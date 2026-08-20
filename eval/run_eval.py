"""LIBERO rollout harness for the dual-system ablation.

    python -m eval.run_eval --checkpoint outputs/train/live/step_0100000.pt \\
        --conditioning frozen --task-indices 5 7 --video-dir outputs/eval/videos

One checkpoint, evaluated under one conditioning override, over a chosen set of
dataset episodes, optionally with one mid-rollout perturbation. Each invocation
covers one cell (or a slice of cells) of the ablation matrix; running the whole
matrix is repeated invocations of this script over different --checkpoint /
--conditioning / --perturb combinations, aggregated downstream by a later
analysis pass (out of scope here).

Perturbation injection does not support "introduce an obstacle" — see
eval/perturbations.py's docstring. Training an offline probe on the logged latents
is likewise out of scope; this harness only *logs* what that probe would need
(--latent-trace).

Examples::

    python -m eval.run_eval --checkpoint outputs/train/live --task-indices 0 --horizon 200
    python -m eval.run_eval --checkpoint outputs/train/live --conditioning zero \\
        --perturb displace_object --perturb-at-step 50 --video
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.env_setup import setup_env  # noqa: E402

setup_env()

from src.models.dual_system import Conditioning, DualSystem  # noqa: E402
from src.observations import from_env  # noqa: E402
from src.utils import DUMMY_ACTION, TaskMapping, episodes_for_task, load_episode, load_exact_init_state  # noqa: E402

from eval.checkpoints import load_checkpoint, warn_if_conditioning_mismatch  # noqa: E402
from eval.logging import EpisodeResult, JsonlResultWriter, summarize, write_latent_trace  # noqa: E402
from eval.perturbations import (  # noqa: E402
    DEFAULT_DISPLACEMENT_RADIUS_M,
    PerturbationKind,
    PerturbationScheduler,
    PerturbationSpec,
    TriggerCondition,
    apply_perturbation,
    snapshot_target_object_pose,
)
from eval.subtasks import SubtaskTracker, subtasks_for  # noqa: E402
from eval.trace import RolloutTracer  # noqa: E402
from eval.trials import DEFAULT_TRIALS_PER_TASK, InitSource, Trial, build_trials  # noqa: E402
from eval.video import RolloutVideoWriter  # noqa: E402

REPO_ID_DEFAULT = "lerobot/libero_10"
DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class EvalConfig:
    checkpoint: Path
    conditioning: str | None = None          # None -> use the checkpoint's trained mode
    task_indices: list[int] | None = None
    episodes: list[int] | None = None        # explicit dataset episode indices; demo source only
    init_source: str = InitSource.BENCHMARK.value
    trials_per_task: int = DEFAULT_TRIALS_PER_TASK
    perturb: str = "none"
    perturb_at_step: int | None = None
    perturb_after_success_steps: int | None = None
    displacement_radius_m: float = DEFAULT_DISPLACEMENT_RADIUS_M
    horizon: int = 600
    settle_steps: int = 10
    post_success_steps: int = 25             # steps to keep running after the goal is
                                              # first satisfied; see run_episode
    camera_size: int = 256
    # Every artifact of a run lands under one directory, at a fixed leaf name, so a run
    # is one thing to find, archive or delete rather than three paths to keep in step.
    output_dir: Path = Path("outputs/eval")
    video: bool = False                      # <output_dir>/videos/<trial>.mp4
    trace: bool = False                      # <output_dir>/traces/<trial>.npz
    latent_trace: bool = False               # <output_dir>/latents/<trial>.npz
    allow_unmatched_episodes: bool = False   # run episodes with no recovered init state
                                              # (object layout will not match the recording)
    device: str = DEFAULT_DEVICE
    seed: int = 0
    repo_id: str = REPO_ID_DEFAULT

    def __post_init__(self) -> None:
        self.checkpoint = Path(self.checkpoint)
        self.output_dir = Path(self.output_dir)

        kind = PerturbationKind(self.perturb)
        trigger_fields = (self.perturb_at_step, self.perturb_after_success_steps)
        if kind is PerturbationKind.NONE:
            if any(f is not None for f in trigger_fields):
                raise ValueError("--perturb-at-step/--perturb-after-success-steps require --perturb != none")
        elif sum(f is not None for f in trigger_fields) != 1:
            raise ValueError("exactly one of --perturb-at-step / --perturb-after-success-steps is required "
                             f"when --perturb={kind.value}")
        if kind is PerturbationKind.UNDO_PROGRESS and self.perturb_after_success_steps is None:
            print("WARNING: --perturb undo_progress without --perturb-after-success-steps triggers on a "
                  "raw step count instead of 'N steps after task completion' — likely not what you want.")


def artifact_dir(cfg: "EvalConfig", kind: str) -> Path:
    """`<output_dir>/<kind>`, created on demand. The leaf names are not configurable."""
    path = cfg.output_dir / kind
    path.mkdir(parents=True, exist_ok=True)
    return path


def build_env(task, camera_size: int, horizon: int):
    """Construct one LIBERO env for `task`, mirroring scripts/replay_episode.py."""
    from libero.libero.envs import OffScreenRenderEnv

    return OffScreenRenderEnv(
        bddl_file_name=task.bddl,
        camera_heights=camera_size,
        camera_widths=camera_size,
        # Headroom over settle_steps + horizon so LIBERO's own horizon cutoff never
        # fires before this loop's does.
        horizon=horizon + 50,
    )


def reset_episode(env, init_state, settle_steps: int) -> dict:
    """Reset, force the trial's initial simulator state, then settle.

    Matches scripts/replay_episode.py's sequence: a different settle count changes
    where objects land and silently produces a different effective task. `init_state`
    may be None only for demo trials run with --allow-unmatched-episodes, which fall
    back to the environment's own randomised reset.
    """
    obs = env.reset()
    if init_state is not None:
        obs = env.set_init_state(init_state)
    for _ in range(settle_steps):
        obs, _reward, _done, _info = env.step(DUMMY_ACTION)
    return obs


class TemporalOffsetBuffer:
    """Feeds System 2 the frame from Δ steps ago, reproducing at eval time what the
    dataloader does at training time (`t - Δ` for the main camera). Holds canonical
    (already `from_env`-processed) per-example main-camera tensors.
    """

    def __init__(self, offset: int) -> None:
        self._offset = offset
        self._frames: list[torch.Tensor] = []

    def push(self, frame: torch.Tensor) -> None:
        self._frames.append(frame)
        if len(self._frames) > self._offset + 1:
            self._frames.pop(0)

    def offset_frame(self) -> torch.Tensor:
        """The frame to feed System 2 *this* step. Clamps to the first frame pushed
        before Δ steps have elapsed — there is no t < 0 frame."""
        return self._frames[0]

    def latest_frame(self) -> torch.Tensor:
        """The most recently pushed frame — used only by the perturbation cosine
        probe, which deliberately measures System 2's *immediate* reaction rather
        than reproducing deployment staleness."""
        return self._frames[-1]


def _episode_is_over(scheduler: PerturbationScheduler, perturbation: PerturbationSpec,
                     steps_taken: int, post_success_steps: int) -> bool:
    """Whether to stop before the horizon, given what the episode still has to observe.

    The environment's own `done` is not consulted. LIBERO sets it from the goal
    predicate, so it ends the episode at the instant of success — which truncates a
    video before the gripper releases, and makes an `after_success_steps` perturbation
    trigger unreachable for any offset above zero.

    An unperturbed episode ends `post_success_steps` after the goal is first satisfied.
    A perturbed one runs to the horizon: before the trigger because the trigger may
    still be pending, and after it because recovery is the thing being measured.
    """
    if perturbation.kind is not PerturbationKind.NONE:
        return False
    first_success = scheduler.first_success_step
    return first_success is not None and steps_taken >= first_success + 1 + post_success_steps


def run_episode(
    model: DualSystem,
    env,
    trial: Trial,
    eval_conditioning: Conditioning,
    trained_conditioning: Conditioning,
    perturbation: PerturbationSpec,
    cfg: EvalConfig,
    checkpoint_path: str,
    device: torch.device,
    checkpoint_step: int | None = None,
) -> EpisodeResult:
    """Run one episode start to finish, writing its own video/latent-trace artifacts
    (if requested) before returning.

    Receding-horizon execution: `model.act()` is called every environment step (the
    "System 1 runs every step" requirement), and only `actions[0, 0]` — the first
    step of the freshly predicted chunk — is ever executed. The rest of the chunk is
    discarded and recomputed next step; there is no chunk queue.
    """
    model.reset()
    obs = reset_episode(env, trial.init_state, cfg.settle_steps)
    # Built after settling, so `achieved_at_reset` describes the configuration the
    # policy actually starts from.
    subtasks = subtasks_for(trial.bddl)
    tracker = SubtaskTracker(subtasks, env)
    tracer = RolloutTracer(env, subtasks) if cfg.trace else None
    snapshot = snapshot_target_object_pose(env)   # baseline for UNDO_PROGRESS
    scheduler = PerturbationScheduler(perturbation)
    rng = perturbation.make_rng()

    buffer = TemporalOffsetBuffer(model.config.temporal_offset)
    canon = from_env(obs).to(device)
    buffer.push(canon.images["image"][0])

    video = None
    if cfg.video:
        video = RolloutVideoWriter(
            artifact_dir(cfg, "videos") / f"{trial.name}.mp4",
            eval_conditioning.value,
        )
    record_latents = cfg.latent_trace
    latents_trace: list[np.ndarray] = [] if record_latents else None
    steps_since_update_trace: list[int] = [] if record_latents else None

    success = False
    cosine_distance: float | None = None
    trigger_step: int | None = None
    first_success_after_trigger_step: int | None = None
    step_index = 0

    try:
        while step_index < cfg.horizon:
            canon = from_env(obs).to(device)
            buffer.push(canon.images["image"][0])
            system2_images = [buffer.offset_frame()]

            actions, z = model.act(canon.images, canon.state, system2_images, [trial.instruction],
                                   conditioning=eval_conditioning)
            if record_latents:
                latents_trace.append(z[0].cpu().numpy())
                steps_since_update_trace.append(model.steps_since_latent_update)

            action_raw = actions[0, 0].cpu().numpy()
            action = np.clip(action_raw, -1.0, 1.0)
            obs, _reward, _done, _info = env.step(action)

            tracker.update(step_index)
            if tracer is not None:
                tracer.record(step_index, obs, action, action_raw=action_raw,
                              latent=z[0].cpu().numpy(),
                              steps_since_update=model.steps_since_latent_update)
            if env.check_success():
                success = True
                if (trigger_step is not None and step_index > trigger_step
                        and first_success_after_trigger_step is None):
                    first_success_after_trigger_step = step_index
            scheduler.observe(step_index, success)

            if scheduler.should_fire():
                trigger_step = step_index
                probe_source = (Conditioning.LIVE if eval_conditioning in
                                (Conditioning.LIVE, Conditioning.FROZEN) else eval_conditioning)
                if eval_conditioning is not Conditioning.ZERO:
                    # Deliberately uses the *immediate* current frame, not the
                    # Δ-offset one: this is an off-cadence diagnostic measuring
                    # "would S2 notice right now", not part of the control path.
                    z_pre = model.compute_latent([buffer.latest_frame()], [trial.instruction], probe_source)
                obs = apply_perturbation(env, perturbation.kind, rng, snapshot, perturbation.displacement_radius_m)
                # A perturbation edits simulator state directly; re-reading here is
                # what makes an undone subtask visible as such rather than as an
                # unexplained drop in the next step's reading.
                tracker.update(step_index)
                if env.check_success():
                    success = True
                if eval_conditioning is not Conditioning.ZERO:
                    new_frame = from_env(obs).to(device).images["image"][0]
                    z_post = model.compute_latent([new_frame], [trial.instruction], probe_source)
                    cosine_distance = float(
                        1.0 - torch.nn.functional.cosine_similarity(z_pre, z_post).item()
                    )

            if video is not None:
                since_perturb = step_index - trigger_step if trigger_step is not None else None
                video.add_frame(obs["agentview_image"], step_index, since_perturb)

            step_index += 1
            if _episode_is_over(scheduler, perturbation, step_index, cfg.post_success_steps):
                break
    finally:
        if video is not None:
            video.close()

    if tracer is not None:
        tracer.write(artifact_dir(cfg, "traces") / f"{trial.name}.npz")

    latent_trace_path = None
    if record_latents and latents_trace:
        latent_trace_path = artifact_dir(cfg, "latents") / f"{trial.name}.npz"
        write_latent_trace(latent_trace_path, np.stack(latents_trace), np.array(steps_since_update_trace))

    recovered = None
    steps_to_recovery = None
    if scheduler.fired:
        recovered = first_success_after_trigger_step is not None
        if recovered:
            steps_to_recovery = first_success_after_trigger_step - trigger_step

    return EpisodeResult(
        task_dataset_index=trial.task_dataset_index, task_instruction=trial.instruction,
        episode_index=trial.index, seed=cfg.seed, checkpoint_path=checkpoint_path,
        trained_conditioning=trained_conditioning.value, eval_conditioning=eval_conditioning.value,
        success=success, steps_run=step_index,
        steps_to_success=scheduler.first_success_step,
        perturbation_kind=perturbation.kind.value,
        perturbation_trigger_step=trigger_step, perturbation_applied=scheduler.fired,
        first_success_step=scheduler.first_success_step,
        latent_cosine_distance=cosine_distance, recovered=recovered, steps_to_recovery=steps_to_recovery,
        subtasks=tracker.as_dicts(), subtasks_achieved=tracker.n_achieved,
        subtasks_total=tracker.n_total,
        horizon=cfg.horizon, init_source=cfg.init_source,
        trials_per_task=cfg.trials_per_task,
        system1_arch=model.config.system1_arch,
        latent_update_period=model.config.latent_update_period,
        temporal_offset=model.config.temporal_offset,
        perturb_at_step=cfg.perturb_at_step,
        displacement_radius_m=(cfg.displacement_radius_m
                               if perturbation.kind is not PerturbationKind.NONE else None),
        checkpoint_step=checkpoint_step,
        video_path=str(video.path) if video is not None else None,
        latent_trace_path=str(latent_trace_path) if latent_trace_path else None,
    )


def main(cfg: EvalConfig) -> list[EpisodeResult]:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    device = torch.device(cfg.device)
    loaded = load_checkpoint(cfg.checkpoint, device=cfg.device)
    eval_conditioning = Conditioning(cfg.conditioning) if cfg.conditioning else loaded.trained_conditioning
    warn_if_conditioning_mismatch(loaded.trained_conditioning, eval_conditioning)

    dataset = LeRobotDataset(cfg.repo_id)
    mapping = TaskMapping.from_dataset(dataset)
    source = InitSource(cfg.init_source)
    trials = build_trials(mapping, dataset, source,
                          task_indices=cfg.task_indices,
                          trials_per_task=cfg.trials_per_task,
                          episodes=cfg.episodes,
                          allow_unmatched=cfg.allow_unmatched_episodes)
    if not trials:
        raise SystemExit("no trials to run")
    print(f"{len(trials)} trials from {source.value} initial states "
          f"across {len({t.task_dataset_index for t in trials})} task(s)\n")

    perturb_kind = PerturbationKind(cfg.perturb)
    trigger = None
    if perturb_kind is not PerturbationKind.NONE:
        trigger = TriggerCondition(at_step=cfg.perturb_at_step,
                                   after_success_steps=cfg.perturb_after_success_steps)

    writer = JsonlResultWriter(cfg.output_dir / "results.jsonl")
    results: list[EpisodeResult] = []
    try:
        for trial in trials:
            task = mapping.by_dataset_index(trial.task_dataset_index)
            env = build_env(task, cfg.camera_size, cfg.horizon)
            spec = PerturbationSpec(kind=perturb_kind, trigger=trigger,
                                    episode_seed=(cfg.seed, trial.index),
                                    displacement_radius_m=cfg.displacement_radius_m)
            try:
                env.seed(0)   # per-trial determinism comes from the forced init state
                              # and perturbation.episode_seed, not the env seed
                result = run_episode(loaded.model, env, trial, eval_conditioning,
                                     loaded.trained_conditioning, spec, cfg,
                                     str(loaded.checkpoint_path), device,
                                     checkpoint_step=loaded.step)
            finally:
                env.close()
            writer.write(result)
            results.append(result)
            reached = next((s["description"] for s in reversed(result.subtasks)
                            if s["first_achieved_step"] is not None
                            and not s["achieved_at_reset"]), "nothing")
            print(f"{trial.name} (task {trial.task_dataset_index}): "
                  f"{'SUCCESS' if result.success else 'failure'} in {result.steps_run} steps, "
                  f"{result.subtasks_achieved}/{result.subtasks_total} subtasks "
                  f"(furthest: {reached})")
    finally:
        writer.close()

    print("\n" + summarize(results))
    return results


def parse_args(argv: list[str] | None = None) -> EvalConfig:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", type=Path, required=True,
                   help="a step_*.pt file, or a run directory containing one")
    p.add_argument("--conditioning", choices=[c.value for c in Conditioning], default=None,
                   help="override the checkpoint's trained conditioning; default: use it as-is")
    p.add_argument("--task-indices", type=int, nargs="+", default=None,
                   help="restrict to these dataset task indices (default: all mapped tasks)")
    p.add_argument("--init-source", choices=[s.value for s in InitSource],
                   default=InitSource.BENCHMARK.value,
                   help="'benchmark' (default) starts from LIBERO's own .pruned_init states — "
                        "the standard protocol, and the only valid source for a reportable "
                        "success rate, since no demonstration starts from one. 'demo' starts "
                        "from a dataset episode's recovered state, which training has seen; "
                        "use it for debugging against a known-good demonstration only")
    p.add_argument("--trials-per-task", type=int, default=DEFAULT_TRIALS_PER_TASK,
                   help=f"benchmark init states per task (default: {DEFAULT_TRIALS_PER_TASK}, "
                        "LIBERO's published protocol)")
    p.add_argument("--episodes", type=int, nargs="+", default=None,
                   help="explicit dataset episode indices; overrides --task-indices")
    p.add_argument("--perturb", choices=[k.value for k in PerturbationKind], default="none")
    p.add_argument("--perturb-at-step", type=int, default=None)
    p.add_argument("--perturb-after-success-steps", type=int, default=None,
                   help="required for --perturb undo_progress in practice")
    p.add_argument("--displacement-radius-m", type=float, default=DEFAULT_DISPLACEMENT_RADIUS_M)
    p.add_argument("--horizon", type=int, default=600)
    p.add_argument("--settle-steps", type=int, default=10)
    p.add_argument("--post-success-steps", type=int, default=25,
                   help="keep stepping this many steps after the goal is first satisfied, "
                        "so a video shows the outcome rather than cutting at the instant "
                        "the predicate flips. Does not change the reported success step; "
                        "ignored when a perturbation is configured, which runs to the horizon")
    p.add_argument("--camera-size", type=int, default=256)
    p.add_argument("--output-dir", type=Path, default=EvalConfig.output_dir,
                   help="everything this run writes goes here: results.jsonl, and the "
                        "videos/ traces/ latents/ subdirectories for whichever artifacts "
                        "are enabled")
    p.add_argument("--video", action="store_true",
                   help="write <output-dir>/videos/<trial>.mp4, captioned")
    p.add_argument("--trace", action="store_true",
                   help="write <output-dir>/traces/<trial>.npz — per-step state, actions, "
                        "subtask flags, object positions, object-to-goal distances. Pair "
                        "with scripts/replay_episode.py --trace for the demonstration "
                        "reference, then classify with scripts/analyze_traces.py")
    p.add_argument("--latent-trace", action="store_true",
                   help="write <output-dir>/latents/<trial>.npz; opt-in, larger artifact")
    p.add_argument("--allow-unmatched-episodes", action="store_true")
    p.add_argument("--device", default=DEFAULT_DEVICE)
    p.add_argument("--seed", type=int, default=0, help="seeds perturbation randomness per episode")
    p.add_argument("--repo-id", default=REPO_ID_DEFAULT)
    args = p.parse_args(argv)
    return EvalConfig(**vars(args))


if __name__ == "__main__":
    main(parse_args())
