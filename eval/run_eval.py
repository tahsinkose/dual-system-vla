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
        --perturb displace_object --perturb-at-step 50 --video-dir outputs/eval/videos
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
    camera_size: int = 256
    video_dir: Path | None = None
    log_path: Path = Path("outputs/eval/results.jsonl")
    latent_trace: bool = False
    allow_unmatched_episodes: bool = False   # run episodes with no recovered init state
                                              # (object layout will not match the recording)
    device: str = DEFAULT_DEVICE
    seed: int = 0
    repo_id: str = REPO_ID_DEFAULT

    def __post_init__(self) -> None:
        self.checkpoint = Path(self.checkpoint)
        if self.video_dir is not None:
            self.video_dir = Path(self.video_dir)
        self.log_path = Path(self.log_path)

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
    snapshot = snapshot_target_object_pose(env)   # baseline for UNDO_PROGRESS
    scheduler = PerturbationScheduler(perturbation)
    rng = perturbation.make_rng()

    buffer = TemporalOffsetBuffer(model.config.temporal_offset)
    canon = from_env(obs).to(device)
    buffer.push(canon.images["image"][0])

    video = None
    if cfg.video_dir is not None:
        video = RolloutVideoWriter(
            cfg.video_dir / f"{trial.name}.mp4",
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

            action = np.clip(actions[0, 0].cpu().numpy(), -1.0, 1.0)
            obs, _reward, done, _info = env.step(action)

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
            if done:
                break
    finally:
        if video is not None:
            video.close()

    latent_trace_path = None
    if record_latents and latents_trace:
        latent_trace_path = cfg.log_path.parent / "latents" / f"{trial.name}.npz"
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

    writer = JsonlResultWriter(cfg.log_path)
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
                                     str(loaded.checkpoint_path), device)
            finally:
                env.close()
            writer.write(result)
            results.append(result)
            print(f"{trial.name} (task {trial.task_dataset_index}): "
                  f"{'SUCCESS' if result.success else 'failure'} in {result.steps_run} steps")
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
    p.add_argument("--camera-size", type=int, default=256)
    p.add_argument("--video-dir", type=Path, default=None)
    p.add_argument("--log-path", type=Path, default=Path("outputs/eval/results.jsonl"))
    p.add_argument("--latent-trace", action="store_true",
                   help="dump a full per-step latent trace per episode; opt-in, larger artifact")
    p.add_argument("--allow-unmatched-episodes", action="store_true")
    p.add_argument("--device", default=DEFAULT_DEVICE)
    p.add_argument("--seed", type=int, default=0, help="seeds perturbation randomness per episode")
    p.add_argument("--repo-id", default=REPO_ID_DEFAULT)
    args = p.parse_args(argv)
    return EvalConfig(**vars(args))


if __name__ == "__main__":
    main(parse_args())
