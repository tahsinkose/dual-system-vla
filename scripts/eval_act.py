"""Roll out a LeRobot ACT checkpoint in LIBERO and report success and subtask progress.


Examples::

    python scripts/eval_act.py --checkpoint outputs/act/task09/checkpoints/last/pretrained_model
    python scripts/eval_act.py --checkpoint outputs/act/task09 --task-index 9 --trials 10
    python scripts/eval_act.py --checkpoint <path> --temporal-ensemble 0 --n-action-steps 20
    python scripts/eval_act.py --checkpoint outputs/act/task09 --video-dir outputs/act/videos
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.env_setup import setup_env  # noqa: E402

setup_env()

import numpy as np  # noqa: E402
import torch  # noqa: E402

from src.observations import from_env  # noqa: E402
from src.utils import TaskMapping  # noqa: E402

from eval.run_eval import build_env, reset_episode  # noqa: E402
from eval.subtasks import SubtaskTracker, subtasks_for  # noqa: E402
from eval.trials import DEFAULT_TRIALS_PER_TASK, InitSource, build_trials  # noqa: E402
from eval.video import RolloutVideoWriter  # noqa: E402

REPO_ID_DEFAULT = "lerobot/libero_10"

# Canonical camera key -> the dataset feature name ACT was trained on.
POLICY_IMAGE_KEYS = {"image": "observation.images.image",
                     "image2": "observation.images.wrist_image"}

# ACT's published inference setting. Each step's action is a weighted average over the
# chunks predicted at preceding steps, which suppresses the per-step jitter that
# otherwise compounds into a failed approach.
DEFAULT_TEMPORAL_ENSEMBLE = 0.01

# Demonstrations of these tasks run 190-410 steps, and a trained policy is slower than
# the demonstrator it imitates; too tight a horizon scores a solved task as a failure.
DEFAULT_HORIZON = 800

# LIBERO ends an episode the moment its goal predicate holds, and that predicate is
# geometric: a book counts as in the caddy once its centre enters the region, while the
# gripper is still closed around it. Stopping there cuts the recording before the
# release, so a successful rollout does not look successful. These extra steps run on
# past the trigger for the benefit of the video; the reported success step is still the
# step the goal was first satisfied.
DEFAULT_POST_SUCCESS_STEPS = 25


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", type=Path, required=True,
                   help="a pretrained_model directory, or a run directory containing one")
    p.add_argument("--task-index", type=int, default=None,
                   help="dataset task index (default: read from the checkpoint's path)")
    p.add_argument("--trials", type=int, default=DEFAULT_TRIALS_PER_TASK)
    p.add_argument("--horizon", type=int, default=DEFAULT_HORIZON)
    p.add_argument("--settle-steps", type=int, default=10)
    p.add_argument("--camera-size", type=int, default=256)
    p.add_argument("--temporal-ensemble", type=float, default=DEFAULT_TEMPORAL_ENSEMBLE,
                   help=f"ACT temporal ensembling coefficient; 0 disables it "
                        f"(default: {DEFAULT_TEMPORAL_ENSEMBLE})")
    p.add_argument("--n-action-steps", type=int, default=None,
                   help="execute this many steps of each predicted chunk open-loop. "
                        "Mutually exclusive with temporal ensembling, which requires a "
                        "prediction every step")
    p.add_argument("--video-dir", type=Path, default=None,
                   help="write one captioned mp4 per trial here; off by default")
    p.add_argument("--post-success-steps", type=int, default=DEFAULT_POST_SUCCESS_STEPS,
                   help=f"keep stepping this many steps after the goal is first "
                        f"satisfied (default: {DEFAULT_POST_SUCCESS_STEPS}). Does not "
                        "change the reported success step")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--repo-id", default=REPO_ID_DEFAULT)
    return p.parse_args(argv)


def resolve_checkpoint(path: Path) -> Path:
    """Accept a `pretrained_model` directory or any run directory above one."""
    if (path / "config.json").is_file():
        return path
    candidates = sorted(path.glob("**/pretrained_model"))
    if not candidates:
        raise SystemExit(f"no pretrained_model directory under {path}")
    return candidates[-1]


def task_index_from_path(path: Path) -> int:
    """Recover the dataset task index from a `task<NN>` component of the run path."""
    for part in reversed(path.parts):
        if part.startswith("task") and part[4:].isdigit():
            return int(part[4:])
    raise SystemExit("could not infer the task index from the checkpoint path; "
                     "pass --task-index")


def load_policy(checkpoint: Path, dataset, args: argparse.Namespace):
    """The policy plus the pre/post processors holding its normalisation statistics.

    The weights alone are not runnable: the saved model consumes normalised
    observations and emits normalised actions, and the statistics that undo that live
    in the processor files beside it.
    """
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.factory import make_policy, make_pre_post_processors

    cfg = PreTrainedConfig.from_pretrained(str(checkpoint))
    cfg.pretrained_path = checkpoint
    cfg.device = args.device
    if args.n_action_steps is not None:
        cfg.n_action_steps = args.n_action_steps
    if args.temporal_ensemble:
        cfg.temporal_ensemble_coeff = args.temporal_ensemble
        cfg.n_action_steps = 1
    else:
        cfg.temporal_ensemble_coeff = None

    policy = make_policy(cfg=cfg, ds_meta=dataset.meta)
    policy.eval()
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=cfg, pretrained_path=checkpoint,
        preprocessor_overrides={"device_processor": {"device": args.device}},
    )
    return policy, preprocessor, postprocessor, cfg


def build_batch(canon, instruction: str, device) -> dict:
    batch = {POLICY_IMAGE_KEYS[key]: value.to(device) for key, value in canon.images.items()}
    batch["observation.state"] = canon.state.to(device)
    batch["task"] = instruction
    return batch


def run_trial(policy, preprocessor, postprocessor, env, trial, args, device):
    """One rollout, writing its video if one was requested.

    Returns ``(success_step_or_None, tracker, steps_run, video_path_or_None)``, where
    the first element is the step the goal was first satisfied — unaffected by how many
    steps the loop runs on afterwards.
    """
    obs = reset_episode(env, trial.init_state, args.settle_steps)
    tracker = SubtaskTracker(subtasks_for(trial.bddl), env)
    policy.reset()

    video = None
    if args.video_dir is not None:
        video = RolloutVideoWriter(args.video_dir / f"{trial.name}.mp4", "act")

    success_step = None
    step = 0
    try:
        while step < args.horizon:
            canonical = from_env(obs)
            with torch.no_grad():
                processed = preprocessor(build_batch(canonical, trial.instruction, device))
                action = postprocessor(policy.select_action(processed))
            action = np.clip(action.squeeze(0).float().cpu().numpy(), -1.0, 1.0)
            obs, _reward, _done, _info = env.step(action)
            tracker.update(step)
            if env.check_success() and success_step is None:
                success_step = step
            if video is not None:
                # No perturbation is injected here, so there is never a banner to draw.
                video.add_frame(obs["agentview_image"], step, None)
            step += 1
            # `_done` is ignored: LIBERO sets it from the goal predicate, which would
            # end the episode at the instant of success. The loop owns termination so
            # that `--post-success-steps` can run on past it.
            if success_step is not None and step >= success_step + 1 + args.post_success_steps:
                break
    finally:
        if video is not None:
            video.close()
    return success_step, tracker, step, (video.path if video is not None else None)


def main(argv: list[str] | None = None) -> int:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    args = parse_args(argv)
    device = torch.device(args.device)
    checkpoint = resolve_checkpoint(args.checkpoint)
    task_index = args.task_index if args.task_index is not None else task_index_from_path(checkpoint)

    dataset = LeRobotDataset(args.repo_id)
    mapping = TaskMapping.from_dataset(dataset)
    task = mapping.by_dataset_index(task_index)
    policy, preprocessor, postprocessor, cfg = load_policy(checkpoint, dataset, args)

    trials = build_trials(mapping, dataset, InitSource.BENCHMARK,
                          task_indices=[task_index], trials_per_task=args.trials)

    print(f"{checkpoint}")
    print(f"task {task_index} (benchmark {task.benchmark_id}): {task.instruction}")
    print(f"{len(trials)} trials, horizon {args.horizon}, "
          f"ensemble={cfg.temporal_ensemble_coeff}, n_action_steps={cfg.n_action_steps}, "
          f"device={args.device}\n")

    successes = 0
    reached_counts: dict[str, int] = {}
    total = 0
    for trial in trials:
        env = build_env(task, args.camera_size, args.horizon)
        try:
            env.seed(0)   # per-trial determinism comes from the forced initial state
            success_step, tracker, steps, _video = run_trial(policy, preprocessor, postprocessor,
                                                             env, trial, args, device)
        finally:
            env.close()
        success = success_step is not None
        successes += success
        total = tracker.n_total
        for record in tracker.records():
            if record.first_achieved_step is not None and not record.achieved_at_reset:
                reached_counts[record.description] = reached_counts.get(record.description, 0) + 1
        furthest = next((r.description for r in reversed(tracker.records())
                         if r.first_achieved_step is not None and not r.achieved_at_reset),
                        "nothing")
        outcome = f"SUCCESS at step {success_step}" if success else f"failure in {steps} steps"
        print(f"  {trial.name}: {outcome} "
              f"({tracker.n_achieved}/{total} subtasks, furthest: {furthest})")

    print(f"\n{successes}/{len(trials)} succeeded ({successes / len(trials):.0%})")
    if reached_counts:
        print("\nsubtask completion (trials reaching each step, in task order):")
        width = max(len(name) for name in reached_counts)
        for subtask in subtasks_for(trials[0].bddl):
            count = reached_counts.get(subtask.description, 0)
            print(f"  {subtask.description:<{width}}  {count:>3d}/{len(trials)}")
    if args.video_dir is not None:
        print(f"\nwrote {len(trials)} video(s) to {args.video_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
