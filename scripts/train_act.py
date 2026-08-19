"""Train a single-task ACT policy per LIBERO task, through LeRobot's own training loop.

This is a control, not part of the dual-system study. ACT is a well-understood
language-free single-task recipe, so training one per task answers a question the
dual-system's own loss curve cannot: whether `lerobot/libero_10` supports behaviour
cloning at all, on this hardware, with a policy whose expected result is published. A
respectable per-task success rate here means a weak dual-system result is a modelling
problem; a poor one here means the data or the action convention is.

Everything below the argument parsing is LeRobot's: `TrainPipelineConfig` +
`lerobot.scripts.lerobot_train.train`, the same entry point `lerobot-train` uses. The
only thing this script adds is the per-task split — one run per task, restricted to
that task's episodes via `DatasetConfig.episodes`, so each policy sees exactly one
instruction and needs no language conditioning.

Runs are sequential and each writes its own checkpoint directory, so a run can be
interrupted and the remaining tasks started separately.

Examples::

    # one task, short run, to check the mechanics
    python scripts/train_act.py --task-indices 5 --steps 2000

    # every task, full runs, on a chosen GPU
    CUDA_VISIBLE_DEVICES=0 python scripts/train_act.py --steps 100000

    # with periodic in-simulator success rate (slow, needs a rendering device)
    python scripts/train_act.py --task-indices 5 --env-eval-freq 20000
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.env_setup import setup_env  # noqa: E402

setup_env()

from src.utils import TaskMapping, episodes_for_task  # noqa: E402

REPO_ID_DEFAULT = "lerobot/libero_10"

# Held-out fraction per task, for offline validation loss. LeRobot holds out the last
# episodes of each task; with ~38 episodes per libero_10 task this is 4 of them.
EVAL_SPLIT = 0.1


@dataclass
class Args:
    task_indices: list[int] | None
    repo_id: str
    output_dir: Path
    steps: int
    batch_size: int
    chunk_size: int
    num_workers: int
    log_freq: int
    save_freq: int
    eval_steps: int
    env_eval_freq: int
    env_eval_episodes: int
    device: str | None
    seed: int


def parse_args(argv: list[str] | None = None) -> Args:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--task-indices", type=int, nargs="+", default=None,
                   help="dataset task indices to train (default: all ten)")
    p.add_argument("--repo-id", default=REPO_ID_DEFAULT)
    p.add_argument("--output-dir", type=Path, default=Path("outputs/act"),
                   help="parent directory; each task gets a task<NN> subdirectory")
    p.add_argument("--steps", type=int, default=100_000)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--chunk-size", type=int, default=100,
                   help="action chunk length, in environment steps. LeRobot's ACT default "
                        "of 100 was tuned for 50 Hz bimanual data; this dataset declares "
                        "10 fps over ~270-step episodes, so 100 spans a large fraction of "
                        "an episode and pads heavily near the end (default: 100)")
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--log-freq", type=int, default=200)
    p.add_argument("--save-freq", type=int, default=20_000)
    p.add_argument("--eval-steps", type=int, default=0,
                   help="offline validation-loss interval on held-out episodes. Off by "
                        "default: LeRobot's per-task holdout reads a `tasks` column that "
                        "this dataset's episode metadata does not carry, so enabling it "
                        "raises before the first step")
    p.add_argument("--env-eval-freq", type=int, default=0,
                   help="run in-simulator rollouts every N steps (0, the default, disables "
                        "them). This is the measurement that actually answers whether the "
                        "data supports learning, but it needs a rendering device and costs "
                        "wall-clock time inside the training loop")
    p.add_argument("--env-eval-episodes", type=int, default=10,
                   help="rollouts per in-simulator evaluation (default: 10, LIBERO's protocol)")
    p.add_argument("--device", default=None, help="default: LeRobot's own device selection")
    p.add_argument("--seed", type=int, default=1000)
    return Args(**vars(p.parse_args(argv)))


def build_env_config(task, args: Args):
    """LIBERO env config for in-training rollouts, keyed to one benchmark task.

    Two alignments matter. `task_ids` indexes the *benchmark*, not the dataset, and the
    two orders differ — the wrong one silently evaluates a different task. And the env
    publishes its wrist camera as `observation.images.image2` while this dataset names
    it `observation.images.wrist_image`; since the policy's features come from the
    dataset, the env's mapping is what has to move.
    """
    from lerobot.envs.configs import LiberoEnv
    from lerobot.utils.constants import OBS_IMAGES

    env = LiberoEnv(
        task=task.suite,
        task_ids=[task.benchmark_id],
        observation_height=256,
        observation_width=256,
    )
    eye_in_hand = next(key for key, value in env.features_map.items()
                       if value == f"{OBS_IMAGES}.image2")
    env.features_map[eye_in_hand] = f"{OBS_IMAGES}.wrist_image"
    return env


def build_config(task, episodes: list[int], run_dir: Path, args: Args):
    """One `TrainPipelineConfig`, restricted to a single task's episodes."""
    from lerobot.configs.default import DatasetConfig
    from lerobot.configs.train import TrainPipelineConfig
    from lerobot.policies.act.configuration_act import ACTConfig

    policy = ACTConfig(
        chunk_size=args.chunk_size,
        # Receding-horizon execution: predict a chunk, run one step, re-predict. This
        # matches how the dual-system harness rolls out, so the two are comparable.
        n_action_steps=1,
        push_to_hub=False,
    )
    if args.device is not None:
        policy.device = args.device

    env = build_env_config(task, args) if args.env_eval_freq > 0 else None
    config = TrainPipelineConfig(
        dataset=DatasetConfig(repo_id=args.repo_id, episodes=episodes,
                              eval_split=EVAL_SPLIT if args.eval_steps > 0 else 0.0),
        policy=policy,
        env=env,
        output_dir=run_dir,
        job_name=f"act_task{task.dataset_index:02d}",
        steps=args.steps,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        log_freq=args.log_freq,
        save_freq=args.save_freq,
        eval_steps=args.eval_steps,
        env_eval_freq=args.env_eval_freq,
        seed=args.seed,
        wandb=_disabled_wandb(),
    )
    if env is not None:
        config.eval.n_episodes = args.env_eval_episodes
        config.eval.batch_size = min(config.eval.batch_size, args.env_eval_episodes)
    return config


def _disabled_wandb():
    from lerobot.configs.default import WandBConfig

    return WandBConfig(enable=False)


def main(argv: list[str] | None = None) -> int:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    from lerobot.scripts.lerobot_train import train

    args = parse_args(argv)
    dataset = LeRobotDataset(args.repo_id)
    mapping = TaskMapping.from_dataset(dataset)
    task_indices = args.task_indices if args.task_indices is not None else mapping.dataset_indices()

    plan = []
    for index in sorted(task_indices):
        task = mapping.by_dataset_index(index)
        episodes = episodes_for_task(dataset, index)
        run_dir = args.output_dir / f"task{index:02d}"
        if run_dir.exists():
            raise SystemExit(f"{run_dir} already exists; LeRobot refuses to overwrite a run "
                             "directory. Remove it or pass a different --output-dir.")
        plan.append((task, episodes, run_dir))

    print(f"training {len(plan)} single-task ACT policies from {args.repo_id}\n")
    for task, episodes, run_dir in plan:
        print(f"  task {task.dataset_index:02d}  n={len(episodes):3d} episodes  "
              f"-> {run_dir}  {task.instruction}")
    print()

    for task, episodes, run_dir in plan:
        print(f"\n=== task {task.dataset_index:02d}: {task.instruction} ===\n")
        train(build_config(task, episodes, run_dir, args))

    print(f"\nwrote {len(plan)} run(s) under {args.output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
