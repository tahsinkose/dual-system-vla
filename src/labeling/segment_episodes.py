"""Segment demonstrations into phases by gripper open/close transitions.

Produces the labels the (separately scoped) bottleneck-analysis probe needs to map
System 2's latent `z` to task phase. The segmentation itself needs no model or
simulator: LIBERO's recorded actions carry the gripper command as a clean bang-bang
±1 signal — verified against real episodes (no intermediate values found) — so a
phase boundary is exactly wherever the demonstrator's own grasp/release command
flips, not a heuristic threshold on a noisy continuous signal.

The core function, `segment_by_gripper_transitions`, is deliberately generic — it
takes only an action array. The same logic applies unmodified both here (over
recorded demonstrations) and later to a live eval rollout's own executed actions when
the probe is trained (§6, out of scope here).

Examples::

    python -m src.labeling.segment_episodes                    # all libero_10 episodes
    python -m src.labeling.segment_episodes --task-indices 5 7 # just some tasks
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.env_setup import setup_env  # noqa: E402

setup_env()

from src.utils import episodes_for_task, load_episode  # noqa: E402

REPO_ID = "lerobot/libero_10"
PHASE_LABELS_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "phase_labels.npz"


def gripper_transition_steps(actions: np.ndarray) -> np.ndarray:
    """Steps where the gripper command's sign flips.

    Each returned index is the first step of the new command. LIBERO's recorded
    gripper channel is an exact ±1 bang-bang signal, so a bare sign comparison is an
    exact detector, not a heuristic threshold on noisy continuous data.
    """
    gripper = actions[:, -1]
    return np.flatnonzero(np.diff(np.sign(gripper)) != 0) + 1


def segment_by_gripper_transitions(actions: np.ndarray) -> np.ndarray:
    """Per-timestep phase label: 0 up to the first transition, incrementing by one at
    each subsequent grasp/release command. Same length as `actions`."""
    labels = np.zeros(len(actions), dtype=np.int64)
    for step in gripper_transition_steps(actions):
        labels[step:] += 1
    return labels


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--task-indices", type=int, nargs="+", default=None,
                   help="restrict to these dataset task indices (default: all ten)")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    args = parse_args(argv)
    dataset = LeRobotDataset(REPO_ID)

    episodes = (sorted(e for t in args.task_indices for e in episodes_for_task(dataset, t))
               if args.task_indices is not None else list(range(dataset.num_episodes)))

    labels_by_episode: dict[str, np.ndarray] = {}
    phase_counts: list[int] = []
    degenerate: list[int] = []
    for episode_index in episodes:
        _instruction, actions, _state = load_episode(dataset, episode_index)
        labels = segment_by_gripper_transitions(actions)
        labels_by_episode[str(episode_index)] = labels
        n_phases = int(labels.max()) + 1
        phase_counts.append(n_phases)
        if n_phases == 1:
            degenerate.append(episode_index)

    PHASE_LABELS_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(PHASE_LABELS_PATH, **labels_by_episode)

    print(f"wrote {PHASE_LABELS_PATH} ({len(labels_by_episode)} episodes)")
    print(f"phases per episode: min {min(phase_counts)}, max {max(phase_counts)}, "
          f"mean {np.mean(phase_counts):.1f}")
    if degenerate:
        print(f"WARNING: {len(degenerate)} episode(s) never transition the gripper "
              f"(single phase, likely a corrupted or degenerate demo): {degenerate}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
