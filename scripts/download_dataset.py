"""Download and verify the dataset slice this study uses.

Fetches `lerobot/libero_10` — the LIBERO-10 (long-horizon) suite — into the repo-local
cache and prints a summary so the download can be confirmed before anything depends
on it.

The download location matters: `src.env_setup` points `HF_LEROBOT_HOME` at
`data/lerobot` inside the repo, and LeRobot reads that variable once at import time.
Fetching the dataset without it lands the files in `~/.cache` instead.

Usage::

    python scripts/download_dataset.py
    python scripts/download_dataset.py --repo-id lerobot/libero    # all four suites
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.env_setup import setup_env  # noqa: E402

setup_env()

from src.utils import TaskMapping, episodes_for_task  # noqa: E402

DEFAULT_REPO_ID = "lerobot/libero_10"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--repo-id", default=DEFAULT_REPO_ID,
                   help=f"HuggingFace dataset to fetch (default: {DEFAULT_REPO_ID})")
    return p.parse_args()


def main() -> int:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.utils.constants import HF_LEROBOT_HOME

    args = parse_args()
    print(f"downloading {args.repo_id} into {HF_LEROBOT_HOME} ...\n")
    dataset = LeRobotDataset(args.repo_id)

    print(f"episodes : {dataset.num_episodes}")
    print(f"frames   : {dataset.num_frames}")
    print(f"fps      : {dataset.meta.fps}")
    print(f"root     : {dataset.root}")

    sample = dataset[0]
    print("\nfeatures:")
    for key in ("observation.images.image", "observation.images.wrist_image",
                "observation.images.image2", "observation.state", "action"):
        if key in sample:
            value = sample[key]
            print(f"  {key:34s} {tuple(value.shape)} {value.dtype}")

    mapping = TaskMapping.from_dataset(dataset)
    print(f"\ntasks ({len(mapping)}), with episode counts:")
    for index in mapping.dataset_indices():
        task = mapping.by_dataset_index(index)
        count = len(episodes_for_task(dataset, index))
        print(f"  dataset[{index}] -> benchmark[{task.benchmark_id}]  n={count:3d}  {task.instruction}")

    print("\nOK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
