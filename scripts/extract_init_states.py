"""Recover each dataset episode's exact initial simulator state.

Limited to the ``lerobot/libero_10`` dataset (task indices 0-9).

The LeRobot LIBERO dataset keeps images, proprioceptive state and actions, but drops
the simulator state the demonstration started from. Without it a replay begins from a
different object layout and fails, even though the recorded actions are correct.

LIBERO's original HDF5 demonstrations do store it, as ``states[0]`` of each demo, so
this script recovers the mapping:

1. Read ``actions`` and ``states[0]`` for every demo in the task's HDF5. The files are
   ~1.4 GB each but almost all of that is images; HDF5 range requests over HTTPS fetch
   only the two datasets needed, a few MB.
2. Match each dataset episode to its demo by comparing action sequences. Actions
   survive the conversion unchanged, so this is an exact join — not a heuristic.
3. Merge the matched states into a single ``data/init_states.npz``, mapping every
   episode index to its 8-float initial state, for replay to load.

Runs once per task and takes a few minutes, dominated by the remote reads.

Examples::

    python scripts/extract_init_states.py --task-index 5
    python scripts/extract_init_states.py    # extracts all 10 libero_10 tasks
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.env_setup import setup_env  # noqa: E402

setup_env()

from src.utils import (  # noqa: E402
    INIT_STATES_DIR,
    TaskMapping,
    episode_slice,
    episodes_for_task,
    init_states_path,
)

HF_DEMO_REPO = "datasets/yifengzhu-hf/LIBERO-datasets"
REPO_ID = "lerobot/libero_10"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--task-index", type=int, choices=range(10),
                   help="libero_10 dataset task index to extract (0-9); default: extract all")
    return p.parse_args()


def demo_file_for(suite_name: str, bddl: str) -> str:
    """Remote path of the HDF5 holding this task's demonstrations."""
    stem = Path(bddl).stem
    return f"{HF_DEMO_REPO}/{suite_name}/{stem}_demo.hdf5"


# Actions round-trip through float32 in the conversion, so allow a little slack.
MATCH_TOLERANCE = 1e-3


def read_demos(remote_path: str, cache_path: Path) -> dict[str, tuple]:
    """Fetch (actions, initial state) for every demo, without keeping the images.

    Downloads the whole ~1.4 GB HDF5 in one streaming pass rather than issuing an
    HTTPS range request per demo: HDF5's chunked layout turns a small logical read into
    many small round trips, and round-trip latency rather than bandwidth dominates over
    a real network. The raw download is discarded once ``actions`` and ``states[0]``
    are extracted into ``cache_path``, which every later call re-reads instead.
    """
    import numpy as np

    if cache_path.exists():
        with np.load(cache_path) as archive:
            names = {key.rsplit("/", 1)[0] for key in archive.files}
            return {n: (archive[f"{n}/actions"], archive[f"{n}/state"]) for n in names}

    import h5py
    from huggingface_hub import HfFileSystem

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    raw_path = cache_path.parent / f"_raw_{cache_path.stem}.hdf5"
    print(f"  downloading {remote_path} (~1.4 GB, one pass)...", flush=True)
    fs = HfFileSystem()
    fs.get_file(remote_path, str(raw_path))

    demos = {}
    with h5py.File(raw_path, "r") as h5:
        group = h5["data"]
        names = list(group)
        for i, name in enumerate(names, 1):
            demos[name] = (group[name]["actions"][:], group[name]["states"][0][:])
            print(f"\r  reading demos from disk: {i}/{len(names)}", end="", flush=True)
    print()
    raw_path.unlink()

    flat = {}
    for name, (actions, state) in demos.items():
        flat[f"{name}/actions"] = actions
        flat[f"{name}/state"] = state
    np.savez_compressed(cache_path, **flat)
    return demos


def match_episode(actions, demos: dict[str, tuple]) -> tuple[str | None, float]:
    """Find the demo whose action sequence matches this episode.

    Equal length alone is *not* sufficient evidence: only 33 of this task's 50 demos
    survived conversion, so an unrelated demo can coincidentally share a length. The
    action values must agree too — a mismatch shows up as a delta of 2.0, the full
    range of the binary gripper channel. Callers reject anything above the tolerance
    rather than storing an initial state belonging to a different demonstration.
    """
    import numpy as np

    best_name, best_error = None, float("inf")
    for name, (demo_actions, _state) in demos.items():
        if len(demo_actions) != len(actions):
            continue
        error = float(np.abs(demo_actions - actions).max())
        if error < best_error:
            best_name, best_error = name, error
    return best_name, best_error


def load_all_states() -> dict:
    """The full episode-index -> initial-state mapping accumulated so far."""
    import numpy as np

    path = init_states_path()
    if not path.exists():
        return {}
    with np.load(path) as archive:
        return {key: archive[key] for key in archive.files}


def save_all_states(states: dict) -> None:
    import numpy as np

    INIT_STATES_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(init_states_path(), **states)


def extract_task(dataset, task_index: int, mapping: TaskMapping, states: dict,
                 progress: str = "") -> tuple[int, bool]:
    """Match this task's episodes into ``states`` in place. Returns (status, changed)."""
    import numpy as np

    episodes = episodes_for_task(dataset, task_index)
    if not episodes:
        print(f"task_index {task_index}: no episodes", file=sys.stderr)
        return 1, False

    if all(str(e) in states for e in episodes):
        print(f"task_index {task_index}{progress}: already extracted", flush=True)
        return 0, False

    task = mapping.by_dataset_index(task_index)
    print(f"\ntask_index {task_index}{progress}: {task.instruction}", flush=True)
    print(f"  {len(episodes)} episodes | suite={task.suite} "
          f"| benchmark_id={task.benchmark_id}", flush=True)
    print("  reading demo actions and initial states over HTTPS (a few minutes)...", flush=True)
    demos = read_demos(demo_file_for(task.suite, task.bddl),
                       INIT_STATES_DIR / f"_demos_task_{task_index:02d}.npz")
    print(f"  {len(demos)} demos in the HDF5", flush=True)

    matched, unmatched = 0, []
    for i, episode_index in enumerate(episodes, 1):
        begin, end = episode_slice(dataset, episode_index)
        actions = np.asarray(dataset.hf_dataset.select(range(begin, end))["action"], dtype=np.float64)
        name, error = match_episode(actions, demos)
        if name is None or error > MATCH_TOLERANCE:
            # Rejecting is the safe outcome: a wrong initial state would make an
            # otherwise-valid replay fail, which is far harder to diagnose than a gap.
            unmatched.append((episode_index, None if name is None else f"{name}@{error:.1f}"))
        else:
            states[str(episode_index)] = demos[name][1]
            matched += 1
        print(f"\r  matching episodes: {i}/{len(episodes)} (matched {matched})", end="", flush=True)
    print()

    if unmatched:
        print(f"  unmatched ({len(unmatched)}): {unmatched}")
        print("  these episodes cannot be replayed faithfully; they are skipped")
    return 0, True


def main() -> int:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    args = parse_args()
    dataset = LeRobotDataset(REPO_ID)
    mapping = TaskMapping.from_dataset(dataset)
    states = load_all_states()

    targets = list(range(10)) if args.task_index is None else [args.task_index]
    status, changed = 0, False
    for i, task_index in enumerate(targets, 1):
        progress = f" [{i}/{len(targets)}]" if len(targets) > 1 else ""
        task_status, task_changed = extract_task(dataset, task_index, mapping, states, progress)
        status |= task_status
        changed = changed or task_changed

    if changed:
        save_all_states(states)
        print(f"\nwrote {init_states_path()} ({len(states)} episodes total)", flush=True)
    return status


if __name__ == "__main__":
    sys.exit(main())
