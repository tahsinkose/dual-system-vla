"""Shared helpers for joining the LeRobot dataset to the LIBERO simulator.

The two sides enumerate the same ten tasks in **different orders**:

* the dataset's ``task_index`` column, used when training;
* the ``libero`` package's task id, used by ``LiberoEnv(task_ids=[...])``,
  ``lerobot-eval --env.task_ids``, and the ``.bddl`` / ``.pruned_init`` lookups.

For ``libero_10`` the two orderings share *no* fixed points, so passing a dataset
index straight to the environment silently evaluates a different task — a different
scene with different objects — and reads as a policy failure rather than a bug.
:class:`TaskMapping` is the single place that reconciles them, keyed on the
instruction string, which is the only field both sides agree on.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# Exact per-episode initial simulator states, recovered by scripts/extract_init_states.py.
INIT_STATES_DIR = REPO_ROOT / "data"

# Held-still action used to let objects settle after a reset, matching LiberoEnv's
# num_steps_wait behaviour (gripper open, no motion).
DUMMY_ACTION = [0, 0, 0, 0, 0, 0, -1]

# The suites searched when resolving an instruction to a benchmark task.
SUITES = ("libero_10", "libero_goal", "libero_object", "libero_spatial", "libero_90")

# This dataset names its wrist camera differently from what LiberoEnv emits. The two
# must be aliased consistently at train and eval time: LeRobot encodes key names into
# normalisation statistics, so an inconsistency corrupts silently instead of raising.
DATASET_TO_ENV_KEYS = {
    "observation.images.image": "observation.images.image",
    "observation.images.wrist_image": "observation.images.image2",
}


@dataclass(frozen=True)
class TaskRef:
    """One task, addressable from either side of the dataset/simulator divide."""

    instruction: str
    suite: str
    benchmark_id: int
    bddl: str
    dataset_index: int | None = None

    def __str__(self) -> str:
        return (f"{self.instruction!r} (dataset_index={self.dataset_index}, "
                f"suite={self.suite}, benchmark_id={self.benchmark_id})")


@lru_cache(maxsize=1)
def _benchmark_tasks() -> dict[str, TaskRef]:
    """Every LIBERO task, keyed by normalised instruction. Cached — it reads from disk."""
    from libero.libero import benchmark, get_libero_path

    bddl_root = get_libero_path("bddl_files")
    tasks: dict[str, TaskRef] = {}
    for suite_name in SUITES:
        suite = benchmark.get_benchmark_dict()[suite_name]()
        for benchmark_id in range(suite.n_tasks):
            task = suite.get_task(benchmark_id)
            key = _normalise(task.language)
            # setdefault: a handful of instructions recur across suites, and the
            # earlier entry in SUITES is the one this project means.
            tasks.setdefault(key, TaskRef(
                instruction=task.language,
                suite=suite_name,
                benchmark_id=benchmark_id,
                bddl=os.path.join(bddl_root, task.problem_folder, task.bddl_file),
            ))
    return tasks


def _normalise(instruction: str) -> str:
    return instruction.strip().lower()


class TaskMapping:
    """Bidirectional map between dataset task indices and LIBERO benchmark tasks.

    Build it once per dataset with :meth:`from_dataset`, then look tasks up by
    whichever identifier is at hand. Never index one side with the other's integer.
    """

    def __init__(self, by_dataset_index: dict[int, TaskRef]) -> None:
        self._by_dataset_index = by_dataset_index
        self._by_instruction = {_normalise(t.instruction): t for t in by_dataset_index.values()}

    @classmethod
    def from_dataset(cls, dataset) -> "TaskMapping":
        """Resolve every task in the dataset to its benchmark counterpart."""
        benchmark_tasks = _benchmark_tasks()
        resolved: dict[int, TaskRef] = {}
        for dataset_index, instruction in dataset_task_instructions(dataset).items():
            key = _normalise(instruction)
            if key not in benchmark_tasks:
                raise KeyError(
                    f"instruction not found in any LIBERO suite: {instruction!r}. "
                    "The dataset and the installed libero package disagree."
                )
            resolved[dataset_index] = replace_dataset_index(benchmark_tasks[key], dataset_index)
        return cls(resolved)

    def by_dataset_index(self, dataset_index: int) -> TaskRef:
        return self._by_dataset_index[dataset_index]

    def by_instruction(self, instruction: str) -> TaskRef:
        return self._by_instruction[_normalise(instruction)]

    def by_episode(self, dataset, episode_index: int) -> TaskRef:
        return self.by_dataset_index(episode_task_index(dataset, episode_index))

    def dataset_indices(self) -> list[int]:
        return sorted(self._by_dataset_index)

    def __len__(self) -> int:
        return len(self._by_dataset_index)

    def format_table(self) -> str:
        lines = [" dataset  benchmark  instruction"]
        for dataset_index in self.dataset_indices():
            task = self._by_dataset_index[dataset_index]
            marker = "  <- same" if task.benchmark_id == dataset_index else ""
            lines.append(f"   [{dataset_index}]  ->  [{task.benchmark_id}]    "
                         f"{task.instruction}{marker}")
        return "\n".join(lines)


def replace_dataset_index(task: TaskRef, dataset_index: int) -> TaskRef:
    from dataclasses import replace

    return replace(task, dataset_index=dataset_index)


# --------------------------------------------------------------------- dataset access


def episode_slice(dataset, episode_index: int) -> tuple[int, int]:
    """Row range of one episode in the underlying table."""
    row = dataset.meta.episodes[episode_index]
    return int(row["dataset_from_index"]), int(row["dataset_to_index"])


def episode_task_index(dataset, episode_index: int) -> int:
    start, _ = episode_slice(dataset, episode_index)
    return int(dataset.hf_dataset[start]["task_index"])


def dataset_task_instructions(dataset) -> dict[int, str]:
    """Map each dataset task index to its instruction, reading one episode per task."""
    instructions: dict[int, str] = {}
    for episode_index in range(dataset.num_episodes):
        task_index = episode_task_index(dataset, episode_index)
        if task_index not in instructions:
            instructions[task_index] = load_episode(dataset, episode_index)[0]
    return instructions


def episodes_for_task(dataset, dataset_task_index: int) -> list[int]:
    """Every episode index belonging to one dataset task."""
    return [i for i in range(dataset.num_episodes)
            if episode_task_index(dataset, i) == dataset_task_index]


def load_episode(dataset, episode_index: int):
    """Return ``(instruction, actions[T,7], state[T,8])`` for one episode.

    Reads the underlying table directly so the video frames are never decoded —
    replay and initial-state extraction need only actions and proprioception.
    """
    import numpy as np

    start, end = episode_slice(dataset, episode_index)
    rows = dataset.hf_dataset.select(range(start, end))
    instruction = rows[0]["task"] if "task" in rows.column_names else None
    if instruction is None:
        instruction = dataset.meta.tasks.iloc[int(rows[0]["task_index"])].name
    actions = np.asarray(rows["action"], dtype=np.float64)
    state = np.asarray(rows["observation.state"], dtype=np.float64)
    return str(instruction), actions, state


# ----------------------------------------------------------------- initial states


def init_states_path() -> Path:
    """Single archive mapping every episode index to its 8-float initial state."""
    return INIT_STATES_DIR / "init_states.npz"


def load_exact_init_state(episode_index: int):
    """This episode's recorded initial simulator state, or None if not yet extracted.

    The dataset does not carry simulator state; these come from LIBERO's original
    demonstrations via scripts/extract_init_states.py.
    """
    import numpy as np

    path = init_states_path()
    if not path.exists():
        return None
    with np.load(path) as archive:
        key = str(episode_index)
        return archive[key] if key in archive else None


# ------------------------------------------------------------------------ rendering


def agentview_upright(frame):
    """Orient a raw agentview render to match the dataset's frames.

    MuJoCo's buffer is rotated 180 degrees relative to how LIBERO recorded these
    demonstrations — not merely flipped vertically. Measured against a dataset frame:
    rot180 gives 0.047 mean absolute error, a vertical flip alone 0.139.
    """
    return frame[::-1, ::-1]


if __name__ == "__main__":
    import sys

    sys.path.insert(0, str(REPO_ROOT))
    from src.env_setup import setup_env

    setup_env()
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    repo = sys.argv[1] if len(sys.argv) > 1 else "lerobot/libero_10"
    mapping = TaskMapping.from_dataset(LeRobotDataset(repo))
    print(f"{repo}: {len(mapping)} tasks\n")
    print(mapping.format_table())
    coinciding = sum(1 for i in mapping.dataset_indices()
                     if mapping.by_dataset_index(i).benchmark_id == i)
    print(f"\nindices that coincide: {coinciding}/{len(mapping)}")
