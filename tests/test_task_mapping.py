"""Tests for the dataset <-> LIBERO benchmark task mapping.

The mapping is the join between training data and the simulator. Getting it wrong
does not raise — it evaluates a different task and looks like a policy failure — so
it is worth pinning down.

Run with::

    python -m pytest tests/ -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.env_setup import setup_env  # noqa: E402

setup_env()

from src.utils import (  # noqa: E402
    DATASET_TO_ENV_KEYS,
    TaskMapping,
    agentview_upright,
    episodes_for_task,
)

REPO_ID = "lerobot/libero_10"


@pytest.fixture(scope="module")
def dataset():
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    return LeRobotDataset(REPO_ID)


@pytest.fixture(scope="module")
def mapping(dataset):
    return TaskMapping.from_dataset(dataset)


def test_covers_every_task(mapping):
    assert len(mapping) == 10
    assert mapping.dataset_indices() == list(range(10))


def test_every_task_resolves_to_libero_10(mapping):
    for index in mapping.dataset_indices():
        assert mapping.by_dataset_index(index).suite == "libero_10"


def test_benchmark_ids_are_a_permutation(mapping):
    """Each dataset task maps to a distinct benchmark id — no collisions, none dropped."""
    ids = [mapping.by_dataset_index(i).benchmark_id for i in mapping.dataset_indices()]
    assert sorted(ids) == list(range(10))


def test_indices_genuinely_differ(mapping):
    """The two orderings share no fixed points.

    This is the whole reason the adapter exists: for libero_10 every dataset index
    denotes a different task than the same integer does on the benchmark side, so
    using one as the other is wrong for every task, not just unlucky ones.
    """
    coinciding = [i for i in mapping.dataset_indices()
                  if mapping.by_dataset_index(i).benchmark_id == i]
    assert coinciding == []


def test_known_pair(mapping):
    """A concrete anchor, verified by replaying episodes of this task to success."""
    task = mapping.by_dataset_index(5)
    assert task.instruction.lower().startswith("put both the alphabet soup and the tomato sauce")
    assert task.benchmark_id == 0


def test_lookups_agree(mapping, dataset):
    """Instruction, dataset index and episode lookups return the same task."""
    for index in mapping.dataset_indices():
        by_index = mapping.by_dataset_index(index)
        assert mapping.by_instruction(by_index.instruction) == by_index

        episode = episodes_for_task(dataset, index)[0]
        assert mapping.by_episode(dataset, episode) == by_index


def test_bddl_files_exist(mapping):
    for index in mapping.dataset_indices():
        assert Path(mapping.by_dataset_index(index).bddl).is_file()


def test_instruction_lookup_is_case_and_space_insensitive(mapping):
    task = mapping.by_dataset_index(0)
    assert mapping.by_instruction(f"  {task.instruction.upper()}  ") == task


def test_wrist_camera_is_aliased_to_the_env_name():
    """The dataset and the env disagree on the wrist camera key."""
    assert DATASET_TO_ENV_KEYS["observation.images.wrist_image"] == "observation.images.image2"
    assert DATASET_TO_ENV_KEYS["observation.images.image"] == "observation.images.image"


def test_agentview_upright_is_rot180():
    import numpy as np

    frame = np.arange(2 * 3 * 3).reshape(2, 3, 3)
    np.testing.assert_array_equal(agentview_upright(frame), frame[::-1, ::-1])
    # A vertical flip alone is not equivalent — that was the original bug.
    assert not np.array_equal(agentview_upright(frame), frame[::-1])
