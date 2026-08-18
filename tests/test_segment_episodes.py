"""Tests for src/labeling/segment_episodes.py.

Fast tests use synthetic action arrays. One `slow` test pins the exact real-data
result found by inspecting sampled episodes (4 gripper transitions, 5 phases) as a
regression check — if the recorded action convention ever changes, this fails loudly
instead of the probe silently training on garbage labels.

Run with::

    python -m pytest tests/test_segment_episodes.py -v -m "not slow"
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.env_setup import setup_env  # noqa: E402

setup_env()

from src.labeling.segment_episodes import (  # noqa: E402
    gripper_transition_steps,
    main,
    segment_by_gripper_transitions,
)


def _actions(gripper_signal: list[float]) -> np.ndarray:
    """A minimal (T, 7) action array with only the gripper channel populated."""
    actions = np.zeros((len(gripper_signal), 7))
    actions[:, -1] = gripper_signal
    return actions


def test_gripper_transition_steps_detects_sign_flips():
    actions = _actions([-1, -1, -1, 1, 1, -1, -1])
    steps = gripper_transition_steps(actions)
    np.testing.assert_array_equal(steps, [3, 5])


def test_gripper_transition_steps_empty_when_no_flip():
    actions = _actions([-1, -1, -1, -1])
    assert gripper_transition_steps(actions).size == 0


def test_gripper_transition_steps_ignores_repeated_same_sign_values():
    actions = _actions([1, 1, 1, 1])
    assert gripper_transition_steps(actions).size == 0


def test_segment_by_gripper_transitions_increments_at_each_transition():
    actions = _actions([-1, -1, -1, 1, 1, -1, -1])
    labels = segment_by_gripper_transitions(actions)
    np.testing.assert_array_equal(labels, [0, 0, 0, 1, 1, 2, 2])


def test_segment_by_gripper_transitions_single_phase_when_never_transitions():
    actions = _actions([-1, -1, -1, -1])
    labels = segment_by_gripper_transitions(actions)
    np.testing.assert_array_equal(labels, [0, 0, 0, 0])


def test_segment_by_gripper_transitions_same_length_as_input():
    actions = _actions([-1, 1, -1, 1, -1])
    assert len(segment_by_gripper_transitions(actions)) == len(actions)


def test_main_writes_phase_labels_npz(monkeypatch, tmp_path):
    class FakeDataset:
        num_episodes = 3

    monkeypatch.setattr("lerobot.datasets.lerobot_dataset.LeRobotDataset", lambda repo_id: FakeDataset())
    episodes = {
        0: (-1, -1, -1, 1, 1),          # 1 transition -> 2 phases
        1: (-1, -1, -1, -1),            # 0 transitions -> 1 phase (degenerate)
        2: (-1, 1, -1, 1, -1, 1),       # 5 transitions -> 6 phases
    }

    def fake_load_episode(dataset, episode_index):
        return "instr", _actions(list(episodes[episode_index])), np.zeros((len(episodes[episode_index]), 8))

    monkeypatch.setattr("src.labeling.segment_episodes.load_episode", fake_load_episode)
    out_path = tmp_path / "phase_labels.npz"
    monkeypatch.setattr("src.labeling.segment_episodes.PHASE_LABELS_PATH", out_path)

    status = main([])
    assert status == 0
    assert out_path.exists()

    with np.load(out_path) as archive:
        assert set(archive.files) == {"0", "1", "2"}
        np.testing.assert_array_equal(archive["1"], [0, 0, 0, 0])   # the degenerate one
        assert int(archive["2"].max()) + 1 == 6


def test_main_respects_task_indices(monkeypatch, tmp_path):
    class FakeDataset:
        num_episodes = 10

    monkeypatch.setattr("lerobot.datasets.lerobot_dataset.LeRobotDataset", lambda repo_id: FakeDataset())
    monkeypatch.setattr("src.labeling.segment_episodes.episodes_for_task",
                        lambda dataset, t: {5: [50, 51], 7: [70]}[t])
    monkeypatch.setattr("src.labeling.segment_episodes.load_episode",
                        lambda dataset, e: ("instr", _actions([-1, 1, -1]), np.zeros((3, 8))))
    out_path = tmp_path / "phase_labels.npz"
    monkeypatch.setattr("src.labeling.segment_episodes.PHASE_LABELS_PATH", out_path)

    main(["--task-indices", "5", "7"])

    with np.load(out_path) as archive:
        assert set(archive.files) == {"50", "51", "70"}


# ------------------------------------------------------------------------- slow


@pytest.mark.slow
def test_real_episode_transition_count_matches_known_good_value():
    """Regression pin: episode 18 ('put mug 1 on plate 1, put mug 2 on plate 2') was
    inspected directly and has exactly 4 gripper transitions (5 phases) — two full
    grasp/release cycles. The flips happen between steps (42, 43), (133, 134),
    (213, 214), (258, 259); gripper_transition_steps reports the first step of the
    new command, i.e. the second of each pair. If the dataset's action convention
    ever changes, this fails loudly instead of the probe silently training on
    garbage labels."""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    from src.utils import load_episode

    dataset = LeRobotDataset("lerobot/libero_10")
    _instruction, actions, _state = load_episode(dataset, 18)
    steps = gripper_transition_steps(actions)
    assert steps.tolist() == [43, 134, 214, 259]
    labels = segment_by_gripper_transitions(actions)
    assert int(labels.max()) + 1 == 5
