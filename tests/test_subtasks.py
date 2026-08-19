"""Checks the hand-written subtask tables against the BDDL files they describe.

A wrong object name or a mistyped predicate in `eval/subtasks.py` would not raise
until a rollout was already running, and a goal literal that does not match its BDDL
file would silently report a subtask that can never complete. Both are caught here,
offline, without a simulator.
"""

from __future__ import annotations

import glob
import os

import pytest

from src.env_setup import setup_env

setup_env()

from eval.subtasks import (  # noqa: E402
    SUBTASKS,
    SUITE,
    GoalCondition,
    Grasp,
    Subtask,
    SubtaskRecord,
    SubtaskTracker,
    subtasks_for,
)


def _parse(bddl_path: str) -> dict:
    import libero.libero.envs.bddl_utils as bddl_utils

    return bddl_utils.robosuite_parse_problem(bddl_path)


def _suite_bddl_paths() -> dict[str, str]:
    from libero.libero import get_libero_path

    root = os.path.join(get_libero_path("bddl_files"), SUITE)
    return {os.path.basename(p): p for p in sorted(glob.glob(os.path.join(root, "*.bddl")))}


SUITE_BDDLS = _suite_bddl_paths()


def test_every_suite_task_has_a_decomposition():
    assert set(SUBTASKS) == set(SUITE_BDDLS)


@pytest.mark.parametrize("name", sorted(SUBTASKS))
def test_goal_conditions_match_the_bddl_goal_state(name):
    """Every goal-condition subtask is a literal the BDDL declares, and vice versa.

    The forward direction keeps `_eval_predicate` from being handed a literal it
    cannot evaluate; the reverse keeps a task's success from depending on a condition
    no subtask reports on.
    """
    goal_state = {tuple(literal) for literal in _parse(SUITE_BDDLS[name])["goal_state"]}
    declared = {s.literal for s in SUBTASKS[name] if isinstance(s, GoalCondition)}
    assert declared == goal_state


@pytest.mark.parametrize("name", sorted(SUBTASKS))
def test_grasp_targets_are_objects_the_problem_defines(name):
    """Grasp targets must be movable object *instances*.

    `_check_grasp` looks the name up in `objects_dict`, which holds only movable
    objects — a fixture name (a stove, a cabinet) parses fine but raises at rollout
    time.
    """
    problem = _parse(SUITE_BDDLS[name])
    instances = {instance for group in problem["objects"].values() for instance in group}
    targets = {s.object_name for s in SUBTASKS[name] if isinstance(s, Grasp)}
    assert targets <= instances


@pytest.mark.parametrize("name", sorted(SUBTASKS))
def test_every_relocated_object_has_a_grasp_subtask(name):
    """An object the goal state moves somewhere must be picked up first.

    Without the grasp step, "never reached the object" and "grasped it but placed it
    badly" both show up as a single unmet place condition.
    """
    problem = _parse(SUITE_BDDLS[name])
    instances = {instance for group in problem["objects"].values() for instance in group}
    relocated = {literal[1] for literal in problem["goal_state"]
                 if literal[0] in ("in", "on") and literal[1] in instances}
    grasped = {s.object_name for s in SUBTASKS[name] if isinstance(s, Grasp)}
    assert relocated == grasped


@pytest.mark.parametrize("name", sorted(SUBTASKS))
def test_subtask_ids_are_unique_within_a_task(name):
    ids = [s.id for s in SUBTASKS[name]]
    assert len(ids) == len(set(ids))


def test_subtasks_for_accepts_a_path_or_a_basename():
    name, path = next(iter(SUITE_BDDLS.items()))
    assert subtasks_for(path) is subtasks_for(name)


def test_subtasks_for_rejects_an_unknown_task():
    with pytest.raises(KeyError, match="no subtask decomposition"):
        subtasks_for("NOT_A_LIBERO_TASK.bddl")


# ------------------------------------------------------------------ tracker latching


class _ScriptedSubtask(Subtask):
    """Reports `True` at exactly the step indices it is given."""

    def __init__(self, name: str, true_at: set[int]) -> None:
        self.description = name
        self._name = name
        self._true_at = true_at
        self.step = -1

    @property
    def id(self) -> str:
        return self._name

    def is_done(self, env) -> bool:
        return self.step in self._true_at


class _Clock:
    """Drives a set of `_ScriptedSubtask`s from one step counter."""

    def __init__(self, subtasks) -> None:
        self._subtasks = subtasks

    def advance(self, step: int) -> None:
        for s in self._subtasks:
            s.step = step


def test_tracker_latches_the_first_step_and_keeps_the_final_reading():
    pick = _ScriptedSubtask("pick", {1, 2})          # released again after step 2
    place = _ScriptedSubtask("place", {3, 4})
    clock = _Clock([pick, place])

    clock.advance(-1)
    tracker = SubtaskTracker((pick, place), env=None)
    for step in range(5):
        clock.advance(step)
        tracker.update(step)

    records = {r.id: r for r in tracker.records()}
    assert records["pick"].first_achieved_step == 1
    assert records["pick"].achieved_at_end is False     # transient, as grasps are
    assert records["place"].first_achieved_step == 3
    assert records["place"].achieved_at_end is True
    assert tracker.n_achieved == 2
    assert tracker.n_total == 2


def test_tracker_separates_reset_state_from_policy_progress():
    """A condition already true at reset is not counted as achieved by the rollout."""
    already = _ScriptedSubtask("already", {-1, 0, 1})
    clock = _Clock([already])

    clock.advance(-1)
    tracker = SubtaskTracker((already,), env=None)
    for step in range(2):
        clock.advance(step)
        tracker.update(step)

    record = tracker.records()[0]
    assert record.achieved_at_reset is True
    assert record.first_achieved_step == 0
    # Excluded from both counts: it was true before the policy acted.
    assert tracker.n_achieved == 0
    assert tracker.n_total == 0


def test_counts_score_progress_out_of_progress_available():
    """KITCHEN_SCENE8's stove is on at reset; a rollout that does nothing scores 0/2,
    not 1/3."""
    stove = _ScriptedSubtask("stove", {-1, 0, 1})
    pick = _ScriptedSubtask("pick", {1})
    place = _ScriptedSubtask("place", set())
    clock = _Clock([stove, pick, place])

    clock.advance(-1)
    tracker = SubtaskTracker((pick, place, stove), env=None)
    for step in range(2):
        clock.advance(step)
        tracker.update(step)

    assert tracker.n_total == 2
    assert tracker.n_achieved == 1
    assert len(tracker.records()) == 3      # the stove still has a record


def test_never_achieved_subtask_reports_none():
    never = _ScriptedSubtask("never", set())
    clock = _Clock([never])

    clock.advance(-1)
    tracker = SubtaskTracker((never,), env=None)
    for step in range(3):
        clock.advance(step)
        tracker.update(step)

    record = tracker.records()[0]
    assert record.achieved_at_reset is False
    assert record.first_achieved_step is None
    assert record.achieved_at_end is False
    assert tracker.n_achieved == 0


def test_as_dicts_is_json_shaped():
    import json

    subtask = _ScriptedSubtask("only", {0})
    subtask.step = -1
    tracker = SubtaskTracker((subtask,), env=None)
    subtask.step = 0
    tracker.update(0)

    payload = tracker.as_dicts()
    assert json.loads(json.dumps(payload)) == payload
    assert set(payload[0]) == set(SubtaskRecord.__dataclass_fields__)
