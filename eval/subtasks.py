"""Ordered subtask decomposition of the libero_10 tasks, checkable in the simulator.

A libero_10 task is long-horizon: "put both the alphabet soup and the tomato sauce in
the basket" is four manipulations, and `env.check_success()` — a conjunction over the
BDDL goal state — reports only whether all of them landed. A rollout that picks and
places the soup and then never touches the sauce is indistinguishable from one that
never moved, which is the wrong resolution for diagnosing where a policy stalls.

Each subtask here answers one such step with `is_done(env)`, evaluated against the
live simulator:

**goal conditions** delegate to the environment's own `_eval_predicate`, passing a
literal taken verbatim from `parsed_problem["goal_state"]`. A place subtask is
therefore true by exactly the benchmark's definition of success — `In`/`On`/`Close`/
`Turnon` are never reimplemented here, and `tests/test_subtasks.py` asserts every
literal in this table is one the BDDL file actually declares.

**grasps** have no BDDL counterpart — the goal state describes where objects end up,
not that the gripper ever held them — so they use robosuite's `_check_grasp` against
the object's contact geoms. This is the distinction that separates "never reached the
object" from "grasped it and dropped it", the two failures a bare success rate merges.

The tables are hand-written rather than derived from the goal state. Ordering is a
semantic judgement a parser cannot make (the drawer must be closed *after* the bowl
goes in, not before), and the grasp steps have no source to be derived from.

Subtasks are tracked independently rather than as a prefix: the goal is a conjunction,
so a policy is free to place the sauce before the soup, and scoring the longest
completed prefix would report no progress for a rollout that solved the task in the
other order.
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass

SUITE = "libero_10"


def _domain(env):
    """The robosuite `BDDLBaseDomain` behind LIBERO's `ControlEnv` wrapper.

    Accepts either the wrapper or the domain itself, so a caller holding one of the
    `libero.libero.envs` env classes does not have to reach through `.env` by hand.
    """
    return getattr(env, "env", env)


class Subtask(ABC):
    """One checkable step of a task, with a stable id and a human-readable name."""

    description: str

    @property
    @abstractmethod
    def id(self) -> str:
        """Stable across runs — it keys per-subtask aggregation over the JSONL logs."""

    @abstractmethod
    def is_done(self, env) -> bool:
        """Whether this step is satisfied *right now*, in the live simulator.

        Not latching: a grasped object can be dropped and a placed object knocked out
        of its region. `SubtaskTracker` is what turns this into "was it ever true".
        """


@dataclass(frozen=True)
class Grasp(Subtask):
    """The gripper is holding `object_name`.

    True while both finger pads contact the object's geoms, so it goes false again on
    release — including the intended release that completes the following place step.
    """

    object_name: str
    description: str

    @property
    def id(self) -> str:
        return f"grasp:{self.object_name}"

    def is_done(self, env) -> bool:
        domain = _domain(env)
        return bool(domain._check_grasp(domain.robots[0].gripper,
                                        domain.objects_dict[self.object_name]))


@dataclass(frozen=True)
class GoalCondition(Subtask):
    """One literal of the task's BDDL goal state.

    `literal` is the parsed form — a lowercased predicate name followed by one or two
    object names, exactly as `parsed_problem["goal_state"]` holds it.
    """

    literal: tuple[str, ...]
    description: str

    @property
    def id(self) -> str:
        return ":".join(self.literal)

    def is_done(self, env) -> bool:
        return bool(_domain(env)._eval_predicate(list(self.literal)))


# Keyed by BDDL file basename, which is what `Trial.bddl` carries and what identifies a
# task independently of its index in any particular dataset.
SUBTASKS: dict[str, tuple[Subtask, ...]] = {
    "KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it.bddl": (
        GoalCondition(("turnon", "flat_stove_1"), "turn on the stove"),
        Grasp("moka_pot_1", "pick up the moka pot"),
        GoalCondition(("on", "moka_pot_1", "flat_stove_1_cook_region"),
                      "place the moka pot on the stove"),
    ),
    # The drawer is open at reset — `(:init (open white_cabinet_1_bottom_region))` — so
    # closing it is the last step, not a matching open/close pair.
    "KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it.bddl": (
        Grasp("akita_black_bowl_1", "pick up the black bowl"),
        GoalCondition(("in", "akita_black_bowl_1", "white_cabinet_1_bottom_region"),
                      "place the black bowl in the bottom drawer"),
        GoalCondition(("close", "white_cabinet_1_bottom_region"), "close the drawer"),
    ),
    # The microwave likewise starts open.
    "KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it.bddl": (
        Grasp("white_yellow_mug_1", "pick up the yellow and white mug"),
        GoalCondition(("in", "white_yellow_mug_1", "microwave_1_heating_region"),
                      "place the mug in the microwave"),
        GoalCondition(("close", "microwave_1"), "close the microwave"),
    ),
    # The stove is already on at reset, so this goal condition is satisfied before the
    # first action. It stays in the table because a rollout that knocks the knob off
    # fails the task, and only a per-condition record shows that as the cause.
    "KITCHEN_SCENE8_put_both_moka_pots_on_the_stove.bddl": (
        Grasp("moka_pot_1", "pick up the first moka pot"),
        GoalCondition(("on", "moka_pot_1", "flat_stove_1_cook_region"),
                      "place the first moka pot on the stove"),
        Grasp("moka_pot_2", "pick up the second moka pot"),
        GoalCondition(("on", "moka_pot_2", "flat_stove_1_cook_region"),
                      "place the second moka pot on the stove"),
        GoalCondition(("turnon", "flat_stove_1"), "keep the stove turned on"),
    ),
    "LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket.bddl": (
        Grasp("alphabet_soup_1", "pick up the alphabet soup"),
        GoalCondition(("in", "alphabet_soup_1", "basket_1_contain_region"),
                      "place the alphabet soup in the basket"),
        Grasp("cream_cheese_1", "pick up the cream cheese box"),
        GoalCondition(("in", "cream_cheese_1", "basket_1_contain_region"),
                      "place the cream cheese box in the basket"),
    ),
    "LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket.bddl": (
        Grasp("alphabet_soup_1", "pick up the alphabet soup"),
        GoalCondition(("in", "alphabet_soup_1", "basket_1_contain_region"),
                      "place the alphabet soup in the basket"),
        Grasp("tomato_sauce_1", "pick up the tomato sauce"),
        GoalCondition(("in", "tomato_sauce_1", "basket_1_contain_region"),
                      "place the tomato sauce in the basket"),
    ),
    "LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket.bddl": (
        Grasp("cream_cheese_1", "pick up the cream cheese box"),
        GoalCondition(("in", "cream_cheese_1", "basket_1_contain_region"),
                      "place the cream cheese box in the basket"),
        Grasp("butter_1", "pick up the butter"),
        GoalCondition(("in", "butter_1", "basket_1_contain_region"),
                      "place the butter in the basket"),
    ),
    "LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate.bddl": (
        Grasp("porcelain_mug_1", "pick up the white mug"),
        GoalCondition(("on", "porcelain_mug_1", "plate_1"),
                      "place the white mug on the left plate"),
        Grasp("white_yellow_mug_1", "pick up the yellow and white mug"),
        GoalCondition(("on", "white_yellow_mug_1", "plate_2"),
                      "place the yellow and white mug on the right plate"),
    ),
    "LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate.bddl": (
        Grasp("porcelain_mug_1", "pick up the white mug"),
        GoalCondition(("on", "porcelain_mug_1", "plate_1"),
                      "place the white mug on the plate"),
        Grasp("chocolate_pudding_1", "pick up the chocolate pudding"),
        GoalCondition(("on", "chocolate_pudding_1", "living_room_table_plate_right_region"),
                      "place the chocolate pudding to the right of the plate"),
    ),
    "STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy.bddl": (
        Grasp("black_book_1", "pick up the book"),
        GoalCondition(("in", "black_book_1", "desk_caddy_1_back_contain_region"),
                      "place the book in the back compartment of the caddy"),
    ),
}


def subtasks_for(bddl: str) -> tuple[Subtask, ...]:
    """The ordered subtasks of the task defined by `bddl` (a path or a basename)."""
    key = os.path.basename(bddl)
    try:
        return SUBTASKS[key]
    except KeyError:
        raise KeyError(f"no subtask decomposition for {key}; eval/subtasks.py covers "
                       f"{SUITE} only") from None


@dataclass
class SubtaskRecord:
    """What one subtask did over one rollout."""

    id: str
    description: str
    achieved_at_reset: bool     # true before the first action — the stove in
                                # KITCHEN_SCENE8 is already on, and crediting that to
                                # the policy would overstate its progress
    first_achieved_step: int | None    # None if never satisfied during the rollout
    achieved_at_end: bool       # the final reading; differs from `first_achieved_step
                                # is not None` for transient steps — every grasp, and
                                # any placement a later manipulation disturbs


class SubtaskTracker:
    """Latches when each subtask of one trial first becomes true.

    Constructed after the environment is reset and settled, so `achieved_at_reset`
    reflects the trial's starting configuration rather than the previous episode's.
    """

    def __init__(self, subtasks: tuple[Subtask, ...], env) -> None:
        self._subtasks = subtasks
        self._env = env
        self._current = [s.is_done(env) for s in subtasks]
        self._at_reset = list(self._current)
        self._first_step: list[int | None] = [None] * len(subtasks)

    def update(self, step_index: int) -> None:
        """Re-evaluate every subtask; record `step_index` for any newly satisfied one."""
        for i, subtask in enumerate(self._subtasks):
            done = subtask.is_done(self._env)
            self._current[i] = done
            if done and self._first_step[i] is None:
                self._first_step[i] = step_index

    @property
    def n_total(self) -> int:
        """Subtasks the policy actually has to accomplish.

        Excludes anything already satisfied at reset, so the counts read as progress
        made out of progress available: KITCHEN_SCENE8 scores out of 4, not 5, because
        its stove is on before the first action. Every subtask, reset-satisfied or not,
        still appears in `records()`.
        """
        return sum(not at_reset for at_reset in self._at_reset)

    @property
    def n_achieved(self) -> int:
        return sum(step is not None and not at_reset
                   for step, at_reset in zip(self._first_step, self._at_reset))

    def records(self) -> list[SubtaskRecord]:
        return [SubtaskRecord(id=s.id, description=s.description,
                              achieved_at_reset=self._at_reset[i],
                              first_achieved_step=self._first_step[i],
                              achieved_at_end=self._current[i])
                for i, s in enumerate(self._subtasks)]

    def as_dicts(self) -> list[dict]:
        return [asdict(r) for r in self.records()]
