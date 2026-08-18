"""Mid-rollout MuJoCo-state perturbations for the error-recovery ablation.

Primitives verified against the installed robosuite/libero source: displace the
target object, force-open the gripper, and undo completed progress by restoring a
pre-episode snapshot. Obstacle injection (the "secondary, novel-skill" perturbation
in the plan) is out of scope for this pass — it requires adding a body to the MJCF at
env-build time, not a runtime qpos edit; the plan's own descope order lists it second.

Every mutation goes through `_refresh`, matching what `set_init_state` does
internally: `sim.forward()` alone updates physics but not the observation dict or
`check_success()`'s cached state.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np


class PerturbationKind(str, Enum):
    NONE = "none"
    DISPLACE_OBJECT = "displace_object"
    FORCE_GRIPPER_OPEN = "force_gripper_open"
    UNDO_PROGRESS = "undo_progress"


DEFAULT_DISPLACEMENT_RADIUS_M = 0.08

# Panda finger joints, fully open. Matched by suffix rather than a hardcoded
# "gripper0_" prefix, since env.robots[0].gripper.joints returns the model's actual
# (possibly differently-prefixed) names.
_FINGER_OPEN_QPOS = {"finger_joint1": 0.04, "finger_joint2": -0.04}


@dataclass(frozen=True)
class TriggerCondition:
    """When a perturbation fires. Exactly one field is set.

    `at_step`: fires the first step where the post-settle step counter reaches this
    value (0-indexed).
    `after_success_steps`: fires this many steps after the *first* observed
    `check_success() == True`, whenever that happens. Needed in practice for
    UNDO_PROGRESS — undoing "completed progress" presupposes progress was completed —
    but not enforced as a hard constraint on the other two kinds here.
    """

    at_step: int | None = None
    after_success_steps: int | None = None

    def __post_init__(self) -> None:
        set_count = sum(x is not None for x in (self.at_step, self.after_success_steps))
        if set_count != 1:
            raise ValueError("exactly one of at_step / after_success_steps must be set")
        for name, value in (("at_step", self.at_step), ("after_success_steps", self.after_success_steps)):
            if value is not None and value < 0:
                raise ValueError(f"{name} must be >= 0")


@dataclass(frozen=True)
class PerturbationSpec:
    """One perturbation to inject, fully describing what/when/how much."""

    kind: PerturbationKind
    trigger: TriggerCondition | None       # None iff kind is NONE
    episode_seed: tuple[int, int]          # (cfg.seed, episode_index) -> rng
    displacement_radius_m: float = DEFAULT_DISPLACEMENT_RADIUS_M

    def __post_init__(self) -> None:
        if (self.kind is PerturbationKind.NONE) != (self.trigger is None):
            raise ValueError("trigger must be set iff kind is not NONE")

    def make_rng(self) -> np.random.Generator:
        return np.random.default_rng(self.episode_seed)


class PerturbationScheduler:
    """Tracks whether/when one episode's perturbation should fire.

    One instance per episode. Call `.observe(step_index, success)` every step before
    `.should_fire()` — the after_success_steps trigger needs to know the first
    success step, which `observe` latches.
    """

    def __init__(self, spec: PerturbationSpec) -> None:
        self._spec = spec
        self._first_success_step: int | None = None
        self._fired = False
        self._fired_at_step: int | None = None
        self._last_step_index: int | None = None

    def observe(self, step_index: int, success: bool) -> None:
        self._last_step_index = step_index
        if success and self._first_success_step is None:
            self._first_success_step = step_index

    def should_fire(self) -> bool:
        """True exactly once: on the first step where the condition holds.

        Returns False forever once already fired, and always False for NONE.
        """
        if self._fired or self._spec.kind is PerturbationKind.NONE:
            return False
        trigger = self._spec.trigger
        step = self._last_step_index
        if trigger.at_step is not None:
            due = step >= trigger.at_step
        else:
            due = (self._first_success_step is not None
                   and step >= self._first_success_step + trigger.after_success_steps)
        if due:
            self._fired = True
            self._fired_at_step = step
        return due

    @property
    def fired(self) -> bool:
        return self._fired

    @property
    def fired_at_step(self) -> int | None:
        return self._fired_at_step

    @property
    def first_success_step(self) -> int | None:
        return self._first_success_step


# --------------------------------------------------------------------- primitives


def target_object_joint_name(env) -> str:
    """Free joint of the primary target object (index 0 of the BDDL obj_of_interest)."""
    target_name = env.obj_of_interest[0]
    obj = env.env.objects_dict[target_name]
    return obj.joints[-1]


def snapshot_target_object_pose(env) -> np.ndarray:
    """7-vector [x,y,z,qw,qx,qy,qz] of the target object's free joint.

    Call once per episode, immediately after the initial reset/set_init_state and
    before any policy step — this is the baseline UNDO_PROGRESS restores.
    """
    joint_name = target_object_joint_name(env)
    return np.array(env.sim.data.get_joint_qpos(joint_name), dtype=np.float64, copy=True)


def _refresh(env) -> dict:
    """Push a direct sim.data mutation through to the observation/success machinery."""
    env.sim.forward()
    return env.set_init_state(env.get_sim_state())


def displace_object(env, rng: np.random.Generator,
                    radius_m: float = DEFAULT_DISPLACEMENT_RADIUS_M) -> dict:
    """Jitter the target object's xy position by `radius_m` in a uniformly random
    planar direction, keeping z and orientation. Returns the refreshed observation."""
    joint_name = target_object_joint_name(env)
    pose = np.array(env.sim.data.get_joint_qpos(joint_name), dtype=np.float64, copy=True)
    angle = rng.uniform(0.0, 2.0 * np.pi)
    pose[0] += radius_m * np.cos(angle)
    pose[1] += radius_m * np.sin(angle)
    env.sim.data.set_joint_qpos(joint_name, pose)
    return _refresh(env)


def force_gripper_open(env) -> dict:
    """Set both Panda finger joints to their fully-open qpos.

    Raises RuntimeError on an unrecognised joint name — only the Panda gripper,
    LIBERO's fixed choice, is supported.
    """
    for joint_name in env.robots[0].gripper.joints:
        for suffix, target in _FINGER_OPEN_QPOS.items():
            if joint_name.endswith(suffix):
                env.sim.data.set_joint_qpos(joint_name, target)
                break
        else:
            raise RuntimeError(
                f"unrecognised gripper joint {joint_name!r}; expected a suffix in "
                f"{list(_FINGER_OPEN_QPOS)} (Panda gripper assumed)"
            )
    return _refresh(env)


def undo_progress(env, snapshot: np.ndarray) -> dict:
    """Restore the target object's free-joint pose to `snapshot`."""
    joint_name = target_object_joint_name(env)
    env.sim.data.set_joint_qpos(joint_name, np.asarray(snapshot, dtype=np.float64))
    return _refresh(env)


def apply_perturbation(env, kind: PerturbationKind, rng: np.random.Generator,
                       snapshot: np.ndarray, radius_m: float) -> dict:
    """Dispatch to the primitive above; returns the refreshed observation dict.

    Callers MUST use this return value for the next `model.act()` call and for the
    post-perturbation cosine-probe frame — the mutation alone does not update the
    environment's cached observation.
    """
    if kind is PerturbationKind.DISPLACE_OBJECT:
        return displace_object(env, rng, radius_m)
    if kind is PerturbationKind.FORCE_GRIPPER_OPEN:
        return force_gripper_open(env)
    if kind is PerturbationKind.UNDO_PROGRESS:
        return undo_progress(env, snapshot)
    raise ValueError(f"cannot apply {kind}")
