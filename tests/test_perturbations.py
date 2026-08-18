"""Tests for eval/perturbations.py.

Fast tests use `tests/_fake_env.py`'s `FakeLiberoEnv`. A handful of `slow` tests
exercise the same primitives against a real LIBERO `OffScreenRenderEnv`, since the
fake env's attribute names are contrived and only a real env proves
`target_object_joint_name`/`force_gripper_open` actually resolve against real
robosuite/LIBERO objects.

Run with::

    python -m pytest tests/test_perturbations.py -v
    python -m pytest tests/test_perturbations.py -v -m "not slow"
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eval.perturbations import (  # noqa: E402
    DEFAULT_DISPLACEMENT_RADIUS_M,
    PerturbationKind,
    PerturbationScheduler,
    PerturbationSpec,
    TriggerCondition,
    apply_perturbation,
    displace_object,
    force_gripper_open,
    snapshot_target_object_pose,
    target_object_joint_name,
    undo_progress,
)
from _fake_env import FakeLiberoEnv  # noqa: E402


# ------------------------------------------------------------------------ primitives


def test_target_object_joint_name_uses_index_zero_of_obj_of_interest():
    env = FakeLiberoEnv(obj_of_interest=["alphabet_soup_1", "basket_1"],
                        joint_name="alphabet_soup_1_joint0")
    assert target_object_joint_name(env) == "alphabet_soup_1_joint0"


def test_displace_object_moves_xy_only():
    env = FakeLiberoEnv()
    before = env.sim.data.get_joint_qpos(env.joint_name).copy()
    rng = np.random.default_rng(0)
    displace_object(env, rng, radius_m=0.1)
    after = env.sim.data.get_joint_qpos(env.joint_name)

    assert after[2] == pytest.approx(before[2])          # z unchanged
    np.testing.assert_allclose(after[3:], before[3:])     # orientation unchanged
    displacement = np.linalg.norm(after[:2] - before[:2])
    assert displacement == pytest.approx(0.1)


def test_displace_object_is_reproducible_with_same_rng_seed():
    env_a = FakeLiberoEnv()
    env_b = FakeLiberoEnv()
    spec_a = PerturbationSpec(kind=PerturbationKind.DISPLACE_OBJECT,
                              trigger=TriggerCondition(at_step=0), episode_seed=(0, 5))
    spec_b = PerturbationSpec(kind=PerturbationKind.DISPLACE_OBJECT,
                              trigger=TriggerCondition(at_step=0), episode_seed=(0, 5))
    spec_c = PerturbationSpec(kind=PerturbationKind.DISPLACE_OBJECT,
                              trigger=TriggerCondition(at_step=0), episode_seed=(0, 6))

    displace_object(env_a, spec_a.make_rng())
    displace_object(env_b, spec_b.make_rng())
    pose_a = env_a.sim.data.get_joint_qpos(env_a.joint_name)
    pose_b = env_b.sim.data.get_joint_qpos(env_b.joint_name)
    np.testing.assert_allclose(pose_a, pose_b)

    env_c = FakeLiberoEnv()
    displace_object(env_c, spec_c.make_rng())
    pose_c = env_c.sim.data.get_joint_qpos(env_c.joint_name)
    assert not np.allclose(pose_a, pose_c)


def test_displace_object_calls_refresh():
    env = FakeLiberoEnv()
    displace_object(env, np.random.default_rng(0))
    assert env.sim.forward_calls == 1
    assert env.set_init_state_calls == 1


def test_force_gripper_open_sets_both_fingers_to_max_magnitude():
    env = FakeLiberoEnv(gripper_joint_names=("robotX_finger_joint1", "robotX_finger_joint2"))
    force_gripper_open(env)
    assert float(env.sim.data.get_joint_qpos("robotX_finger_joint1")) == pytest.approx(0.04)
    assert float(env.sim.data.get_joint_qpos("robotX_finger_joint2")) == pytest.approx(-0.04)


def test_force_gripper_open_rejects_unknown_joint_name():
    env = FakeLiberoEnv()
    env.robots[0].gripper.joints = ["some_other_joint"]
    with pytest.raises(RuntimeError, match="unrecognised gripper joint"):
        force_gripper_open(env)


def test_undo_progress_restores_the_exact_snapshot():
    env = FakeLiberoEnv()
    snapshot = snapshot_target_object_pose(env)
    displace_object(env, np.random.default_rng(0))
    assert not np.allclose(env.sim.data.get_joint_qpos(env.joint_name), snapshot)

    undo_progress(env, snapshot)
    np.testing.assert_allclose(env.sim.data.get_joint_qpos(env.joint_name), snapshot)


@pytest.mark.parametrize("kind", [
    PerturbationKind.DISPLACE_OBJECT, PerturbationKind.FORCE_GRIPPER_OPEN, PerturbationKind.UNDO_PROGRESS,
])
def test_apply_perturbation_dispatches_by_kind(monkeypatch, kind):
    env = FakeLiberoEnv()
    calls = []
    for name in ("displace_object", "force_gripper_open", "undo_progress"):
        def spy(*args, _name=name, **kwargs):
            calls.append(_name)
            return {}
        monkeypatch.setattr(f"eval.perturbations.{name}", spy)

    apply_perturbation(env, kind, np.random.default_rng(0), np.zeros(7), 0.1)
    assert calls == [kind.value]


def test_apply_perturbation_rejects_none():
    with pytest.raises(ValueError):
        apply_perturbation(FakeLiberoEnv(), PerturbationKind.NONE, np.random.default_rng(0), np.zeros(7), 0.1)


# --------------------------------------------------------------------------- specs


def test_trigger_condition_rejects_zero_or_two_fields_set():
    with pytest.raises(ValueError):
        TriggerCondition()
    with pytest.raises(ValueError):
        TriggerCondition(at_step=1, after_success_steps=1)


def test_trigger_condition_rejects_negative_values():
    with pytest.raises(ValueError):
        TriggerCondition(at_step=-1)


def test_perturbation_spec_rejects_trigger_none_mismatch():
    with pytest.raises(ValueError):
        PerturbationSpec(kind=PerturbationKind.NONE, trigger=TriggerCondition(at_step=0), episode_seed=(0, 0))
    with pytest.raises(ValueError):
        PerturbationSpec(kind=PerturbationKind.DISPLACE_OBJECT, trigger=None, episode_seed=(0, 0))


# ---------------------------------------------------------------------- scheduler


def test_scheduler_fires_at_exact_step_for_at_step_trigger():
    spec = PerturbationSpec(kind=PerturbationKind.DISPLACE_OBJECT,
                            trigger=TriggerCondition(at_step=3), episode_seed=(0, 0))
    scheduler = PerturbationScheduler(spec)
    for step in range(3):
        scheduler.observe(step, success=False)
        assert scheduler.should_fire() is False

    scheduler.observe(3, success=False)
    assert scheduler.should_fire() is True
    assert scheduler.fired_at_step == 3

    # one-shot: stays fired, never fires again
    scheduler.observe(4, success=False)
    assert scheduler.should_fire() is False


def test_scheduler_waits_for_first_success_before_after_success_steps_trigger():
    spec = PerturbationSpec(kind=PerturbationKind.UNDO_PROGRESS,
                            trigger=TriggerCondition(after_success_steps=2), episode_seed=(0, 0))
    scheduler = PerturbationScheduler(spec)
    for step in range(5):
        scheduler.observe(step, success=False)
        assert scheduler.should_fire() is False

    scheduler.observe(5, success=True)          # first success at step 5
    assert scheduler.first_success_step == 5
    assert scheduler.should_fire() is False      # 5 + 2 = 7, not yet

    scheduler.observe(6, success=True)
    assert scheduler.should_fire() is False

    scheduler.observe(7, success=True)
    assert scheduler.should_fire() is True
    assert scheduler.fired_at_step == 7


def test_scheduler_never_fires_if_success_never_happens():
    spec = PerturbationSpec(kind=PerturbationKind.UNDO_PROGRESS,
                            trigger=TriggerCondition(after_success_steps=1), episode_seed=(0, 0))
    scheduler = PerturbationScheduler(spec)
    for step in range(50):
        scheduler.observe(step, success=False)
        assert scheduler.should_fire() is False
    assert scheduler.fired is False


def test_scheduler_none_kind_never_fires():
    spec = PerturbationSpec(kind=PerturbationKind.NONE, trigger=None, episode_seed=(0, 0))
    scheduler = PerturbationScheduler(spec)
    scheduler.observe(0, success=True)
    assert scheduler.should_fire() is False


# ------------------------------------------------------------------------- slow


@pytest.mark.slow
def test_displace_object_actually_moves_the_object_in_sim():
    from src.env_setup import setup_env

    setup_env()
    import os

    os.environ.setdefault("MUJOCO_GL", "egl")
    from libero.libero.envs import OffScreenRenderEnv

    from src.utils import TaskMapping

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    dataset = LeRobotDataset("lerobot/libero_10")
    mapping = TaskMapping.from_dataset(dataset)
    task = mapping.by_dataset_index(0)

    env = OffScreenRenderEnv(bddl_file_name=task.bddl, camera_heights=128, camera_widths=128)
    try:
        env.seed(0)
        env.reset()
        joint_name = target_object_joint_name(env)
        before = np.array(env.sim.data.get_joint_qpos(joint_name), copy=True)
        displace_object(env, np.random.default_rng(0), radius_m=0.08)
        after = np.array(env.sim.data.get_joint_qpos(joint_name), copy=True)
        assert np.linalg.norm(after[:2] - before[:2]) == pytest.approx(0.08, abs=1e-6)
    finally:
        env.close()


@pytest.mark.slow
def test_force_gripper_open_changes_gripper_qpos_observation():
    from src.env_setup import setup_env

    setup_env()
    import os

    os.environ.setdefault("MUJOCO_GL", "egl")
    from libero.libero.envs import OffScreenRenderEnv

    from src.utils import TaskMapping

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    dataset = LeRobotDataset("lerobot/libero_10")
    mapping = TaskMapping.from_dataset(dataset)
    task = mapping.by_dataset_index(0)

    env = OffScreenRenderEnv(bddl_file_name=task.bddl, camera_heights=128, camera_widths=128)
    try:
        env.seed(0)
        env.reset()
        obs = force_gripper_open(env)
        assert "robot0_gripper_qpos" in obs
        # Fully open Panda fingers sit near their qpos range limits (0.04 / -0.04).
        assert max(abs(v) for v in obs["robot0_gripper_qpos"]) > 0.03
    finally:
        env.close()


@pytest.mark.slow
def test_check_success_does_not_raise_after_a_refresh():
    from src.env_setup import setup_env

    setup_env()
    import os

    os.environ.setdefault("MUJOCO_GL", "egl")
    from libero.libero.envs import OffScreenRenderEnv

    from src.utils import TaskMapping

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    dataset = LeRobotDataset("lerobot/libero_10")
    mapping = TaskMapping.from_dataset(dataset)
    task = mapping.by_dataset_index(0)

    env = OffScreenRenderEnv(bddl_file_name=task.bddl, camera_heights=128, camera_widths=128)
    try:
        env.seed(0)
        env.reset()
        snapshot = snapshot_target_object_pose(env)
        displace_object(env, np.random.default_rng(0))
        undo_progress(env, snapshot)
        assert isinstance(env.check_success(), (bool, np.bool_))
    finally:
        env.close()
