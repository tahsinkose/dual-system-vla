"""Shared fake LIBERO environment for fast, real-simulator-free tests.

Implements exactly the surface `eval/perturbations.py` and `eval/run_eval.py` touch:
an in-memory qpos dict standing in for MuJoCo's `sim.data`, a scriptable
`check_success()`, and observation dicts shaped like `src/observations.py::from_env`
expects (the flat LIBERO form: `agentview_image`, `robot0_eef_pos`, `robot0_eef_quat`,
`robot0_gripper_qpos`).

Not a mock of any specific real class — a purpose-built double kept as small as the
tests need, so a change here is a deliberate, visible edit rather than a
mock-framework expectation silently drifting from what LIBERO actually does.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

IMAGE_SIZE = 32  # small: these tests never touch a real renderer


def _fake_image() -> np.ndarray:
    return np.zeros((IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8)


class _FakeData:
    """Stands in for `env.sim.data`."""

    def __init__(self, qpos: dict[str, np.ndarray]) -> None:
        self._qpos = qpos

    def get_joint_qpos(self, name: str) -> np.ndarray:
        return self._qpos[name]

    def set_joint_qpos(self, name: str, value) -> None:
        self._qpos[name] = np.asarray(value, dtype=np.float64).copy()


class _FakeSim:
    """Stands in for `env.sim`."""

    def __init__(self, data: _FakeData) -> None:
        self.data = data
        self.forward_calls = 0

    def forward(self) -> None:
        self.forward_calls += 1


class _FakeMujocoObject:
    def __init__(self, joints: list[str]) -> None:
        self.joints = joints


class _FakeGripper:
    def __init__(self, joints: list[str]) -> None:
        self.joints = joints


class _FakeRobot:
    def __init__(self, gripper_joints: list[str]) -> None:
        self.gripper = _FakeGripper(gripper_joints)


class _FakeRobosuiteEnv:
    """Stands in for `env.env` (LIBERO's `BDDLBaseDomain` instance)."""

    def __init__(self, objects_dict: dict[str, _FakeMujocoObject]) -> None:
        self.objects_dict = objects_dict


@dataclass
class FakeLiberoEnv:
    """Minimal double for `OffScreenRenderEnv`.

    `success_at_steps`: the set of `.step()` call indices (0-indexed) at which
    `check_success()` should report True from then on until reset — tests script task
    completion directly rather than simulating real physics.
    """

    obj_of_interest: list[str] = field(default_factory=lambda: ["target_object_1"])
    joint_name: str = "target_object_1_joint0"
    gripper_joint_names: tuple[str, str] = ("gripper0_finger_joint1", "gripper0_finger_joint2")
    success_at_steps: frozenset[int] = field(default_factory=frozenset)
    horizon: int = 10_000

    def __post_init__(self) -> None:
        initial_qpos = {
            self.joint_name: np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]),
            # Single-DOF (hinge/slide) joints store a scalar, unlike the free joint's
            # 7-vector — matching real MuJoCo bindings, where get_joint_qpos on a
            # scalar joint returns a 0-d value, not a 1-element array.
            self.gripper_joint_names[0]: np.array(0.02),
            self.gripper_joint_names[1]: np.array(-0.02),
        }
        self._data = _FakeData(initial_qpos)
        self.sim = _FakeSim(self._data)
        self.env = _FakeRobosuiteEnv({
            name: _FakeMujocoObject([f"{name}_joint0"]) for name in self.obj_of_interest
        })
        self.env.objects_dict[self.obj_of_interest[0]] = _FakeMujocoObject([self.joint_name])
        self.robots = [_FakeRobot(list(self.gripper_joint_names))]

        self._step_count = 0
        self.set_init_state_calls = 0
        self.seed_value: int | None = None
        self.closed = False

    # ---------------------------------------------------------------- observations

    def _observation(self) -> dict:
        obs = {
            "agentview_image": _fake_image(),
            "robot0_eye_in_hand_image": _fake_image(),
            "robot0_eef_pos": np.zeros(3),
            "robot0_eef_quat": np.array([0.0, 0.0, 0.0, 1.0]),
            "robot0_gripper_qpos": np.array([
                float(self._data.get_joint_qpos(self.gripper_joint_names[0])),
                float(self._data.get_joint_qpos(self.gripper_joint_names[1])),
            ]),
        }
        return obs

    # --------------------------------------------------------------- env lifecycle

    def seed(self, value: int) -> None:
        self.seed_value = value

    def reset(self) -> dict:
        self._step_count = 0
        return self._observation()

    def step(self, action) -> tuple[dict, float, bool, dict]:
        self._step_count += 1
        done = self._step_count >= self.horizon
        return self._observation(), 0.0, done, {}

    def check_success(self) -> bool:
        return any(step <= self._step_count - 1 for step in self.success_at_steps) if self._step_count else False

    def close(self) -> None:
        self.closed = True

    # ----------------------------------------------------------- state get/set/refresh

    def get_sim_state(self):
        return {k: v.copy() for k, v in self._data._qpos.items()}

    def set_init_state(self, state) -> dict:
        self.set_init_state_calls += 1
        if isinstance(state, dict):
            for name, value in state.items():
                self._data.set_joint_qpos(name, value)
        return self._observation()
