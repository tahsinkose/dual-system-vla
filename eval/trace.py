"""Per-step rollout traces, in a schema shared with demonstration replays.

A success rate says an episode failed; the subtask records say which step it stalled
at. Neither says *how*. "Grasped the object and held it for 800 steps" covers a policy
frozen in place, one circling the target, one drifting away, and one that simply never
commands the gripper open — four different faults with four different fixes, and video
can only be watched, not measured.

This records the quantities that separate them, every step, for both a policy rollout
and a replayed demonstration of the same task. The demonstration is the reference: it
is what a solved episode looks like, so a rollout's departure from it is the diagnosis.

The arrays are small — a few hundred steps of low-dimensional state — so one `.npz`
per episode is written whole at the end, matching how `write_latent_trace` already
handles the latent sidecar. Images are deliberately absent; the video writer covers
the qualitative channel.

Schema (all arrays share the leading step axis T unless noted)::

    step               (T,)      int32    environment step index
    state              (T, 8)    float32  eef xyz, axis-angle, two gripper joints
    action             (T, 7)    float32  what was executed, after clipping
    action_raw         (T, 7)    float32  what the policy asked for, before clipping
    subtask_done       (T, S)    bool     each subtask's is_done() at this step
    subtask_ids        (S,)      str      column labels for subtask_done
    object_pos         (T, O, 3) float32  obj_of_interest world positions
    object_names       (O,)      str      row labels for object_pos
    goal_distance      (T, G)    float32  object-to-target distance per goal condition
    goal_ids           (G,)      str      column labels for goal_distance
    latent             (T, D)    float16  System 2's latent, when recorded
    steps_since_update (T,)      int16    latent cadence position, when recorded
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from eval.subtasks import GoalCondition, Subtask, _domain

# Goal predicates that relate an object to a place, and so admit a distance. The unary
# ones (`turnon`, `close`) describe articulation state, which `subtask_done` covers.
SPATIAL_PREDICATES = ("in", "on")


def _position_of(env, name: str) -> np.ndarray:
    """World position of a named object, fixture, or region site.

    A goal literal's target may be any of the three — `basket_1_contain_region` is a
    site, `plate_1` is an object — so both lookups are tried.
    """
    domain = _domain(env)
    if name in getattr(domain, "object_sites_dict", {}):
        return np.asarray(domain.sim.data.get_site_xpos(name), dtype=np.float32)
    if name in getattr(domain, "obj_body_id", {}):
        return np.asarray(domain.sim.data.body_xpos[domain.obj_body_id[name]], dtype=np.float32)
    raise KeyError(f"{name!r} is neither a region site nor a body in this environment")


def spatial_goals(subtasks: tuple[Subtask, ...]) -> list[tuple[str, str, str]]:
    """``(label, moved_object, target)`` for every goal condition with a distance."""
    goals = []
    for subtask in subtasks:
        if isinstance(subtask, GoalCondition) and subtask.literal[0] in SPATIAL_PREDICATES:
            _predicate, moved, target = subtask.literal
            goals.append((subtask.id, moved, target))
    return goals


class RolloutTracer:
    """Accumulates one episode's per-step record.

    Reads simulator state directly rather than the observation dict, so a replayed
    demonstration and a policy rollout produce identical columns even though only one
    of them has a policy.
    """

    def __init__(self, env, subtasks: tuple[Subtask, ...]) -> None:
        self._env = env
        self._subtasks = subtasks
        self._goals = spatial_goals(subtasks)
        self._objects = list(_domain(env).obj_of_interest)

        self._step: list[int] = []
        self._state: list[np.ndarray] = []
        self._action: list[np.ndarray] = []
        self._action_raw: list[np.ndarray] = []
        self._done: list[list[bool]] = []
        self._object_pos: list[np.ndarray] = []
        self._goal_distance: list[list[float]] = []
        self._latent: list[np.ndarray] = []
        self._since_update: list[int] = []

    def record(self, step: int, observation: dict, action: np.ndarray,
               action_raw: np.ndarray | None = None,
               latent: np.ndarray | None = None,
               steps_since_update: int | None = None) -> None:
        """One step, called after `env.step` so the state is the action's result."""
        from src.observations import from_env

        self._step.append(step)
        self._state.append(from_env(observation).state[0].cpu().numpy().astype(np.float32))
        action = np.asarray(action, dtype=np.float32)
        self._action.append(action)
        self._action_raw.append(action if action_raw is None
                                else np.asarray(action_raw, dtype=np.float32))
        self._done.append([bool(s.is_done(self._env)) for s in self._subtasks])
        self._object_pos.append(np.stack([_position_of(self._env, name)
                                          for name in self._objects])
                                if self._objects else np.zeros((0, 3), np.float32))
        self._goal_distance.append([
            float(np.linalg.norm(_position_of(self._env, moved) - _position_of(self._env, target)))
            for _label, moved, target in self._goals
        ])
        if latent is not None:
            self._latent.append(np.asarray(latent, dtype=np.float32))
            self._since_update.append(int(steps_since_update or 0))

    def arrays(self) -> dict[str, np.ndarray]:
        payload = {
            "step": np.asarray(self._step, dtype=np.int32),
            "state": np.stack(self._state) if self._state else np.zeros((0, 8), np.float32),
            "action": np.stack(self._action) if self._action else np.zeros((0, 7), np.float32),
            "action_raw": (np.stack(self._action_raw) if self._action_raw
                           else np.zeros((0, 7), np.float32)),
            "subtask_done": np.asarray(self._done, dtype=bool).reshape(len(self._step), -1),
            "subtask_ids": np.asarray([s.id for s in self._subtasks], dtype=object),
            "object_pos": (np.stack(self._object_pos) if self._object_pos
                           else np.zeros((0, 0, 3), np.float32)),
            "object_names": np.asarray(self._objects, dtype=object),
            "goal_distance": np.asarray(self._goal_distance, dtype=np.float32).reshape(
                len(self._step), -1),
            "goal_ids": np.asarray([label for label, _m, _t in self._goals], dtype=object),
        }
        if self._latent:
            payload["latent"] = np.stack(self._latent).astype(np.float16)
            payload["steps_since_update"] = np.asarray(self._since_update, dtype=np.int16)
        return payload

    def write(self, path: Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, **self.arrays())
        return path


def load_trace(path: Path) -> dict[str, np.ndarray]:
    """Read one trace back, with the object-dtype label arrays restored to lists of str."""
    with np.load(Path(path), allow_pickle=True) as archive:
        trace = {key: archive[key] for key in archive.files}
    for key in ("subtask_ids", "object_names", "goal_ids"):
        if key in trace:
            trace[key] = [str(v) for v in trace[key]]
    return trace
