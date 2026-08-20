"""Harvest recovery segments by perturbing a demonstration and scripting the correction.

Behaviour cloning on successful-only demonstrations gives a policy no state that maps to
"something went wrong, back off and re-approach": nothing in the data has ever gone
wrong. This produces that missing mode as training data.

The method is demonstration-anchored, which is what makes the labels trustworthy. A
demonstration is replayed from its exact recorded initial state to a step inside its
transport phase; the simulator state is snapshotted; a perturbation is injected; a
scripted controller closes the resulting pose gap; and control is then handed back to
the demonstration's *own remaining recorded actions*. Only the correction is synthetic —
everything after re-convergence is real demonstrator data, and the correct continuation
is known exactly rather than inferred.

A segment is kept only if it re-achieves the subtask predicate the perturbation broke.
Nothing is hand-labelled, and the acceptance rate is itself a measurement: a low one
says these states are not recoverable, which is a finding rather than a failure.

``--perturb none`` is the null harvest, and the first thing to run: it snapshots and
restores without perturbing or correcting, so the demonstration must still succeed. If
it does not, the state round-trip is disturbing the physics and no segment produced here
would mean anything.

Examples::

    # verification: the round-trip must not change the outcome
    python scripts/harvest_recovery.py --episodes 27 --perturb none

    # one episode, one perturbation, before harvesting at scale
    python scripts/harvest_recovery.py --episodes 27 --perturb displace_object

    # the backbone of the mixture
    python scripts/harvest_recovery.py --perturb displace_object --per-task 20
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.env_setup import setup_env  # noqa: E402

setup_env()

import numpy as np  # noqa: E402

from eval.perturbations import PerturbationKind, apply_perturbation  # noqa: E402
from eval.subtasks import SubtaskTracker, subtasks_for  # noqa: E402
from src.utils import DUMMY_ACTION, TaskMapping, load_episode, load_exact_init_state  # noqa: E402

# Proportional gain in the environment's OSC delta-pose space. Calibrated from the
# traces rather than guessed: successful transport realises ~5.3e-3 m/step at a command
# magnitude of ~0.5, so a unit command moves ~0.01 m and a gain of 100 saturates on a
# 1 cm error — fast enough to close a gap in tens of steps without ringing.
POSITION_GAIN = 100.0

# Rotation is commanded in the same axis-angle space, at a lower gain: the rate limit on
# dimensions 3--5 is roughly a tenth of the positional one, so an equal gain would spend
# every step saturated against it.
ROTATION_GAIN = 10.0

# The demonstrations' per-dimension action standard deviations. Scripted commands are
# smoother and larger than teleop, so they are rate-limited to this envelope; without it
# the mixture teaches a second action *style* rather than a recovery *mode*.
DEMO_ACTION_SIGMA = np.array([0.283, 0.359, 0.367, 0.038, 0.054, 0.087, 0.996])

# A correction that has not converged in this many steps is not going to; the segment is
# rejected rather than left to spin.
MAX_CORRECTION_STEPS = 120

# Convergence threshold on the end-effector position gap, in metres. A proportional law
# against a compliant OSC controller leaves a steady-state error — measured at ~0.018 m
# here — and closing it would need integral action that overshoots into contact. Two
# centimetres is inside the gripper's finger span, so it is a tolerance the grasp can
# absorb rather than a number chosen to make the check pass.
POSE_TOLERANCE_M = 0.02

# Below this the object has left the table and no re-approach will reach it.
MIN_OBJECT_Z = 0.30

# How far to lift before re-approaching, in metres. Enough to clear the object rather
# than dragging across it laterally.
RETREAT_HEIGHT_M = 0.06

# Steps held closed after the re-grasp, so the contact settles before the arm is loaded.
GRASP_SETTLE_STEPS = 8

# Demonstration steps that must remain after an anchor. The handback replays them to
# finish the task, so an anchor too near the end leaves nothing to recover into.
# 15 rather than more: a late-grasping task can have only ~50 steps between
# its grasp and the end of the recording, and a larger margin excludes it
# entirely. Anchors that leave too little to finish are caught by the
# acceptance filter rather than by guessing a bound here.
HANDBACK_MARGIN = 15


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--episodes", type=int, nargs="+", default=None,
                   help="dataset episode indices; default: one matched episode per task")
    p.add_argument("--task-indices", type=int, nargs="+", default=None,
                   help="harvest every matched episode of these tasks, up to --max-episodes")
    p.add_argument("--weight-by", type=Path, default=None, metavar="RESULTS.JSONL",
                   help="allocate --total-segments across tasks in proportion to how "
                        "often the baseline fails each one, so recovery data is densest "
                        "where the policy actually breaks")
    p.add_argument("--total-segments", type=int, default=160,
                   help="segments to harvest in total when --weight-by is given")
    p.add_argument("--floor", type=int, default=0,
                   help="minimum quota per task under --weight-by, so a task the "
                        "baseline never fails is not excluded entirely")
    p.add_argument("--max-episodes", type=int, default=12,
                   help="episodes per task under --task-indices; caps how much any one "
                        "task contributes, so the mixture does not specialise to it")
    p.add_argument("--perturb", default="offcourse",
                   choices=["none", "offcourse"] + [k.value for k in PerturbationKind],
                   help="'offcourse' drives the arm away from the demonstrated "
                        "trajectory, which is the failure the policy actually exhibits; "
                        "'none' is the null harvest that verifies the state round-trip")
    p.add_argument("--offcourse-steps", type=int, default=12,
                   help="steps of off-trajectory command before the correction begins")
    p.add_argument("--displacement-radius-m", type=float, default=0.08)
    p.add_argument("--per-task", type=int, default=1,
                   help="anchor points per episode, spread across its transport phase")
    p.add_argument("--out", type=Path, default=Path("outputs/recovery_segments"))
    p.add_argument("--camera-size", type=int, default=256)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--repo-id", default="lerobot/libero_10")
    return p.parse_args(argv)


def error_weighted_quota(results_path: Path, total: int, floor: int) -> dict[int, int]:
    """Segments per task, in proportion to how often the baseline fails that task.

    Recovery data is only useful where the policy breaks. Harvesting uniformly spends
    most of the budget on tasks that already succeed, and harvesting one task spends it
    all on a single failure mode; weighting by the measured error count puts the data
    where the errors are while keeping every failing task represented.
    """
    from eval.logging import read_results

    failures: dict[int, int] = {}
    for result in read_results(results_path):
        failures.setdefault(result.task_dataset_index, 0)
        failures[result.task_dataset_index] += (not result.success)

    total_failures = sum(failures.values())
    if not total_failures:
        raise SystemExit(f"{results_path} records no failures; nothing to weight by")
    quota = {task: max(floor, round(total * count / total_failures))
             for task, count in failures.items()}
    return {task: n for task, n in quota.items() if n > 0}


def transport_window(trace_path: Path) -> tuple[int, int] | None:
    """``(grasp_step, place_step)`` from a demonstration trace.

    The anchor has to sit between them: before the grasp there is nothing to drop, and
    after the placement the episode is over. Tasks differ by more than a factor of two
    in when those happen, so the window is read per episode rather than assumed.
    """
    from eval.trace import load_trace

    trace = load_trace(trace_path)
    done, ids = trace["subtask_done"], trace["subtask_ids"]
    grasps = [i for i, name in enumerate(ids) if name.startswith("grasp:")]
    if not grasps or not done.size:
        return None
    grasped = np.flatnonzero(done[:, grasps].any(axis=1))
    if not grasped.size:
        return None

    # The window runs from the first grasp to shortly before the recording ends, not to
    # the first goal condition. Goal conditions are not reliably *after* the grasp: an
    # articulation goal can fire long before it (a stove switched on during approach) or
    # be true at reset, and a containment goal can fire the step after the object leaves
    # the table. What is reliable is that the object is held from the grasp onward, which
    # is what makes an off-course excursion a recoverable perturbation.
    start = int(grasped[0])
    end = len(done) - HANDBACK_MARGIN
    return (start, end) if end - start > 20 else None


def eef_quat(env) -> np.ndarray:
    """Current end-effector orientation as a quaternion, copied off the live buffer."""
    return np.array(env.env._get_observations()["robot0_eef_quat"],
                    dtype=np.float64, copy=True)


def orientation_error(env, target_quat: np.ndarray) -> np.ndarray:
    """Axis-angle rotation taking the current orientation to `target_quat`.

    The environment's OSC action space expects exactly this in dimensions 3--5, so the
    controller commands the same quantity the demonstrations do rather than leaving
    those dimensions at zero — which would make scripted segments a distinguishable
    action style rather than a recovery mode.
    """
    from robosuite.utils import transform_utils

    delta = transform_utils.quat_multiply(
        target_quat, transform_utils.quat_inverse(eef_quat(env)))
    return np.asarray(transform_utils.quat2axisangle(delta), dtype=np.float64)


def eef_pose(env) -> np.ndarray:
    """Current end-effector position, in world coordinates.

    Copied deliberately. MuJoCo's `site_xpos` is a live view into the simulator's own
    buffer, so a reference to it silently tracks the arm — and a "snapshot" taken that
    way equals the current pose at every later comparison, making every convergence
    check trivially true.
    """
    return np.array(env.env.sim.data.site_xpos[
        env.env.sim.model.site_name2id("gripper0_grip_site")], dtype=np.float64, copy=True)


def object_position(env) -> np.ndarray:
    """World position of the task's primary object — the one perturbations move."""
    domain = env.env
    name = domain.obj_of_interest[0]
    # Copied for the same reason as `eef_pose`: `body_xpos` is a live view.
    return np.array(domain.sim.data.body_xpos[domain.obj_body_id[name]],
                    dtype=np.float64, copy=True)


def held_object_z(env) -> float:
    return float(object_position(env)[2])


def rate_limited(action: np.ndarray) -> np.ndarray:
    """Clip a scripted command into the demonstrations' per-dimension envelope."""
    return np.clip(action, -DEMO_ACTION_SIGMA * 3, DEMO_ACTION_SIGMA * 3)


def drive_to(env, target: np.ndarray, gripper: float, recorder,
             target_quat: np.ndarray | None = None) -> tuple[bool, float]:
    """P-control the end-effector to `target`. Returns whether it converged.

    Operates in the environment's own OSC delta-pose space, so the commands recorded
    here are in the same units and convention as the demonstrations' — which is what
    makes the segment mixable with them at all.
    """
    for _ in range(MAX_CORRECTION_STEPS):
        error = target - eef_pose(env)
        distance = float(np.linalg.norm(error))
        if distance < POSE_TOLERANCE_M:
            return True, distance
        action = np.zeros(7)
        action[:3] = np.clip(POSITION_GAIN * error, -1.0, 1.0)
        if target_quat is not None:
            action[3:6] = np.clip(ROTATION_GAIN * orientation_error(env, target_quat),
                                  -1.0, 1.0)
        action[6] = gripper
        action = rate_limited(action)
        observation, _reward, _done, _info = env.step(action)
        recorder(observation, action)
    return False, float(np.linalg.norm(target - eef_pose(env)))


class SegmentRecorder:
    """Accumulates the corrective steps in the form the training loader consumes.

    The main camera is kept for every step so the System 2 frame can be emitted at the
    same staleness the dataloader provides during training; a segment that recorded only
    the current frame could not reproduce the input contract.
    """

    def __init__(self, instruction: str) -> None:
        self.instruction = instruction
        self.images: list[np.ndarray] = []
        self.wrist: list[np.ndarray] = []
        self.states: list[np.ndarray] = []
        self.actions: list[np.ndarray] = []

    def __call__(self, observation: dict, action: np.ndarray) -> None:
        from src.observations import from_env

        canonical = from_env(observation)
        self.images.append((canonical.images["image"][0].numpy() * 255).astype(np.uint8))
        self.wrist.append((canonical.images["image2"][0].numpy() * 255).astype(np.uint8))
        self.states.append(canonical.state[0].numpy().astype(np.float32))
        self.actions.append(np.asarray(action, dtype=np.float32))

    def __len__(self) -> int:
        return len(self.actions)

    def arrays(self, metadata: dict) -> dict:
        return {
            "image": np.stack(self.images),          # (T, 3, H, W) uint8
            "wrist_image": np.stack(self.wrist),
            "state": np.stack(self.states),
            "action": np.stack(self.actions),
            "task": np.asarray(self.instruction),
            **{k: np.asarray(v) for k, v in metadata.items()},
        }


def harvest_one(env, task, episode: int, actions, anchor: int, args,
                rng: np.random.Generator) -> tuple[bool, SegmentRecorder | None, str]:
    """One anchor point. Returns ``(accepted, segment, reason)``."""
    init_state = load_exact_init_state(episode)
    env.reset()
    env.set_init_state(init_state)
    for _ in range(10):
        env.step(DUMMY_ACTION)

    subtasks = subtasks_for(task.bddl)
    for step in range(anchor):
        env.step(np.asarray(actions[step], dtype=np.float64))

    # The pose the demonstration was at when it was interrupted: the correction's target,
    # and the state from which its remaining actions are valid again.
    resume_pose = eef_pose(env)
    resume_quat = eef_quat(env)
    kind = (PerturbationKind.NONE if args.perturb == "offcourse"
            else PerturbationKind(args.perturb))

    # How the gripper was holding the object when it was interrupted. The object moves,
    # so re-approaching means reconstructing this relationship at its *new* pose rather
    # than returning to the old location.
    grasp_offset = resume_pose - object_position(env)
    recorder = SegmentRecorder(task.instruction)

    if args.perturb == "offcourse":
        # Drive the arm away from the demonstrated trajectory while it keeps hold of the
        # object. The object stays where the demonstration expects it, so the remaining
        # recorded actions stay valid and the correction is exactly a pose gap — the
        # same gap the policy faces when it drifts or jams mid-transport, which is the
        # failure mode the census finds rather than one invented for the harvest.
        direction = rng.normal(size=3)
        direction /= np.linalg.norm(direction)
        for _ in range(args.offcourse_steps):
            action = np.zeros(7)
            action[:3] = rate_limited(np.concatenate([direction * 0.6, np.zeros(4)]))[:3]
            action[6] = 1.0            # keep holding
            observation, _r, _d, _i = env.step(action)
            recorder(observation, action)
        excursion = float(np.linalg.norm(eef_pose(env) - resume_pose))
        ok, gap = drive_to(env, resume_pose, gripper=1.0, recorder=recorder,
                           target_quat=resume_quat)
        if not ok:
            return False, recorder, f"return stalled {gap:.3f} m short"
        if excursion < POSE_TOLERANCE_M:
            return False, recorder, f"excursion only {excursion:.3f} m — no gap to correct"
    elif kind is not PerturbationKind.NONE:
        apply_perturbation(env, kind, rng, None, args.displacement_radius_m)
        if held_object_z(env) < MIN_OBJECT_Z:
            return False, None, "object left the workspace"

        # Retreat: open and lift clear, so the re-approach is not a lateral drag through
        # whatever the perturbation just did to the contact state.
        retreat = eef_pose(env) + np.array([0.0, 0.0, RETREAT_HEIGHT_M])
        ok, gap = drive_to(env, retreat, gripper=-1.0, recorder=recorder)
        if not ok:
            return False, recorder, f"retreat stalled {gap:.3f} m short"

        # Re-approach the object where it now is, holding the demonstrated grasp offset.
        target = object_position(env) + grasp_offset
        ok, gap = drive_to(env, target, gripper=-1.0, recorder=recorder)
        if not ok:
            return False, recorder, (f"re-approach stalled {gap:.3f} m short of "
                                     f"{np.round(target, 3)}")

        # Close, and let the contact settle before loading the arm.
        for _ in range(GRASP_SETTLE_STEPS):
            action = np.zeros(7)
            action[6] = 1.0
            observation, _r, _d, _i = env.step(action)
            recorder(observation, action)

        # Return to where the demonstration was interrupted; its remaining actions are
        # only valid from that pose.
        ok, gap = drive_to(env, resume_pose, gripper=1.0, recorder=recorder)
        if not ok:
            return False, recorder, f"return stalled {gap:.3f} m short"

    # Hand control back to the demonstration. Everything from here is recorded human
    # data, and the predicate it re-achieves is what accepts the segment.
    tracker = SubtaskTracker(subtasks, env)
    for step in range(anchor, len(actions)):
        env.step(np.asarray(actions[step], dtype=np.float64))
        tracker.update(step)
        if env.check_success():
            return True, recorder, "recovered"
    achieved = tracker.n_achieved
    return False, recorder, f"predicate not re-achieved ({achieved}/{tracker.n_total})"


def main(argv: list[str] | None = None) -> int:
    from libero.libero.envs import OffScreenRenderEnv

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    args = parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    dataset = LeRobotDataset(args.repo_id)
    mapping = TaskMapping.from_dataset(dataset)
    from src.utils import episodes_for_task

    quota: dict[int, int] | None = None
    if args.weight_by:
        quota = error_weighted_quota(args.weight_by, args.total_segments, args.floor)
        print("error-weighted allocation:")
        for task in sorted(quota):
            print(f"  task {task:02d}: {quota[task]:>3d} segments")
        print()

    episodes = args.episodes or []
    if not episodes and (args.task_indices or quota):
        for index in sorted(args.task_indices or quota):
            matched = [e for e in episodes_for_task(dataset, index)
                       if load_exact_init_state(e) is not None]
            episodes.extend(matched[: args.max_episodes])
    if not episodes:
        for index in mapping.dataset_indices():
            match = next((e for e in episodes_for_task(dataset, index)
                          if load_exact_init_state(e) is not None), None)
            if match is not None:
                episodes.append(match)

    accepted = attempted = written = 0
    per_task_written: dict[int, int] = {}
    for episode in episodes:
        task = mapping.by_episode(dataset, episode)
        if quota is not None:
            filled = per_task_written.get(task.dataset_index, 0)
            if filled >= quota.get(task.dataset_index, 0):
                continue
        instruction, actions, _state = load_episode(dataset, episode)
        traces = sorted(Path("outputs/demo_traces").glob(
            f"task{task.dataset_index:02d}_demo_ep*.npz"))
        window = transport_window(traces[0]) if traces else None
        if window is None:
            print(f"  ep{episode} (task {task.dataset_index:02d}): no transport window")
            continue

        # The window comes from one representative trace, but every episode of a task
        # has its own length; an anchor past this episode's actions would index off the
        # end of the handback.
        start, end = window
        end = min(end, len(actions) - HANDBACK_MARGIN)
        if end - start <= 10:
            print(f"  ep{episode} (task {task.dataset_index:02d}): "
                  f"episode too short for an anchor")
            continue
        anchors = np.unique(np.linspace(start + 5, end - 5, args.per_task).astype(int))
        env = OffScreenRenderEnv(bddl_file_name=task.bddl, camera_heights=args.camera_size,
                                 camera_widths=args.camera_size, horizon=len(actions) + 400)
        try:
            env.seed(0)
            for anchor in anchors:
                if quota is not None and per_task_written.get(task.dataset_index, 0) >= \
                        quota.get(task.dataset_index, 0):
                    break
                attempted += 1
                ok, segment, reason = harvest_one(env, task, episode, actions,
                                                  int(anchor), args, rng)
                marker = "accept" if ok else "reject"
                print(f"  ep{episode} (task {task.dataset_index:02d}) anchor {anchor:>3d}: "
                      f"{marker} — {reason}"
                      + (f", {len(segment)} corrective steps" if segment else ""))
                accepted += ok
                if ok and segment is not None and len(segment):
                    written += 1
                    per_task_written[task.dataset_index] = \
                        per_task_written.get(task.dataset_index, 0) + 1
                    path = args.out / f"task{task.dataset_index:02d}_ep{episode}_a{anchor}.npz"
                    np.savez_compressed(path, **segment.arrays({
                        "source": "demonstration_anchored",
                        "task_dataset_index": task.dataset_index,
                        "episode": episode, "anchor": int(anchor),
                        "perturbation": args.perturb,
                    }))
        finally:
            env.close()

    rate = accepted / attempted if attempted else 0.0
    # Acceptance and segments written differ under `--perturb none`: the null harvest
    # verifies the round-trip and correctly produces no corrective steps to keep.
    print(f"\naccepted {accepted}/{attempted} ({rate:.0%}); "
          f"wrote {written} segment(s) to {args.out}")
    if per_task_written:
        print("per task: " + "  ".join(f"t{task:02d}:{n}"
                                       for task, n in sorted(per_task_written.items())))
        if quota is not None:
            short = {t: quota[t] - per_task_written.get(t, 0) for t in quota
                     if per_task_written.get(t, 0) < quota[t]}
            if short:
                print("short of quota (raise --max-episodes or --per-task): "
                      + "  ".join(f"t{t:02d}:-{n}" for t, n in sorted(short.items())))
    return 0


if __name__ == "__main__":
    sys.exit(main())
