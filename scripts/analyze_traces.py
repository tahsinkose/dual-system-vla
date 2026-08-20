"""Classify why traced rollouts failed, against a replayed demonstration.

Reads the `.npz` traces written by `eval/trace.py` and answers the question a success
rate cannot: given that an episode reached subtask N and stopped, *how* did it stop.

Episodes are aligned on the first grasp rather than on step index, because a rollout
and a demonstration of the same task differ in length and in how long the approach
takes. Everything downstream is measured from that event.

Post-grasp faults, distinguished by the goal-distance curve and arm motion:

**stalled**          the end-effector stops moving; the arm is frozen.
**oscillating**      goal distance stops improving but the arm keeps moving; it circles.
**receding**         goal distance grows; the arm carries the object away.
**short-of-target**  the object approaches but never gets as close as a demonstration
                     leaves it, so the goal predicate cannot fire.
**at-target-unmet**  the object reaches demonstration distance and the predicate still
                     does not hold — the goal is not about proximity alone.

Gripper release is deliberately *not* a fault. Demonstrations of these tasks hold the
object to the end: LIBERO stops the recording when the goal predicate fires, which for
a containment goal happens while the gripper is still closed. A rollout that never
opens is doing what it was trained on.

Examples::

    python scripts/analyze_traces.py --traces outputs/eval/traces --demo outputs/demo_traces
    python scripts/analyze_traces.py --traces outputs/eval/traces/task09_init000.npz
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from eval.trace import load_trace  # noqa: E402

# The gripper command is the last action dimension, a bang-bang +/-1 signal in the
# demonstrations: positive closes, negative opens.
GRIPPER_DIM = 6

# Below this the end-effector is not meaningfully moving, in metres per step.
STALL_SPEED_M = 2e-4
# Goal distance must improve by at least this much to count as progress, in metres.
PROGRESS_M = 5e-3
# How much further than a demonstration's final distance still counts as "arrived".
# Demonstrations end around 0.06 m on task 9, so this is a generous band.
TARGET_TOLERANCE_M = 0.03


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--traces", type=Path, required=True,
                   help="a trace .npz, or a directory of them")
    p.add_argument("--demo", type=Path, default=None,
                   help="demonstration trace(s) for the same task, as the reference")
    p.add_argument("--window", type=int, default=50,
                   help="steps at the end of an episode used to judge its final state")
    return p.parse_args(argv)


def trace_paths(target: Path) -> list[Path]:
    if target.is_dir():
        return sorted(target.glob("*.npz"))
    return [target]


def grasp_step(trace: dict) -> int | None:
    """First step any grasp subtask reads true — the alignment anchor."""
    grasp_columns = [i for i, name in enumerate(trace["subtask_ids"]) if name.startswith("grasp:")]
    if not grasp_columns or trace["subtask_done"].size == 0:
        return None
    done = trace["subtask_done"][:, grasp_columns].any(axis=1)
    hits = np.flatnonzero(done)
    return int(hits[0]) if hits.size else None


def eef_speed(trace: dict) -> np.ndarray:
    """Per-step end-effector displacement, in metres."""
    positions = trace["state"][:, :3]
    if len(positions) < 2:
        return np.zeros(len(positions), dtype=np.float32)
    return np.concatenate([[0.0], np.linalg.norm(np.diff(positions, axis=0), axis=1)])


def active_goal(trace: dict, start: int) -> int | None:
    """Index of the goal condition the episode was working on after `start`.

    The one whose distance is smallest at the end: with several objects to relocate,
    that is the one currently in hand or just placed.
    """
    distances = trace["goal_distance"]
    if distances.size == 0 or start >= len(distances):
        return None
    return int(np.argmin(distances[-1]))


def classify(trace: dict, window: int, reference_distance: float | None = None) -> tuple[str, dict]:
    """Return ``(verdict, evidence)`` for one traced episode.

    `reference_distance` is what a demonstration leaves the object at; without it the
    proximity verdicts are skipped, since there is nothing to be short of.
    """
    done = trace["subtask_done"]
    # Success is the task's goal predicate: every *goal condition* true at once. Grasp
    # subtasks are excluded because they are transient by construction — the gripper
    # opens to place the object — so requiring them would report a solved episode as a
    # failure, and requiring the final step alone would miss a goal that is met and then
    # disturbed.
    goals = [i for i, name in enumerate(trace["subtask_ids"])
             if not name.startswith("grasp:")]
    if done.size and goals and bool(done[:, goals].all(axis=1).any()):
        return "solved", {}

    grasp = grasp_step(trace)
    if grasp is None:
        speed = eef_speed(trace)
        moved = float(np.linalg.norm(trace["state"][-1, :3] - trace["state"][0, :3])) if len(trace["state"]) else 0.0
        verdict = "never-approached" if speed.mean() < STALL_SPEED_M else "never-grasped"
        return verdict, {"mean_speed_m": float(speed.mean()), "net_eef_travel_m": moved}

    goal = active_goal(trace, grasp)
    tail = slice(max(grasp, len(trace["step"]) - window), len(trace["step"]))
    speed = eef_speed(trace)[tail]
    gripper = trace["action"][grasp:, GRIPPER_DIM]
    evidence = {
        "grasp_step": grasp,
        "mean_speed_m": float(speed.mean()) if speed.size else 0.0,
        # Recorded because it is informative, not because it is a fault: see the
        # module docstring on why demonstrations never release either.
        "gripper_opened_after_grasp": bool((gripper < 0).any()),
    }
    if goal is None:
        return "no-goal-distance", evidence

    distance = trace["goal_distance"][:, goal]
    evidence["goal"] = trace["goal_ids"][goal]
    evidence["distance_at_grasp_m"] = float(distance[grasp])
    evidence["final_distance_m"] = float(distance[-1])
    evidence["best_distance_m"] = float(distance[grasp:].min())

    best = float(distance[grasp:].min())
    if distance[-1] - distance[grasp] > PROGRESS_M:
        return "receding", evidence
    if speed.size and speed.mean() < STALL_SPEED_M:
        return "stalled", evidence
    if distance[grasp] - best < PROGRESS_M:
        return "oscillating", evidence
    if reference_distance is None:
        return "progressed-but-unmet", evidence
    if best > reference_distance + TARGET_TOLERANCE_M:
        return "short-of-target", evidence
    return "at-target-unmet", evidence


def demo_reference(paths: list[Path]) -> dict | None:
    """Aggregate the demonstrations into the numbers a rollout is judged against."""
    if not paths:
        return None
    finals, grasps, lengths = [], [], []
    for path in paths:
        trace = load_trace(path)
        goal = active_goal(trace, 0)
        if goal is not None:
            finals.append(float(trace["goal_distance"][-1, goal]))
        grasp = grasp_step(trace)
        if grasp is not None:
            grasps.append(grasp)
        lengths.append(len(trace["step"]))
    return {
        "n": len(paths),
        "final_distance_m": float(np.mean(finals)) if finals else float("nan"),
        "grasp_step": float(np.mean(grasps)) if grasps else float("nan"),
        "length": float(np.mean(lengths)),
    }


def task_of(path: Path) -> str:
    """The `taskNN` prefix a trace filename carries, or `""` if it has none.

    Rollouts and demonstrations are matched on it because the target distance a
    demonstration leaves an object at is task-specific: judging a basket task against a
    caddy task's reference would classify correct behaviour as short-of-target.
    """
    stem = path.stem
    return stem[:6] if stem.startswith("task") and stem[4:6].isdigit() else ""


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    rollouts = trace_paths(args.traces)
    if not rollouts:
        raise SystemExit(f"no traces found at {args.traces}")

    demos: dict[str, list[Path]] = {}
    for path in (trace_paths(args.demo) if args.demo else []):
        demos.setdefault(task_of(path), []).append(path)
    references = {task: demo_reference(paths) for task, paths in demos.items()}
    missing = sorted({task_of(p) for p in rollouts} - references.keys())
    if missing:
        print(f"no demonstration reference for {', '.join(missing) or 'untagged traces'}"
              " — proximity verdicts skipped there\n", file=sys.stderr)

    verdicts: dict[str, int] = {}
    per_task: dict[str, dict[str, int]] = {}
    for path in rollouts:
        task = task_of(path)
        reference = references.get(task)
        verdict, evidence = classify(load_trace(path), args.window,
                                     reference["final_distance_m"] if reference else None)
        verdicts[verdict] = verdicts.get(verdict, 0) + 1
        per_task.setdefault(task, {})
        per_task[task][verdict] = per_task[task].get(verdict, 0) + 1
        detail = "  ".join(f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}"
                           for k, v in evidence.items())
        print(f"{path.stem:<24} {verdict:<20} {detail}")

    print("\nverdicts:")
    for verdict, count in sorted(verdicts.items(), key=lambda kv: -kv[1]):
        print(f"  {verdict:<20} {count:>3d}/{len(rollouts)}")

    if len(per_task) > 1:
        print("\nper task:")
        for task in sorted(per_task):
            counts = per_task[task]
            total = sum(counts.values())
            summary = "  ".join(f"{v}:{c}" for v, c in
                                sorted(counts.items(), key=lambda kv: -kv[1]))
            reference = references.get(task)
            anchor = (f"  [demo ends at {reference['final_distance_m']:.3f} m]"
                      if reference else "  [no reference]")
            print(f"  {task or '(untagged)':<10} n={total:<4} {summary}{anchor}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
