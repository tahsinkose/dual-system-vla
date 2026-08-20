"""Compare a solved and a failed rollout of the same task, from their `.npz` traces.

Two layouts, because they answer different questions.

``--layout report`` is two panels sized for a paper column. Panel (a) plots realised
end-effector displacement against commanded displacement, both accumulated from the
shared first-grasp step: slope is the fraction of what the policy asks for that the arm
achieves, so a trajectory that flattens is one whose commands have stopped translating
into motion. Panel (b) is System 2's latent as consecutive-step cosine distance, which
says whether the high-level intent was revised while execution went nowhere.

``--layout full`` is the six-panel diagnostic: goal distance, object height against the
target's own reference height, realised speed, commanded magnitude, gripper command, and
the realised-vs-commanded panel. Subtask transitions are marked on every time axis, so a
release or a re-grasp is visible against whichever curve went flat.

Both layouts are drawn at the size they will be *placed* at, never scaled afterwards: a
figure authored large and shrunk to fit takes its text down with it, and 6 pt set in a
15-inch canvas lands at under 3 pt across a page column.

A policy that stops asking and a policy whose asking stops working score identically in a
success rate and look opposite here.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SUCCESS_COLOUR = "tab:green"
FAILURE_COLOUR = "tab:red"


def load(path: Path) -> dict:
    return dict(np.load(path, allow_pickle=True))


def grasp_step(trace: dict) -> int:
    columns = [i for i, n in enumerate(trace["subtask_ids"]) if n.startswith("grasp:")]
    hits = np.flatnonzero(trace["subtask_done"][:, columns].any(axis=1))
    return int(hits[0]) if hits.size else 0


def subtask_events(trace: dict) -> list[tuple[int, str]]:
    """First step each subtask reads true, with a short label."""
    events = []
    for column, name in enumerate(trace["subtask_ids"]):
        hits = np.flatnonzero(trace["subtask_done"][:, column])
        if hits.size:
            events.append((int(hits[0]), str(name).split(":")[0]))
    return sorted(events)


def cumulative(trace: dict, start: int) -> tuple[np.ndarray, np.ndarray]:
    eef = trace["state"][start:, :3]
    realised = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(eef, axis=0), axis=1))])
    commanded = np.cumsum(np.linalg.norm(trace["action"][start:, :3], axis=1))
    return realised, commanded


def eef_speed(trace: dict) -> np.ndarray:
    eef = trace["state"][:, :3]
    return np.concatenate([[0.0], np.linalg.norm(np.diff(eef, axis=0), axis=1)])


def latent_step_distance(trace: dict, smooth: int) -> np.ndarray:
    latent = trace["latent"].astype(np.float32)
    unit = latent / (np.linalg.norm(latent, axis=1, keepdims=True) + 1e-8)
    distance = 1.0 - (unit[1:] * unit[:-1]).sum(1)
    if smooth > 1:
        distance = np.convolve(distance, np.ones(smooth) / smooth, mode="valid")
    return distance


def mark_subtasks(ax, trace: dict, colour: str, annotate: bool = False) -> None:
    """Vertical rule at each subtask's first-achieved step.

    Labels go on one panel only: repeated on all six they overlap into noise, while the
    rules themselves line up across panels and carry the timing on their own.
    """
    for index, (step, label) in enumerate(subtask_events(trace)):
        ax.axvline(step, color=colour, ls=":", lw=0.7, alpha=0.6)
        if annotate:
            ax.annotate(label, (step, 0.97 - 0.09 * index),
                        xycoords=("data", "axes fraction"), fontsize=6, color=colour,
                        va="top", ha="left", alpha=0.9,
                        xytext=(2, 0), textcoords="offset points")


def plot_realised_vs_commanded(ax, traces) -> None:
    for label, trace, colour in traces:
        realised, commanded = cumulative(trace, grasp_step(trace))
        ax.plot(commanded, realised, colour, lw=1.4, label=label)
        ax.plot(commanded[-1], realised[-1], colour, marker="o", ms=3)
    ax.set_title("realised vs commanded motion, from first grasp", fontsize=8)
    ax.set_xlabel(r"cumulative commanded $|\Delta xyz|$ (action units)", fontsize=7)
    ax.set_ylabel("cumulative realised eef travel (m)", fontsize=7)


def plot_latent(ax, traces, smooth: int) -> None:
    for label, trace, colour in traces:
        ax.plot(latent_step_distance(trace, smooth), colour, lw=1.0, label=label)
    ax.set_title("System 2 latent: consecutive-step cosine distance", fontsize=8)
    ax.set_xlabel("step", fontsize=7)
    ax.set_ylabel(r"$1 - \cos$", fontsize=7)


def report_figure(traces, smooth: int):
    fig, (left, right) = plt.subplots(1, 2, figsize=(6.5, 2.4))
    plot_realised_vs_commanded(left, traces)
    plot_latent(right, traces, smooth)
    return fig, (left, right)


def full_figure(traces, smooth: int, tall: bool = False):
    """Six panels, either 2x3 spanning a page or 3x2 inside one column.

    The tall arrangement costs about half the page area of the wide one and takes a
    single-column float, at the price of ~41 mm panels instead of ~55 mm.
    """
    figsize, shape = ((3.3, 4.9), (3, 2)) if tall else ((6.9, 3.9), (2, 3))
    fig, axes = plt.subplots(*shape, figsize=figsize)
    goal, height, speed, command, gripper, dissociation = axes.ravel()

    for label, trace, colour in traces:
        step = trace["step"]
        target_z = float(trace["object_pos"][0, 1, 2])
        goal.plot(step, trace["goal_distance"][:, 0], colour, label=label)
        height.plot(step, trace["object_pos"][:, 0, 2], colour, label=label)
        speed.plot(step, eef_speed(trace), colour, lw=0.7, label=label)
        command.plot(step, np.linalg.norm(trace["action"][:, :3], axis=1), colour,
                     lw=0.7, label=label)
        gripper.plot(step, trace["action"][:, 6], colour, lw=0.8, label=label)
        for ax in (goal, height, speed, command, gripper):
            mark_subtasks(ax, trace, colour, annotate=ax is goal)

    height.axhline(target_z, ls="--", c="k", lw=0.8)
    height.annotate("target reference height", (0.02, target_z), xycoords=("axes fraction", "data"),
                    fontsize=6, va="bottom")
    plot_realised_vs_commanded(dissociation, traces)
    speed.set_yscale("log")

    titles = [
        (goal, "goal distance to target region", "m"),
        (height, "object height", "m"),
        (speed, r"realised eef speed $|\Delta eef|$", "m/step"),
        (command, r"commanded $|\Delta xyz|$", "action units"),
        (gripper, "gripper command (+1 close)", "cmd"),
    ]
    for ax, title, ylabel in titles:
        ax.set_title(title, fontsize=7)
        ax.set_ylabel(ylabel, fontsize=6)
        ax.set_xlabel("step", fontsize=6)
    for ax in axes.ravel()[:-shape[1]]:
        ax.set_xlabel("")
    if tall:
        # 41 mm of width will not carry five tick labels legibly.
        for ax in (goal, height, speed, command, gripper):
            ax.set_xticks([0, 400, 800])
    dissociation.set_title("realised vs commanded" if tall
                           else "realised vs commanded motion", fontsize=7)
    dissociation.set_xlabel(r"cumulative commanded" if tall
                            else r"cumulative commanded $|\Delta xyz|$", fontsize=6)
    dissociation.set_ylabel("realised eef travel (m)", fontsize=6)
    return fig, axes.ravel(), goal


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--success", type=Path, required=True)
    p.add_argument("--failure", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--layout", choices=("report", "full"), default="report")
    p.add_argument("--tall", action="store_true",
                   help="full layout as 3x2 inside one column, rather than 2x3 spanning")
    p.add_argument("--title", default=None, help="suptitle, full layout only")
    p.add_argument("--smooth", type=int, default=15,
                   help="moving average over the latent cosine distance")
    args = p.parse_args()

    traces = [("success", load(args.success), SUCCESS_COLOUR),
              ("failure", load(args.failure), FAILURE_COLOUR)]

    legend_only = None
    if args.layout == "report":
        fig, axes = report_figure(traces, args.smooth)
    else:
        fig, axes, legend_only = full_figure(traces, args.smooth, tall=args.tall)
        if args.title:
            fig.suptitle(args.title, fontsize=8)

    for ax in axes:
        ax.tick_params(labelsize=5 if legend_only is not None else 6)
        ax.grid(alpha=0.3, lw=0.4)
        if legend_only is None or ax is legend_only:
            ax.legend(fontsize=5 if legend_only is not None else 6, frameon=False)

    fig.tight_layout(pad=0.5, h_pad=1.0, w_pad=1.0)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=200 if args.layout == "report" else 110)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
