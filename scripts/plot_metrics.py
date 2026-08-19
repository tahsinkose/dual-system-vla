"""Plot training curves from one or more `metrics.jsonl` files.

Overlaying runs is the primary use: the ablation compares CKPT-DUAL against
CKPT-STATIC, and their loss curves belong side by side in the report rather than in
two separate images at different scales.

Examples::

    # both training runs on one axis
    python scripts/plot_metrics.py outputs/train/live outputs/train/static

    # a single run, written next to its metrics file
    python scripts/plot_metrics.py outputs/gate/live

    # explicit destination, log-scale loss
    python scripts/plot_metrics.py outputs/train/* --output report/loss.png --log
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("runs", nargs="+", type=Path,
                   help="run directories, or metrics.jsonl files directly")
    p.add_argument("--output", type=Path, default=None,
                   help="where to write the figure (default: alongside the first run)")
    p.add_argument("--log", action="store_true", help="log-scale the loss axis")
    p.add_argument("--smooth", type=int, default=1,
                   help="moving-average window over logged points (default: 1, none)")
    p.add_argument("--metric", default="loss", choices=["loss", "it_per_s"],
                   help="which series to plot (default: loss)")
    return p.parse_args()


def resolve(path: Path) -> Path:
    """Accept either a run directory or the metrics file itself."""
    candidate = path / "metrics.jsonl" if path.is_dir() else path
    if not candidate.is_file():
        raise SystemExit(f"no metrics file at {candidate}")
    return candidate


def read_run(path: Path) -> tuple[dict, list[dict], list[dict]]:
    """Return the run header, its step records, and its validation records.

    A run killed mid-write can leave a truncated final line; it is skipped rather than
    allowed to abort the plot, since partial curves are exactly what one wants to
    inspect after an interrupted run.
    """
    header: dict = {}
    steps: list[dict] = []
    validations: list[dict] = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            print(f"  skipping malformed final line in {path}", file=sys.stderr)
            continue
        kind = record.get("type")
        if kind == "step":
            steps.append(record)
        elif kind == "validation":
            validations.append(record)
        else:
            # Only the run header lands here; folding validation records in would
            # overwrite its fields one pass at a time.
            header.update(record)
    return header, steps, validations


def moving_average(values: list[float], window: int) -> list[float]:
    if window <= 1:
        return values
    out = []
    for i in range(len(values)):
        lo = max(0, i - window + 1)
        chunk = values[lo : i + 1]
        out.append(sum(chunk) / len(chunk))
    return out


def label_for(path: Path, header: dict) -> str:
    conditioning = header.get("conditioning")
    name = path.parent.name if path.name == "metrics.jsonl" else path.stem
    return f"{conditioning} ({name})" if conditioning and conditioning != name else name


def main() -> int:
    import matplotlib

    matplotlib.use("Agg")  # no display on the training host
    import matplotlib.pyplot as plt

    args = parse_args()
    figure, axis = plt.subplots(figsize=(8, 5))

    plotted = 0
    for run in args.runs:
        path = resolve(run)
        header, steps, validations = read_run(path)
        if not steps:
            print(f"  {path} has no step records yet — skipping", file=sys.stderr)
            continue
        label = label_for(path, header)
        xs = [s["step"] for s in steps]
        ys = moving_average([s[args.metric] for s in steps], args.smooth)
        line, = axis.plot(xs, ys, label=f"{label} train", linewidth=1.6)
        plotted += 1

        # Validation on the same axes and in the same colour, dashed. Overfitting is
        # only visible as the *divergence* between the two, so plotting them apart —
        # or on separate figures at different scales — hides the thing worth seeing.
        if validations and args.metric == "loss":
            axis.plot([v["step"] for v in validations],
                      [v["val_loss"] for v in validations],
                      linestyle="--", linewidth=1.4, color=line.get_color(),
                      label=f"{label} val")
            best = min(validations, key=lambda v: v["val_loss"])
            axis.scatter([best["step"]], [best["val_loss"]], color=line.get_color(),
                         zorder=5, s=36, marker="o")

        final = steps[-1]
        summary = (f"{label:20s} steps={final['step']:>7d}  "
                   f"train={final[args.metric]:.4f}")
        if validations:
            best = min(validations, key=lambda v: v["val_loss"])
            summary += f"  val={validations[-1]['val_loss']:.4f}  best={best['val_loss']:.4f} @ {best['step']}"
        print(summary + f"  elapsed={final.get('elapsed_s', 0) / 60:.1f} min")

    if not plotted:
        raise SystemExit("nothing to plot")

    axis.set_xlabel("step")
    axis.set_ylabel({"loss": "masked L1 loss", "it_per_s": "iterations / s"}[args.metric])
    if args.log and args.metric == "loss":
        axis.set_yscale("log")
    axis.grid(alpha=0.3)
    axis.legend()
    axis.set_title(f"{args.metric} — {plotted} run{'s' if plotted > 1 else ''}"
                   + (f", smoothed over {args.smooth}" if args.smooth > 1 else ""))

    output = args.output or resolve(args.runs[0]).parent / f"{args.metric}.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(output, dpi=150)
    print(f"\nwrote {output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
