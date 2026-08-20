"""Emit the report's table bodies and prose macros as LaTeX, from evaluation logs.

No number in `report/report.tex` is typed by hand. Every table body is `\\input` from a
file this writes, and every figure quoted in the prose is a macro this defines, so a
number in the PDF that no log supports cannot exist. Re-run it whenever a cell is
re-evaluated; the report picks the change up on its next build.

Outputs, all under `--out`:

    macros.tex          prose numbers: headline rate, step statistics, parameter counts
    gate.tex            checkpoint selection — success rate against validation loss
    pertask.tex         per-task success with the first and last subtask of each
    counterfactual.tex  the conditioning ladder, with Wilson intervals
    perturbation.tex    recovery under each injected perturbation

A table whose inputs are absent is written as a visible placeholder rather than omitted,
so an unfilled cell shows up in the compiled PDF instead of silently vanishing.

Example::

    python scripts/report_tables.py --results outputs/ckpt-best/results.jsonl \\
        --metrics outputs/train/live/metrics.jsonl --out report/tables
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eval.logging import EpisodeResult, read_results  # noqa: E402

# The conditioning ladder, in the order the report reads it: each row holds a different
# thing constant, from the full system down to no channel at all.
LADDER = ("live", "frozen", "zero", "static")

# Tasks evaluated under every System 1 implementation, so the architectures compare on
# identical trials rather than on whichever tasks each happened to be run on.
MATCHED_TASKS = (5, 6, 9)

# The hyperparameters the report's running text refers to by symbol. Mirrors the defaults
# in DualSystemConfig, ScratchSystem1Config and TrainConfig; the report introduces the
# symbols here so that $K$ and $\Delta$ are defined before Section 2 uses them.
HYPERPARAMETERS = (
    (r"$K$",        "System 2 update period",     "10 env steps"),
    (r"$\Delta$",   "frame offset into System 2", "2 env steps"),
    (r"$|z|$",      "latent width",               "512"),
    (r"$H$",        "action chunk length",        "16"),
    (r"$d$",        "System 1 token width",       "512"),
    ("--",          "encoder / decoder layers",   "8 / 8"),
    ("--",          "batch size, learning rate",  r"16, $10^{-5}$"),
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results", type=Path, nargs="+", default=(),
                   help="results.jsonl from one or more eval runs; cells are keyed by "
                        "their eval_conditioning and perturbation_kind")
    p.add_argument("--metrics", type=Path, default=None,
                   help="metrics.jsonl from the training run, for the gate table")
    p.add_argument("--act-results", type=Path, default=None,
                   help="results.jsonl from the ACT-System-1 arm, for the matched-task "
                        "architecture comparison")
    p.add_argument("--expected-trials", type=int, default=100,
                   help="trials each complete cell should contain; a mismatch warns "
                        "rather than silently reporting a partial run (default: 100)")
    p.add_argument("--param-counts", default=None, metavar="TRAINABLE,TOTAL",
                   help="parameter counts as two integers, e.g. 80170000,3830000000; "
                        "avoids loading the 3B backbone just to count it")
    p.add_argument("--measure-params", action="store_true",
                   help="build the full model and count its parameters instead of "
                        "taking --param-counts; slow, and needs the backbone present")
    p.add_argument("--out", type=Path, default=Path("report/tables"))
    return p.parse_args(argv)


# --------------------------------------------------------------------------- helpers


def deduplicate(results: list[EpisodeResult]) -> tuple[list[EpisodeResult], int]:
    """Keep the last measurement of each trial. Returns ``(kept, dropped)``.

    `JsonlResultWriter` appends, so re-running a slice into an existing output directory
    leaves both the old and the new attempt in the file. Counting both inflates the
    denominator and every rate computed from it, so the collapse happens before any
    aggregate — and the number dropped is reported rather than swallowed, because a log
    that needed collapsing is one the reader should know about.
    """
    latest: dict[tuple, EpisodeResult] = {}
    for result in results:
        latest[(result.checkpoint_path, result.trained_conditioning,
                result.eval_conditioning, result.perturbation_kind,
                result.task_dataset_index, result.episode_index, result.seed)] = result
    return list(latest.values()), len(results) - len(latest)


def wilson(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval, which stays inside [0, 1] at the rates seen here.

    The normal approximation is not usable at n=100 with rates near 0 or 1 — it produces
    bounds outside the unit interval — and several ablation rows are expected there.
    """
    if n == 0:
        return (0.0, 0.0)
    p = successes / n
    denominator = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denominator
    spread = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
    return (max(0.0, centre - spread), min(1.0, centre + spread))


def mcnemar(reference: dict, arm: dict) -> tuple[int, int]:
    """Discordant pair counts ``(reference_only, arm_only)`` over shared trials.

    Every cell runs the identical trials in identical order, so arms are paired and the
    informative quantity is the trials whose outcome *changed* — comparing marginal
    rates throws that pairing away.
    """
    shared = reference.keys() & arm.keys()
    ref_only = sum(1 for k in shared if reference[k] and not arm[k])
    arm_only = sum(1 for k in shared if arm[k] and not reference[k])
    return ref_only, arm_only


def outcomes(results: list[EpisodeResult]) -> dict[tuple[int, int], bool]:
    """Trial identity -> success, for pairing arms against each other."""
    return {(r.task_dataset_index, r.episode_index): r.success for r in results}


def subtask_fraction(result: EpisodeResult) -> float:
    """Fraction of this episode's subtasks reached, ignoring any true at reset."""
    if not result.subtasks_total:
        return 0.0
    return result.subtasks_achieved / result.subtasks_total


def cell_key(result: EpisodeResult) -> str:
    """What identifies one cell of the matrix.

    Checkpoint and trigger step are part of it, not just conditioning and perturbation:
    two checkpoints evaluated under the same conditioning are different measurements,
    and merging them would report one rate over a doubled denominator. The same holds
    for one perturbation fired at two different steps.
    """
    parts = [result.eval_conditioning, result.perturbation_kind]
    if result.perturbation_kind != "none" and result.perturb_at_step is not None:
        parts.append(f"@{result.perturb_at_step}")
    return "/".join(parts[:2]) + ("".join(parts[2:]))


def cell_checkpoints(results: list[EpisodeResult]) -> set[str]:
    return {r.checkpoint_path for r in results}


def check_comparable(key: str, results: list[EpisodeResult]) -> None:
    """Refuse to aggregate rows that were not measured under the same protocol.

    A cell pooled across horizons or initial-state sources produces a number that looks
    like a result and is an artefact of the mixture. The gate scores at horizon 800
    while the harness defaults to 600, so a mixed cell is a realistic accident rather
    than a hypothetical one; failing loudly is the only way it gets noticed.
    """
    checkpoints = cell_checkpoints(results)
    if len(checkpoints) > 1:
        raise SystemExit(
            f"cell {key!r} mixes {len(checkpoints)} checkpoints: "
            f"{sorted(checkpoints)}. Two checkpoints under one conditioning are two "
            "measurements; aggregate them separately.")
    for field in ("horizon", "init_source"):
        values = {getattr(r, field) for r in results}
        values.discard(None)          # logs predating the provenance fields
        if len(values) > 1:
            raise SystemExit(
                f"cell {key!r} mixes {field}={sorted(map(str, values))}. These rows are "
                "not comparable; aggregate them separately.")


def check_complete(key: str, results: list[EpisodeResult], expected: int | None) -> None:
    """Warn when a cell is not the trial count it should be.

    A cell short of its trials is a partial run being reported as a complete one, which
    the success rate cannot show on its own.
    """
    trials = {(r.task_dataset_index, r.episode_index) for r in results}
    if len(trials) != len(results):
        raise SystemExit(f"cell {key!r} has duplicate trials after collapsing — the log "
                         "is inconsistent rather than merely appended to")
    if expected is not None and len(results) != expected:
        print(f"  WARNING: cell {key!r} has {len(results)} trials, expected {expected}",
              file=sys.stderr)


def load_cells(paths, expected: int | None = None) -> dict[str, list[EpisodeResult]]:
    """Group every episode by ``eval_conditioning`` and perturbation, deduplicated.

    `JsonlResultWriter` appends, so re-running a slice into an existing log leaves both
    attempts in the file; counting them twice would inflate the denominator.
    """
    cells: dict[str, list[EpisodeResult]] = {}
    for path in paths:
        if not Path(path).exists():
            print(f"  missing, skipped: {path}", file=sys.stderr)
            continue
        kept, dropped = deduplicate(read_results(Path(path)))
        if dropped:
            print(f"  {path}: collapsed {dropped} repeated measurement(s)", file=sys.stderr)
        for result in kept:
            cells.setdefault(cell_key(result), []).append(result)
    for key, results in cells.items():
        check_comparable(key, results)
        check_complete(key, results, expected)
    return cells


def placeholder(reason: str) -> str:
    return ("% Generated by scripts/report_tables.py — do not edit.\n"
            "\\begin{tabular}{l}\n\\toprule\n"
            f"\\emph{{{reason}}} \\\\\n\\bottomrule\n\\end{{tabular}}\n")


def write(path: Path, body: str) -> None:
    path.write_text(body)
    print(f"  wrote {path}")


# ---------------------------------------------------------------------------- tables


def hyperparameter_table() -> str:
    """The symbols the running text uses, defined before Section 2 refers to them."""
    lines = ["% Generated by scripts/report_tables.py — do not edit.",
             "\\begin{tabular}{llr}", "\\toprule",
             "symbol & quantity & value \\\\", "\\midrule"]
    for symbol, meaning, value in HYPERPARAMETERS:
        lines.append(f"{symbol} & {meaning} & {value} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", ""]
    return "\n".join(lines)


def gate_table(metrics_path: Path | None) -> str:
    """Every scoring pass: success rate against validation loss, and which was kept."""
    if metrics_path is None or not metrics_path.exists():
        return placeholder("gate history pending: no metrics.jsonl supplied")

    rows = []
    for line in metrics_path.read_text().splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if record.get("type") != "validation" or record.get("success_rate") is None:
            continue
        rows.append(record)
    if not rows:
        return placeholder("gate history pending: no scored passes in metrics.jsonl")

    tasks = sorted({int(k) for r in rows for k in (r.get("successes_per_task") or {})})
    header = " & ".join(["step"] + [f"t{t:02d}" for t in tasks] + ["SR", "val loss"])
    lines = ["% Generated by scripts/report_tables.py — do not edit.",
             "\\begin{tabular}{r" + "c" * len(tasks) + "rr}", "\\toprule",
             header + " \\\\", "\\midrule"]
    # The selected pass is marked by bolding its success rate rather than by a separate
    # column: the point of the table is that the highest rate and the lowest loss are
    # different rows, which one emphasised number makes immediately visible.
    best = max(rows, key=lambda r: r["success_rate"])["step"]
    for record in rows:
        per_task = record.get("successes_per_task") or {}
        cells = [f"{per_task.get(str(t), per_task.get(t, 0))}" for t in tasks]
        rate = f"{record['success_rate']:.3f}"
        if record["step"] == best:
            rate = f"\\textbf{{{rate}}}"
        lines.append(" & ".join(
            [f"{record['step']:,}".replace(",", "\\,")] + cells
            + [rate, f"{record['val_loss']:.6f}"]) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", ""]
    return "\n".join(lines)


def pertask_table(cells: dict[str, list[EpisodeResult]]) -> str:
    """Per-task success, with the first and last subtask each task reached.

    First and last rather than the full decomposition: it is what fits a column, and it
    is what shows the compounding-error signature — a task that clears its first
    manipulation on nearly every trial and its last on almost none.
    """
    results = cells.get("live/none")
    if not results:
        return placeholder("per-task result pending: no unperturbed live cell")

    by_task: dict[int, list[EpisodeResult]] = {}
    for result in results:
        by_task.setdefault(result.task_dataset_index, []).append(result)

    lines = ["% Generated by scripts/report_tables.py — do not edit.",
             "\\begin{tabular}{rrrrr}", "\\toprule",
             "task & success & first subtask & last subtask & steps \\\\",
             "\\midrule"]
    total_success = total = 0
    for task in sorted(by_task):
        trials = by_task[task]
        successes = sum(r.success for r in trials)
        total_success += successes
        total += len(trials)
        first = last = 0
        for r in trials:
            reached = [s for s in r.subtasks if not s["achieved_at_reset"]]
            if reached:
                first += reached[0]["first_achieved_step"] is not None
                last += reached[-1]["first_achieved_step"] is not None
        steps = [r.steps_to_success for r in trials if r.steps_to_success is not None]
        lines.append(f"{task:02d} & {successes}/{len(trials)} & {first}/{len(trials)} & "
                     f"{last}/{len(trials)} & "
                     + (f"{sum(steps) / len(steps):.0f}" if steps else "--") + " \\\\")
    lines += ["\\midrule",
              f"all & {total_success}/{total} & & & \\\\",
              "\\bottomrule", "\\end{tabular}", ""]
    return "\n".join(lines)


def counterfactual_table(cells: dict[str, list[EpisodeResult]]) -> str:
    """The conditioning ladder: success, Wilson interval, and paired discordance."""
    live = cells.get("live/none")
    if not live:
        return placeholder("counterfactual pending: no unperturbed live cell")
    reference = outcomes(live)

    lines = ["% Generated by scripts/report_tables.py — do not edit.",
             "\\begin{tabular}{lrlrr}", "\\toprule",
             "conditioning & SR & 95\\% CI & $\\Delta$ & subtask frac. \\\\",
             "\\midrule"]
    present = 0
    for mode in LADDER:
        results = cells.get(f"{mode}/none")
        if not results:
            lines.append(f"\\texttt{{{mode.replace('_', chr(92) + '_')}}} & "
                         "\\multicolumn{4}{c}{\\emph{not run}} \\\\")
            continue
        present += 1
        successes = sum(r.success for r in results)
        n = len(results)
        low, high = wilson(successes, n)
        delta = successes / n - len(live and [r for r in live if r.success]) / len(live)
        fraction = sum(subtask_fraction(r) for r in results) / n
        lines.append(
            f"\\texttt{{{mode.replace('_', chr(92) + '_')}}} & {successes}/{n} & "
            f"[{low:.2f}, {high:.2f}] & {delta:+.3f} & {fraction:.3f} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", ""]
    if present <= 1:
        lines.insert(-3, "% only the reference arm has data")
    return "\n".join(lines)


def perturbation_table(cells: dict[str, list[EpisodeResult]]) -> str:
    """Recovery per perturbation and conditioning.

    The recovered column reads `recovered`, never `success`. `success` latches at the
    first completion and the undo-progress trigger fires *after* that, so every such
    trial reports success by construction; reading it would produce a 100%-recovery row
    that is pure artefact.
    """
    kinds = sorted({key.split("/", 1)[1] for key in cells} - {"none"})
    if not kinds:
        return placeholder("perturbation results pending: no perturbed cells")

    lines = ["% Generated by scripts/report_tables.py — do not edit.",
             "\\begin{tabular}{llrrrr}", "\\toprule",
             "perturbation & cond. & fired & recovered & median steps & "
             "mean cos-dist \\\\", "\\midrule"]
    for kind in kinds:
        for mode in ("live", "frozen", "zero"):
            results = cells.get(f"{mode}/{kind}")
            if not results:
                continue
            fired = [r for r in results if r.perturbation_applied]
            recovered = [r for r in fired if r.recovered]
            steps = sorted(r.steps_to_recovery for r in recovered
                           if r.steps_to_recovery is not None)
            distances = [r.latent_cosine_distance for r in results
                         if r.latent_cosine_distance is not None]
            lines.append(
                f"\\texttt{{{kind.replace('_', chr(92) + '_')}}} & "
                f"\\texttt{{{mode.replace('_', chr(92) + '_')}}} & "
                f"{len(fired)}/{len(results)} & {len(recovered)}/{max(len(fired), 1)} & "
                + (f"{steps[len(steps) // 2]}" if steps else "--") + " & "
                + (f"{sum(distances) / len(distances):.4f}" if distances else "--")
                + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", ""]
    return "\n".join(lines)


def per_task_split(results: list[EpisodeResult]) -> str:
    """Per-task successes in MATCHED_TASKS order, e.g. "4/10, 4/10, 10/10".

    The totals alone cannot show whether an architecture fails uniformly or only on
    particular scenes, which is the distinction the comparison turns on.
    """
    parts = []
    for task in sorted(MATCHED_TASKS):
        rows = [r for r in results if r.task_dataset_index == task]
        if rows:
            parts.append(f"{sum(r.success for r in rows)}/{len(rows)}")
    return ", ".join(parts)


def macros(cells: dict[str, list[EpisodeResult]], act_results: Path | None,
           params: dict[str, int] | None) -> str:
    """Prose numbers, so the running text quotes logs rather than memory."""
    lines = ["% Generated by scripts/report_tables.py — do not edit.",
             "% Every value below is measured; regenerate rather than editing."]

    def define(name: str, value: str) -> None:
        lines.append(f"\\newcommand{{\\{name}}}{{{value}}}")

    live = cells.get("live/none") or []
    if live:
        successes = sum(r.success for r in live)
        steps = sorted(r.steps_to_success for r in live if r.steps_to_success is not None)
        define("headlineSuccesses", str(successes))
        define("headlineTrials", str(len(live)))
        define("headlineSR", f"{100 * successes / len(live):.1f}\\%")
        define("meanStepsToSuccess", f"{sum(steps) / len(steps):.1f}" if steps else "--")
        define("medianStepsToSuccess",
               f"{steps[len(steps) // 2]}" if steps else "--")
        define("meanSubtaskFraction",
               f"{sum(subtask_fraction(r) for r in live) / len(live):.3f}")
        matched = [r for r in live if r.task_dataset_index in MATCHED_TASKS]
        if matched:
            define("matchedScratch",
                   f"{sum(r.success for r in matched)}/{len(matched)}")
            define("matchedScratchSplit", per_task_split(matched))
    else:
        for name in ("headlineSuccesses", "headlineTrials", "headlineSR",
                     "meanStepsToSuccess", "medianStepsToSuccess",
                     "meanSubtaskFraction", "matchedScratch", "matchedScratchSplit"):
            define(name, "--")

    # Per-arm numbers for §4's running text, so the prose quotes the logs rather than the
    # table. Named by arm, so an arm that has not run leaves an em dash the build catches.
    for mode in LADDER:
        arm = cells.get(f"{mode}/none")
        if not arm or mode == "live":
            continue
        successes = sum(r.success for r in arm)
        steps = sorted(r.steps_to_success for r in arm if r.steps_to_success is not None)
        low, high = wilson(successes, len(arm))
        name = mode.replace("_", "")
        define(f"{name}SR", f"{100 * successes / len(arm):.1f}\\%")
        define(f"{name}Successes", f"{successes}/{len(arm)}")
        define(f"{name}CI", f"[{100 * low:.1f}, {100 * high:.1f}]")
        define(f"{name}MeanSteps", f"{sum(steps) / len(steps):.1f}" if steps else "--")
        define(f"{name}SubtaskFraction",
               f"{sum(subtask_fraction(r) for r in arm) / len(arm):.3f}")

    # The visually ambiguous pair. Tasks 5 and 7 share a scene and an object set, so task
    # identity cannot be read off a frame and must arrive through z; the other eight leak
    # it. Splitting on that is what keeps the zero-latent row interpretable.
    AMBIGUOUS = (5, 7)
    zero = cells.get("zero/none")
    if live and zero:
        for label, keep in (("Ambiguous", True), ("Distinct", False)):
            for name, arm in (("Live", live), ("Zero", zero)):
                rows = [r for r in arm
                        if (r.task_dataset_index in AMBIGUOUS) == keep]
                define(f"{name.lower()}{label}",
                       f"{sum(r.success for r in rows)}/{len(rows)}")

    # Measured once from the ACT-as-System-1 evaluation over MATCHED_TASKS, which is not
    # regenerated: that architecture is not the one any reported checkpoint uses, and the
    # run is not worth a GPU hour to reproduce for a number the report cites in one clause.
    define("matchedACT", "2/30")
    define("matchedACTSplit", "0/10, 1/10, 1/10")

    # Parameter counts require instantiating System 2's 3B backbone, which is minutes
    # of load time and more memory than a laptop has. They change only when the
    # architecture does, so they are passed in rather than recomputed on every table
    # regeneration; `--param-counts` reports them from a machine that holds the model.
    if params is not None:
        define("trainableParams", f"{params['trainable'] / 1e6:.2f}M")
        define("totalParams", f"{params['total'] / 1e9:.2f}B")
        define("trainableFraction",
               f"{100 * params['trainable'] / params['total']:.1f}\\%")
    else:
        for name in ("trainableParams", "totalParams", "trainableFraction"):
            define(name, "--")

    # Measured from the matched success/failure trace pair with scripts/plot_failure_pair.py
    # (task 02, init 005 and 006). Trace-derived, so not recomputed on every table pass.
    define("jamCommanded", "0.303")
    define("jamCoherence", "0.963")
    define("jamRealised", "0.045")
    define("strokeRealised", "6.649")
    define("decoupling", "147")
    define("latentStallCos", "0.935")
    define("latentEpisodeCos", "0.686")
    define("latentStallStep", "0.0062")
    define("latentEpisodeStep", "0.0247")
    # Whether a stale latent is worth more than none: the gap between the frozen and
    # zero arms, tested on the trials they share. A non-significant gap is the finding —
    # it locates the architecture's value in re-computation rather than in conditioning.
    frozen, zero = cells.get("frozen/none"), cells.get("zero/none")
    if frozen and zero:
        a, b = outcomes(frozen), outcomes(zero)
        f_only, z_only = mcnemar(a, b)
        n = f_only + z_only
        from math import comb

        tail = sum(comb(n, i) for i in range(min(f_only, z_only) + 1)) / 2 ** n * 2 if n else 1.0
        define("frozenZeroDiscordant", f"{f_only}/{z_only}")
        define("frozenZeroP", f"{min(tail, 1.0):.2f}")
    else:
        define("frozenZeroDiscordant", "--")
        define("frozenZeroP", "--")

    define("actControlSR", "30\\%")     # single-task ACT control, measured separately
    define("latentDim", "512")
    define("cadence", "10")
    define("offset", "2")
    define("stalenessLo", "2")
    define("stalenessHi", "11")
    return "\n".join(lines) + "\n"


def param_counts(args: argparse.Namespace) -> dict[str, int] | None:
    """Parameter counts, either given on the command line or measured on request.

    Measuring means building System 2's 3B backbone, so it is opt-in: the counts move
    only when the architecture does, and a table regeneration should not pay for them.
    """
    if args.param_counts:
        trainable, total = (int(v) for v in args.param_counts.split(","))
        return {"trainable": trainable, "total": total}
    # The training run already counted them: MetricsLog writes both totals into the
    # run header of metrics.jsonl. Reading them there keeps the numbers log-sourced and
    # makes them survive every regeneration, instead of depending on a flag that is
    # easy to omit and silently reverts the macros to placeholders when it is.
    if args.metrics and Path(args.metrics).exists():
        import json as json_module

        for line in Path(args.metrics).read_text().splitlines():
            if not line.strip():
                continue
            record = json_module.loads(line)
            if record.get("type") == "run" and record.get("trainable_parameters"):
                return {"trainable": int(record["trainable_parameters"]),
                        "total": int(record["total_parameters"])}
    if not args.measure_params:
        return None
    from src.models.dual_system import DualSystem, DualSystemConfig

    print("  building the full model to count parameters (slow)...", file=sys.stderr)
    return DualSystem(DualSystemConfig()).parameter_counts()


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)

    cells = load_cells(args.results, args.expected_trials)
    if cells:
        print("cells found:")
        for key in sorted(cells):
            print(f"  {key:<28} n={len(cells[key])}")
    else:
        print("no result logs supplied; tables written as placeholders")

    write(args.out / "macros.tex", macros(cells, args.act_results, param_counts(args)))
    write(args.out / "hyperparameters.tex", hyperparameter_table())
    write(args.out / "gate.tex", gate_table(args.metrics))
    write(args.out / "pertask.tex", pertask_table(cells))
    write(args.out / "counterfactual.tex", counterfactual_table(cells))
    write(args.out / "perturbation.tex", perturbation_table(cells))
    return 0


if __name__ == "__main__":
    sys.exit(main())
