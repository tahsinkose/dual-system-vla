"""JSONL result logging, the latent-trace sidecar, and the run summary.

Written and flushed immediately after every episode — a killed run over the full
five-conditioning x ten-task x several-perturbation matrix should still yield a
readable prefix of results.
"""

from __future__ import annotations

import dataclasses
import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class EpisodeResult:
    # identity
    task_dataset_index: int
    task_instruction: str
    episode_index: int
    seed: int
    checkpoint_path: str
    trained_conditioning: str
    eval_conditioning: str

    # Ablation 1 — counterfactual (success rate + step count)
    success: bool
    steps_run: int
    steps_to_success: int | None        # None if never succeeded; the more sensitive
                                         # long-horizon signal

    # Ablation 2 — error recovery
    perturbation_kind: str              # "none" if unperturbed
    perturbation_trigger_step: int | None
    perturbation_applied: bool          # False if the trigger condition never fired
                                         # (e.g. after_success_steps set but success
                                         # never happened) — NOT an error condition
    first_success_step: int | None      # first step check_success() was ever True
    latent_cosine_distance: float | None   # 1 - cos(z_pre, z_post); None if
                                         # unperturbed, never fired, or eval_conditioning
                                         # is ZERO (cosine undefined for a zero vector)
    recovered: bool | None              # None unless perturbation_applied; else whether
                                         # check_success() was True at any step strictly
                                         # after perturbation_trigger_step
    steps_to_recovery: int | None       # steps from trigger to first post-trigger
                                         # success, if recovered

    # Subtask progress — the resolution `success` alone cannot give on a long-horizon
    # task. One `eval.subtasks.SubtaskRecord` per step of the decomposition, in order.
    subtasks: list[dict] = field(default_factory=list)
    # Both counts exclude subtasks already satisfied at reset, so they read as progress
    # made out of progress available. The full decomposition is still in `subtasks`.
    subtasks_achieved: int = 0
    subtasks_total: int = 0

    # artifacts
    video_path: str | None = None
    latent_trace_path: str | None = None   # only set when --latent-trace was passed

    def to_json_line(self) -> str:
        return json.dumps(asdict(self))


class JsonlResultWriter:
    """Appends one EpisodeResult per call; each call is durable before returning."""

    def __init__(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._file = path.open("a", buffering=1)   # line-buffered

    def write(self, result: EpisodeResult) -> None:
        self._file.write(result.to_json_line() + "\n")
        self._file.flush()
        os.fsync(self._file.fileno())

    def close(self) -> None:
        self._file.close()


def read_results(path: Path) -> list[EpisodeResult]:
    """Parse a JSONL log back into `EpisodeResult`s.

    Unknown keys are dropped rather than raising, so a log written by an older or newer
    harness still reads; a *missing* key raises, since a summary computed over a
    defaulted `success` or `steps_run` would be quietly wrong rather than absent.
    """
    fields = {f.name for f in dataclasses.fields(EpisodeResult)}
    required = {f.name for f in dataclasses.fields(EpisodeResult)
                if f.default is dataclasses.MISSING
                and f.default_factory is dataclasses.MISSING}

    results = []
    with Path(path).open() as handle:
        for number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{number}: not valid JSON — {error}") from error
            missing = required - record.keys()
            if missing:
                raise ValueError(f"{path}:{number}: missing required field(s) "
                                 f"{sorted(missing)}")
            results.append(EpisodeResult(**{k: v for k, v in record.items() if k in fields}))
    return results


def write_latent_trace(path: Path, latents: np.ndarray, steps_since_update: np.ndarray) -> None:
    """Opt-in per-step latent dump for the (separately scoped) bottleneck-analysis probe.

    `latents`: (T, latent_dim), cast to float16 — the larger log artifact, kept as
    compact as the probe's downstream linear-regression accuracy can tolerate.
    `steps_since_update`: (T,) int16 — the latent-update-cadence position at each
    step, so the probe can later group steps sharing one System 2 call.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, latents=latents.astype(np.float16),
                        steps_since_update=steps_since_update.astype(np.int16))


def summarize(results: list[EpisodeResult]) -> str:
    """A convenience per-run success-rate line — NOT the headline-success-rate GATE
    itself. That gate is a decision made over the aggregated JSONL output across the
    full matrix, once every conditioning/checkpoint/task combination has been run; a
    single invocation of this harness only ever covers a slice of that matrix.
    """
    n = len(results)
    if n == 0:
        return "0 episodes run"
    successes = sum(r.success for r in results)
    steps = [r.steps_to_success if r.steps_to_success is not None else r.steps_run for r in results]
    lines = [f"{successes}/{n} succeeded ({successes / n:.1%})",
             f"steps (to success, else run): mean {np.mean(steps):.1f}, median {np.median(steps):.1f}"]
    perturbed = [r for r in results if r.perturbation_applied]
    if perturbed:
        recovered = sum(bool(r.recovered) for r in perturbed)
        lines.append(f"recovery: {recovered}/{len(perturbed)} perturbed episodes recovered "
                     f"({recovered / len(perturbed):.1%})")
    breakdown = subtask_breakdown(results)
    if breakdown:
        lines.append("")
        lines.append(breakdown)
    return "\n".join(lines)


def subtask_breakdown(results: list[EpisodeResult]) -> str:
    """Per-task, per-subtask completion counts over a run.

    The column that matters is where the count falls off: a task whose first pick
    completes 10/10 and whose second pick completes 0/10 is stalling at the subtask
    transition, which a single success rate reports only as a uniform zero.

    Subtasks already satisfied at reset are counted separately — crediting them to the
    policy would make a task look partly solved before it acted.
    """
    per_task: dict[tuple[int, str], list[EpisodeResult]] = {}
    for result in results:
        if result.subtasks:
            key = (result.task_dataset_index, result.task_instruction)
            per_task.setdefault(key, []).append(result)
    if not per_task:
        return ""

    lines = ["subtask completion (episodes achieving each step, in task order):"]
    for (task_index, instruction), group in sorted(per_task.items()):
        lines.append(f"  task {task_index:02d}  {instruction}")
        width = max(len(s["description"]) for r in group for s in r.subtasks)
        for position in range(max(len(r.subtasks) for r in group)):
            steps = [r.subtasks[position] for r in group if position < len(r.subtasks)]
            description = steps[0]["description"]
            if all(s["achieved_at_reset"] for s in steps):
                # Already true before the first action, so it is excluded from the
                # score and there is no completion count to report. What can still go
                # wrong is the policy undoing it — a rollout that knocks the stove off
                # fails the task — so report how many trials ended with it intact.
                held = sum(s["achieved_at_end"] for s in steps)
                lines.append(f"    {description:<{width}}  {held:>3d}/{len(steps):<3d} "
                             "still held at end (true at reset, not scored)")
            else:
                achieved = sum(s["first_achieved_step"] is not None for s in steps)
                lines.append(f"    {description:<{width}}  {achieved:>3d}/{len(steps):<3d}")
    return "\n".join(lines)
