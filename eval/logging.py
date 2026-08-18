"""JSONL result logging, the latent-trace sidecar, and the run summary.

Written and flushed immediately after every episode — a killed run over the full
five-conditioning x ten-task x several-perturbation matrix should still yield a
readable prefix of results.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
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
    return "\n".join(lines)
