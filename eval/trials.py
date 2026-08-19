"""What a single evaluation rollout starts from.

Two sources of initial simulator state, and the distinction decides whether a success
rate means anything:

**benchmark** (default) — LIBERO's own `.pruned_init` states, 50 per task, shipped with
the suite. This is the standard protocol, and the one published numbers are measured
under: train on a task's demonstrations, evaluate that task from the benchmark's
initial states. No demonstration ever starts from one of these, so every evaluated
configuration is unseen even though training used all 379 episodes.

**demo** — the initial state of a specific dataset episode, recovered by
`scripts/extract_init_states.py`. Useful for debugging against a known-good
demonstration (it is what `scripts/replay_episode.py` uses), but **not** a valid
success metric: training used every episode, so these are configurations the policy
has already seen.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from enum import Enum
from functools import lru_cache

import numpy as np


class InitSource(str, Enum):
    BENCHMARK = "benchmark"
    DEMO = "demo"


# LIBERO's published protocol evaluates 10 episodes per task.
DEFAULT_TRIALS_PER_TASK = 10


@dataclass(frozen=True)
class Trial:
    """One rollout: a task, an initial state, and a stable name for logs and videos."""

    task_dataset_index: int
    instruction: str
    bddl: str
    init_state: np.ndarray
    source: InitSource
    index: int          # benchmark init-state index, or dataset episode index

    @property
    def name(self) -> str:
        prefix = "init" if self.source is InitSource.BENCHMARK else "ep"
        return f"task{self.task_dataset_index:02d}_{prefix}{self.index:03d}"


@lru_cache(maxsize=None)
def benchmark_init_states(suite: str, bddl: str) -> np.ndarray:
    """The suite's `.pruned_init` states for one task, shape ``(n, state_dim)``."""
    import torch
    from libero.libero import get_libero_path

    path = os.path.join(get_libero_path("init_states"), suite,
                        os.path.basename(bddl).replace(".bddl", ".pruned_init"))
    if not os.path.exists(path):
        raise FileNotFoundError(f"no benchmark init states for {bddl}: {path}")
    return np.asarray(torch.load(path, weights_only=False))


def build_trials(
    mapping,
    dataset,
    source: InitSource,
    task_indices: list[int] | None = None,
    trials_per_task: int = DEFAULT_TRIALS_PER_TASK,
    episodes: list[int] | None = None,
    allow_unmatched: bool = False,
) -> list[Trial]:
    """Enumerate the rollouts to run, in a fixed order.

    Order is deterministic so that repeated invocations with different
    ``--checkpoint`` / ``--conditioning`` evaluate the identical set — the ablation
    compares modalities, which only means something over the same trials.
    """
    from src.utils import episodes_for_task, load_episode, load_exact_init_state

    tasks = task_indices if task_indices is not None else mapping.dataset_indices()

    if source is InitSource.BENCHMARK:
        trials = []
        for task_index in sorted(tasks):
            task = mapping.by_dataset_index(task_index)
            states = benchmark_init_states(task.suite, task.bddl)
            for init_index in range(min(trials_per_task, len(states))):
                trials.append(Trial(
                    task_dataset_index=task_index,
                    # The dataset's instruction string, which is what training
                    # conditioned System 2 on. Verified identical to the benchmark's
                    # for all ten libero_10 tasks, but taken from the mapping so the
                    # two cannot silently diverge.
                    instruction=task.instruction,
                    bddl=task.bddl,
                    init_state=np.asarray(states[init_index]),
                    source=source,
                    index=init_index,
                ))
        return trials

    # demo source: one trial per dataset episode
    if episodes is not None:
        candidates = sorted(episodes)
    else:
        candidates = sorted(e for t in tasks for e in episodes_for_task(dataset, t))

    trials = []
    dropped = []
    for episode_index in candidates:
        init_state = load_exact_init_state(episode_index)
        if init_state is None and not allow_unmatched:
            dropped.append(episode_index)
            continue
        task = mapping.by_episode(dataset, episode_index)
        instruction, _actions, _state = load_episode(dataset, episode_index)
        trials.append(Trial(
            task_dataset_index=task.dataset_index,
            instruction=instruction,
            bddl=task.bddl,
            init_state=None if init_state is None else np.asarray(init_state),
            source=source,
            index=episode_index,
        ))
    if dropped:
        print(f"skipping {len(dropped)} episode(s) with no recovered init state: {dropped} "
              "(see data/unmatched_episodes_per_task.json; pass --allow-unmatched-episodes "
              "to run them from the env's own reset instead)")
    return trials
