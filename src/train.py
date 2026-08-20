"""Training loop for the dual system.

Produces the two checkpoints the ablation compares:

    python -m src.train --conditioning live    -> CKPT-DUAL
    python -m src.train --conditioning static  -> CKPT-STATIC

Both are trained identically apart from how `z` is produced, which is the point: any
difference in the results is attributable to the conditioning and nothing else.

**The loss lives here, not in the model.** `DualSystem.forward` stays a pure
`observation -> actions` function so the same forward serves training and rollout, so
one checkpoint can be evaluated under all five conditioning modes, and so the objective
can change without touching the architecture.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.env_setup import setup_env  # noqa: E402

setup_env()

from src.models.dual_system import (  # noqa: E402
    SYSTEM1_ARCHS,
    Conditioning,
    DualSystem,
    DualSystemConfig,
)
from src.models.system1 import System1Config  # noqa: E402
from src.models.system1_scratch import ScratchSystem1Config  # noqa: E402
from src.models.system2 import System2Config  # noqa: E402
from src.observations import ModelObservation  # noqa: E402

from eval.trials import DEFAULT_TRIALS_PER_TASK  # noqa: E402

REPO_ID = "lerobot/libero_10"

# Each System 1 carries its own action-chunk length, and they differ: ACT predicts 100
# steps, the scratch encoder-decoder 16. `TrainConfig` defers to whichever is selected,
# so neither runs off-design because the other's number was left in place.
DEFAULT_CHUNK_SIZE = {"act": System1Config.chunk_size,
                      "scratch": ScratchSystem1Config.chunk_size}

# What a checkpoint's weights are shaped by. Continuing a run has to reuse these
# exactly: a different value does not produce a differently-configured continuation, it
# produces a `load_state_dict` failure or — worse, where shapes happen to agree — a
# model being trained under a contract its weights were not learned under.
ARCHITECTURE_FIELDS = ("system1_arch", "chunk_size", "latent_dim", "temporal_offset",
                       "conditioning", "tiny")

# Tasks the in-training rollout gate scores on. Fixed rather than configurable: `best.pt`
# is only comparable across runs if every run was scored on the same set, and these three
# are the ones whose subtask breakdowns separate architectures rather than bottoming out.
GATE_TASK_INDICES = (5, 6, 9)

# The dataset stores System 1's cameras under these names; the wrist camera is renamed
# to the canonical key by the observation adapter.
DATASET_MAIN_CAMERA = "observation.images.image"
DATASET_WRIST_CAMERA = "observation.images.wrist_image"


@dataclass
class TrainConfig:
    conditioning: str = "live"
    output_dir: Path = Path("outputs/train")

    steps: int = 100_000
    batch_size: int = 32
    num_workers: int = 8
    seed: int = 0

    # Both at 1e-5 — ACT's own rate, for its transformer and backbone alike.
    lr_system1: float = 1e-5
    lr_system2: float = 1e-5
    weight_decay: float = 1e-4
    kl_weight: float = 10.0     # weight on the CVAE style latent's KLD term
    grad_clip: float = 1.0

    # Which System 1 to build: "scratch" the from-scratch encoder-decoder in
    # src/models/system1_scratch.py, "act" wraps LeRobot's ACT.
    system1_arch: str = "scratch"

    # None resolves to the selected architecture's default; see DEFAULT_CHUNK_SIZE.
    chunk_size: int | None = None
    # Δ₀ — the floor staleness of System 2's frame, modelling inference latency.
    temporal_offset: int = 2
    latent_dim: int = 512

    log_every: int = 100
    checkpoint_every: int = 5_000
    # Episodes withheld from training, per task, to score on. At the default 0 nothing
    # is withheld and the scoring pass runs in-sample — see `build_datasets`.
    val_fraction: float = 0.0
    val_every: int = 2_000        # steps between scoring passes; 0 disables them
    val_samples: int = 512        # bounds cost; spread across every task

    # In-simulator rollouts on GATE_TASK_INDICES, scored at the same cadence, selecting
    # `best.pt` by success rate. This is expensive — trials_per_task x 3 tasks episodes
    # of up to `gate_horizon` steps each, with a fresh LIBERO env per episode — and it
    # is what makes `best.pt` mean "solved the most tasks" rather than "fit the
    # demonstrations best", which loss alone has repeatedly failed to predict.
    gate_rollouts: bool = True
    gate_trials_per_task: int = DEFAULT_TRIALS_PER_TASK
    # A trained policy runs slower than the demonstrator it imitates, and task 6's
    # demonstrations already average 407 steps; too tight a horizon scores solved
    # episodes as failures.
    gate_horizon: int = 800
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # Restrict to these dataset task indices; None means all ten.
    task_indices: list[int] | None = None
    # Smoke-test mode: overfit this many episodes. Passing a small number here is the
    # §1 training gate — loss must fall, gradients must reach System 2.
    overfit_episodes: int | None = None
    tiny: bool = False   # small random models, for testing the loop itself
    # Checkpoint to continue from. The fields in ARCHITECTURE_FIELDS are then taken
    # from that run's config.json rather than from this one; see `resume_config`.
    resume: Path | None = None

    def __post_init__(self) -> None:
        if self.system1_arch not in SYSTEM1_ARCHS:
            raise ValueError(f"system1_arch must be one of {SYSTEM1_ARCHS}, "
                             f"got {self.system1_arch!r}")
        if self.chunk_size is None:
            self.chunk_size = DEFAULT_CHUNK_SIZE[self.system1_arch]


# ------------------------------------------------------------------------- data


def build_delta_timestamps(cfg: TrainConfig, fps: float) -> dict[str, list[float]]:
    """Ask the dataset for an action chunk and the System 2 / System 1 frame pair.

    The main camera is fetched at two timestamps — `t - Δ` for System 2 and `t` for
    System 1 — so the offset is applied by the loader rather than reconstructed later.
    """
    return {
        "action": [i / fps for i in range(cfg.chunk_size)],
        DATASET_MAIN_CAMERA: [-cfg.temporal_offset / fps, 0.0],
    }


def split_episodes(episodes: list[int], val_fraction: float) -> tuple[list[int], list[int]]:
    """Deterministic train/validation split of one task's episodes.

    Takes every k-th episode for validation rather than sampling randomly. Determinism
    matters more than randomness here: CKPT-DUAL and CKPT-STATIC must validate on the
    identical set for their curves to be comparable, and a stride needs no shared seed
    to guarantee that. Spreading the picks across the task also avoids correlating the
    split with whatever ordering the conversion happened to produce.
    """
    if val_fraction <= 0 or len(episodes) < 2:
        return list(episodes), []
    stride = max(2, round(1 / val_fraction))
    val = [e for i, e in enumerate(episodes) if i % stride == 0]
    # Never hand back an empty training set for a task with very few episodes.
    if len(val) >= len(episodes):
        val = val[:1]
    train = [e for e in episodes if e not in set(val)]
    return train, val


def resume_config(cfg: TrainConfig, explicit: set[str]) -> TrainConfig:
    """Fold a checkpoint's own configuration into `cfg`, for continuing that run.

    The checkpoint's `config.json` — the same sidecar `eval/checkpoints.py` reads — wins
    for everything in ARCHITECTURE_FIELDS, because those shaped the weights being
    loaded. Everything else is a property of *this* invocation: how long to keep going,
    on which device, at what learning rate, where to write.

    `explicit` names the fields the caller actually passed. Silently discarding an
    explicit `--chunk-size` that disagrees with the checkpoint would be the worst
    outcome, so it raises instead; a value that merely came from a default is
    overridden without comment.
    """
    checkpoint = resolve_resume_path(cfg.resume)
    config_path = checkpoint.parent / "config.json"
    if not config_path.exists():
        raise SystemExit(
            f"no config.json beside {checkpoint}. It records the architecture the "
            "weights were trained under, and without it a resume would have to guess."
        )

    saved = json.loads(config_path.read_text())
    conflicts = {name: (saved[name], getattr(cfg, name))
                 for name in ARCHITECTURE_FIELDS
                 if name in explicit and name in saved and saved[name] != getattr(cfg, name)}
    if conflicts:
        detail = "; ".join(f"--{name.replace('_', '-')}: checkpoint has {was!r}, "
                           f"you passed {now!r}" for name, (was, now) in conflicts.items())
        raise SystemExit(f"cannot change architecture while resuming — {detail}")

    merged = {**asdict(cfg)}
    for name in ARCHITECTURE_FIELDS:
        if name in saved:
            merged[name] = saved[name]
    merged["output_dir"] = Path(merged["output_dir"])
    merged["resume"] = checkpoint
    return TrainConfig(**merged)


def resolve_resume_path(path: Path) -> Path:
    """Accept a checkpoint file, or a run directory holding `best.pt`/the latest step."""
    path = Path(path)
    if path.is_file():
        return path
    if path.is_dir():
        best = path / "best.pt"
        if best.is_file():
            return best
        steps = sorted(path.glob("step_*.pt"))
        if steps:
            return steps[-1]
    raise SystemExit(f"no checkpoint at {path}")


def build_datasets(cfg: TrainConfig):
    """Return ``(train_dataset, score_dataset_or_None, score_is_held_out)``.

    Which episodes are *trained on* and which are *scored* are separate choices.

    At the default ``val_fraction == 0`` every episode trains and the scoring pass runs
    over training episodes: the loss it reports is in-sample, tracks the training loss,
    and says nothing about generalisation. It still yields a smooth, fixed-sample series
    to select a checkpoint from, which a per-step training loss over shuffled batches
    does not. Above 0, a stride of each task's episodes is withheld and scored instead,
    giving a held-out loss that reveals overfitting — paid for with that fraction of the
    training data.

    Neither number is the success metric. That is measured by rolling out in the
    simulator from LIBERO's own initial states (see eval/trials.py), which no
    demonstration starts from, and it does not track loss: configurations have scored
    competitively here and failed in simulation.
    """
    from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata

    from src.utils import episodes_for_task

    fps = LeRobotDatasetMetadata(REPO_ID).fps
    deltas = build_delta_timestamps(cfg, fps)
    full = LeRobotDataset(REPO_ID, delta_timestamps=deltas)

    task_indices = cfg.task_indices if cfg.task_indices is not None else None
    if task_indices is None:
        from src.utils import TaskMapping

        task_indices = TaskMapping.from_dataset(full).dataset_indices()

    # Split per task, so validation covers every task rather than whichever ones
    # happen to fall at the end of a global ordering.
    train_episodes: list[int] = []
    val_episodes: list[int] = []
    for task_index in sorted(task_indices):
        task_train, task_val = split_episodes(episodes_for_task(full, task_index),
                                              cfg.val_fraction)
        train_episodes += task_train
        val_episodes += task_val

    if cfg.overfit_episodes is not None:
        # Smoke-test mode deliberately has no scoring pass: the point is to memorise a
        # handful of episodes, so any loss over them is the training loss twice.
        train_episodes = sorted(train_episodes + val_episodes)[: cfg.overfit_episodes]
        train_dataset = LeRobotDataset(REPO_ID, episodes=train_episodes, delta_timestamps=deltas)
        return train_dataset, None, False

    train_dataset = LeRobotDataset(REPO_ID, episodes=sorted(train_episodes), delta_timestamps=deltas)
    if val_episodes:
        score_dataset = LeRobotDataset(REPO_ID, episodes=sorted(val_episodes),
                                       delta_timestamps=deltas)
        return train_dataset, score_dataset, True
    # Nothing withheld: score the training episodes themselves. `validation_subset`
    # strides across them, so the sample spans every task rather than the first few.
    return train_dataset, train_dataset, False


def split_batch(batch: dict) -> tuple[ModelObservation, list, torch.Tensor, torch.Tensor, list[str]]:
    """Unpack one dataloader batch into what each component needs.

    Returns ``(system1_observation, system2_frames, actions, action_is_pad, instructions)``.
    """
    from src.observations import from_dataset

    main = batch[DATASET_MAIN_CAMERA]          # (B, 2, 3, H, W): index 0 is t-Δ, 1 is t
    system2_frames = list(main[:, 0])
    observation = from_dataset({
        DATASET_MAIN_CAMERA: main[:, 1],
        DATASET_WRIST_CAMERA: batch[DATASET_WRIST_CAMERA],
        "observation.state": batch["observation.state"],
    })
    return (observation, system2_frames, batch["action"],
            batch["action_is_pad"], list(batch["task"]))


# ------------------------------------------------------------------------- loss


def masked_l1_loss(
    predicted: torch.Tensor,
    target: torch.Tensor,
    is_pad: torch.Tensor,
) -> torch.Tensor:
    """L1 over valid chunk steps only.

    `is_pad` marks steps where the chunk ran past the end of its episode. Those targets
    are padding, and averaging over them teaches the policy to imitate padding right at
    the end of an episode — where success is decided.
    """
    if predicted.shape != target.shape:
        # Broadcasting would otherwise turn a chunk-size mismatch between the model and
        # the dataloader into a silently wrong loss rather than an error.
        raise ValueError(
            f"prediction {tuple(predicted.shape)} does not match target "
            f"{tuple(target.shape)} — check that System 1's chunk_size equals the "
            "number of action timestamps requested from the dataset"
        )
    error = (predicted - target).abs()               # (B, chunk, action_dim)
    valid = (~is_pad).unsqueeze(-1).to(error.dtype)  # (B, chunk, 1)
    return (error * valid).sum() / valid.expand_as(error).sum().clamp(min=1.0)


def kld_loss(mu: torch.Tensor, log_sigma_x2: torch.Tensor) -> torch.Tensor:
    """KL divergence of the CVAE style latent from a unit Gaussian, summed over dims.

    Regularises the variable System 1's CVAE encoder infers from the ground-truth action
    sequence. Without it the encoder is free to smuggle the whole chunk through the
    style latent, which is available in training and zeroed at rollout — the policy
    would fit the data and then behave differently the moment it was deployed.
    """
    return (-0.5 * (1 + log_sigma_x2 - mu.pow(2) - log_sigma_x2.exp())).sum(-1).mean()


def action_loss(
    predicted: torch.Tensor,
    target: torch.Tensor,
    is_pad: torch.Tensor,
    style_latent_params: tuple[torch.Tensor | None, torch.Tensor | None],
    kl_weight: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    """ACT's objective: masked L1 reconstruction plus the weighted KLD term.

    Both operands are in normalised action space. Computing this on raw actions would
    silently reweight the objective across dimensions — the rotation deltas vary by
    roughly a tenth of what the gripper command does, so the two spaces disagree about
    their relative importance by an order of magnitude.

    Returns the scalar to backpropagate plus its components, which are logged
    separately: a run where the KLD term dominates is a different failure from one where
    reconstruction stalls, and their sum hides which happened.
    """
    l1 = masked_l1_loss(predicted, target, is_pad)
    mu, log_sigma_x2 = style_latent_params
    if mu is None:
        return l1, {"l1": l1.item(), "kld": 0.0}
    kld = kld_loss(mu, log_sigma_x2)
    return l1 + kl_weight * kld, {"l1": l1.item(), "kld": kld.item()}


# --------------------------------------------------------------------- optimiser


def build_optimizer(model: DualSystem, cfg: TrainConfig) -> torch.optim.Optimizer:
    """Separate rates: pretrained-and-adapted System 2 vs from-scratch System 1."""
    system2 = [p for p in model.system2.parameters() if p.requires_grad]
    system1 = [p for p in model.system1.parameters() if p.requires_grad]
    return torch.optim.AdamW(
        [
            {"params": system1, "lr": cfg.lr_system1},
            {"params": system2, "lr": cfg.lr_system2},
        ],
        weight_decay=cfg.weight_decay,
    )


def system1_stats(dataset, cfg: TrainConfig):
    """The normalisation statistics System 1 is trained against.

    Taken from the dataset metadata rather than a sidecar file, so training and every
    later evaluation read one source. `--tiny` skips them: its inputs are random tensors
    whose statistics describe nothing.
    """
    if cfg.tiny or cfg.system1_arch != "act":
        # The scratch architecture trains on raw actions and holds no statistics.
        return None
    from src.models.system1 import NormalisationStats

    config = System1Config(latent_dim=cfg.latent_dim, chunk_size=cfg.chunk_size)
    return NormalisationStats.from_dataset(dataset.meta, config)


def build_model(cfg: TrainConfig, stats=None) -> DualSystem:
    """Assemble the model, optionally with the dataset's normalisation statistics.

    `stats` is omitted when restoring a checkpoint: the statistics are persistent
    buffers, so a strict `load_state_dict` supplies them, and reading them from a
    dataset that may not be the one the run trained on would be the wrong source.
    """
    if cfg.tiny:
        from src.models.dual_system import tiny_config
        if cfg.system1_arch == "act":
            from src.models.system1 import tiny_config as system1_tiny
        else:
            from src.models.system1_scratch import tiny_config as system1_tiny

        config = tiny_config(conditioning=Conditioning(cfg.conditioning),
                             temporal_offset=cfg.temporal_offset,
                             system1_arch=cfg.system1_arch)
        # The dataset hands back chunks of cfg.chunk_size, so the model must predict
        # that many regardless of what the tiny defaults say.
        config.system1 = system1_tiny(latent_dim=config.system2.latent_dim,
                                      chunk_size=cfg.chunk_size)
        return DualSystem(config, system1_stats=stats)
    system1_config = (System1Config if cfg.system1_arch == "act" else ScratchSystem1Config)(
        latent_dim=cfg.latent_dim, chunk_size=cfg.chunk_size)
    return DualSystem(DualSystemConfig(
        system2=System2Config(latent_dim=cfg.latent_dim),
        system1=system1_config,
        system1_arch=cfg.system1_arch,
        conditioning=Conditioning(cfg.conditioning),
        temporal_offset=cfg.temporal_offset,
    ), system1_stats=stats)


# ---------------------------------------------------------------------- metrics


class MetricsLog:
    """Append-only JSONL record of training progress.

    One flushed line per logging interval, so a run that is killed still leaves usable
    history — the ablation compares two checkpoints, and their loss curves are part of
    the result, not just a debugging aid. JSONL rather than TensorBoard or W&B because
    it needs no server or account, diffs cleanly, and can be replotted long afterwards.

    Also records the *instantaneous* rate rather than the cumulative average, which is
    what reveals a run slowing down (thermal throttling, a contended dataloader) — a
    running average hides that behind its own history.
    """

    def __init__(self, path: Path, cfg: TrainConfig, counts: dict[str, int],
                 append: bool = False) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # Truncate for a fresh run, so it does not append to a previous run's curve.
        # A resume appends instead: the continuation is the same curve, and its own
        # header line marks where it picked up.
        with self.path.open("a" if append else "w") as handle:
            handle.write(json.dumps({
                "type": "run",
                "conditioning": cfg.conditioning,
                "steps": cfg.steps,
                "batch_size": cfg.batch_size,
                "chunk_size": cfg.chunk_size,
                "temporal_offset": cfg.temporal_offset,
                "lr_system1": cfg.lr_system1,
                "lr_system2": cfg.lr_system2,
                "seed": cfg.seed,
                "trainable_parameters": counts["trainable"],
                "total_parameters": counts["total"],
            }) + "\n")

    def log_validation(self, step: int, val_loss: float, is_best: bool,
                       held_out: bool = True, success_rate: float | None = None,
                       successes_per_task: dict[int, int] | None = None) -> None:
        with self.path.open("a") as handle:
            handle.write(json.dumps({
                "type": "validation",
                "step": step,
                # False when nothing was withheld from training: the loss is in-sample,
                # and a plot that treats it as a generalisation curve would mislead.
                "held_out": held_out,
                # Present once the rollout gate is running; it, not the loss, is what
                # `is_best` reflects.
                "success_rate": success_rate,
                "successes_per_task": successes_per_task,
                "val_loss": round(val_loss, 6),
                "is_best": is_best,
            }) + "\n")

    def log(self, step: int, loss: float, rate: float, elapsed: float,
            kld: float | None = None) -> None:
        record = {
            "type": "step",
            "step": step,
            "loss": round(loss, 6),
            "it_per_s": round(rate, 3),
            "elapsed_s": round(elapsed, 1),
        }
        if kld is not None:
            record["kld"] = round(kld, 6)
        with self.path.open("a") as handle:
            handle.write(json.dumps(record) + "\n")


def validation_subset(dataset, max_samples: int):
    """A fixed, task-spanning subset.

    Fixed, so successive scoring passes move with the model rather than the sample.
    Samples are ordered by episode, so the stride is what spreads them across tasks.
    """
    from torch.utils.data import Subset

    if max_samples >= len(dataset):
        return dataset
    stride = max(1, len(dataset) // max_samples)
    return Subset(dataset, list(range(0, len(dataset), stride))[:max_samples])


@torch.no_grad()
def validate(model: DualSystem, loader, device: torch.device) -> float:
    """Mean masked L1 over held-out episodes.

    Switches to eval mode and restores training mode afterwards — forgetting the
    restore is a classic silent bug, since training would continue with dropout
    disabled and the loss curve would look *better*, not worse.
    """
    was_training = model.training
    model.eval()
    total, batches = 0.0, 0
    try:
        for batch in loader:
            observation, system2_frames, actions, is_pad, instructions = split_batch(batch)
            observation = observation.to(device)
            actions, is_pad = actions.to(device), is_pad.to(device)
            system2_frames = [frame.to(device) for frame in system2_frames]
            # `eval()` above zeroes the CVAE style latent, so validation measures the
            # reconstruction the deployed policy is actually capable of — the KLD term
            # has no counterpart here and the L1 is directly comparable to training's.
            predicted, _latent, _style = model(observation.images, observation.state,
                                               system2_frames, instructions)
            total += masked_l1_loss(predicted, model.system1.normalise_action(actions),
                                    is_pad).item()
            batches += 1
    finally:
        model.train(was_training)
    return total / max(batches, 1)


@dataclass
class Restored:
    """What a resumed run inherits from the checkpoint it continues."""

    step: int
    path: Path
    # The score `best.pt` was last written at, when the checkpoint records one. Without
    # it a resumed run starts from the sentinel and overwrites `best.pt` on its first
    # scoring pass, however badly that pass went — discarding the bar the earlier run
    # had already cleared.
    success_rate: float | None = None
    val_loss: float | None = None


def restore(model: DualSystem, optimizer: torch.optim.Optimizer, path: Path,
            device: torch.device) -> Restored:
    """Load weights and optimiser state from `path`.

    `strict=True`: a key or shape mismatch means the architecture does not match the one
    that produced these weights, and continuing from a partially-loaded model would
    train something that silently is not the checkpoint.

    Checkpoints written before optimiser state was recorded still load; the moments
    restart, which is announced rather than hidden because it perturbs the first few
    hundred updates.
    """
    checkpoint_path = resolve_resume_path(path)
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(payload["state_dict"], strict=True)
    if "optimizer" in payload:
        optimizer.load_state_dict(payload["optimizer"])
    else:
        print(f"NOTE: {checkpoint_path.name} carries no optimiser state; Adam's moment "
              "estimates restart from zero.")
    return Restored(step=int(payload["step"]), path=checkpoint_path,
                    success_rate=payload.get("success_rate"),
                    val_loss=payload.get("val_loss"))


class RolloutGate:
    """Scores a checkpoint by success rate in the simulator, on a fixed task set.

    Runs the *same* rollout as `eval/run_eval.py` — the harness's `run_episode`, its
    trial construction, LIBERO's own initial states — so a number reported here means
    what the same number means at evaluation time. Anything reimplemented would be free
    to drift from it.

    Trials are enumerated once and reused, so every scoring pass faces an identical set
    and successive rates differ because the policy changed. Environments are built per
    episode rather than cached: LIBERO carries mutable simulator state, and a reused env
    would let one episode's outcome bias the next.
    """

    def __init__(self, dataset, cfg: TrainConfig, device: torch.device) -> None:
        from eval.run_eval import EvalConfig
        from eval.trials import InitSource, build_trials
        from src.utils import TaskMapping

        self._device = device
        self._mapping = TaskMapping.from_dataset(dataset)
        self._trials = build_trials(self._mapping, dataset, InitSource.BENCHMARK,
                                    task_indices=list(GATE_TASK_INDICES),
                                    trials_per_task=cfg.gate_trials_per_task)
        self._conditioning = Conditioning(cfg.conditioning)
        self._eval_cfg = EvalConfig(checkpoint=Path("in-training"),
                                    horizon=cfg.gate_horizon,
                                    log_path=Path(cfg.output_dir) / "gate.jsonl")

    def __len__(self) -> int:
        return len(self._trials)

    @property
    def task_indices(self) -> tuple[int, ...]:
        return GATE_TASK_INDICES

    def score(self, model: DualSystem) -> tuple[float, dict[int, int]]:
        """Returns ``(success_rate, successes_per_task)``.

        Restores training mode afterwards; leaving the model in eval mode would disable
        dropout for the remainder of the run and make the loss curve look better rather
        than worse.
        """
        from eval.perturbations import PerturbationKind, PerturbationSpec
        from eval.run_eval import build_env, run_episode

        was_training = model.training
        model.eval()
        per_task = {index: 0 for index in GATE_TASK_INDICES}
        successes = 0
        try:
            for trial in self._trials:
                task = self._mapping.by_dataset_index(trial.task_dataset_index)
                env = build_env(task, self._eval_cfg.camera_size, self._eval_cfg.horizon)
                try:
                    env.seed(0)   # determinism comes from the forced initial state
                    result = run_episode(
                        model, env, trial,
                        eval_conditioning=self._conditioning,
                        trained_conditioning=self._conditioning,
                        perturbation=PerturbationSpec(kind=PerturbationKind.NONE,
                                                      trigger=None,
                                                      episode_seed=(0, trial.index)),
                        cfg=self._eval_cfg, checkpoint_path="in-training",
                        device=self._device)
                finally:
                    env.close()
                successes += result.success
                per_task[trial.task_dataset_index] += result.success
        finally:
            model.train(was_training)
        return successes / max(len(self._trials), 1), per_task


# ------------------------------------------------------------------------- loop


def train(cfg: TrainConfig) -> Path:
    torch.manual_seed(cfg.seed)
    device = torch.device(cfg.device)

    dataset, score_dataset, score_is_held_out = build_datasets(cfg)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=cfg.batch_size, shuffle=True,
        num_workers=cfg.num_workers, pin_memory=device.type == "cuda", drop_last=True,
    )
    val_loader = None
    if score_dataset is not None and cfg.val_every > 0:
        val_loader = torch.utils.data.DataLoader(
            # Fixed, task-spanning subset; not shuffled, so successive losses differ
            # because the model changed, not because the sample did.
            validation_subset(score_dataset, cfg.val_samples),
            batch_size=cfg.batch_size, shuffle=False,
            num_workers=max(1, cfg.num_workers // 2),
            pin_memory=device.type == "cuda", drop_last=False,
        )

    model = build_model(cfg, stats=system1_stats(dataset, cfg)).to(device)
    optimizer = build_optimizer(model, cfg)
    counts = model.parameter_counts()

    start_step, resumed = 0, None
    if cfg.resume is not None:
        resumed = restore(model, optimizer, cfg.resume, device)
        start_step = resumed.step

    gate = None
    if cfg.gate_rollouts and cfg.val_every > 0 and not cfg.tiny:
        gate = RolloutGate(dataset, cfg, device)

    output_dir = Path(cfg.output_dir) / cfg.conditioning
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config.json").write_text(
        json.dumps({**asdict(cfg), "output_dir": str(cfg.output_dir)}, indent=2, default=str) + "\n"
    )

    print(f"conditioning : {cfg.conditioning}")
    if score_dataset is None:
        scoring = "none"
    elif score_is_held_out:
        scoring = f"{score_dataset.num_episodes} held-out episodes"
    else:
        scoring = "in-sample/nothing with-held"
    print(f"episodes     : {dataset.num_episodes} train")
    print(f"scored on    : {scoring}")
    if gate is not None:
        print(f"best.pt by   : success rate over {len(gate)} rollouts "
              f"(tasks {', '.join(str(i) for i in gate.task_indices)}), every "
              f"{cfg.val_every} steps")
    else:
        print("best.pt by   : loss" if cfg.val_every > 0 else "best.pt by   : not selected")
    print(f"trainable    : {counts['trainable'] / 1e6:.2f}M of {counts['total'] / 1e6:.2f}M")
    print(f"device       : {device}")
    if resumed is not None:
        inherited = ""
        if resumed.success_rate is not None:
            inherited = f", carrying best {resumed.success_rate:.1%}"
        elif resumed.val_loss is not None:
            inherited = f", carrying best loss {resumed.val_loss:.4f}"
        print(f"resumed      : {resumed.path} at step {start_step}{inherited}")
    print(f"metrics      : {output_dir / 'metrics.jsonl'}\n")

    metrics = MetricsLog(output_dir / "metrics.jsonl", cfg, counts,
                         append=cfg.resume is not None)

    model.train()
    step, started = start_step, time.time()
    # Reconstruction and KLD are accumulated apart: their sum hides whether a stalled
    # run is failing to fit the actions or being dominated by the regulariser.
    running = 0.0
    running_kld = 0.0
    last_log_at = started
    # A resumed run inherits the bar its checkpoint was written at, so `best.pt` is only
    # replaced by something that actually beats it. A checkpoint that records neither
    # score leaves the sentinel, and the first scoring pass sets the bar.
    best_val = float("inf")
    best_gate = (-1.0, float("-inf"))
    if resumed is not None:
        if resumed.val_loss is not None:
            best_val = resumed.val_loss
        if resumed.success_rate is not None:
            best_gate = (resumed.success_rate, -(resumed.val_loss or 0.0))
    while step < cfg.steps:
        for batch in loader:
            if step >= cfg.steps:
                break
            observation, system2_frames, actions, is_pad, instructions = split_batch(batch)
            observation = observation.to(device)
            actions, is_pad = actions.to(device), is_pad.to(device)
            system2_frames = [f.to(device) for f in system2_frames]

            predicted, _latent, style_latent = model(
                observation.images, observation.state, system2_frames, instructions,
                actions=actions, action_is_pad=is_pad)
            loss, components = action_loss(
                predicted, model.system1.normalise_action(actions), is_pad,
                style_latent, cfg.kl_weight)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],
                                     cfg.grad_clip)
            optimizer.step()

            running += components["l1"]
            running_kld += components["kld"]
            step += 1

            if step % cfg.log_every == 0:
                now = time.time()
                # Instantaneous rate over this interval, not the cumulative average:
                # a run that slows down should be visible while it happens.
                rate = cfg.log_every / max(now - last_log_at, 1e-9)
                mean_loss = running / cfg.log_every
                mean_kld = running_kld / cfg.log_every
                print(f"step {step:>7d} | l1 {mean_loss:.4f} | kld {mean_kld:.4f} "
                      f"| {rate:.2f} it/s")
                metrics.log(step, mean_loss, rate, now - started, kld=mean_kld)
                running = 0.0
                running_kld = 0.0
                last_log_at = now
            if cfg.val_every > 0 and step % cfg.val_every == 0:
                val_loss = None
                if val_loader is not None:
                    val_loss = validate(model, val_loader, device)
                    label = "val" if score_is_held_out else "in-sample"
                    print(f"step {step:>7d} | {label:>9s} {val_loss:.4f}")

                if gate is not None:
                    rate, per_task = gate.score(model)
                    # Success rate decides, loss only breaks ties: rollouts are the
                    # quantity the study reports, and configurations have scored
                    # competitively on loss here while failing in simulation.
                    improved = (rate, -(val_loss if val_loss is not None else 0.0)) > best_gate
                    marker = "  <- best" if improved else f"  (best {best_gate[0]:.1%})"
                    breakdown = "  ".join(f"t{index}:{count}/{cfg.gate_trials_per_task}"
                                          for index, count in sorted(per_task.items()))
                    print(f"step {step:>7d} |   gate {rate:.1%} of {len(gate)}"
                          f"  [{breakdown}]{marker}")
                    if improved:
                        best_gate = (rate, -(val_loss if val_loss is not None else 0.0))
                else:
                    improved = val_loss is not None and val_loss < best_val
                    if improved:
                        best_val = val_loss
                    rate, per_task = None, None

                if val_loss is not None:
                    metrics.log_validation(step, val_loss, is_best=improved,
                                           held_out=score_is_held_out,
                                           success_rate=rate, successes_per_task=per_task)
                if improved:
                    # A stable filename so evaluation never has to guess which step to
                    # take from a directory of checkpoints.
                    save_checkpoint(model, cfg, step, output_dir,
                                    filename="best.pt", val_loss=val_loss,
                                    optimizer=optimizer, success_rate=rate)
            if step % cfg.checkpoint_every == 0:
                save_checkpoint(model, cfg, step, output_dir, optimizer=optimizer)

    final = save_checkpoint(model, cfg, step, output_dir, optimizer=optimizer)
    print(f"\nfinished at step {step}: {final}")
    return final


def save_checkpoint(model: DualSystem, cfg: TrainConfig, step: int, output_dir: Path,
                    filename: str | None = None, val_loss: float | None = None,
                    optimizer: torch.optim.Optimizer | None = None,
                    success_rate: float | None = None) -> Path:
    path = output_dir / (filename or f"step_{step:07d}.pt")
    payload = {"step": step, "conditioning": cfg.conditioning,
               "state_dict": model.state_dict()}
    if optimizer is not None:
        # Adam's moments are part of the training state, not a nicety: resuming without
        # them restarts the moment estimates and perturbs the first few hundred updates.
        payload["optimizer"] = optimizer.state_dict()
    if val_loss is not None:
        payload["val_loss"] = val_loss
    if success_rate is not None:
        payload["success_rate"] = success_rate
    torch.save(payload, path)
    return path


# -------------------------------------------------------------------------- cli


def parse_args(argv: list[str] | None = None) -> TrainConfig:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--resume", type=Path, default=None,
                   help="continue from a checkpoint, or from a run directory (best.pt, "
                        "else the latest step_*.pt). Architecture settings come from "
                        "that run's config.json; everything else from this command")
    p.add_argument("--conditioning", default=TrainConfig.conditioning,
                   choices=["live", "static"],
                   help="'live' produces CKPT-DUAL, 'static' the naive-baseline CKPT-STATIC")
    p.add_argument("--output-dir", type=Path, default=TrainConfig.output_dir)
    p.add_argument("--steps", type=int, default=TrainConfig.steps)
    p.add_argument("--batch-size", type=int, default=TrainConfig.batch_size)
    p.add_argument("--num-workers", type=int, default=TrainConfig.num_workers)
    p.add_argument("--seed", type=int, default=TrainConfig.seed)
    p.add_argument("--lr-system1", type=float, default=TrainConfig.lr_system1)
    p.add_argument("--lr-system2", type=float, default=TrainConfig.lr_system2)
    p.add_argument("--system1-arch", default=TrainConfig.system1_arch,
                   choices=list(SYSTEM1_ARCHS),
                   help="System 1 implementation: 'scratch' uses the from-scratch "
                        "encoder-decoder (default), 'act' wraps LeRobot's ACT")
    p.add_argument("--chunk-size", type=int, default=None,
                   help="actions predicted per forward pass; defaults to the value the "
                        "selected --system1-arch was designed around ("
                        + ", ".join(f"{a}={n}" for a, n in DEFAULT_CHUNK_SIZE.items()) + ")")
    p.add_argument("--kl-weight", type=float, default=TrainConfig.kl_weight,
                   help="weight on the KLD term regularising the CVAE style latent")
    p.add_argument("--temporal-offset", type=int, default=TrainConfig.temporal_offset,
                   help="Δ₀: floor staleness of System 2's frame, in env steps")
    p.add_argument("--device", default=TrainConfig.device)
    p.add_argument("--task-indices", type=int, nargs="+", default=None,
                   help="restrict to these dataset task indices (default: all ten)")
    p.add_argument("--overfit-episodes", type=int, default=None,
                   help="overfit this many episodes — the §1 training smoke-test gate")
    p.add_argument("--tiny", action="store_true",
                   help="small random models, for exercising the loop itself")
    p.add_argument("--val-fraction", type=float, default=TrainConfig.val_fraction,
                   help="fraction of each task's episodes held out from training and "
                        "scored (default: 0, nothing withheld). At 0 every episode "
                        "trains and the scoring pass runs in-sample: best.pt is still "
                        "written, but the loss says nothing about generalisation. Above "
                        "0 the held-out loss reveals overfitting, at the cost of that "
                        "fraction of the training data. Either way the success metric "
                        "comes from simulator rollouts against LIBERO's own initial "
                        "states")
    p.add_argument("--val-every", type=int, default=TrainConfig.val_every,
                   help="steps between scoring passes; 0 disables them and best.pt")
    p.add_argument("--no-gate-rollouts", dest="gate_rollouts", action="store_false",
                   default=TrainConfig.gate_rollouts,
                   help="select best.pt by loss instead of by in-simulator success rate")
    p.add_argument("--gate-trials-per-task", type=int,
                   default=TrainConfig.gate_trials_per_task,
                   help=f"rollouts per gate task (default: "
                        f"{TrainConfig.gate_trials_per_task}, over tasks "
                        + ", ".join(str(i) for i in GATE_TASK_INDICES) + ")")
    p.add_argument("--gate-horizon", type=int, default=TrainConfig.gate_horizon,
                   help="max env steps per gate rollout")
    p.add_argument("--val-samples", type=int, default=TrainConfig.val_samples,
                   help="validation samples per pass, strided across every task")
    p.add_argument("--log-every", type=int, default=TrainConfig.log_every)
    p.add_argument("--checkpoint-every", type=int, default=TrainConfig.checkpoint_every)
    args = p.parse_args(argv)
    config = TrainConfig(**vars(args))
    if config.resume is None:
        return config
    # Which fields the caller actually named, as opposed to inherited from a default.
    # `resume_config` needs the distinction: an explicit architecture flag that
    # disagrees with the checkpoint is an error, an inherited one is simply replaced.
    tokens = sys.argv[1:] if argv is None else argv
    explicit = {token.lstrip("-").replace("-", "_").split("=")[0]
                for token in tokens if token.startswith("--")}
    return resume_config(config, explicit)


if __name__ == "__main__":
    train(parse_args())
