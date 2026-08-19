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

from src.models.dual_system import Conditioning, DualSystem, DualSystemConfig  # noqa: E402
from src.models.system1 import System1Config  # noqa: E402
from src.models.system2 import System2Config  # noqa: E402
from src.observations import ModelObservation  # noqa: E402

REPO_ID = "lerobot/libero_10"

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

    # System 1 trains from scratch and wants a higher rate than the LoRA adapters
    # sitting on top of an already-good pretrained encoder.
    lr_system1: float = 1e-4
    lr_system2: float = 1e-5
    weight_decay: float = 1e-4
    grad_clip: float = 1.0

    chunk_size: int = 16
    temporal_offset: int = 2
    latent_dim: int = 512

    log_every: int = 100
    checkpoint_every: int = 5_000
    val_fraction: float = 0.1     # per task; 0 disables validation entirely
    val_every: int = 1_000        # steps between validation passes
    val_samples: int = 512        # bounds cost; spread across every task
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # Restrict to these dataset task indices; None means all ten.
    task_indices: list[int] | None = None
    # Smoke-test mode: overfit this many episodes. Passing a small number here is the
    # §1 training gate — loss must fall, gradients must reach System 2.
    overfit_episodes: int | None = None
    tiny: bool = False   # small random models, for testing the loop itself


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


def build_datasets(cfg: TrainConfig):
    """Return ``(train_dataset, val_dataset_or_None)``.

    The validation set exists only to choose a checkpoint and to reveal overfitting.
    It is *not* the success metric: that is measured by rolling out in the simulator
    from LIBERO's own initial states (see eval/trials.py), which no demonstration
    starts from.
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
        # Smoke-test mode deliberately has no validation: the point is to memorise a
        # handful of episodes, so a held-out loss would be meaningless.
        train_episodes = sorted(train_episodes + val_episodes)[: cfg.overfit_episodes]
        val_episodes = []

    train_dataset = LeRobotDataset(REPO_ID, episodes=sorted(train_episodes), delta_timestamps=deltas)
    val_dataset = (LeRobotDataset(REPO_ID, episodes=sorted(val_episodes), delta_timestamps=deltas)
                   if val_episodes else None)
    return train_dataset, val_dataset


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


def build_model(cfg: TrainConfig) -> DualSystem:
    if cfg.tiny:
        from src.models.dual_system import tiny_config
        from src.models.system1 import tiny_config as system1_tiny

        config = tiny_config(conditioning=Conditioning(cfg.conditioning),
                             temporal_offset=cfg.temporal_offset)
        # The dataset hands back chunks of cfg.chunk_size, so the model must predict
        # that many regardless of what the tiny defaults say.
        config.system1 = system1_tiny(latent_dim=config.system2.latent_dim,
                                      chunk_size=cfg.chunk_size)
        return DualSystem(config)
    return DualSystem(DualSystemConfig(
        system2=System2Config(latent_dim=cfg.latent_dim),
        system1=System1Config(latent_dim=cfg.latent_dim, chunk_size=cfg.chunk_size),
        conditioning=Conditioning(cfg.conditioning),
        temporal_offset=cfg.temporal_offset,
    ))


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

    def __init__(self, path: Path, cfg: TrainConfig, counts: dict[str, int]) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # Truncate: a fresh run should not append to a previous run's curve.
        with self.path.open("w") as handle:
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

    def log_validation(self, step: int, val_loss: float, is_best: bool) -> None:
        with self.path.open("a") as handle:
            handle.write(json.dumps({
                "type": "validation",
                "step": step,
                "val_loss": round(val_loss, 6),
                "is_best": is_best,
            }) + "\n")

    def log(self, step: int, loss: float, rate: float, elapsed: float) -> None:
        with self.path.open("a") as handle:
            handle.write(json.dumps({
                "type": "step",
                "step": step,
                "loss": round(loss, 6),
                "it_per_s": round(rate, 3),
                "elapsed_s": round(elapsed, 1),
            }) + "\n")


def validation_subset(dataset, max_samples: int):
    """A fixed subset that spans every task,.
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
            predicted, _latent = model(observation.images, observation.state,
                                       system2_frames, instructions)
            total += masked_l1_loss(predicted, actions, is_pad).item()
            batches += 1
    finally:
        model.train(was_training)
    return total / max(batches, 1)


# ------------------------------------------------------------------------- loop


def train(cfg: TrainConfig) -> Path:
    torch.manual_seed(cfg.seed)
    device = torch.device(cfg.device)

    dataset, val_dataset = build_datasets(cfg)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=cfg.batch_size, shuffle=True,
        num_workers=cfg.num_workers, pin_memory=device.type == "cuda", drop_last=True,
    )
    val_loader = None
    if val_dataset is not None:
        val_loader = torch.utils.data.DataLoader(
            # Fixed, task-spanning subset; not shuffled, so successive validation losses
            # differ because the model changed, not because the sample did.
            validation_subset(val_dataset, cfg.val_samples),
            batch_size=cfg.batch_size, shuffle=False,
            num_workers=max(1, cfg.num_workers // 2),
            pin_memory=device.type == "cuda", drop_last=False,
        )

    model = build_model(cfg).to(device)
    optimizer = build_optimizer(model, cfg)
    counts = model.parameter_counts()

    output_dir = Path(cfg.output_dir) / cfg.conditioning
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config.json").write_text(
        json.dumps({**asdict(cfg), "output_dir": str(cfg.output_dir)}, indent=2, default=str) + "\n"
    )

    print(f"conditioning : {cfg.conditioning}")
    held_out = val_dataset.num_episodes if val_dataset is not None else 0
    print(f"episodes     : {dataset.num_episodes} train | {held_out} validation")
    print(f"trainable    : {counts['trainable'] / 1e6:.2f}M of {counts['total'] / 1e6:.2f}M")
    print(f"device       : {device}")
    print(f"metrics      : {output_dir / 'metrics.jsonl'}\n")

    metrics = MetricsLog(output_dir / "metrics.jsonl", cfg, counts)

    model.train()
    step, started = 0, time.time()
    running = 0.0
    last_log_at = started
    best_val = float("inf")
    while step < cfg.steps:
        for batch in loader:
            if step >= cfg.steps:
                break
            observation, system2_frames, actions, is_pad, instructions = split_batch(batch)
            observation = observation.to(device)
            actions, is_pad = actions.to(device), is_pad.to(device)
            system2_frames = [f.to(device) for f in system2_frames]

            predicted, _latent = model(observation.images, observation.state,
                                       system2_frames, instructions)
            loss = masked_l1_loss(predicted, actions, is_pad)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],
                                     cfg.grad_clip)
            optimizer.step()

            running += loss.item()
            step += 1

            if step % cfg.log_every == 0:
                now = time.time()
                # Instantaneous rate over this interval, not the cumulative average:
                # a run that slows down should be visible while it happens.
                rate = cfg.log_every / max(now - last_log_at, 1e-9)
                mean_loss = running / cfg.log_every
                print(f"step {step:>7d} | train {mean_loss:.4f} | {rate:.2f} it/s")
                metrics.log(step, mean_loss, rate, now - started)
                running = 0.0
                last_log_at = now
            if val_loader is not None and step % cfg.val_every == 0:
                val_loss = validate(model, val_loader, device)
                improved = val_loss < best_val
                marker = "  <- best" if improved else f"  (best {best_val:.4f})"
                print(f"step {step:>7d} |   val {val_loss:.4f}{marker}")
                metrics.log_validation(step, val_loss, is_best=improved)
                if improved:
                    best_val = val_loss
                    # Written to a stable filename so evaluation never has to guess
                    # which step generalised best from a directory of checkpoints.
                    save_checkpoint(model, cfg, step, output_dir,
                                    filename="best.pt", val_loss=val_loss)
            if step % cfg.checkpoint_every == 0:
                save_checkpoint(model, cfg, step, output_dir)

    final = save_checkpoint(model, cfg, step, output_dir)
    print(f"\nfinished at step {step}: {final}")
    return final


def save_checkpoint(model: DualSystem, cfg: TrainConfig, step: int, output_dir: Path,
                    filename: str | None = None, val_loss: float | None = None) -> Path:
    path = output_dir / (filename or f"step_{step:07d}.pt")
    payload = {"step": step, "conditioning": cfg.conditioning,
               "state_dict": model.state_dict()}
    if val_loss is not None:
        payload["val_loss"] = val_loss
    torch.save(payload, path)
    return path


# -------------------------------------------------------------------------- cli


def parse_args(argv: list[str] | None = None) -> TrainConfig:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--conditioning", default="live", choices=["live", "static"],
                   help="'live' produces CKPT-DUAL, 'static' the naive-baseline CKPT-STATIC")
    p.add_argument("--output-dir", type=Path, default=Path("outputs/train"))
    p.add_argument("--steps", type=int, default=100_000)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--lr-system1", type=float, default=1e-4)
    p.add_argument("--lr-system2", type=float, default=1e-5)
    p.add_argument("--chunk-size", type=int, default=16)
    p.add_argument("--temporal-offset", type=int, default=2)
    p.add_argument("--device", default=TrainConfig.device)
    p.add_argument("--task-indices", type=int, nargs="+", default=None,
                   help="restrict to these dataset task indices (default: all ten)")
    p.add_argument("--overfit-episodes", type=int, default=None,
                   help="overfit this many episodes — the §1 training smoke-test gate")
    p.add_argument("--tiny", action="store_true",
                   help="small random models, for exercising the loop itself")
    p.add_argument("--val-fraction", type=float, default=0.1,
                   help="fraction of each task's episodes held out for validation "
                        "(default: 0.1; 0 disables). Used only to pick a checkpoint and "
                        "reveal overfitting — the success metric comes from simulator "
                        "rollouts against LIBERO's own initial states")
    p.add_argument("--val-every", type=int, default=1000, help="steps between validation passes")
    p.add_argument("--val-samples", type=int, default=512,
                   help="validation samples per pass, strided across every task")
    p.add_argument("--log-every", type=int, default=100)
    p.add_argument("--checkpoint-every", type=int, default=5_000)
    args = p.parse_args(argv)
    return TrainConfig(**vars(args))


if __name__ == "__main__":
    train(parse_args())
