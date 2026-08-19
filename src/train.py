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


def build_dataset(cfg: TrainConfig):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata

    from src.utils import episodes_for_task

    fps = LeRobotDatasetMetadata(REPO_ID).fps
    dataset = LeRobotDataset(REPO_ID, delta_timestamps=build_delta_timestamps(cfg, fps))

    episodes = None
    if cfg.task_indices is not None:
        episodes = sorted(e for t in cfg.task_indices for e in episodes_for_task(dataset, t))
    if cfg.overfit_episodes is not None:
        episodes = (episodes or list(range(dataset.num_episodes)))[: cfg.overfit_episodes]

    if episodes is not None:
        dataset = LeRobotDataset(REPO_ID, episodes=episodes,
                                 delta_timestamps=build_delta_timestamps(cfg, fps))
    return dataset


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

    def log(self, step: int, loss: float, rate: float, elapsed: float) -> None:
        with self.path.open("a") as handle:
            handle.write(json.dumps({
                "type": "step",
                "step": step,
                "loss": round(loss, 6),
                "it_per_s": round(rate, 3),
                "elapsed_s": round(elapsed, 1),
            }) + "\n")


# ------------------------------------------------------------------------- loop


def train(cfg: TrainConfig) -> Path:
    torch.manual_seed(cfg.seed)
    device = torch.device(cfg.device)

    dataset = build_dataset(cfg)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=cfg.batch_size, shuffle=True,
        num_workers=cfg.num_workers, pin_memory=device.type == "cuda", drop_last=True,
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
    print(f"episodes     : {dataset.num_episodes} | frames: {dataset.num_frames}")
    print(f"trainable    : {counts['trainable'] / 1e6:.2f}M of {counts['total'] / 1e6:.2f}M")
    print(f"device       : {device}")
    print(f"metrics      : {output_dir / 'metrics.jsonl'}\n")

    metrics = MetricsLog(output_dir / "metrics.jsonl", cfg, counts)

    model.train()
    step, started = 0, time.time()
    running = 0.0
    last_log_at = started
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
                print(f"step {step:>7d} | loss {mean_loss:.4f} | {rate:.2f} it/s")
                metrics.log(step, mean_loss, rate, now - started)
                running = 0.0
                last_log_at = now
            if step % cfg.checkpoint_every == 0:
                save_checkpoint(model, cfg, step, output_dir)

    final = save_checkpoint(model, cfg, step, output_dir)
    print(f"\nfinished at step {step}: {final}")
    return final


def save_checkpoint(model: DualSystem, cfg: TrainConfig, step: int, output_dir: Path) -> Path:
    path = output_dir / f"step_{step:07d}.pt"
    torch.save({"step": step, "conditioning": cfg.conditioning,
                "state_dict": model.state_dict()}, path)
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
    p.add_argument("--log-every", type=int, default=100)
    p.add_argument("--checkpoint-every", type=int, default=5_000)
    args = p.parse_args(argv)
    return TrainConfig(**vars(args))


if __name__ == "__main__":
    train(parse_args())
