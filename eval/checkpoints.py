"""Load a trained DualSystem checkpoint for evaluation.

Reconstructs the model via `src.train.build_model(TrainConfig)`, never by hand-rolling
`DualSystemConfig` — that is the only reliable way to recover whatever chunk_size,
latent_dim, and tiny/real sizing the run actually used.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import torch

from src.models.dual_system import Conditioning, DualSystem
from src.train import TrainConfig, build_model

# The only ablation rows the plan defines. Anything else evaluates a checkpoint in a
# regime it was never trained for (out-of-distribution in latent space, per the
# "naive baseline needs its own training run" reasoning in the plan).
_VALID_ROWS = {
    (Conditioning.LIVE, Conditioning.LIVE),
    (Conditioning.LIVE, Conditioning.FROZEN),
    (Conditioning.LIVE, Conditioning.ZERO),
    (Conditioning.LIVE, Conditioning.SCENE_BLIND),
    (Conditioning.STATIC, Conditioning.STATIC),
}


@dataclass(frozen=True)
class LoadedCheckpoint:
    model: DualSystem
    train_config: TrainConfig
    checkpoint_path: Path
    step: int
    trained_conditioning: Conditioning


def resolve_checkpoint_path(path: Path) -> Path:
    """`path` may be a `step_XXXXXXX.pt` file or its containing run directory.

    For a directory, returns the highest-step `step_*.pt` found directly inside it
    (zero-padded step numbers sort lexicographically, so a plain sort suffices).
    """
    path = Path(path)
    if path.is_file():
        return path
    if path.is_dir():
        candidates = sorted(path.glob("step_*.pt"))
        if candidates:
            return candidates[-1]
        raise FileNotFoundError(
            f"no step_*.pt found in {path}; contents: {sorted(path.iterdir())}"
        )
    raise FileNotFoundError(f"checkpoint path does not exist: {path}")


def load_checkpoint(path: Path, device: str = "cpu") -> LoadedCheckpoint:
    """Load a checkpoint saved by `src.train.save_checkpoint`.

    Reads the sibling `config.json` (same directory) — the exact `TrainConfig` used
    for that run — reconstructs an untrained model via `build_model`, then loads the
    saved weights with `strict=True` so a shape or key mismatch (e.g. evaluating with
    the wrong `--tiny` flag) raises immediately rather than silently loading a partial
    model.

    Raises FileNotFoundError if `config.json` is missing next to the checkpoint.
    """
    checkpoint_path = resolve_checkpoint_path(Path(path))
    config_path = checkpoint_path.parent / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"no config.json next to checkpoint {checkpoint_path}")

    payload = json.loads(config_path.read_text())
    # json.dumps(..., default=str) stringified output_dir; TrainConfig.output_dir is
    # typed Path but dataclasses do not auto-coerce on construction, so this must be
    # wrapped explicitly or downstream Path operations on it would fail.
    payload["output_dir"] = Path(payload["output_dir"])
    train_config = TrainConfig(**payload)

    model = build_model(train_config)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.to(device).eval()

    return LoadedCheckpoint(
        model=model,
        train_config=train_config,
        checkpoint_path=checkpoint_path,
        step=int(checkpoint["step"]),
        trained_conditioning=Conditioning(checkpoint["conditioning"]),
    )


def warn_if_conditioning_mismatch(trained: Conditioning, requested: Conditioning) -> None:
    """Print a non-fatal warning for a nonsensical ablation-table row.

    Not a hard error — this project favours explaining over over-guarding, and a
    researcher may deliberately want to see how far out-of-distribution a mismatched
    combination degrades.
    """
    if (trained, requested) not in _VALID_ROWS:
        print(
            f"WARNING: checkpoint trained under conditioning={trained.value!r} is being "
            f"evaluated under conditioning={requested.value!r}. This is not a defined "
            "ablation-table row — feeding a mode the checkpoint never saw in training "
            "measures distribution shift, not the ablation you likely intend."
        )
