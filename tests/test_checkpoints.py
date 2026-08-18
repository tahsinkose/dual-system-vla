"""Tests for eval/checkpoints.py.

All fast: build a tiny DualSystem, save it exactly as src/train.py would, and load it
back through eval.checkpoints.

Run with::

    python -m pytest tests/test_checkpoints.py -v
"""

from __future__ import annotations

import itertools
import json
import sys
from dataclasses import asdict
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.env_setup import setup_env  # noqa: E402

setup_env()

from src.models.dual_system import Conditioning  # noqa: E402
from src.train import TrainConfig, build_model, save_checkpoint  # noqa: E402

from eval.checkpoints import (  # noqa: E402
    load_checkpoint,
    resolve_checkpoint_path,
    warn_if_conditioning_mismatch,
)


def _write_tiny_checkpoint(tmp_path: Path, conditioning: str = "live", step: int = 100) -> Path:
    """Writes a checkpoint + config.json exactly as src.train.train() would, without
    running the training loop."""
    cfg = TrainConfig(conditioning=conditioning, output_dir=tmp_path, tiny=True, chunk_size=4)
    model = build_model(cfg)
    output_dir = tmp_path / conditioning
    output_dir.mkdir(parents=True, exist_ok=True)
    # Matches train.py's own config.json write exactly, including output_dir as a
    # bare string via default=str — this is the quirk load_checkpoint must handle.
    (output_dir / "config.json").write_text(
        json.dumps({**asdict(cfg), "output_dir": str(cfg.output_dir)}, indent=2, default=str) + "\n"
    )
    return save_checkpoint(model, cfg, step, output_dir)


def test_checkpoint_round_trip_matches_train_py_save_format(tmp_path):
    path = _write_tiny_checkpoint(tmp_path, conditioning="live", step=100)
    loaded = load_checkpoint(path)

    assert loaded.step == 100
    assert loaded.trained_conditioning is Conditioning.LIVE
    assert loaded.checkpoint_path == path

    # A forward pass runs — the reconstructed architecture actually matches the
    # saved weights' shapes.
    cfg = loaded.model.config.system1
    images = {k: torch.rand(1, 3, 64, 64) for k in cfg.camera_keys}
    actions, z = loaded.model.act(images, torch.rand(1, cfg.state_dim),
                                  [torch.rand(3, 64, 64)], ["do the thing"])
    assert actions.shape == (1, cfg.chunk_size, cfg.action_dim)
    assert z.shape == (1, cfg.latent_dim)


def test_load_checkpoint_accepts_a_run_directory(tmp_path):
    path = _write_tiny_checkpoint(tmp_path, conditioning="static", step=50)
    loaded = load_checkpoint(path.parent)
    assert loaded.checkpoint_path == path


def test_resolve_checkpoint_path_picks_highest_step_in_a_run_dir(tmp_path):
    run_dir = tmp_path / "live"
    run_dir.mkdir()
    (run_dir / "step_0000100.pt").touch()
    (run_dir / "step_0005000.pt").touch()
    (run_dir / "step_0000999.pt").touch()

    resolved = resolve_checkpoint_path(run_dir)
    assert resolved.name == "step_0005000.pt"


def test_resolve_checkpoint_path_rejects_empty_directory(tmp_path):
    run_dir = tmp_path / "empty"
    run_dir.mkdir()
    with pytest.raises(FileNotFoundError, match="no step_\\*.pt found"):
        resolve_checkpoint_path(run_dir)


def test_resolve_checkpoint_path_rejects_missing_path(tmp_path):
    with pytest.raises(FileNotFoundError):
        resolve_checkpoint_path(tmp_path / "does-not-exist")


def test_load_checkpoint_requires_sibling_config(tmp_path):
    output_dir = tmp_path / "live"
    output_dir.mkdir()
    cfg = TrainConfig(conditioning="live", output_dir=tmp_path, tiny=True, chunk_size=4)
    model = build_model(cfg)
    path = save_checkpoint(model, cfg, 1, output_dir)   # no config.json written
    with pytest.raises(FileNotFoundError, match="no config.json"):
        load_checkpoint(path)


def test_load_checkpoint_is_strict_about_state_dict_shape(tmp_path):
    """Evaluating with the wrong architecture must raise, not silently partial-load."""
    path = _write_tiny_checkpoint(tmp_path, conditioning="live", step=1)
    config_path = path.parent / "config.json"
    payload = json.loads(config_path.read_text())
    payload["chunk_size"] = payload["chunk_size"] + 1   # architecture mismatch
    config_path.write_text(json.dumps(payload))

    with pytest.raises(RuntimeError):
        load_checkpoint(path)


@pytest.mark.parametrize("trained,requested", list(itertools.product(Conditioning, Conditioning)))
def test_warn_if_conditioning_mismatch_prints_for_invalid_rows_only(capsys, trained, requested):
    valid_rows = {
        (Conditioning.LIVE, Conditioning.LIVE),
        (Conditioning.LIVE, Conditioning.FROZEN),
        (Conditioning.LIVE, Conditioning.ZERO),
        (Conditioning.LIVE, Conditioning.SCENE_BLIND),
        (Conditioning.STATIC, Conditioning.STATIC),
    }
    warn_if_conditioning_mismatch(trained, requested)
    out = capsys.readouterr().out
    if (trained, requested) in valid_rows:
        assert out == ""
    else:
        assert "WARNING" in out
