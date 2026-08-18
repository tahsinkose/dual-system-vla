"""Tests for System 1.

Run with::

    python -m pytest tests/test_system1.py -v
"""

from __future__ import annotations

import inspect
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models.system1 import System1, System1Config, tiny_config  # noqa: E402


@pytest.fixture(scope="module")
def model() -> System1:
    return System1(tiny_config())


def observation(model: System1, batch: int = 2, size: int = 128):
    cfg = model.config
    return (
        {key: torch.rand(batch, 3, size, size) for key in cfg.camera_keys},
        torch.rand(batch, cfg.state_dim),
        torch.rand(batch, cfg.latent_dim),
    )


def test_action_chunk_shape(model):
    images, state, latent = observation(model)
    actions = model(images, state, latent)
    assert actions.shape == (2, model.config.chunk_size, model.config.action_dim)


def test_has_no_language_input(model):
    """The architecture's central constraint, enforced at the signature level.

    If System 1 could read the instruction, it would not need the latent to identify
    the task, and the counterfactual the whole study rests on would be unprovable.
    """
    parameters = set(inspect.signature(System1.forward).parameters)
    assert parameters == {"self", "images", "state", "latent"}
    forbidden = {"text", "instruction", "language", "task", "prompt", "tokens"}
    assert not (parameters & forbidden)


def test_latent_changes_the_action(model):
    """`z` must actually drive the output.

    A model that ignores the latent would still train and still reach some success
    rate — while making every ablation modality produce identical numbers. That failure
    is invisible in the loss curve, so it is asserted here.
    """
    torch.manual_seed(0)
    model.eval()
    images, state, latent = observation(model)
    with torch.no_grad():
        a = model(images, state, latent)
        b = model(images, state, torch.rand_like(latent))
    assert not torch.allclose(a, b, atol=1e-5), "System 1 ignores the latent"


def test_gradients_reach_the_latent(model):
    """Gradients must flow back through `z`, or System 2 never learns.

    End-to-end training depends on this path existing: it is how System 1's loss
    reaches System 2's LoRA adapters.
    """
    images, state, latent = observation(model)
    latent = latent.clone().requires_grad_(True)
    model(images, state, latent).sum().backward()
    assert latent.grad is not None and latent.grad.abs().sum() > 0


def test_missing_camera_raises(model):
    images, state, latent = observation(model)
    del images[model.config.camera_keys[-1]]
    with pytest.raises(KeyError, match="missing camera views"):
        model(images, state, latent)


def test_multi_scale_gives_more_tokens_than_a_single_stage():
    """Multi-scale is the point of the backbone; verify the extra stage contributes."""
    from src.models.system1 import MultiScaleBackbone

    single = MultiScaleBackbone(tiny_config(backbone_stages=("layer4",)))
    multi = MultiScaleBackbone(tiny_config(backbone_stages=("layer3", "layer4")))
    frame = torch.rand(1, 3, 128, 128)
    assert multi(frame).shape[1] > single(frame).shape[1]


def test_rejects_unknown_backbone_stage():
    with pytest.raises(ValueError, match="unknown backbone stage"):
        tiny_config(backbone_stages=("layer9",))


def test_sequence_length_guard_is_explicit():
    """Too many visual tokens should say so, not index past the position embeddings."""
    model = System1(tiny_config(backbone_stages=("layer2", "layer3", "layer4")))
    model.position_embedding.data = model.position_embedding.data[:, :8]
    images, state, latent = observation(model)
    with pytest.raises(ValueError, match="exceeds"):
        model(images, state, latent)


def test_latent_dim_must_match_system2():
    """The two configs share one number; a mismatch should fail loudly at the boundary."""
    model = System1(tiny_config(latent_dim=32))
    images, state, _ = observation(model)
    with pytest.raises(RuntimeError):
        model(images, state, torch.rand(2, 64))


def test_default_size_is_in_the_intended_range():
    counts = System1(System1Config(pretrained_backbone=False)).parameter_counts()
    assert 50e6 < counts["total"] < 100e6, f"unexpected size: {counts['total'] / 1e6:.1f}M"
