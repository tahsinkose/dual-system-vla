"""Tests for the composed dual system.

Run with::

    python -m pytest tests/test_dual_system.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models.dual_system import (  # noqa: E402
    Conditioning,
    DualSystem,
    DualSystemConfig,
    tiny_config,
)
from src.models.system1 import tiny_config as s1_tiny  # noqa: E402
from src.models.system2 import tiny_config as s2_tiny  # noqa: E402

INSTRUCTION = "put both the alphabet soup and the tomato sauce in the basket"


@pytest.fixture(scope="module")
def model() -> DualSystem:
    """Inference mode: System 1's CVAE encoder runs only in training mode, and demands
    the ground-truth action chunk when it does. Tests that need it opt in via
    `supervision()`."""
    return DualSystem(tiny_config()).eval()


def observation(model: DualSystem, batch: int = 1, size: int = 128):
    cfg = model.config.system1
    return (
        {key: torch.rand(batch, 3, size, size) for key in cfg.camera_keys},
        torch.rand(batch, cfg.state_dim),
        [torch.rand(3, size, size) for _ in range(batch)],
        [INSTRUCTION] * batch,
    )


def supervision(model: DualSystem, batch: int = 1):
    """The ground-truth chunk and padding mask System 1's CVAE encoder consumes."""
    cfg = model.config.system1
    return (torch.rand(batch, cfg.chunk_size, cfg.action_dim),
            torch.zeros(batch, cfg.chunk_size, dtype=torch.bool))


# ------------------------------------------------------------------------ wiring


def test_forward_returns_actions_latent_and_style_latent(model):
    images, state, s2_images, instructions = observation(model)
    actions, latent, style_latent = model(images, state, s2_images, instructions)
    assert actions.shape == (1, model.config.system1.chunk_size, model.config.system1.action_dim)
    assert latent.shape == (1, model.config.system1.latent_dim)
    # Zeroed at inference, so its distribution parameters have no value to report.
    assert style_latent == (None, None)


def test_forward_infers_the_style_latent_when_training(model):
    """The CVAE parameters come back so the training loop can add the KLD term.

    The loss lives outside the model — that is what keeps one forward serving both
    training and rollout — so the term's operands have to cross the boundary.
    """
    images, state, s2_images, instructions = observation(model)
    actions, is_pad = supervision(model)
    model.train()
    try:
        _, _, (mu, log_sigma_x2) = model(images, state, s2_images, instructions,
                                         actions=actions, action_is_pad=is_pad)
    finally:
        model.eval()
    assert mu is not None and log_sigma_x2 is not None
    assert mu.shape == (1, model.config.system1.style_latent_dim)


def test_gradients_reach_system2_through_the_latent(model):
    """The architecture's central claim, end to end.

    System 1's loss must shape System 2's adapters, and `z` is the only path. If this
    fails the two halves are training independently and the "dual system" is a
    frozen feature extractor bolted to a policy.
    """
    model.zero_grad(set_to_none=True)
    images, state, s2_images, instructions = observation(model)
    actions, _, _ = model(images, state, s2_images, instructions)
    actions.sum().backward()

    lora = [p for n, p in model.system2.named_parameters()
            if p.requires_grad and "lora" in n.lower()]
    assert lora, "no LoRA parameters found"
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in lora), \
        "System 1's loss did not reach System 2"


def test_mismatched_latent_dims_are_rejected():
    """One number shared by two configs — caught at construction, not at the boundary."""
    with pytest.raises(ValueError, match="latent_dim must agree"):
        DualSystemConfig(system2=s2_tiny(latent_dim=32), system1=s1_tiny(latent_dim=64))


@pytest.mark.parametrize("period,offset", [(0, 2), (1, -1)])
def test_invalid_schedule_rejected(period, offset):
    with pytest.raises(ValueError):
        DualSystemConfig(system2=s2_tiny(latent_dim=8), system1=s1_tiny(latent_dim=8),
                         latent_update_period=period, temporal_offset=offset)


# ------------------------------------------------------------ conditioning modes


def test_zero_conditioning_produces_a_zero_latent(model):
    _, _, s2_images, instructions = observation(model)
    z = model.compute_latent(s2_images, instructions, Conditioning.ZERO)
    assert torch.count_nonzero(z) == 0


def test_static_conditioning_ignores_the_scene(model):
    """The naive baseline: `z` must depend on the instruction alone.

    If the scene leaked in, this would not be the PDF's "static embedding of the
    initial instruction" and the baseline would be measuring something else.
    """
    z_a = model.compute_latent([torch.rand(3, 128, 128)], [INSTRUCTION], Conditioning.STATIC)
    z_b = model.compute_latent([torch.rand(3, 128, 128)], [INSTRUCTION], Conditioning.STATIC)
    assert torch.allclose(z_a, z_b, atol=1e-5), "static conditioning leaked the scene"


def test_static_conditioning_still_reflects_the_instruction(model):
    """Ignoring the scene must not mean ignoring everything."""
    frame = [torch.rand(3, 128, 128)]
    z_a = model.compute_latent(frame, [INSTRUCTION], Conditioning.STATIC)
    z_b = model.compute_latent(frame, ["put both moka pots on the stove"], Conditioning.STATIC)
    assert not torch.allclose(z_a, z_b, atol=1e-5)


def test_live_conditioning_tracks_the_scene(model):
    """The property the error-recovery experiment depends on."""
    z_a = model.compute_latent([torch.rand(3, 128, 128)], [INSTRUCTION], Conditioning.LIVE)
    z_b = model.compute_latent([torch.rand(3, 128, 128)], [INSTRUCTION], Conditioning.LIVE)
    assert not torch.allclose(z_a, z_b, atol=1e-5), "live conditioning ignores the scene"


def test_scene_blind_uses_one_fixed_frame(model):
    model.reset()
    model._scene_blind_frame = None
    z_a = model.compute_latent([torch.rand(3, 128, 128)], [INSTRUCTION], Conditioning.SCENE_BLIND)
    z_b = model.compute_latent([torch.rand(3, 128, 128)], [INSTRUCTION], Conditioning.SCENE_BLIND)
    assert torch.allclose(z_a, z_b, atol=1e-5), "scene-blind latent changed with the scene"


def test_only_live_and_static_are_training_modes():
    training = {m for m in Conditioning if m.is_training_mode}
    assert training == {Conditioning.LIVE, Conditioning.STATIC}


# ------------------------------------------------------------------ rollout cadence


def test_latent_is_held_between_updates(model):
    """System 2 runs every K steps, System 1 every step — the dual-system premise."""
    model.reset()
    period = model.config.latent_update_period
    seen = []
    for _ in range(period * 2):
        images, state, s2_images, instructions = observation(model)
        _, z = model.act(images, state, s2_images, instructions, Conditioning.LIVE)
        seen.append(z.clone())

    # Within the first period the latent must not change despite new frames.
    for z in seen[1:period]:
        assert torch.allclose(seen[0], z, atol=1e-6), "latent changed inside its hold window"
    # And it must refresh once the period elapses.
    assert not torch.allclose(seen[0], seen[period], atol=1e-6), "latent never refreshed"


def test_frozen_never_refreshes(model):
    model.reset()
    seen = []
    for _ in range(model.config.latent_update_period * 3):
        images, state, s2_images, instructions = observation(model)
        _, z = model.act(images, state, s2_images, instructions, Conditioning.FROZEN)
        seen.append(z.clone())
    assert all(torch.allclose(seen[0], z, atol=1e-6) for z in seen), "frozen latent refreshed"


def test_reset_clears_episode_state(model):
    model.reset()
    images, state, s2_images, instructions = observation(model)
    model.act(images, state, s2_images, instructions, Conditioning.LIVE)
    assert model.steps_since_latent_update == 1
    model.reset()
    assert model.steps_since_latent_update == 0
    assert model._cached_latent is None


def test_act_does_not_build_a_graph(model):
    """Rollouts run under no_grad; retaining graphs across an episode would leak memory."""
    model.reset()
    images, state, s2_images, instructions = observation(model)
    actions, latent = model.act(images, state, s2_images, instructions)
    assert not actions.requires_grad and not latent.requires_grad
