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

from src.models.system1 import (  # noqa: E402
    NormalisationStats,
    System1,
    System1Config,
    tiny_config,
)


@pytest.fixture
def model() -> System1:
    """Inference mode by default; tests that exercise the CVAE opt into `.train()`.

    Mode is not incidental here — the CVAE encoder runs only in training mode, so a
    model left in the wrong one changes what `forward` requires of its caller.
    """
    return System1(tiny_config()).eval()


def observation(model: System1, batch: int = 2, size: int = 64):
    cfg = model.config
    return (
        {key: torch.rand(batch, 3, size, size) for key in cfg.camera_keys},
        torch.rand(batch, cfg.state_dim),
        torch.rand(batch, cfg.latent_dim),
    )


def supervision(model: System1, batch: int = 2):
    cfg = model.config
    return (torch.rand(batch, cfg.chunk_size, cfg.action_dim),
            torch.zeros(batch, cfg.chunk_size, dtype=torch.bool))


def test_action_chunk_shape(model):
    images, state, latent = observation(model)
    actions, _ = model(images, state, latent)
    assert actions.shape == (2, model.config.chunk_size, model.config.action_dim)


def test_has_no_language_input(model):
    """The architecture's central constraint, enforced at the signature level.

    If System 1 could read the instruction, it would not need the latent to identify
    the task, and the counterfactual the whole study rests on would be unprovable.
    """
    parameters = set(inspect.signature(System1.forward).parameters)
    assert parameters == {"self", "images", "state", "latent", "actions", "action_is_pad"}
    forbidden = {"text", "instruction", "language", "task", "prompt", "tokens"}
    assert not (parameters & forbidden)


def test_latent_changes_the_action(model):
    """`z` must actually drive the output.

    A model that ignores the latent would still train and still reach some success
    rate — while making every ablation modality produce identical numbers. That failure
    is invisible in the loss curve, so it is asserted here.

    The latent under test is System 2's `z`, not ACT's style latent: the two occupy
    adjacent encoder tokens, and an assertion satisfied by the wrong one would pass
    while `z` was ignored.
    """
    torch.manual_seed(0)
    model.eval()
    images, state, latent = observation(model)
    with torch.no_grad():
        a, _ = model(images, state, latent)
        b, _ = model(images, state, torch.rand_like(latent))
    assert not torch.allclose(a, b, atol=1e-5), "System 1 ignores the latent"


def test_gradients_reach_the_latent(model):
    """Gradients must flow back through `z`, or System 2 never learns.

    End-to-end training depends on this path existing: it is how System 1's loss
    reaches System 2's LoRA adapters.
    """
    images, state, latent = observation(model)
    latent = latent.clone().requires_grad_(True)
    actions, _ = model(images, state, latent)
    actions.sum().backward()
    assert latent.grad is not None and latent.grad.abs().sum() > 0


def test_latent_does_not_reach_the_style_latent(model):
    """`z` must be invisible to the CVAE encoder.

    ACT's style latent is inferred from the action sequence and zeroed at inference. If
    `z` also fed it, part of what `z` contributes during training would vanish at
    rollout — and the ablation would be measuring a latent whose training-time role it
    cannot reproduce. The encoder-token slot `z` occupies is outside the CVAE encoder's
    input, so this holds by construction; the test pins it against that changing.
    """
    model.train()
    images, state, latent = observation(model)
    other = torch.rand_like(latent)
    actions, is_pad = supervision(model)

    # Dropout is active in training mode, so both passes must start from the same RNG
    # state; the alternative latent is drawn beforehand rather than inline.
    torch.manual_seed(0)
    _, (mu, log_sigma_x2) = model(images, state, latent, actions, is_pad)
    torch.manual_seed(0)
    _, (other_mu, other_log_sigma_x2) = model(images, state, other, actions, is_pad)

    torch.testing.assert_close(mu, other_mu)
    torch.testing.assert_close(log_sigma_x2, other_log_sigma_x2)


def test_style_latent_is_inferred_in_training_and_zeroed_at_inference(model):
    """The CVAE's training/inference asymmetry, which the ablation depends on."""
    images, state, latent = observation(model)
    actions, is_pad = supervision(model)

    model.train()
    _, (mu, log_sigma_x2) = model(images, state, latent, actions, is_pad)
    assert mu is not None and log_sigma_x2 is not None
    assert mu.shape == (2, model.config.style_latent_dim)

    model.eval()
    with torch.no_grad():
        _, params = model(images, state, latent)
    assert params == (None, None)


def test_actions_require_a_padding_mask(model):
    """Padded chunk steps must be masked out of the CVAE encoder.

    Without the mask the style latent is inferred partly from padding — silently, since
    the shapes are valid either way.
    """
    model.train()
    images, state, latent = observation(model)
    actions, _ = supervision(model)
    with pytest.raises(ValueError, match="action_is_pad is required"):
        model(images, state, latent, actions)


def test_training_without_actions_says_what_is_missing(model):
    """The CVAE encoder's requirement, stated rather than assert-ed from inside ACT."""
    model.train()
    images, state, latent = observation(model)
    with pytest.raises(ValueError, match="CVAE encoder needs the ground-truth"):
        model(images, state, latent)


def test_missing_camera_raises(model):
    images, state, latent = observation(model)
    del images[model.config.camera_keys[-1]]
    with pytest.raises(KeyError, match="missing camera views"):
        model(images, state, latent)


def test_latent_dim_must_match_system2(model):
    """The two configs share one number; a mismatch should fail loudly at the boundary."""
    images, state, _ = observation(model)
    with pytest.raises(ValueError, match="System 1 and System 2 must agree"):
        model(images, state, torch.rand(2, model.config.latent_dim + 32))


def test_default_size_is_in_the_intended_range():
    """The deployed controller is what the size claim is about.

    `total` includes the CVAE encoder, which runs only during training; `inference` is
    the number that describes System 1 at rollout.
    """
    counts = System1(System1Config(pretrained_backbone=False)).parameter_counts()
    assert counts["inference"] < counts["total"]
    assert counts["vae_encoder"] > 0
    assert 25e6 < counts["inference"] < 45e6, \
        f"unexpected size: {counts['inference'] / 1e6:.1f}M at inference"


def test_backbone_normalisation_is_frozen():
    """BatchNorm statistics must not drift between training and rollout.

    BatchNorm is one of the few layers that behaves *differently* in the two modes:
    batch statistics while training, accumulated running statistics at evaluation. If
    they diverge the policy silently evaluates worse than it trained. Rollouts also
    step a single environment, so batch statistics would be computed over one sample.
    """
    from torchvision.ops.misc import FrozenBatchNorm2d

    backbone = System1(tiny_config()).act.backbone
    norms = [m for m in backbone.modules()
             if isinstance(m, (torch.nn.BatchNorm2d, FrozenBatchNorm2d))]
    assert norms, "no normalisation layers found in the backbone"
    assert all(isinstance(m, FrozenBatchNorm2d) for m in norms), \
        "backbone still uses trainable BatchNorm"


def test_backbone_output_is_mode_independent():
    """The observable consequence: identical features in train and eval mode."""
    model = System1(tiny_config())
    frame = torch.rand(4, 3, 64, 64)

    model.train()
    with torch.no_grad():
        training_features = model.act.backbone(frame)["feature_map"]
    model.eval()
    with torch.no_grad():
        eval_features = model.act.backbone(frame)["feature_map"]

    torch.testing.assert_close(training_features, eval_features)


# ------------------------------------------------------------------- normalisation


def test_normalisation_statistics_are_persistent():
    """A checkpoint must carry its own statistics.

    Restoring weights trained under one normalisation and evaluating under another
    shifts every input and every emitted action, with nothing to raise.
    """
    config = tiny_config()
    stats = NormalisationStats.identity(config)
    stats.action = (torch.full((config.action_dim,), 0.25),
                    torch.full((config.action_dim,), 2.0))
    saved = System1(config, stats).state_dict()

    assert "action_mean" in saved and "action_std" in saved
    restored = System1(config)
    restored.load_state_dict(saved, strict=True)
    torch.testing.assert_close(restored.action_mean, stats.action[0])
    torch.testing.assert_close(restored.action_std, stats.action[1])


def test_action_normalisation_round_trips():
    config = tiny_config()
    stats = NormalisationStats.identity(config)
    stats.action = (torch.linspace(-1, 1, config.action_dim),
                    torch.linspace(0.5, 2.0, config.action_dim))
    model = System1(config, stats)

    raw = torch.rand(3, config.chunk_size, config.action_dim) * 2 - 1
    torch.testing.assert_close(model.unnormalise_action(model.normalise_action(raw)), raw)


def test_zero_variance_statistic_does_not_divide_by_zero():
    """A dimension that never varies in the dataset has std 0."""
    config = tiny_config()
    stats = NormalisationStats.identity(config)
    stats.state = (torch.zeros(config.state_dim), torch.zeros(config.state_dim))
    model = System1(config, stats).eval()

    images, state, latent = observation(model)
    actions, _ = model(images, state, latent)
    assert torch.isfinite(actions).all()


def test_predictions_are_in_normalised_action_space():
    """`forward` predicts in the space the loss is computed in, not the env's.

    ACT is trained against normalised targets; returning unnormalised actions here
    would make the training objective differ from the verified recipe by a per-dimension
    reweighting — the rotation deltas have roughly a tenth the spread of the gripper
    command, so the two spaces weight them an order of magnitude apart.
    """
    config = tiny_config()
    stats = NormalisationStats.identity(config)
    stats.action = (torch.full((config.action_dim,), 5.0),
                    torch.full((config.action_dim,), 0.1))
    model = System1(config, stats)
    model.eval()

    images, state, latent = observation(model)
    with torch.no_grad():
        actions, _ = model(images, state, latent)
    # Unnormalised outputs would sit near the mean of 5.0; normalised ones near 0.
    assert actions.abs().mean() < 1.0
