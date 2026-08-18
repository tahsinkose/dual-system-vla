"""Tests for System 2.

Uses a small randomly-initialised backbone with the real tokenizer geometry, so the
whole code path — chat template, image placeholders, pooling, latent head — is
exercised without downloading multi-GB weights.

Run with::

    python -m pytest tests/test_system2.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models.system2 import System2, System2Config, tiny_config  # noqa: E402

TASK_5_INSTRUCTION = "put both the alphabet soup and the tomato sauce in the basket"
TASK_7_INSTRUCTION = "put both the cream cheese box and the butter in the basket"
INSTRUCTION = TASK_5_INSTRUCTION


@pytest.fixture(scope="module")
def model() -> System2:
    return System2(tiny_config(latent_dim=32))


def test_latent_shape(model):
    z = model([torch.rand(3, 256, 256)], [INSTRUCTION])
    assert z.shape == (1, 32)
    assert z.dtype == torch.float32


def test_batches(model):
    images = [torch.rand(3, 256, 256) for _ in range(3)]
    z = model(images, [INSTRUCTION, "put both moka pots on the stove", "open the drawer"])
    assert z.shape == (3, 32)


def test_mismatched_inputs_raise(model):
    with pytest.raises(ValueError, match="images but"):
        model([torch.rand(3, 256, 256)], [INSTRUCTION, INSTRUCTION])


def test_backbone_frozen_and_lora_trainable(model):
    """Only LoRA adapters and the latent head may receive gradients."""
    trainable = [n for n, p in model.named_parameters() if p.requires_grad]
    assert trainable, "nothing is trainable"
    for name in trainable:
        assert "lora" in name.lower() or "latent_head" in name, f"unexpectedly trainable: {name}"

    counts = model.trainable_parameters()
    assert counts["trainable"] < counts["total"] * 0.05, "far too much of the backbone is trainable"


def test_gradients_reach_the_lora_adapters(model):
    """The whole point of end-to-end training: System 1's loss must shape System 2.

    If the latent were detached, or the backbone ran under no_grad, the latent head
    would still train and the loss would still fall — but System 2 would be a frozen
    feature extractor and the architecture's central claim would be false. This test
    is what makes that failure visible.
    """
    model.zero_grad(set_to_none=True)
    z = model([torch.rand(3, 256, 256)], [INSTRUCTION])
    z.sum().backward()

    lora = {n: p for n, p in model.named_parameters() if p.requires_grad and "lora" in n.lower()}
    assert lora, "no LoRA parameters found"
    with_grad = [n for n, p in lora.items() if p.grad is not None and p.grad.abs().sum() > 0]
    assert with_grad, "gradients did not reach any LoRA adapter — S2 is effectively frozen"

    head = dict(model.latent_head.named_parameters())
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in head.values())


def test_latent_varies_with_the_scene(model):
    """A latent that ignores its inputs would make every ablation modality identical."""
    torch.manual_seed(0)
    a = model([torch.rand(3, 256, 256)], [INSTRUCTION])
    b = model([torch.rand(3, 256, 256)], [INSTRUCTION])
    assert not torch.allclose(a, b, atol=1e-4), "latent is constant across different images"


def test_latent_varies_with_the_instruction(model):
    """Tasks 5 and 7 share a scene, so instruction sensitivity is what separates them."""
    torch.manual_seed(0)
    image = torch.rand(3, 256, 256)
    a = model([image], [TASK_5_INSTRUCTION])
    b = model([image], [TASK_7_INSTRUCTION])
    assert not torch.allclose(a, b, atol=1e-4), "latent ignores the instruction"


@pytest.mark.parametrize("pooling", ["mean", "last"])
def test_pooling_modes(pooling):
    config = tiny_config(latent_dim=16)
    config.pooling = pooling
    z = System2(config)([torch.rand(3, 256, 256)], [INSTRUCTION])
    assert z.shape == (1, 16)


def test_accepts_env_and_dataset_image_formats(model):
    """The env yields uint8 HWC 0-255; the dataset yields float CHW 0-1.

    Both must land on the same representation or train and eval diverge silently.
    """
    import numpy as np

    torch.manual_seed(0)
    chw_float = torch.rand(3, 64, 64)
    hwc_uint8 = (chw_float.permute(1, 2, 0).numpy() * 255).astype(np.uint8)

    a = model([chw_float], [INSTRUCTION])
    b = model([hwc_uint8], [INSTRUCTION])
    # Not bit-identical: uint8 quantises the float frame. Close is the requirement.
    assert torch.allclose(a, b, atol=1e-2), "dataset and env image formats disagree"
