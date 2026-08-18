"""System 2 — the slow semantic reasoner.

Consumes a camera frame and the task instruction, and emits a single continuous
latent vector `z` that conditions System 1. Runs every K environment steps, roughly
1-2 Hz, against System 1's per-step rate.

Design notes:

* **Base model, not the generation wrapper.** `Qwen2_5_VLModel` is loaded rather than
  `Qwen2_5_VLForConditionalGeneration`: `z` is read from hidden states, so the language
  modelling head is never used and would waste ~300M frozen parameters
  (151936 vocab x 2048 hidden) of memory.
* **No text is generated.** One forward pass, pool, project. Autoregressive decoding
  would be far too slow to sit inside a control loop and provides nothing extra —
  everything downstream needs is already in the hidden states.
* **Gradients flow back into S2.** The latent is the only channel between the systems,
  and the whole point of end-to-end training is that System 1's loss shapes what
  System 2 encodes. Nothing here detaches or runs under `no_grad`.
* `Qwen2-VL-2B-Instruct` is the alternative if S2 latency dominates the evaluation matrix.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import nn

DEFAULT_MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"


@dataclass
class System2Config:
    model_id: str = DEFAULT_MODEL_ID
    latent_dim: int = 512
    pooling: str = "mean"  # "mean" over unmasked tokens, or "last" token
    dtype: str = "bfloat16"

    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    # Attention projections of the language tower. The vision tower uses different
    # module names ("qkv", "proj") and is left frozen by default: the latent's job is
    # to summarise a scene the pretrained encoder already represents well, and keeping
    # the adapter surface small keeps two training runs on two GPUs comfortable.
    lora_target_modules: tuple[str, ...] = ("q_proj", "k_proj", "v_proj", "o_proj")

    # Set only for tests: builds a small randomly-initialised model instead of
    # downloading pretrained weights. See tiny_config().
    random_init_config: object | None = field(default=None, repr=False)

    @property
    def torch_dtype(self) -> torch.dtype:
        return getattr(torch, self.dtype)


def tiny_config(latent_dim: int = 32) -> System2Config:
    """A small randomly-initialised stand-in with the *real* tokenizer geometry.

    Vocabulary and the image/video token ids must match the pretrained processor, and
    the vision tower's ``out_hidden_size`` must equal the text hidden size — publicly
    available "tiny-random" Qwen2.5-VL checkpoints get that second constraint wrong and
    fail at the image-placeholder check, so the config is built here instead.
    """
    from transformers import Qwen2_5_VLConfig

    config = Qwen2_5_VLConfig(
        text_config={
            "hidden_size": 64,
            "intermediate_size": 128,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "vocab_size": 151936,
            # Qwen2.5-VL uses multimodal RoPE; the sections must sum to head_dim/2
            # (here 64/4 heads = 16, so 8). The pretrained 3B uses [16, 24, 24] = 64.
            "rope_parameters": {
                "type": "mrope",
                "rope_type": "default",
                "mrope_section": [2, 3, 3],
                "rope_theta": 1000000.0,
            },
        },
        vision_config={
            "hidden_size": 64,
            "intermediate_size": 128,
            "depth": 2,
            "num_heads": 4,
            "out_hidden_size": 64,  # must equal text hidden_size
            "patch_size": 14,
            "spatial_merge_size": 2,
        },
        image_token_id=151655,
        video_token_id=151656,
    )
    return System2Config(latent_dim=latent_dim, dtype="float32", random_init_config=config)


class System2(nn.Module):
    """Vision-language encoder producing one latent vector per observation."""

    def __init__(self, config: System2Config | None = None) -> None:
        super().__init__()
        self.config = config or System2Config()
        backbone, processor, hidden_size = self._build_backbone()
        self.backbone = backbone
        self.processor = processor
        self.hidden_size = hidden_size

        # Kept in float32 regardless of the backbone dtype: it is small, it is trained
        # from scratch, and it is the single path every gradient into S2 must pass
        # through, so it is not worth risking bf16 precision loss here.
        self.latent_head = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, self.config.latent_dim),
        ).to(torch.float32)

    # ------------------------------------------------------------------ construction

    def _build_backbone(self):
        from transformers import AutoProcessor, Qwen2_5_VLModel

        cfg = self.config
        if cfg.random_init_config is not None:
            backbone = Qwen2_5_VLModel(cfg.random_init_config).to(cfg.torch_dtype)
            hidden_size = cfg.random_init_config.text_config.hidden_size
        else:
            backbone = Qwen2_5_VLModel.from_pretrained(cfg.model_id, dtype=cfg.torch_dtype)
            hidden_size = backbone.config.text_config.hidden_size

        processor = AutoProcessor.from_pretrained(
            DEFAULT_MODEL_ID if cfg.random_init_config is not None else cfg.model_id
        )

        for parameter in backbone.parameters():
            parameter.requires_grad_(False)
        backbone = self._apply_lora(backbone)
        return backbone, processor, hidden_size

    def _apply_lora(self, backbone):
        from peft import LoraConfig, get_peft_model

        cfg = self.config
        if cfg.lora_r <= 0:
            return backbone
        lora = LoraConfig(
            r=cfg.lora_r,
            lora_alpha=cfg.lora_alpha,
            lora_dropout=cfg.lora_dropout,
            target_modules=list(cfg.lora_target_modules),
            bias="none",
        )
        return get_peft_model(backbone, lora)

    # ----------------------------------------------------------------------- helpers

    @staticmethod
    def _to_pil(images):
        """Accept PIL images, HWC uint8 arrays, or CHW float tensors in [0, 1].

        The dataset stores float CHW normalised frames while the environment returns
        uint8 HWC ones; both must reach the processor identically, or train and eval
        silently diverge.
        """
        from PIL import Image
        import numpy as np

        out = []
        for image in images:
            if isinstance(image, Image.Image):
                out.append(image)
                continue
            array = image.detach().cpu().numpy() if torch.is_tensor(image) else np.asarray(image)
            if array.ndim == 3 and array.shape[0] in (1, 3) and array.shape[-1] not in (1, 3):
                array = np.transpose(array, (1, 2, 0))  # CHW -> HWC
            if array.dtype != np.uint8:
                scale = 255.0 if float(array.max(initial=0.0)) <= 1.0 else 1.0
                array = np.clip(array * scale, 0, 255).astype(np.uint8)
            out.append(Image.fromarray(array))
        return out

    def _encode(self, images, instructions):
        """Tokenise one image + instruction per batch element."""
        prompts = [
            self.processor.apply_chat_template(
                [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": text}]}],
                tokenize=False,
                add_generation_prompt=True,
            )
            for text in instructions
        ]
        batch = self.processor(text=prompts, images=self._to_pil(images),
                               padding=True, return_tensors="pt")
        return {k: v.to(self.device) if torch.is_tensor(v) else v for k, v in batch.items()}

    def _pool(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        if self.config.pooling == "last":
            # Under causal attention the final position has attended to everything,
            # which makes it a natural summary — but it is also the most
            # generation-flavoured choice, hence not the default.
            lengths = attention_mask.sum(dim=1) - 1
            return hidden_states[torch.arange(hidden_states.size(0)), lengths]
        mask = attention_mask.unsqueeze(-1).to(hidden_states.dtype)
        return (hidden_states * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)

    @property
    def device(self) -> torch.device:
        return next(self.backbone.parameters()).device

    # ----------------------------------------------------------------------- forward

    def forward(self, images, instructions) -> torch.Tensor:
        """Return the latent `z`, shape ``(batch, latent_dim)``.

        Args:
            images: one frame per batch element (PIL, HWC uint8, or CHW float in [0,1]).
            instructions: one task instruction string per batch element.
        """
        if len(images) != len(instructions):
            raise ValueError(f"got {len(images)} images but {len(instructions)} instructions")

        batch = self._encode(images, instructions)
        outputs = self.backbone(**batch, output_hidden_states=True)
        pooled = self._pool(outputs.last_hidden_state, batch["attention_mask"])
        return self.latent_head(pooled.to(torch.float32))

    # ------------------------------------------------------------------- diagnostics

    def trainable_parameters(self) -> dict[str, int]:
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        return {"trainable": trainable, "total": total, "frozen": total - trainable}


if __name__ == "__main__":
    model = System2(tiny_config())
    counts = model.trainable_parameters()
    print(f"latent_dim   : {model.config.latent_dim}")
    print(f"hidden_size  : {model.hidden_size}")
    print(f"parameters   : {counts['trainable']:,} trainable / {counts['total']:,} total")

    z = model([torch.rand(3, 256, 256)], ["put both the alphabet soup and the tomato sauce in the basket"])
    print(f"latent shape : {tuple(z.shape)}")
