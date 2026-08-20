"""System 1, built from scratch — the alternative to wrapping LeRobot's ACT.

Selected with ``--system1-arch scratch``; ``src/models/system1.py`` holds the ACT-backed
default. Both satisfy the same contract, so `DualSystem` and the training loop do not
branch on which is in use.

Two differences from the ACT path are load-bearing rather than incidental. There is no
CVAE, so `forward` returns no style-latent parameters and the KLD term of the loss is
inert. And actions are consumed and emitted unnormalised, so the normalisation
statistics an ACT checkpoint carries as buffers have no counterpart here.

Consumes camera frames, proprioception, and System 2's latent `z`, and emits a chunk of
actions. Runs every environment step (~20 Hz), against System 2's ~1-2 Hz.

Structure, following the DETR/ACT pattern:

* a **shared convolutional backbone** over both camera views, taking feature maps from
  several stages so the encoder sees multiple spatial scales;
* an **encoder** over the concatenation of visual tokens, a proprioception token, and
  the projected latent tokens;
* a **decoder** with one learned query per action-chunk step, cross-attending to that
  memory;
* a linear head mapping each query to a 7-dim action.

The backbone is shared across cameras rather than duplicated: it halves the parameter
count and the two LIBERO views are similar enough in statistics that separate weights
buy little.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18
from torchvision.ops.misc import FrozenBatchNorm2d

# ImageNet statistics from torchvision, required because the backbone is ImageNet-initialised.
IMAGENET_MEAN = ResNet18_Weights.IMAGENET1K_V1.transforms().mean
IMAGENET_STD = ResNet18_Weights.IMAGENET1K_V1.transforms().std

# Output channels of the ResNet-18 stages this model reads from.
RESNET18_STAGE_CHANNELS = {"layer2": 128, "layer3": 256, "layer4": 512}


@dataclass
class ScratchSystem1Config:
    latent_dim: int = 512          # must match System2Config.latent_dim
    state_dim: int = 8             # eef xyz, axis-angle, 2x gripper qpos
    action_dim: int = 7            # 6D eef delta + gripper
    chunk_size: int = 16           # actions predicted per forward pass

    d_model: int = 512
    n_heads: int = 8
    n_encoder_layers: int = 8
    n_decoder_layers: int = 8
    dim_feedforward: int = 2048
    dropout: float = 0.1

    # Which backbone stages feed the encoder. Later stages are semantically richer but
    # spatially coarser; including earlier ones preserves the fine detail that precise
    # manipulation needs.
    backbone_stages: tuple[str, ...] = ("layer3", "layer4")
    pretrained_backbone: bool = True

    # How many tokens the latent expands into. One is enough for a global conditioning
    # signal; more gives the decoder somewhere to attend differentially.
    n_latent_tokens: int = 1

    camera_keys: tuple[str, ...] = ("image", "image2")

    def __post_init__(self) -> None:
        for stage in self.backbone_stages:
            if stage not in RESNET18_STAGE_CHANNELS:
                raise ValueError(f"unknown backbone stage {stage!r}; "
                                 f"expected one of {sorted(RESNET18_STAGE_CHANNELS)}")


def tiny_config(**overrides) -> ScratchSystem1Config:
    """Small configuration for tests — no pretrained download, few layers."""
    defaults = dict(
        latent_dim=32, d_model=64, n_heads=4, n_encoder_layers=2, n_decoder_layers=2,
        dim_feedforward=128, chunk_size=4, pretrained_backbone=False,
    )
    defaults.update(overrides)
    return ScratchSystem1Config(**defaults)


class MultiScaleBackbone(nn.Module):
    """ResNet-18 trunk exposing several stages as flat token sequences."""

    def __init__(self, config: ScratchSystem1Config) -> None:
        super().__init__()
        weights = ResNet18_Weights.IMAGENET1K_V1 if config.pretrained_backbone else None
        # FrozenBatchNorm keeps the ImageNet statistics fixed rather than re-estimating
        # them — what ACT and DETR do, for three reasons that all apply here:
        #   * BatchNorm is one of the few layers that behaves *differently* in train and
        #     eval mode (batch statistics vs accumulated running ones). Drift between the
        #     two shows up as a policy that evaluates worse than it trained, with nothing
        #     obvious to point at.
        #   * Rollouts step one environment at a time, so batch statistics would be
        #     computed over a single sample.
        #   * ImageNet statistics are a better prior than ones re-estimated from a few
        #     hundred episodes of narrow robot data at modest batch sizes.
        resnet = resnet18(weights=weights, norm_layer=FrozenBatchNorm2d)

        self.stem = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        self.layer1, self.layer2 = resnet.layer1, resnet.layer2
        self.layer3, self.layer4 = resnet.layer3, resnet.layer4
        self.stages = config.backbone_stages

        # One 1x1 projection per stage, since each has a different channel count.
        self.projections = nn.ModuleDict({
            stage: nn.Conv2d(RESNET18_STAGE_CHANNELS[stage], config.d_model, kernel_size=1)
            for stage in self.stages
        })

        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1), persistent=False)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """``(B, 3, H, W)`` in [0, 1] -> ``(B, tokens, d_model)``."""
        x = (images - self.mean) / self.std
        x = self.layer1(self.stem(x))
        outputs = {}
        x = self.layer2(x)
        outputs["layer2"] = x
        x = self.layer3(x)
        outputs["layer3"] = x
        x = self.layer4(x)
        outputs["layer4"] = x

        tokens = []
        for stage in self.stages:
            feature = self.projections[stage](outputs[stage])       # B, d_model, h, w
            tokens.append(feature.flatten(2).transpose(1, 2))       # B, h*w, d_model
        return torch.cat(tokens, dim=1)


class ScratchSystem1(nn.Module):
    """Latent-conditioned action-chunk policy. No language input."""

    def __init__(self, config: ScratchSystem1Config | None = None, stats=None) -> None:
        super().__init__()
        # `stats` is accepted and ignored so the two architectures are constructed
        # identically. This one is trained on raw actions, so there is nothing to
        # normalise against and nothing for a checkpoint to carry.
        del stats
        self.config = config or ScratchSystem1Config()
        cfg = self.config

        self.backbone = MultiScaleBackbone(cfg)
        self.state_projection = nn.Linear(cfg.state_dim, cfg.d_model)
        self.latent_projection = nn.Linear(cfg.latent_dim, cfg.d_model * cfg.n_latent_tokens)

        # Learned embeddings marking token provenance (camera i / state / latent), so
        # the encoder can tell an image patch from proprioception after concatenation.
        self.token_type_embedding = nn.Embedding(len(cfg.camera_keys) + 2, cfg.d_model)
        self.position_embedding = nn.Parameter(torch.zeros(1, 2048, cfg.d_model))
        nn.init.trunc_normal_(self.position_embedding, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model, nhead=cfg.n_heads, dim_feedforward=cfg.dim_feedforward,
            dropout=cfg.dropout, batch_first=True, norm_first=True,
        )
        # enable_nested_tensor is incompatible with norm_first and only warns; disabled
        # explicitly since every sequence here is dense anyway (no padding mask).
        self.encoder = nn.TransformerEncoder(encoder_layer, cfg.n_encoder_layers,
                                             enable_nested_tensor=False)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=cfg.d_model, nhead=cfg.n_heads, dim_feedforward=cfg.dim_feedforward,
            dropout=cfg.dropout, batch_first=True, norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, cfg.n_decoder_layers)

        # One query per timestep of the predicted chunk.
        self.action_queries = nn.Parameter(torch.zeros(1, cfg.chunk_size, cfg.d_model))
        nn.init.trunc_normal_(self.action_queries, std=0.02)

        # Left unsquashed: the loss is a regression onto recorded actions that already
        # lie in [-1, 1], and a tanh here would flatten gradients near the range limits
        # exactly where the gripper commands live. Clamping belongs at the env boundary.
        self.action_head = nn.Linear(cfg.d_model, cfg.action_dim)

    # ----------------------------------------------------------------------- forward

    def normalise_action(self, actions: torch.Tensor) -> torch.Tensor:
        """Identity: this architecture is trained on raw actions."""
        return actions

    def unnormalise_action(self, actions: torch.Tensor) -> torch.Tensor:
        """Identity, mirroring `normalise_action`."""
        return actions

    def forward(
        self,
        images: dict[str, torch.Tensor],
        state: torch.Tensor,
        latent: torch.Tensor,
        actions: torch.Tensor | None = None,
        action_is_pad: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, tuple[None, None]]:
        """Predict an action chunk.

        Args:
            images: camera key -> ``(B, 3, H, W)`` float tensors in [0, 1]. Keys must
                cover ``config.camera_keys``.
            state: ``(B, state_dim)`` proprioception.
            latent: ``(B, latent_dim)`` — System 2's output.
            actions, action_is_pad: accepted and ignored. They feed the ACT path's CVAE
                encoder; there is no CVAE here.

        Returns:
            ``((B, chunk_size, action_dim), (None, None))``. The second element is where
            the ACT path returns its style-latent parameters; None makes the KLD term of
            the loss inert without the caller having to know which architecture it holds.

        Note the absence of any text argument: see the module docstring.
        """
        cfg = self.config
        missing = [k for k in cfg.camera_keys if k not in images]
        if missing:
            raise KeyError(f"missing camera views {missing}; got {sorted(images)}")

        batch_size = state.shape[0]
        tokens, type_ids = [], []

        for index, key in enumerate(cfg.camera_keys):
            visual = self.backbone(images[key])
            tokens.append(visual)
            type_ids.append(torch.full((visual.shape[1],), index, device=visual.device))

        state_token = self.state_projection(state).unsqueeze(1)
        tokens.append(state_token)
        type_ids.append(torch.full((1,), len(cfg.camera_keys), device=state.device))

        latent_tokens = self.latent_projection(latent).view(
            batch_size, cfg.n_latent_tokens, cfg.d_model
        )
        tokens.append(latent_tokens)
        type_ids.append(torch.full((cfg.n_latent_tokens,), len(cfg.camera_keys) + 1,
                                   device=latent.device))

        sequence = torch.cat(tokens, dim=1)
        length = sequence.shape[1]
        if length > self.position_embedding.shape[1]:
            raise ValueError(
                f"sequence of {length} tokens exceeds the {self.position_embedding.shape[1]} "
                "positional embeddings; reduce image resolution or backbone stages"
            )
        sequence = (sequence
                    + self.position_embedding[:, :length]
                    + self.token_type_embedding(torch.cat(type_ids).long()).unsqueeze(0))

        memory = self.encoder(sequence)
        del actions, action_is_pad
        queries = self.action_queries.expand(batch_size, -1, -1)
        decoded = self.decoder(queries, memory)
        return self.action_head(decoded), (None, None)

    # ------------------------------------------------------------------- diagnostics

    def parameter_counts(self) -> dict[str, int]:
        groups = {
            "backbone": self.backbone,
            "encoder": self.encoder,
            "decoder": self.decoder,
        }
        counts = {name: sum(p.numel() for p in module.parameters())
                  for name, module in groups.items()}
        counts["total"] = sum(p.numel() for p in self.parameters())
        counts["other"] = counts["total"] - sum(counts[k] for k in groups)
        return counts


if __name__ == "__main__":
    model = ScratchSystem1(ScratchSystem1Config(pretrained_backbone=False))
    cfg = model.config
    counts = model.parameter_counts()
    for name in ("backbone", "encoder", "decoder", "other", "total"):
        print(f"{name:9s} {counts[name] / 1e6:6.1f}M")

    images = {key: torch.rand(2, 3, 256, 256) for key in cfg.camera_keys}
    actions, _ = model(images, torch.rand(2, cfg.state_dim), torch.rand(2, cfg.latent_dim))
    print(f"\naction chunk: {tuple(actions.shape)}")
