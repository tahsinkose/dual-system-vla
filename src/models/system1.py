"""System 1 — the fast reactive controller.

Consumes camera frames, proprioception, and System 2's latent `z`, and emits a chunk of
actions. Runs every environment step (~20 Hz), against System 2's ~1-2 Hz.

The policy itself is LeRobot's ACT (`lerobot.policies.act.modeling_act.ACT`): a
ResNet-18 backbone over each camera, a transformer encoder over the resulting tokens, a
decoder with one learned query per chunk step, and a linear action head, plus a CVAE
whose style latent is inferred from the action sequence during training and zeroed at
inference. This module owns three things ACT does not:

* **the `z` token** — System 2's latent projected into ACT's `dim_model` and inserted
  into its encoder sequence;
* **normalisation** — the dataset statistics ACT is trained against, carried as buffers
  so a checkpoint is self-contained;
* **the calling convention** — positional observations rather than a feature-keyed
  batch dictionary, which is what the rest of this repo passes around.

**There is no language input.** All instruction content reaches this model through `z`
and nowhere else; ACT has no text encoder and no language feature, so the constraint
holds structurally rather than by convention.

Two different things are called "latent" in the code below and they must not be
confused. **`z`** is System 2's semantic latent — the interface this study is about,
live at inference. **The style latent** is ACT's own CVAE variable, inferred from the
ground-truth action sequence and zeroed at inference. `z` never enters the CVAE
encoder, which reads only ``[cls, state, actions]``; a test pins that.

Wrapping the inner `ACT` module rather than `ACTPolicy` keeps the loss out of the model:
`forward` stays a function of observations, so one forward serves both training and
rollout and one checkpoint runs under every conditioning modality. The KLD term that
`ACTPolicy` would have computed belongs to the training loop instead, which is why the
CVAE distribution parameters are returned rather than consumed here.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import nn

from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.act.modeling_act import ACT
from lerobot.utils.constants import ACTION, OBS_ENV_STATE, OBS_IMAGES, OBS_STATE

# Where `z` rides into ACT.
#
# ACT assembles its 1-D encoder tokens from a fixed list — style latent, robot state,
# environment state — with no extension point. The environment-state slot is the one
# that fits: an optional input of arbitrary width, given its own `nn.Linear` projection
# and its own row of `encoder_1d_feature_pos_embed`, appended to the encoder sequence
# and *excluded from the CVAE encoder*. That last property is the reason to prefer the
# slot over a reimplementation: the isolation of `z` from the style latent is enforced
# by upstream's own control flow rather than by code here that could drift out of step
# with it.
#
# The name is upstream's and is inaccurate for this use — nothing about `z` is a
# simulator state. It is aliased once, here, and every reference below goes through the
# alias.
LATENT_SLOT = OBS_ENV_STATE

# Canonical camera key -> the dataset feature name whose statistics normalise it.
DATASET_IMAGE_FEATURES = {"image": "observation.images.image",
                          "image2": "observation.images.wrist_image"}


@dataclass
class System1Config:
    """System 1's own configuration, from which an `ACTConfig` is derived.

    Kept separate from `ACTConfig` so that the constructor surface the rest of the repo
    builds against — and therefore what strict checkpoint loading has to reproduce —
    does not track an upstream dataclass with ~30 fields, most of them irrelevant here
    (normalisation mapping, optimiser preset, inference-time action stepping).
    """

    latent_dim: int = 512          # `z`; must match System2Config.latent_dim
    state_dim: int = 8             # eef xyz, axis-angle, 2x gripper qpos
    action_dim: int = 7            # 6D eef delta + gripper
    chunk_size: int = 100          # actions predicted per forward pass

    dim_model: int = 512
    n_heads: int = 8
    dim_feedforward: int = 3200
    n_encoder_layers: int = 4
    # Upstream ships 1 against the paper's 7: only the first decoder layer was ever
    # used in the original implementation, and LeRobot matches the behaviour rather
    # than the number.
    n_decoder_layers: int = 1
    dropout: float = 0.1
    pre_norm: bool = False

    vision_backbone: str = "resnet18"
    pretrained_backbone: bool = True

    # ACT's CVAE. `style_latent_dim` is deliberately *not* called `latent_dim`.
    use_vae: bool = True
    style_latent_dim: int = 32
    n_vae_encoder_layers: int = 4

    camera_keys: tuple[str, ...] = ("image", "image2")

    def act_config(self) -> ACTConfig:
        """The upstream configuration this wraps.

        `normalization_mapping` is set to IDENTITY throughout: normalisation happens in
        `System1` against buffers it owns, so a checkpoint carries its own statistics
        instead of depending on processor files stored beside it.
        """
        input_features = {
            OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(self.state_dim,)),
            LATENT_SLOT: PolicyFeature(type=FeatureType.ENV, shape=(self.latent_dim,)),
        }
        for key in self.camera_keys:
            input_features[f"observation.images.{key}"] = PolicyFeature(
                type=FeatureType.VISUAL, shape=(3, 256, 256))

        return ACTConfig(
            chunk_size=self.chunk_size,
            n_action_steps=self.chunk_size,
            input_features=input_features,
            output_features={ACTION: PolicyFeature(type=FeatureType.ACTION,
                                                   shape=(self.action_dim,))},
            normalization_mapping={"VISUAL": NormalizationMode.IDENTITY,
                                   "STATE": NormalizationMode.IDENTITY,
                                   "ENV": NormalizationMode.IDENTITY,
                                   "ACTION": NormalizationMode.IDENTITY},
            vision_backbone=self.vision_backbone,
            pretrained_backbone_weights=("ResNet18_Weights.IMAGENET1K_V1"
                                         if self.pretrained_backbone else None),
            pre_norm=self.pre_norm,
            dim_model=self.dim_model,
            n_heads=self.n_heads,
            dim_feedforward=self.dim_feedforward,
            n_encoder_layers=self.n_encoder_layers,
            n_decoder_layers=self.n_decoder_layers,
            use_vae=self.use_vae,
            latent_dim=self.style_latent_dim,
            n_vae_encoder_layers=self.n_vae_encoder_layers,
            dropout=self.dropout,
        )


def tiny_config(**overrides) -> System1Config:
    """Small configuration for tests — no pretrained download, few layers."""
    defaults = dict(
        latent_dim=32, dim_model=64, n_heads=4, dim_feedforward=128,
        n_encoder_layers=2, n_decoder_layers=1, n_vae_encoder_layers=1,
        style_latent_dim=8, chunk_size=4, pretrained_backbone=False,
    )
    defaults.update(overrides)
    return System1Config(**defaults)


@dataclass
class NormalisationStats:
    """Per-feature mean and standard deviation, in the model's input space.

    `images` is keyed by canonical camera key and holds `(3, 1, 1)` tensors; `state`
    and `action` hold `(state_dim,)` and `(action_dim,)`.
    """

    state: tuple[torch.Tensor, torch.Tensor]
    action: tuple[torch.Tensor, torch.Tensor]
    images: dict[str, tuple[torch.Tensor, torch.Tensor]] = field(default_factory=dict)

    @classmethod
    def from_dataset(cls, dataset_meta, config: System1Config) -> "NormalisationStats":
        """Read the statistics LeRobot computed over the dataset.

        These are the same numbers the policy's processor files would hold; taking them
        from the dataset metadata rather than from a sidecar keeps train and eval on one
        source.
        """
        stats = dataset_meta.stats

        def pair(key):
            entry = stats[key]
            return (torch.as_tensor(entry["mean"], dtype=torch.float32),
                    torch.as_tensor(entry["std"], dtype=torch.float32))

        return cls(
            state=pair(OBS_STATE),
            action=pair(ACTION),
            images={key: pair(DATASET_IMAGE_FEATURES[key]) for key in config.camera_keys},
        )

    @classmethod
    def identity(cls, config: System1Config) -> "NormalisationStats":
        """Statistics that leave every input untouched, for tests and bring-up."""
        return cls(
            state=(torch.zeros(config.state_dim), torch.ones(config.state_dim)),
            action=(torch.zeros(config.action_dim), torch.ones(config.action_dim)),
            images={key: (torch.zeros(3, 1, 1), torch.ones(3, 1, 1))
                    for key in config.camera_keys},
        )


# Guards against dividing by a statistic that is zero because a dimension never varies.
_STD_FLOOR = 1e-6


class System1(nn.Module):
    """Latent-conditioned action-chunk policy. No language input."""

    def __init__(self, config: System1Config | None = None,
                 stats: NormalisationStats | None = None) -> None:
        super().__init__()
        self.config = config or System1Config()
        cfg = self.config
        self.act = ACT(cfg.act_config())

        stats = stats or NormalisationStats.identity(cfg)
        self._register_stat("state", *stats.state)
        self._register_stat("action", *stats.action)
        for key in cfg.camera_keys:
            self._register_stat(f"image_{key}", *stats.images[key])

    def _register_stat(self, name: str, mean: torch.Tensor, std: torch.Tensor) -> None:
        """Statistics are persistent buffers: a checkpoint carries its own.

        Loading weights trained under one normalisation and applying another silently
        shifts every input and every action the policy emits, with nothing to raise.
        """
        self.register_buffer(f"{name}_mean", mean.clone().float())
        self.register_buffer(f"{name}_std", std.clone().float().clamp_min(_STD_FLOOR))

    # ----------------------------------------------------------------- normalisation

    def normalise_action(self, actions: torch.Tensor) -> torch.Tensor:
        """Raw actions -> the space the model predicts in."""
        return (actions - self.action_mean) / self.action_std

    def unnormalise_action(self, actions: torch.Tensor) -> torch.Tensor:
        """Model output -> the space the environment accepts."""
        return actions * self.action_std + self.action_mean

    # ----------------------------------------------------------------------- forward

    def forward(
        self,
        images: dict[str, torch.Tensor],
        state: torch.Tensor,
        latent: torch.Tensor,
        actions: torch.Tensor | None = None,
        action_is_pad: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor | None, torch.Tensor | None]]:
        """Predict an action chunk.

        Args:
            images: camera key -> ``(B, 3, H, W)`` float tensors in [0, 1]. Keys must
                cover ``config.camera_keys``.
            state: ``(B, state_dim)`` proprioception, unnormalised.
            latent: ``(B, latent_dim)`` — System 2's `z`.
            actions: ``(B, chunk_size, action_dim)`` ground truth, unnormalised.
                Required in training mode when the CVAE is enabled — it is the CVAE
                encoder's input — and ignored at inference, where the style latent is
                zeroed.
            action_is_pad: ``(B, chunk_size)`` bool, marking chunk steps that ran past
                the end of their episode. Required alongside `actions`: it is the CVAE
                encoder's key-padding mask, and without it the style latent is inferred
                partly from padding.

        Returns:
            ``(actions, (mu, log_sigma_x2))`` — the chunk **in normalised action
            space**, and the style latent's distribution parameters, both `None`
            outside CVAE training. The caller owns the loss, including its KLD term.

        Note the absence of any text argument: see the module docstring.
        """
        cfg = self.config
        missing = [k for k in cfg.camera_keys if k not in images]
        if missing:
            raise KeyError(f"missing camera views {missing}; got {sorted(images)}")
        if latent.shape[-1] != cfg.latent_dim:
            raise ValueError(
                f"latent has width {latent.shape[-1]}, expected {cfg.latent_dim} — "
                "System 1 and System 2 must agree on latent_dim")

        batch = {
            OBS_IMAGES: [(images[key] - getattr(self, f"image_{key}_mean"))
                         / getattr(self, f"image_{key}_std") for key in cfg.camera_keys],
            OBS_STATE: (state - self.state_mean) / self.state_std,
            LATENT_SLOT: latent,
        }
        if actions is None:
            if cfg.use_vae and self.training:
                raise ValueError(
                    "the CVAE encoder needs the ground-truth action chunk in training "
                    "mode; pass `actions` and `action_is_pad`, or call `.eval()` first "
                    "if this is an inference pass")
        else:
            if action_is_pad is None:
                raise ValueError("action_is_pad is required alongside actions: it masks "
                                 "padded steps out of the CVAE encoder")
            batch[ACTION] = self.normalise_action(actions)
            batch["action_is_pad"] = action_is_pad
        return self.act(batch)

    # ------------------------------------------------------------------- diagnostics

    def parameter_counts(self) -> dict[str, int]:
        """Sizes by group.

        `inference` excludes the CVAE encoder, which runs only during training — the
        number that describes the deployed controller.
        """
        groups = {
            "backbone": self.act.backbone,
            "encoder": self.act.encoder,
            "decoder": self.act.decoder,
        }
        counts = {name: sum(p.numel() for p in module.parameters())
                  for name, module in groups.items()}
        counts["total"] = sum(p.numel() for p in self.parameters())
        vae = sum(p.numel() for name, p in self.named_parameters() if "vae_encoder" in name)
        counts["vae_encoder"] = vae
        counts["other"] = counts["total"] - sum(counts[k] for k in groups) - vae
        counts["inference"] = counts["total"] - vae
        return counts


if __name__ == "__main__":
    model = System1(System1Config(pretrained_backbone=False))
    cfg = model.config
    counts = model.parameter_counts()
    for name in ("backbone", "encoder", "decoder", "vae_encoder", "other", "total", "inference"):
        print(f"{name:12s} {counts[name] / 1e6:6.1f}M")

    images = {key: torch.rand(2, 3, 256, 256) for key in cfg.camera_keys}
    model.eval()
    actions, (mu, _) = model(images, torch.rand(2, cfg.state_dim),
                             torch.rand(2, cfg.latent_dim))
    print(f"\naction chunk: {tuple(actions.shape)}  style latent: {mu}")
