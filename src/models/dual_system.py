"""The dual system — System 2 conditioning System 1 through a latent vector.

This module owns what neither half owns alone:

* **composition** — one end-to-end forward, so System 1's regression loss reaches
  System 2's LoRA adapters through `z` and nothing else;
* **the conditioning mode** — how `z` is produced. This is the axis the whole ablation
  varies, so it lives in one place rather than being scattered through the eval loop;
* **the temporal offset Δ** — System 2 reads a frame Δ steps stale;
* **the update period K** — during a rollout `z` is recomputed every K steps and held
  constant in between.

The projection into System 1's token space and the sequence-dim concatenation are not
here: System 1 owns them, since only it knows its own token layout.

Why Δ and K exist
-----------------
At inference `z` is inherently stale: System 2 runs at ~1-2 Hz while System 1 runs
every step, so by the time a latent arrives the world has moved on. Training on
perfectly fresh latents would therefore train a model that never sees its deployment
conditions. Feeding System 2 the frame from Δ steps ago reproduces that staleness at
training time.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from enum import Enum

import torch
from torch import nn

from src.models.system1 import System1, System1Config
from src.models.system1_scratch import ScratchSystem1, ScratchSystem1Config
from src.models.system2 import System2, System2Config


class Conditioning(str, Enum):
    """How `z` is produced. Each value is one row of the ablation table.

    LIVE and STATIC are also *training* modes — they produce the two checkpoints. The
    rest are test-time interventions applied to an already-trained checkpoint.
    """

    LIVE = "live"                # recomputed from the current frame every K steps
    STATIC = "static"            # instruction only, no scene — the naive baseline
    FROZEN = "frozen"            # computed once at t=0 from the initial frame, held
    ZERO = "zero"                # z := 0, what System 1 manages on visual priors alone
    SCENE_BLIND = "scene_blind"  # from a fixed unrelated frame; isolates scene grounding

    @property
    def is_training_mode(self) -> bool:
        return self in (Conditioning.LIVE, Conditioning.STATIC)


# Mid-grey placeholder used by STATIC: a constant image makes `z` a function of the
# instruction alone, which is exactly "a static embedding of the initial instruction".
# Grey rather than black so the vision tower sees a valid, unsaturated input.
PLACEHOLDER_PIXEL = 0.5


# Which System 1 implementation to build. Both satisfy one contract, so nothing
# downstream branches on the choice; see src/models/system1_scratch.py for how they
# differ. Recorded in the checkpoint, so evaluation rebuilds what training used.
SYSTEM1_ARCHS = ("act", "scratch")


@dataclass
class DualSystemConfig:
    system2: System2Config = field(default_factory=System2Config)
    system1: ScratchSystem1Config | System1Config = field(
        default_factory=ScratchSystem1Config)
    system1_arch: str = "scratch"

    conditioning: Conditioning = Conditioning.LIVE
    latent_update_period: int = 10   # K, in environment steps
    temporal_offset: int = 2         # Δ, in environment steps

    def __post_init__(self) -> None:
        if isinstance(self.conditioning, str):
            self.conditioning = Conditioning(self.conditioning)
        if self.system1_arch not in SYSTEM1_ARCHS:
            raise ValueError(f"system1_arch must be one of {SYSTEM1_ARCHS}, "
                             f"got {self.system1_arch!r}")
        expected = System1Config if self.system1_arch == "act" else ScratchSystem1Config
        if not isinstance(self.system1, expected):
            raise TypeError(f"system1_arch={self.system1_arch!r} needs a "
                            f"{expected.__name__}, got {type(self.system1).__name__}")
        if self.system1.latent_dim != self.system2.latent_dim:
            raise ValueError(
                "latent_dim must agree across the two systems: "
                f"System2={self.system2.latent_dim}, System1={self.system1.latent_dim}"
            )
        if self.latent_update_period < 1:
            raise ValueError("latent_update_period must be >= 1")
        if self.temporal_offset < 0:
            raise ValueError("temporal_offset must be >= 0")


def tiny_config(**overrides) -> DualSystemConfig:
    """Small end-to-end configuration for tests, for either System 1 architecture.
    """
    from src.models.system1 import tiny_config as act_tiny
    from src.models.system1_scratch import tiny_config as scratch_tiny
    from src.models.system2 import tiny_config as s2_tiny

    latent_dim = overrides.pop("latent_dim", 32)
    arch = overrides.pop("system1_arch", "scratch")
    s1_tiny = act_tiny if arch == "act" else scratch_tiny

    dual_fields = {f.name for f in fields(DualSystemConfig)}
    dual = {k: v for k, v in overrides.items() if k in dual_fields}
    system1 = {k: v for k, v in overrides.items() if k not in dual_fields}

    return DualSystemConfig(
        system2=s2_tiny(latent_dim=latent_dim),
        system1=s1_tiny(latent_dim=latent_dim, **system1),
        system1_arch=arch,
        **dual,
    )


class DualSystem(nn.Module):
    """System 2 → latent → System 1, trained end-to-end."""

    def __init__(self, config: DualSystemConfig | None = None,
                 system1_stats=None) -> None:
        super().__init__()
        self.config = config or DualSystemConfig()
        self.system2 = System2(self.config.system2)
        # Statistics are needed only when building a model to train: they are persistent
        # buffers, so restoring a checkpoint carries whatever the run was trained with.
        system1_cls = System1 if self.config.system1_arch == "act" else ScratchSystem1
        self.system1 = system1_cls(self.config.system1, system1_stats)

        # Rollout state, reset per episode. Kept off the module's parameters so it
        # never leaks into checkpoints.
        self._cached_latent: torch.Tensor | None = None
        self._steps_since_update = 0
        self._scene_blind_frame: torch.Tensor | None = None

    # ---------------------------------------------------------------- latent sources

    def compute_latent(
        self,
        images,
        instructions: list[str],
        conditioning: Conditioning | None = None,
    ) -> torch.Tensor:
        """Produce `z` under the given conditioning mode.

        `images` are System 2's frames — already temporally offset by the caller, which
        owns the notion of time (the dataset during training, the rollout loop at eval).
        """
        mode = conditioning or self.config.conditioning
        batch_size = len(instructions)

        if mode is Conditioning.ZERO:
            # No System 2 call at all: cheaper, and unambiguous about what is measured.
            reference = next(self.system1.parameters())
            return torch.zeros(batch_size, self.config.system1.latent_dim,
                               device=reference.device, dtype=torch.float32)

        if mode is Conditioning.STATIC:
            images = self._placeholder_like(images)
        elif mode is Conditioning.SCENE_BLIND:
            images = [self._scene_blind_reference(images)] * batch_size

        return self.system2(images, instructions)

    def _placeholder_like(self, images):
        """A constant frame per batch element, so only the instruction informs `z`."""
        first = images[0]
        shape = tuple(first.shape) if torch.is_tensor(first) else (3, 224, 224)
        return [torch.full(shape, PLACEHOLDER_PIXEL, dtype=torch.float32)
                for _ in range(len(images))]

    def _scene_blind_reference(self, images):
        """A fixed frame, captured once, standing in for every observation.

        Distinct from STATIC: the latent still comes from *a* scene, just never this
        one. The difference between the two isolates how much of System 2's value is
        scene grounding rather than the mere presence of a conditioning vector.
        """
        if self._scene_blind_frame is None:
            first = images[0]
            tensor = first if torch.is_tensor(first) else torch.as_tensor(first)
            self._scene_blind_frame = tensor.detach().clone()
        return self._scene_blind_frame

    # ----------------------------------------------------------------------- forward

    def forward(
        self,
        images: dict[str, torch.Tensor],
        state: torch.Tensor,
        system2_images,
        instructions: list[str],
        conditioning: Conditioning | None = None,
        actions: torch.Tensor | None = None,
        action_is_pad: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, tuple[torch.Tensor | None, torch.Tensor | None]]:
        """One end-to-end pass. Returns ``(actions, latent, style_latent_params)``.

        Args:
            images: System 1's cameras at time t — key -> ``(B, 3, H, W)`` in [0, 1].
            state: ``(B, state_dim)`` proprioception at time t.
            system2_images: System 1 sees time t; these are System 2's frames, which the
                caller has already offset to ``t - Δ``.
            instructions: one per batch element. Reaches System 2 only — System 1 has
                no language input.
            conditioning: overrides the configured mode, for evaluating one checkpoint
                under several ablation modalities.
            actions, action_is_pad: the ground-truth chunk and its padding mask, which
                System 1's CVAE encoder consumes during training. Omitted at inference.

        The predicted actions are in **normalised** action space — the space the loss is
        computed in. `act()` is the path that returns environment-ready actions.

        The latent is returned alongside the actions because the ablation needs it:
        the pre/post-perturbation cosine distance is the quantitative evidence that
        System 2 noticed a mid-rollout change. The style latent's distribution
        parameters come back too, because the KLD term that regularises them belongs to
        the training loop rather than to the model.
        """
        latent = self.compute_latent(system2_images, instructions, conditioning)
        predicted, style_latent_params = self.system1(images, state, latent,
                                                      actions, action_is_pad)
        return predicted, latent, style_latent_params

    # ------------------------------------------------------------------ rollout path

    def reset(self) -> None:
        """Clear per-episode state. Call at every environment reset."""
        self._cached_latent = None
        self._steps_since_update = 0

    @torch.no_grad()
    def act(
        self,
        images: dict[str, torch.Tensor],
        state: torch.Tensor,
        system2_images,
        instructions: list[str],
        conditioning: Conditioning | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Rollout step with the latent held between updates. Returns ``(actions, z)``.

        Actions come back in the environment's units, unlike `forward`, which stays in
        the normalised space the loss is computed in.

        The update schedule *is* the dual-system premise: System 2 runs every K steps,
        System 1 every step. FROZEN is the same machinery with an unbounded period —
        computed once, never refreshed.
        """
        mode = conditioning or self.config.conditioning
        period = float("inf") if mode is Conditioning.FROZEN else self.config.latent_update_period

        due = self._cached_latent is None or self._steps_since_update >= period
        if due:
            # FROZEN evaluates a checkpoint trained on live latents, so its one latent
            # is computed the same way a live one would be — only the refresh differs.
            source = Conditioning.LIVE if mode is Conditioning.FROZEN else mode
            self._cached_latent = self.compute_latent(system2_images, instructions, source)
            self._steps_since_update = 0
        self._steps_since_update += 1

        # System 1 is forced into inference mode for the duration, for the same reason
        # this method is `no_grad`: it is the rollout path by definition, and both
        # dropout and the CVAE encoder are training-only machinery whose presence would
        # change the actions a rollout produces without changing anything visible.
        was_training = self.system1.training
        self.system1.eval()
        try:
            # Unnormalised, because this is the path whose output reaches `env.step`.
            predicted, _ = self.system1(images, state, self._cached_latent)
        finally:
            self.system1.train(was_training)
        return self.system1.unnormalise_action(predicted), self._cached_latent

    @property
    def steps_since_latent_update(self) -> int:
        return self._steps_since_update

    # ------------------------------------------------------------------- diagnostics

    def parameter_counts(self) -> dict[str, int]:
        s2 = sum(p.numel() for p in self.system2.parameters())
        s1 = sum(p.numel() for p in self.system1.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"system2": s2, "system1": s1, "total": s2 + s1, "trainable": trainable}


if __name__ == "__main__":
    model = DualSystem(tiny_config())
    counts = model.parameter_counts()
    print(f"system2   {counts['system2'] / 1e6:6.2f}M")
    print(f"system1   {counts['system1'] / 1e6:6.2f}M")
    print(f"trainable {counts['trainable'] / 1e6:6.2f}M of {counts['total'] / 1e6:.2f}M")

    cfg = model.config.system1
    model.eval()
    images = {k: torch.rand(1, 3, 128, 128) for k in cfg.camera_keys}
    actions, latent, _ = model(
        images, torch.rand(1, cfg.state_dim), [torch.rand(3, 128, 128)],
        ["put both the alphabet soup and the tomato sauce in the basket"],
    )
    print(f"\nactions {tuple(actions.shape)}  latent {tuple(latent.shape)}")
