"""Tests for the training loop.

Concentrated on the parts that are wrong *silently* — the padding mask and the
temporal offset both produce a perfectly healthy-looking loss curve when broken.

Run with::

    python -m pytest tests/test_train.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.train import (  # noqa: E402
    DATASET_MAIN_CAMERA,
    DATASET_WRIST_CAMERA,
    TrainConfig,
    build_delta_timestamps,
    build_model,
    build_optimizer,
    action_loss,
    masked_l1_loss,
    parse_args,
    split_batch,
)

FPS = 10.0


# ------------------------------------------------------------------ padding mask


def test_padding_steps_are_excluded():
    """Padded chunk steps must not contribute, however wrong they are."""
    predicted = torch.zeros(1, 4, 7)
    target = torch.zeros(1, 4, 7)
    target[0, 2:] = 100.0                      # nonsense, but flagged as padding
    is_pad = torch.tensor([[False, False, True, True]])
    assert masked_l1_loss(predicted, target, is_pad).item() == pytest.approx(0.0)


def test_loss_is_a_mean_over_valid_steps_only():
    """The divisor is the valid count, not the full chunk.

    Dividing by the full chunk would shrink the loss as episodes near their end,
    quietly down-weighting exactly the steps that decide task success.
    """
    predicted = torch.zeros(1, 4, 7)
    target = torch.ones(1, 4, 7)
    is_pad = torch.tensor([[False, False, True, True]])
    # Every valid element is off by 1, so the mean over valid elements is 1.0.
    assert masked_l1_loss(predicted, target, is_pad).item() == pytest.approx(1.0)


def test_all_padding_does_not_divide_by_zero():
    loss = masked_l1_loss(torch.zeros(1, 2, 7), torch.ones(1, 2, 7),
                          torch.tensor([[True, True]]))
    assert torch.isfinite(loss)


def test_shape_mismatch_raises_instead_of_broadcasting():
    """A chunk-size mismatch must fail loudly, not broadcast into a wrong loss."""
    with pytest.raises(ValueError, match="does not match target"):
        masked_l1_loss(torch.zeros(1, 4, 7), torch.zeros(1, 16, 7),
                       torch.zeros(1, 16, dtype=torch.bool))


# --------------------------------------------------------------- temporal offset


def test_delta_timestamps_request_the_offset_frame_pair():
    """System 2 must be fed `t - Δ` and System 1 `t`, from one sample."""
    cfg = TrainConfig(chunk_size=4, temporal_offset=2)
    deltas = build_delta_timestamps(cfg, FPS)
    assert deltas["action"] == [0.0, 0.1, 0.2, 0.3]
    assert deltas[DATASET_MAIN_CAMERA] == [-0.2, 0.0]


def test_zero_offset_still_requests_two_frames():
    """Δ=0 is a valid ablation of the offset itself; it must not collapse the pair."""
    deltas = build_delta_timestamps(TrainConfig(chunk_size=2, temporal_offset=0), FPS)
    assert deltas[DATASET_MAIN_CAMERA] == [0.0, 0.0]


def test_split_batch_routes_the_two_frames_correctly():
    """The offset is only real if the *older* frame reaches System 2.

    Swapping these silently trains System 2 on the fresh frame and System 1 on the
    stale one — the exact inverse of the deployment condition it exists to reproduce.
    """
    batch_size, size = 2, 8
    older = torch.zeros(batch_size, 3, size, size)      # t - Δ
    newer = torch.ones(batch_size, 3, size, size)       # t
    batch = {
        DATASET_MAIN_CAMERA: torch.stack([older, newer], dim=1),
        DATASET_WRIST_CAMERA: torch.full((batch_size, 3, size, size), 0.5),
        "observation.state": torch.zeros(batch_size, 8),
        "action": torch.zeros(batch_size, 4, 7),
        "action_is_pad": torch.zeros(batch_size, 4, dtype=torch.bool),
        "task": ["do the thing"] * batch_size,
    }
    observation, system2_frames, actions, is_pad, instructions = split_batch(batch)

    assert all(float(f.mean()) == pytest.approx(0.0) for f in system2_frames), \
        "System 2 received the fresh frame instead of the offset one"
    assert float(observation.images["image"].mean()) == pytest.approx(1.0), \
        "System 1 received the stale frame instead of the current one"
    assert float(observation.images["image2"].mean()) == pytest.approx(0.5)
    assert actions.shape == (batch_size, 4, 7)
    assert is_pad.shape == (batch_size, 4)
    assert instructions == ["do the thing"] * batch_size


# ------------------------------------------------------------------- optimisation


def test_optimizer_separates_the_two_learning_rates():
    """System 1 trains from scratch; System 2's adapters sit on a pretrained encoder."""
    cfg = TrainConfig(tiny=True, chunk_size=4, lr_system1=1e-4, lr_system2=1e-5)
    optimizer = build_optimizer(build_model(cfg), cfg)
    assert [g["lr"] for g in optimizer.param_groups] == [1e-4, 1e-5]
    assert all(g["params"] for g in optimizer.param_groups), "an empty parameter group"


def test_only_trainable_parameters_are_optimised():
    """The frozen VLM backbone must not be handed to AdamW."""
    cfg = TrainConfig(tiny=True, chunk_size=4)
    model = build_model(cfg)
    optimized = {id(p) for g in build_optimizer(model, cfg).param_groups for p in g["params"]}
    frozen = [p for p in model.parameters() if not p.requires_grad]
    assert frozen, "expected a frozen backbone"
    assert not (optimized & {id(p) for p in frozen})


def test_tiny_model_chunk_size_follows_the_config():
    """The tiny defaults must not silently disagree with the dataloader."""
    model = build_model(TrainConfig(tiny=True, chunk_size=9))
    assert model.config.system1.chunk_size == 9


def test_a_training_step_moves_system2_adapters():
    """End to end: one optimiser step must change System 2, not just System 1.

    If the latent path were broken the loss would still fall — System 1 alone can fit
    a little — while System 2 sat frozen. That is the failure this catches.
    """
    cfg = TrainConfig(tiny=True, chunk_size=4, lr_system1=1e-3, lr_system2=1e-3)
    model = build_model(cfg)
    optimizer = build_optimizer(model, cfg)

    lora = {n: p for n, p in model.system2.named_parameters()
            if p.requires_grad and "lora" in n.lower()}
    before = {n: p.detach().clone() for n, p in lora.items()}

    images = {k: torch.rand(2, 3, 32, 32) for k in model.config.system1.camera_keys}
    actions = torch.rand(2, 4, 7)
    is_pad = torch.zeros(2, 4, dtype=torch.bool)
    model.train()
    predicted, _latent, style_latent = model(
        images, torch.rand(2, 8), [torch.rand(3, 32, 32)] * 2, ["do the thing"] * 2,
        actions=actions, action_is_pad=is_pad)
    loss, _ = action_loss(predicted, model.system1.normalise_action(actions), is_pad,
                          style_latent, cfg.kl_weight)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

    assert any(not torch.equal(before[n], p) for n, p in lora.items()), \
        "an optimiser step left System 2 unchanged"


# --------------------------------------------------------------------------- cli


def test_cli_produces_the_two_training_conditionings():
    assert parse_args(["--conditioning", "live"]).conditioning == "live"
    assert parse_args(["--conditioning", "static"]).conditioning == "static"


def test_cli_rejects_test_time_only_conditionings():
    """`frozen`, `zero` and `scene_blind` are evaluation interventions, not runs."""
    with pytest.raises(SystemExit):
        parse_args(["--conditioning", "frozen"])


def test_loss_matches_acts_formulation():
    """Our masked L1 is ACT's reconstruction loss, elementwise identical.

    ACT computes it as `(abs_err * valid_mask).sum() / (valid_mask.sum() * action_dim)`.
    Ours expands the mask to element count instead of multiplying — the same divisor,
    but a refactor could easily change one and not the other, so it is pinned here.

    Only the reconstruction half is checked here; `test_objective_matches_acts_full_loss`
    pins the KLD term on top of it.
    """
    import torch.nn.functional as F

    torch.manual_seed(0)
    for _ in range(5):
        predicted, target = torch.randn(3, 8, 7), torch.randn(3, 8, 7)
        is_pad = torch.rand(3, 8) < 0.4

        abs_err = F.l1_loss(target, predicted, reduction="none")
        valid_mask = ~is_pad.unsqueeze(-1)
        num_valid = valid_mask.sum() * abs_err.shape[-1]
        reference = (abs_err * valid_mask).sum() / num_valid.clamp_min(1)

        torch.testing.assert_close(masked_l1_loss(predicted, target, is_pad), reference)


def test_objective_matches_acts_full_loss():
    """The whole objective, not just its reconstruction half.

    System 1 wraps ACT's inner module rather than `ACTPolicy`, which means the KLD term
    `ACTPolicy` would have computed is this training loop's responsibility. Getting the
    weighting or the reduction wrong would still produce a falling curve, so it is
    pinned against upstream's own formulation: sum over the style latent's dimensions,
    mean over the batch, scaled by `kl_weight`.
    """
    import torch.nn.functional as F

    from src.train import action_loss

    torch.manual_seed(0)
    kl_weight = 10.0
    for _ in range(5):
        predicted, target = torch.randn(3, 8, 7), torch.randn(3, 8, 7)
        is_pad = torch.rand(3, 8) < 0.4
        mu, log_sigma_x2 = torch.randn(3, 32), torch.randn(3, 32)

        abs_err = F.l1_loss(target, predicted, reduction="none")
        valid_mask = ~is_pad.unsqueeze(-1)
        num_valid = valid_mask.sum() * abs_err.shape[-1]
        reference_l1 = (abs_err * valid_mask).sum() / num_valid.clamp_min(1)
        reference_kld = (
            -0.5 * (1 + log_sigma_x2 - mu.pow(2) - log_sigma_x2.exp())
        ).sum(-1).mean()

        loss, components = action_loss(predicted, target, is_pad,
                                       (mu, log_sigma_x2), kl_weight)
        torch.testing.assert_close(loss, reference_l1 + kl_weight * reference_kld)
        assert components["l1"] == pytest.approx(reference_l1.item())
        assert components["kld"] == pytest.approx(reference_kld.item())


def test_objective_drops_the_kld_term_at_inference():
    """With the style latent zeroed there is no distribution to regularise.

    Validation runs in eval mode, so its loss must be the reconstruction term alone —
    otherwise it would not be comparable to the training curve's `l1`.
    """
    from src.train import action_loss

    predicted, target = torch.randn(2, 4, 7), torch.randn(2, 4, 7)
    is_pad = torch.zeros(2, 4, dtype=torch.bool)
    loss, components = action_loss(predicted, target, is_pad, (None, None), 10.0)

    torch.testing.assert_close(loss, masked_l1_loss(predicted, target, is_pad))
    assert components["kld"] == 0.0


# ------------------------------------------------------------ validation split


def test_split_is_disjoint_and_covers_everything():
    from src.train import split_episodes

    episodes = list(range(37))
    train, val = split_episodes(episodes, 0.1)
    assert set(train) & set(val) == set(), "an episode is in both splits"
    assert sorted(train + val) == episodes, "episodes were lost or duplicated"
    assert val, "no validation episodes held out"


def test_split_is_deterministic_across_runs():
    """CKPT-DUAL and CKPT-STATIC must validate on the identical set.

    A stride rather than a seeded shuffle, so the two runs agree without having to
    share an RNG seed — otherwise their validation curves would not be comparable.
    """
    from src.train import split_episodes

    episodes = list(range(29))
    assert split_episodes(episodes, 0.1) == split_episodes(episodes, 0.1)


def test_split_never_empties_the_training_set():
    """Tasks have as few as 29 episodes; a degenerate split must not starve training."""
    from src.train import split_episodes

    for count in (2, 3, 5, 29, 49):
        train, val = split_episodes(list(range(count)), 0.1)
        assert train, f"empty training set for {count} episodes"
    # At realistic sizes — the smallest libero_10 task has 29 episodes — training must
    # dominate. Below that the split degenerates gracefully rather than usefully.
    for count in (29, 49):
        train, val = split_episodes(list(range(count)), 0.1)
        assert len(train) > 4 * len(val), f"validation too large at {count} episodes"


def test_zero_fraction_withholds_nothing():
    from src.train import split_episodes

    train, val = split_episodes(list(range(20)), 0.0)
    assert val == [] and train == list(range(20))


def test_scoring_falls_back_to_the_training_episodes_when_nothing_is_withheld(monkeypatch):
    """Training on everything must still produce a best.pt.

    Which episodes train and which are scored are separate choices: at
    `val_fraction=0` every episode trains and the scoring pass runs over those same
    episodes, flagged so the number is never mistaken for a held-out one.
    """
    from src.train import TrainConfig, build_datasets

    built = []

    class _FakeDataset:
        def __init__(self, repo_id, episodes=None, delta_timestamps=None):
            self.episodes = list(episodes) if episodes is not None else None
            built.append(self.episodes)

        @property
        def num_episodes(self):
            return len(self.episodes)

    monkeypatch.setattr("lerobot.datasets.lerobot_dataset.LeRobotDataset", _FakeDataset)
    monkeypatch.setattr("lerobot.datasets.lerobot_dataset.LeRobotDatasetMetadata",
                        lambda repo_id: type("M", (), {"fps": 10})())
    monkeypatch.setattr("src.utils.episodes_for_task", lambda dataset, t: [0, 1, 2, 3, 4])

    train_ds, score_ds, held_out = build_datasets(
        TrainConfig(val_fraction=0.0, task_indices=[0]))
    assert held_out is False
    assert score_ds is train_ds                  # scored on what it trained on
    assert train_ds.episodes == [0, 1, 2, 3, 4]  # nothing withheld

    train_ds, score_ds, held_out = build_datasets(
        TrainConfig(val_fraction=0.5, task_indices=[0]))
    assert held_out is True
    assert score_ds is not train_ds
    assert set(train_ds.episodes) & set(score_ds.episodes) == set()


def test_validation_restores_training_mode():
    """Forgetting the restore would silently continue training with dropout disabled.

    The loss curve would then look *better*, not worse, which is why this is asserted
    rather than left to review.
    """
    from src.train import validate

    cfg = TrainConfig(tiny=True, chunk_size=4)
    model = build_model(cfg)
    model.train()

    batch = {
        DATASET_MAIN_CAMERA: torch.rand(1, 2, 3, 32, 32),
        DATASET_WRIST_CAMERA: torch.rand(1, 3, 32, 32),
        "observation.state": torch.zeros(1, 8),
        "action": torch.zeros(1, 4, 7),
        "action_is_pad": torch.zeros(1, 4, dtype=torch.bool),
        "task": ["do the thing"],
    }
    loss = validate(model, [batch], torch.device("cpu"))
    assert loss >= 0.0
    assert model.training, "validate() left the model in eval mode"


def test_validation_runs_without_gradients():
    """A validation pass must not build a graph — it would leak memory across a run."""
    from src.train import validate

    cfg = TrainConfig(tiny=True, chunk_size=4)
    model = build_model(cfg)
    batch = {
        DATASET_MAIN_CAMERA: torch.rand(1, 2, 3, 32, 32),
        DATASET_WRIST_CAMERA: torch.rand(1, 3, 32, 32),
        "observation.state": torch.zeros(1, 8),
        "action": torch.zeros(1, 4, 7),
        "action_is_pad": torch.zeros(1, 4, dtype=torch.bool),
        "task": ["do the thing"],
    }
    model.zero_grad(set_to_none=True)
    validate(model, [batch], torch.device("cpu"))
    assert all(p.grad is None for p in model.parameters() if p.requires_grad)


def test_validation_subset_spans_every_task():
    """Bounding validation by "first N batches" silently narrows it to one or two tasks.

    Samples are ordered by episode, so a prefix of the validation set covers only the
    earliest tasks — `best.pt` would then be selected on a fraction of the benchmark.
    A stride keeps the cost bounded while covering all of them.
    """
    from src.train import validation_subset

    # Stand-in for the ordered validation set: 10 tasks, 100 contiguous samples each.
    ordered = [t for t in range(10) for _ in range(100)]
    subset = validation_subset(ordered, max_samples=50)
    assert len(set(subset)) == 10, f"subset covers only tasks {sorted(set(subset))}"
    assert len(subset) <= 50


def test_validation_subset_is_fixed_across_calls():
    """Successive losses must differ because the model changed, not the sample."""
    from src.train import validation_subset

    ordered = list(range(1000))
    assert list(validation_subset(ordered, 40)) == list(validation_subset(ordered, 40))


def test_validation_subset_passes_small_sets_through():
    from src.train import validation_subset

    small = list(range(10))
    assert validation_subset(small, 50) is small
