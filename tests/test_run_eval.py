"""Tests for eval/run_eval.py.

Fast tests use `FakeLiberoEnv` (tests/_fake_env.py) plus a `tiny_config()` DualSystem
— no real simulator, no real weights. One `@pytest.mark.slow` test exercises the full
CLI end to end against a real task.

Run with::

    python -m pytest tests/test_run_eval.py -v -m "not slow"
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.env_setup import setup_env  # noqa: E402

setup_env()

from src.models.dual_system import Conditioning, DualSystem, tiny_config  # noqa: E402

from eval.logging import EpisodeResult  # noqa: E402
from eval.perturbations import PerturbationKind, PerturbationSpec, TriggerCondition  # noqa: E402
from eval.run_eval import EvalConfig, TemporalOffsetBuffer, run_episode  # noqa: E402
from eval.trials import InitSource, build_trials  # noqa: E402
from _fake_env import FakeLiberoEnv  # noqa: E402

INSTRUCTION = "put both the alphabet soup and the tomato sauce in the basket"
# A real libero_10 task: the rollout loop looks its subtask decomposition up by BDDL
# basename, so a placeholder name would not resolve.
BDDL = "LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket.bddl"
GRASP_SOUP = "grasp:alphabet_soup_1"
PLACE_SOUP = "in:alphabet_soup_1:basket_1_contain_region"


@pytest.fixture(autouse=True)
def _no_real_init_state_lookup(monkeypatch):
    """FakeLiberoEnv-based tests must not depend on the real, on-disk
    data/init_states.npz (whether a given episode index happens to be matched or
    not there is irrelevant to — and would make brittle — these unit tests)."""
    monkeypatch.setattr("eval.run_eval.load_exact_init_state", lambda episode_index: None)


def _tiny_model() -> DualSystem:
    return DualSystem(tiny_config())


def _spec(kind=PerturbationKind.NONE, trigger=None, seed=(0, 0)) -> PerturbationSpec:
    return PerturbationSpec(kind=kind, trigger=trigger, episode_seed=seed)


def _cfg(**overrides) -> EvalConfig:
    defaults = dict(checkpoint=Path("unused"), horizon=8, settle_steps=0,
                    allow_unmatched_episodes=True)
    defaults.update(overrides)
    return EvalConfig(**defaults)


class _SpyEnv(FakeLiberoEnv):
    def __post_init__(self):
        super().__post_init__()
        self.step_actions: list[np.ndarray] = []

    def step(self, action):
        self.step_actions.append(np.array(action, dtype=np.float64, copy=True))
        return super().step(action)


def _trial(index=0, init_state=None):
    from eval.trials import Trial

    return Trial(task_dataset_index=0, instruction=INSTRUCTION, bddl=BDDL,
                 init_state=init_state, source=InitSource.BENCHMARK, index=index)


def _run(model, env, cfg, spec=None, conditioning=None):
    spec = spec or _spec()
    conditioning = conditioning or Conditioning.LIVE
    return run_episode(model, env, _trial(), eval_conditioning=conditioning,
                       trained_conditioning=Conditioning.LIVE, perturbation=spec,
                       cfg=cfg, checkpoint_path="unused", device=torch.device("cpu"))


# --------------------------------------------------------------- receding horizon


def test_receding_horizon_calls_act_once_per_env_step():
    model = _tiny_model()
    env = _SpyEnv()
    cfg = _cfg(horizon=5)
    real_act = model.act
    call_count = 0

    def counting_act(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        return real_act(*args, **kwargs)

    with patch.object(model, "act", side_effect=counting_act):
        result = _run(model, env, cfg)

    assert result.steps_run == 5
    assert len(env.step_actions) == 5
    assert call_count == 5   # one act() call per env step — no chunk queue


def test_action_is_clipped_before_env_step():
    """System 1's action head is deliberately unsquashed, so it can legitimately
    predict values outside [-1, 1]; the rollout loop, not the model, must clamp."""
    model = _tiny_model()
    env = _SpyEnv()
    cfg = _cfg(horizon=3)
    with torch.no_grad():
        for p in model.system1.act.action_head.parameters():
            p.mul_(100.0)   # force out-of-range predictions
    _run(model, env, cfg)
    assert env.step_actions   # sanity: the loop actually ran
    for action in env.step_actions:
        assert np.all(action >= -1.0 - 1e-6) and np.all(action <= 1.0 + 1e-6)


# ------------------------------------------------------------ temporal offset buffer


def test_temporal_offset_buffer_feeds_frame_t_minus_delta():
    buffer = TemporalOffsetBuffer(offset=2)
    for i in range(6):
        buffer.push(torch.full((1,), float(i)))
        if i >= 2:
            assert buffer.offset_frame().item() == pytest.approx(i - 2)


def test_temporal_offset_buffer_clamps_before_delta_elapses():
    buffer = TemporalOffsetBuffer(offset=3)
    buffer.push(torch.full((1,), 0.0))
    assert buffer.offset_frame().item() == pytest.approx(0.0)
    buffer.push(torch.full((1,), 1.0))
    assert buffer.offset_frame().item() == pytest.approx(0.0)


def test_temporal_offset_buffer_latest_frame_is_most_recent():
    buffer = TemporalOffsetBuffer(offset=1)
    buffer.push(torch.full((1,), 0.0))
    buffer.push(torch.full((1,), 1.0))
    assert buffer.latest_frame().item() == pytest.approx(1.0)


# ------------------------------------------------------------------- perturbations


def test_perturbation_fires_at_configured_step_and_flips_perturbation_applied():
    model = _tiny_model()
    env = FakeLiberoEnv(horizon=100)
    cfg = _cfg(horizon=10)
    spec = _spec(kind=PerturbationKind.DISPLACE_OBJECT, trigger=TriggerCondition(at_step=3))
    result = _run(model, env, cfg, spec=spec)
    assert result.perturbation_applied is True
    assert result.perturbation_trigger_step == 3


def test_perturbation_never_fires_is_not_an_error():
    model = _tiny_model()
    env = FakeLiberoEnv(horizon=100)   # check_success() never True
    cfg = _cfg(horizon=5)
    spec = _spec(kind=PerturbationKind.UNDO_PROGRESS, trigger=TriggerCondition(after_success_steps=2))
    result = _run(model, env, cfg, spec=spec)
    assert result.perturbation_applied is False
    assert result.perturbation_trigger_step is None
    assert result.latent_cosine_distance is None
    assert result.recovered is None


def test_cosine_distance_uses_pre_and_post_perturbation_frames_only():
    model = _tiny_model()
    env = FakeLiberoEnv(horizon=100)
    cfg = _cfg(horizon=6)
    spec = _spec(kind=PerturbationKind.DISPLACE_OBJECT, trigger=TriggerCondition(at_step=2))
    before_pose = env.sim.data.get_joint_qpos(env.joint_name).copy()
    with patch.object(model, "compute_latent", wraps=model.compute_latent) as spy:
        result = _run(model, env, cfg, spec=spec)
    # act() itself calls compute_latent once, at step 0, to seed the initial cached
    # latent (horizon=6 is well under the default K=10 update period, so that's the
    # only "internal" call) — plus exactly 2 more for the pre/post-perturbation probe.
    assert spy.call_count == 3
    # The fake env's agentview_image is static regardless of physical state, so a
    # pixel-equality check on the probe's frames wouldn't be meaningful; check
    # instead that the object genuinely moved between the two probe calls.
    after_pose = env.sim.data.get_joint_qpos(env.joint_name)
    assert not np.allclose(before_pose, after_pose)
    assert result.latent_cosine_distance is not None


def test_zero_conditioning_skips_the_cosine_probe():
    model = _tiny_model()
    env = FakeLiberoEnv(horizon=100)
    cfg = _cfg(horizon=6)
    spec = _spec(kind=PerturbationKind.DISPLACE_OBJECT, trigger=TriggerCondition(at_step=2))
    with patch.object(model, "compute_latent", wraps=model.compute_latent) as spy:
        result = _run(model, env, cfg, spec=spec, conditioning=Conditioning.ZERO)
    # The one call is act()'s own internal ZERO-mode invocation (a cheap early return,
    # no System 2 forward pass) — not the probe, which is skipped entirely under ZERO.
    # If the probe fired too, this would be 3, as in the LIVE case above.
    assert spy.call_count == 1
    assert result.latent_cosine_distance is None
    assert result.perturbation_applied is True


def test_recovered_is_true_only_for_post_trigger_success():
    model = _tiny_model()
    # Success at step 1 (pre-trigger) and step 5 (post-trigger, trigger fires at 3).
    env = FakeLiberoEnv(horizon=100, success_at_steps=frozenset({1, 5}))
    cfg = _cfg(horizon=8)
    spec = _spec(kind=PerturbationKind.DISPLACE_OBJECT, trigger=TriggerCondition(at_step=3))
    result = _run(model, env, cfg, spec=spec)
    assert result.success is True
    assert result.recovered is True
    assert result.steps_to_recovery == 2   # 5 - 3


def test_recovered_is_false_when_no_post_trigger_success():
    model = _tiny_model()
    env = FakeLiberoEnv(horizon=100, success_at_steps=frozenset({1}))   # only pre-trigger
    cfg = _cfg(horizon=8)
    spec = _spec(kind=PerturbationKind.DISPLACE_OBJECT, trigger=TriggerCondition(at_step=3))
    result = _run(model, env, cfg, spec=spec)
    assert result.success is True          # latched from step 1
    assert result.recovered is False
    assert result.steps_to_recovery is None


def test_after_success_trigger_fires_although_the_env_reports_done_at_success():
    """LIBERO derives `done` from its goal predicate, so an episode would otherwise end
    on the step it succeeds — leaving `first_success_step + N` unreachable for any N
    above zero, and the recovery ablation unable to run at all."""
    model = _tiny_model()
    env = FakeLiberoEnv(horizon=100, success_at_steps=frozenset({2}))
    cfg = _cfg(horizon=20)
    spec = _spec(kind=PerturbationKind.UNDO_PROGRESS,
                 trigger=TriggerCondition(after_success_steps=5))
    result = _run(model, env, cfg, spec=spec)
    assert result.first_success_step == 2
    assert result.perturbation_applied is True
    assert result.perturbation_trigger_step == 7      # 2 + 5


def test_unperturbed_episode_runs_on_past_success_for_the_video():
    """The goal predicate can hold before the outcome is visible — a placed object
    counts as contained while the gripper still holds it — so a recording cut at the
    trigger does not show the episode being solved."""
    model = _tiny_model()
    env = FakeLiberoEnv(horizon=100, success_at_steps=frozenset({3}))
    cfg = _cfg(horizon=50, post_success_steps=4)
    result = _run(model, env, cfg)
    assert result.first_success_step == 3
    assert result.steps_run == 8            # 3 + 1 + 4, not the full horizon


def test_unperturbed_episode_stops_promptly_after_success():
    """Running on is bounded: it must not silently become a full-horizon rollout."""
    model = _tiny_model()
    env = FakeLiberoEnv(horizon=100, success_at_steps=frozenset({1}))
    cfg = _cfg(horizon=50, post_success_steps=0)
    result = _run(model, env, cfg)
    assert result.steps_run == 2            # 1 + 1 + 0


def test_undo_progress_precondition_snapshot_is_taken_before_any_step():
    import eval.run_eval as run_eval_module

    model = _tiny_model()
    env = FakeLiberoEnv(horizon=100)
    cfg = _cfg(horizon=3)
    spec = _spec(kind=PerturbationKind.UNDO_PROGRESS, trigger=TriggerCondition(at_step=1))

    call_order: list[str] = []
    real_snapshot = run_eval_module.snapshot_target_object_pose
    real_act = model.act

    def spy_snapshot(*args, **kwargs):
        call_order.append("snapshot")
        return real_snapshot(*args, **kwargs)

    def spy_act(*args, **kwargs):
        call_order.append("act")
        return real_act(*args, **kwargs)

    with patch.object(run_eval_module, "snapshot_target_object_pose", side_effect=spy_snapshot) as spy:
        with patch.object(model, "act", side_effect=spy_act):
            _run(model, env, cfg, spec=spec)

    assert spy.call_count == 1
    assert call_order[0] == "snapshot"   # taken before the first model.act() call


# ---------------------------------------------------------------- subtask logging


def test_subtask_progress_is_recorded_per_episode():
    """A rollout that grasps the soup and places it, then never touches the sauce.

    Its `success` is False — the goal is a conjunction — so only the per-subtask
    record distinguishes it from a rollout that never moved.
    """
    model = _tiny_model()
    env = FakeLiberoEnv(horizon=100, subtask_true_at={
        GRASP_SOUP: frozenset({1, 2, 3}),      # released on placement, as a real grasp is
        PLACE_SOUP: frozenset({4, 5, 6, 7}),
    })
    result = _run(model, env, _cfg(horizon=8))

    assert result.subtasks_total == 4
    assert result.subtasks_achieved == 2
    records = {s["id"]: s for s in result.subtasks}
    assert records[GRASP_SOUP]["first_achieved_step"] == 1
    assert records[GRASP_SOUP]["achieved_at_end"] is False
    assert records[PLACE_SOUP]["first_achieved_step"] == 4
    assert records[PLACE_SOUP]["achieved_at_end"] is True
    assert records["grasp:tomato_sauce_1"]["first_achieved_step"] is None
    assert result.success is False


def test_subtask_records_keep_task_order():
    model = _tiny_model()
    env = FakeLiberoEnv(horizon=100)
    result = _run(model, env, _cfg(horizon=3))
    assert [s["id"] for s in result.subtasks] == [
        GRASP_SOUP, PLACE_SOUP, "grasp:tomato_sauce_1", "in:tomato_sauce_1:basket_1_contain_region",
    ]


def test_subtask_state_at_reset_is_read_before_the_first_action():
    model = _tiny_model()
    env = FakeLiberoEnv(horizon=100, subtask_true_at={PLACE_SOUP: frozenset({-1, 0, 1})})
    result = _run(model, env, _cfg(horizon=3))
    records = {s["id"]: s for s in result.subtasks}
    assert records[PLACE_SOUP]["achieved_at_reset"] is True
    assert records[GRASP_SOUP]["achieved_at_reset"] is False


def test_perturbation_undoing_a_subtask_shows_in_the_final_reading():
    """`achieved_at_end` is re-read at the moment the perturbation edits sim state."""
    model = _tiny_model()
    env = FakeLiberoEnv(horizon=100, subtask_true_at={PLACE_SOUP: frozenset({0, 1, 2})})
    spec = _spec(kind=PerturbationKind.DISPLACE_OBJECT, trigger=TriggerCondition(at_step=3))
    result = _run(model, env, _cfg(horizon=6), spec=spec)
    records = {s["id"]: s for s in result.subtasks}
    assert records[PLACE_SOUP]["first_achieved_step"] == 0
    assert records[PLACE_SOUP]["achieved_at_end"] is False


# ------------------------------------------------------------------- trial building


class _FakeTask:
    dataset_index = 0
    instruction = INSTRUCTION
    bddl = BDDL
    suite = "libero_10"


class _FakeMapping:
    def dataset_indices(self):
        return [0]

    def by_dataset_index(self, index):
        return _FakeTask()

    def by_episode(self, dataset, episode_index):
        return _FakeTask()


def test_demo_trials_skip_unmatched_by_default_and_keep_order(monkeypatch):
    """Episodes without a recovered initial state would start from a different object
    layout than the demonstration, so they are dropped rather than silently run."""
    monkeypatch.setattr("src.utils.episodes_for_task", lambda dataset, t: [5, 3, 1])
    matched = {1, 5}
    monkeypatch.setattr("src.utils.load_exact_init_state",
                        lambda e: object() if e in matched else None)
    monkeypatch.setattr("src.utils.load_episode", lambda dataset, e: ("do the thing", None, None))

    trials = build_trials(_FakeMapping(), None, InitSource.DEMO, task_indices=[0])
    assert [t.index for t in trials] == [1, 5]


def test_demo_trials_allow_unmatched_keeps_everything(monkeypatch):
    monkeypatch.setattr("src.utils.episodes_for_task", lambda dataset, t: [5, 3, 1])
    monkeypatch.setattr("src.utils.load_exact_init_state", lambda e: None)
    monkeypatch.setattr("src.utils.load_episode", lambda dataset, e: ("do the thing", None, None))

    trials = build_trials(_FakeMapping(), None, InitSource.DEMO, task_indices=[0],
                          allow_unmatched=True)
    assert [t.index for t in trials] == [1, 3, 5]


def test_demo_trials_explicit_episodes_win_over_task_indices(monkeypatch):
    monkeypatch.setattr("src.utils.load_exact_init_state", lambda e: object())
    monkeypatch.setattr("src.utils.load_episode", lambda dataset, e: ("do the thing", None, None))

    trials = build_trials(_FakeMapping(), None, InitSource.DEMO, task_indices=[0],
                          episodes=[9, 2])
    assert [t.index for t in trials] == [2, 9]


# ------------------------------------------------------------------- EvalConfig


def test_eval_config_rejects_trigger_flags_without_perturb():
    with pytest.raises(ValueError):
        _cfg(perturb="none", perturb_at_step=5)


def test_eval_config_requires_exactly_one_trigger_when_perturbing():
    with pytest.raises(ValueError):
        _cfg(perturb="displace_object")
    with pytest.raises(ValueError):
        _cfg(perturb="displace_object", perturb_at_step=1, perturb_after_success_steps=1)


def test_eval_config_warns_for_undo_progress_without_after_success_steps(capsys):
    _cfg(perturb="undo_progress", perturb_at_step=5)
    assert "WARNING" in capsys.readouterr().out


# ------------------------------------------------------------------------- slow


@pytest.mark.slow
def test_end_to_end_tiny_model_rollout_on_real_env(tmp_path):
    import json as json_module

    from src.env_setup import setup_env as _setup

    _setup()
    import os

    os.environ.setdefault("MUJOCO_GL", "egl")
    from src.models.dual_system import tiny_config as dual_tiny_config
    from src.train import TrainConfig, build_model, save_checkpoint

    from eval.run_eval import main, parse_args

    from dataclasses import asdict

    output_dir = tmp_path / "checkpoints"
    cfg = TrainConfig(conditioning="live", output_dir=tmp_path, tiny=True, chunk_size=4)
    model = build_model(cfg)
    run_dir = output_dir / "live"
    run_dir.mkdir(parents=True)
    (run_dir / "config.json").write_text(
        json_module.dumps({**asdict(cfg), "output_dir": str(cfg.output_dir)}, default=str)
    )
    checkpoint_path = save_checkpoint(model, cfg, 1, run_dir)

    eval_cfg = parse_args([
        "--checkpoint", str(checkpoint_path),
        "--task-indices", "0",
        "--horizon", "10",
        "--settle-steps", "0",
        "--allow-unmatched-episodes",
        "--episodes", "0",
        "--log-path", str(tmp_path / "results.jsonl"),
        "--video-dir", str(tmp_path / "videos"),
    ])
    results = main(eval_cfg)

    assert len(results) == 1
    lines = (tmp_path / "results.jsonl").read_text().splitlines()
    assert len(lines) == 1
    parsed = json.loads(lines[0])
    assert parsed["task_dataset_index"] == 0
    # The subtask decomposition survives the round trip through the real env and JSONL.
    assert 0 < parsed["subtasks_total"] <= len(parsed["subtasks"])
    assert all("first_achieved_step" in s for s in parsed["subtasks"])

    videos = list((tmp_path / "videos").glob("*.mp4"))
    assert len(videos) == 1
    assert videos[0].stat().st_size > 0
