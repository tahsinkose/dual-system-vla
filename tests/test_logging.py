"""Tests for eval/logging.py.

Run with::

    python -m pytest tests/test_logging.py -v
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eval.logging import (  # noqa: E402
    EpisodeResult,
    JsonlResultWriter,
    read_results,
    subtask_breakdown,
    summarize,
    write_latent_trace,
)


def _subtask(id, description, first_achieved_step=None, achieved_at_reset=False,
             achieved_at_end=False) -> dict:
    return dict(id=id, description=description, achieved_at_reset=achieved_at_reset,
                first_achieved_step=first_achieved_step, achieved_at_end=achieved_at_end)


def _result(**overrides) -> EpisodeResult:
    defaults = dict(
        task_dataset_index=0, task_instruction="do the thing", episode_index=0, seed=0,
        checkpoint_path="outputs/train/live/step_0000100.pt", trained_conditioning="live",
        eval_conditioning="live", success=True, steps_run=42, steps_to_success=42,
        perturbation_kind="none", perturbation_trigger_step=None, perturbation_applied=False,
        first_success_step=42, latent_cosine_distance=None, recovered=None, steps_to_recovery=None,
    )
    defaults.update(overrides)
    return EpisodeResult(**defaults)


def test_episode_result_round_trips_through_json():
    result = _result()
    parsed = json.loads(result.to_json_line())
    assert parsed["task_instruction"] == "do the thing"
    assert parsed["success"] is True
    assert parsed["steps_to_success"] == 42
    assert parsed["video_path"] is None


def test_jsonl_writer_survives_a_simulated_crash(tmp_path):
    path = tmp_path / "results.jsonl"
    writer = JsonlResultWriter(path)
    for i in range(5):
        writer.write(_result(episode_index=i))
    # Simulate a crash: drop the reference without calling .close().
    del writer

    lines = path.read_text().splitlines()
    assert len(lines) == 5
    for i, line in enumerate(lines):
        parsed = json.loads(line)
        assert parsed["episode_index"] == i


def test_jsonl_writer_appends_across_instances(tmp_path):
    path = tmp_path / "results.jsonl"
    writer = JsonlResultWriter(path)
    writer.write(_result(episode_index=0))
    writer.close()

    writer2 = JsonlResultWriter(path)
    writer2.write(_result(episode_index=1))
    writer2.close()

    lines = path.read_text().splitlines()
    assert len(lines) == 2


def test_write_latent_trace_round_trips_shape_and_dtype(tmp_path):
    path = tmp_path / "latents" / "ep0.npz"
    latents = np.random.rand(10, 32).astype(np.float32)
    steps_since_update = np.arange(10)

    write_latent_trace(path, latents, steps_since_update)

    with np.load(path) as archive:
        assert archive["latents"].dtype == np.float16
        assert archive["latents"].shape == (10, 32)
        assert archive["steps_since_update"].dtype == np.int16
        np.testing.assert_array_equal(archive["steps_since_update"], steps_since_update)


def test_summarize_zero_episodes():
    assert summarize([]) == "0 episodes run"


def test_summarize_all_success_no_perturbation():
    results = [_result(episode_index=i, steps_to_success=10 + i) for i in range(3)]
    summary = summarize(results)
    assert "3/3 succeeded (100.0%)" in summary
    assert "recovery" not in summary


def test_summarize_includes_recovery_line_when_perturbed():
    results = [
        _result(episode_index=0, perturbation_kind="displace_object", perturbation_applied=True,
               perturbation_trigger_step=5, recovered=True, steps_to_recovery=3),
        _result(episode_index=1, perturbation_kind="displace_object", perturbation_applied=True,
               perturbation_trigger_step=5, recovered=False, success=False, steps_to_success=None),
    ]
    summary = summarize(results)
    assert "recovery: 1/2 perturbed episodes recovered (50.0%)" in summary


def test_summarize_uses_steps_run_when_never_succeeded():
    results = [_result(success=False, steps_to_success=None, steps_run=99)]
    summary = summarize(results)
    assert "0/1 succeeded (0.0%)" in summary
    assert "mean 99.0" in summary


# ------------------------------------------------------------- subtask breakdown


def _two_step_result(placed: bool, **overrides) -> EpisodeResult:
    return _result(
        success=placed, steps_to_success=10 if placed else None,
        subtasks=[
            _subtask("grasp:soup_1", "pick up the soup", first_achieved_step=3),
            _subtask("in:soup_1:basket", "place the soup in the basket",
                     first_achieved_step=8 if placed else None, achieved_at_end=placed),
        ],
        subtasks_achieved=2 if placed else 1, subtasks_total=2, **overrides)


def test_breakdown_is_empty_without_subtask_records():
    assert subtask_breakdown([_result()]) == ""
    assert "subtask completion" not in summarize([_result()])


def test_breakdown_counts_episodes_reaching_each_step():
    """Where the count drops is the subtask the policy stalls at."""
    results = [_two_step_result(placed=i < 1, episode_index=i) for i in range(4)]
    breakdown = subtask_breakdown(results)
    assert "pick up the soup" in breakdown
    assert "  4/4  " in breakdown          # every episode grasped it
    assert "  1/4  " in breakdown          # only one went on to place it


def test_breakdown_groups_by_task():
    results = [
        _two_step_result(placed=True, task_dataset_index=0, task_instruction="first task"),
        _two_step_result(placed=False, task_dataset_index=5, task_instruction="second task"),
    ]
    breakdown = subtask_breakdown(results)
    assert "task 00  first task" in breakdown
    assert "task 05  second task" in breakdown


def _stove(achieved_at_end: bool) -> dict:
    """KITCHEN_SCENE8's stove: on before the policy acts, and it can be knocked off."""
    return _subtask("turnon:stove_1", "keep the stove on", first_achieved_step=0,
                    achieved_at_reset=True, achieved_at_end=achieved_at_end)


def test_breakdown_reports_survival_not_completion_for_reset_conditions():
    """A completion count is meaningless for something true before the first action.

    What can still go wrong is the policy undoing it, so the count reports how many
    trials ended with the condition intact.
    """
    results = [_result(subtasks=[_stove(achieved_at_end=end)], subtasks_total=0)
               for end in (True, True, False)]
    breakdown = subtask_breakdown(results)
    assert "2/3" in breakdown              # two trials left the stove on
    assert "still held at end" in breakdown
    assert "not scored" in breakdown


def test_breakdown_does_not_credit_a_reset_condition_as_completed():
    """The old form counted the step-0 latch and printed a full 3/3."""
    results = [_result(subtasks=[_stove(achieved_at_end=False)], subtasks_total=0)
               for _ in range(3)]
    assert "3/3" not in subtask_breakdown(results)


def test_summarize_includes_the_breakdown():
    assert "subtask completion" in summarize([_two_step_result(placed=True)])


# --------------------------------------------------------------------- reading back


def test_read_results_round_trips_everything_the_writer_wrote(tmp_path):
    path = tmp_path / "results.jsonl"
    writer = JsonlResultWriter(path)
    written = [
        _result(episode_index=0, subtasks=[_subtask("a", "pick it up", first_achieved_step=3)]),
        _result(episode_index=1, success=False, steps_to_success=None, steps_run=800),
    ]
    for result in written:
        writer.write(result)
    writer.close()

    assert read_results(path) == written


def test_read_results_ignores_blank_lines(tmp_path):
    path = tmp_path / "results.jsonl"
    path.write_text(_result().to_json_line() + "\n\n" + _result(episode_index=1).to_json_line() + "\n")
    assert len(read_results(path)) == 2


def test_read_results_drops_unknown_fields(tmp_path):
    """A log from a harness that recorded something extra must still read."""
    path = tmp_path / "results.jsonl"
    record = json.loads(_result().to_json_line())
    record["some_future_metric"] = 1.5
    path.write_text(json.dumps(record) + "\n")

    assert read_results(path)[0].episode_index == 0


def test_read_results_raises_on_a_missing_required_field(tmp_path):
    """Defaulting `success` or `steps_run` would make the summary quietly wrong.

    A dropped optional field costs nothing; a dropped required one changes the number
    the summary reports, so it has to raise rather than fall back.
    """
    path = tmp_path / "results.jsonl"
    record = json.loads(_result().to_json_line())
    del record["success"]
    path.write_text(json.dumps(record) + "\n")

    with pytest.raises(ValueError, match="missing required field"):
        read_results(path)


def test_read_results_names_the_line_that_is_not_json(tmp_path):
    path = tmp_path / "results.jsonl"
    path.write_text(_result().to_json_line() + "\nnot json at all\n")

    with pytest.raises(ValueError, match=r"results\.jsonl:2"):
        read_results(path)
