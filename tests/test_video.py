"""Tests for eval/video.py.

Run with::

    python -m pytest tests/test_video.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.env_setup import setup_env  # noqa: E402

setup_env()

from eval.video import PERTURBATION_BANNER_FRAMES, RolloutVideoWriter, _burn_text  # noqa: E402


def _frame(size: int = 128) -> np.ndarray:
    return np.zeros((size, size, 3), dtype=np.uint8)


def test_close_is_a_noop_when_no_frames_were_added(tmp_path):
    writer = RolloutVideoWriter(tmp_path / "ep.mp4", conditioning="live")
    writer.close()
    assert not (tmp_path / "ep.mp4").exists()


def test_close_writes_a_file_when_frames_were_added(tmp_path):
    writer = RolloutVideoWriter(tmp_path / "ep.mp4", conditioning="live")
    for step in range(3):
        writer.add_frame(_frame(), step, None)
    writer.close()
    path = tmp_path / "ep.mp4"
    assert path.exists()
    assert path.stat().st_size > 0


def test_burn_text_changes_pixels_in_the_text_region():
    frame = _frame()
    burned = _burn_text(frame, "step 0 | live")
    # The top-left corner (where text is drawn) must differ from the untouched frame;
    # the far bottom-right corner must not (128px gives ample clearance from
    # top-left-anchored text of a dozen characters in the default bitmap font).
    assert not np.array_equal(frame[:16, :16], burned[:16, :16])
    np.testing.assert_array_equal(frame[-4:, -4:], burned[-4:, -4:])


def test_add_frame_perturbation_banner_only_within_the_window():
    writer = RolloutVideoWriter(Path("unused.mp4"), conditioning="live")
    writer.add_frame(_frame(), step_index=10, steps_since_perturbation=0)
    inside = writer._frames[-1]
    writer.add_frame(_frame(), step_index=10, steps_since_perturbation=None)
    outside = writer._frames[-1]
    assert not np.array_equal(inside, outside)


def test_add_frame_banner_stops_after_the_window():
    writer = RolloutVideoWriter(Path("unused.mp4"), conditioning="live")
    writer.add_frame(_frame(), step_index=0, steps_since_perturbation=PERTURBATION_BANNER_FRAMES - 1)
    still_banner = writer._frames[-1]
    writer.add_frame(_frame(), step_index=0, steps_since_perturbation=PERTURBATION_BANNER_FRAMES)
    no_longer_banner = writer._frames[-1]
    assert not np.array_equal(still_banner, no_longer_banner)


def test_add_frame_applies_orientation_correction():
    """agentview_upright must run before burning text — a marker pixel placed in the
    raw frame's top-left corner should land in the diagonally-opposite (bottom-right)
    corner, same technique as
    test_observations.py::test_env_frames_are_rotated_180. A 128px frame keeps that
    corner well clear of the top-left-anchored burned text."""
    raw = _frame()
    raw[0, 0] = (255, 255, 255)   # top-left of the raw (un-rotated) frame

    writer = RolloutVideoWriter(Path("unused.mp4"), conditioning="live")
    writer.add_frame(raw, step_index=0, steps_since_perturbation=None)
    burned = writer._frames[-1]

    assert tuple(burned[-1, -1]) == (255, 255, 255)
    assert tuple(burned[0, 0]) == (0, 0, 0)
