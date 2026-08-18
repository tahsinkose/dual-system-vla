"""Rollout video writer with a per-frame diagnostic overlay.

No generic video writer exists elsewhere in the repo (every script hand-rolls one with
imageio.v2.mimsave); this is the first one that needs to burn text into frames, which
scripts/replay_episode.py's write_video does not.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from src.utils import agentview_upright

PERTURBATION_BANNER_FRAMES = 20   # frames after the trigger step that show "PERTURBED"


class RolloutVideoWriter:
    """Collects one episode's frames in memory, writes one mp4 on close().

    A LIBERO rollout is at most a few hundred 256x256 uint8 frames — comfortably
    memory-resident per episode; nothing here needs to stream to disk incrementally.
    """

    def __init__(self, path: Path, conditioning: str, fps: int = 20) -> None:
        self._path = Path(path)
        self._conditioning = conditioning
        self._fps = fps
        self._frames: list[np.ndarray] = []

    @property
    def path(self) -> Path:
        return self._path

    def add_frame(self, raw_agentview_frame: np.ndarray, step_index: int,
                  steps_since_perturbation: int | None) -> None:
        """`raw_agentview_frame`: HWC uint8, straight from obs['agentview_image'],
        not yet corrected for orientation — `agentview_upright` is applied here.

        `steps_since_perturbation`: `current_step - trigger_step` if a perturbation
        has fired, else None. A "PERTURBED" banner is drawn while
        `0 <= steps_since_perturbation < PERTURBATION_BANNER_FRAMES`.
        """
        frame = agentview_upright(raw_agentview_frame).copy()
        text = f"step {step_index} | {self._conditioning}"
        if steps_since_perturbation is not None and 0 <= steps_since_perturbation < PERTURBATION_BANNER_FRAMES:
            text += " | PERTURBED"
        self._frames.append(_burn_text(frame, text))

    def close(self) -> None:
        """Writes the mp4. No-op — does not create an empty file — if no frames were
        added (e.g. an episode that raised before the first step)."""
        if not self._frames:
            return
        import imageio.v2 as imageio
        self._path.parent.mkdir(parents=True, exist_ok=True)
        imageio.mimsave(str(self._path), self._frames, fps=self._fps)


def _burn_text(frame: np.ndarray, text: str) -> np.ndarray:
    """Draw `text` into the top-left corner of `frame` (HWC uint8)."""
    from PIL import Image, ImageDraw

    image = Image.fromarray(frame)
    draw = ImageDraw.Draw(image)
    # A thin black outline behind white text keeps it legible over both light and
    # dark parts of the scene, without depending on a specific font file being
    # installed (the default bitmap font is used deliberately, for portability).
    x, y = 4, 4
    for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        draw.text((x + dx, y + dy), text, fill=(0, 0, 0))
    draw.text((x, y), text, fill=(255, 255, 255))
    return np.asarray(image)
