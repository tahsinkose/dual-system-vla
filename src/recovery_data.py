"""Harvested recovery segments, served in the layout the training loop already consumes.

`scripts/harvest_recovery.py` writes one `.npz` per segment: a short window in which the
policy is perturbed off the demonstrated trajectory and driven back onto it. The training
loop reads batches from a `LeRobotDataset`, so a segment only becomes trainable once it
emits the same keys with the same conventions — otherwise the two sources disagree about
image scale or which frame System 2 is supposed to see, and the disagreement is silent.

The contract `split_batch` imposes, per sample:

    observation.images.image        (2, 3, H, W)  float32 in [0, 1]; index 0 is t-Δ, 1 is t
    observation.images.wrist_image  (3, H, W)     float32 in [0, 1]
    observation.state               (8,)          float32
    action                          (chunk, 7)    float32
    action_is_pad                   (chunk,)      bool
    task                            str

Two conversions are needed and neither is optional. Segments store frames as uint8 to
keep the harvest on disk, so they are scaled back to [0, 1] here; and they are already
canonical, having been written through `src/observations.from_env`, so no rotation is
applied — applying one would flip recovery frames relative to every demonstration frame.

Chunks run past the end of a short segment. A 19-step segment offers a full 16-step chunk
at only four of its steps, so the remainder are padded and `action_is_pad` marks them.
`masked_l1_loss` divides by the valid count, so a padded chunk supervises the steps it
actually has; `min_valid_actions` drops the tail where too few remain to be worth a
sample.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

MAIN_CAMERA = "observation.images.image"
WRIST_CAMERA = "observation.images.wrist_image"


class RecoverySegmentDataset(Dataset):
    """One sample per usable step of every harvested segment."""

    def __init__(self, root: Path | str, chunk_size: int, temporal_offset: int,
                 min_valid_actions: int = 4) -> None:
        self.root = Path(root)
        self.chunk_size = chunk_size
        self.temporal_offset = temporal_offset
        self.paths = sorted(self.root.glob("*.npz"))
        if not self.paths:
            raise FileNotFoundError(f"no recovery segments under {self.root}")

        # (segment, step) pairs, resolved once so __getitem__ is a single load and slice.
        self.index: list[tuple[int, int]] = []
        self.lengths: list[int] = []
        for segment, path in enumerate(self.paths):
            with np.load(path, allow_pickle=True) as data:
                length = len(data["state"])
            self.lengths.append(length)
            usable = max(0, length - min_valid_actions + 1)
            self.index.extend((segment, step) for step in range(usable))

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, item: int) -> dict:
        segment, step = self.index[item]
        with np.load(self.paths[segment], allow_pickle=True) as data:
            length = len(data["state"])
            stop = min(step + self.chunk_size, length)

            actions = np.zeros((self.chunk_size, data["action"].shape[1]), dtype=np.float32)
            actions[: stop - step] = data["action"][step:stop]
            is_pad = np.ones(self.chunk_size, dtype=bool)
            is_pad[: stop - step] = False

            # System 2 reads Δ steps back. Inside the first Δ steps of a segment there is
            # no earlier frame, so the oldest available one stands in — the alternative,
            # dropping those steps, discards the moment the perturbation lands.
            offset_step = max(0, step - self.temporal_offset)
            main = np.stack([data["image"][offset_step], data["image"][step]])

            return {
                MAIN_CAMERA: torch.from_numpy(main).float().div_(255.0),
                WRIST_CAMERA: torch.from_numpy(data["wrist_image"][step]).float().div_(255.0),
                "observation.state": torch.from_numpy(data["state"][step]).float(),
                "action": torch.from_numpy(actions),
                "action_is_pad": torch.from_numpy(is_pad),
                "task": str(data["task"]),
            }

    def summary(self) -> str:
        frames = sum(self.lengths)
        full = sum(max(0, n - self.chunk_size + 1) for n in self.lengths)
        return (f"{len(self.paths)} segments, {frames} frames, {len(self)} samples "
                f"({full} with a full {self.chunk_size}-step chunk)")
