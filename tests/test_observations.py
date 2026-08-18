"""Tests for the observation adapter.

The decisive test is `test_dataset_and_env_agree_on_the_same_scene`: it forces the
simulator into a known episode's recorded initial state and checks that both paths
produce the same canonical observation. Everything else here is a unit check; that one
is the actual train/eval parity guarantee.

Run with::

    python -m pytest tests/test_observations.py -v
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.env_setup import setup_env  # noqa: E402

os.environ.setdefault("MUJOCO_GL", "egl")
setup_env()

from src.observations import (  # noqa: E402
    CAMERA_KEYS,
    STATE_DIM,
    ModelObservation,
    from_dataset,
    from_env,
)

REPO_ID = "lerobot/libero_10"


# ------------------------------------------------------------------------- units


def synthetic_dataset_sample(batch: int = 2, size: int = 32) -> dict:
    return {
        "observation.images.image": torch.rand(batch, 3, size, size),
        "observation.images.wrist_image": torch.rand(batch, 3, size, size),
        "observation.state": torch.rand(batch, STATE_DIM),
    }


def synthetic_env_observation(batch: int = 2, size: int = 32) -> dict:
    return {
        "pixels": {
            "image": np.random.randint(0, 256, (batch, size, size, 3), dtype=np.uint8),
            "image2": np.random.randint(0, 256, (batch, size, size, 3), dtype=np.uint8),
        },
        "robot_state": {
            "eef": {"pos": np.zeros((batch, 3)), "quat": np.tile([0.0, 0.0, 0.0, 1.0], (batch, 1))},
            "gripper": {"qpos": np.zeros((batch, 2))},
        },
    }


def test_dataset_path_renames_the_wrist_camera():
    obs = from_dataset(synthetic_dataset_sample())
    assert set(obs.images) == set(CAMERA_KEYS)
    assert obs.state.shape == (2, STATE_DIM)


def test_env_path_normalises_layout_and_dtype():
    obs = from_env(synthetic_env_observation())
    assert set(obs.images) == set(CAMERA_KEYS)
    for image in obs.images.values():
        assert image.shape[1] == 3 and image.dtype == torch.float32
        assert 0.0 <= float(image.min()) and float(image.max()) <= 1.0


def test_env_frames_are_rotated_180():
    """The correction lives in exactly one place; assert it is actually applied."""
    frame = np.zeros((1, 4, 4, 3), dtype=np.uint8)
    frame[0, 0, 0] = 255  # top-left corner
    obs = from_env({"pixels": {"image": frame, "image2": frame},
                    "robot_state": {"eef": {"pos": np.zeros((1, 3)),
                                            "quat": np.tile([0.0, 0.0, 0.0, 1.0], (1, 1))},
                                    "gripper": {"qpos": np.zeros((1, 2))}}})
    image = obs.images["image"][0]
    assert float(image[0, -1, -1]) == pytest.approx(1.0), "rot180 not applied"
    assert float(image[0, 0, 0]) == pytest.approx(0.0)


def test_dataset_frames_are_not_rotated():
    """Dataset frames are already upright — rotating them would break parity."""
    frame = torch.zeros(1, 3, 4, 4)
    frame[0, :, 0, 0] = 1.0
    obs = from_dataset({"observation.images.image": frame,
                        "observation.images.wrist_image": frame,
                        "observation.state": torch.zeros(1, STATE_DIM)})
    assert float(obs.images["image"][0, 0, 0, 0]) == pytest.approx(1.0)


def test_single_samples_are_batched():
    obs = from_dataset({
        "observation.images.image": torch.rand(3, 16, 16),
        "observation.images.wrist_image": torch.rand(3, 16, 16),
        "observation.state": torch.rand(STATE_DIM),
    })
    assert obs.batch_size == 1
    assert obs.images["image"].shape == (1, 3, 16, 16)


def test_already_canonical_keys_pass_through():
    """Re-adapting an adapted batch must be a no-op, not a KeyError."""
    obs = from_dataset({
        "image": torch.rand(1, 3, 8, 8),
        "image2": torch.rand(1, 3, 8, 8),
        "observation.state": torch.rand(1, STATE_DIM),
    })
    assert set(obs.images) == set(CAMERA_KEYS)


def test_missing_camera_is_rejected():
    sample = synthetic_dataset_sample()
    del sample["observation.images.wrist_image"]
    with pytest.raises(KeyError, match="missing canonical camera views"):
        from_dataset(sample)


def test_wrong_state_width_is_rejected():
    with pytest.raises(ValueError, match=f"state must be \\(B, {STATE_DIM}\\)"):
        ModelObservation({k: torch.rand(1, 3, 8, 8) for k in CAMERA_KEYS},
                         torch.rand(1, 5)).validate()


def test_flat_libero_env_observation_is_accepted():
    """LIBERO's own OffScreenRenderEnv uses flat robosuite names, not the nested form."""
    obs = from_env({
        "agentview_image": np.zeros((16, 16, 3), dtype=np.uint8),
        "robot0_eye_in_hand_image": np.zeros((16, 16, 3), dtype=np.uint8),
        "robot0_eef_pos": np.zeros(3),
        "robot0_eef_quat": np.array([0.0, 0.0, 0.0, 1.0]),
        "robot0_gripper_qpos": np.zeros(2),
    })
    assert set(obs.images) == set(CAMERA_KEYS)
    assert obs.state.shape == (1, STATE_DIM)


# --------------------------------------------------------------- train/eval parity


@pytest.mark.slow
def test_dataset_and_env_agree_on_the_same_scene():
    """Both paths, one scene, same canonical observation.

    This is the guarantee the adapter exists for. The simulator is forced into a
    known episode's recorded initial simulator state, so the dataset's first frame and
    the environment's observation describe the *same* physical configuration. Any
    disagreement — orientation, scaling, layout, or the quaternion-to-axis-angle
    conversion — would mean the model is trained on one representation and evaluated
    on another, with nothing raising an error.
    """
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from libero.libero.envs import OffScreenRenderEnv

    from src.utils import (
        DUMMY_ACTION,
        TaskMapping,
        episode_slice,
        episodes_for_task,
        load_exact_init_state,
    )

    dataset = LeRobotDataset(REPO_ID)
    mapping = TaskMapping.from_dataset(dataset)

    episode = next((e for e in episodes_for_task(dataset, 5)
                    if load_exact_init_state(e) is not None), None)
    if episode is None:
        pytest.skip("no extracted initial states; run scripts/extract_init_states.py")

    start, _ = episode_slice(dataset, episode)
    reference = from_dataset(dataset[start])

    task = mapping.by_dataset_index(5)
    env = OffScreenRenderEnv(bddl_file_name=task.bddl, camera_heights=256,
                             camera_widths=256, control_freq=20, horizon=500)
    try:
        env.seed(0)
        env.reset()
        env.set_init_state(load_exact_init_state(episode))
        for _ in range(10):
            env.step(DUMMY_ACTION)
        candidate = from_env(env.env._get_observations())
    finally:
        env.close()

    # State must match essentially exactly — it is a deterministic transform.
    torch.testing.assert_close(candidate.state, reference.state, atol=1e-3, rtol=0)

    # Frames cannot be bit-identical: the dataset's are lossily video-encoded. The bar
    # is that they agree far better than any wrong orientation would allow — a rot180
    # error scores ~0.19, a vertical flip ~0.14.
    for key in CAMERA_KEYS:
        error = (candidate.images[key] - reference.images[key]).abs().mean().item()
        assert error < 0.08, f"{key} differs by {error:.3f}; orientation or scaling is wrong"
