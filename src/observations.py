"""One canonical observation, built identically from the dataset and the simulator.

Training reads the LeRobot dataset; evaluation reads the LIBERO environment. The two
describe the same robot in materially different ways, and every difference below is a
silent corruption if handled on only one path — the model would train on one
representation and be evaluated on another, with nothing raising an error:

======================  ==============================  ==============================
                        dataset                         environment
======================  ==============================  ==============================
image layout            CHW float32 in [0, 1]           HWC uint8 in [0, 255]
image orientation       upright                         **rotated 180 degrees**
wrist camera key        ``...images.wrist_image``       ``image2``
end-effector rotation   axis-angle, inside the 8-dim    quaternion, separate field
                        ``observation.state``
======================  ==============================  ==============================

The orientation row is the dangerous one. ``LiberoEnv.render()`` applies
``image[::-1, ::-1]`` "for visualization", but ``_format_raw_obs`` — what a *policy*
receives — passes frames through untouched.

"""

from __future__ import annotations

from dataclasses import dataclass

import torch

# What the model consumes, regardless of source.
CAMERA_KEYS: tuple[str, ...] = ("image", "image2")

# Dataset feature name -> canonical camera key.
DATASET_CAMERA_KEYS = {
    "observation.images.image": "image",
    "observation.images.wrist_image": "image2",
}

STATE_DIM = 8


@dataclass
class ModelObservation:
    """Canonical model input.

    images: canonical camera key -> ``(B, 3, H, W)`` float32 in [0, 1]
    state:  ``(B, 8)`` float32 — eef xyz, axis-angle rotation, two gripper joints
    """

    images: dict[str, torch.Tensor]
    state: torch.Tensor

    def to(self, device) -> "ModelObservation":
        return ModelObservation(
            images={k: v.to(device) for k, v in self.images.items()},
            state=self.state.to(device),
        )

    @property
    def batch_size(self) -> int:
        return self.state.shape[0]

    def validate(self) -> "ModelObservation":
        missing = [k for k in CAMERA_KEYS if k not in self.images]
        if missing:
            raise KeyError(f"missing canonical camera views {missing}; got {sorted(self.images)}")
        if self.state.ndim != 2 or self.state.shape[1] != STATE_DIM:
            raise ValueError(f"state must be (B, {STATE_DIM}); got {tuple(self.state.shape)}")
        for key, image in self.images.items():
            if image.ndim != 4 or image.shape[1] != 3:
                raise ValueError(f"{key} must be (B, 3, H, W); got {tuple(image.shape)}")
            if image.dtype != torch.float32:
                raise TypeError(f"{key} must be float32; got {image.dtype}")
        return self


def _as_batched_chw_float(array, *, rotate180: bool) -> torch.Tensor:
    """Normalise any frame layout to ``(B, 3, H, W)`` float32 in [0, 1]."""
    tensor = array if torch.is_tensor(array) else torch.as_tensor(array)
    if tensor.ndim == 3:
        tensor = tensor.unsqueeze(0)
    if tensor.ndim != 4:
        raise ValueError(f"expected a 3- or 4-dim frame, got {tuple(tensor.shape)}")

    if tensor.shape[-1] == 3 and tensor.shape[1] != 3:      # BHWC -> BCHW
        tensor = tensor.permute(0, 3, 1, 2)

    if rotate180:
        # Applied after the layout fix so the spatial axes are unambiguously (-2, -1).
        tensor = torch.flip(tensor, dims=(-2, -1))

    if tensor.dtype == torch.uint8:
        tensor = tensor.float() / 255.0
    else:
        tensor = tensor.float()
        if float(tensor.max()) > 1.5:   # tolerate float frames still in 0-255
            tensor = tensor / 255.0
    return tensor.contiguous()


def _as_batched_state(array) -> torch.Tensor:
    tensor = array if torch.is_tensor(array) else torch.as_tensor(array)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    return tensor.float()


def from_dataset(sample: dict) -> ModelObservation:
    """Build the canonical observation from a LeRobotDataset sample or batch.

    Dataset frames are already upright CHW floats, so only the wrist camera key is
    renamed. Both the dataset name and the canonical name are accepted, so a batch that
    has already been through this adapter passes unchanged.
    """
    images = {}
    for source, canonical in DATASET_CAMERA_KEYS.items():
        if source in sample:
            images[canonical] = _as_batched_chw_float(sample[source], rotate180=False)
    for canonical in CAMERA_KEYS:
        if canonical not in images and canonical in sample:
            images[canonical] = _as_batched_chw_float(sample[canonical], rotate180=False)

    if "observation.state" not in sample:
        raise KeyError(f"sample has no 'observation.state'; got {sorted(sample)}")
    return ModelObservation(images, _as_batched_state(sample["observation.state"])).validate()


def from_env(observation: dict) -> ModelObservation:
    """Build the canonical observation from a LIBERO environment observation.

    Accepts either the nested form returned by ``lerobot.envs.make_env``
    (``{"pixels": {...}, "robot_state": {...}}``) or the flat form from LIBERO's own
    ``OffScreenRenderEnv`` (``agentview_image``, ``robot0_eef_pos``, ...).

    Frames are rotated 180 degrees here — see the module docstring. This is the single
    place that correction exists, so training and evaluation cannot disagree about it.
    """
    pixels, robot_state = _split_env_observation(observation)

    images = {}
    for canonical, candidates in (
        ("image", ("image", "agentview_image")),
        ("image2", ("image2", "wrist_image", "robot0_eye_in_hand_image")),
    ):
        for name in candidates:
            if name in pixels:
                images[canonical] = _as_batched_chw_float(pixels[name], rotate180=True)
                break

    return ModelObservation(images, _env_state(robot_state)).validate()


def _split_env_observation(observation: dict) -> tuple[dict, dict]:
    if "pixels" in observation:
        return observation["pixels"], observation.get("robot_state", observation)
    return observation, observation


def _env_state(source: dict) -> torch.Tensor:
    """Rebuild the dataset's 8-dim state: eef xyz, axis-angle rotation, gripper joints.

    The environment reports rotation as a quaternion, so it is converted rather than
    copied. Verified exact against a dataset episode replayed from its recorded initial
    simulator state.
    """
    from robosuite.utils import transform_utils

    position = _lookup(source, ("eef/pos", "robot0_eef_pos"), ("eef", "pos"))
    quaternion = _lookup(source, ("eef/quat", "robot0_eef_quat"), ("eef", "quat"))
    gripper = _lookup(source, ("gripper/qpos", "robot0_gripper_qpos"), ("gripper", "qpos"))

    position = _as_batched_state(position)
    quaternion = _as_batched_state(quaternion)
    gripper = _as_batched_state(gripper)

    axis_angle = torch.stack([
        torch.as_tensor(transform_utils.quat2axisangle(q.numpy()), dtype=torch.float32)
        for q in quaternion
    ])
    return torch.cat([position, axis_angle, gripper], dim=1).float()


def _lookup(source: dict, flat_names: tuple[str, ...], nested_path: tuple[str, ...]):
    for name in flat_names:
        if name in source:
            return source[name]
    node = source
    for key in nested_path:
        if not isinstance(node, dict) or key not in node:
            raise KeyError(
                f"cannot find {'/'.join(nested_path)} (or {flat_names}) in the "
                f"environment observation; got {sorted(source)}"
            )
        node = node[key]
    return node
