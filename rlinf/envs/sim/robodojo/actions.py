# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Physical actions independent of agent tool schemas."""

from dataclasses import dataclass

import numpy as np


class InvalidAction(ValueError):
    """A correctable physical target or motion-planning rejection."""


@dataclass(frozen=True)
class JointControls:
    """Absolute native joint controls, shaped (H, A), angles in radians.

    Native ordering is left arm, right arm, left gripper, right gripper.
    Grippers use opening fractions in [0, 1]. The runtime validates arm widths.
    """

    values: np.ndarray

    def __post_init__(self) -> None:
        values = np.array(self.values, dtype=np.float64, copy=True)
        if values.ndim != 2 or not all(values.shape) or not np.isfinite(values).all():
            raise InvalidAction(
                "Expected finite joint controls with shape (positive H, A)."
            )
        if np.any((values[:, -2:] < 0) | (values[:, -2:] > 1)):
            raise InvalidAction("Expected scalar gripper openings in [0, 1].")
        object.__setattr__(self, "values", values)


@dataclass(frozen=True)
class JointTarget:
    """One absolute native joint target, interpolated by the controller."""

    values: np.ndarray

    def __post_init__(self) -> None:
        values = np.array(self.values, dtype=np.float64, copy=True)
        if values.ndim != 1:
            raise InvalidAction("Expected one native joint target with shape (A,).")
        checked = JointControls(values[None])
        object.__setattr__(self, "values", checked.values[0])


@dataclass(frozen=True)
class EefTarget:
    """Absolute EE targets in each arm's base frame, with optional grippers.

    Poses are (x, y, z, qw, qx, qy, qz), metres and unit wxyz quaternions.
    Omitted arms and grippers hold their measured joint values. IK for all
    requested arms must succeed before any physics executes.
    """

    left: tuple[float, ...] | None = None
    right: tuple[float, ...] | None = None
    left_gripper: float | None = None
    right_gripper: float | None = None

    def __post_init__(self) -> None:
        if all(
            getattr(self, name) is None
            for name in ("left", "right", "left_gripper", "right_gripper")
        ):
            raise InvalidAction("Expected at least one EEF or gripper target.")
        for side in ("left", "right"):
            pose = getattr(self, side)
            if pose is not None:
                pose = np.asarray(pose, dtype=np.float64)
                if (
                    pose.shape != (7,)
                    or not np.isfinite(pose).all()
                    or not np.isclose(np.linalg.norm(pose[3:]), 1, atol=1e-5)
                ):
                    raise InvalidAction(
                        "Expected a finite 7D pose with a unit wxyz quaternion."
                    )
                object.__setattr__(self, side, tuple(pose.tolist()))
            grip = getattr(self, side + "_gripper")
            if grip is not None and (
                isinstance(grip, bool) or not np.isfinite(grip) or not 0 <= grip <= 1
            ):
                raise InvalidAction("Expected gripper opening in [0, 1].")


def interpolate_joints(
    current: np.ndarray,
    target: np.ndarray,
    joint_step: float,
    gripper_step: float,
    max_steps: int,
) -> np.ndarray:
    """Generate bounded joint-space targets or reject an oversized motion."""
    current, target = np.asarray(current, dtype=float), np.asarray(target, dtype=float)
    if current.shape != target.shape or not np.isfinite(current).all():
        raise RuntimeError(
            "Expected finite native joint state matching the target shape."
        )
    limits = np.full(len(current), joint_step)
    limits[-2:] = gripper_step
    with np.errstate(over="ignore", invalid="ignore"):
        ratio = np.max(np.abs(target - current) / limits)
    if not np.isfinite(ratio) or ratio > max_steps:
        raise InvalidAction(
            f"Motion exceeds {max_steps} controls; split it into smaller moves."
        )
    count = max(1, int(np.ceil(ratio)))
    return current + (target - current) * (np.arange(1, count + 1)[:, None] / count)
