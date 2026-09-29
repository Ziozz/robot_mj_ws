"""Transport-independent messages used by simulation, tasks and ROS adapters."""

from dataclasses import dataclass
from enum import Enum

import numpy as np
from numpy.typing import NDArray


class ControlMode(str, Enum):
    """Meaning of a joint command; values intentionally match future topics."""

    POSITION = "position"
    VELOCITY = "velocity"
    TORQUE = "torque"


@dataclass(frozen=True)
class JointState:
    names: tuple[str, ...]
    position: NDArray[np.float64]
    velocity: NDArray[np.float64]
    effort: NDArray[np.float64]
    stamp: float


@dataclass(frozen=True)
class JointCommand:
    names: tuple[str, ...]
    values: NDArray[np.float64]
    mode: ControlMode

    def __post_init__(self) -> None:
        values = np.asarray(self.values, dtype=float)
        if values.shape != (len(self.names),):
            raise ValueError(
                f"Expected {len(self.names)} command values, got {values.shape}"
            )
        object.__setattr__(self, "values", values)


@dataclass(frozen=True)
class Pose:
    """Rigid pose in a named frame, using a wxyz quaternion like MuJoCo."""

    position: NDArray[np.float64]
    quaternion: NDArray[np.float64]
    frame_id: str = "world"
    stamp: float = 0.0

    def __post_init__(self) -> None:
        position = np.asarray(self.position, dtype=float)
        quaternion = np.asarray(self.quaternion, dtype=float)
        if position.shape != (3,) or quaternion.shape != (4,):
            raise ValueError("Pose requires position (3,) and quaternion (4,)")
        norm = np.linalg.norm(quaternion)
        if norm < 1e-12:
            raise ValueError("Pose quaternion cannot be zero")
        object.__setattr__(self, "position", position)
        object.__setattr__(self, "quaternion", quaternion / norm)

