"""Bounded, multi-start inverse kinematics for any configured joint group."""

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from robot_mj.interfaces import Pose

from .pinocchio_backend import PinocchioKinematics


@dataclass(frozen=True)
class IKResult:
    configuration: np.ndarray
    position_error: float
    orientation_error: float
    cost: float
    seed_index: int
    evaluations: int


def _rotation_from_pose(pose: Pose) -> np.ndarray:
    # scipy expects xyzw while the stable project interface uses MuJoCo's wxyz.
    w, x, y, z = pose.quaternion
    return Rotation.from_quat([x, y, z, w]).as_matrix()


def solve_ik(
    kinematics: PinocchioKinematics,
    target: Pose,
    seeds: Iterable[np.ndarray],
    *,
    position_tolerance: float = 0.002,
    orientation_tolerance: float = 0.035,
    position_scale: float = 10.0,
    posture_weight: float = 0.02,
    reference: np.ndarray | None = None,
) -> IKResult:
    """Solve a full-pose target and select the best valid multi-start result.

    The target is expressed in the same world frame returned by the MuJoCo
    backend. Joint limits are enforced inside scipy's bounded optimizer. A
    small posture residual resolves redundancy without overriding pose error.
    """
    target_rotation = _rotation_from_pose(target)
    reference = (
        kinematics.home.copy()
        if reference is None
        else np.asarray(reference, dtype=float)
    )
    if reference.shape != (kinematics.dof,):
        raise ValueError(f"Reference must have shape {(kinematics.dof,)}")

    seeds = [np.asarray(seed, dtype=float) for seed in seeds]
    if not seeds:
        raise ValueError("At least one IK seed is required")
    for seed in seeds:
        if seed.shape != (kinematics.dof,):
            raise ValueError(f"Every seed must have shape {(kinematics.dof,)}")

    def pose_error(q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        current = kinematics.forward(q)
        position = current.position - target.position
        current_rotation = _rotation_from_pose(current)
        orientation = Rotation.from_matrix(
            target_rotation.T @ current_rotation
        ).as_rotvec()
        return position, orientation

    def residual(q: np.ndarray) -> np.ndarray:
        position, orientation = pose_error(q)
        return np.r_[
            position_scale * position,
            orientation,
            posture_weight * (q - reference),
        ]

    def analytic_jacobian(q: np.ndarray) -> np.ndarray:
        geometric = kinematics.jacobian(q)
        current_rotation = _rotation_from_pose(kinematics.forward(q))
        orientation = Rotation.from_matrix(
            target_rotation.T @ current_rotation
        ).as_rotvec()
        theta = np.linalg.norm(orientation)
        skew = np.array(
            [
                [0.0, -orientation[2], orientation[1]],
                [orientation[2], 0.0, -orientation[0]],
                [-orientation[1], orientation[0], 0.0],
            ]
        )
        if theta < 1e-5:
            coefficient = 1.0 / 12.0 + theta**2 / 720.0
        else:
            coefficient = (
                1.0 - 0.5 * theta / np.tan(0.5 * theta)
            ) / theta**2
        inverse_left_jacobian = (
            np.eye(3) - 0.5 * skew + coefficient * (skew @ skew)
        )
        orientation_jacobian = (
            inverse_left_jacobian
            @ target_rotation.T
            @ geometric[3:, :]
        )
        return np.vstack(
            (
                position_scale * geometric[:3, :],
                orientation_jacobian,
                posture_weight * np.eye(kinematics.dof),
            )
        )

    valid: list[IKResult] = []
    diagnostics: list[str] = []
    for seed_index, seed in enumerate(seeds):
        seed = np.clip(seed, kinematics.lower, kinematics.upper)
        optimization = least_squares(
            residual,
            seed,
            jac=analytic_jacobian,
            bounds=(kinematics.lower, kinematics.upper),
            max_nfev=500,
            xtol=1e-10,
            ftol=1e-10,
            gtol=1e-10,
        )
        position, orientation = pose_error(optimization.x)
        position_error = float(np.linalg.norm(position))
        orientation_error = float(np.linalg.norm(orientation))
        diagnostics.append(
            f"seed {seed_index}: position={position_error:.4f} m, "
            f"orientation={orientation_error:.4f} rad"
        )
        if (
            position_error <= position_tolerance
            and orientation_error <= orientation_tolerance
        ):
            # Pose feasibility dominates; the final term selects a natural
            # redundant posture among otherwise valid solutions.
            posture_cost = float(np.linalg.norm(optimization.x - reference))
            valid.append(
                IKResult(
                    configuration=optimization.x.copy(),
                    position_error=position_error,
                    orientation_error=orientation_error,
                    cost=position_error + 0.1 * orientation_error + 0.01 * posture_cost,
                    seed_index=seed_index,
                    evaluations=optimization.nfev,
                )
            )
    if not valid:
        raise RuntimeError("No valid IK solution; " + "; ".join(diagnostics))
    return min(valid, key=lambda result: result.cost)

