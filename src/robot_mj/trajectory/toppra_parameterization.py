"""Synchronized fixed-period TOPP-RA trajectory generation and validation."""

from dataclasses import dataclass
from pathlib import Path
import tomllib

import numpy as np
from scipy.interpolate import PchipInterpolator
import toppra as ta
import toppra.algorithm as algo
import toppra.constraint as constraint

from robot_mj.collision import CollisionPhase, HppFclCollisionChecker


@dataclass(frozen=True)
class MotionSettings:
    velocity: np.ndarray
    acceleration: np.ndarray
    safety: float
    control_period: float
    ompl_timeout: float
    ompl_range: float
    edge_joint_step: float
    simplify: bool


@dataclass(frozen=True)
class SynchronizedTrajectory:
    joint_names: tuple[str, ...]
    time: np.ndarray
    position: np.ndarray
    velocity: np.ndarray
    acceleration: np.ndarray
    control_period: float

    @property
    def duration(self) -> float:
        return float(self.time[-1])


def load_motion_settings(path: str | Path, group: str) -> MotionSettings:
    with Path(path).open("rb") as stream:
        raw = tomllib.load(stream)[group]
    ompl = raw["ompl"]
    result = MotionSettings(
        velocity=np.asarray(raw["velocity"], dtype=float),
        acceleration=np.asarray(raw["acceleration"], dtype=float),
        safety=float(raw["safety"]),
        control_period=float(raw["control_period"]),
        ompl_timeout=float(ompl["timeout"]),
        ompl_range=float(ompl["range"]),
        edge_joint_step=float(ompl["edge_joint_step"]),
        simplify=bool(ompl["simplify"]),
    )
    if np.any(result.velocity <= 0.0) or np.any(result.acceleration <= 0.0):
        raise ValueError("Velocity and acceleration limits must be positive")
    if not 0.0 < result.safety <= 1.0 or result.control_period <= 0.0:
        raise ValueError("Invalid safety factor or control period")
    return result


def _remove_duplicate_waypoints(waypoints: np.ndarray) -> np.ndarray:
    lengths = np.linalg.norm(np.diff(waypoints, axis=0), axis=1)
    return waypoints[np.r_[True, lengths > 1e-10]]


def _densify_polyline(waypoints: np.ndarray, maximum_step: float) -> np.ndarray:
    """Add knots along planner edges so the cubic stays near the safe polyline.

    A sparse cubic through OMPL vertices can cut across an obstacle even when
    every original straight edge is valid. Dense collinear knots substantially
    limit that corner cutting; the resulting continuous curve is still fully
    collision-rechecked below.
    """
    dense = [waypoints[0]]
    for start, goal in zip(waypoints[:-1], waypoints[1:]):
        count = max(1, int(np.ceil(np.max(np.abs(goal - start)) / maximum_step)))
        dense.extend(start + alpha * (goal - start)
                     for alpha in np.linspace(0.0, 1.0, count + 1)[1:])
    return np.asarray(dense)


def parameterize_toppra(
    waypoints: np.ndarray,
    joint_names: tuple[str, ...],
    settings: MotionSettings,
    *,
    lower: np.ndarray,
    upper: np.ndarray,
    checker: HppFclCollisionChecker | None = None,
    phase: CollisionPhase = CollisionPhase.FREE_SPACE,
    joint_positions: dict[str, float] | None = None,
) -> SynchronizedTrajectory:
    """Generate one shared rest-to-rest timeline for every joint.

    TOPP-RA returns a continuous q(t). The final timeline is stretched (never
    accelerated) to an integer number of controller periods, matching the
    fixed-period command style used by upperbody-bridge.
    """
    waypoints = _remove_duplicate_waypoints(np.asarray(waypoints, dtype=float))
    if waypoints.ndim != 2 or len(waypoints) < 2:
        raise ValueError("At least two distinct path waypoints are required")
    dof = waypoints.shape[1]
    if len(joint_names) != dof:
        raise ValueError("Joint names do not match path dimension")
    for values, label in (
        (settings.velocity, "velocity"),
        (settings.acceleration, "acceleration"),
        (np.asarray(lower), "lower"),
        (np.asarray(upper), "upper"),
    ):
        if values.shape != (dof,):
            raise ValueError(f"{label} must contain {dof} values")

    waypoints = _densify_polyline(
        waypoints, max(0.005, settings.edge_joint_step * 0.25)
    )

    chord = np.linalg.norm(np.diff(waypoints, axis=0), axis=1)
    path_coordinate = np.r_[0.0, np.cumsum(chord)]
    path_coordinate /= path_coordinate[-1]
    # Shape-preserving Hermite slopes avoid cubic overshoot past a joint limit
    # while retaining continuous velocity along the dense planner polyline.
    pchip = PchipInterpolator(path_coordinate, waypoints, axis=0)
    geometric_path = ta.SimplePath(
        path_coordinate, waypoints, pchip.derivative()(path_coordinate)
    )
    velocity = settings.velocity * settings.safety
    acceleration = settings.acceleration * settings.safety
    constraints = [
        constraint.JointVelocityConstraint(np.column_stack((-velocity, velocity))),
        constraint.JointAccelerationConstraint(
            np.column_stack((-acceleration, acceleration)),
            discretization_scheme=constraint.DiscretizationType.Interpolation,
        ),
    ]
    instance = algo.TOPPRA(constraints, geometric_path, solver_wrapper="seidel")
    continuous = instance.compute_trajectory(0.0, 0.0)
    if continuous is None:
        raise RuntimeError("TOPP-RA found no feasible rest-to-rest parameterization")

    # TOPP-RA enforces constraints on its path grid. A highly curved spline can
    # have a slightly larger sampled peak between grid points. If that happens,
    # globally stretch time (never speed it up) and resample at the same 2 ms
    # controller period. One scalar stretch preserves joint synchronization.
    output_duration = float(continuous.duration)
    for _ in range(8):
        sample_count = max(
            1, int(np.ceil(output_duration / settings.control_period))
        )
        output_duration = sample_count * settings.control_period
        time = np.arange(sample_count + 1, dtype=float) * settings.control_period
        scale = continuous.duration / output_duration
        query_time = time * scale
        position = np.asarray(continuous(query_time, 0))
        velocity_out = np.asarray(continuous(query_time, 1)) * scale
        acceleration_out = np.asarray(continuous(query_time, 2)) * scale**2
        velocity_ratio = float(np.max(
            np.abs(velocity_out) / settings.velocity[None, :]
        ))
        acceleration_ratio = float(np.max(
            np.abs(acceleration_out) / settings.acceleration[None, :]
        ))
        stretch = max(1.0, velocity_ratio, np.sqrt(acceleration_ratio))
        if stretch <= 1.0 + 1e-9:
            break
        output_duration *= 1.002 * stretch

    # Make the rest-to-rest contract exact at the serialized boundaries.
    position[0], position[-1] = waypoints[0], waypoints[-1]
    velocity_out[[0, -1], :] = 0.0
    if np.any(position < np.asarray(lower) - 1e-6) or np.any(
        position > np.asarray(upper) + 1e-6
    ):
        raise RuntimeError(
            "Timed spline violates a joint position limit: "
            f"min margin={np.min(position - np.asarray(lower))}, "
            f"max margin={np.min(np.asarray(upper) - position)}"
        )
    # Remove harmless floating-point excursions at an exactly active bound.
    position = np.clip(position, np.asarray(lower), np.asarray(upper))
    if np.any(np.abs(velocity_out) > settings.velocity + 1e-7):
        raise RuntimeError("Timed trajectory violates a velocity limit")
    if np.any(np.abs(acceleration_out) > settings.acceleration + 1e-6):
        raise RuntimeError(
            "Timed trajectory violates an acceleration limit: "
            f"peak={np.max(np.abs(acceleration_out), axis=0)}, "
            f"limit={settings.acceleration}"
        )

    if checker is not None:
        # Recheck the smoothed path, not merely the original OMPL polyline.
        anchor = 0
        for index in range(1, len(position)):
            far_enough = (
                np.max(np.abs(position[index] - position[anchor]))
                >= settings.edge_joint_step * 0.5
            )
            if far_enough or index == len(position) - 1:
                if not checker.edge_is_valid(
                    position[anchor],
                    position[index],
                    phase,
                    joint_positions=joint_positions,
                    max_joint_step=settings.edge_joint_step * 0.5,
                ):
                    raise RuntimeError(
                        f"TOPP-RA smoothed path collides between samples {anchor} and {index}"
                    )
                anchor = index
    return SynchronizedTrajectory(
        joint_names=joint_names,
        time=time,
        position=position,
        velocity=velocity_out,
        acceleration=acceleration_out,
        control_period=settings.control_period,
    )
