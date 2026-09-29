"""Run OMPL followed by synchronized TOPP-RA and save the result."""

from pathlib import Path

import numpy as np
from ompl import util as ou

from robot_mj.collision import HppFclCollisionChecker
from robot_mj.kinematics import PinocchioKinematics
from robot_mj.planning import OmplPlanner
from robot_mj.robots import RobotDescription
from robot_mj.tasks import build_g1_pick_place_scene
from robot_mj.trajectory import load_motion_settings, parameterize_toppra


def project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def main() -> None:
    ou.setLogLevel(ou.LOG_WARN)
    root = project_root()
    robot = RobotDescription.from_toml(root / "configs/robots/g1.toml", root)
    scene = build_g1_pick_place_scene(
        robot.model_path,
        root / "generated/g1_pick_place.xml",
        root / "configs/tasks/g1_pick_place.toml",
    )
    kinematics = PinocchioKinematics(scene, robot, "right_arm", "right_hand")
    checker = HppFclCollisionChecker(
        kinematics,
        root / "configs/tasks/g1_pick_place.toml",
        root / "configs/collision/g1.toml",
    )
    settings = load_motion_settings(root / "configs/motion/g1.toml", "right_arm")
    planner = OmplPlanner(
        kinematics.lower,
        kinematics.upper,
        checker,
        planner_range=settings.ompl_range,
        edge_joint_step=settings.edge_joint_step,
    )
    # A deterministic collision-free module-integration target. The grasp task
    # will replace this with a configuration returned by 6D grasp IK.
    goal = np.array(
        [-0.028127, -0.261151, 0.582303, 1.510702,
         -0.574489, -0.001821, -0.218212]
    )
    geometric = planner.plan(
        kinematics.home,
        goal,
        timeout=settings.ompl_timeout,
        simplify=settings.simplify,
    )
    trajectory = parameterize_toppra(
        geometric.waypoints,
        kinematics.names,
        settings,
        lower=kinematics.lower,
        upper=kinematics.upper,
        checker=checker,
    )
    destination = root / "generated/right_arm_trajectory.npz"
    np.savez(
        destination,
        joint_names=np.asarray(trajectory.joint_names),
        time=trajectory.time,
        position=trajectory.position,
        velocity=trajectory.velocity,
        acceleration=trajectory.acceleration,
    )
    print(
        f"OMPL: {geometric.raw_waypoint_count} raw -> "
        f"{geometric.simplified_waypoint_count} simplified waypoints"
    )
    print(
        f"TOPP-RA: {len(trajectory.time)} samples, "
        f"{trajectory.duration:.3f} s, dt={trajectory.control_period:.3f} s"
    )
    print(f"max |velocity|: {np.max(np.abs(trajectory.velocity), axis=0)}")
    print(f"max |acceleration|: {np.max(np.abs(trajectory.acceleration), axis=0)}")
    print(f"start velocity: {trajectory.velocity[0]}")
    print(f"end velocity:   {trajectory.velocity[-1]}")
    print(f"Saved: {destination}")


if __name__ == "__main__":
    main()

