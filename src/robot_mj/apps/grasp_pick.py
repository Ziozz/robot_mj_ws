"""Plan and physically verify one G1 three-finger pick."""

import argparse
from pathlib import Path

import numpy as np
from ompl import util as ou

from robot_mj.collision import HppFclCollisionChecker
from robot_mj.grasp import GraspExecutor, GraspPlanner, GraspSettings
from robot_mj.kinematics import PinocchioKinematics
from robot_mj.robots import RobotDescription
from robot_mj.sim import MujocoSimulation
from robot_mj.tasks import build_g1_pick_place_scene
from robot_mj.trajectory import load_motion_settings


def project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def main() -> None:
    parser = argparse.ArgumentParser(description="Collision-checked physical G1 grasp")
    parser.add_argument("--headless", action="store_true", help="do not open viewer")
    parser.add_argument("--plan-only", action="store_true", help="skip physics execution")
    args = parser.parse_args()
    ou.setLogLevel(ou.LOG_WARN)
    root = project_root()
    robot = RobotDescription.from_toml(root / "configs/robots/g1.toml", root)
    scene = build_g1_pick_place_scene(
        robot.model_path, root / "generated/g1_pick_place.xml",
        root / "configs/tasks/g1_pick_place.toml",
    )
    settings = GraspSettings.from_toml(
        root / "configs/grasp/g1_three_finger.toml"
    )
    kinematics = PinocchioKinematics(
        scene, robot, settings.arm_group, settings.end_effector
    )
    checker = HppFclCollisionChecker(
        kinematics, root / "configs/tasks/g1_pick_place.toml",
        root / "configs/collision/g1.toml",
    )
    transit = load_motion_settings(root / "configs/motion/g1.toml", "right_arm")
    approach = load_motion_settings(
        root / "configs/motion/g1.toml", "right_arm_approach"
    )
    truth = MujocoSimulation(scene, robot)
    object_pose = truth.body_pose(settings.target_body)
    plan = GraspPlanner(
        robot, kinematics, checker, settings, transit, approach
    ).plan(object_pose)
    print("Grasp plan")
    print(f"  object xyz:       {object_pose.position}")
    print(f"  pregrasp xyz:     {plan.pregrasp_pose.position}")
    print(f"  grasp TCP xyz:    {plan.grasp_pose.position}")
    print(f"  valid IK branches:{plan.valid_ik_candidates}")
    print(
        f"  OMPL waypoints:   {plan.transit_geometric.raw_waypoint_count} -> "
        f"{plan.transit_geometric.simplified_waypoint_count}"
    )
    print(
        f"  timed phases:     {plan.transit.duration:.3f} s transit + "
        f"{plan.approach.duration:.3f} s approach"
    )
    output = root / "generated/g1_grasp_plan.npz"
    np.savez(
        output,
        transit_time=plan.transit.time,
        transit_position=plan.transit.position,
        approach_time=plan.approach.time,
        approach_position=plan.approach.position,
        pregrasp_pose=np.r_[plan.pregrasp_pose.position, plan.pregrasp_pose.quaternion],
        grasp_pose=np.r_[plan.grasp_pose.position, plan.grasp_pose.quaternion],
    )
    print(f"  saved:            {output}")
    if args.plan_only:
        return

    report = GraspExecutor(
        scene, robot, kinematics, checker, settings, approach
    ).execute(plan, show_viewer=not args.headless)
    print("\nPhysical execution")
    print(f"  planned peak |dq|:  {np.round(report.planned_peak_velocity, 3)} rad/s")
    print(f"  planned peak |ddq|: {np.round(report.planned_peak_acceleration, 3)} rad/s^2")
    print(
        f"  tracking RMS/peak:  {report.tracking_rms:.5f} / "
        f"{report.tracking_peak:.5f} rad"
    )
    print(f"  contacts before close: {report.preclose_contacts}")
    print(f"  forbidden contacts:    {report.forbidden_contacts}")
    print(f"  fingertip contacts:    {report.contact_links}")
    print(f"  cylinder-side contacts:{report.sidewall_contact_links}")
    print(f"  opposition cosine:     {report.opposition_cosine}")
    center_mm = None if report.contact_center_error is None else 1000 * report.contact_center_error
    print(f"  contact center error:  {center_mm} mm")
    print(f"  maximum penetration:   {1000*report.maximum_penetration:.3f} mm")
    print(f"  dual-contact fraction: {report.dual_contact_fraction:.3f}")
    print(
        f"  lift commanded/object/tool: {1000*report.commanded_lift:.1f} / "
        f"{1000*report.achieved_object_lift:.1f} / "
        f"{1000*report.achieved_tool_lift:.1f} mm"
    )
    print(f"  horizontal slip:       {1000*report.horizontal_slip:.3f} mm")
    print(f"  GRASP CONFIRMED:       {report.grasp_confirmed}")
    if not report.grasp_confirmed:
        raise RuntimeError("Physical grasp acceptance criteria were not met")


if __name__ == "__main__":
    main()
