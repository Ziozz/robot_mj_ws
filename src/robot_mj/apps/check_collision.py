"""Standalone diagnostics for state, self and edge collision checking."""

from pathlib import Path

import numpy as np

from robot_mj.collision import HppFclCollisionChecker
from robot_mj.kinematics import PinocchioKinematics
from robot_mj.robots import RobotDescription
from robot_mj.tasks import build_g1_pick_place_scene


def project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def main() -> None:
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

    home_report = checker.check(kinematics.home)
    print(f"Unique robot collision geometries: {len(checker._robot_geometry_ids)}")
    print(f"Geometries moving with right_arm: {len(checker._active_geometry_ids)}")
    print(f"Home collision-free: {not home_report.in_collision}")

    folded = np.array(
        [0.523848, -0.008739, 0.784597, -0.781910,
         -0.332093, -1.480064, -0.019403]
    )
    folded_report = checker.check(folded)
    print("Folded-pose contacts:")
    for contact in folded_report.contacts:
        print(f"  [{contact.category}] {contact.first} <-> {contact.second}")
    print(
        "Home-to-folded edge collision-free: "
        f"{checker.edge_is_valid(kinematics.home, folded)}"
    )


if __name__ == "__main__":
    main()
