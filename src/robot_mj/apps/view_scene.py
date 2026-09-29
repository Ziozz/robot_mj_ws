"""Build and inspect the first G1 pick-and-place scene."""

import argparse
from pathlib import Path
import time

import mujoco.viewer

from robot_mj.robots import RobotDescription
from robot_mj.sim import MujocoSimulation
from robot_mj.tasks import build_g1_pick_place_scene


def project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--seconds", type=float, default=0.2)
    args = parser.parse_args()

    root = project_root()
    robot = RobotDescription.from_toml(root / "configs/robots/g1.toml", root)
    scene = build_g1_pick_place_scene(
        robot.model_path,
        root / "generated/g1_pick_place.xml",
        root / "configs/tasks/g1_pick_place.toml",
    )
    simulation = MujocoSimulation(scene, robot)
    print(f"Loaded {robot.name}: nq={simulation.model.nq}, nv={simulation.model.nv}")
    print(f"Right arm joints: {robot.group('right_arm')}")
    for name in ("pick_box", "pick_cylinder", "pick_sphere"):
        print(f"{name} truth pose: {simulation.body_pose(name)}")

    if args.headless:
        steps = max(1, round(args.seconds / simulation.model.opt.timestep))
        simulation.step(steps)
        print(f"Headless smoke test passed at t={simulation.data.time:.3f}s")
        return

    with mujoco.viewer.launch_passive(simulation.model, simulation.data) as viewer:
        while viewer.is_running():
            start = time.perf_counter()
            simulation.step()
            viewer.sync()
            remaining = simulation.model.opt.timestep - (time.perf_counter() - start)
            if remaining > 0:
                time.sleep(remaining)


if __name__ == "__main__":
    main()
