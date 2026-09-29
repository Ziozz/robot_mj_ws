from pathlib import Path
import tempfile
import tomllib
import unittest

import numpy as np

from robot_mj.control import JointController
from robot_mj.interfaces import ControlMode, JointCommand
from robot_mj.robots import RobotDescription
from robot_mj.sim import MujocoSimulation
from robot_mj.tasks import build_g1_pick_place_scene


ROOT = Path(__file__).resolve().parents[1]


def make_simulation(tmp_path: Path):
    robot = RobotDescription.from_toml(ROOT / "configs/robots/g1.toml", ROOT)
    scene = build_g1_pick_place_scene(
        robot.model_path,
        tmp_path / "scene.xml",
        ROOT / "configs/tasks/g1_pick_place.toml",
    )
    return robot, MujocoSimulation(scene, robot)


class G1SmokeTest(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self.temporary_directory.name)

    def tearDown(self):
        self.temporary_directory.cleanup()

    def test_g1_scene_and_groups_load(self):
        robot, simulation = make_simulation(self.tmp_path)
        self.assertEqual(len(robot.group("right_arm")), 7)
        self.assertEqual(len(robot.group("right_hand")), 7)
        self.assertEqual(simulation.joint_state("right_arm").position.shape, (7,))
        np.testing.assert_allclose(
            simulation.body_pose("pick_cylinder").position,
            [0.34, -0.24, 0.802],
        )
        task_geom_ids = {
            simulation.model.geom(name).id
            for name in (
                "table", "barrier", "pick_box_geom",
                "pick_cylinder_geom", "pick_sphere_geom",
            )
        }
        robot_contacts = [
            contact for contact in simulation.data.contact
            if (contact.geom1 in task_geom_ids) != (contact.geom2 in task_geom_ids)
        ]
        self.assertEqual(robot_contacts, [], "Robot starts intersecting a task geom")

    def test_all_three_control_modes_produce_bounded_effort(self):
        robot, simulation = make_simulation(self.tmp_path)
        controller = JointController(simulation, robot, "right_arm")
        state = simulation.joint_state("right_arm")
        commands = (
            JointCommand(controller.names, state.position + 0.01, ControlMode.POSITION),
            JointCommand(controller.names, np.full(7, 0.1), ControlMode.VELOCITY),
            JointCommand(controller.names, np.full(7, 100.0), ControlMode.TORQUE),
        )
        for command in commands:
            effort = controller.update(command)
            self.assertTrue(np.all(np.abs(effort) <= controller.gains.effort_limit))
            simulation.step()

    def test_group_effort_does_not_overwrite_other_arm(self):
        robot, simulation = make_simulation(self.tmp_path)
        right = robot.group("right_arm")
        left = robot.group("left_arm")
        simulation.set_generalized_effort(left, np.ones(7))
        simulation.set_generalized_effort(right, np.full(7, 2.0))
        for name in left:
            dof = simulation._dof[name]
            self.assertEqual(simulation.data.qfrc_applied[dof], 1.0)

    def test_position_mode_holds_a_small_target(self):
        robot, simulation = make_simulation(self.tmp_path)
        controller = JointController(simulation, robot, "right_arm")
        target = simulation.joint_state("right_arm").position
        target[0] += 0.1
        command = JointCommand(controller.names, target, ControlMode.POSITION)
        for _ in range(1000):
            controller.update(command)
            simulation.step()
        error = target - simulation.joint_state("right_arm").position
        self.assertLess(np.max(np.abs(error)), 1e-3)

    def test_place_goal_footprint_is_clear(self):
        with (ROOT / "configs/tasks/g1_pick_place.toml").open("rb") as stream:
            task = tomllib.load(stream)
        goal = np.asarray(task["place_goal"]["position"][:2])
        goal_radius = float(task["place_goal"]["radius"])
        occupied = [*task.get("obstacles", []), *task["objects"]]
        for item in occupied:
            position = np.asarray(item["position"][:2])
            if item["shape"] == "box":
                footprint_radius = float(np.linalg.norm(item["size"][:2]))
            else:
                footprint_radius = float(item["size"][0])
            clearance = np.linalg.norm(position - goal)
            self.assertGreater(
                clearance,
                goal_radius + footprint_radius + 0.01,
                f"{item['name']} overlaps the place-goal footprint",
            )


if __name__ == "__main__":
    unittest.main()
