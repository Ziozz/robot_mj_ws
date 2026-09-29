import importlib.util
from pathlib import Path
import tempfile
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
PLANNING_AVAILABLE = all(
    importlib.util.find_spec(module) is not None
    for module in ("coal", "ompl", "pinocchio", "toppra")
)


@unittest.skipUnless(PLANNING_AVAILABLE, "install planning extras")
class PlanningTrajectoryTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from ompl import util as ou

        from robot_mj.collision import HppFclCollisionChecker
        from robot_mj.kinematics import PinocchioKinematics
        from robot_mj.planning import OmplPlanner
        from robot_mj.robots import RobotDescription
        from robot_mj.tasks import build_g1_pick_place_scene
        from robot_mj.trajectory import load_motion_settings

        ou.setLogLevel(ou.LOG_WARN)
        cls.temporary_directory = tempfile.TemporaryDirectory()
        robot = RobotDescription.from_toml(ROOT / "configs/robots/g1.toml", ROOT)
        scene = build_g1_pick_place_scene(
            robot.model_path,
            Path(cls.temporary_directory.name) / "scene.xml",
            ROOT / "configs/tasks/g1_pick_place.toml",
        )
        cls.kinematics = PinocchioKinematics(
            scene, robot, "right_arm", "right_hand"
        )
        cls.checker = HppFclCollisionChecker(
            cls.kinematics,
            ROOT / "configs/tasks/g1_pick_place.toml",
            ROOT / "configs/collision/g1.toml",
        )
        cls.settings = load_motion_settings(
            ROOT / "configs/motion/g1.toml", "right_arm"
        )
        cls.planner = OmplPlanner(
            cls.kinematics.lower,
            cls.kinematics.upper,
            cls.checker,
            planner_range=cls.settings.ompl_range,
            edge_joint_step=cls.settings.edge_joint_step,
        )
        cls.goal = np.array(
            [-0.028127, -0.261151, 0.582303, 1.510702,
             -0.574489, -0.001821, -0.218212]
        )

    @classmethod
    def tearDownClass(cls):
        cls.temporary_directory.cleanup()

    def test_ompl_to_toppra_synchronized_pipeline(self):
        from robot_mj.trajectory import parameterize_toppra

        geometric = self.planner.plan(
            self.kinematics.home,
            self.goal,
            timeout=self.settings.ompl_timeout,
            simplify=self.settings.simplify,
        )
        np.testing.assert_allclose(geometric.waypoints[0], self.kinematics.home)
        np.testing.assert_allclose(geometric.waypoints[-1], self.goal)
        trajectory = parameterize_toppra(
            geometric.waypoints,
            self.kinematics.names,
            self.settings,
            lower=self.kinematics.lower,
            upper=self.kinematics.upper,
            checker=self.checker,
        )
        np.testing.assert_allclose(
            np.diff(trajectory.time), self.settings.control_period, atol=1e-12
        )
        np.testing.assert_allclose(trajectory.position[0], self.kinematics.home)
        np.testing.assert_allclose(trajectory.position[-1], self.goal)
        np.testing.assert_allclose(trajectory.velocity[[0, -1]], 0.0, atol=1e-12)
        self.assertTrue(
            np.all(np.abs(trajectory.velocity) <= self.settings.velocity + 1e-7)
        )
        self.assertTrue(
            np.all(
                np.abs(trajectory.acceleration)
                <= self.settings.acceleration + 1e-6
            )
        )

        # With this all-joints-moving test path, every joint leaves its start
        # on the first shared tick and reaches its target on the same final tick.
        moved = np.abs(trajectory.position - trajectory.position[0]) > 1e-12
        first_move = np.argmax(moved, axis=0)
        self.assertTrue(np.all(first_move == first_move[0]))
        before_final = np.abs(trajectory.position[-2] - trajectory.position[-1])
        self.assertTrue(np.all(before_final > 1e-12))


if __name__ == "__main__":
    unittest.main()

