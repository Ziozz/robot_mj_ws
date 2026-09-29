import importlib.util
from pathlib import Path
import tempfile
import unittest

import mujoco
import numpy as np

from robot_mj.robots import RobotDescription
from robot_mj.tasks import build_g1_pick_place_scene


ROOT = Path(__file__).resolve().parents[1]
PINOCCHIO_AVAILABLE = importlib.util.find_spec("pinocchio") is not None


@unittest.skipUnless(PINOCCHIO_AVAILABLE, "install the planning extra for Pinocchio")
class PinocchioKinematicsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from robot_mj.kinematics import PinocchioKinematics

        cls.temporary_directory = tempfile.TemporaryDirectory()
        cls.robot = RobotDescription.from_toml(ROOT / "configs/robots/g1.toml", ROOT)
        cls.scene = build_g1_pick_place_scene(
            cls.robot.model_path,
            Path(cls.temporary_directory.name) / "scene.xml",
            ROOT / "configs/tasks/g1_pick_place.toml",
        )
        cls.kinematics = PinocchioKinematics(
            cls.scene, cls.robot, "right_arm", "right_hand"
        )

    @classmethod
    def tearDownClass(cls):
        cls.temporary_directory.cleanup()

    def _mujoco_pose_and_jacobian(self, q):
        backend = self.kinematics
        simulation = backend.reference
        for name, value in zip(backend.names, q):
            simulation.data.qpos[simulation._qpos[name]] = value
        mujoco.mj_forward(simulation.model, simulation.data)
        body_id = mujoco.mj_name2id(
            simulation.model,
            mujoco.mjtObj.mjOBJ_BODY,
            backend.tool.parent_body,
        )
        rotation = simulation.data.xmat[body_id].reshape(3, 3)
        point = simulation.data.xpos[body_id] + rotation @ backend.tool.position
        jacobian_position = np.zeros((3, simulation.model.nv))
        jacobian_rotation = np.zeros((3, simulation.model.nv))
        mujoco.mj_jac(
            simulation.model,
            simulation.data,
            jacobian_position,
            jacobian_rotation,
            point,
            body_id,
        )
        columns = [simulation._dof[name] for name in backend.names]
        return point, np.vstack((jacobian_position[:, columns], jacobian_rotation[:, columns]))

    def test_fk_and_jacobian_match_mujoco(self):
        backend = self.kinematics
        samples = [
            backend.home,
            np.clip(
                backend.home + np.array([0.1, -0.08, 0.06, -0.1, 0.05, -0.04, 0.03]),
                backend.lower,
                backend.upper,
            ),
        ]
        for q in samples:
            expected_position, expected_jacobian = self._mujoco_pose_and_jacobian(q)
            np.testing.assert_allclose(
                backend.forward(q).position, expected_position, atol=2e-8
            )
            np.testing.assert_allclose(
                backend.jacobian(q), expected_jacobian, atol=2e-7
            )

    def test_multistart_full_pose_ik(self):
        from robot_mj.kinematics import solve_ik

        backend = self.kinematics
        known_configuration = np.clip(
            backend.home
            + np.array([-0.12, -0.10, 0.08, -0.16, 0.06, -0.05, 0.04]),
            backend.lower,
            backend.upper,
        )
        target = backend.forward(known_configuration)
        result = solve_ik(
            backend,
            target,
            seeds=(
                backend.home,
                backend.home + np.array([0.15, -0.15, 0.1, -0.1, 0.1, 0.0, -0.1]),
            ),
        )
        self.assertLess(result.position_error, 0.002)
        self.assertLess(result.orientation_error, 0.035)
        achieved = backend.forward(result.configuration)
        np.testing.assert_allclose(achieved.position, target.position, atol=0.002)


if __name__ == "__main__":
    unittest.main()
