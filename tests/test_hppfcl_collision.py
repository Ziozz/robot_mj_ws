import importlib.util
from pathlib import Path
import tempfile
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
COLLISION_AVAILABLE = (
    importlib.util.find_spec("pinocchio") is not None
    and importlib.util.find_spec("coal") is not None
)


@unittest.skipUnless(COLLISION_AVAILABLE, "install planning extras for HPP-FCL")
class HppFclCollisionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import coal

        from robot_mj.collision import HppFclCollisionChecker
        from robot_mj.kinematics import PinocchioKinematics
        from robot_mj.robots import RobotDescription
        from robot_mj.tasks import build_g1_pick_place_scene

        cls.coal = coal
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

    @classmethod
    def tearDownClass(cls):
        cls.temporary_directory.cleanup()

    def test_home_is_collision_free(self):
        report = self.checker.check(self.kinematics.home)
        self.assertFalse(report.in_collision, report.contacts)

    def test_known_folded_posture_has_self_collision(self):
        q = np.array(
            [0.523848, -0.008739, 0.784597, -0.781910,
             -0.332093, -1.480064, -0.019403]
        )
        report = self.checker.check(q)
        self.assertTrue(any(contact.category == "self" for contact in report.contacts))
        self.assertFalse(self.checker.edge_is_valid(self.kinematics.home, q))

    def test_target_contact_is_allowed_only_for_touch_link_in_grasp(self):
        from robot_mj.collision import CollisionPhase

        checker = self.checker
        checker.check(self.kinematics.home)  # Update robot geometry placements.
        geometry_id = next(
            geometry_id
            for geometry_id in checker._active_geometry_ids
            if checker._link_name(geometry_id) == "right_hand_index_1_link"
        )
        robot_geometry = checker.geometry_model.geometryObjects[geometry_id].geometry
        # Pick a point near the distal pad.  Vertex 100 lies inside the
        # neighbouring index_0 mesh as well, which tests two contacts instead
        # of the intended semantic exception for index_1 only.
        local_surface_point = np.asarray(robot_geometry.vertices())[763]
        transform = checker._robot_transform(geometry_id)
        world_surface_point = (
            transform.getRotation() @ local_surface_point + transform.getTranslation()
        )
        target = checker._environment[checker.target_name]
        original_geometry, original_transform = target.geometry, target.transform
        try:
            target.geometry = self.coal.Sphere(0.00001)
            target.transform = self.coal.Transform3s(np.eye(3), world_surface_point)
            free = checker.check(self.kinematics.home, CollisionPhase.FREE_SPACE)
            grasp = checker.check(self.kinematics.home, CollisionPhase.GRASP)
            self.assertTrue(free.in_collision)
            self.assertTrue(
                any(contact.first == "right_hand_index_1_link" for contact in free.contacts)
            )
            self.assertFalse(grasp.in_collision, grasp.contacts)
        finally:
            target.geometry, target.transform = original_geometry, original_transform


if __name__ == "__main__":
    unittest.main()
