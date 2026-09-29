import unittest

import numpy as np

from robot_mj.interfaces import ControlMode, JointCommand, Pose


class InterfaceTest(unittest.TestCase):
    def test_pose_normalizes_quaternion(self):
        pose = Pose(np.zeros(3), np.array([2.0, 0.0, 0.0, 0.0]))
        np.testing.assert_allclose(pose.quaternion, [1.0, 0.0, 0.0, 0.0])

    def test_joint_command_rejects_wrong_length(self):
        with self.assertRaises(ValueError):
            JointCommand(("j1", "j2"), np.zeros(1), ControlMode.POSITION)


if __name__ == "__main__":
    unittest.main()
