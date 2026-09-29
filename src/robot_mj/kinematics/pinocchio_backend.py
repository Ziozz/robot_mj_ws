"""Pinocchio kinematics backend selected by robot group and end effector."""

from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
import pinocchio as pin

from robot_mj.interfaces import Pose
from robot_mj.robots import RobotDescription
from robot_mj.sim import MujocoSimulation


def _write_pinocchio_compatible_mjcf(scene: Path) -> Path:
    """Remove MuJoCo-only keyframes without changing the executable scene."""
    destination = scene.with_name(f"{scene.stem}_pinocchio.xml")
    tree = ET.parse(scene)
    root = tree.getroot()
    for keyframe in root.findall("keyframe"):
        root.remove(keyframe)
    ET.indent(tree, space="  ")
    tree.write(destination, encoding="unicode")
    return destination


class PinocchioKinematics:
    """Planner-facing FK and geometric Jacobian for one configured joint group.

    Pinocchio's MJCF parser and MuJoCo may choose different fixed-root world
    frames. A rigid world transform is calibrated once at the home posture, so
    users always receive poses in the same world frame as the simulator.
    """

    def __init__(
        self,
        scene: str | Path,
        robot: RobotDescription,
        group: str,
        end_effector: str,
    ) -> None:
        self.scene = Path(scene).resolve()
        self.robot = robot
        self.group = group
        self.names = robot.group(group)
        try:
            self.tool = robot.end_effectors[end_effector]
        except KeyError as exc:
            raise KeyError(f"Unknown end effector {end_effector!r}") from exc

        # MuJoCo remains the world-frame and joint-limit source of truth.
        self.reference = MujocoSimulation(self.scene, robot)
        state = self.reference.joint_state(group)
        self.home = state.position.copy()
        joint_ids = [self.reference._joint_ids[name] for name in self.names]
        self.lower = self.reference.model.jnt_range[joint_ids, 0].copy()
        self.upper = self.reference.model.jnt_range[joint_ids, 1].copy()

        self.pinocchio_scene = _write_pinocchio_compatible_mjcf(self.scene)
        self.model = pin.buildModelFromMJCF(str(self.pinocchio_scene))
        self.data = self.model.createData()
        self._joint_ids = [self.model.getJointId(name) for name in self.names]
        if any(joint_id == 0 for joint_id in self._joint_ids):
            missing = [
                name for name, joint_id in zip(self.names, self._joint_ids)
                if joint_id == 0
            ]
            raise ValueError(f"Pinocchio model is missing joints: {missing}")
        if any(self.model.joints[joint_id].nq != 1 for joint_id in self._joint_ids):
            raise ValueError("A controlled kinematic group must contain scalar joints")
        self._q_indices = np.asarray(
            [self.model.idx_qs[joint_id] for joint_id in self._joint_ids], dtype=int
        )
        self._v_indices = np.asarray(
            [self.model.idx_vs[joint_id] for joint_id in self._joint_ids], dtype=int
        )
        self._frame_id = self.model.getFrameId(self.tool.parent_body)
        if self._frame_id >= self.model.nframes:
            raise ValueError(f"Pinocchio frame {self.tool.parent_body!r} is missing")

        self._q_template = pin.neutral(self.model)
        self._copy_robot_home_to_pinocchio()
        self._tool_local = pin.SE3(
            self._rotation_from_wxyz(self.tool.quaternion), self.tool.position
        )
        pin_home = self._forward_pinocchio(self.home)
        mujoco_home = self._forward_mujoco(self.home)
        self._world_rotation = mujoco_home.rotation @ pin_home.rotation.T
        self._world_translation = (
            mujoco_home.translation
            - self._world_rotation @ pin_home.translation
        )

    @property
    def dof(self) -> int:
        return len(self.names)

    @staticmethod
    def _rotation_from_wxyz(quaternion: np.ndarray) -> np.ndarray:
        matrix = np.empty(9)
        mujoco.mju_quat2Mat(matrix, np.asarray(quaternion, dtype=float))
        return matrix.reshape(3, 3)

    def _copy_robot_home_to_pinocchio(self) -> None:
        """Copy every configured scalar robot joint, not just the active group."""
        copied: set[str] = set()
        for names in self.robot.groups.values():
            for name in names:
                if name in copied:
                    continue
                copied.add(name)
                pin_joint = self.model.getJointId(name)
                mj_joint = self.reference._joint_ids.get(name)
                if pin_joint == 0 or mj_joint is None:
                    continue
                if self.model.joints[pin_joint].nq != 1:
                    continue
                pin_q = self.model.idx_qs[pin_joint]
                mj_q = int(self.reference.model.jnt_qposadr[mj_joint])
                self._q_template[pin_q] = self.reference.data.qpos[mj_q]

    def _configuration(self, q: np.ndarray) -> np.ndarray:
        q = np.asarray(q, dtype=float)
        if q.shape != (self.dof,):
            raise ValueError(f"Expected q shape {(self.dof,)}, got {q.shape}")
        configuration = self._q_template.copy()
        configuration[self._q_indices] = q
        return configuration

    def _forward_pinocchio(self, q: np.ndarray) -> pin.SE3:
        configuration = self._configuration(q)
        pin.forwardKinematics(self.model, self.data, configuration)
        pin.updateFramePlacements(self.model, self.data)
        return self.data.oMf[self._frame_id] * self._tool_local

    def _forward_mujoco(self, q: np.ndarray) -> pin.SE3:
        for name, value in zip(self.names, q):
            self.reference.data.qpos[self.reference._qpos[name]] = value
        mujoco.mj_forward(self.reference.model, self.reference.data)
        body_id = mujoco.mj_name2id(
            self.reference.model, mujoco.mjtObj.mjOBJ_BODY, self.tool.parent_body
        )
        rotation = self.reference.data.xmat[body_id].reshape(3, 3).copy()
        body = pin.SE3(rotation, self.reference.data.xpos[body_id].copy())
        return body * self._tool_local

    def forward(self, q: np.ndarray) -> Pose:
        """Return the configured TCP pose in the MuJoCo world frame."""
        placement = self._forward_pinocchio(q)
        rotation = self._world_rotation @ placement.rotation
        position = self._world_rotation @ placement.translation + self._world_translation
        coefficients = pin.Quaternion(rotation).coeffs()  # xyzw
        quaternion = coefficients[[3, 0, 1, 2]]
        return Pose(position, quaternion)

    def jacobian(self, q: np.ndarray) -> np.ndarray:
        """Return a 6xN [linear; angular] geometric Jacobian at the TCP."""
        configuration = self._configuration(q)
        jacobian = np.asarray(
            pin.computeFrameJacobian(
                self.model,
                self.data,
                configuration,
                self._frame_id,
                pin.ReferenceFrame.LOCAL_WORLD_ALIGNED,
            )[:, self._v_indices]
        ).copy()
        parent_rotation = self.data.oMf[self._frame_id].rotation
        tcp_offset = parent_rotation @ self.tool.position
        jacobian[:3, :] += np.cross(jacobian[3:, :].T, tcp_offset).T
        jacobian[:3, :] = self._world_rotation @ jacobian[:3, :]
        jacobian[3:, :] = self._world_rotation @ jacobian[3:, :]
        return jacobian
