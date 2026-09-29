"""MuJoCo backend implementing state, stepping and generalized-force control."""

from pathlib import Path

import mujoco
import numpy as np

from robot_mj.interfaces import JointState, Pose
from robot_mj.robots import RobotDescription


class MujocoSimulation:
    """Own a compiled model and expose named joints instead of raw indices."""

    def __init__(self, model_path: str | Path, robot: RobotDescription) -> None:
        self.model_path = Path(model_path).resolve()
        self.robot = robot
        self.model = mujoco.MjModel.from_xml_path(str(self.model_path))
        self.data = mujoco.MjData(self.model)
        self._joint_ids: dict[str, int] = {}
        self._qpos: dict[str, int] = {}
        self._dof: dict[str, int] = {}
        self._actuator: dict[str, int] = {}
        self._index_robot()
        self.reset()

    def _index_robot(self) -> None:
        all_names = {name for group in self.robot.groups.values() for name in group}
        for name in all_names:
            joint_id = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_JOINT, name
            )
            if joint_id < 0:
                raise ValueError(f"Robot joint {name!r} is missing from {self.model_path}")
            if self.model.jnt_type[joint_id] != mujoco.mjtJoint.mjJNT_HINGE:
                raise ValueError(f"Controlled joint {name!r} must be a hinge")
            self._joint_ids[name] = joint_id
            self._qpos[name] = int(self.model.jnt_qposadr[joint_id])
            self._dof[name] = int(self.model.jnt_dofadr[joint_id])
            actuator_id = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name
            )
            if actuator_id >= 0:
                self._actuator[name] = actuator_id

    def reset(self) -> None:
        key_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_KEY, self.robot.home_keyframe
        )
        if key_id >= 0:
            mujoco.mj_resetDataKeyframe(self.model, self.data, key_id)
        else:
            mujoco.mj_resetData(self.model, self.data)
        self.data.qfrc_applied[:] = 0.0
        mujoco.mj_forward(self.model, self.data)

    def joint_state(self, group: str) -> JointState:
        names = self.robot.group(group)
        qpos = np.fromiter((self.data.qpos[self._qpos[n]] for n in names), float)
        qvel = np.fromiter((self.data.qvel[self._dof[n]] for n in names), float)
        effort = np.fromiter(
            (self.data.qfrc_actuator[self._dof[n]] + self.data.qfrc_applied[self._dof[n]] for n in names),
            float,
        )
        return JointState(names, qpos, qvel, effort, float(self.data.time))

    def set_generalized_effort(self, names: tuple[str, ...], effort: np.ndarray) -> None:
        """Apply joint torque without touching another controller's group."""
        effort = np.asarray(effort, dtype=float)
        if effort.shape != (len(names),):
            raise ValueError("Effort vector and joint names have different lengths")
        for name, value in zip(names, effort):
            self.data.qfrc_applied[self._dof[name]] = value

    def bias_effort(self, names: tuple[str, ...]) -> np.ndarray:
        """Return MuJoCo gravity/Coriolis bias in the requested joint order."""
        return np.fromiter(
            (self.data.qfrc_bias[self._dof[name]] for name in names), dtype=float
        )

    def isolate_software_control(self, names: tuple[str, ...]) -> None:
        """Disable native actuators for joints controlled through qfrc_applied.

        The G1 source MJCF has position actuators. Merely setting their targets
        to measured positions would leave native velocity damping active and
        contaminate velocity/torque semantics. Zero gain and bias make the
        software controller the only effort source for this joint group.
        """
        for name in names:
            actuator_id = self._actuator.get(name)
            if actuator_id is None:
                continue
            self.model.actuator_gainprm[actuator_id, :] = 0.0
            self.model.actuator_biasprm[actuator_id, :] = 0.0

    def body_pose(self, body_name: str) -> Pose:
        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        if body_id < 0:
            raise KeyError(f"Unknown MuJoCo body {body_name!r}")
        return Pose(
            self.data.xpos[body_id].copy(),
            self.data.xquat[body_id].copy(),
            stamp=float(self.data.time),
        )

    def set_free_body_pose(self, joint_name: str, pose: Pose) -> None:
        """External object-pose input used now by truth data and later by perception."""
        joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        if joint_id < 0 or self.model.jnt_type[joint_id] != mujoco.mjtJoint.mjJNT_FREE:
            raise KeyError(f"{joint_name!r} is not a free joint")
        address = int(self.model.jnt_qposadr[joint_id])
        self.data.qpos[address : address + 3] = pose.position
        self.data.qpos[address + 3 : address + 7] = pose.quaternion
        mujoco.mj_forward(self.model, self.data)

    def step(self, count: int = 1) -> None:
        for _ in range(count):
            mujoco.mj_step(self.model, self.data)
