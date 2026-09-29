"""Unified position, velocity and torque control implemented in joint effort."""

import numpy as np

from robot_mj.interfaces import ControlMode, JointCommand
from robot_mj.robots import RobotDescription
from robot_mj.sim import MujocoSimulation


class JointController:
    def __init__(self, simulation: MujocoSimulation, robot: RobotDescription, group: str):
        self.simulation = simulation
        self.robot = robot
        self.group = group
        self.names = robot.group(group)
        try:
            self.gains = robot.control[group]
        except KeyError as exc:
            raise ValueError(f"No control gains configured for group {group!r}") from exc
        # All three public modes are converted to effort below. Isolating the
        # source MJCF actuator avoids hidden stiffness/damping in torque mode.
        self.simulation.isolate_software_control(self.names)

    def update(self, command: JointCommand) -> np.ndarray:
        if command.names != self.names:
            raise ValueError(
                "Command names must exactly match the configured group order; "
                "this prevents silent joint-order mistakes"
            )
        state = self.simulation.joint_state(self.group)
        if command.mode is ControlMode.POSITION:
            effort = self.simulation.bias_effort(self.names)
            effort += self.gains.kp * (command.values - state.position)
            effort -= self.gains.kd * state.velocity
        elif command.mode is ControlMode.VELOCITY:
            effort = self.simulation.bias_effort(self.names)
            effort += self.gains.velocity_gain * (command.values - state.velocity)
        elif command.mode is ControlMode.TORQUE:
            effort = command.values.copy()
        else:  # Defensive check for deserialized external commands.
            raise ValueError(f"Unsupported control mode: {command.mode}")
        effort = np.clip(effort, -self.gains.effort_limit, self.gains.effort_limit)
        self.simulation.set_generalized_effort(self.names, effort)
        return effort
