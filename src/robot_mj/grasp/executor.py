"""Contact-aware G1 grasp execution using MuJoCo actuator dynamics."""

from dataclasses import dataclass
import time

import mujoco
import numpy as np

from robot_mj.collision import CollisionPhase, HppFclCollisionChecker
from robot_mj.interfaces import Pose
from robot_mj.kinematics import PinocchioKinematics, solve_ik
from robot_mj.robots import RobotDescription
from robot_mj.trajectory import MotionSettings, parameterize_toppra

from .planner import GraspPlan, GraspSettings


@dataclass(frozen=True)
class ExecutionReport:
    grasp_confirmed: bool
    planned_peak_velocity: np.ndarray
    planned_peak_acceleration: np.ndarray
    tracking_rms: float
    tracking_peak: float
    measured_peak_velocity: np.ndarray
    measured_peak_acceleration: np.ndarray
    preclose_contacts: tuple[str, ...]
    forbidden_contacts: tuple[str, ...]
    contact_links: tuple[str, ...]
    sidewall_contact_links: tuple[str, ...]
    opposition_cosine: float | None
    contact_center_error: float | None
    maximum_penetration: float
    dual_contact_fraction: float
    object_displacement_before_lift: float
    commanded_lift: float
    achieved_object_lift: float
    achieved_tool_lift: float
    horizontal_slip: float


def _sample(time_value: float, trajectory) -> np.ndarray:
    value = min(max(time_value, 0.0), trajectory.duration)
    return np.array([
        np.interp(value, trajectory.time, trajectory.position[:, index])
        for index in range(trajectory.position.shape[1])
    ])


class GraspExecutor:
    """Execute a plan, close on measured contact, then prove it by lifting.

    The object is never welded, attached or teleported.  A successful return
    therefore means MuJoCo contact and friction actually carried the target.
    """

    def __init__(self, scene, robot: RobotDescription,
                 kinematics: PinocchioKinematics,
                 checker: HppFclCollisionChecker, settings: GraspSettings,
                 lift_motion: MotionSettings):
        self.scene, self.robot = scene, robot
        self.kinematics, self.checker = kinematics, checker
        self.settings, self.lift_motion = settings, lift_motion

    def _hand_target(self, opened: np.ndarray,
                     fractions: np.ndarray) -> np.ndarray:
        """Convert independent [thumb, index, middle] travel to joint targets."""
        fractions = np.asarray(fractions, dtype=float)
        if fractions.shape != (3,):
            raise ValueError("Hand fractions must be [thumb, index, middle]")
        closed = self.settings.hand.closed
        result = opened.copy()
        for indices, fraction in (
            (slice(0, 3), fractions[0]),
            (slice(3, 5), fractions[1]),
            (slice(5, 7), fractions[2]),
        ):
            result[indices] += fraction * (
                closed[indices] - opened[indices]
            )
        return result

    @staticmethod
    def _pinch_geometry(points: dict[str, np.ndarray], center: np.ndarray):
        thumb = points.get("right_hand_thumb_2_link")
        opponents = [
            points[name] for name in (
                "right_hand_index_1_link", "right_hand_middle_1_link"
            ) if name in points
        ]
        if thumb is None or not opponents:
            return None
        opposing = min(
            opponents,
            key=lambda point: np.linalg.norm(0.5 * (thumb + point) - center),
        )
        first, second = thumb - center, opposing - center
        denominator = np.linalg.norm(first) * np.linalg.norm(second)
        if denominator < 1e-12:
            return None
        return (
            float(np.dot(first, second) / denominator),
            float(np.linalg.norm(0.5 * (thumb + opposing) - center)),
        )

    def execute(self, plan: GraspPlan, *, show_viewer: bool = False) -> ExecutionReport:
        model = mujoco.MjModel.from_xml_path(str(self.scene))
        data = mujoco.MjData(model)
        key = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_KEY, self.robot.home_keyframe
        )
        mujoco.mj_resetDataKeyframe(model, data, key)
        arm_names = self.robot.group(self.settings.arm_group)
        hand_names = self.robot.group(self.settings.hand_group)

        def ids(kind, names):
            return np.array([
                mujoco.mj_name2id(model, kind, name) for name in names
            ], dtype=int)

        arm_joints = ids(mujoco.mjtObj.mjOBJ_JOINT, arm_names)
        hand_joints = ids(mujoco.mjtObj.mjOBJ_JOINT, hand_names)
        arm_actuators = ids(mujoco.mjtObj.mjOBJ_ACTUATOR, arm_names)
        hand_actuators = ids(mujoco.mjtObj.mjOBJ_ACTUATOR, hand_names)
        arm_qpos = model.jnt_qposadr[arm_joints]
        arm_dofs = model.jnt_dofadr[arm_joints]
        hand_qpos = model.jnt_qposadr[hand_joints]
        hand_open = data.qpos[hand_qpos].copy()
        arm_kp = model.actuator_gainprm[arm_actuators, 0].copy()

        # The source hand actuators are extremely stiff.  A low-gain,
        # force-limited position servo gives contact a chance to settle.
        hand = self.settings.hand
        model.actuator_gainprm[hand_actuators, 0] = hand.kp
        model.actuator_biasprm[hand_actuators, 1] = -hand.kp
        model.actuator_forcelimited[hand_actuators] = 1
        model.actuator_forcerange[hand_actuators] = [
            -hand.torque_limit, hand.torque_limit
        ]
        target_body = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_BODY, self.settings.target_body
        )
        target_geoms = np.flatnonzero(model.geom_bodyid == target_body)
        if len(target_geoms) != 1 or model.geom_type[target_geoms[0]] != mujoco.mjtGeom.mjGEOM_CYLINDER:
            raise ValueError("The three-finger side-grasp demo requires one cylinder target geom")
        cylinder_half_height = float(model.geom_size[target_geoms[0], 1])
        mujoco.mj_forward(model, data)
        initial_object = data.xpos[target_body].copy()
        touch_links = set(self.robot.end_effectors[
            self.settings.end_effector
        ].touch_bodies)

        window = None
        wall_start = time.time()
        if show_viewer:
            from mujoco import viewer
            window = viewer.launch_passive(model, data)
            window.cam.lookat[:] = [0.30, -0.18, 0.78]
            window.cam.distance = 1.05
            window.cam.azimuth = 145
            window.cam.elevation = -12

        def sync():
            if window is None:
                return True
            if not window.is_running():
                return False
            window.sync()
            delay = data.time - (time.time() - wall_start)
            if delay > 0.0:
                time.sleep(delay)
            return True

        def arm_command(desired):
            mujoco.mj_forward(model, data)
            target = desired + data.qfrc_bias[arm_dofs] / arm_kp
            data.ctrl[arm_actuators] = np.clip(
                target,
                model.actuator_ctrlrange[arm_actuators, 0],
                model.actuator_ctrlrange[arm_actuators, 1],
            )

        tracking, measured_velocity, measured_acceleration = [], [], []
        previous_velocity = data.qvel[arm_dofs].copy()
        preclose_contacts: set[str] = set()
        forbidden_contacts: set[str] = set()

        def audit_contacts(*, closure: bool):
            active_prefixes = ("right_shoulder", "right_elbow", "right_wrist", "right_hand")
            for contact in data.contact:
                body1 = int(model.geom_bodyid[contact.geom1])
                body2 = int(model.geom_bodyid[contact.geom2])
                name1 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body1) or "world"
                name2 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body2) or "world"
                active1 = name1.startswith(active_prefixes)
                active2 = name2.startswith(active_prefixes)
                if not active1 and not active2:
                    continue
                if target_body in (body1, body2):
                    robot_name = name2 if body1 == target_body else name1
                    if not closure:
                        preclose_contacts.add(f"{robot_name}<->{self.settings.target_body}")
                    elif robot_name not in touch_links:
                        forbidden_contacts.add(
                            f"{robot_name}<->{self.settings.target_body}"
                        )
                elif active1 != active2:
                    other_geom = contact.geom2 if active1 else contact.geom1
                    active_name = name1 if active1 else name2
                    other = mujoco.mj_id2name(
                        model, mujoco.mjtObj.mjOBJ_GEOM, other_geom
                    ) or (name2 if active1 else name1)
                    forbidden_contacts.add(f"{active_name}<->{other}")
                elif active1 and active2:
                    # Parent/child collision is intentionally disabled by most
                    # robot models and does not represent self-collision.
                    if not (
                        int(model.body_parentid[body1]) == body2
                        or int(model.body_parentid[body2]) == body1
                    ):
                        forbidden_contacts.add(f"{name1}<->{name2}")

        def step(desired, hand_command, *, closure=False):
            nonlocal previous_velocity
            arm_command(desired)
            data.ctrl[hand_actuators] = hand_command
            mujoco.mj_step(model, data)
            actual = data.qpos[arm_qpos].copy()
            velocity = data.qvel[arm_dofs].copy()
            tracking.append(actual - desired)
            measured_velocity.append(velocity)
            measured_acceleration.append(
                (velocity - previous_velocity) / model.opt.timestep
            )
            previous_velocity = velocity
            audit_contacts(closure=closure)
            return sync()

        # Global transit uses the open hand and ends at a safe staging posture.
        start_time = data.time
        while data.time - start_time <= plan.transit.duration + 0.30:
            desired = _sample(data.time - start_time, plan.transit)
            if not step(desired, hand_open):
                break

        # Only pre-shape after leaving the hip; the second planned transfer and
        # Cartesian approach were both checked with this exact hand shape.
        preshape_start = data.time
        staging = plan.transit.position[-1]
        while data.time - preshape_start <= hand.pregrasp_pause:
            alpha = min(1.0, (data.time - preshape_start) / hand.pregrasp_pause)
            command = self._hand_target(
                hand_open,
                np.array([alpha * hand.thumb_preshape, 0.0, 0.0]),
            )
            if not step(staging, command):
                break

        # Pre-shape only the thumb while stationary, then use a separately
        # retimed slow Cartesian approach. Both phase boundaries are at rest.
        approach_start = data.time
        while data.time - approach_start <= plan.approach.duration + 0.35:
            desired = _sample(data.time - approach_start, plan.approach)
            if not step(desired, plan.preshape):
                break

        # Close only now. Stop on a centered, opposing measured contact pair.
        ready_start = data.time
        ready_duration = 1.20
        start_fraction = np.array([hand.thumb_preshape, 0.0, 0.0])
        while data.time - ready_start <= ready_duration:
            fraction = min(1.0, (data.time - ready_start) / ready_duration)
            finger_fraction = (
                (1.0 - fraction) * start_fraction
                + fraction * hand.contact_fraction
            )
            command = self._hand_target(
                hand_open, finger_fraction
            )
            step(plan.grasp_configuration, command)

        close_start = data.time
        stopped_target = None
        stopped_time = None
        observed_links: set[str] = set()
        observed_side_links: set[str] = set()
        minimum_distance = 0.0
        geometry = None
        hold_samples = dual_samples = 0
        guarded_arm_target = None
        first_guarded_contact_time = None
        while data.time - close_start <= (
            hand.max_close_fraction / hand.close_rate + hand.hold_time + 1.2
        ):
            elapsed = data.time - close_start
            alpha = min(hand.max_close_fraction, hand.close_rate * elapsed)
            insertion = min(1.0, alpha / hand.max_close_fraction)
            closure_arm_target = (
                (1.0 - insertion) * plan.grasp_configuration
                + insertion * plan.contact_configuration
            )
            if guarded_arm_target is not None:
                closure_arm_target = guarded_arm_target
                finger_alpha = min(
                    hand.max_close_fraction,
                    hand.close_rate * (data.time - first_guarded_contact_time),
                )
                finger_fraction = np.minimum(
                    hand.max_close_fraction,
                    hand.contact_fraction + finger_alpha,
                )
            else:
                finger_fraction = hand.contact_fraction
            command = self._hand_target(
                hand_open, finger_fraction
            ) if stopped_target is None else stopped_target
            step(closure_arm_target, command, closure=True)
            forces: dict[str, float] = {}
            points: dict[str, list[np.ndarray]] = {}
            side_links: set[str] = set()
            for index, contact in enumerate(data.contact):
                body1 = int(model.geom_bodyid[contact.geom1])
                body2 = int(model.geom_bodyid[contact.geom2])
                if target_body not in (body1, body2):
                    continue
                other = body2 if body1 == target_body else body1
                name = mujoco.mj_id2name(
                    model, mujoco.mjtObj.mjOBJ_BODY, other
                ) or "world"
                if name not in touch_links:
                    continue
                wrench = np.zeros(6)
                mujoco.mj_contactForce(model, data, index, wrench)
                forces[name] = forces.get(name, 0.0) + float(wrench[0])
                points.setdefault(name, []).append(np.asarray(contact.pos).copy())
                observed_links.add(name)
                local_point = (
                    data.xmat[target_body].reshape(3, 3).T
                    @ (np.asarray(contact.pos) - data.xpos[target_body])
                )
                # Reject top/bottom-cap contacts: all three fingers must land
                # on the cylindrical side, like an opposed two-finger grasp.
                if abs(local_point[2]) <= cylinder_half_height - 0.001:
                    side_links.add(name)
                    observed_side_links.add(name)
                minimum_distance = min(minimum_distance, float(contact.dist))
            centers = {name: np.mean(values, axis=0)
                       for name, values in points.items()}
            current_geometry = self._pinch_geometry(
                centers, data.xpos[target_body]
            )
            thumb_ok = forces.get("right_hand_thumb_2_link", 0.0) >= hand.contact_force
            opposing_force = max(
                forces.get("right_hand_index_1_link", 0.0),
                forces.get("right_hand_middle_1_link", 0.0),
            )
            opposing_ok = opposing_force >= hand.contact_force
            three_finger_force = all(
                forces.get(name, 0.0) >= hand.contact_force
                for name in touch_links
            )
            all_on_side = touch_links <= side_links
            # Stop Cartesian insertion on the very first legitimate distal
            # side contact. Remaining fingers close compliantly from there;
            # continuing to drive the wrist would create deep penetration.
            if guarded_arm_target is None and any(
                forces.get(name, 0.0) >= hand.contact_force
                and name in side_links for name in touch_links
            ):
                guarded_arm_target = data.qpos[arm_qpos].copy()
                first_guarded_contact_time = data.time
            centered = (
                current_geometry is not None
                and current_geometry[0] <= hand.maximum_opposition_cosine
                and current_geometry[1] <= hand.maximum_center_error
            )
            if (
                thumb_ok and opposing_ok and three_finger_force and all_on_side
                and centered and stopped_target is None
            ):
                stopped_target = self._hand_target(
                    hand_open,
                    np.minimum(
                        hand.max_close_fraction,
                        hand.contact_fraction + hand.squeeze_fraction,
                    ),
                )
                stopped_time = data.time
                geometry = current_geometry
            if stopped_target is not None:
                hold_samples += 1
                if (
                    all(forces.get(name, 0.0) >= hand.hold_force
                        for name in touch_links)
                    and touch_links <= side_links
                ):
                    dual_samples += 1
                if data.time - stopped_time >= hand.hold_time:
                    break

        dual_fraction = dual_samples / hold_samples if hold_samples else 0.0
        pinch_confirmed = (
            stopped_target is not None
            and dual_fraction >= hand.minimum_hold_fraction
            and -minimum_distance <= hand.maximum_penetration
            and not forbidden_contacts
        )
        before_lift = data.xpos[target_body].copy()
        displacement = float(np.linalg.norm(before_lift - initial_object))
        object_lift = tool_lift = horizontal_slip = 0.0

        if pinch_confirmed:
            # Plan a vertical TCP lift while the semantic checker allows only
            # the three configured fingertip links to touch the target.
            lift_waypoints = [data.qpos[arm_qpos].copy()]
            start_pose = self.kinematics.forward(lift_waypoints[0])
            for fraction in np.linspace(0.0, 1.0, 11)[1:]:
                target = Pose(
                    start_pose.position
                    + np.array([0.0, 0.0, fraction * hand.trial_lift_distance]),
                    start_pose.quaternion,
                )
                q = solve_ik(
                    self.kinematics, target, [lift_waypoints[-1]],
                    reference=lift_waypoints[-1], posture_weight=0.002,
                ).configuration
                hand_map = dict(zip(hand_names, stopped_target))
                if not self.checker.edge_is_valid(
                    lift_waypoints[-1], q, CollisionPhase.GRASP,
                    joint_positions=hand_map,
                    max_joint_step=self.lift_motion.edge_joint_step,
                ):
                    raise RuntimeError("Trial-lift path has a forbidden collision")
                lift_waypoints.append(q)
            lift = parameterize_toppra(
                np.asarray(lift_waypoints), arm_names, self.lift_motion,
                lower=self.kinematics.lower, upper=self.kinematics.upper,
                checker=self.checker, phase=CollisionPhase.GRASP,
                joint_positions=dict(zip(hand_names, stopped_target)),
            )
            tool_start = self.kinematics.forward(
                data.qpos[arm_qpos].copy()
            ).position.copy()
            lift_start = data.time
            while data.time - lift_start <= lift.duration + 0.45:
                desired = _sample(data.time - lift_start, lift)
                if not step(desired, stopped_target, closure=True):
                    break
            delta = data.xpos[target_body] - before_lift
            object_lift = float(delta[2])
            horizontal_slip = float(np.linalg.norm(delta[:2]))
            tool_lift = float(
                self.kinematics.forward(data.qpos[arm_qpos]).position[2]
                - tool_start[2]
            )

        tracking_array = np.asarray(tracking)
        measured_v = np.asarray(measured_velocity)
        measured_a = np.asarray(measured_acceleration)
        planned_v = np.maximum(
            np.max(np.abs(plan.transit.velocity), axis=0),
            np.max(np.abs(plan.approach.velocity), axis=0),
        )
        planned_a = np.maximum(
            np.max(np.abs(plan.transit.acceleration), axis=0),
            np.max(np.abs(plan.approach.acceleration), axis=0),
        )
        grasp_confirmed = (
            pinch_confirmed
            and object_lift >= 0.020
            and horizontal_slip <= 0.012
            and not preclose_contacts
            and not forbidden_contacts
        )
        report = ExecutionReport(
            grasp_confirmed=grasp_confirmed,
            planned_peak_velocity=planned_v,
            planned_peak_acceleration=planned_a,
            tracking_rms=float(np.sqrt(np.mean(tracking_array**2))),
            tracking_peak=float(np.max(np.abs(tracking_array))),
            measured_peak_velocity=np.max(np.abs(measured_v), axis=0),
            measured_peak_acceleration=np.max(np.abs(measured_a), axis=0),
            preclose_contacts=tuple(sorted(preclose_contacts)),
            forbidden_contacts=tuple(sorted(forbidden_contacts)),
            contact_links=tuple(sorted(observed_links)),
            sidewall_contact_links=tuple(sorted(observed_side_links)),
            opposition_cosine=None if geometry is None else geometry[0],
            contact_center_error=None if geometry is None else geometry[1],
            maximum_penetration=max(0.0, -minimum_distance),
            dual_contact_fraction=dual_fraction,
            object_displacement_before_lift=displacement,
            commanded_lift=hand.trial_lift_distance,
            achieved_object_lift=object_lift,
            achieved_tool_lift=tool_lift,
            horizontal_slip=horizontal_slip,
        )
        if window is not None:
            print("Execution finished; close the viewer to exit.")
            while window.is_running():
                window.sync()
                time.sleep(0.02)
            window.close()
        return report
