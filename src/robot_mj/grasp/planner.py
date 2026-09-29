"""Object-relative grasp planning with semantic collision phases."""

from dataclasses import dataclass
from pathlib import Path
import tomllib

import numpy as np
from ompl import util as ou
import mujoco
from scipy.spatial.transform import Rotation

from robot_mj.collision import (
    CollisionContact, CollisionPhase, CollisionReport, HppFclCollisionChecker,
)
from robot_mj.interfaces import Pose
from robot_mj.kinematics import PinocchioKinematics, solve_ik
from robot_mj.planning import GeometricPlan, OmplPlanner
from robot_mj.robots import RobotDescription
from robot_mj.trajectory import (
    MotionSettings,
    SynchronizedTrajectory,
    parameterize_toppra,
)


@dataclass(frozen=True)
class HandSettings:
    closed: np.ndarray
    contact_fraction: np.ndarray
    thumb_preshape: float
    close_rate: float
    max_close_fraction: float
    squeeze_fraction: float
    kp: float
    torque_limit: float
    contact_force: float
    hold_force: float
    hold_time: float
    minimum_hold_fraction: float
    maximum_center_error: float
    maximum_opposition_cosine: float
    maximum_penetration: float
    pregrasp_pause: float
    trial_lift_distance: float


@dataclass(frozen=True)
class GraspSettings:
    target_body: str
    target_joint: str
    arm_group: str
    hand_group: str
    end_effector: str
    object_to_grasp: np.ndarray
    closure_translation: np.ndarray
    retreat_distance: float
    pregrasp_direction_object: np.ndarray
    cartesian_steps: int
    natural_posture: np.ndarray
    seed_std: np.ndarray
    random_seed: int
    seed_count: int
    staging_posture: np.ndarray
    hand: HandSettings

    @classmethod
    def from_toml(cls, path: str | Path) -> "GraspSettings":
        with Path(path).open("rb") as stream:
            raw = tomllib.load(stream)
        transform = np.eye(4)
        transform[:3, :3] = np.asarray(raw["object_to_grasp_rotation"], float)
        transform[:3, 3] = np.asarray(raw["object_to_grasp_translation"], float)
        hand = raw["hand"]
        return cls(
            target_body=raw["target_body"], target_joint=raw["target_joint"],
            arm_group=raw["arm_group"], hand_group=raw["hand_group"],
            end_effector=raw["end_effector"], object_to_grasp=transform,
            closure_translation=np.asarray(raw["closure_translation"], float),
            retreat_distance=float(raw["retreat_distance"]),
            pregrasp_direction_object=np.asarray(
                raw["pregrasp_direction_object"], float
            ),
            cartesian_steps=int(raw["cartesian_steps"]),
            natural_posture=np.asarray(raw["natural_posture"], float),
            seed_std=np.asarray(raw["seed_std"], float),
            random_seed=int(raw["random_seed"]), seed_count=int(raw["seed_count"]),
            staging_posture=np.asarray(raw["staging_posture"], float),
            hand=HandSettings(
                closed=np.asarray(hand["closed"], float),
                contact_fraction=np.asarray(hand["contact_fraction"], float),
                **{name: float(hand[name]) for name in (
                    "thumb_preshape", "close_rate", "max_close_fraction",
                    "squeeze_fraction", "kp", "torque_limit", "contact_force",
                    "hold_force", "hold_time", "minimum_hold_fraction",
                    "maximum_center_error", "maximum_opposition_cosine",
                    "maximum_penetration", "pregrasp_pause", "trial_lift_distance",
                )},
            ),
        )


@dataclass(frozen=True)
class GraspPlan:
    object_pose: Pose
    pregrasp_pose: Pose
    grasp_pose: Pose
    pregrasp_configuration: np.ndarray
    grasp_configuration: np.ndarray
    contact_configuration: np.ndarray
    preshape: np.ndarray
    transit_geometric: GeometricPlan
    approach_waypoints: np.ndarray
    transit: SynchronizedTrajectory
    approach: SynchronizedTrajectory
    valid_ik_candidates: int


def _matrix(pose: Pose) -> np.ndarray:
    w, x, y, z = pose.quaternion
    result = np.eye(4)
    result[:3, :3] = Rotation.from_quat([x, y, z, w]).as_matrix()
    result[:3, 3] = pose.position
    return result


def _pose(transform: np.ndarray) -> Pose:
    xyzw = Rotation.from_matrix(transform[:3, :3]).as_quat()
    return Pose(transform[:3, 3], xyzw[[3, 0, 1, 2]])


class _CrossCheckedCollision:
    """Duck-typed checker that requires both HPP-FCL and MuJoCo clearance."""

    def __init__(self, exact, owner: "GraspPlanner", hand: np.ndarray):
        self.exact, self.owner, self.hand = exact, owner, hand

    def check(self, q, phase=CollisionPhase.FREE_SPACE, *, stop_at_first=False,
              joint_positions=None):
        report = self.exact.check(
            q, phase, joint_positions=joint_positions, stop_at_first=stop_at_first
        )
        if report.in_collision:
            return report
        if not self.owner._mujoco_configuration_is_clear(q, self.hand):
            return CollisionReport((CollisionContact(
                "right_arm_hand", "mujoco_scene", "cross_backend"
            ),))
        return report

    def edge_is_valid(self, start, goal, phase=CollisionPhase.FREE_SPACE, *,
                      joint_positions=None, max_joint_step=0.04):
        start, goal = np.asarray(start), np.asarray(goal)
        count = max(1, int(np.ceil(
            np.max(np.abs(goal - start)) / max_joint_step
        )))
        return all(not self.check(
            start + alpha * (goal - start), phase,
            joint_positions=joint_positions, stop_at_first=True,
        ).in_collision for alpha in np.linspace(0.0, 1.0, count + 1))


class GraspPlanner:
    """Plan global transit and straight Cartesian approach as separate phases."""

    def __init__(self, robot: RobotDescription, kinematics: PinocchioKinematics,
                 checker: HppFclCollisionChecker, settings: GraspSettings,
                 transit_motion: MotionSettings, approach_motion: MotionSettings):
        self.robot, self.kinematics, self.checker = robot, kinematics, checker
        self.settings = settings
        self.transit_motion, self.approach_motion = transit_motion, approach_motion
        self.hand_names = robot.group(settings.hand_group)
        self.hand_open = kinematics.reference.joint_state(settings.hand_group).position

    def _hand_target(self, fractions: np.ndarray) -> np.ndarray:
        """Map independent thumb/index/middle travel to seven hand joints.

        The three fingers have different link lengths and transmissions.  A
        single close percentage therefore does not represent a three-point
        grasp: each finger must stop at its mesh-calibrated contact travel.
        """
        fractions = np.asarray(fractions, dtype=float)
        if fractions.shape != (3,):
            raise ValueError("Hand fractions must be [thumb, index, middle]")
        result = self.hand_open.copy()
        for indices, fraction in (
            (slice(0, 3), fractions[0]),
            (slice(3, 5), fractions[1]),
            (slice(5, 7), fractions[2]),
        ):
            result[indices] += fraction * (
                self.settings.hand.closed[indices] - result[indices]
            )
        return result

    def _joint_map(self, hand: np.ndarray) -> dict[str, float]:
        return dict(zip(self.hand_names, hand))

    def _mujoco_path_is_clear(
        self, positions: np.ndarray, hand: np.ndarray
    ) -> bool:
        """Cross-check HPP-FCL output with executable MuJoCo geometry.

        Backend mesh conversion and controller tracking can differ by a few
        millimetres. Planning inflation handles tracking; this independent
        check prevents a conversion discrepancy from reaching execution.
        """
        simulation = self.kinematics.reference
        model, data = simulation.model, simulation.data
        hand_addresses = [simulation._qpos[name] for name in self.hand_names]
        arm_addresses = [simulation._qpos[name] for name in self.kinematics.names]
        active_prefixes = (
            "right_shoulder", "right_elbow", "right_wrist", "right_hand"
        )
        sample_indices = np.unique(np.r_[np.arange(0, len(positions), 10), len(positions) - 1])
        for q in positions[sample_indices]:  # 20 ms plus the exact endpoint.
            if not self._mujoco_configuration_is_clear(q, hand):
                return False
        return True

    def _mujoco_configuration_is_clear(
        self, q: np.ndarray, hand: np.ndarray
    ) -> bool:
        """Return whether one arm/hand state has no MuJoCo task contact."""
        simulation = self.kinematics.reference
        model, data = simulation.model, simulation.data
        hand_addresses = [simulation._qpos[name] for name in self.hand_names]
        arm_addresses = [simulation._qpos[name] for name in self.kinematics.names]
        active_prefixes = (
            "right_shoulder", "right_elbow", "right_wrist", "right_hand"
        )
        data.qpos[arm_addresses] = q
        data.qpos[hand_addresses] = hand
        mujoco.mj_forward(model, data)
        for contact in data.contact:
            first = int(model.geom_bodyid[contact.geom1])
            second = int(model.geom_bodyid[contact.geom2])
            first_name = mujoco.mj_id2name(
                model, mujoco.mjtObj.mjOBJ_BODY, first
            ) or "world"
            second_name = mujoco.mj_id2name(
                model, mujoco.mjtObj.mjOBJ_BODY, second
            ) or "world"
            if first_name.startswith(active_prefixes) != second_name.startswith(active_prefixes):
                return False
        return True

    def _poses(self, object_pose: Pose) -> tuple[Pose, Pose]:
        object_transform = _matrix(object_pose)
        grasp = object_transform @ self.settings.object_to_grasp
        pregrasp = grasp.copy()
        # A side grasp retreats horizontally away from the object. Retiring
        # along tool -Z made this particular hand descend into the tabletop.
        outward = self.settings.pregrasp_direction_object.copy()
        outward /= np.linalg.norm(outward)
        pregrasp[:3, 3] += (
            object_transform[:3, :3] @ outward * self.settings.retreat_distance
        )
        return _pose(pregrasp), _pose(grasp)

    def _seeds(self) -> list[np.ndarray]:
        rng = np.random.default_rng(self.settings.random_seed)
        seeds = [self.settings.natural_posture, self.kinematics.home]
        for _ in range(self.settings.seed_count):
            seeds.append(np.clip(
                self.settings.natural_posture
                + rng.normal(0.0, self.settings.seed_std),
                self.kinematics.lower + 1e-4,
                self.kinematics.upper - 1e-4,
            ))
        return seeds

    def plan(self, object_pose: Pose) -> GraspPlan:
        # Deterministic planning is essential for repeatable simulation and CI.
        ou.RNG.setSeed(self.settings.random_seed)
        pregrasp_pose, grasp_pose = self._poses(object_pose)
        preshape = self._hand_target(np.array([
            self.settings.hand.thumb_preshape, 0.0, 0.0,
        ]))
        hand_map = self._joint_map(preshape)

        # Solve independent redundant branches, reject collision before scoring.
        candidates: list[tuple[float, np.ndarray, np.ndarray]] = []
        for seed in self._seeds():
            try:
                grasp = solve_ik(
                    self.kinematics, grasp_pose, [seed],
                    reference=self.settings.natural_posture, posture_weight=0.005,
                ).configuration
                pregrasp = solve_ik(
                    self.kinematics, pregrasp_pose, [grasp],
                    reference=self.settings.natural_posture, posture_weight=0.005,
                ).configuration
            except RuntimeError:
                continue
            if self.checker.check(
                grasp, CollisionPhase.APPROACH,
                joint_positions=hand_map, stop_at_first=True,
            ).in_collision:
                continue
            if self.checker.check(
                pregrasp, CollisionPhase.FREE_SPACE,
                joint_positions=hand_map, stop_at_first=True,
            ).in_collision:
                continue
            margin = np.min(np.minimum(
                grasp - self.kinematics.lower,
                self.kinematics.upper - grasp,
            ))
            weights = np.array([1.2, 1.2, 1.8, 0.8, 0.7, 0.7, 0.7])
            score = float(np.linalg.norm(
                weights * (grasp - self.settings.natural_posture)
            ) + 0.08 / (margin + 0.05))
            candidates.append((score, pregrasp, grasp))
        if not candidates:
            raise RuntimeError("No natural, collision-free grasp IK branch")
        _, q_pregrasp, q_grasp = min(candidates, key=lambda item: item[0])

        # Solve each Cartesian sample from its predecessor to stay on one IK branch.
        approach = [q_pregrasp]
        for alpha in np.linspace(0.0, 1.0, self.settings.cartesian_steps + 1)[1:]:
            position = ((1.0 - alpha) * pregrasp_pose.position
                        + alpha * grasp_pose.position)
            target = Pose(position, grasp_pose.quaternion)
            q = solve_ik(
                self.kinematics, target, [approach[-1]],
                reference=approach[-1], posture_weight=0.002,
            ).configuration
            if not self.checker.edge_is_valid(
                approach[-1], q, CollisionPhase.APPROACH,
                joint_positions=hand_map,
                max_joint_step=self.approach_motion.edge_joint_step,
            ):
                raise RuntimeError("Cartesian approach intersects a forbidden body")
            approach.append(q)
        q_grasp = approach[-1]

        # The approach ends with no contact. During closure the wrist performs
        # a short guarded insertion toward the three-fingertip circumcenter.
        object_rotation = _matrix(object_pose)[:3, :3]
        contact_pose = Pose(
            grasp_pose.position
            + object_rotation @ self.settings.closure_translation,
            grasp_pose.quaternion,
        )
        q_contact = solve_ik(
            self.kinematics, contact_pose, [q_grasp],
            reference=q_grasp, posture_weight=0.002,
        ).configuration
        # Close all fingers to a common ready shape while the hand is still at
        # the clearance pose. Then perform only the guarded wrist insertion;
        # this avoids sweeping the thumb's proximal link through the cylinder.
        for alpha in np.linspace(0.0, 1.0, 21):
            start_fraction = np.array([
                self.settings.hand.thumb_preshape, 0.0, 0.0,
            ])
            fractions = (
                (1.0 - alpha) * start_fraction
                + alpha * self.settings.hand.contact_fraction
            )
            hand_q = self._hand_target(fractions)
            if self.checker.check(
                q_grasp, CollisionPhase.GRASP,
                joint_positions=self._joint_map(hand_q), stop_at_first=True,
            ).in_collision:
                raise RuntimeError("Finger ready-shape motion has a forbidden collision")
        for alpha in np.linspace(0.0, 1.0, 41):
            q = (1.0 - alpha) * q_grasp + alpha * q_contact
            hand_q = self._hand_target(self.settings.hand.contact_fraction)
            closure_report = self.checker.check(
                q, CollisionPhase.GRASP,
                joint_positions=self._joint_map(hand_q), stop_at_first=True,
            )
            if closure_report.in_collision:
                raise RuntimeError(
                    f"Guarded closure sweep collides at {alpha:.3f}: "
                    f"{closure_report.contacts}"
                )

        # Sampling planners may occasionally return a shortcut that fails our
        # stricter exact edge recheck. Retry with deterministic successive
        # seeds; never execute a path that failed validation.
        planning_errors: list[str] = []
        geometric = None
        transit = None
        for attempt in range(4):
            ompl = OmplPlanner(
                self.kinematics.lower, self.kinematics.upper, self.checker,
                planner_range=self.transit_motion.ompl_range,
                edge_joint_step=self.transit_motion.edge_joint_step,
            )
            try:
                geometric = ompl.plan(
                    self.kinematics.home, self.settings.staging_posture,
                    timeout=self.transit_motion.ompl_timeout,
                    simplify=self.transit_motion.simplify,
                )
                transit = parameterize_toppra(
                    geometric.waypoints, self.kinematics.names,
                    self.transit_motion,
                    lower=self.kinematics.lower, upper=self.kinematics.upper,
                    checker=self.checker,
                )
                break
            except RuntimeError as error:
                planning_errors.append(str(error))
        if geometric is None or transit is None:
            raise RuntimeError("OMPL retries failed: " + "; ".join(planning_errors))
        # After the open-hand global transit, pre-shape at the staging posture,
        # then use a second collision-checked transfer to the Cartesian segment.
        approach_timed = None
        approach_array = None
        transfer_errors: list[str] = []
        cross_checked = _CrossCheckedCollision(self.checker, self, preshape)
        for _ in range(8):
            transfer_planner = OmplPlanner(
                self.kinematics.lower, self.kinematics.upper, cross_checked,
                joint_positions=hand_map,
                planner_range=self.approach_motion.ompl_range,
                edge_joint_step=self.approach_motion.edge_joint_step,
            )
            try:
                transfer = transfer_planner.plan(
                    self.settings.staging_posture, q_pregrasp,
                    timeout=self.transit_motion.ompl_timeout,
                    simplify=True,
                )
                candidate = np.vstack((
                    transfer.waypoints,
                    np.asarray(approach)[1:],
                ))
                timed = parameterize_toppra(
                    candidate, self.kinematics.names, self.approach_motion,
                    lower=self.kinematics.lower, upper=self.kinematics.upper,
                    checker=cross_checked, phase=CollisionPhase.APPROACH,
                    joint_positions=hand_map,
                )
                if not self._mujoco_path_is_clear(timed.position, preshape):
                    raise RuntimeError("MuJoCo geometry rejected the pregrasp transfer")
                approach_array, approach_timed = candidate, timed
                break
            except RuntimeError as error:
                transfer_errors.append(str(error))
        if approach_timed is None or approach_array is None:
            raise RuntimeError(
                "Pregrasp transfer retries failed: " + "; ".join(transfer_errors)
            )
        return GraspPlan(
            object_pose, pregrasp_pose, grasp_pose, q_pregrasp, q_grasp,
            q_contact,
            preshape, geometric, approach_array, transit, approach_timed,
            len(candidates),
        )
