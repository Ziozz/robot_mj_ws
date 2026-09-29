"""Coal/HPP-FCL collision checking with phase-aware allowed contacts."""

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
import tomllib

import coal
import numpy as np
import pinocchio as pin

from robot_mj.interfaces import Pose
from robot_mj.kinematics import PinocchioKinematics


class CollisionPhase(str, Enum):
    FREE_SPACE = "free_space"
    APPROACH = "approach"
    GRASP = "grasp"


@dataclass(frozen=True)
class CollisionContact:
    first: str
    second: str
    category: str


@dataclass(frozen=True)
class CollisionReport:
    contacts: tuple[CollisionContact, ...]

    @property
    def in_collision(self) -> bool:
        return bool(self.contacts)


@dataclass
class _EnvironmentGeometry:
    name: str
    geometry: object
    transform: coal.Transform3s
    is_target: bool = False
    actual_geometry: object | None = None


class HppFclCollisionChecker:
    """Check active-group self/environment collision using exact scene meshes."""

    def __init__(
        self,
        kinematics: PinocchioKinematics,
        task_config: str | Path,
        collision_config: str | Path,
    ) -> None:
        self.kinematics = kinematics
        self.model = kinematics.model
        self.data = self.model.createData()
        self.geometry_model = pin.buildGeomFromMJCF(
            self.model,
            str(kinematics.pinocchio_scene),
            pin.GeometryType.COLLISION,
        )
        self.geometry_data = pin.GeometryData(self.geometry_model)
        self._request = coal.CollisionRequest()
        self._allowed_link_pairs = self._load_allowed_pairs(collision_config)
        self._environment_margin, self._target_margin = self._load_margins(
            collision_config
        )
        self._robot_geometry_ids = self._unique_robot_geometries()
        self._active_joint_ids = set(kinematics._joint_ids)
        self._active_geometry_ids = {
            geometry_id
            for geometry_id in self._robot_geometry_ids
            if self._moves_with_active_group(
                self.geometry_model.geometryObjects[geometry_id].parentJoint
            )
        }
        self._environment, self.target_name = self._load_environment(task_config)
        self._touch_links = set(kinematics.tool.touch_bodies)

    @staticmethod
    def _load_allowed_pairs(path: str | Path) -> set[frozenset[str]]:
        with Path(path).open("rb") as stream:
            raw = tomllib.load(stream)
        return {
            frozenset((entry["first"], entry["second"]))
            for entry in raw.get("allowed_pairs", [])
        }

    @staticmethod
    def _load_margins(path: str | Path) -> tuple[float, float]:
        with Path(path).open("rb") as stream:
            raw = tomllib.load(stream).get("margins", {})
        return (
            float(raw.get("environment", 0.0)),
            float(raw.get("target_before_grasp", 0.0)),
        )

    def _unique_robot_geometries(self) -> tuple[int, ...]:
        unique: list[int] = []
        seen: set[tuple[str, str]] = set()
        for geometry_id, geometry in enumerate(self.geometry_model.geometryObjects):
            frame = self.model.frames[geometry.parentFrame].name
            key = (geometry.name, frame)
            if key in seen:
                continue
            seen.add(key)
            unique.append(geometry_id)
        return tuple(unique)

    def _moves_with_active_group(self, joint_id: int) -> bool:
        while joint_id > 0:
            if joint_id in self._active_joint_ids:
                return True
            joint_id = int(self.model.parents[joint_id])
        return False

    @staticmethod
    def _shape(specification: dict, margin: float = 0.0):
        size = np.asarray(specification["size"], dtype=float)
        if specification["shape"] == "box":
            return coal.Box(*(2.0 * (size + margin)))
        if specification["shape"] == "sphere":
            return coal.Sphere(float(size[0] + margin))
        if specification["shape"] == "cylinder":
            return coal.Cylinder(float(size[0] + margin), float(size[1] + margin))
        raise ValueError(f"Unsupported collision shape {specification['shape']!r}")

    def _load_environment(
        self, task_config: str | Path
    ) -> tuple[dict[str, _EnvironmentGeometry], str]:
        with Path(task_config).open("rb") as stream:
            task = tomllib.load(stream)
        specifications = [
            {
                "name": "table",
                "shape": "box",
                "position": task["table"]["position"],
                "size": task["table"]["half_size"],
            },
            *task.get("obstacles", []),
            *task["objects"],
        ]
        environment: dict[str, _EnvironmentGeometry] = {}
        target_name = ""
        for specification in specifications:
            name = specification["name"]
            is_target = bool(specification.get("grasp_target", False))
            if is_target:
                if target_name:
                    raise ValueError("Task contains more than one grasp target")
                target_name = name
            environment[name] = _EnvironmentGeometry(
                name=name,
                geometry=self._shape(
                    specification,
                    self._target_margin if is_target else self._environment_margin,
                ),
                transform=coal.Transform3s(
                    np.eye(3), np.asarray(specification["position"], dtype=float)
                ),
                is_target=is_target,
                actual_geometry=self._shape(specification, 0.0),
            )
        if not target_name:
            raise ValueError("Task contains no grasp target")
        return environment, target_name

    def set_environment_pose(self, name: str, pose: Pose) -> None:
        """Update a planning-scene object from truth data or an external pose."""
        try:
            item = self._environment[name]
        except KeyError as exc:
            raise KeyError(f"Unknown collision object {name!r}") from exc
        rotation = self.kinematics._rotation_from_wxyz(pose.quaternion)
        item.transform = coal.Transform3s(rotation, pose.position)

    def _link_name(self, geometry_id: int) -> str:
        geometry = self.geometry_model.geometryObjects[geometry_id]
        return self.model.frames[geometry.parentFrame].name

    def _robot_transform(self, geometry_id: int) -> coal.Transform3s:
        """Map Pinocchio geometry placement into the MuJoCo world frame."""
        placement = self.geometry_data.oMg[geometry_id]
        rotation = self.kinematics._world_rotation @ placement.rotation
        translation = (
            self.kinematics._world_rotation @ placement.translation
            + self.kinematics._world_translation
        )
        return coal.Transform3s(rotation, translation)

    def _adjacent(self, first_id: int, second_id: int) -> bool:
        first = self.geometry_model.geometryObjects[first_id].parentJoint
        second = self.geometry_model.geometryObjects[second_id].parentJoint
        if first == second:
            return True
        return (
            int(self.model.parents[first]) == second
            or int(self.model.parents[second]) == first
        )

    def _self_pair_allowed(self, first_id: int, second_id: int) -> bool:
        if self._adjacent(first_id, second_id):
            return True
        names = frozenset((self._link_name(first_id), self._link_name(second_id)))
        return names in self._allowed_link_pairs

    def _target_contact_allowed(
        self, link_name: str, environment_name: str, phase: CollisionPhase
    ) -> bool:
        return (
            phase is CollisionPhase.GRASP
            and environment_name == self.target_name
            and link_name in self._touch_links
        )

    def _collide(self, first_geometry, first_transform, second_geometry, second_transform) -> bool:
        result = coal.CollisionResult()
        return bool(
            coal.collide(
                first_geometry,
                first_transform,
                second_geometry,
                second_transform,
                self._request,
                result,
            )
        )

    def check(
        self,
        q: np.ndarray,
        phase: CollisionPhase = CollisionPhase.FREE_SPACE,
        *,
        joint_positions: dict[str, float] | None = None,
        stop_at_first: bool = False,
    ) -> CollisionReport:
        """Return every disallowed collision at one robot configuration.

        ``q`` contains the planner's active group.  ``joint_positions`` can
        additionally override gripper or other scalar joints.  This matters
        during grasp planning: checking an arm pose with an open hand while
        executing it with a pre-shaped hand is not a valid collision test.
        """
        configuration = self.kinematics._configuration(q)
        if joint_positions:
            for name, value in joint_positions.items():
                joint_id = self.model.getJointId(name)
                if joint_id == 0 or self.model.joints[joint_id].nq != 1:
                    raise KeyError(f"Unknown scalar collision joint {name!r}")
                configuration[self.model.idx_qs[joint_id]] = float(value)
        pin.updateGeometryPlacements(
            self.model,
            self.data,
            self.geometry_model,
            self.geometry_data,
            configuration,
        )
        contacts: list[CollisionContact] = []
        reported: set[tuple[str, str, str]] = set()
        active = sorted(self._active_geometry_ids)
        all_geometry = self._robot_geometry_ids
        for first_id in active:
            for second_id in all_geometry:
                if second_id == first_id:
                    continue
                # Only active-active pairs can be visited twice. A fixed-body
                # id may be smaller than the active id and must not be skipped.
                if second_id in self._active_geometry_ids and second_id < first_id:
                    continue
                if self._self_pair_allowed(first_id, second_id):
                    continue
                first = self.geometry_model.geometryObjects[first_id]
                second = self.geometry_model.geometryObjects[second_id]
                if self._collide(
                    first.geometry,
                    self._robot_transform(first_id),
                    second.geometry,
                    self._robot_transform(second_id),
                ):
                    first_name = self._link_name(first_id)
                    second_name = self._link_name(second_id)
                    key = (first_name, second_name, "self")
                    if key not in reported:
                        reported.add(key)
                        contacts.append(CollisionContact(*key))
                    if stop_at_first:
                        return CollisionReport(tuple(contacts))
        for robot_id in active:
            robot_geometry = self.geometry_model.geometryObjects[robot_id]
            link_name = self._link_name(robot_id)
            for environment in self._environment.values():
                if self._target_contact_allowed(link_name, environment.name, phase):
                    continue
                environment_geometry = (
                    environment.actual_geometry
                    if phase is CollisionPhase.GRASP and environment.is_target
                    else environment.geometry
                )
                if self._collide(
                    robot_geometry.geometry,
                    self._robot_transform(robot_id),
                    environment_geometry,
                    environment.transform,
                ):
                    key = (link_name, environment.name, "environment")
                    if key not in reported:
                        reported.add(key)
                        contacts.append(CollisionContact(*key))
                    if stop_at_first:
                        return CollisionReport(tuple(contacts))
        return CollisionReport(tuple(contacts))

    def edge_is_valid(
        self,
        start: np.ndarray,
        goal: np.ndarray,
        phase: CollisionPhase = CollisionPhase.FREE_SPACE,
        *,
        joint_positions: dict[str, float] | None = None,
        max_joint_step: float = 0.04,
    ) -> bool:
        """Sample the complete edge; valid endpoints alone are insufficient."""
        start, goal = np.asarray(start, float), np.asarray(goal, float)
        steps = max(1, int(np.ceil(np.max(np.abs(goal - start)) / max_joint_step)))
        for alpha in np.linspace(0.0, 1.0, steps + 1):
            if self.check(
                start + alpha * (goal - start),
                phase,
                joint_positions=joint_positions,
                stop_at_first=True,
            ).in_collision:
                return False
        return True
