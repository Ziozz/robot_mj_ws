"""Configuration-backed robot description with no G1-specific planner logic."""

from dataclasses import dataclass
from pathlib import Path
import tomllib

import numpy as np


@dataclass(frozen=True)
class EndEffector:
    parent_body: str
    position: np.ndarray
    quaternion: np.ndarray
    touch_bodies: tuple[str, ...]


@dataclass(frozen=True)
class ControlGains:
    kp: np.ndarray
    kd: np.ndarray
    velocity_gain: np.ndarray
    effort_limit: np.ndarray


class RobotDescription:
    """Names and parameters needed to adapt an arbitrary MuJoCo robot."""

    def __init__(
        self,
        *,
        name: str,
        model_path: Path,
        home_keyframe: str,
        groups: dict[str, tuple[str, ...]],
        end_effectors: dict[str, EndEffector],
        control: dict[str, ControlGains],
    ) -> None:
        self.name = name
        self.model_path = model_path
        self.home_keyframe = home_keyframe
        self.groups = groups
        self.end_effectors = end_effectors
        self.control = control

    @classmethod
    def from_toml(cls, config_path: str | Path, project_root: str | Path | None = None):
        config_path = Path(config_path).resolve()
        with config_path.open("rb") as stream:
            raw = tomllib.load(stream)
        root = Path(project_root).resolve() if project_root else config_path.parents[2]

        groups = {
            name: tuple(value["joints"])
            for name, value in raw.get("groups", {}).items()
        }
        effectors = {
            name: EndEffector(
                parent_body=value["parent_body"],
                position=np.asarray(value["position"], dtype=float),
                quaternion=np.asarray(value["quaternion"], dtype=float),
                touch_bodies=tuple(value.get("touch_bodies", ())),
            )
            for name, value in raw.get("end_effectors", {}).items()
        }
        control = {
            name: ControlGains(
                kp=np.asarray(value["kp"], dtype=float),
                kd=np.asarray(value["kd"], dtype=float),
                velocity_gain=np.asarray(value["velocity_gain"], dtype=float),
                effort_limit=np.asarray(value["effort_limit"], dtype=float),
            )
            for name, value in raw.get("control", {}).items()
        }
        result = cls(
            name=raw["name"],
            model_path=root / raw["model"],
            home_keyframe=raw["home_keyframe"],
            groups=groups,
            end_effectors=effectors,
            control=control,
        )
        result._validate_lengths()
        return result

    def _validate_lengths(self) -> None:
        for group, gains in self.control.items():
            dof = len(self.group(group))
            for name in ("kp", "kd", "velocity_gain", "effort_limit"):
                if getattr(gains, name).shape != (dof,):
                    raise ValueError(f"control.{group}.{name} must contain {dof} values")

    def group(self, name: str) -> tuple[str, ...]:
        try:
            return self.groups[name]
        except KeyError as exc:
            raise KeyError(f"Unknown joint group {name!r}; available: {sorted(self.groups)}") from exc

