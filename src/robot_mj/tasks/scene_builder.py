"""Create a deterministic G1 pick-and-place scene without editing source assets."""

from pathlib import Path
import os
import tomllib
import xml.etree.ElementTree as ET

import numpy as np


def _numbers(values) -> str:
    return " ".join(f"{float(value):.9g}" for value in values)


def _rgba(values) -> str:
    return _numbers(values)


def build_g1_pick_place_scene(
    source: Path, destination: Path, task_config: Path
) -> Path:
    """Lock the unsupported floating base and add task objects and sites."""
    source, destination = Path(source).resolve(), Path(destination).resolve()
    with Path(task_config).open("rb") as stream:
        task = tomllib.load(stream)
    tree = ET.parse(source)
    root = tree.getroot()
    compiler = root.find("compiler")
    if compiler is None:
        raise ValueError("G1 MJCF has no compiler element")
    mesh_dir = (source.parent / compiler.get("meshdir", "meshes")).resolve()
    compiler.set("meshdir", os.path.relpath(mesh_dir, destination.parent))

    pelvis = root.find(".//body[@name='pelvis']")
    freejoint = pelvis.find("freejoint") if pelvis is not None else None
    if freejoint is not None:
        pelvis.remove(freejoint)

    wrist = root.find(".//body[@name='right_wrist_yaw_link']")
    if wrist is None:
        raise ValueError("G1 right wrist body is missing")
    ET.SubElement(
        wrist,
        "site",
        name="right_tcp",
        pos="0.14 -0.003 0",
        size="0.008",
        rgba="0 1 0 1",
    )

    world = root.findall("worldbody")[-1]
    table = task["table"]
    ET.SubElement(
        world, "geom", name="table", type="box",
        pos=_numbers(table["position"]), size=_numbers(table["half_size"]),
        rgba=_rgba(table["rgba"]), contype="1", conaffinity="1",
    )
    for obstacle in task.get("obstacles", []):
        ET.SubElement(
            world, "geom", name=obstacle["name"], type=obstacle["shape"],
            pos=_numbers(obstacle["position"]), size=_numbers(obstacle["size"]),
            rgba=_rgba(obstacle["rgba"]), contype="1", conaffinity="1",
        )

    target = None
    object_qpos = []
    for specification in task["objects"]:
        name = specification["name"]
        position = np.asarray(specification["position"], dtype=float)
        body = ET.SubElement(world, "body", name=name, pos=_numbers(position))
        ET.SubElement(body, "freejoint", name=f"{name}_joint")
        ET.SubElement(
            body, "geom", name=f"{name}_geom", type=specification["shape"],
            size=_numbers(specification["size"]), mass=str(specification["mass"]),
            rgba=_rgba(specification["rgba"]),
            friction=_numbers(specification["friction"]),
        )
        object_qpos.extend((*position, 1.0, 0.0, 0.0, 0.0))
        if specification.get("grasp_target", False):
            if target is not None:
                raise ValueError("Task config must contain exactly one grasp target")
            target = specification
    if target is None:
        raise ValueError("Task config contains no grasp_target object")

    goal_specification = task["place_goal"]
    goal = ET.SubElement(
        world, "body", name="place_goal",
        pos=_numbers(goal_specification["position"]),
    )
    ET.SubElement(
        goal, "geom", name="place_goal_marker", type="cylinder",
        size=_numbers([goal_specification["radius"], 0.002]),
        rgba=_rgba(goal_specification["rgba"]), contype="0", conaffinity="0",
    )

    contact = root.find("contact")
    if contact is None:
        contact = ET.SubElement(root, "contact")
    for side in ("left", "right"):
        ET.SubElement(
            contact, "exclude", body1=f"{side}_wrist_yaw_link",
            body2=f"{side}_hand_thumb_1_link",
        )

    # The original keyframe starts with a 7-qpos floating base. Removing it and
    # appending the object's free joint keeps reset-to-stand deterministic.
    stand = root.find(".//key[@name='stand']")
    if stand is not None:
        old = np.fromstring(stand.get("qpos", ""), sep=" ")
        stand.set("qpos", _numbers(np.r_[old[7:], object_qpos]))

    destination.parent.mkdir(parents=True, exist_ok=True)
    ET.indent(tree, space="  ")
    tree.write(destination, encoding="unicode")
    return destination
