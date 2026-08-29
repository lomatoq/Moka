"""Canonical rig, forward kinematics and topology-restricted skinning.

Coordinates are image pixels, X right / Y down. Rotation deltas are degrees,
clockwise. A bone's setup transform is derived from its start/end joints, never
from guessed Euler axes. Exporters are responsible for coordinate conversion.
"""
from __future__ import annotations
import math
import re
from typing import Any
import numpy as np

SAFE_ID = re.compile(r"^[A-Za-z][A-Za-z0-9_-]{0,63}$")


def template_rig(width: int, height: int, preset: str = "biped", bbox=None) -> dict:
    if preset not in {"biped", "creature", "quadruped"}:
        raise ValueError("Unknown rig preset")
    x, y, w, h = bbox or (width * .16, height * .04, width * .68, height * .92)
    # L/R are the character's anatomical left/right, not the viewer's.
    points = {
        "pelvis": (.50, .56), "chest": (.50, .32), "neck": (.50, .23),
        "head_tip": (.50, .025),
        "shoulder_l": (.69, .31), "elbow_l": (.80, .44),
        "wrist_l": (.90, .58), "hand_l": (.94, .64),
        "shoulder_r": (.31, .31), "elbow_r": (.20, .44),
        "wrist_r": (.10, .58), "hand_r": (.06, .64),
        "hip_l": (.60, .56), "knee_l": (.62, .75),
        "ankle_l": (.65, .93), "toe_l": (.76, .97),
        "hip_r": (.40, .56), "knee_r": (.38, .75),
        "ankle_r": (.35, .93), "toe_r": (.24, .97),
    }
    if preset == "quadruped":
        # Side-view scaffold: front legs use the arm roles, rear legs the leg
        # roles. No claim that a human pose model detects animal anatomy.
        points.update({"pelvis": (.31, .43), "chest": (.67, .37),
            "neck": (.77, .25), "head_tip": (.89, .11),
            "shoulder_l": (.71, .43), "elbow_l": (.76, .66),
            "wrist_l": (.76, .87), "hand_l": (.87, .92),
            "shoulder_r": (.63, .43), "elbow_r": (.65, .64),
            "wrist_r": (.65, .84), "hand_r": (.74, .89),
            "hip_l": (.30, .48), "knee_l": (.40, .66),
            "ankle_l": (.27, .87), "toe_l": (.40, .94),
            "hip_r": (.23, .46), "knee_r": (.30, .64),
            "ankle_r": (.18, .83), "toe_r": (.28, .90)})
    bones = [
        ("root", None, "pelvis", "pelvis", False),
        ("torso", "root", "pelvis", "chest", True),
        ("neck", "torso", "chest", "neck", True),
        ("head", "neck", "neck", "head_tip", True),
    ]
    for side in ("r", "l"):
        bones += [
            (f"collar_{side}", "torso", "chest", f"shoulder_{side}", False),
            (f"upper_arm_{side}", f"collar_{side}", f"shoulder_{side}", f"elbow_{side}", True),
            (f"forearm_{side}", f"upper_arm_{side}", f"elbow_{side}", f"wrist_{side}", True),
            (f"hand_{side}", f"forearm_{side}", f"wrist_{side}", f"hand_{side}", True),
            (f"hip_link_{side}", "root", "pelvis", f"hip_{side}", False),
            (f"thigh_{side}", f"hip_link_{side}", f"hip_{side}", f"knee_{side}", True),
            (f"shin_{side}", f"thigh_{side}", f"knee_{side}", f"ankle_{side}", True),
            (f"foot_{side}", f"shin_{side}", f"ankle_{side}", f"toe_{side}", True),
        ]
    if preset in {"creature", "quadruped"}:
        points.update({"tail_mid": (.12, .42), "tail_tip": (.01, .24)})
        bones += [("tail_base", "root", "pelvis", "tail_mid", True),
                  ("tail_tip", "tail_base", "tail_mid", "tail_tip", True)]
    return {
        "preset": preset, "provenance": "template", "confidence": 0.0,
        "joints": [{"id": k, "x": round(x + a * w, 3), "y": round(y + b * h, 3),
                    "confidence": 0.0, "source": "template"} for k, (a, b) in points.items()],
        "bones": [{"id": n, "parent": p, "start": a, "end": b, "layer": layer}
                  for n, p, a, b, layer in bones],
    }


def validate_rig(rig: dict, width: int = 4096, height: int = 4096) -> None:
    if not isinstance(rig, dict):
        raise ValueError("Rig must be an object")
    joints, bones = rig.get("joints", []), rig.get("bones", [])
    if not 1 <= len(joints) <= 128 or not 1 <= len(bones) <= 64:
        raise ValueError("Rig must contain 1–128 joints and 1–64 bones")
    joint_ids, seen = set(), set()
    for j in joints:
        name = j.get("id", "")
        if not SAFE_ID.fullmatch(name) or name in joint_ids:
            raise ValueError("Joint identifiers must be unique safe names")
        joint_ids.add(name)
        for axis, limit in (("x", width), ("y", height)):
            value = j.get(axis)
            if not isinstance(value, (int, float)) or not math.isfinite(value) or not -limit <= value <= 2 * limit:
                raise ValueError(f"Invalid {axis} coordinate for {name}")
    roots = 0
    for b in bones:
        name = b.get("id", "")
        if not SAFE_ID.fullmatch(name) or name in seen:
            raise ValueError("Bone identifiers must be unique safe names")
        parent = b.get("parent")
        if parent is None:
            roots += 1
        elif parent not in seen:
            raise ValueError("Parent bones must precede children; cyclic rigs are not allowed")
        if b.get("start") not in joint_ids or b.get("end") not in joint_ids:
            raise ValueError(f"Unknown joint for bone {name}")
        radius = b.get("radius")
        if radius is not None and (not isinstance(radius, (int, float)) or not math.isfinite(radius) or not 1 <= radius <= max(width, height)):
            raise ValueError("Bone radius must be a finite positive pixel value")
        seen.add(name)
    if roots != 1 or bones[0]["id"] != "root":
        raise ValueError("Exactly one root named 'root' is required")


def setup_transforms(rig: dict) -> dict[str, dict]:
    joints = {j["id"]: j for j in rig["joints"]}
    out = {}
    for b in rig["bones"]:
        a, z = joints[b["start"]], joints[b["end"]]
        angle = math.atan2(z["y"] - a["y"], z["x"] - a["x"]) if b["start"] != b["end"] else 0.0
        out[b["id"]] = {"x": a["x"], "y": a["y"], "angle": angle,
                           "length": math.hypot(z["x"] - a["x"], z["y"] - a["y"])}
    return out


def local_point(point, transform):
    dx, dy = point[0] - transform["x"], point[1] - transform["y"]
    c, s = math.cos(transform["angle"]), math.sin(transform["angle"])
    return [c * dx + s * dy, -s * dx + c * dy]


def world_point(point, transform):
    c, s = math.cos(transform["angle"]), math.sin(transform["angle"])
    return [transform["x"] + c * point[0] - s * point[1],
            transform["y"] + s * point[0] + c * point[1]]


def pose_transforms(rig: dict, frame: dict | None = None) -> dict:
    frame = frame or {}
    bind, posed = setup_transforms(rig), {}
    angles = frame.get("angles", {})
    root = frame.get("root", [0, 0])
    for b in rig["bones"]:
        own = bind[b["id"]]
        delta = math.radians(angles.get(b["id"], 0))
        if b.get("parent") is None:
            posed[b["id"]] = {**own, "x": own["x"] + root[0], "y": own["y"] + root[1], "angle": own["angle"] + delta}
        else:
            parent = b["parent"]
            offset = local_point([own["x"], own["y"]], bind[parent])
            x, y = world_point(offset, posed[parent])
            posed[b["id"]] = {**own, "x": x, "y": y,
                              "angle": posed[parent]["angle"] + own["angle"] - bind[parent]["angle"] + delta}
    return posed


def make_mesh(alpha: np.ndarray, bbox: list[int], bone_id: str, rig: dict, rigid: bool = False) -> dict:
    """Regular textured grid; only occupied cells are retained.

    Skinning may use the owning bone and directly connected parent/child only.
    Opposite limbs cannot influence each other just because they overlap.
    """
    x, y, w, h = bbox
    if w < 1 or h < 1:
        return {"vertices": [], "uvs": [], "triangles": [], "weights": []}
    step = max(8, int(math.sqrt(w * h / 90)))
    cols, rows = max(1, math.ceil(w / step)), max(1, math.ceil(h / step))
    xs, ys = np.linspace(0, w, cols + 1), np.linspace(0, h, rows + 1)
    vertices = [[round(x + xx, 4), round(y + yy, 4)] for yy in ys for xx in xs]
    uvs = [[round(xx / w, 7), round(yy / h, 7)] for yy in ys for xx in xs]
    triangles = []
    for r in range(rows):
        for c in range(cols):
            patch = alpha[int(ys[r]):max(int(math.ceil(ys[r + 1])), int(ys[r]) + 1),
                          int(xs[c]):max(int(math.ceil(xs[c + 1])), int(xs[c]) + 1)]
            if not patch.size or not np.any(patch):
                continue
            a = r * (cols + 1) + c
            triangles.extend([a, a + 1, a + cols + 2, a, a + cols + 2, a + cols + 1])
    bind = setup_transforms(rig)
    bones = {b["id"]: b for b in rig["bones"]}
    own = bones[bone_id]
    parent = bones.get(own.get("parent"))
    children = [b for b in rig["bones"] if b.get("parent") == bone_id and b.get("layer")]
    length = max(bind[bone_id]["length"], 1)
    weights = []
    for v in vertices:
        t = local_point(v, bind[bone_id])[0] / length
        influences = []
        if not rigid and parent and parent.get("layer") and t < .18:
            influences.append({"bone": parent["id"], "weight": min(.35, max(0, (.18 - t) * .8))})
        if not rigid and len(children) == 1 and t > .82:
            influences.append({"bone": children[0]["id"], "weight": min(.35, max(0, (t - .82) * .8))})
        influences.append({"bone": bone_id, "weight": 1 - sum(i["weight"] for i in influences)})
        weights.append(influences)
    return {"vertices": vertices, "uvs": uvs, "triangles": triangles, "weights": weights}


def deform_mesh(mesh: dict, rig: dict, frame=None) -> list:
    bind, posed = setup_transforms(rig), pose_transforms(rig, frame)
    result = []
    for v, influences in zip(mesh["vertices"], mesh["weights"]):
        p = [0.0, 0.0]
        for item in influences:
            k, w = item["bone"], item["weight"]
            q = world_point(local_point(v, bind[k]), posed[k])
            p[0] += q[0] * w
            p[1] += q[1] * w
        result.append(p)
    return result
