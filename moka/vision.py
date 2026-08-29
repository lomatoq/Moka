"""Rig-conditioned layered decomposition, independent of any AI provider.

The CPU baseline is NOT a learned semantic model. It minimizes an approximate
edge-aware Potts energy with anatomy-derived unary costs, preserves original
visible pixels, and extrapolates only hidden joint overlap. Learned proposals
can improve the unary terms; they never redefine the canonical skeleton.
"""
from __future__ import annotations
import math
from pathlib import Path
from typing import Callable
import cv2
import numpy as np
from PIL import Image
from scipy import ndimage
from .rig import make_mesh, validate_rig


class Cancelled(Exception):
    pass


def foreground(image: Image.Image, mode: str = "auto") -> tuple[Image.Image, str, list[str]]:
    rgba = np.array(image.convert("RGBA"))
    if mode == "keep" or np.min(rgba[..., 3]) < 250:
        return Image.fromarray(rgba), "source-alpha", []
    # Only remove border-connected near-background pixels, never interior white.
    rgb = rgba[..., :3]
    corners = np.concatenate([rgb[:8, :8].reshape(-1, 3), rgb[:8, -8:].reshape(-1, 3),
                              rgb[-8:, :8].reshape(-1, 3), rgb[-8:, -8:].reshape(-1, 3)])
    bg = np.median(corners, axis=0)
    distance = np.linalg.norm(rgb.astype(np.float32) - bg, axis=2)
    candidates = distance < 34
    labels, _ = ndimage.label(candidates)
    border = np.unique(np.concatenate([labels[0], labels[-1], labels[:, 0], labels[:, -1]]))
    background = np.isin(labels, border[border != 0])
    ratio = background.mean()
    if ratio < .02:
        return image.convert("RGBA"), "opaque-unresolved", ["Complex background was not removed. Use a transparent PNG, a mask, or the optional rembg engine."]
    rgba[background, 3] = 0
    # Keep actual source RGB, even for the newly transparent pixels.
    return Image.fromarray(rgba), "border-connected-background", ["Background removal used a flat-border heuristic, not AI segmentation."]


def image_bbox(image: Image.Image):
    box = image.getchannel("A").getbbox()
    if box is None:
        raise ValueError("The image is fully transparent")
    x, y, x2, y2 = box
    return [x, y, x2-x, y2-y]


def _distance_field(xx, yy, a, b):
    vx, vy = b[0] - a[0], b[1] - a[1]
    length2 = max(vx * vx + vy * vy, 1e-6)
    t = ((xx - a[0]) * vx + (yy - a[1]) * vy) / length2
    tc = np.clip(t, 0, 1)
    d2 = (xx - a[0] - tc * vx) ** 2 + (yy - a[1] - tc * vy) ** 2
    return d2, t


def _radius(bone, joints, silhouette_distance, scale):
    a, z = joints[bone["start"]], joints[bone["end"]]
    length = math.hypot(z[0] - a[0], z[1] - a[1])
    name = bone["id"]
    factor = .23
    if name == "head": factor = .64
    elif name == "torso": factor = .44
    elif name == "neck": factor = .38
    elif name.startswith("hand"): factor = .75
    elif name.startswith("foot"): factor = .52
    elif name.startswith("tail"): factor = .19
    default = max(4, length * factor)
    if bone.get("radius") is not None:
        return max(2, bone["radius"] * scale)
    h, w = silhouette_distance.shape
    sampled = []
    for t in (.3, .5, .7):
        x = int(np.clip(a[0] + (z[0] - a[0]) * t, 0, w-1))
        y = int(np.clip(a[1] + (z[1] - a[1]) * t, 0, h-1))
        if silhouette_distance[y, x] > 1:
            sampled.append(silhouette_distance[y, x])
    if sampled:
        return float(np.clip(np.median(sampled), default * .72, default * 1.45))
    return default


def _check(cancel):
    if cancel and cancel():
        raise Cancelled("Operation cancelled; the previous project was preserved")


def partition(image: Image.Image, rig: dict, work_size=640, edge_weight=.9,
              priors: dict | None = None, progress=None, cancel=None):
    """Approximate multi-label MAP using checkerboard ICM, fixed anatomy seeds.

    Pairwise terms prefer boundaries at RGB/Lab edges. Unaries penalize distance
    from physical bone segments, so an 'arm' class can still split at the elbow.
    Processing is bounded at work_size; masks return at source resolution.
    """
    validate_rig(rig, *image.size)
    width, height = image.size
    scale = min(1.0, work_size / max(width, height))
    w, h = max(1, round(width * scale)), max(1, round(height * scale))
    arr = np.array(image.resize((w, h), Image.Resampling.LANCZOS).convert("RGBA"))
    fg = arr[..., 3] > 8
    if fg.sum() < 8:
        raise ValueError("No usable foreground pixels")
    joints = {j["id"]: (j["x"] * scale, j["y"] * scale) for j in rig["joints"]}
    bones = [b for b in rig["bones"] if b.get("layer", True) and b["start"] != b["end"]]
    if not bones:
        raise ValueError("The rig has no drawable bone segments")
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    dt = ndimage.distance_transform_edt(fg)
    unary, radii, axial = [], {}, []
    for b in bones:
        d2, t = _distance_field(xx, yy, joints[b["start"]], joints[b["end"]])
        r = _radius(b, joints, dt, scale)
        radii[b["id"]] = r / scale
        # Additional axial penalty keeps a short hand/foot from eating a forearm.
        u = d2 / max(r * r, 1) + .08 * (t - .5) ** 2
        if priors and b["id"] in priors:
            proposal = np.array(Image.fromarray(priors[b["id"]]).resize((w, h), Image.Resampling.BILINEAR)) / 255
            u += 1.8 * (1 - proposal)
        unary.append(u.astype(np.float32))
        axial.append(t)
    unary = np.stack(unary)
    labels = np.argmin(unary, axis=0).astype(np.int16)
    labels[~fg] = -1
    seeds = np.full((h, w), -1, np.int16)
    # Seeds are chosen only inside the foreground and only where geometry
    # already prefers that bone. A hidden limb is not invented as 'detected'.
    for k, b in enumerate(bones):
        core = fg & (labels == k) & (unary[k] < .10) & (axial[k] > .22) & (axial[k] < .78)
        seeds[core] = k
    lab = cv2.cvtColor(arr[..., :3], cv2.COLOR_RGB2LAB).astype(np.float32) / 255
    dx = np.sum((lab[:, 1:] - lab[:, :-1]) ** 2, axis=2)
    dy = np.sum((lab[1:] - lab[:-1]) ** 2, axis=2)
    wx = edge_weight * np.exp(-dx * 100)
    wy = edge_weight * np.exp(-dy * 100)
    wx *= fg[:, 1:] & fg[:, :-1]
    wy *= fg[1:] & fg[:-1]
    parity = ((xx.astype(int) + yy.astype(int)) % 2)
    ids = np.arange(len(bones))[:, None, None]
    for iteration in range(8):
        _check(cancel)
        changed = 0
        for phase in (0, 1):
            costs = unary.copy()
            costs[:, :, 1:] += wx[None] * (ids != labels[None, :, :-1])
            costs[:, :, :-1] += wx[None] * (ids != labels[None, :, 1:])
            costs[:, 1:, :] += wy[None] * (ids != labels[None, :-1, :])
            costs[:, :-1, :] += wy[None] * (ids != labels[None, 1:, :])
            proposal = np.argmin(costs, axis=0).astype(np.int16)
            update = fg & (parity == phase) & (seeds < 0)
            changed += int(np.count_nonzero(labels[update] != proposal[update]))
            labels[update] = proposal[update]
        if progress:
            progress(.12 + .40 * (iteration+1)/8, f"Refining anatomy boundaries · pass {iteration+1}/8")
        if changed == 0:
            break
    full = cv2.resize(labels.astype(np.float32), (width, height), interpolation=cv2.INTER_NEAREST).astype(np.int16)
    source_alpha = np.array(image.getchannel("A"))
    # Downsampling may lose tiny antialiased islands. Assign them by nearest
    # high-resolution skeleton segment rather than dropping original pixels.
    holes = (source_alpha > 0) & (full < 0)
    if np.any(holes):
        hi_y, hi_x = np.nonzero(holes)
        values = []
        original_joints = {j["id"]: (j["x"], j["y"]) for j in rig["joints"]}
        for b in bones:
            d2, _ = _distance_field(hi_x, hi_y, original_joints[b["start"]], original_joints[b["end"]])
            values.append(d2 / radii[b["id"]] ** 2)
        full[holes] = np.argmin(np.stack(values), axis=0)
    full[source_alpha == 0] = -1
    sorted_costs = np.partition(unary, min(1, len(bones)-1), axis=0)
    margin = sorted_costs[1] - sorted_costs[0] if len(bones) > 1 else np.ones((h, w))
    return full, bones, radii, float(np.mean(margin[fg] < .12))


def extend_texture(rgba: np.ndarray, visible: np.ndarray, extension: np.ndarray) -> np.ndarray:
    """Nearest own-part texture + local Telea smoothing. NOT generative AI.

    Other parts' colors never enter the extrapolation. Known visible texels are
    restored byte-for-byte after filling. Large fills must be reviewed.
    """
    result = np.zeros_like(rgba)
    if not np.any(visible):
        return result
    known = visible > 0
    _, indices = ndimage.distance_transform_edt(~known, return_indices=True)
    nearest = rgba[..., :3][indices[0], indices[1]].copy()
    ext = extension.astype(np.uint8)
    if np.any(ext):
        nearest = cv2.inpaint(nearest, ext, 3, cv2.INPAINT_TELEA)
    result[..., :3] = nearest
    result[..., 3] = np.maximum(visible, ext)
    result[known] = rgba[known]
    result[..., 3][known] = visible[known]
    return result


def default_order(names):
    sequence = ["tail_tip", "tail_base", "upper_arm_r", "forearm_r", "hand_r",
                "thigh_r", "shin_r", "foot_r", "torso", "thigh_l", "shin_l",
                "foot_l", "upper_arm_l", "forearm_l", "hand_l", "neck", "head"]
    known = [n for n in sequence if n in names]
    return known + [n for n in names if n not in known]


def decompose(image: Image.Image, rig: dict, output: Path, *, padding=18,
              work_size=640, edge_weight=.9, rigid=False, priors=None,
              progress: Callable | None = None, cancel=None) -> tuple[list, dict, list]:
    output.mkdir(parents=True, exist_ok=True)
    labels, bones, radii, ambiguous = partition(image, rig, work_size, edge_weight, priors, progress, cancel)
    rgba = np.array(image.convert("RGBA"))
    alpha = rgba[..., 3]
    joints = {j["id"]: (j["x"], j["y"]) for j in rig["joints"]}
    names = default_order([b["id"] for b in bones])
    by_name = {b["id"]: b for b in bones}
    index = {b["id"]: k for k, b in enumerate(bones)}
    rank = {name: i for i, name in enumerate(names)}
    pixel_rank = np.full(labels.shape, -1, np.int16)
    for name, k in index.items():
        pixel_rank[labels == k] = rank[name]
    layers, warnings = [], []
    total_fill, total_visible = 0, int(np.count_nonzero(alpha))
    for ordinal, name in enumerate(names):
        _check(cancel)
        k = index[name]
        visible = np.where(labels == k, alpha, 0).astype(np.uint8)
        area = int(np.count_nonzero(visible))
        if area == 0:
            warnings.append(f"No visible pixels for {name}; check its joints or add a learned/painted mask.")
            continue
        b = by_name[name]
        extension = np.zeros_like(alpha)
        if padding > 0:
            a, z = joints[b["start"]], joints[b["end"]]
            # Extrapolate joint overlap, not an entirely invented limb.
            support = np.zeros_like(alpha)
            rr = max(2, round(min(radii[name], padding * 1.8)))
            for px, py in (a, z):
                cv2.circle(support, (round(px), round(py)), rr, 255, -1)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*int(padding)+1, 2*int(padding)+1))
            close_to_part = cv2.dilate((visible > 0).astype(np.uint8), kernel) > 0
            # Fill only below fully opaque higher-order ORIGINAL pixels. This
            # guarantees rest-pose reconstruction, including alpha boundaries.
            hidden = (pixel_rank > rank[name]) & (alpha == 255)
            extension[(support > 0) & close_to_part & hidden & (visible == 0)] = 255
        box = Image.fromarray(np.maximum(visible, extension)).getbbox()
        x, y, x2, y2 = box
        # Fill in a cropped region to avoid allocating a full distance field
        # per part for 4K input.
        crop = rgba[y:y2, x:x2]
        vis_crop = visible[y:y2, x:x2]
        ext_crop = extension[y:y2, x:x2]
        filled = extend_texture(crop, vis_crop, ext_crop)
        path = f"{name}.png"
        Image.fromarray(filled).save(output / path)
        Image.fromarray(vis_crop).save(output / f"{name}.visible.png")
        Image.fromarray(ext_crop).save(output / f"{name}.fill.png")
        bbox = [x, y, x2-x, y2-y]
        fill_count = int(np.count_nonzero(ext_crop))
        total_fill += fill_count
        mesh = make_mesh(filled[..., 3], bbox, name, rig, rigid)
        layers.append({"id": name, "name": name.replace("_", " "), "bone": name,
            "image": path, "visible_mask": f"{name}.visible.png", "fill_mask": f"{name}.fill.png",
            "bbox": bbox, "order": ordinal, "opacity": 1, "visible": True, "mesh": mesh,
            "visible_pixels": area, "fill_pixels": fill_count,
            "fill_fraction": round(fill_count/max(area+fill_count, 1), 4),
            "needs_review": area < max(20, total_visible*.001) or fill_count > area*.25,
            "provenance": "rig-conditioned-cpu", "completion": "own-texture-edge-extension"})
        if progress:
            progress(.55 + .42 * (ordinal+1)/len(names), f"Building {name} · mask, overlap, weighted mesh")
    quality = {"visible_coverage": round(sum(l["visible_pixels"] for l in layers) / max(total_visible, 1), 6),
               "ambiguous_geometry_fraction": round(ambiguous, 4),
               "filled_pixels": total_fill, "source_pixels": total_visible,
               "expected_parts": len(bones), "produced_parts": len(layers),
               "method": "rig-conditioned-cpu", "automatic_semantic_accuracy": None}
    if rig.get("provenance") == "template":
        warnings.insert(0, "Skeleton is an unverified template, not detected anatomy. Align joints before judging the cut.")
    warnings.append("Hidden overlap uses texture extrapolation, not learned reconstruction. Inspect the amber fill overlay.")
    return layers, quality, warnings


def paint_layer(image: Image.Image, layer: dict, strokes: list[dict], root: Path, rig: dict):
    """Edit masks without destroying unpainted, already completed hidden artwork."""
    w, h = image.size
    rgba = np.array(image.convert("RGBA"))
    art = np.zeros((h, w, 4), np.uint8)
    visible = np.zeros((h, w), np.uint8); fill = np.zeros((h, w), np.uint8)
    x, y, lw, lh = map(int, layer["bbox"])
    art[y:y+lh, x:x+lw] = np.array(Image.open(root/layer["image"]).convert("RGBA"))
    visible[y:y+lh, x:x+lw] = np.array(Image.open(root/layer["visible_mask"]).convert("L"))
    fill[y:y+lh, x:x+lw] = np.array(Image.open(root/layer["fill_mask"]).convert("L"))
    for stroke in strokes:
        points = stroke.get("points", [])
        radius = int(np.clip(stroke.get("radius", 8), 1, 150))
        if len(points) > 20000: raise ValueError("Too many brush points")
        brush = np.zeros((h, w), np.uint8); prev = None
        for point in points:
            if len(point) != 2 or not all(isinstance(v, (int, float)) and math.isfinite(v) for v in point):
                raise ValueError("Invalid brush point")
            pt = (int(np.clip(point[0],0,w-1)),int(np.clip(point[1],0,h-1)))
            cv2.circle(brush,pt,radius,255,-1)
            if prev: cv2.line(brush,prev,pt,255,2*radius)
            prev = pt
        touched = brush > 0
        if stroke.get("erase"):
            art[touched] = 0; visible[touched] = 0; fill[touched] = 0
        else:
            touched &= rgba[...,3] > 0
            art[touched] = rgba[touched]; visible[touched] = rgba[...,3][touched]; fill[touched] = 0
    box = Image.fromarray(art[...,3]).getbbox()
    if box is None: raise ValueError("Mask correction would remove the entire part")
    x,y,x2,y2 = box; crop = art[y:y2,x:x2]
    Image.fromarray(crop).save(root/layer["image"])
    Image.fromarray(visible[y:y2,x:x2]).save(root/layer["visible_mask"])
    Image.fromarray(fill[y:y2,x:x2]).save(root/layer["fill_mask"])
    fill_count = int(np.count_nonzero(fill)); visible_count = int(np.count_nonzero(visible))
    layer.update(bbox=[x,y,x2-x,y2-y],visible_pixels=visible_count,fill_pixels=fill_count,
                 fill_fraction=round(fill_count/max(fill_count+visible_count,1),4),provenance="manual-mask+"+layer.get("provenance", "unknown").replace("manual-mask+", ""))
    layer["mesh"] = make_mesh(crop[...,3],layer["bbox"],layer["bone"],rig)
    return layer


def semantic_candidates(name: str, rig: dict) -> set[str]:
    """Weak semantic hints, resolved at word boundaries ("wear" is not "ear").

    A model's clothing class is not a limb: topwear still needs arm/forearm
    subdivision and legwear needs thigh/shin/foot subdivision. Unknown labels
    remain geometry-driven rather than silently deleting an anatomical part.
    """
    import re
    text = re.sub(r"[^a-z0-9]+", " ", name.lower()).strip()
    words = set(text.split())
    all_ids = {b["id"] for b in rig["bones"] if b.get("layer")}
    exact = {i for i in all_ids if re.sub(r"[^a-z0-9]+", " ", i.lower()).strip() == text}
    if exact: return exact
    def prefix(*names): return {i for i in all_ids if i.lower().startswith(names)}
    group = set()
    if words & {"neck", "neckwear"}:
        group = {"neck"} & all_ids
    elif words & {"hair", "headwear", "face", "irides", "irises", "eyebrow", "eyebrows", "eyewhite", "eyelash", "eyelashes", "eyewear", "ears", "earwear", "ear", "nose", "mouth", "eyes", "eye", "head", "brow", "beard"}:
        group = {"head"} & all_ids
    elif words & {"hand", "hands", "handwear", "glove", "gloves", "palm"}:
        group = prefix("hand")
    elif words & {"forearm", "forearms"} or "lower arm" in text:
        group = prefix("forearm")
    elif "upper arm" in text or "upperarm" in words:
        group = prefix("upper_arm")
    elif words & {"arm", "arms", "sleeve", "sleeves"}:
        group = prefix("upper_arm", "forearm", "hand")
    elif words & {"topwear", "shirt", "jacket", "coat"} or any(t in text for t in ("upper clothing", "upper garment")):
        group = ({"torso"} & all_ids) | prefix("upper_arm", "forearm")
    elif words & {"paw", "paws"}:
        group = prefix("foot", "hand") if rig.get("preset") == "quadruped" else prefix("foot")
    elif words & {"boot", "boots"}:
        group = prefix("foot", "shin")
    elif words & {"foot", "feet", "footwear", "shoe", "shoes"}:
        group = prefix("foot")
    elif words & {"thigh", "thighs"} or "upper leg" in text:
        group = prefix("thigh")
    elif words & {"shin", "shins", "calf", "calves"} or "lower leg" in text:
        group = prefix("shin")
    elif words & {"leg", "legs", "legwear", "bottomwear", "trouser", "trousers", "pants"} or any(t in text for t in ("lower clothing", "lower garment")):
        group = prefix("thigh", "shin", "foot")
    elif words & {"tail", "tails"}: group = prefix("tail")
    elif words & {"wing", "wings"}: group = prefix("wing")
    if group:
        if words & {"left", "l"}: group = {i for i in group if not i.endswith("_r")}
        if words & {"right", "r"}: group = {i for i in group if not i.endswith("_l")}
    return group or all_ids


def split_semantic_layers(inputs, size, rig, output: Path, *, provider="imported", progress=None, cancel=None):
    """Refine an amodal PSD into physical subsegments without discarding fills.

    Unlike taking model masks as final parts, each semantic RGBA layer is split
    AGAIN by the editable skeleton. Colors and hidden art from the provider are
    retained. Accessories and face layers remain separate Spine slots.
    """
    import copy
    import re
    output.mkdir(parents=True, exist_ok=True)
    layers, warnings = [], []
    width, height = size
    for ordinal, (name, art, offset) in enumerate(inputs):
        _check(cancel)
        full = Image.new("RGBA", size)
        full.paste(art.convert("RGBA"), tuple(map(int, offset)))
        if full.getchannel("A").getbbox() is None: continue
        candidates = semantic_candidates(name, rig)
        restricted = copy.deepcopy(rig)
        for b in restricted["bones"]: b["layer"] = b["id"] in candidates
        labels, bones, _, _ = partition(full, restricted, work_size=384, edge_weight=.6, cancel=cancel)
        rgba = np.array(full)
        for k, bone in enumerate(bones):
            mask = np.where(labels == k, rgba[..., 3], 0).astype(np.uint8)
            area = np.count_nonzero(mask)
            if area == 0: continue
            box = Image.fromarray(mask).getbbox()
            x, y, x2, y2 = box
            crop = rgba[y:y2, x:x2].copy(); crop[..., 3] = mask[y:y2, x:x2]
            slug = re.sub(r"[^A-Za-z0-9_-]", "_", name)[:24]
            lid = f"p{ordinal}_{slug}_{bone['id']}"[:64]
            Image.fromarray(crop).save(output / f"{lid}.png")
            Image.fromarray(crop[..., 3]).save(output / f"{lid}.visible.png")
            Image.new("L", (x2-x, y2-y)).save(output / f"{lid}.fill.png")
            bbox = [x, y, x2-x, y2-y]
            layers.append({"id": lid, "name": f"{name} / {bone['id']}", "bone": bone["id"],
                "image": f"{lid}.png", "visible_mask": f"{lid}.visible.png", "fill_mask": f"{lid}.fill.png",
                "bbox": bbox, "order": len(layers), "opacity": 1, "visible": True,
                "mesh": make_mesh(crop[..., 3], bbox, bone["id"], rig),
                "visible_pixels": int(area), "fill_pixels": 0, "fill_fraction": 0,
                "needs_review": provider != "imported", "provenance": f"{provider}+rig-split",
                "completion": "provider-amodal" if provider != "imported" else "imported-art"})
        if progress: progress(.30 + .60 * (ordinal+1)/max(len(inputs), 1), f"Splitting semantic layer {ordinal+1}/{len(inputs)}: {name}")
    # Visibility is defined by inferred draw order; retain amodal colors and
    # mark only regions occluded by an opaque front layer as hidden.
    covered = np.zeros((height, width), bool)
    for layer in reversed(layers):
        x, y, w, h = layer["bbox"]
        a = np.array(Image.open(output / layer["image"]))[..., 3]
        hidden = covered[y:y+h, x:x+w] & (a > 0)
        visible = a.copy(); visible[hidden] = 0
        fill = np.zeros_like(a); fill[hidden] = a[hidden]
        Image.fromarray(visible).save(output / layer["visible_mask"])
        Image.fromarray(fill).save(output / layer["fill_mask"])
        covered[y:y+h, x:x+w] |= a == 255
        layer.update(visible_pixels=int(np.count_nonzero(visible)), fill_pixels=int(np.count_nonzero(fill)),
                     fill_fraction=round(float(np.count_nonzero(fill)/max(np.count_nonzero(a), 1)), 4))
    represented = {l["bone"] for l in layers}
    expected = {b["id"] for b in rig["bones"] if b.get("layer")}
    for missing in sorted(expected - represented): warnings.append(f"No source/model layer covers {missing}; inspect the rig and masks.")
    if provider != "imported": warnings.append("Learned layers can alter visible art. Compare against the original; inferred hidden anatomy is not ground truth.")
    quality = {"method": f"{provider}+rig-split", "expected_parts": len(expected),
               "produced_parts": len(represented), "layer_count": len(layers), "visible_coverage": None,
               "automatic_semantic_accuracy": None}
    return layers, quality, warnings
