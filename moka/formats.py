"""Portable projects, layered PSD, and actual Spine 4.2 weighted-mesh export.
No Spine runtime code is bundled. The format boundary converts Y-down pixels
and clockwise degrees to Spine's Y-up, counter-clockwise setup transforms.
"""
from __future__ import annotations
import io
import json
import math
import struct
import zipfile
import zlib
from pathlib import Path, PurePosixPath
import numpy as np
from PIL import Image
from .rig import SAFE_ID, local_point, setup_transforms
from .validation import MAX_LAYER_PIXELS


def safe_asset(root: Path, relative: str) -> Path:
    if not isinstance(relative, str) or not relative or "\\" in relative:
        raise ValueError("Invalid asset path")
    p = PurePosixPath(relative)
    if p.is_absolute() or ".." in p.parts or ":" in relative:
        raise ValueError("Asset path escapes the project")
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("Asset path escapes the project")
    return path


def _pack(fmt, *args):
    return struct.pack(">" + fmt, *args)


def write_psd(layers: list[tuple[str, Image.Image, tuple[int, int]]], size: tuple[int, int]) -> bytes:
    """8-bit RGB PSD with real editable RGBA layers and a merged preview."""
    width, height = size
    records, pixels = [], []
    composite = Image.new("RGBA", size)
    for name, image, (x, y) in layers:
        temp = Image.new("RGBA", size)
        temp.paste(image.convert("RGBA"), (x, y))
        composite = Image.alpha_composite(composite, temp)
    # PSD records are top-first; our scene order is bottom-first.
    for name, image, (x, y) in reversed(layers):
        a = np.array(image.convert("RGBA"))
        h, w = a.shape[:2]
        channels = [(0, a[..., 0]), (1, a[..., 1]), (2, a[..., 2]), (-1, a[..., 3])]
        payloads = [b"\0\0" + c.tobytes() for _, c in channels]
        record = _pack("iiiiH", y, x, y+h, x+w, 4)
        for (channel_id, _), data in zip(channels, payloads):
            record += _pack("hI", channel_id, len(data))
        record += b"8BIMnorm" + bytes([255, 0, 0, 0])
        short = name.encode("ascii", "replace")[:255]
        pascal = bytes([len(short)]) + short
        pascal += b"\0" * ((-len(pascal)) % 4)
        uni = name.encode("utf-16be")
        unicode_name = _pack("I", len(uni)//2) + uni
        additional = b"8BIMluni" + _pack("I", len(unicode_name)) + unicode_name
        if len(unicode_name) % 2: additional += b"\0"
        extra = _pack("II", 0, 0) + pascal + additional
        records.append(record + _pack("I", len(extra)) + extra)
        pixels.extend(payloads)
    layer_info = _pack("h", -len(layers)) + b"".join(records) + b"".join(pixels)
    if len(layer_info) % 2: layer_info += b"\0"
    layer_mask = _pack("I", len(layer_info)) + layer_info + _pack("I", 0)
    merged = np.array(composite)
    planar = b"".join(merged[..., c].tobytes() for c in range(4))
    return (b"8BPS" + _pack("H", 1) + b"\0"*6 + _pack("HIIHH", 4, height, width, 8, 3)
            + _pack("II", 0, 0) + _pack("I", len(layer_mask)) + layer_mask + b"\0\0" + planar)


class _Reader:
    def __init__(self, data): self.s = io.BytesIO(data)
    def read(self, n):
        if n < 0 or n > 256_000_000: raise ValueError("Invalid PSD block length")
        data = self.s.read(n)
        if len(data) != n: raise ValueError("Truncated PSD")
        return data
    def get(self, fmt):
        n = struct.calcsize(">" + fmt)
        values = struct.unpack(">" + fmt, self.read(n))
        return values[0] if len(values) == 1 else values


def _unpackbits(data: bytes, expected: int) -> bytes:
    out, pos = bytearray(), 0
    while pos < len(data) and len(out) < expected:
        n = data[pos]; pos += 1
        if n <= 127:
            count = n + 1
            out.extend(data[pos:pos+count]); pos += count
        elif n >= 129:
            if pos >= len(data): raise ValueError("Invalid PSD RLE run")
            out.extend(bytes([data[pos]]) * (257-n)); pos += 1
    if len(out) != expected: raise ValueError("PSD channel has an invalid decoded length")
    return bytes(out)


def read_psd_basic(data: bytes):
    """Fallback for 8-bit RGB, normal RGBA layers (raw/RLE/ZIP).

    Complex Photoshop masks/blending are explicitly reported, not silently
    promised to match. psd-tools is preferred by the application when installed.
    """
    r = _Reader(data)
    if r.read(4) != b"8BPS" or r.get("H") != 1: raise ValueError("Only standard PSD version 1 is supported")
    r.read(6)
    channels, h, w, depth, mode = r.get("HIIHH")
    if depth != 8 or mode != 3 or not 1 <= w*h <= 16_777_216 or max(w,h) > 4096: raise ValueError("PSD must be 8-bit RGB and no larger than 16 megapixels")
    r.read(r.get("I")); r.read(r.get("I"))
    section = r.get("I")
    if not section: raise ValueError("PSD has no editable layers")
    info_len = r.get("I")
    if not info_len: raise ValueError("PSD has no editable layers")
    count = abs(r.get("h"))
    if count > 256: raise ValueError("PSD exceeds 256 layers")
    records, warnings = [], []
    for i in range(count):
        top, left, bottom, right, n = r.get("iiiiH")
        if n > 16: raise ValueError("PSD has too many channels")
        ch = [r.get("hI") for _ in range(n)]
        sig, blend = r.read(4), r.read(4)
        opacity, clipping, flags, _ = r.get("BBBB")
        extra_len = r.get("I")
        extra = _Reader(r.read(extra_len))
        mask_len = extra.get("I"); extra.read(mask_len)
        extra.read(extra.get("I"))
        name_len = extra.get("B")
        name = extra.read(name_len).decode("latin1", "replace")
        extra.read((-(name_len+1)) % 4)
        while extra.s.tell() + 12 <= extra_len:
            signature, key, length = extra.read(4), extra.read(4), extra.get("I")
            block = extra.read(length)
            if length % 2 and extra.s.tell() < extra_len: extra.read(1)
            if key == b"luni" and len(block) >= 4:
                nchars = struct.unpack(">I", block[:4])[0]
                name = block[4:4+nchars*2].decode("utf-16be", "replace")
        if blend not in (b"norm", b"pass") or mask_len or clipping:
            warnings.append(f"{name}: complex masks/blending need psd-tools or a flattened RGBA layer")
        records.append((name or f"layer_{i}", left, top, right-left, bottom-top, ch, opacity, flags))
    if sum(max(0,r[3])*max(0,r[4]) for r in records) > MAX_LAYER_PIXELS: raise ValueError("Decoded PSD layers exceed 64 megapixels")
    layers = []
    for name, x, y, lw, lh, ch, opacity, flags in records:
        if lw < 0 or lh < 0 or lw*lh > 16_777_216: raise ValueError("Invalid PSD layer bounds")
        arr = np.zeros((lh, lw, 4), np.uint8); arr[..., 3] = opacity
        for channel_id, length in ch:
            block = _Reader(r.read(length))
            compression = block.get("H")
            payload = block.s.read()
            expected = lw*lh
            if not expected: continue
            if compression == 0: raw = payload
            elif compression == 1:
                if len(payload) < lh*2: raise ValueError("Invalid RLE row table")
                lengths = struct.unpack(">" + "H"*lh, payload[:lh*2])
                pos, rows = lh*2, []
                for rowlen in lengths:
                    rows.append(_unpackbits(payload[pos:pos+rowlen], lw)); pos += rowlen
                raw = b"".join(rows)
            elif compression in (2, 3):
                dec = zlib.decompressobj()
                raw = dec.decompress(payload, expected+1)
                if compression == 3 and len(raw) == expected:
                    pred = np.frombuffer(raw, np.uint8).reshape(lh, lw).astype(np.uint16)
                    raw = (np.cumsum(pred, axis=1) % 256).astype(np.uint8).tobytes()
            else: raise ValueError("Unsupported PSD channel compression")
            if len(raw) != expected: raise ValueError("Invalid PSD channel length")
            target = 3 if channel_id == -1 else channel_id
            if target in (0, 1, 2, 3):
                values = np.frombuffer(raw, np.uint8).reshape(lh, lw)
                arr[..., target] = (values.astype(np.uint16)*opacity//255).astype(np.uint8) if target == 3 else values
        if lw and lh and np.any(arr[..., 3]) and not flags & 2:
            layers.append((name, Image.fromarray(arr), (x, y)))
    return list(reversed(layers)), (w, h), warnings


def read_psd(data: bytes):
    try:
        from psd_tools import PSDImage
    except ImportError:
        return read_psd_basic(data)
    psd = PSDImage.open(io.BytesIO(data))
    if psd.width * psd.height > 16_777_216 or max(psd.size) > 4096: raise ValueError("PSD exceeds 16 megapixels")
    layers, warnings = [], []
    decoded = 0
    def walk(group, depth=0):
        nonlocal decoded
        if depth > 16: raise ValueError("PSD groups are too deeply nested")
        for item in group:
            if not item.is_visible(): continue
            if item.is_group():
                walk(item,depth+1)
            else:
                decoded += max(0,item.width)*max(0,item.height)
                if decoded > MAX_LAYER_PIXELS or len(layers) >= 256: raise ValueError("Decoded PSD exceeds the layer budget")
                img = item.composite()
                if img is not None:
                    layers.append((item.name, img.convert("RGBA"), (item.left, item.top)))
    walk(psd)
    if len(layers) > 256: raise ValueError("PSD exceeds 256 layers")
    if not layers: raise ValueError("PSD has no visible editable layers")
    # psd-tools iteration follows bottom-to-top PSD layer order.
    return layers, (psd.width, psd.height), warnings


def export_psd(project: dict, root: Path) -> bytes:
    layers = []
    for layer in sorted(project["layers"], key=lambda l:l["order"]):
        if not layer.get("visible",True): continue
        image = Image.open(safe_asset(root,layer["image"])).convert("RGBA")
        opacity = float(layer.get("opacity",1))
        if opacity < 1:
            alpha = np.array(image.getchannel("A"),dtype=np.float64)
            image.putalpha(Image.fromarray(np.round(alpha*opacity).astype(np.uint8)))
        layers.append((layer["name"],image,tuple(layer["bbox"][:2])))
    if not layers: raise ValueError("No visible layers to export")
    return write_psd(layers,(project["width"],project["height"]))


def pack_atlas(project: dict, root: Path):
    items = []
    for layer in project["layers"]:
        im = Image.open(safe_asset(root, layer["image"])).convert("RGBA")
        items.append((layer["id"], im))
    if not items: raise ValueError("Cut or import layers before exporting")
    pad = 2
    max_part = max(max(im.size)+2*pad for _, im in items)
    limit = max(2048, max_part)
    if limit > 8192: raise ValueError("An atlas region exceeds 8192 pixels")
    area = sum((im.width+4)*(im.height+4) for _, im in items)
    page_width = min(limit, max(max_part, math.ceil(math.sqrt(area)), 256))
    pages, placements = [], []
    current, x, y, row, used_w = [], 0, 0, 0, 0
    for name, image in sorted(items, key=lambda v: v[1].height, reverse=True):
        w, h = image.size
        if x+w+2*pad > page_width:
            y += row; x = 0; row = 0
        if y+h+2*pad > limit and current:
            pages.append((current, used_w, y+row)); current=[]; x=y=row=used_w=0
        current.append((name, image, x+pad, y+pad))
        x += w+2*pad; row = max(row, h+2*pad); used_w=max(used_w, x)
    if current: pages.append((current, used_w, y+row))
    atlas, files = [], {}
    for page_index, (entries, pw, ph) in enumerate(pages):
        filename = "atlas.png" if page_index == 0 else f"atlas-{page_index}.png"
        canvas = Image.new("RGBA", (pw, ph))
        atlas += [filename, f"size: {pw},{ph}", "format: RGBA8888", "filter: Linear,Linear", "repeat: none", "pma: false"]
        for name, im, x, y in entries:
            canvas.paste(im, (x, y))
            # Extrude edge texels into padding for bilinear sampling.
            canvas.paste(im.crop((0, 0, 1, im.height)).resize((pad, im.height)), (x-pad, y))
            canvas.paste(im.crop((im.width-1, 0, im.width, im.height)).resize((pad, im.height)), (x+im.width, y))
            canvas.paste(im.crop((0, 0, im.width, 1)).resize((im.width, pad)), (x, y-pad))
            canvas.paste(im.crop((0, im.height-1, im.width, im.height)).resize((im.width, pad)), (x, y+im.height))
            atlas += [name, f"  bounds: {x},{y},{im.width},{im.height}", f"  offsets: 0,0,{im.width},{im.height}", "  rotate: false", "  index: -1"]
        atlas.append("")
        b = io.BytesIO(); canvas.save(b, format="PNG"); files[filename] = b.getvalue()
    return "\n".join(atlas), files


def spine_document(project: dict, weighted=True) -> dict:
    rig = project["rig"]
    bind = setup_transforms(rig)
    bones, slots, attachments = [], [], {}
    bone_index = {b["id"]: i for i, b in enumerate(rig["bones"])}
    root_bind = bind["root"]
    for b in rig["bones"]:
        own = bind[b["id"]]
        item = {"name": b["id"], "length": round(own["length"], 5)}
        if b.get("parent"):
            p = bind[b["parent"]]
            offset = local_point([own["x"], own["y"]], p)
            item.update(parent=b["parent"], x=round(offset[0], 5), y=round(-offset[1], 5),
                        rotation=round(-math.degrees(own["angle"]-p["angle"]), 5))
        bones.append(item)
    for layer in sorted(project["layers"], key=lambda l: l["order"]):
        name, bone = layer["id"], layer["bone"]
        if not SAFE_ID.fullmatch(name) or bone not in bind: raise ValueError("Invalid layer/bone reference")
        slot = {"name": name, "bone": bone, "attachment": name}
        alpha = round(255 * float(layer.get("opacity", 1))) if layer.get("visible", True) else 0
        if alpha != 255: slot["color"] = f"ffffff{alpha:02x}"
        slots.append(slot)
        x, y, w, h = layer["bbox"]
        mesh = layer.get("mesh", {})
        if weighted and mesh.get("vertices") and mesh.get("triangles"):
            # Put the rectangular UV hull first, as expected by the editor.
            uvs = mesh["uvs"]
            corners = []
            for target in ((0, 0), (1, 0), (1, 1), (0, 1)):
                idx = min(range(len(uvs)), key=lambda i: (uvs[i][0]-target[0])**2+(uvs[i][1]-target[1])**2)
                if idx not in corners: corners.append(idx)
            ordering = corners + [i for i in range(len(uvs)) if i not in corners]
            remap = {old: new for new, old in enumerate(ordering)}
            vertices = []
            for i in ordering:
                point, influences = mesh["vertices"][i], mesh["weights"][i]
                vertices.append(len(influences))
                for influence in influences:
                    bid, weight = influence["bone"], influence["weight"]
                    p = local_point(point, bind[bid])
                    vertices.extend([bone_index[bid], round(p[0], 6), round(-p[1], 6), round(weight, 7)])
            attachment = {"type": "mesh", "path": name, "uvs": [v for i in ordering for v in uvs[i]],
                          "triangles": [remap[i] for i in mesh["triangles"]], "vertices": vertices,
                          "hull": len(corners), "width": w, "height": h}
        else:
            p = local_point([x+w/2, y+h/2], bind[bone])
            attachment = {"type": "region", "path": name, "x": p[0], "y": -p[1], "width": w, "height": h,
                          "rotation": math.degrees(bind[bone]["angle"])}
        attachments[name] = {name: attachment}
    animations = {}
    for clip in project.get("clips", []):
        timelines = {}
        frames = clip.get("frames", [])
        if not frames: continue
        for bone in bone_index:
            values = [{"time": round(f["time"], 6), "value": round(-float(f.get("angles", {}).get(bone, 0)), 5)} for f in frames]
            if any(abs(v["value"]) > 1e-6 for v in values):
                timelines.setdefault(bone, {})["rotate"] = values
        roots = [{"time": round(f["time"], 6), "x": round(f.get("root", [0, 0])[0], 5), "y": round(-f.get("root", [0, 0])[1], 5)} for f in frames]
        # A root translation timeline also preserves duration for static clips.
        timelines.setdefault("root", {})["translate"] = roots
        name = clip.get("name", "animation")
        suffix = 2; candidate = name
        while candidate in animations: candidate = f"{name}_{suffix}"; suffix += 1
        animations[candidate] = {"bones": timelines}
    return {"skeleton": {"spine": "4.2.00", "x": -root_bind["x"], "y": root_bind["y"]-project["height"],
                         "width": project["width"], "height": project["height"], "images": "./images/", "fps": 30},
            "bones": bones, "slots": slots, "skins": [{"name": "default", "attachments": attachments}],
            "animations": animations}


def export_spine(project: dict, root: Path, weighted=True) -> bytes:
    document = spine_document(project, weighted)
    atlas, pages = pack_atlas(project, root)
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("skeleton.json", json.dumps(document, ensure_ascii=False, separators=(",", ":"), allow_nan=False))
        z.writestr("skeleton.atlas", atlas)
        for name, data in pages.items(): z.writestr(name, data)
        for layer in project["layers"]:
            z.write(safe_asset(root, layer["image"]), f"images/{layer['id']}.png")
        z.writestr("IMPORT.txt", "Spine 4.2 JSON, atlas and editable source images.\nImport skeleton.json using Data Import in a compatible Spine editor.\nUse the matching 4.2 runtime or re-export from your own editor version.\nNo Spine runtime license is included.\n")
    return output.getvalue()


def project_bundle(project: dict, root: Path) -> bytes:
    output = io.BytesIO()
    files = {project["source"]}
    if project.get("semantic_source"): files.add(project["semantic_source"])
    for layer in project.get("layers", []):
        for key in ("image", "visible_mask", "fill_mask"):
            if layer.get(key): files.add(layer[key])
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("project.json", json.dumps(project, ensure_ascii=False, allow_nan=False))
        for name in sorted(files): z.write(safe_asset(root, name), name)
    return output.getvalue()


def checked_zip(data: bytes) -> zipfile.ZipFile:
    z = zipfile.ZipFile(io.BytesIO(data))
    if len(z.infolist()) > 1024 or sum(i.file_size for i in z.infolist()) > 256_000_000:
        z.close(); raise ValueError("Archive exceeds the unpacked size/file limit")
    for entry in z.infolist():
        p = PurePosixPath(entry.filename)
        if p.is_absolute() or ".." in p.parts or "\\" in entry.filename or ":" in entry.filename:
            z.close(); raise ValueError("Unsafe archive path")
        if (entry.external_attr >> 16) & 0o170000 == 0o120000:
            z.close(); raise ValueError("Archive symlinks are not allowed")
    return z
