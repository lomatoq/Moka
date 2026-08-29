"""Bounded portable-data validation. Imported art is data, never executable code."""
from __future__ import annotations
import math
from .rig import SAFE_ID, validate_rig

MAX_LAYER_PIXELS = 64_000_000

def number(v, limit=10_000_000):
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) and abs(v) <= limit

def validate_motion(m):
    if m is None: return
    if not isinstance(m, dict) or m.get('schema') != 'moka.motion/1': raise ValueError('Unsupported motion schema')
    joints, frames = m.get('joints'), m.get('frames')
    if not isinstance(joints, list) or not 1 <= len(joints) <= 256: raise ValueError('Motion needs 1–256 joints')
    names = set()
    for j in joints:
        name = j if isinstance(j, str) else j.get('name') if isinstance(j, dict) else None
        if not isinstance(name, str) or not 1 <= len(name) <= 128 or name in names: raise ValueError('Invalid or duplicate motion joint')
        names.add(name)
    if not isinstance(frames, list) or not 1 <= len(frames) <= 18001: raise ValueError('Motion needs 1–18001 frames')
    if m.get('fps') is not None and (not number(m['fps'], 240) or m['fps'] <= 0): raise ValueError('Invalid motion frame rate')
    previous = -1
    for frame in frames:
        if not isinstance(frame, dict): raise ValueError('Invalid motion frame')
        t, points = frame.get('time'), frame.get('joints')
        if not number(t, 600) or t <= previous: raise ValueError('Motion times must increase and stay within 10 minutes')
        previous = t
        if not isinstance(points, dict) or len(points) > 256: raise ValueError('Invalid motion landmarks')
        for name, point in points.items():
            if name not in names or not isinstance(point, list) or len(point) not in (3, 4) or not all(number(v) for v in point):
                raise ValueError('Invalid motion landmark')
            if len(point) == 4 and not 0 <= point[3] <= 1: raise ValueError('Invalid landmark confidence')
    if 'screen_frames' in m:
        screens = m['screen_frames']
        if not isinstance(screens, list) or len(screens) != len(frames): raise ValueError('Invalid observed motion frames')
        for f in screens:
            points = f.get('joints') if isinstance(f, dict) else None
            if not isinstance(points, dict) or len(points) > 256: raise ValueError('Invalid observed landmarks')
            for name, point in points.items():
                if name not in names or not isinstance(point, list) or not 2 <= len(point) <= 4 or not all(number(v) for v in point):
                    raise ValueError('Invalid observed landmark')


def validate_project(p):
    if not isinstance(p, dict) or p.get('schema') != 'moka.project/1': raise ValueError('Unsupported project schema')
    width, height = p.get('width'), p.get('height')
    if any(not isinstance(v, int) or not 1 <= v <= 4096 for v in (width, height)): raise ValueError('Invalid project dimensions')
    if not isinstance(p.get('name'), str) or len(p['name']) > 100: raise ValueError('Invalid project name')
    validate_rig(p.get('rig'), width, height)
    bones = {b['id'] for b in p['rig']['bones']}
    layers = p.get('layers', [])
    if not isinstance(layers, list) or len(layers) > 256: raise ValueError('At most 256 layers are supported')
    seen, pixels, vertices = set(), 0, 0
    for layer in layers:
        if not isinstance(layer, dict): raise ValueError('Invalid layer')
        lid = layer.get('id')
        if not isinstance(lid, str) or not SAFE_ID.fullmatch(lid) or lid in seen: raise ValueError('Invalid or duplicate layer ID')
        seen.add(lid)
        if layer.get('bone') not in bones: raise ValueError('Unknown attachment bone')
        if not isinstance(layer.get('name'), str) or len(layer['name']) > 256: raise ValueError('Invalid layer name')
        for name in ('image', 'visible_mask', 'fill_mask'):
            if not isinstance(layer.get(name), str) or not layer[name].lower().endswith('.png'): raise ValueError('Layer assets must be PNG files')
        box = layer.get('bbox')
        if not isinstance(box, list) or len(box) != 4 or not all(isinstance(v, int) for v in box): raise ValueError('Invalid layer bounds')
        x, y, w, h = box
        if x < 0 or y < 0 or w < 1 or h < 1 or x+w > width or y+h > height: raise ValueError('Layer extends outside its canvas')
        pixels += w*h
        if pixels > MAX_LAYER_PIXELS: raise ValueError('Decoded layers exceed 64 megapixels')
        if not number(layer.get('opacity', 1), 1) or layer.get('opacity', 1) < 0: raise ValueError('Invalid layer opacity')
        if not number(layer.get('order', 0), 1024) or layer.get('order', 0) < 0: raise ValueError('Invalid layer order')
        mesh = layer.get('mesh')
        if not isinstance(mesh, dict): raise ValueError('Missing attachment mesh')
        v, uv, weights, triangles = (mesh.get(k) for k in ('vertices', 'uvs', 'weights', 'triangles'))
        if not all(isinstance(a, list) for a in (v, uv, weights, triangles)): raise ValueError('Invalid mesh arrays')
        if not 3 <= len(v) <= 65536 or len(uv) != len(v) or len(weights) != len(v): raise ValueError('Invalid mesh vertex count')
        vertices += len(v)
        if vertices > 500000: raise ValueError('Project exceeds the mesh vertex budget')
        if len(triangles) > 393216 or len(triangles) % 3 or not all(isinstance(i, int) and 0 <= i < len(v) for i in triangles):
            raise ValueError('Invalid triangle indices')
        for point, coord, influences in zip(v, uv, weights):
            if not isinstance(point, list) or len(point) != 2 or not all(number(a, 16384) for a in point): raise ValueError('Invalid mesh position')
            if not isinstance(coord, list) or len(coord) != 2 or not all(number(a, 1) and a >= 0 for a in coord): raise ValueError('Invalid UV coordinates')
            if not isinstance(influences, list) or not 1 <= len(influences) <= 4: raise ValueError('Invalid skinning influences')
            total = 0
            for influence in influences:
                if not isinstance(influence, dict) or influence.get('bone') not in bones or not number(influence.get('weight'), 1) or influence['weight'] < 0:
                    raise ValueError('Invalid skinning weight')
                total += influence['weight']
            if abs(total-1) > .002: raise ValueError('Skinning weights must sum to one')
    validate_motion(p.get('source_motion'))
