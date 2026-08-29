"""Local workbench API. Jobs commit atomically, inputs are bounded, assets stay local."""
from __future__ import annotations
import copy
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import io
import json
import math
import os
from pathlib import Path
import re
import shutil
import threading
import time
import uuid
from urllib.parse import urlparse
import zipfile

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from starlette.middleware.trustedhost import TrustedHostMiddleware
from . import __version__
from . import engines
from .demo import make_demo
from .formats import checked_zip, export_psd, export_spine, project_bundle, read_psd, safe_asset
from .rig import SAFE_ID, make_mesh, setup_transforms, template_rig, validate_rig
from .validation import MAX_LAYER_PIXELS, validate_motion, validate_project
from .vision import Cancelled, decompose, foreground, image_bbox, paint_layer, split_semantic_layers

ROOT = Path(__file__).resolve().parents[1]
MAX_UPLOAD = 128 * 1024 * 1024
MAX_PIXELS = 16_777_216
ID = re.compile(r"^[a-f0-9]{32}$")


def now(): return datetime.now(timezone.utc).isoformat()


def finite_tree(value, depth=0):
    if depth > 16: raise ValueError("JSON structure is too deeply nested")
    if isinstance(value, float) and not math.isfinite(value): raise ValueError("NaN and Infinity are not allowed")
    if isinstance(value, dict):
        for k, v in value.items():
            if not isinstance(k, str) or len(k) > 256: raise ValueError("Invalid JSON key")
            finite_tree(v, depth+1)
    elif isinstance(value, list):
        for v in value: finite_tree(v, depth+1)
    elif isinstance(value, str) and len(value) > 10000: raise ValueError("Text field is too large")


def validate_clips(clips, rig):
    if not isinstance(clips, list) or len(clips) > 64: raise ValueError("At most 64 clips are allowed")
    bone_ids = {b["id"] for b in rig["bones"]}
    total = 0
    for clip in clips:
        if not isinstance(clip, dict): raise ValueError("Invalid clip")
        if not isinstance(clip.get("name"), str) or not 1 <= len(clip["name"]) <= 100: raise ValueError("Invalid clip name")
        frames = clip.get("frames", [])
        if not isinstance(frames, list): raise ValueError("Clip frames must be an array")
        total += len(frames)
        if len(frames) > 18001 or total > 36000: raise ValueError("Motion exceeds the frame budget")
        prev = -1
        for frame in frames:
            if not isinstance(frame, dict) or not isinstance(frame.get("angles", {}), dict): raise ValueError("Invalid clip frame")
            t = frame.get("time")
            if not isinstance(t, (int, float)) or not math.isfinite(t) or t < 0 or t > 600 or t <= prev:
                raise ValueError("Clip times must be finite, increasing, and within 10 minutes")
            prev = t
            for k, angle in frame.get("angles", {}).items():
                if k not in bone_ids or not isinstance(angle, (int, float)) or not math.isfinite(angle) or abs(angle) > 100000:
                    raise ValueError("Invalid bone angle")
            root = frame.get("root", [0, 0])
            if not isinstance(root, list) or len(root) != 2 or not all(isinstance(v, (int, float)) and math.isfinite(v) and abs(v) < 1e7 for v in root):
                raise ValueError("Invalid root motion")


class Store:
    def __init__(self, root):
        self.root = Path(root).resolve(); self.root.mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="moka-job")
        self.jobs = {}

    def path(self, pid):
        if not ID.fullmatch(pid): raise HTTPException(404, "Project not found")
        return self.root/pid

    def load(self, pid):
        path = self.path(pid)/"project.json"
        if not path.is_file(): raise HTTPException(404, "Project not found")
        with self.lock: return json.loads(path.read_text("utf-8"))

    def write(self, project, expected=None):
        with self.lock:
            path = self.path(project["id"]); path.mkdir(parents=True, exist_ok=True)
            existing = path/"project.json"
            if expected is not None:
                actual = json.loads(existing.read_text("utf-8"))["revision"]
                if actual != expected: raise HTTPException(409, "The project changed in another tab; reload before applying edits")
            project["revision"] = (expected+1) if expected is not None else project.get("revision", 0)
            project["updated_at"] = now()
            temp = path/f".project-{uuid.uuid4().hex}.tmp"
            temp.write_text(json.dumps(project, ensure_ascii=False, allow_nan=False), "utf-8")
            temp.replace(existing)
        return project

    def busy(self, pid):
        return any(j["project_id"] == pid and j["status"] in ("queued", "running") for j in self.jobs.values())

    def new(self, image, name, rig=None, warnings=None, original=None):
        pid = uuid.uuid4().hex; path = self.path(pid); path.mkdir()
        image.save(path/"source.png")
        if original: original.save(path/"original.png")
        project = {"schema": "moka.project/1", "id": pid, "name": name[:100] or "Untitled character",
            "width": image.width, "height": image.height, "source": "source.png", "revision": 0,
            "rig": rig or template_rig(*image.size, bbox=image_bbox(image)), "layers": [], "clips": [],
            "warnings": warnings or [], "quality": {}, "created_at": now(), "updated_at": now()}
        return self.write(project)


def decode_image(data):
    try:
        with Image.open(io.BytesIO(data)) as im:
            if im.width*im.height > MAX_PIXELS or max(im.size) > 4096:
                raise ValueError("Source images must be at most 4096 px per side and 16 megapixels")
            if getattr(im, "n_frames", 1) > 1: im.seek(0)
            return ImageOps.exif_transpose(im).convert("RGBA")
    except (UnidentifiedImageError, OSError) as exc: raise ValueError("The file is not a supported image") from exc


async def upload_bytes(file: UploadFile):
    pieces, length = [], 0
    while True:
        data = await file.read(1024*1024)
        if not data: break
        length += len(data)
        if length > MAX_UPLOAD: raise HTTPException(413, "Upload exceeds 128 MB")
        pieces.append(data)
    if not length: raise HTTPException(400, "The uploaded file is empty")
    return b"".join(pieces)


def create_app(data_dir=None):
    store = Store(data_dir or os.environ.get("MOKA_DATA_DIR", ROOT/".moka/projects"))
    app = FastAPI(title="Moka", version=__version__)
    app.state.store = store
    allowed = os.environ.get("MOKA_ALLOWED_HOSTS", "localhost,127.0.0.1,[::1],testserver").split(",")
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=allowed)

    @app.middleware("http")
    async def local_guard(request, call_next):
        if request.method not in ("GET", "HEAD", "OPTIONS"):
            origin = request.headers.get("origin")
            if origin and urlparse(origin).netloc != request.headers.get("host"):
                return JSONResponse({"detail": "Cross-origin writes are disabled"}, status_code=403)
            length = request.headers.get("content-length")
            if length and (not length.isdigit() or int(length) > MAX_UPLOAD + 1024*1024):
                return JSONResponse({"detail": "Request exceeds 128 MB"}, status_code=413)
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["X-Frame-Options"] = "DENY"
        return response

    @app.exception_handler(ValueError)
    async def bad_input(request, exc): return JSONResponse({"detail": str(exc)}, status_code=400)

    @app.get("/api/health")
    def health(): return {"app": "moka", "version": __version__, "status": "ok"}

    @app.get("/api/capabilities")
    def capabilities():
        local_three = (ROOT/"vendor/three/build/three.module.js").is_file()
        local_pose = (ROOT/"vendor/mediapipe/vision_bundle.mjs").is_file()
        local_model = (ROOT/"models/pose_landmarker_full.task").is_file()
        return {"engines": engines.capabilities(), "local_only": True,
                "three_base": "/vendor/three" if local_three else "https://cdn.jsdelivr.net/npm/three@0.180.0",
                "pose_module": "/vendor/mediapipe/vision_bundle.mjs" if local_pose else "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.18/vision_bundle.mjs",
                "pose_wasm": "/vendor/mediapipe/wasm" if local_pose else "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.18/wasm",
                "pose_model": "/models/pose_landmarker_full.task" if local_model else "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task",
                "browser_assets_cached": local_pose and local_model and local_three}

    @app.get("/api/projects")
    def projects():
        items = []
        for path in store.root.glob("*/project.json"):
            try:
                p = json.loads(path.read_text("utf-8"))
                items.append({k: p.get(k) for k in ("id", "name", "updated_at", "width", "height")})
            except (OSError, ValueError): pass
        return sorted(items, key=lambda p: p["updated_at"] or "", reverse=True)

    @app.post("/api/demo")
    def demo():
        image, rig, _ = make_demo()
        return store.new(image, "Moka · demo character", rig, ["Synthetic fixture with known joints. It does not demonstrate AI pose-detection accuracy."])

    @app.post("/api/projects")
    async def create_project(file: UploadFile = File(...), preset: str = Form("biped")):
        data = await upload_bytes(file)
        filename = Path(file.filename or "character.png").name
        ext = Path(filename).suffix.lower()
        if ext in (".moka", ".zip"):
            with checked_zip(data) as z:
                if "project.json" in z.namelist():
                    p = json.loads(z.read("project.json")); finite_tree(p)
                    validate_project(p); validate_clips(p.get("clips", []), p["rig"])
                    p.update(id=uuid.uuid4().hex, revision=0)
                    root = store.path(p["id"]); root.mkdir()
                    if not isinstance(p.get("source"), str) or not p["source"].lower().endswith((".png", ".jpg", ".webp")):
                        shutil.rmtree(root, ignore_errors=True); raise ValueError("Invalid source image asset")
                    required = {p["source"]}
                    if p.get("semantic_source"): required.add(p["semantic_source"])
                    for l in p.get("layers", []):
                        if not SAFE_ID.fullmatch(l["id"]): raise ValueError("Invalid layer ID")
                        for key in ("image", "visible_mask", "fill_mask"):
                            if l.get(key): required.add(l[key])
                    try:
                        for name in required:
                            target = safe_asset(root, name)
                            if name not in z.namelist(): raise ValueError("The project is missing an asset")
                            contents = z.read(name)
                            if name.lower().endswith((".png", ".jpg", ".webp")): decode_image(contents)
                            target.parent.mkdir(parents=True, exist_ok=True); target.write_bytes(contents)
                        with Image.open(safe_asset(root,p["source"])) as source:
                            if source.size != (p["width"],p["height"]): raise ValueError("Source dimensions do not match the project")
                        for layer in p["layers"]:
                            for key in ("image", "visible_mask", "fill_mask"):
                                with Image.open(safe_asset(root,layer[key])) as art:
                                    if art.size != tuple(layer["bbox"][2:]): raise ValueError("Layer asset dimensions do not match its bounds")
                        p.setdefault("quality", {}); p.setdefault("warnings", []); p.setdefault("clips", [])
                        return store.write(p)
                    except Exception:
                        shutil.rmtree(root, ignore_errors=True); raise
                pngs = [n for n in z.namelist() if n.lower().endswith(".png") and not n.startswith("__MACOSX/")]
                if not pngs: raise ValueError("ZIP has neither a Moka project nor PNG layers")
                manifest = json.loads(z.read("layers.json")) if "layers.json" in z.namelist() else None
                if len(pngs) > 256: raise ValueError("At most 256 PNG layers are supported")
                inputs, decoded = [], 0
                for name in sorted(pngs):
                    art = decode_image(z.read(name))
                    decoded += art.width*art.height
                    if decoded > MAX_LAYER_PIXELS: raise ValueError("Decoded layers exceed 64 megapixels")
                    entry = next((e for e in (manifest or {}).get("layers", []) if e.get("path") == name), {})
                    inputs.append((entry.get("name", Path(name).stem), art, (entry.get("x", 0), entry.get("y", 0))))
                size = tuple((manifest or {}).get("canvas", inputs[0][1].size))
                if not manifest and any(im.size != size for _, im, _ in inputs):
                    raise ValueError("Cropped PNG layers require layers.json with canvas and x/y offsets")
                if len(size) != 2 or min(size) < 1 or max(size) > 4096: raise ValueError("Invalid layer canvas")
                image = Image.new("RGBA", size)
                for _, art, offset in inputs:
                    full = Image.new("RGBA", size); full.paste(art, tuple(offset)); image = Image.alpha_composite(image, full)
                from .formats import write_psd
                data = write_psd(inputs, size); ext = ".psd"
        if ext == ".psd":
            inputs, size, warnings = read_psd(data)
            image = Image.new("RGBA", size)
            for _, art, offset in inputs:
                full = Image.new("RGBA", size); full.paste(art, offset); image = Image.alpha_composite(image, full)
            p = store.new(image, Path(filename).stem, template_rig(*size, preset, image_bbox(image)), warnings)
            (store.path(p["id"])/"original.psd").write_bytes(data)
            p["semantic_source"] = "original.psd"; p["suggested_engine"] = "imported"
            return store.write(p, p["revision"])
        original = decode_image(data)
        image, method, warnings = foreground(original)
        rig = template_rig(*image.size, preset, image_bbox(image))
        p = store.new(image, Path(filename).stem, rig, warnings, original)
        p["foreground_method"] = method
        return store.write(p, p["revision"])

    @app.get("/api/projects/{pid}")
    def get_project(pid: str): return store.load(pid)

    @app.get("/api/projects/{pid}/assets/{asset:path}")
    def get_asset(pid: str, asset: str):
        store.load(pid)
        path = safe_asset(store.path(pid), asset)
        if path.suffix.lower() not in (".png", ".jpg", ".webp") or not path.is_file(): raise HTTPException(404, "Image asset not found")
        return FileResponse(path, headers={"Cache-Control": "no-cache"})

    @app.put("/api/projects/{pid}")
    def update_project(pid: str, patch: dict):
        finite_tree(patch)
        with store.lock:
            if store.busy(pid): raise HTTPException(409, "A job is modifying this project; finish or cancel it first")
            p = store.load(pid); expected = patch.get("revision")
            if expected != p["revision"]: raise HTTPException(409, "Stale project revision; reload the project")
            if "name" in patch: p["name"] = str(patch["name"])[:100]
            if "rig" in patch:
                validate_rig(patch["rig"], p["width"], p["height"])
                p["rig"] = patch["rig"]
            bones = {b["id"] for b in p["rig"]["bones"]}
            edits = {l["id"]: l for l in patch.get("layer_edits", [])}
            for layer in p["layers"]:
                edit = edits.get(layer["id"], {})
                for key in ("name", "bone", "order", "visible", "opacity"):
                    if key in edit: layer[key] = edit[key]
                if layer["bone"] not in bones: raise ValueError("Layer references an unknown bone")
                if not isinstance(layer["order"], (int, float)) or not 0 <= layer["order"] <= 1024: raise ValueError("Invalid layer order")
                if not isinstance(layer["opacity"], (int, float)) or not 0 <= layer["opacity"] <= 1: raise ValueError("Invalid layer opacity")
                if edit.get("bone") or patch.get("rebuild_mesh"):
                    image = Image.open(safe_asset(store.path(pid), layer["image"])).convert("RGBA")
                    layer["mesh"] = make_mesh(np.array(image)[..., 3], layer["bbox"], layer["bone"], p["rig"], bool(patch.get("rigid", False)))
            if "clips" in patch:
                validate_clips(patch["clips"], p["rig"]); p["clips"] = patch["clips"]
            if "source_motion" in patch:
                m = patch["source_motion"]
                validate_motion(m)
                p["source_motion"] = m
            return store.write(p, expected)

    @app.post("/api/projects/{pid}/preset")
    def set_preset(pid: str, request: dict):
        with store.lock:
            if store.busy(pid): raise HTTPException(409, "A project job is already running")
            p = store.load(pid)
            if request.get("revision") != p["revision"]: raise HTTPException(409, "Stale project revision")
            image = Image.open(safe_asset(store.path(pid), p["source"])).convert("RGBA")
            rig = template_rig(p["width"], p["height"], request.get("preset", "biped"), image_bbox(image))
            new_ids = {b["id"] for b in rig["bones"]}
            # Do not silently orphan custom artwork or animation tracks.
            if any(l["bone"] not in new_ids for l in p["layers"]):
                raise ValueError("This preset would orphan custom parts. Import the source as a new project instead.")
            for clip in p["clips"]:
                for frame in clip["frames"]:
                    frame["angles"] = {k: v for k, v in frame["angles"].items() if k in new_ids}
            p["rig"] = rig
            for layer in p["layers"]:
                art = Image.open(safe_asset(store.path(pid), layer["image"])).convert("RGBA")
                layer["mesh"] = make_mesh(np.array(art)[..., 3], layer["bbox"], layer["bone"], rig)
            p["warnings"].append("Setup scaffold changed. Review the rig and re-run separation if part boundaries changed.")
            return store.write(p, p["revision"])

    @app.post("/api/projects/{pid}/pose")
    def detect(pid: str):
        with store.lock:
            if store.busy(pid): raise HTTPException(409, "A project job is already running")
            p = store.load(pid)
            image = Image.open(safe_asset(store.path(pid), p["source"])).convert("RGBA")
            p["rig"] = engines.detect_pose(image, p["rig"])
            return store.write(p, p["revision"])

    @app.post("/api/projects/{pid}/background")
    def background(pid: str):
        with store.lock:
            if store.busy(pid): raise HTTPException(409, "A project job is already running")
            p = store.load(pid)
            if p["layers"]: raise ValueError("Background removal is only available before cutting; import a new image to preserve existing layer edits")
            image = Image.open(safe_asset(store.path(pid), p["source"])).convert("RGBA")
            result = engines.remove_background(image)
            name = f"foreground-{uuid.uuid4().hex}.png"; result.save(store.path(pid)/name)
            p.update(source=name, foreground_method="rembg-isnet")
            return store.write(p, p["revision"])

    @app.post("/api/projects/{pid}/cut")
    def cut(pid: str, options: dict):
        finite_tree(options)
        with store.lock:
            if store.busy(pid): raise HTTPException(409, "A project job is already running")
            p = store.load(pid)
            if options.get("revision") != p["revision"]: raise HTTPException(409, "Save or reload the current rig first")
            engine = options.get("engine", "cpu")
            if engine not in ("cpu", "sam2", "seethrough", "qwen", "imported"): raise ValueError("Unknown decomposition engine")
            if engine == "imported" and not p.get("semantic_source"): raise ValueError("This project has no imported semantic PSD")
            if engine not in ("cpu", "imported") and not engines.capabilities()[engine]["available"]:
                raise ValueError(f"The {engine} engine is not installed/configured. No existing layers were changed.")
            jid = uuid.uuid4().hex; event = threading.Event()
            job = {"id": jid, "project_id": pid, "status": "queued", "progress": 0,
                   "message": "Queued", "created_at": now(), "error": None, "cancel": event}
            store.jobs[jid] = job
            # Keep job records bounded while retaining active jobs.
            if len(store.jobs) > 256:
                for old_id, old in list(store.jobs.items()):
                    if old["status"] not in ("queued", "running"):
                        del store.jobs[old_id]
                        if len(store.jobs) <= 128: break
        def run():
            root = store.path(pid); work = root/"work"/jid; out = work/"layers"
            work.mkdir(parents=True)
            def progress(value, message): job.update(progress=min(float(value), .99), message=message)
            job.update(status="running", message="Preparing source and skeleton")
            try:
                image = Image.open(safe_asset(root, p["source"])).convert("RGBA")
                if engine in ("seethrough", "qwen", "imported"):
                    if engine == "seethrough": inputs, extra = engines.see_through(image, work, progress, event.is_set)
                    elif engine == "qwen": inputs, extra = engines.qwen_layered(image, work, progress, event.is_set, options.get("layers", 8))
                    else:
                        inputs, _, extra = read_psd(safe_asset(root, p["semantic_source"]).read_bytes())
                    layers, quality, warnings = split_semantic_layers(inputs, image.size, p["rig"], out, provider=engine, progress=progress, cancel=event.is_set)
                    warnings += extra
                else:
                    priors = engines.sam2_priors(image, p["rig"], progress, event.is_set) if engine == "sam2" else None
                    layers, quality, warnings = decompose(image, p["rig"], out,
                        padding=int(np.clip(options.get("padding", 18), 0, 64)),
                        work_size=int(np.clip(options.get("work_size", 640), 256, 1024)),
                        edge_weight=float(np.clip(options.get("edge_weight", .9), 0, 3)),
                        rigid=bool(options.get("rigid", False)), priors=priors, progress=progress, cancel=event.is_set)
                    if engine == "sam2": quality["method"] = "sam2+rig-conditioned-cpu"
                if event.is_set(): raise Cancelled("Operation cancelled; previous layers were preserved")
                if not layers: raise ValueError("The engine produced no usable parts")
                with store.lock:
                    if event.is_set(): raise Cancelled("Operation cancelled; previous layers were preserved")
                    if store.load(pid)["revision"] != p["revision"]: raise ValueError("The source project changed; generated files were not committed")
                    target = root/"layers"/jid; target.parent.mkdir(exist_ok=True)
                    shutil.move(str(out), str(target))
                    for layer in layers:
                        for key in ("image", "visible_mask", "fill_mask"):
                            layer[key] = f"layers/{jid}/{layer[key]}"
                    p.update(layers=layers, quality=quality, warnings=warnings)
                    store.write(p, p["revision"])
                    job.update(status="done", progress=1, message="Layers and weighted rig are ready", revision=p["revision"])
            except Cancelled as exc: job.update(status="cancelled", message=str(exc))
            except Exception as exc: job.update(status="failed", error=str(exc), message="Failed; previous project preserved")
            finally:
                job["finished_at"] = now()
                shutil.rmtree(work, ignore_errors=True)
        store.pool.submit(run)
        return {k: v for k, v in job.items() if k != "cancel"}

    @app.get("/api/jobs/{jid}")
    def get_job(jid: str):
        job = store.jobs.get(jid)
        if not job: raise HTTPException(404, "Job not found")
        return {k: v for k, v in job.items() if k != "cancel"}

    @app.post("/api/jobs/{jid}/cancel")
    def cancel(jid: str):
        job = store.jobs.get(jid)
        if not job: raise HTTPException(404, "Job not found")
        job["cancel"].set()
        return {"status": "cancellation_requested"}

    @app.post("/api/projects/{pid}/mask")
    def mask(pid: str, request: dict):
        finite_tree(request)
        with store.lock:
            if store.busy(pid): raise HTTPException(409, "A project job is already running")
            p = store.load(pid)
            if request.get("revision") != p["revision"]: raise HTTPException(409, "Stale project revision")
            layer = next((l for l in p["layers"] if l["id"] == request.get("layer")), None)
            if layer is None: raise ValueError("Select a valid layer")
            if len(request.get("strokes", [])) > 128: raise ValueError("Too many brush strokes")
            folder = store.path(pid)/"edits"/uuid.uuid4().hex; folder.mkdir(parents=True)
            for key in ("image", "visible_mask", "fill_mask"):
                old = safe_asset(store.path(pid), layer[key]); dest = folder/old.name
                shutil.copyfile(old, dest); layer[key] = dest.relative_to(store.path(pid)).as_posix()
            image = Image.open(safe_asset(store.path(pid), p["source"])).convert("RGBA")
            paint_layer(image, layer, request.get("strokes", []), store.path(pid), p["rig"])
            return store.write(p, p["revision"])

    @app.get("/api/projects/{pid}/export/{kind}")
    def export(pid: str, kind: str, weighted: bool = True):
        p, root = store.load(pid), store.path(pid)
        if kind == "project": data, filename, media = project_bundle(p, root), "character.moka", "application/zip"
        elif kind == "spine": data, filename, media = export_spine(p, root, weighted), "moka-spine-4.2.zip", "application/zip"
        elif kind == "psd":
            if not p["layers"]: raise ValueError("Cut layers before exporting a PSD")
            data, filename, media = export_psd(p, root), "moka-layers.psd", "image/vnd.adobe.photoshop"
        elif kind == "layers":
            output = io.BytesIO()
            with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as z:
                manifest = {"canvas": [p["width"], p["height"]], "layers": []}
                for l in sorted(p["layers"], key=lambda l: l["order"]):
                    path = f"{l['id']}.png"; z.write(safe_asset(root, l["image"]), path)
                    manifest["layers"].append({"name": l["name"], "path": path, "x": l["bbox"][0], "y": l["bbox"][1], "bone": l["bone"]})
                z.writestr("layers.json", json.dumps(manifest, ensure_ascii=False))
            data, filename, media = output.getvalue(), "moka-layers.zip", "application/zip"
        else: raise HTTPException(404, "Unknown export format")
        return Response(data, media_type=media, headers={"Content-Disposition": f'attachment; filename="{filename}"'})

    @app.get("/")
    def index():
        text = (ROOT/"web/index.html").read_text("utf-8")
        base = "/vendor/three" if (ROOT/"vendor/three/build/three.module.js").is_file() else "https://cdn.jsdelivr.net/npm/three@0.180.0"
        return HTMLResponse(text.replace("__THREE_BASE__", base))

    app.mount("/static", StaticFiles(directory=ROOT/"web"), name="static")
    if (ROOT/"vendor").is_dir(): app.mount("/vendor", StaticFiles(directory=ROOT/"vendor"), name="vendor")
    if (ROOT/"models").is_dir(): app.mount("/models", StaticFiles(directory=ROOT/"models"), name="models")
    return app


app = create_app()
