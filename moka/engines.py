"""Optional real inference adapters. Imports are lazy; no paid API is used.

No model weights are bundled. First use may download upstream weights. An
installed Python package is NOT advertised as a validated/loaded model.
"""
from __future__ import annotations
import copy
import gc
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import signal
import time
import uuid
import numpy as np
from PIL import Image
from .formats import read_psd, safe_asset
from .vision import Cancelled

_CACHE = {}


def _has(name):
    try: return importlib.util.find_spec(name) is not None
    except (ModuleNotFoundError, ValueError): return False


def _external(engine):
    python = os.environ.get("MOKA_AI_PYTHON", "")
    configured = os.environ.get("MOKA_AI_ENGINES", "").split(",")
    return bool(python) and engine in configured and Path(python).is_file() and Path(python).resolve() != Path(sys.executable).resolve()


def _external_run(engine, image, rig=None, progress=None, cancel=None, **options):
    """One job per process: model dependency conflicts and GPU allocations stay isolated."""
    root = Path(__file__).resolve().parents[1]
    scratch = root/".moka/engine-jobs"; scratch.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=engine+"-", dir=scratch) as folder:
        work = Path(folder); image.save(work/"input.png")
        request = {"engine":engine,"rig":rig,**options}
        path = work/"request.json"; path.write_text(json.dumps(request), "utf-8")
        env = dict(os.environ); env.pop("MOKA_AI_PYTHON", None); env.pop("MOKA_AI_ENGINES", None)
        with (work/"engine.log").open("w", encoding="utf-8") as log:
            proc = subprocess.Popen([os.environ["MOKA_AI_PYTHON"], "-m", "moka.engine_worker", str(path)],
                cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=os.name != "nt")
            start = time.monotonic()
            try:
                while proc.poll() is None:
                    if cancel and cancel(): raise Cancelled(f"{engine} cancelled; previous layers preserved")
                    if time.monotonic()-start > 3600: raise RuntimeError("Optional model exceeded the one-hour job limit")
                    if progress and (work/"progress.json").is_file():
                        try:
                            state = json.loads((work/"progress.json").read_text("utf-8")); progress(state["value"], state["message"])
                        except (OSError, ValueError): pass
                    time.sleep(.3)
            finally:
                if proc.poll() is None:
                    if os.name != "nt": os.killpg(proc.pid, signal.SIGTERM)
                    else: proc.terminate()
                    try: proc.wait(timeout=8)
                    except subprocess.TimeoutExpired:
                        if os.name != "nt": os.killpg(proc.pid, signal.SIGKILL)
                        else: proc.kill()
                        proc.wait()
        if proc.returncode:
            tail = (work/"engine.log").read_text("utf-8", errors="replace")[-3500:]
            raise RuntimeError(f"Local {engine} failed. Existing project preserved.\n{tail}")
        result = json.loads((work/"result.json").read_text("utf-8"))
        if engine == "dwpose": return result["rig"]
        if engine == "rembg": return Image.open(safe_asset(work, result["image"])).convert("RGBA").copy()
        if engine == "sam2": return {k: np.array(Image.open(safe_asset(work, v)).convert("L")) for k,v in result["masks"].items()}
        return [(p["name"],Image.open(safe_asset(work,p["path"])).convert("RGBA").copy(),tuple(p["offset"])) for p in result["parts"]], result["warnings"]


def capabilities():
    home = os.environ.get("MOKA_SEETHROUGH_HOME", "")
    return {
        "cpu": {"available": True, "label": "Rig-conditioned CPU", "kind": "deterministic", "tested_inference": True},
        "dwpose": {"available": _external("dwpose") or (_has("rtmlib") and _has("onnxruntime")), "label": "DWPose / RTMPose", "kind": "optional-model", "tested_inference": False},
        "rembg": {"available": _external("rembg") or _has("rembg"), "label": "rembg foreground", "kind": "optional-model", "tested_inference": False},
        "sam2": {"available": _external("sam2") or _has("sam2"), "label": "SAM 2.1 + anatomy", "kind": "optional-model", "tested_inference": False},
        "seethrough": {"available": bool(home) and (Path(home)/"inference/scripts/inference_psd.py").is_file(), "label": "See-through + joint split", "kind": "external-local-engine", "tested_inference": False},
        "qwen": {"available": _external("qwen") or (_has("diffusers") and _has("torch")), "label": "Qwen Image Layered + joint split", "kind": "optional-model", "tested_inference": False},
    }


def detect_pose(image: Image.Image, rig: dict):
    if _external("dwpose"): return _external_run("dwpose", image, rig)
    if rig.get("preset") == "quadruped":
        raise ValueError("DWPose is a human model, not an animal detector. Use the quadruped scaffold and adjust joints.")
    if not capabilities()["dwpose"]["available"]:
        raise ValueError("DWPose is not installed. Run the optional AI setup for 'pose', or use browser pose detection.")
    import cv2
    from rtmlib import Wholebody
    if "dwpose" not in _CACHE:
        _CACHE["dwpose"] = Wholebody(to_openpose=False, mode="balanced", backend="onnxruntime", device="cpu")
    rgb = np.array(image.convert("RGB"))
    keypoints, scores = _CACHE["dwpose"](cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    if len(keypoints) == 0: raise ValueError("No reliable human skeleton detected; original joints were preserved")
    person = int(np.argmax(np.mean(scores[:, :17], axis=1)))
    p, c = keypoints[person], scores[person]
    if np.mean(c[[5, 6, 11, 12]]) < .35:
        raise ValueError("Pose confidence is too low for auto-rigging. Adjust the visible scaffold instead.")
    roles = {"shoulder_l": 5, "shoulder_r": 6, "elbow_l": 7, "elbow_r": 8,
             "wrist_l": 9, "wrist_r": 10, "hip_l": 11, "hip_r": 12,
             "knee_l": 13, "knee_r": 14, "ankle_l": 15, "ankle_r": 16}
    updates = {role: (p[i], float(c[i])) for role, i in roles.items() if c[i] >= .25}
    chest, pelvis = (p[5]+p[6])/2, (p[11]+p[12])/2
    nose = p[0]
    updates.update(chest=(chest, float(min(c[5], c[6]))), pelvis=(pelvis, float(min(c[11], c[12]))))
    if c[0] >= .25:
        updates["neck"] = (chest + .42*(nose-chest), float(c[0]))
        updates["head_tip"] = (chest + 1.55*(nose-chest), float(c[0])*.8)
    # COCO WholeBody feet: left big toe/small toe/heel then right equivalents.
    for side, footidx, wrist, elbow in (("l", 17, 9, 7), ("r", 20, 10, 8)):
        if len(p) > footidx and c[footidx] > .25:
            updates[f"toe_{side}"] = (p[footidx], float(c[footidx]))
        updates[f"hand_{side}"] = (p[wrist]+.28*(p[wrist]-p[elbow]), float(c[wrist])*.6)
    out = copy.deepcopy(rig)
    for j in out["joints"]:
        if j["id"] in updates:
            xy, conf = updates[j["id"]]
            j.update(x=float(xy[0]), y=float(xy[1]), confidence=conf, source="dwpose")
    out.update(provenance="dwpose-with-derived-endpoints", confidence=float(np.mean(c[:17])))
    return out


def remove_background(image: Image.Image):
    if _external("rembg"): return _external_run("rembg", image)
    if not _has("rembg"): raise ValueError("rembg is not installed; run tools/setup_ai.py background")
    from rembg import new_session, remove
    if "rembg" not in _CACHE:
        _CACHE["rembg"] = new_session("isnet-general-use")
    return remove(image.convert("RGBA"), session=_CACHE["rembg"]).convert("RGBA")


def sam2_priors(image: Image.Image, rig: dict, progress=None, cancel=None):
    if _external("sam2"): return _external_run("sam2", image, rig, progress, cancel)
    if not _has("sam2"): raise ValueError("SAM 2 is not installed; see the optional AI setup")
    import torch
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if "sam2" not in _CACHE:
        _CACHE["sam2"] = SAM2ImagePredictor.from_pretrained("facebook/sam2.1-hiera-small", device=device)
    predictor = _CACHE["sam2"]
    rgb = np.array(image.convert("RGB"))
    joints = {j["id"]: np.array([j["x"], j["y"]]) for j in rig["joints"]}
    bones = [b for b in rig["bones"] if b.get("layer")]
    results = {}
    with torch.inference_mode():
        predictor.set_image(rgb)
        for i, b in enumerate(bones):
            if cancel and cancel(): raise Cancelled("SAM pass cancelled")
            a, z = joints[b["start"]], joints[b["end"]]
            positives = [a+(z-a)*t for t in (.3, .5, .7)]
            negatives = [(joints[o["start"]]+joints[o["end"]])/2 for o in bones
                         if o["id"] != b["id"] and o["start"] not in (b["start"], b["end"])
                         and o["end"] not in (b["start"], b["end"])]
            coords = np.asarray(positives+negatives, dtype=np.float32)
            labels = np.asarray([1]*len(positives)+[0]*len(negatives), dtype=np.int32)
            masks, scores, _ = predictor.predict(point_coords=coords, point_labels=labels, multimask_output=True)
            best = int(np.argmax(scores))
            results[b["id"]] = masks[best].astype(np.uint8)*255
            if progress: progress(.05+.30*(i+1)/len(bones), f"SAM anatomy proposal {i+1}/{len(bones)}")
    return results


def see_through(image: Image.Image, work: Path, progress=None, cancel=None):
    home = Path(os.environ.get("MOKA_SEETHROUGH_HOME", "")).expanduser().resolve()
    script = home/"inference/scripts/inference_psd.py"
    python = os.environ.get("MOKA_SEETHROUGH_PYTHON", sys.executable)
    if not script.is_file(): raise ValueError("See-through is not configured. Set up the optional local engine first.")
    stem = "moka_" + uuid.uuid4().hex
    source = work/f"{stem}.png"; image.save(source)
    args = [python, str(script), "--srcp", str(source), "--save_to_psd", "--group_offload"]
    log_path = work/"seethrough.log"
    if progress: progress(.03, "Running local See-through; first use may download model weights")
    with log_path.open("w", encoding="utf-8") as log:
        proc = subprocess.Popen(args, cwd=home, stdout=log, stderr=subprocess.STDOUT, shell=False)
        start = time.monotonic()
        try:
            while proc.poll() is None:
                if cancel and cancel(): raise Cancelled("See-through was cancelled")
                if time.monotonic()-start > 3600: raise TimeoutError("See-through exceeded the one-hour safety timeout")
                time.sleep(.3)
        finally:
            if proc.poll() is None:
                proc.terminate()
                try: proc.wait(timeout=8)
                except subprocess.TimeoutExpired: proc.kill(); proc.wait()
    if proc.returncode != 0:
        tail = log_path.read_text(encoding="utf-8", errors="replace")[-2500:]
        raise RuntimeError("See-through failed; no previous layers were replaced.\n" + tail)
    results = list((home/"workspace/layerdiff_output").rglob(f"*{stem}*.psd"))
    if not results:
        # Some upstream revisions put the PSD inside an input-named directory.
        results = [p for p in (home/"workspace/layerdiff_output").rglob("*.psd") if stem in str(p)]
    if not results: raise RuntimeError("See-through completed but no matching PSD was produced; inspect its log")
    layers, size, warnings = read_psd(max(results, key=lambda p: p.stat().st_mtime).read_bytes())
    if size != image.size:
        sx, sy = image.width/size[0], image.height/size[1]
        layers = [(name, art.resize((max(1, round(art.width*sx)), max(1, round(art.height*sy))), Image.Resampling.LANCZOS), (round(x*sx), round(y*sy)))
                  for name, art, (x, y) in layers]
    return layers, warnings


def qwen_layered(image: Image.Image, work: Path, progress=None, cancel=None, layers=8):
    if _external("qwen"): return _external_run("qwen", image, progress=progress, cancel=cancel, layers=layers)
    if not _has("diffusers"): raise ValueError("Qwen Image Layered is not installed")
    import torch
    try:
        from diffusers import QwenImageLayeredPipeline
    except ImportError as exc:
        raise ValueError("Your diffusers version has no QwenImageLayeredPipeline; use the Qwen optional setup") from exc
    if not torch.cuda.is_available():
        raise ValueError("This Qwen adapter requires CUDA. Use See-through, imported PSD, or the CPU pipeline on this machine.")
    if progress: progress(.02, "Loading Qwen Image Layered with CPU offload; weights are not bundled")
    pipeline = QwenImageLayeredPipeline.from_pretrained("Qwen/Qwen-Image-Layered", torch_dtype=torch.bfloat16)
    pipeline.enable_model_cpu_offload()
    def callback(pipe, step, timestep, kwargs):
        if cancel and cancel(): raise Cancelled("Qwen decomposition cancelled")
        if progress: progress(.08+.20*(step+1)/30, f"Qwen decomposition · denoising {step+1}/30")
        return kwargs
    try:
        with torch.inference_mode():
            result = pipeline(image=image.convert("RGBA"), generator=torch.Generator(device="cuda").manual_seed(777),
                true_cfg_scale=4.0, negative_prompt=" ", num_inference_steps=30, num_images_per_prompt=1,
                layers=int(np.clip(layers, 2, 12)), resolution=640, cfg_normalize=True, use_en_prompt=True,
                callback_on_step_end=callback)
        outputs = result.images[0]
        return [(f"qwen_layer_{i}", art.convert("RGBA").resize(image.size, Image.Resampling.LANCZOS), (0, 0)) for i, art in enumerate(outputs)], [
            "Qwen produces generic layers, not guaranteed anatomical labels. Moka splits them by joints; inspect the result."]
    finally:
        del pipeline
        gc.collect()
        torch.cuda.empty_cache()
