"""Isolated optional-engine worker. File protocol; invoked only by the local backend."""
from __future__ import annotations
import json
from pathlib import Path
import sys
import traceback
import numpy as np
from PIL import Image
from . import engines


def main():
    request_path = Path(sys.argv[1]).resolve(); work = request_path.parent
    request = json.loads(request_path.read_text("utf-8")); image = Image.open(work/"input.png").convert("RGBA")
    engine, rig = request["engine"], request.get("rig")
    def progress(value, message):
        temp = work/"progress.tmp"; temp.write_text(json.dumps({"value":value,"message":message}), "utf-8"); temp.replace(work/"progress.json")
    progress(.01, "Loading the isolated local model environment")
    if engine == "dwpose": result = {"rig": engines.detect_pose(image, rig)}
    elif engine == "rembg":
        engines.remove_background(image).save(work/"foreground.png"); result = {"image":"foreground.png"}
    elif engine == "sam2":
        masks = engines.sam2_priors(image, rig, progress)
        result = {"masks": {}}
        for name, mask in masks.items():
            filename = f"mask-{name}.png"; Image.fromarray(mask).save(work/filename); result["masks"][name] = filename
    elif engine == "qwen":
        parts, warnings = engines.qwen_layered(image, work, progress, layers=request.get("layers",8))
        result = {"parts":[], "warnings":warnings}
        for i, (name, art, offset) in enumerate(parts):
            filename = f"layer-{i}.png"; art.save(work/filename); result["parts"].append({"name":name,"path":filename,"offset":offset})
    else: raise ValueError("Unknown isolated engine")
    (work/"result.json").write_text(json.dumps(result, allow_nan=False), "utf-8")


if __name__ == "__main__":
    try: main()
    except Exception:
        traceback.print_exc(); sys.exit(1)
