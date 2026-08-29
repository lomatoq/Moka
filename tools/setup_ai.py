#!/usr/bin/env python3
"""Optional local engines in isolated environments; never modifies system Python."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"tools"))
from launch import environment, install_core
SAM2_COMMIT = "2b90b9f5ceec907a1c18123530e92e794ad901a4"
SEETHROUGH_COMMIT = "7f139bb25c46a0c8ac720d95ddab185fcda5451c"
DIFFUSERS_COMMIT = "c1bf18c92c6285334adcaac7e75ef8946a227f49"


def run(args, cwd=ROOT, env=None):
    print("+ "+" ".join(map(str, args)), flush=True)
    subprocess.run(list(map(str, args)), cwd=cwd, env=env, check=True)


def configure(values):
    folder = ROOT/".moka"; folder.mkdir(exist_ok=True)
    path = folder/"engines.json"
    data = json.loads(path.read_text("utf-8")) if path.is_file() else {}
    if "MOKA_AI_ENGINES" in values:
        values["MOKA_AI_ENGINES"] = ",".join(sorted((set(data.get("MOKA_AI_ENGINES", "").split(",")) | set(values["MOKA_AI_ENGINES"].split(","))) - {""}))
    data.update(values)
    temp = path.with_suffix(".tmp"); temp.write_text(json.dumps(data, indent=2), "utf-8"); temp.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("engine", choices=["pose", "background", "sam2", "qwen", "seethrough"])
    parser.add_argument("--existing-home", type=Path, help="Reuse an already installed See-through repository")
    parser.add_argument("--existing-python", type=Path, help="Python interpreter for the existing See-through environment")
    args = parser.parse_args()
    if args.engine == "seethrough":
        if args.existing_home:
            home = args.existing_home.expanduser().resolve()
            if not (home/"inference/scripts/inference_psd.py").is_file(): raise RuntimeError("Not a See-through repository")
            python = (args.existing_python or Path(sys.executable)).expanduser().resolve()
            if not python.is_file(): raise RuntimeError("See-through Python interpreter was not found")
        else:
            if sys.version_info[:2] != (3, 12): raise RuntimeError("See-through's pinned environment requires Python 3.12. Run this command with python3.12 / py -3.12.")
            if not shutil.which("git"): raise RuntimeError("Git is required for this optional engine")
            home = ROOT/"vendor/see-through"; home.parent.mkdir(exist_ok=True)
            if not home.exists(): run(["git", "clone", "https://github.com/shitagaki-lab/see-through.git", home])
            run(["git", "checkout", SEETHROUGH_COMMIT], cwd=home)
            python = environment(ROOT/".venv-seethrough")
            run([python,"-m","pip","install","torch==2.8.0","torchvision==0.23.0","torchaudio==2.8.0","--index-url","https://download.pytorch.org/whl/cu128"])
            run([python,"-m","pip","install","-r","requirements.txt"], cwd=home)
            if not (home/"assets").exists(): shutil.copytree(home/"common/assets", home/"assets")
        configure({"MOKA_SEETHROUGH_HOME":str(home),"MOKA_SEETHROUGH_PYTHON":str(python)})
    else:
        python = environment(ROOT/".venv-ai"); install_core(python)
        key = {"pose":"dwpose","background":"rembg"}.get(args.engine,args.engine)
        if args.engine == "pose":
            run([python,"-m","pip","install","rtmlib","onnxruntime>=1.20,<2"])
            run([python,"-c","from rtmlib import Wholebody; import onnxruntime"])
        elif args.engine == "background":
            run([python,"-m","pip","install","rembg[cpu]>=2.0.60,<3"])
            run([python,"-c","from rembg import remove, new_session"])
        else:
            if not shutil.which("git"): raise RuntimeError("Git is required for pinned model code")
            run([python,"-m","pip","install","torch==2.8.0","torchvision==0.23.0","--index-url","https://download.pytorch.org/whl/cu128"])
            if args.engine == "sam2":
                env = dict(os.environ, SAM2_BUILD_CUDA="0")
                run([python,"-m","pip","install",f"git+https://github.com/facebookresearch/sam2.git@{SAM2_COMMIT}","huggingface-hub"],env=env)
                run([python,"-c","from sam2.sam2_image_predictor import SAM2ImagePredictor"])
            else:
                run([python,"-m","pip","install",f"git+https://github.com/huggingface/diffusers.git@{DIFFUSERS_COMMIT}","transformers>=4.57,<6","accelerate>=1.10,<2","safetensors","sentencepiece"])
                run([python,"-c","from diffusers import QwenImageLayeredPipeline"])
        configure({"MOKA_AI_PYTHON":str(python),"MOKA_AI_ENGINES":key})
    print("\nEngine configured. Restart Moka. Model weights download on first inference.\nImport checks are not a quality or GPU-memory benchmark.")


if __name__ == "__main__":
    try: main()
    except (OSError, RuntimeError, subprocess.CalledProcessError) as exc:
        raise SystemExit(f"Optional setup failed: {exc}\nThe core Moka environment was not changed.")
