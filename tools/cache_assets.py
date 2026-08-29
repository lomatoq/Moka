#!/usr/bin/env python3
"""Cache pinned browser dependencies. Verifies npm package integrity before extraction."""
from __future__ import annotations
import argparse
import base64
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import shutil
import tarfile
import tempfile
import urllib.parse
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
PACKAGES = [("three", "0.180.0", "three"), ("@mediapipe/tasks-vision", "0.10.18", "mediapipe")]
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task"


def fetch(url, limit=160_000_000):
    if urllib.parse.urlparse(url).scheme != "https": raise ValueError("Downloads must use HTTPS")
    request = urllib.request.Request(url, headers={"User-Agent": "Moka/0.1 local asset installer"})
    with urllib.request.urlopen(request, timeout=120) as r:
        out, total = [], 0
        while True:
            chunk = r.read(1024*1024)
            if not chunk: break
            total += len(chunk)
            if total > limit: raise ValueError("Dependency download exceeds the size limit")
            out.append(chunk)
        return b"".join(out)


def extract_package(data, target):
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as archive:
        members = archive.getmembers()
        if len(members) > 8000 or sum(m.size for m in members) > 300_000_000: raise ValueError("Dependency archive is too large")
        for member in members:
            path = PurePosixPath(member.name)
            if path.is_absolute() or ".." in path.parts or "\\" in member.name or not path.parts or path.parts[0] != "package":
                raise ValueError("Unsafe dependency archive path")
            if not member.isfile():
                if not member.isdir(): raise ValueError("Dependency links are not allowed")
                continue
            relative = Path(*path.parts[1:])
            # Three's examples contain its actual format loaders and dependencies.
            if target.name.startswith("three") and relative.parts[0] not in ("build", "examples", "LICENSE", "package.json"): continue
            dest = target/relative; dest.parent.mkdir(parents=True, exist_ok=True)
            stream = archive.extractfile(member)
            if stream is None: raise ValueError("Could not read dependency member")
            dest.write_bytes(stream.read())


def install_package(name, version, folder, vendor):
    metadata_url = "https://registry.npmjs.org/"+urllib.parse.quote(name, safe="")+"/"+version
    metadata = json.loads(fetch(metadata_url, 5_000_000))
    dist = metadata["dist"]; tarball = dist["tarball"]
    if urllib.parse.urlparse(tarball).hostname != "registry.npmjs.org": raise ValueError("Unexpected npm package host")
    print(f"Downloading {name}@{version}…", flush=True)
    data = fetch(tarball)
    integrity = dist.get("integrity", "")
    if integrity:
        algorithm, expected = integrity.split("-", 1)
        if algorithm not in ("sha512", "sha256"): raise ValueError("Unsupported npm integrity algorithm")
        actual = base64.b64encode(hashlib.new(algorithm, data).digest()).decode()
        if actual != expected: raise ValueError(f"Integrity check failed for {name}")
    elif hashlib.sha1(data).hexdigest() != dist.get("shasum"):
        raise ValueError(f"Checksum failed for {name}")
    with tempfile.TemporaryDirectory(prefix=folder+"-", dir=vendor) as tmp:
        stage = Path(tmp); extract_package(data, stage)
        destination = vendor/folder; backup = vendor/(folder+".previous")
        if backup.exists(): shutil.rmtree(backup)
        if destination.exists(): destination.rename(backup)
        try: shutil.copytree(stage, destination)
        except Exception:
            shutil.rmtree(destination, ignore_errors=True)
            if backup.exists(): backup.rename(destination)
            raise
        shutil.rmtree(backup, ignore_errors=True)
    return {"package": name, "version": version, "integrity": integrity or dist.get("shasum"), "url": tarball}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Check local entry points without downloading")
    args = parser.parse_args()
    required = [ROOT/"vendor/three/build/three.module.js",ROOT/"vendor/three/examples/jsm/loaders/FBXLoader.js",
                ROOT/"vendor/mediapipe/vision_bundle.mjs",ROOT/"models/pose_landmarker_full.task"]
    if args.check:
        for path in required: print(("OK      " if path.is_file() else "MISSING ")+str(path.relative_to(ROOT)))
        raise SystemExit(0 if all(p.is_file() for p in required) else 1)
    vendor = ROOT/"vendor"; vendor.mkdir(exist_ok=True)
    manifest = {"packages": [install_package(*p, vendor) for p in PACKAGES]}
    print("Downloading the versioned MediaPipe pose model…", flush=True)
    data = fetch(MODEL_URL, 100_000_000)
    if len(data) < 1_000_000: raise ValueError("The downloaded model is unexpectedly small")
    models = ROOT/"models"; models.mkdir(exist_ok=True)
    temp = models/"pose_landmarker_full.task.tmp"; temp.write_bytes(data); temp.replace(models/"pose_landmarker_full.task")
    manifest["model"] = {"url": MODEL_URL, "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}
    (vendor/"manifest.json").write_text(json.dumps(manifest, indent=2), "utf-8")
    if not all(p.is_file() for p in required): raise RuntimeError("Package layout changed. Check the missing entry points with --check.")
    print("Browser assets cached. Restart Moka to enable the local asset routes.")


if __name__ == "__main__":
    try: main()
    except Exception as exc: raise SystemExit(f"Asset setup failed: {exc}\nCore image, rig, and BVH tools remain usable.")
