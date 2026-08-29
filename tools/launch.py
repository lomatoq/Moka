#!/usr/bin/env python3
"""Cross-platform bootstrap. Never installs into the system Python environment."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import threading
import time
import urllib.request
import webbrowser

ROOT = Path(__file__).resolve().parents[1]


def venv_python(path):
    return path / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def environment(path=None):
    path = path or ROOT/".venv"
    if not (3, 11) <= sys.version_info[:2] <= (3, 13):
        raise RuntimeError(f"Use Python 3.11, 3.12, or 3.13. Selected interpreter: {sys.executable} ({sys.version.split()[0]})")
    python = venv_python(path)
    if not python.is_file():
        if path.exists():
            backup = path.with_name(path.name+f"-incomplete-{int(time.time())}")
            path.rename(backup)
            print(f"Preserved the incomplete environment as {backup}")
        print(f"Creating local environment with {sys.executable}", flush=True)
        subprocess.run([sys.executable, "-m", "venv", str(path)], check=True, cwd=ROOT)
    return python


def install_core(python):
    requirement = ROOT/"requirements.txt"
    stamp = python.parent.parent/".moka-requirements-sha256"
    digest = hashlib.sha256(requirement.read_bytes()).hexdigest()
    if stamp.is_file() and stamp.read_text().strip() == digest:
        check = subprocess.run([str(python), "-c", "import fastapi,uvicorn,multipart,numpy,scipy,PIL,cv2"], capture_output=True)
        if check.returncode == 0: return
    print("Installing Moka's core dependencies into its local environment…", flush=True)
    subprocess.run([str(python), "-m", "pip", "install", "--disable-pip-version-check", "-r", str(requirement)], check=True, cwd=ROOT)
    stamp.write_text(digest)


def moka_on_port(port):
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/api/health", timeout=.4) as r:
            return json.load(r).get("app") == "moka"
    except Exception: return False


def choose_port(start):
    for port in range(start, min(start+20, 65536)):
        if moka_on_port(port): return port, True
        with socket.socket() as s:
            try: s.bind(("127.0.0.1", port)); return port, False
            except OSError: continue
    raise RuntimeError("No available local port was found")


def main():
    parser = argparse.ArgumentParser(description="Start the Moka web workbench")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--no-browser", action="store_true")
    parser.add_argument("--cache-assets", action="store_true", help="Download official browser libraries and pose model for offline use")
    args = parser.parse_args()
    if not 1024 <= args.port <= 65515: raise RuntimeError("Choose a port between 1024 and 65515")
    python = environment(); install_core(python)
    if args.cache_assets:
        subprocess.run([str(python), str(ROOT/"tools/cache_assets.py")], check=True, cwd=ROOT)
    port, existing = choose_port(args.port)
    url = f"http://127.0.0.1:{port}"
    if existing:
        print(f"Moka is already running: {url}")
        if not args.no_browser: webbrowser.open(url)
        return
    print(f"\nMOKA · {url}\nKeep this window open. Ctrl+C stops the local server.\n", flush=True)
    if not args.no_browser:
        def open_when_ready():
            for _ in range(100):
                if moka_on_port(port): webbrowser.open(url); return
                time.sleep(.2)
        threading.Thread(target=open_when_ready, daemon=True).start()
    process = subprocess.Popen([str(python), "-m", "moka", "--port", str(port)], cwd=ROOT)
    try:
        code = process.wait()
        if code: raise RuntimeError(f"Moka exited with code {code}. Read the error above.")
    except KeyboardInterrupt:
        process.terminate()
        try: process.wait(timeout=8)
        except subprocess.TimeoutExpired: process.kill()


if __name__ == "__main__":
    try: main()
    except (RuntimeError, OSError, subprocess.CalledProcessError) as exc:
        print(f"\nMoka could not start: {exc}\nNo system Python packages were changed.", file=sys.stderr)
        sys.exit(1)
