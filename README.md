# Moka · Character Lab

**A local-first, editable 2D character workbench: artwork → anatomical parts → weighted rig → reusable motion → Spine.**

Moka is a working first implementation, not a claim of production-quality, one-click rigging for every drawing. It deliberately separates **detected anatomy**, **editable scaffolds**, **visible source pixels**, and **inferred hidden artwork**. A template is never presented as a successful AI detection.

## Start

Install **Python 3.12** (3.11–3.13 also work for the core). Download this repository, extract it, and run:

- **Windows:** double-click `Start Moka.bat`.
- **macOS:** run `bash "Start Moka.command"` from the project directory.
- **Any supported desktop:** `python tools/launch.py`.

The launcher creates a project-local `.venv`, installs the small core dependency set, and opens the browser. No administrator rights, Node build, paid API, cloud account, or system-Python changes are required. The first dependency installation needs internet access. Keep the terminal open while using Moka.

**Беларуская падказка:** на Windows адкрывай `Start Moka.bat`. Для першай праверкі націсні **Try demo → Separate & build rig → Wave → Export**. Дэма мае загадзя вядомыя суставы; гэта праверка працоўнага ланцужка, не доказ якасці AI-распазнавання.

## What is implemented

### Artwork and rig

Import PNG, JPG, WebP, layered PSD, positioned PNG-layer ZIP, or a portable `.moka` project. Alpha is preserved; simple border-connected backgrounds can be removed without deleting white interiors. Optional rembg handles harder backgrounds.

The editor provides biped, biped-with-tail, and quadruped **scaffolds**, draggable setup joints, custom bones, mask paint/erase, layer ordering, opacity, attachment mapping, and weighted or rigid meshes. Browser MediaPipe and optional local DWPose provide **actual human pose inference**, rather than assigning confidence to a template. Human detectors can fail on cartoon anatomy; review the proposed joints.

The CPU separator uses joint-conditioned anatomical regions and edge-aware refinement. Arms split into upper arm, forearm, and hand; legs split into thigh, shin, and foot. Visible masks are disjoint. Hidden joint overlap has its own mask and inspection overlay. The baseline extends each part's own texture into overlaps; **this is not learned reconstruction of an unseen hand or face**.

### Optional learned decomposition

Real local adapters are included for **SAM 2.1**, **See-through**, and **Qwen Image Layered**. These are not simulated API buttons. They call the upstream predictors/pipelines or an isolated local process. Model weights are not bundled, and their inference quality and GPU-memory requirements have not been benchmarked in the development container.

Crucially, a model's semantic layer is **not the final anatomical segment**. Imported/learned RGBA layers are refined by the editable rig. For example, `legwear` is eligible for separate thigh/shin/foot parts, and `topwear` for torso/upper-arm/forearm parts. The refined output retains the provider's completed hidden RGBA artwork. Hair, face, and accessory layers can remain separate attachments. Missing anatomical parts are reported instead of silently declared complete.

### Motion

Import BVH, FBX, GLB, embedded glTF, or `moka.motion/1` JSON. BVH is parsed locally; Three.js evaluates FBX/glTF skeleton hierarchies and animation clips before sampling world-space joints. Multiple embedded clips can be selected. External files referenced by 3D assets are blocked: use self-contained files.

MP4/WebM and browser-decodable MOV can be sampled frame by frame through a MediaPipe worker. The source image/video stays on the computer. The first model/library download requires internet unless assets are cached. Missing detections remain missing; no placeholder performance is generated.

Retargeting maps anatomical roles, projects source joint directions, preserves **target bone lengths**, smooths angular tracks, optionally locks feet, and offers reference-pose, view-yaw, mirroring, root-motion, and loop-closure controls. Monocular MediaPipe world landmarks are hip-centered estimates, not recovered global root motion; observed 2D tracks preserve screen-space translation.

There is a timeline, scrubbing, playback speed, editable rotation keys, and clearly labeled procedural Idle/Walk/Wave test clips. This is a **2D target editor**. Importing a 3D source does not turn the target illustration into a volumetric character.

### Export

**Spine 4.2 JSON + atlas + PNG**, including weighted meshes, hierarchy, animation rotation/translation tracks, and an independent Y-down → Y-up coordinate conversion. Also export editable layered PSD, positioned PNG layers, portable `.moka`, and a **planar BVH** of the actual 2D rig.

The planar BVH is not recovered 3D motion. The Spine exporter is tested with independent transform/vertex reconstruction; a manual import/render check in the licensed Spine editor remains an important acceptance test. No Spine runtime is bundled with Moka.

## Offline browser assets

The core editor and BVH path work without CDN dependencies. MediaPipe and Three.js can be cached from their official distributions:

```sh
python tools/cache_assets.py
python tools/cache_assets.py --check
```

Restart Moka afterwards. The cache verifies npm package integrity, preserves upstream license files, and stores a checksum of the versioned pose model. Without a cache, these optional browser modules load from pinned CDN URLs. Neither path uploads the artwork or video.

## Optional AI installation

The optional installer uses a separate `.venv-ai`; See-through uses its own `.venv-seethrough`. It does not install Torch/diffusers into the core environment.

```sh
python tools/setup_ai.py pose
python tools/setup_ai.py background
python tools/setup_ai.py sam2
python tools/setup_ai.py qwen
```

For See-through, use **Python 3.12**:

```sh
python3.12 tools/setup_ai.py seethrough
# Windows: py -3.12 tools/setup_ai.py seethrough
```

An existing working See-through installation can be reused with `--existing-home PATH --existing-python PATH`. Restart Moka after setup. Dependencies are installed when explicitly requested; weights download on first inference. The GPU installers target NVIDIA CUDA, preferably Linux/WSL2. They are not a promise that every upstream dependency builds natively on Windows or works on Apple Silicon. Core and browser tools do not require CUDA.

Successful package imports mean **configured**, not “model quality validated.” Model jobs fail visibly and leave the previous project intact. Inspect hidden artwork before exporting. Never assume a generative layer preserves every visible detail of the original.

## Data, safety, and limits

Projects live in `.moka/projects/`; export `.moka` to back them up. Model paths live in `.moka/engines.json`. The server binds to loopback by default, rejects cross-origin writes, bounds uploads/decoded image data, and checks ZIP/asset paths. It is a single-user desktop tool, **not an authenticated public hosting service**.

Current limits include a 4096 px source side, 128 MB uploads, 256 imported layers, 64 million decoded layer pixels, 64 target bones, and 10-minute imported motion clips. Browser video capture is limited to 180 seconds and 30 fps. A single flat view cannot supply correct new artwork for arbitrary large turns or severe self-occlusion. Draw order is editable but static; dynamic limb crossing needs manual review.

## Tests and further development

```sh
python -m pip install -r requirements.txt -r requirements-dev.txt
python -m pytest -q
node --test tests/*.test.mjs
python -m playwright install chromium
python tests/browser_smoke.py
```

For the optional browser model/3D integration suite, cache browser assets and run `python tests/browser_smoke.py --assets`. Supply `--pose-fixture /path/to/a/full-body-photo.jpg` to exercise real pose detection. The synthetic fixture is original project test art, not a demonstration of detector generalization.

Read [architecture](docs/ARCHITECTURE.md), [validation boundaries](docs/VALIDATION.md), [upstream sources](docs/SOURCES.md), and [agent engineering rules](AGENTS.md) before changing the inverse-graphics pipeline.
