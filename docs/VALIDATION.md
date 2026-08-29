# Validation record and boundaries

## Executed in the development container

**76 Python tests passed** and **17 Node motion tests passed** after implementing the anatomical/semantic and input-validation fixes. The environment used Python 3.13 and Node 22. Detailed live results are reproducible with the commands in README; subsequent CI runs may include additional browser cases.

Verified properties include exact assembled RGBA reconstruction for the original synthetic 15-part fixture; all expected anatomical parts; disjoint visible masks; preservation of visible and unedited hidden pixels; normalized local-chain mesh weights; independently reconstructed Spine weighted vertices and transformed bones; PSD layer naming/order; atlas/project ZIP export; traversal rejection; revision conflicts; cancellation without partial commit; and all 23 See-through V3 semantic-label categories, including the `wear`/`ear` regression.

Motion tests cover BVH hierarchy and Euler order, angular wrapping, reference-pose behavior, target-proportion preservation, scale-correct root motion, projection/mirroring, confidence gaps, two-bone IK, loop closure, and planar BVH round trips.

A managed Chromium source-bundle harness uses the **actual local HTTP API**, not canned responses. Its Canvas fallback path can exercise editor interactions even where top-level loopback navigation is blocked. This harness is explicitly not a substitute for native-module/WebGL/model tests. `tests/browser_smoke.py` runs native modules by default; `--source-mode` is the documented constrained-environment variant.

## What these tests do not establish

The demo has known joints; it proves plumbing/invariants, not automatic pose detection on a random illustration. No comparative benchmark demonstrates that Moka outperforms See-through or an experienced Spine artist.

CUDA inference for SAM2/See-through/Qwen and DWPose/rembg inference were **not executed in the development container**. Their real adapters, isolation, configuration, and failure paths are implemented; model-download compatibility, VRAM use, output quality, and non-anime generalization need hardware/input-specific validation. A configured-engine badge is not a quality certificate.

Spine output is checked at the format/transform boundary, but a licensed Spine editor import/render was not performed here. PSD fallback parsing was exercised locally; the optional psd-tools path is exercised when that dependency is installed, including in CI. Advanced Photoshop blending/group effects are not guaranteed to remain pixel-identical after editable layer extraction.

## Acceptance sequence for real artwork

Use representative humanoids, exaggerated mascots, overlapping legs, a tail, and difficult clothing. Keep the same source and corrected joints for each model. Inspect setup reconstruction, all required physical segments, hidden pixels, joint extremes, left/right assignment, foot contacts, and the exported animation in Spine. Record failure examples. Do not hide missing parts by returning procedural motion or silently switching to a template.
