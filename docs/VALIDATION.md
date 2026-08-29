# Validation record — Moka v0.1

## Verified on GitHub Actions

Code revision: `9837a9fba25b05b85418cff4b5cf8abe17566655`.
Run: https://github.com/lomatoq/Moka/actions/runs/33231374298
Environment: clean Ubuntu runner, Python 3.12, Node 22, Playwright Chromium.

- **77 Python tests passed**, including an independent psd-tools writer/reader and compositing check.
- **17 Node tests passed** for motion parsing, coordinate transforms, interpolation, conditioning, IK, and retarget invariants.
- **Native browser end-to-end scenario passed** with real ES-module loading, WebGL2 rendering, and the real local FastAPI service.
- Actual GLTF and GLB loaders evaluated a skinned, animated fixture through Three AnimationMixer. Eleven sampled world-space frames were checked against a known rotation.
- Actual MediaPipe PoseLandmarker inference returned 33 landmarks from the official photographic test fixture. No pose output was mocked.

The browser scenario drags a setup joint and verifies persisted coordinates, cuts 15 parts, edits two bones at one timestamp, imports a real BVH, retargets it, saves/reloads the project, and downloads/parses a Spine 4.2 ZIP containing two animations. Machine-readable browser evidence is in `browser-validation-ci.json`; the workflow artifact also contains the screenshot, exported Spine ZIP, and server log.

The pinned TensorFlow Lite build writes its exact successful XNNPACK CPU delegate initialization notice through the browser error channel. The test records that exact notice separately as informational. Other application, WASM, inference, and console errors still fail the test.

## Measured core invariants

- The original 768-pixel demo's CPU visible partition covers the source and reassembles its RGBA pixels exactly in the setup pose.
- Visible masks are disjoint; vertex weights sum to one; the bind pose is an identity deformation.
- Joint-overlap completion is drawn only under a front part. Visible source pixels are not repainted by the CPU fill.
- Imported semantic/amodal colors are retained instead of being resampled from the flattened input. Unknown semantic coverage is reported as unknown, not as a fabricated perfect score.
- PSD round-trips cover Unicode names, opacity, channel decoding, and bottom-to-top layer order. An independent psd-tools composite catches writer/reader errors that could otherwise cancel each other.
- Independently evaluated exported Spine bone transforms and weighted vertices match Moka's posed points, including nonzero root translation.
- Upload/path/archive budgets, project revision conflicts, and cancellation preserving previous layers have regression coverage.

These are correctness checks on controlled fixtures, not evidence that arbitrary artwork is segmented perfectly.

## Additional local validation

The managed offline Chromium environment was also tested through the source-bundle/real-API bridge and Canvas fallback. `browser-validation-local.json` records that separate run. It is not substituted for the native browser/real model test above.

## Implemented but not production-validated

- DWPose, SAM2.1, See-through, Qwen-Image-Layered, and neural background-removal adapters have executable integration and installation paths. Their GPU inference and output quality were **not tested on this machine**; no comparative superiority is claimed.
- FBX loading is implemented but an actual production Mixamo FBX was **not exercised** in this validation. GLTF/GLB and BVH were exercised.
- Video decoding, frame sampling, and MediaPipe VIDEO-mode capture are implemented. The real-model check above tests still-image inference, **not an end-to-end recorded performance** or monocular mocap accuracy.
- Spine 4.2 format and transform invariants were tested, but no licensed Spine desktop import/runtime rendering session was available.
- Windows/macOS launchers were inspected, not executed on those operating systems. The clean install test ran on Linux.
- Animal/stylized pose accuracy, crossing-limb cleanup, dynamic draw-order tracks, unseen-view generation, full hidden-limb reconstruction, and 3D target-mesh retargeting are not established capabilities of this release.

CPU completion only extends a part's own texture around joints. It is not neural reconstruction of a fully hidden leg. MediaPipe is a human detector; anatomy scaffolds are explicitly labeled templates, not successful detections.
