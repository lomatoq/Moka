# Pipeline and invariants

## Representation

A `moka.project/1` stores the immutable imported source, editable setup joints, a parent-ordered skeleton, attachment layers, and animation clips. Image coordinates are X-right/Y-down. Bone offsets are clockwise degrees; the rest transform is derived from the two setup endpoints. Root translation is in target pixels.

Each attachment stores a cropped RGBA image, its canvas bounding box, visible/fill alpha masks, owner bone, draw order, opacity, provenance, and a regular alpha-aware triangulated mesh. Vertex weights only blend an owner with its immediate anatomical parent/child; distant/opposite limbs cannot attract the same skin automatically. Custom bones extend the schema for tails, ears, wings, and other parts.

## Segmentation is an inverse problem, not a color threshold

`vision.partition` constructs normalized distances to anatomical bone segments, adds optional SAM proposal costs, and uses an edge-sensitive Lab-space neighborhood term. Checkerboard ICM provides a bounded deterministic baseline. It does not infer correct anatomy when the skeleton itself is misplaced.

`vision.decompose` uses a disjoint visible partition. It adds small joint-overlap caps only where another source part is opaque and in front. Thus completed overlap does not change the assembled setup image. Completion extends texture from that part only. An “ambiguous geometry” score is not a learned probability or semantic accuracy score.

`vision.split_semantic_layers` instead starts from imported/learned **amodal RGBA layers**. Each semantic layer is restricted to plausible rig roles and partitioned again. It retains the provider's hidden colors instead of sampling the flattened original through them. A back-to-front opacity pass separates currently visible and hidden pixels. Unrepresented bones become warnings. This allows an upstream `legwear` class to feed upper/lower/foot segments without asking the upstream model to have those training labels.

See-through and Qwen can alter source appearance or hallucinate hidden art. Their output is not assigned a fabricated exact-reconstruction score. `visible_coverage` is null when that metric has not been measured.

## Motion

All source formats produce `moka.motion/1`: named joints, timestamps, 3D positions/confidence, optional reference pose, and optional observed 2D video frames. BVH evaluation respects declared rotation order. FBX/glTF are evaluated by Three AnimationMixer before sampling world positions. JSON imports pass finite/time/size validation.

Retargeting derives source directions in the selected projection, maps local parent-relative angular changes onto the target's setup directions, unwraps and smooths tracks, and computes FK with the **target's** offsets/lengths. Relative-reference mode uses the selected source frame as neutral. Monocular world-landmark root position is not promoted to global trajectory. Optional foot locking solves a two-segment chain without changing its length. Missing detections are tracked and conditioned, not replaced with demo motion.

The browser renders the same bind-to-pose skin matrices as the Python tests. The WebGL2 path is independent of Spine; Canvas triangles provide a slower fallback. Rendering, pose inference, and BVH/retarget calculations are separated so model work does not block ordinary editor interaction.

## Export

Spine conversion reflects Y, negates rotations, transforms each weighted vertex into the appropriate bind-bone coordinates, and writes modern 4.2 rotation `value` tracks. The atlas is a straight-alpha, padded, multi-page shelf pack with explicit bounds. The PSD writer stores editable RGBA layers and a merged preview. Project ZIPs include the current source, attachments/masks, and imported semantic PSD when present.

`clipToBVH` exports the existing 2D hierarchy into an XY plane. It is useful for inspection/interchange; it is not an inferred volumetric performance.

## Isolation and state

The API is local FastAPI; the frontend is native ES modules. `.venv` contains the editor only. Optional engines run through `.venv-ai`, or the separately pinned See-through environment. An ML subprocess reports progress to its parent; cancellation terminates it, and only a successful job is committed. Project revisions prevent stale multi-tab edits. Inputs have bounded frames, dimensions, decoded layer pixels, nesting, and archive paths.

Large production features intentionally deferred: dynamic draw-order tracks, learned animal pose, view synthesis, automatic 3D target retargeting, a fully featured dope sheet, and production deformation cleanup. No capability should be inferred merely from a file-extension label.
