# Engineering rules for Moka

These are project-specific working principles for a solo-development workflow and a computational-graphics/inverse-systems approach. They are not a copy of an unavailable external SIGMA or solo-gamedev skill.

## Delivery discipline

Keep a runnable, local-first vertical slice. Do not replace functioning components with stubs, synthetic progress, or speculative architecture. A successful import is not successful model inference, and a good synthetic reconstruction is not real-character semantic accuracy. Report exactly which tests ran and on which renderer/model path.

Before edits, read README and the affected code. Preserve user projects and working routes. Update only the layers/results belonging to a completed job. Job cancellation, inference failure, and stale revisions must not destroy prior artwork. Do not download model weights automatically during startup.

## Graphics invariants

The original source remains available. Visible pixels, semantic/amodal artwork, and inferred completion have separate provenance. Do not erase an existing amodal fill while editing a different region. The known-pose CPU fixture must reconstruct the original RGBA exactly.

A semantic class is not an anatomical unit. A single topwear/legwear/body layer may cross several joints. Preserve valid small details, use explicit semantic token matching, and let the editable skeleton drive subsegmentation. Never match `ear` inside `handwear`.

Preserve target lengths during retargeting. Evaluate the source hierarchy rather than copying incompatible Euler channels. Keep coordinate conversions at import/export boundaries, unwrap angles before smoothing, report confidence gaps, and test root-motion and bind-offset behavior independently.

## Verification

Run Python tests, Node motion tests, and a real-server browser smoke test. A frontend screenshot alone is not a functional test. Validate the Spine export with an independent FK/weighted-vertex reconstruction and inspect a real editor import when available. Never introduce Spine runtime source into the independent renderer without explicit license review.

Model benchmarking must use user-representative stylized art, annotated joint/part targets, and output inspection at both setup and extreme poses. Compare CPU, SAM, See-through, and Qwen on identical inputs; report failures as well as successes. Do not tune exclusively to the demo robot.

## Simplicity

No Node build is required for the shipped editor. Keep optional ML dependencies outside the core venv. Favor plain, inspectable formats and small tested modules over new orchestration frameworks. Add capabilities incrementally; do not claim universal 3D/2D rigging from a single image.
