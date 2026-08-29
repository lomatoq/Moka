# Upstream references

Primary references checked while implementing this version (29 August 2026). These are implementation references, not performance endorsements. Upstream software/model licenses apply independently of Moka's own license. Model weights are downloaded only when the relevant feature is explicitly used.

- See-through: https://github.com/shitagaki-lab/see-through ; pinned installer commit `7f139bb25c46a0c8ac720d95ddab185fcda5451c`. Its `common/live2d/scrap_model.py` defines V3 semantic tags. The CLI outputs amodal PSD layers, not a complete Spine rig. https://arxiv.org/abs/2602.03749
- SAM 2: https://github.com/facebookresearch/sam2 ; pinned code commit `2b90b9f5ceec907a1c18123530e92e794ad901a4`. Moka uses the official `SAM2ImagePredictor` with `facebook/sam2.1-hiera-small` and anatomy-conditioned point prompts.
- Qwen Image Layered: https://huggingface.co/Qwen/Qwen-Image-Layered ; https://github.com/huggingface/diffusers ; installer commit `c1bf18c92c6285334adcaac7e75ef8946a227f49`. Generic transparent layer decomposition is followed by our anatomical subdivision; it is not assumed to provide per-bone labels.
- RTMLib/DWPose: https://github.com/Tau-J/rtmlib ; Wholebody, ONNX Runtime CPU mode. Human pose estimation with derived endpoints, not an animal model.
- MediaPipe Pose Landmarker: https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker/web_js ; browser package `@mediapipe/tasks-vision@0.10.18`; versioned full float16 pose model. World landmarks are monocular estimates.
- MediaPipe test image provenance: `pose.jpg` is listed in https://github.com/google-ai-edge/mediapipe/blob/master/mediapipe/tasks/testdata/vision/BUILD and downloadable from the official MediaPipe test-asset bucket. It is used only for optional CI inference and is not included as Moka demo art.
- Three.js: https://threejs.org/docs/ ; package `three@0.180.0`; official `FBXLoader`, `GLTFLoader`, `AnimationMixer`. Moka's own synthetic GLTF/GLB integration fixture has a known animated two-bone hierarchy.
- Spine JSON reference: https://esotericsoftware.com/spine-json-format ; official 4.2 `SkeletonJson.ts` consulted to verify weighted-vertex layout and modern rotation timeline `value` fields: https://github.com/EsotericSoftware/spine-runtimes/blob/4.2/spine-ts/spine-core/src/SkeletonJson.ts . No Spine runtime source is redistributed.
- PSD: https://psd-tools.readthedocs.io/ ; https://www.adobe.com/devnet-apps/photoshop/fileformatashtml/ . Moka's fallback writer/reader handles basic editable 8-bit RGBA PSD layers; psd-tools is preferred for more complex inputs.

Own test fixtures and CPU/renderer/export implementations are part of Moka. They are not copied commercial animation libraries or paid-service integrations.
