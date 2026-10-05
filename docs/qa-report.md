# Independent QA evidence

Validation date: 2026-10-04. This report describes checks actually executed on the cloud CPU environment. It does not establish population recognition accuracy, equivalence with InsightFace, physical CUDA operation, or real employee phone camera behavior.

## Real model inference

`PYTHONPATH=. /workspace/.venvs/face-recog-insightface/bin/python e2e/model_real_inference.py --different-person /tmp/attendance-astronaut.jpg --report /tmp/attendance-model-qa.json`

- Model: `opencv-sface-2021dec-v1`; actual `CPUExecutionProvider` with YuNet detection.
- Existing `face1.jpg` enrollment and `face2.jpg` comparison: cosine **0.8279105424880981**.
- Known different person, public-domain astronaut image bundled with scikit-image (Eileen Collins), compared to `face1.jpg`: cosine **0.2206738293170929**.
- Configured cosine threshold: **0.363**. The genuine pair passes; the distinct-person comparison fails.
- Separately computed OpenCV `FaceRecognizerSF.feature()` reference embeddings match ONNX Runtime embeddings with cosine **1.0** for each of the three fixtures. This independently checks image alignment, channel order, and feature preprocessing for those fixtures.
- Explicit `cuda` selection raises a clear error because the available runtime has no `CUDAExecutionProvider`; it does not silently use CPU. No physical GPU was tested.

The coordinator also ran the preserved original InsightFace `buffalo_l` model on the same three files: genuine cosine **0.865996**, distinct-person cosine **0.04374604**. This is a fixture comparison only. Model score distributions and decision thresholds differ, so these numbers do not demonstrate population accuracy parity. Timings are not compared because the original FaceAnalysis pipeline performs additional tasks.

The negative fixture was generated into `/tmp` from the original environment's `skimage.data.astronaut()` with RGB→BGR conversion and JPEG encoding. It is a known different individual; modified photographs of the enrolled person were not counted as negative examples. No additional biometric fixture or generated employee record is committed.

## Independent API integration

`PYTHONPATH=. /workspace/.venvs/face-recog-insightface/bin/python e2e/run_api_qa.py --different-person /tmp/attendance-astronaut.jpg --report /tmp/attendance-api-qa.json`

**55 checks passed** against a temporary server and fresh disposable SQLite database, using real face inference. The helper deletes its database and stops its server after execution. Credentials are generated randomly at runtime; there are no shipped test accounts or default passwords.

Checks cover first administrator setup and repeat rejection; correct/incorrect login; HttpOnly/SameSite authentication cookie; cross-origin mutation rejection; anonymous and employee/admin boundaries; employee-isolated history; enrollment/reset and enrollment replacement restriction; missing enrollment; malformed, no-face, and multiple-face photographs; known different-person mismatch; invalid client challenge metadata; client identity/similarity injection; genuine enrollment-to-attendance; UTC server timestamps; retry deduplication; incompatible reuse of a request ID; invalid attendance transitions; newest-first history; disabled-account session revocation; reactivation; logout and logged-out cookie replay. All rejected face/challenge cases leave history empty.

Concurrent identical submission requests return one `201` and one `200`, with the same event ID. Concurrent distinct request IDs for check-in return one `201` and one `409`, with only one valid transition recorded.

The coordinator repeated all **55 API checks** using the fresh dedicated `/workspace/.venvs/attendance` environment and also ran the Python suite with **40 tests passing** (37 backend tests and 3 model-downloader tests). The API evidence above therefore applies to the final application dependency environment.

The deployment owner also ran all **55 API checks against PostgreSQL 16.15 in Docker**, using actual CPU inference and disposable validation records: all passed. Concurrent different request IDs returned `201`/`409`; concurrent identical request IDs returned `201`/`200`, preserving exactly one event. Genuine cosine was **0.8279105424880981**, and the known different-person fixture was rejected. The coordinator independently confirmed the deployment health response reported database/models ready and CPU inference. Deployment report: `/tmp/attendance-postgres-api-report.json`.

## Review findings

The first review identified that Vite's `changeOrigin: true` rewrote `Host` to the API host while browser `Origin` retained the frontend host, conflicting with the backend's same-origin mutation check. The frontend now preserves `Host` with `changeOrigin: false`; actual browser setup, employee creation, enrollment, and attendance mutations succeeded through that proxy.

Review also identified that a completed challenge ignored face loss before the final photograph had been captured. The reducer now resets on face loss, multiple faces, or a frame gap even after motion completion; a targeted unit test covers that window. The first phone camera screenshot placed instructions/shutter below the fold and beneath fixed navigation. The frontend now compacts the camera flow and hides its bottom navigation while capturing; updated phone screenshots show the preview and instructions together.

The complete browser run found two minor touch targets: the account avatar was 40×40 pixels and the dismiss-notice close button was 16.27×44. Both were enlarged to at least 44×44. A focused DOM inspection with mocked API data verified those rectangles and no horizontal overflow after the CSS correction; it is separate from the complete workflows with real API inference.

## Browser and challenge scope

Actual Chromium MediaPipe **0.10.32** inference passed using the checked local Face Landmarker task and local npm package WASM. Both original fixtures produced exactly one face, **478 landmarks**, and **52 blendshapes**. Actual 4×4 outputs place translation at indices 12/13/14, with indices 3/7/11 zero and index 15 one, directly confirming column-major storage. Horizontal mirroring reversed the measured yaw: `face1` **−6.886378915→+6.361729898°** and `face2` **−6.559255547→+6.286659272°**; the nose's horizontal displacement relative to the eye midpoint had matching sign. Model lifecycle cleanup (`close`) executed. These are static-image browser inference checks, without a camera or live-person motion claim.

`/opt/codex/runtimes/codex-primary-runtime/dependencies/python/bin/python e2e/browser_models.py --package-dir frontend/node_modules/@mediapipe/tasks-vision --report /tmp/attendance-browser-model-qa.json`

The first independent model run used the identical pinned npm package downloaded separately into `/tmp` while frontend installation completed.

The complete application browser run passed **19 workflow checks** and **57 layout checks** (19 screens at **390×844**, **768×1024**, and **1440×900**), with zero horizontal overflow and zero uncaught browser errors. The frontend owner also verified TypeScript, the production build, and **21 unit tests** covering challenge and lighting behavior.

`/opt/codex/runtimes/codex-primary-runtime/dependencies/python/bin/python e2e/run_browser_qa.py --backend-python /workspace/.venvs/attendance/bin/python --artifacts /tmp/attendance-ui-manual-resume-qa --report /tmp/attendance-ui-manual-resume-qa.json`

Verified workflows include browser setup through the same-origin proxy; administrator account creation, photo enrollment, enrollment reset, account disable/re-enable and confirmation; employee authentication/admin-navigation isolation; enrollment capture and review; empty/populated history; network error and retry; native Chromium camera permission denial with helpful Thai error/retry; camera track cleanup; logout and reload. After a real SFace check-in was accepted, the test deliberately discarded its network response; retrying the retained browser submission returned HTTP 200 with `duplicate: true`, the same request ID, and only one database event.

Lighting tests used an explicitly **simulated static camera** derived from the existing fixture. Black/dim frames displayed the Thai low-light warning, kept enrollment disabled, and produced **zero application `detectForVideo` calls**. After stable bright frames, inference and capture remained paused until the user clicked **“เริ่มตรวจต่อ”**. Bright enrollment then used actual MediaPipe detection and actual backend SFace inference. During attendance, the test reached step 2, dimmed the video, confirmed inference/capture/events stopped, restored brightness, verified continued pause, and explicitly resumed into **step 1 of a new three-action challenge**.

Successful motion completion uses **test-only synthetic pose/landmark/blendshape frames inside the isolated browser process**. The production randomized challenge reducer, UI, camera snapshot, and backend SFace comparison remain active; production source contains no bypass. Actual SFace recorded check-in and check-out and updated next-action/history correctly. A static photograph without synthetic motion did not complete the challenge or create an event. These checks do not establish physical employee motion, real phone camera behavior, population false rejection rates, or spoof resistance. Validate those in a real employee phone pilot.

The browser runner cleans its temporary database, simulated video, logs, and owned services. Screenshots and reports remain outside the checkout in the artifact folder above. These browser helpers target Linux/system Chromium and use the primary Python with Playwright; see `e2e/README.md` for optional `requirements-e2e.txt` installation when using another Python environment.


## MediaPipe pose matrix provenance

The official [MatrixData proto](https://github.com/google-ai-edge/mediapipe/blob/master/mediapipe/framework/formats/matrix_data.proto) defines `COLUMN_MAJOR` as the default storage layout. The [FaceGeometry pipeline](https://github.com/google-ai-edge/mediapipe/blob/master/mediapipe/tasks/cc/vision/face_geometry/libs/geometry_pipeline.cc) serializes the pose with `MatrixDataProtoFromMatrix`. The [matrix serialization implementation](https://github.com/google-ai-edge/mediapipe/blob/master/mediapipe/framework/formats/matrix.cc) clears the layout to its default and copies `Eigen::MatrixXf.data()` (column-major). The [web FaceLandmarker implementation](https://github.com/google-ai-edge/mediapipe/blob/master/mediapipe/tasks/web/vision/face_landmarker/face_landmarker.ts) copies the packed data unchanged. Therefore the frontend's column-major decomposition follows the actual graph's serialization contract, beyond its own constructed-matrix unit test. Actual fixture matrix output and its runtime use are checked separately.
