# Model provenance and licenses

Run `python scripts/prepare_models.py` before starting the backend or building the frontend. It fetches official repositories through verified HTTPS Git/Git LFS, checks the pinned publisher metadata and SHA256, and atomically installs each asset. Existing verified files skip the network. `--verify-only` checks readiness without changing anything; `--only backend` and `--only frontend` select assets. No archives are extracted. Binaries, enrollments and database files remain ignored.

| Asset | Destination | Publisher source and license |
| --- | --- | --- |
| YuNet `face_detection_yunet_2023mar.onnx` | `.local/models` | [OpenCV Zoo YuNet](https://github.com/opencv/opencv_zoo/tree/47534e27c9851bb1128ccc0102f1145e27f23f98/models/face_detection_yunet); its README states **all files in this directory** use [MIT](https://github.com/opencv/opencv_zoo/blob/47534e27c9851bb1128ccc0102f1145e27f23f98/models/face_detection_yunet/LICENSE), copyright 2020 Shiqi Yu. |
| SFace `face_recognition_sface_2021dec.onnx` | `.local/models` | [OpenCV Zoo SFace](https://github.com/opencv/opencv_zoo/tree/47534e27c9851bb1128ccc0102f1145e27f23f98/models/face_recognition_sface); its README states **all files in this directory** use [Apache 2.0](https://github.com/opencv/opencv_zoo/blob/47534e27c9851bb1128ccc0102f1145e27f23f98/models/face_recognition_sface/LICENSE). |
| MediaPipe Face Landmarker float16, version 1 | `frontend/public/models/face_landmarker.task` | The actual binary is [tracked in Google's official samples repository](https://github.com/google-ai-edge/mediapipe-samples/blob/c2518ec444c3a3a99689e5d31eddadc240c83a0c/examples/face_landmarker/ios/FaceLandmarker/face_landmarker.task), under that repository's [Apache 2.0 license](https://github.com/google-ai-edge/mediapipe-samples/blob/c2518ec444c3a3a99689e5d31eddadc240c83a0c/LICENSE), copyright 2022 The TensorFlow Authors. This provenance applies to the distributed asset itself, rather than inferring its license from MediaPipe API code alone. No more restrictive file-specific notice is present in the sample directory or bundle. |

The [official sample's download script](https://github.com/google-ai-edge/mediapipe-samples/blob/c2518ec444c3a3a99689e5d31eddadc240c83a0c/examples/face_landmarker/ios/RunScripts/download_models.sh) identifies the corresponding upstream model as `https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task`. The downloader uses the pinned official GitHub copy so it also works in environments where the Google Storage host is not allowed. The bundle includes face detection, landmarks, geometry metadata and blendshapes.

Pinned integrity values:

| Asset | Bytes | SHA256 |
| --- | ---: | --- |
| YuNet 2023mar | 232589 | `8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4` |
| SFace 2021dec | 38696353 | `0ba9fbfa01b5270c96627c4ef784da859931e02f04419c829e83484087c34e79` |
| Face Landmarker float16 v1 | 3758596 | `64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff` |

YuNet/SFace SHA256 values are publisher Git LFS object IDs at the pinned Zoo commit. Face Landmarker SHA256 is calculated from Google's pinned Git blob `c50c845d113e02ac6e3627b8121db7fb9a35bb4b` and checked on every installation. The expected values are code-reviewed constants; the downloader never accepts a new digest from downloaded content. License texts are retained in `scripts/model_licenses` and copied beside downloaded assets; frontend build output includes the client asset notices.

The client also self-hosts **Noto Sans Thai** from the pinned npm package [`@fontsource/noto-sans-thai` 5.3.0](https://www.npmjs.com/package/@fontsource/noto-sans-thai/v/5.3.0). The installed package's actual `LICENSE` is **SIL Open Font License 1.1**. Its exact copyright/license notice is retained in `frontend/public/licenses/noto-sans-thai-OFL.txt` and served at `/licenses/noto-sans-thai-OFL.txt`; distribute this notice with the bundled font files. The font uses no external font CDN.

YuNet 2023mar uses OpenCV 4.12.0.88. OpenCV Zoo now supplies a dynamic-shape 2026may model for OpenCV 5.x; upgrading OpenCV requires an intentional model/configuration change and inference validation. SFace embeddings are incompatible with the previous InsightFace `buffalo_l` enrollment; enroll again. A threshold of 0.363 is an initial configurable setting, not a claim of equivalent employee accuracy. Validate false acceptance/rejection on a consented local pilot dataset before commercial rollout.

These license notices support redistribution of the selected models. They do not supply consent, a retention policy, or rights to process employee photographs. Keep enrollment embeddings and attendance data in the deployment database with appropriate access and retention controls.
