# Development and deployment

Use the existing checkout. Cloud tasks should not create worktrees or assume processes survive environment publication. Requirements: Python 3.11 or 3.12, Node 22, npm, Git, Git LFS, and HTTPS access to GitHub plus Python/npm registries. Docker Compose 2.30+ is needed for the optional GPU override. No administrator password or employee accounts are shipped.

## Local CPU workflow

```bash
cd /workspace/face-recog-insightface
ATTENDANCE_VENV=/workspace/.venvs/attendance bash scripts/setup_dev.sh
source /workspace/.venvs/attendance/bin/activate
python scripts/prepare_models.py --verify-only
ATTENDANCE_INFERENCE_DEVICE=cpu python scripts/check_runtime.py --image face1.jpg
pytest
python -m unittest discover -s scripts -p 'test_*.py' -v
npm --prefix frontend test
```

Outside the cloud workspace, omit `ATTENDANCE_VENV` and activate `.venv/bin/activate`. The helper installs development dependencies, verifies all three models, uses `npm ci`, runs real detection/embedding on the existing fixture and builds React. Git LFS downloads only the selected pinned models. Run the helper from a shell with the intended Python version active. It changes no tracked fixture, embedding, or bytecode file.

For development, an isolated SQLite database is supported:

```bash
ATTENDANCE_DATABASE_URL=sqlite:///.local/attendance.db \
ATTENDANCE_COOKIE_SECURE=false ATTENDANCE_INFERENCE_DEVICE=cpu \
python -m uvicorn main:app --host 0.0.0.0 --port 8000
```

In another terminal, run `npm --prefix frontend run dev`. Vite serves port 5173 and proxies `/api` to port 8000. A production frontend build is served by FastAPI from `frontend/dist`. Check `GET /api/health` for `database_ready=true` and `models_ready=true`, then use the UI's first administrator setup screen. Check local readiness with HTTP requests; the onboarding UI does not publish loopback previews. Real mobile cameras require HTTPS (or the browser's localhost exception); LAN HTTP IP addresses do not provide the same secure-context camera permission.

## PostgreSQL and CPU containers

Prepare models on the host before building. The frontend model and installed MediaPipe WASM become build assets; backend ONNX files mount read-only from `.local/models`. The image runs as an unprivileged user and checks model integrity and actual SFace initialization/inference before starting. Health checks require both database and models.

```bash
python scripts/prepare_models.py
cp .env.example .env
```

Fill `.env` with a freshly generated PostgreSQL password and its matching connection URL. For the included Compose database, use `postgresql+psycopg://attendance:YOUR_URL_ENCODED_PASSWORD@db:5432/attendance`. URL-encode any reserved password characters. A local random hex password is convenient and avoids URL escaping. Empty credentials fail Compose validation. Protect `.env` permissions and keep it ignored. Changing `POSTGRES_PASSWORD` does not alter an already initialized PostgreSQL volume's password; rotate existing database credentials deliberately.

```bash
docker compose config --quiet
docker compose up --build -d --wait
curl --fail http://127.0.0.1:8000/api/health
docker compose logs --tail=100 app
docker compose down
```

In a cloud environment that supplies an egress proxy, use `DOCKER_CONFIG=/tmp/attendance-docker-config python scripts/build_container.py --tag attendance-validation-app` (substitute your Compose project's app image tag). The helper forwards the existing proxy variables, supplies the proxy's ordinary workspace DNS resolution to the build, and mounts the workspace's trusted CA bundle as a temporary BuildKit secret. It preserves TLS verification and stores no CA or proxy credentials in the image. If the Docker client home is read-only, the writable `DOCKER_CONFIG` directory under `/tmp` holds its build cache/configuration. Preserve any existing registry authentication when a private registry is in use. Start the built image with the same Compose project and `up --no-build`.

For loopback HTTP tests set `ATTENDANCE_COOKIE_SECURE=false`; keep it `true` for HTTPS deployment. Bind to loopback by default and place the app behind an HTTPS reverse proxy. `docker compose down` preserves the named database volume. Database backups contain biometric embeddings and must receive the same access protection. Production uses PostgreSQL; SQLite is a development/test fallback.

The HTTPS reverse proxy must preserve `Host` and send `X-Forwarded-Proto: https`. Set `FORWARDED_ALLOW_IPS` to its explicit source IP(s) as observed by the app container so Uvicorn trusts the forwarded scheme and the application's same-origin checks match the browser's HTTPS origin. The default trusts only `127.0.0.1`. Keep the app inaccessible directly from untrusted clients; do not use a wildcard trust setting for an exposed service.

## Device configuration

| Setting | Behavior |
| --- | --- |
| `ATTENDANCE_INFERENCE_DEVICE=cpu` | YuNet and SFace run on CPU; portable default in the CPU Docker image. |
| `auto` | Try available acceleration for SFace; fall back to CPU if initialization/warmup fails. See health diagnostics for the actual provider. |
| `cuda` | SFace requires a working ONNX Runtime CUDA provider. YuNet detection and alignment remain CPU. Explicit CUDA fails if unavailable; NVIDIA hardware/runtime is required. |
| `coreml` | SFace uses ONNX Runtime CoreML on **native macOS** when its provider initializes and completes warmup. YuNet detection/alignment remain CPU. CoreML may partition graph operations across Apple accelerators and CPU; provider selection is not a claim that all operations use the GPU. |
| `ATTENDANCE_MODEL_DIR` | Backend asset directory, default `.local/models`. |
| `ATTENDANCE_FACE_THRESHOLD` | Initial cosine threshold 0.363; validate for the actual workforce/photos. |
| `ATTENDANCE_DATABASE_URL` | SQLAlchemy PostgreSQL URL with the `psycopg` driver, or isolated development SQLite URL. |
| `ATTENDANCE_COOKIE_SECURE` | `true` for HTTPS deployments; `false` only for local HTTP development. |
| `VITE_MIN_FACE_LUMA` | Frontend **build-time** lighting threshold, default 45, valid range 10–180; rebuild to change it. |

The client samples brightness from the face area, with a central image area as fallback. In poor lighting it pauses inference, motion challenge progress and final photo capture. After brightness exceeds the configured threshold by 10 and remains stable for 500 ms, the **เริ่มตรวจต่อ** button becomes enabled. Inference stays paused until the user presses it; attendance then starts again at step 1. The same manual resume applies to enrollment. This local camera-quality check does not prove liveness, identity, or perfect photo quality. Pilot the threshold on representative employee phones and workplace lighting; exposure, skin tone, backlighting and camera behavior affect measurements. Invalid/out-of-range settings use the default 45.

For Compose, put `VITE_MIN_FACE_LUMA=45` in the root `.env`; Compose explicitly forwards it as a frontend build argument. For native Vite development/builds, export the variable in the invoking shell or set it in ignored `frontend/.env.local`. For the cloud container helper, pass `--min-face-luma 45` or export the variable; that helper does not read Compose's `.env`. Runtime container environment changes cannot alter an already compiled frontend.

### NVIDIA Linux

Install a compatible NVIDIA driver and [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html), then confirm `nvidia-smi`. The optional image uses CUDA 12.4/cuDNN runtime and `requirements-cuda.txt` with `onnxruntime-gpu`. Do not install CPU and GPU ORT distributions together; they share the same Python module.

```bash
docker compose -f docker-compose.yml -f docker-compose.cuda.yml up --build -d --wait
docker compose -f docker-compose.yml -f docker-compose.cuda.yml logs --tail=100 app
```

Compose requests a GPU and the backend executes model warmup; a missing NVIDIA runtime or inactive provider fails explicitly. See [ONNX Runtime CUDA requirements](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html). GPU image/hardware inference was not built or validated in the CPU-only cloud environment.

### Apple Silicon Mac mini

CUDA is for NVIDIA devices and does not use the Mac mini's Apple GPU. Linux Docker on macOS cannot expose Apple CoreML/GPU to this application. Run Python natively in an arm64 macOS virtual environment to use CoreML; the standard `onnxruntime` macOS package may provide `CoreMLExecutionProvider` depending on its version and build. Start with the ordinary CPU install and confirm inference first:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-dev.txt
python scripts/prepare_models.py
ATTENDANCE_INFERENCE_DEVICE=cpu python scripts/check_runtime.py --image face1.jpg
python -c 'import onnxruntime as ort; print(ort.get_available_providers())'
ATTENDANCE_INFERENCE_DEVICE=coreml python scripts/check_runtime.py --image face1.jpg
```

If the CoreML provider is absent, use CPU or install an official compatible ONNX Runtime build. Do not claim GPU support based only on a listed provider: the application requires provider activation and real model warmup. CoreML hardware performance and graph partitioning must be checked on the actual Mac; no Apple hardware is available in this cloud environment. See the official [CoreML provider documentation](https://onnxruntime.ai/docs/execution-providers/CoreML-ExecutionProvider.html).

## Validation limits

The client challenges provide modest client-reported motion checks. The server performs identity comparison, not video liveness/anti-spoof inference, and has no challenge-code endpoint. Browser tests with synthetic camera input do not validate real phone cameras or human motion thresholds. Keep first administrator setup behind your deployment access controls until the administrator creates the account.
