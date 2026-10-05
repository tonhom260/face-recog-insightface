# Independent verification

These scripts complement backend and frontend unit tests. Model files must already be downloaded by `scripts/prepare_models.py`. The API and browser runners create fresh temporary SQLite databases, random account credentials, and owned processes, then stop those processes and remove temporary databases. Never point standalone workflow checks at an employee or production database.

Create the public-domain different-person fixture outside the checkout, using an environment with scikit-image and OpenCV installed:

```bash
/workspace/.venvs/face-recog-insightface/bin/python - <<'PY'
from skimage import data
import cv2
cv2.imwrite('/tmp/attendance-astronaut.jpg', cv2.cvtColor(data.astronaut(), cv2.COLOR_RGB2BGR))
PY
```

Run the actual YuNet/SFace API checks and OpenCV preprocessing reference comparison:

```bash
PYTHONPATH=. /workspace/.venvs/attendance/bin/python e2e/run_api_qa.py \
  --different-person /tmp/attendance-astronaut.jpg --report /tmp/attendance-api-qa.json
PYTHONPATH=. /workspace/.venvs/attendance/bin/python e2e/model_real_inference.py \
  --different-person /tmp/attendance-astronaut.jpg --report /tmp/attendance-model-qa.json
```

The different-person input must depict a known different individual. Altering a photograph of the enrolled person does not create a valid different-person negative. These three fixtures do not measure population accuracy.

These browser helpers target Linux with `/usr/bin/chromium` and ffmpeg installed. Their Python environment needs Playwright. The backend's ordinary dependency set does not include it. To add optional browser tooling to a development/test environment, run `python -m pip install -r requirements-e2e.txt`; installation of Chromium and ffmpeg is separate.

The checks in this report used the cloud primary Python at `/opt/codex/runtimes/codex-primary-runtime/dependencies/python/bin/python`, which already contains Playwright. That is distinct from `/workspace/.venvs/attendance/bin/python` used to start the API. If the backend virtual environment is activated, explicitly use the primary Python path below or install the optional E2E dependencies first. After `npm ci` in `frontend/`, run:

```bash
/opt/codex/runtimes/codex-primary-runtime/dependencies/python/bin/python e2e/browser_models.py --report /tmp/attendance-browser-model-qa.json
/opt/codex/runtimes/codex-primary-runtime/dependencies/python/bin/python e2e/run_browser_qa.py --backend-python /workspace/.venvs/attendance/bin/python \
  --artifacts /tmp/attendance-ui-qa --report /tmp/attendance-ui-qa.json
```

The browser runner requires ports 8000 and 5173 to be free. It leaves existing services alone. Its screenshots and reports are outside the checkout. It checks 390×844, 768×1024, and 1440×900 layouts and exercises setup, authentication, enrollment, administration, history, permission denial, errors, and camera cleanup.

Camera workflows use an explicitly **simulated static video** derived from `face1.jpg`. Bright enrollment uses actual MediaPipe inference. Black/dim frames exercise the client lighting gate. Bright recovery keeps inference and capture paused until the user explicitly clicks the resume button, then attendance starts a fresh three-action challenge. Successful attendance motion uses **test-only synthetic landmarks/pose/blendshapes injected into the browser process**, while the production randomized reducer and actual server SFace comparison remain active. Production source contains no test bypass. The checks do not establish real phone camera handling, physical employee motion thresholds, or spoof resistance; validate those during a real-device pilot.
