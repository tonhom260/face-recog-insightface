# Mobile attendance implementation

Branch: `codex/mobile-attendance`, based on `b06a935`.

## Agreed product scope

- Commercial employee attendance, with ordinary employee users and a deliberately modest fraud-resistance requirement.
- React + TypeScript, designed first for phones, then tablets, then desktop.
- MediaPipe Face Landmarker runs on the client. Randomly choose three distinct actions per attempt from `turn_left`, `turn_right`, `look_down`, `look_up`, `smile`, `blink`. Present one action at a time and check neutral → action → neutral over consecutive frames with a timeout. Reset on face loss.
- React checks camera luminance before motion inference and capture. Low light must show a Thai warning, halt motion/capture, and clear progress. Keep the camera only to monitor recovery. After adequate lighting remains stable, enable **เริ่มตรวจต่อ**; remain paused until the user explicitly presses it, then restart calibration/challenge from step 1. Apply the gate to enrollment too. Sample the face region when available, otherwise a central region; use hysteresis and configurable client thresholds that can be tuned on actual employee phones.
- After the client challenge passes, capture a front-facing photograph. FastAPI runs YuNet + SFace and compares against the authenticated employee's enrollment, 1:1.
- Configure inference using `ATTENDANCE_INFERENCE_DEVICE=auto|cpu|cuda|coreml`. Explicit CUDA/CoreML must fail clearly if the requested execution provider cannot initialize and run; auto may fall back to CPU. CUDA is for NVIDIA; CoreML requires native macOS, with supported partitions using Apple compute units and other partitions possibly using CPU. Linux Docker on Mac cannot use Apple CoreML. YuNet detection and alignment remain CPU. Test actual CPU inference and device selection; do not claim NVIDIA or Apple hardware validation in this CPU environment.
- The backend records the challenge as a client-reported result. It does not perform video liveness/anti-spoofing inference. The UI must not claim certified liveness or guaranteed spoof resistance.
- **No server challenge-code endpoint, nonce acquisition, or pre-camera verification-code request.** The client starts and randomizes its own motion challenge. A client-generated attendance request ID is only for retry deduplication, not authentication or a server-issued challenge.
- Use server timestamps, display Asia/Bangkok time, and prevent duplicate submissions and invalid check-in/check-out transitions.
- Retain existing `/test` and `/send_img` compatibility where practical.

## Work ownership

1. Backend agent: `attendance/`, `main.py`, Python requirements, legacy Python CLI scripts, and `tests/` (excluding browser E2E).
2. Frontend agent: `frontend/`, including React screens, CSS, camera/MediaPipe integration, challenge state machine, and frontend unit tests.
3. Models/deployment agent: `scripts/`, Docker/Compose files, `.devcontainer/`, `.gitignore`, `.env.example`, and `docs/development.md`, `docs/model-licenses.md`.
4. QA agent: `e2e/`, independent integration/browser checks, and review findings; do not edit files owned by others without coordination.
5. Coordinator: API contract, root README, integration fixes after owners report completion, final verification, and review.

All agents share the checkout and branch. Do not create worktrees, switch branches, commit, push, or overwrite another agent's files. Use GPT-6.1 sol with high reasoning as requested.

## Delivery and validation

- Actual SFace inference on included images and enrollment/comparison via the API.
- Compare genuine and different-person fixtures; existing images alone do not establish accuracy parity with InsightFace. Keep the model/version and threshold configurable and require new enrollment when moving from buffalo_l to SFace.
- Authentication/permissions, malformed images, missing/disabled employees, missing enrollment, wrong-person rejection, duplicate retry, invalid attendance transitions, and Bangkok time handling.
- Challenge state-machine tests: random unique actions, ordering, neutral transitions, duration, timeout, face loss, and front-camera mirroring.
- Low-light tests: dark input halts progress/capture, boundary hysteresis prevents flicker, adequate stable light enables manual resume but does not resume inference by itself, explicit resume restarts step 1, and no frame is sent merely for brightness checking. Browser dark/bright camera checks are explicitly simulated.
- Frontend build/type check and browser layout checks at phone, tablet, desktop sizes; test setup/login/enrollment/attendance/history/admin workflows.
- Browser simulated-camera tests are labeled simulated; actual camera handling, false rejects and motion thresholds require a pilot on real employee phones.
- PostgreSQL is the deployment database; lightweight test database support may be used for isolated tests and documented development fallback.
- Download model assets using verified HTTPS and authoritative SHA256/LFS digests. Do not commit model binaries or real employee biometric records, and do not disable verification.
