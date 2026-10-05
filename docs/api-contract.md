# Shared API contract

All application routes use `/api`. All timestamps are ISO 8601 UTC; the UI formats in Asia/Bangkok. JSON errors are `{ "error": { "code": "...", "message": "..." } }`. Use HTTP-only SameSite cookies for persisted authentication; production cookies must be secure. Vite proxies `/api` to FastAPI for same-origin development.

No backend liveness model or server-issued challenge-code flow is in scope. Do not create `/api/challenge` or require any pre-camera challenge acquisition.

## Types

Employee: `{ id: string, employee_code: string, name: string, role: "admin" | "employee", active: boolean, face_enrolled: boolean }`.

AttendanceEvent: `{ id: string, employee_id: string, action: "check_in" | "check_out", occurred_at: string, similarity: number, request_id: string, client_challenge_passed: boolean }`.

## Setup and login

- `GET /api/health` → `{ status: "ok" | "degraded", database_ready: boolean, models_ready: boolean }`.
  May include `inference_device` and `inference_providers` as diagnostic fields. Configure the requested device through `ATTENDANCE_INFERENCE_DEVICE=auto|cpu|cuda|coreml`; explicit CUDA/CoreML requires that provider to initialize and complete inference warmup. CoreML can partition unsupported operations to CPU. The frontend needs only the readiness booleans.
- `GET /api/setup` → `{ needs_setup: boolean }`.
- `POST /api/setup`, JSON `{ employee_code, name, password }` → `{ employee }`, creates the first administrator only and logs them in. Atomic; reject another setup once initialized. No default passwords or shipped employee accounts.
- `POST /api/auth/login`, JSON `{ employee_code, password }` → `{ employee }`, sets session cookie.
- `GET /api/auth/me` → `{ employee }`, HTTP 401 when unauthenticated.
- `POST /api/auth/logout` → HTTP 204, clears session.

## Employees and enrollment

- `GET /api/employees` (admin) → `{ employees: Employee[] }`.
- `POST /api/employees` (admin), JSON `{ employee_code, name, password }` → `{ employee }`, creates an employee account.
- `PATCH /api/employees/{id}` (admin), JSON fields `{ name?, active? }` → `{ employee }`.
- `POST /api/face/enroll`, multipart `image` plus optional `employee_id` (admin-only when another employee) → `{ employee, model_version: string }`.
  Require exactly one usable face. Employees may perform their first enrollment; replacing an existing enrollment requires an admin reset. Admins may enroll/re-enroll employees.
- `DELETE /api/employees/{id}/face` (admin) → HTTP 204, clears enrollment so the employee may enroll again.

## Attendance

- `GET /api/attendance/status` → `{ next_action: "check_in" | "check_out", last_event: AttendanceEvent | null }`.
- `POST /api/attendance`, multipart:
  - `image`: front-facing photograph captured after the client motion challenge.
  - `action`: `check_in` or `check_out`.
  - `client_challenge_passed`: `true`.
  - `challenge_actions`: JSON array of three distinct supported action names from `turn_left`, `turn_right`, `look_down`, `look_up`, `smile`, `blink`.
  - `request_id`: UUID created by the client for this submission and reused if the same submission is retried. Not a challenge/authentication code.
  → `{ event: AttendanceEvent, duplicate: boolean }`.
  The server authenticates the employee, validates the reported metadata, runs SFace comparison, enforces alternating attendance transitions, and records the event in a transaction. Retrying the same employee/request ID returns the existing event without another entry. New submissions return HTTP 201, duplicates HTTP 200. Missing face/enrollment or mismatch must never record an event. Never accept a client-provided similarity score or employee identity override.
- `GET /api/attendance?limit=50` → `{ events: AttendanceEvent[] }`, newest first, authenticated employee only. Admin may pass `employee_id` to filter another employee.

## Important status codes

- 401: no valid session / invalid login.
- 403: inactive account or insufficient permissions.
- 409: setup already completed, duplicate employee code, invalid attendance transition, or enrollment replacement requires admin reset.
- 422: invalid fields/image, absent/multiple faces, enrollment missing, or face mismatch.
- 503: required face models unavailable.

Use stable error codes such as `FACE_MISMATCH`, `FACE_NOT_FOUND`, `MULTIPLE_FACES`, `FACE_NOT_ENROLLED`, `MODELS_UNAVAILABLE`, `INVALID_TRANSITION`. The frontend displays helpful Thai messages without internal system diagnostics.

## Models and local development

Agree model asset filenames through `docs/model-licenses.md` and scripts. Prefer `ATTENDANCE_MODEL_DIR` (default `.local/models`) for YuNet/SFace and `frontend/public/models/face_landmarker.task` for the client asset. MediaPipe WASM is served from the installed npm package through the frontend build, not an unpinned third-party CDN. Frontend API responses and field names must follow this contract; proposed changes go through the coordinator before other agents depend on them.
