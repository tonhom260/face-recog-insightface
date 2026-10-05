from contextlib import asynccontextmanager
from datetime import timezone
import json
import platform
from pathlib import Path
from urllib.parse import urlsplit
from uuid import UUID

import numpy as np
from fastapi import Depends, FastAPI, File, Form, Query, Request, Response, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, ConfigDict, Field, field_validator
from starlette.exceptions import HTTPException
from fastapi.staticfiles import StaticFiles
from sqlalchemy import delete, select, text
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from sqlalchemy.orm import Session

from .config import Settings
from .db import AttendanceEvent, AuthSession, Database, Employee, SetupState
from .errors import APIError
from .face import FaceService
from .security import hash_password, is_expired, issue_session, token_hash, verify_password

ACTIONS = {"turn_left", "turn_right", "look_down", "look_up", "smile", "blink"}


class AccountInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    employee_code: str = Field(min_length=1, max_length=64)
    name: str = Field(min_length=1, max_length=200)
    password: str = Field(min_length=8, max_length=128)

    @field_validator("employee_code", "name")
    @classmethod
    def trim_nonempty(cls, value):
        value = value.strip()
        if not value:
            raise ValueError("Must not be blank")
        return value


class LoginInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    employee_code: str = Field(min_length=1, max_length=64)
    password: str = Field(min_length=1, max_length=128)


class EmployeePatch(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str | None = Field(default=None, min_length=1, max_length=200)
    active: bool | None = None

    @field_validator("name")
    @classmethod
    def name_nonempty(cls, value):
        if value is not None and not value.strip():
            raise ValueError("Must not be blank")
        return value.strip() if value else value


def employee_json(employee):
    return {"id": employee.id, "employee_code": employee.employee_code, "name": employee.name,
            "role": employee.role, "active": employee.active, "face_enrolled": employee.embedding is not None}


def event_json(event):
    timestamp = event.occurred_at
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=timezone.utc)
    return {"id": event.id, "employee_id": event.employee_id, "action": event.action,
            "occurred_at": timestamp.astimezone(timezone.utc).isoformat().replace("+00:00", "Z"),
            "similarity": event.similarity, "request_id": event.request_id,
            "client_challenge_passed": event.client_challenge_passed}


def create_app(settings=None, face_service=None, frontend_dir=None):
    settings = settings or Settings.from_env()
    settings.validate()
    database = Database(settings.database_url)
    face = face_service or FaceService(settings)

    @asynccontextmanager
    async def lifespan(app):
        database.initialize()
        face.load()
        yield
        database.engine.dispose()

    app = FastAPI(title="Employee attendance", lifespan=lifespan)
    app.state.database = database
    app.state.face_service = face
    app.state.settings = settings

    @app.exception_handler(APIError)
    async def api_error(_, exc):
        return JSONResponse(status_code=exc.status, content={"error": {"code": exc.code, "message": exc.message}})

    @app.exception_handler(HTTPException)
    async def http_error(_, exc):
        code = "NOT_FOUND" if exc.status_code == 404 else "HTTP_ERROR"
        return JSONResponse(status_code=exc.status_code, content={"error": {"code": code, "message": str(exc.detail)}})

    @app.exception_handler(RequestValidationError)
    async def validation_error(_, exc):
        # Never echo submitted password, image bytes, or internal details.
        return JSONResponse(status_code=422, content={"error": {"code": "INVALID_FIELDS", "message": "Please check the supplied fields"}})

    @app.middleware("http")
    async def same_origin(request, call_next):
        if request.method in {"POST", "PATCH", "DELETE", "PUT"} and request.url.path.startswith("/api/"):
            origin = request.headers.get("origin")
            referer = request.headers.get("referer")
            candidate = origin or referer
            if candidate:
                parsed = urlsplit(candidate)
                if (parsed.scheme, parsed.netloc) != (request.url.scheme, request.url.netloc):
                    return JSONResponse(status_code=403, content={"error": {"code": "INVALID_ORIGIN", "message": "Request origin is not allowed"}})
            if request.headers.get("sec-fetch-site") == "cross-site":
                return JSONResponse(status_code=403, content={"error": {"code": "INVALID_ORIGIN", "message": "Request origin is not allowed"}})
        response = await call_next(request)
        if request.url.path.startswith("/api/"):
            response.headers["Cache-Control"] = "no-store"
        response.headers["X-Content-Type-Options"] = "nosniff"
        return response

    def db_session(request: Request):
        with database.sessions() as db:
            # SQLite has no row locks. Serialize writes before authentication reads;
            # the deployment PostgreSQL path locks the employee row instead.
            if database.sqlite and request.method in {"POST", "PATCH", "DELETE", "PUT"}:
                db.connection().exec_driver_sql("BEGIN IMMEDIATE")
            try:
                yield db
            finally:
                db.rollback()

    def authenticated(request: Request, db: Session = Depends(db_session)):
        token = request.cookies.get(settings.cookie_name)
        session = db.get(AuthSession, token_hash(token)) if token else None
        if session is None or is_expired(session.expires_at):
            raise APIError(401, "UNAUTHENTICATED", "Please sign in")
        query = select(Employee).where(Employee.id == session.employee_id)
        if request.method in {"POST", "PATCH", "DELETE", "PUT"}:
            query = query.with_for_update()
        employee = db.scalar(query)
        if employee is None:
            raise APIError(401, "UNAUTHENTICATED", "Please sign in")
        if not employee.active:
            raise APIError(403, "ACCOUNT_INACTIVE", "This account is inactive")
        return employee

    def admin(employee: Employee = Depends(authenticated)):
        if employee.role != "admin":
            raise APIError(403, "FORBIDDEN", "Administrator access required")
        return employee

    def find_employee(db, employee_id, lock=False):
        query = select(Employee).where(Employee.id == employee_id)
        if lock:
            query = query.with_for_update()
        employee = db.scalar(query)
        if employee is None:
            raise APIError(404, "EMPLOYEE_NOT_FOUND", "Employee not found")
        return employee

    def latest(db, employee_id):
        return db.scalar(select(AttendanceEvent).where(AttendanceEvent.employee_id == employee_id)
                         .order_by(AttendanceEvent.occurred_at.desc(), AttendanceEvent.id.desc()).limit(1))

    def next_action(event):
        return "check_out" if event and event.action == "check_in" else "check_in"

    def read_image(image):
        data = image.file.read(settings.max_image_bytes + 1)
        if len(data) > settings.max_image_bytes:
            raise APIError(422, "INVALID_IMAGE", "Image exceeds the upload limit")
        return data

    @app.get("/api/health")
    def health():
        try:
            with database.engine.connect() as connection:
                connection.execute(text("SELECT 1"))
            database_ready = True
        except SQLAlchemyError:
            database_ready = False
        ready = database_ready and face.ready
        return {"status": "ok" if ready else "degraded", "database_ready": database_ready,
                "models_ready": face.ready, "inference_device": face.device,
                "inference_providers": face.providers}

    @app.get("/api/setup")
    def setup_status(db: Session = Depends(db_session)):
        return {"needs_setup": db.get(SetupState, 1) is None}

    @app.post("/api/setup", status_code=201)
    def setup(body: AccountInput, response: Response, db: Session = Depends(db_session)):
        if db.get(SetupState, 1):
            raise APIError(409, "SETUP_COMPLETED", "Administrator setup already completed")
        employee = Employee(employee_code=body.employee_code, name=body.name,
                            password_hash=hash_password(body.password), role="admin")
        try:
            # Flush singleton first so parallel PostgreSQL attempts serialize on it.
            db.add(SetupState(id=1))
            db.flush()
            db.add(employee)
            db.flush()
            issue_session(db, employee, settings, response)
            db.commit()
        except IntegrityError as exc:
            db.rollback()
            raise APIError(409, "SETUP_COMPLETED", "Administrator setup already completed") from exc
        return {"employee": employee_json(employee)}

    @app.post("/api/auth/login")
    def login(body: LoginInput, response: Response, request: Request, db: Session = Depends(db_session)):
        employee = db.scalar(select(Employee).where(Employee.employee_code == body.employee_code.strip()))
        if not verify_password(body.password, employee.password_hash if employee else None):
            raise APIError(401, "INVALID_LOGIN", "Invalid employee code or password")
        if not employee.active:
            raise APIError(403, "ACCOUNT_INACTIVE", "This account is inactive")
        previous = request.cookies.get(settings.cookie_name)
        if previous:
            db.execute(delete(AuthSession).where(AuthSession.token_hash == token_hash(previous)))
        issue_session(db, employee, settings, response)
        db.commit()
        return {"employee": employee_json(employee)}

    @app.get("/api/auth/me")
    def me(employee: Employee = Depends(authenticated)):
        return {"employee": employee_json(employee)}

    @app.post("/api/auth/logout", status_code=204)
    def logout(request: Request, response: Response, db: Session = Depends(db_session)):
        token = request.cookies.get(settings.cookie_name)
        if token:
            db.execute(delete(AuthSession).where(AuthSession.token_hash == token_hash(token)))
            db.commit()
        response.delete_cookie(settings.cookie_name, path="/", httponly=True,
                               secure=settings.cookie_secure, samesite="lax")

    @app.get("/api/employees")
    def employees(_: Employee = Depends(admin), db: Session = Depends(db_session)):
        rows = db.scalars(select(Employee).order_by(Employee.employee_code)).all()
        return {"employees": [employee_json(row) for row in rows]}

    @app.post("/api/employees", status_code=201)
    def create_employee(body: AccountInput, _: Employee = Depends(admin), db: Session = Depends(db_session)):
        employee = Employee(employee_code=body.employee_code, name=body.name,
                            password_hash=hash_password(body.password), role="employee")
        try:
            db.add(employee)
            db.commit()
        except IntegrityError as exc:
            db.rollback()
            raise APIError(409, "DUPLICATE_EMPLOYEE_CODE", "Employee code already exists") from exc
        return {"employee": employee_json(employee)}

    @app.patch("/api/employees/{employee_id}")
    def update_employee(employee_id: str, body: EmployeePatch, current: Employee = Depends(admin), db: Session = Depends(db_session)):
        employee = find_employee(db, employee_id, lock=True)
        if body.active is False and employee.id == current.id:
            raise APIError(409, "ADMIN_SELF_DISABLE", "An administrator cannot disable their own account")
        if body.name is not None:
            employee.name = body.name
        if body.active is not None:
            employee.active = body.active
            if not body.active:
                db.execute(delete(AuthSession).where(AuthSession.employee_id == employee.id))
        db.commit()
        return {"employee": employee_json(employee)}

    @app.post("/api/face/enroll")
    def enroll(image: UploadFile = File(...), employee_id: str | None = Form(None),
               current: Employee = Depends(authenticated), db: Session = Depends(db_session)):
        target_id = employee_id or current.id
        if target_id != current.id and current.role != "admin":
            raise APIError(403, "FORBIDDEN", "Administrator access required")
        target = current if target_id == current.id else find_employee(db, target_id, lock=True)
        if not target.active:
            raise APIError(403, "ACCOUNT_INACTIVE", "This account is inactive")
        if target.embedding is not None and current.role != "admin":
            raise APIError(409, "ENROLLMENT_EXISTS", "Ask an administrator to reset face enrollment")
        embedding = face.extract(read_image(image))
        target.embedding = np.asarray(embedding, dtype="<f4").tobytes()
        target.model_version = face.model_version
        db.commit()
        return {"employee": employee_json(target), "model_version": face.model_version}

    @app.delete("/api/employees/{employee_id}/face", status_code=204)
    def reset_face(employee_id: str, _: Employee = Depends(admin), db: Session = Depends(db_session)):
        target = find_employee(db, employee_id, lock=True)
        target.embedding = target.model_version = None
        db.commit()

    @app.get("/api/attendance/status")
    def status(current: Employee = Depends(authenticated), db: Session = Depends(db_session)):
        event = latest(db, current.id)
        return {"next_action": next_action(event), "last_event": event_json(event) if event else None}

    @app.post("/api/attendance", status_code=201)
    def attendance(response: Response, request: Request, image: UploadFile = File(...), action: str = Form(...),
                   client_challenge_passed: str = Form(...), challenge_actions: str = Form(...), request_id: str = Form(...),
                   current: Employee = Depends(authenticated), db: Session = Depends(db_session)):
        try:
            request_id = str(UUID(request_id))
            challenges = json.loads(challenge_actions)
            if action not in {"check_in", "check_out"} or client_challenge_passed != "true":
                raise ValueError()
            if not isinstance(challenges, list) or len(challenges) != 3 or any(not isinstance(a, str) for a in challenges):
                raise ValueError()
            if len(set(challenges)) != 3 or not set(challenges) <= ACTIONS:
                raise ValueError()
        except (ValueError, TypeError, json.JSONDecodeError) as exc:
            raise APIError(422, "INVALID_CHALLENGE", "Report three distinct completed motion actions and a valid request ID") from exc
        # Reject identity/score overrides rather than accepting untrusted values.
        form = request._form
        if form is not None and set(form.keys()) - {"image", "action", "client_challenge_passed", "challenge_actions", "request_id"}:
            raise APIError(422, "INVALID_FIELDS", "Unexpected attendance fields")
        existing = db.scalar(select(AttendanceEvent).where(AttendanceEvent.employee_id == current.id,
                                                          AttendanceEvent.request_id == request_id))
        if existing:
            if existing.action != action or existing.challenge_actions != challenges:
                raise APIError(409, "REQUEST_ID_CONFLICT", "Request ID already belongs to a different submission")
            response.status_code = 200
            return {"event": event_json(existing), "duplicate": True}
        if current.embedding is None or current.model_version != face.model_version:
            raise APIError(422, "FACE_NOT_ENROLLED", "Face enrollment is required for the current model")
        if action != next_action(latest(db, current.id)):
            raise APIError(409, "INVALID_TRANSITION", "Attendance must alternate check-in and check-out")
        embedding = face.extract(read_image(image))
        enrolled = np.frombuffer(current.embedding, dtype="<f4")
        similarity = face.similarity(enrolled, embedding)
        if similarity < settings.face_threshold:
            raise APIError(422, "FACE_MISMATCH", "Face does not match the enrolled employee")
        event = AttendanceEvent(employee_id=current.id, action=action, similarity=similarity,
                                request_id=request_id, client_challenge_passed=True, challenge_actions=challenges)
        db.add(event)
        db.commit()
        return {"event": event_json(event), "duplicate": False}

    @app.get("/api/attendance")
    def history(limit: int = Query(50, ge=1, le=200), employee_id: str | None = None,
                current: Employee = Depends(authenticated), db: Session = Depends(db_session)):
        target_id = employee_id or current.id
        if target_id != current.id and current.role != "admin":
            raise APIError(403, "FORBIDDEN", "Administrator access required")
        find_employee(db, target_id)
        events = db.scalars(select(AttendanceEvent).where(AttendanceEvent.employee_id == target_id)
                           .order_by(AttendanceEvent.occurred_at.desc(), AttendanceEvent.id.desc()).limit(limit)).all()
        return {"events": [event_json(event) for event in events]}

    @app.get("/test")
    def legacy_test(name: str | None = None, age: int | None = None):
        return {"status": "ok", "message": "FastAPI is running correctly!",
                "system_info": {"platform": platform.system(), "processor": platform.processor(), "node": platform.node()}}

    @app.post("/send_img")
    def legacy_image(upload: UploadFile = File(...), name: str | None = Form(None), age: int | None = Form(None)):
        data = read_image(upload)
        return {"status": "ok", "message": "File received", "file_info": {
            "filename": upload.filename, "content_type": upload.content_type, "file_size": len(data)},
            "system_info": {"platform": platform.system(), "processor": platform.processor(), "node": platform.node()}}

    # The deployed image includes the compiled React application. Vite serves it
    # separately in development. Resolve paths before serving any static file.
    frontend = (Path(frontend_dir) if frontend_dir is not None else Path(__file__).resolve().parents[1] / "frontend" / "dist").resolve()
    if (frontend / "index.html").is_file():
        if (frontend / "assets").is_dir():
            app.mount("/assets", StaticFiles(directory=frontend / "assets"), name="assets")

        @app.get("/{path:path}", include_in_schema=False)
        def frontend_page(path: str):
            if path == "api" or path.startswith("api/"):
                raise APIError(404, "NOT_FOUND", "Route not found")
            target = (frontend / path).resolve()
            if frontend.resolve() not in target.parents and target != frontend.resolve():
                raise APIError(404, "NOT_FOUND", "Route not found")
            if target.is_file():
                return FileResponse(target)
            if path.split("/", 1)[0] in {"assets", "models", "wasm"} or target.suffix:
                raise APIError(404, "NOT_FOUND", "Static file not found")
            return FileResponse(frontend / "index.html")

    return app
