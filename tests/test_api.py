from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
import json
import uuid

from fastapi.testclient import TestClient
import pytest
from sqlalchemy import select

from attendance.db import AttendanceEvent, AuthSession, Employee, utcnow


def enroll(client, data=b"right-person", **fields):
    return client.post("/api/face/enroll", data=fields, files={"image": ("face.jpg", data, "image/jpeg")})


def submit(client, request_id=None, action="check_in", data=b"right-person", **overrides):
    fields = {"action": action, "client_challenge_passed": "true", "challenge_actions": json.dumps(["blink", "smile", "turn_left"]),
              "request_id": request_id or str(uuid.uuid4())}
    fields.update(overrides)
    return client.post("/api/attendance", data=fields, files={"image": ("face.jpg", data, "image/jpeg")})


def create_employee(client, code="EMP"):
    r = client.post("/api/employees", json={"employee_code": code, "name": "Employee", "password": "employee-password"})
    assert r.status_code == 201, r.text
    return r.json()["employee"]


def login_employee(client, code="EMP"):
    assert client.post("/api/auth/login", json={"employee_code": code, "password": "employee-password"}).status_code == 200


def test_bootstrap_cookie_and_auth(client, admin, app):
    assert client.get("/api/setup").json() == {"needs_setup": False}
    assert client.get("/api/auth/me").json()["employee"] == admin
    assert client.post("/api/setup", json={"employee_code":"X", "name":"X", "password":"12345678"}).status_code == 409
    token = client.cookies.get("attendance_session")
    with app.state.database.sessions() as db:
        assert db.get(Employee, admin["id"]).password_hash.startswith("$argon2id$")
        assert db.get(AuthSession, token) is None
        assert len(db.scalars(select(AuthSession)).all()) == 1
    response = client.post("/api/auth/login", json={"employee_code":"ADM", "password":"strong-password"})
    cookie = response.headers["set-cookie"].lower()
    assert "httponly" in cookie and "samesite=lax" in cookie
    assert client.post("/api/auth/logout").status_code == 204
    assert client.get("/api/auth/me").status_code == 401
    assert client.post("/api/auth/login",json={"employee_code":"ADM","password":"wrong"}).status_code == 401


def test_no_server_challenge_and_legacy(client):
    assert client.get("/api/challenge").status_code == 404
    assert client.get("/test").json()["status"] == "ok"
    assert client.post("/send_img", files={"upload":("x.jpg",b"data")}).json()["file_info"]["file_size"] == 4


def test_employee_permissions_enrollment_reset(client, admin):
    employee = create_employee(client)
    assert client.post("/api/employees",json={"employee_code":"EMP","name":"Duplicate","password":"12345678"}).status_code == 409
    login_employee(client)
    assert client.get("/api/employees").status_code == 403
    assert enroll(client, employee_id=admin["id"]).status_code == 403
    assert enroll(client).json()["employee"]["face_enrolled"]
    assert enroll(client).status_code == 409
    assert client.delete(f"/api/employees/{employee['id']}/face").status_code == 403
    assert client.get("/api/attendance",params={"employee_id":admin["id"]}).status_code == 403
    client.post("/api/auth/login",json={"employee_code":"ADM","password":"strong-password"})
    assert client.delete(f"/api/employees/{employee['id']}/face").status_code == 204
    assert not client.get("/api/employees").json()["employees"][1]["face_enrolled"]
    assert client.patch(f"/api/employees/{employee['id']}",json={"active":False}).status_code == 200
    assert client.post("/api/auth/login",json={"employee_code":"EMP","password":"employee-password"}).status_code == 403


def test_history_transitions_dedupe_server_time(client, admin, app):
    assert client.get("/api/attendance/status").json() == {"next_action":"check_in","last_event":None}
    assert submit(client).json()["error"]["code"] == "FACE_NOT_ENROLLED"
    assert enroll(client).status_code == 200
    assert submit(client, action="check_out").status_code == 409
    request_id = str(uuid.uuid4())
    first = submit(client,request_id)
    assert first.status_code == 201
    event = first.json()["event"]
    assert event["employee_id"] == admin["id"] and event["similarity"] == 1
    assert event["occurred_at"].endswith("Z")
    assert abs((datetime.fromisoformat(event["occurred_at"].replace("Z","+00:00")) - utcnow()).total_seconds()) < 5
    calls = app.state.face_service.calls
    retry = submit(client, request_id)
    assert retry.status_code == 200 and retry.json() == {"event":event,"duplicate":True}
    assert app.state.face_service.calls == calls
    assert submit(client, request_id, action="check_out").status_code == 409
    assert submit(client).json()["error"]["code"] == "INVALID_TRANSITION"
    assert submit(client,action="check_out").status_code == 201
    history = client.get("/api/attendance").json()["events"]
    assert [e["action"] for e in history] == ["check_out","check_in"]
    assert client.get("/api/attendance/status").json()["next_action"] == "check_in"


@pytest.mark.parametrize("data,code",[(b"wrong-person","FACE_MISMATCH"),(b"malformed","INVALID_IMAGE"),(b"absent","FACE_NOT_FOUND"),(b"multiple","MULTIPLE_FACES")])
def test_rejected_faces_never_record(client,admin,data,code):
    assert enroll(client).status_code == 200
    response=submit(client,data=data)
    assert response.status_code == 422 and response.json()["error"]["code"] == code
    assert client.get("/api/attendance").json()["events"] == []


@pytest.mark.parametrize("override",[{"client_challenge_passed":"false"},{"challenge_actions":"[]"},{"challenge_actions":'["smile","smile","blink"]'},{"challenge_actions":'["smile","blink","unknown"]'},{"challenge_actions":"not-json"},{"request_id":"bad"},{"employee_id":"another"},{"similarity":"1"}])
def test_untrusted_fields(client,admin,override):
    assert enroll(client).status_code == 200
    assert submit(client,**override).status_code == 422
    assert client.get("/api/attendance").json()["events"] == []


def test_session_expiry_and_csrf(client,admin,app):
    response=client.post("/api/auth/logout",headers={"Origin":"https://evil.invalid"})
    assert response.status_code == 403
    assert client.get("/api/auth/me").status_code == 200
    with app.state.database.sessions() as db:
        session=db.scalar(select(AuthSession));session.expires_at=utcnow()-timedelta(seconds=1);db.commit()
    assert client.get("/api/auth/me").status_code == 401


def test_parallel_retries_and_transitions(client,admin,app):
    enroll(client)
    token=client.cookies.get("attendance_session")
    request_id=str(uuid.uuid4())
    def attempt(rid):
        with TestClient(app) as worker:
            worker.cookies.set("attendance_session",token)
            return submit(worker,rid).status_code
    with ThreadPoolExecutor(max_workers=2) as pool:
        results=list(pool.map(attempt,[request_id,request_id]))
    assert sorted(results)==[200,201]
    assert len(client.get("/api/attendance").json()["events"])==1
    assert submit(client,action="check_out").status_code==201
    with ThreadPoolExecutor(max_workers=2) as pool:
        results=list(pool.map(attempt,[str(uuid.uuid4()),str(uuid.uuid4())]))
    assert sorted(results)==[201,409]
    assert len(client.get("/api/attendance").json()["events"])==3


def test_setup_is_atomic_under_parallel_requests(client,app):
    def bootstrap(code):
        with TestClient(app) as worker:
            return worker.post("/api/setup",json={"employee_code":code,"name":code,"password":"12345678"}).status_code
    with ThreadPoolExecutor(max_workers=2) as pool:
        statuses=list(pool.map(bootstrap,["FIRST","SECOND"]))
    assert sorted(statuses)==[201,409]
    with app.state.database.sessions() as db:
        assert len(db.scalars(select(Employee)).all())==1
        assert len(db.scalars(select(AuthSession)).all())==1
