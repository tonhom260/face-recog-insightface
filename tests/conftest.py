import numpy as np
import pytest
from fastapi.testclient import TestClient
from attendance.api import create_app
from attendance.config import Settings
from attendance.errors import APIError
from attendance.face import FaceService


class FakeFace:
    ready = True
    device = "cpu"
    providers = ["CPUExecutionProvider"]
    model_version = "test-sface"
    calls = 0

    def load(self):
        pass

    def extract(self, data):
        self.calls += 1
        if data == b"malformed":
            raise APIError(422, "INVALID_IMAGE", "Invalid image")
        if data == b"absent":
            raise APIError(422, "FACE_NOT_FOUND", "No usable face")
        if data == b"multiple":
            raise APIError(422, "MULTIPLE_FACES", "Multiple faces")
        result = np.zeros(128, dtype=np.float32)
        result[1 if data == b"wrong-person" else 0] = 1
        return result

    similarity = staticmethod(FaceService.similarity)


@pytest.fixture
def app(tmp_path):
    return create_app(Settings(database_url=f"sqlite:///{tmp_path / 'attendance.db'}"), FakeFace())


@pytest.fixture
def client(app):
    with TestClient(app) as client:
        yield client


@pytest.fixture
def admin(client):
    response = client.post("/api/setup", json={"employee_code": "ADM", "name": "Administrator", "password": "strong-password"})
    assert response.status_code == 201, response.text
    return response.json()["employee"]
