from fastapi.testclient import TestClient
from attendance.api import create_app
from attendance.config import Settings
from conftest import FakeFace


def test_built_frontend_spa_and_assets_preserve_api(tmp_path):
    dist=tmp_path/"dist";dist.mkdir()
    (dist/"index.html").write_text("<html>Attendance application</html>")
    (dist/"models").mkdir();(dist/"models"/"face.task").write_bytes(b"model-data")
    (dist/"assets").mkdir();(dist/"assets"/"app.js").write_text("const loaded=true")
    (tmp_path/"outside.txt").write_text("private-data")
    app=create_app(Settings(database_url=f"sqlite:///{tmp_path/'test.db'}"),FakeFace(),frontend_dir=dist)
    with TestClient(app) as client:
        assert "Attendance application" in client.get("/").text
        assert "Attendance application" in client.get("/attendance/history").text
        assert client.get("/models/face.task").content==b"model-data"
        assert client.get("/assets/app.js").text=="const loaded=true"
        for path in ["/api/missing","/models/missing.task","/assets/missing.js","/wasm/missing.wasm","/%2e%2e/outside.txt"]:
            response=client.get(path)
            assert response.status_code==404,(path,response.text)
            assert response.json()["error"]["code"]=="NOT_FOUND"
        assert client.get("/api/health").json()["models_ready"] is True
