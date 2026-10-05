from dataclasses import dataclass
import os
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    database_url: str = "sqlite:///.local/attendance.db"
    model_dir: str = ".local/models"
    inference_device: str = "auto"
    face_threshold: float = 0.363
    cookie_secure: bool = False
    session_hours: int = 24
    max_image_bytes: int = 10 * 1024 * 1024
    cookie_name: str = "attendance_session"

    @classmethod
    def from_env(cls):
        settings = cls(
            database_url=os.getenv("ATTENDANCE_DATABASE_URL", cls.database_url),
            model_dir=os.getenv("ATTENDANCE_MODEL_DIR", cls.model_dir),
            inference_device=os.getenv("ATTENDANCE_INFERENCE_DEVICE", "auto").lower(),
            face_threshold=float(os.getenv("ATTENDANCE_FACE_THRESHOLD", "0.363")),
            cookie_secure=os.getenv("ATTENDANCE_COOKIE_SECURE", "false").lower() in {"1", "true", "yes"},
            session_hours=int(os.getenv("ATTENDANCE_SESSION_HOURS", "24")),
        )
        settings.validate()
        return settings

    def validate(self):
        if self.inference_device not in {"auto", "cpu", "cuda", "coreml"}:
            raise ValueError("ATTENDANCE_INFERENCE_DEVICE must be auto, cpu, cuda, or coreml")
        if not -1 < self.face_threshold < 1:
            raise ValueError("ATTENDANCE_FACE_THRESHOLD must be between -1 and 1")
        if self.session_hours <= 0:
            raise ValueError("ATTENDANCE_SESSION_HOURS must be positive")
        if self.database_url.startswith("postgresql://"):
            raise ValueError("Use postgresql+psycopg:// for ATTENDANCE_DATABASE_URL")
        if self.database_url.startswith("sqlite:///."):
            Path(self.database_url.removeprefix("sqlite:///")).parent.mkdir(parents=True, exist_ok=True)
