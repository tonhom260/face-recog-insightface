from datetime import datetime, timezone
import uuid

from sqlalchemy import Boolean, DateTime, Float, ForeignKey, Integer, JSON, LargeBinary, String, UniqueConstraint, create_engine, event
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker
from sqlalchemy.pool import StaticPool


def utcnow():
    return datetime.now(timezone.utc)


def new_id():
    return str(uuid.uuid4())


class Base(DeclarativeBase):
    pass


class Employee(Base):
    __tablename__ = "employees"
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    employee_code: Mapped[str] = mapped_column(String(64), unique=True, nullable=False)
    name: Mapped[str] = mapped_column(String(200))
    role: Mapped[str] = mapped_column(String(16), default="employee")
    active: Mapped[bool] = mapped_column(Boolean, default=True)
    password_hash: Mapped[str] = mapped_column(String(255))
    embedding: Mapped[bytes | None] = mapped_column(LargeBinary, nullable=True)
    model_version: Mapped[str | None] = mapped_column(String(100), nullable=True)


class SetupState(Base):
    __tablename__ = "setup_state"
    # All setup attempts insert the same primary key; DB uniqueness is the guard.
    id: Mapped[int] = mapped_column(Integer, primary_key=True)


class AuthSession(Base):
    __tablename__ = "auth_sessions"
    token_hash: Mapped[str] = mapped_column(String(64), primary_key=True)
    employee_id: Mapped[str] = mapped_column(ForeignKey("employees.id"), index=True)
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)


class AttendanceEvent(Base):
    __tablename__ = "attendance_events"
    __table_args__ = (UniqueConstraint("employee_id", "request_id", name="uq_attendance_retry"),)
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    employee_id: Mapped[str] = mapped_column(ForeignKey("employees.id"), index=True)
    action: Mapped[str] = mapped_column(String(16))
    occurred_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, index=True)
    similarity: Mapped[float] = mapped_column(Float)
    request_id: Mapped[str] = mapped_column(String(36))
    client_challenge_passed: Mapped[bool] = mapped_column(Boolean)
    challenge_actions: Mapped[list[str]] = mapped_column(JSON)


class Database:
    def __init__(self, url):
        options = {"pool_pre_ping": True}
        self.sqlite = url.startswith("sqlite")
        if self.sqlite:
            options["connect_args"] = {"check_same_thread": False, "timeout": 30}
            if ":memory:" in url:
                options["poolclass"] = StaticPool
        self.engine = create_engine(url, **options)
        if self.sqlite:
            @event.listens_for(self.engine, "connect")
            def configure_sqlite(connection, _):
                connection.execute("PRAGMA foreign_keys=ON")
                connection.execute("PRAGMA busy_timeout=30000")
        self.sessions = sessionmaker(self.engine, expire_on_commit=False)

    def initialize(self):
        Base.metadata.create_all(self.engine)
