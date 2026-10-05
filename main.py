"""ASGI entry point: uvicorn main:app."""
from attendance.api import create_app

app = create_app()
