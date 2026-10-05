from datetime import timedelta, timezone
import hashlib
import secrets

from argon2 import PasswordHasher
from argon2.exceptions import InvalidHashError, VerificationError
from sqlalchemy import delete

from .db import AuthSession, utcnow

hasher = PasswordHasher()
# Used only to equalize absent-account password work; it is never an account.
_dummy_hash = hasher.hash(secrets.token_urlsafe(32))


def hash_password(password):
    return hasher.hash(password)


def verify_password(password, stored_hash):
    try:
        return hasher.verify(stored_hash or _dummy_hash, password) and bool(stored_hash)
    except (VerificationError, InvalidHashError):
        return False


def token_hash(token):
    return hashlib.sha256(token.encode()).hexdigest()


def issue_session(db, employee, settings, response):
    token = secrets.token_urlsafe(32)
    db.execute(delete(AuthSession).where(AuthSession.expires_at <= utcnow()))
    db.add(AuthSession(token_hash=token_hash(token), employee_id=employee.id,
                       expires_at=utcnow() + timedelta(hours=settings.session_hours)))
    response.set_cookie(settings.cookie_name, token, httponly=True,
                        secure=settings.cookie_secure, samesite="lax", path="/",
                        max_age=settings.session_hours * 3600)


def is_expired(expiry):
    if expiry.tzinfo is None:
        expiry = expiry.replace(tzinfo=timezone.utc)
    return expiry <= utcnow()
