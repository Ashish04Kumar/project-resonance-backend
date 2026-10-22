"""Email sign-up and sign-in, with short-lived access tokens and rotating refresh tokens.

- Passwords are hashed with Argon2; a missing account takes the same time as a wrong password.
- The access token (JWT, 15 minutes) goes in `Authorization: Bearer ...`.
- The refresh token (random, 30 days) is stored only as a SHA-256 hash, one row per
  signed-in device, and is replaced on every refresh so a stolen old token stops working.
- Repeated wrong passwords for one email are slowed down (429).
- Google sign-in has its own route that answers 501 until it is integrated.
"""

from __future__ import annotations

import hashlib
import re
import secrets
import time
from collections import defaultdict, deque
from datetime import datetime, timedelta, timezone
from typing import Annotated
from uuid import UUID

import asyncpg
import jwt
from argon2 import PasswordHasher
from argon2.exceptions import InvalidHashError, VerifyMismatchError
from fastapi import APIRouter, Depends, Header, HTTPException, status
from loguru import logger
from pydantic import BaseModel, Field, field_validator

from . import db
from .config import settings

router = APIRouter(prefix="/api/v1/auth", tags=["auth"])

_hasher = PasswordHasher()
# Used when the email has no account, so both paths cost one Argon2 verification.
_DUMMY_HASH = _hasher.hash("not-a-real-password")
_EMAIL = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
_MAX_FAILURES = 5
_FAILURE_WINDOW_SECONDS = 15 * 60
_failures: dict[str, deque[float]] = defaultdict(deque)


# --- Request and response shapes ------------------------------------------------------


def _clean_email(value: str) -> str:
    email = value.strip().lower()
    if len(email) > 254 or not _EMAIL.match(email):
        raise ValueError("Enter a valid email address")
    return email


class SignUpRequest(BaseModel):
    name: str = Field(min_length=1, max_length=60)
    email: str
    password: str = Field(min_length=8, max_length=128)
    device_label: str | None = Field(default=None, max_length=80)

    @field_validator("email")
    @classmethod
    def _email(cls, value: str) -> str:
        return _clean_email(value)

    @field_validator("name")
    @classmethod
    def _name(cls, value: str) -> str:
        name = " ".join(value.split())
        if not name:
            raise ValueError("Enter your name")
        return name


class SignInRequest(BaseModel):
    email: str
    password: str = Field(min_length=1, max_length=128)
    device_label: str | None = Field(default=None, max_length=80)

    @field_validator("email")
    @classmethod
    def _email(cls, value: str) -> str:
        return _clean_email(value)


class RefreshRequest(BaseModel):
    refresh_token: str = Field(min_length=20, max_length=200)


class UserOut(BaseModel):
    id: UUID
    email: str
    name: str
    created_at: datetime


class AuthResponse(BaseModel):
    user: UserOut
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int


# --- Helpers ------------------------------------------------------------------------------


def _pool() -> asyncpg.Pool:
    pool = db.pool()
    if pool is None:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "Accounts are not available right now.")
    return pool


def _secret() -> str:
    if len(settings.auth_jwt_secret) < 32:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "Accounts are not configured.")
    return settings.auth_jwt_secret


def _hash_token(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def _access_token(user_id: UUID, session_id: UUID) -> str:
    now = datetime.now(timezone.utc)
    claims = {
        "sub": str(user_id),
        "sid": str(session_id),
        "typ": "access",
        "iat": now,
        "exp": now + timedelta(minutes=settings.auth_access_token_minutes),
    }
    return jwt.encode(claims, _secret(), algorithm="HS256")


def _user_out(row: asyncpg.Record) -> UserOut:
    return UserOut(id=row["id"], email=row["email"], name=row["display_name"], created_at=row["created_at"])


async def _start_session(connection: asyncpg.Connection, user: asyncpg.Record, device_label: str | None) -> AuthResponse:
    refresh_token = secrets.token_urlsafe(48)
    expires_at = datetime.now(timezone.utc) + timedelta(days=settings.auth_refresh_token_days)
    session_id = await connection.fetchval(
        "INSERT INTO auth_sessions (user_id, refresh_token_hash, device_label, expires_at) VALUES ($1, $2, $3, $4) RETURNING id",
        user["id"],
        _hash_token(refresh_token),
        device_label,
        expires_at,
    )
    return AuthResponse(
        user=_user_out(user),
        access_token=_access_token(user["id"], session_id),
        refresh_token=refresh_token,
        expires_in=settings.auth_access_token_minutes * 60,
    )


def _too_many_failures(email: str) -> bool:
    attempts = _failures[email]
    cutoff = time.monotonic() - _FAILURE_WINDOW_SECONDS
    while attempts and attempts[0] < cutoff:
        attempts.popleft()
    return len(attempts) >= _MAX_FAILURES


# --- Routes ------------------------------------------------------------------------------


@router.post("/signup", response_model=AuthResponse, status_code=status.HTTP_201_CREATED)
async def sign_up(payload: SignUpRequest) -> AuthResponse:
    password_hash = _hasher.hash(payload.password)
    async with _pool().acquire() as connection:
        async with connection.transaction():
            try:
                user = await connection.fetchrow(
                    "INSERT INTO users (email, display_name, password_hash, last_login_at) VALUES ($1, $2, $3, now()) "
                    "RETURNING id, email, display_name, created_at",
                    payload.email,
                    payload.name,
                    password_hash,
                )
            except asyncpg.UniqueViolationError:
                raise HTTPException(
                    status.HTTP_409_CONFLICT, "An account with this email already exists. Try signing in."
                ) from None
            response = await _start_session(connection, user, payload.device_label)
    logger.info("AUTH SIGN UP | user={}", user["id"])
    return response


@router.post("/login", response_model=AuthResponse)
async def sign_in(payload: SignInRequest) -> AuthResponse:
    if _too_many_failures(payload.email):
        raise HTTPException(status.HTTP_429_TOO_MANY_REQUESTS, "Too many attempts. Please try again in a few minutes.")
    async with _pool().acquire() as connection:
        user = await connection.fetchrow(
            "SELECT id, email, display_name, created_at, password_hash FROM users WHERE lower(email) = $1",
            payload.email,
        )
        stored_hash = user["password_hash"] if user and user["password_hash"] else _DUMMY_HASH
        try:
            _hasher.verify(stored_hash, payload.password)
            valid = user is not None and stored_hash != _DUMMY_HASH
        except (VerifyMismatchError, InvalidHashError):
            valid = False
        if not valid:
            _failures[payload.email].append(time.monotonic())
            raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Email or password is incorrect.")
        _failures.pop(payload.email, None)
        async with connection.transaction():
            if _hasher.check_needs_rehash(stored_hash):
                await connection.execute(
                    "UPDATE users SET password_hash = $1, updated_at = now() WHERE id = $2",
                    _hasher.hash(payload.password),
                    user["id"],
                )
            await connection.execute("UPDATE users SET last_login_at = now() WHERE id = $1", user["id"])
            response = await _start_session(connection, user, payload.device_label)
    logger.info("AUTH SIGN IN | user={}", user["id"])
    return response


@router.post("/refresh", response_model=AuthResponse)
async def refresh(payload: RefreshRequest) -> AuthResponse:
    async with _pool().acquire() as connection:
        async with connection.transaction():
            session = await connection.fetchrow(
                "SELECT s.id, s.user_id, u.email, u.display_name, u.created_at, u.id AS uid "
                "FROM auth_sessions s JOIN users u ON u.id = s.user_id "
                "WHERE s.refresh_token_hash = $1 AND s.revoked_at IS NULL AND s.expires_at > now() FOR UPDATE OF s",
                _hash_token(payload.refresh_token),
            )
            if session is None:
                raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Your session has ended. Please sign in again.")
            new_refresh = secrets.token_urlsafe(48)
            await connection.execute(
                "UPDATE auth_sessions SET refresh_token_hash = $1, last_used_at = now(), expires_at = $2 WHERE id = $3",
                _hash_token(new_refresh),
                datetime.now(timezone.utc) + timedelta(days=settings.auth_refresh_token_days),
                session["id"],
            )
    user = UserOut(id=session["uid"], email=session["email"], name=session["display_name"], created_at=session["created_at"])
    return AuthResponse(
        user=user,
        access_token=_access_token(session["uid"], session["id"]),
        refresh_token=new_refresh,
        expires_in=settings.auth_access_token_minutes * 60,
    )


@router.post("/logout", status_code=status.HTTP_204_NO_CONTENT)
async def logout(payload: RefreshRequest) -> None:
    async with _pool().acquire() as connection:
        await connection.execute(
            "UPDATE auth_sessions SET revoked_at = now() WHERE refresh_token_hash = $1 AND revoked_at IS NULL",
            _hash_token(payload.refresh_token),
        )


async def current_user(authorization: Annotated[str | None, Header()] = None) -> UserOut:
    """FastAPI dependency: the signed-in user from a valid, unrevoked access token."""
    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Please sign in.")
    try:
        claims = jwt.decode(authorization[7:].strip(), _secret(), algorithms=["HS256"])
        if claims.get("typ") != "access":
            raise jwt.InvalidTokenError("wrong token type")
        user_id, session_id = UUID(claims["sub"]), UUID(claims["sid"])
    except (jwt.InvalidTokenError, KeyError, ValueError):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Please sign in.") from None
    async with _pool().acquire() as connection:
        row = await connection.fetchrow(
            "SELECT u.id, u.email, u.display_name, u.created_at FROM users u JOIN auth_sessions s ON s.user_id = u.id "
            "WHERE u.id = $1 AND s.id = $2 AND s.revoked_at IS NULL AND s.expires_at > now()",
            user_id,
            session_id,
        )
    if row is None:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Please sign in.")
    return _user_out(row)


@router.get("/me", response_model=UserOut)
async def me(user: Annotated[UserOut, Depends(current_user)]) -> UserOut:
    return user


@router.post("/google", status_code=status.HTTP_501_NOT_IMPLEMENTED)
async def google_sign_in() -> None:
    raise HTTPException(status.HTTP_501_NOT_IMPLEMENTED, "Google sign-in is coming soon.")
