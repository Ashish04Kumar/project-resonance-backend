"""The swipe deck: other people's profiles, sparks, and a reset for testing.

- GET  /api/v1/discover       every other user's finished profile, with whether I've sparked them
- POST /api/v1/sparks         record a spark (Send Spark); passing records nothing
- POST /api/v1/sparks/reset   delete my sparks, so the whole deck shows again

A finished profile has a cover photo, an age and a neighborhood (name is always set).
"""

from __future__ import annotations

from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from loguru import logger
from pydantic import BaseModel

from .auth import UserOut, _pool, current_user
from .profile_store import ProfileOut, _profile_out

router = APIRouter(prefix="/api/v1", tags=["discovery"])

_FINISHED = "p.photo_media_ids->>0 IS NOT NULL AND p.age IS NOT NULL AND p.neighborhood <> ''"


class DiscoverItem(BaseModel):
    user_id: UUID
    profile: ProfileOut
    sparked: bool


class DiscoverOut(BaseModel):
    profiles: list[DiscoverItem]


class SparkIn(BaseModel):
    to_user_id: UUID


class SparkOut(BaseModel):
    to_user_id: UUID
    sparked: bool


class ResetOut(BaseModel):
    cleared: int


@router.get("/discover", response_model=DiscoverOut)
async def discover(user: Annotated[UserOut, Depends(current_user)]) -> DiscoverOut:
    async with _pool().acquire() as connection:
        rows = await connection.fetch(
            f"""
            SELECT p.*, (s.to_user IS NOT NULL) AS sparked
            FROM profiles p
            LEFT JOIN sparks s ON s.from_user = $1 AND s.to_user = p.user_id
            WHERE p.user_id <> $1 AND {_FINISHED}
            ORDER BY p.created_at, p.user_id
            """,
            user.id,
        )
    return DiscoverOut(profiles=[DiscoverItem(user_id=row["user_id"], profile=_profile_out(row), sparked=row["sparked"]) for row in rows])


@router.post("/sparks", response_model=SparkOut, status_code=status.HTTP_201_CREATED)
async def send_spark(payload: SparkIn, user: Annotated[UserOut, Depends(current_user)]) -> SparkOut:
    if payload.to_user_id == user.id:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, "You can't send a spark to yourself.")
    async with _pool().acquire() as connection:
        exists = await connection.fetchval(
            f"SELECT 1 FROM profiles p WHERE p.user_id = $1 AND {_FINISHED}", payload.to_user_id
        )
        if not exists:
            raise HTTPException(status.HTTP_404_NOT_FOUND, "This profile isn't available.")
        await connection.execute(
            "INSERT INTO sparks (from_user, to_user) VALUES ($1, $2) ON CONFLICT DO NOTHING", user.id, payload.to_user_id
        )
    logger.info("SPARK SENT | from={} | to={}", user.id, payload.to_user_id)
    return SparkOut(to_user_id=payload.to_user_id, sparked=True)


@router.post("/sparks/reset", response_model=ResetOut)
async def reset_sparks(user: Annotated[UserOut, Depends(current_user)]) -> ResetOut:
    async with _pool().acquire() as connection:
        result = await connection.execute("DELETE FROM sparks WHERE from_user = $1", user.id)
    cleared = int(result.split()[-1])
    logger.info("SPARKS RESET | user={} | cleared={}", user.id, cleared)
    return ResetOut(cleared=cleared)
