"""Saved profiles and their photo/voice files, one profile per signed-in account.

- GET  /api/v1/profile          the user's saved profile (404 until the first save)
- PUT  /api/v1/profile          create or replace it
- POST /api/v1/profile/media    upload one photo or voice file (raw body, ?kind=photo|voice)
- GET  /api/v1/media/{id}       the file itself

Files are stored on disk under MEDIA_DIR. Their ids are random UUIDs, so the link
works like an unlisted link: it doesn't need a token, but can't be guessed. Upload
and profile changes always need the user's access token.
"""

from __future__ import annotations

import asyncio
import json
import os
from datetime import datetime
from typing import Annotated, Literal
from uuid import UUID, uuid4

import asyncpg
from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from fastapi.responses import FileResponse, RedirectResponse
from loguru import logger
from pydantic import BaseModel, Field, field_validator

from .auth import UserOut, _pool, current_user
from .config import settings

router = APIRouter(prefix="/api/v1/profile", tags=["profile"])
media_router = APIRouter(prefix="/api/v1/media", tags=["media"])

INTENTS = ("Lasting intimacy", "Long-term partner", "Open to see")
MediaKind = Literal["photo", "voice"]
_MAX_BYTES = {"photo": 12 * 1024 * 1024, "voice": 6 * 1024 * 1024}
_HEIC_BRANDS = {b"heic", b"heix", b"hevc", b"hevx", b"heim", b"heis"}
_HEIF_BRANDS = {b"mif1", b"msf1"}


# --- Shapes -------------------------------------------------------------------------------


def _one_line(value: str) -> str:
    return " ".join(value.split())


class VoiceIn(BaseModel):
    title: str = Field(default="", max_length=70)
    media_id: UUID | None = None
    duration_seconds: float = Field(default=0, ge=0, le=60)


class VignetteIn(BaseModel):
    title: str = Field(default="", max_length=80)
    body: str = Field(default="", max_length=400)


class VisibilityIn(BaseModel):
    vibe_deck_highlights: bool = True
    value_pillars: bool = True
    call_reflections: bool = False


class ProfileIn(BaseModel):
    # Required to create a profile: cover photo (photos[0]), name, age and neighborhood.
    name: str = Field(min_length=1, max_length=60)
    age: int = Field(ge=18, le=99)
    neighborhood: str = Field(max_length=120)
    craft: str = Field(default="", max_length=80)
    bio: str = Field(default="", max_length=280)
    intent: Literal["Lasting intimacy", "Long-term partner", "Open to see"] = "Lasting intimacy"
    voice: VoiceIn = Field(default_factory=VoiceIn)
    vignettes: list[VignetteIn] = Field(default_factory=list, max_length=2)
    visibility: VisibilityIn = Field(default_factory=VisibilityIn)
    # Cover photo, photo 2, photo 3: a media id each, or null.
    photos: list[UUID | None] = Field(default_factory=lambda: [None, None, None], min_length=3, max_length=3)

    @field_validator("name", "neighborhood", "craft")
    @classmethod
    def _tidy(cls, value: str) -> str:
        return _one_line(value)

    @field_validator("name")
    @classmethod
    def _name(cls, value: str) -> str:
        if not value:
            raise ValueError("Enter your name")
        return value

    @field_validator("neighborhood")
    @classmethod
    def _neighborhood(cls, value: str) -> str:
        if not value:
            raise ValueError("Choose your neighborhood")
        return value

    @field_validator("photos")
    @classmethod
    def _cover(cls, value: list[UUID | None]) -> list[UUID | None]:
        if value[0] is None:
            raise ValueError("Add a cover photo")
        return value


class MediaOut(BaseModel):
    id: UUID
    url: str


class VoiceOut(BaseModel):
    title: str
    media: MediaOut | None
    duration_seconds: float


class ProfileOut(BaseModel):
    name: str
    age: int | None
    neighborhood: str
    craft: str
    bio: str
    intent: str
    voice: VoiceOut
    vignettes: list[VignetteIn]
    visibility: VisibilityIn
    photos: list[MediaOut | None]
    updated_at: datetime


def _media(media_id: UUID | None) -> MediaOut | None:
    return MediaOut(id=media_id, url=f"/api/v1/media/{media_id}") if media_id else None


def _profile_out(row: asyncpg.Record) -> ProfileOut:
    photos = json.loads(row["photo_media_ids"])
    return ProfileOut(
        name=row["display_name"],
        age=row["age"],
        neighborhood=row["neighborhood"],
        craft=row["craft"],
        bio=row["bio"],
        intent=row["intent"],
        voice=VoiceOut(title=row["voice_title"], media=_media(row["voice_media_id"]), duration_seconds=row["voice_duration_seconds"]),
        vignettes=[VignetteIn(**item) for item in json.loads(row["vignettes"])],
        visibility=VisibilityIn(**json.loads(row["visibility"])),
        photos=[_media(UUID(item)) if item else None for item in photos],
        updated_at=row["updated_at"],
    )


# --- Profile ------------------------------------------------------------------------------


@router.get("", response_model=ProfileOut)
async def get_profile(user: Annotated[UserOut, Depends(current_user)]) -> ProfileOut:
    async with _pool().acquire() as connection:
        row = await connection.fetchrow("SELECT * FROM profiles WHERE user_id = $1", user.id)
    if row is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "No profile yet.")
    return _profile_out(row)


@router.put("", response_model=ProfileOut)
async def save_profile(payload: ProfileIn, user: Annotated[UserOut, Depends(current_user)]) -> ProfileOut:
    photo_ids = [str(item) if item else None for item in payload.photos]
    wanted = {UUID(item): "photo" for item in photo_ids if item}
    if payload.voice.media_id:
        wanted[payload.voice.media_id] = "voice"
    async with _pool().acquire() as connection:
        async with connection.transaction():
            if wanted:
                owned = await connection.fetch(
                    "SELECT id, kind FROM media WHERE user_id = $1 AND id = ANY($2::uuid[])", user.id, list(wanted)
                )
                if {(row["id"], row["kind"]) for row in owned} != set(wanted.items()):
                    raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, "A photo or recording could not be found. Please add it again.")
            row = await connection.fetchrow(
                """
                INSERT INTO profiles (user_id, display_name, age, neighborhood, craft, bio, intent, voice_title,
                                      voice_media_id, voice_duration_seconds, photo_media_ids, vignettes, visibility)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11::jsonb, $12::jsonb, $13::jsonb)
                ON CONFLICT (user_id) DO UPDATE SET
                    display_name = EXCLUDED.display_name, age = EXCLUDED.age, neighborhood = EXCLUDED.neighborhood,
                    craft = EXCLUDED.craft, bio = EXCLUDED.bio, intent = EXCLUDED.intent,
                    voice_title = EXCLUDED.voice_title, voice_media_id = EXCLUDED.voice_media_id,
                    voice_duration_seconds = EXCLUDED.voice_duration_seconds, photo_media_ids = EXCLUDED.photo_media_ids,
                    vignettes = EXCLUDED.vignettes, visibility = EXCLUDED.visibility, updated_at = now()
                RETURNING *
                """,
                user.id,
                payload.name,
                payload.age,
                payload.neighborhood,
                payload.craft,
                payload.bio.strip(),
                payload.intent,
                payload.voice.title.strip(),
                payload.voice.media_id,
                payload.voice.duration_seconds if payload.voice.media_id else 0,
                json.dumps(photo_ids),
                json.dumps([vignette.model_dump() for vignette in payload.vignettes]),
                json.dumps(payload.visibility.model_dump()),
            )
            # Files this user replaced more than a few minutes ago are no longer needed.
            stale = await connection.fetch(
                "DELETE FROM media WHERE user_id = $1 AND NOT (id = ANY($2::uuid[])) "
                "AND created_at < now() - interval '10 minutes' RETURNING storage_path",
                user.id,
                list(wanted),
            )
    for item in stale:
        await asyncio.to_thread(_remove_file, item["storage_path"])
    logger.info("PROFILE SAVED | user={} | photos={} | voice={}", user.id, len([p for p in photo_ids if p]), bool(payload.voice.media_id))
    return _profile_out(row)


# --- Media --------------------------------------------------------------------------------


def _sniff(kind: str, head: bytes) -> tuple[str, str] | None:
    """The file's real type from its first bytes, as (content type, extension).

    Phones don't always declare a type we recognise (Android pickers may send none, or
    a generic one), so the declared Content-Type is ignored and the bytes decide.
    """
    brand = head[8:12] if head[4:8] == b"ftyp" else None
    if kind == "photo":
        if head.startswith(b"\xff\xd8\xff"):
            return "image/jpeg", ".jpg"
        if head.startswith(b"\x89PNG\r\n\x1a\n"):
            return "image/png", ".png"
        if head[:4] == b"RIFF" and head[8:12] == b"WEBP":
            return "image/webp", ".webp"
        if brand in _HEIC_BRANDS:
            return "image/heic", ".heic"
        if brand in _HEIF_BRANDS:
            return "image/heif", ".heif"
        return None
    if brand is None:
        return None
    if brand.startswith(b"3g"):
        return "audio/3gpp", ".3gp"
    return "audio/mp4", ".m4a"  # M4A/MP4 audio


def _remove_file(relative: str) -> None:
    try:
        (settings.media_dir / relative).unlink(missing_ok=True)
    except OSError:
        logger.warning("MEDIA FILE NOT REMOVED | {}", relative)


def _write_file(relative: str, data: bytes) -> None:
    path = settings.media_dir / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(data)
    os.replace(temporary, path)


@router.post("/media", response_model=MediaOut, status_code=status.HTTP_201_CREATED)
async def upload_media(
    request: Request,
    user: Annotated[UserOut, Depends(current_user)],
    kind: Annotated[MediaKind, Query()],
) -> MediaOut:
    declared = int(request.headers.get("content-length") or 0)
    if declared > _MAX_BYTES[kind]:
        raise HTTPException(status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, "This file is too large.")
    data = await request.body()
    if not data or len(data) > _MAX_BYTES[kind]:
        raise HTTPException(status.HTTP_413_REQUEST_ENTITY_TOO_LARGE if data else status.HTTP_400_BAD_REQUEST, "This file is empty or too large.")
    sniffed = _sniff(kind, data[:16])
    if sniffed is None:
        logger.info("MEDIA REJECTED | user={} | kind={} | declared={} | head={}", user.id, kind, request.headers.get("content-type"), data[:12].hex())
        raise HTTPException(
            status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            "Please choose a JPEG, PNG, WebP or HEIC photo." if kind == "photo" else "This recording format isn't supported.",
        )
    content_type, extension = sniffed
    media_id = uuid4()
    relative = f"{user.id}/{media_id}{extension}"
    await asyncio.to_thread(_write_file, relative, data)
    async with _pool().acquire() as connection:
        await connection.execute(
            "INSERT INTO media (id, user_id, kind, content_type, byte_size, storage_path) VALUES ($1, $2, $3, $4, $5, $6)",
            media_id,
            user.id,
            kind,
            content_type,
            len(data),
            relative,
        )
    logger.info("MEDIA UPLOADED | user={} | kind={} | bytes={}", user.id, kind, len(data))
    return MediaOut(id=media_id, url=f"/api/v1/media/{media_id}")


@media_router.get("/{media_id}", response_model=None)
async def get_media(media_id: UUID) -> FileResponse | RedirectResponse:
    async with _pool().acquire() as connection:
        row = await connection.fetchrow("SELECT content_type, storage_path, external_url FROM media WHERE id = $1", media_id)
    if row is not None and row["external_url"]:
        # A photo hosted elsewhere (the dummy profiles use public stock photos).
        return RedirectResponse(row["external_url"], status_code=status.HTTP_307_TEMPORARY_REDIRECT)
    path = settings.media_dir / row["storage_path"] if row else None
    if row is None or path is None or not path.is_file():
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Not found.")
    # A media id never points at different content, so it can be cached for a long time.
    return FileResponse(path, media_type=row["content_type"], headers={"Cache-Control": "private, max-age=31536000, immutable"})
