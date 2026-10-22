"""Saved profile and media tests on the Postgres test database (skipped when it isn't available)."""

from __future__ import annotations

import os
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import asyncpg
import httpx

from app import auth, db, profile_store
from app.main import app

TEST_DATABASE_URL = os.getenv("TEST_DATABASE_URL", "postgresql://bs@/synq_test?host=/var/run/postgresql")
JPEG = b"\xff\xd8\xff\xe0" + b"\x00" * 60
M4A = b"\x00\x00\x00\x20ftypM4A " + b"\x00" * 60


class ProfileStoreTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        try:
            pool = await asyncpg.create_pool(TEST_DATABASE_URL, min_size=1, max_size=4)
        except Exception as error:
            self.skipTest(f"test database unavailable: {error}")
        await db.migrate(pool)
        async with pool.acquire() as connection:
            await connection.execute("TRUNCATE users CASCADE")
        db._pool = pool
        auth._failures.clear()
        self.media_dir = tempfile.TemporaryDirectory()
        self.settings_patch = patch.object(profile_store, "settings", replace(profile_store.settings, media_dir=Path(self.media_dir.name)))
        self.settings_patch.start()
        self.client = httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test")
        self.headers = await self.sign_up("riya@example.com")

    async def asyncTearDown(self) -> None:
        await self.client.aclose()
        self.settings_patch.stop()
        self.media_dir.cleanup()
        if db._pool is not None:
            await db._pool.close()
            db._pool = None

    async def sign_up(self, email: str) -> dict[str, str]:
        response = await self.client.post("/api/v1/auth/signup", json={"name": "Riya", "email": email, "password": "correct horse 1"})
        return {"Authorization": f"Bearer {response.json()['access_token']}"}

    async def upload(self, kind: str, data: bytes, content_type: str, headers: dict[str, str] | None = None) -> httpx.Response:
        return await self.client.post(
            f"/api/v1/profile/media?kind={kind}",
            content=data,
            headers={**(self.headers if headers is None else headers), "Content-Type": content_type},
        )

    async def cover(self) -> str:
        return (await self.upload("photo", JPEG, "image/jpeg")).json()["id"]

    async def profile(self, **overrides: object) -> dict[str, object]:
        body: dict[str, object] = {
            "name": "Riya",
            "age": 27,
            "neighborhood": "Bandra West",
            "craft": "Software engineer",
            "bio": "Coding by day, river walks by dusk.",
            "intent": "Long-term partner",
            "voice": {"title": "Rain on a tin roof", "media_id": None, "duration_seconds": 0},
            "vignettes": [{"title": "Sunday", "body": "Chai and a long walk."}],
            "visibility": {"vibe_deck_highlights": True, "value_pillars": False, "call_reflections": False},
            "photos": [await self.cover(), None, None],
        }
        body.update(overrides)
        return body

    async def test_no_profile_before_the_first_save(self) -> None:
        response = await self.client.get("/api/v1/profile", headers=self.headers)
        self.assertEqual(response.status_code, 404)

    async def test_save_with_photo_and_voice_then_read_back(self) -> None:
        photo = (await self.upload("photo", JPEG, "image/jpeg")).json()
        voice = (await self.upload("voice", M4A, "audio/mp4")).json()
        body = await self.profile(
            photos=[photo["id"], None, None],
            voice={"title": "Rain on a tin roof", "media_id": voice["id"], "duration_seconds": 12.4},
        )
        saved = await self.client.put("/api/v1/profile", json=body, headers=self.headers)
        self.assertEqual(saved.status_code, 200)
        loaded = (await self.client.get("/api/v1/profile", headers=self.headers)).json()
        self.assertEqual(loaded["intent"], "Long-term partner")
        self.assertEqual(loaded["photos"][0]["id"], photo["id"])
        self.assertEqual(loaded["voice"]["media"]["id"], voice["id"])
        self.assertAlmostEqual(loaded["voice"]["duration_seconds"], 12.4, places=1)
        self.assertFalse(loaded["visibility"]["value_pillars"])
        file = await self.client.get(loaded["photos"][0]["url"])
        self.assertEqual((file.status_code, file.content), (200, JPEG))

    async def test_saving_again_replaces_the_profile(self) -> None:
        await self.client.put("/api/v1/profile", json=await self.profile(), headers=self.headers)
        await self.client.put("/api/v1/profile", json=await self.profile(bio="Changed my mind.", age=31), headers=self.headers)
        loaded = (await self.client.get("/api/v1/profile", headers=self.headers)).json()
        self.assertEqual((loaded["bio"], loaded["age"]), ("Changed my mind.", 31))

    async def test_rejects_fake_or_unsupported_files(self) -> None:
        self.assertEqual((await self.upload("photo", b"not an image at all....", "image/jpeg")).status_code, 415)
        self.assertEqual((await self.upload("photo", b"%PDF-1.7 fake document....", "image/jpeg")).status_code, 415)
        self.assertEqual((await self.upload("voice", JPEG, "image/jpeg")).status_code, 415)

    async def test_type_comes_from_the_file_not_the_header(self) -> None:
        # Android pickers may send a generic or missing type; the bytes decide.
        for declared in ("application/octet-stream", "image/jpg", ""):
            response = await self.upload("photo", JPEG, declared)
            self.assertEqual(response.status_code, 201, declared)
        heic = b"\x00\x00\x00\x18ftypheic" + b"\x00" * 60
        uploaded = (await self.upload("photo", heic, "application/octet-stream")).json()
        file = await self.client.get(uploaded["url"])
        self.assertEqual(file.headers["content-type"], "image/heic")

    async def test_cannot_use_someone_elses_photo(self) -> None:
        other = await self.sign_up("other@example.com")
        their_photo = (await self.upload("photo", JPEG, "image/jpeg", headers=other)).json()
        response = await self.client.put("/api/v1/profile", json=await self.profile(photos=[their_photo["id"], None, None]), headers=self.headers)
        self.assertEqual(response.status_code, 422)

    async def test_invalid_profiles_are_rejected(self) -> None:
        too_young = await self.client.put("/api/v1/profile", json=await self.profile(age=17), headers=self.headers)
        three_vignettes = await self.client.put(
            "/api/v1/profile", json=await self.profile(vignettes=[{"title": "a", "body": "b"}] * 3), headers=self.headers
        )
        no_name = await self.client.put("/api/v1/profile", json=await self.profile(name="   "), headers=self.headers)
        self.assertEqual((too_young.status_code, three_vignettes.status_code, no_name.status_code), (422, 422, 422))

    async def test_cover_name_age_and_neighborhood_are_required(self) -> None:
        for missing in ({"photos": [None, None, None]}, {"age": None}, {"neighborhood": "  "}, {"name": ""}):
            response = await self.client.put("/api/v1/profile", json=await self.profile(**missing), headers=self.headers)
            self.assertEqual(response.status_code, 422, missing)
        body = await self.profile()
        del body["age"]
        self.assertEqual((await self.client.put("/api/v1/profile", json=body, headers=self.headers)).status_code, 422)
        optional_empty = await self.profile(craft="", bio="", vignettes=[], voice={"title": "", "media_id": None, "duration_seconds": 0})
        self.assertEqual((await self.client.put("/api/v1/profile", json=optional_empty, headers=self.headers)).status_code, 200)

    async def test_needs_sign_in(self) -> None:
        self.assertEqual((await self.client.get("/api/v1/profile")).status_code, 401)
        self.assertEqual((await self.upload("photo", JPEG, "image/jpeg", headers={})).status_code, 401)


if __name__ == "__main__":
    unittest.main()
