"""Swipe deck API tests on the Postgres test database (skipped when it isn't available)."""

from __future__ import annotations

import os
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch
from uuid import uuid4

import asyncpg
import httpx

from app import auth, db, profile_store
from app.main import app

TEST_DATABASE_URL = os.getenv("TEST_DATABASE_URL", "postgresql://bs@/synq_test?host=/var/run/postgresql")
JPEG = b"\xff\xd8\xff\xe0" + b"\x00" * 60


class DiscoveryTests(unittest.IsolatedAsyncioTestCase):
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
        self.me = await self.sign_up("me@example.com", "Me")

    async def asyncTearDown(self) -> None:
        await self.client.aclose()
        self.settings_patch.stop()
        self.media_dir.cleanup()
        if db._pool is not None:
            await db._pool.close()
            db._pool = None

    async def sign_up(self, email: str, name: str) -> dict[str, str]:
        response = await self.client.post("/api/v1/auth/signup", json={"name": name, "email": email, "password": "correct horse 1"})
        return {"Authorization": f"Bearer {response.json()['access_token']}"}

    async def user_id(self, headers: dict[str, str]) -> str:
        return (await self.client.get("/api/v1/auth/me", headers=headers)).json()["id"]

    async def finish_profile(self, headers: dict[str, str], name: str) -> None:
        cover = (await self.client.post("/api/v1/profile/media?kind=photo", content=JPEG, headers={**headers, "Content-Type": "image/jpeg"})).json()
        body = {"name": name, "age": 28, "neighborhood": "Bandra West", "photos": [cover["id"], None, None]}
        response = await self.client.put("/api/v1/profile", json=body, headers=headers)
        self.assertEqual(response.status_code, 200)

    async def test_discover_lists_other_finished_profiles_only(self) -> None:
        await self.finish_profile(self.me, "Me")
        riya = await self.sign_up("riya@example.com", "Riya")
        await self.finish_profile(riya, "Riya")
        await self.sign_up("nobody@example.com", "No Profile")  # never finishes a profile
        response = await self.client.get("/api/v1/discover", headers=self.me)
        self.assertEqual(response.status_code, 200)
        profiles = response.json()["profiles"]
        self.assertEqual([item["profile"]["name"] for item in profiles], ["Riya"])
        self.assertFalse(profiles[0]["sparked"])

    async def test_spark_is_recorded_and_reset_clears_it(self) -> None:
        riya = await self.sign_up("riya@example.com", "Riya")
        await self.finish_profile(riya, "Riya")
        riya_id = await self.user_id(riya)
        sent = await self.client.post("/api/v1/sparks", json={"to_user_id": riya_id}, headers=self.me)
        self.assertEqual(sent.status_code, 201)
        again = await self.client.post("/api/v1/sparks", json={"to_user_id": riya_id}, headers=self.me)
        self.assertEqual(again.status_code, 201)  # sending twice is harmless
        listed = (await self.client.get("/api/v1/discover", headers=self.me)).json()["profiles"]
        self.assertTrue(listed[0]["sparked"])
        async with db._pool.acquire() as connection:
            self.assertEqual(await connection.fetchval("SELECT count(*) FROM sparks"), 1)
        reset = await self.client.post("/api/v1/sparks/reset", headers=self.me)
        self.assertEqual(reset.json(), {"cleared": 1})
        listed = (await self.client.get("/api/v1/discover", headers=self.me)).json()["profiles"]
        self.assertFalse(listed[0]["sparked"])

    async def test_spark_rejects_self_and_unknown_profiles(self) -> None:
        await self.finish_profile(self.me, "Me")
        to_self = await self.client.post("/api/v1/sparks", json={"to_user_id": await self.user_id(self.me)}, headers=self.me)
        unknown = await self.client.post("/api/v1/sparks", json={"to_user_id": str(uuid4())}, headers=self.me)
        self.assertEqual((to_self.status_code, unknown.status_code), (422, 404))

    async def test_external_photo_redirects(self) -> None:
        user_id = await self.user_id(self.me)
        media_id = uuid4()
        async with db._pool.acquire() as connection:
            await connection.execute(
                "INSERT INTO media (id, user_id, kind, content_type, byte_size, storage_path, external_url) "
                "VALUES ($1, $2, 'photo', 'image/jpeg', 1, '', 'https://images.example.com/a.jpg')",
                media_id, user_id,
            )
        response = await self.client.get(f"/api/v1/media/{media_id}")
        self.assertEqual((response.status_code, response.headers["location"]), (307, "https://images.example.com/a.jpg"))

    async def test_needs_sign_in(self) -> None:
        self.assertEqual((await self.client.get("/api/v1/discover")).status_code, 401)
        self.assertEqual((await self.client.post("/api/v1/sparks/reset")).status_code, 401)


if __name__ == "__main__":
    unittest.main()
