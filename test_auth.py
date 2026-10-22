"""Auth API tests against a real Postgres test database (skipped when it isn't available).

Create it once:  sudo -u postgres createdb -O bs synq_test
"""

from __future__ import annotations

import os
import unittest

import asyncpg
import httpx

from app import auth, db
from app.main import app

TEST_DATABASE_URL = os.getenv("TEST_DATABASE_URL", "postgresql://bs@/synq_test?host=/var/run/postgresql")


class AuthApiTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        try:
            pool = await asyncpg.create_pool(TEST_DATABASE_URL, min_size=1, max_size=4)
        except Exception as error:  # no test database on this machine
            self.skipTest(f"test database unavailable: {error}")
        await db.migrate(pool)
        async with pool.acquire() as connection:
            await connection.execute("TRUNCATE users CASCADE")
        db._pool = pool
        auth._failures.clear()
        self.client = httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test")

    async def asyncTearDown(self) -> None:
        await self.client.aclose()
        if db._pool is not None:
            await db._pool.close()
            db._pool = None

    async def sign_up(self, email: str = "Riya@Example.com", password: str = "correct horse 1") -> httpx.Response:
        return await self.client.post("/api/v1/auth/signup", json={"name": "Riya", "email": email, "password": password})

    async def test_sign_up_returns_tokens_and_normalised_email(self) -> None:
        response = await self.sign_up()
        self.assertEqual(response.status_code, 201)
        body = response.json()
        self.assertEqual(body["user"]["email"], "riya@example.com")
        self.assertTrue(body["access_token"] and body["refresh_token"])

    async def test_duplicate_email_is_rejected_in_any_case(self) -> None:
        await self.sign_up()
        response = await self.sign_up(email="RIYA@example.com")
        self.assertEqual(response.status_code, 409)

    async def test_invalid_input_is_rejected(self) -> None:
        bad_email = await self.sign_up(email="not-an-email")
        short_password = await self.sign_up(password="short")
        self.assertEqual(bad_email.status_code, 422)
        self.assertEqual(short_password.status_code, 422)

    async def test_sign_in_and_me(self) -> None:
        await self.sign_up()
        wrong = await self.client.post("/api/v1/auth/login", json={"email": "riya@example.com", "password": "nope nope"})
        self.assertEqual(wrong.status_code, 401)
        missing = await self.client.post("/api/v1/auth/login", json={"email": "nobody@example.com", "password": "nope nope"})
        self.assertEqual(missing.json()["detail"], wrong.json()["detail"])  # no hint which emails exist
        ok = await self.client.post("/api/v1/auth/login", json={"email": "riya@example.com", "password": "correct horse 1"})
        self.assertEqual(ok.status_code, 200)
        me = await self.client.get("/api/v1/auth/me", headers={"Authorization": f"Bearer {ok.json()['access_token']}"})
        self.assertEqual(me.json()["name"], "Riya")

    async def test_refresh_rotates_and_logout_revokes(self) -> None:
        first = (await self.sign_up()).json()
        rotated = await self.client.post("/api/v1/auth/refresh", json={"refresh_token": first["refresh_token"]})
        self.assertEqual(rotated.status_code, 200)
        reused = await self.client.post("/api/v1/auth/refresh", json={"refresh_token": first["refresh_token"]})
        self.assertEqual(reused.status_code, 401)  # an old refresh token stops working
        new = rotated.json()
        await self.client.post("/api/v1/auth/logout", json={"refresh_token": new["refresh_token"]})
        me = await self.client.get("/api/v1/auth/me", headers={"Authorization": f"Bearer {new['access_token']}"})
        self.assertEqual(me.status_code, 401)  # signing out ends the access token's session too

    async def test_repeated_wrong_passwords_are_slowed_down(self) -> None:
        await self.sign_up()
        for _ in range(5):
            await self.client.post("/api/v1/auth/login", json={"email": "riya@example.com", "password": "wrong pass"})
        blocked = await self.client.post("/api/v1/auth/login", json={"email": "riya@example.com", "password": "correct horse 1"})
        self.assertEqual(blocked.status_code, 429)

    async def test_google_is_not_available_yet(self) -> None:
        response = await self.client.post("/api/v1/auth/google")
        self.assertEqual(response.status_code, 501)


if __name__ == "__main__":
    unittest.main()
