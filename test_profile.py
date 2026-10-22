"""Polish-tone endpoint tests (the LLM is faked; no database needed)."""

from __future__ import annotations

import unittest
from datetime import datetime, timezone
from unittest.mock import patch
from uuid import uuid4

import httpx

from app import pod4_providers
from app.auth import UserOut, current_user
from app.main import app

BIO = "I make pottery and drive along the coast at sunset. Looking for someone kind who likes slow coffee."


class PolishBioTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        app.dependency_overrides[current_user] = lambda: UserOut(
            id=uuid4(), email="riya@example.com", name="Riya", created_at=datetime.now(timezone.utc)
        )
        self.client = httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test")

    async def asyncTearDown(self) -> None:
        await self.client.aclose()
        app.dependency_overrides.clear()

    async def test_returns_the_polished_text(self) -> None:
        async def fake(messages, **_):
            self.assertIn("own voice", messages[0].content)
            self.assertEqual(messages[1].content, BIO)
            return '"I shape clay and chase coastal sunsets. Hoping to meet someone kind who loves slow coffee."', "fake"

        with patch.object(pod4_providers, "complete", side_effect=fake):
            response = await self.client.post("/api/v1/profile/polish-bio", json={"text": BIO})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["text"], "I shape clay and chase coastal sunsets. Hoping to meet someone kind who loves slow coffee.")

    async def test_overlong_answer_is_trimmed_to_about_the_original_length(self) -> None:
        long_reply = "This is a sentence that keeps going. " * 20

        async def fake(*_, **__):
            return long_reply, "fake"

        with patch.object(pod4_providers, "complete", side_effect=fake):
            response = await self.client.post("/api/v1/profile/polish-bio", json={"text": BIO})
        text = response.json()["text"]
        self.assertLessEqual(len(text), len(BIO) + 40)
        self.assertTrue(text.endswith("."))

    async def test_llm_down_is_a_clear_503(self) -> None:
        async def down(*_, **__):
            raise RuntimeError("no llm")

        with patch.object(pod4_providers, "complete", side_effect=down):
            response = await self.client.post("/api/v1/profile/polish-bio", json={"text": BIO})
        self.assertEqual(response.status_code, 503)

    async def test_requires_sign_in_and_some_text(self) -> None:
        app.dependency_overrides.clear()
        signed_out = await self.client.post("/api/v1/profile/polish-bio", json={"text": BIO})
        self.assertIn(signed_out.status_code, (401, 503))
        app.dependency_overrides[current_user] = lambda: UserOut(
            id=uuid4(), email="r@example.com", name="R", created_at=datetime.now(timezone.utc)
        )
        too_short = await self.client.post("/api/v1/profile/polish-bio", json={"text": "hi"})
        self.assertEqual(too_short.status_code, 422)


if __name__ == "__main__":
    unittest.main()
