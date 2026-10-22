from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from app import main
from app.conversation_pods import PODS
from app.models import CreateCallRequest


class ConversationPodTests(unittest.IsolatedAsyncioTestCase):
    def test_only_pods_four_and_five_remain(self) -> None:
        self.assertEqual(set(PODS), {"pod4", "pod5"})

    def test_frontend_request_does_not_select_conversation_pod(self) -> None:
        self.assertNotIn("conversation_pod", CreateCallRequest.model_fields)

    async def test_backend_setting_selects_the_pod_for_a_new_call(self) -> None:
        request = CreateCallRequest(consent=True)
        settings = SimpleNamespace(
            conversation_pod="pod5",
            voice_provider="gemini_live",
            gemini_api_key="configured",
        )
        with (
            patch.object(main, "settings", settings),
            patch.object(main, "start_agent", new_callable=AsyncMock),
            patch.object(main, "client_livekit_url", return_value="ws://test"),
            patch.object(main, "participant_token", return_value="test-token"),
            patch.object(main, "generate_turn", AsyncMock(return_value=("Hi!", "test"))),
        ):
            response = await main.create_call(request)

        session = main.call_store.remove(response.call_id)
        self.assertIsNotNone(session)
        self.assertEqual(session.conversation_pod, "pod5")
        self.assertEqual(response.conversation_pod, "pod5")
