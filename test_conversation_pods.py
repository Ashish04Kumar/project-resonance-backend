from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from app import main
from app.conversation import CallSession, update_live_conversation
from app.conversation_pods import get_conversation_pod
from app.conversation_schemas import TurnObservation
from app.models import CreateCallRequest
from app.voice_agent import _gemini_system_instruction


class ConversationPodTests(unittest.IsolatedAsyncioTestCase):
    def test_frontend_request_does_not_select_conversation_pod(self) -> None:
        self.assertNotIn("conversation_pod", CreateCallRequest.model_fields)

    async def test_backend_setting_selects_the_pod_for_a_new_call(self) -> None:
        request = CreateCallRequest(consent=True)
        settings = SimpleNamespace(
            conversation_pod="pod2",
            voice_provider="gemini_live",
            gemini_api_key="configured",
        )
        with (
            patch.object(main, "settings", settings),
            patch.object(main, "start_agent", new_callable=AsyncMock),
            patch.object(main, "client_livekit_url", return_value="ws://test"),
            patch.object(main, "participant_token", return_value="test-token"),
        ):
            response = await main.create_call(request)

        session = main.call_store.remove(response.call_id)
        self.assertIsNotNone(session)
        self.assertEqual(session.conversation_pod, "pod2")
        self.assertEqual(response.conversation_pod, "pod2")


    def test_pod_two_is_an_independent_copy_with_its_earlier_behavior_preserved(self) -> None:
        stable = get_conversation_pod("pod1")
        candidate = get_conversation_pod("pod2")

        self.assertEqual(stable.status, "stable")
        self.assertEqual(candidate.status, "clone")
        self.assertIsNot(stable.handle_turn, candidate.handle_turn)
        self.assertNotEqual(stable.prompt_addendum, candidate.prompt_addendum)
        self.assertEqual(candidate.prompt_addendum, "")


    def test_pause_follow_up_prompt_is_scoped_to_pod_one(self) -> None:
        stable_session = CallSession("stable-call", "english", conversation_pod="pod1")
        candidate_session = CallSession("candidate-call", "english", conversation_pod="pod2")

        self.assertIn("LONG-PAUSE FOLLOW-UP", _gemini_system_instruction(stable_session))
        self.assertNotIn("LONG-PAUSE FOLLOW-UP", _gemini_system_instruction(candidate_session))


    async def test_both_pods_run_the_same_stable_conductor_without_changing_state_shape(self) -> None:
        observation = TurnObservation(intent="small_talk")
        stable_session = CallSession("stable-call", "english", conversation_pod="pod1")
        candidate_session = CallSession("candidate-call", "english", conversation_pod="pod2")

        stable_decision = await update_live_conversation(stable_session, observation)
        candidate_decision = await update_live_conversation(candidate_session, observation)

        self.assertEqual(stable_decision, candidate_decision)
        self.assertEqual(stable_session.conversation_state, candidate_session.conversation_state)
        self.assertEqual(stable_session.conversation_state.turn_count, 1)

    async def test_pod1_idle_nudge_uses_shared_words_without_advancing_the_user_turn_state(self) -> None:
        session = CallSession("idle-call", "english", conversation_pod="pod1")
        session.latest_user_text = "I like quiet Sunday mornings and making coffee."
        session.idle_nudge_pending = True

        decision = await update_live_conversation(session, TurnObservation(intent="other"))

        self.assertEqual(decision.grounding_detail, session.latest_user_text)
        self.assertIn("long-pause", decision.cue)
        self.assertFalse(session.idle_nudge_pending)
        self.assertEqual(session.conversation_state.turn_count, 0)
