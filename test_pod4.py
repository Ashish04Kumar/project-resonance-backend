"""Tests for Pod 4, the Wave-style Conductor (stages in order, soft skips park a stage)."""

from __future__ import annotations

import ast
import unittest
from pathlib import Path

from app.conversation_pod4 import (
    POD4_BLUEPRINT,
    POD4_GOAL_ORDER,
    Pod4ConversationState,
    Pod4Decision,
    Pod4TurnObservation,
    hard_decline,
    is_flat_reply,
    raised_goals,
    plan_pod4_turn,
    pod4_coverage_report,
)
from app.conversation_pods import get_conversation_pod
from dataclasses import replace
from unittest.mock import patch

from app import pod4_providers
from app.config import settings
from app.conversation import CallSession
from app.conversation_pod4_text import observe_turn, run_pod4_turn

EARLY = 60  # seconds: inside the core segment and before detours are allowed
GOAL_SLOTS = {goal.id: [slot.id for slot in goal.slots] for goal in POD4_BLUEPRINT.goals}


def answer_for(goal_id: str, slots: list[str] | None = None, intent: str = "small_talk") -> tuple[str, Pod4TurnObservation]:
    """A grounded reply stating the given slots; small_talk intent avoids triggering a follow-up."""
    slots = slots if slots is not None else GOAL_SLOTS[goal_id]
    pieces = [f"about {slot.replace('_', ' ')} i would say yes" for slot in slots]
    text = ". ".join(pieces) + "."
    fills = [{"slot": f"{goal_id}.{slot}", "quote": piece, "value": piece} for slot, piece in zip(slots, pieces)]
    outcome = "full" if len(slots) == len(GOAL_SLOTS[goal_id]) else "partial"
    return text, Pod4TurnObservation(intent=intent, outcome=outcome, outcome_confidence=0.9, slots_filled=fills)


async def step(
    state: Pod4ConversationState, text: str, observation: Pod4TurnObservation, elapsed: int = EARLY
) -> tuple[Pod4ConversationState, Pod4Decision]:
    return await plan_pod4_turn(state, observation, user_text=text, elapsed_seconds=elapsed)


def chat() -> Pod4TurnObservation:
    return Pod4TurnObservation(intent="small_talk", outcome="not_applicable", outcome_confidence=0.9)


def skip() -> Pod4TurnObservation:
    return Pod4TurnObservation(intent="skip", outcome="skip", outcome_confidence=1.0)


def no_answer() -> Pod4TurnObservation:
    return Pod4TurnObservation(intent="answer", outcome="no_answer", outcome_confidence=0.9)


async def past_conversation_turns(
    state: Pod4ConversationState, decision: Pod4Decision
) -> tuple[Pod4ConversationState, Pod4Decision]:
    """Reply with small talk through Wave's no-question conversational turns until a stage is asked."""
    for _ in range(4):
        if decision.goal and decision.ask_question:
            break
        state, decision = await step(state, "haha yeah", chat())
    return state, decision


class Pod4Tests(unittest.IsolatedAsyncioTestCase):
    def test_registered_with_its_own_blueprint(self) -> None:
        pod = get_conversation_pod("pod4")
        self.assertEqual(pod.id, "pod4")
        self.assertEqual(pod.prompt_addendum, "")  # Pod 4 writes its own Talker prompt
        self.assertIn("dating_journey.usual_pattern", POD4_BLUEPRINT.slot_keys)
        self.assertEqual(POD4_BLUEPRINT.call.max_call_seconds, 1080)

    def test_providers_follow_env_choice_with_fallback(self) -> None:
        with patch.object(pod4_providers, "_make_stt", side_effect=[RuntimeError("no model"), "sarvam-stt"]):
            service, name = pod4_providers.make_stt()
        self.assertEqual((service, name), ("sarvam-stt", settings.pod4_stt_fallback))
        # Sarvam needs its key; without it the factory refuses instead of failing mid-call.
        with patch.object(pod4_providers, "settings", replace(settings, sarvam_api_key="")):
            with self.assertRaises(RuntimeError):
                pod4_providers._make_tts("sarvam")

    async def test_llm_falls_back_per_request(self) -> None:
        calls = []

        def broken(*_):
            calls.append("primary")
            raise RuntimeError("down")

        def working(*_):
            calls.append("fallback")
            return "hello"

        first_fallback = settings.pod4_llm_fallback.split(",")[0].strip()
        with patch.dict(pod4_providers._LLMS, {settings.pod4_llm: broken, first_fallback: working}):
            text, name = await pod4_providers.complete([pod4_providers.Message("user", "hi")], temperature=0, max_tokens=10)
        self.assertEqual((text, name), ("hello", first_fallback))
        self.assertEqual(calls, ["primary", "fallback"])

    async def test_turn_runs_observe_then_conductor_then_talker(self) -> None:
        session = CallSession("pod4-test", "english", conversation_pod="pod4")
        replies = iter([
            '{"intent": "answer", "tone_cue": "neutral", "tone_confidence": 0.8, "outcome": "partial", '
            '"outcome_confidence": 0.9, "slots_filled": [{"slot": "life_context.location", "quote": "from Lucknow", '
            '"value": "from Lucknow"}], "distress": false, "distress_quote": "", "asr_unclear": false}',
            "Lucknow, nice. What brought you to Ahmedabad?",
        ])
        prompts = []

        async def fake_complete(messages, **_):
            prompts.append(messages[0].content)
            return next(replies), "fake"

        with patch.object(pod4_providers, "complete", side_effect=fake_complete):
            reply, provider, decision = await run_pod4_turn(session, "I'm from Lucknow, moved for work")
        self.assertEqual(reply, "Lucknow, nice. What brought you to Ahmedabad?")
        self.assertIn("location", session.pod4_state.goals["life_context"].slots)
        self.assertIn("Private guidance for this reply", prompts[1])  # the Conductor ran before the Talker
        self.assertEqual(decision.goal, "life_context")

    async def test_observation_without_llm_is_unknown_not_a_dodge(self) -> None:
        session = CallSession("pod4-test-2", "english", conversation_pod="pod4")

        async def down(*_, **__):
            raise RuntimeError("no llm")

        with patch.object(pod4_providers, "complete", side_effect=down):
            observation = await observe_turn(session, "honestly walks and music help me unwind")
            skipped = await observe_turn(session, "skip this")
        self.assertEqual(observation.source, "unknown")
        self.assertEqual(skipped.outcome, "skip")

    async def test_first_stage_is_asked_on_the_first_turn(self) -> None:
        state, decision = await step(Pod4ConversationState(seed=1), "it's going great", chat(), elapsed=10)
        self.assertEqual(decision.response_mode, "ask_goal")
        self.assertEqual(decision.goal, POD4_GOAL_ORDER[0])

    async def test_stages_follow_the_blueprint_order(self) -> None:
        state, decision = await step(Pod4ConversationState(seed=7), "hi", chat())
        asked = [decision.goal]
        for _ in range(3):
            text, observation = answer_for(decision.goal)
            state, decision = await step(state, text, observation)
            state, decision = await past_conversation_turns(state, decision)
            asked.append(decision.goal)
        # relationship_history is sensitive and waits for 4 user turns; the conversational turns get there in time.
        self.assertEqual(asked, list(POD4_GOAL_ORDER[:4]))

    async def test_partial_answer_stays_on_the_stage(self) -> None:
        state, decision = await step(Pod4ConversationState(seed=4), "hi", chat())
        first = decision.goal
        text, observation = answer_for(first, GOAL_SLOTS[first][:1])
        state, decision = await step(state, text, observation)
        self.assertEqual(decision.goal, first)
        self.assertEqual(decision.planning_reason, "continue_stage")
        self.assertEqual(decision.missing, [slot.ask_for for slot in POD4_BLUEPRINT.goal(first).slots[1:]])

    async def test_conversation_turn_after_two_questions_in_a_row(self) -> None:
        state, decision = await step(Pod4ConversationState(seed=4), "hi", chat())
        first = decision.goal
        text, observation = answer_for(first, GOAL_SLOTS[first][:1])
        state, decision = await step(state, text, observation)
        self.assertTrue(decision.ask_question)
        state, decision = await step(state, "haha nice", chat())
        self.assertEqual(decision.planning_reason, "conversation_layer")
        self.assertFalse(decision.ask_question)
        # Next turn softly comes back to the stage; the talk turn didn't count against it.
        asks = state.goals[first].asks
        state, decision = await step(state, "yeah it's been fun", chat())
        self.assertEqual(decision.goal, first)
        self.assertEqual(decision.planning_reason, "return_to_open_stage")
        self.assertEqual(state.goals[first].asks, asks)

    async def test_heavy_tone_gets_a_turn_with_no_new_question(self) -> None:
        state, decision = await step(Pod4ConversationState(seed=2), "hi", chat())
        low = Pod4TurnObservation(intent="answer", tone_cue="sad", tone_confidence=0.9, outcome="not_applicable")
        state, decision = await step(state, "honestly i don't have anyone here, it's lonely", low)
        self.assertEqual(decision.planning_reason, "pause_heavy_tone")
        self.assertFalse(decision.ask_question)

    async def test_full_answer_gets_a_follow_up_before_moving_on(self) -> None:
        state, decision = await step(Pod4ConversationState(seed=3), "hi", chat())
        text, observation = answer_for(decision.goal, intent="answer")
        state, decision = await step(state, text, observation)
        self.assertEqual(decision.response_mode, "follow_up")
        self.assertEqual(decision.goal, POD4_GOAL_ORDER[0])

    async def test_soft_skip_parks_then_comes_back_then_second_skip_drops(self) -> None:
        state, decision = await step(Pod4ConversationState(seed=5), "hi", chat())
        first = decision.goal
        state, decision = await step(state, "skip this", skip())
        self.assertEqual(state.goals[first].status, "parked")
        self.assertEqual(decision.planning_reason, "skip_parked")
        # Close two other stages; then the parked stage comes back as a revisit.
        for _ in range(2):
            state, decision = await past_conversation_turns(state, decision)
            text, observation = answer_for(decision.goal)
            state, decision = await step(state, text, observation)
        state, decision = await past_conversation_turns(state, decision)
        self.assertEqual(decision.goal, first)
        self.assertEqual(decision.ask_kind, "revisit")
        state, decision = await step(state, "skip", skip())
        self.assertEqual(state.goals[first].status, "skipped")

    async def test_hard_decline_drops_a_stage_for_good(self) -> None:
        self.assertTrue(hard_decline("I don't want to talk about that"))
        self.assertFalse(hard_decline("let's skip this"))
        state, decision = await step(Pod4ConversationState(seed=9), "hi", chat())
        first = decision.goal
        state, decision = await step(state, "I don't want to talk about that", skip())
        self.assertEqual(state.goals[first].status, "skipped")
        self.assertEqual(decision.planning_reason, "skip_respected")

    async def test_dodge_moves_on_then_comes_back_then_drops(self) -> None:
        state, decision = await step(Pod4ConversationState(seed=11), "hi", chat())
        first = decision.goal
        state, decision = await step(state, "hmm", no_answer())
        self.assertEqual(state.goals[first].status, "parked")
        self.assertEqual(decision.planning_reason, "pivot_after_dodge")
        self.assertNotEqual(decision.goal, first)
        # Close two other stages; the dodged one then comes back in a different tone.
        for _ in range(2):
            state, decision = await past_conversation_turns(state, decision)
            text, observation = answer_for(decision.goal)
            state, decision = await step(state, text, observation)
        state, decision = await past_conversation_turns(state, decision)
        self.assertEqual(decision.goal, first)
        state, decision = await step(state, "hmm", no_answer())
        self.assertEqual(state.goals[first].status, "not_covered")

    async def test_backup_topics_fill_in_when_main_stages_are_parked(self) -> None:
        state = Pod4ConversationState(seed=31)
        for goal in POD4_BLUEPRINT.goals:
            if goal.must_cover:
                state.goals[goal.id].status = "parked"
                state.goals[goal.id].revisit_after_closed = 99
        state, decision = await step(state, "hi", chat())
        self.assertFalse(POD4_BLUEPRINT.goal(decision.goal).must_cover)

    async def test_parked_stage_left_at_the_end_is_reported_not_covered(self) -> None:
        state, decision = await step(Pod4ConversationState(seed=13), "hi", chat())
        first = decision.goal
        state, _ = await step(state, "skip", skip())
        report = {row["goal"]: row for row in pod4_coverage_report(state, final=True)}
        self.assertEqual(report[first]["status"], "not_covered")
        self.assertEqual(report[first]["reason"], "skipped once and not answered later")

    def test_blueprint_has_the_question_map_stages(self) -> None:
        self.assertIn("family_parents", POD4_GOAL_ORDER)
        self.assertIn("relationship_priorities", POD4_GOAL_ORDER)
        history = POD4_BLUEPRINT.goal("relationship_history")
        self.assertIn("If you don't mind sharing", history.framing)
        self.assertTrue(all(goal.branches and goal.revive for goal in POD4_BLUEPRINT.goals))

    def test_flat_replies_and_raised_topics_are_recognised(self) -> None:
        self.assertTrue(is_flat_reply("Okay, thanks."))
        self.assertTrue(is_flat_reply("Yeah, yeah, I know."))
        self.assertTrue(is_flat_reply("Yeah, you are right."))
        self.assertTrue(is_flat_reply("That makes sense."))
        self.assertFalse(is_flat_reply("I like to play football."))
        self.assertIn("relationship_history", raised_goals("I had a relationship before, we broke up"))
        self.assertIn("family_parents", raised_goals("my parents are pretty strict"))
        self.assertEqual(raised_goals("I like coding"), [])

    async def test_user_raised_sensitive_topic_is_followed_now_with_framing(self) -> None:
        state, decision = await step(Pod4ConversationState(seed=21), "hi", chat())
        first = decision.goal
        text, observation = answer_for(first, GOAL_SLOTS[first][:1])
        state, decision = await step(state, text + " i had a relationship before, my ex and i broke up", observation)
        self.assertEqual(decision.planning_reason, "follow_user_topic")
        self.assertEqual(decision.goal, "relationship_history")
        self.assertIn("If you don't mind sharing", decision.stage_framing)
        self.assertTrue(decision.branches)
        self.assertEqual(state.goals[first].status, "partial")

    async def test_unknown_turns_never_count_as_a_dodge(self) -> None:
        state, decision = await step(Pod4ConversationState(seed=27), "hi", chat())
        first = decision.goal
        unknown = Pod4TurnObservation(source="unknown", intent="other", outcome="not_applicable")
        for _ in range(3):
            state, decision = await step(state, "honestly walks and music help me a lot", unknown)
        self.assertNotIn(state.goals[first].status, ("parked", "not_covered"))
        self.assertNotEqual(decision.planning_reason, "pivot_after_dodge")

    async def test_flat_reply_marked_skip_by_the_model_is_not_a_skip(self) -> None:
        state, decision = await step(Pod4ConversationState(seed=25), "hi", chat())
        first = decision.goal
        state, decision = await step(state, "Okay, thanks.", skip())
        self.assertNotEqual(state.goals[first].status, "parked")
        self.assertNotEqual(decision.response_mode, "respect_skip")

    async def test_two_flat_replies_make_the_ai_take_the_lead(self) -> None:
        state, decision = await step(Pod4ConversationState(seed=23), "hi", chat())
        state, decision = await step(state, "okay", chat())
        self.assertNotEqual(decision.planning_reason, "revive_flat")
        state, decision = await step(state, "yeah", chat())
        self.assertEqual(decision.response_mode, "revive")
        self.assertTrue(decision.ask_question)
        self.assertTrue(decision.revive_line)
        self.assertEqual(state.flat_streak, 0)

    def test_conductor_imports_no_provider(self) -> None:
        banned = {"pipecat", "google", "httpx", "requests", "ollama", "livekit", "openai"}
        for name in ("app/conversation_pod4.py", "app/pod4_blueprint.py"):
            tree = ast.parse(Path(name).read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                modules = (
                    [alias.name for alias in node.names] if isinstance(node, ast.Import)
                    else [node.module or ""] if isinstance(node, ast.ImportFrom) else []
                )
                for module in modules:
                    self.assertNotIn(module.split(".")[0], banned, f"{name} imports {module}")


if __name__ == "__main__":
    unittest.main()
