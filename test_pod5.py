"""Tests for Pod 5, the blueprint-driven Kismet Conductor.

Each test names the requirement IDs it checks from
"Kismet AI Call: Conversation Requirements".
"""

from __future__ import annotations

import ast
import random
import unittest
from pathlib import Path

from app.conversation import CallSession, record_voice_transcript, update_live_conversation
from app.conversation_pod5 import (
    POD5_BLUEPRINT,
    Pod5ConversationState,
    Pod5Decision,
    Pod5TurnObservation,
    plan_pod5_turn,
    pod5_coverage_report,
)
from app.conversation_pods import get_conversation_pod
from app.pod5_blueprint import load_blueprint
from app.pod5_rules import code_outcome, distress_quote
from app.voice_agent import _live_conversation_tools

CORE = 200  # seconds: a time inside the core segment, after rapport is allowed
GOAL_SLOTS = {goal.id: [slot.id for slot in goal.slots] for goal in POD5_BLUEPRINT.goals}


def answer_for(goal_id: str, slots: list[str] | None = None) -> tuple[str, Pod5TurnObservation]:
    """A user reply that states the given slots of a goal, with grounded quotes."""
    slots = slots if slots is not None else GOAL_SLOTS[goal_id]
    pieces = [f"about {slot.replace('_', ' ')} i would say yes" for slot in slots]
    text = ". ".join(pieces) + "."
    fills = [
        {"slot": f"{goal_id}.{slot}", "quote": piece, "value": piece}
        for slot, piece in zip(slots, pieces)
    ]
    outcome = "full" if len(slots) == len(GOAL_SLOTS[goal_id]) else "partial"
    return text, Pod5TurnObservation(intent="answer", outcome=outcome, outcome_confidence=0.9, slots_filled=fills)


async def step(
    state: Pod5ConversationState, text: str, observation: Pod5TurnObservation, elapsed: int = CORE
) -> tuple[Pod5ConversationState, Pod5Decision]:
    return await plan_pod5_turn(state, observation, user_text=text, elapsed_seconds=elapsed)


async def past_casual_turns(
    state: Pod5ConversationState, decision: Pod5Decision
) -> tuple[Pod5ConversationState, Pod5Decision]:
    """Reply with small talk through breather and detour turns until a goal is asked."""
    for _ in range(6):
        if decision.goal and decision.response_mode not in ("detour",):
            break
        state, decision = await step(state, "haha nice", Pod5TurnObservation(intent="small_talk", outcome="off_topic", outcome_confidence=0.9))
    return state, decision


class Pod5RuleTests(unittest.TestCase):
    def test_blueprint_is_valid_and_complete(self) -> None:  # TOP-4, TOP-6
        blueprint = load_blueprint()
        self.assertEqual(len(blueprint.goals), 8)
        self.assertEqual(sum(goal.must_cover for goal in blueprint.goals), 7)
        past = blueprint.goal("past_lessons")
        self.assertFalse(past.must_cover)
        self.assertTrue(past.skippable and past.sensitive)
        self.assertTrue(blueprint.goal("family_future").sensitive)
        for goal in blueprint.goals:
            self.assertTrue(4 <= len(goal.angles) <= 6, goal.id)
            self.assertGreaterEqual(len(goal.slots), 1, goal.id)

    def test_code_classifies_obvious_phrases_first(self) -> None:  # RETRY-1
        for phrase in ("skip", "next", "baad mein", "chhodo yaar", "next question please", "pass"):
            self.assertEqual(code_outcome(phrase), "skip", phrase)
        for phrase in ("pata nahi", "I don't know", "no idea yaar", "pata nahi yaar"):
            self.assertEqual(code_outcome(phrase), "no_answer", phrase)
        long_answer = "Next weekend I am not sure yet but usually I go hiking with my friends and cook"
        self.assertIsNone(code_outcome(long_answer))

    def test_distress_phrases_in_both_languages(self) -> None:  # SAFE-2
        self.assertTrue(distress_quote("Honestly sometimes I want to die"))
        self.assertTrue(distress_quote("kabhi kabhi lagta hai marna chahta hoon"))
        self.assertTrue(distress_quote("he hits me when he is angry"))
        self.assertFalse(distress_quote("this traffic is killing me lol"))

    def test_conductor_modules_never_import_providers(self) -> None:  # PROV-5
        for name in ("conversation_pod5.py", "pod5_rules.py", "pod5_blueprint.py"):
            tree = ast.parse((Path(__file__).parent / "app" / name).read_text())
            imported = {
                (node.module or "") for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)
            } | {alias.name for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names}
            self.assertFalse(any("provider" in module for module in imported), name)

    def test_pod5_is_registered_with_its_own_tool(self) -> None:
        pod = get_conversation_pod("pod5")
        self.assertIn("POD 5 CONDUCTOR", pod.prompt_addendum)
        tool = _live_conversation_tools("pod5").standard_tools[0]
        self.assertIn("slots_filled", tool.properties)
        self.assertIn("distress", tool.properties)


class Pod5ConductorTests(unittest.IsolatedAsyncioTestCase):
    async def test_warm_up_asks_no_goal(self) -> None:  # TIME-3
        state, decision = await step(Pod5ConversationState(seed=1), "hi, just got home", Pod5TurnObservation(intent="small_talk"), 20)
        self.assertEqual(decision.response_mode, "warm_up")
        self.assertIsNone(decision.goal)
        self.assertEqual(state.timeline[0].segment, "warm_up")

    async def test_core_asks_a_goal_for_its_missing_slots(self) -> None:
        state, decision = await step(Pod5ConversationState(seed=1), "it was fine", Pod5TurnObservation(intent="small_talk"))
        self.assertEqual(decision.response_mode, "ask_goal")
        self.assertEqual(len(decision.missing), len(GOAL_SLOTS[decision.goal]))
        self.assertEqual(state.goals[decision.goal].asks, 1)
        self.assertEqual(state.active_goal, decision.goal)

    async def test_full_answer_covers_and_ungrounded_quotes_are_ignored(self) -> None:  # TOP-1, SAFE-5
        state, decision = await step(Pod5ConversationState(seed=2), "hey", Pod5TurnObservation())
        goal = decision.goal
        made_up = Pod5TurnObservation(
            outcome="full",
            outcome_confidence=0.9,
            slots_filled=[{"slot": f"{goal}.{slot}", "quote": "words never said", "value": "x"} for slot in GOAL_SLOTS[goal]],
        )
        state, _ = await step(state, "something else entirely here", made_up)
        self.assertNotEqual(state.goals[goal].status, "covered")
        state2, decision2 = await step(Pod5ConversationState(seed=2), "hey", Pod5TurnObservation())
        text, observation = answer_for(decision2.goal)
        state2, _ = await step(state2, text, observation)
        self.assertEqual(state2.goals[decision2.goal].status, "covered")

    async def test_unprompted_answer_counts_as_covered(self) -> None:  # TOP-2
        state, decision = await step(Pod5ConversationState(seed=3), "hey", Pod5TurnObservation())
        other = next(goal for goal in ("conflict", "values_priorities") if goal != decision.goal)
        text, observation = answer_for(other)
        state, _ = await step(state, text, observation)
        self.assertEqual(state.goals[other].status, "covered")

    async def test_partial_comes_back_after_two_goals_then_drops_after_three_asks(self) -> None:  # RETRY-4, section 7
        state, decision = await step(Pod5ConversationState(seed=4), "hey", Pod5TurnObservation())
        target = decision.goal
        first_slot = GOAL_SLOTS[target][0]
        text, observation = answer_for(target, [first_slot])
        state, decision = await step(state, text, observation)
        self.assertEqual(state.goals[target].status, "partial")
        self.assertNotEqual(decision.goal, target)
        # Two other goals close, then the revisit is due and asks only for the missing part.
        for _ in range(2):
            state, decision = await past_casual_turns(state, decision)
            text, observation = answer_for(decision.goal)
            state, decision = await step(state, text, observation)
        state, decision = await past_casual_turns(state, decision)
        self.assertEqual(decision.goal, target)
        self.assertEqual(decision.ask_kind, "revisit")
        self.assertEqual(len(decision.missing), len(GOAL_SLOTS[target]) - 1)
        vague = Pod5TurnObservation(outcome="partial", outcome_confidence=0.9)
        for _ in range(4):
            if state.goals[target].status == "not_covered":
                break
            state, decision = await step(state, "hmm maybe", vague)
            while decision.goal != target and state.goals[target].status == "partial":
                text, observation = answer_for(decision.goal) if decision.goal else ("ok", Pod5TurnObservation())
                state, decision = await step(state, text, observation)
        self.assertEqual(state.goals[target].status, "not_covered")
        self.assertLessEqual(state.goals[target].asks, POD5_BLUEPRINT.retries.partial_max_asks)

    async def test_no_answer_retries_with_new_angle_then_drops(self) -> None:  # section 7, RETRY-3
        state, decision = await step(Pod5ConversationState(seed=5), "hey", Pod5TurnObservation())
        goal, first_angle = decision.goal, decision.angle
        state, decision = await step(state, "pata nahi", Pod5TurnObservation(outcome="no_answer", outcome_confidence=0.9))
        self.assertEqual(decision.response_mode, "retry")
        self.assertEqual(decision.goal, goal)
        self.assertNotEqual(decision.angle, first_angle)
        state, decision = await step(state, "pata nahi", Pod5TurnObservation(outcome="no_answer", outcome_confidence=0.9))
        self.assertEqual(state.goals[goal].status, "not_covered")
        self.assertEqual(state.goals[goal].asks, POD5_BLUEPRINT.retries.no_answer_max_asks)
        self.assertNotEqual(decision.goal, goal)

    async def test_skip_is_never_asked_again(self) -> None:  # RETRY-8
        state, decision = await step(Pod5ConversationState(seed=6), "hey", Pod5TurnObservation())
        skipped = decision.goal
        state, decision = await step(state, "skip", Pod5TurnObservation(outcome="full", outcome_confidence=0.9))
        self.assertEqual(state.goals[skipped].status, "skipped")
        self.assertEqual(decision.response_mode, "respect_skip")
        for turn in range(30):
            text, observation = answer_for(decision.goal) if decision.goal else ("sure", Pod5TurnObservation())
            state, decision = await step(state, text, observation, CORE + turn * 10)
            self.assertNotEqual(decision.goal, skipped)

    async def test_unsure_classification_counts_as_partial(self) -> None:  # RETRY-2
        state, decision = await step(Pod5ConversationState(seed=7), "hey", Pod5TurnObservation())
        state, _ = await step(state, "kind of not really", Pod5TurnObservation(outcome="no_answer", outcome_confidence=0.3))
        self.assertEqual(state.goals[decision.goal].status, "partial")

    async def test_distress_starts_the_safety_path_and_stays(self) -> None:  # SAFE-2
        state, decision = await step(Pod5ConversationState(seed=8), "hey", Pod5TurnObservation())
        state, decision = await step(state, "honestly I want to die these days", Pod5TurnObservation())
        self.assertEqual(decision.response_mode, "safety")
        self.assertTrue(decision.support_resources)
        self.assertFalse(decision.ask_question)
        state, decision = await step(state, "anyway what next", Pod5TurnObservation(intent="small_talk"))
        self.assertEqual(decision.response_mode, "safety")

    async def test_model_distress_flag_needs_a_real_quote(self) -> None:  # SAFE-2, SAFE-5
        state, decision = await step(
            Pod5ConversationState(seed=8), "work is so hard lately",
            Pod5TurnObservation(distress=True, distress_quote="i give up on everything"),
        )
        self.assertNotEqual(decision.response_mode, "safety")

    async def test_sensitive_goals_wait_for_rapport(self) -> None:  # TOP-5
        for seed in range(1, 30):
            state = Pod5ConversationState(seed=seed)
            _, decision = await step(state, "hey", Pod5TurnObservation(), 70)
            self.assertNotIn(decision.goal, ("family_future", "past_lessons"))

    async def test_asr_repair_is_limited(self) -> None:  # VOICE-8
        state, _ = await step(Pod5ConversationState(seed=9), "hey", Pod5TurnObservation())
        unclear = Pod5TurnObservation(asr_unclear=True)
        modes = []
        for _ in range(3):
            state, decision = await step(state, "mmhmm", unclear)
            modes.append(decision.response_mode)
        self.assertEqual(modes[:2], ["repair", "repair"])
        self.assertNotEqual(modes[2], "repair")

    async def test_same_seed_and_answers_replay_the_same_plan(self) -> None:  # RAND-1
        async def run(seed: int) -> list[tuple[str, str | None, str | None, str | None]]:
            state = Pod5ConversationState(seed=seed)
            plan = []
            decision = Pod5Decision()
            for turn in range(25):
                text, observation = answer_for(decision.goal) if decision.goal else ("nice", Pod5TurnObservation(intent="small_talk"))
                state, decision = await step(state, text, observation, 60 + turn * 30)
                plan.append((decision.response_mode, decision.goal, decision.angle, decision.filler_type))
            return plan

        self.assertEqual(await run(42), await run(42))
        orders = {tuple(goal for _, goal, _, _ in await run(seed) if goal) for seed in range(20)}
        self.assertGreater(len(orders), 5)  # TOP-3: different seeds vary the order

    async def test_wrap_up_last_chance_then_close(self) -> None:  # RETRY-6, TIME-1, TIME-2
        state, decision = await step(Pod5ConversationState(seed=10), "hey", Pod5TurnObservation())
        state, decision = await step(state, "nice", Pod5TurnObservation(intent="small_talk"), 850)
        self.assertEqual(decision.response_mode, "last_chance")
        goal = decision.goal
        state, decision = await step(state, "hmm", Pod5TurnObservation(outcome="partial", outcome_confidence=0.9), 860)
        self.assertEqual(state.goals[goal].status, "not_covered")
        state, decision = await step(state, "ok", Pod5TurnObservation(), 970)
        self.assertEqual(decision.response_mode, "close")
        self.assertFalse(decision.ask_question)
        report = pod5_coverage_report(state, final=True)
        for row in report:
            self.assertIn(row["status"], ("covered", "not_covered", "skipped", "not_asked"))
            if row["status"] == "not_covered":
                self.assertTrue(row["reason"])


class Pod5SimulatedCallTests(unittest.IsolatedAsyncioTestCase):
    async def test_simulated_callers_respect_every_limit(self) -> None:  # DET-1..4, FILL-2, TOP-1, RETRY limits
        styles = ("talkative", "one_word", "half", "off_topic", "skipper")
        for seed in range(40):
            caller = random.Random(seed)
            style = styles[seed % len(styles)]
            state = Pod5ConversationState(seed=seed)
            decision = Pod5Decision()
            detour_turns_in_a_row = 0
            elapsed = 0
            while elapsed < 1020:
                elapsed += caller.randint(15, 35)
                goal = decision.goal if decision.ask_question else None
                if goal and style == "talkative":
                    text, observation = answer_for(goal)
                elif goal and style == "half":
                    text, observation = answer_for(goal, GOAL_SLOTS[goal][:1]) if caller.random() < 0.7 else answer_for(goal)
                elif goal and style == "one_word":
                    text, observation = "pata nahi", Pod5TurnObservation(outcome="no_answer", outcome_confidence=0.9)
                elif goal and style == "skipper" and caller.random() < 0.4:
                    text, observation = "skip", Pod5TurnObservation()
                elif goal and style == "off_topic" and caller.random() < 0.5:
                    text, observation = "did you see the match yesterday", Pod5TurnObservation(outcome="off_topic", outcome_confidence=0.9)
                elif goal:
                    text, observation = answer_for(goal)
                else:
                    text, observation = "haha yeah", Pod5TurnObservation(intent="small_talk")
                state, decision = await step(state, text, observation, elapsed)

                if decision.response_mode == "detour":
                    detour_turns_in_a_row += 1
                    self.assertGreaterEqual(elapsed, POD5_BLUEPRINT.detours.earliest_seconds)
                    self.assertLess(elapsed, POD5_BLUEPRINT.call.wrap_up_seconds)
                    self.assertLessEqual(detour_turns_in_a_row, POD5_BLUEPRINT.detours.max_turns)
                    pending = state.detour.pending_goal if state.detour else None
                    if pending:
                        self.assertFalse(POD5_BLUEPRINT.goal(pending).sensitive)
                else:
                    detour_turns_in_a_row = 0
                if elapsed >= POD5_BLUEPRINT.call.close_seconds:
                    self.assertIn(decision.response_mode, ("close", "safety"))
            self.assertLessEqual(state.detours_used, POD5_BLUEPRINT.detours.max_per_call)
            types = state.recent_filler_types
            for previous, current in zip(types, types[1:]):
                self.assertNotEqual(previous, current, f"seed {seed}")
            for goal in POD5_BLUEPRINT.goals:
                progress = state.goals[goal.id]
                self.assertLessEqual(progress.asks, POD5_BLUEPRINT.retries.partial_max_asks, f"seed {seed} {goal.id}")
            for row in pod5_coverage_report(state, final=True):
                if row["must_cover"]:
                    self.assertIn(row["status"], ("covered", "not_covered", "skipped"), f"seed {seed}")


class Pod5SessionTests(unittest.IsolatedAsyncioTestCase):
    async def test_session_dispatch_and_opener_tracking(self) -> None:  # VOICE-3
        session = CallSession("pod5-call", "english", conversation_pod="pod5")
        await record_voice_transcript(session, "assistant", "Arre wah, that sounds lovely. What do you do on Sundays?")
        await record_voice_transcript(session, "assistant", "Arre wah, that is fun!")
        self.assertEqual(session.pod5_state.opener_repeats, 1)
        await record_voice_transcript(session, "user", "I like slow Sundays with chai")
        decision = await update_live_conversation(session, Pod5TurnObservation(intent="small_talk"))
        self.assertIsInstance(decision, Pod5Decision)
        self.assertIn("arre wah that", decision.must_avoid_openers)


if __name__ == "__main__":
    unittest.main()
