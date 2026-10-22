"""Pod 5 text turn: the local Whisper/Ollama/Piper fallback and the /turn text harness.

The Conductor in `conversation_pod5.py` must never import a provider (PROV-5),
so the LLM calls for this text path live here. Like Pod 3, it first asks a
small model for a JSON observation, falls back to code rules if that fails,
runs the same Conductor, then asks the model to word the reply.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

from loguru import logger

from .conversation_pod5 import (
    POD5_BLUEPRINT,
    POD5_GOAL_LABELS,
    Pod5Decision,
    Pod5TurnObservation,
)
from .pod5_rules import code_outcome, distress_quote
from .providers import ChatMessage, provider_router

if TYPE_CHECKING:
    from .conversation import CallSession


def _slot_list() -> str:
    return "\n".join(
        f"- {goal.slot_key(slot.id)}: {slot.ask_for}" for goal in POD5_BLUEPRINT.goals for slot in goal.slots
    )


def _fallback_observation(user_text: str) -> Pod5TurnObservation:
    """Conservative code-only observation when the model returns no valid JSON."""
    outcome = code_outcome(user_text)
    quote = distress_quote(user_text)
    return Pod5TurnObservation(
        intent="skip" if outcome == "skip" else "answer",
        outcome=outcome or "partial",
        outcome_confidence=0.9 if outcome else 0.5,
        distress=bool(quote),
        distress_quote=quote,
    )


async def _observe_text_turn(session: "CallSession", user_text: str) -> Pod5TurnObservation:
    state = session.pod5_state
    pending = (
        f"The pending question is about {POD5_GOAL_LABELS[state.active_goal]}. "
        if state.active_goal
        else "No goal question is pending; set outcome to not_applicable. "
    )
    messages = [
        ChatMessage(
            "system",
            "Classify the latest caller turn of a casual dating-compatibility call. Return only a JSON object with "
            "keys intent, tone_cue, tone_confidence, outcome, outcome_confidence, slots_filled, distress, "
            "distress_quote, asr_unclear. intent is small_talk, user_question, answer, story, correction, skip, "
            "or other. outcome is full, partial, no_answer, off_topic, skip or not_applicable for the pending "
            "question. slots_filled is a list of {slot, quote, value}: slot must be one of the ids below, quote an "
            "exact short quote from the latest turn, value a short English summary of what they stated. Never infer "
            "facts they did not say. distress is true only for signs of self-harm or abuse. " + pending +
            "Slots:\n" + _slot_list(),
        ),
        ChatMessage("user", user_text),
    ]
    try:
        raw, provider = await provider_router.complete(messages, temperature=0.1)
        start, end = raw.find("{"), raw.rfind("}")
        if start < 0 or end < start:
            raise ValueError("No JSON object in observation response")
        observation = Pod5TurnObservation.model_validate(json.loads(raw[start : end + 1]))
        logger.info(
            "POD5 TEXT OBSERVATION | call={} | provider={} | outcome={} | slots={}",
            session.call_id,
            provider,
            observation.outcome,
            [fill.slot for fill in observation.slots_filled],
        )
        return observation
    except Exception as exc:
        logger.warning("POD5 TEXT OBSERVATION FALLBACK | call={} | {}", session.call_id, exc)
        return _fallback_observation(user_text)


async def generate_pod5_text_turn(session: "CallSession", user_text: str | None = None) -> tuple[str, str]:
    from .conversation import _spoken_only, record_voice_transcript, update_live_conversation

    if user_text:
        await record_voice_transcript(session, "user", user_text)
        observation = await _observe_text_turn(session, user_text)
        decision = await update_live_conversation(session, observation)
    else:
        decision = Pod5Decision(
            response_mode="warm_up",
            ask_question=True,
            cue="Greet them, say briefly that you're Aria, an AI, and start easy small talk about their day.",
            planning_reason="opening_turn",
        )
    assert isinstance(decision, Pod5Decision)

    mode = session.language_mode
    persona_file = "persona_hinglish.md" if mode == "hinglish" else "persona_english.md"
    try:
        persona = (Path(__file__).parent.parent / "prompts" / persona_file).read_text(encoding="utf-8").strip()
    except OSError:
        persona = "Be warm, respectful, curious, concise, and natural."
    prompt = (
        f"{persona}\n\nYou are Aria, an AI companion on a casual call. Say only the next spoken turn: 1 to 2 short "
        "sentences, about 25 words, no lists. Start with a reaction to what they said, then at most one question. "
        "Never give advice, labels or opinions about the user. Be honest that you're an AI if asked.\n"
        f"Language: {mode}.\nPrivate guidance (never say it aloud): "
        f"{json.dumps(decision.model_dump(mode='json', exclude={'planning_reason'}), ensure_ascii=False)}\n"
        f"Recent conversation: {json.dumps(session.transcript[-6:], ensure_ascii=False)}"
    )
    messages = [ChatMessage("system", prompt)]
    if user_text:
        messages.append(ChatMessage("user", user_text))
    try:
        raw_reply, provider = await provider_router.complete(messages, temperature=0.7)
        reply = _spoken_only(raw_reply, has_user_turn=bool(user_text), mode=mode)
    except Exception:
        logger.exception("POD5 TEXT TALKER FAILED | call={}", session.call_id)
        reply = (
            "I hear you. What feels most important to you about that?"
            if mode == "english"
            else "Achha, theek hai. Ismein tumhare liye sabse zyada kya matter karta hai?"
        )
        provider = "pod5-talker-fallback"
    await record_voice_transcript(session, "assistant", reply)
    if session.phase == "closed":
        session.closed = True
    return reply, provider
