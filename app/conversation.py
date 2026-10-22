"""Short-lived in-memory call sessions and the dispatch to the selected conversation pod."""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Literal
from uuid import uuid4

from .config import settings
from .conversation_pods import PodId, get_conversation_pod
from .conversation_pod4 import (
    POD4_BLUEPRINT,
    Pod4ConversationState,
    Pod4Decision,
    Pod4TurnObservation,
    record_pod4_assistant_turn,
)
from .conversation_pod5 import Pod5ConversationState, Pod5Decision, Pod5TurnObservation, record_pod5_assistant_turn
from .models import LanguageMode


def _spoken_only(text: str, *, has_user_turn: bool, mode: LanguageMode) -> str:
    """Keep model meta-commentary from ever reaching Piper or the call transcript."""
    normalized = text.strip()
    lower = normalized.lower()
    meta_phrases = (
        "the user hasn't said", "the user has not said", "the instruction says",
        "instructions say", "we are to", "we must", "i need to", "i should",
        "as the talker", "as an ai language model", "return only", "the prompt",
        "the goal is", "however, the instructions", "let me think",
    )
    if not normalized or len(normalized.split()) > 40 or any(phrase in lower for phrase in meta_phrases):
        if has_user_turn:
            return (
                "Thanks for sharing that. What feels most important to you about it?"
                if mode == "english"
                else "Batane ke liye shukriya. Ismein tumhare liye sabse zaroori kya hai?"
            )
        return (
            "I’m glad we’re here. What does a good weekend look like for you?"
            if mode == "english"
            else "Achha laga tum yahan ho. Tumhare liye ek achha weekend kaisa hota hai?"
        )
    return normalized


@dataclass
class CallSession:
    call_id: str
    language_mode: LanguageMode
    conversation_pod: PodId = "pod4"
    started_at: float = field(default_factory=time.monotonic)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    transcript: list[dict[str, str]] = field(default_factory=list)
    export_transcript: list[dict[str, str]] = field(default_factory=list)
    covered_goals: list[str] = field(default_factory=list)
    rolling_summary: str = ""
    recent_questions: list[str] = field(default_factory=list)
    pod4_state: Pod4ConversationState = field(default_factory=Pod4ConversationState)
    pod5_state: Pod5ConversationState = field(default_factory=Pod5ConversationState)
    latest_user_text: str = ""
    closed: bool = False
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    @property
    def elapsed_seconds(self) -> int:
        return int(time.monotonic() - self.started_at)

    @property
    def phase(self) -> Literal["conversation", "wrap_up", "closed"]:
        max_seconds, wrap_up_seconds = settings.max_call_seconds, settings.wrap_up_seconds
        if self.conversation_pod == "pod4":
            # Pod 4 runs on its own blueprint clock (an 18-minute call).
            clock = POD4_BLUEPRINT.call
            max_seconds = clock.max_call_seconds
            wrap_up_seconds = clock.wrap_up_seconds
        if self.closed or self.elapsed_seconds >= max_seconds:
            return "closed"
        if self.elapsed_seconds >= wrap_up_seconds:
            return "wrap_up"
        return "conversation"


class CallStore:
    """Development-only in-memory sessions; data disappears on process restart."""

    def __init__(self) -> None:
        self._sessions: dict[str, CallSession] = {}

    async def create(
        self, language_mode: LanguageMode, conversation_pod: PodId = "pod4"
    ) -> CallSession:
        session = CallSession(
            call_id=str(uuid4()),
            language_mode=language_mode,
            conversation_pod=conversation_pod,
        )
        self._sessions[session.call_id] = session
        return session

    def get(self, call_id: str) -> CallSession | None:
        return self._sessions.get(call_id)

    def remove(self, call_id: str) -> CallSession | None:
        return self._sessions.pop(call_id, None)


call_store = CallStore()


async def record_voice_transcript(session: CallSession, role: Literal["user", "assistant"], text: str) -> None:
    """Keep Gemini Live captions in the same short-lived session used by call status."""
    cleaned = text.strip()
    if not cleaned:
        return
    async with session.lock:
        turn = {"role": role, "content": cleaned}
        session.transcript.append(turn)
        if settings.export_call_logs:
            session.export_transcript.append(turn.copy())
        if role == "user":
            latest_user_segments: list[str] = []
            for turn in reversed(session.transcript):
                if turn["role"] != "user":
                    break
                latest_user_segments.append(turn["content"])
            session.latest_user_text = " ".join(reversed(latest_user_segments))[-1200:]
        elif "?" in cleaned:
            question = cleaned.rsplit("?", 1)[0].split(".")[-1].strip() + "?"
            if question not in session.recent_questions:
                session.recent_questions.append(question)
                session.recent_questions = session.recent_questions[-10:]
        if role == "assistant" and session.conversation_pod == "pod5":
            # Pod 5 tracks reply openers so they are not reused (VOICE-3).
            record_pod5_assistant_turn(session.pod5_state, cleaned)
        if role == "assistant" and session.conversation_pod == "pod4":
            record_pod4_assistant_turn(session.pod4_state, cleaned)
        if len(session.transcript) > 12:
            removed = session.transcript[:-12]
            session.transcript = session.transcript[-12:]
            user_topics = [turn["content"][:180] for turn in removed if turn["role"] == "user"]
            if user_topics:
                previous = session.rolling_summary.removeprefix("User explicitly shared: ")
                summary_parts = ([previous] if previous else []) + user_topics[-4:]
                session.rolling_summary = "User explicitly shared: " + " | ".join(summary_parts)[-1200:]


async def update_live_conversation(
    session: CallSession,
    observation: Pod4TurnObservation | Pod5TurnObservation,
) -> Pod4Decision | Pod5Decision:
    """Dispatch through the selected pod's turn handler."""
    pod = get_conversation_pod(session.conversation_pod)
    return await pod.handle_turn(session, observation)


async def generate_turn(session: CallSession, user_text: str | None = None) -> tuple[str, str]:
    if session.conversation_pod == "pod5":
        from .conversation_pod5_text import generate_pod5_text_turn

        return await generate_pod5_text_turn(session, user_text)
    from .conversation_pod4_text import generate_pod4_text_turn

    return await generate_pod4_text_turn(session, user_text)
