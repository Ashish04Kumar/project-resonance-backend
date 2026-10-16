"""LangGraph text conversation flow and short-lived in-memory call sessions."""

from __future__ import annotations

import asyncio
import json
import random
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Literal, TypedDict
from uuid import uuid4

from langgraph.graph import END, START, StateGraph

from .config import settings
from .conversation_pods import PodId, get_conversation_pod
from .conversation_schemas import (
    ConductorDecision,
    ConversationState,
    QuestionAngle,
    RelationshipTopic,
    ToneCue,
    TurnObservation,
    TurnIntent,
)
from .models import LanguageMode
from .providers import ChatMessage, provider_router


GOALS = (
    ("weekend_downtime", "weekend and downtime"),
    ("conflict", "conflict and repair"),
    ("partner_wants", "what they want in a partner"),
    ("family_future", "family and future hopes"),
    ("values_priorities", "values and priorities"),
    ("communication_affection", "communication and affection"),
    ("real_life_scenario", "a real-life relationship scenario"),
)
GOAL_IDS = tuple(goal[0] for goal in GOALS)
GOAL_LABELS = dict(GOALS)
OPTIONAL_GOAL = "past_relationship_lessons"

LIVE_GOALS: tuple[RelationshipTopic, ...] = (
    "weekend_downtime",
    "communication_affection",
    "conflict",
    "partner_wants",
    "values_priorities",
    "family_future",
    "real_life_scenario",
)
LIVE_GOAL_LABELS: dict[RelationshipTopic, str] = {
    "weekend_downtime": "everyday life and downtime",
    "communication_affection": "communication and affection",
    "conflict": "handling disagreements and repairing things",
    "partner_wants": "what they want in a relationship",
    "values_priorities": "values and priorities",
    "family_future": "family and future hopes",
    "real_life_scenario": "a practical relationship situation",
}


class FlowState(TypedDict, total=False):
    language_mode: LanguageMode
    transcript: list[dict[str, str]]
    rolling_summary: str
    recent_questions: list[str]
    user_text: str
    elapsed_seconds: int
    covered_goals: list[str]
    current_goal: str
    next_goal: str
    acknowledgement: str
    assistant_text: str
    provider: str
    phase: Literal["conversation", "wrap_up", "closed"]


class LiveConductorState(TypedDict, total=False):
    conversation_state: ConversationState
    observation: TurnObservation
    user_text: str
    elapsed_seconds: int
    recent_questions: list[str]
    decision: ConductorDecision


def _normalized_quote(value: str) -> str:
    return " ".join(value.casefold().split()).strip(" \t\n.,!?\"'“”‘’")


async def _merge_live_observation(state: LiveConductorState) -> LiveConductorState:
    conversation = state["conversation_state"].model_copy(deep=True)
    observation = state["observation"]
    user_text = state.get("user_text", "")
    normalized_user = _normalized_quote(user_text)
    conversation.turn_count += 1
    conversation.turns_since_topic += 1
    conversation.last_tone_cue = observation.tone_cue
    conversation.last_tone_confidence = observation.tone_confidence

    topics_to_skip = list(observation.skip_topics)
    if observation.intent == "skip" and not topics_to_skip and conversation.last_goal:
        topics_to_skip.append(conversation.last_goal)
    for topic in topics_to_skip:
        progress = conversation.topics[topic]
        progress.status = "skipped"

    for signal in observation.topic_signals:
        quote = _normalized_quote(signal.evidence)
        # The model may only persist text that is verifiably present in this turn.
        if not quote or quote not in normalized_user:
            continue
        progress = conversation.topics[signal.topic]
        if progress.status == "skipped":
            continue
        progress.evidence = (progress.evidence + [signal.evidence])[-4:]
        if signal.depth == "clear" and signal.confidence >= 0.65:
            progress.status = "covered"
        elif progress.status == "new":
            progress.status = "opened"

    for preference in observation.explicit_preferences:
        quote = _normalized_quote(preference)
        if (
            quote
            and len(preference) <= 180
            and quote in normalized_user
            and preference not in conversation.explicit_preferences
        ):
            conversation.explicit_preferences = (conversation.explicit_preferences + [preference])[-12:]

    return {"conversation_state": conversation}


def _next_live_angle(progress) -> QuestionAngle:
    angles: tuple[QuestionAngle, ...] = (
        "recent_story",
        "everyday_preference",
        "specific_example",
        "light_hypothetical",
    )
    recent = progress.angles_used[-2:]
    return next((angle for angle in angles if angle not in recent), angles[0])


async def _choose_live_decision(state: LiveConductorState) -> LiveConductorState:
    conversation = state["conversation_state"].model_copy(deep=True)
    observation = state["observation"]
    elapsed = state.get("elapsed_seconds", 0)
    recent_questions = state.get("recent_questions", [])[-3:]
    tone: ToneCue = (
        observation.tone_cue
        if observation.tone_confidence >= 0.65
        else "neutral"
    )

    mode = "chat"
    goal: RelationshipTopic | None = None
    angle: QuestionAngle | None = None
    depth: Literal["light", "follow_up"] = "light"
    ask_question = False
    must_avoid = list(recent_questions)
    cue = "Stay with the user's thread. React naturally; a question is optional."

    if elapsed >= settings.wrap_up_seconds:
        mode = "wrap_up"
        cue = "Gently bring the call toward a close. Reflect one real detail; don't introduce a new topic."
    elif observation.intent == "user_question":
        mode = "answer_user"
        cue = "Answer the user's question directly and naturally. Don't tack on a compatibility question."
    elif observation.intent == "correction":
        mode = "acknowledge_correction"
        cue = "Accept the correction simply, adjust, and continue without defending the earlier assumption."
    elif observation.intent == "skip" or observation.skip_topics:
        mode = "respect_skip"
        skipped = [LIVE_GOAL_LABELS[item] for item in observation.skip_topics]
        must_avoid.extend(skipped)
        cue = "Respect the skip without probing or immediately replacing it with another personal question."
    elif tone in ("sad", "frustrated", "uncertain") and observation.tone_confidence >= 0.78:
        mode = "reflect"
        cue = "Respond to the specific thing they said with calm care. Don't diagnose, force optimism, or turn it into an interview."
    elif observation.intent in ("small_talk", "other"):
        mode = "chat"
        cue = "Have an easy back-and-forth about what they brought up. Don't steer to a relationship topic yet."
    else:
        mode = "chat"
        signal = next(
            (
                item
                for item in reversed(observation.topic_signals)
                if item.topic in conversation.topics
                and conversation.topics[item.topic].status not in ("covered", "skipped")
            ),
            None,
        )
        if signal and conversation.topics[signal.topic].evidence:
            goal = signal.topic
            depth = "follow_up"
            cue = "Stay with the story they just told; respond first, then follow one useful thread if it feels natural."
        elif conversation.turn_count >= 3 and conversation.turns_since_topic >= 2:
            goal = next(
                (
                    topic
                    for topic in LIVE_GOALS
                    if conversation.topics[topic].status == "new"
                ),
                None,
            )
            if goal:
                mode = "explore"
                cue = (
                    f"After a natural reaction, gently open the topic of {LIVE_GOAL_LABELS[goal]} "
                    "using the suggested angle only if it fits. Don't announce a topic or ask it like a form."
                )
        if goal and mode == "chat":
            mode = "explore"

        if goal:
            angle = _next_live_angle(conversation.topics[goal])
            conversation.topics[goal].angles_used = (
                conversation.topics[goal].angles_used + [angle]
            )[-4:]
            conversation.last_goal = goal
            conversation.turns_since_topic = 0
            ask_question = True
            depth = "follow_up" if signal else "light"
            must_avoid.append(f"Do not repeat an angle already used for {LIVE_GOAL_LABELS[goal]}.")

    decision = ConductorDecision(
        response_mode=mode,
        goal=goal,
        angle=angle,
        depth=depth,
        tone_hint=tone,
        must_avoid=must_avoid[-8:],
        ask_question=ask_question,
        cue=cue,
    )
    return {"conversation_state": conversation, "decision": decision}


def _build_live_conductor_graph():
    graph = StateGraph(LiveConductorState)
    graph.add_node("merge_observation", _merge_live_observation)
    graph.add_node("choose_decision", _choose_live_decision)
    graph.add_edge(START, "merge_observation")
    graph.add_edge("merge_observation", "choose_decision")
    graph.add_edge("choose_decision", END)
    return graph.compile()


live_conductor_graph = _build_live_conductor_graph()


def _conductor_prompt(mode: LanguageMode, state: FlowState) -> str:
    goals = "\n".join(f"- {goal_id}: {label}" for goal_id, label in GOALS)
    minutes_left = max(0, (settings.max_call_seconds - state.get("elapsed_seconds", 0)) // 60)
    return f"""You are the Conductor for a private relationship compatibility conversation. You never speak to the user and never write the next question. Return only valid JSON with keys: covered_now (array of goal ids), next_goal (goal id), acknowledgement (short phrase, max 12 words).
Language mode: {mode}. Identify meaning in either English or Hinglish. Required goals:
{goals}
Already covered: {state.get('covered_goals', [])}
Current topic: {state.get('current_goal', '')}
User's latest answer: {state.get('user_text', '')}
Choose the next uncovered goal, adapting naturally to what the user said. Do not infer facts. Past relationships are optional and can be skipped; never select them as a required goal. There are about {minutes_left} minutes left. At 3 minutes or less, choose values_priorities or communication_affection if still uncovered, then wrap up."""


def _talker_prompt(mode: LanguageMode, state: FlowState) -> str:
    persona_path = "persona_hinglish.md" if mode == "hinglish" else "persona_english.md"
    try:
        persona = ( __import__("pathlib").Path(__file__).parent.parent / "prompts" / persona_path).read_text()
    except OSError:
        persona = "Be warm, respectful, curious, and concise."
    recent = state.get("transcript", [])[-12:]
    return f"""{persona}

You are the Talker. Say only the next spoken turn, not labels, stage directions, markdown, or analysis. Usually use 1–2 short sentences and fewer than 25 words. React specifically to the user's words, then ask one natural open question about this goal: {state.get('next_goal', '')}. Do not repeat these recent question openings: {state.get('recent_questions', [])}. Never invent or imply details about the user. No more than one question. Keep it warm, not clinical. You are an AI companion; disclose that if asked. This is a getting-to-know-you conversation, not therapy or a promise of a match.
Language mode: {mode}. Keep Hinglish conversational and naturally code-switched.
Earlier context: {state.get('rolling_summary', '')}
Recent turns: {json.dumps(recent, ensure_ascii=False)}
Conductor acknowledgement hint: {state.get('acknowledgement', '')}
Elapsed seconds: {state.get('elapsed_seconds', 0)}. At 840 seconds or later, warmly signal the last part of the conversation. At 1020 seconds, close kindly without asking another question."""


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


def _parse_conductor(raw: str, user_text: str) -> dict[str, object]:
    try:
        start, end = raw.find("{"), raw.rfind("}")
        data = json.loads(raw[start : end + 1])
        covered = [item for item in data.get("covered_now", []) if item in GOAL_IDS]
        next_goal = data.get("next_goal")
        if next_goal not in GOAL_IDS:
            next_goal = next((goal for goal in GOAL_IDS if goal not in covered), GOAL_IDS[-1])
        return {
            "covered_now": covered,
            "next_goal": next_goal,
            "acknowledgement": str(data.get("acknowledgement", ""))[:100],
        }
    except (ValueError, TypeError, json.JSONDecodeError):
        covered = _keyword_coverage(user_text)
        return {"covered_now": covered, "next_goal": next((g for g in GOAL_IDS if g not in covered), GOAL_IDS[0]), "acknowledgement": ""}


def _keyword_coverage(text: str) -> list[str]:
    normalized = text.lower()
    patterns = {
        "weekend_downtime": r"weekend|relax|downtime|free time|weekend|aaram|chhutti|sunday",
        "conflict": r"conflict|argument|fight|disagree|repair|apolog|gussa|jhag|maafi",
        "partner_wants": r"partner|relationship|looking for|want in|saathi|partner mein",
        "family_future": r"family|children|kids|future|shaadi|bachch|parivaar",
        "values_priorities": r"value|priority|important to me|matter most|belief|principle|usool",
        "communication_affection": r"communicat|affection|love language|feel loved|care|baat kar|pyaar",
        "real_life_scenario": r"scenario|what would you do|if your partner|imagine|maan lo|agar aisa",
    }
    return [goal for goal, pattern in patterns.items() if re.search(pattern, normalized)]


async def _conductor_node(state: FlowState) -> FlowState:
    messages = [ChatMessage("system", _conductor_prompt(state["language_mode"], state))]
    if state.get("user_text"):
        messages.append(ChatMessage("user", state["user_text"]))
    try:
        raw, provider = await provider_router.complete(messages, temperature=0.1)
        decision = _parse_conductor(raw, state.get("user_text", ""))
    except Exception:
        decision = {
            "covered_now": _keyword_coverage(state.get("user_text", "")),
            "next_goal": next(
                (goal for goal in GOAL_IDS if goal not in state.get("covered_goals", [])),
                GOAL_IDS[0],
            ),
            "acknowledgement": "",
        }
        provider = "conductor-fallback"
    covered = list(dict.fromkeys(state.get("covered_goals", []) + decision["covered_now"]))
    next_goal = decision["next_goal"]
    if next_goal in covered:
        next_goal = next((goal for goal in GOAL_IDS if goal not in covered), GOAL_IDS[-1])
    phase: Literal["conversation", "wrap_up", "closed"] = "conversation"
    elapsed = state.get("elapsed_seconds", 0)
    if elapsed >= settings.max_call_seconds:
        phase = "closed"
    elif elapsed >= settings.wrap_up_seconds:
        phase = "wrap_up"
    return {
        "covered_goals": covered,
        "next_goal": next_goal,
        "acknowledgement": str(decision["acknowledgement"]),
        "provider": provider,
        "phase": phase,
    }


async def _talker_node(state: FlowState) -> FlowState:
    if state.get("phase") == "closed":
        text = "Thank you for sharing this time with me. Take care, and I hope the next conversation feels easy and true."
        return {"assistant_text": text, "provider": state.get("provider", "unknown")}
    system = _talker_prompt(state["language_mode"], state)
    messages = [ChatMessage("system", system)]
    for turn in state.get("transcript", [])[-12:]:
        if turn.get("role") in ("user", "assistant"):
            messages.append(ChatMessage(turn["role"], turn["content"]))
    if state.get("user_text"):
        messages.append(ChatMessage("user", state["user_text"]))
    if state.get("phase") == "wrap_up":
        messages[0] = ChatMessage("system", system + " The 14-minute wrap-up has begun. Do not start a new required topic. Briefly reflect the user's own stated themes and invite one final thought about what helps them feel connected. Do not say goodbye yet; the call closes at 17 minutes.")
    try:
        text, provider = await provider_router.complete(messages, temperature=0.7)
    except Exception:
        mode = state["language_mode"]
        text = (
            "Thanks for telling me. What feels most important to you about that?"
            if mode == "english"
            else "Batane ke liye shukriya. Ismein tumhare liye sabse zaroori kya hai?"
        )
        provider = "talker-fallback"
    return {
        "assistant_text": _spoken_only(text, has_user_turn=bool(state.get("user_text")), mode=state["language_mode"]),
        "provider": provider,
    }


def _build_graph():
    graph = StateGraph(FlowState)
    graph.add_node("conductor", _conductor_node)
    graph.add_node("talker", _talker_node)
    graph.add_edge(START, "conductor")
    graph.add_edge("conductor", "talker")
    graph.add_edge("talker", END)
    return graph.compile()


conversation_graph = _build_graph()


@dataclass
class CallSession:
    call_id: str
    language_mode: LanguageMode
    conversation_pod: PodId = "pod1"
    started_at: float = field(default_factory=time.monotonic)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    transcript: list[dict[str, str]] = field(default_factory=list)
    export_transcript: list[dict[str, str]] = field(default_factory=list)
    covered_goals: list[str] = field(default_factory=list)
    rolling_summary: str = ""
    recent_questions: list[str] = field(default_factory=list)
    conversation_state: ConversationState = field(default_factory=ConversationState)
    latest_user_text: str = ""
    idle_nudge_pending: bool = False
    idle_nudge_grounding_detail: str = ""
    idle_nudge_details_used: list[str] = field(default_factory=list)
    closed: bool = False
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    @property
    def elapsed_seconds(self) -> int:
        return int(time.monotonic() - self.started_at)

    @property
    def phase(self) -> Literal["conversation", "wrap_up", "closed"]:
        if self.closed or self.elapsed_seconds >= settings.max_call_seconds:
            return "closed"
        if self.elapsed_seconds >= settings.wrap_up_seconds:
            return "wrap_up"
        return "conversation"


class CallStore:
    """Development-only in-memory sessions; data disappears on process restart."""

    def __init__(self) -> None:
        self._sessions: dict[str, CallSession] = {}

    async def create(
        self, language_mode: LanguageMode, conversation_pod: PodId = "pod1"
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
        if len(session.transcript) > 12:
            removed = session.transcript[:-12]
            session.transcript = session.transcript[-12:]
            user_topics = [turn["content"][:180] for turn in removed if turn["role"] == "user"]
            if user_topics:
                previous = session.rolling_summary.removeprefix("User explicitly shared: ")
                summary_parts = ([previous] if previous else []) + user_topics[-4:]
                session.rolling_summary = "User explicitly shared: " + " | ".join(summary_parts)[-1200:]


async def update_live_conversation(
    session: CallSession, observation: TurnObservation
) -> ConductorDecision:
    """Dispatch through the selected pod's turn handler."""
    async with session.lock:
        if session.idle_nudge_pending:
            session.idle_nudge_pending = False
            recent_detail = session.idle_nudge_grounding_detail or _choose_idle_nudge_detail(session)
            session.idle_nudge_grounding_detail = ""
            tone = (
                session.conversation_state.last_tone_cue
                if session.conversation_state.last_tone_confidence >= 0.65
                else "neutral"
            )
            return ConductorDecision(
                response_mode="chat",
                goal=session.conversation_state.last_goal,
                depth="follow_up",
                tone_hint=tone,
                must_avoid=session.recent_questions[-3:],
                ask_question=True,
                grounding_detail=recent_detail,
                cue=(
                    "long-pause follow-up: briefly reconnect to this real detail, then ask one easy, specific open question that "
                    "invites more. Sound natural, not interview-like. If no detail is available, offer a "
                    "gentle check-in. Don't mention the pause, invent context, or repeat must_avoid. Then "
                    "stop and leave space."
                ),
            )
    pod = get_conversation_pod(session.conversation_pod)
    return await pod.handle_turn(session, observation)


def _choose_idle_nudge_detail(session: CallSession) -> str:
    """Choose a varied, user-grounded detail for Pod 1's silence follow-up."""
    candidates: list[str] = []

    def add(value: str) -> None:
        cleaned = " ".join(value.split()).strip()
        if len(cleaned) < 12:
            return
        if len(cleaned) > 180:
            cleaned = cleaned[:177].rsplit(" ", 1)[0].rstrip(".,!?;:") + "…"
        normalized = cleaned.casefold()
        if not any(existing.casefold() == normalized for existing in candidates):
            candidates.append(cleaned)

    add(session.latest_user_text)

    # Prefer exact details recorded by the conductor; never mine a skipped topic.
    for topic, progress in session.conversation_state.topics.items():
        if progress.status != "skipped":
            for evidence in progress.evidence:
                add(evidence)
    for preference in session.conversation_state.explicit_preferences:
        add(preference)

    # Keep verbatim user material available even when it has not become a topic signal.
    for turn in session.transcript:
        if turn.get("role") != "user":
            continue
        for piece in re.split(r"(?<=[.!?])\s+|\n+", turn.get("content", "")):
            add(piece)

    # The rolling summary contains only user-shared excerpts from older turns.
    if session.rolling_summary:
        summary = session.rolling_summary.removeprefix("User explicitly shared: ")
        for piece in summary.split(" | "):
            add(piece)

    if not candidates:
        return ""

    used = {detail.casefold() for detail in session.idle_nudge_details_used}
    fresh = [detail for detail in candidates if detail.casefold() not in used]
    if not fresh:
        return ""
    selected = random.choice(fresh)
    session.idle_nudge_details_used.append(selected)
    return selected


async def _run_stable_live_conductor(
    session: CallSession, observation: TurnObservation
) -> ConductorDecision:
    """The stable Pod 1 implementation; keep its graph and state unchanged."""
    async with session.lock:
        result = await live_conductor_graph.ainvoke(
            {
                "conversation_state": session.conversation_state,
                "observation": observation,
                "user_text": session.latest_user_text,
                "elapsed_seconds": session.elapsed_seconds,
                "recent_questions": session.recent_questions,
            }
        )
        session.conversation_state = result["conversation_state"]
        session.covered_goals = [
            topic
            for topic, progress in session.conversation_state.topics.items()
            if progress.status == "covered"
        ]
        return result["decision"]


async def _run_pod2_live_conductor(
    session: CallSession, observation: TurnObservation
) -> ConductorDecision:
    """Pod 2 starts as a behavior-identical clone with its own change point."""
    return await _run_stable_live_conductor(session, observation)


async def generate_turn(session: CallSession, user_text: str | None = None) -> tuple[str, str]:
    async with session.lock:
        if session.phase == "closed":
            session.closed = True
        state: FlowState = {
            "language_mode": session.language_mode,
            "transcript": session.transcript[-12:],
            "rolling_summary": session.rolling_summary,
            "recent_questions": session.recent_questions[-10:],
            "user_text": user_text or "",
            "elapsed_seconds": session.elapsed_seconds,
            "covered_goals": session.covered_goals,
            "current_goal": session.transcript[-1].get("goal", "") if session.transcript else GOAL_LABELS[GOAL_IDS[0]],
            "phase": session.phase,
        }
        result = await conversation_graph.ainvoke(state)
        if user_text:
            user_turn = {"role": "user", "content": user_text}
            session.transcript.append(user_turn)
            if settings.export_call_logs:
                session.export_transcript.append(user_turn.copy())
        assistant = result["assistant_text"]
        assistant_turn = {"role": "assistant", "content": assistant, "goal": result.get("next_goal", "")}
        session.transcript.append(assistant_turn)
        if settings.export_call_logs:
            session.export_transcript.append(assistant_turn.copy())
        if "?" in assistant:
            session.recent_questions.append(assistant.rsplit("?", 1)[0].split(".")[-1].strip() + "?")
            session.recent_questions = session.recent_questions[-10:]
        session.covered_goals = result.get("covered_goals", session.covered_goals)
        # Keep only recent verbatim turns. The summary stores only explicit user-shared topics.
        if len(session.transcript) > 12:
            removed = session.transcript[:-12]
            session.transcript = session.transcript[-12:]
            user_topics = [turn["content"][:180] for turn in removed if turn["role"] == "user"]
            if user_topics:
                previous = session.rolling_summary.removeprefix("User explicitly shared: ")
                summary_parts = ([previous] if previous else []) + user_topics[-4:]
                session.rolling_summary = "User explicitly shared: " + " | ".join(summary_parts)[-1200:]
        if result.get("phase") == "closed":
            session.closed = True
        return assistant, result.get("provider", "unknown")
