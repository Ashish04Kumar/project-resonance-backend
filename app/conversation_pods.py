"""Selectable conversation capability bundles for the live voice call."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Awaitable, Callable, Literal, cast

from .conversation_pod4 import Pod4Decision, Pod4TurnObservation
from .conversation_pod5 import Pod5Decision, Pod5TurnObservation

if TYPE_CHECKING:
    from .conversation import CallSession

PodId = Literal["pod4", "pod5"]
TurnHandler = Callable[
    ["CallSession", Pod4TurnObservation | Pod5TurnObservation],
    Awaitable[Pod4Decision | Pod5Decision],
]


async def _pod4_conductor(session: "CallSession", observation: Pod4TurnObservation) -> Pod4Decision:
    from .conversation_pod4 import run_pod4_conductor

    return await run_pod4_conductor(session, observation)


async def _pod5_conductor(session: "CallSession", observation: Pod5TurnObservation) -> Pod5Decision:
    from .conversation_pod5 import run_pod5_conductor

    return await run_pod5_conductor(session, observation)


@dataclass(frozen=True)
class ConversationPod:
    id: PodId
    label: str
    status: Literal["stable", "clone"]
    description: str
    capabilities: tuple[str, ...]
    prompt_addendum: str
    handle_turn: TurnHandler


PODS: dict[PodId, ConversationPod] = {
    "pod4": ConversationPod(
        id="pod4",
        label="Wave conversation on our own voice pipeline",
        status="clone",
        description=(
            "The Wave brain (first built as Pod 6, now removed) running without Gemini Live: speech-to-text, the Conductor, a text "
            "LLM and text-to-speech in order, so the Conductor plans every reply. Each part is plug-and-play from "
            ".env: laptop Whisper and Piper by default with Sarvam Saarika and Bulbul as fallback, and Gemini text "
            "with laptop Ollama as fallback. Stages come from blueprints/pod4_blueprint.yaml."
        ),
        capabilities=(
            "Our own pipeline: STT, then Conductor, then LLM Talker, then TTS on every turn",
            "Pluggable providers per part with a fallback (whisper/sarvam, piper/sarvam, gemini/ollama)",
            "Wave stages, branches, revive lines, skip/dodge rules and backup topics",
            "Silence nudge, safety card and an 18-minute clock with a spoken goodbye",
        ),
        # Pod 4 writes its own Talker prompt in conversation_pod4_text.py; no Gemini Live addendum.
        prompt_addendum="",
        handle_turn=_pod4_conductor,
    ),
    "pod5": ConversationPod(
        id="pod5",
        label="Kismet requirements on Gemini Live",
        status="clone",
        description=(
            "A blueprint-driven Conductor implements the Kismet AI Call conversation "
            "requirements while Gemini Live stays the voice."
        ),
        capabilities=(
            "Gemini Live audio conversation",
            "Goals, slots, angles and limits loaded from blueprints/call_blueprint.yaml",
            "Five answer outcomes with code-first phrase rules and separate retry and follow-up counters",
            "Warm-up, core, wrap-up and close clock with a behind-schedule check",
            "Seeded detours, filler types, goal order and scenario, replayable per call",
            "Code and model safety check on every turn with a fixed support path",
            "Coverage report with not-covered reasons in the call export",
        ),
        prompt_addendum=(
            "POD 5 CONDUCTOR (PRIVATE): You are the Talker only. The Conductor decides what to cover; you decide how "
            "a warm, curious friend would say it. The tool result often arrives after you have already replied. "
            "Never speak again just because a tool result arrived; use the latest `private_talker_guidance` for "
            "your next reply:\n"
            "- response_mode says what this turn is for. ask_goal/revisit/retry/last_chance: ask one easy question "
            "about `goal_label`, only for the `missing` parts, using `angle_hint` as a direction, never as a script. "
            "If `scenario` is set, use that situation. retry: make it easier with an example or two simple options. "
            "detour: a short light tangent of `detour_type` (about `detour_theme` if given). detour_return: come back "
            "to the goal with the given `return_style`. follow_up: go one step deeper into what they just shared. "
            "safety: follow the cue exactly and offer `support_resources`. close: say goodbye, no question.\n"
            "- Start most replies with a short reaction to their words, then ask (about 9 in 10 turns). Word the "
            "`filler_type` naturally (agreement, small reaction, thinking aloud, callback, light aside). A light aside "
            "must never claim a human life, body or experiences.\n"
            "- If `callback_detail` is set, refer back to it naturally in their own terms.\n"
            "- Never begin a reply with any phrase in `must_avoid_openers`, and never repeat `must_avoid`.\n"
            "- 1 to 2 short spoken sentences, about 25 words, no lists. No advice, labels, opinions, diagnosis or "
            "scores about the user. Never assume what any gender wants.\n"
            "- In your very first turn, say briefly and naturally that you're Aria, an AI, then start small talk.\n"
            "- Report what the user stated in the tool call: exact quotes only, and a short English `value` for each "
            "slot, even in Hinglish. Set `asr_unclear` if you couldn't make out their words; set `distress` for any "
            "sign of self-harm or abuse.\n"
            "Never say the Conductor, goals, slots, modes or any internal field aloud."
        ),
        handle_turn=_pod5_conductor,
    ),
}

def get_conversation_pod(pod_id: str) -> ConversationPod:
    if pod_id not in PODS:
        raise ValueError(f"Unknown conversation pod: {pod_id}")
    return PODS[cast(PodId, pod_id)]


def pod_catalog() -> list[dict[str, object]]:
    return [
        {
            "id": pod.id,
            "label": pod.label,
            "status": pod.status,
            "description": pod.description,
            "capabilities": list(pod.capabilities),
        }
        for pod in PODS.values()
    ]
