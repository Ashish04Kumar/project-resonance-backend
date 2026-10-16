"""Selectable conversation capability bundles for the live voice call."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Awaitable, Callable, Literal, cast

from .conversation_schemas import ConductorDecision, TurnObservation

if TYPE_CHECKING:
    from .conversation import CallSession

PodId = Literal["pod1", "pod2"]
TurnHandler = Callable[["CallSession", TurnObservation], Awaitable[ConductorDecision]]


async def _stable_conductor(session: "CallSession", observation: TurnObservation) -> ConductorDecision:
    # Import here to keep the registry independent from the conversation graph module.
    from .conversation import _run_stable_live_conductor

    return await _run_stable_live_conductor(session, observation)


async def _pod2_conductor(session: "CallSession", observation: TurnObservation) -> ConductorDecision:
    # Pod 2 has its own handler seam, while intentionally matching Pod 1 today.
    from .conversation import _run_pod2_live_conductor

    return await _run_pod2_live_conductor(session, observation)


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
    "pod1": ConversationPod(
        id="pod1",
        label="Stable conversation",
        status="stable",
        description="The current, approved live conversation behavior.",
        capabilities=(
            "Gemini Live audio conversation",
            "Pydantic-validated turn observations",
            "LangGraph topic pacing and follow-up guidance",
            "Respect for direct questions, corrections, vulnerability, and skips",
            "One context-grounded follow-up after a long post-turn pause",
        ),
        prompt_addendum=(
            "LONG-PAUSE FOLLOW-UP: After the caller has spoken and you have finished your turn, a private "
            "cue may arrive after sustained silence. Treat it as internal guidance, never as caller speech. "
            "The conductor may provide one actual detail the caller previously shared; reconnect to that "
            "detail briefly, then ask one easy, specific, open question whose purpose is to invite the "
            "caller to share more. Vary which remembered detail you return to, and avoid recent questions. "
            "Do not mention the pause, say that you remembered or analyzed them, invent details, or change "
            "to an unrelated topic. If no grounded detail is available, use one gentle open check-in. "
            "Then stop and leave the floor to the caller; do not issue another nudge until the caller has "
            "spoken again."
        ),
        handle_turn=_stable_conductor,
    ),
    "pod2": ConversationPod(
        id="pod2",
        label="Conversation clone",
        status="clone",
        description=(
            "A complete copy of Pod 1's current conversation behavior, with its own prompt "
            "and turn-handler extension points for later experiments."
        ),
        capabilities=(
            "Gemini Live audio conversation",
            "Pydantic-validated turn observations",
            "LangGraph topic pacing and follow-up guidance",
            "Respect for direct questions, corrections, vulnerability, and skips",
        ),
        prompt_addendum="",
        handle_turn=_pod2_conductor,
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
