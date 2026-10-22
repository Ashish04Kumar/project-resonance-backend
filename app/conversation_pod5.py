"""Pod 5: blueprint-driven Conductor for the Kismet AI call (Gemini Live Talker).

Pod 5 started as a clone of Pod 3 and keeps its shape: Gemini Live sends a
private, Pydantic-validated observation after each user turn, a local
LangGraph graph merges it into temporary call state, then plans the next turn
and returns private guidance. The planner itself is rebuilt around the
blueprint in `blueprints/call_blueprint.yaml` and the requirements in
"Kismet AI Call: Conversation Requirements".

What code decides here, never the LLM:
- which goal comes next, which angle, and what is still missing (TOP-1..6)
- the five answer outcomes and their retry limits (RETRY-1..8)
- the clock: warm-up, core, wrap-up, close (TIME-1..4)
- detours and filler types, rolled with one stored seed per call (DET, FILL, RAND-1)
- the safety path (SAFE-1, SAFE-2)

This module is pure: it never imports a provider or makes a network call
(PROV-5). The text-only fallback that needs an LLM lives in
`conversation_pod5_text.py`.
"""

from __future__ import annotations

import random
from typing import TYPE_CHECKING, Literal, TypedDict

from langgraph.graph import END, START, StateGraph
from loguru import logger
from pydantic import Field

from .conversation_schemas import StrictModel, ToneCue, TurnIntent
from .pod5_blueprint import Blueprint, FillerType, Goal, get_blueprint
from .pod5_rules import code_outcome, distress_quote, is_grounded, opener

if TYPE_CHECKING:
    from .conversation import CallSession


# Validate the blueprint when the backend starts (TOP-6).
POD5_BLUEPRINT: Blueprint = get_blueprint()
POD5_GOAL_ORDER: tuple[str, ...] = POD5_BLUEPRINT.goal_ids
POD5_GOAL_LABELS: dict[str, str] = {goal.id: goal.label for goal in POD5_BLUEPRINT.goals}
POD5_SLOT_KEYS: tuple[str, ...] = POD5_BLUEPRINT.slot_keys

Pod5Outcome = Literal["full", "partial", "no_answer", "off_topic", "skip", "not_applicable"]
GoalStatus = Literal["open", "active", "partial", "covered", "not_covered", "skipped"]
AskKind = Literal["new", "retry", "revisit", "follow_up", "last_chance"]
Segment = Literal["warm_up", "core", "wrap_up", "close"]
ResponseMode = Literal[
    "warm_up",
    "chat",
    "ask_goal",
    "retry",
    "revisit",
    "follow_up",
    "detour",
    "detour_return",
    "answer_user",
    "acknowledge_correction",
    "respect_skip",
    "repair",
    "wrap_up",
    "last_chance",
    "close",
    "safety",
]
CLOSED_STATUSES: tuple[GoalStatus, ...] = ("covered", "not_covered", "skipped")


# --- Observation: what Gemini reports after each user turn -------------------


class Pod5SlotFill(StrictModel):
    # "goal_id.slot_id", one of POD5_SLOT_KEYS.
    slot: str = Field(min_length=3, max_length=80)
    # Exact short quote from the user's latest words; anything else is dropped.
    quote: str = Field(min_length=1, max_length=180)
    # Short English-normalized value so English and Hinglish users match (LANG-4).
    value: str = Field(default="", max_length=120)


class Pod5TurnObservation(StrictModel):
    intent: TurnIntent = "other"
    tone_cue: ToneCue = "neutral"
    tone_confidence: float = Field(default=0.0, ge=0, le=1)
    outcome: Pod5Outcome = "not_applicable"
    outcome_confidence: float = Field(default=0.0, ge=0, le=1)
    slots_filled: list[Pod5SlotFill] = Field(default_factory=list, max_length=6)
    distress: bool = False
    distress_quote: str = Field(default="", max_length=180)
    asr_unclear: bool = False


# --- Call state ---------------------------------------------------------------


class Pod5SlotValue(StrictModel):
    quote: str = Field(max_length=180)
    value: str = Field(default="", max_length=120)
    turn: int = Field(default=0, ge=0)


class Pod5Fact(StrictModel):
    slot: str
    quote: str = Field(max_length=180)
    value: str = Field(default="", max_length=120)
    turn: int = Field(default=0, ge=0)


class Pod5GoalProgress(StrictModel):
    status: GoalStatus = "open"
    slots: dict[str, Pod5SlotValue] = Field(default_factory=dict)
    # Retry counter: every time the Conductor asks this goal (RETRY-5).
    asks: int = Field(default=0, ge=0)
    # Separate counter for follow-ups that deepen a rich answer (RETRY-5).
    followups: int = Field(default=0, ge=0)
    angles_used: list[str] = Field(default_factory=list)
    last_outcome: Pod5Outcome | None = None
    # Partial answers come back once this many goals have closed.
    revisit_after_closed: int | None = None
    last_chance_asked: bool = False
    reason: str = Field(default="", max_length=160)


class Pod5Detour(StrictModel):
    type: Literal["user_thread", "light_tangent", "playful_what_if"]
    theme: str = ""
    pending_goal: str | None = None
    turns: int = Field(default=0, ge=0)
    returning: bool = False
    return_attempts: int = Field(default=0, ge=0)
    started_at_seconds: int = Field(default=0, ge=0)


class Pod5TimelineEntry(StrictModel):
    segment: Segment
    at_seconds: int = Field(ge=0)


def _new_seed() -> int:
    return random.SystemRandom().randrange(1, 2**31)


def _empty_goals() -> dict[str, Pod5GoalProgress]:
    return {goal_id: Pod5GoalProgress() for goal_id in POD5_GOAL_ORDER}


class Pod5ConversationState(StrictModel):
    # RAND-1: every random choice in the call derives from this seed.
    seed: int = Field(default_factory=_new_seed)
    scenario: str = ""
    goals: dict[str, Pod5GoalProgress] = Field(default_factory=_empty_goals)
    active_goal: str | None = None
    active_ask: AskKind | None = None
    active_pending_turns: int = Field(default=0, ge=0)
    goals_closed: int = Field(default=0, ge=0)
    turn_count: int = Field(default=0, ge=0)
    last_tone_cue: ToneCue = "neutral"
    last_tone_confidence: float = Field(default=0.0, ge=0, le=1)
    detours_used: int = Field(default=0, ge=0)
    detour: Pod5Detour | None = None
    last_detour_turn: int = Field(default=-10)
    last_breather_turn: int = Field(default=-10)
    # DET-6: facts said during a detour that did not complete a goal.
    incidental_facts: list[Pod5Fact] = Field(default_factory=list)
    recent_filler_types: list[FillerType] = Field(default_factory=list)
    recent_openers: list[str] = Field(default_factory=list)
    opener_repeats: int = Field(default=0, ge=0)
    callbacks_used: list[str] = Field(default_factory=list)
    last_callback_turn: int = Field(default=0, ge=0)
    asr_repair_attempts: int = Field(default=0, ge=0)
    safety_mode: bool = False
    safety_quote: str = Field(default="", max_length=180)
    safety_resources_sent: bool = False
    segment: Segment = "warm_up"
    timeline: list[Pod5TimelineEntry] = Field(default_factory=list)
    behind_schedule_seconds: int = 0


# --- Decision: private guidance returned to the Talker --------------------------


class Pod5Decision(StrictModel):
    response_mode: ResponseMode = "chat"
    goal: str | None = None
    goal_label: str = ""
    ask_kind: AskKind | None = None
    angle: str | None = None
    angle_hint: str = ""
    missing: list[str] = Field(default_factory=list)
    scenario: str = ""
    detour_type: str = ""
    detour_theme: str = ""
    return_style: str = ""
    filler_type: FillerType | None = None
    callback_detail: str = Field(default="", max_length=180)
    tone_hint: ToneCue = "neutral"
    must_avoid_openers: list[str] = Field(default_factory=list)
    must_avoid: list[str] = Field(default_factory=list, max_length=8)
    ask_question: bool = False
    support_resources: list[str] = Field(default_factory=list)
    segment: Segment = "core"
    seconds_left: int = 0
    cue: str = Field(default="React naturally to what they said.", max_length=700)
    planning_reason: str = Field(default="", max_length=200)


class Pod5GraphState(TypedDict, total=False):
    conversation_state: Pod5ConversationState
    observation: Pod5TurnObservation
    user_text: str
    elapsed_seconds: int
    recent_questions: list[str]
    # Set by the merge step for the planner.
    resolved_outcome: Pod5Outcome
    retry_due: bool
    repair_due: bool
    just_covered: str | None
    opened_goals: list[str]
    decision: Pod5Decision


# --- Helpers --------------------------------------------------------------------


def _rng(state: Pod5ConversationState, purpose: str) -> random.Random:
    """A deterministic random source for one purpose on one turn (RAND-1)."""
    return random.Random(f"{state.seed}:{state.turn_count}:{purpose}")


def _weighted(rng: random.Random, weights: dict[str, float]) -> str:
    items = [(key, weight) for key, weight in weights.items() if weight > 0]
    total = sum(weight for _, weight in items)
    roll = rng.random() * total
    for key, weight in items:
        roll -= weight
        if roll <= 0:
            return key
    return items[-1][0]


def _goal(goal_id: str) -> Goal:
    return POD5_BLUEPRINT.goal(goal_id)


def _missing_slots(goal_id: str, progress: Pod5GoalProgress) -> list[str]:
    return [slot.id for slot in _goal(goal_id).slots if slot.id not in progress.slots]


def _close_goal(state: Pod5ConversationState, goal_id: str, status: GoalStatus, reason: str = "") -> None:
    progress = state.goals[goal_id]
    if progress.status in CLOSED_STATUSES:
        return
    progress.status = status
    progress.reason = reason
    progress.revisit_after_closed = None
    state.goals_closed += 1
    if state.active_goal == goal_id:
        state.active_goal = None
        state.active_ask = None
        state.active_pending_turns = 0


def _segment(elapsed: int) -> Segment:
    clock = POD5_BLUEPRINT.call
    if elapsed >= clock.close_seconds:
        return "close"
    if elapsed >= clock.wrap_up_seconds:
        return "wrap_up"
    if elapsed < clock.segments.warm_up:
        return "warm_up"
    return "core"


def _must_cover_ids() -> list[str]:
    return [goal.id for goal in POD5_BLUEPRINT.goals if goal.must_cover]


def _behind_schedule_seconds(state: Pod5ConversationState, elapsed: int) -> int:
    """How far coverage is behind an even pace across the core window (TIME-4)."""
    clock = POD5_BLUEPRINT.call
    must_cover = _must_cover_ids()
    if not must_cover:
        return 0
    core_start = clock.segments.warm_up
    core_end = clock.wrap_up_seconds
    if not clock.lab_enabled:
        window = core_end - core_start
    else:
        window = clock.segments.core
    per_goal = max(1.0, window / len(must_cover))
    expected = min(len(must_cover), max(0.0, (elapsed - core_start) / per_goal))
    closed = sum(1 for goal_id in must_cover if state.goals[goal_id].status in CLOSED_STATUSES)
    return int((expected - closed) * per_goal)


def _sensitive_allowed(state: Pod5ConversationState, elapsed: int) -> bool:
    clock = POD5_BLUEPRINT.call
    return state.turn_count >= clock.sensitive_after_user_turns or elapsed >= clock.sensitive_after_seconds


def _pick_angle(state: Pod5ConversationState, goal_id: str, purpose: str) -> tuple[str, str]:
    goal = _goal(goal_id)
    progress = state.goals[goal_id]
    unused = [angle for angle in goal.angles if angle.id not in progress.angles_used]
    pool = unused or list(goal.angles)  # TOP-4: no repeat until every angle was used
    angle = _rng(state, f"angle:{goal_id}:{purpose}").choice(pool)
    progress.angles_used.append(angle.id)
    return angle.id, angle.hint


def _pick_filler(state: Pod5ConversationState, has_callback: bool) -> FillerType:
    config = POD5_BLUEPRINT.fillers
    recent = set(state.recent_filler_types[-config.no_repeat_type_within_turns:]) if config.no_repeat_type_within_turns else set()
    weights = {
        kind: weight
        for kind, weight in config.types.items()
        if kind not in recent and (kind != "callback" or has_callback)
    }
    if not weights:
        weights = {kind: weight for kind, weight in config.types.items() if kind != "callback" or has_callback}
    choice: FillerType = _weighted(_rng(state, "filler"), weights)  # type: ignore[assignment]
    state.recent_filler_types = (state.recent_filler_types + [choice])[-10:]
    return choice


def _all_quotes(state: Pod5ConversationState) -> list[tuple[int, str]]:
    quotes: list[tuple[int, str]] = []
    for goal_id, progress in state.goals.items():
        if progress.status == "skipped":
            continue
        for value in progress.slots.values():
            quotes.append((value.turn, value.quote))
    quotes.extend((fact.turn, fact.quote) for fact in state.incidental_facts)
    return quotes


def _pick_callback(state: Pod5ConversationState) -> str:
    """VOICE-7: an earlier detail in the user's own words, never the latest turn."""
    voice = POD5_BLUEPRINT.voice
    if state.turn_count - state.last_callback_turn < voice.callback_every_user_turns:
        return ""
    used = {item.casefold() for item in state.callbacks_used}
    candidates = sorted(
        {quote for turn, quote in _all_quotes(state) if turn < state.turn_count and quote.casefold() not in used}
    )
    if not candidates:
        return ""
    detail = _rng(state, "callback").choice(candidates)
    state.callbacks_used.append(detail)
    state.last_callback_turn = state.turn_count
    return detail


def _scenario(state: Pod5ConversationState) -> str:
    if not state.scenario:
        scenarios = next((goal.scenarios for goal in POD5_BLUEPRINT.goals if goal.scenarios), [])
        if scenarios:
            state.scenario = random.Random(f"{state.seed}:scenario").choice(scenarios)
    return state.scenario


def record_pod5_assistant_turn(state: Pod5ConversationState, text: str) -> None:
    """Remember reply openers so the Talker is told not to reuse them (VOICE-3)."""
    voice = POD5_BLUEPRINT.voice
    start = opener(text, voice.opener_words)
    if not start:
        return
    if start in state.recent_openers[-voice.opener_no_repeat_within_turns:]:
        state.opener_repeats += 1
        logger.warning("POD5 OPENER REPEAT | opener={!r} | repeats={}", start, state.opener_repeats)
    state.recent_openers = (state.recent_openers + [start])[-voice.opener_no_repeat_within_turns:]


# --- Graph step 1: merge the observation -------------------------------------


async def _merge_observation(state: Pod5GraphState) -> Pod5GraphState:
    conversation = state["conversation_state"].model_copy(deep=True)
    observation = state["observation"]
    user_text = state.get("user_text", "")
    retries = POD5_BLUEPRINT.retries
    conversation.turn_count += 1
    conversation.last_tone_cue = observation.tone_cue
    conversation.last_tone_confidence = observation.tone_confidence
    result: Pod5GraphState = {
        "resolved_outcome": "not_applicable",
        "retry_due": False,
        "repair_due": False,
        "just_covered": None,
        "opened_goals": [],
    }

    # Safety first (priority order: safety, honesty, user choice, coverage, time).
    flagged = distress_quote(user_text)
    if not flagged and observation.distress and is_grounded(observation.distress_quote, user_text):
        flagged = observation.distress_quote
    if flagged or conversation.safety_mode:
        if flagged and not conversation.safety_mode:
            conversation.safety_mode = True
            conversation.safety_quote = flagged[:180]
            logger.warning("POD5 SAFETY PATH STARTED | turn={}", conversation.turn_count)
        return {**result, "conversation_state": conversation}

    # RETRY-1: code classifies obvious phrases before the model is trusted.
    code = code_outcome(user_text)
    outcome: Pod5Outcome = code or observation.outcome
    if code is None and outcome != "not_applicable" and observation.outcome_confidence < retries.uncertain_confidence:
        # RETRY-2: an unsure classification counts as partial, so the AI asks less.
        outcome = "partial"

    # VOICE-8: if the words were not heard clearly, ask once casually, at most twice.
    if observation.asr_unclear and code is None:
        conversation.asr_repair_attempts += 1
        if conversation.asr_repair_attempts <= retries.asr_repair_max_attempts:
            return {**result, "conversation_state": conversation, "repair_due": True}
        outcome = "not_applicable"
    conversation.asr_repair_attempts = 0

    # Store only slot quotes that really appear in the user's latest words.
    detour = conversation.detour
    detour_candidates: dict[str, dict[str, Pod5SlotValue]] = {}
    opened: list[str] = []
    for fill in observation.slots_filled:
        goal_id, _, slot_id = fill.slot.partition(".")
        if fill.slot not in POD5_SLOT_KEYS or not is_grounded(fill.quote, user_text):
            continue
        progress = conversation.goals[goal_id]
        if progress.status in ("skipped",) or slot_id in progress.slots:
            continue
        value = Pod5SlotValue(quote=fill.quote, value=fill.value, turn=conversation.turn_count)
        if detour is not None and not detour.returning:
            detour_candidates.setdefault(goal_id, {})[slot_id] = value
        else:
            progress.slots[slot_id] = value
            opened.append(goal_id)

    # DET-6: detour facts cover a goal only if they complete every slot.
    for goal_id, values in detour_candidates.items():
        progress = conversation.goals[goal_id]
        merged = {**progress.slots, **values}
        if not _missing_slots(goal_id, progress.model_copy(update={"slots": merged})):
            progress.slots = merged
            opened.append(goal_id)
        else:
            conversation.incidental_facts.extend(
                Pod5Fact(slot=f"{goal_id}.{slot_id}", quote=value.quote, value=value.value, turn=value.turn)
                for slot_id, value in values.items()
            )

    # TOP-2: any goal whose slots are now all filled counts as covered, even unprompted.
    just_covered: str | None = None
    for goal_id in dict.fromkeys(opened):
        progress = conversation.goals[goal_id]
        if progress.status not in CLOSED_STATUSES and not _missing_slots(goal_id, progress):
            was_active = conversation.active_goal == goal_id
            _close_goal(conversation, goal_id, "covered")
            if was_active:
                just_covered = goal_id

    # A detour ends once the user is back on the pending goal.
    if detour is not None:
        if detour.returning and outcome not in ("off_topic", "not_applicable"):
            conversation.detour = None
        else:
            detour.turns += 1
            if code == "skip" and detour.pending_goal:
                _close_goal(conversation, detour.pending_goal, "skipped", "user skipped")
                conversation.detour = None
                outcome = "skip"

    # Five outcomes for the goal that was just asked (section 7).
    active_id = conversation.active_goal
    retry_due = False
    if active_id and conversation.detour is None:
        progress = conversation.goals[active_id]
        progress.last_outcome = outcome
        if outcome == "skip":
            # RETRY-8: a skip is never asked again, in this call or a resumed one.
            _close_goal(conversation, active_id, "skipped", "user skipped")
        elif conversation.active_ask == "last_chance":
            # RETRY-6: the one wrap-up question failed, so the goal stays not covered.
            _close_goal(conversation, active_id, "not_covered", "no full answer to the wrap-up question")
        elif outcome in ("full", "partial"):
            # "full" with missing slots is still partial: only slots decide coverage.
            if progress.asks >= retries.partial_max_asks:
                _close_goal(conversation, active_id, "not_covered", f"partial after {progress.asks} asks")
            else:
                progress.status = "partial"
                progress.revisit_after_closed = conversation.goals_closed + retries.partial_revisit_after_goals
                conversation.active_goal = None
                conversation.active_ask = None
                conversation.active_pending_turns = 0
        elif outcome in ("no_answer", "off_topic"):
            if progress.asks >= retries.no_answer_max_asks:
                _close_goal(conversation, active_id, "not_covered", f"no answer after {progress.asks} asks")
            else:
                retry_due = True
        else:
            # The user asked something back or corrected us: keep the goal pending.
            conversation.active_pending_turns += 1
            if conversation.active_pending_turns >= 2:
                if progress.asks >= retries.no_answer_max_asks:
                    _close_goal(conversation, active_id, "not_covered", "conversation moved away twice")
                else:
                    retry_due = True

    return {
        **result,
        "conversation_state": conversation,
        "resolved_outcome": outcome,
        "retry_due": retry_due,
        "just_covered": just_covered,
        "opened_goals": [goal_id for goal_id in dict.fromkeys(opened) if conversation.goals[goal_id].status not in CLOSED_STATUSES],
    }


# --- Graph step 2: plan the next turn ------------------------------------------


def _next_goal(
    state: Pod5ConversationState, elapsed: int, opened: list[str], *, behind: bool
) -> tuple[str | None, AskKind]:
    sensitive_ok = _sensitive_allowed(state, elapsed)
    must_cover_open = [
        goal_id for goal_id in _must_cover_ids() if state.goals[goal_id].status not in CLOSED_STATUSES
    ]

    def eligible(goal_id: str) -> bool:
        goal = _goal(goal_id)
        if goal.sensitive and not sensitive_ok:
            return False
        if not goal.must_cover and (must_cover_open or behind):
            return False
        return True

    # 1. Partial answers whose revisit is due (RETRY-4).
    due = [
        goal_id
        for goal_id in POD5_GOAL_ORDER
        if state.goals[goal_id].status == "partial"
        and (state.goals[goal_id].revisit_after_closed or 0) <= state.goals_closed
        and eligible(goal_id)
    ]
    if due:
        return due[0], "revisit"
    # 2. Goals the last answer opened up (TOP-2).
    for goal_id in opened:
        if state.goals[goal_id].status == "open" and eligible(goal_id):
            return goal_id, "new"
    # 3. Warm-first order with a seeded pick among the next few (TOP-3, TOP-5).
    fresh = [goal_id for goal_id in POD5_GOAL_ORDER if state.goals[goal_id].status == "open" and eligible(goal_id)]
    if fresh:
        window = fresh[:3]
        return _rng(state, "next_goal").choice(window), "new"
    # 4. Nothing new left: bring back a waiting partial early rather than idle.
    waiting = [goal_id for goal_id in POD5_GOAL_ORDER if state.goals[goal_id].status == "partial" and eligible(goal_id)]
    if waiting:
        return waiting[0], "revisit"
    return None, "new"


def _missing_ask_for(goal_id: str, progress: Pod5GoalProgress) -> list[str]:
    """What the Talker should ask for: only the slots still missing (RETRY-4)."""
    asks = {slot.id: slot.ask_for for slot in _goal(goal_id).slots}
    return [asks[slot_id] for slot_id in _missing_slots(goal_id, progress)]


def _ask(state: Pod5ConversationState, decision: Pod5Decision, goal_id: str, kind: AskKind) -> None:
    progress = state.goals[goal_id]
    goal = _goal(goal_id)
    progress.asks += 0 if kind == "follow_up" else 1
    if kind == "follow_up":
        progress.followups += 1
    if kind != "follow_up":
        progress.status = "active"
        state.active_goal = goal_id
        state.active_ask = kind
        state.active_pending_turns = 0
    angle_id, hint = _pick_angle(state, goal_id, kind)
    decision.goal = goal_id
    decision.goal_label = goal.label
    decision.ask_kind = kind
    decision.angle = angle_id
    decision.angle_hint = hint
    # A follow-up deepens an answer that is already complete, so nothing is missing.
    decision.missing = [] if kind == "follow_up" else _missing_ask_for(goal_id, progress)
    decision.ask_question = True
    if goal.scenarios:
        decision.scenario = _scenario(state)


async def _plan_turn(state: Pod5GraphState) -> Pod5GraphState:
    conversation = state["conversation_state"].model_copy(deep=True)
    observation = state["observation"]
    elapsed = state.get("elapsed_seconds", 0)
    clock = POD5_BLUEPRINT.call
    detours = POD5_BLUEPRINT.detours
    voice = POD5_BLUEPRINT.voice

    segment = _segment(elapsed)
    if segment != conversation.segment or not conversation.timeline:
        conversation.segment = segment
        conversation.timeline.append(Pod5TimelineEntry(segment=segment, at_seconds=elapsed))
    behind_seconds = _behind_schedule_seconds(conversation, elapsed)
    conversation.behind_schedule_seconds = behind_seconds
    behind = behind_seconds > clock.behind_schedule_seconds
    tone: ToneCue = observation.tone_cue if observation.tone_confidence >= 0.65 else "neutral"

    decision = Pod5Decision(
        tone_hint=tone,
        segment=segment,
        seconds_left=max(0, clock.max_call_seconds - elapsed),
        must_avoid_openers=conversation.recent_openers[-voice.opener_no_repeat_within_turns:],
        must_avoid=list(state.get("recent_questions", [])[-3:]),
    )

    def finish(mode: ResponseMode, cue: str, reason: str, *, filler: bool = True) -> Pod5GraphState:
        decision.response_mode = mode
        decision.cue = cue
        decision.planning_reason = reason
        if filler and mode not in ("safety", "close", "repair"):
            if not decision.callback_detail and mode in ("ask_goal", "revisit", "chat", "follow_up", "wrap_up"):
                decision.callback_detail = _pick_callback(conversation)
            decision.filler_type = _pick_filler(conversation, bool(decision.callback_detail))
        return {"conversation_state": conversation, "decision": decision}

    # Safety path: the script stops for the rest of the call (SAFE-2).
    if conversation.safety_mode:
        decision.support_resources = [
            f"{resource.name}: {resource.contact}" for resource in POD5_BLUEPRINT.safety.resources
        ]
        return finish(
            "safety",
            "Stop the conversation plan completely. Speak calmly and warmly: thank them for telling you, say "
            "plainly that you are an AI and can't give the help they deserve, and encourage them to reach a "
            "trusted person or a helpline now; offer the support numbers. No compatibility questions, no advice "
            "beyond reaching support, no diagnosis.",
            "safety_path",
            filler=False,
        )

    # TIME-1: close by 17 minutes; the last minute is only for a warm goodbye.
    if segment == "close":
        for goal_id in _must_cover_ids():
            if conversation.goals[goal_id].status not in CLOSED_STATUSES:
                _close_goal(conversation, goal_id, "not_covered", "time ran out")
        return finish(
            "close",
            "Close the call warmly now. Reflect one real thing they shared, thank them, and say goodbye. "
            "Do not ask any new question.",
            "time_close",
            filler=False,
        )

    if state.get("repair_due"):
        return finish(
            "repair",
            "You didn't catch their words clearly. Casually ask them to say it again once, like "
            "\"Sorry, thoda dobara bologe?\" in the call's language. Don't guess what they said.",
            "asr_repair",
            filler=False,
        )

    if observation.intent == "user_question":
        if conversation.active_goal:
            decision.goal = conversation.active_goal
            decision.goal_label = POD5_GOAL_LABELS[conversation.active_goal]
        return finish(
            "answer_user",
            "Answer their question directly and honestly first. If a question of yours is still pending, you "
            "may come back to it softly in new words; otherwise just keep the conversation easy.",
            "answer_user_first",
        )
    if observation.intent == "correction":
        return finish(
            "acknowledge_correction",
            "Accept the correction simply, adjust, and carry on without defending the earlier assumption.",
            "correction",
        )

    if state.get("resolved_outcome") == "skip":
        return finish(
            "respect_skip",
            "They want to skip that. Accept it lightly with no pressure and don't ask about it again. Move to "
            "easy conversation; don't immediately fire another personal question.",
            "skip_respected",
        )

    # Detours: a light tangent, then back to the pending goal (DET-1..5).
    detour = conversation.detour
    if detour is not None:
        if detour.turns < detours.max_turns and segment == "core":
            decision.detour_type = detour.type
            decision.detour_theme = detour.theme
            return finish(
                "detour",
                "Stay in the light tangent for one more easy exchange. Keep it short and playful.",
                "detour_continue",
            )
        detour.returning = True
        detour.return_attempts += 1
        pending = detour.pending_goal
        if pending and conversation.goals[pending].status not in CLOSED_STATUSES:
            if detour.return_attempts <= detours.max_return_attempts:
                style = _rng(conversation, "return_style").choice(detours.return_styles)
                kind: AskKind = "revisit" if conversation.goals[pending].status == "partial" else "new"
                if detour.return_attempts == 1:
                    _ask(conversation, decision, pending, kind)
                else:
                    # A second return re-asks the same question, so it is not a new ask.
                    decision.goal = pending
                    decision.goal_label = POD5_GOAL_LABELS[pending]
                    decision.missing = _missing_ask_for(pending, conversation.goals[pending])
                    decision.ask_question = True
                decision.return_style = style
                return finish(
                    "detour_return",
                    f"Come back from the tangent with a {style.replace('_', ' ')} and ask about the goal, "
                    "in fresh words that grow out of what they just said.",
                    f"detour_return_{detour.return_attempts}",
                )
            conversation.detour = None
            _ask(conversation, decision, pending, "new")
            return finish(
                "ask_goal",
                "They keep drifting. Kindly and simply ask the goal question directly now, in one short sentence.",
                "detour_direct_ask",
            )
        conversation.detour = None

    # Wrap-up: no detours, no revisits; one last short question per open goal (RETRY-6).
    if segment == "wrap_up":
        for goal_id in _must_cover_ids():
            progress = conversation.goals[goal_id]
            if progress.status in CLOSED_STATUSES or progress.last_chance_asked:
                continue
            if goal_id == conversation.active_goal and conversation.active_ask == "last_chance":
                continue
            progress.last_chance_asked = True
            _ask(conversation, decision, goal_id, "last_chance")
            return finish(
                "last_chance",
                "We're near the end. Ask one short, easy question for just the missing part of this goal. If they "
                "pass, accept it warmly.",
                "wrap_up_last_chance",
            )
        return finish(
            "wrap_up",
            "Gently start wrapping up. Reflect one or two things they actually said and invite a final thought. "
            "Don't open any new topic.",
            "wrap_up",
        )

    # Warm-up: easy small talk only; volunteered answers still count.
    if segment == "warm_up":
        decision.ask_question = True
        return finish(
            "warm_up",
            "Easy small talk about their day or what they're up to. No relationship questions yet.",
            "warm_up",
        )

    # No answer: retry right away with a new angle and an easier form.
    active_id = conversation.active_goal
    if state.get("retry_due") and active_id:
        _ask(conversation, decision, active_id, "retry")
        return finish(
            "retry",
            "They didn't really answer. Make it easier: give a quick example or offer two simple options to pick "
            "from, using completely different words than before (RETRY-3).",
            "no_answer_retry",
        )

    # A still-active goal (for example after a correction) waits for its answer.
    if active_id:
        return finish(
            "chat",
            "Stay with what they're saying. Your earlier question is still open; you may softly return to it.",
            "awaiting_answer",
        )

    # Deepen a rich answer once, on its own counter (RETRY-5).
    just_covered = state.get("just_covered")
    if just_covered and observation.intent == "story" and not behind:
        progress = conversation.goals[just_covered]
        if progress.followups < _goal(just_covered).max_followups:
            _ask(conversation, decision, just_covered, "follow_up")
            return finish(
                "follow_up",
                "That was a rich answer. React to a specific detail and ask one natural follow-up that goes a bit "
                "deeper into it. Don't move to a new topic yet.",
                "deepen_rich_answer",
            )

    # Ahead of schedule: one casual turn between goals, never two in a row.
    ahead = -behind_seconds > clock.breather_when_ahead_seconds
    if ahead and conversation.turn_count - conversation.last_breather_turn > 1:
        conversation.last_breather_turn = conversation.turn_count
        return finish(
            "chat",
            "React warmly to what they said and keep the conversation easy for this turn. A light question about "
            "what they just shared is fine; don't start a new relationship topic.",
            "breather_ahead_of_schedule",
        )

    goal_id, kind = _next_goal(conversation, elapsed, state.get("opened_goals", []), behind=behind)
    if goal_id is None:
        return finish(
            "chat",
            "Everything needed is covered or set aside. Just keep an easy, warm conversation going.",
            "coverage_complete",
        )

    # DET-2..4: a seeded roll decides whether to take a light detour first.
    goal = _goal(goal_id)
    detour_ok = (
        segment == "core"
        and elapsed >= detours.earliest_seconds
        and conversation.detours_used < detours.max_per_call
        and not behind
        and not goal.sensitive
        and conversation.turn_count - conversation.last_detour_turn > detours.max_turns + 1
    )
    if detour_ok and _rng(conversation, "detour_roll").random() < detours.roll_probability:
        detour_type = _weighted(_rng(conversation, "detour_type"), dict(detours.types))
        theme = (
            _rng(conversation, "detour_theme").choice(detours.tangent_themes)
            if detour_type == "light_tangent"
            else ""
        )
        conversation.detours_used += 1
        conversation.last_detour_turn = conversation.turn_count
        conversation.detour = Pod5Detour(
            type=detour_type,  # type: ignore[arg-type]
            theme=theme,
            pending_goal=goal_id,
            started_at_seconds=elapsed,
        )
        decision.detour_type = detour_type
        decision.detour_theme = theme
        decision.ask_question = True
        cue = {
            "user_thread": "Take a short, curious detour into something they just mentioned that isn't a relationship topic.",
            "light_tangent": f"Take a short, light tangent about {theme} that connects to the moment.",
            "playful_what_if": "Ask one playful, light what-if question that's just for fun.",
        }[detour_type]
        return finish("detour", cue, f"detour_{detour_type}")

    _ask(conversation, decision, goal_id, kind)
    if kind == "revisit":
        cue = (
            "Casually call back to this earlier topic and ask only for the missing part, in new words. Don't repeat "
            "what they already told you."
        )
    else:
        cue = (
            "React to what they just said first, then bridge naturally into this goal using the angle as a "
            "direction, not a script. One easy question; don't announce a topic."
        )
    return finish("revisit" if kind == "revisit" else "ask_goal", cue, f"{kind}_goal")


def _build_graph():
    graph = StateGraph(Pod5GraphState)
    graph.add_node("merge_observation", _merge_observation)
    graph.add_node("plan_turn", _plan_turn)
    graph.add_edge(START, "merge_observation")
    graph.add_edge("merge_observation", "plan_turn")
    graph.add_edge("plan_turn", END)
    return graph.compile()


pod5_conductor_graph = _build_graph()


async def plan_pod5_turn(
    conversation: Pod5ConversationState,
    observation: Pod5TurnObservation,
    *,
    user_text: str,
    elapsed_seconds: int,
    recent_questions: list[str] | None = None,
) -> tuple[Pod5ConversationState, Pod5Decision]:
    """Run one Conductor turn on plain values; used by the call and by tests."""
    result = await pod5_conductor_graph.ainvoke(
        {
            "conversation_state": conversation,
            "observation": observation,
            "user_text": user_text,
            "elapsed_seconds": elapsed_seconds,
            "recent_questions": recent_questions or [],
        }
    )
    return result["conversation_state"], result["decision"]


async def run_pod5_conductor(session: "CallSession", observation: Pod5TurnObservation) -> Pod5Decision:
    """Update only Pod 5's isolated state and produce its next-turn guidance."""
    async with session.lock:
        conversation, decision = await plan_pod5_turn(
            session.pod5_state,
            observation,
            user_text=session.latest_user_text,
            elapsed_seconds=session.elapsed_seconds,
            recent_questions=session.recent_questions,
        )
        session.pod5_state = conversation
        session.covered_goals = [
            goal_id for goal_id, progress in conversation.goals.items() if progress.status == "covered"
        ]
        logger.info(
            "POD5 PLANNER | call={} | seed={} | turn={} | segment={} | behind={}s | mode={} | reason={} | goal={} | "
            "kind={} | angle={} | detours={} | filler={} | statuses={}",
            session.call_id,
            conversation.seed,
            conversation.turn_count,
            decision.segment,
            conversation.behind_schedule_seconds,
            decision.response_mode,
            decision.planning_reason,
            decision.goal or "none",
            decision.ask_kind or "none",
            decision.angle or "none",
            conversation.detours_used,
            decision.filler_type or "none",
            {goal_id: progress.status for goal_id, progress in conversation.goals.items()},
        )
        return decision


def pod5_coverage_report(conversation: Pod5ConversationState, *, final: bool = False) -> list[dict[str, object]]:
    """TOP-1 / RETRY-7: every goal is covered, skipped, or "not covered" with a reason.

    With `final=True`, goals still open when the call ends are reported as not
    covered (never guessed), so the Blueprint can show them for the user to fill in.
    """
    report: list[dict[str, object]] = []
    for goal in POD5_BLUEPRINT.goals:
        progress = conversation.goals[goal.id]
        status: str = progress.status
        reason = progress.reason
        if final and status not in CLOSED_STATUSES:
            if goal.must_cover:
                status, reason = "not_covered", reason or "call ended before this was covered"
            else:
                status, reason = "not_asked", reason or "optional goal not reached"
        report.append(
            {
                "goal": goal.id,
                "label": goal.label,
                "must_cover": goal.must_cover,
                "status": status,
                "reason": reason,
                "asks": progress.asks,
                "followups": progress.followups,
                "slots": {
                    slot.id: progress.slots[slot.id].model_dump(mode="json") if slot.id in progress.slots else None
                    for slot in goal.slots
                },
            }
        )
    return report
