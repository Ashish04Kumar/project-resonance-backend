"""Pod 4: Wave-style Conductor for our own voice pipeline (STT -> Conductor -> LLM -> TTS).

Pod 4 is a clone of Pod 6 (the Wave brain) that runs without Gemini Live: the
Conductor is called on every caller turn before the reply is written. The call flows
through stages in a fixed order (who they are, how dating has gone, what they
want), stays on the person's thread with deeper follow-ups, and keeps a
stage's intent after a soft skip by parking it and bringing it back later
from a different side. A clear "I don't want to talk about it" still drops a
stage for good. Stages and limits come from `blueprints/pod4_blueprint.yaml`.

What code decides here, never the LLM:
- which goal comes next, which angle, and what is still missing (TOP-1..6)
- the five answer outcomes and their retry limits (RETRY-1..8)
- the clock: warm-up, core, wrap-up, close (TIME-1..4)
- detours and filler types, rolled with one stored seed per call (DET, FILL, RAND-1)
- the safety path (SAFE-1, SAFE-2)

This module is pure: it never imports a provider or makes a network call
(PROV-5). The text-only fallback that needs an LLM lives in
`conversation_pod4_text.py`.
"""

from __future__ import annotations

import random
import re
from typing import TYPE_CHECKING, Literal, TypedDict

from langgraph.graph import END, START, StateGraph
from loguru import logger
from pydantic import Field

from .conversation_schemas import StrictModel, ToneCue, TurnIntent
from .pod5_blueprint import FillerType, Goal
from .pod5_rules import code_outcome, distress_quote, is_grounded, normalize, opener
from .pod4_blueprint import WaveBlueprint, WaveGoal, get_pod4_blueprint

if TYPE_CHECKING:
    from .conversation import CallSession


# Validate the blueprint when the backend starts (TOP-6).
POD4_BLUEPRINT: WaveBlueprint = get_pod4_blueprint()
POD4_GOAL_ORDER: tuple[str, ...] = POD4_BLUEPRINT.goal_ids
POD4_GOAL_LABELS: dict[str, str] = {goal.id: goal.label for goal in POD4_BLUEPRINT.goals}
POD4_SLOT_KEYS: tuple[str, ...] = POD4_BLUEPRINT.slot_keys

Pod4Outcome = Literal["full", "partial", "no_answer", "off_topic", "skip", "not_applicable"]
# "parked": soft-skipped once; it comes back later from a different side.
GoalStatus = Literal["open", "active", "partial", "parked", "covered", "not_covered", "skipped"]
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
    "revive",
]
CLOSED_STATUSES: tuple[GoalStatus, ...] = ("covered", "not_covered", "skipped")


# --- Observation: what Gemini reports after each user turn -------------------


class Pod4SlotFill(StrictModel):
    # "goal_id.slot_id", one of POD4_SLOT_KEYS.
    slot: str = Field(min_length=3, max_length=80)
    # Exact short quote from the user's latest words; anything else is dropped.
    quote: str = Field(min_length=1, max_length=180)
    # Short English-normalized value so English and Hinglish users match (LANG-4).
    value: str = Field(default="", max_length=120)


class Pod4TurnObservation(StrictModel):
    # "model": Gemini's own tool call. "extracted": read from the transcript by a text model when
    # Gemini skipped the tool. "unknown": nothing could be read, so the turn must not count as a dodge.
    source: Literal["model", "extracted", "unknown"] = "model"
    intent: TurnIntent = "other"
    tone_cue: ToneCue = "neutral"
    tone_confidence: float = Field(default=0.0, ge=0, le=1)
    outcome: Pod4Outcome = "not_applicable"
    outcome_confidence: float = Field(default=0.0, ge=0, le=1)
    slots_filled: list[Pod4SlotFill] = Field(default_factory=list, max_length=6)
    distress: bool = False
    distress_quote: str = Field(default="", max_length=180)
    asr_unclear: bool = False


# --- Call state ---------------------------------------------------------------


class Pod4SlotValue(StrictModel):
    quote: str = Field(max_length=180)
    value: str = Field(default="", max_length=120)
    turn: int = Field(default=0, ge=0)


class Pod4Fact(StrictModel):
    slot: str
    quote: str = Field(max_length=180)
    value: str = Field(default="", max_length=120)
    turn: int = Field(default=0, ge=0)


class Pod4GoalProgress(StrictModel):
    status: GoalStatus = "open"
    slots: dict[str, Pod4SlotValue] = Field(default_factory=dict)
    # Retry counter: every time the Conductor asks this goal (RETRY-5).
    asks: int = Field(default=0, ge=0)
    # Separate counter for follow-ups that deepen a rich answer (RETRY-5).
    followups: int = Field(default=0, ge=0)
    angles_used: list[str] = Field(default_factory=list)
    last_outcome: Pod4Outcome | None = None
    # Partial answers come back once this many goals have closed.
    revisit_after_closed: int | None = None
    last_chance_asked: bool = False
    # How many times this stage was soft-skipped and parked.
    parks: int = Field(default=0, ge=0)
    reason: str = Field(default="", max_length=160)


class Pod4Detour(StrictModel):
    type: Literal["user_thread", "light_tangent", "playful_what_if"]
    theme: str = ""
    pending_goal: str | None = None
    turns: int = Field(default=0, ge=0)
    returning: bool = False
    return_attempts: int = Field(default=0, ge=0)
    started_at_seconds: int = Field(default=0, ge=0)


class Pod4TimelineEntry(StrictModel):
    segment: Segment
    at_seconds: int = Field(ge=0)


def _new_seed() -> int:
    return random.SystemRandom().randrange(1, 2**31)


def _empty_goals() -> dict[str, Pod4GoalProgress]:
    return {goal_id: Pod4GoalProgress() for goal_id in POD4_GOAL_ORDER}


class Pod4ConversationState(StrictModel):
    # RAND-1: every random choice in the call derives from this seed.
    seed: int = Field(default_factory=_new_seed)
    scenario: str = ""
    goals: dict[str, Pod4GoalProgress] = Field(default_factory=_empty_goals)
    active_goal: str | None = None
    active_ask: AskKind | None = None
    active_pending_turns: int = Field(default=0, ge=0)
    goals_closed: int = Field(default=0, ge=0)
    turn_count: int = Field(default=0, ge=0)
    last_tone_cue: ToneCue = "neutral"
    last_tone_confidence: float = Field(default=0.0, ge=0, le=1)
    detours_used: int = Field(default=0, ge=0)
    detour: Pod4Detour | None = None
    last_detour_turn: int = Field(default=-10)
    last_breather_turn: int = Field(default=-10)
    # DET-6: facts said during a detour that did not complete a goal.
    incidental_facts: list[Pod4Fact] = Field(default_factory=list)
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
    # Question turns in a row; Wave takes a conversational turn after a few.
    consecutive_questions: int = Field(default=0, ge=0)
    # Short flat replies in a row ("okay", "yeah"); the AI takes the lead after a few.
    flat_streak: int = Field(default=0, ge=0)
    last_heavy_pause_turn: int = Field(default=-10)
    # Stages the user brought up themselves; they may come early, even if sensitive.
    raised_goals: list[str] = Field(default_factory=list)
    timeline: list[Pod4TimelineEntry] = Field(default_factory=list)
    behind_schedule_seconds: int = 0


# --- Decision: private guidance returned to the Talker --------------------------


class Pod4Decision(StrictModel):
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
    cue: str = Field(default="React naturally to what they said.", max_length=1000)
    # Stage guidance from the question map: how to ask, what to ask next by reply, a line to revive a flat talk.
    stage_framing: str = Field(default="", max_length=800)
    branches: list[str] = Field(default_factory=list, max_length=8)
    revive_line: str = Field(default="", max_length=300)
    planning_reason: str = Field(default="", max_length=200)


class Pod4GraphState(TypedDict, total=False):
    conversation_state: Pod4ConversationState
    observation: Pod4TurnObservation
    user_text: str
    elapsed_seconds: int
    recent_questions: list[str]
    # Set by the merge step for the planner.
    resolved_outcome: Pod4Outcome
    retry_due: bool
    repair_due: bool
    just_covered: str | None
    opened_goals: list[str]
    # "parked" or "skipped": what a skip did to the stage, so the Talker words it right.
    skip_result: str
    # The stage got a partial answer; ask its missing parts next instead of moving on.
    continue_due: bool
    # Stages the user raised in this turn, in blueprint order.
    raised_now: list[str]
    # The user didn't answer the pending stage; it was parked and the talk moves on.
    dodged: bool
    decision: Pod4Decision


# --- Helpers --------------------------------------------------------------------


def _rng(state: Pod4ConversationState, purpose: str) -> random.Random:
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


def _goal(goal_id: str) -> WaveGoal:
    return POD4_BLUEPRINT.goal(goal_id)


def _missing_slots(goal_id: str, progress: Pod4GoalProgress) -> list[str]:
    return [slot.id for slot in _goal(goal_id).slots if slot.id not in progress.slots]


def _close_goal(state: Pod4ConversationState, goal_id: str, status: GoalStatus, reason: str = "") -> None:
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


# A clear "I don't want to talk about this" ends a stage for good; plain "skip" only parks it.
_HARD_DECLINE_PATTERNS = (
    r"\bdon'?t want to (talk|answer|share|discuss|get into)\b",
    r"\brather not\b",
    r"\bnot comfortable\b",
    r"\b(too )?personal\b",
    r"\bnahi+n? batana\b",
    r"\bnahi+n? baat karni\b",
    r"\bmat poocho\b",
)


def hard_decline(user_text: str) -> bool:
    text = normalize(user_text)
    return any(re.search(pattern, text) for pattern in _HARD_DECLINE_PATTERNS)


# Replies made only of acknowledgment words read as the talk going flat.
_FLAT_WORDS = frozenset(
    "ok okay okk yeah yea yes yep ya haan han hmm hm mm achha acha accha thanks thank you thx perfect nice cool "
    "fine right sure true good alright i know so you are you're youre correct exactly absolutely agree totally "
    "makes sense that that's thats it is it's its".split()
)


def is_flat_reply(user_text: str) -> bool:
    words = re.findall(r"[a-z']+", normalize(user_text))
    return 1 <= len(words) <= 5 and all(word in _FLAT_WORDS for word in words)


def raised_goals(user_text: str) -> list[str]:
    """Stages whose user_cues appear in what the user just said, in blueprint order."""
    text = f" {normalize(user_text)} "
    found = []
    for goal in POD4_BLUEPRINT.goals:
        if any(re.search(rf"\b{re.escape(cue.casefold())}\b", text) for cue in goal.user_cues):
            found.append(goal.id)
    return found


def _skip_goal(state: Pod4ConversationState, goal_id: str, user_text: str, *, dodge: bool = False) -> None:
    """Park a stage so its intent comes back later in a different tone; drop it on a hard no or a repeat.

    A skip is the user saying "skip"; a dodge is a reply that doesn't answer (pata nahi, off topic).
    """
    progress = state.goals[goal_id]
    wave = POD4_BLUEPRINT.wave
    if not dodge and hard_decline(user_text):
        _close_goal(state, goal_id, "skipped", "user declined")
        return
    if progress.parks >= wave.soft_skip_max_revisits:
        if dodge:
            _close_goal(state, goal_id, "not_covered", "no answer, even when asked again later")
        else:
            _close_goal(state, goal_id, "skipped", "user skipped twice")
        return
    progress.parks += 1
    progress.status = "parked"
    progress.revisit_after_closed = state.goals_closed + wave.soft_skip_revisit_after_goals
    if state.active_goal == goal_id:
        state.active_goal = None
        state.active_ask = None
        state.active_pending_turns = 0


def _segment(elapsed: int) -> Segment:
    clock = POD4_BLUEPRINT.call
    if elapsed >= clock.close_seconds:
        return "close"
    if elapsed >= clock.wrap_up_seconds:
        return "wrap_up"
    if elapsed < clock.segments.warm_up:
        return "warm_up"
    return "core"


def _must_cover_ids() -> list[str]:
    return [goal.id for goal in POD4_BLUEPRINT.goals if goal.must_cover]


def _behind_schedule_seconds(state: Pod4ConversationState, elapsed: int) -> int:
    """How far coverage is behind an even pace across the core window (TIME-4)."""
    clock = POD4_BLUEPRINT.call
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


def _sensitive_allowed(state: Pod4ConversationState, elapsed: int) -> bool:
    clock = POD4_BLUEPRINT.call
    return state.turn_count >= clock.sensitive_after_user_turns or elapsed >= clock.sensitive_after_seconds


def _pick_angle(state: Pod4ConversationState, goal_id: str, purpose: str) -> tuple[str, str]:
    goal = _goal(goal_id)
    progress = state.goals[goal_id]
    unused = [angle for angle in goal.angles if angle.id not in progress.angles_used]
    pool = unused or list(goal.angles)  # TOP-4: no repeat until every angle was used
    angle = _rng(state, f"angle:{goal_id}:{purpose}").choice(pool)
    progress.angles_used.append(angle.id)
    return angle.id, angle.hint


def _pick_filler(state: Pod4ConversationState, has_callback: bool) -> FillerType:
    config = POD4_BLUEPRINT.fillers
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


def _all_quotes(state: Pod4ConversationState) -> list[tuple[int, str]]:
    quotes: list[tuple[int, str]] = []
    for goal_id, progress in state.goals.items():
        if progress.status == "skipped":
            continue
        for value in progress.slots.values():
            quotes.append((value.turn, value.quote))
    quotes.extend((fact.turn, fact.quote) for fact in state.incidental_facts)
    return quotes


def _pick_callback(state: Pod4ConversationState) -> str:
    """VOICE-7: an earlier detail in the user's own words, never the latest turn."""
    voice = POD4_BLUEPRINT.voice
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


def _scenario(state: Pod4ConversationState) -> str:
    if not state.scenario:
        scenarios = next((goal.scenarios for goal in POD4_BLUEPRINT.goals if goal.scenarios), [])
        if scenarios:
            state.scenario = random.Random(f"{state.seed}:scenario").choice(scenarios)
    return state.scenario


def record_pod4_assistant_turn(state: Pod4ConversationState, text: str) -> None:
    """Remember reply openers so the Talker is told not to reuse them (VOICE-3)."""
    voice = POD4_BLUEPRINT.voice
    start = opener(text, voice.opener_words)
    if not start:
        return
    if start in state.recent_openers[-voice.opener_no_repeat_within_turns:]:
        state.opener_repeats += 1
        logger.warning("POD4 OPENER REPEAT | opener={!r} | repeats={}", start, state.opener_repeats)
    state.recent_openers = (state.recent_openers + [start])[-voice.opener_no_repeat_within_turns:]


# --- Graph step 1: merge the observation -------------------------------------


async def _merge_observation(state: Pod4GraphState) -> Pod4GraphState:
    conversation = state["conversation_state"].model_copy(deep=True)
    observation = state["observation"]
    user_text = state.get("user_text", "")
    retries = POD4_BLUEPRINT.retries
    conversation.turn_count += 1
    conversation.last_tone_cue = observation.tone_cue
    conversation.last_tone_confidence = observation.tone_confidence
    result: Pod4GraphState = {
        "resolved_outcome": "not_applicable",
        "retry_due": False,
        "repair_due": False,
        "just_covered": None,
        "opened_goals": [],
        "skip_result": "",
        "raised_now": [],
        "dodged": False,
    }

    # Safety first (priority order: safety, honesty, user choice, coverage, time).
    flagged = distress_quote(user_text)
    if not flagged and observation.distress and is_grounded(observation.distress_quote, user_text):
        flagged = observation.distress_quote
    if flagged or conversation.safety_mode:
        if flagged and not conversation.safety_mode:
            conversation.safety_mode = True
            conversation.safety_quote = flagged[:180]
            logger.warning("POD4 SAFETY PATH STARTED | turn={}", conversation.turn_count)
        return {**result, "conversation_state": conversation}

    conversation.flat_streak = conversation.flat_streak + 1 if is_flat_reply(user_text) else 0
    raised_now = [
        goal_id
        for goal_id in raised_goals(user_text)
        if conversation.goals[goal_id].status not in CLOSED_STATUSES and goal_id != conversation.active_goal
    ]
    for goal_id in raised_now:
        if goal_id not in conversation.raised_goals:
            conversation.raised_goals.append(goal_id)
    result["raised_now"] = raised_now

    # RETRY-1: code classifies obvious phrases before the model is trusted.
    code = code_outcome(user_text)
    outcome: Pod4Outcome = code or observation.outcome
    if code is None and outcome == "skip" and is_flat_reply(user_text):
        # "Okay, thanks" or "yeah, I know" is the talk going flat, not a request to skip.
        outcome = "not_applicable"
    if code is None and outcome != "not_applicable" and observation.outcome_confidence < retries.uncertain_confidence:
        # RETRY-2: an unsure classification counts as partial, so the AI asks less.
        outcome = "partial"

    # VOICE-8: if the words were not heard clearly, ask once casually, at most twice.
    # Only short, garbled turns get a "say that again"; a long reply is answered as it is.
    if observation.asr_unclear and code is None and len(user_text.split()) <= 6:
        conversation.asr_repair_attempts += 1
        if conversation.asr_repair_attempts <= retries.asr_repair_max_attempts:
            return {**result, "conversation_state": conversation, "repair_due": True}
        outcome = "not_applicable"
    conversation.asr_repair_attempts = 0

    # Store only slot quotes that really appear in the user's latest words.
    detour = conversation.detour
    detour_candidates: dict[str, dict[str, Pod4SlotValue]] = {}
    opened: list[str] = []
    for fill in observation.slots_filled:
        goal_id, _, slot_id = fill.slot.partition(".")
        if fill.slot not in POD4_SLOT_KEYS or not is_grounded(fill.quote, user_text):
            continue
        progress = conversation.goals[goal_id]
        if progress.status in ("skipped",) or slot_id in progress.slots:
            continue
        value = Pod4SlotValue(quote=fill.quote, value=fill.value, turn=conversation.turn_count)
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
                Pod4Fact(slot=f"{goal_id}.{slot_id}", quote=value.quote, value=value.value, turn=value.turn)
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
                _skip_goal(conversation, detour.pending_goal, user_text)
                conversation.detour = None
                outcome = "skip"

    # Five outcomes for the goal that was just asked (section 7).
    # An unknown turn (no tool call and no extraction) tells us nothing about the pending stage.
    active_id = conversation.active_goal if observation.source != "unknown" else None
    retry_due = False
    dodged = False
    continue_due = False
    skip_result = ""
    if active_id and conversation.detour is None:
        progress = conversation.goals[active_id]
        progress.last_outcome = outcome
        if outcome == "skip":
            # Wave: a soft skip parks the stage (its intent comes back later); a hard no ends it.
            _skip_goal(conversation, active_id, user_text)
            skip_result = progress.status
        elif conversation.active_ask == "last_chance":
            # RETRY-6: the one wrap-up question failed, so the goal stays not covered.
            _close_goal(conversation, active_id, "not_covered", "no full answer to the wrap-up question")
        elif outcome in ("full", "partial"):
            # "full" with missing slots is still partial: only slots decide coverage.
            if progress.asks >= retries.partial_max_asks:
                _close_goal(conversation, active_id, "not_covered", f"partial after {progress.asks} asks")
            else:
                # Wave: stay on this stage and ask for its missing parts next, on the person's thread.
                continue_due = True
        elif outcome in ("no_answer", "off_topic"):
            # Wave: don't push. Let it go with a filler, move to another topic, and come back
            # to this one later in a different tone; a second dodge drops it as not covered.
            _skip_goal(conversation, active_id, user_text, dodge=True)
            dodged = True
        else:
            # The user asked something back or corrected us: keep the goal pending. A reply to a turn
            # where we asked nothing (a conversational turn) doesn't count against the stage.
            if conversation.consecutive_questions > 0:
                conversation.active_pending_turns += 1
            if conversation.active_pending_turns >= 2:
                _skip_goal(conversation, active_id, user_text, dodge=True)
                dodged = True

    return {
        **result,
        "conversation_state": conversation,
        "resolved_outcome": outcome,
        "retry_due": retry_due,
        "continue_due": continue_due,
        "dodged": dodged,
        "raised_now": raised_now,
        "skip_result": skip_result,
        "just_covered": just_covered,
        "opened_goals": [goal_id for goal_id in dict.fromkeys(opened) if conversation.goals[goal_id].status not in CLOSED_STATUSES],
    }


# --- Graph step 2: plan the next turn ------------------------------------------


def _next_goal(
    state: Pod4ConversationState, elapsed: int, opened: list[str], *, behind: bool
) -> tuple[str | None, AskKind]:
    sensitive_ok = _sensitive_allowed(state, elapsed)
    # Optional backup stages fill the time once no must-cover stage is still fresh or in progress
    # (parked or half-answered ones come back later anyway).
    must_cover_open = [goal_id for goal_id in _must_cover_ids() if state.goals[goal_id].status in ("open", "active")]

    def eligible(goal_id: str) -> bool:
        goal = _goal(goal_id)
        # A sensitive stage the user raised themselves may come early.
        if goal.sensitive and not sensitive_ok and goal_id not in state.raised_goals:
            return False
        if not goal.must_cover and must_cover_open:
            return False
        return True

    # 1. Partial answers and parked stages whose comeback is due.
    due = [
        goal_id
        for goal_id in POD4_GOAL_ORDER
        if state.goals[goal_id].status in ("partial", "parked")
        and (state.goals[goal_id].revisit_after_closed or 0) <= state.goals_closed
        and eligible(goal_id)
    ]
    if due:
        return due[0], "revisit"
    # 2. Goals the last answer opened up (TOP-2).
    for goal_id in opened:
        if state.goals[goal_id].status == "open" and eligible(goal_id):
            return goal_id, "new"
    # 3. Wave: stages flow in the blueprint's order, like one natural conversation.
    fresh = [goal_id for goal_id in POD4_GOAL_ORDER if state.goals[goal_id].status == "open" and eligible(goal_id)]
    if fresh:
        return fresh[0], "new"
    # 4. Nothing new left: bring back a waiting partial or parked stage early rather than idle.
    waiting = [
        goal_id
        for goal_id in POD4_GOAL_ORDER
        if state.goals[goal_id].status in ("partial", "parked")
        and eligible(goal_id)
    ]
    if waiting:
        return waiting[0], "revisit"
    return None, "new"


def _missing_ask_for(goal_id: str, progress: Pod4GoalProgress) -> list[str]:
    """What the Talker should ask for: only the slots still missing (RETRY-4)."""
    asks = {slot.id: slot.ask_for for slot in _goal(goal_id).slots}
    return [asks[slot_id] for slot_id in _missing_slots(goal_id, progress)]


def _ask(state: Pod4ConversationState, decision: Pod4Decision, goal_id: str, kind: AskKind) -> None:
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


async def _plan_turn(state: Pod4GraphState) -> Pod4GraphState:
    conversation = state["conversation_state"].model_copy(deep=True)
    observation = state["observation"]
    elapsed = state.get("elapsed_seconds", 0)
    clock = POD4_BLUEPRINT.call
    detours = POD4_BLUEPRINT.detours
    voice = POD4_BLUEPRINT.voice

    segment = _segment(elapsed)
    if segment != conversation.segment or not conversation.timeline:
        conversation.segment = segment
        conversation.timeline.append(Pod4TimelineEntry(segment=segment, at_seconds=elapsed))
    behind_seconds = _behind_schedule_seconds(conversation, elapsed)
    conversation.behind_schedule_seconds = behind_seconds
    behind = behind_seconds > clock.behind_schedule_seconds
    tone: ToneCue = observation.tone_cue if observation.tone_confidence >= 0.65 else "neutral"

    decision = Pod4Decision(
        tone_hint=tone,
        segment=segment,
        seconds_left=max(0, clock.max_call_seconds - elapsed),
        must_avoid_openers=conversation.recent_openers[-voice.opener_no_repeat_within_turns:],
        must_avoid=list(state.get("recent_questions", [])[-3:]),
    )

    def finish(mode: ResponseMode, cue: str, reason: str, *, filler: bool = True) -> Pod4GraphState:
        decision.response_mode = mode
        decision.cue = cue
        decision.planning_reason = reason
        conversation.consecutive_questions = conversation.consecutive_questions + 1 if decision.ask_question else 0
        if decision.goal:
            stage = _goal(decision.goal)
            decision.stage_framing = stage.framing
            decision.branches = [f"if {branch.when}: {branch.ask}" for branch in stage.branches][:8]
            if stage.revive:
                decision.revive_line = _rng(conversation, f"revive:{stage.id}").choice(stage.revive)
        if filler and mode not in ("safety", "close", "repair"):
            if not decision.callback_detail and mode in ("ask_goal", "revisit", "chat", "follow_up", "wrap_up"):
                decision.callback_detail = _pick_callback(conversation)
            decision.filler_type = _pick_filler(conversation, bool(decision.callback_detail))
        return {"conversation_state": conversation, "decision": decision}

    # Safety path: the script stops for the rest of the call (SAFE-2).
    if conversation.safety_mode:
        decision.support_resources = [
            f"{resource.name}: {resource.contact}" for resource in POD4_BLUEPRINT.safety.resources
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
            "You didn't catch their words clearly. Casually ask them to say it again once, in the call's language "
            "(in English, like \"Sorry, could you say that again?\"). Don't guess what they said.",
            "asr_repair",
            filler=False,
        )

    if observation.intent == "user_question":
        if conversation.active_goal:
            decision.goal = conversation.active_goal
            decision.goal_label = POD4_GOAL_LABELS[conversation.active_goal]
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
        if state.get("skip_result") == "parked":
            return finish(
                "respect_skip",
                "They want to skip that for now. Accept it in a few easy words (like \"Fair enough\") with no "
                "pressure, then always hand it back with one easy question, either about something related they "
                "already mentioned or about another part of their life. Don't mention coming back to it.",
                "skip_parked",
            )
        return finish(
            "respect_skip",
            "They clearly don't want to talk about that. Accept it warmly in a few words and never bring it up "
            "again. Then always hand it back with one easy question about something else they've shared.",
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
                    decision.goal_label = POD4_GOAL_LABELS[pending]
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

    wave = POD4_BLUEPRINT.wave
    active_id = conversation.active_goal

    # When they sound low, sit with it before anything else (at most once every three turns).
    if (
        wave.pause_on_heavy_tone
        and tone in ("sad", "tired", "frustrated", "serious")
        and conversation.turn_count - conversation.last_heavy_pause_turn >= 3
    ):
        conversation.last_heavy_pause_turn = conversation.turn_count
        decision.ask_question = False
        return finish(
            "chat",
            "They sound low or serious. Don't ask about a new topic. Sit with what they said like a caring friend: "
            "name it simply and kindly, without fixing or advising, then gently invite them to say more if they want "
            "(for example, asking if they want to talk about it).",
            "pause_heavy_tone",
        )

    # Follow a topic the user raised themselves (for example an ex, or their parents).
    raised = [goal_id for goal_id in state.get("raised_now", []) if conversation.goals[goal_id].status not in CLOSED_STATUSES]
    if raised and segment == "core":
        target = raised[0]
        if active_id and active_id != target:
            # Leave the current stage open; it comes back later.
            current = conversation.goals[active_id]
            current.status = "partial"
            current.revisit_after_closed = conversation.goals_closed + POD4_BLUEPRINT.retries.partial_revisit_after_goals
            conversation.active_goal = None
            conversation.active_ask = None
            conversation.active_pending_turns = 0
        kind: AskKind = "revisit" if conversation.goals[target].status in ("partial", "parked") else "new"
        _ask(conversation, decision, target, kind)
        return finish(
            "ask_goal",
            "They brought this up themselves, so follow their lead now. React warmly to what they said, then ask "
            "about it as the stage framing suggests, picking the branch that fits their words.",
            "follow_user_topic",
        )

    # The talk is going flat: take the lead instead of agreeing again.
    # A reply classified as no answer (like "hmm" to a question) gets a retry instead.
    flat_due = state.get("resolved_outcome") not in ("no_answer", "off_topic") and (
        conversation.flat_streak >= wave.flat_replies_before_revive
        or (conversation.flat_streak >= 1 and conversation.consecutive_questions == 0 and conversation.turn_count > 1)
    )
    if flat_due and segment == "core":
        conversation.flat_streak = 0
        conversation.active_pending_turns = 0
        if active_id:
            decision.goal = active_id
            decision.goal_label = POD4_GOAL_LABELS[active_id]
            decision.missing = _missing_ask_for(active_id, conversation.goals[active_id])
            decision.ask_question = True
        else:
            goal_id, kind = _next_goal(conversation, elapsed, [], behind=behind)
            if goal_id:
                _ask(conversation, decision, goal_id, kind)
        return finish(
            "revive",
            "The talk is going flat. Don't agree or validate again. Take the lead like a friend: first say something "
            "light and genuine, using `revive_line` as a direction or an honest, relatable thought (never claim a "
            "human life), then ease into one question for this stage.",
            "revive_flat",
        )

    # They didn't answer: a human filler that lets it go, then a different topic.
    if state.get("dodged"):
        goal_id, kind = _next_goal(conversation, elapsed, [], behind=behind)
        if goal_id:
            _ask(conversation, decision, goal_id, kind)
            return finish(
                "ask_goal",
                "They didn't really go into that, so don't push or repeat it. Say one warm, human filler line that "
                "lets it go easily (for example: \"Fair, that's one of those things nobody thinks about until "
                "someone asks.\"), then move to this new topic with one easy, light question, connected to "
                "something they said if you can.",
                "pivot_after_dodge",
            )
        return finish(
            "chat",
            "They didn't really go into that, so let it go warmly. Keep them talking with one light, easy question "
            "about something they mentioned earlier.",
            "pivot_nothing_left",
        )

    # No answer: retry right away with a new angle and an easier form.
    if state.get("retry_due") and active_id:
        _ask(conversation, decision, active_id, "retry")
        if conversation.goals[active_id].asks >= POD4_BLUEPRINT.retries.no_answer_max_asks:
            return finish(
                "retry",
                "They still haven't really answered. Ask it plainly and directly now, in one short, kind sentence "
                "with fresh words, so it's easy to answer or to pass.",
                "no_answer_direct",
            )
        return finish(
            "retry",
            "They didn't really answer. Come at it from a different side: connect it to something they said, or "
            "make it easier with a quick example or two simple options. Completely different words than before.",
            "no_answer_retry",
        )

    # Wave: questions need conversation around them. After a few question turns in a row,
    # take one turn that only talks: react, add a layer, leave space.
    if conversation.consecutive_questions >= wave.max_consecutive_questions:
        return finish(
            "chat",
            "Don't ask a new topic question this turn. Talk with them like a friend: react to a specific thing they "
            "said, then add a layer of your own, such as a relatable thought about their situation, a bit of context "
            "about something they mentioned (a festival, a city, a job), or a link to something they said earlier. "
            "Then hand it back with one light, easy question or invitation about what they just said, so they're "
            "never left to restart the talk. Don't open a new topic.",
            "conversation_layer",
        )

    # Wave: a partial answer keeps the stage going; ask the missing parts from their answer.
    if state.get("continue_due") and active_id:
        _ask(conversation, decision, active_id, "revisit")
        return finish(
            "ask_goal",
            "Stay on this stage. Reflect what they just said, then ask about one of the missing parts in a way that "
            "grows out of their answer (for example, if they named a city, ask if they're from there or moved). "
            "Don't repeat what they already told you.",
            "continue_stage",
        )

    # A still-active stage (after a conversational turn or a correction): come back to it softly.
    # This is not a new ask, so it does not use up the stage's retry budget.
    if active_id:
        decision.goal = active_id
        decision.goal_label = POD4_GOAL_LABELS[active_id]
        decision.missing = _missing_ask_for(active_id, conversation.goals[active_id])
        decision.ask_question = True
        return finish(
            "ask_goal",
            "Respond to what they just said first, adding a little of your own. Then softly come back to the open "
            "part of this stage, growing out of their words, in new wording.",
            "return_to_open_stage",
        )

    # Deepen a rich answer once, on its own counter (RETRY-5).
    just_covered = state.get("just_covered")
    rich = observation.intent == "story" or (
        POD4_BLUEPRINT.wave.followup_on_any_full_answer and observation.intent == "answer"
    )
    if just_covered and rich and not behind:
        progress = conversation.goals[just_covered]
        if progress.followups < _goal(just_covered).max_followups:
            _ask(conversation, decision, just_covered, "follow_up")
            return finish(
                "follow_up",
                "Stay on their thread. Reflect a specific detail back, then go one step deeper: ask why, how it "
                "felt, what happened next, or what a vague word they used actually means to them. Don't move to "
                "a new topic yet.",
                "deepen_answer",
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
        conversation.detour = Pod4Detour(
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

    was_parked = conversation.goals[goal_id].status == "parked"
    _ask(conversation, decision, goal_id, kind)
    if was_parked:
        cue = (
            "Bring this up again from a completely different side and in a lighter, different tone, growing out of "
            "something they said recently. Ask only for the missing part, in new words, and never mention that they "
            "skipped or avoided it before."
        )
    elif kind == "revisit":
        cue = (
            "Casually call back to this earlier topic and ask only for the missing part, in new words. Don't repeat "
            "what they already told you."
        )
    else:
        cue = (
            "Briefly reflect what they just said (and, if it's honest, one short observation about what they did "
            "or how they handled it), then bridge naturally into this stage using the angle as a direction, not a "
            "script, and the stage framing if there is one. One easy question; don't announce a topic."
        )
    return finish("revisit" if kind == "revisit" else "ask_goal", cue, f"{kind}_goal")


def _build_graph():
    graph = StateGraph(Pod4GraphState)
    graph.add_node("merge_observation", _merge_observation)
    graph.add_node("plan_turn", _plan_turn)
    graph.add_edge(START, "merge_observation")
    graph.add_edge("merge_observation", "plan_turn")
    graph.add_edge("plan_turn", END)
    return graph.compile()


pod4_conductor_graph = _build_graph()


async def plan_pod4_turn(
    conversation: Pod4ConversationState,
    observation: Pod4TurnObservation,
    *,
    user_text: str,
    elapsed_seconds: int,
    recent_questions: list[str] | None = None,
) -> tuple[Pod4ConversationState, Pod4Decision]:
    """Run one Conductor turn on plain values; used by the call and by tests."""
    result = await pod4_conductor_graph.ainvoke(
        {
            "conversation_state": conversation,
            "observation": observation,
            "user_text": user_text,
            "elapsed_seconds": elapsed_seconds,
            "recent_questions": recent_questions or [],
        }
    )
    return result["conversation_state"], result["decision"]


async def run_pod4_conductor(session: "CallSession", observation: Pod4TurnObservation) -> Pod4Decision:
    """Update only Pod 4's isolated state and produce its next-turn guidance."""
    async with session.lock:
        conversation, decision = await plan_pod4_turn(
            session.pod4_state,
            observation,
            user_text=session.latest_user_text,
            elapsed_seconds=session.elapsed_seconds,
            recent_questions=session.recent_questions,
        )
        session.pod4_state = conversation
        session.covered_goals = [
            goal_id for goal_id, progress in conversation.goals.items() if progress.status == "covered"
        ]
        logger.info(
            "POD4 PLANNER | call={} | seed={} | turn={} | segment={} | behind={}s | mode={} | reason={} | goal={} | "
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


def pod4_coverage_report(conversation: Pod4ConversationState, *, final: bool = False) -> list[dict[str, object]]:
    """TOP-1 / RETRY-7: every goal is covered, skipped, or "not covered" with a reason.

    With `final=True`, goals still open when the call ends are reported as not
    covered (never guessed), so the Blueprint can show them for the user to fill in.
    """
    report: list[dict[str, object]] = []
    for goal in POD4_BLUEPRINT.goals:
        progress = conversation.goals[goal.id]
        status: str = progress.status
        reason = progress.reason
        if final and status == "parked":
            status, reason = "not_covered", "skipped once and not answered later"
        elif final and status not in CLOSED_STATUSES:
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
