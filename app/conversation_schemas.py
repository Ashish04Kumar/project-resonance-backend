"""Strict, short-lived schemas for the live relationship conversation."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


RelationshipTopic = Literal[
    "weekend_downtime",
    "conflict",
    "partner_wants",
    "family_future",
    "values_priorities",
    "communication_affection",
    "real_life_scenario",
]
TurnIntent = Literal[
    "small_talk",
    "user_question",
    "answer",
    "story",
    "correction",
    "skip",
    "other",
]
ToneCue = Literal[
    "neutral",
    "upbeat",
    "curious",
    "playful",
    "tired",
    "sad",
    "frustrated",
    "uncertain",
    "serious",
]
QuestionAngle = Literal[
    "recent_story",
    "everyday_preference",
    "specific_example",
    "light_hypothetical",
]
TopicStatus = Literal["new", "opened", "covered", "skipped"]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class TopicSignal(StrictModel):
    topic: RelationshipTopic
    # Must be quoted from the latest transcript; paraphrased inferences are discarded.
    evidence: str = Field(min_length=1, max_length=180)
    depth: Literal["mention", "partial", "clear"] = "mention"
    confidence: float = Field(default=0.5, ge=0, le=1)


class TurnObservation(StrictModel):
    """Gemini's concise reading of one completed user turn."""

    intent: TurnIntent = "other"
    tone_cue: ToneCue = "neutral"
    tone_confidence: float = Field(default=0.0, ge=0, le=1)
    topic_signals: list[TopicSignal] = Field(default_factory=list, max_length=3)
    skip_topics: list[RelationshipTopic] = Field(default_factory=list, max_length=3)
    explicit_preferences: list[str] = Field(default_factory=list, max_length=3)


class TopicProgress(StrictModel):
    status: TopicStatus = "new"
    evidence: list[str] = Field(default_factory=list, max_length=4)
    angles_used: list[QuestionAngle] = Field(default_factory=list, max_length=4)


def _empty_topic_map() -> dict[RelationshipTopic, TopicProgress]:
    topics: tuple[RelationshipTopic, ...] = (
        "weekend_downtime",
        "conflict",
        "partner_wants",
        "family_future",
        "values_priorities",
        "communication_affection",
        "real_life_scenario",
    )
    return {topic: TopicProgress() for topic in topics}


class ConversationState(StrictModel):
    """Temporary call state; never written to disk or treated as a diagnosis."""

    topics: dict[RelationshipTopic, TopicProgress] = Field(default_factory=_empty_topic_map)
    explicit_preferences: list[str] = Field(default_factory=list, max_length=12)
    recent_angles: list[QuestionAngle] = Field(default_factory=list, max_length=6)
    turn_count: int = Field(default=0, ge=0)
    turns_since_topic: int = Field(default=0, ge=0)
    last_goal: RelationshipTopic | None = None
    last_tone_cue: ToneCue = "neutral"
    last_tone_confidence: float = Field(default=0.0, ge=0, le=1)


class ConductorDecision(StrictModel):
    """Private guidance returned to Gemini's Talker after local state update."""

    response_mode: Literal[
        "chat",
        "answer_user",
        "acknowledge_correction",
        "respect_skip",
        "reflect",
        "explore",
        "wrap_up",
    ] = "chat"
    goal: RelationshipTopic | None = None
    angle: QuestionAngle | None = None
    depth: Literal["light", "follow_up"] = "light"
    tone_hint: ToneCue = "neutral"
    must_avoid: list[str] = Field(default_factory=list, max_length=8)
    ask_question: bool = False
    grounding_detail: str = Field(default="", max_length=180)
    cue: str = Field(default="Stay with the user's thread.", max_length=300)

