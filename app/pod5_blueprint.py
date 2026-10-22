"""Load and validate Pod 5's call blueprint (goals, slots, limits, clock).

The blueprint file decides what the call must learn and every limit the
Conductor follows (requirement TOP-6). It is validated once, at import time,
so a broken file stops the backend at startup instead of during a call.
This module is pure: no network calls and no provider imports (PROV-5).
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


DEFAULT_BLUEPRINT_PATH = Path(__file__).resolve().parents[1] / "blueprints" / "call_blueprint.yaml"

AngleKind = Literal["story", "hypothetical", "preference", "reaction"]
DetourType = Literal["user_thread", "light_tangent", "playful_what_if"]
FillerType = Literal["agreement", "small_reaction", "thinking_aloud", "callback", "light_aside"]
ReturnStyle = Literal["soft_link", "honest_return", "light_joke"]


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class Segments(_Strict):
    warm_up: int = Field(ge=0)
    core: int = Field(ge=0)
    lab_handoff: int = Field(ge=0)
    lab_follow_up: int = Field(ge=0)
    close: int = Field(ge=0)


class CallClock(_Strict):
    max_call_seconds: int = Field(gt=0)
    wrap_up_seconds: int = Field(gt=0)
    close_seconds: int = Field(gt=0)
    lab_enabled: bool = False
    segments: Segments
    behind_schedule_seconds: int = Field(ge=0)
    breather_when_ahead_seconds: int = Field(ge=0)
    sensitive_after_user_turns: int = Field(ge=0)
    sensitive_after_seconds: int = Field(ge=0)

    @model_validator(mode="after")
    def _ordered(self) -> "CallClock":
        if not (self.segments.warm_up < self.wrap_up_seconds < self.close_seconds < self.max_call_seconds):
            raise ValueError("Expected warm_up < wrap_up_seconds < close_seconds < max_call_seconds")
        return self


class Retries(_Strict):
    partial_max_asks: int = Field(ge=1)
    partial_revisit_after_goals: int = Field(ge=0)
    no_answer_max_asks: int = Field(ge=1)
    uncertain_confidence: float = Field(ge=0, le=1)
    asr_repair_max_attempts: int = Field(ge=0)


class Detours(_Strict):
    max_per_call: int = Field(ge=0)
    max_turns: int = Field(ge=1)
    earliest_seconds: int = Field(ge=0)
    roll_probability: float = Field(ge=0, le=1)
    types: dict[DetourType, float]
    tangent_themes: list[str] = Field(min_length=1)
    return_styles: list[ReturnStyle] = Field(min_length=1)
    max_return_attempts: int = Field(ge=1)


class Fillers(_Strict):
    types: dict[FillerType, float]
    no_repeat_type_within_turns: int = Field(ge=0)
    no_repeat_phrase_within_turns: int = Field(ge=0)


class Voice(_Strict):
    min_callbacks_per_call: int = Field(ge=0)
    callback_every_user_turns: int = Field(ge=1)
    opener_no_repeat_within_turns: int = Field(ge=1)
    opener_words: int = Field(ge=1, le=6)


class SafetyResource(_Strict):
    name: str
    contact: str


class Safety(_Strict):
    resources: list[SafetyResource] = Field(min_length=1)


class Slot(_Strict):
    id: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    ask_for: str = Field(min_length=3)


class Angle(_Strict):
    id: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    kind: AngleKind
    hint: str = Field(min_length=3)


class Goal(_Strict):
    id: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    label: str
    must_cover: bool
    skippable: bool
    sensitive: bool
    max_followups: int = Field(ge=0)
    slots: list[Slot] = Field(min_length=1)
    angles: list[Angle] = Field(min_length=4, max_length=6)  # TOP-4
    scenarios: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _unique_ids(self) -> "Goal":
        if len({slot.id for slot in self.slots}) != len(self.slots):
            raise ValueError(f"Goal {self.id} has duplicate slot ids")
        if len({angle.id for angle in self.angles}) != len(self.angles):
            raise ValueError(f"Goal {self.id} has duplicate angle ids")
        return self

    def slot_key(self, slot_id: str) -> str:
        return f"{self.id}.{slot_id}"


class Blueprint(_Strict):
    version: int
    call: CallClock
    retries: Retries
    detours: Detours
    fillers: Fillers
    voice: Voice
    safety: Safety
    goals: list[Goal] = Field(min_length=1)

    @model_validator(mode="after")
    def _unique_goals(self) -> "Blueprint":
        if len({goal.id for goal in self.goals}) != len(self.goals):
            raise ValueError("Blueprint has duplicate goal ids")
        return self

    def goal(self, goal_id: str) -> Goal:
        for goal in self.goals:
            if goal.id == goal_id:
                return goal
        raise KeyError(goal_id)

    @property
    def goal_ids(self) -> tuple[str, ...]:
        return tuple(goal.id for goal in self.goals)

    @property
    def slot_keys(self) -> tuple[str, ...]:
        """Every slot as "goal_id.slot_id", the id the Talker's tool call uses."""
        return tuple(goal.slot_key(slot.id) for goal in self.goals for slot in goal.slots)


def load_blueprint(path: Path | str | None = None) -> Blueprint:
    source = Path(path or os.getenv("SYNQ_POD5_BLUEPRINT", DEFAULT_BLUEPRINT_PATH))
    with source.open(encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    return Blueprint.model_validate(data)


@lru_cache(maxsize=1)
def get_blueprint() -> Blueprint:
    return load_blueprint()
