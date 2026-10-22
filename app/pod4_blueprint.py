"""Load and validate Pod 4's Wave-style blueprint (a copy of Pod 6's, kept separate).

Pod 4 reuses Pod 5's blueprint shape (goals, slots, angles, limits, clock) and
adds a `wave` block for the Wave-style rules: stages in a fixed order, a soft
skip that parks a stage and brings it back later, and deeper follow-ups.
Validated once at import, so a broken file stops the backend at startup.
Pure: no network calls and no provider imports.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import yaml
from pydantic import Field

from .pod5_blueprint import Blueprint, Goal, _Strict


DEFAULT_POD4_BLUEPRINT_PATH = Path(__file__).resolve().parents[1] / "blueprints" / "pod4_blueprint.yaml"


class WaveStyle(_Strict):
    # A soft skip ("skip", "baad mein") parks the stage; it comes back after this many other stages close.
    soft_skip_revisit_after_goals: int = Field(ge=0)
    # How many times a parked stage may come back before a further skip drops it for good.
    soft_skip_max_revisits: int = Field(ge=0)
    # Follow up on any full answer (not only stories), up to each stage's max_followups.
    followup_on_any_full_answer: bool
    # After this many question turns in a row, take one conversational turn with no new question.
    max_consecutive_questions: int = Field(ge=1)
    # When the user sounds sad, tired, frustrated or serious, sit with it for a turn before asking more.
    pause_on_heavy_tone: bool
    # Short flat replies in a row ("okay", "yeah", "thanks") before the AI takes the lead to revive the talk.
    flat_replies_before_revive: int = Field(ge=1)


class Branch(_Strict):
    # What the reply looks like, and the follow-up direction that fits it.
    when: str = Field(min_length=3)
    ask: str = Field(min_length=3)


class WaveGoal(Goal):
    # How a person would ask this stage (tone, permission, honest reason). Sent to the Talker with the ask.
    framing: str = ""
    # Next-question directions that depend on how they answered.
    branches: list[Branch] = Field(default_factory=list)
    # Light lines the AI can lead with when the talk goes flat on this stage.
    revive: list[str] = Field(default_factory=list)
    # Words that mean the user raised this stage themselves; it then comes next, even if sensitive.
    user_cues: list[str] = Field(default_factory=list)


class WaveBlueprint(Blueprint):
    wave: WaveStyle
    goals: list[WaveGoal] = Field(min_length=1)

    def goal(self, goal_id: str) -> WaveGoal:
        for goal in self.goals:
            if goal.id == goal_id:
                return goal
        raise KeyError(goal_id)


def load_pod4_blueprint(path: Path | str | None = None) -> WaveBlueprint:
    source = Path(path or os.getenv("SYNQ_POD4_BLUEPRINT", DEFAULT_POD4_BLUEPRINT_PATH))
    with source.open(encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    return WaveBlueprint.model_validate(data)


@lru_cache(maxsize=1)
def get_pod4_blueprint() -> WaveBlueprint:
    return load_pod4_blueprint()
