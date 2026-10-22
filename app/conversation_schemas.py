"""Shared building blocks for the Pod 4 and Pod 5 call schemas."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict


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


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
