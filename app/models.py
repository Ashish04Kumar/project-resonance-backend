"""HTTP request and response shapes for the text prototype."""

from typing import Literal
from pydantic import BaseModel, Field


LanguageMode = Literal["english", "hinglish"]
ConversationPodId = Literal["pod1", "pod2"]


class CreateCallRequest(BaseModel):
    language_mode: LanguageMode = "english"
    consent: bool
    client_host: str | None = None


class CreateCallResponse(BaseModel):
    call_id: str
    conversation_pod: ConversationPodId
    assistant_text: str
    language_mode: LanguageMode
    elapsed_seconds: int
    phase: Literal["conversation", "wrap_up", "closed"]
    provider: str
    livekit_url: str
    livekit_token: str


class VoiceJoinResponse(BaseModel):
    call_id: str
    assistant_text: str
    livekit_url: str
    livekit_token: str


class TurnRequest(BaseModel):
    text: str = Field(min_length=1, max_length=4000)


class TurnResponse(BaseModel):
    call_id: str
    assistant_text: str
    language_mode: LanguageMode
    elapsed_seconds: int
    phase: Literal["conversation", "wrap_up", "closed"]
    covered_goals: list[str]
    provider: str


class CallStatusResponse(BaseModel):
    call_id: str
    conversation_pod: ConversationPodId
    language_mode: LanguageMode
    elapsed_seconds: int
    phase: Literal["conversation", "wrap_up", "closed"]
    covered_goals: list[str]
