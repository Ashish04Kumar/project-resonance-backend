"""Profile helpers. For now: polish the user's introduction with the configured LLM."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from loguru import logger
from pydantic import BaseModel, Field

from . import pod4_providers
from .auth import UserOut, current_user
from .pod4_providers import Message

router = APIRouter(prefix="/api/v1/profile", tags=["profile"])

MAX_BIO = 280

_POLISH_RULES = (
    "You polish a short dating-profile introduction written by the user. Keep it in their own voice, first person, "
    "and keep every fact they wrote; never add facts, hobbies, names or places they didn't mention. Make it warm, "
    "natural and easy to read. Match their length: about the same length as their text or a little shorter, never "
    "longer, and at most 280 characters. Keep it short. No emojis, hashtags, quotes or lists. Reply with only the "
    "polished introduction."
)


class PolishRequest(BaseModel):
    text: str = Field(min_length=10, max_length=MAX_BIO)


class PolishResponse(BaseModel):
    text: str


def _clean(raw: str, original: str) -> str:
    text = " ".join(raw.strip().strip('"').strip("“”").split())
    limit = min(MAX_BIO, max(len(original) + 40, 60))
    if len(text) > limit:
        # Trim at the last full sentence (or word) that fits.
        cut = text[:limit]
        end = max(cut.rfind(". "), cut.rfind("! "), cut.rfind("? "))
        text = cut[: end + 1] if end > limit // 2 else cut.rsplit(" ", 1)[0].rstrip(",;:") + "."
    return text


@router.post("/polish-bio", response_model=PolishResponse)
async def polish_bio(payload: PolishRequest, user: Annotated[UserOut, Depends(current_user)]) -> PolishResponse:
    original = " ".join(payload.text.split())
    messages = [Message("system", _POLISH_RULES), Message("user", original)]
    try:
        raw, provider = await pod4_providers.complete(messages, temperature=0.6, max_tokens=300)
    except Exception as error:
        logger.warning("PROFILE POLISH FAILED | user={} | {}", user.id, error)
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "Polish tone isn't available right now. Please try again.") from None
    text = _clean(raw, original)
    if not text:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "Polish tone isn't available right now. Please try again.")
    logger.info("PROFILE POLISH | user={} | provider={} | {} -> {} chars", user.id, provider, len(original), len(text))
    return PolishResponse(text=text)
