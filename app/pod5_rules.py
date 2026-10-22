"""Pod 5 code-first rules: phrases that code can classify without an LLM.

RETRY-1: obvious phrases are classified by code before trusting the model.
SAFE-1/2: a phrase-based safety check runs on every user turn. It is a fast
first line only; Gemini's own distress flag is the second line.
Pure functions only; no network calls and no provider imports (PROV-5).
"""

from __future__ import annotations

import re
from typing import Literal


CodeOutcome = Literal["skip", "no_answer"]


def normalize(text: str) -> str:
    """Lower-case, collapse spaces, and trim punctuation for quote matching."""
    return " ".join(text.casefold().split()).strip(" \t\n.,!?\"'“”‘’…")


def is_grounded(quote: str, user_text: str) -> bool:
    """A quote counts only when it really appears in what the user just said."""
    cleaned = normalize(quote)
    return bool(cleaned) and cleaned in normalize(user_text)


# Phrases that clearly mean "skip this question" (English and Hinglish).
_SKIP_PATTERNS = (
    r"\bskip( (it|this|that|this one|this question))?\b",
    r"\bnext( (one|question))?\b",
    r"\bpass\b",
    r"\bbaad mein\b",
    r"\bbaad me\b",
    r"\bchhodo\b",
    r"\bchodo\b",
    r"\brehne do\b",
    r"\bnahi batana\b",
    r"\bnahin batana\b",
    r"\bdon'?t want to (talk|answer|share)\b",
    r"\brather not (say|share|talk)\b",
    r"\bmove on\b",
)
# Phrases that clearly mean "I have no answer".
_NO_ANSWER_PATTERNS = (
    r"\bpata nahi+n?\b",
    r"\bpata nai\b",
    r"\bmalum nahi+n?\b",
    r"\bmaloom nahi+n?\b",
    r"\bi (really )?don'?t know\b",
    r"\bdunno\b",
    r"\bno idea\b",
    r"\bnot sure\b",
    r"\bnever thought about (it|that)\b",
    r"\bkuch nahi+n?\b",
)
# Self-harm or abuse signals. Matched anywhere in the turn, at any length.
_DISTRESS_PATTERNS = (
    r"\bkill myself\b",
    r"\bend (it all|my life)\b",
    r"\bwant to die\b",
    r"\bwanna die\b",
    r"\bsuicid",
    r"\bself[- ]?harm",
    r"\bhurt(ing)? myself\b",
    r"\bcut(ting)? myself\b",
    r"\bno reason to live\b",
    r"\bmarna chaht[ai]\b",
    r"\bmar jaana chaht[ai]\b",
    r"\bmar jaun\b",
    r"\bjeena nahi+n? chaht[ai]\b",
    r"\bkhudkushi\b",
    r"\baatmahatya\b",
    r"\b(he|she|they) (hits|beats|hurts) me\b",
    r"\bbeing abused\b",
    r"\babusing me\b",
    r"\bmujhe maart[ai] h(ai|e)\b",
)

# Only short replies are read as a skip or a no-answer; a long reply that
# happens to contain "next" or "not sure" is a real answer.
_SHORT_REPLY_WORDS = 8


def code_outcome(user_text: str) -> CodeOutcome | None:
    text = normalize(user_text)
    if not text or len(text.split()) > _SHORT_REPLY_WORDS:
        return None
    if any(re.search(pattern, text) for pattern in _SKIP_PATTERNS):
        return "skip"
    if any(re.search(pattern, text) for pattern in _NO_ANSWER_PATTERNS):
        return "no_answer"
    return None


def distress_quote(user_text: str) -> str:
    """Return the matching words when a turn shows self-harm or abuse signals."""
    text = normalize(user_text)
    for pattern in _DISTRESS_PATTERNS:
        match = re.search(pattern, text)
        if match:
            return match.group(0)
    return ""


def opener(text: str, words: int) -> str:
    """The first few words of a reply, used for the no-repeated-openers rule."""
    cleaned = re.sub(r"[^\w\s']", " ", text.casefold())
    return " ".join(cleaned.split()[:words])
