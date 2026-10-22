"""Decide whether a piece of STT text is real speech or noise the STT made up.

Whisper (and other STTs) sometimes return words for a breath, a click or silence; the
classic ones are "Thank you.", "Good bye." and "Thanks for watching." This guard looks at
how much the caller actually spoke since the previous piece of text (from VAD timing) and
how loud it was (from the raw mic audio), so it works the same with any STT provider.

Each piece of text is first matched to the stretch of speech (VAD segment) it came from:
segment-based STTs like Whisper return text in the same order as the segments, but a
segment can give no text at all. So a piece belongs to the first waiting segment long
enough to hold its words (at most ~7 words a second); earlier segments gave nothing.

VAD sometimes flickers off and on in the middle of real speech, leaving a tiny segment, while
the STT still hears the words (it keeps some audio from before VAD's start, the pre-roll). So
each segment also records how loud the mic was over the pre-roll before it: loud means real
speech was going on, and the segment gets PRE_ROLL_SECONDS more room for words.

Rules:
  - empty text: drop;
  - no waiting segment could hold it, at most 4 words, and the audio around it was quiet: drop;
  - YouTube-style phrases ("thanks for watching", "subscribe", subtitle credits): always drop;
  - from under 0.2 s of quiet speech and at most 4 words: drop (a blip);
  - a stock phrase ("good bye", "thank you", "bye", ...) with under 0.8 s of speech, or
    from very quiet audio: drop. A real, clearly spoken "thank you" is kept.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

BLIP_SECONDS = 0.2
STOCK_PHRASE_SECONDS = 0.8
QUIET_DB = -45.0
# Pre-roll louder than this means real speech was going on around the segment.
SPEECH_AROUND_DB = -35.0
PRE_ROLL_SECONDS = 0.8

_ALWAYS_NOISE = (
    "thanks for watching",
    "thank you for watching",
    "please subscribe",
    "like and subscribe",
    "subtitles by",
    "amara.org",
    "transcribed by",
)
_STOCK_PHRASES = {
    "good bye", "goodbye", "bye", "bye bye", "thank you", "thanks", "thank you so much",
    "thank you very much", "you", "okay bye", "see you", "see you next time",
}
_PUNCTUATION = re.compile(r"[^\w\s']", re.UNICODE)


def normalise(text: str) -> str:
    return " ".join(_PUNCTUATION.sub(" ", text.lower()).split())


def check_piece(text: str, speech_seconds: float, loudness_db: float | None) -> tuple[bool, str]:
    """(keep, reason). `speech_seconds` is VAD speech since the previous piece of text."""
    words = normalise(text)
    if not words:
        return False, "empty"
    if any(phrase in words for phrase in _ALWAYS_NOISE):
        return False, "known STT filler phrase"
    if speech_seconds < BLIP_SECONDS and len(words.split()) <= 4:
        return False, f"only {speech_seconds:.2f}s of speech (a blip)"
    if words in _STOCK_PHRASES and (
        speech_seconds < STOCK_PHRASE_SECONDS or (loudness_db is not None and loudness_db < QUIET_DB)
    ):
        return False, "stock phrase from very short or quiet audio (likely made up)"
    return True, "ok"


# --- matching text to speech segments -------------------------------------------------------

MAX_WORDS_PER_SECOND = 7.0


@dataclass
class Segment:
    """One stretch of caller speech between VAD start and stop."""

    ended_at: float
    seconds: float
    loudness_db: float | None
    # Loudness from PRE_ROLL_SECONDS before the segment started until it ended.
    pre_roll_db: float | None = None

    @property
    def speech_around(self) -> bool:
        return self.pre_roll_db is not None and self.pre_roll_db >= SPEECH_AROUND_DB

    @property
    def heard_seconds(self) -> float:
        """How much audio the STT may have heard words in."""
        return self.seconds + (PRE_ROLL_SECONDS if self.speech_around else 0.0)


def word_count(text: str) -> int:
    return len(normalise(text).split())


def match_segment(segments: list[Segment], words: int) -> tuple[Segment | None, int]:
    """(segment the text came from, how many waiting segments to remove).

    The text belongs to the first segment that could hold that many words; segments before it
    produced no text. If none fits, nothing is removed and the segment is None.
    """
    for index, segment in enumerate(segments):
        if words <= MAX_WORDS_PER_SECOND * segment.heard_seconds + 1:
            return segment, index + 1
    return None, 0


def check_with_segments(text: str, segments: list[Segment]) -> tuple[bool, str, Segment | None, int]:
    """Match `text` to a waiting segment and decide whether to keep it.

    Returns (keep, reason, matched segment, how many segments to remove from the queue).
    """
    words = word_count(text)
    segment, used = match_segment(segments, words)
    if segment is None:
        if words <= 4:
            return False, "no matching speech (likely made up)", None, 0
        return True, "ok (no timing)", None, 0
    keep, reason = check_piece(text, segment.heard_seconds, segment.loudness_db)
    return keep, reason, segment, used


# --- streaming STTs (Sarvam) -------------------------------------------------------------

# A streaming STT sends text whenever it finalises: at its own pauses (often while the caller is
# still talking), and again when we flush it on our VAD stop. One stretch of speech can give
# several pieces, and one piece can span several stretches, so pieces are not matched one-to-one.
# Instead each piece is checked against all caller speech in the last STREAM_WINDOW_SECONDS,
# including the stretch still going on.
STREAM_WINDOW_SECONDS = 6.0


def check_streamed(text: str, recent: list[Segment]) -> tuple[bool, str, Segment | None]:
    """(keep, reason, the recent speech taken together) for a piece from a streaming STT."""
    if not recent:
        if word_count(text) <= 4:
            return False, "no recent speech (likely made up)", None
        return True, "ok (no timing)", None
    louds = [segment.loudness_db for segment in recent if segment.loudness_db is not None]
    pre_rolls = [segment.pre_roll_db for segment in recent if segment.pre_roll_db is not None]
    speech = Segment(
        ended_at=max(segment.ended_at for segment in recent),
        seconds=sum(segment.seconds for segment in recent),
        loudness_db=max(louds) if louds else None,
        pre_roll_db=max(pre_rolls) if pre_rolls else None,
    )
    keep, reason = check_piece(text, speech.heard_seconds, speech.loudness_db)
    return keep, reason, speech
