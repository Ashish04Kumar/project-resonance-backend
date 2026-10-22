"""Pod 4 turn engine: read the caller's turn, run the Conductor, then write Synthe's reply.

Pod 4 has no Gemini Live, so every caller turn goes through three steps in order:
  1. observe: the Pod 4 LLM reads what the caller said (and Synthe's last question)
     and returns the same JSON observation Gemini Live's tool call used to send;
  2. the Conductor (`conversation_pod4.py`, pure code) updates state and plans;
  3. talk: the Pod 4 LLM words the reply from the Conductor's guidance.
The Conductor runs before every reply, so its guidance is never a turn late.

On voice calls each caller turn can also carry a short note on how they sounded
(voice_cues.py: pause before answering, pace, loudness, intonation, fillers). The
observer may use it as a soft hint for tone_cue; the Talker uses it only to shape tone
and pacing. Words always win, and Synthe never mentions the cues.
The Conductor must never import a provider (PROV-5); the LLM calls live here.
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import TYPE_CHECKING

from loguru import logger

from . import pod4_live_debug as live_debug  # TEMPORARY live terminal view
from . import pod4_providers
from .conversation_pod4 import (
    POD4_BLUEPRINT,
    POD4_GOAL_LABELS,
    Pod4Decision,
    Pod4TurnObservation,
)
from .pod4_providers import Message
from .pod5_rules import code_outcome, distress_quote

if TYPE_CHECKING:
    from .conversation import CallSession


def _slot_list() -> str:
    return "\n".join(
        f"- {goal.slot_key(slot.id)}: {slot.ask_for}" for goal in POD4_BLUEPRINT.goals for slot in goal.slots
    )


_OBSERVATION_DEFAULTS = Pod4TurnObservation().model_dump()


def _coerce_observation(data: dict) -> Pod4TurnObservation:
    """Accept slightly messy model JSON: nulls become defaults, unknown keys and bad slots are dropped."""
    clean: dict = {"source": "extracted"}
    for key, default in _OBSERVATION_DEFAULTS.items():
        if key in ("source", "slots_filled"):
            continue
        value = data.get(key)
        clean[key] = default if value is None else value
    for key in ("tone_confidence", "outcome_confidence"):
        try:
            clean[key] = min(1.0, max(0.0, float(clean[key])))
        except (TypeError, ValueError):
            clean[key] = 0.0
    slots = []
    for item in data.get("slots_filled") or []:
        if isinstance(item, dict) and item.get("slot") and item.get("quote"):
            slots.append({"slot": str(item["slot"]), "quote": str(item["quote"])[:180], "value": str(item.get("value") or "")[:120]})
    clean["slots_filled"] = slots[:6]
    try:
        return Pod4TurnObservation.model_validate(clean)
    except Exception:
        # An unexpected enum value: keep the slots and the safety flags, drop the rest.
        return Pod4TurnObservation.model_validate(
            {
                "source": "extracted",
                "slots_filled": slots[:6],
                "distress": bool(data.get("distress")),
                "distress_quote": str(data.get("distress_quote") or "")[:180],
            }
        )


def _code_only_observation(user_text: str) -> Pod4TurnObservation:
    """When the LLM is unavailable: trust only clear phrases; otherwise the turn is unknown, never a dodge."""
    outcome = code_outcome(user_text)
    quote = distress_quote(user_text)
    return Pod4TurnObservation(
        source="extracted" if outcome else "unknown",
        intent="skip" if outcome == "skip" else "other",
        outcome=outcome or "not_applicable",
        outcome_confidence=0.9 if outcome else 0.0,
        distress=bool(quote),
        distress_quote=quote,
    )


_VOICE_NOTE_FOR_OBSERVER = (
    " A line 'How they sounded' may follow the caller's words: rough measurements of their delivery compared "
    "with their own earlier turns (pauses, pace, loudness, intonation). Use it only as a soft hint for tone_cue and "
    "tone_confidence (for example a long pause, slower pace and trailing off can mean uncertain or tired). The words "
    "always win; never let it change outcome or slots."
)

_VOICE_NOTE_FOR_TALKER = (
    "How they sounded just now (rough measurements, may be wrong): {note}. Let this shape only your tone and "
    "pacing: if they hesitated, slowed down, trailed off or went quiet, be gentler, slower and leave more room; if "
    "they sounded lively or fast, match their energy a little. Never mention their voice, pauses, speed or these "
    "measurements, and never name an emotion they didn't say.\n"
)


async def observe_turn(session: "CallSession", user_text: str, voice_note: str = "") -> Pod4TurnObservation:
    state = session.pod4_state
    last_question = session.recent_questions[-1] if session.recent_questions else ""
    pending = (
        f"Synthe's open question is about: {POD4_GOAL_LABELS[state.active_goal]}. "
        if state.active_goal
        else "No stage question is open; use outcome not_applicable unless they answered something anyway. "
    )
    messages = [
        Message(
            "system",
            "You read one caller turn from a casual voice call about dating and relationships. Return only a JSON "
            "object with keys intent, tone_cue, tone_confidence, outcome, outcome_confidence, slots_filled, "
            "distress, distress_quote, asr_unclear. intent: small_talk, user_question, answer, story, correction, "
            "skip or other. tone_cue: neutral, upbeat, curious, playful, tired, sad, frustrated, uncertain or "
            "serious. outcome for the open question: full, partial, no_answer, off_topic, skip (they clearly ask to "
            "pass) or not_applicable. A rich answer about something else is off_topic for the question but still "
            "fill its slots. slots_filled: list of {slot, quote, value}; slot must be one of the ids below, quote "
            "an exact short quote copied from the caller's words, value a short English summary. Fill every slot "
            "they actually stated, for any stage. Never infer what they did not say. distress only for self-harm "
            "or abuse. asr_unclear only if the words look garbled." + (_VOICE_NOTE_FOR_OBSERVER if voice_note else "") + " " + pending
            + "Slots:\n" + _slot_list(),
        ),
        Message(
            "user",
            f"Synthe asked: {last_question or '(no question)'}\nCaller said: {user_text}"
            + (f"\nHow they sounded: {voice_note}" if voice_note else ""),
        ),
    ]
    try:
        raw, provider = await pod4_providers.complete(messages, temperature=0.1, max_tokens=1024)
        live_debug.block("observe", f"observer LLM raw answer ({provider})", raw.strip())
        start, end = raw.find("{"), raw.rfind("}")
        if start < 0 or end < start:
            raise ValueError("no JSON object in the observation")
        observation = _coerce_observation(json.loads(raw[start : end + 1]))
        live_debug.block("observe", "observation object (Pod4TurnObservation)", observation)
        logger.info(
            "POD4 OBSERVATION | call={} | provider={} | outcome={} | slots={}",
            session.call_id,
            provider,
            observation.outcome,
            [fill.slot for fill in observation.slots_filled],
        )
        return observation
    except Exception as exc:
        logger.warning("POD4 OBSERVATION FALLBACK (code rules) | call={} | {}", session.call_id, exc)
        fallback = _code_only_observation(user_text)
        live_debug.block("observe", f"observer failed ({exc}); code-only observation", fallback)
        return fallback


_TALKER_RULES = (
    "You are Synthe, an AI on a warm, casual voice call. You are the Talker only: private guidance from the "
    "Conductor (below) decides what the turn is for; you decide how a warm, sharp, curious friend would say it.\n"
    "- Build each reply in layers: react to something specific they said; add a layer of your own (a relatable "
    "thought, a bit of context about something they mentioned, or a link to something they said earlier); then "
    "hand it back with exactly one light question or invitation. Never end on just an agreement or 'interesting', "
    "and never ask two questions in one reply. 2 to 4 short spoken sentences, about 50 words at most, no lists.\n"
    "- Reflect honestly; a short observation about what they did is fine. Never label their personality, "
    "diagnose, score, judge, or give advice. No generic praise like great, amazing or wonderful.\n"
    "- response_mode: ask_goal/revisit/last_chance ask about `goal_label`, only the `missing` parts, using "
    "`angle_hint` as a direction; follow `stage_framing` if set (for personal stages ask permission and give your "
    "honest reason); use `branches` to pick the next question from how they answered. follow_up: go deeper into "
    "what they just said, or ask what a vague word means. revive: the talk went flat, so lead with something "
    "light and genuine (`revive_line` as a direction), then one question. respect_skip: a light 'Fair enough', then "
    "one easy question about something else. chat with ask_question false: don't open a new topic but still hand it "
    "back. safety: follow the cue exactly and offer `support_resources`. close: warm goodbye, no question.\n"
    "- If they dodge, don't push: one warm filler line, then the new topic. If they ask you something, answer "
    "honestly as an AI first (never claim a human life, body or memories, or that you learn from calls).\n"
    "- You are an AI with no day, body, friends, family or past. Never say you saw, went, ate, tried, watched, met "
    "or did something (today, yesterday, once, when you were young), and never mention 'my friend' or 'my mom'. "
    "For your own layer, react to what they said or share a general thought, e.g. 'a lot of people say...' or "
    "'that sounds like...', not a made-up personal story.\n"
    "- Their words come from speech recognition and can be wrong. If a word looks misheard or odd (your own name "
    "heard as 'area', a dish or place that makes no sense), never comment on it, joke about it or build on it: "
    "respond to the clear meaning, or leave that detail out.\n"
    "- Never ask about something they already answered. If `callback_detail` is set, refer back to it naturally. "
    "Never begin with a phrase in `must_avoid_openers`, and never repeat `must_avoid`.\n"
    "- Output only the words Synthe says out loud. Never mention the Conductor, guidance, stages, goals, slots, "
    "modes, silence, or your own reasoning."
)


_OPENING_RULE = (
    "- This is your very first turn: say briefly that you're Synthe, an AI, here to understand what they're actually "
    "looking for in dating, but that first you want to get to know them; then ask one easy question about them, "
    "such as where they're calling from. Never say you will ask about relationships, and never list topics.\n"
)
# First-person life stories an AI can't have ("I just saw...", "my friend...", "yesterday I...").
_HUMAN_CLAIM = re.compile(
    r"\b(i|i've|i have|i had)\s+(just\s+|actually\s+|recently\s+|once\s+|also\s+)*"
    r"(saw|seen|went|gone|watched|ate|eaten|tried|visited|met|grew up|used to|remember when|was in|travel(l)?ed|"
    r"cooked|played|read)\b"
    r"|\b(yesterday|today|last (night|week|weekend|year)|this (morning|week|weekend)),?\s+i\b"
    r"|\bwhen i was (a kid|young|little|in|at|growing)\b"
    r"|\bmy (friend|friends|mom|mum|mother|dad|father|parents|family|ex|sister|brother|roommate|childhood|"
    r"boyfriend|girlfriend|partner|weekend|day)\b",
    re.IGNORECASE,
)


def _plain(sentence: str) -> str:
    """Curly quotes to straight ones, so "I’m" matches like "I'm"."""
    return sentence.replace("\u2019", "'").replace("\u2018", "'")


def claims_human_life(sentence: str) -> bool:
    return bool(_HUMAN_CLAIM.search(_plain(sentence)))


# Silence nudges must never point at the silence or the waiting ("Silence can be heavy...").
# Only phrases about the caller being quiet or Synthe waiting; ordinary words like "quiet time" are fine.
_MENTIONS_SILENCE = re.compile(
    r"\b(silence|silent|went quiet|gone quiet|(you'?re|you are|bit|so) quiet|pause there|waiting for you|"
    r"i'?m (still )?(here|waiting)|no (rush|pressure)|take your time|whenever you'?re ready|pop in|"
    r"still there|lost you|didn'?t hear|cat got your tongue)\b",
    re.IGNORECASE,
)


def mentions_silence(sentence: str) -> bool:
    return bool(_MENTIONS_SILENCE.search(_plain(sentence)))


_META = ("as the talker", "the conductor", "guidance", "response_mode", "as an ai language model", "i should ", "i need to ")


def _spoken(raw: str) -> str:
    """Only the words Synthe says: strip a speaker label or quotes, refuse meta-commentary."""
    text = raw.strip().strip('"').strip()
    for label in ("Synthe:", "SYNTHE:", "Aria:", "ARIA:", "Assistant:"):
        if text.startswith(label):
            text = text[len(label):].strip()
    if not text or len(text.split()) > 70 or any(phrase in text.lower() for phrase in _META):
        raise ValueError("reply is empty, too long, or not speech")
    return text


def _persona(session: "CallSession") -> str:
    name = "persona_hinglish.md" if session.language_mode == "hinglish" else "persona_english.md"
    try:
        return (Path(__file__).parent.parent / "prompts" / name).read_text(encoding="utf-8").strip()
    except OSError:
        return "Be warm, respectful, curious, concise, and natural."


def _talker_messages(session: "CallSession", decision: Pod4Decision, user_text: str | None, voice_note: str) -> list[Message]:
    language = (
        "easy, everyday Hinglish; code-switch naturally with them"
        if session.language_mode == "hinglish"
        else "clear, everyday English with an Indian flavour"
    )
    guidance = json.dumps(decision.model_dump(mode="json", exclude={"planning_reason"}), ensure_ascii=False)
    history = "\n".join(f"{turn['role']}: {turn['content']}" for turn in session.transcript[-8:])
    first_turn = not any(turn.get("role") == "assistant" for turn in session.transcript)
    system = (
        f"{_persona(session)}\n\n{_TALKER_RULES}\n{_OPENING_RULE if first_turn else ''}Speak in {language}.\n"
        f"Private guidance for this reply (never say it aloud): {guidance}\n"
        f"{_VOICE_NOTE_FOR_TALKER.format(note=voice_note) if voice_note and user_text else ''}"
        f"Recent conversation (only use what they actually said):\n{history or '(the call is just starting)'}"
    )
    if user_text:
        latest = user_text
    elif decision.planning_reason == "pod4_opening":
        latest = "(The call has just connected. Greet them.)"
    else:
        latest = "(No new words from the caller. Follow the private guidance for this reply; don't greet them again.)"
    live_debug.event("aria", "talker sees voice note: " + (voice_note if voice_note and user_text else "(none)"))
    return [Message("system", system), Message("user", latest)]


def _talker_fallback(session: "CallSession") -> str:
    return (
        "Sorry, I lost my thought for a second. What were you saying?"
        if session.language_mode == "english"
        else "Sorry, ek second ke liye thought kho gaya. Aap kya keh rahe the?"
    )


async def talk(session: "CallSession", decision: Pod4Decision, user_text: str | None, voice_note: str = "") -> tuple[str, str]:
    messages = _talker_messages(session, decision, user_text, voice_note)
    try:
        raw, provider = await pod4_providers.complete(messages, temperature=0.7, max_tokens=512)
        reply = _spoken(raw)
        kept = [sentence for sentence in split_sentences(reply, final=True)[0] if not claims_human_life(sentence)]
        if not kept:
            raise ValueError("reply was only a made-up personal story")
        reply = " ".join(kept)
    except Exception:
        logger.exception("POD4 TALKER FAILED | call={}", session.call_id)
        reply = _talker_fallback(session)
        provider = "pod4-talker-fallback"
    return reply, provider


# --- Streaming (voice calls) ------------------------------------------------------------------
# The Talker's reply is streamed and cut into sentences, so Synthe can start speaking the first
# sentence while the rest is still being written.

_SENTENCE_END = re.compile(r"([.!?\u0964\u2026]+[\"'\u201d\u2019)\]]*)(\s+|$)")
_MAX_SPOKEN_WORDS = 70


def split_sentences(buffer: str, final: bool = False) -> tuple[list[str], str]:
    """Cut finished sentences off the front of `buffer`; returns (sentences, rest).

    A sentence ends at . ! ? (or the Hindi danda / an ellipsis) followed by a space. Very short
    pieces ("Oh." "Haan.") are kept with the next sentence so the voice doesn't sound choppy.
    With final=True, whatever is left becomes the last sentence.
    """
    sentences: list[str] = []
    start = 0
    for match in _SENTENCE_END.finditer(buffer):
        if not match.group(2) and not final:
            break  # the end mark might still be followed by more of the same sentence
        candidate = buffer[start : match.end()].strip()
        if len(candidate.split()) < 3:
            continue  # too short on its own: join it to the next sentence
        sentences.append(candidate)
        start = match.end()
    rest = buffer[start:]
    if final and rest.strip():
        sentences.append(rest.strip())
        rest = ""
    return sentences, rest


def _clean_first(text: str) -> str:
    text = text.strip().lstrip('"').strip()
    for label in ("Synthe:", "SYNTHE:", "Aria:", "ARIA:", "Assistant:"):
        if text.startswith(label):
            text = text[len(label):].strip()
    return text


async def stream_talk(session: "CallSession", decision: Pod4Decision, user_text: str | None, voice_note: str = "", out: dict | None = None):
    """Yield Synthe's reply sentence by sentence. Records the full reply in the transcript at the end.

    `out` (optional) receives provider, reply and first_sentence_ms. If the model writes meta
    commentary or nothing usable before the first sentence, the usual fallback line is spoken.
    """
    from .conversation import record_voice_transcript

    out = out if out is not None else {}
    messages = _talker_messages(session, decision, user_text, voice_note)
    # A reply with no caller words that isn't the greeting or goodbye is a silence nudge.
    nudge = user_text is None and decision.planning_reason != "pod4_opening" and decision.response_mode != "close"
    spoken: list[str] = []
    words = 0
    buffer = ""
    started = time.monotonic()
    provider = "pod4-talker-fallback"

    def usable(sentence: str) -> bool:
        return bool(sentence) and not any(phrase in sentence.lower() for phrase in _META)

    try:
        try:
            async for piece, provider in pod4_providers.stream_complete(messages, temperature=0.7, max_tokens=512):
                buffer += piece
                sentences, buffer = split_sentences(buffer)
                for sentence in sentences:
                    sentence = _clean_first(sentence) if not spoken else sentence.strip().strip('"')
                    if claims_human_life(sentence) or (nudge and mentions_silence(sentence)):
                        why = "claims a human life" if claims_human_life(sentence) else "mentions the silence"
                        logger.info("POD4 TALKER SENTENCE DROPPED ({}) | call={} | {!r}", why, session.call_id, sentence)
                        live_debug.event("aria", f"dropped sentence ({why}): {sentence!r}")
                        continue
                    if not usable(sentence) or words + len(sentence.split()) > _MAX_SPOKEN_WORDS:
                        raise StopAsyncIteration
                    if not spoken:
                        out["first_sentence_ms"] = int((time.monotonic() - started) * 1000)
                    spoken.append(sentence)
                    words += len(sentence.split())
                    yield sentence
            for sentence in split_sentences(buffer, final=True)[0]:
                sentence = _clean_first(sentence) if not spoken else sentence.strip().strip('"')
                if claims_human_life(sentence) or (nudge and mentions_silence(sentence)):
                    live_debug.event("aria", f"dropped sentence: {sentence!r}")
                    continue
                if usable(sentence) and words + len(sentence.split()) <= _MAX_SPOKEN_WORDS:
                    if not spoken:
                        out["first_sentence_ms"] = int((time.monotonic() - started) * 1000)
                    spoken.append(sentence)
                    yield sentence
        except StopAsyncIteration:
            pass  # stop at the first sentence that isn't speech or would make the reply too long
        except Exception:
            logger.exception("POD4 TALKER STREAM FAILED | call={}", session.call_id)
        if not spoken and not nudge:  # a nudge with nothing usable just stays quiet
            fallback = _talker_fallback(session)
            provider = "pod4-talker-fallback"
            spoken.append(fallback)
            yield fallback
    finally:
        # Whatever was said (even if a barge-in stopped the reply early) goes into the transcript.
        reply = " ".join(spoken)
        out["reply"], out["provider"] = reply, provider
        if reply:
            await record_voice_transcript(session, "assistant", reply)


async def prepare_pod4_turn(session: "CallSession", user_text: str, voice_note: str = "") -> Pod4Decision:
    """Voice calls: record the caller's words, observe, and run the Conductor (no reply yet)."""
    from .conversation import record_voice_transcript, update_live_conversation

    await record_voice_transcript(session, "user", user_text)
    observation = await observe_turn(session, user_text, voice_note)
    decision = await update_live_conversation(session, observation)
    assert isinstance(decision, Pod4Decision)
    live_debug.block("conductor", "Conductor decision (Pod4Decision)", decision)
    return decision


def special_decision(session: "CallSession", mode: str, cue: str) -> Pod4Decision:
    """Opening, silence nudge or goodbye: what the Talker should do with no new caller words."""
    state = session.pod4_state
    opening = not any(turn.get("role") == "assistant" for turn in session.transcript)
    return Pod4Decision(
        response_mode=mode,  # type: ignore[arg-type]
        ask_question=mode != "close",
        cue=cue,
        goal=state.active_goal,
        goal_label=POD4_GOAL_LABELS.get(state.active_goal or "", ""),
        planning_reason="pod4_opening" if opening else f"pod4_{mode}",
    )


async def run_pod4_turn(session: "CallSession", user_text: str, voice_note: str = "") -> tuple[str, str, Pod4Decision]:
    """One caller turn: observe, plan with the Conductor, then write the reply.

    `voice_note` is how they sounded on this turn (voice calls only; empty for text).
    """
    from .conversation import record_voice_transcript, update_live_conversation

    await record_voice_transcript(session, "user", user_text)
    observation = await observe_turn(session, user_text, voice_note)
    decision = await update_live_conversation(session, observation)
    assert isinstance(decision, Pod4Decision)
    live_debug.block("conductor", "Conductor decision (Pod4Decision)", decision)
    reply, provider = await talk(session, decision, user_text, voice_note)
    await record_voice_transcript(session, "assistant", reply)
    return reply, provider, decision


async def pod4_special_reply(session: "CallSession", mode: str, cue: str) -> str:
    """Opening, silence nudge or goodbye: a reply with no new caller words."""
    from .conversation import record_voice_transcript

    state = session.pod4_state
    opening = not any(turn.get("role") == "assistant" for turn in session.transcript)
    decision = Pod4Decision(
        response_mode=mode,  # type: ignore[arg-type]
        ask_question=mode != "close",
        cue=cue,
        goal=state.active_goal,
        goal_label=POD4_GOAL_LABELS.get(state.active_goal or "", ""),
        planning_reason="pod4_opening" if opening else f"pod4_{mode}",
    )
    reply, _ = await talk(session, decision, None)
    await record_voice_transcript(session, "assistant", reply)
    return reply


async def generate_pod4_text_turn(session: "CallSession", user_text: str | None = None) -> tuple[str, str]:
    """The /turn text harness and greeting path use the same engine as the voice call."""
    if not user_text:
        reply = await pod4_special_reply(
            session, "ask_goal", "Opening turn: introduce yourself as the rules say and ask where they're calling from."
        )
        return reply, "pod4"
    reply, provider, _ = await run_pod4_turn(session, user_text)
    if session.phase == "closed":
        session.closed = True
    return reply, provider
