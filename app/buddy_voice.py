"""Buddy live voice: talk to the embedded companion like a call, on Pod 4's own voice pipeline.

Phone -> LiveKit -> Silero VAD -> VadTap -> STT -> BuddyTurnProcessor -> TTS -> LiveKit -> Phone

Everything about listening and speaking is Pod 4's, unchanged: the STT/TTS adapters and their
fallbacks (pod4_providers; Buddy's TTS has its own order and voices: BUDDY_TTS, BUDDY_SARVAM_SPEAKER, BUDDY_PIPER_VOICE), the real-time VAD tap, the STT guard against made-up words, the voice
cues, sentence-by-sentence streaming and barge-in. BuddyTurnProcessor extends Pod 4's turn
processor and replaces only the "who writes the reply" part: instead of Pod 4's
observe -> Conductor -> Talker it asks the companion for the user's current mode (buddy.py:
chat (Buddy) or synth (Umbrella), each on its own thread; saying "umbrella mode" / "buddy mode" switches) with `chat_stream`, and speaks each
sentence as it arrives. If the caller talks over Buddy,
the rest of the reply is dropped and the companion keeps only what was actually said.

Data messages to the phone (LiveKit data channel):
  {"type": "call_caption", "role": "user"|"assistant", "text"}
  {"type": "buddy_expression", "expression"}   the reply's mood, for the robot's face
  {"type": "user_partial", "text"}             what the STT has heard of the caller's turn so far
  {"type": "buddy_mode", "mode"}               the mode changed (an "umbrella mode" / "buddy mode" keyword)
"""

from __future__ import annotations

import asyncio
import json
import re
import time
from contextlib import aclosing

from loguru import logger
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import TTSSpeakFrame, TTSUpdateSettingsFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.audio.vad_processor import VADProcessor
from pipecat.processors.frame_processor import FrameDirection
from pipecat.transports.livekit.transport import LiveKitParams, LiveKitTransport

from . import buddy
from .config import settings
from .conversation import CallSession
from .pod4_providers import _make_tts, make_stt
from .pod4_voice_agent import STT_WAIT_MAX_SECONDS, Pod4TurnProcessor, VadTap
from .stt_guard import BLIP_SECONDS
from .voice_cues import VoiceCues
from . import pod4_live_debug as live_debug  # TEMPORARY live terminal view

# What Buddy is asked to do on turns with no new words from the caller.
GREETING_CUE = (
    "They just joined a voice chat with you. Greet them warmly and briefly, by name if you know it, "
    "and ask one easy question about their day."
)
NUDGE_CUE = (
    "They have been quiet for a while after your reply. Warmly invite them to share more about what they last said, "
    "with one easy question, using only their real words. Don't mention the silence or the waiting."
)

# Markdown the companion sometimes writes (bold, headings, list dashes, code ticks): not to be read aloud.
_MARKDOWN = re.compile(r"[*_#`]+|^\s*(?:[-•]|\d+[.)])\s+", re.MULTILINE)


# Emoji and pictographs: TTS either reads their names or stumbles on them.
_EMOJI = re.compile("[\U0001F000-\U0001FAFF\u2600-\u27BF\uFE0F\u200D]+")


def speakable(text: str) -> str:
    """A sentence as it should be spoken: markdown symbols and emoji removed, spaces tidied."""
    return re.sub(r"\s{2,}", " ", _EMOJI.sub("", _MARKDOWN.sub("", text))).strip()


# Pod 4 waits up to STT_WAIT_MAX_SECONDS for the text of every stretch of speech before answering.
# A tiny sound (a breath, a click) after the real words then held Buddy's answer back ~4 s. For Buddy,
# stretches shorter than SHORT_SPEECH_SECONDS are waited for only SHORT_SPEECH_WAIT_SECONDS.
SHORT_SPEECH_SECONDS = 0.5
SHORT_SPEECH_WAIT_SECONDS = 1.5

# The companion guard's "nothing was saved" offers start like this (see buddy.conversation_reply).
_SAVE_OFFER_STARTS = ("I haven't saved that yet", "I haven't changed anything yet")


def buddy_unsaved() -> str:
    from companion.guard import UNSAVED_REPLY

    return UNSAVED_REPLY


def make_buddy_tts() -> tuple[object, str]:
    """Buddy's TTS, in its own order (BUDDY_TTS, then BUDDY_TTS_FALLBACK): Sarvam with Buddy's
    speaker, or laptop Piper with Buddy's Piper voice. Pod 4's TTS (Synthe) is not affected."""
    order = [settings.buddy_tts] + [name.strip() for name in settings.buddy_tts_fallback.split(",") if name.strip()]
    errors: list[str] = []
    for name in dict.fromkeys(order):
        try:
            if name == "sarvam":
                if not settings.sarvam_api_key:
                    raise RuntimeError("SARVAM_API_KEY is not set in backend/.env")
                from pipecat.services.sarvam.tts import SarvamTTSService

                service = SarvamTTSService(
                    api_key=settings.sarvam_api_key,
                    settings=SarvamTTSService.Settings(
                        model=settings.buddy_sarvam_tts_model,
                        voice=settings.buddy_sarvam_speaker,
                        language=settings.buddy_sarvam_language,
                        pace=settings.buddy_sarvam_pace,
                    ),
                )
                label = f"sarvam {settings.buddy_sarvam_tts_model} {settings.buddy_sarvam_speaker} pace {settings.buddy_sarvam_pace}"
            elif name == "piper":
                service, label = _make_tts("piper", settings.buddy_piper_voice), f"piper {settings.buddy_piper_voice}"
            else:
                raise RuntimeError(f"unknown TTS {name!r}")
            if errors:
                logger.warning("BUDDY TTS | using {} after: {}", label, "; ".join(errors))
            return service, label
        except Exception as error:
            errors.append(f"{name}: {error}")
    raise RuntimeError("No Buddy TTS could start: " + "; ".join(errors))


# Sarvam has no emotion control, so the reply's mood changes only its pace a little
# (multiplied with BUDDY_SARVAM_PACE): slower when sad or concerned, a bit faster when excited.
MOOD_PACE = {"sad": 0.9, "concerned": 0.94, "excited": 1.08, "surprised": 1.05, "playful": 1.04}
SARVAM_PACE_RANGE = (0.5, 2.0)  # Bulbul v3


def mood_pace(base: float, mood: str) -> float:
    """Sarvam pace for a reply with this mood (unknown or calm moods keep the base pace)."""
    low, high = SARVAM_PACE_RANGE
    return round(min(high, max(low, base * MOOD_PACE.get(mood, 1.0))), 2)


# A live chat ends on its own after this long, so a forgotten session doesn't keep the mic room open.
MAX_SESSION_SECONDS = 30 * 60


class BuddyTurnProcessor(Pod4TurnProcessor):
    """Pod 4's turn taking, barge-in and silence nudge, with the companion writing the replies."""

    debug_voice = "buddy"

    def __init__(self, room: str, user_id: str, transport: LiveKitTransport) -> None:
        # Pod 4's processor needs a call session for its log lines; Buddy doesn't use Pod 4's call state.
        super().__init__(CallSession(room, "hinglish", conversation_pod="pod4"), transport)
        self._user_id = user_id
        # Set when the TTS is Sarvam: each reply's mood then nudges its pace (see mood_pace).
        self.sarvam_pace_base: float | None = None
        self._last_pace: float | None = None

    async def _pace_for_mood(self, mood: str) -> None:
        if self.sarvam_pace_base is None:
            return
        pace = mood_pace(self.sarvam_pace_base, mood)
        if pace == self._last_pace:
            return
        from pipecat.services.sarvam.tts import SarvamTTSService

        self._last_pace = pace
        live_debug.event(self.debug_voice, f"voice pace {pace} ({mood or 'no mood'})")
        # Goes down the pipeline before this reply's sentences, so the whole reply uses it.
        await self.push_frame(TTSUpdateSettingsFrame(delta=SarvamTTSService.Settings(pace=pace)), FrameDirection.DOWNSTREAM)

    async def _say_plain(self, text: str) -> None:
        """Speak a fixed line (no LLM), e.g. a mode switch acknowledgement."""
        live_debug.event(self.debug_voice, f"-> TTS: {text!r}")
        await self.push_frame(TTSSpeakFrame(text), FrameDirection.DOWNSTREAM)
        await self._caption("assistant", text)

    async def _speak_companion(self, text: str = "", cue: str = "", cues: VoiceCues | None = None) -> str:
        """Speak the companion's reply sentence by sentence; stop if the caller starts talking.
        Returns what was handed to TTS. Call with the lock held."""
        self._caller_cut_in = False
        said: list[str] = []
        mode = buddy.get_mode(self._user_id)
        bot = await buddy.get_companion(mode)
        stream = bot.chat_stream(
            buddy.MODE_USER[mode],  # companion user "buddy" or "umbrella", whatever the account
            text,
            cue=cue,
            voice_note=cues.describe() if cues else "",
            voice_cues=cues.log_fields() if cues else None,
            spoken=True,
            mode_label=f"resonance buddy live ({mode}, {self._user_id})",
            thread=buddy.COMPANION_THREAD,  # chat and synth are separate users: separate history and memory
        )
        # aclosing: breaking out (barge-in) closes the companion's stream at once, so it saves
        # only the sentences that were spoken.
        async with aclosing(stream) as events:
            async for event in events:
                kind = event.get("type")
                if kind == "expression":
                    await self._transport.send_message(json.dumps({"type": "buddy_expression", "expression": event.get("expression", "")}))
                    await self._pace_for_mood(event.get("expression", ""))
                elif kind == "sentence":
                    sentence = speakable(event["text"])
                    if not sentence:
                        continue
                    if mode == "chat" and sentence.startswith(_SAVE_OFFER_STARTS):
                        # Chat mode has no saving tools: say how to get them instead of the save offer.
                        await self._say_plain(buddy.conversation_reply(buddy_unsaved(), text, mode))
                        return ""
                    if self._caller_cut_in:
                        live_debug.event(self.debug_voice, "caller started talking: rest of the reply dropped")
                        break
                    live_debug.event(self.debug_voice, f"-> TTS: {sentence!r}")
                    said.append(sentence)
                    await self.push_frame(TTSSpeakFrame(sentence), FrameDirection.DOWNSTREAM)
        reply = " ".join(said)
        if reply:
            await self._caption("assistant", reply)
        return reply

    # --- what the STT heard, sent to the phone as it arrives ---------------------------------------

    def _on_transcript(self, text: str, now: float) -> None:
        before = len(self._pending)
        super()._on_transcript(text, now)
        if len(self._pending) > before:  # kept (not dropped as noise): show it live under the robot
            message = json.dumps({"type": "user_partial", "text": " ".join(self._pending)}, ensure_ascii=False)
            self.create_task(self._transport.send_message(message))

    def _stt_still_working(self, now: float) -> bool:
        def wait_for(segment) -> float:
            return STT_WAIT_MAX_SECONDS if segment.seconds >= SHORT_SPEECH_SECONDS else SHORT_SPEECH_WAIT_SECONDS

        self._segments = [segment for segment in self._segments if now - segment.ended_at < wait_for(segment)]
        return any(segment.seconds >= BLIP_SECONDS for segment in self._segments)

    async def _apply_mode_keyword(self, text: str) -> tuple[str, bool]:
        """An "umbrella mode" / "buddy mode" keyword: switch, tell the phone, and acknowledge a keyword-only turn.
        Returns (the words left to answer, whether the turn is already done)."""
        turn = buddy.prepare_turn(self._user_id, text)
        if turn.changed:
            live_debug.event("info", f"BUDDY MODE: {turn.mode}")
            await self._transport.send_message(json.dumps({"type": "buddy_mode", "mode": turn.mode}))
        if turn.ack:
            await self._say_plain(turn.ack)
            return "", True
        return turn.text, False

    async def greet(self) -> None:
        async with self._lock:
            await self._speak_companion(cue=GREETING_CUE)

    # --- one gentle nudge per caller turn (Pod 4's timer, without Pod 4's call stages) -----------

    def _start_nudge_timer(self) -> None:
        if self._nudged_since_caller:
            return
        self._cancel_nudge()
        self._nudge_handle = asyncio.get_running_loop().call_later(settings.conversation_silence_nudge_seconds, self._fire_nudge)

    async def _nudge_after_silence(self) -> None:
        if self._lock.locked():
            return
        self._nudged_since_caller = True
        live_debug.event("conductor", "silence nudge")
        async with self._lock:
            await self._speak_companion(cue=NUDGE_CUE)

    # --- the caller's finished turn ---------------------------------------------------------------

    async def _handle_caller(self, text: str, cues: VoiceCues | None = None) -> None:
        self._cancel_nudge()
        self._nudged_since_caller = False
        await self._caption("user", text)
        self._turn_number += 1
        live_debug.turn_divider(self._turn_number)
        live_debug.event("you", f"full turn: {text!r}")
        if cues:
            live_debug.event("cues", f"note sent to LLM: {cues.describe() or '(nothing unusual: no note)'}")
        started = time.monotonic()
        async with self._lock:
            try:
                rest, done = await self._apply_mode_keyword(text)
                reply = "" if done else await self._speak_companion(rest, cues=cues)
            except Exception:
                logger.exception("BUDDY LIVE TURN FAILED | room={}", self._session.call_id)
                reply = ""
        logger.info("BUDDY LIVE TURN | room={} | total_ms={}", self._session.call_id, int((time.monotonic() - started) * 1000))
        live_debug.event(self.debug_voice, f"reply: {reply!r} ({int((time.monotonic() - started) * 1000)} ms)")


async def run_buddy_voice_agent(
    room: str,
    user_id: str,
    *,
    token: str,
    on_connected: asyncio.Event | None = None,
) -> None:
    transport = LiveKitTransport(
        url=settings.livekit_url,
        token=token,
        room_name=room,
        params=LiveKitParams(audio_in_enabled=True, audio_out_enabled=True),
    )
    stt, stt_name = make_stt(settings.buddy_stt, settings.buddy_stt_fallback)  # Buddy's own STT order
    tts, tts_name = make_buddy_tts()  # Buddy's own voice and order, not Synthe's
    logger.info("STARTING BUDDY LIVE VOICE | room={} | user={} | stt={} | tts={}", room, user_id, stt_name, tts_name)
    live_debug.event("info", f"BUDDY VOICE: {tts_name}")
    turns = BuddyTurnProcessor(room, user_id, transport)
    if tts_name.startswith("sarvam"):
        turns.sarvam_pace_base = settings.buddy_sarvam_pace
        turns._last_pace = settings.buddy_sarvam_pace
    pipeline = Pipeline([transport.input(), VADProcessor(vad_analyzer=SileroVADAnalyzer()), VadTap(turns), stt, turns, tts, transport.output()])
    task = PipelineTask(
        pipeline,
        params=PipelineParams(allow_interruptions=True, audio_in_sample_rate=16000, audio_out_sample_rate=24000, enable_metrics=True),
        conversation_id=room,
    )
    runner = PipelineRunner(handle_sigint=False, handle_sigterm=False)

    @transport.event_handler("on_connected")
    async def connected(_: LiveKitTransport) -> None:
        if on_connected:
            on_connected.set()

    @transport.event_handler("on_first_participant_joined")
    async def greet(_: LiveKitTransport, __: str) -> None:
        logger.info("BUDDY LIVE JOINED | room={}", room)
        live_debug.event("info", "BUDDY LIVE: phone joined the room; Buddy greets")
        await turns.greet()

    @transport.event_handler("on_participant_disconnected")
    async def participant_left(_: LiveKitTransport, __: str) -> None:
        logger.info("BUDDY LIVE LEFT | room={}", room)
        live_debug.event("info", "BUDDY LIVE: phone left the room")
        turns._cancel_nudge()  # no silence nudge into an empty room
        await runner.cancel()

    async def time_limit() -> None:
        await asyncio.sleep(MAX_SESSION_SECONDS)
        logger.info("BUDDY LIVE TIME LIMIT | room={}", room)
        await runner.cancel()

    limit = asyncio.create_task(time_limit())
    try:
        await runner.run(task)
    except asyncio.CancelledError:
        await runner.cancel()
        raise
    finally:
        limit.cancel()
        await runner.cleanup()
