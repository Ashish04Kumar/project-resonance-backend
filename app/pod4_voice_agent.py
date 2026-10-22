"""Pod 4 voice call on our own pipeline (no Gemini Live).

Phone -> LiveKit -> Silero VAD -> STT -> Pod4TurnProcessor -> TTS -> LiveKit -> Phone

Pod4TurnProcessor runs the Pod 4 turn engine on each finished caller utterance
(observe -> Conductor -> Talker), speaks the reply and sends captions. The Talker's
reply is streamed: each sentence goes to TTS as soon as the LLM has written it, so Synthe
starts talking before the whole reply exists. It also
measures how the caller sounded on each turn (voice_cues.py: pauses, pace, loudness,
intonation, fillers, from VAD timing and raw microphone audio) and passes that to the
LLM as soft hints; this works the same with any STT/TTS provider. It also
handles barge-in, a single silence nudge per caller turn, the safety card and a
spoken goodbye on the blueprint clock. STT and TTS come from pod4_providers.
"""

from __future__ import annotations

import asyncio
import json
import time
from collections import deque

import numpy as np
from contextlib import aclosing
from typing import AsyncIterator, Callable

from loguru import logger
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    Frame,
    InputAudioRawFrame,
    TranscriptionFrame,
    TTSSpeakFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.audio.vad_processor import VADProcessor
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.transports.livekit.transport import LiveKitParams, LiveKitTransport

from .config import settings
from .conversation import CallSession
from .conversation_pod4 import POD4_BLUEPRINT
from .conversation_pod4_text import prepare_pod4_turn, special_decision, stream_talk
from .pod4_providers import make_stt, make_tts
from .stt_guard import BLIP_SECONDS, STREAM_WINDOW_SECONDS, Segment, check_streamed, check_with_segments
from .voice_cues import VoiceCues, VoiceCueTracker
from . import pod4_live_debug as live_debug  # TEMPORARY live terminal view


# Whisper sends one transcript per speech segment, so one spoken answer can arrive in pieces,
# each 1-3.5 s after that bit of speech ended. Each finished segment waits in a queue until its
# text arrives (stt_guard.check_with_segments). Synthe answers only when the caller has stopped
# talking, the last piece is at least this old, and no real segment is still waiting for its
# text (given up after STT_WAIT_MAX_SECONDS). A reply already being prepared is never cancelled:
# words that arrive while it is being answered become the next turn.
UTTERANCE_GAP_SECONDS = 0.6
STT_WAIT_MAX_SECONDS = 4.0
# The caller must also have been quiet this long (by real-time VAD) before Synthe answers.
END_SILENCE_SECONDS = 0.7
# A streaming STT (Sarvam) is told to finish its text at every VAD stop (pipecat flushes it) and
# answers within a fraction of a second. Its pieces are not matched to segments one-to-one (see
# stt_guard.check_streamed). After a stretch of speech ends, Synthe waits at most this long for the
# flushed text (Whisper's STT_WAIT_MAX_SECONDS is for a whole transcription)...
STREAM_FLUSH_WAIT_SECONDS = 1.5
# ...and once it has come (text after the VAD stop is the flushed end of the speech), this much
# instead of UTTERANCE_GAP_SECONDS for any last piece.
STREAM_FINAL_GAP_SECONDS = 0.3
# Whisper hears a little audio from before VAD's start (pre-roll); a segment's loudness is measured
# over this much audio before it too, so a real word cut short by VAD isn't mistaken for noise.
PRE_ROLL_SECONDS = 0.8


class Pod4TurnProcessor(FrameProcessor):
    # Label of the AI's lines in the live terminal view ("aria" shows as SYNTHE; Buddy uses "buddy").
    debug_voice = "aria"

    def __init__(self, session: CallSession, transport: LiveKitTransport) -> None:
        super().__init__()
        self._session = session
        self._transport = transport
        self._lock = asyncio.Lock()
        self._bot_speaking = False
        self._user_speaking = False
        self._pending: list[str] = []
        self._answer_task: asyncio.Task | None = None
        self._last_piece_at = 0.0
        # Finished stretches of caller speech still waiting for their STT text, oldest first. Used to
        # tell real words from text the STT made up, and whether more text is still on its way.
        self._segment_started_at: float | None = None
        self._segment_energy = 0.0
        self._segment_samples = 0
        self._segments: list[Segment] = []
        self._last_speech_end = 0.0
        # True with a streaming STT (see stt_streams); False keeps Whisper's segment matching.
        self.streaming_stt = False
        # Streaming only: when the last stretch of real speech ended and when the STT last sent text.
        self._last_real_speech_end = 0.0
        self._last_text_at = 0.0
        # Recent mic energy per 20 ms frame (time, energy, samples) for the pre-roll loudness.
        self._recent_audio: deque[tuple[float, float, int]] = deque(maxlen=150)
        # The silence nudge is a plain timer (no task until it fires).
        self._nudge_handle: asyncio.TimerHandle | None = None
        self._nudged_since_caller = False
        # How the caller sounds, measured from VAD events and raw mic audio (provider-independent).
        self._cues = VoiceCueTracker()
        self._turn_number = 0  # live_debug
        # Set when the caller starts talking: a streaming reply stops sending further sentences.
        self._caller_cut_in = False

    # --- captions and speech ---------------------------------------------------------

    async def _caption(self, role: str, text: str) -> None:
        await self._transport.send_message(json.dumps({"type": "call_caption", "role": role, "text": text}, ensure_ascii=False))

    async def say(self, text: str) -> None:
        await self._caption("assistant", text)
        await self.push_frame(TTSSpeakFrame(text), FrameDirection.DOWNSTREAM)

    async def speak_stream(self, sentences: AsyncIterator[str]) -> None:
        """Send each streamed sentence to TTS as it arrives; stop if the caller starts talking."""
        self._caller_cut_in = False
        async with aclosing(sentences) as stream:
            async for sentence in stream:
                if self._caller_cut_in:
                    live_debug.event(self.debug_voice, "caller started talking: rest of the reply dropped")
                    break
                live_debug.event(self.debug_voice, f"-> TTS: {sentence!r}")
                await self.push_frame(TTSSpeakFrame(sentence), FrameDirection.DOWNSTREAM)

    async def say_special(self, mode: str, cue: str) -> None:
        """Greeting, silence nudge or goodbye, streamed like a normal reply. Call with the lock held."""
        out: dict = {}
        await self.speak_stream(stream_talk(self._session, special_decision(self._session, mode, cue), None, "", out))
        if out.get("reply"):
            await self._caption("assistant", out["reply"])

    @property
    def bot_speaking(self) -> bool:
        return self._bot_speaking

    # --- silence nudge -----------------------------------------------------------------

    def _cancel_nudge(self) -> None:
        if self._nudge_handle is not None:
            self._nudge_handle.cancel()
        self._nudge_handle = None

    def _start_nudge_timer(self) -> None:
        state = self._session.pod4_state
        if self._nudged_since_caller or state.safety_mode or state.segment in ("wrap_up", "close"):
            return
        self._cancel_nudge()
        # A timer, not a sleeping task: Synthe's audio stops/starts between sentences, and cancelling a
        # task that never ran used to print "coroutine was never awaited" warnings.
        self._nudge_handle = asyncio.get_running_loop().call_later(
            settings.conversation_silence_nudge_seconds, self._fire_nudge
        )

    def _fire_nudge(self) -> None:
        self._nudge_handle = None
        if self._bot_speaking or self._user_speaking:
            return
        self.create_task(self._nudge_after_silence())

    async def _nudge_after_silence(self) -> None:
        state = self._session.pod4_state
        if self._lock.locked() or state.safety_mode or state.segment in ("wrap_up", "close"):
            return
        self._nudged_since_caller = True
        if state.flat_streak >= 1:
            mode, cue = "revive", (
                "The caller has gone quiet after short replies. Take the lead like a friend: say something light and "
                "genuine first, then ease into one easy question. Don't mention the silence."
            )
        elif state.active_goal:
            mode, cue = "ask_goal", (
                "The caller has been quiet for a while. Gently come back to your open question in an easier form, "
                "with a tiny example or two simple options, in fresh words. Don't mention the silence."
            )
        else:
            mode, cue = "chat", (
                "The caller has been quiet for a while. Warmly invite them to share more about what they last said, "
                "with one easy question. Don't mention the silence."
            )
        logger.info("POD4 SILENCE NUDGE | call={} | mode={}", self._session.call_id, mode)
        live_debug.event("conductor", f"silence nudge: {mode}")
        async with self._lock:
            await self.say_special(mode, cue)

    # --- frames ------------------------------------------------------------------------

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)
        now = time.monotonic()
        # Caller speech (VAD events and mic audio) is handled by VadTap before the STT, in real time;
        # those frames still pass through here and are just forwarded.
        if isinstance(frame, BotStartedSpeakingFrame):
            self._bot_speaking = True
            self._cues.bot_started(now)
            live_debug.event(self.debug_voice, "started speaking")
            self._cancel_nudge()
        elif isinstance(frame, BotStoppedSpeakingFrame):
            self._bot_speaking = False
            self._cues.bot_stopped(now)
            live_debug.event(self.debug_voice, "stopped speaking")
            self._start_nudge_timer()
        if isinstance(frame, TranscriptionFrame):
            self._on_transcript(frame.text.strip(), now)
            return
        await self.push_frame(frame, direction)

    # --- caller speech, called by VadTap in real time ------------------------------------------

    def on_mic_audio(self, pcm: bytes, sample_rate: int, now: float) -> None:
        self._cues.audio(pcm, sample_rate)
        samples = np.frombuffer(pcm[: len(pcm) // 2 * 2], dtype=np.int16).astype(np.float32) / 32768.0
        energy = float(np.sum(samples * samples))
        self._recent_audio.append((now, energy, len(samples)))
        if self._user_speaking:
            self._segment_energy += energy
            self._segment_samples += len(samples)

    def _loudness_since(self, start: float) -> float | None:
        """Average mic loudness (dBFS) from `start` until now."""
        energy = samples = 0
        for at, frame_energy, frame_samples in self._recent_audio:
            if at >= start:
                energy += frame_energy
                samples += frame_samples
        return 10 * float(np.log10(max(energy / samples, 1e-12))) if samples else None

    async def on_user_started(self, now: float) -> None:
        self._user_speaking = True
        self._caller_cut_in = True
        self._segment_started_at = now
        self._segment_energy, self._segment_samples = 0.0, 0
        self._cues.user_started(now)
        live_debug.event("you", "started speaking" + (" (over Synthe: barge-in)" if self._bot_speaking else ""))
        self._cancel_nudge()
        if self._bot_speaking:
            # Barge-in: stop Synthe's audio so she listens (earphones avoid her own voice triggering this).
            await self.broadcast_interruption()

    def on_user_stopped(self, now: float) -> None:
        self._user_speaking = False
        self._cues.user_stopped(now)
        if self._segment_started_at is not None:
            loudness = (
                10 * float(np.log10(max(self._segment_energy / self._segment_samples, 1e-12)))
                if self._segment_samples
                else None
            )
            pre_roll = self._loudness_since(self._segment_started_at - PRE_ROLL_SECONDS)
            segment = Segment(now, now - self._segment_started_at, loudness, pre_roll)
            self._segments.append(segment)
            if segment.seconds >= BLIP_SECONDS:
                self._last_real_speech_end = now
            self._segment_started_at = None
        self._last_speech_end = now
        live_debug.event("you", "stopped speaking")
        self._ensure_answer_loop()

    # --- turn taking -------------------------------------------------------------------------

    def _on_transcript(self, text: str, now: float) -> None:
        """Match a piece of STT text to its speech segment, keep or drop it, and keep the loop running."""
        if self.streaming_stt:
            keep, reason, segment = self._check_streamed(text, now)
        else:
            keep, reason, segment, used = check_with_segments(text, self._segments)
            del self._segments[:used]
        if not keep:
            logger.info("POD4 STT PIECE DROPPED | call={} | text={!r} | {}", self._session.call_id, text, reason)
            live_debug.event("stt", f"dropped {text!r}: {reason}")
            self._ensure_answer_loop()
            return
        speech = f"{segment.seconds:.1f}s of speech" if segment else "no timing"
        live_debug.event("stt", f"heard: {text!r} ({speech})")
        self._pending.append(text)
        self._last_piece_at = now
        self._cancel_nudge()
        self._ensure_answer_loop()

    def _check_streamed(self, text: str, now: float) -> tuple[bool, str, Segment | None]:
        """Streaming STT: check a piece against all recent caller speech, including the stretch
        still going on (the STT often finalises mid-sentence, before our VAD has stopped)."""
        self._last_text_at = now
        self._segments = [segment for segment in self._segments if now - segment.ended_at < STREAM_WINDOW_SECONDS]
        recent = list(self._segments)
        if self._segment_started_at is not None:
            loudness = (
                10 * float(np.log10(max(self._segment_energy / self._segment_samples, 1e-12)))
                if self._segment_samples
                else None
            )
            pre_roll = self._loudness_since(self._segment_started_at - PRE_ROLL_SECONDS)
            recent.append(Segment(now, now - self._segment_started_at, loudness, pre_roll))
        return check_streamed(text, recent)

    def _waiting_for_stt(self, now: float) -> bool:
        """More text may still come for speech that has already ended."""
        if self.streaming_stt:
            # The flushed text for the last stretch of real speech hasn't come yet (given up after a while).
            return self._last_text_at < self._last_real_speech_end and now - self._last_real_speech_end < STREAM_FLUSH_WAIT_SECONDS
        return self._stt_still_working(now)

    def _stt_still_working(self, now: float) -> bool:
        """A real stretch of speech is still waiting for its text (segments older than
        STT_WAIT_MAX_SECONDS gave none and are forgotten)."""
        self._segments = [segment for segment in self._segments if now - segment.ended_at < STT_WAIT_MAX_SECONDS]
        return any(segment.seconds >= BLIP_SECONDS for segment in self._segments)

    def _ready_to_answer(self, now: float) -> bool:
        gap = UTTERANCE_GAP_SECONDS
        if self.streaming_stt and self._last_piece_at > self._last_speech_end:
            gap = STREAM_FINAL_GAP_SECONDS  # the flushed end of the speech is already here
        return (
            bool(self._pending)
            and not self._user_speaking
            and now - self._last_piece_at >= gap
            and now - self._last_speech_end >= END_SILENCE_SECONDS
            and not self._waiting_for_stt(now)
        )

    def _ensure_answer_loop(self) -> None:
        if self._answer_task is None or self._answer_task.done():
            self._answer_task = self.create_task(self._answer_loop())

    async def _answer_loop(self) -> None:
        """One loop answers turns in order; it is never cancelled by new pieces of text."""
        answered = False
        while True:
            await asyncio.sleep(0.15)
            now = time.monotonic()
            if not self._pending:
                if self._user_speaking or self._waiting_for_stt(now):
                    continue  # more text may still come
                # They made a sound but no words came through (a short "hmm", a cough). Don't leave
                # both sides waiting: count it as their turn and start the silence nudge again.
                if not answered and not self._bot_speaking and not self._lock.locked():
                    live_debug.event("stt", "speech gave no usable words; silence nudge re-armed")
                    self._cues.discard_turn()  # its timing would make the next turn's cues wrong
                    self._nudged_since_caller = False
                    self._start_nudge_timer()
                return
            if not self._ready_to_answer(now):
                continue
            await self._answer_pending()
            answered = True

    async def _answer_pending(self) -> None:
        text = " ".join(self._pending)
        self._pending.clear()
        # Hand the turn's timings/audio over now (so the next turn starts clean), measure off the event loop.
        snapshot = self._cues.take_turn(text, time.monotonic())
        try:
            cues: VoiceCues | None = await asyncio.to_thread(snapshot.measure)
        except Exception:
            logger.exception("POD4 VOICE CUES FAILED | call={}", self._session.call_id)
            cues = None
        await self._handle_caller(text, cues)

    async def _handle_caller(self, text: str, cues: VoiceCues | None = None) -> None:
        self._cancel_nudge()
        self._nudged_since_caller = False
        await self._caption("user", text)
        started = time.monotonic()
        voice_note = cues.describe() if cues else ""
        self._turn_number += 1
        live_debug.turn_divider(self._turn_number)
        live_debug.event("you", f"full turn: {text!r}")
        if cues:
            live_debug.block("cues", "voice measurements (VoiceCues)", cues.log_fields() | {
                "words": cues.words, "longest_pause_s": round(cues.longest_pause_s, 1), "short_reply": cues.short_reply,
                "pace": cues.pace, "volume": cues.volume, "intonation": cues.intonation,
            })
            live_debug.event("cues", f"note sent to LLM: {voice_note or '(nothing unusual: no note)'}")
            logger.info("POD4 VOICE CUES | call={} | {} | note={}", self._session.call_id, cues.log_fields(), voice_note or "-")
        out: dict = {}
        async with self._lock:
            decision = await prepare_pod4_turn(self._session, text, voice_note)
            planned_ms = int((time.monotonic() - started) * 1000)
            state = self._session.pod4_state
            if decision.response_mode == "safety" and not state.safety_resources_sent:
                # SAFE-2: the app shows support resources alongside the calm spoken reply.
                state.safety_resources_sent = True
                await self._transport.send_message(
                    json.dumps({"type": "safety_resources", "resources": decision.support_resources}, ensure_ascii=False)
                )
            await self.speak_stream(stream_talk(self._session, decision, text, voice_note, out))
        first_ms = planned_ms + out.get("first_sentence_ms", 0)
        logger.info(
            "POD4 TURN | call={} | mode={} | reason={} | provider={} | plan_ms={} | first_sentence_ms={} | total_ms={}",
            self._session.call_id,
            decision.response_mode,
            decision.planning_reason,
            out.get("provider"),
            planned_ms,
            first_ms,
            int((time.monotonic() - started) * 1000),
        )
        live_debug.event(
            "aria",
            f"reply ({out.get('provider')}): first sentence sent {first_ms} ms after your turn "
            f"(observe+Conductor {planned_ms} ms); full: {out.get('reply', '')!r}",
        )
        if out.get("reply"):
            await self._caption("assistant", out["reply"])


def stt_streams(stt: object) -> bool:
    """True for a streaming STT (Sarvam); False for one that transcribes each VAD segment (Whisper)."""
    from pipecat.services.stt_service import SegmentedSTTService

    return not isinstance(stt, SegmentedSTTService)


class VadTap(FrameProcessor):
    """Sits between VAD and the STT and passes caller speech events to the turn processor at once.

    A segment-based STT (Whisper) transcribes inside its frame handling, so VAD frames behind it
    reach later processors up to a second late. Watching them here keeps "is the caller talking
    right now?" accurate for turn taking, barge-in and the voice cues, with any STT provider.
    """

    def __init__(self, turns: Pod4TurnProcessor) -> None:
        super().__init__()
        self._turns = turns

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)
        now = time.monotonic()
        if isinstance(frame, InputAudioRawFrame) and frame.num_channels == 1:
            self._turns.on_mic_audio(frame.audio, frame.sample_rate, now)
        elif isinstance(frame, VADUserStartedSpeakingFrame):
            await self._turns.on_user_started(now)
        elif isinstance(frame, VADUserStoppedSpeakingFrame):
            self._turns.on_user_stopped(now)
        await self.push_frame(frame, direction)


async def run_pod4_voice_agent(
    session: CallSession,
    *,
    token: str,
    on_connected: asyncio.Event | None = None,
    on_error: Callable[[Exception], None] | None = None,
) -> None:
    room_name = session.call_id
    transport = LiveKitTransport(
        url=settings.livekit_url,
        token=token,
        room_name=room_name,
        params=LiveKitParams(audio_in_enabled=True, audio_out_enabled=True),
    )
    stt, stt_name = make_stt()
    tts, tts_name = make_tts()
    logger.info(
        "STARTING POD4 PIPELINE CALL | call={} | stt={} | llm={} (fallback {}) | tts={}",
        session.call_id,
        stt_name,
        settings.pod4_llm,
        settings.pod4_llm_fallback,
        tts_name,
    )
    turns = Pod4TurnProcessor(session, transport)
    turns.streaming_stt = stt_streams(stt)
    live_debug.event("stt", f"{stt_name}: {'streaming (pieces checked against recent speech)' if turns.streaming_stt else 'one text per speech segment'}")
    pipeline = Pipeline(
        [
            transport.input(),
            VADProcessor(vad_analyzer=SileroVADAnalyzer()),
            VadTap(turns),
            stt,
            turns,
            tts,
            transport.output(),
        ]
    )
    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            allow_interruptions=True,
            audio_in_sample_rate=16000,
            audio_out_sample_rate=24000,
            enable_metrics=True,
        ),
        conversation_id=session.call_id,
    )
    runner = PipelineRunner(handle_sigint=False, handle_sigterm=False)

    @transport.event_handler("on_connected")
    async def connected(_: LiveKitTransport) -> None:
        logger.info("LIVEKIT TRANSPORT CONNECTED | call={} | room={}", session.call_id, room_name)
        if on_connected:
            on_connected.set()

    @transport.event_handler("on_first_participant_joined")
    async def greet(_: LiveKitTransport, __: str) -> None:
        logger.info("CALLER JOINED | call={} | pod4 greeting", session.call_id)
        if session.transcript:
            cue = "This is a resumed call: welcome them back briefly and pick up from the last thing they shared."
        else:
            cue = "Opening turn: introduce yourself as the rules say and ask where they're calling from."
        async with turns._lock:
            await turns.say_special("ask_goal", cue)

    @transport.event_handler("on_participant_disconnected")
    async def participant_left(_: LiveKitTransport, participant_id: str) -> None:
        logger.info("CALLER LEFT | call={} | participant={}", session.call_id, participant_id)
        await runner.cancel()

    async def goodbye_on_time() -> None:
        clock = POD4_BLUEPRINT.call
        while session.elapsed_seconds < clock.close_seconds:
            await asyncio.sleep(1)
        session.pod4_state.segment = "close"
        for _ in range(15):  # let Synthe finish her current sentence first
            if not turns.bot_speaking and not turns._lock.locked():
                break
            await asyncio.sleep(1)
        logger.info("CALL CLOCK GOODBYE | call={} | elapsed={}s", session.call_id, session.elapsed_seconds)
        async with turns._lock:
            await turns.say_special(
                "close",
                "The call time is up. Warmly reflect one or two real things they shared, thank them for being so "
                "open, and say goodbye. Do not ask any question.",
            )

    goodbye_task = asyncio.create_task(goodbye_on_time())
    try:
        await runner.run(task)
    except asyncio.CancelledError:
        await runner.cancel()
        raise
    except Exception as exc:
        if on_error:
            on_error(exc)
        raise
    finally:
        goodbye_task.cancel()
        await runner.cleanup()
