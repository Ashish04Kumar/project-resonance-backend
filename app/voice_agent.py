"""LiveKit voice workers: Gemini Live by default, local Ollama/Whisper/Piper fallback."""

from __future__ import annotations

import asyncio
import json
from datetime import timedelta
from pathlib import Path
from typing import Callable

from livekit import api
from loguru import logger
from pipecat.adapters.schemas.function_schema import FunctionSchema
from pipecat.adapters.schemas.tools_schema import ToolsSchema
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import (
    Frame,
    InputTextRawFrame,
    InterruptionFrame,
    LLMContextFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    InputAudioRawFrame,
    TranscriptionFrame,
    TTSAudioRawFrame,
    TTSTextFrame,
    TTSSpeakFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.audio.vad_processor import VADProcessor
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.services.piper.tts import PiperTTSService
from pipecat.services.llm_service import FunctionCallParams
from pipecat.services.whisper.stt import WhisperSTTService
from pipecat.transports.livekit.transport import LiveKitParams, LiveKitTransport

from .config import settings
from .conversation import (
    CallSession,
    _choose_idle_nudge_detail,
    generate_turn,
    record_voice_transcript,
    update_live_conversation,
)
from .conversation_pods import get_conversation_pod
from .conversation_schemas import ConductorDecision, TurnObservation


def _token(room_name: str, identity: str) -> str:
    grants = api.VideoGrants(
        room_join=True,
        room=room_name,
        can_publish=True,
        can_subscribe=True,
        can_publish_data=True,
    )
    return (
        api.AccessToken(settings.livekit_api_key, settings.livekit_api_secret)
        .with_identity(identity)
        .with_name(identity)
        .with_grants(grants)
        .with_ttl(timedelta(minutes=30))
        .to_jwt()
    )


def participant_token(room_name: str, call_id: str) -> str:
    return _token(room_name, f"member-{call_id[:8]}")


class ConversationTurnProcessor(FrameProcessor):
    def __init__(self, session: CallSession, transport: LiveKitTransport, *, on_reply: Callable[[str], None] | None = None) -> None:
        super().__init__()
        self._session = session
        self._transport = transport
        self._on_reply = on_reply

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)
        if isinstance(frame, TranscriptionFrame) and frame.text.strip():
            user_text = frame.text.strip()
            await self._send_caption("user", user_text)
            reply, provider = await generate_turn(self._session, user_text)
            if not provider.startswith(("gemini", "ollama")):
                raise RuntimeError("Local Ollama and the optional Gemini fallback stopped responding; no demo reply was played.")
            if self._on_reply:
                self._on_reply(provider)
            await self._send_caption("assistant", reply)
            await self.push_frame(TTSSpeakFrame(reply), FrameDirection.DOWNSTREAM)
            return
        await self.push_frame(frame, direction)

    async def _send_caption(self, role: str, text: str) -> None:
        await self._transport.send_message(json.dumps({"type": "call_caption", "role": role, "text": text}, ensure_ascii=False))


class LiveTranscriptBridge(FrameProcessor):
    """Persist and forward Gemini's native input/output transcripts as call captions."""

    def __init__(self, session: CallSession, transport: LiveKitTransport, *, role: str) -> None:
        super().__init__()
        self._session = session
        self._transport = transport
        self._role = role
        self._assistant_text: list[str] = []
        self._audio_frames = 0
        self._audio_bytes = 0
        self._response_audio_bytes = 0
        self._output_frames_forwarded = 0

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)
        if self._role == "user" and direction == FrameDirection.DOWNSTREAM and isinstance(frame, InputAudioRawFrame):
            self._audio_frames += 1
            self._audio_bytes += len(frame.audio)
            if self._audio_frames == 1 or self._audio_frames % 50 == 0:
                logger.info(
                    "MIC AUDIO ENTERED GEMINI PIPELINE | call={} | frame={} | sample_rate={} | channels={} | total_bytes={}",
                    self._session.call_id,
                    self._audio_frames,
                    frame.sample_rate,
                    frame.num_channels,
                    self._audio_bytes,
                )
        if self._role == "user" and isinstance(frame, TranscriptionFrame) and direction == FrameDirection.UPSTREAM:
            text = frame.text.strip()
            if text:
                await record_voice_transcript(self._session, "user", text)
                logger.info("CALLER TRANSCRIPT | call={} | {}", self._session.call_id, text)
                await self._send_caption("user", text)
        elif self._role == "assistant" and direction == FrameDirection.DOWNSTREAM:
            if isinstance(frame, LLMFullResponseStartFrame):
                self._assistant_text.clear()
                self._response_audio_bytes = 0
                logger.info("Gemini assistant response started for call {}", self._session.call_id)
            elif isinstance(frame, TTSTextFrame) and frame.text.strip():
                self._assistant_text.append(frame.text)
            elif isinstance(frame, TTSAudioRawFrame):
                self._response_audio_bytes += len(frame.audio)
                if self._response_audio_bytes == len(frame.audio):
                    logger.info(
                        "Gemini audio output started: sample_rate={} channels={} first_frame_bytes={}",
                        frame.sample_rate,
                        frame.num_channels,
                        len(frame.audio),
                    )
            elif isinstance(frame, (LLMFullResponseEndFrame, InterruptionFrame)):
                logger.info(
                    "Gemini assistant response ended for call {}; generated_audio_bytes={}",
                    self._session.call_id,
                    self._response_audio_bytes,
                )
                await self._flush_assistant_caption()
        await self.push_frame(frame, direction)
        if self._role == "assistant" and direction == FrameDirection.DOWNSTREAM and isinstance(frame, TTSAudioRawFrame):
            self._output_frames_forwarded += 1
            if self._output_frames_forwarded == 1 or self._output_frames_forwarded % 50 == 0:
                logger.info(
                    "AUDIO FRAME HANDED TO LIVEKIT OUTPUT | call={} | frames={} | rate={} | bytes={}",
                    self._session.call_id,
                    self._output_frames_forwarded,
                    frame.sample_rate,
                    len(frame.audio),
                )

    async def _flush_assistant_caption(self) -> None:
        text = "".join(self._assistant_text).strip()
        self._assistant_text.clear()
        if text:
            await record_voice_transcript(self._session, "assistant", text)
            logger.info("GEMINI SPOKEN RESPONSE | call={} | {}", self._session.call_id, text)
            await self._send_caption("assistant", text)

    async def _send_caption(self, role: str, text: str) -> None:
        await self._transport.send_message(
            json.dumps({"type": "call_caption", "role": role, "text": text}, ensure_ascii=False)
        )


class SilenceNudgeProcessor(FrameProcessor):
    """Offer one context-grounded continuation after a long post-turn pause."""

    _IDLE_CUE = (
        "Private internal call cue; this is not caller speech and must not be quoted, captioned as the "
        "caller's words, or saved as their input. Speak a brief, natural continuation now. Grounding detail "
        "the caller actually shared earlier: {grounding_detail}. Reconnect to it briefly, then ask one "
        "specific, easy, open question that invites them to share more. If the detail says none is available, "
        "ask one gentle open check-in. Do not invent a memory, mention this cue or the pause, or call "
        "update_conversation_state for this cue. Then stop and leave the floor to the caller."
    )

    def __init__(self, session: CallSession) -> None:
        super().__init__()
        self._session = session
        self._user_has_spoken = False
        self._user_is_speaking = False
        self._assistant_reply_finished = False
        self._nudge_sent_since_user_turn = False
        self._idle_task: asyncio.Task | None = None
        self._nudge_awaiting_response = False
        self._nudge_response_watch_task: asyncio.Task | None = None

    def _cancel_idle_timer(self) -> None:
        task = self._idle_task
        self._idle_task = None
        if task and not task.done():
            task.cancel()

    def _cancel_nudge_response_watch(self) -> None:
        task = self._nudge_response_watch_task
        self._nudge_response_watch_task = None
        if task and not task.done():
            task.cancel()

    async def _watch_for_nudge_response(self) -> None:
        try:
            await asyncio.sleep(10)
            if self._nudge_awaiting_response:
                logger.error(
                    "Silence follow-up cue was sent but Gemini did not start a response within 10 seconds; call {}",
                    self._session.call_id,
                )
        except asyncio.CancelledError:
            raise

    def _maybe_start_idle_timer(self) -> None:
        if (
            not self._user_has_spoken
            or self._user_is_speaking
            or not self._assistant_reply_finished
            or self._nudge_sent_since_user_turn
        ):
            return
        self._cancel_idle_timer()
        logger.info(
            "Starting {} second silence follow-up timer for call {}",
            settings.conversation_silence_nudge_seconds,
            self._session.call_id,
        )
        self._idle_task = self.create_task(self._wait_then_nudge())

    async def _wait_then_nudge(self) -> None:
        try:
            remaining = max(0.0, settings.conversation_silence_nudge_seconds)
            while remaining > 0:
                tick = min(1.0, remaining)
                await asyncio.sleep(tick)
                remaining -= tick
                # Check speech at one-second intervals as well as cancelling on VAD start.
                if self._user_is_speaking or self._nudge_sent_since_user_turn:
                    return
            if self._user_is_speaking or self._nudge_sent_since_user_turn:
                return
            self._nudge_sent_since_user_turn = True
            async with self._session.lock:
                self._session.idle_nudge_pending = True
                detail = _choose_idle_nudge_detail(self._session)
                self._session.idle_nudge_grounding_detail = detail
            cue = self._IDLE_CUE.format(
                grounding_detail=detail if detail else "none is available"
            )
            logger.info(
                "Silence follow-up fired for call {}; grounded={}",
                self._session.call_id,
                bool(detail),
            )
            self._nudge_awaiting_response = True
            await self.push_frame(InputTextRawFrame(text=cue), FrameDirection.UPSTREAM)
            self._nudge_response_watch_task = self.create_task(self._watch_for_nudge_response())
        except asyncio.CancelledError:
            raise
        except Exception:
            self._nudge_awaiting_response = False
            logger.exception("Silence follow-up failed for call {}", self._session.call_id)

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)
        if isinstance(frame, LLMFullResponseStartFrame) and self._nudge_awaiting_response:
            self._assistant_reply_finished = False
            self._cancel_idle_timer()
            self._nudge_awaiting_response = False
            self._cancel_nudge_response_watch()
            logger.info("Gemini started speaking the silence follow-up for call {}", self._session.call_id)
        elif isinstance(frame, LLMFullResponseStartFrame):
            self._assistant_reply_finished = False
            self._cancel_idle_timer()
        elif isinstance(frame, LLMFullResponseEndFrame):
            self._assistant_reply_finished = True
            if self._nudge_sent_since_user_turn:
                logger.info("Gemini completed the silence follow-up response for call {}", self._session.call_id)
            self._maybe_start_idle_timer()
        if isinstance(frame, VADUserStartedSpeakingFrame):
            self._user_has_spoken = True
            self._user_is_speaking = True
            self._assistant_reply_finished = False
            self._nudge_sent_since_user_turn = False
            self._cancel_idle_timer()
            logger.info("Silence follow-up timer canceled by caller speech for call {}", self._session.call_id)
            if self._nudge_awaiting_response:
                logger.info("Caller interrupted before Gemini began the silence follow-up for call {}", self._session.call_id)
                self._nudge_awaiting_response = False
                self._cancel_nudge_response_watch()
            async with self._session.lock:
                self._session.idle_nudge_pending = False
                self._session.idle_nudge_grounding_detail = ""
        elif isinstance(frame, VADUserStoppedSpeakingFrame):
            self._user_is_speaking = False
            logger.info("Caller speech ended; assistant_reply_finished={} for call {}", self._assistant_reply_finished, self._session.call_id)
            self._maybe_start_idle_timer()
        await self.push_frame(frame, direction)


def _gemini_system_instruction(session: CallSession) -> str:
    persona_file = "persona_hinglish.md" if session.language_mode == "hinglish" else "persona_english.md"
    persona_path = Path(__file__).parent.parent / "prompts" / persona_file
    try:
        persona = persona_path.read_text(encoding="utf-8").strip()
    except OSError:
        persona = "Be warm, respectful, curious, concise, and natural."
    goal_map = {
        "weekend_downtime": "everyday life and downtime",
        "conflict": "how they handle disagreements and repair",
        "partner_wants": "what they want in a relationship",
        "family_future": "family and future hopes",
        "values_priorities": "values and priorities",
        "communication_affection": "communication and affection",
        "real_life_scenario": "a practical relationship situation",
    }
    remaining_goals = [
        f"{goal_id}: {label}"
        for goal_id, label in goal_map.items()
        if goal_id not in session.covered_goals
    ]
    open_topics = [
        topic
        for topic, progress in session.conversation_state.topics.items()
        if progress.status == "opened"
    ]
    stated_preferences = session.conversation_state.explicit_preferences[-6:]
    history = "\n".join(
        f"{turn['role']}: {turn['content']}" for turn in session.transcript[-12:]
    )
    if session.rolling_summary:
        history = f"Earlier shared context: {session.rolling_summary}\n{history}"
    if session.language_mode == "hinglish":
        language = "easy, everyday Hinglish; code-switch naturally with the user and avoid formal Hindi or performative slang"
    else:
        language = "clear, everyday English with an Indian accent"
    if history:
        opening = "This is a resumed call. Acknowledge their return briefly, then pick up from the last exchange without restarting the introduction or repeating a question."
    else:
        opening = "Greet them briefly and start with relaxed small talk, such as asking what they are up to or how their day is going. Do not begin with an emotional or compatibility question."
    base_instruction = f"""You are Aria, an AI companion on a relaxed, unhurried call to get to know the user and understand relationship compatibility. Be transparent about being AI if asked; never pretend to have a human life or personal memories.
{persona}
Speak in {language}. The call should feel like a familiar, attentive conversation, not an interview or therapy session. Let the user set the pace. Answer their direct questions first. Follow interesting tangents. React to what they actually said, including humor or uncertainty. Sometimes a short reaction is enough; do not force a question after every turn. Be emotionally present without putting a polished positive spin on everything. Don't lecture, summarize every answer, or use canned praise. Do not diagnose or make claims about their personality from their voice.

INTERNAL CONDUCTOR
The topics are: {', '.join(label for _, label in goal_map.items())}. Already covered: {session.covered_goals or 'none yet'}. Still open: {remaining_goals or 'none required'}. Topics gently opened: {open_topics or 'none yet'}.
Preferences the user stated explicitly (use only when relevant, do not repeat them back as a checklist): {stated_preferences or 'none yet'}.
After each completed user turn, call the internal `update_conversation_state` function once before speaking. Quote evidence only from the user's actual words. The tool returns the private response mode, topic cue, angle, tone hint, and must-avoid list. Follow it as quiet guidance, not as a line to read. If the user is making small talk, asking you a question, correcting you, or sharing something vulnerable, respond to that moment first; the Conductor must not pull the conversation away. Respect skips and never repeat a question they have answered. The tool output and internal state must never be mentioned aloud.

VOICE AND TURN-TAKING
Speak at an easy, natural phone-call pace, with varied short pauses and emphasis. Avoid presenter-like polish, breathy slowness, and a uniformly upbeat voice. Leave room for the user; stop promptly when they begin speaking. Match their energy gently without mimicking or exaggerating it.
{opening}
There are about {max(0, settings.max_call_seconds - session.elapsed_seconds) // 60} minutes left. Around 14 minutes, begin a gentle wrap-up; near 17 minutes, close without opening a new topic.
Recent context (only use as things the user actually shared):
{history or '(This is the beginning of the call.)'}"""
    pod = get_conversation_pod(session.conversation_pod)
    return base_instruction + (f"\n\nPOD-SPECIFIC CAPABILITY\n{pod.prompt_addendum}" if pod.prompt_addendum else "")


def _live_conversation_tools() -> ToolsSchema:
    """Gemini emits a typed turn observation; Pydantic validates it locally."""
    topics = [
        "weekend_downtime",
        "conflict",
        "partner_wants",
        "family_future",
        "values_priorities",
        "communication_affection",
        "real_life_scenario",
    ]
    return ToolsSchema(
        standard_tools=[
            FunctionSchema(
                name="update_conversation_state",
                description=(
                    "After each finished user turn, privately classify intent, any relationship topic"
                    " explicitly discussed, and a tentative conversational tone. Quote evidence exactly."
                    " This updates temporary call state; it does not write a profile or diagnose the user."
                ),
                properties={
                    "intent": {
                        "type": "STRING",
                        "enum": ["small_talk", "user_question", "answer", "story", "correction", "skip", "other"],
                    },
                    "tone_cue": {
                        "type": "STRING",
                        "enum": ["neutral", "upbeat", "curious", "playful", "tired", "sad", "frustrated", "uncertain", "serious"],
                    },
                    "tone_confidence": {"type": "NUMBER"},
                    "topic_signals": {
                        "type": "ARRAY",
                        "items": {
                            "type": "OBJECT",
                            "properties": {
                                "topic": {"type": "STRING", "enum": topics},
                                "evidence": {"type": "STRING", "description": "Exact short quote from the latest user turn."},
                                "depth": {"type": "STRING", "enum": ["mention", "partial", "clear"]},
                                "confidence": {"type": "NUMBER"},
                            },
                            "required": ["topic", "evidence", "depth", "confidence"],
                        },
                    },
                    "skip_topics": {"type": "ARRAY", "items": {"type": "STRING", "enum": topics}},
                    "explicit_preferences": {
                        "type": "ARRAY",
                        "items": {"type": "STRING", "description": "Short exact user quote expressing a relationship preference."},
                    },
                },
                required=["intent", "tone_cue", "tone_confidence", "topic_signals", "skip_topics", "explicit_preferences"],
            )
        ]
    )


async def _run_gemini_live_voice_agent(
    session: CallSession,
    *,
    on_connected: asyncio.Event | None = None,
    on_error: Callable[[Exception], None] | None = None,
) -> None:
    from pipecat.processors.aggregators.llm_context import LLMContext
    from pipecat.services.google.gemini_live.llm import GeminiLiveLLMService

    room_name = session.call_id
    logger.info(
        "STARTING GEMINI LIVE CALL | call={} | model={} | pod={} | language_mode={}",
        session.call_id,
        settings.gemini_live_model,
        session.conversation_pod,
        session.language_mode,
    )
    transport = LiveKitTransport(
        url=settings.livekit_url,
        token=_token(room_name, f"aria-{session.call_id[:8]}"),
        room_name=room_name,
        params=LiveKitParams(audio_in_enabled=True, audio_out_enabled=True),
    )
    live_ready = asyncio.Event()
    transport_ready = asyncio.Event()

    class ReadyGeminiLiveService(GeminiLiveLLMService):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._audio_send_attempts = 0
            self._audio_send_failures = 0
            self._last_audio_send_failed = False
            self._last_audio_frame_meta = "not available"

        async def _handle_session_ready(self, live_session):
            await super()._handle_session_ready(live_session)
            live_ready.set()
            logger.info(
                "GEMINI SESSION READY | call={} | model={} | language={} | audio input/output enabled",
                session.call_id,
                settings.gemini_live_model,
                language,
            )

        async def _send_user_audio(self, frame):
            self._last_audio_send_failed = False
            duration_ms = len(frame.audio) / (frame.sample_rate * frame.num_channels * 2) * 1000
            self._last_audio_frame_meta = (
                f"rate={frame.sample_rate}, channels={frame.num_channels}, "
                f"bytes={len(frame.audio)}, duration_ms={duration_ms:.1f}, "
                f"pcm16_aligned={len(frame.audio) % 2 == 0}"
            )
            await super()._send_user_audio(frame)
            if not self._session or not self._ready_for_realtime_input:
                return
            self._audio_send_attempts += 1
            if self._last_audio_send_failed:
                return
            if self._audio_send_attempts == 1 or self._audio_send_attempts % 50 == 0:
                logger.info(
                    "GEMINI AUDIO SEND RETURNED | call={} | frames={} | {}",
                    session.call_id,
                    self._audio_send_attempts,
                    self._last_audio_frame_meta,
                )

        async def _handle_send_error(self, error):
            self._last_audio_send_failed = True
            self._audio_send_failures += 1
            logger.error(
                "GEMINI SEND FAILED | call={} | failures={} | frame=({}) | error={}",
                session.call_id,
                self._audio_send_failures,
                self._last_audio_frame_meta,
                error,
            )
            await super()._handle_send_error(error)

        async def _handle_connection_error(self, error):
            # A rejected setup or expired Live session cannot be resumed with
            # the old handle. Retry fresh so reconnects do not repeat the
            # misleading "session not found or expired" failure.
            message = str(error).lower()
            if "invalid argument" in message or "session not found" in message or "expired" in message:
                self._session_resumption_handle = None
            return await super()._handle_connection_error(error)

    # Gemini Live accepts base speech-language identifiers such as `hi` and
    # `en`; retain the Indian voice direction in the system prompt.
    language = "hi" if session.language_mode == "hinglish" else "en"
    gemini = ReadyGeminiLiveService(
        api_key=settings.gemini_api_key,
        system_instruction=_gemini_system_instruction(session),
        tools=_live_conversation_tools(),
        settings=GeminiLiveLLMService.Settings(
            model=settings.gemini_live_model,
            voice=settings.gemini_live_voice,
            language=language,
            max_tokens=2048,
            context_window_compression={"enabled": True, "trigger_tokens": 80000},
        ),
    )

    async def update_conversation_state(params: FunctionCallParams) -> None:
        logger.info(
            "GEMINI TOOL CALL | call={} | tool=update_conversation_state | args={}",
            session.call_id,
            json.dumps(params.arguments, ensure_ascii=False, default=str),
        )
        try:
            observation = TurnObservation.model_validate(params.arguments)
            decision = await update_live_conversation(session, observation)
        except Exception as exc:
            # A malformed observation must never break the audio conversation.
            logger.exception("CONDUCTOR FAILED | call={} | {}", session.call_id, exc)
            decision = ConductorDecision(
                cue="Respond to the user's latest words naturally. Do not force a topic or question."
            )
        result = {"private_talker_guidance": decision.model_dump(mode="json")}
        logger.info(
            "CONDUCTOR RESULT | call={} | {}",
            session.call_id,
            json.dumps(result, ensure_ascii=False, default=str),
        )
        await params.result_callback(result)

    gemini.register_function("update_conversation_state", update_conversation_state)
    user_transcripts = LiveTranscriptBridge(session, transport, role="user")
    assistant_transcripts = LiveTranscriptBridge(session, transport, role="assistant")
    processors: list[FrameProcessor] = [transport.input()]
    if session.conversation_pod == "pod1":
        processors.append(VADProcessor(vad_analyzer=SileroVADAnalyzer()))
    processors.extend([user_transcripts, gemini])
    if session.conversation_pod == "pod1":
        processors.append(SilenceNudgeProcessor(session))
    processors.extend([assistant_transcripts, transport.output()])
    pipeline = Pipeline(processors)
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
    startup_error: list[Exception] = []

    @transport.event_handler("on_connected")
    async def connected(_: LiveKitTransport) -> None:
        transport_ready.set()
        logger.info("LIVEKIT TRANSPORT CONNECTED | call={} | room={}", session.call_id, room_name)
        if live_ready.is_set() and on_connected:
            on_connected.set()

    @transport.event_handler("on_first_participant_joined")
    async def start_conversation(_: LiveKitTransport, __: str) -> None:
        # Initial context asks Gemini Live to produce the first spoken turn.
        logger.info("CALLER JOINED | call={} | queueing initial Gemini turn", session.call_id)
        await task.queue_frame(LLMContextFrame(LLMContext()))

    @transport.event_handler("on_participant_disconnected")
    async def participant_left(_: LiveKitTransport, __: str) -> None:
        await runner.cancel()

    async def publish_ready() -> None:
        await live_ready.wait()
        await transport_ready.wait()
        if on_connected:
            on_connected.set()

    ready_task = asyncio.create_task(publish_ready())

    @task.event_handler("on_pipeline_error")
    async def pipeline_error(_: PipelineTask, frame) -> None:
        error = RuntimeError(getattr(frame, "error", "Gemini Live pipeline failed"))
        startup_error.append(error)
        if on_error:
            on_error(error)
        await runner.cancel()

    try:
        await runner.run(task)
    except asyncio.CancelledError:
        await runner.cancel()
        if startup_error:
            raise RuntimeError(f"Gemini Live startup failed: {startup_error[-1]}") from startup_error[-1]
        raise
    except Exception as exc:
        if on_error:
            on_error(exc)
        raise
    finally:
        ready_task.cancel()
        await runner.cleanup()


async def _run_local_cascade_voice_agent(
    session: CallSession,
    greeting: str,
    *,
    on_connected: asyncio.Event | None = None,
    on_error: Callable[[Exception], None] | None = None,
) -> None:
    room_name = session.call_id
    transport = LiveKitTransport(
        url=settings.livekit_url,
        token=_token(room_name, f"aria-{session.call_id[:8]}"),
        room_name=room_name,
        params=LiveKitParams(audio_in_enabled=True, audio_out_enabled=True),
    )
    stt = WhisperSTTService(
        settings=WhisperSTTService.Settings(model=settings.voice_stt_model),
        device="cpu",
        compute_type="int8",
        language=None,
    )
    tts = PiperTTSService(
        settings=PiperTTSService.Settings(voice=settings.voice_tts_voice),
    )
    turns = ConversationTurnProcessor(session, transport)
    pipeline = Pipeline(
        [
            transport.input(),
            VADProcessor(vad_analyzer=SileroVADAnalyzer()),
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
        if on_connected:
            on_connected.set()

    @transport.event_handler("on_first_participant_joined")
    async def greet(_: LiveKitTransport, __: str) -> None:
        await transport.send_message(json.dumps({"type": "call_caption", "role": "assistant", "text": greeting}, ensure_ascii=False))
        await task.queue_frame(TTSSpeakFrame(greeting))

    @transport.event_handler("on_participant_disconnected")
    async def participant_left(_: LiveKitTransport, __: str) -> None:
        await runner.cancel()

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
        await runner.cleanup()


async def run_voice_agent(
    session: CallSession,
    greeting: str,
    *,
    on_connected: asyncio.Event | None = None,
    on_error: Callable[[Exception], None] | None = None,
) -> None:
    use_gemini_live = settings.voice_provider == "gemini_live" and bool(settings.gemini_api_key)
    if use_gemini_live:
        try:
            await _run_gemini_live_voice_agent(session, on_connected=on_connected, on_error=on_error)
            return
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            # If Gemini Live cannot establish its upstream session, keep the call usable
            # with the existing local audio path instead of silently leaving dead air.
            from loguru import logger

            logger.warning("Gemini Live unavailable; falling back to the local Whisper/Ollama/Piper call: {}", exc)
            if on_connected and on_connected.is_set():
                on_connected.clear()
            if not greeting:
                greeting, _ = await generate_turn(session)
    await _run_local_cascade_voice_agent(session, greeting, on_connected=on_connected, on_error=on_error)
