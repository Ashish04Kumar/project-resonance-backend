"""LiveKit voice workers: Gemini Live by default, local Ollama/Whisper/Piper fallback."""

from __future__ import annotations

from array import array
import asyncio
import math
import time
import json
from datetime import timedelta
from pathlib import Path
from typing import Awaitable, Callable

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
    generate_turn,
    record_voice_transcript,
    update_live_conversation,
)
from .conversation_pods import get_conversation_pod
from .conversation_pod5 import (
    POD5_BLUEPRINT,
    POD5_GOAL_LABELS,
    POD5_SLOT_KEYS,
    Pod5Decision,
    Pod5TurnObservation,
)

# Pods whose Conductor is driven by a blueprint file (goals, slots, clock).
_BLUEPRINT_PODS = frozenset({"pod5"})


def _blueprint_state(session: CallSession):
    return session.pod5_state


def _time_left_line(session: CallSession) -> str:
    return (
        f"There are about {max(0, settings.max_call_seconds - session.elapsed_seconds) // 60} minutes left. Around "
        "14 minutes, begin a gentle wrap-up; near 17 minutes, close without opening a new topic."
    )


def _blueprint_labels(session: CallSession) -> dict[str, str]:
    return POD5_GOAL_LABELS


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


def _pcm16_rms(audio: bytes) -> float:
    samples = array("h", audio[: len(audio) - len(audio) % 2])
    if not samples:
        return 0.0
    return math.sqrt(sum(sample * sample for sample in samples) / len(samples))


class LiveTranscriptBridge(FrameProcessor):
    """Persist and forward Gemini's native input/output transcripts as call captions."""

    # 16-bit PCM RMS above this counts as the caller making sound (a starting guess to tune).
    _VOICE_RMS_THRESHOLD = 500

    def __init__(
        self,
        session: CallSession,
        transport: LiveKitTransport,
        *,
        role: str,
        on_caller_sound: Callable[[], Awaitable[None]] | None = None,
    ) -> None:
        super().__init__()
        self._session = session
        self._transport = transport
        self._role = role
        # Pod 5: Gemini sends no "caller is speaking" signal until the caller stops,
        # so mic loudness keeps the silence timer from firing over a long answer.
        self._on_caller_sound = on_caller_sound
        self._last_sound_report = 0.0
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
            if self._on_caller_sound and _pcm16_rms(frame.audio) > self._VOICE_RMS_THRESHOLD:
                now = time.monotonic()
                if now - self._last_sound_report > 0.5:
                    self._last_sound_report = now
                    await self._on_caller_sound()
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

    _POD5_CUE = (
        "Private internal call cue; this is not caller speech and must not be quoted, captioned or saved. "
        "The caller has been quiet for a while. {situation} Speak one or two short, warm sentences, about 25 "
        "words, no lists. Never repeat a sentence or question you already said in this call. Do not mention the "
        "silence, this cue or any internal field, do not call "
        "update_conversation_state for this cue, and never claim a human life or body. Then stop and leave "
        "the floor to the caller."
    )

    def _pod5_cue(self) -> str | None:
        """Pod 5 silence cue from the Conductor's state, or None when a nudge would be wrong."""
        state = _blueprint_state(self._session)
        # Never nudge during the safety path, wrap-up or close; the clock owns those.
        if state.safety_mode or state.segment in {"wrap_up", "close"}:
            return None
        if state.active_goal:
            label = _blueprint_labels(self._session)[state.active_goal]
            situation = (
                f"The pending question is about {label}. Gently come back to it in an easier form: give a tiny "
                "example or two simple options to pick from, with fresh words (never repeat your earlier question)."
            )
        else:
            quote = " ".join(self._session.latest_user_text.split())[:140]
            situation = (
                f"Warmly check in and invite them to share more about what they last said: \"{quote}\"."
                if quote
                else "Warmly check in with one easy, open question."
            )
        return self._POD5_CUE.format(situation=situation)

    def __init__(
        self,
        session: CallSession,
        send_cue: Callable[[str], Awaitable[None]] | None = None,
    ) -> None:
        super().__init__()
        self._session = session
        # When set, the cue goes to Gemini as a completed turn so it answers at once
        # instead of waiting to infer the end of a turn from the silent mic stream.
        self._send_cue = send_cue
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

    async def note_user_activity(self) -> None:
        """Pod 5: Gemini heard the caller, so restart the silence clock from now."""
        self._user_has_spoken = True
        self._nudge_sent_since_user_turn = False
        self._cancel_idle_timer()
        if self._nudge_awaiting_response:
            self._nudge_awaiting_response = False
            self._cancel_nudge_response_watch()
        self._maybe_start_idle_timer()

    async def note_caller_sound(self) -> None:
        """Pod 5: sound on the mic pushes the silence clock back, without re-arming a used nudge."""
        if self._idle_task and not self._idle_task.done():
            self._cancel_idle_timer()
            self._maybe_start_idle_timer()

    def _maybe_start_idle_timer(self) -> None:
        if (
            # Pod 5 also nudges after the greeting if the caller never speaks.
            self._user_is_speaking
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
            cue = self._pod5_cue()
            if cue is None:
                return
            self._nudge_sent_since_user_turn = True
            logger.info(
                "SILENCE NUDGE FIRED | call={} | pod={} | active_goal={} | segment={}",
                self._session.call_id,
                self._session.conversation_pod,
                _blueprint_state(self._session).active_goal,
                _blueprint_state(self._session).segment,
            )
            self._nudge_awaiting_response = True
            if self._send_cue:
                await self._send_cue(cue)
            else:
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
    blueprint_state = _blueprint_state(session)
    goal_map = _blueprint_labels(session)
    covered = [goal for goal, progress in blueprint_state.goals.items() if progress.status == "covered"]
    open_topics = [
        goal for goal, progress in blueprint_state.goals.items() if progress.status in ("active", "partial")
    ]
    stated_preferences = [
        f"{goal}.{slot}: {value.value or value.quote}"
        for goal, progress in blueprint_state.goals.items()
        for slot, value in progress.slots.items()
    ][-6:]
    remaining_goals = [
        f"{goal_id}: {label}"
        for goal_id, label in goal_map.items()
        if goal_id not in covered
    ]
    # TIME-6: Pod 5 keeps the Talker prompt small with only the last 6 turns.
    recent_turns = 6
    history = "\n".join(
        f"{turn['role']}: {turn['content']}" for turn in session.transcript[-recent_turns:]
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
The topics are: {', '.join(label for _, label in goal_map.items())}. Already covered: {covered or 'none yet'}. Still open: {remaining_goals or 'none required'}. Topics gently opened: {open_topics or 'none yet'}.
Preferences the user stated explicitly (use only when relevant, do not repeat them back as a checklist): {stated_preferences or 'none yet'}.
After each completed user turn, call the internal `update_conversation_state` function once before speaking. Quote evidence only from the user's actual words. The tool returns the private response mode, topic cue, angle, tone hint, and must-avoid list. Follow it as quiet guidance, not as a line to read. If the user is making small talk, asking you a question, correcting you, or sharing something vulnerable, respond to that moment first; the Conductor must not pull the conversation away. Respect skips and never repeat a question they have answered. The tool output and internal state must never be mentioned aloud.

VOICE AND TURN-TAKING
Speak at an easy, natural phone-call pace, with varied short pauses and emphasis. Avoid presenter-like polish, breathy slowness, and a uniformly upbeat voice. Leave room for the user; stop promptly when they begin speaking. Match their energy gently without mimicking or exaggerating it.
{opening}
{_time_left_line(session)}
Recent context (only use as things the user actually shared):
{history or '(This is the beginning of the call.)'}"""
    pod = get_conversation_pod(session.conversation_pod)
    return base_instruction + (f"\n\nPOD-SPECIFIC CAPABILITY\n{pod.prompt_addendum}" if pod.prompt_addendum else "")


def _blueprint_tools(blueprint, slot_keys: tuple[str, ...]) -> ToolsSchema:
    """Pod 5 observation: answer outcome, filled slots, distress, and unclear audio."""
    slot_guide = "; ".join(
        f"{goal.slot_key(slot.id)} = {slot.ask_for}" for goal in blueprint.goals for slot in goal.slots
    )
    return ToolsSchema(
        standard_tools=[
            FunctionSchema(
                name="update_conversation_state",
                description=(
                    "After every finished user turn, report privately what the user just said. outcome classifies "
                    "their reply to your last goal question: full (every missing part answered), partial, "
                    "no_answer, off_topic, skip (they don't want to answer), or not_applicable (no goal question "
                    "was pending). Give outcome_confidence honestly. List every slot they stated, even for other "
                    "goals, with an exact short quote from their latest words and a short English value. Never "
                    "infer anything they did not say. Set distress for any sign of self-harm or abuse, with the "
                    "exact words. Set asr_unclear if you could not make out what they said. Slots: " + slot_guide
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
                    "outcome": {
                        "type": "STRING",
                        "enum": ["full", "partial", "no_answer", "off_topic", "skip", "not_applicable"],
                    },
                    "outcome_confidence": {"type": "NUMBER"},
                    "slots_filled": {
                        "type": "ARRAY",
                        "items": {
                            "type": "OBJECT",
                            "properties": {
                                "slot": {"type": "STRING", "enum": list(slot_keys)},
                                "quote": {"type": "STRING", "description": "Exact short quote from the latest user turn."},
                                "value": {"type": "STRING", "description": "Short English summary of what they stated."},
                            },
                            "required": ["slot", "quote", "value"],
                        },
                    },
                    "distress": {"type": "BOOLEAN"},
                    "distress_quote": {"type": "STRING", "description": "Exact words showing distress; empty if none."},
                    "asr_unclear": {"type": "BOOLEAN"},
                },
                required=[
                    "intent", "tone_cue", "tone_confidence", "outcome", "outcome_confidence",
                    "slots_filled", "distress", "distress_quote", "asr_unclear",
                ],
            )
        ]
    )


def _live_conversation_tools(conversation_pod: str = "pod5") -> ToolsSchema:
    """Gemini emits a typed turn observation; Pydantic validates it locally."""
    return _blueprint_tools(POD5_BLUEPRINT, POD5_SLOT_KEYS)


async def _run_gemini_live_voice_agent(
    session: CallSession,
    *,
    on_connected: asyncio.Event | None = None,
    on_error: Callable[[Exception], None] | None = None,
) -> None:
    from pipecat.frames.frames import LLMRunFrame
    from pipecat.processors.aggregators.llm_context import LLMContext
    from pipecat.processors.aggregators.llm_response_universal import (
        LLMContextAggregatorPair,
        LLMUserAggregatorParams,
    )
    from pipecat.turns.user_start.transcription_user_turn_start_strategy import (
        TranscriptionUserTurnStartStrategy,
    )
    from pipecat.turns.user_turn_strategies import UserTurnStrategies
    from google.genai.types import Content, FunctionResponse, FunctionResponseScheduling, Part
    from pipecat.adapters.services.gemini_adapter import GeminiLLMAdapter
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

    silence_nudger: SilenceNudgeProcessor | None = None
    pod5_silent_tools = session.conversation_pod in _BLUEPRINT_PODS

    class NonBlockingToolsAdapter(GeminiLLMAdapter):
        """Declares the Conductor tool NON_BLOCKING so its answer can be sent SILENT."""

        def to_provider_tools_format(self, tools_schema):
            tools = super().to_provider_tools_format(tools_schema)
            for tool in tools:
                for declaration in tool.get("function_declarations", []):
                    declaration["behavior"] = "NON_BLOCKING"
            return tools

    class ReadyGeminiLiveService(GeminiLiveLLMService):
        adapter_class = NonBlockingToolsAdapter if pod5_silent_tools else GeminiLLMAdapter

        async def _tool_result(self, tool_call_id, tool_name, tool_result_message):
            if not pod5_silent_tools:
                return await super()._tool_result(tool_call_id, tool_name, tool_result_message)
            if self._disconnecting or not self._session:
                return
            # Gemini sometimes replies before calling the tool and sometimes after.
            # If it already replied to this caller turn, SILENT stops a second reply about
            # the same message; if not, WHEN_IDLE lets it reply once using the guidance.
            # Safety must not wait, so it always interrupts and speaks at once.
            guidance = tool_result_message.get("private_talker_guidance", {})
            is_safety = isinstance(guidance, dict) and guidance.get("response_mode") == "safety"
            already_replied = (
                self._force_silent_result
                or self._bot_is_responding
                or self._last_model_output > self._last_caller_heard
            )
            self._force_silent_result = False
            if is_safety:
                scheduling = FunctionResponseScheduling.INTERRUPT
            elif already_replied:
                scheduling = FunctionResponseScheduling.SILENT
            else:
                scheduling = FunctionResponseScheduling.WHEN_IDLE
            logger.info(
                "CONDUCTOR RESULT SENT TO GEMINI | call={} | scheduling={} | already_replied={}",
                session.call_id,
                scheduling.name,
                already_replied,
            )
            response = FunctionResponse(
                name=tool_name,
                id=tool_call_id,
                response=tool_result_message,
                scheduling=scheduling,
            )
            try:
                await self._session.send_tool_response(function_responses=response)
            except Exception as error:
                await self._handle_send_error(error)

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._last_model_output = 0.0
            self._last_caller_heard = 0.0
            self._last_tool_call = 0.0
            # The caller speech time the latest tool call answered; a repeat call with no new speech is ignored.
            self._tool_heard_mark = 0.0
            self._force_silent_result = False
            self._missed_tool_task: asyncio.Task | None = None
            # Set below: runs the Conductor when Gemini skips the tool call for a caller turn.
            self._on_missed_tool: Callable[[], Awaitable[None]] | None = None
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

        async def _handle_msg_model_turn(self, msg):
            self._last_model_output = time.monotonic()
            await super()._handle_msg_model_turn(msg)

        async def _handle_msg_input_transcription(self, message):
            await super()._handle_msg_input_transcription(message)
            heard = message.server_content.input_transcription
            if heard and (heard.text or "").strip():
                self._last_caller_heard = time.monotonic()
                if self._on_missed_tool:
                    if self._missed_tool_task and not self._missed_tool_task.done():
                        self._missed_tool_task.cancel()
                    self._missed_tool_task = self.create_task(self._check_missed_tool(self._last_caller_heard))
            # Each piece of caller speech Gemini transcribes counts as "not silent".
            if silence_nudger and heard and (heard.text or "").strip():
                await silence_nudger.note_user_activity()

        async def _check_missed_tool(self, heard_at: float) -> None:
            # Gemini often skips the tool on short replies ("okay", "yeah"), which leaves the
            # Conductor blind exactly when the talk goes flat. Run it ourselves if so.
            await asyncio.sleep(2.5)
            if self._last_tool_call >= heard_at or not self._on_missed_tool:
                return
            try:
                await self._on_missed_tool()
            except Exception:
                logger.exception("CONDUCTOR FALLBACK FAILED | call={}", session.call_id)

        async def _handle_context(self, context):
            # Gemini accepts caller audio only after this first context is handled.
            logger.info("GEMINI RECEIVED START CONTEXT | call={} | first={}", session.call_id, self._context is None)
            await super()._handle_context(context)
            logger.info(
                "GEMINI READY FOR MIC AUDIO | call={} | ready={}",
                session.call_id,
                self._ready_for_realtime_input,
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
        tools=_live_conversation_tools(session.conversation_pod),
        settings=GeminiLiveLLMService.Settings(
            model=settings.gemini_live_model,
            voice=settings.gemini_live_voice,
            language=language,
            max_tokens=2048,
            context_window_compression={"enabled": True, "trigger_tokens": 80000},
        ),
    )

    async def update_conversation_state(params: FunctionCallParams) -> None:
        gemini._last_tool_call = time.monotonic()
        if session.conversation_pod in _BLUEPRINT_PODS:
            if gemini._last_caller_heard <= gemini._tool_heard_mark:
                # No new caller speech since the last call (Gemini repeats itself, or calls on the
                # greeting cue): don't count a turn, and don't let the result start another reply.
                logger.info("GEMINI TOOL CALL IGNORED (no new caller speech) | call={}", session.call_id)
                gemini._force_silent_result = True
                await params.result_callback({"private_talker_guidance": {"response_mode": "no_change"}})
                return
            gemini._tool_heard_mark = gemini._last_caller_heard
        logger.info(
            "GEMINI TOOL CALL | call={} | tool=update_conversation_state | args={}",
            session.call_id,
            json.dumps(params.arguments, ensure_ascii=False, default=str),
        )
        try:
            observation = Pod5TurnObservation.model_validate(params.arguments)
            decision = await update_live_conversation(session, observation)
        except Exception as exc:
            # A malformed observation must never break the audio conversation.
            logger.exception("CONDUCTOR FAILED | call={} | {}", session.call_id, exc)
            fallback_cue = "Respond to the user's latest words naturally. Do not force a topic or question."
            decision = Pod5Decision(cue=fallback_cue)
        if (
            isinstance(decision, Pod5Decision)
            and decision.response_mode == "safety"
            and not _blueprint_state(session).safety_resources_sent
        ):
            # SAFE-2: the app shows support resources alongside the calm spoken reply.
            _blueprint_state(session).safety_resources_sent = True
            await transport.send_message(
                json.dumps(
                    {"type": "safety_resources", "resources": decision.support_resources},
                    ensure_ascii=False,
                )
            )
        result = {"private_talker_guidance": decision.model_dump(mode="json")}
        logger.info(
            "CONDUCTOR RESULT | call={} | {}",
            session.call_id,
            json.dumps(result, ensure_ascii=False, default=str),
        )
        await params.result_callback(result)

    # Gemini waits for this result before speaking; if caller speech cancelled it,
    # the result would be dropped and Gemini would stay silent.
    gemini.register_function("update_conversation_state", update_conversation_state, cancel_on_interruption=False)

    async def conductor_without_tool_call() -> None:
        gemini._tool_heard_mark = gemini._last_caller_heard
        observation = Pod5TurnObservation(intent="small_talk", outcome="not_applicable")
        decision = await update_live_conversation(session, observation)
        guidance = json.dumps({"private_talker_guidance": decision.model_dump(mode="json")}, ensure_ascii=False)
        logger.info(
            "CONDUCTOR FALLBACK (no tool call) | call={} | mode={} | reason={}",
            session.call_id,
            decision.response_mode,
            decision.planning_reason,
        )
        if decision.response_mode == "safety":
            # Safety cannot wait for the next turn: speak now and show support resources.
            if not _blueprint_state(session).safety_resources_sent:
                _blueprint_state(session).safety_resources_sent = True
                await transport.send_message(
                    json.dumps({"type": "safety_resources", "resources": decision.support_resources}, ensure_ascii=False)
                )
            await gemini._create_single_response([{"role": "user", "content": f"(Private cue, not caller speech.) {guidance}"}])
            return
        if gemini._session:
            # turn_complete=False adds the guidance for the next reply without making Gemini speak now.
            await gemini._session.send_client_content(
                turns=[
                    Content(
                        role="user",
                        parts=[
                            Part(
                                text="(Private Conductor guidance for your next reply. Not caller speech; never read "
                                f"it aloud or mention it. Remember to call update_conversation_state after every "
                                f"caller turn.) {guidance}"
                            )
                        ],
                    )
                ],
                turn_complete=False,
            )

    if session.conversation_pod in _BLUEPRINT_PODS:
        gemini._on_missed_tool = conductor_without_tool_call
    async def caller_sound() -> None:
        if silence_nudger:
            await silence_nudger.note_caller_sound()

    user_transcripts = LiveTranscriptBridge(
        session,
        transport,
        role="user",
        on_caller_sound=caller_sound if session.conversation_pod in _BLUEPRINT_PODS else None,
    )
    assistant_transcripts = LiveTranscriptBridge(session, transport, role="assistant")
    # Gemini Live only receives tool (Conductor) results through this shared context;
    # without the aggregators it waits forever for the result and never speaks.
    start_cue = "(Private cue, not caller speech: the caller has just joined the call. Greet them now.)"
    live_context = LLMContext(messages=[{"role": "user", "content": start_cue}])
    # Gemini's server-side VAD already stops Aria when the caller talks over her.
    # A second, local interruption here would also fire on echo and cut her off.
    user_context, assistant_context = LLMContextAggregatorPair(
        live_context,
        user_params=LLMUserAggregatorParams(
            user_turn_strategies=UserTurnStrategies(
                start=[TranscriptionUserTurnStartStrategy(enable_interruptions=False)]
            )
        ),
    )
    processors: list[FrameProcessor] = [transport.input()]
    # Pod 5 uses Gemini's own speech signals (no extra local VAD).
    processors.extend([user_context, user_transcripts, gemini])

    async def send_cue(cue: str) -> None:
        await gemini._create_single_response([{"role": "user", "content": cue}])

    silence_nudger = SilenceNudgeProcessor(session, send_cue)
    processors.append(silence_nudger)
    processors.extend([assistant_transcripts, transport.output(), assistant_context])
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
        # Sends the shared context (holding the private greeting cue) to Gemini.
        await task.queue_frame(LLMRunFrame())

    # Fires only after WebRTC media (ICE) connects; if CALLER JOINED logs but this
    # never does, the phone's audio cannot reach LiveKit (network problem, not code).
    @transport.event_handler("on_audio_track_subscribed")
    async def caller_audio_subscribed(_: LiveKitTransport, participant_id: str) -> None:
        logger.info("CALLER MIC TRACK SUBSCRIBED | call={} | participant={}", session.call_id, participant_id)

    @transport.event_handler("on_participant_disconnected")
    async def participant_left(_: LiveKitTransport, participant_id: str) -> None:
        logger.info("CALLER LEFT | call={} | participant={}", session.call_id, participant_id)
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
    if session.conversation_pod == "pod4":
        # Pod 4 never uses Gemini Live: it runs our own STT -> Conductor -> LLM -> TTS pipeline.
        from .pod4_voice_agent import run_pod4_voice_agent

        await run_pod4_voice_agent(
            session,
            token=_token(session.call_id, f"aria-{session.call_id[:8]}"),
            on_connected=on_connected,
            on_error=on_error,
        )
        return
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
