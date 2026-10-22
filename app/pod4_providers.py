"""Pod 4 plug-and-play providers: speech-to-text, the text LLM, and text-to-speech.

Each part is chosen in backend/.env with a primary and a fallback (PROV-1/PROV-2):

    SYNQ_POD4_STT=whisper   SYNQ_POD4_STT_FALLBACK=sarvam    (laptop Whisper, or Sarvam Saarika)
    SYNQ_POD4_TTS=piper     SYNQ_POD4_TTS_FALLBACK=sarvam    (laptop Piper, or Sarvam Bulbul)
    SYNQ_POD4_LLM=gemini    SYNQ_POD4_LLM_FALLBACK=ollama    (Gemini text, Groq, or laptop Ollama)

STT and TTS fall back when the primary cannot start (missing key, library or model).
The LLM falls back per request when the primary errors or times out (PROV-4).
Sarvam's libraries are imported only when Sarvam is chosen, so the laptop-only setup
needs nothing extra. API keys stay on the backend.
"""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen

import numpy as np
from loguru import logger

from .config import settings


# --- Speech-to-text ---------------------------------------------------------------


def _whisper_service_class():
    """Pipecat's Whisper service, with the whole transcription done in a worker thread.

    faster-whisper returns its segments lazily: the real decoding happens while they are read.
    Pipecat reads them on the event loop, which froze the whole call (VAD, turn taking, Aria's
    audio) for up to a second on every utterance. This version reads them inside the thread.
    """
    from pipecat.frames.frames import TranscriptionFrame
    from pipecat.services.whisper.stt import WhisperSTTService
    from pipecat.utils.time import time_now_iso8601

    class ThreadedWhisperSTTService(WhisperSTTService):
        async def run_stt(self, audio: bytes):
            if not self._model:
                async for frame in super().run_stt(audio):
                    yield frame
                return
            await self.start_processing_metrics()
            samples = np.frombuffer(audio, dtype=np.int16).astype(np.float32) / 32768.0
            limit = self._settings.no_speech_prob
            language = self._settings.language

            def transcribe() -> str:
                segments, _ = self._model.transcribe(samples, language=language)
                return "".join(f"{segment.text} " for segment in segments if segment.no_speech_prob < limit)

            text = await asyncio.to_thread(transcribe)
            await self.stop_processing_metrics()
            if text.strip():
                await self._handle_transcription(text, True, language)
                yield TranscriptionFrame(text, self._user_id, time_now_iso8601(), language)

    return ThreadedWhisperSTTService


def _make_stt(name: str):
    if name == "whisper":
        WhisperService = _whisper_service_class()

        # language=None lets Whisper detect English or Hindi per utterance (Hinglish callers).
        return WhisperService(
            settings=WhisperService.Settings(model=settings.pod4_whisper_model),
            device="cpu",
            compute_type="int8",
            language=None,
        )
    if name == "sarvam":
        if not settings.sarvam_api_key:
            raise RuntimeError("SARVAM_API_KEY is not set")
        from pipecat.services.sarvam.stt import SarvamSTTService  # needs the sarvamai package

        kwargs: dict = {"model": settings.pod4_sarvam_stt_model} if settings.pod4_sarvam_stt_model else {}
        if settings.pod4_sarvam_stt_mode:
            kwargs["mode"] = settings.pod4_sarvam_stt_mode
        return SarvamSTTService(api_key=settings.sarvam_api_key, **kwargs)
    raise RuntimeError(f"Unknown Pod 4 STT provider: {name}")


def make_stt(primary: str | None = None, fallback: str | None = None) -> tuple[object, str]:
    """Pod 4's STT with its fallback. Another agent (Buddy) can pass its own order."""
    return _with_fallback("STT", primary or settings.pod4_stt, fallback or settings.pod4_stt_fallback, _make_stt)


# --- Text-to-speech ---------------------------------------------------------------


def _make_tts(name: str, piper_voice: str | None = None):
    if name == "piper":
        from pipecat.services.piper.tts import PiperTTSService

        return PiperTTSService(settings=PiperTTSService.Settings(voice=piper_voice or settings.pod4_piper_voice))
    if name == "sarvam":
        if not settings.sarvam_api_key:
            raise RuntimeError("SARVAM_API_KEY is not set")
        from pipecat.services.sarvam.tts import SarvamTTSService

        tts_settings: dict = {"pace": settings.pod4_sarvam_tts_pace}
        if settings.pod4_sarvam_tts_model:
            tts_settings["model"] = settings.pod4_sarvam_tts_model
        if settings.pod4_sarvam_tts_voice:
            tts_settings["voice"] = settings.pod4_sarvam_tts_voice
        return SarvamTTSService(api_key=settings.sarvam_api_key, settings=SarvamTTSService.Settings(**tts_settings))
    raise RuntimeError(f"Unknown Pod 4 TTS provider: {name}")


def make_tts(piper_voice: str | None = None) -> tuple[object, str]:
    """Pod 4's TTS with its fallback. `piper_voice` lets another agent (Buddy) use a different Piper voice."""
    return _with_fallback("TTS", settings.pod4_tts, settings.pod4_tts_fallback, lambda name: _make_tts(name, piper_voice))


def _with_fallback(kind: str, primary: str, fallback: str, factory) -> tuple[object, str]:
    try:
        service = factory(primary)
        logger.info("POD4 {} | using {}", kind, primary)
        return service, primary
    except Exception as primary_error:
        if not fallback or fallback == primary:
            raise
        logger.warning("POD4 {} | {} could not start ({}); trying {}", kind, primary, primary_error, fallback)
        service = factory(fallback)
        logger.info("POD4 {} | using fallback {}", kind, fallback)
        return service, fallback


# --- Text LLM -----------------------------------------------------------------------


@dataclass
class Message:
    role: str  # system | user | assistant
    content: str


def _gemini_generation_config(temperature: float, max_tokens: int, thinking: bool = True) -> dict[str, object]:
    """Generation settings, with thinking kept low so replies stay fast and cheap."""
    config: dict[str, object] = {"temperature": temperature, "maxOutputTokens": max_tokens}
    if thinking and settings.pod4_gemini_thinking:
        if settings.pod4_gemini_model.startswith("gemini-2.5"):
            config["thinkingConfig"] = {"thinkingBudget": 0}
        else:
            config["thinkingConfig"] = {"thinkingLevel": settings.pod4_gemini_thinking}
    return config


def _gemini_body(messages: list[Message], temperature: float, max_tokens: int, thinking: bool = True) -> bytes:
    system = "\n\n".join(m.content for m in messages if m.role == "system")
    contents: list[dict[str, object]] = []
    for message in messages:
        if message.role == "system":
            continue
        role = "model" if message.role == "assistant" else "user"
        if contents and contents[-1]["role"] == role:
            contents[-1]["parts"].append({"text": message.content})  # type: ignore[union-attr]
        else:
            contents.append({"role": role, "parts": [{"text": message.content}]})
    body: dict[str, object] = {"contents": contents, "generationConfig": _gemini_generation_config(temperature, max_tokens, thinking)}
    if system:
        body["systemInstruction"] = {"parts": [{"text": system}]}
    return json.dumps(body).encode("utf-8")


def _gemini(messages: list[Message], temperature: float, max_tokens: int, thinking: bool = True) -> str:
    if not settings.pod4_gemini_api_key:
        raise RuntimeError("No Gemini key for Pod 4 (SYNQ_POD4_GEMINI_API_KEY or GEMINI_API_KEY)")
    request = Request(
        "https://generativelanguage.googleapis.com/v1beta/models/"
        f"{quote(settings.pod4_gemini_model, safe='')}:generateContent",
        data=_gemini_body(messages, temperature, max_tokens, thinking),
        headers={"Content-Type": "application/json", "x-goog-api-key": settings.pod4_gemini_api_key},
        method="POST",
    )
    try:
        with urlopen(request, timeout=settings.gemini_timeout_seconds) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        if exc.code == 400 and thinking:
            # This model doesn't accept the thinking setting: ask again without it.
            return _gemini(messages, temperature, max_tokens, thinking=False)
        # Keep keys and content out of error messages.
        raise RuntimeError(f"Gemini returned HTTP {exc.code}") from exc
    except (URLError, TimeoutError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Gemini request failed: {exc}") from exc
    candidates = payload.get("candidates", [])
    parts = candidates[0].get("content", {}).get("parts", []) if candidates else []
    text = "".join(part.get("text", "") for part in parts if isinstance(part.get("text"), str))
    if not text.strip():
        raise RuntimeError("Gemini returned no text")
    return text.strip()


def _ollama(messages: list[Message], temperature: float, max_tokens: int) -> str:
    body = json.dumps(
        {
            "model": settings.pod4_ollama_model,
            "stream": False,
            "think": False,
            "keep_alive": "20m",
            "options": {"temperature": temperature, "num_predict": max_tokens},
            "messages": [{"role": m.role, "content": m.content} for m in messages],
        }
    ).encode("utf-8")
    request = Request(
        f"{settings.ollama_url.rstrip('/')}/api/chat",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlopen(request, timeout=settings.llm_timeout_seconds) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (URLError, TimeoutError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Ollama request failed: {exc}") from exc
    content = payload.get("message", {}).get("content")
    if not isinstance(content, str) or not content.strip():
        raise RuntimeError("Ollama returned an empty response")
    return content.strip()


def _groq(messages: list[Message], temperature: float, max_tokens: int, model: str | None = None) -> str:
    """Groq's OpenAI-compatible chat API (fast hosted open models)."""
    if not settings.groq_api_key:
        raise RuntimeError("GROQ_API_KEY is not set")
    model = model or settings.pod4_groq_model
    body: dict[str, object] = {
        "model": model,
        "messages": [{"role": m.role, "content": m.content} for m in messages],
        "temperature": temperature,
        "max_completion_tokens": max_tokens,
    }
    # Keep reasoning short so the call stays fast enough for a voice reply.
    if model.startswith("qwen/"):
        body["reasoning_effort"] = "none"
    elif model.startswith("openai/gpt-oss"):
        body["reasoning_effort"] = "low"
    request = Request(
        "https://api.groq.com/openai/v1/chat/completions",
        data=json.dumps(body).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {settings.groq_api_key}",
            "Content-Type": "application/json",
            "User-Agent": "resonance-backend",
        },
        method="POST",
    )
    for attempt in range(2):
        try:
            with urlopen(request, timeout=settings.gemini_timeout_seconds) as response:
                payload = json.loads(response.read().decode("utf-8"))
            break
        except HTTPError as exc:
            wait = float(exc.headers.get("retry-after") or 0) if exc.code == 429 else 0
            if exc.code == 429 and attempt == 0 and 0 < wait <= 3:
                time.sleep(wait)  # a short rate-limit pause is cheaper than switching models
                continue
            raise RuntimeError(f"Groq returned HTTP {exc.code}") from exc
        except (URLError, TimeoutError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"Groq request failed: {exc}") from exc
    choices = payload.get("choices") or []
    content = choices[0].get("message", {}).get("content") if choices else None
    if not isinstance(content, str) or not content.strip():
        raise RuntimeError("Groq returned no text")
    return content.strip()


def _groq_backup(messages: list[Message], temperature: float, max_tokens: int) -> str:
    return _groq(messages, temperature, max_tokens, model=settings.pod4_groq_backup_model)


_LLMS = {"gemini": _gemini, "groq": _groq, "groq_backup": _groq_backup, "ollama": _ollama}


async def complete(
    messages: list[Message], *, temperature: float, max_tokens: int, timeout: float | None = None
) -> tuple[str, str]:
    """Run the primary LLM, falling back per request. Returns (text, provider name)."""
    # SYNQ_POD4_LLM_FALLBACK may list several, tried in order: for example "groq_backup,ollama".
    fallbacks = [name.strip() for name in settings.pod4_llm_fallback.split(",") if name.strip()]
    order = [settings.pod4_llm] + [name for name in fallbacks if name != settings.pod4_llm]
    last_error: Exception | None = None
    for name in order:
        call = _LLMS.get(name)
        if call is None:
            last_error = RuntimeError(f"Unknown Pod 4 LLM provider: {name}")
            continue
        try:
            # The laptop model on CPU needs longer than a cloud model.
            limit = timeout or (settings.llm_timeout_seconds if name == "ollama" else settings.gemini_timeout_seconds)
            text = await asyncio.wait_for(asyncio.to_thread(call, messages, temperature, max_tokens), limit)
            return text, name
        except Exception as error:
            last_error = error
            logger.warning("POD4 LLM | {} failed: {}", name, repr(error))
    raise RuntimeError(f"All Pod 4 LLM providers failed: {last_error}")


# --- Streaming LLM (Talker replies) ----------------------------------------------------------
# Each streamer is a plain generator that yields text pieces as the provider sends them; it runs
# in a worker thread and the pieces are handed to the event loop through a queue.


def _sse_lines(response):
    """Server-sent events: yield the JSON text after each 'data:' line."""
    for raw in response:
        line = raw.decode("utf-8", errors="ignore").strip()
        if line.startswith("data:"):
            yield line[5:].strip()


def _groq_stream(messages: list[Message], temperature: float, max_tokens: int, model: str | None = None):
    if not settings.groq_api_key:
        raise RuntimeError("GROQ_API_KEY is not set")
    model = model or settings.pod4_groq_model
    body: dict[str, object] = {
        "model": model,
        "messages": [{"role": m.role, "content": m.content} for m in messages],
        "temperature": temperature,
        "max_completion_tokens": max_tokens,
        "stream": True,
    }
    if model.startswith("qwen/"):
        body["reasoning_effort"] = "none"
    elif model.startswith("openai/gpt-oss"):
        body["reasoning_effort"] = "low"
    request = Request(
        "https://api.groq.com/openai/v1/chat/completions",
        data=json.dumps(body).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {settings.groq_api_key}",
            "Content-Type": "application/json",
            "User-Agent": "resonance-backend",
        },
        method="POST",
    )
    try:
        with urlopen(request, timeout=settings.gemini_timeout_seconds) as response:
            for data in _sse_lines(response):
                if data == "[DONE]":
                    return
                choices = json.loads(data).get("choices") or []
                piece = (choices[0].get("delta") or {}).get("content") if choices else None
                if isinstance(piece, str) and piece:
                    yield piece
    except HTTPError as exc:
        raise RuntimeError(f"Groq returned HTTP {exc.code}") from exc
    except (URLError, TimeoutError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Groq stream failed: {exc}") from exc


def _groq_backup_stream(messages: list[Message], temperature: float, max_tokens: int):
    yield from _groq_stream(messages, temperature, max_tokens, model=settings.pod4_groq_backup_model)


def _ollama_stream(messages: list[Message], temperature: float, max_tokens: int):
    body = json.dumps(
        {
            "model": settings.pod4_ollama_model,
            "stream": True,
            "think": False,
            "keep_alive": "20m",
            "options": {"temperature": temperature, "num_predict": max_tokens},
            "messages": [{"role": m.role, "content": m.content} for m in messages],
        }
    ).encode("utf-8")
    request = Request(f"{settings.ollama_url.rstrip('/')}/api/chat", data=body, headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urlopen(request, timeout=settings.llm_timeout_seconds) as response:
            for raw in response:  # one JSON object per line
                if not raw.strip():
                    continue
                chunk = json.loads(raw.decode("utf-8"))
                piece = (chunk.get("message") or {}).get("content")
                if isinstance(piece, str) and piece:
                    yield piece
                if chunk.get("done"):
                    return
    except (URLError, TimeoutError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Ollama stream failed: {exc}") from exc


def _gemini_stream(messages: list[Message], temperature: float, max_tokens: int, thinking: bool = True):
    if not settings.pod4_gemini_api_key:
        raise RuntimeError("No Gemini key for Pod 4 (SYNQ_POD4_GEMINI_API_KEY or GEMINI_API_KEY)")
    request = Request(
        "https://generativelanguage.googleapis.com/v1beta/models/"
        f"{quote(settings.pod4_gemini_model, safe='')}:streamGenerateContent?alt=sse",
        data=_gemini_body(messages, temperature, max_tokens, thinking),
        headers={"Content-Type": "application/json", "x-goog-api-key": settings.pod4_gemini_api_key},
        method="POST",
    )
    try:
        with urlopen(request, timeout=settings.gemini_timeout_seconds) as response:
            for data in _sse_lines(response):
                candidates = json.loads(data).get("candidates") or []
                parts = candidates[0].get("content", {}).get("parts", []) if candidates else []
                # Skip thought summaries if the model sends any; only spoken text goes on.
                piece = "".join(part.get("text", "") for part in parts if isinstance(part.get("text"), str) and not part.get("thought"))
                if piece:
                    yield piece
    except HTTPError as exc:
        if exc.code == 400 and thinking:
            yield from _gemini_stream(messages, temperature, max_tokens, thinking=False)
            return
        raise RuntimeError(f"Gemini returned HTTP {exc.code}") from exc
    except (URLError, TimeoutError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Gemini stream failed: {exc}") from exc


_STREAMS = {"gemini": _gemini_stream, "groq": _groq_stream, "groq_backup": _groq_backup_stream, "ollama": _ollama_stream}

_DONE = object()


async def _pieces_from_thread(generator_factory, limit: float):
    """Run a blocking generator in a thread and yield its pieces on the event loop."""
    loop = asyncio.get_running_loop()
    queue: asyncio.Queue = asyncio.Queue()
    stop = False

    def run() -> None:
        try:
            for piece in generator_factory():
                if stop:
                    return
                loop.call_soon_threadsafe(queue.put_nowait, piece)
            loop.call_soon_threadsafe(queue.put_nowait, _DONE)
        except Exception as error:  # handed to the loop and raised there
            loop.call_soon_threadsafe(queue.put_nowait, error)

    worker = loop.run_in_executor(None, run)
    try:
        while True:
            item = await asyncio.wait_for(queue.get(), limit)
            if item is _DONE:
                return
            if isinstance(item, Exception):
                raise item
            yield item
    finally:
        stop = True  # the thread stops at its next piece; we don't wait for it
        worker.add_done_callback(lambda _: None)


async def stream_complete(messages: list[Message], *, temperature: float, max_tokens: int):
    """Stream the reply as (piece, provider) pairs.

    Falls back to the next provider only if one fails before sending anything; once text has
    started, an error just ends the stream (what was said stays said).
    """
    fallbacks = [name.strip() for name in settings.pod4_llm_fallback.split(",") if name.strip()]
    order = [settings.pod4_llm] + [name for name in fallbacks if name != settings.pod4_llm]
    last_error: Exception | None = None
    for name in order:
        streamer = _STREAMS.get(name)
        if streamer is None:
            last_error = RuntimeError(f"Unknown Pod 4 LLM provider: {name}")
            continue
        limit = settings.llm_timeout_seconds if name == "ollama" else settings.gemini_timeout_seconds
        started = False
        try:
            async for piece in _pieces_from_thread(lambda: streamer(messages, temperature, max_tokens), limit):
                started = True
                yield piece, name
            return
        except Exception as error:
            if started:
                logger.warning("POD4 LLM STREAM | {} stopped mid-reply: {}", name, repr(error))
                return
            last_error = error
            logger.warning("POD4 LLM STREAM | {} failed: {}", name, repr(error))
    raise RuntimeError(f"All Pod 4 LLM providers failed: {last_error}")
