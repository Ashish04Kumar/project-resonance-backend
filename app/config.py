"""Small environment-backed settings module; no secrets are stored in source."""

from dataclasses import dataclass
import os
from pathlib import Path

from dotenv import load_dotenv


load_dotenv(Path(__file__).resolve().parents[1] / ".env")


def _csv(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in value.split(",") if item.strip())


@dataclass(frozen=True)
class Settings:
    llm_provider: str = os.getenv("SYNQ_LLM_PROVIDER", "ollama")
    llm_fallback: str = os.getenv("SYNQ_LLM_FALLBACK", "gemini")
    gemini_api_key: str = os.getenv("GEMINI_API_KEY", "")
    gemini_model: str = os.getenv("SYNQ_GEMINI_MODEL", "gemini-3.8-flash")
    gemini_live_model: str = os.getenv("SYNQ_GEMINI_LIVE_MODEL", "gemini-3.8-live")
    gemini_live_voice: str = os.getenv("SYNQ_GEMINI_LIVE_VOICE", "Kore")
    voice_provider: str = os.getenv("SYNQ_VOICE_PROVIDER", "gemini_live")
    conversation_pod: str = os.getenv("SYNQ_CONVERSATION_POD", "pod1").strip().lower()
    conversation_silence_nudge_seconds: float = float(
        os.getenv("SYNQ_CONVERSATION_SILENCE_NUDGE_SECONDS", "10")
    )
    gemini_timeout_seconds: float = float(os.getenv("SYNQ_GEMINI_TIMEOUT_SECONDS", "30"))
    ollama_url: str = os.getenv("SYNQ_OLLAMA_URL", "http://127.0.0.1:11434")
    ollama_model: str = os.getenv("SYNQ_OLLAMA_MODEL", "smollm2:360m")
    # Keep model failures from blocking a live conversation for too long.
    llm_timeout_seconds: float = float(os.getenv("SYNQ_LLM_TIMEOUT_SECONDS", "45"))
    wrap_up_seconds: int = int(os.getenv("SYNQ_WRAP_UP_SECONDS", "840"))
    max_call_seconds: int = int(os.getenv("SYNQ_MAX_CALL_SECONDS", "1020"))
    allowed_origins: tuple[str, ...] = _csv(
        os.getenv("SYNQ_ALLOWED_ORIGINS", "http://localhost:8081,http://localhost:19006")
    )
    livekit_url: str = os.getenv("SYNQ_LIVEKIT_URL", "ws://127.0.0.1:7880")
    livekit_api_key: str = os.getenv("LIVEKIT_API_KEY", "devkey")
    livekit_api_secret: str = os.getenv("LIVEKIT_API_SECRET", "secret")
    voice_stt_model: str = os.getenv("SYNQ_VOICE_STT_MODEL", "base")
    voice_tts_voice: str = os.getenv("SYNQ_VOICE_TTS_VOICE", "en_US-amy-medium")
    voice_startup_seconds: float = float(os.getenv("SYNQ_VOICE_STARTUP_SECONDS", "120"))
    export_call_logs: bool = os.getenv("SYNQ_EXPORT_CALL_LOGS", "false").strip().lower() in {"1", "true", "yes", "on"}
    call_log_directory: Path = Path(
        os.getenv("SYNQ_CALL_LOG_DIRECTORY", str(Path(__file__).resolve().parents[1] / "call_logs"))
    )


settings = Settings()
