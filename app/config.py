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
    conversation_pod: str = os.getenv("SYNQ_CONVERSATION_POD", "pod4").strip().lower()
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
    # Pod 4: our own voice pipeline. Each part is plug-and-play from .env (primary, then fallback).
    pod4_stt: str = os.getenv("SYNQ_POD4_STT", "whisper").strip().lower()            # whisper | sarvam
    pod4_stt_fallback: str = os.getenv("SYNQ_POD4_STT_FALLBACK", "sarvam").strip().lower()
    pod4_tts: str = os.getenv("SYNQ_POD4_TTS", "piper").strip().lower()              # piper | sarvam
    pod4_tts_fallback: str = os.getenv("SYNQ_POD4_TTS_FALLBACK", "sarvam").strip().lower()
    pod4_llm: str = os.getenv("SYNQ_POD4_LLM", "gemini").strip().lower()             # gemini | groq | ollama
    pod4_llm_fallback: str = os.getenv("SYNQ_POD4_LLM_FALLBACK", "ollama").strip().lower()
    pod4_gemini_api_key: str = os.getenv("SYNQ_POD4_GEMINI_API_KEY", "") or os.getenv("GEMINI_API_KEY", "")
    pod4_gemini_model: str = os.getenv("SYNQ_POD4_GEMINI_MODEL", os.getenv("SYNQ_GEMINI_MODEL", "gemini-3.8-flash"))
    # How much Gemini "thinks" before answering (thinking tokens are billed as output and add delay).
    # Gemini 3+: minimal | low | medium | high. Gemini 2.5 models get a thinking budget of 0 instead.
    pod4_gemini_thinking: str = os.getenv("SYNQ_POD4_GEMINI_THINKING", "low").strip().lower()
    groq_api_key: str = os.getenv("GROQ_API_KEY", "")
    pod4_groq_model: str = os.getenv("SYNQ_POD4_GROQ_MODEL", "qwen/qwen3.8-27b")
    # A second Groq model has its own rate limit, so it is a quick fallback when the first is busy.
    pod4_groq_backup_model: str = os.getenv("SYNQ_POD4_GROQ_BACKUP_MODEL", "openai/gpt-oss-20b")
    pod4_ollama_model: str = os.getenv("SYNQ_POD4_OLLAMA_MODEL", "qwen3:4b")
    pod4_whisper_model: str = os.getenv("SYNQ_POD4_WHISPER_MODEL", os.getenv("SYNQ_VOICE_STT_MODEL", "base"))
    pod4_piper_voice: str = os.getenv("SYNQ_POD4_PIPER_VOICE", os.getenv("SYNQ_VOICE_TTS_VOICE", "en_US-amy-medium"))
    # Buddy's live voice speaks with its own Piper voice, so it never sounds like Synthe (the call).
    buddy_piper_voice: str = os.getenv("BUDDY_PIPER_VOICE", "en_US-ryan-medium")
    # Buddy's own TTS order (piper | sarvam), separate from Pod 4's, and its Sarvam voice.
    # Buddy's own STT order (whisper | sarvam), so changing Pod 4's STT doesn't change Buddy's.
    buddy_stt: str = os.getenv("BUDDY_STT", "whisper").strip().lower()
    buddy_stt_fallback: str = os.getenv("BUDDY_STT_FALLBACK", "sarvam").strip().lower()
    buddy_tts: str = os.getenv("BUDDY_TTS", "piper").strip().lower()
    buddy_tts_fallback: str = os.getenv("BUDDY_TTS_FALLBACK", "sarvam").strip().lower()
    buddy_sarvam_tts_model: str = os.getenv("BUDDY_SARVAM_MODEL", "bulbul:v3")  # v2 was retired by Sarvam (Oct 2026)
    buddy_sarvam_speaker: str = os.getenv("BUDDY_SARVAM_SPEAKER", "priya")
    buddy_sarvam_language: str = os.getenv("BUDDY_SARVAM_LANGUAGE", "en-IN")  # Roman Hinglish reads well as en-IN
    buddy_sarvam_pace: float = float(os.getenv("BUDDY_SARVAM_PACE", "0.88"))  # 1.0 sounded rushed; range 0.5-2.0
    sarvam_api_key: str = os.getenv("SARVAM_API_KEY", "")
    pod4_sarvam_stt_model: str = os.getenv("SYNQ_POD4_SARVAM_STT_MODEL", "")      # empty: library default
    # saaras:v3 only: "translit" writes Hinglish in Roman script (Pod 4's guard, cues and prompts expect it).
    pod4_sarvam_stt_mode: str = os.getenv("SYNQ_POD4_SARVAM_STT_MODE", "")
    pod4_sarvam_tts_model: str = os.getenv("SYNQ_POD4_SARVAM_TTS_MODEL", "")      # empty: library default (Bulbul)
    pod4_sarvam_tts_voice: str = os.getenv("SYNQ_POD4_SARVAM_TTS_VOICE", "")      # empty: model default speaker
    pod4_sarvam_tts_pace: float = float(os.getenv("SYNQ_POD4_SARVAM_TTS_PACE", "1.0"))  # 0.5-2.0 on Bulbul v3
    # Accounts. DATABASE_URL example (local socket, no password): postgresql://bs@/synq?host=/var/run/postgresql
    database_url: str = os.getenv("DATABASE_URL", "")
    # Signs login tokens. Keep it secret and long; changing it signs everyone out.
    auth_jwt_secret: str = os.getenv("AUTH_JWT_SECRET", "")
    auth_access_token_minutes: int = int(os.getenv("AUTH_ACCESS_TOKEN_MINUTES", "15"))
    auth_refresh_token_days: int = int(os.getenv("AUTH_REFRESH_TOKEN_DAYS", "30"))
    # Uploaded profile photos and voice reflections are stored here.
    media_dir: Path = Path(os.getenv("MEDIA_DIR", str(Path(__file__).resolve().parents[1] / "media")))
    export_call_logs: bool = os.getenv("SYNQ_EXPORT_CALL_LOGS", "true").strip().lower() in {"1", "true", "yes", "on"}
    call_log_directory: Path = Path(
        os.getenv("SYNQ_CALL_LOG_DIRECTORY", str(Path(__file__).resolve().parents[1] / "call_logs"))
    )


settings = Settings()
