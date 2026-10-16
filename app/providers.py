"""Provider contracts and a local Ollama-compatible chat adapter."""

from __future__ import annotations

import asyncio
import json
import logging
import re
from dataclasses import dataclass
from typing import Protocol
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen

from .config import settings

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ChatMessage:
    role: str
    content: str


class LlmProvider(Protocol):
    name: str

    async def complete(self, messages: list[ChatMessage], *, temperature: float) -> str: ...


class SttProvider(Protocol):
    async def transcribe(self, audio: bytes, *, language_mode: str) -> str: ...


class TtsProvider(Protocol):
    async def synthesize(self, text: str, *, language_mode: str) -> bytes: ...


class Transport(Protocol):
    async def receive(self) -> bytes: ...

    async def send(self, payload: bytes) -> None: ...


class OllamaChatProvider:
    name = "ollama"

    async def complete(self, messages: list[ChatMessage], *, temperature: float) -> str:
        return await asyncio.to_thread(self._complete_sync, messages, temperature)

    @staticmethod
    def _complete_sync(messages: list[ChatMessage], temperature: float) -> str:
        body = json.dumps(
            {
                "model": settings.ollama_model,
                "stream": False,
                "think": False,
                "keep_alive": "20m",
                "options": {
                    "temperature": temperature,
                    "num_predict": 180 if temperature <= 0.2 else 100,
                },
                "messages": [message.__dict__ for message in messages],
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
            raise RuntimeError(f"Local language model request failed: {exc}") from exc
        content = payload.get("message", {}).get("content")
        if not isinstance(content, str) or not content.strip():
            raise RuntimeError("The language model returned an empty response")
        return content.strip()


class GeminiChatProvider:
    """Gemini text generation through Google's REST API; the key stays server-side."""

    name = "gemini"

    async def complete(self, messages: list[ChatMessage], *, temperature: float) -> str:
        return await asyncio.to_thread(self._complete_sync, messages, temperature)

    @staticmethod
    def _complete_sync(messages: list[ChatMessage], temperature: float) -> str:
        if not settings.gemini_api_key:
            raise RuntimeError("GEMINI_API_KEY is not configured")

        system_parts = [message.content for message in messages if message.role == "system"]
        contents: list[dict[str, object]] = []
        for message in messages:
            if message.role == "system":
                continue
            role = "model" if message.role == "assistant" else "user"
            if contents and contents[-1]["role"] == role:
                parts = contents[-1]["parts"]
                assert isinstance(parts, list)
                parts.append({"text": message.content})
            else:
                contents.append({"role": role, "parts": [{"text": message.content}]})
        if not contents:
            raise RuntimeError("Gemini request has no conversation messages")

        body: dict[str, object] = {
            "contents": contents,
            "generationConfig": {"temperature": temperature, "maxOutputTokens": 180},
        }
        if system_parts:
            body["systemInstruction"] = {"parts": [{"text": "\n\n".join(system_parts)}]}
        request = Request(
            f"https://generativelanguage.googleapis.com/v1beta/models/{quote(settings.gemini_model, safe='')}:generateContent",
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json", "x-goog-api-key": settings.gemini_api_key},
            method="POST",
        )
        try:
            with urlopen(request, timeout=settings.gemini_timeout_seconds) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            # Keep API credentials and request content out of error messages.
            raise RuntimeError(f"Gemini API returned HTTP {exc.code}") from exc
        except (URLError, TimeoutError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"Gemini request failed: {exc}") from exc

        candidates = payload.get("candidates", [])
        parts = candidates[0].get("content", {}).get("parts", []) if candidates else []
        text = "".join(part.get("text", "") for part in parts if isinstance(part.get("text"), str))
        if not text.strip():
            raise RuntimeError("Gemini returned no text (the response may have been blocked)")
        return text.strip()


class DemoProvider:
    """Clearly marked deterministic fallback so the API runs without a model."""

    name = "demo"

    async def complete(self, messages: list[ChatMessage], *, temperature: float) -> str:
        del temperature
        latest_user = next((m.content for m in reversed(messages) if m.role == "user"), "")
        system = messages[0].content.lower() if messages else ""
        if "you are the conductor" in system:
            ids = (
                "weekend_downtime", "conflict", "partner_wants", "family_future",
                "values_priorities", "communication_affection", "real_life_scenario",
            )
            covered_line = re.search(r"already covered: (\[[^\]]*\])", system)
            try:
                covered = set(json.loads(covered_line.group(1))) if covered_line else set()
            except (json.JSONDecodeError, AttributeError):
                covered = set()
            text = latest_user.lower()
            matched: list[str] = []
            demo_patterns = {
                "weekend_downtime": r"weekend|relax|downtime|free time|aaram|chhutti|sunday",
                "conflict": r"conflict|argument|fight|disagree|repair|apolog|gussa|jhag|maafi",
                "partner_wants": r"partner|relationship|looking for|want in|saathi",
                "family_future": r"family|children|kids|future|shaadi|bachch|parivaar",
                "values_priorities": r"value|priority|important|matter most|belief|principle|usool",
                "communication_affection": r"communicat|affection|love language|feel loved|care|pyaar",
                "real_life_scenario": r"scenario|what would you do|if your partner|imagine|maan lo",
            }
            matched = [goal for goal, pattern in demo_patterns.items() if re.search(pattern, text)]
            next_goal = next((goal for goal in ids if goal not in covered and goal not in matched), ids[-1])
            return json.dumps({"covered_now": matched, "next_goal": next_goal, "acknowledgement": "I hear you"})
        if "you are the talker" in system:
            if "14-minute wrap-up has begun" in system:
                return "We have a little time left. Is there anything else you want me to understand about what helps you feel connected?"
            goal_match = re.search(r"about this goal: ([^.]*)", system)
            goal = goal_match.group(1).strip() if goal_match else "what matters to you"
            questions = {
                "weekend and downtime": "What part of a weekend helps you feel most like yourself?",
                "conflict and repair": "When tension comes up, what helps you feel ready to reconnect?",
                "what they want in a partner": "What quality would you most hope to share with a partner?",
                "family and future hopes": "When you picture the years ahead, what feels meaningful to you?",
                "values and priorities": "Which value do you want your everyday choices to reflect?",
                "communication and affection": "What small gesture helps you feel cared for?",
                "a real-life relationship scenario": "If plans changed at the last minute, what would good communication look like?",
            }
            if not latest_user:
                return "I’m an AI companion, here to listen and learn what matters to you. What part of a weekend do you most look forward to?"
            question = questions.get(goal, "What feels most important to you about that?")
            if "hinglish" in system:
                return "Samajh raha hoon. " + question
            return "Thanks for sharing that. " + question
        if "hinglish" in system:
            return (
                "Thanks, is baat se tumhare liye kya sabse zyada matter karta hai?"
                if latest_user
                else "Chalo shuru karte hain—weekend ka woh kaunsa hissa hai jiska tum intezaar karte ho?"
            )
        return (
            "Thanks for sharing that. What matters most to you in that moment?"
            if latest_user
            else "Let’s start with an easy one: what part of a weekend do you most look forward to?"
        )


class ProviderRouter:
    def __init__(self) -> None:
        self.providers: dict[str, LlmProvider] = {
            "gemini": GeminiChatProvider(),
            "ollama": OllamaChatProvider(),
            "demo": DemoProvider(),
        }

    async def complete(self, messages: list[ChatMessage], *, temperature: float) -> tuple[str, str]:
        primary = self.providers.get(settings.llm_provider)
        if primary is None:
            raise RuntimeError(f"Unsupported LLM provider: {settings.llm_provider}")
        if primary.name == "demo":
            return await primary.complete(messages, temperature=temperature), primary.name
        try:
            return await primary.complete(messages, temperature=temperature), primary.name
        except Exception as primary_error:
            fallback = self.providers.get(settings.llm_fallback)
            if fallback is None or fallback.name == primary.name:
                raise
            try:
                answer = await fallback.complete(messages, temperature=temperature)
            except Exception as fallback_error:
                logger.error("Primary LLM %s and fallback %s both failed (%s; %s)", primary.name, fallback.name, primary_error, fallback_error)
                raise RuntimeError("Primary and fallback language models are both unavailable") from fallback_error
            logger.warning("Primary LLM %s failed; using %s fallback: %s", primary.name, fallback.name, primary_error)
            return answer, f"{fallback.name}-fallback"


provider_router = ProviderRouter()
