"""Starts and tracks short-lived Pipecat workers for development voice calls."""

from __future__ import annotations

import asyncio
import ipaddress
from urllib.parse import urlsplit, urlunsplit

from loguru import logger

from .config import settings
from .conversation import CallSession
from .voice_agent import participant_token, run_voice_agent

_agents: dict[str, asyncio.Task[None]] = {}


def client_livekit_url(client_host: str | None) -> str:
    configured = urlsplit(settings.livekit_url)
    if not client_host or configured.hostname not in {"127.0.0.1", "localhost", "::1"}:
        return settings.livekit_url
    address = ipaddress.ip_address(client_host)
    if not address.is_private:
        raise ValueError("The phone must connect through the same private Wi-Fi network.")
    return urlunsplit((configured.scheme, f"{address.compressed}:{configured.port or 7880}", "", "", ""))


async def start_agent(session: CallSession, greeting: str) -> None:
    active = _agents.get(session.call_id)
    if active and not active.done():
        active.cancel()

    connected = asyncio.Event()
    logger.info(
        "VOICE WORKER STARTING | call={} | provider={} | pod={}",
        session.call_id,
        settings.voice_provider if settings.gemini_api_key else "local cascade",
        session.conversation_pod,
    )
    task = asyncio.create_task(run_voice_agent(session, greeting, on_connected=connected))
    _agents[session.call_id] = task
    task.add_done_callback(lambda done: _agents.pop(session.call_id, None) if _agents.get(session.call_id) is done else None)
    connected_wait = asyncio.create_task(connected.wait())
    try:
        done, _ = await asyncio.wait(
            {connected_wait, task},
            timeout=settings.voice_startup_seconds,
            return_when=asyncio.FIRST_COMPLETED,
        )
        if connected_wait in done and connected.is_set():
            logger.info("VOICE WORKER READY | call={} | client can join LiveKit", session.call_id)
            return
        if task in done:
            task.result()
            logger.error("VOICE WORKER EXITED BEFORE READY | call={}", session.call_id)
            raise RuntimeError("The voice agent stopped before it was ready.")
        logger.error("VOICE WORKER STARTUP TIMEOUT | call={} | timeout_seconds={}", session.call_id, settings.voice_startup_seconds)
        raise TimeoutError("Voice agent startup timed out")
    except Exception as exc:
        logger.exception("VOICE WORKER STARTUP FAILED | call={} | {}", session.call_id, exc)
        task.cancel()
        try:
            await task
        except (asyncio.CancelledError, Exception):
            pass
        _agents.pop(session.call_id, None)
        raise RuntimeError(
            "Could not start the voice agent. Check the Gemini Live API key/quota and LiveKit server, then retry."
        ) from exc
    finally:
        if not connected_wait.done():
            connected_wait.cancel()


async def stop_agent(call_id: str) -> None:
    task = _agents.pop(call_id, None)
    if task and not task.done():
        task.cancel()
        try:
            await task
        except (asyncio.CancelledError, Exception):
            pass


def make_participant_token(call_id: str) -> str:
    return participant_token(call_id, call_id)
