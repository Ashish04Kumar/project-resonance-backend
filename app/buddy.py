"""Buddy chat: the companion umbrella, embedded in this backend, in two modes (chat and synth).

The companion, Umbrella (its own project and repo, `~/projects/umbrella`, Python package
`companion`) is installed into this backend's environment as a fixed version
(`pip install /home/bs/projects/umbrella`, later from its git repo at a tag); its code is
not part of this repo and is never edited from here. To pick up a newer companion, install
it again.

- Settings: the companion reads its own `umbrella/.env` (path: BUDDY_COMPANION_ENV).
  Those values are applied only while the companion is created, so this backend's own keys
  (GROQ_API_KEY, GEMINI_API_KEY, ...) are not changed. BUDDY_COMPANION_LLM and
  BUDDY_COMPANION_LLM_FALLBACK in backend/.env override the companion's LLM order for Buddy, and
  BUDDY_COMPANION_NAME its name (the companion calls itself "Umbrella"; in Resonance it is "Buddy").
- Two modes, each its own companion with its own tool list (the list the LLM is offered):
    chat  (default)  only MODE_TOOLS["chat"]: read memory (find_saved) and web_search, no saving.
    synth            every companion tool: saving, reminders, to-dos, lists, notes, clearing.
  Both companions use the same database, but each mode talks to its own companion user (MODE_USER,
  main thread): chat (shown as "Buddy") is "buddy", synth (shown as "Umbrella") is "umbrella", so
  their chat history and saved data are fully separate and show under those names in the playground. Two companions rather than one whose list changes per message, because the
  list is shared by all users talking at the same time. The companion has no per-message tool setting, so tools
  are removed here, from outside; test_buddy.py checks its registry still looks as expected.
- Each account's mode is kept in memory (back to chat when the backend restarts). It changes
  with POST /mode (the Chat / Synth toggle) or by saying "synth" / "chat mode" (mode_command).
- A Resonance account is known here as "resonance:<user id>" (its mode and its live call); its words go to
  the companion users "buddy" / "umbrella", shared by every account.

- POST /api/v1/buddy/chat  {text}  ->  a stream of JSON lines:
    {"type": "delta", "text"}            new reply text as it is written
    {"type": "expression", "expression"}  the reply's mood, for the robot's face
    {"type": "done", "text", "expression"}  the final reply: show this one (it can differ from the deltas)
    {"type": "error", "message"}
    {"type": "mode", "mode"}             the mode changed (a "synth" / "chat mode" keyword)
- GET /api/v1/buddy/mode  ->  {mode};  POST /api/v1/buddy/mode  {mode}
- POST /api/v1/buddy/live  {client_host?}  ->  {room, livekit_url, livekit_token}: a live voice chat
  with Buddy on Pod 4's voice pipeline (buddy_voice.py). One per account; starting again ends the old one.
- POST /api/v1/buddy/live/end

The companion's own prompt still describes its saving tools, so in chat mode, when asked to save
or change something, it answers with its guard's "Where should I save it?" offer. Those tools
are not there in chat mode, so that answer is replaced with NO_SAVING_REPLY (which points to synth).
"""

from __future__ import annotations

import asyncio
import json
import os
import re
from dataclasses import dataclass
from uuid import uuid4
from contextlib import aclosing, contextmanager
from pathlib import Path
from typing import Annotated, Any, AsyncIterator, Iterator, Literal

from dotenv import dotenv_values
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse
from loguru import logger
from pydantic import BaseModel, Field

from .auth import UserOut, current_user
from .config import settings
from .voice_agent import _token
from .voice_sessions import client_livekit_url

router = APIRouter(prefix="/api/v1/buddy", tags=["buddy"])

Mode = Literal["chat", "synth"]
MODES: tuple[Mode, ...] = ("chat", "synth")
DEFAULT_MODE: Mode = "chat"
# The tools each mode keeps (None = every companion tool).
MODE_TOOLS: dict[Mode, frozenset[str] | None] = {"chat": frozenset({"web_search", "find_saved"}), "synth": None}
CONVERSATION_TOOLS = MODE_TOOLS["chat"]
# Each mode talks to its own companion user, on that user's main thread (owner's decision, 21 Oct):
# chat (shown as "Buddy") is companion user "buddy", synth (shown as "Umbrella") is "umbrella". So the
# app's chats show up under those names in the companion playground, and each mode keeps its own chat
# history, memory, notes, reminders, to-dos and lists. Every Resonance account shares these two users
# (fine while only the owner uses the app); BUDDY_CHAT_USER / BUDDY_SYNTH_USER in backend/.env rename them.
MODE_USER: dict[Mode, str] = {"chat": os.getenv("BUDDY_CHAT_USER", "buddy"), "synth": os.getenv("BUDDY_SYNTH_USER", "umbrella")}
COMPANION_THREAD = "main"

# Default: the companion project next to this repo's folder (~/projects/umbrella/.env).
COMPANION_ENV = Path(os.getenv("BUDDY_COMPANION_ENV", str(Path(__file__).resolve().parents[2] / "umbrella" / ".env")))

# Said instead of the companion's "Where should I save it?" / "I haven't changed anything" offers.
NO_SAVING_REPLY = {
    "english": "In Buddy mode I can't save or change things. Switch to Umbrella mode (or say \"umbrella mode\") and I'll do it.",
    "hinglish": "Buddy mode mein main save ya change nahi kar sakta. Umbrella mode on karo (ya \"umbrella mode\" bolo), phir kar dunga.",
}
# Said (without the LLM) when a message only switches the mode.
MODE_ACK = {
    "synth": "Umbrella mode on. Ab main save, reminders, to-dos aur lists bhi sambhal sakta hoon.",
    "chat": "Buddy mode on. Bas baatein karte hain.",
}
# "umbrella mode" / "synth" (also as speech-to-text often writes it) switch to synth (shown as Umbrella);
# "buddy mode" / "chat mode" switch back to chat (shown as Buddy). A bare "buddy" or "umbrella" is just a word.
_SYNTH_WORD = re.compile(r"\b(?:umbrella mode|synth mode|sync mode|synth|sinth)\b[\s,.:;!?-]*", re.IGNORECASE)
_CHAT_WORD = re.compile(r"\b(?:buddy|chat|conversation) mode\b[\s,.:;!?-]*", re.IGNORECASE)


# Backend-side overrides of companion settings, from backend/.env (umbrella/.env is left
# as it is): BUDDY_COMPANION_LLM=gemini sets the companion's COMPANION_LLM for Buddy only.
BUDDY_OVERRIDES = {
    "BUDDY_COMPANION_LLM": "COMPANION_LLM",
    "BUDDY_COMPANION_LLM_FALLBACK": "COMPANION_LLM_FALLBACK",
    "BUDDY_COMPANION_NAME": "COMPANION_NAME",
}

_companions: dict[Mode, Any] = {}
_create_lock = asyncio.Lock()
_modes: dict[str, Mode] = {}  # companion user id -> mode


@contextmanager
def _companion_environment() -> Iterator[None]:
    """Apply umbrella/.env while the companion reads its settings, then restore."""
    values = {key: value for key, value in dotenv_values(COMPANION_ENV).items() if value is not None}
    for ours, theirs in BUDDY_OVERRIDES.items():
        if os.getenv(ours, "").strip():
            values[theirs] = os.environ[ours].strip()
    saved = {key: os.environ.get(key) for key in values}
    os.environ.update(values)
    try:
        yield
    finally:
        for key, old in saved.items():
            if old is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = old


def keep_tools(registry: Any, keep: frozenset[str] | None) -> list[str]:
    """Remove every tool not in `keep` from the companion's registry (in place); None keeps all.

    The companion's agent graph holds this same registry object, so the change applies to it.
    Returns the names that were removed.
    """
    if keep is None:
        return []
    removed = [name for name in list(registry.tools) if name not in keep]
    for name in removed:
        del registry.tools[name]
    return removed


def keep_conversation_tools(registry: Any) -> list[str]:
    return keep_tools(registry, CONVERSATION_TOOLS)


async def get_companion(mode: Mode = DEFAULT_MODE) -> Any:
    """The embedded companion for a mode, created on first use (database connection and settings)."""
    if mode in _companions:
        return _companions[mode]
    async with _create_lock:
        if mode not in _companions:
            from companion import Companion  # installed package; imported lazily so the API starts without it
            from companion.config import Settings

            if not COMPANION_ENV.is_file():
                raise RuntimeError(f"Companion settings not found at {COMPANION_ENV}")
            with _companion_environment():
                companion_settings = Settings()
            bot = await Companion.create(companion_settings)
            keep = MODE_TOOLS[mode]
            removed = keep_tools(bot.tools, keep)
            missing = sorted((keep or frozenset()) - set(bot.tools.names()))
            if missing:
                logger.warning("BUDDY | companion has no tool(s) {}", missing)
            logger.info(
                "BUDDY READY | mode={} | llm={} | tools={} | removed={}",
                mode, companion_settings.llm_order, len(bot.tools.names()), len(removed),
            )
            _companions[mode] = bot
    return _companions[mode]


async def close_companion() -> None:
    for user_id in list(_live):
        await stop_live(user_id)
    for mode in list(_companions):
        await _companions.pop(mode).close()


# --- modes -----------------------------------------------------------------------------------


def get_mode(user_id: str) -> Mode:
    return _modes.get(user_id, DEFAULT_MODE)


def set_mode(user_id: str, mode: Mode) -> None:
    _modes[user_id] = mode


def mode_command(text: str) -> tuple[Mode | None, str]:
    """A mode keyword in a message: (the mode it asks for, the message without it); (None, text) if none."""
    for mode, pattern in (("chat", _CHAT_WORD), ("synth", _SYNTH_WORD)):
        if pattern.search(text):
            return mode, pattern.sub(" ", text, count=1).strip(" ,.-")  # type: ignore[return-value]
    return None, text


@dataclass
class BuddyTurn:
    """What to do with one message: which companion answers what, or just acknowledge a mode switch."""

    mode: Mode
    text: str
    changed: bool
    ack: str = ""


def prepare_turn(user_id: str, text: str) -> BuddyTurn:
    """Apply a mode keyword in the message (if any) and say which mode answers the rest."""
    wanted, rest = mode_command(text)
    changed = wanted is not None and wanted != get_mode(user_id)
    if wanted is not None:
        set_mode(user_id, wanted)
    mode = get_mode(user_id)
    if wanted is not None and len(rest) < 2:  # only the keyword: acknowledge without the LLM
        return BuddyTurn(mode, "", True, MODE_ACK[mode])  # "changed": the phone re-syncs its toggle either way
    return BuddyTurn(mode, rest, changed)


def companion_user(user: UserOut) -> str:
    return f"resonance:{user.id}"


class BuddyChatIn(BaseModel):
    text: str = Field(min_length=1, max_length=2000)


def conversation_reply(reply_text: str, user_text: str, mode: Mode = DEFAULT_MODE) -> str:
    """The reply to show: in chat mode, the companion's save/change offers become NO_SAVING_REPLY."""
    from companion.guard import UNCHANGED_REPLY, UNSAVED_REPLY
    from companion.persona import is_english

    if mode == "chat" and reply_text.strip() in (UNSAVED_REPLY, UNCHANGED_REPLY):
        return NO_SAVING_REPLY["english" if is_english(user_text) else "hinglish"]
    return reply_text


def _line(event: dict[str, Any]) -> bytes:
    return (json.dumps(event, ensure_ascii=False) + "\n").encode("utf-8")


async def _reply_stream(user_id: str, text: str) -> AsyncIterator[bytes]:
    try:
        turn = prepare_turn(user_id, text)
        if turn.changed:
            yield _line({"type": "mode", "mode": turn.mode})
        if turn.ack:
            yield _line({"type": "done", "text": turn.ack, "expression": "happy"})
            return
        bot = await get_companion(turn.mode)
        text = turn.text
        # Chat mode: the companion's spoken style (1-3 short sentences, at most 70 words, no lists or
        # markdown) keeps typed replies crisp. Synth mode keeps the full text style for plans and lists.
        crisp = turn.mode == "chat"
        stream = bot.chat_stream(
            MODE_USER[turn.mode], text, spoken=crisp, mode_label=f"resonance buddy text ({turn.mode}, {user_id})",
            thread=COMPANION_THREAD,
        )
        async with aclosing(stream) as events:
            async for event in events:
                kind = event.get("type")
                if kind == "delta":
                    yield _line({"type": "delta", "text": event.get("text", "")})
                elif kind == "expression":
                    yield _line({"type": "expression", "expression": event.get("expression", "")})
                elif kind == "done":
                    reply = event["reply"]
                    yield _line({"type": "done", "text": conversation_reply(reply.text, text, turn.mode), "expression": reply.expression})
    except Exception:
        logger.exception("BUDDY CHAT FAILED | user={}", user_id)
        yield _line({"type": "error", "message": "Buddy couldn't reply right now. Please try again."})


@router.post("/chat")
async def buddy_chat(payload: BuddyChatIn, user: Annotated[UserOut, Depends(current_user)]) -> StreamingResponse:
    text = payload.text.strip()
    if not text:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, "Say something first.")
    return StreamingResponse(_reply_stream(companion_user(user), text), media_type="application/x-ndjson")


class BuddyModeIO(BaseModel):
    mode: Mode


@router.get("/mode", response_model=BuddyModeIO)
async def read_mode(user: Annotated[UserOut, Depends(current_user)]) -> BuddyModeIO:
    return BuddyModeIO(mode=get_mode(companion_user(user)))


@router.post("/mode", response_model=BuddyModeIO)
async def change_mode(payload: BuddyModeIO, user: Annotated[UserOut, Depends(current_user)]) -> BuddyModeIO:
    set_mode(companion_user(user), payload.mode)
    return payload


# --- live voice ------------------------------------------------------------------------------

# companion user id -> (LiveKit room, the running voice agent)
_live: dict[str, tuple[str, asyncio.Task[None]]] = {}


class BuddyLiveIn(BaseModel):
    client_host: str | None = None


class BuddyLiveOut(BaseModel):
    room: str
    livekit_url: str
    livekit_token: str


async def stop_live(user_id: str) -> None:
    entry = _live.pop(user_id, None)
    if entry and not entry[1].done():
        entry[1].cancel()
        try:
            await entry[1]
        except (asyncio.CancelledError, Exception):
            pass


@router.post("/live", response_model=BuddyLiveOut)
async def start_live(payload: BuddyLiveIn, user: Annotated[UserOut, Depends(current_user)]) -> BuddyLiveOut:
    from . import pod4_live_debug as live_debug
    from .buddy_voice import run_buddy_voice_agent

    user_id = companion_user(user)
    live_debug.event("info", f"BUDDY LIVE: phone asked to start ({user_id})")
    try:
        await get_companion(get_mode(user_id))  # created now, so a missing setup fails here, not mid-chat
        livekit_url = client_livekit_url(payload.client_host)
    except Exception as error:
        logger.exception("BUDDY LIVE CANNOT START | user={}", user_id)
        live_debug.event("info", f"BUDDY LIVE: cannot start: {error}")
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "Buddy's voice isn't available right now.") from error
    await stop_live(user_id)
    room = f"buddy-{uuid4().hex[:12]}"
    connected = asyncio.Event()
    task = asyncio.create_task(
        run_buddy_voice_agent(room, user_id, token=_token(room, f"buddy-{room[-8:]}"), on_connected=connected)
    )
    _live[user_id] = (room, task)
    task.add_done_callback(lambda done: _live.pop(user_id, None) if _live.get(user_id, (None, None))[1] is done else None)
    waiting = asyncio.create_task(connected.wait())
    finished, _ = await asyncio.wait({waiting, task}, timeout=settings.voice_startup_seconds, return_when=asyncio.FIRST_COMPLETED)
    waiting.cancel()
    if not connected.is_set():
        if task in finished and task.exception():
            logger.opt(exception=task.exception()).error("BUDDY LIVE FAILED TO START | room={}", room)
        await stop_live(user_id)
        live_debug.event("info", "BUDDY LIVE: voice agent did not connect to LiveKit")
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "Buddy's voice couldn't start. Please try again.")
    logger.info("BUDDY LIVE READY | room={} | user={}", room, user_id)
    live_debug.event("info", f"BUDDY LIVE: agent ready in {room}, waiting for the phone to join ({livekit_url})")
    return BuddyLiveOut(room=room, livekit_url=livekit_url, livekit_token=_token(room, f"member-{room[-8:]}"))


@router.post("/live/end", status_code=status.HTTP_204_NO_CONTENT)
async def end_live(user: Annotated[UserOut, Depends(current_user)]) -> None:
    from . import pod4_live_debug as live_debug

    live_debug.event("info", "BUDDY LIVE: phone ended the voice chat")
    await stop_live(companion_user(user))
