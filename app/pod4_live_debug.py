"""TEMPORARY: show what Pod 4 does on each turn, live in the terminal. Nothing is saved.

Turn on with SYNQ_POD4_LIVE_DEBUG=1 when starting the backend. It prints, as the call happens:
  - when you start/stop speaking, and each transcript piece the STT returns;
  - the voice-cue measurements and the note sent to the LLM;
  - the observer LLM's raw answer and the parsed observation object;
  - the Conductor's decision object;
  - Aria's reply, and when she starts/stops speaking.

To remove later: delete this file and every line that calls `live_debug` (grep "live_debug").
"""

from __future__ import annotations

import json
import os
import sys
import time

ENABLED = os.getenv("SYNQ_POD4_LIVE_DEBUG", "").strip().lower() in ("1", "true", "yes", "on")

if ENABLED:
    # Keep the terminal readable: hide the usual DEBUG/INFO log lines (pipecat, Piper text,
    # LiveKit) so only this live view and real warnings/errors show.
    from loguru import logger

    logger.remove()
    logger.add(sys.stderr, level="WARNING")

_COLORS = {
    "you": "\033[38;5;117m",      # light blue: the caller
    "stt": "\033[38;5;153m",
    "cues": "\033[38;5;179m",     # amber: voice measurements
    "observe": "\033[38;5;150m",  # green: observer LLM
    "conductor": "\033[38;5;183m",  # violet: Conductor decision
    "aria": "\033[38;5;211m",     # pink: Aria
    "buddy": "\033[38;5;216m",    # peach: Buddy (its own voice, not Synthe's)
    "info": "\033[38;5;245m",
}
_RESET = "\033[0m"
_BOLD = "\033[1m"
_start = time.monotonic()


def _stamp() -> str:
    return f"{time.monotonic() - _start:8.2f}s"


def _label(kind: str) -> str:
    return "SYNTHE" if kind == "aria" else kind.upper()


def _out(text: str) -> None:
    sys.stdout.write(text + "\n")
    sys.stdout.flush()


def event(kind: str, text: str) -> None:
    """One line, e.g. event("you", "started speaking")."""
    if not ENABLED:
        return
    color = _COLORS.get(kind, "")
    _out(f"{_COLORS['info']}{_stamp()}{_RESET} {color}{_BOLD}{_label(kind):>9}{_RESET} {color}{text}{_RESET}")


def block(kind: str, title: str, data: object) -> None:
    """A titled, pretty-printed object (dict, list or pydantic model)."""
    if not ENABLED:
        return
    if hasattr(data, "model_dump"):
        data = data.model_dump(mode="json")  # type: ignore[union-attr]
    body = data if isinstance(data, str) else json.dumps(data, indent=2, ensure_ascii=False, default=str)
    color = _COLORS.get(kind, "")
    _out(f"{_COLORS['info']}{_stamp()}{_RESET} {color}{_BOLD}{_label(kind):>9}{_RESET} {color}{_BOLD}{title}{_RESET}")
    for line in body.splitlines():
        _out(f"{' ' * 19}{color}{line}{_RESET}")


def turn_divider(number: int) -> None:
    if not ENABLED:
        return
    _out(f"\n{_COLORS['info']}{'─' * 30} turn {number} {'─' * 30}{_RESET}")
