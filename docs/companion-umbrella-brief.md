# Brief: Companion "umbrella" and the Buddy AI

This is a hand-over brief for a separate thread. The voice call, call screen and UI work stay in the original thread; everything about the umbrella, the Buddy AI, its backend and its voice tuning happens in the new one.

## What the owner wants

1. **An umbrella = a backend-only Python package** (working name `companion`). It holds the companion's "brain" and grows over time: agent loop, persona, LLM providers, tools, memory, knowledge base, voice (STT/TTS/VAD), perception events, and a shared list of expressions.
2. **Why a package:** no network hop in the middle, so no added delay. Whatever uses it installs it (`pip install`) and calls it directly.
3. **Two ways to use the same package:**
   - **Embedded (in code):** a Python project imports it and runs it in-process, with no server. Examples: the owner's Raspberry Pi pan-tilt robot, or a laptop bot.
   - **Plug-and-play (service):** the same package can also run as a server (e.g. WebSocket/HTTP), so any frontend can connect: the mobile Buddy, a laptop UI, anything.
4. **Frontends are not part of the umbrella.** Each "body" builds its own frontend:
   - **Mobile Buddy:** the Buddy tab in this app (`../resonance-native/src/features/buddy/`), which already has the 3D robot head with idle, listening, thinking, speaking and error expressions, tilt, haptics and a mic button.
   - **The owner's own robot:** its own screen, its own 3D model, its own frontend.
   - No separate frontend package is needed.
5. **Buddy is the first thing built under the umbrella.**
6. **Future body: a pan-tilt robot** with 2 servo motors, a camera, YOLO person detection that turns toward people, running on a Raspberry Pi. It should plug into the same brain.

## What Buddy should be

- **A personal companion chatbot**, separate from the Chemistry Call (it doesn't use the call pods).
- **Mostly a general, friendly chat.** Occasionally (at random, not every time, and with a cooldown) it gives its own point of view based on the relationship understanding from the user's calls.
- **Agentic:** it can use tools when needed, e.g. read the user's profile, read the call insights, search a knowledge base.
- **A knowledge base** it can draw on.
- **Expressions:** besides thinking, speaking and error, add 2–3 more facial expressions. Suggested: happy/smile, curious, concerned, playful (wink). The LLM returns an expression tag with each reply.
- **Memory:** context-window management and long-term memory. The owner is building a separate memory project (a common cloud "umbrella" memory) that will be integrated later. Design a clear memory interface now (remember / recall), with a simple local implementation (Postgres), so the cloud memory can later be swapped in as an adapter.
- **Voice tuning with the agent** also belongs in the new thread.

## What already exists in this repo to reuse (don't break)

- **Pod 4 voice pipeline (`app/`), provider-independent:**
  - STT, LLM and TTS providers with fallbacks (`pod4_providers.py`).
  - Reply streaming sentence by sentence (`stream_complete`, `conversation_pod4_text.stream_talk`).
  - Voice cues (`voice_cues.py`) and the STT guard (`stt_guard.py`).
  - Real-time VAD tap and a threaded Whisper (`pod4_voice_agent.py`).
  - Filters: no made-up personal stories, no silence mentions in nudges, misheard words.
- **Auth and profiles:** auth (`auth.py`), profiles and media (`profile_store.py`), discovery and sparks (`discovery.py`), Postgres migrations (`app/migrations/`).
- **Mobile Buddy robot:** `../resonance-native/src/features/buddy/` (robot, expressions, `robotState.ts` `EXPRESSIONS`), and haptics (`../resonance-native/src/services/haptics.ts`).
- **Docs:** `docs/pod4-architecture.md`, `../resonance-native/docs/screens/buddy.md`.

## Open decisions to ask the owner first

1. **Repo location:** a separate repo for the umbrella, or a folder in this project?
2. **First step for Buddy:** text chat first, or voice first?
3. **Storing call insights:** call insights are currently only kept in temporary memory (see the AGENTS.md decision log). For Buddy to use them, they must be saved per user, which needs updated consent wording. Is that approved?
4. **Knowledge base:** what content goes in it, or start without one?
5. **Naming and expressions:** the name of the package, and which extra expressions.

## Working rules (from AGENTS.md)

- Ask before adding any library.
- Work in small milestones: verify, summarise, then stop.
- Keys stay on the backend.
- Pods 5 and 6 stay unchanged.
- Add a dated line to the AGENTS.md decision log for each owner decision.
- The owner is new to React Native: explain big changes in 1–2 plain sentences first.
