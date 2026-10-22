# Conversation pods

The backend selects a conversation pod from `SYNQ_CONVERSATION_POD` when each call starts. The frontend does not select or know about the pod. A pod owns its instruction addendum and turn handler; pods do not write session data to disk.

## Available pods

- `pod4` — the Wave brain (first built as Pod 6, now removed) that runs on our own voice pipeline instead of Gemini Live: speech-to-text, the Conductor, a text LLM and text-to-speech on every turn, each plug-and-play from `.env` (laptop Whisper/Piper with Sarvam fallback; Groq, Gemini or laptop Ollama for the LLM). Own blueprint in `blueprints/pod4_blueprint.yaml`. See `docs/pod4-architecture.md`.
- `pod5` — implements the Kismet AI call requirements with Gemini Live as the voice. A blueprint-driven Conductor (`blueprints/call_blueprint.yaml`) controls goals, slots, retries, timing, seeded detours and the safety path. See `docs/pod5-kismet-status.md`.

Pods 1, 2, 3 and 6 were removed on 2026-10-22 (owner's decision). Pod 4 is the one in use.

Inspect the registry with `GET /api/v1/conversation-pods`. The call status response reports the pod selected by the backend.

Set the backend's `.env` value before starting the service:

```dotenv
SYNQ_CONVERSATION_POD=pod4
```

Set it to `pod5` to use the Gemini Live call. The backend defaults to `pod4` when the setting is absent. Each pod keeps its own in-memory state, with its final state included in the opt-in call JSON export when `SYNQ_EXPORT_CALL_LOGS=true`.

Restart the backend after changing the setting, then start a new call. An active Gemini Live session has its system instruction fixed when its upstream session starts, so switching an in-progress call is not supported.

Pod 4 starts with its own thresholds (constants prefixed `POD4_` in `conversation_pod4.py`); record any changes to them here.
