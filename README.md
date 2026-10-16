# Project Synq conversation and voice backend

The backend connects real mobile calls to Gemini Live's native speech-to-speech model. If Gemini Live is not configured or cannot start, the existing local call path is used:

```text
Android microphone ⇄ LiveKit WebRTC ⇄ Pipecat ⇄ Gemini Live (audio in/out, live transcript)
                                                    ↓ startup fallback
                           LiveKit ⇄ local Whisper → LangGraph + Ollama → Piper
```

No call audio is saved. By default, transcripts and call state stay in process memory and are cleared when a call ends or the API restarts. You can opt in to exporting transcript text and final conversation state as one JSON file per ended call.

## Local prerequisites

- Python 3.11 or newer
- A Gemini API key in `backend/.env` for the primary real-time audio conversation
- Ollama with `smollm2:360m` downloaded and running for local fallback and text-mode calls
- LiveKit Server running in development mode
- The Android phone and development computer on the same Wi-Fi network

Download the configured Ollama model before starting the backend:

```bash
ollama pull smollm2:360m
```

Install LiveKit Server using its official instructions, then start it so the phone can reach it:

```bash
livekit-server --dev --bind 0.0.0.0
```

Development mode uses `devkey` and `secret`. These are local-only credentials; do not reuse them outside your machine.

Install backend dependencies and run the API:

```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[voice]'
export SYNQ_LLM_PROVIDER=ollama
export SYNQ_OLLAMA_MODEL=smollm2:360m
export SYNQ_LLM_FALLBACK=gemini
# Set these in backend/.env. The backend loads that file; never put the key in the app.
# GEMINI_API_KEY='your-key-from-google-ai-studio'
# SYNQ_VOICE_PROVIDER=gemini_live
# SYNQ_GEMINI_LIVE_MODEL=gemini-3.8-live
# SYNQ_GEMINI_LIVE_VOICE=Kore
# Optional: save full transcript text and final state as JSON when each call ends (default: false).
# SYNQ_EXPORT_CALL_LOGS=true
# SYNQ_CALL_LOG_DIRECTORY=call_logs
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Gemini Live processes microphone audio and returns spoken audio and live transcripts. The API key stays in the backend `.env`. The app does not save call audio or transcripts to disk; captions and the recent conversation are kept in temporary backend memory and cleared when the call ends or the API restarts. Google handles API data according to the account's billing tier. On the free tier, data may be used to improve Google products, so use test answers and avoid sensitive details. If Gemini Live fails during setup, the call falls back to the local Whisper → Ollama → Piper path.

The first local fallback call downloads the Whisper and Piper models. Allow time and disk space for that startup. Defaults are Gemini Live `gemini-3.8-live` with voice `Kore`, Ollama `smollm2:360m`, Whisper `base`, and Piper `en_US-amy-medium`. Override with `SYNQ_GEMINI_LIVE_MODEL`, `SYNQ_GEMINI_LIVE_VOICE`, `SYNQ_OLLAMA_MODEL`, `SYNQ_VOICE_STT_MODEL`, and `SYNQ_VOICE_TTS_VOICE`.

When `SYNQ_EXPORT_CALL_LOGS=true`, the backend writes a call JSON file on `POST /api/v1/calls/{call_id}/end` to `backend/call_logs/` (or `SYNQ_CALL_LOG_DIRECTORY`). Filenames use Indian Standard Time, for example `call_2026-10-17_14-30-00_IST_<call-id>.json`. Each file includes the full available transcript and final LangGraph conversation state. These files may contain sensitive personal conversation content; keep the option off unless you need it and delete exports when finished reviewing them. Raw microphone audio is never written.

## Mobile development build

LiveKit uses native Android WebRTC code and does not work in Expo Go. In another terminal:

```bash
cd mobile
npx expo run:android
npx expo start --lan
```

Open the installed development build on the phone, allow microphone access, and use the profile call slider. Keep the API and LiveKit server running. The app uses the Expo LAN host to reach port 8000 and the LiveKit server on port 7880. If Expo cannot provide a LAN host, set `EXPO_PUBLIC_API_URL=http://<computer-lan-ip>:8000` before starting Metro.

## API endpoints

- `POST /api/v1/calls` creates a consented session, starts the Gemini Live Pipecat worker (or local fallback), and returns a room token. Gemini Live speaks the opening turn after the user joins.
- `POST /api/v1/calls/{call_id}/resume` restarts the worker for a minimized call and returns a fresh room token.
- `POST /api/v1/calls/{call_id}/turn` remains available for text harnesses.
- `POST /api/v1/calls/{call_id}/end` closes the session and clears its in-memory transcript.
- `GET /healthz` checks the API process.

The Gemini Live service returns native audio and both sides' speech transcripts. Its session uses context compression for longer calls. If Gemini Live cannot establish a session, the backend switches to local Whisper/Ollama/Piper and keeps the call on the same LiveKit room.

This is a local prototype. Before real users, add authentication, bounded call creation, robust session storage and retention controls, crisis escalation, provider monitoring, and production LiveKit credentials/TLS.
