# Chemistry Call voice architecture

## Current call path

```text
Android microphone ⇄ LiveKit WebRTC ⇄ Pipecat ⇄ Gemini Live (`gemini-3.8-live`)
Android speaker    ⇄ LiveKit WebRTC ⇄ Pipecat ⇄ Gemini Live native audio
                                              └─ live input/output captions

Startup fallback: LiveKit ⇄ Pipecat ⇄ local Whisper ⇄ LangGraph + Ollama ⇄ Piper
```

The FastAPI backend creates a short-lived call session and LiveKit room token. A Pipecat worker joins the room. Gemini Live handles speech recognition, conversation and speech generation in one streaming audio-to-audio session; it also returns text captions for the temporary call log. Server-side voice activity detection and interruption handling are enabled. Context compression is enabled to support the product's 17-minute call window.

The Gemini Live Talker remains the single streaming audio model. After a user turn, Gemini calls the private `update_conversation_state` tool. Its arguments are validated with Pydantic schemas, then a local LangGraph Conductor merges the observation into temporary call state and returns a short private cue to Gemini. The schemas track conversational intent, a tentative tone cue, quoted topic evidence, explicit preferences, topic coverage, and question angles. The Conductor can defer topic exploration during small talk, answer requests, corrections, vulnerable disclosures, and skips; it spaces out topic openings and avoids recently used question angles. Evidence and preferences are stored only when their quoted text matches the latest user transcript. This adds a model tool-call step before a spoken reply, so its latency must be measured on device; it is not a second LLM inference. If tool calling fails, the Talker receives a generic safe cue and the audio turn continues.

Gemini affective dialog remains enabled to shape vocal expression. The Live API exposes voice selection but no direct speech-rate control, so pacing is guided by the prompt; changing the voice changes vocal character. Captions update the in-memory call transcript, and topic coverage is driven by the validated Conductor state rather than keyword matches in live captions. Pipecat interruptions are enabled so the user can barge in while Gemini is speaking. No call audio or transcript is written to disk. Ending the call clears the session; restarting the API clears all sessions.

Conversation behavior is selected through the backend Pod registry described in `docs/conversation-pods.md`. Pod 4 (the default) runs on our own STT → LLM → TTS pipeline (`docs/pod4-architecture.md`); Pod 5 runs on Gemini Live and uses Gemini's own speech signals, with one blueprint-driven silence nudge. Pod selection is fixed for the lifetime of a call.

## Fallback and settings

If Gemini Live is not configured, or cannot establish its service session during startup, the agent uses the local Whisper → LangGsraph/Ollama → Piper path. The configured Gemini API key is read by the backend from `.env` or the process environment and is never sent to the mobile app.

Defaults:

- Gemini Live voice model: `gemini-3.8-live`
- Gemini Live voice: `Kore`
- Local text fallback: Ollama `smollm2:360m`
- Local speech recognition fallback: Whisper `base`
- Local speech generation fallback: Piper `en_US-amy-medium`
- LiveKit development server: `ws://127.0.0.1:7880`

See `README.md` for setup and environment variables. Expo Go cannot load LiveKit's native Android WebRTC module; use the Android development build.

## Data and consent

Starting a call means the user has accepted the consent notice. In the primary path, live microphone audio is streamed through the backend to Google Gemini Live, which returns spoken audio and text transcriptions. Google processes that data under the API account's current billing tier and terms. Google AI Studio's free tier may use submitted content to improve products; test with non-sensitive content on that tier. In the local fallback, call audio is processed on the development computer. The app stores neither audio recordings nor transcripts on disk; the backend holds recent transcript and call state in memory only.

## Scope and limitations

This is a development prototype. It does not yet provide user authentication, durable call/session storage, production LiveKit credentials and TLS, automated session-resumption tokens, billing safeguards, call analytics, or a production consent/retention system. Gemini Live supports a limited duration per session; Pipecat's context compression addresses context size but does not by itself guarantee connection resumption. Long-call reliability and session renewal need device QA before production.
