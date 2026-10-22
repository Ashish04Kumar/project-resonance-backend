# Pod 4: our own voice pipeline

Pod 4 runs the Wave conversation brain (first built as Pod 6, which was removed on 2026-10-22) without Gemini Live. Every caller turn goes through our own pipeline, and the Conductor plans each reply before it is written. Pod 5 (Gemini Live) is separate and unchanged.

## Flow of one call

```
Phone mic ──LiveKit──▶ Silero VAD ──▶ STT ──▶ Pod4TurnProcessor ──▶ TTS ──LiveKit──▶ Phone speaker
                                                   │
                         ┌─────────────────────────┴─────────────────────────┐
                         │ 1. observe: LLM reads the caller's words → JSON    │
                         │ 2. Conductor (pure code) updates state and plans   │
                         │ 3. talk: LLM writes Aria's reply from the guidance │
                         └────────────────────────────────────────────────────┘
```

- Whisper sends one transcript per speech segment, so the processor waits until the caller has been quiet for 1.2 s (and VAD says they stopped) and answers the whole utterance once.
- If the caller talks over Aria, her audio stops (barge-in). Use earphones so her own voice doesn't trigger it.
- After Aria speaks, a single silence nudge comes after 10 s (not in wrap-up, close or safety).
- At 17 minutes Aria says goodbye on her own; the call is 18 minutes at most.
- On signs of self-harm or abuse, the safety reply is spoken and the app gets the support card.

## Files

| File | What it does |
| --- | --- |
| `app/conversation_pod4.py` | The Conductor (from the old Pod 6): stages, skips and dodges, revive, backup topics, clock. Pure code, no providers. |
| `app/pod4_blueprint.py`, `blueprints/pod4_blueprint.yaml` | Pod 4's own copy of the Wave stages, branches and limits. |
| `app/conversation_pod4_text.py` | The turn engine: observe → Conductor → talk; opening, nudge and goodbye replies; the Talker prompt. |
| `app/pod4_providers.py` | Plug-and-play STT, TTS and LLM adapters with fallbacks. |
| `app/pod4_voice_agent.py` | The live call pipeline on LiveKit. |
| `test_pod4.py` | Unit tests (Conductor rules, provider fallback, turn order). |

## Choosing providers (.env)

| Part | Setting | Options | Default now |
| --- | --- | --- | --- |
| Speech-to-text | `SYNQ_POD4_STT`, `SYNQ_POD4_STT_FALLBACK` | `whisper` (laptop), `sarvam` (Saarika, cloud) | whisper, then sarvam |
| Text-to-speech | `SYNQ_POD4_TTS`, `SYNQ_POD4_TTS_FALLBACK` | `piper` (laptop), `sarvam` (Bulbul, cloud) | piper, then sarvam |
| LLM | `SYNQ_POD4_LLM`, `SYNQ_POD4_LLM_FALLBACK` (comma list) | `gemini`, `groq`, `groq_backup`, `ollama` | groq, then groq_backup, then ollama |

Other settings: `GROQ_API_KEY`, `SYNQ_POD4_GROQ_MODEL` (default `qwen/qwen3.8-27b`), `SYNQ_POD4_GROQ_BACKUP_MODEL` (default `openai/gpt-oss-20b`, its own rate limit), `SYNQ_POD4_GEMINI_API_KEY` (falls back to `GEMINI_API_KEY`), `SYNQ_POD4_OLLAMA_MODEL` (default `qwen3:4b`), `SYNQ_POD4_WHISPER_MODEL`, `SYNQ_POD4_PIPER_VOICE`, `SARVAM_API_KEY` and optional Sarvam model and voice overrides.

- STT and TTS switch to the fallback when the primary cannot start (missing key, library or model).
- The LLM falls back per request; a short Groq rate-limit pause is retried once first.
- Sarvam STT needs the `sarvamai` package, which is not installed yet. Install it and set `SARVAM_API_KEY` before choosing it. Sarvam TTS needs no extra package.

Restart the backend after changing `.env`. Select the pod with `SYNQ_CONVERSATION_POD=pod4`.

## Logs to watch

- `STARTING POD4 PIPELINE CALL`: which STT, LLM and TTS were picked.
- `POD4 OBSERVATION`: what was read from the caller's turn (outcome and slots).
- `POD4 PLANNER`: the Conductor's decision.
- `POD4 TURN … think_ms=`: time from the caller's last word being transcribed to the reply text.
- `POD4 LLM | … failed`: a provider failure and the fallback used.
- `POD4 SILENCE NUDGE`, `CALL CLOCK GOODBYE`.

## Privacy

With Groq or Gemini chosen, the caller's transcribed words go to that provider. With Sarvam chosen, call audio goes to Sarvam. Laptop Whisper, Piper and Ollama keep everything on the machine. Update the consent screen wording before real users.

## Known gaps

- Laptop Whisper (`base`) sometimes drops very short replies ("yeah", "next") and mishears Hinglish; a bigger model or Sarvam/Groq Whisper STT should help.
- Piper's voice is clearly synthetic and English-only; Sarvam Bulbul is the planned upgrade.
- Laptop Qwen (Ollama) on CPU can take longer than the 45 s timeout, so it is only a last resort.


## Voice cues (how the caller sounded)

Each caller turn on a voice call is also measured for *how* it was said (`app/voice_cues.py`), and a short note goes to the LLM with the words:

| Cue | Measured from |
| --- | --- |
| Pause before answering; started talking over Aria | VAD start/stop timing vs Aria's speech |
| Long pauses mid-answer (≥ 0.8 s) | gaps between VAD speech segments in the turn |
| Pace (words per minute) | transcript word count ÷ VAD speaking time |
| Softer/louder, trailed off at the end | loudness of the raw microphone audio (20 ms frames) |
| Flatter/livelier intonation | pitch movement (autocorrelation pitch, 75–400 Hz) |
| Filler sounds (um, hmm, uh) | transcript |
| Very short reply | ≤ 3 words or < 1.2 s of speech |

- **Provider-independent:** only VAD events, raw mic audio and the final transcript text are used, so swapping Whisper/Piper for Sarvam or another paid STT/TTS in `.env` needs no change here.
- **Relative to the caller:** pace, loudness and intonation are compared with the caller's own earlier turns in the call (from the third turn on); before that only the absolute cues (pauses, interruptions, fillers, short replies) are reported.
- **Only what stood out** is written, e.g. `took 4s before answering; spoke slower than their usual; voice trailed off at the end`. Normal turns send nothing.
- **Observer:** may use the note as a soft hint for `tone_cue`/`tone_confidence`; words always win, and it never changes outcome or slots.
- **Talker:** uses it only for tone and pacing (gentler and slower after hesitation, matching energy when lively), and is told never to mention voice, pauses, speed or name an emotion.
- **Cost:** no extra LLM call and no extra model; the maths runs in a worker thread per turn (milliseconds). Audio is kept only while the caller speaks (max 30 s per turn) and dropped right after measuring; nothing is stored. Each turn's numbers are logged as `POD4 VOICE CUES` for tuning.
- Text turns (the /turn harness) have no note.


## Reply streaming

The Talker's reply is streamed so Aria starts speaking before the whole reply is written:

1. The observer (JSON) and the Conductor run first, as before; they need the whole answer, so they are not streamed.
2. The Talker LLM is called with streaming on (`pod4_providers.stream_complete`: Groq and Gemini over server-sent events, Ollama line by line). It falls back to the next provider only if one fails before sending any text; once text has started, an error just ends the reply.
3. `conversation_pod4_text.stream_talk` cuts the stream into sentences (. ! ? । …; pieces under three words are joined to the next) and checks each one (no meta commentary, at most 70 words in total). Each sentence goes to TTS (`TTSSpeakFrame`) as soon as it is complete.
4. If the caller starts talking, the remaining sentences are dropped. Whatever was said is still recorded in the transcript, and the caption shows the full reply when it ends.
5. The greeting, silence nudge and goodbye are streamed the same way.

Logs: `POD4 TURN … plan_ms (observe + Conductor) | first_sentence_ms (caller's turn → first sentence sent to TTS) | total_ms`. The non-streamed `run_pod4_turn` remains for the text harness.


## Turn taking with a segment-based STT

Whisper returns one piece of text per stretch of speech, 1–3.5 s after it ends, so one answer arrives in pieces.

- **Pieces are merged, never cancelled.** Each finished stretch of speech (VAD segment) waits in a queue until its text arrives; a piece of text is matched to the first waiting segment long enough to hold its words (≤ 7 words/s), and earlier segments are taken to have given no text. One answer loop waits until the caller isn't speaking, the last piece is at least 0.6 s old, and no real segment is still waiting for its text (given up after 4 s). Then all pieces become one turn. A reply that is already being prepared is never cancelled; words that arrive meanwhile become the next turn. (Before, a late piece cancelled the reply in progress and the earlier words were lost.)
- **Made-up text is dropped** (`app/stt_guard.py`), using how much the caller actually spoke since the previous piece (VAD) and how loud it was (mic audio):
  - text that no waiting segment could hold, at most 4 words: dropped (no matching speech; e.g. 0.02 s of noise came back as "Good bye.");
  - from under 0.2 s of speech, at most 4 words: dropped (a blip);
  - "thanks for watching", "subscribe", subtitle credits: always dropped;
  - stock phrases ("good bye", "thank you", "bye", …) from under 0.8 s of speech or very quiet audio (below −45 dBFS): dropped. A clearly spoken "thank you" is kept.
  - Dropped pieces are logged as `POD4 STT PIECE DROPPED` with the reason.
- Like the voice cues, this uses only VAD events, raw mic audio and the transcript text, so it works with any STT provider.

- **No stall on sounds without words.** If the caller makes a sound but no usable text comes (a short "hmm", a cough), it counts as their turn and the silence nudge is armed again, so Aria speaks after 10 s instead of both sides waiting. (In a test call this used to freeze the call until Pipecat's 5-minute idle timeout.) The nudge is a plain timer, so Aria's audio stopping between sentences no longer prints warnings.

## No made-up personal stories

Aria is an AI: the Talker prompt forbids saying she saw, went, ate, tried or did anything, or mentioning friends or family. Because models still slip ("I actually just saw a wild bug today"), every reply sentence is also checked (`claims_human_life` in `conversation_pod4_text.py`) and dropped before it is spoken; the rest of the reply goes ahead. Dropped sentences are logged as `POD4 TALKER SENTENCE DROPPED`.


## Real-time speech events (VadTap)

Whisper transcribes inside its own frame handling, so VAD start/stop frames behind it reached the turn processor up to a second late; Aria then answered while the caller was still talking and replied twice to one answer. `VadTap` now sits between the VAD and the STT and hands VAD events and mic audio to the turn processor immediately (turn taking, barge-in, voice cues). Aria also waits until the caller has been quiet for 0.7 s by real-time VAD.

Words cut short by a VAD flicker are kept: each speech segment records how loud the mic was from 0.8 s before it started (the STT's pre-roll). If that was loud, real speech was going on, so the segment gets 0.8 s more room for words and isn't treated as a blip. Only short text from tiny segments in quiet audio is dropped.

Also: after a sound that gave no words, the voice-cue turn is discarded (so the next turn's pause and mid-answer pauses are right); silence-nudge replies drop any sentence that mentions the silence or waiting (and stay quiet if nothing is left); the Talker is told never to comment on words that look misheard (e.g. its own name heard as "area").


## Gemini as the LLM (2026-10-19)

`SYNQ_POD4_LLM=gemini` with `SYNQ_POD4_LLM_FALLBACK=groq,ollama`. Both Gemini calls (observer and streamed Talker) send a low thinking setting: `thinkingLevel` from `SYNQ_POD4_GEMINI_THINKING` (default `low`), or `thinkingBudget: 0` for Gemini 2.5 models, because thinking tokens are billed as output and add delay. If a model rejects the setting (HTTP 400), the call is repeated once without it. Thought parts in the stream are never spoken. Pick the model with `SYNQ_POD4_GEMINI_MODEL` (the default `gemini-3.8-flash` costs more than a Flash-Lite model).
