# Kismet AI Call: Conversation Requirements

Oct 17, 2026 · @ashish

## 1. Purpose and scope

The AI call is a casual 15 to 17 minute voice conversation, in English or Hinglish as the user chooses, that learns what a person wants from a relationship without feeling like a form or a robot. This document lists what the conversation must do, as numbered requirements, so Codex can build it and testers can check it. How it is built is in the architecture document "Kismet AI Call: Voice and Conversation Architecture".

Numbers marked *default* are starting values to tune, not tested ones. A requirement says **must** when it is required and **should** when it is a strong preference.

Priority order if requirements conflict: user safety, then honesty, then the user's own choices (skips, language), then coverage of topics, then time.

## 2. Language

The user chooses English or Hinglish before the call, and the whole call follows that choice.

| ID | Requirement | How to check |
| --- | --- | --- |
| LANG-1 | The call-intro screen must let the user pick English or Hinglish, and the choice is saved as `language_mode` | UI test; value present in call record |
| LANG-2 | Speech recognition, persona, voice, captions and app text must all follow `language_mode` | Run one call per mode; no mixed-up language |
| LANG-3 | In Hinglish mode the AI should mirror the user's Hindi-English mix | Review of 20 recorded calls |
| LANG-4 | Facts must be saved in the same English-normalized fields in both modes, with the user's original quote kept | Match test between an English user and a Hinglish user |
| LANG-5 | The language may change only between Part 1 and Part 2, from Settings, never in the middle of a question | UI test |
| LANG-6 | Each language needs its own persona file with the same warmth rules | File exists and passes the style checks for both |

## 3. Call length and time budget

The call lasts 15 to 17 minutes of talk time. This is an assumption to confirm: the Chemistry Lab games are not counted.

| ID | Requirement | How to check |
| --- | --- | --- |
| TIME-1 | The call must end by 1,020 seconds (17 minutes) with a warm close | Timer in every simulated call |
| TIME-2 | Wrap-up should begin at 840 seconds (14 minutes), *default*; no new detours or revisits after that | Log of wrap-up time |
| TIME-3 | The conversation follows fixed segments, *default* targets: warm-up 60 s, core 520 s, lab handoff 30 s, lab follow-up 330 s, close 60 s | Segment times in the call log |
| TIME-4 | Time targets are soft: a rich answer may run over, and the Conductor makes it up by dropping detours first | Behaviour in a slow-caller test |
| TIME-5 | After Part 1 the call must pause cleanly for the Chemistry Lab and resume in Part 2 with full memory of Part 1 | Pause and resume test |
| TIME-6 | The Talker prompt should stay small: last 6 turns plus a rolling summary refreshed every 5 turns, not the full transcript | Prompt size check at minute 15 |

## 4. Topics

The call covers goals, not fixed questions. The AI chooses the order and the wording; the goals and what must be learned from each are fixed in `call_blueprint.yaml`.

| Goal | Must cover | Skippable | Sensitive (no detours) |
| --- | --- | --- | --- |
| Weekend and downtime style | Yes | No | No |
| How they handle conflict | Yes | No | No |
| What they want from a partner | Yes | No | No |
| Family and future expectations | Yes | No | Yes |
| Past-relationship lessons (stated only) | No | Yes | Yes |
| Values and priorities | Yes | No | No |
| Communication and affection style | Yes | No | No |
| One real-life scenario | Yes | No | No |

The blueprint file so far lists weekend, conflict, family and past lessons in full; the other four follow the same shape, and their slots still need to be written.

| ID | Requirement | How to check |
| --- | --- | --- |
| TOP-1 | Every must-cover goal must be covered, or marked "not covered" with a reason, by the end of the call | Coverage report per call |
| TOP-2 | The order must adapt to what the user just said; a goal answered unprompted counts as covered | Simulated caller who volunteers answers early |
| TOP-3 | The wording of the same goal must differ between calls | Compare 20 calls |
| TOP-4 | Each goal has 4 to 6 angles (story, hypothetical, preference, reaction); an angle should not repeat within a call | Log of angles used |
| TOP-5 | Easy, warm goals should come first; sensitive goals only after rapport has built | Position of family and past-lessons goals in 20 calls |
| TOP-6 | Topics, timing and limits must come from the blueprint file, never be invented by the LLM | Code review; blueprint validated at server start |

## 5. Casual voice behaviour

The AI should sound like a warm friend who is curious, not like an interviewer.

| ID | Requirement | How to check |
| --- | --- | --- |
| VOICE-1 | Replies must be short and spoken: 1 to 2 sentences, about 25 words on average, no lists or headings | Transcript length check |
| VOICE-2 | The AI should react first, then ask: about 9 in 10 turns start with a reaction | Automated check on transcripts |
| VOICE-3 | The AI must not reuse an opening phrase from the last 10 turns | Automated check; zero repeats in a 17 minute call |
| VOICE-4 | The AI must stop speaking within 300 ms, *default*, when the user talks over it, and then listen | Logged on interruption |
| VOICE-5 | The AI should wait patiently when a sentence trails off ("...", "aur") and interrupt the user in under 5% of turns, *default* | Interrupt rate in test calls |
| VOICE-6 | Quiet backchannels ("hmm", "achha") should play during long answers, only on pauses over 700 ms, *default*, never over the user's words | Check against audio log |
| VOICE-7 | The AI should refer back to an earlier detail at least 3 times per call | Callback count |
| VOICE-8 | If the speech recognition is unsure, the AI asks once, casually ("Thoda dobara bologe?"), and never loops more than 2 attempts | Noisy-audio test |
| VOICE-9 | The AI must not give advice, labels or opinions about the user mid-call | Transcript review |
| VOICE-10 | A blind panel of 10 testers per language should rate the call 4 of 5 or higher for sounding natural | Listening test |

## 6. Off-topic detours and fillers

The call may drift into light, casual tangents and small filler reactions so it feels random, but code controls how often, and every detour comes back to the question.

| ID | Requirement | How to check |
| --- | --- | --- |
| DET-1 | At most 3 detours per call, each at most 2 turns, *default* | Count in simulated calls |
| DET-2 | A detour must not start before 90 seconds, *default*, and not in a goal marked sensitive (family, past lessons) | Check position and goal of each detour |
| DET-3 | Detours happen by a seeded roll, *default* 20% on each eligible turn, with types weighted: follow-up on something the user said, light tangent (food, music, travel, festivals, weather), playful what-if | Seeded replay gives the same plan |
| DET-4 | A detour must be skipped when the call is more than 45 seconds behind schedule, *default*, and never after wrap-up | Slow-caller test |
| DET-5 | After a detour the AI must return to the pending goal within 2 turns using a soft link, an honest return or a light joke; if the user keeps drifting, return once more, then ask the goal directly | Return check in transcripts |
| DET-6 | Facts stated in a detour are kept as incidental; they cover a must-cover goal only if every slot of it is filled | Fact check on test calls |
| FILL-1 | Filler type (agreement, small reaction, thinking aloud, callback, light aside) must be chosen by code by weight, and worded by the AI | Code review |
| FILL-2 | No filler type twice within 2 turns, and no filler phrase repeated within 10 turns, *default* | Automated check |
| FILL-3 | A light aside must never claim a human life, body or experiences | Transcript review |
| RAND-1 | Every random choice must use one seed per call, stored with the call, so a bad call can be replayed | Replay test |

## 7. Unanswered, partial, skipped and distress

Every answer is sorted into one of five outcomes. A goal is never forced: it is retried casually, then dropped.

| Outcome | What the AI does | Limit, *default* |
| --- | --- | --- |
| Full answer | Marks the goal covered and moves on | none |
| Partial answer | Comes back after 2 other goals, as a casual callback, and asks only for the missing part | 3 asks in total, then drops it |
| No answer or off-topic | Retries right away with a new angle and an easier form (give an example, pick one of two) | 2 asks in total, then drops it |
| User says skip | Skips now and never asks again | none |
| Distress | Stops the script and moves to the safety path | none |

| ID | Requirement | How to check |
| --- | --- | --- |
| RETRY-1 | Obvious phrases are classified by code first ("skip", "next", "baad mein" mean skip; "pata nahi" means no answer); a small fast LLM call classifies the rest as `{outcome, slots_filled}` | Unit tests on phrases |
| RETRY-2 | If the classifier is not sure, the answer counts as partial, so the AI asks less, not more | Unit test |
| RETRY-3 | A retry must never repeat the same words | Transcript check |
| RETRY-4 | A revisit must ask only for the missing parts | Transcript check |
| RETRY-5 | Follow-ups that deepen a rich answer (`max_followups`) and retries for missing answers are separate counters | Unit test |
| RETRY-6 | At wrap-up, an open must-cover goal gets one short last question; if that fails it stays "not covered" | Wrap-up test |
| RETRY-7 | A dropped goal must be stored as "not covered", shown on the Blueprint, never guessed; the user may fill it in later | Blueprint screen test |
| RETRY-8 | A skip must never be asked again, in this call or a resumed one | Resume test |

## 8. Safety, honesty and privacy

| ID | Requirement | How to check |
| --- | --- | --- |
| SAFE-1 | A safety check must run on every user turn in parallel, without slowing the reply | Latency log with the check on |
| SAFE-2 | On signs of self-harm or abuse, the script stops, the AI uses a calm fixed-intent reply, and the app shows support resources | Crisis simulated-caller test |
| SAFE-3 | The AI must say it is an AI at the start and answer truthfully if asked | Transcript check |
| SAFE-4 | The AI must not diagnose, label the user's personality, or give percentage scores | Transcript and Blueprint review |
| SAFE-5 | Only facts the user stated are saved as facts; inferences carry an evidence quote and stay unapproved until the user accepts them; rejected inferences are never shown again | Blueprint screen test |
| PRIV-1 | Raw audio must not be stored by default | Storage check |
| PRIV-2 | Transcripts are kept only for the user's own Blueprint and deleted on request | Delete-request test |
| PRIV-3 | Analytics events must carry no answers or transcripts | Event schema review |
| PRIV-4 | Call state must be saved after every turn so a dropped network or the Lab pause can resume without losing anything | Kill-and-resume test |

## 9. Providers: free now, paid later

The aim is to build on free or free-credit tools now and move to paid ones later without rewriting. Which provider is best for Hinglish is not known yet and must be decided by a benchmark on your own recordings.

| ID | Requirement | How to check |
| --- | --- | --- |
| PROV-1 | Speech-to-text, text-to-speech, the LLM and the live transport must each sit behind one adapter interface (`SttProvider`, `TtsProvider`, `LlmProvider`, `Transport`) | Code review |
| PROV-2 | One config file must choose the provider per tier: Tier 0 free and self-run, Tier 1 free credits, Tier 2 paid | Switch tier by config only |
| PROV-3 | Every provider must have a Fake that returns canned text or silence, so any phase can run without a live service | Tests run offline |
| PROV-4 | If the primary provider errors or is too slow, the call must fall back to the next one without the user noticing | Fault-injection test |
| PROV-5 | The call-logic code (the Conductor) must never import providers or make network calls | Import check in CI |
| PROV-6 | The voice pipeline must contain no conversation logic; it only moves audio in and out | Code review |
| PROV-7 | Feel of the voice must be judged on Tier 1 or Tier 2, not on local Tier 0 models | Test plan |
| PROV-8 | Cost per 17 minute call must be recorded in every test call, so a budget can be set after the benchmark | Cost log |

## 10. Acceptance tests

A phase is accepted when these pass. Simulated callers are LLMs that play different users: talkative, one-word, emotional, off-topic, rude, silent, half-answering.

| Test | Passes when | Covers |
| --- | --- | --- |
| Coverage run | In 100 simulated calls per language, every must-cover goal is covered or marked "not covered" | TOP-1, TOP-2, RETRY-7 |
| Variety run | 20 calls with different seeds give different goal order, wording and detours; no opener repeats | TOP-3, TOP-4, VOICE-3 |
| Detour run | No more than 3 detours, each back on topic within 2 turns, none in sensitive goals or after wrap-up | DET-1 to DET-6 |
| Silent and half-answer run | No goal asked more than its limit; skips never re-asked; dropped goals shown as not covered | RETRY-1 to RETRY-8 |
| Time run | Every call ends by 17 minutes, including slow callers | TIME-1 to TIME-4 |
| Crisis run | Safety path starts on every crisis script | SAFE-1, SAFE-2 |
| Replay run | The same seed and the same answers replay the same plan | RAND-1 |
| Offline run | The whole text brain passes using Fakes only | PROV-3, PROV-5 |
| Fallback run | Killing the primary provider mid-call does not end the call | PROV-4 |
| Pause and resume run | Part 1, Lab pause, Part 2 keep memory and skips | TIME-5, PRIV-4, RETRY-8 |
| Blind listening | 10 testers per language rate 4 of 5 or higher for natural | VOICE-10 |

The nightly simulated-caller run uses a fixed seed list, and any failure prints the seed so the call can be replayed.

## 11. Open questions for the owner

- [ ] Is 15 to 17 minutes the talk time only, or the whole journey including the Lab games? The segment times assume talk time only.
- [ ] Which goals belong to Part 1 and which to Part 2? Now all 8 are in Part 1, and Part 2 is built from the Lab answers.
- [ ] Should a user be able to change language between Part 1 and Part 2, or is it fixed for the whole call?
- [ ] Are the default numbers acceptable as a start (3 detours, 20% chance, 2 and 3 asks, 14 minute wrap-up), or do you want different ones?
- [ ] The slots for values, wants, affection and scenario goals are not written yet. Should they follow the same style as the first four?
- [ ] Provider choices (speech recognition, voice, LLM) stay open until the benchmark on your own Hinglish and English recordings.
