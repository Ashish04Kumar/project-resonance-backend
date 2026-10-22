# Hybrid call: line bank + live reply

**Status:** proposal, not built · **Date:** 21 Oct 2026 · **Base:** Pod 4, the Synthe call (`docs/pod4-architecture.md`) · **Build as:** Pod 7 in a fork (`SYNQ_CONVERSATION_POD=pod7`)

---

## 1. The idea in one minute

Today, every sentence Synthe says is written by an LLM and turned into speech by cloud TTS, live, on every call.

But much of what Synthe says is the same on every call: the questions for each stage, moving to a new topic, short fillers ("Fair enough"), "Sorry, could you say that again?", the silence nudge, the goodbye. In our call logs, about **50–64% of Synthe's speech** did not use the caller's own words.

So:

- Write those lines **once**, in many versions, review them, and **record their audio once**. This is the **line bank**.
- On each turn, the **LLM still listens and decides**: it writes a short reaction to what the caller said, and **picks the best bank line** (or rejects them all and writes the whole reply itself).
- The call plan, timing and behaviour stay **exactly the same** as today.

**Why:** less live TTS (our biggest cost), fewer LLM tokens, and a faster first sound, without making the call feel scripted.

---

## 2. What stays exactly the same

The fork changes **only how a reply is produced**. Everything else is Pod 4, reused as is.

| Area | Same as today |
| --- | --- |
| Voice pipeline | LiveKit Cloud, VAD, VadTap, STT (with fallback), turn taking, STT guard, voice cues, barge-in |
| Brain | Observer (LLM, JSON) → Conductor (code) → reply. Same decision fields and response modes |
| Call plan | `pod4_blueprint.yaml`: the 15 stages in order, slots, angles, branches, revive lines, must-cover and sensitive stages, skip and dodge rules, detours, fillers, callbacks |
| Clock | 18 minutes at most, wrap-up at 15.5 min, goodbye at 17 min |
| Manners | Named Synthe, greets first, one silence nudge 10 s after a reply, every reply ends with one light question, never repeats a question, no made-up personal stories |
| Safety | Safety mode, the spoken safety reply, the support card on the phone |
| Outputs | Captions, transcript, call log JSON, live terminal view |

**Rule:** for the same caller words, the Conductor must make the same decisions as in Pod 4. Only the wording and where the audio comes from may change.

**Not in scope:** replacing the Conductor with a script, inserting the caller's words into recorded audio, or changing Buddy, Pods 1–6, the app or the companion umbrella.

---

## 3. Where the LLM's judgement sits

Nothing about understanding the caller is hard-coded.

| Step | Who | What it decides |
| --- | --- | --- |
| Understand the turn | **LLM** (observer, as today) | what they meant, tone, what they answered, distress, which branch fits |
| Plan the call | Conductor (code, as today) | which stage, retry or move on, skips, clock, safety |
| Reply | **LLM** (reflection + pick) | a short reaction to their actual words, and which bank line fits best, or none |
| Unusual turns | **LLM** (today's Talker) | their questions, corrections, small talk, anything the bank doesn't fit |
| Bank content | **LLM** offline + people | the LLM writes the versions, people approve them |

The bank only saves the cost of saying the predictable parts again. The LLM still chooses what is said.

---

## 4. One turn, step by step

1. The caller stops talking. STT and turn taking work as today.
2. **Observer** (LLM) reads the turn. New: it also says which blueprint branch fits, if any.
3. **Conductor** (code) decides the next move, as today.
4. **Selector** (code) finds 3–5 bank lines that match the decision.
5. **Reflection + pick** (one LLM call) writes a short reaction and picks one of those lines, or `none`.
6. **Composer** puts the reply together: reaction first, then the bank line.
7. **Speech:** the reaction goes to live TTS; the bank line plays from recorded audio.
8. If the bank has nothing, or the LLM picks `none`, today's Talker writes the whole reply (Pod 4 behaviour).

```
caller ─▶ Observer ─▶ Conductor ─▶ Selector ─▶ Reflection + pick ─▶ Composer
          (LLM)        (code)      (code)          (LLM)                │
                                                          live TTS ◀────┤ reaction
                                                    recorded audio ◀────┘ bank line
```

---

## 5. The line bank

A set of reviewed lines, stored as YAML in the repo (`line_bank/*.yaml`). Audio is stored outside git.

```yaml
- id: life_context.calling_from.new.03
  key: {mode: ask_goal, goal: life_context, angle: calling_from, ask_kind: new}
  lang: en              # en | hinglish
  tone: [neutral, warm]
  text: "So, where are you calling in from today?"
  is_question: true
  status: approved      # draft | approved | retired
```

**Which replies use the bank**

| Response mode | Bank key | Reply made from |
| --- | --- | --- |
| Greeting | `greeting` | bank |
| `ask_goal`, `retry`, `revisit`, `last_chance` | stage + angle + ask kind | reaction + bank question |
| `follow_up` | stage + branch | reaction + bank question |
| `revive`, silence nudge (revive or easier ask) | stage + nudge kind | bank (revive lines are in the blueprint) |
| `repair` | `repair` | bank |
| `respect_skip` | `respect_skip` | bank |
| `detour`, `detour_return` | detour type + theme, or return style | reaction + bank |
| Fillers | filler type | bank |
| `safety` | `safety` | bank only (reviewed wording) |
| `close` | `close` | reaction naming real things they shared + bank goodbye |
| `chat`, `answer_user`, `acknowledge_correction`, silence nudge (chat) | none | live, as today |

**Size:** at least 6 approved versions per key, in each language. About 1,500–2,500 lines in total.

**How lines are made (offline):**

1. An LLM writes draft versions for each key from the blueprint (stage, angle, branch, persona rules).
2. Automatic checks: length, exactly one question, no names or places, no duplicates, right language.
3. A person approves, edits or retires each line. Only `approved` lines are used.
4. The audio for every approved line is recorded (section 7).

---

## 6. Picking and putting the reply together

**Selector (code, no LLM)**

1. Keep lines with the right key, language and `approved` status.
2. Remove lines already used in this call and anything in `must_avoid`.
3. Score them: tone match, part of the call, and a small penalty for lines used often lately.
4. Send the top 3–5 to the LLM to pick from.
5. If the LLM is slow or fails, play the top-scored line, so the call never waits on the pick.
6. Log `BANK MISS` (nothing matched) and `BANK REJECTED` (the LLM picked `none`). These show where the bank needs more lines.

**Reflection + pick (one LLM call)**

- Returns JSON: `reflection` (0 or 1 sentence, at most 15 words, no question) and `pick` (a line id, or `none`).
- Short prompt: persona rules, the caller's last turn, the voice-cue note, the tone hint, the candidates. About 600 tokens, against about 1,580 for today's Talker.
- Same safety checks as today: no made-up personal stories, no meta talk, no comments on misheard words.

**Composer**

- Order: reaction, then the bank line. At most 3 sentences, one question, at the end.
- Drops the reaction if it asks a question or repeats the bank line.
- Writes the final text to captions, transcript and the call log, as today.

---

## 7. Audio: recording, storing, loading, playing

**Recording (offline)**

- Every approved line is recorded once for each **voice profile**: TTS provider, model, speaker, pace and language (today: Sarvam Bulbul v3, `priya`, pace 1.0).
- Format: 24 kHz mono, compressed (Opus), about 10–20 KB per line, so the whole bank is about 20–50 MB per voice.
- File name: `sha256(text + voice profile)`. Changing the voice means re-recording the bank, as an offline job.

**Storing**

- Development: a local folder on the backend.
- Production: object storage (for example S3 or GCS) as the source, with a copy on each voice server's disk. Never fetched over the network in the middle of a turn.

**Loading during a call**

| When | What is loaded into memory |
| --- | --- |
| Voice server starts | Shared lines for the active voice: greeting, repair, fillers, nudges, safety, goodbye (small, used on every call) |
| Call starts | The lines for the first stage |
| Each turn | The lines for the current stage and the next likely stage, in the background |

- Memory is small (a few MB per call), and lines from the server's disk load in well under 50 ms.
- If a line's audio is missing (not recorded yet, or a new voice), that line is spoken with live TTS instead. Logged as `AUDIO MISS`.

**Playing**

- Recorded audio is sent as audio frames into the **same pipeline output** as TTS, so barge-in, captions and the "Synthe is speaking" events work exactly the same.
- Live TTS uses the same voice profile as the recordings, so both parts of a reply sound like one voice.

---

## 8. Speed

Target time from the caller stopping to Synthe's first sound, compared with today:

| Step | Pod 4 today | Hybrid |
| --- | --- | --- |
| Turn taking + STT | same | same |
| Observer (LLM) | ≈ 0.7–2 s | same |
| Conductor + selector | ms | ms |
| Writing the reply | Talker until the first sentence ≈ 0.5–1.5 s | Reflection + pick ≈ 0.4–1 s (shorter prompt and answer) |
| First audio | live TTS of the first sentence | live TTS of the short reaction, or a recorded line at once if there is no reaction |

Rules that protect speed:

- The selector picks candidates while the LLM writes, never after.
- If the reaction isn't ready in time, the bank line starts on its own.
- **Pass rule:** the median time to first sound must not be slower than Pod 4 (section 11).

---

## 9. Cost (estimates)

From `docs/cost-estimate.md` and the Pod 4 call logs:

| Per 18-min call | Pod 4 today | Hybrid target |
| --- | --- | --- |
| Live TTS | ≈ 7,000 characters ≈ ₹21 | ≈ 2,500–3,500 characters ≈ ₹8–10 |
| LLM input per turn | ≈ 2,750 tokens | ≈ 1,800 tokens |
| One-time bank recording | – | ≈ 160,000 characters ≈ ₹480 per voice |

The 50–64% share is a rough estimate from four test calls. The real number comes from the tests in section 11.

---

## 10. Risks

| Risk | What we do |
| --- | --- |
| Sounds scripted or repetitive | Many versions per key, no line twice in a call, penalty for lines used often lately, human review, blind ratings |
| Reaction and bank line don't fit together | The LLM sees the candidates when it writes the reaction; the composer checks; otherwise the Talker writes the reply |
| The LLM rejects bank lines too often (cost goes back up) | Track `BANK REJECTED`; add or rewrite lines for those keys |
| Nothing in the bank for a rare situation | Talker fallback; `BANK MISS` report after each test round |
| Recorded and live voice don't match | One voice profile for both; re-record when the voice changes |
| Wrong branch picked | Use a branch only when the observer is sure; otherwise a general question for the angle |

---

## 11. How we test it

Pod 7 is compared with Pod 4 on the same inputs.

- **Automatic:** 20 scripted caller personas, played into LiveKit by the fake caller (Piper speech). Same personas for both pods.
- **People:** blind ratings of recorded calls: natural, warm, feels scripted (1–5).

| Metric | Pass |
| --- | --- |
| Same Conductor decisions as Pod 4 for the same words | 100% |
| Must-cover stages reached | not fewer than Pod 4 |
| Live TTS characters per call | at least 50% lower |
| LLM tokens per call | at least 30% lower |
| Median time to first sound | not slower than Pod 4 |
| Naturalness (people, average) | at most 0.3 below Pod 4 |
| Same line twice in one call | never |
| Turns with no bank line (miss or rejected) | under 10% |

---

## 12. Plan for the fork

| Step | Result |
| --- | --- |
| M0 | Fork; Pod 7 as a copy of Pod 4; baseline numbers from the fake caller |
| M1 | Bank format, line writer and review flow for 2 stages (`life_context`, `daily_life`) plus shared lines (greeting, repair, fillers, nudges, goodbye) |
| M2 | Selector, reflection + pick and composer in the text test harness (`/turn`), with unit tests and the same-decisions test |
| M3 | Audio recording, loading and playback in the live call; barge-in and captions checked |
| M4 | Fake-caller A/B against Pod 4, first ratings from people |
| M5 | Bank for all 15 stages; go / no-go on section 11 |

**New files in the fork**

| File | What it does |
| --- | --- |
| `app/conversation_pod7.py` | Copy of the Pod 4 Conductor, no logic changes |
| `app/conversation_pod7_text.py` | The turn: selector, reflection + pick, composer |
| `app/line_bank.py` | Load, check and select lines |
| `app/audio_cache.py` | Load recorded audio and send it as frames |
| `line_bank/*.yaml` | The bank |
| `scripts/build_line_bank.py`, `render_line_bank.py` | Write lines, record audio |
| `test_pod7.py` | Selector, composer, same-decisions test, fallbacks |

---

## 13. Open questions

1. First bank: English only, or English and Hinglish together?
2. Who reviews lines, and how many must be approved before M4?
3. Production storage for the audio (which cloud, and is the voice server's disk enough)?
4. Can the reaction be skipped on some turns (for example after one-word answers) to save more?
