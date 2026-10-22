# Pod 5: Kismet call requirements on Gemini Live

Pod 5 runs the Kismet AI call requirements (`docs/kismet-call-requirements.md`) with Gemini Live as the voice. It started as a clone of Pod 3 (removed on 2026-10-22). The Conductor was rebuilt around a blueprint file: code decides what to cover, when to retry, detour or wrap up; Gemini only decides how to say it.

Pod 4 is reserved for the future step-by-step pipeline (separate speech-to-text, LLM and text-to-speech providers). Pods 1–4 are unchanged.

## Files

| File | What it does |
| --- | --- |
| `blueprints/call_blueprint.yaml` | Goals, slots, angles, scenarios, limits, clock, detour and filler weights, safety contacts. Edit this to tune the call. |
| `app/pod5_blueprint.py` | Loads and validates the blueprint when the backend starts. |
| `app/pod5_rules.py` | Code-first phrase rules: skip, no answer, distress; opener extraction. |
| `app/conversation_pod5.py` | The Conductor: state, observation, decision, and the LangGraph merge-then-plan graph. Never imports a provider. |
| `app/conversation_pod5_text.py` | Text path for the local fallback and the `/turn` endpoint. |
| `test_pod5.py` | Unit tests and scripted simulated callers. |

## How one turn works

1. The user speaks. Gemini Live calls the private `update_conversation_state` tool with an observation: intent, the outcome of the pending question, every slot the user stated (exact quote plus a short English value), distress, and unclear audio.
2. Code checks the turn first: distress phrases, then skip and "pata nahi" phrases. Quotes that are not in the user's actual words are dropped.
3. The Conductor updates goal status and counters, then plans the next turn from the clock, the blueprint and the call's seed.
4. Gemini receives `private_talker_guidance` (mode, goal, missing parts, angle, filler type, callback, openers to avoid) and speaks.

Gemini Live sometimes replies before it calls the tool. The tool is declared `NON_BLOCKING`, and the result is sent `SILENT` if Gemini already replied to that caller turn (so it does not reply twice), `WHEN_IDLE` if it has not, and always `INTERRUPT` in safety mode. When Gemini replied first, the guidance applies to its next reply.

## Requirement status

**Done** means implemented in code and checked by unit tests. **Guided** means Gemini is told to do it but code cannot force or verify its speech. Nothing here has been tried in a live call yet.

| Area | Status |
| --- | --- |
| TOP-1 to TOP-6 | Done. 8 goals with slots; adaptive order, unprompted answers count, seeded order and angles, sensitive goals after rapport, all limits from the blueprint. Wording variety (TOP-3) is Gemini's. |
| RETRY-1 to RETRY-6 | Done. Five outcomes, code phrase rules first, unsure counts as partial, revisits ask only for missing slots, separate retry and follow-up counters, one last question at wrap-up. Fresh wording on retries is guided. |
| RETRY-7 | Backend done: the call export has a coverage report with "not covered" reasons. The Blueprint screen does not exist yet. |
| RETRY-8 | Done within a running backend (minimize and resume keep state). Not kept across a backend restart; see PRIV-4. |
| TIME-1, TIME-2, TIME-4 | Done. Close by 1,020 s, wrap-up at 840 s, behind-schedule check drops detours and follow-ups; ahead of schedule adds casual turns between goals. |
| TIME-3 | Partly. Warm-up, core, wrap-up and close are logged. Lab segments are in the blueprint but unused while `lab_enabled: false`. |
| TIME-5 | Not started. Needs the Chemistry Lab and Part 2. |
| TIME-6 | Partly. Gemini's start-up prompt gets the last 6 turns; Gemini Live keeps its own context during a session. |
| DET-1 to DET-6, FILL-1, FILL-2, RAND-1 | Done. Seeded detours (max 3, 2 turns, not before 90 s, not in sensitive goals or wrap-up, return within 2 turns), incidental detour facts, weighted filler types with no repeat within 2 turns, one stored seed per call. Filler wording is Gemini's. |
| FILL-3, VOICE-1, VOICE-2, VOICE-9, SAFE-3, SAFE-4 | Guided in the Pod 5 prompt. |
| VOICE-3 | Openers of the last 10 replies are tracked and sent as `must_avoid_openers`; repeats are counted in the state. Gemini Live audio cannot be retried after it is spoken. |
| VOICE-7 | Callbacks are scheduled every 4 user turns from the user's own quotes. Whether Gemini says them is not verified. |
| VOICE-8 | Done, using Gemini's own "couldn't hear" flag. Gemini Live gives no recognition confidence score. |
| Silence (from `Kismet-Silence-and-State-Loop.md`) | Simple version only: about 10–12 s after Aria finishes speaking (also after the greeting), she speaks once, re-asking a pending goal in an easier form or checking in on the caller's last words. Not in safety, wrap-up or close. The full ladder, mic check, pause and save, sensitive-goal multiplier and state callbacks are not built, and the re-ask is not counted as a retry. Timer: `SYNQ_CONVERSATION_SILENCE_NUDGE_SECONDS`. |
| VOICE-4, VOICE-5, VOICE-6 | Handled by Gemini Live and Pipecat, not by Pod 5. Backchannels while the user speaks are not possible with Gemini Live. |
| SAFE-1, SAFE-2 | Done: a code phrase check plus Gemini's distress flag on every turn; a fixed calm reply and a support card in the app. The check runs inside the tool call, not as a separate parallel service. Helpline numbers must be verified before launch. |
| SAFE-5 | Facts need exact quotes. Inferences and approval are not built yet. |
| PRIV-1 | No raw audio is stored. |
| PRIV-2, PRIV-4 | Not started. Needs the owner's approval to store call state and transcripts. |
| PRIV-3, PROV-1 to PROV-4, PROV-6 to PROV-8 | Not in Pod 5. These belong to the Pod 4 pipeline track. |
| PROV-5 | Done; a test checks that the Conductor imports no provider. |
| LANG-1 to LANG-6 | Language is chosen on the Language screen and passed to the call; English-normalized slot values are guided. LANG-5 waits for Part 2. |
| Acceptance tests | Scripted simulated callers run in the unit tests. LLM-played callers, blind listening and live calls are not done. |

## Run it

Set `SYNQ_CONVERSATION_POD=pod5` in `.env`, restart the backend, and start a new call. Each turn logs a `POD5 PLANNER` line with the seed, segment, mode, goal and statuses. With `SYNQ_EXPORT_CALL_LOGS=true`, the call export has `final_state.coverage_report` and `final_state.seed`.

Run the tests from ``:

```bash
.venv/bin/python -m unittest test_pod5 test_conversation_pods
```
