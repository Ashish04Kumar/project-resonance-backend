# AI cost estimate (21 Oct 2026)

Rough numbers to plan pricing. Usage is measured from our own code and call logs; prices are from the providers' pages on 21 Oct 2026 and will change. 1 USD ≈ ₹88 (assumption).

## Measured usage

| What | Measured |
| --- | --- |
| Synthe call | test calls (~10 min) had ~20 AI turns and ~3,000 spoken characters; a full 18-minute call ≈ 36 turns, ≈ 7,000 characters |
| Synthe LLM per turn | 2 calls: observer ≈ 1,170 + reply ≈ 1,580 input tokens (≈ 2,750), ≈ 160 output tokens |
| Synthe LLM per 18-min call | ≈ 99,000 input + ≈ 5,800 output tokens |
| Buddy turn (Buddy mode) | ≈ 2,250–2,300 input + 20–50 output tokens (grows with history; Umbrella mode ≈ 2,750, and a tool turn needs a second LLM call) |
| Buddy voice reply | ≈ 25 words ≈ 140 characters |

## Prices used

| Provider | Price |
| --- | --- |
| Groq `openai/gpt-oss-120b` | $0.15 in / $0.60 out per 1M tokens |
| Groq `openai/gpt-oss-20b` | $0.075 / $0.30 |
| Groq `qwen/qwen3.8-27b` (Buddy's first LLM today) | $0.80 / $4.00 |
| Gemini 3.5 Flash-Lite | $0.30 / $2.50 |
| Gemini 2.5 Flash-Lite | $0.10 / $0.40 |
| Groq Whisper large-v3-turbo | $0.04 per hour of audio |
| Sarvam TTS Bulbul v3 | ₹30 per 10,000 characters (pay as you go) |
| Sarvam STT | ₹0.49 per minute (pay as you go) |
| LiveKit Cloud | Ship $50/month with 150,000 WebRTC participant-minutes, then $0.0005/min; Scale $500/month with 1.5M |

## Cost per Synthe call (18 min)

| Part | Today's setup | Cost |
| --- | --- | --- |
| LLM | Groq gpt-oss-120b | ≈ ₹1.6 (Qwen fallback: ≈ ₹9) |
| TTS | Sarvam Bulbul v3, ≈ 7,000 chars | **≈ ₹21** |
| STT | laptop Whisper | ₹0 + server (Sarvam: ≈ ₹8.8; Groq Whisper: ≈ ₹0.5) |
| LiveKit | 36 participant-minutes | ≈ ₹1.6 (inside the Ship plan's free minutes) |
| **Total** | | **≈ ₹24 per call**, ~85% of it TTS |

With laptop/self-hosted Piper TTS instead of Sarvam: ≈ ₹3 per call plus server cost.

## Cost per Buddy turn

| LLM | Text turn | + Sarvam voice reply |
| --- | --- | --- |
| Qwen 3.8 27B (today) | ≈ ₹0.18 | ≈ ₹0.60 |
| gpt-oss-120b | ≈ ₹0.03 | ≈ ₹0.45 |
| gpt-oss-20b | ≈ ₹0.02 | ≈ ₹0.44 |
| Gemini 2.5 Flash-Lite | ≈ ₹0.02 | ≈ ₹0.44 |

## Monthly cost at scale

Assumption: each active user does 1 Synthe call, plus 20 typed and 5 voice Buddy turns a day for 30 days.

| | Per active user / month | 100 users | 1,000 users | 10,000 users |
| --- | --- | --- | --- | --- |
| Today's setup (Qwen first, Sarvam TTS) | ≈ ₹230 | ≈ ₹23,000 | ≈ ₹2.3 lakh | ≈ ₹23 lakh |
| Cheaper LLM (gpt-oss-120b), Sarvam TTS | ≈ ₹120 | ≈ ₹12,000 | ≈ ₹1.2 lakh | ≈ ₹12 lakh |
| Cheaper LLM + self-hosted Piper/Whisper | ≈ ₹35 + servers | ≈ ₹3,500 | ≈ ₹35,000 | ≈ ₹3.5 lakh |

Plus LiveKit's plan ($50 or $500 a month) and our own servers (backend, and GPUs for self-hosted speech at scale). These are estimates: heavy users, longer Buddy histories and tool turns cost more.

## Ways to cut cost (biggest first)

1. **TTS is the biggest bill.** Self-host Piper (or another open TTS) on our own server for most speech; keep Sarvam for premium or for Synthe only.
2. **Use the cheap LLM first for Buddy.** gpt-oss-120b is ~5× cheaper than Qwen 3.8 27B on Groq; gpt-oss-20b and Gemini 2.5 Flash-Lite are cheaper still.
3. **Cap Buddy.** Free daily limits (messages and voice minutes); more with a subscription. Voice is ~15× the cost of text.
4. **Halve the call's LLM cost.** Merge Pod 4's observer and reply into one LLM call; trim the prompt; use prompt caching for the part that repeats every turn.
5. **Self-host STT** (Whisper on a GPU server) once volume is steady; Groq Whisper is a cheap cloud option meanwhile.
6. **Measure real usage.** Log tokens, characters and minutes per user so this sheet becomes real numbers, and price the app on them (freemium: the first call free, Buddy voice and heavy use paid).

## Sources

- Groq models and prices: https://console.groq.com/docs/models
- Gemini API prices: https://ai.google.dev/gemini-api/docs/pricing
- Sarvam API prices: https://www.sarvam.ai/api-pricing
- LiveKit Cloud prices: https://livekit.com/pricing
