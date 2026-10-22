# Backend docs

- Conversation pods (which call brain runs) — conversation-pods.md
- Pod 4, the Synthe call on our own voice pipeline (in use) — pod4-architecture.md
- Question map that Pod 4's blueprint follows — pod6-question-map.md
- Pod 5, Kismet requirements on Gemini Live — pod5-kismet-status.md, kismet-call-requirements.md
- Voice call transport (LiveKit, Pipecat, Gemini Live path) — voice-architecture.md
- Hybrid call proposal (line bank + live reply, not built) — hybrid-call-design.md
- AI cost estimate — cost-estimate.md
- Hand-over brief for the Umbrella companion and Buddy — companion-umbrella-brief.md
- Voice samples (Sarvam, Piper) — samples/

App screens and the product overview live in the app project: `../resonance-native/docs/`.

Before building a feature, write its goals, flow, edge cases and acceptance checks in a focused file here, link it above, and update `AGENTS.md` if it changes shared rules.
