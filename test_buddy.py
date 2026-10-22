"""Buddy chat: the embedded companion keeps only conversation tools, its settings don't leak, and the stream works."""

from __future__ import annotations

import asyncio
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import asyncpg
import httpx

from app import auth, buddy, db
from app.main import app

TEST_DATABASE_URL = os.getenv("TEST_DATABASE_URL", "postgresql://bs@/synq_test?host=/var/run/postgresql")


class BuddyAdapterTests(unittest.IsolatedAsyncioTestCase):
    def test_companion_registry_still_has_the_tools_we_keep(self) -> None:
        """Catches a companion update that renames tools or changes how the registry stores them."""
        from companion.config import Settings
        from companion.tools.builtin import build_registry

        with buddy._companion_environment():
            registry = build_registry(Settings())
        self.assertIsInstance(registry.tools, dict)
        self.assertTrue(buddy.CONVERSATION_TOOLS <= set(registry.tools), sorted(registry.tools))
        removed = buddy.keep_conversation_tools(registry)
        self.assertTrue(removed)  # it had write tools (save, reminders, ...) before
        self.assertEqual(set(registry.names()), set(buddy.CONVERSATION_TOOLS))
        self.assertTrue(all(not registry.is_write(name) for name in registry.names()))

    def test_synth_keeps_every_tool_and_chat_only_reads(self) -> None:
        from companion.config import Settings
        from companion.tools.builtin import build_registry

        with buddy._companion_environment():
            synth, chat = build_registry(Settings()), build_registry(Settings())
        self.assertEqual(buddy.keep_tools(synth, buddy.MODE_TOOLS["synth"]), [])
        self.assertTrue({"reminder_add", "todo_add", "ltm_save", "note_save"} <= set(synth.names()))
        buddy.keep_tools(chat, buddy.MODE_TOOLS["chat"])
        self.assertEqual(set(chat.names()), {"find_saved", "web_search"})

    def test_mode_keywords(self) -> None:
        self.assertEqual(buddy.mode_command("synth"), ("synth", ""))
        self.assertEqual(buddy.mode_command("Synth, kal 9 baje mom ko call yaad dilana"), ("synth", "kal 9 baje mom ko call yaad dilana"))
        self.assertEqual(buddy.mode_command("Sync mode."), ("synth", ""))
        self.assertEqual(buddy.mode_command("chat mode please"), ("chat", "please"))
        self.assertEqual(buddy.mode_command("Umbrella mode"), ("synth", ""))
        self.assertEqual(buddy.mode_command("buddy mode, kaise ho"), ("chat", "kaise ho"))
        self.assertEqual(buddy.mode_command("Hi Buddy, umbrella le jaun?"), (None, "Hi Buddy, umbrella le jaun?"))
        self.assertEqual(buddy.mode_command("Synthe ka call kaisa tha?"), (None, "Synthe ka call kaisa tha?"))  # the call bot's name
        self.assertEqual(buddy.mode_command("let's chat about movies"), (None, "let's chat about movies"))

    def test_prepare_turn_switches_and_remembers_the_mode(self) -> None:
        buddy._modes.clear()
        self.assertEqual(buddy.get_mode("resonance:m1"), "chat")
        turn = buddy.prepare_turn("resonance:m1", "synth")
        self.assertEqual((turn.mode, turn.changed, turn.ack), ("synth", True, buddy.MODE_ACK["synth"]))
        turn = buddy.prepare_turn("resonance:m1", "kal ka plan kya hai?")
        self.assertEqual((turn.mode, turn.changed, turn.text, turn.ack), ("synth", False, "kal ka plan kya hai?", ""))
        self.assertEqual(buddy.get_mode("resonance:m2"), "chat")  # other users are not affected
        turn = buddy.prepare_turn("resonance:m1", "chat mode, kaise ho?")
        self.assertEqual((turn.mode, turn.changed, turn.text), ("chat", True, "kaise ho?"))

    def test_companion_settings_do_not_change_backend_keys(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            env = Path(folder) / ".env"
            env.write_text("GROQ_API_KEY=companion-key\nCOMPANION_ONLY_SETTING=1\n")
            with patch.object(buddy, "COMPANION_ENV", env), patch.dict(os.environ, {"GROQ_API_KEY": "backend-key"}):
                with buddy._companion_environment():
                    self.assertEqual(os.environ["GROQ_API_KEY"], "companion-key")
                self.assertEqual(os.environ["GROQ_API_KEY"], "backend-key")
                self.assertNotIn("COMPANION_ONLY_SETTING", os.environ)
            env.write_text("COMPANION_LLM=groq\n")
            with patch.object(buddy, "COMPANION_ENV", env), patch.dict(os.environ, {"BUDDY_COMPANION_LLM": "gemini"}):
                with buddy._companion_environment():
                    self.assertEqual(os.environ["COMPANION_LLM"], "gemini")  # backend override wins

    def test_companion_is_called_buddy_here(self) -> None:
        """The companion calls itself Umbrella; BUDDY_COMPANION_NAME makes it Buddy in Resonance."""
        from companion.config import Settings

        with tempfile.TemporaryDirectory() as folder:
            env = Path(folder) / ".env"
            env.write_text("COMPANION_NAME=Umbrella\n")
            with patch.object(buddy, "COMPANION_ENV", env), patch.dict(os.environ, {"BUDDY_COMPANION_NAME": "Buddy"}):
                with buddy._companion_environment():
                    self.assertEqual(Settings().name, "Buddy")
                self.assertNotIn("COMPANION_NAME", os.environ)

    async def test_stream_sends_text_mood_and_done(self) -> None:
        class FakeReply:
            text, expression = "Hey! Kaisa raha din?", "happy"

        class FakeBot:
            async def chat_stream(self, user_id, text, **kwargs):
                self.call = (user_id, text, kwargs)
                yield {"type": "delta", "text": "Hey! "}
                yield {"type": "expression", "expression": "happy"}
                yield {"type": "sentence", "text": "Hey!"}  # not passed on
                yield {"type": "done", "reply": FakeReply()}

        bot = FakeBot()
        with patch.dict(buddy._companions, {"chat": bot, "synth": bot}):
            lines = [json.loads(line) async for line in buddy._reply_stream("resonance:u1", "hi")]
        self.assertEqual([line["type"] for line in lines], ["delta", "expression", "done"])
        self.assertEqual(lines[-1], {"type": "done", "text": "Hey! Kaisa raha din?", "expression": "happy"})
        self.assertEqual(bot.call[:2], ("buddy", "hi"))   # chat mode talks to companion user "buddy"
        self.assertTrue(bot.call[2]["spoken"])  # chat mode: crisp (spoken-style) replies
        self.assertEqual(bot.call[2]["thread"], "main")

    async def test_save_offers_become_a_plain_answer(self) -> None:
        from companion.guard import UNSAVED_REPLY

        class FakeReply:
            text, expression = UNSAVED_REPLY, "curious"

        class FakeBot:
            async def chat_stream(self, user_id, text, **kwargs):
                yield {"type": "done", "reply": FakeReply()}

        with patch.dict(buddy._companions, {"chat": FakeBot(), "synth": FakeBot()}):
            lines = [json.loads(line) async for line in buddy._reply_stream("resonance:u1", "Remind me tomorrow at 9 to call mom")]
        self.assertEqual(lines[-1]["text"], buddy.NO_SAVING_REPLY["english"])
        self.assertEqual(buddy.conversation_reply("Theek hai!", "kal milte"), "Theek hai!")

    async def test_keyword_only_message_is_acknowledged_without_the_llm(self) -> None:
        buddy._modes.clear()
        lines = [json.loads(line) async for line in buddy._reply_stream("resonance:k1", "synth")]
        self.assertEqual(lines, [{"type": "mode", "mode": "synth"}, {"type": "done", "text": buddy.MODE_ACK["synth"], "expression": "happy"}])

    async def test_keyword_with_words_answers_them_in_the_new_mode(self) -> None:
        class FakeReply:
            text, expression = "Done, 9 baje yaad dila dunga.", "happy"

        class FakeBot:
            def __init__(self, name: str) -> None:
                self.name, self.calls, self.threads = name, [], []

            async def chat_stream(self, user_id, text, **kwargs):
                self.calls.append(text)
                self.threads.append((user_id, kwargs.get("thread")))
                yield {"type": "done", "reply": FakeReply()}

        chat, synth = FakeBot("chat"), FakeBot("synth")
        buddy._modes.clear()
        with patch.dict(buddy._companions, {"chat": chat, "synth": synth}):
            lines = [json.loads(line) async for line in buddy._reply_stream("resonance:k2", "synth, kal 9 baje yaad dilana")]
        self.assertEqual(lines[0], {"type": "mode", "mode": "synth"})
        self.assertEqual((chat.calls, synth.calls), ([], ["kal 9 baje yaad dilana"]))
        self.assertEqual(synth.threads, [("umbrella", "main")])  # synth is its own user: own history and memory

    async def test_failure_becomes_an_error_line(self) -> None:
        class BrokenBot:
            async def chat_stream(self, *args, **kwargs):
                raise RuntimeError("provider down")
                yield {}

        with patch.dict(buddy._companions, {"chat": BrokenBot(), "synth": BrokenBot()}):
            lines = [json.loads(line) async for line in buddy._reply_stream("resonance:u1", "hi")]
        self.assertEqual(lines[-1]["type"], "error")


class BuddyEndpointTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        try:
            pool = await asyncpg.create_pool(TEST_DATABASE_URL, min_size=1, max_size=2)
        except Exception as error:
            self.skipTest(f"test database unavailable: {error}")
        await db.migrate(pool)
        async with pool.acquire() as connection:
            await connection.execute("TRUNCATE users CASCADE")
        db._pool = pool
        auth._failures.clear()
        self.client = httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test")

    async def asyncTearDown(self) -> None:
        await self.client.aclose()
        await db.disconnect()

    async def test_signed_in_chat_streams_json_lines(self) -> None:
        class FakeReply:
            text, expression = "Hello!", "happy"

        class FakeBot:
            async def chat_stream(self, user_id, text, **kwargs):
                self.user_id = user_id
                yield {"type": "delta", "text": "Hello!"}
                yield {"type": "done", "reply": FakeReply()}

        self.assertEqual((await self.client.post("/api/v1/buddy/chat", json={"text": "hi"})).status_code, 401)
        signup = await self.client.post("/api/v1/auth/signup", json={"name": "Me", "email": "me@example.com", "password": "correct horse 1"})
        headers = {"Authorization": f"Bearer {signup.json()['access_token']}"}
        me = (await self.client.get("/api/v1/auth/me", headers=headers)).json()["id"]
        bot = FakeBot()
        with patch.dict(buddy._companions, {"chat": bot, "synth": bot}):
            response = await self.client.post("/api/v1/buddy/chat", json={"text": "hi"}, headers=headers)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["content-type"], "application/x-ndjson")
        lines = [json.loads(line) for line in response.text.splitlines()]
        self.assertEqual(lines[-1]["text"], "Hello!")
        self.assertTrue(me)
        self.assertEqual(bot.user_id, "buddy")   # every account's Buddy chat goes to companion user "buddy"

    async def test_mode_endpoint(self) -> None:
        self.assertEqual((await self.client.get("/api/v1/buddy/mode")).status_code, 401)
        signup = await self.client.post("/api/v1/auth/signup", json={"name": "Me", "email": "mode@example.com", "password": "correct horse 1"})
        headers = {"Authorization": f"Bearer {signup.json()['access_token']}"}
        buddy._modes.clear()
        self.assertEqual((await self.client.get("/api/v1/buddy/mode", headers=headers)).json(), {"mode": "chat"})
        self.assertEqual((await self.client.post("/api/v1/buddy/mode", json={"mode": "synth"}, headers=headers)).json(), {"mode": "synth"})
        self.assertEqual((await self.client.get("/api/v1/buddy/mode", headers=headers)).json(), {"mode": "synth"})
        self.assertEqual((await self.client.post("/api/v1/buddy/mode", json={"mode": "party"}, headers=headers)).status_code, 422)


class FakeTransport:
    def __init__(self) -> None:
        self.messages: list[dict] = []

    async def send_message(self, text: str) -> None:
        self.messages.append(json.loads(text))


class BuddyLiveVoiceTests(unittest.IsolatedAsyncioTestCase):
    """Buddy's live voice: Pod 4's turn processor, with the companion writing the replies."""

    def make(self, events: list[dict], cut_in_after: int | None = None):
        from app.buddy_voice import BuddyTurnProcessor

        class FakeBot:
            async def chat_stream(self, user_id, text="", **kwargs):
                self.call = (user_id, text, kwargs)
                for number, event in enumerate(events):
                    if cut_in_after is not None and number == cut_in_after:
                        turns._caller_cut_in = True  # the caller starts talking over Buddy here
                    yield event

        transport = FakeTransport()
        bot = FakeBot()
        patcher = patch.dict(buddy._companions, {"chat": bot, "synth": bot})
        patcher.start()
        self.addCleanup(patcher.stop)
        buddy._modes.clear()
        turns = BuddyTurnProcessor("buddy-test", "resonance:u1", transport)  # type: ignore[arg-type]
        spoken: list[str] = []

        async def push_frame(frame, direction=None):
            spoken.append(frame.text)

        turns.push_frame = push_frame  # type: ignore[method-assign]
        # Outside a running pipeline there is no pipecat task manager: run tasks on asyncio directly.
        turns.create_task = lambda coroutine, name=None: asyncio.get_running_loop().create_task(coroutine)  # type: ignore[method-assign]
        return turns, bot, transport, spoken

    async def test_each_sentence_is_spoken_with_mood_and_captions(self) -> None:
        events = [
            {"type": "expression", "expression": "happy"},
            {"type": "delta", "text": "Arre wah! "},
            {"type": "sentence", "text": "Arre wah!"},
            {"type": "sentence", "text": "Kaisa raha?"},
            {"type": "done", "reply": None},
        ]
        turns, bot, transport, spoken = self.make(events)
        await turns._handle_caller("aaj interview tha")
        self.assertEqual(spoken, ["Arre wah!", "Kaisa raha?"])
        self.assertEqual(bot.call[:2], ("buddy", "aaj interview tha"))
        self.assertTrue(bot.call[2]["spoken"])
        self.assertIn({"type": "buddy_expression", "expression": "happy"}, transport.messages)
        self.assertEqual(transport.messages[0], {"type": "call_caption", "role": "user", "text": "aaj interview tha"})
        self.assertEqual(transport.messages[-1], {"type": "call_caption", "role": "assistant", "text": "Arre wah! Kaisa raha?"})

    def test_markdown_is_not_read_aloud(self) -> None:
        from app.buddy_voice import speakable

        self.assertEqual(speakable("- Pick a dish like **Misal Pav** or *Puran Poli*."), "Pick a dish like Misal Pav or Puran Poli.")
        self.assertEqual(speakable("## Ideas"), "Ideas")
        self.assertEqual(speakable("Mid-week plans, 2-3 hours."), "Mid-week plans, 2-3 hours.")
        self.assertEqual(speakable("**"), "")
        self.assertEqual(speakable("Oh wow, huge congratulations! 🎉"), "Oh wow, huge congratulations!")
        self.assertEqual(speakable("Chai ☕️ ya coffee?"), "Chai ya coffee?")

    async def test_barge_in_drops_the_rest_of_the_reply(self) -> None:
        events = [{"type": "sentence", "text": "First."}, {"type": "sentence", "text": "Second."}, {"type": "sentence", "text": "Third."}]
        turns, _, transport, spoken = self.make(events, cut_in_after=1)
        await turns._handle_caller("hello")
        self.assertEqual(spoken, ["First."])
        self.assertEqual(transport.messages[-1]["text"], "First.")

    async def test_heard_words_go_to_the_phone_and_tiny_sounds_wait_less(self) -> None:
        from app.buddy_voice import SHORT_SPEECH_WAIT_SECONDS
        from app.stt_guard import Segment

        turns, _, transport, _ = self.make([])
        turns._segments.append(Segment(10.0, 1.4, -25.0, -25.0))
        turns._on_transcript("aaj office mein", 11.0)
        await asyncio.sleep(0)
        self.assertEqual(transport.messages[-1], {"type": "user_partial", "text": "aaj office mein"})
        turns._cancel_nudge()
        # A 0.25 s sound after the words: waited for only briefly, a real 1 s stretch for the full time.
        turns._segments = [Segment(12.0, 0.25, -30.0, -30.0)]
        self.assertTrue(turns._stt_still_working(12.0 + SHORT_SPEECH_WAIT_SECONDS - 0.1))
        self.assertFalse(turns._stt_still_working(12.0 + SHORT_SPEECH_WAIT_SECONDS + 0.1))
        turns._segments = [Segment(12.0, 1.0, -25.0, -25.0)]
        self.assertTrue(turns._stt_still_working(12.0 + SHORT_SPEECH_WAIT_SECONDS + 0.1))
        if turns._answer_task:
            turns._answer_task.cancel()

    async def test_saying_synth_in_voice_switches_mode_and_tells_the_phone(self) -> None:
        turns, bot, transport, spoken = self.make([{"type": "sentence", "text": "Should not be called."}])
        await turns._handle_caller("synth")
        self.assertEqual(buddy.get_mode("resonance:u1"), "synth")
        self.assertIn({"type": "buddy_mode", "mode": "synth"}, transport.messages)
        self.assertEqual(spoken, [buddy.MODE_ACK["synth"]])
        self.assertFalse(hasattr(bot, "call"))  # no LLM for a keyword-only turn

    async def test_chat_mode_voice_replaces_the_save_offer(self) -> None:
        from companion.guard import UNSAVED_REPLY

        turns, _, _, spoken = self.make([{"type": "sentence", "text": UNSAVED_REPLY.split("?")[0] + "?"}])
        await turns._handle_caller("Remind me tomorrow at 9")
        self.assertEqual(spoken, [buddy.NO_SAVING_REPLY["english"]])

    def test_buddy_tts_order_and_voices(self) -> None:
        from dataclasses import replace

        from app import buddy_voice

        piper_first = replace(buddy_voice.settings, buddy_tts="piper", buddy_tts_fallback="sarvam", buddy_piper_voice="en_US-ryan-medium")
        with patch.object(buddy_voice, "settings", piper_first):
            service, label = buddy_voice.make_buddy_tts()
        self.assertEqual(label, "piper en_US-ryan-medium")
        # Sarvam first without a key in backend/.env: falls back to Buddy's Piper voice.
        no_key = replace(piper_first, buddy_tts="sarvam", buddy_tts_fallback="piper", sarvam_api_key="")
        with patch.object(buddy_voice, "settings", no_key):
            _, label = buddy_voice.make_buddy_tts()
        self.assertEqual(label, "piper en_US-ryan-medium")
        with_key = replace(no_key, sarvam_api_key="test-key", buddy_sarvam_speaker="priya")
        with patch.object(buddy_voice, "settings", with_key):
            service, label = buddy_voice.make_buddy_tts()
        self.assertTrue(label.startswith("sarvam bulbul:v3 priya pace "))
        self.assertEqual(service._settings.voice, "priya")

    def test_mood_changes_sarvam_pace_a_little(self) -> None:
        from app.buddy_voice import mood_pace

        self.assertEqual(mood_pace(0.88, "happy"), 0.88)
        self.assertEqual(mood_pace(0.88, ""), 0.88)
        self.assertLess(mood_pace(0.88, "sad"), mood_pace(0.88, "concerned"))
        self.assertLess(mood_pace(0.88, "concerned"), 0.88)
        self.assertGreater(mood_pace(0.88, "excited"), 0.88)
        self.assertEqual(mood_pace(1.9, "excited"), 2.0)  # kept inside Sarvam's range

    async def test_reply_mood_sends_a_pace_update_only_for_sarvam(self) -> None:
        from pipecat.frames.frames import TTSUpdateSettingsFrame

        events = [{"type": "expression", "expression": "sad"}, {"type": "sentence", "text": "Oh, that hurts."}]
        turns, _, _, _ = self.make(events)
        frames: list = []

        async def push_frame(frame, direction=None):
            frames.append(frame)

        turns.push_frame = push_frame  # type: ignore[method-assign]
        await turns._handle_caller("my dog is sick")  # Piper: no pace change
        self.assertFalse(any(isinstance(frame, TTSUpdateSettingsFrame) for frame in frames))
        turns.sarvam_pace_base = turns._last_pace = 0.88
        frames.clear()
        await turns._handle_caller("my dog is sick")
        self.assertIsInstance(frames[0], TTSUpdateSettingsFrame)  # before the sentence
        self.assertEqual(frames[0].delta.pace, 0.79)
        frames.clear()
        await turns._handle_caller("still sad")  # same mood again: no repeat update
        self.assertFalse(any(isinstance(frame, TTSUpdateSettingsFrame) for frame in frames))

    async def test_greeting_and_nudge_are_cue_turns(self) -> None:
        from app.buddy_voice import GREETING_CUE, NUDGE_CUE

        turns, bot, _, spoken = self.make([{"type": "sentence", "text": "Hey!"}])
        await turns.greet()
        self.assertEqual((bot.call[1], bot.call[2]["cue"]), ("", GREETING_CUE))
        await turns._nudge_after_silence()
        self.assertEqual(bot.call[2]["cue"], NUDGE_CUE)
        self.assertTrue(turns._nudged_since_caller)  # only one nudge until the caller speaks again
        self.assertEqual(spoken, ["Hey!", "Hey!"])


if __name__ == "__main__":
    unittest.main()
