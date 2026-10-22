"""Pod 4 reply streaming: providers stream pieces, the Talker cuts them into sentences."""

from __future__ import annotations

import unittest
from dataclasses import replace
from unittest.mock import patch

from app import pod4_providers
from app.config import settings
from app.conversation import CallSession
from app.conversation_pod4 import Pod4Decision
from app.conversation_pod4_text import split_sentences, stream_talk


def decision() -> Pod4Decision:
    return Pod4Decision(response_mode="chat", ask_question=True, cue="chat", planning_reason="test")


def fake_stream(pieces: list[str], provider: str = "fake"):
    async def stream(messages, **_):
        for piece in pieces:
            yield piece, provider

    return stream


class StreamingTests(unittest.IsolatedAsyncioTestCase):
    def test_sentences_are_cut_as_they_finish(self) -> None:
        self.assertEqual(split_sentences("Oh. Pune is lovely. What took you there? And"), (["Oh. Pune is lovely.", "What took you there?"], "And"))
        self.assertEqual(split_sentences("Haan, theek hai. Mumbai mein kitne saal?"), (["Haan, theek hai."], "Mumbai mein kitne saal?"))
        self.assertEqual(split_sentences("Wait... really? I love that city", final=True), (["Wait... really? I love that city"], ""))
        self.assertEqual(split_sentences("Accha laga sunke। Aur batao?", final=True), (["Accha laga sunke।", "Aur batao?"], ""))

    async def test_provider_pieces_stream_from_a_thread(self) -> None:
        def generator(*_):
            yield "Hello "
            yield "there."

        with patch.dict(pod4_providers._STREAMS, {settings.pod4_llm: generator}):
            pieces = [piece async for piece in pod4_providers.stream_complete([], temperature=0, max_tokens=10)]
        self.assertEqual(pieces, [("Hello ", settings.pod4_llm), ("there.", settings.pod4_llm)])

    async def test_falls_back_only_before_the_first_piece(self) -> None:
        def broken(*_):
            raise RuntimeError("down")
            yield ""  # pragma: no cover  (makes this a generator)

        def working(*_):
            yield "ok."

        def breaks_midway(*_):
            yield "Half a "
            raise RuntimeError("connection dropped")

        with patch.object(pod4_providers, "settings", replace(settings, pod4_llm="primary", pod4_llm_fallback="backup")):
            with patch.dict(pod4_providers._STREAMS, {"primary": broken, "backup": working}):
                pieces = [piece async for piece in pod4_providers.stream_complete([], temperature=0, max_tokens=10)]
            self.assertEqual(pieces, [("ok.", "backup")])
            with patch.dict(pod4_providers._STREAMS, {"primary": breaks_midway, "backup": working}):
                pieces = [piece async for piece in pod4_providers.stream_complete([], temperature=0, max_tokens=10)]
            self.assertEqual(pieces, [("Half a ", "primary")])  # no switch once speech started

    async def test_talker_yields_sentences_and_records_the_reply(self) -> None:
        session = CallSession("stream-test", "english", conversation_pod="pod4")
        out: dict = {}
        pieces = ["Aria: Pune, ", "nice choice. What ", "took you there? ", "Work or ", "family?"]
        with patch.object(pod4_providers, "stream_complete", fake_stream(pieces)):
            sentences = [sentence async for sentence in stream_talk(session, decision(), "I'm from Pune", "", out)]
        self.assertEqual(sentences, ["Pune, nice choice.", "What took you there?", "Work or family?"])
        self.assertEqual(out["reply"], "Pune, nice choice. What took you there? Work or family?")
        self.assertEqual(out["provider"], "fake")
        self.assertIn("first_sentence_ms", out)
        self.assertEqual(session.transcript[-1]["content"], out["reply"])

    async def test_meta_commentary_stops_the_reply(self) -> None:
        session = CallSession("stream-meta", "english", conversation_pod="pod4")
        pieces = ["That sounds fun. ", "As the talker I should ask about work. ", "What do you do?"]
        with patch.object(pod4_providers, "stream_complete", fake_stream(pieces)):
            sentences = [sentence async for sentence in stream_talk(session, decision(), "I like hiking", "")]
        self.assertEqual(sentences, ["That sounds fun."])

    async def test_nothing_usable_gives_the_fallback_line(self) -> None:
        session = CallSession("stream-empty", "english", conversation_pod="pod4")

        async def down(messages, **_):
            raise RuntimeError("All Pod 4 LLM providers failed")
            yield  # pragma: no cover

        out: dict = {}
        with patch.object(pod4_providers, "stream_complete", down):
            sentences = [sentence async for sentence in stream_talk(session, decision(), "hello", "", out)]
        self.assertEqual(sentences, ["Sorry, I lost my thought for a second. What were you saying?"])
        self.assertEqual(out["provider"], "pod4-talker-fallback")

    async def test_stopping_early_still_records_what_was_said(self) -> None:
        session = CallSession("stream-cut", "english", conversation_pod="pod4")
        pieces = ["First thought here. ", "Second thought here. ", "Third one too."]
        with patch.object(pod4_providers, "stream_complete", fake_stream(pieces)):
            stream = stream_talk(session, decision(), "hi", "")
            first = await anext(stream)
            await stream.aclose()  # the caller cut in after one sentence
        self.assertEqual(first, "First thought here.")
        self.assertEqual(session.transcript[-1]["content"], "First thought here.")


if __name__ == "__main__":
    unittest.main()
