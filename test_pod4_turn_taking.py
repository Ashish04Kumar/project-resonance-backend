"""Pod 4 turn taking: STT pieces are matched to speech segments and merged into one turn
(never cancelled), made-up text is dropped, and speech with no words never stalls the call."""

from __future__ import annotations

import unittest
from unittest.mock import patch

from pipecat.frames.frames import VADUserStartedSpeakingFrame, VADUserStoppedSpeakingFrame

from app import pod4_providers
from app.conversation import CallSession
from app.conversation_pod4 import Pod4Decision
from app.conversation_pod4_text import claims_human_life, mentions_silence, special_decision, stream_talk
from app.pod4_voice_agent import (
    END_SILENCE_SECONDS,
    STREAM_FLUSH_WAIT_SECONDS,
    STT_WAIT_MAX_SECONDS,
    UTTERANCE_GAP_SECONDS,
    Pod4TurnProcessor,
    VadTap,
    stt_streams,
)
from app.stt_guard import Segment, check_piece, check_streamed, check_with_segments


class SttGuardTests(unittest.TestCase):
    def test_blips_and_made_up_phrases_are_dropped(self) -> None:
        self.assertFalse(check_piece("Good bye.", 0.02, -40)[0])
        self.assertFalse(check_piece("Thanks for watching!", 3.0, -20)[0])
        self.assertFalse(check_piece("Thank you.", 0.5, -25)[0])  # stock phrase, very short
        self.assertFalse(check_piece("Thank you.", 1.2, -55)[0])  # stock phrase, very quiet
        self.assertFalse(check_piece("  ...  ", 2.0, -20)[0])

    def test_real_speech_is_kept(self) -> None:
        self.assertTrue(check_piece("I am calling from Ahmedabad", 1.1, -28)[0])
        self.assertTrue(check_piece("Thank you, that means a lot", 1.6, -25)[0])
        self.assertTrue(check_piece("Thank you.", 1.0, -25)[0])
        self.assertTrue(check_piece("hmm", 0.28, -25)[0])  # a short real "hmm" (the stalled call)

    def test_real_words_in_a_vad_flicker_are_kept(self) -> None:
        # VAD dropped out for 0.1 s mid-sentence, but the mic was loud: the STT heard real words.
        flicker = Segment(490.13, 0.11, -24.0, -22.0)
        keep, _, segment, _ = check_with_segments("I've been working on", [flicker])
        self.assertTrue(keep)
        self.assertIs(segment, flicker)
        # The same tiny segment in a quiet room is a click, and short text from it is dropped.
        click = Segment(20.07, 0.02, -48.0, -46.0)
        self.assertFalse(check_with_segments("Good bye.", [click])[0])

    def test_text_is_matched_to_the_segment_it_came_from(self) -> None:
        real, blip = Segment(18.96, 1.1, -25, -25), Segment(20.07, 0.02, -48, -46)
        # "Good bye." can't come from 0.02 s of sound: no matching speech, dropped.
        keep, _, segment, used = check_with_segments("Good bye.", [blip])
        self.assertEqual((keep, segment, used), (False, None, 0))
        # A segment that gave no text is skipped; the words belong to the next one that can hold them.
        keep, _, segment, used = check_with_segments("I moved here for work last year", [blip, real])
        self.assertEqual((keep, segment, used), (True, real, 2))


class TurnTakingTests(unittest.IsolatedAsyncioTestCase):
    def processor(self) -> Pod4TurnProcessor:
        return Pod4TurnProcessor(CallSession("turns-test", "english", conversation_pod="pod4"), transport=None)  # type: ignore[arg-type]

    def speak(self, turns: Pod4TurnProcessor, start: float, end: float) -> None:
        """What VadTap records when VAD reports speech from start to end."""
        turns._segments.append(Segment(end, end - start, -25.0, -25.0))
        turns._last_speech_end = end

    async def test_long_answer_in_pieces_stays_one_turn(self) -> None:
        # The split from the 12-minute test: two bits of a ramble ended before the first text came back.
        turns = self.processor()
        with patch.object(Pod4TurnProcessor, "_ensure_answer_loop"):
            self.speak(turns, 10.0, 13.0)
            self.speak(turns, 13.4, 17.0)  # the second bit ends before the first text arrives
            turns._on_transcript("I'm a software engineer doing backend for a fintech startup", 14.5)
            # The second bit is still waiting for its text: don't answer yet.
            self.assertFalse(turns._ready_to_answer(14.5 + UTTERANCE_GAP_SECONDS + 1.0))
            turns._on_transcript("the city is okay but traffic is a nightmare", 18.2)
            self.assertTrue(turns._ready_to_answer(18.2 + max(UTTERANCE_GAP_SECONDS, END_SILENCE_SECONDS) + 0.01))
        self.assertEqual(len(turns._pending), 2)

    async def test_a_blip_does_not_add_text(self) -> None:
        turns = self.processor()
        with patch.object(Pod4TurnProcessor, "_ensure_answer_loop"):
            self.speak(turns, 17.86, 18.96)
            turns._on_transcript("I am calling from Ahmedabad", 20.02)
            turns._segments.append(Segment(20.07, 0.02, -48.0, -46.0))  # a quiet click, no speech around it
            turns._on_transcript("Good bye.", 23.69)
        self.assertEqual(turns._pending, ["I am calling from Ahmedabad"])

    async def test_text_that_never_comes_stops_blocking_after_a_while(self) -> None:
        turns = self.processor()
        with patch.object(Pod4TurnProcessor, "_ensure_answer_loop"):
            self.speak(turns, 0.0, 1.5)
            turns._on_transcript("hello there Aria", 2.0)
            self.speak(turns, 2.5, 3.5)  # STT returns nothing for this bit
        self.assertFalse(turns._ready_to_answer(4.0))
        self.assertTrue(turns._ready_to_answer(3.5 + STT_WAIT_MAX_SECONDS + 0.01))

    async def test_words_during_a_reply_wait_for_the_next_turn(self) -> None:
        turns = self.processor()
        answered: list[str] = []

        async def fake_handle(text, cues=None):
            answered.append(text)
            if len(answered) == 1:
                self.speak(turns, 30.0, 31.5)
                turns._on_transcript("and also I love trekking", 31.6)

        with patch.object(Pod4TurnProcessor, "_ensure_answer_loop"), patch.object(turns, "_handle_caller", fake_handle):
            self.speak(turns, 10.0, 12.0)
            turns._on_transcript("I'm from Pune", 12.5)
            await turns._answer_pending()
            self.assertEqual(turns._pending, ["and also I love trekking"])
            await turns._answer_pending()
        self.assertEqual(answered, ["I'm from Pune", "and also I love trekking"])

    async def test_speech_with_no_words_re_arms_the_silence_nudge(self) -> None:
        # The stall from the 12-minute test: a short "hmm" gave no text and nobody spoke again.
        turns = self.processor()
        turns._nudged_since_caller = True
        armed: list[bool] = []
        with patch("app.pod4_voice_agent.asyncio.sleep", return_value=None), patch.object(
            Pod4TurnProcessor, "_start_nudge_timer", lambda self: armed.append(True)
        ):
            await turns._answer_loop()  # nothing pending, nothing waiting
        self.assertEqual(armed, [True])
        self.assertFalse(turns._nudged_since_caller)

    async def test_vad_tap_reports_speech_before_the_stt(self) -> None:
        turns = self.processor()
        turns._ensure_answer_loop = lambda: None  # type: ignore[method-assign]
        tap = VadTap(turns)
        with patch.object(VadTap, "push_frame"), patch("app.pod4_voice_agent.time.monotonic", side_effect=[100.0, 101.25]):
            await tap.process_frame(VADUserStartedSpeakingFrame(), None)  # type: ignore[arg-type]
            self.assertTrue(turns._user_speaking)  # known at once, not after the STT
            await tap.process_frame(VADUserStoppedSpeakingFrame(), None)  # type: ignore[arg-type]
        self.assertEqual(len(turns._segments), 1)
        self.assertAlmostEqual(turns._segments[0].seconds, 1.25)

    async def test_not_ready_while_they_are_still_talking_or_just_stopped(self) -> None:
        turns = self.processor()
        with patch.object(Pod4TurnProcessor, "_ensure_answer_loop"):
            self.speak(turns, 10.0, 12.0)
            turns._on_transcript("I feel this one is a bit too deep", 12.8)
            turns._user_speaking = True  # they carried on ("let's skip it")
            self.assertFalse(turns._ready_to_answer(14.0))
            turns._user_speaking = False
            turns._last_speech_end = 14.0
            self.assertFalse(turns._ready_to_answer(14.0 + END_SILENCE_SECONDS - 0.1))


class StreamingSttTests(unittest.IsolatedAsyncioTestCase):
    """A streaming STT (Sarvam) finalises text at its own pauses, often while the caller is still
    talking, and again when pipecat flushes it on our VAD stop."""

    def processor(self, streaming: bool = True) -> Pod4TurnProcessor:
        turns = Pod4TurnProcessor(CallSession("stream-test", "english", conversation_pod="pod4"), transport=None)  # type: ignore[arg-type]
        turns.streaming_stt = streaming
        return turns

    def start(self, turns: Pod4TurnProcessor, at: float) -> None:
        turns._user_speaking = True
        turns._segment_started_at = at

    def stop(self, turns: Pod4TurnProcessor, at: float) -> None:
        turns.on_user_stopped(at)

    def test_only_segment_based_stts_are_matched_per_segment(self) -> None:
        from pipecat.services.sarvam.stt import SarvamSTTService

        whisper = pod4_providers._whisper_service_class()
        self.assertFalse(stt_streams(whisper.__new__(whisper)))
        self.assertTrue(stt_streams(SarvamSTTService.__new__(SarvamSTTService)))

    def test_guard_checks_a_piece_against_all_recent_speech(self) -> None:
        recent = [Segment(11.0, 1.0, -25.0, -25.0), Segment(12.5, 1.2, -26.0, -30.0)]
        self.assertTrue(check_streamed("day at", recent)[0])  # a short tail of real speech
        self.assertFalse(check_streamed("Thank you.", [])[0])  # nobody spoke: made up
        self.assertFalse(check_streamed("Good bye.", [Segment(20.07, 0.02, -48.0, -46.0)])[0])  # a click
        self.assertFalse(check_streamed("Thanks for watching!", recent)[0])

    async def test_text_while_still_talking_is_kept(self) -> None:
        # "Hi Buddy" came back before our VAD had stopped: with no finished segment it used to be dropped.
        turns = self.processor()
        with patch.object(Pod4TurnProcessor, "_ensure_answer_loop"):
            self.start(turns, 10.0)
            turns._on_transcript("Hi Synthe,", 11.2)
        self.assertEqual(turns._pending, ["Hi Synthe,"])
        whisper = self.processor(streaming=False)
        with patch.object(Pod4TurnProcessor, "_ensure_answer_loop"):
            self.start(whisper, 10.0)
            whisper._on_transcript("Hi Synthe,", 11.2)
        self.assertEqual(whisper._pending, [])  # Whisper never does this, so its rule stays as it was

    async def test_pieces_of_one_sentence_make_one_quick_turn(self) -> None:
        turns = self.processor()
        with patch.object(Pod4TurnProcessor, "_ensure_answer_loop"):
            self.start(turns, 10.0)
            turns._on_transcript("Hi Synthe, I had a really long day", 12.5)  # the STT's own pause
            self.stop(turns, 14.0)
            self.assertFalse(turns._ready_to_answer(14.1))  # the flushed end hasn't come yet
            turns._on_transcript("at work today", 14.25)
            self.assertFalse(turns._ready_to_answer(14.0 + END_SILENCE_SECONDS - 0.05))
            self.assertTrue(turns._ready_to_answer(14.0 + END_SILENCE_SECONDS + 0.01))
        self.assertEqual(turns._pending, ["Hi Synthe, I had a really long day", "at work today"])

    async def test_no_long_wait_when_the_flush_gives_nothing_new(self) -> None:
        # All the words came back before VAD stopped; the old rule then waited 4 s for "missing" text.
        turns = self.processor()
        with patch.object(Pod4TurnProcessor, "_ensure_answer_loop"):
            self.start(turns, 10.0)
            turns._on_transcript("I live in Pune and I love trekking", 12.4)
            self.stop(turns, 12.6)
            self.assertFalse(turns._ready_to_answer(12.6 + END_SILENCE_SECONDS + 0.1))
            self.assertTrue(turns._ready_to_answer(12.6 + STREAM_FLUSH_WAIT_SECONDS + 0.01))
        self.assertLess(STREAM_FLUSH_WAIT_SECONDS, STT_WAIT_MAX_SECONDS)

    async def test_a_late_short_piece_is_not_dropped(self) -> None:
        # A slow STT: the end of the sentence came back 4.5 s after the speech, longer than Whisper's queue keeps it.
        turns = self.processor()
        with patch.object(Pod4TurnProcessor, "_ensure_answer_loop"):
            self.start(turns, 10.0)
            self.stop(turns, 12.0)
            turns._on_transcript("so many tasks", 16.5)
        self.assertEqual(turns._pending, ["so many tasks"])

    async def test_speech_with_no_words_does_not_hold_the_turn(self) -> None:
        turns = self.processor()
        with patch.object(Pod4TurnProcessor, "_ensure_answer_loop"):
            self.start(turns, 10.0)
            self.stop(turns, 11.0)  # a cough: the flush gives no text
        self.assertTrue(turns._waiting_for_stt(11.5))
        self.assertFalse(turns._waiting_for_stt(11.0 + STREAM_FLUSH_WAIT_SECONDS + 0.01))


class HumanClaimTests(unittest.IsolatedAsyncioTestCase):
    async def test_nudges_never_mention_the_silence(self) -> None:
        self.assertTrue(mentions_silence("Silence can be heavy when you're still figuring out the right words."))
        self.assertFalse(mentions_silence("What would a really good day look like for you?"))
        self.assertTrue(mentions_silence("Hey, I\u2019m still here."))  # curly apostrophe from the model
        self.assertFalse(mentions_silence("Do you need a bit more quiet time at home afterward?"))
        session = CallSession("nudge", "english", conversation_pod="pod4")
        session.transcript.append({"role": "assistant", "content": "Hi, I'm Aria."})
        pieces = ["I'm waiting for you to pop in whenever you're ready. ", "What does a slow Sunday look like for you?"]

        async def stream(messages, **_):
            for piece in pieces:
                yield piece, "fake"

        with patch.object(pod4_providers, "stream_complete", stream):
            sentences = [s async for s in stream_talk(session, special_decision(session, "chat", "nudge"), None, "")]
        self.assertEqual(sentences, ["What does a slow Sunday look like for you?"])

    async def test_a_nudge_with_nothing_usable_stays_quiet(self) -> None:
        session = CallSession("nudge-empty", "english", conversation_pod="pod4")
        session.transcript.append({"role": "assistant", "content": "Hi, I'm Aria."})

        async def stream(messages, **_):
            yield "Silence can be awkward, take your time.", "fake"

        with patch.object(pod4_providers, "stream_complete", stream):
            sentences = [s async for s in stream_talk(session, special_decision(session, "chat", "nudge"), None, "")]
        self.assertEqual(sentences, [])

    def test_made_up_personal_stories_are_spotted(self) -> None:
        self.assertTrue(claims_human_life("I actually just saw a really wild bug in a code snippet today."))
        self.assertTrue(claims_human_life("Yesterday I went for a long walk."))
        self.assertTrue(claims_human_life("My friend loves biryani too."))
        self.assertFalse(claims_human_life("I hear you."))
        self.assertFalse(claims_human_life("A lot of people say debugging feels like that."))
        self.assertFalse(claims_human_life("I remember you said Lucknow."))

    async def test_streamed_reply_skips_the_made_up_sentence(self) -> None:
        session = CallSession("claims", "english", conversation_pod="pod4")
        pieces = ["Fair enough, I won't push. ", "I actually just saw a wild bug today. ", "Do side projects ever drive you mad?"]

        async def stream(messages, **_):
            for piece in pieces:
                yield piece, "fake"

        decision = Pod4Decision(response_mode="chat", ask_question=True, cue="chat", planning_reason="test")
        with patch.object(pod4_providers, "stream_complete", stream):
            sentences = [sentence async for sentence in stream_talk(session, decision, "skip", "")]
        self.assertEqual(sentences, ["Fair enough, I won't push.", "Do side projects ever drive you mad?"])


if __name__ == "__main__":
    unittest.main()
