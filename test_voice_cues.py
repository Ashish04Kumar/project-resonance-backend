"""Voice cues: measurements from timing, audio and words, and how they reach the Pod 4 LLM."""

from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np

from app import pod4_providers
from app.conversation import CallSession
from app.conversation_pod4_text import run_pod4_turn
from app.voice_cues import VoiceCueTracker, _Baseline, compute_cues, count_fillers, measure_audio, pitch_track

RATE = 16000


def tone(seconds: float, hz: float | np.ndarray, amplitude: float | np.ndarray = 0.3) -> np.ndarray:
    """A voice-like test sound: a tone (fixed or gliding pitch) with harmonics."""
    t = np.arange(int(seconds * RATE)) / RATE
    freq = np.broadcast_to(hz, t.shape) if np.ndim(hz) else np.full(t.shape, float(hz))
    phase = 2 * np.pi * np.cumsum(freq) / RATE
    wave = np.sin(phase) + 0.5 * np.sin(2 * phase) + 0.25 * np.sin(3 * phase)
    return (wave / 1.75 * amplitude).astype(np.float32)


def pcm(samples: np.ndarray) -> bytes:
    return (np.clip(samples, -1, 1) * 32767).astype(np.int16).tobytes()


def turn(text: str, seconds: float, samples: np.ndarray | None = None, baseline: _Baseline | None = None, **kwargs):
    audio = samples if samples is not None else tone(seconds, 180)
    return compute_cues(
        text=text,
        segments=[(0.0, seconds)],
        pcm=pcm(audio),
        sample_rate=RATE,
        pause_before=kwargs.get("pause_before", 1.0),
        interrupted=kwargs.get("interrupted", False),
        baseline=baseline,
    )


WORDS_20 = " ".join(["word"] * 20)


class VoiceCueTests(unittest.IsolatedAsyncioTestCase):
    def test_pitch_is_measured(self) -> None:
        samples = tone(1.0, 200)
        db = np.full(len(samples) // 320, -20.0)
        pitches = pitch_track(samples, RATE, db > -50)
        self.assertAlmostEqual(float(np.median(pitches)), 200, delta=8)

    def test_flat_vs_lively_intonation(self) -> None:
        flat = measure_audio(tone(3.0, 180), RATE)
        glide = 180 * 2 ** (np.sin(np.arange(3 * RATE) / RATE * 2 * np.pi * 1.2) * 5 / 12)  # +-5 semitones
        lively = measure_audio(tone(3.0, glide), RATE)
        self.assertLess(flat["pitch_spread_st"], 0.5)
        self.assertGreater(lively["pitch_spread_st"], 2.5)

    def test_trailing_off_is_noticed(self) -> None:
        envelope = np.ones(4 * RATE, dtype=np.float32)
        envelope[3 * RATE :] = 0.25  # last second about 12 dB quieter
        self.assertTrue(measure_audio(tone(4.0, 180) * envelope, RATE)["trailed_off"])
        self.assertFalse(measure_audio(tone(4.0, 180), RATE)["trailed_off"])

    def test_labels_compare_with_their_own_usual(self) -> None:
        baseline = _Baseline()
        for _ in range(2):  # their usual: 20 words in 8 s = 150 wpm, normal loudness
            usual = turn(WORDS_20, 8.0, baseline=baseline)
            self.assertEqual(usual.describe(), "")  # nothing to compare with yet / nothing unusual
        slow_soft = turn(WORDS_20, 14.0, tone(14.0, 180, amplitude=0.08), baseline=baseline, pause_before=4.2)
        self.assertEqual(slow_soft.pace, "slower")
        self.assertEqual(slow_soft.volume, "softer")
        note = slow_soft.describe()
        self.assertIn("took 4s before answering", note)
        self.assertIn("spoke slower than their usual", note)
        self.assertIn("softer than their usual", note)

    def test_short_replies_are_not_judged_on_pace_or_pitch(self) -> None:
        baseline = _Baseline(wpm=[150, 150], pitch_spread=[3.0, 3.0])
        cues = turn("haan theek hai", 0.9, baseline=baseline)
        self.assertTrue(cues.short_reply)
        self.assertIsNone(cues.pace)
        self.assertIsNone(cues.intonation)
        self.assertIn("very short reply", cues.describe())

    def test_fillers_and_mid_answer_pauses(self) -> None:
        self.assertEqual(count_fillers("Umm, I think... hmm, uh, maybe"), 3)
        cues = compute_cues(
            text="umm I was thinking hmm about my family and uh my work",
            segments=[(0.0, 2.0), (3.2, 5.0), (5.1, 7.0)],
            pcm=pcm(tone(5.7, 180)),
            sample_rate=RATE,
            pause_before=1.0,
            interrupted=False,
        )
        self.assertEqual((cues.long_pauses, cues.fillers), (1, 3))
        self.assertIn("1 long pause mid-answer", cues.describe())
        self.assertIn("3 filler sounds", cues.describe())

    def test_tracker_times_the_pause_and_spots_interruptions(self) -> None:
        tracker = VoiceCueTracker()
        tracker.bot_started(0.0)
        tracker.bot_stopped(5.0)
        tracker.user_started(8.5)
        tracker.audio(pcm(tone(2.0, 180)), RATE)
        tracker.user_stopped(10.5)
        first = tracker.take_turn("I was thinking about it for a while", 11.0).measure()
        self.assertAlmostEqual(first.pause_before_s, 3.5)
        self.assertFalse(first.interrupted)
        tracker.bot_started(12.0)
        tracker.user_started(13.0)  # talks over Aria
        tracker.user_stopped(15.0)
        second = tracker.take_turn("wait no I meant something else", 15.5).measure()
        self.assertTrue(second.interrupted)
        self.assertIn("started talking while Synthe was still speaking", second.describe())

    def test_a_sound_with_no_words_does_not_skew_the_next_turn(self) -> None:
        tracker = VoiceCueTracker()
        tracker.bot_stopped(10.0)
        tracker.user_started(18.0)  # a short "hmm" that gave no text
        tracker.user_stopped(18.3)
        tracker.discard_turn()
        tracker.bot_started(28.0)  # the silence nudge
        tracker.bot_stopped(35.0)
        tracker.user_started(37.0)
        tracker.audio(pcm(tone(2.0, 180)), RATE)
        tracker.user_stopped(39.0)
        cues = tracker.take_turn("last time I went out with this girl we talked for hours", 39.5).measure()
        self.assertAlmostEqual(cues.pause_before_s, 2.0)
        self.assertEqual(cues.long_pauses, 0)

    def test_audio_outside_speech_is_not_kept(self) -> None:
        tracker = VoiceCueTracker()
        tracker.audio(pcm(tone(1.0, 180)), RATE)  # nobody speaking: dropped
        snapshot = tracker.take_turn("hello", 1.0)
        self.assertEqual(snapshot.pcm, b"")

    async def test_note_reaches_the_observer_and_the_talker(self) -> None:
        session = CallSession("cues-test", "english", conversation_pod="pod4")
        replies = iter(['{"intent": "answer", "outcome": "partial"}', "That sounds like a lot. Take your time."])
        prompts: list[str] = []

        async def fake_complete(messages, **_):
            prompts.append("\n".join(message.content for message in messages))
            return next(replies), "fake"

        note = "took 4s before answering; spoke slower than their usual"
        with patch.object(pod4_providers, "complete", side_effect=fake_complete):
            await run_pod4_turn(session, "I don't know, it's complicated", note)
        self.assertIn(f"How they sounded: {note}", prompts[0])
        self.assertIn(f"How they sounded just now (rough measurements, may be wrong): {note}", prompts[1])
        self.assertIn("Never mention their voice", prompts[1])

    async def test_text_turns_have_no_voice_note(self) -> None:
        session = CallSession("cues-text", "english", conversation_pod="pod4")
        replies = iter(['{"intent": "answer", "outcome": "partial"}', "Nice. Tell me more?"])
        prompts: list[str] = []

        async def fake_complete(messages, **_):
            prompts.append("\n".join(message.content for message in messages))
            return next(replies), "fake"

        with patch.object(pod4_providers, "complete", side_effect=fake_complete):
            await run_pod4_turn(session, "I'm from Pune")
        self.assertNotIn("How they sounded", prompts[0] + prompts[1])


if __name__ == "__main__":
    unittest.main()
