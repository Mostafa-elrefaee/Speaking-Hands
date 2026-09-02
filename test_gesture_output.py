#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Tests for gesture_output.py using scripted fake per-frame predictions.
No camera, no models, no audio hardware needed:  python -m pytest -q
"""
import time
import threading

import pytest

from gesture_output import choose_gesture, GestureStabilizer, SpeechWorker

STABLE = 8  # mirrors app.STABLE_FRAMES


class FakeEngine:
    """Records what is spoken and sleeps to simulate playback time."""
    def __init__(self, seconds_per_word=0.05):
        self.said = []
        self.stopped = False
        self.seconds_per_word = seconds_per_word
        self._pending = None

    def say(self, text):
        self._pending = text

    def runAndWait(self):
        time.sleep(self.seconds_per_word)
        self.said.append(self._pending)

    def stop(self):
        self.stopped = True


def drive(stab, raw_frames, speaker=None):
    """Run a list of raw per-frame values through the stabilizer exactly as
    app.py does; return the list of (frame_idx, spoken_value) events."""
    events = []
    for i, raw in enumerate(raw_frames):
        current, changed = stab.update(raw)
        if changed and current is not None:
            events.append((i, current))
            if speaker is not None:
                speaker.say(current)
    return events


# ---------------------------------------------------------------- priority

def test_sequence_wins_when_valid_else_static():
    assert choose_gesture("wave", "flat_hand") == ("wave", "seq")
    assert choose_gesture(None, "flat_hand") == ("flat_hand", "static")
    assert choose_gesture(None, None) == (None, None)


# --------------------------------------------------------------- stability

def test_static_only_spoken_once_while_held():
    stab = GestureStabilizer(STABLE)
    events = drive(stab, ["hello"] * 60)
    assert events == [(STABLE - 1, "hello")]  # exactly one event, on frame 8
    assert stab.current == "hello"


def test_nothing_before_stabilization():
    stab = GestureStabilizer(STABLE)
    events = drive(stab, ["hello"] * (STABLE - 1))
    assert events == [] and stab.current is None


def test_flicker_shorter_than_stable_frames_is_ignored():
    stab = GestureStabilizer(STABLE)
    frames = ["hello"] * 20 + ["thanks"] * 3 + ["hello"] * 20
    events = drive(stab, frames)
    assert events == [(STABLE - 1, "hello")]  # 3-frame flicker never surfaced
    # a 3-frame flicker back to the SAME value also does not re-speak
    frames = ["hello"] * 20 + ["thanks"] * 3 + ["hello"] * 2 + ["thanks"] * 3
    stab = GestureStabilizer(STABLE)
    assert drive(stab, frames) == [(STABLE - 1, "hello")]


def test_motion_sign_onset_disagreement_is_absorbed():
    """A dynamic sign passes through a static hand-shape ('flat_hand') for a
    few frames before the sequence model becomes confident ('wave').
    Simulates the real priority path: seq None -> static fallback, then seq
    valid -> seq wins."""
    stab = GestureStabilizer(STABLE)
    onset = [choose_gesture(None, "flat_hand")[0]] * 5      # < STABLE frames
    motion = [choose_gesture("wave", "flat_hand")[0]] * 30  # seq authoritative
    events = drive(stab, onset + motion)
    assert events == [(5 + STABLE - 1, "wave")]
    # If the static onset lasts >= STABLE frames it WILL be spoken -- that's
    # the documented tradeoff, made explicit here.
    stab = GestureStabilizer(STABLE)
    events = drive(stab, ["flat_hand"] * STABLE + ["wave"] * 30)
    assert [e[1] for e in events] == ["flat_hand", "wave"]


def test_dropped_hand_gap_then_same_sign_speaks_again():
    """Hands vanish for longer than STABLE frames -> current goes back to
    None (screen clears, nothing spoken) -> same sign again is a new
    stabilization event, so a repeated word is repeated."""
    stab = GestureStabilizer(STABLE)
    frames = ["hello"] * 20 + [None] * 12 + ["hello"] * 20
    events = drive(stab, frames)
    assert [e[1] for e in events] == ["hello", "hello"]
    # ...but a SHORT dropout (< STABLE) is invisible: no clear, no re-speak
    stab = GestureStabilizer(STABLE)
    frames = ["hello"] * 20 + [None] * 4 + ["hello"] * 20
    assert [e[1] for e in drive(stab, frames)] == ["hello"]


def test_rapid_real_gesture_changes_each_spoken_once():
    stab = GestureStabilizer(STABLE)
    frames = ["hello"] * 10 + ["thanks"] * 9 + ["same"] * 8 + ["hello"] * 15
    events = drive(stab, frames)
    assert [e[1] for e in events] == ["hello", "thanks", "same", "hello"]
    # and a gesture held for fewer than STABLE frames is treated as noise
    stab = GestureStabilizer(STABLE)
    frames = ["hello"] * 10 + ["thanks"] * 7 + ["same"] * 10
    assert [e[1] for e in drive(stab, frames)] == ["hello", "same"]


# ----------------------------------------------------------------- speech

def test_speech_is_queued_not_interrupted_and_never_blocks():
    engine = FakeEngine(seconds_per_word=0.15)
    worker = SpeechWorker(engine_factory=lambda: engine).start()
    assert engine.said == []

    stab = GestureStabilizer(STABLE)
    frames = ["hello"] * 10 + ["thanks"] * 9 + ["same"] * 8
    t0 = time.perf_counter()
    events = drive(stab, frames, speaker=worker)
    main_loop_time = time.perf_counter() - t0
    assert main_loop_time < 0.05, "main loop must not wait on playback"
    assert len(events) == 3

    time.sleep(0.15 * 3 + 0.2)
    assert engine.said == ["hello", "thanks", "same"]  # in order, none lost
    worker.close()
    assert engine.stopped


def test_close_mid_utterance_returns_promptly_and_no_exception():
    engine = FakeEngine(seconds_per_word=0.5)
    worker = SpeechWorker(engine_factory=lambda: engine).start()
    worker.say("hello")
    worker.say("thanks")
    time.sleep(0.05)  # "hello" is now in flight
    t0 = time.perf_counter()
    worker.close(timeout=1.0)
    assert time.perf_counter() - t0 < 1.2
    assert threading.active_count() >= 1  # nothing to assert except no crash


def test_backlog_safety_valve_drops_oldest():
    engine = FakeEngine(seconds_per_word=0.3)
    worker = SpeechWorker(engine_factory=lambda: engine, maxsize=2).start()
    for w in ["a", "b", "c", "d"]:
        worker.say(w)
    time.sleep(0.05)
    assert worker.pending <= 2
    worker.close(timeout=2.0)


def test_engine_init_failure_surfaces_with_message():
    def bad_factory():
        raise RuntimeError("pip install pyttsx3")
    with pytest.raises(RuntimeError, match="pip install pyttsx3"):
        SpeechWorker(engine_factory=bad_factory).start()


def test_utterance_failure_does_not_kill_worker():
    class Flaky(FakeEngine):
        def runAndWait(self):
            if self._pending == "boom":
                raise OSError("audio device busy")
            super().runAndWait()
    engine = Flaky(0.02)
    worker = SpeechWorker(engine_factory=lambda: engine).start()
    worker.say("boom")
    worker.say("hello")
    time.sleep(0.3)
    assert engine.said == ["hello"]
    worker.close()


def test_real_pyttsx3_factory_error_is_actionable():
    """Whatever this machine has, create_pyttsx3_engine must either return
    an engine or raise a RuntimeError that tells the user what to install."""
    from gesture_output import create_pyttsx3_engine
    try:
        eng = create_pyttsx3_engine()
        assert hasattr(eng, "say")
    except RuntimeError as e:
        msg = str(e)
        assert "pip install pyttsx3" in msg or "espeak" in msg.lower() \
            or "--no_tts" in msg


def test_subprocess_backend_speaks_every_word_not_just_the_first():
    """Regression for the 'only the first gesture is spoken' pyttsx3 bug:
    the default backend must isolate each word in a fresh process."""
    pytest.importorskip("pyttsx3")
    try:
        worker = SpeechWorker().start()
    except RuntimeError as e:
        pytest.skip(f"no speech backend on this machine: {e}")
    for w in ["hello", "thanks", "same", "hello"]:
        worker.say(w)
    deadline = time.time() + 30
    while len(worker.spoken) < 4 and time.time() < deadline:
        time.sleep(0.1)
    worker.close()
    assert worker.spoken == ["hello", "thanks", "same", "hello"]
