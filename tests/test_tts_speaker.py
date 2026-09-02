"""Tests for tts_speaker.py -- run with:  python -m pytest tests/ -v
No webcam, MediaPipe, or audio device required."""
import os
import sys
import time
import threading

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tts_speaker import GestureStabilizer, TTSSpeaker  # noqa: E402


# ---------------------------------------------------------------- stabilizer

def feed(stab, labels):
    """Return list of (frame_index, spoken_label) events."""
    return [(i, out) for i, l in enumerate(labels)
            if (out := stab.update(l)) is not None]


def test_speaks_once_after_n_consecutive_frames():
    stab = GestureStabilizer(stable_frames=10)
    events = feed(stab, ["hello"] * 50)
    assert events == [(9, "hello")], "must fire exactly once, on frame 10"


def test_does_not_speak_before_threshold():
    stab = GestureStabilizer(stable_frames=10)
    assert feed(stab, ["hello"] * 9) == []


def test_flicker_resets_streak():
    stab = GestureStabilizer(stable_frames=10)
    # 9 hellos, one wrong frame, then 9 more hellos: never stable
    seq = ["hello"] * 9 + ["thanks"] + ["hello"] * 9
    assert feed(stab, seq) == []
    # one more hello completes a fresh streak of 10
    assert stab.update("hello") == "hello"


def test_no_hand_resets_streak_but_not_last_spoken():
    stab = GestureStabilizer(stable_frames=3)
    assert feed(stab, ["hello"] * 3) == [(2, "hello")]
    # hand disappears, then the same word comes back stable: NOT repeated
    seq = [None] * 5 + ["hello"] * 20
    assert feed(stab, seq) == []


def test_new_word_then_old_word_again():
    stab = GestureStabilizer(stable_frames=3)
    seq = (["hello"] * 6 +      # speak hello @2
           ["thanks"] * 6 +     # speak thanks @8
           ["hello"] * 6)       # hello changed back -> speak again @14
    assert feed(stab, seq) == [(2, "hello"), (8, "thanks"), (14, "hello")]


def test_switching_before_stable_is_ignored():
    stab = GestureStabilizer(stable_frames=5)
    seq = ["a", "b", "a", "b", "a", "b", "a", "b"]
    assert feed(stab, seq) == []


def test_reset_allows_repeat():
    stab = GestureStabilizer(stable_frames=2)
    assert feed(stab, ["hi", "hi"]) == [(1, "hi")]
    stab.reset()
    assert feed(stab, ["hi", "hi"]) == [(1, "hi")]


def test_invalid_threshold():
    try:
        GestureStabilizer(stable_frames=0)
    except ValueError:
        return
    assert False, "expected ValueError"


# ------------------------------------------------------------------- speaker

class FakeEngine:
    """Stand-in for pyttsx3's engine: records calls, simulates blocking."""
    def __init__(self, delay=0.05):
        self.spoken = []
        self.threads = []
        self.props = {}
        self.delay = delay
        self._pending = None

    def setProperty(self, k, v):
        self.props[k] = v

    def say(self, text):
        self._pending = text

    def runAndWait(self):
        time.sleep(self.delay)               # pretend audio is playing
        self.spoken.append(self._pending)
        self.threads.append(threading.current_thread().name)

    def stop(self):
        pass


def test_speak_does_not_block_and_runs_on_worker_thread():
    eng = FakeEngine(delay=0.2)
    spk = TTSSpeaker(engine_factory=lambda: eng)
    assert spk.available

    t0 = time.perf_counter()
    assert spk.speak("hello")
    elapsed = time.perf_counter() - t0
    assert elapsed < 0.05, "speak() blocked the caller for %.3fs" % elapsed

    # wait for the background thread to actually speak it
    deadline = time.time() + 2
    while not eng.spoken and time.time() < deadline:
        time.sleep(0.01)
    assert eng.spoken == ["hello"]
    assert eng.threads == ["tts-speaker"]
    assert eng.threads[0] != threading.current_thread().name
    spk.stop()


def test_words_spoken_in_order():
    eng = FakeEngine(delay=0.01)
    spk = TTSSpeaker(engine_factory=lambda: eng)
    for w in ["one", "two", "three"]:
        spk.speak(w)
    deadline = time.time() + 2
    while len(eng.spoken) < 3 and time.time() < deadline:
        time.sleep(0.01)
    spk.stop()
    assert eng.spoken == ["one", "two", "three"]


def test_stop_does_not_drain_backlog():
    """On shutdown (ESC) we want to stop talking promptly, not finish a
    backlog of queued words."""
    eng = FakeEngine(delay=0.3)
    spk = TTSSpeaker(engine_factory=lambda: eng)
    for w in ["one", "two", "three"]:
        spk.speak(w)
    time.sleep(0.05)            # worker is mid-way through "one"
    t0 = time.perf_counter()
    spk.stop()
    assert time.perf_counter() - t0 < 1.0
    assert eng.spoken == ["one"]


def test_queue_full_drops_instead_of_blocking():
    eng = FakeEngine(delay=0.5)
    spk = TTSSpeaker(engine_factory=lambda: eng, max_queue=2)
    results = [spk.speak(str(i)) for i in range(6)]
    # first one is picked up by the worker almost immediately, next two
    # fit in the queue, the rest are dropped -- and none of them blocked
    assert results.count(True) >= 2
    assert results.count(False) >= 1
    spk.stop()


def test_rate_and_volume_applied():
    eng = FakeEngine()
    spk = TTSSpeaker(rate=150, volume=0.8, engine_factory=lambda: eng)
    spk.stop()
    assert eng.props == {'rate': 150, 'volume': 0.8}


def test_graceful_when_engine_init_fails():
    def broken():
        raise RuntimeError("no audio backend")
    spk = TTSSpeaker(engine_factory=broken)
    assert spk.available is False
    assert isinstance(spk.error, RuntimeError)
    assert spk.speak("hello") is False   # no-op, no exception
    spk.stop()


def test_graceful_when_pyttsx3_missing():
    import builtins
    real_import = builtins.__import__

    def fake_import(name, *a, **k):
        if name == "pyttsx3":
            raise ImportError("No module named pyttsx3")
        return real_import(name, *a, **k)

    builtins.__import__ = fake_import
    try:
        spk = TTSSpeaker()          # default factory -> import pyttsx3
        assert spk.available is False
        assert isinstance(spk.error, ImportError)
        spk.stop()
    finally:
        builtins.__import__ = real_import
