"""Prompt-3 §20: repeated / low-confidence / rapidly changing predictions, TTS failure,
queue overflow, long and short audio, inference continuing while TTS plays."""
import time

import numpy as np

from sh_research.config import StabilizerConfig, TTSConfig
from gesture_output import PredictionStabilizer, StableEvent
from sh_research.tts import MockBackend, SpeechEventManager, TTSWorker, label_to_text, normalize_text, join_phrase


def _ev(label, t, conf=0.9, stable_since=None):
    return StableEvent(0, label, conf, t, 5, t - 0.05, t - 0.01, stable_since if stable_since is not None else t)


def _chain(stab_kw=None, tts_kw=None):
    st = PredictionStabilizer(StabilizerConfig(confidence_threshold=0.8, stable_count=3, averaging_window=1, **(stab_kw or {})), ["hello", "thanks", "iloveyou"])
    mgr = SpeechEventManager(TTSConfig(backend="mock", **(tts_kw or {})))
    return st, mgr


def run_stream(st, mgr, probs_seq, dt=0.05, t0=0.0):
    """Exactly what GesturePipeline does per frame: update -> on_validated / tick(current)."""
    out = []
    for i, p in enumerate(probs_seq):
        t = t0 + i * dt
        ev = st.update(np.array(p), now=t, predicted_at=t)
        sp = mgr.on_validated(ev, t) if ev is not None else None
        if sp is None:
            sp = mgr.tick(t, st.current)
        if sp:
            out.append((t, sp.text))
    return out


def test_repeated_predictions_one_speech_event():
    st, mgr = _chain(tts_kw={"duplicate_cooldown_ms": 3000})
    hello = [[0.95, 0.03, 0.02]] * 40  # 2 s of HELLO at 20 FPS
    none = [[0.4, 0.3, 0.3]] * 5       # hands down / low confidence -> current clears
    out = run_stream(st, mgr, hello)
    assert [t for _, t in out] == ["hello"]  # one held sign = one utterance (stabilizer change-detection)
    # released and re-signed within the 3 s cooldown: suppressed by the event manager
    out2 = run_stream(st, mgr, none + hello, t0=2.0)
    assert out2 == [] and mgr.suppressed_duplicates == 1
    # after the cooldown the same sign is spoken again
    out3 = run_stream(st, mgr, none + hello, t0=10.0)
    assert [t for _, t in out3] == ["hello"]


def test_low_confidence_never_speaks():
    st, mgr = _chain()
    out = run_stream(st, mgr, [[0.6, 0.3, 0.1]] * 40)
    assert out == [] and mgr.events_created == 0


def test_rapidly_changing_predictions_do_not_speak():
    st, mgr = _chain()
    flip = [[0.95, 0.03, 0.02], [0.03, 0.95, 0.02]] * 20  # alternates every frame
    assert run_stream(st, mgr, flip) == []


def test_require_prediction_change_and_min_stable_duration():
    st, mgr = _chain(tts_kw={"duplicate_cooldown_ms": 0, "require_prediction_change": True})
    hello = [[0.95, 0.03, 0.02]] * 20
    thanks = [[0.02, 0.95, 0.03]] * 20
    out = run_stream(st, mgr, hello + thanks + hello)
    assert [t for _, t in out] == ["hello", "thanks", "hello"]
    st, mgr = _chain(tts_kw={"minimum_stable_duration_ms": 400})
    out = run_stream(st, mgr, hello)
    assert out and out[0][0] >= 0.4  # first event only after 400 ms of stability


def test_buffered_phrase_mode():
    st, mgr = _chain(tts_kw={"mode": "buffered_phrase", "phrase_max_words": 3, "phrase_timeout_ms": 500,
                             "duplicate_cooldown_ms": 0, "capitalization": "sentence", "punctuation": True,
                             "label_map": {"iloveyou": "I love you"}})
    seq = [[0.95, 0.03, 0.02]] * 5 + [[0.02, 0.95, 0.03]] * 5 + [[0.02, 0.03, 0.95]] * 5
    out = run_stream(st, mgr, seq)
    assert len(out) == 1 and out[0][1] == "Hello thanks I love you."
    st, mgr = _chain(tts_kw={"mode": "buffered_phrase", "phrase_max_words": 5, "phrase_timeout_ms": 300, "duplicate_cooldown_ms": 0})
    out = run_stream(st, mgr, [[0.95, 0.03, 0.02]] * 5 + [[0.6, 0.2, 0.2]] * 20)  # silence -> timeout flush
    # hello held until frame 4 (t=0.20); 'None' stabilises 3 frames later (t=0.35);
    # the 300 ms silence timeout counts from that release -> flush at 0.65
    assert [t for _, t in out] == ["hello"] and 0.6 <= out[0][0] <= 0.75


def test_text_normalisation():
    cfg = TTSConfig(label_map={"i_love_you": "  I   love you "}, capitalization="lower", split_label_words=True)
    assert label_to_text("THANK_YOU", TTSConfig()) == "THANK_YOU"  # default: label spoken as-is (original behaviour)
    assert label_to_text("i_love_you", cfg) == "i love you"
    assert label_to_text("THANK_YOU", cfg) == "thank you"
    assert normalize_text("   ", cfg) is None
    cfg2 = TTSConfig(unknown_label_policy="skip")
    assert label_to_text("unknown", cfg2) is None
    assert join_phrase(["hello", " world "], TTSConfig(capitalization="sentence", punctuation=True)) == "Hello world."


def test_generation_failure_does_not_stop_inference_or_worker():
    w = TTSWorker(TTSConfig(backend="mock", prewarm=False, max_pending_age_ms=0, queue_size=3),
                  MockBackend(latency_s=0.01, fail_texts={"bad"})).start()
    assert w.speak("bad") and w.speak("good")
    time.sleep(0.2)
    w.stop()
    assert w.failed == 1 and w.spoken_count == 1 and w.available is False and w.backend.spoken == ["good"]


def test_queue_overflow_and_stale_expiry():
    w = TTSWorker(TTSConfig(backend="mock", prewarm=False, queue_size=1, overflow_policy="drop_oldest", max_pending_age_ms=0),
                  MockBackend(latency_s=0.15)).start()
    for txt in ["a", "b", "c", "d"]:
        w.speak(txt); time.sleep(0.01)
    time.sleep(0.6); w.stop()
    assert w.backend.spoken == ["a", "d"] and w.dropped_overflow == 2
    # stale expiry: an event that waited too long is never spoken
    w = TTSWorker(TTSConfig(backend="mock", prewarm=False, queue_size=2, max_pending_age_ms=100), MockBackend(latency_s=0.3)).start()
    w.speak("first"); time.sleep(0.01); w.speak("second")
    time.sleep(0.8); w.stop()
    assert w.backend.spoken == ["first"] and w.expired == 1


def test_long_and_short_audio_do_not_block_inference():
    long_b = MockBackend(latency_s=0.02, playback_s=0.6)
    w = TTSWorker(TTSConfig(backend="mock", prewarm=False, max_pending_age_ms=0), long_b).start()
    t0 = time.monotonic()
    w.speak("a long sentence", confirmed_ts=t0)
    steps = 0
    while time.monotonic() - t0 < 0.5:  # "inference" keeps running while audio plays
        np.random.rand(30, 126).sum(); steps += 1
        time.sleep(0.005)
    assert w.is_speaking and steps > 50
    time.sleep(0.3); w.stop()
    r = w.records[0].to_dict()
    assert r["playback_ms"] >= 600 and r["t6_t2_confirmed_to_audio_ms"] < 100
    short_b = MockBackend(latency_s=0.001, playback_s=0.005)
    w = TTSWorker(TTSConfig(backend="mock", prewarm=False, max_pending_age_ms=0, queue_size=2), short_b).start()
    for i in range(3):
        w.speak(f"w{i}"); time.sleep(0.03)
    time.sleep(0.1); w.stop()
    assert w.spoken_count == 3


def test_interrupt_only_when_backend_supports_it():
    b = MockBackend(latency_s=0.01, playback_s=0.5)
    w = TTSWorker(TTSConfig(backend="mock", prewarm=False, interrupt_current_audio=True, max_pending_age_ms=0, queue_size=2), b).start()
    w.speak("one"); time.sleep(0.05); w.speak("two")
    time.sleep(0.8); w.stop()  # let "two" finish (stop() would otherwise cut it too)
    assert b.interrupted == 1 and w.interrupts == 1 and b.spoken == ["one", "two"]
    class NoInterrupt(MockBackend):
        supports_interrupt = False
    b2 = NoInterrupt(latency_s=0.01, playback_s=0.2)
    w = TTSWorker(TTSConfig(backend="mock", prewarm=False, interrupt_current_audio=True, max_pending_age_ms=0, queue_size=2), b2).start()
    w.speak("one"); time.sleep(0.05); w.speak("two"); time.sleep(0.5); w.stop()
    assert b2.interrupted == 0 and w.interrupts == 0


def test_prewarm_is_measured_and_removes_first_call_penalty():
    cold = TTSWorker(TTSConfig(backend="mock", prewarm=False, max_pending_age_ms=0), MockBackend(latency_s=0.01, init_s=0.2)).start()
    cold.speak("x", confirmed_ts=time.monotonic()); time.sleep(0.4); cold.stop()
    warm = TTSWorker(TTSConfig(backend="mock", prewarm=True, max_pending_age_ms=0), MockBackend(latency_s=0.01, init_s=0.2)).start()
    time.sleep(0.3)
    warm.speak("x", confirmed_ts=time.monotonic()); time.sleep(0.2); warm.stop()
    assert cold.prewarm_ms is None and warm.prewarm_ms >= 200
    assert cold.records[0].to_dict()["t5_t4_synthesis_ms"] >= 200 > warm.records[0].to_dict()["t5_t4_synthesis_ms"]


def test_backend_without_optional_methods_is_tolerated():
    class Minimal:
        name = "legacy"
        def speak(self, text):
            t = time.monotonic(); return t, t, t
    w = TTSWorker(TTSConfig(backend="mock", prewarm=True, max_pending_age_ms=0), Minimal()).start()
    w.speak("hi"); time.sleep(0.15); w.stop()
    assert w.spoken_count == 1 and not w.errors
