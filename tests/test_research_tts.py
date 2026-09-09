import time

from sh_research.config import TTSConfig
from sh_research.tts import MockBackend, TTSWorker


def test_speak_is_non_blocking_and_records_latency():
    w = TTSWorker(TTSConfig(backend="mock", duplicate_window_s=0.0, prewarm=False), MockBackend(latency_s=0.05)).start()
    t0 = time.monotonic()
    assert w.speak("hello", confirmed_ts=t0)
    assert time.monotonic() - t0 < 0.01  # returned immediately
    time.sleep(0.2)
    w.stop()
    assert w.spoken_count == 1
    r = w.records[0]
    assert r.total_prediction_to_speech_ms >= 50 and r.tts_generation_ms >= 50
    assert w.stats()["t6_t2_confirmed_to_audio_ms"]["n"] == 1


def test_duplicate_suppression():
    w = TTSWorker(TTSConfig(backend="mock", duplicate_window_s=1.0, queue_size=5), MockBackend(latency_s=0.0)).start()
    results = [w.speak("HELLO") for _ in range(4)]
    time.sleep(0.1)
    w.stop()
    assert results == [True, False, False, False] and w.suppressed_duplicates == 3 and w.spoken_count == 1


def test_bounded_queue_drops_oldest():
    w = TTSWorker(TTSConfig(backend="mock", duplicate_window_s=0.0, queue_size=1, overflow_policy="drop_oldest"),
                  MockBackend(latency_s=0.15))
    w.start()
    w.speak("one"); time.sleep(0.02)  # being spoken
    w.speak("two"); w.speak("three"); w.speak("four")  # only one may wait
    time.sleep(0.5)
    w.stop()
    assert w.backend.spoken == ["one", "four"] and w.dropped_overflow == 2


def test_bounded_queue_drops_newest():
    w = TTSWorker(TTSConfig(backend="mock", duplicate_window_s=0.0, queue_size=1, overflow_policy="drop_newest"),
                  MockBackend(latency_s=0.15)).start()
    w.speak("one"); time.sleep(0.02)
    w.speak("two"); w.speak("three")
    time.sleep(0.5)
    w.stop()
    assert w.backend.spoken == ["one", "two"] and w.dropped_overflow == 1


def test_disabled_tts_does_nothing():
    w = TTSWorker(TTSConfig(enabled=False, backend="mock"), MockBackend()).start()
    assert w.speak("x") is False
    w.stop()
    assert w.spoken_count == 0


def test_backend_error_does_not_kill_worker():
    class Bad:
        name = "bad"
        def speak(self, text):
            raise RuntimeError("boom")
        def close(self):
            pass
    w = TTSWorker(TTSConfig(backend="mock", duplicate_window_s=0.0, queue_size=3, prewarm=False), Bad()).start()
    w.speak("a"); w.speak("b")
    time.sleep(0.3)
    w.stop()
    assert len(w.errors) == 2 and w.spoken_count == 0 and w.failed == 2
