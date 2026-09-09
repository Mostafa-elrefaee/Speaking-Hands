"""Realtime integration: scheduler + latest-frame buffer + GesturePipeline
(the same objects app.py runs) driven by a synthetic camera and synthetic
landmark extractor, with fake TFLite-style classifiers. No camera, MediaPipe
or audio needed."""
import time

import numpy as np
import pytest

from gesture_output import PredictionStabilizer
from sh_research.config import ExperimentConfig, StabilizerConfig
from sh_research.realtime import (FrameSource, GesturePipeline, LatestFrameSlot, PerfTracker, RateScheduler,
                                  SyntheticCamera, SyntheticExtractor)
from sh_research.tts import MockBackend, SpeechEventManager, TTSWorker


class FakeStatic:
    def __init__(self, cost_s=0.0):
        self.cost_s = cost_s

    def __call__(self, vec):
        if self.cost_s:
            time.sleep(self.cost_s)
        return 0


class FakeSequence:
    """Looks like model/sequence_classifier/SequenceClassifier."""
    invalid_value = -1

    def __init__(self, seq_length=10, probs=(0.9, 0.1), cost_s=0.0):
        self.seq_length, self.num_features, self.probs, self.cost_s = seq_length, 84, probs, cost_s
        self.output_details = [{"shape": np.array([1, len(probs)])}]
        self.calls = []

    def predict_proba(self, window):
        if len(window) < self.seq_length:
            return None
        self.calls.append(np.asarray(list(window)[-self.seq_length:]).shape)
        if self.cost_s:
            time.sleep(self.cost_s)
        return np.array(self.probs, dtype=np.float32)


def make_pipeline(cfg, extractor, seq=None, tts=None, perf=None):
    labels = ["wave", "circle"] if seq is not None else ["flat_hand"]
    stab = PredictionStabilizer(cfg.stabilizer, labels)
    return GesturePipeline(cfg, extractor, FakeStatic(), ["flat_hand"], seq, ["wave", "circle"] if seq else None,
                           stab, SpeechEventManager(cfg.tts), tts, perf=perf)


def run_source(cfg, pipeline, camera_fps=30.0, duration_s=1.5, max_frames=None):
    src = FrameSource(cfg.realtime, pipeline.perf,
                      thread=SyntheticCamera(None, fps=camera_fps, shape=(60, 80, 3), max_frames=max_frames))
    src.start()
    t0 = time.monotonic()
    steps = 0
    try:
        while time.monotonic() - t0 < duration_s:
            item = src.next()
            if item is None:
                if src.ended:
                    break
                continue
            pipeline.process(item.frame, item.captured_at, item.late_s, item.acquire_ms)
            pipeline.perf.record_draw(0.0)
            steps += 1
    finally:
        src.close()
    return steps


# --- building blocks -------------------------------------------------------------

def test_slot_keeps_only_newest():
    slot = LatestFrameSlot()
    for i in range(5):
        slot.put(i)
    item = slot.take(timeout=0)
    assert item.frame == 4 and slot.dropped == 4 and slot.take(timeout=0) is None


def test_scheduler_rate_with_fake_clock():
    t = [0.0]
    sched = RateScheduler(20.0, max_catchup_intervals=2, clock=lambda: t[0])

    def sleep(s):
        t[0] += s
    lates = [sched.wait_until_due(sleep) for _ in range(10)]
    assert t[0] == pytest.approx(9 * 0.05, abs=1e-9) and max(lates) == 0
    t[0] += 1.0  # fall far behind -> re-anchor instead of bursting
    sched.wait_until_due(sleep)
    assert sched.reanchors == 1
    before = t[0]
    sched.wait_until_due(sleep)
    assert t[0] - before == pytest.approx(0.05, abs=1e-9)


def test_scheduler_real_clock_20fps():
    sched = RateScheduler(20.0)
    t0 = time.monotonic()
    for _ in range(10):
        sched.wait_until_due()
    assert 0.40 <= time.monotonic() - t0 <= 0.65


# --- stabilizer (probability path) -------------------------------------------------

def test_prediction_stabilizer_default_equals_gesture_stabilizer():
    """averaging=1, voting=0, threshold, stable_count: one event per held sign,
    re-fires only after a different value (or None) stabilises -- the exact
    GestureStabilizer contract app.py always had."""
    st = PredictionStabilizer(StabilizerConfig(confidence_threshold=0.6, stable_count=3), ["a", "b"])
    a = np.array([0.9, 0.1])
    events = [st.update(a, now=0.05 * i) for i in range(10)]
    got = [e for e in events if e is not None]
    assert len(got) == 1 and got[0].label == "a" and got[0].frames_to_confirm == 3 and st.current == "a"
    assert all(st.update(np.array([0.55, 0.45]), now=1 + 0.05 * i) is None for i in range(2))  # below threshold
    assert st.current == "a"  # 2 low-confidence frames < stable_count: current not cleared
    for i in range(3):
        st.update(np.array([0.5, 0.5]), now=2 + 0.05 * i)  # None stabilises after 3 -> screen clears
    assert st.current is None
    assert st.update(a, now=3.0) is None and st.update(a, now=3.05) is None and st.update(a, now=3.1) is not None


def test_stabilizer_cooldown_and_duplicate_suppression():
    cfg = StabilizerConfig(confidence_threshold=0.8, stable_count=2, cooldown_s=1.0)
    st = PredictionStabilizer(cfg, ["a", "b"])
    assert st.update(np.array([0.95, 0.05]), now=1.0) is None
    assert st.update(np.array([0.95, 0.05]), now=1.05) is not None
    assert st.update(np.array([0.05, 0.95]), now=1.10) is None
    assert st.update(np.array([0.05, 0.95]), now=1.15) is None  # b stabilised but inside cooldown
    assert st.suppressed == 1
    cfg = StabilizerConfig(confidence_threshold=0.5, stable_count=1, duplicate_suppression=True, repeat_after_s=2.0)
    st = PredictionStabilizer(cfg, ["a", "b"])
    assert st.update(np.array([0.9, 0.1]), now=0.0) is not None
    st.update(np.array([0.1, 0.1]), now=0.1)  # None -> clears
    assert st.update(np.array([0.9, 0.1]), now=0.2) is None  # same label again within 2 s: suppressed
    assert st.update(np.array([0.9, 0.1]), now=3.0) is None  # held: no change
    st.update(np.array([0.1, 0.1]), now=3.1)
    assert st.update(np.array([0.9, 0.1]), now=3.2) is not None


def test_stabilizer_averaging_and_majority_vote():
    st = PredictionStabilizer(StabilizerConfig(confidence_threshold=0.5, stable_count=1, averaging_window=3), ["a", "b"])
    st.update(np.array([0.9, 0.1]), now=0)
    st.update(np.array([0.9, 0.1]), now=1)
    label, conf = st.smooth(np.array([0.1, 0.9]))  # averaged: (0.9+0.9+0.1)/3 -> still a
    assert label == 0 and conf == pytest.approx(0.6333, abs=1e-3)
    st = PredictionStabilizer(StabilizerConfig(confidence_threshold=0.5, stable_count=1, voting_window=5), ["a", "b"])
    seq = [[0.9, 0.1]] * 3 + [[0.1, 0.9]] * 2
    labels = [st.smooth(np.array(p))[0] for p in seq]
    assert labels == [0] * 5


# --- pipeline ------------------------------------------------------------------------

def test_pipeline_end_to_end_synthetic_20fps():
    cfg = ExperimentConfig()
    cfg.realtime.target_fps = 20
    cfg.stabilizer.stable_count = 2
    cfg.tts.backend = "mock"
    seq = FakeSequence(seq_length=10, probs=(0.8, 0.2), cost_s=0.003)
    tts = TTSWorker(cfg.tts, MockBackend(latency_s=0.01)).start()
    pipe = make_pipeline(cfg, SyntheticExtractor(), seq, tts)
    steps = run_source(cfg, pipe, camera_fps=30.0, duration_s=1.5)
    rep = pipe.report()
    perf = rep["perf"]
    assert steps >= 10 and all(s == (10, 84) for s in seq.calls) and len(seq.calls) >= 10
    assert 17 <= perf["processing_fps_overall"] <= 21.5
    assert perf["camera_fps_overall"] > perf["processing_fps_overall"]
    assert perf["dropped_frames"] > 0  # 30 FPS in, 20 FPS out -> older frames discarded
    assert perf["processing_interval_ms"]["p50"] == pytest.approx(50.0, abs=5.0)
    # static fallback is spoken first (buffer filling), then the motion sign
    assert [e.label for e in pipe.events] == ["flat_hand", "wave"]
    assert rep["speech_events"]["events_created"] == 2
    time.sleep(0.1)
    tts.stop()
    assert rep["tts"]["backend"] == "mock" and tts.spoken_count == 2
    assert tts.stats()["t6_t2_confirmed_to_audio_ms"]["n"] == 2
    for stage in ("capture_ms", "acquire_ms", "frame_age_ms", "schedule_late_ms", "landmark_ms", "window_ms",
                  "static_model_ms", "sequence_model_ms", "model_ms", "probability_ms", "stabilize_ms",
                  "tts_event_ms", "total_ms", "loop_ms", "end_to_end_ms"):
        assert perf[stage]["n"] > 0, stage
    assert perf["end_to_end_ms"]["p50"] >= perf["total_ms"]["p50"]
    assert perf["bottleneck"]["stage"] == "sequence_model_ms"
    assert perf["sustainability"]["sustained"] is True and perf["queue_size"] in (0, 1)
    assert rep["temporal"]["window_span_realtime_s"] == pytest.approx(0.5)


def test_pipeline_reports_unsustained_when_overloaded():
    cfg = ExperimentConfig()
    cfg.realtime.target_fps = 20
    cfg.tts.enabled = False
    slow = SyntheticExtractor(simulated_latency_s=0.09)  # 90 ms > 50 ms interval
    pipe = make_pipeline(cfg, slow, FakeSequence(seq_length=5))
    run_source(cfg, pipe, camera_fps=30.0, duration_s=2.0)
    v = pipe.perf.sustainability()
    assert v["sustained"] is False and "bottleneck landmark_ms" in v["reason"]
    assert pipe.perf.snapshot()["processing_fps_overall"] < 13
    assert pipe.perf.snapshot()["frame_age_ms"]["p50"] < 100  # still processing fresh frames


def test_sync_mode_processes_every_frame():
    """target_fps=0: the original every-frame loop, deterministic."""
    cfg = ExperimentConfig()
    cfg.realtime.target_fps = 0
    cfg.tts.enabled = False

    class Cap:
        def __init__(self, n):
            self.n, self.i = n, 0

        def read(self):
            if self.i >= self.n:
                return False, None
            self.i += 1
            return True, np.zeros((60, 80, 3), np.uint8)
    pipe = make_pipeline(cfg, SyntheticExtractor(), FakeSequence(seq_length=5))
    src = FrameSource(cfg.realtime, pipe.perf, cap=Cap(25)).start()
    n = 0
    while True:
        item = src.next()
        if item is None:
            assert src.ended
            break
        pipe.process(item.frame, item.captured_at)
        n += 1
    assert n == 25 and pipe.perf.snapshot()["frames_processed"] == 25 and pipe.perf.dropped_frames == 0
    assert pipe.perf.sustainability()["sustained"] is None


def test_window_reset_on_hand_loss_and_temporal_gap():
    cfg = ExperimentConfig()
    cfg.realtime.target_fps = 20
    cfg.realtime.max_sequence_gap_intervals = 3
    cfg.stabilizer.max_missing_frames = 2
    cfg.tts.enabled = False
    presence = {"n": 1}
    ex = SyntheticExtractor(hands_present=lambda i: presence["n"])
    pipe = make_pipeline(cfg, ex, FakeSequence(seq_length=10))
    t0 = time.monotonic()
    for i in range(3):
        pipe.process(np.zeros((2, 2, 3), np.uint8), t0 + i * 0.05)
    assert len(pipe.seq_buffer) == 3
    pipe.process(np.zeros((2, 2, 3), np.uint8), t0 + 3 * 0.05 + 0.5)  # 0.5 s stall > 3 x 50 ms
    assert len(pipe.seq_buffer) == 1 and pipe.perf.window_resets == 1
    presence["n"] = 0
    pipe.process(np.zeros((2, 2, 3), np.uint8), t0 + 1.0)
    assert len(pipe.seq_buffer) == 1  # 1 missing frame: keep
    pipe.process(np.zeros((2, 2, 3), np.uint8), t0 + 1.05)
    assert len(pipe.seq_buffer) == 0  # max_missing_frames reached: cleared (app.py MAX_MISSING_FRAMES)


def test_static_only_pipeline_speaks_and_never_blocks_on_tts():
    cfg = ExperimentConfig()
    cfg.realtime.target_fps = 0
    cfg.stabilizer.stable_count = 2
    cfg.tts.backend = "mock"
    tts = TTSWorker(cfg.tts, MockBackend(latency_s=0.3)).start()  # slow speech
    pipe = make_pipeline(cfg, SyntheticExtractor(), None, tts)
    t = time.monotonic()
    for i in range(6):
        pipe.process(np.zeros((2, 2, 3), np.uint8), t + i * 0.01)
    assert time.monotonic() - t < 0.1  # never waited on the 300 ms utterance
    assert [e.label for e in pipe.events] == ["flat_hand"]
    time.sleep(0.5)
    tts.stop()
    assert tts.spoken_count == 1
