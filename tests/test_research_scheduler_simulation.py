"""Prompt-2 §15 validation: scheduler + buffer under simulated frame timelines."""
import pytest

from sh_research.realtime import FrameBuffer
from sh_research.realtime.simulate import (burst_timeline, constant_cost, irregular_timeline, simulate, slowdown_cost,
                                          stable_timeline)


def test_frame_buffer_bounded_fifo_and_newest():
    b = FrameBuffer(3)
    for i in range(5):
        b.put(i, captured_at=float(i))
    assert b.size() == 3 and b.dropped == 2  # 0,1 evicted
    assert b.take(timeout=0, newest=False).frame == 2  # FIFO
    assert b.take(timeout=0, newest=True).frame == 4 and b.dropped == 3  # newest, 3 discarded
    assert b.take(timeout=0) is None
    with pytest.raises(ValueError):
        FrameBuffer(0)


@pytest.mark.parametrize("cam_fps", [24, 30, 60])
def test_stable_camera_holds_20fps(cam_fps):
    r = simulate("stable", stable_timeline(cam_fps, 10), constant_cost(10), 20.0)
    assert abs(r.achieved_fps - 20.0) < 0.2
    assert r.interval_ms["p50"] == pytest.approx(50.0, abs=0.1) and r.interval_ms["p95"] == pytest.approx(50.0, abs=0.1)
    assert r.dropped == r.frames_captured - r.frames_processed  # every unprocessed frame is accounted for
    assert r.frame_age_ms["max"] <= 1000 / cam_fps + 0.1  # never older than one camera interval
    assert r.max_backlog == 1 and r.reanchors == 0


def test_camera_slower_than_target_is_reported_honestly():
    r = simulate("cam15", stable_timeline(15, 10), constant_cost(10), 20.0)
    assert abs(r.achieved_fps - 15.0) < 0.2 and r.dropped == 0


def test_irregular_and_bursty_cameras():
    r = simulate("irregular", irregular_timeline(30, 10, 0.5), constant_cost(10), 20.0)
    assert abs(r.achieved_fps - 20.0) < 0.3 and r.interval_ms["p95"] == pytest.approx(50.0, abs=0.1)
    b = simulate("burst", burst_timeline(30, 10), constant_cost(10), 20.0)
    assert abs(b.achieved_fps - 20.0) < 0.2 and b.max_backlog == 1 and b.frame_age_ms["p95"] < 40


def test_temporary_slowdown_recovers_without_backlog():
    r = simulate("slow", stable_timeline(30, 10), slowdown_cost(10, 120, 4.0, 6.0), 20.0)
    before = [t for t in r.processed_capture_ts if t < 4.0]
    after = [t for t in r.processed_capture_ts if t >= 6.5]
    assert len(before) / 4.0 == pytest.approx(20.0, abs=0.5)
    assert len(after) / 3.5 == pytest.approx(20.0, abs=0.6)  # back to target after the slowdown
    assert r.reanchors > 0 and r.max_backlog == 1
    assert r.frame_age_ms["max"] < 60  # never processed a frame from seconds ago


def test_overload_reports_real_rate_and_keeps_frames_fresh():
    latest = simulate("overload", stable_timeline(30, 10), constant_cost(80), 20.0)
    assert 12 <= latest.achieved_fps <= 13 and latest.interval_ms["p50"] == pytest.approx(80.0, abs=0.5)
    assert latest.frame_age_ms["p95"] < 40
    fifo = simulate("overload-fifo", stable_timeline(30, 10), constant_cost(80), 20.0, max_queue_size=5, latest_frame=False)
    assert fifo.frame_age_ms["p50"] > 100  # bounded FIFO trades freshness for order
    assert fifo.max_backlog == 5 and fifo.achieved_fps == pytest.approx(latest.achieved_fps, abs=0.1)
