"""Deterministic scheduler/buffer simulation on a virtual clock.

Drives ``RateScheduler`` + ``FrameBuffer`` with a scripted camera timeline and
a scripted per-frame processing cost, WITHOUT real sleeping, so scheduler
behaviour under stable FPS, irregular FPS, frame bursts and temporary
slowdowns can be verified exactly and quickly.

The simulation is single-threaded: camera frames are "delivered" into the
buffer whenever the virtual clock passes their timestamp. This models a
capture thread that is never starved (true in the real pipeline because the
capture thread only does cap.read() + put()).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Sequence

import numpy as np

from ..evaluation.latency import percentiles
from .scheduler import FrameBuffer, RateScheduler


# --------------------------------------------------------------------------- #
# Camera timelines (capture timestamps in seconds)
# --------------------------------------------------------------------------- #
def stable_timeline(fps: float, duration_s: float) -> np.ndarray:
    return np.arange(0.0, duration_s, 1.0 / fps)


def irregular_timeline(mean_fps: float, duration_s: float, jitter: float = 0.5, seed: int = 0) -> np.ndarray:
    rng = np.random.RandomState(seed)
    base = 1.0 / mean_fps
    gaps = base * rng.uniform(1 - jitter, 1 + jitter, size=int(duration_s * mean_fps * 1.5))
    ts = np.cumsum(gaps)
    return ts[ts < duration_s]


def burst_timeline(fps: float, duration_s: float, every_s: float = 1.0, burst_frames: int = 8) -> np.ndarray:
    """Stable FPS, but every ``every_s`` a burst of ``burst_frames`` arrives within 1 ms."""
    ts = list(stable_timeline(fps, duration_s))
    t = every_s
    while t < duration_s:
        ts.extend(t + np.arange(burst_frames) * 1e-4)
        t += every_s
    return np.array(sorted(ts))


# --------------------------------------------------------------------------- #
# Processing-cost profiles (seconds per processed frame, as a function of time)
# --------------------------------------------------------------------------- #
def constant_cost(ms: float) -> Callable[[float], float]:
    return lambda t: ms / 1e3


def slowdown_cost(base_ms: float, slow_ms: float, start_s: float, end_s: float) -> Callable[[float], float]:
    """Temporary slowdown: cost jumps to ``slow_ms`` between start_s and end_s."""
    return lambda t: (slow_ms if start_s <= t < end_s else base_ms) / 1e3


# --------------------------------------------------------------------------- #
# Simulation
# --------------------------------------------------------------------------- #
@dataclass
class SimResult:
    scenario: str
    target_fps: float
    duration_s: float
    frames_captured: int
    frames_processed: int
    dropped: int
    skipped_stale: int
    reanchors: int
    achieved_fps: float
    interval_ms: Dict[str, float]
    frame_age_ms: Dict[str, float]
    late_ms: Dict[str, float]
    max_backlog: int
    processed_capture_ts: List[float] = field(default_factory=list, repr=False)

    def summary(self) -> str:
        return (f"{self.scenario:38s} cam {self.frames_captured:4d} proc {self.frames_processed:4d} "
                f"drop {self.dropped:4d} stale {self.skipped_stale:3d} reanchor {self.reanchors:2d} | "
                f"achieved {self.achieved_fps:5.2f} fps | interval p50 {self.interval_ms.get('p50', 0):5.1f} "
                f"p95 {self.interval_ms.get('p95', float('nan')):5.1f} | age p50 {self.frame_age_ms.get('p50', 0):5.1f} "
                f"p95 {self.frame_age_ms.get('p95', float('nan')):5.1f} ms | backlog max {self.max_backlog}")


def simulate(scenario: str, capture_ts: Sequence[float], cost_fn: Callable[[float], float],
             target_fps: float = 20.0, max_queue_size: int = 1, latest_frame: bool = True,
             drop_stale: bool = True, stale_max_age_s: float = 0.25, max_catchup_intervals: int = 2) -> SimResult:
    clock = [0.0]
    now = lambda: clock[0]
    def sleep(s):
        clock[0] += s
    buf: FrameBuffer[int] = FrameBuffer(max_queue_size)
    sched = RateScheduler(target_fps, max_catchup_intervals, clock=now)
    capture_ts = list(capture_ts)
    duration = capture_ts[-1] + 1.0 / target_fps if capture_ts else 0.0
    ci = 0
    def deliver():
        nonlocal ci
        while ci < len(capture_ts) and capture_ts[ci] <= clock[0]:
            buf.put(ci, capture_ts[ci]); ci += 1

    proc_times, ages, lates, cap_of_processed = [], [], [], []
    skipped_stale = max_backlog = 0
    while clock[0] < duration:
        late = sched.wait_until_due(sleep)
        deliver()
        max_backlog = max(max_backlog, buf.size())
        item = buf.take(timeout=0, newest=latest_frame)
        if item is None:
            # nothing fresh: advance to the next capture or next deadline, whichever first
            if ci < len(capture_ts):
                clock[0] = max(clock[0], min(capture_ts[ci], clock[0] + sched.interval))
            else:
                clock[0] += sched.interval
            continue
        age = clock[0] - item.captured_at
        if drop_stale and age > stale_max_age_s:
            skipped_stale += 1
            continue
        ages.append(age * 1e3); lates.append(late * 1e3); cap_of_processed.append(item.captured_at)
        proc_times.append(clock[0])
        clock[0] += cost_fn(clock[0])  # the processing itself
        deliver()
    intervals = np.diff(proc_times) * 1e3 if len(proc_times) > 1 else []
    achieved = len(proc_times) / duration if duration > 0 else 0.0
    return SimResult(scenario, target_fps, duration, len(capture_ts), len(proc_times), buf.dropped, skipped_stale,
                     sched.reanchors, achieved, percentiles(intervals), percentiles(ages), percentiles(lates),
                     max_backlog, cap_of_processed)


def standard_scenarios(target_fps: float = 20.0, duration_s: float = 10.0) -> List[SimResult]:
    return [
        simulate("stable 30fps, cost 10ms", stable_timeline(30, duration_s), constant_cost(10), target_fps),
        simulate("stable 60fps, cost 10ms", stable_timeline(60, duration_s), constant_cost(10), target_fps),
        simulate("stable 24fps, cost 10ms", stable_timeline(24, duration_s), constant_cost(10), target_fps),
        simulate("stable 15fps (camera-limited)", stable_timeline(15, duration_s), constant_cost(10), target_fps),
        simulate("irregular ~30fps ±50%, cost 10ms", irregular_timeline(30, duration_s, 0.5), constant_cost(10), target_fps),
        simulate("bursts of 8 every 1s @30fps", burst_timeline(30, duration_s), constant_cost(10), target_fps),
        simulate("slowdown 10->120ms for 2s", stable_timeline(30, duration_s), slowdown_cost(10, 120, 4.0, 6.0), target_fps),
        simulate("overloaded: cost 80ms (>50ms)", stable_timeline(30, duration_s), constant_cost(80), target_fps),
        simulate("overloaded, FIFO queue 5", stable_timeline(30, duration_s), constant_cost(80), target_fps,
                 max_queue_size=5, latest_frame=False),
    ]
