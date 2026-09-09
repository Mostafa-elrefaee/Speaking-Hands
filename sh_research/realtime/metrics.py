"""Measured (never estimated) realtime performance statistics.

Every latency is a difference of ``time.monotonic()`` readings taken around
the actual stage. FPS values are computed from timestamps of events that
really happened. ``bottleneck()`` names the stage with the largest median
cost and ``sustainability()`` states whether the target rate was actually
held, judged from the measured processing rate -- never from the target.

Stage names (all milliseconds), matching app.py / GesturePipeline:

    capture_ms            cap.read() duration in the capture thread
    acquire_ms            buffer take() duration in the processing loop
    frame_age_ms          capture -> start of processing (queueing delay)
    schedule_late_ms      how late the scheduler tick fired vs. its deadline
    image_preprocess_ms   flip + BGR->RGB before MediaPipe
    landmark_ms           MediaPipe Hands .process()
    feature_ms            landmark_utils.build_combined_vector (84-d vector)
    window_ms             sequence-buffer update + stacking (+ normalisation)
    static_model_ms       KeyPointClassifier TFLite forward pass
    sequence_model_ms     SequenceClassifier TFLite forward pass
    model_ms              static + sequence
    probability_ms        probability smoothing / gate (PredictionStabilizer.smooth)
    stabilize_ms          arbitration + debounce decision (choose_gesture + commit)
    tts_event_ms          speech-event creation + non-blocking enqueue
    total_ms              ML processing for one frame (preprocess .. tts_event)
    draw_ms               UI drawing + imshow (outside total_ms, inside loop_ms)
    loop_ms               total_ms + draw_ms: what one tick really costs
    end_to_end_ms         capture -> prediction available
    (TTS generation / playback-start live in TTSWorker.stats())
"""
from __future__ import annotations

import json
import time
from collections import deque
from pathlib import Path
from typing import Deque, Dict, List, Optional, Tuple

from ..evaluation.latency import percentiles

STAGES: Tuple[str, ...] = (
    "capture_ms", "acquire_ms", "frame_age_ms", "schedule_late_ms", "image_preprocess_ms",
    "landmark_ms", "feature_ms", "window_ms", "static_model_ms", "sequence_model_ms", "model_ms",
    "probability_ms", "stabilize_ms", "tts_event_ms", "total_ms", "draw_ms", "loop_ms", "end_to_end_ms",
)
# Stages that are components of loop_ms (candidates for "the bottleneck").
COMPONENT_STAGES: Tuple[str, ...] = (
    "acquire_ms", "image_preprocess_ms", "landmark_ms", "feature_ms", "window_ms",
    "static_model_ms", "sequence_model_ms", "probability_ms", "stabilize_ms", "tts_event_ms", "draw_ms",
)


class RollingRate:
    """Events per second over the last ``window_s`` seconds."""

    def __init__(self, window_s: float = 2.0) -> None:
        self.window_s = window_s
        self._ts: Deque[float] = deque()
        self.count = 0

    def tick(self, t: Optional[float] = None) -> None:
        t = time.monotonic() if t is None else t
        self._ts.append(t)
        self.count += 1
        cutoff = t - self.window_s
        while self._ts and self._ts[0] < cutoff:
            self._ts.popleft()

    @property
    def fps(self) -> float:
        if len(self._ts) < 2:
            return 0.0
        span = self._ts[-1] - self._ts[0]
        return (len(self._ts) - 1) / span if span > 0 else 0.0


class PerfTracker:
    def __init__(self, window: int = 500, target_fps: float = 20.0, enabled: bool = True) -> None:
        self.window = window
        self.target_fps = target_fps
        self.enabled = enabled
        self.samples: Dict[str, Deque[float]] = {s: deque(maxlen=window) for s in STAGES}
        self.camera = RollingRate()
        self.processing = RollingRate()
        self.intervals: Deque[float] = deque(maxlen=window)  # processing interval ms
        self.dropped_frames = 0
        self.skipped_stale = 0
        self.window_resets = 0
        self.queue_size = 0
        self.started_at = time.monotonic()
        self._last_proc: Optional[float] = None

    # --- recording -----------------------------------------------------------
    def record_capture(self, t: Optional[float] = None, capture_ms: Optional[float] = None) -> None:
        self.camera.tick(t)
        if self.enabled and capture_ms is not None:
            self.samples["capture_ms"].append(capture_ms)

    def record_processing(self, stages: Dict[str, float], t: Optional[float] = None) -> None:
        t = time.monotonic() if t is None else t
        self.processing.tick(t)
        if self._last_proc is not None:
            self.intervals.append((t - self._last_proc) * 1000.0)
        self._last_proc = t
        if not self.enabled:
            return
        for k, v in stages.items():
            if k in self.samples:
                self.samples[k].append(v)

    def record_draw(self, draw_ms: float) -> None:
        """UI cost of the tick just processed (adds loop_ms = total + draw)."""
        if not self.enabled:
            return
        self.samples["draw_ms"].append(draw_ms)
        if self.samples["total_ms"]:
            self.samples["loop_ms"].append(self.samples["total_ms"][-1] + draw_ms)

    # --- analysis ------------------------------------------------------------
    def bottleneck(self) -> Optional[Dict[str, float]]:
        """Component stage with the largest median cost, with its share of the loop."""
        best, best_p50 = None, -1.0
        for k in COMPONENT_STAGES:
            d = self.samples[k]
            if len(d) >= 5:
                p = percentiles(d)["p50"]
                if p > best_p50:
                    best, best_p50 = k, p
        if best is None:
            return None
        ref = self.samples["loop_ms"] if len(self.samples["loop_ms"]) >= 5 else self.samples["total_ms"]
        total = percentiles(ref)["p50"] if len(ref) >= 5 else 0.0
        return {"stage": best, "p50_ms": best_p50, "share_of_total": (best_p50 / total) if total > 0 else 0.0}

    def sustainability(self) -> Dict[str, object]:
        """Was the target rate actually held? Judged from measurements only."""
        up = time.monotonic() - self.started_at
        overall = self.processing.count / up if up > 0 else 0.0
        interval_ms = 1000.0 / self.target_fps if self.target_fps > 0 else 0.0
        total = percentiles(self.samples["total_ms"]) if len(self.samples["total_ms"]) >= 5 else {}
        loop = percentiles(self.samples["loop_ms"]) if len(self.samples["loop_ms"]) >= 5 else total
        cam_overall = self.camera.count / up if up > 0 else 0.0
        limited_by_camera = cam_overall > 0 and cam_overall < self.target_fps * 0.98
        verdict: Dict[str, object] = {
            "target_fps": self.target_fps,
            "achieved_fps": overall,
            "achieved_ratio": overall / self.target_fps if self.target_fps else 0.0,
            "target_interval_ms": interval_ms,
            "processing_total_p50_ms": total.get("p50"),
            "processing_total_p95_ms": total.get("p95"),
            "loop_p50_ms": loop.get("p50"),
            "camera_limited": limited_by_camera,
        }
        if self.target_fps <= 0:
            verdict["sustained"] = None
            verdict["reason"] = "no target rate (every captured frame is processed)"
            return verdict
        # Sustained := achieved >= 95% of target AND median tick cost fits in the interval.
        sustained = overall >= 0.95 * self.target_fps and (loop.get("p50") is None or loop["p50"] < interval_ms)
        verdict["sustained"] = bool(sustained)
        if not sustained:
            if limited_by_camera:
                verdict["reason"] = f"camera delivered only {cam_overall:.1f} FPS"
            elif loop.get("p50") is not None and loop["p50"] >= interval_ms:
                bn = self.bottleneck()
                verdict["reason"] = (f"median tick {loop['p50']:.1f} ms >= interval {interval_ms:.1f} ms"
                                     + (f"; bottleneck {bn['stage']} {bn['p50_ms']:.1f} ms" if bn else ""))
            else:
                verdict["reason"] = "achieved rate below 95% of target (scheduling/jitter or short run)"
        return verdict

    # --- reporting -----------------------------------------------------------
    def snapshot(self) -> Dict[str, object]:
        up = time.monotonic() - self.started_at
        out: Dict[str, object] = {
            "profiling_enabled": self.enabled,
            "uptime_s": up,
            "camera_fps_rolling": self.camera.fps,
            "processing_fps_rolling": self.processing.fps,
            "camera_fps_overall": self.camera.count / up if up > 0 else 0.0,
            "processing_fps_overall": self.processing.count / up if up > 0 else 0.0,
            "frames_captured": self.camera.count,
            "frames_processed": self.processing.count,
            "dropped_frames": self.dropped_frames,
            "skipped_stale_frames": self.skipped_stale,
            "window_resets": self.window_resets,
            "queue_size": self.queue_size,
            "processing_interval_ms": percentiles(self.intervals),
        }
        for k, d in self.samples.items():
            if len(d):
                out[k] = percentiles(d)
        out["bottleneck"] = self.bottleneck()
        out["sustainability"] = self.sustainability()
        return out

    def overlay_lines(self) -> List[str]:
        """Short human-readable lines for the debug overlay / console."""
        s = self.snapshot()

        def p50(k):
            d = s.get(k)
            return f"{d['p50']:.1f} ms" if d and d.get("n") else "-"
        target = f" (target {self.target_fps:.0f})" if self.target_fps > 0 else " (every frame)"
        lines = [
            f"Camera FPS: {s['camera_fps_rolling']:.1f}",
            f"Processing FPS: {s['processing_fps_rolling']:.1f}{target}",
            f"Model: {p50('model_ms')}",
            f"Landmarks: {p50('landmark_ms')}",
            f"Latency: {p50('total_ms')} proc / {p50('end_to_end_ms')} e2e",
            f"Dropped: {s['dropped_frames']}  Queue: {s['queue_size']}",
        ]
        bn = s["bottleneck"]
        if bn:
            lines.append(f"Bottleneck: {bn['stage'].replace('_ms', '')} ({bn['share_of_total']:.0%})")
        return lines

    def one_line(self) -> str:
        return " | ".join(self.overlay_lines())

    def dump(self, path: Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.snapshot(), indent=2))
        return path
