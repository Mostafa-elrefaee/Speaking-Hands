"""Prediction-to-speech benchmark.

Drives the *real* speech subsystem -- gesture_output.PredictionStabilizer ->
SpeechEventManager -> bounded queue -> TTSWorker -> backend -- with a
scripted stream of model predictions delivered at ``target_fps`` in real
time, so every T2..T7 timestamp is measured on the actual backend and thread
hand-offs. No landmark extraction or model inference is involved (those come
from scripts/benchmark_realtime.py / app.py --benchmark).

Prediction stream: signs are "held" for ``hold_s`` seconds each, cycling
through the classes, with a low-confidence gap between signs so the
stabilizer sees a release. Each held sign yields exactly one speech event
(change detection), so ``n_events`` controls the sample count.

Exports ``<out>/tts_benchmark.json`` (metadata + statistics) and
``<out>/tts_benchmark_events.csv`` (one row per spoken event, T0..T7 + deltas).

The default backend (pyttsx3_subprocess) produces AUDIO and needs a working
pyttsx3 driver (SAPI5 on Windows, NSSpeechSynthesizer on macOS, eSpeak on
Linux). Use ``--tts mock`` to measure only the queue / thread overhead.
"""
from __future__ import annotations

import csv
import json
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from gesture_output import PredictionStabilizer

from ..config import ExperimentConfig
from ..evaluation.latency import percentiles
from ..experiments.metadata import benchmark_metadata
from .backends import TTSBackend
from .events import SpeechEventManager
from .worker import DELTA_KEYS, TTSWorker


def prediction_stream(n_classes: int, n_events: int, target_fps: float, hold_s: float = 0.6, gap_s: float = 0.3,
                      confidence: float = 0.95, noise: float = 0.02, seed: int = 0):
    """Yield (probs, is_gap) at the target rate; n_events held signs."""
    rng = np.random.RandomState(seed)
    hold_frames, gap_frames = int(round(hold_s * target_fps)), int(round(gap_s * target_fps))
    for k in range(n_events):
        c = k % n_classes
        for _ in range(hold_frames):
            p = np.full(n_classes, (1 - confidence) / max(n_classes - 1, 1))
            p[c] = confidence
            p = np.clip(p + noise * rng.randn(n_classes), 1e-4, None)
            yield p / p.sum(), False
        for _ in range(gap_frames):
            yield rng.dirichlet(np.ones(n_classes) * 5.0), True  # nothing confident


def run_tts_benchmark(cfg: ExperimentConfig, class_names: List[str], out_dir: Path, n_events: int = 30,
                      backend: Optional[TTSBackend] = None, hold_s: float = 0.6, gap_s: float = 0.3,
                      label: str = "") -> Dict[str, object]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stab = PredictionStabilizer(cfg.stabilizer, class_names)
    mgr = SpeechEventManager(cfg.tts)
    worker = TTSWorker(cfg.tts, backend).start()
    fps = cfg.realtime.target_fps or 20.0
    interval = 1.0 / fps
    stabilize_ms: List[float] = []
    t_start = time.monotonic()
    next_t = t_start
    n_pred = 0
    for probs, _ in prediction_stream(len(class_names), n_events, fps, hold_s, gap_s):
        delay = next_t - time.monotonic()
        if delay > 0:
            time.sleep(delay)
        now = time.monotonic()
        t0 = now - interval  # a frame "captured" one interval earlier (synthetic)
        ts = time.monotonic()
        ev = stab.update(probs, ts, frame_captured_at=t0, predicted_at=now)
        sp = mgr.on_validated(ev, time.monotonic()) if ev is not None else None
        if sp is None:
            sp = mgr.tick(time.monotonic(), stab.current)
        stabilize_ms.append((time.monotonic() - ts) * 1e3)
        if sp is not None:
            worker.enqueue(sp)
        n_pred += 1
        next_t += interval
    elapsed = time.monotonic() - t_start
    deadline = time.monotonic() + 10.0  # let the last utterances finish
    while (worker.pending or worker.is_speaking) and time.monotonic() < deadline:
        time.sleep(0.02)
    worker.stop()

    stats = worker.stats()
    rows = worker.records_as_rows()
    result: Dict[str, object] = {
        "label": label,
        "metadata": benchmark_metadata(cfg, actual_fps=n_pred / elapsed if elapsed > 0 else None,
                                       extractor="none (tts benchmark: scripted predictions)"),
        "stream": {"n_events_requested": n_events, "predictions_delivered": n_pred, "hold_s": hold_s, "gap_s": gap_s,
                   "delivered_fps": n_pred / elapsed if elapsed > 0 else None, "duration_s": elapsed},
        "speech_events": mgr.stats(),
        "stabilizer": {"events": stab.events_emitted, "suppressed": stab.suppressed},
        "tts": stats,
        "stabilization_ms": percentiles(stabilize_ms),
        "latency": {k: stats[k] for k in DELTA_KEYS},
        "primary_metric_confirmed_to_audio_ms": stats["t6_t2_confirmed_to_audio_ms"],
    }
    (out_dir / "tts_benchmark.json").write_text(json.dumps(result, indent=2, default=float, ensure_ascii=False),
                                                encoding="utf-8")
    if rows:
        with open(out_dir / "tts_benchmark_events.csv", "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
    result["paths"] = {"json": str(out_dir / "tts_benchmark.json"), "csv": str(out_dir / "tts_benchmark_events.csv")}
    return result


def summary_line(res: Dict[str, object]) -> str:
    lat, t = res["latency"], res["tts"]

    def p(d, k="p50"):
        return f"{d[k]:.1f}" if d.get("n") and k in d else "-"
    spawn = t.get("process_spawn_ms")
    return (f"{res.get('label', ''):24s} events {t['spoken']:3d} spoken / {t['requested']:3d} requested "
            f"(dup {res['speech_events']['suppressed_duplicates']}, exp {t['expired']}, drop {t['dropped_overflow']}, "
            f"fail {t['failed']}) | prewarm {t['prewarm_ms']:.0f} ms" if t['prewarm_ms'] is not None else
            f"{res.get('label', ''):24s} events {t['spoken']:3d} spoken / {t['requested']:3d} requested | prewarm -") + (
            f" | spawn p50 {p(spawn)}" if spawn else "") + (
            f" | queue-wait p50 {p(lat['t4_t3_queue_wait_ms'])} | synth p50 {p(lat['t5_t4_synthesis_ms'])} | "
            f"confirmed->audio p50 {p(lat['t6_t2_confirmed_to_audio_ms'])} p95 {p(lat['t6_t2_confirmed_to_audio_ms'], 'p95')} "
            f"max {p(lat['t6_t2_confirmed_to_audio_ms'], 'max')} ms")
