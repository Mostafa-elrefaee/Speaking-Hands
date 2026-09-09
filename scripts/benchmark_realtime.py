#!/usr/bin/env python3
"""Headless realtime benchmark: the SAME FrameSource + GesturePipeline app.py
runs, without a window, fed by a synthetic camera at a chosen rate.

  # real TFLite models (the deployed ones), synthetic landmarks (no MediaPipe):
  python scripts/benchmark_realtime.py --out experiments/realtime --duration 20
  # real MediaPipe Hands on generated frames (measures landmark_ms on hand-less
  # images -- a LOWER bound; real hands cost more):
  python scripts/benchmark_realtime.py --extractor mediapipe --duration 20
  # a research run's model + preprocessor, 15 FPS, mock TTS:
  python scripts/benchmark_realtime.py --seq_model experiments/<run> --target_fps 15 --tts mock

For numbers with real hands use  python app.py --benchmark --duration 60
(same report format) -- it needs a camera and someone signing in front of it.
"""
import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np

from _common import add_config_args, build_config
from sh_research.experiments.metadata import benchmark_metadata


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_config_args(p)
    p.add_argument("--out", default="experiments/realtime")
    p.add_argument("--duration", type=float, default=20.0)
    p.add_argument("--camera_fps", type=float, default=30.0)
    p.add_argument("--target_fps", type=float, default=None)
    p.add_argument("--extractor", choices=["synthetic", "mediapipe"], default="synthetic")
    p.add_argument("--frame", default="960x540", help="synthetic frame size WxH")
    p.add_argument("--seq_model", default=None, help=".tflite or research run dir (default: deployed model)")
    p.add_argument("--seq_label", default=None)
    p.add_argument("--tts", choices=["pyttsx3_subprocess", "pyttsx3", "print", "mock", "off"], default="mock")
    p.add_argument("--stable_frames", type=int, default=None)
    a = p.parse_args()

    cfg = build_config(a)
    if a.target_fps is not None:
        cfg.realtime.target_fps = a.target_fps
    if a.stable_frames is not None:
        cfg.stabilizer.stable_count = a.stable_frames
    if a.tts == "off":
        cfg.tts.enabled = False
    else:
        cfg.tts.backend = a.tts
    cfg.realtime.show_window = False

    import app  # the real app: same loaders, same pipeline
    from gesture_output import PredictionStabilizer, SpeechWorker
    from model import KeyPointClassifier
    from sh_research.realtime import FrameSource, GesturePipeline, HandsExtractor, PerfTracker, SyntheticCamera, SyntheticExtractor, make_hands
    from sh_research.tts.events import SpeechEventManager

    keypoint = KeyPointClassifier()
    from dataset_utils import load_labels
    kp_labels = load_labels("model/keypoint_classifier/keypoint_classifier_label.csv")
    seq_model, seq_label, pre = app.resolve_sequence_model(a.seq_model or app.SEQ_MODEL_PATH, a.seq_label or app.SEQ_LABEL_PATH)
    seq_clf, seq_labels = app.load_sequence_classifier(seq_model, seq_label, cfg.stabilizer.confidence_threshold)
    if seq_clf is None:
        pre = None

    w, h = (int(v) for v in a.frame.lower().split("x"))
    if a.extractor == "mediapipe":
        extractor = HandsExtractor(make_hands(cfg.realtime))
        rng = np.random.RandomState(0)
        def frame_fn(i):  # noise frames so MediaPipe does real work (no hands will be found)
            return rng.randint(0, 255, size=(h, w, 3), dtype=np.uint8)
    else:
        extractor = SyntheticExtractor(hands_present=lambda i: 1 if (i % 60) < 45 else 0)  # 0.75 s sign, 0.25 s rest at 60-frame cycles
        frame_fn = None

    stabilizer = PredictionStabilizer(cfg.stabilizer, seq_labels or kp_labels)
    speaker = SpeechWorker(cfg=cfg.tts).start() if cfg.tts.enabled else None
    perf = PerfTracker(cfg.realtime.latency_window, cfg.realtime.target_fps, True)
    pipeline = GesturePipeline(cfg, extractor, keypoint, kp_labels, seq_clf, seq_labels, stabilizer,
                               SpeechEventManager(cfg.tts), speaker, pre, perf)
    source = FrameSource(cfg.realtime, perf, thread=SyntheticCamera(None, fps=a.camera_fps, shape=(h, w, 3), frame_fn=frame_fn)).start()
    print(f"running {a.duration:.0f}s: camera {a.camera_fps} FPS -> target {cfg.realtime.target_fps} FPS, extractor={a.extractor}, "
          f"seq={'yes' if seq_clf else 'no'}, tts={cfg.tts.backend if cfg.tts.enabled else 'off'}")
    t0 = time.monotonic()
    try:
        while time.monotonic() - t0 < a.duration:
            item = source.next()
            if item is None:
                if source.ended:
                    break
                continue
            pipeline.process(item.frame, item.captured_at, item.late_s, item.acquire_ms)
            perf.record_draw(0.0)
    finally:
        source.close()
        report = pipeline.report()
        if speaker is not None:
            time.sleep(0.3)
            report["tts"] = speaker.stats()
            speaker.close()
        pipeline.close()

    meta = {}
    mp = Path(seq_model).parent / "metadata.json"
    if mp.exists():
        meta = json.loads(mp.read_text(encoding="utf-8"))
    report["metadata"] = benchmark_metadata(cfg, meta, actual_fps=report["perf"]["processing_fps_overall"],
                                            extractor=f"{a.extractor} (headless benchmark)", model_dir=str(Path(seq_model).parent))
    report["metadata"]["synthetic_camera_fps"] = a.camera_fps
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    path = out / f"{stamp}_headless_{a.extractor}_realtime_benchmark.json"
    path.write_text(json.dumps(report, indent=2, default=float, ensure_ascii=False), encoding="utf-8")
    if speaker is not None and speaker.records:
        rows = speaker.records_as_rows()
        with open(path.with_name(f"{stamp}_headless_{a.extractor}_speech_events.csv"), "w", newline="", encoding="utf-8") as fh:
            wr = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            wr.writeheader()
            wr.writerows(rows)
    report["target_fps"] = cfg.realtime.target_fps
    app.print_report(report)
    print(f"-> {path}")


if __name__ == "__main__":
    main()
