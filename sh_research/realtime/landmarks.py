"""Landmark extraction: the EXISTING Speaking Hands path, wrapped.

``HandsExtractor`` runs MediaPipe Hands (the ``hands`` object app.py already
creates) and ``landmark_utils.build_combined_vector`` -- the same function
extract_gesture_data.py uses to build training data -- and reports the two
sub-stage timings. ``SyntheticExtractor`` is a stand-in for headless
benchmarks/tests (no MediaPipe): it returns random 84-d vectors and fake
per-hand info so the model / stabilizer / TTS stages still run.
"""
from __future__ import annotations

import time
from typing import List, Optional, Tuple

import numpy as np

from landmark_utils import TOTAL_FEATURES, build_combined_vector

# (vector, per_hand_info, timings)
ExtractResult = Tuple[List[float], list, dict]


class HandsExtractor:
    feature_dim = TOTAL_FEATURES
    name = "mediapipe_hands"

    def __init__(self, hands, flip: bool = True) -> None:
        import cv2 as cv
        self._cv = cv
        self.hands = hands
        self.flip = flip
        self.last_results = None

    def extract(self, frame_bgr: np.ndarray) -> Tuple[np.ndarray, ExtractResult]:
        """-> (display_image (flipped BGR), (vector, per_hand_info, timings))"""
        cv = self._cv
        t0 = time.monotonic()
        image = cv.flip(frame_bgr, 1) if self.flip else frame_bgr  # mirror, as app.py always did
        debug_image = image.copy()
        rgb = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        t1 = time.monotonic()
        results = self.hands.process(rgb)
        t2 = time.monotonic()
        rgb.flags.writeable = True
        self.last_results = results
        vec, per_hand_info = build_combined_vector(debug_image, results.multi_hand_landmarks,
                                                   results.multi_handedness)
        t3 = time.monotonic()
        return debug_image, (vec, per_hand_info, {"image_preprocess_ms": (t1 - t0) * 1e3,
                                                  "landmark_ms": (t2 - t1) * 1e3,
                                                  "feature_ms": (t3 - t2) * 1e3})

    def close(self) -> None:
        close = getattr(self.hands, "close", None)
        if close is not None:
            close()


class SyntheticExtractor:
    """Deterministic fake extractor (optionally slow, optionally hand-less)."""

    feature_dim = TOTAL_FEATURES
    name = "synthetic"

    def __init__(self, simulated_latency_s: float = 0.0, seed: int = 0, hands_present=None,
                 feature_dim: int = TOTAL_FEATURES) -> None:
        self.latency = simulated_latency_s
        self._rng = np.random.RandomState(seed)
        self.hands_present = hands_present  # None = always, or callable(frame_idx) -> int hands
        self.feature_dim = feature_dim
        self.calls = 0

    def extract(self, frame_bgr: np.ndarray) -> Tuple[np.ndarray, ExtractResult]:
        t0 = time.monotonic()
        if self.latency > 0:
            time.sleep(self.latency)
        n = 1 if self.hands_present is None else int(self.hands_present(self.calls))
        self.calls += 1
        vec = (self._rng.rand(self.feature_dim) * 2 - 1).astype(np.float32).tolist() if n else [0.0] * self.feature_dim
        lm = [[10 + i * 5, 20 + i * 3] for i in range(21)]
        info = [("Right", lm, [5, 15, 120, 90])][:n]
        t1 = time.monotonic()
        return frame_bgr, (vec, info, {"image_preprocess_ms": 0.0, "landmark_ms": (t1 - t0) * 1e3,
                                       "feature_ms": 0.0})

    def close(self) -> None:
        pass


def make_hands(rt_cfg):
    """Create the MediaPipe Hands object exactly as app.py did (plus the
    configurable model_complexity)."""
    import mediapipe as mp
    return mp.solutions.hands.Hands(
        static_image_mode=rt_cfg.use_static_image_mode,
        max_num_hands=rt_cfg.max_num_hands,
        model_complexity=rt_cfg.model_complexity,
        min_detection_confidence=rt_cfg.min_detection_confidence,
        min_tracking_confidence=rt_cfg.min_tracking_confidence,
    )
