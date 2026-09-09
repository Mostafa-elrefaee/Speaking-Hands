"""Debug overlay for the preview window (``realtime.debug_overlay_enabled``).

Everything drawn comes from ``PerfTracker.overlay_lines()`` — i.e. measured
values. Headless callers can use ``format_lines`` for the console instead.
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np


def format_lines(perf, prediction: Optional[str] = None, confidence: Optional[float] = None) -> List[str]:
    lines = []
    if prediction is not None:
        lines.append(f"{prediction} ({confidence:.2f})" if confidence is not None else prediction)
    lines.extend(perf.overlay_lines())
    return lines


def draw_overlay(img: np.ndarray, perf, prediction: Optional[str] = None, confidence: Optional[float] = None,
                 origin=(10, 24), line_h: int = 20) -> np.ndarray:
    import cv2
    x, y = origin
    for i, line in enumerate(format_lines(perf, prediction, confidence)):
        scale, color, thick = (0.8, (0, 255, 0), 2) if i == 0 and prediction is not None else (0.5, (255, 255, 255), 1)
        yy = y + i * line_h + (6 if i else 0)
        cv2.putText(img, line, (x + 1, yy + 1), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thick + 1, cv2.LINE_AA)
        cv2.putText(img, line, (x, yy), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)
    return img
