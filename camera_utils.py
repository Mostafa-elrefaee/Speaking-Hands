#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
camera_utils.py -- open a webcam RELIABLY (shared by app.py and record_video.py)

OpenCV on Windows has two common failure modes that a plain
cv2.VideoCapture(0) doesn't protect you from:
  * MSMF backend "opens" the camera but every grab fails
    (error -1072875772 / 0xC00D3E84)
  * DirectShow (or an infrared camera at index 0) returns frames that are
    all black
open_camera() tries DirectShow -> MSMF -> default on Windows, waits until a
frame with an actual image arrives, and only then hands the camera back.
"""
import sys
import time

import cv2

WARMUP_FRAMES = 15          # frames discarded before recording starts
WARMUP_SECONDS = 3.0        # max time to wait for a real (non-black) image
BLACK_MEAN_THRESHOLD = 4.0  # mean pixel value below this = black frame
MAX_CONSECUTIVE_FAILS = 30  # give up if this many grabs fail in a row (~1 s)
OPEN_RETRIES = 3            # attempts per backend when opening the camera




def _backend_list(choice):
    if choice == "dshow":
        return [("DirectShow", cv2.CAP_DSHOW)]
    if choice == "msmf":
        return [("MSMF", cv2.CAP_MSMF)]
    if choice == "any":
        return [("default", cv2.CAP_ANY)]
    if sys.platform.startswith("win"):
        return [("DirectShow", cv2.CAP_DSHOW), ("MSMF", cv2.CAP_MSMF),
                ("default", cv2.CAP_ANY)]
    return [("default", cv2.CAP_ANY)]


def frame_is_black(frame):
    """True for the all-zero frames some backends/IR cameras deliver."""
    return frame is None or float(frame.mean()) < BLACK_MEAN_THRESHOLD


def _warm_up(cap):
    """Read frames for up to WARMUP_SECONDS until a REAL image arrives.
    Returns (got_any_frame, got_real_image)."""
    got_any = False
    end = time.time() + WARMUP_SECONDS
    n = 0
    while time.time() < end:
        ret, frame = cap.read()
        if ret and frame is not None:
            got_any = True
            n += 1
            # give auto-exposure a few frames, then accept the first real image
            if n >= WARMUP_FRAMES and not frame_is_black(frame):
                return True, True
        else:
            time.sleep(0.03)
    return got_any, False


def open_camera(device, backend_choice, width=0, height=0, quiet=False):
    """Open the camera with the first backend that actually delivers a
    REAL image. isOpened() alone is not enough (MSMF may 'open' and then
    fail every grab), and a returned frame is not enough either
    (DirectShow may return all-black frames on some drivers)."""
    for name, api in _backend_list(backend_choice):
        for attempt in range(1, OPEN_RETRIES + 1):
            cap = cv2.VideoCapture(device, api)
            if width:
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            if height:
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            if cap.isOpened():
                got_any, got_real = _warm_up(cap)
                if got_real:
                    if not quiet:
                        print(f"[camera] device {device} via {name} "
                              f"(attempt {attempt})")
                    return cap, name
                if got_any and not quiet:
                    print(f"[camera] {name}: frames arrive but are BLACK, "
                          "trying next backend")
            cap.release()
            time.sleep(0.3)
        if not quiet:
            print(f"[camera] {name}: no usable image after {OPEN_RETRIES} "
                  "attempts")
    raise RuntimeError(
        f"Could not get a real image from camera {device} with any backend.\n"
        "  Common causes on Windows:\n"
        "   - wrong camera index: device 0 is often the infrared/Windows "
        "Hello camera (black). Try --device 1 (or pick it in the menu)\n"
        "   - physical privacy shutter closed, or camera-off key (Fn + "
        "camera icon)\n"
        "   - another program is using the camera (close app.py, browser "
        "tabs, Teams/Zoom, the Camera app)\n"
        "   - Windows Settings > Privacy & security > Camera: allow desktop "
        "apps\n"
        "   - try --backend msmf or --backend dshow explicitly")


def probe_cameras(max_index=4, backend_choice="auto"):
    """Return a list of (index, backend, width, height) for every camera
    index that delivers a real image."""
    found = []
    for idx in range(max_index):
        try:
            cap, name = open_camera(idx, backend_choice, quiet=True)
        except RuntimeError:
            continue
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        found.append((idx, name, w, h))
    return found


