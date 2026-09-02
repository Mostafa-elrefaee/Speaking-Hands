#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
record_video.py -- record short webcam clips for extract_gesture_data.py

USAGE
-----
python record_video.py
    -> asks you interactively for the gesture name, clip length, number of
       clips and camera (press Enter to accept the default shown in [ ]).
       Clips are saved as videos/<label>/<label>_001.mp4, _002, ... and it
       asks whether you want to record another gesture when done.

Command-line arguments still work if you prefer them, e.g.
python record_video.py --label finish --count 5 --seconds 3

Controls while recording:  q = abort

WHY THIS VERSION EXISTS
-----------------------
The previous script did `cap.read()` once, and on Windows the MSMF backend
often fails the first grab (error -1072875772 / 0xC00D3E84) -- the loop
then exited, but the script still printed "Video saved" for a file with
ZERO frames in it. This version:
  * tries DirectShow first on Windows (far more reliable than MSMF), then
    falls back to MSMF / default
  * warms the camera up (discards the first frames) before recording
  * tolerates a few failed grabs instead of quitting on the first one
  * only keeps a file if frames were actually captured; otherwise deletes
    the empty file and exits with an error
  * writes CLEAN frames (the countdown is only drawn on the preview window,
    never into the training video)
  * reports the real number of frames / effective FPS so you know whether a
    clip has enough frames for --seq_length in sequence mode
"""
import os
import sys
import time
import argparse
from pathlib import Path

import cv2

from camera_utils import (open_camera, probe_cameras, MAX_CONSECUTIVE_FAILS)

def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--label", type=str, default=None,
                   help="gesture name; clips go to videos/<label>/<label>_NNN.mp4")
    p.add_argument("--seconds", type=float, default=2.0)
    p.add_argument("--count", type=int, default=1, help="clips to record")
    p.add_argument("--device", type=int, default=0)
    p.add_argument("--width", type=int, default=0, help="0 = camera default")
    p.add_argument("--height", type=int, default=0, help="0 = camera default")
    p.add_argument("--backend", choices=["auto", "dshow", "msmf", "any"],
                   default="auto")
    p.add_argument("--out_dir", type=str, default="videos")
    p.add_argument("--countdown", type=float, default=2.0,
                   help="seconds of 'get ready' before each clip")
    return p.parse_args()


def ask(prompt, default, cast=str, validate=None):
    """Prompt with a default; Enter keeps the default. Re-asks on bad input."""
    while True:
        raw = input(f"{prompt} [{default}]: ").strip()
        if raw == "":
            return default
        try:
            value = cast(raw)
        except ValueError:
            print(f"  please enter a {cast.__name__}")
            continue
        if validate is not None and not validate(value):
            print("  invalid value, try again")
            continue
        return value


def ask_yes_no(prompt, default=True):
    d = "Y/n" if default else "y/N"
    raw = input(f"{prompt} [{d}]: ").strip().lower()
    if raw == "":
        return default
    return raw in ("y", "yes")


def interactive_setup(args):
    """Fill in anything not given on the command line by asking."""
    print("=== Gesture clip recorder ===  (Enter = keep the default)")
    if args.label is None:
        args.label = ask("Gesture name (label)", "hello", str,
                         lambda v: v.replace("_", "").replace("-", "").isalnum())
    args.seconds = ask("Clip length in seconds", args.seconds, float,
                       lambda v: v > 0)
    args.count = ask("How many clips", args.count, int, lambda v: v > 0)
    print("Looking for cameras that give a real (non-black) image...")
    cams = probe_cameras(backend_choice=args.backend)
    if cams:
        for idx, name, w, h in cams:
            print(f"   camera {idx}: {w}x{h} via {name}")
        default_dev = cams[0][0]
    else:
        print("   none found -- check the privacy shutter / camera-off key, "
              "close other apps using the camera, then try anyway")
        default_dev = args.device
    args.device = ask("Camera index", default_dev, int, lambda v: v >= 0)
    print(f"-> {args.count} clip(s) of {args.seconds}s for '{args.label}' "
          f"into {Path(args.out_dir, args.label).resolve()}")
    return args


# ---------------------------------------------------------------------------
# Recording
# ---------------------------------------------------------------------------

def next_free_path(out_dir, label):
    d = Path(out_dir) / label
    d.mkdir(parents=True, exist_ok=True)
    n = 1
    while (d / f"{label}_{n:03d}.mp4").exists() or \
            (d / f"{label}_{n:03d}.avi").exists():
        n += 1
    return d / f"{label}_{n:03d}.mp4"


def open_writer(path, fps, size):
    """mp4v first; MJPG .avi as a fallback that works on every OpenCV
    build."""
    for fourcc_name in ("mp4v", "MJPG"):
        fourcc = cv2.VideoWriter_fourcc(*fourcc_name)
        out_path = path if fourcc_name == "mp4v" else path.with_suffix(".avi")
        writer = cv2.VideoWriter(str(out_path), fourcc, fps, size)
        if writer.isOpened():
            return writer, out_path, fourcc_name
        writer.release()
    raise RuntimeError("Could not create a video writer with any codec.")


def countdown(cap, seconds, text):
    """Preview with a 'get ready' overlay. Returns False if 'q' pressed."""
    end = time.time() + seconds
    while time.time() < end:
        ret, frame = cap.read()
        if not ret or frame is None:
            continue
        preview = frame.copy()
        cv2.putText(preview, f"{text} in {end - time.time():.1f}s", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 200, 255), 2, cv2.LINE_AA)
        cv2.imshow("Recording", preview)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            return False
    return True


def record_clip(cap, out_path, seconds, fps_hint, show=True):
    """Record one clip. Returns the number of frames written (0 = failed,
    and no file is left behind)."""
    frame = None
    for _ in range(MAX_CONSECUTIVE_FAILS):   # first grab may need a retry too
        ret, frame = cap.read()
        if ret and frame is not None:
            break
        time.sleep(0.01)
    if frame is None:
        print("[record] camera stopped delivering frames")
        return 0
    h, w = frame.shape[:2]
    writer, out_path, codec = open_writer(out_path, fps_hint, (w, h))
    print(f"[record] {out_path}  ({w}x{h}, codec {codec}, {seconds}s)")

    frames = 0
    fails = 0
    aborted = False
    start = time.time()
    while True:
        elapsed = time.time() - start
        if elapsed >= seconds:
            break
        ret, frame = cap.read()
        if not ret or frame is None:
            fails += 1
            if fails >= MAX_CONSECUTIVE_FAILS:
                print(f"[record] {fails} consecutive failed grabs, stopping "
                      "this clip early")
                break
            time.sleep(0.01)
            continue
        fails = 0

        writer.write(frame)          # CLEAN frame into the file
        frames += 1

        if show:
            preview = frame.copy()   # overlay only on the preview window
            cv2.putText(preview,
                        f"REC {seconds - elapsed:.1f}s  frames:{frames}",
                        (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2,
                        cv2.LINE_AA)
            cv2.imshow("Recording", preview)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                aborted = True
                break

    wall = time.time() - start
    writer.release()

    if frames == 0 or aborted:
        try:
            os.remove(out_path)
        except OSError:
            pass
        print("[record] aborted by user, file removed" if aborted
              else "[record] ERROR: no frames captured, empty file removed")
        return 0

    eff_fps = frames / wall if wall > 0 else 0.0
    check = cv2.VideoCapture(str(out_path))     # re-open to prove it's readable
    readable = int(check.get(cv2.CAP_PROP_FRAME_COUNT)) if check.isOpened() else 0
    check.release()
    print(f"[record] saved {frames} frames in {wall:.2f}s "
          f"(~{eff_fps:.1f} fps actual; file reports {readable} frames)")
    if eff_fps < 20:
        print("[record] NOTE: actual fps is low -- a 30-frame --seq_length "
              f"window needs ~{30 / max(eff_fps, 1):.1f}s of signing per clip")
    return frames


def record_session(cap, args):
    """Record args.count clips for args.label. Returns how many were saved."""
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps != fps or fps > 120:      # 0, NaN or nonsense
        fps = 30.0

    recorded = 0
    for i in range(args.count):
        out_path = next_free_path(args.out_dir, args.label)
        if not countdown(cap, args.countdown,
                         f"clip {i + 1}/{args.count} '{args.label}'"):
            print("[record] stopped by user")
            break
        if record_clip(cap, out_path, args.seconds, fps) > 0:
            recorded += 1

    print(f"Done: {recorded}/{args.count} clip(s) saved under "
          f"{Path(args.out_dir, args.label).resolve()}")
    return recorded


def main():
    args = get_args()
    used_cli = args.label is not None      # given on the command line?
    if not used_cli:
        args = interactive_setup(args)

    cap = None
    total = 0
    try:
        cap, _backend = open_camera(args.device, args.backend, args.width,
                                    args.height)
        while True:
            total += record_session(cap, args)
            if used_cli:
                break
            if not ask_yes_no("Record another gesture?", default=True):
                break
            args.label = None
            cv2.destroyAllWindows()
            args = interactive_setup(args)   # camera stays open
    except KeyboardInterrupt:
        print("\n[record] interrupted")
    finally:
        if cap is not None:
            cap.release()
        cv2.destroyAllWindows()

    if total == 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
