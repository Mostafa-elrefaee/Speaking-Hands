#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""camera_check.py -- what does each camera index / backend actually give?

python camera_check.py           # probe indices 0-3 with every backend
python camera_check.py --show    # also pop up a window for each working one
"""
import sys
import time
import argparse

import cv2

BACKENDS = [("DirectShow", cv2.CAP_DSHOW), ("MSMF", cv2.CAP_MSMF),
            ("default", cv2.CAP_ANY)]


def probe(index, name, api, show):
    cap = cv2.VideoCapture(index, api)
    if not cap.isOpened():
        print(f"  index {index} {name:10s}: cannot open")
        return
    got, black, means = 0, 0, []
    end = time.time() + 3.0
    last = None
    while time.time() < end:
        ret, frame = cap.read()
        if ret and frame is not None:
            got += 1
            m = float(frame.mean())
            means.append(m)
            if m < 4.0:
                black += 1
            last = frame
        else:
            time.sleep(0.03)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    if got == 0:
        print(f"  index {index} {name:10s}: opens, but NO frames in 3 s")
        return
    print(f"  index {index} {name:10s}: {got} frames in 3 s, {w}x{h}, "
          f"mean brightness first={means[0]:.1f} last={means[-1]:.1f}, "
          f"black frames={black}/{got}"
          + ("   <-- looks GOOD" if black < got else "   <-- all black"))
    if show and last is not None:
        cv2.imshow(f"index {index} {name}", last)
        cv2.waitKey(1500)
        cv2.destroyAllWindows()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--max_index", type=int, default=4)
    p.add_argument("--show", action="store_true")
    args = p.parse_args()
    try:
        cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_ERROR)
    except AttributeError:
        pass
    print(f"OpenCV {cv2.__version__} on {sys.platform}")
    for index in range(args.max_index):
        print(f"camera index {index}:")
        for name, api in BACKENDS:
            probe(index, name, api, args.show)


if __name__ == "__main__":
    main()
