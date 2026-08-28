#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
extract_gesture_data.py

Feed this script a video of someone performing ONE gesture (one or two
hands), and it will:
  1. Read every frame (or every Nth frame) of the video
  2. Run MediaPipe Hands to find up to 2 hands' worth of landmarks
  3. Combine BOTH hands into a single 84-value vector per frame (via
     landmark_utils.build_combined_vector -- the exact same function
     app.py uses at inference time, so the two can never disagree)
  4. Append one labeled row per frame to a clean CSV

This means a gesture that needs both hands together (e.g. hands touching,
or one hand modifying what the other is doing) is learned as ONE gesture,
not as two separate single-hand predictions.

Two output modes:

  --mode static   (default)
      One row per frame: [label_id, <84 combined values>]
      -> model/keypoint_classifier/keypoint.csv
      For a single still hand-shape/pose -> one label.

  --mode sequence
      One row per SEQ_LENGTH-frame window:
      [label_id, frame1's 84 values, frame2's 84 values, ...]
      -> model/sequence_classifier/sequence_data.csv
      For a gesture that involves motion over time -> one label.

It also auto-manages a label CSV: give it a gesture name (e.g. "hello"),
and it looks up (or creates) a numeric ID for that name, so you never
hand-manage label numbers yourself.

USAGE EXAMPLES
--------------
# One video, one gesture (uses both hands automatically if both appear)
python extract_gesture_data.py --video videos/hello.mp4 --label hello

# Motion sequences (for the LSTM/GRU model)
python extract_gesture_data.py --video videos/hello.mp4 --label hello --mode sequence --seq_length 30

# Several clips of the SAME gesture, all in one folder
python extract_gesture_data.py --video_dir videos/hello_clips/ --label hello

# Each video in a folder is its OWN gesture (filename = label)
python extract_gesture_data.py --video_dir videos/
"""

import argparse
import csv
import itertools
import os
from pathlib import Path

import cv2 as cv
import mediapipe as mp

from landmark_utils import build_combined_vector, TOTAL_FEATURES


def get_args():
    parser = argparse.ArgumentParser(
        description="Extract two-hand-combined gesture data from a video."
    )

    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--video", type=str,
                      help="Path to a single video file for one gesture.")
    src.add_argument("--video_dir", type=str,
                      help="Folder of videos. By default each file's name "
                           "(without extension) is used as its own gesture "
                           "label -- pass --label as well to instead treat "
                           "every video in the folder as the SAME gesture "
                           "(e.g. several clips of the same sign).")

    parser.add_argument("--label", type=str, default=None,
                         help="Gesture name, e.g. 'hello'. Required with "
                              "--video. With --video_dir it is OPTIONAL: "
                              "if given, every video in the folder is "
                              "treated as this one gesture; if omitted, "
                              "each video's filename is used as its own "
                              "separate gesture label instead.")

    parser.add_argument("--mode", choices=["static", "sequence"],
                         default="static",
                         help="static = one row per frame (default). "
                              "sequence = one row per N-frame motion window.")
    parser.add_argument("--seq_length", type=int, default=30,
                         help="Frames per sequence window (sequence mode only).")
    parser.add_argument("--stride", type=int, default=None,
                         help="Step between sequence windows (sequence mode "
                              "only). Defaults to seq_length (no overlap).")

    parser.add_argument("--frame_skip", type=int, default=1,
                         help="Only process every Nth frame. Use >1 for "
                              "long static videos to avoid near-duplicate "
                              "samples. Leave at 1 for sequence mode.")
    parser.add_argument("--max_num_hands", type=int, default=2)
    parser.add_argument("--min_detection_confidence", type=float, default=0.7)
    parser.add_argument("--min_tracking_confidence", type=float, default=0.5)
    parser.add_argument("--flip", action="store_true",
                         help="Mirror-flip frames horizontally (only needed "
                              "if your video looks mirrored compared to how "
                              "you'll use the live webcam later).")
    parser.add_argument("--preview", action="store_true",
                         help="Show a window with landmarks while processing "
                              "(press 'q' to stop early).")

    parser.add_argument("--output", type=str,
                         default="model/keypoint_classifier/keypoint.csv",
                         help="CSV to append static rows to.")
    parser.add_argument("--seq_output", type=str,
                         default="model/sequence_classifier/sequence_data.csv",
                         help="CSV to append sequence rows to (sequence mode).")
    parser.add_argument("--label_csv", type=str, default=None,
                         help="Label-name <-> ID lookup file. Auto-created "
                              "and auto-extended as needed. Defaults to "
                              "model/keypoint_classifier/keypoint_classifier_label.csv "
                              "in static mode, or "
                              "model/sequence_classifier/sequence_classifier_label.csv "
                              "in sequence mode.")

    parser.add_argument("--reset", action="store_true",
                         help="Wipe the existing data CSV AND label CSV for "
                              "the selected mode before extracting, so you "
                              "can start a brand new gesture vocabulary "
                              "from scratch. Asks for confirmation first.")

    args = parser.parse_args()
    if args.video and not args.label:
        raise SystemExit("--label is required when using --video")
    if args.label_csv is None:
        args.label_csv = (
            "model/keypoint_classifier/keypoint_classifier_label.csv"
            if args.mode == "static"
            else "model/sequence_classifier/sequence_classifier_label.csv"
        )
    return args


# ---------------------------------------------------------------------------
# Label bookkeeping
# ---------------------------------------------------------------------------

def load_labels(label_csv_path):
    if not os.path.exists(label_csv_path):
        return []
    with open(label_csv_path, encoding="utf-8-sig") as f:
        return [row[0] for row in csv.reader(f) if row]


def get_or_create_label_id(label_name, label_csv_path):
    file_existed = os.path.exists(label_csv_path)
    labels = load_labels(label_csv_path)

    if label_name in labels:
        return labels.index(label_name), labels

    if not file_existed:
        print(f"\n*** WARNING ***")
        print(f"No label file found at: {os.path.abspath(label_csv_path)}")
        print("About to create a BRAND NEW one with only this one label in it.")
        print("If this path is supposed to already contain labels from an "
              "existing trained model, STOP: you are probably running this "
              "script from the wrong working directory.")
        answer = input("Type 'yes' to really create a new label file here, "
                        "anything else to abort: ").strip().lower()
        if answer != "yes":
            raise SystemExit("Aborted -- no files were changed. Run this "
                              "script from your repo root instead.")

    Path(os.path.dirname(label_csv_path) or ".").mkdir(parents=True, exist_ok=True)
    labels.append(label_name)
    with open(label_csv_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        for name in labels:
            writer.writerow([name])
    return len(labels) - 1, labels


# ---------------------------------------------------------------------------
# Core extraction
# ---------------------------------------------------------------------------

def extract_from_video(video_path, label_id, label_name, args, hands, csv_writer,
                        seq_csv_writer):
    cap = cv.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"  !! Could not open video: {video_path}")
        return 0, 0

    frame_idx = 0
    static_rows = 0
    sequence_rows = 0
    sequence_buffer = []  # one shared buffer -- each entry is a combined 84-value frame
    stride = args.stride or args.seq_length
    frames_with_no_hand = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if args.mode == "static" and frame_idx % args.frame_skip != 0:
            frame_idx += 1
            continue

        if args.flip:
            frame = cv.flip(frame, 1)

        rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        results = hands.process(rgb)
        rgb.flags.writeable = True

        combined, per_hand_info = build_combined_vector(
            frame, results.multi_hand_landmarks, results.multi_handedness)

        if per_hand_info:  # at least one hand detected this frame
            if args.mode == "static":
                csv_writer.writerow([label_id, *combined])
                static_rows += 1
            else:  # sequence
                sequence_buffer.append(combined)
                if len(sequence_buffer) >= args.seq_length:
                    window = sequence_buffer[:args.seq_length]
                    flat = list(itertools.chain.from_iterable(window))
                    seq_csv_writer.writerow([label_id, *flat])
                    sequence_rows += 1
                    sequence_buffer = sequence_buffer[stride:]

            if args.preview:
                for side, landmark_list, brect in per_hand_info:
                    for x, y in landmark_list:
                        cv.circle(frame, (x, y), 3, (0, 255, 0), -1)
                    cv.rectangle(frame, (brect[0], brect[1]),
                                 (brect[2], brect[3]), (0, 0, 0), 1)
        else:
            frames_with_no_hand += 1

        if args.preview:
            cv.putText(frame, f"{label_name}", (10, 30),
                       cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv.LINE_AA)
            cv.imshow("Extracting gesture data (q to stop)", frame)
            if cv.waitKey(1) & 0xFF == ord("q"):
                break

        frame_idx += 1

    cap.release()
    if frames_with_no_hand:
        print(f"  ({frames_with_no_hand} frame(s) had no hand detected and "
              f"were skipped)")
    return static_rows, sequence_rows


def main():
    args = get_args()

    # Build the (video_path, label_name) work list
    jobs = []
    if args.video:
        jobs.append((Path(args.video), args.label))
    else:
        video_dir = Path(args.video_dir)
        exts = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
        for path in sorted(video_dir.iterdir()):
            if path.suffix.lower() in exts:
                label_name = args.label if args.label else path.stem
                jobs.append((path, label_name))
        if not jobs:
            raise SystemExit(f"No video files found in {args.video_dir}")

        if args.label:
            print(f"Batch mode: all {len(jobs)} video(s) in {video_dir} "
                  f"will be labeled '{args.label}'.\n")
        else:
            print(f"Batch mode: {len(jobs)} video(s) in {video_dir}, each "
                  f"labeled by its own filename.\n")

    out_path = args.output if args.mode == "static" else args.seq_output
    Path(os.path.dirname(out_path) or ".").mkdir(parents=True, exist_ok=True)

    print(f"Data CSV  : {os.path.abspath(out_path)}")
    print(f"Label CSV : {os.path.abspath(args.label_csv)}")
    print(f"Feature format: combined 2-hand vector, "
          f"{TOTAL_FEATURES} values per frame")
    print("(Double-check these paths look right for your repo before "
          "continuing -- wrong working directory is the #1 cause of a "
          "label mismatch later.)\n")

    if args.reset:
        existing = [p for p in (out_path, args.label_csv) if os.path.exists(p)]
        if existing:
            print("--reset was passed. This will DELETE:")
            for p in existing:
                print(f"  {os.path.abspath(p)}")
            answer = input("Type 'yes' to permanently delete these and "
                            "start a brand new gesture vocabulary: ").strip().lower()
            if answer != "yes":
                raise SystemExit("Aborted -- no files were changed.")
            for p in existing:
                os.remove(p)
                print(f"  deleted {p}")
        else:
            print("--reset was passed, but there was nothing to delete "
                  "at those paths -- already starting fresh.")
        Path(os.path.dirname(args.label_csv) or ".").mkdir(parents=True, exist_ok=True)
        open(args.label_csv, "w").close()
        print()

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=args.max_num_hands,
        min_detection_confidence=args.min_detection_confidence,
        min_tracking_confidence=args.min_tracking_confidence,
    )

    total_static, total_seq = 0, 0
    with open(out_path, "a", newline="") as out_f:
        csv_writer = csv.writer(out_f)
        seq_writer = csv_writer  # same writer object works for either mode

        for video_path, label_name in jobs:
            label_id, labels = get_or_create_label_id(label_name, args.label_csv)
            print(f"[{video_path.name}] label='{label_name}' -> id={label_id}")

            s_rows, q_rows = extract_from_video(
                video_path, label_id, label_name, args, hands, csv_writer, seq_writer)

            if args.mode == "static":
                print(f"  saved {s_rows} frame samples -> {out_path}")
            else:
                print(f"  saved {q_rows} sequence samples "
                      f"({args.seq_length} frames each) -> {out_path}")
            total_static += s_rows
            total_seq += q_rows

    hands.close()
    if args.preview:
        cv.destroyAllWindows()

    print("\nDone.")
    if args.mode == "static":
        print(f"Total new rows written: {total_static}")
    else:
        print(f"Total new sequence rows written: {total_seq}")
    print(f"Current labels in {args.label_csv}: {load_labels(args.label_csv)}")


if __name__ == "__main__":
    main()
