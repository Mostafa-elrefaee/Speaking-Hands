#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
app.py

Live webcam gesture recognition. Both hands (if present) are combined into
ONE 84-value vector per frame and classified as a SINGLE gesture -- so a
sign that needs both hands together (they touch, mirror each other, one
modifies the other, etc.) is recognized as one word, not as two separate
per-hand guesses.

This uses landmark_utils.build_combined_vector(), the exact same function
extract_gesture_data.py uses to build training data, so live inference and
training data can never disagree on the feature format.

NOTE: the original repo's PointHistoryClassifier (tracking a pointing
finger drawing "Stop"/"Clockwise"/"Counter Clockwise"/"Move" in the air)
has been removed. It was built around a single dominant hand's fingertip
and doesn't have a sensible meaning once both hands are combined into one
gesture -- it was also unrelated to sign-language recognition to begin
with.

Motion signs: a rolling deque of combined vectors feeds the optional
SequenceClassifier every frame alongside the static KeyPointClassifier
(sequence wins when confident, static is the fallback). The result is
debounced by gesture_output.GestureStabilizer and that ONE stabilized value
is both drawn on screen and spoken via pyttsx3 on a background thread.
"""
import os
import csv
import copy
import argparse
from collections import deque

import cv2 as cv
import mediapipe as mp

from utils import CvFpsCalc
from model import KeyPointClassifier
from landmark_utils import build_combined_vector, TOTAL_FEATURES
from gesture_output import (choose_gesture, GestureStabilizer, SpeechWorker)
from camera_utils import open_camera

# How many consecutive frames a raw prediction must agree before it becomes
# the "current gesture" that is drawn AND spoken. At ~30 fps, 8 frames is
# ~0.27 s: long enough to swallow 1-3 frame classifier flicker and most of the
# static/sequence disagreement at the onset of a motion sign, short enough
# that a fluent signer holding each word for ~0.4 s+ still gets every word.
STABLE_FRAMES = 8

# Consecutive no-hand frames after which the sequence buffer is reset.
# Shorter dropouts keep the buffer as-is (training data skips no-hand frames
# too, see extract_gesture_data.py), longer gaps mean a new sign is starting
# and stale pre-gap frames would only mislead the sequence model.
MAX_MISSING_FRAMES = 5

SEQ_MODEL_PATH = 'model/sequence_classifier/sequence_classifier.tflite'
SEQ_LABEL_PATH = 'model/sequence_classifier/sequence_classifier_label.csv'


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--width", help='cap width', type=int, default=960)
    parser.add_argument("--height", help='cap height', type=int, default=540)

    parser.add_argument('--use_static_image_mode', action='store_true')
    parser.add_argument("--max_num_hands", type=int, default=2)
    parser.add_argument("--min_detection_confidence",
                        help='min_detection_confidence',
                        type=float,
                        default=0.7)
    parser.add_argument("--min_tracking_confidence",
                        help='min_tracking_confidence',
                        type=float,
                        default=0.5)

    parser.add_argument("--seq_model", type=str, default=SEQ_MODEL_PATH)
    parser.add_argument("--seq_label", type=str, default=SEQ_LABEL_PATH)
    parser.add_argument("--seq_score_th", type=float, default=0.5,
                        help='sequence classifier confidence threshold')
    parser.add_argument("--stable_frames", type=int, default=STABLE_FRAMES)
    parser.add_argument("--no_tts", action='store_true',
                        help='disable spoken output')
    parser.add_argument("--backend", choices=["auto", "dshow", "msmf", "any"],
                        default="auto",
                        help='camera backend (auto = dshow, then msmf on Windows)')

    args = parser.parse_args()

    return args


def main():
    # Argument parsing #################################################################
    args = get_args()

    cap_device = args.device
    cap_width = args.width
    cap_height = args.height

    use_static_image_mode = args.use_static_image_mode
    max_num_hands = args.max_num_hands
    min_detection_confidence = args.min_detection_confidence
    min_tracking_confidence = args.min_tracking_confidence

    use_brect = True

    # Camera preparation ###############################################################
    # open_camera() picks a backend that actually delivers a real image
    # (see camera_utils.py) instead of trusting cv.VideoCapture(0).
    cap, _backend = open_camera(cap_device, args.backend, cap_width, cap_height)

    # Model load #############################################################
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=use_static_image_mode,
        max_num_hands=max_num_hands,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
    )

    keypoint_classifier = KeyPointClassifier()

    # Read labels ###########################################################
    with open('model/keypoint_classifier/keypoint_classifier_label.csv',
              encoding='utf-8-sig') as f:
        keypoint_classifier_labels = csv.reader(f)
        keypoint_classifier_labels = [
            row[0] for row in keypoint_classifier_labels
        ]

    # Sequence (motion) classifier -- optional, degrades to static-only ######
    sequence_classifier, sequence_classifier_labels = load_sequence_classifier(
        args.seq_model, args.seq_label, args.seq_score_th)
    seq_len = sequence_classifier.seq_length if sequence_classifier else 1
    seq_buffer = deque(maxlen=int(seq_len))

    # Stabilizer + TTS: ONE stabilized "current gesture" feeds both ##########
    stabilizer = GestureStabilizer(args.stable_frames)
    speaker = None
    if not args.no_tts:
        speaker = SpeechWorker().start()  # raises with an actionable message

    # FPS Measurement ########################################################
    cvFpsCalc = CvFpsCalc(buffer_len=10)

    try:
        run_loop(cap, hands, keypoint_classifier, keypoint_classifier_labels,
                 sequence_classifier, sequence_classifier_labels, seq_buffer,
                 stabilizer, speaker, cvFpsCalc, use_brect)
    finally:
        cap.release()
        cv.destroyAllWindows()
        if speaker is not None:
            speaker.close()


def load_sequence_classifier(model_path, label_path, score_th):
    """Returns (SequenceClassifier, labels) or (None, None) with ONE clear
    console message if the motion model can't be used yet."""
    if not os.path.exists(model_path) or not os.path.exists(label_path):
        print(f"[info] No sequence classifier found ({model_path} / "
              f"{label_path}). Running STATIC-ONLY. Train one with "
              "extract_gesture_data.py --mode sequence + "
              "train_sequence_classifier.py to enable motion signs.")
        return None, None

    from model.sequence_classifier.sequence_classifier import SequenceClassifier
    clf = SequenceClassifier(model_path=model_path, score_th=score_th)
    with open(label_path, encoding='utf-8-sig') as f:
        labels = [row[0] for row in csv.reader(f) if row]

    # Same guard rails as the training scripts: never trust hand-typed sizes.
    num_classes = int(clf.output_details[0]['shape'][-1])
    if len(labels) < 2:
        print(f"[warn] {label_path} has only {len(labels)} label(s). A "
              "1-class model always predicts that class with 100% "
              "confidence, so it would override the static classifier on "
              "every frame. Extract at least one more gesture (e.g. a 'rest' "
              "class of idle hands) and retrain. Running STATIC-ONLY.")
        return None, None
    if int(clf.num_features) != TOTAL_FEATURES:
        print(f"[warn] {model_path} expects {clf.num_features} features per "
              f"frame but landmark_utils.TOTAL_FEATURES is {TOTAL_FEATURES}. "
              "It was trained on a different vector format -- retrain it. "
              "Running STATIC-ONLY.")
        return None, None
    if num_classes != len(labels):
        print(f"[warn] {model_path} outputs {num_classes} classes but "
              f"{label_path} has {len(labels)} labels. Run "
              "diagnose_label_mismatch.py. Running STATIC-ONLY.")
        return None, None

    print(f"[info] Sequence classifier loaded: seq_length={clf.seq_length}, "
          f"{num_classes} labels, score_th={score_th}")
    return clf, labels


def run_loop(cap, hands, keypoint_classifier, keypoint_classifier_labels,
             sequence_classifier, sequence_classifier_labels, seq_buffer,
             stabilizer, speaker, cvFpsCalc, use_brect):
    mode = 0
    missing_frames = 0

    while True:
        fps = cvFpsCalc.get()

        # Process Key (ESC: end) #################################################
        key = cv.waitKey(10)
        if key == 27:  # ESC
            break
        number, mode = select_mode(key, mode)

        # Camera capture #####################################################
        ret, image = cap.read()
        if not ret:
            break
        image = cv.flip(image, 1)  # Mirror display
        debug_image = copy.deepcopy(image)

        # Detection implementation #############################################################
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

        image.flags.writeable = False
        results = hands.process(image)
        image.flags.writeable = True

        #  ####################################################################
        # Combine whatever hand(s) were detected this frame into ONE gesture
        # vector -- the exact same function used to build training data.
        combined_vector, per_hand_info = build_combined_vector(
            debug_image, results.multi_hand_landmarks, results.multi_handedness)

        raw_gesture, source = None, None
        if per_hand_info:  # at least one hand detected
            missing_frames = 0
            # Write to the dataset file (data-collection mode, 'k' key)
            logging_csv(number, mode, combined_vector)

            # Rolling window for the motion model. Only hand-present frames
            # are appended -- identical to how extract_gesture_data.py builds
            # the training windows. A missing SECOND hand is already
            # zero-filled inside combined_vector.
            seq_buffer.append(combined_vector)

            # ONE static prediction for the whole (one- or two-handed) gesture
            hand_sign_id = keypoint_classifier(combined_vector)
            static_label = keypoint_classifier_labels[hand_sign_id]

            # Sequence prediction (wrapper returns invalid_value until the
            # buffer holds seq_length frames or when confidence is low)
            seq_label = None
            if sequence_classifier is not None:
                seq_id = sequence_classifier(seq_buffer)
                if seq_id != sequence_classifier.invalid_value:
                    seq_label = sequence_classifier_labels[seq_id]

            raw_gesture, source = choose_gesture(seq_label, static_label)

            # Drawing: each hand's skeleton + box
            for side, landmark_list, brect in per_hand_info:
                debug_image = draw_bounding_rect(use_brect, debug_image, brect)
                debug_image = draw_landmarks(debug_image, landmark_list)
                debug_image = draw_hand_tag(debug_image, brect, side)
        else:
            missing_frames += 1
            if missing_frames == MAX_MISSING_FRAMES:
                seq_buffer.clear()

        # Single source of truth: the stabilized gesture drives BOTH the
        # on-screen label and the speech output. `changed` fires exactly once
        # per stabilization to a new value.
        current_gesture, changed = stabilizer.update(raw_gesture)
        if changed and current_gesture is not None and speaker is not None:
            speaker.say(current_gesture)

        debug_image = draw_gesture_text(debug_image, current_gesture)
        debug_image = draw_raw_gesture(debug_image, raw_gesture, source)
        debug_image = draw_info(debug_image, fps, mode, number)

        # Screen reflection #############################################################
        cv.imshow('Hand Gesture Recognition', debug_image)


def select_mode(key, mode):
    number = -1
    if 48 <= key <= 57:  # 0 ~ 9
        number = key - 48
    if key == 110:  # n
        mode = 0
    if key == 107:  # k
        mode = 1
    return number, mode


def logging_csv(number, mode, combined_vector):
    """Live data-collection mode ('k' then a digit key): appends the
    current combined 2-hand vector to keypoint.csv, same format
    extract_gesture_data.py produces. This is a convenience for quick
    single-frame samples; for real gesture videos, prefer
    extract_gesture_data.py."""
    if mode == 1 and (0 <= number <= 9):
        csv_path = 'model/keypoint_classifier/keypoint.csv'
        with open(csv_path, 'a', newline="") as f:
            writer = csv.writer(f)
            writer.writerow([number, *combined_vector])
    return


def draw_landmarks(image, landmark_point):
    if len(landmark_point) > 0:
        # Thumb
        cv.line(image, tuple(landmark_point[2]), tuple(landmark_point[3]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[2]), tuple(landmark_point[3]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[3]), tuple(landmark_point[4]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[3]), tuple(landmark_point[4]),
                (255, 255, 255), 2)

        # Index finger
        cv.line(image, tuple(landmark_point[5]), tuple(landmark_point[6]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[5]), tuple(landmark_point[6]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[6]), tuple(landmark_point[7]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[6]), tuple(landmark_point[7]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[7]), tuple(landmark_point[8]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[7]), tuple(landmark_point[8]),
                (255, 255, 255), 2)

        # Middle finger
        cv.line(image, tuple(landmark_point[9]), tuple(landmark_point[10]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[9]), tuple(landmark_point[10]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[10]), tuple(landmark_point[11]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[10]), tuple(landmark_point[11]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[11]), tuple(landmark_point[12]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[11]), tuple(landmark_point[12]),
                (255, 255, 255), 2)

        # Ring finger
        cv.line(image, tuple(landmark_point[13]), tuple(landmark_point[14]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[13]), tuple(landmark_point[14]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[14]), tuple(landmark_point[15]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[14]), tuple(landmark_point[15]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[15]), tuple(landmark_point[16]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[15]), tuple(landmark_point[16]),
                (255, 255, 255), 2)

        # Little finger
        cv.line(image, tuple(landmark_point[17]), tuple(landmark_point[18]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[17]), tuple(landmark_point[18]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[18]), tuple(landmark_point[19]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[18]), tuple(landmark_point[19]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[19]), tuple(landmark_point[20]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[19]), tuple(landmark_point[20]),
                (255, 255, 255), 2)

        # Palm
        cv.line(image, tuple(landmark_point[0]), tuple(landmark_point[1]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[0]), tuple(landmark_point[1]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[1]), tuple(landmark_point[2]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[1]), tuple(landmark_point[2]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[2]), tuple(landmark_point[5]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[2]), tuple(landmark_point[5]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[5]), tuple(landmark_point[9]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[5]), tuple(landmark_point[9]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[9]), tuple(landmark_point[13]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[9]), tuple(landmark_point[13]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[13]), tuple(landmark_point[17]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[13]), tuple(landmark_point[17]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[17]), tuple(landmark_point[0]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[17]), tuple(landmark_point[0]),
                (255, 255, 255), 2)

    # Key Points
    for index, landmark in enumerate(landmark_point):
        radius = 8 if index in (4, 8, 12, 16, 20) else 5
        cv.circle(image, (landmark[0], landmark[1]), radius, (255, 255, 255), -1)
        cv.circle(image, (landmark[0], landmark[1]), radius, (0, 0, 0), 1)

    return image


def draw_bounding_rect(use_brect, image, brect):
    if use_brect:
        cv.rectangle(image, (brect[0], brect[1]), (brect[2], brect[3]),
                     (0, 0, 0), 1)
    return image


def draw_hand_tag(image, brect, side):
    """Small 'Left'/'Right' label above each hand's box -- purely
    informational now, since the gesture prediction itself is shared
    across both hands (see draw_gesture_text)."""
    cv.rectangle(image, (brect[0], brect[1]), (brect[2], brect[1] - 22),
                 (0, 0, 0), -1)
    cv.putText(image, side, (brect[0] + 5, brect[1] - 4),
               cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv.LINE_AA)
    return image


def draw_gesture_text(image, gesture_text):
    """The ONE predicted gesture for this frame (both hands combined),
    shown prominently near the top of the screen."""
    if not gesture_text:
        return image
    cv.putText(image, "Gesture: " + gesture_text, (10, 60),
               cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 4, cv.LINE_AA)
    cv.putText(image, "Gesture: " + gesture_text, (10, 60),
               cv.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv.LINE_AA)
    return image


def draw_raw_gesture(image, raw_gesture, source):
    """Small per-frame debug line: the un-debounced prediction and which
    classifier produced it. Useful for tuning --seq_score_th/--stable_frames;
    the big label above is the stabilized value users actually get."""
    if raw_gesture is None:
        return image
    cv.putText(image, f"raw: {raw_gesture} ({source})", (10, 130),
               cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv.LINE_AA)
    return image


def draw_info(image, fps, mode, number):
    cv.putText(image, "FPS:" + str(fps), (10, 30), cv.FONT_HERSHEY_SIMPLEX,
               1.0, (0, 0, 0), 4, cv.LINE_AA)
    cv.putText(image, "FPS:" + str(fps), (10, 30), cv.FONT_HERSHEY_SIMPLEX,
               1.0, (255, 255, 255), 2, cv.LINE_AA)

    if mode == 1:
        cv.putText(image, "MODE:Logging Key Point", (10, 90),
                   cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1,
                   cv.LINE_AA)
        if 0 <= number <= 9:
            cv.putText(image, "NUM:" + str(number), (10, 110),
                       cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1,
                       cv.LINE_AA)
    return image


if __name__ == '__main__':
    main()
