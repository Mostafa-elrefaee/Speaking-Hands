#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
landmark_utils.py

Single source of truth for how hand landmarks get turned into a feature
vector -- used by app.py, extract_gesture_data.py, train_keypoint_classifier.py,
and train_sequence_classifier.py. Everything imports these constants and
functions instead of redefining its own copy, specifically so the "model
expects N features but got M" class of bug can't happen again.

TWO-HAND COMBINED GESTURES
---------------------------
A gesture is represented as ONE vector covering BOTH hands, laid out in
fixed slots:

    [ Left hand: 42 values | Right hand: 42 values ]  = 84 values total

If only one hand is present (a one-handed sign), the other hand's 42 slots
are filled with zeros. The model learns to treat an all-zero hand-slot as
"this hand isn't part of the sign," while a real hand's landmarks are
extremely unlikely to normalize to all zeros (only the wrist point, which
is the origin, is ever exactly [0, 0]).
"""
import copy
import itertools

NUM_LANDMARKS = 21
COORDS_PER_LANDMARK = 2
PER_HAND_FEATURES = NUM_LANDMARKS * COORDS_PER_LANDMARK  # 42
NUM_HANDS = 2
TOTAL_FEATURES = PER_HAND_FEATURES * NUM_HANDS  # 84

LEFT_SLICE = slice(0, PER_HAND_FEATURES)
RIGHT_SLICE = slice(PER_HAND_FEATURES, TOTAL_FEATURES)


def calc_bounding_rect(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]
    landmark_array = [[min(int(lm.x * image_width), image_width - 1),
                        min(int(lm.y * image_height), image_height - 1)]
                       for lm in landmarks.landmark]
    xs = [p[0] for p in landmark_array]
    ys = [p[1] for p in landmark_array]
    return [min(xs), min(ys), max(xs), max(ys)]


def calc_landmark_list(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]
    landmark_point = []
    for landmark in landmarks.landmark:
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)
        landmark_point.append([landmark_x, landmark_y])
    return landmark_point


def pre_process_landmark(landmark_list):
    """One hand's 21 [x,y] points -> 42 normalized floats, relative to the
    wrist and scaled to roughly [-1, 1]."""
    temp_landmark_list = copy.deepcopy(landmark_list)

    base_x, base_y = temp_landmark_list[0][0], temp_landmark_list[0][1]
    for point in temp_landmark_list:
        point[0] -= base_x
        point[1] -= base_y

    temp_landmark_list = list(itertools.chain.from_iterable(temp_landmark_list))

    max_value = max(map(abs, temp_landmark_list)) or 1  # avoid /0
    return [v / max_value for v in temp_landmark_list]


def build_combined_vector(image, multi_hand_landmarks, multi_handedness):
    """
    Combines however many hands MediaPipe detected this frame (0, 1, or 2)
    into ONE fixed-length 84-value vector representing the whole gesture.

    Returns: (combined_vector, per_hand_info)
        combined_vector: list of 84 floats
        per_hand_info: list of (side, landmark_list, brect) for drawing,
                       one entry per hand actually detected this frame
    """
    combined = [0.0] * TOTAL_FEATURES
    per_hand_info = []

    if not multi_hand_landmarks:
        return combined, per_hand_info

    for hand_landmarks, handedness in zip(multi_hand_landmarks, multi_handedness):
        side = handedness.classification[0].label  # "Left" or "Right"
        landmark_list = calc_landmark_list(image, hand_landmarks)
        processed = pre_process_landmark(landmark_list)
        brect = calc_bounding_rect(image, hand_landmarks)

        target_slice = LEFT_SLICE if side == "Left" else RIGHT_SLICE
        combined[target_slice] = processed
        per_hand_info.append((side, landmark_list, brect))

    return combined, per_hand_info
