#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Run this from your repo root, e.g.:
    python diagnose_label_mismatch.py
        (checks the keypoint_classifier by default)

    python diagnose_label_mismatch.py \
        --model model/point_history_classifier/point_history_classifier.tflite \
        --label model/point_history_classifier/point_history_classifier_label.csv
        (checks the point_history_classifier instead)

It tells you exactly how many classes the trained model can output vs.
how many labels are listed in the matching label CSV, which is the usual
cause of an IndexError like:
    keypoint_classifier_labels[hand_sign_id]
    point_history_classifier_labels[most_common_fg_id[0][0]]
"""
import argparse
import csv
import tensorflow as tf


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", type=str,
        default="model/keypoint_classifier/keypoint_classifier.tflite")
    parser.add_argument(
        "--label", type=str,
        default="model/keypoint_classifier/keypoint_classifier_label.csv")
    return parser.parse_args()


def main():
    args = get_args()

    interpreter = tf.lite.Interpreter(model_path=args.model)
    interpreter.allocate_tensors()
    output_shape = interpreter.get_output_details()[0]['shape']
    num_model_classes = output_shape[-1]

    with open(args.label, encoding="utf-8-sig") as f:
        labels = [row[0] for row in csv.reader(f) if row]

    print(f"Model ({args.model}) outputs {num_model_classes} classes.")
    print(f"Label file ({args.label}) has {len(labels)} rows: {labels}")

    if num_model_classes == len(labels):
        print("\nThese MATCH -- the crash must be something else "
              "(e.g. a stale .pyc / cached import). Try re-running after a "
              "clean restart.")
    elif num_model_classes > len(labels):
        print(f"\nMISMATCH: the model can predict up to class "
              f"{num_model_classes - 1}, but your label file only has rows "
              f"0-{len(labels) - 1}.")
        print(f"Fix: add the missing label name(s) to {args.label} (one "
              f"per line, in the same order used when the model was "
              f"trained).")
    else:
        print(f"\nMISMATCH: your label file has more rows ({len(labels)}) "
              f"than the model was trained to output ({num_model_classes}).")
        print(f"Fix: either retrain the model with the correct number of "
              f"classes, or remove the extra label row(s) that don't have "
              f"a matching trained class.")


if __name__ == "__main__":
    main()

