#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
train_keypoint_classifier.py

Trains the STATIC hand-shape classifier (single-frame, 21 landmarks) from
model/keypoint_classifier/keypoint.csv, and exports
model/keypoint_classifier/keypoint_classifier.tflite.

This replaces manually editing keypoint_classification.ipynb. The #1 cause
of the "IndexError: list index out of range" crash in app.py is a human
typing the wrong NUM_CLASSES in that notebook, or the label CSV and the
trained model silently drifting out of sync. This script removes that
entire failure mode:

  - The number of classes is ALWAYS read directly from
    keypoint_classifier_label.csv -- never typed in by hand.
  - Before saving anything, it double-checks that the exported .tflite
    model's output size actually matches the label file. If they don't
    match, it refuses to save and tells you why.
  - It warns you (without stopping) if any label in your label file has
    zero training samples, since that class will never be predicted
    correctly.

USAGE
-----
python train_keypoint_classifier.py
python train_keypoint_classifier.py --epochs 500
"""

import argparse
import csv

import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

from landmark_utils import TOTAL_FEATURES


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_csv", type=str,
                         default="model/keypoint_classifier/keypoint.csv")
    parser.add_argument("--label_csv", type=str,
                         default="model/keypoint_classifier/keypoint_classifier_label.csv")
    parser.add_argument("--num_features", type=int, default=TOTAL_FEATURES,
                         help=f"Values per frame. Defaults to "
                              f"{TOTAL_FEATURES} (2 hands x 21 landmarks x "
                              f"2 coords), matching landmark_utils.py -- "
                              f"only override this if you changed that file.")

    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--test_size", type=float, default=0.25)
    parser.add_argument("--random_state", type=int, default=42)

    parser.add_argument("--out_dir", type=str,
                         default="model/keypoint_classifier")
    return parser.parse_args()


def load_labels(label_csv_path):
    with open(label_csv_path, encoding="utf-8-sig") as f:
        return [row[0] for row in csv.reader(f) if row]


def load_dataset(data_csv_path, num_features):
    raw = np.loadtxt(data_csv_path, delimiter=",", dtype="float32")
    if raw.ndim == 1:
        raw = raw.reshape(1, -1)

    y = raw[:, 0].astype(int)
    x = raw[:, 1:]

    if x.shape[1] != num_features:
        raise ValueError(
            f"Each row in {data_csv_path} has {x.shape[1]} feature values, "
            f"but --num_features is {num_features}. These must match."
        )
    return x, y


def build_model(num_features, num_classes):
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(num_features,)),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(20, activation="relu"),
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.Dense(10, activation="relu"),
        tf.keras.layers.Dense(num_classes, activation="softmax"),
    ])
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def main():
    args = get_args()

    labels = load_labels(args.label_csv)
    num_classes = len(labels)
    if num_classes < 2:
        raise SystemExit(
            f"Only found {num_classes} label(s) in {args.label_csv}. "
            f"You need at least 2 gesture classes to train a classifier. "
            f"Run extract_gesture_data.py for a couple of gestures first."
        )
    print(f"Found {num_classes} gesture labels: {labels}")

    x, y = load_dataset(args.data_csv, args.num_features)
    print(f"Loaded {x.shape[0]} samples from {args.data_csv}")

    # Catch the other common failure mode early: a label ID appears in the
    # data that doesn't have a matching row in the label file (or vice
    # versa), instead of letting it surface later as a confusing crash.
    present_ids = set(np.unique(y).tolist())
    expected_ids = set(range(num_classes))
    unknown_ids = present_ids - expected_ids
    if unknown_ids:
        raise SystemExit(
            f"Found label id(s) {sorted(unknown_ids)} in {args.data_csv} "
            f"that have no matching row in {args.label_csv} (which only "
            f"defines ids 0-{num_classes - 1}). Fix the label file (or "
            f"re-extract) before training."
        )
    missing_ids = expected_ids - present_ids
    if missing_ids:
        missing_names = [labels[i] for i in sorted(missing_ids)]
        print(f"WARNING: these labels have ZERO training samples and will "
              f"never be predicted correctly: {missing_names}")
        print("Collect at least a few dozen samples for each before relying "
              "on this model.")

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=args.test_size, random_state=args.random_state,
        stratify=y if len(present_ids) > 1 and min(np.bincount(y)) > 1 else None,
    )

    model = build_model(args.num_features, num_classes)
    model.summary()

    checkpoint_path = f"{args.out_dir}/keypoint_classifier_checkpoint.keras"
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            checkpoint_path, save_best_only=True, monitor="val_loss"),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=30, restore_best_weights=True),
    ]

    model.fit(
        x_train, y_train,
        epochs=args.epochs,
        batch_size=args.batch_size,
        validation_data=(x_test, y_test),
        callbacks=callbacks,
        verbose=2,
    )

    print("\n=== Evaluation on held-out test split ===")
    y_pred = np.argmax(model.predict(x_test), axis=1)
    print(classification_report(y_test, y_pred, target_names=labels,
                                 labels=list(range(num_classes)),
                                 zero_division=0))
    print("Confusion matrix (rows=true, cols=predicted):")
    print(confusion_matrix(y_test, y_pred, labels=list(range(num_classes))))

    # --- Export to TFLite ---
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()

    # --- Self-check before saving anything: does the exported model's
    # output size actually match the label file? This is the exact check
    # that would have caught the earlier IndexError before it ever reached
    # app.py. ---
    interpreter = tf.lite.Interpreter(model_content=tflite_model)
    interpreter.allocate_tensors()
    exported_num_classes = interpreter.get_output_details()[0]['shape'][-1]

    if exported_num_classes != num_classes:
        raise SystemExit(
            f"REFUSING TO SAVE: exported model outputs "
            f"{exported_num_classes} classes but {args.label_csv} has "
            f"{num_classes} labels. This should not be possible -- please "
            f"report this. No files were written."
        )

    tflite_path = f"{args.out_dir}/keypoint_classifier.tflite"
    with open(tflite_path, "wb") as f:
        f.write(tflite_model)

    print(f"\nSelf-check passed: model outputs {exported_num_classes} "
          f"classes, matching {args.label_csv} exactly.")
    print(f"Saved TFLite model -> {tflite_path}")
    print("app.py will now recognize:", labels)


if __name__ == "__main__":
    main()
