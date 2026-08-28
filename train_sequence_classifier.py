#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
train_sequence_classifier.py

Trains an LSTM/GRU classifier on the sequence data produced by:
    extract_gesture_data.py --mode sequence

This is the model that KeyPointClassifier and PointHistoryClassifier don't
give you: one that looks at how the WHOLE hand moves across a window of
frames, so it can learn actual signs rather than a single still pose.

Input CSV format (one row per sample), matching extract_gesture_data.py:
    label_id, f1_x1,f1_y1, ..., f1_x21,f1_y21, f2_x1,f2_y1, ..., fN_x21,fN_y21
i.e. label_id followed by seq_length * 42 floats (21 landmarks * x,y per frame).

Outputs (all under model/sequence_classifier/ by default):
    sequence_classifier.tflite            <- for real-time inference
    sequence_classifier_label.csv         <- already exists (from extraction);
                                              left untouched
    sequence_classifier.keras             <- Keras model, kept for further
                                              training/debugging

USAGE
-----
python train_sequence_classifier.py
python train_sequence_classifier.py --cell gru --epochs 200
"""

import argparse
import csv

import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

from landmark_utils import NUM_LANDMARKS, COORDS_PER_LANDMARK, NUM_HANDS, TOTAL_FEATURES


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_csv", type=str,
                         default="model/sequence_classifier/sequence_data.csv")
    parser.add_argument("--label_csv", type=str,
                         default="model/sequence_classifier/sequence_classifier_label.csv")
    parser.add_argument("--seq_length", type=int, default=30,
                         help="Must match --seq_length used during extraction.")
    parser.add_argument("--num_landmarks", type=int, default=NUM_LANDMARKS)
    parser.add_argument("--coords_per_landmark", type=int, default=COORDS_PER_LANDMARK)
    parser.add_argument("--num_hands", type=int, default=NUM_HANDS,
                         help=f"Defaults to {NUM_HANDS} -- matches "
                              f"landmark_utils.py's combined 2-hand vector.")

    parser.add_argument("--cell", choices=["lstm", "gru"], default="lstm")
    parser.add_argument("--units", type=int, default=64,
                         help="Units in the first recurrent layer (the "
                              "second layer uses half this many).")
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--test_size", type=float, default=0.25)
    parser.add_argument("--random_state", type=int, default=42)

    parser.add_argument("--out_dir", type=str,
                         default="model/sequence_classifier")
    return parser.parse_args()


def load_labels(label_csv_path):
    with open(label_csv_path, encoding="utf-8-sig") as f:
        return [row[0] for row in csv.reader(f) if row]


def load_dataset(data_csv_path, seq_length, num_features):
    raw = np.loadtxt(data_csv_path, delimiter=",", dtype="float32")
    if raw.ndim == 1:  # only one sample in the file
        raw = raw.reshape(1, -1)

    y = raw[:, 0].astype(int)
    x_flat = raw[:, 1:]

    expected_len = seq_length * num_features
    if x_flat.shape[1] != expected_len:
        raise ValueError(
            f"Each row has {x_flat.shape[1]} feature values, but "
            f"seq_length ({seq_length}) x num_features ({num_features}) "
            f"= {expected_len}. Pass the same --seq_length you used in "
            f"extract_gesture_data.py."
        )

    x = x_flat.reshape(-1, seq_length, num_features)
    return x, y


def build_model(seq_length, num_features, num_classes, cell, units, dropout,
                 batch_size=None):
    """
    batch_size=None -> flexible batch size, used for training.
    batch_size=1    -> fixed batch of 1, required for a clean LSTM/GRU ->
                       TFLite conversion (the converter can't handle a
                       dynamic batch dimension with recurrent layers).
    """
    RecurrentLayer = tf.keras.layers.LSTM if cell == "lstm" else tf.keras.layers.GRU

    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(
            batch_input_shape=(batch_size, seq_length, num_features)),
        RecurrentLayer(units, return_sequences=True),
        tf.keras.layers.Dropout(dropout),
        RecurrentLayer(units // 2),
        tf.keras.layers.Dropout(dropout),
        tf.keras.layers.Dense(units // 2, activation="relu"),
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
    num_features = args.num_landmarks * args.coords_per_landmark * args.num_hands

    labels = load_labels(args.label_csv)
    num_classes = len(labels)
    print(f"Found {num_classes} gesture labels: {labels}")

    x, y = load_dataset(args.data_csv, args.seq_length, num_features)
    print(f"Loaded {x.shape[0]} sequence samples, shape={x.shape}")

    if x.shape[0] < num_classes * 4:
        print("WARNING: very little data per class -- collect more videos "
              "for anything beyond a quick smoke test.")

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

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=args.test_size, random_state=args.random_state,
        stratify=y if len(set(y)) > 1 else None,
    )

    model = build_model(args.seq_length, num_features, num_classes,
                         args.cell, args.units, args.dropout, batch_size=None)
    model.summary()

    checkpoint_path = f"{args.out_dir}/sequence_classifier_checkpoint.keras"
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            checkpoint_path, save_best_only=True, monitor="val_loss"),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=20, restore_best_weights=True),
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

    # Save the full Keras model for later fine-tuning/debugging
    keras_path = f"{args.out_dir}/sequence_classifier.keras"
    model.save(keras_path)
    print(f"\nSaved Keras model -> {keras_path}")

    # Export to TFLite for real-time inference (same deployment format as
    # the other two classifiers in this project). LSTM/GRU layers need a
    # FIXED batch size to convert cleanly, so we build a batch_size=1 clone
    # of the trained model and copy the learned weights into it.
    export_model = build_model(args.seq_length, num_features, num_classes,
                                args.cell, args.units, args.dropout,
                                batch_size=1)
    export_model.set_weights(model.get_weights())

    converter = tf.lite.TFLiteConverter.from_keras_model(export_model)
    tflite_model = converter.convert()

    # Self-check before saving: does the exported model's output size
    # actually match the label file it's meant to go with?
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

    tflite_path = f"{args.out_dir}/sequence_classifier.tflite"
    with open(tflite_path, "wb") as f:
        f.write(tflite_model)
    print(f"\nSelf-check passed: model outputs {exported_num_classes} "
          f"classes, matching {args.label_csv} exactly.")
    print(f"Saved TFLite model -> {tflite_path}")


if __name__ == "__main__":
    main()
