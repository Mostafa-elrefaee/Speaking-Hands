#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Place this file at: model/sequence_classifier/sequence_classifier.py

Mirrors the style of KeyPointClassifier / PointHistoryClassifier so it can
be dropped into app.py's real-time loop the same way. Instead of a single
frame, it takes a rolling buffer of the last `seq_length` pre-processed
landmark frames (each an 84-value combined two-hand list from
landmark_utils.build_combined_vector) and predicts which whole sign is being
made. Any of the four research architectures (MLP / GRU / LSTM / GRU+LSTM)
exported by scripts/train.py has the same [1, seq_length, 84] -> softmax
contract and loads here unchanged.
"""
import numpy as np
import tensorflow as tf


class SequenceClassifier(object):
    def __init__(
        self,
        model_path='model/sequence_classifier/sequence_classifier.tflite',
        score_th=0.5,
        invalid_value=-1,
        num_threads=1,
    ):
        self.interpreter = tf.lite.Interpreter(model_path=model_path,
                                               num_threads=num_threads)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        # input shape is (1, seq_length, num_features) -- read it from the
        # model itself so this class never gets out of sync with training
        _, self.seq_length, self.num_features = self.input_details[0]['shape']

        self.score_th = score_th
        self.invalid_value = invalid_value

    def predict_proba(self, landmark_sequence):
        """Full softmax vector (numpy [num_classes]) for the most recent
        self.seq_length frames, or None while the buffer is still filling.
        Used by the research stabilizer / benchmarks; __call__ keeps the
        original argmax-with-threshold contract."""
        if len(landmark_sequence) < self.seq_length:
            return None
        window = list(landmark_sequence)[-self.seq_length:]
        input_array = np.array([window], dtype=np.float32)  # (1, seq_len, num_features)
        input_index = self.input_details[0]['index']
        self.interpreter.set_tensor(input_index, input_array)
        self.interpreter.invoke()
        output_index = self.output_details[0]['index']
        return np.squeeze(self.interpreter.get_tensor(output_index)).astype(np.float32).reshape(-1)

    def __call__(self, landmark_sequence):
        """
        landmark_sequence: a list/deque of the most recent frames, each an
        84-value combined vector. Must contain at least self.seq_length frames;
        only the most recent self.seq_length are used.

        Returns the predicted label id, or self.invalid_value if there
        isn't enough history yet or confidence is below score_th.
        """
        if len(landmark_sequence) < self.seq_length:
            return self.invalid_value

        window = list(landmark_sequence)[-self.seq_length:]
        input_array = np.array([window], dtype=np.float32)  # (1, seq_len, num_features)

        input_index = self.input_details[0]['index']
        self.interpreter.set_tensor(input_index, input_array)
        self.interpreter.invoke()

        output_index = self.output_details[0]['index']
        result = self.interpreter.get_tensor(output_index)
        result = np.squeeze(result)

        result_index = int(np.argmax(result))
        if result[result_index] < self.score_th:
            return self.invalid_value
        return result_index
