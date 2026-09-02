#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
End-to-end test of app.py's live loop WITHOUT a camera, real models, or
audio: python -m pytest -q test_app_integration.py

What is real here: app.run_loop / load_sequence_classifier, the real
KeyPointClassifier + SequenceClassifier wrapper code, landmark_utils,
cv2 drawing, GestureStabilizer, SpeechWorker.
What is faked: tensorflow.lite.Interpreter (scripted outputs), MediaPipe
results (scripted hand landmarks), cv.VideoCapture / waitKey / imshow, and
the pyttsx3 engine (recording fake).
"""
import os
import sys
import csv
import types
import shutil
import subprocess
from collections import deque

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------------------
# Scripted "world" the fakes read from
# ---------------------------------------------------------------------------
SCRIPT = {"hands": 0, "static": 0, "seq_probs": None, "frame": 0}
SEQ_LEN, N_STATIC, N_SEQ = 10, 2, 2
STATIC_LABELS = ["flat_hand", "fist"]
SEQ_LABELS = ["wave", "circle"]


class FakeInterpreter:
    """Stand-in for tf.lite.Interpreter driven by SCRIPT."""
    def __init__(self, model_path, num_threads=1):
        self.is_seq = "sequence" in model_path
        self.inputs = []

    def allocate_tensors(self):
        pass

    def get_input_details(self):
        shape = (np.array([1, SEQ_LEN, 84]) if self.is_seq
                 else np.array([1, 84]))
        return [{"index": 0, "shape": shape}]

    def get_output_details(self):
        n = N_SEQ if self.is_seq else N_STATIC
        return [{"index": 1, "shape": np.array([1, n])}]

    def set_tensor(self, index, value):
        self.last = np.asarray(value)
        self.inputs.append(self.last.shape)

    def invoke(self):
        pass

    def get_tensor(self, index):
        if self.is_seq:
            assert self.last.shape == (1, SEQ_LEN, 84), self.last.shape
            probs = SCRIPT["seq_probs"] or [0.1] * N_SEQ
            return np.array([probs], dtype=np.float32)
        assert self.last.shape == (1, 84), self.last.shape
        out = np.zeros((1, N_STATIC), dtype=np.float32)
        out[0, SCRIPT["static"]] = 1.0
        return out


def _install_fakes(monkeypatch):
    tf = types.ModuleType("tensorflow")
    tf.lite = types.SimpleNamespace(Interpreter=FakeInterpreter)
    monkeypatch.setitem(sys.modules, "tensorflow", tf)

    class Lm:
        def __init__(self, x, y):
            self.x, self.y = x, y

    class HandLandmarks:
        def __init__(self, ox):
            self.landmark = [Lm(ox + 0.01 * i, 0.3 + 0.02 * i) for i in range(21)]

    class Handedness:
        def __init__(self, label):
            self.classification = [types.SimpleNamespace(label=label)]

    class FakeHands:
        def __init__(self, **kw):
            pass

        def process(self, image):
            n = SCRIPT["hands"]
            sides = ["Left", "Right"][:n]
            return types.SimpleNamespace(
                multi_hand_landmarks=[HandLandmarks(0.2 + 0.4 * i)
                                      for i in range(n)] or None,
                multi_handedness=[Handedness(s) for s in sides] or None)

    mp = types.ModuleType("mediapipe")
    mp.solutions = types.SimpleNamespace(
        hands=types.SimpleNamespace(Hands=FakeHands))
    monkeypatch.setitem(sys.modules, "mediapipe", mp)


class ScriptedCap:
    """Fake cv.VideoCapture whose read() advances a frame script."""
    def __init__(self, frames):
        self.frames = frames
        self.i = 0

    def read(self):
        if self.i >= len(self.frames):
            return False, None  # camera "disconnects"
        SCRIPT.update(self.frames[self.i])
        SCRIPT["frame"] = self.i
        self.i += 1
        return True, np.zeros((120, 160, 3), dtype=np.uint8)

    def set(self, *a):
        pass

    def release(self):
        self.released = True


class RecordingEngine:
    def __init__(self):
        self.said = []
        self._p = None

    def say(self, t):
        self._p = t

    def runAndWait(self):
        self.said.append(self._p)

    def stop(self):
        pass


@pytest.fixture
def workdir(tmp_path, monkeypatch):
    """Copy the repo into a temp dir, add dummy model/label files, chdir."""
    dst = tmp_path / "repo"
    shutil.copytree(ROOT, dst, ignore=shutil.ignore_patterns(".git", "__pycache__"))
    with open(dst / "model/keypoint_classifier/keypoint_classifier_label.csv",
              "w", newline="") as f:
        csv.writer(f).writerows([[l] for l in STATIC_LABELS])
    with open(dst / "model/sequence_classifier/sequence_classifier_label.csv",
              "w", newline="") as f:
        csv.writer(f).writerows([[l] for l in SEQ_LABELS])
    (dst / "model/sequence_classifier/sequence_classifier.tflite").write_bytes(b"x")
    monkeypatch.chdir(dst)
    monkeypatch.syspath_prepend(str(dst))
    for m in list(sys.modules):
        if m in ("app", "model", "landmark_utils", "gesture_output", "utils",
                 "camera_utils") \
                or m.startswith("model."):
            monkeypatch.delitem(sys.modules, m, raising=False)
    return dst


def run_app(monkeypatch, frames, stable_frames=4, no_tts=False,
            seq_model="model/sequence_classifier/sequence_classifier.tflite"):
    _install_fakes(monkeypatch)
    import cv2 as cv
    import app
    from gesture_output import SpeechWorker

    engine = RecordingEngine()

    class RecordingSpeechWorker(SpeechWorker):
        def __init__(self, engine_factory=None, maxsize=16):
            super().__init__(engine_factory=lambda: engine, maxsize=maxsize)
    monkeypatch.setattr(app, "SpeechWorker", RecordingSpeechWorker)
    cap = ScriptedCap(frames)
    monkeypatch.setattr(app, "open_camera", lambda *a, **k: (cap, "fake"))
    monkeypatch.setattr(cv, "waitKey", lambda *a: -1)
    shown = []
    monkeypatch.setattr(cv, "imshow", lambda name, img: shown.append(img))
    monkeypatch.setattr(cv, "destroyAllWindows", lambda: None)

    argv = ["app.py", "--stable_frames", str(stable_frames),
            "--seq_model", seq_model]
    if no_tts:
        argv.append("--no_tts")
    monkeypatch.setattr(sys, "argv", argv)

    # spy on the stabilizer so we can read the timeline of current_gesture
    timeline = []
    orig_update = app.GestureStabilizer.update

    def spy(self, raw):
        cur, changed = orig_update(self, raw)
        timeline.append((SCRIPT["frame"], raw, cur, changed))
        return cur, changed
    monkeypatch.setattr(app.GestureStabilizer, "update", spy)

    app.main()
    return engine, timeline, cap, shown


# ---------------------------------------------------------------------------

def F(hands=1, static=0, seq=None, n=1):
    return [{"hands": hands, "static": static, "seq_probs": seq}] * n


def test_full_scenario(workdir, monkeypatch):
    frames = (
        F(hands=1, static=0, n=6)                        # static-only, buffer filling
        + F(hands=1, static=0, seq=[0.9, 0.1], n=12)     # motion: seq confident 'wave'
        + F(hands=0, n=3)                                # short dropout (< MAX_MISSING)
        + F(hands=1, static=0, seq=[0.9, 0.1], n=6)      # still waving
        + F(hands=0, n=8)                                # long gap: buffer reset, screen clears
        + F(hands=2, static=1, seq=[0.2, 0.2], n=5)      # two hands, seq unsure -> static 'fist'
        + F(hands=2, static=0, seq=[0.2, 0.2], n=5)      # quick change -> 'flat_hand'
    )
    engine, timeline, cap, shown = run_app(monkeypatch, frames, stable_frames=4)

    spoken_events = [(f, cur) for f, raw, cur, ch in timeline if ch and cur]
    # The 9-frame static warm-up (>= stable_frames=4) is spoken first -- see
    # test_static_onset_is_spoken_when_held_long_enough for the tradeoff.
    assert spoken_events == [(3, "flat_hand"), (12, "wave"),
                             (38, "fist"), (43, "flat_hand")]
    assert engine.said == ["flat_hand", "wave", "fist", "flat_hand"]

    raw_by_frame = {f: raw for f, raw, cur, ch in timeline}
    # frames 0-5: seq buffer < SEQ_LEN -> static fallback
    assert all(raw_by_frame[f] == "flat_hand" for f in range(6))
    # buffer reaches 10 hand-present frames at frame 9 -> seq becomes authoritative
    assert raw_by_frame[8] == "flat_hand" and raw_by_frame[9] == "wave"
    # short 3-frame dropout never cleared the current gesture
    cur_by_frame = {f: cur for f, raw, cur, ch in timeline}
    assert cur_by_frame[20] == "wave" and cur_by_frame[23] == "wave"
    # long gap: current cleared to None after stable_frames of no hands
    assert cur_by_frame[27 + 3] is None
    assert shown and cap.released


def test_static_onset_is_spoken_when_held_long_enough(workdir, monkeypatch):
    """Documents the tradeoff: a static shape held >= stable_frames before
    the sequence model fires IS spoken. With SEQ_LEN=10 and stable=4 the
    9-frame warm-up in test_full_scenario would also do this -- so check the
    exact first event there."""
    # 'wave' can only appear from frame 9 (buffer full), so 13 'wave' frames
    frames = F(hands=1, static=0, n=6) + F(hands=1, static=0, seq=[0.9, 0.1], n=16)
    engine, timeline, *_ = run_app(monkeypatch, frames, stable_frames=4)
    assert engine.said == ["flat_hand", "wave"]
    engine, timeline, *_ = run_app(monkeypatch, frames, stable_frames=10)
    assert engine.said == ["wave"]  # larger stable window absorbs the onset


def test_missing_sequence_model_degrades_to_static_with_tts(workdir, monkeypatch,
                                                            capsys):
    os.remove(workdir / "model/sequence_classifier/sequence_classifier.tflite")
    frames = F(hands=1, static=1, n=10) + F(hands=1, static=0, n=10)
    engine, timeline, *_ = run_app(monkeypatch, frames, stable_frames=4)
    out = capsys.readouterr().out
    assert out.count("STATIC-ONLY") == 1
    assert engine.said == ["fist", "flat_hand"]


def test_label_mismatch_disables_sequence(workdir, monkeypatch, capsys):
    with open(workdir / "model/sequence_classifier/sequence_classifier_label.csv",
              "a", newline="") as f:
        csv.writer(f).writerow(["extra_label"])
    frames = F(hands=1, static=0, seq=[0.9, 0.1], n=15)
    engine, *_ = run_app(monkeypatch, frames, stable_frames=4)
    assert "diagnose_label_mismatch" in capsys.readouterr().out
    assert engine.said == ["flat_hand"]  # never 'wave': sequence disabled


def test_no_tts_flag(workdir, monkeypatch):
    frames = F(hands=1, static=0, n=10)
    engine, timeline, *_ = run_app(monkeypatch, frames, stable_frames=4, no_tts=True)
    assert engine.said == [] and any(cur == "flat_hand" for _, _, cur, _ in timeline)


def test_missing_pyttsx3_gives_actionable_error(workdir, monkeypatch):
    """Run app.py in a subprocess with pyttsx3 hidden and models faked out;
    it must exit with an actionable message rather than a traceback."""
    code = r'''
import sys, types, builtins
real_import = builtins.__import__
def fake_import(name, *a, **k):
    if name == "pyttsx3": raise ImportError("No module named pyttsx3")
    return real_import(name, *a, **k)
builtins.__import__ = fake_import
from gesture_output import SpeechWorker
try:
    SpeechWorker().start()
except RuntimeError as e:
    print("OK:", e); sys.exit(0)
print("NO ERROR"); sys.exit(1)
'''
    r = subprocess.run([sys.executable, "-c", code], capture_output=True,
                       text=True, cwd=workdir)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "pip install pyttsx3" in r.stdout


def test_single_class_sequence_model_is_refused(workdir, monkeypatch, capsys):
    with open(workdir / "model/sequence_classifier/sequence_classifier_label.csv",
              "w", newline="") as f:
        csv.writer(f).writerow(["wave"])
    monkeypatch.setattr(sys.modules[__name__], "N_SEQ", 1)
    frames = F(hands=1, static=0, seq=[1.0], n=15)
    engine, *_ = run_app(monkeypatch, frames, stable_frames=4)
    out = capsys.readouterr().out
    assert "only 1 label" in out and "STATIC-ONLY" in out
    assert engine.said == ["flat_hand"]
