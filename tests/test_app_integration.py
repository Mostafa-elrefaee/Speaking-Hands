"""Integration test: runs app.main()'s real loop with the webcam, MediaPipe
and the TFLite classifier replaced by scripted stand-ins, and checks that
the speaker is asked to say each gesture exactly once after it has been
stable for --tts_stable_frames frames.

Run with:  python -m pytest tests/ -v
"""
import os
import sys
import shutil
import tempfile
import types

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


# Frame-by-frame script: classifier output id, or None for "no hand".
# labels: 0=hello 1=thanks 2=same
SCRIPT = (
    [0] * 4 +            # hello, not yet stable
    [None] * 2 +         # hand lost -> streak reset
    [0] * 25 +           # hello stable at frame 10 of this run -> speak once
    [1] * 3 + [0] * 3 +  # flicker, nothing spoken
    [1] * 12 +           # thanks stable -> speak once
    [None] * 5 +
    [1] * 20 +           # thanks again after hand lost -> NOT repeated
    [2] * 12 +           # same -> speak once
    [0] * 12             # hello again (after a different word) -> speak
)
EXPECTED_SPOKEN = ["hello", "thanks", "same", "hello"]


@pytest.fixture
def scratch_repo():
    d = tempfile.mkdtemp()
    for name in ("app.py", "tts_speaker.py", "landmark_utils.py"):
        shutil.copy(os.path.join(ROOT, name), d)
    shutil.copytree(os.path.join(ROOT, "utils"), os.path.join(d, "utils"))
    os.makedirs(os.path.join(d, "model", "keypoint_classifier"))
    with open(os.path.join(d, "model", "keypoint_classifier",
                           "keypoint_classifier_label.csv"), "w") as f:
        f.write("hello\nthanks\nsame\n")
    # minimal 'model' package so `from model import KeyPointClassifier` works
    with open(os.path.join(d, "model", "__init__.py"), "w") as f:
        f.write("class KeyPointClassifier: pass\n")
    # Some MediaPipe builds ship only the new `tasks` API; app.py uses the
    # legacy `mp.solutions.hands`. Provide a stub namespace if missing so
    # this test only exercises app.py's own wiring, not MediaPipe itself.
    import mediapipe as mp
    if not hasattr(mp, "solutions"):
        mp.solutions = types.SimpleNamespace(
            hands=types.SimpleNamespace(Hands=None))
    cwd = os.getcwd()
    os.chdir(d)
    sys.path.insert(0, d)
    yield d
    os.chdir(cwd)
    sys.path.remove(d)
    shutil.rmtree(d)
    for m in ("app", "model", "utils", "tts_speaker", "landmark_utils"):
        sys.modules.pop(m, None)


def test_main_loop_speaks_once_per_stable_gesture(scratch_repo, monkeypatch):
    import cv2 as cv
    import app

    frames = iter(SCRIPT)
    state = {"current": None, "frame": 0}
    spoken = []

    # --- camera: synthetic frames, ends when the script runs out ---------
    class FakeCap:
        def set(self, *a): pass
        def read(self):
            try:
                state["current"] = next(frames)
                state["frame"] += 1
            except StopIteration:
                return False, None
            return True, np.zeros((540, 960, 3), dtype=np.uint8)
        def release(self): pass

    monkeypatch.setattr(cv, "VideoCapture", lambda *a: FakeCap())
    monkeypatch.setattr(cv, "imshow", lambda *a: None)
    monkeypatch.setattr(cv, "waitKey", lambda *a: -1)
    monkeypatch.setattr(cv, "destroyAllWindows", lambda: None)

    # --- MediaPipe: we only need .process() to return something ----------
    class FakeHands:
        def process(self, img):
            return types.SimpleNamespace(multi_hand_landmarks=None,
                                         multi_handedness=None)
    monkeypatch.setattr(app.mp.solutions.hands, "Hands",
                        lambda **kw: FakeHands())

    # --- feature vector: fake hand present/absent per script -------------
    lm = [[100 + i * 5, 200 + i * 3] for i in range(21)]
    def fake_build(image, hand_landmarks, handedness):
        if state["current"] is None:
            return [0.0] * 84, []
        return [0.1] * 84, [("Right", lm, [90, 190, 220, 270])]
    monkeypatch.setattr(app, "build_combined_vector", fake_build)

    # --- classifier: scripted ids ---------------------------------------
    class FakeClassifier:
        def __call__(self, vec):
            return state["current"]
    monkeypatch.setattr(app, "KeyPointClassifier", FakeClassifier)

    # --- speaker: record what would be said ------------------------------
    class FakeSpeaker:
        available = True
        def __init__(self, rate=None): pass
        def speak(self, text): spoken.append((state["frame"], text))
        def stop(self): spoken.append("STOPPED")
    monkeypatch.setattr(app, "TTSSpeaker", FakeSpeaker)

    monkeypatch.setattr(sys, "argv", ["app.py", "--tts_stable_frames", "10"])
    app.main()

    assert spoken[-1] == "STOPPED", "speaker.stop() must run on exit"
    words = [w for w in spoken[:-1]]
    assert [w for _, w in words] == EXPECTED_SPOKEN
    # first 'hello' spoken exactly on the 10th consecutive frame after reset
    assert words[0][0] == 4 + 2 + 10
    assert state["frame"] == len(SCRIPT), "loop ran through the whole script"


def test_no_tts_flag_never_creates_speaker(scratch_repo, monkeypatch):
    import cv2 as cv
    import app

    created = []
    monkeypatch.setattr(app, "TTSSpeaker",
                        lambda **kw: created.append(1))
    class Cap:
        def set(self, *a): pass
        def read(self): return False, None
        def release(self): pass
    monkeypatch.setattr(cv, "VideoCapture", lambda *a: Cap())
    monkeypatch.setattr(cv, "waitKey", lambda *a: -1)
    monkeypatch.setattr(cv, "destroyAllWindows", lambda: None)
    monkeypatch.setattr(app.mp.solutions.hands, "Hands", lambda **kw: None)
    monkeypatch.setattr(app, "KeyPointClassifier", lambda: None)
    monkeypatch.setattr(sys, "argv", ["app.py", "--no_tts"])
    app.main()
    assert created == []
