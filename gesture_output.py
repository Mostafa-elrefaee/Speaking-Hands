#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
gesture_output.py

Post-processing that sits between the classifiers and the user in app.py:

    static prediction  ─┐
                        ├─> choose_gesture() ─> GestureStabilizer ─┬─> on-screen text
    sequence prediction ─┘        (priority)      (debounce)       └─> SpeechWorker (TTS)

The stabilizer's `current` value is the ONE source of truth for "what gesture
is being shown right now". Both the screen and the speech output read from
it; there is deliberately no second debounce for speech.

This module has no cv2 / mediapipe / tensorflow imports so it can be unit
tested with plain fake predictions (see test_gesture_output.py).
"""
import sys
import queue
import threading
import subprocess


# ---------------------------------------------------------------------------
# Priority between the two classifiers
# ---------------------------------------------------------------------------

def choose_gesture(seq_label, static_label):
    """
    Pick this frame's raw gesture from the two classifiers' outputs.

    seq_label:    label string from the sequence classifier, or None if it
                  returned invalid_value (not enough history / low confidence)
                  or isn't loaded at all.
    static_label: label string from the static classifier, or None if no
                  hand was detected this frame.

    Returns (label, source) where source is "seq", "static" or None.

    The sequence classifier is authoritative when it speaks up, because:
      * its output is confidence-gated (score_th) while KeyPointClassifier
        always returns an argmax with no threshold, so a valid sequence
        result carries strictly more evidence than a static one;
      * it sees a superset of the information (the current frame plus the
        preceding window), so a motion sign that happens to pass through a
        known static hand-shape would otherwise be misread as that shape.
    The static classifier is the fallback for still signs and for the
    warm-up period before the buffer is full.
    """
    if seq_label is not None:
        return seq_label, "seq"
    if static_label is not None:
        return static_label, "static"
    return None, None


# ---------------------------------------------------------------------------
# Stability / debounce
# ---------------------------------------------------------------------------

class GestureStabilizer(object):
    """
    Promotes a raw per-frame prediction to the "current gesture" only after
    it has been identical for `stable_frames` consecutive frames.

    `None` is a valid raw value meaning "no gesture / no hand" and is
    debounced exactly like any label, so the on-screen text clears (and a
    repeated sign can be spoken again) only after the hands have genuinely
    been gone for a while, not on a single dropped frame.
    """

    def __init__(self, stable_frames):
        if stable_frames < 1:
            raise ValueError("stable_frames must be >= 1")
        self.stable_frames = stable_frames
        self.current = None        # the stabilized value (single source of truth)
        self._candidate = None     # value seen on the most recent frame
        self._run_length = 0       # how many consecutive frames it has been seen

    def update(self, raw):
        """
        Feed this frame's raw prediction. Returns (current, changed) where
        `changed` is True only on the frame where `current` switches to a
        different value -- i.e. exactly one "stabilization event" per
        distinct gesture. Holding the same sign never re-triggers it.
        """
        if raw == self._candidate:
            self._run_length += 1
        else:
            self._candidate = raw
            self._run_length = 1

        changed = False
        if self._run_length >= self.stable_frames and raw != self.current:
            self.current = raw
            changed = True
        return self.current, changed


# ---------------------------------------------------------------------------
# Text-to-speech on a background thread
# ---------------------------------------------------------------------------

def create_pyttsx3_engine():
    """Import + init pyttsx3 with actionable errors instead of tracebacks."""
    try:
        import pyttsx3
    except ImportError:
        raise RuntimeError(
            "Text-to-speech needs the 'pyttsx3' package, which is not "
            "installed. Fix:  pip install pyttsx3   (or run app.py with "
            "--no_tts to disable speech)")
    try:
        return pyttsx3.init()
    except Exception as e:  # pyttsx3 raises plain RuntimeError/OSError
        hint = ""
        if "espeak" in str(e).lower():
            hint = ("  On Linux pyttsx3 needs eSpeak:  "
                    "sudo apt install espeak-ng")
        raise RuntimeError(
            "pyttsx3 is installed but could not start a speech engine: "
            f"{e}.{hint}  (or run app.py with --no_tts)")


# Script run once PER WORD in a fresh Python process. pyttsx3 has a long-
# standing bug where the SECOND engine.runAndWait() on the same engine hangs
# (Windows SAPI5) or goes silent (macOS) -- only Linux/eSpeak is immune. A
# fresh process per utterance sidesteps it on every platform; the ~0.2-0.5 s
# startup happens on the background thread, never in the camera loop.
_SPEAK_SCRIPT = (
    "import sys, pyttsx3\n"
    "e = pyttsx3.init()\n"
    "e.say(sys.argv[1])\n"
    "e.runAndWait()\n"
)


def speak_in_subprocess(text, timeout=15.0):
    """Blocking: speak `text` in an isolated process. Returns the Popen so a
    caller can terminate it on shutdown."""
    return subprocess.Popen([sys.executable, "-c", _SPEAK_SCRIPT, text],
                            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)


class SpeechWorker(object):
    """
    Speaks strings on a daemon thread so the webcam loop never blocks.

    Policy when a new gesture stabilizes while speech is playing: QUEUE it,
    don't interrupt. Every stabilized gesture is a word the signer meant to
    say; cutting one off mid-word (or dropping it) changes the sentence,
    while a short bounded delay does not. Utterances are single words
    (~0.3-0.6 s), so the backlog stays small. `maxsize` is only a safety
    valve: if it is ever hit, the OLDEST queued (not-yet-spoken) word is
    dropped and a warning printed, so speech can't lag unboundedly.

    Backends:
      * default (engine_factory=None): one subprocess per word -- see
        _SPEAK_SCRIPT for why. Robust on Windows / macOS / Linux.
      * engine_factory given: in-process, but a FRESH engine per word
        (engine.stop() + drop the reference after each utterance). Used by
        the tests with a fake engine; also usable on Linux if you'd rather
        avoid process spawns.
    """

    def __init__(self, engine_factory=None, maxsize=16):
        self._engine_factory = engine_factory
        self._queue = queue.Queue(maxsize=maxsize)
        self._ready = threading.Event()
        self._init_error = None
        self._dropped = 0
        self._child = None          # in-flight subprocess (subprocess mode)
        self._closing = False
        self._thread = threading.Thread(target=self._run, name="tts-worker",
                                        daemon=True)
        self.spoken = []  # everything successfully spoken (for tests / debug)

    # -- lifecycle -----------------------------------------------------------

    def start(self, init_timeout=15.0):
        """Start the thread and wait for a one-time engine probe, so a
        missing/broken TTS backend fails loudly here rather than silently
        inside the thread later."""
        self._thread.start()
        if not self._ready.wait(init_timeout):
            raise RuntimeError("Speech engine did not initialise within "
                               f"{init_timeout}s")
        if self._init_error is not None:
            raise self._init_error
        return self

    def close(self, timeout=3.0):
        """Ask the thread to exit; kill any in-flight utterance. Never
        raises; never blocks longer than `timeout` (and the thread is a
        daemon, so even a stuck backend can't keep the process alive)."""
        self._closing = True
        try:
            self._queue.put_nowait(None)
        except queue.Full:
            while True:
                try:
                    self._queue.get_nowait()
                except queue.Empty:
                    break
            self._queue.put_nowait(None)
        child = self._child
        if child is not None:
            try:
                child.terminate()
            except Exception:
                pass
        if self._thread.is_alive():
            self._thread.join(timeout)

    # -- producer side (main loop) --------------------------------------------

    def say(self, text):
        if not text:
            return
        try:
            self._queue.put_nowait(text)
        except queue.Full:
            try:
                dropped = self._queue.get_nowait()
            except queue.Empty:
                dropped = None
            self._dropped += 1
            print(f"[tts] speech backlog full, dropped oldest unspoken "
                  f"word: {dropped!r}")
            self._queue.put_nowait(text)

    @property
    def pending(self):
        return self._queue.qsize()

    # -- worker thread -------------------------------------------------------

    def _probe(self):
        """Create-and-discard one engine so import/driver errors surface."""
        factory = self._engine_factory or create_pyttsx3_engine
        engine = factory()
        try:
            stop = getattr(engine, "stop", None)
            if stop is not None:
                stop()
        except Exception:
            pass
        del engine

    def _speak_one(self, text):
        if self._engine_factory is None:
            self._child = speak_in_subprocess(text)
            try:
                _, err = self._child.communicate()
                rc = self._child.returncode
            finally:
                self._child = None
            if rc != 0 and not self._closing:
                tail = (err or b"").decode(errors="replace").strip().splitlines()
                raise RuntimeError(tail[-1] if tail else f"exit code {rc}")
        else:
            engine = self._engine_factory()
            try:
                engine.say(text)
                engine.runAndWait()
            finally:
                try:
                    stop = getattr(engine, "stop", None)
                    if stop is not None:
                        stop()
                except Exception:
                    pass
                del engine  # let pyttsx3's cached engine die before the next init

    def _run(self):
        try:
            self._probe()
        except Exception as e:
            self._init_error = e
            self._ready.set()
            return
        self._ready.set()

        while True:
            item = self._queue.get()
            if item is None:
                break
            try:
                self._speak_one(item)
                self.spoken.append(item)
            except Exception as e:
                # A failed utterance must not kill the worker or the app.
                if not self._closing:
                    print(f"[tts] failed to speak {item!r}: {e}")
