#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
tts_speaker.py

Offline text-to-speech for the live gesture recognizer (app.py).

Two small pieces:

  GestureStabilizer
      Decides WHEN a gesture should be spoken. The classifier makes a new
      prediction every frame, so speaking the raw output would repeat the
      same word 30 times a second. Instead, a label is only "confirmed"
      once it has been predicted for `stable_frames` consecutive frames,
      and a confirmed label is spoken exactly once -- it is not spoken
      again until the prediction changes to a *different* label and that
      new label stabilizes in turn.

  TTSSpeaker
      Speaks confirmed labels using pyttsx3 (fully offline, no API key).
      pyttsx3's runAndWait() blocks for the duration of the utterance, so
      it must never run inside the webcam loop. All speech happens on a
      dedicated daemon thread fed through a queue; the main loop only ever
      does a non-blocking queue.put().

Both pieces are independent of OpenCV / MediaPipe so they can be unit
tested without a camera (see tests/test_tts_speaker.py).
"""
import queue
import threading


class GestureStabilizer(object):
    """Turns a per-frame stream of predicted labels into 'speak this now'
    events.

    Feed every frame's prediction to update(). Pass None on frames where
    no hand was detected. update() returns the label on the single frame
    where it becomes newly confirmed, and None on every other frame.

    Rules:
      * A label is confirmed after `stable_frames` consecutive identical
        predictions.
      * A confirmed label is returned once. It is not returned again while
        it remains the most recently spoken label, even if the streak is
        broken (hand lost, momentary misclassification) and re-formed.
      * A different label must stabilize before anything is spoken again;
        after that, the original label may be spoken again if it returns.
    """

    def __init__(self, stable_frames=10):
        if stable_frames < 1:
            raise ValueError("stable_frames must be >= 1")
        self.stable_frames = stable_frames
        self._current = None       # label of the running streak
        self._count = 0            # length of the running streak
        self._last_spoken = None   # most recently confirmed label

    def update(self, label):
        """Feed one frame's prediction. Returns the label to speak (once,
        on the frame it becomes confirmed) or None."""
        if label is None:
            # No hand this frame: the streak breaks, but we remember what
            # was last spoken so it isn't repeated when the hand returns.
            self._current = None
            self._count = 0
            return None

        if label == self._current:
            self._count += 1
        else:
            self._current = label
            self._count = 1

        if self._count == self.stable_frames and label != self._last_spoken:
            self._last_spoken = label
            return label
        return None

    def reset(self):
        """Forget everything, including the last spoken label."""
        self._current = None
        self._count = 0
        self._last_spoken = None


class TTSSpeaker(object):
    """Non-blocking offline speech via pyttsx3 on a background thread.

    Usage:
        speaker = TTSSpeaker()
        speaker.speak("hello")     # returns immediately
        ...
        speaker.stop()             # on shutdown

    If pyttsx3 is missing or fails to initialise (no speech backend on the
    machine), the speaker degrades gracefully: `available` is False, a
    warning is printed once, and speak() becomes a no-op so the video feed
    keeps running.
    """

    def __init__(self, rate=None, volume=None, max_queue=5,
                 engine_factory=None):
        """
        rate / volume   optional pyttsx3 properties (words per minute, 0-1)
        max_queue       max utterances waiting to be spoken; if the queue
                        is full, new words are dropped rather than piling
                        up a backlog of stale speech
        engine_factory  callable returning an object with
                        say(text) / runAndWait() -- defaults to
                        pyttsx3.init. Exposed for tests.
        """
        self._queue = queue.Queue(maxsize=max_queue)
        self._rate = rate
        self._volume = volume
        self._engine_factory = engine_factory
        self._stop_event = threading.Event()
        self._ready = threading.Event()
        self.available = False
        self.error = None

        self._thread = threading.Thread(
            target=self._worker, name="tts-speaker", daemon=True)
        self._thread.start()
        # Wait briefly for the engine to initialise so `available` is
        # meaningful right after construction. Cap the wait so a slow
        # backend can't hold up camera start-up.
        self._ready.wait(timeout=5.0)

    # -- public API -------------------------------------------------------

    def speak(self, text):
        """Queue `text` to be spoken. Never blocks the caller."""
        if not text or not self.available:
            return False
        try:
            self._queue.put_nowait(text)
            return True
        except queue.Full:
            return False

    def stop(self, timeout=2.0):
        """Ask the worker to finish and wait briefly for it."""
        self._stop_event.set()
        try:
            self._queue.put_nowait(None)   # wake the worker if idle
        except queue.Full:
            pass
        self._thread.join(timeout=timeout)

    # -- internals --------------------------------------------------------

    def _make_engine(self):
        # The pyttsx3 engine is created INSIDE the worker thread: the
        # driver objects are not thread-safe and must be used from the
        # thread that created them.
        if self._engine_factory is not None:
            engine = self._engine_factory()
        else:
            import pyttsx3
            engine = pyttsx3.init()
        if self._rate is not None:
            engine.setProperty('rate', self._rate)
        if self._volume is not None:
            engine.setProperty('volume', self._volume)
        return engine

    def _worker(self):
        try:
            engine = self._make_engine()
            self.available = True
        except Exception as e:  # ImportError, missing driver, no audio...
            self.error = e
            self.available = False
            print("[TTS] Speech disabled: %s" % e)
            print("[TTS] Install with:  pip install pyttsx3   "
                  "(Linux also needs espeak / espeak-ng)")
            self._ready.set()
            return
        self._ready.set()

        while not self._stop_event.is_set():
            text = self._queue.get()
            if text is None:
                break
            try:
                engine.say(text)
                engine.runAndWait()
            except Exception as e:
                print("[TTS] Failed to speak %r: %s" % (text, e))
        try:
            engine.stop()
        except Exception:
            pass
