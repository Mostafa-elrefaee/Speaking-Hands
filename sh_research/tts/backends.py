"""TTS backends behind one small interface.

    prewarm()      initialise once (probe the engine / pre-spawn a process); returns ms spent
    speak(text)    synthesise + play, BLOCKING; returns (T5 synthesis_done, T6 playback_start, T7 playback_end)
    stop()         interrupt current playback if the backend can do so safely
    is_busy        True while speak() is running
    supports_interrupt
    close()

``speak`` is only ever called from the TTSWorker thread, never from the
inference loop.

The default backend, ``Pyttsx3SubprocessBackend``, IS the original Speaking
Hands backend (gesture_output.SpeechWorker's "one fresh pyttsx3 process per
word"), now with two measurable improvements:

* the child prints START when pyttsx3 fires ``started-utterance`` and END when
  it finishes, so T6 (audio start) is a real measurement, not a guess;
* optional ``prespawn``: the next process is started (Python + pyttsx3.init())
  BEFORE the next word arrives and waits for text on stdin, moving the
  ~0.2-0.5 s interpreter start-up off the prediction-to-speech path. Each
  process still speaks exactly once, so the runAndWait-twice hang never occurs.
"""
from __future__ import annotations

import os
import subprocess
import sys
import threading
import time
from typing import Callable, Optional, Protocol, Tuple

Timestamps = Tuple[Optional[float], Optional[float], Optional[float]]


class TTSBackend(Protocol):
    name: str
    supports_interrupt: bool

    def prewarm(self) -> float: ...

    def speak(self, text: str) -> Timestamps: ...

    def stop(self) -> None: ...

    @property
    def is_busy(self) -> bool: ...

    def close(self) -> None: ...


# --------------------------------------------------------------------------- #
# pyttsx3 helpers (moved here from gesture_output.py, unchanged behaviour)
# --------------------------------------------------------------------------- #
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
# fresh process per utterance sidesteps it on every platform.
#
# Protocol (stdin/stdout, one line each):
#   child -> READY            engine initialised, waiting for text
#   parent -> <text>          the word to speak
#   child -> START            pyttsx3 started-utterance (audio begins)   = T6
#   child -> END              finished-utterance / runAndWait returned   = T7
_SPEAK_SCRIPT = (
    "import sys, pyttsx3\n"
    "rate, vol, voice = float(sys.argv[1]), float(sys.argv[2]), sys.argv[3]\n"
    "e = pyttsx3.init()\n"
    "if rate > 0: e.setProperty('rate', int(rate))\n"
    "if vol > 0: e.setProperty('volume', vol)\n"
    "if voice:\n"
    "    for v in e.getProperty('voices'):\n"
    "        if voice.lower() in (v.id.lower(), (v.name or '').lower()):\n"
    "            e.setProperty('voice', v.id); break\n"
    "out = sys.stdout\n"
    "def _w(s):\n"
    "    out.write(s + '\\n'); out.flush()\n"
    "try:\n"
    "    e.connect('started-utterance', lambda name: _w('START'))\n"
    "except Exception:\n"
    "    pass\n"
    "_w('READY')\n"
    "text = sys.stdin.readline().rstrip('\\r\\n')\n"
    "if not text:\n"
    "    sys.exit(0)\n"
    "e.say(text)\n"
    "e.runAndWait()\n"
    "_w('END')\n"
)


class _Child:
    """One speaking process: spawned, initialised (READY), speaks once."""

    def __init__(self, rate: int, volume: float, voice: str) -> None:
        env = dict(os.environ, PYTHONIOENCODING="utf-8")
        self.proc = subprocess.Popen([sys.executable, "-c", _SPEAK_SCRIPT, str(rate), str(volume), voice],
                                     stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                     env=env, text=True, encoding="utf-8", errors="replace", bufsize=1)
        self.ready_at: Optional[float] = None
        self.spawned_at = time.monotonic()

    def wait_ready(self) -> bool:
        line = self.proc.stdout.readline()
        self.ready_at = time.monotonic()
        return line.strip() == "READY"

    def terminate(self) -> None:
        try:
            self.proc.terminate()
        except Exception:
            pass


class _Base:
    name = "base"
    supports_interrupt = False

    def __init__(self) -> None:
        self._busy = threading.Event()

    @property
    def is_busy(self) -> bool:
        return self._busy.is_set()

    def prewarm(self) -> float:
        return 0.0

    def stop(self) -> None:
        pass

    def close(self) -> None:
        pass


class Pyttsx3SubprocessBackend(_Base):
    """The original Speaking Hands backend (fresh process per word)."""

    name = "pyttsx3_subprocess"
    supports_interrupt = True  # killing the child stops audio; no engine state to corrupt

    def __init__(self, rate: int = 0, volume: float = 0.0, voice: str = "", prespawn: bool = True,
                 timeout_s: float = 30.0) -> None:
        super().__init__()
        self._rate, self._volume, self._voice = int(rate), float(volume), voice
        self.prespawn = prespawn
        self.timeout_s = timeout_s
        self._standby: Optional[_Child] = None
        self._current: Optional[_Child] = None
        self._closing = False
        self.spawn_ms: list = []  # measured spawn->READY times
        self.no_start_marker = 0

    # -- process management -----------------------------------------------------
    def _spawn(self) -> _Child:
        return _Child(self._rate, self._volume, self._voice)

    def _take_child(self) -> _Child:
        child = self._standby
        self._standby = None
        if child is None:
            child = self._spawn()
        if child.ready_at is None:  # cold: wait for the engine to initialise
            if not child.wait_ready():
                err = child.proc.stderr.read()
                raise RuntimeError((err or "").strip().splitlines()[-1] if err and err.strip()
                                   else "speech process failed to initialise")
            self.spawn_ms.append((child.ready_at - child.spawned_at) * 1e3)
        return child

    def _prespawn_next(self) -> None:
        if self.prespawn and not self._closing and self._standby is None:
            self._standby = self._spawn()

    # -- interface ------------------------------------------------------------------
    def prewarm(self) -> float:
        """Probe the engine in-process (surfaces install/driver errors with the
        original actionable messages) and pre-spawn the first process."""
        t = time.monotonic()
        engine = create_pyttsx3_engine()
        try:
            stop = getattr(engine, "stop", None)
            if stop is not None:
                stop()
        except Exception:
            pass
        del engine
        if self.prespawn:
            self._standby = self._spawn()
            if not self._standby.wait_ready():
                self._standby = None
            else:
                self.spawn_ms.append((self._standby.ready_at - self._standby.spawned_at) * 1e3)
        return (time.monotonic() - t) * 1e3

    def speak(self, text: str) -> Timestamps:
        self._busy.set()
        child = None
        try:
            child = self._take_child()
            self._current = child
            self._prespawn_next()  # overlaps the next interpreter start-up with this playback
            watchdog = threading.Timer(self.timeout_s, child.terminate)
            watchdog.daemon = True
            watchdog.start()
            try:
                child.proc.stdin.write(text.replace("\n", " ") + "\n")
                child.proc.stdin.flush()
                t_start: Optional[float] = None
                t_end: Optional[float] = None
                while True:
                    line = child.proc.stdout.readline()
                    if not line:
                        break
                    line = line.strip()
                    if line == "START":
                        t_start = time.monotonic()
                    elif line == "END":
                        t_end = time.monotonic()
                        break
                _, err = child.proc.communicate()
                rc = child.proc.returncode
            finally:
                watchdog.cancel()
            if t_end is None:
                t_end = time.monotonic()
            if rc != 0 and not self._closing:
                tail = (err or "").strip().splitlines()
                raise RuntimeError(tail[-1] if tail else f"exit code {rc}")
            if t_start is None:
                self.no_start_marker += 1
            return t_start, t_start, t_end
        finally:
            self._current = None
            self._busy.clear()

    def stop(self) -> None:
        cur = self._current
        if cur is not None:
            cur.terminate()

    def close(self) -> None:
        self._closing = True
        for c in (self._current, self._standby):
            if c is not None:
                c.terminate()
        self._standby = None


class Pyttsx3Backend(_Base):
    """In-process pyttsx3. A FRESH engine is created per word and dropped
    afterwards (engine.stop() + del), exactly like the old
    SpeechWorker(engine_factory=...) mode, because reusing one engine hangs on
    Windows SAPI5 after the second runAndWait(). ``engine_factory`` lets tests
    inject a fake engine (say / runAndWait / stop)."""

    name = "pyttsx3"
    supports_interrupt = False

    def __init__(self, rate: int = 0, volume: float = 0.0, voice: str = "",
                 engine_factory: Optional[Callable[[], object]] = None) -> None:
        super().__init__()
        self._rate, self._volume, self._voice = rate, volume, voice
        self._factory = engine_factory or create_pyttsx3_engine
        self._started: Optional[float] = None
        self._finished: Optional[float] = None

    def _make_engine(self):
        eng = self._factory()
        if self._rate > 0 and hasattr(eng, "setProperty"):
            eng.setProperty("rate", int(self._rate))
        if self._volume > 0 and hasattr(eng, "setProperty"):
            eng.setProperty("volume", self._volume)
        if self._voice and hasattr(eng, "getProperty"):
            for v in eng.getProperty("voices"):
                if self._voice.lower() in (v.id.lower(), (v.name or "").lower()):
                    eng.setProperty("voice", v.id)
                    break
        if hasattr(eng, "connect"):
            try:
                eng.connect("started-utterance", lambda name: setattr(self, "_started", time.monotonic()))
                eng.connect("finished-utterance", lambda name, completed: setattr(self, "_finished", time.monotonic()))
            except Exception:
                pass
        return eng

    def prewarm(self) -> float:
        """Create-and-discard one engine so import/driver errors surface here."""
        t = time.monotonic()
        eng = self._make_engine()
        self._dispose(eng)
        return (time.monotonic() - t) * 1e3

    @staticmethod
    def _dispose(eng) -> None:
        try:
            stop = getattr(eng, "stop", None)
            if stop is not None:
                stop()
        except Exception:
            pass

    def speak(self, text: str) -> Timestamps:
        self._busy.set()
        try:
            eng = self._make_engine()
            self._started = self._finished = None
            try:
                eng.say(text)
                eng.runAndWait()
            finally:
                self._dispose(eng)
                del eng  # let pyttsx3's cached engine die before the next init
            end = self._finished or time.monotonic()
            start = self._started  # None when the engine has no callbacks (fakes)
            return start, start, end
        finally:
            self._busy.clear()


class PrintBackend(_Base):
    name = "print"

    def speak(self, text: str) -> Timestamps:
        t = time.monotonic()
        print(f"[TTS] {text}", flush=True)
        return t, t, t


class MockBackend(_Base):
    """Simulates an engine: one-off init cost, per-utterance synthesis latency,
    playback proportional to text length (or fixed), interruptible playback."""

    name = "mock"
    supports_interrupt = True

    def __init__(self, latency_s: float = 0.05, playback_s: float = 0.0, init_s: float = 0.0,
                 playback_per_char_s: float = 0.0, fail_texts: Optional[set] = None) -> None:
        super().__init__()
        self.latency_s, self.playback_s, self.init_s, self.per_char = latency_s, playback_s, init_s, playback_per_char_s
        self.fail_texts = fail_texts or set()
        self.spoken: list = []
        self.interrupted = 0
        self._initialised = False
        self._stop_evt = threading.Event()

    def _ensure(self) -> None:
        if not self._initialised:
            time.sleep(self.init_s)
            self._initialised = True

    def prewarm(self) -> float:
        t = time.monotonic()
        self._ensure()
        return (time.monotonic() - t) * 1e3

    def speak(self, text: str) -> Timestamps:
        self._busy.set()
        self._stop_evt.clear()
        try:
            self._ensure()
            if text in self.fail_texts:
                raise RuntimeError(f"synthesis failed for {text!r}")
            time.sleep(self.latency_s)
            gen_done = time.monotonic()
            play_start = gen_done
            dur = self.playback_s + self.per_char * len(text)
            if self._stop_evt.wait(dur):
                self.interrupted += 1
            self.spoken.append(text)
            return gen_done, play_start, time.monotonic()
        finally:
            self._busy.clear()

    def stop(self) -> None:
        self._stop_evt.set()


def make_backend(kind: str, **kw) -> TTSBackend:
    if kind == "pyttsx3_subprocess":
        return Pyttsx3SubprocessBackend(rate=kw.get("rate", 0), volume=kw.get("volume", 0.0),
                                        voice=kw.get("voice", ""), prespawn=kw.get("prespawn", True))
    if kind == "pyttsx3":
        return Pyttsx3Backend(rate=kw.get("rate", 0), volume=kw.get("volume", 0.0), voice=kw.get("voice", ""),
                              engine_factory=kw.get("engine_factory"))
    if kind == "print":
        return PrintBackend()
    if kind == "mock":
        return MockBackend(latency_s=kw.get("mock_latency_s", 0.05), init_s=kw.get("mock_init_s", 0.0))
    raise ValueError(f"Unknown TTS backend: {kind}")
