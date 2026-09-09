"""Decoupled capture / processing with timestamp-based scheduling.

Two pieces:

``FrameBuffer(capacity)``
    Thread-safe bounded buffer between the capture thread and the processing
    loop (``realtime.max_queue_size``). ``put`` drops the OLDEST frame when the
    buffer is full and counts it as dropped, so the buffer can never fall
    behind the camera by more than ``capacity`` frames. ``take(newest=True)``
    (``realtime.latest_frame_strategy``) returns the newest frame and discards
    the rest (also counted as dropped); ``take(newest=False)`` is bounded FIFO.
    With capacity 1 both modes are the classic latest-frame mailbox.
    ``LatestFrameSlot`` is kept as an alias for capacity 1.

``RateScheduler``
    Enforces a target processing rate (default 20 FPS = 50 ms interval)
    using ``time.monotonic()`` deadlines instead of "every N-th frame".
    Camera rate and processing rate are therefore independent: a 30 FPS
    camera with a 20 FPS target is processed at 20 FPS; a 15 FPS camera is
    processed at 15 FPS (every fresh frame, no busy-looping).

    Deadline bookkeeping is "fixed-interval with re-anchoring": after each
    tick the next deadline is ``previous + interval`` (so jitter and variable
    processing time do not accumulate as drift), but if processing falls more
    than ``max_catchup_intervals`` behind, the deadline is re-anchored to
    ``now + interval`` rather than firing a burst of catch-up ticks — bursts
    would only process stale data.
"""
from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Deque, Generic, Optional, TypeVar

T = TypeVar("T")


@dataclass
class TimedFrame(Generic[T]):
    frame: T
    captured_at: float  # monotonic seconds
    seq: int  # capture sequence number


class FrameBuffer(Generic[T]):
    def __init__(self, capacity: int = 1) -> None:
        if capacity < 1:
            raise ValueError("capacity must be >= 1")
        self.capacity = capacity
        self._lock = threading.Lock()
        self._cond = threading.Condition(self._lock)
        self._items: Deque[TimedFrame[T]] = deque()
        self._seq = 0
        self.dropped = 0  # frames never processed (overflow + skipped by newest-take)
        self.received = 0

    def put(self, frame: T, captured_at: Optional[float] = None) -> None:
        with self._cond:
            if len(self._items) >= self.capacity:
                self._items.popleft()
                self.dropped += 1
            self._seq += 1
            self.received += 1
            self._items.append(TimedFrame(frame, captured_at if captured_at is not None else time.monotonic(), self._seq))
            self._cond.notify()

    def take(self, timeout: Optional[float] = None, newest: bool = True) -> Optional[TimedFrame[T]]:
        """Return a frame (newest by default, discarding older ones). None on timeout."""
        with self._cond:
            if not self._items and timeout != 0:
                self._cond.wait(timeout)
            if not self._items:
                return None
            if newest:
                item = self._items.pop()
                self.dropped += len(self._items)
                self._items.clear()
            else:
                item = self._items.popleft()
            return item

    def size(self) -> int:
        with self._lock:
            return len(self._items)

    def peek_age(self) -> Optional[float]:
        with self._lock:
            return None if not self._items else time.monotonic() - self._items[-1].captured_at


class LatestFrameSlot(FrameBuffer[T]):
    """Capacity-1 FrameBuffer (kept for backwards compatibility)."""

    def __init__(self) -> None:
        super().__init__(capacity=1)


class RateScheduler:
    def __init__(self, target_fps: float, max_catchup_intervals: int = 2, clock=time.monotonic) -> None:
        if target_fps <= 0:
            raise ValueError("target_fps must be > 0")
        self.interval = 1.0 / target_fps
        self.max_catchup = max(1, int(max_catchup_intervals))
        self._clock = clock
        self._next: Optional[float] = None
        self.ticks = 0
        self.reanchors = 0
        self.last_late_s = 0.0

    @property
    def target_fps(self) -> float:
        return 1.0 / self.interval

    def reset(self) -> None:
        self._next = None

    def time_until_due(self) -> float:
        now = self._clock()
        if self._next is None:
            self._next = now
        return self._next - now

    def wait_until_due(self, sleep=time.sleep) -> float:
        """Block until the next deadline; returns the lateness in seconds (>=0)."""
        remaining = self.time_until_due()
        if remaining > 0:
            sleep(remaining)
        now = self._clock()
        late = max(now - self._next, 0.0)
        self._advance(now)
        self.last_late_s = late
        return late

    def _advance(self, now: float) -> None:
        self.ticks += 1
        self._next += self.interval
        if now - self._next > self.max_catchup * self.interval:
            # Too far behind: don't fire a burst of catch-up ticks (they would
            # only process stale data). Re-anchor: next tick one interval from now.
            self._next = now + self.interval
            self.reanchors += 1
