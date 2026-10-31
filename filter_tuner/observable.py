"""Tiny observer primitives used to wire the app together without Qt.

``Observable`` holds a value and notifies subscribers on change.
``UndoableObservable`` adds undo/redo with gesture grouping (a mouse drag
becomes one undo step) and coalescing (spinning a spin box becomes one step).
"""

from __future__ import annotations

import time
from typing import Callable, Generic, TypeVar

T = TypeVar("T")
Listener = Callable[[T], None]


class Observable(Generic[T]):
    def __init__(self, value: T):
        self._value = value
        self._listeners: list[Listener] = []

    @property
    def value(self) -> T:
        return self._value

    def set(self, value: T) -> None:
        if value == self._value:
            return
        self._value = value
        self._notify()

    def subscribe(self, fn: Listener, call_now: bool = False) -> Callable[[], None]:
        self._listeners.append(fn)
        if call_now:
            fn(self._value)
        return lambda: self._listeners.remove(fn)

    def _notify(self) -> None:
        for fn in list(self._listeners):
            fn(self._value)


class UndoableObservable(Observable[T]):
    COALESCE_SECONDS = 1.0

    def __init__(self, value: T, limit: int = 500):
        super().__init__(value)
        self._undo: list[T] = []
        self._redo: list[T] = []
        self._limit = limit
        self._gesture_start: T | None = None
        self._last_key: object = None
        self._last_time = 0.0
        self.history_changed = Observable(0)  # bumps on any history change

    # --- recording edits -------------------------------------------------
    def set(self, value: T, coalesce_key: object = None) -> None:
        if value == self._value:
            return
        if self._gesture_start is None:
            now = time.monotonic()
            same = (
                coalesce_key is not None
                and coalesce_key == self._last_key
                and now - self._last_time < self.COALESCE_SECONDS
                and self._undo
            )
            if not same:
                self._push_undo(self._value)
            self._last_key, self._last_time = coalesce_key, now
        self._value = value
        self._notify()

    def reset(self, value: T) -> None:
        """Replace the value and clear history (e.g. after loading a file)."""
        self._undo.clear()
        self._redo.clear()
        self._gesture_start = None
        self._value = value
        self._notify()
        self._bump()

    def begin_gesture(self) -> None:
        if self._gesture_start is None:
            self._gesture_start = self._value

    def end_gesture(self) -> None:
        start, self._gesture_start = self._gesture_start, None
        if start is not None and start != self._value:
            self._push_undo(start)
        self._last_key = None

    # --- undo / redo -------------------------------------------------------
    @property
    def can_undo(self) -> bool:
        return bool(self._undo)

    @property
    def can_redo(self) -> bool:
        return bool(self._redo)

    def undo(self) -> None:
        if not self._undo:
            return
        self._redo.append(self._value)
        self._value = self._undo.pop()
        self._last_key = None
        self._notify()
        self._bump()

    def redo(self) -> None:
        if not self._redo:
            return
        self._undo.append(self._value)
        self._value = self._redo.pop()
        self._last_key = None
        self._notify()
        self._bump()

    def _push_undo(self, value: T) -> None:
        self._undo.append(value)
        if len(self._undo) > self._limit:
            del self._undo[0]
        self._redo.clear()
        self._bump()

    def _bump(self) -> None:
        self.history_changed.set(self.history_changed.value + 1)
