#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Stable public API for integrated VLA-based upper-garment manipulation.

This module intentionally contains no keyboard/UI loop and no robot motion implementation.
The final upper-garment V51 behavior lives in ``upper_runtime_v51.py``. Integration code should
import this module rather than the large runtime directly. In integrated mode an external
frame provider is supplied so V51 never opens the physical camera itself; direct execution
of ``upper_runtime_v51.py`` retains the standalone V51 UI/camera/debug path.
"""
from __future__ import annotations

import sys
import threading
from pathlib import Path
from typing import Any, List, Optional

_MODULE_DIR = Path(__file__).resolve().parent
if str(_MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(_MODULE_DIR))

from shared_empty_board_state import EmptyBoardState
from upper_runtime_v51 import UpperCycleResult, UpperGarmentController as _RuntimeController

__all__ = [
    "EmptyBoardState",
    "UpperCycleResult",
    "UpperProcess",
    "UpperGarmentController",
    "create_process",
    "get_default_process",
    "start",
    "capture_empty_board_state",
    "capture_empty_board",
    "set_empty_board_state",
    "get_empty_board_state",
    "run_cycle",
    "request_stop",
    "close",
]


class UpperProcess:
    """Persistent importable upper-garment process.

    ``start()`` initializes camera/models/serial once.  One shared EmptyBoardState may
    then be captured or supplied by the integration runtime.  ``run_cycle()`` performs
    one complete upper-garment manipulation cycle and returns an UpperCycleResult.
    The same instance may be reused for later garments until ``close()``.
    """

    def __init__(self, argv: Optional[List[str]] = None, *, debug: bool = False,
                 external_frame_provider: Any = None):
        self._controller = _RuntimeController(
            argv=argv,
            debug=debug,
            external_frame_provider=external_frame_provider,
        )
        self._started = False
        self._closed = False
        self._lock = threading.RLock()

    def start(self, timeout: float = 120.0) -> "UpperProcess":
        with self._lock:
            if self._closed:
                raise RuntimeError("UpperProcess is closed")
            if not self._started:
                self._controller.start(timeout=timeout)
                self._started = True
            return self

    def capture_empty_board(self, timeout: float = 10.0) -> EmptyBoardState:
        self.start()
        return self._controller.capture_empty_board(timeout=timeout)

    def capture_empty_board_state(self, timeout: float = 10.0) -> EmptyBoardState:
        return self.capture_empty_board(timeout=timeout)

    def set_empty_board_state(self, state: EmptyBoardState, timeout: float = 10.0) -> EmptyBoardState:
        self.start()
        return self._controller.set_empty_board_state(state, timeout=timeout)

    def get_empty_board_state(self) -> Optional[EmptyBoardState]:
        return self._controller.get_empty_board_state()

    def run_cycle(
        self,
        timeout: Optional[float] = None,
        *,
        empty_board_state: Optional[EmptyBoardState] = None,
    ) -> UpperCycleResult:
        self.start()
        return self._controller.run_cycle(
            timeout=timeout,
            empty_board_state=empty_board_state,
        )

    def request_stop(self, reason: str = "display UI stop requested") -> bool:
        """Forward an integrated stop request to the V51 runtime controller."""
        return bool(self._controller.request_stop(reason))

    def get_current_action(self) -> Optional[str]:
        """Return display-only final executable macro selected by V51, if any."""
        getter = getattr(self._controller, "get_current_action", None)
        if callable(getter):
            return getter()
        control = getattr(self._controller, "_control", None)
        getter = getattr(control, "current_action", None)
        return getter() if callable(getter) else None

    def close(self, timeout: float = 65.0) -> None:
        with self._lock:
            if self._closed:
                return
            try:
                self._controller.close(timeout=timeout)
            finally:
                self._closed = True
                self._started = False

    def __enter__(self) -> "UpperProcess":
        return self.start()

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


UpperGarmentController = UpperProcess


def create_process(argv: Optional[List[str]] = None, *, debug: bool = False,
                   external_frame_provider: Any = None) -> UpperProcess:
    return UpperProcess(
        argv=argv,
        debug=debug,
        external_frame_provider=external_frame_provider,
    )


def self_test() -> int:
    """Device-free API/contract check for integration wiring."""
    runtime_path = _MODULE_DIR / "upper_runtime_v51.py"
    core_path = _MODULE_DIR / "micro_unfold_e156_core_v51.py"
    shared_path = _MODULE_DIR / "shared_empty_board_state.py"
    checks = {
        "runtime_exists": runtime_path.is_file(),
        "micro_core_exists": core_path.is_file(),
        "shared_state_exists": shared_path.is_file(),
        "runtime_external_provider_api": "external_frame_provider" in runtime_path.read_text(encoding="utf-8", errors="replace") if runtime_path.is_file() else False,
        "runtime_shared_capture_adapter": "class ExternalFrameProviderCapture" in runtime_path.read_text(encoding="utf-8", errors="replace") if runtime_path.is_file() else False,
        "runtime_controller_api": "class UpperGarmentController" in runtime_path.read_text(encoding="utf-8", errors="replace") if runtime_path.is_file() else False,
        "runtime_matching_v51_core_loader": "micro_unfold_e156_core_v{match.group(1)}.py" in runtime_path.read_text(encoding="utf-8", errors="replace") if runtime_path.is_file() else False,
        "runtime_standalone_entrypoint": 'if __name__ == "__main__"' in runtime_path.read_text(encoding="utf-8", errors="replace") if runtime_path.is_file() else False,
        "public_run_cycle": callable(getattr(UpperProcess, "run_cycle", None)),
        "public_request_stop": callable(getattr(UpperProcess, "request_stop", None)),
        "public_set_empty": callable(getattr(UpperProcess, "set_empty_board_state", None)),
    }
    for name, ok in checks.items():
        print(f"[UPPER-VLA-SELF-TEST] {name}={bool(ok)}")
    passed = all(checks.values())
    print(f"[UPPER-VLA-SELF-TEST] ALL_PASS={passed}")
    return 0 if passed else 2


_default_lock = threading.RLock()
_default_process: Optional[UpperProcess] = None


def get_default_process(*, argv: Optional[List[str]] = None, debug: bool = False) -> UpperProcess:
    global _default_process
    with _default_lock:
        if _default_process is None:
            _default_process = UpperProcess(argv=argv, debug=debug)
        return _default_process


def start(timeout: float = 120.0, *, argv: Optional[List[str]] = None, debug: bool = False) -> UpperProcess:
    process = get_default_process(argv=argv, debug=debug)
    return process.start(timeout=timeout)


def capture_empty_board_state(timeout: float = 10.0) -> EmptyBoardState:
    return get_default_process().capture_empty_board_state(timeout=timeout)


def capture_empty_board(timeout: float = 10.0) -> EmptyBoardState:
    return capture_empty_board_state(timeout=timeout)


def set_empty_board_state(state: EmptyBoardState, timeout: float = 10.0) -> EmptyBoardState:
    return get_default_process().set_empty_board_state(state, timeout=timeout)


def get_empty_board_state() -> Optional[EmptyBoardState]:
    return get_default_process().get_empty_board_state()


def run_cycle(
    timeout: Optional[float] = None,
    *,
    empty_board_state: Optional[EmptyBoardState] = None,
) -> UpperCycleResult:
    return get_default_process().run_cycle(
        timeout=timeout,
        empty_board_state=empty_board_state,
    )


def request_stop(reason: str = "display UI stop requested") -> bool:
    return bool(get_default_process().request_stop(reason))


def close(timeout: float = 65.0) -> None:
    global _default_process
    with _default_lock:
        process = _default_process
        _default_process = None
    if process is not None:
        process.close(timeout=timeout)
