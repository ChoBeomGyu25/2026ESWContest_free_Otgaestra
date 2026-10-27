from __future__ import annotations

import contextlib
import importlib.util
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import numpy as np


@dataclass(frozen=True)
class BottomProcessResult:
    status: str
    reason: str = ""
    recovery_required: bool = False


class SharedCameraCaptureAdapter:
    def __init__(self, provider: Any, timeout_s: float = 2.0):
        if provider is None:
            raise ValueError("shared camera provider is required")
        self.provider = provider
        self.timeout_s = float(timeout_s)
        self._lock = threading.RLock()
        self._released = False
        self._last_sequence = int(provider.current_sequence())
        self._last_frame = None
        self.release_calls = 0

    def read(self):
        with self._lock:
            if self._released:
                return False, None
            try:
                frame, sequence, _ = self.provider.wait_for_new_frame(
                    int(self._last_sequence), timeout=max(0.01, float(self.timeout_s))
                )
            except TimeoutError:
                return False, None
            except Exception:
                return False, None
            if frame is None:
                return False, None
            arr = np.asarray(frame, dtype=np.uint8)
            if arr.ndim != 3 or arr.size == 0:
                return False, None
            self._last_sequence = int(sequence)
            self._last_frame = arr.copy()
            return True, self._last_frame.copy()

    def isOpened(self):
        return bool((not self._released) and self.provider.is_alive())

    def release(self):
        with self._lock:
            self.release_calls += 1
            self._released = True

    def get(self, prop):
        frame = self._last_frame
        if frame is None:
            return 0.0
        try:
            import cv2
            if prop == cv2.CAP_PROP_FRAME_WIDTH:
                return float(frame.shape[1])
            if prop == cv2.CAP_PROP_FRAME_HEIGHT:
                return float(frame.shape[0])
        except Exception:
            pass
        return 0.0

    def set(self, prop, value):
        return False


class _CvIntegrationGuard:
    def __init__(self):
        self._saved = {}

    def __enter__(self):
        import cv2

        def blocked(name):
            def fn(*args, **kwargs):
                raise RuntimeError(f"BOTTOM integration forbids cv2.{name}()")
            return fn

        for name in ("VideoCapture", "namedWindow", "imshow", "waitKey"):
            if hasattr(cv2, name):
                self._saved[name] = getattr(cv2, name)
                setattr(cv2, name, blocked(name))
        return self

    def __exit__(self, exc_type, exc, tb):
        import cv2
        for name, value in self._saved.items():
            setattr(cv2, name, value)
        self._saved.clear()
        return False


@contextlib.contextmanager
def _temporary_lower_dependency_modules(bottom_dir: Path):
    names = (
        "camera_undistort",
        "step_d25_v2",
        "step_e49_bottom_perception",
        "step_e62_bottom_perception",
    )
    previous: Dict[str, Any] = {}
    inserted = []
    bottom_dir = Path(bottom_dir).resolve()
    if str(bottom_dir) not in sys.path:
        sys.path.insert(0, str(bottom_dir))
        inserted.append(str(bottom_dir))
    try:
        for name in names:
            path = bottom_dir / f"{name}.py"
            if not path.is_file():
                raise FileNotFoundError(f"required lower dependency missing: {path}")
            if name in sys.modules:
                previous[name] = sys.modules[name]
            spec = importlib.util.spec_from_file_location(name, str(path))
            if spec is None or spec.loader is None:
                raise RuntimeError(f"cannot load lower dependency: {path}")
            module = importlib.util.module_from_spec(spec)
            sys.modules[name] = module
            spec.loader.exec_module(module)
        yield
    finally:
        for name in names:
            if name in previous:
                sys.modules[name] = previous[name]
            else:
                sys.modules.pop(name, None)
        for entry in inserted:
            try:
                sys.path.remove(entry)
            except ValueError:
                pass


class BottomProcess:
    def __init__(
        self,
        argv: Optional[Iterable[str]] = None,
        debug: bool = False,
        external_frame_provider: Any = None,
        runtime_path: Optional[str] = None,
    ):
        if external_frame_provider is None:
            raise ValueError("BottomProcess requires external_frame_provider")
        self.argv = list(argv or [])
        self.debug = bool(debug)
        self.external_frame_provider = external_frame_provider
        self.base_dir = Path(__file__).resolve().parent
        self.bottom_dir = self.base_dir / "bottom"
        self.runtime_path = Path(runtime_path).resolve() if runtime_path else self.bottom_dir / "bottom_vla_runtime.py"
        self.module = None
        self.app = None
        self.capture = None
        self._started = False
        self._closed = False
        self._empty_board_ready = False
        self._run_lock = threading.RLock()

    def _required_files(self):
        return (
            "bottom_vla_runtime.py",
            "bottom_vla-23.py",
            "main-33.py",
            "50-1.py",
            "54-3.py",
            "55-5.py",
            "58-3.py",
            "60-15.py",
            "align-11.py",
            "step_e49_bottom_perception.py",
            "step_e62_bottom_perception.py",
            "step_d25_v2.py",
            "camera_undistort.py",
        )

    def validate_installation(self):
        missing = []
        for name in self._required_files():
            path = self.runtime_path if name == "bottom_vla_runtime.py" else self.bottom_dir / name
            if not path.is_file():
                missing.append(str(path))
        if missing:
            raise FileNotFoundError("missing lower integration files: " + ", ".join(missing))
        return True

    def _wait_for_app_idle(self, app: Any, timeout: float, context: str) -> bool:
        waiter = getattr(app, "wait_for_integrated_idle", None)
        if not callable(waiter):
            print(f"[BOTTOM-PROCESS-IDLE-WARN] context={context} runtime has no wait_for_integrated_idle()")
            return True
        try:
            ok = bool(waiter(timeout=max(0.1, float(timeout))))
        except Exception as exc:
            print(f"[BOTTOM-PROCESS-IDLE-WARN] context={context} error={type(exc).__name__}: {exc}")
            return False
        print(f"[BOTTOM-PROCESS-IDLE] context={context} quiescent={ok}")
        return ok

    def _load_runtime(self):
        spec = importlib.util.spec_from_file_location("bottom_vla_v114_integrated_runtime", str(self.runtime_path))
        if spec is None or spec.loader is None:
            raise RuntimeError(f"cannot import bottom runtime: {self.runtime_path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module

    def start(self, timeout: float = 12.0):
        with self._run_lock:
            if self._started:
                return True
            self.validate_installation()
            self.external_frame_provider.start(timeout=float(timeout))
            if not bool(self.external_frame_provider.is_alive()):
                raise RuntimeError("V22 shared camera provider is not alive")
            self.capture = SharedCameraCaptureAdapter(self.external_frame_provider, timeout_s=2.0)
            with _CvIntegrationGuard(), _temporary_lower_dependency_modules(self.bottom_dir):
                self.module = self._load_runtime()
                self.app = self.module.create_integrated_app(
                    self.external_frame_provider,
                    self.capture,
                    argv=self.argv,
                )
                prepare = getattr(self.app, "prepare_integrated_startup_standby", None)
                if not callable(prepare):
                    raise RuntimeError("bottom runtime has no prepare_integrated_startup_standby()")
                if not bool(prepare()):
                    raise RuntimeError("bottom startup standby recovery failed")
            self._started = True
            self._closed = False
            try:
                self.external_frame_provider.set_status("BOTTOM V114 READY - SHARED CAMERA")
            except Exception:
                pass
            print("[BOTTOM-PROCESS-START] V114 integrated runtime ready; V22 remains sole camera owner")
            return True

    def set_empty_board_state(self, state: Any, timeout: float = 15.0):
        del timeout
        with self._run_lock:
            if not self._started or self.app is None:
                raise RuntimeError("BottomProcess.start() must be called first")
            with _CvIntegrationGuard():
                applied = bool(self.app.set_integration_empty_board_state(state))
            self._empty_board_ready = applied
            return applied

    def run_cycle(self, timeout: float = 240.0) -> BottomProcessResult:
        with self._run_lock:
            if not self._started or self.app is None:
                raise RuntimeError("BottomProcess.start() must be called first")
            if not self._empty_board_ready:
                raise RuntimeError("set_empty_board_state() must succeed before run_cycle()")
            try:
                with _CvIntegrationGuard():
                    payload = self.app.run_integrated_cycle(timeout=float(timeout))
            except Exception as exc:
                # Never let an integration/UI exception race serial teardown
                # against a still-running AUTO-JUDGE/planner worker.
                try:
                    self.app.request_integrated_stop()
                except Exception:
                    pass
                idle = self._wait_for_app_idle(
                    self.app, timeout=min(20.0, max(3.0, float(timeout) * 0.10)), context="cycle-exception"
                )
                reason = f"run_integrated_cycle exception {type(exc).__name__}: {exc}; workers_idle={idle}"
                print(f"[BOTTOM-PROCESS-CYCLE-ERROR] {reason}")
                return BottomProcessResult("ERROR", reason, True)
            if isinstance(payload, BottomProcessResult):
                return payload
            if not isinstance(payload, dict):
                return BottomProcessResult("ERROR", f"unexpected bottom result: {payload!r}", True)
            return BottomProcessResult(
                status=str(payload.get("status", "ERROR")),
                reason=str(payload.get("reason", "")),
                recovery_required=bool(payload.get("recovery_required", True)),
            )

    def get_current_action(self) -> Optional[str]:
        """Return only the lower-garment macro actually accepted by the executor.

        V20 deliberately ignores selected_action / planner state.  The runtime
        latches _integration_executing_action only after _start_execution()
        confirms V110-EXEC-ACCEPT, and the value is visible only while the
        physical motion remains busy.
        """
        app = self.app
        if app is None:
            return None
        state_lock = getattr(app, "state_lock", None)
        try:
            if state_lock is None:
                action = getattr(app, "_integration_executing_action", None)
                motion_busy = bool(getattr(app, "motion_busy", False))
            else:
                with state_lock:
                    action = getattr(app, "_integration_executing_action", None)
                    motion_busy = bool(getattr(app, "motion_busy", False))
            value = str(action or "").strip().upper()
            allowed = {"POSITION_ADJUST", "PRESS_SWEEP", "WAIST_PULL_LAYDOWN", "ALIGN"}
            if motion_busy and value in allowed:
                return value
            return None
        except Exception:
            return None

    def prepare_force_fold_standby(self, timeout: float = 35.0) -> bool:
        """After cooperative STOP, verify workers idle then move/verify camera-clear standby."""
        app = self.app
        if app is None:
            return False
        if not self._wait_for_app_idle(app, timeout=max(3.0, float(timeout)), context="force-fold-stop"):
            return False
        prepare = getattr(app, "prepare_integrated_startup_standby", None)
        if not callable(prepare):
            print("[BOTTOM-FORCE-FOLD-STANDBY] runtime has no standby helper")
            return False
        try:
            with _CvIntegrationGuard():
                ok = bool(prepare())
        except Exception as exc:
            print(f"[BOTTOM-FORCE-FOLD-STANDBY] error={type(exc).__name__}: {exc}")
            return False
        print(f"[BOTTOM-FORCE-FOLD-STANDBY] verified={ok}")
        return ok

    def request_stop(self, reason: str = ""):
        # Cooperative stop must never wait for run_cycle()'s long-held _run_lock.
        # Keep a local app reference so close() may clear self.app concurrently
        # without invalidating this immediate Event.set() request.
        app = self.app
        if app is None:
            return False
        try:
            app.request_integrated_stop()
        finally:
            if reason:
                print(f"[BOTTOM-PROCESS-STOP] reason={reason}")
        return True

    def close(self, timeout: float = 65.0):
        # Signal first, before waiting for _run_lock. This lets an active
        # run_integrated_cycle() observe the stop event and release the lock.
        try:
            self.request_stop()
        except Exception:
            pass
        with self._run_lock:
            if self._closed:
                return True
            app = self.app
            if app is not None:
                try:
                    app.request_integrated_stop()
                except Exception:
                    pass
                idle_budget = min(60.0, max(10.0, float(timeout) - 5.0))
                if not self._wait_for_app_idle(app, timeout=idle_budget, context="close-before-serial"):
                    raise TimeoutError(
                        f"BottomProcess.close refused serial teardown because action/planner workers "
                        f"were still active after {idle_budget:.1f}s"
                    )
                failure = []
                def closer():
                    try:
                        app.close()
                    except Exception as exc:
                        failure.append(exc)
                thread = threading.Thread(target=closer, name="bottom-process-close", daemon=True)
                thread.start()
                thread.join(timeout=max(0.1, float(timeout)))
                if thread.is_alive():
                    raise TimeoutError(f"BottomProcess.close timed out after {float(timeout):.1f}s")
                if failure:
                    raise failure[0]
            try:
                self.external_frame_provider.clear_override()
            except Exception:
                pass
            self._closed = True
            self._started = False
            self.app = None
            self.module = None
            print("[BOTTOM-PROCESS-CLOSE] lower runtime closed; shared camera provider was not stopped or released")
            return True
