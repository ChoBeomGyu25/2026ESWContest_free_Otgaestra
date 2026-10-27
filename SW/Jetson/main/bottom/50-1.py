#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Standalone 5-point basket calibration and temporary grasp-point viewer.

Five-point order (fixed):
  P0/V0: visible corner adjacent to the hidden corner on edge A
  P1/V1: visible corner diagonally opposite the hidden corner
  P2/V2: other visible corner adjacent to the hidden corner on edge B
  P3/E0: visible point on straight edge H-V0
  P4/E2: visible point on straight edge H-V2

The hidden corner H is reconstructed as intersection(line(V0,E0), line(V2,E2)).
Basket corner order is H -> V0 -> V1 -> V2. The temporary grasp point is
intersection(diagonal(H,V1), diagonal(V0,V2)).

Keys:
  B       enter/leave basket mode (terminal or OpenCV window)
  C       start/restart calibration (ARM2 torque off)
  click   select current image point
  Enter   save ARM2 T:105 feedback for pending click
  U       cancel pending click, otherwise undo last point
  S       recompute and save JSON (requires 5 points)
  L       reload JSON
  F       ARM2 feedback
  T / N   ARM2 torque off / on
  G       open gripper
  H       prepare ARM2 hover plan above temporary grasp point
  D       prepare torque-monitored descent after hover
  Enter   execute a prepared hover or descent action
  Q/ESC   quit

This file calibrates, visualizes, performs an ARM2-only hover, descends with
shoulder-primary robust contact detection, and—only after confirmed cloth
contact—automatically opens wider, closes the gripper, and lifts vertically.
After the adaptive vertical lift either reaches its preferred height or detects
a real vertical saturation, the last measured Z is accepted as the transit Z.
No second absolute-Z clearance gate is applied after saturation. From that exact
pose it sends one direct T:104 goal to the folding-board center, judges arrival
by XY only, opens the gripper there, and returns ARM2 to the A150 taught standby
pose. It never sends a command to ARM1.

Hover safety sequence:
  H       prepare a hover plan from the loaded 5-point JSON
  Enter   execute the prepared hover plan

Terminal logging:
  Every Python stdout/stderr message is mirrored live to a timestamped UTF-8
  text file. Native execution writes to /home/deca/project_train/aruco_test/dual.
  Docker execution automatically uses the persistent /workspace mirror.

Execution path:
  1) Read the current ARM2 pose with T:105.
  2) If ARM2 is already at the A150 taught standby pose, first raise vertically
     to the configured board safe-hover Z and move to the calibrated ARM2
     board-inner (RED_EXTRA) waypoint. The basket command is blocked until
     that board-safe waypoint is actually reached.
  3) Move horizontally from the board-safe waypoint to the temporary basket
     grasp XY at the same high transit Z. For non-standby starts, retain the
     original direct high-Z basket approach.
  4) Descend vertically only after reaching the target XY.

The five-point Z mean is used as the hover reference. For collision safety,
the final hover Z is also constrained to remain above the highest calibrated
rim point by a configurable minimum clearance.
"""

import argparse
import atexit
import csv
import json
import math
import os
import platform
import select
import shutil
import statistics
import sys
import threading
import time
import traceback
from pathlib import Path
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

os.environ["QT_X11_NO_MITSHM"] = "1"
os.environ["GDK_DISABLE_SHM"] = "1"
os.environ.setdefault("NO_AT_BRIDGE", "1")

import cv2
import numpy as np

try:
    import serial
except Exception:
    serial = None

try:
    import termios
    import tty
except Exception:
    termios = None
    tty = None

BUILD_ID = "D50-BASKET-DIRECT-AIR-DROP-20260813"

POINT_SPECS = (
    ("P0_V0", "visible corner adjacent to hidden corner on edge A"),
    ("P1_V1", "visible corner diagonally opposite the hidden corner"),
    ("P2_V2", "other visible corner adjacent to hidden corner on edge B"),
    ("P3_E0", "point on straight edge H-V0; avoid rounded corner"),
    ("P4_E2", "point on straight edge H-V2; avoid rounded corner"),
)


SESSION_LOG_REQUESTED_DIR = Path("/home/deca/project_train/aruco_test/dual")
SESSION_LOG_DOCKER_MIRROR_DIR = Path("/workspace/project_train/aruco_test/dual")


class _TeeTextStream:
    """Write each Python stdout/stderr message to the terminal and a UTF-8 log file."""

    def __init__(self, terminal_stream, log_stream, lock: threading.RLock):
        self._terminal_stream = terminal_stream
        self._log_stream = log_stream
        self._lock = lock

    def write(self, text):
        if not isinstance(text, str):
            text = str(text)
        with self._lock:
            terminal_result = self._terminal_stream.write(text)
            self._log_stream.write(text)
            # Line-buffered file plus explicit flush prevents long-run data loss.
            self._log_stream.flush()
            return terminal_result

    def flush(self):
        with self._lock:
            self._terminal_stream.flush()
            self._log_stream.flush()

    def isatty(self):
        return bool(getattr(self._terminal_stream, "isatty", lambda: False)())

    def fileno(self):
        return self._terminal_stream.fileno()

    @property
    def encoding(self):
        return getattr(self._terminal_stream, "encoding", "utf-8")

    @property
    def errors(self):
        return getattr(self._terminal_stream, "errors", "replace")

    def __getattr__(self, name):
        return getattr(self._terminal_stream, name)


class SessionTerminalLog:
    """Mirror the complete Python terminal session to a timestamped text file."""

    def __init__(self):
        self._original_stdout = sys.stdout
        self._original_stderr = sys.stderr
        self._log_stream = None
        self._lock = threading.RLock()
        self.path: Optional[Path] = None
        self.requested_dir = SESSION_LOG_REQUESTED_DIR
        self.used_docker_mirror = False
        self._closed = False

    @staticmethod
    def _is_writable_directory(path: Path) -> bool:
        return path.is_dir() and os.access(str(path), os.W_OK)

    def _resolve_directory(self) -> Path:
        # On the host, use the exact directory requested by the operator.
        if self._is_writable_directory(self.requested_dir):
            return self.requested_dir

        # The project is normally mounted into Docker as /workspace/project_train.
        # Writing there persists into the requested host project directory.
        if self._is_writable_directory(SESSION_LOG_DOCKER_MIRROR_DIR):
            self.used_docker_mirror = True
            return SESSION_LOG_DOCKER_MIRROR_DIR

        # Native execution may begin before the destination directory exists.
        self.requested_dir.mkdir(parents=True, exist_ok=True)
        if not self._is_writable_directory(self.requested_dir):
            raise PermissionError(f"session log directory is not writable: {self.requested_dir}")
        return self.requested_dir

    def start(self) -> Path:
        log_dir = self._resolve_directory()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = f"basket_v24_terminal_{timestamp}_pid{os.getpid()}.txt"
        self.path = log_dir / filename
        self._log_stream = self.path.open("x", encoding="utf-8", buffering=1)
        sys.stdout = _TeeTextStream(self._original_stdout, self._log_stream, self._lock)
        sys.stderr = _TeeTextStream(self._original_stderr, self._log_stream, self._lock)
        print("\n========== TERMINAL SESSION LOG STARTED ==========")
        print(f"[SESSION-LOG] file={self.path}")
        print(f"[SESSION-LOG] requested_host_dir={self.requested_dir}")
        if self.used_docker_mirror:
            print(
                "[SESSION-LOG] Docker mirror active: /workspace/project_train/... "
                "persists to the requested host project directory"
            )
        print(f"[SESSION-LOG] started_at={datetime.now().isoformat(timespec='seconds')}")
        print("==================================================\n")
        atexit.register(self.close)
        return self.path

    def close(self, reason: str = "normal-exit") -> None:
        if self._closed:
            return
        self._closed = True
        try:
            if self._log_stream is not None:
                print("\n========== TERMINAL SESSION LOG FINISHED ==========")
                print(f"[SESSION-LOG] reason={reason}")
                print(f"[SESSION-LOG] finished_at={datetime.now().isoformat(timespec='seconds')}")
                print(f"[SESSION-LOG] saved={self.path}")
                print("===================================================")
                sys.stdout.flush()
                sys.stderr.flush()
        finally:
            sys.stdout = self._original_stdout
            sys.stderr = self._original_stderr
            if self._log_stream is not None:
                try:
                    self._log_stream.flush()
                finally:
                    self._log_stream.close()


def _run_main_with_session_log() -> int:
    """Run main while preserving normal, Ctrl+C, and exception termination logs."""
    session_log = SessionTerminalLog()
    try:
        session_log.start()
    except Exception as exc:
        # Logging failure must not prevent robot operation, but it must be obvious.
        print(f"[SESSION-LOG-ERROR] automatic terminal log disabled: {exc}", file=sys.stderr)
        try:
            main()
            return 0
        except SystemExit as exc:
            code = exc.code if isinstance(exc.code, int) else 1
            return int(code)
        except KeyboardInterrupt:
            print("\n[PROGRAM-INTERRUPTED] KeyboardInterrupt", file=sys.stderr)
            return 130

    exit_code = 0
    close_reason = "normal-exit"
    try:
        main()
    except SystemExit as exc:
        exit_code = int(exc.code) if isinstance(exc.code, int) else 1
        close_reason = "normal-system-exit" if exit_code == 0 else "argument-or-system-exit"
    except KeyboardInterrupt:
        close_reason = "keyboard-interrupt"
        exit_code = 130
        print("\n[PROGRAM-INTERRUPTED] KeyboardInterrupt")
    except BaseException:
        close_reason = "unhandled-exception"
        exit_code = 1
        traceback.print_exc()
    finally:
        session_log.close(close_reason)
    return exit_code


def open_camera(index: int, width: int, height: int, backend: str):
    """Open the webcam with the same capture path used by step_a150(5).py.

    Deliberately does not force MJPEG, YUYV, FPS, or another FOURCC.
    The camera driver negotiates the format exactly as in A150.
    """
    b = (backend or "auto").lower()
    if b == "v4l2" and hasattr(cv2, "CAP_V4L2"):
        api = cv2.CAP_V4L2
    elif b == "dshow":
        api = cv2.CAP_DSHOW
    elif b == "any":
        api = cv2.CAP_ANY
    elif platform.system().lower().startswith("win"):
        api = cv2.CAP_DSHOW
    elif hasattr(cv2, "CAP_V4L2"):
        api = cv2.CAP_V4L2
    else:
        api = cv2.CAP_ANY

    cap = cv2.VideoCapture(index, api)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(width))
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(height))
    try:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    except Exception:
        pass
    if not cap.isOpened() and api != cv2.CAP_ANY:
        cap.release()
        cap = cv2.VideoCapture(index, cv2.CAP_ANY)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(width))
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(height))
    return cap


class RoArmSerial:
    """Single persistent ARM2 serial session with startup synchronization.

    The serial port is opened only once for the full hover + descent workflow.
    This prevents the second-script port reopen from resetting the RoArm ESP32
    after the hover pose has already been reached.
    """

    def __init__(self, port: str, baud: int = 115200):
        if serial is None:
            raise RuntimeError("pyserial is not installed: pip3 install pyserial")
        self.ser = serial.Serial()
        self.ser.port = port
        self.ser.baudrate = int(baud)
        self.ser.timeout = 0.15
        self.ser.write_timeout = 1.0
        self.ser.rtscts = False
        self.ser.dsrdtr = False
        try:
            self.ser.dtr = False
            self.ser.rts = False
        except Exception:
            pass
        try:
            self.ser.exclusive = True
        except Exception:
            pass
        self.ser.open()
        self.boot_reset_detected = False

    def synchronize_startup(self, timeout: float = 15.0, quiet_sec: float = 1.2):
        started = time.monotonic()
        last_rx = started
        saw_text = False
        ready_seen = False
        reset_tokens = (
            "POWERON_RESET", "SPI_FAST_FLASH_BOOT", "entry 0x",
            "Initialize LittleFS", "Power up the servos", "Moving BASE_JOINT",
            "Moving SHOULDER_JOINT",
        )
        ready_tokens = ("RoArm-M2 started.", "[missionPlay finished.]")
        print(
            f"[SERIAL-SYNC] waiting for ARM2 startup to finish "
            f"(timeout={float(timeout):.1f}s, quiet={float(quiet_sec):.1f}s)"
        )
        while time.monotonic() - started < max(1.0, float(timeout)):
            raw = self.ser.readline()
            now = time.monotonic()
            if raw:
                text = raw.decode("utf-8", errors="replace").strip()
                if text:
                    saw_text = True
                    last_rx = now
                    print(f"[ARM2] STARTUP: {text}")
                    if any(token in text for token in reset_tokens):
                        self.boot_reset_detected = True
                    if any(token in text for token in ready_tokens):
                        ready_seen = True
                continue
            quiet_enough = now - last_rx >= max(0.3, float(quiet_sec))
            if quiet_enough and (not self.boot_reset_detected or ready_seen):
                break
        if self.boot_reset_detected:
            print(
                "[SERIAL-SYNC] startup/reset chatter drained. "
                "Hover and descent will now use this same serial session."
            )
        elif saw_text:
            print("[SERIAL-SYNC] startup text drained; link is quiet")
        else:
            print("[SERIAL-SYNC] ARM2 link was already quiet")
        self.flush()

    def flush(self):
        try:
            self.ser.reset_input_buffer()
            self.ser.reset_output_buffer()
        except Exception:
            pass

    def send(self, cmd: Dict[str, Any], delay: float = 0.12):
        line = json.dumps(cmd, separators=(",", ":"))
        print(f"[ARM2] SEND: {line}")
        self.ser.write(line.encode("utf-8") + b"\n")
        self.ser.flush()
        time.sleep(max(0.0, float(delay)))

    def feedback(self, timeout: float = 2.5, quiet: bool = False):
        try:
            self.ser.reset_input_buffer()
        except Exception:
            pass
        self.send({"T": 105}, delay=0.04)
        deadline = time.time() + max(0.2, float(timeout))
        while time.time() < deadline:
            raw = self.ser.readline()
            if not raw:
                continue
            text = raw.decode("utf-8", errors="replace").strip()
            if not text:
                continue
            if not quiet:
                print(f"[ARM2] RECV: {text}")
            try:
                obj = json.loads(text)
            except Exception:
                continue
            if int(obj.get("T", -1)) == 1051:
                return obj
        return None

    def feedback_retry(self, timeout: float = 2.5, attempts: int = 5, retry_delay: float = 0.5, quiet: bool = False):
        for attempt in range(1, max(1, int(attempts)) + 1):
            result = self.feedback(timeout=timeout, quiet=quiet)
            if result is not None:
                if attempt > 1:
                    print(f"[SERIAL-SYNC] T:105 recovered on attempt {attempt}")
                return result
            print(f"[SERIAL-SYNC] no T:1051 reply ({attempt}/{attempts})")
            time.sleep(max(0.0, float(retry_delay)))
        return None

    def torque_off(self):
        self.send({"T": 210, "cmd": 0}, delay=0.25)

    def torque_on(self):
        self.send({"T": 210, "cmd": 1}, delay=0.25)

    def gripper_open(self, angle: float, spd: float, acc: float):
        self.send({"T": 106, "cmd": float(angle), "spd": float(spd), "acc": float(acc)}, delay=0.35)

    def move_goal(self, move_command: int, x: float, y: float, z: float, t: float, spd: float):
        self.send(
            {"T": int(move_command), "x": float(x), "y": float(y),
             "z": float(z), "t": float(t), "spd": float(spd)},
            delay=0.05,
        )

    def move_direct(self, x: float, y: float, z: float, t: float):
        """D56-compatible non-blocking Cartesian stream target (T:1041)."""
        line = json.dumps(
            {"T": 1041, "x": float(x), "y": float(y),
             "z": float(z), "t": float(t)},
            separators=(",", ":"),
        )
        self.ser.write(line.encode("utf-8") + b"\n")
        self.ser.flush()

    def close(self):
        try:
            self.ser.close()
        except Exception:
            pass


class TerminalKeyReader:
    """A150-style nonblocking terminal key reader."""

    def __init__(self, enabled: bool = True, raw: bool = True):
        self.enabled = bool(enabled)
        self.raw_requested = bool(raw)
        self.raw_active = False
        self.orig_attrs = None
        self.is_tty = bool(getattr(sys.stdin, "isatty", lambda: False)())
        self.raw_supported = (
            self.enabled
            and self.raw_requested
            and self.is_tty
            and termios is not None
            and tty is not None
            and platform.system().lower() != "windows"
        )
        self.line_supported = self.enabled and self.is_tty
        self.registered = False

    def start(self):
        if not self.enabled:
            return
        if self.raw_supported and not self.raw_active:
            try:
                self.orig_attrs = termios.tcgetattr(sys.stdin.fileno())
                tty.setcbreak(sys.stdin.fileno())
                self.raw_active = True
                if not self.registered:
                    atexit.register(self.stop)
                    self.registered = True
                print("[TERMINAL] raw key mode ON: b/c/u/s/l/f/t/n/g/h/d/Enter/q")
                return
            except Exception as exc:
                print(f"[TERMINAL] raw mode failed: {exc}. Use line mode.")
                self.raw_supported = False
        if self.line_supported:
            print("[TERMINAL] line mode: type key then Enter.")
        else:
            print("[TERMINAL-WARN] stdin is not an interactive TTY.")

    def stop(self):
        if self.raw_active and self.orig_attrs is not None and termios is not None:
            try:
                termios.tcsetattr(sys.stdin.fileno(), termios.TCSADRAIN, self.orig_attrs)
            except Exception:
                pass
        self.raw_active = False

    @staticmethod
    def _line_to_key(text: str) -> int:
        cmd = (text or "").strip().lower()
        if cmd in ("", "enter"):
            return 13
        if cmd in ("esc", "escape"):
            return 27
        if cmd in ("quit", "exit"):
            return ord("q")
        aliases = {
            "basket": ord("b"),
            "calib": ord("c"),
            "undo": ord("u"),
            "save": ord("s"),
            "load": ord("l"),
            "feedback": ord("f"),
            "torqueoff": ord("t"),
            "torqueon": ord("n"),
            "open": ord("g"),
            "hover": ord("h"),
            "descent": ord("d"),
        }
        if len(cmd) == 1:
            return ord(cmd)
        return aliases.get(cmd, 255)

    def read_key(self) -> int:
        if not self.enabled or not self.is_tty:
            return 255
        try:
            ready, _, _ = select.select([sys.stdin], [], [], 0)
            if not ready:
                return 255
            if self.raw_active:
                ch = sys.stdin.read(1)
                if ch in ("\r", "\n"):
                    return 13
                if ch == "\x1b":
                    return 27
                if ch == "\x03":
                    raise KeyboardInterrupt
                return ord(ch.lower()) if ch else 255
            return self._line_to_key(sys.stdin.readline())
        except KeyboardInterrupt:
            raise
        except Exception:
            return 255


def xy(point: Sequence[float]) -> np.ndarray:
    return np.asarray([float(point[0]), float(point[1])], dtype=np.float64)


def line_from_points(a: Sequence[float], b: Sequence[float]) -> np.ndarray:
    p = np.asarray([float(a[0]), float(a[1]), 1.0], dtype=np.float64)
    q = np.asarray([float(b[0]), float(b[1]), 1.0], dtype=np.float64)
    line = np.cross(p, q)
    norm = math.hypot(float(line[0]), float(line[1]))
    if norm <= 1e-12:
        raise RuntimeError("Two points are identical or too close to define a line")
    return line / norm


def intersect(a: Sequence[float], b: Sequence[float], c: Sequence[float], d: Sequence[float]) -> np.ndarray:
    p = np.cross(line_from_points(a, b), line_from_points(c, d))
    if abs(float(p[2])) <= 1e-10:
        raise RuntimeError("Lines are parallel or numerically unstable")
    return np.asarray([p[0] / p[2], p[1] / p[2]], dtype=np.float64)


def line_angle_deg(a: Sequence[float], b: Sequence[float], c: Sequence[float], d: Sequence[float]) -> float:
    v1, v2 = xy(b) - xy(a), xy(d) - xy(c)
    n1, n2 = float(np.linalg.norm(v1)), float(np.linalg.norm(v2))
    if n1 <= 1e-12 or n2 <= 1e-12:
        return 0.0
    cosv = abs(float(np.dot(v1, v2) / (n1 * n2)))
    return float(math.degrees(math.acos(float(np.clip(cosv, -1.0, 1.0)))))


def polygon_area(points: Sequence[Sequence[float]]) -> float:
    pts = np.asarray(points, dtype=np.float64)
    x, y = pts[:, 0], pts[:, 1]
    return 0.5 * abs(float(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1))))


def point_inside(point: Sequence[float], polygon: Sequence[Sequence[float]]) -> bool:
    contour = np.asarray(polygon, dtype=np.float32).reshape(-1, 1, 2)
    return cv2.pointPolygonTest(contour, (float(point[0]), float(point[1])), False) >= 0


def fit_affine(points: List[Dict[str, Any]]) -> Tuple[np.ndarray, np.ndarray, int]:
    pixels = np.asarray([p["pixel_uv"] for p in points], dtype=np.float64)
    robots = np.asarray([[p["arm2_xyz"][0], p["arm2_xyz"][1]] for p in points], dtype=np.float64)
    design = np.c_[pixels, np.ones(len(points), dtype=np.float64)]
    solution, _, rank, _ = np.linalg.lstsq(design, robots, rcond=None)
    pred = design @ solution
    return solution.T, np.linalg.norm(pred - robots, axis=1), int(rank)


def apply_affine(matrix: np.ndarray, point: Sequence[float]) -> np.ndarray:
    return matrix @ np.asarray([float(point[0]), float(point[1]), 1.0], dtype=np.float64)


class BasketCalib:
    def __init__(self):
        self.points: List[Dict[str, Any]] = []
        self.matrix: Optional[np.ndarray] = None
        self.errors: Optional[np.ndarray] = None
        self.rank: Optional[int] = None
        self.geometry: Optional[Dict[str, Any]] = None

    def clear(self):
        self.points.clear()
        self.matrix = None
        self.errors = None
        self.rank = None
        self.geometry = None

    def next_spec(self):
        return POINT_SPECS[len(self.points)] if len(self.points) < 5 else None

    def add(self, pixel_uv: Tuple[int, int], arm2_xyz: Tuple[float, float, float]):
        spec = self.next_spec()
        if spec is None:
            raise RuntimeError("All five points are already saved")
        item = {
            "label": spec[0],
            "description": spec[1],
            "pixel_uv": [float(pixel_uv[0]), float(pixel_uv[1])],
            "arm2_xyz": [float(arm2_xyz[0]), float(arm2_xyz[1]), float(arm2_xyz[2])],
        }
        self.points.append(item)
        self.matrix = self.errors = self.rank = self.geometry = None
        return item

    def undo(self):
        if not self.points:
            print("[CALIB] nothing to undo")
            return
        print(f"[CALIB] undo: {self.points.pop()['label']}")
        self.matrix = self.errors = self.rank = self.geometry = None

    def compute(self, width: int, height: int, args):
        if len(self.points) != 5:
            raise RuntimeError(f"Exactly five points required; current={len(self.points)}")
        p0, p1, p2, p3, p4 = self.points
        v0p, v1p, v2p, e0p, e2p = [xy(p["pixel_uv"]) for p in self.points]
        angle_px = line_angle_deg(v0p, e0p, v2p, e2p)
        if angle_px < args.min_line_angle_deg:
            raise RuntimeError(f"Image edge-line angle too small: {angle_px:.2f}deg")
        hidden_px = intersect(v0p, e0p, v2p, e2p)
        margin_x, margin_y = width * args.max_hidden_margin_ratio, height * args.max_hidden_margin_ratio
        if not (-margin_x <= hidden_px[0] <= width + margin_x and -margin_y <= hidden_px[1] <= height + margin_y):
            raise RuntimeError("Reconstructed hidden corner is implausibly far outside image")
        corners_px = [hidden_px, v0p, v1p, v2p]
        area = polygon_area(corners_px)
        if area < args.min_polygon_area_px:
            raise RuntimeError(f"Basket polygon too small: {area:.1f}px^2")
        center_px = intersect(hidden_px, v1p, v0p, v2p)
        if not point_inside(center_px, corners_px):
            raise RuntimeError("Diagonal intersection is outside reconstructed basket polygon")

        matrix, errors, rank = fit_affine(self.points)
        if rank < 3:
            raise RuntimeError(f"Affine rank is {rank}; points are degenerate")

        v0r, v1r, v2r, e0r, e2r = [xy(p["arm2_xyz"]) for p in self.points]
        angle_robot = line_angle_deg(v0r, e0r, v2r, e2r)
        if angle_robot < args.min_line_angle_deg:
            raise RuntimeError(f"ARM2 edge-line angle too small: {angle_robot:.2f}deg")
        hidden_robot_direct = intersect(v0r, e0r, v2r, e2r)
        center_robot_direct = intersect(hidden_robot_direct, v1r, v0r, v2r)
        hidden_robot_affine = apply_affine(matrix, hidden_px)
        center_robot_affine = apply_affine(matrix, center_px)
        center_delta = float(np.linalg.norm(center_robot_direct - center_robot_affine))

        mean_error, max_error = float(np.mean(errors)), float(np.max(errors))
        if mean_error > args.max_mean_error_mm:
            raise RuntimeError(f"Mean affine error too large: {mean_error:.2f}mm")
        if max_error > args.max_point_error_mm:
            raise RuntimeError(f"Max affine error too large: {max_error:.2f}mm")
        if center_delta > args.max_center_crosscheck_mm:
            raise RuntimeError(f"Center cross-check error too large: {center_delta:.2f}mm")

        z = np.asarray([p["arm2_xyz"][2] for p in self.points], dtype=np.float64)
        self.matrix, self.errors, self.rank = matrix, errors, rank
        self.geometry = {
            "corner_order": ["H_reconstructed", "P0_V0", "P1_V1", "P2_V2"],
            "hidden_corner_pixel_uv": hidden_px.tolist(),
            "basket_corners_pixel_uv": [p.tolist() for p in corners_px],
            "diagonal_1_pixel_uv": [hidden_px.tolist(), v1p.tolist()],
            "diagonal_2_pixel_uv": [v0p.tolist(), v2p.tolist()],
            "temporary_grasp_pixel_uv": center_px.tolist(),
            "hidden_corner_arm2_xy_direct": hidden_robot_direct.tolist(),
            "hidden_corner_arm2_xy_affine": hidden_robot_affine.tolist(),
            "temporary_grasp_arm2_xy_direct": center_robot_direct.tolist(),
            "temporary_grasp_arm2_xy_affine": center_robot_affine.tolist(),
            "temporary_grasp_crosscheck_error_mm": center_delta,
            "image_edge_line_angle_deg": angle_px,
            "robot_edge_line_angle_deg": angle_robot,
            "basket_polygon_area_px2": area,
            "rim_z_mean": float(np.mean(z)),
            "rim_z_min": float(np.min(z)),
            "rim_z_max": float(np.max(z)),
        }
        return self.geometry

    def save(self, path: str, camera: str, width: int, height: int, arm2_port: str):
        if self.matrix is None or self.errors is None or self.geometry is None:
            raise RuntimeError("Calibration is not computed")
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        if output.exists():
            backup = output.with_name(f"{output.stem}.backup_{time.strftime('%Y%m%d_%H%M%S')}{output.suffix}")
            shutil.copy2(output, backup)
            print(f"[CALIB] previous JSON backup: {backup}")
        data = {
            "schema_version": 1,
            "created_or_updated_at_unix": time.time(),
            "type": "basket_arm2_5point_pixel_to_xy_affine",
            "camera": {"device": str(camera), "frame_width": width, "frame_height": height},
            "arm": {"name": "ARM2", "port": arm2_port, "feedback_command": 105},
            "five_point_protocol": [{"label": x, "description": y} for x, y in POINT_SPECS],
            "points": self.points,
            "affine": {
                "direction": "pixel_uv_to_arm2_xy",
                "matrix_2x3": self.matrix.tolist(),
                "rank": int(self.rank),
                "error_mm": self.errors.tolist(),
                "mean_error_mm": float(np.mean(self.errors)),
                "max_error_mm": float(np.max(self.errors)),
            },
            "geometry": self.geometry,
            "temporary_grasp_policy": {
                "current_policy": "intersection_of_reconstructed_basket_diagonals",
                "robot_xy_to_use_later": "geometry.temporary_grasp_arm2_xy_direct",
                "note": "Geometry-only temporary grasp point; no clothing perception yet",
            },
        }
        tmp = output.with_suffix(output.suffix + ".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, output)
        print(f"[CALIB] saved: {output}")

    def load(self, path: str):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if len(data.get("points", [])) != 5:
            raise RuntimeError("JSON must contain exactly five points")
        matrix = np.asarray(data["affine"]["matrix_2x3"], dtype=np.float64)
        if matrix.shape != (2, 3):
            raise RuntimeError(f"Invalid matrix shape: {matrix.shape}")
        self.points = data["points"]
        self.matrix = matrix
        self.errors = np.asarray(data["affine"].get("error_mm", []), dtype=np.float64)
        self.rank = int(data["affine"].get("rank", 3))
        self.geometry = data.get("geometry")
        print(f"[CALIB] loaded: {path}")


def outlined(img, text: str, org: Tuple[int, int], scale: float, color, thickness: int = 2):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thickness + 3, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def cross(img, point: Sequence[float], color, label: str):
    x, y = int(round(float(point[0]))), int(round(float(point[1])))
    s = 18
    cv2.line(img, (x - s, y), (x + s, y), (0, 0, 0), 7)
    cv2.line(img, (x, y - s), (x, y + s), (0, 0, 0), 7)
    cv2.line(img, (x - s, y), (x + s, y), color, 3)
    cv2.line(img, (x, y - s), (x, y + s), color, 3)
    outlined(img, label, (x + 14, y - 14), 0.55, color)


def draw(
    img,
    calib: BasketCalib,
    pending: Optional[Tuple[int, int]],
    basket_mode: bool,
    active: bool,
    hover_plan: Optional[Dict[str, float]],
):
    colors = [(0,180,255),(0,255,0),(255,180,0),(255,0,255),(255,255,0)]
    for i, p in enumerate(calib.points):
        u, v = map(lambda n: int(round(float(n))), p["pixel_uv"])
        cv2.circle(img, (u, v), 9, (0, 0, 0), -1)
        cv2.circle(img, (u, v), 6, colors[i], -1)
        outlined(img, p["label"], (u + 12, v - 12), 0.52, colors[i])
    if pending is not None:
        cv2.circle(img, pending, 11, (0, 0, 0), -1)
        cv2.circle(img, pending, 7, (0, 0, 255), -1)
        outlined(img, "PENDING: move ARM2 tip, then Enter", (pending[0] + 14, pending[1] + 24), 0.52, (0, 0, 255))
    if calib.geometry:
        g = calib.geometry
        corners = np.round(np.asarray(g["basket_corners_pixel_uv"])).astype(np.int32).reshape(-1, 1, 2)
        overlay = img.copy()
        cv2.fillPoly(overlay, [corners], (0, 150, 255))
        cv2.addWeighted(overlay, 0.12, img, 0.88, 0, img)
        cv2.polylines(img, [corners], True, (0, 200, 255), 3, cv2.LINE_AA)
        d1, d2 = g["diagonal_1_pixel_uv"], g["diagonal_2_pixel_uv"]
        cv2.line(img, tuple(np.round(d1[0]).astype(int)), tuple(np.round(d1[1]).astype(int)), (0,255,0), 3, cv2.LINE_AA)
        cv2.line(img, tuple(np.round(d2[0]).astype(int)), tuple(np.round(d2[1]).astype(int)), (255,0,255), 3, cv2.LINE_AA)
        cross(img, g["hidden_corner_pixel_uv"], (0, 0, 255), "HIDDEN-CORNER")
        cross(img, g["temporary_grasp_pixel_uv"], (0, 255, 255), "TEMP-GRASP")
        direct = g["temporary_grasp_arm2_xy_direct"]
        affine = g["temporary_grasp_arm2_xy_affine"]
        delta = g["temporary_grasp_crosscheck_error_mm"]
        outlined(img, f"TEMP-GRASP ARM2 direct=({direct[0]:.1f},{direct[1]:.1f}) affine=({affine[0]:.1f},{affine[1]:.1f}) delta={delta:.1f}mm", (25, 205), 0.58, (0,255,255))
    outlined(img, "BASKET MODE" if basket_mode else "NORMAL VIEW", (25, 40), 0.85, (0,255,255) if basket_mode else (180,180,180))
    outlined(img, f"calibration={'ACTIVE' if active else 'IDLE'} points={len(calib.points)}/5 geometry={'READY' if calib.geometry else 'NO'}", (25, 75), 0.58, (255,255,0))
    if basket_mode and active:
        spec = calib.next_spec()
        message = "Pending click: move ARM2 tip and press Enter" if pending is not None else (f"NEXT {spec[0]}: {spec[1]}" if spec else "Finalizing")
        outlined(img, message, (25, 110), 0.54, (0,255,255))
    elif hover_plan is not None:
        outlined(
            img,
            (
                f"HOVER READY: Enter execute | target=({hover_plan['target_x']:.1f},"
                f"{hover_plan['target_y']:.1f},{hover_plan['target_z']:.1f}) "
                f"safe_z={hover_plan['safe_z']:.1f}"
            ),
            (25, 110),
            0.54,
            (0, 255, 255),
        )
    elif basket_mode and calib.geometry:
        outlined(img, "H prepare hover plan | Enter execute after plan", (25, 110), 0.54, (0,255,255))
    outlined(img, "B mode | C calibrate | H hover plan | Enter confirm | U cancel | L load | Q quit", (25, 145), 0.52, (255,255,255))



@dataclass
class PoseSample:
    timestamp: float
    phase: str
    step_index: int
    commanded_z: float
    x: float
    y: float
    z: float
    tor_b: float
    tor_s: float
    tor_e: float
    tor_h: float

    @property
    def torques(self):
        return self.tor_b, self.tor_s, self.tor_e, self.tor_h


def _finite_float(value: Any, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"Invalid {name}: {value!r}") from exc
    if not math.isfinite(result):
        raise RuntimeError(f"Non-finite {name}: {value!r}")
    return result


def _sample_from_feedback(feedback: Dict[str, Any], phase: str, step_index: int, commanded_z: float) -> PoseSample:
    return PoseSample(
        timestamp=time.time(), phase=str(phase), step_index=int(step_index),
        commanded_z=float(commanded_z),
        x=_finite_float(feedback.get("x"), "x"),
        y=_finite_float(feedback.get("y"), "y"),
        z=_finite_float(feedback.get("z"), "z"),
        tor_b=_finite_float(feedback.get("torB", 0.0), "torB"),
        tor_s=_finite_float(feedback.get("torS", 0.0), "torS"),
        tor_e=_finite_float(feedback.get("torE", 0.0), "torE"),
        tor_h=_finite_float(feedback.get("torH", 0.0), "torH"),
    )


def _median_torques(samples: Sequence[PoseSample]):
    cols = list(zip(*(sample.torques for sample in samples)))
    return tuple(float(statistics.median(col)) for col in cols)


def _torque_delta(current: Sequence[float], reference: Sequence[float]):
    return tuple(float(a) - float(b) for a, b in zip(current, reference))


def _torque_norm(values: Sequence[float]) -> float:
    return math.sqrt(sum(float(v) ** 2 for v in values))


def _dominant_axis(values: Sequence[float]):
    names = ("B", "S", "E", "H")
    idx = max(range(4), key=lambda i: abs(float(values[i])))
    return names[idx], float(values[idx])


def _descending_targets(start_z: float, end_z: float, step_mm: float):
    start, end, step = float(start_z), float(end_z), abs(float(step_mm))
    if step <= 0:
        raise ValueError("step must be positive")
    if start <= end:
        return []
    result = []
    current = start
    while current - step > end:
        current -= step
        result.append(current)
    if not result or abs(result[-1] - end) > 1e-6:
        result.append(end)
    return result


def _gripper_angle(open_percent: float, fully_open: float, fully_closed: float) -> float:
    ratio = min(100.0, max(0.0, float(open_percent))) / 100.0
    return float(fully_closed) + ratio * (float(fully_open) - float(fully_closed))


def _write_probe_header(writer: csv.writer):
    writer.writerow([
        "time_iso", "phase", "step_index", "commanded_z_mm",
        "actual_x_mm", "actual_y_mm", "actual_z_mm", "z_tracking_error_mm",
        "torB", "torS", "torE", "torH",
        "delta_base_B", "delta_base_S", "delta_base_E", "delta_base_H",
        "delta_step_B", "delta_step_S", "delta_step_E", "delta_step_H",
    ])


def _write_probe_sample(writer: csv.writer, sample: PoseSample, delta_base, delta_step):
    writer.writerow([
        datetime.fromtimestamp(sample.timestamp).isoformat(timespec="milliseconds"),
        sample.phase, sample.step_index, f"{sample.commanded_z:.6f}",
        f"{sample.x:.6f}", f"{sample.y:.6f}", f"{sample.z:.6f}",
        f"{sample.z - sample.commanded_z:.6f}",
        f"{sample.tor_b:.3f}", f"{sample.tor_s:.3f}",
        f"{sample.tor_e:.3f}", f"{sample.tor_h:.3f}",
        *[f"{float(v):.3f}" for v in delta_base],
        *[f"{float(v):.3f}" for v in delta_step],
    ])


def _print_probe_sample(sample: PoseSample, baseline, previous):
    delta_base = _torque_delta(sample.torques, baseline)
    delta_step = _torque_delta(sample.torques, previous)
    axis_b, value_b = _dominant_axis(delta_base)
    axis_s, value_s = _dominant_axis(delta_step)
    print(
        f"[{sample.phase}:{sample.step_index:02d}] cmd_z={sample.commanded_z:.2f} "
        f"actual_z={sample.z:.2f} z_error={sample.z-sample.commanded_z:+.2f}mm "
        f"xy=({sample.x:.2f},{sample.y:.2f})"
    )
    print(
        f"  torque raw B={sample.tor_b:+.0f} S={sample.tor_s:+.0f} "
        f"E={sample.tor_e:+.0f} H={sample.tor_h:+.0f}"
    )
    print(
        f"  delta-base B={delta_base[0]:+.0f} S={delta_base[1]:+.0f} "
        f"E={delta_base[2]:+.0f} H={delta_base[3]:+.0f} "
        f"norm={_torque_norm(delta_base):.1f} dominant={axis_b}{value_b:+.0f}"
    )
    print(
        f"  delta-step B={delta_step[0]:+.0f} S={delta_step[1]:+.0f} "
        f"E={delta_step[2]:+.0f} H={delta_step[3]:+.0f} "
        f"norm={_torque_norm(delta_step):.1f} dominant={axis_s}{value_s:+.0f}"
    )
    return delta_base, delta_step


def _line_intersection_2d(a0, a1, b0, b1):
    """Return the infinite-line intersection of a0-a1 and b0-b1."""
    p = np.asarray(a0, dtype=np.float64)
    r = np.asarray(a1, dtype=np.float64) - p
    q = np.asarray(b0, dtype=np.float64)
    v = np.asarray(b1, dtype=np.float64) - q
    cross = float(r[0] * v[1] - r[1] * v[0])
    if abs(cross) < 1e-9:
        raise RuntimeError("folding-board diagonals are parallel or degenerate")
    qp = q - p
    t = float((qp[0] * v[1] - qp[1] * v[0]) / cross)
    return p + t * r


def _load_board_center_arm2(config_path: str):
    """Load board diagonal center and convert it to ARM2 XY using the JSON affine."""
    path = Path(config_path)
    if not path.exists():
        raise RuntimeError(f"board config not found: {config_path}")
    with path.open("r", encoding="utf-8") as fp:
        config = json.load(fp)

    marker_map = config.get("aruco", {}).get("marker_board_mm", {})
    required = ("0", "1", "2", "3")
    if not all(key in marker_map for key in required):
        raise RuntimeError("board config must contain marker_board_mm IDs 0,1,2,3")
    p0 = np.asarray(marker_map["0"], dtype=np.float64)
    p1 = np.asarray(marker_map["1"], dtype=np.float64)
    p2 = np.asarray(marker_map["2"], dtype=np.float64)
    p3 = np.asarray(marker_map["3"], dtype=np.float64)
    if not all(point.shape == (2,) and np.all(np.isfinite(point)) for point in (p0,p1,p2,p3)):
        raise RuntimeError("invalid board marker coordinates")

    # Diagonals are ID0-ID3 and ID1-ID2.
    center_board = _line_intersection_2d(p0, p3, p1, p2)

    arm2_config = config.get("dual_roarm", {}).get("arm2", {})
    affine_raw = arm2_config.get("board_to_roarm_affine_2x3")
    affine = np.asarray(affine_raw, dtype=np.float64)
    if affine.shape != (2, 3) or not np.all(np.isfinite(affine)):
        raise RuntimeError("invalid dual_roarm.arm2.board_to_roarm_affine_2x3")
    center_arm2 = affine[:, :2] @ center_board + affine[:, 2]
    # Board +Y points from ID1/ID3 (lower edge) toward ID0/ID2 (upper edge).
    # Transform only the direction (no translation) into ARM2 XY.
    board_up_arm2 = affine[:, :2] @ np.asarray([0.0, 1.0], dtype=np.float64)
    board_up_norm = float(np.linalg.norm(board_up_arm2))
    if not np.isfinite(board_up_norm) or board_up_norm < 1e-9:
        raise RuntimeError("invalid ARM2 affine board-Y direction")
    board_up_arm2 = board_up_arm2 / board_up_norm

    corners = np.vstack((p0, p1, p2, p3))
    center_mean = np.mean(corners, axis=0)
    diagonal_vs_mean_error = float(np.linalg.norm(center_board - center_mean))

    # Prefer the manually calibrated ARM2 RED_EXTRA point as an inward recovery
    # waypoint.  It is a known-reachable point closer to the robot base than the
    # basket grasp point.  Fall back to a conservative point just inside the
    # board center if the label is unavailable.
    inner_xy = None
    calib_points = arm2_config.get("calib_points", [])
    calib_roarm_points = arm2_config.get("calib_roarm_points", [])
    if isinstance(calib_points, list) and isinstance(calib_roarm_points, list):
        for index, point in enumerate(calib_points):
            if not isinstance(point, dict) or str(point.get("label", "")).upper() != "RED_EXTRA":
                continue
            if index >= len(calib_roarm_points):
                continue
            candidate = np.asarray(calib_roarm_points[index], dtype=np.float64)
            if candidate.shape == (2,) and np.all(np.isfinite(candidate)):
                inner_xy = candidate
                break
    if inner_xy is None:
        inner_xy = np.asarray(
            [float(center_arm2[0]) - 50.0, float(center_arm2[1]) + 20.0],
            dtype=np.float64,
        )

    safe_hover_z = float(arm2_config.get("safe_hover_z", 180.0))
    if not np.isfinite(safe_hover_z):
        safe_hover_z = 180.0
    return {
        "config_path": str(path),
        "board_center_xy": [float(center_board[0]), float(center_board[1])],
        "arm2_center_xy": [float(center_arm2[0]), float(center_arm2[1])],
        "arm2_board_up_unit": [float(board_up_arm2[0]), float(board_up_arm2[1])],
        "arm2_inner_xy": [float(inner_xy[0]), float(inner_xy[1])],
        "safe_hover_z": safe_hover_z,
        "diagonal_vs_corner_mean_error_mm": diagonal_vs_mean_error,
    }


def main():
    print(f"[BUILD] {BUILD_ID}")
    print("[CAMERA] A150 capture path; no FOURCC/FPS forcing")
    print("[WORKFLOW] one serial session: safe hover approach -> torque descent -> grasp -> fast direct lift -> retention gate -> FAILURE auto-regrasp / SUCCESS one-shot board-center transit -> release -> A150 standby")
    print("[SAFETY] ARM2 only; when H starts from A150 standby, V24 preserves the V21 forced standby -> high-Z board-inner/RED_EXTRA waypoint -> basket before descent; direct standby -> basket transit is forbidden")
    print("[AUTO-REGRASP] after every completed lift, V24 samples torS+torE three times; SUCCESS continues to board center, FAILURE blocks transfer and repeats the same monitored basket descent/grasp/lift")
    parser = argparse.ArgumentParser(description="ARM2 basket 5-point affine calibration and diagonal-center visualization")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--backend", default="v4l2", choices=["auto","v4l2","dshow","any"])
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--display", default=":0")
    parser.add_argument("--no-window", action="store_true")
    parser.add_argument("--no-terminal-control", action="store_true")
    parser.add_argument("--auto-run-once", action="store_true", default=False,
                        help="VLA adapter: run B->H->Enter->D->Enter once without operator key input")
    parser.add_argument("--terminal-line-mode", action="store_true")
    parser.add_argument("--send", action="store_true")
    parser.add_argument("--arm2-port", default="/dev/roarm_2")
    parser.add_argument("--baudrate", type=int, default=115200)
    parser.add_argument("--feedback-timeout", type=float, default=2.5)
    parser.add_argument("--grip-open", type=float, default=1.35)
    parser.add_argument("--grip-spd", type=float, default=0.0)
    parser.add_argument("--grip-acc", type=float, default=0.0)
    parser.add_argument("--calib-file", default="basket_arm2_5point_affine.json")
    parser.add_argument("--load-calib", action="store_true", default=True)
    parser.add_argument("--no-load-calib", dest="load_calib", action="store_false")
    parser.add_argument("--max-mean-error-mm", type=float, default=15.0)
    parser.add_argument("--max-point-error-mm", type=float, default=25.0)
    parser.add_argument("--max-center-crosscheck-mm", type=float, default=25.0)
    parser.add_argument("--min-line-angle-deg", type=float, default=8.0)
    parser.add_argument("--min-polygon-area-px", type=float, default=5000.0)
    parser.add_argument("--max-hidden-margin-ratio", type=float, default=0.35)
    parser.add_argument("--move-command", type=int, default=104)
    parser.add_argument("--speed", type=float, default=1.12,
                        help="ARM2 speed for vertical raise and high-Z horizontal transit")
    parser.add_argument("--descent-speed", type=float, default=0.35,
                        help="Slower ARM2 speed for the final vertical descent")
    parser.add_argument("--hover-offset-mm", type=float, default=30.0,
                        help="Offset added to the five-point mean Z")
    parser.add_argument("--min-rim-clearance-mm", type=float, default=30.0,
                        help="Final hover must remain this far above the highest calibrated rim point")
    parser.add_argument("--transit-clearance-mm", type=float, default=100.0,
                        help="Safe-Z margin above final hover/highest rim before XY transit")
    parser.add_argument("--move-wait", type=float, default=0.20,
                        help="Minimum settle delay after each T:104 segment")
    parser.add_argument("--move-timeout", type=float, default=15.0,
                        help="Maximum seconds to wait for each commanded waypoint")
    parser.add_argument("--move-tolerance-mm", type=float, default=25.0,
                        help="3D feedback tolerance before the next segment is allowed")
    parser.add_argument("--move-poll-sec", type=float, default=0.30,
                        help="T:105 polling interval while waiting for a waypoint")
    parser.add_argument("--tool-angle-fallback", type=float, default=2.80,
                        help="T:104 tool angle used only if T:105 has no valid t field")
    parser.add_argument("--x-min", type=float, default=-400.0)
    parser.add_argument("--x-max", type=float, default=400.0)
    parser.add_argument("--y-min", type=float, default=-100.0)
    parser.add_argument("--y-max", type=float, default=500.0)
    parser.add_argument("--z-min", type=float, default=-340.0)
    parser.add_argument("--z-max", type=float, default=350.0)
    parser.add_argument("--startup-timeout", type=float, default=15.0)
    parser.add_argument("--startup-quiet-sec", type=float, default=1.2)
    parser.add_argument("--xy-start-tolerance-mm", type=float, default=35.0)
    parser.add_argument("--basket-floor-z", type=float, default=-325.8368564)
    parser.add_argument("--floor-clearance-mm", type=float, default=15.0)
    parser.add_argument("--gripper-open-percent", type=float, default=30.0)
    parser.add_argument("--grip-fully-open", type=float, default=1.35)
    parser.add_argument("--grip-fully-closed", type=float, default=3.05)
    parser.add_argument("--fast-step-mm", type=float, default=20.0)
    parser.add_argument("--fast-speed", type=float, default=0.90)
    parser.add_argument("--slow-step-mm", type=float, default=5.0)
    parser.add_argument("--slow-speed", type=float, default=0.40)
    parser.add_argument("--probe-poll-sec", type=float, default=0.18)
    parser.add_argument("--probe-settle-wait-sec", type=float, default=0.10)
    parser.add_argument("--probe-step-timeout", type=float, default=5.0)
    parser.add_argument("--probe-z-arrival-tolerance-mm", type=float, default=12.0)
    parser.add_argument("--probe-z-stable-span-mm", type=float, default=2.0)
    parser.add_argument("--probe-settle-samples", type=int, default=3)
    parser.add_argument("--baseline-samples", type=int, default=7)
    parser.add_argument("--baseline-interval-sec", type=float, default=0.10)
    parser.add_argument("--contact-axis-delta", type=float, default=120.0)
    parser.add_argument("--contact-vector-delta", type=float, default=200.0)
    parser.add_argument("--contact-confirm-steps", type=int, default=2)
    parser.add_argument("--hard-axis-delta", type=float, default=300.0)
    parser.add_argument("--stall-min-command-mm", type=float, default=4.0)
    parser.add_argument("--stall-max-actual-mm", type=float, default=1.5)
    parser.add_argument("--stall-confirm-steps", type=int, default=2)
    parser.add_argument("--no-auto-contact-stop", action="store_true")
    parser.add_argument("--gripper-settle-sec", type=float, default=1.2,
                        help="Wait after T:106 before any descent T:104 command")
    parser.add_argument("--fast-hard-se-axis-delta", type=float, default=140.0,
                        help="Fast-approach emergency stop threshold using shoulder/elbow only")
    parser.add_argument("--contact-shoulder-delta", type=float, default=40.0,
                        help="Required shoulder torque decrease from rim baseline")
    parser.add_argument("--contact-elbow-delta", type=float, default=20.0,
                        help="Required absolute elbow torque change from rim baseline")
    parser.add_argument("--contact-z-lag-mm", type=float, default=2.5,
                        help="Auxiliary Z tracking-lag threshold for logging; not sufficient alone to stop")
    parser.add_argument("--post-contact-open-percent", type=float, default=90.0,
                        help="Gripper opening percentage applied after confirmed contact before closing")
    parser.add_argument("--post-contact-open-settle-sec", type=float, default=0.8,
                        help="Wait after widening the gripper at the confirmed contact pose")
    parser.add_argument("--grasp-close-angle", type=float, default=3.05,
                        help="T:106 angle used to close and hold the gripper for the test grasp")
    parser.add_argument("--grasp-close-settle-sec", type=float, default=1.2,
                        help="Wait after closing before the vertical test lift")
    parser.add_argument("--gripper-feedback-tolerance-rad", type=float, default=0.18,
                        help="Allowed T:105 t error when verifying the post-contact opening")
    parser.add_argument("--gripper-feedback-attempts", type=int, default=3,
                        help="Maximum T:106 resend/feedback attempts for post-contact opening")
    parser.add_argument("--gripper-min-close-motion-rad", type=float, default=0.20,
                        help="Minimum t increase after close command before lift is allowed")
    parser.add_argument("--board-config", default="dual_roarm_folding_board_config.json",
                        help="Folding-board JSON containing marker_board_mm and ARM2 board affine")
    parser.add_argument("--pickup-lift-z", type=float, default=180.0,
                        help="Preferred absolute ARM2 transit Z; capped by config safe_hover_z")
    parser.add_argument("--basket-exit-clearance-mm", type=float, default=145.0,
                        help="Minimum gripper Z clearance above highest basket rim before inward XY recovery starts")
    parser.add_argument("--inward-recovery-z-droop-mm", type=float, default=5.0,
                        help="Entry relaxation below basket-exit-clearance-mm after a confirmed vertical lift saturation")
    parser.add_argument("--inward-recovery-travel-extra-droop-mm", type=float, default=5.0,
                        help="Additional Z droop allowed only during each short inward recovery XY hop")
    parser.add_argument("--inward-recovery-hop-mm", type=float, default=35.0,
                        help="Short XY hop toward board center after lift saturation before re-lifting")
    parser.add_argument("--post-saturation-required-rise-mm", type=float, default=120.0,
                        help="Deprecated compatibility option; V18 transfers directly at the achieved lift Z after first saturation")
    parser.add_argument("--recovery-hop-xy-tolerance-mm", type=float, default=10.0,
                        help="Deprecated compatibility option; V16 recovery hops are accepted by actual centerward progress after XY settles")
    parser.add_argument("--inner-xy-step-mm", type=float, default=80.0,
                        help="Maximum constant-Z XY segment length during normal board transit")
    parser.add_argument("--recovery-xy-speed", type=float, default=0.95,
                        help="Deprecated compatibility option; V18 does not use recovery hops")
    parser.add_argument("--vertical-waypoint-tolerance-mm", type=float, default=25.0,
                        help="Dedicated Z-path waypoint tolerance used by adaptive lift")
    parser.add_argument("--vertical-stall-polls", type=int, default=5,
                        help="Consecutive nearly unchanged Z feedback samples used to declare lift saturation")
    parser.add_argument("--vertical-stall-span-mm", type=float, default=1.0,
                        help="Maximum Z span across stall polls before the lift is considered saturated")
    parser.add_argument("--minimum-board-transit-z", type=float, default=120.0,
                        help="Minimum accepted Z for board-center transit if the inner lift also saturates")
    parser.add_argument("--board-descent-step-mm", type=float, default=30.0,
                        help="Maximum vertical descent increment at board center")
    parser.add_argument("--board-release-z", type=float, default=24.913311,
                        help="Absolute ARM2 Z at board center immediately before gripper release")
    parser.add_argument("--test-lift-speed", type=float, default=0.95,
                        help="ARM2 speed for the initial fast vertical garment lift (V18 default 0.95)")
    parser.add_argument("--test-lift-step-mm", type=float, default=25.0,
                        help="Deprecated compatibility option; V18 initial lift uses one direct T:104 to the requested Z")
    parser.add_argument("--board-transit-speed", type=float, default=0.95,
                        help="ARM2 speed for the direct constant-commanded-Z basket-to-board-center transit (V18 default 0.95)")
    parser.add_argument("--direct-transit-clearance-mm", type=float, default=5.0,
                        help="Deprecated compatibility option; V18 accepts the achieved lift Z and does not use an absolute transit-Z gate")
    parser.add_argument("--standby-roarm-x", type=float, default=-3.342226292,
                        help="A150 taught ARM2 standby RoArm X")
    parser.add_argument("--standby-roarm-y", type=float, default=-108.9054583,
                        help="A150 taught ARM2 standby RoArm Y")
    parser.add_argument("--standby-roarm-z", type=float, default=21.07581154,
                        help="A150 taught ARM2 standby RoArm Z")
    parser.add_argument("--standby-roarm-t", type=float, default=2.816388727,
                        help="A150 taught ARM2 standby tool angle")
    parser.add_argument("--standby-gripper-angle", type=float, default=1.57,
                        help="A150 imaging-standby slightly-open gripper angle")
    parser.add_argument("--standby-speed", type=float, default=1.12,
                        help="ARM2 speed for the final return to the A150 taught standby pose")
    parser.add_argument("--board-descent-speed", type=float, default=0.35,
                        help="ARM2 speed for vertical descent at the board center")
    parser.add_argument("--release-open-angle", type=float, default=1.35,
                        help="T:106 angle used to release the garment over the board")
    parser.add_argument("--release-settle-sec", type=float, default=1.0,
                        help="Wait after opening the gripper at the board center")
    parser.add_argument("--release-feedback-tolerance-rad", type=float, default=0.22,
                        help="Allowed T:105 t error when verifying release opening")
    parser.add_argument("--board-entry-extra-mm", type=float, default=0.0,
                        help="Initial V24 behavior: stop at exact board center before air release")
    parser.add_argument("--pre-swing-shake-mm", type=float, default=150.0,
                        help="High-Z fore/aft shake amplitude: front +150mm, back -150mm")
    parser.add_argument("--pre-swing-shake-extra-z-mm", type=float, default=100.0,
                        help="raise another 100mm before the full-stroke shake")
    parser.add_argument("--pre-swing-shake-cycles", type=int, default=2)
    parser.add_argument("--pre-swing-shake-duration-s", type=float, default=2.5,
                        help="Total duration for full-stroke +150/-150 two-cycle shake")
    parser.add_argument("--pre-swing-shake-hz", type=float, default=30.0)
    parser.add_argument("--swing-back-mm", type=float, default=150.0,
                        help="High-Z backswing distance toward ID1/ID3 before laydown")
    parser.add_argument("--swing-forward-mm", type=float, default=300.0,
                        help="Forward laydown distance toward ID0/ID2")
    parser.add_argument("--swing-stream-hz", type=float, default=22.0)
    parser.add_argument("--swing-back-duration-s", type=float, default=1.30)
    parser.add_argument("--swing-forward-duration-s", type=float, default=2.30)
    parser.add_argument("--swing-curve-rise-mm", type=float, default=100.0,
                        help="D56 quarter-ellipse extra Z rise during backswing")
    parser.add_argument("--arm2-board-contact-z", type=float, default=-97.21527656,
                        help="ARM2 folding-board gripper contact Z from the current affine calibration")
    parser.add_argument("--swing-release-clearance-mm", type=float, default=30.0,
                        help="Final gripper height above the V24 board release Z")
    parser.add_argument("--swing-exp-decay", type=float, default=0.25,
                        help="D56-style normalized exponential descent decay")
    parser.add_argument("--swing-vertical-gamma", type=float, default=2.40,
                        help="Larger than D56 default: hold height longer, descend more strongly near the end")
    parser.add_argument("--grasp-retention-threshold", type=float, default=220.0,
                        help="Diagnostic failure threshold for the median of three post-lift torS+torE samples")
    parser.add_argument("--grasp-retention-samples", type=int, default=3,
                        help="Number of post-lift T:105 torque samples used by the diagnostic (default 3)")
    parser.add_argument("--grasp-retention-interval-sec", type=float, default=0.12,
                        help="Delay between post-lift diagnostic torque samples")
    parser.add_argument("--max-auto-regrasp-retries", type=int, default=3,
                        help="Maximum additional automatic regrasp attempts after the first failed lift diagnosis")
    parser.add_argument("--probe-csv", default="")
    args = parser.parse_args()

    board_target = _load_board_center_arm2(args.board_config)
    board_center_x, board_center_y = board_target["arm2_center_xy"]
    board_up_arm2 = np.asarray(board_target["arm2_board_up_unit"], dtype=np.float64)
    board_inner_x, board_inner_y = board_target["arm2_inner_xy"]
    config_safe_hover_z = float(board_target["safe_hover_z"])
    print(
        f"[BOARD] config={board_target['config_path']} "
        f"board_center=({board_target['board_center_xy'][0]:.3f},"
        f"{board_target['board_center_xy'][1]:.3f}) "
        f"ARM2_center=({board_center_x:.3f},{board_center_y:.3f}) "
        f"ARM2_inner=({board_inner_x:.3f},{board_inner_y:.3f}) "
        f"safe_hover_z={config_safe_hover_z:.3f} "
        f"diag-vs-mean={board_target['diagonal_vs_corner_mean_error_mm']:.6f}mm"
    )

    if args.display and not args.no_window:
        os.environ["DISPLAY"] = args.display
    cap = open_camera(args.camera, args.width, args.height, args.backend)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera index {args.camera}")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(
        f"[CAMERA] A150 capture path opened index={args.camera} "
        f"requested={args.width}x{args.height} actual={width}x{height} "
        f"backend={args.backend}"
    )

    arm = RoArmSerial(args.arm2_port, args.baudrate) if args.send else None
    if arm is not None:
        arm.synchronize_startup(args.startup_timeout, args.startup_quiet_sec)
    print(f"[MODE] {'ARM2 connected' if arm else 'vision/load only; add --send for calibration feedback'}")
    calib = BasketCalib()
    if args.load_calib and os.path.exists(args.calib_file):
        try:
            calib.load(args.calib_file)
        except Exception as exc:
            print(f"[CALIB-WARN] load failed: {exc!r}")

    basket_mode = False
    active = False
    pending: Optional[Tuple[int, int]] = None
    hover_plan: Optional[Dict[str, float]] = None
    descent_plan: Optional[Dict[str, float]] = None
    grasp_lift_plan: Optional[Dict[str, float]] = None
    grasp_retention_cycle = 0
    hover_ready = False
    window = "Basket 5-Point Affine Calibration"
    if not args.no_window:
        cv2.namedWindow(window)

    def next_instruction():
        spec = calib.next_spec()
        if spec:
            print(f"\n[NEXT] {spec[0]}: {spec[1]}\n  click image -> move ARM2 tip -> Enter")

    def finalize():
        nonlocal active, pending
        try:
            g = calib.compute(width, height, args)
            calib.save(args.calib_file, str(args.camera), width, height, args.arm2_port)
        except Exception as exc:
            print(f"[CALIB-ERROR] {exc}")
            print("Use U to remove the last point, or C to restart")
            return False
        print("\n========== CALIBRATION RESULT ==========")
        print("affine matrix pixel(u,v)->ARM2(x,y):")
        print(np.array2string(calib.matrix, precision=8))
        print(f"mean_error={float(np.mean(calib.errors)):.2f}mm max_error={float(np.max(calib.errors)):.2f}mm")
        print(f"hidden_corner_pixel_uv={g['hidden_corner_pixel_uv']}")
        print(f"temporary_grasp_pixel_uv={g['temporary_grasp_pixel_uv']}")
        print(f"temporary_grasp_arm2_xy_direct={g['temporary_grasp_arm2_xy_direct']}")
        print(f"temporary_grasp_arm2_xy_affine={g['temporary_grasp_arm2_xy_affine']}")
        print(f"crosscheck_error={g['temporary_grasp_crosscheck_error_mm']:.2f}mm")
        print(f"saved_json={args.calib_file}")
        print("========================================\n")
        active, pending = False, None
        if arm:
            arm.torque_on()
            arm.gripper_open(args.grip_open, args.grip_spd, args.grip_acc)
        return True

    def _require_in_range(name: str, value: float, lower: float, upper: float):
        if not np.isfinite(value) or value < lower or value > upper:
            raise RuntimeError(
                f"{name}={value:.2f} is outside safety range [{lower:.2f}, {upper:.2f}]"
            )

    def prepare_hover_plan():
        nonlocal hover_plan, descent_plan, hover_ready
        descent_plan = None
        hover_ready = False
        if not basket_mode:
            print("[HOVER-BLOCKED] press B to enter BASKET MODE first")
            return False
        if active:
            print("[HOVER-BLOCKED] calibration is active; finish or cancel it first")
            return False
        if arm is None:
            print("[HOVER-BLOCKED] run with --send")
            return False
        if not calib.geometry or len(calib.points) != 5:
            print(f"[HOVER-BLOCKED] load a valid calibration JSON first: {args.calib_file}")
            return False

        feedback = arm.feedback(args.feedback_timeout)
        if feedback is None or not all(k in feedback for k in ("x", "y", "z")):
            print("[HOVER-ERROR] ARM2 T:105 feedback failed")
            return False

        geometry = calib.geometry
        direct_xy = geometry.get("temporary_grasp_arm2_xy_direct")
        if not isinstance(direct_xy, (list, tuple)) or len(direct_xy) != 2:
            print("[HOVER-ERROR] temporary_grasp_arm2_xy_direct missing from JSON")
            return False

        z_values = np.asarray(
            [float(point["arm2_xyz"][2]) for point in calib.points],
            dtype=np.float64,
        )
        if z_values.size != 5 or not np.all(np.isfinite(z_values)):
            print("[HOVER-ERROR] invalid five-point Z values")
            return False

        current_x = float(feedback["x"])
        current_y = float(feedback["y"])
        current_z = float(feedback["z"])
        tool_t = float(feedback.get("t", args.tool_angle_fallback))
        if not np.isfinite(tool_t):
            tool_t = float(args.tool_angle_fallback)

        target_x = float(direct_xy[0])
        target_y = float(direct_xy[1])
        rim_z_mean = float(np.mean(z_values))
        rim_z_max = float(np.max(z_values))

        # Use the five-point mean as the requested reference, but never allow
        # the final hover to fall below the highest rim plus safety clearance.
        target_z_from_mean = rim_z_mean + float(args.hover_offset_mm)
        target_z_from_rim = rim_z_max + float(args.min_rim_clearance_mm)
        target_z = max(target_z_from_mean, target_z_from_rim)

        # XY travel occurs only at a high safe Z. Larger Z is physically higher
        # in the RoArm Cartesian convention used by this project.
        safe_z = max(
            current_z,
            target_z + float(args.transit_clearance_mm),
            rim_z_max + float(args.transit_clearance_mm),
        )

        # V21 safety route: after an automatic place cycle ARM2 returns to the
        # exact A150 taught standby.  A direct standby -> basket XY command can
        # make the inverse-kinematics branch fold the arm through an unsafe
        # posture.  Detect that known standby pose with the SAME tolerance used
        # by the standby arrival gate, and force a board-side waypoint first.
        # The board-side waypoint is the already-loaded ARM2 inner point
        # (preferentially RED_EXTRA), i.e. a calibrated known-reachable point
        # over the folding board rather than a newly invented coordinate.
        standby_xyz = np.asarray(
            [
                float(args.standby_roarm_x),
                float(args.standby_roarm_y),
                float(args.standby_roarm_z),
            ],
            dtype=np.float64,
        )
        current_xyz = np.asarray([current_x, current_y, current_z], dtype=np.float64)
        standby_distance = float(np.linalg.norm(current_xyz - standby_xyz))
        standby_route_required = standby_distance <= float(args.move_tolerance_mm)

        board_safe_x = float(board_inner_x)
        board_safe_y = float(board_inner_y)
        board_safe_z = float(safe_z)
        if standby_route_required:
            # Use the board config's established safe-hover Z for the first
            # horizontal departure from standby.  This intentionally overrides
            # the old ~20 mm transit height that produced the dangerous fold.
            board_safe_z = max(board_safe_z, float(config_safe_hover_z))
            safe_z = board_safe_z

        _require_in_range("target_x", target_x, args.x_min, args.x_max)
        _require_in_range("target_y", target_y, args.y_min, args.y_max)
        _require_in_range("target_z", target_z, args.z_min, args.z_max)
        _require_in_range("safe_z", safe_z, args.z_min, args.z_max)
        if standby_route_required:
            _require_in_range("board_safe_x", board_safe_x, args.x_min, args.x_max)
            _require_in_range("board_safe_y", board_safe_y, args.y_min, args.y_max)
            _require_in_range("board_safe_z", board_safe_z, args.z_min, args.z_max)

        hover_plan = {
            "start_x": current_x,
            "start_y": current_y,
            "start_z": current_z,
            "target_x": target_x,
            "target_y": target_y,
            "target_z": target_z,
            "safe_z": safe_z,
            "tool_t": tool_t,
            "rim_z_mean": rim_z_mean,
            "rim_z_max": rim_z_max,
            "target_z_from_mean": target_z_from_mean,
            "target_z_from_rim": target_z_from_rim,
            "standby_route_required": bool(standby_route_required),
            "standby_distance_mm": float(standby_distance),
            "board_safe_x": board_safe_x,
            "board_safe_y": board_safe_y,
            "board_safe_z": board_safe_z,
        }

        print("\n========== ARM2 HOVER PLAN ==========")
        print(
            f"start=({current_x:.2f},{current_y:.2f},{current_z:.2f}) "
            f"tool_t={tool_t:.3f}"
        )
        print(f"temporary_grasp_xy=({target_x:.2f},{target_y:.2f})")
        print(
            f"rim_z_mean={rim_z_mean:.2f} rim_z_max={rim_z_max:.2f} "
            f"mean+offset={target_z_from_mean:.2f} "
            f"max+clearance={target_z_from_rim:.2f}"
        )
        print(f"final_hover_z={target_z:.2f} safe_transit_z={safe_z:.2f}")
        if standby_route_required:
            print(
                f"[STANDBY-SAFE-ROUTE] current pose is within "
                f"{float(args.move_tolerance_mm):.1f}mm of A150 standby "
                f"(distance={standby_distance:.1f}mm)"
            )
            print("path:")
            print(f"  1. vertical raise at standby XY -> z={board_safe_z:.2f}")
            print(
                f"  2. REQUIRED board-safe waypoint -> "
                f"({board_safe_x:.2f},{board_safe_y:.2f},{board_safe_z:.2f}) "
                "[ARM2 inner/RED_EXTRA; over folding board]"
            )
            print(
                f"  3. only after board-safe arrival, high-Z move to basket -> "
                f"({target_x:.2f},{target_y:.2f},{safe_z:.2f})"
            )
            print(f"  4. final vertical descent -> ({target_x:.2f},{target_y:.2f},{target_z:.2f})")
        else:
            print("path:")
            print(f"  1. vertical raise at current XY -> z={safe_z:.2f}")
            print(f"  2. high-Z horizontal move -> ({target_x:.2f},{target_y:.2f},{safe_z:.2f})")
            print(f"  3. final vertical descent -> ({target_x:.2f},{target_y:.2f},{target_z:.2f})")
        print("ARM1 command: NONE | gripper close: NONE | clothing lift: NONE")
        print("Press Enter to execute, or U to cancel.")
        print("=====================================\n")
        return True

    def wait_for_waypoint(label: str, x: float, y: float, z: float, allow_abort: bool = False):
        if arm is None:
            raise RuntimeError("ARM2 is not connected")
        deadline = time.time() + max(0.5, float(args.move_timeout))
        target = np.asarray([float(x), float(y), float(z)], dtype=np.float64)
        last_error = float("inf")
        last_pose = None
        while time.time() < deadline:
            if allow_abort and _probe_abort_requested():
                raise KeyboardInterrupt(f"operator abort during {label}")
            time.sleep(max(0.05, float(args.move_poll_sec)))
            feedback = arm.feedback(args.feedback_timeout)
            if feedback is None or not all(k in feedback for k in ("x", "y", "z")):
                continue
            actual = np.asarray(
                [float(feedback["x"]), float(feedback["y"]), float(feedback["z"])],
                dtype=np.float64,
            )
            last_pose = actual
            last_error = float(np.linalg.norm(actual - target))
            print(
                f"[{label}] feedback=({actual[0]:.1f},{actual[1]:.1f},{actual[2]:.1f}) "
                f"error={last_error:.1f}mm"
            )
            if last_error <= float(args.move_tolerance_mm):
                time.sleep(max(0.0, float(args.move_wait)))
                return actual
        pose_text = "unavailable" if last_pose is None else (
            f"({last_pose[0]:.1f},{last_pose[1]:.1f},{last_pose[2]:.1f})"
        )
        raise RuntimeError(
            f"{label} waypoint timeout: target=({x:.1f},{y:.1f},{z:.1f}) "
            f"last={pose_text} error={last_error:.1f}mm. "
            "Next motion segment was blocked for safety."
        )

    def wait_for_horizontal_waypoint(
        label: str,
        x: float,
        y: float,
        commanded_z: float,
        min_safe_z: float,
        allow_abort: bool = False,
    ):
        """Confirm horizontal arrival by XY error while independently enforcing safe Z."""
        if arm is None:
            raise RuntimeError("ARM2 is not connected")
        deadline = time.time() + max(0.5, float(args.move_timeout))
        target = np.asarray(
            [float(x), float(y), float(commanded_z)], dtype=np.float64
        )
        last_pose = None
        last_xy_error = float("inf")
        last_z_error = float("inf")
        last_3d_error = float("inf")
        while time.time() < deadline:
            if allow_abort and _probe_abort_requested():
                raise KeyboardInterrupt(f"operator abort during {label}")
            time.sleep(max(0.05, float(args.move_poll_sec)))
            feedback = arm.feedback(args.feedback_timeout)
            if feedback is None or not all(k in feedback for k in ("x", "y", "z")):
                continue
            actual = np.asarray(
                [float(feedback["x"]), float(feedback["y"]), float(feedback["z"])],
                dtype=np.float64,
            )
            last_pose = actual
            last_xy_error = float(np.linalg.norm(actual[:2] - target[:2]))
            last_z_error = abs(float(actual[2]) - float(commanded_z))
            last_3d_error = float(np.linalg.norm(actual - target))
            safe_z_ok = float(actual[2]) >= float(min_safe_z)
            print(
                f"[{label}] feedback=({actual[0]:.1f},{actual[1]:.1f},{actual[2]:.1f}) "
                f"xy_error={last_xy_error:.1f}mm z_error={last_z_error:.1f}mm "
                f"3d_error={last_3d_error:.1f}mm min_safe_z={float(min_safe_z):.1f}mm "
                f"safe_z={'YES' if safe_z_ok else 'NO'}"
            )
            if last_xy_error <= float(args.move_tolerance_mm) and safe_z_ok:
                time.sleep(max(0.0, float(args.move_wait)))
                return actual
        pose_text = "unavailable" if last_pose is None else (
            f"({last_pose[0]:.1f},{last_pose[1]:.1f},{last_pose[2]:.1f})"
        )
        raise RuntimeError(
            f"{label} horizontal waypoint timeout: target_xy=({x:.1f},{y:.1f}) "
            f"commanded_z={commanded_z:.1f} min_safe_z={min_safe_z:.1f} "
            f"last={pose_text} xy_error={last_xy_error:.1f}mm "
            f"z_error={last_z_error:.1f}mm 3d_error={last_3d_error:.1f}mm. "
            "Next motion segment was blocked for safety."
        )

    def wait_for_direct_xy_arrival(
        label: str,
        x: float,
        y: float,
        commanded_z: float,
        allow_abort: bool = False,
    ):
        """Wait until the final board-center transfer is both near target and settled.

        Z tracking remains diagnostic-only during the long direct transfer.  Unlike
        the recovery hop, however, the final release location matters.  Therefore
        V16 does not open the gripper on the first in-tolerance sample; it waits
        until the XY motion has settled while remaining within the normal center
        tolerance.
        """
        if arm is None:
            raise RuntimeError("ARM2 is not connected")
        deadline = time.time() + max(0.5, float(args.move_timeout))
        target_xy = np.asarray([float(x), float(y)], dtype=np.float64)
        last_pose = None
        last_xy_error = float("inf")
        last_z_error = float("inf")
        recent_xy = []
        settle_polls = 3
        settle_span_mm = 2.0
        while time.time() < deadline:
            if allow_abort and _probe_abort_requested():
                raise KeyboardInterrupt(f"operator abort during {label}")
            time.sleep(max(0.05, float(args.move_poll_sec)))
            feedback = arm.feedback(args.feedback_timeout)
            if feedback is None or not all(k in feedback for k in ("x", "y", "z")):
                continue
            actual = np.asarray(
                [float(feedback["x"]), float(feedback["y"]), float(feedback["z"])],
                dtype=np.float64,
            )
            last_pose = actual
            last_xy_error = float(np.linalg.norm(actual[:2] - target_xy))
            last_z_error = abs(float(actual[2]) - float(commanded_z))
            recent_xy.append(actual[:2].copy())
            if len(recent_xy) > settle_polls:
                recent_xy = recent_xy[-settle_polls:]
            settled = False
            xy_span = float("inf")
            if len(recent_xy) == settle_polls:
                anchor = recent_xy[0]
                xy_span = max(float(np.linalg.norm(p - anchor)) for p in recent_xy[1:])
                settled = xy_span <= settle_span_mm
            in_tolerance = last_xy_error <= float(args.move_tolerance_mm)
            print(
                f"[{label}] feedback=({actual[0]:.1f},{actual[1]:.1f},{actual[2]:.1f}) "
                f"xy_error={last_xy_error:.1f}mm z_error={last_z_error:.1f}mm "
                f"in_center_tol={'YES' if in_tolerance else 'NO'} "
                f"settled={'YES' if settled else 'NO'} settle_span={xy_span:.1f}mm"
            )
            if in_tolerance and settled:
                time.sleep(max(0.0, float(args.move_wait)))
                return actual
        pose_text = "unavailable" if last_pose is None else (
            f"({last_pose[0]:.1f},{last_pose[1]:.1f},{last_pose[2]:.1f})"
        )
        raise RuntimeError(
            f"{label} direct XY timeout before settled center arrival: "
            f"target_xy=({x:.1f},{y:.1f}) commanded_z={commanded_z:.1f} "
            f"last={pose_text} xy_error={last_xy_error:.1f}mm z_error={last_z_error:.1f}mm"
        )

    def wait_for_vertical_waypoint_adaptive(label: str, x: float, y: float, z: float):
        """Wait for a lift waypoint and return a stall result instead of hanging."""
        if arm is None:
            raise RuntimeError("ARM2 is not connected")
        deadline = time.time() + max(0.5, float(args.move_timeout))
        target = np.asarray([float(x), float(y), float(z)], dtype=np.float64)
        recent_z = []
        last_pose = None
        last_error = float("inf")
        while time.time() < deadline:
            if _probe_abort_requested():
                raise KeyboardInterrupt(f"operator abort during {label}")
            time.sleep(max(0.05, float(args.move_poll_sec)))
            feedback = arm.feedback(args.feedback_timeout)
            if feedback is None or not all(k in feedback for k in ("x", "y", "z")):
                continue
            actual = np.asarray(
                [float(feedback["x"]), float(feedback["y"]), float(feedback["z"])],
                dtype=np.float64,
            )
            last_pose = actual
            last_error = float(np.linalg.norm(actual - target))
            z_error = abs(float(actual[2]) - float(z))
            print(
                f"[{label}] feedback=({actual[0]:.1f},{actual[1]:.1f},{actual[2]:.1f}) "
                f"error={last_error:.1f}mm z_error={z_error:.1f}mm"
            )
            if z_error <= float(args.vertical_waypoint_tolerance_mm):
                time.sleep(max(0.0, float(args.move_wait)))
                return {"reached": True, "stalled": False, "pose": actual, "error": last_error}
            recent_z.append(float(actual[2]))
            poll_count = max(3, int(args.vertical_stall_polls))
            if len(recent_z) > poll_count:
                recent_z = recent_z[-poll_count:]
            if len(recent_z) == poll_count:
                span = max(recent_z) - min(recent_z)
                if span <= float(args.vertical_stall_span_mm):
                    print(
                        f"[{label}-SATURATED] Z span={span:.2f}mm across {poll_count} polls; "
                        f"last_z={actual[2]:.2f} target_z={z:.2f}"
                    )
                    return {"reached": False, "stalled": True, "pose": actual, "error": last_error}
        if last_pose is None:
            raise RuntimeError(f"{label} feedback unavailable during adaptive lift")
        print(
            f"[{label}-TIMEOUT] last=({last_pose[0]:.1f},{last_pose[1]:.1f},{last_pose[2]:.1f}) "
            f"target_z={z:.1f} error={last_error:.1f}mm"
        )
        return {"reached": False, "stalled": False, "pose": last_pose, "error": last_error}

    def move_xy_segmented(
        label: str,
        start_pose,
        target_x: float,
        target_y: float,
        z: float,
        speed: float,
        close_angle: float,
        min_safe_z: float,
    ):
        start = np.asarray(start_pose, dtype=np.float64)
        dx = float(target_x) - float(start[0])
        dy = float(target_y) - float(start[1])
        distance = math.hypot(dx, dy)
        count = max(1, int(math.ceil(distance / max(10.0, float(args.inner_xy_step_mm)))))
        last = start.copy()
        for index in range(1, count + 1):
            if _probe_abort_requested():
                raise KeyboardInterrupt(f"operator abort during {label}")
            ratio = index / count
            segment_x = float(start[0]) + ratio * dx
            segment_y = float(start[1]) + ratio * dy
            print(
                f"[{label} {index}/{count}] constant-Z target=({segment_x:.2f},{segment_y:.2f},{z:.2f})"
            )
            arm.move_goal(
                args.move_command, segment_x, segment_y, z, close_angle, speed
            )
            last = wait_for_horizontal_waypoint(
                f"{label}:{index}/{count}",
                segment_x,
                segment_y,
                z,
                min_safe_z,
                allow_abort=True,
            )
        return np.asarray(last, dtype=np.float64)

    def wait_for_recovery_hop_progress(
        label: str,
        start_pose,
        board_center_xy,
        commanded_z: float,
        allow_abort: bool = False,
    ):
        """Accept a short inward recovery hop by *actual centerward progress*.

        This hop is not a precision waypoint. Its purpose is only to change the
        arm posture enough that another vertical lift becomes reachable.  V15
        incorrectly required the 35 mm hop target to be reached within 10 mm and
        could wait forever even after the arm had already moved meaningfully
        toward the board.

        Legacy recovery helper retained for compatibility; V18 automatic placement does not call it.  It polls T:105 until the
        XY pose settles, then accepts the actual pose whenever the distance to
        board center has decreased.  The following relift always starts from
        that actual feedback XY.
        """
        if arm is None:
            raise RuntimeError("ARM2 is not connected")
        start = np.asarray(start_pose, dtype=np.float64)
        center = np.asarray(board_center_xy, dtype=np.float64).reshape(2)
        start_distance = float(np.linalg.norm(center - start[:2]))
        deadline = time.time() + max(0.5, float(args.move_timeout))
        recent_xy = []
        last_pose = None
        last_progress = float("-inf")
        settle_polls = 3
        settle_span_mm = 1.5
        min_positive_progress_mm = 1.0

        while time.time() < deadline:
            if allow_abort and _probe_abort_requested():
                raise KeyboardInterrupt(f"operator abort during {label}")
            time.sleep(max(0.05, float(args.move_poll_sec)))
            feedback = arm.feedback(args.feedback_timeout)
            if feedback is None or not all(k in feedback for k in ("x", "y", "z")):
                continue
            actual = np.asarray(
                [float(feedback["x"]), float(feedback["y"]), float(feedback["z"])],
                dtype=np.float64,
            )
            last_pose = actual
            current_distance = float(np.linalg.norm(center - actual[:2]))
            last_progress = float(start_distance - current_distance)
            z_error = abs(float(actual[2]) - float(commanded_z))

            recent_xy.append(actual[:2].copy())
            if len(recent_xy) > settle_polls:
                recent_xy = recent_xy[-settle_polls:]
            settled = False
            xy_span = float("inf")
            if len(recent_xy) == settle_polls:
                anchor = recent_xy[0]
                xy_span = max(float(np.linalg.norm(p - anchor)) for p in recent_xy[1:])
                settled = xy_span <= settle_span_mm

            print(
                f"[{label}] feedback=({actual[0]:.1f},{actual[1]:.1f},{actual[2]:.1f}) "
                f"centerward_progress={last_progress:.1f}mm "
                f"remaining_to_center={current_distance:.1f}mm "
                f"z_error={z_error:.1f}mm "
                f"settled={'YES' if settled else 'NO'} "
                f"settle_span={xy_span:.1f}mm"
            )

            if settled:
                if last_progress > min_positive_progress_mm:
                    print(
                        f"[{label}-ACCEPT] precision waypoint is not required; "
                        f"actual centerward progress={last_progress:.1f}mm. "
                        "Start re-lift from this actual XY."
                    )
                    time.sleep(max(0.0, float(args.move_wait)))
                    return actual
                raise RuntimeError(
                    f"{label} settled without meaningful motion toward board center: "
                    f"progress={last_progress:.1f}mm"
                )

        if last_pose is not None and last_progress > min_positive_progress_mm:
            print(
                f"[{label}-TIMEOUT-ACCEPT] feedback did not satisfy the settle detector, "
                f"but actual centerward progress={last_progress:.1f}mm; "
                "use the latest actual pose and start re-lift."
            )
            return last_pose
        pose_text = "unavailable" if last_pose is None else (
            f"({last_pose[0]:.1f},{last_pose[1]:.1f},{last_pose[2]:.1f})"
        )
        raise RuntimeError(
            f"{label} recovery hop produced no usable centerward progress: "
            f"last={pose_text} progress={last_progress:.1f}mm"
        )

    def move_vertical_adaptive(label: str, start_pose, target_z: float, x: float, y: float, close_angle: float):
        """V18 fast lift: one direct vertical T:104, then judge reached vs physical saturation.

        V16 split the lift into 25 mm waypoints.  That made the motion unnecessarily
        slow and inserted a feedback wait after every small segment.  V18 keeps the
        requested XY fixed, sends one vertical goal at test_lift_speed, and lets the
        existing adaptive feedback monitor terminate on either target reach or a real
        Z plateau.
        """
        start = np.asarray(start_pose, dtype=np.float64)
        target_z = float(target_z)
        print(
            f"[{label}-DIRECT] from_z={float(start[2]):.3f} target_z={target_z:.3f} "
            f"xy=({float(x):.3f},{float(y):.3f}) speed={float(args.test_lift_speed):.2f}"
        )
        arm.move_goal(
            args.move_command, float(x), float(y), target_z, close_angle, args.test_lift_speed
        )
        result = wait_for_vertical_waypoint_adaptive(label, float(x), float(y), target_z)
        last = np.asarray(result["pose"], dtype=np.float64)
        return bool(result["reached"]), last, result

    def move_vertical_strict_segmented(label: str, start_pose, target_z: float, x: float, y: float, speed: float, close_angle: float):
        start = np.asarray(start_pose, dtype=np.float64)
        direction = 1.0 if float(target_z) > float(start[2]) else -1.0
        step_mm = max(5.0, float(args.board_descent_step_mm))
        current_z = float(start[2])
        targets = []
        while direction * (float(target_z) - current_z) > step_mm + 1e-9:
            current_z += direction * step_mm
            targets.append(current_z)
        targets.append(float(target_z))
        last = start.copy()
        for index, segment_z in enumerate(targets, 1):
            if _probe_abort_requested():
                raise KeyboardInterrupt(f"operator abort during {label}")
            print(f"[{label} {index}/{len(targets)}] vertical target_z={segment_z:.3f}")
            arm.move_goal(args.move_command, x, y, segment_z, close_angle, speed)
            last = wait_for_waypoint(
                f"{label}:{index}/{len(targets)}", x, y, segment_z, allow_abort=True
            )
        return np.asarray(last, dtype=np.float64)

    def execute_hover_plan():
        nonlocal hover_plan, hover_ready, descent_plan, grasp_lift_plan
        if hover_plan is None:
            print("[HOVER] press H first to prepare and review the hover plan")
            return False
        if arm is None:
            print("[HOVER-BLOCKED] --send required")
            return False

        plan = dict(hover_plan)
        hover_plan = None
        try:
            arm.torque_on()

            # Segment 1: vertical raise only. Never start horizontal travel at
            # the mean rim height because that could scrape the board/basket.
            standby_safe_route = bool(plan.get("standby_route_required", False))
            hover_raise_label = "HOVER-1/4" if standby_safe_route else "HOVER-1/3"
            if plan["start_z"] < plan["safe_z"] - 1.0:
                print(f"[{hover_raise_label}] vertical raise to safe transit Z")
                arm.move_goal(
                    args.move_command,
                    plan["start_x"],
                    plan["start_y"],
                    plan["safe_z"],
                    plan["tool_t"],
                    args.speed,
                )
                wait_for_waypoint(
                    hover_raise_label,
                    plan["start_x"],
                    plan["start_y"],
                    plan["safe_z"],
                )
            else:
                print(f"[{hover_raise_label}] current Z already at/above safe transit Z; raise skipped")

            hover_horizontal_min_safe_z = (
                float(plan["rim_z_max"]) + float(args.transit_clearance_mm)
            )

            if bool(plan.get("standby_route_required", False)):
                # V21 mandatory safety segment.  Never send the basket target
                # directly from the A150 standby posture.  First unfold/move
                # ARM2 to the calibrated board-inner point at high Z.  The next
                # basket command is sent ONLY after this waypoint is confirmed.
                print("[HOVER-2/4] REQUIRED board-safe waypoint before basket transit")
                arm.move_goal(
                    args.move_command,
                    plan["board_safe_x"],
                    plan["board_safe_y"],
                    plan["board_safe_z"],
                    plan["tool_t"],
                    args.speed,
                )
                wait_for_waypoint(
                    "HOVER-2/4-BOARD-SAFE",
                    plan["board_safe_x"],
                    plan["board_safe_y"],
                    plan["board_safe_z"],
                )

                print("[HOVER-3/4] board-safe confirmed; high-Z horizontal move to temporary grasp XY")
                arm.move_goal(
                    args.move_command,
                    plan["target_x"],
                    plan["target_y"],
                    plan["safe_z"],
                    plan["tool_t"],
                    args.speed,
                )
                wait_for_horizontal_waypoint(
                    "HOVER-3/4-BASKET-TRANSIT",
                    plan["target_x"],
                    plan["target_y"],
                    plan["safe_z"],
                    hover_horizontal_min_safe_z,
                )

                # Descend only after confirmed basket XY arrival.
                print("[HOVER-4/4] vertical descent to final hover Z")
                arm.move_goal(
                    args.move_command,
                    plan["target_x"],
                    plan["target_y"],
                    plan["target_z"],
                    plan["tool_t"],
                    args.descent_speed,
                )
                wait_for_waypoint(
                    "HOVER-4/4",
                    plan["target_x"],
                    plan["target_y"],
                    plan["target_z"],
                )
            else:
                # Original first-run/non-standby route.
                print("[HOVER-2/3] high-Z horizontal move to temporary grasp XY")
                arm.move_goal(
                    args.move_command,
                    plan["target_x"],
                    plan["target_y"],
                    plan["safe_z"],
                    plan["tool_t"],
                    args.speed,
                )
                wait_for_horizontal_waypoint(
                    "HOVER-2/3",
                    plan["target_x"],
                    plan["target_y"],
                    plan["safe_z"],
                    hover_horizontal_min_safe_z,
                )

                # Segment 3: descend only after confirmed XY arrival. No gripper command follows.
                print("[HOVER-3/3] vertical descent to final hover Z")
                arm.move_goal(
                    args.move_command,
                    plan["target_x"],
                    plan["target_y"],
                    plan["target_z"],
                    plan["tool_t"],
                    args.descent_speed,
                )
                wait_for_waypoint(
                    "HOVER-3/3",
                    plan["target_x"],
                    plan["target_y"],
                    plan["target_z"],
                )

            final_feedback = arm.feedback(args.feedback_timeout)
            print("\n========== ARM2 HOVER COMPLETE ==========")
            print(
                f"commanded=({plan['target_x']:.2f},{plan['target_y']:.2f},"
                f"{plan['target_z']:.2f})"
            )
            if final_feedback is not None and all(k in final_feedback for k in ("x", "y", "z")):
                actual = np.asarray(
                    [float(final_feedback["x"]), float(final_feedback["y"]), float(final_feedback["z"])],
                    dtype=np.float64,
                )
                commanded = np.asarray(
                    [plan["target_x"], plan["target_y"], plan["target_z"]],
                    dtype=np.float64,
                )
                error = float(np.linalg.norm(actual - commanded))
                print(
                    f"feedback=({actual[0]:.2f},{actual[1]:.2f},{actual[2]:.2f}) "
                    f"3D_error={error:.2f}mm"
                )
            else:
                print("final feedback unavailable; visually verify the hover position")
            print("Robot remains at hover. No clothing contact or lift has occurred yet.")
            print("Press D to prepare the monitored torque descent in this same serial session.")
            print("=========================================\n")
            hover_ready = True
            descent_plan = None
            grasp_lift_plan = None
            return True
        except Exception as exc:
            hover_ready = False
            print(f"[HOVER-ERROR] execution failed: {exc}")
            return False

    def prepare_descent_plan():
        nonlocal descent_plan
        if not hover_ready:
            print("[DESCENT-BLOCKED] complete H -> Enter hover first in this same program")
            return False
        if arm is None:
            print("[DESCENT-BLOCKED] run with --send")
            return False
        if not calib.geometry or len(calib.points) != 5:
            print("[DESCENT-BLOCKED] valid calibration is not loaded")
            return False
        fb = arm.feedback_retry(args.feedback_timeout, attempts=4, retry_delay=0.4)
        if fb is None:
            print("[DESCENT-ERROR] T:105 feedback failed")
            return False
        target_xy = calib.geometry.get("temporary_grasp_arm2_xy_direct")
        if not isinstance(target_xy, (list, tuple)) or len(target_xy) != 2:
            print("[DESCENT-ERROR] temporary_grasp_arm2_xy_direct missing")
            return False
        tx, ty = float(target_xy[0]), float(target_xy[1])
        cx, cy, cz = float(fb["x"]), float(fb["y"]), float(fb["z"])
        tool_t = float(fb.get("t", args.tool_angle_fallback))
        xy_error = math.hypot(cx - tx, cy - ty)
        z_values = [float(p["arm2_xyz"][2]) for p in calib.points]
        rim_mean_z = float(np.mean(z_values))
        rim_z_max = float(np.max(z_values))
        min_safe_z = float(args.basket_floor_z) + float(args.floor_clearance_mm)
        grip_angle = _gripper_angle(
            args.gripper_open_percent, args.grip_fully_open, args.grip_fully_closed
        )
        if min_safe_z >= rim_mean_z:
            print("[DESCENT-ERROR] floor safety limit must be below rim mean Z")
            return False
        print("\n========== TORQUE DESCENT PLAN ==========")
        print(f"current=({cx:.2f},{cy:.2f},{cz:.2f}) tool_t={tool_t:.3f}")
        print(f"grasp_xy=({tx:.2f},{ty:.2f}) current_xy_error={xy_error:.2f}mm")
        print(f"rim_mean_z={rim_mean_z:.2f}mm rim_z_max={rim_z_max:.2f}mm")
        print(f"basket_floor_z={args.basket_floor_z:.2f}mm")
        print(f"absolute_min_safe_z=floor+{args.floor_clearance_mm:.1f}={min_safe_z:.2f}mm")
        print(f"gripper={args.gripper_open_percent:.1f}% open -> angle={grip_angle:.3f}rad")
        print(f"descent T:104 t will be LOCKED to the same gripper angle {grip_angle:.3f}rad")
        print(f"FAST monitored descent: {args.fast_step_mm:.1f}mm steps to rim mean Z")
        print(f"SLOW torque profile: {args.slow_step_mm:.1f}mm steps to safety limit")
        print("All B/S/E/H torque values will be printed and written to CSV.")
        print("After confirmed contact, open -> close -> adaptive lift -> direct board-center transit -> air release -> A150 standby runs automatically.")
        print("Floor-limit, stall, hard-stop, or operator-abort outcomes cannot arm the grasp/lift test.")
        print("ARM1 command: NONE.")
        print("Press Enter to execute, or U to cancel.")
        print("=========================================\n")
        if xy_error > float(args.xy_start_tolerance_mm):
            print(
                f"[DESCENT-BLOCKED] XY error {xy_error:.1f}mm exceeds "
                f"{args.xy_start_tolerance_mm:.1f}mm"
            )
            return False
        descent_plan = {
            "target_x": tx, "target_y": ty, "start_z": cz, "tool_t": tool_t,
            "rim_mean_z": rim_mean_z, "rim_z_max": rim_z_max,
            "min_safe_z": min_safe_z, "grip_angle": grip_angle,
        }
        return True

    def _probe_abort_requested():
        key = terminal.read_key()
        return key in (ord(' '), ord('x'), ord('q'), 27)

    def _read_probe_sample(phase: str, step_index: int, commanded_z: float):
        deadline = time.time() + max(0.5, float(args.probe_step_timeout))
        recent_z = []
        latest = None
        while time.time() < deadline:
            if _probe_abort_requested():
                raise KeyboardInterrupt("operator abort")
            time.sleep(max(0.05, float(args.probe_poll_sec)))
            fb = arm.feedback(args.feedback_timeout, quiet=True)
            if fb is None:
                continue
            latest = _sample_from_feedback(fb, phase, step_index, commanded_z)
            recent_z.append(latest.z)
            if len(recent_z) > int(args.probe_settle_samples):
                recent_z.pop(0)
            z_close = abs(latest.z - commanded_z) <= float(args.probe_z_arrival_tolerance_mm)
            z_stable = (
                len(recent_z) >= int(args.probe_settle_samples)
                and max(recent_z) - min(recent_z) <= float(args.probe_z_stable_span_mm)
            )
            if z_close or z_stable:
                time.sleep(max(0.0, float(args.probe_settle_wait_sec)))
                fb2 = arm.feedback(args.feedback_timeout, quiet=True)
                if fb2 is not None:
                    latest = _sample_from_feedback(fb2, phase, step_index, commanded_z)
                return latest
        if latest is None:
            raise RuntimeError(f"No T:105 feedback for {phase} step {step_index}")
        print(f"[{phase}:{step_index:02d}] settle timeout; using latest z={latest.z:.2f}")
        return latest

    def _collect_probe_baseline(commanded_z: float, label: str = "rim mean Z"):
        samples = []
        count = max(3, int(args.baseline_samples))
        print(f"[BASELINE] collecting {count} torque samples near {label}")
        for i in range(count):
            fb = arm.feedback(args.feedback_timeout, quiet=True)
            if fb is not None:
                samples.append(_sample_from_feedback(fb, "BASELINE", i, commanded_z))
            time.sleep(max(0.05, float(args.baseline_interval_sec)))
        if len(samples) < 3:
            raise RuntimeError("not enough torque samples for baseline")
        result = _median_torques(samples)
        print(
            f"[BASELINE] median B={result[0]:+.0f} S={result[1]:+.0f} "
            f"E={result[2]:+.0f} H={result[3]:+.0f}"
        )
        return result

    def execute_descent_probe(retry_cycle: Optional[int] = None, retry_attempt: int = 1):
        nonlocal descent_plan, hover_ready, grasp_lift_plan
        if descent_plan is None:
            print("[DESCENT] press D first to prepare and review the plan")
            return False
        if arm is None:
            print("[DESCENT-BLOCKED] --send required")
            return False
        plan = dict(descent_plan)
        descent_plan = None
        required_plan_keys = (
            "target_x", "target_y", "start_z", "tool_t",
            "rim_mean_z", "rim_z_max", "min_safe_z", "grip_angle",
        )
        missing_plan_keys = [key for key in required_plan_keys if key not in plan]
        if missing_plan_keys:
            print(
                "[DESCENT-ERROR] descent plan is missing required keys: "
                + ", ".join(missing_plan_keys)
            )
            grasp_lift_plan = None
            return False
        grasp_lift_plan = None
        csv_path = args.probe_csv or (
            "basket_torque_descent_" + datetime.now().strftime("%Y%m%d_%H%M%S") + ".csv"
        )
        auto_grasp_lift = False
        try:
            arm.torque_on()
            print(
                f"[GRIPPER] set {args.gripper_open_percent:.1f}% open "
                f"angle={plan['grip_angle']:.3f}"
            )
            arm.gripper_open(plan["grip_angle"], args.grip_spd, args.grip_acc)
            print(
                f"[GRIPPER] holding {args.gripper_open_percent:.1f}% open; "
                f"wait {args.gripper_settle_sec:.1f}s before descent"
            )
            time.sleep(max(0.0, float(args.gripper_settle_sec)))
            start_fb = arm.feedback_retry(args.feedback_timeout, attempts=3, retry_delay=0.4)
            if start_fb is None:
                raise RuntimeError("T:105 failed after gripper setup")
            current = _sample_from_feedback(start_fb, "START", 0, float(start_fb["z"]))
            gripper_baseline = _collect_probe_baseline(current.z, label="settled gripper pose")
            start_fb2 = arm.feedback(args.feedback_timeout, quiet=True)
            if start_fb2 is not None:
                current = _sample_from_feedback(start_fb2, "START", 0, float(start_fb2["z"]))
            fast_targets = _descending_targets(
                current.z, max(plan["rim_mean_z"], plan["min_safe_z"]), args.fast_step_mm
            )
            with open(csv_path, "w", newline="", encoding="utf-8") as fh:
                writer = csv.writer(fh)
                _write_probe_header(writer)
                previous = current
                previous_torque = gripper_baseline
                print("\n========== FAST MONITORED APPROACH ==========")
                print("[CONTACT-EVAL] torH is LOG-ONLY; fast emergency check uses shoulder/elbow only")
                for index, target_z in enumerate(fast_targets, 1):
                    target_z = max(float(target_z), plan["min_safe_z"])
                    if _probe_abort_requested():
                        raise KeyboardInterrupt("operator abort")
                    print(f"[FAST:{index:02d}] vertical Z={target_z:.2f}")
                    arm.move_goal(
                        args.move_command, plan["target_x"], plan["target_y"],
                        target_z, plan["grip_angle"], args.fast_speed,
                    )
                    sample = _read_probe_sample("FAST", index, target_z)
                    delta_base, delta_step = _print_probe_sample(
                        sample, gripper_baseline, previous_torque
                    )
                    _write_probe_sample(writer, sample, delta_base, delta_step)
                    fh.flush()
                    fast_se_axis = max(abs(delta_step[1]), abs(delta_step[2]))
                    if fast_se_axis >= float(args.fast_hard_se_axis_delta):
                        print(
                            f"[HARD-STOP] shoulder/elbow jump during fast approach: "
                            f"{fast_se_axis:.1f}"
                        )
                        return False
                    previous, previous_torque = sample, sample.torques

                baseline = _collect_probe_baseline(plan["rim_mean_z"], label="rim mean Z")
                previous_torque = baseline
                fb = arm.feedback(args.feedback_timeout, quiet=True)
                if fb is not None:
                    previous = _sample_from_feedback(fb, "BASELINE", 0, plan["rim_mean_z"])
                slow_start = min(previous.z, plan["rim_mean_z"])
                slow_targets = _descending_targets(
                    slow_start, plan["min_safe_z"], args.slow_step_mm
                )
                print("\n========== SLOW TORQUE DESCENT ==========")
                print(f"[SLOW] {slow_start:.2f} -> hard limit {plan['min_safe_z']:.2f}")
                print("[CONTACT-EVAL] primary axis: shoulder torque decrease from rim baseline")
                print("[CONTACT-EVAL] confirm when shoulder_load>=40 AND (abs(elbow_change)>=20 OR Z_lag>=2.5mm)")
                print("[CONTACT-EVAL] condition must persist for 2 consecutive steps; B/H are log-only")
                torque_count = 0
                stall_count = 0
                contact_confirmed = False
                stop_reason = "absolute floor safety limit reached"
                for index, requested_z in enumerate(slow_targets, 1):
                    target_z = max(float(requested_z), plan["min_safe_z"])
                    if _probe_abort_requested():
                        stop_reason = "operator abort"
                        break
                    before_z = previous.z
                    print(f"[SLOW:{index:02d}] vertical Z={target_z:.2f}")
                    arm.move_goal(
                        args.move_command, plan["target_x"], plan["target_y"],
                        target_z, plan["grip_angle"], args.slow_speed,
                    )
                    sample = _read_probe_sample("SLOW", index, target_z)
                    delta_base, delta_step = _print_probe_sample(sample, baseline, previous_torque)
                    _write_probe_sample(writer, sample, delta_base, delta_step)
                    fh.flush()
                    # Robust contact detection derived from all measured trials.
                    # Shoulder torque decrease is the stable primary signal.
                    # Elbow response can change sign with garment geometry, so its magnitude is used.
                    # Positive Z lag is independent mechanical-resistance evidence.
                    # B is posture-dominated and H reacts to gripper commands, so both remain log-only.
                    shoulder_load = float(baseline[1]) - float(sample.torques[1])
                    elbow_change = float(sample.torques[2]) - float(baseline[2])
                    elbow_change_abs = abs(elbow_change)
                    z_lag = float(sample.z) - float(target_z)
                    shoulder_candidate = (
                        shoulder_load >= float(args.contact_shoulder_delta)
                    )
                    secondary_candidate = (
                        elbow_change_abs >= float(args.contact_elbow_delta)
                        or z_lag >= float(args.contact_z_lag_mm)
                    )
                    torque_candidate = shoulder_candidate and secondary_candidate
                    torque_count = torque_count + 1 if torque_candidate else 0
                    commanded_drop = max(0.0, before_z - target_z)
                    actual_drop = max(0.0, before_z - sample.z)
                    stall_candidate = (
                        commanded_drop >= float(args.stall_min_command_mm)
                        and actual_drop <= float(args.stall_max_actual_mm)
                    )
                    stall_count = stall_count + 1 if stall_candidate else 0
                    if torque_candidate:
                        source = (
                            "ELBOW"
                            if elbow_change_abs >= float(args.contact_elbow_delta)
                            else "Z-LAG"
                        )
                        print(
                            f"[CONTACT-CANDIDATE] shoulder_load={shoulder_load:.1f} "
                            f"elbow_change={elbow_change:+.1f} abs={elbow_change_abs:.1f} "
                            f"z_lag={z_lag:.2f}mm secondary={source} "
                            f"confirm={torque_count}/{args.contact_confirm_steps}"
                        )
                    elif shoulder_candidate:
                        print(
                            f"[CONTACT-AUX] shoulder_load={shoulder_load:.1f} reached, "
                            f"but elbow_abs={elbow_change_abs:.1f} and z_lag={z_lag:.2f}mm "
                            f"are below secondary thresholds"
                        )
                    elif z_lag >= float(args.contact_z_lag_mm):
                        print(
                            f"[CONTACT-AUX] Z lag={z_lag:.2f}mm >= "
                            f"{args.contact_z_lag_mm:.2f}mm, but shoulder_load={shoulder_load:.1f} "
                            f"is below {args.contact_shoulder_delta:.1f}"
                        )
                    if stall_candidate:
                        print(
                            f"[SAFETY-CANDIDATE] Z stall cmd_drop={commanded_drop:.2f} "
                            f"actual_drop={actual_drop:.2f} "
                            f"confirm={stall_count}/{args.stall_confirm_steps}"
                        )
                    hard_contact_load = max(shoulder_load, elbow_change_abs)
                    if hard_contact_load >= float(args.hard_axis_delta):
                        stop_reason = f"hard shoulder/elbow torque stop ({hard_contact_load:.1f})"
                        previous = sample
                        break
                    if not args.no_auto_contact_stop and torque_count >= int(args.contact_confirm_steps):
                        contact_confirmed = True
                        stop_reason = "shoulder-primary clothing contact confirmed"
                        previous = sample
                        break
                    if not args.no_auto_contact_stop and stall_count >= int(args.stall_confirm_steps):
                        stop_reason = "Z-motion stall safety stop"
                        previous = sample
                        break
                    previous, previous_torque = sample, sample.torques
                    if target_z <= plan["min_safe_z"] + 1e-9:
                        stop_reason = "absolute floor safety limit reached"
                        break
                print("\n========== DESCENT STOPPED ==========")
                print(f"reason={stop_reason}")
                print(
                    f"final=({previous.x:.2f},{previous.y:.2f},{previous.z:.2f}) "
                    f"hard_limit={plan['min_safe_z']:.2f}"
                )
                print(f"csv={csv_path}")
                if contact_confirmed:
                    wider_angle = _gripper_angle(
                        args.post_contact_open_percent,
                        args.grip_fully_open,
                        args.grip_fully_closed,
                    )
                    pickup_lift_z = float(args.pickup_lift_z)
                    release_z = float(args.board_release_z)
                    for name, value in (
                        ("pickup_lift_z", pickup_lift_z),
                        ("board_release_z", release_z),
                        ("board_center_x", float(board_center_x)),
                        ("board_center_y", float(board_center_y)),
                    ):
                        if not np.isfinite(value):
                            raise RuntimeError(f"{name} is not finite")
                    if pickup_lift_z > float(args.z_max) or pickup_lift_z < float(args.z_min):
                        raise RuntimeError(
                            f"pickup lift Z {pickup_lift_z:.2f} is outside [{args.z_min:.2f},{args.z_max:.2f}]"
                        )
                    if release_z > pickup_lift_z - 20.0:
                        raise RuntimeError(
                            f"board release Z {release_z:.2f} must be at least 20mm below pickup lift Z {pickup_lift_z:.2f}"
                        )
                    _require_in_range("board_center_x", float(board_center_x), args.x_min, args.x_max)
                    _require_in_range("board_center_y", float(board_center_y), args.y_min, args.y_max)
                    _require_in_range("board_release_z", release_z, args.z_min, args.z_max)
                    grasp_lift_plan = {
                        "target_x": float(plan["target_x"]),
                        "target_y": float(plan["target_y"]),
                        "contact_x": float(previous.x),
                        "contact_y": float(previous.y),
                        "contact_z": float(previous.z),
                        "descent_grip_angle": float(plan["grip_angle"]),
                        "wider_open_angle": float(wider_angle),
                        "close_angle": float(args.grasp_close_angle),
                        "pickup_lift_z": pickup_lift_z,
                        "config_safe_hover_z": float(config_safe_hover_z),
                        "inner_x": float(board_inner_x),
                        "inner_y": float(board_inner_y),
                        "rim_mean_z": float(plan["rim_mean_z"]),
                        "rim_z_max": float(plan["rim_z_max"]),
                        "min_safe_z": float(plan["min_safe_z"]),
                        "board_center_x": float(board_center_x),
                        "board_center_y": float(board_center_y),
                        "board_release_z": release_z,
                        "release_open_angle": float(args.release_open_angle),
                        "contact_reason": str(stop_reason),
                    }
                    auto_grasp_lift = True
                    print("gripper_close=AUTO-PENDING | placement=AUTO-PENDING | release=AUTO-PENDING")
                    print(
                        f"[AUTO-PLACE] open to {args.post_contact_open_percent:.1f}% "
                        f"(angle={wider_angle:.3f}) -> close angle={args.grasp_close_angle:.3f} "
                        f"-> adaptive lift preferred_Z={pickup_lift_z:.3f} "
                        f"-> one-shot ARM2 center=({board_center_x:.3f},{board_center_y:.3f}) "
                        f"-> air release angle={args.release_open_angle:.3f} "
                        f"-> A150 standby=({args.standby_roarm_x:.3f},{args.standby_roarm_y:.3f},{args.standby_roarm_z:.3f})"
                    )
                    print("No additional Enter approval is required.")
                else:
                    grasp_lift_plan = None
                    print("gripper_close=BLOCKED | lift=BLOCKED")
                    print("[GRASP-LIFT-BLOCKED] descent did not end with confirmed clothing contact")
                if auto_grasp_lift:
                    print("ARM2 torque remains ON; automatic grasp/placement starts now")
                else:
                    print("ARM2 torque remains ON at the stopped pose")
                print("=====================================\n")
            hover_ready = False
            if auto_grasp_lift:
                return execute_grasp_lift_test(
                    cycle_index=retry_cycle,
                    attempt_index=max(1, int(retry_attempt)),
                )
            return True
        except KeyboardInterrupt:
            print("\n[OPERATOR-STOP] no further motion commands will be sent")
            print("[STATE] ARM2 torque remains ON; no close and no lift")
            hover_ready = False
            return False
        except Exception as exc:
            print(f"[DESCENT-ERROR] {exc}")
            hover_ready = False
            return False

    def _interruptible_wait(seconds: float, label: str):
        deadline = time.time() + max(0.0, float(seconds))
        while time.time() < deadline:
            if _probe_abort_requested():
                raise KeyboardInterrupt(f"operator abort during {label}")
            time.sleep(min(0.05, max(0.0, deadline - time.time())))

    def _read_gripper_feedback(attempts: int = 3):
        for _ in range(max(1, int(attempts))):
            fb = arm.feedback_retry(args.feedback_timeout, attempts=2, retry_delay=0.20)
            if fb is not None and "t" in fb:
                return fb
            time.sleep(0.15)
        return None

    def _set_and_verify_gripper_open(target_angle: float):
        tolerance = max(0.03, float(args.gripper_feedback_tolerance_rad))
        attempts = max(1, int(args.gripper_feedback_attempts))
        last_fb = None
        for attempt in range(1, attempts + 1):
            arm.gripper_open(target_angle, args.grip_spd, args.grip_acc)
            _interruptible_wait(
                args.post_contact_open_settle_sec,
                f"post-contact gripper opening attempt {attempt}",
            )
            last_fb = _read_gripper_feedback(attempts=2)
            if last_fb is not None and "t" in last_fb:
                error = abs(float(last_fb["t"]) - float(target_angle))
                print(
                    f"[GRIPPER-OPEN-CHECK] attempt={attempt}/{attempts} "
                    f"target={target_angle:.3f} feedback={float(last_fb['t']):.3f} "
                    f"error={error:.3f}rad"
                )
                if error <= tolerance:
                    return last_fb
            else:
                print(
                    f"[GRIPPER-OPEN-CHECK] attempt={attempt}/{attempts} "
                    "feedback unavailable"
                )
        raise RuntimeError(
            f"post-contact gripper did not reach the wider-open angle within {tolerance:.3f}rad"
        )

    def _diagnose_grasp_retention(cycle_index: int, attempt_index: int):
        """Post-lift retention diagnosis using only median(torS + torE).

        V24 returns an explicit status so board transfer can be gated.  Z, torB,
        torH, and gripper angle remain excluded from the decision exactly as validated in V23.
        """
        requested_samples = max(1, int(args.grasp_retention_samples))
        threshold = float(args.grasp_retention_threshold)
        interval_sec = max(0.0, float(args.grasp_retention_interval_sec))
        samples = []

        print("\n========== GRASP RETENTION DIAGNOSIS ==========")
        print(f"cycle={int(cycle_index)} attempt={int(attempt_index)}")
        print("parameter=torS+torE only")
        print(f"sample_count={requested_samples} threshold={threshold:.1f}")

        for sample_index in range(1, requested_samples + 1):
            if _probe_abort_requested():
                raise KeyboardInterrupt("operator abort during grasp-retention diagnosis")
            fb = arm.feedback_retry(
                args.feedback_timeout, attempts=3, retry_delay=0.15
            )
            if fb is None or "torS" not in fb or "torE" not in fb:
                print(
                    f"[GRASP-RETENTION-SAMPLE:{sample_index}/{requested_samples}] "
                    "feedback unavailable"
                )
            else:
                tor_s = float(fb["torS"])
                tor_e = float(fb["torE"])
                score = tor_s + tor_e
                samples.append(score)
                print(
                    f"[GRASP-RETENTION-SAMPLE:{sample_index}/{requested_samples}] "
                    f"torS={tor_s:+.0f} torE={tor_e:+.0f} score={score:.1f}"
                )
            if sample_index < requested_samples and interval_sec > 0.0:
                _interruptible_wait(interval_sec, "grasp-retention sampling")

        if len(samples) != requested_samples:
            print(f"sample_scores={samples}")
            print("median_score=UNAVAILABLE")
            print("result=GRASP DIAGNOSIS UNKNOWN")
            print(
                "[GRASP-RETENTION-UNKNOWN] insufficient valid torque samples; "
                "board-center transfer is blocked"
            )
            print("================================================\n")
            return {
                "status": "UNKNOWN",
                "median_score": None,
                "threshold": threshold,
                "sample_scores": samples,
            }

        median_score = float(statistics.median(samples))
        failed = median_score <= threshold
        status = "FAILURE" if failed else "SUCCESS"
        print(f"sample_scores={[round(value, 1) for value in samples]}")
        print(f"median_score={median_score:.1f}")
        print(f"threshold={threshold:.1f}")
        print(f"result={'GRASP FAILURE' if failed else 'GRASP SUCCESS'}")
        if failed:
            print(
                "[GRASP-FAILURE-DETECTED] post-lift median(torS+torE) is at or "
                "below the threshold; board-center transfer is blocked and auto-regrasp is requested"
            )
        else:
            print(
                "[GRASP-SUCCESS-DETECTED] post-lift median(torS+torE) is above "
                "the threshold; board-center transfer is authorized"
            )
        print("================================================\n")
        return {
            "status": status,
            "median_score": median_score,
            "threshold": threshold,
            "sample_scores": samples,
        }

    def execute_grasp_lift_test(cycle_index: Optional[int] = None, attempt_index: int = 1):
        nonlocal grasp_lift_plan, grasp_retention_cycle, descent_plan
        if grasp_lift_plan is None:
            print("[AUTO-PLACE] no armed plan; complete a confirmed-contact descent first")
            return False
        if arm is None:
            print("[AUTO-PLACE-BLOCKED] --send required")
            grasp_lift_plan = None
            return False

        plan = dict(grasp_lift_plan)
        grasp_lift_plan = None
        current_attempt = max(1, int(attempt_index))
        if cycle_index is None:
            grasp_retention_cycle += 1
            current_cycle = int(grasp_retention_cycle)
        else:
            current_cycle = int(cycle_index)
        try:
            arm.torque_on()
            standby_target = np.asarray(
                [
                    float(args.standby_roarm_x),
                    float(args.standby_roarm_y),
                    float(args.standby_roarm_z),
                ],
                dtype=np.float64,
            )
            if not np.all(np.isfinite(standby_target)) or not np.isfinite(float(args.standby_roarm_t)):
                raise RuntimeError("A150 standby target contains a non-finite value")
            if not (float(args.x_min) <= standby_target[0] <= float(args.x_max)):
                raise RuntimeError(f"standby X {standby_target[0]:.2f} is outside configured X range")
            # A150's directly taught ARM2 standby Y is -108.905mm, slightly beyond
            # this basket script's historical -100mm workspace guard.  Keep a
            # dedicated -150mm floor only for this known taught standby target.
            standby_y_min = min(float(args.y_min), -150.0)
            if not (standby_y_min <= standby_target[1] <= float(args.y_max)):
                raise RuntimeError(f"standby Y {standby_target[1]:.2f} is outside dedicated standby range")
            if not (float(args.z_min) <= standby_target[2] <= float(args.z_max)):
                raise RuntimeError(f"standby Z {standby_target[2]:.2f} is outside configured Z range")

            print("\n========== AUTOMATIC GRASP + RETENTION-GATED CENTER RELEASE ==========")
            print(f"cycle={current_cycle} attempt={current_attempt}")
            print(
                f"confirmed_contact=({plan['contact_x']:.2f},{plan['contact_y']:.2f},"
                f"{plan['contact_z']:.2f})"
            )
            print(
                f"1) widen gripper to {args.post_contact_open_percent:.1f}% "
                f"angle={plan['wider_open_angle']:.3f}"
            )
            print(f"2) close gripper to angle={plan['close_angle']:.3f}")
            print(
                f"3) FAST direct vertical lift toward preferred Z={plan['pickup_lift_z']:.3f} "
                f"at speed={float(args.test_lift_speed):.2f}; if the lift physically saturates, accept that actual Z immediately"
            )
            print(
                f"4) diagnose retention at the achieved Z; only SUCCESS sends ONE direct T:104 "
                f"to board center at speed={float(args.board_transit_speed):.2f}; FAILURE starts auto-regrasp"
            )
            print(f"5) open gripper to angle={plan['release_open_angle']:.3f} at board center")
            print(
                f"6) return ARM2 to A150 standby="
                f"({standby_target[0]:.3f},{standby_target[1]:.3f},{standby_target[2]:.3f}) "
                f"t={float(args.standby_roarm_t):.3f}"
            )
            print("ARM1 command: NONE | board-center descent: NONE | segmented board transit: NONE")
            print("=============================================================\n")

            # Hold the contact pose while opening wider, then close to grasp.
            open_fb = _set_and_verify_gripper_open(plan["wider_open_angle"])
            open_t = float(open_fb["t"])
            print(
                f"[AUTO-PLACE:1/6] wider opening VERIFIED at t={open_t:.3f}; "
                "no lift command has been sent yet"
            )

            arm.gripper_open(plan["close_angle"], args.grip_spd, args.grip_acc)
            print(
                f"[AUTO-PLACE:2/6] close command sent angle={plan['close_angle']:.3f}; "
                f"settle {args.grasp_close_settle_sec:.1f}s"
            )
            _interruptible_wait(args.grasp_close_settle_sec, "gripper closing")

            close_fb = arm.feedback_retry(
                args.feedback_timeout, attempts=4, retry_delay=0.3
            )
            if close_fb is None or not all(k in close_fb for k in ("x", "y", "z", "t")):
                raise RuntimeError("T:105 feedback failed after gripper close")
            close_t = float(close_fb["t"])
            close_motion = close_t - open_t
            print(
                f"[GRIPPER-CLOSE-CHECK] open_t={open_t:.3f} close_feedback={close_t:.3f} "
                f"motion={close_motion:.3f}rad torH={float(close_fb.get('torH', 0.0)):+.0f}"
            )
            if close_motion < float(args.gripper_min_close_motion_rad):
                raise RuntimeError(
                    f"gripper close motion too small ({close_motion:.3f}rad); placement blocked"
                )

            basket_x = float(close_fb["x"])
            basket_y = float(close_fb["y"])
            lift_start_z = float(close_fb["z"])
            preferred_transit_z = min(
                float(plan["pickup_lift_z"]),
                float(plan["config_safe_hover_z"]),
                float(args.z_max),
            )
            if preferred_transit_z <= lift_start_z + 1.0:
                raise RuntimeError(
                    f"preferred transit Z {preferred_transit_z:.2f} is not above grasp Z {lift_start_z:.2f}"
                )
            print(
                f"[DIRECT-TRANSIT-Z] preferred_z={preferred_transit_z:.3f} "
                f"policy=FAST_LIFT_THEN_HOLD_ACHIEVED_Z_AND_DIRECT_XY "
                f"lift_spd={float(args.test_lift_speed):.2f} "
                f"center_spd={float(args.board_transit_speed):.2f} "
                f"basket_rim_max={float(plan['rim_z_max']):.3f} "
                f"board_release_reference={float(plan['board_release_z']):.3f}"
            )

            initial_pose = np.asarray([basket_x, basket_y, lift_start_z], dtype=np.float64)
            basket_reached, lift_pose, lift_result = move_vertical_adaptive(
                "AUTO-PLACE:3/6/BASKET-LIFT",
                initial_pose,
                preferred_transit_z,
                basket_x,
                basket_y,
                plan["close_angle"],
            )
            transit_z = float(lift_pose[2])
            transit_start = np.asarray(lift_pose, dtype=np.float64)
            if basket_reached:
                print(
                    f"[BASKET-LIFT-REACHED] achieved_z={transit_z:.3f} "
                    "-> preferred height reached; start post-lift retention diagnosis"
                )
            else:
                if not bool(lift_result.get("stalled", False)):
                    raise RuntimeError(
                        "initial basket lift did not reach target and did not confirm physical saturation"
                    )
                # V18 policy: the first physical lift saturation is the end of the lift phase.
                # Do NOT demand another +120 mm, do NOT hop, and do NOT re-lift.  The
                # achieved feedback Z becomes the commanded Z for the one-shot direct XY
                # transfer only after retention SUCCESS.  Any unavoidable Z droop during that long move is diagnostic
                # only and never creates another recovery loop.
                print(
                    f"[BASKET-LIFT-SATURATION-ACCEPTED] achieved_z={transit_z:.3f} "
                    "-> no +120mm gate, no hop, no re-lift; start post-lift retention diagnosis"
                )

            # V24: diagnose only after the vertical lift has finished or physically
            # saturated.  The validated threshold is not applied to transient lift
            # samples.  Only SUCCESS authorizes the existing one-shot board transfer.
            diagnosis = _diagnose_grasp_retention(current_cycle, current_attempt)
            diagnosis_status = str(diagnosis.get("status", "UNKNOWN"))

            if diagnosis_status != "SUCCESS":
                # Never send a board-center command after FAILURE/UNKNOWN.  Release any
                # residual cloth directly over the basket while keeping the same serial
                # session and the same basket XY.
                print(
                    f"[BOARD-TRANSFER-BLOCKED] cycle={current_cycle} "
                    f"attempt={current_attempt} diagnosis={diagnosis_status}"
                )
                release_fb = _set_and_verify_gripper_open(float(args.grip_fully_open))
                if release_fb is None or not all(
                    key in release_fb for key in ("x", "y", "z", "t")
                ):
                    raise RuntimeError("post-failure basket release feedback unavailable")

                retry_xy_error = math.hypot(
                    float(release_fb["x"]) - float(plan["target_x"]),
                    float(release_fb["y"]) - float(plan["target_y"]),
                )
                print(
                    f"[AUTO-REGRASP-RELEASE] current=({float(release_fb['x']):.2f},"
                    f"{float(release_fb['y']):.2f},{float(release_fb['z']):.2f}) "
                    f"grasp_xy_error={retry_xy_error:.2f}mm"
                )

                if diagnosis_status == "UNKNOWN":
                    print(
                        "[AUTO-REGRASP-BLOCKED] diagnosis is UNKNOWN; ARM2 remains "
                        "open above the basket and no additional motion is sent"
                    )
                    return False

                max_retries = max(0, int(args.max_auto_regrasp_retries))
                retries_already_used = current_attempt - 1
                if retries_already_used >= max_retries:
                    print(
                        f"[AUTO-REGRASP-EXHAUSTED] cycle={current_cycle} "
                        f"attempts={current_attempt} max_additional_retries={max_retries}"
                    )
                    print(
                        "[STATE] gripper is open; ARM2 remains above the basket; "
                        "board-center transfer and standby return were not commanded"
                    )
                    return False

                if retry_xy_error > float(args.xy_start_tolerance_mm):
                    print(
                        f"[AUTO-REGRASP-BLOCKED] basket XY error {retry_xy_error:.1f}mm "
                        f"exceeds {float(args.xy_start_tolerance_mm):.1f}mm"
                    )
                    print(
                        "[STATE] gripper is open; ARM2 remains above the basket; "
                        "manual H/D recovery is required"
                    )
                    return False

                next_attempt = current_attempt + 1
                descent_plan = {
                    "target_x": float(plan["target_x"]),
                    "target_y": float(plan["target_y"]),
                    "start_z": float(release_fb["z"]),
                    "tool_t": float(release_fb.get("t", args.tool_angle_fallback)),
                    "rim_mean_z": float(plan["rim_mean_z"]),
                    "rim_z_max": float(plan["rim_z_max"]),
                    "min_safe_z": float(plan["min_safe_z"]),
                    "grip_angle": float(plan["descent_grip_angle"]),
                }
                print("\n========== AUTO REGRASP RETRY ==========")
                print(
                    f"cycle={current_cycle} next_attempt={next_attempt} "
                    f"max_additional_retries={max_retries}"
                )
                print(
                    "action=same monitored D->Enter descent -> confirmed contact -> "
                    "widen -> close -> vertical lift -> retention diagnosis"
                )
                print(
                    "board-center transfer remains blocked until diagnosis=SUCCESS"
                )
                print("========================================\n")
                return execute_descent_probe(
                    retry_cycle=current_cycle,
                    retry_attempt=next_attempt,
                )

            # D50-v1: preserve the complete V24 basket grasp/retention gate, but
            # replace its center air-drop with an ARM2-only D56-style swing and
            # exponential laydown.  XY is brought to the board center at the
            # already-achieved high Z before any descent starts.
            center_xy = np.asarray(
                [float(plan["board_center_x"]), float(plan["board_center_y"])],
                dtype=np.float64,
            )
            boardward = center_xy - transit_start[:2]
            boardward_distance = float(np.linalg.norm(boardward))
            if boardward_distance < 1e-6:
                raise RuntimeError("basket-to-board rotation vector is degenerate")
            boardward_unit = boardward / boardward_distance
            entry_extra = max(0.0, float(args.board_entry_extra_mm))
            entry_xy = center_xy + boardward_unit * entry_extra
            board_side_y_min = min(float(args.y_min), -180.0)
            _require_in_range("board_entry_x", float(entry_xy[0]), args.x_min, args.x_max)
            _require_in_range("board_entry_y", float(entry_xy[1]), board_side_y_min, args.y_max)
            direct_distance = float(np.linalg.norm(entry_xy - transit_start[:2]))
            print(
                f"[D50-SWING:4/6] HIGH-Z board rotation distance={direct_distance:.1f}mm "
                f"from=({transit_start[0]:.3f},{transit_start[1]:.3f},{transit_z:.3f}) "
                f"center=({center_xy[0]:.3f},{center_xy[1]:.3f}) "
                f"extra={entry_extra:.1f}mm "
                f"to=({entry_xy[0]:.3f},{entry_xy[1]:.3f},{transit_z:.3f})"
            )
            arm.move_goal(
                args.move_command, float(entry_xy[0]), float(entry_xy[1]),
                transit_z, plan["close_angle"], args.board_transit_speed,
            )
            center_high_pose = wait_for_direct_xy_arrival(
                "D50-BOARD-ROTATED-HIGH", float(entry_xy[0]), float(entry_xy[1]),
                transit_z, allow_abort=True,
            )

            # Initial V24 placement behavior: no shake, no backswing and no
            # exponential descent. Release immediately at the exact board
            # center while retaining the achieved high transit Z.
            board_center_pose = center_high_pose
            if _probe_abort_requested():
                raise KeyboardInterrupt("operator abort before direct center air release")
            print(
                f"[D50-DIRECT-AIR-DROP] board center reached at actual="
                f"({board_center_pose[0]:.3f},{board_center_pose[1]:.3f},"
                f"{board_center_pose[2]:.3f}); NO_SHAKE NO_SWING NO_DESCENT; "
                f"release angle={plan['release_open_angle']:.3f}"
            )
            arm.gripper_open(
                plan["release_open_angle"], args.grip_spd, args.grip_acc
            )
            _interruptible_wait(args.release_settle_sec, "direct center air release")
            release_fb = _read_gripper_feedback(attempts=3)
            if release_fb is None or "t" not in release_fb:
                raise RuntimeError("release gripper feedback unavailable")
            release_error = abs(float(release_fb["t"]) - plan["release_open_angle"])
            print(
                f"[GRIPPER-RELEASE-CHECK] target={plan['release_open_angle']:.3f} "
                f"feedback={float(release_fb['t']):.3f} error={release_error:.3f}rad"
            )
            if release_error > float(args.release_feedback_tolerance_rad):
                raise RuntimeError(
                    f"release opening error {release_error:.3f}rad exceeds "
                    f"{args.release_feedback_tolerance_rad:.3f}rad"
                )
            arm.gripper_open(
                float(args.standby_gripper_angle), args.grip_spd, args.grip_acc
            )
            _interruptible_wait(0.15, "standby gripper preparation")
            arm.move_goal(
                args.move_command,
                float(standby_target[0]), float(standby_target[1]),
                float(standby_target[2]), float(args.standby_roarm_t),
                float(args.standby_speed),
            )
            standby_pose = wait_for_waypoint(
                "ARM2-A150-STANDBY", float(standby_target[0]),
                float(standby_target[1]), float(standby_target[2]),
                allow_abort=True,
            )
            print(
                f"[D50-DIRECT-AIR-DROP-COMPLETE] releaseZ={board_center_pose[2]:.1f}mm "
                f"standby=({standby_pose[0]:.1f},{standby_pose[1]:.1f},"
                f"{standby_pose[2]:.1f})"
            )
            return True

            # Start from measured feedback, not the requested transit Z.  The
            # previous V1 requested 144 mm but actually arrived near 121 mm,
            # which created a discontinuity at the first swing waypoint.
            swing_start_z = float(center_high_pose[2])
            back_mm = max(0.0, float(args.swing_back_mm))
            forward_mm = max(1.0, float(args.swing_forward_mm))
            stream_hz = float(np.clip(float(args.swing_stream_hz), 10.0, 30.0))
            back_duration = max(0.40, float(args.swing_back_duration_s))
            forward_duration = max(0.70, float(args.swing_forward_duration_s))
            back_steps = max(8, int(math.ceil(back_duration * stream_hz)))
            forward_steps = max(14, int(math.ceil(forward_duration * stream_hz)))
            curve_rise = max(0.0, float(args.swing_curve_rise_mm))
            peak_z = min(float(args.z_max), swing_start_z + curve_rise)
            curve_rise = max(0.0, peak_z - swing_start_z)
            final_z = float(args.arm2_board_contact_z) + max(
                0.0, float(args.swing_release_clearance_mm)
            )
            if final_z >= swing_start_z - 10.0:
                raise RuntimeError(
                    f"swing final Z {final_z:.1f} must be at least 10mm below actual start Z {swing_start_z:.1f}"
                )
            up = np.asarray(board_up_arm2, dtype=np.float64)
            swing_origin_xy = np.asarray(center_high_pose[:2], dtype=np.float64)
            back_xy = swing_origin_xy - back_mm * up
            final_xy = back_xy + forward_mm * up
            # The basket script's historical y_min=-100 predates the taught
            # A150 standby at y=-108.905.  Use the same dedicated -150 mm floor
            # already accepted by V24 for that known ARM2 board-side branch;
            # do not weaken the basket approach/descent bounds globally.
            swing_y_min = board_side_y_min
            for label_xy, point_xy in (("swing_back", back_xy), ("swing_final", final_xy)):
                _require_in_range(f"{label_xy}_x", float(point_xy[0]), args.x_min, args.x_max)
                _require_in_range(f"{label_xy}_y", float(point_xy[1]), swing_y_min, args.y_max)
            _require_in_range("swing_final_z", final_z, args.z_min, args.z_max)
            print(
                f"[D50-SWING-WORKSPACE] dedicated board-side Y range="
                f"[{swing_y_min:.1f},{float(args.y_max):.1f}]mm; "
                f"backY={float(back_xy[1]):.1f} finalY={float(final_xy[1]):.1f}"
            )

            print(
                f"[D50-SWING-PLAN] ARM2_ONLY back={back_mm:.1f}mm toward ID1/ID3 "
                f"forward={forward_mm:.1f}mm toward ID0/ID2 "
                f"z={swing_start_z:.1f}->{peak_z:.1f}->{final_z:.1f}mm "
                f"T1041={stream_hz:.1f}Hz CLOSED"
            )

            def stream_waypoints(label, points, duration_s, time_fractions=None):
                if not points:
                    raise RuntimeError(f"{label}: empty waypoint stream")
                started = time.monotonic()
                count = len(points)
                for index, target in enumerate(points, start=1):
                    if _probe_abort_requested():
                        raise KeyboardInterrupt(f"operator abort during {label}")
                    fraction = (
                        float(time_fractions[index - 1])
                        if time_fractions is not None
                        else float(index) / float(count)
                    )
                    deadline = started + float(duration_s) * fraction
                    remaining = deadline - time.monotonic()
                    if remaining > 0.0:
                        time.sleep(remaining)
                    lag = time.monotonic() - deadline
                    if lag > 0.12:
                        raise RuntimeError(
                            f"{label} stream lag {lag * 1000.0:.1f}ms exceeds 120ms"
                        )
                    arm.move_direct(
                        float(target[0]), float(target[1]), float(target[2]),
                        float(plan["close_angle"]),
                    )
                    if index == 1 or index == count or index % 8 == 0:
                        print(
                            f"[{label}] {index:03d}/{count} command=T1041 "
                            f"lag={max(0.0, lag) * 1000.0:.1f}ms"
                        )

            # V7: command full-stroke linear reversals instead of a sinusoid
            # whose peak velocity was too high for the arm to follow.  Travel
            # center -> +A -> -A -> +A -> -A -> center at constant leg speed,
            # with a short endpoint hold to make every reversal physical.
            shake_mm = max(0.0, float(args.pre_swing_shake_mm))
            shake_cycles = max(0, int(args.pre_swing_shake_cycles))
            shake_duration = max(0.20, float(args.pre_swing_shake_duration_s))
            shake_hz = float(np.clip(float(args.pre_swing_shake_hz), 20.0, 30.0))
            if shake_mm > 0.0 and shake_cycles > 0:
                requested_extra_z = max(0.0, float(args.pre_swing_shake_extra_z_mm))
                shake_z = min(float(args.z_max), swing_start_z + requested_extra_z)
                actual_extra_z = max(0.0, shake_z - swing_start_z)
                vertical_duration = max(0.25, actual_extra_z / 220.0)
                vertical_steps = max(6, int(math.ceil(vertical_duration * shake_hz)))
                rise_points = []
                for index in range(1, vertical_steps + 1):
                    alpha = float(index) / float(vertical_steps)
                    s = alpha * alpha * (3.0 - 2.0 * alpha)
                    z = swing_start_z + actual_extra_z * s
                    rise_points.append(np.asarray([
                        swing_origin_xy[0], swing_origin_xy[1], z
                    ], dtype=np.float64))
                if actual_extra_z > 0.0:
                    print(
                        f"[D50-SHAKE-HIGH-Z] raise={actual_extra_z:.1f}mm "
                        f"z={swing_start_z:.1f}->{shake_z:.1f}mm CLOSED"
                    )
                    stream_waypoints(
                        "D50-SHAKE-VERTICAL-RISE", rise_points, vertical_duration
                    )
                shake_points = []
                endpoints = [shake_mm]
                for _ in range(shake_cycles - 1):
                    endpoints.extend([-shake_mm, shake_mm])
                endpoints.extend([-shake_mm, 0.0])
                previous_offset = 0.0
                total_distance = sum(abs(v - p) for p, v in zip([0.0] + endpoints[:-1], endpoints))
                hold_steps = max(1, int(round(0.05 * shake_hz)))
                travel_duration = max(0.40, shake_duration - 0.05 * (len(endpoints) - 1))
                for endpoint_index, endpoint in enumerate(endpoints):
                    leg_distance = abs(endpoint - previous_offset)
                    leg_steps = max(2, int(round(shake_hz * travel_duration * leg_distance / total_distance)))
                    for index in range(1, leg_steps + 1):
                        alpha = float(index) / float(leg_steps)
                        offset_mm = previous_offset + (endpoint - previous_offset) * alpha
                        xy = swing_origin_xy + up * offset_mm
                        shake_points.append(np.asarray([xy[0], xy[1], shake_z], dtype=np.float64))
                    if endpoint_index < len(endpoints) - 1:
                        shake_points.extend([shake_points[-1].copy() for _ in range(hold_steps)])
                    previous_offset = endpoint
                actual_duration = float(len(shake_points)) / shake_hz
                print(
                    f"[D50-FULL-STROKE-SHAKE] ARM2 CLOSED front=+{shake_mm:.0f}mm "
                    f"back=-{shake_mm:.0f}mm cycles={shake_cycles} "
                    f"duration={actual_duration:.2f}s T1041={shake_hz:.1f}Hz "
                    f"LINEAR Z={shake_z:.1f}mm"
                )
                stream_waypoints(
                    "D50-FULL-STROKE-FRONT-BACK-SHAKE", shake_points, actual_duration
                )
                if actual_extra_z > 0.0:
                    fall_points = list(reversed(rise_points[:-1])) + [np.asarray([
                        swing_origin_xy[0], swing_origin_xy[1], swing_start_z
                    ], dtype=np.float64)]
                    stream_waypoints(
                        "D50-SHAKE-VERTICAL-RETURN", fall_points, vertical_duration
                    )

            backswing_points = []
            for index in range(1, back_steps + 1):
                t = float(index) / float(back_steps)
                eased_t = t * t * (3.0 - 2.0 * t)
                theta = 0.5 * math.pi * eased_t
                horizontal_ratio = 1.0 - math.cos(theta)
                rise = math.sin(theta)
                xy = swing_origin_xy + (back_xy - swing_origin_xy) * horizontal_ratio
                z = swing_start_z + curve_rise * rise
                backswing_points.append(np.asarray([xy[0], xy[1], z], dtype=np.float64))
            stream_waypoints("D50-CURVE-ASCENT", backswing_points, back_duration)

            decay = max(0.1, float(args.swing_exp_decay))
            vertical_gamma = max(1.0, float(args.swing_vertical_gamma))
            end_exp = math.exp(-decay)
            forward_points = []
            forward_times = []
            final_slow_start = 0.75
            final_slow_scale = 0.55
            final_duration_scale = final_slow_start + (1.0 - final_slow_start) / final_slow_scale
            for index in range(1, forward_steps + 1):
                t = float(index) / float(forward_steps)
                s = t * t * (3.0 - 2.0 * t)
                xy = back_xy + (final_xy - back_xy) * s
                vertical_s = s ** vertical_gamma
                remain = (math.exp(-decay * vertical_s) - end_exp) / max(1e-9, 1.0 - end_exp)
                remain = float(np.clip(remain, 0.0, 1.0))
                z = final_z + (peak_z - final_z) * remain
                forward_points.append(np.asarray([xy[0], xy[1], z], dtype=np.float64))
                raw_time = t if t <= final_slow_start else final_slow_start + (t - final_slow_start) / final_slow_scale
                forward_times.append(float(raw_time / final_duration_scale))
            stream_waypoints(
                "D50-FORWARD-EXP-LAYDOWN", forward_points,
                forward_duration * final_duration_scale, forward_times,
            )

            arm.move_goal(
                args.move_command, float(final_xy[0]), float(final_xy[1]), final_z,
                plan["close_angle"], args.board_descent_speed,
            )
            board_center_pose = wait_for_waypoint(
                "D50-SWING-FINAL-SUPPORT", float(final_xy[0]), float(final_xy[1]),
                final_z, allow_abort=True,
            )

            if _probe_abort_requested():
                raise KeyboardInterrupt("operator abort before garment release")
            print(
                f"[D50-SWING:5/6] final support reached at actual="
                f"({board_center_pose[0]:.3f},{board_center_pose[1]:.3f},{board_center_pose[2]:.3f}); "
                f"release angle={plan['release_open_angle']:.3f}"
            )
            arm.gripper_open(
                plan["release_open_angle"], args.grip_spd, args.grip_acc
            )
            _interruptible_wait(args.release_settle_sec, "post-swing board release")
            release_fb = _read_gripper_feedback(attempts=3)
            if release_fb is None or "t" not in release_fb:
                raise RuntimeError("release gripper feedback unavailable")
            release_error = abs(float(release_fb["t"]) - plan["release_open_angle"])
            print(
                f"[GRIPPER-RELEASE-CHECK] target={plan['release_open_angle']:.3f} "
                f"feedback={float(release_fb['t']):.3f} error={release_error:.3f}rad"
            )
            if release_error > float(args.release_feedback_tolerance_rad):
                raise RuntimeError(
                    f"release opening error {release_error:.3f}rad exceeds "
                    f"{args.release_feedback_tolerance_rad:.3f}rad"
                )

            print(
                f"[AUTO-PLACE:6/6] set A150 standby gripper={float(args.standby_gripper_angle):.3f} "
                f"and return to taught ARM2 standby"
            )
            arm.gripper_open(
                float(args.standby_gripper_angle), args.grip_spd, args.grip_acc
            )
            _interruptible_wait(0.15, "standby gripper preparation")
            arm.move_goal(
                args.move_command,
                float(standby_target[0]),
                float(standby_target[1]),
                float(standby_target[2]),
                float(args.standby_roarm_t),
                float(args.standby_speed),
            )
            standby_pose = wait_for_waypoint(
                "ARM2-A150-STANDBY",
                float(standby_target[0]),
                float(standby_target[1]),
                float(standby_target[2]),
                allow_abort=True,
            )

            print("\n========== DIRECT CENTER RELEASE COMPLETE ==========")
            print(
                f"board_center_ARM2=({plan['board_center_x']:.3f},"
                f"{plan['board_center_y']:.3f}) release_actual_z={board_center_pose[2]:.3f}"
            )
            print(
                f"standby_command=({standby_target[0]:.3f},{standby_target[1]:.3f},"
                f"{standby_target[2]:.3f}) t={float(args.standby_roarm_t):.3f}"
            )
            print(
                f"standby_feedback=({standby_pose[0]:.2f},{standby_pose[1]:.2f},"
                f"{standby_pose[2]:.2f})"
            )
            print("gripper=A150 slight-open | ARM2 torque remains ON | ARM1 was not commanded")
            print("====================================================\n")
            return True
        except KeyboardInterrupt:
            print("\n[AUTO-PLACE-OPERATOR-STOP] no further motion commands will be sent")
            print("[STATE] ARM2 torque remains ON; gripper command state is unchanged")
            return False
        except Exception as exc:
            print(f"[AUTO-PLACE-ERROR] {exc}")
            print("[STATE] ARM2 torque remains ON; no additional release or motion is executed")
            return False

    def on_mouse(event, x, y, flags, param):
        nonlocal pending
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        if not basket_mode:
            print("[MOUSE] press B first")
        elif not active:
            print("[MOUSE] press C first")
        elif pending is not None:
            print("[MOUSE] pending point exists; Enter to save or U to cancel")
        elif calib.next_spec() is not None:
            pending = (int(x), int(y))
            spec = calib.next_spec()
            print(f"[PENDING] {spec[0]} pixel=({x},{y}); move ARM2 tip and press Enter")

    if not args.no_window:
        cv2.setMouseCallback(window, on_mouse)

    terminal = TerminalKeyReader(
        enabled=not args.no_terminal_control,
        raw=not args.terminal_line_mode,
    )
    terminal.start()

    print("[KEYS] B mode | C calibrate | H hover | D descent | Enter execute prepared H/D | U cancel | Q quit")
    for i, spec in enumerate(POINT_SPECS, 1):
        print(f"  {i}. {spec[0]}: {spec[1]}")

    try:
        if bool(args.auto_run_once):
            basket_mode = True
            print("[VLA-D50-AUTO] BASKET MODE ON; preparing exact V4 hover plan")
            if not prepare_hover_plan():
                raise RuntimeError("VLA D50 automatic hover planning failed")
            if not execute_hover_plan():
                raise RuntimeError("VLA D50 automatic hover execution failed")
            if not prepare_descent_plan():
                raise RuntimeError("VLA D50 automatic descent planning failed")
            if not execute_descent_probe():
                raise RuntimeError("VLA D50 automatic grasp/swing execution failed")
            print("[VLA-D50-AUTO-COMPLETE] basket grasp + V4 big swing laydown completed")
            return 0
        while True:
            ok, frame = cap.read()
            if not ok:
                print("[CAMERA] frame read failed")
                break
            canvas = frame.copy()
            draw(canvas, calib, pending, basket_mode, active, hover_plan)
            tkey = terminal.read_key()
            if not args.no_window:
                cv2.imshow(window, canvas)
                wkey = cv2.waitKey(1) & 0xFF
                key = tkey if tkey != 255 else wkey
            else:
                key = tkey
                time.sleep(0.001)
            if key in (ord('q'), 27):
                break
            elif key in (ord('b'), ord('B')):
                basket_mode = not basket_mode
                if not basket_mode:
                    active, pending, hover_plan, descent_plan, grasp_lift_plan, hover_ready = False, None, None, None, None, False
                print(f"[MODE] BASKET MODE {'ON' if basket_mode else 'OFF'}")
            elif key in (ord('c'), ord('C')):
                if not basket_mode:
                    print("[CALIB-BLOCKED] press B first")
                elif arm is None:
                    print("[CALIB-BLOCKED] run with --send")
                else:
                    calib.clear()
                    pending, active, hover_plan, descent_plan, grasp_lift_plan, hover_ready = None, True, None, None, None, False
                    arm.torque_off()
                    print("[CALIB] started; ARM2 torque OFF")
                    next_instruction()
            elif key in (13, 10):
                if active:
                    if not basket_mode:
                        print("[ENTER] press B first")
                    elif pending is None:
                        print("[ENTER] click the current calibration point first")
                    elif arm is None:
                        print("[ENTER] --send required")
                    else:
                        fb = arm.feedback(args.feedback_timeout)
                        if fb is None or not all(k in fb for k in ('x','y','z')):
                            print("[CALIB-ERROR] T:105 feedback failed")
                        else:
                            item = calib.add(pending, (float(fb['x']), float(fb['y']), float(fb['z'])))
                            print(f"[CALIB] saved {len(calib.points)}/5 {item['label']}: pixel={item['pixel_uv']} ARM2={item['arm2_xyz']}")
                            pending = None
                            if len(calib.points) == 5:
                                print("[CALIB] fifth point saved; auto-finalizing")
                                finalize()
                            else:
                                next_instruction()
                elif hover_plan is not None:
                    execute_hover_plan()
                elif descent_plan is not None:
                    execute_descent_probe()
                elif grasp_lift_plan is not None:
                    print("[ENTER] fallback manual execution of an already-armed plan")
                    execute_grasp_lift_test()
                else:
                    print("[ENTER] no pending action; use H for hover or D for torque descent")
            elif key in (ord('u'), ord('U')):
                if grasp_lift_plan is not None:
                    print("[GRASP-LIFT] armed grasp/lift plan cancelled")
                    grasp_lift_plan = None
                elif descent_plan is not None:
                    print("[DESCENT] pending descent plan cancelled")
                    descent_plan = None
                elif hover_plan is not None:
                    print("[HOVER] pending hover plan cancelled")
                    hover_plan = None
                elif pending is not None:
                    print(f"[CALIB] pending cancelled: {pending}")
                    pending = None
                else:
                    calib.undo()
                    if active:
                        next_instruction()
            elif key in (ord('s'), ord('S')):
                finalize()
            elif key in (ord('l'), ord('L')):
                try:
                    calib.load(args.calib_file)
                    active, pending, hover_plan, descent_plan, grasp_lift_plan, hover_ready = False, None, None, None, None, False
                except Exception as exc:
                    print(f"[CALIB-ERROR] load failed: {exc}")
            elif key in (ord('h'), ord('H')):
                prepare_hover_plan()
            elif key in (ord('d'), ord('D')):
                prepare_descent_plan()
            elif key in (ord('f'), ord('F')):
                print("[FEEDBACK] --send not active" if arm is None else arm.feedback(args.feedback_timeout))
            elif key in (ord('t'), ord('T')):
                print("[TORQUE] --send not active") if arm is None else arm.torque_off()
            elif key in (ord('n'), ord('N')):
                print("[TORQUE] --send not active") if arm is None else arm.torque_on()
            elif key in (ord('g'), ord('G')):
                print("[GRIPPER] --send not active") if arm is None else arm.gripper_open(args.grip_open, args.grip_spd, args.grip_acc)
    finally:
        terminal.stop()
        if arm:
            arm.close()
        cap.release()
        if not args.no_window:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    raise SystemExit(_run_main_with_session_log())
