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
For the ELP OV2710 + H110 camera, the displayed basket-calibration image and
all saved pixel coordinates use the undistorted 1280x720 coordinate system.
Robot motion, torque, gripper, timing, waypoint, and safety policies are unchanged.
After the adaptive vertical lift either reaches its preferred height or detects
a real vertical saturation, the last measured Z is accepted as the transit Z.
FIX11 accepts that first grasp unconditionally: no post-lift torS+torE retention
diagnosis and no automatic regrasp are performed. ARM2 then moves directly to
the board-center handoff, where ARM1 performs the already-tested Z-150 mm grasp
and four-waypoint diagonal rise. After both arms hold the garment at the same
physical height, ARM2 and ARM1 spread outward by 50 mm each. The spread pair is
then recentered without changing separation, follows a synchronized quarter-
ellipse backswing and exponential forward laydown, releases near the calibrated
board surface, and returns to taught standby. Ctrl+C aborts the stream first and
uses the same safe high-Z standby return.

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
import subprocess
import statistics
import sys
import threading
import time
import traceback
from pathlib import Path
from dataclasses import dataclass
from collections import deque
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

os.environ["QT_X11_NO_MITSHM"] = "1"
os.environ["GDK_DISABLE_SHM"] = "1"
os.environ.setdefault("NO_AT_BRIDGE", "1")

import cv2
import numpy as np

# Project layout:
#   dual/<this FIX11 script>
#   dual/undistort/camera_undistort.py
#   dual/undistort/elp_ov2710_1280x720_calibration.npz
SCRIPT_DIR = Path(__file__).resolve().parent
UNDISTORT_DIR = SCRIPT_DIR / "undistort"
if str(UNDISTORT_DIR) not in sys.path:
    sys.path.insert(0, str(UNDISTORT_DIR))

try:
    from camera_undistort import CameraCalibrationError, CameraUndistorter
    _CAMERA_UNDISTORT_IMPORT_ERROR = None
except Exception as _camera_undistort_error:
    CameraCalibrationError = RuntimeError
    CameraUndistorter = None
    _CAMERA_UNDISTORT_IMPORT_ERROR = _camera_undistort_error

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

BUILD_ID = "BASKET-HOVER-TORQUE-AUTO-GRASP-DUAL-HANDOFF-V25-FIX11-RAW-PREVIEW-POST-MASK-V8-LEGACY-H-COMPAT-RETRY-20260806"

POINT_SPECS = (
    ("P0_V0", "visible corner adjacent to hidden corner on edge A"),
    ("P1_V1", "visible corner diagonally opposite the hidden corner"),
    ("P2_V2", "other visible corner adjacent to hidden corner on edge B"),
    ("P3_E0", "point on straight edge H-V0; avoid rounded corner"),
    ("P4_E2", "point on straight edge H-V2; avoid rounded corner"),
)


SESSION_LOG_REQUESTED_DIR = Path("/home/deca/project_train/aruco_test/dual")
SESSION_LOG_DOCKER_MIRROR_DIR = Path("/workspace/project_train/aruco_test/dual")
CAMERA_CALIBRATION_DEFAULT_PATH = str(
    UNDISTORT_DIR / "elp_ov2710_1280x720_calibration.npz"
)
ELP_BASKET_CALIB_DEFAULT_PATH = "elp_ov2710_basket_arm2_5point_affine.json"


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
        filename = f"basket_v25_fix11_rawpreview_postmask_v8_terminal_{timestamp}_pid{os.getpid()}.txt"
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



def _resolve_runtime_file(path_value: str, *, include_models: bool = False) -> str:
    """Resolve project files from dual/ or dual/undistort/ without assuming CWD."""
    raw = os.path.expanduser(str(path_value or ""))
    if not raw:
        return raw
    path = Path(raw)
    if path.is_absolute():
        if path.is_file():
            return str(path)
        # A stale absolute path must not suppress filename-based recovery.
        raw = path.name
    candidates = [
        Path.cwd() / raw,
        SCRIPT_DIR / raw,
        SCRIPT_DIR.parent / raw,
        UNDISTORT_DIR / raw,
        SCRIPT_DIR.parent / "undistort" / raw,
    ]
    if include_models:
        name = Path(raw).name
        candidates.extend([
            SCRIPT_DIR / "models" / name,
            SCRIPT_DIR.parent / "models" / name,
            Path("/workspace/project_train/aruco_test/dual/models") / name,
            Path("/workspace/models") / name,
        ])
    seen = set()
    for candidate in candidates:
        candidate = candidate.expanduser()
        key = str(candidate)
        if key in seen:
            continue
        seen.add(key)
        if candidate.is_file():
            return str(candidate.resolve())
    return str(path if path.is_absolute() else Path(raw))


def _load_json_dict(path_value: str) -> Dict[str, Any]:
    path = Path(path_value)
    if not path.is_file():
        raise RuntimeError(f"JSON file not found: {path_value}")
    with path.open("r", encoding="utf-8") as fp:
        data = json.load(fp)
    if not isinstance(data, dict):
        raise RuntimeError(f"JSON root must be an object: {path_value}")
    return data


def _post_mask_open_camera(args):
    """Open ELP exactly in the D54-v8 capture mode, but only after FIX11 completes."""
    backend = str(args.backend or "v4l2").lower()
    api = cv2.CAP_ANY
    if backend == "v4l2" and hasattr(cv2, "CAP_V4L2"):
        api = cv2.CAP_V4L2
    elif backend == "dshow" and hasattr(cv2, "CAP_DSHOW"):
        api = cv2.CAP_DSHOW
    cap_local = cv2.VideoCapture(int(args.camera), api)
    fourcc = str(args.post_mask_fourcc or "YUYV").upper()
    if len(fourcc) != 4:
        cap_local.release()
        raise RuntimeError("--post-mask-fourcc must contain exactly four characters")
    cap_local.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc))
    cap_local.set(cv2.CAP_PROP_FRAME_WIDTH, int(args.width))
    cap_local.set(cv2.CAP_PROP_FRAME_HEIGHT, int(args.height))
    cap_local.set(cv2.CAP_PROP_FPS, float(args.post_mask_fps))
    try:
        cap_local.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    except Exception:
        pass
    if not cap_local.isOpened():
        cap_local.release()
        raise RuntimeError(f"post-laydown camera open failed: index={args.camera}")
    return cap_local


def _post_mask_clamp(value: float, lower: float, upper: float) -> float:
    return float(min(float(upper), max(float(lower), float(value))))


def _post_mask_robust_thresholds(
    deltas: Sequence[float],
    *,
    low_percentile: float = 65.0,
    min_samples: int = 6,
    calibration_max_delta: float = 9.0,
    stable_min: float = 4.5,
    stable_max: float = 7.5,
    stable_margin: float = 0.8,
    stable_mad_scale: float = 3.0,
    motion_min: float = 10.0,
    motion_max: float = 14.0,
    motion_gap: float = 3.0,
    motion_mad_scale: float = 6.0,
    fixed_stable_threshold: float = 0.0,
) -> Dict[str, Any]:
    """Estimate the stationary camera floor without learning large motion spikes."""
    values = np.asarray(
        [float(v) for v in deltas if np.isfinite(float(v))],
        dtype=np.float64,
    )
    if values.size < max(3, int(min_samples)):
        raise RuntimeError(
            f"noise calibration has too few finite deltas: {values.size} "
            f"< {max(3, int(min_samples))}"
        )

    percentile = _post_mask_clamp(low_percentile, 25.0, 90.0)
    low_cutoff = float(np.percentile(values, percentile))
    low_candidates = values[values <= low_cutoff + 1e-9]
    if low_candidates.size < max(3, int(min_samples) // 2):
        low_candidates = np.sort(values)[: max(3, int(math.ceil(0.5 * values.size)))]

    preliminary_median = float(np.median(low_candidates))
    preliminary_mad = float(np.median(np.abs(low_candidates - preliminary_median)))
    preliminary_sigma = 1.4826 * preliminary_mad
    robust_upper = preliminary_median + max(0.35, 3.0 * preliminary_sigma)
    filtered = low_candidates[low_candidates <= robust_upper + 1e-9]
    if filtered.size < max(3, int(min_samples) // 2):
        filtered = low_candidates

    noise_median = float(np.median(filtered))
    noise_mad = float(np.median(np.abs(filtered - noise_median)))
    robust_sigma = float(1.4826 * noise_mad)
    if noise_median > float(calibration_max_delta):
        raise RuntimeError(
            f"noise calibration low-motion median is too high: "
            f"{noise_median:.3f}>{float(calibration_max_delta):.3f}"
        )

    adaptive_stable = noise_median + max(
        float(stable_margin), float(stable_mad_scale) * robust_sigma
    )
    # V4: the configured stable_max is a preferred ceiling, not permission to
    # place the gate below the measured stationary floor.  Keep at least one
    # stable-margin above noise_median, while preserving room below motion_max.
    stable_upper_effective = max(
        float(stable_max),
        noise_median + max(0.50, float(stable_margin)),
    )
    stable_upper_effective = min(
        stable_upper_effective,
        max(float(stable_min), float(motion_max) - 1.0),
    )
    adaptive_stable = _post_mask_clamp(
        adaptive_stable, stable_min, stable_upper_effective
    )
    fixed_value = float(fixed_stable_threshold)
    stable_threshold = (
        _post_mask_clamp(fixed_value, stable_min, stable_upper_effective)
        if fixed_value > 0.0 else adaptive_stable
    )
    motion_threshold = max(
        stable_threshold + float(motion_gap),
        noise_median + float(motion_mad_scale) * robust_sigma,
        float(motion_min),
    )
    motion_threshold = _post_mask_clamp(motion_threshold, motion_min, motion_max)
    if motion_threshold <= stable_threshold:
        motion_threshold = min(float(motion_max), stable_threshold + max(1.0, float(motion_gap)))

    return {
        "sample_count": int(values.size),
        "low_candidate_count": int(low_candidates.size),
        "filtered_count": int(filtered.size),
        "low_percentile": float(percentile),
        "low_cutoff": float(low_cutoff),
        "noise_median": float(noise_median),
        "noise_mad": float(noise_mad),
        "robust_sigma": float(robust_sigma),
        "adaptive_stable_threshold": float(adaptive_stable),
        "stable_threshold": float(stable_threshold),
        "stable_upper_configured": float(stable_max),
        "stable_upper_effective": float(stable_upper_effective),
        "fixed_stable_override": float(fixed_value) if fixed_value > 0.0 else None,
        "motion_threshold": float(motion_threshold),
        "all_deltas": [float(v) for v in values.tolist()],
        "low_motion_deltas": [float(v) for v in filtered.tolist()],
    }


def _post_mask_uvc_command(args, control: str, value: int) -> Dict[str, Any]:
    tool = shutil.which("v4l2-ctl")
    device = str(args.post_mask_cam_device or f"/dev/video{int(args.camera)}")
    report: Dict[str, Any] = {
        "method": "v4l2-ctl",
        "device": device,
        "control": str(control),
        "requested": int(value),
        "available": bool(tool),
        "set_ok": False,
        "returncode": None,
        "stdout": "",
        "stderr": "",
        "readback": None,
    }
    if tool is None:
        return report
    proc = subprocess.run(
        [tool, "-d", device, "-c", f"{control}={int(value)}"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    report.update({
        "set_ok": proc.returncode == 0,
        "returncode": int(proc.returncode),
        "stdout": str(proc.stdout or "").strip(),
        "stderr": str(proc.stderr or "").strip(),
    })
    readback = subprocess.run(
        [tool, "-d", device, "-C", str(control)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    if readback.returncode == 0:
        raw = str(readback.stdout or "").strip()
        report["readback_raw"] = raw
        if ":" in raw:
            raw = raw.split(":", 1)[1].strip()
        try:
            report["readback"] = float(raw.split()[0])
        except Exception:
            report["readback"] = raw
    else:
        report["readback_error"] = str(readback.stderr or "").strip()
    return report


def _post_mask_opencv_control(cap_local, name: str, prop_id, requested: float) -> Dict[str, Any]:
    item: Dict[str, Any] = {
        "method": "opencv",
        "control": str(name),
        "requested": float(requested),
        "property_available": prop_id is not None,
        "set_return": False,
        "readback": None,
        "error": "",
    }
    if prop_id is None or cap_local is None:
        return item
    try:
        item["property_id"] = int(prop_id)
        item["set_return"] = bool(cap_local.set(int(prop_id), float(requested)))
    except Exception as exc:
        item["error"] = repr(exc)
        return item
    try:
        item["readback"] = float(cap_local.get(int(prop_id)))
    except Exception as exc:
        item["error"] = item["error"] or repr(exc)
    return item


def _post_mask_enable_auto_exposure(args, cap_local=None) -> Dict[str, Any]:
    """Request ELP automatic exposure without forcing exposure/gain values."""
    report: Dict[str, Any] = {
        "mode": "AUTO",
        "v4l2_available": bool(shutil.which("v4l2-ctl")),
        "controls": [],
        "opencv_fallback": [],
        "authoritative": False,
    }
    if report["v4l2_available"]:
        item = _post_mask_uvc_command(args, "exposure_auto", 3)
        report["controls"].append(item)
        report["authoritative"] = bool(item.get("set_ok")) and item.get("readback") in (3, 3.0, "3")
        print(
            f"[POST-MASK-CAMERA-AUTO-V4L2] exposure_auto requested=3 "
            f"set_ok={item.get('set_ok')} readback={item.get('readback')} "
            f"stderr={item.get('stderr') or '-'}"
        )
    else:
        print(
            "[POST-MASK-CAMERA-AUTO-WARN] v4l2-ctl unavailable; requesting "
            "automatic exposure through OpenCV. Install v4l-utils for authoritative readback."
        )

    if not report["authoritative"] and cap_local is not None:
        prop_id = getattr(cv2, "CAP_PROP_AUTO_EXPOSURE", None)
        # V4L2 usually exposes the UVC enum directly (3=auto). Some OpenCV
        # backends use 0.75 for auto, so try it only when the direct enum is not retained.
        for requested in (3.0, 0.75):
            item = _post_mask_opencv_control(
                cap_local, "exposure_auto", prop_id, requested
            )
            report["opencv_fallback"].append(item)
            print(
                f"[POST-MASK-CAMERA-AUTO-OPENCV] requested={requested} "
                f"cap.set={item.get('set_return')} cap.get={item.get('readback')} "
                f"error={item.get('error') or '-'}"
            )
            readback = item.get("readback")
            if item.get("set_return") and readback is not None and (
                abs(float(readback) - requested) <= 0.26
                or float(readback) >= 2.5
            ):
                break
        print(
            "[POST-MASK-CAMERA-AUTO-OPENCV-WARN] OpenCV readback is backend-dependent; "
            "startup brightness statistics will verify the practical result."
        )
    return report


def _post_mask_configure_uvc(args, cap_local=None) -> Dict[str, Any]:
    """V4 camera policy: automatic exposure by default; manual controls are opt-in."""
    manual_requested = bool(getattr(args, "post_mask_manual_camera", False))
    if not manual_requested:
        print(
            "[POST-MASK-CAMERA] mode=AUTO(default); exposure_absolute and gain are not forced"
        )
        return _post_mask_enable_auto_exposure(args, cap_local)

    report: Dict[str, Any] = {
        "mode": "MANUAL",
        "manual_requested": True,
        "v4l2_available": bool(shutil.which("v4l2-ctl")),
        "controls": [],
        "opencv_fallback_warning": False,
    }
    controls = (
        ("exposure_auto", 1, getattr(cv2, "CAP_PROP_AUTO_EXPOSURE", None), 1.0),
        ("exposure_absolute", int(args.post_mask_exposure_abs), getattr(cv2, "CAP_PROP_EXPOSURE", None), float(args.post_mask_exposure_abs)),
        ("gain", int(args.post_mask_gain), getattr(cv2, "CAP_PROP_GAIN", None), float(args.post_mask_gain)),
    )
    if report["v4l2_available"]:
        for name, value, _prop_id, _opencv_value in controls:
            item = _post_mask_uvc_command(args, name, value)
            report["controls"].append(item)
            print(
                f"[POST-MASK-CAMERA-MANUAL-V4L2] control={name} requested={value} "
                f"set_ok={item.get('set_ok')} readback={item.get('readback')} "
                f"stderr={item.get('stderr') or '-'}"
            )
    else:
        print(
            "[POST-MASK-CAMERA-MANUAL-WARN] v4l2-ctl unavailable; trying OpenCV fallback"
        )

    needs_fallback = not report["v4l2_available"] or any(
        not bool(item.get("set_ok")) for item in report["controls"]
    )
    if needs_fallback and cap_local is not None:
        fallback_items = []
        for name, _requested, prop_id, opencv_value in controls:
            item = _post_mask_opencv_control(
                cap_local, name, prop_id, opencv_value
            )
            fallback_items.append(item)
            print(
                f"[POST-MASK-CAMERA-MANUAL-OPENCV] control={name} "
                f"requested={opencv_value} cap.set={item.get('set_return')} "
                f"cap.get={item.get('readback')} error={item.get('error') or '-'}"
            )
        report["opencv_fallback"] = fallback_items
        report["opencv_fallback_warning"] = True
        print(
            "[POST-MASK-CAMERA-MANUAL-OPENCV-WARN] cap.set() success does not guarantee "
            "that the UVC camera retained the requested value."
        )
    return report


def _post_mask_global_brightness_stats(frame: np.ndarray) -> Dict[str, float]:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return {
        "mean": float(np.mean(gray)),
        "median": float(np.median(gray)),
        "p95": float(np.percentile(gray, 95.0)),
        "p99": float(np.percentile(gray, 99.0)),
        "dark_ratio_le20": float(np.count_nonzero(gray <= 20)) / float(gray.size),
        "saturated_ratio_ge250": float(np.count_nonzero(gray >= 250)) / float(gray.size),
    }


def _post_mask_collect_brightness_probe(cap_local, frame_count: int) -> Tuple[Optional[np.ndarray], Dict[str, float]]:
    samples: List[Dict[str, float]] = []
    latest = None
    for _ in range(max(3, int(frame_count))):
        ok, frame = cap_local.read()
        if not ok or frame is None:
            continue
        latest = frame.copy()
        samples.append(_post_mask_global_brightness_stats(frame))
    if not samples:
        return latest, {}
    keys = tuple(samples[0].keys())
    summary = {key: float(np.median([item[key] for item in samples])) for key in keys}
    return latest, summary



class RawPreviewController:
    """Own the ELP camera and HighGUI window for the full program lifetime.

    Normal preview is RAW only.  This thread never imports YOLO and never calls
    CameraUndistorter.correct().  It continuously consumes frames while the
    synchronous FIX11 robot functions run in the main thread, preventing stale
    V4L2 buffers.  After FIX11-COMPLETE, the mask-only stage requests distinct
    newly captured raw frames from this controller for the 0.5 s ROI stability
    gate, then performs exactly one undistort on the accepted snapshot.
    """

    def __init__(self, args, window_name: str = "FIX11 ELP RAW Preview"):
        self.args = args
        self.window_name = str(window_name)
        self._lock = threading.RLock()
        self._condition = threading.Condition(self._lock)
        self._stop = threading.Event()
        self._started = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._frame: Optional[np.ndarray] = None
        self._frame_seq = 0
        self._frame_time = 0.0
        self._keys = deque(maxlen=128)
        self._status = "RAW PREVIEW | model=UNLOADED | undistort=0"
        self._override: Optional[np.ndarray] = None
        self._override_status = ""
        self._error: Optional[BaseException] = None
        self._camera_info: Dict[str, Any] = {}

    @staticmethod
    def _fourcc_text(cap_obj) -> str:
        try:
            value = int(cap_obj.get(cv2.CAP_PROP_FOURCC))
            text = "".join(chr((value >> (8 * i)) & 0xFF) for i in range(4))
            return text.replace("\x00", "").strip() or "UNKNOWN"
        except Exception:
            return "UNKNOWN"

    def start(self, timeout: float = 12.0) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(
            target=self._run,
            name="ELP-Raw-Preview",
            daemon=True,
        )
        self._thread.start()
        if not self._started.wait(max(1.0, float(timeout))):
            raise RuntimeError("raw preview camera thread did not start in time")
        if self._error is not None:
            raise RuntimeError(f"raw preview startup failed: {self._error!r}")
        print(
            f"[RAW-PREVIEW-READY] actual={self._camera_info.get('actual_size')} "
            f"fourcc={self._camera_info.get('fourcc')} "
            f"fps={float(self._camera_info.get('fps', 0.0)):.2f} "
            "model=UNLOADED undistort_calls=0"
        )

    def _draw_status(self, image: np.ndarray, status: str) -> np.ndarray:
        canvas = image.copy()
        h, w = canvas.shape[:2]
        lines = [
            "ELP RAW LIVE PREVIEW - NO YOLO / NO UNDISTORT",
            str(status),
            "B basket | H hover plan | Enter execute | D descent | Q/ESC quit",
        ]
        y = 32
        for index, line in enumerate(lines):
            scale = 0.62 if index == 0 else 0.52
            color = (0, 255, 255) if index == 0 else (255, 255, 255)
            cv2.putText(canvas, line, (18, y), cv2.FONT_HERSHEY_SIMPLEX,
                        scale, (0, 0, 0), 5, cv2.LINE_AA)
            cv2.putText(canvas, line, (18, y), cv2.FONT_HERSHEY_SIMPLEX,
                        scale, color, 2, cv2.LINE_AA)
            y += 30
        cv2.putText(
            canvas,
            f"RAW {w}x{h} seq={self._frame_seq}",
            (18, max(25, h - 18)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (0, 0, 0),
            4,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            f"RAW {w}x{h} seq={self._frame_seq}",
            (18, max(25, h - 18)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (0, 255, 255),
            1,
            cv2.LINE_AA,
        )
        return canvas

    def _run(self) -> None:
        cap_local = None
        window_created = False
        try:
            cap_local = _post_mask_open_camera(self.args)
            control_report = _post_mask_configure_uvc(self.args, cap_local)
            probe_frame, brightness = _post_mask_collect_brightness_probe(
                cap_local, int(self.args.post_mask_camera_brightness_warmup_frames)
            )
            if brightness:
                print(
                    f"[RAW-PREVIEW-BRIGHTNESS] mode={control_report.get('mode')} "
                    f"mean={brightness['mean']:.1f} median={brightness['median']:.1f} "
                    f"p95={brightness['p95']:.1f} p99={brightness['p99']:.1f} "
                    f"dark<=20={brightness['dark_ratio_le20']:.3f} "
                    f"sat>=250={brightness['saturated_ratio_ge250']:.4f}"
                )
            too_dark = bool(brightness) and (
                float(brightness['mean']) < float(self.args.post_mask_dark_mean_min)
                or float(brightness['p95']) < float(self.args.post_mask_dark_p95_min)
            )
            if bool(getattr(self.args, 'post_mask_manual_camera', False)) and too_dark:
                print(
                    "[RAW-PREVIEW-BRIGHTNESS-RECOVERY] manual result is too dark; "
                    "restoring automatic exposure and leaving exposure/gain unrestricted"
                )
                recovery_report = _post_mask_enable_auto_exposure(self.args, cap_local)
                probe_frame, brightness = _post_mask_collect_brightness_probe(
                    cap_local, int(self.args.post_mask_camera_brightness_warmup_frames)
                )
                control_report['dark_manual_recovery'] = recovery_report
                control_report['mode'] = 'AUTO_RECOVERED_FROM_DARK_MANUAL'
                if brightness:
                    print(
                        f"[RAW-PREVIEW-BRIGHTNESS-AFTER-RECOVERY] "
                        f"mean={brightness['mean']:.1f} median={brightness['median']:.1f} "
                        f"p95={brightness['p95']:.1f} p99={brightness['p99']:.1f} "
                        f"dark<=20={brightness['dark_ratio_le20']:.3f} "
                        f"sat>=250={brightness['saturated_ratio_ge250']:.4f}"
                    )
            elif too_dark:
                print(
                    "[RAW-PREVIEW-BRIGHTNESS-WARN] automatic-exposure image is still dark; "
                    "check illumination, USB camera persistence, and install v4l-utils for authoritative controls"
                )
            actual_w = int(cap_local.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_h = int(cap_local.get(cv2.CAP_PROP_FRAME_HEIGHT))
            actual_fps = float(cap_local.get(cv2.CAP_PROP_FPS))
            self._camera_info = {
                "index": int(self.args.camera),
                "actual_size": [actual_w, actual_h],
                "fourcc": self._fourcc_text(cap_local),
                "fps": actual_fps,
                "buffer_size_requested": 1,
                "camera_control_mode": control_report.get("mode"),
                "requested_manual_exposure_absolute": int(self.args.post_mask_exposure_abs),
                "requested_manual_gain": int(self.args.post_mask_gain),
                "camera_control_report": control_report,
                "startup_brightness": brightness,
            }
            if (actual_w, actual_h) != (int(self.args.width), int(self.args.height)):
                raise RuntimeError(
                    f"raw preview size mismatch actual={actual_w}x{actual_h} "
                    f"expected={self.args.width}x{self.args.height}"
                )
            if not bool(self.args.no_window):
                cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
                window_created = True
            if probe_frame is not None:
                with self._condition:
                    self._frame = probe_frame.copy()
                    self._frame_seq += 1
                    self._frame_time = time.monotonic()
                    self._condition.notify_all()
            self._started.set()

            consecutive_failures = 0
            while not self._stop.is_set():
                ok, raw = cap_local.read()
                if not ok or raw is None:
                    consecutive_failures += 1
                    if consecutive_failures >= 30:
                        raise RuntimeError("ELP raw preview failed for 30 consecutive reads")
                    time.sleep(0.01)
                    continue
                consecutive_failures = 0
                with self._condition:
                    self._frame = raw.copy()
                    self._frame_seq += 1
                    self._frame_time = time.monotonic()
                    status = self._status
                    override = None if self._override is None else self._override.copy()
                    override_status = self._override_status
                    self._condition.notify_all()

                if window_created:
                    # A post-mask success override is already a complete single-screen
                    # corrected-image overlay. Do not wrap it in the RAW preview header
                    # or a diagnostic montage. Normal operation remains RAW-only.
                    if override is not None:
                        canvas = override
                    else:
                        canvas = self._draw_status(raw, status)
                    cv2.imshow(self.window_name, canvas)
                    key = cv2.waitKey(1) & 0xFF
                    if key != 255:
                        with self._lock:
                            self._keys.append(int(key))
        except BaseException as exc:
            self._error = exc
            print(f"[RAW-PREVIEW-ERROR] {exc}")
            self._started.set()
        finally:
            if cap_local is not None:
                cap_local.release()
            if window_created:
                try:
                    cv2.destroyWindow(self.window_name)
                except Exception:
                    pass
            with self._condition:
                self._condition.notify_all()

    def camera_info(self) -> Dict[str, Any]:
        with self._lock:
            return dict(self._camera_info)

    def set_status(self, text: str) -> None:
        with self._lock:
            self._status = str(text)

    def show_override(self, image: np.ndarray, status: str) -> None:
        with self._lock:
            self._override = np.asarray(image, dtype=np.uint8).copy()
            self._override_status = str(status)

    def clear_override(self) -> None:
        with self._lock:
            self._override = None
            self._override_status = ""

    def pop_key(self) -> int:
        with self._lock:
            if not self._keys:
                return 255
            return int(self._keys.popleft())

    def current_sequence(self) -> int:
        with self._lock:
            return int(self._frame_seq)

    def wait_for_new_frame(
        self,
        after_sequence: int,
        timeout: float = 2.0,
    ) -> Tuple[np.ndarray, int, float]:
        deadline = time.monotonic() + max(0.05, float(timeout))
        with self._condition:
            while self._frame_seq <= int(after_sequence) and not self._stop.is_set():
                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    break
                self._condition.wait(timeout=remaining)
            if self._frame is None or self._frame_seq <= int(after_sequence):
                if self._error is not None:
                    raise RuntimeError(f"raw preview camera failed: {self._error!r}")
                raise RuntimeError(
                    f"no new raw frame after seq={after_sequence} within {float(timeout):.2f}s"
                )
            return self._frame.copy(), int(self._frame_seq), float(self._frame_time)

    def stop(self, timeout: float = 4.0) -> None:
        self._stop.set()
        with self._condition:
            self._condition.notify_all()
        if self._thread is not None:
            self._thread.join(timeout=max(0.2, float(timeout)))
            if self._thread.is_alive():
                print("[RAW-PREVIEW-WARN] camera thread did not stop within timeout")


def _board_to_pixel_h(H: np.ndarray, board_x: float, board_y: float) -> Optional[Tuple[float, float]]:
    try:
        inv = np.linalg.inv(np.asarray(H, dtype=np.float64))
        src = np.asarray([[[float(board_x), float(board_y)]]], dtype=np.float32)
        dst = cv2.perspectiveTransform(src, inv.astype(np.float32))
        return float(dst[0, 0, 0]), float(dst[0, 0, 1])
    except Exception:
        return None


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
        """Send one non-blocking Cartesian T:1041 target without print/sleep.

        FIX11 controls effective speed with paired waypoint spacing and monotonic
        host timing. Each RoArm owns a separate serial object, so ARM1/ARM2 writes
        can be released concurrently without sharing one serial stream.
        """
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


def _basket_camera_geometry_matches(
    actual: Any,
    expected: Dict[str, Any],
    *,
    allow_legacy: bool = False,
) -> Tuple[bool, str]:
    """Validate that saved basket pixels belong to this runtime camera geometry."""
    if not isinstance(actual, dict) or not actual:
        if allow_legacy:
            return True, "legacy basket calibration explicitly allowed"
        return False, "basket calibration has no camera_geometry metadata"

    required = (
        "undistort_enabled",
        "calibration_id",
        "output_size",
        "alpha",
        "coordinate_space",
    )
    for key in required:
        if key not in actual:
            return False, f"basket calibration camera_geometry missing {key}"
        saved = actual.get(key)
        runtime = expected.get(key)
        if key == "alpha":
            try:
                if abs(float(saved) - float(runtime)) > 1e-6:
                    return False, f"alpha mismatch saved={saved} runtime={runtime}"
            except Exception:
                return False, f"invalid saved alpha={saved!r}"
        elif saved != runtime:
            return False, f"{key} mismatch saved={saved!r} runtime={runtime!r}"
    return True, "camera geometry metadata matched"


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

    def save(
        self,
        path: str,
        camera: str,
        width: int,
        height: int,
        arm2_port: str,
        camera_geometry: Dict[str, Any],
    ):
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
            "camera_geometry": dict(camera_geometry),
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

    def load(
        self,
        path: str,
        expected_camera_geometry: Dict[str, Any],
        *,
        allow_legacy: bool = False,
    ):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        geometry_ok, geometry_reason = _basket_camera_geometry_matches(
            data.get("camera_geometry"),
            expected_camera_geometry,
            allow_legacy=allow_legacy,
        )
        if not geometry_ok:
            raise RuntimeError(
                f"basket calibration rejected: {geometry_reason}. "
                "Re-run C and save five points in the current corrected ELP view."
            )
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
        print(f"[CALIB] loaded: {path} ({geometry_reason})")


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

    dual_roarm = config.get("dual_roarm", {})
    arm2_config = dual_roarm.get("arm2", {})
    arm1_config = dual_roarm.get("arm1", {})

    affine_raw = arm2_config.get("board_to_roarm_affine_2x3")
    affine = np.asarray(affine_raw, dtype=np.float64)
    if affine.shape != (2, 3) or not np.all(np.isfinite(affine)):
        raise RuntimeError("invalid dual_roarm.arm2.board_to_roarm_affine_2x3")
    center_arm2 = affine[:, :2] @ center_board + affine[:, 2]

    arm1_affine_raw = arm1_config.get("board_to_roarm_affine_2x3")
    arm1_affine = np.asarray(arm1_affine_raw, dtype=np.float64)
    if arm1_affine.shape != (2, 3) or not np.all(np.isfinite(arm1_affine)):
        raise RuntimeError("invalid dual_roarm.arm1.board_to_roarm_affine_2x3")
    center_arm1 = arm1_affine[:, :2] @ center_board + arm1_affine[:, 2]

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
        "arm1_center_xy": [float(center_arm1[0]), float(center_arm1[1])],
        "arm2_affine_2x3": affine.tolist(),
        "arm1_affine_2x3": arm1_affine.tolist(),
        "board_bounds": [
            float(np.min(corners[:, 0])), float(np.max(corners[:, 0])),
            float(np.min(corners[:, 1])), float(np.max(corners[:, 1])),
        ],
        "arm2_inner_xy": [float(inner_xy[0]), float(inner_xy[1])],
        "safe_hover_z": safe_hover_z,
        "marker_board_mm": {
            key: [float(marker_map[key][0]), float(marker_map[key][1])]
            for key in required
        },
        "surface_z_plane_by_arm": {
            "arm2": [float(v) for v in arm2_config.get("surface_z_plane_abc", [])],
            "arm1": [float(v) for v in arm1_config.get("surface_z_plane_abc", [])],
        },
        "diagonal_vs_corner_mean_error_mm": diagonal_vs_mean_error,
    }


def main():
    print(f"[BUILD] {BUILD_ID}")
    print("[VISION] startup RAW OpenCV preview is live; TensorRT remains unloaded and live undistort remains disabled")
    print("[ELP-UNDISTORT] after FIX11, raw_H ROI must remain stable for 0.5s; only the accepted RAW snapshot is remapped once")
    print("[WORKFLOW] single ARM2 basket grasp -> board-center HOLD -> ARM1 Z-150mm grasp -> diagonal rise -> 50mm/arm spread -> pair recenter -> synchronized quarter-ellipse/exponential laydown -> release -> standby")
    print("[SAFETY] ARM2 basket mechanics and single-grasp policy are preserved. FIX11 adds only the post-spread laydown. All board/local/Z/radius/stream-step targets are preflighted before recenter motion. Ctrl+C stops waypoint generation before the existing high-Z standby return.")
    print("[SINGLE-GRASP] post-lift torS+torE retention diagnosis is DISABLED; no automatic regrasp is performed")
    parser = argparse.ArgumentParser(description="ARM2 basket 5-point affine calibration and diagonal-center visualization")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--backend", default="v4l2", choices=["auto","v4l2","dshow","any"])
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument(
        "--camera-undistort",
        dest="camera_undistort",
        action="store_true",
        default=True,
        help="Enable ELP OV2710 + H110 lens correction for the displayed/calibrated basket image.",
    )
    parser.add_argument(
        "--no-camera-undistort",
        dest="camera_undistort",
        action="store_false",
        help="Disable lens correction. A basket calibration from corrected coordinates will be rejected.",
    )
    parser.add_argument(
        "--camera-calibration",
        default=CAMERA_CALIBRATION_DEFAULT_PATH,
        help="ELP calibration .npz/.json. Default: undistort/elp_ov2710_1280x720_calibration.npz under the dual folder.",
    )
    parser.add_argument(
        "--camera-undistort-alpha",
        type=float,
        default=0.0,
        help="OpenCV optimal-new-camera-matrix alpha in [0,1]. Verified ELP setting: 0.0.",
    )
    parser.add_argument(
        "--camera-undistort-strict-size",
        dest="camera_undistort_strict_size",
        action="store_true",
        default=True,
        help="Require runtime frame size to exactly match the calibration size.",
    )
    parser.add_argument(
        "--camera-undistort-allow-size-scale",
        dest="camera_undistort_strict_size",
        action="store_false",
        help="Allow intrinsic scaling only when the aspect ratio matches. Not recommended for robot calibration.",
    )
    parser.add_argument(
        "--allow-legacy-basket-calib",
        action="store_true",
        help="Unsafe compatibility option: allow a basket affine without camera-geometry metadata.",
    )
    parser.add_argument("--display", default=":0")
    parser.add_argument("--no-window", action="store_true")
    parser.add_argument("--no-terminal-control", action="store_true")
    parser.add_argument("--terminal-line-mode", action="store_true")
    parser.add_argument("--send", action="store_true")
    parser.add_argument("--arm2-port", default="/dev/roarm_2")
    parser.add_argument("--arm1-port", default="/dev/roarm_1",
                        help="ARM1 port; lazily opened only after ARM2 reaches board-center handoff")
    parser.add_argument("--handoff-settle-sec", type=float, default=3.0,
                        help="ARM2 stationary hold before ARM1 is opened/moved")
    parser.add_argument("--arm1-handoff-below-mm", type=float, default=150.0,
                        help="ARM1 final physical tip height below actual ARM2 handoff Z")
    parser.add_argument("--arm1-z-calibration-offset-mm", type=float, default=0.0,
                        help="Optional ARM1-vs-ARM2 Cartesian Z correction")
    parser.add_argument("--arm1-approach-offset-mm", type=float, default=120.0,
                        help="Board +X lateral safe pre-approach offset from board center")
    parser.add_argument("--arm1-preapproach-clearance-mm", type=float, default=80.0,
                        help="Clearance above final ARM1 target Z for lateral pre-approach")
    parser.add_argument("--arm1-approach-speed", type=float, default=0.70,
                        help="ARM1 speed for safe pre-approach segments")
    parser.add_argument("--arm1-insert-speed", type=float, default=0.30,
                        help="ARM1 speed for final horizontal insertion")
    parser.add_argument("--arm1-rise-spread-mm", type=float, default=120.0,
                        help="Board +X outward separation added while ARM1 rises after grasp")
    parser.add_argument("--arm1-rise-speed", type=float, default=0.45,
                        help="ARM1 speed for post-grasp diagonal rise waypoints")
    parser.add_argument("--dual-air-spread-each-mm", type=float, default=50.0,
                        help="Additional synchronized outward air spread applied to EACH arm after dual hang")
    parser.add_argument("--dual-air-spread-speed", type=float, default=0.35,
                        help="Speed for the synchronized post-grasp air spread")
    parser.add_argument("--dual-air-spread-settle-sec", type=float, default=0.8,
                        help="Settle time after both air-spread waypoints are reached")
    parser.add_argument("--dual-arc-laydown", dest="dual_arc_laydown", action="store_true", default=True,
                        help="Run FIX11 pair recenter + synchronized curved laydown after air spread")
    parser.add_argument("--no-dual-arc-laydown", dest="dual_arc_laydown", action="store_false",
                        help="Stop after the proven FIX8 50mm-per-arm air spread")
    parser.add_argument("--laydown-swing-amplitude-mm", type=float, default=200.0,
                        help="Backswing A; forward phase travels 2A and ends A toward ID0-ID2")
    parser.add_argument("--laydown-curve-rise-mm", type=float, default=50.0,
                        help="Additional Z rise during the quarter-ellipse backswing")
    parser.add_argument("--laydown-min-curve-rise-mm", type=float, default=20.0,
                        help="Block before recenter if common additional Z room is smaller")
    parser.add_argument("--laydown-z-limit-margin-mm", type=float, default=5.0,
                        help="Reserved margin below configured z_max during curved ascent")
    parser.add_argument("--laydown-stream-hz", type=float, default=22.0,
                        help="Nominal T:1041 paired-stream waypoint rate")
    parser.add_argument("--laydown-backswing-duration-sec", type=float, default=1.30,
                        help="Quarter-ellipse backswing duration")
    parser.add_argument("--laydown-forward-duration-sec", type=float, default=2.30,
                        help="Exponential forward-laydown base duration")
    parser.add_argument("--laydown-decay", type=float, default=0.25,
                        help="Normalized exponential Z decay coefficient")
    parser.add_argument("--laydown-vertical-gamma", type=float, default=1.70,
                        help="Gamma above 1 delays descent and keeps the garment high longer")
    parser.add_argument("--laydown-final-slow-zone-ratio", type=float, default=0.25,
                        help="Final forward-trajectory fraction that is slowed")
    parser.add_argument("--laydown-final-slow-speed-scale", type=float, default=0.55,
                        help="Relative speed in the final slow zone")
    parser.add_argument("--laydown-reversal-hold-sec", type=float, default=0.03,
                        help="Short zero-motion hold at the curve peak")
    parser.add_argument("--laydown-recenter-speed", type=float, default=0.30,
                        help="T:104 speed for pair-preserving X recenter")
    parser.add_argument("--laydown-recenter-settle-sec", type=float, default=0.45,
                        help="Settle after pair recenter before T:1041 streaming")
    parser.add_argument("--laydown-recenter-xy-tolerance-mm", type=float, default=15.0,
                        help="Dedicated XY arrival tolerance for the pair recenter")
    parser.add_argument("--laydown-recenter-z-tolerance-mm", type=float, default=10.0,
                        help="Dedicated Z-hold tolerance for the pair recenter")
    parser.add_argument("--laydown-recenter-hard-z-drift-mm", type=float, default=20.0,
                        help="Immediate safety block when settled recenter Z error exceeds this value")
    parser.add_argument("--laydown-recenter-correction-attempts", type=int, default=1,
                        help="Maximum same-target recenter re-command count after the first attempt")
    parser.add_argument("--laydown-recenter-strict-timeout-sec", type=float, default=4.0,
                        help="Maximum strict XY/Z arrival wait per recenter command before stability evaluation")
    parser.add_argument("--laydown-recenter-stable-gap-sec", type=float, default=0.18,
                        help="Gap between consecutive recenter stability samples")
    parser.add_argument("--laydown-recenter-stable-samples", type=int, default=5,
                        help="Number of settled T:105 samples used for actual-pose reanchoring")
    parser.add_argument("--laydown-recenter-stable-xy-span-mm", type=float, default=2.0,
                        help="Maximum pairwise XY span across stability samples")
    parser.add_argument("--laydown-recenter-stable-z-span-mm", type=float, default=1.5,
                        help="Maximum Z peak-to-peak span across stability samples")
    parser.add_argument("--laydown-recenter-stable-delta-mm", type=float, default=5.0,
                        help="Deprecated compatibility option; FIX11 uses separate XY/Z span limits")
    parser.add_argument("--laydown-board-margin-mm", type=float, default=25.0,
                        help="Required marker-board margin for every board waypoint")
    parser.add_argument("--laydown-roarm-radius-max-mm", type=float, default=360.0,
                        help="Maximum local XY radius allowed at every waypoint")
    parser.add_argument("--laydown-local-axis-max-mm", type=float, default=400.0,
                        help="Maximum absolute local X or Y allowed at every waypoint")
    parser.add_argument("--laydown-stream-max-step-mm", type=float, default=24.0,
                        help="Maximum 3D step between consecutive streamed targets")
    parser.add_argument("--laydown-stream-max-lag-sec", type=float, default=0.12,
                        help="Abort stream when host timing falls this far behind")
    parser.add_argument("--laydown-stream-max-write-ms", type=float, default=35.0,
                        help="Abort stream when one serial write exceeds this time")
    parser.add_argument("--laydown-waypoint-log-stride", type=int, default=4,
                        help="Print every Nth streamed paired waypoint")
    parser.add_argument("--laydown-arm2-final-clearance-mm", type=float, default=30.0,
                        help="ARM2 release support height above calibrated board surface")
    parser.add_argument("--laydown-arm1-final-clearance-mm", type=float, default=50.0,
                        help="ARM1 release support height above calibrated board surface")
    parser.add_argument("--laydown-final-lock-speed", type=float, default=0.35,
                        help="T:104 speed used to reassert the last streamed support target")
    parser.add_argument("--laydown-final-lock-wait-sec", type=float, default=0.25,
                        help="Wait after paired final T:104 support lock")
    parser.add_argument("--laydown-final-verify-timeout-sec", type=float, default=1.20,
                        help="Short best-effort T:105 confirmation window after final support lock")
    parser.add_argument("--laydown-final-verify-tolerance-mm", type=float, default=55.0,
                        help="Best-effort final support feedback tolerance; failure never returns to origin")
    parser.add_argument("--laydown-final-settle-sec", type=float, default=0.45,
                        help="Closed-gripper settle at final support before release")
    parser.add_argument("--laydown-release-open-angle", type=float, default=1.35,
                        help="Paired gripper opening used after final support")
    parser.add_argument("--laydown-release-settle-sec", type=float, default=1.0,
                        help="Wait after paired release before automatic standby return")
    parser.add_argument("--arm1-gripper-open-angle", type=float, default=1.35,
                        help="ARM1 remains open throughout the approach path before the final grasp")
    parser.add_argument("--baudrate", type=int, default=115200)
    parser.add_argument("--feedback-timeout", type=float, default=2.5)
    parser.add_argument("--grip-open", type=float, default=1.35)
    parser.add_argument("--grip-spd", type=float, default=0.0)
    parser.add_argument("--grip-acc", type=float, default=0.0)
    parser.add_argument("--calib-file", default=ELP_BASKET_CALIB_DEFAULT_PATH)
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
                        help="Speed for Ctrl+C safe return to the taught standby poses")
    parser.add_argument("--arm1-standby-roarm-x", type=float, default=9.981603127,
                        help="A150 taught ARM1 standby RoArm X")
    parser.add_argument("--arm1-standby-roarm-y", type=float, default=-127.327706,
                        help="A150 taught ARM1 standby RoArm Y")
    parser.add_argument("--arm1-standby-roarm-z", type=float, default=24.8031204,
                        help="A150 taught ARM1 standby RoArm Z")
    parser.add_argument("--arm1-standby-roarm-t", type=float, default=2.899223689,
                        help="A150 taught ARM1 standby tool angle")
    parser.add_argument("--shutdown-clear-z", type=float, default=180.0,
                        help="Minimum high-Z clearance used by Ctrl+C standby return")
    parser.add_argument("--shutdown-gripper-angle", type=float, default=1.57,
                        help="Slightly-open gripper angle used before and after Ctrl+C standby return")
    parser.add_argument("--shutdown-release-settle-sec", type=float, default=0.8,
                        help="Wait after both connected grippers slightly open on Ctrl+C")
    parser.add_argument("--board-descent-speed", type=float, default=0.35,
                        help="ARM2 speed for vertical descent at the board center")
    parser.add_argument("--release-open-angle", type=float, default=1.35,
                        help="T:106 angle used to release the garment over the board")
    parser.add_argument("--release-settle-sec", type=float, default=1.0,
                        help="Wait after opening the gripper at the board center")
    parser.add_argument("--release-feedback-tolerance-rad", type=float, default=0.22,
                        help="Allowed T:105 t error when verifying release opening")
    parser.add_argument("--probe-csv", default="")

    # FIX11 post-laydown mask-only diagnostic. The default operational path keeps
    # the camera closed and the TensorRT engine unloaded until FIX11 has released
    # the garment and both arms have returned to standby.
    parser.add_argument(
        "--post-laydown-mask-only",
        dest="post_laydown_mask_only",
        action="store_true",
        default=True,
        help="After FIX11 completion, capture one stable raw snapshot, undistort once, and infer only the garment mask.",
    )
    parser.add_argument(
        "--no-post-laydown-mask-only",
        dest="post_laydown_mask_only",
        action="store_false",
        help="Disable the post-laydown camera/model stage and retain pure FIX11 behavior.",
    )
    parser.add_argument(
        "--raw-preview",
        dest="raw_preview",
        action="store_true",
        default=True,
        help="Open ELP at startup and continuously display/consume RAW frames. No YOLO or undistort runs before FIX11 completion.",
    )
    parser.add_argument(
        "--no-raw-preview",
        dest="raw_preview",
        action="store_false",
        help="Disable the startup raw preview. Post-mask capture will open the camera only after FIX11 completion.",
    )
    parser.add_argument(
        "--startup-camera",
        dest="raw_preview",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--post-mask-hfile",
        default="elp_ov2710_folding_board_homography_cache.json",
        help="ELP Homography bundle containing corrected H, raw_H, and camera_geometry.",
    )
    parser.add_argument(
        "--post-mask-seg-model",
        default="models/kfashion_yolo26s_seg3_e100_best.engine",
        help="Segmentation TensorRT engine. It is resolved and loaded only after FIX11 completion.",
    )
    parser.add_argument("--post-mask-seg-imgsz", type=int, default=640)
    parser.add_argument("--post-mask-conf-ladder", default="0.12,0.07,0.03")
    parser.add_argument("--post-mask-classes", default="outer,top,bottoms,bottom")
    parser.add_argument("--post-mask-fourcc", default="YUYV")
    parser.add_argument("--post-mask-fps", type=float, default=9.0)
    parser.add_argument("--post-mask-cam-device", default="")
    parser.add_argument("--post-mask-exposure-abs", type=int, default=35)
    parser.add_argument("--post-mask-gain", type=int, default=5)
    parser.add_argument(
        "--post-mask-manual-camera",
        dest="post_mask_manual_camera",
        action="store_true",
        default=False,
        help="Opt in to fixed manual exposure/gain. Default V4 policy is automatic exposure.",
    )
    parser.add_argument(
        "--post-mask-no-manual-camera",
        dest="post_mask_manual_camera",
        action="store_false",
        help="Compatibility alias: keep the V4 default automatic-exposure policy.",
    )
    parser.add_argument("--post-mask-camera-brightness-warmup-frames", type=int, default=12)
    parser.add_argument("--post-mask-dark-mean-min", type=float, default=35.0)
    parser.add_argument("--post-mask-dark-p95-min", type=float, default=80.0)
    parser.add_argument("--post-mask-warmup-frames", type=int, default=8)
    parser.add_argument(
        "--post-mask-stable-threshold", type=float, default=0.0,
        help="Positive value overrides adaptive stability threshold; 0 uses noise calibration.",
    )
    parser.add_argument("--post-mask-stable-sec", type=float, default=0.50)
    parser.add_argument("--post-mask-timeout-sec", type=float, default=12.0)
    parser.add_argument("--post-mask-noise-calibration-sec", type=float, default=1.80)
    parser.add_argument("--post-mask-noise-low-percentile", type=float, default=65.0)
    parser.add_argument("--post-mask-noise-min-samples", type=int, default=8)
    parser.add_argument("--post-mask-noise-max-delta", type=float, default=12.0)
    parser.add_argument("--post-mask-stable-threshold-min", type=float, default=4.5)
    parser.add_argument("--post-mask-stable-threshold-max", type=float, default=10.0)
    parser.add_argument("--post-mask-stable-margin", type=float, default=0.8)
    parser.add_argument("--post-mask-stable-mad-scale", type=float, default=3.0)
    parser.add_argument("--post-mask-motion-threshold-min", type=float, default=10.0)
    parser.add_argument("--post-mask-motion-threshold-max", type=float, default=14.0)
    parser.add_argument("--post-mask-motion-gap", type=float, default=3.0)
    parser.add_argument("--post-mask-motion-mad-scale", type=float, default=6.0)
    parser.add_argument("--post-mask-delta-median-window", type=int, default=5)
    parser.add_argument("--post-mask-ambiguous-hold-sec", type=float, default=0.22)
    parser.add_argument("--post-mask-signature-width", type=int, default=128)
    parser.add_argument("--post-mask-signature-height", type=int, default=96)
    parser.add_argument("--post-mask-roi-inset-mm", type=float, default=10.0)
    parser.add_argument(
        "--post-mask-roi-inset-px", type=int, default=0,
        help="Optional additional pixel erosion after the 10 mm board-space inset.",
    )
    parser.add_argument("--post-mask-marker-exclusion-radius-px", type=int, default=30)
    parser.add_argument(
        "--post-mask-grasp-edge-inset-px",
        type=float,
        default=18.0,
        help="Preferred inward offset from each minor-axis garment boundary for stable grasp markers.",
    )
    parser.add_argument(
        "--post-mask-grasp-search-span-px",
        type=float,
        default=14.0,
        help="Search span around the preferred edge inset using distance-transform clearance.",
    )
    parser.add_argument(
        "--post-mask-grasp-min-clearance-px",
        type=float,
        default=5.0,
        help="Preferred minimum mask-interior clearance radius for each displayed grasp point.",
    )
    parser.add_argument(
        "--post-mask-grasp-axis-step-px",
        type=float,
        default=0.5,
        help="Subpixel sampling step used to intersect the PCA minor axis with the garment mask.",
    )
    parser.add_argument(
        "--post-mask-max-attempts",
        type=int,
        default=0,
        help="Maximum fresh-snapshot mask attempts; 0 keeps retrying until success or operator abort.",
    )
    parser.add_argument(
        "--post-mask-retry-delay-sec",
        type=float,
        default=0.60,
        help="Pause between failed fresh-snapshot attempts. Q/ESC aborts perception only.",
    )
    parser.add_argument("--post-mask-debug-dir", default="post_laydown_mask_debug")
    parser.add_argument(
        "--post-mask-hold-window",
        dest="post_mask_hold_window",
        action="store_true",
        default=True,
        help="Keep the final mask diagnostic window open until Q/ESC.",
    )
    parser.add_argument(
        "--no-post-mask-hold-window",
        dest="post_mask_hold_window",
        action="store_false",
    )
    args = parser.parse_args()

    board_target = _load_board_center_arm2(args.board_config)
    board_center_x, board_center_y = board_target["arm2_center_xy"]
    arm1_center_x, arm1_center_y = board_target["arm1_center_xy"]
    board_inner_x, board_inner_y = board_target["arm2_inner_xy"]
    config_safe_hover_z = float(board_target["safe_hover_z"])
    print(
        f"[BOARD] config={board_target['config_path']} "
        f"board_center=({board_target['board_center_xy'][0]:.3f},"
        f"{board_target['board_center_xy'][1]:.3f}) "
        f"ARM2_center=({board_center_x:.3f},{board_center_y:.3f}) "
        f"ARM1_same_board_center=({arm1_center_x:.3f},{arm1_center_y:.3f}) "
        f"ARM2_inner=({board_inner_x:.3f},{board_inner_y:.3f}) "
        f"safe_hover_z={config_safe_hover_z:.3f} "
        f"diag-vs-mean={board_target['diagonal_vs_corner_mean_error_mm']:.6f}mm"
    )

    if args.display and not args.no_window:
        os.environ["DISPLAY"] = args.display

    # The camera is opened at startup for RAW live preview and continuous buffer
    # consumption.  Homography metadata is loaded now only to validate the saved
    # basket affine.  No CameraUndistorter.correct() and no TensorRT load occurs
    # before FIX11-COMPLETE.
    post_h_path = _resolve_runtime_file(args.post_mask_hfile)
    post_h_bundle: Dict[str, Any] = {}
    try:
        post_h_bundle = _load_json_dict(post_h_path)
    except Exception as exc:
        print(f"[POST-MASK-H-WARN] startup metadata unavailable: {exc}")
    width = int(args.width)
    height = int(args.height)
    camera_geometry_metadata = dict(post_h_bundle.get("camera_geometry", {}) or {})
    if bool(args.camera_undistort):
        camera_geometry_metadata.update({
            "camera_model": "ELP_OV2710_H110",
            "undistort_enabled": True,
            "input_size": list(camera_geometry_metadata.get("input_size", [width, height])),
            "output_size": list(camera_geometry_metadata.get("output_size", [width, height])),
            "alpha": float(camera_geometry_metadata.get("alpha", args.camera_undistort_alpha)),
            "coordinate_space": "undistorted_pixel_uv",
        })
    else:
        camera_geometry_metadata = {
            "camera_model": "UNSPECIFIED_RAW_CAMERA",
            "undistort_enabled": False,
            "calibration_id": "DISABLED",
            "calibration_size": None,
            "input_size": [width, height],
            "output_size": [width, height],
            "alpha": 0.0,
            "scaled_intrinsics": False,
            "map_type": "DISABLED",
            "coordinate_space": "raw_pixel_uv",
        }

    preview: Optional[RawPreviewController] = None
    if bool(args.raw_preview):
        preview = RawPreviewController(args)
        preview.start()
        preview_info = preview.camera_info()
        actual_size = preview_info.get("actual_size", [width, height])
        width, height = int(actual_size[0]), int(actual_size[1])
        print(
            "[STARTUP-RAW-PREVIEW] camera OPEN and continuously consuming raw frames; "
            "OpenCV RAW window active; TensorRT model UNLOADED; undistort calls=0"
        )
    else:
        print(
            "[STARTUP-RAW-PREVIEW-WARN] disabled by --no-raw-preview; "
            "there will be no OpenCV preview before FIX11-COMPLETE"
        )

    arm = RoArmSerial(args.arm2_port, args.baudrate) if args.send else None
    if arm is not None:
        arm.synchronize_startup(args.startup_timeout, args.startup_quiet_sec)
    print(f"[MODE] {'ARM2 connected' if arm else 'vision/load only; add --send for calibration feedback'}")
    # FIX8 invariant: ARM1 MUST NOT be opened during startup.
    # Program startup through the entire V24 basket pipeline therefore uses the
    # exact same single ARM2 serial session as the known-good V24.
    arm1 = None
    calib = BasketCalib()
    if args.load_calib and os.path.exists(args.calib_file):
        try:
            calib.load(
                args.calib_file,
                camera_geometry_metadata,
                allow_legacy=bool(args.allow_legacy_basket_calib),
            )
        except Exception as exc:
            print(f"[CALIB-WARN] load failed: {exc!r}")

    basket_mode = False
    active = False
    pending: Optional[Tuple[int, int]] = None
    hover_plan: Optional[Dict[str, float]] = None
    descent_plan: Optional[Dict[str, float]] = None
    grasp_lift_plan: Optional[Dict[str, float]] = None
    hover_ready = False
    window = "FIX11 ELP RAW Preview"

    def next_instruction():
        spec = calib.next_spec()
        if spec:
            print(f"\n[NEXT] {spec[0]}: {spec[1]}\n  click image -> move ARM2 tip -> Enter")

    def finalize():
        nonlocal active, pending
        try:
            g = calib.compute(width, height, args)
            calib.save(
                args.calib_file,
                str(args.camera),
                width,
                height,
                args.arm2_port,
                camera_geometry_metadata,
            )
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

    def _board_to_arm2_xy(board_x: float, board_y: float):
        affine2 = np.asarray(board_target["arm2_affine_2x3"], dtype=np.float64)
        p = affine2[:, :2] @ np.asarray([float(board_x), float(board_y)], dtype=np.float64) + affine2[:, 2]
        return float(p[0]), float(p[1])

    def _board_to_arm1_xy(board_x: float, board_y: float):
        affine1 = np.asarray(board_target["arm1_affine_2x3"], dtype=np.float64)
        p = affine1[:, :2] @ np.asarray([float(board_x), float(board_y)], dtype=np.float64) + affine1[:, 2]
        return float(p[0]), float(p[1])

    def _arm_xy_to_board(arm_key: str, arm_x: float, arm_y: float):
        key = "arm1_affine_2x3" if arm_key == "arm1" else "arm2_affine_2x3"
        affine = np.asarray(board_target[key], dtype=np.float64)
        linear = affine[:, :2]
        offset = affine[:, 2]
        if abs(float(np.linalg.det(linear))) < 1e-9:
            raise RuntimeError(f"{arm_key} board affine is singular")
        p = np.linalg.solve(
            linear,
            np.asarray([float(arm_x), float(arm_y)], dtype=np.float64) - offset,
        )
        return np.asarray([float(p[0]), float(p[1])], dtype=np.float64)

    def _surface_z(arm_key: str, board_x: float, board_y: float) -> float:
        plane = np.asarray(
            board_target.get("surface_z_plane_by_arm", {}).get(arm_key, []),
            dtype=np.float64,
        )
        if plane.shape != (3,) or not np.all(np.isfinite(plane)):
            raise RuntimeError(f"missing/invalid {arm_key} surface_z_plane_abc in board config")
        return float(plane[0] * float(board_x) + plane[1] * float(board_y) + plane[2])

    def _lazy_open_arm1_for_handoff():
        nonlocal arm1
        if arm1 is not None:
            return arm1
        if not args.send:
            raise RuntimeError("ARM1 handoff approach requires --send")
        print("[ARM1-LAZY] opening ARM1 only now; V24 ARM2 startup/basket path has already completed")
        # Deliberately use the unmodified V24 RoArmSerial implementation.
        arm1 = RoArmSerial(args.arm1_port, args.baudrate)
        arm1.synchronize_startup(args.startup_timeout, args.startup_quiet_sec)
        # Startup text can contain short quiet gaps. Before any ARM1 motion, require
        # actual T:105 feedback without changing the V24 serial class itself.
        deadline = time.monotonic() + 20.0
        last_exc = None
        while time.monotonic() < deadline:
            try:
                fb = arm1.feedback_retry(args.feedback_timeout, attempts=2, retry_delay=0.35)
            except Exception as exc:
                last_exc = exc
                print(f"[ARM1-LAZY] waiting for ARM1 command readiness: {exc}")
                time.sleep(0.5)
                continue
            if fb is not None and all(k in fb for k in ("x", "y", "z")):
                print(
                    f"[ARM1-LAZY] READY pose=({float(fb['x']):.2f},"
                    f"{float(fb['y']):.2f},{float(fb['z']):.2f})"
                )
                return arm1
            time.sleep(0.4)
        raise RuntimeError(f"ARM1 did not become T:105-ready after lazy open: {last_exc}")

    def _wait_for_arm1_waypoint(label: str, x: float, y: float, z: float):
        robot1 = _lazy_open_arm1_for_handoff()
        deadline = time.time() + max(0.5, float(args.move_timeout))
        target = np.asarray([float(x), float(y), float(z)], dtype=np.float64)
        last_pose = None
        last_error = float("inf")
        while time.time() < deadline:
            if _probe_abort_requested():
                raise KeyboardInterrupt(f"operator abort during {label}")
            time.sleep(max(0.05, float(args.move_poll_sec)))
            fb = robot1.feedback(args.feedback_timeout)
            if fb is None or not all(k in fb for k in ("x", "y", "z")):
                continue
            actual = np.asarray([float(fb["x"]), float(fb["y"]), float(fb["z"])], dtype=np.float64)
            last_pose = actual
            last_error = float(np.linalg.norm(actual - target))
            print(f"[{label}] ARM1 feedback=({actual[0]:.1f},{actual[1]:.1f},{actual[2]:.1f}) error={last_error:.1f}mm")
            if last_error <= float(args.move_tolerance_mm):
                time.sleep(max(0.0, float(args.move_wait)))
                return actual
        pose_text = "unavailable" if last_pose is None else f"({last_pose[0]:.1f},{last_pose[1]:.1f},{last_pose[2]:.1f})"
        raise RuntimeError(f"{label} timeout target=({x:.1f},{y:.1f},{z:.1f}) last={pose_text} error={last_error:.1f}mm")

    def _execute_arm1_grasp_and_diagonal_rise(arm2_handoff_pose):
        robot1 = _lazy_open_arm1_for_handoff()
        arm2_pose = np.asarray(arm2_handoff_pose, dtype=np.float64).reshape(3)
        center_board = np.asarray(board_target["board_center_xy"], dtype=np.float64)
        bx0, bx1, by0, by1 = [float(v) for v in board_target["board_bounds"]]

        target_x, target_y = float(arm1_center_x), float(arm1_center_y)
        target_z = float(arm2_pose[2]) - abs(float(args.arm1_handoff_below_mm)) + float(args.arm1_z_calibration_offset_mm)
        _require_in_range("ARM1 target_z", target_z, args.z_min, args.z_max)

        approach_board_x = min(bx1 - 25.0, float(center_board[0]) + max(20.0, float(args.arm1_approach_offset_mm)))
        approach_board_y = float(center_board[1])
        approach_x, approach_y = _board_to_arm1_xy(approach_board_x, approach_board_y)

        fb = robot1.feedback_retry(args.feedback_timeout, attempts=4, retry_delay=0.3)
        if fb is None or not all(k in fb for k in ("x", "y", "z")):
            raise RuntimeError("ARM1 T:105 feedback failed before approach")
        current = np.asarray([float(fb["x"]), float(fb["y"]), float(fb["z"])], dtype=np.float64)
        tool_t = float(fb.get("t", args.tool_angle_fallback))
        if not np.isfinite(tool_t): tool_t = float(args.tool_angle_fallback)
        pre_z = min(float(args.z_max), max(float(current[2]), target_z + max(20.0, float(args.arm1_preapproach_clearance_mm))))

        print("\n========== V25 FIX8 ARM1 APPROACH + GRASP + DIAGONAL RISE ==========")
        print(f"same physical board XY=({center_board[0]:.3f},{center_board[1]:.3f})")
        print(f"ARM2 local center=({board_center_x:.3f},{board_center_y:.3f}); ARM1 local center=({target_x:.3f},{target_y:.3f})")
        print(f"ARM2 actual handoff Z={arm2_pose[2]:.3f}; ARM1 target Z={target_z:.3f} ({abs(float(args.arm1_handoff_below_mm)):.1f}mm below)")
        print(f"ARM1 lateral safe local=({approach_x:.3f},{approach_y:.3f}) pre_z={pre_z:.3f}")
        print("ARM1 approaches OPEN; after final arrival it mirrors V24 post-contact grasp, then rises outward through four explicit waypoints to ARM2 physical height. ARM2 remains CLOSED and stationary; no extra spread/D47/release.")
        print("===================================================\n")

        robot1.torque_on()
        robot1.gripper_open(float(args.arm1_gripper_open_angle), args.grip_spd, args.grip_acc)
        _interruptible_wait(0.4, "ARM1 open")

        if float(current[2]) < pre_z - 1.0:
            robot1.move_goal(args.move_command, float(current[0]), float(current[1]), pre_z, tool_t, float(args.arm1_approach_speed))
            current = _wait_for_arm1_waypoint("ARM1-1/4-VERTICAL-CLEAR", float(current[0]), float(current[1]), pre_z)
        else:
            print("[ARM1-1/4] vertical clear skipped")

        robot1.move_goal(args.move_command, approach_x, approach_y, pre_z, tool_t, float(args.arm1_approach_speed))
        _wait_for_arm1_waypoint("ARM1-2/4-LATERAL-SAFE", approach_x, approach_y, pre_z)
        robot1.move_goal(args.move_command, approach_x, approach_y, target_z, tool_t, float(args.arm1_approach_speed))
        _wait_for_arm1_waypoint("ARM1-3/4-DESCEND", approach_x, approach_y, target_z)
        robot1.move_goal(args.move_command, target_x, target_y, target_z, tool_t, float(args.arm1_insert_speed))
        final_pose = _wait_for_arm1_waypoint("ARM1-4/4-SLOW-INSERT", target_x, target_y, target_z)

        print("\n========== ARM1 FINAL POSITION REACHED ==========")
        print(f"ARM1 final=({final_pose[0]:.2f},{final_pose[1]:.2f},{final_pose[2]:.2f})")
        print("ARM2 remains CLOSED and stationary. Starting ARM1 gripper-only grasp test.")
        print("=================================================\n")

        # Mirror the V24 ARM2 post-contact grasp sequence only:
        # widen to the same configured post-contact opening, settle, then close
        # to the same grasp angle. No torque-based success judgment and no arm motion
        # is allowed before the deliberate post-grasp diagonal rise in FIX8.
        arm1_wider_angle = _gripper_angle(
            args.post_contact_open_percent,
            args.grip_fully_open,
            args.grip_fully_closed,
        )
        print(
            f"[ARM1-GRASP:1/2] widen to {float(args.post_contact_open_percent):.1f}% "
            f"open angle={arm1_wider_angle:.3f}; settle={float(args.post_contact_open_settle_sec):.2f}s"
        )
        robot1.gripper_open(arm1_wider_angle, args.grip_spd, args.grip_acc)
        _interruptible_wait(args.post_contact_open_settle_sec, "ARM1 grasp widen")

        open_fb = robot1.feedback_retry(args.feedback_timeout, attempts=3, retry_delay=0.20)
        if open_fb is not None and "t" in open_fb:
            print(
                f"[ARM1-GRASP-OPEN-FEEDBACK] target={arm1_wider_angle:.3f} "
                f"actual={float(open_fb['t']):.3f}"
            )
        else:
            print("[ARM1-GRASP-OPEN-FEEDBACK] unavailable; continuing gripper-only test")

        print(
            f"[ARM1-GRASP:2/2] close to V24 grasp angle={float(args.grasp_close_angle):.3f}; "
            f"settle={float(args.grasp_close_settle_sec):.2f}s"
        )
        robot1.gripper_open(float(args.grasp_close_angle), args.grip_spd, args.grip_acc)
        _interruptible_wait(args.grasp_close_settle_sec, "ARM1 grasp close")

        close_fb = robot1.feedback_retry(args.feedback_timeout, attempts=3, retry_delay=0.20)
        if close_fb is not None and "t" in close_fb:
            print(
                f"[ARM1-GRASP-CLOSE-FEEDBACK] target={float(args.grasp_close_angle):.3f} "
                f"actual={float(close_fb['t']):.3f}"
            )
        else:
            print("[ARM1-GRASP-CLOSE-FEEDBACK] unavailable")

        # FIX8 pre-spread post-grasp motion: ARM2 stays completely fixed. ARM1 alone moves
        # outward toward its board side while rising to the physical height of ARM2.
        # The XY separation is front-loaded to clear the ARM2 vertical axis early.
        close_pose = np.asarray(final_pose, dtype=np.float64)
        if close_fb is not None and all(k in close_fb for k in ("x", "y", "z")):
            close_pose = np.asarray(
                [float(close_fb["x"]), float(close_fb["y"]), float(close_fb["z"])],
                dtype=np.float64,
            )

        rise_start_z = float(close_pose[2])
        rise_final_z = float(arm2_pose[2]) + float(args.arm1_z_calibration_offset_mm)
        _require_in_range("ARM1 rise_final_z", rise_final_z, args.z_min, args.z_max)
        if rise_final_z <= rise_start_z + 5.0:
            raise RuntimeError(
                f"ARM1 rise target is not meaningfully above grasp pose: "
                f"start_z={rise_start_z:.2f} final_z={rise_final_z:.2f}"
            )

        requested_spread = max(0.0, float(args.arm1_rise_spread_mm))
        max_spread = max(0.0, (bx1 - 25.0) - float(center_board[0]))
        spread_mm = min(requested_spread, max_spread)
        if spread_mm < requested_spread - 1e-6:
            print(
                f"[ARM1-RISE] requested spread {requested_spread:.1f}mm clipped to "
                f"{spread_mm:.1f}mm by board +X safety margin"
            )

        rise_fracs = (40.0 / 150.0, 85.0 / 150.0, 125.0 / 150.0, 1.0)
        spread_fracs = (0.35, 0.65, 0.85, 1.0)
        rise_waypoints = []
        for index, (rise_frac, spread_frac) in enumerate(zip(rise_fracs, spread_fracs), 1):
            board_x = float(center_board[0]) + spread_mm * spread_frac
            board_y = float(center_board[1])
            wp_x, wp_y = _board_to_arm1_xy(board_x, board_y)
            wp_z = rise_start_z + (rise_final_z - rise_start_z) * rise_frac
            _require_in_range(f"ARM1 rise waypoint {index} z", wp_z, args.z_min, args.z_max)
            rise_waypoints.append((wp_x, wp_y, wp_z, board_x, board_y, spread_mm * spread_frac))

        print("\n========== ARM1 POST-GRASP DIAGONAL RISE ==========")
        print(
            f"start local=({close_pose[0]:.2f},{close_pose[1]:.2f},{close_pose[2]:.2f}) "
            f"-> final physical-height target Z={rise_final_z:.2f}"
        )
        print(
            f"outward board +X spread={spread_mm:.1f}mm; speed={float(args.arm1_rise_speed):.2f}; "
            "ARM2 XYZ/gripper remain fixed"
        )
        print("waypoints use front-loaded XY separation: 35% -> 65% -> 85% -> 100%")
        print("===================================================\n")

        current_rise_pose = close_pose.copy()
        for index, (wp_x, wp_y, wp_z, board_x, board_y, spread_now) in enumerate(rise_waypoints, 1):
            print(
                f"[ARM1-RISE:{index}/4] board=({board_x:.2f},{board_y:.2f}) "
                f"spread={spread_now:.1f}mm local=({wp_x:.2f},{wp_y:.2f},{wp_z:.2f})"
            )
            robot1.move_goal(
                args.move_command,
                wp_x,
                wp_y,
                wp_z,
                float(args.grasp_close_angle),
                float(args.arm1_rise_speed),
            )
            current_rise_pose = _wait_for_arm1_waypoint(
                f"ARM1-RISE-{index}/4", wp_x, wp_y, wp_z
            )

        print("\n========== DUAL HANG POSITION REACHED ==========")
        print(
            f"ARM1 rise final=({current_rise_pose[0]:.2f},{current_rise_pose[1]:.2f},"
            f"{current_rise_pose[2]:.2f}) CLOSED"
        )
        print(
            f"ARM2 handoff actual=({arm2_pose[0]:.2f},{arm2_pose[1]:.2f},{arm2_pose[2]:.2f}) CLOSED"
        )
        print("ARM2 remained stationary throughout ARM1 rise. Starting additional synchronized air spread.")
        print("================================================\n")

        # FIX8: once both arms securely hold the garment at the same physical
        # height, move both outward by the same small board-X distance. The two
        # T:104 commands are released through one barrier so neither arm pulls
        # materially earlier than the other.
        spread_each = max(0.0, float(args.dual_air_spread_each_mm))
        arm2_start_board_x = float(center_board[0])
        arm1_start_board_x = float(center_board[0]) + float(spread_mm)
        arm2_target_board_x = arm2_start_board_x - spread_each
        arm1_target_board_x = arm1_start_board_x + spread_each
        safe_min_x = float(bx0) + 25.0
        safe_max_x = float(bx1) - 25.0
        if arm2_target_board_x < safe_min_x or arm1_target_board_x > safe_max_x:
            raise RuntimeError(
                f"dual air spread leaves board safety margin: "
                f"ARM2 bx={arm2_target_board_x:.1f} min={safe_min_x:.1f}, "
                f"ARM1 bx={arm1_target_board_x:.1f} max={safe_max_x:.1f}"
            )

        arm2_spread_x, arm2_spread_y = _board_to_arm2_xy(arm2_target_board_x, float(center_board[1]))
        arm1_spread_x, arm1_spread_y = _board_to_arm1_xy(arm1_target_board_x, float(center_board[1]))
        arm2_fb = arm.feedback_retry(args.feedback_timeout, attempts=3, retry_delay=0.20)
        arm1_fb = robot1.feedback_retry(args.feedback_timeout, attempts=3, retry_delay=0.20)
        if arm2_fb is None or not all(k in arm2_fb for k in ("x", "y", "z")):
            raise RuntimeError("ARM2 feedback unavailable before dual air spread")
        if arm1_fb is None or not all(k in arm1_fb for k in ("x", "y", "z")):
            raise RuntimeError("ARM1 feedback unavailable before dual air spread")
        arm2_spread_z = float(arm2_fb["z"])
        arm1_spread_z = float(arm1_fb["z"])

        print("\n========== SYNCHRONIZED DUAL AIR SPREAD ==========")
        print(
            f"each-arm outward={spread_each:.1f}mm | total extra separation={2.0 * spread_each:.1f}mm | "
            f"speed={float(args.dual_air_spread_speed):.2f}"
        )
        print(
            f"ARM2 board ({arm2_start_board_x:.1f},{float(center_board[1]):.1f}) -> "
            f"({arm2_target_board_x:.1f},{float(center_board[1]):.1f})"
        )
        print(
            f"ARM1 board ({arm1_start_board_x:.1f},{float(center_board[1]):.1f}) -> "
            f"({arm1_target_board_x:.1f},{float(center_board[1]):.1f})"
        )
        print("Both grippers remain CLOSED during this spread stage. FIX11 laydown follows only after both arrivals are verified.")
        print("==================================================\n")

        command_barrier = threading.Barrier(3)
        command_errors = []
        command_lock = threading.Lock()

        def _spread_worker(label, robot_obj, x, y, z):
            try:
                command_barrier.wait(timeout=3.0)
                robot_obj.move_goal(
                    args.move_command, x, y, z,
                    float(args.grasp_close_angle),
                    float(args.dual_air_spread_speed),
                )
            except BaseException as exc:
                with command_lock:
                    command_errors.append((label, exc))

        thread2 = threading.Thread(
            target=_spread_worker,
            args=("ARM2", arm, arm2_spread_x, arm2_spread_y, arm2_spread_z),
            daemon=True,
        )
        thread1 = threading.Thread(
            target=_spread_worker,
            args=("ARM1", robot1, arm1_spread_x, arm1_spread_y, arm1_spread_z),
            daemon=True,
        )
        thread2.start()
        thread1.start()
        try:
            command_barrier.wait(timeout=3.0)
        except threading.BrokenBarrierError as exc:
            command_errors.append(("BARRIER", exc))
        thread2.join(timeout=5.0)
        thread1.join(timeout=5.0)
        if thread2.is_alive() or thread1.is_alive():
            raise RuntimeError("dual air-spread command thread timeout")
        if command_errors:
            label, exc = command_errors[0]
            raise RuntimeError(f"dual air-spread command failed at {label}: {exc}")

        arm2_spread_pose = wait_for_waypoint(
            "DUAL-SPREAD-ARM2",
            arm2_spread_x, arm2_spread_y, arm2_spread_z,
            allow_abort=True,
        )
        arm1_spread_pose = _wait_for_arm1_waypoint(
            "DUAL-SPREAD-ARM1",
            arm1_spread_x, arm1_spread_y, arm1_spread_z,
        )
        _interruptible_wait(args.dual_air_spread_settle_sec, "dual air spread settle")

        print("\n========== DUAL AIR SPREAD COMPLETE ==========")
        print(
            f"ARM2 final=({arm2_spread_pose[0]:.2f},{arm2_spread_pose[1]:.2f},"
            f"{arm2_spread_pose[2]:.2f}) CLOSED"
        )
        print(
            f"ARM1 final=({arm1_spread_pose[0]:.2f},{arm1_spread_pose[1]:.2f},"
            f"{arm1_spread_pose[2]:.2f}) CLOSED"
        )
        print(
            f"initial ARM1 rise separation={spread_mm:.1f}mm; "
            f"additional spread={spread_each:.1f}mm per arm"
        )
        if bool(args.dual_arc_laydown):
            print("Both arms remain CLOSED; FIX11 will now preflight pair recenter and curved laydown.")
        else:
            print("Both arms remain CLOSED and stationary for visual inspection (--no-dual-arc-laydown).")
            print("Ctrl+C will slightly open both connected grippers and return them to taught standby.")
        print("==============================================\n")
        return {
            "arm2_pose": np.asarray(arm2_spread_pose, dtype=np.float64),
            "arm1_pose": np.asarray(arm1_spread_pose, dtype=np.float64),
            "arm2_board_command": np.asarray(
                [arm2_target_board_x, float(center_board[1])], dtype=np.float64
            ),
            "arm1_board_command": np.asarray(
                [arm1_target_board_x, float(center_board[1])], dtype=np.float64
            ),
        }

    post_mask_has_run = False

    def _post_mask_parse_conf_ladder() -> List[float]:
        values: List[float] = []
        seen = set()
        for token in str(args.post_mask_conf_ladder or "").split(","):
            try:
                value = float(token.strip())
            except Exception:
                continue
            value = float(np.clip(value, 0.005, 0.95))
            key = round(value, 5)
            if key not in seen:
                seen.add(key)
                values.append(value)
        return values or [0.12, 0.07, 0.03]

    def _post_mask_load_homography():
        """Load both homographies while accepting the metadata-less V5 legacy cache.

        V7 incorrectly treated missing descriptive camera_geometry fields as a
        fatal condition even when the actual H/raw_H matrices were present and
        valid.  V8 validates the numerical transforms first.  Existing metadata
        is still enforced, but absent fields are filled later from the prepared
        CameraUndistorter runtime metadata.
        """
        resolved = _resolve_runtime_file(args.post_mask_hfile)
        print(f"[POST-MASK-H-PATH] requested={args.post_mask_hfile} resolved={resolved}")
        data = _load_json_dict(resolved)
        if "H" not in data or "raw_H" not in data:
            raise RuntimeError(
                f"ELP Homography bundle must contain both H and raw_H: {resolved}"
            )
        H_value = np.asarray(data["H"], dtype=np.float64)
        raw_H_value = np.asarray(data["raw_H"], dtype=np.float64)
        if H_value.shape != (3, 3) or raw_H_value.shape != (3, 3):
            raise RuntimeError(
                f"H/raw_H must both be 3x3: H={H_value.shape} raw_H={raw_H_value.shape}"
            )
        if not np.all(np.isfinite(H_value)) or not np.all(np.isfinite(raw_H_value)):
            raise RuntimeError("H/raw_H contain NaN or infinity")
        det_h = float(np.linalg.det(H_value))
        det_raw_h = float(np.linalg.det(raw_H_value))
        if abs(det_h) <= 1e-12 or abs(det_raw_h) <= 1e-12:
            raise RuntimeError(
                f"H/raw_H are singular: detH={det_h:.6e} detRawH={det_raw_h:.6e}"
            )

        metadata = dict(data.get("camera_geometry", {}) or {})
        required = ("undistort_enabled", "calibration_id", "output_size", "alpha")
        missing = [key for key in required if key not in metadata]
        if "undistort_enabled" in metadata and not bool(metadata.get("undistort_enabled")):
            raise RuntimeError(
                "post-mask Homography explicitly declares undistort_enabled=false"
            )

        metadata["_legacy_compat"] = bool(missing)
        metadata["_legacy_missing_fields"] = list(missing)
        metadata["_matrix_validation"] = {
            "det_H": det_h,
            "det_raw_H": det_raw_h,
            "finite": True,
            "shape": [3, 3],
        }
        if missing:
            print(
                "[POST-MASK-H-LEGACY-COMPAT] camera_geometry fields are absent "
                f"{missing}; H/raw_H are numerically valid, so runtime camera "
                "metadata will be used instead of aborting"
            )
        else:
            print(
                f"[POST-MASK-H-METADATA] calibration_id={metadata.get('calibration_id')} "
                f"output_size={metadata.get('output_size')} alpha={metadata.get('alpha')}"
            )
        print(
            f"[POST-MASK-H] loaded={resolved} raw_H=YES corrected_H=YES "
            f"detH={det_h:.6e} detRawH={det_raw_h:.6e} "
            f"legacyCompat={bool(missing)}"
        )
        return (
            resolved,
            H_value.astype(np.float32),
            raw_H_value.astype(np.float32),
            metadata,
        )

    def _post_mask_project_board_points(raw_H_value: np.ndarray, board_points):
        projected: List[List[int]] = []
        for board_x, board_y in board_points:
            pixel = _board_to_pixel_h(raw_H_value, float(board_x), float(board_y))
            if pixel is None or not np.all(np.isfinite(pixel)):
                raise RuntimeError(
                    f"raw_H board projection failed for ({board_x},{board_y})"
                )
            projected.append([int(round(pixel[0])), int(round(pixel[1]))])
        return np.asarray(projected, dtype=np.int32)

    def _post_mask_board_roi(image_shape, raw_H_value: np.ndarray):
        """Build exact raw-camera plate masks from the four ArUco board points."""
        h, w = image_shape[:2]
        marker_map = board_target.get("marker_board_mm", {})
        board_points: List[List[float]] = []
        for marker_id in ("0", "1", "2", "3"):
            raw = marker_map.get(marker_id)
            if not isinstance(raw, (list, tuple)) or len(raw) < 2:
                raise RuntimeError(
                    f"marker_board_mm is missing folding-plate corner ID {marker_id}"
                )
            board_points.append([float(raw[0]), float(raw[1])])
        board_points_np = np.asarray(board_points, dtype=np.float32)
        board_polygon_mm = cv2.convexHull(
            board_points_np.reshape(-1, 1, 2)
        ).reshape(-1, 2)
        if board_polygon_mm.shape[0] != 4:
            raise RuntimeError(
                f"folding-plate ArUco polygon must have four corners: "
                f"{board_polygon_mm.tolist()}"
            )

        # Construct the plate and its physical inset in board-millimeter space.
        # This avoids a perspective-dependent raw-pixel erosion and keeps the
        # stability ROI tied to the actual folding plate geometry.
        scale = 2.0  # board-mask pixels per millimeter
        padding = 4.0
        min_x = float(np.min(board_polygon_mm[:, 0]))
        max_x = float(np.max(board_polygon_mm[:, 0]))
        min_y = float(np.min(board_polygon_mm[:, 1]))
        max_y = float(np.max(board_polygon_mm[:, 1]))
        canvas_w = max(32, int(math.ceil((max_x - min_x) * scale + 2.0 * padding + 1.0)))
        canvas_h = max(32, int(math.ceil((max_y - min_y) * scale + 2.0 * padding + 1.0)))
        board_to_canvas = np.asarray(
            [
                [scale, 0.0, -min_x * scale + padding],
                [0.0, scale, -min_y * scale + padding],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        polygon_canvas = cv2.perspectiveTransform(
            board_polygon_mm.reshape(-1, 1, 2),
            board_to_canvas.astype(np.float32),
        ).reshape(-1, 2)
        full_canvas = np.zeros((canvas_h, canvas_w), dtype=np.uint8)
        cv2.fillConvexPoly(
            full_canvas, np.round(polygon_canvas).astype(np.int32), 255
        )

        inset_mm = max(0.0, float(args.post_mask_roi_inset_mm))
        stability_canvas = full_canvas.copy()
        if inset_mm > 0.0:
            radius = max(1, int(round(inset_mm * scale)))
            kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (radius * 2 + 1, radius * 2 + 1)
            )
            stability_canvas = cv2.erode(full_canvas, kernel)
        if cv2.countNonZero(stability_canvas) < 1000:
            raise RuntimeError(
                f"post-mask board inset {inset_mm:.1f}mm removed the plate ROI"
            )

        raw_to_canvas = board_to_canvas @ np.asarray(raw_H_value, dtype=np.float64)
        if abs(float(np.linalg.det(raw_to_canvas))) < 1e-12:
            raise RuntimeError("raw_H to board-mask transform is singular")
        canvas_to_raw = np.linalg.inv(raw_to_canvas)
        full_mask = cv2.warpPerspective(
            full_canvas,
            canvas_to_raw,
            (w, h),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        stability_mask = cv2.warpPerspective(
            stability_canvas,
            canvas_to_raw,
            (w, h),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )

        extra_inset_px = max(0, int(args.post_mask_roi_inset_px))
        if extra_inset_px > 0:
            kernel_size = extra_inset_px * 2 + 1
            eroded = cv2.erode(
                stability_mask,
                cv2.getStructuringElement(
                    cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)
                ),
            )
            if cv2.countNonZero(eroded) >= 1000:
                stability_mask = eroded

        full_polygon = cv2.convexHull(
            _post_mask_project_board_points(
                raw_H_value, board_polygon_mm.tolist()
            ).reshape(-1, 1, 2)
        ).reshape(-1, 2)
        if abs(float(cv2.contourArea(full_polygon.reshape(-1, 1, 2)))) < 1000.0:
            raise RuntimeError(
                f"projected folding-plate polygon is too small: {full_polygon.tolist()}"
            )

        marker_pixels: List[List[int]] = []
        marker_radius = max(0, int(args.post_mask_marker_exclusion_radius_px))
        for marker_id in ("0", "1", "2", "3"):
            raw = marker_map[marker_id]
            point = _post_mask_project_board_points(
                raw_H_value, [(float(raw[0]), float(raw[1]))]
            )[0]
            marker_pixels.append([int(point[0]), int(point[1])])
            if marker_radius > 0:
                cv2.circle(
                    stability_mask,
                    (int(point[0]), int(point[1])),
                    marker_radius,
                    0,
                    -1,
                    cv2.LINE_AA,
                )

        contours, _hierarchy = cv2.findContours(
            stability_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            raise RuntimeError("exact folding-plate stability ROI has no contour")
        stability_polygon = cv2.convexHull(
            max(contours, key=cv2.contourArea)
        ).reshape(-1, 2)

        full_pixels = int(cv2.countNonZero(full_mask))
        stable_pixels = int(cv2.countNonZero(stability_mask))
        if stable_pixels < 1000:
            raise RuntimeError(
                f"exact folding-plate stability ROI is too small: {stable_pixels}px"
            )
        roi_info = {
            "mode": "raw_H_exact_aruco_folding_plate_polygon",
            "board_polygon_mm": board_polygon_mm.astype(float).tolist(),
            "board_bounds_mm": [min_x, max_x, min_y, max_y],
            "board_mask_scale_px_per_mm": float(scale),
            "inset_mm": float(inset_mm),
            "extra_inset_px": int(extra_inset_px),
            "marker_exclusion_radius_px": int(marker_radius),
            "marker_exclusion": bool(marker_radius > 0),
            "full_plate_pixels": int(full_pixels),
            "stability_pixels": int(stable_pixels),
            "coverage_ratio": float(stable_pixels) / float(max(1, full_pixels)),
            "full_polygon_raw_uv": full_polygon.astype(int).tolist(),
            "stability_polygon_raw_uv": stability_polygon.astype(int).tolist(),
        }
        print(
            "[POST-MASK-STABILITY-ROI] "
            f"mode={roi_info['mode']} polygon_mm={roi_info['board_polygon_mm']} "
            f"inset_mm={inset_mm:.1f} marker_exclusion={roi_info['marker_exclusion']} "
            f"valid_pixels={stable_pixels} full_pixels={full_pixels} "
            f"coverage={roi_info['coverage_ratio']:.3f}"
        )
        return (
            stability_mask,
            full_mask,
            full_polygon.astype(np.int32),
            stability_polygon.astype(np.int32),
            marker_pixels,
            roi_info,
        )

    def _post_mask_signature(raw_frame: np.ndarray, raw_roi_u8: np.ndarray):
        """Compute structure only after cropping/masking the exact plate ROI."""
        gray = cv2.cvtColor(raw_frame, cv2.COLOR_BGR2GRAY)
        x, y, bw, bh = cv2.boundingRect(raw_roi_u8)
        if bw <= 4 or bh <= 4:
            raise RuntimeError("exact folding-plate stability ROI has no valid crop")
        crop_gray = gray[y:y + bh, x:x + bw].copy()
        crop_mask = raw_roi_u8[y:y + bh, x:x + bw].copy()
        valid = crop_mask > 0
        if int(np.count_nonzero(valid)) < 100:
            raise RuntimeError("exact folding-plate stability ROI has too few pixels")
        fill_value = int(round(float(np.median(crop_gray[valid]))))
        crop_gray[~valid] = fill_value

        sw = max(48, int(args.post_mask_signature_width))
        sh = max(36, int(args.post_mask_signature_height))
        small_gray = cv2.resize(crop_gray, (sw, sh), interpolation=cv2.INTER_AREA)
        small_mask = cv2.resize(crop_mask, (sw, sh), interpolation=cv2.INTER_NEAREST)
        base = cv2.GaussianBlur(small_gray, (5, 5), 0)
        local_base = cv2.GaussianBlur(base, (21, 21), 0)
        local_contrast = cv2.absdiff(base, local_base).astype(np.float32)
        gx = cv2.Sobel(base, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(base, cv2.CV_32F, 0, 1, ksize=3)
        gradient = cv2.magnitude(gx, gy)
        structural = np.clip(
            3.0 * local_contrast + 0.75 * gradient,
            0.0,
            255.0,
        ).astype(np.uint8)
        structural[small_mask == 0] = 0
        return structural, small_mask

    def _post_mask_signature_delta(sig_a, sig_b) -> float:
        if sig_a is None or sig_b is None:
            return float("inf")
        structural_a, mask_a = sig_a
        structural_b, mask_b = sig_b
        if structural_a.shape != structural_b.shape:
            return float("inf")
        valid = (mask_a > 0) & (mask_b > 0)
        count = int(np.count_nonzero(valid))
        if count < 100:
            return float("inf")
        diff = cv2.absdiff(structural_a, structural_b)[valid].astype(np.float32)
        mean_abs = float(np.mean(diff))
        p85 = float(np.percentile(diff, 85.0))
        p95 = float(np.percentile(diff, 95.0))
        p99 = float(np.percentile(diff, 99.0))
        top_count = min(int(diff.size), max(32, int(round(0.02 * float(diff.size)))))
        top_mean = (
            float(np.mean(np.partition(diff, diff.size - top_count)[-top_count:]))
            if top_count > 0 else 0.0
        )
        changed_ratio = float(np.count_nonzero(diff >= 12.0)) / float(max(1, diff.size))
        return (
            0.30 * mean_abs
            + 0.15 * p85
            + 0.20 * p95
            + 0.15 * p99
            + 0.15 * top_mean
            + 5.0 * changed_ratio
        )

    def _post_mask_brightness_stats(frame: np.ndarray, roi_mask: np.ndarray):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        values = gray[roi_mask > 0]
        if values.size < 1:
            return {
                "roi_mean": None, "roi_median": None, "roi_p95": None,
                "roi_p99": None, "saturated_ratio": None,
                "dark_ratio": None, "global_mean": float(np.mean(gray)),
            }
        return {
            "roi_mean": float(np.mean(values)),
            "roi_median": float(np.median(values)),
            "roi_p95": float(np.percentile(values, 95.0)),
            "roi_p99": float(np.percentile(values, 99.0)),
            "saturated_ratio": float(np.count_nonzero(values >= 250)) / float(values.size),
            "dark_ratio": float(np.count_nonzero(values <= 5)) / float(values.size),
            "global_mean": float(np.mean(gray)),
        }

    def _post_mask_overlay_roi(
        frame: np.ndarray,
        full_polygon: np.ndarray,
        stability_polygon: np.ndarray,
        marker_pixels,
    ) -> np.ndarray:
        overlay = frame.copy()
        tint = frame.copy()
        cv2.fillConvexPoly(tint, full_polygon.astype(np.int32), (255, 120, 0))
        cv2.addWeighted(tint, 0.12, overlay, 0.88, 0.0, overlay)
        cv2.polylines(overlay, [full_polygon.reshape(-1, 1, 2)], True, (255, 180, 0), 3)
        cv2.polylines(
            overlay, [stability_polygon.reshape(-1, 1, 2)], True, (0, 255, 0), 3
        )
        for point in marker_pixels:
            cv2.circle(
                overlay,
                (int(point[0]), int(point[1])),
                max(2, int(args.post_mask_marker_exclusion_radius_px)),
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )
        cv2.putText(
            overlay,
            "BLUE=full plate / GREEN=stability ROI / RED=marker exclusion",
            (18, 32),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        return overlay

    def _post_mask_write_stability_diagnostics(
        run_dir: Path,
        *,
        trace,
        thresholds,
        roi_info,
        stability_mask,
        full_mask,
        full_polygon,
        stability_polygon,
        marker_pixels,
        frames,
        status: str,
        stable_accumulated: float,
    ) -> Dict[str, Any]:
        run_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(run_dir / "stability_roi_mask.png"), stability_mask)
        cv2.imwrite(str(run_dir / "segmentation_roi_mask.png"), full_mask)
        reference = frames.get("first")
        if reference is None:
            reference = frames.get("last")
        if reference is not None:
            cv2.imwrite(
                str(run_dir / "stability_roi_overlay.png"),
                _post_mask_overlay_roi(
                    reference, full_polygon, stability_polygon, marker_pixels
                ),
            )
            segmentation_overlay = reference.copy()
            cv2.polylines(
                segmentation_overlay,
                [full_polygon.reshape(-1, 1, 2)],
                True,
                (0, 255, 255),
                3,
            )
            cv2.imwrite(
                str(run_dir / "segmentation_roi_overlay.png"), segmentation_overlay
            )

        for key in ("first", "best", "last", "max_delta"):
            frame = frames.get(key)
            if frame is None:
                continue
            cv2.imwrite(str(run_dir / f"stability_{key}_raw.png"), frame)
            roi_frame = cv2.bitwise_and(frame, frame, mask=stability_mask)
            cv2.imwrite(str(run_dir / f"stability_{key}_roi.png"), roi_frame)

        fieldnames = [
            "timestamp_monotonic", "sequence", "phase", "raw_delta",
            "decision_delta", "state", "stable_accumulated",
            "roi_mean", "roi_median", "roi_p95", "roi_p99",
            "saturated_ratio", "dark_ratio", "global_mean",
            "global_brightness_shift",
        ]
        with (run_dir / "stability_trace.csv").open(
            "w", encoding="utf-8", newline=""
        ) as fp:
            writer = csv.DictWriter(fp, fieldnames=fieldnames)
            writer.writeheader()
            for row in trace:
                writer.writerow({key: row.get(key) for key in fieldnames})

        finite_decisions = [
            float(row["decision_delta"])
            for row in trace
            if row.get("decision_delta") is not None
            and np.isfinite(float(row["decision_delta"]))
        ]
        finite_raw = [
            float(row["raw_delta"])
            for row in trace
            if row.get("raw_delta") is not None
            and np.isfinite(float(row["raw_delta"]))
        ]
        diagnostics = {
            "status": str(status),
            "thresholds": thresholds,
            "roi": roi_info,
            "trace_count": int(len(trace)),
            "best_decision_delta": min(finite_decisions) if finite_decisions else None,
            "last_decision_delta": finite_decisions[-1] if finite_decisions else None,
            "max_raw_delta": max(finite_raw) if finite_raw else None,
            "stable_accumulated_sec": float(stable_accumulated),
            "files": {
                "trace": "stability_trace.csv",
                "roi_mask": "stability_roi_mask.png",
                "full_plate_mask": "segmentation_roi_mask.png",
                "roi_overlay": "stability_roi_overlay.png",
            },
        }
        with (run_dir / "stability_diagnostics.json").open(
            "w", encoding="utf-8"
        ) as fp:
            json.dump(diagnostics, fp, ensure_ascii=False, indent=2)
        print(
            f"[POST-MASK-STABILITY-DIAGNOSTICS] status={status} saved={run_dir}"
        )
        return diagnostics

    def _post_mask_capture_stable_core(
        read_next_frame,
        *,
        source_label: str,
        raw_H_value: np.ndarray,
        run_dir: Path,
    ):
        warmup = max(1, int(args.post_mask_warmup_frames))
        latest = None
        latest_sequence = -1
        for _index in range(1, warmup + 1):
            latest, latest_sequence, _timestamp = read_next_frame()
        if latest is None:
            raise RuntimeError(f"{source_label} produced no warmup frame")

        (
            stability_mask,
            full_mask,
            full_polygon,
            stability_polygon,
            marker_pixels,
            roi_info,
        ) = _post_mask_board_roi(latest.shape, raw_H_value)

        trace: List[Dict[str, Any]] = []
        frames: Dict[str, Optional[np.ndarray]] = {
            "first": latest.copy(),
            "best": None,
            "last": latest.copy(),
            "max_delta": None,
        }
        previous_signature = _post_mask_signature(latest, stability_mask)
        previous_global_mean = _post_mask_brightness_stats(
            latest, stability_mask
        )["global_mean"]
        calibration_deltas: List[float] = []
        calibration_started = time.monotonic()
        calibration_duration = max(0.8, float(args.post_mask_noise_calibration_sec))
        min_samples = max(5, int(args.post_mask_noise_min_samples))
        max_calibration_duration = max(
            calibration_duration, min_samples / max(1.0, float(args.post_mask_fps)) + 1.0
        )
        print(
            f"[POST-MASK-NOISE-CALIBRATION] source={source_label} "
            f"target={calibration_duration:.2f}s min_samples={min_samples} "
            f"ROIpx={cv2.countNonZero(stability_mask)}"
        )
        while (
            time.monotonic() - calibration_started < max_calibration_duration
            or len(calibration_deltas) < min_samples
        ):
            frame, sequence, timestamp_value = read_next_frame()
            current_signature = _post_mask_signature(frame, stability_mask)
            raw_delta = _post_mask_signature_delta(
                current_signature, previous_signature
            )
            previous_signature = current_signature
            stats = _post_mask_brightness_stats(frame, stability_mask)
            brightness_shift = float(stats["global_mean"] - previous_global_mean)
            previous_global_mean = float(stats["global_mean"])
            calibration_deltas.append(float(raw_delta))
            trace.append({
                "timestamp_monotonic": float(timestamp_value),
                "sequence": int(sequence),
                "phase": "noise_calibration",
                "raw_delta": float(raw_delta),
                "decision_delta": None,
                "state": "CALIBRATION",
                "stable_accumulated": 0.0,
                **stats,
                "global_brightness_shift": brightness_shift,
            })
            latest = frame.copy()
            latest_sequence = int(sequence)
            frames["last"] = latest.copy()
            if (
                time.monotonic() - calibration_started >= calibration_duration
                and len(calibration_deltas) >= min_samples
            ):
                break

        try:
            thresholds = _post_mask_robust_thresholds(
                calibration_deltas,
                low_percentile=float(args.post_mask_noise_low_percentile),
                min_samples=min_samples,
                calibration_max_delta=float(args.post_mask_noise_max_delta),
                stable_min=float(args.post_mask_stable_threshold_min),
                stable_max=float(args.post_mask_stable_threshold_max),
                stable_margin=float(args.post_mask_stable_margin),
                stable_mad_scale=float(args.post_mask_stable_mad_scale),
                motion_min=float(args.post_mask_motion_threshold_min),
                motion_max=float(args.post_mask_motion_threshold_max),
                motion_gap=float(args.post_mask_motion_gap),
                motion_mad_scale=float(args.post_mask_motion_mad_scale),
                fixed_stable_threshold=float(args.post_mask_stable_threshold),
            )
        except Exception as exc:
            threshold_failure = {
                "calibration_error": repr(exc),
                "sample_count": int(len(calibration_deltas)),
                "all_deltas": [float(v) for v in calibration_deltas],
            }
            _post_mask_write_stability_diagnostics(
                run_dir,
                trace=trace,
                thresholds=threshold_failure,
                roi_info=roi_info,
                stability_mask=stability_mask,
                full_mask=full_mask,
                full_polygon=full_polygon,
                stability_polygon=stability_polygon,
                marker_pixels=marker_pixels,
                frames=frames,
                status="NOISE_CALIBRATION_FAILED",
                stable_accumulated=0.0,
            )
            raise RuntimeError(
                f"post-mask noise calibration failed: {exc}"
            ) from exc
        stable_threshold = float(thresholds["stable_threshold"])
        motion_threshold = float(thresholds["motion_threshold"])
        print(
            f"[POST-MASK-ADAPTIVE-THRESHOLD] noiseMedian={thresholds['noise_median']:.3f} "
            f"MAD={thresholds['noise_mad']:.3f} sigma={thresholds['robust_sigma']:.3f} "
            f"stable={stable_threshold:.3f} motion={motion_threshold:.3f} "
            f"configuredStableMax={thresholds['stable_upper_configured']:.3f} "
            f"effectiveStableMax={thresholds['stable_upper_effective']:.3f} "
            f"fixedOverride={thresholds['fixed_stable_override']}"
        )

        window_size = max(3, int(args.post_mask_delta_median_window))
        if window_size % 2 == 0:
            window_size += 1
        recent_deltas = deque(maxlen=window_size)
        required_stable = max(0.0, float(args.post_mask_stable_sec))
        ambiguous_hold = max(0.0, float(args.post_mask_ambiguous_hold_sec))
        timeout = max(1.0, float(args.post_mask_timeout_sec))
        started = time.monotonic()
        previous_time = started
        stable_accumulated = 0.0
        ambiguous_accumulated = 0.0
        last_log = 0.0
        best_decision = float("inf")
        max_raw_delta = -float("inf")

        print(
            f"[POST-MASK-STABILITY] source={source_label} warmup={warmup} "
            f"ROIpx={cv2.countNonZero(stability_mask)} stableThreshold={stable_threshold:.3f} "
            f"motionThreshold={motion_threshold:.3f} medianWindow={window_size} "
            f"required={required_stable:.2f}s timeout={timeout:.1f}s"
        )
        while time.monotonic() - started < timeout:
            frame, sequence, timestamp_value = read_next_frame()
            now = time.monotonic()
            dt = max(0.0, now - previous_time)
            previous_time = now
            current_signature = _post_mask_signature(frame, stability_mask)
            raw_delta = _post_mask_signature_delta(
                current_signature, previous_signature
            )
            previous_signature = current_signature
            recent_deltas.append(float(raw_delta))
            decision_delta = float(np.median(np.asarray(recent_deltas, dtype=np.float64)))

            if decision_delta <= stable_threshold:
                state = "STABLE"
                stable_accumulated += dt
                ambiguous_accumulated = 0.0
            elif decision_delta < motion_threshold:
                state = "AMBIGUOUS"
                ambiguous_accumulated += dt
                if ambiguous_accumulated > ambiguous_hold:
                    stable_accumulated = 0.0
            else:
                state = "MOTION"
                stable_accumulated = 0.0
                ambiguous_accumulated = 0.0

            stats = _post_mask_brightness_stats(frame, stability_mask)
            brightness_shift = float(stats["global_mean"] - previous_global_mean)
            previous_global_mean = float(stats["global_mean"])
            row = {
                "timestamp_monotonic": float(timestamp_value),
                "sequence": int(sequence),
                "phase": "stability_gate",
                "raw_delta": float(raw_delta),
                "decision_delta": float(decision_delta),
                "state": state,
                "stable_accumulated": float(stable_accumulated),
                **stats,
                "global_brightness_shift": brightness_shift,
            }
            trace.append(row)
            latest = frame.copy()
            latest_sequence = int(sequence)
            frames["last"] = latest.copy()
            if decision_delta < best_decision:
                best_decision = decision_delta
                frames["best"] = latest.copy()
            if raw_delta > max_raw_delta:
                max_raw_delta = raw_delta
                frames["max_delta"] = latest.copy()

            if now - last_log >= 0.25:
                print(
                    f"[POST-MASK-STABILITY] seq={sequence} raw={raw_delta:.3f} "
                    f"decision={decision_delta:.3f} state={state} "
                    f"stable={stable_accumulated:.3f}/{required_stable:.3f}s "
                    f"brightnessShift={brightness_shift:+.3f} sat={stats['saturated_ratio']:.4f}"
                )
                last_log = now

            if state == "STABLE" and stable_accumulated >= required_stable:
                diagnostics = _post_mask_write_stability_diagnostics(
                    run_dir,
                    trace=trace,
                    thresholds=thresholds,
                    roi_info=roi_info,
                    stability_mask=stability_mask,
                    full_mask=full_mask,
                    full_polygon=full_polygon,
                    stability_polygon=stability_polygon,
                    marker_pixels=marker_pixels,
                    frames=frames,
                    status="STABLE_ACCEPTED",
                    stable_accumulated=stable_accumulated,
                )
                print(
                    f"[POST-MASK-STABLE] accepted latest RAW frame seq={sequence} "
                    f"after {now-started:.2f}s; decision={decision_delta:.3f}; "
                    f"stable={stable_accumulated:.3f}s"
                )
                return latest.copy(), full_mask, marker_pixels, {
                    "source": str(source_label),
                    "accepted_sequence": int(sequence),
                    "elapsed_sec": float(now - started),
                    "stable_sec": float(stable_accumulated),
                    "last_raw_delta": float(raw_delta),
                    "last_decision_delta": float(decision_delta),
                    "frame_count": int(
                        sum(1 for row_i in trace if row_i.get("phase") == "stability_gate")
                    ),
                    "thresholds": thresholds,
                    "roi": roi_info,
                    "diagnostics": diagnostics,
                }

        diagnostics = _post_mask_write_stability_diagnostics(
            run_dir,
            trace=trace,
            thresholds=thresholds,
            roi_info=roi_info,
            stability_mask=stability_mask,
            full_mask=full_mask,
            full_polygon=full_polygon,
            stability_polygon=stability_polygon,
            marker_pixels=marker_pixels,
            frames=frames,
            status="TIMEOUT",
            stable_accumulated=stable_accumulated,
        )
        raise RuntimeError(
            f"exact folding-plate ROI did not remain stable for {required_stable:.2f}s "
            f"within {timeout:.1f}s; lastDecision="
            f"{diagnostics.get('last_decision_delta')} bestDecision="
            f"{diagnostics.get('best_decision_delta')}"
        )

    def _post_mask_capture_stable_raw(
        post_cap, raw_H_value: np.ndarray, run_dir: Path
    ):
        sequence = 0

        def read_next():
            nonlocal sequence
            ok, frame_raw = post_cap.read()
            if not ok or frame_raw is None:
                raise RuntimeError("post-FIX11 camera frame read failed")
            sequence += 1
            return frame_raw.copy(), int(sequence), float(time.monotonic())

        return _post_mask_capture_stable_core(
            read_next,
            source_label="post_fix11_camera_open",
            raw_H_value=raw_H_value,
            run_dir=run_dir,
        )

    def _post_mask_capture_stable_raw_from_preview(
        preview_source: RawPreviewController,
        raw_H_value: np.ndarray,
        run_dir: Path,
    ):
        sequence = preview_source.current_sequence()

        def read_next():
            nonlocal sequence
            frame, sequence, timestamp_value = preview_source.wait_for_new_frame(
                sequence, timeout=2.0
            )
            return frame, int(sequence), float(timestamp_value)

        return _post_mask_capture_stable_core(
            read_next,
            source_label="continuous_raw_preview",
            raw_H_value=raw_H_value,
            run_dir=run_dir,
        )

    def _post_mask_prepare_e49_runtime(e49_module, H_value: np.ndarray):
        args._board_marker_map = {
            str(k): [float(v[0]), float(v[1])]
            for k, v in board_target.get("marker_board_mm", {}).items()
        }
        args._board_required_ids = [0, 1, 2, 3]
        args._board_roi_source = "dual_roarm_folding_board_config"
        args.board_roi = True
        args.board_roi_strict = True
        args.board_roi_expand_px = 0
        args.board_marker_exclusion_radius_px = int(args.post_mask_marker_exclusion_radius_px)
        args.board_frame_border_exclude_px = 8
        args.board_mask_open_px = 3
        args.board_mask_close_px = 5
        args.board_mask_min_component_px = 1200
        args.board_mask_max_frame_ratio = 0.72
        args.empty_baseline_veto = False
        args.empty_baseline_require = False
        args._empty_baseline_bgr = None
        args._empty_baseline_h = None
        args._empty_baseline_board_points = []
        args._empty_baseline_last_info = {}
        args._board_roi_last_info = {}
        return e49_module

    def _post_mask_pixel_to_board(
        H_value: np.ndarray,
        point_xy: Tuple[float, float],
    ) -> List[float]:
        src = np.asarray([[[float(point_xy[0]), float(point_xy[1])]]], dtype=np.float32)
        dst = cv2.perspectiveTransform(src, np.asarray(H_value, dtype=np.float32))
        return [float(dst[0, 0, 0]), float(dst[0, 0, 1])]

    def _post_mask_largest_component(mask_u8: np.ndarray) -> np.ndarray:
        binary = (np.asarray(mask_u8, dtype=np.uint8) > 0).astype(np.uint8)
        count, labels, stats, _centroids = cv2.connectedComponentsWithStats(binary, 8)
        if count <= 1:
            raise RuntimeError("grasp analysis requires a non-empty garment mask")
        largest_label = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        component = np.zeros_like(binary, dtype=np.uint8)
        component[labels == largest_label] = 255
        if cv2.countNonZero(component) < 500:
            raise RuntimeError("garment mask is too small for short-axis grasp analysis")
        return component

    def _post_mask_pick_axis_side_grasp(
        component: np.ndarray,
        distance_map: np.ndarray,
        center_xy: np.ndarray,
        axis_xy: np.ndarray,
        t_values: np.ndarray,
        inside_values: np.ndarray,
        side_sign: int,
    ) -> Tuple[np.ndarray, float, float, float]:
        side_indices = np.flatnonzero((t_values * float(side_sign) > 0.0) & inside_values)
        if side_indices.size == 0:
            raise RuntimeError(f"minor-axis mask intersection missing side={side_sign:+d}")

        # Use the outermost in-mask sample as the true garment edge on this side.
        if side_sign > 0:
            boundary_index = int(side_indices[np.argmax(t_values[side_indices])])
            inward_direction = -1
        else:
            boundary_index = int(side_indices[np.argmin(t_values[side_indices])])
            inward_direction = +1
        boundary_t = float(t_values[boundary_index])

        # Restrict the search to the same outer mask run. This prevents a trouser
        # crotch gap or another concavity from jumping the point to a different run.
        run_indices = [boundary_index]
        cursor = boundary_index + inward_direction
        while 0 <= cursor < len(t_values) and bool(inside_values[cursor]):
            run_indices.append(cursor)
            cursor += inward_direction
        run_indices = np.asarray(run_indices, dtype=np.int32)
        if run_indices.size < 2:
            raise RuntimeError(f"minor-axis outer run is too thin on side={side_sign:+d}")

        depths = np.abs(t_values[run_indices] - boundary_t)
        run_depth = float(np.max(depths))
        preferred = max(3.0, float(args.post_mask_grasp_edge_inset_px))
        search_span = max(2.0, float(args.post_mask_grasp_search_span_px))
        max_allowed_depth = max(3.0, min(run_depth * 0.70, preferred + search_span))
        min_allowed_depth = min(max_allowed_depth, max(2.0, preferred - search_span))

        candidate_mask = (depths >= min_allowed_depth) & (depths <= max_allowed_depth)
        candidate_indices = run_indices[candidate_mask]
        candidate_depths = depths[candidate_mask]
        if candidate_indices.size == 0:
            candidate_indices = run_indices
            candidate_depths = depths

        min_clearance = max(1.0, float(args.post_mask_grasp_min_clearance_px))
        h, w = component.shape[:2]
        best = None
        for index, depth in zip(candidate_indices.tolist(), candidate_depths.tolist()):
            point = center_xy + float(t_values[index]) * axis_xy
            px = int(np.clip(round(float(point[0])), 0, w - 1))
            py = int(np.clip(round(float(point[1])), 0, h - 1))
            if component[py, px] == 0:
                continue
            clearance = float(distance_map[py, px])
            # Prefer a locally thick cloth patch while retaining the edge-side
            # character. The target-depth penalty prevents drifting to the center.
            clearance_bonus = min(clearance, min_clearance * 3.0)
            target_penalty = 0.12 * abs(float(depth) - preferred)
            inward_penalty = 0.035 * float(depth)
            score = clearance_bonus - target_penalty - inward_penalty
            satisfies = clearance >= min_clearance
            key = (1 if satisfies else 0, score, clearance, -abs(float(depth) - preferred))
            if best is None or key > best[0]:
                best = (key, np.asarray([float(px), float(py)], dtype=np.float64), float(t_values[index]), float(depth), clearance)

        if best is None:
            raise RuntimeError(f"no in-mask minor-axis grasp candidate on side={side_sign:+d}")
        _key, point_xy, point_t, inward_depth, clearance = best
        return point_xy, point_t, inward_depth, clearance

    def _post_mask_analyze_short_axis_grasps(
        mask_u8: np.ndarray,
        H_value: np.ndarray,
    ) -> Dict[str, Any]:
        component = _post_mask_largest_component(mask_u8)
        ys, xs = np.nonzero(component)
        samples_xy = np.column_stack((xs, ys)).astype(np.float64)
        center_xy = np.mean(samples_xy, axis=0)
        centered = samples_xy - center_xy
        covariance = np.cov(centered, rowvar=False)
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        order = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[order]
        eigenvectors = eigenvectors[:, order]
        major_axis = np.asarray(eigenvectors[:, 0], dtype=np.float64)
        minor_axis = np.asarray(eigenvectors[:, 1], dtype=np.float64)
        major_axis /= max(1e-12, float(np.linalg.norm(major_axis)))
        minor_axis /= max(1e-12, float(np.linalg.norm(minor_axis)))

        # Deterministic color assignment: the positive minor-axis direction is
        # the direction with positive X, or positive Y when it is nearly vertical.
        if float(minor_axis[0]) < -1e-9 or (
            abs(float(minor_axis[0])) <= 1e-9 and float(minor_axis[1]) < 0.0
        ):
            minor_axis *= -1.0

        h, w = component.shape[:2]
        diagonal = float(math.hypot(w, h))
        step = float(np.clip(float(args.post_mask_grasp_axis_step_px), 0.25, 2.0))
        t_values = np.arange(-diagonal, diagonal + step, step, dtype=np.float64)

        # A concave garment can place the PCA centroid in a neck/crotch gap. Keep
        # the PCA minor-axis direction, but search a small family of parallel lines
        # along the major axis and choose the most balanced, longest valid crossing.
        major_projection = centered @ major_axis
        major_low = float(np.percentile(major_projection, 8.0))
        major_high = float(np.percentile(major_projection, 92.0))
        offsets = np.unique(np.concatenate((
            np.asarray([0.0], dtype=np.float64),
            np.linspace(major_low, major_high, 41, dtype=np.float64),
        )))
        best_line = None
        major_scale = max(1.0, major_high - major_low)
        for offset in offsets:
            origin = center_xy + float(offset) * major_axis
            line_points = origin[None, :] + t_values[:, None] * minor_axis[None, :]
            xi = np.rint(line_points[:, 0]).astype(np.int32)
            yi = np.rint(line_points[:, 1]).astype(np.int32)
            valid = (xi >= 0) & (xi < w) & (yi >= 0) & (yi < h)
            inside_i = np.zeros_like(valid, dtype=bool)
            inside_i[valid] = component[yi[valid], xi[valid]] > 0
            negative_i = np.flatnonzero((t_values < 0.0) & inside_i)
            positive_i = np.flatnonzero((t_values > 0.0) & inside_i)
            if negative_i.size == 0 or positive_i.size == 0:
                continue
            neg_t = float(np.min(t_values[negative_i]))
            pos_t = float(np.max(t_values[positive_i]))
            span = pos_t - neg_t
            side_balance = min(abs(neg_t), abs(pos_t)) / max(1e-6, max(abs(neg_t), abs(pos_t)))
            occupancy = float(np.count_nonzero(inside_i)) * step
            center_penalty = abs(float(offset)) / major_scale
            score = span + 0.20 * occupancy + 24.0 * side_balance - 8.0 * center_penalty
            candidate = (score, span, side_balance, -abs(float(offset)), origin, inside_i, neg_t, pos_t, float(offset))
            if best_line is None or candidate[:4] > best_line[:4]:
                best_line = candidate

        if best_line is None:
            raise RuntimeError(
                "no PCA-minor-axis parallel crossing intersects garment on both sides"
            )
        _score, axis_length_px, _balance, _offset_rank, axis_origin, inside, negative_boundary_t, positive_boundary_t, axis_offset = best_line
        negative_boundary = axis_origin + negative_boundary_t * minor_axis
        positive_boundary = axis_origin + positive_boundary_t * minor_axis
        if axis_length_px < 18.0:
            raise RuntimeError(f"garment minor axis is too short: {axis_length_px:.1f}px")

        distance_map = cv2.distanceTransform(component, cv2.DIST_L2, 5)
        red_point, red_t, red_inset, red_clearance = _post_mask_pick_axis_side_grasp(
            component, distance_map, axis_origin, minor_axis, t_values, inside, -1
        )
        blue_point, blue_t, blue_inset, blue_clearance = _post_mask_pick_axis_side_grasp(
            component, distance_map, axis_origin, minor_axis, t_values, inside, +1
        )
        separation_px = float(np.linalg.norm(blue_point - red_point))
        if separation_px < 12.0:
            raise RuntimeError(f"computed grasp-point separation is too small: {separation_px:.1f}px")

        angle_deg = float(math.degrees(math.atan2(float(minor_axis[1]), float(minor_axis[0]))))
        return {
            "method": "PCA_MINOR_AXIS_OUTERMOST_RUN_DISTANCE_TRANSFORM_INSET",
            "component_pixel_count": int(cv2.countNonZero(component)),
            "centroid_px": [float(center_xy[0]), float(center_xy[1])],
            "axis_origin_px": [float(axis_origin[0]), float(axis_origin[1])],
            "axis_major_offset_px": float(axis_offset),
            "major_axis_unit_px": [float(major_axis[0]), float(major_axis[1])],
            "minor_axis_unit_px": [float(minor_axis[0]), float(minor_axis[1])],
            "pca_eigenvalues": [float(eigenvalues[0]), float(eigenvalues[1])],
            "minor_axis_angle_deg": angle_deg,
            "minor_axis_boundary_length_px": float(axis_length_px),
            "minor_axis_boundary_red_px": [float(negative_boundary[0]), float(negative_boundary[1])],
            "minor_axis_boundary_blue_px": [float(positive_boundary[0]), float(positive_boundary[1])],
            "red_grasp": {
                "pixel": [float(red_point[0]), float(red_point[1])],
                "board_mm": _post_mask_pixel_to_board(H_value, (float(red_point[0]), float(red_point[1]))),
                "axis_t_px": float(red_t),
                "inset_from_boundary_px": float(red_inset),
                "clearance_radius_px": float(red_clearance),
            },
            "blue_grasp": {
                "pixel": [float(blue_point[0]), float(blue_point[1])],
                "board_mm": _post_mask_pixel_to_board(H_value, (float(blue_point[0]), float(blue_point[1]))),
                "axis_t_px": float(blue_t),
                "inset_from_boundary_px": float(blue_inset),
                "clearance_radius_px": float(blue_clearance),
            },
            "grasp_separation_px": separation_px,
            "grasp_separation_board_mm": float(np.linalg.norm(
                np.asarray(_post_mask_pixel_to_board(H_value, (float(blue_point[0]), float(blue_point[1]))), dtype=np.float64)
                - np.asarray(_post_mask_pixel_to_board(H_value, (float(red_point[0]), float(red_point[1]))), dtype=np.float64)
            )),
        }

    def _post_mask_save_retry_attempt(
        attempt_dir: Path,
        raw_snapshot: np.ndarray,
        raw_roi_u8: np.ndarray,
        corrected: np.ndarray,
        attempt_info: Dict[str, Any],
    ) -> None:
        """Persist a failed attempt without showing a FAIL result window."""
        attempt_dir.mkdir(parents=True, exist_ok=True)
        raw_roi_frame = cv2.bitwise_and(raw_snapshot, raw_snapshot, mask=raw_roi_u8)
        cv2.imwrite(str(attempt_dir / "01_raw_latest.png"), raw_snapshot)
        cv2.imwrite(str(attempt_dir / "02_raw_roi.png"), raw_roi_frame)
        cv2.imwrite(str(attempt_dir / "03_corrected_snapshot.png"), corrected)
        with (attempt_dir / "attempt_result.json").open("w", encoding="utf-8") as fp:
            json.dump(attempt_info, fp, ensure_ascii=False, indent=2)
        print(f"[POST-MASK-RETRY-ARTIFACTS] saved={attempt_dir}")

    def _post_mask_make_single_overlay(
        corrected: np.ndarray,
        final_mask,
        grasp_analysis: Dict[str, Any],
    ) -> np.ndarray:
        """Draw only the garment contour and the two requested X grasp markers."""
        overlay = corrected.copy()
        cv2.drawContours(overlay, [final_mask.contour], -1, (0, 255, 0), 3, cv2.LINE_AA)
        red_point = tuple(np.round(grasp_analysis["red_grasp"]["pixel"]).astype(int))
        blue_point = tuple(np.round(grasp_analysis["blue_grasp"]["pixel"]).astype(int))
        cv2.drawMarker(
            overlay, red_point, (0, 0, 255), cv2.MARKER_TILTED_CROSS, 34, 5, cv2.LINE_AA
        )
        cv2.drawMarker(
            overlay, blue_point, (255, 0, 0), cv2.MARKER_TILTED_CROSS, 34, 5, cv2.LINE_AA
        )
        cv2.putText(
            overlay,
            "RED X",
            (red_point[0] + 10, red_point[1] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            overlay,
            "BLUE X",
            (blue_point[0] + 10, blue_point[1] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (255, 0, 0),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            overlay,
            f"{final_mask.class_name} conf={float(final_mask.confidence):.3f}",
            (20, 38),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.72,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
        return overlay

    def _post_mask_save_and_show(
        run_dir: Path,
        raw_snapshot: np.ndarray,
        raw_roi_u8: np.ndarray,
        corrected: np.ndarray,
        final_mask,
        result_info: Dict[str, Any],
        grasp_analysis: Dict[str, Any],
    ) -> None:
        run_dir.mkdir(parents=True, exist_ok=True)
        raw_roi_frame = cv2.bitwise_and(raw_snapshot, raw_snapshot, mask=raw_roi_u8)
        mask_u8 = np.asarray(final_mask.mask_u8, dtype=np.uint8)
        overlay = _post_mask_make_single_overlay(corrected, final_mask, grasp_analysis)

        cv2.imwrite(str(run_dir / "01_raw_latest.png"), raw_snapshot)
        cv2.imwrite(str(run_dir / "02_raw_roi.png"), raw_roi_frame)
        cv2.imwrite(str(run_dir / "03_corrected_snapshot.png"), corrected)
        cv2.imwrite(str(run_dir / "04_final_mask.png"), mask_u8)
        cv2.imwrite(str(run_dir / "05_single_contour_grasp_overlay.png"), overlay)
        with (run_dir / "result.json").open("w", encoding="utf-8") as fp:
            json.dump(result_info, fp, ensure_ascii=False, indent=2)
        print(f"[POST-MASK-ARTIFACTS] saved={run_dir}")

        if args.no_window:
            return
        if preview is not None:
            preview.show_override(
                overlay,
                "MASK SUCCESS | green contour + red/blue X grasp points | Q/ESC/Enter close",
            )
            if bool(args.post_mask_hold_window):
                print(
                    "[POST-MASK-WINDOW] single corrected-image overlay is shown; "
                    "Q/ESC/Enter returns to RAW preview"
                )
                while True:
                    key = read_operator_key()
                    if key in (ord("q"), ord("Q"), 27, 13, 10):
                        break
                    time.sleep(0.02)
            preview.clear_override()
            preview.set_status("RAW PREVIEW RESUMED | post-mask success | no robot command")
            return

        window_name = "FIX11 Post-Laydown Garment Contour + Grasp X"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.imshow(window_name, overlay)
        cv2.waitKey(1)
        if bool(args.post_mask_hold_window):
            print("[POST-MASK-WINDOW] Q/ESC/Enter closes the single result window")
            while True:
                tkey = terminal.read_key()
                wkey = cv2.waitKey(20) & 0xFF
                key = tkey if tkey != 255 else wkey
                if key in (ord("q"), ord("Q"), 27, 13, 10):
                    break
                if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                    break
        cv2.destroyWindow(window_name)

    def _post_mask_retry_pause(attempt_index: int) -> None:
        delay = max(0.0, float(args.post_mask_retry_delay_sec))
        if preview is not None:
            preview.set_status(
                f"NO MASK ON ATTEMPT {attempt_index} | acquiring a new stable frame | Q/ESC abort"
            )
        deadline = time.monotonic() + delay
        while time.monotonic() < deadline:
            key = read_operator_key()
            if key in (ord("q"), ord("Q"), 27):
                raise KeyboardInterrupt("operator stopped post-mask retry loop")
            time.sleep(0.02)

    def _run_post_laydown_mask_only() -> bool:
        nonlocal post_mask_has_run
        if not bool(args.post_laydown_mask_only):
            print("[POST-MASK] disabled; pure FIX11 completion retained")
            return True
        if post_mask_has_run:
            print("[POST-MASK] stage already finished; duplicate request ignored")
            return True

        post_cap = None
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        debug_root = Path(args.post_mask_debug_dir)
        if not debug_root.is_absolute():
            debug_root = SCRIPT_DIR / debug_root
        run_dir = debug_root / timestamp
        run_dir.mkdir(parents=True, exist_ok=True)
        result_info: Dict[str, Any] = {
            "build_id": BUILD_ID,
            "fix11_complete_at": datetime.now().isoformat(timespec="milliseconds"),
            "robot_commands_after_fix11_complete": 0,
            "success": False,
            "attempts": [],
            "retry_policy": {
                "fresh_stable_snapshot_each_attempt": True,
                "max_attempts": int(args.post_mask_max_attempts),
                "zero_means_unlimited": True,
                "retry_delay_sec": float(args.post_mask_retry_delay_sec),
                "failure_window_shown": False,
            },
        }

        try:
            print("\n========== POST-LAYDOWN MASK RETRY STAGE ==========")
            print("[POST-MASK-SAFETY] FIX11 is complete; no T:104/T:1041/T:106 command is permitted")
            print("[POST-MASK-RETRY] every failed confidence ladder acquires a NEW stable RAW snapshot")
            h_path, H_value, raw_H_value, h_metadata = _post_mask_load_homography()
            result_info["homography_path"] = h_path
            result_info["homography_camera_geometry"] = h_metadata

            if preview is not None:
                camera_info_runtime = preview.camera_info()
                actual_w, actual_h = [
                    int(v) for v in camera_info_runtime.get("actual_size", [args.width, args.height])
                ]
                print(
                    f"[POST-MASK-CAMERA] reusing continuously consumed startup RAW camera "
                    f"actual={actual_w}x{actual_h} fourcc={camera_info_runtime.get('fourcc')} "
                    f"fps={float(camera_info_runtime.get('fps', 0.0)):.2f}"
                )
                result_info["camera"] = camera_info_runtime
            else:
                post_cap = _post_mask_open_camera(args)
                control_report = _post_mask_configure_uvc(args, post_cap)
                _probe_frame, fallback_brightness = _post_mask_collect_brightness_probe(
                    post_cap, int(args.post_mask_camera_brightness_warmup_frames)
                )
                actual_w = int(post_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                actual_h = int(post_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                actual_fps = float(post_cap.get(cv2.CAP_PROP_FPS))
                fourcc_value = int(post_cap.get(cv2.CAP_PROP_FOURCC))
                fourcc_text = "".join(chr((fourcc_value >> (8 * i)) & 0xFF) for i in range(4))
                fourcc_text = fourcc_text.replace("\x00", "").strip() or "UNKNOWN"
                print(
                    f"[POST-MASK-CAMERA] fallback camera remains open for all retries "
                    f"actual={actual_w}x{actual_h} fourcc={fourcc_text} fps={actual_fps:.2f}"
                )
                result_info["camera"] = {
                    "index": int(args.camera),
                    "actual_size": [actual_w, actual_h],
                    "fourcc": fourcc_text,
                    "fps": actual_fps,
                    "camera_control_mode": control_report.get("mode"),
                    "camera_control_report": control_report,
                    "startup_brightness": fallback_brightness,
                }

            if CameraUndistorter is None:
                raise RuntimeError(
                    f"camera_undistort.py import failed: {_CAMERA_UNDISTORT_IMPORT_ERROR!r}"
                )
            calibration_path = _resolve_runtime_file(args.camera_calibration)
            undistorter = CameraUndistorter(
                calibration_path,
                alpha=float(args.camera_undistort_alpha),
                strict_size=bool(args.camera_undistort_strict_size),
            )
            camera_prepared = undistorter.prepare((actual_w, actual_h))
            runtime_metadata = camera_prepared.to_metadata()

            # Enforce every field that the Homography cache actually contains.
            # For the metadata-less V5 cache, fill the descriptive fields from
            # the runtime CameraUndistorter instead of terminating before the
            # first segmentation attempt.
            effective_h_metadata = dict(h_metadata)
            if "undistort_enabled" in effective_h_metadata:
                if not bool(effective_h_metadata.get("undistort_enabled")):
                    raise RuntimeError(
                        "camera/Homography geometry mismatch: undistort_enabled=false"
                    )
            else:
                effective_h_metadata["undistort_enabled"] = True

            for key in ("calibration_id", "output_size", "alpha"):
                actual = runtime_metadata.get(key)
                if key not in effective_h_metadata:
                    effective_h_metadata[key] = actual
                    print(
                        f"[POST-MASK-H-METADATA-FILL] {key}={actual!r} "
                        "source=runtime CameraUndistorter"
                    )
                    continue
                expected = effective_h_metadata.get(key)
                matched = (
                    abs(float(expected) - float(actual)) <= 1e-6
                    if key == "alpha" else expected == actual
                )
                if not matched:
                    raise RuntimeError(
                        f"camera/Homography geometry mismatch {key}: "
                        f"H={expected!r} runtime={actual!r}"
                    )

            runtime_output_size = runtime_metadata.get("output_size")
            if runtime_output_size != [int(actual_w), int(actual_h)]:
                raise RuntimeError(
                    f"runtime undistort output_size={runtime_output_size!r} does not "
                    f"match camera frame {[int(actual_w), int(actual_h)]!r}"
                )
            h_metadata = effective_h_metadata
            result_info["homography_camera_geometry_effective"] = effective_h_metadata
            result_info["camera_calibration"] = calibration_path
            result_info["runtime_camera_geometry"] = runtime_metadata
            print(
                "[POST-MASK-H-RUNTIME-COMPAT-OK] numerical H/raw_H valid; "
                f"output_size={runtime_output_size} alpha={runtime_metadata.get('alpha')} "
                f"legacyCompat={bool(effective_h_metadata.get('_legacy_compat'))}"
            )

            seg_model = None
            e49_bottom = None
            target_names: List[str] = []
            model_names: Dict[str, str] = {}
            undistort_calls = 0
            attempt_index = 0
            max_attempts = max(0, int(args.post_mask_max_attempts))

            while True:
                if max_attempts > 0 and attempt_index >= max_attempts:
                    result_info["stop_reason"] = "configured maximum attempts reached"
                    result_info["undistort_calls"] = int(undistort_calls)
                    with (run_dir / "result.json").open("w", encoding="utf-8") as fp:
                        json.dump(result_info, fp, ensure_ascii=False, indent=2)
                    post_mask_has_run = True
                    print(
                        f"[POST-MASK-STOPPED] max attempts={max_attempts} reached; "
                        "robot remains at standby"
                    )
                    return False

                key = read_operator_key()
                if key in (ord("q"), ord("Q"), 27):
                    raise KeyboardInterrupt("operator stopped post-mask retry loop")

                attempt_index += 1
                attempt_dir = run_dir / f"attempt_{attempt_index:03d}"
                attempt_info: Dict[str, Any] = {
                    "attempt": int(attempt_index),
                    "started_at": datetime.now().isoformat(timespec="milliseconds"),
                    "success": False,
                }
                print(f"\n[POST-MASK-ATTEMPT] {attempt_index} fresh stable-frame judgment starts")
                if preview is not None:
                    preview.set_status(
                        f"POST-MASK ATTEMPT {attempt_index} | waiting for a new stable board frame"
                    )

                if preview is not None:
                    raw_snapshot, raw_roi_u8, marker_pixels, stability = (
                        _post_mask_capture_stable_raw_from_preview(
                            preview, raw_H_value, attempt_dir
                        )
                    )
                else:
                    raw_snapshot, raw_roi_u8, marker_pixels, stability = (
                        _post_mask_capture_stable_raw(post_cap, raw_H_value, attempt_dir)
                    )
                attempt_info["stability"] = stability
                attempt_info["raw_marker_pixels"] = marker_pixels
                print(
                    f"[POST-MASK-SNAPSHOT] attempt={attempt_index} fresh RAW sequence="
                    f"{stability.get('accepted_sequence')} source={stability.get('source')}"
                )

                undistort_calls += 1
                print(
                    f"[POST-MASK-UNDISTORT] attempt={attempt_index} call_index={undistort_calls}; "
                    "remapping this accepted RAW snapshot exactly once"
                )
                corrected = undistorter.correct(raw_snapshot)
                corrected_brightness = _post_mask_global_brightness_stats(corrected)
                attempt_info["corrected_brightness"] = corrected_brightness
                attempt_info["undistort_call_index"] = int(undistort_calls)
                print(
                    f"[POST-MASK-CORRECTED-BRIGHTNESS] attempt={attempt_index} "
                    f"mean={corrected_brightness['mean']:.1f} "
                    f"median={corrected_brightness['median']:.1f} "
                    f"p95={corrected_brightness['p95']:.1f} "
                    f"dark<=20={corrected_brightness['dark_ratio_le20']:.3f}"
                )

                # Preserve the V5 contract: the TensorRT model is loaded only after
                # the first stable RAW snapshot has been accepted and corrected.
                if seg_model is None:
                    seg_path = _resolve_runtime_file(
                        args.post_mask_seg_model, include_models=True
                    )
                    if not Path(seg_path).is_file():
                        raise RuntimeError(f"segmentation model not found: {seg_path}")
                    print(f"[POST-MASK-MODEL] one-time delayed load starts: {seg_path}")
                    try:
                        from ultralytics import YOLO
                    except Exception as exc:
                        raise RuntimeError(
                            f"ultralytics import failed after FIX11 completion: {exc!r}"
                        ) from exc
                    seg_model = YOLO(seg_path, task="segment")
                    model_names_raw = getattr(seg_model, "names", {}) or {}
                    if isinstance(model_names_raw, dict):
                        model_names = {
                            str(int(k)) if isinstance(k, (int, np.integer)) else str(k): str(v)
                            for k, v in model_names_raw.items()
                        }
                    elif isinstance(model_names_raw, (list, tuple)):
                        model_names = {str(i): str(v) for i, v in enumerate(model_names_raw)}
                    else:
                        model_names = {}
                    print(f"[POST-MASK-MODEL-LOADED] one-time names={model_names}")
                    result_info["segmentation_model"] = seg_path
                    result_info["model_names"] = model_names

                    try:
                        import step_e49_bottom_perception as e49_bottom_module
                    except Exception as exc:
                        raise RuntimeError(
                            f"step_e49_bottom_perception import failed: {exc!r}"
                        ) from exc
                    e49_bottom = _post_mask_prepare_e49_runtime(
                        e49_bottom_module, H_value
                    )
                    available_names = list(model_names.values())
                    requested = e49_bottom.parse_class_names(args.post_mask_classes)
                    target_names = [name for name in requested if name in available_names]
                    if not target_names:
                        target_names = available_names
                        print(
                            f"[POST-MASK-CLASS-WARN] requested={requested} absent; "
                            f"using all model classes={target_names}"
                        )
                    result_info["target_classes"] = target_names

                final_mask = None
                ladder_log: List[Dict[str, Any]] = []
                for conf in _post_mask_parse_conf_ladder():
                    print(
                        f"[POST-MASK-E49-CALL] attempt={attempt_index} "
                        f"infer_bottoms_mask conf={conf:.3f}"
                    )
                    mask_i, status_i = e49_bottom.infer_bottoms_mask(
                        seg_model,
                        corrected,
                        H_value,
                        int(args.post_mask_seg_imgsz),
                        float(conf),
                        target_class_names=target_names,
                        args=args,
                    )
                    entry = {
                        "confidence": float(conf),
                        "mask": mask_i is not None,
                        "status": str(status_i),
                    }
                    ladder_log.append(entry)
                    print(
                        f"[POST-MASK-E49-RETURN] attempt={attempt_index} "
                        f"conf={conf:.3f} mask={mask_i is not None} status={status_i}"
                    )
                    if mask_i is not None:
                        final_mask = mask_i
                        break
                attempt_info["confidence_ladder"] = ladder_log

                retry_reason = None
                roi_info = None
                area_ratio = None
                mask_area = 0
                roi_area = 0
                grasp_analysis = None

                if final_mask is None:
                    retry_reason = "E49 returned no garment mask on this fresh snapshot"
                else:
                    valid_roi, roi_info = e49_bottom.build_board_valid_mask(
                        corrected.shape, H_value, args
                    )
                    roi_area = int(cv2.countNonZero(valid_roi))
                    mask_area = int(
                        cv2.countNonZero(
                            np.asarray(final_mask.mask_u8, dtype=np.uint8)
                        )
                    )
                    area_ratio = float(mask_area) / float(max(1, roi_area))
                    x, y, bw, bh = cv2.boundingRect(final_mask.contour)
                    bbox_ratio_w = float(bw) / float(max(1, corrected.shape[1]))
                    bbox_ratio_h = float(bh) / float(max(1, corrected.shape[0]))
                    if area_ratio > 0.72 or (
                        bbox_ratio_w >= 0.93 and bbox_ratio_h >= 0.93
                    ):
                        retry_reason = (
                            f"gross board-leak mask rejected areaRatio={area_ratio:.3f} "
                            f"bboxRatio=({bbox_ratio_w:.3f},{bbox_ratio_h:.3f})"
                        )
                    else:
                        try:
                            grasp_analysis = _post_mask_analyze_short_axis_grasps(
                                np.asarray(final_mask.mask_u8, dtype=np.uint8), H_value
                            )
                        except Exception as exc:
                            retry_reason = f"short-axis grasp analysis rejected: {exc}"

                if retry_reason is not None:
                    attempt_info["retry_reason"] = str(retry_reason)
                    attempt_info["completed_at"] = datetime.now().isoformat(
                        timespec="milliseconds"
                    )
                    if final_mask is not None:
                        attempt_info["candidate_mask"] = {
                            "class_name": str(final_mask.class_name),
                            "confidence": float(final_mask.confidence),
                            "pixel_count": int(mask_area),
                            "board_roi_pixel_count": int(roi_area),
                            "board_roi_area_ratio": area_ratio,
                        }
                    _post_mask_save_retry_attempt(
                        attempt_dir,
                        raw_snapshot,
                        raw_roi_u8,
                        corrected,
                        attempt_info,
                    )
                    result_info["attempts"].append(attempt_info)
                    result_info["undistort_calls"] = int(undistort_calls)
                    with (run_dir / "result.json").open("w", encoding="utf-8") as fp:
                        json.dump(result_info, fp, ensure_ascii=False, indent=2)
                    print(
                        f"[POST-MASK-RETRY] attempt={attempt_index} reason={retry_reason}; "
                        "no FAIL window, no robot command, acquiring a new frame"
                    )
                    _post_mask_retry_pause(attempt_index)
                    continue

                print(
                    f"[POST-MASK-SHORT-AXIS] method={grasp_analysis['method']} "
                    f"length={grasp_analysis['minor_axis_boundary_length_px']:.1f}px "
                    f"angle={grasp_analysis['minor_axis_angle_deg']:.1f}deg"
                )
                print(
                    f"[POST-MASK-GRASP-RED] px={grasp_analysis['red_grasp']['pixel']} "
                    f"board={grasp_analysis['red_grasp']['board_mm']}"
                )
                print(
                    f"[POST-MASK-GRASP-BLUE] px={grasp_analysis['blue_grasp']['pixel']} "
                    f"board={grasp_analysis['blue_grasp']['board_mm']}"
                )

                x, y, bw, bh = cv2.boundingRect(final_mask.contour)
                attempt_info.update({
                    "success": True,
                    "completed_at": datetime.now().isoformat(timespec="milliseconds"),
                    "mask": {
                        "class_name": str(final_mask.class_name),
                        "confidence": float(final_mask.confidence),
                        "area_px": float(final_mask.area_px),
                        "pixel_count": int(mask_area),
                        "board_roi_pixel_count": int(roi_area),
                        "board_roi_area_ratio": float(area_ratio),
                        "bbox_xywh": [int(x), int(y), int(bw), int(bh)],
                        "solidity": float(final_mask.solidity),
                        "center_px": [float(v) for v in final_mask.center_px],
                        "center_board": [float(v) for v in final_mask.center_board],
                    },
                    "e49_board_roi": roi_info,
                    "short_axis_grasp_analysis": grasp_analysis,
                })
                result_info["attempts"].append(attempt_info)
                result_info.update({
                    "success": True,
                    "successful_attempt": int(attempt_index),
                    "undistort_calls": int(undistort_calls),
                    "mask": attempt_info["mask"],
                    "e49_board_roi": roi_info,
                    "short_axis_grasp_analysis": grasp_analysis,
                    "completed_at": datetime.now().isoformat(timespec="milliseconds"),
                })
                _post_mask_save_and_show(
                    run_dir,
                    raw_snapshot,
                    raw_roi_u8,
                    corrected,
                    final_mask,
                    result_info,
                    grasp_analysis,
                )
                post_mask_has_run = True
                print(
                    f"[POST-MASK-SUCCESS] attempt={attempt_index} "
                    f"class={final_mask.class_name} conf={float(final_mask.confidence):.3f} "
                    f"area={float(final_mask.area_px):.0f}px ROIratio={float(area_ratio):.3f}"
                )
                print(
                    "[POST-MASK-SAFETY] success; robot command count after FIX11-COMPLETE remains 0"
                )
                print("===================================================\n")
                return True

        except KeyboardInterrupt:
            post_mask_has_run = True
            result_info["stop_reason"] = "operator abort"
            result_info["stopped_at"] = datetime.now().isoformat(timespec="milliseconds")
            try:
                with (run_dir / "result.json").open("w", encoding="utf-8") as fp:
                    json.dump(result_info, fp, ensure_ascii=False, indent=2)
            except Exception:
                pass
            print(
                "[POST-MASK-STOPPED] retry stage closed by operator; "
                "robot remains at FIX11 standby"
            )
            if preview is not None:
                preview.clear_override()
                preview.set_status("POST-MASK STOPPED | RAW preview live | robot at standby")
            return False
        except Exception as exc:
            post_mask_has_run = True
            result_info["fatal_error"] = repr(exc)
            result_info["stopped_at"] = datetime.now().isoformat(timespec="milliseconds")
            try:
                with (run_dir / "result.json").open("w", encoding="utf-8") as fp:
                    json.dump(result_info, fp, ensure_ascii=False, indent=2)
            except Exception:
                pass
            print(f"[POST-MASK-FATAL] {exc}")
            print(
                "[POST-MASK-SAFETY] fatal perception setup error; no robot command was sent"
            )
            if preview is not None:
                preview.clear_override()
                preview.set_status("POST-MASK STOPPED | RAW preview live | robot at standby")
            return False
        finally:
            if post_cap is not None:
                post_cap.release()

    def _execute_dual_arc_laydown(spread_state):
        """FIX11 post-spread pair recenter and synchronized curved laydown with safe actual-pose reanchoring."""
        robot1 = _lazy_open_arm1_for_handoff()
        if not bool(args.dual_arc_laydown):
            print("[FIX11-LAYDOWN] disabled; retaining FIX8 final hold")
            return True

        # Re-read both real poses. Their actual local XY values are inverted back
        # to board coordinates before the pair midpoint is recentered.
        fb2 = arm.feedback_retry(args.feedback_timeout, attempts=4, retry_delay=0.20)
        fb1 = robot1.feedback_retry(args.feedback_timeout, attempts=4, retry_delay=0.20)
        if fb2 is None or not all(k in fb2 for k in ("x", "y", "z")):
            raise RuntimeError("ARM2 feedback unavailable before FIX11 laydown")
        if fb1 is None or not all(k in fb1 for k in ("x", "y", "z")):
            raise RuntimeError("ARM1 feedback unavailable before FIX11 laydown")

        actual_pose = {
            "arm2": np.asarray([float(fb2["x"]), float(fb2["y"]), float(fb2["z"])], dtype=np.float64),
            "arm1": np.asarray([float(fb1["x"]), float(fb1["y"]), float(fb1["z"])], dtype=np.float64),
        }
        # T:104/T:1041 ``t`` is the Cartesian tool/wrist orientation, not the
        # T:106 gripper close angle. Preserve each arm's measured orientation
        # throughout recenter, curved streaming, and final support lock.
        tool_t_by_arm = {
            "arm2": float(fb2.get("t", args.tool_angle_fallback)),
            "arm1": float(fb1.get("t", args.tool_angle_fallback)),
        }
        for arm_key in ("arm2", "arm1"):
            if not np.isfinite(tool_t_by_arm[arm_key]):
                tool_t_by_arm[arm_key] = float(args.tool_angle_fallback)
        print(
            f"[FIX11-TOOL-T] preserve ARM2={tool_t_by_arm['arm2']:.6f}rad "
            f"ARM1={tool_t_by_arm['arm1']:.6f}rad; grippers remain independently CLOSED "
            f"by T:106 at {float(args.grasp_close_angle):.3f}rad"
        )
        actual_board = {
            arm_key: _arm_xy_to_board(arm_key, pose[0], pose[1])
            for arm_key, pose in actual_pose.items()
        }
        center_board = np.asarray(board_target["board_center_xy"], dtype=np.float64)
        pair_mid = 0.5 * (actual_board["arm2"] + actual_board["arm1"])
        recenter_delta = np.asarray([float(center_board[0] - pair_mid[0]), 0.0], dtype=np.float64)
        start_board = {
            arm_key: actual_board[arm_key] + recenter_delta
            for arm_key in ("arm2", "arm1")
        }
        start_separation = float(np.linalg.norm(start_board["arm1"] - start_board["arm2"]))

        marker = board_target.get("marker_board_mm", {})
        m0 = np.asarray(marker.get("0"), dtype=np.float64)
        m1 = np.asarray(marker.get("1"), dtype=np.float64)
        m2 = np.asarray(marker.get("2"), dtype=np.float64)
        m3 = np.asarray(marker.get("3"), dtype=np.float64)
        if not all(p.shape == (2,) and np.all(np.isfinite(p)) for p in (m0, m1, m2, m3)):
            raise RuntimeError("invalid marker geometry for FIX11 laydown direction")
        final_direction = 0.5 * (m0 + m2) - 0.5 * (m1 + m3)
        norm = float(np.linalg.norm(final_direction))
        if norm <= 1e-6:
            raise RuntimeError("ID1-ID3 to ID0-ID2 direction is degenerate")
        final_direction /= norm
        back_direction = -final_direction

        amplitude = max(20.0, float(args.laydown_swing_amplitude_mm))
        requested_rise = max(0.0, float(args.laydown_curve_rise_mm))
        z_margin = max(0.0, float(args.laydown_z_limit_margin_mm))
        common_available_rise = min(
            float(args.z_max) - z_margin - float(actual_pose[k][2])
            for k in ("arm2", "arm1")
        )
        curve_rise = min(requested_rise, common_available_rise)
        if curve_rise < float(args.laydown_min_curve_rise_mm) - 1e-6:
            raise RuntimeError(
                f"FIX11 common curve-rise room too small: {curve_rise:.1f}mm "
                f"< {float(args.laydown_min_curve_rise_mm):.1f}mm"
            )

        back_board = {
            k: start_board[k] + back_direction * amplitude for k in ("arm2", "arm1")
        }
        final_board = {
            k: start_board[k] + final_direction * amplitude for k in ("arm2", "arm1")
        }
        final_z = {
            "arm2": _surface_z("arm2", final_board["arm2"][0], final_board["arm2"][1])
                    + float(args.laydown_arm2_final_clearance_mm),
            "arm1": _surface_z("arm1", final_board["arm1"][0], final_board["arm1"][1])
                    + float(args.laydown_arm1_final_clearance_mm),
        }
        peak_z = {
            k: float(actual_pose[k][2]) + curve_rise for k in ("arm2", "arm1")
        }
        for k in ("arm2", "arm1"):
            _require_in_range(f"{k} FIX11 final_z", final_z[k], args.z_min, args.z_max)
            _require_in_range(f"{k} FIX11 peak_z", peak_z[k], args.z_min, args.z_max)
            if final_z[k] >= peak_z[k] - 10.0:
                raise RuntimeError(
                    f"{k} final support Z {final_z[k]:.1f} is not sufficiently below peak {peak_z[k]:.1f}"
                )

        hz = float(np.clip(float(args.laydown_stream_hz), 10.0, 30.0))
        back_duration = max(0.40, float(args.laydown_backswing_duration_sec))
        forward_base_duration = max(0.70, float(args.laydown_forward_duration_sec))
        back_steps = max(8, int(math.ceil(back_duration * hz)))
        forward_steps = max(14, int(math.ceil(forward_base_duration * hz)))
        decay = max(0.10, float(args.laydown_decay))
        vertical_gamma = max(1.0, float(args.laydown_vertical_gamma))
        slow_ratio = float(np.clip(float(args.laydown_final_slow_zone_ratio), 0.0, 0.80))
        slow_scale = float(np.clip(float(args.laydown_final_slow_speed_scale), 0.10, 1.0))
        slow_start_t = 1.0 - slow_ratio
        duration_scale = slow_start_t + slow_ratio / max(1e-6, slow_scale)
        forward_duration = float(forward_base_duration * duration_scale)

        def _exp_remaining(progress: float) -> float:
            p = float(np.clip(float(progress), 0.0, 1.0))
            end = math.exp(-decay)
            return float(np.clip(
                (math.exp(-decay * p) - end) / max(1e-9, 1.0 - end),
                0.0, 1.0,
            ))

        def _local_target(arm_key: str, board_xy, z: float):
            if arm_key == "arm1":
                x, y = _board_to_arm1_xy(float(board_xy[0]), float(board_xy[1]))
            else:
                x, y = _board_to_arm2_xy(float(board_xy[0]), float(board_xy[1]))
            return np.asarray([x, y, float(z)], dtype=np.float64)

        recenter_targets = {
            k: _local_target(k, start_board[k], float(actual_pose[k][2]))
            for k in ("arm2", "arm1")
        }
        backswing = []
        ascent_values = []
        for index in range(1, back_steps + 1):
            t = float(index) / float(back_steps)
            eased = t * t * (3.0 - 2.0 * t)
            theta = 0.5 * math.pi * eased
            horizontal_ratio = 1.0 - math.cos(theta)
            rise_ratio = math.sin(theta)
            per_arm = {}
            for k in ("arm2", "arm1"):
                board_at = start_board[k] + back_direction * amplitude * horizontal_ratio
                z_at = float(actual_pose[k][2]) + curve_rise * rise_ratio
                per_arm[k] = {
                    "board": np.asarray(board_at, dtype=np.float64),
                    "local": _local_target(k, board_at, z_at),
                }
            backswing.append(per_arm)
            ascent_values.append(rise_ratio)

        forward = []
        remain_values = []
        forward_time_fractions = []
        for index in range(1, forward_steps + 1):
            t = float(index) / float(forward_steps)
            smooth = t * t * (3.0 - 2.0 * t)
            vertical_s = float(smooth ** vertical_gamma)
            remain = _exp_remaining(vertical_s)
            if slow_ratio <= 1e-9 or t <= slow_start_t:
                raw_time = t
            else:
                raw_time = slow_start_t + (t - slow_start_t) / slow_scale
            forward_time_fractions.append(float(raw_time / duration_scale))
            per_arm = {}
            for k in ("arm2", "arm1"):
                board_at = back_board[k] + final_direction * (2.0 * amplitude * smooth)
                z_at = final_z[k] + (peak_z[k] - final_z[k]) * remain
                per_arm[k] = {
                    "board": np.asarray(board_at, dtype=np.float64),
                    "local": _local_target(k, board_at, z_at),
                }
            forward.append(per_arm)
            remain_values.append(remain)

        bx0, bx1, by0, by1 = [float(v) for v in board_target["board_bounds"]]
        margin = max(0.0, float(args.laydown_board_margin_mm))
        radius_max = max(50.0, float(args.laydown_roarm_radius_max_mm))
        axis_max = max(50.0, float(args.laydown_local_axis_max_mm))
        step_max = max(2.0, float(args.laydown_stream_max_step_mm))

        def _check_board(label: str, board_xy):
            x, y = float(board_xy[0]), float(board_xy[1])
            if not (bx0 + margin <= x <= bx1 - margin and by0 + margin <= y <= by1 - margin):
                raise RuntimeError(
                    f"{label} board point ({x:.1f},{y:.1f}) outside margin {margin:.1f}mm"
                )

        def _check_local(label: str, local):
            x, y, z = [float(v) for v in local]
            if not all(np.isfinite([x, y, z])):
                raise RuntimeError(f"{label} nonfinite local target")
            if max(abs(x), abs(y)) > axis_max + 1e-6:
                raise RuntimeError(f"{label} local axis exceeds {axis_max:.1f}mm: ({x:.1f},{y:.1f})")
            radius = math.hypot(x, y)
            if radius > radius_max + 1e-6:
                raise RuntimeError(f"{label} local XY radius {radius:.1f}>{radius_max:.1f}mm")
            _require_in_range(f"{label} z", z, args.z_min, args.z_max)

        for k in ("arm2", "arm1"):
            _check_board(f"{k} recenter", start_board[k])
            _check_board(f"{k} back", back_board[k])
            _check_board(f"{k} final", final_board[k])
            _check_local(f"{k} recenter", recenter_targets[k])
        for phase_name, sequence in (("BACK", backswing), ("FORWARD", forward)):
            previous = {
                k: recenter_targets[k].copy() if phase_name == "BACK"
                else backswing[-1][k]["local"].copy()
                for k in ("arm2", "arm1")
            }
            for index, per_arm in enumerate(sequence, 1):
                sep = float(np.linalg.norm(per_arm["arm1"]["board"] - per_arm["arm2"]["board"]))
                if abs(sep - start_separation) > 1e-3:
                    raise RuntimeError(f"{phase_name} pair separation changed at {index}: {sep:.3f}mm")
                for k in ("arm2", "arm1"):
                    _check_board(f"{phase_name}-{index}-{k}", per_arm[k]["board"])
                    _check_local(f"{phase_name}-{index}-{k}", per_arm[k]["local"])
                    step = float(np.linalg.norm(per_arm[k]["local"] - previous[k]))
                    if step > step_max + 1e-6:
                        raise RuntimeError(
                            f"{phase_name}-{index}-{k} stream step {step:.1f}>{step_max:.1f}mm"
                        )
                    previous[k] = per_arm[k]["local"].copy()

        print("\n========== FIX11 ROUGH PREFLIGHT BEFORE RECENTER ==========")
        print(
            f"actual board ARM2=({actual_board['arm2'][0]:.1f},{actual_board['arm2'][1]:.1f}) "
            f"ARM1=({actual_board['arm1'][0]:.1f},{actual_board['arm1'][1]:.1f})"
        )
        print(
            f"pair midpoint X={pair_mid[0]:.1f} -> board center X={center_board[0]:.1f}; "
            f"shared recenter dX={recenter_delta[0]:+.1f}mm; separation preserved={start_separation:.1f}mm"
        )
        print(
            f"A={amplitude:.1f}mm | back={amplitude:.1f}mm | forward={2.0*amplitude:.1f}mm | "
            f"curveRise={curve_rise:.1f}mm | stream={hz:.1f}Hz"
        )
        print(
            f"final support: ARM2 surface+{float(args.laydown_arm2_final_clearance_mm):.1f}="
            f"{final_z['arm2']:.1f}mm; ARM1 surface+{float(args.laydown_arm1_final_clearance_mm):.1f}="
            f"{final_z['arm1']:.1f}mm"
        )
        print(
            f"forward final slow zone={slow_ratio*100.0:.0f}% at {slow_scale:.2f}x; "
            f"forward duration={forward_base_duration:.2f}->{forward_duration:.2f}s; "
            f"all {back_steps + forward_steps} nominal pair targets passed rough preflight"
        )
        print("=====================================================\n")

        def _pair_t104(label: str, targets, speed: float):
            barrier = threading.Barrier(3)
            errors = []
            lock = threading.Lock()
            def one(arm_key: str, robot_obj):
                try:
                    barrier.wait(timeout=2.0)
                    target = targets[arm_key]
                    robot_obj.move_goal(
                        args.move_command,
                        float(target[0]), float(target[1]), float(target[2]),
                        float(tool_t_by_arm[arm_key]), float(speed),
                    )
                except Exception as exc:
                    with lock:
                        errors.append((arm_key, exc))
            threads = [
                threading.Thread(target=one, args=("arm2", arm), daemon=True),
                threading.Thread(target=one, args=("arm1", robot1), daemon=True),
            ]
            for thread in threads:
                thread.start()
            try:
                barrier.wait(timeout=2.0)
            except KeyboardInterrupt:
                for thread in threads:
                    thread.join(timeout=1.0)
                raise
            except Exception as exc:
                errors.append(("barrier", exc))
            for thread in threads:
                thread.join(timeout=4.0)
            if any(thread.is_alive() for thread in threads):
                raise RuntimeError(f"{label} paired T:104 thread timeout")
            if errors:
                raise RuntimeError(f"{label} paired T:104 failed: {errors[0]}")

        stream_abort = threading.Event()

        def _stream_pair(label: str, sequence, duration: float, profiles, time_fractions=None):
            if not sequence:
                raise RuntimeError(f"{label} has no waypoints")
            point_count = len(sequence)
            duration = max(0.20, float(duration))
            period = duration / float(point_count)
            if time_fractions is not None:
                if len(time_fractions) != point_count:
                    raise RuntimeError(f"{label} time-fraction count mismatch")
                previous_fraction = 0.0
                for value in time_fractions:
                    if not np.isfinite(value) or value <= previous_fraction:
                        raise RuntimeError(f"{label} time fractions are not strictly increasing")
                    previous_fraction = float(value)
                if abs(float(time_fractions[-1]) - 1.0) > 1e-5:
                    raise RuntimeError(f"{label} final time fraction is not 1")
            max_lag = max(0.01, float(args.laydown_stream_max_lag_sec))
            max_write = max(0.005, float(args.laydown_stream_max_write_ms) / 1000.0)
            log_stride = max(1, int(args.laydown_waypoint_log_stride))
            start = time.monotonic()
            max_lag_seen = max_write_seen = max_skew_seen = 0.0
            print(
                f"[{label}-START] T1041 points={point_count} duration={duration:.2f}s "
                f"nominalRate={1.0/period:.1f}Hz"
            )
            try:
                for index, per_arm in enumerate(sequence, 1):
                    if stream_abort.is_set() or _probe_abort_requested():
                        stream_abort.set()
                        raise KeyboardInterrupt(f"operator abort during {label}")
                    fraction = float(index) / float(point_count) if time_fractions is None else float(time_fractions[index - 1])
                    deadline = start + fraction * duration
                    sleep_s = deadline - time.monotonic()
                    if sleep_s > 0.0:
                        time.sleep(sleep_s)
                    send_start = time.monotonic()
                    lag = max(0.0, send_start - deadline)
                    max_lag_seen = max(max_lag_seen, lag)
                    if lag > max_lag:
                        raise RuntimeError(
                            f"{label} schedule lag {lag*1000.0:.1f}>{max_lag*1000.0:.1f}ms at {index}/{point_count}"
                        )
                    barrier = threading.Barrier(3)
                    timing = {}
                    errors = []
                    lock = threading.Lock()
                    def one(arm_key: str, robot_obj):
                        try:
                            barrier.wait(timeout=1.0)
                            target = per_arm[arm_key]["local"]
                            t0 = time.monotonic()
                            robot_obj.move_direct(
                                float(target[0]), float(target[1]), float(target[2]),
                                float(tool_t_by_arm[arm_key]),
                            )
                            t1 = time.monotonic()
                            with lock:
                                timing[arm_key] = (t0, t1)
                        except BaseException as exc:
                            with lock:
                                errors.append((arm_key, exc))
                    threads = [
                        threading.Thread(target=one, args=("arm2", arm), daemon=True),
                        threading.Thread(target=one, args=("arm1", robot1), daemon=True),
                    ]
                    for thread in threads:
                        thread.start()
                    try:
                        barrier.wait(timeout=1.0)
                        for thread in threads:
                            thread.join(timeout=max(0.20, period * 2.0))
                    except KeyboardInterrupt:
                        stream_abort.set()
                        for thread in threads:
                            thread.join(timeout=1.0)
                        raise
                    except Exception as exc:
                        errors.append(("barrier", exc))
                        for thread in threads:
                            thread.join(timeout=max(0.20, period * 2.0))
                    if any(thread.is_alive() for thread in threads):
                        raise RuntimeError(f"{label} send thread timeout at {index}/{point_count}")
                    if errors or len(timing) != 2:
                        raise RuntimeError(f"{label} paired send failed at {index}: {errors[:1]}")
                    skew = abs(timing["arm2"][0] - timing["arm1"][0])
                    write_time = max(timing[k][1] - timing[k][0] for k in ("arm2", "arm1"))
                    max_skew_seen = max(max_skew_seen, skew)
                    max_write_seen = max(max_write_seen, write_time)
                    if write_time > max_write:
                        raise RuntimeError(
                            f"{label} serial write {write_time*1000.0:.1f}>{max_write*1000.0:.1f}ms at {index}"
                        )
                    if index == 1 or index == point_count or index % log_stride == 0:
                        profile = float(profiles[index - 1]) if profiles is not None else 0.0
                        print(
                            f"[{label}] {index:03d}/{point_count} lag={lag*1000.0:.1f}ms "
                            f"write={write_time*1000.0:.1f}ms skew={skew*1000.0:.1f}ms profile={profile:.3f}"
                        )
            except KeyboardInterrupt:
                stream_abort.set()
                raise
            print(
                f"[{label}-END] maxLag={max_lag_seen*1000.0:.1f}ms "
                f"maxWrite={max_write_seen*1000.0:.1f}ms maxSkew={max_skew_seen*1000.0:.1f}ms"
            )

        recenter_xy_tol = max(1.0, float(args.laydown_recenter_xy_tolerance_mm))
        recenter_z_tol = max(1.0, float(args.laydown_recenter_z_tolerance_mm))
        recenter_hard_z = max(recenter_z_tol, float(args.laydown_recenter_hard_z_drift_mm))
        recenter_corrections = max(0, int(args.laydown_recenter_correction_attempts))
        recenter_strict_timeout = max(0.5, float(args.laydown_recenter_strict_timeout_sec))
        stable_gap = max(0.05, float(args.laydown_recenter_stable_gap_sec))
        stable_sample_count = max(3, int(args.laydown_recenter_stable_samples))
        stable_xy_span_limit = max(0.5, float(args.laydown_recenter_stable_xy_span_mm))
        stable_z_span_limit = max(0.5, float(args.laydown_recenter_stable_z_span_mm))

        def _feedback_xyz(robot_obj, label: str, attempts: int = 3):
            fb = robot_obj.feedback_retry(
                min(float(args.feedback_timeout), 0.60),
                attempts=max(1, int(attempts)),
                retry_delay=0.12,
                quiet=True,
            )
            if fb is None or not all(k in fb for k in ("x", "y", "z")):
                raise RuntimeError(f"{label} T:105 feedback unavailable")
            pose = np.asarray(
                [float(fb["x"]), float(fb["y"]), float(fb["z"])],
                dtype=np.float64,
            )
            if not np.all(np.isfinite(pose)):
                raise RuntimeError(f"{label} nonfinite T:105 pose: {pose}")
            return pose

        def _recenter_metrics(actual, target):
            actual = np.asarray(actual, dtype=np.float64)
            target = np.asarray(target, dtype=np.float64)
            return {
                "xy_error": float(np.linalg.norm(actual[:2] - target[:2])),
                "z_error": float(abs(actual[2] - target[2])),
                "z_signed": float(actual[2] - target[2]),
                "error_3d": float(np.linalg.norm(actual - target)),
            }

        def _wait_recenter_pair(attempt_label: str):
            # FIX11 bounds the strict gate. If the arms settle slightly below the
            # commanded Z under garment load, the final retry proceeds to the
            # measured-pose stability gate instead of polling indefinitely.
            deadline = time.monotonic() + recenter_strict_timeout
            latest = {}
            passed = {"arm2": False, "arm1": False}
            while time.monotonic() < deadline:
                if _probe_abort_requested():
                    raise KeyboardInterrupt(f"operator abort during {attempt_label}")
                for arm_key, robot_obj in (("arm2", arm), ("arm1", robot1)):
                    remaining = max(0.10, deadline - time.monotonic())
                    fb = robot_obj.feedback(min(0.35, remaining), quiet=True)
                    if fb is None or not all(k in fb for k in ("x", "y", "z")):
                        continue
                    actual = np.asarray(
                        [float(fb["x"]), float(fb["y"]), float(fb["z"])],
                        dtype=np.float64,
                    )
                    latest[arm_key] = actual
                    metrics = _recenter_metrics(actual, recenter_targets[arm_key])
                    print(
                        f"[{attempt_label}-{arm_key.upper()}] "
                        f"actual=({actual[0]:.1f},{actual[1]:.1f},{actual[2]:.1f}) "
                        f"xyErr={metrics['xy_error']:.1f}/{recenter_xy_tol:.1f}mm "
                        f"zErr={metrics['z_error']:.1f}/{recenter_z_tol:.1f}mm "
                        f"zSigned={metrics['z_signed']:+.1f}mm"
                    )
                    passed[arm_key] = bool(
                        metrics["xy_error"] <= recenter_xy_tol
                        and metrics["z_error"] <= recenter_z_tol
                    )
                if all(passed.values()):
                    return True, latest
                time.sleep(max(0.05, min(0.15, float(args.move_poll_sec))))
            return False, latest

        def _pairwise_xy_span(samples):
            points = np.asarray(samples, dtype=np.float64)[:, :2]
            maximum = 0.0
            for i in range(len(points)):
                for j in range(i + 1, len(points)):
                    maximum = max(maximum, float(np.linalg.norm(points[i] - points[j])))
            return maximum

        def _collect_recenter_stability(label: str):
            samples = {"arm2": [], "arm1": []}
            for sample_index in range(1, stable_sample_count + 1):
                if _probe_abort_requested():
                    raise KeyboardInterrupt(f"operator abort during {label}")
                samples["arm2"].append(
                    _feedback_xyz(arm, f"{label}-S{sample_index}-ARM2")
                )
                samples["arm1"].append(
                    _feedback_xyz(robot1, f"{label}-S{sample_index}-ARM1")
                )
                if sample_index < stable_sample_count:
                    _interruptible_wait(stable_gap, f"{label} sample gap")

            anchors = {}
            accepted = True
            for arm_key in ("arm2", "arm1"):
                stack = np.vstack(samples[arm_key])
                anchor = np.median(stack, axis=0)
                xy_span = _pairwise_xy_span(stack)
                z_span = float(np.ptp(stack[:, 2]))
                metrics = _recenter_metrics(anchor, recenter_targets[arm_key])
                xy_ok = metrics["xy_error"] <= recenter_xy_tol
                z_safe = metrics["z_error"] <= recenter_hard_z
                stable_ok = (
                    xy_span <= stable_xy_span_limit
                    and z_span <= stable_z_span_limit
                )
                print(
                    f"[FIX11-RECENTER-STABLE-{arm_key.upper()}] "
                    f"median=({anchor[0]:.1f},{anchor[1]:.1f},{anchor[2]:.1f}) "
                    f"xyErr={metrics['xy_error']:.1f}/{recenter_xy_tol:.1f}mm "
                    f"zErr={metrics['z_error']:.1f}/{recenter_hard_z:.1f}mm "
                    f"xySpan={xy_span:.2f}/{stable_xy_span_limit:.2f}mm "
                    f"zSpan={z_span:.2f}/{stable_z_span_limit:.2f}mm"
                )
                if not xy_ok:
                    print(
                        f"[FIX11-RECENTER-BLOCKED] {arm_key} XY remains outside "
                        f"the {recenter_xy_tol:.1f}mm recenter tolerance"
                    )
                if not z_safe:
                    print(
                        f"[FIX11-RECENTER-BLOCKED] {arm_key} Z drift "
                        f"{metrics['z_signed']:+.1f}mm exceeds the "
                        f"{recenter_hard_z:.1f}mm safety limit"
                    )
                if not stable_ok:
                    print(
                        f"[FIX11-RECENTER-BLOCKED] {arm_key} feedback is not settled"
                    )
                accepted = accepted and xy_ok and z_safe and stable_ok
                anchors[arm_key] = anchor.astype(np.float64, copy=True)
            return accepted, anchors

        recentered_pose = None
        recenter_accept_mode = None
        last_failure = "unknown"
        for attempt_index in range(recenter_corrections + 1):
            attempt_no = attempt_index + 1
            label = f"FIX11-RECENTER-A{attempt_no}"
            if attempt_index == 0:
                print("[FIX11-RECENTER] pair-preserving shared X translation")
            else:
                print(
                    f"[FIX11-RECENTER-CORRECTION] re-send same XYZ/tool_t target "
                    f"({attempt_index}/{recenter_corrections})"
                )
            _pair_t104(label, recenter_targets, float(args.laydown_recenter_speed))
            arrived, latest = _wait_recenter_pair(label)

            for arm_key, actual in latest.items():
                metrics = _recenter_metrics(actual, recenter_targets[arm_key])
                if metrics["z_error"] > recenter_hard_z:
                    raise RuntimeError(
                        f"{arm_key} recenter hard Z drift "
                        f"{metrics['z_signed']:+.1f}mm exceeds {recenter_hard_z:.1f}mm"
                    )

            if arrived:
                print(
                    f"[FIX11-RECENTER-STRICT-PASS] attempt={attempt_no}; "
                    "collecting measured stability samples before curve rebuild"
                )
                _interruptible_wait(
                    args.laydown_recenter_settle_sec,
                    f"FIX11 recenter settle attempt {attempt_no}",
                )
                accepted, anchors = _collect_recenter_stability(
                    f"FIX11-RECENTER-STABILITY-A{attempt_no}"
                )
                if accepted:
                    recentered_pose = anchors
                    recenter_accept_mode = "STRICT_THEN_ACTUAL_MEDIAN"
                    break
                last_failure = "strict arrival passed but measured pose did not settle safely"
            else:
                print(
                    f"[FIX11-RECENTER-STRICT-FAIL] attempt={attempt_no}; "
                    f"strict XY/Z gate was not reached within {recenter_strict_timeout:.1f}s"
                )
                last_failure = "dedicated XY/Z arrival tolerances not reached"

            if attempt_index < recenter_corrections:
                continue

            # Final-attempt fallback: accept only a stationary, XY-aligned pose
            # whose Z droop remains inside the existing hard safety envelope.
            _interruptible_wait(
                args.laydown_recenter_settle_sec,
                "FIX11 final recenter settle before actual-pose evaluation",
            )
            accepted, anchors = _collect_recenter_stability(
                "FIX11-RECENTER-FINAL-ACTUAL"
            )
            if accepted:
                recentered_pose = anchors
                recenter_accept_mode = "SAFE_STABLE_ACTUAL_REANCHOR"
                print(
                    "[FIX11-RECENTER-REANCHOR] strict target was not reached, "
                    "but both arms are XY-aligned, stationary, and inside the hard Z limit"
                )
                break
            raise RuntimeError(
                f"FIX11 recenter blocked after {attempt_no} attempt(s): {last_failure}; "
                "actual-pose stability gate also failed"
            )

        if recentered_pose is None:
            raise RuntimeError("FIX11 recenter ended without a safe settled pose")

        for arm_key in ("arm2", "arm1"):
            nominal = recenter_targets[arm_key]
            actual = recentered_pose[arm_key]
            metrics = _recenter_metrics(actual, nominal)
            print(
                f"[FIX11-RECENTER-ANCHOR-{arm_key.upper()}] "
                f"nominal=({nominal[0]:.2f},{nominal[1]:.2f},{nominal[2]:.2f}) "
                f"actual=({actual[0]:.2f},{actual[1]:.2f},{actual[2]:.2f}) "
                f"dZ={metrics['z_signed']:+.2f}mm mode={recenter_accept_mode}"
            )

        # The exact curved path is regenerated from the measured, settled recenter
        # pose. This removes any first-stream jump and preserves the actual pair
        # separation rather than assuming perfect T:104 arrival.
        actual_pose = {
            "arm2": recentered_pose["arm2"].copy(),
            "arm1": recentered_pose["arm1"].copy(),
        }
        actual_board = {
            arm_key: _arm_xy_to_board(arm_key, pose[0], pose[1])
            for arm_key, pose in actual_pose.items()
        }
        start_board = {
            "arm2": actual_board["arm2"].copy(),
            "arm1": actual_board["arm1"].copy(),
        }
        pair_mid_after = 0.5 * (start_board["arm2"] + start_board["arm1"])
        start_separation = float(np.linalg.norm(start_board["arm1"] - start_board["arm2"]))

        common_available_rise = min(
            float(args.z_max) - z_margin - float(actual_pose[k][2])
            for k in ("arm2", "arm1")
        )
        curve_rise = min(requested_rise, common_available_rise)
        if curve_rise < float(args.laydown_min_curve_rise_mm) - 1e-6:
            raise RuntimeError(
                f"FIX11 post-recenter curve-rise room too small: {curve_rise:.1f}mm "
                f"< {float(args.laydown_min_curve_rise_mm):.1f}mm"
            )

        back_board = {
            k: start_board[k] + back_direction * amplitude for k in ("arm2", "arm1")
        }
        final_board = {
            k: start_board[k] + final_direction * amplitude for k in ("arm2", "arm1")
        }
        final_z = {
            "arm2": _surface_z("arm2", final_board["arm2"][0], final_board["arm2"][1])
                    + float(args.laydown_arm2_final_clearance_mm),
            "arm1": _surface_z("arm1", final_board["arm1"][0], final_board["arm1"][1])
                    + float(args.laydown_arm1_final_clearance_mm),
        }
        peak_z = {
            k: float(actual_pose[k][2]) + curve_rise for k in ("arm2", "arm1")
        }
        for k in ("arm2", "arm1"):
            _require_in_range(f"{k} FIX11 final_z", final_z[k], args.z_min, args.z_max)
            _require_in_range(f"{k} FIX11 peak_z", peak_z[k], args.z_min, args.z_max)
            if final_z[k] >= peak_z[k] - 10.0:
                raise RuntimeError(
                    f"{k} final support Z {final_z[k]:.1f} is not sufficiently below "
                    f"post-recenter peak {peak_z[k]:.1f}"
                )

        backswing = []
        ascent_values = []
        for index in range(1, back_steps + 1):
            t = float(index) / float(back_steps)
            eased = t * t * (3.0 - 2.0 * t)
            theta = 0.5 * math.pi * eased
            horizontal_ratio = 1.0 - math.cos(theta)
            rise_ratio = math.sin(theta)
            per_arm = {}
            for k in ("arm2", "arm1"):
                board_at = start_board[k] + back_direction * amplitude * horizontal_ratio
                z_at = float(actual_pose[k][2]) + curve_rise * rise_ratio
                per_arm[k] = {
                    "board": np.asarray(board_at, dtype=np.float64),
                    "local": _local_target(k, board_at, z_at),
                }
            backswing.append(per_arm)
            ascent_values.append(rise_ratio)

        forward = []
        remain_values = []
        for index in range(1, forward_steps + 1):
            t = float(index) / float(forward_steps)
            smooth = t * t * (3.0 - 2.0 * t)
            vertical_s = float(smooth ** vertical_gamma)
            remain = _exp_remaining(vertical_s)
            per_arm = {}
            for k in ("arm2", "arm1"):
                board_at = back_board[k] + final_direction * (2.0 * amplitude * smooth)
                z_at = final_z[k] + (peak_z[k] - final_z[k]) * remain
                per_arm[k] = {
                    "board": np.asarray(board_at, dtype=np.float64),
                    "local": _local_target(k, board_at, z_at),
                }
            forward.append(per_arm)
            remain_values.append(remain)

        # Final preflight uses the measured recenter pose as the true anchor.
        for k in ("arm2", "arm1"):
            _check_board(f"{k} actual recenter anchor", start_board[k])
            _check_board(f"{k} exact back", back_board[k])
            _check_board(f"{k} exact final", final_board[k])
            _check_local(f"{k} actual recenter local", actual_pose[k])
        for phase_name, sequence in (("BACK-FINAL", backswing), ("FORWARD-FINAL", forward)):
            previous = {
                k: actual_pose[k].copy() if phase_name == "BACK-FINAL"
                else backswing[-1][k]["local"].copy()
                for k in ("arm2", "arm1")
            }
            for index, per_arm in enumerate(sequence, 1):
                sep = float(np.linalg.norm(per_arm["arm1"]["board"] - per_arm["arm2"]["board"]))
                if abs(sep - start_separation) > 1e-3:
                    raise RuntimeError(
                        f"{phase_name} pair separation changed at {index}: {sep:.3f}mm"
                    )
                for k in ("arm2", "arm1"):
                    _check_board(f"{phase_name}-{index}-{k}", per_arm[k]["board"])
                    _check_local(f"{phase_name}-{index}-{k}", per_arm[k]["local"])
                    step = float(np.linalg.norm(per_arm[k]["local"] - previous[k]))
                    if step > step_max + 1e-6:
                        raise RuntimeError(
                            f"{phase_name}-{index}-{k} actual-anchor step "
                            f"{step:.1f}>{step_max:.1f}mm"
                        )
                    previous[k] = per_arm[k]["local"].copy()

        z_adjust = {
            k: float(actual_pose[k][2] - recenter_targets[k][2])
            for k in ("arm2", "arm1")
        }
        print("\n========== FIX11 FINAL PREFLIGHT AFTER RECENTER ==========")
        print(
            f"actual anchors ARM2=({start_board['arm2'][0]:.1f},{start_board['arm2'][1]:.1f}) "
            f"ARM1=({start_board['arm1'][0]:.1f},{start_board['arm1'][1]:.1f}); "
            f"midpointX={pair_mid_after[0]:.1f}mm centerResidual="
            f"{pair_mid_after[0]-center_board[0]:+.1f}mm separation={start_separation:.1f}mm"
        )
        print(
            f"actual reanchor Z adjustment ARM2={z_adjust['arm2']:+.1f}mm "
            f"ARM1={z_adjust['arm1']:+.1f}mm; tool_t preserved "
            f"ARM2={tool_t_by_arm['arm2']:.4f} ARM1={tool_t_by_arm['arm1']:.4f}"
        )
        print(
            f"curveRise={curve_rise:.1f}mm; final support ARM2={final_z['arm2']:.1f}mm "
            f"ARM1={final_z['arm1']:.1f}mm; "
            f"all {back_steps + forward_steps} regenerated pair targets passed final preflight"
        )
        print("==========================================================\n")

        _stream_pair(
            "FIX11-CURVE-ASCENT", backswing, back_duration, ascent_values
        )
        _interruptible_wait(args.laydown_reversal_hold_sec, "FIX11 reversal hold")
        _stream_pair(
            "FIX11-FORWARD-EXP-LAYDOWN", forward, forward_duration,
            remain_values, forward_time_fractions,
        )

        final_targets = {
            k: forward[-1][k]["local"].copy() for k in ("arm2", "arm1")
        }
        print("[FIX11-FINAL-LOCK] reassert final support CLOSED with paired T:104")
        _pair_t104(
            "FIX11-FINAL-LOCK", final_targets, float(args.laydown_final_lock_speed)
        )
        _interruptible_wait(args.laydown_final_lock_wait_sec, "FIX11 final lock wait")
        verify_deadline = time.monotonic() + max(0.20, float(args.laydown_final_verify_timeout_sec))
        verify_tolerance = max(5.0, float(args.laydown_final_verify_tolerance_mm))
        verified = {}
        while time.monotonic() < verify_deadline and len(verified) < 2:
            if _probe_abort_requested():
                raise KeyboardInterrupt("operator abort during FIX11 final support verification")
            for arm_key, robot_obj in (("arm2", arm), ("arm1", robot1)):
                if arm_key in verified:
                    continue
                remaining = max(0.10, verify_deadline - time.monotonic())
                fb = robot_obj.feedback(min(0.35, remaining), quiet=True)
                if fb is None or not all(k in fb for k in ("x", "y", "z")):
                    continue
                actual = np.asarray(
                    [float(fb["x"]), float(fb["y"]), float(fb["z"])], dtype=np.float64
                )
                error = float(np.linalg.norm(actual - final_targets[arm_key]))
                print(
                    f"[FIX11-FINAL-{arm_key.upper()}] actual=({actual[0]:.1f},{actual[1]:.1f},{actual[2]:.1f}) "
                    f"error={error:.1f}mm tolerance={verify_tolerance:.1f}mm"
                )
                if error <= verify_tolerance:
                    verified[arm_key] = actual
            if len(verified) < 2:
                time.sleep(0.05)
        if len(verified) < 2:
            missing = [k for k in ("arm2", "arm1") if k not in verified]
            print(
                f"[FIX11-FINAL-FEEDBACK-WARN] short confirmation incomplete for {missing}; "
                "no return-to-origin recovery will run"
            )
        else:
            print("[FIX11-FINAL-FEEDBACK] both final support targets confirmed")
        _interruptible_wait(args.laydown_final_settle_sec, "FIX11 final support settle")

        release_angle = float(args.laydown_release_open_angle)
        print(f"[FIX11-RELEASE] paired OPEN angle={release_angle:.3f}")
        barrier = threading.Barrier(3)
        errors = []
        lock = threading.Lock()
        def release_one(label: str, robot_obj):
            try:
                barrier.wait(timeout=2.0)
                robot_obj.gripper_open(release_angle, args.grip_spd, args.grip_acc)
            except Exception as exc:
                with lock:
                    errors.append((label, exc))
        threads = [
            threading.Thread(target=release_one, args=("ARM2", arm), daemon=True),
            threading.Thread(target=release_one, args=("ARM1", robot1), daemon=True),
        ]
        for thread in threads:
            thread.start()
        try:
            barrier.wait(timeout=2.0)
        except KeyboardInterrupt:
            for thread in threads:
                thread.join(timeout=1.0)
            raise
        except Exception as exc:
            errors.append(("BARRIER", exc))
        for thread in threads:
            thread.join(timeout=4.0)
        if errors or any(thread.is_alive() for thread in threads):
            raise RuntimeError(f"FIX11 paired release failed: {errors[:1]}")
        _interruptible_wait(args.laydown_release_settle_sec, "FIX11 release settle")

        if not _safe_return_to_standby(
            reason="automatic FIX11 laydown complete",
            release_first=False,
            release_angle=release_angle,
        ):
            raise RuntimeError("FIX11 automatic standby return did not fully complete")
        print("[FIX11-COMPLETE] curved laydown, release, and automatic standby completed")
        if preview is not None:
            preview.set_status("FIX11 COMPLETE | robot at standby | starting RAW ROI stability gate")
        try:
            mask_ok = _run_post_laydown_mask_only()
            print(f"[POST-MASK-RESULT] {'SUCCESS' if mask_ok else 'STOPPED'}; FIX11 robot completion remains valid")
        except BaseException as exc:
            print(f"[POST-MASK-FATAL-BOUNDARY] {exc!r}; FIX11 robot completion remains valid")
        return True

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
        key = read_operator_key()
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

    def execute_descent_probe():
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
                    print("gripper_close=AUTO-PENDING | first-grasp lift/transfer=AUTO-PENDING")
                    print(
                        f"[AUTO-PLACE] open to {args.post_contact_open_percent:.1f}% "
                        f"(angle={wider_angle:.3f}) -> close angle={args.grasp_close_angle:.3f} "
                        f"-> adaptive lift preferred_Z={pickup_lift_z:.3f} "
                        f"-> one-shot ARM2 center=({board_center_x:.3f},{board_center_y:.3f}) "
                        f"-> FIX11 handoff/air-spread/curved-laydown continues at board center; no basket retention retry"
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
                return execute_grasp_lift_test()
            return True
        except KeyboardInterrupt:
            print("\n[OPERATOR-STOP] propagating interrupt to Ctrl+C safe-return handler")
            hover_ready = False
            raise
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

    def execute_grasp_lift_test():
        nonlocal grasp_lift_plan, return_completed
        return_completed = False
        if grasp_lift_plan is None:
            print("[AUTO-PLACE] no armed plan; complete a confirmed-contact descent first")
            return False
        if arm is None:
            print("[AUTO-PLACE-BLOCKED] --send required")
            grasp_lift_plan = None
            return False

        plan = dict(grasp_lift_plan)
        grasp_lift_plan = None
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

            print("\n========== AUTOMATIC SINGLE GRASP + DIRECT CENTER HANDOFF ==========")
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
                f"4) accept the first grasp unconditionally after lift and send ONE direct T:104 "
                f"to board center at speed={float(args.board_transit_speed):.2f}; no retention test, no regrasp"
            )
            print("5) at board center keep ARM2 CLOSED, wait 3s, then run the existing ARM1 Z-150mm grasp + diagonal-rise handoff")
            print("No post-lift retention diagnosis | no automatic regrasp | no ARM2 release | no standby return")
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

            verify_target = float(plan["close_angle"])
            verify_primary = 2.80
            verify_motion_min = 0.30
            verify_torque_min = 220.0
            verify_prev = float(close_t)
            verify_cloth_contact = False
            verify_accepted = close_t >= verify_primary

            for verify_attempt in range(1, 3):
                if verify_accepted:
                    break

                arm.gripper_open(verify_target, 0.0, 0.0)
                _interruptible_wait(0.25, f"basket close verify {verify_attempt}")

                verify_fb = arm.feedback_retry(
                    args.feedback_timeout, attempts=4, retry_delay=0.20
                )
                if verify_fb is None or not all(k in verify_fb for k in ("x", "y", "z", "t")):
                    raise RuntimeError("T:105 feedback failed during upper basket close verification")

                verify_t = float(verify_fb["t"])
                verify_motion = verify_t - float(open_t)
                verify_tor_h = abs(float(verify_fb.get("torH", 0.0)))
                stable_cloth = abs(verify_t - verify_prev) <= 0.06

                verify_cloth_contact = bool(
                    stable_cloth
                    and verify_motion >= verify_motion_min
                    and verify_tor_h >= verify_torque_min
                )

                print(
                    f"[BASKET-CLOSE-VERIFY-FIX133] attempt={verify_attempt} "
                    f"target={verify_target:.3f} actual={verify_t:.3f} "
                    f"motion={verify_motion:.3f}rad absTorH={verify_tor_h:.0f} "
                    f"stableCloth={stable_cloth} clothContact={verify_cloth_contact}"
                )

                close_fb = verify_fb
                close_t = verify_t

                verify_accepted = bool(
                    close_t >= verify_primary
                    or verify_cloth_contact
                )

                verify_prev = verify_t

            if not verify_accepted:
                raise RuntimeError(
                    f"upper basket close not secured before final latch: "
                    f"actual={close_t:.3f} motion={close_t-float(open_t):.3f}rad"
                )

            pre_final_t = float(close_t)
            pre_final_cloth_contact = bool(verify_cloth_contact)

            final_close_target = 3.32
            arm.gripper_open(final_close_target, 0.0, 0.0)
            _interruptible_wait(0.36, "upper basket final close")

            arm.send({"T": 107, "tor": 1000}, delay=0.55)

            final_fb = arm.feedback_retry(
                args.feedback_timeout, attempts=4, retry_delay=0.20
            )
            if final_fb is None or not all(k in final_fb for k in ("x", "y", "z", "t")):
                raise RuntimeError("T:105 feedback failed after upper basket final torque latch")

            final_t = float(final_fb["t"])
            final_motion = final_t - float(open_t)
            final_tor_h = abs(float(final_fb.get("torH", 0.0)))

            final_cloth_latch = bool(
                pre_final_cloth_contact
                and final_motion >= verify_motion_min
                and final_t >= pre_final_t - 0.08
            )

            final_ok = bool(
                final_t >= verify_primary
                or final_cloth_latch
            )

            print(
                f"[FIX133-UPPER-BASKET-GRIP-LATCH] "
                f"finalClose={final_close_target:.2f}rad "
                f"torque=1000 feedback={final_t:.3f}rad "
                f"motion={final_motion:.3f}rad absTorH={final_tor_h:.0f} "
                f"clothLatch={final_cloth_latch} liftAuthorized={final_ok}"
            )

            if not final_ok:
                raise RuntimeError(
                    f"upper basket final grip latch unverified: "
                    f"actual={final_t:.3f} motion={final_motion:.3f}rad; lift blocked"
                )

            close_fb = final_fb
            close_t = final_t

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
                    "-> preferred height reached; first grasp accepted, continue directly to board center"
                )
            else:
                if not bool(lift_result.get("stalled", False)):
                    raise RuntimeError(
                        "initial basket lift did not reach target and did not confirm physical saturation"
                    )
                # V18 policy: the first physical lift saturation is the end of the lift phase.
                # Do NOT demand another +120 mm, do NOT hop, and do NOT re-lift.  The
                # achieved feedback Z becomes the commanded Z for the one-shot direct XY
                # transfer immediately under FIX8 single-grasp policy. Any unavoidable Z droop during that long move is diagnostic
                # only and never creates another recovery loop.
                print(
                    f"[BASKET-LIFT-SATURATION-ACCEPTED] achieved_z={transit_z:.3f} "
                    "-> no +120mm gate, no hop, no re-lift; first grasp accepted, continue directly to board center"
                )

            # FIX8 single-grasp policy: the first basket grasp is always accepted
            # once the existing close command and lift phase have completed.
            # No post-lift torS+torE retention diagnosis is performed, and no
            # automatic release/regrasp loop can run.
            print(
                f"[SINGLE-GRASP-ACCEPTED] achieved_z={transit_z:.3f}; "
                "retention diagnosis skipped -> direct board-center handoff authorized"
            )

            direct_distance = float(
                np.linalg.norm(
                    np.asarray(
                        [float(plan["board_center_x"]), float(plan["board_center_y"])],
                        dtype=np.float64,
                    ) - transit_start[:2]
                )
            )
            print(
                f"[AUTO-PLACE:4/6] DIRECT board-center goal distance={direct_distance:.1f}mm "
                f"from=({transit_start[0]:.3f},{transit_start[1]:.3f},{transit_z:.3f}) "
                f"to=({plan['board_center_x']:.3f},{plan['board_center_y']:.3f},{transit_z:.3f})"
            )
            # Exactly one T:104 command is used for the basket-to-board transfer.
            # V18: after the initial fast lift, either normal target reach OR the first
            # confirmed physical saturation immediately authorizes this transfer.
            # The achieved lift Z is reused as the commanded transit Z.  During the long
            # direct move, Z tracking is diagnostic-only; unavoidable droop is ignored.
            # Final release still waits for settled XY arrival near the board center.
            arm.move_goal(
                args.move_command,
                plan["board_center_x"],
                plan["board_center_y"],
                transit_z,
                plan["close_angle"],
                args.board_transit_speed,
            )
            board_center_pose = wait_for_direct_xy_arrival(
                "BOARD-CENTER-DIRECT",
                plan["board_center_x"],
                plan["board_center_y"],
                transit_z,
                allow_abort=True,
            )

            if _probe_abort_requested():
                raise KeyboardInterrupt("operator abort before V25 handoff hold")
            print(
                f"[V25-HANDOFF] ARM2 reached board-center handoff actual="
                f"({board_center_pose[0]:.3f},{board_center_pose[1]:.3f},{board_center_pose[2]:.3f}); "
                f"gripper remains CLOSED"
            )
            print(f"[V25-HANDOFF] ARM2 stationary hold {float(args.handoff_settle_sec):.1f}s")
            _interruptible_wait(args.handoff_settle_sec, "ARM2 handoff settle")
            spread_state = _execute_arm1_grasp_and_diagonal_rise(board_center_pose)
            if bool(args.dual_arc_laydown):
                _execute_dual_arc_laydown(spread_state)
                print("[V25-STOP] FIX11 scope complete: curved laydown + release + automatic standby")
            else:
                print("[V25-STOP] FIX8-compatible scope complete: dual hang + synchronized 50mm-per-arm air spread; both arms CLOSED and stationary")
            return True
        except KeyboardInterrupt:
            print("\n[AUTO-PLACE-OPERATOR-STOP] propagating interrupt to Ctrl+C safe-return handler")
            raise
        except Exception as exc:
            print(f"[AUTO-PLACE-ERROR] {exc}")
            print("[STATE] ARM2 torque remains ON; no additional release or motion is executed")
            return False

    return_in_progress = False
    return_completed = False

    def _shutdown_wait_waypoint(robot_obj, label: str, x: float, y: float, z: float):
        deadline = time.time() + max(4.0, float(args.move_timeout))
        target = np.asarray([float(x), float(y), float(z)], dtype=np.float64)
        last_pose = None
        last_error = float("inf")
        while time.time() < deadline:
            time.sleep(max(0.08, float(args.move_poll_sec)))
            fb = robot_obj.feedback(args.feedback_timeout, quiet=True)
            if fb is None or not all(k in fb for k in ("x", "y", "z")):
                continue
            actual = np.asarray([float(fb["x"]), float(fb["y"]), float(fb["z"])], dtype=np.float64)
            last_pose = actual
            last_error = float(np.linalg.norm(actual - target))
            print(
                f"[{label}] feedback=({actual[0]:.1f},{actual[1]:.1f},{actual[2]:.1f}) "
                f"error={last_error:.1f}mm"
            )
            if last_error <= float(args.move_tolerance_mm):
                time.sleep(max(0.0, float(args.move_wait)))
                return actual
        pose_text = "unavailable" if last_pose is None else f"({last_pose[0]:.1f},{last_pose[1]:.1f},{last_pose[2]:.1f})"
        raise RuntimeError(
            f"{label} timeout target=({x:.1f},{y:.1f},{z:.1f}) "
            f"last={pose_text} error={last_error:.1f}mm"
        )

    def _safe_return_to_standby(
        reason: str,
        release_first: bool,
        release_angle: float,
    ) -> bool:
        nonlocal return_in_progress, return_completed
        if return_completed:
            print("[SAFE-RETURN] already completed")
            return True
        if return_in_progress:
            print("[SAFE-RETURN] already in progress")
            return False
        return_in_progress = True
        success = True
        try:
            if not args.send:
                print("[SAFE-RETURN] --send inactive; no robot commands required")
                return_completed = True
                return True

            connected = []
            if arm1 is not None:
                connected.append((
                    "ARM1", arm1,
                    np.asarray([
                        float(args.arm1_standby_roarm_x),
                        float(args.arm1_standby_roarm_y),
                        float(args.arm1_standby_roarm_z),
                    ], dtype=np.float64),
                    float(args.arm1_standby_roarm_t),
                ))
            else:
                print("[SAFE-RETURN] ARM1 serial was never opened")
            if arm is not None:
                connected.append((
                    "ARM2", arm,
                    np.asarray([
                        float(args.standby_roarm_x),
                        float(args.standby_roarm_y),
                        float(args.standby_roarm_z),
                    ], dtype=np.float64),
                    float(args.standby_roarm_t),
                ))
            if not connected:
                print("[SAFE-RETURN] no connected arms")
                return_completed = True
                return True

            print("\n========== SAFE RETURN TO STANDBY START ==========")
            print(f"reason={reason}")
            print(
                f"release_first={release_first} release_angle={float(release_angle):.2f}rad -> "
                f"vertical clear >= {float(args.shutdown_clear_z):.1f}mm -> high-Z standby XY -> standby Z"
            )
            print("Whole-arm torque remains ON so the arms can hold standby safely.")
            print("==================================================\n")

            if release_first:
                barrier = threading.Barrier(len(connected) + 1)
                errors = []
                lock = threading.Lock()
                def open_worker(label, robot_obj):
                    try:
                        barrier.wait(timeout=3.0)
                        robot_obj.gripper_open(
                            float(release_angle), float(args.grip_spd), float(args.grip_acc)
                        )
                    except BaseException as exc:
                        with lock:
                            errors.append((label, exc))
                threads = []
                for label, robot_obj, _, _ in connected:
                    thread = threading.Thread(target=open_worker, args=(label, robot_obj), daemon=True)
                    threads.append(thread)
                    thread.start()
                try:
                    barrier.wait(timeout=3.0)
                except BaseException as exc:
                    errors.append(("BARRIER", exc))
                for thread in threads:
                    thread.join(timeout=4.0)
                if errors or any(thread.is_alive() for thread in threads):
                    success = False
                    print(f"[SAFE-RETURN-WARN] initial gripper release issue: {errors[:1]}")
                time.sleep(max(0.0, float(args.shutdown_release_settle_sec)))

            # ARM1 first, then ARM2. Each returns via vertical clear, high-Z
            # standby XY, then its directly taught standby Z.
            for label, robot_obj, standby_xyz, standby_t in connected:
                try:
                    fb = robot_obj.feedback_retry(
                        args.feedback_timeout, attempts=4, retry_delay=0.25, quiet=True
                    )
                    if fb is None or not all(k in fb for k in ("x", "y", "z")):
                        raise RuntimeError("T:105 feedback unavailable")
                    current = np.asarray(
                        [float(fb["x"]), float(fb["y"]), float(fb["z"])],
                        dtype=np.float64,
                    )
                    clear_z = min(
                        float(args.z_max),
                        max(float(args.shutdown_clear_z), float(current[2]), float(standby_xyz[2])),
                    )
                    print(
                        f"[SAFE-RETURN] {label} current=({current[0]:.1f},{current[1]:.1f},{current[2]:.1f}) "
                        f"standby=({standby_xyz[0]:.1f},{standby_xyz[1]:.1f},{standby_xyz[2]:.1f}) "
                        f"clear_z={clear_z:.1f}"
                    )
                    if float(current[2]) < clear_z - 2.0:
                        robot_obj.move_goal(
                            args.move_command,
                            float(current[0]), float(current[1]), clear_z,
                            standby_t, float(args.standby_speed),
                        )
                        _shutdown_wait_waypoint(
                            robot_obj, f"RETURN-{label}-1/3-VERTICAL-CLEAR",
                            float(current[0]), float(current[1]), clear_z,
                        )
                    else:
                        print(f"[RETURN-{label}-1/3] vertical clear skipped")

                    robot_obj.move_goal(
                        args.move_command,
                        float(standby_xyz[0]), float(standby_xyz[1]), clear_z,
                        standby_t, float(args.standby_speed),
                    )
                    _shutdown_wait_waypoint(
                        robot_obj, f"RETURN-{label}-2/3-HIGH-STANDBY-XY",
                        float(standby_xyz[0]), float(standby_xyz[1]), clear_z,
                    )
                    robot_obj.move_goal(
                        args.move_command,
                        float(standby_xyz[0]), float(standby_xyz[1]), float(standby_xyz[2]),
                        standby_t, float(args.standby_speed),
                    )
                    _shutdown_wait_waypoint(
                        robot_obj, f"RETURN-{label}-3/3-STANDBY",
                        float(standby_xyz[0]), float(standby_xyz[1]), float(standby_xyz[2]),
                    )
                    robot_obj.gripper_open(
                        float(args.shutdown_gripper_angle),
                        float(args.grip_spd), float(args.grip_acc),
                    )
                    print(f"[SAFE-RETURN] {label} standby complete; gripper slightly open")
                except BaseException as exc:
                    success = False
                    print(f"[SAFE-RETURN-ERROR] {label}: {exc}")

            return_completed = bool(success)
            print(
                f"========== SAFE RETURN TO STANDBY {'FINISHED' if success else 'INCOMPLETE'} ==========\n"
            )
            return success
        finally:
            return_in_progress = False

    def _ctrlc_safe_return_to_standby(reason: str = "KeyboardInterrupt"):
        return _safe_return_to_standby(
            reason=reason,
            release_first=True,
            release_angle=float(args.shutdown_gripper_angle),
        )

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

    terminal = TerminalKeyReader(
        enabled=not args.no_terminal_control,
        raw=not args.terminal_line_mode,
    )
    terminal.start()

    def read_operator_key() -> int:
        terminal_key = terminal.read_key()
        if terminal_key != 255:
            return int(terminal_key)
        if preview is not None:
            return int(preview.pop_key())
        return 255

    def set_preview_status(message: str) -> None:
        if preview is not None:
            preview.set_status(message)

    print("[KEYS] B mode | H hover | D descent | Enter execute prepared H/D | U cancel | Q quit | Ctrl+C abort+safe return")
    print("[RAW-PREVIEW] C/mouse recalibration is disabled in this runtime because the live window is intentionally RAW; use the dedicated undistorted calibration script when calibration is required.")
    for i, spec in enumerate(POINT_SPECS, 1):
        print(f"  {i}. {spec[0]}: {spec[1]}")

    keyboard_interrupt_received = False
    try:
        while True:
            key = read_operator_key()
            if key == 255:
                time.sleep(0.01)
                continue
            if key in (ord('q'), 27):
                break
            elif key in (ord('b'), ord('B')):
                basket_mode = not basket_mode
                if not basket_mode:
                    active, pending, hover_plan, descent_plan, grasp_lift_plan, hover_ready = False, None, None, None, None, False
                print(f"[MODE] BASKET MODE {'ON' if basket_mode else 'OFF'}")
                set_preview_status(f"BASKET MODE {'ON' if basket_mode else 'OFF'} | RAW preview | model unloaded")
            elif key in (ord('c'), ord('C')):
                print(
                    "[CALIB-BLOCKED] this runtime window is RAW-only by design. "
                    "Do not save raw mouse pixels into the existing undistorted basket affine. "
                    "Use the dedicated FIX11/ELP undistorted calibration script if the basket moves."
                )
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
                    set_preview_status("ROBOT MOTION: HOVER | RAW camera continues live | no inference")
                    execute_hover_plan()
                    set_preview_status("HOVER COMPLETE | RAW preview | press D to prepare descent")
                elif descent_plan is not None:
                    set_preview_status("ROBOT MOTION: PICK/SPREAD/LAYDOWN | RAW camera continues live | no inference")
                    execute_descent_probe()
                    set_preview_status("ROBOT CYCLE RETURNED | RAW preview live")
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
                    calib.load(
                        args.calib_file,
                        camera_geometry_metadata,
                        allow_legacy=bool(args.allow_legacy_basket_calib),
                    )
                    active, pending, hover_plan, descent_plan, grasp_lift_plan, hover_ready = False, None, None, None, None, False
                except Exception as exc:
                    print(f"[CALIB-ERROR] load failed: {exc}")
            elif key in (ord('h'), ord('H')):
                if prepare_hover_plan():
                    set_preview_status("HOVER PLAN READY | press Enter | RAW preview live")
            elif key in (ord('d'), ord('D')):
                if prepare_descent_plan():
                    set_preview_status("DESCENT PLAN READY | press Enter | RAW preview live")
            elif key in (ord('f'), ord('F')):
                print("[FEEDBACK] --send not active" if arm is None else arm.feedback(args.feedback_timeout))
            elif key in (ord('t'), ord('T')):
                print("[TORQUE] --send not active") if arm is None else arm.torque_off()
            elif key in (ord('n'), ord('N')):
                print("[TORQUE] --send not active") if arm is None else arm.torque_on()
            elif key in (ord('g'), ord('G')):
                print("[GRIPPER] --send not active") if arm is None else arm.gripper_open(args.grip_open, args.grip_spd, args.grip_acc)
    except KeyboardInterrupt:
        keyboard_interrupt_received = True
        print("\n[CTRL+C] safe standby return requested")
        raise
    finally:
        terminal.stop()
        if keyboard_interrupt_received:
            try:
                _ctrlc_safe_return_to_standby("KeyboardInterrupt")
            except BaseException as exc:
                print(f"[CTRLC-STANDBY-FATAL] {exc}")
        if arm1:
            arm1.close()
        if arm:
            arm.close()
        if preview is not None:
            preview.stop()
        if not args.no_window:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    raise SystemExit(_run_main_with_session_log())
