#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Bottom-garment rotation and position-adjustment controller.

This module provides corrected-camera perception support, contour-based dual-arm
rotation planning, and the D58 circumcenter position-adjustment interface used by
the integrated lower-garment runtime.

Robot geometry, safety checks, motion sequencing, calibrated coordinates, and
runtime parameters are intentionally preserved.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import platform
import select
import shutil
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

# Docker / X11 stability before cv2 import.
os.environ.setdefault("QT_X11_NO_MITSHM", "1")
os.environ.setdefault("GDK_DISABLE_SHM", "1")
os.environ.setdefault("NO_AT_BRIDGE", "1")

import cv2
import numpy as np

try:
    import serial
except Exception:
    serial = None

try:
    from ultralytics import YOLO
except Exception:
    YOLO = None

try:
    import termios
    import tty
except Exception:
    termios = None
    tty = None


class TerminalKeyReader:
    """Non-blocking terminal single-key input, independent of OpenCV window focus.

    I/L/R/Q are consumed immediately in an interactive TTY. Enter is returned as
    key code 13. If raw terminal mode cannot be enabled, it falls back to line mode
    so `i<Enter>` / `l<Enter>` / blank Enter still work.
    """
    def __init__(self) -> None:
        self.is_tty = bool(getattr(sys.stdin, "isatty", lambda: False)())
        self.raw_active = False
        self.orig_attrs = None

    def start(self) -> None:
        if not self.is_tty:
            print("[D57-8-TERMINAL] stdin is not a TTY; OpenCV keys remain available")
            return
        if termios is not None and tty is not None:
            try:
                self.orig_attrs = termios.tcgetattr(sys.stdin.fileno())
                tty.setcbreak(sys.stdin.fileno())
                self.raw_active = True
                print("[D57-8-TERMINAL] RAW KEY MODE ON: I/L/R/Q and Enter work directly in terminal")
                return
            except Exception as exc:
                print(f"[D57-8-TERMINAL] raw mode unavailable ({exc}); using line mode")
        print("[D57-8-TERMINAL] LINE MODE: type i/l/r/q then Enter; blank Enter executes")

    def stop(self) -> None:
        if self.raw_active and self.orig_attrs is not None and termios is not None:
            try:
                termios.tcsetattr(sys.stdin.fileno(), termios.TCSADRAIN, self.orig_attrs)
            except Exception:
                pass
        self.raw_active = False

    def read_key(self) -> int:
        if not self.is_tty:
            return 255
        try:
            ready, _, _ = select.select([sys.stdin], [], [], 0.0)
        except Exception:
            return 255
        if not ready:
            return 255
        try:
            if self.raw_active:
                ch = sys.stdin.read(1)
                if ch in ("\r", "\n"):
                    return 13
                if ch == "\x1b":
                    return 27
                if ch == "\x03":
                    raise KeyboardInterrupt
                return ord(ch.lower()) if ch else 255
            line = sys.stdin.readline()
            text = (line or "").strip().lower()
            if text in ("", "enter"):
                return 13
            if text in ("q", "quit", "exit"):
                return ord("q")
            if text in ("i", "infer"):
                return ord("i")
            if text in ("l", "lock"):
                return ord("l")
            if text in ("r", "reset", "discard"):
                return ord("r")
            return 255
        except KeyboardInterrupt:
            raise
        except Exception:
            return 255


THIS_DIR = Path(__file__).resolve().parent
PARENT_DIR = THIS_DIR.parent
for p in (THIS_DIR, PARENT_DIR, THIS_DIR / "undistort", PARENT_DIR / "undistort"):
    if p.is_dir() and str(p) not in sys.path:
        sys.path.insert(0, str(p))

# D57-6: load the exact undistort helper located beside this script first.
# This prevents a different camera_undistort.py in the parent directory from
# shadowing the module used by the deployed main/D54 path.
try:
    import importlib.util as _importlib_util
    _local_undistort_py = THIS_DIR / "camera_undistort.py"
    if _local_undistort_py.is_file():
        _spec = _importlib_util.spec_from_file_location("d57_exact_camera_undistort", str(_local_undistort_py))
        if _spec is None or _spec.loader is None:
            raise ImportError(f"cannot load {_local_undistort_py}")
        _mod = _importlib_util.module_from_spec(_spec)
        sys.modules[_spec.name] = _mod
        _spec.loader.exec_module(_mod)
        CameraUndistorter = _mod.CameraUndistorter
        CAMERA_UNDISTORT_IMPORT_ERROR = None
        CAMERA_UNDISTORT_MODULE_PATH = str(_local_undistort_py)
    else:
        from camera_undistort import CameraUndistorter
        CAMERA_UNDISTORT_IMPORT_ERROR = None
        CAMERA_UNDISTORT_MODULE_PATH = str(getattr(sys.modules.get("camera_undistort"), "__file__", "camera_undistort"))
except Exception as _undistort_import_error:
    CameraUndistorter = None
    CAMERA_UNDISTORT_IMPORT_ERROR = _undistort_import_error
    CAMERA_UNDISTORT_MODULE_PATH = "UNAVAILABLE"

DEFAULT_CONFIG = "/workspace/project_train/aruco_test/dual/dual_roarm_folding_board_config.json"
DEFAULT_HFILE = "/workspace/project_train/aruco_test/dual/undistort/elp_ov2710_folding_board_homography_cache.json"
DEFAULT_CALIBRATION = "/workspace/project_train/aruco_test/dual/undistort/elp_ov2710_1280x720_calibration.npz"
DEFAULT_CAMERA_CONTROLS = "/workspace/project_train/aruco_test/dual/elp_ov2710_camera_controls.json"


# -----------------------------------------------------------------------------
# Constants / data
# -----------------------------------------------------------------------------

KPT_NAMES = [
    "waist_img_left",      # 0
    "waist_center",        # 1
    "waist_img_right",     # 2
    "crotch",              # 3
    "img_left_hem_outer",  # 4
    "img_left_hem_inner",  # 5
    "img_right_hem_inner", # 6
    "img_right_hem_outer", # 7
]
KPT = {name: i for i, name in enumerate(KPT_NAMES)}
HFLIP_REMAP = np.asarray([2, 1, 0, 3, 7, 6, 5, 4], dtype=np.int32)


@dataclass
class PoseResult:
    keypoints_px: Dict[str, np.ndarray]
    keypoints_board: Dict[str, np.ndarray]
    keypoint_conf: Dict[str, float]
    waist_left: np.ndarray
    waist_center: np.ndarray
    waist_right: np.ndarray
    crotch: Optional[np.ndarray]
    left_hem_center: Optional[np.ndarray]
    right_hem_center: Optional[np.ndarray]
    lower_center: Optional[np.ndarray]
    score: float
    tested: int
    source: str


@dataclass
class MaskResult:
    mask_u8: np.ndarray
    contour: np.ndarray
    area_px: float
    center_px: np.ndarray
    center_board: np.ndarray
    class_name: str
    confidence: float
    solidity: float


@dataclass
class AxisResult:
    p0: np.ndarray
    p1: np.ndarray
    unit: np.ndarray
    length_mm: float
    angle_deg: float
    source: str
    waist_center: np.ndarray
    lower_ref: np.ndarray


@dataclass
class GripPair:
    arm2: np.ndarray
    arm1: np.ndarray
    station: np.ndarray
    pair_distance_mm: float
    station_ratio: float
    inset_mm: float


@dataclass
class D57Plan:
    ok: bool
    mode: str
    reason: str
    axis: Optional[AxisResult] = None
    left_fold_x: Optional[float] = None
    angle_error_deg: float = 999.0
    signed_delta_deg: float = 0.0
    axis_offset_mm: float = 999.0
    reference_y_mm: float = 0.0
    grip: Optional[GripPair] = None
    pivot: Optional[np.ndarray] = None
    action_delta_deg: float = 0.0
    target_arm2: Optional[np.ndarray] = None
    target_arm1: Optional[np.ndarray] = None
    waypoints: List[Dict[str, Any]] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)
    overlay: Optional[np.ndarray] = None
    frame: Optional[np.ndarray] = None
    created_at: float = field(default_factory=time.time)


# -----------------------------------------------------------------------------
# Small helpers
# -----------------------------------------------------------------------------

def resolve_file(value: str) -> str:
    raw = os.path.expanduser(str(value or "").strip())
    if not raw:
        return raw
    if os.path.isabs(raw) and os.path.exists(raw):
        return os.path.abspath(raw)
    candidates = [
        Path(raw), THIS_DIR / raw, PARENT_DIR / raw,
        THIS_DIR / "undistort" / raw, PARENT_DIR / "undistort" / raw,
        Path("/workspace/project_train/aruco_test/dual") / raw,
        Path("/workspace/project_train/aruco_test/dual/undistort") / raw,
    ]
    for p in candidates:
        if p.is_file():
            return str(p.resolve())
    return raw


def load_json(path: str) -> Optional[Dict[str, Any]]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


def normalize_deg(v: float) -> float:
    return float((float(v) + 180.0) % 360.0 - 180.0)


def unit(v: Sequence[float], fallback=(1.0, 0.0)) -> np.ndarray:
    a = np.asarray(v, dtype=np.float32).reshape(2)
    n = float(np.linalg.norm(a))
    if n < 1e-6:
        return np.asarray(fallback, dtype=np.float32)
    return (a / n).astype(np.float32)


def rotate_point(p: Sequence[float], pivot: Sequence[float], deg: float) -> np.ndarray:
    p = np.asarray(p, np.float32).reshape(2)
    c = np.asarray(pivot, np.float32).reshape(2)
    t = math.radians(float(deg))
    R = np.asarray([[math.cos(t), -math.sin(t)], [math.sin(t), math.cos(t)]], np.float32)
    return (c + R @ (p - c)).astype(np.float32)


def perspective_points(H: np.ndarray, pts: np.ndarray) -> np.ndarray:
    a = np.asarray(pts, dtype=np.float32).reshape(-1, 1, 2)
    return cv2.perspectiveTransform(a, H).reshape(-1, 2)


def pixel_to_board(H: np.ndarray, x: float, y: float) -> np.ndarray:
    return perspective_points(H, np.asarray([[x, y]], np.float32))[0]


def board_to_pixel(H: np.ndarray, x: float, y: float) -> Optional[np.ndarray]:
    try:
        inv = np.linalg.inv(H)
        return perspective_points(inv.astype(np.float32), np.asarray([[x, y]], np.float32))[0]
    except Exception:
        return None


def as_pt(p: Sequence[float]) -> Tuple[int, int]:
    return int(round(float(p[0]))), int(round(float(p[1])))


def draw_text(img: np.ndarray, text: str, xy: Tuple[int, int], color=(0,255,0), scale=0.55, thick=2):
    cv2.putText(img, text, xy, cv2.FONT_HERSHEY_SIMPLEX, scale, (0,0,0), thick+2, cv2.LINE_AA)
    cv2.putText(img, text, xy, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)


def draw_point(img: np.ndarray, p: Sequence[float], color, label: str, radius=7):
    q = as_pt(p)
    cv2.circle(img, q, radius+2, (0,0,0), -1)
    cv2.circle(img, q, radius, color, -1)
    draw_text(img, label, (q[0]+10, q[1]-8), color, 0.48, 1)



def draw_cross(img: np.ndarray, p: Sequence[float], color, label: str, size: int = 14):
    """Draw a high-contrast X marker for the raw contour intersection."""
    x, y = as_pt(p)
    ss = max(6, int(size))
    cv2.line(img, (x-ss, y-ss), (x+ss, y+ss), (0,0,0), 5, cv2.LINE_AA)
    cv2.line(img, (x-ss, y+ss), (x+ss, y-ss), (0,0,0), 5, cv2.LINE_AA)
    cv2.line(img, (x-ss, y-ss), (x+ss, y+ss), color, 2, cv2.LINE_AA)
    cv2.line(img, (x-ss, y+ss), (x+ss, y-ss), color, 2, cv2.LINE_AA)
    draw_text(img, str(label), (x+ss+5, y-ss-2), color, 0.48, 2)

def _fourcc_text(value: float) -> str:
    try:
        v = int(round(float(value)))
        return "".join(chr((v >> (8 * i)) & 0xFF) for i in range(4))
    except Exception:
        return "????"


def open_camera(index: int, width: int, height: int, backend: str):
    """Open the ELP camera exactly like the source used by main-16.

    IMPORTANT: do NOT force MJPG/FPS here. main-16 delegates camera opening to
    D56.open_camera(), which only selects backend + width/height + buffersize.
    Forcing another UVC mode can change the sensor/scaler geometry even at the
    same nominal 1280x720 size and make the stored calibration look ineffective.
    """
    backend = str(backend).lower()
    api = cv2.CAP_ANY
    if backend == "v4l2" and hasattr(cv2, "CAP_V4L2"):
        api = cv2.CAP_V4L2
    elif backend == "dshow" and hasattr(cv2, "CAP_DSHOW"):
        api = cv2.CAP_DSHOW
    cap = cv2.VideoCapture(int(index), api)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(width))
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(height))
    try:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    except Exception:
        pass
    if not cap.isOpened():
        raise RuntimeError(f"camera open failed: index={index}")
    try:
        aw = int(round(cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
        ah = int(round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        fourcc = _fourcc_text(cap.get(cv2.CAP_PROP_FOURCC))
        print(f"[D57-8-CAMERA-OPEN] main-compatible mode size={aw}x{ah} fourcc={fourcc!r} fps={fps:.2f}; FOURCC/FPS NOT FORCED")
    except Exception:
        pass
    return cap


def _parse_v4l2_value(text: str) -> Any:
    # Keep this parser exactly compatible with the main controller.
    # v4l2-ctl may return enum controls as e.g.
    #   auto_exposure: 1 (Manual Mode)
    # Only the leading numeric token is the actual control value.
    try:
        token = str(text).split(":", 1)[1].strip().split()[0]
    except Exception:
        return None
    try:
        return int(token, 0)
    except Exception:
        try:
            return float(token)
        except Exception:
            return token


def apply_camera_controls(path: str, device: str, strict: bool = True) -> Dict[str, Any]:
    """Same fixed ELP OV2710 control-profile policy used by main."""
    profile = Path(resolve_file(path))
    if not profile.is_file():
        raise RuntimeError(f"camera controls JSON not found: {profile}")
    payload = load_json(str(profile))
    controls = payload.get("controls") if isinstance(payload, dict) else None
    if not isinstance(controls, dict):
        raise RuntimeError("camera controls JSON must contain a numeric 'controls' object")
    declared_device = str(payload.get("device") or device)
    if strict and declared_device != str(device):
        raise RuntimeError(f"camera device mismatch: profile={declared_device}, capture={device}")
    if shutil.which("v4l2-ctl") is None:
        raise RuntimeError("v4l2-ctl is required to apply the fixed camera profile")
    actual = {}
    mismatches = []
    for name, value in controls.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise RuntimeError(f"camera control {name!r} is not numeric: {value!r}")
        value_text = str(int(value)) if float(value).is_integer() else repr(float(value))
        proc = subprocess.run(
            ["v4l2-ctl", "-d", declared_device, f"--set-ctrl={name}={value_text}"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=5.0, check=False,
        )
        if proc.returncode != 0 and strict:
            raise RuntimeError(f"camera control {name} failed: {(proc.stderr or proc.stdout).strip()}")
    for name, expected in controls.items():
        proc = subprocess.run(
            ["v4l2-ctl", "-d", declared_device, f"--get-ctrl={name}"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=5.0, check=False,
        )
        value = _parse_v4l2_value(proc.stdout) if proc.returncode == 0 else None
        actual[str(name)] = value
        try:
            equal = abs(float(value) - float(expected)) <= 1e-9
        except Exception:
            equal = str(value) == str(expected)
        if not equal:
            mismatches.append({"name": name, "expected": expected, "actual": value})
    if mismatches and strict:
        raise RuntimeError(f"camera controls readback mismatch: {mismatches}")
    print(f"[D57-8-CAMERA-CONTROL] fixed profile applied: {actual}")
    return {"enabled": True, "device": declared_device, "actual_controls": actual, "mismatches": mismatches}


def camera_geometry_matches(actual: Dict[str, Any], expected: Dict[str, Any]) -> Tuple[bool, str]:
    """Reject raw/legacy/mismatched H for corrected D57 geometry."""
    if not actual:
        return False, "cache has no camera_geometry metadata"
    for key in ("undistort_enabled", "calibration_id", "output_size", "alpha"):
        if key not in actual:
            return False, f"cache metadata missing {key}"
        av, ev = actual.get(key), expected.get(key)
        if key == "alpha":
            try:
                if abs(float(av) - float(ev)) > 1e-6:
                    return False, f"alpha mismatch cache={av} runtime={ev}"
            except Exception:
                return False, f"invalid alpha metadata cache={av!r} runtime={ev!r}"
        elif av != ev:
            return False, f"{key} mismatch cache={av!r} runtime={ev!r}"
    return True, "corrected camera geometry matched"


def load_corrected_h(path: str, expected_geometry: Dict[str, Any]) -> Tuple[Optional[np.ndarray], Dict[str, Any]]:
    d = load_json(path) or {}
    if "H" not in d:
        return None, d
    ok, reason = camera_geometry_matches(dict(d.get("camera_geometry", {}) or {}), expected_geometry)
    if not ok:
        print(f"[D57-8-H-REJECT] {reason}; press L with ID0-ID3 visible")
        return None, d
    try:
        H = np.asarray(d["H"], np.float32).reshape(3, 3)
    except Exception:
        return None, d
    if not np.all(np.isfinite(H)):
        return None, d
    print(f"[D57-8-H] corrected H accepted: {reason}")
    return H, d


def save_corrected_h_preserve_bundle(path: str, H: np.ndarray, camera_geometry: Dict[str, Any]) -> None:
    """Update corrected H without deleting main's raw_H or other cache fields."""
    obj = load_json(path) or {}
    obj["schema_version"] = max(2, int(obj.get("schema_version", 2) or 2))
    obj["H"] = np.asarray(H, float).tolist()
    obj["camera_geometry"] = copy.deepcopy(camera_geometry)
    tmp = str(path) + ".d57tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)
    print(f"[D57-8-H] corrected H updated while preserving raw_H/bundle fields: {path}")


# -----------------------------------------------------------------------------
# Board config / calibration
# -----------------------------------------------------------------------------

def board_marker_map(config: Dict[str, Any]) -> Dict[str, List[float]]:
    marker = ((config.get("aruco") or {}).get("marker_board_mm") or {})
    if len(marker) < 4:
        # Current project board fallback.
        marker = {
            "0": [0.0, 0.0], "1": [0.0, -780.0],
            "2": [649.0, 0.0], "3": [649.0, -780.0],
        }
    return marker


def board_bounds(config: Dict[str, Any]) -> Tuple[float,float,float,float]:
    pts = np.asarray(list(board_marker_map(config).values()), np.float32)
    return float(pts[:,0].min()), float(pts[:,0].max()), float(pts[:,1].min()), float(pts[:,1].max())


def load_h(path: str) -> Optional[np.ndarray]:
    d = load_json(path)
    if not d or "H" not in d:
        return None
    try:
        H = np.asarray(d["H"], np.float32).reshape(3,3)
        return H if np.all(np.isfinite(H)) else None
    except Exception:
        return None


def save_h(path: str, H: np.ndarray, camera_geometry: Optional[Dict[str, Any]] = None):
    obj: Dict[str, Any] = {"schema_version": 2, "H": np.asarray(H, float).tolist()}
    if camera_geometry:
        obj["camera_geometry"] = copy.deepcopy(camera_geometry)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    print(f"[D57-H] saved {path}")


def aruco_lock(frame: np.ndarray, config: Dict[str, Any]) -> Optional[np.ndarray]:
    marker = board_marker_map(config)
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    if hasattr(cv2.aruco, "ArucoDetector"):
        detector = cv2.aruco.ArucoDetector(dictionary, cv2.aruco.DetectorParameters())
        corners, ids, _ = detector.detectMarkers(frame)
    else:
        corners, ids, _ = cv2.aruco.detectMarkers(frame, dictionary, parameters=cv2.aruco.DetectorParameters_create())
    if ids is None:
        print("[D57-H] no ArUco IDs")
        return None
    centers: Dict[int,np.ndarray] = {}
    for c, mid in zip(corners, ids.flatten()):
        centers[int(mid)] = np.asarray(c, np.float32).reshape(4,2).mean(axis=0)
    req = [0,1,2,3]
    if not all(i in centers and str(i) in marker for i in req):
        print(f"[D57-H] incomplete IDs: have={sorted(centers)}")
        return None
    src = np.asarray([centers[i] for i in req], np.float32)
    dst = np.asarray([marker[str(i)] for i in req], np.float32)
    H, _ = cv2.findHomography(src, dst)
    return None if H is None else np.asarray(H, np.float32)


def board_mask_image(H: np.ndarray, config: Dict[str, Any], shape, shrink_px: int = 10) -> np.ndarray:
    h,w = shape[:2]
    inv = np.linalg.inv(H)
    marker = board_marker_map(config)
    corners = np.asarray([marker[str(i)] for i in [0,2,3,1]], np.float32)
    px = perspective_points(inv.astype(np.float32), corners)
    m = np.zeros((h,w), np.uint8)
    cv2.fillConvexPoly(m, np.round(px).astype(np.int32), 255)
    k = max(0, int(shrink_px))
    if k:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*k+1,2*k+1))
        m = cv2.erode(m, kernel, iterations=1)
    return m


def board_to_arm_xy(config: Dict[str, Any], arm_key: str, bx: float, by: float) -> Tuple[float,float]:
    ac = ((config.get("dual_roarm") or {}).get(arm_key) or {})
    Mraw = ac.get("board_to_roarm_affine_2x3")
    if Mraw is None:
        raise RuntimeError(f"{arm_key} board_to_roarm_affine_2x3 missing")
    M = np.asarray(Mraw, float).reshape(2,3)
    out = M @ np.asarray([float(bx),float(by),1.0], float)
    off = ac.get("roarm_xy_offset", [0.0,0.0])
    try:
        out[0] += float(off[0]); out[1] += float(off[1])
    except Exception:
        pass
    return float(out[0]), float(out[1])


def arm_cfg(config: Dict[str,Any], key: str) -> Dict[str,Any]:
    return ((config.get("dual_roarm") or {}).get(key) or {})


def workspace_limits(config: Dict[str,Any], dead_half: float) -> Tuple[float,float,float]:
    xmin,xmax,_,_ = board_bounds(config)
    dual = config.get("dual_roarm") or {}
    split = float(dual.get("split_board_x", 0.5*(xmin+xmax)))
    return split, split-float(dead_half), split+float(dead_half)


def arm_for_x(config: Dict[str,Any], x: float, dead_half: float) -> Optional[str]:
    _, a2max, a1min = workspace_limits(config, dead_half)
    if float(x) <= a2max:
        return "arm2"
    if float(x) >= a1min:
        return "arm1"
    return None


# -----------------------------------------------------------------------------
# Segmentation
# -----------------------------------------------------------------------------

def infer_mask(model, frame: np.ndarray, H: np.ndarray, config: Dict[str,Any], args) -> Tuple[Optional[MaskResult], str]:
    target = {x.strip().lower() for x in str(args.seg_classes).split(",") if x.strip()}
    board_roi = board_mask_image(H, config, frame.shape, shrink_px=int(args.board_roi_shrink_px))
    best = None
    messages=[]
    for conf in [float(v) for v in str(args.seg_conf_ladder).split(",") if str(v).strip()]:
        try:
            r = model.predict(source=frame, imgsz=int(args.seg_imgsz), conf=conf, retina_masks=True, verbose=False)[0]
        except Exception as exc:
            messages.append(f"conf={conf:.3f}:{type(exc).__name__}")
            continue
        if r.boxes is None or len(r.boxes)==0 or r.masks is None:
            messages.append(f"conf={conf:.3f}:no-box")
            continue
        names = r.names or {}
        for i in range(len(r.boxes)):
            cname = str(names.get(int(r.boxes.cls[i].item()), int(r.boxes.cls[i].item()))).lower()
            if target and cname not in target:
                continue
            c = float(r.boxes.conf[i].item())
            try:
                data = r.masks.data[i].detach().cpu().numpy()
            except Exception:
                continue
            if data.shape[:2] != frame.shape[:2]:
                data = cv2.resize(data, (frame.shape[1],frame.shape[0]), interpolation=cv2.INTER_LINEAR)
            mask = ((data>0.5).astype(np.uint8)*255)
            mask = cv2.bitwise_and(mask, board_roi)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5,5),np.uint8), iterations=1)
            contours,_ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                continue
            cnt=max(contours,key=cv2.contourArea)
            area=float(cv2.contourArea(cnt))
            if area < float(args.seg_min_area_px):
                continue
            m2=np.zeros_like(mask); cv2.drawContours(m2,[cnt],-1,255,-1)
            M=cv2.moments(cnt)
            if abs(M["m00"])>1e-6:
                center=np.asarray([M["m10"]/M["m00"],M["m01"]/M["m00"]],np.float32)
            else:
                center=cnt.reshape(-1,2).astype(np.float32).mean(axis=0)
            cb=pixel_to_board(H,float(center[0]),float(center[1]))
            hull=cv2.convexHull(cnt); ha=float(cv2.contourArea(hull))
            sol=area/ha if ha>1 else 0.0
            item=MaskResult(m2,cnt,area,center,cb,cname,c,sol)
            # prefer semantic bottoms, then area/conf
            semantic_bonus = 2.0 if cname in {"bottoms","bottom","pants","trousers"} else 1.0
            rank=semantic_bonus*1e9+area+1e4*c
            if best is None or rank>best[0]:
                best=(rank,item,conf)
        if best is not None:
            break
    if best is None:
        return None, "SEG_FAILED " + ";".join(messages[-4:])
    _, item, used_conf = best
    return item, f"OK class={item.class_name} conf={item.confidence:.3f} ladder={used_conf:.3f} area={item.area_px:.0f}"


# -----------------------------------------------------------------------------
# Bottom pose TTA
# -----------------------------------------------------------------------------

def rotate_image(frame: np.ndarray, deg: float) -> Tuple[np.ndarray,np.ndarray]:
    h,w=frame.shape[:2]; cx=w/2.0; cy=h/2.0
    M=cv2.getRotationMatrix2D((cx,cy),float(deg),1.0)
    ca=abs(float(M[0,0])); sa=abs(float(M[0,1]))
    nw=int(round(h*sa+w*ca)); nh=int(round(h*ca+w*sa))
    M[0,2]+=nw/2.0-cx; M[1,2]+=nh/2.0-cy
    out=cv2.warpAffine(frame,M,(nw,nh),flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT,borderValue=(0,0,0))
    M3=np.vstack([M,[0,0,1]]).astype(np.float32)
    return out,np.linalg.inv(M3).astype(np.float32)


def hflip_image(frame: np.ndarray) -> Tuple[np.ndarray,np.ndarray]:
    h,w=frame.shape[:2]
    inv=np.asarray([[-1,0,w-1],[0,1,0],[0,0,1]],np.float32)
    return frame[:,::-1].copy(),inv


def inverse_pts(pts: np.ndarray, inv3: np.ndarray) -> np.ndarray:
    p=np.asarray(pts,np.float32).reshape(-1,2)
    h=np.c_[p,np.ones((len(p),1),np.float32)]
    return (inv3@h.T).T[:,:2].astype(np.float32)


def read_pose_instance(result) -> Tuple[Optional[np.ndarray],Optional[np.ndarray]]:
    if result.keypoints is None:
        return None,None
    try:
        xy=result.keypoints.xy.detach().cpu().numpy()
        if len(xy)==0:
            return None,None
        idx=0
        if result.boxes is not None and len(result.boxes)>0:
            conf=result.boxes.conf.detach().cpu().numpy().astype(float)
            idx=int(np.argmax(conf[:len(xy)]))
        kxy=np.asarray(xy[idx],np.float32)
        try:
            cf=result.keypoints.conf.detach().cpu().numpy()
            kcf=np.asarray(cf[idx],np.float32)
        except Exception:
            kcf=np.ones((len(kxy),),np.float32)
        return kxy,kcf
    except Exception:
        return None,None


def score_pose(kxy: np.ndarray, kcf: np.ndarray, frame_shape, mask: Optional[MaskResult], min_conf: float) -> Tuple[float,Dict[str,Any]]:
    h,w=frame_shape[:2]
    valid=[]
    for i in range(min(8,len(kxy))):
        x,y=map(float,kxy[i]); c=float(kcf[i]) if i<len(kcf) else 0.0
        valid.append(c>=min_conf and 1<x<w-1 and 1<y<h-1 and np.isfinite(x+y))
    valid += [False]*(8-len(valid))
    visible=sum(valid); waist=sum(valid[i] for i in [0,1,2]); hems=sum(valid[i] for i in [4,5,6,7])
    score=2.2*visible+5.0*waist+1.5*hems+(3.0 if valid[3] else 0.0)
    reasons=[]
    if waist==3:
        ww=float(np.linalg.norm(kxy[2]-kxy[0]))
        mid=0.5*(kxy[0]+kxy[2]); off=float(np.linalg.norm(kxy[1]-mid))
        if ww>=35: score+=4.0
        else: score-=5.0; reasons.append("waist-small")
        if off<=max(25.0,0.38*ww): score+=3.0
        else: score-=3.0; reasons.append("waist-center")
    if mask is not None:
        ds=[]
        for i,v in enumerate(valid):
            if v:
                ds.append(float(cv2.pointPolygonTest(mask.contour,(float(kxy[i,0]),float(kxy[i,1])),True)))
        if ds:
            ratio=sum(d>=-22.0 for d in ds)/len(ds)
            score += 7.0*ratio
            if ratio<0.55: score-=6.0; reasons.append("mask-disagree")
    return float(score),{"visible":visible,"waist":waist,"hems":hems,"reasons":reasons}


def infer_pose(model, frame: np.ndarray, H: np.ndarray, mask: Optional[MaskResult], args) -> Tuple[Optional[PoseResult],str]:
    angles=[float(x) for x in str(args.pose_tta_angles).split(",") if x.strip()]
    flips=[x.strip().lower() for x in str(args.pose_tta_flips).split(",") if x.strip()]
    best=None; tested=0
    for deg in angles:
        rot,inv_r=rotate_image(frame,deg)
        for flip in flips:
            work=rot; inv=inv_r; remap=False
            if flip=="h":
                work,inv_f=hflip_image(rot); inv=inv_r@inv_f; remap=True
            try:
                r=model.predict(source=work,imgsz=int(args.pose_imgsz),conf=float(args.pose_conf),verbose=False)[0]
            except Exception:
                continue
            tested+=1
            kxy,kcf=read_pose_instance(r)
            if kxy is None or len(kxy)<8:
                continue
            orig=inverse_pts(kxy,inv)
            if remap:
                orig=orig[HFLIP_REMAP].copy(); kcf=kcf[HFLIP_REMAP].copy()
            score,meta=score_pose(orig,kcf,frame.shape,mask,float(args.pose_kpt_conf))
            if best is None or score>best[0]:
                best=(score,orig,kcf,deg,flip,meta)
    if best is None:
        return None,f"POSE_FAILED tested={tested}"
    score,kxy,kcf,deg,flip,meta=best
    px:Dict[str,np.ndarray]={}; bd:Dict[str,np.ndarray]={}; cf:Dict[str,float]={}
    h,w=frame.shape[:2]
    for name,i in KPT.items():
        x,y=map(float,kxy[i]); c=float(kcf[i])
        if c<float(args.pose_kpt_conf) or not (1<x<w-1 and 1<y<h-1):
            continue
        px[name]=np.asarray([x,y],np.float32); bd[name]=pixel_to_board(H,x,y); cf[name]=c
    if "waist_center" not in bd:
        return None,f"POSE_NO_WAIST score={score:.1f}"
    # waist edges can be reconstructed from whichever edge is available only as a last fallback.
    wc=bd["waist_center"]
    wl=bd.get("waist_img_left"); wr=bd.get("waist_img_right")
    if wl is None or wr is None:
        return None,f"POSE_WAIST_EDGES_MISSING score={score:.1f}"
    crotch=bd.get("crotch")
    lpts=[bd[n] for n in ("img_left_hem_outer","img_left_hem_inner") if n in bd]
    rpts=[bd[n] for n in ("img_right_hem_inner","img_right_hem_outer") if n in bd]
    lh=np.mean(np.stack(lpts),axis=0).astype(np.float32) if lpts else None
    rh=np.mean(np.stack(rpts),axis=0).astype(np.float32) if rpts else None
    lower=(0.5*(lh+rh)).astype(np.float32) if lh is not None and rh is not None else None
    out=PoseResult(px,bd,cf,wl,wc,wr,crotch,lh,rh,lower,float(score),tested,f"TTA {deg:+.0f}/{flip}")
    return out,f"OK score={score:.1f} tested={tested} source={out.source} visible={meta['visible']} hems={meta['hems']}"


# -----------------------------------------------------------------------------
# Physical LEFT_FOLD seam
# -----------------------------------------------------------------------------

def load_left_fold_from_layout(path: str) -> Optional[float]:
    d=load_json(path)
    if not d:
        return None
    refs=d.get("reference_lines_board_mm") or {}
    line=refs.get("LEFT_FOLD") if isinstance(refs,dict) else None
    try:
        x=float((line or {}).get("x_mm"))
        return x if np.isfinite(x) else None
    except Exception:
        return None


def rectify_board(frame: np.ndarray, H: np.ndarray, config: Dict[str,Any], px_per_mm: float=1.0):
    xmin,xmax,ymin,ymax=board_bounds(config)
    s=max(0.4,float(px_per_mm)); W=max(50,int(round((xmax-xmin)*s))); HH=max(50,int(round((ymax-ymin)*s)))
    # Board coordinates -> rectified pixels: x grows right, board y=top(max) -> pixel y=0.
    T=np.asarray([[s,0,-xmin*s],[0,-s,ymax*s],[0,0,1]],np.float32)
    img_to_rect=T@H
    out=cv2.warpPerspective(frame,img_to_rect,(W,HH),flags=cv2.INTER_LINEAR)
    return out,{"xmin":xmin,"xmax":xmax,"ymin":ymin,"ymax":ymax,"scale":s}


def detect_left_fold_from_empty(empty: np.ndarray, H: np.ndarray, config: Dict[str,Any], args) -> Tuple[Optional[float],Dict[str,Any]]:
    try:
        rect,meta=rectify_board(empty,H,config,float(args.seam_px_per_mm))
        gray=cv2.cvtColor(rect,cv2.COLOR_BGR2GRAY); gray=cv2.GaussianBlur(gray,(5,5),0)
        h,w=gray.shape[:2]; s=float(meta["scale"]); xmin=float(meta["xmin"])
        lo_frac=float(args.seam_search_lo_frac); hi_frac=float(args.seam_search_hi_frac)
        x0=int(np.clip(round(w*lo_frac),5,w-10)); x1=int(np.clip(round(w*hi_frac),x0+8,w-5))
        y0=int(round(h*0.06)); y1=int(round(h*0.94))
        roi=gray[y0:y1,x0:x1]
        gx=np.abs(cv2.Sobel(roi,cv2.CV_32F,1,0,ksize=3))
        edge=np.mean(gx,axis=0)
        # Persistent vertical lines get an additional Hough support bonus.
        ed8=cv2.Canny(gray,50,140)
        lines=cv2.HoughLinesP(ed8,1,np.pi/180,threshold=max(50,int(0.22*h)),minLineLength=max(80,int(0.50*h)),maxLineGap=35)
        support=np.zeros_like(edge,dtype=np.float32)
        if lines is not None:
            for ln in lines[:,0,:]:
                xa,ya,xb,yb=map(float,ln); dx=abs(xb-xa); dy=abs(yb-ya)
                if dy<80 or dx>0.16*dy: continue
                xc=int(round(0.5*(xa+xb)))
                if x0<=xc<x1:
                    j=xc-x0; rad=max(2,int(round(3*s)))
                    support[max(0,j-rad):min(len(support),j+rad+1)] += float(dy/h)
        def norm(v):
            v=np.asarray(v,np.float32); a=float(np.percentile(v,10)); b=float(np.percentile(v,99));
            return np.clip((v-a)/max(1e-6,b-a),0,1)
        en=norm(edge); sn=norm(support) if np.max(support)>0 else support
        # Avoid selecting only one noisy high pixel: smooth along x.
        k=max(3,int(round(5*s))|1)
        score=cv2.GaussianBlur((0.78*en+0.22*sn).reshape(1,-1),(k,1),0).reshape(-1)
        j=int(np.argmax(score)); peak=float(score[j]); xp=x0+j
        x_mm=xmin+xp/s
        if peak<float(args.seam_min_score):
            return None,{"ok":False,"reason":f"weak seam score={peak:.2f}","score":peak}
        return float(x_mm),{"ok":True,"score":peak,"rect_x_px":xp,"search_px":[x0,x1],"source":"EMPTY_BOARD_VERTICAL_EDGE"}
    except Exception as exc:
        return None,{"ok":False,"reason":repr(exc)}


def resolve_left_fold(H: np.ndarray, config: Dict[str,Any], args) -> Tuple[Optional[float],Dict[str,Any]]:
    if args.left_fold_x is not None:
        return float(args.left_fold_x),{"source":"CLI_OVERRIDE","ok":True}
    layout_path=resolve_file(args.plate_layout)
    if layout_path and os.path.isfile(layout_path):
        x=load_left_fold_from_layout(layout_path)
        if x is not None:
            # If empty board is available, prefer actual visible seam over stale raw layout.
            ep=resolve_file(args.empty_board)
            if ep and os.path.isfile(ep):
                empty=cv2.imread(ep,cv2.IMREAD_COLOR)
                if empty is not None:
                    xd,rep=detect_left_fold_from_empty(empty,H,config,args)
                    if xd is not None:
                        # Do not accept an unrelated edge far away from the saved LEFT_FOLD.
                        if abs(xd-x)<=float(args.seam_layout_guard_mm):
                            rep.update({"layout_x":x,"source":"EMPTY_EDGE_NEAR_LAYOUT"})
                            return xd,rep
            return x,{"source":"C11_LAYOUT","ok":True,"layout_path":layout_path}
    ep=resolve_file(args.empty_board)
    if ep and os.path.isfile(ep):
        empty=cv2.imread(ep,cv2.IMREAD_COLOR)
        if empty is not None:
            return detect_left_fold_from_empty(empty,H,config,args)
    return None,{"ok":False,"reason":"LEFT_FOLD unavailable: use --left-fold-x or provide c11_plate_layout.json / empty-board image"}


# -----------------------------------------------------------------------------
# Axis / mask board samples / grip pair
# -----------------------------------------------------------------------------

def build_body_axis(pose: PoseResult, mask: MaskResult, args) -> Tuple[Optional[AxisResult],str]:
    wc=np.asarray(pose.waist_center,np.float32)
    if pose.lower_center is not None:
        lr=np.asarray(pose.lower_center,np.float32)
        d=lr-wc; L=float(np.linalg.norm(d))
        if L>=float(args.axis_min_hem_length_mm):
            u=unit(d); return AxisResult(wc,lr,u,L,math.degrees(math.atan2(float(u[1]),float(u[0]))),"WAIST_TO_HEM_CENTER",wc,lr),"OK"
    if pose.crotch is not None:
        cr=np.asarray(pose.crotch,np.float32); d=cr-wc; L=float(np.linalg.norm(d))
        if L>=float(args.axis_min_crotch_length_mm):
            u=unit(d); return AxisResult(wc,cr,u,L,math.degrees(math.atan2(float(u[1]),float(u[0]))),"WAIST_TO_CROTCH",wc,cr),"OK"
    waist=np.asarray(pose.waist_right-pose.waist_left,np.float32); wu=unit(waist)
    n=np.asarray([-wu[1],wu[0]],np.float32)
    # direct perpendicular toward mask center
    if float(np.dot(mask.center_board-wc,n))<0: n=-n
    L=max(float(args.axis_min_crotch_length_mm),float(np.linalg.norm(mask.center_board-wc))*1.7)
    p1=wc+n*L
    return AxisResult(wc,p1,n,L,math.degrees(math.atan2(float(n[1]),float(n[0]))),"WAIST_PERP_MASK_DIRECTED",wc,p1),"OK_FALLBACK"


def choose_vertical_delta(current_deg: float) -> Tuple[float,float]:
    dpos=normalize_deg(90.0-current_deg); dneg=normalize_deg(-90.0-current_deg)
    if abs(dpos)<=abs(dneg): return float(dpos),90.0
    return float(dneg),-90.0


def axis_x_at_y(axis: AxisResult, yref: float) -> float:
    u=axis.unit; p=axis.p0
    if abs(float(u[1]))<0.18:
        return float(0.5*(axis.p0[0]+axis.p1[0]))
    s=(float(yref)-float(p[1]))/float(u[1])
    return float(p[0]+s*float(u[0]))


def mask_board_samples(mask: MaskResult, H: np.ndarray, stride_px: int=5) -> np.ndarray:
    ys,xs=np.where(mask.mask_u8>0)
    if len(xs)==0: return np.zeros((0,2),np.float32)
    step=max(1,int(stride_px))
    idx=np.arange(0,len(xs),step,dtype=int)
    px=np.c_[xs[idx].astype(np.float32),ys[idx].astype(np.float32)]
    return perspective_points(H,px)


def nearest_sample(samples: np.ndarray, p: np.ndarray, max_dist: float=28.0) -> Optional[np.ndarray]:
    if samples is None or len(samples)==0: return None
    d=np.linalg.norm(samples-p.reshape(1,2),axis=1); i=int(np.argmin(d))
    return samples[i].astype(np.float32) if float(d[i])<=float(max_dist) else None


def board_inside(config: Dict[str,Any], p: np.ndarray, margin: float) -> bool:
    xmin,xmax,ymin,ymax=board_bounds(config); x,y=map(float,p)
    return xmin+margin<=x<=xmax-margin and ymin+margin<=y<=ymax-margin


def roarm_safe(config: Dict[str,Any], arm: str, p: np.ndarray, z: float, args) -> bool:
    try: x,y=board_to_arm_xy(config,arm,float(p[0]),float(p[1]))
    except Exception: return False
    r=math.hypot(x,y)
    return r<=float(args.roarm_xy_radius_max_mm) and float(args.min_z)<=z<=float(args.max_z)


def point_safe(config: Dict[str,Any], arm: str, p: np.ndarray, z: float, args) -> bool:
    return (arm_for_x(config,float(p[0]),float(args.center_dead_half_width))==arm
            and board_inside(config,p,float(args.board_margin_mm))
            and roarm_safe(config,arm,p,z,args))


def _mask_signed_inside_px(mask: MaskResult, H: np.ndarray, p: np.ndarray) -> float:
    pix=board_to_pixel(H,float(p[0]),float(p[1]))
    if pix is None: return -1e9
    try:
        return float(cv2.pointPolygonTest(mask.contour,(float(pix[0]),float(pix[1])),True))
    except Exception:
        return -1e9


def _nearest_safe_section_mask_point(samples: np.ndarray, desired: np.ndarray, mask: MaskResult, H: np.ndarray,
                                     config: Dict[str,Any], arm: str, contact_z_mm: float, lift_z_mm: float,
                                     axis_origin: np.ndarray, axis_u: np.ndarray, station_s: float, args,
                                     max_snap_mm: float) -> Tuple[Optional[np.ndarray],Dict[str,Any]]:
    """Snap one contour-seeded target to real, deep garment support on the SAME body cross-section.

    The outer contour supplies the physical left/right edge.  The actual gripper target
    must then be inside the segmentation mask, stay close to the requested body-axis
    station, remain in that arm's private workspace, and be reachable both at CONTACT
    and LOW-LIFT Z.  This is intentionally pair-aware geometry for D57-8 rotation.
    """
    if samples is None or len(samples)==0:
        return None,{"reason":"no mask samples"}
    desired=np.asarray(desired,np.float32).reshape(2)
    axis_origin=np.asarray(axis_origin,np.float32).reshape(2)
    axis_u=unit(np.asarray(axis_u,np.float32).reshape(2))
    d=np.linalg.norm(samples-desired.reshape(1,2),axis=1)
    order=np.argsort(d)
    min_inside=float(args.rotate_grip_min_inside_px)
    max_station=float(args.rotate_grip_max_station_snap_mm)
    best=None; best_meta=None
    for idx in order[:min(len(order),2200)]:
        dist=float(d[idx])
        if dist>float(max_snap_mm): break
        q=np.asarray(samples[int(idx)],np.float32)
        station=float(np.dot(q-axis_origin,axis_u))
        station_err=abs(station-float(station_s))
        if station_err>max_station: continue
        if arm_for_x(config,float(q[0]),float(args.center_dead_half_width))!=arm: continue
        if not board_inside(config,q,float(args.board_margin_mm)): continue
        if not roarm_safe(config,arm,q,float(contact_z_mm),args): continue
        if not roarm_safe(config,arm,q,float(lift_z_mm),args): continue
        inside=_mask_signed_inside_px(mask,H,q)
        if inside<min_inside: continue
        # Stay near the contour-derived target and same section, while preferring deep cloth.
        score=dist + 1.35*station_err - 0.70*min(inside,30.0)
        if best is None or score<float(best_meta["score"]):
            best=q.copy(); best_meta={
                "snap_mm":dist,"inside_px":inside,"station_err_mm":station_err,"score":score
            }
    if best is None:
        return None,{"reason":f"no safe same-section mask point within {max_snap_mm:.0f}mm"}
    return best,best_meta


def _contour_board_points(mask: MaskResult, H: np.ndarray) -> np.ndarray:
    """Return a DENSE physical outer contour in board mm.

    infer_mask() keeps a CHAIN_APPROX_SIMPLE contour for drawing, which can contain
    only a few vertices on long straight edges.  D57 grip slicing needs real boundary
    samples along the whole outline, so rebuild the authoritative outer boundary from
    mask_u8 with CHAIN_APPROX_NONE before converting to board coordinates.
    """
    if mask is None or mask.mask_u8 is None:
        return np.zeros((0,2),np.float32)
    try:
        contours,_=cv2.findContours((np.asarray(mask.mask_u8,np.uint8)>0).astype(np.uint8)*255,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
        if not contours:
            return np.zeros((0,2),np.float32)
        dense=max(contours,key=cv2.contourArea)
        px=np.asarray(dense,np.float32).reshape(-1,2)
        if len(px)<4:
            return np.zeros((0,2),np.float32)
        bd=perspective_points(H,px)
    except Exception:
        return np.zeros((0,2),np.float32)
    bd=np.asarray(bd,np.float32).reshape(-1,2)
    good=np.all(np.isfinite(bd),axis=1)
    return bd[good]


def find_contour_rotate_grip_candidates(mask: MaskResult, axis: AxisResult, H: np.ndarray,
                                        config: Dict[str,Any], args) -> Tuple[List[Tuple[GripPair,Dict[str,Any]]],str]:
    """D57-8 OUTER-CONTOUR grasp planner.

    1) Convert the actually detected garment OUTER CONTOUR into board coordinates.
    2) Slice that contour with several lines perpendicular to BODY AXIS.
    3) For each slice, take its physical left/right outer edges.
    4) Move both edges inward toward the same slice center and snap to deep real mask support.
    5) Keep only ARM2-left + ARM1-right pairs that are simultaneously reachable.

    This prevents the D57-5 failure where each arm could land on a different body/leg
    station and prevents D57-6 from being hard-bound to an unreachable waist endpoint.
    """
    contour=_contour_board_points(mask,H)
    samples=mask_board_samples(mask,H,int(args.mask_sample_stride_px))
    if len(contour)<12 or len(samples)<100:
        return [],f"insufficient contour/mask samples contour={len(contour)} mask={len(samples)}"

    origin=np.asarray(axis.waist_center,np.float32).reshape(2)
    u=unit(np.asarray(axis.unit,np.float32).reshape(2))
    n=np.asarray([-u[1],u[0]],np.float32)
    cs=(contour-origin.reshape(1,2))@u.reshape(2,1); cs=cs.reshape(-1)
    cn=(contour-origin.reshape(1,2))@n.reshape(2,1); cn=cn.reshape(-1)

    ratios=[]
    for token in str(args.rotate_grip_station_ratios).split(','):
        try: ratios.append(float(token.strip()))
        except Exception: pass
    if not ratios: ratios=[0.28,0.36,0.44,0.52,0.22,0.60]

    band=max(5.0,float(args.rotate_grip_section_band_mm))
    inset=max(5.0,float(args.rotate_grip_inset_mm))
    max_snap=max(10.0,float(args.rotate_grip_snap_mm))
    min_width=float(args.rotate_grip_min_raw_width_mm)
    max_width=float(args.rotate_grip_max_raw_width_mm)
    zc2=contact_z(config,"arm2",args); zc1=contact_z(config,"arm1",args)
    zl2=zc2+float(args.low_lift_mm); zl1=zc1+float(args.low_lift_mm)
    results=[]

    print(f"[D57-8-CONTOUR] contourPts={len(contour)} maskPts={len(samples)} axis={axis.source} angle={axis.angle_deg:+.1f}deg")
    for ratio in ratios:
        if not (0.05<=ratio<=0.82):
            continue
        s0=float(axis.length_mm)*float(ratio)
        idx=np.where(np.abs(cs-s0)<=band)[0]
        if len(idx)<4:
            print(f"[D57-8-SECTION] ratio={ratio:.2f} s={s0:.1f} -> MISS contourBandPts={len(idx)}")
            continue
        band_pts=contour[idx]; band_n=cn[idx]
        raw_a=np.asarray(band_pts[int(np.argmin(band_n))],np.float32)
        raw_b=np.asarray(band_pts[int(np.argmax(band_n))],np.float32)
        # Physical arm assignment is board-x based, never pose-name based.
        raw2,raw1=(raw_a,raw_b) if float(raw_a[0])<=float(raw_b[0]) else (raw_b,raw_a)
        raw_width=float(np.linalg.norm(raw1-raw2))
        if raw_width<min_width or raw_width>max_width:
            print(f"[D57-8-SECTION] ratio={ratio:.2f} rawWidth={raw_width:.1f} -> REJECT width")
            continue
        center=0.5*(raw2+raw1)
        v2=center-raw2; v1=center-raw1
        if np.linalg.norm(v2)<1e-6 or np.linalg.norm(v1)<1e-6:
            continue
        desired2=raw2+unit(v2)*inset
        desired1=raw1+unit(v1)*inset
        a2,m2=_nearest_safe_section_mask_point(samples,desired2,mask,H,config,"arm2",zc2,zl2,origin,u,s0,args,max_snap)
        a1,m1=_nearest_safe_section_mask_point(samples,desired1,mask,H,config,"arm1",zc1,zl1,origin,u,s0,args,max_snap)
        if a2 is None or a1 is None:
            print(f"[D57-8-SECTION] ratio={ratio:.2f} raw2={np.round(raw2,1).tolist()} raw1={np.round(raw1,1).tolist()} -> BLOCK A2={a2 is not None} A1={a1 is not None} m2={m2} m1={m1}")
            continue
        pair=float(np.linalg.norm(a1-a2))
        station2=float(np.dot(a2-origin,u)); station1=float(np.dot(a1-origin,u))
        section_mismatch=abs(station1-station2)
        if not (float(args.grip_min_pair_separation_mm)<=pair<=float(args.grip_max_pair_separation_mm)):
            print(f"[D57-8-SECTION] ratio={ratio:.2f} pair={pair:.1f} -> REJECT separation")
            continue
        if section_mismatch>float(args.rotate_grip_pair_station_mismatch_mm):
            print(f"[D57-8-SECTION] ratio={ratio:.2f} stationMismatch={section_mismatch:.1f} -> REJECT same-section")
            continue
        # Prefer upper/mid body slices, deep mask points, small snap, and a compact pair.
        preferred=0.36
        score=(
            90.0*abs(float(ratio)-preferred)
            + 0.10*abs(pair-float(args.rotate_grip_preferred_pair_mm))
            + float(m2['snap_mm'])+float(m1['snap_mm'])
            + 1.5*section_mismatch
            - 0.45*(min(float(m2['inside_px']),25.0)+min(float(m1['inside_px']),25.0))
        )
        gp=GripPair(a2.astype(np.float32),a1.astype(np.float32),center.astype(np.float32),pair,float(ratio),float(inset))
        meta={
            "score":float(score),"ratio":float(ratio),"station_s_mm":s0,
            "raw_arm2":raw2.tolist(),"raw_arm1":raw1.tolist(),"raw_width_mm":raw_width,
            "section_mismatch_mm":section_mismatch,"arm2":m2,"arm1":m1,
        }
        print(f"[D57-8-SECTION-OK] ratio={ratio:.2f} rawWidth={raw_width:.1f} A2={np.round(a2,1).tolist()} A1={np.round(a1,1).tolist()} pair={pair:.1f} stationMismatch={section_mismatch:.1f} score={score:.1f}")
        results.append((gp,meta))

    results.sort(key=lambda item:float(item[1].get('score',1e9)))
    if not results:
        return [],"no same-cross-section contour pair is simultaneously safe for ARM2+ARM1"
    return results,"OK"


def rotate_contour_footprint_safe(mask: MaskResult, H: np.ndarray, pivot: np.ndarray,
                                  delta_deg: float, config: Dict[str,Any], args) -> Tuple[bool,Dict[str,Any]]:
    """Approximate cloth-footprint guard using the detected OUTER CONTOUR in board mm."""
    contour=_contour_board_points(mask,H)
    if len(contour)<8:
        return False,{"reason":"no contour board points"}
    t=math.radians(float(delta_deg)); c=math.cos(t); ss=math.sin(t)
    R=np.asarray([[c,-ss],[ss,c]],np.float32)
    pv=np.asarray(pivot,np.float32).reshape(2)
    rot=(pv.reshape(1,2)+(contour-pv.reshape(1,2))@R.T).astype(np.float32)
    xmin,xmax,ymin,ymax=board_bounds(config); m=max(0.0,float(args.rotate_contour_board_margin_mm))
    inside=((rot[:,0]>=xmin+m)&(rot[:,0]<=xmax-m)&(rot[:,1]>=ymin+m)&(rot[:,1]<=ymax-m))
    ratio=float(np.mean(inside)) if len(inside) else 0.0
    ok=ratio>=float(args.rotate_contour_min_inside_ratio)
    return ok,{"inside_ratio":ratio,"points":len(rot),"margin_mm":m,
              "rot_x":[float(np.min(rot[:,0])),float(np.max(rot[:,0]))],
              "rot_y":[float(np.min(rot[:,1])),float(np.max(rot[:,1]))]}


# -----------------------------------------------------------------------------
# Planner
# -----------------------------------------------------------------------------

def build_waypoints_rotate(grip: GripPair, pivot: np.ndarray, delta: float, config: Dict[str,Any], args) -> Tuple[Optional[List[Dict[str,Any]]],str,np.ndarray,np.ndarray,float]:
    max_step=max(3.0,float(args.rotation_step_deg)); n=max(1,int(math.ceil(abs(delta)/max_step)))
    pair0=float(np.linalg.norm(grip.arm1-grip.arm2)); out=[]
    prev2=grip.arm2.copy(); prev1=grip.arm1.copy()
    z2=contact_z(config,"arm2",args)+float(args.low_lift_mm); z1=contact_z(config,"arm1",args)+float(args.low_lift_mm)
    max_pair=pair0
    for i in range(1,n+1):
        d=float(delta)*i/n; p2=rotate_point(grip.arm2,pivot,d); p1=rotate_point(grip.arm1,pivot,d)
        pair=float(np.linalg.norm(p1-p2)); max_pair=max(max_pair,pair)
        if pair>pair0+float(args.distance_max_stretch_mm):
            return None,f"distance stretch {pair:.1f}>{pair0+args.distance_max_stretch_mm:.1f}",p2,p1,max_pair
        if not point_safe(config,"arm2",p2,z2,args) or not point_safe(config,"arm1",p1,z1,args):
            return None,f"unsafe waypoint {i}/{n}",p2,p1,max_pair
        l2=float(np.linalg.norm(p2-prev2)); l1=float(np.linalg.norm(p1-prev1)); mx=max(l2,l1,1e-6)
        sbase=float(args.rotate_speed)
        s2=max(float(args.sync_min_scale),l2/mx)*sbase
        s1=max(float(args.sync_min_scale),l1/mx)*sbase
        out.append({"i":i,"n":n,"delta_deg":d,"arm2":p2,"arm1":p1,"move2":l2,"move1":l1,"speed2":s2,"speed1":s1,"pair":pair})
        prev2,prev1=p2,p1
    return out,"OK",out[-1]["arm2"],out[-1]["arm1"],max_pair


def make_overlay(frame: np.ndarray, mask: MaskResult, pose: PoseResult, plan: D57Plan, H: np.ndarray, config: Dict[str,Any]) -> np.ndarray:
    out=frame.copy(); ov=out.copy(); cv2.drawContours(ov,[mask.contour],-1,(0,180,255),-1); out=cv2.addWeighted(ov,0.18,out,0.82,0)
    cv2.drawContours(out,[mask.contour],-1,(0,220,255),2)
    # pose keypoints
    for name,p in pose.keypoints_px.items():
        col=(255,180,0) if "waist" in name else ((0,255,255) if "hem" in name else (0,0,255))
        cv2.circle(out,as_pt(p),4,col,-1)
    if plan.left_fold_x is not None:
        _,_,ymin,ymax=board_bounds(config)
        p0=board_to_pixel(H,float(plan.left_fold_x),ymin); p1=board_to_pixel(H,float(plan.left_fold_x),ymax)
        if p0 is not None and p1 is not None:
            cv2.line(out,as_pt(p0),as_pt(p1),(255,255,255),5)
            draw_text(out,f"LEFT_FOLD x={plan.left_fold_x:.1f}mm",(as_pt(p1)[0]+8,52),(255,255,255),0.55,2)
    if plan.axis is not None:
        # extend body axis to board y limits for visual authority.
        ax=plan.axis; _,_,ymin,ymax=board_bounds(config)
        pts=[]
        for yy in (ymin,ymax):
            xx=axis_x_at_y(ax,yy); p=board_to_pixel(H,xx,yy)
            if p is not None: pts.append(p)
        if len(pts)==2:
            cv2.line(out,as_pt(pts[0]),as_pt(pts[1]),(0,255,0),4)
        for p,label in ((ax.p0,"WAIST-C"),(ax.p1,"LOWER-REF")):
            q=board_to_pixel(H,float(p[0]),float(p[1]));
            if q is not None: draw_point(out,q,(0,255,0),label)
    if plan.grip is not None:
        raw2=plan.metrics.get("grip_raw_arm2"); raw1=plan.metrics.get("grip_raw_arm1")
        for label,raw,final,col in (("OUTER-A2",raw2,plan.grip.arm2,(255,180,80)),("OUTER-A1",raw1,plan.grip.arm1,(255,120,255))):
            if raw is not None:
                rp=np.asarray(raw,np.float32); a=board_to_pixel(H,float(rp[0]),float(rp[1])); b=board_to_pixel(H,float(final[0]),float(final[1]))
                if a is not None:
                    draw_cross(out,a,col,label)
                if a is not None and b is not None:
                    cv2.line(out,as_pt(a),as_pt(b),col,2,cv2.LINE_AA)
        for key,p,col in (("A2",plan.grip.arm2,(255,80,80)),("A1",plan.grip.arm1,(255,0,255))):
            q=board_to_pixel(H,float(p[0]),float(p[1]));
            if q is not None: draw_point(out,q,col,f"D57 {key} GRIP",8)
        if plan.pivot is not None:
            q=board_to_pixel(H,float(plan.pivot[0]),float(plan.pivot[1]));
            if q is not None: draw_point(out,q,(0,165,255),"PIVOT",6)
    if plan.target_arm2 is not None and plan.target_arm1 is not None and plan.grip is not None:
        for src,dst,col in ((plan.grip.arm2,plan.target_arm2,(255,80,80)),(plan.grip.arm1,plan.target_arm1,(255,0,255))):
            a=board_to_pixel(H,float(src[0]),float(src[1])); b=board_to_pixel(H,float(dst[0]),float(dst[1]))
            if a is not None and b is not None:
                cv2.arrowedLine(out,as_pt(a),as_pt(b),col,4,tipLength=0.18); cv2.circle(out,as_pt(b),7,col,-1)
    color=(0,255,0) if plan.ok else ((0,220,255) if plan.mode=="ALIGNED" else (0,0,255))
    draw_text(out,f"D57-8 {plan.mode} | {'PLAN OK' if plan.ok else plan.reason[:70]}",(24,32),color,0.68,2)
    draw_text(out,f"axis={plan.axis.source if plan.axis else '-'} angleErr={plan.angle_error_deg:.1f}deg offsetDiag={plan.axis_offset_mm:+.1f}mm",(24,59),color,0.56,2)
    if plan.mode=="ROTATE":
        draw_text(out,f"rotate={plan.action_delta_deg:+.1f}deg steps={len(plan.waypoints)} LOW-LIFT",(24,85),color,0.55,2)
    return out


def build_plan(frame: np.ndarray, mask: MaskResult, pose: PoseResult, H: np.ndarray, config: Dict[str,Any], left_fold_x: float, args) -> D57Plan:
    """Build exactly one ROTATE-only plan. LEFT_FOLD X is display context; its direction is vertical."""
    axis,why=build_body_axis(pose,mask,args)
    if axis is None:
        return D57Plan(False,"BLOCKED",f"body axis failed: {why}",frame=frame)

    waist_w=float(np.linalg.norm(pose.waist_right-pose.waist_left))
    hem_gap=float(np.linalg.norm(pose.right_hem_center-pose.left_hem_center)) if pose.left_hem_center is not None and pose.right_hem_center is not None else 0.0
    hem_ratio=hem_gap/max(waist_w,1.0)
    # D57 is ROTATE-only.  A small hem/waist ratio can simply mean the two trouser
    # legs are lying close together; it must not veto an otherwise very flat, long,
    # well-observed garment.  Readiness is therefore based on the actual spread mask
    # solidity + a trustworthy body-axis length.  hem_ratio remains diagnostic only.
    if mask.solidity<float(args.min_solidity) or axis.length_mm<float(args.min_axis_length_mm):
        reason=f"ROTATE_NOT_READY solidity={mask.solidity:.2f} axisLen={axis.length_mm:.0f} hem/waist={hem_ratio:.2f}(diag-only)"
        p=D57Plan(False,"NOT_READY",reason,axis=axis,left_fold_x=float(left_fold_x),frame=frame,
                  metrics={"solidity":mask.solidity,"hem_ratio":hem_ratio,"waist_width":waist_w,
                           "hem_ratio_gate":False})
        p.overlay=make_overlay(frame,mask,pose,p,H,config); return p
    if hem_ratio<float(args.min_hem_gap_ratio):
        print(f"[D57-8-READY-WARN] hem/waist={hem_ratio:.2f} < {float(args.min_hem_gap_ratio):.2f}; diagnostic only, ROTATE planning continues")

    signed_delta,target_axis=choose_vertical_delta(axis.angle_deg)
    angle_err=abs(signed_delta)
    # Offset is diagnostic only in D57-8; D57 no longer translates the trousers.
    yref=float(0.5*(axis.waist_center[1]+axis.lower_ref[1])); xref=axis_x_at_y(axis,yref); offset=xref-float(left_fold_x)
    base=D57Plan(False,"BLOCKED","",axis=axis,left_fold_x=float(left_fold_x),angle_error_deg=float(angle_err),
                 signed_delta_deg=float(signed_delta),axis_offset_mm=float(offset),reference_y_mm=yref,frame=frame,
                 metrics={"target_axis_deg":target_axis,"axis_x_ref":xref,"solidity":mask.solidity,
                          "hem_ratio":hem_ratio,"waist_width":waist_w,"offset_diagnostic_only":True})

    if angle_err<=float(args.angle_tolerance_deg):
        base.mode="ALIGNED"; base.reason="BODY AXIS already parallel to LEFT_FOLD; D57 ROTATE NO_ACTION"
        base.overlay=make_overlay(frame,mask,pose,base,H,config); return base

    candidates,gwhy=find_contour_rotate_grip_candidates(mask,axis,H,config,args)
    if not candidates:
        base.reason=f"no safe contour dual grip: {gwhy}"; base.overlay=make_overlay(frame,mask,pose,base,H,config); return base

    requested=float(np.clip(signed_delta,-float(args.max_rotation_deg),float(args.max_rotation_deg)))
    chosen=None
    # Pair selection and rotation-path selection are solved together.  A visually good
    # grasp is rejected if either arm cannot follow the complete rigid rotation arc.
    for cand_index,(grip,gmeta) in enumerate(candidates):
        pivot=(0.5*(grip.arm2+grip.arm1)).astype(np.float32)
        for scale in [1.0,0.90,0.80,0.70,0.60,0.50,0.40,0.30]:
            d=requested*scale
            if abs(d)<float(args.min_rotation_action_deg):
                continue
            w,rs,p2,p1,maxpair=build_waypoints_rotate(grip,pivot,d,config,args)
            if w is None:
                continue
            footprint_ok=True; footprint_rep=None
            for wp in w:
                footprint_ok,footprint_rep=rotate_contour_footprint_safe(mask,H,pivot,float(wp["delta_deg"]),config,args)
                if not footprint_ok:
                    break
            if not footprint_ok:
                print(f"[D57-8-FOOTPRINT-BLOCK] candidate={cand_index} delta={d:+.1f} rep={footprint_rep}")
                continue
            # Final place target must also be contact-safe for both arms.
            if not point_safe(config,"arm2",p2,contact_z(config,"arm2",args)+float(args.place_clearance_mm),args):
                continue
            if not point_safe(config,"arm1",p1,contact_z(config,"arm1",args)+float(args.place_clearance_mm),args):
                continue
            chosen=(grip,gmeta,pivot,d,w,p2,p1,maxpair,scale,cand_index)
            break
        if chosen is not None:
            break

    if chosen is None:
        base.reason=f"all contour grasp pairs fail full rotate-path preflight requested={requested:+.1f}deg"
        base.overlay=make_overlay(frame,mask,pose,base,H,config); return base

    grip,gmeta,pivot,d,w,p2,p1,maxpair,scale,cand_index=chosen
    base.mode="ROTATE"; base.grip=grip; base.pivot=pivot; base.action_delta_deg=float(d)
    base.waypoints=w; base.target_arm2=p2; base.target_arm1=p1
    base.metrics.update({
        "rotation_scale":float(scale),"pair_start_mm":grip.pair_distance_mm,"pair_max_mm":maxpair,
        "grip_candidate_index":int(cand_index),"grip_station_ratio":float(grip.station_ratio),
        "grip_section_mismatch_mm":float(gmeta.get('section_mismatch_mm',0.0)),
        "grip_raw_arm2":gmeta.get('raw_arm2'),"grip_raw_arm1":gmeta.get('raw_arm1'),
        "rotated_contour_inside_ratio":float((footprint_rep or {}).get("inside_ratio",1.0)),
    })
    base.ok=True; base.reason="OK_ROTATE_ONLY"
    print(f"[D57-8-ROTATE-PLAN] requested={requested:+.1f}deg chosen={d:+.1f}deg scale={scale:.2f} candidate={cand_index} stationRatio={grip.station_ratio:.2f} pivot={np.round(pivot,1).tolist()}")
    base.overlay=make_overlay(frame,mask,pose,base,H,config); return base


# -----------------------------------------------------------------------------
# Robot execution
# -----------------------------------------------------------------------------

class RoArm:
    def __init__(self, port: str, label: str, baud=115200):
        if serial is None: raise RuntimeError("pyserial unavailable")
        self.label=label; self.lock=threading.RLock(); self.ser=serial.Serial(port,int(baud),timeout=0.18)
        try: self.ser.setDTR(False); self.ser.setRTS(False)
        except Exception: pass
        time.sleep(0.25)
    def send(self,obj:Dict[str,Any],delay=0.0,stage=""):
        line=json.dumps(obj,separators=(",",":"))
        with self.lock:
            print(f"[D57-SERIAL] {self.label} stage={stage} {line}")
            self.ser.write((line+"\n").encode("utf-8")); self.ser.flush()
            if delay>0: time.sleep(delay)
    def move(self,x,y,z,t=3.14,spd=0.8,stage=""):
        self.send({"T":104,"x":float(x),"y":float(y),"z":float(z),"t":float(t),"spd":float(spd)},stage=stage)
    def grip(self,cmd,stage=""):
        self.send({"T":106,"cmd":float(cmd),"spd":0,"acc":0},stage=stage)
    def feedback(self,timeout=1.0) -> Optional[Dict[str,Any]]:
        with self.lock:
            try: self.ser.reset_input_buffer()
            except Exception: pass
            self.ser.write(b'{"T":105}\n'); self.ser.flush(); start=time.time()
            while time.time()-start<timeout:
                raw=self.ser.readline()
                if not raw: continue
                try: d=json.loads(raw.decode("utf-8",errors="ignore").strip())
                except Exception: continue
                if int(d.get("T",-1))==1051: return d
        return None
    def close(self):
        try: self.ser.close()
        except Exception: pass


def contact_z(config: Dict[str,Any], key: str, args) -> float:
    override=args.arm1_grip_z if key=="arm1" else args.arm2_grip_z
    if override is not None: return float(override)
    ac=arm_cfg(config,key)
    if ac.get("grip_z") is not None: return float(ac["grip_z"])
    return -125.0 if key=="arm1" else -115.0


def hover_z(config: Dict[str,Any], key: str, args) -> float:
    if args.hover_z is not None: return float(args.hover_z)
    ac=arm_cfg(config,key)
    return float(ac.get("safe_hover_z",180.0))


def tool_t(config: Dict[str,Any], key: str) -> float:
    ac=arm_cfg(config,key); return float(ac.get("tool_t_rad",3.14))


def pair_roarm(config: Dict[str,Any], p2: np.ndarray, p1: np.ndarray, z2: float, z1: float) -> Dict[str,Tuple[float,float,float]]:
    x2,y2=board_to_arm_xy(config,"arm2",float(p2[0]),float(p2[1])); x1,y1=board_to_arm_xy(config,"arm1",float(p1[0]),float(p1[1]))
    return {"arm2":(x2,y2,float(z2)),"arm1":(x1,y1,float(z1))}


def parallel_call(func2,func1):
    errs=[]
    def wrap(fn):
        try: fn()
        except Exception as exc: errs.append(exc)
    t2=threading.Thread(target=wrap,args=(func2,),daemon=True); t1=threading.Thread(target=wrap,args=(func1,),daemon=True)
    t2.start(); t1.start(); t2.join(); t1.join()
    if errs: raise errs[0]


def move_pair(arms: Dict[str,RoArm], config: Dict[str,Any], p2: np.ndarray, p1: np.ndarray, z2: float, z1: float, speed2: float, speed1: float, stage: str, wait: float):
    xyz=pair_roarm(config,p2,p1,z2,z1)
    parallel_call(
        lambda: arms["arm2"].move(*xyz["arm2"],t=tool_t(config,"arm2"),spd=speed2,stage=stage),
        lambda: arms["arm1"].move(*xyz["arm1"],t=tool_t(config,"arm1"),spd=speed1,stage=stage),
    )
    time.sleep(max(0.0,float(wait)))


def grip_both(arms: Dict[str,RoArm], cmd: float, stage: str, wait: float=0.0):
    parallel_call(lambda: arms["arm2"].grip(cmd,stage),lambda: arms["arm1"].grip(cmd,stage))
    if wait>0: time.sleep(wait)


def obvious_open_feedback(arm: RoArm, threshold: float) -> Tuple[bool,Optional[float]]:
    d=arm.feedback(timeout=0.8)
    if not d: return False,None
    try:
        v=float(d.get("t")); return bool(v<float(threshold)),v
    except Exception: return False,None


def _query_gripper_t_retry(arm: RoArm, attempts: int=3, pause_s: float=0.10) -> Optional[float]:
    for _ in range(max(1,int(attempts))):
        d=arm.feedback(timeout=0.9)
        if d is not None:
            try: return float(d.get("t"))
            except Exception: pass
        time.sleep(max(0.0,float(pause_s)))
    return None


def close_fully_at_contact(arms: Dict[str,RoArm], args) -> Tuple[bool,str]:
    """C120/B31 style: LIMITED CLOSE 3.05 -> HOLD 3.14 while XYZ stays fixed."""
    print("[D57-8-GRIP] XYZ FIXED -> LIMITED_CLOSE -> HOLD before any lift")
    grip_both(arms,float(args.grip_close),"D57_LIMITED_CLOSE_AT_GRIP")
    time.sleep(float(args.close_limited_wait_s))
    for i in range(max(1,int(args.close_repeat))):
        grip_both(arms,float(args.grip_hold),f"D57_HOLD_AT_GRIP_{i+1}")
        if i+1<int(args.close_repeat): time.sleep(float(args.close_repeat_gap_s))
    time.sleep(float(args.close_final_hold_s))

    v2=_query_gripper_t_retry(arms["arm2"],int(args.feedback_query_retries),float(args.feedback_query_gap_s))
    v1=_query_gripper_t_retry(arms["arm1"],int(args.feedback_query_retries),float(args.feedback_query_gap_s))
    threshold=float(args.obvious_open_feedback_rad)
    print(f"[D57-8-GRIP-FEEDBACK] A2={v2} A1={v1} obviousOpenThreshold={threshold:.2f}")
    obvious=[]
    if v2 is not None and v2<threshold: obvious.append("ARM2")
    if v1 is not None and v1<threshold: obvious.append("ARM1")
    if not obvious:
        if v2 is None or v1 is None:
            print("[D57-8-GRIP-FEEDBACK-WARN] feedback missing on one arm; repeated HOLD command is authoritative")
        return True,"OK"

    # Clearly-open feedback is a real fault. Keep contact XYZ fixed and retry HOLD only.
    for j in range(int(args.close_extra_retries)):
        print(f"[D57-8-GRIP-RETRY] {j+1}/{args.close_extra_retries} obviousOpen={obvious}; NO ARM MOTION")
        grip_both(arms,float(args.grip_hold),f"D57_HOLD_RETRY_{j+1}")
        time.sleep(float(args.close_retry_wait_s))
        v2=_query_gripper_t_retry(arms["arm2"],2,float(args.feedback_query_gap_s))
        v1=_query_gripper_t_retry(arms["arm1"],2,float(args.feedback_query_gap_s))
        obvious=[]
        if v2 is not None and v2<threshold: obvious.append("ARM2")
        if v1 is not None and v1<threshold: obvious.append("ARM1")
        print(f"[D57-8-GRIP-FEEDBACK] retry={j+1} A2={v2} A1={v1} obvious={obvious}")
        if not obvious: return True,"OK_AFTER_RETRY"
    return False,f"GRIPPER_STILL_OBVIOUSLY_OPEN A2={v2} A1={v1}"


def standby(arms: Dict[str,RoArm], args):
    # Project-standard imaging standby; direct RoArm local coordinates.
    try:
        parallel_call(
            lambda: arms["arm2"].move(float(args.arm2_standby_x),float(args.arm2_standby_y),float(args.arm2_standby_z),t=float(args.arm2_standby_t),spd=float(args.standby_speed),stage="D57_STANDBY"),
            lambda: arms["arm1"].move(float(args.arm1_standby_x),float(args.arm1_standby_y),float(args.arm1_standby_z),t=float(args.arm1_standby_t),spd=float(args.standby_speed),stage="D57_STANDBY"),
        ); time.sleep(float(args.standby_wait_s))
    except Exception as exc: print(f"[D57-STANDBY-WARN] {exc!r}")


def failed_contact_recovery(arms: Dict[str,RoArm], config: Dict[str,Any], grip: GripPair, args):
    print("[D57-RECOVERY] close failed -> OPEN -> vertical retract -> standby")
    grip_both(arms,float(args.grip_open),"D57_FAIL_OPEN",wait=0.35)
    z2=hover_z(config,"arm2",args); z1=hover_z(config,"arm1",args)
    move_pair(arms,config,grip.arm2,grip.arm1,z2,z1,float(args.free_speed),float(args.free_speed),"D57_FAIL_VERTICAL_CLEAR",1.5)
    standby(arms,args)


def execute_plan(plan: D57Plan, config: Dict[str,Any], args) -> bool:
    if not plan.ok or plan.grip is None or plan.mode!="ROTATE":
        print(f"[D57-EXEC] blocked mode={plan.mode} reason={plan.reason}"); return False
    if not bool(args.send):
        print("[D57-DRY] --send not set; no robot command"); return True
    arms={"arm1":RoArm(args.arm1_port,"ARM1"),"arm2":RoArm(args.arm2_port,"ARM2")}
    try:
        g=plan.grip; p2=g.arm2.copy(); p1=g.arm1.copy()
        cz2=contact_z(config,"arm2",args); cz1=contact_z(config,"arm1",args)
        hz2=hover_z(config,"arm2",args); hz1=hover_z(config,"arm1",args)
        lz2=cz2+float(args.low_lift_mm); lz1=cz1+float(args.low_lift_mm)
        place2=cz2+float(args.place_clearance_mm); place1=cz1+float(args.place_clearance_mm)
        print(f"[D57-8-SEQUENCE] OUTER-CONTOUR SAME-SECTION GRIP -> OPEN -> HOVER -> VERTICAL CONTACT -> CLOSE/HOLD -> LOW_LIFT({args.low_lift_mm:.0f}mm) -> DUAL RIGID ROTATE({plan.action_delta_deg:+.1f}deg) -> PLACE -> OPEN -> VERTICAL RETRACT")
        print(f"[D57-8-GRIP-LOCKED] A2={np.round(p2,1).tolist()} A1={np.round(p1,1).tolist()} pair={g.pair_distance_mm:.1f}mm stationRatio={g.station_ratio:.2f}")

        grip_both(arms,float(args.grip_open),"D57_OPEN_BEFORE_HOVER",float(args.pre_open_wait_s))
        move_pair(arms,config,p2,p1,hz2,hz1,float(args.free_speed),float(args.free_speed),"D57_HOVER_OPEN",float(args.hover_wait_s))
        grip_both(arms,float(args.grip_open),"D57_OPEN_BEFORE_DESCENT",float(args.open_before_descent_wait_s))
        move_pair(arms,config,p2,p1,cz2,cz1,float(args.near_speed),float(args.near_speed),"D57_CONTACT_OPEN_VERTICAL",float(args.contact_move_wait_s))
        time.sleep(float(args.contact_hold_s))
        ok,why=close_fully_at_contact(arms,args)
        if not ok:
            print(f"[D57-8-GRIP-BLOCK] {why}"); failed_contact_recovery(arms,config,g,args); return False

        # No pull, tension, or pre-slack is applied; the held garment is only lifted and rotated.
        move_pair(arms,config,p2,p1,lz2,lz1,float(args.lift_speed),float(args.lift_speed),"D57_LOW_LIFT_CLOSED",float(args.low_lift_wait_s))
        pivot=0.5*(p2+p1)
        sg=GripPair(p2,p1,pivot,float(np.linalg.norm(p1-p2)),g.station_ratio,g.inset_mm)
        way,why,_,_,_=build_waypoints_rotate(sg,pivot,float(plan.action_delta_deg),config,args)
        if way is None:
            raise RuntimeError("post-lift rotation preflight failed: "+why)
        for wp in way:
            print(f"[D57-8-ROTATE] {wp['i']}/{wp['n']} delta={wp['delta_deg']:+.1f}deg A2={np.round(wp['arm2'],1).tolist()} A1={np.round(wp['arm1'],1).tolist()} pair={wp['pair']:.1f}mm")
            move_pair(arms,config,wp["arm2"],wp["arm1"],lz2,lz1,wp["speed2"],wp["speed1"],f"D57_ROTATE_{wp['i']:02d}",float(args.rotation_step_wait_s))
        end2,end1=way[-1]["arm2"],way[-1]["arm1"]

        move_pair(arms,config,end2,end1,place2,place1,float(args.near_speed),float(args.near_speed),"D57_PLACE_CLOSED",float(args.place_move_wait_s))
        time.sleep(float(args.place_settle_s))
        grip_both(arms,float(args.release_open),"D57_RELEASE",float(args.release_wait_s))
        move_pair(arms,config,end2,end1,hz2,hz1,float(args.vertical_retract_speed),float(args.vertical_retract_speed),"D57_VERTICAL_RETRACT_OPEN",float(args.vertical_retract_wait_s))
        standby(arms,args)
        print("[D57-8-EXEC-DONE] ROTATE-only physical round complete; press I for FRESH REJUDGE")
        return True
    except Exception as exc:
        print(f"[D57-8-EXEC-ERROR] {exc!r}")
        try:
            grip_both(arms,float(args.grip_open),"D57_ERROR_OPEN",0.3); standby(arms,args)
        except Exception:
            pass
        return False
    finally:
        for a in arms.values(): a.close()


# -----------------------------------------------------------------------------
# CLI / main
# -----------------------------------------------------------------------------

def parser_build() -> argparse.ArgumentParser:
    p=argparse.ArgumentParser(description="D57-8 outer-contour same-section dual-arm ROTATE-only pants alignment to LEFT_FOLD direction")
    p.add_argument("--config",default=DEFAULT_CONFIG)
    p.add_argument("--hfile",default=DEFAULT_HFILE)
    p.add_argument("--camera",type=int,default=0); p.add_argument("--width",type=int,default=1280); p.add_argument("--height",type=int,default=720); p.add_argument("--backend",default="v4l2")
    p.add_argument("--camera-undistort",dest="camera_undistort",action="store_true",default=True); p.add_argument("--no-camera-undistort",dest="camera_undistort",action="store_false")
    p.add_argument("--camera-calibration",default=DEFAULT_CALIBRATION); p.add_argument("--camera-undistort-alpha",type=float,default=0.0)
    p.add_argument("--camera-device",default="/dev/video0"); p.add_argument("--camera-controls-json",default=DEFAULT_CAMERA_CONTROLS)
    p.add_argument("--camera-controls-enable",action="store_true",default=True); p.add_argument("--no-camera-controls-enable",dest="camera_controls_enable",action="store_false")
    p.add_argument("--camera-controls-strict",action="store_true",default=True); p.add_argument("--no-camera-controls-strict",dest="camera_controls_strict",action="store_false")
    p.add_argument("--camera-controls-stabilization-frames",type=int,default=12)
    p.add_argument("--seg-model",default="/workspace/project_train/aruco_test/dual/models/kfashion_yolo26s_seg3_e100_best.engine")
    p.add_argument("--seg-imgsz",type=int,default=640); p.add_argument("--seg-conf-ladder",default="0.12,0.07,0.03"); p.add_argument("--seg-classes",default="bottoms,bottom,pants,trousers,outer")
    p.add_argument("--seg-min-area-px",type=float,default=2500); p.add_argument("--board-roi-shrink-px",type=int,default=10)
    p.add_argument("--pose-model",default="/workspace/project_train/yolo26/bottom_pose8_yolo26m_e40_best.engine"); p.add_argument("--pose-imgsz",type=int,default=640); p.add_argument("--pose-conf",type=float,default=0.10); p.add_argument("--pose-kpt-conf",type=float,default=0.20)
    p.add_argument("--pose-tta-angles",default="0,180,90,-90,45,-45"); p.add_argument("--pose-tta-flips",default="none,h")
    p.add_argument("--plate-layout",default="c11_plate_layout.json"); p.add_argument("--empty-board",default="c68_elp_undistorted_empty_board.png"); p.add_argument("--left-fold-x",type=float,default=None)
    p.add_argument("--seam-px-per-mm",type=float,default=1.0); p.add_argument("--seam-search-lo-frac",type=float,default=0.18); p.add_argument("--seam-search-hi-frac",type=float,default=0.48); p.add_argument("--seam-min-score",type=float,default=0.28); p.add_argument("--seam-layout-guard-mm",type=float,default=70.0)
    p.add_argument("--angle-tolerance-deg",type=float,default=6.0); p.add_argument("--offset-tolerance-mm",type=float,default=25.0)
    p.add_argument("--axis-min-hem-length-mm",type=float,default=150.0); p.add_argument("--axis-min-crotch-length-mm",type=float,default=70.0); p.add_argument("--min-axis-length-mm",type=float,default=150.0)
    p.add_argument("--min-solidity",type=float,default=0.64); p.add_argument("--min-hem-gap-ratio",type=float,default=0.38,help="diagnostic only in D57-8; does not block ROTATE")
    p.add_argument("--grip-station-ratios",default="0.28,0.36,0.44,0.52,0.22,0.60",help="legacy alias; D57-8 uses --rotate-grip-station-ratios")
    p.add_argument("--grip-station-band-mm",type=float,default=24.0,help="legacy compatibility")
    p.add_argument("--grip-inset-mm",type=float,default=30.0,help="legacy compatibility")
    p.add_argument("--grip-snap-radius-mm",type=float,default=32.0,help="legacy compatibility")
    p.add_argument("--mask-sample-stride-px",type=int,default=3)
    p.add_argument("--rotate-grip-station-ratios",default="0.28,0.36,0.44,0.52,0.22,0.60")
    p.add_argument("--rotate-grip-section-band-mm",type=float,default=22.0)
    p.add_argument("--rotate-grip-inset-mm",type=float,default=28.0)
    p.add_argument("--rotate-grip-snap-mm",type=float,default=38.0)
    p.add_argument("--rotate-grip-min-inside-px",type=float,default=3.0)
    p.add_argument("--rotate-grip-max-station-snap-mm",type=float,default=20.0)
    p.add_argument("--rotate-grip-pair-station-mismatch-mm",type=float,default=24.0)
    p.add_argument("--rotate-grip-min-raw-width-mm",type=float,default=95.0)
    p.add_argument("--rotate-grip-max-raw-width-mm",type=float,default=520.0)
    p.add_argument("--rotate-grip-preferred-pair-mm",type=float,default=230.0)
    p.add_argument("--rotate-contour-board-margin-mm",type=float,default=6.0)
    p.add_argument("--rotate-contour-min-inside-ratio",type=float,default=0.965)
    p.add_argument("--grip-min-pair-separation-mm",type=float,default=95.0); p.add_argument("--grip-max-pair-separation-mm",type=float,default=430.0)
    p.add_argument("--center-dead-half-width",type=float,default=40.0); p.add_argument("--board-margin-mm",type=float,default=18.0); p.add_argument("--roarm-xy-radius-max-mm",type=float,default=420.0); p.add_argument("--min-z",type=float,default=-180.0); p.add_argument("--max-z",type=float,default=420.0)
    p.add_argument("--max-rotation-deg",type=float,default=45.0); p.add_argument("--min-rotation-action-deg",type=float,default=5.0); p.add_argument("--rotation-step-deg",type=float,default=12.0); p.add_argument("--distance-max-stretch-mm",type=float,default=4.0)
    p.add_argument("--low-lift-mm",type=float,default=75.0); p.add_argument("--pre-slack-mm",type=float,default=0.0,help="legacy compatibility only; D57-8 never uses pre-slack"); p.add_argument("--place-clearance-mm",type=float,default=7.0)
    p.add_argument("--free-speed",type=float,default=1.12); p.add_argument("--near-speed",type=float,default=0.62); p.add_argument("--lift-speed",type=float,default=0.45); p.add_argument("--rotate-speed",type=float,default=0.68); p.add_argument("--vertical-retract-speed",type=float,default=1.0); p.add_argument("--sync-min-scale",type=float,default=0.50)
    p.add_argument("--rotation-step-wait-s",type=float,default=0.28)
    p.add_argument("--hover-z",type=float,default=None); p.add_argument("--arm1-grip-z",type=float,default=None); p.add_argument("--arm2-grip-z",type=float,default=None)
    p.add_argument("--grip-open",type=float,default=1.35); p.add_argument("--grip-close",type=float,default=3.05); p.add_argument("--grip-hold",type=float,default=3.14); p.add_argument("--release-open",type=float,default=1.35)
    p.add_argument("--close-repeat",type=int,default=2); p.add_argument("--close-limited-wait-s",type=float,default=0.25); p.add_argument("--close-repeat-gap-s",type=float,default=0.12); p.add_argument("--close-final-hold-s",type=float,default=0.28); p.add_argument("--close-extra-retries",type=int,default=3); p.add_argument("--close-retry-wait-s",type=float,default=0.45); p.add_argument("--obvious-open-feedback-rad",type=float,default=1.80)
    p.add_argument("--feedback-query-retries",type=int,default=3); p.add_argument("--feedback-query-gap-s",type=float,default=0.10)
    p.add_argument("--pre-open-wait-s",type=float,default=0.08); p.add_argument("--hover-wait-s",type=float,default=1.1); p.add_argument("--open-before-descent-wait-s",type=float,default=0.10); p.add_argument("--contact-move-wait-s",type=float,default=1.4); p.add_argument("--low-lift-wait-s",type=float,default=1.0)
    p.add_argument("--contact-hold-s",type=float,default=0.15); p.add_argument("--place-settle-s",type=float,default=0.40); p.add_argument("--release-wait-s",type=float,default=0.60)
    p.add_argument("--place-move-wait-s",type=float,default=1.40); p.add_argument("--vertical-retract-wait-s",type=float,default=1.30)
    p.add_argument("--arm1-port",default="/dev/roarm_1"); p.add_argument("--arm2-port",default="/dev/roarm_2")
    p.add_argument("--arm1-standby-x",type=float,default=-10.47886); p.add_argument("--arm1-standby-y",type=float,default=227.544386); p.add_argument("--arm1-standby-z",type=float,default=170.929303); p.add_argument("--arm1-standby-t",type=float,default=2.124563)
    p.add_argument("--arm2-standby-x",type=float,default=2.870034); p.add_argument("--arm2-standby-y",type=float,default=-233.859636); p.add_argument("--arm2-standby-z",type=float,default=102.23829); p.add_argument("--arm2-standby-t",type=float,default=1.356039); p.add_argument("--standby-speed",type=float,default=1.12); p.add_argument("--standby-wait-s",type=float,default=1.3)
    p.add_argument("--send",action="store_true",default=False); p.add_argument("--no-window",action="store_true",default=False); p.add_argument("--save-overlay",default="d57_7_plan.jpg")
    return p


def d57_legacy_main() -> int:
    args=parser_build().parse_args()
    if YOLO is None: raise RuntimeError("ultralytics import failed")
    args.config=resolve_file(args.config); args.hfile=resolve_file(args.hfile); args.camera_calibration=resolve_file(args.camera_calibration); args.camera_controls_json=resolve_file(args.camera_controls_json)
    config=load_json(args.config)
    if not config: raise RuntimeError(f"config not found/invalid: {args.config}")
    if not args.camera_undistort:
        raise RuntimeError("D57-8 requires ELP undistortion; --no-camera-undistort is intentionally blocked")
    if CameraUndistorter is None:
        raise RuntimeError(f"camera_undistort.py unavailable: {CAMERA_UNDISTORT_IMPORT_ERROR!r}")
    if not os.path.isfile(args.camera_calibration):
        raise RuntimeError(f"camera calibration not found: {args.camera_calibration}")

    # Match main: open camera first, apply the fixed ELP profile, then prepare the
    # undistorter against the ACTUAL frame size rather than trusting CLI dimensions.
    cap=open_camera(args.camera,args.width,args.height,args.backend)
    if not cap.isOpened(): raise RuntimeError("camera open failed")
    if args.camera_controls_enable:
        apply_camera_controls(args.camera_controls_json,args.camera_device,args.camera_controls_strict)
    for _ in range(max(0,int(args.camera_controls_stabilization_frames))):
        cap.read()
    ok_probe,probe=cap.read()
    if not ok_probe or probe is None:
        raise RuntimeError("camera opened but initial frame read failed")
    actual_size=(int(probe.shape[1]),int(probe.shape[0]))
    if actual_size != (int(args.width),int(args.height)):
        raise RuntimeError(f"camera size {actual_size} != requested {(args.width,args.height)}")
    und=CameraUndistorter(args.camera_calibration,alpha=float(args.camera_undistort_alpha),strict_size=True)
    und.prepare(actual_size)
    cam_meta=und.info().to_metadata()
    print(f"[D57-8-UNDISTORT-MODULE] {CAMERA_UNDISTORT_MODULE_PATH}")
    print("[D57-8-UNDISTORT] "+und.status_line())
    # Verify that the actual runtime frame is really passing through the remap.
    probe_corrected = und.correct(probe)
    if probe_corrected.shape != probe.shape:
        raise RuntimeError(f"undistort output shape {probe_corrected.shape} != raw {probe.shape}")
    _mad = float(np.mean(cv2.absdiff(probe, probe_corrected)))
    _maxdiff = int(np.max(cv2.absdiff(probe, probe_corrected)))
    print(f"[D57-8-UNDISTORT-CHECK] raw->corrected meanAbsDiff={_mad:.3f} maxDiff={_maxdiff}; DISPLAY/ARUCO/SEG/POSE=corrected only")
    print(f"[D57-8-CAMERA] actual={actual_size} geometry=CORRECTED_ONLY")

    # D57 is corrected-frame only. Never silently pair a corrected image with raw_H.
    H,_h_bundle=load_corrected_h(args.hfile,cam_meta)
    if H is None: print("[D57-8-H] corrected cache unavailable/mismatched; press L after launch")
    seg_path=resolve_file(args.seg_model); pose_path=resolve_file(args.pose_model)
    print(f"[D57-8-MODEL] SEG task=segment path={seg_path}")
    seg=YOLO(seg_path, task="segment")
    print(f"[D57-8-MODEL] POSE task=pose path={pose_path}")
    pose_model=YOLO(pose_path, task="pose")
    locked:Optional[D57Plan]=None; last_canvas=None; window="D57-8 BOTTOM ROTATE ONLY"
    print("\n[D57-8 KEYS] TERMINAL/GUI: I=infer+lock | Enter=execute | L=relock H | R=discard | Q/ESC=quit")
    print("[D57-8 POLICY] OUTER CONTOUR GRIP -> DUAL LOW-LIFT ROTATE; LEFT_FOLD supplies ORIENTATION ONLY")
    if args.send: print("[D57-8] PHYSICAL SEND=ON")
    else: print("[D57-8] DRY PREVIEW ONLY (--send to move robot)")
    terminal=TerminalKeyReader()
    terminal.start()
    try:
        while True:
            ok,raw=cap.read()
            if not ok or raw is None: time.sleep(0.02); continue
            frame=und.correct(raw)
            canvas=frame.copy() if locked is None or locked.overlay is None else locked.overlay.copy()
            draw_text(canvas,"ELP UNDISTORT: ON | MAIN-COMPAT CAMERA | CORRECTED ONLY",(20,56),(0,255,255),0.50,2)
            if locked is None:
                draw_text(canvas,"D57-8 ROTATE ONLY | press I to build fresh plan",(20,30),(0,255,0),0.65,2)
            else:
                age=time.time()-locked.created_at; draw_text(canvas,f"I-LOCK age={age:.1f}s | Enter={'EXECUTE' if locked.ok else 'BLOCKED'} | R=discard",(20,frame.shape[0]-20),(0,255,255),0.50,2)
            last_canvas=canvas
            if not args.no_window:
                cv2.imshow(window,canvas)
                gui_key=cv2.waitKey(1)&0xFF
            else:
                gui_key=255
            term_key=terminal.read_key()
            key=term_key if term_key != 255 else gui_key
            if term_key != 255:
                _label = "ENTER" if term_key in (10,13) else ("ESC" if term_key == 27 else chr(term_key).upper() if 0 <= term_key < 256 else str(term_key))
                print(f"[D57-8-KEY] terminal={_label}")
            if key in (ord('q'),27): break
            if key==ord('r'):
                locked=None; print("[D57] plan discarded"); continue
            if key==ord('l'):
                Hn=aruco_lock(frame,config)
                if Hn is not None:
                    H=Hn; save_corrected_h_preserve_bundle(args.hfile,H,cam_meta); locked=None; print("[D57-8-H] locked corrected-frame H")
                continue
            if key==ord('i'):
                if H is None: print("[D57-I] H unavailable; press L"); continue
                t0=time.time(); m,ms=infer_mask(seg,frame,H,config,args); print(f"[D57-MASK] {ms}")
                if m is None: locked=None; continue
                po,ps=infer_pose(pose_model,frame,H,m,args); print(f"[D57-POSE] {ps}")
                if po is None: locked=None; continue
                fold,fr=resolve_left_fold(H,config,args); print(f"[D57-LEFT-FOLD] x={fold} info={fr}")
                if fold is None: locked=None; continue
                locked=build_plan(frame.copy(),m,po,H,config,float(fold),args)
                if locked.overlay is not None and args.save_overlay:
                    try: cv2.imwrite(args.save_overlay,locked.overlay)
                    except Exception: pass
                print(f"[D57-PLAN] ok={locked.ok} mode={locked.mode} reason={locked.reason} angleErr={locked.angle_error_deg:.1f}deg offsetDiag={locked.axis_offset_mm:+.1f}mm actionRot={locked.action_delta_deg:+.1f} time={(time.time()-t0)*1000:.0f}ms")
                if locked.grip is not None:
                    print(f"[D57-8-CONTOUR-GRIP] A2={np.round(locked.grip.arm2,1).tolist()} A1={np.round(locked.grip.arm1,1).tolist()} sep={locked.grip.pair_distance_mm:.1f}mm station={locked.grip.station_ratio:.2f} inset={locked.grip.inset_mm:.1f}mm")
                continue
            if key in (10,13):
                if locked is None: print("[D57-ENTER] no I-locked plan"); continue
                if locked.mode=="ALIGNED": print("[D57-ENTER] already aligned; NO_ACTION"); locked=None; continue
                if not locked.ok: print(f"[D57-ENTER] blocked: {locked.reason}"); continue
                # Locked plan is intentionally one-shot. Clear before moving so it cannot execute twice.
                plan=locked; locked=None
                execute_plan(plan,config,args)
                print("[D57] physical round finished -> press I for FRESH REJUDGE")
                continue
    finally:
        terminal.stop()
        cap.release()
        if not args.no_window: cv2.destroyAllWindows()
    return 0


# =============================================================================
# D58-2 CIRCUMCENTER POSITION CORRECTION
#   Camera/geometry: current corrected ELP pipeline inherited from D57-8 only.
#   B34/B46 contribution: vacancy-side circumcenter target + contour-inset contact idea.
#   Motion: single-arm grip -> low lift -> short translation -> place -> fresh rejudge.
# =============================================================================

@dataclass
class D58Plan:
    ok: bool = False
    reason: str = "UNINITIALIZED"
    created_at: float = field(default_factory=time.time)
    center_board: Optional[np.ndarray] = None
    mask_center_board: Optional[np.ndarray] = None
    pose_center_board: Optional[np.ndarray] = None
    target_board: Optional[np.ndarray] = None
    target_source: str = "NONE"
    side_a: str = ""
    side_b: str = ""
    gaps: Dict[str, float] = field(default_factory=dict)
    triangle: Optional[np.ndarray] = None
    contour_vertex: Optional[np.ndarray] = None
    circum_radius_mm: Optional[float] = None
    pull_unit: Optional[np.ndarray] = None
    selected_arm: Optional[str] = None
    grip_board: Optional[np.ndarray] = None
    end_board: Optional[np.ndarray] = None
    move_mm: float = 0.0
    grip_inset_mm: float = 0.0
    overlay: Optional[np.ndarray] = None


def _d58_unit(v: np.ndarray) -> Optional[np.ndarray]:
    a=np.asarray(v,np.float32).reshape(2)
    n=float(np.linalg.norm(a))
    if n < 1e-6:
        return None
    return (a/n).astype(np.float32)


def _d58_pose_center(pose: Optional[PoseResult]) -> Optional[np.ndarray]:
    if pose is None:
        return None
    try:
        w=np.asarray(pose.waist_center,np.float32).reshape(2)
        if pose.lower_center is not None:
            l=np.asarray(pose.lower_center,np.float32).reshape(2)
            return (0.5*(w+l)).astype(np.float32)
        if pose.crotch is not None:
            c=np.asarray(pose.crotch,np.float32).reshape(2)
            return (0.5*(w+c)).astype(np.float32)
        return w
    except Exception:
        return None


def _d58_robust_board_points(mask: MaskResult, H: np.ndarray, stride_px: int=3) -> np.ndarray:
    pts=mask_board_samples(mask,H,stride_px=max(1,int(stride_px)))
    if pts is None or len(pts)==0:
        return np.zeros((0,2),np.float32)
    good=np.all(np.isfinite(pts),axis=1)
    return np.asarray(pts[good],np.float32)


def _d58_gap_info(samples: np.ndarray, config: Dict[str,Any], pct: float=2.0) -> Dict[str,Any]:
    xmin,xmax,ymin,ymax=board_bounds(config)
    q=float(np.clip(pct,0.0,15.0))
    xlo=float(np.percentile(samples[:,0],q)); xhi=float(np.percentile(samples[:,0],100.0-q))
    ylo=float(np.percentile(samples[:,1],q)); yhi=float(np.percentile(samples[:,1],100.0-q))
    gaps={
        'LEFT': max(0.0,xlo-xmin),
        'RIGHT': max(0.0,xmax-xhi),
        'BOTTOM': max(0.0,ylo-ymin),
        'TOP': max(0.0,ymax-yhi),
    }
    ordered=sorted(gaps.keys(),key=lambda k:gaps[k],reverse=True)
    return {'gaps':gaps,'ordered':ordered,'robust_bounds':[xlo,xhi,ylo,yhi]}


def _d58_side_midpoint(side: str, config: Dict[str,Any]) -> np.ndarray:
    xmin,xmax,ymin,ymax=board_bounds(config)
    if side=='LEFT': return np.asarray([xmin,0.5*(ymin+ymax)],np.float32)
    if side=='RIGHT': return np.asarray([xmax,0.5*(ymin+ymax)],np.float32)
    if side=='BOTTOM': return np.asarray([0.5*(xmin+xmax),ymin],np.float32)
    return np.asarray([0.5*(xmin+xmax),ymax],np.float32)


def _d58_side_dir(side: str) -> np.ndarray:
    return {
        'LEFT':np.asarray([-1.0,0.0],np.float32),
        'RIGHT':np.asarray([1.0,0.0],np.float32),
        'BOTTOM':np.asarray([0.0,-1.0],np.float32),
        'TOP':np.asarray([0.0,1.0],np.float32),
    }[side]


def _d58_choose_two_sides(info: Dict[str,Any]) -> Tuple[Optional[str],Optional[str]]:
    ordered=list(info.get('ordered',[]))
    if len(ordered)<2:
        return None,None
    a=ordered[0]
    opposite={('LEFT','RIGHT'),('RIGHT','LEFT'),('TOP','BOTTOM'),('BOTTOM','TOP')}
    # Prefer two adjacent vacancies; opposite sides make a weak/degenerate triangle.
    for b in ordered[1:]:
        if (a,b) not in opposite:
            return a,b
    return a,ordered[1]


def _d58_circumcenter(a: np.ndarray,b: np.ndarray,c: np.ndarray) -> Tuple[Optional[np.ndarray],Optional[float],float]:
    a=np.asarray(a,float).reshape(2); b=np.asarray(b,float).reshape(2); c=np.asarray(c,float).reshape(2)
    area2=float(np.cross(b-a,c-a))
    tri_area=0.5*abs(area2)
    d=2.0*(a[0]*(b[1]-c[1])+b[0]*(c[1]-a[1])+c[0]*(a[1]-b[1]))
    if abs(d)<1e-8:
        return None,None,tri_area
    aa=float(np.dot(a,a)); bb=float(np.dot(b,b)); cc=float(np.dot(c,c))
    ux=(aa*(b[1]-c[1])+bb*(c[1]-a[1])+cc*(a[1]-b[1]))/d
    uy=(aa*(c[0]-b[0])+bb*(a[0]-c[0])+cc*(b[0]-a[0]))/d
    center=np.asarray([ux,uy],np.float32)
    radius=float(np.linalg.norm(center-a))
    if not np.all(np.isfinite(center)) or not math.isfinite(radius):
        return None,None,tri_area
    return center,radius,tri_area


def _d58_contour_board(mask: MaskResult,H: np.ndarray) -> np.ndarray:
    px=np.asarray(mask.contour,np.float32).reshape(-1,2)
    if len(px)==0:
        return np.zeros((0,2),np.float32)
    return perspective_points(H,px)


def _d58_choose_triangle_vertex(contour_board: np.ndarray, center: np.ndarray,
                                side_a: str, side_b: str, m1: np.ndarray, m2: np.ndarray,
                                edge_band_mm: float, equal_tol_mm: float) -> Optional[np.ndarray]:
    if contour_board is None or len(contour_board)<4:
        return None
    direction=_d58_unit(_d58_side_dir(side_a)+_d58_side_dir(side_b))
    if direction is None:
        return None
    proj=(contour_board-center.reshape(1,2))@direction
    mx=float(np.max(proj)); band=max(20.0,float(edge_band_mm))
    idx=np.where(proj>=mx-band)[0]
    cand=contour_board[idx] if len(idx) else contour_board[[int(np.argmax(proj))]]
    d1=np.linalg.norm(cand-m1.reshape(1,2),axis=1)
    d2=np.linalg.norm(cand-m2.reshape(1,2),axis=1)
    bal=np.abs(d1-d2); best_bal=float(np.min(bal))
    keep=np.where(bal<=best_bal+max(0.0,float(equal_tol_mm)))[0]
    if len(keep)==0:
        keep=np.asarray([int(np.argmin(bal))])
    avg=0.5*(d1[keep]+d2[keep])
    return np.asarray(cand[keep[int(np.argmin(avg))]],np.float32)


def _d58_local_mask_ratio(mask_u8: np.ndarray, pxy: Tuple[float,float], radius: int=13) -> float:
    h,w=mask_u8.shape[:2]; x=int(round(float(pxy[0]))); y=int(round(float(pxy[1]))); r=max(3,int(radius))
    x0=max(0,x-r); x1=min(w,x+r+1); y0=max(0,y-r); y1=min(h,y+r+1)
    if x1<=x0 or y1<=y0: return 0.0
    patch=mask_u8[y0:y1,x0:x1]
    yy,xx=np.ogrid[y0:y1,x0:x1]
    circle=((xx-x)**2+(yy-y)**2)<=r*r
    den=max(1,int(np.count_nonzero(circle)))
    return float(np.count_nonzero((patch>0)&circle))/float(den)


def _d58_mask_support(mask: MaskResult,H: np.ndarray,p: np.ndarray,args) -> Tuple[bool,float,float]:
    pix=board_to_pixel(H,float(p[0]),float(p[1]))
    if pix is None: return False,-999.0,0.0
    try: signed=float(cv2.pointPolygonTest(mask.contour,(float(pix[0]),float(pix[1])),True))
    except Exception: signed=-999.0
    lr=_d58_local_mask_ratio(mask.mask_u8,pix,int(args.d58_local_mask_radius_px))
    ok=signed>=float(args.d58_min_inside_px) and lr>=float(args.d58_min_local_mask_ratio)
    return bool(ok),signed,lr


def _d58_arm_point_safe(config: Dict[str,Any],arm: str,p: np.ndarray,z: float,args,require_workspace: bool=True) -> bool:
    if not board_inside(config,p,float(args.d58_board_margin_mm)):
        return False
    if require_workspace and arm_for_x(config,float(p[0]),float(args.center_dead_half_width))!=arm:
        return False
    try:
        x,y=board_to_arm_xy(config,arm,float(p[0]),float(p[1]))
    except Exception:
        return False
    if math.hypot(x,y)>float(args.d58_roarm_xy_radius_max_mm):
        return False
    return float(args.min_z)<=float(z)<=float(args.max_z)


def _d58_contact_candidates(mask: MaskResult,H: np.ndarray,config: Dict[str,Any],center: np.ndarray,
                            pull_u: np.ndarray,requested_move: float,args,preferred_vertex: Optional[np.ndarray]=None) -> Tuple[Optional[Dict[str,Any]],str]:
    contour=_d58_contour_board(mask,H)
    if contour is None or len(contour)<4:
        return None,'contour unavailable'
    proj=(contour-center.reshape(1,2))@pull_u.reshape(2)
    mx=float(np.max(proj)); band=max(25.0,float(args.d58_contact_edge_band_mm))
    ids=np.where(proj>=mx-band)[0]
    if len(ids)==0: ids=np.asarray([int(np.argmax(proj))])
    # Evaluate the most advanced contour points first, then select for real mask support + arm reach.
    ids=ids[np.argsort(proj[ids])[::-1]]
    desired_anchor=None
    if preferred_vertex is not None:
        pv=np.asarray(preferred_vertex,np.float32).reshape(2)
        pin=_d58_unit(center-pv)
        if pin is not None:
            desired_anchor=(pv+pin*float(args.d58_contact_inset_mm)).astype(np.float32)
            nearest_idx=int(np.argmin(np.linalg.norm(contour-pv.reshape(1,2),axis=1)))
            ids=np.concatenate([np.asarray([nearest_idx],dtype=int),ids[ids!=nearest_idx]])
    best=None
    max_eval=min(int(args.d58_contact_max_candidates),len(ids))
    preferred_arm='arm1' if float(pull_u[0])>float(args.d58_horizontal_arm_deadband) else ('arm2' if float(pull_u[0])<-float(args.d58_horizontal_arm_deadband) else None)
    for ii in ids[:max_eval]:
        raw=np.asarray(contour[int(ii)],np.float32)
        inward=_d58_unit(center-raw)
        if inward is None: continue
        for inset in (float(args.d58_contact_inset_mm),float(args.d58_contact_inset_mm)+12.0,float(args.d58_contact_inset_mm)+24.0):
            grip=(raw+inward*inset).astype(np.float32)
            okmask,signed,local=_d58_mask_support(mask,H,grip,args)
            if not okmask: continue
            arm=arm_for_x(config,float(grip[0]),float(args.center_dead_half_width))
            if arm not in ('arm1','arm2'): continue
            cz=contact_z(config,arm,args)
            hz=hover_z(config,arm,args)
            clear_z=min(float(hz),float(cz)+float(args.d58_post_drag_clear_mm))
            if not _d58_arm_point_safe(config,arm,grip,cz,args): continue
            # D58-2 preflight follows the real execution: same-Z drag first,
            # then vertical clear only at the final XY.
            chosen=0.0; end=None
            d=float(requested_move)
            while d>=float(args.d58_min_move_mm)-1e-6:
                q=(grip+pull_u*d).astype(np.float32)
                if (_d58_arm_point_safe(config,arm,q,cz,args) and
                    _d58_arm_point_safe(config,arm,q,clear_z,args)):
                    chosen=d; end=q; break
                d-=5.0
            if end is None: continue
            try:
                rx,ry=board_to_arm_xy(config,arm,float(grip[0]),float(grip[1])); r0=math.hypot(rx,ry)
                ex,ey=board_to_arm_xy(config,arm,float(end[0]),float(end[1])); r1=math.hypot(ex,ey)
            except Exception: continue
            arm_bonus=18.0 if preferred_arm==arm else 0.0
            anchor_pen=0.0 if desired_anchor is None else 0.18*float(np.linalg.norm(grip-desired_anchor))
            score=(65.0*local + 2.0*signed + 0.20*chosen + arm_bonus
                   -0.035*max(r0,r1) -0.08*inset - anchor_pen)
            item={'arm':arm,'raw':raw,'grip':grip,'end':end,'move_mm':float(chosen),'inset_mm':float(inset),
                  'local':float(local),'inside_px':float(signed),'score':float(score),'r0':float(r0),'r1':float(r1)}
            if best is None or item['score']>best['score']:
                best=item
    if best is None:
        return None,'no contour-inset contact is safe for same-arm low-lift translation'
    return best,'OK'


def _d58_weighted_fallback_target(gaps: Dict[str,float],side_a: str,side_b: str,config: Dict[str,Any],margin: float) -> np.ndarray:
    m1=_d58_side_midpoint(side_a,config); m2=_d58_side_midpoint(side_b,config)
    w1=max(1.0,float(gaps.get(side_a,1.0))); w2=max(1.0,float(gaps.get(side_b,1.0)))
    side=(w1*m1+w2*m2)/(w1+w2)
    xmin,xmax,ymin,ymax=board_bounds(config); bc=np.asarray([0.5*(xmin+xmax),0.5*(ymin+ymax)],np.float32)
    t=(0.68*side+0.32*bc).astype(np.float32)
    t[0]=float(np.clip(t[0],xmin+margin,xmax-margin)); t[1]=float(np.clip(t[1],ymin+margin,ymax-margin))
    return t


def build_d58_plan(frame: np.ndarray,mask: MaskResult,pose: Optional[PoseResult],H: np.ndarray,
                    config: Dict[str,Any],args) -> D58Plan:
    plan=D58Plan()
    samples=_d58_robust_board_points(mask,H,int(args.mask_sample_stride_px))
    if len(samples)<30:
        plan.reason='MASK_BOARD_SAMPLES_TOO_FEW'; return plan
    gapinfo=_d58_gap_info(samples,config,float(args.d58_gap_percentile))
    gaps=gapinfo['gaps']; plan.gaps={k:float(v) for k,v in gaps.items()}
    sa,sb=_d58_choose_two_sides(gapinfo)
    if sa is None or sb is None:
        plan.reason='VACANCY_SIDES_UNAVAILABLE'; return plan
    plan.side_a,plan.side_b=sa,sb
    if float(gaps.get(sa,0.0))<float(args.d58_min_vacancy_gap_mm):
        plan.reason=f"NO_POSITION_CORRECTION majorGap={float(gaps.get(sa,0.0)):.1f}mm"; return plan

    mask_center=np.median(samples,axis=0).astype(np.float32)
    pose_center=_d58_pose_center(pose)
    if pose_center is not None and board_inside(config,pose_center,0.0):
        w=float(np.clip(args.d58_pose_center_weight,0.0,0.5))
        center=((1.0-w)*mask_center+w*pose_center).astype(np.float32)
    else:
        pose_center=None; center=mask_center.copy()
    plan.mask_center_board=mask_center; plan.pose_center_board=pose_center; plan.center_board=center

    m1=_d58_side_midpoint(sa,config); m2=_d58_side_midpoint(sb,config)
    contour_board=_d58_contour_board(mask,H)
    vertex=_d58_choose_triangle_vertex(contour_board,center,sa,sb,m1,m2,
                                       float(args.d58_contour_edge_band_mm),float(args.d58_contour_equal_tolerance_mm))
    if vertex is None:
        plan.reason='CIRCUMCENTER_VERTEX_UNAVAILABLE'; return plan
    plan.contour_vertex=vertex
    circ,radius,area=_d58_circumcenter(m1,m2,vertex)
    plan.triangle=np.vstack([m1,m2,vertex]).astype(np.float32)
    target=None; source=''
    if circ is not None and radius is not None:
        if (area>=float(args.d58_min_triangle_area_mm2) and
            radius<=float(args.d58_max_circumradius_mm) and
            board_inside(config,circ,float(args.d58_circumcenter_safe_margin_mm))):
            target=circ.astype(np.float32); source='CIRCUMCENTER'; plan.circum_radius_mm=float(radius)
    if target is None:
        target=_d58_weighted_fallback_target(gaps,sa,sb,config,float(args.d58_circumcenter_safe_margin_mm))
        source=f'WEIGHTED_FALLBACK(area={area:.0f},R={radius if radius is not None else float("nan"):.1f})'
    plan.target_board=target; plan.target_source=source
    delta=target-center; dist=float(np.linalg.norm(delta)); u=_d58_unit(delta)
    if u is None or dist<float(args.d58_done_center_error_mm):
        plan.reason=f'NO_ACTION centerTargetError={dist:.1f}mm'; return plan
    span_x=float(np.ptp(samples[:,0])) if len(samples) else 0.0
    span_y=float(np.ptp(samples[:,1])) if len(samples) else 0.0
    garment_span=max(span_x,span_y)
    move_cap=float(args.d58_max_move_mm)
    if garment_span>=float(getattr(args,'d58_long_garment_span_mm',480.0)):
        move_cap=max(move_cap,float(getattr(args,'d58_long_garment_max_move_mm',165.0)))
    request=min(move_cap,max(float(args.d58_min_move_mm),dist*float(args.d58_center_target_gain)))
    print(f"[D58-3-MOVE-SCALE] garmentSpan={garment_span:.1f}mm targetError={dist:.1f}mm request={request:.1f}mm cap={move_cap:.1f}mm boardInnerMargin={float(args.d58_board_margin_mm):.1f}mm")
    contact,why=_d58_contact_candidates(mask,H,config,center,u,request,args,preferred_vertex=vertex)
    if contact is None:
        plan.reason='CONTACT_UNAVAILABLE: '+why; return plan
    plan.pull_unit=u; plan.selected_arm=str(contact['arm']); plan.grip_board=np.asarray(contact['grip'],np.float32)
    plan.end_board=np.asarray(contact['end'],np.float32); plan.move_mm=float(contact['move_mm']); plan.grip_inset_mm=float(contact['inset_mm'])
    plan.ok=True; plan.reason='OK'
    print(f"[D58-2-VACANCY] gaps="+','.join(f"{k}:{v:.1f}" for k,v in sorted(gaps.items()))+f" selected={sa}+{sb}")
    print(f"[D58-2-CIRC] source={source} center={np.round(center,1).tolist()} target={np.round(target,1).tolist()} radius={plan.circum_radius_mm} triArea={area:.0f}")
    print(f"[D58-2-GRIP] arm={plan.selected_arm.upper()} grip={np.round(plan.grip_board,1).tolist()} end={np.round(plan.end_board,1).tolist()} move={plan.move_mm:.1f}mm inset={plan.grip_inset_mm:.1f}mm local={contact['local']:.2f} reachR={contact['r0']:.0f}->{contact['r1']:.0f}")
    plan.overlay=make_d58_overlay(frame,mask,plan,H,config)
    return plan


def _d58_draw_board_point(img: np.ndarray,H: np.ndarray,p: Optional[np.ndarray],color,label,r=8):
    if p is None: return
    q=board_to_pixel(H,float(p[0]),float(p[1]))
    if q is None: return
    x,y=int(round(q[0])),int(round(q[1])); cv2.circle(img,(x,y),r+3,(0,0,0),-1); cv2.circle(img,(x,y),r,color,-1)
    cv2.putText(img,label,(x+10,y-10),cv2.FONT_HERSHEY_SIMPLEX,0.48,(0,0,0),3); cv2.putText(img,label,(x+10,y-10),cv2.FONT_HERSHEY_SIMPLEX,0.48,color,1)


def make_d58_overlay(frame: np.ndarray,mask: MaskResult,plan: D58Plan,H: np.ndarray,config: Dict[str,Any]) -> np.ndarray:
    out=frame.copy(); ov=out.copy(); cv2.drawContours(ov,[mask.contour],-1,(90,160,255),-1); out=cv2.addWeighted(ov,0.20,out,0.80,0)
    cv2.drawContours(out,[mask.contour],-1,(255,255,255),2)
    if plan.triangle is not None:
        pp=[]
        for b in plan.triangle:
            q=board_to_pixel(H,float(b[0]),float(b[1]));
            if q is not None: pp.append((int(round(q[0])),int(round(q[1]))))
        if len(pp)==3:
            cv2.polylines(out,[np.asarray(pp,np.int32).reshape(-1,1,2)],True,(255,0,255),2)
    if plan.target_source.startswith('CIRCUMCENTER') and plan.target_board is not None and plan.circum_radius_mm is not None:
        pts=[]
        for a in np.linspace(0,2*math.pi,72,endpoint=False):
            b=plan.target_board+float(plan.circum_radius_mm)*np.asarray([math.cos(a),math.sin(a)],np.float32)
            q=board_to_pixel(H,float(b[0]),float(b[1]))
            if q is not None: pts.append((int(round(q[0])),int(round(q[1]))))
        if len(pts)>8: cv2.polylines(out,[np.asarray(pts,np.int32).reshape(-1,1,2)],True,(180,80,255),1)
    _d58_draw_board_point(out,H,plan.mask_center_board,(0,180,255),'MASK CENTER',7)
    _d58_draw_board_point(out,H,plan.pose_center_board,(255,180,0),'POSE CENTER',6)
    _d58_draw_board_point(out,H,plan.center_board,(0,255,255),'CENTER',8)
    _d58_draw_board_point(out,H,plan.contour_vertex,(255,0,255),'TRI VERTEX',7)
    _d58_draw_board_point(out,H,plan.target_board,(0,255,0),'CIRC TARGET',9)
    _d58_draw_board_point(out,H,plan.grip_board,(0,0,255),'GRIP',9)
    _d58_draw_board_point(out,H,plan.end_board,(255,0,0),'END',9)
    if plan.grip_board is not None and plan.end_board is not None:
        a=board_to_pixel(H,float(plan.grip_board[0]),float(plan.grip_board[1])); b=board_to_pixel(H,float(plan.end_board[0]),float(plan.end_board[1]))
        if a is not None and b is not None: cv2.arrowedLine(out,(int(a[0]),int(a[1])),(int(b[0]),int(b[1])),(0,255,255),3,tipLength=0.12)
    draw_text(out,f"D58-2 CIRC POSITION | {'READY' if plan.ok else 'BLOCKED'} | {plan.reason}",(18,28),(0,255,0) if plan.ok else (0,0,255),0.58,2)
    draw_text(out,f"gaps L={plan.gaps.get('LEFT',0):.0f} R={plan.gaps.get('RIGHT',0):.0f} T={plan.gaps.get('TOP',0):.0f} B={plan.gaps.get('BOTTOM',0):.0f} | sides={plan.side_a}+{plan.side_b}",(18,52),(255,255,0),0.46,1)
    if plan.ok:
        draw_text(out,f"{plan.selected_arm.upper()} | move={plan.move_mm:.0f}mm inset={plan.grip_inset_mm:.0f}mm | target={plan.target_source}",(18,74),(0,255,255),0.48,1)
    return out


def _d58_query_gripper_t(arm: RoArm,args) -> Optional[float]:
    return _query_gripper_t_retry(arm,int(args.feedback_query_retries),float(args.feedback_query_gap_s))


def _d58_single_standby(arm: RoArm,arm_key: str,args):
    if arm_key=='arm1':
        arm.move(float(args.arm1_standby_x),float(args.arm1_standby_y),float(args.arm1_standby_z),t=float(args.arm1_standby_t),spd=float(args.standby_speed),stage='D58_STANDBY')
    else:
        arm.move(float(args.arm2_standby_x),float(args.arm2_standby_y),float(args.arm2_standby_z),t=float(args.arm2_standby_t),spd=float(args.standby_speed),stage='D58_STANDBY')
    time.sleep(float(args.standby_wait_s))


def execute_d58_plan(plan: D58Plan,config: Dict[str,Any],args) -> bool:
    if not plan.ok or plan.selected_arm not in ('arm1','arm2') or plan.grip_board is None or plan.end_board is None:
        print(f"[D58-2-EXEC] blocked: {plan.reason}"); return False
    if not bool(args.send):
        print('[D58-2-DRY] --send not set; preview only'); return True
    key=str(plan.selected_arm); label='ARM1' if key=='arm1' else 'ARM2'; port=args.arm1_port if key=='arm1' else args.arm2_port
    arm=RoArm(port,label)
    try:
        gp=plan.grip_board; ep=plan.end_board
        gx,gy=board_to_arm_xy(config,key,float(gp[0]),float(gp[1])); ex,ey=board_to_arm_xy(config,key,float(ep[0]),float(ep[1]))
        cz=contact_z(config,key,args); hz=hover_z(config,key,args); tt=tool_t(config,key)
        mid_ratio=float(np.clip(args.d58_drag_mid_ratio,0.20,0.80))
        mb=gp+(ep-gp)*mid_ratio
        mx,my=board_to_arm_xy(config,key,float(mb[0]),float(mb[1]))
        clear_z=min(float(hz),float(cz)+float(args.d58_post_drag_clear_mm))
        print(f"[D58-2-SEQUENCE] {label} OPEN -> HOVER -> CONTACT OPEN -> CLOSE/HOLD -> LOW-Z MID DRAG -> LOW-Z SWEEP({plan.move_mm:.0f}) -> POST-DRAG CLEAR -> OPEN -> VERTICAL RETRACT")
        print(f"[D58-2-DRAG-Z] contact={cz:.1f} mid={cz:.1f} sweep={cz:.1f} prePullLift=0.0mm clearAfter={clear_z-cz:.1f}mm")
        arm.grip(float(args.grip_open),'D58_OPEN'); time.sleep(float(args.pre_open_wait_s))
        arm.move(gx,gy,hz,t=tt,spd=float(args.free_speed),stage='D58_HOVER_OPEN'); time.sleep(float(args.hover_wait_s))
        arm.grip(float(args.grip_open),'D58_OPEN_BEFORE_DESCENT'); time.sleep(float(args.open_before_descent_wait_s))
        arm.move(gx,gy,cz,t=tt,spd=float(args.near_speed),stage='D58_CONTACT_OPEN'); time.sleep(float(args.contact_move_wait_s))
        arm.grip(float(args.grip_close),'D58_LIMITED_CLOSE'); time.sleep(float(args.close_limited_wait_s))
        for i in range(max(1,int(args.close_repeat))):
            arm.grip(float(args.grip_hold),f'D58_HOLD_{i+1}')
            if i+1<int(args.close_repeat): time.sleep(float(args.close_repeat_gap_s))
        time.sleep(float(args.close_final_hold_s))
        # T105 is unreliable on the current serial setup and can add a visible
        # stall before every drag. HOLD is already reasserted above, so continue
        # after the deterministic close-settle barrier without T105.
        print(f"[D58-3-GRIP-CLOSE] {label} T105_SKIPPED timedHoldComplete=True")
        # Keep the grasped cloth at the same contact Z through MID and SWEEP.
        # Do not lift before pulling; clear vertically only after the drag completes.
        arm.move(mx,my,cz,t=tt,spd=float(args.d58_translate_speed),stage='D58_LOW_Z_MID_CLOSED'); time.sleep(float(args.d58_mid_wait_s))
        arm.move(ex,ey,cz,t=tt,spd=float(args.d58_translate_speed),stage='D58_LOW_Z_SWEEP_CLOSED'); time.sleep(float(args.d58_translate_wait_s))
        arm.move(ex,ey,clear_z,t=tt,spd=float(args.d58_post_drag_clear_speed),stage='D58_POST_DRAG_CLEAR_CLOSED'); time.sleep(float(args.d58_post_drag_clear_wait_s))
        arm.grip(float(args.release_open),'D58_RELEASE_AFTER_CLEAR'); time.sleep(float(args.release_wait_s))
        arm.move(ex,ey,hz,t=tt,spd=float(args.vertical_retract_speed),stage='D58_VERTICAL_RETRACT_OPEN'); time.sleep(float(args.vertical_retract_wait_s)); _d58_single_standby(arm,key,args)
        print('[D58-2-EXEC-DONE] one circumcenter position correction completed; press I for FRESH REJUDGE')
        return True
    except Exception as exc:
        print(f"[D58-2-EXEC-ERROR] {exc!r}")
        try: arm.grip(float(args.grip_open),'D58_ERROR_OPEN'); time.sleep(0.3); _d58_single_standby(arm,key,args)
        except Exception: pass
        return False
    finally:
        arm.close()


def parser58() -> argparse.ArgumentParser:
    p=parser_build()
    p.description='D58-3 bottom circumcenter position correction: long-garment scaling, full-board bounds, no T105 grip stall'
    p.set_defaults(save_overlay='d58_3_plan.jpg')
    p.add_argument('--d58-gap-percentile',type=float,default=2.0)
    p.add_argument('--d58-min-vacancy-gap-mm',type=float,default=30.0)
    p.add_argument('--d58-pose-center-weight',type=float,default=0.25)
    p.add_argument('--d58-contour-edge-band-mm',type=float,default=95.0)
    p.add_argument('--d58-contour-equal-tolerance-mm',type=float,default=12.0)
    p.add_argument('--d58-min-triangle-area-mm2',type=float,default=2500.0)
    p.add_argument('--d58-max-circumradius-mm',type=float,default=430.0)
    p.add_argument('--d58-circumcenter-safe-margin-mm',type=float,default=0.0)
    p.add_argument('--d58-done-center-error-mm',type=float,default=35.0)
    p.add_argument('--d58-center-target-gain',type=float,default=1.00)
    p.add_argument('--d58-min-move-mm',type=float,default=30.0)
    p.add_argument('--d58-max-move-mm',type=float,default=105.0)
    p.add_argument('--d58-long-garment-span-mm',type=float,default=480.0)
    p.add_argument('--d58-long-garment-max-move-mm',type=float,default=165.0)
    p.add_argument('--d58-contact-edge-band-mm',type=float,default=90.0)
    p.add_argument('--d58-contact-inset-mm',type=float,default=40.0)
    p.add_argument('--d58-contact-max-candidates',type=int,default=180)
    p.add_argument('--d58-min-inside-px',type=float,default=3.0)
    p.add_argument('--d58-local-mask-radius-px',type=int,default=13)
    p.add_argument('--d58-min-local-mask-ratio',type=float,default=0.62)
    p.add_argument('--d58-horizontal-arm-deadband',type=float,default=0.22)
    p.add_argument('--d58-board-margin-mm',type=float,default=0.0)
    p.add_argument('--d58-roarm-xy-radius-max-mm',type=float,default=395.0)
    # D58-2 motion: D55-style low-Z drag. No pre-pull lift.
    p.add_argument('--d58-drag-mid-ratio',type=float,default=0.55)
    p.add_argument('--d58-translate-speed',type=float,default=0.42)
    p.add_argument('--d58-mid-wait-s',type=float,default=0.45)
    p.add_argument('--d58-translate-wait-s',type=float,default=0.75)
    p.add_argument('--d58-post-drag-clear-mm',type=float,default=65.0)
    p.add_argument('--d58-post-drag-clear-speed',type=float,default=0.55)
    p.add_argument('--d58-post-drag-clear-wait-s',type=float,default=0.55)
    return p


def main() -> int:
    args=parser58().parse_args()
    if YOLO is None: raise RuntimeError('ultralytics import failed')
    args.config=resolve_file(args.config); args.hfile=resolve_file(args.hfile); args.camera_calibration=resolve_file(args.camera_calibration); args.camera_controls_json=resolve_file(args.camera_controls_json)
    config=load_json(args.config)
    if not config: raise RuntimeError(f'config not found/invalid: {args.config}')
    if not args.camera_undistort: raise RuntimeError('D58-2 requires corrected ELP geometry; --no-camera-undistort is blocked')
    if CameraUndistorter is None: raise RuntimeError(f'camera_undistort.py unavailable: {CAMERA_UNDISTORT_IMPORT_ERROR!r}')
    if not os.path.isfile(args.camera_calibration): raise RuntimeError(f'camera calibration not found: {args.camera_calibration}')
    cap=open_camera(args.camera,args.width,args.height,args.backend)
    if not cap.isOpened(): raise RuntimeError('camera open failed')
    if args.camera_controls_enable: apply_camera_controls(args.camera_controls_json,args.camera_device,args.camera_controls_strict)
    for _ in range(max(0,int(args.camera_controls_stabilization_frames))): cap.read()
    ok_probe,probe=cap.read()
    if not ok_probe or probe is None: raise RuntimeError('camera opened but initial frame read failed')
    actual_size=(int(probe.shape[1]),int(probe.shape[0]))
    if actual_size!=(int(args.width),int(args.height)): raise RuntimeError(f'camera size {actual_size} != requested {(args.width,args.height)}')
    und=CameraUndistorter(args.camera_calibration,alpha=float(args.camera_undistort_alpha),strict_size=True); und.prepare(actual_size); cam_meta=und.info().to_metadata()
    print(f'[D58-2-UNDISTORT-MODULE] {CAMERA_UNDISTORT_MODULE_PATH}'); print('[D58-2-UNDISTORT] '+und.status_line())
    probe_corrected=und.correct(probe); mad=float(np.mean(cv2.absdiff(probe,probe_corrected))); md=int(np.max(cv2.absdiff(probe,probe_corrected)))
    print(f'[D58-2-UNDISTORT-CHECK] raw->corrected meanAbsDiff={mad:.3f} maxDiff={md}; DISPLAY/ARUCO/SEG/POSE/PLAN=corrected only')
    H,_=load_corrected_h(args.hfile,cam_meta)
    if H is None: print('[D58-2-H] corrected cache unavailable/mismatched; press L')
    seg_path=resolve_file(args.seg_model); pose_path=resolve_file(args.pose_model)
    print(f'[D58-2-MODEL] SEG task=segment path={seg_path}'); seg=YOLO(seg_path,task='segment')
    print(f'[D58-2-MODEL] POSE task=pose path={pose_path}'); pose_model=YOLO(pose_path,task='pose')
    locked:Optional[D58Plan]=None; terminal=TerminalKeyReader(); terminal.start(); window='D58-2 CIRCUMCENTER POSITION'; action_count=0
    print('\n[D58-2 KEYS] I=infer+lock | Enter=execute one correction | L=relock corrected H | R=discard | Q/ESC=quit')
    print('[D58-3 POLICY] corrected camera | circumcenter target | full board bounds (0mm inner exclusion) | long-garment adaptive move | same-Z low drag | no T105 grip stall')
    if args.send: print('[D58-2] PHYSICAL SEND=ON')
    else: print('[D58-2] PREVIEW ONLY (--send to move robot)')
    try:
        while True:
            ok,raw=cap.read()
            if not ok or raw is None: time.sleep(0.02); continue
            frame=und.correct(raw)
            canvas=frame.copy() if locked is None or locked.overlay is None else locked.overlay.copy()
            draw_text(canvas,'ELP UNDISTORT: ON | CORRECTED H ONLY',(18,28),(0,255,255),0.50,2)
            if locked is None: draw_text(canvas,'D58-2 | press I to calculate circumcenter position correction',(18,52),(0,255,0),0.52,2)
            else: draw_text(canvas,f"I-LOCK | Enter={'EXECUTE' if locked.ok else 'BLOCKED'} | {locked.reason}",(18,frame.shape[0]-18),(0,255,255),0.48,2)
            txt=f'ACTIONS: {action_count}'; (tw,th),_=cv2.getTextSize(txt,cv2.FONT_HERSHEY_SIMPLEX,0.55,2); cv2.putText(canvas,txt,(frame.shape[1]-tw-18,28),cv2.FONT_HERSHEY_SIMPLEX,0.55,(0,0,0),4); cv2.putText(canvas,txt,(frame.shape[1]-tw-18,28),cv2.FONT_HERSHEY_SIMPLEX,0.55,(0,255,0),2)
            if not args.no_window: cv2.imshow(window,canvas); gui=cv2.waitKey(1)&0xFF
            else: gui=255
            tk=terminal.read_key(); key=tk if tk!=255 else gui
            if tk!=255:
                lab='ENTER' if tk in (10,13) else ('ESC' if tk==27 else chr(tk).upper() if 0<=tk<256 else str(tk)); print(f'[D58-2-KEY] terminal={lab}')
            if key in (ord('q'),27): break
            if key==ord('r'): locked=None; print('[D58-2] plan discarded'); continue
            if key==ord('l'):
                hn=aruco_lock(frame,config)
                if hn is not None: H=hn; save_corrected_h_preserve_bundle(args.hfile,H,cam_meta); locked=None; print('[D58-2-H] corrected-frame H relocked')
                continue
            if key==ord('i'):
                if H is None: print('[D58-2-I] H unavailable; press L'); continue
                t0=time.time(); m,ms=infer_mask(seg,frame,H,config,args); print(f'[D58-2-MASK] {ms}')
                if m is None: locked=None; continue
                po,ps=infer_pose(pose_model,frame,H,m,args); print(f'[D58-2-POSE] {ps}')
                if po is None: print('[D58-2-POSE-WARN] pose unavailable -> MASK CENTER fallback is allowed')
                locked=build_d58_plan(frame.copy(),m,po,H,config,args)
                if locked.overlay is None: locked.overlay=make_d58_overlay(frame.copy(),m,locked,H,config)
                if locked.overlay is not None and args.save_overlay:
                    try: cv2.imwrite(args.save_overlay,locked.overlay)
                    except Exception: pass
                print(f'[D58-2-PLAN] ok={locked.ok} reason={locked.reason} arm={locked.selected_arm} move={locked.move_mm:.1f}mm target={locked.target_source} time={(time.time()-t0)*1000:.0f}ms')
                continue
            if key in (10,13):
                if locked is None: print('[D58-2-ENTER] no I-locked plan'); continue
                if not locked.ok: print(f'[D58-2-ENTER] blocked: {locked.reason}'); continue
                plan=locked; locked=None
                if args.send: action_count+=1
                execute_d58_plan(plan,config,args)
                print('[D58-2] round finished -> press I for FRESH REJUDGE')
                continue
    finally:
        terminal.stop(); cap.release()
        if not args.no_window: cv2.destroyAllWindows()
    return 0


if __name__=="__main__":
    raise SystemExit(main())
