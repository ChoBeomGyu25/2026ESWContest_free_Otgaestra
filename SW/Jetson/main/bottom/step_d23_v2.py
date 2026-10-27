#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
step_d23_v2.py

Bottom/Pants stage-1~4 controller for Jetson Orin Nano Super + dual RoArm-M2-S.

Goal covered by this file:
  1) Detect pants and initialize state.
  2) Robot arm picks pants and places it on the folding board area.
  3) Align waist line near the lower reference line.
  4) Align crotch point to the vertical center slit.

This file is intentionally conservative:
  - --send defaults to OFF.
  - --dry-run defaults to ON.
  - --hover-only defaults to ON.
  - ENTER confirmation is required by default before any planned motion.
  - No simultaneous two-arm low-Z motion is used.
  - ARM2 is left workspace only, ARM1 is right workspace only.
  - Center dead-zone is blocked for low-Z contact/grip points.

Assumptions:
  - Pants are visible from the overhead camera in the ArUco board coordinate frame.
  - Existing dual_roarm_big_table_config.json and dual_roarm_affine_cache.json are available.
  - Pixel -> board-mm Homography is locked or loadable from dual_big_table_homography_cache.json.
  - The first version does not solve off-board basket pickup. For basket pickup, a separate
    basket coordinate calibration is required.

Recommended first checks:
  python3 step_b35_bottom_stage1_4.py --demo
  python3 -m py_compile step_b35_bottom_stage1_4.py
  python3 step_b35_bottom_stage1_4.py --help

Recommended live preview, no robot motion:
  python3 step_b35_bottom_stage1_4.py \
    --pose-model /workspace/project_train/aruco_test/dual/best.engine \
    --seg-model /workspace/project_train/aruco_test/dual/models/kfashion_yolo26s_seg3_e100_best.engine \
    --no-send --dry-run --hover-only

To allow hover-only robot preview after verifying calibration:
  python3 step_b35_bottom_stage1_4.py --send --dry-run --hover-only

To allow actual low-Z action, only after hover-only verification:
  python3 step_b35_bottom_stage1_4.py --send --no-dry-run --no-hover-only --enter-confirm
"""

import argparse
import json
import math
import os
import select
import sys
import time
import threading
import copy
import subprocess
import shutil
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Sequence, Tuple

# Docker + X11 stability. Keep before cv2 import.
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




STEP_D23_BUILD = "2026-07-16-d23-v2-center-outward-closed-press-v1"
TERMINAL_CONFIRM_ACTIVE = threading.Event()

# D23_v2 changes over D23:
#   1) Every moving contact is checked against a pants-center radial direction.
#      Any inward motion that would reduce distance from the garment center is blocked.
#   2) Moderate/irregular wrinkles and lower-leg asymmetry use a closed-gripper
#      center-to-outside press sweep instead of opening and gripping by default.
#   3) Grip-pull is reserved for clearly severe, long, dark and well-fitted LINE/CURVE
#      wrinkles; unavailable or inward grip plans fall back to closed press.
#   4) The legacy unconstrained fallback is disabled by default, so NO_ACTION is
#      preferred over a motion that can bunch a trouser leg inward.
#   5) D23 waistband veto, staged pose TTA and safe parallel high-Z motion remain.

BOTTOM_POSE_KPT_NAMES = [
    "waist_img_left",       # 0
    "waist_center",         # 1
    "waist_img_right",      # 2
    "crotch",               # 3
    "img_left_hem_outer",   # 4
    "img_left_hem_inner",   # 5
    "img_right_hem_inner",  # 6
    "img_right_hem_outer",  # 7
]
BOTTOM_POSE_KPT = {name: i for i, name in enumerate(BOTTOM_POSE_KPT_NAMES)}

# D2-style horizontal-flip semantic remap.
# After hflip, image-left/right keypoint meanings must be restored to the original frame.
HFLIP_REMAP = np.asarray([2, 1, 0, 3, 7, 6, 5, 4], dtype=np.int32)


DEFAULT_MARKER_BOARD_MM = {
    "0": [0.0, 0.0],
    "1": [0.0, -496.0],
    "2": [492.0, 5.0],
    "3": [494.0, -490.0],
}


class BottomStage(Enum):
    DETECT_INIT = auto()
    PICK_PLACE_TO_BOARD = auto()
    ALIGN_WAIST_REF = auto()
    ALIGN_CROTCH_SLIT = auto()
    DUAL_WRINKLE_STRETCH = auto()
    DONE = auto()
    ERROR = auto()


@dataclass
class BottomSafetyConfig:
    board_x_min: float = 0.0
    board_x_max: float = 494.0
    board_y_min: float = -496.0
    board_y_max: float = 5.0
    board_margin_mm: float = 18.0

    split_board_x: float = 247.0
    center_dead_half_width: float = 40.0
    private_workspace_margin_mm: float = 8.0

    hover_z: float = 180.0
    approach_z: float = 180.0
    grip_z: float = -115.0
    arm1_grip_z: Optional[float] = -125.0
    arm2_grip_z: Optional[float] = None
    lift_z: float = -85.0
    arm1_lift_z: Optional[float] = -85.0
    arm2_lift_z: Optional[float] = -85.0
    place_z: float = -105.0
    clear_z: float = 180.0
    min_safe_contact_z: float = -140.0

    tool_t_rad: float = 3.14
    speed: float = 0.12
    drag_speed: float = 0.35
    grip_open: float = 1.35
    grip_close: float = 3.05
    grip_hold: float = 3.15
    grip_down_wait: float = 0.8
    grip_open_wait: float = 1.0
    grip_contact_wait: float = 0.15
    grip_close_repeat: int = 1
    grip_close_wait: float = 0.45
    grip_close_settle: float = 0.45
    place_settle_wait: float = 0.8
    place_open_wait: float = 1.0

    imaging_standby: bool = True
    standby_after_motion: bool = True
    standby_gripper_angle: float = 1.57
    arm1_standby_roarm_x: float = -10.47886
    arm1_standby_roarm_y: float = 227.544386
    arm1_standby_roarm_z: float = 170.929303
    arm1_standby_roarm_t: float = 2.124563
    arm2_standby_roarm_x: float = 2.870034
    arm2_standby_roarm_y: float = -233.859636
    arm2_standby_roarm_z: float = 102.23829
    arm2_standby_roarm_t: float = 1.356039

    max_single_xy_move_mm: float = 140.0
    max_stage_retry: int = 3

    waist_ref_y: float = -390.0
    center_slit_x: float = 247.0
    waist_y_tolerance_mm: float = 25.0
    crotch_x_tolerance_mm: float = 25.0

    pose_kpt_conf: float = 0.20
    pose_center_mask_max_dist_mm: float = 120.0
    min_pants_axis_len_mm: float = 80.0
    min_mask_area_px: float = 1200.0

    send: bool = False
    dry_run: bool = True
    hover_only: bool = True
    enter_confirm: bool = True


@dataclass
class MotionPoint:
    arm_key: str
    board_xy: Tuple[float, float]
    roarm_xyz: Optional[Tuple[float, float, float]] = None
    label: str = ""


@dataclass
class MotionPlan:
    name: str
    action: str
    stage: BottomStage
    ok: bool
    reason: str
    arm_key: Optional[str] = None
    grip_board_xy: Optional[Tuple[float, float]] = None
    target_board_xy: Optional[Tuple[float, float]] = None
    sequence: List[MotionPoint] = field(default_factory=list)
    safety_warnings: List[str] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BottomPoseBoard:
    keypoints_board: Dict[str, Tuple[float, float]]
    keypoints_px: Dict[str, Tuple[float, float]]
    keypoint_conf: Dict[str, float]
    waist_left: np.ndarray
    waist_center: np.ndarray
    waist_right: np.ndarray
    crotch: np.ndarray
    left_hem_center: np.ndarray
    right_hem_center: np.ndarray
    lower_center: np.ndarray
    pose_center: np.ndarray
    waist_angle_deg: float
    pants_axis_angle_deg: float
    pants_axis_len_mm: float
    waist_width_mm: float
    hem_gap_mm: float
    valid: bool
    reason: str
    tta_angle_deg: float = 0.0
    tta_flip_mode: str = "none"
    tta_score: float = 0.0
    tta_state: str = "SINGLE"
    tta_reason: str = ""
    tta_tested_count: int = 1


@dataclass
class BottomMaskBoard:
    mask_u8: np.ndarray
    contour: np.ndarray
    area_px: float
    center_px: np.ndarray
    center_board: np.ndarray
    class_name: str
    confidence: float
    solidity: float


@dataclass
class BottomObservation:
    pose: Optional[BottomPoseBoard]
    mask: Optional[BottomMaskBoard]
    fused_center_board: Optional[np.ndarray]
    valid: bool
    reason: str
    frame: Optional[np.ndarray] = None


@dataclass
class WrinkleHeatmapPlan:
    heatmap: np.ndarray
    inner_mask: np.ndarray
    binary: np.ndarray
    candidates: List[Dict[str, Any]]
    threshold: float
    wrinkle_ratio: float
    status: str
    summary: str
    created_at: float = field(default_factory=time.time)


@dataclass
class DualWrinkleStretchPlan:
    """D16 shape-aware wrinkle policy plan."""
    ok: bool
    reason: str
    action: str = "NO_ACTION"
    candidate_index: int = -1
    center_px: Optional[Tuple[float, float]] = None
    center_board: Optional[Tuple[float, float]] = None
    major_axis_px: Optional[Tuple[float, float]] = None
    normal_axis_board: Optional[Tuple[float, float]] = None
    arm_points: Dict[str, Dict[str, Tuple[float, ...]]] = field(default_factory=dict)
    metrics: Dict[str, Any] = field(default_factory=dict)
    safety_warnings: List[str] = field(default_factory=list)


# B34-style feedback memory: nearby bad directions are reversed/penalized on the next X action.
B34_PRESS_HISTORY: List[Dict[str, Any]] = []

# D23 staged-TTA hint.  A stable garment usually keeps the same successful
# transform between cycles, so testing this first removes most redundant TensorRT
# pose calls while preserving the full fallback search for difficult orientations.
D23_TTA_HINT_LOCK = threading.Lock()
D23_TTA_HINT: Dict[str, Any] = {
    "valid": False,
    "angle": 0.0,
    "flip_mode": "none",
    "success_count": 0,
}


class RoArmSerial:
    """Minimal RoArm serial wrapper compatible with Waveshare JSON serial commands."""

    def __init__(self, port: str, baudrate: int = 115200, label: str = "arm"):
        if serial is None:
            raise RuntimeError("pyserial is not installed. Run: pip3 install pyserial")
        self.port = port
        self.label = label
        self.ser = serial.Serial(port, baudrate=baudrate, timeout=0.15)
        try:
            self.ser.setDTR(False)
            self.ser.setRTS(False)
        except Exception:
            pass
        time.sleep(0.5)

    def send(self, cmd: Dict[str, Any], delay: float = 0.18) -> None:
        line = json.dumps(cmd, separators=(",", ":"))
        print(f"[{self.label}] SEND {line}")
        self.ser.write((line + "\n").encode("utf-8"))
        self.ser.flush()
        time.sleep(delay)

    def move_goal(self, x: float, y: float, z: float, t: float, spd: float, move_command: int = 104) -> None:
        self.send({"T": int(move_command), "x": float(x), "y": float(y), "z": float(z), "t": float(t), "spd": float(spd)})

    def set_gripper(self, angle_rad: float, delay: float = 0.25) -> None:
        self.send({"T": 106, "cmd": float(angle_rad), "spd": 0, "acc": 0}, delay=delay)

    def close(self) -> None:
        try:
            self.ser.close()
        except Exception:
            pass


# -----------------------------------------------------------------------------
# General geometry / config / Homography helpers
# -----------------------------------------------------------------------------

def load_json_if_exists(path: str) -> Optional[Dict[str, Any]]:
    if not path or not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_homography(path: str) -> Optional[np.ndarray]:
    data = load_json_if_exists(path)
    if not data or "H" not in data:
        return None
    return np.asarray(data["H"], dtype=np.float32)


def save_homography(path: str, H: np.ndarray) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"H": np.asarray(H, dtype=float).tolist()}, f, indent=2)
    print(f"[H] saved: {path}")


def pixel_to_board(H: np.ndarray, u: float, v: float) -> Tuple[float, float]:
    p = np.array([[[float(u), float(v)]]], dtype=np.float32)
    out = cv2.perspectiveTransform(p, H)
    return float(out[0, 0, 0]), float(out[0, 0, 1])


def board_to_pixel(H: np.ndarray, bx: float, by: float) -> Optional[Tuple[float, float]]:
    try:
        Hinv = np.linalg.inv(H)
        p = np.array([[[float(bx), float(by)]]], dtype=np.float32)
        out = cv2.perspectiveTransform(p, Hinv)
        return float(out[0, 0, 0]), float(out[0, 0, 1])
    except Exception:
        return None


def get_dictionary(name: str = "DICT_4X4_50"):
    if not hasattr(cv2.aruco, name):
        raise ValueError(f"Unknown ArUco dictionary: {name}")
    return cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, name))


def make_aruco_detector(dictionary):
    if hasattr(cv2.aruco, "ArucoDetector"):
        params = cv2.aruco.DetectorParameters()
        return cv2.aruco.ArucoDetector(dictionary, params)
    params = cv2.aruco.DetectorParameters_create()
    return dictionary, params


def detect_markers(frame: np.ndarray, detector_obj):
    if isinstance(detector_obj, tuple):
        dictionary, params = detector_obj
        return cv2.aruco.detectMarkers(frame, dictionary, parameters=params)
    return detector_obj.detectMarkers(frame)


def compute_homography(corners, ids, marker_board_mm: Dict[str, List[float]], required_ids: Sequence[int]):
    if ids is None:
        return None, {}
    centers = {}
    for c, marker_id in zip(corners, ids.flatten()):
        pts = c.reshape(4, 2)
        centers[int(marker_id)] = pts.mean(axis=0)
    if not all(i in centers for i in required_ids):
        return None, centers
    img_pts = np.array([centers[i] for i in required_ids], dtype=np.float32)
    board_pts = np.array([marker_board_mm[str(i)] for i in required_ids], dtype=np.float32)
    H, _ = cv2.findHomography(img_pts, board_pts)
    return H, centers


def normalize_angle_deg(angle: float) -> float:
    return float((float(angle) + 180.0) % 360.0 - 180.0)


def unit_vec(v: np.ndarray, fallback=(1.0, 0.0)) -> np.ndarray:
    arr = np.asarray(v, dtype=np.float32).reshape(2)
    n = float(np.linalg.norm(arr))
    if n < 1e-6:
        return np.asarray(fallback, dtype=np.float32)
    return arr / n


def board_bounds_from_config(config: Optional[Dict[str, Any]]) -> Tuple[float, float, float, float]:
    marker = DEFAULT_MARKER_BOARD_MM
    if config and isinstance(config.get("aruco"), dict):
        marker = config.get("aruco", {}).get("marker_board_mm", marker)
    pts = np.asarray(list(marker.values()), dtype=np.float32)
    return float(np.min(pts[:, 0])), float(np.max(pts[:, 0])), float(np.min(pts[:, 1])), float(np.max(pts[:, 1]))


def load_affine_cache_into_config(config: Dict[str, Any], cache_path: str) -> Dict[str, Any]:
    cache = load_json_if_exists(cache_path)
    if not cache:
        print(f"[AFFINE] no cache loaded: {cache_path}")
        return config
    # Accept either full config-like cache or only dual_roarm entry.
    if "dual_roarm" in cache:
        dual = cache["dual_roarm"]
    else:
        dual = cache
    config.setdefault("dual_roarm", {})
    for arm_key in ("arm1", "arm2"):
        if arm_key in dual:
            config["dual_roarm"].setdefault(arm_key, {})
            for k, v in dual[arm_key].items():
                if k in ("board_to_roarm_affine_2x3", "roarm_xy_offset", "calib_roarm_points", "calib_roarm_z"):
                    config["dual_roarm"][arm_key][k] = v
    print(f"[AFFINE] loaded: {cache_path}")
    return config


def board_to_arm_xy(config: Dict[str, Any], arm_key: str, bx: float, by: float) -> Tuple[float, float]:
    arm_cfg = config.get("dual_roarm", {}).get(arm_key, {})
    M_raw = arm_cfg.get("board_to_roarm_affine_2x3")
    if M_raw is None:
        raise RuntimeError(f"{arm_key} affine is not calibrated. Need board_to_roarm_affine_2x3.")
    M = np.asarray(M_raw, dtype=float).reshape(2, 3)
    out = M @ np.asarray([float(bx), float(by), 1.0], dtype=float)
    off = arm_cfg.get("roarm_xy_offset", [0.0, 0.0])
    ox = float(off[0]) if isinstance(off, (list, tuple)) and len(off) >= 1 else 0.0
    oy = float(off[1]) if isinstance(off, (list, tuple)) and len(off) >= 2 else 0.0
    return float(out[0] + ox), float(out[1] + oy)


def select_arm_for_board_x(cfg: BottomSafetyConfig, bx: float) -> Optional[str]:
    left_max = cfg.split_board_x - cfg.center_dead_half_width - cfg.private_workspace_margin_mm
    right_min = cfg.split_board_x + cfg.center_dead_half_width + cfg.private_workspace_margin_mm
    if bx <= left_max:
        return "arm2"
    if bx >= right_min:
        return "arm1"
    return None


def point_in_board(cfg: BottomSafetyConfig, p: Sequence[float], margin: Optional[float] = None) -> bool:
    m = cfg.board_margin_mm if margin is None else float(margin)
    x, y = float(p[0]), float(p[1])
    return (cfg.board_x_min + m <= x <= cfg.board_x_max - m and
            cfg.board_y_min + m <= y <= cfg.board_y_max - m)


def clamp_board_point(cfg: BottomSafetyConfig, p: Sequence[float], margin: Optional[float] = None) -> np.ndarray:
    m = cfg.board_margin_mm if margin is None else float(margin)
    x = float(np.clip(float(p[0]), cfg.board_x_min + m, cfg.board_x_max - m))
    y = float(np.clip(float(p[1]), cfg.board_y_min + m, cfg.board_y_max - m))
    return np.asarray([x, y], dtype=np.float32)


def arm_contact_z(cfg: BottomSafetyConfig, arm_key: str) -> float:
    if arm_key == "arm1" and cfg.arm1_grip_z is not None:
        return float(cfg.arm1_grip_z)
    if arm_key == "arm2" and cfg.arm2_grip_z is not None:
        return float(cfg.arm2_grip_z)
    return float(cfg.grip_z)


def arm_lift_z(cfg: BottomSafetyConfig, arm_key: str) -> float:
    if arm_key == "arm1" and cfg.arm1_lift_z is not None:
        return float(cfg.arm1_lift_z)
    if arm_key == "arm2" and cfg.arm2_lift_z is not None:
        return float(cfg.arm2_lift_z)
    return float(cfg.lift_z)


# -----------------------------------------------------------------------------
# Perception: segmentation + bottom pose
# -----------------------------------------------------------------------------

def result_mask_to_u8(result, index: int, image_shape) -> Optional[np.ndarray]:
    if getattr(result, "masks", None) is None or result.masks is None:
        return None
    h, w = image_shape[:2]
    try:
        data = result.masks.data[index].detach().cpu().numpy()
        if data.shape[:2] != (h, w):
            data = cv2.resize(data, (w, h), interpolation=cv2.INTER_LINEAR)
        return ((data > 0.5).astype(np.uint8) * 255)
    except Exception:
        return None


def infer_bottoms_mask(seg_model, frame: np.ndarray, H: Optional[np.ndarray], imgsz: int, conf: float,
                       target_class_names: Sequence[str] = ("bottoms",)) -> Tuple[Optional[BottomMaskBoard], str]:
    if seg_model is None:
        return None, "segmentation model is None"
    if H is None:
        return None, "Homography not locked"
    try:
        result = seg_model.predict(source=frame, imgsz=int(imgsz), conf=float(conf), retina_masks=True, verbose=False)[0]
    except Exception as e:
        return None, f"seg predict error: {repr(e)}"

    if getattr(result, "boxes", None) is None or result.boxes is None or len(result.boxes) == 0:
        return None, "no segmentation boxes"

    names = getattr(result, "names", {}) or {}
    best = None
    best_reason = "no bottoms class mask"
    for i in range(len(result.boxes)):
        cls_id = int(result.boxes.cls[i].item())
        cls_name = str(names.get(cls_id, cls_id))
        if cls_name not in set(target_class_names):
            continue
        c = float(result.boxes.conf[i].item())
        mask = result_mask_to_u8(result, i, frame.shape)
        if mask is None:
            best_reason = f"{cls_name} has no mask"
            continue
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            best_reason = f"{cls_name} mask has no contour"
            continue
        contour = max(contours, key=cv2.contourArea)
        area = float(cv2.contourArea(contour))
        if area <= 1.0:
            best_reason = f"{cls_name} mask area too small"
            continue
        M = cv2.moments(contour)
        if abs(M["m00"]) > 1e-6:
            center_px = np.asarray([M["m10"] / M["m00"], M["m01"] / M["m00"]], dtype=np.float32)
        else:
            center_px = contour.reshape(-1, 2).astype(np.float32).mean(axis=0)
        bx, by = pixel_to_board(H, float(center_px[0]), float(center_px[1]))
        hull = cv2.convexHull(contour)
        hull_area = float(cv2.contourArea(hull))
        solidity = area / hull_area if hull_area > 1.0 else 0.0
        item = BottomMaskBoard(
            mask_u8=mask,
            contour=contour,
            area_px=area,
            center_px=center_px,
            center_board=np.asarray([bx, by], dtype=np.float32),
            class_name=cls_name,
            confidence=c,
            solidity=float(solidity),
        )
        if best is None or item.area_px > best.area_px:
            best = item
    if best is None:
        return None, best_reason
    return best, "OK"


def read_pose_keypoints(result, frame_shape, kpt_conf: float) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], str]:
    if getattr(result, "keypoints", None) is None or result.keypoints is None:
        return None, None, "no keypoints"
    try:
        kxy_all = result.keypoints.xy.detach().cpu().numpy()
    except Exception:
        return None, None, "cannot read keypoints.xy"
    if kxy_all is None or len(kxy_all) == 0:
        return None, None, "no pose instances"

    kconf_all = None
    try:
        if getattr(result.keypoints, "conf", None) is not None:
            kconf_all = result.keypoints.conf.detach().cpu().numpy()
    except Exception:
        kconf_all = None

    det_idx = 0
    try:
        if result.boxes is not None and len(result.boxes) > 0:
            confs = result.boxes.conf.detach().cpu().numpy().astype(float)
            det_idx = int(np.argmax(confs[:len(kxy_all)]))
    except Exception:
        det_idx = 0
    kxy = np.asarray(kxy_all[det_idx], dtype=np.float32)
    kcf = None if kconf_all is None else np.asarray(kconf_all[det_idx], dtype=np.float32)

    if kxy.shape[0] < 8:
        return None, None, f"expected 8 keypoints, got {kxy.shape[0]}"
    if kcf is None:
        kcf = np.ones((kxy.shape[0],), dtype=np.float32)
    return kxy, kcf, "OK"


def kpt_valid_xy_conf(kxy: np.ndarray, kcf: np.ndarray, idx: int, image_shape, min_conf: float) -> bool:
    h, w = image_shape[:2]
    if idx >= len(kxy):
        return False
    x, y = float(kxy[idx, 0]), float(kxy[idx, 1])
    c = float(kcf[idx]) if idx < len(kcf) else 0.0
    return bool(np.isfinite(x) and np.isfinite(y) and c >= min_conf and 1.0 < x < w - 1.0 and 1.0 < y < h - 1.0)



# -----------------------------------------------------------------------------
# D2-style rotation + horizontal-flip TTA for bottom pose
# -----------------------------------------------------------------------------

def parse_pose_tta_angles(text: str) -> List[float]:
    values: List[float] = []
    for token in str(text or '').split(','):
        token = token.strip()
        if not token:
            continue
        values.append(float(token))
    if not values:
        values = [0.0]
    result: List[float] = []
    seen = set()
    for value in values:
        key = round(float(value), 4)
        if key not in seen:
            seen.add(key)
            result.append(float(value))
    return result


def parse_pose_tta_flip_modes(text: str) -> List[str]:
    result: List[str] = []
    for token in str(text or '').split(','):
        token = token.strip().lower()
        if token in ('none', 'h') and token not in result:
            result.append(token)
    if not result:
        result = ['none']
    if 'none' not in result:
        result.insert(0, 'none')
    return result


def rotate_image_and_inverse(frame: np.ndarray, angle_deg: float) -> Tuple[np.ndarray, np.ndarray]:
    h, w = frame.shape[:2]
    cx, cy = w / 2.0, h / 2.0
    M = cv2.getRotationMatrix2D((cx, cy), float(angle_deg), 1.0)
    cos_v = abs(float(M[0, 0]))
    sin_v = abs(float(M[0, 1]))
    new_w = int(round(h * sin_v + w * cos_v))
    new_h = int(round(h * cos_v + w * sin_v))
    M[0, 2] += new_w / 2.0 - cx
    M[1, 2] += new_h / 2.0 - cy
    rotated = cv2.warpAffine(
        frame, M, (new_w, new_h), flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0),
    )
    M3 = np.vstack([M, [0.0, 0.0, 1.0]]).astype(np.float32)
    return rotated, np.linalg.inv(M3).astype(np.float32)


def horizontal_flip_and_inverse(frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    _h, w = frame.shape[:2]
    flipped = frame[:, ::-1].copy()
    inverse = np.asarray([
        [-1.0, 0.0, float(w - 1)],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ], dtype=np.float32)
    return flipped, inverse


def inverse_transform_keypoints(kxy: np.ndarray, inverse_3x3: np.ndarray) -> np.ndarray:
    pts = np.asarray(kxy, dtype=np.float32)
    homog = np.concatenate([pts, np.ones((len(pts), 1), dtype=np.float32)], axis=1)
    restored = (np.asarray(inverse_3x3, dtype=np.float32) @ homog.T).T
    return restored[:, :2].astype(np.float32)


def remap_hflip_bottom_keypoints(kxy: np.ndarray, kcf: Optional[np.ndarray]) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    restored_xy = np.asarray(kxy, dtype=np.float32)[HFLIP_REMAP].copy()
    restored_cf = None if kcf is None else np.asarray(kcf, dtype=np.float32)[HFLIP_REMAP].copy()
    return restored_xy, restored_cf


def _tta_valid(kxy: np.ndarray, kcf: np.ndarray, idx: int, image_shape, threshold: float) -> bool:
    return kpt_valid_xy_conf(kxy, kcf, idx, image_shape, threshold)


def _tta_mean(kxy: np.ndarray, kcf: np.ndarray, indices: Sequence[int], image_shape, threshold: float) -> Optional[np.ndarray]:
    points = [np.asarray(kxy[i], dtype=np.float32) for i in indices if _tta_valid(kxy, kcf, i, image_shape, threshold)]
    if not points:
        return None
    return np.mean(np.stack(points, axis=0), axis=0)


def _safe_unit(v: np.ndarray) -> Optional[np.ndarray]:
    a = np.asarray(v, dtype=np.float32)
    n = float(np.linalg.norm(a))
    if n < 1e-6:
        return None
    return a / n


def _angle_between_deg(v1: np.ndarray, v2: np.ndarray) -> Optional[float]:
    u1 = _safe_unit(v1)
    u2 = _safe_unit(v2)
    if u1 is None or u2 is None:
        return None
    return float(np.degrees(np.arccos(np.clip(float(np.dot(u1, u2)), -1.0, 1.0))))


def score_bottom_pose_tta(kxy: np.ndarray, kcf: np.ndarray, frame_shape, mask: Optional[BottomMaskBoard], args) -> Dict[str, Any]:
    threshold = float(args.pose_kpt_conf)
    valid = np.asarray([
        _tta_valid(kxy, kcf, i, frame_shape, threshold)
        for i in range(min(8, len(kxy)))
    ], dtype=bool)
    if len(valid) < 8:
        valid = np.pad(valid, (0, 8 - len(valid)), constant_values=False)

    visible_count = int(valid.sum())
    waist_count = int(valid[[0, 1, 2]].sum())
    hem_count = int(valid[[4, 5, 6, 7]].sum())
    crotch_visible = bool(valid[3])
    visible_conf = [float(kcf[i]) for i in range(min(8, len(kcf))) if valid[i]]
    mean_conf = float(np.mean(visible_conf)) if visible_conf else 0.0

    geometry = 0.0
    penalty = 0.0
    reasons: List[str] = []
    waist_width = None
    hem_gap = None
    leg_open = None
    waist_axis_dot = None

    wl = np.asarray(kxy[0], dtype=np.float32) if valid[0] else None
    wc = np.asarray(kxy[1], dtype=np.float32) if valid[1] else None
    wr = np.asarray(kxy[2], dtype=np.float32) if valid[2] else None
    cr = np.asarray(kxy[3], dtype=np.float32) if valid[3] else None
    lhc = _tta_mean(kxy, kcf, (4, 5), frame_shape, threshold)
    rhc = _tta_mean(kxy, kcf, (6, 7), frame_shape, threshold)
    lower = 0.5 * (lhc + rhc) if lhc is not None and rhc is not None else (lhc if lhc is not None else rhc)

    if wl is not None and wc is not None and wr is not None:
        waist_vec = wr - wl
        waist_width = float(np.linalg.norm(waist_vec))
        waist_mid = 0.5 * (wl + wr)
        center_offset = float(np.linalg.norm(wc - waist_mid))
        if waist_width >= float(args.tta_min_waist_width_px):
            geometry += 1.0
        else:
            penalty += 2.5
            reasons.append('waist width small')
        max_offset = max(25.0, waist_width * float(args.tta_max_waist_center_offset_ratio))
        if center_offset <= max_offset:
            geometry += 1.5
        else:
            penalty += 2.5
            reasons.append('waist center inconsistent')
        if cr is not None:
            wc_dist = float(np.linalg.norm(cr - wc))
            if wc_dist >= max(25.0, waist_width * float(args.tta_min_waist_to_crotch_ratio)):
                geometry += 1.5
            else:
                penalty += 2.0
                reasons.append('crotch too close to waist')
        if lower is not None:
            uw = _safe_unit(waist_vec)
            ua = _safe_unit(lower - wc)
            if uw is not None and ua is not None:
                waist_axis_dot = abs(float(np.dot(uw, ua)))
                if waist_axis_dot <= float(args.tta_max_waist_axis_dot):
                    geometry += 2.0
                else:
                    penalty += 2.0
                    reasons.append('waist-axis geometry weak')
        if cr is not None and lower is not None:
            axis = lower - wc
            ua = _safe_unit(axis)
            axis_len = float(np.linalg.norm(axis))
            if ua is not None and axis_len > 1e-6:
                projection = float(np.dot(cr - wc, ua))
                if 0.15 * axis_len <= projection <= 0.90 * axis_len:
                    geometry += 1.2
                else:
                    penalty += 1.8
                    reasons.append('crotch projection inconsistent')
    else:
        penalty += 3.0
        reasons.append('waist 3 points incomplete')

    if lhc is not None and rhc is not None:
        hem_gap = float(np.linalg.norm(rhc - lhc))
    if cr is not None and lhc is not None and rhc is not None:
        leg_open = _angle_between_deg(lhc - cr, rhc - cr)

    need_pre_spread = False
    if hem_count < int(args.tta_pre_spread_min_hem_visible):
        need_pre_spread = True
        reasons.append('hem visible 부족')
    if not crotch_visible:
        need_pre_spread = True
        reasons.append('crotch 불안정')
    if waist_width is not None and hem_gap is not None and waist_width > 1e-6:
        if hem_gap / waist_width < float(args.tta_pre_spread_hem_gap_ratio):
            need_pre_spread = True
            reasons.append('다리 간격 좁음')
    if leg_open is not None and leg_open < float(args.tta_pre_spread_open_angle_deg):
        need_pre_spread = True
        reasons.append('다리 벌어짐 각도 작음')

    # Segmentation agreement is evaluated in the original image coordinates.
    mask_bonus = 0.0
    if mask is not None:
        signed_distances: List[float] = []
        for i in range(8):
            if valid[i]:
                signed_distances.append(float(cv2.pointPolygonTest(mask.contour, (float(kxy[i, 0]), float(kxy[i, 1])), True)))
        if signed_distances:
            inside_near = sum(d >= -float(args.tta_mask_max_outside_px) for d in signed_distances)
            inside_ratio = inside_near / len(signed_distances)
            mask_bonus += 5.0 * inside_ratio
            if inside_ratio < float(args.tta_mask_min_inside_ratio):
                penalty += 4.0
                reasons.append(f'pose-mask agreement low({inside_ratio:.2f})')

    score = (
        3.5 * waist_count + 2.0 * float(crotch_visible) + 1.0 * hem_count
        + 2.0 * mean_conf + 1.4 * visible_count + 3.2 * geometry
        + mask_bonus - 2.2 * penalty - (4.0 if need_pre_spread else 0.0)
    )

    if waist_count == 3 and geometry >= 3.0 and not need_pre_spread:
        state = 'OK'
        state_reason = '허리/가랑이/밑단 구조가 비교적 일관적'
    elif need_pre_spread:
        state = 'NEED_PRE_SPREAD'
        state_reason = ', '.join(dict.fromkeys(reasons)) or 'pre-spread 필요'
    else:
        state = 'WEAK_POSE'
        state_reason = ', '.join(dict.fromkeys(reasons)) or 'geometry score weak'

    return {
        'score': float(score), 'state': state, 'state_reason': state_reason,
        'visible_count': visible_count, 'waist_visible_count': waist_count,
        'hem_visible_count': hem_count, 'crotch_visible': crotch_visible,
        'waist_width_px': waist_width, 'hem_gap_px': hem_gap,
        'leg_open_angle_deg': leg_open, 'waist_axis_dot': waist_axis_dot,
        'mean_conf': mean_conf,
    }


def infer_bottom_pose_keypoints_tta(pose_model, frame: np.ndarray, mask: Optional[BottomMaskBoard], args) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Dict[str, Any], str]:
    """D23 staged pose TTA.

    D22 evaluated every angle x flip combination on every observation.  D23 first
    tries the last successful transform, then a small common-orientation set.  A
    candidate is accepted early only when the original D22 geometry state is OK,
    enough keypoints are visible, mean confidence is adequate, and the total score
    passes a configurable floor.  Otherwise the untouched full TTA set is evaluated.
    """
    global D23_TTA_HINT

    full_angles = list(getattr(args, 'pose_tta_angles_list', parse_pose_tta_angles(args.angles)))
    full_flips = list(getattr(args, 'pose_tta_flip_modes_list', parse_pose_tta_flip_modes(args.flip_modes)))
    fast_enabled = bool(getattr(args, 'd23_pose_tta_fast_first', True))
    fast_angles = parse_pose_tta_angles(getattr(args, 'd23_pose_tta_fast_angles', '0,180'))
    fast_flips = parse_pose_tta_flip_modes(getattr(args, 'd23_pose_tta_fast_flip_modes', 'none'))

    tested = 0
    errors: List[str] = []
    best = None
    attempted = set()

    def evaluate_one(angle: float, flip_mode: str):
        nonlocal tested, best
        key = (round(float(angle), 4), str(flip_mode))
        if key in attempted:
            return None
        attempted.add(key)
        tested += 1

        rotated, rotation_inv = rotate_image_and_inverse(frame, float(angle))
        work = rotated
        total_inv = rotation_inv
        post_remap = False
        if flip_mode == 'h':
            work, flip_inv = horizontal_flip_and_inverse(rotated)
            total_inv = rotation_inv @ flip_inv
            post_remap = True
        try:
            result = pose_model.predict(
                source=work, imgsz=int(args.pose_imgsz),
                conf=float(args.pose_conf), verbose=False,
            )[0]
        except Exception as exc:
            errors.append(f'{angle}/{flip_mode}:{type(exc).__name__}')
            return None
        kxy_aug, kcf, _status = read_pose_keypoints(
            result, work.shape, float(args.pose_kpt_conf)
        )
        if kxy_aug is None or kcf is None:
            return None
        kxy_orig = inverse_transform_keypoints(kxy_aug, total_inv)
        if post_remap:
            kxy_orig, kcf = remap_hflip_bottom_keypoints(kxy_orig, kcf)
        summary = score_bottom_pose_tta(kxy_orig, kcf, frame.shape, mask, args)
        candidate = {
            'kxy': kxy_orig, 'kcf': kcf,
            'angle': float(angle), 'flip_mode': str(flip_mode),
            'summary': summary,
        }
        if best is None or float(summary['score']) > float(best['summary']['score']):
            best = candidate
        return candidate

    def fast_acceptable(candidate: Optional[Dict[str, Any]]) -> bool:
        if candidate is None:
            return False
        summary = candidate.get('summary', {})
        return bool(
            str(summary.get('state', 'WEAK_POSE')) == 'OK'
            and int(summary.get('visible_count', 0)) >= int(getattr(args, 'd23_pose_tta_fast_min_visible', 7))
            and float(summary.get('mean_conf', 0.0)) >= float(getattr(args, 'd23_pose_tta_fast_min_mean_conf', 0.32))
            and float(summary.get('score', -1e9)) >= float(getattr(args, 'd23_pose_tta_fast_min_score', 28.0))
        )

    # 1) Last successful transform first.
    if fast_enabled:
        with D23_TTA_HINT_LOCK:
            hint = dict(D23_TTA_HINT)
        if bool(hint.get('valid', False)):
            cand = evaluate_one(float(hint.get('angle', 0.0)), str(hint.get('flip_mode', 'none')))
            if fast_acceptable(cand):
                meta = dict(cand['summary'])
                meta.update({
                    'angle': cand['angle'], 'flip_mode': cand['flip_mode'],
                    'tested_count': tested, 'fast_path': 'hint',
                })
                with D23_TTA_HINT_LOCK:
                    D23_TTA_HINT.update({
                        'valid': True, 'angle': cand['angle'],
                        'flip_mode': cand['flip_mode'],
                        'success_count': int(D23_TTA_HINT.get('success_count', 0)) + 1,
                    })
                return cand['kxy'], cand['kcf'], meta, 'OK'

        # 2) Cheap common transforms.
        for angle in fast_angles:
            for flip_mode in fast_flips:
                cand = evaluate_one(float(angle), str(flip_mode))
                if fast_acceptable(cand):
                    meta = dict(cand['summary'])
                    meta.update({
                        'angle': cand['angle'], 'flip_mode': cand['flip_mode'],
                        'tested_count': tested, 'fast_path': 'common',
                    })
                    with D23_TTA_HINT_LOCK:
                        D23_TTA_HINT.update({
                            'valid': True, 'angle': cand['angle'],
                            'flip_mode': cand['flip_mode'],
                            'success_count': int(D23_TTA_HINT.get('success_count', 0)) + 1,
                        })
                    return cand['kxy'], cand['kcf'], meta, 'OK'

    # 3) Full D22 fallback.  Already-tried combinations are skipped.
    for angle in full_angles:
        for flip_mode in full_flips:
            evaluate_one(float(angle), str(flip_mode))

    if best is None:
        suffix = f" errors={errors[:3]}" if errors else ''
        return None, None, {'tested_count': tested, 'fast_path': 'failed'}, f'no TTA pose candidate; tested={tested}{suffix}'

    meta = dict(best['summary'])
    meta.update({
        'angle': best['angle'], 'flip_mode': best['flip_mode'],
        'tested_count': tested, 'fast_path': 'full',
    })
    if str(meta.get('state', 'WEAK_POSE')) == 'OK':
        with D23_TTA_HINT_LOCK:
            D23_TTA_HINT.update({
                'valid': True, 'angle': best['angle'],
                'flip_mode': best['flip_mode'],
                'success_count': int(D23_TTA_HINT.get('success_count', 0)) + 1,
            })
    return best['kxy'], best['kcf'], meta, 'OK'

def infer_bottom_pose(pose_model, frame: np.ndarray, H: Optional[np.ndarray], imgsz: int, conf: float,
                      kpt_conf: float, mask: Optional[BottomMaskBoard] = None, args=None) -> Tuple[Optional[BottomPoseBoard], str]:
    if pose_model is None:
        return None, "pose model is None"
    if H is None:
        return None, "Homography not locked"
    tta_meta: Dict[str, Any] = {
        "angle": 0.0, "flip_mode": "none", "score": 0.0,
        "state": "SINGLE", "state_reason": "single inference", "tested_count": 1,
    }
    if args is not None and bool(getattr(args, "pose_tta", True)):
        kxy, kcf, tta_meta, status = infer_bottom_pose_keypoints_tta(pose_model, frame, mask, args)
        if kxy is None or kcf is None:
            return None, status
    else:
        try:
            result = pose_model.predict(source=frame, imgsz=int(imgsz), conf=float(conf), verbose=False)[0]
        except Exception as e:
            return None, f"pose predict error: {repr(e)}"
        kxy, kcf, status = read_pose_keypoints(result, frame.shape, kpt_conf)
        if kxy is None or kcf is None:
            return None, status

    keypoints_px: Dict[str, Tuple[float, float]] = {}
    keypoints_board: Dict[str, Tuple[float, float]] = {}
    keypoint_conf: Dict[str, float] = {}
    valid_names = []
    for name, idx in BOTTOM_POSE_KPT.items():
        if not kpt_valid_xy_conf(kxy, kcf, idx, frame.shape, kpt_conf):
            continue
        x, y = float(kxy[idx, 0]), float(kxy[idx, 1])
        bx, by = pixel_to_board(H, x, y)
        keypoints_px[name] = (x, y)
        keypoints_board[name] = (bx, by)
        keypoint_conf[name] = float(kcf[idx])
        valid_names.append(name)

    required = ["waist_center", "crotch"]
    missing_required = [n for n in required if n not in keypoints_board]
    hem_names = ["img_left_hem_outer", "img_left_hem_inner", "img_right_hem_inner", "img_right_hem_outer"]
    valid_hem_count = sum(1 for n in hem_names if n in keypoints_board)
    waist_edge_ok = "waist_img_left" in keypoints_board and "waist_img_right" in keypoints_board
    if missing_required:
        return None, f"missing required bottom pose keypoints: {missing_required}"
    if valid_hem_count < 3:
        return None, f"not enough hem keypoints: {valid_hem_count}/4"

    def p(name: str) -> np.ndarray:
        return np.asarray(keypoints_board[name], dtype=np.float32)

    waist_center = p("waist_center")
    crotch = p("crotch")

    if waist_edge_ok:
        waist_left = p("waist_img_left")
        waist_right = p("waist_img_right")
    else:
        # Fallback: construct approximate waist endpoints from mask orientation if edges are missing.
        waist_left = waist_center + np.asarray([-35.0, 0.0], dtype=np.float32)
        waist_right = waist_center + np.asarray([35.0, 0.0], dtype=np.float32)

    left_hems = [p(n) for n in ("img_left_hem_outer", "img_left_hem_inner") if n in keypoints_board]
    right_hems = [p(n) for n in ("img_right_hem_inner", "img_right_hem_outer") if n in keypoints_board]
    if not left_hems or not right_hems:
        # If one leg side is missing, split all hem points by x as a fallback.
        all_hems = [p(n) for n in hem_names if n in keypoints_board]
        all_hems = sorted(all_hems, key=lambda v: float(v[0]))
        mid = len(all_hems) // 2
        left_hems = all_hems[:mid]
        right_hems = all_hems[mid:]
    left_hem_center = np.mean(np.asarray(left_hems, dtype=np.float32), axis=0)
    right_hem_center = np.mean(np.asarray(right_hems, dtype=np.float32), axis=0)
    lower_center = 0.5 * (left_hem_center + right_hem_center)
    pose_center = np.mean(np.asarray([waist_center, crotch, left_hem_center, right_hem_center], dtype=np.float32), axis=0)

    waist_vec = waist_right - waist_left
    pants_axis = lower_center - waist_center
    waist_angle = float(np.degrees(np.arctan2(float(waist_vec[1]), float(waist_vec[0]))))
    pants_axis_angle = float(np.degrees(np.arctan2(float(pants_axis[1]), float(pants_axis[0]))))
    pants_axis_len = float(np.linalg.norm(pants_axis))
    waist_width = float(np.linalg.norm(waist_vec))
    hem_gap = float(np.linalg.norm(right_hem_center - left_hem_center))

    valid = True
    reasons = []
    if pants_axis_len < 80.0:
        valid = False
        reasons.append(f"pants_axis too short {pants_axis_len:.1f}mm")
    if mask is not None:
        dist = float(np.linalg.norm(pose_center - mask.center_board))
        if dist > 120.0:
            valid = False
            reasons.append(f"pose/mask center far {dist:.1f}mm")
        # Crotch should be inside or near the mask. Pixel test is safer than board-only test.
        if "crotch" in keypoints_px:
            cx, cy = keypoints_px["crotch"]
            inside = cv2.pointPolygonTest(mask.contour, (float(cx), float(cy)), True)
            if inside < -28.0:
                valid = False
                reasons.append(f"crotch outside mask {inside:.1f}px")

    # Keep weak/pre-spread TTA results visible on screen, but block robot planning by default.
    if (args is not None and bool(getattr(args, "pose_tta", True))
            and bool(getattr(args, "tta_require_ok_for_motion", True))
            and str(tta_meta.get("state", "WEAK_POSE")) != "OK"):
        valid = False
        reasons.append(
            f"TTA {tta_meta.get('state')}: {tta_meta.get('state_reason', 'geometry weak')}"
        )

    pose = BottomPoseBoard(
        keypoints_board=keypoints_board,
        keypoints_px=keypoints_px,
        keypoint_conf=keypoint_conf,
        waist_left=waist_left,
        waist_center=waist_center,
        waist_right=waist_right,
        crotch=crotch,
        left_hem_center=left_hem_center,
        right_hem_center=right_hem_center,
        lower_center=lower_center,
        pose_center=pose_center,
        waist_angle_deg=waist_angle,
        pants_axis_angle_deg=pants_axis_angle,
        pants_axis_len_mm=pants_axis_len,
        waist_width_mm=waist_width,
        hem_gap_mm=hem_gap,
        valid=valid,
        reason="OK" if valid else "; ".join(reasons),
        tta_angle_deg=float(tta_meta.get("angle", 0.0)),
        tta_flip_mode=str(tta_meta.get("flip_mode", "none")),
        tta_score=float(tta_meta.get("score", 0.0)),
        tta_state=str(tta_meta.get("state", "SINGLE")),
        tta_reason=str(tta_meta.get("state_reason", "")),
        tta_tested_count=int(tta_meta.get("tested_count", 1)),
    )
    return pose, "OK" if valid else pose.reason



def _largest_component_with_pose(mask_u8: np.ndarray, pose: BottomPoseBoard) -> Optional[np.ndarray]:
    """Select the connected component containing the most pose keypoints."""
    num, labels, stats, _ = cv2.connectedComponentsWithStats((mask_u8 > 0).astype(np.uint8), 8)
    if num <= 1:
        return None

    best_label = None
    best_score = -1e9
    pts = list(pose.keypoints_px.values())

    for label in range(1, num):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < 100:
            continue
        comp = (labels == label).astype(np.uint8)
        inside = 0
        for x, y in pts:
            xi = int(round(x))
            yi = int(round(y))
            if 0 <= yi < comp.shape[0] and 0 <= xi < comp.shape[1] and comp[yi, xi] > 0:
                inside += 1
        score = inside * 100000.0 + area
        if score > best_score:
            best_score = score
            best_label = label

    if best_label is None:
        return None
    return ((labels == best_label).astype(np.uint8) * 255)


def build_pose_grabcut_fallback_mask(
    frame: np.ndarray,
    H: np.ndarray,
    pose: Optional[BottomPoseBoard],
    args,
) -> Tuple[Optional[BottomMaskBoard], str]:
    """Build a conservative temporary pants mask from pose seeds + GrabCut.

    This is used only when the trained segmentation model produces no bottoms mask.
    D3-style CLAHE is still used later only for wrinkle heatmap extraction.
    """
    if pose is None:
        return None, "fallback: pose 없음"
    if str(getattr(pose, "tta_state", "WEAK_POSE")) != "OK":
        return None, f"fallback: pose TTA state={pose.tta_state}"

    h, w = frame.shape[:2]
    pts = np.asarray(list(pose.keypoints_px.values()), dtype=np.float32)
    if len(pts) < int(getattr(args, "fallback_mask_min_pose_points", 6)):
        return None, f"fallback: pose points 부족 {len(pts)}"

    pad = max(8, int(getattr(args, "fallback_mask_bbox_pad_px", 55)))
    x0 = max(1, int(np.floor(pts[:, 0].min())) - pad)
    y0 = max(1, int(np.floor(pts[:, 1].min())) - pad)
    x1 = min(w - 2, int(np.ceil(pts[:, 0].max())) + pad)
    y1 = min(h - 2, int(np.ceil(pts[:, 1].max())) + pad)
    if x1 - x0 < 20 or y1 - y0 < 20:
        return None, "fallback: pose ROI가 너무 작음"

    gc = np.full((h, w), cv2.GC_BGD, dtype=np.uint8)
    gc[y0:y1 + 1, x0:x1 + 1] = cv2.GC_PR_BGD

    # Pose skeleton is definite foreground.
    seed_radius = max(3, int(getattr(args, "fallback_mask_seed_radius_px", 12)))
    line_thickness = max(3, int(getattr(args, "fallback_mask_seed_line_px", 18)))
    for x, y in pts:
        cv2.circle(gc, (int(round(x)), int(round(y))), seed_radius, cv2.GC_FGD, -1)

    kp = pose.keypoints_px
    links = [
        ("waist_img_left", "waist_center"),
        ("waist_center", "waist_img_right"),
        ("waist_center", "crotch"),
        ("crotch", "img_left_hem_inner"),
        ("crotch", "img_right_hem_inner"),
        ("img_left_hem_outer", "img_left_hem_inner"),
        ("img_right_hem_inner", "img_right_hem_outer"),
    ]
    for a, b in links:
        if a in kp and b in kp:
            pa = tuple(np.round(kp[a]).astype(int))
            pb = tuple(np.round(kp[b]).astype(int))
            cv2.line(gc, pa, pb, cv2.GC_FGD, line_thickness)

    # Convex hull is probable foreground, not definite foreground.
    hull = cv2.convexHull(pts.astype(np.int32).reshape(-1, 1, 2))
    probable = np.zeros((h, w), dtype=np.uint8)
    cv2.fillConvexPoly(probable, hull.reshape(-1, 2), 255)
    dilate_px = max(3, int(getattr(args, "fallback_mask_hull_dilate_px", 28)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_px * 2 + 1, dilate_px * 2 + 1))
    probable = cv2.dilate(probable, kernel)
    gc[(probable > 0) & (gc != cv2.GC_FGD)] = cv2.GC_PR_FGD

    bg_model = np.zeros((1, 65), np.float64)
    fg_model = np.zeros((1, 65), np.float64)
    try:
        cv2.grabCut(
            frame,
            gc,
            None,
            bg_model,
            fg_model,
            max(1, int(getattr(args, "fallback_mask_grabcut_iters", 4))),
            cv2.GC_INIT_WITH_MASK,
        )
    except Exception as exc:
        return None, f"fallback GrabCut 오류: {exc!r}"

    mask = np.where((gc == cv2.GC_FGD) | (gc == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)

    close_px = max(1, int(getattr(args, "fallback_mask_close_px", 9)))
    open_px = max(1, int(getattr(args, "fallback_mask_open_px", 3)))
    close_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_px, close_px))
    open_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_px, open_px))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_k)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, open_k)

    selected = _largest_component_with_pose(mask, pose)
    if selected is None:
        return None, "fallback: pose와 연결된 component 없음"
    mask = selected

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, "fallback: contour 없음"
    contour = max(contours, key=cv2.contourArea)
    area = float(cv2.contourArea(contour))
    frame_area = float(h * w)
    min_area = float(getattr(args, "fallback_mask_min_area_px", 1800.0))
    max_ratio = float(getattr(args, "fallback_mask_max_frame_ratio", 0.70))
    if area < min_area:
        return None, f"fallback: mask area 작음 {area:.0f}px"
    if area > frame_area * max_ratio:
        return None, f"fallback: mask area 과대 {area / frame_area:.2f}"

    # Safety validation: enough pose points must lie inside or very near the mask.
    max_outside = float(getattr(args, "fallback_mask_max_kpt_outside_px", 22.0))
    inside_count = 0
    for x, y in pose.keypoints_px.values():
        d = cv2.pointPolygonTest(contour, (float(x), float(y)), True)
        if d >= -max_outside:
            inside_count += 1
    min_inside = int(getattr(args, "fallback_mask_min_inside_points", 6))
    if inside_count < min_inside:
        return None, f"fallback: pose-mask 일치 부족 {inside_count}/{len(pose.keypoints_px)}"

    M = cv2.moments(contour)
    if abs(M["m00"]) > 1e-6:
        center_px = np.asarray([M["m10"] / M["m00"], M["m01"] / M["m00"]], dtype=np.float32)
    else:
        center_px = contour.reshape(-1, 2).astype(np.float32).mean(axis=0)
    bx, by = pixel_to_board(H, float(center_px[0]), float(center_px[1]))

    hull2 = cv2.convexHull(contour)
    hull_area = float(cv2.contourArea(hull2))
    solidity = area / hull_area if hull_area > 1.0 else 0.0
    min_solidity = float(getattr(args, "fallback_mask_min_solidity", 0.35))
    if solidity < min_solidity:
        return None, f"fallback: solidity 낮음 {solidity:.2f}"

    return BottomMaskBoard(
        mask_u8=mask,
        contour=contour,
        area_px=area,
        center_px=center_px,
        center_board=np.asarray([bx, by], dtype=np.float32),
        class_name="bottoms_pose_grabcut",
        confidence=0.0,
        solidity=float(solidity),
    ), f"POSE_GRABCUT_OK inside={inside_count} area={area:.0f}px solidity={solidity:.2f}"

def infer_bottom_observation(seg_model, pose_model, frame: np.ndarray, H: Optional[np.ndarray], args,
                             cfg: BottomSafetyConfig) -> BottomObservation:
    if H is None:
        return BottomObservation(None, None, None, False, "Homography not locked", frame)

    # 1) Original trained segmentation path.
    mask, mask_status = infer_bottoms_mask(seg_model, frame, H, args.seg_imgsz, args.seg_conf)

    # 2) Pose is inferred from the raw/manual-exposure image, never the CLAHE heatmap image.
    pose, pose_status = infer_bottom_pose(
        pose_model, frame, H,
        args.pose_imgsz, args.pose_conf, args.pose_kpt_conf,
        mask=mask, args=args,
    )

    fallback_status = ""
    if mask is None and bool(getattr(args, "pose_grabcut_fallback", True)):
        mask, fallback_status = build_pose_grabcut_fallback_mask(frame, H, pose, args)
        if mask is not None:
            print(f"[MASK-FALLBACK] {fallback_status}")
        else:
            print(f"[MASK-FALLBACK-BLOCKED] {fallback_status}")

    fused = None
    valid = True
    reasons = []

    if mask is None:
        valid = False
        detail = fallback_status if fallback_status else mask_status
        reasons.append(f"mask invalid: {detail}")
    elif mask.area_px < cfg.min_mask_area_px:
        valid = False
        reasons.append(f"mask area too small: {mask.area_px:.0f}px")

    if pose is None:
        valid = False
        reasons.append(f"pose invalid: {pose_status}")
    elif not pose.valid:
        valid = False
        reasons.append(f"pose invalid: {pose.reason}")

    if pose is not None and mask is not None:
        # Re-check pose-mask consistency for both trained and fallback masks.
        inside = 0
        for x, y in pose.keypoints_px.values():
            d = cv2.pointPolygonTest(mask.contour, (float(x), float(y)), True)
            if d >= -float(getattr(args, "fallback_mask_max_kpt_outside_px", 22.0)):
                inside += 1
        if inside < int(getattr(args, "fallback_mask_min_inside_points", 6)):
            valid = False
            reasons.append(f"pose/mask keypoint consistency low: {inside}/{len(pose.keypoints_px)}")

        fused = 0.5 * pose.pose_center + 0.5 * mask.center_board
    elif mask is not None:
        fused = mask.center_board.copy()

    return BottomObservation(
        pose=pose,
        mask=mask,
        fused_center_board=fused,
        valid=valid,
        reason="OK" if valid else " | ".join(reasons),
        frame=frame,
    )


# -----------------------------------------------------------------------------
# B34-style texture wrinkle heatmap (visualization only)
# -----------------------------------------------------------------------------

def _odd_ksize(value: int, minimum: int = 3) -> int:
    k = max(int(minimum), int(value))
    return k if k % 2 == 1 else k + 1


def build_wrinkle_heatmap(frame: np.ndarray, mask_u8: Optional[np.ndarray], args) -> Optional[WrinkleHeatmapPlan]:
    """Build a texture/contrast heatmap inside the garment mask.

    This follows the B34 idea: CLAHE on the luminance channel, low-frequency
    subtraction, optional multi-orientation Gabor response, mask erosion,
    percentile thresholding, and connected-component filtering. It is not a
    trained wrinkle detector and must not be used as a robot-motion safety gate.
    """
    if not bool(getattr(args, 'wrinkle_heatmap_mode', True)):
        return None
    if frame is None or mask_u8 is None:
        return None
    if mask_u8.shape[:2] != frame.shape[:2]:
        mask = cv2.resize(mask_u8, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)
    else:
        mask = mask_u8.copy()
    mask = (mask > 0).astype(np.uint8) * 255

    erode_px = max(0, int(getattr(args, 'wrinkle_heatmap_erode_px', 12)))
    if erode_px > 0:
        ek = _odd_ksize(erode_px * 2 + 1)
        inner_mask = cv2.erode(mask, np.ones((ek, ek), np.uint8), iterations=1)
    else:
        inner_mask = mask
    if cv2.countNonZero(inner_mask) < 100:
        return WrinkleHeatmapPlan(
            heatmap=np.zeros(frame.shape[:2], np.uint8),
            inner_mask=inner_mask,
            binary=np.zeros(frame.shape[:2], np.uint8),
            candidates=[], threshold=255.0, wrinkle_ratio=0.0,
            status='MASK_TOO_SMALL', summary='wrinkle heatmap: inner mask too small'
        )

    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    lch = lab[:, :, 0]
    tile = max(2, int(getattr(args, 'wrinkle_heatmap_clahe_tile', 8)))
    clip = max(0.1, float(getattr(args, 'wrinkle_heatmap_clahe_clip', 2.0)))
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(tile, tile))
    enhanced = clahe.apply(lch)

    blur_k = _odd_ksize(int(getattr(args, 'wrinkle_heatmap_blur_ksize', 41)))
    low = cv2.GaussianBlur(enhanced, (blur_k, blur_k), 0)
    high = cv2.absdiff(enhanced, low).astype(np.float32)

    score = high
    if bool(getattr(args, 'wrinkle_heatmap_gabor', True)):
        gk = _odd_ksize(int(getattr(args, 'wrinkle_heatmap_gabor_ksize', 17)))
        sigma = max(0.5, float(getattr(args, 'wrinkle_heatmap_gabor_sigma', 4.0)))
        lambd = max(1.0, float(getattr(args, 'wrinkle_heatmap_gabor_lambda', 8.0)))
        responses = []
        for theta in (0.0, np.pi / 4.0, np.pi / 2.0, 3.0 * np.pi / 4.0):
            kernel = cv2.getGaborKernel((gk, gk), sigma, theta, lambd, 0.5, 0.0, ktype=cv2.CV_32F)
            resp = cv2.filter2D(enhanced.astype(np.float32), cv2.CV_32F, kernel)
            responses.append(np.abs(resp))
        gabor = np.maximum.reduce(responses)
        valid_g = gabor[inner_mask > 0]
        if valid_g.size:
            gscale = float(np.percentile(valid_g, 99.0))
            if gscale > 1e-6:
                gabor = np.clip(gabor * (255.0 / gscale), 0.0, 255.0)
        score = 0.70 * high + 0.30 * gabor

    valid = score[inner_mask > 0]
    if valid.size == 0:
        return None
    scale = float(np.percentile(valid, 99.5))
    if scale <= 1e-6:
        heat = np.zeros(frame.shape[:2], np.uint8)
    else:
        heat = np.clip(score * (255.0 / scale), 0.0, 255.0).astype(np.uint8)
    heat[inner_mask == 0] = 0

    pct = float(np.clip(float(getattr(args, 'wrinkle_heatmap_percentile', 93.0)), 50.0, 99.9))
    valid_heat = heat[inner_mask > 0]
    percentile_thr = float(np.percentile(valid_heat, pct)) if valid_heat.size else 255.0
    score_min = float(getattr(args, 'wrinkle_heatmap_score_min', 35.0))
    thr = max(score_min, percentile_thr)
    binary = np.zeros_like(heat)
    binary[(heat >= thr) & (inner_mask > 0)] = 255
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8), iterations=1)

    n, labels, stats, cents = cv2.connectedComponentsWithStats(binary, connectivity=8)
    min_area = max(1.0, float(getattr(args, 'wrinkle_heatmap_min_blob_area', 45.0)))
    kept = np.zeros_like(binary)
    candidates: List[Dict[str, Any]] = []
    for idx in range(1, n):
        area = float(stats[idx, cv2.CC_STAT_AREA])
        if area < min_area:
            continue
        comp = labels == idx
        kept[comp] = 255
        x = int(stats[idx, cv2.CC_STAT_LEFT])
        y = int(stats[idx, cv2.CC_STAT_TOP])
        w = int(stats[idx, cv2.CC_STAT_WIDTH])
        h = int(stats[idx, cv2.CC_STAT_HEIGHT])
        cx, cy = map(float, cents[idx])
        vals = heat[comp]
        ys_comp, xs_comp = np.where(comp)
        pts = np.column_stack([xs_comp, ys_comp]).astype(np.float32)
        major_axis = np.asarray([1.0, 0.0], dtype=np.float32)
        minor_axis = np.asarray([0.0, 1.0], dtype=np.float32)
        major_len = float(max(w, h))
        minor_len = float(max(1, min(w, h)))
        if len(pts) >= 5:
            centered = pts - pts.mean(axis=0, keepdims=True)
            cov = np.cov(centered.T)
            eigvals, eigvecs = np.linalg.eigh(cov)
            order = np.argsort(eigvals)[::-1]
            major_axis = eigvecs[:, order[0]].astype(np.float32)
            major_axis = major_axis / max(1e-6, float(np.linalg.norm(major_axis)))
            minor_axis = np.asarray([-major_axis[1], major_axis[0]], dtype=np.float32)
            proj_major = centered @ major_axis
            proj_minor = centered @ minor_axis
            major_len = float(np.ptp(proj_major)) if proj_major.size else major_len
            minor_len = float(np.ptp(proj_minor)) if proj_minor.size else minor_len
        linearity = float(major_len / max(1.0, minor_len))
        dist_map = cv2.distanceTransform((inner_mask > 0).astype(np.uint8), cv2.DIST_L2, 5)
        cxi = int(np.clip(round(cx), 0, dist_map.shape[1] - 1))
        cyi = int(np.clip(round(cy), 0, dist_map.shape[0] - 1))
        edge_distance_px = float(dist_map[cyi, cxi])
        mean_score = float(np.mean(vals)) if vals.size else 0.0
        max_score = float(np.max(vals)) if vals.size else 0.0
        priority = float(mean_score * math.sqrt(max(1.0, area)) * min(4.0, max(1.0, linearity)))
        candidates.append({
            'area_px': area, 'bbox': [x, y, w, h], 'center_px': [cx, cy],
            'mean_score': mean_score, 'max_score': max_score,
            'major_axis_px': [float(major_axis[0]), float(major_axis[1])],
            'minor_axis_px': [float(minor_axis[0]), float(minor_axis[1])],
            'major_length_px': major_len, 'minor_length_px': minor_len,
            'linearity': linearity, 'edge_distance_px': edge_distance_px,
            'priority_score': priority,
        })
    candidates.sort(key=lambda c: (float(c.get('priority_score', 0.0)),
                                   float(c['max_score']), float(c['area_px'])), reverse=True)
    max_candidates = max(1, int(getattr(args, 'wrinkle_heatmap_max_candidates', 6)))
    candidates = candidates[:max_candidates]
    denom = max(1, cv2.countNonZero(inner_mask))
    ratio = float(cv2.countNonZero(kept)) / float(denom)
    summary = f'wrinkle ratio={ratio:.3f} blobs={len(candidates)} thr={thr:.1f} pct={pct:.1f}'
    return WrinkleHeatmapPlan(heat, inner_mask, kept, candidates, thr, ratio, 'OK', summary)



def _d19_board_polygon_to_mask(H: np.ndarray, points_board: Sequence[Sequence[float]], image_shape) -> np.ndarray:
    out = np.zeros(image_shape[:2], dtype=np.uint8)
    if H is None or not points_board:
        return out
    pts_px = []
    for p in points_board:
        q = board_to_pixel(H, float(p[0]), float(p[1]))
        if q is None:
            return out
        pts_px.append([int(round(q[0])), int(round(q[1]))])
    if len(pts_px) >= 3:
        cv2.fillPoly(out, [np.asarray(pts_px, dtype=np.int32)], 255)
    return out


def _d19_pose_ignore_mask(obs: Optional[BottomObservation], H: np.ndarray, args) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Pose-relative natural-structure mask for pants.

    A124 excludes pose-known garment structures from the final wrinkle decision.
    D19 applies the same idea to bottoms: the elastic waistband band and the
    crotch junction are not actionable wrinkle regions.
    """
    shape = obs.frame.shape if obs is not None and obs.frame is not None else None
    if shape is None and obs is not None and obs.mask is not None:
        shape = obs.mask.mask_u8.shape
    if shape is None:
        return np.zeros((1, 1), dtype=np.uint8), {"valid": False, "reason": "no image shape"}
    ignore = np.zeros(shape[:2], dtype=np.uint8)
    info: Dict[str, Any] = {
        "valid": False,
        "waistband_enabled": bool(getattr(args, "d19_ignore_waistband_wrinkles", True)),
        "crotch_enabled": bool(getattr(args, "d19_ignore_crotch_wrinkles", True)),
        "waistband_px": 0,
        "crotch_px": 0,
    }
    if obs is None or obs.pose is None or not obs.pose.valid or H is None:
        info["reason"] = "pose invalid"
        return ignore, info

    p = obs.pose
    wl = np.asarray(p.waist_left, dtype=np.float32)
    wc = np.asarray(p.waist_center, dtype=np.float32)
    wr = np.asarray(p.waist_right, dtype=np.float32)
    cr = np.asarray(p.crotch, dtype=np.float32)
    lower = np.asarray(p.lower_center, dtype=np.float32)

    body_u = _safe_unit(lower - wc)
    lateral_u = _safe_unit(wr - wl)
    # Keep a consistent right-handed frame even when pose labels are noisy.
    if abs(float(np.dot(body_u, lateral_u))) > 0.55:
        lateral_u = np.asarray([-body_u[1], body_u[0]], dtype=np.float32)
    if float(np.dot(wr - wl, lateral_u)) < 0.0:
        lateral_u = -lateral_u

    if bool(getattr(args, "d19_ignore_waistband_wrinkles", True)):
        back = max(0.0, float(getattr(args, "d19_waistband_back_mm", 18.0)))
        base_forward = max(0.0, float(getattr(args, "d19_waistband_forward_mm", 82.0)))
        waist_to_crotch = float(np.linalg.norm(cr - wc))
        ratio_forward = waist_to_crotch * max(0.0, float(getattr(args, "d23_waistband_forward_ratio", 0.70)))
        forward = max(base_forward, min(
            float(getattr(args, "d23_waistband_forward_max_mm", 105.0)),
            ratio_forward,
        ))
        side = max(0.0, float(getattr(args, "d19_waistband_side_expand_mm", 30.0)))
        info["waist_to_crotch_mm"] = waist_to_crotch
        info["waistband_forward_used_mm"] = float(forward)
        left = wl - lateral_u * side
        right = wr + lateral_u * side
        poly = [
            left - body_u * back,
            right - body_u * back,
            right + body_u * forward,
            left + body_u * forward,
        ]
        waist_mask = _d19_board_polygon_to_mask(H, poly, shape)
        ignore[waist_mask > 0] = 255
        info["waistband_px"] = int(cv2.countNonZero(waist_mask))

    if bool(getattr(args, "d19_ignore_crotch_wrinkles", True)):
        axial = max(10.0, float(getattr(args, "d19_crotch_axial_radius_mm", 72.0)))
        lateral = max(10.0, float(getattr(args, "d19_crotch_lateral_radius_mm", 78.0)))
        forward_shift = float(getattr(args, "d19_crotch_forward_shift_mm", 12.0))
        center = cr + body_u * forward_shift
        ellipse = []
        for theta in np.linspace(0.0, 2.0 * math.pi, 40, endpoint=False):
            ellipse.append(
                center
                + body_u * (math.cos(float(theta)) * axial)
                + lateral_u * (math.sin(float(theta)) * lateral)
            )
        crotch_mask = _d19_board_polygon_to_mask(H, ellipse, shape)
        ignore[crotch_mask > 0] = 255
        info["crotch_px"] = int(cv2.countNonZero(crotch_mask))

    if obs.mask is not None and obs.mask.mask_u8.shape[:2] == ignore.shape[:2]:
        ignore = cv2.bitwise_and(ignore, obs.mask.mask_u8)
    info["valid"] = True
    info["ignored_px"] = int(cv2.countNonZero(ignore))
    info["reason"] = "OK"
    return ignore, info


def _d19_component_geometry(pts_px: np.ndarray, H: np.ndarray) -> Dict[str, float]:
    pts_px = np.asarray(pts_px, dtype=np.float32).reshape(-1, 2)
    if len(pts_px) < 3:
        return {"major_length_mm": 0.0, "minor_length_mm": 0.0, "linearity": 1.0}
    centered_px = pts_px - pts_px.mean(axis=0, keepdims=True)
    try:
        cov_px = np.cov(centered_px.T)
        vals_px, vecs_px = np.linalg.eigh(cov_px)
        major_px_u = _safe_unit(vecs_px[:, int(np.argmax(vals_px))])
        normal_px_u = np.asarray([-major_px_u[1], major_px_u[0]], dtype=np.float32)
        major_px = float(np.ptp(centered_px @ major_px_u))
        minor_px = float(np.ptp(centered_px @ normal_px_u))
    except Exception:
        major_px = 0.0
        minor_px = 1.0

    major_mm = 0.0
    minor_mm = 0.0
    if H is not None and len(pts_px) >= 5:
        try:
            board = cv2.perspectiveTransform(pts_px.reshape(-1, 1, 2), H).reshape(-1, 2)
            center_b = board.mean(axis=0, keepdims=True)
            X = board - center_b
            cov = np.cov(X.T)
            vals, vecs = np.linalg.eigh(cov)
            major_u = _safe_unit(vecs[:, int(np.argmax(vals))])
            normal_u = np.asarray([-major_u[1], major_u[0]], dtype=np.float32)
            major_mm = float(np.ptp(X @ major_u))
            minor_mm = float(np.ptp(X @ normal_u))
        except Exception:
            pass
    linearity = float((major_mm if major_mm > 0.0 else major_px) /
                      max(1.0, (minor_mm if minor_mm > 0.0 else minor_px)))
    return {
        "major_length_px": float(major_px),
        "minor_length_px": float(minor_px),
        "major_length_mm": float(major_mm),
        "minor_length_mm": float(minor_mm),
        "linearity": float(linearity),
    }



def _d20_local_shadow_map(frame: np.ndarray, effective_inner: np.ndarray, args) -> Dict[str, Any]:
    """Estimate dark fold shadows relative to nearby cloth, not absolute black level.

    For black pants, a fixed grayscale cutoff is unstable.  D20 compares each
    pixel with a local Gaussian brightness baseline and keeps only the negative
    residual (local_baseline - current_luminance).  The residual is normalized
    with a per-frame robust percentile but never with a scale smaller than
    d20_shadow_min_scale_raw, so sensor noise is not exaggerated on a uniformly
    black garment.
    """
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    lch = lab[:, :, 0].astype(np.float32)

    blur_k = _odd_ksize(int(getattr(args, "d20_shadow_blur_ksize", 51)), 9)
    local_base = cv2.GaussianBlur(lch, (blur_k, blur_k), 0)
    shadow_raw = np.maximum(local_base - lch, 0.0)
    shadow_raw[effective_inner == 0] = 0.0

    valid = shadow_raw[effective_inner > 0]
    if valid.size:
        p90 = float(np.percentile(valid, 90.0))
        p98 = float(np.percentile(valid, 98.0))
        p995 = float(np.percentile(valid, 99.5))
    else:
        p90 = p98 = p995 = 0.0

    min_scale = max(1.0, float(getattr(args, "d20_shadow_min_scale_raw", 8.0)))
    scale = max(min_scale, p995)
    shadow_norm = np.clip(shadow_raw * (255.0 / scale), 0.0, 255.0).astype(np.uint8)
    shadow_norm[effective_inner == 0] = 0

    support_raw = max(
        float(getattr(args, "d20_shadow_support_raw_min", 3.0)),
        p90,
    )
    return {
        "raw": shadow_raw,
        "norm": shadow_norm,
        "local_base": local_base,
        "scale_raw": float(scale),
        "p90_raw": p90,
        "p98_raw": p98,
        "p995_raw": p995,
        "support_raw_threshold": float(support_raw),
    }


def _d20_wrinkle_tier(area: float, area_ratio: float, major_mm: float, major_px: float,
                       dark_mean: float, dark_max: float, dark_support: float,
                       args) -> Tuple[int, str, Dict[str, bool]]:
    """Return 3=large/long/dark, 2=large-or-long dark, 1=smaller actionable, 0=ignore."""
    strong_area = max(1.0, float(getattr(args, "d20_strong_area_px", 240.0)))
    strong_area_ratio = max(0.0, float(getattr(args, "d20_strong_area_ratio", 0.0015)))
    strong_len_mm = max(1.0, float(getattr(args, "d20_strong_length_mm", 65.0)))
    strong_len_px = max(1.0, float(getattr(args, "d20_strong_length_px", 85.0)))
    medium_area = max(1.0, float(getattr(args, "d20_medium_area_px", 150.0)))
    medium_area_ratio = max(0.0, float(getattr(args, "d20_medium_area_ratio", 0.0009)))
    medium_len_mm = max(1.0, float(getattr(args, "d20_medium_length_mm", 45.0)))
    medium_len_px = max(1.0, float(getattr(args, "d20_medium_length_px", 58.0)))

    strong_dark = bool(
        (dark_mean >= float(getattr(args, "d20_strong_dark_mean", 38.0))
         or dark_max >= float(getattr(args, "d20_strong_dark_max", 105.0)))
        and dark_support >= float(getattr(args, "d20_strong_dark_support", 0.10))
    )
    medium_dark = bool(
        (dark_mean >= float(getattr(args, "d20_medium_dark_mean", 20.0))
         or dark_max >= float(getattr(args, "d20_medium_dark_max", 65.0)))
        and dark_support >= float(getattr(args, "d20_medium_dark_support", 0.06))
    )

    large = bool(area >= strong_area or area_ratio >= strong_area_ratio)
    long = bool(
        major_mm >= strong_len_mm if major_mm > 0.0 else major_px >= strong_len_px
    )
    medium_size = bool(area >= medium_area or area_ratio >= medium_area_ratio)
    medium_len = bool(
        major_mm >= medium_len_mm if major_mm > 0.0 else major_px >= medium_len_px
    )

    require_dark = bool(getattr(args, "d20_require_dark_shadow", True))
    if large and long and (strong_dark or not require_dark):
        return 3, "LARGE_LONG_DARK", {
            "large": large, "long": long, "strong_dark": strong_dark,
            "medium_size": medium_size, "medium_len": medium_len,
            "medium_dark": medium_dark,
        }
    if (large or long) and (medium_dark or not require_dark):
        return 2, "LARGE_OR_LONG", {
            "large": large, "long": long, "strong_dark": strong_dark,
            "medium_size": medium_size, "medium_len": medium_len,
            "medium_dark": medium_dark,
        }
    if medium_size and medium_len and (medium_dark or not require_dark):
        return 1, "SMALLER_ACTIONABLE", {
            "large": large, "long": long, "strong_dark": strong_dark,
            "medium_size": medium_size, "medium_len": medium_len,
            "medium_dark": medium_dark,
        }
    return 0, "FINE_IGNORED", {
        "large": large, "long": long, "strong_dark": strong_dark,
        "medium_size": medium_size, "medium_len": medium_len,
        "medium_dark": medium_dark,
    }


def build_d20_wrinkle_heatmap(frame: np.ndarray, obs: Optional[BottomObservation],
                               H: np.ndarray, args) -> Optional[WrinkleHeatmapPlan]:
    """Natural-zone filtering + adaptive local-shadow ranking for black pants.

    Processing order:
      1) base texture heatmap,
      2) remove waistband/crotch,
      3) measure local dark-shadow depth,
      4) classify LARGE_LONG_DARK -> LARGE_OR_LONG -> SMALLER_ACTIONABLE,
      5) discard fine wrinkles by default.
    """
    mask_u8 = obs.mask.mask_u8 if obs is not None and obs.mask is not None else None
    base = build_wrinkle_heatmap(frame, mask_u8, args)
    if base is None:
        return None

    ignore, ignore_info = _d19_pose_ignore_mask(obs, H, args)
    if ignore.shape[:2] != base.binary.shape[:2]:
        ignore = cv2.resize(
            ignore, (base.binary.shape[1], base.binary.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )

    effective_inner = base.inner_mask.copy()
    effective_inner[ignore > 0] = 0

    filtered_binary = base.binary.copy()
    filtered_binary[effective_inner == 0] = 0
    filtered_heat = base.heatmap.copy()
    filtered_heat[effective_inner == 0] = 0

    shadow = _d20_local_shadow_map(frame, effective_inner, args)
    shadow_raw = shadow["raw"]
    shadow_norm = shadow["norm"]
    support_raw_threshold = float(shadow["support_raw_threshold"])

    n, labels, stats, cents = cv2.connectedComponentsWithStats(
        (filtered_binary > 0).astype(np.uint8), connectivity=8
    )
    mask_area = (
        float(obs.mask.area_px)
        if obs is not None and obs.mask is not None
        else 1.0
    )

    # Very small connected components are never considered.
    hard_min_area = max(
        float(getattr(args, "wrinkle_heatmap_min_blob_area", 45.0)),
        float(getattr(args, "d20_hard_min_component_area_px", 70.0)),
    )

    dist_map = cv2.distanceTransform(
        (effective_inner > 0).astype(np.uint8), cv2.DIST_L2, 5
    )
    kept = np.zeros_like(filtered_binary)
    candidates: List[Dict[str, Any]] = []
    rejected_fine = 0
    rejected_dark = 0

    for idx in range(1, int(n)):
        area = float(stats[idx, cv2.CC_STAT_AREA])
        if area < hard_min_area:
            rejected_fine += 1
            continue

        comp = labels == idx
        ys_comp, xs_comp = np.where(comp)
        pts = np.column_stack([xs_comp, ys_comp]).astype(np.float32)
        geom = _d19_component_geometry(pts, H)

        major_mm = float(geom.get("major_length_mm", 0.0))
        major_px = float(geom.get("major_length_px", 0.0))
        linearity = float(geom.get("linearity", 1.0))
        area_ratio = area / max(1.0, mask_area)

        dark_vals = shadow_norm[comp]
        dark_raw_vals = shadow_raw[comp]
        dark_mean = float(np.mean(dark_vals)) if dark_vals.size else 0.0
        dark_max = float(np.max(dark_vals)) if dark_vals.size else 0.0
        dark_raw_mean = float(np.mean(dark_raw_vals)) if dark_raw_vals.size else 0.0
        dark_raw_max = float(np.max(dark_raw_vals)) if dark_raw_vals.size else 0.0
        dark_support = (
            float(np.count_nonzero(dark_raw_vals >= support_raw_threshold))
            / float(max(1, dark_raw_vals.size))
        )

        tier, tier_name, tier_checks = _d20_wrinkle_tier(
            area, area_ratio, major_mm, major_px,
            dark_mean, dark_max, dark_support, args,
        )
        if tier <= 0 and bool(getattr(args, "d20_ignore_fine_wrinkles", True)):
            if not tier_checks.get("medium_dark", False):
                rejected_dark += 1
            else:
                rejected_fine += 1
            continue

        # Optional compatibility mode can retain fine components as tier 1.
        if tier <= 0:
            tier = 1
            tier_name = "FINE_COMPAT"

        kept[comp] = 255

        x = int(stats[idx, cv2.CC_STAT_LEFT])
        y = int(stats[idx, cv2.CC_STAT_TOP])
        w = int(stats[idx, cv2.CC_STAT_WIDTH])
        h = int(stats[idx, cv2.CC_STAT_HEIGHT])
        cx, cy = map(float, cents[idx])
        vals = filtered_heat[comp]

        centered = pts - pts.mean(axis=0, keepdims=True)
        major_axis = np.asarray([1.0, 0.0], dtype=np.float32)
        minor_axis = np.asarray([0.0, 1.0], dtype=np.float32)
        if len(pts) >= 5:
            try:
                cov = np.cov(centered.T)
                vals_e, vecs = np.linalg.eigh(cov)
                major_axis = _safe_unit(vecs[:, int(np.argmax(vals_e))])
                minor_axis = np.asarray(
                    [-major_axis[1], major_axis[0]], dtype=np.float32
                )
            except Exception:
                pass

        cxi = int(np.clip(round(cx), 0, dist_map.shape[1] - 1))
        cyi = int(np.clip(round(cy), 0, dist_map.shape[0] - 1))
        mean_score = float(np.mean(vals)) if vals.size else 0.0
        max_score = float(np.max(vals)) if vals.size else 0.0

        strong_area = max(1.0, float(getattr(args, "d20_strong_area_px", 240.0)))
        strong_len = max(1.0, float(getattr(args, "d20_strong_length_mm", 65.0)))
        length_for_score = major_mm if major_mm > 0.0 else major_px * 0.75

        # The million-scale tier term guarantees:
        # large+long+dark first, then major wrinkles, then smaller actionable ones.
        priority = (
            float(tier) * 1_000_000.0
            + min(4.0, area / strong_area) * 15_000.0
            + min(4.0, length_for_score / strong_len) * 12_000.0
            + min(4.0, dark_mean / max(1.0, float(getattr(args, "d20_strong_dark_mean", 38.0)))) * 18_000.0
            + min(4.0, dark_max / max(1.0, float(getattr(args, "d20_strong_dark_max", 105.0)))) * 5_000.0
            + min(4.0, max(1.0, linearity)) * 1_000.0
            + mean_score * 10.0
        )

        candidates.append({
            "area_px": area,
            "area_ratio": float(area_ratio),
            "bbox": [x, y, w, h],
            "center_px": [cx, cy],
            "mean_score": mean_score,
            "max_score": max_score,
            "major_axis_px": [
                float(major_axis[0]), float(major_axis[1])
            ],
            "minor_axis_px": [
                float(minor_axis[0]), float(minor_axis[1])
            ],
            "major_length_px": float(geom.get("major_length_px", 0.0)),
            "minor_length_px": float(geom.get("minor_length_px", 0.0)),
            "major_length_mm": major_mm,
            "minor_length_mm": float(geom.get("minor_length_mm", 0.0)),
            "linearity": linearity,
            "edge_distance_px": float(dist_map[cyi, cxi]),
            "priority_score": float(priority),

            # D20 diagnostics / policy fields.
            "d20_priority_tier": int(tier),
            "d20_tier_name": str(tier_name),
            "d20_shadow_mean": dark_mean,
            "d20_shadow_max": dark_max,
            "d20_shadow_raw_mean": dark_raw_mean,
            "d20_shadow_raw_max": dark_raw_max,
            "d20_shadow_support_ratio": dark_support,
            "d20_shadow_scale_raw": float(shadow["scale_raw"]),
            "d20_significant": True,
        })

    candidates.sort(
        key=lambda c: (
            int(c.get("d20_priority_tier", 0)),
            float(c.get("d20_shadow_mean", 0.0)),
            float(c.get("major_length_mm", 0.0)),
            float(c.get("area_px", 0.0)),
            float(c.get("priority_score", 0.0)),
        ),
        reverse=True,
    )

    max_candidates = max(
        1, int(getattr(args, "wrinkle_heatmap_max_candidates", 6))
    )
    if bool(getattr(args, "d23_waist_structure_veto", True)):
        max_candidates *= max(1, int(getattr(args, "d23_pre_veto_candidate_multiplier", 3)))
    candidates = candidates[:max_candidates]

    denom = max(1, cv2.countNonZero(effective_inner))
    ratio = float(cv2.countNonZero(kept)) / float(denom)
    ignored_ratio = (
        float(cv2.countNonZero(ignore))
        / float(max(1, cv2.countNonZero(base.inner_mask)))
    )
    tier_counts = {
        "large_long_dark": sum(
            1 for c in candidates
            if int(c.get("d20_priority_tier", 0)) == 3
        ),
        "large_or_long": sum(
            1 for c in candidates
            if int(c.get("d20_priority_tier", 0)) == 2
        ),
        "smaller_actionable": sum(
            1 for c in candidates
            if int(c.get("d20_priority_tier", 0)) == 1
        ),
    }

    summary = (
        f"D20 actionable={ratio:.3f} "
        f"L3={tier_counts['large_long_dark']} "
        f"L2={tier_counts['large_or_long']} "
        f"L1={tier_counts['smaller_actionable']} "
        f"ignored_natural={ignored_ratio:.3f} "
        f"fine={rejected_fine} darkweak={rejected_dark} "
        f"shadowScale={float(shadow['scale_raw']):.1f}"
    )

    out = WrinkleHeatmapPlan(
        filtered_heat,
        effective_inner,
        kept,
        candidates,
        base.threshold,
        ratio,
        "OK",
        summary,
    )
    out.d19_ignore_mask = ignore
    out.d19_ignore_info = ignore_info
    out.d20_shadow_map = shadow_norm
    out.d20_shadow_info = {
        k: v for k, v in shadow.items()
        if k not in ("raw", "norm", "local_base")
    }
    out.d20_rejected_fine = int(rejected_fine)
    out.d20_rejected_dark = int(rejected_dark)
    out.d20_tier_counts = tier_counts
    return out


def _d20_print_wrinkle_ranking(heat: Optional[WrinkleHeatmapPlan]) -> None:
    if heat is None:
        print("[D20-RANK] no heatmap")
        return
    print(
        f"[D20-RANK] {heat.summary}"
    )
    if not heat.candidates:
        print("[D20-RANK] actionable wrinkle candidate 없음")
        return
    for i, c in enumerate(heat.candidates[:6], 1):
        print(
            f"[D20-RANK] #{i} tier={c.get('d20_tier_name')} "
            f"area={float(c.get('area_px', 0.0)):.0f}px "
            f"len={float(c.get('major_length_mm', 0.0)):.1f}mm "
            f"darkMean={float(c.get('d20_shadow_mean', 0.0)):.1f} "
            f"darkMax={float(c.get('d20_shadow_max', 0.0)):.1f} "
            f"darkSupport={float(c.get('d20_shadow_support_ratio', 0.0)):.2f}"
        )




# =============================================================================
# D21: wrinkle-geometry classification, edge grip planning, and stabilization
# =============================================================================
D21_MEMORY_LOCK = threading.RLock()
D21_WRINKLE_TRACKS: Dict[int, Dict[str, Any]] = {}
D21_NEXT_TRACK_ID = 1


def _d21_skeletonize_component(component_u8: np.ndarray) -> np.ndarray:
    img = ((component_u8 > 0).astype(np.uint8) * 255)
    skel = np.zeros_like(img)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    for _ in range(256):
        opened = cv2.morphologyEx(img, cv2.MORPH_OPEN, element)
        temp = cv2.subtract(img, opened)
        eroded = cv2.erode(img, element)
        skel = cv2.bitwise_or(skel, temp)
        img = eroded
        if cv2.countNonZero(img) == 0:
            break
    return skel


def _d21_circle_from_three(a: np.ndarray, b: np.ndarray, c: np.ndarray):
    a = np.asarray(a, dtype=np.float64); b = np.asarray(b, dtype=np.float64); c = np.asarray(c, dtype=np.float64)
    d = 2.0 * (a[0]*(b[1]-c[1]) + b[0]*(c[1]-a[1]) + c[0]*(a[1]-b[1]))
    if abs(d) < 1e-8:
        return None
    aa = float(np.dot(a, a)); bb = float(np.dot(b, b)); cc = float(np.dot(c, c))
    ux = (aa*(b[1]-c[1]) + bb*(c[1]-a[1]) + cc*(a[1]-b[1])) / d
    uy = (aa*(c[0]-b[0]) + bb*(a[0]-c[0]) + cc*(b[0]-a[0])) / d
    center = np.asarray([ux, uy], dtype=np.float32)
    radius = float(np.linalg.norm(center - a))
    if not np.isfinite(radius):
        return None
    return center, radius


def _d21_refine_circle(points: np.ndarray):
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    if len(pts) < 3:
        return None
    A = np.column_stack([pts[:, 0], pts[:, 1], np.ones(len(pts))])
    b = -(pts[:, 0] ** 2 + pts[:, 1] ** 2)
    try:
        coef, *_ = np.linalg.lstsq(A, b, rcond=None)
    except Exception:
        return None
    cx, cy = -0.5 * coef[0], -0.5 * coef[1]
    r2 = cx*cx + cy*cy - coef[2]
    if r2 <= 1e-6 or not np.isfinite(r2):
        return None
    return np.asarray([cx, cy], dtype=np.float32), float(math.sqrt(r2))


def _d21_fit_circle_ransac(points_board: np.ndarray, args) -> Dict[str, Any]:
    pts = np.asarray(points_board, dtype=np.float32).reshape(-1, 2)
    if len(pts) < 8:
        return {"ok": False, "reason": "too few skeleton points"}
    max_pts = max(20, int(getattr(args, "d21_fit_max_points", 220)))
    if len(pts) > max_pts:
        step = max(1, int(math.ceil(len(pts) / float(max_pts))))
        pts = pts[::step]
    seed = int(abs(float(np.sum(pts))) * 1000.0) % (2**32 - 1)
    rng = np.random.default_rng(seed)
    iters = max(30, int(getattr(args, "d21_circle_ransac_iters", 120)))
    inlier_tol = float(getattr(args, "d21_circle_inlier_tol_mm", 7.0))
    min_r = float(getattr(args, "d21_curve_min_radius_mm", 25.0))
    max_r = float(getattr(args, "d21_curve_max_radius_mm", 900.0))
    best = None
    for _ in range(iters):
        idx = rng.choice(len(pts), size=3, replace=False)
        circle = _d21_circle_from_three(pts[idx[0]], pts[idx[1]], pts[idx[2]])
        if circle is None:
            continue
        center, radius = circle
        if radius < min_r or radius > max_r:
            continue
        residual = np.abs(np.linalg.norm(pts - center.reshape(1,2), axis=1) - radius)
        inliers = residual <= inlier_tol
        count = int(np.count_nonzero(inliers))
        if count < 5:
            continue
        med = float(np.median(residual[inliers]))
        score = count - 0.15 * med
        if best is None or score > best[0]:
            best = (score, center, radius, inliers, residual)
    if best is None:
        return {"ok": False, "reason": "circle RANSAC failed"}
    _, center, radius, inliers, _ = best
    refined = _d21_refine_circle(pts[inliers])
    if refined is not None:
        center, radius = refined
    residual = np.abs(np.linalg.norm(pts - center.reshape(1,2), axis=1) - radius)
    inliers = residual <= inlier_tol
    return {
        "ok": True,
        "center_board": center,
        "radius_mm": float(radius),
        "median_residual_mm": float(np.median(residual[inliers])) if np.any(inliers) else 999.0,
        "mean_residual_mm": float(np.mean(residual[inliers])) if np.any(inliers) else 999.0,
        "inlier_ratio": float(np.count_nonzero(inliers)) / float(max(1, len(pts))),
    }


def _d21_pca_line(points_board: np.ndarray) -> Dict[str, Any]:
    pts = np.asarray(points_board, dtype=np.float32).reshape(-1, 2)
    if len(pts) < 3:
        return {"ok": False}
    center = pts.mean(axis=0)
    centered = pts - center
    try:
        cov = np.cov(centered.T)
        vals, vecs = np.linalg.eigh(cov)
        tangent = _safe_unit(vecs[:, int(np.argmax(vals))])
    except Exception:
        tangent = np.asarray([1.0, 0.0], dtype=np.float32)
    normal = np.asarray([-tangent[1], tangent[0]], dtype=np.float32)
    residual = np.abs(centered @ normal)
    proj = centered @ tangent
    p0 = center + tangent * float(np.min(proj))
    p1 = center + tangent * float(np.max(proj))
    return {
        "ok": True,
        "center_board": center,
        "tangent_board": tangent,
        "normal_board": normal,
        "median_residual_mm": float(np.median(residual)),
        "mean_residual_mm": float(np.mean(residual)),
        "length_mm": float(np.ptp(proj)),
        "endpoint0_board": p0,
        "endpoint1_board": p1,
    }


def _d21_component_geometry_model(component_u8: np.ndarray, H: np.ndarray, args) -> Dict[str, Any]:
    ys, xs = np.where(component_u8 > 0)
    if len(xs) < 8:
        return {"shape_class": "IRREGULAR", "reason": "component too small"}
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    crop = component_u8[y0:y1, x0:x1]
    skel_crop = _d21_skeletonize_component(crop)
    sy, sx = np.where(skel_crop > 0)
    if len(sx) < 8:
        sx, sy = xs - x0, ys - y0
    pts_px = np.column_stack([sx + x0, sy + y0]).astype(np.float32)
    if len(pts_px) > 500:
        pts_px = pts_px[::max(1, int(math.ceil(len(pts_px)/500.0)))]
    pts_board = cv2.perspectiveTransform(pts_px.reshape(-1,1,2), H).reshape(-1,2)
    line = _d21_pca_line(pts_board)
    circle = _d21_fit_circle_ransac(pts_board, args)
    if not line.get("ok", False):
        return {"shape_class": "IRREGULAR", "reason": "line fit failed"}

    chord_a = np.asarray(line["endpoint0_board"], dtype=np.float32)
    chord_b = np.asarray(line["endpoint1_board"], dtype=np.float32)
    chord = chord_b - chord_a
    chord_len = float(np.linalg.norm(chord))
    chord_u = _safe_unit(chord)
    chord_n = np.asarray([-chord_u[1], chord_u[0]], dtype=np.float32)
    chord_dist = np.abs((pts_board - chord_a.reshape(1,2)) @ chord_n)
    sagitta = float(np.max(chord_dist)) if len(chord_dist) else 0.0
    arc_mid_board = np.asarray(pts_board[int(np.argmax(chord_dist))], dtype=np.float32) if len(chord_dist) else np.asarray(line["center_board"], dtype=np.float32)

    circle_res = float(circle.get("median_residual_mm", 999.0))
    line_res = float(line.get("median_residual_mm", 999.0))
    inlier = float(circle.get("inlier_ratio", 0.0))
    radius = float(circle.get("radius_mm", 1e9))
    arc_len = float(line.get("length_mm", 0.0))
    arc_angle_deg = 0.0
    radial = None
    if circle.get("ok", False):
        cc = np.asarray(circle["center_board"], dtype=np.float32)
        va, vb = chord_a - cc, chord_b - cc
        ua, ub = _safe_unit(va), _safe_unit(vb)
        arc_angle_deg = float(np.degrees(np.arccos(float(np.clip(np.dot(ua, ub), -1.0, 1.0)))))
        arc_len = float(radius * math.radians(arc_angle_deg))
        radial = _safe_unit(cc - arc_mid_board)

    curve_ok = bool(
        circle.get("ok", False)
        and inlier >= float(getattr(args, "d21_curve_min_inlier_ratio", 0.62))
        and circle_res <= float(getattr(args, "d21_curve_max_residual_mm", 8.0))
        and sagitta >= float(getattr(args, "d21_curve_min_sagitta_mm", 10.0))
        and arc_len >= float(getattr(args, "d21_curve_min_arc_length_mm", 50.0))
        and radius <= float(getattr(args, "d21_curve_max_radius_mm", 900.0))
        and circle_res <= max(1.0, line_res * float(getattr(args, "d21_curve_vs_line_error_ratio", 0.90)))
    )
    line_ok = bool(
        line_res <= float(getattr(args, "d21_line_max_residual_mm", 8.0))
        or sagitta <= float(getattr(args, "d21_line_max_sagitta_mm", 10.0))
    )

    if curve_ok and radial is not None:
        shape_class = "CURVE"
        normal = radial
        tangent = np.asarray([-normal[1], normal[0]], dtype=np.float32)
        confidence = float(np.clip(0.55*inlier + 0.45*(1.0-circle_res/max(1.0, float(getattr(args,"d21_curve_max_residual_mm",8.0)))), 0.0, 1.0))
    elif line_ok:
        shape_class = "LINE"
        normal = np.asarray(line["normal_board"], dtype=np.float32)
        tangent = np.asarray(line["tangent_board"], dtype=np.float32)
        confidence = float(np.clip(1.0-line_res/max(1.0,float(getattr(args,"d21_line_max_residual_mm",8.0))),0.0,1.0))
    else:
        shape_class = "IRREGULAR"
        normal = np.asarray(line["normal_board"], dtype=np.float32)
        tangent = np.asarray(line["tangent_board"], dtype=np.float32)
        confidence = 0.0

    return {
        "shape_class": shape_class,
        "center_board": np.asarray(line["center_board"], dtype=np.float32),
        "tangent_board": tangent,
        "normal_board": normal,
        "line_residual_mm": line_res,
        "circle_residual_mm": circle_res,
        "circle_inlier_ratio": inlier,
        "circle_center_board": None if not circle.get("ok", False) else np.asarray(circle["center_board"], dtype=np.float32),
        "circle_radius_mm": radius,
        "arc_mid_board": arc_mid_board,
        "arc_length_mm": arc_len,
        "arc_angle_deg": arc_angle_deg,
        "sagitta_mm": sagitta,
        "chord_length_mm": chord_len,
        "fit_confidence": confidence,
        "skeleton_points_px": pts_px,
    }


def _d21_pose_signature(cand: Dict[str, Any], obs: Optional[BottomObservation], H: np.ndarray) -> Dict[str, float]:
    cp = np.asarray(cand.get("center_px", [0.0,0.0]), dtype=np.float32)
    cb = np.asarray(pixel_to_board(H, float(cp[0]), float(cp[1])), dtype=np.float32)
    out = {"center_x": float(cb[0]), "center_y": float(cb[1]), "pose_s": 0.0, "pose_d": 0.0, "pose_valid": 0.0}
    if obs is not None and obs.pose is not None and obs.pose.valid:
        wc = np.asarray(obs.pose.waist_center, dtype=np.float32)
        lower = np.asarray(obs.pose.lower_center, dtype=np.float32)
        axis = lower - wc; axis_len = max(1.0, float(np.linalg.norm(axis))); axis_u = _safe_unit(axis)
        lat_u = np.asarray([-axis_u[1], axis_u[0]], dtype=np.float32)
        waist = max(1.0, float(obs.pose.waist_width_mm))
        out.update({
            "pose_s": float(np.dot(cb-wc, axis_u)/axis_len),
            "pose_d": float(np.dot(cb-wc, lat_u)/waist),
            "pose_valid": 1.0,
            "axis_len": axis_len,
            "waist_width": waist,
        })
    return out


def _d21_angle_diff_deg(a: float, b: float) -> float:
    return abs(float((float(a)-float(b)+90.0)%180.0-90.0))


def _d21_track_distance(track: Dict[str, Any], sig: Dict[str, float], cand: Dict[str, Any], args) -> float:
    if float(track.get("pose_valid",0.0)) > 0.5 and float(sig.get("pose_valid",0.0)) > 0.5:
        ds = (float(track.get("pose_s",0.0))-float(sig.get("pose_s",0.0))) * float(sig.get("axis_len",250.0))
        dd = (float(track.get("pose_d",0.0))-float(sig.get("pose_d",0.0))) * float(sig.get("waist_width",200.0))
        pos = math.hypot(ds,dd)
    else:
        pos = math.hypot(float(track.get("center_x",0.0))-float(sig.get("center_x",0.0)), float(track.get("center_y",0.0))-float(sig.get("center_y",0.0)))
    length = float(cand.get("major_length_mm",0.0))
    len_rel = abs(length-float(track.get("length_mm",length))) / max(20.0,float(track.get("length_mm",length)))
    angle = math.degrees(math.atan2(float(cand.get("major_axis_px",[1,0])[1]), float(cand.get("major_axis_px",[1,0])[0])))
    ad = _d21_angle_diff_deg(angle,float(track.get("angle_deg",angle)))
    return pos + 30.0*len_rel + 0.5*ad


def _d21_assign_track(cand: Dict[str, Any], obs: Optional[BottomObservation], H: np.ndarray, args) -> Dict[str, Any]:
    global D21_NEXT_TRACK_ID
    sig = _d21_pose_signature(cand, obs, H)
    angle = math.degrees(math.atan2(float(cand.get("major_axis_px",[1,0])[1]), float(cand.get("major_axis_px",[1,0])[0])))
    with D21_MEMORY_LOCK:
        best_id, best_dist = None, 1e9
        for tid, tr in D21_WRINKLE_TRACKS.items():
            d = _d21_track_distance(tr, sig, cand, args)
            if d < best_dist:
                best_dist, best_id = d, tid
        if best_id is None or best_dist > float(getattr(args,"d21_track_match_distance_mm",35.0)):
            best_id = D21_NEXT_TRACK_ID; D21_NEXT_TRACK_ID += 1
            D21_WRINKLE_TRACKS[best_id] = {
                "track_id": best_id, **sig, "length_mm": float(cand.get("major_length_mm",0.0)),
                "area_px": float(cand.get("area_px",0.0)), "dark_mean": float(cand.get("d20_shadow_mean",0.0)),
                "angle_deg": float(angle), "action_count": 0, "unchanged_count": 0,
                "fine_unchanged_count": 0, "resistant_unchanged_count": 0,
                "static": False, "static_reason": "", "resolved": False,
            }
        tr = D21_WRINKLE_TRACKS[best_id]
        tr.update({**sig, "last_seen": time.time(), "length_mm": float(cand.get("major_length_mm",0.0)),
                   "area_px": float(cand.get("area_px",0.0)), "dark_mean": float(cand.get("d20_shadow_mean",0.0)),
                   "angle_deg": float(angle)})
        cand["d21_track_id"] = int(best_id)
        cand["d21_static_intrinsic"] = bool(tr.get("static",False))
        cand["d21_unchanged_count"] = int(tr.get("unchanged_count",0))
        cand["d21_fine_unchanged_count"] = int(tr.get("fine_unchanged_count",0))
        cand["d21_resistant_unchanged_count"] = int(tr.get("resistant_unchanged_count",0))
        return dict(tr)


def _d21_wrinkle_severity(cand: Dict[str, Any], args) -> Tuple[float, float]:
    geom = cand.get("d21_geometry", {}) or {}
    length = float(geom.get("arc_length_mm", cand.get("major_length_mm",0.0))) if geom.get("shape_class") == "CURVE" else float(cand.get("major_length_mm",0.0))
    area = float(cand.get("area_px",0.0)); dark = float(cand.get("d20_shadow_mean",0.0)); dark_max=float(cand.get("d20_shadow_max",0.0))
    sag = float(geom.get("sagitta_mm",0.0)); fit=float(geom.get("fit_confidence",0.0))
    length_s=float(np.clip(length/max(1.0,float(getattr(args,"d21_severity_length_ref_mm",120.0))),0,1))
    area_s=float(np.clip(area/max(1.0,float(getattr(args,"d21_severity_area_ref_px",550.0))),0,1))
    dark_s=float(np.clip(0.7*dark/max(1.0,float(getattr(args,"d21_severity_dark_ref",60.0))) + 0.3*dark_max/max(1.0,float(getattr(args,"d21_severity_dark_max_ref",160.0))),0,1))
    curve_s=float(np.clip(sag/max(1.0,float(getattr(args,"d21_severity_sagitta_ref_mm",45.0))),0,1)) if geom.get("shape_class") == "CURVE" else float(np.clip((float(cand.get("linearity",1.0))-1.0)/3.0,0,1))
    severity=float(np.clip(0.30*length_s+0.25*dark_s+0.20*area_s+0.15*curve_s+0.10*fit,0,1))
    pmin=float(getattr(args,"d21_pull_min_mm",45.0)); pmax=float(getattr(args,"d21_pull_max_mm",140.0))
    pull=float(np.clip(pmin + severity*(pmax-pmin), pmin, pmax))
    return severity,pull



def _d23_waist_structure_guard(
    cand: Dict[str, Any],
    obs: Optional[BottomObservation],
    H: Optional[np.ndarray],
    args,
) -> Dict[str, Any]:
    """Hard-veto a known waistband structural crease.

    Pixel masking alone can split a long waistband shadow and leave a residual
    component immediately below the masked band.  D23 therefore also checks the
    component in pose coordinates.  A line is treated as waistband structure only
    when it is near the waist, spans a meaningful portion of the waist, and runs
    approximately parallel to the detected waist line.
    """
    out = {
        "matched": False,
        "reason": "guard unavailable",
        "axial_mm": 0.0,
        "lateral_mm": 0.0,
        "parallel_error_deg": 180.0,
        "length_mm": float(cand.get("major_length_mm", 0.0)),
    }
    if not bool(getattr(args, "d23_waist_structure_veto", True)):
        out["reason"] = "disabled"
        return out
    if obs is None or obs.pose is None or not obs.pose.valid or H is None:
        out["reason"] = "pose invalid"
        return out

    p = obs.pose
    wl = np.asarray(p.waist_left, dtype=np.float32)
    wc = np.asarray(p.waist_center, dtype=np.float32)
    wr = np.asarray(p.waist_right, dtype=np.float32)
    cr = np.asarray(p.crotch, dtype=np.float32)
    lower = np.asarray(p.lower_center, dtype=np.float32)
    body_u = _safe_unit(lower - wc)
    waist_u = _safe_unit(wr - wl)
    if abs(float(np.dot(body_u, waist_u))) > 0.55:
        waist_u = np.asarray([-body_u[1], body_u[0]], dtype=np.float32)

    cp = np.asarray(cand.get("center_px", [0.0, 0.0]), dtype=np.float32)
    cb = np.asarray(pixel_to_board(H, float(cp[0]), float(cp[1])), dtype=np.float32)
    rel = cb - wc
    axial = float(np.dot(rel, body_u))
    lateral = abs(float(np.dot(rel, waist_u)))

    geom = cand.get("d21_geometry", {}) or {}
    tangent = np.asarray(geom.get("tangent_board", [1.0, 0.0]), dtype=np.float32)
    tangent = _safe_unit(tangent)
    parallel_error = float(np.degrees(np.arccos(np.clip(abs(float(np.dot(tangent, waist_u))), -1.0, 1.0))))
    length = float(
        geom.get("arc_length_mm", cand.get("major_length_mm", 0.0))
        if str(geom.get("shape_class", "")) == "CURVE"
        else cand.get("major_length_mm", 0.0)
    )
    waist_width = max(1.0, float(np.linalg.norm(wr - wl)))
    waist_to_crotch = max(1.0, float(np.linalg.norm(cr - wc)))

    back = max(0.0, float(getattr(args, "d23_waist_structure_back_mm", 24.0)))
    forward = max(
        float(getattr(args, "d23_waist_structure_forward_mm", 92.0)),
        min(
            float(getattr(args, "d23_waist_structure_forward_max_mm", 115.0)),
            waist_to_crotch * float(getattr(args, "d23_waist_structure_forward_ratio", 0.78)),
        ),
    )
    side_limit = 0.5 * waist_width + float(getattr(args, "d23_waist_structure_side_expand_mm", 38.0))
    angle_limit = float(getattr(args, "d23_waist_structure_parallel_deg", 28.0))
    min_length = max(
        float(getattr(args, "d23_waist_structure_min_length_mm", 35.0)),
        waist_width * float(getattr(args, "d23_waist_structure_min_length_ratio", 0.22)),
    )

    checks = {
        "axial": -back <= axial <= forward,
        "lateral": lateral <= side_limit,
        "parallel": parallel_error <= angle_limit,
        "length": length >= min_length,
    }
    matched = bool(all(checks.values()))
    out.update({
        "matched": matched,
        "reason": "pose-relative waistband line" if matched else "not waistband structure",
        "checks": checks,
        "center_board": [float(cb[0]), float(cb[1])],
        "center_px": [float(cp[0]), float(cp[1])],
        "axial_mm": axial,
        "lateral_mm": lateral,
        "parallel_error_deg": parallel_error,
        "length_mm": length,
        "waist_width_mm": waist_width,
        "waist_to_crotch_mm": waist_to_crotch,
        "limits": {
            "back_mm": back,
            "forward_mm": forward,
            "side_limit_mm": side_limit,
            "parallel_deg": angle_limit,
            "min_length_mm": min_length,
        },
    })
    return out


def _d23v2_reference_center(obs: Optional[BottomObservation]) -> Optional[np.ndarray]:
    """Return the center from which all smoothing motions should expand outward."""
    if obs is None:
        return None
    if obs.fused_center_board is not None:
        c = np.asarray(obs.fused_center_board, dtype=np.float32).reshape(2)
        if np.all(np.isfinite(c)):
            return c
    if obs.pose is not None:
        # Keep the reference in the body core rather than at a hem endpoint.
        wc = np.asarray(obs.pose.waist_center, dtype=np.float32)
        cr = np.asarray(obs.pose.crotch, dtype=np.float32)
        lower = np.asarray(obs.pose.lower_center, dtype=np.float32)
        c = 0.25 * wc + 0.40 * cr + 0.35 * lower
        if np.all(np.isfinite(c)):
            return c.astype(np.float32)
    if obs.mask is not None:
        c = np.asarray(obs.mask.center_board, dtype=np.float32).reshape(2)
        if np.all(np.isfinite(c)):
            return c
    return None


def _d23v2_outward_direction(
    reference: np.ndarray,
    point: np.ndarray,
    candidate_axis: np.ndarray,
    args,
) -> Tuple[Optional[np.ndarray], Dict[str, float]]:
    """Choose the sign of a candidate axis that points away from pants center.

    The selected direction is primarily normal to the wrinkle, but it is blended
    with the center->candidate radial vector so a normal that points inward cannot
    be selected merely because the heatmap geometry is ambiguous.
    """
    ref = np.asarray(reference, dtype=np.float32).reshape(2)
    p = np.asarray(point, dtype=np.float32).reshape(2)
    axis = _safe_unit(np.asarray(candidate_axis, dtype=np.float32).reshape(2))
    radial_vec = p - ref
    radial_norm = float(np.linalg.norm(radial_vec))
    if radial_norm < float(getattr(args, "d23v2_min_reference_radius_mm", 12.0)):
        return None, {"radial_norm_mm": radial_norm, "reason_code": -1.0}
    radial = radial_vec / radial_norm
    if float(np.dot(axis, radial)) < 0.0:
        axis = -axis
    normal_weight = float(np.clip(getattr(args, "d23v2_press_normal_weight", 0.78), 0.50, 1.0))
    direction = _safe_unit(normal_weight * axis + (1.0 - normal_weight) * radial)
    outward_cos = float(np.dot(direction, radial))
    min_cos = float(getattr(args, "d23v2_min_outward_cos", 0.15))
    if outward_cos < min_cos:
        return None, {
            "radial_norm_mm": radial_norm,
            "outward_cos": outward_cos,
            "min_outward_cos": min_cos,
            "reason_code": -2.0,
        }
    return direction, {
        "radial_norm_mm": radial_norm,
        "outward_cos": outward_cos,
        "min_outward_cos": min_cos,
        "reason_code": 1.0,
    }


def _d23v2_motion_outward_report(
    reference: np.ndarray,
    source: np.ndarray,
    target: np.ndarray,
    args,
) -> Dict[str, Any]:
    ref = np.asarray(reference, dtype=np.float32).reshape(2)
    src = np.asarray(source, dtype=np.float32).reshape(2)
    dst = np.asarray(target, dtype=np.float32).reshape(2)
    delta = dst - src
    move_mm = float(np.linalg.norm(delta))
    src_r = float(np.linalg.norm(src - ref))
    dst_r = float(np.linalg.norm(dst - ref))
    radial_gain = dst_r - src_r
    radial = _safe_unit(src - ref) if src_r > 1e-6 else _safe_unit(dst - ref)
    direction = _safe_unit(delta) if move_mm > 1e-6 else radial
    outward_cos = float(np.dot(direction, radial)) if move_mm > 1e-6 else 1.0
    min_gain = float(getattr(args, "d23v2_min_radial_gain_mm", 12.0))
    min_cos = float(getattr(args, "d23v2_min_outward_cos", 0.15))
    moving = move_mm >= 3.0
    ok = bool((not moving) or (radial_gain >= min_gain and outward_cos >= min_cos))
    return {
        "ok": ok,
        "moving": moving,
        "move_mm": move_mm,
        "source_radius_mm": src_r,
        "target_radius_mm": dst_r,
        "radial_gain_mm": radial_gain,
        "outward_cos": outward_cos,
        "min_radial_gain_mm": min_gain,
        "min_outward_cos": min_cos,
    }


def _d23v2_validate_center_outward_plan(
    plan: Optional[DualWrinkleStretchPlan],
    obs: Optional[BottomObservation],
    args,
) -> Tuple[bool, Dict[str, Any]]:
    """Final invariant applied to every selected D23_v2 robot plan."""
    report: Dict[str, Any] = {"ok": False, "reason": "plan missing", "arms": {}}
    if plan is None or not plan.ok:
        return False, report
    if not bool(getattr(args, "d23v2_center_outward_policy", True)):
        return True, {"ok": True, "reason": "policy disabled", "arms": {}}
    reference = _d23v2_reference_center(obs)
    if reference is None:
        return False, {"ok": False, "reason": "pants center unavailable", "arms": {}}

    checked = 0
    all_ok = True
    for arm_key, points in plan.arm_points.items():
        if not isinstance(points, dict):
            continue
        role = str(points.get("role", ""))
        if role in ("hold", "hold_press", "passive", "passive_hold"):
            continue
        if "source_board" not in points or "target_board" not in points:
            continue
        arm_report = _d23v2_motion_outward_report(
            reference,
            np.asarray(points["source_board"], dtype=np.float32),
            np.asarray(points["target_board"], dtype=np.float32),
            args,
        )
        report["arms"][arm_key] = {"role": role, **arm_report}
        if arm_report["moving"]:
            checked += 1
            all_ok = all_ok and bool(arm_report["ok"])

    if checked == 0:
        report.update({"ok": False, "reason": "no moving source/target pair"})
        return False, report
    report.update({
        "ok": bool(all_ok),
        "reason": "all moving contacts expand from pants center" if all_ok else "inward/radial-gain guard failed",
        "reference_center_board": [float(reference[0]), float(reference[1])],
        "checked_motion_count": checked,
    })
    return bool(all_ok), report


def build_d21_wrinkle_heatmap(frame: np.ndarray, obs: Optional[BottomObservation], H: np.ndarray, args) -> Optional[WrinkleHeatmapPlan]:
    """Build D21 geometry candidates and separate residual fine wrinkles from robot actions.

    D21_v4 keeps T1/fine wrinkles in diagnostics, but removes them from the motion
    candidate list and action binary.  Therefore a visually acceptable amount of
    shallow residual texture does not force another grip/pull cycle.
    """
    base = build_d20_wrinkle_heatmap(frame, obs, H, args)
    if base is None:
        return None
    n, labels = cv2.connectedComponents((base.binary>0).astype(np.uint8), connectivity=8)
    residual_binary = base.binary.copy()
    action_binary = base.binary.copy()
    all_candidates=[]; action_candidates=[]; fine_candidates=[]; static_ignored=0; waistband_ignored=0
    waistband_guard_reports: List[Dict[str, Any]] = []
    min_action_tier=max(1,int(getattr(args,"d21v4_action_min_tier",2)))
    severity_override=float(getattr(args,"d21v4_action_severity_override",0.55))

    for cand in base.candidates:
        cx,cy = [int(round(float(v))) for v in cand.get("center_px",[0,0])]
        cx=int(np.clip(cx,0,labels.shape[1]-1)); cy=int(np.clip(cy,0,labels.shape[0]-1))
        lid=int(labels[cy,cx])
        if lid <= 0:
            continue
        comp=((labels==lid).astype(np.uint8)*255)
        geom=_d21_component_geometry_model(comp,H,args)
        cand["d21_geometry"]=geom
        cand["d21_shape_class"]=str(geom.get("shape_class","IRREGULAR"))
        cand["d21_component_label"]=lid

        # D23 semantic hard veto: known waistband structure is never sent to the
        # robot and never blocks the finish gate.  This happens before track/static
        # memory so the system does not need to pull the waistband twice to learn it.
        waist_guard = _d23_waist_structure_guard(cand, obs, H, args)
        cand["d23_waist_structure"] = waist_guard
        if bool(waist_guard.get("matched", False)):
            residual_binary[labels==lid] = 0
            action_binary[labels==lid] = 0
            waistband_ignored += 1
            waistband_guard_reports.append(waist_guard)
            print(
                f"[D23-WAIST-VETO] component ignored: axial={float(waist_guard.get('axial_mm',0.0)):.1f}mm "
                f"parallelErr={float(waist_guard.get('parallel_error_deg',180.0)):.1f}deg "
                f"len={float(waist_guard.get('length_mm',0.0)):.1f}mm"
            )
            continue

        tr=_d21_assign_track(cand,obs,H,args)
        severity,pull=_d21_wrinkle_severity(cand,args)
        cand["d21_severity"]=severity; cand["d21_pull_mm"]=pull
        if bool(tr.get("static",False)):
            residual_binary[labels==lid]=0
            action_binary[labels==lid]=0
            static_ignored+=1
            continue

        all_candidates.append(cand)
        tier=int(cand.get("d20_priority_tier",0))
        motion_worthy=bool(tier>=min_action_tier or severity>severity_override)
        cand["d21v4_motion_worthy"]=motion_worthy
        if motion_worthy:
            action_candidates.append(cand)
        else:
            fine_candidates.append(cand)
            action_binary[labels==lid]=0

    sort_key=lambda c:(int(c.get("d20_priority_tier",0)), float(c.get("d21_severity",0.0)), float(c.get("priority_score",0.0)))
    all_candidates.sort(key=sort_key, reverse=True)
    action_candidates.sort(key=sort_key, reverse=True)
    fine_candidates.sort(key=sort_key, reverse=True)

    max_candidates=max(1,int(getattr(args,"wrinkle_heatmap_max_candidates",6)))
    base.binary=action_binary
    base.candidates=action_candidates[:max_candidates]
    base.wrinkle_ratio=float(cv2.countNonZero(action_binary))/float(max(1,cv2.countNonZero(base.inner_mask)))
    base.d21_static_ignored_count=int(static_ignored)
    base.d23_waist_structure_ignored_count=int(waistband_ignored)
    base.d23_waist_structure_reports=waistband_guard_reports
    base.d21v4_all_candidates=all_candidates[:max_candidates]
    base.d21v4_fine_candidates=fine_candidates[:max_candidates]
    base.d21v4_residual_binary=residual_binary
    base.d21v4_residual_ratio=float(cv2.countNonZero(residual_binary))/float(max(1,cv2.countNonZero(base.inner_mask)))
    base.d21v4_fine_ignored_count=len(fine_candidates)
    base.summary=(
        f"D23 actionRatio={base.wrinkle_ratio:.3f} action={len(base.candidates)} "
        f"waistVeto={waistband_ignored} fineAllowed={len(fine_candidates)} "
        f"staticIgnored={static_ignored}; "+base.summary
    )
    return base



def _d21v4_good_enough_finish_report(
    heat: Optional[WrinkleHeatmapPlan],
    obs: Optional[BottomObservation],
    H: Optional[np.ndarray],
    args,
) -> Dict[str, Any]:
    """Allow shallow residual wrinkles while blocking real shape defects.

    Completion is blocked by T3 wrinkles, too many T2 wrinkles, excessive
    severity/actionable area, poor pants shape, or left/right asymmetry.  T1
    wrinkles are intentionally allowed and are not selected for motion.
    """
    enabled=bool(getattr(args,"d21v4_good_enough_finish",True))
    all_candidates=(
        list(getattr(heat,"d21v4_all_candidates",[]))
        if heat is not None else []
    )
    if heat is not None and not all_candidates:
        all_candidates=list(heat.candidates)

    tier3=[c for c in all_candidates if int(c.get("d20_priority_tier",0))>=3]
    tier2=[c for c in all_candidates if int(c.get("d20_priority_tier",0))==2]
    tier1=[c for c in all_candidates if int(c.get("d20_priority_tier",0))<=1]
    strong=tier3+tier2
    max_severity=max([float(c.get("d21_severity",0.0)) for c in strong] or [0.0])

    action_ratio=float(heat.wrinkle_ratio) if heat is not None else 1.0
    mask_area=float(obs.mask.area_px) if obs is not None and obs.mask is not None else 0.0
    max_action_area=max([float(c.get("area_px",0.0)) for c in (heat.candidates if heat is not None else [])] or [0.0])
    max_blob_ratio=max_action_area/max(1.0,mask_area)

    shape=_d19_finish_shape_report(obs,args,H)
    symmetry=shape.get("symmetry",{}) or {}
    shape_score=float(shape.get("shape_score",0.0))
    limits={
        "max_t3":max(0,int(getattr(args,"d21v4_finish_max_t3_count",0))),
        "max_t2":max(0,int(getattr(args,"d21v4_finish_max_t2_count",1))),
        "max_severity":float(getattr(args,"d21v4_finish_max_severity",0.55)),
        "action_ratio":float(getattr(args,"d21v4_finish_actionable_ratio",0.030)),
        "max_blob_ratio":float(getattr(args,"d21v4_finish_max_blob_ratio",0.015)),
        "min_shape_score":float(getattr(args,"d21v4_finish_min_shape_score",0.72)),
    }
    checks={
        "t3":len(tier3)<=limits["max_t3"],
        "t2":len(tier2)<=limits["max_t2"],
        "severity":max_severity<=limits["max_severity"],
        "action_ratio":action_ratio<=limits["action_ratio"],
        "max_blob_ratio":max_blob_ratio<=limits["max_blob_ratio"],
        "shape_score":shape_score>=limits["min_shape_score"],
        "shape_observation":bool(
            obs is not None
            and obs.mask is not None
            and obs.pose is not None
            and bool(obs.pose.valid)
        ),
        "symmetry":bool(symmetry.get("good",False)),
    }
    good=bool(enabled and all(checks.values()))
    remaining=[k for k,v in checks.items() if not v]
    report={
        "good":good,"enabled":enabled,"checks":checks,"limits":limits,
        "t3_count":len(tier3),"t2_count":len(tier2),"t1_allowed_count":len(tier1),
        "max_severity":max_severity,"actionable_ratio":action_ratio,
        "max_blob_ratio":max_blob_ratio,"shape_score":shape_score,
        "symmetry_score":float(symmetry.get("symmetry_score",0.0)),
        "remaining":remaining,
        "summary":(
            f"{'GOOD_ENOUGH' if good else 'NOT_READY'} "
            f"T3={len(tier3)} T2={len(tier2)} T1allowed={len(tier1)} "
            f"sev={max_severity:.2f} actionRatio={action_ratio:.3f} "
            f"maxBlob={max_blob_ratio:.3f} shape={shape_score:.2f} "
            f"sym={float(symmetry.get('symmetry_score',0.0)):.2f} "
            f"lowerIoU={float(symmetry.get('lower_reflected_iou',0.0)):.2f} "
            f"widthMin={float(symmetry.get('lower_width_balance_min',0.0)):.2f}"
        ),
    }
    if heat is not None:
        heat.d21v4_good_enough_report=report
    return report


def _d21v4_print_good_enough_banner(report: Dict[str,Any], required: int, cycle_count: int) -> None:
    print("\n"+"="*82)
    print("바지를 다 폈습니다.")
    print("바지 형태와 좌우대칭이 정상이며, 남은 주름은 허용 가능한 미세 주름입니다.")
    print(f"실용 종료 조건을 {required}회 연속 확인하여 자동 동작을 종료합니다. (사이클 {cycle_count}회)")
    print(f"최종 판정: {report.get('summary','')}")
    print("="*82+"\n")

def _d21_print_geometry_ranking(heat: Optional[WrinkleHeatmapPlan]) -> None:
    if heat is None:
        return
    print(f"[D21-GEOMETRY] {heat.summary}")
    for i,c in enumerate(heat.candidates[:6],1):
        g=c.get("d21_geometry",{}) or {}
        print(f"[D21-GEOMETRY] #{i} track={c.get('d21_track_id')} class={g.get('shape_class')} severity={float(c.get('d21_severity',0)):.2f} pull={float(c.get('d21_pull_mm',0)):.1f}mm lineErr={float(g.get('line_residual_mm',999)):.1f} circleErr={float(g.get('circle_residual_mm',999)):.1f} sagitta={float(g.get('sagitta_mm',0)):.1f} radius={float(g.get('circle_radius_mm',0)):.1f}")


def _d21_contour_board_points(obs: BottomObservation, H: np.ndarray) -> np.ndarray:
    # Use dense boundary pixels rather than CHAIN_APPROX_SIMPLE vertices.  A rectangular
    # mask may otherwise provide only four corners, which can place the grip far from
    # the selected wrinkle.
    mask = ((obs.mask.mask_u8 > 0).astype(np.uint8) * 255)
    eroded = cv2.erode(mask, np.ones((3, 3), np.uint8), iterations=1)
    boundary = cv2.subtract(mask, eroded)
    ys, xs = np.where(boundary > 0)
    if len(xs) < 8:
        pts = np.asarray(obs.mask.contour, dtype=np.float32).reshape(-1, 2)
    else:
        pts = np.column_stack([xs, ys]).astype(np.float32)
        if len(pts) > 12000:
            pts = pts[::max(1, int(math.ceil(len(pts) / 12000.0)))]
    return cv2.perspectiveTransform(pts.reshape(-1, 1, 2), H).reshape(-1, 2)


def _d21_edge_point(contour_board: np.ndarray, center: np.ndarray, tangent: np.ndarray, normal: np.ndarray, sign: float, band_mm: float, inset_mm: float) -> Optional[np.ndarray]:
    rel=contour_board-center.reshape(1,2)
    along=np.abs(rel@tangent)
    side=(rel@normal)*float(sign)
    valid=(along<=band_mm)&(side>5.0)
    if not np.any(valid):
        valid=side>5.0
    if not np.any(valid):
        return None
    valid_idx = np.where(valid)[0]
    side_valid = side[valid_idx]
    max_side = float(np.max(side_valid))
    # First stay near the outermost edge, then choose the point whose tangent
    # coordinate is closest to the wrinkle center.
    edge_tol = 8.0
    near_edge_idx = valid_idx[side_valid >= max_side - edge_tol]
    if len(near_edge_idx) == 0:
        near_edge_idx = valid_idx
    chosen_idx = near_edge_idx[int(np.argmin(along[near_edge_idx]))]
    q = contour_board[chosen_idx]
    return np.asarray(q-normal*float(sign)*inset_mm,dtype=np.float32)


def _d21_make_geometry_plan(obs: BottomObservation, cand: Dict[str,Any], ci: int, H: np.ndarray, config: Dict[str,Any], cfg: BottomSafetyConfig, args) -> Optional[DualWrinkleStretchPlan]:
    geom=cand.get("d21_geometry",{}) or {}; shape=str(geom.get("shape_class","IRREGULAR"))
    if shape not in ("LINE","CURVE"):
        return None
    center=np.asarray(geom.get("arc_mid_board") if shape=="CURVE" else geom.get("center_board"),dtype=np.float32)
    normal=_safe_unit(np.asarray(geom.get("normal_board"),dtype=np.float32)); tangent=_safe_unit(np.asarray(geom.get("tangent_board"),dtype=np.float32))
    contour=_d21_contour_board_points(obs,H)
    band=max(float(getattr(args,"d21_edge_band_mm",65.0)),0.35*float(geom.get("arc_length_mm",cand.get("major_length_mm",0.0))))
    active_in=float(getattr(args,"d21_active_edge_inset_mm",24.0)); hold_in=float(getattr(args,"d21_hold_edge_inset_mm",34.0))
    severity,pull=_d21_wrinkle_severity(cand,args)
    preferred=[1.0,-1.0] if shape=="CURVE" else [1.0,-1.0]
    # CURVE normal already points toward fitted circle center, so +1 is preferred.
    best=None
    for sign in preferred:
        active=_d21_edge_point(contour,center,tangent,normal,sign,band,active_in)
        hold=_d21_edge_point(contour,center,tangent,normal,-sign,band,hold_in)
        if active is None or hold is None:
            continue
        active_arm=select_arm_for_board_x(cfg,float(active[0])); hold_arm=select_arm_for_board_x(cfg,float(hold[0]))
        if active_arm is None or hold_arm is None or active_arm==hold_arm:
            continue
        active_snap=find_nearest_mask_board_point(obs.mask,H,active,cfg,preferred_arm=active_arm,max_radius_mm=float(getattr(args,"d21_edge_snap_radius_mm",42.0)))
        hold_snap=find_nearest_mask_board_point(obs.mask,H,hold,cfg,preferred_arm=hold_arm,max_radius_mm=float(getattr(args,"d21_edge_snap_radius_mm",42.0)))
        if active_snap is None or hold_snap is None:
            continue
        active=np.asarray(active_snap,dtype=np.float32); hold=np.asarray(hold_snap,dtype=np.float32)
        direction=normal*float(sign)
        reference = _d23v2_reference_center(obs)
        if reference is None:
            continue
        radial = _safe_unit(active - reference)
        outward_cos = float(np.dot(_safe_unit(direction), radial))
        if bool(getattr(args, "d23v2_center_outward_policy", True)) and outward_cos < float(getattr(args, "d23v2_min_outward_cos", 0.15)):
            continue
        target=clamp_board_point(cfg,active+direction*pull,margin=float(getattr(args,"b34_board_margin_mm",35.0)))
        if select_arm_for_board_x(cfg,float(target[0]))!=active_arm:
            continue
        actual=float(np.dot(target-active,direction))
        sep0=float(np.linalg.norm(active-hold)); sep1=float(np.linalg.norm(target-hold))
        if actual<float(getattr(args,"d21_min_actual_pull_mm",35.0)) or sep1-sep0<float(getattr(args,"d21_min_separation_gain_mm",25.0)):
            continue
        radial_report = _d23v2_motion_outward_report(reference, active, target, args)
        if bool(getattr(args, "d23v2_center_outward_policy", True)) and not bool(radial_report.get("ok", False)):
            continue
        score=actual+0.5*(sep1-sep0)+80.0*max(0.0,outward_cos)+(35.0 if shape=="CURVE" and sign>0 else 0.0)
        if best is None or score>best[0]:
            best=(score,active,hold,target,active_arm,hold_arm,direction,actual,sep0,sep1,outward_cos,radial_report)
    if best is None:
        return None
    _,active,hold,target,active_arm,hold_arm,direction,actual,sep0,sep1,outward_cos,radial_report=best
    active_contact=float(getattr(args,"d21_active_grip_z",arm_contact_z(cfg,active_arm)) if getattr(args,"d21_active_grip_z",None) is not None else arm_contact_z(cfg,active_arm))
    hold_z=float(getattr(args,"d21_hold_press_z",-104.0)); lift_mm=float(getattr(args,"d21_grip_test_lift_mm",8.0)); pull_z=active_contact+lift_mm
    clear_z=float(getattr(args,"b34_clear_z",-25.0)); mid=active+0.55*(target-active); release_z=active_contact+float(getattr(args,"d21_release_above_contact_mm",2.0))
    arm_points={
      active_arm:{"role":"active_grip","source_board":tuple(map(float,active)),"target_board":tuple(map(float,target)),
                  "hover":_roarm_pose_for_board(config,cfg,active_arm,active,clear_z),"down":_roarm_pose_for_board(config,cfg,active_arm,active,active_contact),
                  "test_lift":_roarm_pose_for_board(config,cfg,active_arm,active,pull_z),"mid":_roarm_pose_for_board(config,cfg,active_arm,mid,pull_z),
                  "sweep":_roarm_pose_for_board(config,cfg,active_arm,target,pull_z),"release":_roarm_pose_for_board(config,cfg,active_arm,target,release_z),
                  "lift":_roarm_pose_for_board(config,cfg,active_arm,target,clear_z)},
      hold_arm:{"role":"hold_press","source_board":tuple(map(float,hold)),"target_board":tuple(map(float,hold)),
                "hover":_roarm_pose_for_board(config,cfg,hold_arm,hold,clear_z),"down":_roarm_pose_for_board(config,cfg,hold_arm,hold,hold_z),
                "test_lift":_roarm_pose_for_board(config,cfg,hold_arm,hold,hold_z),"mid":_roarm_pose_for_board(config,cfg,hold_arm,hold,hold_z),
                "sweep":_roarm_pose_for_board(config,cfg,hold_arm,hold,hold_z),"release":_roarm_pose_for_board(config,cfg,hold_arm,hold,hold_z),
                "lift":_roarm_pose_for_board(config,cfg,hold_arm,hold,clear_z)} }
    action="D21_ARC_RADIAL_GRIP_PULL" if shape=="CURVE" else "D21_LINE_NORMAL_GRIP_PULL"
    return DualWrinkleStretchPlan(True,f"D21 {shape} geometry grip-pull",action=action,candidate_index=ci,center_board=tuple(map(float,center)),normal_axis_board=tuple(map(float,direction)),arm_points=arm_points,metrics={
        "policy_family":"D21_GEOMETRY_GRIP_PULL","d21_shape_class":shape,"d21_track_id":int(cand.get("d21_track_id",-1)),
        "d21_severity":severity,"d20_priority_tier":int(cand.get("d20_priority_tier",3)),"d20_tier_name":str(cand.get("d20_tier_name","")),
        "d21_requested_pull_mm":pull,"d21_actual_pull_mm":actual,"d21_initial_pair_sep_mm":sep0,"d21_final_pair_sep_mm":sep1,
        "d23v2_outward_cos":outward_cos,"d23v2_motion_report":radial_report,
        "active_arm":active_arm,"hold_arm":hold_arm,"d21_candidate_area_px":float(cand.get("area_px",0.0)),
        "d21_candidate_length_mm":float(cand.get("major_length_mm",0.0)),"d21_candidate_dark_mean":float(cand.get("d20_shadow_mean",0.0)),
        "d21_signature":_d21_pose_signature(cand,obs,H),"d21_geometry":{k:(v.tolist() if isinstance(v,np.ndarray) else v) for k,v in geom.items() if k!="skeleton_points_px"}
    })


def build_d21_geometry_grip_plan(obs: BottomObservation, heat: Optional[WrinkleHeatmapPlan], H: np.ndarray, config: Dict[str,Any], cfg: BottomSafetyConfig, args) -> Optional[DualWrinkleStretchPlan]:
    if heat is None or not heat.candidates:
        return None
    proposals=[]
    for ci,cand in enumerate(heat.candidates):
        plan=_d21_make_geometry_plan(obs,cand,ci,H,config,cfg,args)
        if plan is not None and plan.ok:
            proposals.append((float(cand.get("d21_severity",0.0)),plan))
    if not proposals:
        return None
    proposals.sort(key=lambda x:x[0],reverse=True)
    return proposals[0][1]



def _d22_board_axis_to_pixel(
    H: np.ndarray,
    center_board: Sequence[float],
    axis_board: Sequence[float],
) -> Optional[np.ndarray]:
    try:
        c = np.asarray(center_board, dtype=np.float32)
        a = _safe_unit(np.asarray(axis_board, dtype=np.float32))
        p0 = board_to_pixel(H, float(c[0]), float(c[1]))
        p1 = board_to_pixel(H, float(c[0] + 20.0 * a[0]), float(c[1] + 20.0 * a[1]))
        if p0 is None or p1 is None:
            return None
        return _safe_unit(np.asarray(p1, dtype=np.float32) - np.asarray(p0, dtype=np.float32))
    except Exception:
        return None


def _d22_wrinkle_action_mode(cand: Dict[str, Any], args) -> Tuple[str, Dict[str, Any]]:
    geom = cand.get("d21_geometry", {}) or {}
    shape = str(geom.get("shape_class", "IRREGULAR"))
    severity = float(cand.get("d21_severity", 0.0))
    tier = int(cand.get("d20_priority_tier", 0))
    length = float(geom.get("arc_length_mm", cand.get("major_length_mm", 0.0))) if shape == "CURVE" else float(cand.get("major_length_mm", 0.0))
    dark = float(cand.get("d20_shadow_mean", 0.0))
    fit = float(geom.get("fit_confidence", 0.0))

    # D23_v2: pressing is the normal action.  Opening/gripping is reserved for
    # an unmistakably severe geometric wrinkle, because a weak grip action can
    # bunch a trouser leg instead of flattening it.
    grip_tier = int(getattr(args, "d23v2_grip_min_tier", 3))
    grip_severity = float(getattr(args, "d23v2_grip_min_severity", 0.72))
    grip_length = float(getattr(args, "d23v2_grip_min_length_mm", 90.0))
    grip_dark = float(getattr(args, "d23v2_grip_min_dark_mean", 30.0))
    grip_fit = float(getattr(args, "d23v2_grip_min_fit_confidence", 0.35))
    severe_all = bool(
        tier >= grip_tier
        and severity >= grip_severity
        and length >= grip_length
        and dark >= grip_dark
        and fit >= grip_fit
    )
    old_style_strong = bool(
        tier >= int(getattr(args, "d22_grip_min_tier", 3))
        or (
            severity >= float(getattr(args, "d22_grip_min_severity", 0.58))
            and length >= float(getattr(args, "d22_grip_min_length_mm", 70.0))
            and dark >= float(getattr(args, "d22_grip_min_dark_mean", 24.0))
        )
    )
    require_all = bool(getattr(args, "d23v2_grip_require_all", True))
    strong = severe_all if require_all else old_style_strong
    fit_ok = fit >= (grip_fit if require_all else float(getattr(args, "d22_grip_min_fit_confidence", 0.25)))
    mode = "GRIP_PULL" if shape in ("LINE", "CURVE") and strong and fit_ok else "PRESS_SWEEP"
    return mode, {
        "shape": shape,
        "severity": severity,
        "tier": tier,
        "length_mm": length,
        "dark_mean": dark,
        "fit_confidence": fit,
        "strong": strong,
        "fit_ok": fit_ok,
        "severe_all": severe_all,
        "grip_require_all": require_all,
    }

def _d22_make_press_sweep_plan(
    obs: BottomObservation,
    heat: Optional[WrinkleHeatmapPlan],
    cand: Dict[str, Any],
    ci: int,
    H: np.ndarray,
    config: Dict[str, Any],
    cfg: BottomSafetyConfig,
    args,
) -> Optional[DualWrinkleStretchPlan]:
    """Build a closed-gripper press that starts on the center side and moves outward."""
    if obs is None or obs.mask is None or H is None:
        return None
    reference = _d23v2_reference_center(obs)
    if reference is None:
        return None

    geom = cand.get("d21_geometry", {}) or {}
    cp = np.asarray(cand.get("center_px", [0.0, 0.0]), dtype=np.float32)
    if "center_board" in geom:
        problem = np.asarray(geom.get("arc_mid_board") if geom.get("shape_class") == "CURVE" else geom.get("center_board"), dtype=np.float32)
    else:
        problem = np.asarray(pixel_to_board(H, float(cp[0]), float(cp[1])), dtype=np.float32)

    tangent = _safe_unit(np.asarray(geom.get("tangent_board", [1.0, 0.0]), dtype=np.float32))
    normal = np.asarray([-tangent[1], tangent[0]], dtype=np.float32)
    geom_normal = geom.get("normal_board")
    if geom_normal is not None:
        normal = _safe_unit(np.asarray(geom_normal, dtype=np.float32))

    direction, direction_report = _d23v2_outward_direction(reference, problem, normal, args)
    if direction is None:
        return None

    severity = float(cand.get("d21_severity", 0.0))
    dmin = float(getattr(args, "d22_press_sweep_min_mm", 45.0))
    dmax = float(getattr(args, "d22_press_sweep_max_mm", 85.0))
    distance = float(np.clip(dmin + severity * (dmax - dmin), dmin, dmax))
    inward_start = float(getattr(args, "d23v2_press_start_inward_mm", 24.0))
    desired_start = problem - direction * inward_start
    target_guess = problem + direction * max(10.0, distance - inward_start)

    preferred_arm = select_arm_for_board_x(cfg, float(target_guess[0]))
    if preferred_arm is None:
        preferred_arm = select_arm_for_board_x(cfg, float(problem[0]))
    if preferred_arm is None:
        return None

    start_point = find_nearest_mask_board_point(
        obs.mask, H, desired_start, cfg,
        preferred_arm=preferred_arm,
        max_radius_mm=float(getattr(args, "d23v2_press_snap_radius_mm", 75.0)),
    )
    if start_point is None:
        start_point = find_nearest_mask_board_point(
            obs.mask, H, problem, cfg,
            preferred_arm=preferred_arm,
            max_radius_mm=float(getattr(args, "d23v2_press_snap_radius_mm", 75.0)),
        )
    if start_point is None:
        return None
    start_point = np.asarray(start_point, dtype=np.float32)

    margin = float(getattr(args, "b34_board_margin_mm", 35.0))
    target = clamp_board_point(cfg, start_point + direction * distance, margin=margin)
    if select_arm_for_board_x(cfg, float(target[0])) != preferred_arm:
        # Shorten rather than cross the center workspace boundary.
        found = None
        for scale in (0.80, 0.65, 0.50, 0.38):
            q = clamp_board_point(cfg, start_point + direction * (distance * scale), margin=margin)
            if select_arm_for_board_x(cfg, float(q[0])) == preferred_arm:
                found = q
                break
        if found is None:
            return None
        target = found

    motion_report = _d23v2_motion_outward_report(reference, start_point, target, args)
    if not bool(motion_report.get("ok", False)):
        return None
    actual = float(np.linalg.norm(target - start_point))
    min_actual = float(getattr(args, "d22_press_min_actual_mm", 20.0))
    if actual < min_actual:
        return None

    press_z = float(getattr(args, "b34_press_z", -104.0))
    clear_z = float(getattr(args, "b34_clear_z", -25.0))
    mid_ratio = float(np.clip(getattr(args, "b34_micro_mid_ratio", 0.55), 0.25, 0.80))
    mid = start_point + (target - start_point) * mid_ratio
    arm_points = {
        preferred_arm: {
            "role": "center_out_press",
            "detected_source_board": (float(problem[0]), float(problem[1])),
            "source_board": (float(start_point[0]), float(start_point[1])),
            "target_board": (float(target[0]), float(target[1])),
            "hover": _roarm_pose_for_board(config, cfg, preferred_arm, start_point, clear_z),
            "down": _roarm_pose_for_board(config, cfg, preferred_arm, start_point, press_z),
            "mid": _roarm_pose_for_board(config, cfg, preferred_arm, mid, press_z),
            "sweep": _roarm_pose_for_board(config, cfg, preferred_arm, target, press_z),
            "lift": _roarm_pose_for_board(config, cfg, preferred_arm, target, clear_z),
        }
    }
    metrics = _dual_plan_metrics(heat, obs)
    metrics.update({
        "policy_family": "D23V2_CENTER_OUT",
        "d22_motion_mode": "PRESS_SWEEP",
        "d23v2_motion_mode": "CLOSED_PRESS_OUTWARD",
        "d23v2_reference_center_board": [float(reference[0]), float(reference[1])],
        "d23v2_direction_report": direction_report,
        "d23v2_motion_report": motion_report,
        "d23v2_press_gripper": float(getattr(args, "d23v2_press_gripper", 2.95)),
        "d22_press_distance_mm": actual,
        "active_arm": preferred_arm,
        "d21_track_id": int(cand.get("d21_track_id", -1)),
        "d21_signature": _d21_pose_signature(cand, obs, H),
        "d21_shape_class": str(geom.get("shape_class", "IRREGULAR")),
        "d21_severity": severity,
        "d20_priority_tier": int(cand.get("d20_priority_tier", 0)),
        "d21_candidate_area_px": float(cand.get("area_px", 0.0)),
        "d21_candidate_length_mm": float(cand.get("major_length_mm", 0.0)),
        "d21_candidate_dark_mean": float(cand.get("d20_shadow_mean", 0.0)),
        "single_arm_press_allowed": True,
    })
    return DualWrinkleStretchPlan(
        True,
        f"D23_v2 center-side PRESS -> outward; severity={severity:.2f} move={actual:.1f}mm",
        action="D23V2_CENTER_OUT_PRESS",
        candidate_index=ci,
        center_px=(float(cp[0]), float(cp[1])),
        center_board=(float(problem[0]), float(problem[1])),
        normal_axis_board=(float(direction[0]), float(direction[1])),
        arm_points=arm_points,
        metrics=metrics,
    )

def build_d22_hybrid_wrinkle_plan(
    obs: BottomObservation,
    heat: Optional[WrinkleHeatmapPlan],
    H: np.ndarray,
    config: Dict[str, Any],
    cfg: BottomSafetyConfig,
    args,
) -> Optional[DualWrinkleStretchPlan]:
    """Choose grip-pull or press-sweep from wrinkle geometry and severity."""
    if heat is None or not heat.candidates:
        return None
    proposals: List[Tuple[float, DualWrinkleStretchPlan]] = []
    warnings: List[str] = []
    for ci, cand in enumerate(heat.candidates):
        mode, decision = _d22_wrinkle_action_mode(cand, args)
        plan = None
        if mode == "GRIP_PULL":
            plan = _d21_make_geometry_plan(obs, cand, ci, H, config, cfg, args)
            if plan is not None and plan.ok:
                outward_ok, outward_report = _d23v2_validate_center_outward_plan(plan, obs, args)
                plan.metrics["d23v2_center_outward_validation"] = outward_report
                if not outward_ok:
                    warnings.append(f"W{ci+1}: inward GRIP_PULL rejected; fallback PRESS_SWEEP")
                    plan = None
                else:
                    plan.metrics.update({
                        "policy_family": "D23V2_HYBRID",
                        "d22_motion_mode": "GRIP_PULL",
                        "d22_decision": decision,
                    })
                    plan.reason = f"D23_v2 severe {decision['shape']} -> outward GRIP_PULL; " + plan.reason
            if (plan is None or not plan.ok) and bool(getattr(args, "d22_grip_fallback_to_press", True)):
                warnings.append(f"W{ci+1}: GRIP_PULL unavailable; fallback PRESS_SWEEP")
                plan = _d22_make_press_sweep_plan(obs, heat, cand, ci, H, config, cfg, args)
        else:
            plan = _d22_make_press_sweep_plan(obs, heat, cand, ci, H, config, cfg, args)

        if plan is None or not plan.ok:
            warnings.append(f"W{ci+1}: {mode} plan unavailable")
            continue
        score = (
            1_000_000.0 * int(cand.get("d20_priority_tier", 0))
            + 100_000.0 * float(cand.get("d21_severity", 0.0))
            + float(cand.get("priority_score", 0.0))
            + (20000.0 if str(plan.metrics.get("d22_motion_mode")) == "GRIP_PULL" else 0.0)
        )
        plan.metrics["d22_policy_score"] = float(score)
        plan.safety_warnings = list(warnings)
        proposals.append((score, plan))
    if not proposals:
        return None
    proposals.sort(key=lambda item: item[0], reverse=True)
    return proposals[0][1]


def _d22_build_lower_leg_asymmetry_plan(
    obs: BottomObservation,
    H: np.ndarray,
    config: Dict[str, Any],
    cfg: BottomSafetyConfig,
    args,
    symmetry: Dict[str, Any],
    shape_metrics: Dict[str, Any],
) -> Optional[DualWrinkleStretchPlan]:
    """Correct a narrow/folded lower-leg band with a center-to-outside press.

    D22 used an active grip plus passive hold here.  On loose trouser legs that
    can drag fabric inward.  D23_v2 starts on the center side of the narrow band,
    keeps the gripper closed, and presses toward the outer leg boundary.
    """
    if not bool(getattr(args, "d22_asymmetry_correction", True)):
        return None
    if not isinstance(symmetry, dict) or bool(symmetry.get("good", False)):
        return None
    worst = symmetry.get("worst_band")
    if not isinstance(worst, dict):
        return None

    active_raw = np.asarray(worst.get("active_board"), dtype=np.float32)
    axis_center = np.asarray(worst.get("center_board"), dtype=np.float32)
    if active_raw.shape != (2,) or axis_center.shape != (2,):
        return None
    outward = _safe_unit(active_raw - axis_center)
    if float(np.linalg.norm(active_raw - axis_center)) < 5.0:
        return None
    tangent = np.asarray([-outward[1], outward[0]], dtype=np.float32)
    px = board_to_pixel(H, float(active_raw[0]), float(active_raw[1]))
    if px is None:
        return None

    imbalance = float(np.clip(1.0 - float(worst.get("balance", 0.0)), 0.0, 1.0))
    pseudo = {
        "center_px": [float(px[0]), float(px[1])],
        "major_axis_px": [1.0, 0.0],
        "major_length_mm": float(getattr(args, "d23v2_asymmetry_virtual_length_mm", 70.0)),
        "area_px": float(max(100.0, 450.0 * imbalance)),
        "d20_shadow_mean": 0.0,
        "d20_priority_tier": 2,
        "d21_severity": float(np.clip(0.35 + 0.55 * imbalance, 0.0, 1.0)),
        "d21_geometry": {
            "shape_class": "IRREGULAR",
            "center_board": active_raw,
            "tangent_board": tangent,
            "normal_board": outward,
            "fit_confidence": 1.0,
        },
    }
    plan = _d22_make_press_sweep_plan(obs, None, pseudo, -1, H, config, cfg, args)
    if plan is None or not plan.ok:
        return None
    plan.action = "D23V2_LOWER_LEG_OUTWARD_PRESS"
    plan.reason = (
        f"D23_v2 lower-leg narrow band -> closed PRESS outward; "
        f"balance={float(worst.get('balance', 0.0)):.2f}"
    )
    plan.metrics.update(dict(shape_metrics))
    plan.metrics.update({
        "policy_family": "D23V2_LOWER_LEG_SHAPE_PRESS",
        "primary_issue": "LOWER_LEG_MASK_ASYMMETRY",
        "d22_worst_band_balance": float(worst.get("balance", 0.0)),
        "symmetry": symmetry,
    })
    return plan

def _d21_execute_geometry_grip_pull(plan: DualWrinkleStretchPlan, arms: Dict[str,RoArmSerial], cfg: BottomSafetyConfig, move_command: int, args) -> bool:
    active=str(plan.metrics.get("active_arm","")); hold=str(plan.metrics.get("hold_arm",""))
    if active not in arms or hold not in arms or active==hold:
        return False
    ap=plan.arm_points[active]; hp=plan.arm_points[hold]
    fast=bool(getattr(args,"d23_fast_motion",True))
    wait_scale=float(getattr(args,"d23_motion_wait_scale",0.55)) if fast else 1.0
    speed=float(getattr(args,"d21_grip_pull_speed",0.20))
    if fast:
        speed=max(speed,float(getattr(args,"d23_grip_pull_speed_min",0.35)))
    active_open=float(getattr(args,"d21_active_grip_open",1.35)); active_close=float(getattr(args,"d21_active_grip_close",3.05)); hold_angle=float(getattr(args,"d21_hold_gripper",3.05))
    repeat=max(1,int(getattr(args,"d21_active_close_repeat",2)))
    if fast:
        repeat=min(repeat,max(1,int(getattr(args,"d23_fast_close_repeat",1))))

    def wait_arg(name, default, floor=0.06):
        value=max(0.0,float(getattr(args,name,default)))
        return max(float(floor),value*wait_scale) if fast else value

    def move(arm,key,label):
        xyz=plan.arm_points[arm][key]; print(f"[D23-{label}] {arm.upper()} ({xyz[0]:.1f},{xyz[1]:.1f},{xyz[2]:.1f})")
        arms[arm].move_goal(float(xyz[0]),float(xyz[1]),float(xyz[2]),cfg.tool_t_rad,speed,move_command=move_command)

    try:
        _run_parallel([
            ("active-open",lambda: arms[active].set_gripper(active_open,delay=0.12 if fast else 0.18)),
            ("hold-set",lambda: arms[hold].set_gripper(hold_angle,delay=0.12 if fast else 0.18)),
        ])

        # High-Z approaches can be sent in parallel.  Low-Z contact remains hold-first.
        if bool(getattr(args,"d23_parallel_high_z",True)):
            _run_parallel([
                ("hold-hover",lambda: move(hold,"hover","HOLD-HOVER")),
                ("active-hover",lambda: move(active,"hover","ACTIVE-HOVER")),
            ])
            time.sleep(max(
                wait_arg("d21_stage_wait_s",0.70,0.16),
                float(getattr(args,"d23_high_z_wait_s",0.55)),
            ))
        else:
            move(hold,"hover","HOLD-HOVER"); time.sleep(wait_arg("d21_stage_wait_s",0.70,0.16))
            move(active,"hover","ACTIVE-HOVER"); time.sleep(wait_arg("d21_stage_wait_s",0.70,0.16))

        if cfg.hover_only:
            return True
        move(hold,"down","HOLD-PRESS"); time.sleep(wait_arg("d21_hold_settle_s",0.65,0.20))
        move(active,"down","ACTIVE-DOWN"); time.sleep(wait_arg("d21_down_wait_s",0.65,0.20))
        for i in range(repeat):
            print(f"[D23-ACTIVE-GRIP] close {i+1}/{repeat} angle={active_close:.3f}")
            arms[active].set_gripper(active_close,delay=0.14 if fast else 0.20)
            time.sleep(wait_arg("d21_close_wait_s",0.30,0.10))
        time.sleep(wait_arg("d21_close_settle_s",0.45,0.14))
        move(active,"test_lift","TEST-LIFT"); time.sleep(wait_arg("d21_test_lift_wait_s",0.45,0.16))
        move(active,"mid","MID-PULL"); time.sleep(wait_arg("d21_pull_wait_s",0.65,0.20))
        move(active,"sweep","FULL-PULL"); time.sleep(wait_arg("d21_pull_wait_s",0.65,0.20))
        time.sleep(wait_arg("d21_tension_hold_s",0.40,0.12))
        move(active,"release","LOW-RELEASE"); time.sleep(wait_arg("d21_release_settle_s",0.30,0.10))
        arms[active].set_gripper(active_open,delay=0.14 if fast else 0.20)
        time.sleep(wait_arg("d21_release_open_wait_s",0.35,0.12))
        move(active,"lift","ACTIVE-CLEAR"); time.sleep(wait_arg("d21_clear_wait_s",0.55,0.18))
        move(hold,"lift","HOLD-CLEAR-LAST"); time.sleep(wait_arg("d21_clear_wait_s",0.55,0.18))
        return True
    except Exception as exc:
        print(f"[D23-GRIP-PULL-ERROR] {exc!r}"); return False

def _d21_find_after_candidate(before_sig: Dict[str,float], after_heat: Optional[WrinkleHeatmapPlan], after_obs: Optional[BottomObservation], H: np.ndarray, args):
    if after_heat is None:
        return None
    pseudo={"pose_s":before_sig.get("pose_s",0.0),"pose_d":before_sig.get("pose_d",0.0),"pose_valid":before_sig.get("pose_valid",0.0),"center_x":before_sig.get("center_x",0.0),"center_y":before_sig.get("center_y",0.0),"axis_len":before_sig.get("axis_len",250.0),"waist_width":before_sig.get("waist_width",200.0),"length_mm":0.0,"angle_deg":0.0}
    best=None
    for c in after_heat.candidates:
        sig=_d21_pose_signature(c,after_obs,H)
        if pseudo["pose_valid"]>0.5 and sig["pose_valid"]>0.5:
            pos=math.hypot((pseudo["pose_s"]-sig["pose_s"])*sig.get("axis_len",250.0),(pseudo["pose_d"]-sig["pose_d"])*sig.get("waist_width",200.0))
        else:
            pos=math.hypot(pseudo["center_x"]-sig["center_x"],pseudo["center_y"]-sig["center_y"])
        if best is None or pos<best[0]: best=(pos,c,sig)
    if best is None or best[0]>float(getattr(args,"d21_track_match_distance_mm",35.0)):
        return None
    return best



def _d21v3_fine_wrinkle_report(
    before_metrics: Dict[str, Any],
    after_candidate: Dict[str, Any],
    args,
) -> Dict[str, Any]:
    """Decide whether an unchanged wrinkle is mild enough to become intrinsic.

    Unchanged alone is not sufficient.  D21_v3 requires the wrinkle to remain
    below configurable tier, severity, length, area and relative-darkness limits.
    This prevents a failed manipulation on a severe wrinkle from being learned
    as a garment-specific structural wrinkle.
    """
    require_fine = bool(getattr(args, "d21v3_static_require_fine", True))

    before_tier = int(before_metrics.get("d20_priority_tier", 3))
    after_tier = int(after_candidate.get("d20_priority_tier", before_tier))
    before_severity = float(before_metrics.get("d21_severity", 1.0))
    after_severity = float(after_candidate.get("d21_severity", before_severity))

    before_length = float(before_metrics.get("d21_candidate_length_mm", 0.0))
    after_length = float(after_candidate.get("major_length_mm", before_length))
    before_area = float(before_metrics.get("d21_candidate_area_px", 0.0))
    after_area = float(after_candidate.get("area_px", before_area))
    before_dark = float(before_metrics.get("d21_candidate_dark_mean", 0.0))
    after_dark = float(after_candidate.get("d20_shadow_mean", before_dark))

    values = {
        "max_tier": max(before_tier, after_tier),
        "max_severity": max(before_severity, after_severity),
        "max_length_mm": max(before_length, after_length),
        "max_area_px": max(before_area, after_area),
        "max_dark_mean": max(before_dark, after_dark),
    }
    limits = {
        "max_tier": int(getattr(args, "d21v3_static_max_tier", 1)),
        "max_severity": float(getattr(args, "d21v3_static_max_severity", 0.45)),
        "max_length_mm": float(getattr(args, "d21v3_static_max_length_mm", 65.0)),
        "max_area_px": float(getattr(args, "d21v3_static_max_area_px", 240.0)),
        "max_dark_mean": float(getattr(args, "d21v3_static_max_dark_mean", 38.0)),
    }
    checks = {
        "tier": values["max_tier"] <= limits["max_tier"],
        "severity": values["max_severity"] <= limits["max_severity"],
        "length": values["max_length_mm"] <= limits["max_length_mm"],
        "area": values["max_area_px"] <= limits["max_area_px"],
        "darkness": values["max_dark_mean"] <= limits["max_dark_mean"],
    }
    fine = bool(all(checks.values())) if require_fine else True
    return {
        "fine": fine,
        "require_fine": require_fine,
        "checks": checks,
        "values": values,
        "limits": limits,
    }


def _d21v3_remaining_wrinkle_gate(
    heat: Optional[WrinkleHeatmapPlan],
    args,
    newly_static_track_id: int = -1,
) -> Dict[str, Any]:
    """Block stable-shape finish while any severe/non-fine wrinkle remains."""
    candidates = [] if heat is None else list(heat.candidates)
    fine_count = 0
    nonfine_count = 0
    nonfine_tracks: List[int] = []
    details: List[Dict[str, Any]] = []

    for cand in candidates:
        tid = int(cand.get("d21_track_id", -1))
        if tid >= 0 and tid == int(newly_static_track_id):
            continue
        pseudo_before = {
            "d20_priority_tier": int(cand.get("d20_priority_tier", 3)),
            "d21_severity": float(cand.get("d21_severity", 1.0)),
            "d21_candidate_length_mm": float(cand.get("major_length_mm", 0.0)),
            "d21_candidate_area_px": float(cand.get("area_px", 0.0)),
            "d21_candidate_dark_mean": float(cand.get("d20_shadow_mean", 0.0)),
        }
        rep = _d21v3_fine_wrinkle_report(pseudo_before, cand, args)
        if bool(rep.get("fine", False)):
            fine_count += 1
        else:
            nonfine_count += 1
            nonfine_tracks.append(tid)
        details.append({
            "track_id": tid,
            "fine": bool(rep.get("fine", False)),
            "tier": int(cand.get("d20_priority_tier", 3)),
            "severity": float(cand.get("d21_severity", 1.0)),
            "length_mm": float(cand.get("major_length_mm", 0.0)),
            "area_px": float(cand.get("area_px", 0.0)),
            "dark_mean": float(cand.get("d20_shadow_mean", 0.0)),
        })

    max_nonfine = max(0, int(getattr(args, "d21v3_finish_max_nonfine_count", 0)))
    max_fine = max(0, int(getattr(args, "d21v3_finish_max_fine_count", 1)))
    good = bool(nonfine_count <= max_nonfine and fine_count <= max_fine)
    return {
        "good": good,
        "nonfine_count": nonfine_count,
        "fine_count": fine_count,
        "max_nonfine_count": max_nonfine,
        "max_fine_count": max_fine,
        "nonfine_tracks": nonfine_tracks,
        "details": details,
    }


def _d21_record_action_result(plan: DualWrinkleStretchPlan, after_heat: Optional[WrinkleHeatmapPlan], after_obs: Optional[BottomObservation], H: np.ndarray, args) -> Dict[str,Any]:
    tid=int(plan.metrics.get("d21_track_id",-1)); before_sig=plan.metrics.get("d21_signature",{}) or {}
    result={"track_id":tid,"status":"NO_TRACK"}
    if tid<0:
        return result
    match=_d21_find_after_candidate(before_sig,after_heat,after_obs,H,args)
    with D21_MEMORY_LOCK:
        tr=D21_WRINKLE_TRACKS.get(tid)
        if tr is None:
            return result
        tr["action_count"]=int(tr.get("action_count",0))+1
        if match is None:
            tr["resolved"]=True
            tr["unchanged_count"]=0
            tr["fine_unchanged_count"]=0
            tr["resistant_unchanged_count"]=0
            result.update({"status":"RESOLVED_OR_MOVED_AWAY","action_count":tr["action_count"]})
            return result

        dist,c,sig=match
        b_len=float(plan.metrics.get("d21_candidate_length_mm",0.0))
        b_area=float(plan.metrics.get("d21_candidate_area_px",0.0))
        b_dark=float(plan.metrics.get("d21_candidate_dark_mean",0.0))
        a_len=float(c.get("major_length_mm",0.0))
        a_area=float(c.get("area_px",0.0))
        a_dark=float(c.get("d20_shadow_mean",0.0))
        len_change=abs(a_len-b_len)/max(20.0,b_len)
        area_change=abs(a_area-b_area)/max(80.0,b_area)
        dark_change=abs(a_dark-b_dark)/max(10.0,b_dark)
        unchanged=bool(
            dist<=float(getattr(args,"d21_static_center_shift_mm",25.0))
            and len_change<=float(getattr(args,"d21_static_length_rel_change",0.15))
            and area_change<=float(getattr(args,"d21_static_area_rel_change",0.20))
            and dark_change<=float(getattr(args,"d21_static_dark_rel_change",0.12))
        )

        fine_report = _d21v3_fine_wrinkle_report(plan.metrics, c, args)
        is_fine = bool(fine_report.get("fine", False))
        tr["unchanged_count"] = int(tr.get("unchanged_count",0))+1 if unchanged else 0

        if unchanged and is_fine:
            tr["fine_unchanged_count"] = int(tr.get("fine_unchanged_count",0))+1
            tr["resistant_unchanged_count"] = 0
        elif unchanged:
            # Severe/non-fine unchanged wrinkles are treated as failed or
            # ineffective manipulation, never as garment-intrinsic wrinkles.
            tr["fine_unchanged_count"] = 0
            tr["resistant_unchanged_count"] = int(tr.get("resistant_unchanged_count",0))+1
        else:
            tr["fine_unchanged_count"] = 0
            tr["resistant_unchanged_count"] = 0

        tr.update({
            **sig,
            "length_mm":a_len,
            "area_px":a_area,
            "dark_mean":a_dark,
            "last_action_at":time.time(),
            "last_fine_report":fine_report,
            "last_tier":int(c.get("d20_priority_tier",3)),
            "last_severity":float(c.get("d21_severity",plan.metrics.get("d21_severity",1.0))),
        })

        required=max(2,int(getattr(args,"d21_static_required_actions",2)))
        if tr["fine_unchanged_count"]>=required:
            tr["static"]=True
            tr["static_reason"]="UNCHANGED_AND_FINE"
            status="STATIC_INTRINSIC_FINE_REGISTERED"
            print(
                f"[D21_V3-STATIC] track={tid} 실제 동작 {required}회 후 변화가 없고 "
                f"주름 수준도 미세함 -> 고유 주름으로 등록하고 이후 무시"
            )
        elif unchanged and not is_fine:
            tr["static"] = False
            tr["static_reason"] = ""
            status="STRONG_UNCHANGED_NOT_STATIC"
            vals=fine_report.get("values",{})
            print(
                f"[D21_V3-RESISTANT] track={tid} 변화는 없지만 심한 주름이므로 고유 주름으로 등록하지 않음 | "
                f"tier={vals.get('max_tier')} severity={float(vals.get('max_severity',0.0)):.2f} "
                f"len={float(vals.get('max_length_mm',0.0)):.1f}mm "
                f"area={float(vals.get('max_area_px',0.0)):.0f}px "
                f"dark={float(vals.get('max_dark_mean',0.0)):.1f}"
            )
        else:
            status="FINE_UNCHANGED_COUNT" if unchanged and is_fine else "CHANGED"

        result.update({
            "status":status,
            "action_count":tr["action_count"],
            "unchanged_count":tr["unchanged_count"],
            "fine_unchanged_count":int(tr.get("fine_unchanged_count",0)),
            "resistant_unchanged_count":int(tr.get("resistant_unchanged_count",0)),
            "static":bool(tr.get("static",False)),
            "is_fine":is_fine,
            "fine_report":fine_report,
            "center_shift_mm":dist,
            "length_rel_change":len_change,
            "area_rel_change":area_change,
            "dark_rel_change":dark_change,
        })
    return result




def _d22_symmetry_report(
    obs: Optional[BottomObservation],
    H: Optional[np.ndarray],
    args,
) -> Dict[str, Any]:
    """Strict lower-leg silhouette symmetry about waist-mid -> crotch axis.

    D21_v2 was dominated by the broad waist/body region.  D22 evaluates only
    the mask from the crotch toward both hems and uses hard lower-leg checks:
      * board-aligned reflected IoU,
      * left/right area balance below the crotch,
      * width balance in several longitudinal bands,
      * worst-band profile error,
      * pose hem and leg-length balance.
    No weighted total score can hide a failed lower-leg hard check.
    """
    invalid = {
        "good": False,
        "valid": False,
        "reason": "invalid pose/mask/H",
        "symmetry_score": 0.0,
        "reflected_overlap": 0.0,
        "side_area_balance": 0.0,
        "hem_axis_balance": 0.0,
        "leg_length_balance": 0.0,
        "lower_reflected_iou": 0.0,
        "lower_width_balance_median": 0.0,
        "lower_width_balance_min": 0.0,
        "lower_profile_error_p90": 1.0,
        "band_reports": [],
        "remaining": ["invalid"],
    }
    if (
        obs is None or obs.pose is None or obs.mask is None or H is None
        or not bool(obs.pose.valid)
    ):
        return invalid

    p = obs.pose
    waist_mid = 0.5 * (
        np.asarray(p.waist_left, dtype=np.float32)
        + np.asarray(p.waist_right, dtype=np.float32)
    )
    crotch = np.asarray(p.crotch, dtype=np.float32)
    lower_center = np.asarray(p.lower_center, dtype=np.float32)
    axis = crotch - waist_mid
    axis_len = float(np.linalg.norm(axis))
    min_axis = float(getattr(args, "d22_symmetry_axis_min_mm", 35.0))
    if axis_len < min_axis:
        return {**invalid, "reason": f"symmetry axis too short: {axis_len:.1f}mm"}
    u = axis / max(axis_len, 1e-6)
    # Ensure positive longitudinal coordinate points from crotch toward hems.
    if float(np.dot(lower_center - crotch, u)) < 0.0:
        u = -u
    n = np.asarray([-u[1], u[0]], dtype=np.float32)

    mask = (np.asarray(obs.mask.mask_u8, dtype=np.uint8) > 0).astype(np.uint8) * 255
    if cv2.countNonZero(mask) < 100:
        return {**invalid, "reason": "mask too small"}

    ys, xs = np.where(mask > 0)
    pts_px = np.column_stack([xs, ys]).astype(np.float32)
    max_points = max(10000, int(getattr(args, "d22_symmetry_max_sample_points", 100000)))
    if len(pts_px) > max_points:
        step = max(1, int(math.ceil(len(pts_px) / float(max_points))))
        pts_px = pts_px[::step]
    pts_board = cv2.perspectiveTransform(
        pts_px.reshape(-1, 1, 2), np.asarray(H, dtype=np.float32)
    ).reshape(-1, 2)

    rel_c = pts_board - crotch.reshape(1, 2)
    longitudinal = rel_c @ u
    lateral = (pts_board - waist_mid.reshape(1, 2)) @ n
    lower_start = float(getattr(args, "d22_symmetry_lower_start_mm", -4.0))
    lower_sel = longitudinal >= lower_start
    lp = pts_board[lower_sel]
    lt = longitudinal[lower_sel]
    ld = lateral[lower_sel]
    if len(lp) < 250:
        return {**invalid, "reason": f"lower-leg mask too small: {len(lp)} points"}

    axis_band = max(0.0, float(getattr(args, "d22_symmetry_ignore_axis_band_mm", 5.0)))
    side_valid = np.abs(ld) >= axis_band
    pos_count = int(np.count_nonzero(ld[side_valid] > 0.0))
    neg_count = int(np.count_nonzero(ld[side_valid] < 0.0))
    side_balance = float(min(pos_count, neg_count) / max(1, max(pos_count, neg_count)))

    # Board-aligned occupancy grid and true symmetric IoU against its mirror.
    grid_mm = max(1.0, float(getattr(args, "d22_symmetry_grid_mm", 4.0)))
    t_lo = float(max(lower_start, np.percentile(lt, 1.0)))
    t_hi = float(np.percentile(lt, 99.0))
    d_abs = float(max(20.0, np.percentile(np.abs(ld), 99.0)))
    rows = max(8, int(math.ceil((t_hi - t_lo) / grid_mm)) + 1)
    cols_half = max(8, int(math.ceil(d_abs / grid_mm)) + 2)
    cols = 2 * cols_half + 1
    grid = np.zeros((rows, cols), dtype=np.uint8)
    ri = np.floor((lt - t_lo) / grid_mm).astype(np.int32)
    ci = np.rint(ld / grid_mm).astype(np.int32) + cols_half
    ok = (ri >= 0) & (ri < rows) & (ci >= 0) & (ci < cols)
    grid[ri[ok], ci[ok]] = 255
    close_cells = max(0, int(getattr(args, "d22_symmetry_grid_close_cells", 1)))
    if close_cells > 0:
        k = 2 * close_cells + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        grid = cv2.morphologyEx(grid, cv2.MORPH_CLOSE, kernel, iterations=1)
    mirrored = np.fliplr(grid)
    inter = int(np.count_nonzero((grid > 0) & (mirrored > 0)))
    union = int(np.count_nonzero((grid > 0) | (mirrored > 0)))
    lower_iou = float(inter / max(1, union))

    # Longitudinal band profiles. A missing/folded leg creates a low hard minimum.
    band_count = max(3, int(getattr(args, "d22_symmetry_band_count", 6)))
    min_band_points = max(20, int(getattr(args, "d22_symmetry_min_points_per_band", 60)))
    band_edges = np.linspace(max(0.0, t_lo), t_hi, band_count + 1)
    band_reports = []
    balances = []
    profile_errors = []
    worst = None
    for bi in range(band_count):
        a, b = float(band_edges[bi]), float(band_edges[bi + 1])
        sel = (lt >= a) & (lt < b if bi < band_count - 1 else lt <= b)
        dvals = ld[sel]
        pos = dvals[dvals > axis_band]
        neg = -dvals[dvals < -axis_band]
        valid_band = len(pos) >= min_band_points and len(neg) >= min_band_points
        if valid_band:
            pos_w = float(np.percentile(pos, 97.0))
            neg_w = float(np.percentile(neg, 97.0))
            bal = float(min(pos_w, neg_w) / max(1.0, max(pos_w, neg_w)))
            err = float(abs(pos_w - neg_w) / max(1.0, max(pos_w, neg_w)))
        else:
            pos_w = float(np.percentile(pos, 97.0)) if len(pos) else 0.0
            neg_w = float(np.percentile(neg, 97.0)) if len(neg) else 0.0
            bal, err = 0.0, 1.0
        mid_t = 0.5 * (a + b)
        narrow_sign = 1.0 if pos_w < neg_w else -1.0
        narrow_w = pos_w if narrow_sign > 0 else neg_w
        wide_w = neg_w if narrow_sign > 0 else pos_w
        center_board = crotch + u * mid_t
        active_board = center_board + n * narrow_sign * max(4.0, 0.82 * narrow_w)
        hold_board = center_board - n * narrow_sign * max(8.0, 0.72 * wide_w)
        item = {
            "band": bi,
            "t_start_mm": a,
            "t_end_mm": b,
            "positive_width_mm": pos_w,
            "negative_width_mm": neg_w,
            "balance": bal,
            "profile_error": err,
            "valid": valid_band,
            "narrow_side_sign": narrow_sign,
            "center_board": [float(center_board[0]), float(center_board[1])],
            "active_board": [float(active_board[0]), float(active_board[1])],
            "hold_board": [float(hold_board[0]), float(hold_board[1])],
        }
        band_reports.append(item)
        balances.append(bal)
        profile_errors.append(err)
        if worst is None or bal < float(worst[0]):
            worst = (bal, item)

    width_median = float(np.median(balances)) if balances else 0.0
    width_min = float(np.min(balances)) if balances else 0.0
    profile_p90 = float(np.percentile(profile_errors, 90.0)) if profile_errors else 1.0

    def axis_distance(pt: np.ndarray) -> float:
        return abs(float(np.dot(np.asarray(pt, dtype=np.float32) - waist_mid, n)))
    left_hem = np.asarray(p.left_hem_center, dtype=np.float32)
    right_hem = np.asarray(p.right_hem_center, dtype=np.float32)
    left_d, right_d = axis_distance(left_hem), axis_distance(right_hem)
    hem_balance = float(min(left_d, right_d) / max(1.0, max(left_d, right_d)))
    left_len = float(np.linalg.norm(left_hem - crotch))
    right_len = float(np.linalg.norm(right_hem - crotch))
    leg_balance = float(min(left_len, right_len) / max(1.0, max(left_len, right_len)))

    score = float(
        0.35 * lower_iou
        + 0.25 * width_median
        + 0.15 * width_min
        + 0.10 * side_balance
        + 0.08 * hem_balance
        + 0.07 * leg_balance
    )
    limits = {
        "lower_reflected_iou_min": float(getattr(args, "d22_lower_reflected_iou_min", 0.62)),
        "lower_width_balance_median_min": float(getattr(args, "d22_lower_width_balance_median_min", 0.78)),
        "lower_width_balance_min_min": float(getattr(args, "d22_lower_width_balance_min_min", 0.60)),
        "lower_profile_error_p90_max": float(getattr(args, "d22_lower_profile_error_p90_max", 0.38)),
        "lower_side_area_balance_min": float(getattr(args, "d22_lower_side_area_balance_min", 0.74)),
        "hem_axis_balance_min": float(getattr(args, "d22_hem_axis_balance_min", 0.74)),
        "leg_length_balance_min": float(getattr(args, "d22_leg_length_balance_min", 0.78)),
        "symmetry_score_min": float(getattr(args, "d22_symmetry_score_min", 0.70)),
    }
    checks = {
        "lower_reflected_iou": lower_iou >= limits["lower_reflected_iou_min"],
        "lower_width_balance_median": width_median >= limits["lower_width_balance_median_min"],
        "lower_width_balance_min": width_min >= limits["lower_width_balance_min_min"],
        "lower_profile_error_p90": profile_p90 <= limits["lower_profile_error_p90_max"],
        "lower_side_area_balance": side_balance >= limits["lower_side_area_balance_min"],
        "hem_axis_balance": hem_balance >= limits["hem_axis_balance_min"],
        "leg_length_balance": leg_balance >= limits["leg_length_balance_min"],
        "symmetry_score": score >= limits["symmetry_score_min"],
    }
    required = bool(getattr(args, "d22_require_lower_leg_symmetry", True))
    good = bool(all(checks.values())) if required else True
    worst_item = worst[1] if worst is not None else None
    report = {
        "good": good,
        "valid": True,
        "required": required,
        "reason": "OK" if good else "lower-leg silhouette asymmetry",
        "symmetry_score": score,
        "reflected_overlap": lower_iou,
        "side_area_balance": side_balance,
        "hem_axis_balance": hem_balance,
        "leg_length_balance": leg_balance,
        "lower_reflected_iou": lower_iou,
        "lower_width_balance_median": width_median,
        "lower_width_balance_min": width_min,
        "lower_profile_error_p90": profile_p90,
        "band_reports": band_reports,
        "worst_band": worst_item,
        "axis_start_board": [float(waist_mid[0]), float(waist_mid[1])],
        "axis_end_board": [float(lower_center[0]), float(lower_center[1])],
        "crotch_board": [float(crotch[0]), float(crotch[1])],
        "checks": checks,
        "limits": limits,
        "remaining": [name for name, passed in checks.items() if not passed],
    }
    try:
        setattr(obs, "d22_symmetry", report)
        setattr(obs, "d21v2_symmetry", report)  # compatibility with older overlay code.
    except Exception:
        pass
    return report


def _d22_print_symmetry(report: Dict[str, Any], label: str = "CURRENT") -> None:
    print(
        f"[D23-SYMMETRY] {label} good={bool(report.get('good', False))} "
        f"score={float(report.get('symmetry_score', 0.0)):.3f} "
        f"lowerIoU={float(report.get('lower_reflected_iou', 0.0)):.3f} "
        f"widthMed={float(report.get('lower_width_balance_median', 0.0)):.3f} "
        f"widthMin={float(report.get('lower_width_balance_min', 0.0)):.3f} "
        f"profileP90={float(report.get('lower_profile_error_p90', 1.0)):.3f} "
        f"side={float(report.get('side_area_balance', 0.0)):.3f} "
        f"remaining={report.get('remaining', [])}"
    )


def _d22_draw_symmetry_overlay(
    image: np.ndarray,
    H: Optional[np.ndarray],
    obs: Optional[BottomObservation],
) -> np.ndarray:
    report = getattr(obs, "d22_symmetry", None) if obs is not None else None
    if not isinstance(report, dict) or H is None:
        return image
    out = image
    try:
        color = (0, 255, 0) if bool(report.get("good", False)) else (0, 0, 255)
        a, b = report.get("axis_start_board"), report.get("axis_end_board")
        if a is not None and b is not None:
            pa = board_to_pixel(H, float(a[0]), float(a[1]))
            pb = board_to_pixel(H, float(b[0]), float(b[1]))
            if pa is not None and pb is not None:
                cv2.line(out, (int(pa[0]), int(pa[1])), (int(pb[0]), int(pb[1])), color, 3)
        for band in report.get("band_reports", []):
            if not isinstance(band, dict):
                continue
            c = band.get("center_board")
            if c is None:
                continue
            pc = board_to_pixel(H, float(c[0]), float(c[1]))
            if pc is not None:
                bal = float(band.get("balance", 0.0))
                bc = (0, 180, 255) if bal >= float(report.get("limits", {}).get("lower_width_balance_min_min", 0.60)) else (0, 0, 255)
                cv2.circle(out, (int(pc[0]), int(pc[1])), 4, bc, -1)
        label = (
            f"LOWER LR {'PASS' if report.get('good', False) else 'FAIL'} "
            f"IoU={float(report.get('lower_reflected_iou', 0.0)):.2f} "
            f"Wmed={float(report.get('lower_width_balance_median', 0.0)):.2f} "
            f"Wmin={float(report.get('lower_width_balance_min', 0.0)):.2f}"
        )
        cv2.rectangle(out, (18, 112), (720, 153), (0, 0, 0), -1)
        cv2.putText(out, label, (28, 141), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)
    except Exception:
        pass
    return out


def _d21_shape_stability_report(before_obs: Optional[BottomObservation], after_obs: Optional[BottomObservation], H: Optional[np.ndarray], args) -> Dict[str,Any]:
    bs=_d17_pose_shape_metrics(before_obs,args); a_s=_d17_pose_shape_metrics(after_obs,args)
    if before_obs is None or after_obs is None or before_obs.mask is None or after_obs.mask is None or not bs.get("valid",False) or not a_s.get("valid",False):
        return {"stable":False,"reason":"invalid pose/mask"}
    area_rel=abs(float(after_obs.mask.area_px)-float(before_obs.mask.area_px))/max(1.0,float(before_obs.mask.area_px))
    center=float(np.linalg.norm(after_obs.mask.center_board-before_obs.mask.center_board))
    waist_rel=abs(float(a_s.get("waist_width_mm",0))-float(bs.get("waist_width_mm",0)))/max(1.0,float(bs.get("waist_width_mm",0)))
    hem_rel=abs(float(a_s.get("hem_gap_mm",0))-float(bs.get("hem_gap_mm",0)))/max(1.0,float(bs.get("hem_gap_mm",0)))
    score_delta=abs(float(a_s.get("shape_score",0))-float(bs.get("shape_score",0)))
    acceptable=float(a_s.get("shape_score",0))>=float(getattr(args,"d21_stable_finish_min_shape_score",0.72))
    symmetry = _d22_symmetry_report(after_obs, H, args)
    symmetry_good = bool(symmetry.get("good", False))
    stable=bool(area_rel<=float(getattr(args,"d21_stable_mask_area_rel",0.025)) and center<=float(getattr(args,"d21_stable_center_shift_mm",15.0)) and waist_rel<=float(getattr(args,"d21_stable_waist_rel",0.05)) and hem_rel<=float(getattr(args,"d21_stable_hem_rel",0.05)) and score_delta<=float(getattr(args,"d21_stable_shape_score_delta",0.03)) and acceptable and symmetry_good)
    return {"stable":stable,"acceptable":acceptable,"symmetry_good":symmetry_good,"symmetry":symmetry,"mask_area_rel":area_rel,"center_shift_mm":center,"waist_width_rel":waist_rel,"hem_gap_rel":hem_rel,"shape_score_delta":score_delta,"after_shape_score":float(a_s.get("shape_score",0.0))}


def _d21_print_stable_finish_banner(report: Dict[str,Any], streak: int, cycle_count: int) -> None:
    print("\n"+"="*82)
    print("바지를 다 폈습니다.")
    print(f"최근 {streak}회의 실제 펴기 동작에서 형태가 안정적이고 좌우 실루엣 대칭 조건을 만족했습니다.")
    print("두 번 움직여도 변하지 않은 주름은 고유 주름으로 등록하여 종료 판단에서 제외했습니다.")
    print(f"자동 사이클={cycle_count} | shape stability={json.dumps(report,ensure_ascii=False,default=str)}")
    print("="*82+"\n")

def _d19_finish_shape_report(obs: Optional[BottomObservation], args, H: Optional[np.ndarray] = None) -> Dict[str, Any]:
    base = _d17_pose_shape_metrics(obs, args)
    if not bool(base.get("valid", False)):
        return {**base, "shape_good": False, "remaining": ["pose_invalid"]}
    checks = {
        "shape_score": float(base.get("shape_score", 0.0)) >= float(getattr(args, "d19_finish_shape_score_good", 0.78)),
        "hem_gap": float(getattr(args, "d19_finish_shape_hem_gap_ratio_min", 0.52))
                   <= float(base.get("hem_gap_ratio", 0.0))
                   <= float(getattr(args, "d19_finish_shape_hem_gap_ratio_max", 1.40)),
        "leg_balance": float(base.get("leg_balance", 0.0)) >= float(getattr(args, "d19_finish_shape_leg_balance_min", 0.80)),
        "waist_center": float(base.get("waist_center_offset_ratio", 99.0)) <= float(getattr(args, "d19_finish_shape_waist_center_offset_max", 0.16)),
        "waist_line": float(base.get("waist_line_error_ratio", 99.0)) <= float(getattr(args, "d19_finish_shape_waist_line_error_max", 0.16)),
        "crotch_axis": float(base.get("crotch_axis_offset_ratio", 99.0)) <= float(getattr(args, "d19_finish_shape_crotch_axis_offset_max", 0.24)),
        "body_axis": float(base.get("axis_to_waist_ratio", 0.0)) >= float(getattr(args, "d19_finish_shape_axis_to_waist_min", 0.72)),
    }
    symmetry = _d22_symmetry_report(obs, H, args)
    checks["left_right_symmetry"] = bool(symmetry.get("good", False))
    good = bool(all(checks.values()))
    return {
        **base,
        "shape_good": good,
        "symmetry": symmetry,
        "hard_checks": checks,
        "remaining": [name for name, ok in checks.items() if not ok],
    }



def _d20_finish_report(heat: Optional[WrinkleHeatmapPlan],
                       obs: Optional[BottomObservation],
                       args, H: Optional[np.ndarray] = None) -> Dict[str, Any]:
    """Practical finish: only actionable D20 tiers count.

    Fine/weak-dark components have already been excluded by
    build_d20_wrinkle_heatmap.  Finish therefore depends on:
      - no LARGE_LONG_DARK component,
      - no LARGE_OR_LONG component,
      - at most a small configurable number of SMALLER_ACTIONABLE components,
      - low actionable area ratio and max blob ratio,
      - acceptable pants pose/shape.
    """
    candidates = [] if heat is None else list(heat.candidates)
    mask_area = (
        float(obs.mask.area_px)
        if obs is not None and obs.mask is not None
        else 0.0
    )
    actionable_ratio = float(heat.wrinkle_ratio) if heat is not None else 1.0
    max_area = max(
        [float(c.get("area_px", 0.0)) for c in candidates] or [0.0]
    )
    max_blob_ratio = max_area / max(1.0, mask_area)

    tier3 = [
        c for c in candidates
        if int(c.get("d20_priority_tier", 0)) == 3
    ]
    tier2 = [
        c for c in candidates
        if int(c.get("d20_priority_tier", 0)) == 2
    ]
    tier1 = [
        c for c in candidates
        if int(c.get("d20_priority_tier", 0)) == 1
    ]

    limits = {
        "large_long_dark_count": int(
            getattr(args, "d20_finish_max_large_long_dark_count", 0)
        ),
        "large_or_long_count": int(
            getattr(args, "d20_finish_max_large_or_long_count", 0)
        ),
        "smaller_actionable_count": int(
            getattr(args, "d20_finish_max_smaller_actionable_count", 1)
        ),
        "total_actionable_count": int(
            getattr(args, "d20_finish_max_total_actionable_count", 2)
        ),
        "actionable_ratio": float(
            getattr(args, "d20_finish_actionable_ratio", 0.035)
        ),
        "max_blob_ratio": float(
            getattr(args, "d20_finish_max_blob_ratio", 0.015)
        ),
    }

    checks = {
        "large_long_dark_count": (
            len(tier3) <= limits["large_long_dark_count"]
        ),
        "large_or_long_count": (
            len(tier2) <= limits["large_or_long_count"]
        ),
        "smaller_actionable_count": (
            len(tier1) <= limits["smaller_actionable_count"]
        ),
        "total_actionable_count": (
            len(candidates) <= limits["total_actionable_count"]
        ),
        "actionable_ratio": (
            actionable_ratio <= limits["actionable_ratio"]
        ),
        "max_blob_ratio": (
            max_blob_ratio <= limits["max_blob_ratio"]
        ),
    }

    heat_good = bool(all(checks.values()))
    shape = _d19_finish_shape_report(obs, args, H)
    require_shape = bool(getattr(args, "d19_finish_require_shape", True))
    shape_good = (
        bool(shape.get("shape_good", False))
        if require_shape
        else True
    )
    good = bool(heat_good and shape_good)

    remaining = [name for name, ok in checks.items() if not ok]
    if require_shape and not shape_good:
        remaining.append("pants_shape")

    summary = (
        f"{'FINISH' if good else 'REMAIN'} "
        f"L3={len(tier3)} L2={len(tier2)} L1={len(tier1)} "
        f"total={len(candidates)} ratio={actionable_ratio:.3f} "
        f"maxBlob={max_blob_ratio:.3f} "
        f"shape={float(shape.get('shape_score', 0.0)):.3f}"
    )

    report = {
        "good": good,
        "heat_good": heat_good,
        "shape": shape,
        "checks": checks,
        "limits": limits,
        "remaining": remaining,
        "summary": summary,
        "metrics": {
            "large_long_dark_count": len(tier3),
            "large_or_long_count": len(tier2),
            "smaller_actionable_count": len(tier1),
            "total_actionable_count": len(candidates),
            "actionable_ratio": actionable_ratio,
            "max_blob_ratio": max_blob_ratio,
            "fine_wrinkles_ignored": int(
                getattr(heat, "d20_rejected_fine", 0)
            ) if heat is not None else 0,
            "weak_dark_wrinkles_ignored": int(
                getattr(heat, "d20_rejected_dark", 0)
            ) if heat is not None else 0,
        },
    }
    if heat is not None:
        heat.d20_finish_debug = report
    return report


def _d20_print_finish_banner(report: Dict[str, Any],
                              required: int,
                              cycle_count: int) -> None:
    print("\n" + "=" * 78)
    print("바지를 다 폈습니다.")
    print(
        f"종료 조건을 {required}회 연속 만족하여 자동 주름 펴기 동작을 종료합니다. "
        f"(자동 사이클 {cycle_count}회)"
    )
    print(f"최종 판정: {report.get('summary', '')}")
    print("허리 고무줄·가랑이·미세 주름은 제외하되, 좌우 실루엣 대칭은 반드시 확인했습니다.")
    print("=" * 78 + "\n")



def draw_wrinkle_heatmap_overlay(img: np.ndarray, plan: Optional[WrinkleHeatmapPlan], args, stale: bool = False) -> np.ndarray:
    if not bool(getattr(args, 'wrinkle_heatmap_draw', True)) or plan is None:
        return img
    heat = plan.heatmap
    mask = plan.inner_mask
    if heat.shape[:2] != img.shape[:2]:
        heat = cv2.resize(heat, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
    colored = cv2.applyColorMap(heat, cv2.COLORMAP_JET)
    alpha = float(np.clip(float(getattr(args, 'wrinkle_heatmap_alpha', 0.42)), 0.0, 0.9))
    m = mask > 0
    if np.any(m):
        blended = cv2.addWeighted(img, 1.0 - alpha, colored, alpha, 0.0)
        img[m] = blended[m]

    if bool(getattr(args, "d23_draw_waist_veto", True)):
        ignore = getattr(plan, "d19_ignore_mask", None)
        if isinstance(ignore, np.ndarray):
            if ignore.shape[:2] != img.shape[:2]:
                ignore = cv2.resize(ignore, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
            contours, _ = cv2.findContours((ignore > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(img, contours, -1, (255, 0, 255), 2)
        for report in list(getattr(plan, "d23_waist_structure_reports", [])):
            cp = report.get("center_px") if isinstance(report, dict) else None
            if cp is None:
                continue
            cx, cy = int(round(float(cp[0]))), int(round(float(cp[1])))
            cv2.line(img, (cx - 10, cy - 10), (cx + 10, cy + 10), (255, 0, 255), 3)
            cv2.line(img, (cx - 10, cy + 10), (cx + 10, cy - 10), (255, 0, 255), 3)
            cv2.putText(img, "WAIST VETO", (cx + 12, cy - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 3)
            cv2.putText(img, "WAIST VETO", (cx + 12, cy - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 0, 255), 1)

    for i, item in enumerate(plan.candidates, 1):
        x, y, w, h = map(int, item['bbox'])
        cx, cy = map(int, map(round, item['center_px']))
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 255), 1)
        cv2.circle(img, (cx, cy), 6, (0, 255, 255), 2)
        tier_name = str(item.get("d20_tier_name", ""))
        tier_code = "3" if tier_name == "LARGE_LONG_DARK" else ("2" if tier_name == "LARGE_OR_LONG" else "1")
        shape_code = {"LINE": "L", "CURVE": "C", "IRREGULAR": "I"}.get(str(item.get("d21_shape_class", "")), "?")
        wlabel = f"W{i}-T{tier_code}-{shape_code}"
        cv2.putText(img, wlabel, (cx + 7, cy - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3)
        cv2.putText(img, wlabel, (cx + 7, cy - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
    label = ('STALE ' if stale else '') + plan.summary
    y = max(28, img.shape[0] - 18)
    cv2.putText(img, label[:110], (18, y), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 0, 0), 4)
    cv2.putText(img, label[:110], (18, y), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 255, 255), 1)
    return img



# -----------------------------------------------------------------------------
# D8: dual-arm wrinkle stretching
# -----------------------------------------------------------------------------

def _safe_unit(v: np.ndarray, fallback=(1.0, 0.0)) -> np.ndarray:
    a = np.asarray(v, dtype=np.float32).reshape(2)
    n = float(np.linalg.norm(a))
    if n < 1e-6:
        return np.asarray(fallback, dtype=np.float32)
    return a / n


def _pixel_axis_to_board(H: np.ndarray, center_px: Sequence[float], axis_px: Sequence[float],
                         sample_px: float = 24.0) -> Optional[np.ndarray]:
    c = np.asarray(center_px, dtype=np.float32)
    a = _safe_unit(np.asarray(axis_px, dtype=np.float32))
    try:
        b0 = np.asarray(pixel_to_board(H, float(c[0]), float(c[1])), dtype=np.float32)
        b1 = np.asarray(pixel_to_board(H, float(c[0] + a[0] * sample_px),
                                      float(c[1] + a[1] * sample_px)), dtype=np.float32)
    except Exception:
        return None
    d = b1 - b0
    if float(np.linalg.norm(d)) < 1e-6:
        return None
    return _safe_unit(d)


def _roarm_pose_for_board(config: Dict[str, Any], cfg: BottomSafetyConfig, arm_key: str,
                          board_xy: Sequence[float], z: float) -> Tuple[float, float, float]:
    rx, ry = board_to_arm_xy(config, arm_key, float(board_xy[0]), float(board_xy[1]))
    return float(rx), float(ry), float(z)


def _dual_plan_metrics(heat: Optional[WrinkleHeatmapPlan],
                       obs: Optional[BottomObservation]) -> Dict[str, float]:
    if heat is None:
        return {
            "wrinkle_ratio": 1.0,
            "max_blob_area_px": 0.0,
            "max_blob_score": 0.0,
            "mask_area_px": 0.0,
            "max_blob_ratio": 1.0,
            "blob_count": 99.0,
        }
    max_area = max([float(c.get("area_px", 0.0)) for c in heat.candidates] or [0.0])
    max_score = max([float(c.get("max_score", 0.0)) for c in heat.candidates] or [0.0])
    mask_area = float(obs.mask.area_px) if obs is not None and obs.mask is not None else 0.0
    max_blob_ratio = max_area / max(1.0, mask_area)
    return {
        "wrinkle_ratio": float(heat.wrinkle_ratio),
        "max_blob_area_px": max_area,
        "max_blob_score": max_score,
        "mask_area_px": mask_area,
        "max_blob_ratio": float(max_blob_ratio),
        "blob_count": float(len(heat.candidates)),
    }


def _b34_history_direction_adjust(center_board: np.ndarray, direction: np.ndarray, args) -> Tuple[float, str]:
    radius = float(getattr(args, "b34_feedback_repeat_radius_mm", 55.0))
    penalty = float(getattr(args, "b34_feedback_same_bad_penalty", 120.0))
    bonus = float(getattr(args, "b34_feedback_reverse_bonus", 25.0))
    for item in reversed(B34_PRESS_HISTORY):
        c = np.asarray(item.get("center_board", [1e9, 1e9]), dtype=np.float32)
        if float(np.linalg.norm(c - center_board)) > radius:
            continue
        last_dir = _safe_unit(np.asarray(item.get("direction_board", [1.0, 0.0]), dtype=np.float32))
        state = str(item.get("state", ""))
        dot = float(np.dot(_safe_unit(direction), last_dir))
        if state in ("WORSENED_STOP", "NO_MEANINGFUL_CHANGE"):
            if dot > 0.5:
                return -penalty, "same_as_recent_bad_direction"
            if dot < -0.5:
                return bonus, "reversed_after_bad_feedback"
        break
    return 0.0, "no_feedback_adjustment"


def _remember_b34_feedback(plan: "DualWrinkleStretchPlan", evaluation: Dict[str, Any], args) -> None:
    if plan is None or not plan.ok or plan.center_board is None or plan.normal_axis_board is None:
        return
    B34_PRESS_HISTORY.append({
        "center_board": [float(plan.center_board[0]), float(plan.center_board[1])],
        "direction_board": [float(plan.normal_axis_board[0]), float(plan.normal_axis_board[1])],
        "state": str(evaluation.get("state", "")),
        "evaluation": dict(evaluation),
        "time": time.time(),
    })
    max_n = max(1, int(getattr(args, "b34_feedback_history_max", 10)))
    del B34_PRESS_HISTORY[:-max_n]



def _d13_mask_board_points(mask: BottomMaskBoard, H: np.ndarray,
                           max_points: int = 24000) -> np.ndarray:
    ys, xs = np.where(mask.mask_u8 > 0)
    if len(xs) < 10:
        return np.empty((0, 2), dtype=np.float32)
    pts = np.column_stack([xs, ys]).astype(np.float32)
    if len(pts) > max_points:
        step = max(1, int(math.ceil(len(pts) / float(max_points))))
        pts = pts[::step]
    return cv2.perspectiveTransform(pts.reshape(-1, 1, 2), H).reshape(-1, 2)


def _d13_local_mask_ratio(mask_u8: np.ndarray, H: np.ndarray,
                          board_xy: Sequence[float], radius_px: int = 18) -> float:
    pp = board_to_pixel(H, float(board_xy[0]), float(board_xy[1]))
    if pp is None:
        return 0.0
    cx, cy = int(round(pp[0])), int(round(pp[1]))
    h, w = mask_u8.shape[:2]
    r = max(2, int(radius_px))
    x0, x1 = max(0, cx - r), min(w, cx + r + 1)
    y0, y1 = max(0, cy - r), min(h, cy + r + 1)
    if x1 <= x0 or y1 <= y0:
        return 0.0
    roi = mask_u8[y0:y1, x0:x1] > 0
    yy, xx = np.ogrid[y0:y1, x0:x1]
    circle = (xx - cx) ** 2 + (yy - cy) ** 2 <= r ** 2
    denom = int(np.count_nonzero(circle))
    if denom <= 0:
        return 0.0
    return float(np.count_nonzero(roi & circle)) / float(denom)


def _d13_choose_normal_toward_empty_side(
    center_board: np.ndarray,
    normal_board: np.ndarray,
    board_pts: np.ndarray,
    cfg: BottomSafetyConfig,
    linearity: float,
    min_ratio: float,
) -> Tuple[np.ndarray, str]:
    """Literal B34 idea: choose one of the two wrinkle-normal directions toward empty board space."""
    pts = np.asarray(board_pts, dtype=np.float32).reshape(-1, 2)
    min_bx, max_bx, min_by, max_by = (
        float(cfg.board_x_min), float(cfg.board_x_max),
        float(cfg.board_y_min), float(cfg.board_y_max),
    )
    if len(pts) >= 10:
        lo_x = float(np.percentile(pts[:, 0], 2.0))
        hi_x = float(np.percentile(pts[:, 0], 98.0))
        lo_y = float(np.percentile(pts[:, 1], 2.0))
        hi_y = float(np.percentile(pts[:, 1], 98.0))
    else:
        lo_x = hi_x = float(center_board[0])
        lo_y = hi_y = float(center_board[1])

    gaps = {
        "left": max(0.0, lo_x - min_bx),
        "right": max(0.0, max_bx - hi_x),
        "top": max(0.0, max_by - hi_y),
        "bottom": max(0.0, lo_y - min_by),
    }
    side_vec = {
        "left": np.array([-1.0, 0.0], dtype=np.float32),
        "right": np.array([1.0, 0.0], dtype=np.float32),
        "top": np.array([0.0, 1.0], dtype=np.float32),
        "bottom": np.array([0.0, -1.0], dtype=np.float32),
    }

    if linearity < min_ratio:
        side = max(gaps, key=gaps.get)
        return side_vec[side], f"fallback_to_emptiest_{side}(gap={gaps[side]:.1f},ratio={linearity:.2f})"

    n = _safe_unit(normal_board)

    def score(u: np.ndarray) -> Tuple[float, str]:
        point_gap = {
            "left": max(0.0, float(center_board[0]) - min_bx),
            "right": max(0.0, max_bx - float(center_board[0])),
            "top": max(0.0, max_by - float(center_board[1])),
            "bottom": max(0.0, float(center_board[1]) - min_by),
        }
        terms = {}
        total = 0.0
        for side, sv in side_vec.items():
            align = max(0.0, float(np.dot(u, sv)))
            term = align * (float(gaps[side]) + 0.25 * float(point_gap[side]))
            terms[side] = term
            total += term
        best = max(terms, key=terms.get)
        return total, f"toward_empty_{best}(gap={gaps[best]:.1f},score={total:.1f})"

    s_pos, r_pos = score(n)
    s_neg, r_neg = score(-n)
    if s_neg > s_pos:
        return -n, f"normal_to_empty/{r_neg}/ratio={linearity:.2f}"
    return n, f"normal_to_empty/{r_pos}/ratio={linearity:.2f}"


def _d13_edge_contact_from_ray(
    mask: BottomMaskBoard,
    H: np.ndarray,
    board_pts: np.ndarray,
    origin: np.ndarray,
    direction: np.ndarray,
    cfg: BottomSafetyConfig,
    args,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], str]:
    """B34 edge-ray contact: wrinkle center -> garment edge -> inward inset contact."""
    pts = np.asarray(board_pts, dtype=np.float32).reshape(-1, 2)
    if len(pts) < 10:
        return None, None, "no cloth board points"
    u = _safe_unit(direction)
    origin = np.asarray(origin, dtype=np.float32).reshape(2)
    margin = float(getattr(args, "b34_board_margin_mm", 35.0))

    safe = (
        (pts[:, 0] >= cfg.board_x_min + margin) &
        (pts[:, 0] <= cfg.board_x_max - margin) &
        (pts[:, 1] >= cfg.board_y_min + margin) &
        (pts[:, 1] <= cfg.board_y_max - margin)
    )
    pts_safe = pts[safe]
    if len(pts_safe) < 10:
        pts_safe = pts

    rel = pts_safe - origin.reshape(1, 2)
    proj = rel @ u
    perp = np.abs(rel[:, 0] * float(u[1]) - rel[:, 1] * float(u[0]))
    forward = proj > 5.0
    if int(np.count_nonzero(forward)) < 3:
        return None, None, "no cloth points along chosen normal"

    max_proj = float(np.max(proj[forward]))
    ray_w = float(getattr(args, "b34_edge_ray_width_mm", 42.0))
    edge_band = float(getattr(args, "b34_edge_band_mm", 55.0))
    near = forward & (proj >= max_proj - edge_band) & (perp <= ray_w)
    if int(np.count_nonzero(near)) < 3:
        near = forward & (proj >= max_proj - edge_band * 1.8)
    if int(np.count_nonzero(near)) < 1:
        return None, None, "normal ray did not hit garment edge"

    cands = pts_safe[near]
    crel = cands - origin.reshape(1, 2)
    cproj = crel @ u
    cperp = np.abs(crel[:, 0] * float(u[1]) - crel[:, 1] * float(u[0]))
    edge = cands[int(np.argmin(-cproj + 0.30 * cperp))].astype(np.float32)

    inset0 = float(getattr(args, "b34_edge_contact_inset_mm", 32.0))
    min_local = float(getattr(args, "b34_min_local_mask_ratio", 0.18))
    radius_px = int(getattr(args, "b34_local_mask_radius_px", 18))
    trials = [inset0, inset0 * 1.3, inset0 * 1.6, max(12.0, inset0 * 0.65)]

    best = None
    best_ratio = -1.0
    best_note = ""
    for inset in trials:
        p = edge - u * float(inset)
        p = clamp_board_point(cfg, p, margin=margin)
        ratio = _d13_local_mask_ratio(mask.mask_u8, H, p, radius_px)
        if ratio > best_ratio:
            best, best_ratio, best_note = p, ratio, f"edge_inset={inset:.1f}"
        if ratio >= min_local:
            return p.astype(np.float32), edge, f"{best_note},local={ratio:.2f}"

        nearest = find_nearest_mask_board_point(
            mask, H, p, cfg, preferred_arm=None,
            max_radius_mm=float(getattr(args, "b34_contact_snap_mm", 35.0))
        )
        if nearest is not None:
            ratio2 = _d13_local_mask_ratio(mask.mask_u8, H, nearest, radius_px)
            if ratio2 > best_ratio:
                best, best_ratio, best_note = np.asarray(nearest, dtype=np.float32), ratio2, f"snapped_edge_inset={inset:.1f}"
            if ratio2 >= min_local:
                return np.asarray(nearest, dtype=np.float32), edge, f"{best_note},local={ratio2:.2f}"

    if best is not None and best_ratio >= max(0.08, min_local * 0.55):
        return np.asarray(best, dtype=np.float32), edge, f"low_local_allowed {best_note},local={best_ratio:.2f}"
    return None, edge, f"edge contact local too low best={best_ratio:.2f}"


def _d13_outward_fallback_contact(
    mask: BottomMaskBoard,
    H: np.ndarray,
    problem: np.ndarray,
    direction: np.ndarray,
    cfg: BottomSafetyConfig,
    args,
) -> Tuple[Optional[np.ndarray], str]:
    u = _safe_unit(direction)
    offset = float(getattr(args, "b34_outward_fallback_offset_mm", 28.0))
    desired = clamp_board_point(
        cfg, np.asarray(problem, dtype=np.float32) + u * offset,
        margin=float(getattr(args, "b34_board_margin_mm", 35.0))
    )
    local = _d13_local_mask_ratio(
        mask.mask_u8, H, desired,
        int(getattr(args, "b34_local_mask_radius_px", 18))
    )
    min_local = float(getattr(args, "b34_outward_fallback_min_local", 0.10))
    if local >= min_local:
        return desired.astype(np.float32), f"outward_fallback_offset={offset:.1f},local={local:.2f}"

    snapped = find_nearest_mask_board_point(
        mask, H, desired, cfg, preferred_arm=None,
        max_radius_mm=float(getattr(args, "b34_contact_snap_mm", 35.0))
    )
    if snapped is None:
        return None, f"fallback snap failed local={local:.2f}"
    snapped = np.asarray(snapped, dtype=np.float32)
    proj = float(np.dot(snapped - np.asarray(problem, dtype=np.float32), u))
    local2 = _d13_local_mask_ratio(
        mask.mask_u8, H, snapped,
        int(getattr(args, "b34_local_mask_radius_px", 18))
    )
    if proj < max(6.0, offset * 0.45) or local2 < max(0.06, min_local * 0.55):
        return None, f"fallback snap rejected proj={proj:.1f},local={local2:.2f}"
    return snapped, f"outward_fallback_snapped proj={proj:.1f},local={local2:.2f}"


def _d13_validate_outward_path(
    problem: np.ndarray,
    start: np.ndarray,
    end: np.ndarray,
    direction: np.ndarray,
    args,
) -> Tuple[bool, str, Dict[str, float]]:
    u = _safe_unit(direction)
    p = np.asarray(problem, dtype=np.float32)
    s = np.asarray(start, dtype=np.float32)
    e = np.asarray(end, dtype=np.float32)

    ps = float(np.dot(s - p, u))
    pe = float(np.dot(e - p, u))
    se = float(np.dot(e - s, u))
    sweep = float(np.linalg.norm(e - s))

    min_contact = float(getattr(args, "b34_min_contact_outward_mm", 12.0))
    min_sweep = float(getattr(args, "b34_min_sweep_mm", 20.0))
    ok = ps >= min_contact and pe > ps and se >= min_sweep and sweep >= min_sweep
    reason = (
        f"P->S={ps:.1f}mm P->E={pe:.1f}mm S->E={se:.1f}mm sweep={sweep:.1f}mm"
    )
    return ok, reason, {
        "problem_to_start_mm": ps,
        "problem_to_end_mm": pe,
        "start_to_end_proj_mm": se,
        "sweep_mm": sweep,
    }



def _d13v3_find_passive_hold(
    mask: BottomMaskBoard,
    H: np.ndarray,
    problem: np.ndarray,
    direction: np.ndarray,
    active_start: np.ndarray,
    active_end: np.ndarray,
    active_arm: str,
    cfg: BottomSafetyConfig,
    args,
) -> Tuple[Optional[str], Optional[np.ndarray], str, Dict[str, float]]:
    """Find a cloth-supported opposite-arm anchor approximately symmetric to the active contact."""
    hold_arm = "arm1" if active_arm == "arm2" else "arm2"
    u = _safe_unit(direction)
    p = np.asarray(problem, dtype=np.float32)
    s = np.asarray(active_start, dtype=np.float32)
    e = np.asarray(active_end, dtype=np.float32)

    active_offset = max(0.0, float(np.dot(s - p, u)))
    min_offset = float(getattr(args, "b34_passive_hold_offset_mm", 60.0))
    max_offset = float(getattr(args, "b34_passive_hold_max_offset_mm", 100.0))
    hold_offset = float(np.clip(active_offset, min_offset, max_offset))

    strict_snap = float(getattr(args, "b34_passive_hold_max_snap_mm", 35.0))
    fallback_snap = float(getattr(args, "b34_passive_hold_fallback_max_snap_mm", 110.0))
    min_local = float(getattr(args, "b34_passive_hold_min_local_mask_ratio", 0.10))
    radius_px = int(getattr(args, "b34_local_mask_radius_px", 18))
    margin = float(getattr(args, "b34_board_margin_mm", 35.0))

    best = None
    best_score = -1e9
    best_note = ""

    # Try the symmetric point first, then move it slightly toward the wrinkle line
    # when the exact symmetric point has weak mask support.
    for scale in (1.0, 0.85, 0.70, 0.55):
        desired = p - u * (hold_offset * scale)
        desired = clamp_board_point(cfg, desired, margin=margin)

        for snap_limit, tag in ((strict_snap, "strict"), (fallback_snap, "fallback")):
            snapped = find_nearest_mask_board_point(
                mask, H, desired, cfg,
                preferred_arm=hold_arm,
                max_radius_mm=snap_limit,
            )
            if snapped is None:
                continue
            snapped = np.asarray(snapped, dtype=np.float32)

            if select_arm_for_board_x(cfg, float(snapped[0])) != hold_arm:
                continue
            if not point_in_board(cfg, snapped, margin=margin):
                continue

            local = _d13_local_mask_ratio(mask.mask_u8, H, snapped, radius_px)
            snap_mm = float(np.linalg.norm(snapped - desired))
            pair_dist = float(np.linalg.norm(snapped - s))
            end_to_hold = float(np.linalg.norm(snapped - e))

            min_pair = float(getattr(args, "b34_passive_hold_min_pair_distance_mm", 90.0))
            min_end_to_hold = float(getattr(args, "b34_passive_hold_min_sweep_clearance_mm", 90.0))
            if pair_dist < min_pair or end_to_hold < min_end_to_hold:
                continue

            score = local * 100.0 - snap_mm * 0.25 + pair_dist * 0.01
            if score > best_score:
                best_score = score
                best = snapped
                best_note = (
                    f"{tag}; scale={scale:.2f}; local={local:.2f}; "
                    f"snap={snap_mm:.1f}mm; pair={pair_dist:.1f}mm; "
                    f"end_to_hold={end_to_hold:.1f}mm"
                )

    if best is None:
        return hold_arm, None, "opposite-arm cloth-supported hold point not found", {}

    best_local = _d13_local_mask_ratio(mask.mask_u8, H, best, radius_px)
    if best_local < min_local:
        return hold_arm, None, f"hold local mask ratio too low: {best_local:.2f} < {min_local:.2f}", {}

    metrics = {
        "hold_local_mask_ratio": float(best_local),
        "hold_pair_distance_mm": float(np.linalg.norm(best - s)),
        "hold_to_active_end_mm": float(np.linalg.norm(best - e)),
        "hold_offset_mm": float(np.linalg.norm(best - p)),
    }
    return hold_arm, best.astype(np.float32), best_note, metrics


def _d14_inset_active_contact(
    raw_point: np.ndarray,
    problem: np.ndarray,
    pull_direction: np.ndarray,
    requested_inset_mm: float,
    min_outward_mm: float,
) -> Tuple[np.ndarray, float]:
    """Move only the robot command point inward; perception coordinates stay unchanged."""
    raw = np.asarray(raw_point, dtype=np.float32)
    p = np.asarray(problem, dtype=np.float32)
    u = _safe_unit(pull_direction)
    outward_proj = float(np.dot(raw - p, u))
    max_safe_inset = max(0.0, outward_proj - float(min_outward_mm) - 2.0)
    used = min(max(0.0, float(requested_inset_mm)), max_safe_inset)
    return (raw - u * used).astype(np.float32), float(used)


def _d14_inset_hold_contact(
    raw_point: np.ndarray,
    problem: np.ndarray,
    pull_direction: np.ndarray,
    requested_inset_mm: float,
) -> Tuple[np.ndarray, float]:
    """Move the passive hold command toward the wrinkle/garment interior."""
    raw = np.asarray(raw_point, dtype=np.float32)
    p = np.asarray(problem, dtype=np.float32)
    u = _safe_unit(pull_direction)
    inward_capacity = max(0.0, float(np.dot(p - raw, u)))
    used = min(max(0.0, float(requested_inset_mm)), max(0.0, inward_capacity * 0.65))
    return (raw + u * used).astype(np.float32), float(used)


def _d14_finish_metrics_good(metrics: Dict[str, float], obs: Optional[BottomObservation], args) -> bool:
    if obs is None or not obs.valid or obs.mask is None:
        return False
    max_ratio = float(getattr(args, "finish_wrinkle_ratio", 0.035))
    max_blob_ratio = float(getattr(args, "finish_max_blob_ratio", 0.012))
    max_blob_count = int(getattr(args, "finish_max_blob_count", 2))
    return bool(
        float(metrics.get("wrinkle_ratio", 1.0)) <= max_ratio
        and float(metrics.get("max_blob_ratio", 1.0)) <= max_blob_ratio
        and int(round(float(metrics.get("blob_count", 99.0)))) <= max_blob_count
    )

def _d16_build_hold_pull_base(obs: BottomObservation, heat: Optional[WrinkleHeatmapPlan],
                                    H: np.ndarray, config: Dict[str, Any],
                                    cfg: BottomSafetyConfig, args) -> DualWrinkleStretchPlan:
    """B34-style active-pull/passive-hold adapted to the pants mask.

    Sequence:
      dominant wrinkle -> wrinkle normal -> active contact on one side ->
      opposite-arm passive hold on the other side -> active outward sweep.
    """
    if obs is None or obs.mask is None or not obs.valid:
        return DualWrinkleStretchPlan(False, f"invalid observation: {None if obs is None else obs.reason}")
    if heat is None or heat.status != "OK" or not heat.candidates:
        return DualWrinkleStretchPlan(False, "B34 heatmap wrinkle candidate 없음")
    if bool(getattr(args, "tta_require_ok_for_motion", True)) and obs.pose is not None:
        if str(obs.pose.tta_state) != "OK":
            return DualWrinkleStretchPlan(False, f"pose TTA state={obs.pose.tta_state}; robot motion blocked")

    board_pts = _d13_mask_board_points(obs.mask, H)
    if len(board_pts) < 10:
        return DualWrinkleStretchPlan(False, "pants mask board points 부족")

    margin = float(getattr(args, "b34_board_margin_mm", 35.0))
    requested_sweep = float(getattr(args, "b34_targeted_sweep_mm", 120.0))
    max_sweep = float(getattr(args, "b34_max_targeted_sweep_mm", 160.0))
    adaptive_pull = bool(getattr(args, "b34_adaptive_pull", True))
    adaptive_max_bonus = float(getattr(args, "b34_adaptive_pull_max_bonus_mm", 25.0))
    min_actual_pull = float(getattr(args, "b34_min_actual_pull_mm", 60.0))
    motion_inset = max(0.0, float(getattr(args, "robot_motion_inset_mm", 18.0)))
    hold_inset = max(0.0, float(getattr(args, "robot_hold_inset_mm", 12.0)))
    min_ratio = float(getattr(args, "b34_min_major_ratio_for_edge_pull", 1.15))
    press_z = float(getattr(args, "b34_press_z", -104.0))
    hold_z = float(getattr(args, "b34_passive_hold_z", press_z))
    clear_z = float(getattr(args, "b34_clear_z", -25.0))
    mid_ratio = float(np.clip(getattr(args, "b34_micro_mid_ratio", 0.55), 0.1, 0.9))

    warnings: List[str] = []
    proposals: List[Tuple[float, DualWrinkleStretchPlan]] = []

    for ci, cand in enumerate(heat.candidates):
        try:
            center_px = np.asarray(cand["center_px"], dtype=np.float32)
            problem = np.asarray(
                pixel_to_board(H, float(center_px[0]), float(center_px[1])),
                dtype=np.float32,
            )

            major_px = _safe_unit(
                np.asarray(cand.get("major_axis_px", [1.0, 0.0]), dtype=np.float32)
            )
            normal_px = np.array([-major_px[1], major_px[0]], dtype=np.float32)
            normal_board = _pixel_axis_to_board(H, center_px, normal_px)
            if normal_board is None:
                warnings.append(f"W{ci+1}: normal conversion failed")
                continue

            linearity = float(cand.get("linearity", 1.0))
            direction, direction_reason = _d13_choose_normal_toward_empty_side(
                problem, normal_board, board_pts, cfg, linearity, min_ratio
            )
            direction = _safe_unit(direction)

            raw_active_start, edge, contact_reason = _d13_edge_contact_from_ray(
                obs.mask, H, board_pts, problem, direction, cfg, args
            )
            if raw_active_start is None:
                raw_active_start, fb_reason = _d13_outward_fallback_contact(
                    obs.mask, H, problem, direction, cfg, args
                )
                contact_reason = f"edge_ray_failed/{contact_reason};{fb_reason}"
            if raw_active_start is None:
                warnings.append(f"W{ci+1}: active contact unavailable ({contact_reason})")
                continue

            min_contact_outward = float(getattr(args, "b34_min_contact_outward_mm", 12.0))
            active_start, active_inset_used = _d14_inset_active_contact(
                raw_active_start, problem, direction, motion_inset, min_contact_outward
            )

            mask_area = max(1.0, float(obs.mask.area_px))
            candidate_blob_ratio = float(cand.get("area_px", 0.0)) / mask_area
            adaptive_bonus = 0.0
            if adaptive_pull:
                ratio_bonus = max(0.0, float(heat.wrinkle_ratio) - 0.035) * 420.0
                blob_bonus = max(0.0, candidate_blob_ratio - 0.004) * 900.0
                adaptive_bonus = min(adaptive_max_bonus, ratio_bonus + blob_bonus)
            candidate_sweep = float(np.clip(
                requested_sweep + adaptive_bonus, 20.0, max_sweep
            ))

            active_end = clamp_board_point(
                cfg, active_start + direction * candidate_sweep, margin=margin
            )
            path_ok, path_reason, path_metrics = _d13_validate_outward_path(
                problem, active_start, active_end, direction, args
            )
            if not path_ok:
                warnings.append(f"W{ci+1}: outward guard failed ({path_reason})")
                continue

            active_arm = select_arm_for_board_x(cfg, float(active_start[0]))
            end_arm = select_arm_for_board_x(cfg, float(active_end[0]))
            if active_arm is None or end_arm != active_arm:
                warnings.append(f"W{ci+1}: active start/end workspace mismatch")
                continue
            if not point_in_board(cfg, active_start, margin=margin) or not point_in_board(cfg, active_end, margin=margin):
                warnings.append(f"W{ci+1}: active start/end outside safe board")
                continue

            active_local = _d13_local_mask_ratio(
                obs.mask.mask_u8, H, active_start,
                int(getattr(args, "b34_local_mask_radius_px", 18))
            )
            min_local = float(getattr(args, "b34_min_local_mask_ratio", 0.18))
            if active_local < max(0.06, min_local * 0.55):
                warnings.append(f"W{ci+1}: active local mask ratio {active_local:.2f} too low")
                continue

            hold_arm, raw_hold_point, hold_reason, hold_metrics = _d13v3_find_passive_hold(
                obs.mask, H, problem, direction,
                active_start, active_end, active_arm, cfg, args
            )
            if raw_hold_point is None or hold_arm is None:
                warnings.append(f"W{ci+1}: passive hold unavailable ({hold_reason})")
                continue

            hold_point, hold_inset_used = _d14_inset_hold_contact(
                raw_hold_point, problem, direction, hold_inset
            )
            if select_arm_for_board_x(cfg, float(hold_point[0])) != hold_arm:
                warnings.append(f"W{ci+1}: inward-adjusted hold changed arm workspace")
                continue
            hold_local_after_inset = _d13_local_mask_ratio(
                obs.mask.mask_u8, H, hold_point,
                int(getattr(args, "b34_local_mask_radius_px", 18))
            )
            if hold_local_after_inset < float(getattr(args, "b34_passive_hold_min_local_mask_ratio", 0.10)):
                warnings.append(
                    f"W{ci+1}: inward-adjusted hold local={hold_local_after_inset:.2f} too low"
                )
                continue

            active_mid = active_start + (active_end - active_start) * mid_ratio
            actual_sweep = float(np.linalg.norm(active_end - active_start))
            if actual_sweep < min_actual_pull:
                warnings.append(
                    f"W{ci+1}: actual pull {actual_sweep:.1f}mm < {min_actual_pull:.1f}mm"
                )
                continue

            arm_points = {
                active_arm: {
                    "role": "active",
                    "detected_source_board": (float(raw_active_start[0]), float(raw_active_start[1])),
                    "source_board": (float(active_start[0]), float(active_start[1])),
                    "target_board": (float(active_end[0]), float(active_end[1])),
                    "hover": _roarm_pose_for_board(config, cfg, active_arm, active_start, clear_z),
                    "down": _roarm_pose_for_board(config, cfg, active_arm, active_start, press_z),
                    "mid": _roarm_pose_for_board(config, cfg, active_arm, active_mid, press_z),
                    "sweep": _roarm_pose_for_board(config, cfg, active_arm, active_end, press_z),
                    "lift": _roarm_pose_for_board(config, cfg, active_arm, active_end, clear_z),
                },
                hold_arm: {
                    "role": "hold",
                    "detected_source_board": (float(raw_hold_point[0]), float(raw_hold_point[1])),
                    "source_board": (float(hold_point[0]), float(hold_point[1])),
                    "target_board": (float(hold_point[0]), float(hold_point[1])),
                    "hover": _roarm_pose_for_board(config, cfg, hold_arm, hold_point, clear_z),
                    "down": _roarm_pose_for_board(config, cfg, hold_arm, hold_point, hold_z),
                    "mid": _roarm_pose_for_board(config, cfg, hold_arm, hold_point, hold_z),
                    "sweep": _roarm_pose_for_board(config, cfg, hold_arm, hold_point, hold_z),
                    "lift": _roarm_pose_for_board(config, cfg, hold_arm, hold_point, clear_z),
                },
            }

            max_score = float(cand.get("max_score", cand.get("score", 0.0)))
            mean_score = float(cand.get("mean_score", 0.0))
            score = (
                max_score * 1.2
                + 28.0
                + 0.05 * float(cand.get("area_px", 0.0))
                + 10.0 * float(hold_metrics.get("hold_local_mask_ratio", 0.0))
            )

            metrics = {
                "active_arm": active_arm,
                "hold_arm": hold_arm,
                "wrinkle_ratio": float(heat.wrinkle_ratio),
                "candidate_area_px": float(cand.get("area_px", 0.0)),
                "candidate_major_length_px": float(cand.get("major_length_px", 0.0)),
                "candidate_linearity": linearity,
                "candidate_edge_distance_px": float(cand.get("edge_distance_px", 0.0)),
                "heat_max": max_score,
                "heat_mean": mean_score,
                "active_local_mask_ratio": active_local,
                "direction_reason": direction_reason,
                "contact_reason": contact_reason,
                "hold_reason": hold_reason,
                "edge_board": None if edge is None else [float(edge[0]), float(edge[1])],
                "press_z": press_z,
                "hold_z": hold_z,
                "clear_z": clear_z,
                "requested_pull_mm": requested_sweep,
                "adaptive_pull_bonus_mm": adaptive_bonus,
                "candidate_requested_pull_mm": candidate_sweep,
                "max_pull_mm": max_sweep,
                "actual_pull_mm": actual_sweep,
                "active_motion_inset_requested_mm": motion_inset,
                "active_motion_inset_used_mm": active_inset_used,
                "hold_motion_inset_requested_mm": hold_inset,
                "hold_motion_inset_used_mm": hold_inset_used,
                "hold_local_after_inset": hold_local_after_inset,
                **hold_metrics,
                **path_metrics,
            }

            proposals.append((
                score,
                DualWrinkleStretchPlan(
                    True,
                    (
                        f"W{ci+1} B34 active-pull/passive-hold; "
                        f"active={active_arm.upper()} hold={hold_arm.upper()}; "
                        f"pull={actual_sweep:.1f}mm"
                    ),
                    candidate_index=ci,
                    center_px=(float(center_px[0]), float(center_px[1])),
                    center_board=(float(problem[0]), float(problem[1])),
                    major_axis_px=(float(major_px[0]), float(major_px[1])),
                    normal_axis_board=(float(direction[0]), float(direction[1])),
                    arm_points=arm_points,
                    metrics=metrics,
                    safety_warnings=list(warnings),
                )
            ))
        except Exception as exc:
            warnings.append(f"W{ci+1}: planner error {exc!r}")

    if proposals:
        proposals.sort(key=lambda item: item[0], reverse=True)
        return proposals[0][1]

    reason = "B34 active-pull/passive-hold 후보 없음"
    if warnings:
        reason += "; " + " | ".join(warnings[-4:])
    return DualWrinkleStretchPlan(False, reason, safety_warnings=warnings)



def _d16_single_candidate_heat(heat: WrinkleHeatmapPlan, cand: Dict[str, Any]) -> WrinkleHeatmapPlan:
    return WrinkleHeatmapPlan(
        heatmap=heat.heatmap,
        inner_mask=heat.inner_mask,
        binary=heat.binary,
        candidates=[cand],
        threshold=heat.threshold,
        wrinkle_ratio=heat.wrinkle_ratio,
        status=heat.status,
        summary=heat.summary,
        created_at=heat.created_at,
    )



def _d16_aggression_scale(args) -> float:
    """Bounded multiplier for pull distances; preserves board/workspace clipping."""
    requested = float(getattr(args, "d16_aggression_scale", 1.15))
    maximum = float(getattr(args, "d16_max_aggression_scale", 1.30))
    return float(np.clip(requested, 1.0, max(1.0, maximum)))



def _d16_candidate_length_mm(cand: Dict[str, Any], H: np.ndarray) -> float:
    center = np.asarray(cand.get('center_px', [0.0, 0.0]), dtype=np.float32)
    axis = _safe_unit(np.asarray(cand.get('major_axis_px', [1.0, 0.0]), dtype=np.float32))
    half = max(1.0, float(cand.get('major_length_px', 0.0)) * 0.5)
    p0 = center - axis * half
    p1 = center + axis * half
    try:
        b0 = np.asarray(pixel_to_board(H, float(p0[0]), float(p0[1])), dtype=np.float32)
        b1 = np.asarray(pixel_to_board(H, float(p1[0]), float(p1[1])), dtype=np.float32)
    except Exception:
        return 0.0
    return float(np.linalg.norm(b1 - b0))


def _d16_classify(cand: Dict[str, Any], H: np.ndarray, mask_area_px: float, args) -> Dict[str, Any]:
    length_mm = _d16_candidate_length_mm(cand, H)
    linearity = max(1.0, float(cand.get('linearity', 1.0)))
    width_mm = length_mm / linearity
    area_px = max(0.0, float(cand.get('area_px', 0.0)))
    blob_ratio = area_px / max(1.0, float(mask_area_px))
    edge_px = float(cand.get('edge_distance_px', 999.0))

    broad_round = (
        linearity < float(getattr(args, 'd16_round_linearity_max', 1.30))
        and blob_ratio >= float(getattr(args, 'd16_round_blob_ratio_min', 0.008))
    )
    edge = edge_px <= float(getattr(args, 'd16_edge_distance_px', 12.0))
    long_linear = (
        length_mm >= float(getattr(args, 'd16_long_wrinkle_min_mm', 120.0))
        and linearity >= float(getattr(args, 'd16_long_linearity_min', 1.80))
    )
    short_linear = (
        length_mm >= float(getattr(args, 'd16_short_wrinkle_min_mm', 35.0))
        and linearity >= float(getattr(args, 'd16_short_linearity_min', 1.30))
        and not long_linear
    )

    if broad_round:
        wrinkle_type, preferred = 'BROAD_ROUND', 'SHORT_PRE_SMOOTH'
    elif edge:
        wrinkle_type, preferred = 'EDGE', 'EDGE_SWEEP_WITH_HOLD'
    elif long_linear:
        wrinkle_type, preferred = 'LONG_LINEAR', 'DUAL_OPPOSITE_STRETCH'
    elif short_linear:
        wrinkle_type, preferred = 'SHORT_LINEAR', 'ACTIVE_PULL_PASSIVE_HOLD'
    else:
        wrinkle_type, preferred = 'AMBIGUOUS', 'SHORT_PRE_SMOOTH'

    return {
        'wrinkle_type': wrinkle_type,
        'preferred_action': preferred,
        'length_mm': length_mm,
        'width_mm': width_mm,
        'linearity': linearity,
        'blob_ratio': blob_ratio,
        'edge_distance_px': edge_px,
        'area_px': area_px,
    }


def _d16_call_hold_pull(obs, heat, cand, H, config, cfg, args,
                        action: str, pull_mm: float, min_pull_mm: float):
    import copy
    local_args = copy.copy(args)
    local_args.b34_targeted_sweep_mm = float(pull_mm)
    local_args.b34_max_targeted_sweep_mm = max(
        float(pull_mm), float(getattr(args, 'b34_max_targeted_sweep_mm', 160.0))
    )
    local_args.b34_min_actual_pull_mm = float(min_pull_mm)
    plan = _d16_build_hold_pull_base(
        obs, _d16_single_candidate_heat(heat, cand), H, config, cfg, local_args
    )
    if plan.ok:
        plan.action = action
        plan.metrics['policy_action'] = action
    return plan


def _d16_make_dual_plan(obs, heat, cand, ci, feat, H, config, cfg, args):
    center_px = np.asarray(cand['center_px'], dtype=np.float32)
    center = np.asarray(pixel_to_board(H, float(center_px[0]), float(center_px[1])), dtype=np.float32)
    major_px = _safe_unit(np.asarray(cand.get('major_axis_px', [1.0, 0.0]), dtype=np.float32))
    normal_px = np.asarray([-major_px[1], major_px[0]], dtype=np.float32)
    normal = _pixel_axis_to_board(H, center_px, normal_px)
    if normal is None:
        return DualWrinkleStretchPlan(False, 'dual normal conversion failed')
    normal = _safe_unit(normal)

    offset = float(getattr(args, 'd16_dual_grip_offset_mm', 48.0))
    inset = float(getattr(args, 'd16_dual_grip_inset_mm', 12.0))
    snap = float(getattr(args, 'd16_dual_snap_radius_mm', 85.0))
    aggression = _d16_aggression_scale(args)
    stretch = float(np.clip(
        float(getattr(args, 'd16_dual_stretch_mm', 85.0)) * aggression,
        10.0,
        getattr(args, 'd16_dual_max_stretch_mm', 110.0),
    ))
    min_actual = float(getattr(args, 'd16_dual_min_actual_stretch_mm', 35.0))
    min_sep = float(getattr(args, 'd16_dual_min_grip_separation_mm', 100.0))
    margin = float(getattr(args, 'b34_board_margin_mm', 35.0))
    press_z = float(getattr(args, 'b34_press_z', -104.0))
    clear_z = float(getattr(args, 'b34_clear_z', -25.0))
    mid_ratio = float(np.clip(getattr(args, 'b34_micro_mid_ratio', 0.55), 0.2, 0.8))

    raw_a = center - normal * offset
    raw_b = center + normal * offset
    if raw_a[0] <= raw_b[0]:
        raw = {'arm2': raw_a, 'arm1': raw_b}
        signs = {'arm2': -1.0, 'arm1': 1.0}
    else:
        raw = {'arm2': raw_b, 'arm1': raw_a}
        signs = {'arm2': 1.0, 'arm1': -1.0}

    sources, targets, actuals = {}, {}, {}
    arm_points = {}
    for arm_key in ('arm2', 'arm1'):
        inward = _safe_unit(center - raw[arm_key])
        desired = raw[arm_key] + inward * inset
        source = find_nearest_mask_board_point(
            obs.mask, H, desired, cfg,
            preferred_arm=arm_key, max_radius_mm=snap
        )
        if source is None:
            return DualWrinkleStretchPlan(False, f'{arm_key} dual grip unavailable')
        source = np.asarray(source, dtype=np.float32)
        target = clamp_board_point(
            cfg, source + normal * signs[arm_key] * stretch, margin=margin
        )
        if select_arm_for_board_x(cfg, float(source[0])) != arm_key:
            return DualWrinkleStretchPlan(False, f'{arm_key} source workspace mismatch')
        if select_arm_for_board_x(cfg, float(target[0])) != arm_key:
            return DualWrinkleStretchPlan(False, f'{arm_key} target workspace mismatch')
        actual = float(np.dot(target - source, normal * signs[arm_key]))
        if actual < min_actual:
            return DualWrinkleStretchPlan(
                False, f'{arm_key} actual dual stretch {actual:.1f}mm < {min_actual:.1f}mm'
            )
        local = _d13_local_mask_ratio(
            obs.mask.mask_u8, H, source,
            int(getattr(args, 'b34_local_mask_radius_px', 18))
        )
        if local < float(getattr(args, 'd16_dual_min_local_mask_ratio', 0.12)):
            return DualWrinkleStretchPlan(False, f'{arm_key} local mask ratio {local:.2f} too low')
        sources[arm_key], targets[arm_key], actuals[arm_key] = source, target, actual

    sep = float(np.linalg.norm(sources['arm1'] - sources['arm2']))
    if sep < min_sep:
        return DualWrinkleStretchPlan(False, f'dual grip separation {sep:.1f}mm < {min_sep:.1f}mm')

    for arm_key in ('arm2', 'arm1'):
        source, target = sources[arm_key], targets[arm_key]
        mid = source + (target - source) * mid_ratio
        arm_points[arm_key] = {
            'role': 'dual_active',
            'detected_source_board': (float(raw[arm_key][0]), float(raw[arm_key][1])),
            'source_board': (float(source[0]), float(source[1])),
            'target_board': (float(target[0]), float(target[1])),
            'hover': _roarm_pose_for_board(config, cfg, arm_key, source, clear_z),
            'down': _roarm_pose_for_board(config, cfg, arm_key, source, press_z),
            'mid': _roarm_pose_for_board(config, cfg, arm_key, mid, press_z),
            'sweep': _roarm_pose_for_board(config, cfg, arm_key, target, press_z),
            'lift': _roarm_pose_for_board(config, cfg, arm_key, target, clear_z),
        }

    metrics = _dual_plan_metrics(heat, obs)
    metrics.update(feat)
    metrics.update({
        'policy_action': 'DUAL_OPPOSITE_STRETCH',
        'grip_separation_mm': sep,
        'arm2_actual_stretch_mm': actuals['arm2'],
        'arm1_actual_stretch_mm': actuals['arm1'],
        'requested_stretch_each_mm': stretch,
        'aggression_scale': aggression,
    })
    return DualWrinkleStretchPlan(
        True,
        f"W{ci+1} LONG_LINEAR -> DUAL_OPPOSITE_STRETCH; length={feat['length_mm']:.1f}mm",
        action='DUAL_OPPOSITE_STRETCH', candidate_index=ci,
        center_px=(float(center_px[0]), float(center_px[1])),
        center_board=(float(center[0]), float(center[1])),
        major_axis_px=(float(major_px[0]), float(major_px[1])),
        normal_axis_board=(float(normal[0]), float(normal[1])),
        arm_points=arm_points, metrics=metrics,
    )


def _d16_make_pre_smooth_plan(obs, heat, cand, ci, feat, H, config, cfg, args):
    center_px = np.asarray(cand['center_px'], dtype=np.float32)
    center = np.asarray(pixel_to_board(H, float(center_px[0]), float(center_px[1])), dtype=np.float32)
    major_px = _safe_unit(np.asarray(cand.get('major_axis_px', [1.0, 0.0]), dtype=np.float32))
    normal_px = np.asarray([-major_px[1], major_px[0]], dtype=np.float32)
    normal = _pixel_axis_to_board(H, center_px, normal_px)
    if normal is None:
        return DualWrinkleStretchPlan(False, 'pre-smooth normal conversion failed')
    board_pts = _d13_mask_board_points(obs.mask, H)
    direction, direction_reason = _d13_choose_normal_toward_empty_side(
        center, normal, board_pts, cfg, feat['linearity'],
        float(getattr(args, 'b34_min_major_ratio_for_edge_pull', 1.15))
    )
    direction = _safe_unit(direction)
    raw_start, contact_reason = _d13_outward_fallback_contact(
        obs.mask, H, center, direction, cfg, args
    )
    if raw_start is None:
        return DualWrinkleStretchPlan(False, f'pre-smooth contact unavailable ({contact_reason})')
    start, inset_used = _d14_inset_active_contact(
        raw_start, center, direction,
        float(getattr(args, 'robot_motion_inset_mm', 18.0)),
        float(getattr(args, 'b34_min_contact_outward_mm', 12.0))
    )
    margin = float(getattr(args, 'b34_board_margin_mm', 35.0))
    aggression = _d16_aggression_scale(args)
    pre_distance = float(getattr(args, 'd16_pre_smooth_mm', 60.0)) * aggression
    pre_distance = min(
        pre_distance,
        float(getattr(args, 'd16_pre_smooth_max_mm', 80.0))
    )
    target = clamp_board_point(
        cfg, start + direction * pre_distance,
        margin=margin
    )
    arm_key = select_arm_for_board_x(cfg, float(start[0]))
    if arm_key is None or select_arm_for_board_x(cfg, float(target[0])) != arm_key:
        return DualWrinkleStretchPlan(False, 'pre-smooth workspace mismatch')
    actual = float(np.dot(target - start, direction))
    min_actual = float(getattr(args, 'd16_pre_smooth_min_actual_mm', 20.0))
    if actual < min_actual:
        return DualWrinkleStretchPlan(False, f'pre-smooth actual {actual:.1f}mm < {min_actual:.1f}mm')
    press_z = float(getattr(args, 'b34_press_z', -104.0))
    clear_z = float(getattr(args, 'b34_clear_z', -25.0))
    mid = start + (target - start) * float(np.clip(
        getattr(args, 'b34_micro_mid_ratio', 0.55), 0.2, 0.8
    ))
    arm_points = {
        arm_key: {
            'role': 'pre_smooth',
            'detected_source_board': (float(raw_start[0]), float(raw_start[1])),
            'source_board': (float(start[0]), float(start[1])),
            'target_board': (float(target[0]), float(target[1])),
            'hover': _roarm_pose_for_board(config, cfg, arm_key, start, clear_z),
            'down': _roarm_pose_for_board(config, cfg, arm_key, start, press_z),
            'mid': _roarm_pose_for_board(config, cfg, arm_key, mid, press_z),
            'sweep': _roarm_pose_for_board(config, cfg, arm_key, target, press_z),
            'lift': _roarm_pose_for_board(config, cfg, arm_key, target, clear_z),
        }
    }
    metrics = _dual_plan_metrics(heat, obs)
    metrics.update(feat)
    metrics.update({
        'policy_action': 'SHORT_PRE_SMOOTH',
        'active_arm': arm_key,
        'actual_pull_mm': actual,
        'direction_reason': direction_reason,
        'contact_reason': contact_reason,
        'active_motion_inset_used_mm': inset_used,
        'aggression_scale': aggression,
        'requested_pre_smooth_mm': pre_distance,
    })
    return DualWrinkleStretchPlan(
        True,
        f"W{ci+1} {feat['wrinkle_type']} -> SHORT_PRE_SMOOTH; pull={actual:.1f}mm",
        action='SHORT_PRE_SMOOTH', candidate_index=ci,
        center_px=(float(center_px[0]), float(center_px[1])),
        center_board=(float(center[0]), float(center[1])),
        major_axis_px=(float(major_px[0]), float(major_px[1])),
        normal_axis_board=(float(direction[0]), float(direction[1])),
        arm_points=arm_points, metrics=metrics,
    )




def _d18_dense_actual_mask_point(
    obs: BottomObservation,
    H: np.ndarray,
    cfg: BottomSafetyConfig,
    desired_board: np.ndarray,
    expected_arm: str,
    args,
) -> Tuple[Optional[np.ndarray], str, Dict[str, float]]:
    """Choose an actual segmentation-mask pixel near the visual target.

    Selection order:
      1) only real foreground pixels from the small segmentation mask,
      2) only pixels in the expected arm's private workspace,
      3) prefer pixels deep inside the mask,
      4) prefer pixels with dense local mask support,
      5) remain near the original pose/wrinkle target.

    The returned board point is therefore derived from an actual mask pixel,
    not from a bbox, rotated box, or an unverified geometric offset.
    """
    if obs is None or obs.mask is None:
        return None, "mask missing", {}

    desired = np.asarray(desired_board, dtype=np.float32).reshape(2)
    mask_u8 = np.asarray(obs.mask.mask_u8, dtype=np.uint8)
    if mask_u8.ndim != 2 or cv2.countNonZero(mask_u8) < 10:
        return None, "mask empty", {}

    max_points = max(1000, int(getattr(args, "d18_mask_max_sample_points", 24000)))
    snap_radius_mm = float(getattr(args, "d18_mask_snap_radius_mm", 75.0))
    min_inside_px = float(getattr(args, "d18_mask_snap_min_inside_px", 4.0))
    min_local = float(getattr(args, "d18_mask_snap_min_local_ratio", 0.22))
    local_radius = max(3, int(getattr(args, "d18_mask_snap_local_radius_px", 18)))
    inside_weight = float(getattr(args, "d18_mask_snap_inside_weight", 2.8))
    local_weight = float(getattr(args, "d18_mask_snap_local_weight", 45.0))

    ys, xs = np.where(mask_u8 > 0)
    if len(xs) < 10:
        return None, "no foreground pixels", {}

    if len(xs) > max_points:
        step = max(1, int(math.ceil(len(xs) / float(max_points))))
        xs = xs[::step]
        ys = ys[::step]

    pts_px = np.column_stack([xs, ys]).astype(np.float32)
    pts_board = cv2.perspectiveTransform(
        pts_px.reshape(-1, 1, 2), H
    ).reshape(-1, 2)

    dist_map = cv2.distanceTransform(
        (mask_u8 > 0).astype(np.uint8), cv2.DIST_L2, 5
    )

    rows = []
    for idx, board_pt in enumerate(pts_board):
        bx, by = float(board_pt[0]), float(board_pt[1])
        if not point_in_board(cfg, board_pt, margin=cfg.board_margin_mm):
            continue
        if select_arm_for_board_x(cfg, bx) != expected_arm:
            continue

        d_mm = float(np.linalg.norm(board_pt - desired))
        if d_mm > snap_radius_mm:
            continue

        px = pts_px[idx]
        x = int(np.clip(round(float(px[0])), 0, mask_u8.shape[1] - 1))
        y = int(np.clip(round(float(px[1])), 0, mask_u8.shape[0] - 1))
        inside_px = float(dist_map[y, x])

        # Fast first-stage score. Local ratio is evaluated only for the best
        # geometric/interior candidates.
        pre_score = d_mm - inside_weight * inside_px
        rows.append((pre_score, d_mm, inside_px, idx))

    if not rows:
        return None, (
            f"no actual mask pixel within {snap_radius_mm:.1f}mm "
            f"for {expected_arm}"
        ), {}

    rows.sort(key=lambda row: row[0])
    rows = rows[: min(420, len(rows))]

    best = None
    best_score = float("inf")
    best_info = None
    valid_best = None
    valid_score = float("inf")

    for _, d_mm, inside_px, idx in rows:
        board_pt = np.asarray(pts_board[idx], dtype=np.float32)
        local = _d13_local_mask_ratio(
            mask_u8, H, board_pt, local_radius
        )
        score = d_mm - inside_weight * inside_px - local_weight * local

        if score < best_score:
            best_score = score
            best = board_pt
            best_info = (d_mm, inside_px, local)

        if inside_px >= min_inside_px and local >= min_local and score < valid_score:
            valid_score = score
            valid_best = board_pt
            valid_info = (d_mm, inside_px, local)

    if valid_best is not None:
        d_mm, inside_px, local = valid_info
        return valid_best.astype(np.float32), (
            f"actual dense mask pixel OK: snap={d_mm:.1f}mm "
            f"inside={inside_px:.1f}px local={local:.2f}"
        ), {
            "mask_snap_distance_mm": float(d_mm),
            "mask_snap_inside_px": float(inside_px),
            "mask_snap_local_ratio": float(local),
            "mask_snap_strict_pass": 1.0,
        }

    # Conservative fallback is optional. By default D18 fails closed instead
    # of pressing an edge pixel with weak local cloth support.
    if bool(getattr(args, "d18_mask_snap_allow_weak_fallback", False)) and best is not None:
        d_mm, inside_px, local = best_info
        return best.astype(np.float32), (
            f"weak actual-mask fallback: snap={d_mm:.1f}mm "
            f"inside={inside_px:.1f}px local={local:.2f}"
        ), {
            "mask_snap_distance_mm": float(d_mm),
            "mask_snap_inside_px": float(inside_px),
            "mask_snap_local_ratio": float(local),
            "mask_snap_strict_pass": 0.0,
        }

    if best_info is None:
        return None, "mask candidate evaluation failed", {}

    d_mm, inside_px, local = best_info
    return None, (
        f"actual mask pixels found but none were dense enough: "
        f"best snap={d_mm:.1f}mm inside={inside_px:.1f}px "
        f"local={local:.2f}; required inside>={min_inside_px:.1f}px "
        f"local>={min_local:.2f}"
    ), {
        "mask_snap_distance_mm": float(d_mm),
        "mask_snap_inside_px": float(inside_px),
        "mask_snap_local_ratio": float(local),
        "mask_snap_strict_pass": 0.0,
    }



def _d16v3_board_mask_distance_px(
    mask: BottomMaskBoard,
    H: np.ndarray,
    board_xy: Sequence[float],
) -> Optional[float]:
    px = board_to_pixel(H, float(board_xy[0]), float(board_xy[1]))
    if px is None:
        return None
    return float(
        cv2.pointPolygonTest(
            mask.contour,
            (float(px[0]), float(px[1])),
            True,
        )
    )


def _d16v3_correct_grip_point(
    obs: BottomObservation,
    H: np.ndarray,
    cfg: BottomSafetyConfig,
    raw_point: np.ndarray,
    expected_arm: str,
    wrinkle_center: np.ndarray,
    role: str,
    args,
) -> Tuple[Optional[np.ndarray], str, Dict[str, float]]:
    """D18 grip correction.

    raw visual target
      -> nearest dense actual segmentation-mask pixel
      -> additional inward correction
      -> final mask/interior/workspace validation
    """
    if obs.mask is None:
        return None, "mask missing", {}

    visual_raw = np.asarray(raw_point, dtype=np.float32).reshape(2)
    wrinkle_center = np.asarray(wrinkle_center, dtype=np.float32).reshape(2)
    mask_center = np.asarray(obs.mask.center_board, dtype=np.float32).reshape(2)

    mask_first = bool(getattr(args, "d18_mask_first_grip", True))
    mask_required = bool(getattr(args, "d18_mask_snap_required", True))
    base_metrics: Dict[str, float] = {
        "visual_raw_x": float(visual_raw[0]),
        "visual_raw_y": float(visual_raw[1]),
    }

    raw = visual_raw.copy()
    snap_reason = "mask-first disabled"
    if mask_first:
        snapped, snap_reason, snap_metrics = _d18_dense_actual_mask_point(
            obs, H, cfg, visual_raw, expected_arm, args
        )
        base_metrics.update(snap_metrics)
        if snapped is None:
            if mask_required:
                return None, f"mask-first grip failed: {snap_reason}", base_metrics
        else:
            raw = np.asarray(snapped, dtype=np.float32)
            base_metrics.update({
                "mask_snapped_x": float(raw[0]),
                "mask_snapped_y": float(raw[1]),
                "visual_to_mask_snap_mm": float(np.linalg.norm(raw - visual_raw)),
            })

    if role == "hold":
        anchor = 0.55 * wrinkle_center + 0.45 * mask_center
        requested = float(getattr(args, "d16_hold_grip_inset_mm", 28.0))
    else:
        anchor = 0.70 * wrinkle_center + 0.30 * mask_center
        requested = float(getattr(args, "d16_grip_inset_mm", 28.0))

    direction = _safe_unit(anchor - raw)
    max_inset = float(getattr(args, "d16_grip_max_inset_mm", 85.0))
    min_inside_px = float(getattr(args, "d16_grip_min_inside_px", 2.0))
    min_local = float(getattr(args, "d16_grip_min_local_ratio", 0.18))
    local_radius = int(getattr(args, "d16_grip_local_radius_px", 18))
    margin = float(getattr(args, "b34_board_margin_mm", 35.0))

    candidate_distances = [
        requested,
        requested * 0.75,
        requested * 1.20,
        requested * 1.50,
        max(8.0, requested * 0.45),
        min(max_inset, requested * 2.00),
        0.0,  # The already-dense actual mask pixel is also a valid candidate.
    ]

    best = None
    best_score = -1e9
    best_info = ""
    seen = set()

    for dist in candidate_distances:
        dist = float(np.clip(dist, 0.0, max_inset))
        key = round(dist, 2)
        if key in seen:
            continue
        seen.add(key)

        cand = clamp_board_point(cfg, raw + direction * dist, margin=margin)
        if select_arm_for_board_x(cfg, float(cand[0])) != expected_arm:
            continue

        signed_px = _d16v3_board_mask_distance_px(obs.mask, H, cand)
        if signed_px is None:
            continue
        local = _d13_local_mask_ratio(
            obs.mask.mask_u8, H, cand, local_radius
        )

        score = signed_px * 4.0 + local * 100.0 - abs(dist - requested) * 0.08
        if score > best_score:
            best_score = score
            best = cand.astype(np.float32)
            best_info = (
                f"inset={dist:.1f}mm signed={signed_px:.1f}px "
                f"local={local:.2f}"
            )

        if signed_px >= min_inside_px and local >= min_local:
            metrics = dict(base_metrics)
            metrics.update({
                "requested_inset_mm": requested,
                "used_inset_from_mask_mm": float(np.linalg.norm(cand - raw)),
                "used_total_from_visual_mm": float(np.linalg.norm(cand - visual_raw)),
                "inside_distance_px": signed_px,
                "local_mask_ratio": local,
            })
            return cand.astype(np.float32), (
                f"{snap_reason}; inward correction OK: "
                f"inset={dist:.1f}mm inside={signed_px:.1f}px "
                f"local={local:.2f}"
            ), metrics

    desired = clamp_board_point(
        cfg, raw + direction * requested, margin=margin
    )
    snapped = find_nearest_mask_board_point(
        obs.mask,
        H,
        desired,
        cfg,
        preferred_arm=expected_arm,
        max_radius_mm=float(getattr(args, "d16_grip_snap_radius_mm", 65.0)),
    )
    if snapped is not None:
        snapped = np.asarray(snapped, dtype=np.float32)
        refine = clamp_board_point(
            cfg,
            snapped + direction * float(
                getattr(args, "d16_grip_snap_refine_mm", 10.0)
            ),
            margin=margin,
        )
        for option in (refine, snapped):
            if select_arm_for_board_x(cfg, float(option[0])) != expected_arm:
                continue
            signed_px = _d16v3_board_mask_distance_px(obs.mask, H, option)
            local = _d13_local_mask_ratio(
                obs.mask.mask_u8, H, option, local_radius
            )
            if (
                signed_px is not None
                and signed_px >= -1.0
                and local >= max(0.10, min_local * 0.65)
            ):
                metrics = dict(base_metrics)
                metrics.update({
                    "requested_inset_mm": requested,
                    "used_inset_from_mask_mm": float(np.linalg.norm(option - raw)),
                    "used_total_from_visual_mm": float(np.linalg.norm(option - visual_raw)),
                    "inside_distance_px": float(signed_px),
                    "local_mask_ratio": float(local),
                })
                return option.astype(np.float32), (
                    f"{snap_reason}; secondary mask snap: "
                    f"inside={signed_px:.1f}px local={local:.2f}"
                ), metrics

    if best is not None:
        best_signed = _d16v3_board_mask_distance_px(obs.mask, H, best)
        best_local = _d13_local_mask_ratio(
            obs.mask.mask_u8, H, best, local_radius
        )
        if (
            best_signed is not None
            and best_signed >= -1.0
            and best_local >= 0.10
        ):
            metrics = dict(base_metrics)
            metrics.update({
                "requested_inset_mm": requested,
                "used_inset_from_mask_mm": float(np.linalg.norm(best - raw)),
                "used_total_from_visual_mm": float(np.linalg.norm(best - visual_raw)),
                "inside_distance_px": float(best_signed),
                "local_mask_ratio": float(best_local),
            })
            return best.astype(np.float32), (
                f"{snap_reason}; best conservative candidate: {best_info}"
            ), metrics

    return None, (
        f"no safe D18 mask-first grip for {expected_arm}/{role}; "
        f"snap={snap_reason}; inset={requested:.1f}mm"
    ), base_metrics

def _d16v3_shift_target_with_source(
    cfg: BottomSafetyConfig,
    arm_key: str,
    corrected_source: np.ndarray,
    original_source: np.ndarray,
    original_target: np.ndarray,
    margin: float,
    min_motion_mm: float,
) -> Tuple[Optional[np.ndarray], float]:
    """Preserve the original motion vector after correcting the source point.

    If the full vector leaves the arm workspace, progressively shorten it.
    """
    delta = np.asarray(original_target, dtype=np.float32) - np.asarray(
        original_source, dtype=np.float32
    )
    if float(np.linalg.norm(delta)) < 1e-6:
        return corrected_source.astype(np.float32), 0.0

    for scale in (1.0, 0.90, 0.80, 0.70, 0.60, 0.50, 0.40, 0.30):
        target = clamp_board_point(
            cfg,
            corrected_source + delta * float(scale),
            margin=margin,
        )
        if select_arm_for_board_x(cfg, float(target[0])) != arm_key:
            continue
        actual = float(np.linalg.norm(target - corrected_source))
        if actual >= min_motion_mm:
            return target.astype(np.float32), actual
    return None, 0.0


def _d16v3_apply_grip_correction(
    plan: DualWrinkleStretchPlan,
    obs: BottomObservation,
    H: np.ndarray,
    config: Dict[str, Any],
    cfg: BottomSafetyConfig,
    args,
) -> Tuple[Optional[DualWrinkleStretchPlan], str]:
    """Apply inward correction to BOTH robot contacts.

    Perception points remain in detected_source_board; only source_board and
    robot poses are changed.
    """
    if not plan.ok:
        return None, plan.reason
    if obs.mask is None:
        return None, "mask missing"

    # Every successful D16_V3 action must include both arms.
    if "arm1" not in plan.arm_points or "arm2" not in plan.arm_points:
        return None, "two-arm invariant failed: ARM1/ARM2 points are both required"

    center = np.asarray(
        plan.center_board if plan.center_board is not None else obs.mask.center_board,
        dtype=np.float32,
    )
    margin = float(getattr(args, "b34_board_margin_mm", 35.0))
    mid_ratio = float(np.clip(getattr(args, "b34_micro_mid_ratio", 0.55), 0.2, 0.8))
    min_motion = float(getattr(args, "d16_v3_min_corrected_motion_mm", 25.0))

    correction_metrics: Dict[str, Any] = {}

    for arm_key in ("arm2", "arm1"):
        p = plan.arm_points[arm_key]
        role = str(p.get("role", "active"))
        raw = np.asarray(
            p.get("detected_source_board", p["source_board"]),
            dtype=np.float32,
        )
        old_source = np.asarray(p["source_board"], dtype=np.float32)
        old_target = np.asarray(p["target_board"], dtype=np.float32)

        corrected, reason, metrics = _d16v3_correct_grip_point(
            obs,
            H,
            cfg,
            raw,
            arm_key,
            center,
            role,
            args,
        )
        if corrected is None:
            return None, f"{arm_key} grip correction failed: {reason}"

        is_hold = role == "hold" or float(np.linalg.norm(old_target - old_source)) < 1e-6
        if is_hold:
            corrected_target = corrected.copy()
            actual_motion = 0.0
        else:
            corrected_target, actual_motion = _d16v3_shift_target_with_source(
                cfg,
                arm_key,
                corrected,
                old_source,
                old_target,
                margin,
                min_motion,
            )
            if corrected_target is None:
                return None, (
                    f"{arm_key} corrected target cannot stay in its workspace"
                )

        hover_z = float(p["hover"][2])
        down_z = float(p["down"][2])
        lift_z = float(p["lift"][2])
        mid = corrected + (corrected_target - corrected) * mid_ratio

        p["detected_source_board"] = (
            float(raw[0]), float(raw[1])
        )
        p["source_board"] = (
            float(corrected[0]), float(corrected[1])
        )
        p["target_board"] = (
            float(corrected_target[0]), float(corrected_target[1])
        )
        p["hover"] = _roarm_pose_for_board(
            config, cfg, arm_key, corrected, hover_z
        )
        p["down"] = _roarm_pose_for_board(
            config, cfg, arm_key, corrected, down_z
        )
        p["mid"] = _roarm_pose_for_board(
            config, cfg, arm_key, mid, down_z
        )
        p["sweep"] = _roarm_pose_for_board(
            config, cfg, arm_key, corrected_target, down_z
        )
        p["lift"] = _roarm_pose_for_board(
            config, cfg, arm_key, corrected_target, lift_z
        )

        correction_metrics[f"{arm_key}_grip_correction_reason"] = reason
        correction_metrics[f"{arm_key}_raw_to_corrected_mm"] = float(
            np.linalg.norm(corrected - raw)
        )
        correction_metrics[f"{arm_key}_corrected_motion_mm"] = actual_motion
        for key, value in metrics.items():
            correction_metrics[f"{arm_key}_{key}"] = value

    plan.metrics.update(correction_metrics)
    plan.metrics["both_arms_required"] = True
    plan.reason += "; two-arm inward grip correction applied"
    return plan, "OK"


def _d16v3_make_dual_variant(
    obs,
    heat,
    cand,
    ci,
    feat,
    H,
    config,
    cfg,
    args,
    stretch_mm: float,
):
    local_args = copy.copy(args)
    local_args.d16_dual_stretch_mm = float(stretch_mm)
    local_args.d16_dual_max_stretch_mm = max(
        float(stretch_mm),
        float(getattr(args, "d16_dual_max_stretch_mm", 110.0)),
    )
    return _d16_make_dual_plan(
        obs, heat, cand, ci, feat, H, config, cfg, local_args
    )


def _d17_point_line_distance(point: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    p = np.asarray(point, dtype=np.float32)
    aa = np.asarray(a, dtype=np.float32)
    bb = np.asarray(b, dtype=np.float32)
    v = bb - aa
    n = float(np.linalg.norm(v))
    if n < 1e-6:
        return float(np.linalg.norm(p - aa))
    return float(abs(np.cross(v, p - aa)) / n)


def _d17_pose_shape_metrics(obs: Optional[BottomObservation], args) -> Dict[str, Any]:
    """Score how close the detected keypoints are to a stable pants silhouette."""
    if obs is None or obs.pose is None or not obs.pose.valid:
        return {
            "valid": False,
            "shape_good": False,
            "shape_score": 0.0,
            "primary_issue": "POSE_INVALID",
        }

    p = obs.pose
    wl = np.asarray(p.waist_left, dtype=np.float32)
    wc = np.asarray(p.waist_center, dtype=np.float32)
    wr = np.asarray(p.waist_right, dtype=np.float32)
    cr = np.asarray(p.crotch, dtype=np.float32)
    lh = np.asarray(p.left_hem_center, dtype=np.float32)
    rh = np.asarray(p.right_hem_center, dtype=np.float32)
    lower = np.asarray(p.lower_center, dtype=np.float32)

    waist_width = max(1.0, float(np.linalg.norm(wr - wl)))
    hem_gap = max(0.0, float(np.linalg.norm(rh - lh)))
    axis = lower - wc
    axis_len = max(1.0, float(np.linalg.norm(axis)))
    axis_u = _safe_unit(axis)
    lateral_u = np.asarray([-axis_u[1], axis_u[0]], dtype=np.float32)

    waist_mid = 0.5 * (wl + wr)
    waist_center_offset_ratio = float(np.linalg.norm(wc - waist_mid) / waist_width)
    waist_line_error_ratio = _d17_point_line_distance(wc, wl, wr) / waist_width
    crotch_axis_offset_ratio = _d17_point_line_distance(cr, wc, lower) / waist_width

    left_leg_len = float(np.linalg.norm(lh - cr))
    right_leg_len = float(np.linalg.norm(rh - cr))
    leg_balance = min(left_leg_len, right_leg_len) / max(1.0, max(left_leg_len, right_leg_len))
    hem_gap_ratio = hem_gap / waist_width
    axis_to_waist_ratio = axis_len / waist_width

    left_lat = float(np.dot(lh - lower, lateral_u))
    right_lat = float(np.dot(rh - lower, lateral_u))
    hem_side_balance = min(abs(left_lat), abs(right_lat)) / max(1.0, max(abs(left_lat), abs(right_lat)))

    conf_vals = []
    for name in BOTTOM_POSE_KPT_NAMES:
        if name in p.keypoint_conf:
            conf_vals.append(float(p.keypoint_conf[name]))
    mean_kpt_conf = float(np.mean(conf_vals)) if conf_vals else 0.0

    hem_min = float(getattr(args, "d17_shape_hem_gap_ratio_min", 0.58))
    hem_max = float(getattr(args, "d17_shape_hem_gap_ratio_max", 1.30))
    leg_min = float(getattr(args, "d17_shape_leg_balance_min", 0.84))
    waist_off_max = float(getattr(args, "d17_shape_waist_center_offset_max", 0.12))
    waist_line_max = float(getattr(args, "d17_shape_waist_line_error_max", 0.10))
    crotch_off_max = float(getattr(args, "d17_shape_crotch_axis_offset_max", 0.18))
    axis_ratio_min = float(getattr(args, "d17_shape_axis_to_waist_min", 0.80))
    good_thr = float(getattr(args, "d17_shape_score_good", 0.82))

    def upper_score(value: float, limit: float) -> float:
        return float(np.clip(1.0 - value / max(1e-6, limit * 1.6), 0.0, 1.0))

    if hem_gap_ratio < hem_min:
        hem_score = float(np.clip(hem_gap_ratio / max(1e-6, hem_min), 0.0, 1.0))
    elif hem_gap_ratio > hem_max:
        hem_score = float(np.clip(hem_max / max(1e-6, hem_gap_ratio), 0.0, 1.0))
    else:
        hem_score = 1.0

    leg_score = float(np.clip(leg_balance / max(1e-6, leg_min), 0.0, 1.0))
    waist_score = 0.5 * upper_score(waist_center_offset_ratio, waist_off_max) + \
                  0.5 * upper_score(waist_line_error_ratio, waist_line_max)
    crotch_score = upper_score(crotch_axis_offset_ratio, crotch_off_max)
    axis_score = float(np.clip(axis_to_waist_ratio / max(1e-6, axis_ratio_min), 0.0, 1.0))
    side_score = float(np.clip(hem_side_balance / 0.80, 0.0, 1.0))
    conf_score = float(np.clip(mean_kpt_conf / 0.55, 0.0, 1.0))

    shape_score = (
        0.25 * hem_score
        + 0.20 * leg_score
        + 0.20 * waist_score
        + 0.15 * crotch_score
        + 0.10 * axis_score
        + 0.05 * side_score
        + 0.05 * conf_score
    )

    issue_severity = {
        "HEM_COLLAPSED": max(0.0, (hem_min - hem_gap_ratio) / max(hem_min, 1e-6)),
        "WAIST_BENT": max(
            waist_center_offset_ratio / max(waist_off_max, 1e-6),
            waist_line_error_ratio / max(waist_line_max, 1e-6),
        ) - 1.0,
        "LEG_ASYMMETRY": max(0.0, (leg_min - leg_balance) / max(leg_min, 1e-6)),
        "CROTCH_OFF_AXIS": max(0.0, crotch_axis_offset_ratio / max(crotch_off_max, 1e-6) - 1.0),
        "BODY_COLLAPSED": max(0.0, (axis_ratio_min - axis_to_waist_ratio) / max(axis_ratio_min, 1e-6)),
    }
    primary_issue = max(issue_severity.items(), key=lambda kv: kv[1])[0]
    if max(issue_severity.values()) <= 0.0 and shape_score < good_thr:
        primary_issue = "GENERAL_SHAPE"

    shape_good = bool(
        shape_score >= good_thr
        and hem_min <= hem_gap_ratio <= hem_max
        and leg_balance >= leg_min
        and waist_center_offset_ratio <= waist_off_max
        and waist_line_error_ratio <= waist_line_max
        and crotch_axis_offset_ratio <= crotch_off_max
        and axis_to_waist_ratio >= axis_ratio_min
    )

    return {
        "valid": True,
        "shape_good": shape_good,
        "shape_score": float(shape_score),
        "primary_issue": primary_issue,
        "waist_width_mm": waist_width,
        "hem_gap_mm": hem_gap,
        "hem_gap_ratio": hem_gap_ratio,
        "left_leg_len_mm": left_leg_len,
        "right_leg_len_mm": right_leg_len,
        "leg_balance": leg_balance,
        "waist_center_offset_ratio": waist_center_offset_ratio,
        "waist_line_error_ratio": waist_line_error_ratio,
        "crotch_axis_offset_ratio": crotch_axis_offset_ratio,
        "axis_to_waist_ratio": axis_to_waist_ratio,
        "hem_side_balance": hem_side_balance,
        "mean_kpt_conf": mean_kpt_conf,
        "pants_axis_u": [float(axis_u[0]), float(axis_u[1])],
        "pants_lateral_u": [float(lateral_u[0]), float(lateral_u[1])],
    }


def _d17_build_dual_shape_plan(
    obs: BottomObservation,
    H: np.ndarray,
    config: Dict[str, Any],
    cfg: BottomSafetyConfig,
    args,
    action: str,
    raw_a: np.ndarray,
    raw_b: np.ndarray,
    center: np.ndarray,
    delta_a: np.ndarray,
    delta_b: np.ndarray,
    shape_metrics: Dict[str, Any],
) -> Optional[DualWrinkleStretchPlan]:
    raw_a = np.asarray(raw_a, dtype=np.float32)
    raw_b = np.asarray(raw_b, dtype=np.float32)
    center = np.asarray(center, dtype=np.float32)
    target_a = raw_a + np.asarray(delta_a, dtype=np.float32)
    target_b = raw_b + np.asarray(delta_b, dtype=np.float32)

    if float(raw_a[0]) <= float(raw_b[0]):
        assignment = {
            "arm2": (raw_a, target_a),
            "arm1": (raw_b, target_b),
        }
    else:
        assignment = {
            "arm2": (raw_b, target_b),
            "arm1": (raw_a, target_a),
        }

    clear_z = float(getattr(args, "b34_clear_z", -25.0))
    contact_z = float(getattr(args, "b34_press_z", -104.0))
    mid_ratio = float(np.clip(getattr(args, "b34_micro_mid_ratio", 0.55), 0.25, 0.80))
    arm_points: Dict[str, Dict[str, Any]] = {}

    for arm_key, (raw, target) in assignment.items():
        mid = raw + (target - raw) * mid_ratio
        arm_points[arm_key] = {
            "role": "dual_active",
            "detected_source_board": (float(raw[0]), float(raw[1])),
            "source_board": (float(raw[0]), float(raw[1])),
            "target_board": (float(target[0]), float(target[1])),
            "hover": _roarm_pose_for_board(config, cfg, arm_key, raw, clear_z),
            "down": _roarm_pose_for_board(config, cfg, arm_key, raw, contact_z),
            "mid": _roarm_pose_for_board(config, cfg, arm_key, mid, contact_z),
            "sweep": _roarm_pose_for_board(config, cfg, arm_key, target, contact_z),
            "lift": _roarm_pose_for_board(config, cfg, arm_key, target, clear_z),
        }

    plan = DualWrinkleStretchPlan(
        True,
        f"D17 pose-shape correction: {action}",
        action=action,
        candidate_index=-1,
        center_board=(float(center[0]), float(center[1])),
        arm_points=arm_points,
        metrics={
            **shape_metrics,
            "policy_family": "POSE_SHAPE",
            "both_arms_required": True,
        },
    )
    corrected, reason = _d16v3_apply_grip_correction(
        plan, obs, H, config, cfg, args
    )
    if corrected is None:
        return None
    corrected.reason += f"; {reason}"
    return corrected


def _d17_build_hold_shape_plan(
    obs: BottomObservation,
    H: np.ndarray,
    config: Dict[str, Any],
    cfg: BottomSafetyConfig,
    args,
    action: str,
    active_raw: np.ndarray,
    hold_raw: np.ndarray,
    active_delta: np.ndarray,
    center: np.ndarray,
    shape_metrics: Dict[str, Any],
) -> Optional[DualWrinkleStretchPlan]:
    active_raw = np.asarray(active_raw, dtype=np.float32)
    hold_raw = np.asarray(hold_raw, dtype=np.float32)
    active_target = active_raw + np.asarray(active_delta, dtype=np.float32)
    center = np.asarray(center, dtype=np.float32)

    active_arm = "arm2" if float(active_raw[0]) <= float(hold_raw[0]) else "arm1"
    hold_arm = "arm1" if active_arm == "arm2" else "arm2"

    clear_z = float(getattr(args, "b34_clear_z", -25.0))
    active_z = float(getattr(args, "b34_press_z", -104.0))
    hold_z = float(getattr(args, "b34_passive_hold_z", active_z))
    mid_ratio = float(np.clip(getattr(args, "b34_micro_mid_ratio", 0.55), 0.25, 0.80))
    active_mid = active_raw + (active_target - active_raw) * mid_ratio

    arm_points = {
        active_arm: {
            "role": "active",
            "detected_source_board": (float(active_raw[0]), float(active_raw[1])),
            "source_board": (float(active_raw[0]), float(active_raw[1])),
            "target_board": (float(active_target[0]), float(active_target[1])),
            "hover": _roarm_pose_for_board(config, cfg, active_arm, active_raw, clear_z),
            "down": _roarm_pose_for_board(config, cfg, active_arm, active_raw, active_z),
            "mid": _roarm_pose_for_board(config, cfg, active_arm, active_mid, active_z),
            "sweep": _roarm_pose_for_board(config, cfg, active_arm, active_target, active_z),
            "lift": _roarm_pose_for_board(config, cfg, active_arm, active_target, clear_z),
        },
        hold_arm: {
            "role": "hold",
            "detected_source_board": (float(hold_raw[0]), float(hold_raw[1])),
            "source_board": (float(hold_raw[0]), float(hold_raw[1])),
            "target_board": (float(hold_raw[0]), float(hold_raw[1])),
            "hover": _roarm_pose_for_board(config, cfg, hold_arm, hold_raw, clear_z),
            "down": _roarm_pose_for_board(config, cfg, hold_arm, hold_raw, hold_z),
            "mid": _roarm_pose_for_board(config, cfg, hold_arm, hold_raw, hold_z),
            "sweep": _roarm_pose_for_board(config, cfg, hold_arm, hold_raw, hold_z),
            "lift": _roarm_pose_for_board(config, cfg, hold_arm, hold_raw, clear_z),
        },
    }

    plan = DualWrinkleStretchPlan(
        True,
        f"D17 pose-shape correction: {action}",
        action=action,
        candidate_index=-1,
        center_board=(float(center[0]), float(center[1])),
        arm_points=arm_points,
        metrics={
            **shape_metrics,
            "policy_family": "POSE_SHAPE",
            "active_arm": active_arm,
            "hold_arm": hold_arm,
            "both_arms_required": True,
        },
    )
    corrected, reason = _d16v3_apply_grip_correction(
        plan, obs, H, config, cfg, args
    )
    if corrected is None:
        return None
    corrected.reason += f"; {reason}"
    return corrected


def _d17_build_pose_shape_plan(
    obs: BottomObservation,
    H: np.ndarray,
    config: Dict[str, Any],
    cfg: BottomSafetyConfig,
    args,
    shape_metrics: Dict[str, Any],
) -> Optional[DualWrinkleStretchPlan]:
    if not shape_metrics.get("valid", False) or shape_metrics.get("shape_good", False):
        return None
    if obs.pose is None:
        return None

    p = obs.pose
    wl = np.asarray(p.waist_left, dtype=np.float32)
    wr = np.asarray(p.waist_right, dtype=np.float32)
    wc = np.asarray(p.waist_center, dtype=np.float32)
    cr = np.asarray(p.crotch, dtype=np.float32)
    lh = np.asarray(p.left_hem_center, dtype=np.float32)
    rh = np.asarray(p.right_hem_center, dtype=np.float32)
    lower = np.asarray(p.lower_center, dtype=np.float32)

    axis_u = _safe_unit(lower - wc)
    lateral_u = np.asarray([-axis_u[1], axis_u[0]], dtype=np.float32)
    if float(np.dot(rh - lh, lateral_u)) < 0.0:
        lateral_u = -lateral_u

    aggressive = float(getattr(args, "d17_pose_pull_scale", 1.0))
    hem_pull = float(getattr(args, "d17_pose_hem_pull_mm", 130.0)) * aggressive
    waist_pull = float(getattr(args, "d17_pose_waist_pull_mm", 115.0)) * aggressive
    leg_pull = float(getattr(args, "d17_pose_leg_pull_mm", 145.0)) * aggressive
    body_pull = float(getattr(args, "d17_pose_body_pull_mm", 120.0)) * aggressive

    issue = str(shape_metrics.get("primary_issue", "GENERAL_SHAPE"))
    attempts: List[Tuple[str, Any]] = []

    hem_plan = lambda: _d17_build_dual_shape_plan(
        obs, H, config, cfg, args,
        "POSE_HEM_OPEN_STRETCH",
        lh, rh, lower,
        -lateral_u * hem_pull,
        lateral_u * hem_pull,
        shape_metrics,
    )
    waist_plan = lambda: _d17_build_dual_shape_plan(
        obs, H, config, cfg, args,
        "POSE_WAIST_OPEN_STRETCH",
        wl, wr, wc,
        -lateral_u * waist_pull,
        lateral_u * waist_pull,
        shape_metrics,
    )
    body_plan = lambda: _d17_build_dual_shape_plan(
        obs, H, config, cfg, args,
        "POSE_BODY_LENGTH_STRETCH",
        lh, rh, lower,
        axis_u * body_pull - lateral_u * (0.25 * body_pull),
        axis_u * body_pull + lateral_u * (0.25 * body_pull),
        shape_metrics,
    )

    left_len = float(shape_metrics.get("left_leg_len_mm", 0.0))
    right_len = float(shape_metrics.get("right_leg_len_mm", 0.0))
    if left_len <= right_len:
        leg_plan = lambda: _d17_build_hold_shape_plan(
            obs, H, config, cfg, args,
            "POSE_LEG_SYMMETRY_PULL",
            lh, rh,
            _safe_unit(lh - cr) * leg_pull,
            lower,
            shape_metrics,
        )
    else:
        leg_plan = lambda: _d17_build_hold_shape_plan(
            obs, H, config, cfg, args,
            "POSE_LEG_SYMMETRY_PULL",
            rh, lh,
            _safe_unit(rh - cr) * leg_pull,
            lower,
            shape_metrics,
        )

    if issue == "HEM_COLLAPSED":
        attempts = [("hem", hem_plan), ("leg", leg_plan), ("waist", waist_plan)]
    elif issue == "WAIST_BENT":
        attempts = [("waist", waist_plan), ("hem", hem_plan), ("leg", leg_plan)]
    elif issue in ("LEG_ASYMMETRY", "CROTCH_OFF_AXIS"):
        attempts = [("leg", leg_plan), ("hem", hem_plan), ("waist", waist_plan)]
    elif issue == "BODY_COLLAPSED":
        attempts = [("body", body_plan), ("hem", hem_plan), ("waist", waist_plan)]
    else:
        attempts = [("hem", hem_plan), ("waist", waist_plan), ("leg", leg_plan), ("body", body_plan)]

    for _, builder in attempts:
        try:
            plan = builder()
        except Exception as exc:
            print(f"[D20-SHAPE] builder failed: {exc!r}")
            continue
        if plan is not None and plan.ok:
            plan.reason += (
                f"; shape_score={shape_metrics.get('shape_score', 0.0):.3f} "
                f"issue={issue}"
            )
            return plan
    return None

def build_dual_wrinkle_stretch_plan(obs: BottomObservation, heat: Optional[WrinkleHeatmapPlan],
                                    H: np.ndarray, config: Dict[str, Any],
                                    cfg: BottomSafetyConfig, args) -> DualWrinkleStretchPlan:
    """D16_V3 two-arm-only wrinkle policy.

    LONG_LINEAR:
      DUAL_OPPOSITE_STRETCH, fallback ACTIVE_PULL_PASSIVE_HOLD

    SHORT_LINEAR:
      ACTIVE_PULL_PASSIVE_HOLD, fallback short DUAL_OPPOSITE_STRETCH

    EDGE:
      EDGE_SWEEP_WITH_HOLD, fallback short DUAL_OPPOSITE_STRETCH

    BROAD_ROUND / AMBIGUOUS:
      TWO_ARM_PRE_SMOOTH (hold + short active pull), fallback short dual stretch

    No one-arm plan is allowed.
    """
    if obs is None or obs.mask is None or not obs.valid:
        return DualWrinkleStretchPlan(
            False,
            f"invalid observation: {None if obs is None else obs.reason}",
        )
    if heat is None or heat.status != "OK" or not heat.candidates:
        return DualWrinkleStretchPlan(False, "D16_V3 usable wrinkle candidate 없음")
    if bool(getattr(args, "tta_require_ok_for_motion", True)) and obs.pose is not None:
        if str(obs.pose.tta_state) != "OK":
            return DualWrinkleStretchPlan(
                False,
                f"pose TTA state={obs.pose.tta_state}; robot motion blocked",
            )

    warnings: List[str] = []
    proposals: List[Tuple[float, DualWrinkleStretchPlan]] = []
    mask_area = max(1.0, float(obs.mask.area_px))
    aggression = _d16_aggression_scale(args)

    for ci, cand in enumerate(heat.candidates):
        feat = _d16_classify(cand, H, mask_area, args)
        preferred = str(feat["preferred_action"])
        attempts: List[Tuple[str, Any]] = []

        if preferred == "DUAL_OPPOSITE_STRETCH":
            attempts = [
                (
                    "DUAL_OPPOSITE_STRETCH",
                    lambda: _d16v3_make_dual_variant(
                        obs, heat, cand, ci, feat, H, config, cfg, args,
                        float(getattr(args, "d16_dual_stretch_mm", 85.0)),
                    ),
                ),
                (
                    "ACTIVE_PULL_PASSIVE_HOLD",
                    lambda: _d16_call_hold_pull(
                        obs, heat, cand, H, config, cfg, args,
                        "ACTIVE_PULL_PASSIVE_HOLD",
                        float(getattr(args, "d16_long_fallback_pull_mm", 135.0)) * aggression,
                        float(getattr(args, "d16_hold_pull_min_actual_mm", 45.0)),
                    ),
                ),
            ]

        elif preferred == "EDGE_SWEEP_WITH_HOLD":
            attempts = [
                (
                    "EDGE_SWEEP_WITH_HOLD",
                    lambda: _d16_call_hold_pull(
                        obs, heat, cand, H, config, cfg, args,
                        "EDGE_SWEEP_WITH_HOLD",
                        float(getattr(args, "d16_edge_sweep_mm", 130.0)) * aggression,
                        float(getattr(args, "d16_hold_pull_min_actual_mm", 45.0)),
                    ),
                ),
                (
                    "DUAL_OPPOSITE_STRETCH",
                    lambda: _d16v3_make_dual_variant(
                        obs, heat, cand, ci, feat, H, config, cfg, args,
                        float(getattr(args, "d16_v3_edge_fallback_dual_mm", 65.0)),
                    ),
                ),
            ]

        elif preferred == "ACTIVE_PULL_PASSIVE_HOLD":
            attempts = [
                (
                    "ACTIVE_PULL_PASSIVE_HOLD",
                    lambda: _d16_call_hold_pull(
                        obs, heat, cand, H, config, cfg, args,
                        "ACTIVE_PULL_PASSIVE_HOLD",
                        float(getattr(args, "d16_short_pull_mm", 115.0)) * aggression,
                        float(getattr(args, "d16_hold_pull_min_actual_mm", 45.0)),
                    ),
                ),
                (
                    "DUAL_OPPOSITE_STRETCH",
                    lambda: _d16v3_make_dual_variant(
                        obs, heat, cand, ci, feat, H, config, cfg, args,
                        float(getattr(args, "d16_v3_short_fallback_dual_mm", 60.0)),
                    ),
                ),
            ]

        else:
            # Broad/ambiguous wrinkle: still use both arms.
            attempts = [
                (
                    "TWO_ARM_PRE_SMOOTH",
                    lambda: _d16_call_hold_pull(
                        obs, heat, cand, H, config, cfg, args,
                        "TWO_ARM_PRE_SMOOTH",
                        float(getattr(args, "d16_v3_two_arm_pre_smooth_mm", 70.0)) * aggression,
                        float(getattr(args, "d16_v3_pre_smooth_min_actual_mm", 30.0)),
                    ),
                ),
                (
                    "DUAL_OPPOSITE_STRETCH",
                    lambda: _d16v3_make_dual_variant(
                        obs, heat, cand, ci, feat, H, config, cfg, args,
                        float(getattr(args, "d16_v3_round_fallback_dual_mm", 50.0)),
                    ),
                ),
            ]

        chosen: Optional[DualWrinkleStretchPlan] = None

        for attempt_name, builder in attempts:
            try:
                plan = builder()
            except Exception as exc:
                warnings.append(
                    f"W{ci+1}: {attempt_name} builder error {exc!r}"
                )
                continue

            if plan is None or not plan.ok:
                why = "none" if plan is None else plan.reason
                warnings.append(
                    f"W{ci+1}: {attempt_name} unavailable ({why})"
                )
                continue

            plan.action = attempt_name
            plan.candidate_index = ci
            plan.metrics.update(feat)
            plan.metrics["preferred_action"] = preferred
            plan.metrics["selected_two_arm_action"] = attempt_name
            plan.metrics["d20_priority_tier"] = int(cand.get("d20_priority_tier", 0))
            plan.metrics["d20_tier_name"] = str(cand.get("d20_tier_name", ""))
            plan.metrics["d20_shadow_mean"] = float(cand.get("d20_shadow_mean", 0.0))
            plan.metrics["d20_shadow_max"] = float(cand.get("d20_shadow_max", 0.0))
            plan.metrics["d20_shadow_support_ratio"] = float(cand.get("d20_shadow_support_ratio", 0.0))
            plan.metrics["d21_track_id"] = int(cand.get("d21_track_id", -1))
            plan.metrics["d21_candidate_area_px"] = float(cand.get("area_px", 0.0))
            plan.metrics["d21_candidate_length_mm"] = float(cand.get("major_length_mm", 0.0))
            plan.metrics["d21_candidate_dark_mean"] = float(cand.get("d20_shadow_mean", 0.0))
            plan.metrics["d21_signature"] = _d21_pose_signature(cand, obs, H)
            plan.metrics["d21_shape_class"] = str(cand.get("d21_shape_class", "IRREGULAR"))

            corrected, correction_reason = _d16v3_apply_grip_correction(
                plan,
                obs,
                H,
                config,
                cfg,
                args,
            )
            if corrected is None:
                warnings.append(
                    f"W{ci+1}: {attempt_name} grip correction failed "
                    f"({correction_reason})"
                )
                continue

            # Final invariant: no successful action may omit either arm.
            if (
                "arm1" not in corrected.arm_points
                or "arm2" not in corrected.arm_points
            ):
                warnings.append(
                    f"W{ci+1}: {attempt_name} rejected by two-arm invariant"
                )
                continue

            chosen = corrected
            break

        if chosen is None:
            continue

        action_bonus = {
            "DUAL_OPPOSITE_STRETCH": 85.0,
            "EDGE_SWEEP_WITH_HOLD": 70.0,
            "ACTIVE_PULL_PASSIVE_HOLD": 60.0,
            "TWO_ARM_PRE_SMOOTH": 35.0,
        }.get(chosen.action, 0.0)

        score = (
            float(cand.get("priority_score", 0.0))
            + float(cand.get("max_score", 0.0)) * 0.6
            + float(feat["length_mm"]) * 0.18
            + float(feat["blob_ratio"]) * 1200.0
            + action_bonus
        )
        chosen.metrics["policy_score"] = float(score)
        chosen.metrics["all_wrinkles_use_two_arms"] = True
        chosen.safety_warnings = list(warnings)
        proposals.append((score, chosen))

    if proposals:
        proposals.sort(key=lambda item: item[0], reverse=True)
        selected = proposals[0][1]
        selected.reason += f"; selected_score={proposals[0][0]:.1f}"
        return selected

    reason = (
        "D16_V3 two-arm-only policy에서 실행 가능한 후보 없음"
    )
    if warnings:
        reason += "; " + " | ".join(warnings[-8:])
    return DualWrinkleStretchPlan(
        False,
        reason,
        safety_warnings=warnings,
    )

def print_dual_wrinkle_plan(plan: DualWrinkleStretchPlan) -> None:
    print("\n========== D23_V2 CENTER-OUTWARD WRINKLE PLAN ==========")
    print(f"ok={plan.ok} action={plan.action} candidate={plan.candidate_index + 1}")
    print(f"reason={plan.reason}")
    print(f"metrics={json.dumps(plan.metrics, ensure_ascii=False, default=str)}")
    for arm_key, p in plan.arm_points.items():
        print(
            f"{arm_key.upper()} role={p.get('role')} "
            f"start={p.get('source_board')} -> target={p.get('target_board')}"
        )
    for warning in plan.safety_warnings[-8:]:
        print('warning:', warning)
    print('===================================================\n')


def _run_parallel(functions: Sequence[Tuple[str, Any]]) -> None:
    errors: List[Tuple[str, BaseException]] = []
    lock = threading.Lock()

    def runner(name, fn):
        try:
            fn()
        except BaseException as e:
            with lock:
                errors.append((name, e))

    threads = []
    for name, fn in functions:
        th = threading.Thread(target=runner, args=(name, fn), daemon=True)
        th.start()
        threads.append(th)
    for th in threads:
        th.join()
    if errors:
        name, err = errors[0]
        raise RuntimeError(f"{name}: {repr(err)}")


def _dual_move(arms: Dict[str, RoArmSerial], plan: DualWrinkleStretchPlan,
               pose_key: str, cfg: BottomSafetyConfig, move_command: int,
               speed: float, parallel: bool = True) -> None:
    calls = []
    for arm_key in ("arm2", "arm1"):
        xyz = plan.arm_points[arm_key][pose_key]
        arm = arms[arm_key]
        calls.append((
            f"{arm_key}-{pose_key}",
            lambda arm=arm, xyz=xyz: arm.move_goal(
                float(xyz[0]), float(xyz[1]), float(xyz[2]),
                cfg.tool_t_rad, float(speed), move_command=move_command)
        ))
        print(f"[D8-MOVE] {arm_key.upper()} {pose_key}: ({xyz[0]:.1f},{xyz[1]:.1f},{xyz[2]:.1f})")
    if parallel:
        _run_parallel(calls)
    else:
        for name, fn in calls:
            fn()


def _dual_gripper(arms: Dict[str, RoArmSerial], angle: float, delay: float = 0.25) -> None:
    _run_parallel([
        ("arm2-gripper", lambda: arms["arm2"].set_gripper(float(angle), delay=delay)),
        ("arm1-gripper", lambda: arms["arm1"].set_gripper(float(angle), delay=delay)),
    ])


def _d16_execute_hold_pull_base(plan: DualWrinkleStretchPlan,
                                      arms: Dict[str, RoArmSerial],
                                      cfg: BottomSafetyConfig,
                                      move_command: int = 104,
                                      args=None) -> bool:
    """D17 true-grip hold/pull: approach open, close at contact, hold until active clears."""
    if not plan.ok:
        return False

    active_arm = str(plan.metrics.get("active_arm", ""))
    hold_arm = str(plan.metrics.get("hold_arm", ""))
    if active_arm not in plan.arm_points or active_arm not in arms:
        print(f"[D17-EXEC] active arm unavailable: {active_arm}")
        return False
    if hold_arm not in plan.arm_points or hold_arm not in arms or active_arm == hold_arm:
        print(f"[D17-EXEC] hold arm unavailable/invalid: {hold_arm}")
        return False

    active_gripper = float(getattr(args, "b34_press_gripper", 3.15))
    hold_gripper = float(getattr(args, "b34_passive_hold_gripper", 3.18))
    fast = bool(getattr(args, "d23_fast_motion", True))
    speed = float(getattr(args, "b34_action_speed", 0.26))
    if fast:
        speed = max(speed, float(getattr(args, "d23_hold_pull_speed_min", 0.38)))
    repeat = max(1, int(getattr(args, "d17_grip_close_repeat", 2)))
    if fast:
        repeat = min(repeat, max(1, int(getattr(args, "d23_fast_close_repeat", 1))))
    close_wait = float(getattr(args, "d17_grip_close_wait_s", 0.30))
    close_settle = float(getattr(args, "d17_grip_close_settle_s", 0.55))
    contact_open_wait = float(getattr(args, "d17_contact_open_wait_s", 0.20))

    sec_per_mm = float(getattr(args, "d16_motion_sec_per_mm", 0.018))
    min_wait = float(getattr(args, "d16_min_stage_wait_s", 0.65))
    max_wait = float(getattr(args, "d16_max_stage_wait_s", 4.20))
    base_wait = float(getattr(args, "d16_stage_base_wait_s", 0.22))
    hold_settle = float(getattr(args, "d16_hold_settle_s", 0.80))
    down_extra = float(getattr(args, "d16_down_extra_wait_s", 0.30))
    pull_extra = float(getattr(args, "d16_pull_extra_wait_s", 0.35))
    lift_extra = float(getattr(args, "d16_lift_extra_wait_s", 0.65))
    hold_release_delay = float(getattr(args, "d16_hold_release_delay_s", 0.55))
    if fast:
        scale = float(getattr(args, "d23_motion_wait_scale", 0.55))
        sec_per_mm *= scale
        base_wait *= scale
        hold_settle *= scale
        down_extra *= scale
        pull_extra *= scale
        lift_extra *= scale
        hold_release_delay *= scale
        close_wait *= scale
        close_settle *= scale
        contact_open_wait *= scale
        min_wait = min(min_wait, float(getattr(args, "d23_fast_min_stage_wait_s", 0.32)))
        max_wait = min(max_wait, float(getattr(args, "d23_fast_max_stage_wait_s", 2.40)))
    reassert_hold = bool(getattr(args, "d16_reassert_hold", True))

    ap = plan.arm_points[active_arm]
    hp = plan.arm_points[hold_arm]

    def wait_for(a, b, extra=0.0):
        dist = float(np.linalg.norm(np.asarray(b, dtype=np.float32) - np.asarray(a, dtype=np.float32)))
        factor = 0.22 / max(0.05, speed)
        return float(np.clip(base_wait + dist * sec_per_mm * factor + extra, min_wait, max_wait))

    def move(arm_key, pose_key, label):
        xyz = plan.arm_points[arm_key][pose_key]
        print(f"[D17-{label}] {arm_key.upper()} ({xyz[0]:.1f},{xyz[1]:.1f},{xyz[2]:.1f})")
        arms[arm_key].move_goal(
            float(xyz[0]), float(xyz[1]), float(xyz[2]),
            cfg.tool_t_rad, speed, move_command=move_command
        )

    def close_at_contact(arm_key: str, angle: float, label: str):
        print(f"[D17-{label}] close={angle:.3f} repeat={repeat}")
        for i in range(repeat):
            arms[arm_key].set_gripper(angle, delay=0.20)
            time.sleep(max(0.0, close_wait))
        time.sleep(max(0.0, close_settle))

    try:
        _run_parallel([
            ("active-open", lambda: arms[active_arm].set_gripper(cfg.grip_open, delay=0.18)),
            ("hold-open", lambda: arms[hold_arm].set_gripper(cfg.grip_open, delay=0.18)),
        ])

        if bool(getattr(args, "d23_parallel_high_z", True)):
            _run_parallel([
                ("hold-hover", lambda: move(hold_arm, "hover", "HOLD-HOVER")),
                ("active-hover", lambda: move(active_arm, "hover", "ACTIVE-HOVER")),
            ])
            time.sleep(max(min_wait, float(getattr(args, "d23_high_z_wait_s", 0.55))))
        else:
            move(hold_arm, "hover", "HOLD-HOVER")
            time.sleep(min_wait)
            move(active_arm, "hover", "ACTIVE-HOVER")
            time.sleep(min_wait)

        if cfg.hover_only:
            move(active_arm, "lift", "ACTIVE-END-HOVER")
            time.sleep(wait_for(ap["hover"], ap["lift"]))
            _run_parallel([
                ("active-open", lambda: arms[active_arm].set_gripper(cfg.grip_open, delay=0.15)),
                ("hold-open", lambda: arms[hold_arm].set_gripper(cfg.grip_open, delay=0.15)),
            ])
            return True

        # 1) Hold arm contacts and truly grips first.
        move(hold_arm, "down", "HOLD-DOWN")
        time.sleep(wait_for(hp["hover"], hp["down"], down_extra))
        arms[hold_arm].set_gripper(cfg.grip_open, delay=0.12)
        time.sleep(contact_open_wait)
        close_at_contact(hold_arm, hold_gripper, "HOLD-GRIP")
        time.sleep(hold_settle)

        # 2) Active arm contacts and grips.
        move(active_arm, "down", "ACTIVE-DOWN")
        time.sleep(wait_for(ap["hover"], ap["down"], down_extra))
        arms[active_arm].set_gripper(cfg.grip_open, delay=0.12)
        time.sleep(contact_open_wait)
        close_at_contact(active_arm, active_gripper, "ACTIVE-GRIP")

        if reassert_hold:
            arms[hold_arm].set_gripper(hold_gripper, delay=0.15)
            arms[hold_arm].move_goal(
                float(hp["down"][0]), float(hp["down"][1]), float(hp["down"][2]),
                cfg.tool_t_rad, speed, move_command=move_command
            )
            time.sleep(0.25)

        # 3) Full active pull while hold arm remains down and closed.
        move(active_arm, "mid", "ACTIVE-MID-PULL")
        time.sleep(wait_for(ap["down"], ap["mid"], pull_extra))
        if reassert_hold:
            arms[hold_arm].set_gripper(hold_gripper, delay=0.12)

        move(active_arm, "sweep", "ACTIVE-FULL-PULL")
        time.sleep(wait_for(ap["mid"], ap["sweep"], pull_extra))
        if reassert_hold:
            arms[hold_arm].set_gripper(hold_gripper, delay=0.12)

        move(active_arm, "lift", "ACTIVE-LIFT")
        time.sleep(wait_for(ap["sweep"], ap["lift"], lift_extra))
        print("[D17-SYNC] active pull/lift complete; hold remains down and closed")
        time.sleep(hold_release_delay)

        # 4) Hold rises last.
        move(hold_arm, "lift", "HOLD-LIFT-LAST")
        time.sleep(wait_for(hp["down"], hp["lift"], lift_extra))

        _run_parallel([
            ("active-open", lambda: arms[active_arm].set_gripper(cfg.grip_open, delay=0.18)),
            ("hold-open", lambda: arms[hold_arm].set_gripper(cfg.grip_open, delay=0.18)),
        ])
        return True
    except Exception as exc:
        print(f"[D17-HOLD-PULL-ERROR] {exc!r}")
        return False


def _d16_wait_seconds(a, b, speed: float, args, extra: float = 0.0) -> float:
    dist = float(np.linalg.norm(np.asarray(b, dtype=np.float32) - np.asarray(a, dtype=np.float32)))
    sec_per_mm = float(getattr(args, 'd16_motion_sec_per_mm', 0.025))
    base = float(getattr(args, 'd16_stage_base_wait_s', 0.35))
    min_wait = float(getattr(args, 'd16_min_stage_wait_s', 0.85))
    max_wait = float(getattr(args, 'd16_max_stage_wait_s', 4.80))
    if bool(getattr(args, 'd23_fast_motion', True)):
        scale = float(getattr(args, 'd23_motion_wait_scale', 0.55))
        sec_per_mm *= scale
        base *= scale
        extra *= scale
        min_wait = min(min_wait, float(getattr(args, 'd23_fast_min_stage_wait_s', 0.32)))
        max_wait = min(max_wait, float(getattr(args, 'd23_fast_max_stage_wait_s', 2.40)))
    speed_factor = 0.18 / max(0.05, float(speed))
    return float(np.clip(base + dist * sec_per_mm * speed_factor + extra, min_wait, max_wait))


def _d16_execute_dual(plan, arms, cfg, move_command, args):
    """D17 dual true-grip stretch: both arms close at contact, then pull apart."""
    keys = ("arm2", "arm1")
    if any(k not in plan.arm_points or k not in arms for k in keys):
        print("[D17-EXEC] dual action requires ARM1 and ARM2")
        return False

    fast = bool(getattr(args, "d23_fast_motion", True))
    speed = float(getattr(args, "b34_action_speed", 0.26))
    if fast:
        speed = max(speed, float(getattr(args, "d23_dual_speed_min", 0.38)))
    gripper = float(getattr(args, "d16_dual_gripper", 3.15))
    repeat = max(1, int(getattr(args, "d17_grip_close_repeat", 2)))
    if fast:
        repeat = min(repeat, max(1, int(getattr(args, "d23_fast_close_repeat", 1))))
    close_wait = float(getattr(args, "d17_grip_close_wait_s", 0.30))
    close_settle = float(getattr(args, "d17_grip_close_settle_s", 0.55))
    contact_open_wait = float(getattr(args, "d17_contact_open_wait_s", 0.20))
    if fast:
        scale = float(getattr(args, "d23_motion_wait_scale", 0.55))
        close_wait *= scale
        close_settle *= scale
        contact_open_wait *= scale
    down_extra = float(getattr(args, "d16_down_extra_wait_s", 0.30))
    pull_extra = float(getattr(args, "d16_pull_extra_wait_s", 0.35))
    lift_extra = float(getattr(args, "d16_lift_extra_wait_s", 0.65))

    def move(k, pose_key, label):
        xyz = plan.arm_points[k][pose_key]
        print(f"[D17-{label}] {k.upper()} ({xyz[0]:.1f},{xyz[1]:.1f},{xyz[2]:.1f})")
        arms[k].move_goal(
            float(xyz[0]), float(xyz[1]), float(xyz[2]),
            cfg.tool_t_rad, speed, move_command=move_command
        )

    def move_parallel(pose_key, label):
        _run_parallel([
            (f"{k}-{pose_key}", lambda k=k: move(k, pose_key, label))
            for k in keys
        ])

    try:
        _dual_gripper(arms, cfg.grip_open, delay=0.18)
        if bool(getattr(args, "d23_parallel_high_z", True)):
            move_parallel("hover", "DUAL-HOVER")
            time.sleep(max(
                _d16_wait_seconds(
                    plan.arm_points["arm2"]["lift"], plan.arm_points["arm2"]["hover"],
                    speed, args, 0.0,
                ),
                float(getattr(args, "d23_high_z_wait_s", 0.55)),
            ))
        else:
            move("arm2", "hover", "DUAL-ARM2-HOVER")
            time.sleep(float(getattr(args, "d16_min_stage_wait_s", 0.65)))
            move("arm1", "hover", "DUAL-ARM1-HOVER")
            time.sleep(float(getattr(args, "d16_min_stage_wait_s", 0.65)))

        if cfg.hover_only:
            move_parallel("lift", "DUAL-END-HOVER")
            time.sleep(1.0)
            _dual_gripper(arms, cfg.grip_open, delay=0.15)
            return True

        for k in keys:
            p = plan.arm_points[k]
            move(k, "down", f"DUAL-{k.upper()}-DOWN")
            time.sleep(_d16_wait_seconds(p["hover"], p["down"], speed, args, down_extra))
            arms[k].set_gripper(cfg.grip_open, delay=0.12)
            time.sleep(contact_open_wait)

        for i in range(repeat):
            print(f"[D17-DUAL-GRIP] repeat {i+1}/{repeat} angle={gripper:.3f}")
            _dual_gripper(arms, gripper, delay=0.20)
            time.sleep(close_wait)
        time.sleep(close_settle)

        move_parallel("mid", "DUAL-MID-STRETCH")
        time.sleep(max(
            _d16_wait_seconds(plan.arm_points[k]["down"], plan.arm_points[k]["mid"], speed, args, pull_extra)
            for k in keys
        ))
        move_parallel("sweep", "DUAL-FULL-STRETCH")
        time.sleep(max(
            _d16_wait_seconds(plan.arm_points[k]["mid"], plan.arm_points[k]["sweep"], speed, args, pull_extra)
            for k in keys
        ))
        move_parallel("lift", "DUAL-LIFT")
        time.sleep(max(
            _d16_wait_seconds(plan.arm_points[k]["sweep"], plan.arm_points[k]["lift"], speed, args, lift_extra)
            for k in keys
        ))
        _dual_gripper(arms, cfg.grip_open, delay=0.18)
        return True
    except Exception as exc:
        print(f"[D17-DUAL-ERROR] {exc!r}")
        return False


def _d16_execute_pre_smooth(plan, arms, cfg, move_command, args):
    arm_key = str(plan.metrics.get('active_arm', ''))
    if arm_key not in plan.arm_points or arm_key not in arms:
        print(f'[D16-EXEC] pre-smooth arm unavailable: {arm_key}')
        return False
    p = plan.arm_points[arm_key]
    fast = bool(getattr(args, 'd23_fast_motion', True))
    speed = float(getattr(args, 'b34_action_speed', 0.18))
    if fast:
        speed = max(speed, float(getattr(args, 'd23_press_speed_min', 0.45)))
    gripper = float(plan.metrics.get('d23v2_press_gripper', getattr(args, 'd23v2_press_gripper', getattr(args, 'b34_press_gripper', 3.05))))
    down_extra = float(getattr(args, 'd16_down_extra_wait_s', 0.45))
    pull_extra = float(getattr(args, 'd16_pull_extra_wait_s', 0.60))
    lift_extra = float(getattr(args, 'd16_lift_extra_wait_s', 0.90))

    def move(pose_key, label):
        xyz = p[pose_key]
        print(f'[D16-{label}] {arm_key.upper()} ({xyz[0]:.1f},{xyz[1]:.1f},{xyz[2]:.1f})')
        arms[arm_key].move_goal(float(xyz[0]), float(xyz[1]), float(xyz[2]),
                                cfg.tool_t_rad, speed, move_command=move_command)

    print(f'[D23V2-CLOSED-PRESS] {arm_key.upper()} gripper stays closed at {gripper:.3f}; no pre-grip opening')
    arms[arm_key].set_gripper(gripper, delay=0.18)
    move('hover', 'PRE-HOVER'); time.sleep(max(_d16_wait_seconds(p['lift'], p['hover'], speed, args, 0.0), float(getattr(args, 'd23_high_z_wait_s', 0.55))))
    if cfg.hover_only:
        move('lift', 'PRE-END-HOVER'); time.sleep(_d16_wait_seconds(p['hover'], p['lift'], speed, args, 0.0))
        arms[arm_key].set_gripper(cfg.grip_open, delay=0.15)
        return True
    move('down', 'PRE-DOWN'); time.sleep(_d16_wait_seconds(p['hover'], p['down'], speed, args, down_extra))
    move('mid', 'PRE-MID'); time.sleep(_d16_wait_seconds(p['down'], p['mid'], speed, args, pull_extra))
    move('sweep', 'PRE-SWEEP'); time.sleep(_d16_wait_seconds(p['mid'], p['sweep'], speed, args, pull_extra))
    move('lift', 'PRE-LIFT'); time.sleep(_d16_wait_seconds(p['sweep'], p['lift'], speed, args, lift_extra))
    arms[arm_key].set_gripper(cfg.grip_open, delay=0.15)
    return True


def execute_dual_wrinkle_stretch_plan(plan: DualWrinkleStretchPlan,
                                      arms: Dict[str, RoArmSerial],
                                      cfg: BottomSafetyConfig,
                                      move_command: int = 104,
                                      args=None,
                                      skip_confirm: bool = False) -> bool:
    print_dual_wrinkle_plan(plan)
    if not plan.ok:
        print(f"[D17-EXEC] blocked: {plan.reason}")
        return False
    single_arm_actions = {"D22_PRESS_SWEEP", "D23V2_CENTER_OUT_PRESS", "D23V2_LOWER_LEG_OUTWARD_PRESS"}
    if plan.action not in single_arm_actions and ("arm1" not in plan.arm_points or "arm2" not in plan.arm_points):
        print("[D22-EXEC] blocked: this action requires ARM1 and ARM2")
        return False
    if plan.action in single_arm_actions and len(plan.arm_points) != 1:
        print("[D22-EXEC] blocked: PRESS_SWEEP requires exactly one safe active arm")
        return False
    if cfg.dry_run or not cfg.send:
        print(f"[D17-EXEC] preview only: send={cfg.send} dry_run={cfg.dry_run}")
        return True
    if not skip_confirm and not confirm_if_needed(cfg, f"D17 {plan.action}"):
        print("[D17-EXEC] cancelled")
        return False

    try:
        dual_actions = {
            "DUAL_OPPOSITE_STRETCH",
            "POSE_HEM_OPEN_STRETCH",
            "POSE_WAIST_OPEN_STRETCH",
            "POSE_BODY_LENGTH_STRETCH",
        }
        hold_actions = {
            "ACTIVE_PULL_PASSIVE_HOLD",
            "EDGE_SWEEP_WITH_HOLD",
            "TWO_ARM_PRE_SMOOTH",
            "POSE_LEG_SYMMETRY_PULL",
            "D22_LOWER_LEG_SYMMETRY_PULL",
        }

        geometry_actions = {"D21_LINE_NORMAL_GRIP_PULL", "D21_ARC_RADIAL_GRIP_PULL"}
        if plan.action in geometry_actions:
            ok = _d21_execute_geometry_grip_pull(plan, arms, cfg, move_command, args)
        elif plan.action in {"D22_PRESS_SWEEP", "D23V2_CENTER_OUT_PRESS", "D23V2_LOWER_LEG_OUTWARD_PRESS"}:
            ok = _d16_execute_pre_smooth(plan, arms, cfg, move_command, args)
        elif plan.action in dual_actions:
            ok = _d16_execute_dual(plan, arms, cfg, move_command, args)
        elif plan.action in hold_actions:
            ok = _d16_execute_hold_pull_base(plan, arms, cfg, move_command, args)
        else:
            print(f"[D17-EXEC] unknown action: {plan.action}")
            return False

        if ok and cfg.standby_after_motion:
            move_arms_to_standby(
                arms, cfg, move_command=move_command,
                reason=f"after D17 {plan.action}",
            )
        print(f"[D17-EXEC] completed action={plan.action} ok={ok}")
        return bool(ok)
    except Exception as exc:
        print(f"[D17-EXEC] error: {exc!r}")
        return False

def evaluate_wrinkle_change(before_heat: Optional[WrinkleHeatmapPlan],
                            after_heat: Optional[WrinkleHeatmapPlan],
                            before_obs: Optional[BottomObservation],
                            after_obs: Optional[BottomObservation],
                            args) -> Dict[str, Any]:
    before = _dual_plan_metrics(before_heat, before_obs)
    after = _dual_plan_metrics(after_heat, after_obs)

    ratio_drop = float(before["wrinkle_ratio"] - after["wrinkle_ratio"])
    blob_drop = float(before["max_blob_area_px"] - after["max_blob_area_px"])
    ratio_rel = ratio_drop / max(1e-6, float(before["wrinkle_ratio"]))
    blob_rel = blob_drop / max(1.0, float(before["max_blob_area_px"]))
    mask_ratio = float(after["mask_area_px"]) / max(1.0, float(before["mask_area_px"]))

    center_shift = 0.0
    if before_obs is not None and after_obs is not None:
        if before_obs.mask is not None and after_obs.mask is not None:
            center_shift = float(np.linalg.norm(
                after_obs.mask.center_board - before_obs.mask.center_board
            ))

    min_ratio_improve = float(getattr(args, "wrinkle_min_ratio_improvement", 0.05))
    min_blob_improve = float(getattr(args, "wrinkle_min_blob_improvement", 0.10))
    max_center_shift = float(getattr(args, "wrinkle_max_center_shift_mm", 45.0))
    min_mask_ratio = float(getattr(args, "wrinkle_min_mask_area_ratio", 0.90))

    shape_ok = bool(mask_ratio >= min_mask_ratio and center_shift <= max_center_shift)
    improved = bool(
        shape_ok and (
            ratio_rel >= min_ratio_improve
            or blob_rel >= min_blob_improve
        )
    )
    absolute_good = _d14_finish_metrics_good(after, after_obs, args)
    worsened = bool(
        not shape_ok
        or ratio_rel < -0.08
        or blob_rel < -0.15
    )

    if absolute_good:
        state = "GOOD_ENOUGH_CANDIDATE_NEEDS_CONFIRMATION"
    elif improved:
        state = "IMPROVED_BUT_WRINKLES_REMAIN"
    elif worsened:
        state = "WORSENED_REPLAN"
    else:
        state = "WRINKLES_REMAIN_REPLAN"

    return {
        "state": state,
        "absolute_good": bool(absolute_good),
        "before": before,
        "after": after,
        "ratio_relative_improvement": ratio_rel,
        "blob_relative_improvement": blob_rel,
        "mask_area_ratio": mask_ratio,
        "center_shift_mm": center_shift,
        "finish_thresholds": {
            "wrinkle_ratio_max": float(getattr(args, "finish_wrinkle_ratio", 0.035)),
            "max_blob_ratio_max": float(getattr(args, "finish_max_blob_ratio", 0.012)),
            "blob_count_max": int(getattr(args, "finish_max_blob_count", 2)),
        },
    }


def draw_dual_wrinkle_plan_overlay(img: np.ndarray, H: Optional[np.ndarray],
                                   plan: Optional[DualWrinkleStretchPlan]) -> np.ndarray:
    if H is None or plan is None:
        return img
    color = (0, 255, 0) if plan.ok else (0, 0, 255)
    if plan.center_px is not None:
        c = tuple(int(round(v)) for v in plan.center_px)
        cv2.drawMarker(img, c, (0, 255, 255), cv2.MARKER_TILTED_CROSS, 28, 2)
    for arm_key, arm_color in (("arm2", (255, 80, 80)), ("arm1", (80, 80, 255))):
        p = plan.arm_points.get(arm_key)
        if not p:
            continue
        sp = board_to_pixel(H, float(p["source_board"][0]), float(p["source_board"][1]))
        tp = board_to_pixel(H, float(p["target_board"][0]), float(p["target_board"][1]))
        if sp and tp:
            s = (int(round(sp[0])), int(round(sp[1])))
            t = (int(round(tp[0])), int(round(tp[1])))
            raw_board = p.get("detected_source_board")
            if raw_board is not None:
                raw_px = board_to_pixel(H, float(raw_board[0]), float(raw_board[1]))
                if raw_px is not None:
                    rr = (int(round(raw_px[0])), int(round(raw_px[1])))
                    cv2.drawMarker(img, rr, (180, 180, 180), cv2.MARKER_CROSS, 18, 2)
                    cv2.line(img, rr, s, (180, 180, 180), 1)
            role = str(p.get("role", ""))
            cv2.circle(img, s, 9, arm_color, -1)
            if role in ("active", "dual_active", "pre_smooth"):
                cv2.circle(img, t, 9, (0, 255, 0), 2)
                cv2.arrowedLine(img, s, t, arm_color, 3, tipLength=0.2)
            else:
                cv2.circle(img, s, 15, (255, 255, 255), 2)
            cv2.putText(img, f"{arm_key.upper()} {role.upper()}", (s[0] + 6, s[1] - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, arm_color, 2)
    geom = plan.metrics.get("d21_geometry", {}) if isinstance(plan.metrics, dict) else {}
    if isinstance(geom, dict) and geom.get("circle_center_board") is not None:
        try:
            cc = geom.get("circle_center_board")
            cp = board_to_pixel(H, float(cc[0]), float(cc[1]))
            mid = geom.get("arc_mid_board")
            mp = None if mid is None else board_to_pixel(H, float(mid[0]), float(mid[1]))
            if cp is not None:
                cci = (int(round(cp[0])), int(round(cp[1])))
                cv2.drawMarker(img, cci, (255, 0, 255), cv2.MARKER_CROSS, 24, 2)
                cv2.putText(img, "ARC CENTER", (cci[0] + 7, cci[1] - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 0, 255), 1)
            if cp is not None and mp is not None:
                mpi = (int(round(mp[0])), int(round(mp[1])))
                cv2.arrowedLine(img, mpi, cci, (255, 0, 255), 2, tipLength=0.15)
        except Exception:
            pass
    txt = f"D21 {plan.action} ok={plan.ok} {plan.reason[:68]}"
    cv2.putText(img, txt, (25, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.56, (0, 0, 0), 4)
    cv2.putText(img, txt, (25, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.56, color, 2)
    return img


# -----------------------------------------------------------------------------
# Planning: stages 1~4
# -----------------------------------------------------------------------------

def find_nearest_mask_board_point(mask: BottomMaskBoard, H: np.ndarray, desired_board: np.ndarray,
                                  cfg: BottomSafetyConfig, preferred_arm: Optional[str] = None,
                                  max_radius_mm: float = 90.0) -> Optional[np.ndarray]:
    ys, xs = np.where(mask.mask_u8 > 0)
    if len(xs) < 10:
        return None
    pts_px = np.column_stack([xs, ys]).astype(np.float32)
    # Subsample for speed.
    if len(pts_px) > 20000:
        pts_px = pts_px[:: max(1, int(math.ceil(len(pts_px) / 20000.0)))]
    pts_board = cv2.perspectiveTransform(pts_px.reshape(-1, 1, 2), H).reshape(-1, 2)
    valid = []
    for i, p in enumerate(pts_board):
        if not point_in_board(cfg, p, margin=cfg.board_margin_mm):
            continue
        arm = select_arm_for_board_x(cfg, float(p[0]))
        if preferred_arm is not None and arm != preferred_arm:
            continue
        if arm is None:
            continue
        d = float(np.linalg.norm(p - desired_board))
        if d <= float(max_radius_mm):
            valid.append((d, i))
    if not valid:
        return None
    _, idx = min(valid, key=lambda x: x[0])
    return np.asarray(pts_board[idx], dtype=np.float32)


def choose_safe_grip_point(obs: BottomObservation, H: np.ndarray, cfg: BottomSafetyConfig,
                           desired: np.ndarray, preferred_arm: Optional[str] = None) -> Tuple[Optional[str], Optional[np.ndarray], str]:
    if obs.mask is None:
        return None, None, "no mask for grip snapping"
    desired = clamp_board_point(cfg, desired)
    arm = select_arm_for_board_x(cfg, float(desired[0]))
    if preferred_arm is not None:
        arm = preferred_arm
    if arm is not None and point_in_board(cfg, desired):
        # Still snap to the nearest actual mask point, but close to desired.
        snapped = find_nearest_mask_board_point(obs.mask, H, desired, cfg, preferred_arm=arm, max_radius_mm=80.0)
        if snapped is not None:
            return arm, snapped, "snapped desired to actual mask point"
        return arm, desired, "used desired point; no nearby mask snap found"

    # Desired is in the center dead-zone. Try both private workspaces and choose closer snapped point.
    choices = []
    for candidate_arm in ("arm2", "arm1"):
        snapped = find_nearest_mask_board_point(obs.mask, H, desired, cfg, preferred_arm=candidate_arm, max_radius_mm=130.0)
        if snapped is not None:
            choices.append((float(np.linalg.norm(snapped - desired)), candidate_arm, snapped))
    if not choices:
        return None, None, "desired grip in dead-zone and no private-workspace mask point found"
    _, arm, snapped = min(choices, key=lambda x: x[0])
    return arm, snapped, f"desired in dead-zone; selected nearest {arm} mask point"


def limit_delta(delta: np.ndarray, max_norm: float) -> np.ndarray:
    d = float(np.linalg.norm(delta))
    if d <= max_norm or d < 1e-6:
        return delta.astype(np.float32)
    return (delta / d * float(max_norm)).astype(np.float32)


def resolve_sequence(config: Dict[str, Any], cfg: BottomSafetyConfig, arm_key: str,
                     grip_board: np.ndarray, target_board: np.ndarray,
                     action_name: str, contact_z: Optional[float] = None) -> Tuple[List[MotionPoint], List[str]]:
    warnings: List[str] = []
    contact = arm_contact_z(cfg, arm_key) if contact_z is None else float(contact_z)
    lift = arm_lift_z(cfg, arm_key)
    if contact < cfg.min_safe_contact_z:
        warnings.append(f"contact_z {contact:.1f} < min_safe_contact_z {cfg.min_safe_contact_z:.1f}; blocked by caller")
    pts = [
        ("hover_grip", grip_board, cfg.hover_z),
        ("approach_grip", grip_board, cfg.approach_z),
        ("contact_grip", grip_board, contact),
        ("lift_grip", grip_board, lift),
        ("lift_target", target_board, lift),
        ("place_target", target_board, cfg.place_z),
        ("clear_target", target_board, cfg.clear_z),
    ]
    sequence = []
    for label, b, z in pts:
        if not point_in_board(cfg, b, margin=0.0):
            warnings.append(f"{label} board target outside board: ({b[0]:.1f},{b[1]:.1f})")
            continue
        if select_arm_for_board_x(cfg, float(b[0])) != arm_key:
            warnings.append(f"{label} violates workspace: {arm_key} cannot use x={b[0]:.1f}")
            continue
        try:
            rx, ry = board_to_arm_xy(config, arm_key, float(b[0]), float(b[1]))
        except Exception as e:
            warnings.append(f"{label} affine resolve failed: {repr(e)}")
            continue
        sequence.append(MotionPoint(arm_key=arm_key, board_xy=(float(b[0]), float(b[1])), roarm_xyz=(rx, ry, float(z)), label=f"{action_name}:{label}"))
    return sequence, warnings


def build_pick_place_plan(obs: BottomObservation, H: np.ndarray, config: Dict[str, Any], cfg: BottomSafetyConfig) -> MotionPlan:
    if not obs.valid or obs.pose is None or obs.mask is None or obs.fused_center_board is None:
        return MotionPlan("pick_place", "STOP", BottomStage.PICK_PLACE_TO_BOARD, False, f"invalid observation: {obs.reason}")

    # Pick near fused center, but snap to actual mask and private workspace.
    desired_grip = obs.fused_center_board.copy()
    arm_key, grip, grip_reason = choose_safe_grip_point(obs, H, cfg, desired_grip)
    if arm_key is None or grip is None:
        return MotionPlan("pick_place", "STOP", BottomStage.PICK_PLACE_TO_BOARD, False, grip_reason)

    # Combined coarse shift: put waist y near lower reference and crotch x near slit.
    delta = np.asarray([
        cfg.center_slit_x - float(obs.pose.crotch[0]),
        cfg.waist_ref_y - float(obs.pose.waist_center[1]),
    ], dtype=np.float32)
    delta = limit_delta(delta, cfg.max_single_xy_move_mm)
    target = clamp_board_point(cfg, grip + delta)

    # The target must remain in the same arm private workspace for this one-arm pick/place.
    if select_arm_for_board_x(cfg, float(target[0])) != arm_key:
        # Keep x within that arm and prioritize y placement. This avoids crossing the center dead-zone.
        target = target.copy()
        if arm_key == "arm2":
            target[0] = min(target[0], cfg.split_board_x - cfg.center_dead_half_width - cfg.private_workspace_margin_mm)
        else:
            target[0] = max(target[0], cfg.split_board_x + cfg.center_dead_half_width + cfg.private_workspace_margin_mm)
        target = clamp_board_point(cfg, target)

    seq, warnings = resolve_sequence(config, cfg, arm_key, grip, target, "PICK_PLACE")
    ok = len(seq) == 7 and not any("violates" in w or "failed" in w or "outside" in w for w in warnings)
    reason = f"{grip_reason}; delta=({delta[0]:.1f},{delta[1]:.1f})mm"
    return MotionPlan(
        name="pick_place",
        action="PICK_PLACE_TO_BOARD",
        stage=BottomStage.PICK_PLACE_TO_BOARD,
        ok=ok,
        reason=reason,
        arm_key=arm_key,
        grip_board_xy=(float(grip[0]), float(grip[1])),
        target_board_xy=(float(target[0]), float(target[1])),
        sequence=seq,
        safety_warnings=warnings,
        metrics={
            "waist_y": float(obs.pose.waist_center[1]),
            "waist_ref_y": float(cfg.waist_ref_y),
            "crotch_x": float(obs.pose.crotch[0]),
            "center_slit_x": float(cfg.center_slit_x),
        },
    )


def build_pose_keypoint_grasp_plan(obs: BottomObservation, H: np.ndarray, config: Dict[str, Any],
                                   cfg: BottomSafetyConfig, kpt_index: int) -> MotionPlan:
    """Plan a conservative one-arm grasp test for a selected bottom pose keypoint.

    The selected keypoint is a visual target, not always a safe contact point.
    If the keypoint is in the center dead-zone, the contact point is snapped to
    the nearest actual mask pixel in ARM1/ARM2 private workspace.
    """
    if obs.pose is None or obs.mask is None:
        return MotionPlan("pose_kpt_grasp", "STOP", BottomStage.PICK_PLACE_TO_BOARD, False,
                          f"need pose+mask first: {obs.reason}")
    if kpt_index < 0 or kpt_index >= len(BOTTOM_POSE_KPT_NAMES):
        return MotionPlan("pose_kpt_grasp", "STOP", BottomStage.PICK_PLACE_TO_BOARD, False,
                          f"invalid keypoint index: {kpt_index}")
    kpt_name = BOTTOM_POSE_KPT_NAMES[int(kpt_index)]
    raw = obs.pose.keypoints_board.get(kpt_name)
    if raw is None:
        return MotionPlan("pose_kpt_grasp", "STOP", BottomStage.PICK_PLACE_TO_BOARD, False,
                          f"selected keypoint is not visible: {kpt_index}:{kpt_name}",
                          metrics={"selected_kpt_index": int(kpt_index), "selected_kpt_name": kpt_name})

    desired = np.asarray(raw, dtype=np.float32)
    arm_key, grip, grip_reason = choose_safe_grip_point(obs, H, cfg, desired)
    if arm_key is None or grip is None:
        return MotionPlan("pose_kpt_grasp", "STOP", BottomStage.PICK_PLACE_TO_BOARD, False,
                          f"{kpt_index}:{kpt_name} cannot be gripped safely: {grip_reason}",
                          metrics={"selected_kpt_index": int(kpt_index), "selected_kpt_name": kpt_name})

    # Grip test: target is the same XY. Full mode closes, lifts slightly, lowers,
    # opens, and clears. Hover-only mode stays at safe Z only.
    target = grip.copy()
    seq, warnings = resolve_sequence(config, cfg, arm_key, grip, target, f"KPT{kpt_index}_{kpt_name.upper()}")
    ok = len(seq) == 7 and not any("violates" in w or "failed" in w or "outside" in w for w in warnings)
    return MotionPlan(
        name="pose_kpt_grasp",
        action=f"GRASP_KPT_{kpt_index}_{kpt_name}",
        stage=BottomStage.PICK_PLACE_TO_BOARD,
        ok=ok,
        reason=f"selected {kpt_index}:{kpt_name}; {grip_reason}",
        arm_key=arm_key,
        grip_board_xy=(float(grip[0]), float(grip[1])),
        target_board_xy=(float(target[0]), float(target[1])),
        sequence=seq,
        safety_warnings=warnings,
        metrics={
            "selected_kpt_index": int(kpt_index),
            "selected_kpt_name": kpt_name,
            "raw_kpt_board_xy": [float(desired[0]), float(desired[1])],
            "grip_board_xy": [float(grip[0]), float(grip[1])],
            "snap_distance_mm": float(np.linalg.norm(grip - desired)),
        },
    )


def build_waist_align_plan(obs: BottomObservation, H: np.ndarray, config: Dict[str, Any], cfg: BottomSafetyConfig) -> MotionPlan:
    if not obs.valid or obs.pose is None or obs.mask is None:
        return MotionPlan("align_waist", "STOP", BottomStage.ALIGN_WAIST_REF, False, f"invalid observation: {obs.reason}")
    err_y = float(obs.pose.waist_center[1] - cfg.waist_ref_y)
    if abs(err_y) <= cfg.waist_y_tolerance_mm:
        return MotionPlan("align_waist", "DONE", BottomStage.ALIGN_WAIST_REF, True, f"waist already aligned: error_y={err_y:.1f}mm", metrics={"error_y": err_y})

    desired_grip = obs.pose.waist_center.copy()
    # Avoid exact waist edge if it is in the dead-zone; snap to nearby cloth point.
    arm_key, grip, grip_reason = choose_safe_grip_point(obs, H, cfg, desired_grip)
    if arm_key is None or grip is None:
        return MotionPlan("align_waist", "STOP", BottomStage.ALIGN_WAIST_REF, False, grip_reason, metrics={"error_y": err_y})

    delta = np.asarray([0.0, -err_y], dtype=np.float32)
    delta = limit_delta(delta, min(cfg.max_single_xy_move_mm, 95.0))
    target = clamp_board_point(cfg, grip + delta)
    if select_arm_for_board_x(cfg, float(target[0])) != arm_key:
        return MotionPlan("align_waist", "STOP", BottomStage.ALIGN_WAIST_REF, False,
                          f"waist align target crosses workspace: grip x={grip[0]:.1f}, target x={target[0]:.1f}",
                          arm_key=arm_key, grip_board_xy=(float(grip[0]), float(grip[1])), metrics={"error_y": err_y})

    seq, warnings = resolve_sequence(config, cfg, arm_key, grip, target, "WAIST_ALIGN", contact_z=cfg.grip_z)
    ok = len(seq) == 7 and not any("violates" in w or "failed" in w or "outside" in w for w in warnings)
    return MotionPlan(
        name="align_waist",
        action="DRAG_WAIST_TO_REF",
        stage=BottomStage.ALIGN_WAIST_REF,
        ok=ok,
        reason=f"error_y={err_y:.1f}mm; {grip_reason}",
        arm_key=arm_key,
        grip_board_xy=(float(grip[0]), float(grip[1])),
        target_board_xy=(float(target[0]), float(target[1])),
        sequence=seq,
        safety_warnings=warnings,
        metrics={"error_y": err_y, "delta_y": float(delta[1])},
    )


def build_crotch_align_plan(obs: BottomObservation, H: np.ndarray, config: Dict[str, Any], cfg: BottomSafetyConfig) -> MotionPlan:
    if not obs.valid or obs.pose is None or obs.mask is None:
        return MotionPlan("align_crotch", "STOP", BottomStage.ALIGN_CROTCH_SLIT, False, f"invalid observation: {obs.reason}")
    err_x = float(obs.pose.crotch[0] - cfg.center_slit_x)
    if abs(err_x) <= cfg.crotch_x_tolerance_mm:
        return MotionPlan("align_crotch", "DONE", BottomStage.ALIGN_CROTCH_SLIT, True, f"crotch already aligned: error_x={err_x:.1f}mm", metrics={"error_x": err_x})

    # Do NOT grip the crotch point directly. Use a safer point between pose center and lower center.
    desired_grip = 0.55 * obs.pose.pose_center + 0.45 * obs.pose.lower_center
    preferred = "arm2" if desired_grip[0] < cfg.center_slit_x else "arm1"
    arm_key, grip, grip_reason = choose_safe_grip_point(obs, H, cfg, desired_grip, preferred_arm=preferred)
    if arm_key is None or grip is None:
        return MotionPlan("align_crotch", "STOP", BottomStage.ALIGN_CROTCH_SLIT, False, grip_reason, metrics={"error_x": err_x})

    delta = np.asarray([-err_x, 0.0], dtype=np.float32)
    delta = limit_delta(delta, min(cfg.max_single_xy_move_mm, 85.0))
    target = clamp_board_point(cfg, grip + delta)
    if select_arm_for_board_x(cfg, float(target[0])) != arm_key:
        # Keep inside workspace; if this happens, one-arm lateral drag is unsafe.
        return MotionPlan("align_crotch", "STOP", BottomStage.ALIGN_CROTCH_SLIT, False,
                          f"crotch align target crosses center/dead-zone: {arm_key} grip x={grip[0]:.1f}, target x={target[0]:.1f}",
                          arm_key=arm_key, grip_board_xy=(float(grip[0]), float(grip[1])), metrics={"error_x": err_x})

    seq, warnings = resolve_sequence(config, cfg, arm_key, grip, target, "CROTCH_ALIGN", contact_z=cfg.grip_z)
    ok = len(seq) == 7 and not any("violates" in w or "failed" in w or "outside" in w for w in warnings)
    return MotionPlan(
        name="align_crotch",
        action="DRAG_CROTCH_TO_SLIT",
        stage=BottomStage.ALIGN_CROTCH_SLIT,
        ok=ok,
        reason=f"error_x={err_x:.1f}mm; {grip_reason}",
        arm_key=arm_key,
        grip_board_xy=(float(grip[0]), float(grip[1])),
        target_board_xy=(float(target[0]), float(target[1])),
        sequence=seq,
        safety_warnings=warnings,
        metrics={"error_x": err_x, "delta_x": float(delta[0])},
    )


# -----------------------------------------------------------------------------
# Execution and overlay
# -----------------------------------------------------------------------------

def print_plan(plan: MotionPlan) -> None:
    print("\n========== BOTTOM STAGE PLAN ==========")
    print(f"stage={plan.stage.name} action={plan.action} ok={plan.ok}")
    print(f"reason={plan.reason}")
    print(f"arm={plan.arm_key} grip={plan.grip_board_xy} target={plan.target_board_xy}")
    if plan.metrics:
        print(f"metrics={json.dumps(plan.metrics, ensure_ascii=False, default=str)}")
    if plan.safety_warnings:
        print("warnings:")
        for w in plan.safety_warnings:
            print("  -", w)
    for p in plan.sequence:
        print(f"  {p.label:24s} board=({p.board_xy[0]:7.1f},{p.board_xy[1]:7.1f}) roarm={p.roarm_xyz}")
    print("=====================================\n")


def confirm_if_needed(cfg: BottomSafetyConfig, label: str) -> bool:
    if not cfg.enter_confirm:
        return True
    TERMINAL_CONFIRM_ACTIVE.set()
    try:
        ans = input(f"[CONFIRM] {label} 실행하려면 Enter, 취소하려면 n 입력: ").strip().lower()
        return ans == ""
    finally:
        TERMINAL_CONFIRM_ACTIVE.clear()


def standby_roarm_pose(cfg: BottomSafetyConfig, arm_key: str) -> Tuple[float, float, float, float]:
    if arm_key == "arm1":
        return (
            float(cfg.arm1_standby_roarm_x),
            float(cfg.arm1_standby_roarm_y),
            float(cfg.arm1_standby_roarm_z),
            float(cfg.arm1_standby_roarm_t),
        )
    return (
        float(cfg.arm2_standby_roarm_x),
        float(cfg.arm2_standby_roarm_y),
        float(cfg.arm2_standby_roarm_z),
        float(cfg.arm2_standby_roarm_t),
    )


def move_arm_to_standby(arm_key: str, arms: Dict[str, RoArmSerial], cfg: BottomSafetyConfig,
                        move_command: int = 104, reason: str = "standby") -> bool:
    arm = arms.get(arm_key)
    if arm is None:
        print(f"[STANDBY] {arm_key.upper()} skipped: serial arm not connected")
        return False
    x, y, z, t = standby_roarm_pose(cfg, arm_key)
    print(f"[STANDBY] {arm_key.upper()} {reason}: roarm=({x:.1f},{y:.1f},{z:.1f}) "
          f"t={t:.3f} gripper={cfg.standby_gripper_angle:.3f}")
    arm.set_gripper(float(cfg.standby_gripper_angle), delay=0.18)
    arm.move_goal(x, y, z, t, cfg.speed, move_command=move_command)
    return True


def move_arms_to_standby(arms: Dict[str, RoArmSerial], cfg: BottomSafetyConfig,
                         move_command: int = 104, reason: str = "standby") -> bool:
    if not cfg.imaging_standby:
        print("[STANDBY] skipped: imaging_standby=False")
        return False
    if cfg.dry_run or not cfg.send:
        print(f"[STANDBY] dry-run/send guard: send={cfg.send} dry_run={cfg.dry_run}. "
              "No standby motion command sent.")
        return False
    any_ok = False
    # Same order as the older script: ARM2 left side, then ARM1 right side.
    for arm_key in ("arm2", "arm1"):
        try:
            any_ok = move_arm_to_standby(arm_key, arms, cfg, move_command=move_command, reason=reason) or any_ok
            time.sleep(0.15)
        except Exception as e:
            print(f"[STANDBY] {arm_key.upper()} failed: {repr(e)}")
    return any_ok


def execute_motion_plan(plan: MotionPlan, arms: Dict[str, RoArmSerial], cfg: BottomSafetyConfig, move_command: int = 104) -> bool:
    print_plan(plan)
    if not plan.ok:
        print(f"[EXEC] blocked: {plan.reason}")
        return False
    if cfg.dry_run:
        print("[EXEC] dry-run=True 이므로 실제 명령을 보내지 않습니다.")
        return True
    if not cfg.send:
        print("[EXEC] send=False 이므로 실제 명령을 보내지 않습니다.")
        return True
    if plan.arm_key not in arms:
        print(f"[EXEC] missing serial arm: {plan.arm_key}")
        return False
    if not confirm_if_needed(cfg, f"{plan.action} / {plan.arm_key}"):
        print("[EXEC] cancelled by user")
        return False

    arm = arms[plan.arm_key]
    try:
        arm.set_gripper(cfg.grip_open, delay=0.25)
        time.sleep(max(0.0, float(cfg.grip_contact_wait)))
        if cfg.hover_only:
            for p in plan.sequence:
                if p.roarm_xyz is None:
                    continue
                if "hover" in p.label or "clear" in p.label or "lift_target" in p.label:
                    x, y, z = p.roarm_xyz
                    safe_z = max(float(z), cfg.clear_z if "clear" in p.label else cfg.hover_z)
                    arm.move_goal(x, y, safe_z, cfg.tool_t_rad, cfg.speed, move_command=move_command)
                    time.sleep(max(0.0, float(cfg.grip_down_wait) * 0.5))
            print("[EXEC] hover-only 완료. low-Z, gripper close, drag/place는 수행하지 않았습니다.")
            if cfg.standby_after_motion:
                move_arms_to_standby(arms, cfg, move_command=move_command, reason="after hover-only motion")
            return True

        # Full one-arm sequence, intentionally sequential and simple.
        by_label = {p.label.split(":")[-1]: p for p in plan.sequence if p.roarm_xyz is not None}
        for key in ("hover_grip", "approach_grip", "contact_grip"):
            x, y, z = by_label[key].roarm_xyz
            arm.move_goal(x, y, z, cfg.tool_t_rad, cfg.speed, move_command=move_command)
            time.sleep(max(0.0, float(cfg.grip_down_wait)))

        # Match the proven T-shirt pickup pattern: re-assert OPEN at the actual
        # contact point, wait for the gripper/fabric to settle, then close.
        print(f"[EXEC] OPEN at contact, wait {cfg.grip_open_wait:.2f}s before close")
        arm.set_gripper(cfg.grip_open, delay=0.25)
        time.sleep(max(0.0, float(cfg.grip_open_wait)))
        time.sleep(max(0.0, float(cfg.grip_contact_wait)))

        close_repeat = max(1, int(cfg.grip_close_repeat))
        for i in range(close_repeat):
            print(f"[EXEC] LIMITED CLOSE {i + 1}/{close_repeat}: angle={cfg.grip_close:.3f}")
            arm.set_gripper(cfg.grip_close, delay=0.30)
            time.sleep(max(0.0, float(cfg.grip_close_wait)))
        if abs(float(cfg.grip_hold) - float(cfg.grip_close)) > 1e-6:
            print(f"[EXEC] HOLD before lift: angle={cfg.grip_hold:.3f}")
            arm.set_gripper(cfg.grip_hold, delay=0.30)
        time.sleep(max(0.0, float(cfg.grip_close_settle)))

        for key in ("lift_grip", "lift_target", "place_target"):
            x, y, z = by_label[key].roarm_xyz
            arm.move_goal(x, y, z, cfg.tool_t_rad, cfg.speed if key != "lift_target" else cfg.drag_speed, move_command=move_command)
            time.sleep(max(0.0, float(cfg.place_settle_wait)))
        arm.set_gripper(cfg.grip_open, delay=0.45)
        time.sleep(max(0.0, float(cfg.place_open_wait)))
        x, y, z = by_label["clear_target"].roarm_xyz
        arm.move_goal(x, y, z, cfg.tool_t_rad, cfg.speed, move_command=move_command)
        if cfg.standby_after_motion:
            move_arms_to_standby(arms, cfg, move_command=move_command, reason="after low-Z motion")
        return True
    except Exception as e:
        print(f"[EXEC] error: {repr(e)}")
        return False


def draw_bottom_overlay(frame: np.ndarray, H: Optional[np.ndarray], obs: Optional[BottomObservation], cfg: BottomSafetyConfig,
                        plan: Optional[MotionPlan] = None, wrinkle_plan: Optional[WrinkleHeatmapPlan] = None,
                        args=None, motion_busy: bool = False, motion_name: str = "") -> np.ndarray:
    img = frame.copy()
    if H is not None:
        # Draw key board reference lines.
        for bx, by, label, color in [
            (cfg.center_slit_x, cfg.board_y_min + 20, "SLIT", (255, 0, 255)),
            (cfg.center_slit_x, cfg.board_y_max - 20, "", (255, 0, 255)),
        ]:
            pass
        p1 = board_to_pixel(H, cfg.center_slit_x, cfg.board_y_min + cfg.board_margin_mm)
        p2 = board_to_pixel(H, cfg.center_slit_x, cfg.board_y_max - cfg.board_margin_mm)
        if p1 and p2:
            cv2.line(img, (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])), (255, 0, 255), 2)
            cv2.putText(img, "CENTER SLIT", (int(p2[0]) + 6, int(p2[1]) + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 0, 255), 2)
        q1 = board_to_pixel(H, cfg.board_x_min + cfg.board_margin_mm, cfg.waist_ref_y)
        q2 = board_to_pixel(H, cfg.board_x_max - cfg.board_margin_mm, cfg.waist_ref_y)
        if q1 and q2:
            cv2.line(img, (int(q1[0]), int(q1[1])), (int(q2[0]), int(q2[1])), (0, 180, 255), 2)
            cv2.putText(img, "WAIST REF", (int(q1[0]) + 6, int(q1[1]) - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 180, 255), 2)

    if args is not None:
        img = draw_wrinkle_heatmap_overlay(img, wrinkle_plan, args, stale=bool(motion_busy))

    if obs is not None and obs.mask is not None:
        overlay = img.copy()
        cv2.drawContours(overlay, [obs.mask.contour], -1, (60, 170, 255), -1)
        img = cv2.addWeighted(overlay, 0.25, img, 0.75, 0)
        cv2.drawContours(img, [obs.mask.contour], -1, (0, 220, 255), 2)
        cx, cy = obs.mask.center_px
        cv2.circle(img, (int(cx), int(cy)), 6, (0, 0, 255), -1)
        cv2.putText(img, "MASK-C", (int(cx) + 8, int(cy) - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    if obs is not None and obs.pose is not None:
        colors = {
            "waist_img_left": (0, 255, 255),
            "waist_center": (0, 180, 255),
            "waist_img_right": (0, 255, 255),
            "crotch": (255, 0, 255),
            "img_left_hem_outer": (255, 180, 0),
            "img_left_hem_inner": (255, 180, 0),
            "img_right_hem_inner": (255, 180, 0),
            "img_right_hem_outer": (255, 180, 0),
        }
        for name, p in obs.pose.keypoints_px.items():
            x, y = int(round(p[0])), int(round(p[1]))
            cv2.circle(img, (x, y), 5, (0, 0, 0), -1)
            cv2.circle(img, (x, y), 3, colors.get(name, (255, 255, 255)), -1)
            cv2.putText(img, name.replace("img_", "")[:10], (x + 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 0, 0), 2)
            cv2.putText(img, name.replace("img_", "")[:10], (x + 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.38, colors.get(name, (255, 255, 255)), 1)
        # Draw pants axis in board -> pixel.
        if H is not None:
            a = board_to_pixel(H, float(obs.pose.waist_center[0]), float(obs.pose.waist_center[1]))
            b = board_to_pixel(H, float(obs.pose.lower_center[0]), float(obs.pose.lower_center[1]))
            c = board_to_pixel(H, float(obs.pose.crotch[0]), float(obs.pose.crotch[1]))
            if a and b:
                cv2.arrowedLine(img, (int(a[0]), int(a[1])), (int(b[0]), int(b[1])), (0, 255, 0), 3, tipLength=0.2)
            if c:
                cv2.circle(img, (int(c[0]), int(c[1])), 7, (255, 0, 255), 2)

        tta_msg = (f"TTA angle={obs.pose.tta_angle_deg:+.0f} flip={obs.pose.tta_flip_mode} "
                   f"score={obs.pose.tta_score:.1f} state={obs.pose.tta_state} "
                   f"tested={obs.pose.tta_tested_count}")
        y_tta = max(24, img.shape[0] - 20)
        cv2.putText(img, tta_msg[:120], (25, y_tta), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 4)
        cv2.putText(img, tta_msg[:120], (25, y_tta), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (0, 255, 0) if obs.pose.tta_state == "OK" else (0, 180, 255), 2)

    if plan is not None and H is not None:
        if plan.grip_board_xy:
            gp = board_to_pixel(H, plan.grip_board_xy[0], plan.grip_board_xy[1])
            if gp:
                cv2.drawMarker(img, (int(gp[0]), int(gp[1])), (255, 255, 255), cv2.MARKER_CROSS, 24, 3)
                cv2.putText(img, f"GRIP {plan.arm_key}", (int(gp[0]) + 8, int(gp[1]) + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
        if plan.target_board_xy:
            tp = board_to_pixel(H, plan.target_board_xy[0], plan.target_board_xy[1])
            if tp:
                cv2.drawMarker(img, (int(tp[0]), int(tp[1])), (0, 255, 0), cv2.MARKER_TILTED_CROSS, 24, 3)
                cv2.putText(img, "TARGET", (int(tp[0]) + 8, int(tp[1]) + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
            if plan.grip_board_xy and tp:
                gp = board_to_pixel(H, plan.grip_board_xy[0], plan.grip_board_xy[1])
                if gp:
                    cv2.arrowedLine(img, (int(gp[0]), int(gp[1])), (int(tp[0]), int(tp[1])), (0, 255, 0), 3, tipLength=0.2)
        cv2.putText(img, f"PLAN {plan.action} ok={plan.ok} {plan.reason[:70]}", (25, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 0, 0), 4)
        cv2.putText(img, f"PLAN {plan.action} ok={plan.ok} {plan.reason[:70]}", (25, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 255, 0) if plan.ok else (0, 0, 255), 2)
    elif obs is not None:
        cv2.putText(img, f"OBS valid={obs.valid} {obs.reason[:90]}", (25, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 0, 0), 4)
        cv2.putText(img, f"OBS valid={obs.valid} {obs.reason[:90]}", (25, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 255, 0) if obs.valid else (0, 0, 255), 2)
    if motion_busy:
        msg = f"ROBOT MOTION RUNNING - camera stays live [{motion_name}]"
        cv2.putText(img, msg[:105], (25, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 0, 0), 4)
        cv2.putText(img, msg[:105], (25, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 165, 255), 2)
    return img


# -----------------------------------------------------------------------------
# Main loop / demo
# -----------------------------------------------------------------------------

def make_safety_config_from_args(args, config: Optional[Dict[str, Any]]) -> BottomSafetyConfig:
    bx0, bx1, by0, by1 = board_bounds_from_config(config)
    split_x = 247.0
    try:
        split_x = float(config.get("dual_roarm", {}).get("split_board_x", split_x)) if config else split_x
    except Exception:
        pass
    return BottomSafetyConfig(
        board_x_min=bx0,
        board_x_max=bx1,
        board_y_min=by0,
        board_y_max=by1,
        split_board_x=split_x,
        center_dead_half_width=float(args.center_dead_half_width),
        hover_z=float(args.hover_z),
        approach_z=float(args.approach_z),
        grip_z=float(args.grip_z),
        arm1_grip_z=None if args.arm1_grip_z is None else float(args.arm1_grip_z),
        arm2_grip_z=None if args.arm2_grip_z is None else float(args.arm2_grip_z),
        lift_z=float(args.lift_z),
        arm1_lift_z=None if args.arm1_lift_z is None else float(args.arm1_lift_z),
        arm2_lift_z=None if args.arm2_lift_z is None else float(args.arm2_lift_z),
        place_z=float(args.place_z),
        clear_z=float(args.clear_z),
        min_safe_contact_z=float(args.min_safe_contact_z),
        speed=float(args.speed),
        drag_speed=float(args.drag_speed),
        grip_open=float(args.grip_open),
        grip_close=float(args.grip_close),
        grip_hold=float(args.grip_hold),
        grip_down_wait=float(args.grip_down_wait),
        grip_open_wait=float(args.grip_open_wait),
        grip_contact_wait=float(args.grip_contact_wait),
        grip_close_repeat=int(args.grip_close_repeat),
        grip_close_wait=float(args.grip_close_wait),
        grip_close_settle=float(args.grip_close_settle),
        place_settle_wait=float(args.place_settle_wait),
        place_open_wait=float(args.place_open_wait),
        imaging_standby=bool(args.imaging_standby),
        standby_after_motion=bool(args.standby_after_motion),
        standby_gripper_angle=float(args.standby_gripper_angle),
        arm1_standby_roarm_x=float(args.arm1_standby_roarm_x),
        arm1_standby_roarm_y=float(args.arm1_standby_roarm_y),
        arm1_standby_roarm_z=float(args.arm1_standby_roarm_z),
        arm1_standby_roarm_t=float(args.arm1_standby_roarm_t),
        arm2_standby_roarm_x=float(args.arm2_standby_roarm_x),
        arm2_standby_roarm_y=float(args.arm2_standby_roarm_y),
        arm2_standby_roarm_z=float(args.arm2_standby_roarm_z),
        arm2_standby_roarm_t=float(args.arm2_standby_roarm_t),
        waist_ref_y=float(args.waist_ref_y),
        center_slit_x=float(args.center_slit_x),
        waist_y_tolerance_mm=float(args.waist_y_tolerance_mm),
        crotch_x_tolerance_mm=float(args.crotch_x_tolerance_mm),
        pose_kpt_conf=float(args.pose_kpt_conf),
        send=bool(args.send),
        dry_run=bool(args.dry_run),
        hover_only=bool(args.hover_only),
        enter_confirm=bool(args.enter_confirm),
    )


def _resolve_model_path(requested: str, alternatives: Sequence[str], label: str) -> str:
    requested = os.path.expanduser(str(requested or ""))
    if requested and os.path.exists(requested):
        return requested
    for candidate in alternatives:
        candidate = os.path.expanduser(str(candidate))
        if candidate and os.path.exists(candidate):
            print(f"[MODEL-PATH] {label}: requested path missing; using {candidate}")
            return candidate
    return requested


def load_models(args):
    if YOLO is None:
        raise RuntimeError("ultralytics is not available. Install/import YOLO first.")
    args.seg_model = _resolve_model_path(
        args.seg_model,
        [
            "/workspace/project_train/aruco_test/dual/models/kfashion_yolo26s_seg3_e100_best.engine",
            "/home/deca/project_train/aruco_test/dual/models/kfashion_yolo26s_seg3_e100_best.engine",
        ],
        "segmentation",
    )
    args.pose_model = _resolve_model_path(
        args.pose_model,
        [
            "/home/deca/project_train/yolo26-pose-small-down/best.engine",
            "/workspace/project_train/yolo26-pose-small-down/best.engine",
            "/workspace/project_train/aruco_test/dual/best.engine",
        ],
        "bottom-pose",
    )
    print(f"[MODEL] loading seg: {args.seg_model}")
    seg = YOLO(args.seg_model, task="segment")
    print(f"[MODEL] loading bottom pose: {args.pose_model}")
    pose = YOLO(args.pose_model, task="pose")
    return seg, pose


def _uvc_device(args) -> str:
    return str(getattr(args, "cam_device", "") or f"/dev/video{int(args.camera)}")


def _uvc_set(args, name: str, value: int) -> bool:
    tool = shutil.which("v4l2-ctl")
    if tool is None:
        return False
    proc = subprocess.run(
        [tool, "-d", _uvc_device(args), "-c", f"{name}={int(value)}"],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False,
    )
    if proc.returncode != 0:
        msg = (proc.stderr or proc.stdout or "unsupported").strip().replace("\n", " ")
        print(f"[CAM-WARN] {name}={value}: {msg}")
        return False
    return True


def _uvc_get(args, name: str) -> Optional[int]:
    tool = shutil.which("v4l2-ctl")
    if tool is None:
        return None
    proc = subprocess.run(
        [tool, "-d", _uvc_device(args), "-C", name],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False,
    )
    if proc.returncode != 0:
        return None
    m = re.search(r":\s*(-?\d+)", proc.stdout or "")
    return None if m is None else int(m.group(1))


def configure_usb_camera(args) -> None:
    """Initialize UVC controls. D16 can auto-tune before every robot action."""
    if bool(getattr(args, "cam_no_manual", False)):
        print("[CAM] UVC control disabled")
        return
    if shutil.which("v4l2-ctl") is None:
        print("[CAM-WARN] v4l2-ctl not found; install v4l-utils for exposure control")
        return

    if bool(getattr(args, "cam_auto_adjust", True)):
        print("[CAM-AUTO] startup auto-adjust enabled")
        return

    controls = [
        ("exposure_auto", 1),
        ("exposure_absolute", int(getattr(args, "cam_exposure_abs", 35))),
        ("gain", int(getattr(args, "cam_gain", 5))),
    ]
    for name, value in controls:
        if _uvc_set(args, name, value):
            print(f"[CAM] {name}={value}: OK")


def auto_adjust_usb_camera(args, reason: str = "motion") -> None:
    """Temporarily enable hardware auto exposure, then lock the settled values."""
    if bool(getattr(args, "cam_no_manual", False)):
        return
    if not bool(getattr(args, "cam_auto_adjust", True)):
        return
    if shutil.which("v4l2-ctl") is None:
        print("[CAM-AUTO-WARN] v4l2-ctl unavailable; auto exposure skipped")
        return

    settle = max(0.2, float(getattr(args, "cam_auto_settle_s", 1.20)))
    print(f"[CAM-AUTO] {reason}: enable auto exposure for {settle:.2f}s")
    if not _uvc_set(args, "exposure_auto", 3):
        print("[CAM-AUTO-WARN] exposure_auto=3 unsupported; using configured manual values")
        _uvc_set(args, "exposure_auto", 1)
        _uvc_set(args, "exposure_absolute", int(getattr(args, "cam_exposure_abs", 35)))
        _uvc_set(args, "gain", int(getattr(args, "cam_gain", 5)))
        return

    time.sleep(settle)
    exposure = _uvc_get(args, "exposure_absolute")
    gain = _uvc_get(args, "gain")

    if exposure is None:
        exposure = int(getattr(args, "cam_exposure_abs", 35))
    if gain is None:
        gain = int(getattr(args, "cam_gain", 5))

    exposure = int(np.clip(
        exposure,
        int(getattr(args, "cam_auto_exposure_min", 10)),
        int(getattr(args, "cam_auto_exposure_max", 100)),
    ))
    gain = int(np.clip(
        gain,
        int(getattr(args, "cam_auto_gain_min", 0)),
        int(getattr(args, "cam_auto_gain_max", 20)),
    ))

    _uvc_set(args, "exposure_auto", 1)
    ok_e = _uvc_set(args, "exposure_absolute", exposure)
    ok_g = _uvc_set(args, "gain", gain)
    if ok_e:
        args.cam_exposure_abs = exposure
    if ok_g:
        args.cam_gain = gain
    print(
        f"[CAM-AUTO] {reason}: locked exposure={exposure} gain={gain} "
        f"(ok_exposure={ok_e}, ok_gain={ok_g})"
    )


def open_camera(args):
    backend = str(args.backend).lower()
    api = cv2.CAP_ANY
    if backend == "v4l2" and hasattr(cv2, "CAP_V4L2"):
        api = cv2.CAP_V4L2
    elif backend == "dshow" and hasattr(cv2, "CAP_DSHOW"):
        api = cv2.CAP_DSHOW
    cap = cv2.VideoCapture(int(args.camera), api)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(args.width))
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(args.height))
    try:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    except Exception:
        pass
    if not cap.isOpened():
        raise RuntimeError(f"camera open failed: index={args.camera}")
    return cap


def read_terminal_key() -> int:
    """Read one line-mode command from bash without blocking the camera loop.

    OpenCV's cv2.waitKey only receives keys while the preview window has focus.
    In Docker/X11 setups that is easy to miss, so this lets the user type
    L<Enter>, I<Enter>, etc. in the bash terminal as a fallback.
    """
    try:
        if not sys.stdin or not sys.stdin.isatty():
            return 255
        readable, _, _ = select.select([sys.stdin], [], [], 0.0)
        if not readable:
            return 255
        line = sys.stdin.readline()
    except Exception:
        return 255
    if not line:
        return 255
    token = line.strip().lower()
    if not token:
        return 255
    aliases = {
        "l": ord("l"),
        "i": ord("i"),
        "p": ord("p"),
        "g": ord("g"),
        "s": ord("s"),
        "w": ord("w"),
        "c": ord("c"),
        "f": ord("f"),
        "h": ord("h"),
        "x": ord("x"),
        "q": ord("q"),
        "esc": 27,
        "exit": ord("q"),
        "quit": ord("q"),
    }
    for digit in "01234567":
        aliases[digit] = ord(digit)
    key = aliases.get(token, aliases.get(token[0], 255))
    if key != 255:
        print(f"[TERMINAL-KEY] {token} -> {chr(key) if key != 27 else 'ESC'}")
    return key


def setup_arms(args) -> Dict[str, RoArmSerial]:
    arms: Dict[str, RoArmSerial] = {}
    if not args.send:
        return arms
    arms["arm1"] = RoArmSerial(args.arm1_port, baudrate=115200, label="ARM1")
    arms["arm2"] = RoArmSerial(args.arm2_port, baudrate=115200, label="ARM2")
    # Open grippers to a safe angle on startup.
    try:
        arms["arm1"].set_gripper(float(args.grip_open), delay=0.15)
        arms["arm2"].set_gripper(float(args.grip_open), delay=0.15)
    except Exception as e:
        print(f"[ARM] startup gripper open failed: {repr(e)}")
    return arms


def run_once(frame, H, seg_model, pose_model, args, config, cfg, which: str) -> Tuple[Optional[BottomObservation], Optional[MotionPlan], np.ndarray]:
    obs = infer_bottom_observation(seg_model, pose_model, frame, H, args, cfg)
    plan = None
    if which == "pick":
        plan = build_pick_place_plan(obs, H, config, cfg)
    elif which == "waist":
        plan = build_waist_align_plan(obs, H, config, cfg)
    elif which == "crotch":
        plan = build_crotch_align_plan(obs, H, config, cfg)
    canvas = draw_bottom_overlay(frame, H, obs, cfg, plan)
    return obs, plan, canvas


def run_full_stage_1_to_4(cap, H, seg_model, pose_model, args, config, cfg, arms) -> bool:
    stages = ["pick", "waist", "crotch"]
    for st in stages:
        ok_frame, frame = cap.read()
        if not ok_frame:
            print("[FULL] camera read failed")
            return False
        obs, plan, _ = run_once(frame, H, seg_model, pose_model, args, config, cfg, st)
        if obs is None or plan is None:
            print(f"[FULL] {st}: no plan")
            return False
        print_plan(plan)
        if plan.action == "DONE":
            continue
        if not plan.ok:
            print(f"[FULL] {st} blocked: {plan.reason}")
            return False
        if not execute_motion_plan(plan, arms, cfg, move_command=args.move_command):
            return False
        time.sleep(float(args.after_action_settle_s))
    print("[FULL] Stage 1~4 sequence finished.")
    return True


def demo() -> None:
    """Synthetic planning demo without camera/model/serial."""
    H = np.asarray([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    config = {
        "aruco": {"marker_board_mm": DEFAULT_MARKER_BOARD_MM},
        "dual_roarm": {
            "split_board_x": 247.0,
            "arm1": {"board_to_roarm_affine_2x3": [[1, 0, -247], [0, 1, 0]]},
            "arm2": {"board_to_roarm_affine_2x3": [[1, 0, -247], [0, 1, 0]]},
        },
    }
    cfg = BottomSafetyConfig(send=False, dry_run=True, hover_only=True)
    mask_img = np.zeros((720, 1280), dtype=np.uint8)
    # In demo, pixel == board is not realistic but planning still compiles and runs.
    contour = np.asarray([[[120, 120]], [[390, 120]], [[390, 430]], [[120, 430]]], dtype=np.int32)
    cv2.drawContours(mask_img, [contour], -1, 255, -1)
    mask = BottomMaskBoard(mask_img, contour, float(cv2.contourArea(contour)), np.asarray([255, 275], dtype=np.float32),
                           np.asarray([255, -300], dtype=np.float32), "bottoms", 0.9, 0.85)
    pose = BottomPoseBoard(
        keypoints_board={}, keypoints_px={}, keypoint_conf={},
        waist_left=np.asarray([150, -340], dtype=np.float32),
        waist_center=np.asarray([255, -340], dtype=np.float32),
        waist_right=np.asarray([360, -340], dtype=np.float32),
        crotch=np.asarray([275, -260], dtype=np.float32),
        left_hem_center=np.asarray([185, -80], dtype=np.float32),
        right_hem_center=np.asarray([335, -80], dtype=np.float32),
        lower_center=np.asarray([260, -80], dtype=np.float32),
        pose_center=np.asarray([263, -205], dtype=np.float32),
        waist_angle_deg=0.0, pants_axis_angle_deg=88.0, pants_axis_len_mm=260.0,
        waist_width_mm=210.0, hem_gap_mm=150.0, valid=True, reason="OK",
    )
    obs = BottomObservation(pose, mask, 0.5 * pose.pose_center + 0.5 * mask.center_board, True, "OK", None)
    for builder in (build_pick_place_plan, build_waist_align_plan, build_crotch_align_plan):
        plan = builder(obs, H, config, cfg)
        print_plan(plan)


def main() -> int:
    parser = argparse.ArgumentParser(description="D23_v2 bottom control: center-outward-only motion + closed press priority + severe-only grip")
    parser.add_argument("--demo", action="store_true", help="Run synthetic planning demo and exit.")

    parser.add_argument("--config", default="dual_roarm_big_table_config.json")
    parser.add_argument("--calib-cache", default="dual_roarm_affine_cache.json")
    parser.add_argument("--hfile", default="dual_big_table_homography_cache.json")
    parser.add_argument("--load-h", action="store_true", default=True)
    parser.add_argument("--no-load-h", dest="load_h", action="store_false")
    parser.add_argument("--auto-lock", action="store_true", default=True)
    parser.add_argument("--save-h-on-lock", action="store_true", default=True)

    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--backend", choices=["auto", "v4l2", "dshow", "any"], default="v4l2")
    parser.add_argument("--cam-device", default="", help="UVC device path; empty uses /dev/video<camera>")
    parser.add_argument("--cam-exposure-abs", type=int, default=35)
    parser.add_argument("--cam-gain", type=int, default=5)
    parser.add_argument("--cam-no-manual", action="store_true", help="Do not change UVC exposure/gain")
    parser.add_argument("--cam-auto-adjust", dest="cam_auto_adjust", action="store_true", default=True,
                        help="Temporarily enable UVC auto exposure before each robot action, then lock it.")
    parser.add_argument("--no-cam-auto-adjust", dest="cam_auto_adjust", action="store_false")
    parser.add_argument("--cam-auto-settle-s", type=float, default=1.20)
    parser.add_argument("--cam-auto-exposure-min", type=int, default=10)
    parser.add_argument("--cam-auto-exposure-max", type=int, default=100)
    parser.add_argument("--cam-auto-gain-min", type=int, default=0)
    parser.add_argument("--cam-auto-gain-max", type=int, default=20)
    parser.add_argument("--no-window", action="store_true")
    parser.add_argument("--once", action="store_true",
                        help="Run one camera inference and exit. Useful in Docker/no-GUI mode; no L/I key press needed.")
    parser.add_argument("--warmup-frames", type=int, default=8,
                        help="Number of camera frames to discard before --once inference.")
    parser.add_argument("--debug-image", default="bottom_stage1_4_debug.jpg",
                        help="Path to save the --once overlay/debug image. Empty string disables saving.")

    parser.add_argument("--seg-model", default="/workspace/project_train/aruco_test/dual/models/kfashion_yolo26s_seg3_e100_best.engine")
    parser.add_argument("--seg-imgsz", type=int, default=640)
    parser.add_argument("--seg-conf", type=float, default=0.25)
    parser.add_argument("--pose-grabcut-fallback", dest="pose_grabcut_fallback", action="store_true", default=True)
    parser.add_argument("--no-pose-grabcut-fallback", dest="pose_grabcut_fallback", action="store_false")
    parser.add_argument("--fallback-mask-min-pose-points", type=int, default=6)
    parser.add_argument("--fallback-mask-bbox-pad-px", type=int, default=55)
    parser.add_argument("--fallback-mask-seed-radius-px", type=int, default=12)
    parser.add_argument("--fallback-mask-seed-line-px", type=int, default=18)
    parser.add_argument("--fallback-mask-hull-dilate-px", type=int, default=28)
    parser.add_argument("--fallback-mask-grabcut-iters", type=int, default=4)
    parser.add_argument("--fallback-mask-close-px", type=int, default=9)
    parser.add_argument("--fallback-mask-open-px", type=int, default=3)
    parser.add_argument("--fallback-mask-min-area-px", type=float, default=1800.0)
    parser.add_argument("--fallback-mask-max-frame-ratio", type=float, default=0.70)
    parser.add_argument("--fallback-mask-max-kpt-outside-px", type=float, default=22.0)
    parser.add_argument("--fallback-mask-min-inside-points", type=int, default=6)
    parser.add_argument("--fallback-mask-min-solidity", type=float, default=0.35)
    parser.add_argument("--pose-model", default="/home/deca/project_train/yolo26-pose-small-down/best.engine")
    parser.add_argument("--pose-imgsz", type=int, default=640)
    parser.add_argument("--pose-conf", type=float, default=0.10)
    parser.add_argument("--pose-kpt-conf", type=float, default=0.20)

    # D2-style arbitrary-orientation pose TTA. Pose is restored to the original frame before Homography conversion.
    parser.add_argument("--pose-tta", dest="pose_tta", action="store_true", default=True)
    parser.add_argument("--no-pose-tta", dest="pose_tta", action="store_false")
    parser.add_argument("--angles", "--pose-tta-angles", dest="angles", default="0,180,90,-90,45,-45")
    parser.add_argument("--flip-modes", "--pose-tta-flip-modes", dest="flip_modes", default="none,h")
    parser.add_argument("--tta-min-waist-width-px", type=float, default=40.0)
    parser.add_argument("--tta-max-waist-center-offset-ratio", type=float, default=0.35)
    parser.add_argument("--tta-min-waist-to-crotch-ratio", type=float, default=0.35)
    parser.add_argument("--tta-max-waist-axis-dot", type=float, default=0.45)
    parser.add_argument("--tta-pre-spread-hem-gap-ratio", type=float, default=0.42)
    parser.add_argument("--tta-pre-spread-open-angle-deg", type=float, default=22.0)
    parser.add_argument("--tta-pre-spread-min-hem-visible", type=int, default=2)
    parser.add_argument("--tta-mask-max-outside-px", type=float, default=28.0)
    parser.add_argument("--tta-mask-min-inside-ratio", type=float, default=0.60)
    parser.add_argument("--tta-require-ok-for-motion", dest="tta_require_ok_for_motion", action="store_true", default=True)
    parser.add_argument("--no-tta-require-ok-for-motion", dest="tta_require_ok_for_motion", action="store_false")

    # D23 staged TTA: use the last successful/common transforms first, then full fallback.
    parser.add_argument("--d23-pose-tta-fast-first", dest="d23_pose_tta_fast_first", action="store_true", default=True)
    parser.add_argument("--no-d23-pose-tta-fast-first", dest="d23_pose_tta_fast_first", action="store_false")
    parser.add_argument("--d23-pose-tta-fast-angles", default="0,180")
    parser.add_argument("--d23-pose-tta-fast-flip-modes", default="none")
    parser.add_argument("--d23-pose-tta-fast-min-score", type=float, default=28.0)
    parser.add_argument("--d23-pose-tta-fast-min-visible", type=int, default=7)
    parser.add_argument("--d23-pose-tta-fast-min-mean-conf", type=float, default=0.32)

    # B34-style wrinkle heatmap. Applied only to visualization, not pose input.
    parser.add_argument("--wrinkle-heatmap-mode", dest="wrinkle_heatmap_mode", action="store_true", default=True)
    parser.add_argument("--no-wrinkle-heatmap-mode", dest="wrinkle_heatmap_mode", action="store_false")
    parser.add_argument("--wrinkle-heatmap-draw", dest="wrinkle_heatmap_draw", action="store_true", default=True)
    parser.add_argument("--no-wrinkle-heatmap-draw", dest="wrinkle_heatmap_draw", action="store_false")
    parser.add_argument("--wrinkle-heatmap-alpha", type=float, default=0.42)
    parser.add_argument("--wrinkle-heatmap-clahe-clip", type=float, default=2.0)
    parser.add_argument("--wrinkle-heatmap-clahe-tile", type=int, default=8)
    parser.add_argument("--wrinkle-heatmap-blur-ksize", type=int, default=41)
    parser.add_argument("--wrinkle-heatmap-erode-px", type=int, default=12)
    parser.add_argument("--wrinkle-heatmap-percentile", type=float, default=93.0)
    parser.add_argument("--wrinkle-heatmap-min-blob-area", type=float, default=45.0)
    parser.add_argument("--wrinkle-heatmap-max-candidates", type=int, default=6)
    parser.add_argument("--wrinkle-heatmap-gabor", dest="wrinkle_heatmap_gabor", action="store_true", default=True)
    parser.add_argument("--no-wrinkle-heatmap-gabor", dest="wrinkle_heatmap_gabor", action="store_false")
    parser.add_argument("--wrinkle-heatmap-gabor-ksize", type=int, default=17)
    parser.add_argument("--wrinkle-heatmap-gabor-lambda", type=float, default=8.0)
    parser.add_argument("--wrinkle-heatmap-gabor-sigma", type=float, default=4.0)
    parser.add_argument("--wrinkle-heatmap-score-min", type=float, default=35.0)

    # D16: B34 active-pull/passive-hold adapted to the pants mask.
    parser.add_argument("--b34-targeted-sweep-mm", type=float, default=120.0,
                        help="Base active-arm pull distance; D16 uses a stronger default.")
    parser.add_argument("--b34-max-targeted-sweep-mm", type=float, default=160.0,
                        help="Safety cap for adaptive pull distance.")
    parser.add_argument("--b34-adaptive-pull", dest="b34_adaptive_pull", action="store_true", default=True)
    parser.add_argument("--no-b34-adaptive-pull", dest="b34_adaptive_pull", action="store_false")
    parser.add_argument("--b34-adaptive-pull-max-bonus-mm", type=float, default=25.0)
    parser.add_argument("--b34-min-actual-pull-mm", type=float, default=60.0)
    parser.add_argument("--robot-motion-inset-mm", type=float, default=18.0,
                        help="Shift active robot contact inward from the detected point; perception is unchanged.")
    parser.add_argument("--robot-hold-inset-mm", type=float, default=12.0,
                        help="Shift passive hold contact inward from the detected point.")
    parser.add_argument("--b34-edge-contact-inset-mm", type=float, default=32.0)
    parser.add_argument("--b34-edge-ray-width-mm", type=float, default=42.0)
    parser.add_argument("--b34-edge-band-mm", type=float, default=55.0)
    parser.add_argument("--b34-min-major-ratio-for-edge-pull", type=float, default=1.15)
    parser.add_argument("--b34-outward-fallback-offset-mm", type=float, default=28.0)
    parser.add_argument("--b34-outward-fallback-min-local", type=float, default=0.10)
    parser.add_argument("--b34-contact-snap-mm", type=float, default=35.0)
    parser.add_argument("--b34-local-mask-radius-px", type=int, default=18)
    parser.add_argument("--b34-min-local-mask-ratio", type=float, default=0.18)
    parser.add_argument("--b34-board-margin-mm", type=float, default=35.0)
    parser.add_argument("--b34-min-contact-outward-mm", type=float, default=12.0)
    parser.add_argument("--b34-min-sweep-mm", type=float, default=20.0)
    parser.add_argument("--b34-press-gripper", type=float, default=3.15)
    parser.add_argument("--b34-passive-hold-gripper", type=float, default=3.18)
    parser.add_argument("--b34-passive-hold-z", type=float, default=-104.0)
    parser.add_argument("--b34-passive-hold-offset-mm", type=float, default=60.0)
    parser.add_argument("--b34-passive-hold-max-offset-mm", type=float, default=100.0)
    parser.add_argument("--b34-passive-hold-min-local-mask-ratio", type=float, default=0.10)
    parser.add_argument("--b34-passive-hold-max-snap-mm", type=float, default=35.0)
    parser.add_argument("--b34-passive-hold-fallback-max-snap-mm", type=float, default=110.0)
    parser.add_argument("--b34-passive-hold-min-pair-distance-mm", type=float, default=90.0)
    parser.add_argument("--b34-passive-hold-min-sweep-clearance-mm", type=float, default=90.0)
    parser.add_argument("--b34-passive-hold-before-active-wait-s", type=float, default=2.50)
    parser.add_argument("--b34-passive-hold-after-active-lift-wait-s", type=float, default=0.45)
    # D18: pose/wrinkle point -> actual dense segmentation-mask pixel -> inward correction.
    parser.add_argument("--d18-mask-first-grip", dest="d18_mask_first_grip",
                        action="store_true", default=True)
    parser.add_argument("--no-d18-mask-first-grip", dest="d18_mask_first_grip",
                        action="store_false")
    parser.add_argument("--d18-mask-snap-required", dest="d18_mask_snap_required",
                        action="store_true", default=True)
    parser.add_argument("--no-d18-mask-snap-required", dest="d18_mask_snap_required",
                        action="store_false")
    parser.add_argument("--d18-mask-snap-radius-mm", type=float, default=75.0)
    parser.add_argument("--d18-mask-snap-min-inside-px", type=float, default=4.0)
    parser.add_argument("--d18-mask-snap-min-local-ratio", type=float, default=0.22)
    parser.add_argument("--d18-mask-snap-local-radius-px", type=int, default=18)
    parser.add_argument("--d18-mask-snap-inside-weight", type=float, default=2.8)
    parser.add_argument("--d18-mask-snap-local-weight", type=float, default=45.0)
    parser.add_argument("--d18-mask-max-sample-points", type=int, default=24000)
    parser.add_argument("--d18-mask-snap-allow-weak-fallback",
                        action="store_true", default=False)

    parser.add_argument("--d16-grip-inset-mm", type=float, default=28.0,
                        help="Move active/dual robot contact inward from the visual target.")
    parser.add_argument("--d16-hold-grip-inset-mm", type=float, default=28.0,
                        help="Move passive-hold contact inward from the visual target.")
    parser.add_argument("--d16-grip-max-inset-mm", type=float, default=85.0)
    parser.add_argument("--d16-grip-min-inside-px", type=float, default=2.0)
    parser.add_argument("--d16-grip-min-local-ratio", type=float, default=0.18)
    parser.add_argument("--d16-grip-local-radius-px", type=int, default=18)
    parser.add_argument("--d16-grip-snap-radius-mm", type=float, default=65.0)
    parser.add_argument("--d16-grip-snap-refine-mm", type=float, default=10.0)
    parser.add_argument("--d16-v3-min-corrected-motion-mm", type=float, default=25.0)

    parser.add_argument("--d16-v3-two-arm-pre-smooth-mm", type=float, default=70.0)
    parser.add_argument("--d16-v3-pre-smooth-min-actual-mm", type=float, default=30.0)
    parser.add_argument("--d16-v3-edge-fallback-dual-mm", type=float, default=65.0)
    parser.add_argument("--d16-v3-short-fallback-dual-mm", type=float, default=60.0)
    parser.add_argument("--d16-v3-round-fallback-dual-mm", type=float, default=50.0)
    parser.add_argument("--d16-aggression-scale", type=float, default=1.30,
                        help="Multiply pull distances; bounded by --d16-max-aggression-scale.")
    parser.add_argument("--d16-max-aggression-scale", type=float, default=1.50)
    parser.add_argument("--d16-long-wrinkle-min-mm", type=float, default=120.0)
    parser.add_argument("--d16-short-wrinkle-min-mm", type=float, default=35.0)
    parser.add_argument("--d16-long-linearity-min", type=float, default=1.80)
    parser.add_argument("--d16-short-linearity-min", type=float, default=1.30)
    parser.add_argument("--d16-round-linearity-max", type=float, default=1.30)
    parser.add_argument("--d16-round-blob-ratio-min", type=float, default=0.008)
    parser.add_argument("--d16-edge-distance-px", type=float, default=12.0)
    parser.add_argument("--d16-dual-grip-offset-mm", type=float, default=48.0)
    parser.add_argument("--d16-dual-grip-inset-mm", type=float, default=12.0)
    parser.add_argument("--d16-dual-snap-radius-mm", type=float, default=85.0)
    parser.add_argument("--d16-dual-min-grip-separation-mm", type=float, default=100.0)
    parser.add_argument("--d16-dual-stretch-mm", type=float, default=105.0)
    parser.add_argument("--d16-dual-max-stretch-mm", type=float, default=140.0)
    parser.add_argument("--d16-dual-min-actual-stretch-mm", type=float, default=35.0)
    parser.add_argument("--d16-dual-min-local-mask-ratio", type=float, default=0.12)
    parser.add_argument("--d16-dual-gripper", type=float, default=3.15)
    parser.add_argument("--d16-short-pull-mm", type=float, default=140.0)
    parser.add_argument("--d16-edge-sweep-mm", type=float, default=155.0)
    parser.add_argument("--d16-long-fallback-pull-mm", type=float, default=160.0)
    parser.add_argument("--d16-hold-pull-min-actual-mm", type=float, default=45.0)
    parser.add_argument("--d16-pre-smooth-mm", type=float, default=60.0)
    parser.add_argument("--d16-pre-smooth-max-mm", type=float, default=80.0)
    parser.add_argument("--d16-pre-smooth-min-actual-mm", type=float, default=20.0)
    parser.add_argument("--d16-motion-sec-per-mm", type=float, default=0.018)
    parser.add_argument("--d16-min-stage-wait-s", type=float, default=0.65)
    parser.add_argument("--d16-max-stage-wait-s", type=float, default=4.20)
    parser.add_argument("--d16-stage-base-wait-s", type=float, default=0.22)
    parser.add_argument("--d16-hold-settle-s", type=float, default=0.80)
    parser.add_argument("--d16-down-extra-wait-s", type=float, default=0.30)
    parser.add_argument("--d16-pull-extra-wait-s", type=float, default=0.35)
    parser.add_argument("--d16-lift-extra-wait-s", type=float, default=0.70)
    parser.add_argument("--d16-hold-release-delay-s", type=float, default=0.55)
    parser.add_argument("--d16-reassert-hold", dest="d16_reassert_hold",
                        action="store_true", default=True)
    parser.add_argument("--no-d16-reassert-hold", dest="d16_reassert_hold",
                        action="store_false")
    parser.add_argument("--b34-press-z", type=float, default=-104.0)
    parser.add_argument("--b34-clear-z", type=float, default=-25.0)
    parser.add_argument("--b34-action-speed", type=float, default=0.26)
    parser.add_argument("--b34-micro-mid-ratio", type=float, default=0.55)
    parser.add_argument("--b34-micro-mid-wait-s", type=float, default=0.035)
    parser.add_argument("--b34-press-down-hold-s", type=float, default=0.08)
    parser.add_argument("--b34-wait-before-down-s", type=float, default=0.03)
    parser.add_argument("--b34-wait-after-down-s", type=float, default=0.04)
    parser.add_argument("--b34-wait-after-sweep-s", type=float, default=0.04)
    parser.add_argument("--b34-wait-after-clear-s", type=float, default=0.05)
    parser.add_argument("--b34-wait-after-open-s", type=float, default=0.08)
    parser.add_argument("--b34-feedback-repeat-radius-mm", type=float, default=55.0)
    parser.add_argument("--b34-feedback-same-bad-penalty", type=float, default=120.0)
    parser.add_argument("--b34-feedback-reverse-bonus", type=float, default=25.0)
    parser.add_argument("--b34-feedback-history-max", type=int, default=10)
    parser.add_argument("--wrinkle-min-ratio-improvement", type=float, default=0.08)
    parser.add_argument("--wrinkle-min-blob-improvement", type=float, default=0.15)
    parser.add_argument("--wrinkle-max-center-shift-mm", type=float, default=35.0)
    parser.add_argument("--wrinkle-min-mask-area-ratio", type=float, default=0.93)
    parser.add_argument("--wrinkle-after-settle-s", type=float, default=1.0)
    parser.add_argument("--finish-wrinkle-ratio", type=float, default=0.035)
    parser.add_argument("--finish-max-blob-ratio", type=float, default=0.012)
    parser.add_argument("--finish-max-blob-count", type=int, default=2)
    parser.add_argument("--finish-confirm-cycles", type=int, default=2,
                        help="Require consecutive fresh camera-clear confirmations before FINISHED_CONFIRMED.")

    # D19: ignore natural pants structures and stop at a practical finish state.
    parser.add_argument("--d19-ignore-waistband-wrinkles", dest="d19_ignore_waistband_wrinkles",
                        action="store_true", default=True)
    parser.add_argument("--no-d19-ignore-waistband-wrinkles", dest="d19_ignore_waistband_wrinkles",
                        action="store_false")
    parser.add_argument("--d19-waistband-back-mm", type=float, default=18.0)
    parser.add_argument("--d19-waistband-forward-mm", type=float, default=82.0)
    parser.add_argument("--d19-waistband-side-expand-mm", type=float, default=30.0)
    parser.add_argument("--d23-waistband-forward-ratio", type=float, default=0.70)
    parser.add_argument("--d23-waistband-forward-max-mm", type=float, default=105.0)
    parser.add_argument("--d23-waist-structure-veto", dest="d23_waist_structure_veto", action="store_true", default=True)
    parser.add_argument("--no-d23-waist-structure-veto", dest="d23_waist_structure_veto", action="store_false")
    parser.add_argument("--d23-waist-structure-back-mm", type=float, default=24.0)
    parser.add_argument("--d23-waist-structure-forward-mm", type=float, default=92.0)
    parser.add_argument("--d23-waist-structure-forward-ratio", type=float, default=0.78)
    parser.add_argument("--d23-waist-structure-forward-max-mm", type=float, default=115.0)
    parser.add_argument("--d23-waist-structure-side-expand-mm", type=float, default=38.0)
    parser.add_argument("--d23-waist-structure-parallel-deg", type=float, default=28.0)
    parser.add_argument("--d23-waist-structure-min-length-mm", type=float, default=35.0)
    parser.add_argument("--d23-waist-structure-min-length-ratio", type=float, default=0.22)
    parser.add_argument("--d23-pre-veto-candidate-multiplier", type=int, default=3)
    parser.add_argument("--d23-draw-waist-veto", dest="d23_draw_waist_veto", action="store_true", default=True)
    parser.add_argument("--no-d23-draw-waist-veto", dest="d23_draw_waist_veto", action="store_false")
    parser.add_argument("--d19-ignore-crotch-wrinkles", dest="d19_ignore_crotch_wrinkles",
                        action="store_true", default=True)
    parser.add_argument("--no-d19-ignore-crotch-wrinkles", dest="d19_ignore_crotch_wrinkles",
                        action="store_false")
    parser.add_argument("--d19-crotch-axial-radius-mm", type=float, default=72.0)
    parser.add_argument("--d19-crotch-lateral-radius-mm", type=float, default=78.0)
    parser.add_argument("--d19-crotch-forward-shift-mm", type=float, default=12.0)

    parser.add_argument("--d19-action-min-blob-area-px", type=float, default=120.0)
    parser.add_argument("--d19-action-min-blob-area-ratio", type=float, default=0.0008)
    parser.add_argument("--d19-action-min-major-len-mm", type=float, default=45.0)
    parser.add_argument("--d19-action-min-major-len-px", type=float, default=58.0)
    parser.add_argument("--d19-action-broad-min-area-px", type=float, default=220.0)

    parser.add_argument("--d19-finish-wrinkle-ratio", type=float, default=0.050)
    parser.add_argument("--d19-finish-max-blob-ratio", type=float, default=0.020)
    parser.add_argument("--d19-finish-max-blob-count", type=int, default=3)
    parser.add_argument("--d19-finish-meaningful-blob-area-px", type=float, default=140.0)
    parser.add_argument("--d19-finish-meaningful-blob-area-ratio", type=float, default=0.0010)
    parser.add_argument("--d19-finish-max-meaningful-blob-count", type=int, default=1)
    parser.add_argument("--d19-finish-blob-area-cv", type=float, default=1.55)
    parser.add_argument("--d19-finish-long-wrinkle-len-mm", type=float, default=55.0)
    parser.add_argument("--d19-finish-long-wrinkle-len-px", type=float, default=70.0)
    parser.add_argument("--d19-finish-long-wrinkle-aspect", type=float, default=2.0)
    parser.add_argument("--d19-finish-long-wrinkle-min-area-px", type=float, default=120.0)
    parser.add_argument("--d19-finish-max-long-wrinkle-count", type=int, default=0)
    parser.add_argument("--d19-finish-require-shape", dest="d19_finish_require_shape",
                        action="store_true", default=True)
    parser.add_argument("--no-d19-finish-require-shape", dest="d19_finish_require_shape",
                        action="store_false")
    parser.add_argument("--d19-finish-shape-score-good", type=float, default=0.78)
    parser.add_argument("--d19-finish-shape-hem-gap-ratio-min", type=float, default=0.52)
    parser.add_argument("--d19-finish-shape-hem-gap-ratio-max", type=float, default=1.40)
    parser.add_argument("--d19-finish-shape-leg-balance-min", type=float, default=0.80)
    parser.add_argument("--d19-finish-shape-waist-center-offset-max", type=float, default=0.16)
    parser.add_argument("--d19-finish-shape-waist-line-error-max", type=float, default=0.16)
    parser.add_argument("--d19-finish-shape-crotch-axis-offset-max", type=float, default=0.24)
    parser.add_argument("--d19-finish-shape-axis-to-waist-min", type=float, default=0.72)

    # D20: adaptive local-dark-shadow ranking for black pants.
    parser.add_argument("--d20-require-dark-shadow", dest="d20_require_dark_shadow",
                        action="store_true", default=True,
                        help="Require locally darker shadow evidence before treating a component as actionable.")
    parser.add_argument("--no-d20-require-dark-shadow", dest="d20_require_dark_shadow",
                        action="store_false")
    parser.add_argument("--d20-ignore-fine-wrinkles", dest="d20_ignore_fine_wrinkles",
                        action="store_true", default=True,
                        help="Ignore fine, short, small, or weak-dark components.")
    parser.add_argument("--no-d20-ignore-fine-wrinkles", dest="d20_ignore_fine_wrinkles",
                        action="store_false")

    parser.add_argument("--d20-shadow-blur-ksize", type=int, default=51,
                        help="Local brightness baseline window for relative dark-shadow depth.")
    parser.add_argument("--d20-shadow-min-scale-raw", type=float, default=8.0,
                        help="Minimum LAB-L residual normalization scale; prevents black-fabric noise amplification.")
    parser.add_argument("--d20-shadow-support-raw-min", type=float, default=3.0)

    parser.add_argument("--d20-strong-area-px", type=float, default=240.0)
    parser.add_argument("--d20-strong-area-ratio", type=float, default=0.0015)
    parser.add_argument("--d20-strong-length-mm", type=float, default=65.0)
    parser.add_argument("--d20-strong-length-px", type=float, default=85.0)
    parser.add_argument("--d20-strong-dark-mean", type=float, default=38.0)
    parser.add_argument("--d20-strong-dark-max", type=float, default=105.0)
    parser.add_argument("--d20-strong-dark-support", type=float, default=0.10)

    parser.add_argument("--d20-medium-area-px", type=float, default=150.0)
    parser.add_argument("--d20-medium-area-ratio", type=float, default=0.0009)
    parser.add_argument("--d20-medium-length-mm", type=float, default=45.0)
    parser.add_argument("--d20-medium-length-px", type=float, default=58.0)
    parser.add_argument("--d20-medium-dark-mean", type=float, default=20.0)
    parser.add_argument("--d20-medium-dark-max", type=float, default=65.0)
    parser.add_argument("--d20-medium-dark-support", type=float, default=0.06)
    parser.add_argument("--d20-hard-min-component-area-px", type=float, default=70.0)

    # D20 practical finish. Fine/weak-dark wrinkles are already excluded.
    parser.add_argument("--d20-finish-max-large-long-dark-count", type=int, default=0)
    parser.add_argument("--d20-finish-max-large-or-long-count", type=int, default=0)
    parser.add_argument("--d20-finish-max-smaller-actionable-count", type=int, default=1)
    parser.add_argument("--d20-finish-max-total-actionable-count", type=int, default=2)
    parser.add_argument("--d20-finish-actionable-ratio", type=float, default=0.035)
    parser.add_argument("--d20-finish-max-blob-ratio", type=float, default=0.015)

    # D21 geometry classification and edge-grip pull.
    parser.add_argument("--d21-geometry-grip-pull", dest="d21_geometry_grip_pull", action="store_true", default=True)
    parser.add_argument("--no-d21-geometry-grip-pull", dest="d21_geometry_grip_pull", action="store_false")
    parser.add_argument("--d21-line-max-residual-mm", type=float, default=8.0)
    parser.add_argument("--d21-line-max-sagitta-mm", type=float, default=10.0)
    parser.add_argument("--d21-curve-min-inlier-ratio", type=float, default=0.62)
    parser.add_argument("--d21-curve-max-residual-mm", type=float, default=8.0)
    parser.add_argument("--d21-curve-min-sagitta-mm", type=float, default=10.0)
    parser.add_argument("--d21-curve-min-arc-length-mm", type=float, default=50.0)
    parser.add_argument("--d21-curve-min-radius-mm", type=float, default=25.0)
    parser.add_argument("--d21-curve-max-radius-mm", type=float, default=900.0)
    parser.add_argument("--d21-curve-vs-line-error-ratio", type=float, default=0.90)
    parser.add_argument("--d21-circle-ransac-iters", type=int, default=120)
    parser.add_argument("--d21-circle-inlier-tol-mm", type=float, default=7.0)
    parser.add_argument("--d21-fit-max-points", type=int, default=220)
    parser.add_argument("--d21-edge-band-mm", type=float, default=65.0)
    parser.add_argument("--d21-active-edge-inset-mm", type=float, default=24.0)
    parser.add_argument("--d21-hold-edge-inset-mm", type=float, default=34.0)
    parser.add_argument("--d21-edge-snap-radius-mm", type=float, default=42.0)
    parser.add_argument("--d21-pull-min-mm", type=float, default=45.0)
    parser.add_argument("--d21-pull-max-mm", type=float, default=140.0)
    parser.add_argument("--d21-min-actual-pull-mm", type=float, default=35.0)
    parser.add_argument("--d21-min-separation-gain-mm", type=float, default=25.0)
    parser.add_argument("--d21-severity-length-ref-mm", type=float, default=120.0)
    parser.add_argument("--d21-severity-area-ref-px", type=float, default=550.0)
    parser.add_argument("--d21-severity-dark-ref", type=float, default=60.0)
    parser.add_argument("--d21-severity-dark-max-ref", type=float, default=160.0)
    parser.add_argument("--d21-severity-sagitta-ref-mm", type=float, default=45.0)

    # Active arm truly grips; opposite arm presses the outer side down.
    parser.add_argument("--d21-active-grip-z", type=float, default=None)
    parser.add_argument("--d21-hold-press-z", type=float, default=-104.0)
    parser.add_argument("--d21-grip-test-lift-mm", type=float, default=8.0)
    parser.add_argument("--d21-release-above-contact-mm", type=float, default=2.0)
    parser.add_argument("--d21-active-grip-open", type=float, default=1.35)
    parser.add_argument("--d21-active-grip-close", type=float, default=3.05)
    parser.add_argument("--d21-hold-gripper", type=float, default=3.05)
    parser.add_argument("--d21-active-close-repeat", type=int, default=2)
    parser.add_argument("--d21-grip-pull-speed", type=float, default=0.20)
    parser.add_argument("--d21-stage-wait-s", type=float, default=0.70)
    parser.add_argument("--d21-hold-settle-s", type=float, default=0.65)
    parser.add_argument("--d21-down-wait-s", type=float, default=0.65)
    parser.add_argument("--d21-close-wait-s", type=float, default=0.30)
    parser.add_argument("--d21-close-settle-s", type=float, default=0.45)
    parser.add_argument("--d21-test-lift-wait-s", type=float, default=0.45)
    parser.add_argument("--d21-pull-wait-s", type=float, default=0.65)
    parser.add_argument("--d21-tension-hold-s", type=float, default=0.40)
    parser.add_argument("--d21-release-settle-s", type=float, default=0.30)
    parser.add_argument("--d21-release-open-wait-s", type=float, default=0.35)
    parser.add_argument("--d21-clear-wait-s", type=float, default=0.55)

    # Two-action stabilization logic.
    parser.add_argument("--d21-static-required-actions", type=int, default=2)
    parser.add_argument("--d21-track-match-distance-mm", type=float, default=35.0)
    parser.add_argument("--d21-static-center-shift-mm", type=float, default=25.0)
    parser.add_argument("--d21-static-length-rel-change", type=float, default=0.15)
    parser.add_argument("--d21-static-area-rel-change", type=float, default=0.20)
    parser.add_argument("--d21-static-dark-rel-change", type=float, default=0.12)

    # D21_v3: unchanged wrinkles become intrinsic only when their level is mild.
    parser.add_argument("--d21v3-static-require-fine", dest="d21v3_static_require_fine",
                        action="store_true", default=True)
    parser.add_argument("--no-d21v3-static-require-fine", dest="d21v3_static_require_fine",
                        action="store_false")
    parser.add_argument("--d21v3-static-max-tier", type=int, default=1)
    parser.add_argument("--d21v3-static-max-severity", type=float, default=0.45)
    parser.add_argument("--d21v3-static-max-length-mm", type=float, default=65.0)
    parser.add_argument("--d21v3-static-max-area-px", type=float, default=240.0)
    parser.add_argument("--d21v3-static-max-dark-mean", type=float, default=38.0)
    parser.add_argument("--d21v3-finish-max-nonfine-count", type=int, default=0,
                        help="Stable-shape finish is blocked while this many severe/non-fine wrinkles remain.")
    parser.add_argument("--d21v3-finish-max-fine-count", type=int, default=1)


    # D21_v4: practical GOOD_ENOUGH finish. T1 residual wrinkles are allowed and
    # are excluded from motion; only shape defects and meaningful T2/T3 wrinkles block finish.
    parser.add_argument("--d21v4-good-enough-finish", dest="d21v4_good_enough_finish",
                        action="store_true", default=True)
    parser.add_argument("--no-d21v4-good-enough-finish", dest="d21v4_good_enough_finish",
                        action="store_false")
    parser.add_argument("--d21v4-action-min-tier", type=int, default=2)
    parser.add_argument("--d21v4-action-severity-override", type=float, default=0.55)
    parser.add_argument("--d21v4-finish-max-t3-count", type=int, default=0)
    parser.add_argument("--d21v4-finish-max-t2-count", type=int, default=1)
    parser.add_argument("--d21v4-finish-max-severity", type=float, default=0.55)
    parser.add_argument("--d21v4-finish-actionable-ratio", type=float, default=0.030)
    parser.add_argument("--d21v4-finish-max-blob-ratio", type=float, default=0.015)
    parser.add_argument("--d21v4-finish-min-shape-score", type=float, default=0.72)
    parser.add_argument("--d21v4-finish-confirm-cycles", type=int, default=2)
    parser.add_argument("--d21-stable-required-actions", type=int, default=2)
    parser.add_argument("--d21-stable-mask-area-rel", type=float, default=0.025)
    parser.add_argument("--d21-stable-center-shift-mm", type=float, default=15.0)
    parser.add_argument("--d21-stable-waist-rel", type=float, default=0.05)
    parser.add_argument("--d21-stable-hem-rel", type=float, default=0.05)
    parser.add_argument("--d21-stable-shape-score-delta", type=float, default=0.03)
    parser.add_argument("--d21-stable-finish-min-shape-score", type=float, default=0.72)

    # D21_v2: finish only when the segmentation silhouette is sufficiently
    # symmetric about waist-mid -> crotch axis.
    parser.add_argument("--d21v2-require-symmetry", dest="d21v2_require_symmetry",
                        action="store_true", default=True)
    parser.add_argument("--no-d21v2-require-symmetry", dest="d21v2_require_symmetry",
                        action="store_false")
    parser.add_argument("--d21v2-symmetry-score-min", type=float, default=0.74)
    parser.add_argument("--d21v2-reflected-overlap-min", type=float, default=0.70)
    parser.add_argument("--d21v2-side-area-balance-min", type=float, default=0.78)
    parser.add_argument("--d21v2-hem-axis-balance-min", type=float, default=0.76)
    parser.add_argument("--d21v2-leg-length-balance-min", type=float, default=0.80)
    parser.add_argument("--d21v2-symmetry-axis-min-mm", type=float, default=35.0)
    parser.add_argument("--d21v2-symmetry-ignore-axis-band-mm", type=float, default=6.0)
    parser.add_argument("--d21v2-symmetry-tolerance-px", type=int, default=8)
    parser.add_argument("--d21v2-symmetry-close-px", type=int, default=5)
    parser.add_argument("--d21v2-symmetry-max-sample-points", type=int, default=30000)
    # D22: strict lower-leg silhouette symmetry. The broad waist/body area cannot
    # compensate for a folded or narrow leg.
    parser.add_argument("--d22-require-lower-leg-symmetry", dest="d22_require_lower_leg_symmetry",
                        action="store_true", default=True)
    parser.add_argument("--no-d22-require-lower-leg-symmetry", dest="d22_require_lower_leg_symmetry",
                        action="store_false")
    parser.add_argument("--d22-symmetry-axis-min-mm", type=float, default=35.0)
    parser.add_argument("--d22-symmetry-lower-start-mm", type=float, default=-4.0)
    parser.add_argument("--d22-symmetry-ignore-axis-band-mm", type=float, default=5.0)
    parser.add_argument("--d22-symmetry-grid-mm", type=float, default=4.0)
    parser.add_argument("--d22-symmetry-grid-close-cells", type=int, default=1)
    parser.add_argument("--d22-symmetry-band-count", type=int, default=6)
    parser.add_argument("--d22-symmetry-min-points-per-band", type=int, default=60)
    parser.add_argument("--d22-symmetry-max-sample-points", type=int, default=100000)
    parser.add_argument("--d22-lower-reflected-iou-min", type=float, default=0.62)
    parser.add_argument("--d22-lower-width-balance-median-min", type=float, default=0.78)
    parser.add_argument("--d22-lower-width-balance-min-min", type=float, default=0.60)
    parser.add_argument("--d22-lower-profile-error-p90-max", type=float, default=0.38)
    parser.add_argument("--d22-lower-side-area-balance-min", type=float, default=0.74)
    parser.add_argument("--d22-hem-axis-balance-min", type=float, default=0.74)
    parser.add_argument("--d22-leg-length-balance-min", type=float, default=0.78)
    parser.add_argument("--d22-symmetry-score-min", type=float, default=0.70)
    parser.add_argument("--d22-asymmetry-correction", dest="d22_asymmetry_correction",
                        action="store_true", default=True)
    parser.add_argument("--no-d22-asymmetry-correction", dest="d22_asymmetry_correction",
                        action="store_false")
    parser.add_argument("--d22-asymmetry-pull-min-mm", type=float, default=55.0)
    parser.add_argument("--d22-asymmetry-pull-max-mm", type=float, default=105.0)

    # D22 hybrid wrinkle action: strong geometric wrinkles are gripped and pulled;
    # moderate/irregular wrinkles are pressed and swept at low Z.
    parser.add_argument("--d22-hybrid-wrinkle-policy", dest="d22_hybrid_wrinkle_policy",
                        action="store_true", default=True)
    parser.add_argument("--no-d22-hybrid-wrinkle-policy", dest="d22_hybrid_wrinkle_policy",
                        action="store_false")
    parser.add_argument("--d22-grip-min-tier", type=int, default=3)
    parser.add_argument("--d22-grip-min-severity", type=float, default=0.58)
    parser.add_argument("--d22-grip-min-length-mm", type=float, default=70.0)
    parser.add_argument("--d22-grip-min-dark-mean", type=float, default=24.0)
    parser.add_argument("--d22-grip-min-fit-confidence", type=float, default=0.25)
    parser.add_argument("--d22-grip-fallback-to-press", dest="d22_grip_fallback_to_press",
                        action="store_true", default=True)
    parser.add_argument("--no-d22-grip-fallback-to-press", dest="d22_grip_fallback_to_press",
                        action="store_false")
    parser.add_argument("--d22-press-sweep-min-mm", type=float, default=45.0)
    parser.add_argument("--d22-press-sweep-max-mm", type=float, default=85.0)
    parser.add_argument("--d22-press-min-actual-mm", type=float, default=20.0)

    # D23_v2: center-outward invariant and closed-gripper press priority.
    parser.add_argument("--d23v2-center-outward-policy", dest="d23v2_center_outward_policy",
                        action="store_true", default=True)
    parser.add_argument("--no-d23v2-center-outward-policy", dest="d23v2_center_outward_policy",
                        action="store_false")
    parser.add_argument("--d23v2-min-reference-radius-mm", type=float, default=12.0)
    parser.add_argument("--d23v2-min-outward-cos", type=float, default=0.15)
    parser.add_argument("--d23v2-min-radial-gain-mm", type=float, default=12.0)
    parser.add_argument("--d23v2-press-normal-weight", type=float, default=0.78)
    parser.add_argument("--d23v2-press-start-inward-mm", type=float, default=24.0)
    parser.add_argument("--d23v2-press-snap-radius-mm", type=float, default=75.0)
    parser.add_argument("--d23v2-press-gripper", type=float, default=2.95)
    parser.add_argument("--d23v2-grip-require-all", dest="d23v2_grip_require_all",
                        action="store_true", default=True)
    parser.add_argument("--no-d23v2-grip-require-all", dest="d23v2_grip_require_all",
                        action="store_false")
    parser.add_argument("--d23v2-grip-min-tier", type=int, default=3)
    parser.add_argument("--d23v2-grip-min-severity", type=float, default=0.72)
    parser.add_argument("--d23v2-grip-min-length-mm", type=float, default=90.0)
    parser.add_argument("--d23v2-grip-min-dark-mean", type=float, default=30.0)
    parser.add_argument("--d23v2-grip-min-fit-confidence", type=float, default=0.35)
    parser.add_argument("--d23v2-shape-asymmetry-press-first", dest="d23v2_shape_asymmetry_press_first",
                        action="store_true", default=True)
    parser.add_argument("--no-d23v2-shape-asymmetry-press-first", dest="d23v2_shape_asymmetry_press_first",
                        action="store_false")
    parser.add_argument("--d23v2-asymmetry-virtual-length-mm", type=float, default=70.0)
    parser.add_argument("--d23v2-allow-legacy-fallback", dest="d23v2_allow_legacy_fallback",
                        action="store_true", default=False)
    parser.add_argument("--no-d23v2-allow-legacy-fallback", dest="d23v2_allow_legacy_fallback",
                        action="store_false")

    # D23 speed tuning copied from the upper-body team's proven principles:
    # parallel high-Z commands, one close command, faster press speed, shorter waits.
    parser.add_argument("--d23-fast-motion", dest="d23_fast_motion", action="store_true", default=True)
    parser.add_argument("--no-d23-fast-motion", dest="d23_fast_motion", action="store_false")
    parser.add_argument("--d23-parallel-high-z", dest="d23_parallel_high_z", action="store_true", default=True)
    parser.add_argument("--no-d23-parallel-high-z", dest="d23_parallel_high_z", action="store_false")
    parser.add_argument("--d23-motion-wait-scale", type=float, default=0.68)
    parser.add_argument("--d23-fast-min-stage-wait-s", type=float, default=0.32)
    parser.add_argument("--d23-fast-max-stage-wait-s", type=float, default=2.40)
    parser.add_argument("--d23-fast-close-repeat", type=int, default=2)
    parser.add_argument("--d23-high-z-wait-s", type=float, default=0.55)
    parser.add_argument("--d23-grip-pull-speed-min", type=float, default=0.28)
    parser.add_argument("--d23-hold-pull-speed-min", type=float, default=0.30)
    parser.add_argument("--d23-dual-speed-min", type=float, default=0.32)
    parser.add_argument("--d23-press-speed-min", type=float, default=0.36)

    parser.add_argument("--auto-infer-before-motion", dest="auto_infer_before_motion",
                        action="store_true", default=True)
    parser.add_argument("--no-auto-infer-before-motion", dest="auto_infer_before_motion",
                        action="store_false")
    parser.add_argument("--motion-preinfer-settle-s", type=float, default=0.35)

    # D17 pants-shape target and fully automatic repeat loop.
    parser.add_argument("--d17-shape-score-good", type=float, default=0.82)
    parser.add_argument("--d17-shape-hem-gap-ratio-min", type=float, default=0.58)
    parser.add_argument("--d17-shape-hem-gap-ratio-max", type=float, default=1.30)
    parser.add_argument("--d17-shape-leg-balance-min", type=float, default=0.84)
    parser.add_argument("--d17-shape-waist-center-offset-max", type=float, default=0.12)
    parser.add_argument("--d17-shape-waist-line-error-max", type=float, default=0.10)
    parser.add_argument("--d17-shape-crotch-axis-offset-max", type=float, default=0.18)
    parser.add_argument("--d17-shape-axis-to-waist-min", type=float, default=0.80)

    parser.add_argument("--d17-pose-pull-scale", type=float, default=1.0)
    parser.add_argument("--d17-pose-hem-pull-mm", type=float, default=130.0)
    parser.add_argument("--d17-pose-waist-pull-mm", type=float, default=115.0)
    parser.add_argument("--d17-pose-leg-pull-mm", type=float, default=145.0)
    parser.add_argument("--d17-pose-body-pull-mm", type=float, default=120.0)

    parser.add_argument("--d17-grip-close-repeat", type=int, default=2)
    parser.add_argument("--d17-grip-close-wait-s", type=float, default=0.30)
    parser.add_argument("--d17-grip-close-settle-s", type=float, default=0.55)
    parser.add_argument("--d17-contact-open-wait-s", type=float, default=0.20)

    parser.add_argument("--d17-auto-loop", dest="d17_auto_loop", action="store_true", default=True,
                        help="Automatically repeat shape/wrinkle actions after Homography is ready.")
    parser.add_argument("--no-d17-auto-loop", dest="d17_auto_loop", action="store_false")
    parser.add_argument("--d17-auto-confirm-once", dest="d17_auto_confirm_once",
                        action="store_true", default=True,
                        help="Ask once before starting the automatic physical loop.")
    parser.add_argument("--no-d17-auto-confirm-once", dest="d17_auto_confirm_once",
                        action="store_false")
    parser.add_argument("--d17-auto-start-delay-s", type=float, default=5.0)
    parser.add_argument("--d19-auto-cycle-settle-s", type=float, default=1.20)
    parser.add_argument("--d17-auto-failure-limit", type=int, default=3)
    parser.add_argument("--d17-auto-stall-limit", type=int, default=3)
    parser.add_argument("--d17-auto-max-cycles", type=int, default=0,
                        help="0 means no fixed cycle limit; stop by finish/stall/safety conditions.")
    parser.add_argument("--d17-shape-min-improvement", type=float, default=0.025)

    # Keep OpenCV live while robot commands and confirmation waits run in a worker.
    parser.add_argument("--async-motion", dest="async_motion", action="store_true", default=True)
    parser.add_argument("--no-async-motion", dest="async_motion", action="store_false")
    parser.add_argument("--auto-reinfer-after-motion", dest="auto_reinfer_after_motion", action="store_true", default=True)
    parser.add_argument("--no-auto-reinfer-after-motion", dest="auto_reinfer_after_motion", action="store_false")
    parser.add_argument("--motion-refresh-settle-s", type=float, default=0.9)

    # Short launch modes. All perception/model/tuning paths are already defaults.
    parser.add_argument("--physical-auto", action="store_true", default=False,
                        help="Short preset for actual automatic operation: send, no dry-run, no hover-only.")
    parser.add_argument("--hover-check", action="store_true", default=False,
                        help="Short preset for robot hover-only path verification.")

    parser.add_argument("--send", action="store_true", default=False)
    parser.add_argument("--no-send", dest="send", action="store_false")
    parser.add_argument("--dry-run", action="store_true", default=True)
    parser.add_argument("--no-dry-run", dest="dry_run", action="store_false")
    parser.add_argument("--hover-only", action="store_true", default=True)
    parser.add_argument("--no-hover-only", dest="hover_only", action="store_false")
    parser.add_argument("--enter-confirm", action="store_true", default=True)
    parser.add_argument("--no-enter-confirm", dest="enter_confirm", action="store_false")

    parser.add_argument("--arm1-port", default="/dev/roarm_1")
    parser.add_argument("--arm2-port", default="/dev/roarm_2")
    parser.add_argument("--move-command", type=int, choices=[104, 1041], default=104)

    parser.add_argument("--center-dead-half-width", type=float, default=40.0)
    parser.add_argument("--waist-ref-y", type=float, default=-390.0)
    parser.add_argument("--center-slit-x", type=float, default=247.0)
    parser.add_argument("--waist-y-tolerance-mm", type=float, default=25.0)
    parser.add_argument("--crotch-x-tolerance-mm", type=float, default=25.0)

    parser.add_argument("--hover-z", type=float, default=180.0)
    parser.add_argument("--approach-z", type=float, default=180.0)
    parser.add_argument("--grip-z", type=float, default=-115.0)
    parser.add_argument("--arm1-grip-z", type=float, default=-125.0)
    parser.add_argument("--arm2-grip-z", type=float, default=None)
    parser.add_argument("--lift-z", type=float, default=-85.0)
    parser.add_argument("--arm1-lift-z", type=float, default=-85.0)
    parser.add_argument("--arm2-lift-z", type=float, default=-85.0)
    parser.add_argument("--place-z", type=float, default=-105.0)
    parser.add_argument("--clear-z", type=float, default=180.0)
    parser.add_argument("--min-safe-contact-z", type=float, default=-140.0)
    parser.add_argument("--speed", type=float, default=0.12)
    parser.add_argument("--drag-speed", type=float, default=0.35)
    parser.add_argument("--grip-open", type=float, default=1.35)
    parser.add_argument("--grip-close", type=float, default=3.05)
    parser.add_argument("--grip-hold", type=float, default=3.15)
    parser.add_argument("--grip-down-wait", type=float, default=0.8)
    parser.add_argument("--grip-open-wait", type=float, default=1.0)
    parser.add_argument("--grip-contact-wait", type=float, default=0.15)
    parser.add_argument("--grip-close-repeat", type=int, default=1)
    parser.add_argument("--grip-close-wait", type=float, default=0.45)
    parser.add_argument("--grip-close-settle", type=float, default=0.45)
    parser.add_argument("--place-settle-wait", type=float, default=0.8)
    parser.add_argument("--place-open-wait", type=float, default=1.0)
    parser.add_argument("--after-action-settle-s", type=float, default=0.9)

    # Same direct RoArm standby pose as step_b34_v10.py.
    # D18: clear both robot arms from the camera view before every decision image.
    parser.add_argument("--d18-force-imaging-standby-before-infer",
                        dest="d18_force_imaging_standby_before_infer",
                        action="store_true", default=True)
    parser.add_argument("--no-d18-force-imaging-standby-before-infer",
                        dest="d18_force_imaging_standby_before_infer",
                        action="store_false")
    parser.add_argument("--d18-imaging-standby-wait-s", type=float, default=1.10,
                        help="Estimated wait for both arms to reach camera-clear standby.")
    parser.add_argument("--d18-imaging-frame-flush-s", type=float, default=0.25,
                        help="Extra live-frame settling time after standby.")
    parser.add_argument("--d18-block-infer-if-standby-fails",
                        dest="d18_block_infer_if_standby_fails",
                        action="store_true", default=True)
    parser.add_argument("--no-d18-block-infer-if-standby-fails",
                        dest="d18_block_infer_if_standby_fails",
                        action="store_false")

    parser.add_argument("--imaging-standby", action="store_true", default=True)
    parser.add_argument("--no-imaging-standby", dest="imaging_standby", action="store_false")
    parser.add_argument("--standby-after-motion", action="store_true", default=True)
    parser.add_argument("--no-standby-after-motion", dest="standby_after_motion", action="store_false")
    parser.add_argument("--standby-gripper-angle", type=float, default=1.57)
    parser.add_argument("--arm1-standby-roarm-x", type=float, default=-10.47886)
    parser.add_argument("--arm1-standby-roarm-y", type=float, default=227.544386)
    parser.add_argument("--arm1-standby-roarm-z", type=float, default=170.929303)
    parser.add_argument("--arm1-standby-roarm-t", type=float, default=2.124563)
    parser.add_argument("--arm2-standby-roarm-x", type=float, default=2.870034)
    parser.add_argument("--arm2-standby-roarm-y", type=float, default=-233.859636)
    parser.add_argument("--arm2-standby-roarm-z", type=float, default=102.23829)
    parser.add_argument("--arm2-standby-roarm-t", type=float, default=1.356039)

    # Apply short-mode defaults before parse so any explicit CLI option still wins.
    if "--physical-auto" in sys.argv:
        parser.set_defaults(
            send=True,
            dry_run=False,
            hover_only=False,
            enter_confirm=True,
            d17_auto_loop=True,
            d17_auto_confirm_once=True,
            d17_auto_start_delay_s=3.0,
            d17_auto_failure_limit=5,
            d17_auto_stall_limit=6,
            d17_auto_max_cycles=0,
            cam_auto_adjust=False,
            cam_auto_settle_s=0.20,
            standby_after_motion=False,
            auto_reinfer_after_motion=False,
            motion_preinfer_settle_s=0.12,
            wrinkle_after_settle_s=0.45,
            d19_auto_cycle_settle_s=0.30,
            d18_imaging_standby_wait_s=0.65,
            d18_imaging_frame_flush_s=0.15,
            d23_fast_motion=True,
            d23_pose_tta_fast_first=True,
        )
    elif "--hover-check" in sys.argv:
        parser.set_defaults(
            send=True,
            dry_run=False,
            hover_only=True,
            enter_confirm=True,
            d17_auto_loop=True,
            d17_auto_confirm_once=True,
            d17_auto_start_delay_s=3.0,
            cam_auto_adjust=False,
            cam_auto_settle_s=0.20,
            standby_after_motion=False,
            auto_reinfer_after_motion=False,
            motion_preinfer_settle_s=0.12,
            wrinkle_after_settle_s=0.45,
            d19_auto_cycle_settle_s=0.30,
            d18_imaging_standby_wait_s=0.65,
            d18_imaging_frame_flush_s=0.15,
            d23_fast_motion=True,
            d23_pose_tta_fast_first=True,
        )

    args = parser.parse_args()
    args.pose_tta_angles_list = parse_pose_tta_angles(args.angles)
    args.pose_tta_flip_modes_list = parse_pose_tta_flip_modes(args.flip_modes)
    print(f"[BUILD] {STEP_D23_BUILD}")
    print(
        f"[D23-SYMMETRY] required={args.d22_require_lower_leg_symmetry} "
        f"lowerIoU>={args.d22_lower_reflected_iou_min:.2f} "
        f"widthMed>={args.d22_lower_width_balance_median_min:.2f} "
        f"widthMin>={args.d22_lower_width_balance_min_min:.2f}"
    )
    print(
        f"[D23-HYBRID] enabled={args.d22_hybrid_wrinkle_policy} "
        f"gripTier>={args.d22_grip_min_tier} gripSeverity>={args.d22_grip_min_severity:.2f} "
        f"press={args.d22_press_sweep_min_mm:.0f}~{args.d22_press_sweep_max_mm:.0f}mm"
    )
    print(
        f"[D23-WAIST-VETO] enabled={args.d23_waist_structure_veto} "
        f"forward={args.d23_waist_structure_forward_mm:.0f}mm "
        f"parallel<={args.d23_waist_structure_parallel_deg:.0f}deg"
    )
    print(
        f"[D23-SPEED] fast_motion={args.d23_fast_motion} wait_scale={args.d23_motion_wait_scale:.2f} "
        f"parallel_high_z={args.d23_parallel_high_z} staged_tta={args.d23_pose_tta_fast_first}"
    )
    print(
        f"[D23V2-CENTER-OUT] enabled={args.d23v2_center_outward_policy} "
        f"minCos={args.d23v2_min_outward_cos:.2f} minGain={args.d23v2_min_radial_gain_mm:.0f}mm "
        f"closedPress={args.d23v2_press_gripper:.2f} severeOnlyGrip={args.d23v2_grip_require_all} "
        f"legacyFallback={args.d23v2_allow_legacy_fallback}"
    )
    if args.physical_auto:
        print("[D23-PRESET] physical-auto enabled")
    elif args.hover_check:
        print("[D23-PRESET] hover-check enabled")
    print(f"[D20-MODEL] segmentation={args.seg_model}")
    print(
        f"[D20-SHADOW] local-relative=True require_dark={args.d20_require_dark_shadow} "
        f"ignore_fine={args.d20_ignore_fine_wrinkles} "
        f"strong(area>={args.d20_strong_area_px:.0f}px,len>={args.d20_strong_length_mm:.0f}mm,"
        f"darkMean>={args.d20_strong_dark_mean:.0f}) "
        f"medium(area>={args.d20_medium_area_px:.0f}px,len>={args.d20_medium_length_mm:.0f}mm,"
        f"darkMean>={args.d20_medium_dark_mean:.0f})"
    )
    print(
        f"[D20-FILTER] waistband={args.d19_ignore_waistband_wrinkles} "
        f"crotch={args.d19_ignore_crotch_wrinkles} "
        f"min_blob={args.d19_action_min_blob_area_px:.0f}px "
        f"min_len={args.d19_action_min_major_len_mm:.0f}mm"
    )
    print(
        f"[D20] imaging_standby_before_infer="
        f"{args.d18_force_imaging_standby_before_infer} "
        f"mask_first_grip={args.d18_mask_first_grip} "
        f"mask_snap_required={args.d18_mask_snap_required}"
    )
    print(f"[POSE-TTA] enabled={args.pose_tta} angles={args.pose_tta_angles_list} flips={args.pose_tta_flip_modes_list}")
    print(
        f"[D23-GOOD-ENOUGH] enabled={args.d21v4_good_enough_finish} "
        f"actionTier>={args.d21v4_action_min_tier} T3<={args.d21v4_finish_max_t3_count} "
        f"T2<={args.d21v4_finish_max_t2_count} maxSeverity={args.d21v4_finish_max_severity:.2f} "
        f"actionRatio<={args.d21v4_finish_actionable_ratio:.3f} "
        f"T1=residual_allowed"
    )
    print(
        f"[D21] geometry_grip_pull={args.d21_geometry_grip_pull} "
        f"pull={args.d21_pull_min_mm:.0f}~{args.d21_pull_max_mm:.0f}mm "
        f"static_after={args.d21_static_required_actions} actions "
        f"fine_only=True maxTier={args.d21v3_static_max_tier} "
        f"maxSeverity={args.d21v3_static_max_severity:.2f} "
        f"stable_finish_after={args.d21_stable_required_actions} actions"
    )

    if args.demo:
        demo()
        return 0

    config = load_json_if_exists(args.config) or {"aruco": {"marker_board_mm": DEFAULT_MARKER_BOARD_MM}, "dual_roarm": {"split_board_x": 247.0}}
    config = load_affine_cache_into_config(config, args.calib_cache)
    cfg = make_safety_config_from_args(args, config)

    H = load_homography(args.hfile) if args.load_h else None
    if H is not None:
        print(f"[H] loaded: {args.hfile}")

    seg_model, pose_model = load_models(args)
    arms = setup_arms(args)
    cap = open_camera(args)
    configure_usb_camera(args)
    for _ in range(8):
        cap.read()
        time.sleep(0.03)
    auto_adjust_usb_camera(args, reason="startup")
    for _ in range(6):
        cap.read()
        time.sleep(0.03)
    if cfg.imaging_standby:
        move_arms_to_standby(arms, cfg, move_command=args.move_command, reason="startup imaging standby")

    dictionary_name = config.get("aruco", {}).get("dictionary", "DICT_4X4_50")
    detector = make_aruco_detector(get_dictionary(dictionary_name))
    required_ids = config.get("aruco", {}).get("required_ids", [0, 1, 2, 3])
    marker_board_mm = config.get("aruco", {}).get("marker_board_mm", DEFAULT_MARKER_BOARD_MM)

    latest_obs: Optional[BottomObservation] = None
    latest_plan: Optional[MotionPlan] = None
    latest_wrinkle_plan: Optional[WrinkleHeatmapPlan] = None
    latest_dual_wrinkle_plan: Optional[DualWrinkleStretchPlan] = None
    latest_live_frame: Optional[np.ndarray] = None
    selected_kpt_index = 1
    window = "BOTTOM_STAGE_1_4_D8"

    state_lock = threading.RLock()
    frame_lock = threading.RLock()
    inference_lock = threading.Lock()
    motion_lock = threading.RLock()
    motion_busy = False
    motion_name = ""
    perception_busy = False
    d14_finish_good_streak = 0
    d17_auto_enabled = bool(args.d17_auto_loop)
    d17_auto_confirmed = not (bool(args.d17_auto_confirm_once) and bool(cfg.enter_confirm))
    d17_auto_finished = False
    d17_auto_cycle_count = 0
    d21_shape_stable_action_streak = 0
    d21v4_good_enough_streak = 0
    d21_total_real_actions = 0
    d17_auto_failure_count = 0
    d17_auto_stall_count = 0
    d17_auto_next_at = time.time() + max(0.0, float(args.d17_auto_start_delay_s))
    d17_auto_last_status = "WAITING_FOR_H"
    worker_threads: List[threading.Thread] = []

    def get_latest_frame_copy() -> Optional[np.ndarray]:
        with frame_lock:
            return None if latest_live_frame is None else latest_live_frame.copy()

    def fresh_snapshot_before_motion(label: str) -> Optional[np.ndarray]:
        # D18 imaging discipline:
        # never capture a decision image while either robot arm may cover the pants.
        if bool(getattr(args, "d18_force_imaging_standby_before_infer", True)):
            if not bool(cfg.imaging_standby):
                msg = (
                    f"[D20-VISION-CLEAR] {label}: imaging_standby=False; "
                    "decision image is unsafe"
                )
                print(msg)
                if bool(getattr(args, "d18_block_infer_if_standby_fails", True)):
                    return None
            elif cfg.send and not cfg.dry_run:
                print(
                    f"[D20-VISION-CLEAR] {label}: moving ARM2/ARM1 "
                    "to camera-clear standby"
                )
                standby_ok = move_arms_to_standby(
                    arms,
                    cfg,
                    move_command=args.move_command,
                    reason=f"D18 before {label} inference",
                )
                if (
                    not standby_ok
                    and bool(getattr(args, "d18_block_infer_if_standby_fails", True))
                ):
                    print(
                        f"[D20-VISION-CLEAR] {label}: standby command failed; "
                        "inference blocked"
                    )
                    return None
                time.sleep(
                    max(
                        0.0,
                        float(
                            getattr(
                                args,
                                "d18_imaging_standby_wait_s",
                                1.10,
                            )
                        ),
                    )
                )
            else:
                print(
                    f"[D20-VISION-CLEAR] {label}: preview mode; "
                    "standby command not sent"
                )

            time.sleep(
                max(
                    0.0,
                    float(
                        getattr(args, "d18_imaging_frame_flush_s", 0.25)
                    ),
                )
            )

        if bool(getattr(args, "auto_infer_before_motion", True)):
            auto_adjust_usb_camera(args, reason=f"{label}-pre")
            time.sleep(
                max(
                    0.0,
                    float(getattr(args, "motion_preinfer_settle_s", 0.35)),
                )
            )

        snap = get_latest_frame_copy()
        if snap is None:
            print(f"[D20-AUTO-INFER] {label}: no fresh live frame")
            return None
        print(
            f"[D20-AUTO-INFER] {label}: camera-clear fresh frame captured"
        )
        return snap

    def publish_observation(obs: BottomObservation, plan: Optional[MotionPlan], source_frame: np.ndarray) -> None:
        nonlocal latest_obs, latest_plan, latest_wrinkle_plan
        heat = None
        if obs is not None and obs.mask is not None:
            try:
                heat = build_d21_wrinkle_heatmap(source_frame, obs, H, args)
            except Exception as e:
                print(f"[HEATMAP-ERROR] {repr(e)}")
        with state_lock:
            latest_obs = obs
            latest_plan = plan
            latest_wrinkle_plan = heat
        if heat is not None:
            print(f"[HEATMAP] {heat.summary}")

    def print_observation(obs: BottomObservation) -> None:
        print(f"[OBS] valid={obs.valid} reason={obs.reason}")
        if obs.pose is not None:
            po = obs.pose
            print(f"[POSE] waist_y={po.waist_center[1]:.1f} crotch_x={po.crotch[0]:.1f} "
                  f"axis={po.pants_axis_angle_deg:+.1f} len={po.pants_axis_len_mm:.1f} "
                  f"hem_gap={po.hem_gap_mm:.1f}")
            print(f"[POSE-TTA] angle={po.tta_angle_deg:+.1f} flip={po.tta_flip_mode} "
                  f"score={po.tta_score:.2f} state={po.tta_state} tested={po.tta_tested_count} "
                  f"reason={po.tta_reason}")
        if obs.mask is not None:
            m = obs.mask
            print(f"[MASK] area={m.area_px:.0f}px center=({m.center_board[0]:.1f},{m.center_board[1]:.1f}) "
                  f"solidity={m.solidity:.2f}")

    def infer_and_publish(snapshot: np.ndarray, H_snapshot: np.ndarray, label: str,
                          plan_builder=None) -> Tuple[BottomObservation, Optional[MotionPlan]]:
        nonlocal perception_busy
        with state_lock:
            perception_busy = True
        try:
            with inference_lock:
                obs = infer_bottom_observation(seg_model, pose_model, snapshot, H_snapshot, args, cfg)
            plan = None if plan_builder is None else plan_builder(obs)
            publish_observation(obs, plan, snapshot)
            print(f"[INFER] {label}")
            print_observation(obs)
            if plan is not None:
                print_plan(plan)
            return obs, plan
        finally:
            with state_lock:
                perception_busy = False

    def start_inference_task(label: str, snapshot: np.ndarray, H_snapshot: np.ndarray) -> bool:
        nonlocal perception_busy
        with state_lock:
            if perception_busy:
                print(f"[INFER-BUSY] {label} blocked: another inference is running")
                return False
            perception_busy = True
        def worker():
            try:
                infer_and_publish(snapshot, H_snapshot, label)
            except Exception as e:
                print(f"[INFER-ERROR] {label}: {repr(e)}")
        th = threading.Thread(target=worker, name=f"infer-{label}", daemon=True)
        worker_threads.append(th)
        th.start()
        return True

    def run_plan_action(label: str, snapshot: Optional[np.ndarray], H_snapshot: np.ndarray,
                        builder) -> bool:
        if bool(getattr(args, "auto_infer_before_motion", True)) or snapshot is None:
            snapshot = fresh_snapshot_before_motion(label)
        if snapshot is None:
            return False
        obs, plan = infer_and_publish(snapshot, H_snapshot, f"{label}-pre", builder)
        if plan is None:
            print(f"[MOTION] {label}: no plan")
            return False
        return execute_motion_plan(plan, arms, cfg, move_command=args.move_command)

    def run_full_async(H_snapshot: np.ndarray) -> bool:
        stages = [
            ("pick", lambda obs: build_pick_place_plan(obs, H_snapshot, config, cfg)),
            ("waist", lambda obs: build_waist_align_plan(obs, H_snapshot, config, cfg)),
            ("crotch", lambda obs: build_crotch_align_plan(obs, H_snapshot, config, cfg)),
        ]
        for stage_name, builder in stages:
            snapshot = fresh_snapshot_before_motion(f"FULL-{stage_name}")
            if snapshot is None:
                print(f"[FULL] {stage_name}: no live frame")
                return False
            _obs, plan = infer_and_publish(snapshot, H_snapshot, f"FULL-{stage_name}", builder)
            if plan is None:
                return False
            if plan.action == "DONE":
                continue
            if not plan.ok:
                print(f"[FULL] {stage_name} blocked: {plan.reason}")
                return False
            if not execute_motion_plan(plan, arms, cfg, move_command=args.move_command):
                return False
            time.sleep(float(args.after_action_settle_s))
        print("[FULL] Stage 1~4 sequence finished.")
        return True

    def run_dual_wrinkle_cycle(H_snapshot: np.ndarray,
                               auto_mode: bool = False,
                               skip_confirm: bool = False) -> str:
        nonlocal latest_dual_wrinkle_plan, latest_obs, latest_wrinkle_plan
        nonlocal d14_finish_good_streak, d17_auto_stall_count
        nonlocal d21_shape_stable_action_streak, d21_total_real_actions
        nonlocal d21v4_good_enough_streak

        before_frame = fresh_snapshot_before_motion("D17-AUTO" if auto_mode else "D17-X")
        if before_frame is None:
            return "FAILED"

        with inference_lock:
            before_obs = infer_bottom_observation(
                seg_model, pose_model, before_frame, H_snapshot, args, cfg
            )
        before_heat = build_d21_wrinkle_heatmap(
            before_frame, before_obs, H_snapshot, args
        )
        before_wrinkle = _dual_plan_metrics(before_heat, before_obs)
        before_finish = _d20_finish_report(before_heat, before_obs, args, H_snapshot)
        before_shape = before_finish.get("shape", {})
        before_symmetry = before_shape.get("symmetry", {})

        print("[D23V2-PRE] staged-TTA + waistband-veto + center-outward motion gate")
        print_observation(before_obs)
        print("[D20-FINISH-REPORT]", json.dumps(before_finish, ensure_ascii=False, default=str))
        _d20_print_wrinkle_ranking(before_heat)
        _d21_print_geometry_ranking(before_heat)
        _d22_print_symmetry(before_symmetry, "BEFORE")

        good_enough_before = _d21v4_good_enough_finish_report(
            before_heat, before_obs, H_snapshot, args
        )
        print("[D23-GOOD-ENOUGH]", json.dumps(good_enough_before, ensure_ascii=False, default=str))

        wrinkle_good = bool(before_finish.get("heat_good", False))
        shape_good = bool(before_shape.get("shape_good", False))

        # Completion requires the E22-style multi-check gate plus pants shape,
        # after waistband/crotch/tiny wrinkle removal.
        if bool(before_finish.get("good", False)):
            d14_finish_good_streak += 1
            required = max(2, int(getattr(args, "finish_confirm_cycles", 2)))
            print(
                f"[D20-FINISH-CHECK] good={d14_finish_good_streak}/{required} "
                f"summary={before_finish.get('summary', '')}"
            )
            with state_lock:
                latest_obs = before_obs
                latest_plan = None
                latest_wrinkle_plan = before_heat
                latest_dual_wrinkle_plan = None
            if d14_finish_good_streak >= required:
                print("[D20-FINISH] FINISHED_CONFIRMED: practical actionable-wrinkle gate passed")
                _d20_print_finish_banner(
                    before_finish,
                    required,
                    d17_auto_cycle_count,
                )
                return "FINISHED"
            print("[D20-FINISH] VERIFY_AGAIN: one more camera-clear confirmation")
            return "VERIFY"

        d14_finish_good_streak = 0

        if bool(good_enough_before.get("good",False)):
            d21v4_good_enough_streak += 1
            required=max(2,int(getattr(args,"d21v4_finish_confirm_cycles",2)))
            print(
                f"[D23-GOOD-ENOUGH-CHECK] {d21v4_good_enough_streak}/{required} "
                f"{good_enough_before.get('summary','')}"
            )
            with state_lock:
                latest_obs=before_obs
                latest_plan=None
                latest_wrinkle_plan=before_heat
                latest_dual_wrinkle_plan=None
            if d21v4_good_enough_streak>=required:
                _d21v4_print_good_enough_banner(
                    good_enough_before,required,d17_auto_cycle_count
                )
                return "FINISHED"
            print("[D23-GOOD-ENOUGH] VERIFY_AGAIN: 잔주름 허용 종료를 한 번 더 확인합니다.")
            return "VERIFY"
        d21v4_good_enough_streak=0

        # D22: lower-leg silhouette is a hard shape condition.  First try the
        # existing pose-shape correction; if sparse pose points miss the folded
        # leg, use the worst lower-leg mask band to pull the narrow side outward.
        before_plan = None
        if not shape_good:
            # D23_v2 handles a narrow/folded leg with a closed center-out press
            # before considering any pose-shape grip operation.
            if (bool(getattr(args, "d23v2_shape_asymmetry_press_first", True))
                    and not bool(before_symmetry.get("good", False))):
                before_plan = _d22_build_lower_leg_asymmetry_plan(
                    before_obs, H_snapshot, config, cfg, args,
                    before_symmetry, before_shape,
                )
            if before_plan is None:
                before_plan = _d17_build_pose_shape_plan(
                    before_obs, H_snapshot, config, cfg, args, before_shape
                )
            if before_plan is not None:
                print(
                    f"[D23V2-POLICY] SHAPE selected: {before_plan.action} "
                    f"shape={before_shape.get('shape_score', 0.0):.3f} "
                    f"lowerIoU={float(before_symmetry.get('lower_reflected_iou', 0.0)):.3f}"
                )

        # If the pants silhouette is acceptable, select the manipulation mode
        # from wrinkle geometry and severity. Strong LINE/CURVE -> grip-pull;
        # moderate or irregular -> low-Z press-sweep.
        if before_plan is None and bool(getattr(args, "d22_hybrid_wrinkle_policy", True)):
            before_plan = build_d22_hybrid_wrinkle_plan(
                before_obs, before_heat, H_snapshot, config, cfg, args
            )
            if before_plan is not None:
                print(
                    f"[D23V2-POLICY] HYBRID selected: {before_plan.action} "
                    f"mode={before_plan.metrics.get('d22_motion_mode', 'NA')}"
                )

        if before_plan is None:
            if bool(getattr(args, "d23v2_allow_legacy_fallback", False)):
                before_plan = build_dual_wrinkle_stretch_plan(
                    before_obs, before_heat, H_snapshot, config, cfg, args
                )
                print(f"[D23V2-POLICY] LEGACY FALLBACK selected: {before_plan.action}")
            else:
                before_plan = DualWrinkleStretchPlan(
                    False,
                    "D23_v2 safe center-outward plan 없음; inward-capable legacy fallback disabled",
                )
                print("[D23V2-POLICY] NO_ACTION: safe center-outward candidate unavailable")

        if before_plan is not None and before_plan.ok:
            outward_ok, outward_report = _d23v2_validate_center_outward_plan(
                before_plan, before_obs, args
            )
            before_plan.metrics["d23v2_center_outward_validation"] = outward_report
            if not outward_ok:
                print("[D23V2-INWARD-BLOCK]", json.dumps(outward_report, ensure_ascii=False, default=str))
                before_plan = DualWrinkleStretchPlan(
                    False,
                    "D23_v2 final guard blocked an inward or insufficiently outward motion",
                    safety_warnings=[json.dumps(outward_report, ensure_ascii=False, default=str)],
                )
            else:
                print("[D23V2-OUTWARD-OK]", json.dumps(outward_report, ensure_ascii=False, default=str))

        with state_lock:
            latest_obs = before_obs
            latest_plan = None
            latest_wrinkle_plan = before_heat
            latest_dual_wrinkle_plan = before_plan

        print_dual_wrinkle_plan(before_plan)
        if not before_plan.ok:
            return "FAILED"

        executed = execute_dual_wrinkle_stretch_plan(
            before_plan, arms, cfg,
            move_command=args.move_command,
            args=args,
            skip_confirm=skip_confirm,
        )
        if not executed:
            return "FAILED"
        if cfg.dry_run or not cfg.send or cfg.hover_only:
            print("[D17] preview/hover-only complete; automatic preview stops after one cycle")
            return "PREVIEW"

        time.sleep(max(
            max(0.0, float(args.wrinkle_after_settle_s)),
            max(0.0, float(args.d19_auto_cycle_settle_s)),
        ))
        after_frame = fresh_snapshot_before_motion("D17-POST")
        if after_frame is None:
            return "FAILED"

        with inference_lock:
            after_obs = infer_bottom_observation(
                seg_model, pose_model, after_frame, H_snapshot, args, cfg
            )
        after_heat = build_d21_wrinkle_heatmap(
            after_frame, after_obs, H_snapshot, args
        )
        after_shape = _d17_pose_shape_metrics(after_obs, args)
        evaluation = evaluate_wrinkle_change(
            before_heat, after_heat, before_obs, after_obs, args
        )

        shape_delta = float(after_shape.get("shape_score", 0.0)) - float(
            before_shape.get("shape_score", 0.0)
        )
        wrinkle_improved = bool(
            str(evaluation.get("state", "")).startswith("IMPROVED")
            or float(evaluation.get("ratio_relative_improvement", 0.0)) >=
               float(getattr(args, "wrinkle_min_ratio_improvement", 0.05))
            or float(evaluation.get("blob_relative_improvement", 0.0)) >=
               float(getattr(args, "wrinkle_min_blob_improvement", 0.10))
        )
        shape_improved = shape_delta >= float(args.d17_shape_min_improvement)
        overall_improved = bool(wrinkle_improved or shape_improved)

        if overall_improved:
            d17_auto_stall_count = 0
        else:
            d17_auto_stall_count += 1

        after_wrinkle = _dual_plan_metrics(after_heat, after_obs)
        after_finish = _d20_finish_report(after_heat, after_obs, args, H_snapshot)
        after_both_good = bool(after_finish.get("good", False))
        if after_both_good:
            d14_finish_good_streak = 1
        else:
            d14_finish_good_streak = 0

        d21_total_real_actions += 1
        static_result = _d21_record_action_result(
            before_plan, after_heat, after_obs, H_snapshot, args
        )
        shape_stability = _d21_shape_stability_report(before_obs, after_obs, H_snapshot, args)
        newly_static_tid = (
            int(static_result.get("track_id", -1))
            if bool(static_result.get("static", False))
            else -1
        )
        remaining_wrinkle_gate = _d21v3_remaining_wrinkle_gate(
            after_heat,
            args,
            newly_static_track_id=newly_static_tid,
        )
        good_enough_after = _d21v4_good_enough_finish_report(
            after_heat, after_obs, H_snapshot, args
        )
        stable_and_safe_to_finish = bool(
            shape_stability.get("stable", False)
            and good_enough_after.get("good", False)
        )
        if stable_and_safe_to_finish:
            d21_shape_stable_action_streak += 1
        else:
            d21_shape_stable_action_streak = 0
        _d22_print_symmetry(shape_stability.get("symmetry", {}), "AFTER")
        print(
            f"[D23-GOOD-ENOUGH-AFTER] {good_enough_after.get('summary','')}"
        )
        print(
            "[D21_V3-FINISH-GATE] "
            f"shape_stable={bool(shape_stability.get('stable',False))} "
            f"remaining_good={bool(remaining_wrinkle_gate.get('good',False))} "
            f"nonfine={int(remaining_wrinkle_gate.get('nonfine_count',0))} "
            f"fine={int(remaining_wrinkle_gate.get('fine_count',0))} "
            f"nonfine_tracks={remaining_wrinkle_gate.get('nonfine_tracks',[])}"
        )
        print(
            f"[D21-STABILITY] stable={d21_shape_stable_action_streak}/"
            f"{max(2,int(args.d21_stable_required_actions))} "
            f"report={json.dumps(shape_stability,ensure_ascii=False,default=str)}"
        )

        combined = {
            "action": before_plan.action,
            "before_shape": before_shape,
            "after_shape": after_shape,
            "shape_score_delta": shape_delta,
            "wrinkle_evaluation": evaluation,
            "overall_improved": overall_improved,
            "stall_count": d17_auto_stall_count,
            "both_good_once": after_both_good,
            "after_finish": after_finish,
            "d21_static_result": static_result,
            "d21_shape_stability": shape_stability,
            "d21v3_remaining_wrinkle_gate": remaining_wrinkle_gate,
            "d21v4_good_enough_after": good_enough_after,
            "d21_shape_stable_action_streak": d21_shape_stable_action_streak,
            "d21_total_real_actions": d21_total_real_actions,
        }
        print("[D20-EVAL]", json.dumps(combined, ensure_ascii=False, default=str))

        if before_plan.candidate_index >= 0:
            _remember_b34_feedback(before_plan, evaluation, args)

        with state_lock:
            latest_obs = after_obs
            latest_plan = None
            latest_wrinkle_plan = after_heat
            latest_dual_wrinkle_plan = None

        stable_required = max(2, int(args.d21_stable_required_actions))
        if d21_shape_stable_action_streak >= stable_required:
            _d21v4_print_good_enough_banner(
                good_enough_after, stable_required, d17_auto_cycle_count
            )
            return "FINISHED"

        if d17_auto_stall_count >= max(1, int(args.d17_auto_stall_limit)):
            print("[D20-AUTO] PAUSED: repeated cycles made no meaningful improvement")
            return "STALL"
        return "ACTION_OK"


    def run_d17_auto_once(H_snapshot: np.ndarray) -> str:
        nonlocal d17_auto_enabled, d17_auto_confirmed, d17_auto_finished
        nonlocal d17_auto_cycle_count, d17_auto_failure_count
        nonlocal d17_auto_next_at, d17_auto_last_status

        if not d17_auto_confirmed:
            if not confirm_if_needed(cfg, "D17 자동 반복 시작"):
                d17_auto_enabled = False
                d17_auto_last_status = "CANCELLED"
                print("[D20-AUTO] cancelled before first automatic action")
                return "CANCELLED"
            d17_auto_confirmed = True

        d17_auto_cycle_count += 1
        status = run_dual_wrinkle_cycle(
            H_snapshot,
            auto_mode=True,
            skip_confirm=True,
        )
        d17_auto_last_status = status

        if status == "FINISHED":
            d17_auto_enabled = False
            d17_auto_finished = True
            print(f"[D20-AUTO] completed after {d17_auto_cycle_count} cycles")
        elif status == "PREVIEW":
            d17_auto_enabled = False
            print("[D20-AUTO] preview mode: automatic repetition paused after one cycle")
        elif status in ("FAILED", "CANCELLED"):
            d17_auto_failure_count += 1
            if d17_auto_failure_count >= max(1, int(args.d17_auto_failure_limit)):
                d17_auto_enabled = False
                print("[D20-AUTO] PAUSED: repeated planning/execution failures")
        elif status == "STALL":
            d17_auto_enabled = False
        else:
            d17_auto_failure_count = 0

        max_cycles = int(args.d17_auto_max_cycles)
        if max_cycles > 0 and d17_auto_cycle_count >= max_cycles:
            d17_auto_enabled = False
            print(f"[D20-AUTO] stopped at configured max cycles={max_cycles}")

        d17_auto_next_at = time.time() + max(0.2, float(args.d19_auto_cycle_settle_s))
        return status


    def start_motion_task(label: str, fn) -> bool:
        nonlocal motion_busy, motion_name
        with motion_lock:
            if motion_busy:
                print(f"[MOTION-BUSY] {label} blocked: {motion_name} is running")
                return False
            motion_busy = True
            motion_name = label

        def worker():
            nonlocal motion_busy, motion_name
            try:
                print(f"[MOTION] {label} started; OpenCV camera loop remains live")
                fn()
                if bool(args.auto_reinfer_after_motion):
                    time.sleep(max(0.0, float(args.motion_refresh_settle_s)))
                    snap = get_latest_frame_copy()
                    with state_lock:
                        H_after = None if H is None else np.asarray(H, dtype=np.float32).copy()
                    if snap is not None and H_after is not None:
                        try:
                            infer_and_publish(snap, H_after, f"{label}-post")
                        except Exception as e:
                            print(f"[POST-INFER-ERROR] {label}: {repr(e)}")
            except Exception as e:
                print(f"[MOTION-ERROR] {label}: {repr(e)}")
            finally:
                with motion_lock:
                    motion_busy = False
                    motion_name = ""
                print(f"[MOTION] {label} worker finished; this does NOT mean the pants are fully spread")

        if bool(args.async_motion):
            th = threading.Thread(target=worker, name=f"motion-{label}", daemon=True)
            worker_threads.append(th)
            th.start()
        else:
            worker()
        return True

    print("\n[KEYS]")
    print("  L : lock/save Homography from visible ArUco markers")
    print("  I : async infer rotation/flip-TTA pose + mask + refresh wrinkle heatmap")
    print("  H : toggle wrinkle heatmap overlay")
    print("  0~7 : select bottom pose keypoint for grasp test")
    print("  G/P/W/C/F : plan and run in worker thread; camera remains live")
    print("  X : run one D17 cycle manually (auto loop normally runs without X)")
    print("  A : pause/resume D19 automatic repeat loop")
    print("  S : async move both RoArms to standby")
    print("  Q/ESC : quit (blocked while robot motion is running)\n")
    print(f"[KPT] selected={selected_kpt_index}:{BOTTOM_POSE_KPT_NAMES[selected_kpt_index]}")
    print(f"[SAFETY] send={cfg.send} dry_run={cfg.dry_run} hover_only={cfg.hover_only} "
          f"enter_confirm={cfg.enter_confirm} async_motion={args.async_motion}")
    print(
        f"[D20-AUTO] enabled={d17_auto_enabled} start_delay={args.d17_auto_start_delay_s:.1f}s "
        f"confirm_once={args.d17_auto_confirm_once} max_cycles={args.d17_auto_max_cycles}"
    )

    try:
        if args.once:
            frame = None
            ok = False
            for _ in range(max(1, int(args.warmup_frames))):
                ok, frame = cap.read()
                if not ok:
                    time.sleep(0.03)
            if not ok or frame is None:
                print("[ONCE] camera read failed")
                return 2

            if H is None and args.auto_lock:
                corners, ids, _ = detect_markers(frame, detector)
                H_live, centers = compute_homography(corners, ids, marker_board_mm, required_ids)
                if H_live is not None:
                    H = H_live
                    print("[ONCE] Homography auto-locked from current frame")
                    if args.save_h_on_lock:
                        save_homography(args.hfile, H)

            if H is None:
                print("[ONCE] Homography not locked")
                return 3
            latest_obs = infer_bottom_observation(seg_model, pose_model, frame, H, args, cfg)
            latest_wrinkle_plan = build_d21_wrinkle_heatmap(frame, latest_obs, H, args) if latest_obs.mask is not None else None
            canvas = draw_bottom_overlay(frame, H, latest_obs, cfg, None, latest_wrinkle_plan, args, False, "")
            print_observation(latest_obs)
            if args.debug_image:
                cv2.imwrite(args.debug_image, canvas)
                print(f"[ONCE] overlay saved: {args.debug_image}")
            return 0 if latest_obs.valid else 4

        while True:
            ok, frame = cap.read()
            if not ok:
                print("[CAM] read failed")
                break

            raw_frame = frame.copy()
            with frame_lock:
                latest_live_frame = raw_frame.copy()

            corners, ids, _ = detect_markers(frame, detector)
            if ids is not None:
                H_live, centers = compute_homography(corners, ids, marker_board_mm, required_ids)
                if H is None and args.auto_lock and H_live is not None:
                    H = H_live
                    print("[H] auto-locked from live ArUco")
                cv2.aruco.drawDetectedMarkers(frame, corners, ids)
            else:
                H_live = None

            # D17 starts and repeats automatically once Homography is available.
            if d17_auto_enabled and H is not None and time.time() >= d17_auto_next_at:
                with motion_lock:
                    d17_busy_now = bool(motion_busy)
                with state_lock:
                    d17_infer_now = bool(perception_busy)
                if not d17_busy_now and not d17_infer_now:
                    H_auto = np.asarray(H, dtype=np.float32).copy()
                    next_cycle_no = d17_auto_cycle_count + 1
                    start_motion_task(
                        f"d23-auto-cycle-{next_cycle_no}",
                        lambda Hs=H_auto: run_d17_auto_once(Hs),
                    )

            with state_lock:
                obs_view = latest_obs
                plan_view = latest_plan
                heat_view = latest_wrinkle_plan
                dual_view = latest_dual_wrinkle_plan
                infer_view = perception_busy
            with motion_lock:
                motion_view = motion_busy
                motion_label = motion_name

            canvas = draw_bottom_overlay(
                frame, H, obs_view, cfg, plan_view, heat_view, args,
                motion_busy=motion_view, motion_name=motion_label,
            )
            canvas = draw_dual_wrinkle_plan_overlay(canvas, H, dual_view)
            canvas = _d22_draw_symmetry_overlay(canvas, H, obs_view)
            if d17_auto_finished:
                cv2.rectangle(canvas, (15, 15), (760, 92), (0, 0, 0), -1)
                cv2.rectangle(canvas, (15, 15), (760, 92), (0, 255, 0), 3)
                cv2.putText(
                    canvas,
                    "PANTS SPREAD COMPLETE - AUTO STOPPED",
                    (32, 64),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.88,
                    (0, 255, 0),
                    2,
                )
            if infer_view:
                cv2.putText(canvas, "PERCEPTION RUNNING IN BACKGROUND", (25, 102),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 0, 0), 4)
                cv2.putText(canvas, "PERCEPTION RUNNING IN BACKGROUND", (25, 102),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 0), 2)

            if not args.no_window:
                cv2.imshow(window, canvas)
                key = cv2.waitKey(1) & 0xFF
            else:
                key = 255
                time.sleep(0.01)
            terminal_key = 255 if TERMINAL_CONFIRM_ACTIVE.is_set() else read_terminal_key()
            if key == 255 and terminal_key != 255:
                key = terminal_key

            if key in (ord("q"), 27):
                with motion_lock:
                    busy_now = motion_busy
                    running = motion_name
                if busy_now:
                    print(f"[QUIT-BLOCKED] robot motion is running: {running}")
                    continue
                break

            if ord("0") <= key <= ord("7"):
                selected_kpt_index = int(chr(key))
                print(f"[KPT] selected={selected_kpt_index}:{BOTTOM_POSE_KPT_NAMES[selected_kpt_index]}")
                with state_lock:
                    latest_plan = None
                continue

            if key == ord("h"):
                args.wrinkle_heatmap_draw = not bool(args.wrinkle_heatmap_draw)
                print(f"[HEATMAP] draw={args.wrinkle_heatmap_draw}")
                continue

            if key == ord("s"):
                start_motion_task("standby", lambda: move_arms_to_standby(
                    arms, cfg, move_command=args.move_command, reason="manual S"))
                continue

            if key == ord("l"):
                with motion_lock:
                    if motion_busy:
                        print("[H] lock blocked during robot motion")
                        continue
                if H_live is None:
                    print("[H] cannot lock: need ArUco ID0~ID3 visible")
                else:
                    H = np.asarray(H_live, dtype=np.float32).copy()
                    if args.save_h_on_lock:
                        save_homography(args.hfile, H)
                    print("[H] locked")
                continue

            if key == ord("i"):
                if H is None:
                    print("[I] Homography not locked")
                    continue
                with motion_lock:
                    if motion_busy:
                        print("[I] blocked during robot motion; post-motion re-infer is automatic")
                        continue
                with state_lock:
                    latest_dual_wrinkle_plan = None
                start_inference_task("manual-I", raw_frame.copy(), np.asarray(H, dtype=np.float32).copy())
                continue

            if key == ord("a"):
                d17_auto_enabled = not bool(d17_auto_enabled)
                d17_auto_finished = False if d17_auto_enabled else d17_auto_finished
                d17_auto_next_at = time.time() + 1.0
                print(
                    f"[D20-AUTO] {'RESUMED' if d17_auto_enabled else 'PAUSED'} "
                    f"cycle={d17_auto_cycle_count} status={d17_auto_last_status}"
                )
                continue

            if key == ord("x"):
                if H is None:
                    print("[X] Homography not locked")
                    continue
                H_snap = np.asarray(H, dtype=np.float32).copy()
                start_motion_task(
                    "d17-manual-cycle",
                    lambda: run_dual_wrinkle_cycle(H_snap, auto_mode=False, skip_confirm=False),
                )
                continue

            if key in (ord("g"), ord("p"), ord("w"), ord("c"), ord("f")):
                if H is None:
                    print(f"[{chr(key).upper()}] Homography not locked")
                    continue
                snap = raw_frame.copy()
                H_snap = np.asarray(H, dtype=np.float32).copy()
                kpt_snap = int(selected_kpt_index)

                if key == ord("g"):
                    builder = lambda obs, ki=kpt_snap: build_pose_keypoint_grasp_plan(obs, H_snap, config, cfg, ki)
                    start_motion_task("keypoint-grasp", lambda: run_plan_action("G", None, H_snap, builder))
                elif key == ord("p"):
                    builder = lambda obs: build_pick_place_plan(obs, H_snap, config, cfg)
                    start_motion_task("pick-place", lambda: run_plan_action("P", None, H_snap, builder))
                elif key == ord("w"):
                    builder = lambda obs: build_waist_align_plan(obs, H_snap, config, cfg)
                    start_motion_task("waist-align", lambda: run_plan_action("W", None, H_snap, builder))
                elif key == ord("c"):
                    builder = lambda obs: build_crotch_align_plan(obs, H_snap, config, cfg)
                    start_motion_task("crotch-align", lambda: run_plan_action("C", None, H_snap, builder))
                else:
                    start_motion_task("full-stage-1to4", lambda: run_full_async(H_snap))
                continue
    finally:
        # Q is blocked during motion, but still wait briefly for any inference-only worker.
        for th in list(worker_threads):
            if th.is_alive() and not th.name.startswith("motion-"):
                th.join(timeout=2.0)
        cap.release()
        if not args.no_window:
            cv2.destroyAllWindows()
        for arm in arms.values():
            arm.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
