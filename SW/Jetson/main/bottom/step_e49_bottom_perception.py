#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
step_e49_bottom_perception.py

E49 combined bottom-garment perception with empty-board and strict ArUco-board ROI sanitation.

Extracted from step_e43_bottom_keypoint_wiggle_test.py and intentionally limited to:
  - segmentation and garment-mask selection
  - glare/specular suppression
  - wrinkle Heatmap and local-shadow analysis
  - bottom pose inference, TTA, consensus, and temporal stabilization
  - mask topology, PCA/oriented-rectangle geometry
  - crotch concavity and waistband evidence
  - mask-primary/fallback landmark reconstruction
  - final pose geometry scoring
  - construction of BottomObservation

Robot action classification, path planning, serial execution, camera runtime, and UI are
not included. The public boundary of this module is infer_bottom_observation(), which
returns perception/geometry results without selecting or executing an action.

E49 preserves the E45 closed-crotch pipeline and the E47 empty folding-board
baseline stage. It additionally constrains segmentation masks to the quadrilateral
formed by the configured ArUco marker centers, excludes marker patches, and performs
connected-component/Pose-consistency sanitation. Strict ROI failure never falls back
to the unsanitized YOLO mask, so unsafe off-board detections cannot drive motion.
"""

from __future__ import annotations

import math
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

CODE_VERSION = "E49"
CODE_BUILD_ID = "E49-BOTTOM-PERCEPTION-BOARD-ROI-V1-20260724"
SOURCE_BASELINE = "step_e47_bottom_perception.py"


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
BOTTOM_POSE_KPT_SHORT_NAMES = [
    "WAIST_L", "WAIST_C", "WAIST_R", "CROTCH",
    "L_HEM_OUT", "L_HEM_IN", "R_HEM_IN", "R_HEM_OUT",
]
HFLIP_REMAP = np.asarray([2, 1, 0, 3, 7, 6, 5, 4], dtype=np.int32)

class BottomActionDecision:
    action: str
    reason: str
    confidence: float = 0.0
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
    inference_status: str = ""
    geometry_reliable: bool = False
    geometry_score: float = 0.0
    recovery_required: bool = False
    geometry_metrics: Dict[str, Any] = field(default_factory=dict)
    crotch_state: str = "UNKNOWN"
    crotch_confidence: float = 0.0
    polarity_metrics: Dict[str, Any] = field(default_factory=dict)
    pre_spread_required: bool = False


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
    empty_baseline_info: Dict[str, Any] = field(default_factory=dict)
    board_roi_info: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BottomObservation:
    pose: Optional[BottomPoseBoard]
    mask: Optional[BottomMaskBoard]
    fused_center_board: Optional[np.ndarray]
    valid: bool
    reason: str
    frame: Optional[np.ndarray] = None
    wrinkle: Optional[Dict[str, Any]] = None
    action_decision: Optional[BottomActionDecision] = None
    shape_debug: Optional[Dict[str, Any]] = None
    glare: Optional[Dict[str, Any]] = None
    crotch_state: str = "UNKNOWN"
    pants_polarity: Optional[Dict[str, Any]] = None
    pre_spread_required: bool = False
    empty_baseline: Optional[Dict[str, Any]] = None
    board_roi: Optional[Dict[str, Any]] = None

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


def parse_class_names(raw: Any) -> List[str]:
    if raw is None:
        return []
    if isinstance(raw, (list, tuple)):
        items = raw
    else:
        items = str(raw).split(",")
    out = []
    for item in items:
        name = str(item).strip()
        if name and name not in out:
            out.append(name)
    return out



def _empty_baseline_board_shift_px(H_current: np.ndarray, H_saved: np.ndarray,
                                   board_points: Sequence[Sequence[float]]) -> Optional[Dict[str, float]]:
    distances: List[float] = []
    try:
        for raw in board_points:
            bx, by = float(raw[0]), float(raw[1])
            p_now = board_to_pixel(H_current, bx, by)
            p_old = board_to_pixel(H_saved, bx, by)
            if p_now is None or p_old is None:
                continue
            distances.append(float(np.linalg.norm(np.asarray(p_now) - np.asarray(p_old))))
    except Exception:
        return None
    if not distances:
        return None
    return {
        "mean_px": float(np.mean(distances)),
        "max_px": float(np.max(distances)),
    }


def _board_roi_from_reference(H: np.ndarray, board_points: Sequence[Sequence[float]],
                              image_shape) -> np.ndarray:
    h, w = image_shape[:2]
    roi = np.zeros((h, w), dtype=np.uint8)
    pixels: List[List[int]] = []
    for raw in board_points:
        p = board_to_pixel(H, float(raw[0]), float(raw[1]))
        if p is not None:
            pixels.append([int(round(p[0])), int(round(p[1]))])
    if len(pixels) >= 3:
        hull = cv2.convexHull(np.asarray(pixels, dtype=np.int32).reshape(-1, 1, 2))
        cv2.fillConvexPoly(roi, hull, 255)
    else:
        roi[:] = 255
    return roi



def _runtime_board_marker_map(args) -> Tuple[Dict[str, Sequence[float]], List[int], str]:
    """Return Config-provided marker geometry without using a legacy board fallback."""
    marker_map = getattr(args, "_board_marker_map", None) if args is not None else None
    required_ids = getattr(args, "_board_required_ids", None) if args is not None else None
    if not isinstance(marker_map, dict) or len(marker_map) < 3:
        return {}, [], "marker geometry unavailable"
    ids: List[int] = []
    for value in (required_ids or [0, 1, 2, 3]):
        try:
            marker_id = int(value)
        except Exception:
            continue
        if str(marker_id) in marker_map or marker_id in marker_map:
            ids.append(marker_id)
    if len(ids) < 3:
        return marker_map, ids, "fewer than three configured markers"
    return marker_map, ids, str(getattr(args, "_board_roi_source", "folding-board-config"))


def build_board_roi_polygon_px(image_shape, H: Optional[np.ndarray], args) -> Tuple[Optional[np.ndarray], List[Tuple[int, int]], Dict[str, Any]]:
    """Project configured marker centers into the camera frame and build their convex hull."""
    h, w = image_shape[:2]
    marker_map, required_ids, source = _runtime_board_marker_map(args)
    info: Dict[str, Any] = {
        "enabled": bool(getattr(args, "board_roi", True)),
        "source": source,
        "required_ids": list(required_ids),
        "marker_points_px": [],
        "polygon_px": None,
        "valid": False,
        "reason": "not evaluated",
    }
    if not info["enabled"]:
        info.update({"valid": True, "reason": "disabled"})
        return None, [], info
    if H is None:
        info["reason"] = "homography missing"
        return None, [], info
    if len(required_ids) < 3:
        info["reason"] = source
        return None, [], info
    points: List[Tuple[int, int]] = []
    used_ids: List[int] = []
    for marker_id in required_ids:
        raw = marker_map.get(str(marker_id), marker_map.get(marker_id))
        if raw is None or len(raw) < 2:
            continue
        q = board_to_pixel(np.asarray(H, dtype=np.float32), float(raw[0]), float(raw[1]))
        if q is None or not np.all(np.isfinite(q)):
            continue
        x = int(np.clip(round(float(q[0])), -2 * w, 3 * w))
        y = int(np.clip(round(float(q[1])), -2 * h, 3 * h))
        points.append((x, y))
        used_ids.append(int(marker_id))
    info["marker_points_px"] = [[int(x), int(y)] for x, y in points]
    info["used_ids"] = used_ids
    if len(points) < 3:
        info["reason"] = f"only {len(points)} marker projections available"
        return None, points, info
    hull = cv2.convexHull(np.asarray(points, dtype=np.int32).reshape(-1, 1, 2)).reshape(-1, 2)
    info.update({
        "valid": True,
        "reason": "OK",
        "polygon_px": hull.astype(int).tolist(),
    })
    return hull.astype(np.int32), points, info


def build_board_valid_mask(image_shape, H: Optional[np.ndarray], args) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Build the strict ArUco-board valid region and remove marker patches/borders."""
    h, w = image_shape[:2]
    enabled = bool(getattr(args, "board_roi", True))
    strict = bool(getattr(args, "board_roi_strict", True))
    border = max(0, int(getattr(args, "board_frame_border_exclude_px", 8)))
    polygon, marker_points, info = build_board_roi_polygon_px(image_shape, H, args)
    info.update({"strict": strict, "border_exclude_px": border})
    if not enabled:
        valid = np.full((h, w), 255, dtype=np.uint8)
    elif polygon is None or len(polygon) < 3:
        valid = np.zeros((h, w), dtype=np.uint8) if strict else np.full((h, w), 255, dtype=np.uint8)
        info["reason"] = f"{info.get('reason', 'ROI unavailable')}; {'strict block' if strict else 'non-strict full-frame fallback'}"
        info["valid"] = not strict
    else:
        valid = np.zeros((h, w), dtype=np.uint8)
        cv2.fillConvexPoly(valid, polygon, 255)
        expand = max(0, int(getattr(args, "board_roi_expand_px", 0)))
        if expand > 0:
            k = 2 * expand + 1
            valid = cv2.dilate(valid, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
        info["expand_px"] = expand

    marker_radius = max(0, int(getattr(args, "board_marker_exclusion_radius_px", 30)))
    if marker_radius > 0:
        for point in marker_points:
            cv2.circle(valid, tuple(map(int, point)), marker_radius, 0, -1)
    if border > 0:
        valid[:border, :] = 0
        valid[-border:, :] = 0
        valid[:, :border] = 0
        valid[:, -border:] = 0
    info.update({
        "marker_exclusion_radius_px": marker_radius,
        "valid_pixel_count": int(cv2.countNonZero(valid)),
        "valid_frame_ratio": float(cv2.countNonZero(valid)) / float(max(1, h * w)),
    })
    return valid, info


def _mask_iou_u8(a: np.ndarray, b: np.ndarray) -> float:
    if a is None or b is None or a.shape != b.shape:
        return 0.0
    aa, bb = np.asarray(a) > 0, np.asarray(b) > 0
    union = int(np.count_nonzero(aa | bb))
    return float(np.count_nonzero(aa & bb)) / float(union) if union > 0 else 0.0


def sanitize_board_mask_u8(mask_u8: np.ndarray, H: Optional[np.ndarray], image_shape, args,
                           pose: Optional[BottomPoseBoard] = None,
                           require_pose: bool = False,
                           select_component: bool = True) -> Tuple[Optional[np.ndarray], Dict[str, Any]]:
    """Clip a candidate to the ArUco board and select one safe connected component."""
    raw = ((np.asarray(mask_u8, dtype=np.uint8) > 0).astype(np.uint8) * 255)
    raw_count = int(cv2.countNonZero(raw))
    info: Dict[str, Any] = {
        "enabled": bool(getattr(args, "board_roi", True)),
        "strict": bool(getattr(args, "board_roi_strict", True)),
        "valid": False,
        "accepted": False,
        "reason": "raw mask missing" if raw_count <= 0 else "not evaluated",
        "raw_area_px": raw_count,
        "require_pose": bool(require_pose),
        "select_component": bool(select_component),
    }
    if raw_count <= 0:
        return None, info
    if not info["enabled"]:
        info.update({"valid": True, "accepted": True, "reason": "board ROI disabled", "selected_area_px": raw_count})
        return raw, info

    valid_mask, roi_info = build_board_valid_mask(image_shape, H, args)
    info["roi"] = roi_info
    if int(cv2.countNonZero(valid_mask)) <= 0:
        info["reason"] = f"board ROI unavailable: {roi_info.get('reason', '')}"
        if info["strict"]:
            return None, info
        info.update({"valid": True, "accepted": True, "reason": "non-strict raw fallback", "selected_area_px": raw_count})
        return raw, info

    mask = cv2.bitwise_and(raw, valid_mask)
    board_count = int(cv2.countNonZero(mask))
    info["board_clipped_area_px"] = board_count
    info["board_keep_ratio"] = float(board_count) / float(max(1, raw_count))
    if board_count <= 0:
        info["reason"] = "mask has no pixels inside ArUco board ROI"
        return None, info

    def _morph_kernel(value: Any) -> np.ndarray:
        k = max(1, int(value))
        if k % 2 == 0:
            k += 1
        return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))

    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, _morph_kernel(getattr(args, "board_mask_open_px", 3)))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, _morph_kernel(getattr(args, "board_mask_close_px", 5)))
    h, w = image_shape[:2]
    num, labels, stats, _ = cv2.connectedComponentsWithStats((mask > 0).astype(np.uint8), 8)
    if num <= 1:
        info["reason"] = "no component after board/marker exclusion"
        return None, info

    pose_points: List[Tuple[float, float]] = []
    if pose is not None and isinstance(getattr(pose, "keypoints_px", None), dict):
        pose_points = [(float(x), float(y)) for x, y in pose.keypoints_px.values()]
    pose_hull_area = 0.0
    if len(pose_points) >= 3:
        pose_hull = cv2.convexHull(np.asarray(pose_points, dtype=np.float32).reshape(-1, 1, 2))
        pose_hull_area = float(cv2.contourArea(pose_hull))

    min_component = max(300, int(getattr(args, "board_mask_min_component_px", 1200)))
    min_inside = max(0, int(getattr(args, "board_mask_min_pose_inside", 5)))
    pose_gate_active = bool(require_pose and len(pose_points) >= max(1, min_inside))
    info["pose_point_count"] = int(len(pose_points))
    info["pose_gate_active"] = pose_gate_active
    max_frame_ratio = float(np.clip(getattr(args, "board_mask_max_frame_ratio", 0.72), 0.01, 1.0))
    max_pose_hull_ratio = max(1.0, float(getattr(args, "board_mask_max_pose_hull_ratio", 6.0)))
    best_label = None
    accepted_labels: List[int] = []
    best_score = -1e30
    best_diag: Dict[str, Any] = {}
    reject_counts = {"small": 0, "frame": 0, "pose": 0, "hull": 0}
    for label in range(1, int(num)):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < min_component:
            reject_counts["small"] += 1
            continue
        frame_ratio = float(area) / float(max(1, h * w))
        if frame_ratio > max_frame_ratio:
            reject_counts["frame"] += 1
            continue
        comp = labels == label
        inside = 0
        for x, y in pose_points:
            xi, yi = int(round(x)), int(round(y))
            if 0 <= xi < w and 0 <= yi < h and bool(comp[yi, xi]):
                inside += 1
        if pose_gate_active and inside < min_inside:
            reject_counts["pose"] += 1
            continue
        hull_ratio = float(area) / max(1.0, pose_hull_area) if pose_hull_area > 1.0 else 1.0
        if pose_hull_area > 1.0 and hull_ratio > max_pose_hull_ratio:
            reject_counts["hull"] += 1
            continue
        accepted_labels.append(int(label))
        x = int(stats[label, cv2.CC_STAT_LEFT])
        y = int(stats[label, cv2.CC_STAT_TOP])
        ww = int(stats[label, cv2.CC_STAT_WIDTH])
        hh = int(stats[label, cv2.CC_STAT_HEIGHT])
        border_touch = int(x <= 2) + int(y <= 2) + int(x + ww >= w - 2) + int(y + hh >= h - 2)
        score = area - border_touch * 500000.0
        if pose_points:
            score += inside * 1000000.0 - max(0.0, hull_ratio - 3.0) * 30000.0
        if score > best_score:
            best_score = score
            best_label = label
            best_diag = {
                "pose_inside": int(inside), "area_px": int(area),
                "frame_ratio": frame_ratio, "pose_hull_ratio": hull_ratio,
                "border_touch": int(border_touch),
            }
    info["rejected_components"] = reject_counts
    info["component_count"] = int(num - 1)
    if not accepted_labels or best_label is None:
        info["reason"] = "no safe board/Pose-consistent component"
        return None, info
    if select_component:
        selected = ((labels == int(best_label)).astype(np.uint8) * 255)
        info.update(best_diag)
        reason = "OK selected best board/Pose component"
    else:
        selected = (np.isin(labels, np.asarray(accepted_labels, dtype=np.int32)).astype(np.uint8) * 255)
        info.update({
            "pose_inside": 0,
            "area_px": int(cv2.countNonZero(selected)),
            "frame_ratio": float(cv2.countNonZero(selected)) / float(max(1, h * w)),
            "pose_hull_ratio": 1.0,
            "border_touch": 0,
            "retained_component_count": int(len(accepted_labels)),
        })
        reason = "OK retained pre-Pose board components"
    selected_count = int(cv2.countNonZero(selected))
    info.update({
        "valid": True,
        "accepted": True,
        "reason": reason,
        "selected_area_px": selected_count,
        "selected_keep_ratio_from_raw": float(selected_count) / float(max(1, raw_count)),
    })
    return selected, info


def rebuild_bottom_mask_from_u8(mask_u8: np.ndarray, H: np.ndarray, raw: BottomMaskBoard,
                                board_info: Dict[str, Any]) -> Optional[BottomMaskBoard]:
    mask = ((np.asarray(mask_u8, dtype=np.uint8) > 0).astype(np.uint8) * 255)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)
    area = float(cv2.contourArea(contour))
    if area <= 10.0:
        return None
    clean = np.zeros_like(mask)
    cv2.drawContours(clean, [contour], -1, 255, -1)
    moments = cv2.moments(contour)
    if abs(moments["m00"]) > 1e-6:
        center_px = np.asarray([moments["m10"] / moments["m00"], moments["m01"] / moments["m00"]], dtype=np.float32)
    else:
        center_px = contour.reshape(-1, 2).astype(np.float32).mean(axis=0)
    bx, by = pixel_to_board(H, float(center_px[0]), float(center_px[1]))
    hull = cv2.convexHull(contour)
    hull_area = float(cv2.contourArea(hull))
    return BottomMaskBoard(
        mask_u8=clean,
        contour=contour,
        area_px=area,
        center_px=center_px,
        center_board=np.asarray([bx, by], dtype=np.float32),
        class_name=str(raw.class_name),
        confidence=float(raw.confidence),
        solidity=float(area / hull_area if hull_area > 1.0 else 0.0),
        empty_baseline_info=dict(getattr(raw, "empty_baseline_info", {}) or {}),
        board_roi_info=dict(board_info or {}),
    )


def evaluate_empty_baseline_candidate(frame: np.ndarray, candidate_mask: np.ndarray,
                                      H: Optional[np.ndarray], args) -> Tuple[Optional[np.ndarray], Dict[str, Any]]:
    """Conservatively validate/refine one YOLO mask against an empty-board reference.

    The reference never creates a garment mask by itself. It may reject a YOLO
    candidate that is nearly identical to the empty folding board, or refine an
    accepted candidate when the changed support remains sufficiently large.
    """
    info: Dict[str, Any] = {
        "enabled": bool(getattr(args, "empty_baseline_veto", True)),
        "available": False,
        "compatible": False,
        "active": False,
        "accepted": True,
        "rejected": False,
        "refined": False,
        "reason": "disabled",
        "mean_ab_diff": 0.0,
        "changed_ab_ratio": 0.0,
        "combined_ratio": 0.0,
        "refined_keep_ratio": 1.0,
        "illumination_l_shift": 0.0,
        "homography_mean_shift_px": None,
        "homography_max_shift_px": None,
    }
    mask = np.asarray(candidate_mask, dtype=np.uint8)
    if not info["enabled"]:
        return mask, info

    baseline = getattr(args, "_empty_baseline_bgr", None)
    baseline_h = getattr(args, "_empty_baseline_h", None)
    board_points = getattr(args, "_empty_baseline_board_points", None)
    if baseline is None:
        info["reason"] = "baseline unavailable"
        if bool(getattr(args, "empty_baseline_require", False)):
            info.update({"accepted": False, "rejected": True})
            return None, info
        return mask, info
    info["available"] = True
    baseline = np.asarray(baseline, dtype=np.uint8)
    if baseline.shape != frame.shape:
        info["reason"] = f"resolution mismatch baseline={baseline.shape} current={frame.shape}"
        if bool(getattr(args, "empty_baseline_require", False)):
            info.update({"accepted": False, "rejected": True})
            return None, info
        return mask, info
    if H is None or baseline_h is None or not board_points:
        info["reason"] = "baseline homography metadata unavailable"
        if bool(getattr(args, "empty_baseline_require", False)):
            info.update({"accepted": False, "rejected": True})
            return None, info
        return mask, info

    shift = _empty_baseline_board_shift_px(
        np.asarray(H, dtype=np.float32), np.asarray(baseline_h, dtype=np.float32), board_points,
    )
    if shift is None:
        info["reason"] = "homography compatibility check failed"
        if bool(getattr(args, "empty_baseline_require", False)):
            info.update({"accepted": False, "rejected": True})
            return None, info
        return mask, info
    info["homography_mean_shift_px"] = float(shift["mean_px"])
    info["homography_max_shift_px"] = float(shift["max_px"])
    max_h_shift = max(0.5, float(getattr(args, "empty_baseline_max_h_shift_px", 8.0)))
    if float(shift["max_px"]) > max_h_shift:
        info["reason"] = (
            f"homography moved max={float(shift['max_px']):.1f}px > {max_h_shift:.1f}px; recapture E baseline"
        )
        if bool(getattr(args, "empty_baseline_require", False)):
            info.update({"accepted": False, "rejected": True})
            return None, info
        return mask, info
    info["compatible"] = True

    candidate = mask > 0
    candidate_count = int(np.count_nonzero(candidate))
    if candidate_count <= 0:
        info.update({"accepted": False, "rejected": True, "reason": "empty YOLO mask"})
        return None, info

    base_lab = cv2.cvtColor(baseline, cv2.COLOR_BGR2LAB).astype(np.float32)
    curr_lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB).astype(np.float32)
    board_roi = _board_roi_from_reference(np.asarray(H, dtype=np.float32), board_points, frame.shape)

    # Estimate a small global illumination shift from board pixels outside the
    # candidate. This preserves dark/gray clothing contrast while compensating
    # for minor lamp or exposure drift.
    exclusion_px = max(3, int(getattr(args, "empty_baseline_illumination_exclusion_px", 17)))
    exclusion_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (exclusion_px | 1, exclusion_px | 1))
    excluded = cv2.dilate(mask, exclusion_kernel, iterations=1) > 0
    illumination_support = (board_roi > 0) & (~excluded)
    l_shift = 0.0
    if int(np.count_nonzero(illumination_support)) >= 500:
        l_delta = curr_lab[:, :, 0] - base_lab[:, :, 0]
        l_shift = float(np.median(l_delta[illumination_support]))
        limit = max(0.0, float(getattr(args, "empty_baseline_max_l_shift", 20.0)))
        l_shift = float(np.clip(l_shift, -limit, limit))
    info["illumination_l_shift"] = l_shift

    d_l = np.abs((curr_lab[:, :, 0] - l_shift) - base_lab[:, :, 0])
    d_a = curr_lab[:, :, 1] - base_lab[:, :, 1]
    d_b = curr_lab[:, :, 2] - base_lab[:, :, 2]
    d_ab = np.sqrt(d_a * d_a + d_b * d_b)
    l_weight = max(0.0, float(getattr(args, "empty_baseline_l_weight", 0.25)))
    combined = l_weight * d_l + d_ab

    ab_threshold = max(0.0, float(getattr(args, "empty_change_ab_threshold", 10.0)))
    combined_threshold = max(0.0, float(getattr(args, "empty_change_combined_threshold", 14.0)))
    changed_ab = d_ab >= ab_threshold
    changed_combined = combined >= combined_threshold
    changed = (changed_ab | changed_combined).astype(np.uint8) * 255
    changed = cv2.bitwise_and(changed, board_roi)
    morph_px = max(1, int(getattr(args, "empty_change_morph_px", 5))) | 1
    morph_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_px, morph_px))
    changed = cv2.morphologyEx(changed, cv2.MORPH_OPEN, morph_kernel, iterations=1)
    changed = cv2.morphologyEx(changed, cv2.MORPH_CLOSE, morph_kernel, iterations=2)

    info["mean_ab_diff"] = float(np.mean(d_ab[candidate]))
    info["changed_ab_ratio"] = float(np.count_nonzero(changed_ab & candidate)) / float(candidate_count)
    info["combined_ratio"] = float(np.count_nonzero(changed_combined & candidate)) / float(candidate_count)
    min_mean_ab = max(0.0, float(getattr(args, "empty_change_min_mean_ab_diff", 4.5)))
    min_changed_ratio = max(0.0, float(getattr(args, "empty_change_min_ratio", 0.035)))
    min_combined_ratio = max(0.0, float(getattr(args, "empty_change_min_combined_ratio", 0.035)))

    background_like = (
        info["mean_ab_diff"] < min_mean_ab
        and info["changed_ab_ratio"] < min_changed_ratio
        and info["combined_ratio"] < min_combined_ratio
    )
    info["active"] = True
    if background_like:
        info.update({
            "accepted": False,
            "rejected": True,
            "reason": (
                "background-like YOLO candidate "
                f"meanAB={info['mean_ab_diff']:.2f} "
                f"abRatio={info['changed_ab_ratio']:.3f} "
                f"combinedRatio={info['combined_ratio']:.3f}"
            ),
        })
        return None, info

    info["reason"] = "accepted by empty-board change evidence"
    if not bool(getattr(args, "empty_refine_mask", True)):
        return mask, info

    dilate_px = max(1, int(getattr(args, "empty_refine_dilate_px", 11))) | 1
    dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_px, dilate_px))
    support = cv2.dilate(changed, dilate_kernel, iterations=1)
    refined = cv2.bitwise_and(mask, support)
    refined = cv2.morphologyEx(refined, cv2.MORPH_CLOSE, morph_kernel, iterations=1)
    keep_ratio = float(np.count_nonzero(refined)) / float(candidate_count)
    info["refined_keep_ratio"] = keep_ratio
    min_keep = float(np.clip(getattr(args, "empty_refine_min_keep_ratio", 0.25), 0.0, 1.0))
    min_area = max(1, int(getattr(args, "empty_refine_min_area_px", 1200)))
    if keep_ratio >= min_keep and int(np.count_nonzero(refined)) >= min_area:
        info.update({"refined": True, "reason": f"accepted and refined keep={keep_ratio:.3f}"})
        return refined, info

    info["reason"] = f"accepted; refinement skipped keep={keep_ratio:.3f} < {min_keep:.3f}"
    return mask, info


def infer_bottoms_mask(seg_model, frame: np.ndarray, H: Optional[np.ndarray], imgsz: int, conf: float,
                       target_class_names: Sequence[str] = ("bottoms",), args=None) -> Tuple[Optional[BottomMaskBoard], str]:
    if args is not None:
        args._empty_baseline_last_info = {
            "enabled": bool(getattr(args, "empty_baseline_veto", True)),
            "available": getattr(args, "_empty_baseline_bgr", None) is not None,
            "active": False,
            "accepted": False,
            "reason": "segmentation not evaluated",
        }
        args._board_roi_last_info = {
            "enabled": bool(getattr(args, "board_roi", True)),
            "valid": False,
            "accepted": False,
            "reason": "segmentation not evaluated",
        }
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

    targets = parse_class_names(target_class_names) or ["bottoms"]
    allow_all = "*" in targets or "all" in targets
    target_set = set(targets)
    target_rank = {name: idx for idx, name in enumerate(targets)}
    names = getattr(result, "names", {}) or {}
    best = None
    best_rank = 10**6
    best_score = -1.0
    best_reason = f"no configured clothing class mask: {','.join(targets)}"
    for i in range(len(result.boxes)):
        cls_id = int(result.boxes.cls[i].item())
        cls_name = str(names.get(cls_id, cls_id))
        cls_id_name = str(cls_id)
        if not allow_all and cls_name not in target_set and cls_id_name not in target_set:
            continue
        c = float(result.boxes.conf[i].item())
        mask = result_mask_to_u8(result, i, frame.shape)
        if mask is None:
            best_reason = f"{cls_name} has no mask"
            continue
        empty_info: Dict[str, Any] = {}
        board_info: Dict[str, Any] = {}
        if args is not None:
            mask, empty_info = evaluate_empty_baseline_candidate(frame, mask, H, args)
            args._empty_baseline_last_info = dict(empty_info)
            if mask is None:
                best_reason = f"{cls_name} rejected by empty baseline: {empty_info.get('reason', 'background-like')}"
                continue
            mask, board_info = sanitize_board_mask_u8(
                mask, H, frame.shape, args, pose=None, require_pose=False, select_component=False,
            )
            args._board_roi_last_info = dict(board_info)
            if mask is None:
                best_reason = f"{cls_name} rejected by board ROI: {board_info.get('reason', 'outside board')}"
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
            empty_baseline_info=dict(empty_info),
            board_roi_info=dict(board_info),
        )
        rank = target_rank.get(cls_name, target_rank.get(cls_id_name, len(targets)))
        score = area * max(c, 0.05) * max(solidity, 0.20)
        if best is None or rank < best_rank or (rank == best_rank and score > best_score):
            best = item
            best_rank = rank
            best_score = score
    if best is None:
        return None, best_reason
    if best_rank == 0:
        return best, "OK"
    return best, f"OK fallback clothing class={best.class_name}"


def suppress_specular_reflections(
        frame: np.ndarray, garment_mask: Optional[np.ndarray], args,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Repair compact local highlights before pose/Heatmap inference.

    A bright table or an ArUco white cell is not automatically glare. A pixel
    must be bright relative to its own neighborhood, and connected components
    that are too large are rejected. When a garment mask is available, only
    highlights inside that mask are repaired.
    """
    info: Dict[str, Any] = {
        "enabled": bool(getattr(args, "glare_suppression", True)),
        "applied": False,
        "pixel_count": 0,
        "ratio": 0.0,
        "component_count": 0,
        "source": "garment" if garment_mask is not None else "frame",
        "mask_u8": None,
    }
    if frame is None or not info["enabled"]:
        return frame, info

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    value = hsv[:, :, 2].astype(np.float32)
    saturation = hsv[:, :, 1].astype(np.float32)
    blur_k = odd_kernel_size(getattr(args, "glare_local_blur_ksize", 31), 9)
    local_base = cv2.GaussianBlur(value, (blur_k, blur_k), 0)
    local_rise = value - local_base

    value_min = float(getattr(args, "glare_value_min", 218.0))
    rise_min = float(getattr(args, "glare_local_rise_min", 24.0))
    saturation_max = float(getattr(args, "glare_saturation_max", 125.0))
    hard_value = float(getattr(args, "glare_hard_value", 250.0))
    candidate = (
        (value >= value_min)
        & (local_rise >= rise_min)
        & ((saturation <= saturation_max) | (value >= hard_value))
    )

    h, w = frame.shape[:2]
    if garment_mask is not None and np.asarray(garment_mask).shape[:2] == (h, w):
        support = np.asarray(garment_mask, dtype=np.uint8) > 0
    else:
        support = np.zeros((h, w), dtype=bool)
        margin_x = max(1, int(round(0.06 * w)))
        margin_y = max(1, int(round(0.06 * h)))
        support[margin_y:h - margin_y, margin_x:w - margin_x] = True
    candidate &= support

    raw = candidate.astype(np.uint8) * 255
    raw = cv2.morphologyEx(
        raw, cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
        iterations=1,
    )
    count, labels, stats, _ = cv2.connectedComponentsWithStats(raw, connectivity=8)
    filtered = np.zeros_like(raw)
    support_area = max(1, int(np.count_nonzero(support)))
    min_area = max(2, int(getattr(args, "glare_component_min_area_px", 6)))
    max_area = max(
        min_area,
        int(round(support_area * float(getattr(
            args, "glare_component_max_area_ratio", 0.035,
        )))),
    )
    kept = 0
    for label in range(1, int(count)):
        area = int(stats[label, cv2.CC_STAT_AREA])
        width = int(stats[label, cv2.CC_STAT_WIDTH])
        height = int(stats[label, cv2.CC_STAT_HEIGHT])
        if area < min_area or area > max_area:
            continue
        if width > int(0.45 * w) or height > int(0.45 * h):
            continue
        filtered[labels == label] = 255
        kept += 1

    pixel_count = int(cv2.countNonZero(filtered))
    ratio = float(pixel_count) / float(support_area)
    info.update({
        "pixel_count": pixel_count,
        "ratio": ratio,
        "component_count": int(kept),
        "mask_u8": filtered,
    })
    min_ratio = max(0.0, float(getattr(args, "glare_min_apply_ratio", 0.00008)))
    if pixel_count == 0 or ratio < min_ratio:
        return frame, info

    dilate_px = max(0, int(getattr(args, "glare_inpaint_dilate_px", 2)))
    repair_mask = filtered
    if dilate_px > 0:
        k = 2 * dilate_px + 1
        repair_mask = cv2.dilate(
            repair_mask,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)),
            iterations=1,
        )
        repair_mask[~support] = 0
    radius = max(1.0, float(getattr(args, "glare_inpaint_radius_px", 4.0)))
    corrected = cv2.inpaint(frame, repair_mask, radius, cv2.INPAINT_TELEA)
    info["applied"] = True
    info["repair_mask_u8"] = repair_mask
    return corrected, info


def odd_kernel_size(value: Any, minimum: int = 3) -> int:
    k = max(int(value), int(minimum))
    if k % 2 == 0:
        k += 1
    return k


def normalize_masked_u8(values: np.ndarray, mask_u8: np.ndarray,
                        low_percentile: float = 2.0, high_percentile: float = 98.0,
                        min_range: float = 1e-6) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    valid = mask_u8 > 0
    out = np.zeros(arr.shape[:2], dtype=np.uint8)
    if not np.any(valid):
        return out
    vals = arr[valid]
    lo = float(np.percentile(vals, low_percentile))
    hi = float(np.percentile(vals, high_percentile))
    if not np.isfinite(lo) or not np.isfinite(hi):
        return out
    span = max(float(min_range), hi - lo)
    scaled = (arr - lo) * (255.0 / span)
    out = np.clip(scaled, 0, 255).astype(np.uint8)
    out[~valid] = 0
    return out


def build_local_shadow_map(frame: np.ndarray, inner_mask: np.ndarray, args) -> Dict[str, Any]:
    """Measure locally dark fold shadows without requiring a globally bright garment."""
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    luminance = lab[:, :, 0].astype(np.float32)
    blur_k = odd_kernel_size(getattr(args, "wrinkle_shadow_blur_ksize", 51), 9)
    local_base = cv2.GaussianBlur(luminance, (blur_k, blur_k), 0)
    raw = np.maximum(local_base - luminance, 0.0)
    raw[inner_mask == 0] = 0.0

    values = raw[inner_mask > 0]
    if values.size:
        p90 = float(np.percentile(values, 90.0))
        p98 = float(np.percentile(values, 98.0))
        p995 = float(np.percentile(values, 99.5))
    else:
        p90 = p98 = p995 = 0.0
    min_scale = max(1.0, float(getattr(args, "wrinkle_shadow_min_scale_raw", 8.0)))
    scale = max(min_scale, p995)
    normalized = np.clip(raw * (255.0 / scale), 0.0, 255.0).astype(np.uint8)
    normalized[inner_mask == 0] = 0
    support_threshold = max(
        float(getattr(args, "wrinkle_shadow_support_raw_min", 3.0)),
        p90,
    )
    return {
        "raw": raw,
        "normalized": normalized,
        "scale_raw": float(scale),
        "p90_raw": p90,
        "p98_raw": p98,
        "p995_raw": p995,
        "support_raw_threshold": float(support_threshold),
    }


def classify_wrinkle_geometry(area_px: float, mask_area_px: float,
                              major_length_px: float, minor_length_px: float,
                              edge_distance_px: float, shadow_mean: float,
                              shadow_max: float, shadow_support: float,
                              args) -> Dict[str, Any]:
    """Classify action geometry while keeping weak/fine candidates for finish checks."""
    linearity = float(major_length_px / max(1.0, minor_length_px))
    area_ratio = float(area_px / max(1.0, mask_area_px))
    edge_limit = float(getattr(args, "wrinkle_type_edge_distance_px", 15.0))
    long_linearity = float(getattr(args, "wrinkle_type_long_linearity", 2.60))
    short_linearity = float(getattr(args, "wrinkle_type_short_linearity", 1.65))
    long_length = float(getattr(args, "wrinkle_type_long_length_px", 65.0))
    short_length = float(getattr(args, "wrinkle_type_short_length_px", 32.0))

    if edge_distance_px <= edge_limit:
        wrinkle_type = "EDGE"
        pull_scale = float(getattr(args, "wrinkle_type_edge_pull_scale", 0.72))
    elif linearity >= long_linearity and major_length_px >= long_length:
        wrinkle_type = "LONG_LINEAR"
        pull_scale = float(getattr(args, "wrinkle_type_long_pull_scale", 1.15))
    elif linearity >= short_linearity and major_length_px >= short_length:
        wrinkle_type = "SHORT_LINEAR"
        pull_scale = float(getattr(args, "wrinkle_type_short_pull_scale", 0.78))
    else:
        wrinkle_type = "BROAD_ROUND"
        pull_scale = float(getattr(args, "wrinkle_type_round_pull_scale", 0.62))

    strong_size = bool(
        area_px >= float(getattr(args, "wrinkle_tier_strong_area_px", 220.0))
        or area_ratio >= float(getattr(args, "wrinkle_tier_strong_area_ratio", 0.0015))
    )
    strong_length = bool(major_length_px >= float(getattr(args, "wrinkle_tier_strong_length_px", 72.0)))
    medium_size = bool(
        area_px >= float(getattr(args, "wrinkle_tier_medium_area_px", 110.0))
        or area_ratio >= float(getattr(args, "wrinkle_tier_medium_area_ratio", 0.0008))
    )
    shadow_supported = bool(
        (shadow_mean >= float(getattr(args, "wrinkle_tier_shadow_mean", 22.0))
         or shadow_max >= float(getattr(args, "wrinkle_tier_shadow_max", 70.0)))
        and shadow_support >= float(getattr(args, "wrinkle_tier_shadow_support", 0.06))
    )
    if strong_size and strong_length:
        tier, tier_name = 3, "LARGE_LONG"
    elif strong_size or strong_length:
        tier, tier_name = 2, "LARGE_OR_LONG"
    else:
        tier, tier_name = 1, "FINE_OR_LOCAL"
    if shadow_supported:
        tier_name += "_DARK"

    return {
        "wrinkle_type": wrinkle_type,
        "pull_scale": float(np.clip(pull_scale, 0.35, 1.40)),
        "priority_tier": int(tier),
        "priority_tier_name": tier_name,
        "area_ratio": area_ratio,
        "shadow_supported": shadow_supported,
        "medium_size": medium_size,
    }


def build_wrinkle_heatmap(frame: np.ndarray, mask_u8: np.ndarray, args) -> Optional[Dict[str, Any]]:
    if frame is None or mask_u8 is None:
        return None
    mask = ((mask_u8 > 0).astype(np.uint8) * 255)
    if int(cv2.countNonZero(mask)) < 50:
        return None

    erode_px = max(0, int(getattr(args, "wrinkle_heatmap_erode_px", 12)))
    inner = mask.copy()
    if erode_px > 0:
        k = 2 * erode_px + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        eroded = cv2.erode(mask, kernel, iterations=1)
        if cv2.countNonZero(eroded) > max(50, int(0.20 * cv2.countNonZero(mask))):
            inner = eroded

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    clahe_clip = float(getattr(args, "wrinkle_heatmap_clahe_clip", 2.0))
    clahe_tile = max(2, int(getattr(args, "wrinkle_heatmap_clahe_tile", 8)))
    clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=(clahe_tile, clahe_tile))
    base = clahe.apply(gray)

    blur_k = odd_kernel_size(getattr(args, "wrinkle_heatmap_blur_ksize", 41), 5)
    blur = cv2.GaussianBlur(base, (blur_k, blur_k), 0)
    high_freq = cv2.absdiff(base, blur)
    # Keep a non-CLAHE photometric signal. Normalized heatmaps can strongly
    # respond to printed fabric patterns even when there is no physical fold.
    raw_blur = cv2.GaussianBlur(gray, (blur_k, blur_k), 0)
    raw_abs_contrast = cv2.absdiff(gray, raw_blur)

    sx = cv2.Sobel(base, cv2.CV_32F, 1, 0, ksize=3)
    sy = cv2.Sobel(base, cv2.CV_32F, 0, 1, ksize=3)
    grad = cv2.magnitude(sx, sy)
    lap = np.abs(cv2.Laplacian(base, cv2.CV_32F, ksize=3))

    feature_min_range = float(getattr(args, "wrinkle_heatmap_min_normalize_range", 8.0))
    high_u8 = normalize_masked_u8(high_freq, inner, min_range=feature_min_range)
    grad_u8 = normalize_masked_u8(grad, inner, min_range=feature_min_range)
    lap_u8 = normalize_masked_u8(lap, inner, min_range=feature_min_range)
    combined = (
        0.45 * high_u8.astype(np.float32)
        + 0.35 * grad_u8.astype(np.float32)
        + 0.20 * lap_u8.astype(np.float32)
    )

    if bool(getattr(args, "wrinkle_heatmap_gabor", True)):
        gabor_k = odd_kernel_size(getattr(args, "wrinkle_heatmap_gabor_ksize", 17), 7)
        sigma = float(getattr(args, "wrinkle_heatmap_gabor_sigma", 4.0))
        lambd = float(getattr(args, "wrinkle_heatmap_gabor_lambda", 8.0))
        src = base.astype(np.float32)
        gabor_max = np.zeros_like(src, dtype=np.float32)
        for theta in (0.0, math.pi / 4.0, math.pi / 2.0, 3.0 * math.pi / 4.0):
            kernel = cv2.getGaborKernel((gabor_k, gabor_k), sigma, theta, lambd, 0.5, 0, ktype=cv2.CV_32F)
            resp = np.abs(cv2.filter2D(src, cv2.CV_32F, kernel))
            gabor_max = np.maximum(gabor_max, resp)
        gabor_u8 = normalize_masked_u8(gabor_max, inner, min_range=feature_min_range)
        combined = 0.78 * combined + 0.22 * gabor_u8.astype(np.float32)

    shadow = build_local_shadow_map(frame, inner, args)
    if bool(getattr(args, "wrinkle_shadow_mode", True)):
        shadow_weight = float(np.clip(getattr(args, "wrinkle_shadow_weight", 0.24), 0.0, 0.60))
        combined = (1.0 - shadow_weight) * combined + shadow_weight * shadow["normalized"].astype(np.float32)

    heat = normalize_masked_u8(combined, inner, min_range=feature_min_range)
    vals = heat[inner > 0]
    if vals.size == 0:
        return None
    percentile = float(getattr(args, "wrinkle_heatmap_percentile", 93.0))
    threshold = int(np.clip(
        max(float(getattr(args, "wrinkle_heatmap_min_threshold", 42.0)), np.percentile(vals, percentile)),
        1, 254,
    ))
    binary = np.zeros_like(heat, dtype=np.uint8)
    binary[(heat >= threshold) & (inner > 0)] = 255
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, np.ones((3, 3), dtype=np.uint8), iterations=1)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    min_area = float(getattr(args, "wrinkle_heatmap_min_blob_area", 30.0))
    max_candidates = max(0, int(getattr(args, "wrinkle_heatmap_max_candidates", 6)))
    candidates = []
    mask_area = float(max(1, cv2.countNonZero(mask)))
    distance_map = cv2.distanceTransform((inner > 0).astype(np.uint8), cv2.DIST_L2, 5)
    shadow_raw = shadow["raw"]
    shadow_norm = shadow["normalized"]
    shadow_support_threshold = float(shadow["support_raw_threshold"])
    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area < min_area:
            continue
        M = cv2.moments(contour)
        if abs(M["m00"]) > 1e-6:
            cx = float(M["m10"] / M["m00"])
            cy = float(M["m01"] / M["m00"])
        else:
            pts = contour.reshape(-1, 2).astype(np.float32)
            cx, cy = float(np.mean(pts[:, 0])), float(np.mean(pts[:, 1]))
        tmp = np.zeros_like(binary)
        cv2.drawContours(tmp, [contour], -1, 255, -1)
        score = float(cv2.mean(heat, mask=tmp)[0])
        x, y, w, h = cv2.boundingRect(contour)
        ys_comp, xs_comp = np.where(tmp > 0)
        pts = np.column_stack([xs_comp, ys_comp]).astype(np.float32)
        major_axis = np.asarray([1.0, 0.0], dtype=np.float32)
        minor_axis = np.asarray([0.0, 1.0], dtype=np.float32)
        major_len = float(max(w, h))
        minor_len = float(max(1, min(w, h)))
        if len(pts) >= 5:
            centered = pts - pts.mean(axis=0, keepdims=True)
            try:
                cov = np.cov(centered.T)
                eigvals, eigvecs = np.linalg.eigh(cov)
                order = np.argsort(eigvals)[::-1]
                major_axis = eigvecs[:, order[0]].astype(np.float32)
                major_axis = unit_vec(major_axis)
                minor_axis = np.asarray([-major_axis[1], major_axis[0]], dtype=np.float32)
                proj_major = centered @ major_axis
                proj_minor = centered @ minor_axis
                major_len = float(np.ptp(proj_major)) if proj_major.size else major_len
                minor_len = float(np.ptp(proj_minor)) if proj_minor.size else minor_len
            except Exception:
                pass
        linearity = float(major_len / max(1.0, minor_len))
        cxi = int(np.clip(round(cx), 0, distance_map.shape[1] - 1))
        cyi = int(np.clip(round(cy), 0, distance_map.shape[0] - 1))
        edge_distance = float(distance_map[cyi, cxi])
        shadow_values = shadow_norm[tmp > 0]
        shadow_raw_values = shadow_raw[tmp > 0]
        raw_contrast_values = raw_abs_contrast[tmp > 0].astype(np.float32)
        shadow_mean = float(np.mean(shadow_values)) if shadow_values.size else 0.0
        shadow_max = float(np.max(shadow_values)) if shadow_values.size else 0.0
        shadow_raw_mean = float(np.mean(shadow_raw_values)) if shadow_raw_values.size else 0.0
        shadow_raw_max = float(np.max(shadow_raw_values)) if shadow_raw_values.size else 0.0
        shadow_support = (
            float(np.count_nonzero(shadow_raw_values >= shadow_support_threshold))
            / float(max(1, shadow_raw_values.size))
        )
        raw_contrast_threshold = float(getattr(args, "actual_wrinkle_contrast_pixel_threshold", 7.0))
        mean_abs_contrast = float(np.mean(raw_contrast_values)) if raw_contrast_values.size else 0.0
        max_abs_contrast = float(np.max(raw_contrast_values)) if raw_contrast_values.size else 0.0
        contrast_support = (
            float(np.count_nonzero(raw_contrast_values >= raw_contrast_threshold))
            / float(max(1, raw_contrast_values.size))
        )
        geometry = classify_wrinkle_geometry(
            area, mask_area, major_len, minor_len, edge_distance,
            shadow_mean, shadow_max, shadow_support, args,
        )
        priority = float(
            int(geometry["priority_tier"]) * 1_000_000.0
            + score * math.sqrt(max(1.0, area)) * min(4.0, max(1.0, linearity))
            + shadow_mean * 35.0
            + shadow_max * 8.0
        )
        candidates.append({
            "center_px": (cx, cy),
            "area_px": area,
            "score": score,
            "bbox": (int(x), int(y), int(w), int(h)),
            "major_axis_px": (float(major_axis[0]), float(major_axis[1])),
            "minor_axis_px": (float(minor_axis[0]), float(minor_axis[1])),
            "major_length_px": major_len,
            "minor_length_px": minor_len,
            "linearity": linearity,
            "edge_distance_px": edge_distance,
            "shadow_mean": shadow_mean,
            "shadow_max": shadow_max,
            "shadow_raw_mean": shadow_raw_mean,
            "shadow_raw_max": shadow_raw_max,
            "shadow_support_ratio": shadow_support,
            "mean_abs_contrast": mean_abs_contrast,
            "max_abs_contrast": max_abs_contrast,
            "contrast_support_ratio": contrast_support,
            **geometry,
            "priority_score": priority,
        })
    candidates.sort(
        key=lambda item: (
            float(item.get("priority_score", 0.0)),
            float(item.get("score", 0.0)),
            float(item.get("area_px", 0.0)),
        ),
        reverse=True,
    )
    finish_candidates = list(candidates)
    if max_candidates:
        candidates = candidates[:max_candidates]

    gray_values = gray[inner > 0].astype(np.float32)
    if gray_values.size:
        lum_p05 = float(np.percentile(gray_values, 5.0))
        lum_p95 = float(np.percentile(gray_values, 95.0))
        luminance_span = lum_p95 - lum_p05
        clipped_ratio = float(np.mean((gray_values <= 3.0) | (gray_values >= 252.0)))
        texture_p95 = float(np.percentile(high_freq[inner > 0], 95.0))
    else:
        lum_p05 = lum_p95 = luminance_span = texture_p95 = 0.0
        clipped_ratio = 1.0
    reliable = bool(
        luminance_span >= float(getattr(args, "finish_min_luminance_span", 10.0))
        and clipped_ratio <= float(getattr(args, "finish_max_clipped_ratio", 0.35))
        and texture_p95 >= float(getattr(args, "finish_min_texture_p95", 0.5))
    )

    return {
        "heatmap": heat,
        "binary": binary,
        "inner_mask": inner,
        "threshold": threshold,
        "percentile": percentile,
        "candidates": candidates,
        "finish_candidates": finish_candidates,
        "total_candidate_count": len(finish_candidates),
        "wrinkle_ratio": float(cv2.countNonZero(binary)) / float(max(1, cv2.countNonZero(inner))),
        "local_shadow": shadow_norm,
        "shadow_scale_raw": float(shadow["scale_raw"]),
        "quality": {
            "reliable": reliable,
            "luminance_p05": lum_p05,
            "luminance_p95": lum_p95,
            "luminance_span": luminance_span,
            "clipped_ratio": clipped_ratio,
            "texture_p95_raw": texture_p95,
        },
    }


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


def parse_float_csv(text: Any, default: Sequence[float]) -> List[float]:
    vals: List[float] = []
    for raw in str(text or "").split(","):
        raw = raw.strip()
        if not raw:
            continue
        try:
            vals.append(float(raw))
        except ValueError:
            continue
    if not vals:
        vals = [float(v) for v in default]
    out: List[float] = []
    seen = set()
    for v in vals:
        key = round(float(v), 4)
        if key not in seen:
            seen.add(key)
            out.append(float(v))
    return out


def parse_flip_modes(text: Any) -> List[str]:
    vals: List[str] = []
    for raw in str(text or "").split(","):
        mode = raw.strip().lower()
        if mode in ("none", "h") and mode not in vals:
            vals.append(mode)
    if not vals:
        vals = ["none"]
    if "none" not in vals:
        vals.insert(0, "none")
    return vals


def rotate_image_and_inverse(frame: np.ndarray, angle_deg: float,
                             interpolation: Optional[int] = None
                             ) -> Tuple[np.ndarray, np.ndarray]:
    h, w = frame.shape[:2]
    cx, cy = w / 2.0, h / 2.0
    M = cv2.getRotationMatrix2D((cx, cy), float(angle_deg), 1.0)
    cos = abs(float(M[0, 0]))
    sin = abs(float(M[0, 1]))
    new_w = int(round((h * sin) + (w * cos)))
    new_h = int(round((h * cos) + (w * sin)))
    M[0, 2] += (new_w / 2.0) - cx
    M[1, 2] += (new_h / 2.0) - cy
    if interpolation is None:
        interpolation = cv2.INTER_LINEAR
    rotated = cv2.warpAffine(
        frame, M, (new_w, new_h), flags=int(interpolation),
        borderMode=cv2.BORDER_CONSTANT, borderValue=0,
    )
    M3 = np.vstack([M, [0.0, 0.0, 1.0]]).astype(np.float32)
    return rotated, np.linalg.inv(M3).astype(np.float32)


def prepare_mask_guided_pose_view(
        frame: np.ndarray, mask: Optional[BottomMaskBoard], angle_deg: float, args,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """Deskew a tight garment ROI while retaining an exact view-to-frame map."""
    def full_frame_fallback() -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        fallback_image, fallback_inverse = rotate_image_and_inverse(frame, angle_deg)
        return fallback_image, fallback_inverse, {
            "mode": "full-frame",
            "input_wh": (int(fallback_image.shape[1]), int(fallback_image.shape[0])),
            "mask_fill_ratio": 0.0,
        }

    if not bool(getattr(args, "pose_mask_roi_tta", True)):
        return full_frame_fallback()
    if mask is None or getattr(mask, "mask_u8", None) is None:
        return full_frame_fallback()

    binary = np.asarray(mask.mask_u8) > 0
    if binary.shape != frame.shape[:2]:
        return full_frame_fallback()
    ys, xs = np.where(binary)
    if len(xs) < int(getattr(args, "pose_mask_roi_min_area_px", 1200)):
        return full_frame_fallback()

    h, w = frame.shape[:2]
    mask_x0, mask_x1 = int(np.min(xs)), int(np.max(xs)) + 1
    mask_y0, mask_y1 = int(np.min(ys)), int(np.max(ys)) + 1
    mask_width = max(1, mask_x1 - mask_x0)
    mask_height = max(1, mask_y1 - mask_y0)
    padding = max(
        int(getattr(args, "pose_mask_roi_min_padding_px", 24)),
        int(round(max(mask_width, mask_height) * float(getattr(
            args, "pose_mask_roi_padding_ratio", 0.12,
        )))),
    )
    crop_x0 = max(0, mask_x0 - padding)
    crop_y0 = max(0, mask_y0 - padding)
    crop_x1 = min(w, mask_x1 + padding)
    crop_y1 = min(h, mask_y1 + padding)
    if crop_x1 - crop_x0 < 64 or crop_y1 - crop_y0 < 64:
        return full_frame_fallback()

    frame_crop = frame[crop_y0:crop_y1, crop_x0:crop_x1]
    mask_crop = np.asarray(mask.mask_u8[crop_y0:crop_y1, crop_x0:crop_x1], dtype=np.uint8)
    rotated_frame, rotated_to_crop = rotate_image_and_inverse(
        frame_crop, angle_deg, interpolation=cv2.INTER_LINEAR,
    )
    rotated_mask, _ = rotate_image_and_inverse(
        mask_crop, angle_deg, interpolation=cv2.INTER_NEAREST,
    )
    rotated_binary = np.asarray(rotated_mask) > 0
    rotated_ys, rotated_xs = np.where(rotated_binary)
    if len(rotated_xs) < int(getattr(args, "pose_mask_roi_min_area_px", 1200)):
        return full_frame_fallback()

    rotated_mask_x0 = int(np.min(rotated_xs))
    rotated_mask_x1 = int(np.max(rotated_xs)) + 1
    rotated_mask_y0 = int(np.min(rotated_ys))
    rotated_mask_y1 = int(np.max(rotated_ys)) + 1
    rotated_width = max(1, rotated_mask_x1 - rotated_mask_x0)
    rotated_height = max(1, rotated_mask_y1 - rotated_mask_y0)
    final_padding = max(
        int(getattr(args, "pose_mask_roi_final_min_padding_px", 18)),
        int(round(max(rotated_width, rotated_height) * float(getattr(
            args, "pose_mask_roi_final_padding_ratio", 0.08,
        )))),
    )
    view_x0 = max(0, rotated_mask_x0 - final_padding)
    view_y0 = max(0, rotated_mask_y0 - final_padding)
    view_x1 = min(rotated_frame.shape[1], rotated_mask_x1 + final_padding)
    view_y1 = min(rotated_frame.shape[0], rotated_mask_y1 + final_padding)
    if view_x1 - view_x0 < 64 or view_y1 - view_y0 < 64:
        return full_frame_fallback()

    view = rotated_frame[view_y0:view_y1, view_x0:view_x1].copy()
    view_mask = rotated_binary[view_y0:view_y1, view_x0:view_x1]
    crop_to_frame = np.asarray([
        [1.0, 0.0, float(crop_x0)],
        [0.0, 1.0, float(crop_y0)],
        [0.0, 0.0, 1.0],
    ], dtype=np.float32)
    view_to_rotated = np.asarray([
        [1.0, 0.0, float(view_x0)],
        [0.0, 1.0, float(view_y0)],
        [0.0, 0.0, 1.0],
    ], dtype=np.float32)
    view_to_frame = crop_to_frame @ rotated_to_crop @ view_to_rotated
    fill_ratio = float(np.count_nonzero(view_mask)) / float(max(1, view_mask.size))
    return view, np.asarray(view_to_frame, dtype=np.float32), {
        "mode": "mask-roi",
        "input_wh": (int(view.shape[1]), int(view.shape[0])),
        "source_crop_xyxy": (crop_x0, crop_y0, crop_x1, crop_y1),
        "rotated_crop_xyxy": (view_x0, view_y0, view_x1, view_y1),
        "mask_fill_ratio": fill_ratio,
    }


def hflip_image_and_inverse(frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    h, w = frame.shape[:2]
    flipped = frame[:, ::-1].copy()
    Finv = np.asarray([
        [-1.0, 0.0, float(w - 1)],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ], dtype=np.float32)
    return flipped, Finv


def transform_keypoints(kxy: np.ndarray, Tinv3: np.ndarray) -> np.ndarray:
    pts_h = np.concatenate([kxy.astype(np.float32), np.ones((len(kxy), 1), dtype=np.float32)], axis=1)
    out = (Tinv3 @ pts_h.T).T
    return out[:, :2].astype(np.float32)


def remap_hflip_keypoints(kxy: np.ndarray, kcf: Optional[np.ndarray]) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    out_xy = kxy[HFLIP_REMAP].copy()
    out_cf = None if kcf is None else kcf[HFLIP_REMAP].copy()
    return out_xy, out_cf


def mean_valid_kpts(kxy: np.ndarray, kcf: Optional[np.ndarray], idxs: Sequence[int],
                   min_conf: float, image_shape) -> Optional[np.ndarray]:
    pts = []
    for idx in idxs:
        if idx >= len(kxy):
            continue
        conf = None if kcf is None or idx >= len(kcf) else float(kcf[idx])
        if valid_kpt_xy_conf(kxy[idx], conf, image_shape, min_conf):
            pts.append(np.asarray(kxy[idx], dtype=np.float32))
    if not pts:
        return None
    return np.mean(np.stack(pts, axis=0), axis=0)


def valid_kpt_xy_conf(pt: np.ndarray, conf: Optional[float], image_shape, min_conf: float) -> bool:
    h, w = image_shape[:2]
    x, y = float(pt[0]), float(pt[1])
    if not (np.isfinite(x) and np.isfinite(y)):
        return False
    # Match step_d2.py exactly: points on the 1px boundary remain usable.
    if x < 1.0 or y < 1.0 or x >= float(w) or y >= float(h):
        return False
    if conf is not None and float(conf) < float(min_conf):
        return False
    return True


def safe_norm(v: np.ndarray) -> float:
    return float(np.linalg.norm(np.asarray(v, dtype=np.float32)))


def safe_unit(v: np.ndarray) -> Optional[np.ndarray]:
    n = safe_norm(v)
    if n < 1e-6:
        return None
    return np.asarray(v, dtype=np.float32) / n


def angle_between_deg(v1: np.ndarray, v2: np.ndarray) -> Optional[float]:
    u1 = safe_unit(v1)
    u2 = safe_unit(v2)
    if u1 is None or u2 is None:
        return None
    return float(np.degrees(np.arccos(float(np.clip(np.dot(u1, u2), -1.0, 1.0)))))


def _mask_pca_geometry_px(mask: Optional[BottomMaskBoard], args) -> Optional[Dict[str, Any]]:
    """Return cached PCA geometry for mask/pose agreement checks only.

    PCA axes are undirected, so this helper must never decide waist-versus-hem
    polarity. The one-waist/two-hems silhouette profile remains responsible for
    that semantic decision.
    """
    if mask is None or getattr(mask, "mask_u8", None) is None:
        return None
    binary = np.asarray(mask.mask_u8) > 0
    area_px = int(np.count_nonzero(binary))
    min_area = int(getattr(args, "pose_geometry_pca_min_area_px", 1800))
    if area_px < min_area:
        return None

    max_points = max(2000, int(getattr(args, "pose_geometry_pca_max_points", 40000)))
    signature = (tuple(binary.shape), area_px, max_points)
    cached = getattr(mask, "_pose_geometry_pca_cache", None)
    if isinstance(cached, dict) and cached.get("signature") == signature:
        return cached

    sample_step = max(1, int(math.ceil(math.sqrt(float(area_px) / float(max_points)))))
    ys, xs = np.where(binary[::sample_step, ::sample_step])
    if len(xs) < 200:
        return None
    points = np.column_stack([
        xs.astype(np.float32) * float(sample_step),
        ys.astype(np.float32) * float(sample_step),
    ])
    center = np.mean(points, axis=0).astype(np.float32)
    centered = points - center
    covariance = np.cov(centered.T)
    if covariance.shape != (2, 2) or not np.all(np.isfinite(covariance)):
        return None
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    major_value = max(1e-6, float(eigenvalues[order[0]]))
    minor_value = max(1e-6, float(eigenvalues[order[1]]))
    major = safe_unit(np.asarray(eigenvectors[:, order[0]], dtype=np.float32))
    minor = safe_unit(np.asarray(eigenvectors[:, order[1]], dtype=np.float32))
    if major is None or minor is None:
        return None
    major_projection = centered @ major
    minor_projection = centered @ minor
    major_span = float(np.percentile(major_projection, 99.0) - np.percentile(major_projection, 1.0))
    minor_span = float(np.percentile(minor_projection, 99.0) - np.percentile(minor_projection, 1.0))
    axis_ratio = float(math.sqrt(major_value / minor_value))
    reliable = bool(
        axis_ratio >= float(getattr(args, "pose_geometry_pca_min_axis_ratio", 1.10))
        and major_span >= float(getattr(args, "pose_geometry_pca_min_span_px", 80.0))
        and minor_span >= float(getattr(args, "pose_geometry_pca_min_width_px", 55.0))
    )
    result = {
        "signature": signature,
        "center_px": center,
        "major_axis_px": major,
        "minor_axis_px": minor,
        "major_span_px": major_span,
        "minor_span_px": minor_span,
        "axis_ratio": axis_ratio,
        "reliable": reliable,
    }
    setattr(mask, "_pose_geometry_pca_cache", result)
    return result


def _mask_oriented_rectangle_px(
        mask: Optional[BottomMaskBoard], args,
) -> Optional[Dict[str, Any]]:
    """Return an angle-independent four-edge rectangle around the pants mask."""
    if mask is None or getattr(mask, "contour", None) is None:
        return None
    contour = np.asarray(mask.contour, dtype=np.float32).reshape(-1, 1, 2)
    if len(contour) < 4:
        return None
    try:
        rect = cv2.minAreaRect(contour)
        center = np.asarray(rect[0], dtype=np.float32)
        corners = np.asarray(cv2.boxPoints(rect), dtype=np.float32)
    except Exception:
        return None
    if corners.shape != (4, 2) or not np.all(np.isfinite(corners)):
        return None
    angles = np.arctan2(corners[:, 1] - center[1], corners[:, 0] - center[0])
    corners = corners[np.argsort(angles)]
    edges: List[Dict[str, Any]] = []
    for index in range(4):
        start = corners[index]
        end = corners[(index + 1) % 4]
        midpoint = 0.5 * (start + end)
        tangent = safe_unit(end - start)
        inward = safe_unit(center - midpoint)
        if tangent is None or inward is None:
            return None
        edges.append({
            "index": int(index),
            "start_px": start.astype(np.float32),
            "end_px": end.astype(np.float32),
            "mid_px": midpoint.astype(np.float32),
            "tangent": tangent.astype(np.float32),
            "inward": inward.astype(np.float32),
            "length_px": safe_norm(end - start),
        })
    pair_axes: List[Dict[str, Any]] = []
    for first_index in (0, 1):
        opposite_index = (first_index + 2) % 4
        first_mid = np.asarray(edges[first_index]["mid_px"], dtype=np.float32)
        opposite_mid = np.asarray(edges[opposite_index]["mid_px"], dtype=np.float32)
        axis = safe_unit(opposite_mid - first_mid)
        if axis is None:
            continue
        pair_axes.append({
            "pair_index": int(first_index),
            "negative_edge_index": int(first_index),
            "positive_edge_index": int(opposite_index),
            "axis": axis.astype(np.float32),
            "span_px": safe_norm(opposite_mid - first_mid),
            "negative_edge_length_px": float(edges[first_index]["length_px"]),
            "positive_edge_length_px": float(edges[opposite_index]["length_px"]),
        })
    if len(pair_axes) != 2:
        return None
    return {
        "center_px": center,
        "corners_px": corners,
        "edges": edges,
        "pair_axes": pair_axes,
        "size_wh": tuple(float(value) for value in rect[1]),
        "angle_deg": float(rect[2]),
    }


def _pose_pca_pair_alignment(waist_axis: Optional[np.ndarray], body_axis: Optional[np.ndarray],
                             pca: Optional[Dict[str, Any]]) -> Optional[float]:
    """Match orthogonal pose axes to either undirected PCA-axis assignment."""
    waist_u = None if waist_axis is None else safe_unit(waist_axis)
    body_u = None if body_axis is None else safe_unit(body_axis)
    if waist_u is None or body_u is None or pca is None:
        return None
    major = safe_unit(np.asarray(pca["major_axis_px"], dtype=np.float32))
    minor = safe_unit(np.asarray(pca["minor_axis_px"], dtype=np.float32))
    if major is None or minor is None:
        return None
    assignment_a = 0.5 * (abs(float(np.dot(body_u, major))) + abs(float(np.dot(waist_u, minor))))
    assignment_b = 0.5 * (abs(float(np.dot(body_u, minor))) + abs(float(np.dot(waist_u, major))))
    return float(max(assignment_a, assignment_b))


def pose_tta_summary(kxy: np.ndarray, kcf: Optional[np.ndarray], min_conf: float,
                     image_shape, args, mask: Optional[BottomMaskBoard] = None) -> Dict[str, Any]:
    valid = np.zeros((len(BOTTOM_POSE_KPT_NAMES),), dtype=bool)
    visible_count = 0
    conf_sum = 0.0
    conf_count = 0
    for idx in range(min(len(kxy), len(BOTTOM_POSE_KPT_NAMES))):
        conf = None if kcf is None or idx >= len(kcf) else float(kcf[idx])
        if valid_kpt_xy_conf(kxy[idx], conf, image_shape, min_conf):
            valid[idx] = True
            visible_count += 1
            conf_sum += 1.0 if conf is None else float(conf)
            conf_count += 1

    waist_count = int(sum(1 for idx in (0, 1, 2) if valid[idx]))
    hem_count = int(sum(1 for idx in (4, 5, 6, 7) if valid[idx]))
    crotch_visible = bool(valid[3])
    mean_conf = conf_sum / max(1, conf_count)
    waist_conf_values = [
        1.0 if kcf is None or idx >= len(kcf) else float(kcf[idx])
        for idx in (0, 1, 2) if valid[idx]
    ]
    waist_mean_conf = (
        float(np.mean(waist_conf_values)) if waist_conf_values else 0.0
    )
    geometry_score = 0.0
    penalties = 0.0
    hard_failures: List[str] = []

    wl = kxy[0] if valid[0] else None
    wc = kxy[1] if valid[1] else None
    wr = kxy[2] if valid[2] else None
    cr = kxy[3] if valid[3] else None
    lhc = mean_valid_kpts(kxy, kcf, [4, 5], min_conf, image_shape)
    rhc = mean_valid_kpts(kxy, kcf, [6, 7], min_conf, image_shape)
    lower = None
    if lhc is not None and rhc is not None:
        lower = 0.5 * (lhc + rhc)
    elif lhc is not None:
        lower = lhc
    elif rhc is not None:
        lower = rhc

    waist_width = None
    hem_gap = None
    leg_open = None
    waist_axis_dot = None
    waist_center_offset_ratio = None
    crotch_axis_t = None
    crotch_lateral_ratio = None
    hem_gap_ratio = None
    waist_vec = None
    if wl is not None and wc is not None and wr is not None:
        waist_vec = wr - wl
        waist_width = safe_norm(waist_vec)
        waist_mid = 0.5 * (wl + wr)
        center_offset = safe_norm(wc - waist_mid)
        waist_center_offset_ratio = center_offset / max(1.0, waist_width)
        if waist_width >= float(getattr(args, "pose_tta_min_waist_width_px", 40.0)):
            geometry_score += 1.0
        else:
            penalties += 2.5
            hard_failures.append("waist_width")
        max_offset = max(25.0, waist_width * float(getattr(args, "pose_tta_max_waist_center_offset_ratio", 0.35)))
        if center_offset <= max_offset:
            geometry_score += 1.5
        else:
            penalties += 2.5
            if waist_center_offset_ratio > float(getattr(args, "pose_geometry_max_waist_center_offset_ratio", 0.60)):
                hard_failures.append("waist_center")
        if cr is not None:
            waist_to_crotch = safe_norm(cr - wc)
            if waist_to_crotch >= max(25.0, waist_width * float(getattr(args, "pose_tta_min_waist_to_crotch_ratio", 0.35))):
                geometry_score += 1.5
            else:
                penalties += 2.0
        if lower is not None:
            u_waist = safe_unit(waist_vec)
            u_axis = safe_unit(lower - wc)
            if u_waist is not None and u_axis is not None:
                waist_axis_dot = abs(float(np.dot(u_waist, u_axis)))
                if waist_axis_dot <= float(getattr(args, "pose_tta_max_waist_axis_dot", 0.45)):
                    geometry_score += 2.0
                else:
                    penalties += 2.0
                    if waist_axis_dot > float(getattr(args, "pose_geometry_max_waist_axis_dot", 0.72)):
                        hard_failures.append("waist_axis")
        if cr is not None and lower is not None:
            axis = lower - wc
            axis_u = safe_unit(axis)
            axis_len = safe_norm(axis)
            if axis_u is not None and axis_len > 1e-6:
                proj = float(np.dot(cr - wc, axis_u))
                crotch_axis_t = proj / axis_len
                lateral_vec = np.asarray([-axis_u[1], axis_u[0]], dtype=np.float32)
                crotch_lateral_ratio = abs(float(np.dot(cr - wc, lateral_vec))) / max(1.0, waist_width)
                if 0.15 * axis_len <= proj <= 0.90 * axis_len:
                    geometry_score += 1.2
                else:
                    penalties += 1.8
                if not (
                    float(getattr(args, "pose_geometry_crotch_t_min", 0.04))
                    <= crotch_axis_t
                    <= float(getattr(args, "pose_geometry_crotch_t_max", 1.05))
                ):
                    hard_failures.append("crotch_longitudinal")
                if crotch_lateral_ratio > float(getattr(args, "pose_geometry_max_crotch_lateral_ratio", 0.70)):
                    hard_failures.append("crotch_lateral")
    else:
        penalties += 3.0
        hard_failures.append("waist_missing")

    if lhc is not None and rhc is not None:
        hem_gap = safe_norm(rhc - lhc)
        if waist_width is not None and waist_width > 1e-6:
            hem_gap_ratio = hem_gap / waist_width
    if cr is not None and lhc is not None and rhc is not None:
        leg_open = angle_between_deg(lhc - cr, rhc - cr)

    need_pre_spread = False
    if hem_count < int(getattr(args, "pose_tta_pre_spread_min_hem_visible", 2)):
        need_pre_spread = True
    if not crotch_visible:
        need_pre_spread = True
    if waist_width is not None and hem_gap is not None and waist_width > 1e-6:
        if hem_gap / waist_width < float(getattr(args, "pose_tta_pre_spread_hem_gap_ratio", 0.42)):
            need_pre_spread = True
    if leg_open is not None and leg_open < float(getattr(args, "pose_tta_pre_spread_open_angle_deg", 22.0)):
        need_pre_spread = True

    # D2 strict keeps the model landmarks unchanged. Geometry validation may rank
    # or reject a candidate, but it never rewrites waist/crotch/hem semantics.
    d2_strict = bool(getattr(args, "pose_d2_strict", True))
    geometry_validation = bool(getattr(args, "pose_geometry_validation", True))
    mask_bonus = 0.0
    mask_inside_ratio = None
    mask_mean_signed_distance = None
    pca_pair_alignment = None
    pca_axis_ratio = None
    pca_reliable = False
    if ((geometry_validation or not d2_strict) and mask is not None
            and getattr(mask, "contour", None) is not None):
        signed_distances = []
        for idx in range(min(8, len(kxy))):
            if not valid[idx]:
                continue
            signed_distances.append(float(cv2.pointPolygonTest(
                mask.contour,
                (float(kxy[idx, 0]), float(kxy[idx, 1])),
                True,
            )))
        if signed_distances:
            max_outside = float(getattr(args, "pose_tta_mask_max_outside_px", 20.0))
            mask_inside_ratio = float(np.mean(
                np.asarray(signed_distances, dtype=np.float32) >= -max_outside
            ))
            mask_mean_signed_distance = float(np.mean(np.clip(
                np.asarray(signed_distances, dtype=np.float32),
                -max_outside * 2.0,
                max_outside * 2.0,
            )))
            mask_weight = float(getattr(
                args,
                "pose_geometry_strict_mask_weight" if d2_strict else "pose_tta_mask_weight",
                3.0 if d2_strict else 6.0,
            ))
            mask_bonus += mask_weight * mask_inside_ratio
            mask_bonus += 0.02 * mask_mean_signed_distance
            if mask_inside_ratio < float(getattr(args, "pose_tta_mask_min_inside_ratio", 0.75)):
                penalties += float(getattr(args, "pose_tta_mask_low_agreement_penalty", 4.0))
            if mask_inside_ratio < float(getattr(args, "pose_geometry_min_mask_inside_ratio", 0.60)):
                hard_failures.append("mask_support")

        pca = _mask_pca_geometry_px(mask, args)
        if pca is not None:
            pca_axis_ratio = float(pca.get("axis_ratio", 1.0))
            pca_reliable = bool(pca.get("reliable", False))
            body_axis = None if lower is None or wc is None else np.asarray(lower - wc, dtype=np.float32)
            pca_pair_alignment = _pose_pca_pair_alignment(waist_vec, body_axis, pca)
            if pca_pair_alignment is not None:
                mask_bonus += float(getattr(args, "pose_geometry_pca_score_weight", 4.0)) * max(
                    0.0, (pca_pair_alignment - 0.45) / 0.55,
                )
                if (
                    pca_reliable
                    and pca_pair_alignment < float(getattr(args, "pose_geometry_pca_min_alignment", 0.50))
                ):
                    penalties += float(getattr(args, "pose_geometry_pca_penalty", 3.0))
                    hard_failures.append("mask_pca_axes")

    if visible_count < int(getattr(args, "pose_geometry_min_visible", 7)):
        hard_failures.append("visible")
    if waist_count < 3:
        hard_failures.append("waist_points")
    if not crotch_visible:
        hard_failures.append("crotch")
    if hem_count < int(getattr(args, "pose_geometry_min_hem_visible", 3)):
        hard_failures.append("hem_points")

    hard_failures = list(dict.fromkeys(hard_failures))
    structure_score = float(np.clip(
        (
            0.18 * (visible_count / 8.0)
            + 0.16 * (waist_count / 3.0)
            + 0.10 * float(crotch_visible)
            + 0.12 * (hem_count / 4.0)
            + 0.24 * np.clip(geometry_score / 7.2, 0.0, 1.0)
            + 0.10 * (mask_inside_ratio if mask_inside_ratio is not None else 0.70)
            + 0.10 * (pca_pair_alignment if pca_pair_alignment is not None else 0.70)
        ),
        0.0,
        1.0,
    ))
    structure_reliable = bool(
        not hard_failures
        and structure_score >= float(getattr(args, "pose_geometry_tta_min_score", 0.62))
    )

    score = 0.0
    score += 3.5 * waist_count
    score += 2.0 * float(crotch_visible)
    score += 1.0 * hem_count
    # step_d2.py scores confidence only on the three semantic waist points.
    score += 2.0 * waist_mean_conf
    score += 1.4 * visible_count
    score += 3.2 * geometry_score
    score += mask_bonus
    score -= 2.2 * penalties
    if need_pre_spread:
        score -= 4.0
    return {
        "score": float(score),
        "visible_count": int(visible_count),
        "waist_count": int(waist_count),
        "hem_count": int(hem_count),
        "crotch_visible": bool(crotch_visible),
        "geometry_score": float(geometry_score),
        "structure_score": structure_score,
        "structure_reliable": structure_reliable,
        "hard_failures": hard_failures,
        "penalties": float(penalties),
        "need_pre_spread": bool(need_pre_spread),
        "waist_axis_dot": waist_axis_dot,
        "waist_center_offset_ratio": waist_center_offset_ratio,
        "crotch_axis_t": crotch_axis_t,
        "crotch_lateral_ratio": crotch_lateral_ratio,
        "hem_gap_ratio": hem_gap_ratio,
        "mean_conf": float(mean_conf),
        "waist_mean_conf": float(waist_mean_conf),
        "d2_strict": bool(d2_strict),
        "mask_inside_ratio": mask_inside_ratio,
        "mask_mean_signed_distance": mask_mean_signed_distance,
        "pca_pair_alignment": pca_pair_alignment,
        "pca_axis_ratio": pca_axis_ratio,
        "pca_reliable": pca_reliable,
    }


def _wrap_rotation_deg(angle_deg: float) -> float:
    return float((float(angle_deg) + 180.0) % 360.0 - 180.0)


def _append_unique_rotation(angles: List[float], angle_deg: float,
                            tolerance_deg: float = 0.05) -> None:
    wrapped = _wrap_rotation_deg(angle_deg)
    if any(abs(_wrap_rotation_deg(wrapped - old)) <= tolerance_deg for old in angles):
        return
    angles.append(wrapped)


def _weighted_median_1d(values: np.ndarray, weights: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float32).reshape(-1)
    weights = np.asarray(weights, dtype=np.float32).reshape(-1)
    if values.size == 0:
        return 0.0
    order = np.argsort(values)
    sorted_values = values[order]
    sorted_weights = np.maximum(weights[order], 1e-6)
    cutoff = 0.5 * float(np.sum(sorted_weights))
    index = int(np.searchsorted(np.cumsum(sorted_weights), cutoff, side="left"))
    return float(sorted_values[min(index, len(sorted_values) - 1)])


def _pose_candidate_axis(kxy: np.ndarray, kcf: Optional[np.ndarray],
                         min_conf: float, image_shape) -> Optional[np.ndarray]:
    waist = mean_valid_kpts(kxy, kcf, [0, 1, 2], min_conf, image_shape)
    hem = mean_valid_kpts(kxy, kcf, [4, 5, 6, 7], min_conf, image_shape)
    if waist is None or hem is None:
        return None
    return safe_unit(np.asarray(hem - waist, dtype=np.float32))


def _canonicalize_pose_lateral_order(
        kxy: np.ndarray, kcf: np.ndarray, directed_axis: Optional[np.ndarray],
) -> Tuple[np.ndarray, np.ndarray]:
    """Assign left/right in a garment-local frame after waist polarity is known."""
    axis = None if directed_axis is None else safe_unit(directed_axis)
    out_xy = np.asarray(kxy, dtype=np.float32).copy()
    out_cf = np.asarray(kcf, dtype=np.float32).copy()
    if axis is None or len(out_xy) < 8:
        return out_xy, out_cf

    # For an upright image axis=(0,+1), garment-right is (+1,0). This
    # right-handed convention rotates continuously with the pants.
    garment_right = np.asarray([axis[1], -axis[0]], dtype=np.float32)

    def reorder(indices: Sequence[int]) -> None:
        ordered = sorted(
            [int(index) for index in indices],
            key=lambda index: float(np.dot(out_xy[index], garment_right)),
        )
        source_xy = out_xy.copy()
        source_cf = out_cf.copy()
        for destination, source in zip(indices, ordered):
            out_xy[int(destination)] = source_xy[int(source)]
            out_cf[int(destination)] = source_cf[int(source)]

    reorder((0, 1, 2))
    reorder((4, 5, 6, 7))
    return out_xy, out_cf


def _fuse_pose_candidates(
        candidates: Sequence[Dict[str, Any]], image_shape, min_conf: float, args,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any], List[Dict[str, Any]]]:
    if not candidates:
        raise ValueError("cannot fuse an empty pose candidate list")
    ordered = sorted(
        candidates,
        key=lambda candidate: float(candidate["summary"].get("score", -1e9)),
        reverse=True,
    )
    top_score = float(ordered[0]["summary"].get("score", 0.0))
    max_drop = max(0.0, float(getattr(
        args, "pose_canonical_consensus_max_score_drop", 12.0,
    )))
    max_candidates = max(1, int(getattr(args, "pose_canonical_consensus_max_candidates", 6)))
    selected = [
        candidate for candidate in ordered
        if float(candidate["summary"].get("score", -1e9)) >= top_score - max_drop
    ][:max_candidates]
    temperature = max(0.5, float(getattr(args, "pose_canonical_consensus_temperature", 4.0)))

    point_count = max(len(np.asarray(candidate["kxy"])) for candidate in selected)
    fused_xy = np.zeros((point_count, 2), dtype=np.float32)
    fused_cf = np.zeros((point_count,), dtype=np.float32)
    dispersions: List[float] = []
    fallback_xy = np.asarray(selected[0]["kxy"], dtype=np.float32)
    fallback_cf = np.asarray(selected[0]["kcf"], dtype=np.float32)

    for keypoint_index in range(point_count):
        points: List[np.ndarray] = []
        confidences: List[float] = []
        weights: List[float] = []
        for candidate in selected:
            candidate_xy = np.asarray(candidate["kxy"], dtype=np.float32)
            candidate_cf = np.asarray(candidate["kcf"], dtype=np.float32)
            if keypoint_index >= len(candidate_xy):
                continue
            point_conf = (
                float(candidate_cf[keypoint_index])
                if keypoint_index < len(candidate_cf) else 0.0
            )
            if not valid_kpt_xy_conf(
                    candidate_xy[keypoint_index], point_conf, image_shape, min_conf):
                continue
            score_delta = float(candidate["summary"].get("score", top_score)) - top_score
            candidate_weight = math.exp(max(-12.0, score_delta / temperature))
            points.append(candidate_xy[keypoint_index])
            confidences.append(point_conf)
            weights.append(candidate_weight * max(0.05, point_conf))

        if not points:
            if keypoint_index < len(fallback_xy):
                fused_xy[keypoint_index] = fallback_xy[keypoint_index]
                fused_cf[keypoint_index] = (
                    fallback_cf[keypoint_index]
                    if keypoint_index < len(fallback_cf) else 0.0
                )
            dispersions.append(float("inf"))
            continue

        point_array = np.asarray(points, dtype=np.float32)
        weight_array = np.asarray(weights, dtype=np.float32)
        fused_point = np.asarray([
            _weighted_median_1d(point_array[:, 0], weight_array),
            _weighted_median_1d(point_array[:, 1], weight_array),
        ], dtype=np.float32)
        distances = np.linalg.norm(point_array - fused_point[None, :], axis=1)
        dispersion = _weighted_median_1d(distances, weight_array)
        confidence = float(np.average(
            np.asarray(confidences, dtype=np.float32), weights=weight_array,
        ))
        dispersion_scale = max(5.0, float(getattr(
            args, "pose_canonical_consensus_conf_scale_px", 45.0,
        )))
        fused_xy[keypoint_index] = fused_point
        fused_cf[keypoint_index] = confidence * math.exp(-dispersion / dispersion_scale)
        dispersions.append(float(dispersion))

    finite_dispersion = [value for value in dispersions if np.isfinite(value)]
    metrics = {
        "candidate_count": len(selected),
        "mean_dispersion_px": (
            float(np.mean(finite_dispersion)) if finite_dispersion else float("inf")
        ),
        "max_dispersion_px": (
            float(np.max(finite_dispersion)) if finite_dispersion else float("inf")
        ),
        "per_keypoint_dispersion_px": dispersions,
    }
    return fused_xy, fused_cf, metrics, selected


def evaluate_waistband_heat_evidence(
        wrinkle: Optional[Dict[str, Any]], samples: Optional[Dict[str, Any]],
        axis: np.ndarray, args,
) -> Dict[str, Any]:
    """Compare dense transverse Heatmap bands at both garment-axis ends."""
    result: Dict[str, Any] = {
        "available": False,
        "reliable": False,
        "selected_end": None,
        "negative_score": 0.0,
        "positive_score": 0.0,
        "score_margin": 0.0,
        "reason": "waistband Heatmap unavailable",
    }
    if not bool(getattr(args, "pose_waistband_heat", True)):
        result["reason"] = "waistband Heatmap disabled"
        return result
    if not isinstance(wrinkle, dict) or samples is None:
        return result
    heat = wrinkle.get("heatmap")
    if heat is None:
        return result
    heat = np.asarray(heat, dtype=np.uint8)
    if heat.ndim != 2 or heat.size == 0:
        return result
    axis = safe_unit(axis)
    if axis is None:
        result["reason"] = "waistband axis invalid"
        return result

    points_px = np.asarray(samples.get("points_px"), dtype=np.float32)
    center_px = np.asarray(samples.get("center_px"), dtype=np.float32)
    if points_px.ndim != 2 or points_px.shape[1] != 2 or len(points_px) < 200:
        result["reason"] = "waistband mask samples unavailable"
        return result
    h, w = heat.shape[:2]
    xs = np.rint(points_px[:, 0]).astype(np.int32)
    ys = np.rint(points_px[:, 1]).astype(np.int32)
    inside = (xs >= 0) & (xs < w) & (ys >= 0) & (ys < h)
    if int(np.count_nonzero(inside)) < 200:
        result["reason"] = "waistband samples outside Heatmap"
        return result
    points_px = points_px[inside]
    xs = xs[inside]
    ys = ys[inside]
    heat_values = heat[ys, xs].astype(np.float32) / 255.0
    binary = wrinkle.get("binary")
    if binary is not None and np.asarray(binary).shape[:2] == heat.shape[:2]:
        binary_values = np.asarray(binary, dtype=np.uint8)[ys, xs] > 0
    else:
        binary_values = np.zeros((len(points_px),), dtype=bool)
    raw_threshold = float(wrinkle.get("threshold", 128.0))
    relaxed_threshold = float(np.clip(
        raw_threshold * float(getattr(args, "pose_waistband_heat_threshold_scale", 0.86)),
        24.0, 230.0,
    )) / 255.0
    hot = binary_values | (heat_values >= relaxed_threshold)

    rel = points_px - center_px
    lateral = np.asarray([axis[1], -axis[0]], dtype=np.float32)
    v = rel @ axis
    u = rel @ lateral
    v_min = float(np.percentile(v, 0.5))
    v_max = float(np.percentile(v, 99.5))
    v_span = v_max - v_min
    u_min = float(np.percentile(u, 0.5))
    u_max = float(np.percentile(u, 99.5))
    u_span = u_max - u_min
    if v_span < 40.0 or u_span < 35.0:
        result["reason"] = "waistband axis span too small"
        return result
    t = (v - v_min) / max(v_span, 1e-6)
    end_ratio = float(np.clip(getattr(
        args, "pose_waistband_heat_end_ratio", 0.30,
    ), 0.18, 0.42))
    longitudinal_bins = max(6, int(getattr(
        args, "pose_waistband_heat_longitudinal_bins", 12,
    )))
    lateral_bins = max(6, int(getattr(
        args, "pose_waistband_heat_lateral_bins", 12,
    )))
    min_bin_points = max(8, int(getattr(
        args, "pose_waistband_heat_min_bin_points", 18,
    )))

    def score_end(lo_t: float, hi_t: float) -> Dict[str, Any]:
        edges = np.linspace(float(lo_t), float(hi_t), longitudinal_bins + 1)
        records: List[Dict[str, float]] = []
        for bin_index in range(longitudinal_bins):
            lo = float(edges[bin_index])
            hi = float(edges[bin_index + 1])
            selected = (t >= lo) & (t <= hi if bin_index == longitudinal_bins - 1 else t < hi)
            count = int(np.count_nonzero(selected))
            if count < min_bin_points:
                continue
            selected_u = u[selected]
            selected_hot = hot[selected]
            selected_heat = heat_values[selected]
            local_u_lo = float(np.percentile(selected_u, 1.0))
            local_u_hi = float(np.percentile(selected_u, 99.0))
            local_u_span = local_u_hi - local_u_lo
            if local_u_span < 12.0:
                continue
            coverage = float(np.mean(selected_hot))
            strength = float(np.percentile(selected_heat, 80.0))
            occupancy = 0
            valid_lateral_bins = 0
            u_edges = np.linspace(local_u_lo, local_u_hi, lateral_bins + 1)
            for lateral_index in range(lateral_bins):
                ulo = float(u_edges[lateral_index])
                uhi = float(u_edges[lateral_index + 1])
                lateral_selected = (
                    (selected_u >= ulo)
                    & (selected_u <= uhi if lateral_index == lateral_bins - 1 else selected_u < uhi)
                )
                lateral_count = int(np.count_nonzero(lateral_selected))
                if lateral_count < 2:
                    continue
                valid_lateral_bins += 1
                hot_count = int(np.count_nonzero(selected_hot[lateral_selected]))
                required_hot = max(1, int(math.ceil(0.08 * lateral_count)))
                if hot_count >= required_hot:
                    occupancy += 1
            broad_coverage = float(occupancy) / max(1.0, float(valid_lateral_bins))
            sorted_u = np.sort(selected_u)
            split_gap = max(
                float(getattr(args, "pose_mask_geometry_min_gap_px", 6.0)),
                0.05 * local_u_span,
            )
            section_count = 1 + int(np.count_nonzero(np.diff(sorted_u) > split_gap))
            single_section = 1.0 if section_count == 1 else 0.0
            width_ratio = float(np.clip(local_u_span / max(u_span, 1.0), 0.0, 1.0))
            density_breadth = math.sqrt(max(0.0, coverage * broad_coverage))
            score = (
                0.50 * density_breadth
                + 0.20 * coverage
                + 0.15 * strength
                + 0.10 * single_section
                + 0.05 * width_ratio
            )
            records.append({
                "index": float(bin_index),
                "t": 0.5 * (lo + hi),
                "score": float(score),
                "coverage": float(coverage),
                "broad_coverage": float(broad_coverage),
                "strength": float(strength),
                "single_section": float(single_section),
                "width_ratio": float(width_ratio),
            })
        if not records:
            return {
                "score": 0.0, "peak_t": 0.5 * (lo_t + hi_t),
                "coverage": 0.0, "broad_coverage": 0.0,
                "strength": 0.0, "single_section": 0.0,
            }
        peak = max(records, key=lambda item: float(item["score"]))
        neighbors = [
            item for item in records
            if abs(float(item["index"]) - float(peak["index"])) <= 1.0
        ]
        neighbor_score = float(np.mean([
            float(item["score"]) for item in neighbors
        ]))
        end_score = 0.72 * float(peak["score"]) + 0.28 * neighbor_score
        return {
            "score": float(end_score),
            "peak_t": float(peak["t"]),
            "coverage": float(peak["coverage"]),
            "broad_coverage": float(peak["broad_coverage"]),
            "strength": float(peak["strength"]),
            "single_section": float(peak["single_section"]),
        }

    negative = score_end(0.01, end_ratio)
    positive = score_end(1.0 - end_ratio, 0.99)
    negative_score = float(negative["score"])
    positive_score = float(positive["score"])
    selected_end = "negative" if negative_score >= positive_score else "positive"
    selected = negative if selected_end == "negative" else positive
    selected_score = max(negative_score, positive_score)
    margin = abs(negative_score - positive_score)
    min_score = float(getattr(args, "pose_waistband_heat_min_score", 0.30))
    min_margin = float(getattr(args, "pose_waistband_heat_min_margin", 0.07))
    min_coverage = float(getattr(args, "pose_waistband_heat_min_coverage", 0.06))
    min_broad = float(getattr(args, "pose_waistband_heat_min_broad_coverage", 0.42))
    single_section_ok = bool(
        not getattr(args, "pose_waistband_heat_require_single_section", True)
        or float(selected["single_section"]) >= 0.5
    )
    reliable = bool(
        selected_score >= min_score
        and margin >= min_margin
        and float(selected["coverage"]) >= min_coverage
        and float(selected["broad_coverage"]) >= min_broad
        and single_section_ok
    )
    result.update({
        "available": True,
        "reliable": reliable,
        "selected_end": selected_end,
        "selected_score": float(selected_score),
        "negative_score": float(negative_score),
        "positive_score": float(positive_score),
        "score_margin": float(margin),
        "peak_t": float(selected["peak_t"]),
        "coverage": float(selected["coverage"]),
        "broad_coverage": float(selected["broad_coverage"]),
        "strength": float(selected["strength"]),
        "single_section": float(selected["single_section"]),
        "single_section_ok": bool(single_section_ok),
        "quality_reliable": bool(
            isinstance(wrinkle.get("quality"), dict)
            and wrinkle["quality"].get("reliable", False)
        ),
        "reason": "dense transverse waistband" if reliable else "waistband evidence weak or tied",
    })
    return result


def _estimate_crotch_from_directed_profile(
        samples: Dict[str, Any], profile: Optional[Dict[str, Any]],
        axis: np.ndarray, args,
) -> Tuple[np.ndarray, bool, Dict[str, float]]:
    """Estimate a crotch after waistband polarity is known; only a split is precise."""
    points_px = np.asarray(samples["points_px"], dtype=np.float32)
    center_px = np.asarray(samples["center_px"], dtype=np.float32)
    image_shape = tuple(samples["image_shape"])
    axis = safe_unit(axis)
    if axis is None:
        return center_px.copy(), False, {"split_t": 0.0, "split_gap_ratio": 0.0}
    profile_lateral = (
        None if profile is None else safe_unit(np.asarray(
            profile.get("lateral"), dtype=np.float32,
        ))
    )
    lateral = (
        profile_lateral if profile_lateral is not None
        else np.asarray([axis[1], -axis[0]], dtype=np.float32)
    )
    rel = points_px - center_px
    v = rel @ axis
    v_min = float(np.percentile(v, 0.5))
    v_max = float(np.percentile(v, 99.5))
    v_span = max(1e-6, v_max - v_min)
    split = None if profile is None else profile.get("first_stable_split")
    point_reliable = split is not None
    split_gap_ratio = 0.0
    if split is not None:
        split_t, left, right, split_gap_ratio = split
        crotch_u = 0.5 * (float(left[1]) + float(right[0]))
        crotch_t = float(split_t)
    else:
        crotch_t = float(np.clip(getattr(
            args, "pose_waistband_heat_crotch_center_t", 0.58,
        ), 0.42, 0.78))
        crotch_u = 0.0
    point = (
        center_px
        + axis * (v_min + crotch_t * v_span)
        + lateral * float(crotch_u)
    ).astype(np.float32)
    if len(image_shape) >= 2:
        point[0] = float(np.clip(point[0], 1.0, image_shape[1] - 2.0))
        point[1] = float(np.clip(point[1], 1.0, image_shape[0] - 2.0))
    return point, bool(point_reliable), {
        "split_t": float(crotch_t),
        "split_gap_ratio": float(split_gap_ratio),
    }



def estimate_closed_crotch_geometry(
        mask: Optional[BottomMaskBoard], samples: Optional[Dict[str, Any]],
        hypothesis: Optional[Dict[str, Any]], pose_hint_px: Optional[np.ndarray],
        pose_hint_conf: float, args,
) -> Optional[Dict[str, Any]]:
    """Infer a hidden crotch only after waist-to-hem polarity is reliable.

    This function never replaces an observed concavity or stable mask split.  It is
    used only when the waistband end is reliable but the leg opening is closed or
    occluded.  The estimate combines four weak, independent cues:
      * first sustained mask-width contraction from body to legs,
      * first stable two-ridge pattern in the mask distance transform,
      * a low-confidence pose crotch projected onto the directed body axis,
      * a configurable waist-to-hem longitudinal prior.

    The returned point is explicitly marked inferred and pre-spread remains required.
    """
    if not bool(getattr(args, "pose_closed_crotch_inference", True)):
        return None
    if mask is None or samples is None or not isinstance(hypothesis, dict):
        return None
    if bool(hypothesis.get("point_reliable", False)):
        return None
    if not bool(hypothesis.get("orientation_reliable", False)):
        return None

    axis = safe_unit(np.asarray(hypothesis.get("axis"), dtype=np.float32))
    if axis is None:
        return None
    points_px = np.asarray(samples.get("points_px"), dtype=np.float32)
    center_px = np.asarray(samples.get("center_px"), dtype=np.float32)
    image_shape = tuple(samples.get("image_shape", mask.mask_u8.shape[:2]))
    if points_px.ndim != 2 or points_px.shape[1] != 2 or len(points_px) < 500:
        return None

    lateral = np.asarray([axis[1], -axis[0]], dtype=np.float32)
    rel = points_px - center_px
    v = rel @ axis
    u = rel @ lateral
    v_min = float(np.percentile(v, 0.5))
    v_max = float(np.percentile(v, 99.5))
    v_span = v_max - v_min
    u_lo = float(np.percentile(u, 0.5))
    u_hi = float(np.percentile(u, 99.5))
    lateral_span = u_hi - u_lo
    if v_span < float(getattr(args, "pose_mask_geometry_min_axis_px", 80.0)):
        return None
    if lateral_span < float(getattr(args, "pose_mask_geometry_min_width_px", 60.0)):
        return None
    t = (v - v_min) / max(v_span, 1e-6)

    band = max(0.010, float(getattr(
        args, "pose_closed_crotch_profile_band_ratio", 0.022,
    )))
    sample_ts = np.linspace(0.22, 0.76, 28)
    width_records: List[Dict[str, float]] = []
    for sample_t in sample_ts:
        selected = np.asarray(u[np.abs(t - float(sample_t)) <= band], dtype=np.float32)
        if selected.size < max(10, int(getattr(
            args, "pose_mask_geometry_min_section_points", 12,
        ))):
            continue
        lo = float(np.percentile(selected, 2.0))
        hi = float(np.percentile(selected, 98.0))
        width_records.append({
            "t": float(sample_t),
            "lo": lo,
            "hi": hi,
            "width": max(0.0, hi - lo),
            "center_u": 0.5 * (lo + hi),
        })

    width_candidate_t = None
    width_candidate_score = 0.0
    if len(width_records) >= 8:
        body_widths = [
            rec["width"] for rec in width_records
            if 0.22 <= rec["t"] <= 0.38 and rec["width"] > 0.0
        ]
        body_ref = float(np.median(body_widths)) if body_widths else max(
            rec["width"] for rec in width_records
        )
        drop_ratio = float(np.clip(getattr(
            args, "pose_closed_crotch_width_drop_ratio", 0.86,
        ), 0.60, 0.98))
        min_drop = float(np.clip(getattr(
            args, "pose_closed_crotch_min_width_drop", 0.08,
        ), 0.02, 0.35))
        for rec_index in range(1, len(width_records) - 1):
            rec = width_records[rec_index]
            nxt = width_records[rec_index + 1]
            if rec["t"] < 0.28:
                continue
            ratio_now = rec["width"] / max(1.0, body_ref)
            ratio_next = nxt["width"] / max(1.0, body_ref)
            previous = width_records[rec_index - 1]["width"]
            derivative = (rec["width"] - previous) / max(1.0, body_ref)
            if ratio_now <= drop_ratio and ratio_next <= drop_ratio and derivative < -0.015:
                width_candidate_t = float(rec["t"])
                width_candidate_score = float(np.clip(
                    (1.0 - min(ratio_now, ratio_next) - min_drop)
                    / max(0.05, 1.0 - drop_ratio),
                    0.0, 1.0,
                ))
                break
        if width_candidate_t is None:
            gradients: List[Tuple[float, float]] = []
            for rec_index in range(1, len(width_records)):
                current = width_records[rec_index]
                previous = width_records[rec_index - 1]
                if not (0.28 <= current["t"] <= 0.70):
                    continue
                gradient = (current["width"] - previous["width"]) / max(1.0, body_ref)
                gradients.append((gradient, float(current["t"])))
            if gradients:
                gradient, gradient_t = min(gradients, key=lambda item: item[0])
                if gradient <= -min_drop:
                    width_candidate_t = gradient_t
                    width_candidate_score = float(np.clip(-gradient / 0.30, 0.0, 1.0))

    ridge_candidate_t = None
    ridge_candidate_u = 0.0
    ridge_candidate_score = 0.0
    ridge_records: List[Dict[str, float]] = []
    binary = (np.asarray(mask.mask_u8, dtype=np.uint8) > 0).astype(np.uint8)
    if binary.shape[:2] == image_shape[:2] and cv2.countNonZero(binary) > 100:
        distance = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
        grid_count = max(81, int(getattr(
            args, "pose_closed_crotch_ridge_grid_points", 161,
        )))
        if grid_count % 2 == 0:
            grid_count += 1
        u_grid = np.linspace(-0.50 * lateral_span, 0.50 * lateral_span, grid_count)
        peak_ratio = float(np.clip(getattr(
            args, "pose_closed_crotch_ridge_peak_ratio", 0.34,
        ), 0.15, 0.70))
        min_separation = float(getattr(
            args, "pose_closed_crotch_ridge_min_separation_ratio", 0.16,
        )) * lateral_span
        for sample_t in sample_ts:
            base = center_px + axis * (v_min + float(sample_t) * v_span)
            line_points = base[None, :] + u_grid[:, None] * lateral[None, :]
            xs = np.rint(line_points[:, 0]).astype(np.int32)
            ys = np.rint(line_points[:, 1]).astype(np.int32)
            valid = (
                (xs >= 0) & (xs < distance.shape[1])
                & (ys >= 0) & (ys < distance.shape[0])
            )
            signal = np.zeros((grid_count,), dtype=np.float32)
            signal[valid] = distance[ys[valid], xs[valid]]
            if float(np.max(signal)) < 2.0:
                continue
            smooth = np.convolve(signal, np.ones((7,), dtype=np.float32) / 7.0, mode="same")
            floor = peak_ratio * float(np.max(smooth))
            peaks = [
                index for index in range(2, grid_count - 2)
                if smooth[index] >= floor
                and smooth[index] >= smooth[index - 1]
                and smooth[index] >= smooth[index + 1]
            ]
            pairs: List[Tuple[float, int, int]] = []
            for left_index in peaks:
                for right_index in peaks:
                    if right_index <= left_index:
                        continue
                    separation = float(u_grid[right_index] - u_grid[left_index])
                    if separation < min_separation:
                        continue
                    if float(u_grid[left_index]) >= 0.0 or float(u_grid[right_index]) <= 0.0:
                        continue
                    valley = float(np.min(smooth[left_index:right_index + 1]))
                    weaker_peak = max(1e-6, min(
                        float(smooth[left_index]), float(smooth[right_index]),
                    ))
                    saddle_ratio = float(np.clip(valley / weaker_peak, 0.0, 1.0))
                    separation_score = float(np.clip(
                        separation / max(1.0, 0.42 * lateral_span), 0.0, 1.0,
                    ))
                    score = separation_score * (1.0 - saddle_ratio)
                    pairs.append((score, left_index, right_index))
            if not pairs:
                continue
            score, left_index, right_index = max(pairs, key=lambda item: item[0])
            ridge_records.append({
                "t": float(sample_t),
                "score": float(score),
                "center_u": 0.5 * float(u_grid[left_index] + u_grid[right_index]),
                "left_u": float(u_grid[left_index]),
                "right_u": float(u_grid[right_index]),
            })

        ridge_min_score = float(np.clip(getattr(
            args, "pose_closed_crotch_ridge_min_score", 0.18,
        ), 0.05, 0.70))
        for record_index in range(len(ridge_records) - 1):
            current = ridge_records[record_index]
            following = ridge_records[record_index + 1]
            if (
                current["score"] >= ridge_min_score
                and following["score"] >= ridge_min_score
                and following["t"] - current["t"] <= 0.035
            ):
                ridge_candidate_t = float(current["t"])
                ridge_candidate_u = float(current["center_u"])
                ridge_candidate_score = float(np.clip(
                    0.5 * (current["score"] + following["score"]) / 0.55,
                    0.0, 1.0,
                ))
                break

    pose_hint_t = None
    pose_hint_u = None
    pose_hint_quality = 0.0
    if pose_hint_px is not None:
        hint = np.asarray(pose_hint_px, dtype=np.float32).reshape(-1)
        if hint.shape == (2,) and np.all(np.isfinite(hint)):
            hint_rel = hint - center_px
            raw_t = (float(np.dot(hint_rel, axis)) - v_min) / max(v_span, 1e-6)
            if 0.20 <= raw_t <= 0.82:
                pose_hint_t = float(raw_t)
                pose_hint_u = float(np.dot(hint_rel, lateral))
                conf_floor = float(getattr(
                    args, "pose_closed_crotch_pose_hint_min_conf", 0.12,
                ))
                pose_hint_quality = float(np.clip(
                    (float(pose_hint_conf) - conf_floor) / max(0.10, 0.55 - conf_floor),
                    0.0, 1.0,
                ))

    prior_t = float(np.clip(getattr(
        args, "pose_closed_crotch_prior_t", 0.54,
    ), 0.38, 0.70))
    weighted_t: List[Tuple[float, float, str]] = [(prior_t, 0.14, "prior")]
    if width_candidate_t is not None:
        weighted_t.append((width_candidate_t, 0.28 + 0.18 * width_candidate_score, "width"))
    if ridge_candidate_t is not None:
        weighted_t.append((ridge_candidate_t, 0.40 + 0.28 * ridge_candidate_score, "ridge"))
    if pose_hint_t is not None:
        weighted_t.append((pose_hint_t, 0.08 + 0.18 * pose_hint_quality, "pose"))
    weight_sum = sum(item[1] for item in weighted_t)
    crotch_t = float(sum(item[0] * item[1] for item in weighted_t) / max(1e-6, weight_sum))
    crotch_t = float(np.clip(crotch_t, 0.34, 0.76))

    crotch_u = float(ridge_candidate_u if ridge_candidate_t is not None else 0.0)
    if pose_hint_u is not None and ridge_candidate_t is None:
        pose_u_limit = 0.18 * lateral_span
        crotch_u = float(np.clip(pose_hint_u, -pose_u_limit, pose_u_limit)) * (
            0.30 + 0.35 * pose_hint_quality
        )

    evidence = hypothesis.get("waistband_evidence") or {}
    selected_score = float(evidence.get("selected_score", 0.0))
    score_margin = float(evidence.get("score_margin", 0.0))
    min_score = float(getattr(args, "pose_waistband_heat_min_score", 0.30))
    orientation_score = float(np.clip(
        0.70 * (selected_score - min_score) / max(0.10, 1.0 - min_score)
        + 0.30 * score_margin / 0.25,
        0.0, 1.0,
    ))
    geometry_score = max(width_candidate_score, ridge_candidate_score)
    prior_agreement = float(np.clip(1.0 - abs(crotch_t - prior_t) / 0.24, 0.0, 1.0))
    confidence = float(np.clip(
        0.42 * orientation_score
        + 0.30 * geometry_score
        + 0.16 * pose_hint_quality
        + 0.12 * prior_agreement,
        0.0, 1.0,
    ))
    if width_candidate_t is None and ridge_candidate_t is None:
        confidence = min(confidence, float(getattr(
            args, "pose_closed_crotch_prior_only_conf_cap", 0.62,
        )))
    min_confidence = float(np.clip(getattr(
        args, "pose_closed_crotch_min_confidence", 0.50,
    ), 0.25, 0.85))

    point_px = (
        center_px
        + axis * (v_min + crotch_t * v_span)
        + lateral * crotch_u
    ).astype(np.float32)
    point_px[0] = float(np.clip(point_px[0], 1.0, image_shape[1] - 2.0))
    point_px[1] = float(np.clip(point_px[1], 1.0, image_shape[0] - 2.0))
    return {
        "usable": bool(confidence >= min_confidence),
        "state": "INFERRED_CLOSED_GEOMETRY",
        "point_px": point_px,
        "axis": axis,
        "lateral": lateral,
        "crotch_t": float(crotch_t),
        "crotch_u": float(crotch_u),
        "confidence": float(confidence),
        "minimum_confidence": float(min_confidence),
        "pre_spread_required": True,
        "observed": False,
        "waistband_score": selected_score,
        "waistband_margin": score_margin,
        "width_candidate_t": width_candidate_t,
        "width_candidate_score": float(width_candidate_score),
        "ridge_candidate_t": ridge_candidate_t,
        "ridge_candidate_u": float(ridge_candidate_u),
        "ridge_candidate_score": float(ridge_candidate_score),
        "pose_hint_t": pose_hint_t,
        "pose_hint_quality": float(pose_hint_quality),
        "prior_t": float(prior_t),
        "weighted_sources": [item[2] for item in weighted_t],
        "width_records": width_records,
        "ridge_records": ridge_records,
        "reason": (
            "waistband polarity + hidden leg-transition inference"
            if confidence >= min_confidence
            else "closed-crotch evidence below confidence threshold"
        ),
    }


def _waistband_first_axis_hypothesis(
        mask: Optional[BottomMaskBoard], samples: Optional[Dict[str, Any]],
        wrinkle: Optional[Dict[str, Any]], args, allow_weak: bool = False,
) -> Optional[Dict[str, Any]]:
    if samples is None:
        return None
    pca = _mask_pca_geometry_px(mask, args)
    rectangle = _mask_oriented_rectangle_px(mask, args)
    if pca is None and rectangle is None:
        return None
    points_px = np.asarray(samples["points_px"], dtype=np.float32)
    center_px = np.asarray(samples["center_px"], dtype=np.float32)
    hypotheses: List[Dict[str, Any]] = []
    axis_candidates: List[Tuple[str, np.ndarray, Optional[Dict[str, Any]]]] = []
    if rectangle is not None:
        for pair in rectangle["pair_axes"]:
            axis_candidates.append((
                f"rectangle-edge-pair-{int(pair['pair_index'])}",
                np.asarray(pair["axis"], dtype=np.float32),
                pair,
            ))
    if pca is not None:
        axis_candidates.extend([
            ("pca-major", np.asarray(pca["major_axis_px"], dtype=np.float32), None),
            ("pca-minor", np.asarray(pca["minor_axis_px"], dtype=np.float32), None),
        ])

    unique_axes: List[Tuple[str, np.ndarray, Optional[Dict[str, Any]]]] = []
    for axis_name, raw_axis, rectangle_pair in axis_candidates:
        raw_axis = safe_unit(np.asarray(raw_axis, dtype=np.float32))
        if raw_axis is None:
            continue
        if any(abs(float(np.dot(raw_axis, existing_axis))) >= 0.985
               for _, existing_axis, _ in unique_axes):
            continue
        unique_axes.append((axis_name, raw_axis, rectangle_pair))

    for axis_name, raw_axis, rectangle_pair in unique_axes:
        evidence = evaluate_waistband_heat_evidence(
            wrinkle, samples, raw_axis, args,
        )
        evidence_reliable = bool(evidence.get("reliable", False))
        weak_usable = bool(
            allow_weak
            and evidence.get("available", False)
            and float(evidence.get("selected_score", 0.0))
            >= float(getattr(args, "pose_rectangle_weak_heat_min_score", 0.16))
            and float(evidence.get("score_margin", 0.0))
            >= float(getattr(args, "pose_rectangle_weak_heat_min_margin", 0.015))
        )
        if not evidence_reliable and not weak_usable:
            continue
        # raw negative end is the waist -> raw axis already points waist-to-hem.
        directed_axis = raw_axis if evidence["selected_end"] == "negative" else -raw_axis
        profile = _mask_axis_profile(points_px, center_px, directed_axis, None, args)
        point_px, point_reliable, split_metrics = _estimate_crotch_from_directed_profile(
            samples, profile, directed_axis, args,
        )
        topology_bonus = 0.0
        if profile is not None:
            topology_bonus = (
                0.12 * float(profile.get("waist_single_rate", 0.0))
                + 0.10 * float(profile.get("hem_split_rate", 0.0))
                + (0.12 if profile.get("first_stable_split") is not None else 0.0)
            )
        score = (
            float(evidence.get("selected_score", 0.0))
            + 1.5 * float(evidence.get("score_margin", 0.0))
            + topology_bonus
            + (0.18 if rectangle_pair is not None else 0.0)
            + (0.10 if evidence_reliable else -0.10)
        )
        waist_edge_index = None
        hem_edge_index = None
        if rectangle_pair is not None:
            if evidence["selected_end"] == "negative":
                waist_edge_index = int(rectangle_pair["negative_edge_index"])
                hem_edge_index = int(rectangle_pair["positive_edge_index"])
            else:
                waist_edge_index = int(rectangle_pair["positive_edge_index"])
                hem_edge_index = int(rectangle_pair["negative_edge_index"])
        hypotheses.append({
            "score": float(score),
            "axis_name": axis_name,
            "axis": np.asarray(directed_axis, dtype=np.float32),
            "profile": profile,
            "point_px": point_px,
            "point_reliable": bool(point_reliable),
            "split_metrics": split_metrics,
            "waistband_evidence": evidence,
            "orientation_reliable": bool(evidence_reliable),
            "rectangle_pair": rectangle_pair,
            "waist_edge_index": waist_edge_index,
            "hem_edge_index": hem_edge_index,
        })
    if not hypotheses:
        return None
    best = max(hypotheses, key=lambda item: float(item["score"]))
    point_reliable = bool(best["point_reliable"])
    rectangle_selected = best.get("rectangle_pair") is not None
    orientation_reliable = bool(best.get("orientation_reliable", False))
    source_prefix = "rectangle-waistband" if rectangle_selected else "waistband"
    closed_crotch = None
    if not point_reliable and orientation_reliable:
        provisional = {
            "axis": np.asarray(best["axis"], dtype=np.float32),
            "point_reliable": False,
            "orientation_reliable": True,
            "waistband_evidence": best["waistband_evidence"],
        }
        closed_crotch = estimate_closed_crotch_geometry(
            mask, samples, provisional, None, 0.0, args,
        )
    selected_point = (
        np.asarray(closed_crotch["point_px"], dtype=np.float32)
        if isinstance(closed_crotch, dict) and closed_crotch.get("usable", False)
        else np.asarray(best["point_px"], dtype=np.float32)
    )
    return {
        "reliable": True,
        "orientation_reliable": orientation_reliable,
        "point_reliable": point_reliable,
        "reason": (
            f"{source_prefix} first with stable leg split"
            if point_reliable else f"{source_prefix} first; crotch remains estimated"
        ),
        "source": (
            f"{source_prefix}-guided-split"
            if point_reliable else f"{source_prefix}-first-pose"
        ),
        "point_px": selected_point,
        "axis": np.asarray(best["axis"], dtype=np.float32),
        "profile": best["profile"],
        "depth_px": 0.0,
        "chord_px": 0.0,
        "depth_chord_ratio": float(best["split_metrics"].get("split_gap_ratio", 0.0)),
        "stable_scale_count": 0,
        "score": float(best["score"]),
        "score_margin": float(best["waistband_evidence"].get("score_margin", 0.0)),
        "waistband_evidence": best["waistband_evidence"],
        "axis_name": str(best["axis_name"]),
        "crotch_occluded": not point_reliable,
        "closed_crotch": closed_crotch,
        "closed_crotch_inference_usable": bool(
            isinstance(closed_crotch, dict) and closed_crotch.get("usable", False)
        ),
        "rectangle": rectangle,
        "waist_edge_index": best.get("waist_edge_index"),
        "hem_edge_index": best.get("hem_edge_index"),
    }


def detect_stable_crotch_concavity(
        mask: Optional[BottomMaskBoard], samples: Optional[Dict[str, Any]], args,
        wrinkle: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Find the crotch as a deep concavity that survives contour simplification.

    Convexity defects are evaluated at several Douglas-Peucker scales. A true
    crotch remains near the same pixel while small folds, mask noise, and fabric
    texture disappear or move. The returned axis is directed from the opposite
    waist side toward the crotch/hem opening, so it also resolves 180-degree
    pose ambiguity without using image-up or image-down.
    """
    result: Dict[str, Any] = {
        "reliable": False,
        "point_reliable": False,
        "reason": "crotch gap not observable",
        "source": "unresolved",
        "point_px": None,
        "axis": None,
        "profile": None,
        "waistband_evidence": None,
        "raw_candidate_count": 0,
        "stable_scale_count": 0,
        "crotch_occluded": True,
    }
    if not bool(getattr(args, "pose_crotch_concavity", True)):
        result["reason"] = "crotch-first detector disabled"
        return result
    if mask is None or getattr(mask, "contour", None) is None:
        result["reason"] = "pants contour unavailable"
        return result

    contour = np.asarray(mask.contour)
    if contour.size < 12:
        result["reason"] = "pants contour too short"
        return result
    contour = np.rint(contour.reshape(-1, 1, 2)).astype(np.int32)
    try:
        perimeter = float(cv2.arcLength(contour, True))
        _bx, _by, bbox_w, bbox_h = cv2.boundingRect(contour)
    except Exception:
        result["reason"] = "pants contour geometry failed"
        return result
    min_span = max(1.0, float(min(bbox_w, bbox_h)))
    if perimeter < 20.0 or min_span < 20.0:
        result["reason"] = "pants contour too small"
        return result

    center_px = (
        np.asarray(samples["center_px"], dtype=np.float32)
        if samples is not None and samples.get("center_px") is not None
        else np.asarray(mask.center_px, dtype=np.float32)
    )
    image_shape = (
        tuple(samples["image_shape"])
        if samples is not None and samples.get("image_shape") is not None
        else tuple(np.asarray(mask.mask_u8).shape[:2])
    )
    rectangle_hypothesis = _waistband_first_axis_hypothesis(
        mask, samples, wrinkle, args,
    )
    if bool(
        isinstance(rectangle_hypothesis, dict)
        and str(rectangle_hypothesis.get("source", "")).startswith("rectangle-waistband")
        and rectangle_hypothesis.get("orientation_reliable", False)
        and rectangle_hypothesis.get("point_reliable", False)
    ):
        rectangle_hypothesis["raw_candidate_count"] = 0
        rectangle_hypothesis["epsilon_scale_count"] = 0
        rectangle_hypothesis["cluster_count"] = 0
        return rectangle_hypothesis
    epsilon_ratios = parse_float_csv(
        getattr(args, "pose_crotch_concavity_epsilons", "0.002,0.004,0.007,0.010"),
        [0.002, 0.004, 0.007, 0.010],
    )
    epsilon_ratios = sorted({
        float(np.clip(value, 0.0005, 0.03)) for value in epsilon_ratios
    })
    min_depth = max(
        float(getattr(args, "pose_crotch_concavity_min_depth_px", 12.0)),
        min_span * float(getattr(args, "pose_crotch_concavity_min_depth_ratio", 0.035)),
    )
    min_chord = max(
        float(getattr(args, "pose_crotch_concavity_min_chord_px", 42.0)),
        min_span * float(getattr(args, "pose_crotch_concavity_min_chord_ratio", 0.12)),
    )
    min_depth_chord = float(getattr(
        args, "pose_crotch_concavity_min_depth_chord_ratio", 0.08,
    ))
    min_inward_ratio = float(getattr(
        args, "pose_crotch_concavity_min_inward_ratio", 0.08,
    ))
    border_margin = max(0.0, float(getattr(
        args, "pose_crotch_concavity_border_margin_px", 8.0,
    )))
    entries: List[Dict[str, Any]] = []

    for scale_index, epsilon_ratio in enumerate(epsilon_ratios):
        try:
            approx = cv2.approxPolyDP(
                contour, max(0.75, perimeter * float(epsilon_ratio)), True,
            )
            if approx is None or len(approx) < 4:
                continue
            hull_indices = cv2.convexHull(approx, returnPoints=False)
            if hull_indices is None or len(hull_indices) < 3:
                continue
            defects = cv2.convexityDefects(approx, hull_indices)
        except Exception:
            continue
        if defects is None:
            continue
        vertices = np.asarray(approx, dtype=np.float32).reshape(-1, 2)
        for defect in np.asarray(defects).reshape(-1, 4):
            start_index, end_index, far_index, depth_fixed = [int(value) for value in defect]
            if not (0 <= start_index < len(vertices)
                    and 0 <= end_index < len(vertices)
                    and 0 <= far_index < len(vertices)):
                continue
            start = vertices[start_index]
            end = vertices[end_index]
            far = vertices[far_index]
            depth_px = float(depth_fixed) / 256.0
            chord_px = safe_norm(end - start)
            if depth_px < min_depth or chord_px < min_chord:
                continue
            depth_chord_ratio = depth_px / max(chord_px, 1e-6)
            if depth_chord_ratio < min_depth_chord:
                continue
            if len(image_shape) >= 2 and border_margin > 0.0:
                height, width = int(image_shape[0]), int(image_shape[1])
                if (float(far[0]) < border_margin
                        or float(far[1]) < border_margin
                        or float(far[0]) > float(width - 1) - border_margin
                        or float(far[1]) > float(height - 1) - border_margin):
                    continue

            chord_mid = 0.5 * (start + end)
            opening_axis = safe_unit(chord_mid - far)
            radial_axis = safe_unit(far - center_px)
            if opening_axis is None or radial_axis is None:
                continue
            radial_alignment = float(np.dot(opening_axis, radial_axis))
            inward_gain = safe_norm(chord_mid - center_px) - safe_norm(far - center_px)
            if radial_alignment <= 0.05 or inward_gain < min_inward_ratio * depth_px:
                continue
            combined_axis = safe_unit(0.70 * opening_axis + 0.30 * radial_axis)
            if combined_axis is None:
                continue
            corner_angle = angle_between_deg(start - far, end - far)
            if corner_angle is not None and corner_angle > 168.0:
                continue
            inward_score = float(np.clip(inward_gain / max(depth_px, 1.0), 0.0, 1.5))
            geometric_score = (
                8.0 * depth_px / min_span
                + 1.5 * chord_px / min_span
                + 2.0 * depth_chord_ratio
                + 0.8 * max(0.0, radial_alignment)
                + 0.5 * inward_score
            )
            entries.append({
                "scale_index": int(scale_index),
                "epsilon_ratio": float(epsilon_ratio),
                "point_px": far.astype(np.float32),
                "axis": combined_axis.astype(np.float32),
                "start_px": start.astype(np.float32),
                "end_px": end.astype(np.float32),
                "chord_mid_px": chord_mid.astype(np.float32),
                "depth_px": float(depth_px),
                "chord_px": float(chord_px),
                "depth_chord_ratio": float(depth_chord_ratio),
                "radial_alignment": float(radial_alignment),
                "inward_gain_px": float(inward_gain),
                "geometric_score": float(geometric_score),
            })

    result["raw_candidate_count"] = int(len(entries))
    result["epsilon_scale_count"] = int(len(epsilon_ratios))
    if not entries:
        waistband_hypothesis = _waistband_first_axis_hypothesis(
            mask, samples, wrinkle, args,
        )
        if waistband_hypothesis is not None:
            waistband_hypothesis["raw_candidate_count"] = 0
            waistband_hypothesis["epsilon_scale_count"] = int(len(epsilon_ratios))
            return waistband_hypothesis
        return result

    cluster_radius = max(
        float(getattr(args, "pose_crotch_concavity_cluster_px", 16.0)),
        min_span * float(getattr(args, "pose_crotch_concavity_cluster_ratio", 0.045)),
    )
    clusters: List[List[Dict[str, Any]]] = []
    for entry in sorted(entries, key=lambda item: float(item["depth_px"]), reverse=True):
        assigned = False
        for cluster in clusters:
            cluster_point = np.mean(np.stack(
                [np.asarray(item["point_px"], dtype=np.float32) for item in cluster], axis=0,
            ), axis=0)
            cluster_axis = safe_unit(np.mean(np.stack(
                [np.asarray(item["axis"], dtype=np.float32) for item in cluster], axis=0,
            ), axis=0))
            axis_dot = (
                float(np.dot(cluster_axis, np.asarray(entry["axis"], dtype=np.float32)))
                if cluster_axis is not None else -1.0
            )
            if safe_norm(np.asarray(entry["point_px"]) - cluster_point) <= cluster_radius and axis_dot >= 0.55:
                cluster.append(entry)
                assigned = True
                break
        if not assigned:
            clusters.append([entry])

    min_scales = max(2, int(getattr(
        args, "pose_crotch_concavity_min_stable_scales", 2,
    )))
    ranked_clusters: List[Dict[str, Any]] = []
    points_px = (
        np.asarray(samples["points_px"], dtype=np.float32)
        if samples is not None and samples.get("points_px") is not None else None
    )
    for cluster in clusters:
        scale_count = len({int(item["scale_index"]) for item in cluster})
        if scale_count < min_scales:
            continue
        weights = np.asarray([
            max(1.0, float(item["depth_px"])) for item in cluster
        ], dtype=np.float32)
        point_stack = np.stack([
            np.asarray(item["point_px"], dtype=np.float32) for item in cluster
        ], axis=0)
        axis_stack = np.stack([
            np.asarray(item["axis"], dtype=np.float32) for item in cluster
        ], axis=0)
        point_px = np.average(point_stack, axis=0, weights=weights).astype(np.float32)
        axis = safe_unit(np.average(axis_stack, axis=0, weights=weights))
        if axis is None:
            continue
        representative = max(cluster, key=lambda item: float(item["depth_px"]))
        depth_px = float(np.median([float(item["depth_px"]) for item in cluster]))
        chord_px = float(np.median([float(item["chord_px"]) for item in cluster]))
        depth_chord_ratio = depth_px / max(chord_px, 1e-6)
        stability_rate = float(scale_count) / max(1.0, float(len(epsilon_ratios)))
        radial_alignment = float(np.median([
            float(item["radial_alignment"]) for item in cluster
        ]))
        inward_gain = float(np.median([
            float(item["inward_gain_px"]) for item in cluster
        ]))
        profile = None
        topology_bonus = 0.0
        if points_px is not None:
            profile = _mask_axis_profile(points_px, center_px, axis, None, args)
            if profile is not None:
                topology_bonus = (
                    0.35 * float(profile.get("waist_single_rate", 0.0))
                    + 0.25 * float(profile.get("hem_split_rate", 0.0))
                    + (0.15 if profile.get("first_stable_split") is not None else 0.0)
                )
        geometric_score = (
            8.0 * depth_px / min_span
            + 2.0 * depth_chord_ratio
            + 1.5 * stability_rate
            + 0.8 * max(0.0, radial_alignment)
            + 0.5 * float(np.clip(inward_gain / max(depth_px, 1.0), 0.0, 1.5))
            + topology_bonus
        )
        waistband_evidence = evaluate_waistband_heat_evidence(
            wrinkle, samples, axis, args,
        )
        waistband_reliable = bool(waistband_evidence.get("reliable", False))
        waistband_compatible = bool(
            waistband_reliable
            and waistband_evidence.get("selected_end") == "negative"
        )
        waistband_adjustment = 0.0
        if waistband_reliable:
            margin = float(waistband_evidence.get("score_margin", 0.0))
            if waistband_compatible:
                waistband_adjustment = (
                    float(getattr(args, "pose_waistband_heat_crotch_bonus", 1.8))
                    + 3.0 * margin
                )
            else:
                waistband_adjustment = -(
                    float(getattr(args, "pose_waistband_heat_waist_penalty", 3.5))
                    + 3.0 * margin
                )
        score = geometric_score + waistband_adjustment
        ranked_clusters.append({
            "reliable": True,
            "point_reliable": True,
            "reason": (
                "stable crotch concavity opposite dense waistband"
                if waistband_compatible else "stable deep crotch concavity"
            ),
            "source": (
                "crotch+waistband" if waistband_compatible else "crotch-concavity"
            ),
            "point_px": point_px,
            "axis": axis.astype(np.float32),
            "chord_start_px": np.asarray(representative["start_px"], dtype=np.float32),
            "chord_end_px": np.asarray(representative["end_px"], dtype=np.float32),
            "chord_mid_px": np.asarray(representative["chord_mid_px"], dtype=np.float32),
            "depth_px": float(depth_px),
            "chord_px": float(chord_px),
            "depth_chord_ratio": float(depth_chord_ratio),
            "stability_rate": float(stability_rate),
            "stable_scale_count": int(scale_count),
            "score": float(score),
            "geometric_score": float(geometric_score),
            "waistband_adjustment": float(waistband_adjustment),
            "waistband_evidence": waistband_evidence,
            "waistband_compatible": bool(waistband_compatible),
            "profile": profile,
            "raw_candidate_count": int(len(entries)),
            "epsilon_scale_count": int(len(epsilon_ratios)),
            "cluster_count": int(len(clusters)),
            "crotch_occluded": False,
        })

    if not ranked_clusters:
        result["reason"] = "no concavity survives multiple contour scales"
        waistband_hypothesis = _waistband_first_axis_hypothesis(
            mask, samples, wrinkle, args,
        )
        if waistband_hypothesis is not None:
            waistband_hypothesis["raw_candidate_count"] = int(len(entries))
            waistband_hypothesis["epsilon_scale_count"] = int(len(epsilon_ratios))
            waistband_hypothesis["cluster_count"] = int(len(clusters))
            return waistband_hypothesis
        return result
    compatible_clusters = [
        item for item in ranked_clusters
        if bool(item.get("waistband_compatible", False))
    ]
    selection_pool = compatible_clusters or ranked_clusters
    if not compatible_clusters and any(
            bool((item.get("waistband_evidence") or {}).get("reliable", False))
            for item in ranked_clusters):
        waistband_hypothesis = _waistband_first_axis_hypothesis(
            mask, samples, wrinkle, args,
        )
        if waistband_hypothesis is not None:
            waistband_hypothesis["raw_candidate_count"] = int(len(entries))
            waistband_hypothesis["epsilon_scale_count"] = int(len(epsilon_ratios))
            waistband_hypothesis["cluster_count"] = int(len(clusters))
            return waistband_hypothesis
    selection_pool.sort(key=lambda item: float(item["score"]), reverse=True)
    best = dict(selection_pool[0])
    best["score_margin"] = float(
        float(best["score"]) - float(selection_pool[1]["score"])
        if len(selection_pool) > 1 else float(best["score"])
    )
    if len(selection_pool) > 1:
        second = selection_pool[1]
        depth_ratio = float(second["depth_px"]) / max(float(best["depth_px"]), 1e-6)
        ambiguous_margin = float(getattr(
            args, "pose_crotch_concavity_min_score_margin", 0.16,
        ))
        far_apart = safe_norm(
            np.asarray(best["point_px"]) - np.asarray(second["point_px"])
        ) > 1.25 * cluster_radius
        if far_apart and depth_ratio >= 0.88 and float(best["score_margin"]) < ambiguous_margin:
            best["reliable"] = False
            best["reason"] = "two similarly deep crotch concavities"
            best["crotch_occluded"] = True
    return best


def _mask_guided_pose_setup(
        mask: Optional[BottomMaskBoard], samples: Optional[Dict[str, Any]],
        args, fallback_angles: Sequence[float],
        wrinkle: Optional[Dict[str, Any]] = None,
) -> Tuple[List[float], Dict[str, Any]]:
    angle_offsets = parse_float_csv(
        getattr(args, "pose_mask_guided_angle_offsets", "0,-7,7"), [0.0],
    )
    info: Dict[str, Any] = {
        "source": "discrete",
        "directed_axis": None,
        "profile": None,
        "axis_name": None,
        "crotch_concavity": None,
        "angle_offsets": list(angle_offsets),
    }
    if samples is None:
        return [float(angle) for angle in fallback_angles], info

    angles: List[float] = []

    def append_directed_pair(canonical_rotation: float) -> None:
        for offset in angle_offsets:
            _append_unique_rotation(angles, canonical_rotation + float(offset))
            _append_unique_rotation(angles, canonical_rotation + 180.0 + float(offset))

    concavity = detect_stable_crotch_concavity(
        mask, samples, args, wrinkle=wrinkle,
    )
    info["crotch_concavity"] = concavity
    if bool(concavity.get("reliable", False)):
        directed_axis = safe_unit(np.asarray(concavity["axis"], dtype=np.float32))
        if directed_axis is not None:
            info.update({
                "source": str(concavity.get("source", "crotch-concavity")),
                "directed_axis": directed_axis,
                "profile": concavity.get("profile"),
                "axis_name": str(concavity.get(
                    "axis_name", "stable-crotch-concavity",
                )),
            })
            if bool(getattr(args, "pose_mask_guided_tta", True)):
                axis_angle = math.degrees(math.atan2(
                    float(directed_axis[1]), float(directed_axis[0]),
                ))
                append_directed_pair(axis_angle - 90.0)
                if bool(getattr(args, "pose_mask_guided_keep_discrete", False)):
                    for angle in fallback_angles:
                        _append_unique_rotation(angles, float(angle))
                return angles, info

    if not bool(getattr(args, "pose_mask_guided_tta", True)):
        return [float(angle) for angle in fallback_angles], info

    pca = _mask_pca_geometry_px(mask, args)
    if pca is None:
        return [float(angle) for angle in fallback_angles], info

    axis_candidates = [
        ("pca-major", np.asarray(pca["major_axis_px"], dtype=np.float32)),
        ("pca-minor", np.asarray(pca["minor_axis_px"], dtype=np.float32)),
    ]
    require_concavity = bool(
        getattr(args, "pose_crotch_concavity", True)
        and getattr(args, "pose_crotch_concavity_required", True)
    )
    ranked_profiles: List[Tuple[float, str, Dict[str, Any]]] = []
    if not require_concavity:
        for axis_name, axis in axis_candidates:
            profile = evaluate_bottom_mask_polarity(samples, axis, args)
            if profile is None:
                continue
            rank_score = (
                float(profile.get("score", 0.0))
                + min(3.0, float(profile.get("score_margin", 0.0)))
            )
            ranked_profiles.append((rank_score, axis_name, profile))
    ranked_profiles.sort(key=lambda item: item[0], reverse=True)

    if ranked_profiles:
        _, axis_name, profile = ranked_profiles[0]
        directed_axis = safe_unit(np.asarray(profile["axis"], dtype=np.float32))
        if directed_axis is not None:
            axis_angle = math.degrees(math.atan2(
                float(directed_axis[1]), float(directed_axis[0]),
            ))
            # cv2 image rotation subtracts this angle from an image-space vector.
            canonical_rotation = axis_angle - 90.0
            append_directed_pair(canonical_rotation)
            info.update({
                "source": "mask-topology",
                "directed_axis": directed_axis,
                "profile": profile,
                "axis_name": axis_name,
            })

            secondary_gap = max(0.0, float(getattr(
                args, "pose_mask_guided_secondary_score_gap", 0.75,
            )))
            if len(ranked_profiles) > 1:
                secondary_rank, _, secondary_profile = ranked_profiles[1]
                if ranked_profiles[0][0] - secondary_rank <= secondary_gap:
                    secondary_axis = safe_unit(np.asarray(
                        secondary_profile["axis"], dtype=np.float32,
                    ))
                    if secondary_axis is not None:
                        secondary_angle = math.degrees(math.atan2(
                            float(secondary_axis[1]), float(secondary_axis[0]),
                        )) - 90.0
                        append_directed_pair(secondary_angle)
    else:
        # The axis is still useful for exact-angle deskewing even when waist/hem
        # topology is ambiguous. Both axis assignments and both polarities are kept.
        for _, axis in axis_candidates:
            axis = safe_unit(axis)
            if axis is None:
                continue
            axis_angle = math.degrees(math.atan2(float(axis[1]), float(axis[0])))
            canonical_rotation = axis_angle - 90.0
            _append_unique_rotation(angles, canonical_rotation)
            _append_unique_rotation(angles, canonical_rotation + 180.0)
        info["source"] = (
            "crotch-occluded-axis-undirected"
            if require_concavity else "mask-axis-undirected"
        )

    if bool(getattr(args, "pose_mask_guided_keep_discrete", False)):
        for angle in fallback_angles:
            _append_unique_rotation(angles, float(angle))
    if not angles:
        angles = [float(angle) for angle in fallback_angles]
    return angles, info


def _axis_distance_from_cardinal_deg(axis: Optional[np.ndarray]) -> float:
    unit = None if axis is None else safe_unit(axis)
    if unit is None:
        return 0.0
    angle = math.degrees(math.atan2(float(unit[1]), float(unit[0])))
    nearest_cardinal = round(angle / 90.0) * 90.0
    return abs(_wrap_rotation_deg(angle - nearest_cardinal))


def _apply_diagonal_mask_primary_landmarks(
        pose_xy: np.ndarray, pose_cf: np.ndarray,
        samples: Optional[Dict[str, Any]], profile: Optional[Dict[str, Any]],
        directed_axis: Optional[np.ndarray], concavity: Optional[Dict[str, Any]], args,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any], bool]:
    metrics: Dict[str, Any] = {"applied": False}
    if not bool(getattr(args, "pose_diagonal_mask_primary", True)):
        return pose_xy, pose_cf, metrics, False
    if samples is None or directed_axis is None:
        return pose_xy, pose_cf, metrics, False
    concavity_reliable = bool(
        isinstance(concavity, dict) and concavity.get("reliable", False)
    )
    concavity_point_reliable = bool(
        concavity_reliable and concavity.get("point_reliable", True)
    )
    if not concavity_point_reliable and not _mask_profile_reliable(profile, args):
        return pose_xy, pose_cf, metrics, False

    diagonal_deg = _axis_distance_from_cardinal_deg(directed_axis)
    min_diagonal = max(0.0, float(getattr(
        args, "pose_diagonal_mask_primary_min_angle_deg", 12.0,
    )))
    if diagonal_deg < min_diagonal:
        metrics.update({"diagonal_deg": diagonal_deg, "reason": "near-cardinal"})
        return pose_xy, pose_cf, metrics, False

    predicted_side = None
    if len(pose_xy) >= 3:
        predicted_side = (
            np.asarray(pose_xy[2], dtype=np.float32)
            - np.asarray(pose_xy[0], dtype=np.float32)
        )
    if concavity_point_reliable:
        recovered = mask_landmarks_from_crotch_concavity(
            samples, concavity, args,
        )
        if recovered is None and _mask_profile_reliable(profile, args):
            recovered = mask_landmarks_from_polarity(
                samples, profile, predicted_side, args,
            )
    else:
        recovered = mask_landmarks_from_polarity(
            samples, profile, predicted_side, args,
        )
    if recovered is None:
        metrics.update({"diagonal_deg": diagonal_deg, "reason": "mask-landmarks-failed"})
        return pose_xy, pose_cf, metrics, False

    mask_xy, mask_metrics = recovered
    mask_cf = np.full(
        (len(BOTTOM_POSE_KPT_NAMES),),
        float(getattr(args, "pose_mask_geometry_fallback_conf", 0.45)),
        dtype=np.float32,
    )
    mask_xy, mask_cf = _canonicalize_pose_lateral_order(
        mask_xy, mask_cf, directed_axis,
    )
    out_xy, out_cf = _canonicalize_pose_lateral_order(
        pose_xy, pose_cf, directed_axis,
    )
    if len(out_xy) < 8 or len(mask_xy) < 8:
        return pose_xy, pose_cf, metrics, False

    boundary_weight = float(np.clip(getattr(
        args, "pose_diagonal_mask_boundary_weight", 0.92,
    ), 0.0, 1.0))
    center_weight = float(np.clip(getattr(
        args, "pose_diagonal_mask_center_weight", 0.78,
    ), 0.0, 1.0))
    crotch_weight = float(np.clip(getattr(
        args, "pose_diagonal_mask_crotch_weight", 0.88,
    ), 0.0, 1.0))
    max_blend_distance = max(1.0, float(getattr(
        args, "pose_diagonal_mask_max_blend_distance_px", 65.0,
    )))
    weights_by_index = {
        0: boundary_weight,
        1: center_weight,
        2: boundary_weight,
        3: crotch_weight,
        4: boundary_weight,
        5: boundary_weight,
        6: boundary_weight,
        7: boundary_weight,
    }
    source_xy = np.asarray(out_xy, dtype=np.float32).copy()
    source_cf = np.asarray(out_cf, dtype=np.float32).copy()
    discrepancies: List[float] = []
    applied_shifts: List[float] = []
    used_weights: List[float] = []
    for index in range(8):
        discrepancy = safe_norm(mask_xy[index] - source_xy[index])
        weight = float(weights_by_index[index])
        # A distant pose point is treated as a systematic hallucination rather
        # than allowed to pull a strong silhouette landmark off the garment.
        if discrepancy > max_blend_distance:
            weight = 1.0
        out_xy[index] = (
            (1.0 - weight) * source_xy[index]
            + weight * np.asarray(mask_xy[index], dtype=np.float32)
        )
        pose_conf = float(source_cf[index]) if index < len(source_cf) else 0.0
        out_cf[index] = max(float(mask_cf[index]), min(0.85, pose_conf))
        discrepancies.append(float(discrepancy))
        applied_shifts.append(safe_norm(out_xy[index] - source_xy[index]))
        used_weights.append(weight)

    out_xy, out_cf = _canonicalize_pose_lateral_order(
        out_xy, out_cf, directed_axis,
    )
    metrics.update({
        "applied": True,
        "source": str(mask_metrics.get("source", "mask-topology")),
        "diagonal_deg": float(diagonal_deg),
        "mean_pose_mask_discrepancy_px": float(np.mean(discrepancies)),
        "max_pose_mask_discrepancy_px": float(np.max(discrepancies)),
        "mean_applied_shift_px": float(np.mean(applied_shifts)),
        "max_applied_shift_px": float(np.max(applied_shifts)),
        "weights": used_weights,
        "split_t": float(mask_metrics.get("split_t", 0.0)),
        "split_gap_ratio": float(mask_metrics.get("split_gap_ratio", 0.0)),
        "left_hem_t": float(mask_metrics.get("left_hem_t", 0.0)),
        "right_hem_t": float(mask_metrics.get("right_hem_t", 0.0)),
        "crotch_depth_px": float(mask_metrics.get("crotch_depth_px", 0.0)),
        "crotch_stable_scales": int(mask_metrics.get("crotch_stable_scales", 0)),
    })
    return out_xy, out_cf, metrics, True


def _temporally_stabilize_pose(
        kxy: np.ndarray, kcf: np.ndarray, mask: Optional[BottomMaskBoard],
        image_shape, min_conf: float, args,
) -> Tuple[np.ndarray, np.ndarray, int, bool]:
    if not bool(getattr(args, "pose_temporal_stabilization", True)) or mask is None:
        return kxy, kcf, 1, False

    lock = getattr(args, "_pose_semantic_history_lock", None)
    if lock is None:
        lock = threading.Lock()
        setattr(args, "_pose_semantic_history_lock", lock)
    now = time.monotonic()
    center = np.asarray(mask.center_px, dtype=np.float32)
    area = max(1.0, float(mask.area_px))

    with lock:
        history = list(getattr(args, "_pose_semantic_history", []))
        max_age = max(0.2, float(getattr(args, "pose_temporal_max_age_s", 4.0)))
        history = [item for item in history if now - float(item["time"]) <= max_age]
        reset = False
        if history:
            previous = history[-1]
            center_shift = safe_norm(center - np.asarray(previous["center"], dtype=np.float32))
            area_change = abs(area - float(previous["area"])) / max(area, float(previous["area"]), 1.0)
            common = min(len(kxy), len(previous["kxy"]), 8)
            point_shift = float(np.median(np.linalg.norm(
                np.asarray(kxy[:common], dtype=np.float32)
                - np.asarray(previous["kxy"][:common], dtype=np.float32),
                axis=1,
            ))) if common else float("inf")
            reset = bool(
                center_shift > float(getattr(args, "pose_temporal_reset_center_px", 80.0))
                or area_change > float(getattr(args, "pose_temporal_reset_area_ratio", 0.35))
                or point_shift > float(getattr(args, "pose_temporal_max_keypoint_shift_px", 35.0))
            )
        if reset:
            history = []
        history.append({
            "time": now,
            "center": center.copy(),
            "area": area,
            "kxy": np.asarray(kxy, dtype=np.float32).copy(),
            "kcf": np.asarray(kcf, dtype=np.float32).copy(),
        })
        history_limit = max(1, int(getattr(args, "pose_temporal_history", 3)))
        history = history[-history_limit:]
        setattr(args, "_pose_semantic_history", history)

        if len(history) <= 1:
            return kxy, kcf, len(history), reset
        temporal_candidates = [
            {
                "kxy": item["kxy"],
                "kcf": item["kcf"],
                "summary": {"score": 0.0},
            }
            for item in history
        ]
        fused_xy, fused_cf, _, _ = _fuse_pose_candidates(
            temporal_candidates, image_shape, min_conf, args,
        )
        return fused_xy, fused_cf, len(history), reset


def infer_best_pose_with_tta(pose_model, frame: np.ndarray, imgsz: int, conf: float,
                             kpt_conf: float, args,
                             mask: Optional[BottomMaskBoard] = None,
                             wrinkle: Optional[Dict[str, Any]] = None,
                             ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], str, Dict[str, Any]]:
    fallback_angles = parse_float_csv(
        getattr(args, "pose_tta_angles", "0,180,90,-90,45,-45"), [0.0],
    )
    flip_modes = parse_flip_modes(getattr(args, "pose_tta_flips", "none,h"))
    mask_polarity_samples = None
    if bool(getattr(args, "pose_mask_polarity_check", True)):
        mask_polarity_samples = prepare_bottom_mask_polarity(mask, args)
    angles, guided = _mask_guided_pose_setup(
        mask, mask_polarity_samples, args, fallback_angles, wrinkle=wrinkle,
    )
    concavity = guided.get("crotch_concavity")
    concavity_reliable = bool(
        isinstance(concavity, dict) and concavity.get("reliable", False)
    )
    semantic_source = (
        str(concavity.get("source", "unresolved"))
        if isinstance(concavity, dict) else "unresolved"
    )
    waistband_evidence = (
        concavity.get("waistband_evidence")
        if isinstance(concavity, dict) else None
    )
    waistband_reliable = bool(
        isinstance(waistband_evidence, dict)
        and waistband_evidence.get("reliable", False)
    )
    closed_crotch = (
        concavity.get("closed_crotch")
        if isinstance(concavity, dict) else None
    )
    closed_crotch_usable = bool(
        isinstance(closed_crotch, dict) and closed_crotch.get("usable", False)
    )
    strict_crotch_unresolved = bool(
        getattr(args, "pose_crotch_concavity", True)
        and getattr(args, "pose_crotch_concavity_required", True)
        and not concavity_reliable
    )
    candidates: List[Dict[str, Any]] = []
    for angle in angles:
        rotated, rot_inv, view_meta = prepare_mask_guided_pose_view(
            frame, mask, float(angle), args,
        )
        for flip_mode in flip_modes:
            work_img = rotated
            Tinv = rot_inv
            do_hflip = flip_mode == "h"
            if do_hflip:
                work_img, flip_inv = hflip_image_and_inverse(rotated)
                Tinv = rot_inv @ flip_inv
            try:
                result = pose_model.predict(source=work_img, imgsz=int(imgsz), conf=float(conf), verbose=False)[0]
            except Exception:
                continue
            kxy_aug, kcf, status = read_pose_keypoints(result, work_img.shape, kpt_conf)
            if kxy_aug is None or kcf is None:
                continue
            kxy_orig = transform_keypoints(kxy_aug, Tinv)
            if do_hflip:
                kxy_orig, kcf = remap_hflip_keypoints(kxy_orig, kcf)
                if kcf is None:
                    kcf = np.ones((kxy_orig.shape[0],), dtype=np.float32)
            summary = pose_tta_summary(
                kxy_orig, kcf, kpt_conf, frame.shape, args, mask=mask,
            )
            summary["mask_polarity_dot"] = None
            summary["mask_polarity_profile"] = None
            predicted_axis = _pose_candidate_axis(
                kxy_orig, kcf, kpt_conf, frame.shape,
            )
            directed_axis = guided.get("directed_axis")
            if predicted_axis is not None and directed_axis is not None:
                summary["mask_polarity_dot"] = float(np.dot(
                    predicted_axis, np.asarray(directed_axis, dtype=np.float32),
                ))
                summary["mask_polarity_profile"] = guided.get("profile")
            elif (
                not strict_crotch_unresolved
                and mask_polarity_samples is not None
                and predicted_axis is not None
            ):
                polarity = evaluate_bottom_mask_polarity(
                    mask_polarity_samples, predicted_axis, args,
                )
                if polarity is not None:
                    summary["mask_polarity_dot"] = float(polarity["dot"])
                    summary["mask_polarity_profile"] = polarity
            cand = {
                "kxy": kxy_orig,
                "kcf": kcf,
                "angle": float(angle),
                "flip": flip_mode,
                "view_meta": dict(view_meta),
                "summary": summary,
            }
            candidates.append(cand)
    if not candidates:
        return None, None, "TTA no pose instances", {}

    reliable_candidates = [
        candidate for candidate in candidates
        if bool(candidate["summary"].get("structure_reliable", False))
    ]
    base_pool = reliable_candidates if reliable_candidates else candidates
    selection_status = "geometry-filtered" if reliable_candidates else "geometry-unresolved"
    aligned_dot = float(getattr(args, "pose_mask_polarity_min_dot", 0.25))
    aligned = [
        candidate for candidate in base_pool
        if candidate["summary"].get("mask_polarity_dot") is not None
        and float(candidate["summary"]["mask_polarity_dot"]) >= aligned_dot
        and int(candidate["summary"].get("visible_count", 0)) >= 7
        and int(candidate["summary"].get("waist_count", 0)) == 3
        and bool(candidate["summary"].get("crotch_visible", False))
        and int(candidate["summary"].get("hem_count", 0)) >= 3
    ]

    semantic_axis = guided.get("directed_axis")
    if semantic_axis is None and aligned:
        semantic_axis = aligned[0]["summary"].get("mask_polarity_profile", {}).get("axis")
    if semantic_axis is not None:
        for candidate in aligned:
            candidate["kxy"], candidate["kcf"] = _canonicalize_pose_lateral_order(
                candidate["kxy"], candidate["kcf"], semantic_axis,
            )

    best_base = max(base_pool, key=lambda candidate: float(candidate["summary"]["score"]))
    consensus_metrics = {
        "candidate_count": 0,
        "mean_dispersion_px": float("inf"),
        "max_dispersion_px": float("inf"),
    }
    polarity_status = ""
    semantic_reliable = False
    selected_for_margin = aligned if aligned else base_pool

    if aligned:
        fused_xy, fused_cf, consensus_metrics, fused_candidates = _fuse_pose_candidates(
            aligned, frame.shape, kpt_conf, args,
        )
        fused_xy, fused_cf = _canonicalize_pose_lateral_order(
            fused_xy, fused_cf, semantic_axis,
        )
        min_consensus = max(1, int(getattr(args, "pose_canonical_min_consensus", 2)))
        profile = guided.get("profile")
        if profile is None and fused_candidates:
            profile = fused_candidates[0]["summary"].get("mask_polarity_profile")
        profile_margin = (
            float(profile.get("score_margin", 0.0)) if isinstance(profile, dict) else 0.0
        )
        strong_profile = bool(
            concavity_reliable
            or profile_margin >= float(getattr(
                args, "pose_canonical_strong_profile_margin", 1.50,
            ))
        )
        agreement_ok = bool(
            float(consensus_metrics["mean_dispersion_px"])
            <= float(getattr(args, "pose_canonical_max_mean_dispersion_px", 35.0))
            and float(consensus_metrics["max_dispersion_px"])
            <= float(getattr(args, "pose_canonical_max_point_dispersion_px", 55.0))
        )
        semantic_reliable = bool(
            agreement_ok
            and (int(consensus_metrics["candidate_count"]) >= min_consensus or strong_profile)
        )
        fused_xy, fused_cf, mask_primary_metrics, mask_primary_applied = (
            _apply_diagonal_mask_primary_landmarks(
                fused_xy,
                fused_cf,
                mask_polarity_samples,
                profile,
                semantic_axis,
                concavity,
                args,
            )
        )
        if mask_primary_applied:
            semantic_reliable = True
        closed_crotch_applied = False
        if closed_crotch_usable and not bool(concavity.get("point_reliable", False)):
            pose_crotch_conf = float(fused_cf[3]) if len(fused_cf) > 3 else 0.0
            replace_max_conf = float(getattr(
                args, "pose_closed_crotch_replace_max_pose_conf", 0.38,
            ))
            if len(fused_xy) > 3 and pose_crotch_conf <= replace_max_conf:
                blend = float(np.clip(getattr(
                    args, "pose_closed_crotch_blend_weight", 0.72,
                ), 0.50, 0.95))
                inferred_point = np.asarray(closed_crotch["point_px"], dtype=np.float32)
                fused_xy[3] = (
                    blend * inferred_point
                    + (1.0 - blend) * np.asarray(fused_xy[3], dtype=np.float32)
                ).astype(np.float32)
                fused_cf[3] = max(
                    pose_crotch_conf,
                    float(closed_crotch.get("confidence", 0.0))
                    * float(getattr(args, "pose_closed_crotch_conf_scale", 0.58)),
                )
                closed_crotch_applied = True
        fused_xy, fused_cf, temporal_count, temporal_reset = _temporally_stabilize_pose(
            fused_xy, fused_cf, mask, frame.shape, kpt_conf, args,
        )
        fused_summary = pose_tta_summary(
            fused_xy, fused_cf, kpt_conf, frame.shape, args, mask=mask,
        )
        fused_summary.update({
            "mask_polarity_dot": float(np.mean([
                float(candidate["summary"]["mask_polarity_dot"])
                for candidate in fused_candidates
            ])),
            "mask_polarity_profile": profile,
            "semantic_consensus": dict(consensus_metrics),
            "semantic_consensus_reliable": semantic_reliable,
            "temporal_count": int(temporal_count),
            "temporal_reset": bool(temporal_reset),
            "canonical_source": (
                f"{semantic_source}-mask-primary"
                if mask_primary_applied else guided.get("source")
            ),
            "mask_primary": dict(mask_primary_metrics),
            "crotch_concavity": concavity,
            "crotch_occluded": bool(
                isinstance(concavity, dict) and concavity.get("crotch_occluded", False)
            ),
            "closed_crotch_inference": closed_crotch,
            "closed_crotch_inferred": bool(closed_crotch_applied),
            "closed_crotch_usable": bool(closed_crotch_usable),
            "crotch_state": (
                "INFERRED_CLOSED_GEOMETRY" if closed_crotch_applied
                else (
                    "OBSERVED_CONCAVITY"
                    if bool(concavity_reliable and concavity.get("point_reliable", False))
                    and not str(semantic_source).startswith("waistband")
                    else (
                        "OBSERVED_MASK_SPLIT"
                        if bool(concavity_reliable and concavity.get("point_reliable", False))
                        else "POSE_CLOSED_CROTCH"
                    )
                )
            ),
            "crotch_confidence": float(
                closed_crotch.get("confidence", 0.0)
                if closed_crotch_applied else (fused_cf[3] if len(fused_cf) > 3 else 0.0)
            ),
            "pre_spread_required": bool(
                isinstance(concavity, dict) and concavity.get("crotch_occluded", False)
            ),
        })
        best = {
            "kxy": fused_xy,
            "kcf": fused_cf,
            "angle": float(fused_candidates[0]["angle"]),
            "flip": "consensus",
            "view_meta": dict(fused_candidates[0].get("view_meta", {})),
            "summary": fused_summary,
        }
        if semantic_source.startswith("waistband"):
            polarity_status = (
                "polarity=WAISTBAND-FIRST("
                f"score={float(waistband_evidence.get('selected_score', 0.0)):.2f},"
                f"margin={float(waistband_evidence.get('score_margin', 0.0)):.2f},"
                f"crotch={'split' if bool(concavity.get('point_reliable', False)) else 'pose'},"
                f"n={int(consensus_metrics['candidate_count'])})"
            )
        elif waistband_reliable:
            polarity_status = (
                "polarity=CROTCH+WAISTBAND("
                f"depth={float(concavity.get('depth_px', 0.0)):.1f}px,"
                f"waist={float(waistband_evidence.get('selected_score', 0.0)):.2f},"
                f"margin={float(waistband_evidence.get('score_margin', 0.0)):.2f})"
            )
        elif mask_primary_applied and str(mask_primary_metrics.get("source")) == "crotch-concavity":
            polarity_status = (
                "polarity=CROTCH-FIRST("
                f"depth={float(mask_primary_metrics.get('crotch_depth_px', 0.0)):.1f}px,"
                f"stable={int(mask_primary_metrics.get('crotch_stable_scales', 0))},"
                f"shift={float(mask_primary_metrics['mean_applied_shift_px']):.1f}px)"
            )
        elif mask_primary_applied:
            polarity_status = (
                "polarity=MASK-PRIMARY("
                f"diag={float(mask_primary_metrics['diagonal_deg']):.1f}deg,"
                f"shift={float(mask_primary_metrics['mean_applied_shift_px']):.1f}px)"
            )
        elif concavity_reliable:
            polarity_status = (
                "polarity=CROTCH-FIRST-CONSENSUS("
                f"depth={float(concavity.get('depth_px', 0.0)):.1f}px,"
                f"stable={int(concavity.get('stable_scale_count', 0))},"
                f"n={int(consensus_metrics['candidate_count'])})"
            )
        else:
            polarity_status = (
                f"polarity={'CONSENSUS' if semantic_reliable else 'UNRESOLVED'}("
                f"n={int(consensus_metrics['candidate_count'])},"
                f"mean={float(consensus_metrics['mean_dispersion_px']):.1f}px,"
                f"max={float(consensus_metrics['max_dispersion_px']):.1f}px)"
            )
    else:
        profile = guided.get("profile")
        if profile is None:
            profile = best_base["summary"].get("mask_polarity_profile")
        predicted_side = None
        if len(best_base["kxy"]) >= 3:
            predicted_side = (
                np.asarray(best_base["kxy"][2], dtype=np.float32)
                - np.asarray(best_base["kxy"][0], dtype=np.float32)
            )
        recovered = None
        recovery_source = ""
        if bool(getattr(args, "pose_canonical_mask_fallback", True)):
            if concavity_reliable and bool(concavity.get("point_reliable", True)):
                recovered = mask_landmarks_from_crotch_concavity(
                    mask_polarity_samples, concavity, args,
                )
                if recovered is not None:
                    recovery_source = semantic_source
            if (
                recovered is None
                and closed_crotch_usable
                and bool(getattr(args, "pose_closed_crotch_mask_fallback", True))
            ):
                pose_hint_px = (
                    np.asarray(best_base["kxy"][3], dtype=np.float32)
                    if len(best_base["kxy"]) > 3 else None
                )
                pose_hint_conf = (
                    float(best_base["kcf"][3])
                    if len(best_base["kcf"]) > 3 else 0.0
                )
                recovered = mask_landmarks_from_closed_crotch(
                    mask, mask_polarity_samples, concavity,
                    pose_hint_px, pose_hint_conf, args,
                )
                if recovered is not None:
                    recovery_source = "closed-crotch-polarity"
            if recovered is None and not strict_crotch_unresolved:
                recovered = mask_landmarks_from_polarity(
                    mask_polarity_samples, profile, predicted_side, args,
                )
                if recovered is not None:
                    recovery_source = "mask-topology"
        if recovered is not None and semantic_axis is not None:
            recovered_kxy, recovered_metrics = recovered
            recovered_conf = np.full(
                (len(BOTTOM_POSE_KPT_NAMES),),
                float(getattr(args, "pose_mask_geometry_fallback_conf", 0.45)),
                dtype=np.float32,
            )
            recovered_kxy, recovered_conf = _canonicalize_pose_lateral_order(
                recovered_kxy, recovered_conf, semantic_axis,
            )
            recovered_summary = pose_tta_summary(
                recovered_kxy, recovered_conf, kpt_conf, frame.shape,
                args, mask=mask,
            )
            recovered_summary.update({
                "mask_polarity_dot": 1.0,
                "mask_polarity_profile": profile,
                "semantic_consensus": dict(consensus_metrics),
                "semantic_consensus_reliable": True,
                "temporal_count": 1,
                "temporal_reset": False,
                "canonical_source": (
                    f"{recovery_source}-mask-fallback"
                    if recovery_source != "mask-topology" else "mask-landmarks"
                ),
                "crotch_concavity": concavity,
                "crotch_occluded": bool(recovered_metrics.get("crotch_occluded", False)),
                "closed_crotch_inference": recovered_metrics.get(
                    "closed_crotch", closed_crotch,
                ),
                "closed_crotch_inferred": bool(recovered_metrics.get("crotch_inferred", False)),
                "closed_crotch_usable": bool(
                    recovered_metrics.get("crotch_inferred", False)
                    and float(recovered_metrics.get("crotch_confidence", 0.0))
                    >= float(getattr(args, "pose_closed_crotch_min_confidence", 0.50))
                ),
                "crotch_state": str(recovered_metrics.get(
                    "crotch_state",
                    "INFERRED_CLOSED_GEOMETRY"
                    if recovery_source == "closed-crotch-polarity" else "OBSERVED_MASK_SPLIT",
                )),
                "crotch_confidence": float(recovered_metrics.get(
                    "crotch_confidence",
                    closed_crotch.get("confidence", 0.0)
                    if isinstance(closed_crotch, dict) else 0.0,
                )),
                "pre_spread_required": bool(recovered_metrics.get(
                    "pre_spread_required", False,
                )),
            })
            best = {
                "kxy": recovered_kxy,
                "kcf": recovered_conf,
                "angle": float(best_base["angle"]),
                "flip": "mask",
                "view_meta": dict(best_base.get("view_meta", {})),
                "summary": recovered_summary,
            }
            semantic_reliable = True
            if recovery_source == "closed-crotch-polarity":
                polarity_status = (
                    "polarity=CLOSED-CROTCH-FALLBACK("
                    f"conf={float(recovered_metrics.get('crotch_confidence', 0.0)):.2f},"
                    f"t={float(recovered_metrics.get('split_t', 0.0)):.2f},"
                    f"hems={float(recovered_metrics.get('left_hem_t', 0.0)):.2f}/"
                    f"{float(recovered_metrics.get('right_hem_t', 0.0)):.2f})"
                )
            elif recovery_source.startswith("waistband"):
                polarity_status = (
                    "polarity=WAISTBAND-FIRST-FALLBACK("
                    f"waist={float(waistband_evidence.get('selected_score', 0.0)):.2f},"
                    f"margin={float(waistband_evidence.get('score_margin', 0.0)):.2f},"
                    f"split={float(recovered_metrics.get('split_t', 0.0)):.2f})"
                )
            elif recovery_source in {"crotch-concavity", "crotch+waistband"}:
                polarity_status = (
                    f"polarity={'CROTCH+WAISTBAND' if recovery_source == 'crotch+waistband' else 'CROTCH-FIRST'}-FALLBACK("
                    f"depth={float(recovered_metrics.get('crotch_depth_px', 0.0)):.1f}px,"
                    f"stable={int(recovered_metrics.get('crotch_stable_scales', 0))},"
                    f"hems={float(recovered_metrics.get('left_hem_t', 0.0)):.2f}/"
                    f"{float(recovered_metrics.get('right_hem_t', 0.0)):.2f})"
                )
            else:
                polarity_status = (
                    "polarity=MASK-FALLBACK("
                    f"split={float(recovered_metrics['split_t']):.2f})"
                )
        else:
            best = best_base
            best["summary"]["semantic_consensus"] = dict(consensus_metrics)
            best["summary"]["semantic_consensus_reliable"] = False
            best["summary"]["canonical_source"] = guided.get("source")
            best["summary"]["need_pre_spread"] = True
            best["summary"]["crotch_concavity"] = concavity
            best["summary"]["crotch_occluded"] = bool(
                not concavity_reliable
                or (isinstance(concavity, dict) and concavity.get("crotch_occluded", False))
            )
            unresolved_reason = (
                str(concavity.get("reason", "crotch gap not observable"))
                if isinstance(concavity, dict)
                else "crotch gap not observable"
            )
            polarity_status = f"polarity=UNRESOLVED(CROTCH_OCCLUDED:{unresolved_reason})"

    s = best["summary"]
    other_scores = [
        float(candidate["summary"]["score"])
        for candidate in selected_for_margin
        if candidate is not best
    ]
    score_margin = (
        float(s["score"]) - max(other_scores)
        if other_scores else float(s["score"])
    )
    s["tta_score_margin"] = float(score_margin)
    s["tta_selection_status"] = selection_status
    mode_name = "MASK-CANON-TTA" if bool(getattr(args, "pose_mask_guided_tta", True)) else (
        "D2-TTA" if bool(s.get("d2_strict", False)) else "FUSED-TTA"
    )
    best_view_meta = dict(best.get("view_meta", {}))
    best_view_wh = best_view_meta.get("input_wh")
    view_status = str(best_view_meta.get("mode", "fused"))
    if isinstance(best_view_wh, (tuple, list)) and len(best_view_wh) == 2:
        view_status += f":{int(best_view_wh[0])}x{int(best_view_wh[1])}"
    if float(best_view_meta.get("mask_fill_ratio", 0.0)) > 0.0:
        view_status += f"/{float(best_view_meta['mask_fill_ratio']):.2f}"
    status_prefix = f"{mode_name} {polarity_status}".strip()
    status = (
        f"{status_prefix} angle={best['angle']:+.1f} flip={best['flip']} "
        f"score={float(s['score']):.1f} "
        f"visible={s['visible_count']}/8 waist={s['waist_count']}/3 "
        f"crotch={'Y' if s['crotch_visible'] else 'N'} hem={s['hem_count']}/4 "
        f"geom={'OK' if bool(s.get('structure_reliable', False)) else 'LOW'} "
        f"gscore={float(s.get('structure_score', 0.0)):.2f} margin={score_margin:.1f} "
        f"canon={str(s.get('canonical_source', guided.get('source')))} "
        f"view={view_status}"
    )
    if s.get("mask_inside_ratio") is not None:
        status += f" mask={float(s['mask_inside_ratio']):.2f}"
    return (
        np.asarray(best["kxy"], dtype=np.float32),
        np.asarray(best["kcf"], dtype=np.float32),
        status,
        dict(s),
    )


def kpt_valid_xy_conf(kxy: np.ndarray, kcf: np.ndarray, idx: int, image_shape, min_conf: float) -> bool:
    h, w = image_shape[:2]
    if idx >= len(kxy):
        return False
    x, y = float(kxy[idx, 0]), float(kxy[idx, 1])
    c = float(kcf[idx]) if idx < len(kcf) else 0.0
    return bool(
        np.isfinite(x) and np.isfinite(y)
        and c >= min_conf
        and x >= 1.0 and y >= 1.0
        and x < float(w) and y < float(h)
    )


def midpoint_tuple(a: Tuple[float, float], b: Tuple[float, float]) -> Tuple[float, float]:
    return ((float(a[0]) + float(b[0])) * 0.5, (float(a[1]) + float(b[1])) * 0.5)


def mean_tuple(points: Sequence[Tuple[float, float]]) -> Tuple[float, float]:
    arr = np.asarray(points, dtype=np.float32)
    return (float(np.mean(arr[:, 0])), (float(np.mean(arr[:, 1]))))


def sort_keypoint_items_by_image_x(keypoints_board: Dict[str, Tuple[float, float]],
                                   keypoints_px: Dict[str, Tuple[float, float]],
                                   names: Sequence[str]) -> List[Tuple[str, Tuple[float, float], Tuple[float, float]]]:
    items = []
    for name in names:
        if name in keypoints_board and name in keypoints_px:
            items.append((name, keypoints_board[name], keypoints_px[name]))
    items.sort(key=lambda item: float(item[2][0]))
    return items


def maybe_fix_bottom_pose_orientation(keypoints_board: Dict[str, Tuple[float, float]],
                                      keypoints_px: Dict[str, Tuple[float, float]],
                                      keypoint_conf: Dict[str, float],
                                      enable: bool = True,
                                      flip_ratio: float = 1.15,
                                      flip_margin_mm: float = 20.0,
                                      orientation_ref_y: Optional[float] = None,
                                      flip_ref_margin_mm: float = 35.0
                                      ) -> Tuple[Dict[str, Tuple[float, float]],
                                                 Dict[str, Tuple[float, float]],
                                                 Dict[str, float],
                                                 str]:
    """Correct upside-down pose labels when the model treats the visual top as the waist."""
    if not enable:
        return keypoints_board, keypoints_px, keypoint_conf, ""

    waist_names = ["waist_img_left", "waist_center", "waist_img_right"]
    hem_names = ["img_left_hem_outer", "img_left_hem_inner", "img_right_hem_inner", "img_right_hem_outer"]
    if "waist_center" not in keypoints_board or "crotch" not in keypoints_board:
        return keypoints_board, keypoints_px, keypoint_conf, ""
    hem_items = sort_keypoint_items_by_image_x(keypoints_board, keypoints_px, hem_names)
    waist_items = sort_keypoint_items_by_image_x(keypoints_board, keypoints_px, waist_names)
    if len(hem_items) < 3 or len(waist_items) < 2:
        return keypoints_board, keypoints_px, keypoint_conf, ""

    crotch = np.asarray(keypoints_board["crotch"], dtype=np.float32)
    predicted_waist_center = np.asarray(keypoints_board["waist_center"], dtype=np.float32)
    predicted_hem_center = np.mean(np.asarray([item[1] for item in hem_items], dtype=np.float32), axis=0)
    dist_to_predicted_waist = float(np.linalg.norm(crotch - predicted_waist_center))
    dist_to_predicted_hem = float(np.linalg.norm(crotch - predicted_hem_center))
    should_flip = (
        dist_to_predicted_waist > dist_to_predicted_hem * float(flip_ratio)
        and (dist_to_predicted_waist - dist_to_predicted_hem) > float(flip_margin_mm)
    )
    waist_ref_dist = None
    hem_ref_dist = None
    if should_flip and orientation_ref_y is not None:
        waist_ref_dist = abs(float(predicted_waist_center[1]) - float(orientation_ref_y))
        hem_ref_dist = abs(float(predicted_hem_center[1]) - float(orientation_ref_y))
        should_flip = hem_ref_dist + float(flip_ref_margin_mm) < waist_ref_dist
    if not should_flip:
        return keypoints_board, keypoints_px, keypoint_conf, ""

    fixed_board = dict(keypoints_board)
    fixed_px = dict(keypoints_px)
    fixed_conf = dict(keypoint_conf)

    def set_point(name: str, board_xy: Tuple[float, float], px_xy: Tuple[float, float], conf: float) -> None:
        fixed_board[name] = (float(board_xy[0]), float(board_xy[1]))
        fixed_px[name] = (float(px_xy[0]), float(px_xy[1]))
        fixed_conf[name] = float(conf)

    semantic_waist_left = hem_items[0]
    semantic_waist_right = hem_items[-1]
    semantic_waist_board = mean_tuple([item[1] for item in hem_items])
    semantic_waist_px = mean_tuple([item[2] for item in hem_items])
    semantic_waist_conf = float(np.mean([keypoint_conf.get(item[0], 0.0) for item in hem_items]))
    set_point("waist_img_left", semantic_waist_left[1], semantic_waist_left[2], keypoint_conf.get(semantic_waist_left[0], semantic_waist_conf))
    set_point("waist_center", semantic_waist_board, semantic_waist_px, semantic_waist_conf)
    set_point("waist_img_right", semantic_waist_right[1], semantic_waist_right[2], keypoint_conf.get(semantic_waist_right[0], semantic_waist_conf))

    hem_left = waist_items[0]
    hem_right = waist_items[-1]
    hem_mid_board = waist_items[len(waist_items) // 2][1]
    hem_mid_px = waist_items[len(waist_items) // 2][2]
    hem_conf = float(np.mean([keypoint_conf.get(item[0], 0.0) for item in waist_items]))
    set_point("img_left_hem_outer", hem_left[1], hem_left[2], keypoint_conf.get(hem_left[0], hem_conf))
    set_point("img_left_hem_inner", midpoint_tuple(hem_left[1], hem_mid_board), midpoint_tuple(hem_left[2], hem_mid_px), hem_conf)
    set_point("img_right_hem_inner", midpoint_tuple(hem_mid_board, hem_right[1]), midpoint_tuple(hem_mid_px, hem_right[2]), hem_conf)
    set_point("img_right_hem_outer", hem_right[1], hem_right[2], keypoint_conf.get(hem_right[0], hem_conf))

    note = (
        "orientation corrected: predicted waist looked like hem "
        f"(crotch distances waist={dist_to_predicted_waist:.1f}mm, hem={dist_to_predicted_hem:.1f}mm)"
    )
    if waist_ref_dist is not None and hem_ref_dist is not None:
        note += f"; waist_ref distances waist={waist_ref_dist:.1f}mm, hem={hem_ref_dist:.1f}mm"
    return fixed_board, fixed_px, fixed_conf, note


def _mask_cross_section_intervals(u: np.ndarray, t: np.ndarray, sample_t: float,
                                  band_ratio: float, gap_px: float,
                                  min_points: int, min_width_px: float
                                  ) -> List[Tuple[float, float, int]]:
    selected = np.asarray(u[np.abs(t - float(sample_t)) <= float(band_ratio)], dtype=np.float32)
    if selected.size < max(4, int(min_points)):
        return []
    selected.sort()
    split_at = np.where(np.diff(selected) > float(gap_px))[0] + 1
    groups = np.split(selected, split_at)
    intervals: List[Tuple[float, float, int]] = []
    for group in groups:
        if group.size < max(3, int(min_points)):
            continue
        lo = float(np.percentile(group, 2.0))
        hi = float(np.percentile(group, 98.0))
        if hi - lo < float(min_width_px):
            continue
        intervals.append((lo, hi, int(group.size)))
    intervals.sort(key=lambda item: item[0])
    return intervals


def _outer_leg_intervals(intervals: Sequence[Tuple[float, float, int]]
                         ) -> Optional[Tuple[Tuple[float, float, int],
                                             Tuple[float, float, int]]]:
    if len(intervals) < 2:
        return None
    ordered = sorted(intervals, key=lambda item: item[0])
    left = ordered[0]
    right = ordered[-1]
    if float(right[0]) <= float(left[1]):
        return None
    return left, right


def _mask_axis_profile(points_px: np.ndarray, center_px: np.ndarray,
                       axis: np.ndarray, pose_axis: Optional[np.ndarray],
                       args) -> Optional[Dict[str, Any]]:
    axis = safe_unit(axis)
    if axis is None:
        return None
    lateral = np.asarray([-axis[1], axis[0]], dtype=np.float32)
    rel = np.asarray(points_px, dtype=np.float32) - np.asarray(center_px, dtype=np.float32)
    v = rel @ axis
    u = rel @ lateral
    v_min = float(np.percentile(v, 0.5))
    v_max = float(np.percentile(v, 99.5))
    v_span = v_max - v_min
    lateral_span = float(np.percentile(u, 99.5) - np.percentile(u, 0.5))
    if v_span < float(getattr(args, "pose_mask_geometry_min_axis_px", 80.0)):
        return None
    if lateral_span < float(getattr(args, "pose_mask_geometry_min_width_px", 60.0)):
        return None
    t = (v - v_min) / max(v_span, 1e-6)
    band = float(getattr(args, "pose_mask_geometry_band_ratio", 0.018))
    gap_px = max(
        float(getattr(args, "pose_mask_geometry_min_gap_px", 6.0)),
        lateral_span * float(getattr(args, "pose_mask_geometry_gap_width_ratio", 0.025)),
    )
    min_width = max(
        float(getattr(args, "pose_mask_geometry_min_interval_width_px", 8.0)),
        lateral_span * 0.035,
    )
    min_points = int(getattr(args, "pose_mask_geometry_min_section_points", 12))

    def intervals_at(sample: float) -> List[Tuple[float, float, int]]:
        return _mask_cross_section_intervals(
            u, t, sample, band, gap_px, min_points, min_width,
        )

    waist_samples = (0.06, 0.12, 0.18, 0.24)
    hem_samples = (0.76, 0.84, 0.90, 0.95)
    waist_sections = [intervals_at(sample) for sample in waist_samples]
    hem_sections = [intervals_at(sample) for sample in hem_samples]
    waist_single_rate = float(np.mean([len(items) == 1 for items in waist_sections]))

    hem_split_data: List[Tuple[float, Tuple[float, float, int],
                               Tuple[float, float, int], float]] = []
    for sample, items in zip(hem_samples, hem_sections):
        pair = _outer_leg_intervals(items)
        if pair is None:
            continue
        left, right = pair
        total_width = max(1.0, float(right[1]) - float(left[0]))
        gap_ratio = max(0.0, float(right[0]) - float(left[1])) / total_width
        hem_split_data.append((float(sample), left, right, float(gap_ratio)))
    hem_split_rate = float(len(hem_split_data)) / float(len(hem_samples))
    mean_gap_ratio = (
        float(np.mean([item[3] for item in hem_split_data]))
        if hem_split_data else 0.0
    )

    split_scan: List[Tuple[float, Tuple[float, float, int],
                           Tuple[float, float, int], float]] = []
    for sample in np.linspace(0.24, 0.88, 33):
        pair = _outer_leg_intervals(intervals_at(float(sample)))
        if pair is None:
            continue
        left, right = pair
        total_width = max(1.0, float(right[1]) - float(left[0]))
        gap_ratio = max(0.0, float(right[0]) - float(left[1])) / total_width
        if gap_ratio >= float(getattr(args, "pose_mask_geometry_min_split_gap_ratio", 0.025)):
            split_scan.append((float(sample), left, right, float(gap_ratio)))

    first_stable_split = None
    max_scan_step = 0.031
    for index in range(len(split_scan) - 1):
        current = split_scan[index]
        following = split_scan[index + 1]
        if float(following[0]) - float(current[0]) <= max_scan_step:
            first_stable_split = current
            break

    pose_alignment = 0.0
    pose_direction = 0.0
    if pose_axis is not None:
        pose_alignment = abs(float(np.dot(axis, pose_axis)))
        pose_direction = max(0.0, float(np.dot(axis, pose_axis)))
    score = (
        3.0 * waist_single_rate
        + 5.0 * hem_split_rate
        + 3.0 * mean_gap_ratio
        + 0.75 * pose_alignment
        + 0.25 * pose_direction
        + (0.75 if first_stable_split is not None else 0.0)
    )
    return {
        "score": float(score),
        "axis": axis,
        "lateral": lateral,
        "v_min": v_min,
        "v_max": v_max,
        "v_span": v_span,
        "lateral_span": lateral_span,
        "waist_single_rate": waist_single_rate,
        "hem_split_rate": hem_split_rate,
        "mean_gap_ratio": mean_gap_ratio,
        "first_stable_split": first_stable_split,
    }


def prepare_bottom_mask_polarity(
        mask: Optional[BottomMaskBoard], args,
) -> Optional[Dict[str, Any]]:
    """Sample the selected mask once for candidate-axis polarity checks."""
    if mask is None or getattr(mask, "mask_u8", None) is None:
        return None
    binary = np.asarray(mask.mask_u8) > 0
    area_px = int(np.count_nonzero(binary))
    if area_px < int(getattr(args, "pose_mask_geometry_min_area_px", 1800)):
        return None

    max_points = max(10000, int(getattr(args, "pose_mask_polarity_max_points", 40000)))
    sample_step = max(1, int(math.ceil(math.sqrt(float(area_px) / float(max_points)))))
    sampled = binary[::sample_step, ::sample_step]
    ys, xs = np.where(sampled)
    if len(xs) < 500:
        return None
    points_px = np.column_stack([
        xs.astype(np.float32) * float(sample_step),
        ys.astype(np.float32) * float(sample_step),
    ])
    center_px = np.mean(points_px, axis=0).astype(np.float32)
    return {
        "points_px": points_px,
        "center_px": center_px,
        "image_shape": binary.shape,
    }


def _mask_profile_reliable(profile: Optional[Dict[str, Any]], args) -> bool:
    if profile is None:
        return False
    if float(profile["score"]) < float(getattr(args, "pose_mask_geometry_min_score", 6.0)):
        return False
    if float(profile["waist_single_rate"]) < float(
            getattr(args, "pose_mask_geometry_min_waist_single_rate", 0.50)):
        return False
    if float(profile["hem_split_rate"]) < float(
            getattr(args, "pose_mask_geometry_min_hem_split_rate", 0.50)):
        return False
    return profile["first_stable_split"] is not None


def evaluate_bottom_mask_polarity(
        samples: Optional[Dict[str, Any]], pose_axis: np.ndarray, args,
) -> Optional[Dict[str, Any]]:
    """Compare the mask along a pose axis in both semantic directions."""
    if samples is None:
        return None
    axis = safe_unit(pose_axis)
    if axis is None:
        return None
    points_px = np.asarray(samples["points_px"], dtype=np.float32)
    center_px = np.asarray(samples["center_px"], dtype=np.float32)
    forward = _mask_axis_profile(points_px, center_px, axis, None, args)
    reverse = _mask_axis_profile(points_px, center_px, -axis, None, args)
    ranked = [
        (1.0, forward),
        (-1.0, reverse),
    ]
    ranked = [item for item in ranked if item[1] is not None]
    if not ranked:
        return None
    ranked.sort(key=lambda item: float(item[1]["score"]), reverse=True)
    sign, best_profile = ranked[0]
    other_score = float(ranked[1][1]["score"]) if len(ranked) > 1 else -1e9
    score_margin = float(best_profile["score"]) - other_score
    if not _mask_profile_reliable(best_profile, args):
        return None
    if score_margin < float(getattr(args, "pose_mask_polarity_min_profile_margin", 0.75)):
        return None
    result = dict(best_profile)
    result["dot"] = float(sign)
    result["axis"] = np.asarray(axis * float(sign), dtype=np.float32)
    result["score_margin"] = float(score_margin)
    result["forward_score"] = None if forward is None else float(forward["score"])
    result["reverse_score"] = None if reverse is None else float(reverse["score"])
    return result


def mask_landmarks_from_crotch_concavity(
        samples: Optional[Dict[str, Any]], concavity: Optional[Dict[str, Any]], args,
) -> Optional[Tuple[np.ndarray, Dict[str, Any]]]:
    """Recover 0~7 after fixing keypoint 3 from the stable crotch notch.

    Each leg is scanned to its own longitudinal endpoint. This avoids the old
    assumption that both hems must intersect one shared far-end cross-section,
    which is fragile for diagonal poses and unequal or partly folded legs.
    """
    if samples is None or not isinstance(concavity, dict):
        return None
    if not bool(concavity.get("reliable", False)):
        return None
    if not bool(concavity.get("point_reliable", True)):
        return None
    axis = safe_unit(np.asarray(concavity.get("axis"), dtype=np.float32))
    crotch_point = np.asarray(concavity.get("point_px"), dtype=np.float32)
    if axis is None or crotch_point.shape != (2,) or not np.all(np.isfinite(crotch_point)):
        return None

    points_px = np.asarray(samples["points_px"], dtype=np.float32)
    center_px = np.asarray(samples["center_px"], dtype=np.float32)
    image_shape = tuple(samples["image_shape"])
    if len(points_px) < 500 or len(image_shape) < 2:
        return None
    # For upright pants axis=(0,+1), garment-right=(+1,0).
    lateral = np.asarray([axis[1], -axis[0]], dtype=np.float32)
    rel = points_px - center_px
    v = rel @ axis
    u = rel @ lateral
    v_min = float(np.percentile(v, 0.5))
    v_max = float(np.percentile(v, 99.5))
    v_span = v_max - v_min
    lateral_span = float(np.percentile(u, 99.5) - np.percentile(u, 0.5))
    if v_span < float(getattr(args, "pose_mask_geometry_min_axis_px", 80.0)):
        return None
    if lateral_span < float(getattr(args, "pose_mask_geometry_min_width_px", 60.0)):
        return None
    t = (v - v_min) / max(v_span, 1e-6)
    crotch_rel = crotch_point - center_px
    crotch_v = float(np.dot(crotch_rel, axis))
    crotch_u = float(np.dot(crotch_rel, lateral))
    crotch_t = (crotch_v - v_min) / max(v_span, 1e-6)
    if not (
        float(getattr(args, "pose_crotch_landmark_min_t", 0.24))
        <= crotch_t
        <= float(getattr(args, "pose_crotch_landmark_max_t", 0.92))
    ):
        return None

    band = max(0.008, float(getattr(
        args, "pose_crotch_landmark_band_ratio", 0.018,
    )))
    gap_px = max(
        float(getattr(args, "pose_mask_geometry_min_gap_px", 6.0)),
        lateral_span * float(getattr(args, "pose_mask_geometry_gap_width_ratio", 0.025)),
    )
    min_width = max(
        float(getattr(args, "pose_mask_geometry_min_interval_width_px", 8.0)),
        lateral_span * 0.035,
    )
    min_points = max(6, int(getattr(
        args, "pose_mask_geometry_min_section_points", 12,
    )))

    def intervals_at(sample: float) -> List[Tuple[float, float, int]]:
        return _mask_cross_section_intervals(
            u, t, sample, band, gap_px, min_points, min_width,
        )

    waist_limit = min(0.30, max(0.10, crotch_t - 0.13))
    waist_options: List[Tuple[float, float, Tuple[float, float, int]]] = []
    for sample in np.linspace(0.025, waist_limit, 13):
        intervals = intervals_at(float(sample))
        if len(intervals) != 1:
            continue
        interval = intervals[0]
        width = float(interval[1]) - float(interval[0])
        midpoint = 0.5 * (float(interval[0]) + float(interval[1]))
        score = (
            width / max(lateral_span, 1.0)
            - 1.5 * abs(float(sample) - 0.08)
            - 0.10 * abs(midpoint - crotch_u) / max(lateral_span, 1.0)
        )
        waist_options.append((float(score), float(sample), interval))
    if not waist_options:
        return None
    _, waist_t, waist_interval = max(waist_options, key=lambda item: item[0])

    hem_start_t = max(
        float(getattr(args, "pose_crotch_landmark_hem_scan_min_t", 0.52)),
        crotch_t + float(getattr(args, "pose_crotch_landmark_after_crotch_t", 0.045)),
    )
    if hem_start_t >= 0.97:
        return None
    section_samples = np.linspace(hem_start_t, 0.985, 32)
    width_fraction = float(np.clip(getattr(
        args, "pose_crotch_landmark_min_hem_width_fraction", 0.35,
    ), 0.10, 0.90))

    def independent_hem_section(side: int) -> Optional[Tuple[float, float, float, int, float]]:
        side_selector = u <= crotch_u if side < 0 else u >= crotch_u
        sections: List[Tuple[int, float, float, float, int, float]] = []
        for sample_index, sample in enumerate(section_samples):
            selected = np.asarray(
                u[side_selector & (np.abs(t - float(sample)) <= band)],
                dtype=np.float32,
            )
            if selected.size < min_points:
                continue
            lo = float(np.percentile(selected, 2.0))
            hi = float(np.percentile(selected, 98.0))
            width = hi - lo
            if width < min_width:
                continue
            sections.append((sample_index, float(sample), lo, hi, int(selected.size), width))
        if not sections:
            return None
        max_width = max(float(section[5]) for section in sections)
        width_floor = max(min_width, width_fraction * max_width)
        eligible = [section for section in sections if float(section[5]) >= width_floor]
        if not eligible:
            return None
        eligible_indices = {int(section[0]) for section in eligible}
        stable = [
            section for section in eligible
            if int(section[0]) - 1 in eligible_indices
            or int(section[0]) + 1 in eligible_indices
        ]
        chosen = max(stable or eligible, key=lambda section: float(section[1]))
        return (
            float(chosen[1]), float(chosen[2]), float(chosen[3]),
            int(chosen[4]), float(chosen[5]),
        )

    left_hem = independent_hem_section(-1)
    right_hem = independent_hem_section(+1)
    if left_hem is None or right_hem is None:
        return None
    left_t, left_lo, left_hi, left_count, left_width = left_hem
    right_t, right_lo, right_hi, right_count, right_width = right_hem

    def point_from_local(u_value: float, t_value: float) -> np.ndarray:
        v_value = v_min + float(t_value) * v_span
        point = center_px + axis * v_value + lateral * float(u_value)
        point[0] = float(np.clip(point[0], 1.0, image_shape[1] - 2.0))
        point[1] = float(np.clip(point[1], 1.0, image_shape[0] - 2.0))
        return point.astype(np.float32)

    crotch_clipped = crotch_point.copy()
    crotch_clipped[0] = float(np.clip(crotch_clipped[0], 1.0, image_shape[1] - 2.0))
    crotch_clipped[1] = float(np.clip(crotch_clipped[1], 1.0, image_shape[0] - 2.0))
    waist_left = float(waist_interval[0])
    waist_right = float(waist_interval[1])
    landmarks = np.stack([
        point_from_local(waist_left, waist_t),
        point_from_local(0.5 * (waist_left + waist_right), waist_t),
        point_from_local(waist_right, waist_t),
        crotch_clipped.astype(np.float32),
        point_from_local(left_lo, left_t),
        point_from_local(left_hi, left_t),
        point_from_local(right_lo, right_t),
        point_from_local(right_hi, right_t),
    ], axis=0).astype(np.float32)
    hem_gap_ratio = max(0.0, right_lo - left_hi) / max(lateral_span, 1.0)
    metrics: Dict[str, Any] = {
        "source": str(concavity.get("source", "crotch-concavity")),
        "split_t": float(crotch_t),
        "split_gap_ratio": float(concavity.get("depth_chord_ratio", 0.0)),
        "hem_gap_ratio": float(hem_gap_ratio),
        "left_hem_t": float(left_t),
        "right_hem_t": float(right_t),
        "independent_hem_t_delta": float(abs(left_t - right_t)),
        "left_hem_width_px": float(left_width),
        "right_hem_width_px": float(right_width),
        "left_section_points": int(left_count),
        "right_section_points": int(right_count),
        "crotch_depth_px": float(concavity.get("depth_px", 0.0)),
        "crotch_stable_scales": int(concavity.get("stable_scale_count", 0)),
    }
    return landmarks, metrics


def mask_landmarks_from_polarity(
        samples: Optional[Dict[str, Any]], polarity: Optional[Dict[str, Any]],
        predicted_side: Optional[np.ndarray], args,
) -> Optional[Tuple[np.ndarray, Dict[str, float]]]:
    """Recover all eight landmarks only after a strong reversed-axis decision."""
    if samples is None or polarity is None:
        return None
    axis = safe_unit(np.asarray(polarity["axis"], dtype=np.float32))
    if axis is None:
        return None
    points_px = np.asarray(samples["points_px"], dtype=np.float32)
    center_px = np.asarray(samples["center_px"], dtype=np.float32)
    image_shape = tuple(samples["image_shape"])
    # Directed waist->hem polarity fixes the garment-local handedness. For an
    # upright image this vector points to image-right, so local ordering remains
    # continuous when the pants rotate through 45/90/180 degrees.
    lateral = np.asarray([axis[1], -axis[0]], dtype=np.float32)

    rel = points_px - center_px
    v = rel @ axis
    u = rel @ lateral
    v_min = float(np.percentile(v, 0.5))
    v_max = float(np.percentile(v, 99.5))
    v_span = max(1e-6, v_max - v_min)
    t = (v - v_min) / v_span
    lateral_span = float(np.percentile(u, 99.5) - np.percentile(u, 0.5))
    band = float(getattr(args, "pose_mask_geometry_band_ratio", 0.018))
    gap_px = max(
        float(getattr(args, "pose_mask_geometry_min_gap_px", 6.0)),
        lateral_span * float(getattr(args, "pose_mask_geometry_gap_width_ratio", 0.025)),
    )
    min_width = max(
        float(getattr(args, "pose_mask_geometry_min_interval_width_px", 8.0)),
        lateral_span * 0.035,
    )
    min_points = int(getattr(args, "pose_mask_geometry_min_section_points", 12))

    def intervals_at(sample: float) -> List[Tuple[float, float, int]]:
        return _mask_cross_section_intervals(
            u, t, sample, band, gap_px, min_points, min_width,
        )

    waist_options = []
    for sample in np.linspace(0.04, 0.18, 8):
        items = intervals_at(float(sample))
        if len(items) != 1:
            continue
        interval = items[0]
        width = float(interval[1]) - float(interval[0])
        score = width / max(lateral_span, 1.0) - 2.0 * abs(float(sample) - 0.08)
        waist_options.append((score, float(sample), interval))
    if not waist_options:
        return None
    _, waist_t, waist_interval = max(waist_options, key=lambda item: item[0])

    hem_options = []
    for sample in np.linspace(0.78, 0.97, 11):
        pair = _outer_leg_intervals(intervals_at(float(sample)))
        if pair is None:
            continue
        left, right = pair
        total_width = max(1.0, float(right[1]) - float(left[0]))
        gap_ratio = max(0.0, float(right[0]) - float(left[1])) / total_width
        score = 2.5 * gap_ratio - abs(float(sample) - 0.92)
        hem_options.append((score, float(sample), left, right, gap_ratio))
    if not hem_options:
        return None
    _, hem_t, left_hem, right_hem, hem_gap_ratio = max(
        hem_options, key=lambda item: item[0],
    )

    split_options = []
    for sample in np.linspace(0.24, 0.88, 33):
        pair = _outer_leg_intervals(intervals_at(float(sample)))
        if pair is None:
            continue
        left, right = pair
        total_width = max(1.0, float(right[1]) - float(left[0]))
        gap_ratio = max(0.0, float(right[0]) - float(left[1])) / total_width
        if gap_ratio >= float(getattr(args, "pose_mask_geometry_min_split_gap_ratio", 0.025)):
            split_options.append((float(sample), left, right, float(gap_ratio)))
    stable_split = None
    for index in range(len(split_options) - 1):
        current = split_options[index]
        following = split_options[index + 1]
        if float(following[0]) - float(current[0]) <= 0.031:
            stable_split = current
            break
    if stable_split is None:
        return None
    split_t, split_left, split_right, split_gap_ratio = stable_split

    def point_from_local(u_value: float, t_value: float) -> np.ndarray:
        v_value = v_min + float(t_value) * v_span
        point = center_px + axis * v_value + lateral * float(u_value)
        point[0] = float(np.clip(point[0], 1.0, image_shape[1] - 2.0))
        point[1] = float(np.clip(point[1], 1.0, image_shape[0] - 2.0))
        return point.astype(np.float32)

    waist_left = float(waist_interval[0])
    waist_right = float(waist_interval[1])
    crotch_u = 0.5 * (float(split_left[1]) + float(split_right[0]))
    landmarks = np.stack([
        point_from_local(waist_left, waist_t),
        point_from_local(0.5 * (waist_left + waist_right), waist_t),
        point_from_local(waist_right, waist_t),
        point_from_local(crotch_u, float(split_t)),
        point_from_local(float(left_hem[0]), hem_t),
        point_from_local(float(left_hem[1]), hem_t),
        point_from_local(float(right_hem[0]), hem_t),
        point_from_local(float(right_hem[1]), hem_t),
    ], axis=0).astype(np.float32)
    metrics = {
        "split_t": float(split_t),
        "split_gap_ratio": float(split_gap_ratio),
        "hem_gap_ratio": float(hem_gap_ratio),
    }
    return landmarks, metrics



def mask_landmarks_from_closed_crotch(
        mask: Optional[BottomMaskBoard], samples: Optional[Dict[str, Any]],
        hypothesis: Optional[Dict[str, Any]], pose_hint_px: Optional[np.ndarray],
        pose_hint_conf: float, args,
) -> Optional[Tuple[np.ndarray, Dict[str, Any]]]:
    """Recover all eight landmarks while keeping G3 explicitly inferred."""
    if mask is None or samples is None or not isinstance(hypothesis, dict):
        return None
    closed = hypothesis.get("closed_crotch")
    if not isinstance(closed, dict) or not closed.get("usable", False):
        closed = estimate_closed_crotch_geometry(
            mask, samples, hypothesis, pose_hint_px, pose_hint_conf, args,
        )
    elif pose_hint_px is not None:
        # Re-evaluate once with the weak pose hint; reliable observed crotches never
        # reach this fallback, so this cannot degrade the existing primary path.
        hinted = estimate_closed_crotch_geometry(
            mask, samples, hypothesis, pose_hint_px, pose_hint_conf, args,
        )
        if isinstance(hinted, dict) and hinted.get("usable", False):
            closed = hinted
    if not isinstance(closed, dict) or not closed.get("usable", False):
        return None

    axis = safe_unit(np.asarray(closed.get("axis"), dtype=np.float32))
    if axis is None:
        return None
    lateral = np.asarray([axis[1], -axis[0]], dtype=np.float32)
    points_px = np.asarray(samples["points_px"], dtype=np.float32)
    center_px = np.asarray(samples["center_px"], dtype=np.float32)
    image_shape = tuple(samples["image_shape"])
    rel = points_px - center_px
    v = rel @ axis
    u = rel @ lateral
    v_min = float(np.percentile(v, 0.5))
    v_max = float(np.percentile(v, 99.5))
    v_span = max(1e-6, v_max - v_min)
    t = (v - v_min) / v_span
    lateral_span = float(np.percentile(u, 99.5) - np.percentile(u, 0.5))
    if v_span < 50.0 or lateral_span < 35.0:
        return None

    band = max(0.010, float(getattr(
        args, "pose_mask_geometry_band_ratio", 0.018,
    )))
    gap_px = max(
        float(getattr(args, "pose_mask_geometry_min_gap_px", 6.0)),
        lateral_span * float(getattr(args, "pose_mask_geometry_gap_width_ratio", 0.025)),
    )
    min_width = max(5.0, lateral_span * 0.025)
    min_points = max(5, int(getattr(
        args, "pose_mask_geometry_min_section_points", 12,
    )) // 2)

    def intervals_at(sample: float) -> List[Tuple[float, float, int]]:
        return _mask_cross_section_intervals(
            u, t, sample, band, gap_px, min_points, min_width,
        )

    waist_options: List[Tuple[float, float, Tuple[float, float, int]]] = []
    for sample in np.linspace(0.025, 0.22, 15):
        intervals = intervals_at(float(sample))
        if not intervals:
            continue
        interval = max(intervals, key=lambda item: float(item[1]) - float(item[0]))
        width = float(interval[1]) - float(interval[0])
        single_bonus = 0.28 if len(intervals) == 1 else 0.0
        score = width / max(lateral_span, 1.0) + single_bonus - 0.40 * float(sample)
        waist_options.append((score, float(sample), interval))
    if not waist_options:
        return None
    _, waist_t, waist_interval = max(waist_options, key=lambda item: item[0])
    waist_lo, waist_hi = float(waist_interval[0]), float(waist_interval[1])

    crotch_t = float(closed["crotch_t"])
    crotch_u = float(closed.get("crotch_u", 0.0))
    hem_start = max(0.66, crotch_t + 0.08)

    split_hem_options: List[Tuple[float, float, Tuple[float, float, int], Tuple[float, float, int]]] = []
    for sample in np.linspace(hem_start, 0.985, 25):
        pair = _outer_leg_intervals(intervals_at(float(sample)))
        if pair is None:
            continue
        left, right = pair
        total = max(1.0, float(right[1]) - float(left[0]))
        gap = max(0.0, float(right[0]) - float(left[1])) / total
        split_hem_options.append((gap, float(sample), left, right))

    hem_measured_split = bool(split_hem_options)
    if split_hem_options:
        _, hem_t, left_interval, right_interval = max(
            split_hem_options,
            key=lambda item: (float(item[0]), float(item[1])),
        )
        left_t = right_t = float(hem_t)
        left_lo, left_hi = float(left_interval[0]), float(left_interval[1])
        right_lo, right_hi = float(right_interval[0]), float(right_interval[1])
    else:
        def independent_hem(side: int) -> Optional[Tuple[float, float, float]]:
            side_mask = u <= crotch_u if side < 0 else u >= crotch_u
            records: List[Tuple[float, float, float, float]] = []
            for sample in np.linspace(hem_start, 0.985, 28):
                selected = np.asarray(
                    u[side_mask & (np.abs(t - float(sample)) <= band)],
                    dtype=np.float32,
                )
                if selected.size < min_points:
                    continue
                lo = float(np.percentile(selected, 2.0))
                hi = float(np.percentile(selected, 98.0))
                width = hi - lo
                if width >= min_width:
                    records.append((float(sample), lo, hi, width))
            if not records:
                return None
            max_width = max(item[3] for item in records)
            eligible = [item for item in records if item[3] >= max(min_width, 0.30 * max_width)]
            sample, lo, hi, _ = max(eligible or records, key=lambda item: item[0])
            return float(sample), float(lo), float(hi)

        left = independent_hem(-1)
        right = independent_hem(+1)
        if left is None or right is None:
            return None
        left_t, left_lo, left_hi = left
        right_t, right_lo, right_hi = right

    def point_from_local(u_value: float, t_value: float) -> np.ndarray:
        point = center_px + axis * (v_min + float(t_value) * v_span) + lateral * float(u_value)
        point[0] = float(np.clip(point[0], 1.0, image_shape[1] - 2.0))
        point[1] = float(np.clip(point[1], 1.0, image_shape[0] - 2.0))
        return point.astype(np.float32)

    landmarks = np.stack([
        point_from_local(waist_lo, waist_t),
        point_from_local(0.5 * (waist_lo + waist_hi), waist_t),
        point_from_local(waist_hi, waist_t),
        np.asarray(closed["point_px"], dtype=np.float32),
        point_from_local(left_lo, left_t),
        point_from_local(left_hi, left_t),
        point_from_local(right_lo, right_t),
        point_from_local(right_hi, right_t),
    ], axis=0).astype(np.float32)
    return landmarks, {
        "source": "waistband-closed-crotch",
        "crotch_state": "INFERRED_CLOSED_GEOMETRY",
        "crotch_confidence": float(closed.get("confidence", 0.0)),
        "crotch_reliable": False,
        "crotch_inferred": True,
        "crotch_occluded": True,
        "pre_spread_required": True,
        "split_t": float(crotch_t),
        "split_gap_ratio": 0.0,
        "hem_measured_split": bool(hem_measured_split),
        "left_hem_t": float(left_t),
        "right_hem_t": float(right_t),
        "closed_crotch": closed,
    }


def mask_landmarks_from_rectangle_fallback(
        mask: Optional[BottomMaskBoard], samples: Optional[Dict[str, Any]],
        wrinkle: Optional[Dict[str, Any]], args,
) -> Optional[Tuple[np.ndarray, Dict[str, Any]]]:
    """Create usable G0~G7 from a Heatmap-directed rectangle and mask slices.

    This is intentionally tolerant: a visible leg split gives a measured G3,
    while attached/overlapped legs use a marked approximate G3 and conservative
    hem quartiles. The robot still snaps every contact back onto dense mask data.
    """
    if mask is None or samples is None:
        return None
    hypothesis = _waistband_first_axis_hypothesis(
        mask, samples, wrinkle, args, allow_weak=True,
    )
    orientation_source = "rectangle-heat"
    if hypothesis is not None:
        axis = safe_unit(np.asarray(hypothesis.get("axis"), dtype=np.float32))
    else:
        axis = None

    points_px = np.asarray(samples.get("points_px"), dtype=np.float32)
    center_px = np.asarray(samples.get("center_px"), dtype=np.float32)
    image_shape = tuple(samples.get("image_shape", mask.mask_u8.shape[:2]))
    if points_px.ndim != 2 or points_px.shape[1] != 2 or len(points_px) < 200:
        return None

    if axis is None:
        rectangle = _mask_oriented_rectangle_px(mask, args)
        candidates: List[Tuple[float, np.ndarray, Optional[Dict[str, Any]]]] = []
        if rectangle is not None:
            for pair in rectangle["pair_axes"]:
                raw_axis = safe_unit(np.asarray(pair["axis"], dtype=np.float32))
                if raw_axis is None:
                    continue
                for signed_axis in (raw_axis, -raw_axis):
                    profile = _mask_axis_profile(
                        points_px, center_px, signed_axis, None, args,
                    )
                    score = -1.0 if profile is None else float(profile.get("score", 0.0))
                    candidates.append((score, signed_axis, profile))
        if candidates:
            _, axis, _ = max(candidates, key=lambda item: item[0])
            orientation_source = "rectangle-topology"
    if axis is None:
        pca = _mask_pca_geometry_px(mask, args)
        if pca is None:
            return None
        axis = safe_unit(np.asarray(pca["major_axis_px"], dtype=np.float32))
        orientation_source = "rectangle-undirected-last-resort"
    if axis is None:
        return None

    lateral = np.asarray([axis[1], -axis[0]], dtype=np.float32)
    rel = points_px - center_px
    v = rel @ axis
    u = rel @ lateral
    v_min = float(np.percentile(v, 0.5))
    v_max = float(np.percentile(v, 99.5))
    v_span = v_max - v_min
    lateral_span = float(np.percentile(u, 99.5) - np.percentile(u, 0.5))
    if v_span < 35.0 or lateral_span < 30.0:
        return None
    t = (v - v_min) / max(v_span, 1e-6)
    band = max(0.012, float(getattr(args, "pose_mask_geometry_band_ratio", 0.018)))
    gap_px = max(
        float(getattr(args, "pose_mask_geometry_min_gap_px", 6.0)),
        lateral_span * float(getattr(args, "pose_mask_geometry_gap_width_ratio", 0.025)),
    )
    min_width = max(5.0, lateral_span * 0.025)
    min_points = max(5, int(getattr(args, "pose_mask_geometry_min_section_points", 12)) // 2)

    def intervals_at(sample: float) -> List[Tuple[float, float, int]]:
        return _mask_cross_section_intervals(
            u, t, sample, band, gap_px, min_points, min_width,
        )

    waist_options: List[Tuple[float, float, Tuple[float, float, int]]] = []
    for sample in np.linspace(0.025, 0.24, 16):
        intervals = intervals_at(float(sample))
        if not intervals:
            continue
        interval = max(intervals, key=lambda item: float(item[1]) - float(item[0]))
        width = float(interval[1]) - float(interval[0])
        single_bonus = 0.25 if len(intervals) == 1 else 0.0
        score = width / max(lateral_span, 1.0) + single_bonus - 0.35 * float(sample)
        waist_options.append((float(score), float(sample), interval))
    if waist_options:
        _, waist_t, waist_interval = max(waist_options, key=lambda item: item[0])
        waist_lo, waist_hi = float(waist_interval[0]), float(waist_interval[1])
    else:
        selected = u[t <= 0.24]
        if selected.size < min_points:
            selected = u
        waist_t = 0.08
        waist_lo = float(np.percentile(selected, 2.0))
        waist_hi = float(np.percentile(selected, 98.0))

    profile = _mask_axis_profile(points_px, center_px, axis, None, args)
    stable_split = None if profile is None else profile.get("first_stable_split")
    closed_crotch = None
    crotch_state = "UNKNOWN"
    crotch_confidence = 0.0
    if stable_split is not None:
        split_t, split_left, split_right, split_gap_ratio = stable_split
        crotch_u = 0.5 * (float(split_left[1]) + float(split_right[0]))
        crotch_reliable = True
        crotch_state = "OBSERVED_MASK_SPLIT"
        crotch_confidence = 0.90
    else:
        closed_crotch = estimate_closed_crotch_geometry(
            mask, samples, hypothesis, None, 0.0, args,
        )
        if isinstance(closed_crotch, dict) and closed_crotch.get("usable", False):
            split_t = float(closed_crotch["crotch_t"])
            crotch_u = float(closed_crotch.get("crotch_u", 0.0))
            split_gap_ratio = 0.0
            crotch_reliable = False
            crotch_state = "INFERRED_CLOSED_GEOMETRY"
            crotch_confidence = float(closed_crotch.get("confidence", 0.0))
        else:
            split_t = float(np.clip(getattr(
                args, "pose_rectangle_fallback_crotch_t", 0.58,
            ), 0.40, 0.78))
            crotch_u = 0.0
            split_gap_ratio = 0.0
            crotch_reliable = False
            crotch_state = "INFERRED_POLARITY_PRIOR"
            crotch_confidence = float(getattr(
                args, "pose_rectangle_fallback_prior_conf", 0.28,
            ))

    hem_start = max(0.62, float(split_t) + 0.04)

    def independent_hem(side: int) -> Tuple[float, float, float, bool]:
        side_selected = u <= float(crotch_u) if side < 0 else u >= float(crotch_u)
        sections: List[Tuple[float, float, float, int]] = []
        for sample in np.linspace(hem_start, 0.985, 28):
            values = np.asarray(u[side_selected & (np.abs(t - float(sample)) <= band)], dtype=np.float32)
            if values.size < min_points:
                continue
            lo = float(np.percentile(values, 2.0))
            hi = float(np.percentile(values, 98.0))
            if hi - lo >= min_width:
                sections.append((float(sample), lo, hi, int(values.size)))
        if sections:
            widths = [item[2] - item[1] for item in sections]
            floor = max(min_width, 0.30 * max(widths))
            eligible = [item for item in sections if item[2] - item[1] >= floor]
            sample, lo, hi, _ = max(eligible or sections, key=lambda item: item[0])
            return sample, lo, hi, True
        values = np.asarray(u[side_selected & (t >= 0.70)], dtype=np.float32)
        if values.size < min_points:
            values = np.asarray(u[side_selected], dtype=np.float32)
        if values.size < 2:
            if side < 0:
                return 0.90, -0.48 * lateral_span, -0.08 * lateral_span, False
            return 0.90, 0.08 * lateral_span, 0.48 * lateral_span, False
        return (
            0.90,
            float(np.percentile(values, 3.0)),
            float(np.percentile(values, 97.0)),
            False,
        )

    left_t, left_lo, left_hi, left_measured = independent_hem(-1)
    right_t, right_lo, right_hi, right_measured = independent_hem(+1)

    def point_from_local(u_value: float, t_value: float) -> np.ndarray:
        point = center_px + axis * (v_min + float(t_value) * v_span) + lateral * float(u_value)
        point[0] = float(np.clip(point[0], 1.0, image_shape[1] - 2.0))
        point[1] = float(np.clip(point[1], 1.0, image_shape[0] - 2.0))
        return point.astype(np.float32)

    landmarks = np.stack([
        point_from_local(waist_lo, waist_t),
        point_from_local(0.5 * (waist_lo + waist_hi), waist_t),
        point_from_local(waist_hi, waist_t),
        point_from_local(crotch_u, float(split_t)),
        point_from_local(left_lo, left_t),
        point_from_local(left_hi, left_t),
        point_from_local(right_lo, right_t),
        point_from_local(right_hi, right_t),
    ], axis=0).astype(np.float32)
    return landmarks, {
        "source": orientation_source,
        "axis": axis,
        "waist_t": float(waist_t),
        "crotch_t": float(split_t),
        "crotch_reliable": bool(crotch_reliable),
        "crotch_state": str(crotch_state),
        "crotch_confidence": float(crotch_confidence),
        "crotch_inferred": bool(not crotch_reliable),
        "crotch_occluded": bool(not crotch_reliable),
        "pre_spread_required": bool(not crotch_reliable),
        "closed_crotch": closed_crotch,
        "split_gap_ratio": float(split_gap_ratio),
        "left_hem_t": float(left_t),
        "right_hem_t": float(right_t),
        "left_hem_measured": bool(left_measured),
        "right_hem_measured": bool(right_measured),
        "hypothesis": hypothesis,
    }


def build_rectangle_fallback_pose(
        mask: Optional[BottomMaskBoard], H: Optional[np.ndarray], image_shape,
        wrinkle: Optional[Dict[str, Any]], args, trigger_reason: str,
) -> Optional[BottomPoseBoard]:
    if (
        mask is None or H is None
        or not bool(getattr(args, "pose_rectangle_fallback", True))
    ):
        return None
    samples = prepare_bottom_mask_polarity(mask, args)
    recovered = mask_landmarks_from_rectangle_fallback(mask, samples, wrinkle, args)
    if recovered is None:
        return None
    landmarks, metrics = recovered
    if landmarks.shape != (8, 2) or not np.all(np.isfinite(landmarks)):
        return None

    confidence = float(np.clip(getattr(
        args, "pose_rectangle_fallback_conf", 0.30,
    ), 0.21, 0.70))
    keypoints_px: Dict[str, Tuple[float, float]] = {}
    keypoints_board: Dict[str, Tuple[float, float]] = {}
    keypoint_conf: Dict[str, float] = {}
    try:
        for index, name in enumerate(BOTTOM_POSE_KPT_NAMES):
            px = np.asarray(landmarks[index], dtype=np.float32)
            board = pixel_to_board(H, float(px[0]), float(px[1]))
            keypoints_px[name] = (float(px[0]), float(px[1]))
            keypoints_board[name] = (float(board[0]), float(board[1]))
            keypoint_conf[name] = confidence
    except Exception:
        return None

    def board_point(name: str) -> np.ndarray:
        return np.asarray(keypoints_board[name], dtype=np.float32)

    waist_left = board_point("waist_img_left")
    waist_center = board_point("waist_center")
    waist_right = board_point("waist_img_right")
    crotch = board_point("crotch")
    left_hem_center = 0.5 * (
        board_point("img_left_hem_outer") + board_point("img_left_hem_inner")
    )
    right_hem_center = 0.5 * (
        board_point("img_right_hem_inner") + board_point("img_right_hem_outer")
    )
    lower_center = 0.5 * (left_hem_center + right_hem_center)
    pose_center = np.mean(np.stack([
        waist_center, crotch, left_hem_center, right_hem_center,
    ]), axis=0)
    waist_vec = waist_right - waist_left
    body_axis = lower_center - waist_center
    axis_len = safe_norm(body_axis)
    if axis_len < float(getattr(args, "pose_rectangle_fallback_min_axis_mm", 45.0)):
        return None

    source = str(metrics.get("source", "rectangle-mask"))
    crotch_reliable = bool(metrics.get("crotch_reliable", False))
    crotch_state = str(metrics.get(
        "crotch_state", "OBSERVED_MASK_SPLIT" if crotch_reliable else "INFERRED_POLARITY_PRIOR",
    ))
    crotch_confidence = float(metrics.get(
        "crotch_confidence", 0.90 if crotch_reliable else 0.28,
    ))
    status = (
        f"RECTANGLE-MASK-FALLBACK source={source} "
        f"crotch={'measured' if crotch_reliable else 'approximate'}"
    )
    geometry_metrics = {
        "reliable": False,
        "recovery_required": True,
        "geometry_score": 0.35 if crotch_reliable else 0.24,
        "recovery_reasons": ["pose network unreliable", "rectangle-mask fallback active"],
        "rectangle_fallback": True,
        "rectangle_metrics": metrics,
        "trigger_reason": str(trigger_reason),
    }
    return BottomPoseBoard(
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
        waist_angle_deg=float(np.degrees(np.arctan2(waist_vec[1], waist_vec[0]))),
        pants_axis_angle_deg=float(np.degrees(np.arctan2(body_axis[1], body_axis[0]))),
        pants_axis_len_mm=float(axis_len),
        waist_width_mm=safe_norm(waist_vec),
        hem_gap_mm=safe_norm(right_hem_center - left_hem_center),
        valid=True,
        reason=f"{status}; trigger={trigger_reason}",
        inference_status=status,
        geometry_reliable=False,
        geometry_score=float(geometry_metrics["geometry_score"]),
        recovery_required=True,
        geometry_metrics=geometry_metrics,
        crotch_state=crotch_state,
        crotch_confidence=crotch_confidence,
        polarity_metrics={
            "source": source,
            "rectangle_metrics": metrics,
        },
        pre_spread_required=bool(metrics.get("pre_spread_required", not crotch_reliable)),
    )


def refine_bottom_pose_from_mask_geometry(
        keypoints_board: Dict[str, Tuple[float, float]],
        keypoints_px: Dict[str, Tuple[float, float]],
        keypoint_conf: Dict[str, float],
        mask: Optional[BottomMaskBoard],
        H: np.ndarray,
        args,
) -> Tuple[Dict[str, Tuple[float, float]],
           Dict[str, Tuple[float, float]],
           Dict[str, float], str, bool]:
    """Refine the 8 landmarks from the color-independent pants silhouette.

    The waist end should remain one connected cross-section, while the hem end
    should split into two legs. This gives the longitudinal direction even when
    the shorts are rotated or their color differs from the pose training set.
    """
    if mask is None or not bool(getattr(args, "pose_mask_geometry_refine", True)):
        return keypoints_board, keypoints_px, keypoint_conf, "", False

    filled = np.zeros_like(mask.mask_u8, dtype=np.uint8)
    cv2.drawContours(filled, [mask.contour], -1, 255, -1)
    area_px = int(cv2.countNonZero(filled))
    if area_px < int(getattr(args, "pose_mask_geometry_min_area_px", 1800)):
        return keypoints_board, keypoints_px, keypoint_conf, "", False

    max_points = max(10000, int(getattr(args, "pose_mask_geometry_max_points", 180000)))
    sample_step = max(1, int(math.ceil(math.sqrt(float(area_px) / float(max_points)))))
    sampled = filled[::sample_step, ::sample_step]
    ys, xs = np.where(sampled > 0)
    if len(xs) < 500:
        return keypoints_board, keypoints_px, keypoint_conf, "", False
    points_px = np.column_stack([
        xs.astype(np.float32) * float(sample_step),
        ys.astype(np.float32) * float(sample_step),
    ])
    center_px = np.mean(points_px, axis=0).astype(np.float32)
    rel = points_px - center_px
    covariance = np.cov(rel.T)
    if covariance.shape != (2, 2) or not np.all(np.isfinite(covariance)):
        return keypoints_board, keypoints_px, keypoint_conf, "", False
    _eigvals, eigvecs = np.linalg.eigh(covariance)

    def mean_existing(names: Sequence[str]) -> Optional[np.ndarray]:
        values = [
            np.asarray(keypoints_px[name], dtype=np.float32)
            for name in names if name in keypoints_px
        ]
        if not values:
            return None
        return np.mean(np.stack(values, axis=0), axis=0)

    predicted_waist = mean_existing(
        ("waist_img_left", "waist_center", "waist_img_right")
    )
    predicted_hem = mean_existing(
        ("img_left_hem_outer", "img_left_hem_inner",
         "img_right_hem_inner", "img_right_hem_outer")
    )
    pose_axis = None
    if predicted_waist is not None and predicted_hem is not None:
        pose_axis = safe_unit(predicted_hem - predicted_waist)

    profiles: List[Dict[str, Any]] = []
    for eigen_index in (1, 0):
        base_axis = np.asarray(eigvecs[:, eigen_index], dtype=np.float32)
        for sign in (1.0, -1.0):
            profile = _mask_axis_profile(
                points_px, center_px, base_axis * sign, pose_axis, args,
            )
            if profile is not None:
                profiles.append(profile)
    if not profiles:
        return keypoints_board, keypoints_px, keypoint_conf, "", False
    profiles.sort(key=lambda item: float(item["score"]), reverse=True)
    best = profiles[0]
    second_score = float(profiles[1]["score"]) if len(profiles) > 1 else -1e9
    if float(best["score"]) < float(getattr(args, "pose_mask_geometry_min_score", 6.0)):
        return keypoints_board, keypoints_px, keypoint_conf, "", False
    if float(best["waist_single_rate"]) < float(
            getattr(args, "pose_mask_geometry_min_waist_single_rate", 0.50)):
        return keypoints_board, keypoints_px, keypoint_conf, "", False
    if float(best["hem_split_rate"]) < float(
            getattr(args, "pose_mask_geometry_min_hem_split_rate", 0.50)):
        return keypoints_board, keypoints_px, keypoint_conf, "", False
    if best["first_stable_split"] is None:
        return keypoints_board, keypoints_px, keypoint_conf, "", False
    if float(best["score"]) - second_score < float(
            getattr(args, "pose_mask_geometry_min_score_margin", 0.20)):
        return keypoints_board, keypoints_px, keypoint_conf, "", False

    axis = np.asarray(best["axis"], dtype=np.float32)
    lateral = np.asarray([-axis[1], axis[0]], dtype=np.float32)
    predicted_side = None
    if "waist_img_left" in keypoints_px and "waist_img_right" in keypoints_px:
        predicted_side = (
            np.asarray(keypoints_px["waist_img_right"], dtype=np.float32)
            - np.asarray(keypoints_px["waist_img_left"], dtype=np.float32)
        )
    elif "img_left_hem_outer" in keypoints_px and "img_right_hem_outer" in keypoints_px:
        predicted_side = (
            np.asarray(keypoints_px["img_right_hem_outer"], dtype=np.float32)
            - np.asarray(keypoints_px["img_left_hem_outer"], dtype=np.float32)
        )
    if predicted_side is not None and safe_norm(predicted_side) > 1e-6:
        if float(np.dot(lateral, predicted_side)) < 0.0:
            lateral = -lateral
    elif float(lateral[0]) < 0.0:
        lateral = -lateral

    rel = points_px - center_px
    v = rel @ axis
    u = rel @ lateral
    v_min = float(np.percentile(v, 0.5))
    v_max = float(np.percentile(v, 99.5))
    v_span = max(1e-6, v_max - v_min)
    t = (v - v_min) / v_span
    lateral_span = float(np.percentile(u, 99.5) - np.percentile(u, 0.5))
    band = float(getattr(args, "pose_mask_geometry_band_ratio", 0.018))
    gap_px = max(
        float(getattr(args, "pose_mask_geometry_min_gap_px", 6.0)),
        lateral_span * float(getattr(args, "pose_mask_geometry_gap_width_ratio", 0.025)),
    )
    min_width = max(
        float(getattr(args, "pose_mask_geometry_min_interval_width_px", 8.0)),
        lateral_span * 0.035,
    )
    min_points = int(getattr(args, "pose_mask_geometry_min_section_points", 12))

    def intervals_at(sample: float) -> List[Tuple[float, float, int]]:
        return _mask_cross_section_intervals(
            u, t, sample, band, gap_px, min_points, min_width,
        )

    waist_options = []
    for sample in np.linspace(0.04, 0.18, 8):
        items = intervals_at(float(sample))
        if len(items) != 1:
            continue
        interval = items[0]
        width = float(interval[1]) - float(interval[0])
        option_score = width / max(lateral_span, 1.0) - 2.0 * abs(float(sample) - 0.08)
        waist_options.append((option_score, float(sample), interval))
    if not waist_options:
        return keypoints_board, keypoints_px, keypoint_conf, "", False
    _, waist_t, waist_interval = max(waist_options, key=lambda item: item[0])

    hem_options = []
    for sample in np.linspace(0.78, 0.97, 11):
        pair = _outer_leg_intervals(intervals_at(float(sample)))
        if pair is None:
            continue
        left, right = pair
        total_width = max(1.0, float(right[1]) - float(left[0]))
        gap_ratio = max(0.0, float(right[0]) - float(left[1])) / total_width
        option_score = 2.5 * gap_ratio - abs(float(sample) - 0.92)
        hem_options.append((option_score, float(sample), left, right, gap_ratio))
    if not hem_options:
        return keypoints_board, keypoints_px, keypoint_conf, "", False
    _, hem_t, left_hem_interval, right_hem_interval, hem_gap_ratio = max(
        hem_options, key=lambda item: item[0],
    )

    final_split_scan = []
    for sample in np.linspace(0.24, 0.88, 33):
        pair = _outer_leg_intervals(intervals_at(float(sample)))
        if pair is None:
            continue
        left, right = pair
        total_width = max(1.0, float(right[1]) - float(left[0]))
        gap_ratio = max(0.0, float(right[0]) - float(left[1])) / total_width
        if gap_ratio >= float(getattr(args, "pose_mask_geometry_min_split_gap_ratio", 0.025)):
            final_split_scan.append((float(sample), left, right, float(gap_ratio)))
    stable_split = None
    for index in range(len(final_split_scan) - 1):
        current = final_split_scan[index]
        following = final_split_scan[index + 1]
        if float(following[0]) - float(current[0]) <= 0.031:
            stable_split = current
            break
    if stable_split is None:
        return keypoints_board, keypoints_px, keypoint_conf, "", False
    split_t, split_left, split_right, split_gap_ratio = stable_split

    def point_from_local(u_value: float, t_value: float) -> np.ndarray:
        v_value = v_min + float(t_value) * v_span
        point = center_px + axis * v_value + lateral * float(u_value)
        point[0] = float(np.clip(point[0], 1.0, filled.shape[1] - 2.0))
        point[1] = float(np.clip(point[1], 1.0, filled.shape[0] - 2.0))
        return point.astype(np.float32)

    waist_left_u = float(waist_interval[0])
    waist_right_u = float(waist_interval[1])
    crotch_u = 0.5 * (float(split_left[1]) + float(split_right[0]))
    geometry_points = {
        "waist_img_left": point_from_local(waist_left_u, waist_t),
        "waist_center": point_from_local(0.5 * (waist_left_u + waist_right_u), waist_t),
        "waist_img_right": point_from_local(waist_right_u, waist_t),
        "crotch": point_from_local(crotch_u, float(split_t)),
        "img_left_hem_outer": point_from_local(float(left_hem_interval[0]), hem_t),
        "img_left_hem_inner": point_from_local(float(left_hem_interval[1]), hem_t),
        "img_right_hem_inner": point_from_local(float(right_hem_interval[0]), hem_t),
        "img_right_hem_outer": point_from_local(float(right_hem_interval[1]), hem_t),
    }

    mask_weight = float(np.clip(
        float(getattr(args, "pose_mask_geometry_weight", 0.85)), 0.0, 1.0,
    ))
    max_blend_distance = float(getattr(
        args, "pose_mask_geometry_max_blend_distance_px", 75.0,
    ))
    geometry_conf = float(np.clip(
        float(getattr(args, "pose_mask_geometry_fallback_conf", 0.45)), 0.0, 1.0,
    ))
    refined_px = dict(keypoints_px)
    refined_board = dict(keypoints_board)
    refined_conf = dict(keypoint_conf)
    full_geometry_count = 0

    for name, geometry_point in geometry_points.items():
        final_point = np.asarray(geometry_point, dtype=np.float32)
        original = keypoints_px.get(name)
        semantic_ok = False
        if original is not None:
            original_array = np.asarray(original, dtype=np.float32)
            original_t = (
                float(np.dot(original_array - center_px, axis)) - v_min
            ) / v_span
            if name.startswith("waist_"):
                semantic_ok = original_t <= 0.30
            elif name == "crotch":
                semantic_ok = abs(original_t - float(split_t)) <= 0.22
            else:
                semantic_ok = original_t >= 0.62
            distance = float(np.linalg.norm(original_array - geometry_point))
            if semantic_ok and distance <= max_blend_distance:
                final_point = (
                    mask_weight * geometry_point
                    + (1.0 - mask_weight) * original_array
                ).astype(np.float32)
            else:
                full_geometry_count += 1
        else:
            full_geometry_count += 1

        bx, by = pixel_to_board(H, float(final_point[0]), float(final_point[1]))
        if not (np.isfinite(bx) and np.isfinite(by)):
            return keypoints_board, keypoints_px, keypoint_conf, "", False
        refined_px[name] = (float(final_point[0]), float(final_point[1]))
        refined_board[name] = (float(bx), float(by))
        refined_conf[name] = max(float(refined_conf.get(name, 0.0)), geometry_conf)

    axis_angle = float(np.degrees(np.arctan2(float(axis[1]), float(axis[0]))))
    note = (
        "mask-geometry refined "
        f"(axis={axis_angle:+.1f}deg split={float(split_t):.2f} "
        f"waist1={float(best['waist_single_rate']):.2f} "
        f"hem2={float(best['hem_split_rate']):.2f} "
        f"gaps={float(split_gap_ratio):.2f}/{float(hem_gap_ratio):.2f} "
        f"full={full_geometry_count}/8)"
    )
    return refined_board, refined_px, refined_conf, note, True


def evaluate_final_bottom_pose_geometry(
        keypoints_board: Dict[str, Tuple[float, float]],
        keypoints_px: Dict[str, Tuple[float, float]],
        keypoint_conf: Dict[str, float],
        mask: Optional[BottomMaskBoard],
        tta_summary: Optional[Dict[str, Any]],
        args,
) -> Dict[str, Any]:
    """Validate final landmarks without changing their semantic labels."""
    summary = dict(tta_summary or {})
    required = (
        "waist_img_left", "waist_center", "waist_img_right", "crotch",
        "img_left_hem_outer", "img_left_hem_inner",
        "img_right_hem_inner", "img_right_hem_outer",
    )
    visible_count = sum(name in keypoints_board for name in required)
    missing = [name for name in required if name not in keypoints_board]
    hard_failures: List[str] = []
    if visible_count < int(getattr(args, "pose_geometry_min_visible", 7)):
        hard_failures.append("visible_points")
    for name in ("waist_img_left", "waist_center", "waist_img_right", "crotch"):
        if name not in keypoints_board:
            hard_failures.append(name)

    def board_mean(names: Sequence[str]) -> Optional[np.ndarray]:
        points = [
            np.asarray(keypoints_board[name], dtype=np.float32)
            for name in names if name in keypoints_board
        ]
        return None if not points else np.mean(np.stack(points, axis=0), axis=0)

    def px_mean(names: Sequence[str]) -> Optional[np.ndarray]:
        points = [
            np.asarray(keypoints_px[name], dtype=np.float32)
            for name in names if name in keypoints_px
        ]
        return None if not points else np.mean(np.stack(points, axis=0), axis=0)

    left_hem_names = ("img_left_hem_outer", "img_left_hem_inner")
    right_hem_names = ("img_right_hem_inner", "img_right_hem_outer")
    left_hem = board_mean(left_hem_names)
    right_hem = board_mean(right_hem_names)
    left_hem_px = px_mean(left_hem_names)
    right_hem_px = px_mean(right_hem_names)

    waist_center_offset_ratio = float("inf")
    waist_axis_dot = 1.0
    crotch_axis_t = float("nan")
    crotch_lateral_ratio = float("inf")
    hem_gap_ratio = 0.0
    axis_to_waist_ratio = 0.0
    waist_width = 0.0
    body_axis_len = 0.0
    if all(name in keypoints_board for name in ("waist_img_left", "waist_center", "waist_img_right", "crotch")) \
            and left_hem is not None and right_hem is not None:
        waist_left = np.asarray(keypoints_board["waist_img_left"], dtype=np.float32)
        waist_center = np.asarray(keypoints_board["waist_center"], dtype=np.float32)
        waist_right = np.asarray(keypoints_board["waist_img_right"], dtype=np.float32)
        crotch = np.asarray(keypoints_board["crotch"], dtype=np.float32)
        lower = 0.5 * (left_hem + right_hem)
        waist_vec = waist_right - waist_left
        body_axis = lower - waist_center
        waist_width = safe_norm(waist_vec)
        body_axis_len = safe_norm(body_axis)
        waist_u = safe_unit(waist_vec)
        body_u = safe_unit(body_axis)
        if waist_u is None or body_u is None:
            hard_failures.append("degenerate_axes")
        else:
            waist_center_offset_ratio = safe_norm(
                waist_center - 0.5 * (waist_left + waist_right)
            ) / max(1.0, waist_width)
            waist_axis_dot = abs(float(np.dot(waist_u, body_u)))
            crotch_axis_t = float(np.dot(crotch - waist_center, body_u)) / max(1.0, body_axis_len)
            lateral = np.asarray([-body_u[1], body_u[0]], dtype=np.float32)
            crotch_lateral_ratio = abs(float(np.dot(crotch - waist_center, lateral))) / max(1.0, waist_width)
            hem_gap_ratio = safe_norm(right_hem - left_hem) / max(1.0, waist_width)
            axis_to_waist_ratio = body_axis_len / max(1.0, waist_width)

            if waist_center_offset_ratio > float(getattr(args, "pose_geometry_max_waist_center_offset_ratio", 0.60)):
                hard_failures.append("waist_center")
            if waist_axis_dot > float(getattr(args, "pose_geometry_max_waist_axis_dot", 0.72)):
                hard_failures.append("waist_body_axes")
            if not (
                float(getattr(args, "pose_geometry_crotch_t_min", 0.04))
                <= crotch_axis_t
                <= float(getattr(args, "pose_geometry_crotch_t_max", 1.05))
            ):
                hard_failures.append("crotch_longitudinal")
            if crotch_lateral_ratio > float(getattr(args, "pose_geometry_max_crotch_lateral_ratio", 0.70)):
                hard_failures.append("crotch_lateral")
            if waist_width < float(getattr(args, "pose_geometry_min_waist_width_mm", 60.0)):
                hard_failures.append("waist_width")
            if body_axis_len < float(getattr(args, "pose_geometry_min_axis_mm", 80.0)):
                hard_failures.append("body_axis")
    else:
        hard_failures.append("incomplete_structure")

    mask_inside_ratio = None
    if mask is not None and getattr(mask, "contour", None) is not None:
        signed = [
            float(cv2.pointPolygonTest(mask.contour, tuple(map(float, keypoints_px[name])), True))
            for name in required if name in keypoints_px
        ]
        if signed:
            max_outside = float(getattr(args, "pose_tta_mask_max_outside_px", 20.0))
            mask_inside_ratio = float(np.mean(np.asarray(signed, dtype=np.float32) >= -max_outside))
            if mask_inside_ratio < float(getattr(args, "pose_geometry_min_mask_inside_ratio", 0.60)):
                hard_failures.append("mask_support")

    pca = _mask_pca_geometry_px(mask, args)
    pca_alignment = None
    if (
        pca is not None
        and all(name in keypoints_px for name in ("waist_img_left", "waist_img_right", "waist_center"))
        and left_hem_px is not None
        and right_hem_px is not None
    ):
        waist_axis_px = (
            np.asarray(keypoints_px["waist_img_right"], dtype=np.float32)
            - np.asarray(keypoints_px["waist_img_left"], dtype=np.float32)
        )
        lower_px = 0.5 * (left_hem_px + right_hem_px)
        body_axis_px = lower_px - np.asarray(keypoints_px["waist_center"], dtype=np.float32)
        pca_alignment = _pose_pca_pair_alignment(waist_axis_px, body_axis_px, pca)
        if (
            bool(pca.get("reliable", False))
            and pca_alignment is not None
            and pca_alignment < float(getattr(args, "pose_geometry_pca_min_alignment", 0.50))
        ):
            hard_failures.append("mask_pca_axes")

    mean_conf = float(np.mean(list(keypoint_conf.values()))) if keypoint_conf else 0.0
    component_scores = {
        "visible": float(np.clip(visible_count / 8.0, 0.0, 1.0)),
        "confidence": float(np.clip(mean_conf / 0.55, 0.0, 1.0)),
        "waist_center": float(np.clip(1.0 - waist_center_offset_ratio / 0.75, 0.0, 1.0)),
        "axis_orthogonality": float(np.clip(1.0 - waist_axis_dot / 0.85, 0.0, 1.0)),
        "crotch_lateral": float(np.clip(1.0 - crotch_lateral_ratio / 0.85, 0.0, 1.0)),
        "hem_open": float(np.clip(hem_gap_ratio / 0.55, 0.0, 1.0)),
        "mask_support": float(mask_inside_ratio if mask_inside_ratio is not None else 0.70),
        "pca_alignment": float(pca_alignment if pca_alignment is not None else 0.70),
        "tta_structure": 1.0 if bool(summary.get("structure_reliable", False)) else 0.45,
    }
    geometry_score = float(
        0.12 * component_scores["visible"]
        + 0.08 * component_scores["confidence"]
        + 0.14 * component_scores["waist_center"]
        + 0.14 * component_scores["axis_orthogonality"]
        + 0.12 * component_scores["crotch_lateral"]
        + 0.10 * component_scores["hem_open"]
        + 0.12 * component_scores["mask_support"]
        + 0.10 * component_scores["pca_alignment"]
        + 0.08 * component_scores["tta_structure"]
    )
    hard_failures = list(dict.fromkeys(hard_failures))
    closed_crotch_inferred = bool(summary.get("closed_crotch_inferred", False))
    min_score = float(getattr(
        args,
        "pose_geometry_closed_crotch_min_score"
        if closed_crotch_inferred else "pose_geometry_final_min_score",
        0.56 if closed_crotch_inferred else 0.62,
    ))
    reliable = bool(not hard_failures and geometry_score >= min_score)
    need_pre_spread = bool(summary.get("need_pre_spread", False))
    crotch_occluded = bool(summary.get("crotch_occluded", False))
    if crotch_occluded:
        need_pre_spread = True
    if hem_gap_ratio < float(getattr(args, "pose_tta_pre_spread_hem_gap_ratio", 0.42)):
        need_pre_spread = True
    recovery_required = bool(not reliable or need_pre_spread)
    recovery_reasons = list(hard_failures)
    if need_pre_spread:
        recovery_reasons.append("collapsed_or_hidden_hem")
    if crotch_occluded:
        recovery_reasons.append("crotch_occluded")
    if geometry_score < min_score:
        recovery_reasons.append("geometry_score")
    recovery_reasons = list(dict.fromkeys(recovery_reasons))
    return {
        "reliable": reliable,
        "geometry_score": geometry_score,
        "score_required": min_score,
        "recovery_required": recovery_required,
        "recovery_reasons": recovery_reasons,
        "hard_failures": hard_failures,
        "component_scores": component_scores,
        "visible_count": visible_count,
        "missing": missing,
        "mean_conf": mean_conf,
        "waist_width_mm": waist_width,
        "body_axis_len_mm": body_axis_len,
        "axis_to_waist_ratio": axis_to_waist_ratio,
        "waist_center_offset_ratio": waist_center_offset_ratio,
        "waist_axis_dot": waist_axis_dot,
        "crotch_axis_t": crotch_axis_t,
        "crotch_lateral_ratio": crotch_lateral_ratio,
        "hem_gap_ratio": hem_gap_ratio,
        "mask_inside_ratio": mask_inside_ratio,
        "pca_alignment": pca_alignment,
        "pca_axis_ratio": None if pca is None else float(pca.get("axis_ratio", 1.0)),
        "pca_reliable": False if pca is None else bool(pca.get("reliable", False)),
        "tta_score_margin": summary.get("tta_score_margin"),
        "tta_structure_reliable": bool(summary.get("structure_reliable", False)),
        "crotch_state": str(summary.get("crotch_state", "UNKNOWN")),
        "crotch_confidence": float(summary.get("crotch_confidence", 0.0)),
        "closed_crotch_inferred": bool(closed_crotch_inferred),
        "pre_spread_required": bool(summary.get("pre_spread_required", False)),
        "closed_crotch_inference": summary.get("closed_crotch_inference"),
    }


def infer_bottom_pose(pose_model, frame: np.ndarray, H: Optional[np.ndarray], imgsz: int, conf: float,
                      kpt_conf: float, mask: Optional[BottomMaskBoard] = None,
                      wrinkle: Optional[Dict[str, Any]] = None,
                      orientation_fix: bool = True, flip_ratio: float = 1.15,
                      flip_margin_mm: float = 20.0,
                      orientation_ref_y: Optional[float] = None,
                      flip_ref_margin_mm: float = 35.0,
                      tta_args=None) -> Tuple[Optional[BottomPoseBoard], str]:
    if H is None:
        return None, "Homography not locked"

    def rectangle_fallback(reason: str) -> Tuple[Optional[BottomPoseBoard], str]:
        recovered = build_rectangle_fallback_pose(
            mask, H, frame.shape, wrinkle, tta_args, trigger_reason=reason,
        )
        if recovered is not None:
            return recovered, f"OK; {recovered.inference_status}; trigger={reason}"
        return None, reason

    if pose_model is None:
        return rectangle_fallback("pose model is None")
    d2_strict = bool(getattr(tta_args, "pose_d2_strict", True))
    tta_summary: Dict[str, Any] = {}
    if bool(getattr(tta_args, "pose_tta", True)):
        try:
            kxy, kcf, status, tta_summary = infer_best_pose_with_tta(
                pose_model, frame, int(imgsz), float(conf), float(kpt_conf),
                tta_args, mask=mask, wrinkle=wrinkle,
            )
        except Exception as e:
            return rectangle_fallback(f"pose TTA error: {repr(e)}")
    else:
        try:
            result = pose_model.predict(source=frame, imgsz=int(imgsz), conf=float(conf), verbose=False)[0]
        except Exception as e:
            return rectangle_fallback(f"pose predict error: {repr(e)}")
        kxy, kcf, status = read_pose_keypoints(result, frame.shape, kpt_conf)
        if kxy is not None and kcf is not None:
            tta_summary = pose_tta_summary(
                kxy, kcf, float(kpt_conf), frame.shape, tta_args, mask=mask,
            )
    if kxy is None or kcf is None:
        return rectangle_fallback(status)

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

    geometry_note = ""
    geometry_applied = False
    if mask is not None and not d2_strict:
        (
            keypoints_board,
            keypoints_px,
            keypoint_conf,
            geometry_note,
            geometry_applied,
        ) = refine_bottom_pose_from_mask_geometry(
            keypoints_board,
            keypoints_px,
            keypoint_conf,
            mask,
            H,
            tta_args,
        )

    required = ["waist_center", "crotch"]
    missing_required = [n for n in required if n not in keypoints_board]
    hem_names = ["img_left_hem_outer", "img_left_hem_inner", "img_right_hem_inner", "img_right_hem_outer"]
    valid_hem_count = sum(1 for n in hem_names if n in keypoints_board)
    waist_edge_ok = "waist_img_left" in keypoints_board and "waist_img_right" in keypoints_board
    if missing_required:
        return rectangle_fallback(f"missing required bottom pose keypoints: {missing_required}")
    if valid_hem_count < 3:
        return rectangle_fallback(f"not enough hem keypoints: {valid_hem_count}/4")

    orientation_note = ""
    if not d2_strict and not geometry_applied:
        keypoints_board, keypoints_px, keypoint_conf, orientation_note = maybe_fix_bottom_pose_orientation(
            keypoints_board,
            keypoints_px,
            keypoint_conf,
            enable=bool(orientation_fix),
            flip_ratio=float(flip_ratio),
            flip_margin_mm=float(flip_margin_mm),
            orientation_ref_y=orientation_ref_y,
            flip_ref_margin_mm=float(flip_ref_margin_mm),
        )
    waist_edge_ok = "waist_img_left" in keypoints_board and "waist_img_right" in keypoints_board

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

    geometry_validation = bool(getattr(tta_args, "pose_geometry_validation", True))
    geometry_report = evaluate_final_bottom_pose_geometry(
        keypoints_board,
        keypoints_px,
        keypoint_conf,
        mask,
        tta_summary,
        tta_args,
    )
    geometry_reliable = bool(geometry_report.get("reliable", False)) if geometry_validation else True
    geometry_score = float(geometry_report.get("geometry_score", 0.0))
    crotch_state = str(tta_summary.get("crotch_state", "UNKNOWN"))
    crotch_confidence = float(tta_summary.get(
        "crotch_confidence", keypoint_conf.get("crotch", 0.0),
    ))
    closed_crotch_usable = bool(
        tta_summary.get("closed_crotch_usable", False)
        or (
            tta_summary.get("closed_crotch_inferred", False)
            and crotch_confidence >= float(getattr(
                tta_args, "pose_closed_crotch_min_confidence", 0.50,
            ))
        )
    )
    pre_spread_required = bool(
        tta_summary.get("pre_spread_required", False)
        or tta_summary.get("crotch_occluded", False)
    )
    recovery_required = bool(geometry_report.get("recovery_required", False)) if geometry_validation else bool(
        tta_summary.get("need_pre_spread", False)
    )
    status += (
        f" final-geom={'OK' if geometry_reliable else 'LOW'}"
        f"({geometry_score:.2f}/{float(geometry_report.get('score_required', 0.62)):.2f})"
    )

    valid = True
    reasons = []
    if "polarity=UNRESOLVED" in status:
        valid = False
        if "CROTCH_OCCLUDED" in status:
            reasons.append("crotch gap occluded; pre-spread and re-infer")
        else:
            reasons.append("mask/pose direction conflict unresolved")
    elif bool(tta_summary.get("crotch_occluded", False)) and not closed_crotch_usable:
        valid = False
        reasons.append("waistband direction is reliable but crotch remains unresolved; pre-spread and re-infer")
    elif bool(tta_summary.get("crotch_occluded", False)) and closed_crotch_usable:
        status += (
            f" closed-crotch={crotch_state}"
            f"({crotch_confidence:.2f},pre-spread)"
        )
    if pants_axis_len < 80.0:
        valid = False
        reasons.append(f"pants_axis too short {pants_axis_len:.1f}mm")
    if geometry_validation and not geometry_reliable:
        valid = False
        geometry_reasons = list(geometry_report.get("recovery_reasons", []))
        reasons.append(
            "pose geometry unreliable"
            + (f": {','.join(str(item) for item in geometry_reasons[:5])}" if geometry_reasons else "")
        )
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

    ok_parts = ["OK", status]
    if geometry_note:
        ok_parts.append(geometry_note)
    if orientation_note:
        ok_parts.append(orientation_note)
    ok_reason = "; ".join([p for p in ok_parts if p])
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
        reason=ok_reason if valid else "; ".join(
            reasons
            + ([geometry_note] if geometry_note else [])
            + ([orientation_note] if orientation_note else [])
        ),
        inference_status=status,
        geometry_reliable=geometry_reliable,
        geometry_score=geometry_score,
        recovery_required=recovery_required,
        geometry_metrics=geometry_report,
        crotch_state=crotch_state,
        crotch_confidence=crotch_confidence,
        polarity_metrics={
            "canonical_source": tta_summary.get("canonical_source"),
            "closed_crotch_inference": tta_summary.get("closed_crotch_inference"),
            "crotch_concavity": tta_summary.get("crotch_concavity"),
        },
        pre_spread_required=pre_spread_required,
    )
    if not valid:
        recovered, recovered_status = rectangle_fallback(pose.reason)
        if recovered is not None:
            return recovered, recovered_status
    return pose, "OK" if valid else pose.reason



def infer_bottom_observation(
        seg_model,
        pose_model,
        frame: np.ndarray,
        H: Optional[np.ndarray],
        args,
        cfg: Optional[Any] = None,
) -> BottomObservation:
    """Run the complete E45 perception/geometry pipeline.

    This function deliberately performs no action classification. A future
    bottom_policy module should consume the returned BottomObservation and set
    action_decision independently.

    Args:
        seg_model: Ultralytics segmentation model or a compatible object.
        pose_model: Ultralytics pose model or a compatible object.
        frame: BGR camera frame.
        H: Pixel-to-board homography.
        args: Namespace/config object containing the existing E44 perception
            parameters. Missing common parameters receive conservative defaults.
        cfg: Optional runtime/safety config. Only min_mask_area_px and
            waist_ref_y are read, so the perception module does not depend on
            the robot-control dataclass.
    """
    if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
        return BottomObservation(None, None, None, False, "empty camera frame", frame)
    if H is None:
        return BottomObservation(None, None, None, False, "Homography not locked", frame)

    seg_classes = parse_class_names(getattr(args, "seg_classes", "bottoms")) or ["bottoms"]
    seg_imgsz = int(getattr(args, "seg_imgsz", 640))
    seg_conf = float(getattr(args, "seg_conf", 0.25))
    pose_imgsz = int(getattr(args, "pose_imgsz", 640))
    pose_conf = float(getattr(args, "pose_conf", 0.25))
    pose_kpt_conf = float(getattr(args, "pose_kpt_conf", 0.20))

    mask, mask_status = infer_bottoms_mask(
        seg_model,
        frame,
        H,
        seg_imgsz,
        seg_conf,
        target_class_names=seg_classes,
        args=args,
    )

    analysis_frame, glare = suppress_specular_reflections(
        frame,
        None if mask is None else mask.mask_u8,
        args,
    )
    if (
        mask is None
        and bool(glare.get("applied", False))
        and bool(getattr(args, "glare_retry_segmentation", True))
    ):
        retry_mask, retry_status = infer_bottoms_mask(
            seg_model,
            analysis_frame,
            H,
            seg_imgsz,
            seg_conf,
            target_class_names=seg_classes,
            args=args,
        )
        if retry_mask is not None:
            mask = retry_mask
            mask_status = f"{retry_status}; glare-corrected retry"
            glare["segmentation_retry_recovered"] = True
        else:
            glare["segmentation_retry_recovered"] = False

    wrinkle = None
    if mask is not None and bool(getattr(args, "wrinkle_heatmap_mode", True)):
        wrinkle = build_wrinkle_heatmap(analysis_frame, mask.mask_u8, args)

    cfg_waist_ref_y = float(getattr(cfg, "waist_ref_y", -390.0)) if cfg is not None else -390.0
    orientation_ref_y = (
        float(getattr(args, "pose_orientation_ref_y"))
        if getattr(args, "pose_orientation_ref_y", None) is not None
        else (
            cfg_waist_ref_y
            if bool(getattr(args, "pose_orientation_use_waist_ref", False))
            else None
        )
    )
    def _infer_pose_for_mask(current_mask: Optional[BottomMaskBoard], current_wrinkle):
        return infer_bottom_pose(
            pose_model,
            analysis_frame,
            H,
            pose_imgsz,
            pose_conf,
            pose_kpt_conf,
            mask=current_mask,
            wrinkle=current_wrinkle,
            orientation_fix=bool(getattr(args, "pose_orientation_fix", False)),
            flip_ratio=float(getattr(args, "pose_flip_ratio", 1.15)),
            flip_margin_mm=float(getattr(args, "pose_flip_margin_mm", 20.0)),
            orientation_ref_y=orientation_ref_y,
            flip_ref_margin_mm=float(getattr(args, "pose_flip_ref_margin_mm", 35.0)),
            tta_args=args,
        )

    pose, pose_status = _infer_pose_for_mask(mask, wrinkle)

    # Second sanitation pass: now prefer the board component containing the Pose.
    # If this changes the mask substantially, run Pose once more on the final mask.
    if mask is not None and bool(getattr(args, "board_roi", True)):
        previous_mask = mask
        sanitized_u8, post_board_info = sanitize_board_mask_u8(
            previous_mask.mask_u8, H, frame.shape, args, pose=pose, require_pose=(pose is not None),
        )
        args._board_roi_last_info = dict(post_board_info)
        if sanitized_u8 is None:
            if bool(getattr(args, "board_roi_strict", True)):
                mask = None
                mask_status = f"board ROI post-Pose sanitation blocked: {post_board_info.get('reason', '')}"
            else:
                previous_mask.board_roi_info = dict(post_board_info)
        else:
            post_iou = _mask_iou_u8(previous_mask.mask_u8, sanitized_u8)
            post_board_info["pre_post_iou"] = float(post_iou)
            rebuilt = rebuild_bottom_mask_from_u8(sanitized_u8, H, previous_mask, post_board_info)
            if rebuilt is None:
                if bool(getattr(args, "board_roi_strict", True)):
                    mask = None
                    mask_status = "board ROI component rebuild failed"
            else:
                mask = rebuilt
                rerun_threshold = float(np.clip(getattr(args, "board_mask_pose_rerun_iou", 0.86), 0.0, 1.0))
                if post_iou < rerun_threshold:
                    post_board_info["pose_rerun"] = True
                    analysis_frame, glare = suppress_specular_reflections(frame, mask.mask_u8, args)
                    wrinkle = (
                        build_wrinkle_heatmap(analysis_frame, mask.mask_u8, args)
                        if bool(getattr(args, "wrinkle_heatmap_mode", True)) else None
                    )
                    pose, pose_status = _infer_pose_for_mask(mask, wrinkle)
                else:
                    post_board_info["pose_rerun"] = False
                mask.board_roi_info = dict(post_board_info)

    fused: Optional[np.ndarray] = None
    valid = True
    reasons: List[str] = []
    min_mask_area_px = float(
        getattr(cfg, "min_mask_area_px", getattr(args, "min_mask_area_px", 1200.0))
        if cfg is not None
        else getattr(args, "min_mask_area_px", 1200.0)
    )
    if mask is None:
        valid = False
        reasons.append(f"mask invalid: {mask_status}")
    elif mask.area_px < min_mask_area_px:
        valid = False
        reasons.append(f"mask area too small: {mask.area_px:.0f}px")

    if pose is None:
        valid = False
        reasons.append(f"pose invalid: {pose_status}")
    elif not pose.valid:
        valid = False
        reasons.append(f"pose invalid: {pose.reason}")

    if pose is not None and mask is not None:
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
        wrinkle=wrinkle,
        action_decision=None,
        shape_debug=None,
        glare=glare,
        crotch_state=(pose.crotch_state if pose is not None else "UNKNOWN"),
        pants_polarity=(pose.polarity_metrics if pose is not None else None),
        pre_spread_required=bool(
            pose is not None and pose.pre_spread_required
        ),
        empty_baseline=(
            dict(mask.empty_baseline_info)
            if mask is not None
            else dict(getattr(args, "_empty_baseline_last_info", {}) or {
                "enabled": bool(getattr(args, "empty_baseline_veto", True)),
                "available": getattr(args, "_empty_baseline_bgr", None) is not None,
                "accepted": False,
                "reason": mask_status,
            })
        ),
        board_roi=(
            dict(mask.board_roi_info)
            if mask is not None
            else dict(getattr(args, "_board_roi_last_info", {}) or {
                "enabled": bool(getattr(args, "board_roi", True)),
                "accepted": False,
                "reason": mask_status,
            })
        ),
    )


def get_build_info() -> Dict[str, str]:
    """Return immutable version metadata for runtime logs and handoff checks."""
    return {
        "version": CODE_VERSION,
        "build_id": CODE_BUILD_ID,
        "source_baseline": SOURCE_BASELINE,
    }


__all__ = [
    "CODE_VERSION",
    "CODE_BUILD_ID",
    "SOURCE_BASELINE",
    "BOTTOM_POSE_KPT_NAMES",
    "BOTTOM_POSE_KPT",
    "BOTTOM_POSE_KPT_SHORT_NAMES",
    "BottomActionDecision",
    "BottomPoseBoard",
    "BottomMaskBoard",
    "BottomObservation",
    "pixel_to_board",
    "board_to_pixel",
    "infer_bottoms_mask",
    "evaluate_empty_baseline_candidate",
    "suppress_specular_reflections",
    "build_local_shadow_map",
    "build_wrinkle_heatmap",
    "classify_wrinkle_geometry",
    "prepare_bottom_mask_polarity",
    "evaluate_bottom_mask_polarity",
    "detect_stable_crotch_concavity",
    "evaluate_waistband_heat_evidence",
    "estimate_closed_crotch_geometry",
    "mask_landmarks_from_crotch_concavity",
    "mask_landmarks_from_closed_crotch",
    "mask_landmarks_from_polarity",
    "mask_landmarks_from_rectangle_fallback",
    "build_rectangle_fallback_pose",
    "refine_bottom_pose_from_mask_geometry",
    "evaluate_final_bottom_pose_geometry",
    "infer_best_pose_with_tta",
    "infer_bottom_pose",
    "infer_bottom_observation",
    "get_build_info",
]


if __name__ == "__main__":
    info = get_build_info()
    print(f"[BUILD] {info['build_id']}")
    print("[MODULE] combined perception+geometry; no robot action execution")
