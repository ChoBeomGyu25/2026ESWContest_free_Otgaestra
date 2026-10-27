# v36: planner geometry remains byte-equivalent to v32-v35 behavior; v36 changes only main-file target review/correction and overlay control flow.
# v32: target-local recovery reuses the already-computed RGB crease maps; no camera/Pose/Seg/full-frame heavy perception is rerun.
# v31: v30 target-guided E156 preserved; pair search optimized via prepared crease mask, vectorized cheap filters, exact branch-and-bound, and tier-shared metadata caches.
#!/usr/bin/env python3
# v51: keep v49 Action/BMR/top-K Target selection/validation, but restore the full post-Target MICRO planner geometry to v44 behavior.
# v51: ACTIVE/HOLD search, pull_hint, local outer-normal direction, pull distance, endpoint, v39/v40/v41/v42 reach/recovery and all hard safety thresholds are v44-carried; v49 fold context is admission/provenance only.
# v48: Target-direct now validates the Target against the already-computed same-X READY-CCA fold geometry before execution.
# v48: validated Target-direct uses the matched crease normal (BODY->flap, centroid-away sign), reuses E122 same-fold pair semantics, and caps E111 contour-normal drift; no new camera/Seg/Pose/wrinkle inference is added.
# v49: same-X cached local fold context from the main runtime is authoritative for Target/direction validation; READY-CCA is retained only as optional soft support and never hard-blocks Target.
# v49: all v31-v42 bounded recovery, 420mm reach, 65mm path-clearance, pair/board/contact-Z safety, and FULL-E156-OFF defaults are unchanged.
# v43: planner/executor core intentionally preserves v42 geometry, fast paths, E156-off default, cached PULL recovery, and every hard safety threshold; v43 orchestration changes live in the main runtime.
# v41: preserves v40 Target-direct/recoveries and adds bounded PULL-direction/length reach recovery, FAST target-local before FULL E156, and explicit cache reuse.
# v40: preserves v39 Target-direct + reach-aware HOLD recovery and adds a final bounded ACTIVE+HOLD+PULL recovery before E156 fallback.
# v40: exact failure reasons are split into ACTIVE/HOLD/PULL reach, path-clearance, pull-endpoint, pair/board categories; all hard safety remains unchanged.
# v39: preserves v38 Target-direct primary + E156 fallback, adding only bounded reach-aware HOLD recovery after normal Target-direct HOLD/pair failure.
# v39: hard 420mm reach, 65mm path clearance, 66..126mm pair spacing, board/contact-Z/executor semantics remain unchanged.
# -*- coding: utf-8 -*-
# v30: micro_target_v2 is used during safe crease localization/candidate search as well as final safe-plan selection.
# v30: ACTIVE/HOLD pair search rejects same-crease candidates that cannot satisfy the final RGB-crease crossing semantics, so search continues instead of stopping on a mechanically-safe but role-invalid pair.
# v30: target-guided same-crease anchor/reach recovery expands search without relaxing the 420mm reach, 65mm path-clearance, board, spacing, or final role gates.
"""E156 Micro-Unfold perception/planner core adapted for vla_0856 integration.

V14 keeps V13's local fold detection and hard mechanics, fixes the retained-plan
fallthrough bug, and adds bounded SAME-CREASE anchor search. A mechanically-safe
directional plan is never lost merely because later alternative search has no
components. If the exact crease center is unreachable, anchors are sampled only at
relative positions along that SAME real crease before the existing small near-normal
fan is tried. No unrelated crease, garment-state gate, or safety relaxation is added.
Board bounds, 66..126mm pair spacing, 65mm path clearance and the 420mm hard reach
limit remain mandatory.

V14 intentionally makes no planner/geometry change; execution safety changes live in the main runtime.

Full-resolution exact-plan masks stay local to planning/validation only and are
never copied into the frozen pending action.  This module
still owns no camera, segmentation model, Pose model, UI, serial port or AUTO-X
macro-state policy.
"""
from __future__ import annotations
import copy
import hashlib
import json
import math
import time
from types import SimpleNamespace
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np

E156_CORE_VERSION = "E156-VLA-CORE-V51-V49-TARGET-V44-FULL-PLANNER-GEOMETRY"


def _training_json_safe(value):
    if isinstance(value, dict):
        return {str(k): _training_json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_training_json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _board_to_arm_xy(config: Dict[str, Any], arm_key: str, bx: float, by: float) -> Tuple[float, float]:
    arm_cfg = dict(config.get("dual_roarm", {}).get(str(arm_key), {}) or {})
    M_raw = arm_cfg.get("board_to_roarm_affine_2x3")
    if M_raw is None:
        raise RuntimeError(f"{arm_key} affine unavailable")
    M = np.asarray(M_raw, dtype=np.float64).reshape(2, 3)
    out = M @ np.asarray([float(bx), float(by), 1.0], dtype=np.float64)
    off = arm_cfg.get("roarm_xy_offset", [0.0, 0.0])
    ox = float(off[0]) if isinstance(off, (list, tuple)) and len(off) >= 1 else 0.0
    oy = float(off[1]) if isinstance(off, (list, tuple)) and len(off) >= 2 else 0.0
    return float(out[0] + ox), float(out[1] + oy)


def _fix6_local_xy(config, arm_key, board_xy):
    rx, ry = _board_to_arm_xy(config, str(arm_key), float(board_xy[0]), float(board_xy[1]))
    return np.asarray([rx, ry], dtype=np.float64)


def _fix37_local_xy_to_board(config, arm_key, local_xy):
    arm_cfg = dict(config.get("dual_roarm", {}).get(str(arm_key), {}) or {})
    M_raw = arm_cfg.get("board_to_roarm_affine_2x3")
    if M_raw is None:
        raise RuntimeError(f"{arm_key} affine unavailable for inverse")
    M = np.asarray(M_raw, dtype=np.float64).reshape(2, 3)
    A = M[:, :2]
    if abs(float(np.linalg.det(A))) <= 1e-9:
        raise RuntimeError(f"{arm_key} affine singular")
    off = arm_cfg.get("roarm_xy_offset", [0.0, 0.0])
    ox = float(off[0]) if isinstance(off, (list, tuple)) and len(off) >= 1 else 0.0
    oy = float(off[1]) if isinstance(off, (list, tuple)) and len(off) >= 2 else 0.0
    local = np.asarray([float(local_xy[0]) - ox, float(local_xy[1]) - oy], dtype=np.float64)
    return np.linalg.solve(A, local - M[:, 2]).astype(np.float64)


def _fix5_grasp_pixel_to_board(H_value, point_xy):
    src = np.asarray([[[float(point_xy[0]), float(point_xy[1])]]], dtype=np.float32)
    dst = cv2.perspectiveTransform(src, np.asarray(H_value, dtype=np.float32))
    return [float(dst[0, 0, 0]), float(dst[0, 0, 1])]


def _fix12_transform_points(H_value, points_xy):
    pts = np.asarray(points_xy, dtype=np.float32).reshape(-1, 1, 2)
    dst = cv2.perspectiveTransform(pts, np.asarray(H_value, dtype=np.float32))
    return dst.reshape(-1, 2).astype(np.float64)


def standby_roarm_pose(cfg, arm_key):
    d = dict(getattr(cfg, "standby_xyz", {}) or {})
    xyz = d.get(str(arm_key))
    if xyz is None:
        raise RuntimeError(f"{arm_key} standby XYZ unavailable")
    return [float(xyz[0]), float(xyz[1]), float(xyz[2]), float(getattr(cfg, "standby_t", 3.14))]


def arm_contact_z(cfg, arm_key):
    d = dict(getattr(cfg, "contact_z", {}) or {})
    if str(arm_key) not in d or d.get(str(arm_key)) is None:
        raise RuntimeError(f"{arm_key} contact Z unavailable")
    return float(d[str(arm_key)])


def _micro_unfold_hold_z_offset(args, hold_arm):
    """Return the effective HOLD Z offset for the selected hold arm.

    ARM1 keeps the proven common HOLD offset unchanged.  ARM2 needs slightly
    firmer board contact in the physical setup, so only ARM2-as-HOLD is lowered
    by 5 mm.  The correction is applied during planning (not in the executor)
    so the frozen plan/binding and the commanded HOLD_DOWN Z remain identical.
    """
    base_offset = float(getattr(args, "micro_unfold_hold_z_offset_mm", 23.0))
    if str(hold_arm) == "arm2":
        return base_offset - 5.0
    return base_offset


def _default_micro_args() -> SimpleNamespace:
    # Only values whose E156 defaults changed across revisions are explicitly fixed here.
    # Every other helper accesses getattr(args, key, exact_E156_default), preserving E156 behavior.
    return SimpleNamespace(
        micro_unfold_pose_fold_kpt_conf=0.20,
        micro_unfold_pose_deform_kpt_conf=0.20,
        micro_unfold_pose_fold_inside_px=22.0,
        micro_unfold_pose_fold_armpit_inside_px=30.0,
        micro_unfold_e125_pose_neck_inside_px=18.0,
        micro_unfold_pose_fold_corridor_width_px=15.0,
        micro_unfold_hold_z_offset_mm=23.0,
        micro_unfold_active_z_offset_mm=26.0,
        micro_unfold_active_start_clearance_mm=18.0,
        micro_unfold_max_below_contact_mm=2.0,
        micro_unfold_max_pull_mm=264.0,
        micro_unfold_hold_boundary_factor=2.0,
        micro_unfold_outer_normal_window_px=22,
        micro_unfold_min_pair_sep_mm=70.0,
        micro_unfold_peel_xy_mm=35.0,
        micro_unfold_peel_z_mm=55.0,
        # V5 empirical reach margins learned from real RoArm logs. The original
        # 420 mm hard safety gate remains authoritative; these SOFT margins only
        # shorten an already board/outside-cloth-safe endpoint before execution.
        micro_unfold_pull_soft_radius_mm=390.0,
        micro_unfold_peel_soft_radius_mm=400.0,
        # E128 general crease planner budget. Eight anchors/component preserves broad
        # crease coverage while preventing a noisy 720p skeleton from spawning
        # hundreds of near-duplicate expensive candidates.
        micro_unfold_e128_max_anchors_per_component=8,
        micro_unfold_vla_slim_full_plan=True,
        # V6 strict joint-localization defaults.  These are VLA-only admission
        # gates; E156 mechanical reach/path/board safety remains unchanged.
        vla_gate_allow_single_neck=False,
        micro_unfold_vla_require_rgb_pose_corroboration=True,
        micro_unfold_vla_joint_min_exec_pixels=12,
        micro_unfold_vla_joint_min_branch_overlap=0.18,
        micro_unfold_vla_joint_min_orientation_alignment=0.45,
        micro_unfold_vla_joint_deform_min_conf=0.36,
        micro_unfold_vla_joint_deform_region_min_conf=0.20,
        micro_unfold_vla_joint_roi_radius_px=170.0,
        micro_unfold_vla_joint_branch_dilate_px=20,
        micro_unfold_vla_joint_min_branch_span_px=18.0,
        # V7 RGB-adaptive rescue.  These scores are dimensionless outputs of the
        # garment-relative E94/E122 pipeline; no absolute RGB value or board-center
        # distance is used to decide rescue eligibility.
        micro_unfold_vla_rgb_rescue_enable=True,
        micro_unfold_vla_rgb_rescue_edge_min_quality=0.46,
        micro_unfold_vla_rgb_rescue_global_min_quality=0.55,
        micro_unfold_vla_rgb_rescue_global_min_branch_score=0.60,
        micro_unfold_vla_rgb_rescue_neck_distance_ratio=0.55,
        micro_unfold_vla_rgb_rescue_neck_parallel_alignment=0.72,
        # V8: quality is no longer a single hard gate. These ratios are relative
        # to the existing garment-normalized V7 thresholds and therefore remain
        # independent of raw garment brightness/color.
        micro_unfold_vla_rgb_rescue_fused_min_support=0.90,
        micro_unfold_vla_rgb_rescue_min_primary_ratio=0.72,
        # V8: if one exact normal fails mechanics, keep the SAME crease/anchor and
        # search only a small fan around that normal. All original hard safety and
        # final same-crease role checks remain mandatory.
        micro_unfold_vla_near_normal_fan_deg=(-12.0, -8.0, -4.0, 4.0, 8.0, 12.0),
        micro_unfold_vla_late_reach_recovery=True,
        micro_unfold_vla_dual_reach_recovery=True,
        micro_unfold_vla_edge_flap_rescue_enable=True,
        micro_unfold_vla_exact_crease_rerank_max=6,
        micro_unfold_vla_pose_local_score_percentile=85.0,
        micro_unfold_active_arm1_closed_ref=2.900,
        micro_unfold_active_arm2_closed_ref=2.890,
        micro_unfold_hold_arm1_gripper=2.818,
        micro_unfold_hold_arm2_gripper=2.772,
        micro_unfold_active_release_deg=45.0,
        # v38 Target-direct primary planner.  These bounds are deliberately local
        # and keep the proven E156 mechanical gates authoritative.
        micro_unfold_v38_target_direct_enable=True,
        micro_unfold_v38_active_search_radius_mm=40.0,
        micro_unfold_v38_active_boundary_target_mm=25.0,
        micro_unfold_v38_active_max_candidates=8,
        micro_unfold_v38_hold_pair_min_mm=66.0,
        micro_unfold_v38_hold_pair_max_mm=126.0,
        micro_unfold_v38_hold_pair_target_mm=90.0,
        micro_unfold_v38_hold_boundary_target_mm=20.0,
        micro_unfold_v38_hold_max_candidates=18,
        micro_unfold_v38_pair_max_trials=24,
        # v39 bounded recovery: invoked only after the normal v38 HOLD search is
        # empty or its shortlisted pairs fail physical arm reach/path finalization.
        # These settings change search coverage/order only; no safety threshold is relaxed.
        micro_unfold_v39_reach_hold_recovery_enable=True,
        micro_unfold_v39_recovery_cloud_max=8000,
        micro_unfold_v39_recovery_preselect_max=320,
        micro_unfold_v39_recovery_hold_max_candidates=24,
        micro_unfold_v39_recovery_pair_max_trials=36,
        # v41: failure-only PULL reach recovery. Search geometry only; hard
        # 420mm reach, board margin, garment-exit and existing path/pair safety remain unchanged.
        micro_unfold_v41_pull_direction_recovery_enable=True,
        micro_unfold_v41_pull_direction_fan_deg=(-12.0,-8.0,-4.0,4.0,8.0,12.0),
        micro_unfold_v41_pull_direction_max_trials=48,
        micro_unfold_v41_fast_target_local_enable=True,
        micro_unfold_v41_fast_target_local_max_candidates=2,
        micro_unfold_v41_fast_target_local_anchor_trials=1,
        # v42: FULL E156 code is retained for debug/manual comparison, but AUTO
        # fast-path exhaustion returns BLOCKED immediately by default.
        micro_unfold_v42_full_e156_fallback_enable=False,
        micro_unfold_v42_pull_recovery_cache_enable=True,
        micro_unfold_v42_pull_cheap_reject_enable=True,
        # v48 lightweight same-X Target/crease validation. Components are injected by
        # the main runtime from the READY-CCA result already computed for this snapshot.
        # No new heatmap/CCA/Pose/Seg inference is launched here.
        micro_unfold_v48_target_cca_validation_enable=True,
        micro_unfold_v48_cca_measurement_valid=False,
        micro_unfold_v48_cca_components=(),
        micro_unfold_v48_cca_allow_medium_fallback=True,
        micro_unfold_v48_cca_require_when_valid=True,
        micro_unfold_v48_target_cca_max_perp_mm=45.0,
        micro_unfold_v48_target_cca_min_perp_mm=20.0,
        micro_unfold_v48_target_cca_width_pad_mm=12.0,
        micro_unfold_v48_target_cca_end_margin_mm=30.0,
        micro_unfold_v48_direction_validation_enable=True,
        micro_unfold_v48_boundary_normal_max_delta_deg=35.0,
        micro_unfold_v48_fast_same_cca_enable=True,
        micro_unfold_v48_fast_same_cca_anchor_mm=30.0,
        micro_unfold_v48_fast_same_cca_axis_min_cos=0.75,
        # v49 main-runtime local validator injects one same-X cached local crease
        # frame.  READY-CCA remains soft diagnostics only; this context is the only
        # direction frame allowed to behave as Target validation authority.
        micro_unfold_v49_local_fold_context=None,
    )


def _merged_micro_args(overrides: Optional[Dict[str, Any]] = None):
    ns = _default_micro_args()
    for k, v in dict(overrides or {}).items():
        setattr(ns, str(k), v)
    return ns


def make_runtime_cfg(config: Dict[str, Any], *, contact_z: Dict[str, float], standby_xyz: Dict[str, Any],
                     hover_z: float = 180.0, min_safe_contact_z: float = -140.0):
    pts = np.asarray(list(config.get("aruco", {}).get("marker_board_mm", {}).values()), dtype=np.float64)
    if pts.ndim != 2 or pts.shape[0] < 4 or pts.shape[1] < 2:
        raise RuntimeError("E156 VLA core requires four board-marker coordinates")
    return SimpleNamespace(
        board_x_min=float(np.min(pts[:, 0])), board_x_max=float(np.max(pts[:, 0])),
        board_y_min=float(np.min(pts[:, 1])), board_y_max=float(np.max(pts[:, 1])),
        contact_z={str(k): float(v) for k, v in dict(contact_z).items()},
        standby_xyz={str(k): [float(x) for x in v[:3]] for k, v in dict(standby_xyz).items()},
        hover_z=float(hover_z), min_safe_contact_z=float(min_safe_contact_z), standby_t=3.14,
    )


def build_pose_state_from_vla(pose_plan: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    raw = dict((pose_plan or {}).get("pose_keypoints_px", {}) or {}) if isinstance(pose_plan, dict) else {}
    named, raw_named, points = {}, {}, []
    for name in _E102_TSHIRT_POSE_KPT_NAMES:
        item = raw.get(name)
        if not isinstance(item, (list, tuple)) or len(item) < 2:
            continue
        try:
            x, y = float(item[0]), float(item[1])
            c = float(item[2]) if len(item) >= 3 else 1.0
        except Exception:
            continue
        if not (np.isfinite(x) and np.isfinite(y) and np.isfinite(c)):
            continue
        q = {"name": str(name), "x": x, "y": y, "conf": c}
        raw_named[str(name)] = dict(q)
        named[str(name)] = dict(q)
        points.append(dict(q))
    return {
        "valid_count": int(len(named)), "named_points": named, "raw_named_points": raw_named,
        "points": points, "status": "VLA_CACHED_POSE" if named else "VLA_POSE_UNAVAILABLE",
        "source": str((pose_plan or {}).get("pose_inference_source", "VLA_CACHE")) if isinstance(pose_plan, dict) else "NONE",
    }


def quick_gate(frame_bgr, mask_u8, pose_plan=None, *, structural_hint: bool = False,
               pose_deform_min_conf: float = 0.20, arg_overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Cheap high-recall gate that avoids heavy RGB planning on weak pose noise.

    v4 specifically blocks the real-log failure mode: one NECK intrusion plus a
    contour-derived ~0.33 deformation score is not enough by itself. Strong intrinsic
    chain/torso deformation, multiple intrusions, a strong non-neck intrusion, or an
    already-strong structural hint can still open the E156 planner.
    """
    frame = np.asarray(frame_bgr, dtype=np.uint8)
    mask = ((np.asarray(mask_u8, dtype=np.uint8) > 0).astype(np.uint8) * 255)
    if frame.ndim != 3 or mask.ndim != 2 or frame.shape[:2] != mask.shape[:2] or cv2.countNonZero(mask) < 500:
        return {"open": False, "reason": "BAD_FRAME_OR_MASK", "pose_state": build_pose_state_from_vla(pose_plan)}
    args_micro = _merged_micro_args(arg_overrides)
    pose_state = build_pose_state_from_vla(pose_plan)
    intrusion = _e117_pose_fold_intrusion_state(frame.shape, mask, pose_state, args=args_micro)
    deform = _e120_pose_deformation_state(frame.shape, mask, pose_state, args=args_micro, intrusion_state=intrusion)

    items = [q for q in list(intrusion.get("intrusions", []) or []) if isinstance(q, dict)]
    intrusion_count = int(len(items))
    intrusion_conf = float(intrusion.get("confidence", 0.0) or 0.0)
    non_neck = [q for q in items if str(q.get("group", "")) != "NECK"]
    deform_conf = float(deform.get("confidence", 0.0) or 0.0)
    deform_mode = str(deform.get("mode", "NONE") or "NONE")
    deform_available = bool(deform.get("available", False) and deform_conf >= float(pose_deform_min_conf))

    deform_strong_thr = float(np.clip(getattr(args_micro, "vla_gate_deform_strong_conf", 0.36), 0.0, 1.0))
    single_intr_thr = float(np.clip(getattr(args_micro, "vla_gate_single_intrusion_min_conf", 0.42), 0.0, 1.0))
    single_neck_deform_thr = float(np.clip(getattr(args_micro, "vla_gate_single_neck_deform_min_conf", 0.45), 0.0, 1.0))

    multi_intrusion = bool(intrusion_count >= 2 and intrusion_conf >= 0.20)
    strong_single_non_neck = bool(intrusion_count == 1 and len(non_neck) == 1 and intrusion_conf >= single_intr_thr)
    intrinsic_deform = bool(
        deform_available and deform_conf >= deform_strong_thr
        and deform_mode in {"TORSO_QUAD", "LEFT_CHAIN", "RIGHT_CHAIN"}
    )
    # A single neckline point is accepted only with substantially stronger support;
    # this is the exact false-positive seen in cycle_20260909_062136_921.
    allow_single_neck = bool(getattr(args_micro, "vla_gate_allow_single_neck", False))
    strict_joint = bool(getattr(args_micro, "micro_unfold_vla_require_rgb_pose_corroboration", False))
    single_neck_supported = bool(
        allow_single_neck and intrusion_count == 1 and len(non_neck) == 0
        and deform_available and deform_conf >= single_neck_deform_thr
    )
    structural_ok = bool(structural_hint)
    non_neck_multi = bool(multi_intrusion and len(non_neck) >= 1)
    if strict_joint:
        # V6: a pure neckline signal or structural-only hint is not a localization
        # hypothesis.  Heavy RGB work starts only when Pose can actually propose
        # a non-neck region, or strong intrinsic TORSO/CHAIN deformation can.
        opened = bool(non_neck_multi or strong_single_non_neck or intrinsic_deform or single_neck_supported)
    else:
        opened = bool(multi_intrusion or strong_single_non_neck or intrinsic_deform or single_neck_supported or structural_ok)

    reasons = []
    if non_neck_multi if strict_joint else multi_intrusion:
        reasons.append(f"MULTI_INTRUSION:{intrusion_count}:{intrusion_conf:.3f}")
    if strong_single_non_neck: reasons.append(f"STRONG_NON_NECK_INTRUSION:{intrusion_conf:.3f}")
    if intrinsic_deform: reasons.append(f"INTRINSIC_DEFORM:{deform_mode}:{deform_conf:.3f}")
    if single_neck_supported: reasons.append(f"SUPPORTED_SINGLE_NECK:{deform_conf:.3f}")
    if structural_ok and not strict_joint: reasons.append("STRONG_EXISTING_STRUCTURAL_HINT")
    if strict_joint and intrusion_count > 0 and len(non_neck) == 0 and not intrinsic_deform:
        reject_reason = f"STRICT_PURE_NECK_REJECT:intr={intrusion_conf:.3f},deform={deform_mode}:{deform_conf:.3f}"
    elif strict_joint and structural_ok and not (non_neck_multi or strong_single_non_neck or intrinsic_deform or single_neck_supported):
        reject_reason = "STRICT_STRUCTURAL_ONLY_REJECT"
    elif not opened and intrusion_count == 1 and len(non_neck) == 0:
        reject_reason = f"WEAK_SINGLE_NECK:intr={intrusion_conf:.3f},deform={deform_mode}:{deform_conf:.3f}"
    elif not opened and deform_available and deform_mode == "CONTOUR_INTRUSION":
        reject_reason = f"WEAK_CONTOUR_ONLY_DEFORM:{deform_conf:.3f}"
    else:
        reject_reason = "NO_MICRO_HINT"

    return {
        "open": opened, "reason": "+".join(reasons) if reasons else reject_reason,
        "pose_state": pose_state,
        "gate_policy": "V6_STRICT_POSE_LOCALIZATION_GATE" if strict_joint else "V4_STRONG_EVIDENCE_GATE",
        "intrusion_count": intrusion_count, "intrusion_confidence": intrusion_conf,
        "deform_mode": deform_mode, "deform_confidence": deform_conf,
        "structural_hint": structural_ok,
        "pose_intrusion": _training_json_safe({k:v for k,v in intrusion.items() if k not in ("corridor_mask","fold_region_mask")}),
        "pose_deformation": _training_json_safe({k:v for k,v in deform.items() if k != "prior_mask"}),
    }

def _plan_summary(plan: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not isinstance(plan, dict):
        return None
    keys = (
        "state", "method", "active_arm", "hold_arm", "active_px", "hold_px",
        "active_board", "hold_board", "pull_board", "active_local_xy", "hold_local_xy",
        "pull_local_xy", "pull_distance_mm", "pair_separation_mm", "hold_action_z",
        "active_start_z", "active_floor_z", "hold_gripper", "active_closed_ref_nominal",
        "peel_local_xy", "peel_xy_mm", "peel_z_mm", "execution_evidence_ok",
        "execution_evidence_reason", "e119_pose_deform_mode", "e119_pose_deform_confidence",
        "e120_pose_fold_region_hint", "e125_primary_fold_source", "e116_crease_source",
        "active_crease_distance_mm", "hold_crease_distance_mm",
        "vla_joint_localization_policy", "vla_joint_pose_source", "vla_joint_pose_group",
        "vla_joint_pose_confidence", "vla_joint_branch_ids", "vla_joint_selected_branch_id",
        "vla_joint_branch_overlap", "vla_joint_orientation_alignment", "vla_joint_branch_fold_score",
        "vla_joint_role_validation", "vla_rgb_rescue_used", "vla_rgb_rescue_mode",
        "vla_rgb_rescue_confidence", "vla_rgb_rescue_exact_quality",
        "vla_rgb_rescue_branch_score", "vla_rgb_rescue_neck_distance_ratio",
        "vla_rgb_rescue_neck_parallel_alignment",
        "v12_plan_score", "v12_candidate_rank", "v12_candidate_count_total",
        "v12_candidate_count_ready", "v12_crease_span_relative",
        "v12_crease_center_support", "v12_active_interior_support",
        "v12_selected_crease_line_board", "v12_selection_policy",
        "v30_target_guided_planner", "v30_target_joint_distance_mm",
        "v30_target_seed_distance_mm", "v30_pair_cross_valid",
        "v30_target_anchor_fraction",
        "v38_planner_source", "v38_target_direct_attempted", "v38_target_direct_used",
        "v38_target_board", "v38_active_inset_mm", "v38_active_boundary_distance_mm",
        "v38_active_search_candidate_count", "v38_hold_candidate_count",
        "v38_pair_candidate_count", "v38_pair_separation_mm",
        "v38_path_clearance_mm", "v38_max_reach_radius_mm",
        "v38_fallback_used", "v38_fallback_reason",
        "v39_reach_hold_recovery_used", "v39_recovery_trigger",
        "v39_recovery_raw_candidate_count", "v39_recovery_feasible_candidate_count",
        "v39_recovery_hold_candidate_count", "v39_recovery_pair_candidate_count",
        "v39_recovery_selected_assignment_preview",
        "v40_pull_recovery_used", "v40_recovery_trigger",
        "v40_recovery_active_shortlist_count", "v40_recovery_hold_candidate_count",
        "v40_recovery_pair_candidate_count", "v40_recovery_exact_trial_count",
        "v41_pull_direction_recovery_used", "v41_pull_direction_delta_deg",
        "v41_pull_direction_trial_count", "v41_pull_direction_backoff_mm",
        "v41_fast_target_local_used", "v41_fast_target_local_source",
        "v41_fast_target_local_distance_mm", "v41_fast_target_local_candidate_rank",
        "v42_pair_duplicate_skips", "v42_pull_recovery_cache_hits",
        "v42_pull_recovery_cheap_rejects", "v42_shared_geometry_cache",
        "v51_v44_full_planner_geometry", "v51_v44_planner_scope", "v51_target_authority",
        "v48_target_validation_used", "v48_target_validation_pass",
        "v48_target_validation_reason", "v48_target_validation_component_id",
        "v48_target_validation_component_rule", "v48_target_validation_perp_mm",
        "v48_target_validation_axial_overrun_mm", "v48_target_validation_segment_distance_mm",
        "v48_target_validation_ms", "v48_direction_validation_used",
        "v48_direction_validation_pass", "v48_direction_validation_reason",
        "v48_crease_anchor_board", "v48_crease_tangent_board", "v48_crease_normal_board",
        "v48_same_fold_pair_validation", "v48_same_fold_pair_validation_ms", "v48_direction_validation_ms",
        "v40_selected_pull_radius_mm", "v40_failure_reason_counts",
    )
    return {k: _training_json_safe(plan.get(k)) for k in keys if k in plan}


def binding_id(plan: Optional[Dict[str, Any]], generation: Any = None) -> Optional[str]:
    summary = _plan_summary(plan)
    if summary is None:
        return None
    payload = {"core": E156_CORE_VERSION, "generation": generation, "plan": summary}
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:24]


def plan_summary(plan: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    return copy.deepcopy(_plan_summary(plan))


# v4: ONLY these small execution/provenance fields are allowed to escape the core
# into pending_auto_x_action. Full-resolution masks/maps stay local to planning.
_FROZEN_EXECUTION_KEYS = (
    "state", "method", "active_arm", "hold_arm", "active_px", "hold_px",
    "active_board", "hold_board", "pull_board", "active_local_xy", "hold_local_xy",
    "pull_local_xy", "pull_distance_mm", "pair_separation_mm", "hold_action_z",
    "active_start_z", "active_floor_z", "hold_gripper", "active_closed_ref_nominal",
    "peel_local_xy", "peel_xy_mm", "peel_z_mm", "execution_evidence_ok",
    "execution_evidence_reason", "e119_pose_deform_mode", "e119_pose_deform_confidence",
    "e120_pose_fold_region_hint", "e125_primary_fold_source", "e116_crease_source",
    "active_crease_distance_mm", "hold_crease_distance_mm",
    "vla_joint_localization_policy", "vla_joint_pose_source", "vla_joint_pose_group",
    "vla_joint_pose_confidence", "vla_joint_branch_ids", "vla_joint_selected_branch_id",
    "vla_joint_branch_overlap", "vla_joint_orientation_alignment", "vla_joint_branch_fold_score",
    "vla_joint_role_validation", "vla_rgb_rescue_used", "vla_rgb_rescue_mode",
    "vla_rgb_rescue_confidence", "vla_rgb_rescue_exact_quality",
    "vla_rgb_rescue_branch_score", "vla_rgb_rescue_neck_distance_ratio",
    "vla_rgb_rescue_neck_parallel_alignment",
    "v12_plan_score", "v12_candidate_rank", "v12_candidate_count_total",
    "v12_candidate_count_ready", "v12_crease_span_relative",
    "v12_crease_center_support", "v12_active_interior_support",
    "v12_selected_crease_line_board", "v12_selection_policy",
    "v13_selected_crease_line_px", "v13_fold_candidate_summaries",
    "v13_same_crease_anchor_search_used", "v13_same_crease_anchor_fraction",
    "v13_same_crease_anchor_count", "v30_target_guided_planner",
    "v30_target_joint_distance_mm", "v30_target_seed_distance_mm",
    "v30_pair_cross_valid", "v30_target_anchor_fraction",
    "v38_planner_source", "v38_target_direct_attempted", "v38_target_direct_used",
    "v38_target_board", "v38_active_inset_mm", "v38_active_boundary_distance_mm",
    "v38_active_search_candidate_count", "v38_hold_candidate_count",
    "v38_pair_candidate_count", "v38_pair_separation_mm",
    "v38_path_clearance_mm", "v38_max_reach_radius_mm",
    "v38_fallback_used", "v38_fallback_reason",
    "v39_reach_hold_recovery_used", "v39_recovery_trigger",
    "v39_recovery_raw_candidate_count", "v39_recovery_feasible_candidate_count",
    "v39_recovery_hold_candidate_count", "v39_recovery_pair_candidate_count",
    "v39_recovery_selected_assignment_preview",
    "v40_pull_recovery_used", "v40_recovery_trigger",
    "v40_recovery_active_shortlist_count", "v40_recovery_hold_candidate_count",
    "v40_recovery_pair_candidate_count", "v40_recovery_exact_trial_count",
    "v41_pull_direction_recovery_used", "v41_pull_direction_delta_deg",
    "v41_pull_direction_trial_count", "v41_pull_direction_backoff_mm",
    "v41_fast_target_local_used", "v41_fast_target_local_source",
    "v41_fast_target_local_distance_mm", "v41_fast_target_local_candidate_rank",
    "v42_pair_duplicate_skips", "v42_pull_recovery_cache_hits",
    "v42_pull_recovery_cheap_rejects", "v42_shared_geometry_cache",
    "v51_v44_full_planner_geometry", "v51_v44_planner_scope", "v51_target_authority",
    "v48_target_validation_used", "v48_target_validation_pass",
    "v48_target_validation_reason", "v48_target_validation_component_id",
    "v48_target_validation_component_rule", "v48_target_validation_perp_mm",
    "v48_target_validation_axial_overrun_mm", "v48_target_validation_segment_distance_mm",
    "v48_target_validation_ms", "v48_direction_validation_used",
    "v48_direction_validation_pass", "v48_direction_validation_reason",
    "v48_crease_anchor_board", "v48_crease_tangent_board", "v48_crease_normal_board",
    "v48_same_fold_pair_validation", "v48_same_fold_pair_validation_ms", "v48_direction_validation_ms",
    "v40_selected_pull_radius_mm", "v40_failure_reason_counts",
    "e156_vla_core_version",
)


def freeze_execution_plan(plan: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not isinstance(plan, dict):
        return None
    out = {k: _training_json_safe(plan.get(k)) for k in _FROZEN_EXECUTION_KEYS if k in plan}
    # Fail closed: a frozen object must carry the exact minimum executor contract.
    required = (
        "state", "active_arm", "hold_arm", "active_local_xy", "hold_local_xy",
        "pull_local_xy", "hold_action_z", "active_start_z", "hold_gripper",
    )
    if any(k not in out for k in required):
        return None
    return out



def _v6_unit_xy(v):
    try:
        a=np.asarray(v,dtype=np.float64).reshape(2)
        n=float(np.linalg.norm(a))
        if not np.all(np.isfinite(a)) or n<=1e-6:
            return None
        return a/n
    except Exception:
        return None



def _v30_target_board(args):
    if not bool(getattr(args, 'micro_unfold_target_hint_enable', False)):
        return None
    try:
        q=np.asarray(getattr(args, 'micro_unfold_target_board_mm', None),dtype=np.float64).reshape(2)
        return q if np.all(np.isfinite(q)) else None
    except Exception:
        return None


def _v30_mask_target_distance_mm(mask_u8, H_value, target_board, max_samples=160):
    try:
        m=((np.asarray(mask_u8,dtype=np.uint8)>0).astype(np.uint8)*255)
        yy,xx=np.where(m>0)
        if xx.size<=0:
            return float('inf')
        if xx.size>int(max_samples):
            idx=np.linspace(0,xx.size-1,int(max_samples),dtype=np.int64)
            pts=np.column_stack([xx[idx],yy[idx]]).astype(np.float32)
        else:
            pts=np.column_stack([xx,yy]).astype(np.float32)
        b=_fix12_transform_points(H_value,pts)
        t=np.asarray(target_board,dtype=np.float64).reshape(2)
        return float(np.min(np.linalg.norm(b-t[None,:],axis=1)))
    except Exception:
        return float('inf')


def _v30_segment_crosses_crease(active_px, hold_px, crease_u8):
    try:
        crease=(np.asarray(crease_u8,dtype=np.uint8)>0)
        if crease.ndim!=2 or not np.any(crease):
            return False
        a=np.asarray(active_px,dtype=np.float64).reshape(2)
        hpt=np.asarray(hold_px,dtype=np.float64).reshape(2)
        v=a-hpt
        dist=float(np.linalg.norm(v))
        if dist<=1e-6:
            return False
        n=int(np.clip(round(dist*1.4),25,220))
        t=np.linspace(0.0,1.0,n,dtype=np.float64)
        pts=hpt[None,:]+t[:,None]*v[None,:]
        hh,ww=crease.shape
        xi=np.clip(np.rint(pts[:,0]).astype(np.int32),0,ww-1)
        yi=np.clip(np.rint(pts[:,1]).astype(np.int32),0,hh-1)
        near=cv2.dilate((crease.astype(np.uint8)*255),cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(13,13)),iterations=1)>0
        return bool(np.any(near[yi,xi]))
    except Exception:
        return False


def _v31_prepare_crease_near(crease_u8):
    """Build the 13x13 near-crease mask once per selected crease.

    v30 rebuilt this full-frame dilation for every ACTIVE x HOLD pair.  v31 keeps
    the exact same 13x13 morphology and crossing semantics, but reuses the result.
    """
    try:
        crease=(np.asarray(crease_u8,dtype=np.uint8)>0)
        if crease.ndim!=2 or not np.any(crease):
            return None
        kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(13,13))
        return cv2.dilate((crease.astype(np.uint8)*255),kernel,iterations=1)>0
    except Exception:
        return None


def _v31_segment_crosses_crease_prepared(active_px, hold_px, near_mask):
    """v30 crossing test using an already-dilated crease mask."""
    try:
        near=np.asarray(near_mask,dtype=bool)
        if near.ndim!=2 or not np.any(near):
            return False
        a=np.asarray(active_px,dtype=np.float64).reshape(2)
        hpt=np.asarray(hold_px,dtype=np.float64).reshape(2)
        v=a-hpt
        dist=float(np.linalg.norm(v))
        if dist<=1e-6:
            return False
        n=int(np.clip(round(dist*1.4),25,220))
        t=np.linspace(0.0,1.0,n,dtype=np.float64)
        pts=hpt[None,:]+t[:,None]*v[None,:]
        hh,ww=near.shape
        xi=np.clip(np.rint(pts[:,0]).astype(np.int32),0,ww-1)
        yi=np.clip(np.rint(pts[:,1]).astype(np.int32),0,hh-1)
        return bool(np.any(near[yi,xi]))
    except Exception:
        return False


def _v6_branch_tangent(mask_u8):
    """Return the PCA tangent of one thin RGB crease branch."""
    try:
        yy,xx=np.where(np.asarray(mask_u8,dtype=np.uint8)>0)
        if xx.size<6:
            return None
        pts=np.column_stack([xx,yy]).astype(np.float64)
        pts-=np.mean(pts,axis=0,keepdims=True)
        cov=pts.T@pts
        vals,vecs=np.linalg.eigh(cov)
        return _v6_unit_xy(vecs[:,int(np.argmax(vals))])
    except Exception:
        return None


def _v6_circle_roi(shape, center_xy, radius_px, garment_u8):
    h,w=shape[:2]
    out=np.zeros((h,w),dtype=np.uint8)
    try:
        c=np.asarray(center_xy,dtype=np.float64).reshape(2)
        if not np.all(np.isfinite(c)):
            return out
        cv2.circle(out,(int(round(float(c[0]))),int(round(float(c[1])))),max(4,int(round(float(radius_px)))),255,-1,cv2.LINE_AA)
        return cv2.bitwise_and(out,np.asarray(garment_u8,dtype=np.uint8))
    except Exception:
        return out


def _v6_intrusion_group_candidate(depth_state, garment_u8, group, items):
    """Rebuild the original E125 coarse intrusion ROI for one non-neck group."""
    h,w=np.asarray(garment_u8).shape[:2]
    roi=np.zeros((h,w),dtype=np.uint8)
    vecs=[]
    for q in list(items or []):
        if not isinstance(q,dict):
            continue
        try:
            p=np.asarray([float(q.get('x')),float(q.get('y'))],dtype=np.float64)
            b=np.asarray(q.get('nearest_contour_px'),dtype=np.float64).reshape(2)
            dist=float(q.get('inside_distance_px',0.0) or 0.0)
            if not np.all(np.isfinite(p)) or not np.all(np.isfinite(b)):
                continue
        except Exception:
            continue
        rw=int(np.clip(round(max(18.0,0.70*dist)),18,58))
        pi=tuple(np.rint(p).astype(int)); bi=tuple(np.rint(b).astype(int))
        cv2.line(roi,pi,bi,255,rw,cv2.LINE_AA)
        cv2.circle(roi,pi,max(9,rw//2),255,-1,cv2.LINE_AA)
        cv2.circle(roi,bi,max(7,rw//3),255,-1,cv2.LINE_AA)
        u=_v6_unit_xy(p-b)
        if u is not None:
            vecs.append(u)
    roi=cv2.bitwise_and(roi,np.asarray(garment_u8,dtype=np.uint8))
    if cv2.countNonZero(roi)>=16:
        roi=cv2.morphologyEx(roi,cv2.MORPH_CLOSE,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(9,9)))
        roi=cv2.bitwise_and(roi,np.asarray(garment_u8,dtype=np.uint8))
    direction=None
    if vecs:
        # Fold-boundary orientation is sign-invariant, so align vectors before averaging.
        base=np.asarray(vecs[0],dtype=np.float64)
        aligned=[]
        for u in vecs:
            uu=np.asarray(u,dtype=np.float64)
            if float(np.dot(base,uu))<0.0:
                uu=-uu
            aligned.append(uu)
        direction=_v6_unit_xy(np.mean(np.asarray(aligned,dtype=np.float64),axis=0))
    scores=dict(depth_state.get('pose_fold_group_scores',{}) or {})
    conf=float(scores.get(str(group),depth_state.get('pose_fold_intrusion_confidence',0.0)) or 0.0)
    return {'source':'POSE_INTRUSION','group':str(group),'confidence':float(conf),'roi':roi,'direction':direction,'items':list(items or [])}


def _v6_roi_overlap(a_u8,b_u8):
    a=np.asarray(a_u8,dtype=np.uint8)>0; b=np.asarray(b_u8,dtype=np.uint8)>0
    na=int(np.count_nonzero(a)); nb=int(np.count_nonzero(b))
    if na<=0 or nb<=0:
        return 0.0
    inter=int(np.count_nonzero(a&b))
    return float(inter)/float(max(1,min(na,nb)))


def _v6_filter_e122_pool(depth_state, mask_u8, args, H_value, pool, kept):
    """Replace the same-I E122 cache with only RGB branches corroborating the chosen Pose ROI."""
    mask=np.asarray(mask_u8,dtype=np.uint8)
    filtered=copy.copy(dict(pool or {}))
    branches=[q for q in list(kept or []) if isinstance(q,dict)]
    raw=np.zeros_like(mask); preview=np.zeros_like(mask); execution=np.zeros_like(mask)
    score_map=np.zeros(mask.shape,dtype=np.float32)
    for q in branches:
        bm=np.asarray(q.get('mask',np.zeros_like(mask)),dtype=np.uint8)>0
        raw[bm]=255
        fs=float(q.get('fold_score',0.0) or 0.0)
        score_map[bm]=np.maximum(score_map[bm],fs)
        if bool(q.get('preview_ok',False)):
            preview[bm]=255
        if bool(q.get('execution_ok',False)):
            execution[bm]=255
    filtered['branches']=branches
    filtered['branch_info']=[{k:_training_json_safe(v) for k,v in q.items() if k!='mask'} for q in branches[:24]]
    filtered['raw_skeleton']=raw
    filtered['preview_skeleton']=preview
    filtered['execution_skeleton']=execution
    filtered['rejected_skeleton']=np.zeros_like(mask)
    filtered['contour_filtered_raw_skeleton']=raw.copy()
    filtered['contour_veto_rejected_skeleton']=np.zeros_like(mask)
    filtered['branch_score_map']=score_map
    filtered['available']=bool(cv2.countNonZero(preview)>=8)
    filtered['raw_branch_count']=int(len(branches))
    filtered['preview_branch_count']=int(sum(bool(q.get('preview_ok',False)) for q in branches))
    filtered['execution_branch_count']=int(sum(bool(q.get('execution_ok',False)) for q in branches))
    filtered['structural_branch_count']=int(sum(str(q.get('source',''))=='STRUCTURAL' for q in branches))
    filtered['sensitive_branch_count']=int(sum(str(q.get('source',''))=='SENSITIVE' for q in branches))
    filtered['rescue_branch_count']=int(sum(bool(q.get('structural_rescue',False)) for q in branches))
    filtered['vla_joint_filtered']=True
    try:
        sig=_e153_e122_cache_signature(depth_state,mask,args,H_value)
        depth_state['_e153_e122_pool_cache']={'signature':sig,'pool':filtered}
    except Exception:
        # The cached pool already exists from this same-I call; mutating it is safe fallback.
        depth_state['_e153_e122_pool_cache']={'signature':None,'pool':filtered}
    depth_state.pop('_e153_edge_split_cache',None)
    _e153_publish_e122_pool(depth_state,filtered,mask)
    return filtered


def _v6_joint_pose_rgb_localize(depth_state, mask_u8, H_value, args):
    """V6 fail-closed location authority for VLA MICRO_UNFOLD.

    Pose/intrusion/deformation may only propose a region.  A physically executable
    E122 RGB Fold-Crease must overlap that same region and have a compatible normal.
    The selected same-fold RGB branches replace the same-I branch cache, preventing
    unrelated FOLD_NORMAL/EDGE fallbacks from escaping to another garment region.
    """
    require=bool(getattr(args,'micro_unfold_vla_require_rgb_pose_corroboration',True))
    if not require:
        return {'ok':True,'reason':'JOINT_LOCALIZATION_DISABLED','policy':'LEGACY'}
    mask=((np.asarray(mask_u8,dtype=np.uint8)>0).astype(np.uint8)*255)
    if cv2.countNonZero(mask)<500:
        return {'ok':False,'reason':'JOINT_BAD_GARMENT_MASK','policy':'V6_RGB_POSE_SAME_ROI'}

    min_exec=max(6,int(getattr(args,'micro_unfold_vla_joint_min_exec_pixels',12)))
    min_overlap=float(np.clip(getattr(args,'micro_unfold_vla_joint_min_branch_overlap',0.18),0.0,1.0))
    min_orient=float(np.clip(getattr(args,'micro_unfold_vla_joint_min_orientation_alignment',0.45),0.0,1.0))
    deform_min=float(np.clip(getattr(args,'micro_unfold_vla_joint_deform_min_conf',0.36),0.0,1.0))
    deform_region_min=float(np.clip(getattr(args,'micro_unfold_vla_joint_deform_region_min_conf',0.20),0.0,1.0))
    roi_radius=max(60.0,float(getattr(args,'micro_unfold_vla_joint_roi_radius_px',170.0)))
    dilate_px=max(0,int(getattr(args,'micro_unfold_vla_joint_branch_dilate_px',20)))
    min_span=max(8.0,float(getattr(args,'micro_unfold_vla_joint_min_branch_span_px',18.0)))

    pool=_e122_build_scored_crease_pool(depth_state,mask,args,H_value=H_value)
    exec_branches=[]
    for q in list((pool or {}).get('branches',[]) or []):
        if not isinstance(q,dict) or not bool(q.get('execution_ok',False)):
            continue
        bm=np.asarray(q.get('mask',np.zeros_like(mask)),dtype=np.uint8)
        if cv2.countNonZero(bm)<min_exec or float(q.get('span_px',0.0) or 0.0)<min_span:
            continue
        exec_branches.append(q)
    if not exec_branches:
        return {
            'ok':False,'reason':'JOINT_NO_E122_EXECUTION_BRANCH','policy':'V6_RGB_POSE_SAME_ROI',
            'rgb_execution_branch_count':0,'raw_branch_count':int((pool or {}).get('raw_branch_count',0)),
        }

    candidates=[]
    deform_mode=str(depth_state.get('pose_deform_mode','NONE') or 'NONE').upper()
    deform_conf=float(depth_state.get('pose_deform_confidence',0.0) or 0.0)
    deform_group=str(depth_state.get('pose_deform_fold_region_hint','NONE') or 'NONE').upper()
    deform_region_conf=float(depth_state.get('pose_deform_fold_region_confidence',0.0) or 0.0)
    deform_anchor=depth_state.get('pose_deform_fold_anchor_px')
    deform_dir=_v6_unit_xy(depth_state.get('pose_deform_fold_direction_px'))
    strong_deform=bool(
        deform_mode in {'TORSO_QUAD','LEFT_CHAIN','RIGHT_CHAIN'} and
        deform_conf>=deform_min and deform_region_conf>=deform_region_min and
        deform_group!='NONE' and deform_anchor is not None
    )
    deform_roi=None
    if strong_deform:
        circle=_v6_circle_roi(mask.shape,deform_anchor,roi_radius,mask)
        prior=((np.asarray(depth_state.get('pose_deform_mask',np.zeros_like(mask)),dtype=np.uint8)>0).astype(np.uint8)*255)
        inter=cv2.bitwise_and(circle,prior) if cv2.countNonZero(prior)>=16 else np.zeros_like(mask)
        deform_roi=inter if cv2.countNonZero(inter)>=24 else circle
        deform_roi=cv2.bitwise_and(deform_roi,mask)
        if cv2.countNonZero(deform_roi)>=24:
            candidates.append({
                'source':'POSE_DEFORMATION','group':deform_group,
                'confidence':float(max(deform_conf,deform_region_conf)),
                'roi':deform_roi,'direction':deform_dir,'items':[],
            })

    # Raw contour intrusion may propose a region only when it is not pure NECK.
    # Under strong TORSO/CHAIN deformation, an intrusion region must agree spatially
    # with that stronger deformation ROI; otherwise it cannot steal location authority.
    intr=[q for q in list(depth_state.get('pose_fold_intrusions',[]) or []) if isinstance(q,dict)]
    groups={}
    for q in intr:
        g=str(q.get('group','OTHER') or 'OTHER').upper()
        if g=='NECK':
            continue
        groups.setdefault(g,[]).append(q)
    for g,items in groups.items():
        cand=_v6_intrusion_group_candidate(depth_state,mask,g,items)
        roi=np.asarray(cand.get('roi',np.zeros_like(mask)),dtype=np.uint8)
        if cv2.countNonZero(roi)<24:
            continue
        if strong_deform and deform_roi is not None:
            agree=_v6_roi_overlap(roi,deform_roi)
            if agree<0.08:
                print(f"[E156-V6-JOINT-INTRUSION-SUPPRESS] group={g} overlapWithStrongDeform={agree:.3f}<0.080")
                continue
            cand['strong_deform_overlap']=float(agree)
        candidates.append(cand)

    if not candidates:
        return {
            'ok':False,'reason':'JOINT_NO_NON_NECK_POSE_REGION','policy':'V6_RGB_POSE_SAME_ROI',
            'rgb_execution_branch_count':int(len(exec_branches)),'strong_deform':bool(strong_deform),
            'deform_mode':deform_mode,'deform_group':deform_group,'deform_confidence':float(deform_conf),
        }

    matches=[]
    k=None
    if dilate_px>0:
        k=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2*dilate_px+1,2*dilate_px+1))
    for ci,cand in enumerate(candidates):
        roi=np.asarray(cand['roi'],dtype=np.uint8)
        roi_near=cv2.dilate(roi,k,iterations=1) if k is not None else roi
        direction=_v6_unit_xy(cand.get('direction'))
        pose_conf=float(cand.get('confidence',0.0) or 0.0)
        for q in exec_branches:
            bm=np.asarray(q.get('mask',np.zeros_like(mask)),dtype=np.uint8)>0
            n=int(np.count_nonzero(bm))
            if n<=0:
                continue
            overlap=float(np.count_nonzero(bm & (roi_near>0)))/float(n)
            tangent=_v6_branch_tangent(bm.astype(np.uint8)*255)
            if direction is None or tangent is None:
                orient=1.0
            else:
                tangent_dot=float(np.clip(abs(float(np.dot(tangent,direction))),0.0,1.0))
                orient=float(math.sqrt(max(0.0,1.0-tangent_dot*tangent_dot)))
            if overlap+1e-9<min_overlap or orient+1e-9<min_orient:
                continue
            _v30_target=_v30_target_board(args)
            _v30_target_dist=(_v30_mask_target_distance_mm(bm.astype(np.uint8)*255,H_value,_v30_target)
                              if _v30_target is not None else float('inf'))
            fold_score=float(np.clip(q.get('fold_score',0.0) or 0.0,0.0,1.0))
            rgb_support=float(np.clip(q.get('rgb_execution_support',q.get('raw_crease_strength',0.0)) or 0.0,0.0,1.0))
            score=float(0.34*fold_score+0.28*overlap+0.18*orient+0.12*np.clip(pose_conf,0.0,1.0)+0.08*rgb_support)
            if str(cand.get('source'))=='POSE_DEFORMATION':
                score+=0.035
            matches.append({
                'score':float(score),'candidate_index':int(ci),'branch':q,
                'branch_id':int(q.get('branch_id',-1)),'branch_overlap':float(overlap),
                'orientation_alignment':float(orient),'fold_score':float(fold_score),
                'rgb_execution_support':float(rgb_support),
                'v30_target_distance_mm':None if not np.isfinite(_v30_target_dist) else float(_v30_target_dist),
            })
    if not matches:
        return {
            'ok':False,'reason':'JOINT_NO_SAME_ROI_RGB_EXECUTION_CREASE','policy':'V6_RGB_POSE_SAME_ROI',
            'pose_candidate_count':int(len(candidates)),'rgb_execution_branch_count':int(len(exec_branches)),
            'strong_deform':bool(strong_deform),
        }

    _v30_target=_v30_target_board(args)
    _v30_target_max=max(20.0,float(getattr(args,'micro_unfold_target_match_max_mm',120.0)))
    _v30_in_window=[m for m in matches if m.get('v30_target_distance_mm') is not None and float(m['v30_target_distance_mm'])<=_v30_target_max]
    if _v30_target is not None and _v30_in_window:
        matches.sort(key=lambda z:(0 if z.get('v30_target_distance_mm') is not None and float(z['v30_target_distance_mm'])<=_v30_target_max else 1,
                                   float(z.get('v30_target_distance_mm',1e9) if z.get('v30_target_distance_mm') is not None else 1e9),
                                   -float(z['score'])))
    else:
        matches.sort(key=lambda z:float(z['score']),reverse=True)
    best=matches[0]
    selected=candidates[int(best['candidate_index'])]
    selected_roi=np.asarray(selected['roi'],dtype=np.uint8)
    # Retain up to three mechanically interchangeable RGB boundaries, but ONLY
    # from the chosen Pose ROI and only when each independently passes the joint gate.
    same=[m for m in matches if int(m['candidate_index'])==int(best['candidate_index'])]
    if _v30_target is not None and any(m.get('v30_target_distance_mm') is not None and float(m['v30_target_distance_mm'])<=_v30_target_max for m in same):
        same.sort(key=lambda z:(0 if z.get('v30_target_distance_mm') is not None and float(z['v30_target_distance_mm'])<=_v30_target_max else 1,
                                float(z.get('v30_target_distance_mm',1e9) if z.get('v30_target_distance_mm') is not None else 1e9),
                                -float(z['score'])))
    else:
        same.sort(key=lambda z:float(z['score']),reverse=True)
    kept=[]; kept_ids=[]
    for m in same[:3]:
        q=m['branch']
        bid=int(q.get('branch_id',-1))
        if bid in kept_ids:
            continue
        kept.append(q); kept_ids.append(bid)
    filtered=_v6_filter_e122_pool(depth_state,mask,args,H_value,pool,kept)
    if cv2.countNonZero(np.asarray(filtered.get('execution_skeleton',np.zeros_like(mask)),dtype=np.uint8))<min_exec:
        return {'ok':False,'reason':'JOINT_FILTERED_RGB_CREASE_EMPTY','policy':'V6_RGB_POSE_SAME_ROI'}

    # Publish only the selected spatial hypothesis.  Raw NECK/other-region Pose
    # intrusions remain diagnostic in the returned gate, but cannot steer motion.
    selected_group=str(selected.get('group','NONE'))
    selected_source=str(selected.get('source','UNKNOWN'))
    if selected_source=='POSE_INTRUSION':
        depth_state['pose_fold_intrusions']=[q for q in list(selected.get('items',[]) or []) if str(q.get('group','')).upper()==selected_group]
    else:
        depth_state['pose_fold_intrusions']=[]
        depth_state['pose_fold_intrusion_available']=False
    depth_state['vla_joint_fold_region_valid']=True
    depth_state['vla_joint_fold_region_mask']=selected_roi.copy()
    depth_state['vla_joint_pose_source']=selected_source
    depth_state['vla_joint_pose_group']=selected_group
    depth_state['vla_joint_pose_confidence']=float(selected.get('confidence',0.0) or 0.0)
    depth_state['vla_joint_branch_ids']=list(kept_ids)
    depth_state['vla_joint_selected_branch_id']=int(best['branch_id'])
    depth_state['vla_joint_selected_branch_overlap']=float(best['branch_overlap'])
    depth_state['vla_joint_selected_orientation_alignment']=float(best['orientation_alignment'])
    depth_state['vla_joint_selected_branch_score']=float(best['fold_score'])
    depth_state['vla_joint_selected_score']=float(best['score'])
    depth_state['v30_target_guided_planner']=bool(_v30_target is not None)
    depth_state['v30_target_joint_distance_mm']=best.get('v30_target_distance_mm')
    depth_state['vla_joint_strict_active']=True
    depth_state['vla_joint_selected_rgb_skeleton']=np.asarray(filtered.get('execution_skeleton',np.zeros_like(mask)),dtype=np.uint8)
    print(
        f"[E156-V6-JOINT-READY] pose={selected_source}:{selected_group} "
        f"poseConf={float(selected.get('confidence',0.0)):.3f} branch={int(best['branch_id'])} "
        f"foldScore={float(best['fold_score']):.3f} overlap={float(best['branch_overlap']):.3f} "
        f"normalAlign={float(best['orientation_alignment']):.3f} kept={kept_ids}"
        + (f" targetDist={float(best['v30_target_distance_mm']):.1f}mm" if best.get('v30_target_distance_mm') is not None else '')
    )
    return {
        'ok':True,'reason':'JOINT_RGB_POSE_CORROBORATED','policy':'V6_RGB_POSE_SAME_ROI',
        'pose_source':selected_source,'pose_group':selected_group,
        'pose_confidence':float(selected.get('confidence',0.0) or 0.0),
        'selected_branch_id':int(best['branch_id']),'kept_branch_ids':list(kept_ids),
        'branch_overlap':float(best['branch_overlap']),
        'orientation_alignment':float(best['orientation_alignment']),
        'branch_fold_score':float(best['fold_score']),
        'joint_score':float(best['score']),
        'v30_target_guided_planner':bool(_v30_target is not None),
        'v30_target_joint_distance_mm':best.get('v30_target_distance_mm'),
        'pose_candidate_count':int(len(candidates)),
        'rgb_execution_branch_count':int(len(exec_branches)),
        'strong_deform':bool(strong_deform),
    }



def _v7_neckline_relation(depth_state, candidate):
    """Return scale-relative neckline proximity/orientation for one RGB crease.

    Distance is divided by the detected NECK_LEFT--NECK_RIGHT span, so this veto
    scales with the observed garment instead of using a fixed pixel/mm location.
    """
    out={'available':False,'distance_ratio':999.0,'parallel_alignment':0.0}
    try:
        pose=dict(depth_state.get('tshirt_pose_state',{}) or {})
        named=dict(pose.get('named_points',{}) or {})
        nl=dict(named.get('neck_left',{}) or {}); nr=dict(named.get('neck_right',{}) or {})
        p1=np.asarray([float(nl.get('x')),float(nl.get('y'))],dtype=np.float64)
        p2=np.asarray([float(nr.get('x')),float(nr.get('y'))],dtype=np.float64)
        v=p2-p1; span=float(np.linalg.norm(v))
        if not np.all(np.isfinite(p1)) or not np.all(np.isfinite(p2)) or span<=8.0:
            return out
        u=v/span
        anchor=np.asarray(candidate.get('anchor_px'),dtype=np.float64).reshape(2)
        if not np.all(np.isfinite(anchor)):
            return out
        along=float(np.clip(np.dot(anchor-p1,u),0.0,span))
        proj=p1+along*u
        dist=float(np.linalg.norm(anchor-proj))
        tangent=_v6_unit_xy(candidate.get('tangent_px'))
        parallel=float(abs(np.dot(tangent,u))) if tangent is not None else 0.0
        out.update({'available':True,'distance_ratio':float(dist/max(span,1e-6)),
                    'parallel_alignment':float(np.clip(parallel,0.0,1.0)),
                    'distance_px':float(dist),'neck_span_px':float(span)})
    except Exception:
        pass
    return out


def _v7_candidate_branch_matches(candidate_mask, branches, *, dilate_px=8):
    """Map an exact/raw crease component back to same-I E122 execution branches."""
    cm=((np.asarray(candidate_mask,dtype=np.uint8)>0).astype(np.uint8)*255)
    if cv2.countNonZero(cm)<4:
        return []
    k=max(0,int(dilate_px))
    near=cv2.dilate(cm,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2*k+1,2*k+1)),iterations=1)>0 if k>0 else (cm>0)
    direct=cm>0
    out=[]
    for q in list(branches or []):
        if not isinstance(q,dict):
            continue
        bm=np.asarray(q.get('mask',np.zeros_like(cm)),dtype=np.uint8)>0
        n=int(np.count_nonzero(bm))
        if n<=0:
            continue
        near_frac=float(np.count_nonzero(bm & near))/float(n)
        direct_frac=float(np.count_nonzero(bm & direct))/float(n)
        if near_frac<0.30 and direct_frac<0.08:
            continue
        score=float(0.70*near_frac+0.20*direct_frac+0.10*np.clip(q.get('fold_score',0.0) or 0.0,0.0,1.0))
        out.append({'score':score,'near_fraction':near_frac,'direct_fraction':direct_frac,'branch':q})
    out.sort(key=lambda z:float(z.get('score',0.0)),reverse=True)
    return out



def _v11_pose_local_score_crease_candidates(depth_state, mask_u8, args):
    """Recover a real low-contrast fold near a strong Pose-deformation anchor.

    This does NOT infer whether the whole garment is spread/finished and it does
    not threshold absolute RGB brightness.  It reuses the garment-relative
    ``fold_crease_score`` already computed for this snapshot, takes a local
    percentile around the Pose deformation anchor, then requires bilateral RGB,
    seam, geometry and Pose-normal agreement before publishing a real crease.
    """
    mask=((np.asarray(mask_u8,dtype=np.uint8)>0).astype(np.uint8)*255)
    score=np.asarray(depth_state.get('fold_crease_score',np.zeros(mask.shape,dtype=np.float32)),dtype=np.float32)
    frame=depth_state.get('frame')
    pose_conf=float(depth_state.get('pose_deform_confidence',0.0) or 0.0)
    region_conf=float(depth_state.get('pose_deform_fold_region_confidence',0.0) or 0.0)
    min_pose=float(np.clip(getattr(args,'micro_unfold_vla_joint_deform_region_min_conf',0.20),0.0,1.0))
    if frame is None or score.shape!=mask.shape or not bool(depth_state.get('pose_deform_available',False)) or max(pose_conf,region_conf)<min_pose:
        return []
    try:
        anchor=np.asarray(depth_state.get('pose_deform_fold_anchor_px'),dtype=np.float64).reshape(2)
        direction=_v6_unit_xy(depth_state.get('pose_deform_fold_direction_px'))
    except Exception:
        return []
    if direction is None or not np.all(np.isfinite(anchor)):
        return []
    ys,xs=np.where(mask>0)
    if xs.size<500:
        return []
    bw=float(xs.max()-xs.min()+1); bh=float(ys.max()-ys.min()+1)
    # Radius scales with the observed garment and is capped by the existing joint ROI.
    radius=min(float(getattr(args,'micro_unfold_vla_joint_roi_radius_px',170.0)),max(60.0,0.30*min(bw,bh)))
    hh,ww=mask.shape
    yy,xx=np.ogrid[:hh,:ww]
    roi=(((xx-anchor[0])**2+(yy-anchor[1])**2)<=radius*radius) & (mask>0) & np.isfinite(score) & (score>0)
    vals=score[roi]
    if vals.size<128:
        return []
    # Adaptive to the current garment/local texture: no fixed brightness or gradient value.
    pct=float(np.clip(getattr(args,'micro_unfold_vla_pose_local_score_percentile',85.0),70.0,95.0))
    threshold=float(np.percentile(vals,pct))
    bw_score=((score>=threshold)&roi).astype(np.uint8)*255
    bw_score=cv2.morphologyEx(bw_score,cv2.MORPH_CLOSE,np.ones((3,3),np.uint8),iterations=1)
    sk=_e94_skeletonize(bw_score)
    min_pixels=max(6,int(getattr(args,'micro_unfold_e122_branch_min_px',8)))
    recs=_e122_split_crease_branches(sk,min_pixels=min_pixels,return_meta=True) if cv2.countNonZero(sk)>=min_pixels else []
    if not recs:
        return []
    structural=np.asarray(depth_state.get('fold_crease_structural_skeleton',np.zeros_like(mask)),dtype=np.uint8)
    deform=np.asarray(depth_state.get('pose_deform_mask',np.zeros_like(mask)),dtype=np.uint8)
    seam=np.asarray(depth_state.get('pose_seam_exclusion_mask',np.zeros_like(mask)),dtype=np.uint8)
    min_span=max(8.0,float(getattr(args,'micro_unfold_vla_joint_min_branch_span_px',18.0)))
    min_align=float(np.clip(getattr(args,'micro_unfold_vla_joint_min_orientation_alignment',0.45),0.0,1.0))
    rescue_thr=float(getattr(args,'micro_unfold_e123_structural_rescue_score',0.34))
    out=[]
    zero=np.zeros(mask.shape,dtype=np.float32)
    for idx,r in enumerate(recs,1):
        met=_e122_score_fold_branch(r['mask'],frame,zero,mask,score,structural,deform,pose_conf,seam,shared_ctx=None,branch_meta=r)
        span=float(met.get('span_px',0.0) or 0.0); valid=float(met.get('valid_sample_ratio',0.0) or 0.0)
        seam_overlap=float(met.get('seam_overlap',0.0) or 0.0); fold=float(met.get('fold_score',0.0) or 0.0)
        deform_support=float(met.get('pose_deform_support',0.0) or 0.0)
        ctr=np.asarray(met.get('centroid_px',[np.nan,np.nan]),dtype=np.float64).reshape(2)
        if not np.all(np.isfinite(ctr)):
            continue
        proximity=float(np.clip(1.0-np.linalg.norm(ctr-anchor)/max(1.0,radius),0.0,1.0))
        try:
            exact=_e145_select_exact_fold_normal(depth_state,mask,np.asarray(r['mask'],dtype=np.uint8),np.zeros_like(mask))
        except Exception:
            exact=None
        if not isinstance(exact,dict):
            continue
        normal=_v6_unit_xy(exact.get('normal_px'))
        align=abs(float(np.dot(normal,direction))) if normal is not None else 0.0
        rgb_exec=bool(float(met.get('rgb_bilateral_step',0.0) or 0.0)>=3.0 or float(met.get('raw_crease_strength',0.0) or 0.0)>=0.28)
        if not (span>=min_span and valid>=0.50 and seam_overlap<0.18 and fold>=rescue_thr and deform_support*pose_conf>=0.10 and align>=min_align and proximity>=0.15 and rgb_exec):
            continue
        fs=float(np.clip(fold+0.055,0.0,1.0))
        branch={
            'branch_id':int(10000+idx),'source':'STRUCTURAL','fold_score_raw':float(fold),'fold_score':fs,
            **{k:v for k,v in met.items() if k!='fold_score'},
            'source_bonus':0.055,'structural_rescue':True,'depth_execution_ok':True,'rgb_execution_ok':True,
            'unidepth_fold_support':0.0,'contour_near_veto':False,'contour_near_raw':False,'edge_flap_rescue':True,
            'preview_ok':True,'execution_ok':True,'mask':np.asarray(r['mask'],dtype=np.uint8),
            'v11_pose_local_score_rescue':True,'v11_pose_local_percentile':float(pct),
            'v11_pose_local_threshold':float(threshold),'v11_pose_normal_alignment':float(align),
            'v11_pose_anchor_proximity':float(proximity),
        }
        rank_score=float(0.34*fs+0.24*float(np.clip(exact.get('quality',0.0) or 0.0,0.0,1.0))+0.22*align+0.20*proximity)
        out.append({'branch':branch,'exact':exact,'rank_score':rank_score,'alignment':align,'proximity':proximity})
    out.sort(key=lambda z:float(z.get('rank_score',0.0)),reverse=True)
    if out:
        top=out[0]
        print(f"[E156-V11-POSE-LOCAL-RGB] candidates={len(out)} anchor=({anchor[0]:.1f},{anchor[1]:.1f}) radius={radius:.1f}px pct={pct:.0f} topFold={float(top['branch'].get('fold_score',0.0)):.3f} topQ={float(top['exact'].get('quality',0.0)):.3f} align={float(top['alignment']):.3f}")
    return out

def _v8_rgb_adaptive_rescue_localize(depth_state, mask_u8, H_value, args, primary_joint=None):
    """V7 high-recall / fail-closed RGB rescue when Pose localization fails.

    The base E94/E122 crease map is already normalized against the current garment
    interior.  Rescue therefore never thresholds raw brightness/color or CENTER
    error.  It admits either (A) a coherent E150 edge/flap cluster with an exact
    normal, or (B) a stronger global exact Fold-Crease from STRUCTURAL E122
    execution branches.  A neckline-relative veto and the normal planner's hard
    board/reach/path checks remain mandatory.
    """
    if not bool(getattr(args,'micro_unfold_vla_rgb_rescue_enable',True)):
        return {'ok':False,'reason':'RGB_RESCUE_DISABLED','policy':'V8_RGB_ADAPTIVE_RESCUE'}
    mask=((np.asarray(mask_u8,dtype=np.uint8)>0).astype(np.uint8)*255)
    if cv2.countNonZero(mask)<500:
        return {'ok':False,'reason':'RGB_RESCUE_BAD_GARMENT_MASK','policy':'V8_RGB_ADAPTIVE_RESCUE'}

    min_exec=max(6,int(getattr(args,'micro_unfold_vla_joint_min_exec_pixels',12)))
    min_span=max(8.0,float(getattr(args,'micro_unfold_vla_joint_min_branch_span_px',18.0)))
    edge_min_q=float(np.clip(getattr(args,'micro_unfold_vla_rgb_rescue_edge_min_quality',0.46),0.0,1.0))
    global_min_q=float(np.clip(getattr(args,'micro_unfold_vla_rgb_rescue_global_min_quality',0.55),0.0,1.0))
    global_min_branch=float(np.clip(getattr(args,'micro_unfold_vla_rgb_rescue_global_min_branch_score',0.60),0.0,1.0))
    neck_ratio_lim=max(0.05,float(getattr(args,'micro_unfold_vla_rgb_rescue_neck_distance_ratio',0.55)))
    neck_parallel_lim=float(np.clip(getattr(args,'micro_unfold_vla_rgb_rescue_neck_parallel_alignment',0.72),0.0,1.0))

    pool=_e122_build_scored_crease_pool(depth_state,mask,args,H_value=H_value)
    structural=[]; structural_exec=np.zeros_like(mask)
    for q in list((pool or {}).get('branches',[]) or []):
        if not isinstance(q,dict) or not bool(q.get('execution_ok',False)) or str(q.get('source','')).upper()!='STRUCTURAL':
            continue
        bm=np.asarray(q.get('mask',np.zeros_like(mask)),dtype=np.uint8)
        if cv2.countNonZero(bm)<min_exec or float(q.get('span_px',0.0) or 0.0)<min_span:
            continue
        structural.append(q); structural_exec[bm>0]=255

    # V11 low-contrast flap recovery: when Pose deformation provides a concrete
    # local anchor but normal E122 thinning missed that fold, recover candidates
    # from the already garment-relative fold-score map around that anchor.
    pose_local_items=_v11_pose_local_score_crease_candidates(depth_state,mask,args)
    for _pl in list(pose_local_items or []):
        q=dict(_pl.get('branch',{}) or {})
        bm=np.asarray(q.get('mask',np.zeros_like(mask)),dtype=np.uint8)
        if cv2.countNonZero(bm)<min_exec:
            continue
        structural.append(q); structural_exec[bm>0]=255
    if not structural or cv2.countNonZero(structural_exec)<min_exec:
        return {'ok':False,'reason':'RGB_RESCUE_NO_STRUCTURAL_EXECUTION_CREASE','policy':'V8_RGB_ADAPTIVE_RESCUE',
                'structural_execution_branch_count':int(len(structural))}

    def evaluate(exact, mode, extra_score=0.0):
        if not isinstance(exact,dict):
            return None
        qmin=edge_min_q if str(mode).startswith('EDGE') else global_min_q
        quality=float(np.clip(exact.get('quality',0.0) or 0.0,0.0,1.0))
        neck=_v7_neckline_relation(depth_state,exact)
        if bool(neck.get('available',False)) and float(neck.get('distance_ratio',999.0))<=neck_ratio_lim and float(neck.get('parallel_alignment',0.0))>=neck_parallel_lim:
            return None
        matches=_v7_candidate_branch_matches(exact.get('crease_mask',np.zeros_like(mask)),structural,dilate_px=8)
        if not matches:
            return None
        bestm=matches[0]; bestq=bestm['branch']
        branch_score=float(np.clip(bestq.get('fold_score',0.0) or 0.0,0.0,1.0))
        near_fraction=float(np.clip(bestm.get('near_fraction',0.0) or 0.0,0.0,1.0))
        direct_fraction=float(np.clip(bestm.get('direct_fraction',0.0) or 0.0,0.0,1.0))
        # V8: do not let exact-normal quality alone veto an otherwise coherent
        # garment-relative Fold-Crease. Normalize every cue to the already-existing
        # V7 relative thresholds, then require fused support. This is color/brightness
        # adaptive because E94/E122 quality and branch scores are garment-relative.
        fused_min=float(np.clip(getattr(args,'micro_unfold_vla_rgb_rescue_fused_min_support',0.90),0.60,1.20))
        primary_ratio_min=float(np.clip(getattr(args,'micro_unfold_vla_rgb_rescue_min_primary_ratio',0.72),0.40,1.00))
        q_support=float(np.clip(quality/max(1e-6,qmin),0.0,1.25))
        b_support=float(np.clip(branch_score/max(1e-6,global_min_branch),0.0,1.25))
        overlap_support=float(np.clip(near_fraction/0.80,0.0,1.25))
        direct_support=float(np.clip(direct_fraction/0.65,0.0,1.25))
        extra_support=float(np.clip(extra_score/max(1e-6,global_min_branch),0.0,1.25))
        fused_support=float(0.30*q_support+0.30*b_support+0.22*overlap_support+0.10*direct_support+0.08*extra_support)
        primary_ok=bool(q_support>=primary_ratio_min or b_support>=primary_ratio_min)
        if (not primary_ok) or fused_support+1e-9<fused_min:
            return None
        kept=[]; kept_ids=[]
        for m in matches[:3]:
            if float(m.get('near_fraction',0.0))<0.40:
                continue
            q=m['branch']; bid=int(q.get('branch_id',-1))
            if bid in kept_ids:
                continue
            kept.append(q); kept_ids.append(bid)
        if not kept:
            kept=[bestq]; kept_ids=[int(bestq.get('branch_id',-1))]
        seam=max(float(np.clip(q.get('seam_overlap',0.0) or 0.0,0.0,1.0)) for q in kept)
        # E122's semantic seam veto has already acted upstream.  A large residual
        # seam overlap is nevertheless rejected here rather than becoming motion authority.
        if seam>0.30:
            return None
        confidence=float(np.clip(0.38*quality+0.34*branch_score+0.18*near_fraction+0.06*direct_fraction+0.04*np.clip(extra_score,0.0,1.0),0.0,1.0))
        return {'exact':exact,'mode':str(mode),'quality':quality,'branch_score':branch_score,
                'confidence':confidence,'fused_support':float(fused_support),
                'quality_support':float(q_support),'branch_support':float(b_support),
                'near_fraction':float(near_fraction),'direct_fraction':float(direct_fraction),
                'kept':kept,'kept_ids':kept_ids,'best_match':bestm,
                'neck':neck,'seam_overlap':seam,'extra_score':float(extra_score)}

    candidates=[]
    # Highest authority in V11 rescue: a REAL RGB crease recovered around the
    # strong Pose-deformation anchor. Pose supplies location only; RGB still has
    # to provide the bilateral/crease geometry and the normal must agree.
    for _ri,_pl in enumerate(list(pose_local_items or []),start=1):
        cand=evaluate(_pl.get('exact'),'POSE_LOCAL_SCORE',extra_score=float(depth_state.get('pose_deform_confidence',0.0) or 0.0))
        if cand is not None:
            cand['exact_rank']=int(_ri); cand['pose_local_rank_score']=float(_pl.get('rank_score',0.0) or 0.0)
            candidates.append(cand)
            print(f"[E156-V11-POSE-LOCAL-ACCEPT] rank={_ri} quality={float(cand.get('quality',0.0)):.3f} branch={float(cand.get('branch_score',0.0)):.3f} fused={float(cand.get('fused_support',0.0)):.3f}")

    # If a strong spatial Pose hypothesis exists, do not abandon that location and
    # jump to an unrelated global wrinkle.  Global/edge rescue is reserved for
    # cases where Pose does not provide a reliable fold anchor.
    _pose_anchor_expected=bool(
        depth_state.get('pose_deform_available',False)
        and float(depth_state.get('pose_deform_fold_region_confidence',0.0) or 0.0)>=float(np.clip(getattr(args,'micro_unfold_vla_joint_deform_region_min_conf',0.20),0.0,1.0))
        and depth_state.get('pose_deform_fold_anchor_px') is not None
    )
    if not candidates and not _pose_anchor_expected:
        # Prefer a coherent contour-adjacent flap cluster when it exists.  E150 itself
        # is relative/geometry based; it does not use garment color as an absolute gate.
        full_exec=np.asarray((pool or {}).get('execution_skeleton',np.zeros_like(mask)),dtype=np.uint8)
        try:
            edge=_e150_select_edge_crease_cluster(depth_state,mask,H_value,full_exec,np.zeros_like(mask)) if cv2.countNonZero(full_exec)>=min_exec else None
        except Exception as exc:
            edge=None
            print(f'[E156-V11-RGB-RESCUE-EDGE-WARN] {exc!r}')
        if isinstance(edge,dict) and bool(edge.get('cluster_found',False)):
            cl=dict(edge.get('cluster',{}) or {}); cl_score=float(np.clip(cl.get('score',0.0) or 0.0,0.0,1.0))
            for exact in list(edge.get('candidates',[]) or [])[:3]:
                cand=evaluate(exact,'EDGE_FLAP_CLUSTER',extra_score=cl_score)
                if cand is not None:
                    candidates.append(cand)

        # Do not let one neck-like/high-score component monopolize global rescue.
        try:
            exact_list=_v11_rank_exact_fold_normals(
                depth_state, mask, structural_exec, np.zeros_like(mask),
                max_candidates=max(1,int(getattr(args,'micro_unfold_vla_exact_crease_rerank_max',6))), H_value=H_value, args=args)
        except Exception as exc:
            exact_list=[]
            print(f'[E156-V11-RGB-RESCUE-RERANK-WARN] {exc!r}')
        for _ri, exact in enumerate(list(exact_list or []), start=1):
            cand=evaluate(exact,'GLOBAL_EXACT_RERANK',extra_score=0.0)
            if cand is not None:
                cand['exact_rank']=int(_ri)
                candidates.append(cand)
                print(f"[E156-V11-RGB-RESCUE-RERANK-ACCEPT] rank={_ri} quality={float(cand.get('quality',0.0)):.3f} branch={float(cand.get('branch_score',0.0)):.3f} fused={float(cand.get('fused_support',0.0)):.3f}")

    if not candidates:
        return {'ok':False,'reason':'RGB_RESCUE_NO_NON_NECK_ADAPTIVE_FOLD','policy':'V8_RGB_ADAPTIVE_RESCUE',
                'structural_execution_branch_count':int(len(structural)),
                'primary_joint_reason':str((primary_joint or {}).get('reason','NONE'))}
    candidates.sort(key=lambda z:float(z.get('confidence',0.0)),reverse=True)
    chosen=candidates[0]; exact=chosen['exact']
    filtered=_v6_filter_e122_pool(depth_state,mask,args,H_value,pool,chosen['kept'])
    selected_skel=np.asarray(filtered.get('execution_skeleton',np.zeros_like(mask)),dtype=np.uint8)
    if cv2.countNonZero(selected_skel)<min_exec:
        return {'ok':False,'reason':'RGB_RESCUE_FILTERED_CREASE_EMPTY','policy':'V8_RGB_ADAPTIVE_RESCUE'}

    # In rescue mode Pose remains diagnostic only.  Clear its motion-authority
    # fields so E125/E133 cannot jump back to NECK or a conflicting Pose ROI.
    z=np.zeros_like(mask)
    depth_state['pose_fold_intrusions']=[]
    depth_state['pose_fold_intrusion_available']=False
    depth_state['pose_fold_intrusion_confidence']=0.0
    depth_state['pose_fold_intrusion_mask']=z.copy()
    depth_state['pose_fold_region_mask']=z.copy()
    depth_state['pose_fold_region_confidence']=0.0
    depth_state['pose_fold_primary_group']='NONE'
    depth_state['pose_fold_active_groups']=[]
    depth_state['pose_deform_available']=False
    depth_state['pose_deform_confidence']=0.0
    depth_state['pose_deform_mask']=z.copy()
    depth_state['pose_deform_mode']='NONE'
    depth_state['pose_deform_side_hint']='NONE'
    depth_state['pose_deform_fold_region_hint']='NONE'
    depth_state['pose_deform_fold_region_confidence']=0.0
    depth_state['pose_deform_fold_anchor_px']=None
    depth_state['pose_deform_fold_direction_px']=None
    depth_state['vla_joint_fold_region_valid']=False
    depth_state['vla_joint_fold_region_mask']=z.copy()
    depth_state['vla_joint_pose_source']='RGB_ADAPTIVE_RESCUE'
    depth_state['vla_joint_pose_group']=str(chosen['mode'])
    depth_state['vla_joint_pose_confidence']=float(chosen['confidence'])
    depth_state['vla_joint_branch_ids']=list(chosen['kept_ids'])
    depth_state['vla_joint_selected_branch_id']=int(chosen['kept_ids'][0])
    depth_state['vla_joint_selected_branch_overlap']=float(chosen['best_match'].get('near_fraction',0.0))
    depth_state['vla_joint_selected_orientation_alignment']=1.0
    depth_state['vla_joint_selected_branch_score']=float(chosen['branch_score'])
    depth_state['vla_joint_selected_score']=float(chosen['confidence'])
    depth_state['vla_joint_strict_active']=True
    depth_state['vla_joint_rgb_rescue']=True
    depth_state['vla_joint_selected_rgb_skeleton']=selected_skel.copy()
    depth_state['vla_rgb_rescue_anchor_px']=np.asarray(exact.get('anchor_px'),dtype=np.float64)
    depth_state['vla_rgb_rescue_normal_px']=np.asarray(exact.get('normal_px'),dtype=np.float64)
    depth_state['vla_rgb_rescue_tangent_px']=np.asarray(exact.get('tangent_px'),dtype=np.float64)
    depth_state['vla_rgb_rescue_exact_crease_mask']=np.asarray(exact.get('crease_mask'),dtype=np.uint8)
    depth_state['vla_rgb_rescue_mode']=str(chosen['mode'])
    depth_state['vla_rgb_rescue_confidence']=float(chosen['confidence'])
    depth_state['vla_rgb_rescue_exact_quality']=float(chosen['quality'])
    depth_state['vla_rgb_rescue_branch_score']=float(chosen['branch_score'])
    depth_state['vla_rgb_rescue_fused_support']=float(chosen.get('fused_support',0.0))
    depth_state['vla_rgb_rescue_neck_distance_ratio']=float(chosen['neck'].get('distance_ratio',999.0))
    depth_state['vla_rgb_rescue_neck_parallel_alignment']=float(chosen['neck'].get('parallel_alignment',0.0))
    print(
        f"[E156-V10-RGB-RESCUE-READY] mode={chosen['mode']} conf={chosen['confidence']:.3f} "
        f"Q={chosen['quality']:.3f} branchScore={chosen['branch_score']:.3f} fused={chosen.get('fused_support',0.0):.3f} kept={chosen['kept_ids']} "
        f"neckRatio={float(chosen['neck'].get('distance_ratio',999.0)):.3f} "
        f"neckParallel={float(chosen['neck'].get('parallel_alignment',0.0)):.3f}"
    )
    return {
        'ok':True,'reason':'RGB_ADAPTIVE_FOLD_RESCUE','policy':'V8_RGB_ADAPTIVE_RESCUE',
        'pose_source':'RGB_ADAPTIVE_RESCUE','pose_group':str(chosen['mode']),
        'pose_confidence':float(chosen['confidence']),
        'selected_branch_id':int(chosen['kept_ids'][0]),'kept_branch_ids':list(chosen['kept_ids']),
        'branch_overlap':float(chosen['best_match'].get('near_fraction',0.0)),
        'orientation_alignment':1.0,'branch_fold_score':float(chosen['branch_score']),
        'joint_score':float(chosen['confidence']),
        'rgb_execution_branch_count':int(len(structural)),
        'rgb_rescue_used':True,'rgb_rescue_mode':str(chosen['mode']),
        'rgb_rescue_exact_quality':float(chosen['quality']),
        'rgb_rescue_branch_score':float(chosen['branch_score']),
        'rgb_rescue_fused_support':float(chosen.get('fused_support',0.0)),
        'rgb_rescue_neck_distance_ratio':float(chosen['neck'].get('distance_ratio',999.0)),
        'rgb_rescue_neck_parallel_alignment':float(chosen['neck'].get('parallel_alignment',0.0)),
        'primary_joint_reason':str((primary_joint or {}).get('reason','NONE')),
    }


def _v10_exact_plan_mask(full_plan, depth_state, plan_key, state_key):
    """Prefer the exact real crease/region that generated this plan; else joint state.

    A synthetic/virtual directional split must never replace the mandatory RGB
    evidence mask in the final crossing check.
    """
    pm = full_plan.get(plan_key) if isinstance(full_plan, dict) else None
    plan_mask_allowed = True
    if plan_key == 'selected_crease_component' and isinstance(full_plan, dict):
        crease_source = str(full_plan.get('e116_crease_source', full_plan.get('crease_source', '')) or '').upper()
        if 'VIRTUAL' in crease_source:
            plan_mask_allowed = False
    if pm is not None and plan_mask_allowed:
        arr = np.asarray(pm, dtype=np.uint8)
        if arr.ndim == 2 and np.any(arr > 0):
            return (arr > 0), f"PLAN:{plan_key}"
    sm = depth_state.get(state_key) if isinstance(depth_state, dict) else None
    if sm is not None:
        arr = np.asarray(sm, dtype=np.uint8)
        if arr.ndim == 2 and np.any(arr > 0):
            return (arr > 0), f"STATE:{state_key}"
    return np.empty((0, 0), dtype=bool), "MISSING"


def _v7_validate_joint_plan_roles(full_plan, depth_state):
    """V10 final fail-closed role check tied to the crease that made the plan.

    V9 could localize one RGB branch, build a mechanically safe plan from another
    exact selected crease, then validate crossing against the earlier joint-state
    skeleton.  V10 first validates against ``selected_crease_component`` carried by
    the actual plan.  This keeps the final check independent and fail-closed while
    preventing a branch-ID mismatch from discarding a valid same-crease plan.
    """
    if not bool(depth_state.get('vla_joint_strict_active',False)):
        return True,{'reason':'JOINT_NOT_ACTIVE'}
    crease, crease_source = _v10_exact_plan_mask(
        full_plan, depth_state, 'selected_crease_component', 'vla_joint_selected_rgb_skeleton')
    if crease.ndim!=2 or not np.any(crease):
        return False,{'reason':'JOINT_SELECTED_RGB_CREASE_MISSING','crease_source':crease_source}
    try:
        a=np.asarray(full_plan.get('active_px'),dtype=np.float64).reshape(2)
        hpt=np.asarray(full_plan.get('hold_px'),dtype=np.float64).reshape(2)
    except Exception:
        return False,{'reason':'JOINT_ROLE_POINT_MISSING','crease_source':crease_source}
    hh,ww=crease.shape
    v=a-hpt; dist=float(np.linalg.norm(v))
    if dist<=1e-6:
        return False,{'reason':'JOINT_ROLE_PAIR_COLLAPSED','crease_source':crease_source}
    n=int(np.clip(round(dist*1.4),25,220))
    t=np.linspace(0.0,1.0,n,dtype=np.float64)
    pts=hpt[None,:]+t[:,None]*v[None,:]
    xi=np.clip(np.rint(pts[:,0]).astype(np.int32),0,ww-1); yi=np.clip(np.rint(pts[:,1]).astype(np.int32),0,hh-1)
    near=cv2.dilate((crease.astype(np.uint8)*255),cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(13,13)),iterations=1)>0
    crosses=bool(np.any(near[yi,xi]))

    if bool(depth_state.get('vla_joint_rgb_rescue',False)):
        normal=_v6_unit_xy(full_plan.get('v12_crease_normal_px', depth_state.get('vla_rgb_rescue_normal_px')))
        if normal is None:
            return False,{'reason':'RGB_RESCUE_ROLE_NORMAL_MISSING','pair_crosses_selected_rgb_crease':bool(crosses),'crease_source':crease_source}
        outward_projection=float(np.dot(v/dist,normal))
        try:
            # Use the plan's exact crease anchor when available so branch identity
            # stays consistent through localization -> pair generation -> validation.
            anchor_raw=full_plan.get('crease_anchor_px',depth_state.get('vla_rgb_rescue_anchor_px'))
            anchor=np.asarray(anchor_raw,dtype=np.float64).reshape(2)
            active_signed=float(np.dot(a-anchor,normal))
            hold_signed=float(np.dot(hpt-anchor,normal))
        except Exception:
            return False,{'reason':'RGB_RESCUE_ROLE_ANCHOR_MISSING','pair_crosses_selected_rgb_crease':bool(crosses),'crease_source':crease_source}
        straddles=bool(active_signed>0.0 and hold_signed<0.0)
        ok=bool(crosses and outward_projection>0.0 and straddles)
        reason='OK' if ok else (
            f'crossesRGB={crosses} outwardProjection={outward_projection:.3f} '
            f'activeSigned={active_signed:.2f} holdSigned={hold_signed:.2f}')
        return ok,{
            'reason':reason,'mode':'RGB_ADAPTIVE_RESCUE',
            'crease_source':crease_source,
            'pair_crosses_selected_rgb_crease':bool(crosses),
            'active_minus_hold_outward_projection':float(outward_projection),
            'active_signed_from_crease_px':float(active_signed),
            'hold_signed_from_crease_px':float(hold_signed),
            'straddles_selected_rgb_crease':bool(straddles),
        }

    roi, roi_source = _v10_exact_plan_mask(
        full_plan, depth_state, 'primary_fold_region', 'vla_joint_fold_region_mask')
    if roi.shape!=crease.shape or not np.any(roi):
        return False,{'reason':'JOINT_ROLE_MAP_MISSING','crease_source':crease_source,'roi_source':roi_source}
    def support(p,rad=14):
        x=int(round(float(p[0]))); y=int(round(float(p[1])))
        x0=max(0,x-rad); x1=min(ww,x+rad+1); y0=max(0,y-rad); y1=min(hh,y+rad+1)
        if x0>=x1 or y0>=y1:
            return 0.0
        yy,xx=np.ogrid[y0:y1,x0:x1]
        disk=((xx-x)**2+(yy-y)**2)<=rad*rad
        vals=roi[y0:y1,x0:x1][disk]
        return float(np.mean(vals)) if vals.size else 0.0
    a_sup=support(a); h_sup=support(hpt)
    ok=bool(a_sup>=0.08 and (a_sup-h_sup)>=0.03 and h_sup<=0.70 and crosses)
    reason='OK' if ok else f'A={a_sup:.2f} H={h_sup:.2f} delta={a_sup-h_sup:.2f} crossesRGB={crosses}'
    return ok,{
        'reason':reason,'active_fold_support':float(a_sup),'hold_fold_support':float(h_sup),
        'support_delta':float(a_sup-h_sup),'pair_crosses_selected_rgb_crease':bool(crosses),
        'crease_source':crease_source,'roi_source':roi_source,
    }

def _v6_validate_joint_plan_roles(full_plan, depth_state):
    """Final fail-closed check: ACTIVE must be on the proposed flap ROI and HOLD on BODY side."""
    if not bool(depth_state.get('vla_joint_strict_active',False)):
        return True,{'reason':'JOINT_NOT_ACTIVE'}
    roi=np.asarray(depth_state.get('vla_joint_fold_region_mask',np.empty((0,0),dtype=np.uint8)),dtype=np.uint8)>0
    crease=np.asarray(depth_state.get('vla_joint_selected_rgb_skeleton',np.empty((0,0),dtype=np.uint8)),dtype=np.uint8)>0
    if roi.ndim!=2 or crease.shape!=roi.shape or not np.any(roi) or not np.any(crease):
        return False,{'reason':'JOINT_ROLE_MAP_MISSING'}
    try:
        a=np.asarray(full_plan.get('active_px'),dtype=np.float64).reshape(2)
        hpt=np.asarray(full_plan.get('hold_px'),dtype=np.float64).reshape(2)
    except Exception:
        return False,{'reason':'JOINT_ROLE_POINT_MISSING'}
    hh,ww=roi.shape
    def support(p,rad=14):
        x=int(round(float(p[0]))); y=int(round(float(p[1])))
        x0=max(0,x-rad); x1=min(ww,x+rad+1); y0=max(0,y-rad); y1=min(hh,y+rad+1)
        if x0>=x1 or y0>=y1:
            return 0.0
        yy,xx=np.ogrid[y0:y1,x0:x1]
        disk=((xx-x)**2+(yy-y)**2)<=rad*rad
        vals=roi[y0:y1,x0:x1][disk]
        return float(np.mean(vals)) if vals.size else 0.0
    a_sup=support(a); h_sup=support(hpt)
    v=a-hpt; dist=float(np.linalg.norm(v))
    if dist<=1e-6:
        return False,{'reason':'JOINT_ROLE_PAIR_COLLAPSED','active_support':a_sup,'hold_support':h_sup}
    n=int(np.clip(round(dist*1.4),25,220))
    t=np.linspace(0.0,1.0,n,dtype=np.float64)
    pts=hpt[None,:]+t[:,None]*v[None,:]
    xi=np.clip(np.rint(pts[:,0]).astype(np.int32),0,ww-1); yi=np.clip(np.rint(pts[:,1]).astype(np.int32),0,hh-1)
    near=cv2.dilate((crease.astype(np.uint8)*255),cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(13,13)),iterations=1)>0
    crosses=bool(np.any(near[yi,xi]))
    # Deliberately modest support thresholds: Pose ROI is coarse, but ACTIVE must
    # still be more flap-like than HOLD and their physical segment must cross the
    # corroborating RGB fold boundary.
    ok=bool(a_sup>=0.08 and (a_sup-h_sup)>=0.03 and h_sup<=0.70 and crosses)
    reason='OK' if ok else f'A={a_sup:.2f} H={h_sup:.2f} delta={a_sup-h_sup:.2f} crossesRGB={crosses}'
    return ok,{
        'reason':reason,'active_fold_support':float(a_sup),'hold_fold_support':float(h_sup),
        'support_delta':float(a_sup-h_sup),'pair_crosses_selected_rgb_crease':bool(crosses),
    }


def _v38_min_boundary_distance_mm(board_xy, contour_board):
    q = np.asarray(board_xy, dtype=np.float64).reshape(2)
    c = np.asarray(contour_board, dtype=np.float64).reshape(-1, 2)
    if c.size == 0:
        return float("inf"), None
    d2 = np.sum((c - q[None, :]) ** 2, axis=1)
    i = int(np.argmin(d2))
    return float(math.sqrt(max(0.0, float(d2[i])))), c[i].copy()


def _v48_target_cca_context(target_board, garment_centroid_board, args):
    """v49 compatibility name: resolve cached-local fold context; CCA is soft only.

    v48 made READY-CCA a hard Target authority.  Real logs showed that assumption can
    both reject a correct flap and accept a strong seam/neck crease.  v49 therefore
    accepts direction authority only from the main runtime's already-computed same-X
    local heatmap/crease validation.  READY-CCA summaries are still inspected for
    diagnostics/soft support, but they can never hard-block or authorize a Target.
    """
    t0 = time.perf_counter()
    target = np.asarray(target_board, dtype=np.float64).reshape(2)
    centroid = np.asarray(garment_centroid_board, dtype=np.float64).reshape(2)
    local = getattr(args, "micro_unfold_v49_local_fold_context", None)
    diag = {
        "enabled": True,
        "measurement_valid": bool(getattr(args, "micro_unfold_v48_cca_measurement_valid", False)),
        "pass": True, "hard_block": False, "reason": "V49_LOCAL_CONTEXT_UNAVAILABLE",
        "component_count": 0, "usable_count": 0, "pool": "SOFT_ONLY",
        "selected_component_id": None, "selected_rule": None,
        "target_perp_mm": None, "target_axial_overrun_mm": None,
        "target_segment_distance_mm": None, "adaptive_perp_limit_mm": None,
        "cca_authority": False, "local_context_authority": False,
        "elapsed_ms": 0.0,
    }

    if isinstance(local, dict):
        try:
            anchor = np.asarray(local.get("anchor_board"), dtype=np.float64).reshape(2)
            tangent = np.asarray(local.get("tangent_board"), dtype=np.float64).reshape(2)
            normal = np.asarray(local.get("normal_board"), dtype=np.float64).reshape(2)
            tn = float(np.linalg.norm(tangent)); nn = float(np.linalg.norm(normal))
            if not (np.all(np.isfinite(anchor)) and np.all(np.isfinite(tangent)) and np.all(np.isfinite(normal)) and tn > 1e-9 and nn > 1e-9):
                raise ValueError("non-finite/collapsed local fold frame")
            tangent = tangent / tn; normal = normal / nn
            # Keep BODY->flap sign stable even if a serialized context lost its sign.
            if float(np.dot(normal, anchor - centroid)) < 0.0:
                normal = -normal
            target_dist = float(np.linalg.norm(target - anchor))
            declared = float(local.get("target_segment_distance_mm", target_dist) or target_dist)
            limit = max(15.0, float(local.get("adaptive_perp_limit_mm", getattr(args, "micro_unfold_target_match_max_mm", 120.0)) or 120.0))
            # A stale context is the only v49 hard block: it means the Target changed
            # after validation but its old direction frame was accidentally retained.
            stale = bool(target_dist > max(limit + 10.0, 1.35 * limit))
            if stale:
                diag.update({
                    "pass": False, "hard_block": True, "reason": "V49_STALE_LOCAL_FOLD_CONTEXT",
                    "target_perp_mm": target_dist, "target_segment_distance_mm": target_dist,
                    "adaptive_perp_limit_mm": limit, "local_context_authority": True,
                })
                diag["elapsed_ms"] = 1000.0 * (time.perf_counter() - t0)
                return None, diag
            ctx = {
                "component_id": int(local.get("component_id", -49) or -49),
                "component_rule": str(local.get("component_rule", "V49_CACHED_LOCAL_WRINKLE")),
                "anchor_board": anchor,
                "tangent_board": tangent,
                "normal_board": normal,
                "component_center_board": anchor.copy(),
                "component_endpoints_board": [anchor - tangent * 30.0, anchor + tangent * 30.0],
                "target_perp_mm": float(local.get("target_perp_mm", declared) or declared),
                "target_axial_overrun_mm": float(local.get("target_axial_overrun_mm", 0.0) or 0.0),
                "target_segment_distance_mm": float(declared),
                "adaptive_perp_limit_mm": float(limit),
                "target_validation_ms": float(local.get("target_validation_ms", 0.0) or 0.0),
                "source": str(local.get("source", "V49_CACHED_LOCAL")),
                "local_score": float(local.get("local_score", 0.0) or 0.0),
                "cca_soft_support": copy.deepcopy(local.get("cca_soft_support")),
            }
            diag.update({
                "pass": True, "hard_block": False, "reason": "V49_CACHED_LOCAL_FOLD_OK",
                "selected_component_id": int(ctx["component_id"]),
                "selected_rule": str(ctx["component_rule"]),
                "target_perp_mm": float(ctx["target_perp_mm"]),
                "target_axial_overrun_mm": float(ctx["target_axial_overrun_mm"]),
                "target_segment_distance_mm": float(ctx["target_segment_distance_mm"]),
                "adaptive_perp_limit_mm": float(ctx["adaptive_perp_limit_mm"]),
                "local_context_authority": True,
                "local_score": float(ctx["local_score"]),
                "cca_soft_support": _training_json_safe(ctx.get("cca_soft_support")),
            })
            diag["elapsed_ms"] = 1000.0 * (time.perf_counter() - t0)
            ctx["target_validation_ms"] = float(diag["elapsed_ms"] + float(ctx.get("target_validation_ms", 0.0) or 0.0))
            return ctx, diag
        except Exception as exc:
            diag.update({"pass": False, "hard_block": True, "reason": f"V49_LOCAL_FOLD_CONTEXT_INVALID:{exc!r}",
                         "local_context_authority": True})
            diag["elapsed_ms"] = 1000.0 * (time.perf_counter() - t0)
            return None, diag

    # CCA retained only for observability.  Compute the nearest usable summary, but
    # never return it as a direction context and never hard-block Target-direct.
    raw = list(getattr(args, "micro_unfold_v48_cca_components", ()) or ())
    diag["component_count"] = int(len(raw))
    best = None
    for idx, item in enumerate(raw):
        if not isinstance(item, dict):
            continue
        rule = str(item.get("cca_long_rule") or "none")
        if rule not in {"strong", "medium_thin"}:
            continue
        try:
            eps = item.get("major_board_endpoints")
            if isinstance(eps, (list, tuple)) and len(eps) >= 2:
                a = np.asarray(eps[0], dtype=np.float64).reshape(2)
                b = np.asarray(eps[1], dtype=np.float64).reshape(2)
            else:
                center = np.asarray(item.get("center_board", item.get("component_center_board")), dtype=np.float64).reshape(2)
                axis = np.asarray(item.get("major_board_unit"), dtype=np.float64).reshape(2)
                axis /= max(1e-9, float(np.linalg.norm(axis)))
                ll = max(1.0, float(item.get("major_len_mm", 0.0) or 0.0))
                a, b = center - axis * (0.5 * ll), center + axis * (0.5 * ll)
            v = b - a; vv = float(np.dot(v, v))
            if vv <= 1e-9:
                continue
            tau_raw = float(np.dot(target - a, v) / vv)
            tau = float(np.clip(tau_raw, 0.0, 1.0))
            anchor = a + tau * v
            dist = float(np.linalg.norm(target - anchor))
            over = float(max(0.0, -tau_raw, tau_raw - 1.0) * math.sqrt(vv))
            key = (dist + 0.35 * over, 0 if rule == "strong" else 1)
            if best is None or key < best[0]:
                best = (key, idx, item, dist, over)
        except Exception:
            continue
    diag["usable_count"] = int(sum(1 for x in raw if isinstance(x, dict) and str(x.get("cca_long_rule") or "none") in {"strong", "medium_thin"}))
    if best is not None:
        _, idx, item, dist, over = best
        diag.update({
            "reason": "V49_LOCAL_CONTEXT_UNAVAILABLE_CCA_SOFT_ONLY",
            "selected_component_id": int(item.get("component_id", item.get("label_id", idx)) or idx),
            "selected_rule": str(item.get("cca_long_rule") or "none"),
            "target_segment_distance_mm": float(dist), "target_perp_mm": float(dist),
            "target_axial_overrun_mm": float(over), "cca_authority": False,
        })
    else:
        diag["reason"] = "V49_LOCAL_CONTEXT_UNAVAILABLE_NO_CCA_SOFT_SUPPORT"
    diag["elapsed_ms"] = 1000.0 * (time.perf_counter() - t0)
    return None, diag


def _v48_direction_hint_for_active(active, fold_ctx, legacy_outward, args):
    """Return the validated same-crease normal for Target-direct, or legacy hint if disabled."""
    t0=time.perf_counter()
    if not bool(getattr(args,"micro_unfold_v48_direction_validation_enable",True)) or not isinstance(fold_ctx,dict):
        out=np.asarray(legacy_outward,dtype=np.float64).reshape(2)
        n=float(np.linalg.norm(out))
        return ((out/n) if n>1e-9 else None),{"used":False,"pass":bool(n>1e-9),"reason":"LEGACY_OUTWARD","elapsed_ms":1000.0*(time.perf_counter()-t0)}
    try:
        normal=np.asarray(fold_ctx["normal_board"],dtype=np.float64).reshape(2)
        tangent=np.asarray(fold_ctx["tangent_board"],dtype=np.float64).reshape(2)
        anchor=np.asarray(fold_ctx["anchor_board"],dtype=np.float64).reshape(2)
        board=np.asarray(active["board"],dtype=np.float64).reshape(2)
        normal/=max(1e-9,float(np.linalg.norm(normal))); tangent/=max(1e-9,float(np.linalg.norm(tangent)))
        signed=float(np.dot(board-anchor,normal)); tangent_offset=float(abs(np.dot(board-anchor,tangent)))
        d={"used":True,"pass":True,"reason":"CCA_CREASE_NORMAL","active_signed_mm":signed,
           "active_tangent_offset_mm":tangent_offset,"elapsed_ms":1000.0*(time.perf_counter()-t0)}
        return normal,d
    except Exception as exc:
        return None,{"used":True,"pass":False,"reason":f"DIRECTION_CONTEXT_INVALID:{exc!r}","elapsed_ms":1000.0*(time.perf_counter()-t0)}


def _v48_fast_exact_matches_fold_context(exact, H_value, fold_ctx, args):
    """Cheap same-component guard for FAST target-local candidates after Target-direct blocks mechanically."""
    if not bool(getattr(args,"micro_unfold_v48_fast_same_cca_enable",True)) or not isinstance(fold_ctx,dict):
        return True,{"used":False,"reason":"DISABLED_OR_NO_CONTEXT"}
    try:
        anchor_px=np.asarray(exact.get("anchor_px"),dtype=np.float64).reshape(2)
        tangent_px=np.asarray(exact.get("tangent_px"),dtype=np.float64).reshape(2)
        anchor_b=np.asarray(_fix53_px_to_board(H_value,anchor_px),dtype=np.float64).reshape(2)
        p0=np.asarray(_fix53_px_to_board(H_value,anchor_px-tangent_px*16.0),dtype=np.float64).reshape(2)
        p1=np.asarray(_fix53_px_to_board(H_value,anchor_px+tangent_px*16.0),dtype=np.float64).reshape(2)
        exact_t=p1-p0; exact_t/=max(1e-9,float(np.linalg.norm(exact_t)))
        a,b=[np.asarray(q,dtype=np.float64).reshape(2) for q in fold_ctx["component_endpoints_board"]]
        v=b-a; vv=float(np.dot(v,v)); tau=float(np.clip(np.dot(anchor_b-a,v)/max(vv,1e-9),0.0,1.0)); nearest=a+tau*v
        dist=float(np.linalg.norm(anchor_b-nearest))
        ref_t=np.asarray(fold_ctx["tangent_board"],dtype=np.float64).reshape(2); ref_t/=max(1e-9,float(np.linalg.norm(ref_t)))
        align=float(abs(np.dot(exact_t,ref_t)))
        max_dist=max(10.0,float(getattr(args,"micro_unfold_v48_fast_same_cca_anchor_mm",30.0)))
        min_cos=float(np.clip(getattr(args,"micro_unfold_v48_fast_same_cca_axis_min_cos",0.75),0.0,1.0))
        ok=bool(dist<=max_dist+1e-6 and align>=min_cos-1e-6)
        return ok,{"used":True,"reason":"OK" if ok else "FAST_CREASE_NOT_LOCAL_FOLD_MATCH","anchor_distance_mm":dist,
                  "axis_abs_cos":align,"max_anchor_distance_mm":max_dist,"min_axis_abs_cos":min_cos}
    except Exception as exc:
        return False,{"used":True,"reason":f"FAST_LOCAL_FOLD_MATCH_EXCEPTION:{exc!r}"}


def _v38_mask_geometry(mask_u8, H_value):
    mask = ((np.asarray(mask_u8, dtype=np.uint8) > 0).astype(np.uint8) * 255)
    if mask.ndim != 2 or cv2.countNonZero(mask) < 500:
        raise RuntimeError("TARGET_DIRECT_BAD_GARMENT_MASK")
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        raise RuntimeError("TARGET_DIRECT_OUTER_CONTOUR_MISSING")
    contour = max(contours, key=cv2.contourArea).reshape(-1, 2).astype(np.float64)
    if contour.shape[0] < 8:
        raise RuntimeError("TARGET_DIRECT_OUTER_CONTOUR_TOO_SMALL")
    outer_contour_board = _fix12_transform_points(H_value, contour.astype(np.float32))
    # ACTIVE/HOLD interior depth must respect every real mask boundary (including
    # concave/internal cutouts), while pull direction remains tied to the true outer silhouette.
    all_contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    boundary_px = np.concatenate([c.reshape(-1,2) for c in all_contours if len(c) > 0], axis=0).astype(np.float32) if all_contours else contour.astype(np.float32)
    boundary_board = _fix12_transform_points(H_value, boundary_px)
    dt = cv2.distanceTransform(mask, cv2.DIST_L2, 5).astype(np.float32)
    mm = cv2.moments(mask, binaryImage=True)
    if abs(float(mm.get("m00", 0.0))) <= 1e-6:
        raise RuntimeError("TARGET_DIRECT_GARMENT_CENTROID_MISSING")
    centroid_px = np.asarray([float(mm["m10"] / mm["m00"]), float(mm["m01"] / mm["m00"])], dtype=np.float64)
    centroid_board = _fix53_px_to_board(H_value, centroid_px)
    return mask, contour, boundary_board, outer_contour_board, dt, centroid_px, centroid_board


def _v38_point_reachable_by_any_arm(config, board_xy, max_radius=420.0):
    ok = False
    best = float("inf")
    for arm in ("arm1", "arm2"):
        try:
            r = float(np.linalg.norm(_fix6_local_xy(config, arm, board_xy)))
            best = min(best, r)
            if r <= float(max_radius):
                ok = True
        except Exception:
            continue
    return bool(ok), float(best)


def _v38_pair_has_reach_assignment(config, active_board, hold_board, max_radius=420.0):
    for active_arm, hold_arm in (("arm1", "arm2"), ("arm2", "arm1")):
        try:
            ar = float(np.linalg.norm(_fix6_local_xy(config, active_arm, active_board)))
            hr = float(np.linalg.norm(_fix6_local_xy(config, hold_arm, hold_board)))
            if max(ar, hr) <= float(max_radius):
                return True
        except Exception:
            pass
    return False


def _v38_target_local_active_candidates(mask, dt, boundary_board, outer_contour_board, H_value, config, cfg, target_board, args):
    target = np.asarray(target_board, dtype=np.float64).reshape(2)
    search_mm = float(np.clip(getattr(args, "micro_unfold_v38_active_search_radius_mm", 40.0), 20.0, 40.0))
    desired_depth = float(np.clip(getattr(args, "micro_unfold_v38_active_boundary_target_mm", 25.0), 15.0, 35.0))
    target_px = _fix89_board_to_px(H_value, target)
    # Convert the board-mm local bound only to an image ROI.  The final admission is
    # always checked again in true board-mm after Homography transformation.
    px_r = 80.0
    try:
        px_dx = _fix89_board_to_px(H_value, target + np.asarray([search_mm, 0.0]))
        px_dy = _fix89_board_to_px(H_value, target + np.asarray([0.0, search_mm]))
        px_r = float(np.clip(1.35 * max(np.linalg.norm(px_dx-target_px), np.linalg.norm(px_dy-target_px)) + 8.0, 35.0, 220.0))
    except Exception:
        pass
    h, w = mask.shape
    x0=max(0,int(math.floor(float(target_px[0]-px_r)))); x1=min(w,int(math.ceil(float(target_px[0]+px_r)))+1)
    y0=max(0,int(math.floor(float(target_px[1]-px_r)))); y1=min(h,int(math.ceil(float(target_px[1]+px_r)))+1)
    yy, xx = np.where(mask[y0:y1, x0:x1] > 0)
    if xx.size <= 0:
        return [], 0
    pts = np.column_stack([xx+x0, yy+y0]).astype(np.float32)
    board = _fix12_transform_points(H_value, pts)
    d_target = np.linalg.norm(board-target[None,:], axis=1)
    inside_board = np.asarray([_fix53_board_inside(cfg, q, margin=25.0) for q in board], dtype=bool)
    keep = np.isfinite(d_target) & (d_target <= search_mm + 1e-6) & inside_board
    if not np.any(keep):
        return [], 0
    pts=pts[keep]; board=board[keep]; d_target=d_target[keep]
    xi=np.clip(np.rint(pts[:,0]).astype(np.int32),0,w-1); yi=np.clip(np.rint(pts[:,1]).astype(np.int32),0,h-1)
    depths_px=dt[yi,xi].astype(np.float64)
    raw_count=int(len(pts))
    # Bound exact contour-distance work while retaining the nearest point that is
    # likely to satisfy the requested ~25mm interior depth.  The actual decision
    # below still uses exact board-mm distance to the transformed outer contour.
    n_each=min(96,len(pts))
    idx_near=np.argsort(d_target)[:n_each]
    idx_deep=np.argsort(-depths_px)[:n_each]
    px_per_mm=1.0
    try:
        p10x=_fix89_board_to_px(H_value,target+np.asarray([10.0,0.0]))
        p10y=_fix89_board_to_px(H_value,target+np.asarray([0.0,10.0]))
        px_per_mm=max(0.25,min(float(np.linalg.norm(p10x-target_px))/10.0,float(np.linalg.norm(p10y-target_px))/10.0))
    except Exception:
        pass
    desired_depth_px=desired_depth*px_per_mm
    prelim=d_target+4.0*np.maximum(0.0,desired_depth_px-depths_px)/max(px_per_mm,1e-6)
    idx_inset=np.argsort(prelim)[:min(128,len(pts))]
    # Keep a small spatially diverse sample as well. This prevents perspective/axis
    # anisotropy in pixel distance from skipping the nearest true board-mm interior point.
    idx_uniform=np.linspace(0,len(pts)-1,min(256,len(pts)),dtype=np.int64)
    idx=np.unique(np.concatenate([idx_near,idx_deep,idx_inset,idx_uniform]))
    out=[]
    for i in idx.tolist():
        q=board[i]
        reachable, best_r = _v38_point_reachable_by_any_arm(config,q)
        if not reachable:
            continue
        bdist, _ = _v38_min_boundary_distance_mm(q, boundary_board)
        _, nearest_boundary = _v38_min_boundary_distance_mm(q, outer_contour_board)
        if not np.isfinite(bdist):
            continue
        out.append({
            "px":np.asarray(pts[i],dtype=np.float64), "board":np.asarray(q,dtype=np.float64),
            "target_distance_mm":float(d_target[i]), "boundary_distance_mm":float(bdist),
            "nearest_boundary_board":None if nearest_boundary is None else np.asarray(nearest_boundary,dtype=np.float64),
            "best_reach_radius_mm":float(best_r),
        })
    # Preserve the semantic Target itself when it is already a safe garment-interior point.
    try:
        tx,ty=int(round(float(target_px[0]))),int(round(float(target_px[1])))
        if 0<=tx<w and 0<=ty<h and mask[ty,tx]>0 and _fix53_board_inside(cfg,target,margin=25.0):
            reachable,best_r=_v38_point_reachable_by_any_arm(config,target)
            if reachable:
                bdist,_=_v38_min_boundary_distance_mm(target,boundary_board)
                _,nearest_boundary=_v38_min_boundary_distance_mm(target,outer_contour_board)
                out.append({"px":np.asarray(target_px,dtype=np.float64),"board":target.copy(),
                            "target_distance_mm":0.0,"boundary_distance_mm":float(bdist),
                            "nearest_boundary_board":None if nearest_boundary is None else np.asarray(nearest_boundary,dtype=np.float64),
                            "best_reach_radius_mm":float(best_r)})
    except Exception:
        pass
    # First preference: minimum Target correction that reaches the desired interior depth.
    def rank(c):
        short=max(0.0,desired_depth-float(c["boundary_distance_mm"]))
        tier=0 if short<=1e-6 else 1
        return (tier, float(c["target_distance_mm"])+4.0*short, -float(c["boundary_distance_mm"]), float(c["best_reach_radius_mm"]))
    out.sort(key=rank)
    dedup=[]
    for c in out:
        if any(float(np.linalg.norm(c["board"]-d["board"]))<3.0 for d in dedup):
            continue
        dedup.append(c)
    max_c=max(1,min(16,int(getattr(args,"micro_unfold_v38_active_max_candidates",8))))
    return dedup[:max_c], raw_count


def _v38_hold_candidates(mask, dt, boundary_board, H_value, config, cfg, active, outward_u, args):
    active_board=np.asarray(active["board"],dtype=np.float64).reshape(2)
    pair_min=float(np.clip(getattr(args,"micro_unfold_v38_hold_pair_min_mm",66.0),60.0,90.0))
    pair_max=float(np.clip(getattr(args,"micro_unfold_v38_hold_pair_max_mm",126.0),100.0,140.0))
    pair_target=float(np.clip(getattr(args,"micro_unfold_v38_hold_pair_target_mm",90.0),pair_min,pair_max))
    desired_depth=float(np.clip(getattr(args,"micro_unfold_v38_hold_boundary_target_mm",20.0),10.0,35.0))
    # HOLD search is bounded around ACTIVE, not a garment-wide exhaustive search.
    apx=np.asarray(active.get("px"),dtype=np.float64).reshape(2)
    px_r=220.0
    try:
        px_dx=_fix89_board_to_px(H_value,active_board+np.asarray([pair_max,0.0]))
        px_dy=_fix89_board_to_px(H_value,active_board+np.asarray([0.0,pair_max]))
        px_r=float(np.clip(1.20*max(np.linalg.norm(px_dx-apx),np.linalg.norm(px_dy-apx))+10.0,80.0,360.0))
    except Exception:
        pass
    mh,mw=mask.shape
    x0=max(0,int(math.floor(float(apx[0]-px_r)))); x1=min(mw,int(math.ceil(float(apx[0]+px_r)))+1)
    y0=max(0,int(math.floor(float(apx[1]-px_r)))); y1=min(mh,int(math.ceil(float(apx[1]+px_r)))+1)
    yy,xx=np.where(mask[y0:y1,x0:x1]>0)
    if xx.size<=0:
        return [],0
    xx=xx+x0; yy=yy+y0
    max_cloud=8000
    if xx.size>max_cloud:
        ids=np.linspace(0,xx.size-1,max_cloud,dtype=np.int64)
        xx=xx[ids]; yy=yy[ids]
    pts=np.column_stack([xx,yy]).astype(np.float32)
    board=_fix12_transform_points(H_value,pts)
    vec=board-active_board[None,:]
    sep=np.linalg.norm(vec,axis=1)
    inside_board=np.asarray([_fix53_board_inside(cfg,q,margin=25.0) for q in board],dtype=bool)
    keep=np.isfinite(sep)&(sep>=pair_min-1e-6)&(sep<=pair_max+1e-6)&inside_board
    if not np.any(keep):
        return [],0
    pts=pts[keep]; board=board[keep]; vec=vec[keep]; sep=sep[keep]
    unit=vec/np.maximum(sep[:,None],1e-9)
    inward=-np.asarray(outward_u,dtype=np.float64).reshape(2)
    align=np.sum(unit*inward[None,:],axis=1)
    h,w=mask.shape
    xi=np.clip(np.rint(pts[:,0]).astype(np.int32),0,w-1); yi=np.clip(np.rint(pts[:,1]).astype(np.int32),0,h-1)
    depths_px=dt[yi,xi].astype(np.float64)
    # Cheap rank first, exact board-mm contour depth only for the best bounded pool.
    cheap=np.abs(sep-pair_target)-28.0*align-0.8*depths_px
    take=np.argsort(cheap)[:min(160,len(cheap))]
    raw_count=int(len(pts))
    out=[]
    for i in take.tolist():
        q=board[i]
        if not _v38_pair_has_reach_assignment(config,active_board,q):
            continue
        bdist,_=_v38_min_boundary_distance_mm(q,boundary_board)
        short=max(0.0,desired_depth-bdist)
        score=(abs(float(sep[i])-pair_target)+34.0*(1.0-float(align[i]))+3.0*short)
        out.append({"px":np.asarray(pts[i],dtype=np.float64),"board":np.asarray(q,dtype=np.float64),
                    "pair_sep_mm":float(sep[i]),"inward_alignment":float(align[i]),
                    "boundary_distance_mm":float(bdist),"score":float(score)})
    out.sort(key=lambda c:(float(c["score"]),-float(c["boundary_distance_mm"])))
    dedup=[]
    for c in out:
        if any(float(np.linalg.norm(c["board"]-d["board"]))<5.0 for d in dedup):
            continue
        dedup.append(c)
    max_c=max(2,min(32,int(getattr(args,"micro_unfold_v38_hold_max_candidates",18))))
    return dedup[:max_c],raw_count



def _v39_board_points_to_arm_local(config, arm_key, board_points):
    """Vectorized board->RoArm XY for bounded v39 recovery ranking only."""
    pts=np.asarray(board_points,dtype=np.float64).reshape(-1,2)
    arm_cfg=dict(config.get("dual_roarm",{}).get(str(arm_key),{}) or {})
    M_raw=arm_cfg.get("board_to_roarm_affine_2x3")
    if M_raw is None:
        raise RuntimeError(f"{arm_key} affine unavailable")
    M=np.asarray(M_raw,dtype=np.float64).reshape(2,3)
    off=arm_cfg.get("roarm_xy_offset",[0.0,0.0])
    ox=float(off[0]) if isinstance(off,(list,tuple)) and len(off)>=1 else 0.0
    oy=float(off[1]) if isinstance(off,(list,tuple)) and len(off)>=2 else 0.0
    hom=np.column_stack([pts,np.ones((len(pts),1),dtype=np.float64)])
    out=hom@M.T
    out[:,0]+=ox; out[:,1]+=oy
    return out


def _v39_point_to_segment_distance_many(points, seg_a, seg_b):
    """Match v38 finalizer's 61-sample active-approach clearance test, vectorized."""
    pts=np.asarray(points,dtype=np.float64).reshape(-1,2)
    a=np.asarray(seg_a,dtype=np.float64).reshape(2); b=np.asarray(seg_b,dtype=np.float64).reshape(2)
    ts=np.linspace(0.0,1.0,61,dtype=np.float64)
    path=a[None,:]+ts[:,None]*(b-a)[None,:]
    # Bounded cloud <= 8k, so this exact-to-v38 sampled test stays small (~8 MB).
    d=np.linalg.norm(pts[:,None,:]-path[None,:,:],axis=2)
    return np.min(d,axis=1)


def _v39_reach_aware_hold_candidates(mask, dt, boundary_board, H_value, config, cfg, active, outward_u, args):
    """Bounded recovery for v38 NO_SAFE_HOLD / arm reach-path misses.

    Search bounds and all hard safety remain identical to v38.  The difference is
    ranking: candidates are cheaply evaluated against BOTH concrete arm assignments
    before the shortlist is truncated, so a farther-but-still-legal 110..126 mm HOLD
    is not discarded merely because 90 mm candidates rank better semantically.
    """
    if not bool(getattr(args,"micro_unfold_v39_reach_hold_recovery_enable",True)):
        return [],{"raw_candidate_count":0,"feasible_candidate_count":0,"reason":"RECOVERY_DISABLED"}
    active_board=np.asarray(active["board"],dtype=np.float64).reshape(2)
    apx=np.asarray(active.get("px"),dtype=np.float64).reshape(2)
    pair_min=float(np.clip(getattr(args,"micro_unfold_v38_hold_pair_min_mm",66.0),60.0,90.0))
    pair_max=float(np.clip(getattr(args,"micro_unfold_v38_hold_pair_max_mm",126.0),100.0,140.0))
    pair_target=float(np.clip(getattr(args,"micro_unfold_v38_hold_pair_target_mm",90.0),pair_min,pair_max))
    desired_depth=float(np.clip(getattr(args,"micro_unfold_v38_hold_boundary_target_mm",20.0),10.0,35.0))
    mh,mw=mask.shape
    px_r=220.0
    try:
        px_dx=_fix89_board_to_px(H_value,active_board+np.asarray([pair_max,0.0]))
        px_dy=_fix89_board_to_px(H_value,active_board+np.asarray([0.0,pair_max]))
        px_r=float(np.clip(1.20*max(np.linalg.norm(px_dx-apx),np.linalg.norm(px_dy-apx))+10.0,80.0,360.0))
    except Exception:
        pass
    x0=max(0,int(math.floor(float(apx[0]-px_r)))); x1=min(mw,int(math.ceil(float(apx[0]+px_r)))+1)
    y0=max(0,int(math.floor(float(apx[1]-px_r)))); y1=min(mh,int(math.ceil(float(apx[1]+px_r)))+1)
    yy,xx=np.where(mask[y0:y1,x0:x1]>0)
    if xx.size<=0:
        return [],{"raw_candidate_count":0,"feasible_candidate_count":0,"reason":"NO_MASK_PIXELS"}
    xx=xx+x0; yy=yy+y0
    cloud_max=max(1000,min(12000,int(getattr(args,"micro_unfold_v39_recovery_cloud_max",8000))))
    if xx.size>cloud_max:
        ids=np.linspace(0,xx.size-1,cloud_max,dtype=np.int64); xx=xx[ids]; yy=yy[ids]
    pts=np.column_stack([xx,yy]).astype(np.float32)
    board=_fix12_transform_points(H_value,pts)
    vec=board-active_board[None,:]; sep=np.linalg.norm(vec,axis=1)
    inside=np.asarray([_fix53_board_inside(cfg,q,margin=25.0) for q in board],dtype=bool)
    keep=np.isfinite(sep)&(sep>=pair_min-1e-6)&(sep<=pair_max+1e-6)&inside
    if not np.any(keep):
        return [],{"raw_candidate_count":0,"feasible_candidate_count":0,"reason":"NO_PAIR_RANGE_CANDIDATES"}
    pts=pts[keep]; board=board[keep]; vec=vec[keep]; sep=sep[keep]
    raw_count=int(len(board))
    unit=vec/np.maximum(sep[:,None],1e-9)
    inward=-np.asarray(outward_u,dtype=np.float64).reshape(2)
    align=np.sum(unit*inward[None,:],axis=1)
    xi=np.clip(np.rint(pts[:,0]).astype(np.int32),0,mw-1); yi=np.clip(np.rint(pts[:,1]).astype(np.int32),0,mh-1)
    depths_px=np.asarray(dt[yi,xi],dtype=np.float64)

    feasible=np.zeros(len(board),dtype=bool)
    best_reach=np.full(len(board),np.inf,dtype=np.float64)
    best_path=np.full(len(board),-np.inf,dtype=np.float64)
    best_assignment=np.full(len(board),"",dtype=object)
    for active_arm,hold_arm in (("arm1","arm2"),("arm2","arm1")):
        try:
            a_local=_fix6_local_xy(config,active_arm,active_board)
            a_radius=float(np.linalg.norm(a_local))
            if a_radius>420.0:
                continue
            h_local=_v39_board_points_to_arm_local(config,hold_arm,board)
            h_radius=np.linalg.norm(h_local,axis=1)
            a_standby=np.asarray(standby_roarm_pose(cfg,active_arm)[:2],dtype=np.float64)
            a_standby_board=_fix37_local_xy_to_board(config,active_arm,a_standby)
            path_clear=_v39_point_to_segment_distance_many(board,a_standby_board,active_board)
            ok=(h_radius<=420.0)&(path_clear>=65.0)
            reach=np.maximum(a_radius,h_radius)
            improve=ok&((~feasible)|(reach<best_reach-1e-6)|((np.abs(reach-best_reach)<=1e-6)&(path_clear>best_path)))
            feasible|=ok
            best_reach[improve]=reach[improve]; best_path[improve]=path_clear[improve]
            best_assignment[improve]=f"{active_arm}->{hold_arm}"
        except Exception:
            continue
    feasible_count=int(np.count_nonzero(feasible))
    if feasible_count<=0:
        return [],{"raw_candidate_count":raw_count,"feasible_candidate_count":0,"reason":"NO_REACH_PATH_FEASIBLE_HOLD"}
    ids=np.where(feasible)[0]
    # Recovery ranking is intentionally reach/path-first. Pair-target and inward
    # alignment are only tie-breakers after the hard-feasible set is established.
    order=sorted(ids.tolist(),key=lambda i:(float(best_reach[i]),-float(best_path[i]),abs(float(sep[i])-pair_target),-float(align[i]),-float(depths_px[i])))
    pre=max(32,min(640,int(getattr(args,"micro_unfold_v39_recovery_preselect_max",320))))
    order=order[:pre]
    out=[]
    for i in order:
        q=board[i]
        bdist,_=_v38_min_boundary_distance_mm(q,boundary_board)
        if not np.isfinite(bdist):
            continue
        short=max(0.0,desired_depth-float(bdist))
        # Boundary depth still matters, but cannot hide a physically superior arm assignment.
        score=(0.30*float(best_reach[i])-0.20*float(best_path[i])+0.35*abs(float(sep[i])-pair_target)
               +12.0*(1.0-float(align[i]))+2.5*short)
        out.append({"px":np.asarray(pts[i],dtype=np.float64),"board":np.asarray(q,dtype=np.float64),
                    "pair_sep_mm":float(sep[i]),"inward_alignment":float(align[i]),
                    "boundary_distance_mm":float(bdist),"score":float(score),
                    "v39_recovery":True,"v39_recovery_best_reach_mm":float(best_reach[i]),
                    "v39_recovery_path_clearance_mm":float(best_path[i]),
                    "v39_recovery_assignment_preview":str(best_assignment[i])})
    out.sort(key=lambda c:(float(c["score"]),float(c["v39_recovery_best_reach_mm"]),-float(c["v39_recovery_path_clearance_mm"])))
    dedup=[]
    for c in out:
        if any(float(np.linalg.norm(c["board"]-d["board"]))<5.0 for d in dedup):
            continue
        dedup.append(c)
    max_c=max(4,min(40,int(getattr(args,"micro_unfold_v39_recovery_hold_max_candidates",24))))
    return dedup[:max_c],{"raw_candidate_count":raw_count,"feasible_candidate_count":feasible_count,
                          "shortlist_count":int(min(len(dedup),max_c)),"reason":"OK"}


def _v42_prepare_target_direct_cache(mask, H_value, outer_contour_px=None):
    """One same-snapshot cache shared by normal/v39/v40 Target-direct trials.

    It stores only deterministic geometry and bounded recovery memoization.  No
    safety threshold or planner semantics are changed.
    """
    c={"v42_pull_recovery_cache":{},"v42_ray_exit_cache":{},
       "v42_stats":{"pull_recovery_cache_hits":0,"pull_recovery_cheap_rejects":0,
                    "ray_exit_cache_hits":0,"pair_duplicate_skips":0}}
    try:
        mask_bool=(np.asarray(mask,dtype=np.uint8)>0)
        c["stable_mask"]=cv2.dilate(mask_bool.astype(np.uint8)*255,np.ones((9,9),np.uint8),iterations=1)>0
        if outer_contour_px is not None:
            arr=np.asarray(outer_contour_px,dtype=np.float64).reshape(-1,2)
            if len(arr)>0:
                c["outer_contour"]=arr
        if c.get("outer_contour") is None:
            contours,_=cv2.findContours(mask_bool.astype(np.uint8)*255,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
            if contours:
                c["outer_contour"]=max(contours,key=cv2.contourArea).reshape(-1,2).astype(np.float64)
        c["invH"]=_e156_inverse_h(H_value)
    except Exception:
        # Keep memoization even if one invariant geometry item cannot be prepared.
        pass
    return c


def _v42_trial_pair_key(active, hold):
    try:
        a=np.asarray(active["board"],dtype=np.float64).reshape(2)
        h=np.asarray(hold["board"],dtype=np.float64).reshape(2)
        return tuple(round(float(x),4) for x in (a[0],a[1],h[0],h[1]))
    except Exception:
        return None


def _v41_rotate_unit(v, deg):
    a=np.asarray(v,dtype=np.float64).reshape(2)
    n=float(np.linalg.norm(a))
    if n<=1e-9:
        return None
    a=a/n; th=math.radians(float(deg)); c=math.cos(th); ss=math.sin(th)
    return np.asarray([c*a[0]-ss*a[1],ss*a[0]+c*a[1]],dtype=np.float64)


def _v42_pull_reach_endpoint_recovery(mask, H_value, config, cfg, active_board, hold_board,
                                       base_pull_u, base_outside_margin_mm, active_arm, args,
                                       cache=None):
    """Failure-only PULL endpoint recovery with v42 bounded memoization/pruning.

    Search coverage is the same small angular fan/backoff family as v41.  v42 only
    avoids exact duplicate work, reuses ray exits, and skips a direction when the
    arm-affine radius is already >420mm at the minimum tested pull and is moving
    monotonically farther away.  Hard reach/board/garment-exit rules remain exact.
    """
    diag={"used":False,"trial_count":0,"reason":"DISABLED","cache_hit":False,
          "cheap_reject_count":0,"ray_exit_cache_hits":0}
    if not bool(getattr(args,"micro_unfold_v41_pull_direction_recovery_enable",True)):
        return None,diag
    diag["used"]=True
    mask_bool=np.asarray(mask,dtype=np.uint8)>0; h,w=mask_bool.shape[:2]
    c=cache if isinstance(cache,dict) else {}
    try:
        stable=np.asarray(c.get("stable_mask"),dtype=bool) if c.get("stable_mask") is not None else cv2.dilate(mask_bool.astype(np.uint8)*255,np.ones((9,9),np.uint8),iterations=1)>0
        invH=np.asarray(c.get("invH"),dtype=np.float64) if c.get("invH") is not None else _e156_inverse_h(H_value)
    except Exception:
        stable=cv2.dilate(mask_bool.astype(np.uint8)*255,np.ones((9,9),np.uint8),iterations=1)>0
        invH=_e156_inverse_h(H_value)
    active=np.asarray(active_board,dtype=np.float64).reshape(2); hold=np.asarray(hold_board,dtype=np.float64).reshape(2)
    base=np.asarray(base_pull_u,dtype=np.float64).reshape(2); base/=max(1e-9,float(np.linalg.norm(base)))
    step=float(np.clip(float(getattr(args,'micro_unfold_boundary_sample_mm',2.0)),1.0,5.0))
    max_pull=max(60.0,float(getattr(args,'micro_unfold_max_pull_mm',264.0)))
    min_pull=max(20.0,float(getattr(args,'micro_unfold_pull_mm',30.0)))
    length_factor=float(np.clip(float(getattr(args,'micro_unfold_hold_boundary_factor',2.0)),1.2,2.5))
    outside_margin=max(8.0,min(15.0,float(base_outside_margin_mm)))
    search_limit=max(360.0,max_pull+100.0)
    fan=list(getattr(args,"micro_unfold_v41_pull_direction_fan_deg",(-12.0,-8.0,-4.0,4.0,8.0,12.0)) or ())
    fan=[float(x) for x in fan if 0.1<=abs(float(x))<=18.0]
    max_trials=max(12,min(72,int(getattr(args,"micro_unfold_v41_pull_direction_max_trials",48))))

    # Exact same pair/arm/direction basis can reappear across normal/v39/v40 passes.
    memo=c.setdefault("v42_pull_recovery_cache",{}) if bool(getattr(args,"micro_unfold_v42_pull_recovery_cache_enable",True)) else {}
    memo_key=(str(active_arm),)+tuple(round(float(x),4) for x in (active[0],active[1],hold[0],hold[1],base[0],base[1],outside_margin,min_pull,max_pull))
    if memo_key in memo:
        stats=c.setdefault("v42_stats",{})
        stats["pull_recovery_cache_hits"]=int(stats.get("pull_recovery_cache_hits",0))+1
        rec,old_diag=memo[memo_key]
        out_diag=copy.deepcopy(old_diag); out_diag["cache_hit"]=True
        return (None if rec is None else copy.deepcopy(rec)),out_diag

    ray_cache=c.setdefault("v42_ray_exit_cache",{})
    stats=c.setdefault("v42_stats",{})
    def ray_exit(origin,u):
        key=tuple(round(float(x),5) for x in (origin[0],origin[1],u[0],u[1],search_limit,step))
        if key in ray_cache:
            diag["ray_exit_cache_hits"]+=1
            stats["ray_exit_cache_hits"]=int(stats.get("ray_exit_cache_hits",0))+1
            return ray_cache[key]
        out=_e130_ray_exit(stable.astype(np.uint8)*255,H_value,origin,u,max_mm=search_limit,step_mm=step,stable_mask=stable,invH=invH)
        ray_cache[key]=out
        return out

    # Good candidates first: predict arm radius at a short probe distance.  This
    # changes only evaluation order, not admission/safety criteria.
    ordered=[]
    for deg in fan:
        u=_v41_rotate_unit(base,deg)
        if u is None or float(np.dot(u,base))<math.cos(math.radians(18.0)):
            continue
        try:
            probe=_fix6_local_xy(config,active_arm,active+u*max(min_pull,60.0))
            predicted=float(np.linalg.norm(probe))
        except Exception:
            predicted=9999.0
        ordered.append((predicted,abs(float(deg)),float(deg),u))
    ordered.sort(key=lambda z:(z[0],z[1]))

    candidates=[]; trials=0
    for _,_,deg,u in ordered:
        # Safe cheap reject for affine board->arm geometry: if the radius is already
        # outside 420mm at the minimum distance this recovery can ever test and its
        # local radial derivative is non-negative, every longer point is even worse.
        if bool(getattr(args,"micro_unfold_v42_pull_cheap_reject_enable",True)):
            try:
                l1=np.asarray(_fix6_local_xy(config,active_arm,active+u*min_pull),dtype=np.float64)
                l2=np.asarray(_fix6_local_xy(config,active_arm,active+u*(min_pull+1.0)),dtype=np.float64)
                deriv=l2-l1
                if float(np.linalg.norm(l1))>420.0 and float(np.dot(l1,deriv))>=0.0:
                    diag["cheap_reject_count"]+=1
                    stats["pull_recovery_cheap_rejects"]=int(stats.get("pull_recovery_cheap_rejects",0))+1
                    continue
            except Exception:
                pass
        ae=ray_exit(active,u); he=ray_exit(hold,u)
        if ae is None or he is None:
            continue
        active_exit=float(ae[0]); hold_exit=float(he[0])
        minimum_outside=active_exit+outside_margin
        desired=min(max_pull,max(min_pull,length_factor*hold_exit,minimum_outside))
        if desired+1e-6<minimum_outside:
            continue
        lengths=[]; q=float(desired)
        while q>=minimum_outside-1e-6 and len(lengths)<12:
            lengths.append(q); q-=10.0
        if not lengths or lengths[-1]>minimum_outside+1e-6:
            lengths.append(float(minimum_outside))
        for dist in lengths:
            if trials>=max_trials: break
            trials+=1
            bp=active+u*float(dist)
            if not _fix53_board_inside(cfg,bp,margin=20.0):
                continue
            try:
                px=_fix89_board_to_px(H_value,bp); xi,yi=int(round(float(px[0]))),int(round(float(px[1])))
                outside=not (0<=xi<w and 0<=yi<h and stable[yi,xi])
                local=_fix6_local_xy(config,active_arm,bp); radius=float(np.linalg.norm(local))
            except Exception:
                continue
            if (not outside) or radius>420.0:
                continue
            backoff=max(0.0,float(desired)-float(dist))
            score=2.0*abs(float(deg))+0.12*backoff+0.20*max(0.0,radius-380.0)
            candidates.append((score,radius,abs(float(deg)),backoff,{"pull_board":bp,"pull_px":np.asarray(px,dtype=np.float64),"pull_u":u,"pull_distance_mm":float(dist),"active_exit_mm":active_exit,"hold_exit_mm":hold_exit,"outside_margin_mm":float(dist-active_exit),"direction_delta_deg":float(deg),"pull_local_xy":np.asarray(local,dtype=np.float64)}))
        if trials>=max_trials: break
    diag["trial_count"]=int(trials)
    if not candidates:
        diag["reason"]="NO_REACHABLE_OUTWARD_FAN_ENDPOINT"
        if memo is not None:
            memo[memo_key]=(None,copy.deepcopy(diag))
        return None,diag
    candidates.sort(key=lambda z:(z[0],z[1],z[2],z[3]))
    out=candidates[0][4]; diag.update({"reason":"OK","selected_direction_delta_deg":float(out["direction_delta_deg"]),"selected_pull_radius_mm":float(candidates[0][1]),"selected_backoff_mm":float(candidates[0][3])})
    if memo is not None:
        memo[memo_key]=(copy.deepcopy(out),copy.deepcopy(diag))
    return out,diag


def _v38_finalize_target_direct_pair(mask, H_value, config, cfg, target_board, active, hold, outward_u, args,
                                     _v40_e111_cache=None, _v40_log_pull=True):
    active_board=np.asarray(active["board"],dtype=np.float64).reshape(2)
    hold_board=np.asarray(hold["board"],dtype=np.float64).reshape(2)
    pair_sep=float(np.linalg.norm(active_board-hold_board))
    pair_min=float(getattr(args,"micro_unfold_v38_hold_pair_min_mm",66.0)); pair_max=float(getattr(args,"micro_unfold_v38_hold_pair_max_mm",126.0))
    if not (pair_min-1e-6<=pair_sep<=pair_max+1e-6):
        raise RuntimeError(f"TARGET_DIRECT_PAIR_SPACING_FAIL:{pair_sep:.1f}mm")
    if not (_fix53_board_inside(cfg,active_board,margin=25.0) and _fix53_board_inside(cfg,hold_board,margin=25.0)):
        raise RuntimeError("TARGET_DIRECT_BOARD_FAIL")
    hint=np.asarray(outward_u,dtype=np.float64).reshape(2)
    hn=float(np.linalg.norm(hint))
    if hn<=1e-9:
        raise RuntimeError("TARGET_DIRECT_PULL_ENDPOINT_FAIL:PULL_HINT_COLLAPSED")
    hint/=hn
    try:
        _cache=dict(_v40_e111_cache or {})
        (pull_board,used_pull,garment_exit_mm,outside_margin_used,pull_endpoint_px,pull_u,
         hold_boundary_distance_mm,boundary_normal_delta_deg,boundary_normal_ok)=_e111_find_outer_normal_pull_endpoint(
            mask,H_value,cfg,active_board,hold_board,hint,args,log_final=bool(_v40_log_pull),
            stable_mask_cached=_cache.get("stable_mask"), outer_contour_cached=_cache.get("outer_contour"),
            invH_cached=_cache.get("invH"))
    except Exception as exc:
        raise RuntimeError(f"TARGET_DIRECT_PULL_ENDPOINT_FAIL:{exc}") from exc

    assignment_candidates=[]
    fail_counts={"ACTIVE_REACH_FAIL":0,"HOLD_REACH_FAIL":0,"PULL_REACH_FAIL":0,
                 "PATH_CLEARANCE_FAIL":0,"ASSIGNMENT_EXCEPTION":0}
    for active_arm,hold_arm in (("arm1","arm2"),("arm2","arm1")):
        try:
            a_local=_fix6_local_xy(config,active_arm,active_board)
            h_local=_fix6_local_xy(config,hold_arm,hold_board)
            a_radius=float(np.linalg.norm(a_local)); h_radius=float(np.linalg.norm(h_local))
            if a_radius>420.0:
                fail_counts["ACTIVE_REACH_FAIL"]+=1
                continue
            if h_radius>420.0:
                fail_counts["HOLD_REACH_FAIL"]+=1
                continue

            trial_pull_board=np.asarray(pull_board,dtype=np.float64).copy(); trial_used_pull=float(used_pull)
            trial_pull_px=np.asarray(pull_endpoint_px,dtype=np.float64).copy(); p_local=_fix6_local_xy(config,active_arm,trial_pull_board)
            trial_pull_u=np.asarray(pull_u,dtype=np.float64).copy()
            trial_garment_exit=float(garment_exit_mm); trial_hold_boundary=float(hold_boundary_distance_mm)
            trial_boundary_delta=float(boundary_normal_delta_deg); trial_boundary_ok=bool(boundary_normal_ok)
            v41_pull_recovery_used=False; v41_pull_recovery_diag={"trial_count":0}
            soft_pull_radius=float(np.clip(getattr(args,"micro_unfold_pull_soft_radius_mm",390.0),340.0,420.0))
            min_outside_dist=float(garment_exit_mm)+max(8.0,min(15.0,float(outside_margin_used)))
            if float(np.linalg.norm(p_local))>soft_pull_radius:
                while trial_used_pull-5.0>=min_outside_dist-1e-6:
                    trial_used_pull-=5.0
                    q=active_board+np.asarray(pull_u,dtype=np.float64)*trial_used_pull
                    if not _fix53_board_inside(cfg,q,margin=20.0):
                        continue
                    try:
                        qpx=_fix89_board_to_px(H_value,q); qx,qy=int(round(float(qpx[0]))),int(round(float(qpx[1])))
                        outside=not (0<=qx<mask.shape[1] and 0<=qy<mask.shape[0] and mask[qy,qx]>0)
                    except Exception:
                        outside=False
                    qlocal=_fix6_local_xy(config,active_arm,q)
                    if outside and float(np.linalg.norm(qlocal))<=soft_pull_radius:
                        trial_pull_board=q; trial_pull_px=np.asarray(qpx,dtype=np.float64); p_local=qlocal; break
            p_radius=float(np.linalg.norm(p_local))
            if p_radius>420.0:
                rec,rec_diag=_v42_pull_reach_endpoint_recovery(
                    mask,H_value,config,cfg,active_board,hold_board,pull_u,outside_margin_used,active_arm,args,cache=_v40_e111_cache)
                v41_pull_recovery_diag=dict(rec_diag or {})
                if isinstance(rec,dict):
                    trial_pull_board=np.asarray(rec["pull_board"],dtype=np.float64); trial_pull_px=np.asarray(rec["pull_px"],dtype=np.float64)
                    trial_pull_u=np.asarray(rec["pull_u"],dtype=np.float64); trial_used_pull=float(rec["pull_distance_mm"])
                    trial_garment_exit=float(rec["active_exit_mm"]); trial_hold_boundary=float(rec["hold_exit_mm"])
                    trial_boundary_delta=float(boundary_normal_delta_deg)+float(rec["direction_delta_deg"]); trial_boundary_ok=True
                    p_local=np.asarray(rec["pull_local_xy"],dtype=np.float64); p_radius=float(np.linalg.norm(p_local))
                    v41_pull_recovery_used=True
                    print(f"[MICRO-TARGET-DIRECT-PULL-REACH-RECOVERY-V41] arm={active_arm.upper()} angle={float(rec['direction_delta_deg']):+.1f}deg pull={trial_used_pull:.1f}mm radius={p_radius:.1f}mm trials={int(rec_diag.get('trial_count',0))}")
                if p_radius>420.0:
                    print(f"[MICRO-TARGET-DIRECT-PULL-REACH-RECOVERY-BLOCKED-V41] arm={active_arm.upper()} radius={p_radius:.1f}mm trials={int(v41_pull_recovery_diag.get('trial_count',0) or 0)} reason={v41_pull_recovery_diag.get('reason','NO_ENDPOINT')}")
                    fail_counts["PULL_REACH_FAIL"]+=1
                    continue

            a_standby=np.asarray(standby_roarm_pose(cfg,active_arm)[:2],dtype=np.float64)
            h_standby=np.asarray(standby_roarm_pose(cfg,hold_arm)[:2],dtype=np.float64)
            travel=float(np.linalg.norm(a_local-a_standby)+np.linalg.norm(h_local-h_standby))
            a_standby_board=_fix37_local_xy_to_board(config,active_arm,a_standby)
            ts=np.linspace(0.0,1.0,61,dtype=np.float64)
            path=a_standby_board[None,:]+ts[:,None]*(active_board-a_standby_board)[None,:]
            path_clear=float(np.min(np.linalg.norm(path-hold_board[None,:],axis=1)))
            if path_clear<65.0:
                fail_counts["PATH_CLEARANCE_FAIL"]+=1
                continue
            assignment_candidates.append({"active_arm":active_arm,"hold_arm":hold_arm,
                "active_local_xy":a_local,"hold_local_xy":h_local,"pull_local_xy":p_local,
                "pull_board_effective":np.asarray(trial_pull_board,dtype=np.float64),
                "pull_endpoint_px_effective":np.asarray(trial_pull_px,dtype=np.float64),
                "pull_unit_board_effective":np.asarray(trial_pull_u,dtype=np.float64),
                "garment_exit_distance_mm_effective":float(trial_garment_exit),
                "hold_boundary_distance_mm_effective":float(trial_hold_boundary),
                "boundary_normal_delta_deg_effective":float(trial_boundary_delta),
                "boundary_normal_ok_effective":bool(trial_boundary_ok),
                "pull_distance_mm_effective":float(trial_used_pull),
                "v41_pull_direction_recovery_used":bool(v41_pull_recovery_used),
                "v41_pull_direction_delta_deg":float(v41_pull_recovery_diag.get("selected_direction_delta_deg",0.0) if v41_pull_recovery_used else 0.0),
                "v41_pull_direction_trial_count":int(v41_pull_recovery_diag.get("trial_count",0) or 0),
                "v41_pull_direction_backoff_mm":float(v41_pull_recovery_diag.get("selected_backoff_mm",0.0) if v41_pull_recovery_used else 0.0),
                "v42_pull_recovery_cache_hit":bool(v41_pull_recovery_diag.get("cache_hit",False)),
                "v42_pull_recovery_cheap_rejects":int(v41_pull_recovery_diag.get("cheap_reject_count",0) or 0),
                "pull_soft_backoff_mm":float(max(0.0,float(used_pull)-float(trial_used_pull))),
                "travel_cost_mm":travel,"active_path_hold_clearance_mm":path_clear,
                "active_radius_mm":a_radius,"hold_radius_mm":h_radius,"pull_radius_mm":p_radius})
        except Exception:
            fail_counts["ASSIGNMENT_EXCEPTION"]+=1
            continue
    if not assignment_candidates:
        # Prefer the most specific observed physical blocker.  Counts are embedded in
        # the reason so one log line is enough to distinguish the old aggregate failure.
        order=("PULL_REACH_FAIL","ACTIVE_REACH_FAIL","HOLD_REACH_FAIL","PATH_CLEARANCE_FAIL","ASSIGNMENT_EXCEPTION")
        primary=next((k for k in order if fail_counts.get(k,0)>0),"ARM_ASSIGNMENT_FAIL")
        detail=",".join(f"{k}={int(v)}" for k,v in fail_counts.items())
        raise RuntimeError(f"TARGET_DIRECT_{primary}:{detail}")
    assignment=min(assignment_candidates,key=lambda item:float(item["travel_cost_mm"]))
    active_arm=str(assignment["active_arm"]); hold_arm=str(assignment["hold_arm"])
    pull_board=np.asarray(assignment["pull_board_effective"],dtype=np.float64)
    pull_endpoint_px=np.asarray(assignment["pull_endpoint_px_effective"],dtype=np.float64)
    pull_u=np.asarray(assignment.get("pull_unit_board_effective",pull_u),dtype=np.float64)
    pull_u/=max(1e-9,float(np.linalg.norm(pull_u)))
    garment_exit_mm=float(assignment.get("garment_exit_distance_mm_effective",garment_exit_mm))
    hold_boundary_distance_mm=float(assignment.get("hold_boundary_distance_mm_effective",hold_boundary_distance_mm))
    boundary_normal_delta_deg=float(assignment.get("boundary_normal_delta_deg_effective",boundary_normal_delta_deg))
    boundary_normal_ok=bool(assignment.get("boundary_normal_ok_effective",boundary_normal_ok))
    used_pull=float(assignment["pull_distance_mm_effective"])
    outside_margin_used=max(0.0,float(used_pull)-float(garment_exit_mm))
    pull_local=np.asarray(assignment["pull_local_xy"],dtype=np.float64)
    peel_xy_request=max(0.0,float(getattr(args,"micro_unfold_peel_xy_mm",35.0)))
    peel_z_rise=max(50.0,float(getattr(args,"micro_unfold_peel_z_mm",55.0)))
    try:
        local_probe=_fix6_local_xy(config,active_arm,pull_board+np.asarray(pull_u,dtype=np.float64)*10.0)
        peel_local_u=np.asarray(local_probe-pull_local,dtype=np.float64); peel_local_u/=max(1e-9,float(np.linalg.norm(peel_local_u)))
    except Exception:
        peel_local_u=np.asarray([1.0,0.0],dtype=np.float64)
    peel_xy_used=float(peel_xy_request); peel_local_xy=pull_local.copy()
    while peel_xy_used>0.0:
        candidate_local=pull_local+peel_local_u*peel_xy_used
        if float(np.linalg.norm(candidate_local))<=float(np.clip(getattr(args,"micro_unfold_peel_soft_radius_mm",400.0),350.0,420.0)):
            peel_local_xy=candidate_local; break
        peel_xy_used-=5.0
    if peel_xy_used<=0.0:
        peel_xy_used=0.0; peel_local_xy=pull_local.copy()
    active_contact_z=float(arm_contact_z(cfg,active_arm)); hold_contact_z=float(arm_contact_z(cfg,hold_arm))
    hold_z_offset=_micro_unfold_hold_z_offset(args,hold_arm); active_z_offset=float(getattr(args,"micro_unfold_active_z_offset_mm",26.0))
    hold_action_z=hold_contact_z+hold_z_offset
    start_clearance=max(0.0,float(getattr(args,"micro_unfold_active_start_clearance_mm",18.0)))
    active_start_z=active_contact_z+start_clearance+active_z_offset
    max_below=max(0.0,float(getattr(args,"micro_unfold_max_below_contact_mm",2.0)))
    raw_floor=max(float(getattr(cfg,"min_safe_contact_z",-140.0)),active_contact_z-max_below)
    active_floor_z=raw_floor+active_z_offset
    def _angle(name,default): return float(getattr(args,name,default))
    hold_gripper=_angle("micro_unfold_hold_arm1_gripper" if hold_arm=="arm1" else "micro_unfold_hold_arm2_gripper",2.818 if hold_arm=="arm1" else 2.772)
    active_closed_ref_nominal=_angle("micro_unfold_active_arm1_closed_ref" if active_arm=="arm1" else "micro_unfold_active_arm2_closed_ref",2.900 if active_arm=="arm1" else 2.890)
    release_deg=float(getattr(args,"micro_unfold_active_release_deg",45.0))
    active_release=float(active_closed_ref_nominal-np.deg2rad(release_deg))
    max_reach=max(float(assignment["active_radius_mm"]),float(assignment["hold_radius_mm"]),float(assignment["pull_radius_mm"]))
    plan={
        "state":"READY","method":"V38_TARGET_DIRECT_E111_OUTER_NORMAL",
        "active_px":np.asarray(active["px"],dtype=np.float64).tolist(),"hold_px":np.asarray(hold["px"],dtype=np.float64).tolist(),
        "active_board":active_board.tolist(),"hold_board":hold_board.tolist(),"pull_board":pull_board.tolist(),
        "pull_unit_board":np.asarray(pull_u,dtype=np.float64).tolist(),"pull_hint_unit_board":hint.tolist(),
        "pull_distance_mm":float(used_pull),"pair_separation_mm":float(pair_sep),
        "garment_exit_distance_mm":float(garment_exit_mm),"outside_margin_used_mm":float(outside_margin_used),
        "hold_boundary_distance_mm":float(hold_boundary_distance_mm),"boundary_normal_delta_deg":float(boundary_normal_delta_deg),
        "boundary_normal_ok":bool(boundary_normal_ok),"pull_endpoint_px":pull_endpoint_px.tolist(),
        **assignment,
        "active_contact_z":active_contact_z,"hold_contact_z":hold_contact_z,
        "hold_z_offset_mm":hold_z_offset,"active_z_offset_mm":active_z_offset,
        "hold_action_z":float(hold_action_z),"active_start_z":float(active_start_z),"active_floor_z":float(active_floor_z),
        "hold_gripper":float(hold_gripper),"active_open":float(active_closed_ref_nominal),"active_close":float(active_closed_ref_nominal),
        "active_release":float(active_release),"active_gripper_stroke_deg":0.0,"active_approach_closed":float(active_closed_ref_nominal),
        "active_closed_ref_nominal":float(active_closed_ref_nominal),"peel_local_xy":np.asarray(peel_local_xy,dtype=np.float64).tolist(),
        "peel_xy_mm":float(peel_xy_used),"peel_z_mm":float(peel_z_rise),
        "execution_evidence_ok":True,"execution_evidence_reason":"V41_TARGET_DIRECT_PHYSICAL_SAFETY_OK",
        "vla_joint_localization_policy":"V38_TARGET_MODEL_AUTHORITY_NO_CREASE_REJUDGE",
        "vla_joint_role_validation":{"reason":"BYPASSED_BY_V38_TARGET_DIRECT","semantic_crease_revalidation":False},
        "v38_planner_source":"TARGET_DIRECT","v38_target_direct_attempted":True,"v38_target_direct_used":True,
        "v38_target_board":np.asarray(target_board,dtype=np.float64).tolist(),
        "v38_active_inset_mm":float(active["target_distance_mm"]),
        "v38_active_boundary_distance_mm":float(active["boundary_distance_mm"]),
        "v38_pair_separation_mm":float(pair_sep),"v38_path_clearance_mm":float(assignment["active_path_hold_clearance_mm"]),
        "v38_max_reach_radius_mm":float(max_reach),"v38_fallback_used":False,"v38_fallback_reason":"",
        "v39_reach_hold_recovery_used":bool(hold.get("v39_recovery",False)),
        "v39_recovery_trigger":str(hold.get("v39_recovery_trigger","") or ""),
        "v39_recovery_selected_assignment_preview":str(hold.get("v39_recovery_assignment_preview","") or ""),
        "v40_pull_recovery_used":bool(hold.get("v40_pull_recovery",False)),
        "v40_recovery_trigger":str(hold.get("v40_recovery_trigger","") or ""),
        "v40_selected_pull_radius_mm":float(assignment["pull_radius_mm"]),
        "v41_pull_direction_recovery_used":bool(assignment.get("v41_pull_direction_recovery_used",False)),
        "v41_pull_direction_delta_deg":float(assignment.get("v41_pull_direction_delta_deg",0.0) or 0.0),
        "v41_pull_direction_trial_count":int(assignment.get("v41_pull_direction_trial_count",0) or 0),
        "v41_pull_direction_backoff_mm":float(assignment.get("v41_pull_direction_backoff_mm",0.0) or 0.0),
        "v42_pull_recovery_cache_hits":int(bool(assignment.get("v42_pull_recovery_cache_hit",False))),
        "v42_pull_recovery_cheap_rejects":int(assignment.get("v42_pull_recovery_cheap_rejects",0) or 0),
        "v42_shared_geometry_cache":True,
        "v12_plan_score":0.0,"v12_candidate_count_ready":1,"v12_candidate_count_total":1,
        "e156_vla_core_version":E156_CORE_VERSION,
    }
    return plan



def _v40_failure_key(reason):
    text=str(reason or "UNKNOWN")
    for key in ("TARGET_DIRECT_ACTIVE_REACH_FAIL","TARGET_DIRECT_HOLD_REACH_FAIL",
                "TARGET_DIRECT_PULL_REACH_FAIL","TARGET_DIRECT_PATH_CLEARANCE_FAIL",
                "TARGET_DIRECT_PULL_ENDPOINT_FAIL","TARGET_DIRECT_PAIR_SPACING_FAIL",
                "TARGET_DIRECT_BOARD_FAIL","TARGET_DIRECT_ARM_ASSIGNMENT_FAIL"):
        if text.startswith(key):
            return key
    if "NO_SAFE_HOLD" in text:
        return "TARGET_DIRECT_NO_SAFE_HOLD"
    if "NO_SAFE_ACTIVE" in text:
        return "TARGET_DIRECT_NO_SAFE_ACTIVE"
    return text.split(":",1)[0][:96]


def _v40_diverse_exact_trials(trials, max_trials, per_active=2):
    """Bound exact E111/finalizer work while preserving ACTIVE diversity."""
    if not trials:
        return []
    limit=max(1,int(max_trials)); per=max(1,int(per_active))
    ordered=sorted(trials,key=lambda z:float(z[0]))
    chosen=[]; used=set(); counts={}
    # First pass: guarantee a small number from each ACTIVE, in global score order.
    for item in ordered:
        ai=int(item[1]); c=counts.get(ai,0)
        if c>=per:
            continue
        ident=(ai,round(float(item[3]["board"][0]),3),round(float(item[3]["board"][1]),3))
        if ident in used:
            continue
        chosen.append(item); used.add(ident); counts[ai]=c+1
        if len(chosen)>=limit:
            return chosen
    # Fill the remainder by global score.
    for item in ordered:
        ai=int(item[1]); ident=(ai,round(float(item[3]["board"][0]),3),round(float(item[3]["board"][1]),3))
        if ident in used:
            continue
        chosen.append(item); used.add(ident)
        if len(chosen)>=limit:
            break
    return chosen


def _v40_target_local_active_candidates(mask, dt, boundary_board, outer_contour_board, H_value, config, cfg, target_board, args):
    """Spatially diverse ACTIVE shortlist for the final v40 recovery only.

    The normal v38 shortlist intentionally stays Target-near. Here we preserve the
    same <=40mm bound but sample two radial bands x eight angular sectors so a
    20..40mm ACTIVE that makes the actual PULL reachable is not hidden by many
    near-duplicate points within ~10mm of Target.
    """
    target=np.asarray(target_board,dtype=np.float64).reshape(2)
    search_mm=float(np.clip(getattr(args,"micro_unfold_v38_active_search_radius_mm",40.0),20.0,40.0))
    desired_depth=float(np.clip(getattr(args,"micro_unfold_v38_active_boundary_target_mm",25.0),15.0,35.0))
    max_c=max(10,min(16,int(getattr(args,"micro_unfold_v40_recovery_active_max_candidates",16))))
    target_px=_fix89_board_to_px(H_value,target)
    px_r=80.0
    try:
        px_dx=_fix89_board_to_px(H_value,target+np.asarray([search_mm,0.0]))
        px_dy=_fix89_board_to_px(H_value,target+np.asarray([0.0,search_mm]))
        px_r=float(np.clip(1.35*max(np.linalg.norm(px_dx-target_px),np.linalg.norm(px_dy-target_px))+8.0,35.0,220.0))
    except Exception:
        pass
    h,w=mask.shape
    x0=max(0,int(math.floor(float(target_px[0]-px_r)))); x1=min(w,int(math.ceil(float(target_px[0]+px_r)))+1)
    y0=max(0,int(math.floor(float(target_px[1]-px_r)))); y1=min(h,int(math.ceil(float(target_px[1]+px_r)))+1)
    yy,xx=np.where(mask[y0:y1,x0:x1]>0)
    if xx.size<=0:
        return [],0
    pts=np.column_stack([xx+x0,yy+y0]).astype(np.float32)
    # Bound the cloud before Homography/exact-boundary work; local recovery only.
    cloud_max=max(1500,min(6000,int(getattr(args,"micro_unfold_v40_recovery_active_cloud_max",5000))))
    if len(pts)>cloud_max:
        ids=np.linspace(0,len(pts)-1,cloud_max,dtype=np.int64); pts=pts[ids]
    board=_fix12_transform_points(H_value,pts)
    delta=board-target[None,:]
    d_target=np.linalg.norm(delta,axis=1)
    inside=np.asarray([_fix53_board_inside(cfg,q,margin=25.0) for q in board],dtype=bool)
    keep=np.isfinite(d_target)&(d_target<=search_mm+1e-6)&inside
    if not np.any(keep):
        return [],0
    pts=pts[keep]; board=board[keep]; delta=delta[keep]; d_target=d_target[keep]
    raw_count=int(len(board))
    xi=np.clip(np.rint(pts[:,0]).astype(np.int32),0,w-1); yi=np.clip(np.rint(pts[:,1]).astype(np.int32),0,h-1)
    depths_px=np.asarray(dt[yi,xi],dtype=np.float64)
    px_per_mm=1.0
    try:
        p10x=_fix89_board_to_px(H_value,target+np.asarray([10.0,0.0])); p10y=_fix89_board_to_px(H_value,target+np.asarray([0.0,10.0]))
        px_per_mm=max(0.25,min(float(np.linalg.norm(p10x-target_px))/10.0,float(np.linalg.norm(p10y-target_px))/10.0))
    except Exception:
        pass
    desired_px=desired_depth*px_per_mm
    approx_short=np.maximum(0.0,desired_px-depths_px)/max(px_per_mm,1e-6)
    approx_score=d_target+4.0*approx_short
    angle=np.arctan2(delta[:,1],delta[:,0])
    sector=np.mod(np.floor((angle+math.pi)/(2.0*math.pi)*8.0).astype(np.int32),8)
    # Three radial bands deliberately expose farther 26..40mm alternatives that the
    # normal Target-near shortlist never reaches. Normal path already covered the inner region.
    radial=np.zeros(len(d_target),dtype=np.int32)
    radial[d_target>=0.30*search_mm]=1
    radial[d_target>=0.65*search_mm]=2

    exact_idx=[]
    per_bin=max(8,min(32,int(getattr(args,"micro_unfold_v40_recovery_active_preselect_per_bin",24))))
    for r in (0,1,2):
        for sec in range(8):
            ids=np.where((radial==r)&(sector==sec))[0]
            if ids.size<=0:
                continue
            order=ids[np.argsort(approx_score[ids])[:min(per_bin,len(ids))]]
            exact_idx.extend(order.tolist())
    # Always retain the globally closest/deepest points too.
    exact_idx.extend(np.argsort(approx_score)[:min(64,len(board))].tolist())
    exact_idx=np.unique(np.asarray(exact_idx,dtype=np.int64))
    candidates=[]
    for i in exact_idx.tolist():
        q=np.asarray(board[i],dtype=np.float64)
        reachable,best_r=_v38_point_reachable_by_any_arm(config,q)
        if not reachable:
            continue
        bdist,_=_v38_min_boundary_distance_mm(q,boundary_board)
        _,nearest_outer=_v38_min_boundary_distance_mm(q,outer_contour_board)
        if not np.isfinite(bdist):
            continue
        short=max(0.0,desired_depth-float(bdist))
        quality=float(d_target[i])+4.0*short+0.03*float(best_r)
        candidates.append({"px":np.asarray(pts[i],dtype=np.float64),"board":q,
                           "target_distance_mm":float(d_target[i]),"boundary_distance_mm":float(bdist),
                           "nearest_boundary_board":None if nearest_outer is None else np.asarray(nearest_outer,dtype=np.float64),
                           "best_reach_radius_mm":float(best_r),"v40_quality":quality,
                           "v40_sector":int(sector[i]),"v40_radial_band":int(radial[i])})
    if not candidates:
        return [],raw_count
    # Best candidate per spatial cell gives deliberate ACTIVE diversity.
    cell_best={}
    for c in candidates:
        key=(int(c["v40_radial_band"]),int(c["v40_sector"]))
        if key not in cell_best or float(c["v40_quality"])<float(cell_best[key]["v40_quality"]):
            cell_best[key]=c
    # Recovery should add geometry not already exhausted by the normal path: prefer
    # outer then mid radial cells, while keeping quality ordering inside each band.
    selected=[]
    for band in (2,1,0):
        band_items=[c for c in cell_best.values() if int(c["v40_radial_band"])==band]
        band_items.sort(key=lambda c:(float(c["v40_quality"]),float(c["target_distance_mm"])))
        selected.extend(band_items)
    # Preserve the exact semantic Target when it is itself a valid interior/reachable point.
    target_candidate=None
    try:
        tx,ty=int(round(float(target_px[0]))),int(round(float(target_px[1])))
        if 0<=tx<w and 0<=ty<h and mask[ty,tx]>0 and _fix53_board_inside(cfg,target,margin=25.0):
            reachable,best_r=_v38_point_reachable_by_any_arm(config,target)
            if reachable:
                bdist,_=_v38_min_boundary_distance_mm(target,boundary_board)
                _,nearest_outer=_v38_min_boundary_distance_mm(target,outer_contour_board)
                target_candidate={"px":np.asarray(target_px,dtype=np.float64),"board":target.copy(),
                                  "target_distance_mm":0.0,"boundary_distance_mm":float(bdist),
                                  "nearest_boundary_board":None if nearest_outer is None else np.asarray(nearest_outer,dtype=np.float64),
                                  "best_reach_radius_mm":float(best_r),"v40_quality":4.0*max(0.0,desired_depth-float(bdist)),
                                  "v40_sector":-1,"v40_radial_band":0}
    except Exception:
        pass
    if target_candidate is not None:
        selected=[target_candidate]+[c for c in selected if np.linalg.norm(np.asarray(c["board"])-target)>=5.0]
    # Dedup only near-identical board points; sector/radial diversity is otherwise retained.
    out=[]
    for c in selected:
        if any(float(np.linalg.norm(np.asarray(c["board"])-np.asarray(d["board"])))<5.0 for d in out):
            continue
        out.append(c)
        if len(out)>=max_c:
            break
    return out,raw_count


def _v40_pull_aware_recovery(mask, dt, boundary_board, outer_contour_board, H_value, config, cfg,
                              target, centroid_board, args, outward_for, _v42_e111_cache=None):
    """Final bounded recovery: choose ACTIVE+HOLD with actual PULL reach in the loop.

    It runs only after v38 normal search and v39 reach-aware HOLD recovery both fail.
    Hard 40mm Target correction, 66..126mm pair, 420mm reach, 65mm path and board
    safety are unchanged. Expensive E111/finalizer work is capped to a diverse shortlist.
    """
    diag={"used":False,"reason":"RECOVERY_DISABLED","active_shortlist_count":0,
          "hold_candidate_count":0,"pair_candidate_count":0,"exact_trial_count":0,
          "failure_reason_counts":{}}
    if not bool(getattr(args,"micro_unfold_v40_pull_recovery_enable",True)):
        return None,None,None,diag,[]
    diag["used"]=True
    # Reuse the same local ACTIVE generator, but allow a larger bounded shortlist only here.
    rargs=copy.copy(args)
    setattr(rargs,"micro_unfold_v38_active_max_candidates",
            max(10,min(16,int(getattr(args,"micro_unfold_v40_recovery_active_max_candidates",16)))))
    # Keep HOLD cloud cheap; exact PULL checks happen only after global ranking/diversification.
    setattr(rargs,"micro_unfold_v39_recovery_hold_max_candidates",
            max(6,min(24,int(getattr(args,"micro_unfold_v40_recovery_hold_per_active",12)))))
    active_candidates,active_raw=_v40_target_local_active_candidates(
        mask,dt,boundary_board,outer_contour_board,H_value,config,cfg,target,rargs)
    diag["active_raw_candidate_count"]=int(active_raw)
    diag["active_shortlist_count"]=int(len(active_candidates))
    if not active_candidates:
        diag["reason"]="TARGET_DIRECT_NO_SAFE_ACTIVE"
        return None,None,None,diag,[]

    trials=[]; hold_count=0; cheap_feasible=0
    for ai,active in enumerate(active_candidates):
        outward=outward_for(active)
        if outward is None:
            continue
        holds,hdiag=_v39_reach_aware_hold_candidates(mask,dt,boundary_board,H_value,config,cfg,active,outward,rargs)
        cheap_feasible+=int(hdiag.get("feasible_candidate_count",0) or 0)
        hold_count+=int(len(holds))
        for h in holds:
            h=dict(h); h["v40_pull_recovery"]=True
            h["v40_recovery_trigger"]="V39_RECOVERY_EXHAUSTED"
            # Still cheap: actual E111 PULL endpoint is deliberately deferred to exact shortlist.
            score=(0.30*float(h.get("v39_recovery_best_reach_mm",999.0))
                   -0.25*float(h.get("v39_recovery_path_clearance_mm",65.0))
                   +0.55*float(active.get("target_distance_mm",40.0))
                   +0.08*float(h.get("score",0.0)))
            trials.append((score,ai,active,h,np.asarray(outward,dtype=np.float64)))
    diag["hold_candidate_count"]=int(hold_count)
    diag["cheap_feasible_candidate_count"]=int(cheap_feasible)
    diag["pair_candidate_count"]=int(len(trials))
    if not trials:
        diag["reason"]="TARGET_DIRECT_NO_SAFE_HOLD"
        return None,None,None,diag,[]

    exact_max=max(12,min(36,int(getattr(args,"micro_unfold_v40_recovery_exact_pair_max_trials",36))))
    per_active=max(1,min(3,int(getattr(args,"micro_unfold_v40_recovery_exact_per_active",2))))
    exact_trials=_v40_diverse_exact_trials(trials,exact_max,per_active=per_active)
    diag["exact_trial_count"]=int(len(exact_trials))
    # v42: reuse the exact same snapshot-invariant E111/PULL memoization cache
    # already used by normal + v39 Target-direct trials.
    e111_cache=_v42_e111_cache if isinstance(_v42_e111_cache,dict) else _v42_prepare_target_direct_cache(mask,H_value)
    failures=[]; counts={}
    for _,ai,active,hold,outward in exact_trials:
        try:
            plan=_v38_finalize_target_direct_pair(mask,H_value,config,cfg,target,active,hold,outward,args,
                                                  _v40_e111_cache=e111_cache,_v40_log_pull=False)
            plan["v40_pull_recovery_used"]=True
            plan["v40_recovery_trigger"]="V39_RECOVERY_EXHAUSTED"
            plan["v40_recovery_active_shortlist_count"]=int(len(active_candidates))
            plan["v40_recovery_hold_candidate_count"]=int(hold_count)
            plan["v40_recovery_pair_candidate_count"]=int(len(trials))
            plan["v40_recovery_exact_trial_count"]=int(len(exact_trials))
            plan["v40_failure_reason_counts"]=_training_json_safe(counts)
            diag["reason"]="OK"; diag["failure_reason_counts"]=_training_json_safe(counts)
            return plan,ai,active,diag,failures
        except Exception as exc:
            reason=str(exc); failures.append(reason)
            key=_v40_failure_key(reason); counts[key]=int(counts.get(key,0))+1
    diag["failure_reason_counts"]=_training_json_safe(counts)
    # Most frequent exact blocker is more useful than the first arbitrary trial failure.
    if counts:
        diag["reason"]=max(counts.items(),key=lambda kv:(int(kv[1]),kv[0]))[0]
    else:
        diag["reason"]="TARGET_DIRECT_V40_RECOVERY_EXHAUSTED"
    return None,None,None,diag,failures


def _v38_prepare_target_direct_plan(frame_bgr, mask_u8, H_value, config, cfg, args):
    t0=time.perf_counter()
    target=_v30_target_board(args)
    diag={"attempted":False,"status":"NOT_ATTEMPTED","reason":"TARGET_HINT_DISABLED"}
    if target is None or not bool(getattr(args,"micro_unfold_v38_target_direct_enable",True)):
        return {"status":"NOT_ATTEMPTED","plan":None,"reason":diag["reason"],"diagnostic":diag}
    diag.update({"attempted":True,"target_board":np.asarray(target,dtype=np.float64).tolist(),
                 "v39_reach_hold_recovery_used":False,"v40_pull_recovery_used":False,
                 "v40_failure_reason_counts":{}})
    try:
        mask, contour, boundary_board, outer_contour_board, dt, centroid_px, centroid_board=_v38_mask_geometry(mask_u8,H_value)

        # v51 contract:
        #   WHERE to unfold = v49 cached-local/top-K Target validation.
        #   HOW to unfold   = byte-carried v44 planner geometry below.
        # The local fold context is diagnostic/admission evidence only.  It must not
        # rotate pull direction, reject a v44-valid pair, shorten pull distance, or
        # alter the v44 ACTIVE/HOLD/endpoint/recovery search.
        fold_ctx,v48_target_diag=_v48_target_cca_context(target,centroid_board,args)
        diag["v48_target_validation"]=_training_json_safe(v48_target_diag)
        diag["v48_target_validation_ms"]=float(v48_target_diag.get("elapsed_ms",0.0) or 0.0)
        diag["v48_target_validation_hard_block"]=bool(v48_target_diag.get("hard_block",False))
        setattr(args,"micro_unfold_v48_selected_fold_context",
                copy.deepcopy(fold_ctx) if isinstance(fold_ctx,dict) else None)
        print(
            f"[V51-TARGET-VALIDATE] pass={bool(v48_target_diag.get('pass',False))} "
            f"hard={bool(v48_target_diag.get('hard_block',False))} "
            f"reason={v48_target_diag.get('reason')} pool={v48_target_diag.get('pool')} "
            f"comp={v48_target_diag.get('selected_component_id')} "
            f"rule={v48_target_diag.get('selected_rule')} "
            f"perp={v48_target_diag.get('target_perp_mm')} "
            f"end={v48_target_diag.get('target_axial_overrun_mm')} "
            f"time={float(v48_target_diag.get('elapsed_ms',0.0)):.2f}ms"
        )
        if bool(v48_target_diag.get("hard_block",False)):
            raise RuntimeError(
                f"V51_LOCAL_FOLD_CONTEXT_BLOCK:{str(v48_target_diag.get('reason') or 'UNKNOWN')}"
            )

        # From here through ACTIVE/HOLD selection, outer-normal pull direction,
        # pull distance, endpoint generation and reach/recovery, the logic is v44.
        # v42: one cache for all normal/v39/v40 trials from this same frozen snapshot.
        direct_e111_cache=_v42_prepare_target_direct_cache(mask,H_value,outer_contour_px=contour)
        tried_pair_keys=set()
        diag["v42_pair_duplicate_skips"]=0
        diag["v42_shared_geometry_cache"]=True
        active_candidates,active_raw=_v38_target_local_active_candidates(mask,dt,boundary_board,outer_contour_board,H_value,config,cfg,target,args)
        diag["active_search_candidate_count"]=int(active_raw); diag["active_shortlist_count"]=int(len(active_candidates))
        if not active_candidates:
            raise RuntimeError("TARGET_DIRECT_NO_SAFE_ACTIVE")

        def outward_for(active):
            nearest=np.asarray(active.get("nearest_boundary_board"),dtype=np.float64).reshape(2) if active.get("nearest_boundary_board") is not None else None
            if nearest is not None:
                outward=nearest-np.asarray(active["board"],dtype=np.float64)
            else:
                outward=np.asarray(active["board"],dtype=np.float64)-np.asarray(centroid_board,dtype=np.float64)
            on=float(np.linalg.norm(outward))
            if on<=1e-6:
                outward=np.asarray(active["board"],dtype=np.float64)-np.asarray(centroid_board,dtype=np.float64); on=float(np.linalg.norm(outward))
            return (outward/on) if on>1e-6 else None

        pair_trials=[]; hold_total=0
        for ai,active in enumerate(active_candidates):
            outward=outward_for(active)
            if outward is None:
                continue
            holds,hold_raw=_v38_hold_candidates(mask,dt,boundary_board,H_value,config,cfg,active,outward,args)
            hold_total+=int(hold_raw)
            for h in holds:
                pair_score=(float(active["target_distance_mm"])+0.55*float(h["score"])-0.25*float(active["boundary_distance_mm"]))
                pair_trials.append((pair_score,ai,active,h,outward.copy()))
        diag["hold_candidate_count"]=int(hold_total); diag["pair_candidate_count"]=int(len(pair_trials))
        pair_trials.sort(key=lambda z:float(z[0]))
        max_trials=max(1,min(48,int(getattr(args,"micro_unfold_v38_pair_max_trials",24))))
        errors=[]

        def try_trials(trials,limit,recovery_trigger=""):
            local_errors=[]; examined=0
            for _,ai,active,hold,outward in trials:
                if examined>=int(limit):
                    break
                pair_key=_v42_trial_pair_key(active,hold)
                if pair_key is not None and pair_key in tried_pair_keys:
                    diag["v42_pair_duplicate_skips"]=int(diag.get("v42_pair_duplicate_skips",0))+1
                    stats=direct_e111_cache.setdefault("v42_stats",{})
                    stats["pair_duplicate_skips"]=int(stats.get("pair_duplicate_skips",0))+1
                    continue
                if pair_key is not None:
                    tried_pair_keys.add(pair_key)
                examined+=1
                try:
                    if recovery_trigger:
                        hold=dict(hold); hold["v39_recovery_trigger"]=str(recovery_trigger)
                    plan=_v38_finalize_target_direct_pair(mask,H_value,config,cfg,target,active,hold,outward,args,
                                                          _v40_e111_cache=direct_e111_cache,_v40_log_pull=False)
                    stats=direct_e111_cache.get("v42_stats",{})
                    plan["v42_pair_duplicate_skips"]=int(diag.get("v42_pair_duplicate_skips",0))
                    plan["v42_pull_recovery_cache_hits"]=int(stats.get("pull_recovery_cache_hits",0) or 0)
                    plan["v42_pull_recovery_cheap_rejects"]=int(stats.get("pull_recovery_cheap_rejects",0) or 0)
                    plan["v42_shared_geometry_cache"]=True
                    plan["v38_active_search_candidate_count"]=int(active_raw)
                    plan["v38_hold_candidate_count"]=int(hold_total)
                    plan["v38_pair_candidate_count"]=int(len(pair_trials))
                    plan["v12_candidate_count_total"]=int(len(pair_trials))
                    plan["v39_recovery_raw_candidate_count"]=int(diag.get("v39_recovery_raw_candidate_count",0) or 0)
                    plan["v39_recovery_feasible_candidate_count"]=int(diag.get("v39_recovery_feasible_candidate_count",0) or 0)
                    plan["v39_recovery_hold_candidate_count"]=int(diag.get("v39_recovery_hold_candidate_count",0) or 0)
                    plan["v39_recovery_pair_candidate_count"]=int(diag.get("v39_recovery_pair_candidate_count",0) or 0)
                    return plan,ai,active,local_errors
                except Exception as exc:
                    local_errors.append(str(exc))
            return None,None,None,local_errors

        plan,ai,active,normal_errors=try_trials(pair_trials,max_trials)
        errors.extend(normal_errors)
        recovery_trigger=""
        if plan is None and bool(getattr(args,"micro_unfold_v39_reach_hold_recovery_enable",True)):
            # Recover when v38 produced no HOLD at all OR its bounded shortlist could
            # not survive the unchanged final reach/path/pull checks.
            recovery_trigger=("NO_SAFE_HOLD" if not pair_trials else "PAIR_FINALIZE_REACH_PATH_BLOCKED")
            recovery_trials=[]; rec_raw=0; rec_feasible=0; rec_holds=0
            for rai,ractive in enumerate(active_candidates):
                routward=outward_for(ractive)
                if routward is None:
                    continue
                rhs,rdiag=_v39_reach_aware_hold_candidates(mask,dt,boundary_board,H_value,config,cfg,ractive,routward,args)
                rec_raw+=int(rdiag.get("raw_candidate_count",0) or 0)
                rec_feasible+=int(rdiag.get("feasible_candidate_count",0) or 0)
                rec_holds+=int(len(rhs))
                for h in rhs:
                    # Reach/path-first recovery score; active Target correction stays a mild penalty.
                    rscore=(float(h.get("v39_recovery_best_reach_mm",999.0))
                            -0.35*float(h.get("v39_recovery_path_clearance_mm",65.0))
                            +0.20*float(ractive["target_distance_mm"])+0.05*float(h["score"]))
                    recovery_trials.append((rscore,rai,ractive,h,routward.copy()))
            recovery_trials.sort(key=lambda z:float(z[0]))
            diag.update({"v39_reach_hold_recovery_used":True,"v39_recovery_trigger":recovery_trigger,
                         "v39_recovery_raw_candidate_count":int(rec_raw),
                         "v39_recovery_feasible_candidate_count":int(rec_feasible),
                         "v39_recovery_hold_candidate_count":int(rec_holds),
                         "v39_recovery_pair_candidate_count":int(len(recovery_trials))})
            print(f"[MICRO-TARGET-DIRECT-REACH-HOLD-RECOVERY-V39] trigger={recovery_trigger} raw={rec_raw} "
                  f"feasible={rec_feasible} holds={rec_holds} pairs={len(recovery_trials)}")
            rec_limit=max(1,min(72,int(getattr(args,"micro_unfold_v39_recovery_pair_max_trials",36))))
            plan,ai,active,rec_errors=try_trials(recovery_trials,rec_limit,recovery_trigger=recovery_trigger)
            errors.extend(rec_errors)
            if plan is not None:
                plan["v38_pair_candidate_count"]=int(len(pair_trials))
                plan["v12_candidate_count_total"]=int(len(pair_trials)+len(recovery_trials))

        # v40: only after normal Target-direct AND v39 HOLD recovery are exhausted,
        # retry a larger-but-bounded ACTIVE shortlist and include the actual PULL endpoint
        # in the exact physical decision. No safety threshold is relaxed.
        if plan is None and bool(getattr(args,"micro_unfold_v40_pull_recovery_enable",True)):
            v40_plan,v40_ai,v40_active,v40_diag,v40_errors=_v40_pull_aware_recovery(
                mask,dt,boundary_board,outer_contour_board,H_value,config,cfg,target,centroid_board,args,outward_for,
                _v42_e111_cache=direct_e111_cache)
            diag.update({
                "v40_pull_recovery_used":bool(v40_diag.get("used",False)),
                "v40_recovery_trigger":"V39_RECOVERY_EXHAUSTED",
                "v40_recovery_active_shortlist_count":int(v40_diag.get("active_shortlist_count",0) or 0),
                "v40_recovery_hold_candidate_count":int(v40_diag.get("hold_candidate_count",0) or 0),
                "v40_recovery_pair_candidate_count":int(v40_diag.get("pair_candidate_count",0) or 0),
                "v40_recovery_exact_trial_count":int(v40_diag.get("exact_trial_count",0) or 0),
                "v40_failure_reason_counts":copy.deepcopy(v40_diag.get("failure_reason_counts",{})),
            })
            print(f"[MICRO-TARGET-DIRECT-ACTIVE-HOLD-PULL-RECOVERY-V40] active={diag['v40_recovery_active_shortlist_count']} "
                  f"holds={diag['v40_recovery_hold_candidate_count']} pairs={diag['v40_recovery_pair_candidate_count']} "
                  f"exact={diag['v40_recovery_exact_trial_count']} result={v40_diag.get('reason')} "
                  f"failures={diag['v40_failure_reason_counts']}")
            errors.extend(v40_errors)
            if v40_plan is not None:
                plan,ai,active=v40_plan,v40_ai,v40_active
                plan["v38_pair_candidate_count"]=int(len(pair_trials))
                plan["v12_candidate_count_total"]=int(len(pair_trials)+int(diag.get("v39_recovery_pair_candidate_count",0) or 0)+int(diag.get("v40_recovery_pair_candidate_count",0) or 0))

        if plan is not None:
            # Post-geometry provenance only; these fields do not feed planner math.
            plan["v51_v44_full_planner_geometry"]=True
            plan["v51_v44_planner_scope"]="ACTIVE_HOLD_PULL_HINT_OUTER_NORMAL_DISTANCE_ENDPOINT_RECOVERY"
            plan["v51_target_authority"]="V49_TOPK_CACHED_LOCAL"
            plan["v48_target_validation_used"]=bool(isinstance(fold_ctx,dict))
            plan["v48_target_validation_pass"]=bool(v48_target_diag.get("pass",False))
            plan["v48_target_validation_reason"]=str(v48_target_diag.get("reason",""))
            plan["v48_target_validation_component_id"]=v48_target_diag.get("selected_component_id")
            plan["v48_target_validation_component_rule"]=v48_target_diag.get("selected_rule")
            plan["v48_target_validation_perp_mm"]=v48_target_diag.get("target_perp_mm")
            plan["v48_target_validation_axial_overrun_mm"]=v48_target_diag.get("target_axial_overrun_mm")
            plan["v48_target_validation_segment_distance_mm"]=v48_target_diag.get("target_segment_distance_mm")
            plan["v48_target_validation_ms"]=float(v48_target_diag.get("elapsed_ms",0.0) or 0.0)
            plan["v48_direction_validation_used"]=False
            plan["v48_direction_validation_pass"]=False
            plan["v48_direction_validation_reason"]="V51_V44_PLANNER_GEOMETRY_DIRECTION_NOT_OVERRIDDEN"
            if isinstance(fold_ctx,dict):
                plan["v48_crease_anchor_board"]=_training_json_safe(fold_ctx.get("anchor_board"))
                plan["v48_crease_tangent_board"]=_training_json_safe(fold_ctx.get("tangent_board"))
                plan["v48_crease_normal_board"]=_training_json_safe(fold_ctx.get("normal_board"))

            diag.update({"status":"READY","reason":"OK","selected_active_index":int(ai),
                         "active_inset_mm":float(active["target_distance_mm"]),
                         "active_boundary_distance_mm":float(active["boundary_distance_mm"]),
                         "pair_separation_mm":float(plan["pair_separation_mm"]),
                         "path_clearance_mm":float(plan["v38_path_clearance_mm"]),
                         "max_reach_radius_mm":float(plan["v38_max_reach_radius_mm"])})
            stats=direct_e111_cache.get("v42_stats",{})
            diag["v42_pull_recovery_cache_hits"]=int(stats.get("pull_recovery_cache_hits",0) or 0)
            diag["v42_pull_recovery_cheap_rejects"]=int(stats.get("pull_recovery_cheap_rejects",0) or 0)
            diag["elapsed_ms"]=1000.0*(time.perf_counter()-t0)
            if bool(plan.get("v40_pull_recovery_used",False)):
                tag="PULL-RECOVERY-READY-V40"
            elif bool(plan.get("v39_reach_hold_recovery_used",False)):
                tag="HOLD-RECOVERY-READY-V40"
            else:
                tag="READY-V40"
            print(f"[MICRO-TARGET-DIRECT-{tag}] target={diag['target_board']} inset={diag['active_inset_mm']:.1f}mm "
                  f"boundary={diag['active_boundary_distance_mm']:.1f}mm pair={diag['pair_separation_mm']:.1f}mm "
                  f"path={diag['path_clearance_mm']:.1f}mm reach={diag['max_reach_radius_mm']:.1f}mm time={diag['elapsed_ms']:.1f}ms")
            print(
                f"[V51-V44-PLANNER] source=TARGET_DIRECT pull={float(plan.get('pull_distance_mm',0.0)):.1f}mm "
                f"dirRecovery={bool(plan.get('v41_pull_direction_recovery_used',False))} "
                f"delta={float(plan.get('v41_pull_direction_delta_deg',0.0) or 0.0):+.1f}deg "
                f"backoff={float(plan.get('v41_pull_direction_backoff_mm',0.0) or 0.0):.1f}mm"
            )
            return {"status":"READY","plan":plan,"reason":"OK","diagnostic":diag}

        reason="TARGET_DIRECT_NO_SAFE_HOLD" if not pair_trials else "TARGET_DIRECT_PHYSICAL_BLOCKED"
        # Aggregate all exact-finalizer blockers; do not hide PULL failures behind the old aggregate reason.
        counts={}
        for item in errors:
            key=_v40_failure_key(item); counts[key]=int(counts.get(key,0))+1
        if counts:
            reason=max(counts.items(),key=lambda kv:(int(kv[1]),kv[0]))[0]
        diag["v40_failure_reason_counts"]=_training_json_safe(counts)
        stats=direct_e111_cache.get("v42_stats",{})
        diag["v42_pull_recovery_cache_hits"]=int(stats.get("pull_recovery_cache_hits",0) or 0)
        diag["v42_pull_recovery_cheap_rejects"]=int(stats.get("pull_recovery_cheap_rejects",0) or 0)
        diag.update({"status":"BLOCKED","reason":reason,"trial_errors":errors[:12],
                     "elapsed_ms":1000.0*(time.perf_counter()-t0)})
        print(f"[MICRO-TARGET-DIRECT-BLOCKED-V40] reason={reason} pairs={len(pair_trials)} "
              f"holdRecovery={diag.get('v39_reach_hold_recovery_used')} pullRecovery={diag.get('v40_pull_recovery_used')} "
              f"failureCounts={diag.get('v40_failure_reason_counts')} dupSkip={diag.get('v42_pair_duplicate_skips',0)} "
              f"pullCacheHit={diag.get('v42_pull_recovery_cache_hits',0)} cheapReject={diag.get('v42_pull_recovery_cheap_rejects',0)} "
              f"time={diag['elapsed_ms']:.1f}ms")
        return {"status":"BLOCKED","plan":None,"reason":reason,"diagnostic":diag}
    except Exception as exc:
        reason=str(exc)
        diag.update({"status":"BLOCKED","reason":reason,"elapsed_ms":1000.0*(time.perf_counter()-t0)})
        print(f"[MICRO-TARGET-DIRECT-BLOCKED-V40] reason={reason} time={diag['elapsed_ms']:.1f}ms")
        return {"status":"BLOCKED","plan":None,"reason":reason,"diagnostic":diag}



def _v41_fast_target_local_plan(frame_bgr, mask_u8, pose_plan, H_value, config, cfg, args, *, rgb_pre=None, depth_state=None):
    """Bounded target-local crease path before FULL E156 candidate search.

    Only the RGB/crease state needed by the existing v32 target-local exact-normal
    recovery is prepared.  It never runs global v31/v32 candidate enumeration.
    """
    t0=time.perf_counter(); diag={"attempted":False,"status":"DISABLED","reason":"DISABLED"}
    if not bool(getattr(args,"micro_unfold_v41_fast_target_local_enable",True)) or _v30_target_board(args) is None:
        return {"status":"BLOCKED","plan":None,"reason":"FAST_TARGET_LOCAL_DISABLED","diagnostic":diag,"rgb_pre":rgb_pre,"depth_state":depth_state}
    diag["attempted"]=True
    frame=np.asarray(frame_bgr,dtype=np.uint8); mask=((np.asarray(mask_u8,dtype=np.uint8)>0).astype(np.uint8)*255)
    stage={}
    try:
        if rgb_pre is None:
            t=time.perf_counter(); rgb_pre=_e156_precompute_rgb_heavy(frame,mask); stage["rgb_precompute_ms"]=1000.0*(time.perf_counter()-t)
        else:
            stage["rgb_precompute_ms"]=0.0; stage["rgb_cache_reused"]=True
        if depth_state is None:
            t=time.perf_counter(); pose_state=build_pose_state_from_vla(pose_plan)
            depth_state=_fix53_depth_geometry_compute(frame,mask,pose_state=pose_state,e156_rgb_precompute=rgb_pre)
            stage["geometry_ms"]=1000.0*(time.perf_counter()-t)
        else:
            stage["geometry_ms"]=0.0; stage["geometry_cache_reused"]=True
        if not isinstance(depth_state,dict) or not bool(depth_state.get("available",False)):
            raise RuntimeError(str((depth_state or {}).get("reason","FAST_TARGET_LOCAL_GEOMETRY_FAILED")))
        max_c=max(1,min(3,int(getattr(args,"micro_unfold_v41_fast_target_local_max_candidates",2))))
        t=time.perf_counter(); locals_=_v32_target_local_exact_normals(depth_state,mask,H_value,args,max_candidates=max_c,exclude_masks=None)
        stage["target_local_seed_ms"]=1000.0*(time.perf_counter()-t); diag["candidate_count"]=int(len(locals_))
        exec_crease=np.asarray(depth_state.get('e122_fold_execution_skeleton',depth_state.get('fold_crease_skeleton',np.zeros_like(mask))),dtype=np.uint8)
        preview=np.asarray(depth_state.get('e122_fold_preview_skeleton',depth_state.get('fold_crease_skeleton',np.zeros_like(mask))),dtype=np.uint8)
        errors=[]
        for rank,exact in enumerate(locals_,1):
            local_state=dict(depth_state); fake=_v32_fake_from_exact(exact,group_prefix='V41_FAST_TARGET_LOCAL')
            pair=None
            try:
                pair=_e130_directional_pair(local_state,mask,H_value,args,config,cfg,fake,exec_crease,preview)
            except Exception as exc:
                errors.append(f"R{rank}:E130:{exc!r}")
            if not isinstance(pair,dict):
                try:
                    pair=_v13_try_same_crease_anchor_pairs(local_state,mask,H_value,args,config,cfg,fake,exec_crease,preview,
                        label=f'V41_FAST_TARGET_LOCAL_R{rank}',max_trials=max(1,min(2,int(getattr(args,'micro_unfold_v41_fast_target_local_anchor_trials',1)))))
                except Exception as exc:
                    errors.append(f"R{rank}:ANCHOR:{exc!r}")
            if not isinstance(pair,dict):
                continue
            try:
                full=_fix53_plan_points(local_state,np.asarray(H_value,dtype=np.float32),config,cfg,args,preselected_crease_pair=pair)
            except Exception as exc:
                errors.append(f"R{rank}:FINALIZE:{exc!r}"); continue
            if not isinstance(full,dict) or not bool(full.get("execution_evidence_ok",True)):
                errors.append(f"R{rank}:EVIDENCE:{(full or {}).get('execution_evidence_reason','UNKNOWN') if isinstance(full,dict) else 'NONE'}"); continue
            full["v38_planner_source"]="FAST_TARGET_LOCAL"
            full["v38_target_direct_attempted"]=True; full["v38_target_direct_used"]=False
            full["v38_target_board"]=_training_json_safe(_v30_target_board(args)); full["v38_fallback_used"]=False
            full["v41_fast_target_local_used"]=True
            full["v41_fast_target_local_source"]=str(exact.get('v32_target_local_source','UNKNOWN'))
            full["v41_fast_target_local_distance_mm"]=_training_json_safe(exact.get('v32_target_local_distance_mm'))
            full["v41_fast_target_local_candidate_rank"]=int(rank)
            full["vla_joint_localization_policy"]="V41_FAST_TARGET_LOCAL_TARGET_AUTHORITY"
            full["vla_joint_role_validation"]={"reason":"BYPASSED_BY_V41_FAST_TARGET_LOCAL","semantic_target_authority":True}
            full["execution_evidence_reason"]="V41_FAST_TARGET_LOCAL_PHYSICAL_SAFETY_OK"
            full["e156_vla_core_version"]=E156_CORE_VERSION
            diag.update({"status":"READY","reason":"OK","selected_rank":int(rank),"source":str(exact.get('v32_target_local_source','UNKNOWN')),
                         "distance_mm":_training_json_safe(exact.get('v32_target_local_distance_mm'))})
            stage["total_ms"]=1000.0*(time.perf_counter()-t0); diag["stage_times_ms"]=_training_json_safe(stage)
            print(f"[MICRO-FAST-TARGET-LOCAL-READY-V41] rank={rank} source={diag['source']} dist={diag.get('distance_mm')} time={stage['total_ms']:.1f}ms")
            return {"status":"READY","plan":full,"reason":"OK","diagnostic":diag,"rgb_pre":rgb_pre,"depth_state":depth_state}
        diag.update({"status":"BLOCKED","reason":"FAST_TARGET_LOCAL_NO_SAFE_PLAN","errors":errors[:8]})
    except Exception as exc:
        diag.update({"status":"BLOCKED","reason":f"FAST_TARGET_LOCAL_EXCEPTION:{exc!r}"})
    stage["total_ms"]=1000.0*(time.perf_counter()-t0); diag["stage_times_ms"]=_training_json_safe(stage)
    print(f"[MICRO-FAST-TARGET-LOCAL-BLOCKED-V41] reason={diag.get('reason')} candidates={diag.get('candidate_count',0)} time={stage['total_ms']:.1f}ms")
    return {"status":"BLOCKED","plan":None,"reason":str(diag.get('reason')),"diagnostic":diag,"rgb_pre":rgb_pre,"depth_state":depth_state}



def prepare_plan(frame_bgr, mask_u8, pose_plan, H_value, config, runtime_cfg, *,
                 structural_hint: bool = False, pose_deform_min_conf: float = 0.20,
                 generation: Any = None, arg_overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Build one frozen MICRO plan: Target-direct -> FAST target-local -> optional FULL E156."""
    global _E92_MICRO_ARGS
    t0 = time.perf_counter()
    stage = {}
    globals()["_E156_LAST_DIRECTIONAL_PAIR_MS"] = 0.0
    globals()["_E156_LAST_BRANCH_MS"] = 0.0

    # v38: the Action Model + Target Model already decided WHERE MICRO should act.
    # Build the light Target-direct physical plan before any Pose/RGB crease semantics.
    # Only a physical/safety BLOCK falls through to the byte-preserved E156 path.
    _E92_MICRO_ARGS = _merged_micro_args(arg_overrides)
    direct_diag = None
    if _v30_target_board(_E92_MICRO_ARGS) is not None and bool(getattr(_E92_MICRO_ARGS, "micro_unfold_v38_target_direct_enable", True)):
        td0=time.perf_counter()
        try:
            direct=_v38_prepare_target_direct_plan(frame_bgr,mask_u8,np.asarray(H_value,dtype=np.float32),config,runtime_cfg,_E92_MICRO_ARGS)
        except Exception as exc:
            direct={"status":"BLOCKED","plan":None,"reason":f"TARGET_DIRECT_EXCEPTION:{exc!r}",
                    "diagnostic":{"attempted":True,"status":"BLOCKED","reason":f"TARGET_DIRECT_EXCEPTION:{exc!r}"}}
        stage["target_direct"] = 1000.0*(time.perf_counter()-td0)
        direct_diag=copy.deepcopy(direct.get("diagnostic") if isinstance(direct.get("diagnostic"),dict) else {})
        if str(direct.get("status") or "").upper()=="READY" and isinstance(direct.get("plan"),dict):
            full_plan=direct["plan"]
            full_plan["state"]="READY"
            full_plan["e156_vla_core_version"]=E156_CORE_VERSION
            summary=plan_summary(full_plan)
            frozen=freeze_execution_plan(full_plan)
            if not isinstance(frozen,dict):
                direct_diag["status"]="BLOCKED"; direct_diag["reason"]="V40_FROZEN_EXECUTION_CONTRACT_INCOMPLETE"
            else:
                bid=binding_id(frozen,generation=generation)
                stage["finalize"]=0.0
                stage["target_validation_ms"]=float((direct_diag or {}).get("v48_target_validation_ms",0.0) or 0.0)
                stage["direction_validation_ms"]=float(full_plan.get("v48_direction_validation_ms",(direct_diag or {}).get("v48_direction_validation_ms",0.0)) or 0.0)
                stage["total"]=1000.0*(time.perf_counter()-t0)
                gate={"open":True,"reason":"V41_TARGET_DIRECT_READY","v38_target_direct":_training_json_safe(direct_diag),
                      "planner_source":"TARGET_DIRECT"}
                return {"status":"READY","plan":frozen,"diagnostic_plan":None,"binding_id":bid,"gate":gate,
                        "reason":"V41_TARGET_DIRECT_PHYSICAL_SAFETY_OK","summary":summary,
                        "elapsed_s":float(time.perf_counter()-t0),"stage_times_ms":stage,"core_version":E156_CORE_VERSION}
        if isinstance(direct_diag,dict):
            stage["target_validation_ms"]=float(direct_diag.get("v48_target_validation_ms",0.0) or 0.0)
            stage["direction_validation_ms"]=float(direct_diag.get("v48_direction_validation_ms",0.0) or 0.0)
        if bool((direct_diag or {}).get("v48_target_validation_hard_block",False)):
            stage["fast_target_local_skipped_v48_target_mismatch"]=True
            stage["full_e156_skipped"]=True
            stage["total"]=1000.0*(time.perf_counter()-t0)
            reason=str((direct_diag or {}).get("reason") or direct.get("reason") or "V51_LOCAL_FOLD_CONTEXT_BLOCK")
            gate={"open":False,"reason":reason,"planner_source":"V51_LOCAL_FOLD_CONTEXT_BLOCK",
                  "v38_target_direct":_training_json_safe(direct_diag),
                  "v48_target_validation_hard_block":True,
                  "v42_full_e156_fallback_enabled":False}
            print(f"[V51-TARGET-HARD-BLOCK] reason={reason}; FAST/FULL skipped because cached local fold context is stale/invalid")
            return {"status":"BLOCKED","plan":None,"diagnostic_plan":None,"binding_id":None,
                    "gate":gate,"reason":reason,"summary":None,
                    "elapsed_s":float(time.perf_counter()-t0),"stage_times_ms":stage,
                    "core_version":E156_CORE_VERSION}
        print(f"[MICRO-TARGET-DIRECT-NEXT-V51] reason={str((direct_diag or {}).get('reason') or direct.get('reason') or 'BLOCKED')}")

    # v51: after v49 Target validation, use the exact v44 FAST target-local planner before FULL E156
    # target-local exact-normal idea with only the minimum RGB/crease state.  The
    # generated RGB/geometry cache is retained and reused if FULL E156 is still needed.
    fast_rgb_pre=None; fast_depth_state=None; fast_diag=None
    fast_reason="FAST_TARGET_LOCAL_NOT_ATTEMPTED"
    if direct_diag is not None and _v30_target_board(_E92_MICRO_ARGS) is not None and bool(getattr(_E92_MICRO_ARGS,"micro_unfold_v41_fast_target_local_enable",True)):
        ft0=time.perf_counter()
        fast=_v41_fast_target_local_plan(frame_bgr,mask_u8,pose_plan,np.asarray(H_value,dtype=np.float32),config,runtime_cfg,_E92_MICRO_ARGS)
        stage["fast_target_local"]=1000.0*(time.perf_counter()-ft0)
        fast_diag=copy.deepcopy(fast.get("diagnostic") if isinstance(fast.get("diagnostic"),dict) else {})
        fast_rgb_pre=fast.get("rgb_pre"); fast_depth_state=fast.get("depth_state")
        if str(fast.get("status") or "").upper()=="READY" and isinstance(fast.get("plan"),dict):
            full_plan=fast["plan"]; full_plan["state"]="READY"; full_plan["e156_vla_core_version"]=E156_CORE_VERSION
            # FAST planner above is the exact v44 implementation.  v49 Target
            # validation is copied only as provenance after geometry is complete.
            full_plan["v51_v44_full_planner_geometry"]=True
            full_plan["v51_v44_planner_scope"]="FAST_TARGET_LOCAL_V44_EXACT"
            full_plan["v51_target_authority"]="V49_TOPK_CACHED_LOCAL"
            if isinstance(direct_diag,dict):
                _tv=direct_diag.get("v48_target_validation")
                if isinstance(_tv,dict):
                    full_plan["v48_target_validation_used"]=True
                    full_plan["v48_target_validation_pass"]=bool(_tv.get("pass",False))
                    full_plan["v48_target_validation_reason"]=str(_tv.get("reason",""))
                    full_plan["v48_target_validation_component_id"]=_tv.get("selected_component_id")
                    full_plan["v48_target_validation_component_rule"]=_tv.get("selected_rule")
                    full_plan["v48_target_validation_perp_mm"]=_tv.get("target_perp_mm")
                    full_plan["v48_target_validation_axial_overrun_mm"]=_tv.get("target_axial_overrun_mm")
                    full_plan["v48_target_validation_segment_distance_mm"]=_tv.get("target_segment_distance_mm")
                    full_plan["v48_target_validation_ms"]=float(_tv.get("elapsed_ms",0.0) or 0.0)
            full_plan["v48_direction_validation_used"]=False
            full_plan["v48_direction_validation_pass"]=False
            full_plan["v48_direction_validation_reason"]="V51_V44_FAST_PLANNER_GEOMETRY_DIRECTION_NOT_OVERRIDDEN"
            print(
                f"[V51-V44-PLANNER] source=FAST_TARGET_LOCAL pull={float(full_plan.get('pull_distance_mm',0.0)):.1f}mm "
                f"dirRecovery={bool(full_plan.get('v41_pull_direction_recovery_used',False))} "
                f"delta={float(full_plan.get('v41_pull_direction_delta_deg',0.0) or 0.0):+.1f}deg "
                f"backoff={float(full_plan.get('v41_pull_direction_backoff_mm',0.0) or 0.0):.1f}mm"
            )
            summary=plan_summary(full_plan); frozen=freeze_execution_plan(full_plan)
            if isinstance(frozen,dict):
                bid=binding_id(frozen,generation=generation); stage["finalize"]=0.0; stage["total"]=1000.0*(time.perf_counter()-t0)
                gate={"open":True,"reason":"V41_FAST_TARGET_LOCAL_READY","v38_target_direct":_training_json_safe(direct_diag),
                      "v41_fast_target_local":_training_json_safe(fast_diag),"planner_source":"FAST_TARGET_LOCAL"}
                return {"status":"READY","plan":frozen,"diagnostic_plan":None,"binding_id":bid,"gate":gate,
                        "reason":"V41_FAST_TARGET_LOCAL_PHYSICAL_SAFETY_OK","summary":summary,
                        "elapsed_s":float(time.perf_counter()-t0),"stage_times_ms":stage,"core_version":E156_CORE_VERSION}
        fast_reason=str((fast_diag or {}).get('reason') or fast.get('reason') or 'BLOCKED')

    # v42 AUTO policy: after bounded Target-direct + FAST target-local are exhausted,
    # return BLOCKED immediately.  FULL E156 remains available only via explicit override.
    if not bool(getattr(_E92_MICRO_ARGS,"micro_unfold_v42_full_e156_fallback_enable",False)):
        stage["full_e156_skipped"]=True
        stage["total"]=1000.0*(time.perf_counter()-t0)
        reason="V42_FAST_PATHS_EXHAUSTED_FULL_E156_DISABLED"
        gate={"open":False,"reason":reason,"planner_source":"FAST_PATHS_ONLY",
              "v38_target_direct":_training_json_safe(direct_diag),
              "v41_fast_target_local":_training_json_safe(fast_diag),
              "v42_full_e156_fallback_enabled":False,
              "v42_fast_target_local_last_reason":fast_reason}
        print(f"[MICRO-FULL-E156-SKIPPED-V42] direct={str((direct_diag or {}).get('reason') or 'NOT_ATTEMPTED')} "
              f"fast={fast_reason} -> BLOCKED immediately")
        return {"status":"BLOCKED","plan":None,"diagnostic_plan":None,"binding_id":None,
                "gate":gate,"reason":reason,"summary":None,
                "elapsed_s":float(time.perf_counter()-t0),"stage_times_ms":stage,
                "core_version":E156_CORE_VERSION}
    print(f"[MICRO-FAST-TARGET-LOCAL-FALLBACK-FULL-E156-V42] reason={fast_reason} debugOverride=ON")

    # Existing v37 FULL E156 fallback is retained byte-semantically for explicit debug/manual override only.
    tg = time.perf_counter()
    gate = quick_gate(frame_bgr, mask_u8, pose_plan, structural_hint=structural_hint,
                      pose_deform_min_conf=pose_deform_min_conf, arg_overrides=arg_overrides)
    stage["gate"] = 1000.0 * (time.perf_counter() - tg)
    if isinstance(direct_diag,dict):
        gate["v38_target_direct"]=_training_json_safe(direct_diag)
        gate["v38_fallback_used"]=True
    if isinstance(fast_diag,dict):
        gate["v41_fast_target_local"]=_training_json_safe(fast_diag)
        gate["v41_fast_target_local_attempted"]=True
    strict_gate_open=bool(gate.get('open',False))
    rescue_enabled=bool(getattr(_E92_MICRO_ARGS,'micro_unfold_vla_rgb_rescue_enable',True))
    gate['strict_pose_gate_open']=bool(strict_gate_open)
    gate['rgb_rescue_probe']=bool((not strict_gate_open) and rescue_enabled)
    print(f"[E156-V12-GATE] strictOpen={strict_gate_open} rescueProbe={gate['rgb_rescue_probe']} reason={gate.get('reason')} "
          f"intr={gate.get('intrusion_count')} deform={gate.get('deform_mode')}:{gate.get('deform_confidence')} "
          f"struct={gate.get('structural_hint')} time={stage['gate']:.1f}ms")
    if not strict_gate_open and not rescue_enabled:
        stage["total"] = 1000.0 * (time.perf_counter() - t0)
        return {"status": "NOT_APPLICABLE", "plan": None, "binding_id": None, "gate": gate,
                "reason":str(gate.get('reason','NO_MICRO_HINT')),
                "elapsed_s": float((time.perf_counter()-t0)), "stage_times_ms": stage,
                "core_version": E156_CORE_VERSION}

    frame = np.asarray(frame_bgr, dtype=np.uint8)
    mask = ((np.asarray(mask_u8, dtype=np.uint8) > 0).astype(np.uint8) * 255)
    try:
        tr = time.perf_counter()
        if fast_rgb_pre is not None:
            rgb_pre=fast_rgb_pre; stage["rgb_heavy"]=0.0; stage["rgb_heavy_reused_from_fast_target_local"]=True
        else:
            rgb_pre = _e156_precompute_rgb_heavy(frame, mask)
            stage["rgb_heavy"] = 1000.0 * (time.perf_counter() - tr)

        td = time.perf_counter()
        if isinstance(fast_depth_state,dict) and bool(fast_depth_state.get("available",False)):
            depth_state=fast_depth_state; stage["geometry"]=0.0; stage["geometry_reused_from_fast_target_local"]=True
        else:
            depth_state = _fix53_depth_geometry_compute(frame, mask, pose_state=gate.get("pose_state"), e156_rgb_precompute=rgb_pre)
            stage["geometry"] = 1000.0 * (time.perf_counter() - td)
        if not isinstance(depth_state, dict) or not bool(depth_state.get("available", False)):
            stage["branch"] = float(globals().get("_E156_LAST_BRANCH_MS", 0.0) or 0.0)
            stage["directional_pair"] = float(globals().get("_E156_LAST_DIRECTIONAL_PAIR_MS", 0.0) or 0.0)
            stage["total"] = 1000.0 * (time.perf_counter() - t0)
            return {"status": "BLOCKED", "plan": None, "binding_id": None, "gate": gate,
                    "reason": str((depth_state or {}).get("reason", "E156_RGB_GEOMETRY_FAILED")),
                    "elapsed_s": float((time.perf_counter()-t0)), "stage_times_ms": stage,
                    "core_version": E156_CORE_VERSION}

        tj = time.perf_counter()
        primary_joint = _v6_joint_pose_rgb_localize(depth_state, mask, H_value, _E92_MICRO_ARGS)
        stage["joint_localization_primary"] = 1000.0 * (time.perf_counter() - tj)
        gate["joint_localization_primary"] = _training_json_safe(primary_joint)
        joint=primary_joint

        if not bool(primary_joint.get('ok',False)) and rescue_enabled:
            rr=time.perf_counter()
            rescue=_v8_rgb_adaptive_rescue_localize(depth_state,mask,H_value,_E92_MICRO_ARGS,primary_joint=primary_joint)
            stage['rgb_adaptive_rescue']=1000.0*(time.perf_counter()-rr)
            gate['rgb_adaptive_rescue']=_training_json_safe(rescue)
            if bool(rescue.get('ok',False)):
                joint=rescue
        stage["joint_localization"] = float(stage.get("joint_localization_primary",0.0)+stage.get('rgb_adaptive_rescue',0.0))
        gate["joint_localization"] = _training_json_safe(joint)

        if not bool(joint.get("ok", False)):
            stage["branch"] = float(globals().get("_E156_LAST_BRANCH_MS", 0.0) or 0.0)
            stage["directional_pair"] = float(globals().get("_E156_LAST_DIRECTIONAL_PAIR_MS", 0.0) or 0.0)
            stage["total"] = 1000.0 * (time.perf_counter() - t0)
            reason=str((gate.get('rgb_adaptive_rescue') or {}).get('reason') or primary_joint.get('reason') or 'V12_LOCALIZATION_FAILED')
            status='BLOCKED' if strict_gate_open else 'NOT_APPLICABLE'
            print(f"[E156-V12-LOCALIZATION-{status}] primary={primary_joint.get('reason')} rescue={reason} time={stage['joint_localization']:.1f}ms")
            return {"status": status, "plan": None, "binding_id": None, "gate": gate,
                    "reason": reason, "summary": None, "elapsed_s": float((time.perf_counter()-t0)),
                    "stage_times_ms": stage, "core_version": E156_CORE_VERSION}

        tp = time.perf_counter()
        full_plan = _fix53_plan_points(depth_state, np.asarray(H_value, dtype=np.float32), config, runtime_cfg, _E92_MICRO_ARGS)
        stage["plan_points"] = 1000.0 * (time.perf_counter() - tp)
        stage["branch"] = float(globals().get("_E156_LAST_BRANCH_MS", 0.0) or 0.0)
        stage["directional_pair"] = float(globals().get("_E156_LAST_DIRECTIONAL_PAIR_MS", 0.0) or 0.0)

        tf = time.perf_counter()
        execution_ok = bool(full_plan.get("execution_evidence_ok", True))
        role_ok, role_diag = _v7_validate_joint_plan_roles(full_plan, depth_state)
        if execution_ok and not role_ok:
            execution_ok = False
            full_plan["execution_evidence_ok"] = False
            full_plan["execution_evidence_reason"] = "V7_JOINT_FLAP_BODY_ROLE_MISMATCH:" + str(role_diag.get("reason", "UNKNOWN"))
        full_plan["vla_joint_localization_policy"] = str(joint.get("policy", "V8_RGB_POSE_OR_ADAPTIVE_RESCUE"))
        full_plan["vla_joint_pose_source"] = str(joint.get("pose_source", "NONE"))
        full_plan["vla_joint_pose_group"] = str(joint.get("pose_group", "NONE"))
        full_plan["vla_joint_pose_confidence"] = float(joint.get("pose_confidence", 0.0) or 0.0)
        full_plan["vla_joint_branch_ids"] = list(joint.get("kept_branch_ids", []) or [])
        full_plan["vla_joint_selected_branch_id"] = int(joint.get("selected_branch_id", -1))
        full_plan["vla_joint_branch_overlap"] = float(joint.get("branch_overlap", 0.0) or 0.0)
        full_plan["vla_joint_orientation_alignment"] = float(joint.get("orientation_alignment", 0.0) or 0.0)
        full_plan["vla_joint_branch_fold_score"] = float(joint.get("branch_fold_score", 0.0) or 0.0)
        full_plan["vla_joint_role_validation"] = _training_json_safe(role_diag)
        rgb_rescue=bool(joint.get('rgb_rescue_used',False) or depth_state.get('vla_joint_rgb_rescue',False))
        full_plan['vla_rgb_rescue_used']=bool(rgb_rescue)
        if rgb_rescue:
            full_plan['vla_rgb_rescue_mode']=str(depth_state.get('vla_rgb_rescue_mode',joint.get('rgb_rescue_mode','UNKNOWN')))
            full_plan['vla_rgb_rescue_confidence']=float(depth_state.get('vla_rgb_rescue_confidence',joint.get('joint_score',0.0)) or 0.0)
            full_plan['vla_rgb_rescue_exact_quality']=float(depth_state.get('vla_rgb_rescue_exact_quality',joint.get('rgb_rescue_exact_quality',0.0)) or 0.0)
            full_plan['vla_rgb_rescue_branch_score']=float(depth_state.get('vla_rgb_rescue_branch_score',joint.get('rgb_rescue_branch_score',0.0)) or 0.0)
            full_plan['vla_rgb_rescue_neck_distance_ratio']=float(depth_state.get('vla_rgb_rescue_neck_distance_ratio',joint.get('rgb_rescue_neck_distance_ratio',999.0)) or 0.0)
            full_plan['vla_rgb_rescue_neck_parallel_alignment']=float(depth_state.get('vla_rgb_rescue_neck_parallel_alignment',joint.get('rgb_rescue_neck_parallel_alignment',0.0)) or 0.0)
        if isinstance(direct_diag, dict):
            full_plan["v38_planner_source"] = "E156_FALLBACK"
            full_plan["v38_target_direct_attempted"] = True
            full_plan["v38_target_direct_used"] = False
            full_plan["v38_target_board"] = _training_json_safe(direct_diag.get("target_board"))
            full_plan["v38_fallback_used"] = True
            full_plan["v38_fallback_reason"] = str(direct_diag.get("reason") or "TARGET_DIRECT_BLOCKED")
        if isinstance(fast_diag,dict):
            full_plan["v41_fast_target_local_used"]=False
            full_plan["v41_fast_target_local_source"]=""
            full_plan["v41_fast_target_local_distance_mm"]=None
            full_plan["v41_fast_target_local_candidate_rank"]=0
        full_plan["state"] = "READY" if execution_ok else "DIAGNOSTIC"
        full_plan["e156_vla_core_version"] = E156_CORE_VERSION
        full_plan["e156_vla_gate"] = _training_json_safe({k:v for k,v in gate.items() if k != "pose_state"})
        summary = plan_summary(full_plan)
        frozen = freeze_execution_plan(full_plan) if execution_ok else None
        if execution_ok and not isinstance(frozen, dict):
            execution_ok = False
            full_plan["execution_evidence_ok"] = False
            full_plan["execution_evidence_reason"] = "V12_FROZEN_EXECUTION_CONTRACT_INCOMPLETE"
            summary = plan_summary(full_plan)
        bid = binding_id(frozen, generation=generation) if execution_ok else None
        stage["finalize"] = 1000.0 * (time.perf_counter() - tf)
        stage["total"] = 1000.0 * (time.perf_counter() - t0)
        status = "READY" if execution_ok else "BLOCKED"
        print(f"[E156-V12-LATENCY] {stage}")
        print(f"[E156-V12-FROZEN-SIZE] keys={0 if frozen is None else len(frozen)} fullMapsPersisted=NO")
        return {"status": status, "plan": frozen if execution_ok else None,
                "diagnostic_plan": None, "binding_id": bid, "gate": gate,
                "reason": str(full_plan.get("execution_evidence_reason", "OK" if execution_ok else "WEAK_EVIDENCE")),
                "summary": summary, "elapsed_s": float((time.perf_counter()-t0)),
                "stage_times_ms": stage, "core_version": E156_CORE_VERSION}
    except Exception as exc:
        stage["branch"] = float(globals().get("_E156_LAST_BRANCH_MS", 0.0) or 0.0)
        stage["directional_pair"] = float(globals().get("_E156_LAST_DIRECTIONAL_PAIR_MS", 0.0) or 0.0)
        stage["total"] = 1000.0 * (time.perf_counter() - t0)
        return {"status": "BLOCKED", "plan": None, "binding_id": None, "gate": gate,
                "reason": f"E156_PLAN_EXCEPTION:{exc!r}", "elapsed_s": float((time.perf_counter()-t0)),
                "stage_times_ms": stage, "core_version": E156_CORE_VERSION}


# ---- Exact E156 helper body begins ----

# =============================================================================
# FIX53 GARMENT-AGNOSTIC UNIDEPTH MICRO-UNFOLD
# I = fresh MASK+UniDepth plan only, X = execute iterative Z micro-unfold.
# Pose/class semantics are deliberately not used.
# =============================================================================
_FIX53_MICRO_UNFOLD_PLAN = None
_FIX53_MICRO_UNFOLD_STATUS = "IDLE"
# E110 display ownership is intentionally independent from whether a READY/DIAGNOSTIC
# plan already exists. I sets this before synchronous inference, so inherited D38/D42
# yellow-mask overlays cannot remain latched while the Micro-Unfold worker is busy.
_E110_MICRO_VIEW_ACTIVE = False
_E110_MICRO_VIEW_STATE = "IDLE"
# E102: these names must live inside the injected D50 module.  The wrapper-level
# FIX17 tuple is not visible to exec()'d helper functions at runtime.
_E102_TSHIRT_POSE_KPT_NAMES = (
    "shoulder_left", "neck_left", "neck_right", "shoulder_right",
    "sleeve_right_top", "sleeve_right_bottom", "armpit_right",
    "waist_right", "waist_left", "armpit_left",
    "sleeve_left_bottom", "sleeve_left_top",
)


def _fix84_robust_unit_map(values, domain):
    """Robust 0..1 normalization inside the exact garment domain."""
    arr = np.asarray(values, dtype=np.float32)
    dom = np.asarray(domain, dtype=bool) & np.isfinite(arr)
    out = np.zeros(arr.shape, dtype=np.float32)
    vals = arr[dom]
    if vals.size < 16:
        return out
    lo = float(np.percentile(vals, 20.0))
    hi = float(np.percentile(vals, 95.0))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo + 1e-6:
        lo = float(np.min(vals)); hi = float(np.max(vals))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo + 1e-6:
        return out
    out[dom] = np.clip((arr[dom] - lo) / (hi - lo), 0.0, 1.0)
    return out


def _fix86_top1_point_score(relief_mm, prominence_mm, top1_mask):
    """Pixel score only inside donor-validated strict TOP1; never creates candidates."""
    relief = np.asarray(relief_mm, dtype=np.float32)
    prom = np.asarray(prominence_mm, dtype=np.float32)
    dom = np.asarray(top1_mask, dtype=bool) & np.isfinite(relief) & np.isfinite(prom)
    score = np.zeros(relief.shape, dtype=np.float32)
    if int(np.count_nonzero(dom)) < 8:
        return score
    rn = _fix84_robust_unit_map(np.maximum(relief, 0.0), dom)
    pn = _fix84_robust_unit_map(np.maximum(prom, 0.0), dom)
    score[dom] = 0.60 * rn[dom] + 0.40 * pn[dom]
    return score.astype(np.float32)


def _fix89_connected_from_seed(domain_bool, seed_bool):
    """Keep only domain components that touch the strict UniDepth seed."""
    domain = (np.asarray(domain_bool, dtype=bool).astype(np.uint8) * 255)
    seed = np.asarray(seed_bool, dtype=bool)
    n, cc = cv2.connectedComponents((domain > 0).astype(np.uint8), connectivity=8)
    keep = np.zeros(seed.shape, dtype=bool)
    if n <= 1:
        return keep
    seed_touch = cv2.dilate((seed.astype(np.uint8) * 255), np.ones((7,7), np.uint8), iterations=1) > 0
    for lab in range(1, int(n)):
        comp = (cc == lab)
        if np.any(comp & seed_touch):
            keep |= comp
    return keep


def _fix90_contour_arc_indices(n, centre_idx, left_count, right_count):
    """Indices for the contour arc that runs left -> seed-nearest point -> right."""
    n = int(n)
    if n <= 2:
        return []
    left_count = int(np.clip(int(left_count), 1, max(1, n // 2 - 1)))
    right_count = int(np.clip(int(right_count), 1, max(1, n // 2 - 1)))
    return [int((centre_idx + off) % n) for off in range(-left_count, right_count + 1)]


def _fix90_mask_lobe_from_contour(garment_mask_u8, strict_seed_bool, relief_mm, prominence_mm, args=None):
    """Infer one plausible folded flap from MASK silhouette, then validate it with UniDepth.

    Design principle:
      - MASK is the boundary authority.  A local contour arc around the reliable
        depth seed is closed with a chord to form candidate flap/lobe polygons.
      - UniDepth does NOT define how far the flap grows.  It only selects/validates
        the MASK-defined lobe using seed coverage and candidate-vs-body height.
      - This prevents E91's failure mode where a positive-relief flood swallowed
        >50% of the garment and left no flat HOLD region.
    """
    garment = np.asarray(garment_mask_u8, dtype=np.uint8) > 0
    seed = np.asarray(strict_seed_bool, dtype=bool) & garment
    relief = np.asarray(relief_mm, dtype=np.float32)
    prom = np.asarray(prominence_mm, dtype=np.float32)
    info = {"available": False, "reason": "UNINITIALIZED"}
    empty = np.zeros(garment.shape, dtype=bool)
    seed_area = int(np.count_nonzero(seed))
    garment_area = int(np.count_nonzero(garment))
    if seed_area < 8 or garment_area < 500:
        info.update(reason="SEED_OR_GARMENT_TOO_SMALL", seed_area_px=seed_area, garment_area_px=garment_area)
        return empty, info

    contours, _ = cv2.findContours((garment.astype(np.uint8) * 255), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        info.update(reason="NO_GARMENT_CONTOUR")
        return empty, info
    contour = max(contours, key=cv2.contourArea).reshape(-1, 2)
    n = int(contour.shape[0])
    if n < 40:
        info.update(reason="CONTOUR_TOO_SHORT", contour_points=n)
        return empty, info

    sy, sx = np.where(seed)
    seed_centroid = np.asarray([float(np.mean(sx)), float(np.mean(sy))], dtype=np.float64)
    d2 = np.sum((contour.astype(np.float64) - seed_centroid[None, :]) ** 2, axis=1)
    k = int(np.argmin(d2))
    seed_boundary_dist = float(np.sqrt(max(0.0, float(d2[k]))))

    # If the reliable depth seed is deep inside the garment, the outer silhouette
    # cannot safely define the fold boundary.  Let the caller use depth-seed fallback.
    max_seed_boundary_px = float(getattr(args, "micro_unfold_mask_lobe_max_seed_boundary_px", 115.0)) if args is not None else 115.0
    if seed_boundary_dist > max_seed_boundary_px:
        info.update(
            reason="SEED_TOO_FAR_FROM_MASK_CONTOUR",
            seed_boundary_distance_px=seed_boundary_dist,
            max_seed_boundary_distance_px=max_seed_boundary_px,
            contour_points=n,
        )
        return empty, info

    finite_g = garment & np.isfinite(relief) & np.isfinite(prom)
    seed_rel = np.maximum(relief[seed & np.isfinite(relief)], 0.0)
    seed_p95 = float(np.percentile(seed_rel, 95.0)) if seed_rel.size else 0.0
    relief_gate = max(3.0, 0.22 * max(seed_p95, 1.0))
    min_ratio = float(getattr(args, "micro_unfold_mask_lobe_min_ratio", 0.025)) if args is not None else 0.025
    max_ratio = float(getattr(args, "micro_unfold_mask_lobe_max_ratio", 0.42)) if args is not None else 0.42
    min_seed_cov = float(getattr(args, "micro_unfold_mask_lobe_min_seed_coverage", 0.45)) if args is not None else 0.45
    min_depth_contrast = float(getattr(args, "micro_unfold_mask_lobe_min_depth_contrast_mm", 1.0)) if args is not None else 1.0

    # CHAIN_APPROX_NONE is roughly one contour pixel per sample.  Generate both
    # symmetric and asymmetric arc/chord caps because real folds are triangular.
    fracs = (0.035, 0.050, 0.070, 0.095, 0.125, 0.165, 0.215)
    offsets = sorted({max(8, int(round(n * f))) for f in fracs if int(round(n * f)) < n // 2})
    candidates = []
    seed_dil = cv2.dilate((seed.astype(np.uint8) * 255), np.ones((5, 5), np.uint8), iterations=1) > 0

    for left_count in offsets:
        for right_count in offsets:
            if left_count + right_count >= int(0.48 * n):
                continue
            idx = _fix90_contour_arc_indices(n, k, left_count, right_count)
            if len(idx) < 6:
                continue
            arc = contour[np.asarray(idx, dtype=np.int32)]
            poly = np.asarray(arc, dtype=np.int32).reshape(-1, 1, 2)
            candidate_u8 = np.zeros(garment.shape, dtype=np.uint8)
            cv2.fillPoly(candidate_u8, [poly], 255)
            cand = (candidate_u8 > 0) & garment
            area = int(np.count_nonzero(cand))
            if area < max(80, int(0.55 * seed_area)):
                continue
            ratio = float(area) / float(max(1, garment_area))
            if ratio < min_ratio or ratio > max_ratio:
                continue
            overlap = int(np.count_nonzero(cand & seed_dil))
            seed_cov = min(1.0, float(overlap) / float(max(1, seed_area)))
            if seed_cov < min_seed_cov:
                continue

            valid_c = cand & finite_g
            valid_body = garment & (~cand) & finite_g
            if int(np.count_nonzero(valid_c)) < 20 or int(np.count_nonzero(valid_body)) < 50:
                continue
            c_rel = np.maximum(relief[valid_c], 0.0)
            b_rel = np.maximum(relief[valid_body], 0.0)
            c_prom = np.maximum(prom[valid_c], 0.0)
            cand_rel_p70 = float(np.percentile(c_rel, 70.0))
            body_rel_med = float(np.median(b_rel))
            depth_contrast = cand_rel_p70 - body_rel_med
            raised_fraction = float(np.mean(c_rel >= relief_gate)) if c_rel.size else 0.0
            prom_p70 = float(np.percentile(c_prom, 70.0)) if c_prom.size else 0.0

            left_pt = arc[0].astype(np.float64)
            right_pt = arc[-1].astype(np.float64)
            chord = float(np.linalg.norm(right_pt - left_pt))
            arc_steps = np.linalg.norm(np.diff(arc.astype(np.float64), axis=0), axis=1)
            arc_len = float(np.sum(arc_steps))
            arc_chord_ratio = arc_len / max(1.0, chord)

            # UniDepth validates the MASK lobe. Very weak height contrast is a
            # rejection, not an invitation to expand farther.
            if depth_contrast < min_depth_contrast and raised_fraction < 0.18:
                continue
            depth_norm = float(np.clip(depth_contrast / max(8.0, 0.35 * max(seed_p95, 8.0)), 0.0, 1.8))
            prom_norm = float(np.clip(prom_p70 / max(1.0, 0.15 * max(seed_p95, 5.0)), 0.0, 1.5))
            # Strong seed containment and depth validation dominate.  A mild
            # area penalty favours the local folded flap over the main body.
            score = (
                4.2 * seed_cov
                + 1.7 * depth_norm
                + 1.3 * raised_fraction
                + 0.35 * min(2.5, arc_chord_ratio)
                + 0.20 * prom_norm
                - 1.6 * ratio
            )
            candidates.append({
                "score": float(score),
                "mask": cand,
                "area_px": area,
                "garment_ratio": ratio,
                "seed_coverage": seed_cov,
                "depth_contrast_mm": float(depth_contrast),
                "raised_fraction": float(raised_fraction),
                "prominence_p70_mm": float(prom_p70),
                "arc_chord_ratio": float(arc_chord_ratio),
                "left_count": int(left_count),
                "right_count": int(right_count),
                "left_px": [int(left_pt[0]), int(left_pt[1])],
                "right_px": [int(right_pt[0]), int(right_pt[1])],
            })

    if not candidates:
        info.update(
            reason="NO_DEPTH_VALIDATED_MASK_LOBE",
            seed_area_px=seed_area,
            garment_area_px=garment_area,
            seed_boundary_distance_px=seed_boundary_dist,
            relief_gate_mm=float(relief_gate),
            candidate_trials=int(len(offsets) * len(offsets)),
        )
        return empty, info

    candidates.sort(key=lambda item: float(item["score"]), reverse=True)
    best = candidates[0]
    best_mask = np.asarray(best.pop("mask"), dtype=bool)
    # Close only tiny raster gaps. Never flood by depth outside MASK geometry.
    best_u8 = cv2.morphologyEx((best_mask.astype(np.uint8) * 255), cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    best_mask = (best_u8 > 0) & garment
    best_area = int(np.count_nonzero(best_mask))
    best_ratio = float(best_area) / float(max(1, garment_area))
    info.update(
        available=True,
        reason="MASK_LOBE_DEPTH_VALIDATED",
        contour_points=n,
        seed_boundary_distance_px=float(seed_boundary_dist),
        seed_area_px=seed_area,
        garment_area_px=garment_area,
        relief_gate_mm=float(relief_gate),
        fold_area_px=best_area,
        fold_garment_ratio=best_ratio,
        candidate_count=int(len(candidates)),
        best=best,
        alternatives=[{k: v for k, v in item.items() if k != "mask"} for item in candidates[1:4]],
    )
    return best_mask, info


def _fix90_fuse_mask_unidepth(relief_mm, prominence_mm, garment_mask_u8, strict_seed_bool, args=None):
    """MASK boundary first, UniDepth height validation second; conservative fallback."""
    garment = np.asarray(garment_mask_u8, dtype=np.uint8) > 0
    seed = np.asarray(strict_seed_bool, dtype=bool) & garment
    geom, geom_info = _fix90_mask_lobe_from_contour(
        garment_mask_u8, seed, relief_mm, prominence_mm, args=args
    )
    if bool((geom_info or {}).get("available", False)) and int(np.count_nonzero(geom)) >= 8:
        return geom, {
            "fusion_mode": "MASK_LOBE_X_UNIDEPTH",
            "mask_geometry_available": True,
            "mask_geometry": geom_info,
            "fold_area_px": int(np.count_nonzero(geom)),
            "fold_garment_ratio": float(np.count_nonzero(geom)) / float(max(1, np.count_nonzero(garment))),
        }, geom

    # Internal folds or weak silhouette cues: preserve donor reliability but do
    # not repeat E91's unconstrained connected relief flood.  A tiny morphology
    # close/dilate makes the seed graspable while remaining local.
    seed_u8 = (seed.astype(np.uint8) * 255)
    if cv2.countNonZero(seed_u8) > 0:
        local = cv2.morphologyEx(seed_u8, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
        local = cv2.dilate(local, np.ones((5, 5), np.uint8), iterations=1)
        fallback = (local > 0) & garment
    else:
        fallback = seed
    return fallback, {
        "fusion_mode": "UNIDEPTH_SEED_FALLBACK",
        "mask_geometry_available": False,
        "mask_geometry": geom_info,
        "fold_area_px": int(np.count_nonzero(fallback)),
        "fold_garment_ratio": float(np.count_nonzero(fallback)) / float(max(1, np.count_nonzero(garment))),
    }, geom

def _e94_robust_unit_map(values, domain_bool, lo_pct=45.0, hi_pct=98.0):
    arr = np.asarray(values, dtype=np.float32)
    domain = np.asarray(domain_bool, dtype=bool) & np.isfinite(arr)
    out = np.zeros(arr.shape, dtype=np.float32)
    vals = arr[domain]
    if vals.size < 16:
        return out
    lo = float(np.percentile(vals, float(lo_pct)))
    hi = float(np.percentile(vals, float(hi_pct)))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo + 1e-6:
        return out
    out[domain] = np.clip((arr[domain] - lo) / (hi - lo), 0.0, 1.0)
    return out


def _e94_skeletonize(binary_u8):
    """Small OpenCV-only morphology skeleton; avoids an ximgproc dependency."""
    img = ((np.asarray(binary_u8, dtype=np.uint8) > 0).astype(np.uint8) * 255)
    skel = np.zeros_like(img)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    for _ in range(128):
        eroded = cv2.erode(img, element)
        opened = cv2.dilate(eroded, element)
        edge = cv2.subtract(img, opened)
        skel = cv2.bitwise_or(skel, edge)
        img = eroded
        if cv2.countNonZero(img) == 0:
            break
    return skel


def _e101_pose_point(pose_state, name, min_conf=0.20):
    if not isinstance(pose_state, dict):
        return None
    p = dict((pose_state.get("named_points", {}) or {}).get(str(name), {}) or {})
    try:
        c = float(p.get("conf", 0.0)); x = float(p.get("x")); y = float(p.get("y"))
    except Exception:
        return None
    if not (np.isfinite(x) and np.isfinite(y) and np.isfinite(c)) or c < float(min_conf):
        return None
    return np.asarray([x, y], dtype=np.float64)


def _e102_pose_state_crop_tta(frame_bgr, garment_mask_u8, H_value=None, runtime_config=None):
    """Recover T-shirt pose for seam veto when the legacy full-frame call returns zero instances.

    The pose engine may be sensitive to garment scale/orientation.  E102 first reuses the
    existing production runner on the unchanged frozen I frame.  If that yields fewer than
    two semantic points, it retries a tight garment crop at 0/180/90CW/90CCW and maps all
    recovered keypoints back to the original 1280x720 coordinates.  This is seam-veto only;
    it never changes segmentation, Homography, or robot motion by itself.
    """
    runner = globals().get("_FIX21_RUN_A150_TSHIRT_POSE_ON_FRAME")
    getter = globals().get("_FIX21_GET_A150_TSHIRT_POSE_STATE")
    if not callable(runner) or not callable(getter):
        return {"valid_count":0,"named_points":{},"points":[],"status":"POSE_RUNNER_UNAVAILABLE","tta_mode":"NONE"}
    frame=np.asarray(frame_bgr,dtype=np.uint8)
    mask=((np.asarray(garment_mask_u8,dtype=np.uint8)>0).astype(np.uint8)*255)
    h,w=frame.shape[:2]

    def score_state(st):
        named=dict((st or {}).get("named_points",{}) or {})
        vc=int((st or {}).get("valid_count",len(named)) or 0)
        conf=sum(float((p or {}).get("conf",0.0) or 0.0) for p in named.values())
        # Neck/shoulder/armpit points are especially valuable for seam veto.
        semantic=sum(1 for k in ("neck_left","neck_right","shoulder_left","shoulder_right","armpit_left","armpit_right") if k in named)
        return (semantic,vc,conf)

    def mapped_state(st, mode, x0, y0, cw, ch):
        st=dict(st or {})
        named_src=dict(st.get("named_points",{}) or {})
        pts_src=list(st.get("points",[]) or [])
        raw_named_src=dict(st.get("raw_named_points",{}) or {})
        raw_pts_src=list(st.get("raw_points",[]) or [])
        def inv_xy(xp,yp):
            if mode=="CROP0": x,y=xp,yp
            elif mode=="CROP180": x,y=(cw-1.0-xp),(ch-1.0-yp)
            elif mode=="CROP90CW": x,y=yp,(ch-1.0-xp)
            elif mode=="CROP90CCW": x,y=(cw-1.0-yp),xp
            else: x,y=xp,yp
            return float(x+x0),float(y+y0)
        named={}
        for k,p0 in named_src.items():
            try:
                p=dict(p0); x,y=inv_xy(float(p["x"]),float(p["y"])); p["x"]=x; p["y"]=y
                if 0.0<=x<float(w) and 0.0<=y<float(h): named[str(k)]=p
            except Exception: pass
        points=[]
        for p0 in pts_src:
            try:
                p=dict(p0); x,y=inv_xy(float(p["x"]),float(p["y"])); p["x"]=x; p["y"]=y
                if 0.0<=x<float(w) and 0.0<=y<float(h): points.append(p)
            except Exception: pass
        raw_named={}
        for k,p0 in raw_named_src.items():
            try:
                p=dict(p0); x,y=inv_xy(float(p["x"]),float(p["y"])); p["x"]=x; p["y"]=y
                if 0.0<=x<float(w) and 0.0<=y<float(h): raw_named[str(k)]=p
            except Exception: pass
        raw_points=[]
        for p0 in raw_pts_src:
            try:
                p=dict(p0); x,y=inv_xy(float(p["x"]),float(p["y"])); p["x"]=x; p["y"]=y
                if 0.0<=x<float(w) and 0.0<=y<float(h): raw_points.append(p)
            except Exception: pass
        st["named_points"]=named; st["points"]=points; st["valid_count"]=len(named)
        st["raw_named_points"]=raw_named; st["raw_points"]=raw_points
        st["expected_count"]=12; st["status"]="POSE_TTA_RECOVERED" if named else "POSE_TTA_EMPTY"
        st["tta_mode"]=mode; st["roi_mode"]="GARMENT_CROP_TTA"
        return st

    # First preserve the exact production/full-frame attempt.
    try:
        runner(frame.copy(), H_value, runtime_config)
        base=dict(getter() or {})
    except Exception as exc:
        base={"valid_count":0,"named_points":{},"points":[],"status":f"FULL_POSE_ERROR:{exc!r}"}
    if int(base.get("valid_count",0) or 0)>=2:
        base["tta_mode"]="FULL"
        return base

    nz=cv2.findNonZero(mask)
    if nz is None:
        return base
    x,y,bw,bh=cv2.boundingRect(nz)
    margin=max(24,int(round(0.10*max(bw,bh))))
    x0=max(0,x-margin); y0=max(0,y-margin); x1=min(w,x+bw+margin); y1=min(h,y+bh+margin)
    crop=frame[y0:y1,x0:x1].copy(); ch,cw=crop.shape[:2]
    if cw<32 or ch<32:
        return base
    trials=[
        ("CROP0",crop),
        ("CROP180",cv2.rotate(crop,cv2.ROTATE_180)),
        ("CROP90CW",cv2.rotate(crop,cv2.ROTATE_90_CLOCKWISE)),
        ("CROP90CCW",cv2.rotate(crop,cv2.ROTATE_90_COUNTERCLOCKWISE)),
    ]
    best=base; best_score=score_state(base)
    for mode,img in trials:
        try:
            runner(img.copy(), None, runtime_config)
            st=mapped_state(getter(),mode,x0,y0,cw,ch)
            sc=score_state(st)
            print(f"[E102-POSE-TTA] mode={mode} valid={int(st.get('valid_count',0))}/12 semantic={sc[0]} confSum={sc[2]:.2f}")
            if sc>best_score:
                best,best_score=st,sc
            # Four useful semantic points are sufficient; avoid needless TensorRT passes.
            if sc[0]>=4 and sc[1]>=4:
                break
        except Exception as exc:
            print(f"[E102-POSE-TTA-WARN] mode={mode} error={exc!r}")
    if int(best.get("valid_count",0) or 0)>0:
        print(f"[E102-POSE-TTA-SELECT] mode={best.get('tta_mode','FULL')} valid={int(best.get('valid_count',0))}/12")
    else:
        print("[E102-POSE-TTA-SELECT] no pose recovered; RGB crease continues without semantic seam veto")
    return best


def _e101_quad_bezier(p0, pc, p1, samples=48):
    p0=np.asarray(p0,dtype=np.float64); pc=np.asarray(pc,dtype=np.float64); p1=np.asarray(p1,dtype=np.float64)
    t=np.linspace(0.0,1.0,max(8,int(samples)),dtype=np.float64)[:,None]
    pts=(1.0-t)**2*p0[None,:]+2.0*(1.0-t)*t*pc[None,:]+t**2*p1[None,:]
    return np.rint(pts).astype(np.int32).reshape(-1,1,2)


def _e101_pose_semantic_masks(image_shape, garment_mask_u8, pose_state, args=None):
    """Build pose-derived regions that are *not* fold creases.

    The T-shirt pose provides semantic keypoints for neck, shoulders, sleeve edges,
    armpits and waist. E118 hard-vetoes ONLY the neckline. Shoulder/armhole and
    sleeve-hem sewing lines are deliberately left visible to the Fold Crease detector,
    because a real upper/sleeve fold can overlap those structures. The remaining pose
    landmarks are still used to build a torso BODY-core and later pose-count split.
    """
    h,w=image_shape[:2]
    garment=((np.asarray(garment_mask_u8,dtype=np.uint8)>0).astype(np.uint8)*255)
    seam=np.zeros((h,w),dtype=np.uint8)
    body_core=np.zeros((h,w),dtype=np.uint8)
    if not isinstance(pose_state,dict) or int(pose_state.get('valid_count',0) or 0)<2:
        return {"available":False,"reason":"POSE_UNAVAILABLE","seam_mask":seam,"body_core_mask":body_core,"segments":[]}
    min_conf=float(getattr(args,'micro_unfold_pose_seam_kpt_conf',0.20)) if args is not None else 0.20
    soft_neck_conf=float(getattr(args,'micro_unfold_neck_soft_veto_kpt_conf',0.08)) if args is not None else 0.08
    radius=max(6,int(round(float(getattr(args,'micro_unfold_pose_seam_radius_px',18.0)) if args is not None else 18.0)))
    pts={name:_e101_pose_point(pose_state,name,min_conf) for name in _E102_TSHIRT_POSE_KPT_NAMES}
    raw_named=dict((pose_state.get('raw_named_points',{}) or {})) if isinstance(pose_state,dict) else {}
    segs=[]
    def draw_curve(label,p0,pc,p1,rad):
        if p0 is None or pc is None or p1 is None: return
        poly=_e101_quad_bezier(p0,pc,p1,52)
        cv2.polylines(seam,[poly],False,255,max(3,int(2*rad)),cv2.LINE_AA)
        segs.append(label)
    def draw_line(label,p0,p1,rad):
        if p0 is None or p1 is None: return
        cv2.line(seam,tuple(np.rint(p0).astype(int)),tuple(np.rint(p1).astype(int)),255,max(3,int(2*rad)),cv2.LINE_AA)
        segs.append(label)

    nl,nr=pts.get('neck_left'),pts.get('neck_right')
    wl,wr=pts.get('waist_left'),pts.get('waist_right')
    al,ar=pts.get('armpit_left'),pts.get('armpit_right')
    sl,sr=pts.get('shoulder_left'),pts.get('shoulder_right')
    neck_veto_mode='HARD' if (nl is not None and nr is not None) else 'NONE'

    # E112: low-confidence neck points are allowed ONLY as a semantic seam veto.
    # They are never copied into `pts`, never create BODY-core geometry and never
    # authorize HOLD/ACTIVE coordinates. This specifically covers the recurrent
    # 0.08~0.20 neck confidences observed when CROP90CCW recovers the shirt but
    # the production 0.20 keypoint gate drops the collar endpoints.
    if nl is None or nr is None:
        def _raw_neck(name):
            q=dict(raw_named.get(name,{}) or {})
            try:
                x=float(q.get('x')); y=float(q.get('y')); c=float(q.get('conf',0.0))
            except Exception:
                return None,None
            if not (np.isfinite(x) and np.isfinite(y) and np.isfinite(c)) or c < soft_neck_conf:
                return None,None
            xi=int(np.clip(round(x),0,w-1)); yi=int(np.clip(round(y),0,h-1))
            gd=cv2.dilate(garment,np.ones((9,9),np.uint8),iterations=1)
            if gd[yi,xi] <= 0:
                return None,None
            return np.asarray([x,y],dtype=np.float64),c
        snl,cnl=_raw_neck('neck_left'); snr,cnr=_raw_neck('neck_right')
        if snl is not None and snr is not None:
            sep=float(np.linalg.norm(snr-snl))
            nz=cv2.findNonZero(garment)
            bw=float(cv2.boundingRect(nz)[2]) if nz is not None else float(w)
            plausible=bool(12.0 <= sep <= max(34.0,0.48*bw))
            # If reliable shoulders exist, the neck must remain narrower than the shoulder span.
            if plausible and sl is not None and sr is not None:
                shoulder_sep=float(np.linalg.norm(sr-sl))
                plausible=bool(shoulder_sep>1e-6 and sep <= 0.82*shoulder_sep)
            if plausible:
                nl,nr=snl,snr
                neck_veto_mode='SOFT'
                print(f"[E112-NECK-SOFT-VETO] accepted neck_left={cnl:.3f} neck_right={cnr:.3f} sep={sep:.1f}px threshold={soft_neck_conf:.2f} | seam-veto-only")

    if nl is not None and nr is not None:
        nm=0.5*(nl+nr)
        body_ref=None
        if wl is not None and wr is not None: body_ref=0.5*(wl+wr)
        elif al is not None and ar is not None: body_ref=0.5*(al+ar)
        elif neck_veto_mode=='SOFT':
            gm=cv2.moments(garment,binaryImage=True)
            if abs(float(gm.get('m00',0.0)))>1e-6:
                body_ref=np.asarray([gm['m10']/gm['m00'],gm['m01']/gm['m00']],dtype=np.float64)
        if body_ref is not None:
            d=body_ref-nm; dn=float(np.linalg.norm(d))
            if dn>1e-6:
                d=d/dn; width=float(np.linalg.norm(nr-nl))
                # Collar stitching is not one exact mathematical arc.  Cover a
                # narrow family of body-side semicircular curves so the real
                # neckline seam is vetoed even when the pose endpoints are a few
                # pixels off or the shirt is perspectively distorted.
                # E119: keep a neckline veto, but make it deliberately narrow.
                # E118's five deep body-side arcs could erase a real upper/sleeve
                # fold that happened to pass near the collar/shoulder.
                for j,frac in enumerate((0.18,0.30,0.42)):
                    ctrl=nm+d*np.clip(frac*width,8.0,62.0)
                    draw_curve('NECKLINE' if j==1 else f'NECKLINE_BAND_{j}',nl,ctrl,nr,max(6,int(round(radius*0.62))))
                draw_line('NECKLINE_CHORD',nl,nr,max(5,int(round(radius*0.52))))
                # The collar seam can be much broader than one mathematical arc,
                # especially after perspective distortion.  Build a pose-semantic
                # body-side collar zone and veto *all* crease evidence in it.
                # This is intentionally semantic: a true fold elsewhere is not
                # removed, while the known neckline construction seam is.
                e=(nr-nl)/max(1e-6,width)
                n=d
                # E119 narrowed semantic collar zone.  It still removes the
                # neckline itself, but no longer blankets most of the upper torso.
                half=0.56*width
                body_depth=np.clip(0.42*width,24.0,62.0)
                back_depth=np.clip(0.07*width,5.0,12.0)
                zone=np.asarray([
                    nm-e*half-n*back_depth,
                    nm+e*half-n*back_depth,
                    nm+e*half+n*body_depth,
                    nm-e*half+n*body_depth,
                ],dtype=np.float64)
                cv2.fillConvexPoly(seam,np.rint(zone).astype(np.int32),255,cv2.LINE_AA)
                segs.append('NECKLINE_SOFT_VETO_ZONE' if neck_veto_mode=='SOFT' else 'NECKLINE_SEMANTIC_ZONE')
        else:
            draw_line('NECKLINE_CHORD',nl,nr,radius)

    # E118: user explicitly cancelled the shoulder/armhole/sleeve sewing-line veto.
    # Do NOT draw ARMHOLE_* or SLEEVE_HEM_* into the exclusion mask. This is crucial
    # for upper/sleeve folds where the real fold boundary overlaps or runs close to
    # those construction seams. Neckline exclusion above remains unchanged.
    # Torso semantic core for BODY/FLAP orientation.
    core_pts=[]
    for name in ('shoulder_left','shoulder_right','armpit_right','waist_right','waist_left','armpit_left'):
        p=pts.get(name)
        if p is not None: core_pts.append(p)
    if len(core_pts)>=4:
        hull=cv2.convexHull(np.rint(np.asarray(core_pts,dtype=np.float64)).astype(np.int32))
        cv2.fillConvexPoly(body_core,hull,255,cv2.LINE_AA)
        body_core=cv2.bitwise_and(body_core,garment)
        if cv2.countNonZero(body_core)>100:
            body_core=cv2.erode(body_core,np.ones((9,9),np.uint8),iterations=1)

    # Restrict seam veto to garment vicinity; no pose-derived mask can create a fold.
    gar_dil=cv2.dilate(garment,np.ones((7,7),np.uint8),iterations=1)
    seam=cv2.bitwise_and(seam,gar_dil)
    return {
        "available":bool(len(segs)>0),"reason":"POSE_SEAM_GEOMETRY" if segs else "POSE_KEYPOINTS_INSUFFICIENT",
        "seam_mask":seam,"body_core_mask":body_core,"segments":segs,"radius_px":float(radius),
        "pose_valid_count":int(pose_state.get('valid_count',0) or 0),"pose_status":str(pose_state.get('status','')),
        "neck_veto_mode":str(neck_veto_mode),"neck_soft_conf_threshold":float(soft_neck_conf),
    }



def _e126_cleanup_micro_garment_mask(mask_u8):
    """LEGACY E126 helper; E127 AUTO never calls this function.

    Historical E126 contour stabilizer.

    The inherited segmentation/fallback algorithm itself is unchanged.  This only
    removes thin foreground tails that are connected to the garment by a very narrow
    bridge (common with empty-board fallback reflections/nearby foreground).  A large
    change is rejected automatically so sleeves/true cloth are not silently clipped.
    """
    raw=((np.asarray(mask_u8,dtype=np.uint8)>0).astype(np.uint8)*255)
    raw_area=int(cv2.countNonZero(raw))
    if raw_area<500:
        return raw, {'accepted':False,'reason':'TOO_SMALL','raw_area':raw_area,'clean_area':raw_area,'removed_ratio':0.0}
    ker11=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(11,11))
    core=cv2.morphologyEx(raw,cv2.MORPH_OPEN,ker11,iterations=1)
    n,lab,stats,_=cv2.connectedComponentsWithStats((core>0).astype(np.uint8),8)
    if n<=1:
        return raw, {'accepted':False,'reason':'NO_CORE_COMPONENT','raw_area':raw_area,'clean_area':raw_area,'removed_ratio':0.0}
    lid=1+int(np.argmax(stats[1:,cv2.CC_STAT_AREA]))
    main=(lab==lid).astype(np.uint8)*255
    # Restore original contour detail only near the robust main core.
    support=cv2.dilate(main,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(15,15)),iterations=1)
    clean=cv2.bitwise_and(raw,support)
    clean=cv2.morphologyEx(clean,cv2.MORPH_CLOSE,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5)),iterations=1)
    clean_area=int(cv2.countNonZero(clean))
    removed=max(0.0,float(raw_area-clean_area)/float(max(1,raw_area)))
    # A contour cleanup is allowed to remove only a small appendage.  If it needs
    # to delete a large fraction, keep the original mask and let the normal sanity
    # gates handle the frame instead.
    accepted=bool(clean_area>=500 and 0.0<=removed<=0.14 and clean_area<=raw_area*1.02)
    return (clean if accepted else raw), {
        'accepted':accepted,'reason':'THIN_TAIL_REMOVED' if accepted and removed>0.002 else ('NO_SIGNIFICANT_CHANGE' if accepted else 'CHANGE_TOO_LARGE'),
        'raw_area':raw_area,'clean_area':clean_area,'removed_ratio':removed,
    }


def _e117_pose_fold_intrusion_state(image_shape, garment_mask_u8, pose_state, args=None):
    """E125 pose-first folded-part prior from contour intrusion of garment landmarks.

    Most learned boundary points represent semantic garment boundaries.  If a point
    that should remain close to a *current mask boundary* moves unusually deep into
    the cloth, the anatomical part is likely folded inward.  E148 explicitly excludes
    ARMPIT_RIGHT/ARMPIT_LEFT from this intrusion test because the natural armhole
    concavity makes their contour depth non-diagnostic even on a flat shirt.  Unlike E117, NECK_LEFT and
    NECK_RIGHT are included.  This is safe because the distance transform measures
    distance to *any* mask background (including a visible neck opening), not only
    to the external silhouette: a normal open neckline stays close to background,
    while an occluded/folded neckline can become deeply embedded in the mask.

    The returned ``fold_region_mask`` is a coarse manipulation ROI between each
    intruded landmark and its nearest current cloth boundary.  It is not itself a
    Fold-Crease and cannot bypass robot reach/path/Z safety.
    """
    h, w = image_shape[:2]
    garment = ((np.asarray(garment_mask_u8, dtype=np.uint8) > 0).astype(np.uint8) * 255)
    corridor = np.zeros((h, w), dtype=np.uint8)
    empty_region = np.zeros((h, w), dtype=np.uint8)
    if cv2.countNonZero(garment) < 500 or not isinstance(pose_state, dict):
        return {
            "available": False, "reason": "POSE_OR_MASK_UNAVAILABLE",
            "corridor_mask": corridor, "fold_region_mask": empty_region,
            "intrusions": [], "count": 0, "confidence": 0.0,
            "primary_group": "NONE", "group_scores": {},
        }

    named = dict(pose_state.get("named_points", {}) or {})
    inside = cv2.distanceTransform(garment, cv2.DIST_L2, 5).astype(np.float32)
    # E127 pose-analysis contour policy: use the E123-raw authoritative mask; non-neck landmarks use its largest external contour.
    # E125 used RETR_LIST for every landmark, so tiny internal mask holes/noise could
    # become the "nearest cloth contour" and bend the pose-fold corridor/region.
    # Non-neck landmarks now use only the largest external silhouette.  Neck points
    # may still use an internal collar/background boundary because a real neckline
    # opening is semantically meaningful.
    outer_contours, _ = cv2.findContours(garment, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not outer_contours:
        return {
            "available": False, "reason": "NO_CLOTH_CONTOUR",
            "corridor_mask": corridor, "fold_region_mask": empty_region,
            "intrusions": [], "count": 0, "confidence": 0.0,
            "primary_group": "NONE", "group_scores": {},
        }
    outer = max(outer_contours, key=cv2.contourArea).reshape(-1, 2).astype(np.float64)
    all_contours, _ = cv2.findContours(garment, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    all_parts = [c.reshape(-1, 2).astype(np.float64) for c in all_contours if len(c) >= 2 and cv2.contourArea(c) >= 18.0]
    all_boundary_pts = np.concatenate(all_parts, axis=0) if all_parts else outer.copy()

    conf_min = float(getattr(args, "micro_unfold_pose_fold_kpt_conf", 0.20)) if args is not None else 0.20
    edge_thr = float(getattr(args, "micro_unfold_pose_fold_inside_px", 22.0)) if args is not None else 22.0
    armpit_thr = float(getattr(args, "micro_unfold_pose_fold_armpit_inside_px", 30.0)) if args is not None else 30.0
    neck_thr = float(getattr(args, "micro_unfold_e125_pose_neck_inside_px", 18.0)) if args is not None else 18.0
    corridor_w = max(5, int(round(float(getattr(args, "micro_unfold_pose_fold_corridor_width_px", 15.0)) if args is not None else 15.0)))

    expected = {
        "shoulder_left": edge_thr, "shoulder_right": edge_thr,
        "sleeve_left_top": edge_thr, "sleeve_left_bottom": edge_thr,
        "sleeve_right_top": edge_thr, "sleeve_right_bottom": edge_thr,
        "waist_left": edge_thr, "waist_right": edge_thr,
        "armpit_left": armpit_thr, "armpit_right": armpit_thr,
        # E125: neckline can itself be folded.  A visible collar opening is mask
        # background and therefore keeps normal neck points close to a boundary.
        "neck_left": neck_thr, "neck_right": neck_thr,
    }
    group_of = {
        "neck_left": "NECK", "neck_right": "NECK",
        "shoulder_left": "UPPER_LEFT", "sleeve_left_top": "UPPER_LEFT",
        "sleeve_left_bottom": "UPPER_LEFT", "armpit_left": "UPPER_LEFT",
        "shoulder_right": "UPPER_RIGHT", "sleeve_right_top": "UPPER_RIGHT",
        "sleeve_right_bottom": "UPPER_RIGHT", "armpit_right": "UPPER_RIGHT",
        "waist_left": "LOWER_LEFT", "waist_right": "LOWER_RIGHT",
    }

    # E148: armpit landmarks are naturally concave garment landmarks.  Even on a
    # fully spread T-shirt they can sit well inside the external Cloth Contour, so
    # depth-to-contour is not valid Fold evidence for ID6/ID9.  They remain in the
    # pose state and are still used downstream by TORSO_QUAD / left-right
    # asymmetry / chain geometry; only POSE_INTRUSION admission is forbidden here.
    _e148_intrusion_excluded = {"armpit_right", "armpit_left"}

    intrusions = []
    group_items = {}
    group_masks = {}
    for name, threshold in expected.items():
        q = dict(named.get(name, {}) or {})
        try:
            x = float(q.get("x")); y = float(q.get("y")); conf = float(q.get("conf", 0.0))
        except Exception:
            continue
        if not (np.isfinite(x) and np.isfinite(y) and np.isfinite(conf)) or conf < conf_min:
            continue
        xi = int(np.clip(round(x), 0, w - 1)); yi = int(np.clip(round(y), 0, h - 1))
        if garment[yi, xi] <= 0:
            continue
        dist = float(inside[yi, xi])
        if name in _e148_intrusion_excluded:
            print(
                f"[E148-POSE-INTRUSION-SKIP] kpt={name} inside={dist:.1f}px "
                f"reason=NATURAL_ARMPIT_CONCAVITY; retained_for=TORSO_QUAD/ASYMMETRY"
            )
            continue
        if dist <= float(threshold):
            continue
        p = np.asarray([x, y], dtype=np.float64)
        # E126: ordinary garment landmarks use the external cloth silhouette.
        # Only neck_left/right may reference an internal collar/background contour.
        _boundary_ref = all_boundary_pts if str(name).startswith("neck_") else outer
        d2 = np.sum((_boundary_ref - p[None, :]) ** 2, axis=1)
        nearest = _boundary_ref[int(np.argmin(d2))]
        strength = float(np.clip((dist - float(threshold)) / max(10.0, 1.10 * float(threshold)), 0.0, 1.0))
        group = str(group_of.get(name, "OTHER"))
        item = {
            "name": str(name), "group": group, "x": float(x), "y": float(y), "conf": float(conf),
            "inside_distance_px": float(dist), "threshold_px": float(threshold),
            "strength": float(strength),
            "nearest_contour_px": [float(nearest[0]), float(nearest[1])],
        }
        intrusions.append(item)
        group_items.setdefault(group, []).append(item)

        cv2.line(corridor, tuple(np.rint(p).astype(int)), tuple(np.rint(nearest).astype(int)), 255, corridor_w, cv2.LINE_AA)
        cv2.circle(corridor, tuple(np.rint(p).astype(int)), max(7, corridor_w), 255, -1, cv2.LINE_AA)

        gm = group_masks.setdefault(group, np.zeros_like(garment))
        region_w = int(np.clip(round(max(18.0, 0.70 * dist)), 18, 58))
        cv2.line(gm, tuple(np.rint(p).astype(int)), tuple(np.rint(nearest).astype(int)), 255, region_w, cv2.LINE_AA)
        cv2.circle(gm, tuple(np.rint(p).astype(int)), max(9, region_w // 2), 255, -1, cv2.LINE_AA)
        cv2.circle(gm, tuple(np.rint(nearest).astype(int)), max(7, region_w // 3), 255, -1, cv2.LINE_AA)

    corridor = cv2.bitwise_and(corridor, garment)
    if not intrusions:
        return {
            "available": False, "reason": "NO_INTRUDED_BOUNDARY_KPT",
            "corridor_mask": corridor, "fold_region_mask": empty_region,
            "intrusions": [], "count": 0, "confidence": 0.0,
            "primary_group": "NONE", "group_scores": {},
        }

    garment_area = float(max(1, cv2.countNonZero(garment)))
    group_scores = {}
    group_regions = {}
    for group, items in group_items.items():
        gm = np.asarray(group_masks.get(group, np.zeros_like(garment)), dtype=np.uint8)
        # When two or more related landmarks intrude, bridge them with a hull so
        # the region represents a folded anatomical patch rather than isolated dots.
        hull_pts = []
        for it in items:
            hull_pts.append([float(it["x"]), float(it["y"])])
            hull_pts.append(list(it["nearest_contour_px"]))
        if len(hull_pts) >= 4:
            hp = np.rint(np.asarray(hull_pts, dtype=np.float64)).astype(np.int32)
            hull = cv2.convexHull(hp)
            hull_mask = np.zeros_like(garment)
            cv2.fillConvexPoly(hull_mask, hull, 255, cv2.LINE_AA)
            hull_ratio = float(cv2.countNonZero(cv2.bitwise_and(hull_mask, garment))) / garment_area
            if hull_ratio <= 0.42:
                gm = cv2.bitwise_or(gm, hull_mask)
        gm = cv2.bitwise_and(gm, garment)
        gm = cv2.morphologyEx(gm, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
        strengths = [float(it.get("strength", 0.0)) for it in items]
        confs = [float(it.get("conf", 0.0)) for it in items]
        med_strength = float(np.median(strengths)) if strengths else 0.0
        mean_conf = float(np.mean(confs)) if confs else 0.0
        score = float(np.clip((0.12 + 0.12 * min(3, len(items)) + 0.62 * med_strength) * (0.55 + 0.45 * mean_conf), 0.0, 1.0))
        # A single neck point can be slightly interior from mask/pose noise.  A
        # neck pair (or neck + upper-side anomaly) remains fully authoritative.
        if group == "NECK" and len(items) == 1:
            score *= 0.74
        area_ratio = float(cv2.countNonZero(gm)) / garment_area
        if area_ratio < 0.008 or area_ratio > 0.48:
            score *= 0.55
        group_scores[group] = float(score)
        group_regions[group] = gm

    primary_group = max(group_scores, key=group_scores.get) if group_scores else "NONE"
    confidence = float(group_scores.get(primary_group, 0.0))

    # E125 MULTI-FOLD: more than one anatomical part can be folded at the same
    # time (for example NECK + UPPER_LEFT in the physical tests). Start from the
    # strongest group, then merge other credible groups when they are spatially
    # adjacent to it or nearly as strong. This keeps pose-first semantics without
    # turning every weak interior keypoint into one giant folded region.
    fold_region = np.asarray(group_regions.get(primary_group, np.zeros_like(garment)), dtype=np.uint8).copy()
    active_groups = [str(primary_group)] if primary_group != "NONE" else []
    merge_floor = max(0.34, 0.72 * confidence)
    for group, score in sorted(group_scores.items(), key=lambda kv: float(kv[1]), reverse=True):
        if group == primary_group or float(score) < merge_floor:
            continue
        gm = np.asarray(group_regions.get(group, np.zeros_like(garment)), dtype=np.uint8)
        if cv2.countNonZero(gm) < 16:
            continue
        near = bool(cv2.countNonZero(cv2.bitwise_and(
            cv2.dilate(fold_region, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (41, 41)), iterations=1), gm
        )) > 0)
        similarly_strong = bool(float(score) >= 0.86 * max(1e-6, confidence))
        if not (near or similarly_strong):
            continue
        merged = cv2.bitwise_or(fold_region, gm)
        merged_ratio = float(cv2.countNonZero(merged)) / garment_area
        if merged_ratio <= 0.50:
            fold_region = merged
            active_groups.append(str(group))

    fold_region = cv2.bitwise_and(fold_region, garment)
    return {
        "available": bool(cv2.countNonZero(fold_region) >= 24 and confidence > 0.0),
        "reason": "BOUNDARY_KEYPOINT_INTRUSION",
        "corridor_mask": corridor,
        "fold_region_mask": fold_region,
        "intrusions": intrusions,
        "count": int(len(intrusions)),
        "confidence": float(confidence),
        "primary_group": str(primary_group),
        "active_groups": list(active_groups),
        "group_scores": {str(k): float(v) for k, v in group_scores.items()},
    }


def _e125_build_primary_fold_region(depth_state, mask_u8, args):
    """E125 folded-part hierarchy: POSE intrusion -> UniDepth raised region -> crease fallback.

    Returns one coarse PRIMARY FOLD REGION.  The region is deliberately decided
    *before* Fold-Crease selection so an internal wrinkle cannot become SELECTED
    CREASE and drag HOLD onto the already-folded flap.
    """
    mask = ((np.asarray(mask_u8, dtype=np.uint8) > 0).astype(np.uint8) * 255)
    h, w = mask.shape[:2]
    empty = np.zeros_like(mask)
    garment_area = float(max(1, cv2.countNonZero(mask)))

    pose_region = ((np.asarray(depth_state.get("pose_fold_region_mask", empty), dtype=np.uint8) > 0).astype(np.uint8) * 255)
    pose_conf = float(depth_state.get("pose_fold_region_confidence", depth_state.get("pose_fold_intrusion_confidence", 0.0)) or 0.0)
    pose_group = str(depth_state.get("pose_fold_primary_group", "NONE"))
    pose_active_groups = list(depth_state.get("pose_fold_active_groups", []) or [])
    pose_min = float(getattr(args, "micro_unfold_e125_pose_region_min_conf", 0.34))
    pose_ratio = float(cv2.countNonZero(pose_region)) / garment_area if cv2.countNonZero(pose_region) else 0.0
    if cv2.countNonZero(pose_region) >= 24 and pose_conf >= pose_min and 0.008 <= pose_ratio <= 0.48:
        region = cv2.bitwise_and(pose_region, mask)
        source = "POSE_INTRUSION"
        confidence = pose_conf
        group = "+".join(str(g) for g in pose_active_groups) if pose_active_groups else pose_group
        depth_diag = {}
    else:
        # Independent UniDepth fallback.  Use the raw TOP3 seed labels rather than
        # the old crease-dominated selected FUSED hypothesis.
        labels = np.asarray(depth_state.get("strict_candidate_labels", empty), dtype=np.uint8)
        relief = np.asarray(depth_state.get("local_relief_mm", np.zeros(mask.shape, dtype=np.float32)), dtype=np.float32)
        prom = np.asarray(depth_state.get("prominence_mm", np.zeros(mask.shape, dtype=np.float32)), dtype=np.float32)
        if relief.shape != mask.shape:
            relief = cv2.resize(relief, (w, h), interpolation=cv2.INTER_LINEAR)
        if prom.shape != mask.shape:
            prom = cv2.resize(prom, (w, h), interpolation=cv2.INTER_LINEAR)
        garment = mask > 0
        finite = garment & np.isfinite(relief) & np.isfinite(prom)
        body_vals = relief[finite]
        body_med = float(np.median(body_vals)) if body_vals.size else 0.0
        inside = cv2.distanceTransform(mask, cv2.DIST_L2, 5).astype(np.float32)
        best = None
        for rank in (1, 2, 3):
            seed = (labels == rank) & garment & np.isfinite(relief)
            n = int(np.count_nonzero(seed))
            if n < 8:
                continue
            rv = np.asarray(relief[seed], dtype=np.float32)
            pv = np.maximum(np.asarray(prom[seed], dtype=np.float32), 0.0)
            rp80 = float(np.percentile(rv, 80.0))
            rp55 = float(np.percentile(rv, 55.0))
            pp80 = float(np.percentile(pv, 80.0)) if pv.size else 0.0
            contrast = float(rp55 - body_med)
            raised_frac = float(np.mean(rv >= body_med + 0.8))
            interior = float(np.median(inside[seed])) if n else 0.0
            relief_s = float(np.clip(np.tanh(max(0.0, rp80) / 5.0), 0.0, 1.0))
            prom_s = float(np.clip(np.tanh(max(0.0, pp80) / 4.0), 0.0, 1.0))
            contrast_s = float(np.clip(np.tanh(max(0.0, contrast) / 3.0), 0.0, 1.0))
            interior_s = float(np.clip(interior / 26.0, 0.0, 1.0))
            score = float(np.clip(0.42 * relief_s + 0.24 * prom_s + 0.22 * contrast_s + 0.08 * raised_frac + 0.04 * interior_s, 0.0, 1.0))
            item = dict(rank=rank, score=score, relief_p80_mm=rp80, relief_p55_mm=rp55,
                        prom_p80_mm=pp80, depth_contrast_mm=contrast,
                        raised_fraction=raised_frac, interior_px=interior, seed=seed)
            if best is None or score > best["score"]:
                best = item

        depth_min_score = float(getattr(args, "micro_unfold_e125_depth_region_min_score", 0.42))
        depth_min_relief = float(getattr(args, "micro_unfold_e125_depth_region_min_relief_mm", 1.20))
        if best is not None and float(best["score"]) >= depth_min_score and (
            float(best["relief_p80_mm"]) >= depth_min_relief or float(best["depth_contrast_mm"]) >= 0.80
        ):
            seed = np.asarray(best["seed"], dtype=bool)
            near = cv2.dilate((seed.astype(np.uint8) * 255), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (41, 41)), iterations=1) > 0
            r_floor = max(body_med + 0.45, min(float(best["relief_p55_mm"]), body_med + 2.6))
            p_floor = max(0.25, min(1.8, 0.28 * float(best["prom_p80_mm"])))
            region_bool = garment & near & np.isfinite(relief) & (
                (relief >= r_floor) | (np.isfinite(prom) & (prom >= p_floor))
            )
            region_bool |= seed
            region = (region_bool.astype(np.uint8) * 255)
            region = cv2.morphologyEx(region, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
            # Keep only components touching the original UniDepth seed.
            nc, ll, ss, _ = cv2.connectedComponentsWithStats((region > 0).astype(np.uint8), 8)
            kept = np.zeros_like(region)
            for lid in range(1, nc):
                comp = ll == lid
                if np.count_nonzero(comp & seed) > 0:
                    kept[comp] = 255
            if cv2.countNonZero(kept) >= 24:
                region = cv2.bitwise_and(kept, mask)
                source = "UNIDEPTH_RAISED"
                confidence = float(best["score"])
                group = f"C{int(best['rank'])}"
                depth_diag = {k: v for k, v in best.items() if k != "seed"}
            else:
                region = empty.copy(); source = "CREASE_CONTOUR_FALLBACK"; confidence = 0.0; group = "NONE"; depth_diag = {}
        else:
            region = empty.copy(); source = "CREASE_CONTOUR_FALLBACK"; confidence = 0.0; group = "NONE"; depth_diag = {}

    region = cv2.bitwise_and(np.asarray(region, dtype=np.uint8), mask)
    # Internal region boundary only; remove the garment's external silhouette so
    # SELECTED CREASE is encouraged at the fold/body interface, not cloth edge.
    if cv2.countNonZero(region) >= 8:
        dil = cv2.dilate(region, np.ones((3, 3), np.uint8), iterations=1)
        ero = cv2.erode(region, np.ones((3, 3), np.uint8), iterations=1)
        boundary = cv2.subtract(dil, ero)
        ginside = cv2.distanceTransform(mask, cv2.DIST_L2, 5).astype(np.float32)
        internal = (boundary > 0) & (mask > 0) & (ginside >= 4.0)
        boundary = (internal.astype(np.uint8) * 255)
        if cv2.countNonZero(boundary) < 8:
            boundary = cv2.bitwise_and(boundary if cv2.countNonZero(boundary) else cv2.subtract(dil, ero), mask)
    else:
        boundary = empty.copy()
    authoritative = bool(source in ("POSE_INTRUSION", "UNIDEPTH_RAISED") and cv2.countNonZero(region) >= 24)
    return {
        "available": bool(authoritative), "authoritative": bool(authoritative),
        "source": str(source), "confidence": float(confidence), "group": str(group),
        "region_mask": region, "boundary_mask": boundary,
        "area_ratio": float(cv2.countNonZero(region)) / garment_area if cv2.countNonZero(region) else 0.0,
        "depth_diagnostics": depth_diag,
    }


def _e129_rank_pose_intrusion_active_pixels(depth_state, mask_u8, fold_region_u8, args=None):
    """Rank graspable ACTIVE pixels inside an authoritative pose-intrusion fold region.

    Pose intrusion owns *where* the folded flap is. UniDepth only refines which
    point inside that semantic region is most raised/graspable. FUSED FOLD is
    deliberately not read here. The returned list is image-space only so it can
    also drive the pre-HOLD diagnostic overlay.
    """
    mask=((np.asarray(mask_u8,dtype=np.uint8)>0).astype(np.uint8)*255)
    region=((np.asarray(fold_region_u8,dtype=np.uint8)>0).astype(np.uint8)*255)
    region=cv2.bitwise_and(region,mask)
    h,w=mask.shape[:2]
    if cv2.countNonZero(region)<16:
        return []
    relief=np.asarray(depth_state.get('local_relief_mm',np.zeros(mask.shape,dtype=np.float32)),dtype=np.float32)
    prom=np.asarray(depth_state.get('prominence_mm',np.zeros(mask.shape,dtype=np.float32)),dtype=np.float32)
    if relief.shape!=mask.shape:
        relief=cv2.resize(relief,(w,h),interpolation=cv2.INTER_LINEAR)
    if prom.shape!=mask.shape:
        prom=cv2.resize(prom,(w,h),interpolation=cv2.INTER_LINEAR)
    garment_inside=cv2.distanceTransform(mask,cv2.DIST_L2,5).astype(np.float32)
    region_inside=cv2.distanceTransform(region,cv2.DIST_L2,5).astype(np.float32)
    region_support=cv2.blur((region>0).astype(np.float32),(15,15))
    finite=(region>0)&np.isfinite(relief)&np.isfinite(prom)&(garment_inside>=3.0)
    ys,xs=np.nonzero(finite)
    if ys.size==0:
        return []
    rr=relief[ys,xs]
    pp=np.maximum(prom[ys,xs],0.0)
    # Local relief is the main depth cue; prominence stabilizes a raised flap.
    raised_mm=0.72*rr+0.28*pp
    raised_support=np.clip(0.5+0.5*np.tanh(raised_mm/3.2),0.0,1.0)
    pose_support=np.clip(region_support[ys,xs],0.0,1.0)
    garment_support=np.clip(garment_inside[ys,xs]/22.0,0.0,1.0)
    region_clear=np.clip(region_inside[ys,xs]/12.0,0.0,1.0)
    # Pose owns location (46%). UniDepth is the strongest refinement (34%).
    base=0.46*pose_support+0.34*raised_support+0.13*garment_support+0.07*region_clear
    order=np.argsort(-base)
    max_keep=int(getattr(args,'micro_unfold_e129_pose_active_max_candidates',90)) if args is not None else 90
    min_gap=float(getattr(args,'micro_unfold_e129_pose_active_nms_px',8.0)) if args is not None else 8.0
    picked=[]
    for oi in order.tolist():
        p=np.asarray([float(xs[oi]),float(ys[oi])],dtype=np.float64)
        if any(float(np.linalg.norm(p-q['px']))<min_gap for q in picked):
            continue
        picked.append({
            'px':p,
            'score':float(base[oi]),
            'pose_region_support':float(pose_support[oi]),
            'raised_support':float(raised_support[oi]),
            'raised_mm':float(raised_mm[oi]),
            'garment_interior_support':float(garment_support[oi]),
            'region_interior_support':float(region_clear[oi]),
        })
        if len(picked)>=max_keep:
            break
    return picked



def _e130_direct_intrusion_direction(depth_state, mask_u8, H_value, primary_group="NONE"):
    """Return direct folded-side direction from the actually intruded pose landmarks.

    E130 deliberately does not use E120's torso-quad deformation direction here.
    The user's physical observation is simpler and more reliable for manipulation:
    semantic boundary landmarks that moved inside the current Cloth Contour indicate
    *which side* was folded.  Their centroid is therefore direction evidence only,
    not the full folded-area mask.
    """
    mask=((np.asarray(mask_u8,dtype=np.uint8)>0).astype(np.uint8)*255)
    if cv2.countNonZero(mask)<50:
        return None
    mm=cv2.moments(mask,binaryImage=True)
    gc=np.asarray([float(mm['m10']/max(1e-6,mm['m00'])),float(mm['m01']/max(1e-6,mm['m00']))],dtype=np.float64)
    groups={g for g in str(primary_group).split('+') if g and g!='NONE'}
    items=[]
    for q in list(depth_state.get('pose_fold_intrusions',[]) or []):
        if not isinstance(q,dict):
            continue
        if groups and str(q.get('group','')) not in groups:
            continue
        try:
            x=float(q.get('x')); y=float(q.get('y')); c=float(q.get('conf',0.0)); st=float(q.get('strength',0.0))
        except Exception:
            continue
        if not (np.isfinite(x) and np.isfinite(y)):
            continue
        wt=max(0.05,c)*(0.35+0.65*max(0.0,st))
        items.append((wt,np.asarray([x,y],dtype=np.float64),str(q.get('name','?'))))
    if not items:
        return None
    sw=sum(v[0] for v in items)
    ep=sum((v[0]*v[1] for v in items),np.zeros(2,dtype=np.float64))/max(1e-9,sw)
    dpx=ep-gc
    nd=float(np.linalg.norm(dpx))
    if nd<8.0:
        return None
    dpx/=nd
    try:
        gb=_fix53_px_to_board(H_value,gc)
        eb=_fix53_px_to_board(H_value,gc+dpx*40.0)
        ub=np.asarray(eb-gb,dtype=np.float64)
        ub/=max(1e-9,float(np.linalg.norm(ub)))
    except Exception:
        return None
    return {'garment_center_px':gc,'evidence_px':ep,'unit_px':dpx,'unit_board':ub,
            'names':[v[2] for v in items],'weight':float(sw)}


_E156_INV_H_CACHE_KEY = None
_E156_INV_H_CACHE = None


def _e156_inverse_h(H_value):
    """Cache inverse Homography for the frozen X judgment.

    v3 recomputed np.linalg.inv(H) inside every 2mm ray sample. The same H is
    immutable during one judgment, so this is exact reuse, not an approximation.
    """
    global _E156_INV_H_CACHE_KEY, _E156_INV_H_CACHE
    Hm = np.asarray(H_value, dtype=np.float64)
    if Hm.shape != (3, 3):
        raise RuntimeError(f"E156 invalid Homography shape: {Hm.shape}")
    key = Hm.tobytes()
    if _E156_INV_H_CACHE_KEY != key or _E156_INV_H_CACHE is None:
        _E156_INV_H_CACHE = np.linalg.inv(Hm)
        _E156_INV_H_CACHE_KEY = key
    return _E156_INV_H_CACHE


def _e130_ray_exit(mask_u8,H_value,origin_board,unit_board,max_mm=420.0,step_mm=2.0,
                   stable_mask=None, invH=None):
    """Exact E130 3-outside-sample exit test, vectorized in one perspectiveTransform."""
    mask=np.asarray(mask_u8,dtype=np.uint8)>0
    h,w=mask.shape[:2]
    stable=(np.asarray(stable_mask,dtype=bool) if stable_mask is not None else
            cv2.dilate((mask.astype(np.uint8)*255),np.ones((5,5),np.uint8),iterations=1)>0)
    u=np.asarray(unit_board,dtype=np.float64).reshape(2)
    u/=max(1e-9,float(np.linalg.norm(u)))
    step=float(step_mm)
    ds=np.arange(0.0,float(max_mm)+1e-6,step,dtype=np.float64)
    bps=np.asarray(origin_board,dtype=np.float64).reshape(1,2)+ds[:,None]*u.reshape(1,2)
    inv = _e156_inverse_h(H_value) if invH is None else np.asarray(invH,dtype=np.float64)
    pxs=cv2.perspectiveTransform(bps.astype(np.float32).reshape(-1,1,2),inv.astype(np.float64)).reshape(-1,2)
    xi=np.rint(pxs[:,0]).astype(np.int32); yi=np.rint(pxs[:,1]).astype(np.int32)
    inb=(xi>=0)&(xi<w)&(yi>=0)&(yi<h)
    inside=np.zeros((len(ds),),dtype=bool)
    ids=np.flatnonzero(inb)
    if ids.size:
        inside[ids]=stable[yi[ids],xi[ids]]
    outside_run=0; last_px=None
    for i,is_inside in enumerate(inside.tolist()):
        if is_inside:
            outside_run=0; last_px=pxs[i]
        else:
            outside_run+=1
            if outside_run>=3:
                bd=max(0.0,float(ds[i])-2.0*step)
                bb=np.asarray(origin_board,dtype=np.float64)+u*bd
                try:
                    bpx=cv2.perspectiveTransform(bb.astype(np.float32).reshape(1,1,2),inv.astype(np.float64))[0,0]
                except Exception:
                    bpx=last_px
                return float(bd),np.asarray(bb,dtype=np.float64),None if bpx is None else np.asarray(bpx,dtype=np.float64)
    return None


def _e130_find_active_pull_endpoint(mask_u8,H_value,cfg,active_board,pull_u,args,*,stable_mask=None,invH=None):
    """E130 endpoint geometry independent of HOLD; safe to compute once per ACTIVE candidate."""
    u=np.asarray(pull_u,dtype=np.float64).reshape(2)
    u/=max(1e-9,float(np.linalg.norm(u)))
    step=float(np.clip(float(getattr(args,'micro_unfold_boundary_sample_mm',2.0)),1.0,5.0))
    max_pull=max(60.0,float(getattr(args,'micro_unfold_max_pull_mm',264.0)))
    min_pull=max(20.0,float(getattr(args,'micro_unfold_pull_mm',30.0)))
    factor=float(np.clip(float(getattr(args,'micro_unfold_e130_active_boundary_factor',2.0)),1.4,2.5))
    aex=_e130_ray_exit(mask_u8,H_value,active_board,u,max_mm=max_pull+180.0,step_mm=step,
                       stable_mask=stable_mask,invH=invH)
    if aex is None:
        raise RuntimeError('E130 ACTIVE axis does not reach Cloth Contour')
    active_boundary=float(aex[0])
    target=max(min_pull,factor*active_boundary)
    minimum_outside=active_boundary+6.0
    desired=min(max_pull,max(target,minimum_outside))
    if desired+1e-6<minimum_outside:
        raise RuntimeError(f'E130 max pull {max_pull:.1f}mm cannot clear Cloth Contour at {active_boundary:.1f}mm')
    mask=np.asarray(mask_u8,dtype=np.uint8)>0; h,w=mask.shape[:2]
    inv = _e156_inverse_h(H_value) if invH is None else np.asarray(invH,dtype=np.float64)
    # Exact old-loop candidate set: desired, desired-step, ... while
    # trial >= minimum_outside-1e-6. Do not admit one extra sub-boundary point.
    ntrial=max(1,int(np.floor((float(desired)-(float(minimum_outside)-1e-6))/step))+1)
    trials=float(desired)-step*np.arange(ntrial,dtype=np.float64)
    trials=trials[trials>=float(minimum_outside)-1e-6]
    if trials.size==0:
        trials=np.asarray([float(desired)],dtype=np.float64)
    bps=np.asarray(active_board,dtype=np.float64).reshape(1,2)+trials[:,None]*u.reshape(1,2)
    board_ok=((bps[:,0]>=float(cfg.board_x_min)+20.0)&(bps[:,0]<=float(cfg.board_x_max)-20.0)&
              (bps[:,1]>=float(cfg.board_y_min)+20.0)&(bps[:,1]<=float(cfg.board_y_max)-20.0))
    pxs=cv2.perspectiveTransform(bps.astype(np.float32).reshape(-1,1,2),inv.astype(np.float64)).reshape(-1,2)
    xi=np.rint(pxs[:,0]).astype(np.int32); yi=np.rint(pxs[:,1]).astype(np.int32)
    inb=(xi>=0)&(xi<w)&(yi>=0)&(yi<h)
    garment=np.zeros((len(trials),),dtype=bool)
    ids=np.flatnonzero(inb)
    if ids.size: garment[ids]=mask[yi[ids],xi[ids]]
    valid=np.flatnonzero(board_ok & inb & (~garment))
    if valid.size<=0:
        raise RuntimeError('E130 cannot place 2x ACTIVE-boundary endpoint inside board safety')
    j=int(valid[0]); trial=float(trials[j]); chosen=np.asarray(bps[j],dtype=np.float64); chosen_px=np.asarray(pxs[j],dtype=np.float64)
    outside=max(0.0,trial-active_boundary)
    return chosen,trial,active_boundary,float(outside),chosen_px,u


def _e130_find_axis_pull_endpoint(mask_u8,H_value,cfg,active_board,hold_board,pull_u,args,log_final=False,
                                  stable_mask=None,invH=None):
    pull_board,trial,active_boundary,outside,chosen_px,u=_e130_find_active_pull_endpoint(
        mask_u8,H_value,cfg,active_board,pull_u,args,stable_mask=stable_mask,invH=invH)
    step=float(np.clip(float(getattr(args,'micro_unfold_boundary_sample_mm',2.0)),1.0,5.0))
    max_pull=max(60.0,float(getattr(args,'micro_unfold_max_pull_mm',264.0)))
    hexit=_e130_ray_exit(mask_u8,H_value,hold_board,u,max_mm=max_pull+220.0,step_mm=step,
                         stable_mask=stable_mask,invH=invH)
    hold_boundary=float(hexit[0]) if hexit is not None else float(active_boundary+np.linalg.norm(np.asarray(active_board)-np.asarray(hold_board)))
    factor=float(np.clip(float(getattr(args,'micro_unfold_e130_active_boundary_factor',2.0)),1.4,2.5))
    if log_final:
        target=max(max(20.0,float(getattr(args,'micro_unfold_pull_mm',30.0))),factor*active_boundary)
        print(f'[E130-PULL] mode=POSE/DIRECTION_AXIS ACTIVE->contour={active_boundary:.1f}mm target={factor:.2f}x={target:.1f}mm HOLD->contour={hold_boundary:.1f}mm requested={trial:.1f}mm max={max_pull:.1f}mm')
    return (np.asarray(pull_board,dtype=np.float64),float(trial),active_boundary,float(outside),
            np.asarray(chosen_px,dtype=np.float64),u,hold_boundary,0.0,True)

def _e130_directional_pair(depth_state,mask_u8,H_value,args,config,cfg,fold_region_info,exec_crease,preview_crease):
    """Deterministic body->crease->ACTIVE->fold-direction->contour planner.

    Authority order:
      1. Pose intrusion chooses the outward fold/unfold axis (direction only).
      2. The nearest strong Fold Crease encountered when moving inward from that
         evidence becomes the BODY/fold boundary.
      3. ACTIVE is placed just outside that crease, biased toward the garment
         centre and refined by UniDepth.
      4. HOLD is placed much farther on the BODY side.
      5. Pull continues along the same axis for ~2 x ACTIVE-to-contour distance.

    If pose intrusion is absent, the same deterministic construction uses the
    authoritative UniDepth region, then selected depth candidate, then strongest
    crease/relief as a last-resort direction.  This is the E130 "I->READY" path:
    perception confidence no longer blocks execution once a valid garment mask and
    mechanically feasible dual-arm pair exist.
    """
    mask=((np.asarray(mask_u8,dtype=np.uint8)>0).astype(np.uint8)*255)
    if cv2.countNonZero(mask)<500:
        return None
    h,w=mask.shape[:2]
    mm=cv2.moments(mask,binaryImage=True)
    gc=np.asarray([float(mm['m10']/max(1e-6,mm['m00'])),float(mm['m01']/max(1e-6,mm['m00']))],dtype=np.float64)
    try: gc_board=np.asarray(_fix53_px_to_board(H_value,gc),dtype=np.float64)
    except Exception: return None
    source=str(fold_region_info.get('source','CREASE_CONTOUR_FALLBACK'))
    group=str(fold_region_info.get('group','NONE'))
    conf=float(fold_region_info.get('confidence',0.0) or 0.0)
    region=((np.asarray(fold_region_info.get('region_mask',np.zeros_like(mask)),dtype=np.uint8)>0).astype(np.uint8)*255)

    # E145: when Pose cannot provide an unfold direction, the selected Fold-Crease
    # itself owns the direction.  Its PCA tangent is converted to ONE exact normal
    # by _e145_select_exact_fold_normal().  No angular perturbation is permitted.
    _e145_axis_raw=fold_region_info.get('explicit_axis_unit_px',None)
    _e145_evidence_raw=fold_region_info.get('explicit_evidence_px',None)
    _e145_exact_axis=bool(source=='RGB_CREASE_NORMAL_EXACT' and _e145_axis_raw is not None and _e145_evidence_raw is not None)

    direct=_e130_direct_intrusion_direction(depth_state,mask,H_value,group) if source=='POSE_INTRUSION' else None
    evidence_px=None; u_px=None; u_board=None; axis_source=source
    if _e145_exact_axis:
        try:
            u_px=np.asarray(_e145_axis_raw,dtype=np.float64).reshape(2)
            _nu=float(np.linalg.norm(u_px))
            if (not np.all(np.isfinite(u_px))) or _nu<1e-9:
                return None
            u_px=u_px/_nu
            evidence_px=np.asarray(_e145_evidence_raw,dtype=np.float64).reshape(2)
            if not np.all(np.isfinite(evidence_px)):
                return None
            _axis_origin_raw=fold_region_info.get('explicit_crease_anchor_px',None)
            _axis_origin=(np.asarray(_axis_origin_raw,dtype=np.float64).reshape(2)
                          if _axis_origin_raw is not None else gc.copy())
            if not np.all(np.isfinite(_axis_origin)):
                _axis_origin=gc.copy()
            b0=np.asarray(_fix53_px_to_board(H_value,_axis_origin),dtype=np.float64)
            b2=np.asarray(_fix53_px_to_board(H_value,_axis_origin+u_px*40.0),dtype=np.float64)
            u_board=b2-b0
            u_board/=max(1e-9,float(np.linalg.norm(u_board)))
            axis_source='RGB_CREASE_NORMAL_EXACT'
        except Exception:
            return None
    elif isinstance(direct,dict):
        evidence_px=np.asarray(direct['evidence_px'],dtype=np.float64); u_px=np.asarray(direct['unit_px'],dtype=np.float64); u_board=np.asarray(direct['unit_board'],dtype=np.float64)
        axis_source='POSE_INTRUSION_DIRECTION'
    elif cv2.countNonZero(region)>=8:
        ys,xs=np.where(region>0)
        relief=np.asarray(depth_state.get('local_relief_mm',np.zeros(mask.shape,dtype=np.float32)),dtype=np.float32)
        wt=np.maximum(relief[ys,xs],0.0)+0.5 if relief.shape==mask.shape else np.ones_like(xs,dtype=np.float32)
        sw=float(np.sum(wt)); evidence_px=np.asarray([float(np.sum(xs*wt)/max(1e-6,sw)),float(np.sum(ys*wt)/max(1e-6,sw))],dtype=np.float64)
    if evidence_px is None:
        labels=np.asarray(depth_state.get('strict_candidate_labels',np.zeros_like(mask)),dtype=np.uint8)
        sel=int(depth_state.get('selected_donor_rank',depth_state.get('selected_rank',1)) or 1)
        ys,xs=np.where(labels==sel)
        if xs.size>=4:
            evidence_px=np.asarray([float(np.mean(xs)),float(np.mean(ys))],dtype=np.float64); axis_source='UNIDEPTH_SELECTED_DIRECTION'
    if evidence_px is None:
        pool=np.asarray(exec_crease if cv2.countNonZero(np.asarray(exec_crease,dtype=np.uint8))>=8 else preview_crease,dtype=np.uint8)
        ys,xs=np.where(pool>0)
        if xs.size>=4:
            d2=(xs-gc[0])**2+(ys-gc[1])**2; j=int(np.argmax(d2)); evidence_px=np.asarray([float(xs[j]),float(ys[j])],dtype=np.float64); axis_source='CREASE_DIRECTION_FALLBACK'
    if evidence_px is None:
        relief=np.asarray(depth_state.get('local_relief_mm',np.zeros(mask.shape,dtype=np.float32)),dtype=np.float32)
        inside=cv2.distanceTransform(mask,cv2.DIST_L2,5)
        valid=(mask>0)&np.isfinite(relief)&(inside>=4.0)
        if np.any(valid):
            yy,xx=np.where(valid); vals=relief[valid]; j=int(np.argmax(vals)); evidence_px=np.asarray([float(xx[j]),float(yy[j])],dtype=np.float64); axis_source='MAX_RELIEF_DIRECTION_FALLBACK'
    if evidence_px is None:
        cs,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
        if not cs: return None
        pts=max(cs,key=cv2.contourArea).reshape(-1,2).astype(np.float64); j=int(np.argmax(np.linalg.norm(pts-gc[None,:],axis=1))); evidence_px=pts[j]; axis_source='CONTOUR_DIRECTION_FALLBACK'
    if u_px is None:
        dpx=np.asarray(evidence_px-gc,dtype=np.float64); nd=float(np.linalg.norm(dpx))
        if nd<8.0: return None
        u_px=dpx/nd
        try:
            b2=np.asarray(_fix53_px_to_board(H_value,gc+u_px*40.0),dtype=np.float64); u_board=b2-gc_board; u_board/=max(1e-9,float(np.linalg.norm(u_board)))
        except Exception: return None
    if _e145_exact_axis:
        try:
            _proj_origin=np.asarray(fold_region_info.get('explicit_crease_anchor_px',gc),dtype=np.float64).reshape(2)
            if not np.all(np.isfinite(_proj_origin)):
                _proj_origin=gc.copy()
        except Exception:
            _proj_origin=gc.copy()
        evidence_proj=float(np.dot(evidence_px-_proj_origin,u_px))
    else:
        evidence_proj=float(np.dot(evidence_px-gc,u_px))
    if evidence_proj<10.0:
        return None

    # Choose the Fold Crease immediately inward from the direction evidence. This
    # makes pose a direction cue while the crease determines the true folded extent.
    crease_pool=np.asarray(exec_crease if cv2.countNonZero(np.asarray(exec_crease,dtype=np.uint8))>=8 else preview_crease,dtype=np.uint8)
    if cv2.countNonZero(crease_pool)<8:
        # E144: only contour-veto-surviving raw crease may be a fallback.
        crease_pool=((np.asarray(depth_state.get('e144_fold_contour_filtered_skeleton',np.zeros_like(mask)),dtype=np.uint8)>0).astype(np.uint8)*255)
    score_map=np.asarray(depth_state.get('fold_crease_score',np.zeros(mask.shape,dtype=np.float32)),dtype=np.float32)
    bestc=None; selected_comp=np.zeros_like(mask,dtype=np.uint8)
    _e145_explicit_crease=False
    if _e145_exact_axis:
        try:
            _ec=((np.asarray(fold_region_info.get('explicit_crease_mask',np.zeros_like(mask)),dtype=np.uint8)>0).astype(np.uint8)*255)
            _ea=np.asarray(fold_region_info.get('explicit_crease_anchor_px',None),dtype=np.float64).reshape(2)
            if cv2.countNonZero(_ec)>=8 and np.all(np.isfinite(_ea)):
                selected_comp=cv2.bitwise_and(_ec,mask)
                anchor_px=_ea.copy()
                anchor_proj=float(np.dot(anchor_px-gc,u_px))
                real_crease=True
                _e145_explicit_crease=True
        except Exception:
            _e145_explicit_crease=False

    if (not _e145_explicit_crease) and cv2.countNonZero(crease_pool)>=8:
        nc,ll,ss,_=cv2.connectedComponentsWithStats((crease_pool>0).astype(np.uint8),8)
        for lid in range(1,nc):
            yy,xx=np.where(ll==lid)
            if xx.size<8: continue
            pts=np.column_stack([xx,yy]).astype(np.float64)
            rel=pts-gc[None,:]; proj=rel@u_px; lat=np.abs(rel[:,0]*(-u_px[1])+rel[:,1]*u_px[0])
            valid=(proj>max(8.0,0.08*evidence_proj))&(proj<evidence_proj-3.0)&(lat<max(36.0,0.24*evidence_proj))
            if not np.any(valid): continue
            ids=np.where(valid)[0]; # closest credible crease inward from pose evidence
            # Point closest to axis and relatively close to evidence.
            metric=0.65*lat[ids]+0.35*np.maximum(0.0,evidence_proj-proj[ids])
            j=ids[int(np.argmin(metric))]; cp=pts[j]; cproj=float(proj[j]); clat=float(lat[j])
            vals=score_map[ll==lid] if score_map.shape==mask.shape else np.asarray([])
            cscore=float(np.mean(vals[np.isfinite(vals)])) if vals.size and np.any(np.isfinite(vals)) else 0.35
            span=float(max(ss[lid,cv2.CC_STAT_WIDTH],ss[lid,cv2.CC_STAT_HEIGHT]))
            inward_gap=max(0.0,evidence_proj-cproj)
            q=0.38*np.clip(cscore,0.0,1.0)+0.26*np.exp(-abs(inward_gap-55.0)/65.0)+0.22*np.exp(-clat/34.0)+0.14*np.clip(span/120.0,0.0,1.0)
            # Strongly favor the *outermost inward* credible crease, i.e. the first
            # one encountered when tracing from the intruded pose evidence toward body.
            q+=0.16*np.clip(cproj/max(1.0,evidence_proj),0.0,1.0)
            if bestc is None or q>bestc[0]: bestc=(float(q),cp,lid,cproj,clat)
    if _e145_explicit_crease:
        pass
    elif bestc is not None:
        _,anchor_px,lid,anchor_proj,_=bestc; selected_comp[ll==lid]=255; real_crease=True
    else:
        # E145 exact Fold-Crease fallback is not allowed to invent a virtual crease.
        if _e145_exact_axis:
            return None
        anchor_proj=float(np.clip(evidence_proj-55.0,0.34*evidence_proj,0.76*evidence_proj))
        anchor_px=gc+u_px*anchor_proj; real_crease=False
        # Virtual boundary perpendicular to the directional axis for diagnostics.
        tan_px=np.asarray([-u_px[1],u_px[0]],dtype=np.float64)
        p0=anchor_px-tan_px*260.0; p1=anchor_px+tan_px*260.0
        cv2.line(selected_comp,tuple(np.rint(p0).astype(int)),tuple(np.rint(p1).astype(int)),255,5,cv2.LINE_AA)
        selected_comp=cv2.bitwise_and(selected_comp,mask)
    try: anchor_board=np.asarray(_fix53_px_to_board(H_value,anchor_px),dtype=np.float64)
    except Exception: return None
    tangent_board=np.asarray([-u_board[1],u_board[0]],dtype=np.float64)

    # Expanded actual folded extent for overlay: crease -> outward Cloth Contour.
    yygrid,xxgrid=np.indices(mask.shape,dtype=np.float32)
    if _e145_exact_axis and _e145_explicit_crease:
        # Exact normal means the BODY/FLAP split plane must pass through the
        # selected Fold-Crease itself, not through a garment-centroid radial line.
        projgrid=(xxgrid-float(anchor_px[0]))*float(u_px[0])+(yygrid-float(anchor_px[1]))*float(u_px[1])
        fold_extent=((mask>0)&(projgrid>=-2.0)).astype(np.uint8)*255
    else:
        projgrid=(xxgrid-float(gc[0]))*float(u_px[0])+(yygrid-float(gc[1]))*float(u_px[1])
        fold_extent=((mask>0)&(projgrid>=float(anchor_proj)-2.0)).astype(np.uint8)*255
    # Keep the component containing direction evidence when possible.
    ex=int(np.clip(round(float(evidence_px[0])),0,w-1)); ey=int(np.clip(round(float(evidence_px[1])),0,h-1))
    if fold_extent[ey,ex]>0:
        nc,lab,st,_=cv2.connectedComponentsWithStats((fold_extent>0).astype(np.uint8),8)
        lid=int(lab[ey,ex]);
        if lid>0: fold_extent=((lab==lid).astype(np.uint8)*255)

    relief=np.asarray(depth_state.get('local_relief_mm',np.zeros(mask.shape,dtype=np.float32)),dtype=np.float32)
    prom=np.asarray(depth_state.get('prominence_mm',np.zeros(mask.shape,dtype=np.float32)),dtype=np.float32)
    inside=cv2.distanceTransform(mask,cv2.DIST_L2,5).astype(np.float32)
    crease_dist=cv2.distanceTransform(((selected_comp==0).astype(np.uint8)*255),cv2.DIST_L2,5).astype(np.float32) if cv2.countNonZero(selected_comp)>0 else np.full(mask.shape,99.0,dtype=np.float32)

    # ACTIVE: just outward from the selected crease; inward-biased (close to body
    # centre) with UniDepth only refining the grasp patch.
    active_candidates=[]
    evidence_board=np.asarray(_fix53_px_to_board(H_value,evidence_px),dtype=np.float64)
    gap_board=float(np.dot(evidence_board-anchor_board,u_board))
    for dn in (10.0,12.0,16.0,20.0,24.0,30.0,36.0,42.0,48.0):
        if gap_board>18.0 and dn>gap_board-5.0: continue
        for dt in (-36.0,-24.0,-12.0,0.0,12.0,24.0,36.0):
            ab=anchor_board+u_board*dn+tangent_board*dt
            try: ap=np.asarray(_fix89_board_to_px(H_value,ab),dtype=np.float64)
            except Exception: continue
            xi,yi=int(round(ap[0])),int(round(ap[1]))
            if not (0<=xi<w and 0<=yi<h and mask[yi,xi]>0 and inside[yi,xi]>=3.0): continue
            rv=float(relief[yi,xi]) if relief.shape==mask.shape and np.isfinite(relief[yi,xi]) else 0.0
            pv=float(prom[yi,xi]) if prom.shape==mask.shape and np.isfinite(prom[yi,xi]) else 0.0
            raised=float(np.clip(0.5+0.5*np.tanh((0.75*rv+0.25*max(0.0,pv))/3.0),0.0,1.0))
            inward_pref=float(np.exp(-max(0.0,dn-14.0)/22.0))
            interior=float(np.clip(inside[yi,xi]/20.0,0.0,1.0))
            axis=float(np.exp(-abs(dt)/24.0))
            score=0.42*inward_pref+0.30*raised+0.18*interior+0.10*axis
            active_candidates.append((score,ab,ap,dn,dt,rv,pv))
    if not active_candidates:
        return None
    active_candidates.sort(key=lambda z:z[0],reverse=True)

    # V10 same-crease reach recovery may move ACTIVE tangentially as well as HOLD.
    # The normal-side role is unchanged (dn stays positive/outward), and the exact
    # crease/axis remains authoritative.  These candidates are consulted only after
    # the proven base pools fail the unchanged 420mm hard reach gate.
    active_candidates_reach=list(active_candidates)
    for dn in (10.0,14.0,18.0,22.0,28.0,34.0,40.0,46.0):
        if gap_board>18.0 and dn>gap_board-5.0:
            continue
        for dt in (-72.0,-60.0,-48.0,48.0,60.0,72.0):
            ab=anchor_board+u_board*dn+tangent_board*dt
            try:
                ap=np.asarray(_fix89_board_to_px(H_value,ab),dtype=np.float64)
            except Exception:
                continue
            xi,yi=int(round(ap[0])),int(round(ap[1]))
            if not (0<=xi<w and 0<=yi<h and mask[yi,xi]>0 and inside[yi,xi]>=3.0):
                continue
            rv=float(relief[yi,xi]) if relief.shape==mask.shape and np.isfinite(relief[yi,xi]) else 0.0
            pv=float(prom[yi,xi]) if prom.shape==mask.shape and np.isfinite(prom[yi,xi]) else 0.0
            raised=float(np.clip(0.5+0.5*np.tanh((0.75*rv+0.25*max(0.0,pv))/3.0),0.0,1.0))
            inward_pref=float(np.exp(-max(0.0,dn-14.0)/22.0))
            interior=float(np.clip(inside[yi,xi]/20.0,0.0,1.0))
            # Stronger tangent penalty than the base pool: this is reach recovery,
            # not a new preferred geometry.
            axis=float(np.exp(-abs(dt)/34.0))
            score=0.42*inward_pref+0.30*raised+0.18*interior+0.10*axis-0.04*np.clip((abs(dt)-36.0)/36.0,0.0,1.0)
            active_candidates_reach.append((score,ab,ap,dn,dt,rv,pv))
    active_candidates_reach.sort(key=lambda z:z[0],reverse=True)

    # HOLD: well inside BODY, farther from the crease than ACTIVE and biased toward
    # flat, central, high-interior cloth.  V9 keeps the original HOLD pool as the
    # first-choice behavior.  A second, same-crease tangent pool is prepared only
    # for bounded reach recovery after the original pool fails the unchanged 420mm
    # hard robot-radius gate.
    hold_candidates=[]
    hold_candidates_reach=[]
    gc_dist_scale=max(80.0,float(np.sqrt(cv2.countNonZero(mask))))

    def _v9_add_hold_candidate(dst,hd,dt):
        hb=anchor_board-u_board*float(hd)+tangent_board*float(dt)
        try: hp=np.asarray(_fix89_board_to_px(H_value,hb),dtype=np.float64)
        except Exception: return
        xi,yi=int(round(hp[0])),int(round(hp[1]))
        if not (0<=xi<w and 0<=yi<h and mask[yi,xi]>0 and inside[yi,xi]>=4.0): return
        rv=float(relief[yi,xi]) if relief.shape==mask.shape and np.isfinite(relief[yi,xi]) else 0.0
        pv=float(prom[yi,xi]) if prom.shape==mask.shape and np.isfinite(prom[yi,xi]) else 0.0
        flat=float(np.exp(-(abs(rv)+0.7*max(0.0,pv))/4.0))
        interior=float(np.clip(inside[yi,xi]/24.0,0.0,1.0))
        center=float(np.exp(-float(np.linalg.norm(hp-gc))/max(40.0,gc_dist_scale)))
        dist_pref=float(np.exp(-abs(float(hd)-68.0)/24.0))
        clear=float(np.clip(crease_dist[yi,xi]/24.0,0.0,1.0))
        score=0.28*dist_pref+0.24*flat+0.22*interior+0.16*center+0.10*clear
        dst.append((score,hb,hp,float(hd),float(dt),rv,pv))

    _v9_hold_depths=(46.0,52.0,58.0,64.0,72.0,82.0,92.0,102.0)
    _v9_base_hold_tangent=(-50.0,-35.0,-20.0,-10.0,0.0,10.0,20.0,35.0,50.0)
    for hd in _v9_hold_depths:
        for dt in _v9_base_hold_tangent:
            _v9_add_hold_candidate(hold_candidates,hd,dt)
    if not hold_candidates:
        return None
    hold_candidates.sort(key=lambda z:z[0],reverse=True)

    # Use the existing 126mm pair-separation ceiling to bound the recovery fan.
    # This is a search-space extension, not a relaxed safety threshold.
    _v9_pair_sep_hi=126.0
    _v9_extra_abs=tuple(float(_v9_pair_sep_hi*r) for r in (0.52,0.64,0.76,0.88))
    _v9_extra_hold_tangent=tuple(-x for x in reversed(_v9_extra_abs))+_v9_extra_abs
    for hd in _v9_hold_depths:
        for dt in _v9_extra_hold_tangent:
            _v9_add_hold_candidate(hold_candidates_reach,hd,dt)
    hold_candidates_reach.sort(key=lambda z:z[0],reverse=True)

    # v4 mechanically-feasible joint search: exact same safety equations, but
    # invariant work is cached. v3 recomputed a 720p dilation + hundreds of
    # board->pixel transforms + Homography inversions for every ACTIVE x HOLD pair.
    joint_t0=time.perf_counter()
    stable_ray=cv2.dilate(((mask>0).astype(np.uint8)*255),np.ones((5,5),np.uint8),iterations=1)>0
    invH=_e156_inverse_h(H_value)
    ts61=np.linspace(0.0,1.0,61)

    # v31 invariant caches shared by BASE / tier1 / tier2 searches.  The same
    # candidate tuples are reused across tiers, so there is no reason to rebuild
    # board/local transforms, active pull paths or HOLD arm metadata.
    _v31_active_meta_cache={}
    _v31_hold_meta_cache={}
    _v31_cross_cache={}
    _v31_path_clear_cache={}
    _v31_standby_local={}
    _v31_standby_board={}
    for _arm in ('arm1','arm2'):
        try:
            _st=np.asarray(standby_roarm_pose(cfg,_arm)[:2],dtype=np.float64)
            _v31_standby_local[_arm]=_st
            _v31_standby_board[_arm]=_fix37_local_xy_to_board(config,_arm,_st)
        except Exception:
            pass

    def _v10_active_meta_from(candidates,limit=None):
        out=[]
        seq=candidates if limit is None else candidates[:int(limit)]
        for ac in seq:
            _ck=id(ac)
            if _ck in _v31_active_meta_cache:
                _cached=_v31_active_meta_cache[_ck]
                if _cached is not None:
                    out.append(_cached)
                continue
            _,ab,ap,ad,adt,ar,aprom=ac
            try:
                pull_board,used,exit_mm,outside,pull_px,pu=_e130_find_active_pull_endpoint(
                    mask,H_value,cfg,ab,u_board,args,stable_mask=stable_ray,invH=invH)
            except Exception:
                _v31_active_meta_cache[_ck]=None
                continue
            arm_meta={}
            for active_arm in ('arm1','arm2'):
                try:
                    al=_fix6_local_xy(config,active_arm,ab); pl=_fix6_local_xy(config,active_arm,pull_board)
                    ast=_v31_standby_local.get(active_arm)
                    astb=_v31_standby_board.get(active_arm)
                    if ast is None or astb is None:
                        ast=np.asarray(standby_roarm_pose(cfg,active_arm)[:2],dtype=np.float64)
                        astb=_fix37_local_xy_to_board(config,active_arm,ast)
                    path=astb[None,:]+ts61[:,None]*(ab-astb)[None,:]
                    arm_meta[active_arm]={
                        'al':al,'pl':pl,'ast':ast,'path_board':path,
                        'active_pull_radius':max(float(np.linalg.norm(al)),float(np.linalg.norm(pl))),
                        'active_travel':float(np.linalg.norm(al-ast)),
                    }
                except Exception:
                    pass
            if arm_meta:
                _entry=(ac,(pull_board,used,exit_mm,outside,pull_px,pu),arm_meta)
                _v31_active_meta_cache[_ck]=_entry
                out.append(_entry)
            else:
                _v31_active_meta_cache[_ck]=None
        return out

    active_meta=_v10_active_meta_from(active_candidates,limit=30)

    def _v9_hold_meta_from(candidates,limit=None):
        out=[]
        seq=candidates if limit is None else candidates[:int(limit)]
        for hc in seq:
            _ck=id(hc)
            if _ck in _v31_hold_meta_cache:
                _cached=_v31_hold_meta_cache[_ck]
                if _cached is not None:
                    out.append(_cached)
                continue
            _,hb,hp,hd,hdt,hr,hprom=hc
            arms_h={}
            for hold_arm in ('arm1','arm2'):
                try:
                    hl=_fix6_local_xy(config,hold_arm,hb)
                    hst=_v31_standby_local.get(hold_arm)
                    if hst is None:
                        hst=np.asarray(standby_roarm_pose(cfg,hold_arm)[:2],dtype=np.float64)
                    arms_h[hold_arm]={
                        'hl':hl,'hst':hst,'hold_radius':float(np.linalg.norm(hl)),
                        'hold_travel':float(np.linalg.norm(hl-hst)),
                    }
                except Exception:
                    pass
            if arms_h:
                _entry=(hc,arms_h)
                _v31_hold_meta_cache[_ck]=_entry
                out.append(_entry)
            else:
                _v31_hold_meta_cache[_ck]=None
        return out

    _v30_role_gate_active=bool(depth_state.get('vla_joint_strict_active',False))
    _v30_role_crease=selected_comp.copy()
    if ('VIRTUAL' in str('E130_DIRECTIONAL_REAL_CREASE' if real_crease else 'E130_DIRECTIONAL_VIRTUAL_CREASE').upper()) or cv2.countNonZero(_v30_role_crease)<8:
        _v30_state_crease=np.asarray(depth_state.get('vla_joint_selected_rgb_skeleton',np.zeros_like(mask)),dtype=np.uint8)
        if cv2.countNonZero(_v30_state_crease)>=8:
            _v30_role_crease=_v30_state_crease
    # v31: exact same v30 13x13 crossing mask, constructed ONCE for this selected
    # crease and reused through BASE/tier1/tier2.
    _v31_role_crease_near=_v31_prepare_crease_near(_v30_role_crease) if _v30_role_gate_active else None

    def _v9_search_pairs(ameta_pool,hmeta,*,mode):
        # v31 optimization rules:
        #   * 66..126mm separation + crease-side role are filtered as NumPy matrices.
        #   * the v30 13x13 crease-cross test uses one prepared mask and a cross cache.
        #   * candidates are visited by a strict score upper bound; target distance is
        #     only a tie-breaker.  Early exit is allowed only when no unseen pair can
        #     mathematically equal/beat the current q, preserving the v30 optimum.
        #   * 420mm hard reach and 65mm path clearance are unchanged.
        best_local=None
        best_local_orig=None
        na=len(ameta_pool); nh=len(hmeta)
        pairs_checked=int(na*nh); pairs_sep_pass=0; assignment_checks=0
        reject_radius=0; reject_path=0; reject_missing=0; reject_cross=0; reject_side=0
        best_unsafe_radius=float('inf'); best_unsafe_clear=-1.0
        early_exit=False; candidate_pairs=0; cross_checks=0; path_checks=0
        if na<=0 or nh<=0:
            return None,{
                'mode':str(mode),'pairs':pairs_checked,'sep_pass':0,'assignment_checks':0,
                'reject_missing':0,'reject_radius':0,'reject_path':0,'reject_cross':0,'reject_side':0,
                'best_unsafe_radius':None,'best_unsafe_clear':-1.0,'candidate_pairs':0,
                'cross_checks':0,'path_checks':0,'early_exit':False,
            }

        try:
            a_board=np.stack([np.asarray(z[0][1],dtype=np.float64).reshape(2) for z in ameta_pool],axis=0)
            h_board=np.stack([np.asarray(z[0][1],dtype=np.float64).reshape(2) for z in hmeta],axis=0)
            a_score=np.asarray([float(z[0][0]) for z in ameta_pool],dtype=np.float64)
            h_score=np.asarray([float(z[0][0]) for z in hmeta],dtype=np.float64)
            sep_mat=np.linalg.norm(a_board[:,None,:]-h_board[None,:,:],axis=2)
            sep_mask=(sep_mat>=66.0)&(sep_mat<=126.0)
            pairs_sep_pass=int(np.count_nonzero(sep_mask))
            valid=sep_mask.copy()
            if _v30_role_gate_active:
                a_side=(a_board-anchor_board[None,:])@u_board
                h_side=(h_board-anchor_board[None,:])@u_board
                side_mask=(a_side[:,None]>0.0)&(h_side[None,:]<0.0)
                reject_side=int(np.count_nonzero(sep_mask & (~side_mask)))
                valid &= side_mask
            ij=np.argwhere(valid)
            candidate_pairs=int(ij.shape[0])
            if candidate_pairs<=0:
                return None,{
                    'mode':str(mode),'pairs':pairs_checked,'sep_pass':pairs_sep_pass,
                    'assignment_checks':0,'reject_missing':0,'reject_radius':0,'reject_path':0,
                    'reject_cross':0,'reject_side':reject_side,'best_unsafe_radius':None,
                    'best_unsafe_clear':-1.0,'candidate_pairs':0,'cross_checks':0,
                    'path_checks':0,'early_exit':False,
                }
            # q <= active_score + hold_score + 0.20(sep) + 0.10(radius); travel can only subtract.
            upper=a_score[ij[:,0]]+h_score[ij[:,1]]+0.30
            target=_v30_target_board(args)
            if target is not None:
                mids=0.5*(a_board[ij[:,0]]+h_board[ij[:,1]])
                tdist=np.linalg.norm(mids-np.asarray(target,dtype=np.float64).reshape(1,2),axis=1)
            else:
                tdist=np.zeros(candidate_pairs,dtype=np.float64)
            # Primary key = descending exact upper bound. Target proximity only breaks ties.
            order=np.lexsort((tdist,-upper))
        except Exception:
            # Fail closed to a deterministic scalar prefilter if vector setup ever fails.
            _tmp=[]
            pairs_sep_pass=0; reject_side=0
            for ai,(ac,_,_) in enumerate(ameta_pool):
                ab=np.asarray(ac[1],dtype=np.float64)
                for hi,(hc,_) in enumerate(hmeta):
                    hb=np.asarray(hc[1],dtype=np.float64)
                    sep=float(np.linalg.norm(ab-hb))
                    if not (66.0<=sep<=126.0):
                        continue
                    pairs_sep_pass+=1
                    if _v30_role_gate_active:
                        if not (float(np.dot(ab-anchor_board,u_board))>0.0 and float(np.dot(hb-anchor_board,u_board))<0.0):
                            reject_side+=1
                            continue
                    ub=float(ac[0])+float(hc[0])+0.30
                    _tmp.append((ai,hi,ub,float(ai*max(1,nh)+hi)))
            _tmp.sort(key=lambda z:(-z[2],z[3]))
            ij=np.asarray([[z[0],z[1]] for z in _tmp],dtype=np.int64)
            upper=np.asarray([z[2] for z in _tmp],dtype=np.float64)
            order=np.arange(len(_tmp),dtype=np.int64)
            candidate_pairs=int(len(_tmp))

        for _rank,oi in enumerate(order):
            ai=int(ij[int(oi),0]); hi=int(ij[int(oi),1])
            ac,pull_core,ameta=ameta_pool[ai]
            hc,harm=hmeta[hi]
            _,ab,ap,ad,adt,ar,aprom=ac
            _,hb,hp,hd,hdt,hr,hprom=hc
            sep=float(np.linalg.norm(ab-hb))

            if _v30_role_gate_active:
                _xkey=(id(ac),id(hc))
                if _xkey in _v31_cross_cache:
                    _cross_ok=bool(_v31_cross_cache[_xkey])
                else:
                    cross_checks+=1
                    _cross_ok=bool(_v31_segment_crosses_crease_prepared(ap,hp,_v31_role_crease_near))
                    _v31_cross_cache[_xkey]=_cross_ok
                if not _cross_ok:
                    reject_cross+=1
                    continue

            assignments=[]
            for active_arm,hold_arm in (('arm1','arm2'),('arm2','arm1')):
                assignment_checks+=1
                ainfo=ameta.get(active_arm); hinfo=harm.get(hold_arm)
                if not isinstance(ainfo,dict) or not isinstance(hinfo,dict):
                    reject_missing+=1
                    continue
                maxrad=max(float(ainfo['active_pull_radius']),float(hinfo['hold_radius']))
                best_unsafe_radius=min(best_unsafe_radius,float(maxrad))
                if maxrad>420.0:
                    reject_radius+=1
                    continue
                _pkey=(id(ac),str(active_arm),id(hc))
                if _pkey in _v31_path_clear_cache:
                    clear=float(_v31_path_clear_cache[_pkey])
                else:
                    path_checks+=1
                    clear=float(np.min(np.linalg.norm(np.asarray(ainfo['path_board'])-hb[None,:],axis=1)))
                    _v31_path_clear_cache[_pkey]=clear
                best_unsafe_clear=max(best_unsafe_clear,float(clear))
                if clear<65.0:
                    reject_path+=1
                    continue
                travel=float(ainfo['active_travel'])+float(hinfo['hold_travel'])
                assignments.append((travel,active_arm,hold_arm,maxrad,clear))
            if assignments:
                assignments.sort(key=lambda z:z[0])
                travel,aa,ha,maxrad,pathclear=assignments[0]
                q=float(ac[0]+hc[0]+0.20*np.exp(-abs(sep-88.0)/18.0)+0.10*np.clip((420.0-maxrad)/70.0,0.0,1.0)-0.00015*travel)
                orig=int(ai*nh+hi)
                if (best_local is None or q>best_local[0] or (q==best_local[0] and (best_local_orig is None or orig<best_local_orig))):
                    best_local=(q,ac,hc,pull_core,aa,ha,pathclear,maxrad)
                    best_local_orig=orig

            # Strict branch-and-bound: only stop if best q is GREATER than every
            # remaining mathematical upper bound. Equality keeps scanning so v30
            # first-in-original-order tie behavior remains reproducible.
            if best_local is not None and (_rank+1)<len(order):
                next_upper=float(upper[int(order[_rank+1])])
                if float(best_local[0])>next_upper+1e-12:
                    early_exit=True
                    break

        return best_local,{
            'mode':str(mode),'pairs':int(pairs_checked),'sep_pass':int(pairs_sep_pass),
            'assignment_checks':int(assignment_checks),'reject_missing':int(reject_missing),
            'reject_radius':int(reject_radius),'reject_path':int(reject_path),
            'reject_cross':int(reject_cross),'reject_side':int(reject_side),
            'best_unsafe_radius':None if not np.isfinite(best_unsafe_radius) else float(best_unsafe_radius),
            'best_unsafe_clear':float(best_unsafe_clear),'candidate_pairs':int(candidate_pairs),
            'cross_checks':int(cross_checks),'path_checks':int(path_checks),'early_exit':bool(early_exit),
        }

    hold_meta=_v9_hold_meta_from(hold_candidates,limit=40)
    best,diag=_v9_search_pairs(active_meta,hold_meta,mode='BASE')
    recovery_used=False
    recovery_mode='NONE'
    recovery_diag=None
    dual_recovery_diag=None
    reach_meta=[]

    # First recovery tier: preserve V9 behavior and move only HOLD tangentially.
    if (best is None and (
            int(diag.get('reject_radius',0))>0 or int(diag.get('reject_cross',0))>0 or int(diag.get('reject_side',0))>0)
            and bool(getattr(args,'micro_unfold_vla_late_reach_recovery',True))
            and hold_candidates_reach):
        reach_meta=_v9_hold_meta_from(hold_candidates_reach,limit=None)
        best,recovery_diag=_v9_search_pairs(active_meta,reach_meta,mode='SAME_CREASE_HOLD_TANGENT')
        if best is not None:
            recovery_used=True
            recovery_mode='SAME_CREASE_HOLD_TANGENT'
            print(
                f"[E156-V10-REACH-RECOVERY-ACCEPT] mode=HOLD_TANGENT sameCrease=True "
                f"hard420=PASS path65=PASS baseBestRadius={diag.get('best_unsafe_radius')} "
                f"candidates={len(reach_meta)}"
            )
        else:
            print(
                f"[E156-V10-REACH-RECOVERY-TIER1-BLOCK] sameCrease=True hard420=UNCHANGED "
                f"baseBestRadius={diag.get('best_unsafe_radius')} recoveryBestRadius="
                f"{None if recovery_diag is None else recovery_diag.get('best_unsafe_radius')}"
            )

    # V10 second recovery tier: if reach is still the blocker, move BOTH ACTIVE
    # and HOLD along the SAME selected crease tangent. The fold side, crease mask,
    # normal axis, 66..126mm pair spacing, 65mm path clearance and 420mm hard reach
    # remain unchanged. Both arm assignments are still evaluated for every pair.
    geometry_still_blocked = bool(
        best is None and (
            int(diag.get('reject_radius',0))>0 or int(diag.get('reject_cross',0))>0 or int(diag.get('reject_side',0))>0 or
            int((recovery_diag or {}).get('reject_radius',0))>0 or int((recovery_diag or {}).get('reject_cross',0))>0 or int((recovery_diag or {}).get('reject_side',0))>0
        )
    )
    if (geometry_still_blocked
            and bool(getattr(args,'micro_unfold_vla_dual_reach_recovery',True))):
        active_meta_reach=_v10_active_meta_from(active_candidates_reach,limit=72)
        # Keep high-quality base HOLDs and the extra same-crease tangent HOLD pool.
        combo_holds=list(hold_candidates[:56])+list(hold_candidates_reach)
        combo_hold_meta=_v9_hold_meta_from(combo_holds,limit=None)
        best,dual_recovery_diag=_v9_search_pairs(
            active_meta_reach,combo_hold_meta,mode='SAME_CREASE_ACTIVE_HOLD_TANGENT')
        if best is not None:
            recovery_used=True
            recovery_mode='SAME_CREASE_ACTIVE_HOLD_TANGENT'
            print(
                f"[E156-V10-REACH-RECOVERY-ACCEPT] mode=ACTIVE_HOLD_TANGENT sameCrease=True "
                f"hard420=PASS path65=PASS activeCandidates={len(active_meta_reach)} "
                f"holdCandidates={len(combo_hold_meta)}"
            )
        else:
            print(
                f"[E156-V10-REACH-RECOVERY-BLOCK] sameCrease=True hard420=UNCHANGED "
                f"baseBestRadius={diag.get('best_unsafe_radius')} tier1BestRadius="
                f"{None if recovery_diag is None else recovery_diag.get('best_unsafe_radius')} "
                f"tier2BestRadius={None if dual_recovery_diag is None else dual_recovery_diag.get('best_unsafe_radius')}"
            )

    joint_ms=1000.0*(time.perf_counter()-joint_t0)
    globals()['_E156_LAST_DIRECTIONAL_PAIR_MS']=float(joint_ms)
    if recovery_mode=='SAME_CREASE_ACTIVE_HOLD_TANGENT' and isinstance(dual_recovery_diag,dict):
        final_diag=dual_recovery_diag
    elif recovery_mode=='SAME_CREASE_HOLD_TANGENT' and isinstance(recovery_diag,dict):
        final_diag=recovery_diag
    else:
        final_diag=diag
    print(
        f"[E156-V10-PAIR-TIME] activeBase={len(active_meta)}/{min(30,len(active_candidates))} "
        f"activeRecovery={len(active_candidates_reach)} holdBase={len(hold_meta)}/{min(40,len(hold_candidates))} "
        f"holdRecovery={len(hold_candidates_reach)} mode={recovery_mode if recovery_used else 'BASE'} "
        f"pairs={final_diag.get('pairs')} sepPass={final_diag.get('sep_pass')} "
        f"assignmentChecks={final_diag.get('assignment_checks')} total={joint_ms:.1f}ms"
    )
    print(
        f"[E156-V31-PAIR-OPT] mode={final_diag.get('mode')} candidatePairs={final_diag.get('candidate_pairs')} "
        f"crossChecks={final_diag.get('cross_checks')} pathChecks={final_diag.get('path_checks')} "
        f"earlyExit={final_diag.get('early_exit')} crossCache={len(_v31_cross_cache)} "
        f"pathCache={len(_v31_path_clear_cache)} activeMetaCache={len(_v31_active_meta_cache)} "
        f"holdMetaCache={len(_v31_hold_meta_cache)}"
    )
    if best is None:
        last_diag=dual_recovery_diag if isinstance(dual_recovery_diag,dict) else (recovery_diag if isinstance(recovery_diag,dict) else diag)
        print(
            f"[E156-V10-PAIR-BLOCK-DIAG] missing={last_diag.get('reject_missing')} "
            f"radius420={last_diag.get('reject_radius')} path65={last_diag.get('reject_path')} "
            f"cross={last_diag.get('reject_cross')} side={last_diag.get('reject_side')} "
            f"bestRadius={last_diag.get('best_unsafe_radius')} bestPathClear={last_diag.get('best_unsafe_clear'):.1f}"
        )
        return None
    q,ac,hc,pull_core,aa,ha,pathclear,maxrad=best
    _,ab,ap,ad,adt,ar,aprom=ac; _,hb,hp,hd,hdt,hr,hprom=hc
    pull_board,used,exit_mm,outside,pull_px,pu=pull_core
    _step=float(np.clip(float(getattr(args,'micro_unfold_boundary_sample_mm',2.0)),1.0,5.0))
    _max_pull=max(60.0,float(getattr(args,'micro_unfold_max_pull_mm',264.0)))
    _hexit=_e130_ray_exit(mask,H_value,hb,pu,max_mm=_max_pull+220.0,step_mm=_step,stable_mask=stable_ray,invH=invH)
    hold_exit=float(_hexit[0]) if _hexit is not None else float(exit_mm+np.linalg.norm(np.asarray(ab)-np.asarray(hb)))
    bang=0.0; bok=True
    sep=float(np.linalg.norm(ab-hb))
    print(f'[E130-DIRECTIONAL-PAIR] axis={axis_source} group={group} realCrease={real_crease} evidenceProj={evidence_proj:.1f}px ACTIVE={ad:.1f}mm HOLD={hd:.1f}mm sep={sep:.1f}mm activeExit={exit_mm:.1f}mm pull={used:.1f}mm pathClear={pathclear:.1f}mm READY=FORCED')
    return {
        'quality':float(q),'crease_source':'E130_DIRECTIONAL_REAL_CREASE' if real_crease else 'E130_DIRECTIONAL_VIRTUAL_CREASE',
        'active_px':np.asarray(ap,dtype=np.float64),'hold_px':np.asarray(hp,dtype=np.float64),'garment_center_px':gc.copy(),
        'active_board':np.asarray(ab,dtype=np.float64),'hold_board':np.asarray(hb,dtype=np.float64),'crease_anchor_px':np.asarray(anchor_px,dtype=np.float64),
        'component_label':-130,'component_span_px':float(cv2.countNonZero(selected_comp)),'component_strength':0.85 if real_crease else 0.55,
        'fused_support':0.0,'body_support':1.0,'coarse_lobe_support':0.5,'coarse_lobe_quality':0.0,'surface_asymmetry':0.5,'seam_penalty':0.0,'reach_support':1.0,
        'crease_flap_soft_support':1.0,'pose_fold_support':1.0 if source=='POSE_INTRUSION' else 0.5,'pose_fold_role_support':1.0,
        'pose_deform_available':bool(depth_state.get('pose_deform_available',False)),'pose_deform_confidence':float(depth_state.get('pose_deform_confidence',0.0) or 0.0),
        'pose_deform_mode':str(depth_state.get('pose_deform_mode','NONE')),'pose_deform_side_hint':str(depth_state.get('pose_deform_side_hint','NONE')),
        'pose_deform_anchor_support':1.0,'pose_deform_role_support':1.0,'pose_deform_authoritative':bool(source=='POSE_INTRUSION'),
        'pose_fold_region_hint':str(group),'pose_fold_region_confidence':float(max(conf,0.55 if source=='POSE_INTRUSION' else conf)),
        'pose_direction_alignment':1.0,'pose_axis_progress':1.0,'pose_axis_support':1.0,
        'pose_split_lobe_count':0,'pose_split_body_count':0,'pose_split_assigned':0,'pose_split_gap':0,'pose_split_confidence':0.0,
        'pose_split_preferred_active_region':'NONE','pose_split_authoritative':False,'pose_count_role_support':0.5,
        'depth_role_probe_valid':True,'depth_role_probe_delta_mm':float(ar-hr),'depth_role_probe_confidence':float(np.tanh((ar-hr)/2.0)),
        'orientation_evidence_ok':True,'rescue_evidence_ok':True,'hold_clean_score':float(hc[0]),'hold_crease_clear_px':float(crease_dist[int(round(hp[1])),int(round(hp[0]))]),
        'hold_crease_density':0.0,'hold_clean_advantage':0.5,'depth_role_valid':True,'active_depth_role_mm':float(ar),'hold_depth_role_mm':float(hr),
        'depth_role_delta_mm':float(ar-hr),'depth_role_confidence':float(np.tanh((ar-hr)/2.0)),
        'active_crease_distance_mm':float(ad),'hold_crease_distance_mm':float(hd),'pair_separation_mm':float(sep),
        'pair_spacing_mode':'E130_DIRECTIONAL_AXIS','outer_boundary_perpendicular_angle_deg':90.0,'hold_to_active_outward_cos':1.0,
        'hold_normal_target_error_mm':0.0,'active_depth_reference':float(ar),'hold_band_mode':'E130_BODY_FAR_FROM_CREASE',
        'outer_boundary_mode':'E130_DIRECTION_TO_CLOTH_CONTOUR','active_inside_flap':True,'hold_inside_body':True,
        'selected_crease_component':selected_comp.copy(),'coarse_lobe_region':fold_extent.copy(),'coarse_body_region':cv2.bitwise_and(mask,cv2.bitwise_not(fold_extent)),
        'crease_flap_region':fold_extent.copy(),'crease_body_region':cv2.bitwise_and(mask,cv2.bitwise_not(fold_extent)),
        'crease_partition_info':{'mode':'E130_DIRECTIONAL_AXIS','axis_source':axis_source,'real_crease':bool(real_crease),'evidence_px':evidence_px.tolist()},
        'primary_fold_region_source':('POSE_DIRECTION_CREASE_EXTENT' if source=='POSE_INTRUSION' else 'DIRECTION_CREASE_EXTENT'),
        'primary_fold_region_confidence':float(max(conf,0.55)),'primary_fold_region_group':str(group),'primary_fold_region_authoritative':True,
        'primary_fold_region':fold_extent.copy(),'primary_fold_boundary':selected_comp.copy(),
        'active_primary_fold_support':1.0,'hold_primary_fold_support':0.0,'fold_region_role_support':1.0,'fold_boundary_component_support':1.0,
        'fold_region_component_overlap':1.0,'fold_region_component_deep_overlap':0.0,
        'active_selection_mode':'E130_CREASE_INWARD_ACTIVE_UNIDEPTH','active_pose_depth_authority':1.0 if source=='POSE_INTRUSION' else 0.6,
        'active_pose_region_raised_mm':float(ar),'e130_directional_plan':True,
        'e156_v9_reach_recovery_used':bool(recovery_used),
        'e156_v9_reach_recovery_mode':str(recovery_mode),
        'e156_v10_reach_recovery_used':bool(recovery_used),
        'e156_v10_reach_recovery_mode':str(recovery_mode),
        'e130_unfold_unit_board':np.asarray(u_board,dtype=np.float64),
        'e130_direction_evidence_px':np.asarray(evidence_px,dtype=np.float64),'e130_active_to_contour_mm':float(exit_mm),
        'prechecked_pull_board':np.asarray(pull_board,dtype=np.float64),'prechecked_pull_distance_mm':float(used),
        'prechecked_garment_exit_mm':float(exit_mm),'prechecked_outside_used_mm':float(outside),'prechecked_pull_px':np.asarray(pull_px,dtype=np.float64),
        'prechecked_pull_unit_board':np.asarray(pu,dtype=np.float64),'prechecked_hold_boundary_distance_mm':float(hold_exit),
        'prechecked_boundary_normal_delta_deg':0.0,'prechecked_boundary_normal_ok':True,
        'execution_evidence_ok':True,'execution_evidence_reason':'E130_FORCED_READY_VALID_MASK_AND_MECHANICS',
        'orientation_confidence':1.0,
        'v30_target_guided_planner':bool(_v30_target_board(args) is not None),
        'v30_target_joint_distance_mm':depth_state.get('v30_target_joint_distance_mm'),
        'v30_pair_cross_valid':bool((not _v30_role_gate_active) or _v31_segment_crosses_crease_prepared(ap,hp,_v31_role_crease_near)),
        'v31_pair_search_optimized':True,
    }


def _e133_rotate_unit_px(u, deg):
    """Rotate a 2-D image-space unit vector by ``deg`` degrees."""
    v=np.asarray(u,dtype=np.float64).reshape(2)
    nv=float(np.linalg.norm(v))
    if nv<1e-9:
        return None
    v=v/nv
    a=np.deg2rad(float(deg)); c=float(np.cos(a)); sn=float(np.sin(a))
    out=np.asarray([c*v[0]-sn*v[1], sn*v[0]+c*v[1]],dtype=np.float64)
    out/=max(1e-9,float(np.linalg.norm(out)))
    return out


def _e133_axis_region(mask_u8, gc_px, u_px, preferred_proj_px=None):
    """Create a tiny synthetic authoritative region on a requested outward axis.

    E130 only needs a region centroid to derive a direction when Pose intrusion is
    unavailable.  E133 uses this helper to test several nearby UniDepth axes without
    pretending that the tiny disk is the full folded part.
    """
    mask=((np.asarray(mask_u8,dtype=np.uint8)>0).astype(np.uint8)*255)
    h,w=mask.shape[:2]
    gc=np.asarray(gc_px,dtype=np.float64).reshape(2)
    u=np.asarray(u_px,dtype=np.float64).reshape(2)
    u/=max(1e-9,float(np.linalg.norm(u)))
    # Find the last stable cloth point from garment centre along this axis.
    stable=cv2.erode(mask,np.ones((3,3),np.uint8),iterations=1)
    last=None; exit_proj=0.0
    for d in np.arange(8.0,520.0,2.0):
        q=gc+u*float(d); x=int(round(q[0])); y=int(round(q[1]))
        if not (0<=x<w and 0<=y<h) or stable[y,x]<=0:
            break
        last=q.copy(); exit_proj=float(d)
    if last is None or exit_proj<24.0:
        return None
    if preferred_proj_px is None or not np.isfinite(preferred_proj_px):
        proj=0.68*exit_proj
    else:
        proj=float(np.clip(preferred_proj_px,0.35*exit_proj,0.82*exit_proj))
    proj=float(np.clip(proj,18.0,max(18.0,exit_proj-6.0)))
    q=gc+u*proj
    reg=np.zeros_like(mask,dtype=np.uint8)
    cv2.circle(reg,tuple(np.rint(q).astype(int)),8,255,-1,cv2.LINE_AA)
    reg=cv2.bitwise_and(reg,mask)
    if cv2.countNonZero(reg)<8:
        cv2.circle(reg,tuple(np.rint(q).astype(int)),14,255,-1,cv2.LINE_AA)
        reg=cv2.bitwise_and(reg,mask)
    return reg if cv2.countNonZero(reg)>=8 else None


def _e133_collect_depth_base_axes(depth_state, mask_u8, fold_region_info):
    """Return meaningful UniDepth/crease outward axes before angular expansion."""
    mask=((np.asarray(mask_u8,dtype=np.uint8)>0).astype(np.uint8)*255)
    if cv2.countNonZero(mask)<50:
        return []
    mm=cv2.moments(mask,binaryImage=True)
    gc=np.asarray([float(mm['m10']/max(1e-6,mm['m00'])),float(mm['m01']/max(1e-6,mm['m00']))],dtype=np.float64)
    out=[]
    def add(name,p,priority):
        if p is None: return
        p=np.asarray(p,dtype=np.float64).reshape(2)
        d=p-gc; nd=float(np.linalg.norm(d))
        if not np.all(np.isfinite(d)) or nd<12.0: return
        u=d/nd
        # Deduplicate nearly parallel axes; keep earlier/higher-priority source.
        for q in out:
            if float(np.dot(u,q['u']))>0.985:
                return
        out.append({'name':str(name),'u':u,'proj':float(nd),'priority':float(priority)})

    # 1) Authoritative raised-region centroid, depth weighted.
    region=np.asarray(fold_region_info.get('region_mask',np.zeros_like(mask)),dtype=np.uint8)
    if cv2.countNonZero(region)>=8:
        ys,xs=np.where(region>0)
        relief=np.asarray(depth_state.get('local_relief_mm',np.zeros(mask.shape,dtype=np.float32)),dtype=np.float32)
        if relief.shape==mask.shape:
            wt=np.maximum(relief[ys,xs],0.0)+0.5
        else:
            wt=np.ones_like(xs,dtype=np.float64)
        sw=float(np.sum(wt))
        if sw>1e-6:
            add('UNIDEPTH_REGION', [float(np.sum(xs*wt)/sw),float(np.sum(ys*wt)/sw)], 1.00)

    # 2) Each strict UniDepth candidate centroid, selected candidate first.
    labels=np.asarray(depth_state.get('strict_candidate_labels',np.zeros_like(mask)),dtype=np.uint8)
    sel=int(depth_state.get('selected_donor_rank',depth_state.get('selected_rank',1)) or 1)
    ranks=[sel]+[r for r in (1,2,3) if r!=sel]
    for k,r in enumerate(ranks):
        yy,xx=np.where(labels==int(r))
        if xx.size>=4:
            add(f'UNIDEPTH_C{r}', [float(np.mean(xx)),float(np.mean(yy))], 0.94-0.05*k)

    # 3) Maximum local relief point inside cloth.
    relief=np.asarray(depth_state.get('local_relief_mm',np.zeros(mask.shape,dtype=np.float32)),dtype=np.float32)
    inside=cv2.distanceTransform(mask,cv2.DIST_L2,5)
    valid=(mask>0)&np.isfinite(relief)&(inside>=4.0)
    if np.any(valid):
        yy,xx=np.where(valid); vv=relief[valid]; j=int(np.argmax(vv))
        add('MAX_RELIEF',[float(xx[j]),float(yy[j])],0.74)

    # 4) Outermost strong crease point as a geometry fallback.
    crease=np.asarray(depth_state.get('e122_fold_execution_skeleton',np.zeros_like(mask)),dtype=np.uint8)
    if cv2.countNonZero(crease)<8:
        crease=np.asarray(depth_state.get('fold_crease_structural_skeleton',np.zeros_like(mask)),dtype=np.uint8)
    yy,xx=np.where(crease>0)
    if xx.size>=4:
        d2=(xx-gc[0])**2+(yy-gc[1])**2; j=int(np.argmax(d2))
        add('CREASE_OUTER',[float(xx[j]),float(yy[j])],0.58)
    out.sort(key=lambda q:q['priority'],reverse=True)
    return out


def _e133_directional_pair(depth_state,mask_u8,H_value,args,config,cfg,fold_region_info,exec_crease,preview_crease):
    """E133 direction arbitration.

    1) Any pose landmark already admitted by E117 as a contour intrusion is a
       directional cue even when E119 reports side/region NONE.  Group-specific
       intrusion axes are attempted in confidence/strength order.
    2) If no pose axis yields a mechanically feasible pair, UniDepth is no longer
       represented by one centroid direction.  Several depth-derived axes and
       angular perturbations are tested.  The closest-to-evidence feasible axis wins.
    3) A coarse all-angle geometric sweep is the last-resort I->READY attempt.
       It still cannot bypass board/reach/65-mm inter-arm hard safety.
    """
    mask=((np.asarray(mask_u8,dtype=np.uint8)>0).astype(np.uint8)*255)
    if cv2.countNonZero(mask)<500:
        return None
    mm=cv2.moments(mask,binaryImage=True)
    gc=np.asarray([float(mm['m10']/max(1e-6,mm['m00'])),float(mm['m01']/max(1e-6,mm['m00']))],dtype=np.float64)

    # ------------------------------------------------------------------
    # A. POSE ANY-INTRUSION AUTHORITY.
    # E117 already required each listed point to be inside the current cloth by
    # more than its semantic threshold.  Do not throw that direction evidence away
    # merely because torso/side classification says NONE.
    # ------------------------------------------------------------------
    intr=[q for q in list(depth_state.get('pose_fold_intrusions',[]) or []) if isinstance(q,dict)]
    groups={}
    for q in intr:
        try:
            conf=float(q.get('conf',0.0)); strength=float(q.get('strength',0.0))
            excess=float(q.get('inside_distance_px',0.0))-float(q.get('threshold_px',0.0))
        except Exception:
            continue
        if conf<0.20 or excess<=0.5:
            continue
        g=str(q.get('group','OTHER'))
        merit=max(0.01,conf)*(0.20+0.80*max(0.0,strength))*(1.0+0.015*max(0.0,excess))
        groups.setdefault(g,[]).append((merit,q))
    ranked=[]
    for g,items in groups.items():
        merit=float(sum(v[0] for v in items))
        ranked.append((merit,g,items))
    ranked.sort(reverse=True,key=lambda z:z[0])
    for merit,g,items in ranked:
        fake=dict(fold_region_info or {})
        fake['source']='POSE_INTRUSION'; fake['group']=str(g)
        fake['confidence']=float(np.clip(0.42+0.18*min(2,len(items))+0.18*np.tanh(merit),0.0,0.98))
        print(f"[E133-POSE-AXIS-TRY] group={g} points={len(items)} merit={merit:.3f} sideClass={str(depth_state.get('pose_deform_side_hint','NONE'))} -> authority=ANY_INTRUDED_KPT")
        try:
            pair=_e130_directional_pair(depth_state,mask,H_value,args,config,cfg,fake,exec_crease,preview_crease)
        except Exception as exc:
            print(f"[E133-POSE-AXIS-REJECT] group={g} reason={exc!r}")
            pair=None
        if isinstance(pair,dict):
            pair['e133_axis_mode']='POSE_ANY_INTRUSION'
            pair['e133_axis_group']=str(g)
            pair['e133_pose_side_class_bypassed']=True
            print(f"[E133-POSE-AXIS-ACCEPT] group={g} -> READY; E119 side/region confidence cannot veto this admitted intrusion direction")
            return pair

    # ------------------------------------------------------------------
    # B. UNIDEPTH MULTI-AXIS SEARCH.
    # Try the original evidence axis first, then ±15..±90 degrees.  Multiple
    # candidate centroids seed independent bases.  This fixes the E132 case where
    # one depth-centroid axis puts ACTIVE or HOLD outside the shirt despite many
    # excellent depthOK crease branches being present.
    # ------------------------------------------------------------------
    bases=_e133_collect_depth_base_axes(depth_state,mask,fold_region_info)
    angle_order=(0.0,-15.0,15.0,-30.0,30.0,-45.0,45.0,-60.0,60.0,-75.0,75.0,-90.0,90.0)
    seen=[]; attempts=0
    max_attempts=int(np.clip(int(getattr(args,'micro_unfold_e133_axis_attempts',36)),12,72)) if args is not None else 36
    for base in bases:
        for ang in angle_order:
            u=_e133_rotate_unit_px(base['u'],ang)
            if u is None: continue
            if any(float(np.dot(u,v))>0.992 for v in seen):
                continue
            seen.append(u); attempts+=1
            if attempts>max_attempts: break
            reg=_e133_axis_region(mask,gc,u,base.get('proj'))
            if reg is None: continue
            fake={'source':'UNIDEPTH_MULTIAXIS','group':str(base['name']),
                  'confidence':float(fold_region_info.get('confidence',0.0) or 0.0),
                  'region_mask':reg}
            print(f"[E133-DEPTH-AXIS-TRY] base={base['name']} rotate={ang:+.0f}deg attempt={attempts}/{max_attempts}")
            try:
                pair=_e130_directional_pair(depth_state,mask,H_value,args,config,cfg,fake,exec_crease,preview_crease)
            except Exception as exc:
                pair=None
            if isinstance(pair,dict):
                pair['e133_axis_mode']='UNIDEPTH_MULTIAXIS'
                pair['e133_axis_group']=str(base['name'])
                pair['e133_axis_rotation_deg']=float(ang)
                pair['primary_fold_region_source']='UNIDEPTH_MULTIAXIS_CREASE_EXTENT'
                print(f"[E133-DEPTH-AXIS-ACCEPT] base={base['name']} rotate={ang:+.0f}deg -> READY after {attempts} axis tries")
                return pair
        if attempts>max_attempts: break

    # ------------------------------------------------------------------
    # C. LAST-RESORT GEOMETRIC AXIS SWEEP.
    # User requested I->X whenever cloth is present.  We therefore attempt evenly
    # spaced axes only after Pose and UniDepth directions fail.  This is still a
    # planning fallback, not a safety bypass: E130 retains board/reach/path gates.
    # ------------------------------------------------------------------
    force_geo=bool(getattr(args,'micro_unfold_e133_geometric_last_resort',True)) if args is not None else True
    if force_geo:
        # If a depth base exists, start around it; otherwise use image +X as phase.
        phase=np.asarray(bases[0]['u'],dtype=np.float64) if bases else np.asarray([1.0,0.0],dtype=np.float64)
        for ang in range(0,360,30):
            u=_e133_rotate_unit_px(phase,float(ang))
            if u is None: continue
            if any(float(np.dot(u,v))>0.992 for v in seen): continue
            reg=_e133_axis_region(mask,gc,u,None)
            if reg is None: continue
            fake={'source':'GEOMETRIC_LAST_RESORT','group':f'A{ang:03d}','confidence':0.0,'region_mask':reg}
            print(f"[E133-GEOMETRIC-AXIS-TRY] rotate={ang:+03d}deg perceptionVeto=OFF hardSafety=ON")
            try:
                pair=_e130_directional_pair(depth_state,mask,H_value,args,config,cfg,fake,exec_crease,preview_crease)
            except Exception:
                pair=None
            if isinstance(pair,dict):
                pair['e133_axis_mode']='GEOMETRIC_LAST_RESORT'
                pair['e133_axis_rotation_deg']=float(ang)
                pair['execution_evidence_reason']='E133_GEOMETRIC_LAST_RESORT_MECHANICALLY_SAFE'
                print(f"[E133-GEOMETRIC-AXIS-ACCEPT] rotate={ang:+03d}deg -> READY; only physical hard-safety can still block X")
                return pair
    print(f"[E133-DIRECTIONAL-FAIL] poseGroups={len(ranked)} depthBases={len(bases)} depthAxisTries={attempts}; no board/reach/path-safe pair exists")
    return None

def _e120_pose_deformation_state(image_shape, garment_mask_u8, pose_state, args=None, intrusion_state=None):
    """E120 directional fold prior from deformation of the learned 12-point topology.

    In addition to E119's chain/intrusion evidence, E120 explicitly recognizes the
    four fold directions used in the physical tests:
      * LOWER_LEFT  : WAIST_LEFT corner (8) becomes unusually obtuse vs WAIST_RIGHT.
      * LOWER_RIGHT : WAIST_RIGHT corner (7) becomes unusually obtuse vs WAIST_LEFT.
      * UPPER_LEFT  : 0/1/11/10/9 side compresses/straightens or intrudes inward.
      * UPPER_RIGHT : 2/3/4/5/6 side compresses/straightens or intrudes inward.

    ``fold_direction_px`` points from the garment BODY centre toward the inferred
    folded side.  Downstream Fold-Crease selection uses it as a direction prior;
    pose alone never paints a cyan crease and never bypasses robot safety checks.
    """
    h, w = image_shape[:2]
    garment = ((np.asarray(garment_mask_u8, dtype=np.uint8) > 0).astype(np.uint8) * 255)
    prior = np.zeros((h, w), dtype=np.uint8)
    empty = {
        "available": False, "confidence": 0.0, "mode": "NONE", "side_hint": "NONE",
        "fold_region_hint": "NONE", "fold_region_confidence": 0.0,
        "fold_anchor_px": None, "fold_direction_px": None,
        "prior_mask": prior, "left_chain_score": 0.0, "right_chain_score": 0.0,
        "torso_quad_score": 0.0, "torso_max_angle_deg": 0.0,
        "torso_max_angle_vertex": "NONE", "torso_angles_deg": {},
        "lower_left_score": 0.0, "lower_right_score": 0.0,
        "upper_left_score": 0.0, "upper_right_score": 0.0,
        "intrusion_score": 0.0, "intrusion_count": 0, "signals": [],
    }
    if cv2.countNonZero(garment) < 500 or not isinstance(pose_state, dict):
        return empty

    named = dict(pose_state.get("named_points", {}) or {})
    conf_min = float(getattr(args, "micro_unfold_pose_deform_kpt_conf", 0.20)) if args is not None else 0.20
    mm = cv2.moments(garment, binaryImage=True)
    garment_center = np.asarray([
        float(mm['m10']/max(1e-6,mm['m00'])), float(mm['m01']/max(1e-6,mm['m00']))
    ], dtype=np.float64)

    def getp(name):
        q = dict(named.get(name, {}) or {})
        try:
            x = float(q.get("x")); y = float(q.get("y")); c = float(q.get("conf", 0.0))
        except Exception:
            return None
        if not (np.isfinite(x) and np.isfinite(y) and np.isfinite(c)) or c < conf_min:
            return None
        return np.asarray([x, y], dtype=np.float64)

    def chain_state(names):
        pts = [getp(n) for n in names]
        valid = [(n, p) for n, p in zip(names, pts) if p is not None]
        result = {"score":0.0,"valid":len(valid),"linearity_ratio":0.0,
                  "pca_minor_ratio":1.0,"span_px":0.0,"points":valid}
        if len(valid) < 4:
            return result
        arr = np.asarray([p for _, p in valid], dtype=np.float64)
        path = float(np.sum(np.linalg.norm(arr[1:]-arr[:-1], axis=1)))
        chord = float(np.linalg.norm(arr[-1]-arr[0]))
        if path <= 1e-6:
            return result
        line_ratio = float(np.clip(chord/path,0.0,1.0))
        ctr = np.mean(arr,axis=0)
        cov = (arr-ctr).T@(arr-ctr)
        ev = np.sort(np.maximum(np.linalg.eigvalsh(cov),1e-9))
        pca_minor = float(np.sqrt(ev[0]/ev[-1])) if ev[-1] > 0 else 1.0
        span = float(np.sqrt(ev[-1]/max(1,len(arr)))*2.0)
        straight = float(np.clip((line_ratio-0.68)/0.26,0.0,1.0))
        thin = float(np.clip((0.26-pca_minor)/0.22,0.0,1.0))
        span_ok = float(np.clip((span-30.0)/55.0,0.0,1.0))
        score = float(np.clip((0.56*straight+0.44*thin)*(0.42+0.58*span_ok),0.0,1.0))
        result.update(score=score,linearity_ratio=line_ratio,pca_minor_ratio=pca_minor,span_px=span)
        return result

    left_names = ("neck_left","shoulder_left","sleeve_left_top","sleeve_left_bottom","armpit_left")
    right_names = ("neck_right","shoulder_right","sleeve_right_top","sleeve_right_bottom","armpit_right")
    ls, rs = chain_state(left_names), chain_state(right_names)
    l_abs, r_abs = float(ls['score']), float(rs['score'])
    left_chain = float(np.clip(0.55*l_abs+0.45*np.clip(l_abs-0.50*r_abs,0.0,1.0),0.0,1.0))
    right_chain = float(np.clip(0.55*r_abs+0.45*np.clip(r_abs-0.50*l_abs,0.0,1.0),0.0,1.0))

    intr = intrusion_state if isinstance(intrusion_state,dict) else _e117_pose_fold_intrusion_state(
        image_shape, garment, pose_state, args=args
    )
    intrusions = list(intr.get('intrusions',[]) or []) if isinstance(intr,dict) else []
    intrusion_score = 0.0
    left_upper_intr = right_upper_intr = 0.0
    left_all_intr = right_all_intr = 0.0
    if intrusions:
        prior = cv2.bitwise_or(prior, np.asarray(intr.get('corridor_mask',np.zeros_like(garment)),dtype=np.uint8))
        strengths=[float(q.get('strength',0.0)) for q in intrusions]
        intrusion_score=float(np.clip(0.30+0.16*len(intrusions)+0.45*(max(strengths) if strengths else 0.0),0.0,1.0))
        for q in intrusions:
            n=str(q.get('name','')); s=max(0.15,float(q.get('strength',0.0)))
            if '_left' in n:
                left_all_intr += s
                if any(k in n for k in ('shoulder','sleeve','armpit')): left_upper_intr += s
            elif '_right' in n:
                right_all_intr += s
                if any(k in n for k in ('shoulder','sleeve','armpit')): right_upper_intr += s

    # Torso 9-8-7-6 geometry.  Relative waist-angle asymmetry is deliberately
    # more sensitive than E119's absolute 128deg rule, because case1-1 showed a
    # clearly folded waist corner while its absolute maximum was only ~107deg.
    quad_names=("armpit_left","waist_left","waist_right","armpit_right")
    quad_pts=[getp(n) for n in quad_names]
    angles=[]; angle_map={}; quad_score=0.0; max_angle=0.0; max_vertex='NONE'
    ll_score=rr_score=0.0
    if all(p is not None for p in quad_pts):
        qa=np.asarray(quad_pts,dtype=np.float64)
        for k,name in enumerate(quad_names):
            prev=qa[(k-1)%4]-qa[k]; nxt=qa[(k+1)%4]-qa[k]
            den=float(np.linalg.norm(prev)*np.linalg.norm(nxt))
            ang=180.0 if den<=1e-6 else float(np.degrees(np.arccos(np.clip(float(np.dot(prev,nxt))/den,-1.0,1.0))))
            angles.append(ang); angle_map[name]=ang
        kmax=int(np.argmax(angles)); max_angle=float(angles[kmax]); max_vertex=str(quad_names[kmax])
        quad_score=float(np.clip((max_angle-108.0)/48.0,0.0,1.0))
        wl=float(angle_map.get('waist_left',0.0)); wr=float(angle_map.get('waist_right',0.0))
        ll_abs=float(np.clip((wl-88.0)/42.0,0.0,1.0)); rr_abs=float(np.clip((wr-88.0)/42.0,0.0,1.0))
        ll_asym=float(np.clip((wl-wr-8.0)/34.0,0.0,1.0)); rr_asym=float(np.clip((wr-wl-8.0)/34.0,0.0,1.0))
        ll_score=float(np.clip(0.52*ll_abs+0.48*ll_asym,0.0,1.0))
        rr_score=float(np.clip(0.52*rr_abs+0.48*rr_asym,0.0,1.0))

    # Upper-fold score: learned chain collapse + upper-side contour intrusion.
    ul_intr=float(np.clip(left_upper_intr/1.8,0.0,1.0)); ur_intr=float(np.clip(right_upper_intr/1.8,0.0,1.0))
    ul_score=float(np.clip(0.62*left_chain+0.38*ul_intr,0.0,1.0))
    ur_score=float(np.clip(0.62*right_chain+0.38*ur_intr,0.0,1.0))

    region_scores={
        'LOWER_LEFT':ll_score,'LOWER_RIGHT':rr_score,
        'UPPER_LEFT':ul_score,'UPPER_RIGHT':ur_score,
    }
    fold_region=max(region_scores,key=region_scores.get)
    region_conf=float(region_scores[fold_region])
    sorted_scores=sorted(region_scores.values(),reverse=True)
    region_gap=float(sorted_scores[0]-sorted_scores[1]) if len(sorted_scores)>1 else region_conf
    if region_conf < 0.18:
        fold_region='NONE'; region_conf=0.0

    # If a clearly abnormal waist vertex exists, let it own LOWER direction even
    # when a generic armpit intrusion would otherwise cancel LEFT/RIGHT votes.
    if max_vertex in ('waist_left','waist_right') and max_angle>=102.0:
        forced='LOWER_LEFT' if max_vertex=='waist_left' else 'LOWER_RIGHT'
        forced_score=ll_score if forced=='LOWER_LEFT' else rr_score
        if forced_score>=max(0.18, region_conf-0.10):
            fold_region=forced; region_conf=max(region_conf,forced_score)

    # Direction anchor for each anatomical quadrant.
    def mean_valid(names):
        pts=[getp(n) for n in names]; pts=[p for p in pts if p is not None]
        return np.mean(np.asarray(pts,dtype=np.float64),axis=0) if pts else None
    anchors={
        'LOWER_LEFT':getp('waist_left'), 'LOWER_RIGHT':getp('waist_right'),
        'UPPER_LEFT':mean_valid(('shoulder_left','sleeve_left_top','sleeve_left_bottom','armpit_left')),
        'UPPER_RIGHT':mean_valid(('shoulder_right','sleeve_right_top','sleeve_right_bottom','armpit_right')),
    }
    fold_anchor=anchors.get(fold_region) if fold_region!='NONE' else None
    fold_dir=None
    if fold_anchor is not None:
        v=np.asarray(fold_anchor,dtype=np.float64)-garment_center
        nv=float(np.linalg.norm(v))
        if nv>1e-6: fold_dir=v/nv

    signals=[]
    if left_chain>=0.28: signals.append(f'LEFT_CHAIN:{left_chain:.2f}')
    if right_chain>=0.28: signals.append(f'RIGHT_CHAIN:{right_chain:.2f}')
    if intrusions: signals.append('INTRUSION:'+','.join(str(q.get('name','?')) for q in intrusions[:5]))
    if max_angle>0: signals.append(f'TORSO:{max_vertex}:{max_angle:.1f}')
    if fold_region!='NONE': signals.append(f'REGION:{fold_region}:{region_conf:.2f}')

    # Draw topology evidence.  For lower folds, add a corridor from the abnormal
    # waist corner toward BODY centre so a weak crease inside that corridor can be
    # rescued.  For upper folds, the chain/intrusion evidence already covers the arm.
    chain_radius=max(10,int(round(float(getattr(args,'micro_unfold_pose_deform_chain_radius_px',24.0)) if args is not None else 24.0)))
    for label, st, score in (('LEFT',ls,left_chain),('RIGHT',rs,right_chain)):
        if score>=0.30 and len(st['points'])>=4:
            arr=np.rint(np.asarray([p for _,p in st['points']],dtype=np.float64)).astype(np.int32)
            cv2.polylines(prior,[arr],False,255,max(7,int(round(0.55*chain_radius))),cv2.LINE_AA)
            for pp in arr: cv2.circle(prior,tuple(pp.tolist()),chain_radius,255,-1,cv2.LINE_AA)
    if fold_anchor is not None and fold_dir is not None:
        a=tuple(np.rint(fold_anchor).astype(int).tolist())
        inward=np.asarray(fold_anchor,dtype=np.float64)-fold_dir*min(150.0,max(70.0,0.55*float(np.linalg.norm(np.asarray(fold_anchor)-garment_center))))
        b=tuple(np.rint(inward).astype(int).tolist())
        cv2.line(prior,a,b,255,max(20,int(round(1.15*chain_radius))),cv2.LINE_AA)
        cv2.circle(prior,a,max(22,chain_radius+4),255,-1,cv2.LINE_AA)

    prior=cv2.bitwise_and(prior,garment)
    if cv2.countNonZero(prior)>0:
        prior=cv2.dilate(prior,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(17,17)),iterations=1)
        prior=cv2.bitwise_and(prior,garment)

    side_hint='NONE'
    if fold_region.endswith('LEFT'): side_hint='LEFT'
    elif fold_region.endswith('RIGHT'): side_hint='RIGHT'
    else:
        lv=left_chain+0.25*left_all_intr; rv=right_chain+0.25*right_all_intr
        if lv>rv+0.12: side_hint='LEFT'
        elif rv>lv+0.12: side_hint='RIGHT'

    strongest=max(left_chain,right_chain,intrusion_score,quad_score,region_conf)
    conf=float(np.clip(0.60*strongest+0.28*region_conf+0.12*min(1.0,region_gap*2.0),0.0,1.0))
    if conf<0.20 or cv2.countNonZero(prior)<8:
        return {**empty,
            'left_chain_score':left_chain,'right_chain_score':right_chain,
            'torso_quad_score':quad_score,'torso_max_angle_deg':max_angle,'torso_max_angle_vertex':max_vertex,
            'torso_angles_deg':angle_map,'lower_left_score':ll_score,'lower_right_score':rr_score,
            'upper_left_score':ul_score,'upper_right_score':ur_score,
            'intrusion_score':intrusion_score,'intrusion_count':len(intrusions),'signals':signals,
            'side_hint':side_hint,'fold_region_hint':fold_region,'fold_region_confidence':region_conf,
            'fold_anchor_px':None if fold_anchor is None else [float(fold_anchor[0]),float(fold_anchor[1])],
            'fold_direction_px':None if fold_dir is None else [float(fold_dir[0]),float(fold_dir[1])],
        }

    mode = 'TORSO_QUAD' if fold_region.startswith('LOWER_') and region_conf>=0.18 else (
        'LEFT_CHAIN' if fold_region=='UPPER_LEFT' and left_chain>=ul_intr else
        'RIGHT_CHAIN' if fold_region=='UPPER_RIGHT' and right_chain>=ur_intr else
        'CONTOUR_INTRUSION'
    )
    return {
        'available':True,'confidence':conf,'mode':mode,'side_hint':side_hint,
        'fold_region_hint':fold_region,'fold_region_confidence':region_conf,
        'fold_anchor_px':None if fold_anchor is None else [float(fold_anchor[0]),float(fold_anchor[1])],
        'fold_direction_px':None if fold_dir is None else [float(fold_dir[0]),float(fold_dir[1])],
        'prior_mask':prior.astype(np.uint8),'left_chain_score':left_chain,'right_chain_score':right_chain,
        'left_chain_linearity':float(ls.get('linearity_ratio',0.0)),'right_chain_linearity':float(rs.get('linearity_ratio',0.0)),
        'left_chain_pca_minor_ratio':float(ls.get('pca_minor_ratio',1.0)),'right_chain_pca_minor_ratio':float(rs.get('pca_minor_ratio',1.0)),
        'torso_quad_score':quad_score,'torso_max_angle_deg':max_angle,'torso_max_angle_vertex':max_vertex,
        'torso_angles_deg':angle_map,'lower_left_score':ll_score,'lower_right_score':rr_score,
        'upper_left_score':ul_score,'upper_right_score':ur_score,
        'intrusion_score':intrusion_score,'intrusion_count':int(len(intrusions)),'signals':signals,
        'left_vote':float(left_chain+0.25*left_all_intr),'right_vote':float(right_chain+0.25*right_all_intr),
    }

def _e118_pose_region_split_counts(garment_mask_u8, region_a_u8, region_b_u8, pose_state, args=None):
    """Count learned T-shirt pose landmarks on the two cloth regions cut by one crease.

    E118's role rule is deliberately simple and geometric:
      * Fold Crease + cloth contour defines two candidate cloth regions.
      * The region with FEWER valid anatomical landmarks is the folded/ACTIVE side.
      * The region with MORE landmarks is the BODY/HOLD side.

    Points are assigned with a small dilation margin because several learned landmarks
    naturally sit directly on the cloth contour.  If a point falls in both dilated
    regions, it is assigned to whichever original region is spatially closer.
    """
    garment=((np.asarray(garment_mask_u8,dtype=np.uint8)>0).astype(np.uint8)*255)
    a=((np.asarray(region_a_u8,dtype=np.uint8)>0).astype(np.uint8)*255)
    b=((np.asarray(region_b_u8,dtype=np.uint8)>0).astype(np.uint8)*255)
    empty={
        "available":False,"count_a":0,"count_b":0,"weight_a":0.0,"weight_b":0.0,
        "assigned":0,"gap":0,"confidence":0.0,"preferred_active_region":"NONE",
        "authoritative":False,"assignments":[]
    }
    if cv2.countNonZero(garment)<500 or cv2.countNonZero(a)<50 or cv2.countNonZero(b)<50 or not isinstance(pose_state,dict):
        return empty
    named=dict(pose_state.get("named_points",{}) or {})
    conf_min=float(getattr(args,"micro_unfold_pose_split_kpt_conf",0.20)) if args is not None else 0.20
    margin=max(1,int(round(float(getattr(args,"micro_unfold_pose_split_assign_margin_px",8.0)) if args is not None else 8.0)))
    min_assigned=max(2,int(getattr(args,"micro_unfold_pose_split_min_assigned",6)) if args is not None else 6)
    min_gap=max(1,int(getattr(args,"micro_unfold_pose_split_min_count_gap",2)) if args is not None else 2)
    k=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2*margin+1,2*margin+1))
    ad=cv2.dilate(a,k,iterations=1)>0
    bd=cv2.dilate(b,k,iterations=1)>0
    # Distance to each ORIGINAL region, used only when dilation makes membership ambiguous.
    da=cv2.distanceTransform(((a==0).astype(np.uint8)*255),cv2.DIST_L2,5)
    db=cv2.distanceTransform(((b==0).astype(np.uint8)*255),cv2.DIST_L2,5)
    h,w=garment.shape[:2]
    ca=cb=0
    wa=wb=0.0
    assignments=[]
    for idx,name in enumerate(_E102_TSHIRT_POSE_KPT_NAMES):
        q=dict(named.get(name,{}) or {})
        try:
            x=float(q.get("x")); y=float(q.get("y")); c=float(q.get("conf",0.0))
        except Exception:
            continue
        if not (np.isfinite(x) and np.isfinite(y) and np.isfinite(c)) or c<conf_min:
            continue
        xi=int(np.clip(round(x),0,w-1)); yi=int(np.clip(round(y),0,h-1))
        # Permit a few pixels outside segmentation because edge landmarks often land
        # just outside the YOLO mask. Region dilation still decides the ownership.
        ina=bool(ad[yi,xi]); inb=bool(bd[yi,xi])
        owner=None
        if ina and not inb:
            owner="A"
        elif inb and not ina:
            owner="B"
        elif ina and inb:
            owner="A" if float(da[yi,xi])<=float(db[yi,xi]) else "B"
        else:
            # If both miss, accept only a very near region to avoid inventing ownership.
            va=float(da[yi,xi]); vb=float(db[yi,xi])
            if min(va,vb)<=float(margin+3):
                owner="A" if va<=vb else "B"
        if owner=="A":
            ca+=1; wa+=max(0.0,min(1.0,c))
        elif owner=="B":
            cb+=1; wb+=max(0.0,min(1.0,c))
        else:
            continue
        assignments.append({"index":int(idx),"name":str(name),"owner":str(owner),"conf":float(c),"x":float(x),"y":float(y)})
    assigned=int(ca+cb)
    gap=int(abs(ca-cb))
    if assigned<=0:
        return empty
    conf=float(np.clip(gap/max(1.0,float(assigned)),0.0,1.0))
    pref="A" if ca<cb else ("B" if cb<ca else "NONE")
    authoritative=bool(assigned>=min_assigned and gap>=min_gap and pref!="NONE")
    return {
        "available":True,"count_a":int(ca),"count_b":int(cb),"weight_a":float(wa),"weight_b":float(wb),
        "assigned":int(assigned),"gap":int(gap),"confidence":float(conf),
        "preferred_active_region":str(pref),"authoritative":bool(authoritative),
        "assignments":assignments,
    }



_E149_CV_KERNEL_CACHE = {}

def _e149_cv_kernel(shape_name, kx, ky=None):
    """E149 OPT5: cache immutable OpenCV morphology kernels."""
    ky=int(kx if ky is None else ky)
    key=(str(shape_name),int(kx),int(ky))
    ker=_E149_CV_KERNEL_CACHE.get(key)
    if ker is None:
        if str(shape_name)=='ellipse':
            ker=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(int(kx),int(ky)))
        elif str(shape_name)=='cross':
            ker=cv2.getStructuringElement(cv2.MORPH_CROSS,(int(kx),int(ky)))
        else:
            ker=np.ones((int(ky),int(kx)),np.uint8)
        _E149_CV_KERNEL_CACHE[key]=ker
    return ker

def _e149_garment_roi_bounds(mask_u8, pad=64):
    gm=np.asarray(mask_u8,dtype=np.uint8)>0
    yy,xx=np.where(gm); h,w=gm.shape[:2]
    if xx.size==0:
        return 0,0,w,h
    p=max(0,int(pad))
    return max(0,int(xx.min())-p), max(0,int(yy.min())-p), min(w,int(xx.max())+1+p), min(h,int(yy.max())+1+p)

def _e149_paste_map(full_shape, roi_arr, bounds):
    x0,y0,x1,y1=bounds
    arr=np.asarray(roi_arr)
    out=np.zeros(full_shape,dtype=arr.dtype)
    out[y0:y1,x0:x1]=arr
    return out

def _e94_detect_fold_crease(frame_bgr, garment_mask_u8, relief_mm, prominence_mm=None, args=None, pose_state=None, pose_deform_state=None, e156_precomputed=None):
    """E118 dual Fold-Crease detector.

    Two maps are deliberately separated:
      * SENSITIVE: E114-like high recall. It is shown in cyan and may contain fine wrinkles.
      * STRUCTURAL: E113-like conservative thresholds. AUTO planning uses this first.

    If no structural component survives, the sensitive map may be used only as a
    *rescue* source; E116 downstream then requires strong direct UniDepth role
    evidence before X can become READY. E118 hard-excludes the pose-derived neckline
    only; armhole/sleeve sewing lines remain eligible Fold Crease evidence.
    """
    frame=np.asarray(frame_bgr,dtype=np.uint8)
    garment=((np.asarray(garment_mask_u8,dtype=np.uint8)>0).astype(np.uint8)*255)
    if frame.ndim!=3 or frame.shape[:2]!=garment.shape[:2]:
        return {"available":False,"reason":"CREASE_SHAPE_MISMATCH"}
    garment_area=int(cv2.countNonZero(garment))
    if garment_area<500:
        return {"available":False,"reason":"CREASE_GARMENT_TOO_SMALL"}

    pose_sem=_e101_pose_semantic_masks(frame.shape,garment,pose_state,args=args)
    seam_excl=np.asarray(pose_sem.get('seam_mask',np.zeros_like(garment)),dtype=np.uint8)>0
    body_core=np.asarray(pose_sem.get('body_core_mask',np.zeros_like(garment)),dtype=np.uint8)

    # E119: a fold can terminate at / run very near a sleeve or outer contour.
    # E118 globally discarded that boundary band, which fragmented many upper-arm
    # folds.  Keep a smaller normal exclusion and relax it further only where the
    # 12-point pose topology itself shows a fold-like deformation.
    if not isinstance(pose_deform_state,dict):
        pose_deform_state=_e120_pose_deformation_state(frame.shape,garment,pose_state,args=args)
    deform_mask=np.asarray(pose_deform_state.get('prior_mask',np.zeros_like(garment)),dtype=np.uint8)
    deform_conf=float(pose_deform_state.get('confidence',0.0) or 0.0)
    deform_zone=cv2.dilate(deform_mask,_e149_cv_kernel('ellipse',25),iterations=1)>0 if cv2.countNonZero(deform_mask)>=8 else np.zeros_like(garment,dtype=bool)

    _e156_pre=e156_precomputed if isinstance(e156_precomputed,dict) else {}
    _e156_inside=np.asarray(_e156_pre.get('inside_dist',np.empty((0,0),dtype=np.float32)),dtype=np.float32)
    inside_dist=_e156_inside if _e156_inside.shape==garment.shape else cv2.distanceTransform(garment,cv2.DIST_L2,5).astype(np.float32)
    boundary_exclude=max(3.0,float(getattr(args,'micro_unfold_crease_boundary_exclude_px',6.0)) if args is not None else 6.0)
    base_domain=(garment>0)&(~seam_excl)
    domain=base_domain&(inside_dist>=boundary_exclude)
    if deform_conf>=0.28 and np.any(deform_zone):
        domain |= base_domain&deform_zone&(inside_dist>=3.0)
    if int(np.count_nonzero(domain))<300:
        domain=base_domain&(inside_dist>=max(3.0,0.55*boundary_exclude))
    if int(np.count_nonzero(domain))<150:
        return {"available":False,"reason":"CREASE_INTERIOR_DOMAIN_TOO_SMALL","pose_semantic":pose_sem}

    # E149 OPT4: heavy RGB filtering only on garment bbox + 64px context.
    # The full-size domain/coordinates remain authoritative; only expensive image maps
    # are computed on the ROI and pasted back before the unchanged threshold logic.
    _e156_pre_ok=bool(
        isinstance(_e156_pre,dict)
        and tuple(_e156_pre.get('shape',()))==tuple(garment.shape)
        and isinstance(_e156_pre.get('roi',None),(list,tuple))
        and len(_e156_pre.get('roi',()))==4
    )
    if _e156_pre_ok:
        _e149_roi=tuple(int(v) for v in _e156_pre['roi'])
        _x0,_y0,_x1,_y1=_e149_roi
        _domain_r=domain[_y0:_y1,_x0:_x1]
        _Lr=np.asarray(_e156_pre['Lr'],dtype=np.float32)
        _dark_r=np.asarray(_e156_pre['dark_r'],dtype=np.float32)
        _bright_r=np.asarray(_e156_pre['bright_r'],dtype=np.float32)
        _broad_dark_r=np.asarray(_e156_pre['broad_dark_r'],dtype=np.float32)
        _broad_step_r=np.asarray(_e156_pre['broad_step_r'],dtype=np.float32)
        _rgb_grad_r=np.asarray(_e156_pre['rgb_grad_r'],dtype=np.float32)
        _coarse_grad_r=np.asarray(_e156_pre['coarse_grad_r'],dtype=np.float32)
        _depth_grad_r=np.asarray(_e156_pre['depth_grad_r'],dtype=np.float32)
        _zero_depth=bool(_e156_pre.get('zero_depth',True))
        _roi_t0=time.perf_counter()
        _e156_pre_ms=float(_e156_pre.get('heavy_ms',0.0))
    else:
        _e149_roi=_e149_garment_roi_bounds(garment,pad=64)
        _x0,_y0,_x1,_y1=_e149_roi
        _fr=frame[_y0:_y1,_x0:_x1]
        _domain_r=domain[_y0:_y1,_x0:_x1]
        _roi_t0=time.perf_counter()
        _Lr=cv2.cvtColor(_fr,cv2.COLOR_BGR2LAB)[:,:,0].astype(np.float32)
        _fine_r=cv2.GaussianBlur(_Lr,(5,5),0)
        _dark_r=np.zeros_like(_Lr,dtype=np.float32); _bright_r=np.zeros_like(_Lr,dtype=np.float32)
        for k in (11,21,35,55):
            ker=_e149_cv_kernel('ellipse',k)
            _dark_r=np.maximum(_dark_r,cv2.morphologyEx(_fine_r,cv2.MORPH_CLOSE,ker)-_fine_r)
            _bright_r=np.maximum(_bright_r,_fine_r-cv2.morphologyEx(_fine_r,cv2.MORPH_OPEN,ker))

        _g9=cv2.GaussianBlur(_Lr,(9,9),0); _g31=cv2.GaussianBlur(_Lr,(31,31),0); _g51=cv2.GaussianBlur(_Lr,(51,51),0)
        _broad_dark_r=np.maximum(np.maximum(_g31-_g9,_g51-_g9),0.0).astype(np.float32)
        _broad_step_r=np.maximum(np.abs(_g31-_g9),np.abs(_g51-_g9)).astype(np.float32)
        _gx=cv2.Scharr(_fine_r,cv2.CV_32F,1,0); _gy=cv2.Scharr(_fine_r,cv2.CV_32F,0,1)
        _rgb_grad_r=cv2.magnitude(_gx,_gy)
        _coarse_r=cv2.GaussianBlur(_Lr,(0,0),3.0)
        _cgx=cv2.Scharr(_coarse_r,cv2.CV_32F,1,0); _cgy=cv2.Scharr(_coarse_r,cv2.CV_32F,0,1)
        _coarse_grad_r=cv2.magnitude(_cgx,_cgy)

        relief=np.asarray(relief_mm,dtype=np.float32)
        if relief.shape!=garment.shape:
            relief=cv2.resize(relief,(garment.shape[1],garment.shape[0]),interpolation=cv2.INTER_LINEAR)
        relief_clean=np.where(np.isfinite(relief),relief,0.0).astype(np.float32)
        # E149 OPT3: the E137+ compatibility depth is zero. Gaussian/Scharr/normalization
        # would produce exactly zero, so do not launch those OpenCV kernels.
        _zero_depth=not bool(np.any(relief_clean!=0.0))
        if _zero_depth:
            _depth_grad_r=np.zeros_like(_Lr,dtype=np.float32)
        else:
            _relief_r=relief_clean[_y0:_y1,_x0:_x1]
            _relief_s=cv2.GaussianBlur(_relief_r,(7,7),0)
            _dgx=cv2.Scharr(_relief_s,cv2.CV_32F,1,0); _dgy=cv2.Scharr(_relief_s,cv2.CV_32F,0,1)
            _depth_grad_r=cv2.magnitude(_dgx,_dgy)

    # Percentile normalization is evaluated on exactly the same garment domain.
    _dark_n_r=_e94_robust_unit_map(_dark_r,_domain_r,38.0,98.8)
    _bright_n_r=_e94_robust_unit_map(_bright_r,_domain_r,55.0,99.0)
    _broad_n_r=_e94_robust_unit_map(_broad_dark_r,_domain_r,38.0,98.8)
    _broad_step_n_r=_e94_robust_unit_map(_broad_step_r,_domain_r,42.0,98.8)
    _grad_n_r=_e94_robust_unit_map(_rgb_grad_r,_domain_r,48.0,98.5)
    _coarse_grad_n_r=_e94_robust_unit_map(_coarse_grad_r,_domain_r,45.0,98.5)
    _depth_grad_n_r=np.zeros_like(_depth_grad_r,dtype=np.float32) if _zero_depth else _e94_robust_unit_map(_depth_grad_r,_domain_r,55.0,98.5)

    L=_e149_paste_map(garment.shape,_Lr,_e149_roi)
    dark_n=_e149_paste_map(garment.shape,_dark_n_r,_e149_roi)
    bright_n=_e149_paste_map(garment.shape,_bright_n_r,_e149_roi)
    broad_n=_e149_paste_map(garment.shape,_broad_n_r,_e149_roi)
    broad_step_n=_e149_paste_map(garment.shape,_broad_step_n_r,_e149_roi)
    grad_n=_e149_paste_map(garment.shape,_grad_n_r,_e149_roi)
    coarse_grad_n=_e149_paste_map(garment.shape,_coarse_grad_n_r,_e149_roi)
    depth_grad_n=_e149_paste_map(garment.shape,_depth_grad_n_r,_e149_roi)
    combined=(0.18*dark_n+0.18*broad_n+0.08*bright_n+0.20*broad_step_n+
              0.14*grad_n+0.14*coarse_grad_n+0.08*depth_grad_n).astype(np.float32)
    _roi_finalize_sec=time.perf_counter()-_roi_t0
    _roi_sec=(float(_e156_pre_ms)/1000.0) if _e156_pre_ok else _roi_finalize_sec
    _roi_frac=((_x1-_x0)*(_y1-_y0))/float(max(1,garment.shape[0]*garment.shape[1]))
    _e156_src='PRECOMPUTED' if _e156_pre_ok else 'INLINE'
    print(f"[E149-RGB-ROI] xyxy=({_x0},{_y0},{_x1},{_y1}) size={_x1-_x0}x{_y1-_y0} pixels={100.0*_roi_frac:.1f}% pad=64 zeroDepth={_zero_depth} heavyMaps={_roi_sec:.3f}s source={_e156_src} finalize={_roi_finalize_sec:.3f}s")
    vals=combined[domain]
    if vals.size<32:
        return {"available":False,"reason":"CREASE_NO_SCORE_DOMAIN","pose_semantic":pose_sem}

    dark_gate=max(0.14,float(np.percentile(dark_n[domain],56.0)))
    broad_gate=max(0.16,float(np.percentile(broad_n[domain],54.0)))
    step_gate=max(0.14,float(np.percentile(broad_step_n[domain],52.0)))
    coarse_gate=max(0.15,float(np.percentile(coarse_grad_n[domain],56.0)))

    def _variant(name, gate_pct, hard_floor, low_floor, min_span, max_width, max_components):
        gate_pct=float(np.clip(gate_pct,68.0,94.0))
        gate=max(float(hard_floor),float(np.percentile(vals,gate_pct)))
        cand=domain&(combined>=gate)&((dark_n>=dark_gate)|(broad_n>=broad_gate)|(broad_step_n>=step_gate)|(coarse_grad_n>=coarse_gate)|(grad_n>=0.66))
        low_gate=max(float(low_floor),float(np.percentile(vals,max(50.0,gate_pct-24.0))))
        low_support=domain&(combined>=low_gate)&((dark_n>=0.72*dark_gate)|(broad_n>=0.68*broad_gate)|(broad_step_n>=0.66*step_gate)|(coarse_grad_n>=0.70*coarse_gate)|(grad_n>=0.58))
        # E119 pose-deformation rescue: nearby RGB evidence may start slightly
        # below the global percentile, but it must still carry actual dark/broad/
        # gradient crease evidence.  Pose alone never paints a cyan line.
        if deform_conf>=0.38 and np.any(deform_zone):
            prior_gate=max(float(low_floor),0.78*float(gate))
            prior_seed=domain&deform_zone&(combined>=prior_gate)&(
                (dark_n>=0.58*dark_gate)|(broad_n>=0.56*broad_gate)|
                (broad_step_n>=0.54*step_gate)|(coarse_grad_n>=0.56*coarse_gate)|(grad_n>=0.54)
            )
            cand |= prior_seed
            low_support |= domain&deform_zone&(combined>=max(float(low_floor),0.86*float(low_gate)))
        grown=cand.copy(); k3=_e149_cv_kernel('ones',3)
        for _ in range(18):
            nxt=(cv2.dilate(grown.astype(np.uint8),k3,iterations=1)>0)&low_support
            merged=grown|nxt
            if int(np.count_nonzero(merged))==int(np.count_nonzero(grown)):
                break
            grown=merged
        raw=(grown.astype(np.uint8)*255)
        # E122: do NOT weld nearby crease/wrinkle ridges with large 9x9 / 3x13 /
        # 13x3 closings.  Keep only one tiny gap-fill; branch/junction logic below
        # is responsible for continuity. This preserves a true structural fold as
        # an independent polyline even when a nearby wrinkle intersects it visually.
        raw=cv2.morphologyEx(raw,cv2.MORPH_CLOSE,np.ones((3,3),np.uint8),iterations=1)
        raw[~domain]=0

        ncc,labs,stats,_=cv2.connectedComponentsWithStats((raw>0).astype(np.uint8),8)
        comps=[]
        for lid in range(1,ncc):
            area=int(stats[lid,cv2.CC_STAT_AREA])
            if area<16 or area>max(1800,int(0.12*garment_area)):
                continue
            comp=labs==lid; yy,xx=np.where(comp)
            if xx.size<2:
                continue
            pts=np.column_stack([xx,yy]).astype(np.float64)
            ctr=pts.mean(axis=0); cov=np.cov((pts-ctr).T) if pts.shape[0]>=3 else np.eye(2)
            ev,_=np.linalg.eigh(cov); ev=np.sort(np.maximum(ev,1e-6))[::-1]
            elong=float(np.sqrt(ev[0]/ev[1])) if ev.size>=2 else 1.0
            span=float(np.hypot(float(xx.max()-xx.min()),float(yy.max()-yy.min())))
            if span<float(min_span):
                continue
            cu8=(comp.astype(np.uint8)*255); cs,_=cv2.findContours(cu8,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
            per=float(sum(cv2.arcLength(c,True) for c in cs)) if cs else 0.0
            length_est=max(span,0.5*per); width_est=float(area)/max(1.0,length_est)
            if width_est>float(max_width):
                continue
            mean_comb=float(np.mean(combined[comp])); mean_broad=float(np.mean(broad_n[comp])); mean_dark=float(np.mean(dark_n[comp])); mean_step=float(np.mean(broad_step_n[comp])); mean_depth=float(np.mean(depth_grad_n[comp]))
            prior_support=float(np.mean(deform_zone[comp])) if np.any(deform_zone) else 0.0
            line_score=(0.29*mean_comb+0.15*mean_broad+0.07*mean_dark+0.17*mean_step+0.03*mean_depth+
                        0.12*min(1.0,length_est/180.0)+0.05*min(1.0,span/160.0)+0.03*min(1.0,elong/6.0)+
                        0.14*prior_support*deform_conf)
            comps.append({'id':int(lid),'area_px':area,'span_px':span,'length_est_px':length_est,'width_est_px':width_est,
                          'elongation':elong,'mean_score':mean_comb,'mean_broad':mean_broad,'mean_dark':mean_dark,'mean_step':mean_step,
                          'mean_depth_grad':mean_depth,'pose_deform_support':float(prior_support),
                          'component_score':float(line_score),'variant':str(name)})
        comps.sort(key=lambda x:float(x['component_score']),reverse=True)
        kept=comps[:int(max_components)]
        band=np.zeros_like(garment)
        for item in kept:
            band[labs==int(item['id'])]=255
        skel=_e94_skeletonize(band) if cv2.countNonZero(band)>=12 else np.zeros_like(garment)
        if cv2.countNonZero(skel)<8 and cv2.countNonZero(band)>=12:
            skel=band.copy()
        return {
            'name':str(name),'available':bool(cv2.countNonZero(skel)>=8),'gate':float(gate),'gate_pct':float(gate_pct),'low_gate':float(low_gate),
            'band_mask':band.astype(np.uint8),'skeleton_mask':skel.astype(np.uint8),'components':kept,
            'component_count':int(len(kept)),'band_area_px':int(cv2.countNonZero(band)),'skeleton_area_px':int(cv2.countNonZero(skel)),
            'min_span_px':float(min_span),'max_width_px':float(max_width),'max_components':int(max_components),
        }

    sensitive=_variant(
        'SENSITIVE',
        float(getattr(args,'micro_unfold_crease_percentile',70.0)) if args is not None else 70.0,
        float(getattr(args,'micro_unfold_crease_sensitive_hard_floor',0.18)) if args is not None else 0.18,
        float(getattr(args,'micro_unfold_crease_sensitive_low_floor',0.11)) if args is not None else 0.11,
        max(14.0,float(getattr(args,'micro_unfold_crease_min_span_px',18.0)) if args is not None else 18.0),
        max(8.0,float(getattr(args,'micro_unfold_crease_max_width_px',42.0)) if args is not None else 42.0),
        int(np.clip(int(getattr(args,'micro_unfold_crease_max_components',10)) if args is not None else 10,1,12)),
    )
    structural=_variant(
        'STRUCTURAL',
        float(getattr(args,'micro_unfold_structural_crease_percentile',78.0)) if args is not None else 78.0,
        float(getattr(args,'micro_unfold_structural_crease_hard_floor',0.25)) if args is not None else 0.25,
        float(getattr(args,'micro_unfold_structural_crease_low_floor',0.16)) if args is not None else 0.16,
        max(18.0,float(getattr(args,'micro_unfold_structural_crease_min_span_px',24.0)) if args is not None else 24.0),
        max(8.0,float(getattr(args,'micro_unfold_structural_crease_max_width_px',32.0)) if args is not None else 32.0),
        int(np.clip(int(getattr(args,'micro_unfold_structural_crease_max_components',10)) if args is not None else 10,1,12)),
    )

    primary=structural if structural['available'] else sensitive
    if not primary['available']:
        return {
            'available':False,'reason':'NO_RELIABLE_INTERNAL_CREASE','score_map':combined,
            'band_mask':np.zeros_like(garment),'skeleton_mask':np.zeros_like(garment),'components':[],
            'sensitive_band_mask':sensitive['band_mask'],'sensitive_skeleton_mask':sensitive['skeleton_mask'],'sensitive_components':sensitive['components'],
            'structural_band_mask':structural['band_mask'],'structural_skeleton_mask':structural['skeleton_mask'],'structural_components':structural['components'],
            'threshold':float(sensitive['gate']),'threshold_percentile':float(sensitive['gate_pct']),'domain_area_px':int(np.count_nonzero(domain)),
            'pose_semantic':pose_sem,'pose_seam_exclusion_mask':np.asarray(pose_sem.get('seam_mask',np.zeros_like(garment)),dtype=np.uint8),'pose_body_core_mask':body_core,
            'pose_deform_mask':deform_mask.astype(np.uint8),
            'pose_deform_info':{k:v for k,v in pose_deform_state.items() if k!='prior_mask'},
            '_e149_rgb_l_map':L.astype(np.float32),'_e149_inside_map':inside_dist.astype(np.float32),'_e149_roi_bounds':list(_e149_roi),'_e149_roi_sec':float(_roi_sec),
        }

    selection_source='STRUCTURAL' if structural['available'] else 'SENSITIVE_RESCUE'
    return {
        'available':True,'reason':f'E116_{selection_source}_RGB_POSE_FILTERED_CREASE','selection_source':selection_source,
        'threshold':float(primary['gate']),'threshold_percentile':float(primary['gate_pct']),'domain_area_px':int(np.count_nonzero(domain)),
        'band_area_px':int(primary['band_area_px']),'skeleton_area_px':int(primary['skeleton_area_px']),
        'component_count':int(primary['component_count']),'components':primary['components'],'boundary_exclude_px':boundary_exclude,
        'score_map':combined.astype(np.float32),'band_mask':primary['band_mask'],'skeleton_mask':primary['skeleton_mask'],
        'sensitive_threshold':float(sensitive['gate']),'sensitive_low_gate':float(sensitive['low_gate']),
        'sensitive_band_mask':sensitive['band_mask'],'sensitive_skeleton_mask':sensitive['skeleton_mask'],'sensitive_components':sensitive['components'],
        'sensitive_component_count':int(sensitive['component_count']),'sensitive_band_area_px':int(sensitive['band_area_px']),'sensitive_skeleton_area_px':int(sensitive['skeleton_area_px']),
        'structural_threshold':float(structural['gate']),'structural_low_gate':float(structural['low_gate']),
        'structural_band_mask':structural['band_mask'],'structural_skeleton_mask':structural['skeleton_mask'],'structural_components':structural['components'],
        'structural_component_count':int(structural['component_count']),'structural_band_area_px':int(structural['band_area_px']),'structural_skeleton_area_px':int(structural['skeleton_area_px']),
        'pose_semantic':{k:v for k,v in pose_sem.items() if k not in ('seam_mask','body_core_mask')},
        'pose_seam_exclusion_mask':np.asarray(pose_sem.get('seam_mask',np.zeros_like(garment)),dtype=np.uint8),
        'pose_body_core_mask':body_core.astype(np.uint8),
        'pose_deform_mask':deform_mask.astype(np.uint8),
        'pose_deform_info':{k:v for k,v in pose_deform_state.items() if k!='prior_mask'},
        '_e149_rgb_l_map':L.astype(np.float32),'_e149_inside_map':inside_dist.astype(np.float32),'_e149_roi_bounds':list(_e149_roi),'_e149_roi_sec':float(_roi_sec),
    }



def _e122_split_crease_branches(skeleton_u8, min_pixels=10):
    """E123 conservative skeleton branch splitter.

    E122 removed every >=3-neighbour pixel and then dilated those junctions by
    3x3.  On a real one-pixel OpenCV skeleton, diagonal/staircase geometry often
    has three 8-neighbours even without a true T/X junction, so most of the
    1700~2500px crease could disappear.  E123:
      * works per original connected component,
      * treats degree>=4 as a true junction candidate,
      * removes only the junction pixel (no halo dilation), and
      * falls back to the original component whenever splitting loses too much
        of that component or produces no useful child run.
    This guarantees that branch decomposition cannot turn a strong existing
    structural crease component into an empty candidate pool.
    """
    sk=((np.asarray(skeleton_u8,dtype=np.uint8)>0).astype(np.uint8))
    min_pixels=max(6,int(min_pixels))
    if int(np.count_nonzero(sk))<min_pixels:
        return []

    n0, labels0, stats0, _ = cv2.connectedComponentsWithStats(sk,8)
    branches=[]
    for lab0 in range(1,n0):
        comp=(labels0==lab0).astype(np.uint8)
        comp_area=int(stats0[lab0,cv2.CC_STAT_AREA])
        if comp_area<min_pixels:
            continue
        yy0,xx0=np.where(comp>0)
        if xx0.size<min_pixels:
            continue
        comp_span=float(np.hypot(float(xx0.max()-xx0.min()),float(yy0.max()-yy0.min())))
        if comp_span<12.0:
            continue

        degree=(cv2.filter2D(comp,cv2.CV_16S,np.ones((3,3),np.uint8),borderType=cv2.BORDER_CONSTANT)
                -comp.astype(np.int16))
        junction=((comp>0)&(degree>=4))
        # If there is no convincing graph junction, preserve the full component.
        if int(np.count_nonzero(junction))==0:
            branches.append((comp*255).astype(np.uint8))
            continue

        cut=comp.copy()
        cut[junction]=0
        n1, labels1, stats1, _ = cv2.connectedComponentsWithStats(cut,8)
        children=[]
        kept_pixels=0
        for lab1 in range(1,n1):
            area=int(stats1[lab1,cv2.CC_STAT_AREA])
            if area<min_pixels:
                continue
            child=(labels1==lab1).astype(np.uint8)
            yy,xx=np.where(child>0)
            if xx.size<min_pixels:
                continue
            span=float(np.hypot(float(xx.max()-xx.min()),float(yy.max()-yy.min())))
            if span<12.0:
                continue
            children.append((child*255).astype(np.uint8))
            kept_pixels+=area

        coverage=float(kept_pixels)/float(max(1,comp_area))
        # Split only when the decomposition actually preserves the component.
        # Otherwise the original component is much safer than E122's empty pool.
        if children and coverage>=0.55:
            branches.extend(children)
        else:
            branches.append((comp*255).astype(np.uint8))
    return branches


def _e122_patch_median(arr, valid_mask, cx, cy, radius=3):
    h, w = arr.shape[:2]
    x0=max(0,int(round(cx))-int(radius)); x1=min(w,int(round(cx))+int(radius)+1)
    y0=max(0,int(round(cy))-int(radius)); y1=min(h,int(round(cy))+int(radius)+1)
    if x1<=x0 or y1<=y0:
        return None
    yy,xx=np.ogrid[y0:y1,x0:x1]
    disk=((xx-float(cx))**2+(yy-float(cy))**2)<=float(radius*radius)
    vm=np.asarray(valid_mask[y0:y1,x0:x1],dtype=bool)&disk
    vals=np.asarray(arr[y0:y1,x0:x1])[vm]
    vals=vals[np.isfinite(vals)]
    if vals.size<4:
        return None
    return float(np.median(vals))


def _e122_score_fold_branch(branch_u8, frame_bgr, relief_mm, garment_u8, raw_score_map,
                            structural_u8, pose_deform_u8, pose_deform_conf,
                            seam_u8):
    """Classify one high-recall crease branch as wrinkle vs fold boundary.

    The key evidence is *bilateral*: a fold boundary should separate two surfaces
    whose relative depth/orientation/appearance relation is reasonably consistent
    along the branch.  A local wrinkle often produces a ridge/valley but not a
    stable side-A vs side-B surface relationship over the full line.
    """
    branch=np.asarray(branch_u8,dtype=np.uint8)>0
    garment=np.asarray(garment_u8,dtype=np.uint8)>0
    relief=np.asarray(relief_mm,dtype=np.float32)
    if relief.shape!=branch.shape:
        relief=cv2.resize(relief,(branch.shape[1],branch.shape[0]),interpolation=cv2.INTER_LINEAR)
    frame=np.asarray(frame_bgr,dtype=np.uint8)
    L=cv2.cvtColor(frame,cv2.COLOR_BGR2LAB)[:,:,0].astype(np.float32)
    relief_clean=np.where(np.isfinite(relief),relief,0.0).astype(np.float32)
    relief_s=cv2.GaussianBlur(relief_clean,(7,7),0)
    dzdx=cv2.Sobel(relief_s,cv2.CV_32F,1,0,ksize=3)
    dzdy=cv2.Sobel(relief_s,cv2.CV_32F,0,1,ksize=3)
    inside=cv2.distanceTransform((garment.astype(np.uint8)*255),cv2.DIST_L2,5).astype(np.float32)
    yy,xx=np.where(branch)
    pts=np.column_stack([xx,yy]).astype(np.float64)
    if len(pts)<8:
        return {'fold_score':0.0,'valid_samples':0,'reason':'BRANCH_TOO_SHORT'}
    ctr=pts.mean(axis=0)
    X=pts-ctr[None,:]
    cov=X.T@X
    vals,vecs=np.linalg.eigh(cov)
    tangent_global=vecs[:,int(np.argmax(vals))]
    tangent_global/=max(1e-9,float(np.linalg.norm(tangent_global)))
    proj=X@tangent_global
    span=float(np.max(proj)-np.min(proj))
    linearity=float(np.clip(1.0-float(vals[0])/max(1e-6,float(vals[-1])),0.0,1.0))
    order=np.argsort(proj)
    ordered=pts[order]
    sample_count=int(np.clip(round(span/18.0)+3,7,17))
    sample_indices=np.unique(np.rint(np.linspace(0,len(ordered)-1,sample_count)).astype(np.int32))
    valid_surface=garment & np.isfinite(relief)
    depth_deltas=[]; rgb_deltas=[]; normal_jumps=[]; depth_signs=[]
    sample_interiors=[]
    offsets=(10.0,16.0,22.0)
    for si in sample_indices.tolist():
        p=np.asarray(ordered[int(si)],dtype=np.float64)
        d2=np.sum((pts-p[None,:])**2,axis=1)
        neigh=pts[np.argsort(d2)[:min(19,len(pts))]]
        if len(neigh)>=5:
            c=neigh.mean(axis=0); Y=neigh-c[None,:]
            vv,ee=np.linalg.eigh(Y.T@Y)
            tangent=ee[:,int(np.argmax(vv))]
            tangent/=max(1e-9,float(np.linalg.norm(tangent)))
        else:
            tangent=tangent_global.copy()
        normal=np.asarray([-tangent[1],tangent[0]],dtype=np.float64)
        per_depth=[]; per_rgb=[]; per_normal=[]; per_sign=[]
        for off in offsets:
            pa=p+normal*off; pb=p-normal*off
            for q in (pa,pb):
                if not (3<=q[0]<branch.shape[1]-3 and 3<=q[1]<branch.shape[0]-3):
                    break
            else:
                ax,ay=int(round(pa[0])),int(round(pa[1])); bx,by=int(round(pb[0])),int(round(pb[1]))
                if inside[ay,ax]<3.0 or inside[by,bx]<3.0:
                    continue
                za=_e122_patch_median(relief,valid_surface,pa[0],pa[1],3)
                zb=_e122_patch_median(relief,valid_surface,pb[0],pb[1],3)
                la=_e122_patch_median(L,garment,pa[0],pa[1],3)
                lb=_e122_patch_median(L,garment,pb[0],pb[1],3)
                if za is None or zb is None:
                    continue
                per_depth.append(abs(float(za-zb)))
                per_sign.append(float(za-zb))
                if la is not None and lb is not None:
                    per_rgb.append(abs(float(la-lb)))
                ga=np.asarray([float(dzdx[ay,ax]),float(dzdy[ay,ax])],dtype=np.float64)
                gb=np.asarray([float(dzdx[by,bx]),float(dzdy[by,bx])],dtype=np.float64)
                na=np.asarray([-ga[0],-ga[1],1.0],dtype=np.float64); nb=np.asarray([-gb[0],-gb[1],1.0],dtype=np.float64)
                na/=max(1e-9,float(np.linalg.norm(na))); nb/=max(1e-9,float(np.linalg.norm(nb)))
                per_normal.append(float(np.clip(1.0-np.dot(na,nb),0.0,1.0)))
        if per_depth:
            depth_deltas.append(float(np.median(per_depth)))
            depth_signs.append(float(np.median(per_sign)))
            rgb_deltas.append(float(np.median(per_rgb)) if per_rgb else 0.0)
            normal_jumps.append(float(np.median(per_normal)) if per_normal else 0.0)
            xi=int(np.clip(round(p[0]),0,branch.shape[1]-1)); yi=int(np.clip(round(p[1]),0,branch.shape[0]-1))
            sample_interiors.append(float(inside[yi,xi]))
    nvalid=len(depth_deltas)
    valid_ratio=float(nvalid)/float(max(1,len(sample_indices)))
    depth_step=float(np.median(depth_deltas)) if depth_deltas else 0.0
    rgb_step=float(np.median(rgb_deltas)) if rgb_deltas else 0.0
    normal_jump=float(np.median(normal_jumps)) if normal_jumps else 0.0
    strong_sign=np.asarray([v for v in depth_signs if abs(float(v))>=0.20],dtype=np.float64)
    if strong_sign.size>=3:
        pos=float(np.mean(strong_sign>0)); neg=float(np.mean(strong_sign<0)); side_consistency=max(pos,neg)
    else:
        side_consistency=0.50
    raw_score=np.asarray(raw_score_map,dtype=np.float32)
    rv=raw_score[branch & np.isfinite(raw_score)]
    raw_strength=float(np.mean(rv)) if rv.size else 0.0
    struct=np.asarray(structural_u8,dtype=np.uint8)>0
    if np.any(struct):
        struct_near=cv2.dilate((struct.astype(np.uint8)*255),np.ones((5,5),np.uint8),iterations=1)>0
        structural_support=float(np.mean(struct_near[branch]))
    else:
        structural_support=0.0
    deform=np.asarray(pose_deform_u8,dtype=np.uint8)>0
    if np.any(deform):
        deform_near=cv2.dilate((deform.astype(np.uint8)*255),cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(25,25)),iterations=1)>0
        deform_support=float(np.mean(deform_near[branch]))
    else:
        deform_support=0.0
    seam=np.asarray(seam_u8,dtype=np.uint8)>0
    seam_overlap=float(np.mean(seam[branch])) if np.any(seam) else 0.0
    interior_med=float(np.median(sample_interiors)) if sample_interiors else 0.0
    depth_score=float(np.tanh(max(0.0,depth_step)/1.8))
    rgb_score=float(np.tanh(max(0.0,rgb_step)/18.0))
    normal_score=float(np.clip(normal_jump/0.55,0.0,1.0))
    # E124: 0.50 is the no-preference baseline for sign consistency. Do not
    # give every wrinkle half credit merely because the sign test is undecided.
    side_consistency_score=float(np.clip((side_consistency-0.50)/0.35,0.0,1.0))
    # Moderate UniDepth increase: bilateral height is primary, while local
    # orientation and persistent side ordering confirm a true folded surface.
    geometry=float(np.clip(0.56*depth_score+0.24*normal_score+0.20*side_consistency_score,0.0,1.0))
    appearance=float(np.clip(0.55*rgb_score+0.45*raw_strength,0.0,1.0))
    length_score=float(np.clip(span/150.0,0.0,1.0))
    interior_score=float(np.clip(interior_med/18.0,0.0,1.0))
    structure=float(np.clip(0.42*length_score+0.28*linearity+0.20*interior_score+0.10*structural_support,0.0,1.0))
    pose_term=float(np.clip(0.50+0.50*deform_support*float(np.clip(pose_deform_conf,0.0,1.0)),0.0,1.0))
    noise_penalty=float(0.10*(1.0-valid_ratio))
    seam_penalty=float(0.38*seam_overlap)
    # E124 raises UniDepth geometry from 0.40 -> 0.48 while retaining
    # appearance/continuity/pose safeguards against monocular-depth artifacts.
    fold_score=float(np.clip(0.48*geometry+0.16*appearance+0.23*structure+0.13*pose_term-seam_penalty-noise_penalty,0.0,1.0))
    unidepth_fold_support=float(np.clip(0.60*depth_score+0.20*normal_score+0.20*side_consistency_score,0.0,1.0))
    return {
        'fold_score':fold_score,'geometry_score':geometry,'appearance_score':appearance,'structure_score':structure,
        'pose_score':pose_term,'depth_step_mm':depth_step,'depth_step_score':depth_score,
        'surface_normal_jump':normal_jump,'surface_normal_score':normal_score,'rgb_bilateral_step':rgb_step,
        'rgb_bilateral_score':rgb_score,'side_consistency':float(side_consistency),'side_consistency_score':float(side_consistency_score),
        'unidepth_fold_support':float(unidepth_fold_support),'raw_crease_strength':raw_strength,
        'structural_support':structural_support,'pose_deform_support':deform_support,'seam_overlap':seam_overlap,
        'seam_penalty':seam_penalty,'noise_penalty':noise_penalty,'valid_sample_ratio':valid_ratio,'valid_samples':int(nvalid),
        'span_px':span,'linearity':linearity,'interior_median_px':interior_med,
        'centroid_px':[float(ctr[0]),float(ctr[1])],
    }



def _e122_build_scored_crease_pool(depth_state, mask_u8, args):
    """E123 STRUCTURAL-first branch pool with controlled sensitive rescue.

    STRUCTURAL and SENSITIVE skeletons are never OR-welded.  Structural branches
    are scored first. Sensitive branches are added only when they are spatially
    novel relative to structural evidence.  A long/interior/non-seam structural
    branch may enter a conservative rescue tier even when the bilateral score is
    below the still-uncalibrated strong threshold.
    """
    mask=((np.asarray(mask_u8,dtype=np.uint8)>0).astype(np.uint8)*255)
    sensitive=((np.asarray(depth_state.get('fold_crease_skeleton',np.zeros_like(mask)),dtype=np.uint8)>0).astype(np.uint8)*255)
    structural=((np.asarray(depth_state.get('fold_crease_structural_skeleton',np.zeros_like(mask)),dtype=np.uint8)>0).astype(np.uint8)*255)
    frame=depth_state.get('frame')
    relief=np.asarray(depth_state.get('local_relief_mm',np.zeros(mask.shape,dtype=np.float32)),dtype=np.float32)
    raw_score=np.asarray(depth_state.get('fold_crease_score',np.zeros(mask.shape,dtype=np.float32)),dtype=np.float32)
    pose_deform=np.asarray(depth_state.get('pose_deform_mask',np.zeros_like(mask)),dtype=np.uint8)
    pose_conf=float(depth_state.get('pose_deform_confidence',0.0) or 0.0)
    seam=np.asarray(depth_state.get('pose_seam_exclusion_mask',np.zeros_like(mask)),dtype=np.uint8)
    preview_threshold=float(getattr(args,'micro_unfold_e122_preview_score',0.34))
    exec_threshold=float(getattr(args,'micro_unfold_e122_exec_score',0.48))
    rescue_threshold=float(getattr(args,'micro_unfold_e123_structural_rescue_score',0.34))
    rescue_min_span=float(getattr(args,'micro_unfold_e123_structural_rescue_min_span_px',34.0))
    min_pixels=max(6,int(getattr(args,'micro_unfold_e122_branch_min_px',8)))
    empty={
        'available':False,'raw_skeleton':np.zeros_like(mask),'preview_skeleton':np.zeros_like(mask),'execution_skeleton':np.zeros_like(mask),
        'rejected_skeleton':np.zeros_like(mask),'branch_score_map':np.zeros(mask.shape,dtype=np.float32),
        'branch_unidepth_map':np.zeros(mask.shape,dtype=np.float32),'branches':[],
        'preview_threshold':preview_threshold,'execution_threshold':exec_threshold,'rescue_threshold':rescue_threshold,
        'structural_branch_count':0,'sensitive_branch_count':0,'rescue_branch_count':0,
    }
    if frame is None or (cv2.countNonZero(structural)<min_pixels and cv2.countNonZero(sensitive)<min_pixels):
        return empty

    structural_branches=_e122_split_crease_branches(structural,min_pixels=min_pixels) if cv2.countNonZero(structural)>=min_pixels else []
    sensitive_branches=_e122_split_crease_branches(sensitive,min_pixels=min_pixels) if cv2.countNonZero(sensitive)>=min_pixels else []

    # Sensitive evidence is useful for recall, but do not duplicate a structural
    # branch that already covers the same place.
    structural_union=np.zeros_like(mask)
    for b in structural_branches:
        structural_union[np.asarray(b)>0]=255
    structural_near=(cv2.dilate(structural_union,np.ones((7,7),np.uint8),iterations=1)>0) if cv2.countNonZero(structural_union)>0 else np.zeros(mask.shape,dtype=bool)
    novel_sensitive=[]
    for b in sensitive_branches:
        bb=np.asarray(b,dtype=np.uint8)>0
        area=max(1,int(np.count_nonzero(bb)))
        overlap=float(np.count_nonzero(bb & structural_near))/float(area)
        if overlap<0.62:
            novel_sensitive.append(b)

    sources=[('STRUCTURAL',b) for b in structural_branches]+[('SENSITIVE',b) for b in novel_sensitive]
    if not sources:
        return empty

    scored=[]; raw_out=np.zeros_like(mask); preview=np.zeros_like(mask); execution=np.zeros_like(mask); rejected=np.zeros_like(mask)
    score_map=np.zeros(mask.shape,dtype=np.float32)
    unidepth_map=np.zeros(mask.shape,dtype=np.float32)
    rescue_count=0
    for idx,(source,b) in enumerate(sources,1):
        metrics=_e122_score_fold_branch(b,frame,relief,mask,raw_score,structural,pose_deform,pose_conf,seam)
        raw_fold=float(metrics.get('fold_score',0.0))
        # Structural evidence has already passed the stricter upstream gate; give
        # it a small prior rather than making it re-prove the same fact from depth.
        source_bonus=0.045 if source=='STRUCTURAL' else 0.0
        score=float(np.clip(raw_fold+source_bonus,0.0,1.0))
        span=float(metrics.get('span_px',0.0))
        valid_ratio=float(metrics.get('valid_sample_ratio',0.0))
        seam_overlap=float(metrics.get('seam_overlap',0.0))
        interior=float(metrics.get('interior_median_px',0.0))
        depth_step=float(metrics.get('depth_step_mm',0.0))
        side_cons=float(metrics.get('side_consistency',0.50))
        normal_s=float(metrics.get('surface_normal_score',0.0))
        unidepth_support=float(metrics.get('unidepth_fold_support',0.0))
        exec_min_depth=float(getattr(args,'micro_unfold_e124_exec_min_depth_mm',0.50))
        exec_min_side=float(getattr(args,'micro_unfold_e124_exec_min_side_consistency',0.56))
        exec_min_normal=float(getattr(args,'micro_unfold_e124_exec_min_normal_score',0.18))
        rescue_min_depth=float(getattr(args,'micro_unfold_e124_rescue_min_depth_mm',0.75))
        rescue_min_side=float(getattr(args,'micro_unfold_e124_rescue_min_side_consistency',0.60))
        depth_exec_ok=bool(depth_step>=exec_min_depth and (side_cons>=exec_min_side or normal_s>=exec_min_normal))
        # E124: a long structural line alone is no longer enough for execution.
        # Rescue requires a persistent bilateral UniDepth height discontinuity.
        structural_rescue=bool(
            source=='STRUCTURAL' and score>=rescue_threshold and span>=rescue_min_span and
            valid_ratio>=0.25 and seam_overlap<0.18 and interior>=3.0 and
            depth_step>=rescue_min_depth and side_cons>=rescue_min_side
        )
        preview_ok=bool(score>=preview_threshold or structural_rescue)
        execution_ok=bool((score>=exec_threshold and depth_exec_ok) or structural_rescue)
        if structural_rescue and score<exec_threshold:
            rescue_count+=1
        item={
            'branch_id':int(idx),'source':source,'fold_score_raw':raw_fold,'fold_score':score,
            **{k:v for k,v in metrics.items() if k!='fold_score'},
            'source_bonus':float(source_bonus),'structural_rescue':bool(structural_rescue),
            'depth_execution_ok':bool(depth_exec_ok),'unidepth_fold_support':float(unidepth_support),
            'preview_ok':preview_ok,'execution_ok':execution_ok,'mask':np.asarray(b,dtype=np.uint8).copy(),
        }
        scored.append(item)
        bb=np.asarray(b,dtype=np.uint8)>0
        raw_out[bb]=255; score_map[bb]=np.maximum(score_map[bb],score)
        unidepth_map[bb]=np.maximum(unidepth_map[bb],float(unidepth_support))
        if preview_ok: preview[bb]=255
        else: rejected[bb]=255
        if execution_ok: execution[bb]=255

    scored.sort(key=lambda q:(bool(q.get('execution_ok',False)),float(q.get('fold_score',0.0)),float(q.get('span_px',0.0))),reverse=True)
    info=[{k:v for k,v in q.items() if k!='mask'} for q in scored[:24]]
    return {
        'available':bool(cv2.countNonZero(preview)>=8),'raw_skeleton':raw_out,'preview_skeleton':preview,'execution_skeleton':execution,
        'rejected_skeleton':rejected,'branch_score_map':score_map,'branch_unidepth_map':unidepth_map,
        'branches':scored,'branch_info':info,
        'preview_threshold':preview_threshold,'execution_threshold':exec_threshold,'rescue_threshold':rescue_threshold,
        'raw_branch_count':int(len(scored)),'preview_branch_count':int(sum(bool(q['preview_ok']) for q in scored)),
        'execution_branch_count':int(sum(bool(q['execution_ok']) for q in scored)),
        'structural_branch_count':int(len(structural_branches)),'sensitive_branch_count':int(len(novel_sensitive)),
        'rescue_branch_count':int(rescue_count),
    }


def _e122_validate_same_fold_pair(anchor_board, active_board, hold_board, fold_normal_board,
                                  fold_tangent_board, args):
    """Require HOLD/ACTIVE to straddle the same local fold and cross it normally."""
    anchor=np.asarray(anchor_board,dtype=np.float64); active=np.asarray(active_board,dtype=np.float64); hold=np.asarray(hold_board,dtype=np.float64)
    normal=np.asarray(fold_normal_board,dtype=np.float64); tangent=np.asarray(fold_tangent_board,dtype=np.float64)
    normal/=max(1e-9,float(np.linalg.norm(normal))); tangent/=max(1e-9,float(np.linalg.norm(tangent)))
    pair=active-hold; sep=float(np.linalg.norm(pair))
    if sep<=1e-6:
        return False,{'reason':'PAIR_COLLAPSED'}
    pair_u=pair/sep
    a_signed=float(np.dot(active-anchor,normal)); h_signed=float(np.dot(hold-anchor,normal))
    opposite=bool(a_signed>=8.0 and h_signed<=-8.0)
    normal_cos=float(abs(np.dot(pair_u,normal)))
    tangent_shift=float(abs(np.dot(pair,tangent)))
    tangent_ratio=float(tangent_shift/max(1e-6,sep))
    # Distance from fold anchor to HOLD->ACTIVE segment.
    v=pair; tau=float(np.clip(np.dot(anchor-hold,v)/max(1e-9,float(np.dot(v,v))),0.0,1.0))
    nearest=hold+tau*v; anchor_cross=float(np.linalg.norm(nearest-anchor))
    min_cos=float(getattr(args,'micro_unfold_e122_pair_normal_cos',0.72))
    max_tangent_ratio=float(getattr(args,'micro_unfold_e122_pair_max_tangent_ratio',0.55))
    max_cross=float(getattr(args,'micro_unfold_e122_pair_anchor_cross_mm',18.0))
    ok=bool(opposite and normal_cos>=min_cos and tangent_ratio<=max_tangent_ratio and anchor_cross<=max_cross)
    reason='OK' if ok else f'opposite={opposite} normalCos={normal_cos:.2f}/{min_cos:.2f} tangentRatio={tangent_ratio:.2f}/{max_tangent_ratio:.2f} anchorCross={anchor_cross:.1f}/{max_cross:.1f}mm'
    return ok,{
        'reason':reason,'opposite_sides':opposite,'pair_normal_cos':normal_cos,'pair_tangent_shift_mm':tangent_shift,
        'pair_tangent_ratio':tangent_ratio,'pair_anchor_cross_mm':anchor_cross,'active_signed_mm':a_signed,'hold_signed_mm':h_signed,
    }

def _e126_validate_primary_fold_region_pair(active_px, hold_px, fold_region_u8, fold_boundary_u8):
    """Region-first pair gate used when POSE/UniDepth already identified a fold.

    Unlike the E122 same-crease gate, this does not require the HOLD->ACTIVE segment
    to be almost perpendicular to one tiny local crease fragment.  It requires the
    segment to go from BODY/outside into the known folded region and to cross that
    region's boundary.  Robot reach/path/board/Z safety remains checked separately.
    """
    region=(np.asarray(fold_region_u8,dtype=np.uint8)>0)
    boundary=((np.asarray(fold_boundary_u8,dtype=np.uint8)>0).astype(np.uint8)*255)
    h,w=region.shape[:2]
    a=np.asarray(active_px,dtype=np.float64).reshape(2); hpt=np.asarray(hold_px,dtype=np.float64).reshape(2)
    if not np.all(np.isfinite(a)) or not np.all(np.isfinite(hpt)):
        return False,{'reason':'NONFINITE_POINT'}
    def _local(p,rad):
        x=int(round(float(p[0]))); y=int(round(float(p[1])))
        x0=max(0,x-rad); x1=min(w,x+rad+1); y0=max(0,y-rad); y1=min(h,y+rad+1)
        if x0>=x1 or y0>=y1: return 0.0
        yy,xx=np.ogrid[y0:y1,x0:x1]
        d=((xx-x)**2+(yy-y)**2)<=rad*rad
        vals=region[y0:y1,x0:x1][d]
        return float(np.mean(vals)) if vals.size else 0.0
    active_support=_local(a,10); hold_support=_local(hpt,11)
    delta=active_support-hold_support
    v=a-hpt; dist=float(np.linalg.norm(v))
    if dist<8.0:
        return False,{'reason':'PAIR_TOO_SHORT','active_support':active_support,'hold_support':hold_support,'delta':delta}
    n=int(np.clip(round(dist*1.5),31,181))
    ts=np.linspace(0.0,1.0,n,dtype=np.float64)
    pts=hpt[None,:]+ts[:,None]*v[None,:]
    xi=np.clip(np.rint(pts[:,0]).astype(np.int32),0,w-1); yi=np.clip(np.rint(pts[:,1]).astype(np.int32),0,h-1)
    rv=region[yi,xi].astype(np.uint8)
    # Crossing is accepted either by direct boundary contact or by a clear 0->1
    # fold-region membership transition along the physical pair segment.
    bd=cv2.dilate(boundary,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(11,11)),iterations=1)>0 if cv2.countNonZero(boundary)>0 else np.zeros_like(region)
    boundary_hit=bool(np.any(bd[yi,xi]))
    transition=bool(np.any((rv[:-1]==0)&(rv[1:]==1))) if rv.size>=2 else False
    first=float(np.mean(rv[:max(3,n//4)])); last=float(np.mean(rv[-max(3,n//4):]))
    ok=bool(active_support>=0.14 and hold_support<=0.55 and delta>=0.05 and (boundary_hit or transition) and last>=first+0.08)
    reason='OK' if ok else f'A={active_support:.2f} H={hold_support:.2f} d={delta:.2f} boundary={boundary_hit} transition={transition} first={first:.2f} last={last:.2f}'
    return ok,{
        'reason':reason,'active_support':active_support,'hold_support':hold_support,'delta':delta,
        'boundary_hit':boundary_hit,'transition':transition,'segment_start_fold':first,'segment_end_fold':last,
    }


def _e94_crease_support_for_hypothesis(crease_state, seed_bool, fold_bool, seed_centroid_px, garment_center_px):
    """Return how strongly an internal crease supports one raised hypothesis."""
    if not isinstance(crease_state, dict) or not bool(crease_state.get("available", False)):
        return {
            "crease_support": 0.0, "crease_between_score": 0.0,
            "crease_proximity_score": 0.0, "crease_boundary_score": 0.0,
            "seed_crease_distance_px": 999.0, "segment_crease_distance_px": 999.0,
        }
    crease = np.asarray(crease_state.get("skeleton_mask"), dtype=np.uint8) > 0
    if int(np.count_nonzero(crease)) < 4:
        return {
            "crease_support": 0.0, "crease_between_score": 0.0,
            "crease_proximity_score": 0.0, "crease_boundary_score": 0.0,
            "seed_crease_distance_px": 999.0, "segment_crease_distance_px": 999.0,
        }
    inv = ((~crease).astype(np.uint8) * 255)
    dist = cv2.distanceTransform(inv, cv2.DIST_L2, 5).astype(np.float32)
    seed = np.asarray(seed_bool, dtype=bool)
    fold = np.asarray(fold_bool, dtype=bool)
    seed_vals = dist[seed]
    seed_vals = seed_vals[np.isfinite(seed_vals)]
    seed_dist = float(np.percentile(seed_vals, 10.0)) if seed_vals.size else 999.0
    proximity = float(np.clip(1.0 - seed_dist / 90.0, 0.0, 1.0))

    fold_u8 = (fold.astype(np.uint8) * 255)
    border = (cv2.dilate(fold_u8, np.ones((21,21), np.uint8), iterations=1) > 0) & \
             ~(cv2.erode(fold_u8, np.ones((21,21), np.uint8), iterations=1) > 0)
    near = cv2.dilate(fold_u8, np.ones((31,31), np.uint8), iterations=1) > 0
    crease_near_n = int(np.count_nonzero(crease & near))
    crease_border_n = int(np.count_nonzero(crease & border))
    boundary_score = float(np.clip(crease_border_n / float(max(10, crease_near_n)), 0.0, 1.0))

    a = np.asarray(seed_centroid_px, dtype=np.float64)
    b = np.asarray(garment_center_px, dtype=np.float64)
    seg_len = float(np.linalg.norm(b-a))
    seg_min = 999.0
    if seg_len > 4.0:
        n = int(np.clip(round(seg_len * 1.5), 24, 420))
        ts = np.linspace(0.12, 0.88, n, dtype=np.float64)
        pts = a[None,:] + ts[:,None] * (b-a)[None,:]
        xi = np.clip(np.rint(pts[:,0]).astype(np.int32), 0, dist.shape[1]-1)
        yi = np.clip(np.rint(pts[:,1]).astype(np.int32), 0, dist.shape[0]-1)
        seg_min = float(np.min(dist[yi, xi]))
    between = float(np.clip(1.0 - seg_min / 20.0, 0.0, 1.0))
    support = float(np.clip(0.50*between + 0.30*proximity + 0.20*boundary_score, 0.0, 1.0))
    return {
        "crease_support": support,
        "crease_between_score": between,
        "crease_proximity_score": proximity,
        "crease_boundary_score": boundary_score,
        "seed_crease_distance_px": seed_dist,
        "segment_crease_distance_px": seg_min,
    }


def _fix89_board_to_px(H_value, board_xy):
    Hm = np.asarray(H_value, dtype=np.float64)
    if Hm.shape != (3,3):
        raise RuntimeError(f"E91 invalid Homography shape for board->pixel: {Hm.shape}")
    invH = _e156_inverse_h(Hm)
    p = np.asarray([[[float(board_xy[0]), float(board_xy[1])]]], dtype=np.float32)
    q = cv2.perspectiveTransform(p, invH)[0,0]
    return np.asarray([float(q[0]), float(q[1])], dtype=np.float64)


def _fix89_find_outside_pull_endpoint(mask_u8, H_value, cfg, active_board, pull_u, args, log_final=False):
    """Follow HOLD->ACTIVE ray until it exits garment, then continue outside it."""
    mask = np.asarray(mask_u8, dtype=np.uint8) > 0
    h, w = mask.shape[:2]
    # Dilate a little so tiny segmentation holes cannot be mistaken for garment exit.
    stable_mask = cv2.dilate((mask.astype(np.uint8)*255), np.ones((9,9), np.uint8), iterations=1) > 0
    step = float(np.clip(float(getattr(args, 'micro_unfold_boundary_sample_mm', 2.0)), 1.0, 5.0))
    outside_margin = max(8.0, float(getattr(args, 'micro_unfold_outside_margin_mm', 35.0)))
    max_pull = max(60.0, float(getattr(args, 'micro_unfold_max_pull_mm', 220.0)))
    min_pull = max(20.0, float(getattr(args, 'micro_unfold_pull_mm', 30.0)))

    boundary_d = None
    outside_run = 0
    d = 0.0
    while d <= max_pull + 1e-6:
        bp = np.asarray(active_board, dtype=np.float64) + np.asarray(pull_u, dtype=np.float64) * d
        try:
            px = _fix89_board_to_px(H_value, bp)
            xi, yi = int(round(px[0])), int(round(px[1]))
            inside_img = 0 <= xi < w and 0 <= yi < h
            on_garment = bool(inside_img and stable_mask[yi, xi])
        except Exception:
            on_garment = False
        if on_garment:
            outside_run = 0
        else:
            outside_run += 1
            if outside_run >= 3:
                boundary_d = max(0.0, d - 2.0*step)
                break
        d += step
    if boundary_d is None:
        raise RuntimeError(f"E91 HOLD->ACTIVE ray did not leave garment within {max_pull:.1f}mm")

    pull_scale = float(np.clip(float(getattr(args, 'micro_unfold_pull_scale', 1.20)), 1.0, 1.50))
    base_desired = max(min_pull, boundary_d + outside_margin)
    desired = min(base_desired * pull_scale, max_pull)
    if bool(log_final):
        print(f"[E109-PULL] base={base_desired:.1f}mm scale={pull_scale:.2f} requested={desired:.1f}mm max={max_pull:.1f}mm")
    # Keep the endpoint physically inside the folding board while still outside cloth.
    chosen = None
    chosen_px = None
    trial = desired
    while trial >= boundary_d + 6.0 - 1e-6:
        bp = np.asarray(active_board, dtype=np.float64) + np.asarray(pull_u, dtype=np.float64) * trial
        if _fix53_board_inside(cfg, bp, margin=20.0):
            px = _fix89_board_to_px(H_value, bp)
            xi, yi = int(round(px[0])), int(round(px[1]))
            outside_cloth = not (0 <= xi < w and 0 <= yi < h and stable_mask[yi, xi])
            if outside_cloth:
                chosen = bp
                chosen_px = px
                break
        trial -= step
    if chosen is None:
        raise RuntimeError("E91 cannot place ACTIVE pull endpoint outside garment while preserving board safety margin")
    return chosen, float(trial), float(boundary_d), float(max(0.0, trial-boundary_d)), chosen_px


def _e111_find_outer_normal_pull_endpoint(mask_u8, H_value, cfg, active_board, hold_board, pull_hint_u, args, log_final=False,
                                            stable_mask_cached=None, outer_contour_cached=None, invH_cached=None):
    """E111 ACTIVE trajectory: local garment-boundary normal + ~2x HOLD->boundary length.

    The Fold-Crease pair still decides which side is ACTIVE.  Starting from that
    outward hint, find the garment boundary reached from ACTIVE, estimate the
    local outer-contour tangent there, and use its outward normal as the actual
    pull direction.  Length is targeted from HOLD to that same outer boundary:
    ``2 * HOLD_to_boundary``.  Existing board/max-pull safety clipping remains.
    """
    mask = np.asarray(mask_u8, dtype=np.uint8) > 0
    h, w = mask.shape[:2]
    stable_mask = (np.asarray(stable_mask_cached, dtype=bool) if stable_mask_cached is not None else
                   cv2.dilate((mask.astype(np.uint8) * 255), np.ones((9,9), np.uint8), iterations=1) > 0)
    invH_local = _e156_inverse_h(H_value) if invH_cached is None else np.asarray(invH_cached, dtype=np.float64)
    active_board = np.asarray(active_board, dtype=np.float64).reshape(2)
    hold_board = np.asarray(hold_board, dtype=np.float64).reshape(2)
    hint = np.asarray(pull_hint_u, dtype=np.float64).reshape(2)
    hn = float(np.linalg.norm(hint))
    if hn <= 1e-9:
        raise RuntimeError("E111 pull hint collapsed")
    hint = hint / hn

    step = float(np.clip(float(getattr(args, 'micro_unfold_boundary_sample_mm', 2.0)), 1.0, 5.0))
    max_pull = max(60.0, float(getattr(args, 'micro_unfold_max_pull_mm', 264.0)))
    min_pull = max(20.0, float(getattr(args, 'micro_unfold_pull_mm', 30.0)))
    length_factor = float(np.clip(float(getattr(args, 'micro_unfold_hold_boundary_factor', 2.0)), 1.2, 2.5))
    normal_window = int(np.clip(int(getattr(args, 'micro_unfold_outer_normal_window_px', 22)), 8, 50))
    search_limit = max(360.0, max_pull + 100.0)

    def _sample_inside(board_xy):
        try:
            px = _fix89_board_to_px(H_value, board_xy)
            xi, yi = int(round(float(px[0]))), int(round(float(px[1])))
            return bool(0 <= xi < w and 0 <= yi < h and stable_mask[yi, xi]), px
        except Exception:
            return False, None

    def _ray_exit(origin, unit, limit):
        # V5: exact same 3-consecutive-outside rule/sampling step as E111 v4,
        # but transform the whole ray in one OpenCV call. This removes hundreds
        # of Python perspectiveTransform calls per E128 candidate.
        return _e130_ray_exit(
            stable_mask.astype(np.uint8) * 255, H_value, origin, unit,
            max_mm=float(limit), step_mm=step, stable_mask=stable_mask, invH=invH_local
        )

    # First locate the relevant outer boundary in the Fold-Crease outward direction.
    hint_exit = _ray_exit(active_board, hint, search_limit)
    if hint_exit is None:
        raise RuntimeError(f"E111 ACTIVE outward hint did not leave garment within {search_limit:.1f}mm")
    _, _, hint_boundary_px = hint_exit

    pull_u = hint.copy()
    normal_hint_angle_deg = 0.0
    contour_normal_ok = False
    try:
        if outer_contour_cached is not None:
            contour = np.asarray(outer_contour_cached, dtype=np.float64).reshape(-1, 2)
            contours = [contour.reshape(-1, 1, 2).astype(np.int32)]
        else:
            contours, _ = cv2.findContours((mask.astype(np.uint8) * 255), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            contour = max(contours, key=cv2.contourArea).reshape(-1, 2).astype(np.float64) if contours else None
        if contour is not None and len(contour) >= 3 and hint_boundary_px is not None:
            bpix = np.asarray(hint_boundary_px, dtype=np.float64).reshape(2)
            idx = int(np.argmin(np.sum((contour - bpix[None, :]) ** 2, axis=1)))
            npts = len(contour)
            ids = [(idx + k) % npts for k in range(-normal_window, normal_window + 1)]
            local = contour[np.asarray(ids, dtype=np.int32)]
            ctr = local.mean(axis=0)
            X = local - ctr[None, :]
            vals, vecs = np.linalg.eigh(X.T @ X)
            tangent_px = vecs[:, int(np.argmax(vals))]
            tangent_px = tangent_px / max(1e-9, float(np.linalg.norm(tangent_px)))
            p_a = bpix - tangent_px * 16.0
            p_b = bpix + tangent_px * 16.0
            b_a = _fix53_px_to_board(H_value, p_a)
            b_b = _fix53_px_to_board(H_value, p_b)
            tangent_board = np.asarray(b_b - b_a, dtype=np.float64)
            tangent_board /= max(1e-9, float(np.linalg.norm(tangent_board)))
            normals = [np.asarray([-tangent_board[1], tangent_board[0]], dtype=np.float64),
                       np.asarray([ tangent_board[1],-tangent_board[0]], dtype=np.float64)]
            scored = []
            for nvec in normals:
                nvec /= max(1e-9, float(np.linalg.norm(nvec)))
                align = float(np.dot(nvec, hint))
                if align <= 0.05:
                    continue
                ex = _ray_exit(active_board, nvec, search_limit)
                if ex is None:
                    continue
                # Prefer the true local normal that remains compatible with the Fold-Crease outward side.
                scored.append((align, -float(ex[0]), nvec, ex))
            if scored:
                scored.sort(key=lambda q: (q[0], q[1]), reverse=True)
                pull_u = np.asarray(scored[0][2], dtype=np.float64)
                pull_u /= max(1e-9, float(np.linalg.norm(pull_u)))
                normal_hint_angle_deg = float(np.degrees(np.arccos(np.clip(np.dot(pull_u, hint), -1.0, 1.0))))
                contour_normal_ok = True
    except Exception:
        contour_normal_ok = False
        pull_u = hint.copy()
        normal_hint_angle_deg = 0.0

    active_exit = _ray_exit(active_board, pull_u, search_limit)
    hold_exit = _ray_exit(hold_board, pull_u, search_limit)
    if active_exit is None or hold_exit is None:
        # If the local normal is geometrically awkward, preserve the Fold-Crease outward side
        # rather than producing an unsafe endpoint.
        pull_u = hint.copy()
        normal_hint_angle_deg = 0.0
        contour_normal_ok = False
        active_exit = _ray_exit(active_board, pull_u, search_limit)
        hold_exit = _ray_exit(hold_board, pull_u, search_limit)
    if active_exit is None or hold_exit is None:
        raise RuntimeError("E111 could not resolve ACTIVE/HOLD distance to the corresponding garment boundary")

    active_boundary_d = float(active_exit[0])
    hold_boundary_d = float(hold_exit[0])
    target_two_x = max(min_pull, length_factor * hold_boundary_d)
    minimum_outside = active_boundary_d + 6.0
    desired = max(target_two_x, minimum_outside)
    desired = min(desired, max_pull)
    if desired + 1e-6 < minimum_outside:
        raise RuntimeError(
            f"E111 max pull {max_pull:.1f}mm cannot clear garment boundary at {active_boundary_d:.1f}mm"
        )

    chosen = None
    chosen_px = None
    trial = float(desired)
    while trial >= minimum_outside - 1e-6:
        bp = active_board + pull_u * trial
        if _fix53_board_inside(cfg, bp, margin=20.0):
            inside_now, px = _sample_inside(bp)
            if not inside_now:
                chosen = bp
                chosen_px = px if px is not None else _fix89_board_to_px(H_value, bp)
                break
        trial -= step
    if chosen is None:
        raise RuntimeError("E111 cannot place boundary-normal ACTIVE endpoint outside garment while preserving board safety")

    outside_used = max(0.0, float(trial) - active_boundary_d)
    if bool(log_final):
        mode = "OUTER_NORMAL" if contour_normal_ok else "CREASE_HINT_FALLBACK"
        print(
            f"[E111-PULL] mode={mode} normalDelta={normal_hint_angle_deg:.1f}deg "
            f"HOLD->boundary={hold_boundary_d:.1f}mm target={length_factor:.2f}x={target_two_x:.1f}mm "
            f"ACTIVE->boundary={active_boundary_d:.1f}mm requested={trial:.1f}mm max={max_pull:.1f}mm"
        )
    return (np.asarray(chosen, dtype=np.float64), float(trial), active_boundary_d,
            float(outside_used), np.asarray(chosen_px, dtype=np.float64),
            np.asarray(pull_u, dtype=np.float64), hold_boundary_d,
            float(normal_hint_angle_deg), bool(contour_normal_ok))


_E117_UNIDEPTH_BASE_CACHE = None


# E119 caches the post-UniDepth features too.  E118 reused only the raw UniDepth
# base, then paid most of the cost again for mask-relief, multi-scale prominence
# and TOP-candidate extraction on every I.
_E119_DEPTH_FEATURE_CACHE = None
_E120_CREASE_TEMP_CACHE = None

def _e120_pose_signature(pose_state):
    pts=[]
    named=dict((pose_state or {}).get('named_points',{}) or {}) if isinstance(pose_state,dict) else {}
    for name in _E102_TSHIRT_POSE_KPT_NAMES:
        q=dict(named.get(name,{}) or {})
        try:
            x=float(q.get('x')); y=float(q.get('y')); c=float(q.get('conf',0.0))
        except Exception:
            x=y=np.nan; c=0.0
        pts.append([x,y,c])
    return np.asarray(pts,dtype=np.float32)

def _e120_stabilize_crease_state(state, frame_bgr, mask_u8, pose_state, args):
    """Same-scene hysteresis for Fold Crease without freezing stale geometry.

    Previous crease pixels survive only when the *current* RGB score map still has
    low-threshold support there.  Scene matching also checks mask IoU, grayscale MAD
    and median pose displacement, so a genuinely changed garment is not inherited.
    """
    global _E120_CREASE_TEMP_CACHE
    if not isinstance(state,dict):
        return state, False, 999.0, 0.0, 999.0
    try:
        g,m=_e117_depth_cache_signature(frame_bgr,mask_u8)
        ps=_e120_pose_signature(pose_state)
        cache=_E120_CREASE_TEMP_CACHE
        hit=False; mad=999.0; iou=0.0; pose_move=999.0
        if isinstance(cache,dict):
            pg=np.asarray(cache.get('gray_small'),dtype=np.uint8); pm=np.asarray(cache.get('mask_small'),dtype=bool)
            if pg.shape==g.shape and pm.shape==m.shape:
                union=m|pm; inter=m&pm
                iou=float(np.count_nonzero(inter))/float(max(1,np.count_nonzero(union)))
                roi=union if np.count_nonzero(union)>=50 else np.ones_like(union,dtype=bool)
                mad=float(np.mean(np.abs(g.astype(np.float32)-pg.astype(np.float32))[roi]))
                oldps=np.asarray(cache.get('pose_sig'),dtype=np.float32)
                if oldps.shape==ps.shape:
                    good=(ps[:,2]>=0.20)&(oldps[:,2]>=0.20)&np.isfinite(ps[:,0])&np.isfinite(oldps[:,0])
                    pose_move=float(np.median(np.linalg.norm(ps[good,:2]-oldps[good,:2],axis=1))) if np.count_nonzero(good)>=4 else 999.0
                age=float(time.perf_counter()-float(cache.get('time',-1e9)))
                hit=bool(iou>=float(getattr(args,'micro_unfold_crease_stable_min_mask_iou',0.978)) and
                         mad<=float(getattr(args,'micro_unfold_crease_stable_max_mad',14.0)) and
                         pose_move<=float(getattr(args,'micro_unfold_crease_stable_max_pose_move_px',6.0)) and
                         age<=float(getattr(args,'micro_unfold_crease_stable_max_age_s',30.0)))
        if hit:
            score=np.asarray(state.get('score_map',np.zeros_like(mask_u8,dtype=np.float32)),dtype=np.float32)
            cur_s=np.asarray(state.get('sensitive_skeleton_mask',state.get('skeleton_mask',np.zeros_like(mask_u8))),dtype=np.uint8)
            cur_t=np.asarray(state.get('structural_skeleton_mask',np.zeros_like(mask_u8)),dtype=np.uint8)
            prev_s=np.asarray(cache.get('sensitive',np.zeros_like(cur_s)),dtype=np.uint8)
            prev_t=np.asarray(cache.get('structural',np.zeros_like(cur_t)),dtype=np.uint8)
            support_s=(score>=float(getattr(args,'micro_unfold_crease_stable_score_floor',0.115))) & (np.asarray(mask_u8)>0)
            support_t=(score>=float(getattr(args,'micro_unfold_crease_stable_struct_score_floor',0.145))) & (np.asarray(mask_u8)>0)
            sup_s=cv2.dilate((support_s.astype(np.uint8)*255),np.ones((5,5),np.uint8),iterations=1)>0
            sup_t=cv2.dilate((support_t.astype(np.uint8)*255),np.ones((5,5),np.uint8),iterations=1)>0
            stable_s=cv2.bitwise_or(cur_s, ((prev_s>0)&sup_s).astype(np.uint8)*255)
            stable_t=cv2.bitwise_or(cur_t, ((prev_t>0)&sup_t).astype(np.uint8)*255)
            state=dict(state)
            state['sensitive_skeleton_mask']=stable_s.astype(np.uint8)
            state['structural_skeleton_mask']=stable_t.astype(np.uint8)
            if cv2.countNonZero(stable_t)>=8:
                state['skeleton_mask']=stable_t.astype(np.uint8); state['selection_source']='STRUCTURAL_STABLE'
                state['available']=True
            elif cv2.countNonZero(stable_s)>=8:
                state['skeleton_mask']=stable_s.astype(np.uint8); state['selection_source']='SENSITIVE_STABLE_RESCUE'
                state['available']=True
            state['sensitive_skeleton_area_px']=int(cv2.countNonZero(stable_s))
            state['structural_skeleton_area_px']=int(cv2.countNonZero(stable_t))
            state['temporal_stable_hit']=True
        _E120_CREASE_TEMP_CACHE={
            'gray_small':g.copy(),'mask_small':m.copy(),'pose_sig':ps.copy(),'time':time.perf_counter(),
            'sensitive':np.asarray(state.get('sensitive_skeleton_mask',state.get('skeleton_mask',np.zeros_like(mask_u8))),dtype=np.uint8).copy(),
            'structural':np.asarray(state.get('structural_skeleton_mask',np.zeros_like(mask_u8)),dtype=np.uint8).copy(),
        }
        return state, hit, mad, iou, pose_move
    except Exception:
        return state, False, 999.0, 0.0, 999.0

def _e119_feature_cache_match(cache, frame_bgr, mask_u8, args):
    if not isinstance(cache, dict) or "features" not in cache:
        return False, 999.0, 0.0
    if not bool(getattr(args, "micro_unfold_depth_cache", True)):
        return False, 999.0, 0.0
    try:
        g, m = _e117_depth_cache_signature(frame_bgr, mask_u8)
        pg = np.asarray(cache.get("gray_small"), dtype=np.uint8)
        pm = np.asarray(cache.get("mask_small"), dtype=bool)
        if pg.shape != g.shape or pm.shape != m.shape:
            return False, 999.0, 0.0
        union = m | pm
        inter = m & pm
        iou = float(np.count_nonzero(inter)) / float(max(1, np.count_nonzero(union)))
        roi = union if np.count_nonzero(union) >= 50 else np.ones_like(union, dtype=bool)
        mad = float(np.mean(np.abs(g.astype(np.float32)-pg.astype(np.float32))[roi]))
        age = float(time.perf_counter()-float(cache.get("time",-1e9)))
        # Features are a little more scene-sensitive than the raw base, but still
        # tolerate the real ELP noise that made E118's first cache ineffective.
        max_mad = min(float(getattr(args,"micro_unfold_depth_cache_max_mad",5.0)), 4.5)
        min_iou = max(float(getattr(args,"micro_unfold_depth_cache_min_mask_iou",0.94)), 0.945)
        max_age = float(getattr(args,"micro_unfold_depth_cache_max_age_s",180.0))
        return bool(mad<=max_mad and iou>=min_iou and age<=max_age), mad, iou
    except Exception:
        return False, 999.0, 0.0


def _e117_depth_cache_signature(frame_bgr, mask_u8):
    frame = np.asarray(frame_bgr, dtype=np.uint8)
    mask = ((np.asarray(mask_u8, dtype=np.uint8) > 0).astype(np.uint8) * 255)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_s = cv2.resize(gray, (160, 90), interpolation=cv2.INTER_AREA)
    mask_s = cv2.resize(mask, (160, 90), interpolation=cv2.INTER_NEAREST) > 0
    return gray_s, mask_s

def _e117_depth_cache_match(cache, frame_bgr, mask_u8, args):
    if not isinstance(cache, dict) or "base" not in cache:
        return False, 999.0, 0.0
    if not bool(getattr(args, "micro_unfold_depth_cache", True)):
        return False, 999.0, 0.0
    try:
        g, m = _e117_depth_cache_signature(frame_bgr, mask_u8)
        pg = np.asarray(cache.get("gray_small"), dtype=np.uint8)
        pm = np.asarray(cache.get("mask_small"), dtype=bool)
        if pg.shape != g.shape or pm.shape != m.shape:
            return False, 999.0, 0.0
        union = m | pm
        inter = m & pm
        iou = float(np.count_nonzero(inter)) / float(max(1, np.count_nonzero(union)))
        roi = union if np.count_nonzero(union) >= 50 else np.ones_like(union, dtype=bool)
        mad = float(np.mean(np.abs(g.astype(np.float32) - pg.astype(np.float32))[roi]))
        age = float(time.perf_counter() - float(cache.get("time", -1e9)))
        max_mad = float(getattr(args, "micro_unfold_depth_cache_max_mad", 1.6))
        min_iou = float(getattr(args, "micro_unfold_depth_cache_min_mask_iou", 0.945))
        max_age = float(getattr(args, "micro_unfold_depth_cache_max_age_s", 30.0))
        return bool(mad <= max_mad and iou >= min_iou and age <= max_age), mad, iou
    except Exception:
        return False, 999.0, 0.0

# E137: the legacy UniDepth TOP3 geometry implementation was removed here.
# The RGB/Pose compatibility implementation is defined later in this helper source.

def _fix53_px_to_board(H_value, px):
    return np.asarray(_fix5_grasp_pixel_to_board(H_value, px), dtype=np.float64)


def _fix53_board_inside(cfg, p, margin=25.0):
    x, y = float(p[0]), float(p[1])
    return bool(
        float(cfg.board_x_min) + margin <= x <= float(cfg.board_x_max) - margin
        and float(cfg.board_y_min) + margin <= y <= float(cfg.board_y_max) - margin
    )


def _e92_select_active_point(depth_state, garment_mask_u8):
    """Select ACTIVE in two stages: raised-core gate first, body-side choice second."""
    mask = np.asarray(garment_mask_u8, dtype=np.uint8) > 0
    labels = np.asarray(depth_state.get("visual_top2_labels", depth_state.get("labels")), dtype=np.uint8)
    score = np.asarray(
        depth_state.get("visual_top2_score", np.maximum(depth_state.get("prominence_mm"), 0.0)),
        dtype=np.float32,
    )
    region = (labels == 1) & mask
    if int(np.count_nonzero(region)) < 8:
        raise RuntimeError("E92 selected FUSED FOLD is too small for ACTIVE")
    mm = cv2.moments((mask.astype(np.uint8) * 255), binaryImage=True)
    if abs(float(mm.get("m00", 0.0))) <= 1e-6:
        raise RuntimeError("E92 garment centroid unavailable")
    gc = np.asarray([float(mm["m10"]/mm["m00"]), float(mm["m01"]/mm["m00"])], dtype=np.float64)
    args_micro = globals().get("_E92_MICRO_ARGS")
    percentile = float(getattr(args_micro, "micro_unfold_active_core_percentile", 72.0)) if args_micro is not None else 72.0
    percentile = float(np.clip(percentile, 55.0, 90.0))

    vals0 = score[region]
    vals0 = vals0[np.isfinite(vals0)]
    core = region.copy()
    gate = 0.0
    if vals0.size >= 8:
        gate = float(np.percentile(vals0, percentile))
        core = region & np.isfinite(score) & (score >= gate)
        if int(np.count_nonzero(core)) < 6:
            gate = float(np.percentile(vals0, max(55.0, percentile - 10.0)))
            core = region & np.isfinite(score) & (score >= gate)

    inside = cv2.distanceTransform((region.astype(np.uint8) * 255), cv2.DIST_L2, 5).astype(np.float32)
    garment_inside = cv2.distanceTransform((mask.astype(np.uint8) * 255), cv2.DIST_L2, 5).astype(np.float32)
    # E97: the executable ACTIVE should live on the body-side interior of the
    # raised fold, not on the outer garment silhouette.  Enforce a useful
    # garment-edge clearance first and relax it only when the fused fold is too
    # thin to provide enough grasp pixels.
    edge_clearance = float(getattr(args_micro, "micro_unfold_active_mask_edge_clearance_px", 16.0)) if args_micro is not None else 16.0
    edge_clearance = float(np.clip(edge_clearance, 4.0, 40.0))
    graspable = core & (inside >= 3.0) & (garment_inside >= edge_clearance)
    if int(np.count_nonzero(graspable)) < 4:
        graspable = core & (inside >= 2.0) & (garment_inside >= max(8.0, 0.65 * edge_clearance))
    if int(np.count_nonzero(graspable)) < 4:
        graspable = core & (inside >= 1.5) & (garment_inside >= 5.0)
    if int(np.count_nonzero(graspable)) < 4:
        graspable = core if int(np.count_nonzero(core)) >= 4 else region

    # E96 MANUAL EXACT-CANDIDATE MODE.  When the operator presses 1/2/3, the
    # displayed C1/C2/C3 seed marker is the requested ACTIVE grasp location.
    # We therefore do NOT run the normal body-side / crease-standoff selector.
    # The marker is snapped only to the nearest safe pixel of that selected
    # strict seed (or, if needed, the selected fused fold).
    if bool(depth_state.get("manual_override", False)) and depth_state.get("manual_active_target_px") is not None:
        target = np.asarray(depth_state.get("manual_active_target_px"), dtype=np.float64).reshape(2)
        strict_seed = np.asarray(depth_state.get("strict_seed_labels", np.zeros(mask.shape, dtype=np.uint8)), dtype=np.uint8) > 0
        garment_inside = cv2.distanceTransform((mask.astype(np.uint8) * 255), cv2.DIST_L2, 5).astype(np.float32)
        manual_domain = strict_seed & mask & (garment_inside >= 3.0)
        if int(np.count_nonzero(manual_domain)) < 3:
            manual_domain = strict_seed & mask
        if int(np.count_nonzero(manual_domain)) < 3:
            manual_domain = region & mask & (garment_inside >= 3.0)
        if int(np.count_nonzero(manual_domain)) < 3:
            manual_domain = region & mask
        my, mx = np.where(manual_domain)
        if mx.size == 0:
            raise RuntimeError("E96 manual C candidate has no safe ACTIVE pixel")
        dist2 = (mx.astype(np.float64)-target[0])**2 + (my.astype(np.float64)-target[1])**2
        k = int(np.argmin(dist2))
        active_px = np.asarray([float(mx[k]), float(my[k])], dtype=np.float64)
        snap = float(np.sqrt(float(dist2[k])))
        return active_px, gc, {
            "active_core_percentile": float(percentile),
            "active_core_gate": float(gate),
            "active_core_area_px": int(np.count_nonzero(core)),
            "active_graspable_area_px": int(np.count_nonzero(manual_domain)),
            "active_center_min_distance_px": float(np.linalg.norm(active_px-gc)),
            "active_selected_score": float(score[int(round(active_px[1])), int(round(active_px[0]))]) if np.isfinite(score[int(round(active_px[1])), int(round(active_px[0]))]) else 0.0,
            "active_crease_used": False,
            "active_crease_distance_px": 999.0,
            "active_crease_target_px": float(crease_target) if 'crease_target' in locals() else 28.0,
            "manual_exact_candidate": True,
            "manual_active_target_px": target.tolist(),
            "manual_active_snap_distance_px": float(snap),
        }

    yy, xx = np.where(graspable)
    if xx.size == 0:
        raise RuntimeError("E92 ACTIVE raised-core candidate set is empty")
    d_center = np.hypot(xx.astype(np.float64)-gc[0], yy.astype(np.float64)-gc[1])
    min_dist = float(np.min(d_center))
    # Height has already been enforced by the raised-core gate. Only now use the
    # user's body-side preference. Within a small near-centre band, choose the
    # highest score as the tie-breaker instead of mixing pixels and unitless depth.
    # E97: HOLD is intentionally toward garment centre, so centre-side ACTIVE
    # candidates are also the candidates that minimize HOLD<->ACTIVE distance.
    # Keep a modest band (rather than a single nearest pixel) so UniDepth/crease
    # can still choose a robust grasp within the FUSED FOLD.
    body_band_px = float(getattr(args_micro, "micro_unfold_active_body_band_px", 24.0)) if args_micro is not None else 24.0
    body_band_px = float(np.clip(body_band_px, 8.0, 60.0))
    near = d_center <= (min_dist + body_band_px)
    near_indices = np.flatnonzero(near)
    if near_indices.size == 0:
        near_indices = np.arange(xx.size)
    near_scores = score[yy[near_indices], xx[near_indices]].astype(np.float64)
    near_scores = np.where(np.isfinite(near_scores), near_scores, -1e9)

    # E94: if a reliable internal crease exists, keep the grasp on the raised
    # side but do not pinch the thin crease itself.  Prefer a configurable
    # standoff (~28 px) from the detected line, then the body-side location,
    # with depth score only as the tie-breaker because the raised-core gate has
    # already enforced height.
    crease_used = False
    active_crease_distance = 999.0
    crease_target = float(np.clip(float(getattr(args_micro, "micro_unfold_crease_active_target_px", 14.0)) if args_micro is not None else 14.0, 6.0, 50.0))
    crease_min = float(np.clip(float(getattr(args_micro, "micro_unfold_crease_active_min_px", 6.0)) if args_micro is not None else 6.0, 3.0, crease_target))
    crease_max = float(max(crease_target + 10.0, float(getattr(args_micro, "micro_unfold_crease_active_max_px", 45.0)) if args_micro is not None else 45.0))
    crease_skel = np.asarray(depth_state.get("fold_crease_skeleton", np.zeros(mask.shape, dtype=np.uint8)), dtype=np.uint8)
    candidate_pool = near_indices
    if crease_skel.shape == mask.shape and cv2.countNonZero(crease_skel) >= 8:
        crease_dist_map = cv2.distanceTransform(((crease_skel == 0).astype(np.uint8) * 255), cv2.DIST_L2, 5).astype(np.float32)
        all_crease_dist = crease_dist_map[yy, xx].astype(np.float64)
        crease_ok = (all_crease_dist >= crease_min) & (all_crease_dist <= crease_max)
        # E97: crease evidence may refine ACTIVE only inside the body-side band.
        # This prevents a visually strong outer-edge crease from pulling ACTIVE
        # away from HOLD and onto the garment silhouette.
        crease_idx = np.flatnonzero(crease_ok & near)
        if crease_idx.size < 4:
            widened_body = d_center <= (min_dist + 1.5 * body_band_px)
            crease_idx = np.flatnonzero(crease_ok & widened_body)
        if crease_idx.size >= 4:
            cdist = all_crease_dist[crease_idx]
            ccenter = d_center[crease_idx]
            cscores = score[yy[crease_idx], xx[crease_idx]].astype(np.float64)
            cscores = np.where(np.isfinite(cscores), cscores, -1e9)
            if np.max(cscores) > np.min(cscores) + 1e-9:
                sh = (cscores - np.min(cscores)) / (np.max(cscores) - np.min(cscores))
            else:
                sh = np.zeros(cscores.shape, dtype=np.float64)
            dc = (ccenter - np.min(ccenter)) / max(1.0, float(np.ptp(ccenter)))
            objective = 0.35 * (np.abs(cdist - crease_target) / max(1.0, crease_target)) + 0.50 * dc - 0.15 * sh
            idx_best = int(crease_idx[int(np.argmin(objective))])
            crease_used = True
            active_crease_distance = float(all_crease_dist[idx_best])
        else:
            local_best = int(np.argmax(near_scores))
            idx_best = int(near_indices[local_best])
            active_crease_distance = float(crease_dist_map[int(yy[idx_best]), int(xx[idx_best])])
    else:
        local_best = int(np.argmax(near_scores))
        idx_best = int(near_indices[local_best])
    active_px = np.asarray([float(xx[idx_best]), float(yy[idx_best])], dtype=np.float64)
    return active_px, gc, {
        "active_core_percentile": float(percentile),
        "active_core_gate": float(gate),
        "active_core_area_px": int(np.count_nonzero(core)),
        "active_graspable_area_px": int(np.count_nonzero(graspable)),
        "active_center_min_distance_px": float(min_dist),
        "active_body_band_px": float(body_band_px),
        "active_mask_edge_clearance_px": float(garment_inside[int(round(active_px[1])), int(round(active_px[0]))]),
        "active_selected_score": float(score[int(round(active_px[1])), int(round(active_px[0]))]),
        "active_crease_used": bool(crease_used),
        "active_crease_distance_px": float(active_crease_distance),
        "active_crease_target_px": float(crease_target),
    }


def _fix92_make_diagnostic_plan(depth_state, H_value):
    """Publish all successful E92 perception evidence before HOLD planning."""
    # E122 branch classification is perception, not robot planning. Compute it
    # here as well so even a later SAME-FOLD/robot-safety block leaves the useful
    # gray/cyan boundary diagnostics visible to the operator.
    try:
        _e122_args=globals().get('_E92_MICRO_ARGS')
        _e122_mask=np.asarray(depth_state.get('garment_mask'),dtype=np.uint8)
        _e122_pool=_e122_build_scored_crease_pool(depth_state,_e122_mask,_e122_args,H_value=H_value)
        _e153_publish_e122_pool(depth_state,_e122_pool,_e122_mask)
        print('[E153-ZEROCOPY-PUBLISH] diagnostic E122 maps published by immutable reference')
    except Exception as _e122_diag_exc:
        print(f"[E122-DIAGNOSTIC-BRANCH-WARN] {_e122_diag_exc!r}")
    plan = {
        "state": "DIAGNOSTIC",
        "method": "E92_TOP3_HYPOTHESIS_DIAGNOSTIC_ONLY",
        "H": np.asarray(H_value, dtype=np.float64).tolist(),
        "hold_px": None,
        "active_px": None,
        "hold_arm": "",
        "active_arm": "",
        "diagnostic_reason": "HOLD_NOT_PLANNED_YET",
        "garment_mask": np.asarray(depth_state.get("garment_mask"), dtype=np.uint8).copy(),
        "visual_top2_labels": np.asarray(depth_state.get("visual_top2_labels", depth_state.get("labels")), dtype=np.uint8).copy(),
        "strict_seed_labels": np.asarray(depth_state.get("strict_seed_labels", np.zeros_like(depth_state.get("garment_mask"))), dtype=np.uint8).copy(),
        "strict_candidate_labels": np.asarray(depth_state.get("strict_candidate_labels", np.zeros_like(depth_state.get("garment_mask"))), dtype=np.uint8).copy(),
        "mask_fold_candidate": np.asarray(depth_state.get("mask_fold_candidate", np.zeros_like(depth_state.get("garment_mask"))), dtype=np.uint8).copy(),
        "candidate_diagnostics": _training_json_safe(depth_state.get("candidate_diagnostics", [])),
        "selected_candidate_id": str(depth_state.get("selected_candidate_id", "NONE")),
        "manual_override": bool(depth_state.get("manual_override", False)),
        "manual_override_candidate_id": str(depth_state.get("manual_override_candidate_id", "")),
        "hypothesis_score_gap": float(depth_state.get("hypothesis_score_gap", 9.999)),
        "hypothesis_ambiguous": bool(depth_state.get("hypothesis_ambiguous", False)),
        "fusion_mode": str(depth_state.get("fusion_mode", "UNKNOWN")),
        "fold_crease_mask": np.asarray(depth_state.get("fold_crease_mask", np.zeros_like(depth_state.get("garment_mask"))), dtype=np.uint8).copy(),
        "fold_crease_skeleton": np.asarray(depth_state.get("e122_fold_preview_skeleton", depth_state.get("fold_crease_skeleton", np.zeros_like(depth_state.get("garment_mask")))), dtype=np.uint8).copy(),
        "e122_rejected_crease_skeleton": np.asarray(depth_state.get("e122_fold_rejected_skeleton", np.zeros_like(depth_state.get("garment_mask"))), dtype=np.uint8).copy(),
        "e122_fold_branch_info": _training_json_safe(depth_state.get("e122_fold_branch_info", [])),
        "fold_crease_info": _training_json_safe(depth_state.get("fold_crease_info", {})),
        "baseline_depth": depth_state,
    }
    try:
        _e129_args=globals().get('_E92_MICRO_ARGS')
        _e129_fold=_e125_build_primary_fold_region(depth_state,plan["garment_mask"],_e129_args)
        _e129_region=np.asarray(_e129_fold.get('region_mask',np.zeros_like(plan["garment_mask"])),dtype=np.uint8)
        _pose_diag=bool(_e129_fold.get('authoritative',False) and str(_e129_fold.get('source',''))=='POSE_INTRUSION')
        if _pose_diag:
            _pcands=_e129_rank_pose_intrusion_active_pixels(depth_state,plan["garment_mask"],_e129_region,_e129_args)
        else:
            _pcands=[]
        if _pcands:
            active_px=np.asarray(_pcands[0]['px'],dtype=np.float64)
            mm=cv2.moments(np.asarray(plan["garment_mask"],dtype=np.uint8),binaryImage=True)
            gc=np.asarray([float(mm['m10']/max(1e-6,mm['m00'])),float(mm['m01']/max(1e-6,mm['m00']))],dtype=np.float64)
            plan["active_px"]=active_px.tolist()
            plan["garment_center_px"]=gc.tolist()
            plan["active_selection_mode"]='E129_POSE_INTRUSION_UNIDEPTH_ACTIVE_DIAGNOSTIC'
            plan["active_candidate_source"]='POSE_INTRUSION'
            plan["e129_active_pose_depth_authority"]=float(_pcands[0].get('score',0.0))
            plan["e129_active_pose_region_raised_mm"]=float(_pcands[0].get('raised_mm',0.0))
            plan["e125_primary_fold_region"]=_e129_region.copy()
            plan["e125_primary_fold_boundary"]=np.asarray(_e129_fold.get('boundary_mask',np.zeros_like(_e129_region)),dtype=np.uint8).copy()
            plan["e125_primary_fold_source"]='POSE_INTRUSION'
            plan["e125_primary_fold_confidence"]=float(_e129_fold.get('confidence',0.0))
            plan["e125_primary_fold_group"]=str(_e129_fold.get('group','NONE'))
            # Also publish to depth_state so the overlay remains coherent if final HOLD fails.
            depth_state['e125_primary_fold_region_mask']=_e129_region.copy()
            depth_state['e125_primary_fold_boundary_mask']=np.asarray(_e129_fold.get('boundary_mask',np.zeros_like(_e129_region)),dtype=np.uint8).copy()
            depth_state['e125_primary_fold_source']='POSE_INTRUSION'
            depth_state['e125_primary_fold_confidence']=float(_e129_fold.get('confidence',0.0))
            depth_state['e125_primary_fold_group']=str(_e129_fold.get('group','NONE'))
            print(f"[E129-DIAGNOSTIC-ACTIVE] POSE_INTRUSION->UNIDEPTH px=({active_px[0]:.1f},{active_px[1]:.1f}) authority={float(_pcands[0].get('score',0.0)):.3f} raised={float(_pcands[0].get('raised_mm',0.0)):+.2f}mm FUSED=NOT_USED")
        else:
            active_px, gc, active_meta = _e92_select_active_point(depth_state, plan["garment_mask"])
            plan["active_px"] = active_px.tolist()
            plan["garment_center_px"] = gc.tolist()
            plan.update(active_meta)
        if bool(plan.get("hypothesis_ambiguous", False)):
            plan["diagnostic_reason"] = "FOLD_AMBIGUOUS"
    except Exception as exc:
        plan["diagnostic_active_error"] = repr(exc)
    return plan

def _e93_override_depth_state(depth_state, donor_rank, args=None):
    """Force C1/C2/C3 using only the most recent I result.

    No camera, segmentation, or UniDepth inference is performed here.  The
    stored strict candidate seed is re-fused against the already-computed
    relief/prominence/MASK maps, then downstream ACTIVE/HOLD planning is run
    normally.  Manual choice removes only the E92 *ranking ambiguity* block;
    it does not bypass fold-size, board, workspace, pair-clearance, HOLD, or
    pull-endpoint safety gates.
    """
    if not isinstance(depth_state, dict) or not bool(depth_state.get("available", False)):
        raise RuntimeError("E93 manual selection requires a valid most-recent I result")
    rank = int(donor_rank)
    if rank not in (1, 2, 3):
        raise RuntimeError(f"E93 manual candidate rank must be 1/2/3, got {rank}")

    diagnostics_src = list(depth_state.get("candidate_diagnostics", []) or [])
    chosen_diag = None
    for item in diagnostics_src:
        try:
            if int(item.get("donor_rank", 0)) == rank:
                chosen_diag = dict(item)
                break
        except Exception:
            continue
    if chosen_diag is None:
        available_ids = []
        for item in diagnostics_src:
            try:
                available_ids.append(f"C{int(item.get('donor_rank', 0))}")
            except Exception:
                pass
        raise RuntimeError(
            f"E93 C{rank} unavailable in the last I judgment; retained={available_ids or ['NONE']}"
        )

    garment = ((np.asarray(depth_state.get("garment_mask"), dtype=np.uint8) > 0).astype(np.uint8) * 255)
    relief = np.asarray(depth_state.get("local_relief_mm"), dtype=np.float32)
    prom = np.asarray(depth_state.get("prominence_mm"), dtype=np.float32)
    strict_all = np.asarray(depth_state.get("strict_candidate_labels"), dtype=np.uint8)
    if garment.shape != relief.shape or garment.shape != prom.shape or garment.shape != strict_all.shape:
        raise RuntimeError("E93 stored I geometry mismatch; press I again")

    seed = (strict_all == rank) & (garment > 0)
    seed_area = int(np.count_nonzero(seed))
    if seed_area < 8:
        raise RuntimeError(f"E93 C{rank} strict seed is unavailable/too small ({seed_area}px); press I again")

    args_micro = args if args is not None else globals().get("_E92_MICRO_ARGS")
    fold_region, fusion_info, mask_fold = _fix90_fuse_mask_unidepth(
        relief, prom, garment, seed, args=args_micro
    )
    fold_region = np.asarray(fold_region, dtype=bool) & (garment > 0)
    fold_area = int(np.count_nonzero(fold_region))
    if fold_area < 8:
        raise RuntimeError(f"E93 C{rank} fusion produced no usable fold region ({fold_area}px)")

    labels = np.zeros(garment.shape, dtype=np.uint8)
    labels[fold_region] = 1
    fold_score = _fix86_top1_point_score(relief, prom, fold_region)

    diagnostics = []
    for item in diagnostics_src:
        row = dict(item)
        try:
            row_rank = int(row.get("donor_rank", 0))
        except Exception:
            row_rank = 0
        row["selected"] = bool(row_rank == rank)
        row["manual_selected"] = bool(row_rank == rank)
        diagnostics.append(row)

    selected_score = float(chosen_diag.get("final_score", depth_state.get("hypothesis_score", 0.0)))
    fusion_mode = str((fusion_info or {}).get("fusion_mode", "UNKNOWN"))
    out = dict(depth_state)
    out.update({
        "labels": labels.copy(),
        "strict_seed_labels": (seed.astype(np.uint8) * 255),
        "mask_fold_candidate": (np.asarray(mask_fold, dtype=bool).astype(np.uint8) * 255),
        "fusion_mode": fusion_mode,
        "candidate_diagnostics": _training_json_safe(diagnostics),
        "selected_candidate_id": f"C{rank}",
        "selected_donor_rank": int(rank),
        "hypothesis_score": selected_score,
        "hypothesis_score_gap": 9.999,
        "hypothesis_centroid_gap_px": 0.0,
        "hypothesis_ambiguous": False,
        "visual_top2_labels": labels.copy(),
        "visual_top2_score": np.asarray(fold_score, dtype=np.float32).copy(),
        "visual_top2_count": 1,
        "visual_top2_info": [{
            "rank": 1,
            "donor_rank": int(rank),
            "candidate_id": f"C{rank}",
            "centroid_px": list(chosen_diag.get("fold_centroid_px", chosen_diag.get("seed_centroid_px", [0.0, 0.0]))),
            "seed_centroid_px": list(chosen_diag.get("seed_centroid_px", [0.0, 0.0])),
            "area_px": int(fold_area),
            "strict_seed_area_px": int(seed_area),
            "hypothesis_score": float(selected_score),
            "relief_p95_mm": float(chosen_diag.get("relief_p95_mm", 0.0)),
            "prominence_p90_mm": float(chosen_diag.get("prominence_p90_mm", 0.0)),
            "fusion": _training_json_safe(fusion_info),
            "manual_override": True,
        }],
        "fold_region_expansion": _training_json_safe(fusion_info),
        "manual_override": True,
        "manual_override_candidate_id": f"C{rank}",
        "manual_override_rank": int(rank),
        # E96: the numbered C1/C2/C3 marker itself becomes the manual ACTIVE anchor.
        # ACTIVE is snapped only a few pixels, if necessary, to the nearest safe
        # pixel of the selected strict seed; no body-side/crease re-ranking is
        # allowed to move it to another part of the fused fold.
        "manual_active_target_px": list(chosen_diag.get("seed_centroid_px", [0.0, 0.0])),
        "manual_override_source": "MOST_RECENT_I_FROZEN_DEPTH_STATE_EXACT_CANDIDATE_ACTIVE",
    })
    print(
        f"[E93-MANUAL-CANDIDATE] selected=C{rank} from frozen I result | "
        f"seed={seed_area}px fold={fold_area}px mode={fusion_mode} score={selected_score:.3f} | "
        "ACTIVE=nearest-safe-pixel-to-displayed-C-marker | NO new camera/SEG/UniDepth inference"
    )
    return out



def _e98_local_mm_per_px(H_value, px):
    p = np.asarray(px, dtype=np.float64).reshape(2)
    probes = np.asarray([p, p + [6.0, 0.0], p + [0.0, 6.0]], dtype=np.float64)
    b = _fix12_transform_points(H_value, probes)
    sx = float(np.linalg.norm(b[1]-b[0])) / 6.0
    sy = float(np.linalg.norm(b[2]-b[0])) / 6.0
    vals = [v for v in (sx, sy) if np.isfinite(v) and v > 1e-6]
    return float(np.median(vals)) if vals else 1.0


def _e98_outer_contour(mask_u8):
    m = ((np.asarray(mask_u8, dtype=np.uint8) > 0).astype(np.uint8) * 255)
    cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cs:
        raise RuntimeError('E98 garment outer contour unavailable')
    return max(cs, key=lambda c: cv2.contourArea(c)).reshape(-1,2).astype(np.float64)


def _e98_try_crease_partition(mask_u8, crease_u8, pose_body_core_u8=None):
    """Use a dominant crease + outer MASK to split garment into BODY and FLAP.

    The detected crease is extended from both PCA endpoints to their nearest
    garment outer-boundary points. The resulting barrier is removed from the
    garment mask. The component containing the garment centroid is BODY; the
    strongest plausible opposite component is the crease-to-MASK FLAP region.
    """
    mask = ((np.asarray(mask_u8, dtype=np.uint8) > 0).astype(np.uint8) * 255)
    crease = ((np.asarray(crease_u8, dtype=np.uint8) > 0).astype(np.uint8) * 255)
    if cv2.countNonZero(mask) < 500:
        return None
    # E130 does not require a pre-qualified Fold Crease to authorize planning.
    # The directional planner can construct a virtual BODY/fold boundary when
    # pose/depth supplies direction but the crease detector is weak.
    outer = _e98_outer_contour(mask)
    mm = cv2.moments(mask, binaryImage=True)
    gc = np.asarray([float(mm['m10']/max(1e-6,mm['m00'])), float(mm['m01']/max(1e-6,mm['m00']))], dtype=np.float64)
    # Connect small gaps first, then rank connected crease components by span.
    work = cv2.morphologyEx(crease, cv2.MORPH_CLOSE, np.ones((7,7),np.uint8), iterations=1)
    n, labs, stats, _ = cv2.connectedComponentsWithStats((work>0).astype(np.uint8), 8)
    comps=[]
    for lab in range(1,n):
        yy,xx=np.where(labs==lab)
        if xx.size < 10: continue
        pts=np.column_stack([xx,yy]).astype(np.float64)
        ctr=pts.mean(axis=0)
        cov=np.cov((pts-ctr).T) if pts.shape[0]>=3 else np.eye(2)
        vals,vecs=np.linalg.eigh(cov); axis=vecs[:,int(np.argmax(vals))]
        proj=(pts-ctr)@axis
        p0=pts[int(np.argmin(proj))]; p1=pts[int(np.argmax(proj))]
        span=float(np.linalg.norm(p1-p0))
        if span < 28.0: continue
        comps.append((span,lab,p0,p1,pts))
    comps.sort(key=lambda q:q[0], reverse=True)
    garment_area=float(cv2.countNonZero(mask))
    pose_core=np.zeros_like(mask,dtype=np.uint8) if pose_body_core_u8 is None else ((np.asarray(pose_body_core_u8,dtype=np.uint8)>0).astype(np.uint8)*255)
    if pose_core.shape!=mask.shape: pose_core=cv2.resize(pose_core,(mask.shape[1],mask.shape[0]),interpolation=cv2.INTER_NEAREST)
    pose_core=cv2.bitwise_and(pose_core,mask)
    pose_core_area=float(cv2.countNonZero(pose_core))
    best=None
    for span,lab,p0,p1,pts in comps[:8]:
        nearest=[]
        for ep in (p0,p1):
            d2=np.sum((outer-ep[None,:])**2,axis=1)
            j=int(np.argmin(d2)); nearest.append((outer[j].copy(), float(np.sqrt(d2[j]))))
        if max(nearest[0][1], nearest[1][1]) > 135.0:
            continue
        if float(np.linalg.norm(nearest[0][0]-nearest[1][0])) < 22.0:
            continue
        for thickness in (5,7,9,11):
            barrier=np.zeros_like(mask)
            barrier[labs==lab]=255
            cv2.line(barrier, tuple(np.rint(p0).astype(int)), tuple(np.rint(nearest[0][0]).astype(int)), 255, thickness, cv2.LINE_AA)
            cv2.line(barrier, tuple(np.rint(p1).astype(int)), tuple(np.rint(nearest[1][0]).astype(int)), 255, thickness, cv2.LINE_AA)
            barrier=cv2.dilate(barrier,np.ones((thickness,thickness),np.uint8),iterations=1)
            cut=((mask>0)&(barrier==0)).astype(np.uint8)
            nc,ll,ss,_=cv2.connectedComponentsWithStats(cut,8)
            if nc < 3: continue
            gx=int(np.clip(round(gc[0]),0,mask.shape[1]-1)); gy=int(np.clip(round(gc[1]),0,mask.shape[0]-1))
            body_assignment_mode='MASK_CENTROID'
            body_lab=0
            if pose_core_area>=80.0:
                ids=list(range(1,nc))
                overlaps=[int(np.count_nonzero((ll==lid)&(pose_core>0))) for lid in ids]
                if overlaps and max(overlaps)>=20:
                    body_lab=int(ids[int(np.argmax(overlaps))]); body_assignment_mode='POSE_TORSO_CORE'
            if body_lab<=0:
                body_lab=int(ll[gy,gx])
            if body_lab<=0:
                ids=list(range(1,nc)); body_lab=max(ids,key=lambda z:int(ss[z,cv2.CC_STAT_AREA]))
            body=(ll==body_lab)
            body_core_overlap=float(np.count_nonzero(body&(pose_core>0)))/max(1.0,pose_core_area) if pose_core_area>0 else 0.0
            flap_candidates=[]
            for lid in range(1,nc):
                if lid==body_lab: continue
                area=int(ss[lid,cv2.CC_STAT_AREA]); ratio=area/max(1.0,garment_area)
                if ratio < 0.025 or ratio > 0.58: continue
                reg=(ll==lid)
                # Must touch the crease barrier; otherwise it is merely a mask island.
                touch=int(np.count_nonzero(reg & (cv2.dilate(barrier,np.ones((9,9),np.uint8),iterations=1)>0)))
                if touch < 8: continue
                flap_core=float(np.count_nonzero(reg&(pose_core>0)))/max(1.0,pose_core_area) if pose_core_area>0 else 0.0
                flap_candidates.append((area,ratio,touch,reg,lid,flap_core))
            if not flap_candidates: continue
            # A true FLAP should not own the pose-defined torso core.
            flap_candidates.sort(key=lambda q:(q[0]-2200.0*q[5],q[2]),reverse=True)
            area,ratio,touch,flap,lid,flap_core_overlap=flap_candidates[0]
            # Prefer long crease + plausible flap, and reward BODY agreement with pose torso core.
            quality=float(span) + 180.0*min(ratio,0.35) - 80.0*max(0.0,ratio-0.45) + 220.0*body_core_overlap - 260.0*flap_core_overlap
            item={
                'flap_region':(flap.astype(np.uint8)*255),
                'body_region':(body.astype(np.uint8)*255),
                'selected_crease':((labs==lab).astype(np.uint8)*255),
                'barrier':barrier,
                'crease_span_px':float(span),'flap_ratio':float(ratio),
                'endpoint_boundary_dist_px':[float(nearest[0][1]),float(nearest[1][1])],
                'garment_center_px':gc.tolist(),'quality':quality,'body_assignment_mode':body_assignment_mode,'body_pose_core_overlap':float(body_core_overlap),'flap_pose_core_overlap':float(flap_core_overlap),
            }
            if best is None or quality>best['quality']:
                best=item
    return best


def _e109_select_crease_outer_lobe_pair(depth_state, mask_u8, H_value, args, config=None, cfg=None):
    """E109 AUTO selector: Fold Crease + outer-contour lobe geometry is primary.

    Design contract
    ---------------
    1) FOLD CREASE component quality and the coarse shape enclosed by that crease
       plus the garment outer contour dominate candidate ranking.
    2) The coarse lobe is used ONLY for side ownership (ACTIVE side vs BODY/HOLD
       side). It never hard-clips the 30/40 mm grasp points.
    3) FUSED FOLD is a weak 1% reference/tie-breaker.
    4) Pose seam veto is already applied upstream. When pose is unavailable, a
       bilateral RGB/depth similarity test penalizes neckline/sewing-seam-like
       crease components, especially tiny near-boundary lobes.
    5) Every retained pair is prechecked for board/reach/path/pull feasibility.
    """
    mask = ((np.asarray(mask_u8, dtype=np.uint8) > 0).astype(np.uint8) * 255)
    structural_crease = ((np.asarray(depth_state.get('fold_crease_structural_skeleton', np.zeros_like(mask)), dtype=np.uint8) > 0).astype(np.uint8) * 255)
    sensitive_crease = ((np.asarray(depth_state.get('fold_crease_skeleton', np.zeros_like(mask)), dtype=np.uint8) > 0).astype(np.uint8) * 255)
    pose_fold_mask = ((np.asarray(depth_state.get('pose_fold_intrusion_mask', np.zeros_like(mask)), dtype=np.uint8) > 0).astype(np.uint8) * 255)
    pose_fold_bool = pose_fold_mask > 0
    pose_fold_available = bool(depth_state.get('pose_fold_intrusion_available', False)) and cv2.countNonZero(pose_fold_mask) >= 8
    # E125: identify the folded part first.  Pose contour intrusion has first
    # authority; independent UniDepth raised regions are second; only then does
    # crease+contour geometry become the fallback.
    fold_region_info = _e125_build_primary_fold_region(depth_state, mask, args)
    primary_fold_region = ((np.asarray(fold_region_info.get('region_mask', np.zeros_like(mask)), dtype=np.uint8) > 0).astype(np.uint8) * 255)
    primary_fold_bool = primary_fold_region > 0
    primary_fold_boundary = ((np.asarray(fold_region_info.get('boundary_mask', np.zeros_like(mask)), dtype=np.uint8) > 0).astype(np.uint8) * 255)
    primary_fold_boundary_bool = primary_fold_boundary > 0
    primary_fold_source = str(fold_region_info.get('source', 'CREASE_CONTOUR_FALLBACK'))
    primary_fold_conf = float(fold_region_info.get('confidence', 0.0) or 0.0)
    primary_fold_group = str(fold_region_info.get('group', 'NONE'))
    primary_fold_authoritative = bool(fold_region_info.get('authoritative', False))
    depth_state['e125_primary_fold_region_mask'] = primary_fold_region.copy()
    depth_state['e125_primary_fold_boundary_mask'] = primary_fold_boundary.copy()
    depth_state['e125_primary_fold_source'] = primary_fold_source
    depth_state['e125_primary_fold_confidence'] = float(primary_fold_conf)
    depth_state['e125_primary_fold_group'] = primary_fold_group
    depth_state['e125_primary_fold_info'] = _training_json_safe({k:v for k,v in fold_region_info.items() if k not in ('region_mask','boundary_mask')})
    print(
        f"[E126-FOLD-REGION] source={primary_fold_source} group={primary_fold_group} "
        f"conf={primary_fold_conf:.3f} area={cv2.countNonZero(primary_fold_region)}px "
        f"boundary={cv2.countNonZero(primary_fold_boundary)}px authoritative={primary_fold_authoritative}"
    )
    pose_deform_mask = ((np.asarray(depth_state.get('pose_deform_mask', np.zeros_like(mask)), dtype=np.uint8) > 0).astype(np.uint8) * 255)
    pose_deform_bool = pose_deform_mask > 0
    pose_deform_conf = float(depth_state.get('pose_deform_confidence', 0.0) or 0.0)
    pose_deform_available = bool(depth_state.get('pose_deform_available', False)) and cv2.countNonZero(pose_deform_mask) >= 8 and pose_deform_conf >= 0.22
    pose_deform_mode = str(depth_state.get('pose_deform_mode', 'NONE'))
    pose_deform_side_hint = str(depth_state.get('pose_deform_side_hint', 'NONE'))
    pose_fold_region_hint = str(depth_state.get('pose_deform_fold_region_hint', 'NONE'))
    pose_fold_region_conf = float(depth_state.get('pose_deform_fold_region_confidence', 0.0) or 0.0)
    pose_fold_anchor_px_raw = depth_state.get('pose_deform_fold_anchor_px', None)
    pose_fold_direction_px_raw = depth_state.get('pose_deform_fold_direction_px', None)
    print(
        f"[E122-POSE-BRIDGE] region={pose_fold_region_hint}:{pose_fold_region_conf:.2f} "
        f"anchor={'Y' if pose_fold_anchor_px_raw is not None else 'N'} "
        f"direction={'Y' if pose_fold_direction_px_raw is not None else 'N'}"
    )
    # E122: never OR structural/sensitive skeletons into one connected-component
    # pool. Split the high-recall skeleton at graph junctions, score each branch
    # bilaterally, and pass only branch-qualified fold boundaries to the planner.
    e122_pool=_e122_build_scored_crease_pool(depth_state,mask,args,H_value=H_value)
    _e153_publish_e122_pool(depth_state,e122_pool,mask)
    print('[E153-ZEROCOPY-PUBLISH] planner reused cached E122 maps; no 720p recopy')
    _exec=np.asarray(e122_pool.get('execution_skeleton',np.zeros_like(mask)),dtype=np.uint8)
    _preview=np.asarray(e122_pool.get('preview_skeleton',np.zeros_like(mask)),dtype=np.uint8)
    if cv2.countNonZero(_exec)>=8:
        base_crease=_exec.copy(); base_crease_source='E122_EXECUTION_BRANCHES'
    elif cv2.countNonZero(_preview)>=8:
        base_crease=_preview.copy(); base_crease_source='E122_PREVIEW_BRANCHES'
    else:
        base_crease=np.zeros_like(mask,dtype=np.uint8); base_crease_source='E122_NO_FOLD_BRANCH'
    _e122_score=np.asarray(e122_pool.get('branch_score_map',np.zeros(mask.shape,dtype=np.float32)),dtype=np.float32)
    _e124_depth_map=np.asarray(e122_pool.get('branch_unidepth_map',np.zeros(mask.shape,dtype=np.float32)),dtype=np.float32)

    # E125: once a folded region exists, do not let an arbitrary interior wrinkle
    # become SELECTED CREASE. Retain raw crease only near the region boundary and
    # remove pixels deep inside the folded flap. If nothing remains, use the
    # region's internal boundary as a virtual selected-crease candidate.
    crease=base_crease.copy(); crease_source=base_crease_source
    if primary_fold_authoritative and cv2.countNonZero(primary_fold_boundary)>=8:
        bnear=cv2.dilate(primary_fold_boundary,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(45,45)),iterations=1)>0
        deep=cv2.erode(primary_fold_region,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(19,19)),iterations=1)>0
        filtered=((base_crease>0)&bnear&(~deep)).astype(np.uint8)*255
        if cv2.countNonZero(filtered)>=8:
            crease=filtered
            crease_source=base_crease_source+'_E125_REGION_BOUNDARY'
        else:
            crease=primary_fold_boundary.copy()
            crease_source='E125_VIRTUAL_FOLD_REGION_BOUNDARY'
    _e125_virtual = str(crease_source).startswith('E125_VIRTUAL')
    score_map=_e122_score if np.any(_e122_score>0) else np.asarray(depth_state.get('fold_crease_score',np.zeros(mask.shape,dtype=np.float32)),dtype=np.float32)
    if _e125_virtual:
        score_map=score_map.copy()
        score_map[crease>0]=np.maximum(score_map[crease>0],float(np.clip(primary_fold_conf,0.0,1.0)))
        _e124_depth_map=_e124_depth_map.copy()
        _e124_depth_map[crease>0]=np.maximum(_e124_depth_map[crease>0],float(np.clip(primary_fold_conf if primary_fold_source=='UNIDEPTH_RAISED' else 0.45,0.0,1.0)))
    print(
        f"[E124-FOLD-BRANCH-POOL] source={crease_source} raw={int(e122_pool.get('raw_branch_count',0))} "
        f"struct={int(e122_pool.get('structural_branch_count',0))} sensNovel={int(e122_pool.get('sensitive_branch_count',0))} "
        f"preview={int(e122_pool.get('preview_branch_count',0))}@{float(e122_pool.get('preview_threshold',0.34)):.2f} "
        f"exec={int(e122_pool.get('execution_branch_count',0))}@{float(e122_pool.get('execution_threshold',0.48)):.2f} "
        f"rescue={int(e122_pool.get('rescue_branch_count',0))} pixels={cv2.countNonZero(crease)}"
    )
    for _bi in list(e122_pool.get('branch_info',[]))[:6]:
        print(
            f"[E124-BRANCH] id={int(_bi.get('branch_id',-1))} src={str(_bi.get('source','?'))} "
            f"score={float(_bi.get('fold_score',0.0)):.3f} raw={float(_bi.get('fold_score_raw',0.0)):.3f} "
            f"span={float(_bi.get('span_px',0.0)):.1f}px valid={float(_bi.get('valid_sample_ratio',0.0)):.2f} "
            f"depth={float(_bi.get('depth_step_mm',0.0)):.2f}mm side={float(_bi.get('side_consistency',0.5)):.2f} "
            f"U={float(_bi.get('unidepth_fold_support',0.0)):.2f} depthOK={bool(_bi.get('depth_execution_ok',False))} "
            f"seam={float(_bi.get('seam_overlap',0.0)):.2f} rescue={bool(_bi.get('structural_rescue',False))} "
            f"preview={bool(_bi.get('preview_ok',False))} exec={bool(_bi.get('execution_ok',False))}"
        )
    pose_core = ((np.asarray(depth_state.get('pose_body_core_mask', np.zeros_like(mask)), dtype=np.uint8) > 0).astype(np.uint8) * 255)
    pose_seam = ((np.asarray(depth_state.get('pose_seam_exclusion_mask', np.zeros_like(mask)), dtype=np.uint8) > 0).astype(np.uint8) * 255)
    if cv2.countNonZero(mask) < 500 or cv2.countNonZero(crease) < 8:
        return None

    h, w = mask.shape[:2]
    garment_area = float(cv2.countNonZero(mask))
    if primary_fold_authoritative and cv2.countNonZero(primary_fold_boundary)>0:
        _b0=((primary_fold_boundary==0).astype(np.uint8)*255)
        primary_fold_boundary_dist=cv2.distanceTransform(_b0,cv2.DIST_L2,5).astype(np.float32)
        primary_fold_boundary_near=(cv2.dilate(primary_fold_boundary,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(29,29)),iterations=1)>0)
        primary_fold_deep=(cv2.erode(primary_fold_region,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(17,17)),iterations=1)>0)
    else:
        primary_fold_boundary_dist=np.full(mask.shape,999.0,dtype=np.float32)
        primary_fold_boundary_near=np.zeros(mask.shape,dtype=bool)
        primary_fold_deep=np.zeros(mask.shape,dtype=bool)
    neck_fold_override=bool(primary_fold_authoritative and primary_fold_source=='POSE_INTRUSION' and 'NECK' in set(str(primary_fold_group).split('+')) and primary_fold_conf>=0.34)
    mm = cv2.moments(mask, binaryImage=True)
    gc = np.asarray([float(mm['m10']/max(1e-6,mm['m00'])), float(mm['m01']/max(1e-6,mm['m00']))], dtype=np.float64)
    gc_board = _fix53_px_to_board(H_value, gc)
    inside = cv2.distanceTransform(mask, cv2.DIST_L2, 5).astype(np.float32)

    # E120 directional pose guide in board coordinates.
    pose_fold_dir_board = None
    pose_fold_anchor_px = None
    try:
        if pose_fold_anchor_px_raw is not None:
            pose_fold_anchor_px=np.asarray(pose_fold_anchor_px_raw,dtype=np.float64).reshape(2)
        if pose_fold_direction_px_raw is not None:
            _dpx=np.asarray(pose_fold_direction_px_raw,dtype=np.float64).reshape(2)
            _dn=float(np.linalg.norm(_dpx))
            if _dn>1e-6:
                _dpx=_dpx/_dn
                _b2=_fix53_px_to_board(H_value,gc+_dpx*40.0)
                pose_fold_dir_board=np.asarray(_b2-gc_board,dtype=np.float64)
                pose_fold_dir_board/=max(1e-9,float(np.linalg.norm(pose_fold_dir_board)))
    except Exception:
        pose_fold_dir_board=None

    # E133 PRIMARY manipulation geometry.  Any admitted Pose contour intrusion is
    # direction evidence even if E119's side/region classifier reports NONE.  When
    # Pose cannot yield a mechanically safe pair, UniDepth supplies multiple nearby
    # direction axes instead of one centroid axis.  Perception confidence never
    # blocks X once a board/reach/path-safe pair is found.
    try:
        _e133_pair=_e133_directional_pair(
            depth_state,mask,H_value,args,config,cfg,fold_region_info,_exec,_preview
        )
    except Exception as _e133_exc:
        print(f"[E133-DIRECTIONAL-WARN] {_e133_exc!r}")
        _e133_pair=None
    _v12_directional_candidate = _e133_pair if isinstance(_e133_pair,dict) else None
    if isinstance(_v12_directional_candidate,dict):
        print(
            f"[E156-V12-DIRECTIONAL-DEFER] safe directional plan retained as candidate; "
            f"src={_v12_directional_candidate.get('crease_source')} Q={float(_v12_directional_candidate.get('quality',0.0)):.3f}; "
            "continue building alternative complete plans before final selection"
        )
    if _v12_directional_candidate is None and bool(depth_state.get('e151_edge_cluster_hard_block',False)):
        _reason=str(depth_state.get('e151_directional_hard_block_reason','E151_EDGE_CLUSTER_HARD_BLOCK'))
        _info=dict(depth_state.get('e151_edge_cluster_block_info',{}) or {})
        print(
            f"[E151-LEGACY-PREVIEW-SUPPRESS] reason={_reason} side={str(_info.get('side','NONE'))} "
            f"score={float(_info.get('score',0.0)):.3f}; generic E122 HOLD/ACTIVE preview search skipped"
        )
        return None

    # E121: initialize pose state BEFORE the candidate-level neckline guard.
    # E120 referenced pose_state_for_split here before assignment; the try/except
    # hid that NameError and effectively disabled this second neck-line rejection.
    pose_state_for_split = dict(depth_state.get('tshirt_pose_state', {}) or {})
    # Candidate-level neckline guard. Pixelwise seam masking alone can miss a
    # parallel line a few pixels away from NECK_LEFT--NECK_RIGHT.
    neck_dist_map=np.full(mask.shape,999.0,dtype=np.float32); neck_tangent_px=None
    try:
        _named=dict((pose_state_for_split or {}).get('named_points',{}) or {})
        _nl=dict(_named.get('neck_left',{}) or {}); _nr=dict(_named.get('neck_right',{}) or {})
        _p1=np.asarray([float(_nl.get('x')),float(_nl.get('y'))],dtype=np.float64)
        _p2=np.asarray([float(_nr.get('x')),float(_nr.get('y'))],dtype=np.float64)
        if np.all(np.isfinite(_p1)) and np.all(np.isfinite(_p2)) and float(np.linalg.norm(_p2-_p1))>8.0:
            _nm=np.zeros_like(mask,dtype=np.uint8)
            cv2.line(_nm,tuple(np.rint(_p1).astype(int)),tuple(np.rint(_p2).astype(int)),255,5,cv2.LINE_AA)
            neck_dist_map=cv2.distanceTransform(((_nm==0).astype(np.uint8)*255),cv2.DIST_L2,5).astype(np.float32)
            neck_tangent_px=(_p2-_p1)/float(np.linalg.norm(_p2-_p1))
    except Exception:
        pass

    relief = np.asarray(depth_state.get('local_relief_mm', np.zeros(mask.shape,dtype=np.float32)), dtype=np.float32)
    prom = np.asarray(depth_state.get('prominence_mm', np.zeros(mask.shape,dtype=np.float32)), dtype=np.float32)
    finite = np.isfinite(relief) & np.isfinite(prom)
    flat = np.abs(np.where(np.isfinite(relief),relief,0.0)) + 0.70*np.maximum(np.where(np.isfinite(prom),prom,0.0),0.0)
    # E137+ has UniDepth disabled. In that RGB+Pose-only runtime these two maps are
    # exactly zero, so running thousands of median/p70 calculations cannot affect
    # ranking and only burns CPU. Detect that invariant once per frozen snapshot and
    # short-circuit depth-only statistics while preserving the same neutral scores.
    _e128_depth_runtime_available = bool(depth_state.get('unidepth_enabled', False))
    if _e128_depth_runtime_available:
        _depth_nonzero = bool(np.any(np.abs(relief) > 1e-6) or np.any(np.abs(prom) > 1e-6))
        _e128_depth_runtime_available = bool(_depth_nonzero)
    if not _e128_depth_runtime_available:
        print('[E156-V5-E128-NODEPTH-FASTPATH] UniDepth=OFF/zero maps -> skip local depth percentiles; neutral depth role preserved')
    # E112 HOLD cleanliness maps: HOLD should live on calm BODY cloth, not on
    # another detected crease or a multi-layer fold. Distance is measured from
    # a lightly dilated version of every detected Fold Crease component.
    crease_for_hold = crease if cv2.countNonZero(crease) >= 8 else sensitive_crease
    crease_near = cv2.dilate(crease_for_hold, np.ones((5,5),np.uint8), iterations=1)
    crease_dist_px = cv2.distanceTransform(((crease_near==0).astype(np.uint8)*255), cv2.DIST_L2, 5).astype(np.float32)

    frame = depth_state.get('frame')
    L = None
    if frame is not None:
        try:
            fb = np.asarray(frame, dtype=np.uint8)
            if fb.shape[:2] == mask.shape:
                L = cv2.cvtColor(fb, cv2.COLOR_BGR2LAB)[:,:,0].astype(np.float32)
        except Exception:
            L = None

    # E117: FUSED is weak reference only (1% in final ranking).
    labels = np.asarray(depth_state.get('visual_top2_labels', depth_state.get('labels', np.zeros_like(mask))), dtype=np.uint8)
    fused = ((labels == 1) & (mask > 0))
    if np.count_nonzero(fused) >= 8:
        fused_dist_px = cv2.distanceTransform(((~fused).astype(np.uint8)*255), cv2.DIST_L2, 5).astype(np.float32)
    else:
        fused_dist_px = np.full(mask.shape, 999.0, dtype=np.float32)

    active_target = float(getattr(args, 'micro_unfold_e99_active_crease_mm', 30.0))
    hold_target = float(getattr(args, 'micro_unfold_e99_hold_crease_mm', 40.0))
    pair_min = float(getattr(args, 'micro_unfold_e99_pair_min_mm', 50.0))
    pair_max = float(getattr(args, 'micro_unfold_e99_pair_max_mm', 100.0))
    pair_target = float(getattr(args, 'micro_unfold_e99_pair_target_mm', 70.0))

    # Old E102 partition survives at only 2% as a historical soft tie-break.
    soft_part = None
    try:
        p_old = _e98_try_crease_partition(mask, crease, pose_core)
        if isinstance(p_old, dict):
            soft_part = p_old
    except Exception:
        soft_part = None
    soft_flap = (np.asarray(soft_part.get('flap_region'), dtype=np.uint8) > 0) if isinstance(soft_part,dict) else np.zeros_like(mask,dtype=bool)
    soft_body = (np.asarray(soft_part.get('body_region'), dtype=np.uint8) > 0) if isinstance(soft_part,dict) else np.zeros_like(mask,dtype=bool)
    soft_selected = (np.asarray(soft_part.get('selected_crease'), dtype=np.uint8) > 0) if isinstance(soft_part,dict) else np.zeros_like(mask,dtype=bool)

    pose_bool = pose_core > 0
    seam_bool = pose_seam > 0
    # pose_state_for_split already initialized before neckline guard (E121).
    outer_cs, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not outer_cs:
        return None
    outer_contour = max(outer_cs, key=cv2.contourArea)
    outer_pts = outer_contour.reshape(-1,2).astype(np.float64)

    # V5 E128 immutable same-X geometry cache. These values are exact for the
    # frozen observation and avoid redoing 9x9 dilation, Homography inversion,
    # standby inverse-affine conversion, and affine setup for every candidate.
    _e128_invH = _e156_inverse_h(H_value)
    _e128_e111_stable = cv2.dilate(mask, np.ones((9,9), np.uint8), iterations=1) > 0
    _e128_arm_geom = {}
    if config is not None and cfg is not None:
        for _ak in ("arm1", "arm2"):
            try:
                _acfg = dict(config.get("dual_roarm", {}).get(_ak, {}) or {})
                _M = np.asarray(_acfg.get("board_to_roarm_affine_2x3"), dtype=np.float64).reshape(2,3)
                _off = _acfg.get("roarm_xy_offset", [0.0, 0.0])
                _offv = np.asarray([float(_off[0]), float(_off[1])], dtype=np.float64) if isinstance(_off,(list,tuple)) and len(_off)>=2 else np.zeros(2,dtype=np.float64)
                _sb_local = np.asarray(standby_roarm_pose(cfg, _ak)[:2], dtype=np.float64)
                _sb_board = _fix37_local_xy_to_board(config, _ak, _sb_local)
                _e128_arm_geom[_ak] = {"M":_M, "off":_offv, "standby_local":_sb_local, "standby_board":_sb_board}
            except Exception:
                pass

    def _e128_local_xy(arm_key, board_xy):
        g = _e128_arm_geom.get(str(arm_key))
        if not isinstance(g, dict):
            return _fix6_local_xy(config, arm_key, board_xy)
        b = np.asarray(board_xy, dtype=np.float64).reshape(2)
        return (g["M"] @ np.asarray([b[0], b[1], 1.0], dtype=np.float64) + g["off"]).astype(np.float64)

    def px_inside(p, min_inside=4.0):
        x,y=int(round(float(p[0]))),int(round(float(p[1])))
        return bool(0<=x<w and 0<=y<h and mask[y,x]>0 and float(inside[y,x])>=float(min_inside))

    _local_region_support_cache = {}
    def local_region_support(region_bool, p, rad=9):
        if region_bool is None or not np.any(region_bool):
            return 0.0
        x,y=int(round(float(p[0]))),int(round(float(p[1])))
        rr=int(max(1,rad))
        key=(id(region_bool),x,y,rr)
        cached=_local_region_support_cache.get(key)
        if cached is not None:
            return float(cached)
        x0,x1=max(0,x-rr),min(w,x+rr+1); y0,y1=max(0,y-rr),min(h,y+rr+1)
        if x1<=x0 or y1<=y0:
            return 0.0
        disk=np.zeros((y1-y0,x1-x0),dtype=np.uint8)
        cv2.circle(disk,(x-x0,y-y0),rr,1,-1)
        den=max(1,int(np.count_nonzero(disk)))
        out=float(np.count_nonzero(region_bool[y0:y1,x0:x1] & (disk>0)))/float(den)
        _local_region_support_cache[key]=out
        return out

    def local_primary_fold_support(p, rad=11):
        if not primary_fold_authoritative or not np.any(primary_fold_bool):
            return 0.5
        return local_region_support(primary_fold_bool,p,rad=rad)

    def local_pose_support(p, rad=13):
        return local_region_support(pose_bool, p, rad=rad) if cv2.countNonZero(pose_core)>0 else 0.0

    def local_pose_fold_support(p, rad=30):
        # Neutral 0.5 when the pose does not report a folded-in boundary point;
        # otherwise reward crease anchors close to the intrusion corridor.
        if not pose_fold_available:
            return 0.5
        raw = local_region_support(pose_fold_bool, p, rad=rad)
        return float(np.clip(0.20 + 2.6*raw, 0.0, 1.0))

    def local_pose_deform_support(p, rad=34):
        # E119 primary fold-location prior from topology deformation.
        if not pose_deform_available:
            return 0.5
        raw = local_region_support(pose_deform_bool, p, rad=rad)
        return float(np.clip(0.12 + 2.9*raw*max(0.35,pose_deform_conf), 0.0, 1.0))

    def local_patch_stats(p, rad=9):
        x,y=int(round(float(p[0]))),int(round(float(p[1])))
        x0,x1=max(0,x-rad),min(w,x+rad+1); y0,y1=max(0,y-rad),min(h,y+rad+1)
        if x1<=x0 or y1<=y0:
            return (0.0,0.0,0.0)
        m=(mask[y0:y1,x0:x1]>0)
        if np.count_nonzero(m)<8:
            return (0.0,0.0,0.0)
        if L is not None:
            vals=L[y0:y1,x0:x1][m]
            lmean=float(np.mean(vals)); lstd=float(np.std(vals))
        else:
            lmean=0.0; lstd=0.0
        if _e128_depth_runtime_available:
            rv=relief[y0:y1,x0:x1][m & np.isfinite(relief[y0:y1,x0:x1])]
            rmed=float(np.median(rv)) if rv.size else 0.0
        else:
            rmed=0.0
        return lmean,lstd,rmed

    def _hold_clean_metrics_uncached(p, rad=None):
        rad=max(8,int(round(float(rad if rad is not None else getattr(args,'micro_unfold_hold_clean_radius_px',22.0)))))
        x,y=int(round(float(p[0]))),int(round(float(p[1])))
        if not (0<=x<w and 0<=y<h and mask[y,x]>0):
            return {'score':0.0,'crease_clear_px':0.0,'crease_density':1.0,'flat_support':0.0,'inside_support':0.0}
        x0,x1=max(0,x-rad),min(w,x+rad+1); y0,y1=max(0,y-rad),min(h,y+rad+1)
        yy,xx=np.ogrid[y0:y1,x0:x1]
        disk=((xx-x)**2+(yy-y)**2)<=rad*rad
        gm=(mask[y0:y1,x0:x1]>0)&disk
        den=max(1,int(np.count_nonzero(gm)))
        crease_density=float(np.count_nonzero((crease_near[y0:y1,x0:x1]>0)&gm))/float(den)
        clear_px=float(crease_dist_px[y,x])
        clear_support=float(np.clip(clear_px/30.0,0.0,1.0))
        if _e128_depth_runtime_available:
            fv=flat[y0:y1,x0:x1][gm & np.isfinite(flat[y0:y1,x0:x1])]
            fmed=float(np.median(fv)) if fv.size else 99.0
            flat_support=float(1.0/(1.0+max(0.0,fmed)/10.0))
        else:
            # Zero depth maps imply the exact old neutral flat value fmed=0.
            fmed=0.0
            flat_support=1.0
        density_support=float(1.0-np.clip(crease_density/0.14,0.0,1.0))
        inside_support=float(np.clip(float(inside[y,x])/26.0,0.0,1.0))
        score=float(0.38*clear_support+0.28*density_support+0.20*flat_support+0.14*inside_support)
        return {'score':score,'crease_clear_px':clear_px,'crease_density':crease_density,'flat_support':flat_support,'inside_support':inside_support,'flat_median':fmed}

    _hold_clean_metrics_cache = {}
    def hold_clean_metrics(p, rad=None):
        rr=max(8,int(round(float(rad if rad is not None else getattr(args,'micro_unfold_hold_clean_radius_px',22.0)))))
        x,y=int(round(float(p[0]))),int(round(float(p[1])))
        key=(x,y,rr)
        cached=_hold_clean_metrics_cache.get(key)
        if cached is not None:
            return cached
        out=_hold_clean_metrics_uncached(p, rad=rr)
        _hold_clean_metrics_cache[key]=out
        return out

    def _depth_role_metrics_uncached(p, rad=None):
        """Robust local UniDepth raised-surface evidence for ACTIVE/HOLD role assignment.

        Positive local_relief/prominence means the cloth is locally raised above its
        neighborhood. E115 compares patches, not single pixels, so small depth noise
        does not flip the two roles.
        """
        rad=max(5,int(round(float(rad if rad is not None else getattr(args,'micro_unfold_depth_role_radius_px',10.0)))))
        x,y=int(round(float(p[0]))),int(round(float(p[1])))
        if not (0<=x<w and 0<=y<h and mask[y,x]>0):
            return {'valid':False,'raised_mm':0.0,'relief_med_mm':0.0,'relief_p70_mm':0.0,'prom_p70_mm':0.0,'sample_count':0}
        x0,x1=max(0,x-rad),min(w,x+rad+1); y0,y1=max(0,y-rad),min(h,y+rad+1)
        yy,xx=np.ogrid[y0:y1,x0:x1]
        disk=((xx-x)**2+(yy-y)**2)<=rad*rad
        gm=(mask[y0:y1,x0:x1]>0)&disk
        rr=relief[y0:y1,x0:x1]
        pp=prom[y0:y1,x0:x1]
        valid=gm & np.isfinite(rr) & np.isfinite(pp)
        n=int(np.count_nonzero(valid))
        if n<14:
            return {'valid':False,'raised_mm':0.0,'relief_med_mm':0.0,'relief_p70_mm':0.0,'prom_p70_mm':0.0,'sample_count':n}
        rv=np.asarray(rr[valid],dtype=np.float32)
        pv=np.maximum(np.asarray(pp[valid],dtype=np.float32),0.0)
        rmed=float(np.median(rv))
        rp70=float(np.percentile(rv,70.0))
        pp70=float(np.percentile(pv,70.0))
        # Keep metric units (mm). Relief dominates; prominence stabilizes a raised fold.
        raised=float(0.55*rp70+0.25*rmed+0.20*pp70)
        return {'valid':True,'raised_mm':raised,'relief_med_mm':rmed,'relief_p70_mm':rp70,'prom_p70_mm':pp70,'sample_count':n}

    _depth_role_metrics_cache = {}
    _depth_role_neutral={'valid':False,'raised_mm':0.0,'relief_med_mm':0.0,'relief_p70_mm':0.0,'prom_p70_mm':0.0,'sample_count':0}
    def depth_role_metrics(p, rad=None):
        if not _e128_depth_runtime_available:
            return _depth_role_neutral
        rr=max(5,int(round(float(rad if rad is not None else getattr(args,'micro_unfold_depth_role_radius_px',10.0)))))
        x,y=int(round(float(p[0]))),int(round(float(p[1])))
        key=(x,y,rr)
        cached=_depth_role_metrics_cache.get(key)
        if cached is not None:
            return cached
        out=_depth_role_metrics_uncached(p, rad=rr)
        _depth_role_metrics_cache[key]=out
        return out

    def search_clean_hold(anchor_board, hold_u, tangent_board, lobe_bool, body_bool, lobe_quality,
                          preferred_tangent_mm=0.0, active_board_for_pair=None,
                          active_normal_mm=None, desired_pair_mm=None):
        """Find HOLD on BODY, jointly conditioned on the already selected ACTIVE.

        E127 selected the cleanest HOLD independently and only afterwards tested
        pair separation/reach. The real log showed that this ordering discarded
        otherwise usable plans. E128 keeps all physical hard limits but moves
        those geometric preferences into HOLD selection.
        """
        nmin=float(getattr(args,'micro_unfold_hold_clean_normal_min_mm',32.0))
        nmax=float(getattr(args,'micro_unfold_hold_clean_normal_max_mm',58.0))
        tmax=float(getattr(args,'micro_unfold_hold_clean_tangent_mm',25.0))
        pair_conditioned = active_board_for_pair is not None
        if primary_fold_authoritative:
            nmin=max(18.0,nmin-14.0); nmax=max(nmax,78.0); tmax=max(tmax,34.0)

        pref=float(np.clip(float(preferred_tangent_mm),-tmax,tmax))
        target_pair=float(
            desired_pair_mm if desired_pair_mm is not None else
            (82.0 if primary_fold_authoritative else pair_target)
        )
        target_pair=float(np.clip(target_pair,pair_min+2.0,pair_max-2.0))

        desired_dn=float(hold_target)
        if pair_conditioned:
            try:
                ab=np.asarray(active_board_for_pair,dtype=np.float64)
                anch=np.asarray(anchor_board,dtype=np.float64)
                au=-np.asarray(hold_u,dtype=np.float64)
                au/=max(1e-9,float(np.linalg.norm(au)))
                active_dn=max(0.0,float(np.dot(ab-anch,au)))
                desired_dn=float(np.clip(target_pair-active_dn,nmin,nmax))
            except Exception:
                desired_dn=float(np.clip(hold_target,nmin,nmax))

        nd=np.unique(np.clip(np.asarray([
            desired_dn-12.0,desired_dn-6.0,desired_dn,desired_dn+6.0,desired_dn+12.0,
            hold_target-8.0,hold_target,hold_target+8.0,nmin,nmax
        ],dtype=np.float64),nmin,nmax))
        td=np.unique(np.clip(np.asarray([
            pref-tmax*0.50,pref-tmax*0.25,pref,pref+tmax*0.25,pref+tmax*0.50,0.0
        ],dtype=np.float64),-tmax,tmax))

        def _approach_pair_support(active_b, hold_b):
            # Pre-score the same 420 mm / 65 mm geometry used later by role_precheck.
            # Pull-end reach remains a later hard check after E111 computes it.
            if config is None or cfg is None:
                return True,1.0,999.0,0.0
            best_local=None
            for active_arm,hold_arm in (('arm1','arm2'),('arm2','arm1')):
                try:
                    a_local=_e128_local_xy(active_arm,active_b)
                    h_local=_e128_local_xy(hold_arm,hold_b)
                    max_radius=max(float(np.linalg.norm(a_local)),float(np.linalg.norm(h_local)))
                    if max_radius>420.0:
                        continue
                    a_standby_board=np.asarray(_e128_arm_geom.get(active_arm,{}).get("standby_board",
                        _fix37_local_xy_to_board(config,active_arm,np.asarray(standby_roarm_pose(cfg,active_arm)[:2],dtype=np.float64))),dtype=np.float64)
                    ts=np.linspace(0.0,1.0,61,dtype=np.float64)
                    path=a_standby_board[None,:]+ts[:,None]*(np.asarray(active_b)-a_standby_board)[None,:]
                    path_clear=float(np.min(np.linalg.norm(path-np.asarray(hold_b)[None,:],axis=1)))
                    if path_clear<65.0:
                        continue
                    clear_support=float(np.clip((path_clear-65.0)/35.0,0.0,1.0))
                    radius_support=1.0 if max_radius<=370.0 else float(np.clip((420.0-max_radius)/50.0,0.0,1.0))
                    support=float(0.62*clear_support+0.38*radius_support)
                    item=(support,path_clear,max_radius)
                    if best_local is None or item[0]>best_local[0]:
                        best_local=item
                except Exception:
                    continue
            if best_local is None:
                return False,0.0,0.0,999.0
            return True,float(best_local[0]),float(best_local[1]),float(best_local[2])

        best=None
        for dn in nd:
            for dt in td:
                hb=np.asarray(anchor_board,dtype=np.float64)+np.asarray(hold_u,dtype=np.float64)*float(dn)+np.asarray(tangent_board,dtype=np.float64)*float(dt)
                try: hp=_fix89_board_to_px(H_value,hb)
                except Exception: continue
                if not px_inside(hp,6.0): continue
                hx0,hy0=int(round(float(hp[0]))),int(round(float(hp[1])))
                if not finite[hy0,hx0]: continue

                pair_sep=None; pair_support=0.5; approach_support=0.5; approach_clear=999.0; approach_radius=0.0
                if pair_conditioned:
                    ab=np.asarray(active_board_for_pair,dtype=np.float64)
                    pair_sep=float(np.linalg.norm(ab-hb))
                    if not (pair_min<=pair_sep<=pair_max):
                        continue
                    pair_support=float(np.exp(-abs(pair_sep-target_pair)/12.0))
                    approach_ok,approach_support,approach_clear,approach_radius=_approach_pair_support(ab,hb)
                    if not approach_ok:
                        continue

                lsup=local_region_support(lobe_bool,hp,10) if np.any(lobe_bool) else 0.0
                bsup=local_region_support(body_bool,hp,10) if np.any(body_bool) else 0.0
                cm=hold_clean_metrics(hp)
                target_support=float(np.exp(-abs(float(dn)-desired_dn)/16.0))*float(np.exp(-abs(float(dt)-pref)/24.0))
                body_bonus=(0.5 if not np.any(body_bool) else bsup)
                lobe_hold_penalty=float(lobe_quality)*float(np.clip((lsup-0.20)/0.55,0.0,1.0))
                deform_hold_penalty=(local_region_support(pose_deform_bool,hp,22)*pose_deform_conf) if pose_deform_available else 0.0
                fold_hold_support=local_primary_fold_support(hp,rad=12) if primary_fold_authoritative else 0.0
                hold_fold_max=float(getattr(args,'micro_unfold_e126_hold_max_fold_support',0.50) if primary_fold_authoritative else getattr(args,'micro_unfold_e125_hold_max_fold_support',0.35))
                if primary_fold_authoritative and fold_hold_support>hold_fold_max:
                    continue
                fold_outside_support=(1.0-fold_hold_support) if primary_fold_authoritative else 0.5

                if pair_conditioned:
                    score=float(
                        0.40*cm['score'] + 0.08*body_bonus + 0.06*target_support +
                        0.18*fold_outside_support + 0.18*pair_support + 0.10*approach_support -
                        0.05*lobe_hold_penalty - 0.14*deform_hold_penalty
                    )
                else:
                    score=float(0.58*cm['score']+0.10*body_bonus+0.08*target_support+0.24*fold_outside_support-
                                0.06*lobe_hold_penalty-0.18*deform_hold_penalty)
                item={'board':hb,'px':np.asarray(hp,dtype=np.float64),'normal_mm':float(dn),'tangent_mm':float(dt),
                      'score':score,'lobe_support':float(lsup),'body_support':float(bsup),
                      'pose_deform_penalty':float(deform_hold_penalty),'fold_region_support':float(fold_hold_support),'metrics':cm,
                      'e128_pair_conditioned':bool(pair_conditioned),'e128_pair_target_mm':float(target_pair),
                      'e128_pair_sep_mm':None if pair_sep is None else float(pair_sep),
                      'e128_approach_support':float(approach_support),'e128_approach_clear_mm':float(approach_clear),
                      'e128_approach_max_radius_mm':float(approach_radius)}
                if best is None or score>best['score']:
                    best=item
        return best


    def search_active_grasp(anchor_board, active_u, tangent_board, lobe_bool, lobe_quality):
        # E122: ACTIVE is no longer one fixed +30mm pixel. Search a compact
        # normal/tangent grid while preserving the nominated folded side.
        nmin=max(18.0,float(getattr(args,'micro_unfold_e122_active_normal_min_mm',22.0)))
        nmax=max(nmin+4.0,float(getattr(args,'micro_unfold_e122_active_normal_max_mm',46.0)))
        tmax=max(0.0,float(getattr(args,'micro_unfold_e122_active_tangent_mm',18.0)))
        if primary_fold_authoritative:
            # E126: the experiment showed any safe point on the folded flap can
            # unfold it if direction is correct. Search closer to a narrow fold
            # boundary instead of demanding >=22mm of flap width.
            nmin=14.0; nmax=max(nmax,50.0); tmax=max(tmax,24.0)
        nd=np.unique(np.clip(np.asarray([active_target-8.0,active_target-3.0,active_target,active_target+6.0,active_target+12.0,nmin,nmax],dtype=np.float64),nmin,nmax))
        td=np.asarray([-tmax,-0.5*tmax,0.0,0.5*tmax,tmax],dtype=np.float64)
        best=None
        for dn in nd:
            for dt in td:
                ab=np.asarray(anchor_board,dtype=np.float64)+np.asarray(active_u,dtype=np.float64)*float(dn)+np.asarray(tangent_board,dtype=np.float64)*float(dt)
                try: ap=_fix89_board_to_px(H_value,ab)
                except Exception: continue
                if not px_inside(ap,5.0): continue
                ax0,ay0=int(round(float(ap[0]))),int(round(float(ap[1])))
                if not finite[ay0,ax0]: continue
                interior=float(np.clip(float(inside[ay0,ax0])/18.0,0.0,1.0))
                if interior<0.18: continue
                dm=depth_role_metrics(ap)
                raised=float(dm.get('raised_mm',0.0)) if dm.get('valid',False) else 0.0
                raised_support=float(np.clip(0.5+0.5*np.tanh(raised/3.0),0.0,1.0))
                lsup=local_region_support(lobe_bool,ap,10) if np.any(lobe_bool) else 0.5
                dsup=(local_region_support(pose_deform_bool,ap,24)*pose_deform_conf) if pose_deform_available else 0.5
                fold_active_support=local_primary_fold_support(ap,rad=11) if primary_fold_authoritative else 0.5
                active_fold_min=float(getattr(args,'micro_unfold_e126_active_min_fold_support',0.16) if primary_fold_authoritative else getattr(args,'micro_unfold_e125_active_min_fold_support',0.30))
                if primary_fold_authoritative and fold_active_support<active_fold_min:
                    continue
                target=float(np.exp(-abs(float(dn)-active_target)/13.0))*float(np.exp(-abs(float(dt))/30.0))
                # E125: the known folded region outranks local wrinkle strength.
                score=float(0.14*target+0.18*interior+0.24*raised_support+0.07*((1.0-lobe_quality)*0.5+lobe_quality*lsup)+0.10*dsup+0.27*fold_active_support)
                item={'board':ab,'px':np.asarray(ap,dtype=np.float64),'normal_mm':float(dn),'tangent_mm':float(dt),
                      'score':score,'interior_support':interior,'raised_support':raised_support,'lobe_support':float(lsup),'pose_deform_support':float(dsup),
                      'fold_region_support':float(fold_active_support),'selection_mode':'CREASE_LOCAL_ACTIVE',
                      'pose_depth_authority':0.0}
                if best is None or score>best['score']:
                    best=item
        return best

    # E129: pre-rank the semantic POSE_INTRUSION region once. This candidate pool
    # never reads FUSED FOLD. Each crease/boundary anchor then filters the pool to
    # the folded side of that anchor, preserving BODY<->FOLD pairing geometry.
    _e129_pose_active_authority=bool(
        primary_fold_authoritative and primary_fold_source=='POSE_INTRUSION' and
        cv2.countNonZero(primary_fold_region)>=24
    )
    _e129_pose_active_pool=(
        _e129_rank_pose_intrusion_active_pixels(depth_state,mask,primary_fold_region,args)
        if _e129_pose_active_authority else []
    )
    if _e129_pose_active_authority:
        print(
            f"[E129-POSE-ACTIVE-POOL] source=POSE_INTRUSION group={primary_fold_group} "
            f"conf={primary_fold_conf:.3f} candidates={len(_e129_pose_active_pool)} "
            "authority=POSE_REGION->UNIDEPTH FUSED=REF_ONLY"
        )

    def search_pose_intrusion_active_grasp(anchor_board, active_u, tangent_board):
        if not _e129_pose_active_authority or not _e129_pose_active_pool:
            return None
        nmin=float(getattr(args,'micro_unfold_e129_pose_active_min_normal_mm',4.0))
        nmax=float(getattr(args,'micro_unfold_e129_pose_active_max_normal_mm',95.0))
        tmax=float(getattr(args,'micro_unfold_e129_pose_active_max_tangent_mm',72.0))
        best=None
        anchor=np.asarray(anchor_board,dtype=np.float64)
        au=np.asarray(active_u,dtype=np.float64)
        tu=np.asarray(tangent_board,dtype=np.float64)
        for cand in _e129_pose_active_pool:
            ap=np.asarray(cand['px'],dtype=np.float64)
            try:
                ab=np.asarray(_fix53_px_to_board(H_value,ap),dtype=np.float64)
            except Exception:
                continue
            delta=ab-anchor
            normal_mm=float(np.dot(delta,au))
            tangent_mm=float(np.dot(delta,tu))
            # Candidate must truly lie on this anchor's folded side. This is a
            # loose corridor, not the old fixed +30mm crease grasp rule.
            if normal_mm<nmin or normal_mm>nmax or abs(tangent_mm)>tmax:
                continue
            fold_support=local_primary_fold_support(ap,rad=11)
            if fold_support<0.24:
                continue
            dm=depth_role_metrics(ap)
            local_raised=float(dm.get('raised_mm',cand.get('raised_mm',0.0))) if dm.get('valid',False) else float(cand.get('raised_mm',0.0))
            raised_support=float(np.clip(0.5+0.5*np.tanh(local_raised/3.0),0.0,1.0))
            normal_support=float(np.exp(-abs(normal_mm-34.0)/42.0))
            tangent_support=float(np.exp(-abs(tangent_mm)/48.0))
            # Pose region is authoritative; UniDepth picks the best patch within it.
            authority=float(np.clip(
                0.46*fold_support+0.34*raised_support+
                0.12*float(cand.get('garment_interior_support',0.0))+
                0.04*normal_support+0.04*tangent_support,0.0,1.0
            ))
            item={
                'board':ab,'px':ap,'normal_mm':normal_mm,'tangent_mm':tangent_mm,
                'score':authority,'interior_support':float(cand.get('garment_interior_support',0.0)),
                'raised_support':raised_support,'raised_mm':local_raised,'lobe_support':0.5,
                'pose_deform_support':float(cand.get('pose_region_support',fold_support)),
                'fold_region_support':float(fold_support),
                'selection_mode':'POSE_INTRUSION_UNIDEPTH_ACTIVE',
                'pose_depth_authority':authority,
            }
            if best is None or authority>best['score']:
                best=item
        return best

    def ray_extent_mm(origin_board, unit_board, max_mm=180.0, step_mm=5.0):
        # V5 exact vectorization of the old 5mm first-outside sampler.
        u=np.asarray(unit_board,dtype=np.float64).reshape(2)
        un=float(np.linalg.norm(u))
        if un<=1e-9: return 0.0
        u/=un
        ds=np.arange(float(step_mm),float(max_mm)+1e-6,float(step_mm),dtype=np.float64)
        if ds.size==0: return 0.0
        bps=np.asarray(origin_board,dtype=np.float64).reshape(1,2)+ds[:,None]*u.reshape(1,2)
        try:
            pxs=cv2.perspectiveTransform(bps.astype(np.float32).reshape(-1,1,2),_e128_invH).reshape(-1,2)
        except Exception:
            return 0.0
        xi=np.rint(pxs[:,0]).astype(np.int32); yi=np.rint(pxs[:,1]).astype(np.int32)
        inb=(xi>=0)&(xi<w)&(yi>=0)&(yi<h)
        inside_v=np.zeros(len(ds),dtype=bool)
        ids=np.flatnonzero(inb)
        if ids.size: inside_v[ids]=mask[yi[ids],xi[ids]]>0
        bad=np.flatnonzero(~inside_v)
        if bad.size==0: return float(ds[-1])
        j=int(bad[0])
        return 0.0 if j==0 else float(ds[j-1])

    def role_precheck(active_board, hold_board, pull_board):
        if config is None or cfg is None:
            return True, {'travel':0.0,'path_clear':999.0,'radii':[0.0,0.0,0.0],'max_radius':0.0,'reach_support':1.0}
        best=None
        for active_arm,hold_arm in (('arm1','arm2'),('arm2','arm1')):
            try:
                a_local=_e128_local_xy(active_arm,active_board)
                h_local=_e128_local_xy(hold_arm,hold_board)
                p_local=_e128_local_xy(active_arm,pull_board)
                radii=[float(np.linalg.norm(a_local)),float(np.linalg.norm(h_local)),float(np.linalg.norm(p_local))]
                max_radius=max(radii)
                if max_radius>420.0: continue
                a_standby=np.asarray(_e128_arm_geom.get(active_arm,{}).get("standby_local",standby_roarm_pose(cfg,active_arm)[:2]),dtype=np.float64)
                h_standby=np.asarray(_e128_arm_geom.get(hold_arm,{}).get("standby_local",standby_roarm_pose(cfg,hold_arm)[:2]),dtype=np.float64)
                a_standby_board=np.asarray(_e128_arm_geom.get(active_arm,{}).get("standby_board",_fix37_local_xy_to_board(config,active_arm,a_standby)),dtype=np.float64)
                ts=np.linspace(0.0,1.0,61,dtype=np.float64)
                path=a_standby_board[None,:]+ts[:,None]*(np.asarray(active_board)-a_standby_board)[None,:]
                path_clear=float(np.min(np.linalg.norm(path-np.asarray(hold_board)[None,:],axis=1)))
                if path_clear<65.0: continue
                travel=float(np.linalg.norm(a_local-a_standby)+np.linalg.norm(h_local-h_standby))
                # Prefer margin below 370 mm without changing the 420 mm hard gate.
                reach_support=float(np.clip((420.0-max_radius)/50.0,0.0,1.0)) if max_radius>370.0 else 1.0
                item={'active_arm':active_arm,'hold_arm':hold_arm,'travel':travel,'path_clear':path_clear,'radii':radii,
                      'max_radius':float(max_radius),'reach_support':float(reach_support)}
                rank=(float(reach_support),-travel)
                if best is None or rank>(float(best['reach_support']),-float(best['travel'])):
                    best=item
            except Exception:
                continue
        return (best is not None),best

    def component_outer_lobe(comp_mask, pts):
        """Coarse crease+outer-contour partition used only for side ownership."""
        result={'available':False,'quality':0.0,'lobe_ratio':0.0,
                'lobe_region':np.zeros_like(mask,dtype=np.uint8),
                'body_region':np.zeros_like(mask,dtype=np.uint8),
                'endpoint_boundary_dist_px':[999.0,999.0]}
        try:
            ctr=pts.mean(axis=0); X=pts-ctr[None,:]
            vals,vecs=np.linalg.eigh(X.T@X)
            axis=vecs[:,int(np.argmax(vals))]
            proj=X@axis
            p0=pts[int(np.argmin(proj))]; p1=pts[int(np.argmax(proj))]
            nearest=[]
            for ep in (p0,p1):
                d2=np.sum((outer_pts-ep[None,:])**2,axis=1)
                j=int(np.argmin(d2)); nearest.append((outer_pts[j].copy(),float(np.sqrt(d2[j]))))
            d0,d1=nearest[0][1],nearest[1][1]
            result['endpoint_boundary_dist_px']=[float(d0),float(d1)]
            if max(d0,d1)>145.0 or float(np.linalg.norm(nearest[0][0]-nearest[1][0]))<18.0:
                return result
            best_local=None
            for thickness in (5,7,9):
                barrier=np.zeros_like(mask,dtype=np.uint8)
                barrier[comp_mask]=255
                cv2.line(barrier,tuple(np.rint(p0).astype(int)),tuple(np.rint(nearest[0][0]).astype(int)),255,thickness,cv2.LINE_AA)
                cv2.line(barrier,tuple(np.rint(p1).astype(int)),tuple(np.rint(nearest[1][0]).astype(int)),255,thickness,cv2.LINE_AA)
                barrier=cv2.dilate(barrier,np.ones((thickness,thickness),np.uint8),iterations=1)
                cut=((mask>0)&(barrier==0)).astype(np.uint8)
                nc,ll,ss,_=cv2.connectedComponentsWithStats(cut,8)
                if nc<3: continue
                regions=[]
                touch_band=cv2.dilate(barrier,np.ones((7,7),np.uint8),iterations=1)>0
                for lid in range(1,nc):
                    area=int(ss[lid,cv2.CC_STAT_AREA])
                    if area<180: continue
                    reg=(ll==lid)
                    touch=int(np.count_nonzero(reg & touch_band))
                    if touch<6: continue
                    regions.append((area,touch,lid,reg))
                if len(regions)<2: continue
                regions.sort(key=lambda q:q[0],reverse=True)
                body=regions[0]
                lobe=None
                for rr in regions[1:]:
                    ratio=float(rr[0])/max(1.0,garment_area)
                    if 0.020<=ratio<=0.48:
                        lobe=rr; break
                if lobe is None: continue
                ratio=float(lobe[0])/max(1.0,garment_area)
                # Clear single folds usually create a non-tiny side lobe. Keep a broad soft profile.
                if ratio<0.040:
                    ratio_support=float(np.clip((ratio-0.020)/0.020,0.0,1.0))*0.55
                elif ratio<=0.35:
                    ratio_support=1.0
                else:
                    ratio_support=float(np.clip((0.48-ratio)/0.13,0.0,1.0))
                closure_support=float(np.exp(-(0.5*(d0+d1))/70.0))
                size_separation=float(np.clip((float(body[0])-float(lobe[0]))/max(1.0,0.45*garment_area),0.0,1.0))
                q=0.55*ratio_support+0.30*closure_support+0.15*size_separation
                item={'available':True,'quality':float(q),'lobe_ratio':float(ratio),
                      'lobe_region':(lobe[3].astype(np.uint8)*255),'body_region':(body[3].astype(np.uint8)*255),
                      'endpoint_boundary_dist_px':[float(d0),float(d1)]}
                if best_local is None or q>best_local['quality']:
                    best_local=item
            if best_local is not None:
                return best_local
        except Exception:
            pass
        return result

    ncc,cc_labels,stats,_=cv2.connectedComponentsWithStats((crease>0).astype(np.uint8),8)
    components=[]
    for lab in range(1,ncc):
        area=int(stats[lab,cv2.CC_STAT_AREA])
        if area<8: continue
        ys,xs=np.where(cc_labels==lab)
        if xs.size<8: continue
        pts=np.column_stack([xs,ys]).astype(np.float64)
        ctr=pts.mean(axis=0)
        cov=np.cov((pts-ctr).T) if len(pts)>=3 else np.eye(2)
        vals,_vec=np.linalg.eigh(cov)
        span=float(2.0*np.sqrt(max(0.0,float(vals[-1]))))
        if span<18.0: continue
        comp_mask=(cc_labels==lab)
        lobe_info=component_outer_lobe(comp_mask,pts)
        pose_split=_e118_pose_region_split_counts(
            mask,
            np.asarray(lobe_info.get('lobe_region',np.zeros_like(mask)),dtype=np.uint8),
            np.asarray(lobe_info.get('body_region',np.zeros_like(mask)),dtype=np.uint8),
            pose_state_for_split,
            args=args,
        ) if bool(lobe_info.get('available',False)) else {
            'available':False,'count_a':0,'count_b':0,'assigned':0,'gap':0,'confidence':0.0,
            'preferred_active_region':'NONE','authoritative':False,'assignments':[]
        }
        components.append((lab,area,span,pts,lobe_info,pose_split))
    # V13 retained-plan invariant: a complete safe directional candidate must not
    # disappear merely because the legacy component pool is empty. V12 returned
    # None here, causing _fix53_plan_points() to raise even after READY mechanics.
    if not components and not isinstance(_v12_directional_candidate,dict):
        return None

    candidates=[]
    if isinstance(_v12_directional_candidate,dict):
        candidates.append(_v12_directional_candidate)
    orientation_rejected=0
    depth_role_rejected=0
    _e126_reject={'active':0,'hold':0,'fold_role':0,'extent':0,'sep':0,'same_fold':0,'region_pair':0,'reach_path':0,'depth':0}
    for lab,area,span,pts,lobe_info,pose_split in components:
        comp_mask=(cc_labels==lab)
        sv=score_map[comp_mask & np.isfinite(score_map)]
        comp_strength=float(np.mean(sv)) if sv.size else 0.0
        _uv=_e124_depth_map[comp_mask & np.isfinite(_e124_depth_map)]
        comp_unidepth_support=float(np.mean(_uv)) if _uv.size else 0.0
        _idv=inside[comp_mask]
        comp_interior_median=float(np.median(_idv)) if _idv.size else 0.0
        comp_interior_frac=float(np.mean(_idv>=10.0)) if _idv.size else 0.0
        if primary_fold_authoritative:
            comp_fold_overlap=float(np.mean(primary_fold_bool[comp_mask]))
            comp_boundary_support=float(np.mean(primary_fold_boundary_near[comp_mask])) if np.any(primary_fold_boundary_near) else 0.0
            comp_deep_overlap=float(np.mean(primary_fold_deep[comp_mask])) if np.any(primary_fold_deep) else 0.0
            # This is the central E125 protection: a crease deep inside the known
            # folded flap must not generate HOLD/ACTIVE even if visually strong.
            if comp_deep_overlap>0.45 and comp_boundary_support<0.20:
                continue
        else:
            comp_fold_overlap=0.0; comp_boundary_support=0.5; comp_deep_overlap=0.0
        # Reject almost-pure outer-silhouette artifacts. A true fold may touch the
        # contour at its endpoints, but should have a meaningful interior segment.
        if comp_interior_frac < 0.06 and comp_interior_median < 5.0:
            continue
        comp_selected_overlap=float(np.count_nonzero(comp_mask & soft_selected))/max(1.0,float(np.count_nonzero(comp_mask))) if np.any(soft_selected) else 0.0
        pose_seam_overlap=float(np.count_nonzero(comp_mask & seam_bool))/max(1.0,float(np.count_nonzero(comp_mask))) if np.any(seam_bool) else 0.0
        # If Pose recovered and explicitly marked this component as seam, do not revive it downstream.
        if pose_seam_overlap>0.12 and not neck_fold_override:
            continue
        lobe_bool=np.asarray(lobe_info.get('lobe_region',np.zeros_like(mask)),dtype=np.uint8)>0
        body_bool=np.asarray(lobe_info.get('body_region',np.zeros_like(mask)),dtype=np.uint8)>0
        lobe_quality=float(lobe_info.get('quality',0.0)) if bool(lobe_info.get('available',False)) else 0.0
        lobe_ratio=float(lobe_info.get('lobe_ratio',0.0))

        # Spread anchors along the actual Fold Crease shape.
        order=np.argsort(pts[:,1]+0.17*pts[:,0])
        pts_ord=pts[order]
        max_anchors=max(6,int(getattr(args,'micro_unfold_e128_max_anchors_per_component',8)))
        stride=max(1,int(np.ceil(len(pts_ord)/float(max_anchors))))
        for p in pts_ord[::stride][:max_anchors]:
            d2=np.sum((pts-p[None,:])**2,axis=1)
            neigh=pts[np.argsort(d2)[:min(45,len(pts))]]
            if len(neigh)<7: continue
            c=np.mean(neigh,axis=0); X=neigh-c[None,:]
            cov=X.T@X
            vals,vecs=np.linalg.eigh(cov)
            if float(vals[-1])<=1e-6: continue
            linearity=float(np.clip(1.0-float(vals[0])/max(1e-6,float(vals[-1])),0.0,1.0))
            if linearity<0.18: continue
            tangent_px=vecs[:,int(np.argmax(vals))]; tangent_px/=max(1e-6,float(np.linalg.norm(tangent_px)))
            p_board=_fix53_px_to_board(H_value,p)
            p2_board=_fix53_px_to_board(H_value,p+tangent_px*16.0)
            tangent_board=np.asarray(p2_board-p_board,dtype=np.float64)
            tbn=float(np.linalg.norm(tangent_board))
            if tbn<=1e-6: continue
            tangent_board/=tbn
            normal0=np.asarray([-tangent_board[1],tangent_board[0]],dtype=np.float64)

            # Near-crease bilateral statistics for pose-independent seam fallback.
            try:
                side_a_px=_fix89_board_to_px(H_value,np.asarray(p_board)+normal0*12.0)
                side_b_px=_fix89_board_to_px(H_value,np.asarray(p_board)-normal0*12.0)
                a_l,a_s,a_r=local_patch_stats(side_a_px,8)
                b_l,b_s,b_r=local_patch_stats(side_b_px,8)
                l_asym=min(1.0,abs(a_l-b_l)/24.0) if L is not None else 0.0
                s_asym=min(1.0,abs(a_s-b_s)/20.0) if L is not None else 0.0
                r_asym=min(1.0,abs(a_r-b_r)/8.0)
                surface_asym=float(np.clip(0.55*l_asym+0.15*s_asym+0.30*r_asym,0.0,1.0))
            except Exception:
                surface_asym=0.35
            px0=int(np.clip(round(p[0]),0,w-1)); py0=int(np.clip(round(p[1]),0,h-1))
            anchor_inside=float(inside[py0,px0])
            edge_near=float(np.exp(-anchor_inside/38.0))
            tiny_lobe=float(np.clip((0.065-lobe_ratio)/0.045,0.0,1.0)) if lobe_quality>0.0 else 0.35
            seam_like=float(np.clip((1.0-surface_asym)*(0.35+0.65*edge_near)*(0.55+0.45*tiny_lobe),0.0,1.0))
            seam_penalty=0.07*seam_like
            # Stronger geometric neckline rejection: near + parallel to neck segment.
            neck_dist=float(neck_dist_map[py0,px0]) if neck_dist_map.shape==mask.shape else 999.0
            neck_parallel=0.0
            if neck_tangent_px is not None:
                neck_parallel=float(abs(np.dot(tangent_px,neck_tangent_px)))
            neck_candidate_penalty=float(np.exp(-neck_dist/16.0))*float(np.clip((neck_parallel-0.45)/0.45,0.0,1.0))
            pose_fold_support = local_pose_fold_support(p, rad=32)
            pose_deform_anchor_support = local_pose_deform_support(p, rad=38)
            primary_anchor_support=local_primary_fold_support(p,rad=8) if primary_fold_authoritative else 0.5
            primary_anchor_boundary_dist=float(primary_fold_boundary_dist[py0,px0]) if primary_fold_authoritative else 0.0
            if primary_fold_authoritative and primary_anchor_support>0.72 and primary_anchor_boundary_dist>12.0:
                continue
            _neck_local_override=bool(neck_fold_override and primary_anchor_support>=0.22)
            if neck_dist < 10.0 and neck_parallel > 0.72 and pose_deform_anchor_support < 0.55 and not _neck_local_override:
                continue
            seam_penalty += (0.035 if _neck_local_override else 0.22)*neck_candidate_penalty

            # E117 ROLE ORDER: compare the SAME two opposite-side points first.
            # The higher probe determines ACTIVE direction; only after that do we
            # relocate HOLD inside the clean BODY band. This avoids E115/E116's
            # pathological case where both directions were rejected because each
            # direction searched a different HOLD point.
            probe_mm = max(20.0, float(getattr(args,'micro_unfold_depth_role_probe_mm',34.0)))
            probe_metrics = {}
            for _sgn in (1.0,-1.0):
                try:
                    _pb=np.asarray(p_board)+normal0*_sgn*probe_mm
                    _ppx=_fix89_board_to_px(H_value,_pb)
                    if px_inside(_ppx,4.0):
                        probe_metrics[_sgn]=depth_role_metrics(_ppx)
                except Exception:
                    pass
            probe_valid = bool(1.0 in probe_metrics and -1.0 in probe_metrics and probe_metrics[1.0].get('valid',False) and probe_metrics[-1.0].get('valid',False))
            probe_delta = float(probe_metrics.get(1.0,{}).get('raised_mm',0.0)-probe_metrics.get(-1.0,{}).get('raised_mm',0.0)) if probe_valid else 0.0
            probe_scale=max(0.50,float(getattr(args,'micro_unfold_depth_role_scale_mm',2.0)))
            probe_conf=float(np.tanh(abs(probe_delta)/probe_scale)) if probe_valid else 0.0
            probe_min=max(0.25,float(getattr(args,'micro_unfold_depth_role_min_delta_mm',0.70)))
            preferred_sign=(1.0 if probe_delta>=0.0 else -1.0) if probe_valid and abs(probe_delta)>=probe_min else None

            # E119: ask the learned pose-deformation prior which side of this same
            # crease normal actually lies inside the folded anatomical region.
            deform_probe_support={}
            if pose_deform_available:
                for _sgn in (1.0,-1.0):
                    try:
                        _pb=np.asarray(p_board)+normal0*_sgn*probe_mm
                        _ppx=_fix89_board_to_px(H_value,_pb)
                        deform_probe_support[_sgn]=local_region_support(pose_deform_bool,_ppx,rad=24)
                    except Exception:
                        deform_probe_support[_sgn]=0.0
            deform_probe_delta=float(deform_probe_support.get(1.0,0.0)-deform_probe_support.get(-1.0,0.0))
            deform_preferred_sign=(1.0 if deform_probe_delta>=0.0 else -1.0) if (
                pose_deform_available and pose_deform_conf>=0.34 and abs(deform_probe_delta)>=0.035
            ) else None
            # E125 PRIMARY FOLD REGION owns side orientation before all older
            # pose/depth/lobe votes.
            region_probe_support={}
            region_preferred_sign=None
            if primary_fold_authoritative:
                for _sgn in (1.0,-1.0):
                    try:
                        _pb=np.asarray(p_board)+normal0*_sgn*probe_mm
                        _ppx=_fix89_board_to_px(H_value,_pb)
                        region_probe_support[_sgn]=local_primary_fold_support(_ppx,rad=14)
                    except Exception:
                        region_probe_support[_sgn]=0.0
                _region_delta=float(region_probe_support.get(1.0,0.0)-region_probe_support.get(-1.0,0.0))
                if abs(_region_delta)>=0.06:
                    region_preferred_sign=1.0 if _region_delta>=0.0 else -1.0
            # Older topology direction remains secondary when the fold region is
            # unavailable/ambiguous.
            directional_preferred_sign=None
            if pose_fold_dir_board is not None and pose_fold_region_conf>=0.34:
                _a1=float(np.dot(normal0,pose_fold_dir_board))
                directional_preferred_sign=1.0 if _a1>=0.0 else -1.0
            primary_preferred_sign=(region_preferred_sign if region_preferred_sign is not None else
                                    (directional_preferred_sign if directional_preferred_sign is not None else
                                     (deform_preferred_sign if deform_preferred_sign is not None else preferred_sign)))
            active_sign_order=(primary_preferred_sign,-primary_preferred_sign) if primary_preferred_sign is not None else (1.0,-1.0)

            for active_sign in active_sign_order:
                active_u=normal0*active_sign; hold_u=-active_u
                pose_direction_alignment=0.0
                pose_axis_progress=0.5
                pose_axis_support=0.5
                if pose_fold_dir_board is not None and pose_fold_region_conf>=0.18:
                    pose_direction_alignment=float(np.dot(active_u,pose_fold_dir_board))
                    # A strong quadrant prior owns unfold direction. The opposite
                    # normal is not allowed to become ACTIVE merely because depth
                    # or a coarse lobe happened to look stronger.
                    if pose_fold_region_conf>=0.60 and pose_direction_alignment < -0.05:
                        continue
                    if pose_fold_anchor_px is not None:
                        try:
                            _fab=_fix53_px_to_board(H_value,pose_fold_anchor_px)
                            _fd=max(1e-6,float(np.linalg.norm(_fab-gc_board)))
                            pose_axis_progress=float(np.dot(np.asarray(p_board)-gc_board,pose_fold_dir_board)/_fd)
                        except Exception:
                            pose_axis_progress=0.5
                    dir_s=float(np.clip((pose_direction_alignment+0.15)/1.15,0.0,1.0))
                    prog_s=float(np.exp(-abs(pose_axis_progress-0.48)/0.42))
                    pose_axis_support=float(np.clip(0.68*dir_s+0.32*prog_s,0.0,1.0))
                # E129 authority order for ACTIVE:
                #   POSE_INTRUSION region -> UniDepth raised/graspable point -> local crease fallback.
                # FUSED FOLD is never consulted by this pose-authoritative path.
                active_pick=None
                if _e129_pose_active_authority:
                    active_pick=search_pose_intrusion_active_grasp(p_board,active_u,tangent_board)
                if not isinstance(active_pick,dict):
                    active_pick=search_active_grasp(p_board,active_u,tangent_board,lobe_bool,lobe_quality)
                    if isinstance(active_pick,dict) and _e129_pose_active_authority:
                        active_pick=dict(active_pick)
                        active_pick['selection_mode']='POSE_REGION_LOCAL_FALLBACK'
                if not isinstance(active_pick,dict):
                    _e126_reject['active']+=1
                    continue
                active_board=np.asarray(active_pick['board'],dtype=np.float64)
                active_px=np.asarray(active_pick['px'],dtype=np.float64)
                ax,ay=int(round(active_px[0])),int(round(active_px[1]))
                if not finite[ay,ax]: continue

                # E112: do not force HOLD to the one 40 mm pixel. Search a small
                # BODY-side patch for the cleanest crease-free, flat, interior point.
                hold_pick=search_clean_hold(
                    p_board,hold_u,tangent_board,lobe_bool,body_bool,lobe_quality,
                    preferred_tangent_mm=float(active_pick.get('tangent_mm',0.0)),
                    active_board_for_pair=active_board,
                    active_normal_mm=float(active_pick.get('normal_mm',active_target)),
                    desired_pair_mm=(82.0 if primary_fold_authoritative else pair_target),
                )
                if not isinstance(hold_pick,dict):
                    _e126_reject['hold']+=1
                    continue
                hold_board=np.asarray(hold_pick['board'],dtype=np.float64)
                hold_px=np.asarray(hold_pick['px'],dtype=np.float64)
                hx,hy=int(round(hold_px[0])),int(round(hold_px[1]))
                if not finite[hy,hx]: continue
                active_fold_support=float(active_pick.get('fold_region_support',local_primary_fold_support(active_px,11) if primary_fold_authoritative else 0.5))
                hold_fold_support=float(hold_pick.get('fold_region_support',local_primary_fold_support(hold_px,12) if primary_fold_authoritative else 0.0))
                if primary_fold_authoritative:
                    active_fold_min=float(getattr(args,'micro_unfold_e126_active_min_fold_support',0.16))
                    hold_fold_max=float(getattr(args,'micro_unfold_e126_hold_max_fold_support',0.50))
                    role_delta_min=float(getattr(args,'micro_unfold_e126_pair_min_fold_delta',0.05))
                    if active_fold_support<active_fold_min or hold_fold_support>hold_fold_max or (active_fold_support-hold_fold_support)<role_delta_min:
                        _e126_reject['fold_role']+=1
                        continue
                fold_region_role_support=float(np.clip(0.5+0.65*(active_fold_support-hold_fold_support),0.0,1.0)) if primary_fold_authoritative else 0.5
                active_extent=ray_extent_mm(p_board,active_u); hold_extent=ray_extent_mm(p_board,hold_u)
                _active_margin=0.5 if primary_fold_authoritative else 4.0
                _hold_margin=1.0 if primary_fold_authoritative else 4.0
                if active_extent<float(active_pick.get('normal_mm',active_target))+_active_margin or hold_extent<min(float(hold_pick['normal_mm']),hold_target)+_hold_margin:
                    _e126_reject['extent']+=1
                    continue
                sep=float(np.linalg.norm(active_board-hold_board))
                if not (pair_min<=sep<=pair_max):
                    _e126_reject['sep']+=1
                    continue
                same_fold_ok,same_fold_metrics=_e122_validate_same_fold_pair(
                    p_board,active_board,hold_board,active_u,tangent_board,args
                )
                # E126 region-first override: if POSE/UniDepth already owns the
                # folded region, a valid BODY->FOLD boundary crossing is more
                # relevant than perpendicularity to one short/noisy crease branch.
                if primary_fold_authoritative:
                    region_pair_ok,region_pair_metrics=_e126_validate_primary_fold_region_pair(
                        active_px,hold_px,primary_fold_region,primary_fold_boundary
                    )
                    if not region_pair_ok:
                        _e126_reject['region_pair']+=1
                        continue
                    if not same_fold_ok:
                        same_fold_metrics=dict(same_fold_metrics)
                        same_fold_metrics['e126_region_override']=True
                        same_fold_metrics['e126_region_pair']=region_pair_metrics
                        # Quality term only; physical safety is checked below.
                        same_fold_metrics['pair_normal_cos']=max(0.55,float(same_fold_metrics.get('pair_normal_cos',0.0)))
                        same_fold_ok=True
                elif not same_fold_ok:
                    _e126_reject['same_fold']+=1
                    continue

                # E115 direct UniDepth role evidence. The two physical candidate
                # locations are compared after geometry proposes them. The locally
                # higher/raised point should be ACTIVE; the lower/flatter point HOLD.
                active_depth_role=depth_role_metrics(active_px)
                hold_depth_role=depth_role_metrics(hold_px)
                depth_role_valid=bool(active_depth_role.get('valid',False) and hold_depth_role.get('valid',False))
                depth_role_delta=float(active_depth_role.get('raised_mm',0.0)-hold_depth_role.get('raised_mm',0.0)) if depth_role_valid else 0.0
                depth_role_scale=max(0.50,float(getattr(args,'micro_unfold_depth_role_scale_mm',2.0)))
                depth_role_conf=float(np.tanh(depth_role_delta/depth_role_scale)) if depth_role_valid else 0.0
                depth_role_support=float(0.5+0.5*depth_role_conf) if depth_role_valid else 0.5
                depth_role_strong=max(0.50,float(getattr(args,'micro_unfold_depth_role_strong_mm',1.5)))
                # E124: a nominated ACTIVE that is clearly lower than HOLD is
                # inconsistent with a raised folded flap. Only exceptionally
                # strong pose deformation may override this contradiction.
                hard_depth_contradiction=max(0.60,float(getattr(args,'micro_unfold_e124_depth_hard_contradiction_mm',1.20)))
                if depth_role_valid and depth_role_delta<=-hard_depth_contradiction and not (pose_deform_available and pose_deform_conf>=0.78):
                    depth_role_rejected += 1
                    _e126_reject['depth']+=1
                    continue
                role_probe_matches = bool(preferred_sign is None or float(active_sign)==float(preferred_sign))
                # E119: depth contradiction is only a mild penalty when topology
                # deformation clearly identifies the folded side.
                pose_deform_sign_matches = bool(deform_preferred_sign is None or float(active_sign)==float(deform_preferred_sign))
                role_mismatch_penalty = float(
                    0.0 if role_probe_matches else
                    ((0.035 + 0.035*probe_conf) if (deform_preferred_sign is not None and pose_deform_sign_matches)
                     else (0.10 + 0.08*probe_conf))
                )
                rescue_min_depth=max(depth_role_strong,float(getattr(args,'micro_unfold_sensitive_rescue_min_depth_mm',1.5)))
                rescue_min_span=max(24.0,float(getattr(args,'micro_unfold_sensitive_rescue_min_span_px',36.0)))
                deform_rescue_ok = bool(
                    pose_deform_available and pose_deform_conf>=0.42 and
                    pose_deform_sign_matches and pose_deform_anchor_support>=0.38 and span>=max(24.0,0.75*rescue_min_span)
                )
                rescue_evidence_ok = bool(
                    crease_source != 'SENSITIVE_RESCUE' or deform_rescue_ok or
                    (preferred_sign is not None and role_probe_matches and abs(probe_delta)>=rescue_min_depth and probe_conf>=0.55 and span>=rescue_min_span)
                )

                # Fold Crease + outer contour side ownership. This is a side vote, not a grasp-domain gate.
                if lobe_quality>0.0:
                    a_lobe=local_region_support(lobe_bool,active_px,10)
                    h_lobe=local_region_support(lobe_bool,hold_px,10)
                    a_body=local_region_support(body_bool,active_px,10)
                    h_body=local_region_support(body_bool,hold_px,10)
                    correct=0.5*(a_lobe+h_body)
                    reversed_=0.5*(a_body+h_lobe)
                    ownership=float(np.clip(0.5+0.5*(correct-reversed_),0.0,1.0))
                    lobe_support=(1.0-lobe_quality)*0.5+lobe_quality*ownership
                    # E115: even a clear coarse lobe is not allowed to hard-reject
                    # role orientation. It remains a structural side vote; direct
                    # UniDepth at the actual pair decides raised ACTIVE vs lower HOLD.
                else:
                    a_lobe=h_lobe=a_body=h_body=0.0
                    ownership=0.5; lobe_support=0.5

                # E118 dominant role cue: Fold Crease + cloth contour has already
                # split this component into two regions. The region containing fewer
                # learned anatomical pose landmarks is the folded/ACTIVE side.
                pose_split_available=bool(pose_split.get('available',False)) and lobe_quality>0.0
                pose_split_authoritative=bool(pose_split.get('authoritative',False)) and lobe_quality>0.0
                pose_split_conf=float(pose_split.get('confidence',0.0)) if pose_split_available else 0.0
                pose_split_pref=str(pose_split.get('preferred_active_region','NONE'))
                if pose_split_pref=='A':  # A == coarse lobe region
                    pose_count_correct=0.5*(a_lobe+h_body)
                    pose_count_reversed=0.5*(a_body+h_lobe)
                elif pose_split_pref=='B':  # B == coarse body region
                    pose_count_correct=0.5*(a_body+h_lobe)
                    pose_count_reversed=0.5*(a_lobe+h_body)
                else:
                    pose_count_correct=pose_count_reversed=0.5
                pose_count_role_raw=float(np.clip(0.5+0.5*(pose_count_correct-pose_count_reversed),0.0,1.0))
                pose_count_role_support=float((1.0-pose_split_conf)*0.5 + pose_split_conf*pose_count_role_raw) if pose_split_available else 0.5
                pose_count_signal=float(2.0*pose_count_role_support-1.0)
                # E119 keeps the E118 pose-count rule only as a tie-break.  A
                # partition can change dramatically when the selected crease changes,
                # so it must not overpower direct topology-deformation evidence.
                pose_count_mismatch_penalty=float(
                    (0.06+0.05*pose_split_conf) if (pose_split_authoritative and pose_count_role_raw<0.45) else 0.0
                )

                # E117 pose-fold side vote. If a boundary-semantic pose point has
                # moved deep inside the silhouette, the side of this crease that
                # contains that intrusion is more plausibly the folded/ACTIVE side.
                if pose_fold_available:
                    pose_active_raw=local_region_support(pose_fold_bool,active_px,24)
                    pose_hold_raw=local_region_support(pose_fold_bool,hold_px,24)
                    pose_fold_role_support=float(np.clip(0.5+2.2*(pose_active_raw-pose_hold_raw),0.0,1.0))
                else:
                    pose_active_raw=0.0; pose_hold_raw=0.0; pose_fold_role_support=0.5

                # E119 primary folded-side evidence: ACTIVE should lie inside the
                # region highlighted by the deformed learned garment topology,
                # while HOLD should remain on the cleaner non-deformed BODY side.
                if pose_deform_available:
                    pose_deform_active_raw=local_region_support(pose_deform_bool,active_px,28)
                    pose_deform_hold_raw=local_region_support(pose_deform_bool,hold_px,28)
                    pose_deform_role_support=float(np.clip(
                        0.5 + 2.8*pose_deform_conf*(pose_deform_active_raw-pose_deform_hold_raw),
                        0.0, 1.0
                    ))
                else:
                    pose_deform_active_raw=0.0; pose_deform_hold_raw=0.0; pose_deform_role_support=0.5
                pose_deform_signal=float(2.0*pose_deform_role_support-1.0)
                pose_deform_authoritative=bool(
                    pose_deform_available and pose_deform_conf>=0.38 and
                    (abs(deform_probe_delta)>=0.035 or abs(pose_deform_signal)>=0.10)
                )

                # BODY orientation secondary evidence: Pose first, then centroid and side extent.
                pose_body=local_pose_support(hold_px)-local_pose_support(active_px)
                d_hold=float(np.linalg.norm(hold_board-gc_board)); d_active=float(np.linalg.norm(active_board-gc_board))
                centroid_body=(d_active-d_hold)/max(40.0,pair_target)
                extent_body=(hold_extent-active_extent)/80.0
                body_raw=3.8*pose_body+1.05*centroid_body+0.65*extent_body
                body_support=float(1.0/(1.0+np.exp(-np.clip(body_raw,-6.0,6.0))))

                # E129: FUSED FOLD is diagnostic-only whenever pose intrusion
                # already identified the folded anatomical region. It gets ZERO
                # ranking authority in that case.
                if _e129_pose_active_authority:
                    fused_support=0.50
                    fused_quality_weight=0.0
                else:
                    active_fused_dist=float(fused_dist_px[ay,ax])
                    anchor_fused_dist=float(fused_dist_px[py0,px0])
                    fused_support=float(np.exp(-min(active_fused_dist,0.70*anchor_fused_dist)/32.0)) if np.any(fused) else 0.45
                    if fused[ay,ax]: fused_support=min(1.0,fused_support+0.10)
                    fused_quality_weight=0.01

                # Old CREASE FLAP REGION is diagnostic only; it no longer contributes to E117 quality.
                flap_soft=0.5
                if isinstance(soft_part,dict):
                    agreement=0.5*(1.0 if soft_flap[ay,ax] else 0.0)+0.5*(1.0 if soft_body[hy,hx] else 0.0)
                    trust=float(np.clip(comp_selected_overlap/0.35,0.0,1.0))
                    flap_soft=(1.0-trust)*0.5+trust*agreement

                hold_clean=float(hold_pick['score'])
                hold_clean_metrics_v=dict(hold_pick.get('metrics',{}) or {})
                flat_support=float(hold_clean_metrics_v.get('flat_support',0.0))
                # Compare BODY cleanliness in both possible side assignments. A small-lobe
                # vote may be wrong on asymmetric/multi-layer cloth; a clearly cleaner HOLD
                # side is allowed to overturn it.
                opposite_hold=search_clean_hold(p_board,-hold_u,tangent_board,lobe_bool,body_bool,lobe_quality)
                opposite_clean=float(opposite_hold['score']) if isinstance(opposite_hold,dict) else 0.0
                clean_adv=float(np.clip(hold_clean-opposite_clean,-1.0,1.0))
                lobe_signal=float((2.0*ownership-1.0)*lobe_quality)
                body_signal=float(2.0*body_support-1.0)
                geometry_orientation=float(np.clip(0.25*lobe_signal+0.25*body_signal+0.50*clean_adv,-1.0,1.0))
                min_orient=float(getattr(args,'micro_unfold_orientation_min_confidence',0.05))
                if pose_deform_authoritative:
                    # E119 primary hierarchy: topology deformation -> crease/clean
                    # geometry -> pose-count tie-break -> UniDepth weak confirmation.
                    deform_w=float(np.clip(0.58+0.12*(pose_deform_conf-0.38),0.56,0.68))
                    count_w=0.08 if pose_split_authoritative else 0.03
                    depth_w=0.18 if depth_role_valid else 0.0
                    geom_w=max(0.08,1.0-deform_w-count_w-depth_w)
                    norm=max(1e-6,deform_w+count_w+depth_w+geom_w)
                    orientation_conf=float(np.clip(
                        (deform_w*pose_deform_signal + count_w*pose_count_signal +
                         depth_w*depth_role_conf + geom_w*geometry_orientation)/norm,
                        -1.0,1.0
                    ))
                elif pose_split_authoritative:
                    # E118 pose-count logic survives only as a fallback.  It no
                    # longer has 72% authority because region membership changed
                    # too easily when the wrong crease component was selected.
                    pose_w=0.28
                    depth_w=0.25 if depth_role_valid else 0.0
                    geom_w=max(0.0,1.0-pose_w-depth_w)
                    orientation_conf=float(np.clip(
                        pose_w*pose_count_signal + depth_w*depth_role_conf + geom_w*geometry_orientation,
                        -1.0,1.0
                    ))
                elif depth_role_valid:
                    depth_w=0.50
                    orientation_conf=float(np.clip(depth_w*depth_role_conf+(1.0-depth_w)*geometry_orientation,-1.0,1.0))
                    if depth_role_delta>=depth_role_strong:
                        orientation_conf=max(orientation_conf,min_orient+0.02)
                else:
                    depth_w=0.0
                    orientation_conf=geometry_orientation
                orientation_ok = bool(orientation_conf >= min_orient)
                if not orientation_ok:
                    orientation_rejected += 1

                span_support=float(np.clip(span/170.0,0.0,1.0))
                strength_support=float(np.clip(comp_strength/1.20,0.0,1.0))
                crease_shape=0.58*span_support+0.34*strength_support+0.08*linearity
                pair_support=float(np.exp(-abs(sep-pair_target)/18.0))

                pull_u=(active_board-hold_board)/max(1e-6,sep)
                try:
                    if cfg is not None:
                        pull_board,used_pull,garment_exit_mm,outside_used,pull_px,pull_u,hold_boundary_mm,boundary_normal_delta_deg,boundary_normal_ok=_e111_find_outer_normal_pull_endpoint(
                            mask,H_value,cfg,active_board,hold_board,pull_u,args,log_final=False,
                            stable_mask_cached=_e128_e111_stable, outer_contour_cached=outer_pts, invH_cached=_e128_invH)
                    else:
                        used_pull=max(80.0,float(getattr(args,'micro_unfold_pull_mm',30.0))*float(getattr(args,'micro_unfold_pull_scale',1.20)))
                        pull_board=active_board+pull_u*used_pull; garment_exit_mm=0.0; outside_used=0.0; pull_px=_fix89_board_to_px(H_value,pull_board)
                        hold_boundary_mm=float(used_pull)/2.0; boundary_normal_delta_deg=0.0; boundary_normal_ok=False
                except Exception: continue
                feasible,role_hint=role_precheck(active_board,hold_board,pull_board)
                if not feasible:
                    _e126_reject['reach_path']+=1
                    continue
                reach_support=float(role_hint.get('reach_support',1.0)) if isinstance(role_hint,dict) else 1.0

                # E119: the selected crease must be geometrically plausible AND
                # spatially agree with deformation of the learned 12-point garment
                # topology. Pose-count is only a small fallback; FUSED remains 1%.
                pose_split_structure=float(pose_split_conf if pose_split_available else 0.0)
                interiority=float(np.clip(0.45*min(1.0,comp_interior_median/18.0)+0.55*comp_interior_frac,0.0,1.0))
                quality=(
                    0.14*crease_shape +
                    0.05*lobe_support +
                    0.15*comp_boundary_support +
                    0.17*fold_region_role_support +
                    0.07*pose_deform_anchor_support +
                    0.08*pose_deform_role_support +
                    0.09*pose_axis_support +
                    0.05*interiority +
                    0.02*pose_split_structure +
                    0.02*pose_count_role_support +
                    0.02*pose_fold_support +
                    0.02*pose_fold_role_support +
                    0.04*body_support +
                    fused_quality_weight*fused_support +
                    (0.12*float(active_pick.get('pose_depth_authority',0.0)) if _e129_pose_active_authority else 0.0) +
                    0.02*surface_asym +
                    0.05*hold_clean +
                    0.07*float(active_pick.get('score',0.0)) +
                    0.04*float(same_fold_metrics.get('pair_normal_cos',0.0)) +
                    0.09*comp_unidepth_support +
                    0.08*depth_role_support +
                    0.02*pair_support +
                    0.05*max(0.0,orientation_conf) -
                    role_mismatch_penalty -
                    pose_count_mismatch_penalty -
                    ((0.010 if deform_rescue_ok else 0.045) if str(crease_source).startswith('SENSITIVE') else 0.0) -
                    seam_penalty -
                    0.035*(1.0-reach_support) -
                    0.00020*float(role_hint.get('travel',0.0) if isinstance(role_hint,dict) else 0.0)
                )
                candidates.append({
                    'quality':float(quality),'crease_source':str(crease_source),'active_px':np.asarray(active_px,dtype=np.float64),'hold_px':np.asarray(hold_px,dtype=np.float64),
                    'primary_fold_region_source':str(primary_fold_source),'primary_fold_region_confidence':float(primary_fold_conf),
                    'primary_fold_region_group':str(primary_fold_group),'primary_fold_region_authoritative':bool(primary_fold_authoritative),
                    'active_primary_fold_support':float(active_fold_support),'hold_primary_fold_support':float(hold_fold_support),
                    'fold_region_role_support':float(fold_region_role_support),'fold_boundary_component_support':float(comp_boundary_support),
                    'fold_region_component_overlap':float(comp_fold_overlap),'fold_region_component_deep_overlap':float(comp_deep_overlap),
                    'garment_center_px':gc.copy(),'active_board':np.asarray(active_board,dtype=np.float64),'hold_board':np.asarray(hold_board,dtype=np.float64),
                    'crease_anchor_px':np.asarray(p,dtype=np.float64),'crease_tangent_board':tangent_board.copy(),'crease_normal_board':active_u.copy(),
                    'component_label':int(lab),'component_span_px':float(span),'component_strength':float(comp_strength),'local_linearity':float(linearity),
                    'fused_support':float(fused_support),'fused_quality_weight':float(fused_quality_weight),
                    'active_selection_mode':str(active_pick.get('selection_mode','CREASE_LOCAL_ACTIVE')),
                    'active_pose_depth_authority':float(active_pick.get('pose_depth_authority',0.0)),
                    'active_pose_region_raised_mm':float(active_pick.get('raised_mm',0.0)),
                    'body_side_score':float(body_raw),'body_support':float(body_support),
                    'coarse_lobe_support':float(lobe_support),'coarse_lobe_quality':float(lobe_quality),'coarse_lobe_ownership':float(ownership),
                    'coarse_lobe_ratio':float(lobe_ratio),'surface_asymmetry':float(surface_asym),'seam_like':float(seam_like),'seam_penalty':float(seam_penalty),
                    'reach_support':float(reach_support),'crease_flap_soft_support':float(flap_soft),
                    'pose_fold_support':float(pose_fold_support),'pose_fold_intrusion_available':bool(pose_fold_available),
                    'pose_fold_role_support':float(pose_fold_role_support),'pose_fold_active_raw':float(pose_active_raw),'pose_fold_hold_raw':float(pose_hold_raw),
                    'pose_deform_available':bool(pose_deform_available),'pose_deform_confidence':float(pose_deform_conf),
                    'pose_deform_mode':str(pose_deform_mode),'pose_deform_side_hint':str(pose_deform_side_hint),
                    'pose_deform_anchor_support':float(pose_deform_anchor_support),
                    'pose_deform_role_support':float(pose_deform_role_support),
                    'pose_deform_active_raw':float(pose_deform_active_raw),'pose_deform_hold_raw':float(pose_deform_hold_raw),
                    'pose_deform_authoritative':bool(pose_deform_authoritative),
                    'pose_deform_probe_delta':float(deform_probe_delta),
                    'pose_deform_preferred_sign':None if deform_preferred_sign is None else float(deform_preferred_sign),
                    'pose_fold_region_hint':str(pose_fold_region_hint),'pose_fold_region_confidence':float(pose_fold_region_conf),
                    'pose_direction_alignment':float(pose_direction_alignment),'pose_axis_progress':float(pose_axis_progress),'pose_axis_support':float(pose_axis_support),
                    'component_interior_median_px':float(comp_interior_median),'component_interior_fraction':float(comp_interior_frac),
                    'neck_segment_distance_px':float(neck_dist),'neck_parallel':float(neck_parallel),'neck_candidate_penalty':float(neck_candidate_penalty),
                    'pose_split_available':bool(pose_split_available),'pose_split_authoritative':bool(pose_split_authoritative),
                    'pose_split_lobe_count':int(pose_split.get('count_a',0)),'pose_split_body_count':int(pose_split.get('count_b',0)),
                    'pose_split_assigned':int(pose_split.get('assigned',0)),'pose_split_gap':int(pose_split.get('gap',0)),
                    'pose_split_confidence':float(pose_split_conf),'pose_split_preferred_active_region':str(pose_split_pref),
                    'pose_count_role_support':float(pose_count_role_support),'pose_count_role_raw':float(pose_count_role_raw),
                    'pose_count_mismatch_penalty':float(pose_count_mismatch_penalty),
                    'depth_role_probe_valid':bool(probe_valid),'depth_role_probe_delta_mm':float(probe_delta),'depth_role_probe_confidence':float(probe_conf),
                    'depth_role_probe_preferred_sign':None if preferred_sign is None else float(preferred_sign),'depth_role_probe_matches':bool(role_probe_matches),
                    'role_mismatch_penalty':float(role_mismatch_penalty),'orientation_evidence_ok':bool(orientation_ok),'rescue_evidence_ok':bool(rescue_evidence_ok),
                    'hold_clean_score':float(hold_clean),'hold_crease_clear_px':float(hold_clean_metrics_v.get('crease_clear_px',0.0)),
                    'hold_crease_density':float(hold_clean_metrics_v.get('crease_density',1.0)),'hold_clean_advantage':float(clean_adv),
                    'hold_search_tangent_mm':float(hold_pick.get('tangent_mm',0.0)),
                    'e128_pair_conditioned':bool(hold_pick.get('e128_pair_conditioned',False)),
                    'e128_pair_target_mm':float(hold_pick.get('e128_pair_target_mm',82.0 if primary_fold_authoritative else pair_target)),
                    'e128_pair_search_sep_mm':float(hold_pick.get('e128_pair_sep_mm',sep) if hold_pick.get('e128_pair_sep_mm',None) is not None else sep),
                    'e128_pair_approach_clear_mm':float(hold_pick.get('e128_approach_clear_mm',999.0)),
                    'orientation_confidence':float(orientation_conf),
                    'e122_active_search_score':float(active_pick.get('score',0.0)),'e122_active_search_tangent_mm':float(active_pick.get('tangent_mm',0.0)),
                    'e122_branch_fold_score':float(comp_strength),'e124_branch_unidepth_support':float(comp_unidepth_support),
                    'e122_same_fold_gate':dict(same_fold_metrics),
                    'geometry_orientation_confidence':float(geometry_orientation),'depth_role_valid':bool(depth_role_valid),
                    'active_depth_role_mm':float(active_depth_role.get('raised_mm',0.0)),'hold_depth_role_mm':float(hold_depth_role.get('raised_mm',0.0)),
                    'depth_role_delta_mm':float(depth_role_delta),'depth_role_confidence':float(depth_role_conf),'depth_role_support':float(depth_role_support),
                    'active_depth_role_relief_med_mm':float(active_depth_role.get('relief_med_mm',0.0)),'hold_depth_role_relief_med_mm':float(hold_depth_role.get('relief_med_mm',0.0)),
                    'active_depth_role_prom_p70_mm':float(active_depth_role.get('prom_p70_mm',0.0)),'hold_depth_role_prom_p70_mm':float(hold_depth_role.get('prom_p70_mm',0.0)),
                    'active_crease_distance_mm':float(active_pick.get('normal_mm',active_target)),'hold_crease_distance_mm':float(hold_pick.get('normal_mm',hold_target)),
                    'pair_separation_mm':float(sep),'pair_spacing_mode':'E109_LOCAL_CREASE_NORMAL','outer_boundary_perpendicular_angle_deg':90.0,
                    'hold_to_active_outward_cos':1.0,'hold_normal_target_error_mm':0.0,'active_depth_reference':float(fused_support),
                    'active_mask_extent_mm':float(active_extent),'hold_mask_extent_mm':float(hold_extent),'pose_body_evidence':float(pose_body),
                    'centroid_body_evidence':float(centroid_body),'mask_extent_body_evidence':float(extent_body),'hold_band_mode':'DIRECT_40MM_FROM_LOCAL_CREASE',
                    'outer_boundary_mode':'FOLD_CREASE_PLUS_OUTER_LOBE_PRIMARY','active_inside_flap':bool(soft_flap[ay,ax]) if isinstance(soft_part,dict) else False,
                    'hold_inside_body':bool(soft_body[hy,hx]) if isinstance(soft_part,dict) else False,
                    'prechecked_pull_board':np.asarray(pull_board,dtype=np.float64),'prechecked_pull_distance_mm':float(used_pull),
                    'prechecked_garment_exit_mm':float(garment_exit_mm),'prechecked_outside_used_mm':float(outside_used),'prechecked_pull_px':np.asarray(pull_px,dtype=np.float64),
                    'prechecked_role_hint':role_hint,
                    'prechecked_pull_unit_board':np.asarray(pull_u,dtype=np.float64),
                    'prechecked_hold_boundary_distance_mm':float(hold_boundary_mm),
                    'prechecked_boundary_normal_delta_deg':float(boundary_normal_delta_deg),
                    'prechecked_boundary_normal_ok':bool(boundary_normal_ok),
                    'coarse_lobe_region':np.asarray(lobe_info.get('lobe_region',np.zeros_like(mask)),dtype=np.uint8).copy(),
                    'coarse_body_region':np.asarray(lobe_info.get('body_region',np.zeros_like(mask)),dtype=np.uint8).copy(),
                    'selected_crease_component':(comp_mask.astype(np.uint8)*255).copy(),
                    'primary_fold_region':primary_fold_region.copy(),
                    'primary_fold_boundary':primary_fold_boundary.copy(),
                    'pose_deform_region':pose_deform_mask.copy(),
                    'crease_flap_region':np.asarray(soft_part.get('flap_region',np.zeros_like(mask)),dtype=np.uint8).copy() if isinstance(soft_part,dict) else np.zeros_like(mask,dtype=np.uint8),
                    'crease_body_region':np.asarray(soft_part.get('body_region',np.zeros_like(mask)),dtype=np.uint8).copy() if isinstance(soft_part,dict) else np.zeros_like(mask,dtype=np.uint8),
                    'crease_partition_info':({'mode':'E109_OLD_FLAP_2PCT_SOFT_REF'} if isinstance(soft_part,dict) else {'mode':'E109_NO_OLD_FLAP_REF'}),
                })

    if not candidates and (not primary_fold_authoritative) and pose_fold_dir_board is not None and pose_fold_region_conf>=0.18:
        # E120 fallback requested by the operator: when the fold direction is known
        # from pose but a robust selected crease pair cannot be built, still publish
        # HOLD/ACTIVE on one straight unfold axis. Prefer a nearby SENSITIVE crease;
        # otherwise use a virtual boundary and keep X blocked.
        try:
            fold_u=np.asarray(pose_fold_dir_board,dtype=np.float64)
            fold_u/=max(1e-9,float(np.linalg.norm(fold_u)))
            extent=max(80.0,ray_extent_mm(gc_board,fold_u))
            virtual_dist=float(np.clip(0.48*extent,45.0,max(45.0,extent-48.0)))
            virtual_board=gc_board+fold_u*virtual_dist
            virtual_px=_fix89_board_to_px(H_value,virtual_board)
            p_board=np.asarray(virtual_board,dtype=np.float64); p=np.asarray(virtual_px,dtype=np.float64)
            real_crease=False; real_exec_crease=False; selected_comp=np.zeros_like(mask,dtype=np.uint8); nearest_px_dist=999.0
            _pose_axis_pool=_exec if cv2.countNonZero(_exec)>=8 else _preview
            if cv2.countNonZero(_pose_axis_pool)>=8:
                ys,xs=np.where(_pose_axis_pool>0)
                if xs.size:
                    cpx=np.column_stack([xs,ys]).astype(np.float64)
                    # Filter obvious neckline candidates before nearest-axis choice.
                    valid=[]
                    for cp in cpx:
                        cx=int(np.clip(round(cp[0]),0,w-1)); cy=int(np.clip(round(cp[1]),0,h-1))
                        if float(neck_dist_map[cy,cx])<11.0: continue
                        try:
                            cb=_fix53_px_to_board(H_value,cp)
                            along=float(np.dot(cb-gc_board,fold_u))
                            perp=float(np.linalg.norm((cb-gc_board)-fold_u*along))
                            if 0.10*extent<=along<=0.90*extent and perp<=120.0:
                                valid.append((abs(along-virtual_dist)+0.30*perp,cp,cb))
                        except Exception: pass
                    if valid:
                        valid.sort(key=lambda z:z[0]); _,cp,cb=valid[0]
                        nearest_px_dist=float(np.linalg.norm(cp-virtual_px))
                        if nearest_px_dist<=95.0:
                            p=np.asarray(cp,dtype=np.float64); p_board=np.asarray(cb,dtype=np.float64); real_crease=True
                            _cxi=int(np.clip(round(float(p[0])),0,w-1)); _cyi=int(np.clip(round(float(p[1])),0,h-1))
                            real_exec_crease=bool(cv2.countNonZero(_exec)>=8 and _exec[_cyi,_cxi]>0)
                            cv2.circle(selected_comp,tuple(np.rint(p).astype(int)),8,255,-1,cv2.LINE_AA)
            active_board=p_board+fold_u*active_target
            hold_board=p_board-fold_u*hold_target
            active_px=_fix89_board_to_px(H_value,active_board); hold_px=_fix89_board_to_px(H_value,hold_board)
            if px_inside(active_px,4.0) and px_inside(hold_px,4.0):
                sep=float(np.linalg.norm(active_board-hold_board))
                pull_board,used_pull,garment_exit_mm,outside_used,pull_px,pull_u,hold_boundary_mm,bnd_ang,bnd_ok=_e111_find_outer_normal_pull_endpoint(
                    mask,H_value,cfg,active_board,hold_board,fold_u,args,log_final=False,
                    stable_mask_cached=_e128_e111_stable, outer_contour_cached=outer_pts, invH_cached=_e128_invH
                ) if cfg is not None else (active_board+fold_u*120.0,120.0,0.0,0.0,active_px+np.asarray([1.0,0.0]),fold_u,60.0,0.0,False)
                feasible,role_hint=role_precheck(active_board,hold_board,pull_board)
                if feasible:
                    hc=hold_clean_metrics(hold_px)
                    crease_support=1.0 if real_crease else 0.0
                    exec_ok=bool(real_exec_crease and pose_fold_region_conf>=0.50 and float(hc.get('score',0.0))>=0.28)
                    fallback={
                        'quality':float(0.20+0.34*pose_fold_region_conf+0.18*crease_support+0.12*float(hc.get('score',0.0))),
                        'crease_source':'POSE_GUIDED_SENSITIVE_AXIS' if real_crease else 'POSE_GUIDED_VIRTUAL_AXIS',
                        'active_px':np.asarray(active_px,dtype=np.float64),'hold_px':np.asarray(hold_px,dtype=np.float64),
                        'garment_center_px':gc.copy(),'active_board':np.asarray(active_board,dtype=np.float64),'hold_board':np.asarray(hold_board,dtype=np.float64),
                        'crease_anchor_px':np.asarray(p,dtype=np.float64),'component_label':-1,'component_span_px':0.0,'component_strength':0.0,
                        'fused_support':0.0,'body_support':0.5,'coarse_lobe_support':0.5,'coarse_lobe_quality':0.0,'surface_asymmetry':0.0,'seam_penalty':0.0,
                        'reach_support':float(role_hint.get('reach_support',1.0)) if isinstance(role_hint,dict) else 1.0,'crease_flap_soft_support':0.5,
                        'pose_fold_support':0.5,'pose_fold_role_support':0.5,
                        'pose_deform_available':bool(pose_deform_available),'pose_deform_confidence':float(pose_deform_conf),
                        'pose_deform_mode':str(pose_deform_mode),'pose_deform_side_hint':str(pose_deform_side_hint),
                        'pose_deform_anchor_support':1.0,'pose_deform_role_support':1.0,'pose_deform_authoritative':True,
                        'pose_fold_region_hint':str(pose_fold_region_hint),'pose_fold_region_confidence':float(pose_fold_region_conf),
                        'pose_direction_alignment':1.0,'pose_axis_progress':float(np.clip(virtual_dist/max(1.0,extent),0.0,1.0)),'pose_axis_support':1.0,
                        'pose_split_lobe_count':0,'pose_split_body_count':0,'pose_split_assigned':0,'pose_split_gap':0,'pose_split_confidence':0.0,
                        'pose_split_preferred_active_region':'NONE','pose_split_authoritative':False,'pose_count_role_support':0.5,
                        'depth_role_probe_valid':False,'depth_role_probe_delta_mm':0.0,'depth_role_probe_confidence':0.0,
                        'orientation_evidence_ok':True,'rescue_evidence_ok':bool(real_crease),'orientation_confidence':float(0.45+0.45*pose_fold_region_conf),
                        'hold_clean_score':float(hc.get('score',0.0)),'hold_crease_clear_px':float(hc.get('crease_clear_px',0.0)),
                        'hold_crease_density':float(hc.get('crease_density',1.0)),'hold_clean_advantage':0.0,
                        'depth_role_valid':False,'active_depth_role_mm':0.0,'hold_depth_role_mm':0.0,'depth_role_delta_mm':0.0,'depth_role_confidence':0.0,
                        'active_crease_distance_mm':float(active_target),'hold_crease_distance_mm':float(hold_target),'pair_separation_mm':float(sep),
                        'pair_spacing_mode':'E120_POSE_GUIDED_AXIS','outer_boundary_perpendicular_angle_deg':90.0,'hold_to_active_outward_cos':1.0,
                        'hold_normal_target_error_mm':0.0,'active_depth_reference':0.0,'hold_band_mode':'POSE_GUIDED_OPPOSITE_SIDE',
                        'outer_boundary_mode':'POSE_QUADRANT_DIRECTION','active_inside_flap':False,'hold_inside_body':True,
                        'prechecked_pull_board':np.asarray(pull_board,dtype=np.float64),'prechecked_pull_distance_mm':float(used_pull),
                        'prechecked_garment_exit_mm':float(garment_exit_mm),'prechecked_outside_used_mm':float(outside_used),'prechecked_pull_px':np.asarray(pull_px,dtype=np.float64),
                        'prechecked_role_hint':role_hint,'prechecked_pull_unit_board':np.asarray(pull_u,dtype=np.float64),
                        'prechecked_hold_boundary_distance_mm':float(hold_boundary_mm),'prechecked_boundary_normal_delta_deg':float(bnd_ang),'prechecked_boundary_normal_ok':bool(bnd_ok),
                        'coarse_lobe_region':np.zeros_like(mask,dtype=np.uint8),'coarse_body_region':np.zeros_like(mask,dtype=np.uint8),
                        'selected_crease_component':selected_comp,'pose_deform_region':pose_deform_mask.copy(),
                        'crease_flap_region':np.zeros_like(mask,dtype=np.uint8),'crease_body_region':np.zeros_like(mask,dtype=np.uint8),
                        'crease_partition_info':{'mode':'E120_POSE_GUIDED_AXIS','real_sensitive_crease':bool(real_crease),'nearest_px':float(nearest_px_dist)},
                        'execution_evidence_ok':bool(exec_ok),
                        'execution_evidence_reason':'OK' if exec_ok else ('E120_POSE_AXIS_PREVIEW_ONLY_NO_RELIABLE_CREASE' if not real_crease else 'E120_POSE_AXIS_PREVIEW_ONLY_WEAK_SUPPORT'),
                    }
                    print(f"[E122-POSE-AXIS-FALLBACK] region={pose_fold_region_hint}:{pose_fold_region_conf:.2f} qualifiedCrease={real_crease} execCrease={real_exec_crease} nearest={nearest_px_dist:.1f}px exec={exec_ok}")
                    return fallback
        except Exception as exc:
            print(f"[E120-POSE-AXIS-FALLBACK-WARN] {exc!r}")
    if not candidates:
        print('[E128-PAIR-REJECT] '+ ' '.join(f'{k}={v}' for k,v in _e126_reject.items()))
        return None
    print('[E128-PAIR-REJECT] '+ ' '.join(f'{k}={v}' for k,v in _e126_reject.items()) + f' accepted={len(candidates)}')

    # V12: finish/validate every mechanically-safe retained candidate BEFORE
    # selecting one.  This removes the V11 failure mode where the highest raw
    # quality crease was selected first and then failed the final exact-crease
    # role check while a valid alternative already existed in ``candidates``.
    min_q=float(getattr(args,'micro_unfold_e116_min_plan_quality',0.22))
    max_span=max(1.0,max(float(q.get('component_span_px',0.0) or 0.0) for q in candidates))
    garment_dist=cv2.distanceTransform((mask>0).astype(np.uint8),cv2.DIST_L2,5).astype(np.float32)

    def _v12_candidate_evidence(q):
        weak_depth=abs(float(q.get('depth_role_probe_confidence',q.get('depth_role_confidence',0.0)))) < 0.25
        weak_lobe=float(q.get('coarse_lobe_quality',0.0)) < 0.20
        strong_deform=bool(
            q.get('pose_deform_authoritative',False) and
            float(q.get('pose_deform_confidence',0.0))>=0.38 and
            float(q.get('pose_deform_anchor_support',0.0))>=0.34
        )
        strong_primary_region=bool(
            q.get('primary_fold_region_authoritative',False) and
            float(q.get('primary_fold_region_confidence',0.0))>=0.40 and
            float(q.get('active_primary_fold_support',0.0))>=0.24 and
            float(q.get('hold_primary_fold_support',1.0))<=0.48 and
            float(q.get('fold_region_role_support',0.0))>=0.54
        )
        src=str(q.get('crease_source',''))
        if src.startswith('E125_VIRTUAL'):
            branch_exec=bool(strong_primary_region and float(q.get('fold_boundary_component_support',0.0))>=0.20)
        else:
            branch_exec=not src.startswith('E122_PREVIEW')
        preview_rescue=bool(
            (not branch_exec) and
            float(q.get('e122_branch_fold_score',0.0))>=0.30 and
            float(q.get('quality',0.0))>=max(0.24,min_q) and
            bool(q.get('orientation_evidence_ok',True)) and
            bool(q.get('rescue_evidence_ok',True)) and
            (strong_primary_region or strong_deform or abs(float(q.get('depth_role_probe_confidence',0.0)))>=0.35 or float(q.get('coarse_lobe_quality',0.0))>=0.24)
        )
        ok=bool(
            (branch_exec or preview_rescue) and
            float(q.get('quality',0.0))>=min_q and
            bool(q.get('orientation_evidence_ok',True)) and
            bool(q.get('rescue_evidence_ok',True)) and
            (strong_primary_region or strong_deform or not (weak_depth and weak_lobe))
        )
        return ok,branch_exec,preview_rescue

    def _v12_crease_relative_geometry(q):
        cm=np.asarray(q.get('selected_crease_component',np.zeros_like(mask)),dtype=np.uint8)
        anchor=np.asarray(q.get('crease_anchor_px',[np.nan,np.nan]),dtype=np.float64).reshape(2)
        center_support=0.0
        line_board=None
        line_px=None
        normal_px=_v6_unit_xy(np.asarray(q.get('active_px'),dtype=np.float64)-np.asarray(q.get('hold_px'),dtype=np.float64))
        try:
            yy,xx=np.where(cm>0)
            if xx.size>=6 and np.all(np.isfinite(anchor)):
                pts=np.column_stack([xx,yy]).astype(np.float64)
                tangent=_v6_branch_tangent(cm)
                if tangent is not None:
                    proj=pts@tangent
                    lo=float(np.percentile(proj,5.0)); hi=float(np.percentile(proj,95.0))
                    ap=float(anchor@tangent); mid=0.5*(lo+hi); half=max(1.0,0.5*(hi-lo))
                    center_support=float(np.clip(1.0-abs(ap-mid)/half,0.0,1.0))
                    p0=np.mean(pts,axis=0)+tangent*(lo-float(np.mean(pts,axis=0)@tangent))
                    p1=np.mean(pts,axis=0)+tangent*(hi-float(np.mean(pts,axis=0)@tangent))
                    line_px=[np.asarray(p0,dtype=np.float64).reshape(2).tolist(),np.asarray(p1,dtype=np.float64).reshape(2).tolist()]
                    try:
                        b0=_fix53_px_to_board(H_value,p0); b1=_fix53_px_to_board(H_value,p1)
                        line_board=[np.asarray(b0,dtype=np.float64).reshape(2).tolist(),np.asarray(b1,dtype=np.float64).reshape(2).tolist()]
                    except Exception:
                        line_board=None
        except Exception:
            pass
        try:
            ap=np.asarray(q.get('active_px'),dtype=np.float64).reshape(2)
            ax=int(np.clip(round(float(ap[0])),0,mask.shape[1]-1)); ay=int(np.clip(round(float(ap[1])),0,mask.shape[0]-1))
            # Relative to the current garment only; this is a ranking modifier,
            # not a new hard threshold.
            vals=garment_dist[mask>0]
            ref=float(np.percentile(vals,75.0)) if vals.size else 1.0
            interior=float(np.clip(float(garment_dist[ay,ax])/max(1.0,ref),0.0,1.0))
        except Exception:
            interior=0.0
        return center_support,interior,line_board,line_px,normal_px

    evaluated=[]
    for idx,q0 in enumerate(candidates):
        q=q0
        evidence_ok,branch_exec,preview_rescue=_v12_candidate_evidence(q)
        q['execution_evidence_ok']=bool(evidence_ok)
        q['execution_evidence_reason']='OK' if evidence_ok else (
            f"E126_PREVIEW_ONLY branchExec={branch_exec} previewPairRescue={preview_rescue} "
            f"branchScore={float(q.get('e122_branch_fold_score',0.0)):.3f} Q={float(q.get('quality',0.0)):.3f}"
        )
        center_support,interior_support,line_board,line_px,normal_px=_v12_crease_relative_geometry(q)
        if normal_px is not None:
            q['v12_crease_normal_px']=np.asarray(normal_px,dtype=np.float64)
        if line_board is not None:
            q['v12_selected_crease_line_board']=line_board
        if line_px is not None:
            q['v13_selected_crease_line_px']=line_px
        q['v12_crease_center_support']=float(center_support)
        q['v12_active_interior_support']=float(interior_support)
        q['v12_crease_span_relative']=float(np.clip(float(q.get('component_span_px',0.0) or 0.0)/max_span,0.0,1.0))
        role_ok=False; role_diag={'reason':'EVIDENCE_NOT_READY'}
        if evidence_ok:
            try:
                role_ok,role_diag=_v7_validate_joint_plan_roles(q,depth_state)
            except Exception as exc:
                role_ok=False; role_diag={'reason':f'V12_ROLE_EXCEPTION:{exc!r}'}
        q['v12_candidate_role_ok']=bool(role_ok)
        q['v12_candidate_role_validation']=_training_json_safe(role_diag)
        # Keep the established E109 quality dominant. The V12 modifiers are all
        # relative/current-garment terms and only stabilize ties: longer coherent
        # crease, anchor away from endpoints, ACTIVE away from silhouette, reach margin.
        modifier=(
            0.65 +
            0.15*float(q['v12_crease_span_relative']) +
            0.10*float(center_support) +
            0.05*float(interior_support) +
            0.05*float(np.clip(q.get('reach_support',1.0),0.0,1.0))
        )
        q['v12_plan_score']=float(max(0.0,float(q.get('quality',0.0)))*modifier)
        if evidence_ok and role_ok:
            evaluated.append(q)
        print(
            f"[E156-V12-PLAN-CAND] i={idx+1}/{len(candidates)} src={q.get('crease_source')} "
            f"Q={float(q.get('quality',0.0)):.3f} planScore={float(q['v12_plan_score']):.3f} "
            f"spanRel={float(q['v12_crease_span_relative']):.2f} center={float(center_support):.2f} "
            f"interior={float(interior_support):.2f} reach={float(q.get('reach_support',1.0)):.2f} "
            f"evidence={bool(evidence_ok)} role={bool(role_ok)} reason={role_diag.get('reason')}"
        )

    # V13 compact candidate provenance for MICRO target-fold data collection.
    # No full-resolution masks escape the core; only click/ranking-friendly points.
    _v13_candidate_summaries=[]
    for _ci,_cq in enumerate(candidates,1):
        _v13_candidate_summaries.append({
            'candidate_index':int(_ci),
            'source':str(_cq.get('crease_source','UNKNOWN')),
            'quality':float(_cq.get('quality',0.0) or 0.0),
            'plan_score':float(_cq.get('v12_plan_score',0.0) or 0.0),
            'ready':bool(_cq.get('execution_evidence_ok',False) and _cq.get('v12_candidate_role_ok',False)),
            'role_reason':str((_cq.get('v12_candidate_role_validation') or {}).get('reason','UNKNOWN')) if isinstance(_cq.get('v12_candidate_role_validation'),dict) else 'UNKNOWN',
            'crease_anchor_px':_training_json_safe(_cq.get('crease_anchor_px')),
            'crease_line_px':_training_json_safe(_cq.get('v13_selected_crease_line_px')),
            'active_px':_training_json_safe(_cq.get('active_px')),
            'hold_px':_training_json_safe(_cq.get('hold_px')),
            'component_span_px':float(_cq.get('component_span_px',0.0) or 0.0),
            'pose_group':str(_cq.get('pose_fold_region_hint','NONE')),
            'pose_confidence':float(_cq.get('pose_fold_region_confidence',0.0) or 0.0),
            'branch_fold_score':float(_cq.get('e122_branch_fold_score',0.0) or 0.0),
            'target_hint_distance_mm':_training_json_safe(_cq.get('v25_target_hint_distance_mm')),
        })

    # v25: the learned MICRO target selects WHICH already-safe crease to use.
    # Mechanical candidate generation, reach/path/board/role validation are unchanged.
    target_hint_enabled = bool(getattr(args, 'micro_unfold_target_hint_enable', False))
    target_hint_board = getattr(args, 'micro_unfold_target_board_mm', None)
    target_hint_max_mm = max(20.0, float(getattr(args, 'micro_unfold_target_match_max_mm', 120.0)))
    if target_hint_enabled:
        try:
            target_xy = np.asarray(target_hint_board, dtype=np.float64).reshape(2)
            if not np.all(np.isfinite(target_xy)):
                raise ValueError('non-finite target')
        except Exception:
            target_xy = None
        if target_xy is None:
            # A TARGET decision without a usable coordinate is not allowed to fall back
            # to the rule-based best crease.
            if evaluated:
                best_diag = sorted(evaluated, key=lambda q: float(q.get('v12_plan_score', 0.0)), reverse=True)[0]
                best_diag['execution_evidence_ok'] = False
                best_diag['execution_evidence_reason'] = 'V25_TARGET_HINT_INVALID'
                best_diag['v25_target_hint_enabled'] = True
                best_diag['v25_target_hint_board_mm'] = _training_json_safe(target_hint_board)
                best_diag['v12_selection_policy'] = 'TARGET_MODEL_INVALID_FAIL_CLOSED'
                return best_diag
        else:
            def _segment_distance_mm(q):
                line = q.get('v12_selected_crease_line_board')
                try:
                    a = np.asarray(line[0], dtype=np.float64).reshape(2)
                    b = np.asarray(line[1], dtype=np.float64).reshape(2)
                    ab = b - a
                    den = float(np.dot(ab, ab))
                    if den <= 1e-9:
                        return float(np.linalg.norm(target_xy - a))
                    t = float(np.clip(np.dot(target_xy - a, ab) / den, 0.0, 1.0))
                    p = a + t * ab
                    return float(np.linalg.norm(target_xy - p))
                except Exception:
                    return float('inf')
            for q in evaluated:
                d = _segment_distance_mm(q)
                q['v25_target_hint_enabled'] = True
                q['v25_target_hint_board_mm'] = [float(target_xy[0]), float(target_xy[1])]
                q['v25_target_hint_distance_mm'] = None if not np.isfinite(d) else float(d)
                q['v25_target_hint_match_max_mm'] = float(target_hint_max_mm)
            matched = [q for q in evaluated if q.get('v25_target_hint_distance_mm') is not None and float(q['v25_target_hint_distance_mm']) <= target_hint_max_mm]
            if not matched and evaluated:
                best_diag = min(evaluated, key=lambda q: float(q.get('v25_target_hint_distance_mm')) if q.get('v25_target_hint_distance_mm') is not None else 1e9)
                best_diag['execution_evidence_ok'] = False
                best_diag['execution_evidence_reason'] = (
                    f"V25_TARGET_NO_SAFE_CREASE_WITHIN_{target_hint_max_mm:.1f}MM:"
                    f"nearest={best_diag.get('v25_target_hint_distance_mm')}"
                )
                best_diag['v12_selection_policy'] = 'TARGET_MODEL_NO_MATCH_FAIL_CLOSED'
                print(
                    f"[E156-V25-TARGET-BLOCK] target=({target_xy[0]:.1f},{target_xy[1]:.1f})mm "
                    f"nearest={best_diag.get('v25_target_hint_distance_mm')}mm max={target_hint_max_mm:.1f}mm"
                )
                return best_diag
            if matched:
                # All retained items already passed E156 mechanical/role safety. The
                # learned target owns fold selection; original E156 score is tie-break only.
                matched.sort(key=lambda q: (float(q.get('v25_target_hint_distance_mm', 1e9)), -float(q.get('v12_plan_score', 0.0))))
                evaluated = matched
                print(
                    f"[E156-V25-TARGET-MATCH] target=({target_xy[0]:.1f},{target_xy[1]:.1f})mm "
                    f"matched={len(matched)} max={target_hint_max_mm:.1f}mm "
                    f"bestDist={float(matched[0].get('v25_target_hint_distance_mm',0.0)):.1f}mm"
                )

        # Distances are computed after the compact provenance list is first built.
        # Refresh only this diagnostic field so saved/debug output reflects the exact
        # learned-target association used for control.
        for _summary,_candidate in zip(_v13_candidate_summaries,candidates):
            _summary['target_hint_distance_mm']=_training_json_safe(_candidate.get('v25_target_hint_distance_mm'))

    if not evaluated:
        # Preserve the best diagnostic candidate/reason, but never publish it READY.
        candidates.sort(key=lambda q:float(q.get('v12_plan_score',q.get('quality',0.0))),reverse=True)
        best=candidates[0]
        best['alternative_count']=int(len(candidates))
        best['v12_candidate_count_total']=int(len(candidates))
        best['v12_candidate_count_ready']=0
        best['v12_candidate_rank']=0
        best['v12_selection_policy']='COMPLETE_PLAN_THEN_RANK_FAIL_CLOSED'
        best['v13_fold_candidate_summaries']=copy.deepcopy(_v13_candidate_summaries)
        best['execution_evidence_ok']=False
        if str(best.get('execution_evidence_reason','OK'))=='OK':
            best['execution_evidence_reason']='V12_NO_COMPLETE_ROLE_VALID_PLAN'
        print(f"[E156-V12-PLAN-BLOCK] candidates={len(candidates)} ready=0 topReason={best.get('execution_evidence_reason')}")
        return best

    if not target_hint_enabled:
        evaluated.sort(key=lambda q:float(q.get('v12_plan_score',0.0)),reverse=True)
    for rank,q in enumerate(evaluated,1):
        q['v12_candidate_rank']=int(rank)
    best=evaluated[0]
    best['alternative_count']=int(len(candidates))
    best['v12_candidate_count_total']=int(len(candidates))
    best['v12_candidate_count_ready']=int(len(evaluated))
    best['v12_selection_policy']='TARGET_MODEL_NEAREST_SAFE_CREASE' if target_hint_enabled else 'COMPLETE_PLAN_THEN_RANK'
    best['v13_fold_candidate_summaries']=copy.deepcopy(_v13_candidate_summaries)
    best['top_quality_gap']=float(best['v12_plan_score']-evaluated[1]['v12_plan_score']) if len(evaluated)>1 else 9.999
    best['execution_evidence_ok']=True
    best['execution_evidence_reason']='OK'
    if bool(_v12_candidate_evidence(best)[2]):
        print(
            f"[E123-PREVIEW-PAIR-RESCUE] branch={float(best.get('e122_branch_fold_score',0.0)):.3f} "
            f"Q={float(best.get('quality',0.0)):.3f}; complete-plan role validation passed"
        )
    print(
        f"[E156-V12-PLAN-SELECT] ready={len(evaluated)}/{len(candidates)} rank=1 "
        f"score={float(best['v12_plan_score']):.3f} Q={float(best.get('quality',0.0)):.3f} "
        f"span={float(best.get('component_span_px',0.0)):.1f}px center={float(best.get('v12_crease_center_support',0.0)):.2f} "
        f"src={best.get('crease_source')}"
    )
    return best


# Backward-compatible planner hook: E109 replaces only AUTO crease-pair selection.
def _e99_select_crease_mask_pair(depth_state, mask_u8, H_value, args, config=None, cfg=None):
    return _e109_select_crease_outer_lobe_pair(depth_state, mask_u8, H_value, args, config=config, cfg=cfg)

def _fix53_plan_points(depth_state, H_value, config, cfg, args, preselected_crease_pair=None):
    """E92 HOLD/ACTIVE geometry: TOP3 fold hypothesis -> raised-core ACTIVE -> local body HOLD."""
    if not isinstance(depth_state, dict) or not bool(depth_state.get("available", False)):
        raise RuntimeError(
            f"FIX53 depth unavailable: {depth_state.get('reason') if isinstance(depth_state, dict) else 'NONE'}"
        )
    # E98: UniDepth ambiguity is diagnostic only when crease geometry can supply
    # an explicit FLAP/BODY partition. The depth fallback still honors ambiguity.

    mask = ((np.asarray(depth_state["garment_mask"], dtype=np.uint8) > 0).astype(np.uint8) * 255)
    labels = np.asarray(depth_state.get("visual_top2_labels", depth_state["labels"]), dtype=np.uint8)
    relief = np.asarray(depth_state["local_relief_mm"], dtype=np.float32)
    prom = np.asarray(depth_state["prominence_mm"], dtype=np.float32)
    if mask.shape != labels.shape or mask.shape != relief.shape or mask.shape != prom.shape:
        raise RuntimeError("E92 depth/mask geometry shape mismatch")
    if cv2.countNonZero(mask) < 500:
        raise RuntimeError("E92 garment mask too small")

    # E98 crease-first geometry. Manual 1/2/3 remains an explicit operator override;
    # otherwise the crease+MASK partition is authoritative and FUSED FOLD is only diagnostic.
    crease_pair = copy.deepcopy(preselected_crease_pair) if isinstance(preselected_crease_pair,dict) else None
    if crease_pair is not None:
        print(f"[E156-V41-PRESELECTED-CREASE-PAIR] source={crease_pair.get('crease_source','UNKNOWN')} pair={float(crease_pair.get('pair_separation_mm',0.0) or 0.0):.1f}mm")
    if crease_pair is None and not bool(depth_state.get("manual_override", False)):
        try:
            crease_pair = _e99_select_crease_mask_pair(depth_state, mask, H_value, args, config=config, cfg=cfg)
        except Exception as exc:
            _msg=str(exc)
            if any(tok in _msg for tok in ("E112_ORIENTATION_AMBIGUOUS", "E115_DEPTH_ROLE_CONTRADICTION", "E115_ORIENTATION_AMBIGUOUS", "E116_")):
                print(f"[E116-CREASE-BLOCK] {_msg}")
                raise
            print(f"[E116-CREASE-WARN] {exc!r}; falling back to legacy depth/MASK point planner")
            crease_pair = None

    if isinstance(crease_pair, dict):
        active_rank = 0
        active_px = np.asarray(crease_pair["active_px"], dtype=np.float64)
        hold_px = np.asarray(crease_pair["hold_px"], dtype=np.float64)
        garment_center = np.asarray(crease_pair["garment_center_px"], dtype=np.float64)
        active_board = np.asarray(crease_pair["active_board"], dtype=np.float64)
        hold_board = np.asarray(crease_pair["hold_board"], dtype=np.float64)
        active_meta = {
            "active_selection_mode": str(crease_pair.get("active_selection_mode", "E125_FOLD_REGION_FIRST_BOUNDARY_PAIR")),
            "e129_active_pose_depth_authority": float(crease_pair.get("active_pose_depth_authority", 0.0)),
            "e129_active_pose_region_raised_mm": float(crease_pair.get("active_pose_region_raised_mm", 0.0)),
            "e130_directional_plan": bool(crease_pair.get("e130_directional_plan", False)),
            "e130_active_to_contour_mm": float(crease_pair.get("e130_active_to_contour_mm", 0.0)),
            "e130_direction_evidence_px": _training_json_safe(crease_pair.get("e130_direction_evidence_px", None)),
            "e125_primary_fold_source": str(crease_pair.get("primary_fold_region_source", depth_state.get("e125_primary_fold_source", "CREASE_CONTOUR_FALLBACK"))),
            "e125_primary_fold_confidence": float(crease_pair.get("primary_fold_region_confidence", depth_state.get("e125_primary_fold_confidence", 0.0))),
            "e125_primary_fold_group": str(crease_pair.get("primary_fold_region_group", depth_state.get("e125_primary_fold_group", "NONE"))),
            "e125_active_fold_support": float(crease_pair.get("active_primary_fold_support", 0.0)),
            "e125_hold_fold_support": float(crease_pair.get("hold_primary_fold_support", 0.0)),
            "e125_fold_region_role_support": float(crease_pair.get("fold_region_role_support", 0.5)),
            "e125_fold_boundary_support": float(crease_pair.get("fold_boundary_component_support", 0.0)),
            "e116_crease_source": str(crease_pair.get("crease_source", "UNKNOWN")),
            "e117_crease_source": str(crease_pair.get("crease_source", "UNKNOWN")),
            "execution_evidence_ok": bool(crease_pair.get("execution_evidence_ok", True)),
            "execution_evidence_reason": str(crease_pair.get("execution_evidence_reason", "OK")),
            "e117_pose_fold_support": float(crease_pair.get("pose_fold_support", 0.5)),
            "e117_pose_fold_role_support": float(crease_pair.get("pose_fold_role_support", 0.5)),
            "e119_pose_deform_available": bool(crease_pair.get("pose_deform_available", False)),
            "e119_pose_deform_confidence": float(crease_pair.get("pose_deform_confidence", 0.0)),
            "e119_pose_deform_mode": str(crease_pair.get("pose_deform_mode", "NONE")),
            "e119_pose_deform_side_hint": str(crease_pair.get("pose_deform_side_hint", "NONE")),
            "e119_pose_deform_anchor_support": float(crease_pair.get("pose_deform_anchor_support", 0.0)),
            "e119_pose_deform_role_support": float(crease_pair.get("pose_deform_role_support", 0.5)),
            "e119_pose_deform_authoritative": bool(crease_pair.get("pose_deform_authoritative", False)),
            "e120_pose_fold_region_hint": str(crease_pair.get("pose_fold_region_hint", depth_state.get("pose_deform_fold_region_hint", "NONE"))),
            "e120_pose_fold_region_confidence": float(crease_pair.get("pose_fold_region_confidence", depth_state.get("pose_deform_fold_region_confidence", 0.0))),
            "e120_pose_direction_alignment": float(crease_pair.get("pose_direction_alignment", 0.0)),
            "e120_pose_axis_support": float(crease_pair.get("pose_axis_support", 0.5)),
            "e118_pose_split_lobe_count": int(crease_pair.get("pose_split_lobe_count", 0)),
            "e118_pose_split_body_count": int(crease_pair.get("pose_split_body_count", 0)),
            "e118_pose_split_confidence": float(crease_pair.get("pose_split_confidence", 0.0)),
            "e118_pose_split_authoritative": bool(crease_pair.get("pose_split_authoritative", False)),
            "e118_pose_split_preferred_active_region": str(crease_pair.get("pose_split_preferred_active_region", "NONE")),
            "e118_pose_count_role_support": float(crease_pair.get("pose_count_role_support", 0.5)),
            "e117_depth_role_probe_valid": bool(crease_pair.get("depth_role_probe_valid", False)),
            "e117_depth_role_probe_delta_mm": float(crease_pair.get("depth_role_probe_delta_mm", 0.0)),
            "e117_depth_role_probe_confidence": float(crease_pair.get("depth_role_probe_confidence", 0.0)),
            "active_crease_distance_mm": float(crease_pair["active_crease_distance_mm"]),
            "active_depth_reference": float(crease_pair.get("active_depth_reference", 0.0)),
            "active_fused_fold_required": False,
            "e109_quality": float(crease_pair.get("quality", 0.0)),
            "e109_fused_support": float(crease_pair.get("fused_support", 0.0)),
            "e109_body_support": float(crease_pair.get("body_support", 0.0)),
            "e109_lobe_support": float(crease_pair.get("coarse_lobe_support", 0.5)),
            "e109_lobe_quality": float(crease_pair.get("coarse_lobe_quality", 0.0)),
            "e109_surface_asymmetry": float(crease_pair.get("surface_asymmetry", 0.0)),
            "e109_seam_penalty": float(crease_pair.get("seam_penalty", 0.0)),
            "e109_reach_support": float(crease_pair.get("reach_support", 1.0)),
            "e109_flap_soft_support": float(crease_pair.get("crease_flap_soft_support", 0.5)),
            "e112_hold_clean_score": float(crease_pair.get("hold_clean_score", 0.0)),
            "e112_hold_crease_clear_px": float(crease_pair.get("hold_crease_clear_px", 0.0)),
            "e112_hold_crease_density": float(crease_pair.get("hold_crease_density", 1.0)),
            "e112_hold_clean_advantage": float(crease_pair.get("hold_clean_advantage", 0.0)),
            "e112_orientation_confidence": float(crease_pair.get("orientation_confidence", 0.0)),
            "e115_depth_role_valid": bool(crease_pair.get("depth_role_valid", False)),
            "e115_active_depth_role_mm": float(crease_pair.get("active_depth_role_mm", 0.0)),
            "e115_hold_depth_role_mm": float(crease_pair.get("hold_depth_role_mm", 0.0)),
            "e115_depth_role_delta_mm": float(crease_pair.get("depth_role_delta_mm", 0.0)),
            "e115_depth_role_confidence": float(crease_pair.get("depth_role_confidence", 0.0)),
            "e109_crease_component_label": int(crease_pair.get("component_label", -1)),
            "e109_crease_component_span_px": float(crease_pair.get("component_span_px", 0.0)),
            "e109_pair_alternative_count": int(crease_pair.get("alternative_count", 1)),
            "v12_plan_score": float(crease_pair.get("v12_plan_score", crease_pair.get("quality", 0.0))),
            "v12_candidate_rank": int(crease_pair.get("v12_candidate_rank", 0)),
            "v12_candidate_count_total": int(crease_pair.get("v12_candidate_count_total", crease_pair.get("alternative_count", 1))),
            "v12_candidate_count_ready": int(crease_pair.get("v12_candidate_count_ready", 1 if crease_pair.get("execution_evidence_ok", False) else 0)),
            "v12_crease_span_relative": float(crease_pair.get("v12_crease_span_relative", 0.0)),
            "v12_crease_center_support": float(crease_pair.get("v12_crease_center_support", 0.0)),
            "v12_active_interior_support": float(crease_pair.get("v12_active_interior_support", 0.0)),
            "v12_selected_crease_line_board": _training_json_safe(crease_pair.get("v12_selected_crease_line_board")),
            "v12_selection_policy": str(crease_pair.get("v12_selection_policy", "LEGACY_SINGLE_BEST")),
            "v12_crease_normal_px": _training_json_safe(crease_pair.get("v12_crease_normal_px")),
        }
        active_center_r = float(np.linalg.norm(active_px - garment_center))
        hold_alignment = 1.0
        hold_score = 0.0
        hold_target_error = float(crease_pair.get("hold_normal_target_error_mm", 0.0))
        selected_cone_deg = 90.0
        selected_min_sep_mm = float(crease_pair.get("pair_separation_mm", np.linalg.norm(hold_board-active_board)))
        selected_local_radius_mm = 0.0
        hold_target_board = hold_board.copy()
        hold_target_distance = float(np.linalg.norm(hold_board-active_board))
        print(
            f"[E137-CREASE-RGB-PAIR] source={str(crease_pair.get('crease_source','UNKNOWN'))} ACTIVE crease={float(crease_pair['active_crease_distance_mm']):.1f}mm "
            f"HOLD crease={float(crease_pair['hold_crease_distance_mm']):.1f}mm "
            f"sep={float(crease_pair['pair_separation_mm']):.1f}mm "
            f"Q={float(crease_pair.get('quality',0.0)):.3f} "
            f"lobe={float(crease_pair.get('coarse_lobe_support',0.5)):.3f}/q{float(crease_pair.get('coarse_lobe_quality',0.0)):.3f} "
            f"body={float(crease_pair.get('body_support',0.0)):.3f} fused={float(crease_pair.get('fused_support',0.0)):.3f} "
            f"holdClean={float(crease_pair.get('hold_clean_score',0.0)):.3f} clear={float(crease_pair.get('hold_crease_clear_px',0.0)):.1f}px "
            f"density={float(crease_pair.get('hold_crease_density',1.0)):.3f} cleanAdv={float(crease_pair.get('hold_clean_advantage',0.0)):+.3f} "
            f"poseDeform={str(crease_pair.get('pose_deform_mode','NONE'))}:{float(crease_pair.get('pose_deform_confidence',0.0)):.3f} "
            f"role={float(crease_pair.get('pose_deform_role_support',0.5)):.3f} anchor={float(crease_pair.get('pose_deform_anchor_support',0.0)):.3f} "
            f"poseSplit L/B={int(crease_pair.get('pose_split_lobe_count',0))}/{int(crease_pair.get('pose_split_body_count',0))} "
            f"gap={int(crease_pair.get('pose_split_gap',0))} conf={float(crease_pair.get('pose_split_confidence',0.0)):.3f} "
            f"depthRole A={float(crease_pair.get('active_depth_role_mm',0.0)):+.2f} H={float(crease_pair.get('hold_depth_role_mm',0.0)):+.2f} "
            f"d={float(crease_pair.get('depth_role_delta_mm',0.0)):+.2f}mm conf={float(crease_pair.get('depth_role_confidence',0.0)):+.3f} "
            f"orient={float(crease_pair.get('orientation_confidence',0.0)):+.3f} neckPenalty={float(crease_pair.get('seam_penalty',0.0)):.3f} "
            f"reach={float(crease_pair.get('reach_support',1.0)):.3f} alts={int(crease_pair.get('alternative_count',1))} "
            f"region={str(crease_pair.get('pose_fold_region_hint','NONE'))}:{float(crease_pair.get('pose_fold_region_confidence',0.0)):.2f} "
            f"axis={float(crease_pair.get('pose_direction_alignment',0.0)):+.2f} "
            f"| E137 POSE+RGB-CREASE DIRECTION ONLY"
        )
    else:
        if not bool(depth_state.get("manual_override", False)):
            if bool(depth_state.get('e151_edge_cluster_hard_block',False)):
                _info=dict(depth_state.get('e151_edge_cluster_block_info',{}) or {})
                raise RuntimeError(
                    f"E151_EDGE_CLUSTER_HARD_BLOCK side={str(_info.get('side','NONE'))} "
                    f"score={float(_info.get('score',0.0)):.3f}: selected Pose/edge cluster has no board/reach/path-safe exact-normal pair; "
                    "legacy PREVIEW HOLD/ACTIVE suppressed; press I after the garment/crease geometry changes."
                )
            _e122_prev=np.asarray(depth_state.get('e122_fold_preview_skeleton',np.zeros_like(mask)),dtype=np.uint8)
            _e122_exec=np.asarray(depth_state.get('e122_fold_execution_skeleton',np.zeros_like(mask)),dtype=np.uint8)
            _why=(
                f"E151 AUTO exhausted Pose authority + lower-fold exact normal + edge-cluster exact normal + generic exact Fold-Crease normal; "
                f"previewPixels={cv2.countNonZero(_e122_prev)} execPixels={cv2.countNonZero(_e122_exec)}. "
                "Only bounded same-crease near-normal fan + HOLD tangent reach recovery exist; no unrelated-crease/depth fallback; press I after correcting MASK/Pose/RGB crease perception."
            )
            raise RuntimeError(_why)
        if bool(depth_state.get("hypothesis_ambiguous", False)):
            raise RuntimeError(
                "E98 FOLD_AMBIGUOUS in UniDepth fallback: crease partition unavailable and depth hypotheses are ambiguous; X blocked"
            )
        active_rank = 1 if np.any(labels == 1) else None
        if active_rank is None:
            raise RuntimeError("E92 NO RELIABLE FOLD HYPOTHESIS: no selected fused fold")

        active_px, garment_center, active_meta = _e92_select_active_point(depth_state, mask)
        active_board = _fix53_px_to_board(H_value, active_px)
        garment_center_board = _fix53_px_to_board(H_value, garment_center)
        active_to_center_px = garment_center - active_px
        active_center_dist_px = float(np.linalg.norm(active_to_center_px))
        if active_center_dist_px <= 1e-6:
            raise RuntimeError("E92 ACTIVE collapsed onto garment centroid; direction undefined")
        inward_u_px = active_to_center_px / active_center_dist_px

        # ------------------------------------------------------------------
        # HOLD: make ACTIVE-relative geometry authoritative before global flatness.
        # Aim roughly 90 mm from ACTIVE toward the body, without overshooting the
        # garment centroid. Search locally first, then widen. E91 separation tiers
        # remain unchanged and are still the safety gate.
        # ------------------------------------------------------------------
        body_vec_board = garment_center_board - active_board
        body_dist_board = float(np.linalg.norm(body_vec_board))
        if body_dist_board <= 1e-6:
            raise RuntimeError("E92 ACTIVE->garment-centre board vector collapsed")
        body_u_board = body_vec_board / body_dist_board
        requested_hold_target = max(35.0, float(getattr(args, "micro_unfold_hold_target_mm", 70.0)))
        hold_target_distance = min(requested_hold_target, 0.82 * body_dist_board)
        hold_target_distance = max(min(35.0, 0.70 * body_dist_board), hold_target_distance)
        hold_target_board = active_board + body_u_board * hold_target_distance

        interior = cv2.erode(mask, np.ones((11, 11), np.uint8), iterations=1) > 0
        strict_all = np.asarray(depth_state.get("strict_candidate_labels", np.zeros_like(labels)), dtype=np.uint8) > 0
        raised_any = ((labels > 0) | strict_all).astype(np.uint8) * 255
        raised_exclusion = cv2.dilate(raised_any, np.ones((35, 35), np.uint8), iterations=1) > 0
        finite = np.isfinite(relief) & np.isfinite(prom)
        flat_domain = interior & (~raised_exclusion) & finite
        if int(np.count_nonzero(flat_domain)) < 50:
            raised_exclusion = cv2.dilate(raised_any, np.ones((21, 21), np.uint8), iterations=1) > 0
            flat_domain = interior & (~raised_exclusion) & finite
        if int(np.count_nonzero(flat_domain)) < 25:
            raise RuntimeError("E92 cannot find a flat main-body HOLD domain away from C1/C2/C3 raised seeds")

        flatness = np.abs(relief) + 0.70 * np.maximum(prom, 0.0)
        fv = flatness[flat_domain]
        flat_gate = float(np.percentile(fv, 34.0))
        low_flat = flat_domain & (flatness <= flat_gate)
        if int(np.count_nonzero(low_flat)) < 20:
            low_flat = flat_domain

        distance_inside = cv2.distanceTransform(mask, cv2.DIST_L2, 5).astype(np.float32)
        hy, hx = np.where(low_flat)
        if hx.size == 0:
            raise RuntimeError("E92 flat HOLD candidate set is empty")

        requested_min_sep = max(45.0, float(getattr(args, "micro_unfold_min_pair_sep_mm", 70.0)))
        active_center_r = float(np.linalg.norm(active_px - garment_center))
        sep_tiers = []
        for sep in (requested_min_sep, min(requested_min_sep, 60.0), min(requested_min_sep, 50.0)):
            sep = max(45.0, float(sep))
            if not any(abs(sep - old_sep) < 1e-6 for old_sep in sep_tiers):
                sep_tiers.append(sep)

        base_local_radius = max(35.0, float(getattr(args, "micro_unfold_hold_local_radius_mm", 65.0)))
        local_radius_tiers = [base_local_radius, max(base_local_radius, 90.0), max(base_local_radius, 120.0), float("inf")]
        dedup_radius = []
        for r in local_radius_tiers:
            if not any((np.isinf(r) and np.isinf(q)) or abs(r-q) < 1e-6 for q in dedup_radius):
                dedup_radius.append(float(r))
        local_radius_tiers = dedup_radius

        hold_px = None; hold_board = None; hold_alignment = None; hold_score = None
        hold_target_error = float("inf")
        selected_cone_deg = None; selected_min_sep_mm = None; selected_local_radius_mm = float("inf")

        for min_sep in sep_tiers:
            for local_radius in local_radius_tiers:
                for cone_deg in (55.0, 70.0, 85.0, 105.0):
                    cos_gate = float(np.cos(np.deg2rad(cone_deg)))
                    candidates_hold = []
                    for x, y in zip(hx.tolist(), hy.tolist()):
                        p = np.asarray([float(x), float(y)], dtype=np.float64)
                        vec = p - active_px
                        vec_len = float(np.linalg.norm(vec))
                        if vec_len <= 1e-6:
                            continue
                        cos_inward = float(np.dot(vec / vec_len, inward_u_px))
                        if cos_inward < cos_gate:
                            continue
                        center_r = float(np.linalg.norm(p - garment_center))
                        if center_r > active_center_r * 1.05 + 8.0:
                            continue
                        b = _fix53_px_to_board(H_value, p)
                        if not _fix53_board_inside(cfg, b, margin=25.0):
                            continue
                        sep_mm = float(np.linalg.norm(b - active_board))
                        if sep_mm < min_sep:
                            continue
                        target_error = float(np.linalg.norm(b - hold_target_board))
                        if np.isfinite(local_radius) and target_error > local_radius:
                            continue
                        f = float(flatness[int(y), int(x)])
                        inside_px = float(distance_inside[int(y), int(x)])
                        # Flatness remains authoritative, but local target error now
                        # steers HOLD toward the fold base/body instead of arbitrary
                        # globally-flat garment centre pixels.
                        geom = (
                            0.024 * target_error
                            + 0.006 * center_r
                            + 0.65 * (1.0 - cos_inward)
                            - 0.035 * min(inside_px, 30.0)
                        )
                        total = f + geom
                        candidates_hold.append((total, target_error, f, -inside_px, center_r, -cos_inward, p, b, cos_inward))
                    if candidates_hold:
                        candidates_hold.sort(key=lambda q: q[:6])
                        best = candidates_hold[0]
                        hold_score = float(best[0]); hold_target_error = float(best[1])
                        hold_px = best[6]; hold_board = best[7]; hold_alignment = float(best[8])
                        selected_cone_deg = float(cone_deg); selected_min_sep_mm = float(min_sep)
                        selected_local_radius_mm = float(local_radius)
                        break
                if hold_px is not None:
                    break
            if hold_px is not None:
                break

        if hold_px is None:
            raise RuntimeError(
                "E92 no safe flat inward HOLD after local-body search and E91 separation fallback "
                f"tiers={[round(v,1) for v in sep_tiers]}mm"
            )
    if not _fix53_board_inside(cfg, active_board, margin=25.0):
        raise RuntimeError(f"E92 ACTIVE board point outside 25mm margin: {active_board.tolist()}")

    pull_vec = active_board - hold_board
    pair_sep = float(np.linalg.norm(pull_vec))
    if pair_sep <= 1e-6:
        raise RuntimeError("E92 HOLD/ACTIVE pair collapsed")
    _e130_directional = bool(isinstance(crease_pair, dict) and crease_pair.get('e130_directional_plan', False))
    if _e130_directional:
        pull_hint_u = np.asarray(crease_pair.get('e130_unfold_unit_board', pull_vec / pair_sep), dtype=np.float64)
        pull_hint_u /= max(1e-9, float(np.linalg.norm(pull_hint_u)))
        (pull_board, used_pull, garment_exit_mm, outside_margin_used, pull_endpoint_px,
         pull_u, hold_boundary_distance_mm, boundary_normal_delta_deg, boundary_normal_ok) = _e130_find_axis_pull_endpoint(
            mask, H_value, cfg, active_board, hold_board, pull_hint_u, args, log_final=True
        )
    else:
        pull_hint_u = pull_vec / pair_sep
        (pull_board, used_pull, garment_exit_mm, outside_margin_used, pull_endpoint_px,
         pull_u, hold_boundary_distance_mm, boundary_normal_delta_deg, boundary_normal_ok) = _e111_find_outer_normal_pull_endpoint(
            mask, H_value, cfg, active_board, hold_board, pull_hint_u, args, log_final=True
        )
    # Role assignment remains reversible exactly like manual case1/case2.
    assignment_candidates = []
    for active_arm, hold_arm in (("arm1", "arm2"), ("arm2", "arm1")):
        try:
            a_local = _fix6_local_xy(config, active_arm, active_board)
            h_local = _fix6_local_xy(config, hold_arm, hold_board)
            # V5 empirical reach backoff. Keep the original 420mm hard gate, but
            # shorten only the ACTIVE pull endpoint when it sits in the repeatedly
            # observed near-saturation annulus. The endpoint must remain outside
            # garment and inside the existing board margin.
            trial_pull_board = np.asarray(pull_board, dtype=np.float64).copy()
            trial_used_pull = float(used_pull)
            trial_pull_px = np.asarray(pull_endpoint_px, dtype=np.float64).copy()
            p_local = _fix6_local_xy(config, active_arm, trial_pull_board)
            soft_pull_radius = float(np.clip(getattr(args, "micro_unfold_pull_soft_radius_mm", 390.0), 340.0, 420.0))
            min_outside_dist = float(garment_exit_mm) + max(8.0, min(15.0, float(outside_margin_used)))
            if float(np.linalg.norm(p_local)) > soft_pull_radius:
                while trial_used_pull - 5.0 >= min_outside_dist - 1e-6:
                    trial_used_pull -= 5.0
                    q = np.asarray(active_board, dtype=np.float64) + np.asarray(pull_u, dtype=np.float64) * trial_used_pull
                    if not _fix53_board_inside(cfg, q, margin=20.0):
                        continue
                    try:
                        qpx = _fix89_board_to_px(H_value, q)
                        qx,qy=int(round(float(qpx[0]))),int(round(float(qpx[1])))
                        outside = not (0<=qx<mask.shape[1] and 0<=qy<mask.shape[0] and mask[qy,qx]>0)
                    except Exception:
                        outside = False
                    qlocal = _fix6_local_xy(config, active_arm, q)
                    if outside and float(np.linalg.norm(qlocal)) <= soft_pull_radius:
                        trial_pull_board=q; trial_pull_px=np.asarray(qpx,dtype=np.float64); p_local=qlocal
                        break
            a_radius = float(np.linalg.norm(a_local))
            h_radius = float(np.linalg.norm(h_local))
            p_radius = float(np.linalg.norm(p_local))
            if max(a_radius, h_radius, p_radius) > 420.0:
                continue
            a_standby = np.asarray(standby_roarm_pose(cfg, active_arm)[:2], dtype=np.float64)
            h_standby = np.asarray(standby_roarm_pose(cfg, hold_arm)[:2], dtype=np.float64)
            travel = float(np.linalg.norm(a_local - a_standby) + np.linalg.norm(h_local - h_standby))
            a_standby_board = _fix37_local_xy_to_board(config, active_arm, a_standby)
            ts = np.linspace(0.0, 1.0, 61, dtype=np.float64)
            path = a_standby_board[None, :] + ts[:, None] * (active_board - a_standby_board)[None, :]
            path_clear = float(np.min(np.linalg.norm(path - hold_board[None, :], axis=1)))
            if path_clear < 65.0:
                continue
            assignment_candidates.append({
                "active_arm": active_arm,
                "hold_arm": hold_arm,
                "active_local_xy": a_local,
                "hold_local_xy": h_local,
                "pull_local_xy": p_local,
                "pull_board_effective": np.asarray(trial_pull_board,dtype=np.float64),
                "pull_endpoint_px_effective": np.asarray(trial_pull_px,dtype=np.float64),
                "pull_distance_mm_effective": float(trial_used_pull),
                "pull_soft_backoff_mm": float(max(0.0, float(used_pull)-float(trial_used_pull))),
                "travel_cost_mm": travel,
                "active_path_hold_clearance_mm": path_clear,
                "active_radius_mm": a_radius,
                "hold_radius_mm": h_radius,
                "pull_radius_mm": p_radius,
            })
        except Exception:
            continue
    if not assignment_candidates:
        raise RuntimeError("E91 neither ARM1/ARM2 role assignment passed radius/path-clearance gates")
    assignment = min(assignment_candidates, key=lambda item: float(item["travel_cost_mm"]))

    active_arm = str(assignment["active_arm"])
    hold_arm = str(assignment["hold_arm"])
    if "pull_board_effective" in assignment:
        original_pull = float(used_pull)
        pull_board = np.asarray(assignment["pull_board_effective"], dtype=np.float64)
        pull_endpoint_px = np.asarray(assignment["pull_endpoint_px_effective"], dtype=np.float64)
        used_pull = float(assignment["pull_distance_mm_effective"])
        outside_margin_used = max(0.0, float(used_pull) - float(garment_exit_mm))
        if original_pull - used_pull > 0.1:
            print(f"[E156-V5-REACH-BACKOFF] ACTIVE={active_arm.upper()} pull {original_pull:.1f}->{used_pull:.1f}mm "
                  f"softRadius={float(getattr(args,'micro_unfold_pull_soft_radius_mm',390.0)):.1f}mm")

    # E111 short diagonal peel endpoint. The main pull remains board-safe; the peel
    # is already rising, so constrain it by the ACTIVE arm's local reach rather than
    # forcing another garment/board-boundary contact.
    peel_xy_request = max(0.0, float(getattr(args, "micro_unfold_peel_xy_mm", 35.0)))
    peel_z_rise = max(50.0, float(getattr(args, "micro_unfold_peel_z_mm", 55.0)))
    pull_local = np.asarray(assignment["pull_local_xy"], dtype=np.float64)
    try:
        local_probe = _fix6_local_xy(config, active_arm, np.asarray(pull_board, dtype=np.float64) + np.asarray(pull_u, dtype=np.float64) * 10.0)
        peel_local_u = np.asarray(local_probe - pull_local, dtype=np.float64)
        peel_local_u /= max(1e-9, float(np.linalg.norm(peel_local_u)))
    except Exception:
        peel_local_u = np.asarray([1.0, 0.0], dtype=np.float64)
    peel_xy_used = float(peel_xy_request)
    peel_local_xy = pull_local.copy()
    while peel_xy_used > 0.0:
        candidate_local = pull_local + peel_local_u * peel_xy_used
        if float(np.linalg.norm(candidate_local)) <= float(np.clip(getattr(args, "micro_unfold_peel_soft_radius_mm", 400.0), 350.0, 420.0)):
            peel_local_xy = candidate_local
            break
        peel_xy_used -= 5.0
    if peel_xy_used <= 0.0:
        peel_xy_used = 0.0
        peel_local_xy = pull_local.copy()

    active_contact_z = float(arm_contact_z(cfg, active_arm))
    hold_contact_z = float(arm_contact_z(cfg, hold_arm))
    # E131/V51 physical tuning: base HOLD offset is +23 mm.  ARM2-as-HOLD is
    # lowered by 5 mm (effective +18 mm) for firmer board contact; ARM1 HOLD is
    # unchanged. ACTIVE remains Z-locked during the main XY sweep; retry floor
    # shifts upward with the same ACTIVE offset.
    hold_z_offset = _micro_unfold_hold_z_offset(args, hold_arm)
    active_z_offset = float(getattr(args, "micro_unfold_active_z_offset_mm", 26.0))
    hold_action_z = hold_contact_z + hold_z_offset
    start_clearance = max(0.0, float(getattr(args, "micro_unfold_active_start_clearance_mm", 18.0)))
    active_start_z = active_contact_z + start_clearance + active_z_offset
    max_below = max(0.0, float(getattr(args, "micro_unfold_max_below_contact_mm", 2.0)))
    raw_floor = max(float(getattr(cfg, "min_safe_contact_z", -140.0)), active_contact_z - max_below)
    active_floor_z = raw_floor + active_z_offset

    def _angle(name, default):
        return float(getattr(args, name, default))

    hold_gripper = _angle(
        "micro_unfold_hold_arm1_gripper" if hold_arm == "arm1" else "micro_unfold_hold_arm2_gripper",
        2.818 if hold_arm == "arm1" else 2.772,
    )
    # E131 keeps E100 physical CLOSED_REF semantics but removes the contact
    # OPEN20 -> CLOSE15 acquisition stroke entirely. ACTIVE stays at the
    # physical CLOSED_REF through descent, Z-locked pull and diagonal peel.
    # Full release remains a separate angle after the peel.
    active_closed_ref_nominal = _angle(
        "micro_unfold_active_arm1_closed_ref" if active_arm == "arm1" else "micro_unfold_active_arm2_closed_ref",
        2.900 if active_arm == "arm1" else 2.890,
    )
    active_approach_closed = float(active_closed_ref_nominal)
    # Legacy OPEN/CLOSE CLI options remain parsed for command-line compatibility,
    # but E131 deliberately does not execute them.  Store CLOSED_REF for both
    # compatibility fields so diagnostics cannot imply a pinch that never occurs.
    release_deg = float(getattr(args, "micro_unfold_active_release_deg", 45.0))
    active_open = float(active_closed_ref_nominal)
    active_close = float(active_closed_ref_nominal)
    active_release = float(active_closed_ref_nominal - np.deg2rad(release_deg))
    active_gripper_stroke_deg = 0.0

    _vla_slim_full_plan = bool(getattr(args, "micro_unfold_vla_slim_full_plan", True))

    plan_out = {
        "state": "READY",
        "method": "E112_E111_NECK_SOFT_VETO_HOLD_CLEAN_ORIENTATION_PEEL",
        "active_rank": int(active_rank),
        "active_candidate_id": str(depth_state.get("selected_candidate_id", "C?")),
        "active_donor_rank": int(depth_state.get("selected_donor_rank", 1)),
        "manual_override": bool(depth_state.get("manual_override", False)),
        "manual_override_candidate_id": str(depth_state.get("manual_override_candidate_id", "")),
        "hypothesis_score": float(depth_state.get("hypothesis_score", 0.0)),
        "hypothesis_score_gap": float(depth_state.get("hypothesis_score_gap", 9.999)),
        "hypothesis_ambiguous": bool(depth_state.get("hypothesis_ambiguous", False)),
        "garment_center_px": garment_center.tolist(),
        "active_px": active_px.tolist(),
        "hold_px": hold_px.tolist(),
        "active_board": active_board.tolist(),
        "hold_board": hold_board.tolist(),
        "pull_board": pull_board.tolist(),
        "pull_unit_board": np.asarray(pull_u, dtype=np.float64).tolist(),
        "pull_hint_unit_board": np.asarray(pull_hint_u, dtype=np.float64).tolist(),
        "e130_directional_plan": bool(active_meta.get("e130_directional_plan", False)),
        "e130_active_to_contour_mm": float(active_meta.get("e130_active_to_contour_mm", garment_exit_mm)),
        "pull_distance_mm": float(used_pull),
        "hold_boundary_distance_mm": float(hold_boundary_distance_mm),
        "boundary_normal_delta_deg": float(boundary_normal_delta_deg),
        "boundary_normal_ok": bool(boundary_normal_ok),
        "garment_exit_distance_mm": float(garment_exit_mm),
        "outside_margin_used_mm": float(outside_margin_used),
        "pull_endpoint_px": np.asarray(pull_endpoint_px, dtype=np.float64).tolist(),
        "pair_separation_mm": pair_sep,
        "hold_inward_alignment_cos": float(hold_alignment),
        "hold_inward_cone_deg": float(selected_cone_deg),
        "hold_min_separation_gate_mm": float(selected_min_sep_mm),
        "hold_selection_score": float(hold_score),
        "hold_target_board": np.asarray(hold_target_board, dtype=np.float64).tolist(),
        "hold_target_distance_mm": float(hold_target_distance),
        "hold_target_error_mm": float(hold_target_error),
        "hold_local_search_radius_mm": float(selected_local_radius_mm),
        "active_center_distance_px": float(active_center_r),
        **active_meta,
        "hold_center_distance_px": float(np.linalg.norm(hold_px - garment_center)),
        "active_prominence_mm": float(prom[int(round(active_px[1])), int(round(active_px[0]))]),
        "active_relief_mm": float(relief[int(round(active_px[1])), int(round(active_px[0]))]),
        "hold_prominence_mm": float(prom[int(round(hold_px[1])), int(round(hold_px[0]))]),
        "hold_relief_mm": float(relief[int(round(hold_px[1])), int(round(hold_px[0]))]),
        **assignment,
        "active_contact_z": active_contact_z,
        "hold_contact_z": hold_contact_z,
        "hold_z_offset_mm": float(hold_z_offset),
        "active_z_offset_mm": float(active_z_offset),
        "hold_action_z": float(hold_action_z),
        "active_start_z": active_start_z,
        "active_floor_z": active_floor_z,
        "hold_gripper": hold_gripper,
        "active_open": active_open,
        "active_close": active_close,
        "active_release": active_release,
        "active_gripper_stroke_deg": float(active_gripper_stroke_deg),
        "peel_local_xy": np.asarray(peel_local_xy, dtype=np.float64).tolist(),
        "peel_xy_mm": float(peel_xy_used),
        "peel_z_mm": float(peel_z_rise),
        "active_approach_closed": float(active_approach_closed),
        "active_closed_ref_nominal": float(active_closed_ref_nominal),
        **({} if _vla_slim_full_plan else {
            "coarse_lobe_region": np.asarray(crease_pair.get("coarse_lobe_region", np.zeros_like(labels)), dtype=np.uint8).copy() if isinstance(crease_pair, dict) else np.zeros_like(labels, dtype=np.uint8),
            "coarse_body_region": np.asarray(crease_pair.get("coarse_body_region", np.zeros_like(labels)), dtype=np.uint8).copy() if isinstance(crease_pair, dict) else np.zeros_like(labels, dtype=np.uint8),
            "selected_crease_component": np.asarray(crease_pair.get("selected_crease_component", np.zeros_like(labels)), dtype=np.uint8).copy() if isinstance(crease_pair, dict) else np.zeros_like(labels, dtype=np.uint8),
            "e125_primary_fold_region": np.asarray(crease_pair.get("primary_fold_region", depth_state.get("e125_primary_fold_region_mask", np.zeros_like(labels))), dtype=np.uint8).copy() if isinstance(crease_pair, dict) else np.asarray(depth_state.get("e125_primary_fold_region_mask", np.zeros_like(labels)), dtype=np.uint8).copy(),
            "e125_primary_fold_boundary": np.asarray(crease_pair.get("primary_fold_boundary", depth_state.get("e125_primary_fold_boundary_mask", np.zeros_like(labels))), dtype=np.uint8).copy() if isinstance(crease_pair, dict) else np.asarray(depth_state.get("e125_primary_fold_boundary_mask", np.zeros_like(labels)), dtype=np.uint8).copy(),
            "crease_flap_region": np.asarray(crease_pair.get("crease_flap_region", np.zeros_like(labels)), dtype=np.uint8).copy() if isinstance(crease_pair, dict) else np.zeros_like(labels, dtype=np.uint8),
            "crease_body_region": np.asarray(crease_pair.get("crease_body_region", np.zeros_like(labels)), dtype=np.uint8).copy() if isinstance(crease_pair, dict) else np.zeros_like(labels, dtype=np.uint8),
        }),
        "crease_partition_info": _training_json_safe(crease_pair.get("crease_partition_info", {})) if isinstance(crease_pair, dict) else {},
        "active_crease_distance_mm": float(crease_pair.get("active_crease_distance_mm", active_meta.get("active_crease_distance_px", 999.0))) if isinstance(crease_pair, dict) else float(active_meta.get("active_crease_distance_px", 999.0)),
        "hold_crease_distance_mm": float(crease_pair.get("hold_crease_distance_mm", 999.0)) if isinstance(crease_pair, dict) else 999.0,
        "outer_boundary_perpendicular_angle_deg": float(crease_pair.get("outer_boundary_perpendicular_angle_deg", 0.0)) if isinstance(crease_pair, dict) else 0.0,
        "pair_spacing_mode": str(crease_pair.get("pair_spacing_mode", "RGB_GEOMETRY")) if isinstance(crease_pair, dict) else "RGB_GEOMETRY",
        "outer_boundary_mode": str(crease_pair.get("outer_boundary_mode", "RGB_GEOMETRY")) if isinstance(crease_pair, dict) else "RGB_GEOMETRY",
        "hold_band_mode": str(crease_pair.get("hold_band_mode", "RGB_GEOMETRY")) if isinstance(crease_pair, dict) else "RGB_GEOMETRY",
        "active_inside_flap": bool(crease_pair.get("active_inside_flap", False)) if isinstance(crease_pair, dict) else False,
        "hold_inside_body": bool(crease_pair.get("hold_inside_body", False)) if isinstance(crease_pair, dict) else False,
        "selected_candidate_id": str(depth_state.get("selected_candidate_id", "NONE")),
        "fusion_mode": str(depth_state.get("fusion_mode", "UNKNOWN")),
        **({} if _vla_slim_full_plan else {
            "visual_top2_labels": np.asarray(depth_state.get("visual_top2_labels", labels), dtype=np.uint8).copy(),
            "visual_top2_info": _training_json_safe(depth_state.get("visual_top2_info", [])),
            "strict_seed_labels": np.asarray(depth_state.get("strict_seed_labels", np.zeros_like(labels)), dtype=np.uint8).copy(),
            "strict_candidate_labels": np.asarray(depth_state.get("strict_candidate_labels", np.zeros_like(labels)), dtype=np.uint8).copy(),
            "candidate_diagnostics": _training_json_safe(depth_state.get("candidate_diagnostics", [])),
            "mask_fold_candidate": np.asarray(depth_state.get("mask_fold_candidate", np.zeros_like(labels)), dtype=np.uint8).copy(),
            "garment_mask": np.asarray(depth_state.get("garment_mask", mask), dtype=np.uint8).copy(),
            "fold_crease_mask": np.asarray(depth_state.get("fold_crease_mask", np.zeros_like(labels)), dtype=np.uint8).copy(),
            "fold_crease_skeleton": np.asarray(depth_state.get("e122_fold_preview_skeleton", depth_state.get("fold_crease_skeleton", np.zeros_like(labels))), dtype=np.uint8).copy(),
            "e122_rejected_crease_skeleton": np.asarray(depth_state.get("e122_fold_rejected_skeleton", np.zeros_like(labels)), dtype=np.uint8).copy(),
            "e122_fold_branch_info": _training_json_safe(depth_state.get("e122_fold_branch_info", [])),
            "fold_crease_info": _training_json_safe(depth_state.get("fold_crease_info", {})),
            "baseline_depth": depth_state,
        }),
    }

    # V12 local-only exact-plan provenance. Full masks exist only until the final
    # role validator finishes; freeze_execution_plan() keeps only the compact crease line.
    if isinstance(crease_pair, dict):
        _exact=np.asarray(crease_pair.get("selected_crease_component",np.zeros_like(labels)),dtype=np.uint8)
        _roi=np.asarray(crease_pair.get("primary_fold_region",depth_state.get("e125_primary_fold_region_mask",np.zeros_like(labels))),dtype=np.uint8)
        if _exact.shape==labels.shape and cv2.countNonZero(_exact)>0:
            plan_out["selected_crease_component"]=_exact.copy()
            plan_out["v11_exact_crease_pixels"]=int(cv2.countNonZero(_exact))
        if _roi.shape==labels.shape and cv2.countNonZero(_roi)>0:
            plan_out["primary_fold_region"]=_roi.copy()
        _anchor=crease_pair.get("crease_anchor_px",None)
        if _anchor is not None:
            try: plan_out["crease_anchor_px"]=np.asarray(_anchor,dtype=np.float64).reshape(2).tolist()
            except Exception: pass
        plan_out["v11_exact_crease_source"]=str(crease_pair.get("crease_source","UNKNOWN"))
        plan_out["v11_exact_component_label"]=int(crease_pair.get("component_label",-1))
    return plan_out


def _fix53_deformation_report(before_state, after_state, plan, args):
    if not bool((after_state or {}).get("available", False)):
        return {"changed": False, "valid": False, "reason": "AFTER_DEPTH_UNAVAILABLE"}
    bmask = np.asarray(before_state["garment_mask"], dtype=np.uint8) > 0
    amask = np.asarray(after_state["garment_mask"], dtype=np.uint8) > 0
    if bmask.shape != amask.shape:
        return {"changed": False, "valid": False, "reason": "AFTER_SHAPE_MISMATCH"}
    inter = int(np.count_nonzero(bmask & amask))
    union = int(np.count_nonzero(bmask | amask))
    mask_iou = float(inter) / float(max(1, union))

    rank = int(plan.get("active_rank", 1))
    before_labels = np.asarray(before_state.get("visual_top2_labels", before_state["labels"]), dtype=np.uint8)
    after_labels = np.asarray(after_state.get("visual_top2_labels", after_state["labels"]), dtype=np.uint8)
    if rank == 0:
        # E108 AUTO geometry is not defined by CREASE FLAP REGION.  Use the selected
        # FUSED FOLD as the deformation-reference region when available.
        active_region = (before_labels == 1)
        if int(np.count_nonzero(active_region)) < 8 and plan.get("crease_flap_region") is not None:
            active_region = np.asarray(plan.get("crease_flap_region"), dtype=np.uint8) > 0
    else:
        active_region = before_labels == rank
    raised_after = after_labels > 0
    active_area = int(np.count_nonzero(active_region))
    raised_overlap = float(np.count_nonzero(active_region & raised_after)) / float(max(1, active_area)) if rank != 0 else 1.0

    br = np.asarray(before_state["local_relief_mm"], dtype=np.float32)
    ar = np.asarray(after_state["local_relief_mm"], dtype=np.float32)
    bp = np.asarray(before_state["prominence_mm"], dtype=np.float32)
    ap = np.asarray(after_state["prominence_mm"], dtype=np.float32)
    valid_roi = active_region & np.isfinite(br) & np.isfinite(ar) & np.isfinite(bp) & np.isfinite(ap)
    if int(np.count_nonzero(valid_roi)) < 8:
        return {"changed": False, "valid": False, "reason": "ACTIVE_ROI_TOO_SMALL",
                "mask_iou": mask_iou, "raised_overlap": raised_overlap}
    relief_before = float(np.median(br[valid_roi]))
    relief_after = float(np.median(ar[valid_roi]))
    prom_before = float(np.median(bp[valid_roi]))
    prom_after = float(np.median(ap[valid_roi]))
    relief_abs_delta = float(np.median(np.abs(ar[valid_roi] - br[valid_roi])))
    prominence_abs_delta = float(np.median(np.abs(ap[valid_roi] - bp[valid_roi])))
    relief_drop = relief_before - relief_after
    prominence_drop = prom_before - prom_after

    depth_gate = max(0.5, float(getattr(args, "micro_unfold_depth_change_mm", 1.5)))
    overlap_gate = float(getattr(args, "micro_unfold_raised_overlap_threshold", 0.65))
    mask_iou_gate = float(getattr(args, "micro_unfold_mask_iou_threshold", 0.955))
    depth_changed = bool(
        relief_abs_delta >= depth_gate
        or abs(relief_drop) >= depth_gate
        or prominence_abs_delta >= depth_gate
        or abs(prominence_drop) >= depth_gate
    )
    raised_changed = bool(raised_overlap <= overlap_gate)
    silhouette_changed = bool(mask_iou <= mask_iou_gate)
    changed = bool(depth_changed or (raised_changed and silhouette_changed))
    return {
        "changed": changed, "valid": True, "reason": "CHANGED" if changed else "UNCHANGED",
        "mask_iou": mask_iou, "raised_overlap": raised_overlap,
        "relief_before_mm": relief_before, "relief_after_mm": relief_after,
        "relief_drop_mm": relief_drop, "relief_abs_delta_mm": relief_abs_delta,
        "prominence_before_mm": prom_before, "prominence_after_mm": prom_after,
        "prominence_drop_mm": prominence_drop, "prominence_abs_delta_mm": prominence_abs_delta,
        "depth_gate_mm": depth_gate, "overlap_gate": overlap_gate, "mask_iou_gate": mask_iou_gate,
        "signals": {"depth_changed": depth_changed, "raised_changed": raised_changed,
                    "silhouette_changed": silhouette_changed},
    }


def _fix53_draw_overlay(canvas):
    plan = globals().get("_FIX53_MICRO_UNFOLD_PLAN")
    status = str(globals().get("_FIX53_MICRO_UNFOLD_STATUS", "IDLE"))
    if not isinstance(plan, dict) or str(plan.get("state", "")) not in ("READY", "DIAGNOSTIC"):
        return canvas
    out = canvas.copy()
    try:
        # E109: `canvas` is already the clean latest RAW RGB frame supplied by the display loop.
        # Inference geometry remains frozen from the last I judgment; only the background stays live.
        # E92: always show the successful garment MASK even if HOLD planning fails.
        garment_vis = plan.get("garment_mask")
        if garment_vis is not None:
            gv = ((np.asarray(garment_vis, dtype=np.uint8) > 0).astype(np.uint8) * 255)
            if gv.shape[:2] == out.shape[:2] and cv2.countNonZero(gv) > 0:
                gcs, _ = cv2.findContours(gv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if gcs:
                    # E126 display uses the authoritative largest external cloth
                    # contour only; small detached fallback artifacts are diagnostic
                    # noise and must not look like garment geometry.
                    _gmain=max(gcs,key=cv2.contourArea)
                    cv2.drawContours(out, [_gmain], -1, (0, 220, 255), 2, cv2.LINE_AA)
        # E102 pose-derived known sewing seams excluded from Fold Crease.
        # Overlay runs in the preview loop, where no local `depth_state` exists.
        # Always read the frozen I-state stored inside the plan instead.
        _e102_depth_vis = plan.get("baseline_depth", {}) if isinstance(plan, dict) else {}
        if not isinstance(_e102_depth_vis, dict):
            _e102_depth_vis = {}
        seam_excl=np.asarray(_e102_depth_vis.get("pose_seam_exclusion_mask",np.zeros(out.shape[:2],dtype=np.uint8)),dtype=np.uint8)
        if seam_excl.shape[:2]==out.shape[:2] and cv2.countNonZero(seam_excl)>0:
            seam_cs,_=cv2.findContours((seam_excl>0).astype(np.uint8)*255,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(out,seam_cs,-1,(180,80,255),1,cv2.LINE_AA)
            _pf_group=str(plan.get("e125_primary_fold_group",_e102_depth_vis.get("e125_primary_fold_group","NONE")))
            _pf_src=str(plan.get("e125_primary_fold_source",_e102_depth_vis.get("e125_primary_fold_source","")))
            _neck_override=("NECK" in set(_pf_group.split("+")) and (_pf_src=="POSE_INTRUSION" or _pf_src.startswith("POSE_DIRECTION")))
            _neck_text="NECK FOLD OVERRIDE" if _neck_override else "NECKLINE EXCLUDED"
            cv2.putText(out,_neck_text,(18,176),cv2.FONT_HERSHEY_SIMPLEX,0.48,(180,80,255),2,cv2.LINE_AA)

        # E117: show all available 12-point T-shirt pose landmarks as blue dots.
        # Diagnostic-only: pose points never authorize a robot target by themselves.
        _pose117 = dict(_e102_depth_vis.get("tshirt_pose_state", {}) or {})
        _named117 = dict(_pose117.get("named_points", {}) or {})
        _intruded117 = {str(q.get("name","")) for q in list(_e102_depth_vis.get("pose_fold_intrusions", []) or []) if isinstance(q,dict)}
        for _idx117, _name117 in enumerate(_E102_TSHIRT_POSE_KPT_NAMES):
            _q117 = dict(_named117.get(_name117, {}) or {})
            try:
                _x117=float(_q117.get("x")); _y117=float(_q117.get("y")); _c117=float(_q117.get("conf",0.0))
            except Exception:
                continue
            if not (np.isfinite(_x117) and np.isfinite(_y117) and np.isfinite(_c117)) or _c117 < 0.20:
                continue
            _p117=(int(round(_x117)),int(round(_y117)))
            cv2.circle(out,_p117,6,(255,255,255),-1,cv2.LINE_AA)
            cv2.circle(out,_p117,4,(255,0,0),-1,cv2.LINE_AA)
            if _name117 in _intruded117:
                cv2.circle(out,_p117,9,(255,255,0),2,cv2.LINE_AA)
            cv2.putText(out,str(_idx117),(_p117[0]+6,_p117[1]-5),cv2.FONT_HERSHEY_SIMPLEX,0.36,(255,255,255),2,cv2.LINE_AA)
            cv2.putText(out,str(_idx117),(_p117[0]+6,_p117[1]-5),cv2.FONT_HERSHEY_SIMPLEX,0.36,(255,0,0),1,cv2.LINE_AA)

        # E130 FOLD EXTENT: lime outline is the actual directional fold extent from selected crease to Cloth Contour.
        # Pose intrusion is direction evidence only; it is no longer treated as the full folded area.
        _fr125 = plan.get("e125_primary_fold_region")
        if _fr125 is None:
            _fr125 = _e102_depth_vis.get("e125_primary_fold_region_mask")
        if _fr125 is not None:
            _fr125=np.asarray(_fr125,dtype=np.uint8)
            if _fr125.shape[:2]==out.shape[:2] and cv2.countNonZero(_fr125)>0:
                _fcs125,_=cv2.findContours(((_fr125>0).astype(np.uint8)*255),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
                if _fcs125:
                    cv2.drawContours(out,_fcs125,-1,(0,255,128),3,cv2.LINE_AA)
                    _fm=max(_fcs125,key=cv2.contourArea); _mm=cv2.moments(_fm)
                    if abs(float(_mm.get("m00",0.0)))>1e-6:
                        _fx=int(round(_mm["m10"]/_mm["m00"])); _fy=int(round(_mm["m01"]/_mm["m00"]))
                        _src=str(plan.get("e125_primary_fold_source",_e102_depth_vis.get("e125_primary_fold_source","?")))
                        _cf=float(plan.get("e125_primary_fold_confidence",_e102_depth_vis.get("e125_primary_fold_confidence",0.0)) or 0.0)
                        cv2.putText(out,f"PRIMARY FOLD {_src} {_cf:.2f}",(_fx+8,_fy-10),cv2.FONT_HERSHEY_SIMPLEX,0.46,(0,0,0),4,cv2.LINE_AA)
                        cv2.putText(out,f"PRIMARY FOLD {_src} {_cf:.2f}",(_fx+8,_fy-10),cv2.FONT_HERSHEY_SIMPLEX,0.46,(0,255,128),2,cv2.LINE_AA)

        # E109: show only the coarse crease+outer-contour lobe as a thin diagnostic outline.
        # It is used for side ownership, never as a hard grasp-domain mask. No filled MASK is drawn.
        lobe_vis = plan.get("coarse_lobe_region")
        if lobe_vis is not None:
            lv = np.asarray(lobe_vis, dtype=np.uint8)
            if lv.shape[:2] == out.shape[:2] and cv2.countNonZero(lv) > 0:
                lcs, _ = cv2.findContours(lv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if lcs:
                    cv2.drawContours(out, lcs, -1, (145,145,145), 1, cv2.LINE_AA)
                    largest_l=max(lcs,key=cv2.contourArea); ml=cv2.moments(largest_l)
                    if abs(float(ml.get("m00",0.0)))>1e-6:
                        lx=int(round(ml["m10"]/ml["m00"])); ly=int(round(ml["m01"]/ml["m00"]))
                        cv2.putText(out,"CREASE+OUTER LOBE",(lx+8,ly+18),cv2.FONT_HERSHEY_SIMPLEX,0.40,(145,145,145),1,cv2.LINE_AA)
        # E109 clean diagnostics: no translucent garment MASK / heatmap / seed fills.
        # FUSED FOLD remains only as a thin weak-reference contour.
        visual_labels = plan.get("visual_top2_labels")
        if visual_labels is not None:
            vl = np.asarray(visual_labels, dtype=np.uint8)
            if vl.shape[:2] == out.shape[:2]:
                region = ((vl == 1).astype(np.uint8) * 255)
                if cv2.countNonZero(region) > 0:
                    contours, _ = cv2.findContours(region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    if contours:
                        cv2.drawContours(out, contours, -1, (180,0,180), 3, cv2.LINE_AA)
                    mmf = cv2.moments(region, binaryImage=True)
                    if abs(float(mmf.get("m00",0.0))) > 1e-6:
                        cx=int(round(float(mmf["m10"]/mmf["m00"]))); cy=int(round(float(mmf["m01"]/mmf["m00"])))
                        _fused_ref_only=bool(str(plan.get("e125_primary_fold_source",_e102_depth_vis.get("e125_primary_fold_source","")))=="POSE_INTRUSION")
                        _fused_txt="FUSED REF ONLY" if _fused_ref_only else "FUSED FOLD"
                        cv2.putText(out,_fused_txt,(cx+8,cy-8),cv2.FONT_HERSHEY_SIMPLEX,0.52,(255,255,255),4,cv2.LINE_AA)
                        cv2.putText(out,_fused_txt,(cx+8,cy-8),cv2.FONT_HERSHEY_SIMPLEX,0.52,(180,0,180),2,cv2.LINE_AA)
        # E122 branch classifier: raw high-recall branches rejected as wrinkles/weak
        # boundaries are gray; PREVIEW/EXECUTION-qualified fold boundaries are cyan.
        rejected_vis = plan.get("e122_rejected_crease_skeleton")
        if rejected_vis is not None:
            rj=np.asarray(rejected_vis,dtype=np.uint8)
            if rj.shape[:2]==out.shape[:2] and cv2.countNonZero(rj)>0:
                rj_show=cv2.dilate(((rj>0).astype(np.uint8)*255),np.ones((3,3),np.uint8),iterations=1)
                rcs,_=cv2.findContours(rj_show,cv2.RETR_LIST,cv2.CHAIN_APPROX_NONE)
                if rcs: cv2.drawContours(out,rcs,-1,(115,115,115),1,cv2.LINE_AA)
        for _bi in plan.get("e122_fold_branch_info",[]) if isinstance(plan.get("e122_fold_branch_info",[]),(list,tuple)) else []:
            try:
                _c=_bi.get('centroid_px'); _s=float(_bi.get('fold_score',0.0)); _bid=int(_bi.get('branch_id',0))
                if _c is None: continue
                _p=(int(round(float(_c[0]))),int(round(float(_c[1]))))
                _col=(255,255,0) if bool(_bi.get('preview_ok',False)) else (145,145,145)
                cv2.putText(out,f"B{_bid}:{_s:.2f}",(_p[0]+5,_p[1]-5),cv2.FONT_HERSHEY_SIMPLEX,0.34,(0,0,0),3,cv2.LINE_AA)
                cv2.putText(out,f"B{_bid}:{_s:.2f}",(_p[0]+5,_p[1]-5),cv2.FONT_HERSHEY_SIMPLEX,0.34,_col,1,cv2.LINE_AA)
            except Exception:
                pass
        # E122 fold-boundary candidate: cyan curve. Preview may remain visible while
        # X is blocked until the branch passes the higher execution threshold.
        crease_vis = plan.get("fold_crease_skeleton")
        if crease_vis is not None:
            cr = np.asarray(crease_vis, dtype=np.uint8)
            if cr.shape[:2] == out.shape[:2] and cv2.countNonZero(cr) > 0:
                cr_show = cv2.dilate(((cr > 0).astype(np.uint8) * 255), np.ones((3,3), np.uint8), iterations=1)
                ccs, _ = cv2.findContours(cr_show, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
                if ccs:
                    cv2.drawContours(out, ccs, -1, (255,255,0), 2, cv2.LINE_AA)
                    largest_cr = max(ccs, key=lambda c: cv2.arcLength(c, True))
                    mmc = cv2.moments(largest_cr)
                    if abs(float(mmc.get("m00",0.0))) > 1e-6:
                        crx=int(round(mmc["m10"]/mmc["m00"])); cry=int(round(mmc["m01"]/mmc["m00"]))
                        cv2.putText(out, "FOLD BOUNDARY", (crx+8,cry-8), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0,0,0), 4, cv2.LINE_AA)
                        cv2.putText(out, "FOLD BOUNDARY", (crx+8,cry-8), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255,255,0), 2, cv2.LINE_AA)
        # E112: highlight the component that actually produced HOLD/ACTIVE.
        selected_cr = plan.get("selected_crease_component")
        if selected_cr is not None:
            scr=np.asarray(selected_cr,dtype=np.uint8)
            if scr.shape[:2]==out.shape[:2] and cv2.countNonZero(scr)>0:
                scr_show=cv2.dilate(((scr>0).astype(np.uint8)*255),np.ones((3,3),np.uint8),iterations=1)
                scs,_=cv2.findContours(scr_show,cv2.RETR_LIST,cv2.CHAIN_APPROX_NONE)
                if scs:
                    cv2.drawContours(out,scs,-1,(0,0,0),7,cv2.LINE_AA)
                    cv2.drawContours(out,scs,-1,(0,255,0),4,cv2.LINE_AA)
                    largest_sc=max(scs,key=lambda c:cv2.arcLength(c,True))
                    msc=cv2.moments(largest_sc)
                    if abs(float(msc.get("m00",0.0)))>1e-6:
                        sx=int(round(msc["m10"]/msc["m00"])); sy=int(round(msc["m01"]/msc["m00"]))
                        cv2.putText(out,"SELECTED CREASE",(sx+8,sy+18),cv2.FONT_HERSHEY_SIMPLEX,0.44,(0,0,0),4,cv2.LINE_AA)
                        cv2.putText(out,"SELECTED CREASE",(sx+8,sy+18),cv2.FONT_HERSHEY_SIMPLEX,0.44,(0,255,0),2,cv2.LINE_AA)
        # E119: topology-deformation is the primary folded-side cue.  Pose-count
        # split remains visible one line below as a secondary diagnostic.
        _pd_mode=str(plan.get("e119_pose_deform_mode","NONE"))
        _pd_conf=float(plan.get("e119_pose_deform_confidence",0.0) or 0.0)
        _pd_side=str(plan.get("e119_pose_deform_side_hint","NONE"))
        _pd_role=float(plan.get("e119_pose_deform_role_support",0.5) or 0.5)
        if _pd_conf>0.0 or _pd_mode!="NONE":
            _pd_txt=f"POSE DEFORM {_pd_mode} conf={_pd_conf:.2f} side={_pd_side} role={_pd_role:.2f}"
            cv2.putText(out,_pd_txt,(18,154),cv2.FONT_HERSHEY_SIMPLEX,0.44,(0,0,0),4,cv2.LINE_AA)
            cv2.putText(out,_pd_txt,(18,154),cv2.FONT_HERSHEY_SIMPLEX,0.44,(255,180,0),2,cv2.LINE_AA)
        _ps_l=int(plan.get("e118_pose_split_lobe_count",0) or 0)
        _ps_b=int(plan.get("e118_pose_split_body_count",0) or 0)
        _ps_conf=float(plan.get("e118_pose_split_confidence",0.0) or 0.0)
        if (_ps_l+_ps_b)>0:
            _ps_txt=f"POSE COUNT L/B={_ps_l}/{_ps_b} conf={_ps_conf:.2f} (secondary)"
            cv2.putText(out,_ps_txt,(18,176),cv2.FONT_HERSHEY_SIMPLEX,0.42,(0,0,0),4,cv2.LINE_AA)
            cv2.putText(out,_ps_txt,(18,176),cv2.FONT_HERSHEY_SIMPLEX,0.42,(255,180,0),2,cv2.LINE_AA)
        # E92 diagnostic only: show the three UniDepth seed centroids without
        # authorizing three ACTIVE points. The final large magenta X below is the
        # single selected executable ACTIVE candidate.
        candidate_diag = plan.get("candidate_diagnostics", [])
        candidate_colors = {1: (0, 0, 255), 2: (255, 0, 0), 3: (0, 255, 255)}
        for item in candidate_diag if isinstance(candidate_diag, (list, tuple)) else []:
            try:
                rank = int(item.get("donor_rank", 0))
                p = item.get("seed_centroid_px")
                if rank not in candidate_colors or p is None:
                    continue
                cp = tuple(np.round(np.asarray(p, dtype=np.float64)).astype(int).tolist())
                color = candidate_colors[rank]
                is_selected = bool(item.get("selected", False))
                cv2.circle(out, cp, 11 if is_selected else 6, color, 3 if is_selected else 2, cv2.LINE_AA)
                if is_selected and bool(plan.get("manual_override", False)):
                    cv2.circle(out, cp, 15, (255,0,255), 2, cv2.LINE_AA)
                label = f"C{rank}{' -> ACTIVE' if (is_selected and bool(plan.get('manual_override', False))) else ('*' if is_selected else '')}"
                cv2.putText(out, label, (cp[0]+8, cp[1]+18), cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 2, cv2.LINE_AA)
            except Exception:
                continue
        hp = None
        ap = None
        if plan.get("hold_px") is not None:
            hp = tuple(np.round(np.asarray(plan["hold_px"], dtype=np.float64)).astype(int).tolist())
        if plan.get("active_px") is not None:
            ap = tuple(np.round(np.asarray(plan["active_px"], dtype=np.float64)).astype(int).tolist())
        H_value = np.asarray(plan.get("H"), dtype=np.float64) if plan.get("H") is not None else None
        # E147 visualization-only change: show the exact planned ACTIVE -> pull
        # endpoint trajectory whenever HOLD/ACTIVE geometry exists, even when the
        # plan was later downgraded to DIAGNOSTIC/PREVIEW-ONLY.  The endpoint is
        # the same ``pull_board`` consumed by X; no motion/planning coordinate is
        # recomputed here.  READY remains executable, DIAGNOSTIC is display-only.
        pull_px = None
        _plan_state = str(plan.get("state", "DIAGNOSTIC"))
        if _plan_state in ("READY", "DIAGNOSTIC") and H_value is not None and H_value.shape == (3, 3) and plan.get("pull_board") is not None:
            H_inv = np.linalg.inv(H_value)
            pull_px_arr = _fix12_transform_points(H_inv, np.asarray([plan["pull_board"]], dtype=np.float64))[0]
            pull_px = tuple(np.round(pull_px_arr).astype(int).tolist())
        _sel_mode = str(plan.get("active_selection_mode", ""))
        auto_crease_pair = ("FOLD_CREASE" in _sel_mode) and not bool(plan.get("manual_override", False))
        hold_color = (255,0,0) if auto_crease_pair else (0,255,0)
        active_color = (0,255,255) if auto_crease_pair else (255,0,255)
        if hp is not None:
            cv2.drawMarker(out, hp, hold_color, cv2.MARKER_CROSS, 36, 5, cv2.LINE_AA)
            cv2.putText(out, f"HOLD {str(plan.get('hold_arm','')).upper()}", (hp[0]+12,hp[1]-12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.58, hold_color, 2, cv2.LINE_AA)
        _e151_cluster_block=bool(plan.get('e151_edge_cluster_hard_block',False))
        if ap is not None:
            _pose_active_mode=('POSE_INTRUSION' in _sel_mode)
            if not _e151_cluster_block:
                cv2.drawMarker(out, ap, active_color, cv2.MARKER_TILTED_CROSS, 38, 5, cv2.LINE_AA)
                if str(plan.get("state")) == "READY":
                    active_label = f"ACTIVE {str(plan.get('active_arm','')).upper()} POSE+RGB" if _pose_active_mode else f"ACTIVE {str(plan.get('active_arm','')).upper()} FOLD"
                else:
                    active_label = "ACTIVE DIR+RGB" if _pose_active_mode else "ACTIVE CANDIDATE"
                cv2.putText(out, active_label, (ap[0]+12,ap[1]-12), cv2.FONT_HERSHEY_SIMPLEX, 0.58, active_color, 2, cv2.LINE_AA)
        if hp is not None and ap is not None and not _e151_cluster_block:
            cv2.line(out, hp, ap, (0,165,255), 3, cv2.LINE_AA)
        if ap is not None and pull_px is not None and not _e151_cluster_block:
            # READY: solid/thick orange = the trajectory X will execute.
            # DIAGNOSTIC: thinner amber = geometrically planned preview only; X
            # may still be blocked by the higher-level execution gate.
            _traj_ready = (_plan_state == "READY")
            _traj_color = (0,128,255) if _traj_ready else (0,190,255)
            _traj_thick = 5 if _traj_ready else 3
            cv2.arrowedLine(out, ap, pull_px, _traj_color, _traj_thick, cv2.LINE_AA, tipLength=0.20)
            cv2.circle(out, pull_px, 8 if _traj_ready else 6, _traj_color, 2, cv2.LINE_AA)
            _mx=int(round((ap[0]+pull_px[0])*0.5)); _my=int(round((ap[1]+pull_px[1])*0.5))
            _pd=float(plan.get("pull_distance_mm",0.0) or 0.0)
            _traj_txt=(f"UNFOLD PATH {_pd:.0f}mm" if _traj_ready else f"PREVIEW PATH {_pd:.0f}mm | X BLOCKED")
            cv2.putText(out,_traj_txt,(_mx+8,_my-8),cv2.FONT_HERSHEY_SIMPLEX,0.43,(0,0,0),4,cv2.LINE_AA)
            cv2.putText(out,_traj_txt,(_mx+8,_my-8),cv2.FONT_HERSHEY_SIMPLEX,0.43,_traj_color,2,cv2.LINE_AA)
        if _e151_cluster_block:
            _e151_info=dict(plan.get('e151_edge_cluster_block_info',{}) or {})
            _e151_txt=f"EDGE CLUSTER {str(_e151_info.get('side','NONE'))} FOUND | NO SAFE EXACT-NORMAL PAIR | X BLOCKED"
            cv2.putText(out,_e151_txt,(18,198),cv2.FONT_HERSHEY_SIMPLEX,0.50,(0,0,0),5,cv2.LINE_AA)
            cv2.putText(out,_e151_txt,(18,198),cv2.FONT_HERSHEY_SIMPLEX,0.50,(0,165,255),2,cv2.LINE_AA)
        if bool(plan.get("hypothesis_ambiguous", False)):
            cv2.putText(out, "FOLD_AMBIGUOUS | X BLOCKED", (18, 132), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0,0,0), 5, cv2.LINE_AA)
            cv2.putText(out, "FOLD_AMBIGUOUS | X BLOCKED", (18, 132), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0,165,255), 2, cv2.LINE_AA)
        state_txt = str(plan.get("state", "DIAGNOSTIC"))
        manual_candidate = str(plan.get("manual_override_candidate_id", "")) if bool(plan.get("manual_override", False)) else ""
        manual_txt = f" | MANUAL={manual_candidate}" if manual_candidate else ""
        _pose_active_mode=('POSE_INTRUSION' in _sel_mode)
        mode_txt = ("POSE_ACTIVE" if _pose_active_mode else ("CREASE_AUTO" if auto_crease_pair else (f"MANUAL_{manual_candidate}" if manual_candidate else "RGB_GEOMETRY")))
        banner = f"E155 RGB MICRO-UNFOLD | UNIDEPTH OFF | BBOX BRANCH FAST-I | E154 LOWER/EDGE | SINGLE-PASS CACHE | SYNC DOWN + FAST MOTION | LIVE RGB | MODE={mode_txt}{manual_txt} | STATE={state_txt} | I=NEW X=EXECUTE"
        cv2.putText(out, banner, (18, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.54, (0,0,0), 4, cv2.LINE_AA)
        cv2.putText(out, banner, (18, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.54, (0,255,255), 2, cv2.LINE_AA)
        help_txt = "RGB-only: Pose direction -> Fold Crease -> HOLD/ACTIVE | 1/2/3 disabled | X executes READY"
        cv2.putText(out, help_txt, (18, 126), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0,0,0), 4, cv2.LINE_AA)
        cv2.putText(out, help_txt, (18, 126), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0,255,255), 2, cv2.LINE_AA)
        if manual_candidate:
            confirm_txt = f"SELECTED {manual_candidate} -> ACTIVE | magenta X = executable grasp | press X to run"
            cv2.putText(out, confirm_txt, (18, 152), cv2.FONT_HERSHEY_SIMPLEX, 0.54, (0,0,0), 4, cv2.LINE_AA)
            cv2.putText(out, confirm_txt, (18, 152), cv2.FONT_HERSHEY_SIMPLEX, 0.54, (255,0,255), 2, cv2.LINE_AA)
    except Exception as exc:
        print(f"[E102-OVERLAY-WARN] {exc!r}")
        return canvas
    return out


# =============================================================================
# E137 RGB + POSE + FOLD-CREASE OVERRIDES -- UNIDEPTH REMOVED
# =============================================================================

def _e144_outer_contour_inside_map(garment_u8):
    """Pixel distance to the EXTERNAL Cloth Contour; internal mask holes are ignored."""
    garment=((np.asarray(garment_u8,dtype=np.uint8)>0).astype(np.uint8)*255)
    cs,_=cv2.findContours(garment,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    if not cs:
        return cv2.distanceTransform(garment,cv2.DIST_L2,5).astype(np.float32)
    outer=np.zeros_like(garment)
    cv2.drawContours(outer,[max(cs,key=cv2.contourArea)],-1,255,thickness=cv2.FILLED)
    return cv2.distanceTransform(outer,cv2.DIST_L2,5).astype(np.float32)


def _e144_branch_contour_distance_stats(branch_u8, garment_u8, H_value,
                                        threshold_mm=10.0, inside_px=None):
    """Robust physical distance of one Fold-Crease branch from Cloth Contour.

    A true interior fold may touch the silhouette at one endpoint, so minimum
    distance is intentionally NOT used as the veto criterion.  Median distance
    plus the fraction of branch pixels inside the 10-mm contour band are used.
    """
    branch=np.asarray(branch_u8,dtype=np.uint8)>0
    garment=((np.asarray(garment_u8,dtype=np.uint8)>0).astype(np.uint8)*255)
    yy,xx=np.where(branch)
    if xx.size<3 or H_value is None:
        return {'available':False,'median_mm':999.0,'p75_mm':999.0,
                'near_ratio':0.0,'mm_per_px':None}
    inside=np.asarray(inside_px,dtype=np.float32) if inside_px is not None else np.empty((0,0),dtype=np.float32)
    if inside.shape!=garment.shape:
        inside=_e144_outer_contour_inside_map(garment)
    ctr=np.asarray([float(np.mean(xx)),float(np.mean(yy))],dtype=np.float64)
    try:
        mm_per_px=float(_e98_local_mm_per_px(H_value,ctr))
    except Exception:
        mm_per_px=float('nan')
    if not np.isfinite(mm_per_px) or mm_per_px<=1e-6:
        return {'available':False,'median_mm':999.0,'p75_mm':999.0,
                'near_ratio':0.0,'mm_per_px':None}
    d_mm=np.asarray(inside[yy,xx],dtype=np.float64)*mm_per_px
    d_mm=d_mm[np.isfinite(d_mm)]
    if d_mm.size<3:
        return {'available':False,'median_mm':999.0,'p75_mm':999.0,
                'near_ratio':0.0,'mm_per_px':mm_per_px}
    th=max(0.0,float(threshold_mm))
    return {
        'available':True,
        'median_mm':float(np.median(d_mm)),
        'p75_mm':float(np.percentile(d_mm,75.0)),
        'near_ratio':float(np.mean(d_mm<=th)) if th>0.0 else 0.0,
        'mm_per_px':float(mm_per_px),
    }


def _e149_build_branch_shared_ctx(frame_bgr, garment_u8, structural_u8, pose_deform_u8, seam_u8, base_rgb=None):
    """E149 OPT1: compute branch-invariant maps once for one fresh I frame."""
    frame=np.asarray(frame_bgr,dtype=np.uint8)
    garment=np.asarray(garment_u8,dtype=np.uint8)>0
    if frame.ndim!=3 or frame.shape[:2]!=garment.shape:
        return {}
    base=base_rgb if isinstance(base_rgb,dict) else {}
    L=np.asarray(base.get('L',np.empty((0,0),dtype=np.float32)),dtype=np.float32)
    inside=np.asarray(base.get('inside',np.empty((0,0),dtype=np.float32)),dtype=np.float32)
    if L.shape!=garment.shape:
        L=cv2.cvtColor(frame,cv2.COLOR_BGR2LAB)[:,:,0].astype(np.float32)
    if inside.shape!=garment.shape:
        inside=cv2.distanceTransform((garment.astype(np.uint8)*255),cv2.DIST_L2,5).astype(np.float32)
    struct=np.asarray(structural_u8,dtype=np.uint8)>0
    if np.any(struct):
        struct_near=cv2.dilate((struct.astype(np.uint8)*255),_e149_cv_kernel('ones',5),iterations=1)>0
    else:
        struct_near=np.zeros(garment.shape,dtype=bool)
    deform=np.asarray(pose_deform_u8,dtype=np.uint8)>0
    if np.any(deform):
        deform_near=cv2.dilate((deform.astype(np.uint8)*255),_e149_cv_kernel('ellipse',25),iterations=1)>0
    else:
        deform_near=np.zeros(garment.shape,dtype=bool)
    seam=np.asarray(seam_u8,dtype=np.uint8)>0
    return {'L':L,'inside':inside,'garment':garment,'struct_near':struct_near,'deform_near':deform_near,'seam':seam}


def _e122_score_fold_branch(branch_u8, frame_bgr, relief_mm, garment_u8, raw_score_map,
                            structural_u8, pose_deform_u8, pose_deform_conf,
                            seam_u8, shared_ctx=None):
    """E149 RGB-only fold-boundary score; E137 math with invariant maps hoisted out."""
    branch=np.asarray(branch_u8,dtype=np.uint8)>0
    frame=np.asarray(frame_bgr,dtype=np.uint8)
    ctx=shared_ctx if isinstance(shared_ctx,dict) else {}
    garment=np.asarray(ctx.get('garment',np.asarray(garment_u8,dtype=np.uint8)>0),dtype=bool)
    if frame.ndim!=3 or frame.shape[:2]!=branch.shape:
        return {'fold_score':0.0,'valid_samples':0,'reason':'RGB_SHAPE_MISMATCH'}
    L=np.asarray(ctx.get('L',np.empty((0,0),dtype=np.float32)),dtype=np.float32)
    inside=np.asarray(ctx.get('inside',np.empty((0,0),dtype=np.float32)),dtype=np.float32)
    if L.shape!=branch.shape:
        L=cv2.cvtColor(frame,cv2.COLOR_BGR2LAB)[:,:,0].astype(np.float32)
    if inside.shape!=branch.shape:
        inside=cv2.distanceTransform((garment.astype(np.uint8)*255),cv2.DIST_L2,5).astype(np.float32)
    yy,xx=np.where(branch)
    pts=np.column_stack([xx,yy]).astype(np.float64)
    if len(pts)<8:
        return {'fold_score':0.0,'valid_samples':0,'reason':'BRANCH_TOO_SHORT'}
    ctr=pts.mean(axis=0)
    X=pts-ctr[None,:]
    cov=X.T@X
    vals,vecs=np.linalg.eigh(cov)
    tangent_global=vecs[:,int(np.argmax(vals))]
    tangent_global/=max(1e-9,float(np.linalg.norm(tangent_global)))
    proj=X@tangent_global
    span=float(np.max(proj)-np.min(proj))
    linearity=float(np.clip(1.0-float(vals[0])/max(1e-6,float(vals[-1])),0.0,1.0))
    order=np.argsort(proj); ordered=pts[order]
    sample_count=int(np.clip(round(span/18.0)+3,7,17))
    sample_indices=np.unique(np.rint(np.linspace(0,len(ordered)-1,sample_count)).astype(np.int32))
    rgb_deltas=[]; rgb_signs=[]; sample_interiors=[]
    offsets=(10.0,16.0,22.0)
    for si in sample_indices.tolist():
        p=np.asarray(ordered[int(si)],dtype=np.float64)
        d2=np.sum((pts-p[None,:])**2,axis=1)
        neigh=pts[np.argsort(d2)[:min(19,len(pts))]]
        if len(neigh)>=5:
            c=neigh.mean(axis=0); Y=neigh-c[None,:]
            vv,ee=np.linalg.eigh(Y.T@Y)
            tangent=ee[:,int(np.argmax(vv))]
            tangent/=max(1e-9,float(np.linalg.norm(tangent)))
        else:
            tangent=tangent_global.copy()
        normal=np.asarray([-tangent[1],tangent[0]],dtype=np.float64)
        per_rgb=[]; per_sign=[]
        for off in offsets:
            pa=p+normal*off; pb=p-normal*off
            ax,ay=int(round(pa[0])),int(round(pa[1])); bx,by=int(round(pb[0])),int(round(pb[1]))
            if not (3<=ax<branch.shape[1]-3 and 3<=ay<branch.shape[0]-3 and 3<=bx<branch.shape[1]-3 and 3<=by<branch.shape[0]-3):
                continue
            if inside[ay,ax]<3.0 or inside[by,bx]<3.0:
                continue
            la=_e122_patch_median(L,garment,pa[0],pa[1],3)
            lb=_e122_patch_median(L,garment,pb[0],pb[1],3)
            if la is None or lb is None:
                continue
            per_rgb.append(abs(float(la-lb)))
            per_sign.append(float(la-lb))
        if per_rgb:
            rgb_deltas.append(float(np.median(per_rgb)))
            rgb_signs.append(float(np.median(per_sign)))
            xi=int(np.clip(round(p[0]),0,branch.shape[1]-1)); yi=int(np.clip(round(p[1]),0,branch.shape[0]-1))
            sample_interiors.append(float(inside[yi,xi]))

    nvalid=len(rgb_deltas)
    valid_ratio=float(nvalid)/float(max(1,len(sample_indices)))
    rgb_step=float(np.median(rgb_deltas)) if rgb_deltas else 0.0
    strong_sign=np.asarray([v for v in rgb_signs if abs(float(v))>=2.0],dtype=np.float64)
    if strong_sign.size>=3:
        pos=float(np.mean(strong_sign>0)); neg=float(np.mean(strong_sign<0)); side_consistency=max(pos,neg)
    else:
        side_consistency=0.50
    side_score=float(np.clip((side_consistency-0.50)/0.35,0.0,1.0))

    raw_score=np.asarray(raw_score_map,dtype=np.float32)
    rv=raw_score[branch & np.isfinite(raw_score)]
    raw_strength=float(np.mean(rv)) if rv.size else 0.0
    struct_near=np.asarray(ctx.get('struct_near',np.empty((0,0),dtype=bool)),dtype=bool)
    if struct_near.shape==branch.shape:
        structural_support=float(np.mean(struct_near[branch])) if np.any(struct_near) else 0.0
    else:
        struct=np.asarray(structural_u8,dtype=np.uint8)>0
        structural_support=float(np.mean((cv2.dilate((struct.astype(np.uint8)*255),_e149_cv_kernel('ones',5),iterations=1)>0)[branch])) if np.any(struct) else 0.0
    deform_near=np.asarray(ctx.get('deform_near',np.empty((0,0),dtype=bool)),dtype=bool)
    if deform_near.shape==branch.shape:
        deform_support=float(np.mean(deform_near[branch])) if np.any(deform_near) else 0.0
    else:
        deform=np.asarray(pose_deform_u8,dtype=np.uint8)>0
        deform_support=float(np.mean((cv2.dilate((deform.astype(np.uint8)*255),_e149_cv_kernel('ellipse',25),iterations=1)>0)[branch])) if np.any(deform) else 0.0
    seam=np.asarray(ctx.get('seam',np.asarray(seam_u8,dtype=np.uint8)>0),dtype=bool)
    seam_overlap=float(np.mean(seam[branch])) if np.any(seam) else 0.0
    interior_med=float(np.median(sample_interiors)) if sample_interiors else 0.0

    rgb_score=float(np.tanh(max(0.0,rgb_step)/14.0))
    appearance=float(np.clip(0.62*rgb_score+0.38*raw_strength,0.0,1.0))
    length_score=float(np.clip(span/150.0,0.0,1.0))
    interior_score=float(np.clip(interior_med/18.0,0.0,1.0))
    structure=float(np.clip(0.38*length_score+0.28*linearity+0.20*interior_score+0.14*structural_support,0.0,1.0))
    pose_term=float(np.clip(0.30+0.70*deform_support*float(np.clip(pose_deform_conf,0.0,1.0)),0.0,1.0))
    noise_penalty=float(0.08*(1.0-valid_ratio))
    seam_penalty=float(0.42*seam_overlap)
    fold_score=float(np.clip(
        0.36*appearance+0.36*structure+0.18*pose_term+0.10*side_score
        -seam_penalty-noise_penalty,0.0,1.0
    ))
    return {
        'fold_score':fold_score,'geometry_score':structure,'appearance_score':appearance,'structure_score':structure,
        'pose_score':pose_term,'depth_step_mm':0.0,'depth_step_score':0.0,
        'surface_normal_jump':0.0,'surface_normal_score':0.0,'rgb_bilateral_step':rgb_step,
        'rgb_bilateral_score':rgb_score,'side_consistency':float(side_consistency),'side_consistency_score':float(side_score),
        'unidepth_fold_support':0.0,'raw_crease_strength':raw_strength,
        'structural_support':structural_support,'pose_deform_support':deform_support,'seam_overlap':seam_overlap,
        'seam_penalty':seam_penalty,'noise_penalty':noise_penalty,'valid_sample_ratio':valid_ratio,'valid_samples':int(nvalid),
        'span_px':span,'linearity':linearity,'interior_median_px':interior_med,
        'centroid_px':[float(ctr[0]),float(ctr[1])],
        'rgb_execution_support':float(np.clip(0.50*rgb_score+0.30*raw_strength+0.20*side_score,0.0,1.0)),
    }


def _e153_e122_cache_signature(depth_state, mask_u8, args, H_value):
    """Cheap per-I signature. Cache is stored on the fresh depth_state itself.

    The state object is rebuilt for each explicit I, so object identities + the
    immutable planning thresholds are enough to prevent cross-scene reuse.
    """
    mask=np.asarray(mask_u8,dtype=np.uint8)
    try:
        H=np.asarray(H_value,dtype=np.float64)
        hsig=tuple(np.round(H.reshape(-1),9).tolist()) if H.shape==(3,3) else ()
    except Exception:
        hsig=()
    return (
        id(depth_state), id(depth_state.get('frame')), id(depth_state.get('garment_mask')),
        tuple(mask.shape), int(cv2.countNonZero(mask)),
        float(getattr(args,'micro_unfold_e122_preview_score',0.34)),
        float(getattr(args,'micro_unfold_e122_exec_score',0.48)),
        float(getattr(args,'micro_unfold_e123_structural_rescue_score',0.34)),
        float(getattr(args,'micro_unfold_e123_structural_rescue_min_span_px',34.0)),
        int(getattr(args,'micro_unfold_e122_branch_min_px',8)),
        bool(getattr(args,'micro_unfold_vla_edge_flap_rescue_enable',True)),
        int(getattr(args,'micro_unfold_vla_exact_crease_rerank_max',6)),
        float(getattr(args,'micro_unfold_vla_pose_local_score_percentile',85.0)),
        hsig,
    )


def _e153_publish_e122_pool(depth_state, pool, mask_u8):
    """Publish immutable E122 products by reference instead of recopying 720p maps.

    Downstream code treats these maps as read-only. The final overlay plan still
    takes its own snapshot where required, preserving visible behaviour.
    """
    mask=np.asarray(mask_u8,dtype=np.uint8)
    z=np.zeros_like(mask)
    depth_state['e122_fold_preview_skeleton']=np.asarray(pool.get('preview_skeleton',z),dtype=np.uint8)
    depth_state['e122_fold_execution_skeleton']=np.asarray(pool.get('execution_skeleton',z),dtype=np.uint8)
    depth_state['e122_fold_rejected_skeleton']=np.asarray(pool.get('rejected_skeleton',z),dtype=np.uint8)
    depth_state['e122_fold_branch_info']=list(pool.get('branch_info',[]))
    depth_state['e144_fold_contour_filtered_skeleton']=np.asarray(pool.get('contour_filtered_raw_skeleton',z),dtype=np.uint8)
    depth_state['e144_fold_contour_veto_skeleton']=np.asarray(pool.get('contour_veto_rejected_skeleton',z),dtype=np.uint8)


def _e122_build_scored_crease_pool(depth_state, mask_u8, args, H_value=None):
    """E153: E149 scoring math unchanged; one immutable pool is computed per I."""
    _e149_pool_t0=time.perf_counter()
    mask=((np.asarray(mask_u8,dtype=np.uint8)>0).astype(np.uint8)*255)
    sensitive=((np.asarray(depth_state.get('fold_crease_skeleton',np.zeros_like(mask)),dtype=np.uint8)>0).astype(np.uint8)*255)
    structural=((np.asarray(depth_state.get('fold_crease_structural_skeleton',np.zeros_like(mask)),dtype=np.uint8)>0).astype(np.uint8)*255)
    frame=depth_state.get('frame')
    raw_score=np.asarray(depth_state.get('fold_crease_score',np.zeros(mask.shape,dtype=np.float32)),dtype=np.float32)
    pose_deform=np.asarray(depth_state.get('pose_deform_mask',np.zeros_like(mask)),dtype=np.uint8)
    pose_conf=float(depth_state.get('pose_deform_confidence',0.0) or 0.0)
    seam=np.asarray(depth_state.get('pose_seam_exclusion_mask',np.zeros_like(mask)),dtype=np.uint8)
    preview_threshold=float(getattr(args,'micro_unfold_e122_preview_score',0.34))
    exec_threshold=float(getattr(args,'micro_unfold_e122_exec_score',0.48))
    rescue_threshold=float(getattr(args,'micro_unfold_e123_structural_rescue_score',0.34))
    rescue_min_span=float(getattr(args,'micro_unfold_e123_structural_rescue_min_span_px',34.0))
    min_pixels=max(6,int(getattr(args,'micro_unfold_e122_branch_min_px',8)))
    # E153 OPT1: diagnostic and physical planner ask for the exact same pool.
    # Build it once on the fresh I-state, then return the immutable object.
    _e153_sig=_e153_e122_cache_signature(depth_state,mask,args,H_value)
    _e153_cached=depth_state.get('_e153_e122_pool_cache')
    if isinstance(_e153_cached,dict) and _e153_cached.get('signature')==_e153_sig and isinstance(_e153_cached.get('pool'),dict):
        _pool=_e153_cached['pool']
        print(f"[E153-E122-CACHE-HIT] branches={int(_pool.get('raw_branch_count',0))} diagnostic/planner reuse=ON recompute=0")
        return _pool
    print('[E153-E122-CACHE-MISS] fresh I -> computing Fold-branch pool once')
    # E144 is intentionally fixed to the requested physical ~1 cm band.
    contour_veto_mm=10.0
    contour_veto_ratio=0.65
    empty={
        'available':False,'raw_skeleton':np.zeros_like(mask),'preview_skeleton':np.zeros_like(mask),'execution_skeleton':np.zeros_like(mask),
        'rejected_skeleton':np.zeros_like(mask),'branch_score_map':np.zeros(mask.shape,dtype=np.float32),
        'branch_unidepth_map':np.zeros(mask.shape,dtype=np.float32),'branches':[],
        'preview_threshold':preview_threshold,'execution_threshold':exec_threshold,'rescue_threshold':rescue_threshold,
        'structural_branch_count':0,'sensitive_branch_count':0,'rescue_branch_count':0,
        'contour_filtered_raw_skeleton':np.zeros_like(mask),'contour_veto_rejected_skeleton':np.zeros_like(mask),
        'contour_veto_count':0,'contour_veto_mm':contour_veto_mm,'contour_veto_ratio':contour_veto_ratio,
    }
    if frame is None or (cv2.countNonZero(structural)<min_pixels and cv2.countNonZero(sensitive)<min_pixels):
        return empty
    structural_branches=_e122_split_crease_branches(structural,min_pixels=min_pixels) if cv2.countNonZero(structural)>=min_pixels else []
    sensitive_branches=_e122_split_crease_branches(sensitive,min_pixels=min_pixels) if cv2.countNonZero(sensitive)>=min_pixels else []
    structural_union=np.zeros_like(mask)
    for b in structural_branches:
        structural_union[np.asarray(b)>0]=255
    structural_near=(cv2.dilate(structural_union,_e149_cv_kernel('ones',7),iterations=1)>0) if cv2.countNonZero(structural_union)>0 else np.zeros(mask.shape,dtype=bool)
    novel_sensitive=[]
    for b in sensitive_branches:
        bb=np.asarray(b,dtype=np.uint8)>0; area=max(1,int(np.count_nonzero(bb)))
        overlap=float(np.count_nonzero(bb & structural_near))/float(area)
        if overlap<0.62:
            novel_sensitive.append(b)
    sources=[('STRUCTURAL',b) for b in structural_branches]+[('SENSITIVE',b) for b in novel_sensitive]
    if not sources:
        return empty

    shared_ctx=_e149_build_branch_shared_ctx(
        frame,mask,structural,pose_deform,seam,
        base_rgb={'L':depth_state.get('_e149_rgb_l_map'),'inside':depth_state.get('_e149_inside_map')}
    )
    _e149_shared_ms=1000.0*(time.perf_counter()-_e149_pool_t0)

    # Distance to the external silhouette is computed once per I and reused by branches.
    contour_inside=np.asarray(depth_state.get('_e153_contour_inside_map',np.empty((0,0),dtype=np.float32)),dtype=np.float32)
    if contour_inside.shape!=mask.shape:
        contour_inside=_e144_outer_contour_inside_map(mask)
        depth_state['_e153_contour_inside_map']=contour_inside
        print('[E153-CONTOUR-MAP] MISS -> external-contour distance map computed once')
    else:
        print('[E153-CONTOUR-MAP] HIT -> shared with same-I consumers')
    scored=[]; raw_out=np.zeros_like(mask); preview=np.zeros_like(mask); execution=np.zeros_like(mask); rejected=np.zeros_like(mask)
    contour_filtered_raw=np.zeros_like(mask); contour_veto_rejected=np.zeros_like(mask); contour_veto_count=0
    score_map=np.zeros(mask.shape,dtype=np.float32); zero_map=np.zeros(mask.shape,dtype=np.float32); rescue_count=0
    for idx,(source,b) in enumerate(sources,1):
        bb=np.asarray(b,dtype=np.uint8)>0
        contour_stat=_e144_branch_contour_distance_stats(
            b,mask,H_value,threshold_mm=contour_veto_mm,inside_px=contour_inside
        )
        contour_veto=bool(
            contour_stat.get('available',False)
            and float(contour_stat.get('median_mm',999.0))<=contour_veto_mm
            and float(contour_stat.get('near_ratio',0.0))>=contour_veto_ratio
        )
        if contour_veto:
            contour_veto_count+=1
            contour_veto_rejected[bb]=255
            # Keep it diagnostic-only; it contributes zero executable Fold-Crease evidence.
            metrics={
                'fold_score':0.0,'geometry_score':0.0,'appearance_score':0.0,'structure_score':0.0,
                'pose_score':0.0,'depth_step_mm':0.0,'depth_step_score':0.0,
                'surface_normal_jump':0.0,'surface_normal_score':0.0,'rgb_bilateral_step':0.0,
                'rgb_bilateral_score':0.0,'side_consistency':0.5,'side_consistency_score':0.0,
                'unidepth_fold_support':0.0,'raw_crease_strength':0.0,'structural_support':0.0,
                'pose_deform_support':0.0,'seam_overlap':0.0,'seam_penalty':0.0,'noise_penalty':0.0,
                'valid_sample_ratio':0.0,'valid_samples':0,'span_px':0.0,'linearity':0.0,
                'interior_median_px':0.0,
                'centroid_px':[float(np.mean(np.where(bb)[1])) if np.any(bb) else 0.0,
                               float(np.mean(np.where(bb)[0])) if np.any(bb) else 0.0],
                'rgb_execution_support':0.0,
            }
            print(
                f"[E144-CONTOUR-CREASE-VETO] branch={idx} src={source} "
                f"median={float(contour_stat.get('median_mm',999.0)):.1f}mm "
                f"p75={float(contour_stat.get('p75_mm',999.0)):.1f}mm "
                f"within10.0mm={100.0*float(contour_stat.get('near_ratio',0.0)):.0f}% "
                f"gate=65% -> IGNORE"
            )
        else:
            contour_filtered_raw[bb]=255
            # E141 scorer is untouched for all surviving branches.
            metrics=_e122_score_fold_branch(b,frame,zero_map,mask,raw_score,structural,pose_deform,pose_conf,seam,shared_ctx=shared_ctx)

        raw_fold=float(metrics.get('fold_score',0.0)); source_bonus=(0.055 if source=='STRUCTURAL' else 0.0) if not contour_veto else 0.0
        score=float(np.clip(raw_fold+source_bonus,0.0,1.0))
        span=float(metrics.get('span_px',0.0)); valid_ratio=float(metrics.get('valid_sample_ratio',0.0))
        seam_overlap=float(metrics.get('seam_overlap',0.0)); interior=float(metrics.get('interior_median_px',0.0))
        rgb_step=float(metrics.get('rgb_bilateral_step',0.0)); raw_strength=float(metrics.get('raw_crease_strength',0.0))
        deform_support=float(metrics.get('pose_deform_support',0.0))
        rgb_exec_ok=bool(rgb_step>=3.0 or raw_strength>=0.28 or deform_support*pose_conf>=0.16)
        structural_rescue=bool(
            (not contour_veto) and source=='STRUCTURAL' and score>=rescue_threshold and span>=rescue_min_span and
            valid_ratio>=0.25 and seam_overlap<0.18 and interior>=3.0 and rgb_exec_ok
        )
        preview_ok=bool((score>=preview_threshold or structural_rescue) and not contour_veto)
        execution_ok=bool(((score>=exec_threshold and rgb_exec_ok) or structural_rescue) and not contour_veto)
        if structural_rescue and score<exec_threshold:
            rescue_count+=1
        item={
            'branch_id':int(idx),'source':source,'fold_score_raw':raw_fold,'fold_score':score,
            **{k:v for k,v in metrics.items() if k!='fold_score'},
            'source_bonus':float(source_bonus),'structural_rescue':bool(structural_rescue),
            'depth_execution_ok':True,'rgb_execution_ok':bool(rgb_exec_ok),'unidepth_fold_support':0.0,
            'contour_distance_median_mm':float(contour_stat.get('median_mm',999.0)),
            'contour_distance_p75_mm':float(contour_stat.get('p75_mm',999.0)),
            'contour_within_veto_ratio':float(contour_stat.get('near_ratio',0.0)),
            'contour_near_veto':bool(contour_veto),
            'preview_ok':preview_ok,'execution_ok':execution_ok,'mask':np.asarray(b,dtype=np.uint8).copy(),
        }
        scored.append(item)
        raw_out[bb]=255; score_map[bb]=np.maximum(score_map[bb],score)
        if preview_ok: preview[bb]=255
        else: rejected[bb]=255
        if execution_ok: execution[bb]=255
    scored.sort(key=lambda q:(bool(q.get('execution_ok',False)),float(q.get('fold_score',0.0)),float(q.get('span_px',0.0))),reverse=True)
    info=[{k:v for k,v in q.items() if k!='mask'} for q in scored[:24]]
    print(
        f"[E144-CONTOUR-VETO-SUMMARY] rejectedBranches={contour_veto_count}/{len(sources)} "
        f"threshold=10.0mm ratioGate=0.65"
    )
    _e149_pool_ms=1000.0*(time.perf_counter()-_e149_pool_t0)
    print(f"[E149-TIME-BRANCH] branches={len(sources)} sharedPrep={_e149_shared_ms:.1f}ms total={_e149_pool_ms:.1f}ms invariantMaps=ON")
    _e153_result={
        'available':bool(cv2.countNonZero(preview)>=8),'raw_skeleton':raw_out,'preview_skeleton':preview,'execution_skeleton':execution,
        'rejected_skeleton':rejected,'branch_score_map':score_map,'branch_unidepth_map':zero_map,
        'branches':scored,'branch_info':info,'preview_threshold':preview_threshold,'execution_threshold':exec_threshold,
        'rescue_threshold':rescue_threshold,'raw_branch_count':int(len(scored)),
        'preview_branch_count':int(sum(bool(q['preview_ok']) for q in scored)),
        'execution_branch_count':int(sum(bool(q['execution_ok']) for q in scored)),
        'structural_branch_count':int(len(structural_branches)),'sensitive_branch_count':int(len(novel_sensitive)),
        'rescue_branch_count':int(rescue_count),'e137_rgb_only':True,
        'contour_filtered_raw_skeleton':contour_filtered_raw,
        'contour_veto_rejected_skeleton':contour_veto_rejected,
        'contour_veto_count':int(contour_veto_count),'contour_veto_mm':contour_veto_mm,
        'contour_veto_ratio':contour_veto_ratio,'e144_contour_veto':True,
    }
    depth_state['_e153_e122_pool_cache']={'signature':_e153_sig,'pool':_e153_result}
    print(f"[E153-E122-CACHE-STORE] branches={len(scored)} poolMs={_e149_pool_ms:.1f} next same-I call will be O(1)")
    return _e153_result


def _e125_build_primary_fold_region(depth_state, mask_u8, args):
    """E137/V6 primary region: validated VLA joint ROI, else legacy Pose intrusion."""
    mask=((np.asarray(mask_u8,dtype=np.uint8)>0).astype(np.uint8)*255)
    empty=np.zeros_like(mask); garment_area=float(max(1,cv2.countNonZero(mask)))
    # V6 strict mode: only the Pose ROI already corroborated by an E122 RGB
    # execution branch may become authoritative.  This prevents a raw NECK or
    # unrelated Pose intrusion from reclaiming control later in E125/E146.
    if bool(depth_state.get('vla_joint_fold_region_valid',False)):
        joint=((np.asarray(depth_state.get('vla_joint_fold_region_mask',empty),dtype=np.uint8)>0).astype(np.uint8)*255)
        joint=cv2.bitwise_and(joint,mask)
        ratio=float(cv2.countNonZero(joint))/garment_area if cv2.countNonZero(joint) else 0.0
        if cv2.countNonZero(joint)>=24 and 0.008<=ratio<=0.58:
            dil=cv2.dilate(joint,np.ones((3,3),np.uint8),iterations=1)
            ero=cv2.erode(joint,np.ones((3,3),np.uint8),iterations=1)
            boundary=cv2.subtract(dil,ero)
            ginside=cv2.distanceTransform(mask,cv2.DIST_L2,5).astype(np.float32)
            boundary=(((boundary>0)&(mask>0)&(ginside>=4.0)).astype(np.uint8)*255)
            return {
                'available':True,'authoritative':True,'source':'VLA_JOINT_RGB_POSE',
                'confidence':float(depth_state.get('vla_joint_pose_confidence',0.0) or 0.0),
                'group':str(depth_state.get('vla_joint_pose_group','NONE')),
                'region_mask':joint,'boundary_mask':boundary,'area_ratio':float(ratio),
                'depth_diagnostics':{},'e137_rgb_only':True,'vla_joint_validated':True,
            }
    pose_region=((np.asarray(depth_state.get('pose_fold_region_mask',empty),dtype=np.uint8)>0).astype(np.uint8)*255)
    pose_conf=float(depth_state.get('pose_fold_region_confidence',depth_state.get('pose_fold_intrusion_confidence',0.0)) or 0.0)
    pose_group=str(depth_state.get('pose_fold_primary_group','NONE'))
    pose_active_groups=list(depth_state.get('pose_fold_active_groups',[]) or [])
    pose_min=float(getattr(args,'micro_unfold_e125_pose_region_min_conf',0.34))
    pose_ratio=float(cv2.countNonZero(pose_region))/garment_area if cv2.countNonZero(pose_region) else 0.0
    if cv2.countNonZero(pose_region)>=24 and pose_conf>=pose_min and 0.008<=pose_ratio<=0.48:
        region=cv2.bitwise_and(pose_region,mask); source='POSE_INTRUSION'; confidence=pose_conf
        group='+'.join(str(g) for g in pose_active_groups) if pose_active_groups else pose_group
        authoritative=True
    else:
        region=empty.copy(); source='RGB_CREASE_GEOMETRY'; confidence=0.0; group='NONE'; authoritative=False
    if cv2.countNonZero(region)>=8:
        dil=cv2.dilate(region,np.ones((3,3),np.uint8),iterations=1); ero=cv2.erode(region,np.ones((3,3),np.uint8),iterations=1)
        boundary=cv2.subtract(dil,ero)
        ginside=cv2.distanceTransform(mask,cv2.DIST_L2,5).astype(np.float32)
        boundary=(((boundary>0)&(mask>0)&(ginside>=4.0)).astype(np.uint8)*255)
    else:
        boundary=empty.copy()
    return {
        'available':bool(authoritative),'authoritative':bool(authoritative),'source':source,
        'confidence':float(confidence),'group':str(group),'region_mask':region,'boundary_mask':boundary,
        'area_ratio':float(cv2.countNonZero(region))/garment_area if cv2.countNonZero(region) else 0.0,
        'depth_diagnostics':{},'e137_rgb_only':True,
    }


def _e129_rank_pose_intrusion_active_pixels(depth_state, mask_u8, fold_region_u8, args=None):
    """E137 Pose-region ACTIVE ranking with no depth refinement."""
    mask=((np.asarray(mask_u8,dtype=np.uint8)>0).astype(np.uint8)*255)
    region=((np.asarray(fold_region_u8,dtype=np.uint8)>0).astype(np.uint8)*255); region=cv2.bitwise_and(region,mask)
    if cv2.countNonZero(region)<16: return []
    garment_inside=cv2.distanceTransform(mask,cv2.DIST_L2,5).astype(np.float32)
    region_inside=cv2.distanceTransform(region,cv2.DIST_L2,5).astype(np.float32)
    region_support=cv2.blur((region>0).astype(np.float32),(15,15))
    finite=(region>0)&(garment_inside>=3.0)
    ys,xs=np.nonzero(finite)
    if ys.size==0: return []
    pose_support=np.clip(region_support[ys,xs],0.0,1.0)
    garment_support=np.clip(garment_inside[ys,xs]/22.0,0.0,1.0)
    region_clear=np.clip(region_inside[ys,xs]/12.0,0.0,1.0)
    base=0.56*pose_support+0.28*garment_support+0.16*region_clear
    order=np.argsort(-base); max_keep=int(getattr(args,'micro_unfold_e129_pose_active_max_candidates',90)) if args is not None else 90
    min_gap=float(getattr(args,'micro_unfold_e129_pose_active_nms_px',8.0)) if args is not None else 8.0
    picked=[]
    for oi in order.tolist():
        p=np.asarray([float(xs[oi]),float(ys[oi])],dtype=np.float64)
        if any(float(np.linalg.norm(p-q['px']))<min_gap for q in picked): continue
        picked.append({'px':p,'score':float(base[oi]),'pose_region_support':float(pose_support[oi]),
                       'raised_support':0.0,'raised_mm':0.0,'garment_interior_support':float(garment_support[oi]),
                       'region_interior_support':float(region_clear[oi])})
        if len(picked)>=max_keep: break
    return picked


def _e137_collect_rgb_base_axes(state, mask_u8):
    """Collect direction axes from Pose deformation and RGB crease geometry."""
    mask=((np.asarray(mask_u8,dtype=np.uint8)>0).astype(np.uint8)*255)
    if cv2.countNonZero(mask)<50: return []
    mm=cv2.moments(mask,binaryImage=True)
    gc=np.asarray([float(mm['m10']/max(1e-6,mm['m00'])),float(mm['m01']/max(1e-6,mm['m00']))],dtype=np.float64)
    out=[]
    def add_vec(name,u,priority,proj=120.0):
        if u is None: return
        u=np.asarray(u,dtype=np.float64).reshape(2); nu=float(np.linalg.norm(u))
        if not np.all(np.isfinite(u)) or nu<1e-6: return
        u=u/nu
        for q in out:
            if float(np.dot(u,q['u']))>0.985: return
        out.append({'name':str(name),'u':u,'proj':float(proj),'priority':float(priority)})
    def add_point(name,p,priority):
        if p is None: return
        p=np.asarray(p,dtype=np.float64).reshape(2); d=p-gc; nd=float(np.linalg.norm(d))
        if nd<12.0 or not np.all(np.isfinite(d)): return
        add_vec(name,d/nd,priority,nd)

    # Strong pose-deformation direction is useful even without an admitted contour intrusion.
    pd=state.get('pose_deform_fold_direction_px',None)
    pc=float(state.get('pose_deform_fold_region_confidence',state.get('pose_deform_confidence',0.0)) or 0.0)
    if pd is not None and pc>=0.20:
        try: add_vec('POSE_DEFORM',pd,0.96,120.0)
        except Exception: pass

    crease=np.asarray(state.get('e122_fold_execution_skeleton',np.zeros_like(mask)),dtype=np.uint8)
    if cv2.countNonZero(crease)<8:
        crease=np.asarray(state.get('e122_fold_preview_skeleton',np.zeros_like(mask)),dtype=np.uint8)
    if cv2.countNonZero(crease)<8:
        # E144: raw multi-axis fallback is allowed only after Cloth-Contour veto.
        crease=np.asarray(state.get('e144_fold_contour_filtered_skeleton',np.zeros_like(mask)),dtype=np.uint8)
    if cv2.countNonZero(crease)>=8:
        nc,ll,ss,cent=cv2.connectedComponentsWithStats((crease>0).astype(np.uint8),8)
        comps=[]
        for lid in range(1,nc):
            area=int(ss[lid,cv2.CC_STAT_AREA])
            if area<6: continue
            yy,xx=np.where(ll==lid); pts=np.column_stack([xx,yy]).astype(np.float64)
            if len(pts)<6: continue
            ctr=pts.mean(axis=0); X=pts-ctr[None,:]; vv,ee=np.linalg.eigh(X.T@X)
            tangent=ee[:,int(np.argmax(vv))]; tangent/=max(1e-9,float(np.linalg.norm(tangent)))
            normal=np.asarray([-tangent[1],tangent[0]],dtype=np.float64)
            radial=ctr-gc
            if float(np.dot(normal,radial))<0: normal=-normal
            span=float(max(ss[lid,cv2.CC_STAT_WIDTH],ss[lid,cv2.CC_STAT_HEIGHT]))
            comps.append((span,ctr,normal))
        comps.sort(reverse=True,key=lambda z:z[0])
        for k,(span,ctr,normal) in enumerate(comps[:4]):
            add_point(f'CREASE_RADIAL_{k+1}',ctr,0.88-0.04*k)
            add_vec(f'CREASE_NORMAL_OUT_{k+1}',normal,0.84-0.04*k,max(60.0,float(np.linalg.norm(ctr-gc))))
            add_vec(f'CREASE_NORMAL_IN_{k+1}',-normal,0.70-0.04*k,max(60.0,float(np.linalg.norm(ctr-gc))))

    cs,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    if cs:
        pts=max(cs,key=cv2.contourArea).reshape(-1,2).astype(np.float64)
        d=np.linalg.norm(pts-gc[None,:],axis=1)
        if d.size:
            for j in np.argsort(-d)[:8]:
                add_point('CONTOUR_EXTREME',pts[int(j)],0.45)
    out.sort(key=lambda q:q['priority'],reverse=True)
    return out


def _e146_lower_pose_fold_normal(depth_state, mask_u8, exec_crease, preview_crease):
    """Use LOWER_LEFT/LOWER_RIGHT pose asymmetry to select the matching Fold Boundary.

    Lower hem landmarks (ID7=waist_right, ID8=waist_left) naturally remain near the
    current Cloth Contour even when the hem is folded.  Therefore this path does NOT
    require contour intrusion.  Instead it uses the E120 TORSO_QUAD lower-region
    asymmetry as a selector and the corresponding waist landmark as a FLAP-side
    anchor.  The actual manipulation axis is still the exact normal of ONE surviving
    Fold-Crease component; no angular search is permitted.
    """
    mask=((np.asarray(mask_u8,dtype=np.uint8)>0).astype(np.uint8)*255)
    if cv2.countNonZero(mask)<500:
        return None

    region=str(depth_state.get('pose_deform_fold_region_hint','NONE'))
    region_conf=float(depth_state.get('pose_deform_fold_region_confidence',0.0) or 0.0)
    if region not in ('LOWER_LEFT','LOWER_RIGHT') or region_conf < 0.18:
        return None

    anchor_raw=depth_state.get('pose_deform_fold_anchor_px',None)
    pose_state=dict(depth_state.get('tshirt_pose_state',{}) or {})
    named=dict(pose_state.get('named_points',{}) or {})
    semantic='waist_left' if region=='LOWER_LEFT' else 'waist_right'
    anchor=None
    try:
        q=dict(named.get(semantic,{}) or {})
        x=float(q.get('x')); y=float(q.get('y')); c=float(q.get('conf',0.0))
        if np.isfinite(x) and np.isfinite(y) and np.isfinite(c) and c>=0.20:
            anchor=np.asarray([x,y],dtype=np.float64)
    except Exception:
        anchor=None
    if anchor is None and anchor_raw is not None:
        try:
            a=np.asarray(anchor_raw,dtype=np.float64).reshape(2)
            if np.all(np.isfinite(a)):
                anchor=a
        except Exception:
            anchor=None
    if anchor is None:
        print(f'[E146-LOWER-FOLD-BLOCK] region={region}:{region_conf:.2f} but ID7/8 flap anchor unavailable')
        return None

    mm=cv2.moments(mask,binaryImage=True)
    gc=np.asarray([float(mm['m10']/max(1e-6,mm['m00'])),float(mm['m01']/max(1e-6,mm['m00']))],dtype=np.float64)
    radial=anchor-gc
    radial_len=float(np.linalg.norm(radial))
    if radial_len<30.0:
        return None
    radial/=radial_len

    score_map=np.asarray(depth_state.get('fold_crease_score',np.zeros(mask.shape,dtype=np.float32)),dtype=np.float32)
    inside=cv2.distanceTransform(mask,cv2.DIST_L2,5).astype(np.float32)
    pose_prior=((np.asarray(depth_state.get('pose_deform_mask',np.zeros_like(mask)),dtype=np.uint8)>0).astype(np.uint8)*255)
    if pose_prior.shape!=mask.shape:
        pose_prior=cv2.resize(pose_prior,(mask.shape[1],mask.shape[0]),interpolation=cv2.INTER_NEAREST)
    if cv2.countNonZero(pose_prior)>0:
        pose_prior=cv2.dilate(pose_prior,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(31,31)),iterations=1)
        pose_prior=cv2.bitwise_and(pose_prior,mask)

    pools=[]
    ex=np.asarray(exec_crease,dtype=np.uint8)
    pv=np.asarray(preview_crease,dtype=np.uint8)
    raw=np.asarray(depth_state.get('e144_fold_contour_filtered_skeleton',np.zeros_like(mask)),dtype=np.uint8)
    if cv2.countNonZero(ex)>=8:
        pools.append(('EXECUTION',ex,1.00))
    if cv2.countNonZero(pv)>=8:
        pools.append(('PREVIEW',pv,0.84))
    if cv2.countNonZero(raw)>=8:
        pools.append(('E144_FILTERED_RAW',raw,0.66))
    if not pools:
        print(f'[E146-LOWER-FOLD-BLOCK] region={region}:{region_conf:.2f} no contour-veto-surviving Fold Boundary')
        return None

    h,w=mask.shape[:2]
    best=None
    seen_masks=[]
    for source,crease,source_weight in pools:
        crease=((np.asarray(crease,dtype=np.uint8)>0).astype(np.uint8)*255)
        nc,ll,ss,_=cv2.connectedComponentsWithStats((crease>0).astype(np.uint8),8)
        for lid in range(1,nc):
            yy,xx=np.where(ll==lid)
            if xx.size<8:
                continue
            comp=((ll==lid).astype(np.uint8)*255)
            # Avoid evaluating the same geometry again when PREVIEW includes EXECUTION.
            duplicate=False
            for old in seen_masks:
                inter=cv2.countNonZero(cv2.bitwise_and(comp,old))
                if inter/max(1,cv2.countNonZero(comp))>0.80:
                    duplicate=True; break
            if duplicate:
                continue
            seen_masks.append(comp)

            pts=np.column_stack([xx,yy]).astype(np.float64)
            ctr=pts.mean(axis=0)
            X=pts-ctr[None,:]
            vals,vecs=np.linalg.eigh(X.T@X)
            tangent=np.asarray(vecs[:,int(np.argmax(vals))],dtype=np.float64)
            nt=float(np.linalg.norm(tangent))
            if nt<1e-9:
                continue
            tangent/=nt
            normal=np.asarray([-tangent[1],tangent[0]],dtype=np.float64)

            # The lower waist landmark is a FLAP-side anchor.  It may sit on or a
            # few pixels outside the segmentation contour; that is explicitly OK.
            to_anchor=anchor-ctr
            na=float(np.linalg.norm(to_anchor))
            if na<8.0:
                continue
            if float(np.dot(normal,to_anchor))<0.0:
                normal=-normal
            normal_anchor_cos=float(np.dot(normal,to_anchor/na))

            # The Fold Boundary must lie between BODY centre and the lower landmark,
            # and reasonably close to that anatomical ray/corridor.
            rel=ctr-gc
            proj=float(np.dot(rel,radial))
            lateral=float(abs(rel[0]*(-radial[1])+rel[1]*radial[0]))
            if proj < max(10.0,0.12*radial_len) or proj > radial_len+24.0:
                continue
            lateral_limit=max(58.0,0.32*radial_len)
            if lateral>lateral_limit:
                continue

            # Require actual cloth on both sides of the exact crease normal.  The
            # outward side is the ID7/8 side, not the garment-centroid heuristic.
            outward_px=0.0; last_inside=None
            for d in np.arange(2.0,260.0,2.0):
                q=ctr+normal*float(d); xi=int(round(q[0])); yi=int(round(q[1]))
                if not (0<=xi<w and 0<=yi<h) or mask[yi,xi]<=0:
                    break
                outward_px=float(d); last_inside=q.copy()
            inward_px=0.0
            for d in np.arange(2.0,260.0,2.0):
                q=ctr-normal*float(d); xi=int(round(q[0])); yi=int(round(q[1]))
                if not (0<=xi<w and 0<=yi<h) or mask[yi,xi]<=0:
                    break
                inward_px=float(d)
            if outward_px<14.0 or inward_px<24.0 or last_inside is None:
                continue

            sv=score_map[ll==lid] if score_map.shape==mask.shape else np.asarray([])
            smean=float(np.mean(sv[np.isfinite(sv)])) if sv.size and np.any(np.isfinite(sv)) else 0.35
            span=float(max(ss[lid,cv2.CC_STAT_WIDTH],ss[lid,cv2.CC_STAT_HEIGHT]))
            iv=inside[ll==lid]
            interior=float(np.median(iv[np.isfinite(iv)])) if iv.size and np.any(np.isfinite(iv)) else 0.0
            anchor_dist=float(np.min(np.linalg.norm(pts-anchor[None,:],axis=1)))
            pose_overlap=0.0
            if cv2.countNonZero(pose_prior)>0:
                pose_overlap=float(np.count_nonzero((comp>0)&(pose_prior>0)))/max(1.0,float(cv2.countNonZero(comp)))

            # LOWER selector: Pose anatomical agreement outranks branch length.
            ray_score=float(np.exp(-lateral/max(28.0,0.20*radial_len)))
            anchor_score=float(np.exp(-anchor_dist/125.0))
            normal_score=float(np.clip((normal_anchor_cos-0.10)/0.80,0.0,1.0))
            pose_score=float(np.clip(0.58*pose_overlap+0.42*anchor_score,0.0,1.0))
            quality=(
                0.30*np.clip(smean,0.0,1.0)+
                0.24*pose_score+
                0.16*ray_score+
                0.12*normal_score+
                0.08*np.clip(span/120.0,0.0,1.0)+
                0.05*np.clip(interior/22.0,0.0,1.0)+
                0.05*np.clip(outward_px/90.0,0.0,1.0)
            )*float(source_weight)*(0.82+0.18*np.clip(region_conf/0.45,0.0,1.0))

            # A lower Fold Boundary should have at least one direct anatomical link:
            # corridor overlap OR close distance to ID7/8.  This prevents an unrelated
            # central wrinkle from winning solely because it is long.
            if pose_overlap<0.03 and anchor_dist>155.0:
                continue
            if best is None or quality>best[0]:
                best=(float(quality),source,int(lid),ctr,tangent,normal,outward_px,inward_px,
                      smean,span,interior,comp,pose_overlap,anchor_dist,lateral,normal_anchor_cos)

    if best is None:
        print(
            f'[E146-LOWER-FOLD-BLOCK] region={region}:{region_conf:.2f} anchor={semantic} '
            'Fold Boundary exists but none matches the lower Pose corridor/ID7-8 geometry'
        )
        return None

    (quality,source,lid,ctr,tangent,normal,outward_px,inward_px,smean,span,interior,
     comp,pose_overlap,anchor_dist,lateral,normal_anchor_cos)=best
    evidence_dist=float(np.clip(0.72*outward_px,16.0,max(16.0,outward_px-4.0)))
    evidence=ctr+normal*evidence_dist
    reg=np.zeros_like(mask,dtype=np.uint8)
    cv2.circle(reg,tuple(np.rint(evidence).astype(int)),8,255,-1,cv2.LINE_AA)
    reg=cv2.bitwise_and(reg,mask)
    if cv2.countNonZero(reg)<8:
        cv2.circle(reg,tuple(np.rint(evidence).astype(int)),14,255,-1,cv2.LINE_AA)
        reg=cv2.bitwise_and(reg,mask)
    if cv2.countNonZero(reg)<8:
        return None

    print(
        f'[E146-LOWER-FOLD-SELECT] region={region}:{region_conf:.2f} flapAnchor={semantic} '
        f'source={source} component={lid} quality={quality:.3f} foldScore={smean:.3f} '
        f'poseOverlap={pose_overlap:.2f} anchorDist={anchor_dist:.1f}px lateral={lateral:.1f}px '
        f'normalToAnchorCos={normal_anchor_cos:.2f} normal=({normal[0]:+.4f},{normal[1]:+.4f}) '
        f'outwardRoom={outward_px:.1f}px inwardRoom={inward_px:.1f}px rotationSearch=OFF'
    )
    return {
        'source':source,'component_id':int(lid),'quality':float(quality),
        'anchor_px':np.asarray(ctr,dtype=np.float64),'tangent_px':np.asarray(tangent,dtype=np.float64),
        'normal_px':np.asarray(normal,dtype=np.float64),'evidence_px':np.asarray(evidence,dtype=np.float64),
        'region_mask':reg,'crease_mask':comp,'outward_room_px':float(outward_px),
        'inward_room_px':float(inward_px),'fold_score':float(smean),'span_px':float(span),
        'lower_pose_region':str(region),'lower_pose_confidence':float(region_conf),
        'lower_pose_anchor_name':str(semantic),'lower_pose_anchor_px':np.asarray(anchor,dtype=np.float64),
        'pose_overlap':float(pose_overlap),'anchor_distance_px':float(anchor_dist),
    }




def _e154_direct_lower_corner_rescue(depth_state, mask_u8, exec_crease, preview_crease):
    """Direct ID7/ID8 lower-corner Fold-Boundary rescue without TORSO_QUAD authority.

    This is deliberately weaker than a real LOWER_LEFT/LOWER_RIGHT TORSO_QUAD
    decision.  It is used only when that lower-region gate is absent.  ID7
    (waist_right) and ID8 (waist_left) are treated as FLAP-side anchors because
    they normally remain on/near Cloth Contour even when the lower corner is
    folded.  A candidate must still be a real E144-veto-surviving Fold Boundary
    located between BODY centre and the corresponding corner anchor.  Motion is
    attempted only on the component's exact PCA normal; no angle search is added.
    """
    mask=((np.asarray(mask_u8,dtype=np.uint8)>0).astype(np.uint8)*255)
    if cv2.countNonZero(mask)<500:
        return []

    pose_state=dict(depth_state.get('tshirt_pose_state',{}) or {})
    named=dict(pose_state.get('named_points',{}) or {})
    inside=cv2.distanceTransform(mask,cv2.DIST_L2,5).astype(np.float32)
    zeros=np.zeros_like(mask,dtype=np.uint8)
    h,w=mask.shape[:2]
    out=[]

    # ID7 is waist_right, ID8 is waist_left in this 12-point shirt model.
    for id_label,semantic,region in (
        ('ID7','waist_right','LOWER_RIGHT'),
        ('ID8','waist_left','LOWER_LEFT'),
    ):
        try:
            q=dict(named.get(semantic,{}) or {})
            x=float(q.get('x')); y=float(q.get('y')); conf=float(q.get('conf',0.0))
        except Exception:
            continue
        if not (np.isfinite(x) and np.isfinite(y) and np.isfinite(conf)) or conf<0.20:
            continue
        anchor=np.asarray([x,y],dtype=np.float64)
        xi=int(np.clip(round(x),0,w-1)); yi=int(np.clip(round(y),0,h-1))
        # A true lower-corner anchor is expected on/near Cloth Contour.  If the
        # pose point is outside the current mask, contour distance is treated as 0.
        anchor_edge_px=float(inside[yi,xi]) if mask[yi,xi]>0 else 0.0
        if anchor_edge_px>42.0:
            print(
                f'[E154-LOWER-CORNER-SKIP] {id_label}={semantic} conf={conf:.2f} '
                f'anchorInside={anchor_edge_px:.1f}px > 42px; not a contour-side lower anchor'
            )
            continue

        forced=dict(depth_state)
        forced['pose_deform_fold_region_hint']=region
        # 0.18 is exactly the original E146 lower gate.  This does not create a
        # TORSO_QUAD claim; it only lets the existing anatomical Fold-Boundary
        # selector evaluate this corner using the same geometry.
        forced['pose_deform_fold_region_confidence']=0.18
        forced['pose_deform_fold_anchor_px']=anchor.tolist()
        # Direct rescue must stand on ID7/8 + real crease geometry, not on an
        # unrelated shoulder/sleeve deformation mask from the same frame.
        forced['pose_deform_mask']=zeros
        try:
            cand=_e146_lower_pose_fold_normal(forced,mask,exec_crease,preview_crease)
        except Exception as exc:
            print(f'[E154-LOWER-CORNER-REJECT] {id_label} selectorError={exc!r}')
            cand=None
        if not isinstance(cand,dict):
            continue

        quality=float(cand.get('quality',0.0) or 0.0)
        fold_score=float(cand.get('fold_score',0.0) or 0.0)
        anchor_dist=float(cand.get('anchor_distance_px',999.0) or 999.0)
        span=float(cand.get('span_px',0.0) or 0.0)
        # Conservative direct-rescue gate.  TORSO_QUAD is absent, so require a
        # reasonably strong real Fold Boundary and close anatomical association.
        if quality<0.24 or fold_score<0.28 or anchor_dist>150.0 or span<10.0:
            print(
                f'[E154-LOWER-CORNER-REJECT] {id_label}={semantic} quality={quality:.3f} '
                f'foldScore={fold_score:.3f} anchorDist={anchor_dist:.1f}px span={span:.1f}px'
            )
            continue

        direct_score=float(
            quality
            + 0.07*np.clip(conf,0.0,1.0)
            + 0.06*np.exp(-anchor_dist/90.0)
            + 0.04*np.clip(fold_score,0.0,1.0)
        )
        c=dict(cand)
        c['e154_direct_corner']=True
        c['e154_corner_id']=id_label
        c['e154_corner_semantic']=semantic
        c['e154_anchor_confidence']=float(conf)
        c['e154_anchor_inside_px']=float(anchor_edge_px)
        c['e154_direct_score']=float(direct_score)
        out.append(c)
        print(
            f'[E154-LOWER-CORNER-CAND] {id_label}={semantic} region={region} '
            f'directScore={direct_score:.3f} quality={quality:.3f} foldScore={fold_score:.3f} '
            f'anchorDist={anchor_dist:.1f}px anchorInside={anchor_edge_px:.1f}px '
            f"component={int(cand.get('component_id',-1))}"
        )

    out.sort(key=lambda q:float(q.get('e154_direct_score',0.0)),reverse=True)
    if not out:
        print('[E154-LOWER-CORNER-NONE] ID7/ID8 direct lower rescue found no qualifying BODY-to-corner Fold Boundary')
    return out

def _e150_select_edge_crease_cluster(depth_state, mask_u8, H_value, exec_crease, preview_crease):
    """Detect a LOCAL cluster of multiple Fold-Crease branches near Cloth Contour.

    This is a rescue cue, not a relaxation of E144.  Branches already vetoed by
    E144's ~0-10 mm contour-leakage rule are never revived.  Only surviving
    branches whose robust contour distance lies mainly in the ~10-40 mm interior
    band are clustered.  A cluster needs multiple nearby branches, enough total
    skeleton support, reasonable Fold scores, low seam overlap, and compact
    spatial concentration.  Pose asymmetry is a side-ranking authority when LEFT/RIGHT confidence is >=0.20; it never invents a crease or rotates a normal.

    For a qualified cluster, up to three REAL member Fold Boundaries are ranked.
    Their manipulation axis is the exact PCA normal; the normal sign is chosen
    toward the nearest Cloth Contour (FLAP side).  No angle sweep is generated.
    """
    mask=((np.asarray(mask_u8,dtype=np.uint8)>0).astype(np.uint8)*255)
    if H_value is None or cv2.countNonZero(mask)<500:
        return None

    # IMPORTANT: this skeleton already excludes E144 hard-vetoed contour leakage.
    raw=np.asarray(depth_state.get('e144_fold_contour_filtered_skeleton',np.zeros_like(mask)),dtype=np.uint8)
    ex=np.asarray(exec_crease,dtype=np.uint8)
    pv=np.asarray(preview_crease,dtype=np.uint8)
    if cv2.countNonZero(raw)<8:
        raw=ex if cv2.countNonZero(ex)>=8 else pv
    raw=((raw>0).astype(np.uint8)*255)
    if cv2.countNonZero(raw)<18:
        return None

    _e153_edge_cache=depth_state.get('_e153_edge_split_cache')
    if isinstance(_e153_edge_cache,dict) and _e153_edge_cache.get('raw_id')==id(depth_state.get('e144_fold_contour_filtered_skeleton')):
        branches=_e153_edge_cache.get('branches',[])
        print(f'[E153-EDGE-BRANCH-CACHE] HIT branches={len(branches)}')
    else:
        branches=_e122_split_crease_branches(raw,min_pixels=6)
        depth_state['_e153_edge_split_cache']={'raw_id':id(depth_state.get('e144_fold_contour_filtered_skeleton')),'branches':branches}
        print(f'[E153-EDGE-BRANCH-CACHE] MISS splitOnce branches={len(branches)}')
    if len(branches)<3:
        return None

    seam=((np.asarray(depth_state.get('pose_seam_exclusion_mask',np.zeros_like(mask)),dtype=np.uint8)>0).astype(np.uint8)*255)
    if seam.shape!=mask.shape:
        seam=cv2.resize(seam,(mask.shape[1],mask.shape[0]),interpolation=cv2.INTER_NEAREST)
    seam_near=cv2.dilate(seam,_e149_cv_kernel('ellipse',13),iterations=1)>0 if cv2.countNonZero(seam)>0 else np.zeros(mask.shape,dtype=bool)
    score_map=np.asarray(depth_state.get('fold_crease_score',np.zeros(mask.shape,dtype=np.float32)),dtype=np.float32)
    info=list(depth_state.get('e122_fold_branch_info',[]) or [])
    contour_inside=np.asarray(depth_state.get('_e153_contour_inside_map',np.empty((0,0),dtype=np.float32)),dtype=np.float32)
    if contour_inside.shape!=mask.shape:
        contour_inside=_e144_outer_contour_inside_map(mask)
        depth_state['_e153_contour_inside_map']=contour_inside
        print('[E153-EDGE-CONTOUR-CACHE] MISS')
    else:
        print('[E153-EDGE-CONTOUR-CACHE] HIT')

    mm=cv2.moments(mask,binaryImage=True)
    gc=np.asarray([float(mm['m10']/max(1e-6,mm['m00'])),float(mm['m01']/max(1e-6,mm['m00']))],dtype=np.float64)
    h,w=mask.shape[:2]

    # Natural armhole concavity is a known false-positive zone.  Do not hard-veto
    # it (a real fold can occur there), but apply a penalty unless Pose supports it.
    pose_state=dict(depth_state.get('tshirt_pose_state',{}) or {})
    named=dict(pose_state.get('named_points',{}) or {})
    armpits=[]
    for nm in ('armpit_left','armpit_right'):
        try:
            q=dict(named.get(nm,{}) or {}); x=float(q.get('x')); y=float(q.get('y')); c=float(q.get('conf',0.0))
            if np.isfinite(x) and np.isfinite(y) and c>=0.20:
                armpits.append(np.asarray([x,y],dtype=np.float64))
        except Exception:
            pass

    pside=str(depth_state.get('pose_deform_side_hint','NONE')).upper()
    pregion=str(depth_state.get('pose_deform_fold_region_hint','NONE')).upper()
    pconf=float(depth_state.get('pose_deform_confidence',0.0) or 0.0)
    pregconf=float(depth_state.get('pose_deform_fold_region_confidence',0.0) or 0.0)
    pose_conf=max(pconf,pregconf)

    def _nearest_info_score(ctr, fallback):
        best_d=1e9; best=None
        for q in info:
            try:
                c=np.asarray(q.get('centroid_px'),dtype=np.float64).reshape(2)
                d=float(np.linalg.norm(c-ctr))
                if d<best_d:
                    best_d=d; best=q
            except Exception:
                continue
        if best is not None and best_d<=18.0:
            try: return float(best.get('fold_score',fallback)),str(best.get('source','?'))
            except Exception: pass
        return float(fallback),'RAW'

    items=[]
    for bi,b in enumerate(branches,1):
        bb=np.asarray(b,dtype=np.uint8)>0
        yy,xx=np.where(bb)
        if xx.size<6:
            continue
        st10=_e144_branch_contour_distance_stats(b,mask,H_value,threshold_mm=10.0,inside_px=contour_inside)
        st40=_e144_branch_contour_distance_stats(b,mask,H_value,threshold_mm=40.0,inside_px=contour_inside)
        if not bool(st40.get('available',False)):
            continue
        med=float(st40.get('median_mm',999.0)); near40=float(st40.get('near_ratio',0.0)); near10=float(st10.get('near_ratio',0.0))
        # 0-10 mm is NOT rescued.  The component must live predominantly in the
        # immediately-inboard band and not just touch it at one endpoint.
        if not (10.0 < med <= 42.0 and near40>=0.55):
            continue
        ctr=np.asarray([float(np.mean(xx)),float(np.mean(yy))],dtype=np.float64)
        try:
            bctr=np.asarray(_fix53_px_to_board(H_value,ctr),dtype=np.float64).reshape(2)
        except Exception:
            continue
        sv=score_map[bb] if score_map.shape==mask.shape else np.asarray([])
        raw_score=float(np.mean(sv[np.isfinite(sv)])) if sv.size and np.any(np.isfinite(sv)) else 0.35
        fold_score,source=_nearest_info_score(ctr,raw_score)
        seam_ratio=float(np.mean(seam_near[yy,xx])) if xx.size else 0.0
        if seam_ratio>=0.55:
            continue
        X=np.column_stack([xx,yy]).astype(np.float64)-ctr[None,:]
        vals,vecs=np.linalg.eigh(X.T@X)
        tangent=np.asarray(vecs[:,int(np.argmax(vals))],dtype=np.float64)
        nt=float(np.linalg.norm(tangent))
        if nt<1e-9:
            continue
        tangent/=nt
        span=float(max(float(np.ptp(xx)) if xx.size else 0.0,float(np.ptp(yy)) if yy.size else 0.0))
        items.append({
            'id':int(bi),'mask':(bb.astype(np.uint8)*255),'centroid_px':ctr,'centroid_board':bctr,
            'fold_score':float(fold_score),'source':source,'pixels':int(xx.size),'span_px':span,
            'contour_median_mm':med,'contour_near40':near40,'contour_near10':near10,
            'seam_ratio':seam_ratio,'tangent_px':tangent,
        })
    if len(items)<3:
        print(f'[E150-EDGE-CLUSTER-NONE] inBandBranches={len(items)} need>=3 band=10-40mm; E144 0-10mm veto preserved')
        return None

    # Union-find in calibrated board millimetres.  This prevents a chain of edge
    # creases from different garment corners from being treated as one cluster.
    parent=list(range(len(items)))
    def find(i):
        while parent[i]!=i:
            parent[i]=parent[parent[i]]; i=parent[i]
        return i
    def union(a,b):
        ra,rb=find(a),find(b)
        if ra!=rb: parent[rb]=ra
    link_mm=72.0
    for i in range(len(items)):
        for j in range(i+1,len(items)):
            if float(np.linalg.norm(items[i]['centroid_board']-items[j]['centroid_board']))<=link_mm:
                union(i,j)
    groups={}
    for i in range(len(items)):
        groups.setdefault(find(i),[]).append(items[i])

    clusters=[]
    for members in groups.values():
        if len(members)<3:
            continue
        weights=np.asarray([max(1,m['pixels']) for m in members],dtype=np.float64)
        bpts=np.asarray([m['centroid_board'] for m in members],dtype=np.float64)
        cboard=np.average(bpts,axis=0,weights=weights)
        radius=float(np.max(np.linalg.norm(bpts-cboard[None,:],axis=1))) if len(members)>1 else 0.0
        if radius>92.0:
            continue
        total_px=int(sum(m['pixels'] for m in members))
        if total_px<42:
            continue
        mean_fold=float(np.average([m['fold_score'] for m in members],weights=weights))
        max_fold=float(max(m['fold_score'] for m in members))
        med_cont=float(np.average([m['contour_median_mm'] for m in members],weights=weights))
        seam_ratio=float(np.average([m['seam_ratio'] for m in members],weights=weights))
        if mean_fold<0.34 and max_fold<0.50:
            continue
        cpx=np.average(np.asarray([m['centroid_px'] for m in members],dtype=np.float64),axis=0,weights=weights)
        side='LEFT' if cpx[0]<gc[0] else 'RIGHT'

        # E151: LEFT/RIGHT Pose is no longer a tiny 6% tie-breaker.  Once Pose
        # supplies a meaningful side (>=0.20), it becomes cluster-side ranking
        # authority.  It does NOT create a crease and does NOT alter the exact
        # normal; it only tells us which already-detected edge cluster belongs to
        # the deformed anatomical side.
        pose_side='NONE'
        if pside in ('LEFT','RIGHT'):
            pose_side=pside
        elif pregion.endswith('_LEFT'):
            pose_side='LEFT'
        elif pregion.endswith('_RIGHT'):
            pose_side='RIGHT'
        pose_side_authority=bool(pose_side in ('LEFT','RIGHT') and pose_conf>=0.20)
        pose_match=bool(pose_side_authority and side==pose_side)
        pose_opp=bool(pose_side_authority and side!=pose_side)
        pose_support=1.0 if pose_match else (0.0 if pose_opp else 0.50)

        armpit_dist=min([float(np.linalg.norm(cpx-a)) for a in armpits],default=999.0)
        armpit_penalty=0.12 if (armpit_dist<42.0 and not (pose_match and pose_conf>=0.28)) else 0.0
        count_s=float(np.clip(len(members)/5.0,0.0,1.0))
        length_s=float(np.clip(total_px/150.0,0.0,1.0))
        fold_s=float(np.clip(mean_fold/0.70,0.0,1.0))
        edge_s=float(np.clip(1.0-abs(med_cont-26.0)/22.0,0.0,1.0))
        compact_s=float(np.clip(1.0-radius/92.0,0.0,1.0))

        # Preserve the E150 visual-density score as the base, then apply a much
        # stronger anatomical side term.  At the observed ~0.30 Pose confidence,
        # a same-side cluster receives about +0.13 while the opposite side loses
        # about -0.19.  This is intentionally large enough that a dense wrong-side
        # cluster cannot beat a credible same-side fold cluster merely by branch count.
        base_score=(0.24*count_s+0.20*length_s+0.22*fold_s+0.16*edge_s+0.12*compact_s+0.03
                    -0.22*seam_ratio-armpit_penalty)
        pose_strength=float(np.clip((pose_conf-0.20)/0.40,0.0,1.0)) if pose_side_authority else 0.0
        pose_side_term=(0.12+0.08*pose_strength) if pose_match else ((-0.18-0.05*pose_strength) if pose_opp else 0.0)
        cluster_score=float(base_score+pose_side_term)
        if cluster_score<0.54:
            continue
        clusters.append({
            'score':float(cluster_score),'base_score':float(base_score),'pose_side_term':float(pose_side_term),
            'members':members,'center_px':cpx,'center_board':cboard,
            'radius_mm':radius,'total_pixels':total_px,'mean_fold':mean_fold,'max_fold':max_fold,
            'contour_median_mm':med_cont,'seam_ratio':seam_ratio,'side':side,
            'pose_support':pose_support,'pose_match':bool(pose_match),'pose_opposite':bool(pose_opp),
            'pose_side':str(pose_side),'pose_side_authority':bool(pose_side_authority),'pose_confidence':float(pose_conf),
            'armpit_distance_px':armpit_dist,
        })
    if not clusters:
        print(f'[E151-EDGE-CLUSTER-NONE] inBandBranches={len(items)} no cluster passed density/compactness/seam/pose-side gates')
        return None

    # E151 side authority: if there is at least one qualified cluster on the Pose
    # side, opposite-side clusters do not compete at all.  Only when no qualified
    # same-side cluster exists may the RGB-only/opposite pool be considered.
    pose_side='NONE'
    if pside in ('LEFT','RIGHT'):
        pose_side=pside
    elif pregion.endswith('_LEFT'):
        pose_side='LEFT'
    elif pregion.endswith('_RIGHT'):
        pose_side='RIGHT'
    pose_side_authority=bool(pose_side in ('LEFT','RIGHT') and pose_conf>=0.20)
    selection_pool=clusters
    if pose_side_authority:
        same_side=[c for c in clusters if str(c.get('side'))==pose_side]
        if same_side:
            selection_pool=same_side
            print(f'[E151-EDGE-POSE-SIDE-AUTHORITY] poseSide={pose_side} conf={pose_conf:.3f} sameSideClusters={len(same_side)} totalClusters={len(clusters)} -> opposite side excluded from ranking')
        else:
            print(f'[E151-EDGE-POSE-SIDE-FALLBACK] poseSide={pose_side} conf={pose_conf:.3f} no qualified same-side cluster; opposite/RGB cluster pool allowed')

    for c in sorted(clusters,key=lambda q:q['score'],reverse=True)[:6]:
        print(
            f"[E151-EDGE-CLUSTER-CAND] side={c['side']} score={c['score']:.3f} base={c.get('base_score',0.0):.3f} "
            f"poseTerm={c.get('pose_side_term',0.0):+.3f} poseMatch={bool(c.get('pose_match',False))} "
            f"members={len(c['members'])} contourMed={c['contour_median_mm']:.1f}mm meanFold={c['mean_fold']:.3f}"
        )
    selection_pool.sort(key=lambda c:c['score'],reverse=True)
    cluster=selection_pool[0]

    # Pick REAL Fold Boundaries from the chosen cluster.  Prefer the innermost
    # member (larger contour distance) because it most plausibly separates BODY
    # from the folded edge flap.  Exact normal points to the shorter contour exit.
    candidates=[]
    for m in cluster['members']:
        ctr=np.asarray(m['centroid_px'],dtype=np.float64)
        tangent=np.asarray(m['tangent_px'],dtype=np.float64)
        normal0=np.asarray([-tangent[1],tangent[0]],dtype=np.float64)
        rooms=[]
        for sign in (1.0,-1.0):
            n=normal0*sign; room=0.0; last=None
            for d in np.arange(2.0,260.0,2.0):
                q=ctr+n*float(d); xi=int(round(q[0])); yi=int(round(q[1]))
                if not (0<=xi<w and 0<=yi<h) or mask[yi,xi]<=0:
                    break
                room=float(d); last=q.copy()
            rooms.append((room,n,last))
        rooms.sort(key=lambda z:z[0])
        outward_px,normal,last_inside=rooms[0]; inward_px=rooms[1][0]
        if outward_px<14.0 or inward_px<28.0 or last_inside is None:
            continue
        # Edge-cluster geometry should have a clearly nearer silhouette on FLAP side.
        if outward_px>0.82*inward_px and outward_px>72.0:
            continue
        contour_rank=float(np.clip((m['contour_median_mm']-10.0)/32.0,0.0,1.0))
        room_asym=float(np.clip(1.0-outward_px/max(1.0,inward_px),0.0,1.0))
        rep_score=(0.38*np.clip(m['fold_score']/0.75,0.0,1.0)+0.24*contour_rank+
                   0.16*np.clip(m['span_px']/80.0,0.0,1.0)+0.12*room_asym+
                   0.10*np.clip(1.0-float(np.linalg.norm(m['centroid_board']-cluster['center_board']))/92.0,0.0,1.0))
        evidence_dist=float(np.clip(0.68*outward_px,10.0,max(10.0,outward_px-4.0)))
        evidence=ctr+normal*evidence_dist
        reg=np.zeros_like(mask,dtype=np.uint8)
        cv2.circle(reg,tuple(np.rint(evidence).astype(int)),8,255,-1,cv2.LINE_AA)
        reg=cv2.bitwise_and(reg,mask)
        if cv2.countNonZero(reg)<8:
            cv2.circle(reg,tuple(np.rint(evidence).astype(int)),14,255,-1,cv2.LINE_AA)
            reg=cv2.bitwise_and(reg,mask)
        if cv2.countNonZero(reg)<8:
            continue
        candidates.append({
            'source':'EDGE_CREASE_CLUSTER','component_id':int(m['id']),'quality':float(rep_score),
            'anchor_px':ctr,'tangent_px':tangent,'normal_px':normal,'evidence_px':evidence,
            'region_mask':reg,'crease_mask':np.asarray(m['mask'],dtype=np.uint8),
            'outward_room_px':float(outward_px),'inward_room_px':float(inward_px),
            'fold_score':float(m['fold_score']),'span_px':float(m['span_px']),
            'contour_distance_mm':float(m['contour_median_mm']),
        })
    if not candidates:
        print(f"[E150-EDGE-CLUSTER-BLOCK] score={cluster['score']:.3f} members={len(cluster['members'])} cluster found but no member has valid exact-normal BODY/FLAP room")
        return {'cluster_found':True,'candidates':[],'cluster':cluster}
    candidates.sort(key=lambda q:q['quality'],reverse=True)
    candidates=candidates[:3]
    print(
        f"[E151-EDGE-CLUSTER-SELECT] side={cluster['side']} score={cluster['score']:.3f} base={float(cluster.get('base_score',0.0)):.3f} "
        f"poseSide={str(cluster.get('pose_side','NONE'))}:{float(cluster.get('pose_confidence',0.0)):.2f} poseTerm={float(cluster.get('pose_side_term',0.0)):+.3f} "
        f"members={len(cluster['members'])} totalPx={cluster['total_pixels']} radius={cluster['radius_mm']:.1f}mm "
        f"contourMed={cluster['contour_median_mm']:.1f}mm meanFold={cluster['mean_fold']:.3f} "
        f"seam={cluster['seam_ratio']:.2f} representatives={len(candidates)}"
    )
    return {'cluster_found':True,'candidates':candidates,'cluster':cluster}

def _e145_select_exact_fold_normal(depth_state, mask_u8, exec_crease, preview_crease):
    """Select ONE Fold-Crease component and return its exact outward normal.

    E145 deliberately performs no angular sweep.  Candidate authority is:
      execution-qualified Fold-Crease -> preview Fold-Crease -> E144 contour-veto
      surviving raw crease.  Only the single best component is used.

    Normal sign is deterministic: the garment centroid is treated as BODY-side,
    so the selected normal points away from the centroid, toward the likely flap.
    This sign selection does not create a radial *axis*; the axis itself remains
    exactly perpendicular to the selected crease tangent.
    """
    mask=((np.asarray(mask_u8,dtype=np.uint8)>0).astype(np.uint8)*255)
    if cv2.countNonZero(mask)<500:
        return None
    mm=cv2.moments(mask,binaryImage=True)
    gc=np.asarray([float(mm['m10']/max(1e-6,mm['m00'])),float(mm['m01']/max(1e-6,mm['m00']))],dtype=np.float64)
    score_map=np.asarray(depth_state.get('fold_crease_score',np.zeros(mask.shape,dtype=np.float32)),dtype=np.float32)
    inside=cv2.distanceTransform(mask,cv2.DIST_L2,5).astype(np.float32)

    pools=[]
    ex=np.asarray(exec_crease,dtype=np.uint8)
    pv=np.asarray(preview_crease,dtype=np.uint8)
    raw=np.asarray(depth_state.get('e144_fold_contour_filtered_skeleton',np.zeros_like(mask)),dtype=np.uint8)
    if cv2.countNonZero(ex)>=8: pools.append(('EXECUTION',ex,1.00))
    elif cv2.countNonZero(pv)>=8: pools.append(('PREVIEW',pv,0.82))
    elif cv2.countNonZero(raw)>=8: pools.append(('E144_FILTERED_RAW',raw,0.62))
    if not pools:
        print('[E145-FOLD-NORMAL-BLOCK] no contour-veto-surviving Fold-Crease exists')
        return None

    source,crease,source_weight=pools[0]
    crease=((crease>0).astype(np.uint8)*255)
    nc,ll,ss,_=cv2.connectedComponentsWithStats((crease>0).astype(np.uint8),8)
    candidates=[]
    h,w=mask.shape[:2]
    for lid in range(1,nc):
        yy,xx=np.where(ll==lid)
        if xx.size<8: continue
        pts=np.column_stack([xx,yy]).astype(np.float64)
        ctr=pts.mean(axis=0)
        X=pts-ctr[None,:]
        cov=X.T@X
        vals,vecs=np.linalg.eigh(cov)
        tangent=np.asarray(vecs[:,int(np.argmax(vals))],dtype=np.float64)
        nt=float(np.linalg.norm(tangent))
        if nt<1e-9: continue
        tangent/=nt
        normal=np.asarray([-tangent[1],tangent[0]],dtype=np.float64)
        # Choose only the sign with BODY/garment centroid behind the crease.
        if float(np.dot(normal,ctr-gc))<0.0:
            normal=-normal

        # The exact normal must actually have cloth on the outward/flap side.
        outward_px=0.0
        last_inside=None
        for d in np.arange(2.0,260.0,2.0):
            q=ctr+normal*float(d)
            xi=int(round(q[0])); yi=int(round(q[1]))
            if not (0<=xi<w and 0<=yi<h) or mask[yi,xi]<=0:
                break
            outward_px=float(d); last_inside=q.copy()
        inward_px=0.0
        for d in np.arange(2.0,260.0,2.0):
            q=ctr-normal*float(d)
            xi=int(round(q[0])); yi=int(round(q[1]))
            if not (0<=xi<w and 0<=yi<h) or mask[yi,xi]<=0:
                break
            inward_px=float(d)
        if outward_px<18.0 or inward_px<24.0 or last_inside is None:
            continue

        sv=score_map[ll==lid] if score_map.shape==mask.shape else np.asarray([])
        smean=float(np.mean(sv[np.isfinite(sv)])) if sv.size and np.any(np.isfinite(sv)) else 0.35
        span=float(max(ss[lid,cv2.CC_STAT_WIDTH],ss[lid,cv2.CC_STAT_HEIGHT]))
        iv=inside[ll==lid]
        interior=float(np.median(iv[np.isfinite(iv)])) if iv.size and np.any(np.isfinite(iv)) else 0.0
        # Score selects WHICH Fold-Crease component to trust; it never rotates it.
        quality=(0.52*np.clip(smean,0.0,1.0)+
                 0.20*np.clip(span/140.0,0.0,1.0)+
                 0.16*np.clip(outward_px/100.0,0.0,1.0)+
                 0.12*np.clip(interior/22.0,0.0,1.0))*float(source_weight)
        comp=((ll==lid).astype(np.uint8)*255)
        candidates.append((float(quality),int(lid),ctr,tangent,normal,outward_px,inward_px,smean,span,interior,comp))

    if not candidates:
        print(f'[E145-FOLD-NORMAL-BLOCK] source={source} components={max(0,nc-1)} but no component has valid BODY/FLAP normal room')
        return None
    candidates.sort(key=lambda z:z[0],reverse=True)
    quality,lid,ctr,tangent,normal,outward_px,inward_px,smean,span,interior,comp=candidates[0]

    # Evidence lies on the outward side of the SAME exact normal.  It is only a
    # direction marker for E130; the selected crease mask/anchor are passed explicitly.
    evidence_dist=float(np.clip(0.72*outward_px,18.0,max(18.0,outward_px-4.0)))
    evidence=ctr+normal*evidence_dist
    reg=np.zeros_like(mask,dtype=np.uint8)
    cv2.circle(reg,tuple(np.rint(evidence).astype(int)),8,255,-1,cv2.LINE_AA)
    reg=cv2.bitwise_and(reg,mask)
    if cv2.countNonZero(reg)<8:
        cv2.circle(reg,tuple(np.rint(evidence).astype(int)),14,255,-1,cv2.LINE_AA)
        reg=cv2.bitwise_and(reg,mask)
    if cv2.countNonZero(reg)<8:
        print('[E145-FOLD-NORMAL-BLOCK] exact-normal evidence region could not be kept inside garment')
        return None

    print(
        f'[E145-FOLD-NORMAL-SELECT] source={source} component={lid} '
        f'quality={quality:.3f} foldScore={smean:.3f} span={span:.1f}px '
        f'normal=({normal[0]:+.4f},{normal[1]:+.4f}) '
        f'outwardRoom={outward_px:.1f}px inwardRoom={inward_px:.1f}px '
        'rotationSearch=OFF'
    )
    return {
        'source':source,'component_id':int(lid),'quality':float(quality),
        'anchor_px':np.asarray(ctr,dtype=np.float64),'tangent_px':np.asarray(tangent,dtype=np.float64),
        'normal_px':np.asarray(normal,dtype=np.float64),'evidence_px':np.asarray(evidence,dtype=np.float64),
        'region_mask':reg,'crease_mask':comp,'outward_room_px':float(outward_px),
        'inward_room_px':float(inward_px),'fold_score':float(smean),'span_px':float(span),
    }



def _v11_rank_exact_fold_normals(depth_state, mask_u8, exec_crease, preview_crease, max_candidates=6, H_value=None, args=None):
    ex=((np.asarray(exec_crease,dtype=np.uint8)>0).astype(np.uint8)*255)
    pv=((np.asarray(preview_crease,dtype=np.uint8)>0).astype(np.uint8)*255)
    use_exec=bool(cv2.countNonZero(ex)>=8)
    work=ex.copy() if use_exec else pv.copy()
    out=[]
    for _ in range(max(1,int(max_candidates))):
        if cv2.countNonZero(work)<8:
            break
        q=_e145_select_exact_fold_normal(depth_state,mask_u8,work,np.zeros_like(work,dtype=np.uint8))
        if not isinstance(q,dict):
            break
        out.append(q)
        cm=np.asarray(q.get('crease_mask',np.zeros_like(work)),dtype=np.uint8)>0
        if not np.any(cm):
            break
        rm=cv2.dilate((cm.astype(np.uint8)*255),cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5)),iterations=1)>0
        work[rm]=0

    target=_v30_target_board(args) if args is not None else None
    if target is not None and H_value is not None:
        raw=ex if use_exec else pv
        n,lab,stats,_=cv2.connectedComponentsWithStats((raw>0).astype(np.uint8),8)
        nearest=None
        for lid in range(1,n):
            if int(stats[lid,cv2.CC_STAT_AREA])<8:
                continue
            comp=((lab==lid).astype(np.uint8)*255)
            d=_v30_mask_target_distance_mm(comp,H_value,target)
            if nearest is None or d<nearest[0]:
                nearest=(float(d),comp)
        if nearest is not None:
            d,comp=nearest
            duplicate=False
            for q in out:
                qm=(np.asarray(q.get('crease_mask',np.zeros_like(comp)),dtype=np.uint8)>0)
                cm=(comp>0)
                inter=int(np.count_nonzero(qm & cm)); union=int(np.count_nonzero(qm | cm))
                if union>0 and float(inter)/float(union)>=0.45:
                    q['v30_target_seed_distance_mm']=float(d)
                    q['v30_target_guided_planner']=True
                    duplicate=True
                    break
            if not duplicate:
                tq=_e145_select_exact_fold_normal(depth_state,mask_u8,comp,np.zeros_like(comp,dtype=np.uint8))
                if isinstance(tq,dict):
                    tq['v30_target_seed_distance_mm']=float(d)
                    tq['v30_target_guided_planner']=True
                    out.insert(0,tq)
            if out:
                target_items=[q for q in out if q.get('v30_target_seed_distance_mm') is not None]
                other=[q for q in out if q.get('v30_target_seed_distance_mm') is None]
                target_items.sort(key=lambda q:float(q.get('v30_target_seed_distance_mm',1e9)))
                out=target_items+other
                print(f"[E156-V30-TARGET-CREASE-SEED] pool={'EXECUTION' if use_exec else 'PREVIEW'} nearest={d:.1f}mm candidates={len(out)}")
    if out:
        print(f"[E156-V11-CREASE-RERANK] pool={'EXECUTION' if use_exec else 'PREVIEW'} candidates={len(out)} ids={[int(q.get('component_id',-1)) for q in out]}")
    return out

def _v9_try_same_crease_near_normal_pair(depth_state, mask_u8, H_value, args, config, cfg,
                                          fake, exec_crease, preview_crease, *, label):
    """Try a tiny direction fan while preserving the exact same crease authority.

    The crease mask, anchor, evidence point, and ROI are copied unchanged. Only the
    explicit normal vector is rotated. Each candidate still passes the complete E130
    board/reach/path mechanics and the caller's final same-crease role validation.
    """
    base=np.asarray(fake.get('explicit_axis_unit_px'),dtype=np.float64).reshape(2)
    nb=float(np.linalg.norm(base))
    if (not np.all(np.isfinite(base))) or nb<1e-9:
        return None
    base=base/nb
    fan=getattr(args,'micro_unfold_vla_near_normal_fan_deg',(-12.0,-8.0,-4.0,4.0,8.0,12.0))
    try:
        fan=[float(x) for x in list(fan)]
    except Exception:
        fan=[-12.0,-8.0,-4.0,4.0,8.0,12.0]
    fan=[x for x in fan if np.isfinite(x) and 0.5<=abs(x)<=15.0]
    fan=sorted(set(fan),key=lambda x:(abs(x),x))
    for deg in fan:
        u=_e133_rotate_unit_px(base,deg)
        if u is None:
            continue
        alt=dict(fake)
        alt['explicit_axis_unit_px']=np.asarray(u,dtype=np.float64)
        try:
            pair=_e130_directional_pair(depth_state,mask_u8,H_value,args,config,cfg,alt,exec_crease,preview_crease)
        except Exception as exc:
            print(f'[E156-V9-NEAR-NORMAL-REJECT] label={label} rot={deg:+.1f}deg reason={exc!r}')
            pair=None
        if isinstance(pair,dict):
            pair['e133_axis_rotation_deg']=float(deg)
            pair['e156_v9_same_crease_near_normal']=True
            pair['e156_v9_same_crease_near_normal_label']=str(label)
            print(f'[E156-V9-NEAR-NORMAL-ACCEPT] label={label} rot={deg:+.1f}deg sameCrease=True hardSafety=PASS')
            return pair
    print(f'[E156-V9-NEAR-NORMAL-BLOCK] label={label} tested={fan} sameCrease=True hardSafety=NO_PAIR')
    return None



def _v13_try_same_crease_anchor_pairs(depth_state, mask_u8, H_value, args, config, cfg,
                                       fake, exec_crease, preview_crease, *, label, max_trials=None):
    cm=((np.asarray(fake.get('explicit_crease_mask',np.zeros_like(mask_u8)),dtype=np.uint8)>0).astype(np.uint8)*255)
    if cv2.countNonZero(cm)<8:
        return None
    axis=_v6_unit_xy(fake.get('explicit_axis_unit_px'))
    if axis is None:
        return None
    yy,xx=np.where(cm>0)
    if xx.size<8:
        return None
    pts=np.column_stack([xx,yy]).astype(np.float64)
    tangent=_v6_branch_tangent(cm)
    if tangent is None:
        return None
    proj=pts@tangent
    lo=float(np.percentile(proj,5.0)); hi=float(np.percentile(proj,95.0))
    if hi-lo<6.0:
        return None
    base_anchor=np.asarray(fake.get('explicit_crease_anchor_px',np.mean(pts,axis=0)),dtype=np.float64).reshape(2)
    base_evidence=np.asarray(fake.get('explicit_evidence_px',base_anchor+axis*24.0),dtype=np.float64).reshape(2)
    evidence_dist=float(np.dot(base_evidence-base_anchor,axis))
    if not np.isfinite(evidence_dist) or evidence_dist<8.0:
        evidence_dist=24.0

    fractions=[]
    target=_v30_target_board(args)
    target_frac=None
    if target is not None:
        try:
            tp=np.asarray(_fix89_board_to_px(H_value,target),dtype=np.float64).reshape(2)
            tf=(float(tp@tangent)-lo)/(hi-lo)
            target_frac=float(np.clip(tf,0.06,0.94))
            for off in (0.0,-0.08,0.08,-0.16,0.16):
                fractions.append(float(np.clip(target_frac+off,0.06,0.94)))
        except Exception:
            target_frac=None
    fractions.extend((0.40,0.60,0.30,0.70,0.22,0.78,0.14,0.86))
    dedup=[]
    for f in fractions:
        if not any(abs(float(f)-q)<0.025 for q in dedup):
            dedup.append(float(f))
    fractions=tuple(dedup)

    mean=np.mean(pts,axis=0)
    tested=0
    for frac in fractions:
        if max_trials is not None and tested>=max(1,int(max_trials)):
            break
        target_proj=lo+float(frac)*(hi-lo)
        anchor=mean+tangent*(target_proj-float(mean@tangent))
        j=int(np.argmin(np.sum((pts-anchor[None,:])**2,axis=1)))
        anchor=pts[j].copy()
        if float(np.linalg.norm(anchor-base_anchor))<3.0:
            continue
        evidence=anchor+axis*evidence_dist
        reg=np.zeros_like(cm,dtype=np.uint8)
        cv2.circle(reg,tuple(np.rint(evidence).astype(int)),8,255,-1,cv2.LINE_AA)
        reg=cv2.bitwise_and(reg,np.asarray(mask_u8,dtype=np.uint8))
        alt=dict(fake)
        alt['explicit_crease_anchor_px']=np.asarray(anchor,dtype=np.float64)
        alt['explicit_evidence_px']=np.asarray(evidence,dtype=np.float64)
        alt['region_mask']=reg
        tested+=1
        try:
            pair=_e130_directional_pair(depth_state,mask_u8,H_value,args,config,cfg,alt,exec_crease,preview_crease)
        except Exception as exc:
            print(f'[E156-V30-ANCHOR-REJECT] label={label} frac={frac:.2f} reason={exc!r}')
            pair=None
        if isinstance(pair,dict):
            pair['v13_same_crease_anchor_search_used']=True
            pair['v13_same_crease_anchor_fraction']=float(frac)
            pair['v13_same_crease_anchor_count']=int(tested)
            pair['v13_same_crease_anchor_label']=str(label)
            pair['v30_target_guided_planner']=bool(target is not None)
            pair['v30_target_anchor_fraction']=None if target_frac is None else float(target_frac)
            print(f'[E156-V30-ANCHOR-ACCEPT] label={label} frac={frac:.2f} targetFrac={target_frac} sameCrease=True exactNormal=True hardSafety=PASS')
            return pair
    print(f'[E156-V30-ANCHOR-BLOCK] label={label} tested={tested} targetFrac={target_frac} sameCrease=True exactNormal=True hardSafety=NO_PAIR')
    return None

def _e133_directional_pair(depth_state,mask_u8,H_value,args,config,cfg,fold_region_info,exec_crease,preview_crease):
    """E156 V9 direction arbitration.

    Authority:
      1) admitted Pose contour intrusion (legacy strong Pose authority),
      2) LOWER_LEFT/LOWER_RIGHT TORSO_QUAD asymmetry + matching Fold Boundary,
      3) DIRECT ID7/ID8 lower-corner rescue when TORSO_QUAD is absent,
      4) E150 EDGE_CREASE_CLUSTER rescue,
      5) one generic exact Fold-Crease normal,
      6) BLOCK.

    E154 changes two failure semantics only.  A missing TORSO_QUAD no longer
    prevents ID7/ID8 from anchoring a real lower Fold Boundary.  Also, an EDGE
    cluster whose exact normals all fail hard mechanics is no longer an immediate
    global block: the generic exact-normal path is still tried.  If that final
    path also fails, the existing E151 hard-block flag is set only then so legacy
    PREVIEW-ONLY HOLD/ACTIVE cannot overwrite the diagnostic display.

    V8 permits only a tiny near-normal fan around an already-selected exact
    Fold-Crease when the 0-degree normal fails mechanics. The crease/anchor never
    changes; radial-axis, contour-extreme, unrelated-crease, and geometric
    last-resort searches remain forbidden.
    """
    mask=((np.asarray(mask_u8,dtype=np.uint8)>0).astype(np.uint8)*255)
    if cv2.countNonZero(mask)<500: return None
    depth_state.pop('e151_directional_hard_block_reason',None)
    depth_state.pop('e151_edge_cluster_hard_block',None)
    depth_state.pop('e151_edge_cluster_block_info',None)
    depth_state.pop('e154_edge_fallthrough_info',None)

    # A. Strong admitted Pose contour intrusion keeps first authority.
    intr=[q for q in list(depth_state.get('pose_fold_intrusions',[]) or []) if isinstance(q,dict)]
    groups={}
    for q in intr:
        try:
            conf=float(q.get('conf',0.0)); strength=float(q.get('strength',0.0)); excess=float(q.get('inside_distance_px',0.0))-float(q.get('threshold_px',0.0))
        except Exception: continue
        if conf<0.20 or excess<=0.5: continue
        g=str(q.get('group','OTHER')); merit=max(0.01,conf)*(0.20+0.80*max(0.0,strength))*(1.0+0.015*max(0.0,excess))
        groups.setdefault(g,[]).append((merit,q))
    ranked=[]
    for g,items in groups.items(): ranked.append((float(sum(v[0] for v in items)),g,items))
    ranked.sort(reverse=True,key=lambda z:z[0])
    for merit,g,items in ranked:
        fake=dict(fold_region_info or {}); fake['source']='POSE_INTRUSION'; fake['group']=str(g)
        fake['confidence']=float(np.clip(0.42+0.18*min(2,len(items))+0.18*np.tanh(merit),0.0,0.98))
        print(f'[E146-POSE-AXIS-TRY] group={g} points={len(items)} merit={merit:.3f} authority=ANY_INTRUDED_KPT')
        try: pair=_e130_directional_pair(depth_state,mask,H_value,args,config,cfg,fake,exec_crease,preview_crease)
        except Exception as exc:
            print(f'[E146-POSE-AXIS-REJECT] group={g} reason={exc!r}'); pair=None
        if isinstance(pair,dict):
            pair['e133_axis_mode']='POSE_ANY_INTRUSION'; pair['e133_axis_group']=str(g); pair['e137_rgb_only']=True
            pair.update({'depth_role_probe_valid':False,'depth_role_probe_delta_mm':0.0,'depth_role_probe_confidence':0.0,
                         'depth_role_valid':False,'active_depth_role_mm':0.0,'hold_depth_role_mm':0.0,
                         'depth_role_delta_mm':0.0,'depth_role_confidence':0.0,'active_depth_reference':0.0,
                         'e124_branch_unidepth_support':0.0,'active_pose_depth_authority':0.0,
                         'active_selection_mode':'E154_POSE_THEN_LOWER_CORNER_EDGE_OR_EXACT'})
            pair['execution_evidence_reason']='E146_POSE_RGB_CREASE_MECHANICALLY_SAFE'
            print(f'[E146-POSE-AXIS-ACCEPT] group={g} -> READY')
            return pair

    # B. Original authoritative TORSO_QUAD lower-fold path is unchanged.
    lower_region=str(depth_state.get('pose_deform_fold_region_hint','NONE'))
    lower_conf=float(depth_state.get('pose_deform_fold_region_confidence',0.0) or 0.0)
    lower_gate=bool(lower_region in ('LOWER_LEFT','LOWER_RIGHT') and lower_conf>=0.18)
    if lower_gate:
        lower=_e146_lower_pose_fold_normal(depth_state,mask,exec_crease,preview_crease)
        if isinstance(lower,dict):
            fake={
                'source':'RGB_CREASE_NORMAL_EXACT','group':str(lower_region),
                'confidence':float(max(lower_conf,float(lower.get('quality',0.0)))),
                'region_mask':np.asarray(lower['region_mask'],dtype=np.uint8),
                'explicit_axis_unit_px':np.asarray(lower['normal_px'],dtype=np.float64),
                'explicit_evidence_px':np.asarray(lower['evidence_px'],dtype=np.float64),
                'explicit_crease_anchor_px':np.asarray(lower['anchor_px'],dtype=np.float64),
                'explicit_crease_mask':np.asarray(lower['crease_mask'],dtype=np.uint8),
            }
            print(
                f"[E146-LOWER-FOLD-NORMAL-TRY] region={lower_region}:{lower_conf:.2f} "
                f"component={int(lower.get('component_id',-1))} flapAnchor={str(lower.get('lower_pose_anchor_name','ID7/8'))} "
                f"normal=({float(lower['normal_px'][0]):+.4f},{float(lower['normal_px'][1]):+.4f}) rotation=0deg only; hardSafety=ON"
            )
            try: pair=_e130_directional_pair(depth_state,mask,H_value,args,config,cfg,fake,exec_crease,preview_crease)
            except Exception as exc:
                print(f'[E146-LOWER-FOLD-NORMAL-REJECT] reason={exc!r}'); pair=None
            if not isinstance(pair,dict):
                pair=_v13_try_same_crease_anchor_pairs(depth_state,mask,H_value,args,config,cfg,fake,exec_crease,preview_crease,label=f'LOWER_{lower_region}')
            if not isinstance(pair,dict):
                pair=_v9_try_same_crease_near_normal_pair(depth_state,mask,H_value,args,config,cfg,fake,exec_crease,preview_crease,label=f'LOWER_{lower_region}')
            if isinstance(pair,dict):
                pair['e133_axis_mode']='LOWER_POSE_ASYM_EXACT_FOLD_NORMAL'; pair['e133_axis_group']=str(lower_region); pair.setdefault('e133_axis_rotation_deg',0.0)
                pair['primary_fold_region_source']='LOWER_POSE_ASYM_FOLD_BOUNDARY'; pair['e137_rgb_only']=True
                pair['lower_pose_anchor_name']=str(lower.get('lower_pose_anchor_name',''))
                pair['lower_pose_anchor_px']=np.asarray(lower.get('lower_pose_anchor_px'),dtype=np.float64)
                pair.update({'depth_role_probe_valid':False,'depth_role_probe_delta_mm':0.0,'depth_role_probe_confidence':0.0,
                             'depth_role_valid':False,'active_depth_role_mm':0.0,'hold_depth_role_mm':0.0,
                             'depth_role_delta_mm':0.0,'depth_role_confidence':0.0,'active_depth_reference':0.0,
                             'e124_branch_unidepth_support':0.0,'active_pose_depth_authority':0.0,
                             'active_selection_mode':'E146_LOWER_POSE_ASYM_ID7_ID8_FOLD_BOUNDARY'})
                pair['execution_evidence_reason']='E146_LOWER_POSE_ASYM_FOLD_BOUNDARY_MECHANICALLY_SAFE'
                print(f'[E146-LOWER-FOLD-NORMAL-ACCEPT] {lower_region} -> READY; ID7/8 used as FLAP anchor, contour intrusion not required, same-crease near-normal fallback allowed after exact-normal failure')
                return pair
            # Keep the original strong lower-authority semantics: a real TORSO_QUAD
            # match must not silently jump to an unrelated crease.
            print(
                f'[E146-DIRECTIONAL-BLOCK] lowerRegion={lower_region}:{lower_conf:.2f} '
                f"selectedFoldComponent={int(lower.get('component_id',-1))} exact lower normal has no "
                'board/reach/path-safe HOLD/ACTIVE pair; unrelated crease fallback forbidden'
            )
            return None
        print(
            f'[E146-LOWER-FOLD-FALLTHROUGH] region={lower_region}:{lower_conf:.2f} '
            'Pose lower asymmetry exists but no matching Fold Boundary was selected; direct ID7/ID8 rescue remains available'
        )

    # B2. E154 direct lower-corner rescue.  This is intentionally weaker than
    # TORSO_QUAD and therefore falls through if its exact-normal pair is unsafe.
    direct_lower=_e154_direct_lower_corner_rescue(depth_state,mask,exec_crease,preview_crease)
    if direct_lower:
        for rank,lower in enumerate(direct_lower[:2],1):
            region=str(lower.get('lower_pose_region','LOWER_DIRECT'))
            id_label=str(lower.get('e154_corner_id','ID7/8'))
            fake={
                'source':'RGB_CREASE_NORMAL_EXACT','group':f'E154_{region}_{id_label}',
                'confidence':float(max(float(lower.get('quality',0.0)),float(lower.get('e154_direct_score',0.0)))),
                'region_mask':np.asarray(lower['region_mask'],dtype=np.uint8),
                'explicit_axis_unit_px':np.asarray(lower['normal_px'],dtype=np.float64),
                'explicit_evidence_px':np.asarray(lower['evidence_px'],dtype=np.float64),
                'explicit_crease_anchor_px':np.asarray(lower['anchor_px'],dtype=np.float64),
                'explicit_crease_mask':np.asarray(lower['crease_mask'],dtype=np.uint8),
            }
            print(
                f"[E154-LOWER-CORNER-NORMAL-TRY] rank={rank}/{min(2,len(direct_lower))} {id_label} "
                f"component={int(lower.get('component_id',-1))} foldScore={float(lower.get('fold_score',0.0)):.3f} "
                f"normal=({float(lower['normal_px'][0]):+.4f},{float(lower['normal_px'][1]):+.4f}) rotation=0deg only; hardSafety=ON"
            )
            try: pair=_e130_directional_pair(depth_state,mask,H_value,args,config,cfg,fake,exec_crease,preview_crease)
            except Exception as exc:
                print(f'[E154-LOWER-CORNER-NORMAL-REJECT] {id_label} reason={exc!r}'); pair=None
            if not isinstance(pair,dict):
                pair=_v13_try_same_crease_anchor_pairs(depth_state,mask,H_value,args,config,cfg,fake,exec_crease,preview_crease,label=f'DIRECT_{id_label}')
            if not isinstance(pair,dict):
                pair=_v9_try_same_crease_near_normal_pair(depth_state,mask,H_value,args,config,cfg,fake,exec_crease,preview_crease,label=f'DIRECT_{id_label}')
            if isinstance(pair,dict):
                pair['e133_axis_mode']='LOWER_CORNER_DIRECT_EXACT_FOLD_NORMAL'; pair['e133_axis_group']=str(fake['group']); pair.setdefault('e133_axis_rotation_deg',0.0)
                pair['primary_fold_region_source']='LOWER_CORNER_DIRECT_FOLD_BOUNDARY'; pair['e137_rgb_only']=True
                pair['lower_pose_anchor_name']=str(lower.get('lower_pose_anchor_name',''))
                pair['lower_pose_anchor_px']=np.asarray(lower.get('lower_pose_anchor_px'),dtype=np.float64)
                pair['e154_corner_id']=id_label
                pair.update({'depth_role_probe_valid':False,'depth_role_probe_delta_mm':0.0,'depth_role_probe_confidence':0.0,
                             'depth_role_valid':False,'active_depth_role_mm':0.0,'hold_depth_role_mm':0.0,
                             'depth_role_delta_mm':0.0,'depth_role_confidence':0.0,'active_depth_reference':0.0,
                             'e124_branch_unidepth_support':0.0,'active_pose_depth_authority':0.0,
                             'active_selection_mode':'E154_DIRECT_ID7_ID8_FOLD_BOUNDARY'})
                pair['execution_evidence_reason']='E154_DIRECT_LOWER_CORNER_MECHANICALLY_SAFE'
                print(f'[E154-LOWER-CORNER-NORMAL-ACCEPT] {id_label} -> READY; TORSO_QUAD not required; exact Fold Boundary normal only')
                return pair
        print('[E154-LOWER-CORNER-FALLTHROUGH] qualifying ID7/ID8 Fold Boundary found but no direct exact-normal pair passed hard mechanics; continuing to EDGE/generic exact-normal')

    # C. EDGE cluster may try its real member boundaries.  Unlike E151, failure is
    # recorded as pending evidence and does NOT stop the generic exact-normal path.
    edge_failed_info=None
    edge_cluster=_e150_select_edge_crease_cluster(depth_state,mask,H_value,exec_crease,preview_crease)
    if isinstance(edge_cluster,dict) and bool(edge_cluster.get('cluster_found',False)):
        edge_candidates=list(edge_cluster.get('candidates',[]) or [])[:3]
        cluster_info=dict(edge_cluster.get('cluster',{}) or {})
        if not edge_candidates:
            edge_failed_info={'reason':'E154_EDGE_CLUSTER_NO_EXACT_NORMAL_REPRESENTATIVE','score':float(cluster_info.get('score',0.0)),'side':str(cluster_info.get('side','NONE')),'members':int(len(cluster_info.get('members',[]) or [])),'tested_normals':0}
            print('[E154-EDGE-FALLTHROUGH] EDGE cluster has no exact-normal representative; generic exact Fold-Crease will still be tried')
        else:
            for rank,edge in enumerate(edge_candidates,1):
                fake={
                    'source':'RGB_CREASE_NORMAL_EXACT','group':f'EDGE_CLUSTER_{rank}',
                    'confidence':float(max(float(edge.get('quality',0.0)),float(cluster_info.get('score',0.0)))),
                    'region_mask':np.asarray(edge['region_mask'],dtype=np.uint8),
                    'explicit_axis_unit_px':np.asarray(edge['normal_px'],dtype=np.float64),
                    'explicit_evidence_px':np.asarray(edge['evidence_px'],dtype=np.float64),
                    'explicit_crease_anchor_px':np.asarray(edge['anchor_px'],dtype=np.float64),
                    'explicit_crease_mask':np.asarray(edge['crease_mask'],dtype=np.uint8),
                }
                print(
                    f"[E151-EDGE-NORMAL-TRY] rank={rank}/{len(edge_candidates)} component={int(edge.get('component_id',-1))} "
                    f"foldScore={float(edge.get('fold_score',0.0)):.3f} contour={float(edge.get('contour_distance_mm',0.0)):.1f}mm "
                    f"normal=({float(edge['normal_px'][0]):+.4f},{float(edge['normal_px'][1]):+.4f}) rotation=0deg only; hardSafety=ON"
                )
                try: pair=_e130_directional_pair(depth_state,mask,H_value,args,config,cfg,fake,exec_crease,preview_crease)
                except Exception as exc:
                    print(f'[E151-EDGE-NORMAL-REJECT] rank={rank} reason={exc!r}'); pair=None
                if not isinstance(pair,dict):
                    pair=_v13_try_same_crease_anchor_pairs(depth_state,mask,H_value,args,config,cfg,fake,exec_crease,preview_crease,label=f'EDGE_{rank}')
                if not isinstance(pair,dict):
                    pair=_v9_try_same_crease_near_normal_pair(depth_state,mask,H_value,args,config,cfg,fake,exec_crease,preview_crease,label=f'EDGE_{rank}')
                if isinstance(pair,dict):
                    pair['e133_axis_mode']='EDGE_CREASE_CLUSTER_EXACT_NORMAL'; pair['e133_axis_group']=str(fake['group']); pair.setdefault('e133_axis_rotation_deg',0.0)
                    pair['primary_fold_region_source']='EDGE_CREASE_CLUSTER_FOLD_BOUNDARY'; pair['e137_rgb_only']=True
                    pair['e150_edge_cluster_score']=float(cluster_info.get('score',0.0)); pair['e150_edge_cluster_members']=int(len(cluster_info.get('members',[]) or [])); pair['e150_edge_cluster_side']=str(cluster_info.get('side','NONE'))
                    pair.update({'depth_role_probe_valid':False,'depth_role_probe_delta_mm':0.0,'depth_role_probe_confidence':0.0,
                                 'depth_role_valid':False,'active_depth_role_mm':0.0,'hold_depth_role_mm':0.0,
                                 'depth_role_delta_mm':0.0,'depth_role_confidence':0.0,'active_depth_reference':0.0,
                                 'e124_branch_unidepth_support':0.0,'active_pose_depth_authority':0.0,
                                 'active_selection_mode':'E150_EDGE_CREASE_CLUSTER_EXACT_NORMAL'})
                    pair['execution_evidence_reason']='E150_EDGE_CREASE_CLUSTER_MECHANICALLY_SAFE'
                    print(f'[E151-EDGE-NORMAL-ACCEPT] rank={rank} -> READY; cluster FLAP side is nearest Cloth Contour; same-crease near-normal fallback allowed after exact-normal failure')
                    return pair
            edge_failed_info={'reason':'E154_EDGE_CLUSTER_EXACT_NORMALS_FAILED_HARD_SAFETY','score':float(cluster_info.get('score',0.0)),'side':str(cluster_info.get('side','NONE')),'members':int(len(cluster_info.get('members',[]) or [])),'tested_normals':int(len(edge_candidates))}
            print(
                f"[E154-EDGE-FALLTHROUGH] EDGE_CREASE_CLUSTER side={str(cluster_info.get('side','NONE'))} score={float(cluster_info.get('score',0.0)):.3f} "
                f"members={len(cluster_info.get('members',[]) or [])} all {len(edge_candidates)} exact normals failed hard mechanics; generic exact Fold-Crease remains enabled"
            )
        depth_state['e154_edge_fallthrough_info']=dict(edge_failed_info or {})

    # D. Generic exact Fold-Crease remains the final executable fallback.
    # V11 tries several distinct REAL RGB crease components from the same snapshot.
    # A top branch with ACTIVE=0 / hard reach failure no longer suppresses other
    # execution-qualified folds. Every candidate still uses exact/small-fan geometry
    # and the unchanged board/reach/path safety checks.
    _rerank_max=max(1,int(getattr(args,'micro_unfold_vla_exact_crease_rerank_max',6)))
    exact_ranked=_v11_rank_exact_fold_normals(depth_state,mask,exec_crease,preview_crease,_rerank_max,H_value=H_value,args=args)
    exact=exact_ranked[0] if exact_ranked else None
    for _rank,exact_cand in enumerate(exact_ranked,1):
        fake={
            'source':'RGB_CREASE_NORMAL_EXACT','group':f"FOLD_NORMAL_{int(exact_cand.get('component_id',-1))}",
            'confidence':float(exact_cand.get('quality',0.0)),
            'region_mask':np.asarray(exact_cand['region_mask'],dtype=np.uint8),
            'explicit_axis_unit_px':np.asarray(exact_cand['normal_px'],dtype=np.float64),
            'explicit_evidence_px':np.asarray(exact_cand['evidence_px'],dtype=np.float64),
            'explicit_crease_anchor_px':np.asarray(exact_cand['anchor_px'],dtype=np.float64),
            'explicit_crease_mask':np.asarray(exact_cand['crease_mask'],dtype=np.uint8),
        }
        print(f"[E156-V11-CREASE-RERANK-TRY] rank={_rank}/{len(exact_ranked)} component={int(exact_cand.get('component_id',-1))} Q={float(exact_cand.get('quality',0.0)):.3f} fold={float(exact_cand.get('fold_score',0.0)):.3f}")
        try: pair=_e130_directional_pair(depth_state,mask,H_value,args,config,cfg,fake,exec_crease,preview_crease)
        except Exception as exc:
            print(f'[E156-V11-CREASE-RERANK-REJECT] rank={_rank} exact reason={exc!r}'); pair=None
        if not isinstance(pair,dict):
            pair=_v13_try_same_crease_anchor_pairs(depth_state,mask,H_value,args,config,cfg,fake,exec_crease,preview_crease,label=f'GENERIC_EXACT_R{_rank}')
        if not isinstance(pair,dict):
            pair=_v9_try_same_crease_near_normal_pair(depth_state,mask,H_value,args,config,cfg,fake,exec_crease,preview_crease,label=f'GENERIC_EXACT_R{_rank}')
        if isinstance(pair,dict):
            pair['e133_axis_mode']='RGB_CREASE_NORMAL_EXACT'; pair['e133_axis_group']=str(fake['group']); pair.setdefault('e133_axis_rotation_deg',0.0)
            pair['primary_fold_region_source']='RGB_CREASE_EXACT_NORMAL_EXTENT'; pair['e137_rgb_only']=True
            pair['e156_v11_reranked_crease_rank']=int(_rank); pair['e156_v11_reranked_crease_count']=int(len(exact_ranked))
            pair['v30_target_guided_planner']=bool(_v30_target_board(args) is not None)
            pair['v30_target_seed_distance_mm']=exact_cand.get('v30_target_seed_distance_mm')
            pair.update({'depth_role_probe_valid':False,'depth_role_probe_delta_mm':0.0,'depth_role_probe_confidence':0.0,
                         'depth_role_valid':False,'active_depth_role_mm':0.0,'hold_depth_role_mm':0.0,
                         'depth_role_delta_mm':0.0,'depth_role_confidence':0.0,'active_depth_reference':0.0,
                         'e124_branch_unidepth_support':0.0,'active_pose_depth_authority':0.0,
                         'active_selection_mode':'E156_V11_RERANKED_EXACT_FOLD_CREASE_NORMAL'})
            pair['execution_evidence_reason']='E156_V11_RERANKED_EXACT_FOLD_NORMAL_MECHANICALLY_SAFE'
            print(f'[E156-V11-CREASE-RERANK-ACCEPT] rank={_rank} -> READY')
            return pair
        print(f'[E156-V11-CREASE-RERANK-NEXT] rank={_rank} no safe pair; trying next real RGB crease')
    if not exact_ranked:
        print(f'[E146-DIRECTIONAL-BLOCK] poseGroups={len(ranked)} exactFoldNormal=UNAVAILABLE')

    # Only NOW, after direct lower rescue + EDGE + generic exact-normal are all
    # exhausted, arm the old suppression flag so a diagnostic legacy pair cannot
    # masquerade as the selected HOLD/ACTIVE result.
    if edge_failed_info:
        _reason=str(edge_failed_info.get('reason','E154_EDGE_CLUSTER_AND_GENERIC_FAILED'))
        depth_state['e151_directional_hard_block_reason']=_reason
        depth_state['e151_edge_cluster_hard_block']=True
        depth_state['e151_edge_cluster_block_info']=dict(edge_failed_info)
        print(
            f"[E154-FINAL-PREVIEW-SUPPRESS] EDGE evidence failed mechanics and generic exact-normal also failed; "
            f"side={str(edge_failed_info.get('side','NONE'))} score={float(edge_failed_info.get('score',0.0)):.3f}; legacy PREVIEW-ONLY HOLD/ACTIVE remains suppressed"
        )
        return None

    if isinstance(exact,dict):
        print(
            f"[E146-DIRECTIONAL-BLOCK] poseGroups={len(ranked)} selectedFoldComponent={int(exact.get('component_id',-1))} "
            'exact normal has no board/reach/path-safe HOLD/ACTIVE pair; no rotation/geometric fallback permitted'
        )
    return None


# v32 keeps the entire v31 planner as the primary fast path.  The wrapper below
# runs only after that path has exhausted all of its existing exact/anchor/fan
# searches.  This makes the new work bounded and failure-only.
_e133_directional_pair_v31_primary = _e133_directional_pair


def _v32_fake_from_exact(exact_cand, group_prefix='V32'):
    return {
        'source':'RGB_CREASE_NORMAL_EXACT',
        'group':f"{group_prefix}_{int(exact_cand.get('component_id',-1))}",
        'confidence':float(exact_cand.get('quality',0.0)),
        'region_mask':np.asarray(exact_cand['region_mask'],dtype=np.uint8),
        'explicit_axis_unit_px':np.asarray(exact_cand['normal_px'],dtype=np.float64),
        'explicit_evidence_px':np.asarray(exact_cand['evidence_px'],dtype=np.float64),
        'explicit_crease_anchor_px':np.asarray(exact_cand['anchor_px'],dtype=np.float64),
        'explicit_crease_mask':np.asarray(exact_cand['crease_mask'],dtype=np.uint8),
    }


def _v32_make_flipped_fake(fake, mask_u8):
    axis=_v6_unit_xy(fake.get('explicit_axis_unit_px'))
    if axis is None:
        return None
    anchor=np.asarray(fake.get('explicit_crease_anchor_px'),dtype=np.float64).reshape(2)
    evidence=np.asarray(fake.get('explicit_evidence_px',anchor+axis*24.0),dtype=np.float64).reshape(2)
    ed=float(abs(np.dot(evidence-anchor,axis)))
    if not np.isfinite(ed) or ed<8.0:
        ed=24.0
    flipped=-np.asarray(axis,dtype=np.float64)
    new_evidence=anchor+flipped*ed
    mask=((np.asarray(mask_u8,dtype=np.uint8)>0).astype(np.uint8)*255)
    reg=np.zeros_like(mask,dtype=np.uint8)
    for rr in (8,14):
        reg.fill(0)
        cv2.circle(reg,tuple(np.rint(new_evidence).astype(int)),int(rr),255,-1,cv2.LINE_AA)
        reg=cv2.bitwise_and(reg,mask)
        if cv2.countNonZero(reg)>=8:
            break
    if cv2.countNonZero(reg)<8:
        return None
    alt=dict(fake)
    alt['explicit_axis_unit_px']=flipped
    alt['explicit_evidence_px']=new_evidence
    alt['region_mask']=reg
    return alt


def _v32_try_flipped_normal_pair(depth_state, mask_u8, H_value, args, config, cfg,
                                  fake, exec_crease, preview_crease, *, label):
    alt=_v32_make_flipped_fake(fake,mask_u8)
    if not isinstance(alt,dict):
        return None
    try:
        pair=_e130_directional_pair(depth_state,mask_u8,H_value,args,config,cfg,alt,exec_crease,preview_crease)
    except Exception as exc:
        print(f'[E156-V32-FLIP-REJECT] label={label} exact reason={exc!r}')
        pair=None
    if not isinstance(pair,dict) and bool(getattr(args,'micro_unfold_v32_flipped_anchor_retry',False)):
        pair=_v13_try_same_crease_anchor_pairs(
            depth_state,mask_u8,H_value,args,config,cfg,alt,exec_crease,preview_crease,
            label=f'V32_FLIP_{label}',max_trials=max(1,int(getattr(args,'micro_unfold_v32_flipped_anchor_max_trials',2))))
    if isinstance(pair,dict):
        pair['e156_v32_flipped_normal_retry']=True
        pair['e156_v32_flipped_normal_label']=str(label)
        print(f'[E156-V32-FLIP-ACCEPT] label={label} sameCrease=True roleSign=FLIPPED hardSafety=PASS')
        return pair
    print(f'[E156-V32-FLIP-BLOCK] label={label} sameCrease=True roleSign=FLIPPED hardSafety=NO_PAIR')
    return None


def _v32_target_local_exact_normals(depth_state, mask_u8, H_value, args, *, max_candidates=3, exclude_masks=None):
    target=_v30_target_board(args)
    if target is None:
        return []
    mask=((np.asarray(mask_u8,dtype=np.uint8)>0).astype(np.uint8)*255)
    if cv2.countNonZero(mask)<500:
        return []
    try:
        target=np.asarray(target,dtype=np.float64).reshape(2)
        tp=np.asarray(_fix89_board_to_px(H_value,target),dtype=np.float64).reshape(2)
    except Exception:
        return []
    radius_mm=max(20.0,float(getattr(args,'micro_unfold_target_match_max_mm',120.0)))
    radii=[]
    for d in ((radius_mm,0.0),(-radius_mm,0.0),(0.0,radius_mm),(0.0,-radius_mm)):
        try:
            qp=np.asarray(_fix89_board_to_px(H_value,target+np.asarray(d,dtype=np.float64)),dtype=np.float64).reshape(2)
            radii.append(float(np.linalg.norm(qp-tp)))
        except Exception:
            pass
    radius_px=float(np.clip(max(radii) if radii else 140.0,24.0,260.0))
    roi=np.zeros_like(mask,dtype=np.uint8)
    cv2.circle(roi,tuple(np.rint(tp).astype(int)),int(math.ceil(radius_px)),255,-1,cv2.LINE_AA)
    roi=cv2.bitwise_and(roi,mask)
    if cv2.countNonZero(roi)<8:
        return []

    sources=[
        ('STRUCTURAL',np.asarray(depth_state.get('fold_crease_structural_skeleton',np.zeros_like(mask)),dtype=np.uint8)),
        ('SENSITIVE',np.asarray(depth_state.get('fold_crease_skeleton',np.zeros_like(mask)),dtype=np.uint8)),
        ('E144_FILTERED',np.asarray(depth_state.get('e144_fold_contour_filtered_skeleton',np.zeros_like(mask)),dtype=np.uint8)),
    ]
    excluded=[]
    for q in list(exclude_masks or []):
        try:
            qm=(np.asarray(q,dtype=np.uint8)>0)
            if qm.shape==mask.shape and np.any(qm):
                excluded.append(qm)
        except Exception:
            pass
    out=[]
    for source_name,src in sources:
        if src.shape!=mask.shape or cv2.countNonZero(src)<8:
            continue
        local=cv2.bitwise_and(((src>0).astype(np.uint8)*255),roi)
        if cv2.countNonZero(local)<8:
            continue
        nc,lab,stats,_=cv2.connectedComponentsWithStats((local>0).astype(np.uint8),8)
        comps=[]
        for lid in range(1,nc):
            if int(stats[lid,cv2.CC_STAT_AREA])<8:
                continue
            comp=((lab==lid).astype(np.uint8)*255)
            d=_v30_mask_target_distance_mm(comp,H_value,target)
            if not np.isfinite(d) or float(d)>radius_mm:
                continue
            cm=(comp>0)
            duplicate=False
            for em in excluded:
                inter=int(np.count_nonzero(cm & em)); union=int(np.count_nonzero(cm | em))
                if union>0 and float(inter)/float(union)>=0.45:
                    duplicate=True; break
            if duplicate:
                continue
            comps.append((float(d),-int(stats[lid,cv2.CC_STAT_AREA]),comp))
        comps.sort(key=lambda z:(z[0],z[1]))
        for d,_,comp in comps[:max(2,int(max_candidates)*2)]:
            exact=_e145_select_exact_fold_normal(depth_state,mask,comp,np.zeros_like(comp,dtype=np.uint8))
            if not isinstance(exact,dict):
                continue
            exact['v32_target_local_source']=str(source_name)
            exact['v32_target_local_distance_mm']=float(d)
            exact['v30_target_seed_distance_mm']=float(d)
            exact['v30_target_guided_planner']=True
            cm=(np.asarray(exact.get('crease_mask',comp),dtype=np.uint8)>0)
            duplicate=False
            for old in out:
                om=(np.asarray(old.get('crease_mask',np.zeros_like(mask)),dtype=np.uint8)>0)
                inter=int(np.count_nonzero(cm & om)); union=int(np.count_nonzero(cm | om))
                if union>0 and float(inter)/float(union)>=0.45:
                    duplicate=True; break
            if duplicate:
                continue
            out.append(exact)
            if len(out)>=max(1,int(max_candidates)):
                break
        if len(out)>=max(1,int(max_candidates)):
            break
    out.sort(key=lambda q:(float(q.get('v32_target_local_distance_mm',1e9)),-float(q.get('quality',0.0))))
    out=out[:max(1,int(max_candidates))]
    print(
        f"[E156-V32-TARGET-LOCAL-SEED] target=({float(target[0]):.1f},{float(target[1]):.1f}) "
        f"radius={radius_mm:.1f}mm/{radius_px:.1f}px candidates={len(out)} "
        f"dist={[round(float(q.get('v32_target_local_distance_mm',999.0)),1) for q in out]}"
    )
    return out


def _v32_restore_suppression(depth_state, saved):
    keys=('e151_directional_hard_block_reason','e151_edge_cluster_hard_block','e151_edge_cluster_block_info','e154_edge_fallthrough_info')
    for k in keys:
        depth_state.pop(k,None)
    for k,v in saved.items():
        depth_state[k]=v


def _v32_log_stage(primary_ms, flipped_ms, target_ms, total_ms, outcome):
    print(
        f'[E156-V32-STAGE-TIME] primary={primary_ms:.1f}ms flipped={flipped_ms:.1f}ms '
        f'targetLocal={target_ms:.1f}ms total={total_ms:.1f}ms outcome={outcome}'
    )


def _e133_directional_pair(depth_state,mask_u8,H_value,args,config,cfg,fold_region_info,exec_crease,preview_crease):
    total_t0=time.perf_counter()
    primary_t0=time.perf_counter()
    pair=_e133_directional_pair_v31_primary(
        depth_state,mask_u8,H_value,args,config,cfg,fold_region_info,exec_crease,preview_crease)
    primary_ms=1000.0*(time.perf_counter()-primary_t0)
    if isinstance(pair,dict):
        pair['e156_v32_recovery_stage']='PRIMARY_V31'
        pair['e156_v32_primary_ms']=float(primary_ms)
        _v32_log_stage(primary_ms,0.0,0.0,1000.0*(time.perf_counter()-total_t0),'PRIMARY_READY')
        return pair

    if not bool(getattr(args,'micro_unfold_v32_bounded_recovery_enable',True)):
        _v32_log_stage(primary_ms,0.0,0.0,1000.0*(time.perf_counter()-total_t0),'PRIMARY_BLOCKED_RECOVERY_DISABLED')
        return None

    saved={}
    for k in ('e151_directional_hard_block_reason','e151_edge_cluster_hard_block','e151_edge_cluster_block_info','e154_edge_fallthrough_info'):
        if k in depth_state:
            saved[k]=copy.deepcopy(depth_state[k])
            depth_state.pop(k,None)

    rerank_max=max(1,min(12,int(getattr(args,'micro_unfold_vla_exact_crease_rerank_max',6))))
    flip_max=max(1,min(rerank_max,int(getattr(args,'micro_unfold_v32_flipped_max_candidates',2))))
    exact_ranked=_v11_rank_exact_fold_normals(
        depth_state,mask_u8,exec_crease,preview_crease,rerank_max,H_value=H_value,args=args)

    flipped_ms=0.0
    if bool(getattr(args,'micro_unfold_v32_flipped_normal_enable',True)) and exact_ranked:
        ft0=time.perf_counter()
        for rank,exact in enumerate(exact_ranked[:flip_max],1):
            fake=_v32_fake_from_exact(exact,group_prefix='V32_FLIP')
            pair=_v32_try_flipped_normal_pair(
                depth_state,mask_u8,H_value,args,config,cfg,fake,exec_crease,preview_crease,label=f'R{rank}')
            if isinstance(pair,dict):
                flipped_ms=1000.0*(time.perf_counter()-ft0)
                pair['e156_v32_recovery_stage']='FLIPPED_NORMAL'
                pair['e156_v32_primary_ms']=float(primary_ms)
                pair['e156_v32_flipped_ms']=float(flipped_ms)
                pair['v30_target_guided_planner']=bool(_v30_target_board(args) is not None)
                _v32_log_stage(primary_ms,flipped_ms,0.0,1000.0*(time.perf_counter()-total_t0),'FLIPPED_READY')
                return pair
        flipped_ms=1000.0*(time.perf_counter()-ft0)

    target_ms=0.0
    if bool(getattr(args,'micro_unfold_v32_target_local_enable',True)) and _v30_target_board(args) is not None:
        tt0=time.perf_counter()
        exclude=[q.get('crease_mask') for q in exact_ranked if isinstance(q,dict) and q.get('crease_mask') is not None]
        locals_=_v32_target_local_exact_normals(
            depth_state,mask_u8,H_value,args,
            max_candidates=max(1,int(getattr(args,'micro_unfold_v32_target_local_max_candidates',2))),
            exclude_masks=exclude)
        for rank,exact in enumerate(locals_,1):
            fake=_v32_fake_from_exact(exact,group_prefix='V32_TARGET_LOCAL')
            try:
                pair=_e130_directional_pair(depth_state,mask_u8,H_value,args,config,cfg,fake,exec_crease,preview_crease)
            except Exception as exc:
                print(f'[E156-V32-TARGET-LOCAL-REJECT] rank={rank} exact reason={exc!r}')
                pair=None
            if not isinstance(pair,dict):
                pair=_v13_try_same_crease_anchor_pairs(
                    depth_state,mask_u8,H_value,args,config,cfg,fake,exec_crease,preview_crease,
                    label=f'V32_TARGET_LOCAL_R{rank}',
                    max_trials=max(1,int(getattr(args,'micro_unfold_v32_target_local_anchor_max_trials',2))))
            if not isinstance(pair,dict) and bool(getattr(args,'micro_unfold_v32_flipped_normal_enable',True)):
                pair=_v32_try_flipped_normal_pair(
                    depth_state,mask_u8,H_value,args,config,cfg,fake,exec_crease,preview_crease,
                    label=f'TARGET_LOCAL_R{rank}')
            if isinstance(pair,dict):
                target_ms=1000.0*(time.perf_counter()-tt0)
                pair['e156_v32_recovery_stage']='TARGET_LOCAL'
                pair['e156_v32_primary_ms']=float(primary_ms)
                pair['e156_v32_flipped_ms']=float(flipped_ms)
                pair['e156_v32_target_local_ms']=float(target_ms)
                pair['v32_target_local_distance_mm']=exact.get('v32_target_local_distance_mm')
                pair['v32_target_local_source']=exact.get('v32_target_local_source')
                pair['v30_target_guided_planner']=True
                print(
                    f"[E156-V32-TARGET-LOCAL-ACCEPT] rank={rank} source={exact.get('v32_target_local_source')} "
                    f"dist={float(exact.get('v32_target_local_distance_mm',999.0)):.1f}mm hardSafety=PASS"
                )
                _v32_log_stage(primary_ms,flipped_ms,target_ms,1000.0*(time.perf_counter()-total_t0),'TARGET_LOCAL_READY')
                return pair
        target_ms=1000.0*(time.perf_counter()-tt0)

    _v32_restore_suppression(depth_state,saved)
    _v32_log_stage(primary_ms,flipped_ms,target_ms,1000.0*(time.perf_counter()-total_t0),'BLOCKED')
    return None


def _fix53_depth_geometry_compute(frame_bgr, mask_u8, pose_state=None, e156_rgb_precompute=None):
    """E137 compatibility entry point: RGB/Pose/crease only, ZERO UniDepth work."""
    try:
        t0=time.perf_counter(); args_micro=globals().get('_E92_MICRO_ARGS')
        frame=np.asarray(frame_bgr,dtype=np.uint8); mask=((np.asarray(mask_u8,dtype=np.uint8)>0).astype(np.uint8)*255)
        h0,w0=frame.shape[:2]
        if cv2.countNonZero(mask)<500:
            return {'available':False,'reason':'RGB_GARMENT_TOO_SMALL'}
        zeros_f=np.zeros((h0,w0),dtype=np.float32); zeros_u=np.zeros((h0,w0),dtype=np.uint8)

        pose_fold_state=_e117_pose_fold_intrusion_state(frame.shape,mask,pose_state,args=args_micro)
        if bool(pose_fold_state.get('available',False)):
            desc=','.join(str(q.get('name','?')) for q in list(pose_fold_state.get('intrusions',[]))[:6])
            print(f"[E137-POSE-FOLD] boundary-kpt intrusion count={int(pose_fold_state.get('count',0))} names={desc}")
        else:
            print('[E137-POSE-FOLD] no admitted contour intrusion; Pose deformation / RGB crease geometry will decide direction')
        pose_deform_state=_e120_pose_deformation_state(frame.shape,mask,pose_state,args=args_micro,intrusion_state=pose_fold_state)
        print(
            f"[E137-POSE-DEFORM] available={bool(pose_deform_state.get('available',False))} "
            f"mode={str(pose_deform_state.get('mode','NONE'))} conf={float(pose_deform_state.get('confidence',0.0)):.3f} "
            f"side={str(pose_deform_state.get('side_hint','NONE'))} region={str(pose_deform_state.get('fold_region_hint','NONE'))}:"
            f"{float(pose_deform_state.get('fold_region_confidence',0.0)):.2f}"
        )
        # The inherited crease detector is already overwhelmingly RGB-first
        # (LAB ridge/step/gradient); passing zero relief removes its former 8% depth term.
        crease_state=_e94_detect_fold_crease(frame,mask,zeros_f,zeros_f,args=args_micro,pose_state=pose_state,pose_deform_state=pose_deform_state,e156_precomputed=e156_rgb_precompute)
        crease_state,stable_hit,stable_mad,stable_iou,stable_pose=_e120_stabilize_crease_state(crease_state,frame,mask,pose_state,args_micro)
        print(f"[E137-CREASE-STABLE] hit={stable_hit} MAD={stable_mad:.3f} maskIoU={stable_iou:.3f} poseMove={stable_pose:.2f}px")
        if bool(crease_state.get('available',False)):
            print(
                f"[E137-RGB-CREASE] source={str(crease_state.get('selection_source','NONE'))} "
                f"STRUCT components={int(crease_state.get('structural_component_count',0))} skeleton={int(crease_state.get('structural_skeleton_area_px',0))}px | "
                f"SENSITIVE components={int(crease_state.get('sensitive_component_count',0))} skeleton={int(crease_state.get('sensitive_skeleton_area_px',0))}px"
            )
        else:
            print(f"[E137-RGB-CREASE] unavailable: {crease_state.get('reason','UNKNOWN')}")
        elapsed=time.perf_counter()-t0
        state={
            'available':True,'frame':frame.copy(),'mask':mask.copy(),'garment_mask':mask.copy(),
            'local_relief_mm':zeros_f.copy(),'prominence_mm':zeros_f.copy(),
            'labels':zeros_u.copy(),'strict_seed_labels':zeros_u.copy(),'strict_candidate_labels':zeros_u.copy(),
            'mask_fold_candidate':zeros_u.copy(),'fusion_mode':'RGB_POSE_CREASE_ONLY','candidate_count':0,
            'candidate_info':{'mode':'E137_NO_UNIDEPTH'},'candidates':[],'candidate_diagnostics':[],
            'selected_candidate_id':'RGB_CREASE','selected_donor_rank':0,'hypothesis_score':0.0,
            'hypothesis_score_gap':9.999,'hypothesis_centroid_gap_px':0.0,'hypothesis_ambiguous':False,
            'visual_top2_labels':zeros_u.copy(),'visual_top2_info':[],'visual_top2_score':np.asarray(crease_state.get('score_map',zeros_f),dtype=np.float32),
            'visual_top2_count':0,'fold_region_expansion':{},
            'fold_crease_mask':np.asarray(crease_state.get('sensitive_band_mask',crease_state.get('band_mask',zeros_u)),dtype=np.uint8),
            'fold_crease_skeleton':np.asarray(crease_state.get('sensitive_skeleton_mask',crease_state.get('skeleton_mask',zeros_u)),dtype=np.uint8),
            'fold_crease_primary_skeleton':np.asarray(crease_state.get('skeleton_mask',zeros_u),dtype=np.uint8),
            'fold_crease_structural_mask':np.asarray(crease_state.get('structural_band_mask',zeros_u),dtype=np.uint8),
            'fold_crease_structural_skeleton':np.asarray(crease_state.get('structural_skeleton_mask',zeros_u),dtype=np.uint8),
            'fold_crease_selection_source':str(crease_state.get('selection_source','NONE')),
            'fold_crease_score':np.asarray(crease_state.get('score_map',zeros_f),dtype=np.float32),
            '_e149_rgb_l_map':np.asarray(crease_state.get('_e149_rgb_l_map',zeros_f),dtype=np.float32),
            '_e149_inside_map':np.asarray(crease_state.get('_e149_inside_map',zeros_f),dtype=np.float32),
            '_e149_roi_bounds':_training_json_safe(crease_state.get('_e149_roi_bounds',[0,0,w0,h0])),
            'pose_seam_exclusion_mask':np.asarray(crease_state.get('pose_seam_exclusion_mask',zeros_u),dtype=np.uint8),
            'pose_body_core_mask':np.asarray(crease_state.get('pose_body_core_mask',zeros_u),dtype=np.uint8),
            'tshirt_pose_state':_training_json_safe(pose_state or {}),
            'pose_fold_intrusion_mask':np.asarray(pose_fold_state.get('corridor_mask',zeros_u),dtype=np.uint8),
            'pose_fold_intrusions':_training_json_safe(pose_fold_state.get('intrusions',[])),
            'pose_fold_intrusion_available':bool(pose_fold_state.get('available',False)),
            'pose_fold_intrusion_confidence':float(pose_fold_state.get('confidence',0.0)),
            'pose_fold_primary_group':str(pose_fold_state.get('primary_group','NONE')),
            'pose_fold_active_groups':_training_json_safe(pose_fold_state.get('active_groups',[])),
            'pose_fold_group_scores':_training_json_safe(pose_fold_state.get('group_scores',{})),
            'pose_fold_region_mask':np.asarray(pose_fold_state.get('fold_region_mask',zeros_u),dtype=np.uint8),
            'pose_fold_region_confidence':float(pose_fold_state.get('confidence',0.0)),
            'pose_deform_mask':np.asarray(pose_deform_state.get('prior_mask',zeros_u),dtype=np.uint8),
            'pose_deform_available':bool(pose_deform_state.get('available',False)),
            'pose_deform_confidence':float(pose_deform_state.get('confidence',0.0)),
            'pose_deform_mode':str(pose_deform_state.get('mode','NONE')),
            'pose_deform_side_hint':str(pose_deform_state.get('side_hint','NONE')),
            'pose_deform_fold_region_hint':str(pose_deform_state.get('fold_region_hint','NONE')),
            'pose_deform_fold_region_confidence':float(pose_deform_state.get('fold_region_confidence',0.0)),
            'pose_deform_fold_anchor_px':_training_json_safe(pose_deform_state.get('fold_anchor_px',None)),
            'pose_deform_fold_direction_px':_training_json_safe(pose_deform_state.get('fold_direction_px',None)),
            'pose_deform_info':_training_json_safe({k:v for k,v in pose_deform_state.items() if k!='prior_mask'}),
            'depth_cache_hit':False,'depth_feature_cache_hit':False,'depth_cache_scene_mad':0.0,'depth_cache_mask_iou':0.0,
            'fold_crease_info':_training_json_safe({k:v for k,v in crease_state.items() if k not in ('band_mask','skeleton_mask','sensitive_band_mask','sensitive_skeleton_mask','structural_band_mask','structural_skeleton_mask','score_map','pose_seam_exclusion_mask','pose_body_core_mask')}),
            'strict_top1_only':False,'top3_hypothesis_ranking':False,'fold_region_from_strict_seed':False,
            'mask_unidepth_fusion':False,'inference_sec':0.0,'source_inference_sec':0.0,'total_sec':float(elapsed),
            'model_name':'NONE_RGB_POSE_ONLY','uses_exact_d50_mask':True,'unidepth_enabled':False,'e137_rgb_only':True,
        }
        print(f"[E137-RGB-CAPTURE] garment={cv2.countNonZero(mask)}px UniDepth=OFF total={elapsed:.3f}s")
        print(f"[E149-TIME-I-RGB] total={elapsed:.3f}s roi={crease_state.get('_e149_roi_bounds',None)} heavyMaps={float(crease_state.get('_e149_roi_sec',0.0)):.3f}s")
        return state
    except Exception as exc:
        return {'available':False,'reason':'E137_RGB_POSE_RUNTIME_ERROR','error':repr(exc)}





# =============================================================================
# E156: PARALLEL POSE / RGB-HEAVY PRECOMPUTE
# The expensive OpenCV RGB maps are pose-independent. They are computed while
# TensorRT pose inference runs, then E94 applies the unchanged pose-dependent
# domain/normalization/threshold logic after the join.
# =============================================================================
def _e156_precompute_rgb_heavy(frame_bgr, garment_mask_u8):
    t0=time.perf_counter()
    frame=np.asarray(frame_bgr,dtype=np.uint8)
    garment=((np.asarray(garment_mask_u8,dtype=np.uint8)>0).astype(np.uint8)*255)
    if frame.ndim!=3 or frame.shape[:2]!=garment.shape[:2] or cv2.countNonZero(garment)<500:
        return {'available':False,'reason':'E156_PRECOMPUTE_BAD_INPUT'}
    roi=_e149_garment_roi_bounds(garment,pad=64)
    x0,y0,x1,y1=[int(v) for v in roi]
    fr=frame[y0:y1,x0:x1]
    Lr=cv2.cvtColor(fr,cv2.COLOR_BGR2LAB)[:,:,0].astype(np.float32)
    fine=cv2.GaussianBlur(Lr,(5,5),0)
    dark=np.zeros_like(Lr,dtype=np.float32); bright=np.zeros_like(Lr,dtype=np.float32)
    for k in (11,21,35,55):
        ker=_e149_cv_kernel('ellipse',k)
        dark=np.maximum(dark,cv2.morphologyEx(fine,cv2.MORPH_CLOSE,ker)-fine)
        bright=np.maximum(bright,fine-cv2.morphologyEx(fine,cv2.MORPH_OPEN,ker))
    g9=cv2.GaussianBlur(Lr,(9,9),0); g31=cv2.GaussianBlur(Lr,(31,31),0); g51=cv2.GaussianBlur(Lr,(51,51),0)
    broad_dark=np.maximum(np.maximum(g31-g9,g51-g9),0.0).astype(np.float32)
    broad_step=np.maximum(np.abs(g31-g9),np.abs(g51-g9)).astype(np.float32)
    gx=cv2.Scharr(fine,cv2.CV_32F,1,0); gy=cv2.Scharr(fine,cv2.CV_32F,0,1)
    rgb_grad=cv2.magnitude(gx,gy)
    coarse=cv2.GaussianBlur(Lr,(0,0),3.0)
    cgx=cv2.Scharr(coarse,cv2.CV_32F,1,0); cgy=cv2.Scharr(coarse,cv2.CV_32F,0,1)
    coarse_grad=cv2.magnitude(cgx,cgy)
    # E137 has no depth. Keep the exact E149 zero-depth map semantics.
    depth_grad=np.zeros_like(Lr,dtype=np.float32)
    inside=cv2.distanceTransform(garment,cv2.DIST_L2,5).astype(np.float32)
    ms=1000.0*(time.perf_counter()-t0)
    return {
        'available':True,'shape':tuple(garment.shape),'roi':tuple(roi),'Lr':Lr,'dark_r':dark,
        'bright_r':bright,'broad_dark_r':broad_dark,'broad_step_r':broad_step,
        'rgb_grad_r':rgb_grad,'coarse_grad_r':coarse_grad,'depth_grad_r':depth_grad,
        'inside_dist':inside,'zero_depth':True,'heavy_ms':float(ms),
    }


# =============================================================================
# E155: ROI/BBOX BRANCH PIPELINE OPTIMIZATION
# - Preserve E154 branch decomposition/scoring/planning semantics.
# - Run per-component junction filtering inside each connected-component bbox.
# - Carry branch coordinates/metadata forward so scorer/veto do not rescan 720p.
# - Keep full-frame branch masks only for downstream compatibility/overlay.
# =============================================================================

def _e155_branch_record(local_u8, x0, y0, full_shape):
    loc=(np.asarray(local_u8,dtype=np.uint8)>0)
    yy_l,xx_l=np.where(loc)
    yy=(yy_l.astype(np.int32)+int(y0))
    xx=(xx_l.astype(np.int32)+int(x0))
    h,w=int(full_shape[0]),int(full_shape[1])
    full=np.zeros((h,w),dtype=np.uint8)
    if xx.size:
        view=full[int(y0):int(y0)+loc.shape[0],int(x0):int(x0)+loc.shape[1]]
        view[loc]=255
        pts=np.column_stack([xx,yy]).astype(np.float64)
        bbox=(int(xx.min()),int(yy.min()),int(xx.max())+1,int(yy.max())+1)
    else:
        pts=np.empty((0,2),dtype=np.float64); bbox=(0,0,0,0)
    return {
        'mask':full,'yy':yy,'xx':xx,'pts':pts,'area':int(xx.size),'bbox':bbox,
        'bbox_area':int(max(0,bbox[2]-bbox[0])*max(0,bbox[3]-bbox[1])) if xx.size else 0,
        'scan_key':int(np.min(yy.astype(np.int64)*int(w)+xx.astype(np.int64))) if xx.size else (1<<62),
    }


def _e122_split_crease_branches(skeleton_u8, min_pixels=10, return_meta=False):
    """E155 exact-semantics E123 splitter with per-component bbox processing.

    Connectivity, degree>=4 junction rule, 12px span rule and 0.55 coverage rule
    are unchanged.  Only the work domain changes from a 1280x720 temporary mask
    per component to that component's tight bounding box.
    """
    sk=((np.asarray(skeleton_u8,dtype=np.uint8)>0).astype(np.uint8))
    min_pixels=max(6,int(min_pixels))
    if int(np.count_nonzero(sk))<min_pixels:
        return []
    h,w=sk.shape[:2]
    n0,labels0,stats0,_=cv2.connectedComponentsWithStats(sk,8)
    records=[]
    k3=_e149_cv_kernel('ones',3)
    for lab0 in range(1,n0):
        comp_area=int(stats0[lab0,cv2.CC_STAT_AREA])
        if comp_area<min_pixels:
            continue
        x0=int(stats0[lab0,cv2.CC_STAT_LEFT]); y0=int(stats0[lab0,cv2.CC_STAT_TOP])
        bw=int(stats0[lab0,cv2.CC_STAT_WIDTH]); bh=int(stats0[lab0,cv2.CC_STAT_HEIGHT])
        if bw<=0 or bh<=0:
            continue
        comp=(labels0[y0:y0+bh,x0:x0+bw]==lab0).astype(np.uint8)
        yy0,xx0=np.where(comp>0)
        if xx0.size<min_pixels:
            continue
        comp_span=float(np.hypot(float(xx0.max()-xx0.min()),float(yy0.max()-yy0.min())))
        if comp_span<12.0:
            continue
        degree=(cv2.filter2D(comp,cv2.CV_16S,k3,borderType=cv2.BORDER_CONSTANT)-comp.astype(np.int16))
        junction=((comp>0)&(degree>=4))
        if int(np.count_nonzero(junction))==0:
            records.append(_e155_branch_record(comp,x0,y0,(h,w)))
            continue
        cut=comp.copy(); cut[junction]=0
        n1,labels1,stats1,_=cv2.connectedComponentsWithStats(cut,8)
        children=[]; kept_pixels=0
        for lab1 in range(1,n1):
            area=int(stats1[lab1,cv2.CC_STAT_AREA])
            if area<min_pixels:
                continue
            cx0=int(stats1[lab1,cv2.CC_STAT_LEFT]); cy0=int(stats1[lab1,cv2.CC_STAT_TOP])
            cw=int(stats1[lab1,cv2.CC_STAT_WIDTH]); ch=int(stats1[lab1,cv2.CC_STAT_HEIGHT])
            child=(labels1[cy0:cy0+ch,cx0:cx0+cw]==lab1).astype(np.uint8)
            yy,xx=np.where(child>0)
            if xx.size<min_pixels:
                continue
            span=float(np.hypot(float(xx.max()-xx.min()),float(yy.max()-yy.min())))
            if span<12.0:
                continue
            children.append(_e155_branch_record(child,x0+cx0,y0+cy0,(h,w)))
            kept_pixels+=area
        coverage=float(kept_pixels)/float(max(1,comp_area))
        if children and coverage>=0.55:
            children.sort(key=lambda r:int(r.get('scan_key',1<<62)))
            records.extend(children)
        else:
            records.append(_e155_branch_record(comp,x0,y0,(h,w)))
    if return_meta:
        return records
    return [r['mask'] for r in records]


def _e155_branch_contour_distance_stats(branch_meta, branch_u8, garment_u8, H_value, threshold_mm=10.0, inside_px=None):
    """E144 contour statistic using already-known branch coordinates."""
    meta=branch_meta if isinstance(branch_meta,dict) else {}
    yy=np.asarray(meta.get('yy',np.empty((0,),dtype=np.int32)),dtype=np.int32)
    xx=np.asarray(meta.get('xx',np.empty((0,),dtype=np.int32)),dtype=np.int32)
    garment=((np.asarray(garment_u8,dtype=np.uint8)>0).astype(np.uint8)*255)
    if xx.size<3 or yy.size!=xx.size:
        return _e144_branch_contour_distance_stats(branch_u8,garment,H_value,threshold_mm=threshold_mm,inside_px=inside_px)
    inside=np.asarray(inside_px,dtype=np.float32) if inside_px is not None else np.empty((0,0),dtype=np.float32)
    if inside.shape!=garment.shape:
        inside=_e144_outer_contour_inside_map(garment)
    ctr=np.asarray([float(np.mean(xx)),float(np.mean(yy))],dtype=np.float64)
    try:
        mm_per_px=float(_e98_local_mm_per_px(H_value,ctr))
    except Exception:
        mm_per_px=float('nan')
    if not np.isfinite(mm_per_px) or mm_per_px<=1e-6:
        return {'available':False,'median_mm':999.0,'p75_mm':999.0,'near_ratio':0.0,'mm_per_px':None}
    d_mm=np.asarray(inside[yy,xx],dtype=np.float64)*mm_per_px
    d_mm=d_mm[np.isfinite(d_mm)]
    if d_mm.size<3:
        return {'available':False,'median_mm':999.0,'p75_mm':999.0,'near_ratio':0.0,'mm_per_px':mm_per_px}
    th=max(0.0,float(threshold_mm))
    return {
        'available':True,'median_mm':float(np.median(d_mm)),'p75_mm':float(np.percentile(d_mm,75.0)),
        'near_ratio':float(np.mean(d_mm<=th)) if th>0.0 else 0.0,'mm_per_px':float(mm_per_px),
    }


def _e122_score_fold_branch(branch_u8, frame_bgr, relief_mm, garment_u8, raw_score_map,
                            structural_u8, pose_deform_u8, pose_deform_conf,
                            seam_u8, shared_ctx=None, branch_meta=None):
    """E155: E149 scoring equation unchanged; branch coordinates are reused."""
    frame=np.asarray(frame_bgr,dtype=np.uint8)
    ctx=shared_ctx if isinstance(shared_ctx,dict) else {}
    if 'garment' in ctx:
        garment=np.asarray(ctx['garment'],dtype=bool)
    else:
        garment=np.asarray(garment_u8,dtype=np.uint8)>0
    h,w=garment.shape[:2]
    if frame.ndim!=3 or frame.shape[:2]!=(h,w):
        return {'fold_score':0.0,'valid_samples':0,'reason':'RGB_SHAPE_MISMATCH'}
    meta=branch_meta if isinstance(branch_meta,dict) else {}
    yy=np.asarray(meta.get('yy',np.empty((0,),dtype=np.int32)),dtype=np.int32)
    xx=np.asarray(meta.get('xx',np.empty((0,),dtype=np.int32)),dtype=np.int32)
    pts=np.asarray(meta.get('pts',np.empty((0,2),dtype=np.float64)),dtype=np.float64)
    if pts.ndim!=2 or pts.shape[1:]!=(2,) or len(pts)!=len(xx) or len(xx)<1:
        branch=np.asarray(branch_u8,dtype=np.uint8)>0
        yy,xx=np.where(branch)
        pts=np.column_stack([xx,yy]).astype(np.float64)
    else:
        branch=None
    if len(pts)<8:
        return {'fold_score':0.0,'valid_samples':0,'reason':'BRANCH_TOO_SHORT'}
    L=np.asarray(ctx.get('L',np.empty((0,0),dtype=np.float32)),dtype=np.float32)
    inside=np.asarray(ctx.get('inside',np.empty((0,0),dtype=np.float32)),dtype=np.float32)
    if L.shape!=(h,w):
        L=cv2.cvtColor(frame,cv2.COLOR_BGR2LAB)[:,:,0].astype(np.float32)
    if inside.shape!=(h,w):
        inside=cv2.distanceTransform((garment.astype(np.uint8)*255),cv2.DIST_L2,5).astype(np.float32)
    ctr=pts.mean(axis=0)
    X=pts-ctr[None,:]
    cov=X.T@X
    vals,vecs=np.linalg.eigh(cov)
    tangent_global=vecs[:,int(np.argmax(vals))]
    tangent_global/=max(1e-9,float(np.linalg.norm(tangent_global)))
    proj=X@tangent_global
    span=float(np.max(proj)-np.min(proj))
    linearity=float(np.clip(1.0-float(vals[0])/max(1e-6,float(vals[-1])),0.0,1.0))
    order=np.argsort(proj); ordered=pts[order]
    sample_count=int(np.clip(round(span/18.0)+3,7,17))
    sample_indices=np.unique(np.rint(np.linspace(0,len(ordered)-1,sample_count)).astype(np.int32))
    rgb_deltas=[]; rgb_signs=[]; sample_interiors=[]
    offsets=(10.0,16.0,22.0)
    for si in sample_indices.tolist():
        p=np.asarray(ordered[int(si)],dtype=np.float64)
        d2=np.sum((pts-p[None,:])**2,axis=1)
        neigh=pts[np.argsort(d2)[:min(19,len(pts))]]
        if len(neigh)>=5:
            c=neigh.mean(axis=0); Y=neigh-c[None,:]
            vv,ee=np.linalg.eigh(Y.T@Y)
            tangent=ee[:,int(np.argmax(vv))]
            tangent/=max(1e-9,float(np.linalg.norm(tangent)))
        else:
            tangent=tangent_global.copy()
        normal=np.asarray([-tangent[1],tangent[0]],dtype=np.float64)
        per_rgb=[]; per_sign=[]
        for off in offsets:
            pa=p+normal*off; pb=p-normal*off
            ax,ay=int(round(pa[0])),int(round(pa[1])); bx,by=int(round(pb[0])),int(round(pb[1]))
            if not (3<=ax<w-3 and 3<=ay<h-3 and 3<=bx<w-3 and 3<=by<h-3):
                continue
            if inside[ay,ax]<3.0 or inside[by,bx]<3.0:
                continue
            la=_e122_patch_median(L,garment,pa[0],pa[1],3)
            lb=_e122_patch_median(L,garment,pb[0],pb[1],3)
            if la is None or lb is None:
                continue
            per_rgb.append(abs(float(la-lb))); per_sign.append(float(la-lb))
        if per_rgb:
            rgb_deltas.append(float(np.median(per_rgb))); rgb_signs.append(float(np.median(per_sign)))
            xi=int(np.clip(round(p[0]),0,w-1)); yi=int(np.clip(round(p[1]),0,h-1))
            sample_interiors.append(float(inside[yi,xi]))
    nvalid=len(rgb_deltas)
    valid_ratio=float(nvalid)/float(max(1,len(sample_indices)))
    rgb_step=float(np.median(rgb_deltas)) if rgb_deltas else 0.0
    strong_sign=np.asarray([v for v in rgb_signs if abs(float(v))>=2.0],dtype=np.float64)
    if strong_sign.size>=3:
        pos=float(np.mean(strong_sign>0)); neg=float(np.mean(strong_sign<0)); side_consistency=max(pos,neg)
    else:
        side_consistency=0.50
    side_score=float(np.clip((side_consistency-0.50)/0.35,0.0,1.0))
    raw_score=np.asarray(raw_score_map,dtype=np.float32)
    rv=raw_score[yy,xx]
    rv=rv[np.isfinite(rv)]
    raw_strength=float(np.mean(rv)) if rv.size else 0.0
    struct_near=np.asarray(ctx.get('struct_near',np.empty((0,0),dtype=bool)),dtype=bool)
    if struct_near.shape==(h,w):
        structural_support=float(np.mean(struct_near[yy,xx])) if (bool(ctx['_has_struct_near']) if '_has_struct_near' in ctx else bool(np.any(struct_near))) else 0.0
    else:
        if branch is None: branch=np.asarray(branch_u8,dtype=np.uint8)>0
        struct=np.asarray(structural_u8,dtype=np.uint8)>0
        structural_support=float(np.mean((cv2.dilate((struct.astype(np.uint8)*255),_e149_cv_kernel('ones',5),iterations=1)>0)[yy,xx])) if np.any(struct) else 0.0
    deform_near=np.asarray(ctx.get('deform_near',np.empty((0,0),dtype=bool)),dtype=bool)
    if deform_near.shape==(h,w):
        deform_support=float(np.mean(deform_near[yy,xx])) if (bool(ctx['_has_deform_near']) if '_has_deform_near' in ctx else bool(np.any(deform_near))) else 0.0
    else:
        deform=np.asarray(pose_deform_u8,dtype=np.uint8)>0
        deform_support=float(np.mean((cv2.dilate((deform.astype(np.uint8)*255),_e149_cv_kernel('ellipse',25),iterations=1)>0)[yy,xx])) if np.any(deform) else 0.0
    seam=np.asarray(ctx.get('seam',np.asarray(seam_u8,dtype=np.uint8)>0),dtype=bool)
    seam_overlap=float(np.mean(seam[yy,xx])) if (bool(ctx['_has_seam']) if '_has_seam' in ctx else bool(np.any(seam))) else 0.0
    interior_med=float(np.median(sample_interiors)) if sample_interiors else 0.0
    rgb_score=float(np.tanh(max(0.0,rgb_step)/14.0))
    appearance=float(np.clip(0.62*rgb_score+0.38*raw_strength,0.0,1.0))
    length_score=float(np.clip(span/150.0,0.0,1.0))
    interior_score=float(np.clip(interior_med/18.0,0.0,1.0))
    structure=float(np.clip(0.38*length_score+0.28*linearity+0.20*interior_score+0.14*structural_support,0.0,1.0))
    pose_term=float(np.clip(0.30+0.70*deform_support*float(np.clip(pose_deform_conf,0.0,1.0)),0.0,1.0))
    noise_penalty=float(0.08*(1.0-valid_ratio)); seam_penalty=float(0.42*seam_overlap)
    fold_score=float(np.clip(0.36*appearance+0.36*structure+0.18*pose_term+0.10*side_score-seam_penalty-noise_penalty,0.0,1.0))
    return {
        'fold_score':fold_score,'geometry_score':structure,'appearance_score':appearance,'structure_score':structure,
        'pose_score':pose_term,'depth_step_mm':0.0,'depth_step_score':0.0,
        'surface_normal_jump':0.0,'surface_normal_score':0.0,'rgb_bilateral_step':rgb_step,
        'rgb_bilateral_score':rgb_score,'side_consistency':float(side_consistency),'side_consistency_score':float(side_score),
        'unidepth_fold_support':0.0,'raw_crease_strength':raw_strength,
        'structural_support':structural_support,'pose_deform_support':deform_support,'seam_overlap':seam_overlap,
        'seam_penalty':seam_penalty,'noise_penalty':noise_penalty,'valid_sample_ratio':valid_ratio,'valid_samples':int(nvalid),
        'span_px':span,'linearity':linearity,'interior_median_px':interior_med,
        'centroid_px':[float(ctr[0]),float(ctr[1])],
        'rgb_execution_support':float(np.clip(0.50*rgb_score+0.30*raw_strength+0.20*side_score,0.0,1.0)),
    }


def _e122_build_scored_crease_pool(depth_state, mask_u8, args, H_value=None):
    """E155: E154/E153 semantics with bbox splitter + metadata reuse."""
    t0=time.perf_counter()
    mask=((np.asarray(mask_u8,dtype=np.uint8)>0).astype(np.uint8)*255)
    sensitive=((np.asarray(depth_state.get('fold_crease_skeleton',np.zeros_like(mask)),dtype=np.uint8)>0).astype(np.uint8)*255)
    structural=((np.asarray(depth_state.get('fold_crease_structural_skeleton',np.zeros_like(mask)),dtype=np.uint8)>0).astype(np.uint8)*255)
    frame=depth_state.get('frame')
    raw_score=np.asarray(depth_state.get('fold_crease_score',np.zeros(mask.shape,dtype=np.float32)),dtype=np.float32)
    pose_deform=np.asarray(depth_state.get('pose_deform_mask',np.zeros_like(mask)),dtype=np.uint8)
    pose_conf=float(depth_state.get('pose_deform_confidence',0.0) or 0.0)
    seam=np.asarray(depth_state.get('pose_seam_exclusion_mask',np.zeros_like(mask)),dtype=np.uint8)
    preview_threshold=float(getattr(args,'micro_unfold_e122_preview_score',0.34))
    exec_threshold=float(getattr(args,'micro_unfold_e122_exec_score',0.48))
    rescue_threshold=float(getattr(args,'micro_unfold_e123_structural_rescue_score',0.34))
    rescue_min_span=float(getattr(args,'micro_unfold_e123_structural_rescue_min_span_px',34.0))
    min_pixels=max(6,int(getattr(args,'micro_unfold_e122_branch_min_px',8)))
    sig=_e153_e122_cache_signature(depth_state,mask,args,H_value)
    cached=depth_state.get('_e153_e122_pool_cache')
    if isinstance(cached,dict) and cached.get('signature')==sig and isinstance(cached.get('pool'),dict):
        pool=cached['pool']
        print(f"[E153-E122-CACHE-HIT] branches={int(pool.get('raw_branch_count',0))} diagnostic/planner reuse=ON recompute=0")
        return pool
    print('[E153-E122-CACHE-MISS] fresh I -> computing Fold-branch pool once')
    contour_veto_mm=10.0; contour_veto_ratio=0.65
    empty={
        'available':False,'raw_skeleton':np.zeros_like(mask),'preview_skeleton':np.zeros_like(mask),'execution_skeleton':np.zeros_like(mask),
        'rejected_skeleton':np.zeros_like(mask),'branch_score_map':np.zeros(mask.shape,dtype=np.float32),
        'branch_unidepth_map':np.zeros(mask.shape,dtype=np.float32),'branches':[],
        'preview_threshold':preview_threshold,'execution_threshold':exec_threshold,'rescue_threshold':rescue_threshold,
        'structural_branch_count':0,'sensitive_branch_count':0,'rescue_branch_count':0,
        'contour_filtered_raw_skeleton':np.zeros_like(mask),'contour_veto_rejected_skeleton':np.zeros_like(mask),
        'contour_veto_count':0,'contour_veto_mm':contour_veto_mm,'contour_veto_ratio':contour_veto_ratio,
    }
    if frame is None or (cv2.countNonZero(structural)<min_pixels and cv2.countNonZero(sensitive)<min_pixels):
        return empty
    tsplit=time.perf_counter()
    srec=_e122_split_crease_branches(structural,min_pixels=min_pixels,return_meta=True) if cv2.countNonZero(structural)>=min_pixels else []
    qrec=_e122_split_crease_branches(sensitive,min_pixels=min_pixels,return_meta=True) if cv2.countNonZero(sensitive)>=min_pixels else []
    split_ms=1000.0*(time.perf_counter()-tsplit)
    structural_union=np.zeros_like(mask)
    for r in srec:
        if r['xx'].size: structural_union[r['yy'],r['xx']]=255
    structural_near=(cv2.dilate(structural_union,_e149_cv_kernel('ones',7),iterations=1)>0) if cv2.countNonZero(structural_union)>0 else np.zeros(mask.shape,dtype=bool)
    novel=[]
    for r in qrec:
        area=max(1,int(r['area']))
        overlap=float(np.count_nonzero(structural_near[r['yy'],r['xx']]))/float(area)
        if overlap<0.62: novel.append(r)
    sources=[('STRUCTURAL',r) for r in srec]+[('SENSITIVE',r) for r in novel]
    if not sources:
        return empty
    shared_ctx=_e149_build_branch_shared_ctx(frame,mask,structural,pose_deform,seam,base_rgb={'L':depth_state.get('_e149_rgb_l_map'),'inside':depth_state.get('_e149_inside_map')})
    if isinstance(shared_ctx,dict):
        sn=np.asarray(shared_ctx.get('struct_near',np.empty((0,0),dtype=bool)),dtype=bool)
        dn=np.asarray(shared_ctx.get('deform_near',np.empty((0,0),dtype=bool)),dtype=bool)
        sm=np.asarray(shared_ctx.get('seam',np.empty((0,0),dtype=bool)),dtype=bool)
        shared_ctx['_has_struct_near']=bool(np.any(sn)) if sn.shape==mask.shape else False
        shared_ctx['_has_deform_near']=bool(np.any(dn)) if dn.shape==mask.shape else False
        shared_ctx['_has_seam']=bool(np.any(sm)) if sm.shape==mask.shape else False
    shared_ms=1000.0*(time.perf_counter()-t0)
    contour_inside=np.asarray(depth_state.get('_e153_contour_inside_map',np.empty((0,0),dtype=np.float32)),dtype=np.float32)
    if contour_inside.shape!=mask.shape:
        contour_inside=_e144_outer_contour_inside_map(mask); depth_state['_e153_contour_inside_map']=contour_inside
        print('[E153-CONTOUR-MAP] MISS -> external-contour distance map computed once')
    else:
        print('[E153-CONTOUR-MAP] HIT -> shared with same-I consumers')
    scored=[]; raw_out=np.zeros_like(mask); preview=np.zeros_like(mask); execution=np.zeros_like(mask); rejected=np.zeros_like(mask)
    contour_filtered_raw=np.zeros_like(mask); contour_veto_rejected=np.zeros_like(mask); contour_veto_count=0
    score_map=np.zeros(mask.shape,dtype=np.float32); zero_map=np.zeros(mask.shape,dtype=np.float32); rescue_count=0
    tscore=time.perf_counter()
    for idx,(source,r) in enumerate(sources,1):
        b=r['mask']; yy=r['yy']; xx=r['xx']
        contour_stat=_e155_branch_contour_distance_stats(r,b,mask,H_value,threshold_mm=contour_veto_mm,inside_px=contour_inside)
        contour_near=bool(contour_stat.get('available',False) and float(contour_stat.get('median_mm',999.0))<=contour_veto_mm and float(contour_stat.get('near_ratio',0.0))>=contour_veto_ratio)
        # V11: score the real branch before deciding whether a near-contour line is
        # merely the silhouette or an actual folded flap boundary.
        raw_metrics=_e122_score_fold_branch(b,frame,zero_map,mask,raw_score,structural,pose_deform,pose_conf,seam,shared_ctx=shared_ctx,branch_meta=r)
        _raw_fold=float(raw_metrics.get('fold_score',0.0) or 0.0); _span=float(raw_metrics.get('span_px',0.0) or 0.0)
        _valid=float(raw_metrics.get('valid_sample_ratio',0.0) or 0.0); _seam=float(raw_metrics.get('seam_overlap',0.0) or 0.0)
        _rgb_step=float(raw_metrics.get('rgb_bilateral_step',0.0) or 0.0); _raw_strength=float(raw_metrics.get('raw_crease_strength',0.0) or 0.0)
        _side=float(raw_metrics.get('side_consistency',0.5) or 0.5); _deform=float(raw_metrics.get('pose_deform_support',0.0) or 0.0)
        _rgb_exec=bool(_rgb_step>=3.0 or _raw_strength>=0.28 or _deform*pose_conf>=0.16)
        edge_flap_rescue=bool(
            bool(getattr(args,'micro_unfold_vla_edge_flap_rescue_enable',True)) and contour_near and source=='STRUCTURAL'
            and _seam<0.18 and _span>=max(18.0,0.65*rescue_min_span) and _raw_fold>=0.90*rescue_threshold
            and _valid>=0.25 and _side>=0.55 and _rgb_exec and (_deform*pose_conf)>=0.10
        )
        contour_veto=bool(contour_near and not edge_flap_rescue)
        if contour_veto:
            contour_veto_count+=1; contour_veto_rejected[yy,xx]=255
            metrics={
                'fold_score':0.0,'geometry_score':0.0,'appearance_score':0.0,'structure_score':0.0,'pose_score':0.0,
                'depth_step_mm':0.0,'depth_step_score':0.0,'surface_normal_jump':0.0,'surface_normal_score':0.0,
                'rgb_bilateral_step':0.0,'rgb_bilateral_score':0.0,'side_consistency':0.5,'side_consistency_score':0.0,
                'unidepth_fold_support':0.0,'raw_crease_strength':0.0,'structural_support':0.0,'pose_deform_support':0.0,
                'seam_overlap':0.0,'seam_penalty':0.0,'noise_penalty':0.0,'valid_sample_ratio':0.0,'valid_samples':0,
                'span_px':0.0,'linearity':0.0,'interior_median_px':0.0,
                'centroid_px':[float(np.mean(xx)) if xx.size else 0.0,float(np.mean(yy)) if yy.size else 0.0],
                'rgb_execution_support':0.0,
            }
            print(f"[E144-CONTOUR-CREASE-VETO] branch={idx} src={source} median={float(contour_stat.get('median_mm',999.0)):.1f}mm p75={float(contour_stat.get('p75_mm',999.0)):.1f}mm within10.0mm={100.0*float(contour_stat.get('near_ratio',0.0)):.0f}% gate=65% -> IGNORE")
        else:
            contour_filtered_raw[yy,xx]=255; metrics=raw_metrics
            if edge_flap_rescue:
                print(f"[E156-V11-EDGE-FLAP-RESCUE] branch={idx} fold={_raw_fold:.3f} span={_span:.1f}px valid={_valid:.2f} rgbStep={_rgb_step:.1f} side={_side:.2f} deform={_deform:.2f}x{pose_conf:.2f} -> KEEP")
        raw_fold=float(metrics.get('fold_score',0.0)); source_bonus=(0.055 if source=='STRUCTURAL' else 0.0) if not contour_veto else 0.0
        score=float(np.clip(raw_fold+source_bonus,0.0,1.0))
        span=float(metrics.get('span_px',0.0)); valid_ratio=float(metrics.get('valid_sample_ratio',0.0)); seam_overlap=float(metrics.get('seam_overlap',0.0)); interior=float(metrics.get('interior_median_px',0.0)); rgb_step=float(metrics.get('rgb_bilateral_step',0.0)); raw_strength=float(metrics.get('raw_crease_strength',0.0)); deform_support=float(metrics.get('pose_deform_support',0.0))
        rgb_exec_ok=bool(rgb_step>=3.0 or raw_strength>=0.28 or deform_support*pose_conf>=0.16)
        structural_rescue=bool((not contour_veto) and source=='STRUCTURAL' and (score>=rescue_threshold or edge_flap_rescue) and (span>=rescue_min_span or edge_flap_rescue) and valid_ratio>=0.25 and seam_overlap<0.18 and (interior>=3.0 or edge_flap_rescue) and rgb_exec_ok)
        preview_ok=bool((score>=preview_threshold or structural_rescue) and not contour_veto)
        execution_ok=bool(((score>=exec_threshold and rgb_exec_ok) or structural_rescue) and not contour_veto)
        if structural_rescue and score<exec_threshold: rescue_count+=1
        item={
            'branch_id':int(idx),'source':source,'fold_score_raw':raw_fold,'fold_score':score,
            **{k:v for k,v in metrics.items() if k!='fold_score'},'source_bonus':float(source_bonus),
            'structural_rescue':bool(structural_rescue),'depth_execution_ok':True,'rgb_execution_ok':bool(rgb_exec_ok),
            'unidepth_fold_support':0.0,'contour_distance_median_mm':float(contour_stat.get('median_mm',999.0)),
            'contour_distance_p75_mm':float(contour_stat.get('p75_mm',999.0)),'contour_within_veto_ratio':float(contour_stat.get('near_ratio',0.0)),
            'contour_near_veto':bool(contour_veto),'contour_near_raw':bool(contour_near),'edge_flap_rescue':bool(edge_flap_rescue),'preview_ok':preview_ok,'execution_ok':execution_ok,
            'mask':b,
        }
        scored.append(item)
        raw_out[yy,xx]=255; score_map[yy,xx]=np.maximum(score_map[yy,xx],score)
        if preview_ok: preview[yy,xx]=255
        else: rejected[yy,xx]=255
        if execution_ok: execution[yy,xx]=255
    score_ms=1000.0*(time.perf_counter()-tscore)
    scored.sort(key=lambda q:(bool(q.get('execution_ok',False)),float(q.get('fold_score',0.0)),float(q.get('span_px',0.0))),reverse=True)
    info=[{k:v for k,v in q.items() if k!='mask'} for q in scored[:24]]
    print(f"[E144-CONTOUR-VETO-SUMMARY] rejectedBranches={contour_veto_count}/{len(sources)} threshold=10.0mm ratioGate=0.65")
    total_ms=1000.0*(time.perf_counter()-t0)
    local_px=sum(int(r.get('bbox_area',0)) for _,r in sources); full_equiv=max(1,len(sources)*mask.shape[0]*mask.shape[1])
    print(f"[E155-BRANCH-ROI] branches={len(sources)} split={split_ms:.1f}ms sharedPrep={shared_ms:.1f}ms scoreLoop={score_ms:.1f}ms localBBoxPixels={local_px} fullMaskEquivalent={full_equiv} ratio={100.0*local_px/full_equiv:.2f}%")
    print(f"[E149-TIME-BRANCH] branches={len(sources)} sharedPrep={shared_ms:.1f}ms total={total_ms:.1f}ms invariantMaps=ON bboxMeta=ON")
    globals()['_E156_LAST_BRANCH_MS']=float(total_ms)
    result={
        'available':bool(cv2.countNonZero(preview)>=8),'raw_skeleton':raw_out,'preview_skeleton':preview,'execution_skeleton':execution,
        'rejected_skeleton':rejected,'branch_score_map':score_map,'branch_unidepth_map':zero_map,'branches':scored,'branch_info':info,
        'preview_threshold':preview_threshold,'execution_threshold':exec_threshold,'rescue_threshold':rescue_threshold,
        'raw_branch_count':int(len(scored)),'preview_branch_count':int(sum(bool(q['preview_ok']) for q in scored)),
        'execution_branch_count':int(sum(bool(q['execution_ok']) for q in scored)),'structural_branch_count':int(len(srec)),
        'sensitive_branch_count':int(len(novel)),'rescue_branch_count':int(rescue_count),'e137_rgb_only':True,
        'contour_filtered_raw_skeleton':contour_filtered_raw,'contour_veto_rejected_skeleton':contour_veto_rejected,
        'contour_veto_count':int(contour_veto_count),'contour_veto_mm':contour_veto_mm,'contour_veto_ratio':contour_veto_ratio,
        'e144_contour_veto':True,'e155_bbox_branch_pipeline':True,'e155_split_ms':float(split_ms),'e155_score_ms':float(score_ms),'e155_total_ms':float(total_ms),
    }
    depth_state['_e153_e122_pool_cache']={'signature':sig,'pool':result}
    print(f"[E153-E122-CACHE-STORE] branches={len(scored)} poolMs={total_ms:.1f} next same-I call will be O(1)")
    return result


# =============================================================================
# E156: EXACT BATCH BILATERAL RGB SAMPLER
# The E155 scoring equation and sample locations are unchanged. Nearest-neighbor
# distance work and 7x7 disk patch medians are batched with NumPy.
# =============================================================================
_e155_score_fold_branch_reference = _e122_score_fold_branch

def _e156_patch_median_batch_exact(arr, valid_mask, centers_xy, radius=3):
    arr=np.asarray(arr)
    valid=np.asarray(valid_mask,dtype=bool)
    c=np.asarray(centers_xy,dtype=np.float64).reshape(-1,2)
    n=int(c.shape[0]); out=np.full((n,),np.nan,dtype=np.float64)
    if n<=0: return out
    r=int(radius); h,w=arr.shape[:2]
    xr=np.rint(c[:,0]).astype(np.int32); yr=np.rint(c[:,1]).astype(np.int32)
    inb=(xr-r>=0)&(xr+r<w)&(yr-r>=0)&(yr+r<h)
    ids=np.flatnonzero(inb)
    if ids.size<=0: return out
    cc=c[ids]; xri=xr[ids]; yri=yr[ids]
    offs=np.arange(-r,r+1,dtype=np.int32)
    dx,dy=np.meshgrid(offs,offs,indexing='xy')
    xs=xri[:,None,None]+dx[None,:,:]; ys=yri[:,None,None]+dy[None,:,:]
    vals=np.asarray(arr[ys,xs],dtype=np.float64)
    disk=((xs.astype(np.float64)-cc[:,0,None,None])**2+(ys.astype(np.float64)-cc[:,1,None,None])**2)<=float(r*r)
    vm=valid[ys,xs]&disk&np.isfinite(vals)
    counts=np.sum(vm,axis=(1,2))
    flat=vals.reshape(len(ids),-1); keep=vm.reshape(len(ids),-1)
    work=np.where(keep,flat,np.nan)
    nofinite=counts==0
    if np.any(nofinite): work[nofinite,0]=0.0
    med=np.nanmedian(work,axis=1)
    med[counts<4]=np.nan
    out[ids]=med
    return out


def _e122_score_fold_branch(branch_u8, frame_bgr, relief_mm, garment_u8, raw_score_map,
                            structural_u8, pose_deform_u8, pose_deform_conf,
                            seam_u8, shared_ctx=None, branch_meta=None):
    """E156 vectorized implementation of the exact E155 Fold-branch score."""
    frame=np.asarray(frame_bgr,dtype=np.uint8)
    ctx=shared_ctx if isinstance(shared_ctx,dict) else {}
    garment=np.asarray(ctx['garment'],dtype=bool) if 'garment' in ctx else (np.asarray(garment_u8,dtype=np.uint8)>0)
    h,w=garment.shape[:2]
    if frame.ndim!=3 or frame.shape[:2]!=(h,w):
        return {'fold_score':0.0,'valid_samples':0,'reason':'RGB_SHAPE_MISMATCH'}
    meta=branch_meta if isinstance(branch_meta,dict) else {}
    yy=np.asarray(meta.get('yy',np.empty((0,),dtype=np.int32)),dtype=np.int32)
    xx=np.asarray(meta.get('xx',np.empty((0,),dtype=np.int32)),dtype=np.int32)
    pts=np.asarray(meta.get('pts',np.empty((0,2),dtype=np.float64)),dtype=np.float64)
    branch=None
    if pts.ndim!=2 or pts.shape[1:]!=(2,) or len(pts)!=len(xx) or len(xx)<1:
        branch=np.asarray(branch_u8,dtype=np.uint8)>0; yy,xx=np.where(branch); pts=np.column_stack([xx,yy]).astype(np.float64)
    if len(pts)<8: return {'fold_score':0.0,'valid_samples':0,'reason':'BRANCH_TOO_SHORT'}
    L=np.asarray(ctx.get('L',np.empty((0,0),dtype=np.float32)),dtype=np.float32)
    inside=np.asarray(ctx.get('inside',np.empty((0,0),dtype=np.float32)),dtype=np.float32)
    if L.shape!=(h,w): L=cv2.cvtColor(frame,cv2.COLOR_BGR2LAB)[:,:,0].astype(np.float32)
    if inside.shape!=(h,w): inside=cv2.distanceTransform((garment.astype(np.uint8)*255),cv2.DIST_L2,5).astype(np.float32)
    ctr=pts.mean(axis=0); X=pts-ctr[None,:]; cov=X.T@X
    vals,vecs=np.linalg.eigh(cov); tangent_global=vecs[:,int(np.argmax(vals))]; tangent_global/=max(1e-9,float(np.linalg.norm(tangent_global)))
    proj=X@tangent_global; span=float(np.max(proj)-np.min(proj)); linearity=float(np.clip(1.0-float(vals[0])/max(1e-6,float(vals[-1])),0.0,1.0))
    order=np.argsort(proj); ordered=pts[order]
    sample_count=int(np.clip(round(span/18.0)+3,7,17)); sample_indices=np.unique(np.rint(np.linspace(0,len(ordered)-1,sample_count)).astype(np.int32))
    samples=np.asarray(ordered[sample_indices],dtype=np.float64); ns=int(len(samples)); k=min(19,len(pts))
    # Same np.argsort nearest-neighbor rule as E155, but all sample rows at once.
    d2=np.sum((samples[:,None,:]-pts[None,:,:])**2,axis=2); nn=np.argsort(d2,axis=1)[:,:k]
    normals=np.empty((ns,2),dtype=np.float64)
    for i in range(ns):
        neigh=pts[nn[i]]
        if len(neigh)>=5:
            c=neigh.mean(axis=0); Y=neigh-c[None,:]; vv,ee=np.linalg.eigh(Y.T@Y); tangent=ee[:,int(np.argmax(vv))]; tangent/=max(1e-9,float(np.linalg.norm(tangent)))
        else: tangent=tangent_global.copy()
        normals[i]=(-tangent[1],tangent[0])
    offs=np.asarray((10.0,16.0,22.0),dtype=np.float64)
    pa=samples[:,None,:]+normals[:,None,:]*offs[None,:,None]; pb=samples[:,None,:]-normals[:,None,:]*offs[None,:,None]
    pa2=pa.reshape(-1,2); pb2=pb.reshape(-1,2)
    ax=np.rint(pa2[:,0]).astype(np.int32); ay=np.rint(pa2[:,1]).astype(np.int32); bx=np.rint(pb2[:,0]).astype(np.int32); by=np.rint(pb2[:,1]).astype(np.int32)
    center_ok=(ax>=3)&(ax<w-3)&(ay>=3)&(ay<h-3)&(bx>=3)&(bx<w-3)&(by>=3)&(by<h-3)
    ids=np.flatnonzero(center_ok)
    if ids.size:
        center_ok[ids] &= (inside[ay[ids],ax[ids]]>=3.0)&(inside[by[ids],bx[ids]]>=3.0)
    la=_e156_patch_median_batch_exact(L,garment,pa2,3); lb=_e156_patch_median_batch_exact(L,garment,pb2,3)
    pair_ok=center_ok&np.isfinite(la)&np.isfinite(lb)
    diff=(la-lb).reshape(ns,3); ok2=pair_ok.reshape(ns,3)
    rgb_deltas=[]; rgb_signs=[]; sample_interiors=[]
    for i in range(ns):
        vv=diff[i][ok2[i]]
        if vv.size:
            rgb_deltas.append(float(np.median(np.abs(vv)))); rgb_signs.append(float(np.median(vv)))
            p=samples[i]; xi=int(np.clip(round(float(p[0])),0,w-1)); yi=int(np.clip(round(float(p[1])),0,h-1)); sample_interiors.append(float(inside[yi,xi]))
    nvalid=len(rgb_deltas); valid_ratio=float(nvalid)/float(max(1,len(sample_indices))); rgb_step=float(np.median(rgb_deltas)) if rgb_deltas else 0.0
    strong_sign=np.asarray([v for v in rgb_signs if abs(float(v))>=2.0],dtype=np.float64)
    if strong_sign.size>=3: pos=float(np.mean(strong_sign>0)); neg=float(np.mean(strong_sign<0)); side_consistency=max(pos,neg)
    else: side_consistency=0.50
    side_score=float(np.clip((side_consistency-0.50)/0.35,0.0,1.0))
    raw_score=np.asarray(raw_score_map,dtype=np.float32); rv=raw_score[yy,xx]; rv=rv[np.isfinite(rv)]; raw_strength=float(np.mean(rv)) if rv.size else 0.0
    struct_near=np.asarray(ctx.get('struct_near',np.empty((0,0),dtype=bool)),dtype=bool)
    if struct_near.shape==(h,w): structural_support=float(np.mean(struct_near[yy,xx])) if (bool(ctx['_has_struct_near']) if '_has_struct_near' in ctx else bool(np.any(struct_near))) else 0.0
    else:
        if branch is None: branch=np.asarray(branch_u8,dtype=np.uint8)>0
        struct=np.asarray(structural_u8,dtype=np.uint8)>0; structural_support=float(np.mean((cv2.dilate((struct.astype(np.uint8)*255),_e149_cv_kernel('ones',5),iterations=1)>0)[yy,xx])) if np.any(struct) else 0.0
    deform_near=np.asarray(ctx.get('deform_near',np.empty((0,0),dtype=bool)),dtype=bool)
    if deform_near.shape==(h,w): deform_support=float(np.mean(deform_near[yy,xx])) if (bool(ctx['_has_deform_near']) if '_has_deform_near' in ctx else bool(np.any(deform_near))) else 0.0
    else:
        deform=np.asarray(pose_deform_u8,dtype=np.uint8)>0; deform_support=float(np.mean((cv2.dilate((deform.astype(np.uint8)*255),_e149_cv_kernel('ellipse',25),iterations=1)>0)[yy,xx])) if np.any(deform) else 0.0
    seam=np.asarray(ctx.get('seam',np.asarray(seam_u8,dtype=np.uint8)>0),dtype=bool); seam_overlap=float(np.mean(seam[yy,xx])) if (bool(ctx['_has_seam']) if '_has_seam' in ctx else bool(np.any(seam))) else 0.0
    interior_med=float(np.median(sample_interiors)) if sample_interiors else 0.0; rgb_score=float(np.tanh(max(0.0,rgb_step)/14.0)); appearance=float(np.clip(0.62*rgb_score+0.38*raw_strength,0.0,1.0))
    length_score=float(np.clip(span/150.0,0.0,1.0)); interior_score=float(np.clip(interior_med/18.0,0.0,1.0)); structure=float(np.clip(0.38*length_score+0.28*linearity+0.20*interior_score+0.14*structural_support,0.0,1.0))
    pose_term=float(np.clip(0.30+0.70*deform_support*float(np.clip(pose_deform_conf,0.0,1.0)),0.0,1.0)); noise_penalty=float(0.08*(1.0-valid_ratio)); seam_penalty=float(0.42*seam_overlap)
    fold_score=float(np.clip(0.36*appearance+0.36*structure+0.18*pose_term+0.10*side_score-seam_penalty-noise_penalty,0.0,1.0))
    return {'fold_score':fold_score,'geometry_score':structure,'appearance_score':appearance,'structure_score':structure,'pose_score':pose_term,'depth_step_mm':0.0,'depth_step_score':0.0,'surface_normal_jump':0.0,'surface_normal_score':0.0,'rgb_bilateral_step':rgb_step,'rgb_bilateral_score':rgb_score,'side_consistency':float(side_consistency),'side_consistency_score':float(side_score),'unidepth_fold_support':0.0,'raw_crease_strength':raw_strength,'structural_support':structural_support,'pose_deform_support':deform_support,'seam_overlap':seam_overlap,'seam_penalty':seam_penalty,'noise_penalty':noise_penalty,'valid_sample_ratio':valid_ratio,'valid_samples':int(nvalid),'span_px':span,'linearity':linearity,'interior_median_px':interior_med,'centroid_px':[float(ctr[0]),float(ctr[1])],'rgb_execution_support':float(np.clip(0.50*rgb_score+0.30*raw_strength+0.20*side_score,0.0,1.0))}

# ---- Exact E156 helper body ends ----

# Initialize compatibility args once; prepare_plan refreshes this per call.
_E92_MICRO_ARGS = _default_micro_args()
