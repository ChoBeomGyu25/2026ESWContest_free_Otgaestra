#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
step_d25.py

Reference-free bottom/pants finish evaluator.

This file never opens RoArm serial ports and never sends robot commands.
It combines the YOLO26m bottom-pose engine, segmentation contour derivatives,
convexity defects and pose-relative topology to evaluate:

  1) one coherent waist region,
  2) one expected crotch concavity,
  3) two exposed legs and two visible hems,
  4) pose/geometry-fused eight keypoints,
  5) unexpected macro boundary folds or overlap,
  6) broad contrast-supported internal folds only.

No per-garment normal image or user registration is used. Fine wrinkles, elastic
waist gathers, center-rise seams and ordinary hem lines do not block finish.

Snapshot keys:
  Q / ESC : quit
  SPACE   : force one fresh snapshot and evaluate it
  C       : clear the frozen result
  L       : lock/update Homography from currently detected ArUco markers
  R       : clear Homography in memory
  S       : save current overlay and JSON report

Dependency:
  Keep step_d23_v2.py in the same directory.

Recommended:
  python3 step_d25.py

No-window one-shot snapshot:
  python3 step_d25.py --once --no-window
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import os
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Deque, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

try:
    import step_d23_v2 as d23
except Exception as exc:  # pragma: no cover - clear runtime error for Jetson use
    raise RuntimeError(
        "step_d25.py requires step_d23_v2.py in the same directory. "
        f"Import failed: {exc!r}"
    ) from exc


STEP_D24_BUILD = "2026-07-17-d25-contour-differential-topology-medium-pose"

STATUS_COLORS = {
    "REJUDGE": (0, 190, 255),
    "NOT_READY_SHAPE": (0, 0, 255),
    "NOT_READY_WRINKLE": (0, 80, 255),
    "READY_PENDING": (0, 220, 220),
    "READY_GOOD_ENOUGH": (0, 210, 0),
}

TIER_COLORS = {
    1: (255, 170, 40),   # blue-ish: allowed fine wrinkle
    2: (0, 165, 255),    # orange
    3: (0, 0, 255),      # red
}


@dataclass
class StabilityEntry:
    timestamp: float
    mask_area_px: float
    mask_center_board: np.ndarray
    pose_axis_deg: float
    valid_kpts: int
    mean_kpt_conf: float


@dataclass
class D24State:
    history: Deque[StabilityEntry]
    ready_streak: int = 0
    evaluation_count: int = 0
    latest_status: str = "REJUDGE"
    latest_report: Dict[str, Any] = field(default_factory=dict)
    latest_obs: Optional[Any] = None
    latest_heat: Optional[Any] = None
    latest_overlay: Optional[np.ndarray] = None
    latest_eval_frame: Optional[np.ndarray] = None
    latest_h: Optional[np.ndarray] = None
    last_eval_started: float = 0.0
    force_evaluate: bool = False
    paused: bool = False

    def reset_temporal(self) -> None:
        self.history.clear()
        self.ready_streak = 0
        self.latest_status = "REJUDGE"
        self.latest_report = {
            "status": "REJUDGE",
            "reason": "temporal history reset",
        }


# -----------------------------------------------------------------------------
# Argument/config construction
# -----------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="D24-v6 event snapshot tester: pants shape + real-fold-only finish + waist/gather/center-rise structure ignore; no robot motion"
    )

    # Camera / files.
    p.add_argument("--config", default="dual_roarm_big_table_config.json")
    p.add_argument("--hfile", default="dual_big_table_homography_cache.json")
    p.add_argument("--load-h", dest="load_h", action="store_true", default=True)
    p.add_argument("--no-load-h", dest="load_h", action="store_false")
    p.add_argument("--save-h-on-lock", dest="save_h_on_lock", action="store_true", default=True)
    p.add_argument("--no-save-h-on-lock", dest="save_h_on_lock", action="store_false")
    p.add_argument("--auto-lock", dest="auto_lock", action="store_true", default=True)
    p.add_argument("--no-auto-lock", dest="auto_lock", action="store_false")

    p.add_argument("--camera", type=int, default=0)
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--backend", choices=["auto", "v4l2", "dshow", "any"], default="v4l2")
    p.add_argument("--cam-device", default="")
    p.add_argument("--cam-exposure-abs", type=int, default=35)
    p.add_argument("--cam-gain", type=int, default=5)
    p.add_argument("--cam-no-manual", action="store_true")
    p.add_argument("--cam-auto-adjust", dest="cam_auto_adjust", action="store_true", default=False)
    p.add_argument("--cam-auto-settle-s", type=float, default=0.40)
    p.add_argument("--cam-auto-exposure-min", type=int, default=10)
    p.add_argument("--cam-auto-exposure-max", type=int, default=100)
    p.add_argument("--cam-auto-gain-min", type=int, default=0)
    p.add_argument("--cam-auto-gain-max", type=int, default=20)
    p.add_argument("--warmup-frames", type=int, default=10)
    p.add_argument("--no-window", action="store_true")
    p.add_argument("--once", action="store_true")
    p.add_argument("--image", default="", help="Evaluate one saved image instead of the camera.")
    p.add_argument("--output", default="d25_finish_overlay.jpg")
    p.add_argument("--snapshot-dir", default="d25_finish_samples")
    p.add_argument("--log-jsonl", default="d25_finish_log.jsonl")

    # Models and pose inference.
    p.add_argument(
        "--seg-model",
        default="/workspace/project_train/aruco_test/dual/models/kfashion_yolo26s_seg3_e100_best.engine",
    )
    p.add_argument("--seg-imgsz", type=int, default=640)
    p.add_argument("--seg-conf", type=float, default=0.25)
    p.add_argument(
        "--pose-model",
        default="/workspace/project_train/yolo26/bottom_pose8_yolo26m_e40_best.engine",
    )
    p.add_argument("--pose-imgsz", type=int, default=640)
    p.add_argument("--pose-conf", type=float, default=0.10)
    p.add_argument("--pose-kpt-conf", type=float, default=0.20)
    p.add_argument("--pose-tta", dest="pose_tta", action="store_true", default=True)
    p.add_argument("--no-pose-tta", dest="pose_tta", action="store_false")
    p.add_argument("--angles", default="0,180,90,-90,45,-45")
    p.add_argument("--flip-modes", default="none,h")

    # Staged TTA from D23: fast common orientations, full fallback only if weak.
    p.add_argument("--d23-pose-tta-fast-first", dest="d23_pose_tta_fast_first", action="store_true", default=True)
    p.add_argument("--no-d23-pose-tta-fast-first", dest="d23_pose_tta_fast_first", action="store_false")
    p.add_argument("--d23-pose-tta-fast-angles", default="0,180")
    p.add_argument("--d23-pose-tta-fast-flip-modes", default="none")
    p.add_argument("--d23-pose-tta-fast-min-score", type=float, default=28.0)
    p.add_argument("--d23-pose-tta-fast-min-visible", type=int, default=7)
    p.add_argument("--d23-pose-tta-fast-min-mean-conf", type=float, default=0.32)

    # Pose TTA geometry and mask fallback.
    p.add_argument("--tta-min-waist-width-px", type=float, default=40.0)
    p.add_argument("--tta-max-waist-center-offset-ratio", type=float, default=0.35)
    p.add_argument("--tta-min-waist-to-crotch-ratio", type=float, default=0.35)
    p.add_argument("--tta-max-waist-axis-dot", type=float, default=0.45)
    p.add_argument("--tta-pre-spread-hem-gap-ratio", type=float, default=0.42)
    p.add_argument("--tta-pre-spread-open-angle-deg", type=float, default=22.0)
    p.add_argument("--tta-pre-spread-min-hem-visible", type=int, default=2)
    p.add_argument("--tta-mask-max-outside-px", type=float, default=28.0)
    p.add_argument("--tta-mask-min-inside-ratio", type=float, default=0.60)
    p.add_argument("--tta-require-ok-for-motion", dest="tta_require_ok_for_motion", action="store_true", default=True)
    p.add_argument("--no-tta-require-ok-for-motion", dest="tta_require_ok_for_motion", action="store_false")

    p.add_argument("--pose-grabcut-fallback", dest="pose_grabcut_fallback", action="store_true", default=True)
    p.add_argument("--no-pose-grabcut-fallback", dest="pose_grabcut_fallback", action="store_false")
    p.add_argument("--fallback-mask-min-pose-points", type=int, default=6)
    p.add_argument("--fallback-mask-bbox-pad-px", type=int, default=55)
    p.add_argument("--fallback-mask-seed-radius-px", type=int, default=12)
    p.add_argument("--fallback-mask-seed-line-px", type=int, default=18)
    p.add_argument("--fallback-mask-hull-dilate-px", type=int, default=28)
    p.add_argument("--fallback-mask-grabcut-iters", type=int, default=4)
    p.add_argument("--fallback-mask-close-px", type=int, default=9)
    p.add_argument("--fallback-mask-open-px", type=int, default=3)
    p.add_argument("--fallback-mask-min-area-px", type=float, default=1800.0)
    p.add_argument("--fallback-mask-max-frame-ratio", type=float, default=0.70)
    p.add_argument("--fallback-mask-max-kpt-outside-px", type=float, default=22.0)
    p.add_argument("--fallback-mask-min-inside-points", type=int, default=6)
    p.add_argument("--fallback-mask-min-solidity", type=float, default=0.35)

    # Continuous evaluation and temporal stability.
    p.add_argument("--eval-interval-s", type=float, default=0.75)
    p.add_argument("--burst-frame-count", type=int, default=5)
    p.add_argument("--history-count", type=int, default=3)
    p.add_argument("--ready-confirm-count", type=int, default=2)
    p.add_argument("--min-valid-keypoints", type=int, default=7)
    p.add_argument("--min-mean-kpt-conf", type=float, default=0.45)
    p.add_argument("--stable-mask-area-rel", type=float, default=0.05)
    p.add_argument("--stable-center-shift-mm", type=float, default=8.0)
    p.add_argument("--stable-pose-axis-spread-deg", type=float, default=6.0)

    # Base heatmap.
    p.add_argument("--wrinkle-heatmap-mode", dest="wrinkle_heatmap_mode", action="store_true", default=True)
    p.add_argument("--no-wrinkle-heatmap-mode", dest="wrinkle_heatmap_mode", action="store_false")
    p.add_argument("--wrinkle-heatmap-draw", dest="wrinkle_heatmap_draw", action="store_true", default=True)
    p.add_argument("--no-wrinkle-heatmap-draw", dest="wrinkle_heatmap_draw", action="store_false")
    p.add_argument("--wrinkle-heatmap-alpha", type=float, default=0.30)
    p.add_argument("--wrinkle-heatmap-clahe-clip", type=float, default=2.0)
    p.add_argument("--wrinkle-heatmap-clahe-tile", type=int, default=8)
    p.add_argument("--wrinkle-heatmap-blur-ksize", type=int, default=41)
    p.add_argument("--wrinkle-heatmap-erode-px", type=int, default=12)
    p.add_argument("--wrinkle-heatmap-percentile", type=float, default=93.0)
    p.add_argument("--wrinkle-heatmap-min-blob-area", type=float, default=45.0)
    p.add_argument("--wrinkle-heatmap-max-candidates", type=int, default=12)
    p.add_argument("--wrinkle-heatmap-gabor", dest="wrinkle_heatmap_gabor", action="store_true", default=True)
    p.add_argument("--no-wrinkle-heatmap-gabor", dest="wrinkle_heatmap_gabor", action="store_false")
    p.add_argument("--wrinkle-heatmap-gabor-ksize", type=int, default=17)
    p.add_argument("--wrinkle-heatmap-gabor-lambda", type=float, default=8.0)
    p.add_argument("--wrinkle-heatmap-gabor-sigma", type=float, default=4.0)
    p.add_argument("--wrinkle-heatmap-score-min", type=float, default=35.0)

    # D24 robust response guard. This supplements P93 rather than allowing P93 alone.
    p.add_argument("--d24-robust-k", type=float, default=2.7)
    p.add_argument("--d24-robust-score-min", type=float, default=35.0)
    p.add_argument("--d24-candidate-min-robust-support", type=float, default=0.10)

    # Natural structure ignore zones.
    p.add_argument("--d19-ignore-waistband-wrinkles", dest="d19_ignore_waistband_wrinkles", action="store_true", default=True)
    p.add_argument("--no-d19-ignore-waistband-wrinkles", dest="d19_ignore_waistband_wrinkles", action="store_false")
    p.add_argument("--d19-waistband-back-mm", type=float, default=18.0)
    p.add_argument("--d19-waistband-forward-mm", type=float, default=72.0)
    p.add_argument("--d19-waistband-side-expand-mm", type=float, default=28.0)
    p.add_argument("--d23-waistband-forward-ratio", type=float, default=0.70)
    p.add_argument("--d23-waistband-forward-max-mm", type=float, default=95.0)
    p.add_argument("--d23-waist-structure-veto", dest="d23_waist_structure_veto", action="store_true", default=True)
    p.add_argument("--no-d23-waist-structure-veto", dest="d23_waist_structure_veto", action="store_false")
    p.add_argument("--d23-waist-structure-back-mm", type=float, default=22.0)
    p.add_argument("--d23-waist-structure-forward-mm", type=float, default=78.0)
    p.add_argument("--d23-waist-structure-forward-ratio", type=float, default=0.70)
    p.add_argument("--d23-waist-structure-forward-max-mm", type=float, default=98.0)
    p.add_argument("--d23-waist-structure-side-expand-mm", type=float, default=32.0)
    p.add_argument("--d23-waist-structure-parallel-deg", type=float, default=25.0)
    p.add_argument("--d23-waist-structure-min-length-mm", type=float, default=35.0)
    p.add_argument("--d23-waist-structure-min-length-ratio", type=float, default=0.22)
    p.add_argument("--d23-pre-veto-candidate-multiplier", type=int, default=3)
    p.add_argument("--d23-draw-waist-veto", dest="d23_draw_waist_veto", action="store_true", default=True)
    p.add_argument("--no-d23-draw-waist-veto", dest="d23_draw_waist_veto", action="store_false")

    p.add_argument("--d19-ignore-crotch-wrinkles", dest="d19_ignore_crotch_wrinkles", action="store_true", default=True)
    p.add_argument("--no-d19-ignore-crotch-wrinkles", dest="d19_ignore_crotch_wrinkles", action="store_false")
    p.add_argument("--d19-crotch-axial-radius-mm", type=float, default=56.0)
    p.add_argument("--d19-crotch-lateral-radius-mm", type=float, default=66.0)
    p.add_argument("--d19-crotch-forward-shift-mm", type=float, default=10.0)

    # Conditional seam veto: do not erase the entire side/hem area; require line agreement.
    p.add_argument("--d24-seam-veto", dest="d24_seam_veto", action="store_true", default=True)
    p.add_argument("--no-d24-seam-veto", dest="d24_seam_veto", action="store_false")
    p.add_argument("--d24-seam-band-mm", type=float, default=18.0)
    p.add_argument("--d24-seam-parallel-deg", type=float, default=24.0)
    p.add_argument("--d24-seam-min-linearity", type=float, default=3.0)
    p.add_argument("--d24-seam-min-length-mm", type=float, default=30.0)
    p.add_argument("--d24-seam-min-support-ratio", type=float, default=0.70)
    p.add_argument("--d24-seam-max-sample-points", type=int, default=1800)

    # D20 meaningful wrinkle classification.
    p.add_argument("--d20-require-dark-shadow", dest="d20_require_dark_shadow", action="store_true", default=True)
    p.add_argument("--no-d20-require-dark-shadow", dest="d20_require_dark_shadow", action="store_false")
    p.add_argument("--d20-ignore-fine-wrinkles", dest="d20_ignore_fine_wrinkles", action="store_true", default=True)
    p.add_argument("--no-d20-ignore-fine-wrinkles", dest="d20_ignore_fine_wrinkles", action="store_false")
    p.add_argument("--d20-shadow-blur-ksize", type=int, default=51)
    p.add_argument("--d20-shadow-min-scale-raw", type=float, default=8.0)
    p.add_argument("--d20-shadow-support-raw-min", type=float, default=3.0)
    p.add_argument("--d20-hard-min-component-area-px", type=float, default=70.0)

    # T3 strong criteria.
    p.add_argument("--d20-strong-area-px", type=float, default=240.0)
    p.add_argument("--d20-strong-area-ratio", type=float, default=0.015)
    p.add_argument("--d20-strong-length-mm", type=float, default=80.0)
    p.add_argument("--d20-strong-length-px", type=float, default=95.0)
    p.add_argument("--d20-strong-dark-mean", type=float, default=38.0)
    p.add_argument("--d20-strong-dark-max", type=float, default=105.0)
    p.add_argument("--d20-strong-dark-support", type=float, default=0.10)

    # T2 medium criteria.
    p.add_argument("--d20-medium-area-px", type=float, default=150.0)
    p.add_argument("--d20-medium-area-ratio", type=float, default=0.005)
    p.add_argument("--d20-medium-length-mm", type=float, default=45.0)
    p.add_argument("--d20-medium-length-px", type=float, default=58.0)
    p.add_argument("--d20-medium-dark-mean", type=float, default=20.0)
    p.add_argument("--d20-medium-dark-max", type=float, default=65.0)
    p.add_argument("--d20-medium-dark-support", type=float, default=0.06)

    # D21 candidate severity / T1 allowance.
    p.add_argument("--d21v4-action-min-tier", type=int, default=2)
    p.add_argument("--d21v4-action-severity-override", type=float, default=0.55)
    p.add_argument("--d21v4-good-enough-finish", dest="d21v4_good_enough_finish", action="store_true", default=True)
    p.add_argument("--d21-severity-length-ref-mm", type=float, default=120.0)
    p.add_argument("--d21-severity-area-ref-px", type=float, default=550.0)
    p.add_argument("--d21-severity-dark-ref", type=float, default=60.0)
    p.add_argument("--d21-severity-dark-max-ref", type=float, default=160.0)
    p.add_argument("--d21-severity-sagitta-ref-mm", type=float, default=45.0)
    p.add_argument("--d21-pull-min-mm", type=float, default=25.0)
    p.add_argument("--d21-pull-max-mm", type=float, default=95.0)

    # Shape thresholds from the agreed algorithm.
    p.add_argument("--finish-shape-score-min", type=float, default=0.75)
    p.add_argument("--finish-hem-gap-ratio-min", type=float, default=0.55)
    p.add_argument("--finish-hem-gap-ratio-max", type=float, default=1.35)
    p.add_argument("--finish-leg-balance-min", type=float, default=0.82)
    p.add_argument("--finish-waist-center-offset-max", type=float, default=0.15)
    p.add_argument("--finish-waist-line-error-max", type=float, default=0.16)
    p.add_argument("--finish-crotch-axis-offset-max", type=float, default=0.22)
    p.add_argument("--finish-axis-to-waist-min", type=float, default=0.75)

    # Lower silhouette thresholds.
    p.add_argument("--d22-require-lower-leg-symmetry", dest="d22_require_lower_leg_symmetry", action="store_true", default=True)
    p.add_argument("--no-d22-require-lower-leg-symmetry", dest="d22_require_lower_leg_symmetry", action="store_false")
    p.add_argument("--d22-symmetry-axis-min-mm", type=float, default=35.0)
    p.add_argument("--d22-symmetry-lower-start-mm", type=float, default=-4.0)
    p.add_argument("--d22-symmetry-ignore-axis-band-mm", type=float, default=5.0)
    p.add_argument("--d22-symmetry-grid-mm", type=float, default=4.0)
    p.add_argument("--d22-symmetry-grid-close-cells", type=int, default=1)
    p.add_argument("--d22-symmetry-band-count", type=int, default=6)
    p.add_argument("--d22-symmetry-min-points-per-band", type=int, default=60)
    p.add_argument("--d22-symmetry-max-sample-points", type=int, default=100000)
    p.add_argument("--d22-lower-reflected-iou-min", type=float, default=0.68)
    p.add_argument("--d22-lower-width-balance-median-min", type=float, default=0.75)
    p.add_argument("--d22-lower-width-balance-min-min", type=float, default=0.60)
    p.add_argument("--d22-lower-profile-error-p90-max", type=float, default=0.38)
    p.add_argument("--d22-lower-side-area-balance-min", type=float, default=0.74)
    p.add_argument("--d22-hem-axis-balance-min", type=float, default=0.74)
    p.add_argument("--d22-leg-length-balance-min", type=float, default=0.82)
    p.add_argument("--d22-symmetry-score-min", type=float, default=0.70)

    # D24-v2 final policy: pants shape is mandatory, but only clearly major
    # wrinkles block READY. T1/T2 counts and total active ratio are diagnostic only.
    p.add_argument("--finish-max-t3-count", type=int, default=0)
    p.add_argument("--finish-max-t2-count", type=int, default=999)  # display/compatibility only
    p.add_argument("--finish-max-severity", type=float, default=1.0)  # display/compatibility only
    p.add_argument("--finish-actionable-ratio", type=float, default=1.0)  # diagnostic only
    p.add_argument("--finish-max-blob-ratio", type=float, default=1.0)  # diagnostic only

    # Waistline-relative fine-wrinkle allowance. A candidate smaller/weaker than
    # the detected waistband structure is explicitly marked as allowed.
    p.add_argument("--d24v2-waist-relative-enable", dest="d24v2_waist_relative_enable", action="store_true", default=True)
    p.add_argument("--no-d24v2-waist-relative-enable", dest="d24v2_waist_relative_enable", action="store_false")
    p.add_argument("--d24v2-waist-length-ratio-ignore", type=float, default=0.40)
    p.add_argument("--d24v2-waist-area-ratio-ignore", type=float, default=0.30)
    p.add_argument("--d24v2-waist-response-ratio-ignore", type=float, default=0.85)
    p.add_argument("--d24v2-waist-ref-back-mm", type=float, default=18.0)
    p.add_argument("--d24v2-waist-ref-forward-mm", type=float, default=72.0)
    p.add_argument("--d24v2-waist-ref-side-expand-mm", type=float, default=28.0)
    p.add_argument("--d24v2-waist-ref-parallel-deg", type=float, default=30.0)
    p.add_argument("--d24v2-waist-ref-min-length-ratio", type=float, default=0.25)
    p.add_argument("--d24v2-waist-ref-min-area-px", type=float, default=45.0)
    p.add_argument("--d24v2-waist-ref-threshold-scale", type=float, default=0.78)

    # Absolute major-wrinkle overrides. These block READY even when the candidate
    # is smaller than the waistband reference.
    p.add_argument("--d24v2-major-length-mm", type=float, default=80.0)
    p.add_argument("--d24v2-major-area-ratio", type=float, default=0.015)
    p.add_argument("--d24v2-major-severity", type=float, default=0.60)
    p.add_argument("--d24v2-major-severity-min-length-mm", type=float, default=60.0)
    p.add_argument("--d24v2-major-waist-span-ratio", type=float, default=0.50)
    p.add_argument("--d24v2-major-waist-span-min-area-ratio", type=float, default=0.005)

    # D24-v5: waistband-connected radial gathers are natural elastic-waist structures.
    p.add_argument("--d24v5-waist-gather-ignore", dest="d24v5_waist_gather_ignore", action="store_true", default=True)
    p.add_argument("--no-d24v5-waist-gather-ignore", dest="d24v5_waist_gather_ignore", action="store_false")
    p.add_argument("--d24v5-waist-gather-depth-ratio", type=float, default=0.45,
                   help="Maximum gather depth as a fraction of waist-to-crotch distance.")
    p.add_argument("--d24v5-waist-gather-depth-max-mm", type=float, default=105.0)
    p.add_argument("--d24v5-waist-gather-connect-mm", type=float, default=22.0,
                   help="Allowed gap from the existing waistband ignore boundary.")
    p.add_argument("--d24v5-waist-gather-radial-angle-deg", type=float, default=32.0)
    p.add_argument("--d24v5-waist-gather-max-length-ratio", type=float, default=0.35)
    p.add_argument("--d24v5-waist-gather-max-length-mm", type=float, default=78.0)
    p.add_argument("--d24v5-waist-gather-max-area-ratio", type=float, default=0.012)
    p.add_argument("--d24v5-waist-gather-side-expand-mm", type=float, default=20.0)
    p.add_argument("--d24v5-waist-gather-min-upper-support", type=float, default=0.70)

    # D24-v5: fine wrinkles never block finish. Only broad/deep fold-like components do.
    p.add_argument("--d24v5-fold-min-length-mm", type=float, default=75.0)
    p.add_argument("--d24v5-fold-min-width-mm", type=float, default=7.0)
    p.add_argument("--d24v5-fold-min-area-ratio", type=float, default=0.0045)
    p.add_argument("--d24v5-fold-min-severity", type=float, default=0.45)
    p.add_argument("--d24v5-fold-min-mean-response", type=float, default=75.0)
    p.add_argument("--d24v5-fold-broad-min-length-mm", type=float, default=45.0)
    p.add_argument("--d24v5-fold-broad-min-width-mm", type=float, default=13.0)
    p.add_argument("--d24v5-fold-broad-min-area-ratio", type=float, default=0.0075)
    p.add_argument("--d24v5-fold-large-area-ratio", type=float, default=0.018)
    p.add_argument("--d24v5-draw-allowed-fine", dest="d24v5_draw_allowed_fine", action="store_true", default=False)
    p.add_argument("--no-d24v5-draw-allowed-fine", dest="d24v5_draw_allowed_fine", action="store_false")

    # D24-v6: pose-relative front/center-rise seam from waist center toward crotch.
    # This narrow structure band is erased BEFORE CCA so a bright seam cannot merge
    # with nearby waistband gathers and become one large false fold component.
    p.add_argument("--d24v6-center-rise-ignore", dest="d24v6_center_rise_ignore", action="store_true", default=True)
    p.add_argument("--no-d24v6-center-rise-ignore", dest="d24v6_center_rise_ignore", action="store_false")
    p.add_argument("--d24v6-center-rise-half-width-mm", type=float, default=18.0)
    p.add_argument("--d24v6-center-rise-start-mm", type=float, default=48.0,
                   help="Distance below waist center where the narrow center-rise seam mask starts.")
    p.add_argument("--d24v6-center-rise-end-before-crotch-mm", type=float, default=0.0)
    p.add_argument("--d24v6-center-rise-end-expand-mm", type=float, default=12.0)

    # D24-v6 color-robust fold gate. A large normalized heatmap component alone is
    # no longer a blocker; broad geometry must also have real local intensity contrast.
    p.add_argument("--d24v6-fold-min-abs-contrast", type=float, default=7.0,
                   help="Minimum mean absolute high-pass contrast in original gray levels.")
    p.add_argument("--d24v6-fold-min-contrast-support", type=float, default=0.28,
                   help="Minimum fraction of component pixels above the absolute-contrast pixel threshold.")
    p.add_argument("--d24v6-fold-contrast-pixel-threshold", type=float, default=8.0)
    p.add_argument("--d24v6-fold-extreme-width-mm", type=float, default=22.0)
    p.add_argument("--d24v6-fold-extreme-area-ratio", type=float, default=0.016)

    # D24-v4 event-driven snapshot gate. The camera remains live, but heavy
    # pose/segmentation/heatmap inference runs only after a confirmed scene change
    # followed by a stillness interval.
    p.add_argument("--auto-snapshot-on-change", dest="auto_snapshot_on_change", action="store_true", default=True)
    p.add_argument("--no-auto-snapshot-on-change", dest="auto_snapshot_on_change", action="store_false")
    p.add_argument("--startup-auto-snapshot", dest="startup_auto_snapshot", action="store_true", default=True)
    p.add_argument("--no-startup-auto-snapshot", dest="startup_auto_snapshot", action="store_false")
    p.add_argument("--change-mean-threshold", type=float, default=3.5,
                   help="Brightness-compensated mean board difference needed to arm a new snapshot.")
    p.add_argument("--change-pixel-threshold", type=float, default=12.0,
                   help="Per-pixel difference used to calculate changed-area ratio.")
    p.add_argument("--change-active-ratio", type=float, default=0.018,
                   help="Fraction of board pixels that must change when mean difference alone is small.")
    p.add_argument("--change-min-frames", type=int, default=3,
                   help="Consecutive changed frames required before waiting for stillness.")
    p.add_argument("--change-cooldown-s", type=float, default=0.60,
                   help="Ignore change triggers briefly after each completed snapshot.")
    p.add_argument("--change-board-shrink-px", type=int, default=10,
                   help="Ignore a narrow band at the ArUco board boundary during change detection.")
    p.add_argument("--change-preview-width", type=int, default=160)
    p.add_argument("--change-preview-height", type=int, default=90)
    p.add_argument("--settle-motion-threshold", type=float, default=2.5,
                   help="Frame-to-frame mean difference below which the pants are considered still.")
    p.add_argument("--settle-seconds", type=float, default=0.80,
                   help="How long the view must remain still before automatic snapshot inference.")
    p.add_argument("--settle-min-frames", type=int, default=5)
    p.add_argument("--capture-flush-frames", type=int, default=2)
    p.add_argument("--pause-auto", dest="pause_auto", action="store_true", default=False)

    p.add_argument("--panel-width", type=int, default=680)
    p.add_argument("--window-name", default="D25_PANTS_TOPOLOGY_FINISH")
    return p


def finalize_args(args: argparse.Namespace) -> argparse.Namespace:
    args.pose_tta_angles_list = d23.parse_pose_tta_angles(args.angles)
    args.pose_tta_flip_modes_list = d23.parse_pose_tta_flip_modes(args.flip_modes)

    # Names expected by D23 shape helpers.
    args.d19_finish_require_shape = True
    args.d19_finish_shape_score_good = float(args.finish_shape_score_min)
    args.d19_finish_shape_hem_gap_ratio_min = float(args.finish_hem_gap_ratio_min)
    args.d19_finish_shape_hem_gap_ratio_max = float(args.finish_hem_gap_ratio_max)
    args.d19_finish_shape_leg_balance_min = float(args.finish_leg_balance_min)
    args.d19_finish_shape_waist_center_offset_max = float(args.finish_waist_center_offset_max)
    args.d19_finish_shape_waist_line_error_max = float(args.finish_waist_line_error_max)
    args.d19_finish_shape_crotch_axis_offset_max = float(args.finish_crotch_axis_offset_max)
    args.d19_finish_shape_axis_to_waist_min = float(args.finish_axis_to_waist_min)

    # Geometry helper defaults that are accessed directly in a few old functions.
    args.d21_curve_min_inlier_ratio = 0.62
    args.d21_curve_max_residual_mm = 8.0
    args.d21_curve_min_sagitta_mm = 10.0
    args.d21_curve_min_arc_length_mm = 50.0
    args.d21_curve_min_radius_mm = 25.0
    args.d21_curve_max_radius_mm = 900.0
    args.d21_curve_vs_line_error_ratio = 0.90
    args.d21_line_max_residual_mm = 8.0
    args.d21_line_max_sagitta_mm = 10.0
    args.d21_circle_ransac_iters = 120
    args.d21_circle_inlier_tol_mm = 7.0
    args.d21_fit_max_points = 220
    args.d21_track_match_distance_mm = 35.0
    args.d21_static_required_actions = 999999  # No automatic structural learning in D24.
    args.d21v3_static_require_fine = True
    args.d21v3_static_max_tier = 1
    args.d21v3_static_max_severity = 0.45
    args.d21v3_static_max_length_mm = 65.0
    args.d21v3_static_max_area_px = 240.0
    args.d21v3_static_max_dark_mean = 38.0
    return args


# -----------------------------------------------------------------------------
# Geometry / temporal stability
# -----------------------------------------------------------------------------

def angle_diff_180(a_deg: float, b_deg: float) -> float:
    return abs((float(a_deg) - float(b_deg) + 90.0) % 180.0 - 90.0)


def pose_axis_angle(obs: Any) -> float:
    if obs is None or obs.pose is None:
        return 0.0
    v = np.asarray(obs.pose.lower_center, np.float32) - np.asarray(obs.pose.waist_center, np.float32)
    return float(math.degrees(math.atan2(float(v[1]), float(v[0]))))


def observation_entry(obs: Any) -> StabilityEntry:
    confs = list(obs.pose.keypoint_conf.values()) if obs and obs.pose else []
    return StabilityEntry(
        timestamp=time.time(),
        mask_area_px=float(obs.mask.area_px),
        mask_center_board=np.asarray(obs.mask.center_board, np.float32).reshape(2),
        pose_axis_deg=pose_axis_angle(obs),
        valid_kpts=len(confs),
        mean_kpt_conf=float(np.mean(confs)) if confs else 0.0,
    )


def temporal_stability_report(history: Sequence[StabilityEntry], args) -> Dict[str, Any]:
    need = max(2, int(args.history_count))
    if len(history) < need:
        return {
            "good": False,
            "reason": f"history {len(history)}/{need}",
            "history_count": len(history),
            "required": need,
        }

    entries = list(history)[-need:]
    areas = np.asarray([e.mask_area_px for e in entries], np.float32)
    centers = np.asarray([e.mask_center_board for e in entries], np.float32)
    angles = [e.pose_axis_deg for e in entries]

    med_area = float(np.median(areas))
    area_rel = float(np.max(np.abs(areas - med_area)) / max(1.0, med_area))
    med_center = np.median(centers, axis=0)
    center_shift = float(np.max(np.linalg.norm(centers - med_center.reshape(1, 2), axis=1)))
    angle_spread = max(angle_diff_180(a, b) for a in angles for b in angles)

    checks = {
        "mask_area": area_rel <= float(args.stable_mask_area_rel),
        "mask_center": center_shift <= float(args.stable_center_shift_mm),
        "pose_axis": angle_spread <= float(args.stable_pose_axis_spread_deg),
    }
    return {
        "good": bool(all(checks.values())),
        "reason": "OK" if all(checks.values()) else "observation changed",
        "checks": checks,
        "history_count": len(entries),
        "mask_area_rel": area_rel,
        "center_shift_mm": center_shift,
        "pose_axis_spread_deg": float(angle_spread),
        "limits": {
            "mask_area_rel": float(args.stable_mask_area_rel),
            "center_shift_mm": float(args.stable_center_shift_mm),
            "pose_axis_spread_deg": float(args.stable_pose_axis_spread_deg),
        },
    }


# -----------------------------------------------------------------------------
# Conditional side/inner/hem seam filtering
# -----------------------------------------------------------------------------

def point_segment_distances(points: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, np.float32).reshape(-1, 2)
    a = np.asarray(a, np.float32).reshape(2)
    b = np.asarray(b, np.float32).reshape(2)
    ab = b - a
    denom = float(np.dot(ab, ab))
    if denom < 1e-8:
        return np.linalg.norm(pts - a.reshape(1, 2), axis=1)
    t = np.clip(((pts - a.reshape(1, 2)) @ ab) / denom, 0.0, 1.0)
    proj = a.reshape(1, 2) + t[:, None] * ab.reshape(1, 2)
    return np.linalg.norm(pts - proj, axis=1)


def expected_seam_segments(obs: Any) -> List[Dict[str, Any]]:
    if obs is None or obs.pose is None:
        return []
    kb = obs.pose.keypoints_board
    segments: List[Dict[str, Any]] = []

    def add(name: str, n1: str, n2: str) -> None:
        if n1 in kb and n2 in kb:
            a = np.asarray(kb[n1], np.float32)
            b = np.asarray(kb[n2], np.float32)
            if float(np.linalg.norm(b - a)) >= 12.0:
                segments.append({"name": name, "a": a, "b": b})

    add("LEFT_OUTER_SEAM", "waist_img_left", "img_left_hem_outer")
    add("RIGHT_OUTER_SEAM", "waist_img_right", "img_right_hem_outer")
    add("LEFT_INNER_SEAM", "crotch", "img_left_hem_inner")
    add("RIGHT_INNER_SEAM", "crotch", "img_right_hem_inner")
    add("LEFT_HEM", "img_left_hem_outer", "img_left_hem_inner")
    add("RIGHT_HEM", "img_right_hem_inner", "img_right_hem_outer")
    return segments


def candidate_component_mask(heat: Any, cand: Dict[str, Any]) -> Optional[np.ndarray]:
    binary = getattr(heat, "d21v4_residual_binary", None)
    if not isinstance(binary, np.ndarray):
        binary = getattr(heat, "binary", None)
    if not isinstance(binary, np.ndarray):
        return None
    n, labels = cv2.connectedComponents((binary > 0).astype(np.uint8), connectivity=8)
    cx, cy = [int(round(float(v))) for v in cand.get("center_px", [0.0, 0.0])]
    cx = int(np.clip(cx, 0, labels.shape[1] - 1))
    cy = int(np.clip(cy, 0, labels.shape[0] - 1))
    lid = int(labels[cy, cx])
    if lid <= 0 or lid >= n:
        # Search within the candidate bbox if its center fell in a morphology hole.
        x, y, w, h = [int(v) for v in cand.get("bbox", [0, 0, 0, 0])]
        roi = labels[max(0, y):min(labels.shape[0], y + h), max(0, x):min(labels.shape[1], x + w)]
        vals, counts = np.unique(roi[roi > 0], return_counts=True)
        if len(vals) == 0:
            return None
        lid = int(vals[int(np.argmax(counts))])
    return ((labels == lid).astype(np.uint8) * 255)


def component_board_points(component: np.ndarray, H: np.ndarray, max_points: int) -> np.ndarray:
    ys, xs = np.where(component > 0)
    pts = np.column_stack([xs, ys]).astype(np.float32)
    if len(pts) > max_points:
        step = max(1, int(math.ceil(len(pts) / float(max_points))))
        pts = pts[::step]
    if len(pts) == 0:
        return np.empty((0, 2), np.float32)
    return cv2.perspectiveTransform(pts.reshape(-1, 1, 2), H).reshape(-1, 2)


def axis_angle_error_deg(v1: np.ndarray, v2: np.ndarray) -> float:
    a = np.asarray(v1, np.float32).reshape(2)
    b = np.asarray(v2, np.float32).reshape(2)
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    if na < 1e-6 or nb < 1e-6:
        return 180.0
    c = float(np.clip(abs(np.dot(a / na, b / nb)), 0.0, 1.0))
    return float(math.degrees(math.acos(c)))


def seam_veto_report(cand: Dict[str, Any], heat: Any, obs: Any, H: np.ndarray, args) -> Dict[str, Any]:
    out: Dict[str, Any] = {"matched": False, "name": "", "reason": "not a seam"}
    if not bool(args.d24_seam_veto):
        return out

    length = float(cand.get("major_length_mm", 0.0))
    linearity = float(cand.get("linearity", 1.0))
    if length < float(args.d24_seam_min_length_mm) or linearity < float(args.d24_seam_min_linearity):
        return {**out, "reason": "short or non-linear"}

    comp = candidate_component_mask(heat, cand)
    if comp is None:
        return {**out, "reason": "component unavailable"}
    pts = component_board_points(comp, H, max(100, int(args.d24_seam_max_sample_points)))
    if len(pts) < 8:
        return {**out, "reason": "too few component points"}

    geom = cand.get("d21_geometry", {}) or {}
    tangent = geom.get("tangent_board")
    if tangent is None:
        center_px = cand.get("center_px", [0.0, 0.0])
        tangent = d23._pixel_axis_to_board(H, center_px, cand.get("major_axis_px", [1.0, 0.0]))
    if tangent is None:
        return {**out, "reason": "axis unavailable"}
    tangent = np.asarray(tangent, np.float32)

    best = None
    for seg in expected_seam_segments(obs):
        seg_vec = np.asarray(seg["b"], np.float32) - np.asarray(seg["a"], np.float32)
        angle_error = axis_angle_error_deg(tangent, seg_vec)
        dists = point_segment_distances(pts, seg["a"], seg["b"])
        support = float(np.count_nonzero(dists <= float(args.d24_seam_band_mm))) / float(max(1, len(dists)))
        score = support - 0.005 * angle_error
        item = {
            "name": seg["name"],
            "support_ratio": support,
            "parallel_error_deg": angle_error,
            "score": score,
            "a": [float(seg["a"][0]), float(seg["a"][1])],
            "b": [float(seg["b"][0]), float(seg["b"][1])],
        }
        if best is None or score > float(best["score"]):
            best = item

    if best is None:
        return out
    matched = bool(
        float(best["support_ratio"]) >= float(args.d24_seam_min_support_ratio)
        and float(best["parallel_error_deg"]) <= float(args.d24_seam_parallel_deg)
    )
    return {
        **best,
        "matched": matched,
        "reason": "pose-relative seam" if matched else "seam agreement insufficient",
        "length_mm": length,
        "linearity": linearity,
    }


# -----------------------------------------------------------------------------
# D24 robust heat support and final evaluation
# -----------------------------------------------------------------------------

def robust_heat_threshold(heat: Any, args) -> Dict[str, float]:
    if heat is None:
        return {"threshold": 255.0, "median": 0.0, "mad": 0.0, "sigma": 0.0}
    values = heat.heatmap[heat.inner_mask > 0].astype(np.float32)
    if values.size == 0:
        return {"threshold": 255.0, "median": 0.0, "mad": 0.0, "sigma": 0.0}
    med = float(np.median(values))
    mad = float(np.median(np.abs(values - med)))
    sigma = max(1.0, 1.4826 * mad)
    threshold = max(float(args.d24_robust_score_min), med + float(args.d24_robust_k) * sigma)
    return {"threshold": threshold, "median": med, "mad": mad, "sigma": sigma}


def candidate_robust_support(cand: Dict[str, Any], heat: Any, threshold: float) -> float:
    comp = candidate_component_mask(heat, cand)
    if comp is None:
        return 0.0
    vals = heat.heatmap[comp > 0]
    if vals.size == 0:
        return 0.0
    return float(np.count_nonzero(vals.astype(np.float32) >= float(threshold))) / float(vals.size)


def observation_quality_report(obs: Any, args) -> Dict[str, Any]:
    if obs is None or obs.mask is None or obs.pose is None:
        return {
            "good": False,
            "reason": "pose or mask missing",
            "valid_keypoints": 0,
            "mean_kpt_conf": 0.0,
        }
    confs = list(obs.pose.keypoint_conf.values())
    valid_kpts = len(confs)
    mean_conf = float(np.mean(confs)) if confs else 0.0
    checks = {
        "observation_valid": bool(obs.valid),
        "pose_valid": bool(obs.pose.valid),
        "mask_area": float(obs.mask.area_px) >= 1200.0,
        "valid_keypoints": valid_kpts >= int(args.min_valid_keypoints),
        "mean_kpt_conf": mean_conf >= float(args.min_mean_kpt_conf),
    }
    return {
        "good": bool(all(checks.values())),
        "reason": "OK" if all(checks.values()) else "observation quality low",
        "checks": checks,
        "valid_keypoints": valid_kpts,
        "mean_kpt_conf": mean_conf,
        "pose_tta_state": str(obs.pose.tta_state),
        "pose_tta_score": float(obs.pose.tta_score),
        "pose_tta_tested": int(obs.pose.tta_tested_count),
        "mask_area_px": float(obs.mask.area_px),
    }


def _d24v2_waist_reference(obs: Any, heat: Any, H: np.ndarray, robust: Dict[str, float], args) -> Dict[str, Any]:
    """Measure a visible waistband structural response as a per-garment ruler.

    D23 already removes pose-known waistband candidates from robot/finish
    candidates. D24-v2 independently samples the raw heatmap inside a narrow
    pose-relative waistband band, so small residual wrinkles can be compared
    with the garment's own natural waistband response.
    """
    out: Dict[str, Any] = {
        "valid": False,
        "reason": "waist reference unavailable",
        "length_mm": 0.0,
        "area_px": 0.0,
        "mean_response": 0.0,
        "max_response": 0.0,
        "waist_width_mm": 0.0,
        "component_count": 0,
    }
    if not bool(getattr(args, "d24v2_waist_relative_enable", True)):
        out["reason"] = "disabled"
        return out
    if obs is None or obs.pose is None or not obs.pose.valid or heat is None or H is None:
        return out

    p = obs.pose
    wl = np.asarray(p.waist_left, np.float32)
    wc = np.asarray(p.waist_center, np.float32)
    wr = np.asarray(p.waist_right, np.float32)
    lower = np.asarray(p.lower_center, np.float32)
    body_u = d23._safe_unit(lower - wc)
    waist_u = d23._safe_unit(wr - wl)
    if abs(float(np.dot(body_u, waist_u))) > 0.55:
        waist_u = np.asarray([-body_u[1], body_u[0]], np.float32)

    waist_width = max(1.0, float(np.linalg.norm(wr - wl)))
    half_width = 0.5 * waist_width + float(args.d24v2_waist_ref_side_expand_mm)
    back = max(0.0, float(args.d24v2_waist_ref_back_mm))
    forward = max(0.0, float(args.d24v2_waist_ref_forward_mm))
    corners = [
        wc - waist_u * half_width - body_u * back,
        wc + waist_u * half_width - body_u * back,
        wc + waist_u * half_width + body_u * forward,
        wc - waist_u * half_width + body_u * forward,
    ]
    band = d23._d19_board_polygon_to_mask(H, corners, heat.heatmap.shape)
    band = cv2.bitwise_and((band > 0).astype(np.uint8) * 255, (heat.inner_mask > 0).astype(np.uint8) * 255)
    out["waist_width_mm"] = waist_width
    out["band_area_px"] = int(cv2.countNonZero(band))
    if cv2.countNonZero(band) < 30:
        out["reason"] = "waist band too small"
        return out

    threshold = max(
        20.0,
        float(robust.get("threshold", 255.0)) * float(args.d24v2_waist_ref_threshold_scale),
    )
    binary = np.zeros_like(heat.heatmap, np.uint8)
    binary[(heat.heatmap.astype(np.float32) >= threshold) & (band > 0)] = 255
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8), iterations=1)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)

    n, labels, stats, _ = cv2.connectedComponentsWithStats((binary > 0).astype(np.uint8), 8)
    candidates: List[Dict[str, Any]] = []
    min_area = max(1.0, float(args.d24v2_waist_ref_min_area_px))
    min_length = waist_width * float(args.d24v2_waist_ref_min_length_ratio)
    parallel_limit = float(args.d24v2_waist_ref_parallel_deg)
    for label in range(1, n):
        area = float(stats[label, cv2.CC_STAT_AREA])
        if area < min_area:
            continue
        comp = ((labels == label).astype(np.uint8) * 255)
        geom = d23._d21_component_geometry_model(comp, H, args)
        tangent = d23._safe_unit(np.asarray(geom.get("tangent_board", [1.0, 0.0]), np.float32))
        parallel_error = float(np.degrees(np.arccos(np.clip(abs(float(np.dot(tangent, waist_u))), -1.0, 1.0))))
        shape_class = str(geom.get("shape_class", ""))
        length = float(geom.get("arc_length_mm", 0.0)) if shape_class == "CURVE" else float(geom.get("major_length_mm", 0.0))
        vals = heat.heatmap[comp > 0].astype(np.float32)
        mean_response = float(np.mean(vals)) if vals.size else 0.0
        max_response = float(np.max(vals)) if vals.size else 0.0
        if parallel_error > parallel_limit or length < min_length:
            continue
        score = length * math.sqrt(max(1.0, area)) * max(0.1, mean_response / 255.0)
        candidates.append({
            "length_mm": length,
            "area_px": area,
            "mean_response": mean_response,
            "max_response": max_response,
            "parallel_error_deg": parallel_error,
            "score": score,
        })

    out["component_count"] = len(candidates)
    out["threshold"] = threshold
    if not candidates:
        out["reason"] = "no reliable parallel waistband component"
        return out
    best = max(candidates, key=lambda x: float(x["score"]))
    out.update(best)
    out["valid"] = True
    out["reason"] = "pose-relative waistband component"
    return out


def _d24v2_candidate_metrics(cand: Dict[str, Any], heat: Any, obs: Any, waist_ref: Dict[str, Any]) -> Dict[str, Any]:
    comp = candidate_component_mask(heat, cand)
    vals = heat.heatmap[comp > 0].astype(np.float32) if comp is not None else np.asarray([], np.float32)
    geom = cand.get("d21_geometry", {}) or {}
    shape_class = str(geom.get("shape_class", ""))
    length = float(geom.get("arc_length_mm", cand.get("major_length_mm", 0.0))) if shape_class == "CURVE" else float(cand.get("major_length_mm", 0.0))
    area = float(cand.get("area_px", 0.0))
    mask_area = max(1.0, float(obs.mask.area_px))
    waist_width = max(1.0, float(waist_ref.get("waist_width_mm", getattr(obs.pose, "waist_width_mm", 1.0))))
    return {
        "length_mm": length,
        "area_px": area,
        "area_ratio": area / mask_area,
        "mean_response": float(np.mean(vals)) if vals.size else float(cand.get("mean_score", 0.0)),
        "max_response": float(np.max(vals)) if vals.size else float(cand.get("max_score", 0.0)),
        "severity": float(cand.get("d21_severity", 0.0)),
        "waist_span_ratio": length / waist_width,
        "tier": int(cand.get("d20_priority_tier", 0)),
    }


def _d24v2_relative_and_major(metrics: Dict[str, Any], waist_ref: Dict[str, Any], args) -> Tuple[bool, bool, List[str]]:
    smaller_than_waist = False
    if bool(waist_ref.get("valid", False)):
        smaller_than_waist = bool(
            float(metrics["length_mm"]) < float(waist_ref["length_mm"]) * float(args.d24v2_waist_length_ratio_ignore)
            and float(metrics["area_px"]) < float(waist_ref["area_px"]) * float(args.d24v2_waist_area_ratio_ignore)
            and float(metrics["mean_response"]) < float(waist_ref["mean_response"]) * float(args.d24v2_waist_response_ratio_ignore)
        )

    reasons: List[str] = []
    if float(metrics["length_mm"]) >= float(args.d24v2_major_length_mm):
        reasons.append("long")
    if float(metrics["area_ratio"]) >= float(args.d24v2_major_area_ratio):
        reasons.append("large_area")
    if (
        float(metrics["severity"]) >= float(args.d24v2_major_severity)
        and float(metrics["length_mm"]) >= float(args.d24v2_major_severity_min_length_mm)
    ):
        reasons.append("severe")
    if (
        float(metrics["waist_span_ratio"]) >= float(args.d24v2_major_waist_span_ratio)
        and float(metrics["area_ratio"]) >= float(args.d24v2_major_waist_span_min_area_ratio)
    ):
        reasons.append("cross_span")
    major = bool(reasons)
    return smaller_than_waist, major, reasons


def evaluate_d24(obs: Any, heat: Any, H: np.ndarray, state: D24State, args) -> Dict[str, Any]:
    state.evaluation_count += 1
    quality = observation_quality_report(obs, args)

    if not quality["good"]:
        state.ready_streak = 0
        return {
            "status": "REJUDGE", "reason": "OBSERVATION_INVALID",
            "evaluation": state.evaluation_count, "quality": quality,
            "stability": {"good": False, "reason": "not appended"},
            "shape": {}, "wrinkle": {},
        }

    state.history.append(observation_entry(obs))
    stability = temporal_stability_report(state.history, args)
    if not stability["good"]:
        state.ready_streak = 0
        return {
            "status": "REJUDGE", "reason": "OBSERVATION_UNSTABLE",
            "evaluation": state.evaluation_count, "quality": quality,
            "stability": stability, "shape": {}, "wrinkle": {},
        }

    # D24-v2 hard gate 1: pants geometry/silhouette must look spread.
    shape = d23._d19_finish_shape_report(obs, args, H)
    if not bool(shape.get("shape_good", False)):
        state.ready_streak = 0
        return {
            "status": "NOT_READY_SHAPE", "reason": "PANTS_SHAPE_FAILED",
            "evaluation": state.evaluation_count, "quality": quality,
            "stability": stability, "shape": shape, "wrinkle": {},
        }

    if heat is None:
        state.ready_streak = 0
        return {
            "status": "REJUDGE", "reason": "HEATMAP_UNAVAILABLE",
            "evaluation": state.evaluation_count, "quality": quality,
            "stability": stability, "shape": shape, "wrinkle": {},
        }

    robust = robust_heat_threshold(heat, args)
    waist_ref = _d24v2_waist_reference(obs, heat, H, robust, args)
    candidates = list(getattr(heat, "d21v4_all_candidates", [])) or list(getattr(heat, "candidates", []))

    major_blockers: List[Dict[str, Any]] = []
    relative_allowed: List[Dict[str, Any]] = []
    non_major_allowed: List[Dict[str, Any]] = []
    t1: List[Dict[str, Any]] = []
    t2: List[Dict[str, Any]] = []
    seam_ignored: List[Dict[str, Any]] = []
    robust_rejected: List[Dict[str, Any]] = []

    for original in candidates:
        cand = dict(original)
        seam = seam_veto_report(cand, heat, obs, H, args)
        cand["d24_seam"] = seam
        if bool(seam.get("matched", False)):
            seam_ignored.append(cand)
            continue

        support = candidate_robust_support(cand, heat, float(robust["threshold"]))
        cand["d24_robust_support"] = support
        if support < float(args.d24_candidate_min_robust_support):
            robust_rejected.append(cand)
            continue

        metrics = _d24v2_candidate_metrics(cand, heat, obs, waist_ref)
        smaller, major, reasons = _d24v2_relative_and_major(metrics, waist_ref, args)
        cand["d24v2_metrics"] = metrics
        cand["d24v2_smaller_than_waist"] = smaller
        cand["d24v2_major"] = major
        cand["d24v2_major_reasons"] = reasons

        if major:
            major_blockers.append(cand)
        elif smaller:
            relative_allowed.append(cand)
        else:
            non_major_allowed.append(cand)

        if int(metrics["tier"]) <= 1:
            t1.append(cand)
        elif int(metrics["tier"]) == 2:
            t2.append(cand)

    # Only major blockers influence READY. Fine/T2 counts and the total heatmap
    # ratio remain visible diagnostics but never keep a well-shaped pair of pants
    # in NOT_READY forever.
    mask_area = max(1.0, float(obs.mask.area_px))
    effective_area = max(1.0, float(cv2.countNonZero(heat.inner_mask)))
    blocker_area = float(sum(float(c.get("area_px", 0.0)) for c in major_blockers))
    blocker_ratio = blocker_area / effective_area
    max_blob_ratio = max([float(c.get("area_px", 0.0)) / mask_area for c in major_blockers] or [0.0])
    max_severity = max([float((c.get("d24v2_metrics", {}) or {}).get("severity", 0.0)) for c in major_blockers] or [0.0])
    checks = {"major_wrinkle_count": len(major_blockers) <= int(args.finish_max_t3_count)}
    wrinkle_good = bool(all(checks.values()))
    wrinkle_report = {
        "policy": "SHAPE_PLUS_MAJOR_WRINKLE_ONLY",
        "good": wrinkle_good,
        "checks": checks,
        "t3_count": len(major_blockers),
        "t2_count": len(t2),
        "t1_allowed_count": len(t1),
        "relative_allowed_count": len(relative_allowed),
        "non_major_allowed_count": len(non_major_allowed),
        "seam_ignored_count": len(seam_ignored),
        "robust_rejected_count": len(robust_rejected),
        "waist_structure_ignored_count": int(getattr(heat, "d23_waist_structure_ignored_count", 0)),
        "fine_ignored_count": int(getattr(heat, "d21v4_fine_ignored_count", 0)),
        "max_severity": max_severity,
        "actionable_ratio": blocker_ratio,
        "max_blob_ratio": max_blob_ratio,
        "waist_reference": waist_ref,
        "robust": robust,
        "limits": {
            "max_major_count": int(args.finish_max_t3_count),
            "major_length_mm": float(args.d24v2_major_length_mm),
            "major_area_ratio": float(args.d24v2_major_area_ratio),
            "waist_length_ignore_ratio": float(args.d24v2_waist_length_ratio_ignore),
            "waist_area_ignore_ratio": float(args.d24v2_waist_area_ratio_ignore),
            "waist_response_ignore_ratio": float(args.d24v2_waist_response_ratio_ignore),
        },
        "remaining": [k for k, v in checks.items() if not v],
        "t1": t1,
        "t2": t2,
        "t3": major_blockers,
        "major_blockers": major_blockers,
        "relative_allowed": relative_allowed,
        "non_major_allowed": non_major_allowed,
        "seam_ignored": seam_ignored,
        "robust_rejected": robust_rejected,
    }

    if not wrinkle_good:
        state.ready_streak = 0
        return {
            "status": "NOT_READY_WRINKLE", "reason": "MAJOR_WRINKLE_REMAINS",
            "evaluation": state.evaluation_count, "quality": quality,
            "stability": stability, "shape": shape, "wrinkle": wrinkle_report,
            "ready_streak": 0, "ready_required": int(args.ready_confirm_count),
        }

    state.ready_streak += 1
    required = max(1, int(args.ready_confirm_count))
    status = "READY_GOOD_ENOUGH" if state.ready_streak >= required else "READY_PENDING"
    return {
        "status": status, "reason": "SHAPE_OK_AND_NO_MAJOR_WRINKLE",
        "evaluation": state.evaluation_count, "quality": quality,
        "stability": stability, "shape": shape, "wrinkle": wrinkle_report,
        "ready_streak": state.ready_streak, "ready_required": required,
    }


# -----------------------------------------------------------------------------
# Visualization / logging
# -----------------------------------------------------------------------------

def put_text(img: np.ndarray, text: str, xy: Tuple[int, int], color, scale=0.55, thickness=1) -> None:
    x, y = xy
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thickness + 3, cv2.LINE_AA)
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def draw_candidate_box(img: np.ndarray, cand: Dict[str, Any], label: str, color) -> None:
    x, y, w, h = [int(round(float(v))) for v in cand.get("bbox", [0, 0, 0, 0])]
    cx, cy = [int(round(float(v))) for v in cand.get("center_px", [x, y])]
    cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
    cv2.circle(img, (cx, cy), 5, color, -1)
    put_text(img, label, (cx + 7, cy - 7), color, scale=0.45, thickness=1)


def draw_d24_overlay(frame: np.ndarray, obs: Any, heat: Any, report: Dict[str, Any], H: np.ndarray, cfg, args) -> np.ndarray:
    # Draw pose/mask without the old yellow candidate boxes.
    base = d23.draw_bottom_overlay(
        frame, H, obs, cfg, plan=None, wrinkle_plan=None, args=None,
        motion_busy=False, motion_name="",
    )

    # Low-alpha heatmap within the valid inner ROI.
    if heat is not None and bool(args.wrinkle_heatmap_draw):
        colored = cv2.applyColorMap(heat.heatmap, cv2.COLORMAP_JET)
        alpha = float(np.clip(args.wrinkle_heatmap_alpha, 0.0, 0.7))
        m = heat.inner_mask > 0
        blended = cv2.addWeighted(base, 1.0 - alpha, colored, alpha, 0.0)
        base[m] = blended[m]

        ignore = getattr(heat, "structure_ignore_mask", None)
        if isinstance(ignore, np.ndarray):
            contours, _ = cv2.findContours((ignore > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(base, contours, -1, (150, 150, 150), 2)
        center_seam = getattr(heat, "center_rise_ignore_mask", None)
        if isinstance(center_seam, np.ndarray) and cv2.countNonZero(center_seam) > 0:
            contours, _ = cv2.findContours((center_seam > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(base, contours, -1, (190, 190, 190), 2)
            ys, xs = np.where(center_seam > 0)
            if len(xs):
                put_text(base, "IGNORE CENTER RISE SEAM", (int(np.mean(xs))+8, int(np.mean(ys))), (190,190,190), 0.42, 1)

    wrinkle = report.get("wrinkle", {}) or {}
    if bool(getattr(args, "d24v5_draw_allowed_fine", False)):
        for i, cand in enumerate(wrinkle.get("t1", []), 1):
            draw_candidate_box(base, cand, f"FINE ALLOW {i}", TIER_COLORS[1])
        for i, cand in enumerate(wrinkle.get("t2", []), 1):
            draw_candidate_box(base, cand, f"WRINKLE ALLOW {i}", TIER_COLORS[2])
    for i, cand in enumerate(wrinkle.get("t3", []), 1):
        draw_candidate_box(base, cand, f"FOLD {i}", TIER_COLORS[3])
    for cand in wrinkle.get("waist_gather_ignored", []):
        draw_candidate_box(base, cand, "IGNORE WAIST GATHER", (170, 170, 170))
    for cand in wrinkle.get("seam_ignored", []):
        name = str((cand.get("d24_seam", {}) or {}).get("name", "SEAM"))
        draw_candidate_box(base, cand, f"IGNORE {name}", (160, 160, 160))

    # Shape contour color.
    shape_good = bool((report.get("shape", {}) or {}).get("shape_good", False))
    if obs is not None and obs.mask is not None:
        cv2.drawContours(base, [obs.mask.contour], -1, (0, 220, 0) if shape_good else (0, 0, 255), 3)

    status = str(report.get("status", "REJUDGE"))
    color = STATUS_COLORS.get(status, (255, 255, 255))
    cv2.rectangle(base, (10, 10), (min(base.shape[1] - 10, 850), 66), (0, 0, 0), -1)
    put_text(base, status, (25, 48), color, scale=1.0, thickness=2)

    return append_status_panel(base, report, args)


def bool_mark(v: Any) -> str:
    return "PASS" if bool(v) else "FAIL"


def append_status_panel(image: np.ndarray, report: Dict[str, Any], args) -> np.ndarray:
    h = max(image.shape[0], 760)
    panel_w = max(520, int(args.panel_width))
    panel = np.zeros((h, panel_w, 3), np.uint8)
    panel[:] = (24, 24, 24)
    y = 34

    def line(text: str, color=(230, 230, 230), scale=0.47, gap=24) -> None:
        nonlocal y
        put_text(panel, text[:100], (18, y), color, scale=scale, thickness=1)
        y += gap

    status = str(report.get("status", "REJUDGE"))
    line(f"D24 FINISH: {status}", STATUS_COLORS.get(status, (255, 255, 255)), 0.72, 34)
    line(f"reason: {report.get('reason', '')}", (210, 210, 210), 0.47, 28)

    quality = report.get("quality", {}) or {}
    line("[1] OBSERVATION", (255, 220, 120), 0.52, 25)
    line(f"quality {bool_mark(quality.get('good'))}  kpts={quality.get('valid_keypoints', 0)}  meanConf={float(quality.get('mean_kpt_conf', 0.0)):.2f}")
    line(f"TTA={quality.get('pose_tta_state', '-')} score={float(quality.get('pose_tta_score', 0.0)):.1f} tested={quality.get('pose_tta_tested', 0)}")

    stability = report.get("stability", {}) or {}
    line("[2] TEMPORAL STABILITY", (255, 220, 120), 0.52, 25)
    line(f"stable {bool_mark(stability.get('good'))} history={stability.get('history_count', 0)}/{stability.get('required', args.history_count)}")
    line(f"areaDelta={float(stability.get('mask_area_rel', 0.0)):.3f}  center={float(stability.get('center_shift_mm', 0.0)):.1f}mm")
    line(f"axisSpread={float(stability.get('pose_axis_spread_deg', 0.0)):.1f}deg")

    shape = report.get("shape", {}) or {}
    sym = shape.get("symmetry", {}) or {}
    hard = shape.get("hard_checks", {}) or {}
    line("[3] PANTS SHAPE", (255, 220, 120), 0.52, 25)
    line(f"shape {bool_mark(shape.get('shape_good'))} score={float(shape.get('shape_score', 0.0)):.2f}")
    line(f"hemGap={float(shape.get('hem_gap_ratio', 0.0)):.2f}  legBalance={float(shape.get('leg_balance', 0.0)):.2f}")
    line(f"crotchAxis={float(shape.get('crotch_axis_offset_ratio', 0.0)):.2f}  bodyAxis={float(shape.get('axis_to_waist_ratio', 0.0)):.2f}")
    line(f"lowerIoU={float(sym.get('lower_reflected_iou', 0.0)):.2f}  widthMed={float(sym.get('lower_width_balance_median', 0.0)):.2f}")
    line(f"widthMin={float(sym.get('lower_width_balance_min', 0.0)):.2f}  symmetry={float(sym.get('symmetry_score', 0.0)):.2f}")
    if shape.get("remaining"):
        line("fail: " + ", ".join(str(x) for x in shape.get("remaining", [])), (80, 120, 255), 0.43, 23)

    wrinkle = report.get("wrinkle", {}) or {}
    line("[4] SHAPE + FOLD ONLY", (255, 220, 120), 0.52, 25)
    line(f"foldGate {bool_mark(wrinkle.get('good'))}  FOLD={wrinkle.get('fold_count', wrinkle.get('t3_count', 0))}  fineAllowed={wrinkle.get('fine_allowed_count', 0)}")
    line(f"waistGatherIgnored={wrinkle.get('waist_gather_ignored_count', 0)}  seamIgnored={wrinkle.get('seam_ignored_count', 0)}")
    line(f"centerRiseIgnoredPx={wrinkle.get('center_rise_ignored_px', 0)}  brightTextureLargeArea=diagnosticOnly", (190,190,190), 0.40, 23)
    line(f"foldRatio={float(wrinkle.get('actionable_ratio', 0.0)):.3f}  maxFoldBlob={float(wrinkle.get('max_blob_ratio', 0.0)):.3f}")
    line(f"waistBandVeto={wrinkle.get('waist_structure_ignored_count', 0)}  robustReject={wrinkle.get('robust_rejected_count', 0)}")
    waist_ref = wrinkle.get("waist_reference", {}) or {}
    line(f"waistRef={bool_mark(waist_ref.get('valid'))} len={float(waist_ref.get('length_mm',0.0)):.1f}mm area={float(waist_ref.get('area_px',0.0)):.0f}px")
    line(f"waistResp={float(waist_ref.get('mean_response',0.0)):.1f} reason={str(waist_ref.get('reason',''))[:36]}", (190,190,190), 0.40, 23)
    robust = wrinkle.get("robust", {}) or {}
    line(f"robustThr={float(robust.get('threshold', 0.0)):.1f} med={float(robust.get('median', 0.0)):.1f} sigma={float(robust.get('sigma', 0.0)):.1f}")
    if wrinkle.get("remaining"):
        line("fail: " + ", ".join(str(x) for x in wrinkle.get("remaining", [])), (80, 120, 255), 0.43, 23)

    line("[5] CONFIRMATION", (255, 220, 120), 0.52, 25)
    line(f"READY streak={report.get('ready_streak', 0)}/{report.get('ready_required', args.ready_confirm_count)}")
    line("Keys: SPACE evaluate | C reset | S save | L lock H | R reset H | Q quit", (180, 180, 180), 0.39, 24)

    canvas = np.zeros((h, image.shape[1] + panel_w, 3), np.uint8)
    canvas[:image.shape[0], :image.shape[1]] = image
    canvas[:, image.shape[1]:] = panel
    return canvas


def json_safe(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items() if k not in {"t1", "t2", "t3", "major_blockers", "fold_blockers", "waist_gather_ignored", "fine_allowed", "relative_allowed", "non_major_allowed", "seam_ignored", "robust_rejected"}}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    return value


def append_jsonl(path: str, report: Dict[str, Any]) -> None:
    if not path:
        return
    record = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "unix_time": time.time(),
        **json_safe(report),
    }
    try:
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception as exc:
        print(f"[D25-LOG-WARN] {exc!r}")


def save_sample(state: D24State, args) -> None:
    if state.latest_overlay is None:
        print("[D25-SAVE] no overlay yet")
        return
    out_dir = Path(args.snapshot_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    stem = f"d24_{ts}_{state.latest_status}"
    image_path = out_dir / f"{stem}.jpg"
    json_path = out_dir / f"{stem}.json"
    cv2.imwrite(str(image_path), state.latest_overlay)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_safe(state.latest_report), f, ensure_ascii=False, indent=2)
    print(f"[D25-SAVE] image={image_path} report={json_path}")


# -----------------------------------------------------------------------------
# Evaluation worker / main loop
# -----------------------------------------------------------------------------

def median_frame(frames: Sequence[np.ndarray]) -> np.ndarray:
    if len(frames) == 1:
        return frames[0].copy()
    stack = np.stack(frames, axis=0).astype(np.float32)
    return np.median(stack, axis=0).astype(np.uint8)


def make_cfg(config: Dict[str, Any]) -> Any:
    cfg = d23.BottomSafetyConfig()
    xmin, xmax, ymin, ymax = d23.board_bounds_from_config(config)
    cfg.board_x_min = xmin
    cfg.board_x_max = xmax
    cfg.board_y_min = ymin
    cfg.board_y_max = ymax
    cfg.split_board_x = float(config.get("dual_roarm", {}).get("split_board_x", 247.0))
    return cfg


def evaluate_frame(frame: np.ndarray, H: np.ndarray, seg_model, pose_model, state: D24State, args, cfg) -> Tuple[Any, Any, Dict[str, Any], np.ndarray]:
    obs = d23.infer_bottom_observation(seg_model, pose_model, frame, H, args, cfg)
    heat = None
    if obs is not None and obs.mask is not None:
        heat = d23.build_d21_wrinkle_heatmap(frame, obs, H, args)
    report = evaluate_d24(obs, heat, H, state, args)
    overlay = draw_d24_overlay(frame, obs, heat, report, H, cfg, args)
    return obs, heat, report, overlay


def print_report(report: Dict[str, Any]) -> None:
    status = report.get("status", "REJUDGE")
    q = report.get("quality", {}) or {}
    st = report.get("stability", {}) or {}
    sh = report.get("shape", {}) or {}
    sy = sh.get("symmetry", {}) or {}
    wr = report.get("wrinkle", {}) or {}
    print(
        f"[D25] status={status} reason={report.get('reason')} "
        f"kpts={q.get('valid_keypoints',0)} conf={float(q.get('mean_kpt_conf',0.0)):.2f} "
        f"stable={st.get('good',False)} areaDelta={float(st.get('mask_area_rel',0.0)):.3f} "
        f"centerShift={float(st.get('center_shift_mm',0.0)):.1f}mm "
        f"shape={float(sh.get('shape_score',0.0)):.2f} "
        f"lowerIoU={float(sy.get('lower_reflected_iou',0.0)):.2f} "
        f"widthMin={float(sy.get('lower_width_balance_min',0.0)):.2f} "
        f"MAJOR={wr.get('t3_count',0)} T2shown={wr.get('t2_count',0)} waistAllowed={wr.get('relative_allowed_count',0)} "
        f"majorRatio={float(wr.get('actionable_ratio',0.0)):.3f} maxMajorBlob={float(wr.get('max_blob_ratio',0.0)):.3f} "
        f"ready={report.get('ready_streak',0)}/{report.get('ready_required','-')}"
    )


def detect_homography(frame: np.ndarray, detector, marker_board_mm, required_ids) -> Tuple[Optional[np.ndarray], Dict[int, np.ndarray]]:
    corners, ids, _ = d23.detect_markers(frame, detector)
    return d23.compute_homography(corners, ids, marker_board_mm, required_ids)


def run_single_image(args, config, cfg, H, seg_model, pose_model) -> int:
    frame = cv2.imread(args.image)
    if frame is None:
        print(f"[D25-ERROR] cannot read image: {args.image}")
        return 2
    if H is None:
        print("[D25-ERROR] Homography is required for --image. Load --hfile first.")
        return 2
    state = D24State(history=collections.deque(maxlen=max(2, int(args.history_count))))
    # Repeat the same image only for deterministic offline inspection. Mark the
    # report as one-shot; live temporal stability should be tested with a camera.
    for _ in range(max(2, int(args.history_count))):
        obs, heat, report, overlay = evaluate_frame(frame, H, seg_model, pose_model, state, args, cfg)
    state.latest_overlay = overlay
    state.latest_report = report
    state.latest_status = str(report.get("status", "REJUDGE"))
    cv2.imwrite(args.output, overlay)
    print_report(report)
    print(f"[D25] saved: {args.output}")
    return 0


def main() -> int:
    args = finalize_args(build_parser().parse_args())
    print(f"[BUILD] {STEP_D24_BUILD}")
    print("[D25-SAFETY] perception-only: no serial port, no robot command, no low-Z motion")
    print(
        f"[D25-SHAPE] score>={args.finish_shape_score_min:.2f} "
        f"hemGap={args.finish_hem_gap_ratio_min:.2f}~{args.finish_hem_gap_ratio_max:.2f} "
        f"legBalance>={args.finish_leg_balance_min:.2f} lowerIoU>={args.d22_lower_reflected_iou_min:.2f}"
    )
    print(
        f"[D24V6-FINISH] only broad contrast-supported folds block READY: count<={args.finish_max_t3_count} "
        f"length>={args.d24v2_major_length_mm:.0f}mm OR areaRatio>={args.d24v2_major_area_ratio:.3f}; "
        f"waist-relative ignore len<{args.d24v2_waist_length_ratio_ignore:.2f} area<{args.d24v2_waist_area_ratio_ignore:.2f} response<{args.d24v2_waist_response_ratio_ignore:.2f}"
    )
    print(
        f"[D24V6-STRUCTURE] waistband/crotch + radial waist-gather ignore; conditionalSeam={args.d24_seam_veto} "
        f"seamSupport>={args.d24_seam_min_support_ratio:.2f}"
    )

    config = d23.load_json_if_exists(args.config) or {
        "aruco": {
            "dictionary": "DICT_4X4_50",
            "required_ids": [0, 1, 2, 3],
            "marker_board_mm": d23.DEFAULT_MARKER_BOARD_MM,
        },
        "dual_roarm": {"split_board_x": 247.0},
    }
    cfg = make_cfg(config)
    H = d23.load_homography(args.hfile) if args.load_h else None
    if H is not None:
        print(f"[H] loaded: {args.hfile}")

    seg_model, pose_model = d23.load_models(args)

    if args.image:
        return run_single_image(args, config, cfg, H, seg_model, pose_model)

    detector = d23.make_aruco_detector(
        d23.get_dictionary(config.get("aruco", {}).get("dictionary", "DICT_4X4_50"))
    )
    required_ids = config.get("aruco", {}).get("required_ids", [0, 1, 2, 3])
    marker_board_mm = config.get("aruco", {}).get("marker_board_mm", d23.DEFAULT_MARKER_BOARD_MM)

    cap = d23.open_camera(args)
    d23.configure_usb_camera(args)
    for _ in range(max(0, int(args.warmup_frames))):
        cap.read()
        time.sleep(0.02)

    frame_buffer: Deque[np.ndarray] = collections.deque(maxlen=max(1, int(args.burst_frame_count)))
    state = D24State(history=collections.deque(maxlen=max(2, int(args.history_count))))
    state.latest_h = H
    state_lock = threading.RLock()
    worker_lock = threading.Lock()
    stop_event = threading.Event()
    last_schedule = 0.0

    def worker(frames: List[np.ndarray], H_snapshot: np.ndarray) -> None:
        nonlocal last_schedule
        try:
            med = median_frame(frames)
            with state_lock:
                state.last_eval_started = time.time()
            obs, heat, report, overlay = evaluate_frame(
                med, H_snapshot, seg_model, pose_model, state, args, cfg
            )
            with state_lock:
                state.latest_obs = obs
                state.latest_heat = heat
                state.latest_report = report
                state.latest_status = str(report.get("status", "REJUDGE"))
                state.latest_overlay = overlay
                state.latest_eval_frame = med
                state.latest_h = H_snapshot.copy()
            print_report(report)
            append_jsonl(args.log_jsonl, report)
        except Exception as exc:
            print(f"[D25-WORKER-ERROR] {exc!r}")
            with state_lock:
                state.ready_streak = 0
                state.latest_status = "REJUDGE"
                state.latest_report = {
                    "status": "REJUDGE",
                    "reason": f"worker error: {exc!r}",
                }
        finally:
            worker_lock.release()

    try:
        while not stop_event.is_set():
            ok, frame = cap.read()
            if not ok or frame is None:
                time.sleep(0.03)
                continue
            frame_buffer.append(frame.copy())

            H_candidate, marker_centers = detect_homography(frame, detector, marker_board_mm, required_ids)
            if H is None and bool(args.auto_lock) and H_candidate is not None:
                H = H_candidate
                with state_lock:
                    state.latest_h = H.copy()
                if bool(args.save_h_on_lock):
                    d23.save_homography(args.hfile, H)
                print("[H] auto-locked from ArUco markers")

            now = time.time()
            with state_lock:
                force = bool(state.force_evaluate)
                state.force_evaluate = False
                paused = bool(state.paused)
            due = force or (now - last_schedule >= max(0.20, float(args.eval_interval_s)))
            if (
                not paused and due and H is not None
                and len(frame_buffer) >= max(1, int(args.burst_frame_count))
                and worker_lock.acquire(blocking=False)
            ):
                last_schedule = now
                frames = [f.copy() for f in frame_buffer]
                threading.Thread(target=worker, args=(frames, H.copy()), daemon=True).start()

            with state_lock:
                shown = state.latest_overlay.copy() if state.latest_overlay is not None else frame.copy()
                latest_report = dict(state.latest_report)

            if state.latest_overlay is None:
                status = "WAITING FOR FIRST EVALUATION" if H is not None else "LOCK HOMOGRAPHY: show IDs 0,1,2,3"
                put_text(shown, status, (25, 45), (0, 220, 255), scale=0.75, thickness=2)

            if not args.no_window:
                cv2.imshow(args.window_name, shown)
                key = cv2.waitKey(1) & 0xFF
            else:
                key = 255

            terminal_key = d23.read_terminal_key()
            if terminal_key != 255:
                key = terminal_key

            if key in (27, ord("q"), ord("Q")):
                break
            if key == ord(" "):
                with state_lock:
                    state.force_evaluate = True
            elif key in (ord("c"), ord("C")):
                with state_lock:
                    state.reset_temporal()
                print("[D25] temporal history and READY streak cleared")
            elif key in (ord("s"), ord("S")):
                with state_lock:
                    save_sample(state, args)
            elif key in (ord("p"), ord("P")):
                with state_lock:
                    state.paused = not state.paused
                    print(f"[D25] paused={state.paused}")
            elif key in (ord("l"), ord("L")):
                if H_candidate is None:
                    print(f"[H] lock failed; visible marker IDs={sorted(marker_centers.keys())}")
                else:
                    H = H_candidate
                    with state_lock:
                        state.latest_h = H.copy()
                        state.reset_temporal()
                    if bool(args.save_h_on_lock):
                        d23.save_homography(args.hfile, H)
                    print("[H] locked/updated")
            elif key in (ord("r"), ord("R")):
                H = None
                with state_lock:
                    state.latest_h = None
                    state.reset_temporal()
                print("[H] reset in memory")

            if args.once and state.evaluation_count >= max(2, int(args.history_count)):
                # Save the most recent state after enough history exists.
                with state_lock:
                    if state.latest_overlay is not None:
                        cv2.imwrite(args.output, state.latest_overlay)
                        print(f"[D25] once overlay saved: {args.output}")
                break

    finally:
        stop_event.set()
        cap.release()
        if not args.no_window:
            cv2.destroyAllWindows()

    return 0




# =============================================================================
# D24-v3: A119 heatmap transplant + frozen snapshot evaluation
# =============================================================================

@dataclass
class A119BottomHeatmapPlan:
    """A119-style heatmap container compatible with D24-v2 visualization helpers."""
    heatmap: np.ndarray
    inner_mask: np.ndarray
    binary: np.ndarray
    candidates: List[Dict[str, Any]]
    threshold: float
    wrinkle_ratio: float
    status: str
    summary: str
    ready_heatmap: np.ndarray
    ready_inner_mask: np.ndarray
    ready_threshold: float
    ready_roi_valid: bool
    d21v4_all_candidates: List[Dict[str, Any]] = field(default_factory=list)
    d21v4_residual_binary: Optional[np.ndarray] = None
    structure_ignore_mask: Optional[np.ndarray] = None
    structure_info: Dict[str, Any] = field(default_factory=dict)
    d23_waist_structure_ignored_count: int = 0
    d21v4_fine_ignored_count: int = 0
    highpass_abs: Optional[np.ndarray] = None
    center_rise_ignore_mask: Optional[np.ndarray] = None
    center_rise_info: Dict[str, Any] = field(default_factory=dict)


def _a119_odd_ksize(value: int, minimum: int = 3) -> int:
    k = max(int(minimum), int(value))
    return k if k % 2 == 1 else k + 1


def _a119_normalize_u8(arr: np.ndarray, mask: Optional[np.ndarray]) -> np.ndarray:
    """A119 2~98 percentile normalization inside the selected garment ROI."""
    a = np.asarray(arr, dtype=np.float32)
    out = np.zeros(a.shape, dtype=np.uint8)
    if mask is None:
        valid = np.isfinite(a)
    else:
        valid = (np.asarray(mask) > 0) & np.isfinite(a)
    if not np.any(valid):
        return out
    vals = a[valid]
    lo = float(np.percentile(vals, 2.0))
    hi = float(np.percentile(vals, 98.0))
    if hi <= lo + 1e-6:
        return out
    scaled = np.clip((a - lo) * (255.0 / (hi - lo)), 0.0, 255.0).astype(np.uint8)
    scaled[~valid] = 0
    return scaled


def _a119_px_per_mm(H: np.ndarray, center_board: Sequence[float]) -> Optional[float]:
    try:
        c = np.asarray(center_board, dtype=np.float32).reshape(2)
        p0 = d23.board_to_pixel(H, float(c[0]), float(c[1]))
        p1 = d23.board_to_pixel(H, float(c[0] + 50.0), float(c[1]))
        p2 = d23.board_to_pixel(H, float(c[0]), float(c[1] + 50.0))
        scales: List[float] = []
        if p0 is not None and p1 is not None:
            scales.append(float(np.linalg.norm(np.asarray(p1) - np.asarray(p0))) / 50.0)
        if p0 is not None and p2 is not None:
            scales.append(float(np.linalg.norm(np.asarray(p2) - np.asarray(p0))) / 50.0)
        return float(np.median(scales)) if scales else None
    except Exception:
        return None


def _a119_inner_roi(mask_u8: np.ndarray, px_per_mm: Optional[float], margin_mm: float,
                    min_keep_ratio: float = 0.10) -> Tuple[np.ndarray, float]:
    mask = ((np.asarray(mask_u8) > 0).astype(np.uint8) * 255)
    if margin_mm <= 0.0 or px_per_mm is None or px_per_mm <= 1e-6:
        return mask, 0.0
    dist = cv2.distanceTransform((mask > 0).astype(np.uint8), cv2.DIST_L2, 5)
    base = int(cv2.countNonZero(mask))
    min_keep = max(200, int(base * float(np.clip(min_keep_ratio, 0.02, 0.95))))
    for factor in (1.0, 0.80, 0.60, 0.45):
        margin_px = max(1, int(round(float(margin_mm) * factor * float(px_per_mm))))
        roi = (((dist >= float(margin_px)) & (mask > 0)).astype(np.uint8) * 255)
        if int(cv2.countNonZero(roi)) >= min_keep:
            return roi, float(margin_mm) * factor
    return mask, 0.0


def _a119_component_geometry(component: np.ndarray, H: np.ndarray) -> Dict[str, Any]:
    ys, xs = np.where(component > 0)
    if len(xs) < 5:
        return {
            "center_px": [0.0, 0.0], "major_axis_px": [1.0, 0.0],
            "minor_axis_px": [0.0, 1.0], "major_length_px": 0.0,
            "minor_length_px": 1.0, "major_length_mm": 0.0,
            "minor_length_mm": 1.0, "linearity": 1.0,
            "tangent_board": [1.0, 0.0], "normal_board": [0.0, 1.0],
        }
    pts_px = np.column_stack([xs, ys]).astype(np.float32)
    center_px = pts_px.mean(axis=0)
    centered_px = pts_px - center_px.reshape(1, 2)
    try:
        vals, vecs = np.linalg.eigh(np.cov(centered_px.T))
        major_axis_px = vecs[:, int(np.argmax(vals))].astype(np.float32)
        major_axis_px /= max(1e-6, float(np.linalg.norm(major_axis_px)))
    except Exception:
        major_axis_px = np.asarray([1.0, 0.0], np.float32)
    minor_axis_px = np.asarray([-major_axis_px[1], major_axis_px[0]], np.float32)
    pmaj_px = centered_px @ major_axis_px
    pmin_px = centered_px @ minor_axis_px
    major_len_px = float(np.ptp(pmaj_px)) if pmaj_px.size else 0.0
    minor_len_px = float(np.ptp(pmin_px)) if pmin_px.size else 1.0

    sample = pts_px
    if len(sample) > 2500:
        sample = sample[::max(1, int(math.ceil(len(sample) / 2500.0)))]
    try:
        pts_board = cv2.perspectiveTransform(sample.reshape(-1, 1, 2), H).reshape(-1, 2)
        center_board = pts_board.mean(axis=0)
        centered_board = pts_board - center_board.reshape(1, 2)
        vals_b, vecs_b = np.linalg.eigh(np.cov(centered_board.T))
        tangent = vecs_b[:, int(np.argmax(vals_b))].astype(np.float32)
        tangent /= max(1e-6, float(np.linalg.norm(tangent)))
        normal = np.asarray([-tangent[1], tangent[0]], np.float32)
        pmaj = centered_board @ tangent
        pmin = centered_board @ normal
        major_len_mm = float(np.ptp(pmaj)) if pmaj.size else 0.0
        minor_len_mm = float(np.ptp(pmin)) if pmin.size else 1.0
    except Exception:
        center_board = np.asarray([0.0, 0.0], np.float32)
        tangent = np.asarray([1.0, 0.0], np.float32)
        normal = np.asarray([0.0, 1.0], np.float32)
        major_len_mm = 0.0
        minor_len_mm = 1.0

    return {
        "center_px": [float(center_px[0]), float(center_px[1])],
        "center_board": [float(center_board[0]), float(center_board[1])],
        "major_axis_px": [float(major_axis_px[0]), float(major_axis_px[1])],
        "minor_axis_px": [float(minor_axis_px[0]), float(minor_axis_px[1])],
        "major_length_px": major_len_px,
        "minor_length_px": minor_len_px,
        "major_length_mm": major_len_mm,
        "minor_length_mm": minor_len_mm,
        "linearity": float(major_len_mm / max(1.0, minor_len_mm)),
        "tangent_board": [float(tangent[0]), float(tangent[1])],
        "normal_board": [float(normal[0]), float(normal[1])],
    }


def _d24v6_center_rise_structure_mask(obs: Any, H: np.ndarray, image_shape, args) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Return a narrow pose-relative center-rise seam mask.

    The mask follows waist_center -> crotch in board coordinates, so it is
    independent of the pants rotation in the camera image. Only a narrow strip
    is erased; a genuinely broad fold crossing the seam remains as residual
    components on one or both sides after CCA is rebuilt.
    """
    out = np.zeros(image_shape[:2], dtype=np.uint8)
    info: Dict[str, Any] = {"valid": False, "ignored_px": 0, "reason": "disabled"}
    if not bool(getattr(args, "d24v6_center_rise_ignore", True)):
        return out, info
    if obs is None or obs.pose is None or not obs.pose.valid or H is None:
        info["reason"] = "pose invalid"
        return out, info
    wc = np.asarray(obs.pose.waist_center, np.float32).reshape(2)
    cr = np.asarray(obs.pose.crotch, np.float32).reshape(2)
    axis = cr - wc
    length = float(np.linalg.norm(axis))
    if length < 25.0:
        info["reason"] = "waist-crotch axis too short"
        return out, info
    u = axis / length
    n = np.asarray([-u[1], u[0]], np.float32)
    start_mm = float(np.clip(float(getattr(args, "d24v6_center_rise_start_mm", 48.0)), 0.0, max(0.0, length - 5.0)))
    end_before = max(0.0, float(getattr(args, "d24v6_center_rise_end_before_crotch_mm", 0.0)))
    end_expand = max(0.0, float(getattr(args, "d24v6_center_rise_end_expand_mm", 12.0)))
    half_w = max(4.0, float(getattr(args, "d24v6_center_rise_half_width_mm", 18.0)))
    start = wc + u * start_mm
    end = cr - u * end_before + u * end_expand
    poly = [start - n * half_w, start + n * half_w, end + n * half_w, end - n * half_w]
    out = d23._d19_board_polygon_to_mask(H, poly, image_shape)
    if obs.mask is not None and obs.mask.mask_u8.shape[:2] == out.shape[:2]:
        out = cv2.bitwise_and(out, obs.mask.mask_u8)
    info.update({
        "valid": True,
        "reason": "pose-relative center-rise seam",
        "ignored_px": int(cv2.countNonZero(out)),
        "axis_length_mm": length,
        "start_mm": start_mm,
        "half_width_mm": half_w,
        "start_board": [float(start[0]), float(start[1])],
        "end_board": [float(end[0]), float(end[1])],
    })
    return out, info


def _d24v6_component_contrast(cand: Dict[str, Any], heat: Any, args) -> Dict[str, float]:
    """Measure original-scale local contrast rather than normalized heat only."""
    comp = candidate_component_mask(heat, cand)
    hp = getattr(heat, "highpass_abs", None)
    if comp is None or not isinstance(hp, np.ndarray) or hp.shape[:2] != comp.shape[:2]:
        return {"mean_abs_contrast": 0.0, "contrast_support": 0.0}
    vals = hp[comp > 0].astype(np.float32)
    if vals.size == 0:
        return {"mean_abs_contrast": 0.0, "contrast_support": 0.0}
    pix_thr = float(getattr(args, "d24v6_fold_contrast_pixel_threshold", 8.0))
    return {
        "mean_abs_contrast": float(np.mean(vals)),
        "contrast_support": float(np.count_nonzero(vals >= pix_thr)) / float(vals.size),
    }


def build_a119_bottom_heatmap(frame: np.ndarray, obs: Any, H: np.ndarray, args) -> Optional[A119BottomHeatmapPlan]:
    """Transplant A119's heatmap and READY CCA front-end to bottom garments.

    A119 channels:
      30% high-pass shading + 30% Sobel magnitude + 20% Laplacian
      + 20% max response from four Gabor orientations.

    D24-v3 then applies the bottom-specific pose structure mask before connected
    component analysis. The raw heatmap remains available for waistband-relative
    comparison, while candidates are generated only from the cleaned binary.
    """
    if frame is None or obs is None or obs.mask is None or H is None:
        return None
    mask_u8 = ((obs.mask.mask_u8 > 0).astype(np.uint8) * 255)
    if cv2.countNonZero(mask_u8) < 200:
        return None

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame.copy()
    clahe = cv2.createCLAHE(
        clipLimit=max(0.1, float(args.wrinkle_heatmap_clahe_clip)),
        tileGridSize=(max(2, int(args.wrinkle_heatmap_clahe_tile)),) * 2,
    )
    eq = clahe.apply(gray)
    blur_k = _a119_odd_ksize(int(args.wrinkle_heatmap_blur_ksize), 7)
    illum = cv2.GaussianBlur(eq, (blur_k, blur_k), 0)
    high = cv2.absdiff(eq, illum).astype(np.float32)
    sx = cv2.Sobel(eq, cv2.CV_32F, 1, 0, ksize=3)
    sy = cv2.Sobel(eq, cv2.CV_32F, 0, 1, ksize=3)
    grad = cv2.magnitude(sx, sy)
    lap = np.abs(cv2.Laplacian(eq, cv2.CV_32F, ksize=3))

    gabor_resp = np.zeros_like(grad, dtype=np.float32)
    if bool(args.wrinkle_heatmap_gabor):
        gk = _a119_odd_ksize(int(args.wrinkle_heatmap_gabor_ksize), 7)
        sigma = max(0.5, float(args.wrinkle_heatmap_gabor_sigma))
        lambd = max(2.0, float(args.wrinkle_heatmap_gabor_lambda))
        for theta in (0.0, np.pi / 4.0, np.pi / 2.0, 3.0 * np.pi / 4.0):
            kernel = cv2.getGaborKernel((gk, gk), sigma, theta, lambd, gamma=0.45, psi=0, ktype=cv2.CV_32F)
            resp = cv2.filter2D(eq, cv2.CV_32F, kernel)
            gabor_resp = np.maximum(gabor_resp, np.abs(resp))

    center_board = np.asarray(obs.mask.center_board, np.float32).reshape(2)
    px_per_mm = _a119_px_per_mm(H, center_board)
    # A119 READY branch uses a wider 15 mm inner region than its 30 mm action ROI.
    ready_margin_mm = 15.0
    inner_mask, used_margin_mm = _a119_inner_roi(mask_u8, px_per_mm, ready_margin_mm, 0.10)

    high_n = _a119_normalize_u8(high, inner_mask)
    grad_n = _a119_normalize_u8(grad, inner_mask)
    lap_n = _a119_normalize_u8(lap, inner_mask)
    gabor_n = _a119_normalize_u8(gabor_resp, inner_mask) if bool(args.wrinkle_heatmap_gabor) else np.zeros_like(high_n)
    combined = (
        0.30 * high_n.astype(np.float32)
        + 0.30 * grad_n.astype(np.float32)
        + 0.20 * lap_n.astype(np.float32)
        + 0.20 * gabor_n.astype(np.float32)
    )
    heat = _a119_normalize_u8(combined, inner_mask)
    heat[inner_mask <= 0] = 0
    vals = heat[inner_mask > 0]
    if vals.size == 0:
        return None
    threshold = max(
        float(args.wrinkle_heatmap_score_min),
        float(np.percentile(vals, np.clip(float(args.wrinkle_heatmap_percentile), 50.0, 99.5))),
    )

    # A119 READY-only spatial median + opening + dilation.
    heat_ready = cv2.medianBlur(heat, 3)
    heat_ready[inner_mask <= 0] = 0
    binary_pre = np.zeros_like(heat_ready, np.uint8)
    binary_pre[(heat_ready >= threshold) & (inner_mask > 0)] = 255
    binary_pre = cv2.morphologyEx(
        binary_pre, cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=1,
    )
    binary_pre = cv2.dilate(
        binary_pre, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=1,
    )

    # Bottom-specific static structure removal: waistband + crotch. Side/inner/hem
    # seams are still conditionally vetoed later using pose-relative line agreement.
    structure_mask, structure_info = d23._d19_pose_ignore_mask(obs, H, args)
    if structure_mask.shape != binary_pre.shape:
        structure_mask = cv2.resize(structure_mask, (binary_pre.shape[1], binary_pre.shape[0]), interpolation=cv2.INTER_NEAREST)
    center_rise_mask, center_rise_info = _d24v6_center_rise_structure_mask(obs, H, binary_pre.shape, args)
    if center_rise_mask.shape != binary_pre.shape:
        center_rise_mask = cv2.resize(center_rise_mask, (binary_pre.shape[1], binary_pre.shape[0]), interpolation=cv2.INTER_NEAREST)
    structure_mask = cv2.bitwise_or(structure_mask, center_rise_mask)
    structure_info = dict(structure_info or {})
    structure_info["center_rise"] = center_rise_info
    structure_info["center_rise_px"] = int(cv2.countNonZero(center_rise_mask))
    structure_info["ignored_px"] = int(cv2.countNonZero(structure_mask))
    cleaned_binary = binary_pre.copy()
    # D24-v6: structure pixels are removed first; CCA below is rebuilt only on residuals.
    cleaned_binary[structure_mask > 0] = 0

    n, labels, stats, centers = cv2.connectedComponentsWithStats((cleaned_binary > 0).astype(np.uint8), 8)
    min_blob = max(1.0, float(args.wrinkle_heatmap_min_blob_area))
    mask_area = max(1.0, float(obs.mask.area_px))
    candidates: List[Dict[str, Any]] = []
    kept = np.zeros_like(cleaned_binary)
    for label in range(1, n):
        area = float(stats[label, cv2.CC_STAT_AREA])
        if area < min_blob:
            continue
        component = ((labels == label).astype(np.uint8) * 255)
        contours, _ = cv2.findContours(component, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        contour = max(contours, key=cv2.contourArea)
        geom = _a119_component_geometry(component, H)
        x = int(stats[label, cv2.CC_STAT_LEFT])
        y = int(stats[label, cv2.CC_STAT_TOP])
        w = int(stats[label, cv2.CC_STAT_WIDTH])
        h = int(stats[label, cv2.CC_STAT_HEIGHT])
        cx, cy = map(float, centers[label])
        component_vals = heat_ready[component > 0].astype(np.float32)
        mean_score = float(np.mean(component_vals)) if component_vals.size else 0.0
        max_score = float(np.max(component_vals)) if component_vals.size else 0.0
        length_mm = float(geom["major_length_mm"])
        aspect = float(geom["linearity"])
        strong = bool(area >= 120.0 and length_mm >= 80.0 and aspect >= 2.70)
        medium = bool((not strong) and area >= 120.0 and 45.0 <= length_mm < 80.0 and aspect >= 4.00)
        tier = 3 if strong else (2 if medium else 1)
        severity = float(np.clip(
            0.45 * min(1.0, length_mm / 80.0)
            + 0.35 * min(1.0, (area / mask_area) / 0.015)
            + 0.20 * min(1.0, mean_score / 180.0),
            0.0, 1.0,
        ))
        priority = float(mean_score * math.sqrt(max(1.0, area)) * min(4.0, max(1.0, aspect)))
        item = {
            "area_px": area,
            "bbox": [x, y, w, h],
            "center_px": [cx, cy],
            "mean_score": mean_score,
            "max_score": max_score,
            "major_axis_px": geom["major_axis_px"],
            "minor_axis_px": geom["minor_axis_px"],
            "major_length_px": float(geom["major_length_px"]),
            "minor_length_px": float(geom["minor_length_px"]),
            "major_length_mm": length_mm,
            "minor_length_mm": float(geom["minor_length_mm"]),
            "linearity": aspect,
            "priority_score": priority,
            "d20_priority_tier": tier,
            "d21_severity": severity,
            "a119_cca_rule": "strong" if strong else ("medium_thin" if medium else "fine"),
            "d21_geometry": {
                "shape_class": "LINE",
                "center_board": geom["center_board"],
                "tangent_board": geom["tangent_board"],
                "normal_board": geom["normal_board"],
                "arc_length_mm": length_mm,
                "chord_length_mm": length_mm,
                "fit_confidence": float(np.clip((aspect - 1.0) / 4.0, 0.0, 1.0)),
            },
            "contour": contour,
        }
        candidates.append(item)
        kept[component > 0] = 255

    candidates.sort(key=lambda c: (int(c.get("d20_priority_tier", 0)), float(c.get("priority_score", 0.0))), reverse=True)
    # Keep enough components for finish judgment, not only the top action candidates.
    candidates = candidates[:max(12, int(args.wrinkle_heatmap_max_candidates))]
    inner_px = max(1, int(cv2.countNonZero(inner_mask)))
    ratio = float(cv2.countNonZero(kept)) / float(inner_px)
    summary = (
        f"A119 heatmap thr={threshold:.1f} roi={used_margin_mm:.1f}mm "
        f"active={ratio:.3f} blobs={len(candidates)} "
        f"strong={sum(c.get('a119_cca_rule') == 'strong' for c in candidates)} "
        f"medium={sum(c.get('a119_cca_rule') == 'medium_thin' for c in candidates)}"
    )
    return A119BottomHeatmapPlan(
        heatmap=heat_ready,
        inner_mask=inner_mask,
        binary=kept,
        candidates=candidates,
        threshold=float(threshold),
        wrinkle_ratio=ratio,
        status="OK",
        summary=summary,
        ready_heatmap=heat_ready,
        ready_inner_mask=inner_mask,
        ready_threshold=float(threshold),
        ready_roi_valid=True,
        d21v4_all_candidates=candidates,
        d21v4_residual_binary=kept,
        structure_ignore_mask=structure_mask,
        structure_info=structure_info,
        d23_waist_structure_ignored_count=int(structure_info.get("ignored_px", 0) > 0),
        d21v4_fine_ignored_count=sum(int(c.get("d20_priority_tier", 0)) <= 1 for c in candidates),
        highpass_abs=high.astype(np.float32),
        center_rise_ignore_mask=center_rise_mask,
        center_rise_info=center_rise_info,
    )



def _d24v5_waist_gather_report(cand: Dict[str, Any], heat: Any, obs: Any,
                               H: np.ndarray, metrics: Dict[str, Any], args) -> Dict[str, Any]:
    """Recognize elastic-waist gathers without deleting the whole upper pants area.

    A match must be short, remain inside a pose-relative upper band, touch either
    the waistband or the existing waistband-ignore boundary, and point radially
    away from its nearest waist-line anchor. Long/large components fail closed.
    """
    out: Dict[str, Any] = {
        "matched": False,
        "reason": "not waistband gather",
        "depth_mm": 0.0,
        "min_axial_mm": 0.0,
        "max_axial_mm": 0.0,
        "radial_error_deg": 180.0,
        "upper_support_ratio": 0.0,
    }
    if not bool(getattr(args, "d24v5_waist_gather_ignore", True)):
        return {**out, "reason": "disabled"}
    if obs is None or obs.pose is None or not obs.pose.valid or H is None:
        return {**out, "reason": "pose unavailable"}

    comp = candidate_component_mask(heat, cand)
    if comp is None:
        return {**out, "reason": "component unavailable"}
    pts = component_board_points(comp, H, max(300, int(getattr(args, "d24_seam_max_sample_points", 1800))))
    if len(pts) < 8:
        return {**out, "reason": "too few component points"}

    p = obs.pose
    try:
        wl = np.asarray(p.waist_left, np.float32).reshape(2)
        wc = np.asarray(p.waist_center, np.float32).reshape(2)
        wr = np.asarray(p.waist_right, np.float32).reshape(2)
        crotch = np.asarray(p.crotch, np.float32).reshape(2)
    except Exception:
        return {**out, "reason": "waist/crotch unavailable"}

    body_u = d23._safe_unit(crotch - wc)
    waist_u = d23._safe_unit(wr - wl)
    if abs(float(np.dot(body_u, waist_u))) > 0.55:
        waist_u = np.asarray([-body_u[1], body_u[0]], np.float32)
    waist_width = max(1.0, float(np.linalg.norm(wr - wl)))
    waist_to_crotch = max(1.0, float(np.linalg.norm(crotch - wc)))
    depth_limit = min(
        float(args.d24v5_waist_gather_depth_max_mm),
        waist_to_crotch * float(args.d24v5_waist_gather_depth_ratio),
    )
    # D23/D19 already removes a fixed waistband band. A residual gather may start
    # at that erased boundary rather than at axial=0, so allow contact with either.
    erased_forward = max(
        float(getattr(args, "d19_waistband_forward_mm", 0.0)),
        float(getattr(args, "d23_waist_structure_forward_mm", 0.0)),
    )
    connect_limit = min(depth_limit, erased_forward + float(args.d24v5_waist_gather_connect_mm))

    rel = pts - wc.reshape(1, 2)
    axial = rel @ body_u
    lateral = rel @ waist_u
    min_axial = float(np.min(axial))
    max_axial = float(np.max(axial))
    lateral_limit = 0.5 * waist_width + float(args.d24v5_waist_gather_side_expand_mm)
    in_upper = (
        (axial >= -8.0)
        & (axial <= depth_limit)
        & (np.abs(lateral) <= lateral_limit)
    )
    upper_support = float(np.count_nonzero(in_upper)) / float(max(1, len(pts)))

    geom = cand.get("d21_geometry", {}) or {}
    tangent = geom.get("tangent_board")
    if tangent is None:
        tangent = d23._pixel_axis_to_board(H, cand.get("center_px", [0.0, 0.0]), cand.get("major_axis_px", [1.0, 0.0]))
    tangent = d23._safe_unit(np.asarray(tangent if tangent is not None else [1.0, 0.0], np.float32))
    center_board = np.mean(pts, axis=0).astype(np.float32)
    center_rel = center_board - wc
    center_lateral = float(np.clip(np.dot(center_rel, waist_u), -0.5 * waist_width, 0.5 * waist_width))
    waist_anchor = wc + waist_u * center_lateral
    radial = d23._safe_unit(center_board - waist_anchor)
    radial_error = axis_angle_error_deg(tangent, radial)

    length = float(metrics.get("length_mm", 0.0))
    area_ratio = float(metrics.get("area_ratio", 0.0))
    max_length = min(
        float(args.d24v5_waist_gather_max_length_mm),
        waist_width * float(args.d24v5_waist_gather_max_length_ratio),
    )
    checks = {
        "touches_waist_boundary": min_axial <= connect_limit,
        "inside_upper_gather_band": max_axial <= depth_limit + 6.0,
        "upper_support": upper_support >= float(args.d24v5_waist_gather_min_upper_support),
        "radial_direction": radial_error <= float(args.d24v5_waist_gather_radial_angle_deg),
        "short": length <= max_length,
        "small_area": area_ratio <= float(args.d24v5_waist_gather_max_area_ratio),
    }
    matched = bool(all(checks.values()))
    failed = [k for k, v in checks.items() if not v]
    return {
        **out,
        "matched": matched,
        "reason": "waistband-connected radial gather" if matched else ",".join(failed),
        "checks": checks,
        "depth_mm": depth_limit,
        "connect_limit_mm": connect_limit,
        "min_axial_mm": min_axial,
        "max_axial_mm": max_axial,
        "radial_error_deg": radial_error,
        "upper_support_ratio": upper_support,
        "length_mm": length,
        "max_length_mm": max_length,
        "area_ratio": area_ratio,
        "waist_width_mm": waist_width,
        "waist_to_crotch_mm": waist_to_crotch,
        "anchor_board": [float(waist_anchor[0]), float(waist_anchor[1])],
        "center_board": [float(center_board[0]), float(center_board[1])],
    }


def _d24v6_fold_report(cand: Dict[str, Any], metrics: Dict[str, Any], heat: Any, args) -> Dict[str, Any]:
    """Block finish only for broad, contrast-supported physical folds.

    A normalized A119 heatmap can make pale fabric texture look strong. Therefore
    component area alone never blocks READY in v6. A candidate must be broad and
    have original-gray local contrast, unless it is extremely wide and large.
    """
    length = float(metrics.get("length_mm", 0.0))
    width = float(cand.get("minor_length_mm", 0.0))
    area_ratio = float(metrics.get("area_ratio", 0.0))
    severity = float(metrics.get("severity", 0.0))
    mean_response = float(metrics.get("mean_response", 0.0))
    contrast = _d24v6_component_contrast(cand, heat, args)
    mean_abs = float(contrast.get("mean_abs_contrast", 0.0))
    contrast_support = float(contrast.get("contrast_support", 0.0))
    photo_ok = bool(
        mean_abs >= float(args.d24v6_fold_min_abs_contrast)
        and contrast_support >= float(args.d24v6_fold_min_contrast_support)
    )
    broad_geometry = bool(
        length >= float(args.d24v5_fold_broad_min_length_mm)
        and width >= float(args.d24v5_fold_broad_min_width_mm)
        and area_ratio >= float(args.d24v5_fold_broad_min_area_ratio)
    )
    long_geometry = bool(
        length >= float(args.d24v5_fold_min_length_mm)
        and width >= float(args.d24v5_fold_min_width_mm)
        and area_ratio >= float(args.d24v5_fold_min_area_ratio)
    )
    normalized_strength = bool(
        severity >= float(args.d24v5_fold_min_severity)
        or mean_response >= float(args.d24v5_fold_min_mean_response)
    )
    extreme_geometry = bool(
        width >= float(args.d24v6_fold_extreme_width_mm)
        and area_ratio >= float(args.d24v6_fold_extreme_area_ratio)
    )
    broad_fold = bool(broad_geometry and photo_ok)
    long_deep_fold = bool(long_geometry and photo_ok and normalized_strength)
    extreme_fold = bool(extreme_geometry and mean_abs >= 0.70 * float(args.d24v6_fold_min_abs_contrast))
    reasons: List[str] = []
    if broad_fold:
        reasons.append("broad_contrast_fold")
    if long_deep_fold:
        reasons.append("long_deep_contrast_fold")
    if extreme_fold:
        reasons.append("extreme_wide_fold")
    return {
        "is_fold": bool(reasons),
        "reasons": reasons,
        "length_mm": length,
        "width_mm": width,
        "area_ratio": area_ratio,
        "severity": severity,
        "mean_response": mean_response,
        "mean_abs_contrast": mean_abs,
        "contrast_support": contrast_support,
        "photo_ok": photo_ok,
        "broad_geometry": broad_geometry,
        "long_geometry": long_geometry,
        "extreme_geometry": extreme_geometry,
        "broad_fold": broad_fold,
        "long_deep_fold": long_deep_fold,
        "extreme_fold": extreme_fold,
        "large_area_diagnostic_only": area_ratio >= float(args.d24v5_fold_large_area_ratio),
    }

def evaluate_d24_snapshot(obs: Any, heat: Any, H: np.ndarray, state: D24State, args) -> Dict[str, Any]:
    """One frozen photo -> pants-shape and fold-only finish decision."""
    state.evaluation_count += 1
    quality = observation_quality_report(obs, args)
    stability = {
        "good": True,
        "reason": "single frozen snapshot",
        "history_count": 1,
        "mask_area_rel": 0.0,
        "center_shift_mm": 0.0,
        "pose_axis_spread_deg": 0.0,
    }
    if not quality["good"]:
        state.ready_streak = 0
        return {
            "status": "REJUDGE", "reason": "OBSERVATION_INVALID",
            "evaluation": state.evaluation_count, "quality": quality,
            "stability": stability, "shape": {}, "wrinkle": {},
            "snapshot_mode": True,
        }

    # Pants silhouette/pose is the hard finish condition.
    shape = d23._d19_finish_shape_report(obs, args, H)
    if not bool(shape.get("shape_good", False)):
        state.ready_streak = 0
        return {
            "status": "NOT_READY_SHAPE", "reason": "PANTS_SHAPE_FAILED",
            "evaluation": state.evaluation_count, "quality": quality,
            "stability": stability, "shape": shape, "wrinkle": {},
            "snapshot_mode": True,
        }
    if heat is None:
        state.ready_streak = 0
        return {
            "status": "REJUDGE", "reason": "A119_HEATMAP_UNAVAILABLE",
            "evaluation": state.evaluation_count, "quality": quality,
            "stability": stability, "shape": shape, "wrinkle": {},
            "snapshot_mode": True,
        }

    robust = robust_heat_threshold(heat, args)
    waist_ref = _d24v2_waist_reference(obs, heat, H, robust, args)
    candidates = list(getattr(heat, "d21v4_all_candidates", [])) or list(getattr(heat, "candidates", []))

    fold_blockers: List[Dict[str, Any]] = []
    waist_gather_ignored: List[Dict[str, Any]] = []
    seam_ignored: List[Dict[str, Any]] = []
    fine_allowed: List[Dict[str, Any]] = []
    robust_rejected: List[Dict[str, Any]] = []
    t1: List[Dict[str, Any]] = []
    t2: List[Dict[str, Any]] = []

    for original in candidates:
        cand = dict(original)
        seam = seam_veto_report(cand, heat, obs, H, args)
        cand["d24_seam"] = seam
        if bool(seam.get("matched", False)):
            seam_ignored.append(cand)
            continue

        support = candidate_robust_support(cand, heat, float(robust["threshold"]))
        cand["d24_robust_support"] = support
        if support < float(args.d24_candidate_min_robust_support):
            robust_rejected.append(cand)
            continue

        metrics = _d24v2_candidate_metrics(cand, heat, obs, waist_ref)
        cand["d24v2_metrics"] = metrics

        gather = _d24v5_waist_gather_report(cand, heat, obs, H, metrics, args)
        cand["d24v5_waist_gather"] = gather
        if bool(gather.get("matched", False)):
            waist_gather_ignored.append(cand)
            continue

        fold = _d24v6_fold_report(cand, metrics, heat, args)
        cand["d24v6_fold"] = fold
        if bool(fold.get("is_fold", False)):
            fold_blockers.append(cand)
        else:
            fine_allowed.append(cand)

        tier = int(metrics.get("tier", 0))
        if tier <= 1:
            t1.append(cand)
        elif tier == 2:
            t2.append(cand)

    mask_area = max(1.0, float(obs.mask.area_px))
    effective_area = max(1.0, float(cv2.countNonZero(heat.inner_mask)))
    blocker_area = float(sum(float(c.get("area_px", 0.0)) for c in fold_blockers))
    blocker_ratio = blocker_area / effective_area
    max_blob_ratio = max([float(c.get("area_px", 0.0)) / mask_area for c in fold_blockers] or [0.0])
    max_severity = max([float((c.get("d24v2_metrics", {}) or {}).get("severity", 0.0)) for c in fold_blockers] or [0.0])
    checks = {"fold_count": len(fold_blockers) <= int(args.finish_max_t3_count)}
    wrinkle_good = bool(all(checks.values()))

    wrinkle_report = {
        "policy": "PANTS_SHAPE_PLUS_REAL_FOLD_ONLY_WITH_CENTER_RISE_IGNORE",
        "good": wrinkle_good,
        "checks": checks,
        "t3_count": len(fold_blockers),
        "fold_count": len(fold_blockers),
        "t2_count": len(t2),
        "t1_allowed_count": len(t1),
        "fine_allowed_count": len(fine_allowed),
        "relative_allowed_count": 0,
        "non_major_allowed_count": len(fine_allowed),
        "waist_gather_ignored_count": len(waist_gather_ignored),
        "center_rise_ignored_px": int((getattr(heat, "center_rise_info", {}) or {}).get("ignored_px", 0)),
        "seam_ignored_count": len(seam_ignored),
        "robust_rejected_count": len(robust_rejected),
        "waist_structure_ignored_count": int(getattr(heat, "d23_waist_structure_ignored_count", 0)),
        "fine_ignored_count": len(fine_allowed) + int(getattr(heat, "d21v4_fine_ignored_count", 0)),
        "max_severity": max_severity,
        "actionable_ratio": blocker_ratio,
        "max_blob_ratio": max_blob_ratio,
        "waist_reference": waist_ref,
        "robust": robust,
        "a119_summary": str(getattr(heat, "summary", "")),
        "limits": {
            "max_fold_count": int(args.finish_max_t3_count),
            "gather_depth_ratio": float(args.d24v5_waist_gather_depth_ratio),
            "gather_radial_angle_deg": float(args.d24v5_waist_gather_radial_angle_deg),
            "fold_min_length_mm": float(args.d24v5_fold_min_length_mm),
            "fold_min_width_mm": float(args.d24v5_fold_min_width_mm),
            "fold_min_area_ratio": float(args.d24v5_fold_min_area_ratio),
            "fold_large_area_ratio_diagnostic_only": float(args.d24v5_fold_large_area_ratio),
            "fold_min_abs_contrast": float(args.d24v6_fold_min_abs_contrast),
            "center_rise_half_width_mm": float(args.d24v6_center_rise_half_width_mm),
        },
        "remaining": [k for k, v in checks.items() if not v],
        "t1": t1,
        "t2": t2,
        "t3": fold_blockers,
        "major_blockers": fold_blockers,
        "fold_blockers": fold_blockers,
        "waist_gather_ignored": waist_gather_ignored,
        "relative_allowed": [],
        "non_major_allowed": fine_allowed,
        "fine_allowed": fine_allowed,
        "seam_ignored": seam_ignored,
        "robust_rejected": robust_rejected,
    }

    if not wrinkle_good:
        state.ready_streak = 0
        return {
            "status": "NOT_READY_WRINKLE", "reason": "REAL_FOLD_REMAINS",
            "evaluation": state.evaluation_count, "quality": quality,
            "stability": stability, "shape": shape, "wrinkle": wrinkle_report,
            "ready_streak": 0, "ready_required": 1, "snapshot_mode": True,
        }

    state.ready_streak = 1
    return {
        "status": "READY_GOOD_ENOUGH", "reason": "PANTS_SHAPE_OK_AND_NO_REAL_FOLD",
        "evaluation": state.evaluation_count, "quality": quality,
        "stability": stability, "shape": shape, "wrinkle": wrinkle_report,
        "ready_streak": 1, "ready_required": 1, "snapshot_mode": True,
    }

def evaluate_frame_snapshot(frame: np.ndarray, H: np.ndarray, seg_model, pose_model,
                            state: D24State, args, cfg) -> Tuple[Any, Any, Dict[str, Any], np.ndarray]:
    obs = d23.infer_bottom_observation(seg_model, pose_model, frame, H, args, cfg)
    heat = build_a119_bottom_heatmap(frame, obs, H, args) if obs is not None and obs.mask is not None else None
    report = evaluate_d24_snapshot(obs, heat, H, state, args)
    overlay = draw_d24_overlay(frame, obs, heat, report, H, cfg, args)
    put_text(overlay, "FROZEN SNAPSHOT / SHAPE + FOLD ONLY", (25, overlay.shape[0] - 22), (255, 255, 255), 0.62, 2)
    return obs, heat, report, overlay


def _save_captured_source(frame: np.ndarray, args, status: str = "CAPTURE") -> str:
    out_dir = Path(args.snapshot_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    path = out_dir / f"d24v6_{ts}_{status}_source.jpg"
    cv2.imwrite(str(path), frame)
    return str(path)


def run_single_image_snapshot(args, config, cfg, H, seg_model, pose_model) -> int:
    frame = cv2.imread(args.image)
    if frame is None:
        print(f"[D24V3-ERROR] cannot read image: {args.image}")
        return 2
    if H is None:
        print("[D24V3-ERROR] Homography is required for --image. Load --hfile first.")
        return 2
    state = D24State(history=collections.deque(maxlen=1))
    obs, heat, report, overlay = evaluate_frame_snapshot(frame, H, seg_model, pose_model, state, args, cfg)
    state.latest_obs, state.latest_heat = obs, heat
    state.latest_report, state.latest_overlay = report, overlay
    state.latest_status = str(report.get("status", "REJUDGE"))
    state.latest_eval_frame = frame.copy()
    state.latest_h = H.copy()
    cv2.imwrite(args.output, overlay)
    print_report(report)
    print(f"[D24V3] frozen result saved: {args.output}")
    append_jsonl(args.log_jsonl, report)
    return 0


def main_snapshot() -> int:
    args = finalize_args(build_parser().parse_args())
    # D24-v3 intentionally makes one decision per capture. No temporal streak is
    # accumulated from repeated camera frames.
    args.history_count = 1
    args.ready_confirm_count = 1
    print(f"[BUILD] {STEP_D24_BUILD}")
    print("[D24V6] EVENT SNAPSHOT: state change -> stillness -> one frozen evaluation")
    print("[D24V6] FINISH: pants shape is mandatory; fine wrinkles and waistband gathers are allowed; folds block READY")
    print("[D24V6-SAFETY] perception-only: no serial port and no robot command")
    print(
        f"[D24V6-GATHER] depth<={args.d24v5_waist_gather_depth_ratio:.2f}*(waist-crotch), "
        f"radialErr<={args.d24v5_waist_gather_radial_angle_deg:.0f}deg, "
        f"len<={args.d24v5_waist_gather_max_length_mm:.0f}mm"
    )
    print(
        f"[D24V6-FOLD] longFold len>={args.d24v5_fold_min_length_mm:.0f}mm "
        f"width>={args.d24v5_fold_min_width_mm:.0f}mm area>={args.d24v5_fold_min_area_ratio:.4f}; fine lines allowed"
    )

    config = d23.load_json_if_exists(args.config) or {
        "aruco": {
            "dictionary": "DICT_4X4_50",
            "required_ids": [0, 1, 2, 3],
            "marker_board_mm": d23.DEFAULT_MARKER_BOARD_MM,
        },
        "dual_roarm": {"split_board_x": 247.0},
    }
    cfg = make_cfg(config)
    H = d23.load_homography(args.hfile) if args.load_h else None
    if H is not None:
        print(f"[H] loaded: {args.hfile}")
    seg_model, pose_model = d23.load_models(args)
    if args.image:
        return run_single_image_snapshot(args, config, cfg, H, seg_model, pose_model)

    detector = d23.make_aruco_detector(
        d23.get_dictionary(config.get("aruco", {}).get("dictionary", "DICT_4X4_50"))
    )
    required_ids = config.get("aruco", {}).get("required_ids", [0, 1, 2, 3])
    marker_board_mm = config.get("aruco", {}).get("marker_board_mm", d23.DEFAULT_MARKER_BOARD_MM)
    cap = d23.open_camera(args)
    d23.configure_usb_camera(args)
    for _ in range(max(0, int(args.warmup_frames))):
        cap.read()
        time.sleep(0.02)

    frame_buffer: Deque[np.ndarray] = collections.deque(maxlen=max(1, int(args.burst_frame_count)))
    state = D24State(history=collections.deque(maxlen=1))
    state.latest_h = None if H is None else H.copy()
    last_live: Optional[np.ndarray] = None
    auto_once_done = False
    live_window = args.window_name + "_LIVE"
    result_window = args.window_name + "_SNAPSHOT"

    def capture_and_judge() -> None:
        nonlocal H
        if H is None:
            print("[D24V3] capture blocked: Homography is not locked")
            return
        if not frame_buffer:
            print("[D24V3] capture blocked: no camera frame")
            return
        # Capture a fresh burst only after SPACE. The rolling live preview is not
        # used for judgment, so the result cannot drift from background frames.
        frames: List[np.ndarray] = []
        for _ in range(2):
            cap.read()
            time.sleep(0.015)
        for _ in range(max(1, int(args.burst_frame_count))):
            ok_cap, frame_cap = cap.read()
            if ok_cap and frame_cap is not None:
                frames.append(frame_cap.copy())
            time.sleep(0.018)
        if not frames:
            print("[D24V3] capture blocked: fresh camera burst failed")
            return
        frozen = median_frame(frames)
        source_path = _save_captured_source(frozen, args)
        print(f"[D24V3-CAPTURE] frozen photo={source_path} frames={len(frames)}")
        obs, heat, report, overlay = evaluate_frame_snapshot(frozen, H.copy(), seg_model, pose_model, state, args, cfg)
        report["captured_source"] = source_path
        report["captured_frame_count"] = len(frames)
        state.latest_obs = obs
        state.latest_heat = heat
        state.latest_report = report
        state.latest_status = str(report.get("status", "REJUDGE"))
        state.latest_overlay = overlay
        state.latest_eval_frame = frozen
        state.latest_h = H.copy()
        print_report(report)
        if heat is not None:
            print(f"[D24V3-A119] {heat.summary}")
        append_jsonl(args.log_jsonl, report)

    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                time.sleep(0.03)
                continue
            last_live = frame.copy()
            frame_buffer.append(frame.copy())
            H_candidate, marker_centers = detect_homography(frame, detector, marker_board_mm, required_ids)
            if H is None and bool(args.auto_lock) and H_candidate is not None:
                H = H_candidate
                state.latest_h = H.copy()
                if bool(args.save_h_on_lock):
                    d23.save_homography(args.hfile, H)
                print("[H] auto-locked from ArUco markers")

            live = frame.copy()
            put_text(live, "SPACE: CAPTURE PHOTO + JUDGE", (25, 38), (0, 255, 255), 0.76, 2)
            put_text(live, "Result does NOT update until next SPACE", (25, 70), (255, 255, 255), 0.58, 1)
            htxt = "H: LOCKED" if H is not None else f"H: WAIT markers {sorted(marker_centers.keys())}"
            put_text(live, htxt, (25, 100), (0, 220, 0) if H is not None else (0, 0, 255), 0.55, 1)
            if state.latest_report:
                put_text(live, f"LAST: {state.latest_status}", (25, 130), STATUS_COLORS.get(state.latest_status, (255,255,255)), 0.62, 2)

            if not args.no_window:
                cv2.imshow(live_window, live)
                if state.latest_overlay is not None:
                    cv2.imshow(result_window, state.latest_overlay)
                key = cv2.waitKey(1) & 0xFF
            else:
                key = 255
            terminal_key = d23.read_terminal_key()
            if terminal_key != 255:
                key = terminal_key

            if args.once and not auto_once_done and H is not None and len(frame_buffer) >= max(1, int(args.burst_frame_count)):
                capture_and_judge()
                auto_once_done = True
                if state.latest_overlay is not None:
                    cv2.imwrite(args.output, state.latest_overlay)
                    print(f"[D24V3] once overlay saved: {args.output}")
                break

            if key in (27, ord("q"), ord("Q")):
                break
            if key == ord(" "):
                capture_and_judge()
            elif key in (ord("c"), ord("C")):
                state.reset_temporal()
                state.latest_overlay = None
                state.latest_eval_frame = None
                state.latest_obs = None
                state.latest_heat = None
                print("[D24V3] frozen result cleared")
            elif key in (ord("s"), ord("S")):
                save_sample(state, args)
            elif key in (ord("l"), ord("L")):
                if H_candidate is None:
                    print(f"[H] lock failed; visible marker IDs={sorted(marker_centers.keys())}")
                else:
                    H = H_candidate
                    state.latest_h = H.copy()
                    if bool(args.save_h_on_lock):
                        d23.save_homography(args.hfile, H)
                    print("[H] locked/updated")
            elif key in (ord("r"), ord("R")):
                H = None
                state.latest_h = None
                print("[H] reset in memory")
    finally:
        cap.release()
        if not args.no_window:
            cv2.destroyAllWindows()
    return 0



# =============================================================================
# D24-v4: event-driven change -> stillness -> one snapshot inference
# =============================================================================

def _v4_board_mask_small(frame_shape, H: Optional[np.ndarray], config: Dict[str, Any], args) -> np.ndarray:
    """Return a low-resolution board-interior mask for cheap change detection."""
    out_w = max(32, int(args.change_preview_width))
    out_h = max(24, int(args.change_preview_height))
    full = np.zeros(frame_shape[:2], dtype=np.uint8)
    if H is None:
        full[:] = 255
    else:
        try:
            xmin, xmax, ymin, ymax = d23.board_bounds_from_config(config)
            corners_board = [
                (xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax),
            ]
            corners_px = []
            for bx, by in corners_board:
                p = d23.board_to_pixel(H, float(bx), float(by))
                if p is None:
                    raise RuntimeError("board_to_pixel failed")
                corners_px.append([int(round(p[0])), int(round(p[1]))])
            cv2.fillConvexPoly(full, np.asarray(corners_px, np.int32), 255)
            shrink = max(0, int(args.change_board_shrink_px))
            if shrink > 0:
                k = 2 * shrink + 1
                full = cv2.erode(full, np.ones((k, k), np.uint8), iterations=1)
        except Exception:
            full[:] = 255
    small = cv2.resize(full, (out_w, out_h), interpolation=cv2.INTER_NEAREST)
    return small > 0


def _v4_change_gray(frame: np.ndarray, H: Optional[np.ndarray], config: Dict[str, Any], args) -> Tuple[np.ndarray, np.ndarray]:
    out_w = max(32, int(args.change_preview_width))
    out_h = max(24, int(args.change_preview_height))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    gray = cv2.resize(gray, (out_w, out_h), interpolation=cv2.INTER_AREA).astype(np.float32)
    mask = _v4_board_mask_small(frame.shape, H, config, args)
    if int(np.count_nonzero(mask)) < 100:
        mask = np.ones(gray.shape, dtype=bool)
    return gray, mask


def _v4_diff_metrics(current: np.ndarray, reference: np.ndarray, mask: np.ndarray, pixel_threshold: float) -> Dict[str, float]:
    if current is None or reference is None or current.shape != reference.shape:
        return {"mean": 999.0, "active_ratio": 1.0, "p90": 999.0, "brightness_offset": 0.0}
    valid = mask if mask is not None and mask.shape == current.shape else np.ones(current.shape, dtype=bool)
    if int(np.count_nonzero(valid)) < 10:
        valid = np.ones(current.shape, dtype=bool)
    # Cancel global exposure drift before measuring garment/shape changes.
    offset = float(np.median(current[valid]) - np.median(reference[valid]))
    adjusted = np.clip(current - offset, 0.0, 255.0)
    diff = np.abs(adjusted - reference)
    values = diff[valid]
    return {
        "mean": float(np.mean(values)) if values.size else 0.0,
        "active_ratio": float(np.mean(values >= float(pixel_threshold))) if values.size else 0.0,
        "p90": float(np.percentile(values, 90.0)) if values.size else 0.0,
        "brightness_offset": offset,
    }


def _v4_save_captured_source(frame: np.ndarray, args, reason: str) -> str:
    out_dir = Path(args.snapshot_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    safe = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in str(reason))[:40]
    path = out_dir / f"d25_{ts}_{safe}_source.jpg"
    cv2.imwrite(str(path), frame)
    return str(path)


def main_event_snapshot() -> int:
    args = finalize_args(build_parser().parse_args())
    args.history_count = 1
    args.ready_confirm_count = 1
    print(f"[BUILD] {STEP_D24_BUILD}")
    print("[D25] EVENT SNAPSHOT: change detected -> wait until still -> one A119 heatmap judgment")
    print("[D25] Result stays frozen until the pants/scene changes again")
    print("[D25-SAFETY] perception-only: no serial port and no robot command")
    print(
        f"[D25-GATE] change mean>={args.change_mean_threshold:.2f} OR "
        f"active>={args.change_active_ratio:.3f} for {args.change_min_frames} frames; "
        f"still motion<={args.settle_motion_threshold:.2f} for {args.settle_seconds:.2f}s"
    )

    config = d23.load_json_if_exists(args.config) or {
        "aruco": {
            "dictionary": "DICT_4X4_50",
            "required_ids": [0, 1, 2, 3],
            "marker_board_mm": d23.DEFAULT_MARKER_BOARD_MM,
        },
        "dual_roarm": {"split_board_x": 247.0},
    }
    cfg = make_cfg(config)
    H = d23.load_homography(args.hfile) if args.load_h else None
    if H is not None:
        print(f"[H] loaded: {args.hfile}")
    seg_model, pose_model = d23.load_models(args)
    if args.image:
        return run_single_image_snapshot(args, config, cfg, H, seg_model, pose_model)

    detector = d23.make_aruco_detector(
        d23.get_dictionary(config.get("aruco", {}).get("dictionary", "DICT_4X4_50"))
    )
    required_ids = config.get("aruco", {}).get("required_ids", [0, 1, 2, 3])
    marker_board_mm = config.get("aruco", {}).get("marker_board_mm", d23.DEFAULT_MARKER_BOARD_MM)
    cap = d23.open_camera(args)
    d23.configure_usb_camera(args)
    for _ in range(max(0, int(args.warmup_frames))):
        cap.read()
        time.sleep(0.02)

    state = D24State(history=collections.deque(maxlen=1))
    state.latest_h = None if H is None else H.copy()
    state_lock = threading.RLock()
    analysis_lock = threading.Lock()

    # State machine values.
    phase = "WAIT_INITIAL_STILL" if bool(args.startup_auto_snapshot) else "MONITOR_CHANGE"
    reference_gray: Optional[np.ndarray] = None
    reference_mask: Optional[np.ndarray] = None
    previous_gray: Optional[np.ndarray] = None
    change_frames = 0
    stable_frames = 0
    stable_since: Optional[float] = None
    last_capture_at = 0.0
    last_change_metrics = {"mean": 0.0, "active_ratio": 0.0, "p90": 0.0, "brightness_offset": 0.0}
    last_motion = 999.0
    pending_reason = "startup"
    auto_paused = bool(args.pause_auto)
    capture_requested = False
    once_finished = False
    live_window = args.window_name + "_LIVE"
    result_window = args.window_name + "_SNAPSHOT"

    def set_phase(new_phase: str) -> None:
        nonlocal phase
        phase = str(new_phase)

    def collect_frozen_burst() -> Optional[np.ndarray]:
        frames: List[np.ndarray] = []
        for _ in range(max(0, int(args.capture_flush_frames))):
            cap.read()
            time.sleep(0.012)
        for _ in range(max(1, int(args.burst_frame_count))):
            ok_cap, frame_cap = cap.read()
            if ok_cap and frame_cap is not None:
                frames.append(frame_cap.copy())
            time.sleep(0.018)
        if not frames:
            return None
        return median_frame(frames)

    def worker(frozen: np.ndarray, H_snapshot: np.ndarray, reason: str) -> None:
        nonlocal reference_gray, reference_mask, previous_gray
        nonlocal change_frames, stable_frames, stable_since, last_capture_at
        nonlocal last_change_metrics, last_motion, once_finished
        try:
            source_path = _v4_save_captured_source(frozen, args, reason)
            print(f"[D25-CAPTURE] reason={reason} source={source_path} burst={max(1,int(args.burst_frame_count))}")
            obs, heat, report, overlay = evaluate_frame_snapshot(
                frozen, H_snapshot, seg_model, pose_model, state, args, cfg
            )
            report["captured_source"] = source_path
            report["capture_reason"] = reason
            report["event_snapshot_mode"] = True
            report["change_gate"] = dict(last_change_metrics)
            report["still_motion"] = float(last_motion)
            ref_gray, ref_mask = _v4_change_gray(frozen, H_snapshot, config, args)
            with state_lock:
                state.latest_obs = obs
                state.latest_heat = heat
                state.latest_report = report
                state.latest_status = str(report.get("status", "REJUDGE"))
                state.latest_overlay = overlay
                state.latest_eval_frame = frozen.copy()
                state.latest_h = H_snapshot.copy()
                reference_gray = ref_gray
                reference_mask = ref_mask
                previous_gray = ref_gray.copy()
                change_frames = 0
                stable_frames = 0
                stable_since = None
                last_capture_at = time.time()
                last_change_metrics = {"mean": 0.0, "active_ratio": 0.0, "p90": 0.0, "brightness_offset": 0.0}
                last_motion = 0.0
                set_phase("MONITOR_CHANGE")
            print_report(report)
            if heat is not None:
                print(f"[D25-A119] {heat.summary}")
            append_jsonl(args.log_jsonl, report)
            if bool(args.once):
                cv2.imwrite(args.output, overlay)
                print(f"[D25] once overlay saved: {args.output}")
                once_finished = True
        except Exception as exc:
            print(f"[D25-WORKER-ERROR] {exc!r}")
            with state_lock:
                state.latest_status = "REJUDGE"
                state.latest_report = {"status": "REJUDGE", "reason": f"worker error: {exc!r}"}
                last_capture_at = time.time()
                set_phase("MONITOR_CHANGE" if reference_gray is not None else "WAIT_INITIAL_STILL")
        finally:
            analysis_lock.release()

    def start_snapshot(reason: str, force: bool = False) -> bool:
        nonlocal last_capture_at
        if H is None:
            print("[D25] snapshot blocked: Homography is not locked")
            return False
        if not force and (time.time() - last_capture_at) < max(0.0, float(args.change_cooldown_s)):
            return False
        if not analysis_lock.acquire(blocking=False):
            print("[D25] snapshot skipped: analysis already running")
            return False
        frozen = collect_frozen_burst()
        if frozen is None:
            analysis_lock.release()
            print("[D25] snapshot blocked: fresh camera burst failed")
            return False
        set_phase("ANALYZING")
        threading.Thread(target=worker, args=(frozen, H.copy(), reason), daemon=True).start()
        return True

    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                time.sleep(0.03)
                continue

            H_candidate, marker_centers = detect_homography(frame, detector, marker_board_mm, required_ids)
            if H is None and bool(args.auto_lock) and H_candidate is not None:
                H = H_candidate
                with state_lock:
                    state.latest_h = H.copy()
                if bool(args.save_h_on_lock):
                    d23.save_homography(args.hfile, H)
                print("[H] auto-locked from ArUco markers")
                previous_gray = None
                stable_since = None
                stable_frames = 0
                set_phase("WAIT_INITIAL_STILL" if bool(args.startup_auto_snapshot) else "MONITOR_CHANGE")

            current_gray = None
            current_mask = None
            if H is not None:
                current_gray, current_mask = _v4_change_gray(frame, H, config, args)

            now = time.time()
            if (
                H is not None and current_gray is not None and not auto_paused
                and bool(args.auto_snapshot_on_change) and phase != "ANALYZING"
            ):
                if phase == "MONITOR_CHANGE":
                    if reference_gray is None:
                        set_phase("WAIT_INITIAL_STILL")
                        previous_gray = current_gray.copy()
                        stable_since = None
                        stable_frames = 0
                    elif (now - last_capture_at) >= max(0.0, float(args.change_cooldown_s)):
                        mask = current_mask
                        if reference_mask is not None and reference_mask.shape == current_mask.shape:
                            mask = current_mask & reference_mask
                        last_change_metrics = _v4_diff_metrics(
                            current_gray, reference_gray, mask, float(args.change_pixel_threshold)
                        )
                        changed = (
                            last_change_metrics["mean"] >= float(args.change_mean_threshold)
                            or last_change_metrics["active_ratio"] >= float(args.change_active_ratio)
                        )
                        if changed:
                            change_frames += 1
                        else:
                            change_frames = max(0, change_frames - 1)
                        if change_frames >= max(1, int(args.change_min_frames)):
                            pending_reason = "pants_state_changed"
                            set_phase("WAIT_STILL")
                            previous_gray = current_gray.copy()
                            stable_since = None
                            stable_frames = 0
                            print(
                                f"[D25-CHANGE] confirmed mean={last_change_metrics['mean']:.2f} "
                                f"active={last_change_metrics['active_ratio']:.3f}; waiting for stillness"
                            )
                elif phase in ("WAIT_INITIAL_STILL", "WAIT_STILL"):
                    if previous_gray is None:
                        previous_gray = current_gray.copy()
                        stable_since = None
                        stable_frames = 0
                        last_motion = 999.0
                    else:
                        motion_report = _v4_diff_metrics(
                            current_gray, previous_gray, current_mask, float(args.change_pixel_threshold)
                        )
                        last_motion = float(motion_report["mean"])
                        previous_gray = current_gray.copy()
                        if last_motion <= float(args.settle_motion_threshold):
                            stable_frames += 1
                            if stable_since is None:
                                stable_since = now
                        else:
                            stable_frames = 0
                            stable_since = None
                    stable_elapsed = 0.0 if stable_since is None else now - stable_since
                    if (
                        stable_since is not None
                        and stable_elapsed >= max(0.05, float(args.settle_seconds))
                        and stable_frames >= max(1, int(args.settle_min_frames))
                    ):
                        reason = "startup_still" if phase == "WAIT_INITIAL_STILL" else pending_reason
                        if start_snapshot(reason=reason, force=False):
                            stable_since = None
                            stable_frames = 0

            if capture_requested:
                capture_requested = False
                start_snapshot(reason="manual_space", force=True)

            live = frame.copy()
            phase_color = {
                "MONITOR_CHANGE": (0, 220, 0),
                "WAIT_INITIAL_STILL": (0, 220, 255),
                "WAIT_STILL": (0, 220, 255),
                "ANALYZING": (255, 180, 0),
            }.get(phase, (255, 255, 255))
            put_text(live, f"AUTO SNAPSHOT: {phase}", (25, 38), phase_color, 0.74, 2)
            put_text(
                live,
                f"change mean={last_change_metrics['mean']:.2f}/{args.change_mean_threshold:.2f} "
                f"active={last_change_metrics['active_ratio']:.3f}/{args.change_active_ratio:.3f} "
                f"frames={change_frames}/{max(1,int(args.change_min_frames))}",
                (25, 70), (255, 255, 255), 0.48, 1,
            )
            stable_elapsed = 0.0 if stable_since is None else max(0.0, now - stable_since)
            put_text(
                live,
                f"motion={last_motion:.2f}/{args.settle_motion_threshold:.2f} "
                f"still={stable_elapsed:.2f}/{args.settle_seconds:.2f}s frames={stable_frames}/{max(1,int(args.settle_min_frames))}",
                (25, 98), (255, 255, 255), 0.48, 1,
            )
            htxt = "H: LOCKED" if H is not None else f"H: WAIT markers {sorted(marker_centers.keys())}"
            put_text(live, htxt, (25, 126), (0, 220, 0) if H is not None else (0, 0, 255), 0.50, 1)
            with state_lock:
                latest_status = state.latest_status
                latest_overlay = None if state.latest_overlay is None else state.latest_overlay.copy()
            if latest_overlay is not None:
                put_text(live, f"FROZEN RESULT: {latest_status}", (25, 154), STATUS_COLORS.get(latest_status, (255,255,255)), 0.58, 2)
            put_text(live, "SPACE force | P pause auto | C reset | S save | Q quit", (25, live.shape[0]-18), (220,220,220), 0.48, 1)
            if auto_paused:
                put_text(live, "AUTO CHANGE WATCH PAUSED", (25, 188), (0, 0, 255), 0.62, 2)

            if not args.no_window:
                cv2.imshow(live_window, live)
                if latest_overlay is not None:
                    cv2.imshow(result_window, latest_overlay)
                key = cv2.waitKey(1) & 0xFF
            else:
                key = 255
            terminal_key = d23.read_terminal_key()
            if terminal_key != 255:
                key = terminal_key

            if once_finished:
                break
            if key in (27, ord("q"), ord("Q")):
                break
            if key == ord(" "):
                capture_requested = True
            elif key in (ord("p"), ord("P")):
                auto_paused = not auto_paused
                print(f"[D25] auto change watch {'PAUSED' if auto_paused else 'RUNNING'}")
            elif key in (ord("c"), ord("C")):
                with state_lock:
                    state.reset_temporal()
                    state.latest_overlay = None
                    state.latest_eval_frame = None
                    state.latest_obs = None
                    state.latest_heat = None
                reference_gray = None
                reference_mask = None
                previous_gray = None
                change_frames = 0
                stable_frames = 0
                stable_since = None
                pending_reason = "startup"
                set_phase("WAIT_INITIAL_STILL" if bool(args.startup_auto_snapshot) else "MONITOR_CHANGE")
                print("[D25] frozen result/reference cleared; waiting for a stable initial state")
            elif key in (ord("s"), ord("S")):
                save_sample(state, args)
            elif key in (ord("l"), ord("L")):
                if H_candidate is None:
                    print(f"[H] lock failed; visible marker IDs={sorted(marker_centers.keys())}")
                else:
                    H = H_candidate
                    with state_lock:
                        state.latest_h = H.copy()
                    if bool(args.save_h_on_lock):
                        d23.save_homography(args.hfile, H)
                    reference_gray = None
                    reference_mask = None
                    previous_gray = None
                    stable_since = None
                    stable_frames = 0
                    set_phase("WAIT_INITIAL_STILL")
                    print("[H] locked/updated; change reference cleared")
            elif key in (ord("r"), ord("R")):
                H = None
                with state_lock:
                    state.latest_h = None
                reference_gray = None
                reference_mask = None
                previous_gray = None
                stable_since = None
                stable_frames = 0
                set_phase("WAIT_INITIAL_STILL")
                print("[H] reset in memory")
    finally:
        cap.release()
        if not args.no_window:
            cv2.destroyAllWindows()
    return 0



# =============================================================================
# D25: reference-free contour differential / topology finish evaluator
# =============================================================================
# D25 deliberately does NOT register a per-garment normal reference.  The
# decision is based on topology shared by short pants: one waist region, one
# crotch concavity, two exposed legs, two hems, and no unexpected macro fold.

_D25_BASE_BUILD_PARSER = build_parser
_D25_BASE_DRAW_OVERLAY = draw_d24_overlay


def build_parser() -> argparse.ArgumentParser:
    """Extend the D24-v6 camera/heatmap parser with D25 geometry controls."""
    p = _D25_BASE_BUILD_PARSER()
    p.description = (
        "D25 reference-free pants finish evaluator: medium pose + segmentation "
        "contour derivatives + topology + macro-fold gate; perception only"
    )
    p.set_defaults(
        pose_model="/workspace/project_train/yolo26/bottom_pose8_yolo26m_e40_best.engine",
        d23_pose_tta_fast_first=False,
        min_valid_keypoints=5,
        min_mean_kpt_conf=0.28,
        output="d25_finish_overlay.jpg",
        snapshot_dir="d25_finish_samples",
        log_jsonl="d25_finish_log.jsonl",
        window_name="D25_PANTS_TOPOLOGY_FINISH",
    )

    # Mask smoothing and contour differentiation.
    p.add_argument("--d25-mask-close-px", type=int, default=7)
    p.add_argument("--d25-mask-open-px", type=int, default=3)
    p.add_argument("--d25-contour-resample-points", type=int, default=420)
    p.add_argument("--d25-contour-smooth-window", type=int, default=13)
    p.add_argument("--d25-profile-bins", type=int, default=72)
    p.add_argument("--d25-canonical-max-side", type=int, default=440)

    # Reference-free garment topology. These are intentionally permissive: a
    # normal garment need not be perfectly left/right symmetric.
    p.add_argument("--d25-shape-score-min", type=float, default=0.62)
    p.add_argument("--d25-waist-one-row-ratio-min", type=float, default=0.52)
    p.add_argument("--d25-crotch-gap-ratio-min", type=float, default=0.045)
    p.add_argument("--d25-crotch-y-min", type=float, default=0.20)
    p.add_argument("--d25-crotch-y-max", type=float, default=0.78)
    p.add_argument("--d25-lower-two-leg-row-ratio-min", type=float, default=0.22)
    p.add_argument("--d25-hem-two-leg-row-ratio-min", type=float, default=0.14)
    p.add_argument("--d25-leg-area-balance-min", type=float, default=0.38)
    p.add_argument("--d25-min-hem-width-waist-ratio", type=float, default=0.075)
    p.add_argument("--d25-min-solidity", type=float, default=0.50)
    p.add_argument("--d25-min-topology-checks", type=int, default=5,
                   help="Minimum passed structural checks among waist/crotch/legs/hems/balance/solidity.")

    # Outer-boundary macro-fold detection. Waist gathers, the expected crotch
    # concavity and hem corners are excluded before evaluating unexpected dents.
    p.add_argument("--d25-defect-min-depth-ratio", type=float, default=0.065)
    p.add_argument("--d25-defect-hard-depth-ratio", type=float, default=0.16)
    p.add_argument("--d25-max-moderate-unexpected-defects", type=int, default=1)
    p.add_argument("--d25-high-curvature-threshold", type=float, default=0.11,
                   help="Dimensionless |curvature| multiplied by canonical waist width.")
    p.add_argument("--d25-high-curvature-arc-ratio-max", type=float, default=0.16)

    # Pose/geometry fusion. Geometry is allowed to recover a weak/wrong TTA
    # orientation, while a strong matching medium-pose result stabilizes labels.
    p.add_argument("--d25-pose-geometry-fuse", dest="d25_pose_geometry_fuse", action="store_true", default=True)
    p.add_argument("--no-d25-pose-geometry-fuse", dest="d25_pose_geometry_fuse", action="store_false")
    p.add_argument("--d25-fuse-max-distance-waist-ratio", type=float, default=0.22)
    p.add_argument("--d25-geometry-weight", type=float, default=0.62)
    p.add_argument("--d25-min-pose-order-score", type=float, default=0.30)
    p.add_argument("--d25-draw-geometry", dest="d25_draw_geometry", action="store_true", default=True)
    p.add_argument("--no-d25-draw-geometry", dest="d25_draw_geometry", action="store_false")
    return p


def observation_quality_report(obs: Any, args) -> Dict[str, Any]:
    """D25 quality gate: do not discard a usable mask only because TTA is weak.

    Pose confidence remains diagnostic and helps semantic point fusion.  The
    contour topology is the hard geometric verifier, so pose.valid/obs.valid are
    advisory rather than mandatory here.
    """
    if obs is None or obs.mask is None:
        return {
            "good": False, "reason": "mask missing", "valid_keypoints": 0,
            "mean_kpt_conf": 0.0, "checks": {"mask_present": False},
        }
    pose = getattr(obs, "pose", None)
    confs = list(getattr(pose, "keypoint_conf", {}).values()) if pose is not None else []
    valid_kpts = len(confs)
    mean_conf = float(np.mean(confs)) if confs else 0.0
    pose_present = pose is not None
    pose_support = bool(
        pose_present
        and valid_kpts >= int(args.min_valid_keypoints)
        and mean_conf >= float(args.min_mean_kpt_conf)
    )
    checks = {
        "mask_area": float(obs.mask.area_px) >= 1200.0,
        "mask_solidity_loose": float(getattr(obs.mask, "solidity", 0.0)) >= 0.30,
        "pose_support": pose_support,
    }
    # Medium pose is expected, but geometry can still diagnose the mask when the
    # pose is weak. Require either usable pose support or a large, coherent mask.
    geometry_recoverable = bool(
        float(obs.mask.area_px) >= 2400.0
        and float(getattr(obs.mask, "solidity", 0.0)) >= 0.42
    )
    good = bool(checks["mask_area"] and checks["mask_solidity_loose"] and (pose_support or geometry_recoverable))
    return {
        "good": good,
        "reason": "OK" if good else "mask/pose quality low",
        "checks": checks,
        "geometry_recoverable": geometry_recoverable,
        "observation_valid_advisory": bool(getattr(obs, "valid", False)),
        "pose_valid_advisory": bool(getattr(pose, "valid", False)) if pose_present else False,
        "valid_keypoints": valid_kpts,
        "mean_kpt_conf": mean_conf,
        "pose_tta_state": str(getattr(pose, "tta_state", "MISSING")) if pose_present else "MISSING",
        "pose_tta_score": float(getattr(pose, "tta_score", 0.0)) if pose_present else 0.0,
        "pose_tta_tested": int(getattr(pose, "tta_tested_count", 0)) if pose_present else 0,
        "mask_area_px": float(obs.mask.area_px),
    }


def _d25_odd(v: int, minimum: int = 1) -> int:
    v = max(int(minimum), int(v))
    return v if v % 2 == 1 else v + 1


def _d25_clean_mask(mask_u8: np.ndarray, args) -> np.ndarray:
    mask = ((np.asarray(mask_u8) > 0).astype(np.uint8) * 255)
    close_k = _d25_odd(int(args.d25_mask_close_px))
    open_k = _d25_odd(int(args.d25_mask_open_px))
    if close_k > 1:
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((close_k, close_k), np.uint8))
    if open_k > 1:
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((open_k, open_k), np.uint8))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    if n <= 1:
        return mask
    idx = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return ((labels == idx).astype(np.uint8) * 255)


def _d25_largest_contour(mask_u8: np.ndarray) -> Optional[np.ndarray]:
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None
    return max(contours, key=cv2.contourArea)


def _d25_resample_closed(points: np.ndarray, count: int) -> np.ndarray:
    pts = np.asarray(points, np.float32).reshape(-1, 2)
    if len(pts) < 4:
        return pts.copy()
    closed = np.vstack([pts, pts[0]])
    seg = np.linalg.norm(np.diff(closed, axis=0), axis=1)
    total = float(np.sum(seg))
    if total <= 1e-6:
        return pts.copy()
    cumulative = np.concatenate([[0.0], np.cumsum(seg)])
    targets = np.linspace(0.0, total, max(16, int(count)), endpoint=False)
    out = np.empty((len(targets), 2), np.float32)
    j = 0
    for i, t in enumerate(targets):
        while j + 1 < len(cumulative) and cumulative[j + 1] <= t:
            j += 1
        denom = max(1e-6, float(cumulative[j + 1] - cumulative[j]))
        a = float((t - cumulative[j]) / denom)
        out[i] = closed[j] * (1.0 - a) + closed[j + 1] * a
    return out


def _d25_circular_smooth(points: np.ndarray, window: int) -> np.ndarray:
    pts = np.asarray(points, np.float32).reshape(-1, 2)
    if len(pts) < 7:
        return pts.copy()
    w = _d25_odd(min(int(window), max(3, len(pts) // 3)))
    pad = w // 2
    ext = np.vstack([pts[-pad:], pts, pts[:pad]])
    kernel = np.ones(w, np.float32) / float(w)
    x = np.convolve(ext[:, 0], kernel, mode="valid")
    y = np.convolve(ext[:, 1], kernel, mode="valid")
    return np.column_stack([x, y]).astype(np.float32)


def _d25_contour_differential(points: np.ndarray, width_scale: float, args) -> Dict[str, Any]:
    pts = np.asarray(points, np.float32).reshape(-1, 2)
    if len(pts) < 12:
        return {"valid": False, "reason": "too few contour samples"}
    d1 = 0.5 * (np.roll(pts, -1, axis=0) - np.roll(pts, 1, axis=0))
    d2 = np.roll(pts, -1, axis=0) - 2.0 * pts + np.roll(pts, 1, axis=0)
    speed = np.linalg.norm(d1, axis=1)
    cross = d1[:, 0] * d2[:, 1] - d1[:, 1] * d2[:, 0]
    curvature = cross / np.maximum(speed ** 3, 1e-6)
    k_norm = np.abs(curvature) * max(1.0, float(width_scale))
    high = k_norm >= float(args.d25_high_curvature_threshold)
    turning = np.arctan2(
        d1[:, 0] * np.roll(d1, -1, axis=0)[:, 1] - d1[:, 1] * np.roll(d1, -1, axis=0)[:, 0],
        np.sum(d1 * np.roll(d1, -1, axis=0), axis=1),
    )
    return {
        "valid": True,
        "mean_abs_curvature_scaled": float(np.mean(k_norm)),
        "p90_abs_curvature_scaled": float(np.percentile(k_norm, 90.0)),
        "high_curvature_arc_ratio": float(np.mean(high)),
        "total_abs_turning_rad": float(np.sum(np.abs(turning))),
        "curvature_scaled": k_norm,
        "high_mask": high,
    }


def _d25_mask_center(mask: np.ndarray) -> np.ndarray:
    m = cv2.moments(mask, binaryImage=True)
    if abs(float(m["m00"])) > 1e-6:
        return np.asarray([m["m10"] / m["m00"], m["m01"] / m["m00"]], np.float32)
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return np.asarray([0.0, 0.0], np.float32)
    return np.asarray([float(np.mean(xs)), float(np.mean(ys))], np.float32)


def _d25_unit(v: np.ndarray) -> Optional[np.ndarray]:
    a = np.asarray(v, np.float32).reshape(2)
    n = float(np.linalg.norm(a))
    return None if n < 1e-6 else a / n


def _d25_pose_axis_candidates(obs: Any, contour: np.ndarray) -> List[Tuple[str, np.ndarray]]:
    candidates: List[Tuple[str, np.ndarray]] = []
    pose = getattr(obs, "pose", None)
    if pose is not None:
        kp = getattr(pose, "keypoints_px", {}) or {}
        wc = kp.get("waist_center")
        hem_pts = [kp.get(n) for n in (
            "img_left_hem_outer", "img_left_hem_inner",
            "img_right_hem_inner", "img_right_hem_outer",
        ) if kp.get(n) is not None]
        if wc is not None and hem_pts:
            lower = np.mean(np.asarray(hem_pts, np.float32), axis=0)
            u = _d25_unit(lower - np.asarray(wc, np.float32))
            if u is not None:
                candidates.append(("pose_waist_to_hems", u))
        wl, wr = kp.get("waist_img_left"), kp.get("waist_img_right")
        if wl is not None and wr is not None:
            waist_u = _d25_unit(np.asarray(wr, np.float32) - np.asarray(wl, np.float32))
            if waist_u is not None:
                candidates.append(("pose_waist_normal", np.asarray([-waist_u[1], waist_u[0]], np.float32)))

    pts = np.asarray(contour, np.float32).reshape(-1, 2)
    if len(pts) >= 4:
        _, eigvec, _ = cv2.PCACompute2(pts, mean=None)
        for i, name in enumerate(("pca_major", "pca_minor")):
            if i < len(eigvec):
                u = _d25_unit(eigvec[i])
                if u is not None:
                    candidates.append((name, u))

    unique: List[Tuple[str, np.ndarray]] = []
    for name, axis in candidates:
        if any(abs(float(np.dot(axis, old))) > 0.997 for _, old in unique):
            continue
        unique.append((name, axis))
    return unique


def _d25_affine_canonical(mask: np.ndarray, axis: np.ndarray, sign: float, args) -> Optional[Dict[str, Any]]:
    v = _d25_unit(np.asarray(axis, np.float32) * float(sign))
    if v is None:
        return None
    lateral = np.asarray([-v[1], v[0]], np.float32)
    contour = _d25_largest_contour(mask)
    if contour is None:
        return None
    pts = np.asarray(contour, np.float32).reshape(-1, 2)
    proj_x = pts @ lateral
    proj_y = pts @ v
    xmin, xmax = float(np.min(proj_x)), float(np.max(proj_x))
    ymin, ymax = float(np.min(proj_y)), float(np.max(proj_y))
    xr, yr = xmax - xmin, ymax - ymin
    if xr < 20.0 or yr < 20.0:
        return None
    margin = 14
    max_side = max(240, int(args.d25_canonical_max_side))
    scale = min((max_side - 2 * margin) / xr, (max_side - 2 * margin) / yr)
    out_w = max(64, int(round(xr * scale)) + 2 * margin)
    out_h = max(64, int(round(yr * scale)) + 2 * margin)
    M = np.asarray([
        [lateral[0] * scale, lateral[1] * scale, margin - xmin * scale],
        [v[0] * scale, v[1] * scale, margin - ymin * scale],
    ], np.float32)
    can = cv2.warpAffine(mask, M, (out_w, out_h), flags=cv2.INTER_NEAREST, borderValue=0)
    can = ((can > 0).astype(np.uint8) * 255)
    inv = cv2.invertAffineTransform(M)
    return {
        "mask": can, "M": M, "M_inv": inv, "axis_px": v,
        "lateral_px": lateral, "scale": float(scale),
    }


def _d25_runs(binary_row: np.ndarray, close_px: int = 2) -> List[Tuple[int, int]]:
    row = (np.asarray(binary_row).reshape(-1) > 0).astype(np.uint8) * 255
    if close_px > 0:
        k = 2 * int(close_px) + 1
        row = cv2.morphologyEx(row.reshape(1, -1), cv2.MORPH_CLOSE, np.ones((1, k), np.uint8)).reshape(-1)
    idx = np.flatnonzero(row > 0)
    if idx.size == 0:
        return []
    cuts = np.where(np.diff(idx) > 1)[0]
    starts = np.r_[idx[0], idx[cuts + 1]]
    ends = np.r_[idx[cuts], idx[-1]]
    return [(int(a), int(b)) for a, b in zip(starts, ends)]


def _d25_transform_points(M: np.ndarray, points: Sequence[Sequence[float]]) -> np.ndarray:
    pts = np.asarray(points, np.float32).reshape(-1, 2)
    if pts.size == 0:
        return pts
    return cv2.transform(pts.reshape(1, -1, 2), np.asarray(M, np.float32)).reshape(-1, 2)


def _d25_profile_analysis(can: np.ndarray, args) -> Dict[str, Any]:
    h, w = can.shape[:2]
    nbin = max(40, int(args.d25_profile_bins))
    ys = np.linspace(0, h, nbin + 1).astype(np.int32)
    raw: List[Dict[str, Any]] = []
    max_span = 1.0
    for i in range(nbin):
        y0, y1 = int(ys[i]), max(int(ys[i]) + 1, int(ys[i + 1]))
        row = np.any(can[y0:y1] > 0, axis=0).astype(np.uint8)
        runs = _d25_runs(row, close_px=max(1, int(round(w * 0.004))))
        span = float(runs[-1][1] - runs[0][0] + 1) if runs else 0.0
        max_span = max(max_span, span)
        raw.append({"i": i, "y": 0.5 * (y0 + y1), "yn": (i + 0.5) / nbin, "runs_all": runs, "span": span})

    min_run = max(3.0, 0.055 * max_span)
    min_gap = max(3.0, 0.035 * max_span)
    for row in raw:
        runs = [(a, b) for a, b in row["runs_all"] if (b - a + 1) >= min_run]
        runs = sorted(runs, key=lambda ab: ab[0])
        row["runs"] = runs
        row["count"] = len(runs)
        row["split"] = False
        row["gap"] = 0.0
        row["gap_center"] = None
        if len(runs) >= 2:
            # Use the two widest non-overlapping runs; retain spatial order.
            pair = sorted(sorted(runs, key=lambda ab: (ab[1] - ab[0]), reverse=True)[:2], key=lambda ab: ab[0])
            gap = float(pair[1][0] - pair[0][1] - 1)
            if gap >= min_gap:
                row["split"] = True
                row["gap"] = gap
                row["gap_center"] = 0.5 * (pair[0][1] + pair[1][0])
                row["pair"] = pair

    def rows_between(a: float, b: float) -> List[Dict[str, Any]]:
        return [r for r in raw if a <= float(r["yn"]) <= b and r["span"] > 0]

    top = rows_between(0.04, 0.23)
    top_one_ratio = float(np.mean([r["count"] == 1 for r in top])) if top else 0.0
    top_widths = [r["span"] for r in top if r["span"] > 0]
    waist_width = float(np.median(top_widths)) if top_widths else float(max_span)

    split_flags = np.asarray([bool(r["split"]) for r in raw], np.uint8)
    onset_idx = None
    for i in range(max(1, int(0.15 * nbin)), int(0.82 * nbin)):
        win = split_flags[i:min(nbin, i + 6)]
        if len(win) >= 4 and int(np.sum(win)) >= 4:
            onset_idx = i
            break
    if onset_idx is None:
        # A shallow but real crotch may yield intermittent split rows.
        candidates = [i for i, f in enumerate(split_flags) if f and 0.16 <= raw[i]["yn"] <= 0.84]
        onset_idx = candidates[0] if candidates else None
    crotch_y = float(raw[onset_idx]["yn"]) if onset_idx is not None else -1.0

    lower_start = max(0.42, crotch_y + 0.04) if crotch_y >= 0 else 0.48
    lower = rows_between(lower_start, 0.93)
    hem = rows_between(0.76, 0.97)
    lower_two_ratio = float(np.mean([r["split"] for r in lower])) if lower else 0.0
    hem_two_ratio = float(np.mean([r["split"] for r in hem])) if hem else 0.0
    gap_rows = [r for r in raw if r["split"] and (crotch_y < 0 or r["yn"] >= crotch_y)]
    max_gap = max([float(r["gap"]) for r in gap_rows] or [0.0])
    crotch_gap_ratio = max_gap / max(1.0, waist_width)
    gap_centers = [float(r["gap_center"]) for r in gap_rows if r.get("gap_center") is not None]
    split_x = float(np.median(gap_centers)) if gap_centers else 0.5 * w

    y_lower_px = int(round(lower_start * h))
    lower_mask = can.copy()
    lower_mask[:max(0, y_lower_px)] = 0
    yy, xx = np.where(lower_mask > 0)
    left_area = int(np.sum(xx < split_x)) if len(xx) else 0
    right_area = int(np.sum(xx >= split_x)) if len(xx) else 0
    leg_balance = float(min(left_area, right_area) / max(1, max(left_area, right_area)))

    hem_pairs = [r.get("pair") for r in hem if r.get("split") and r.get("pair")]
    left_hem_w = float(np.median([p[0][1] - p[0][0] + 1 for p in hem_pairs])) if hem_pairs else 0.0
    right_hem_w = float(np.median([p[1][1] - p[1][0] + 1 for p in hem_pairs])) if hem_pairs else 0.0
    min_hem_waist_ratio = min(left_hem_w, right_hem_w) / max(1.0, waist_width)

    # Geometry points in canonical coordinates.
    top_rows_with_runs = [r for r in top if r["runs"]]
    if top_rows_with_runs:
        waist_y = float(np.median([r["y"] for r in top_rows_with_runs]))
        waist_l = float(np.median([r["runs"][0][0] for r in top_rows_with_runs]))
        waist_r = float(np.median([r["runs"][-1][1] for r in top_rows_with_runs]))
    else:
        waist_y, waist_l, waist_r = 0.08 * h, 0.2 * w, 0.8 * w
    waist_c = 0.5 * (waist_l + waist_r)

    if onset_idx is not None:
        near = [r for r in raw[max(0, onset_idx - 2):min(nbin, onset_idx + 5)] if r.get("split")]
        crotch_row = near[0] if near else raw[onset_idx]
        crotch_x = float(crotch_row.get("gap_center") if crotch_row.get("gap_center") is not None else split_x)
        crotch_y_px = float(crotch_row["y"])
    else:
        crotch_x, crotch_y_px = split_x, 0.55 * h

    hem_rows = [r for r in hem if r.get("split") and r.get("pair")]
    if hem_rows:
        hr = max(hem_rows, key=lambda r: float(r["yn"]))
        pair = hr["pair"]
        hem_y = float(hr["y"])
        hem_points = [
            (float(pair[0][0]), hem_y), (float(pair[0][1]), hem_y),
            (float(pair[1][0]), hem_y), (float(pair[1][1]), hem_y),
        ]
    else:
        hem_y = 0.92 * h
        hem_points = [(0.15*w, hem_y), (0.35*w, hem_y), (0.65*w, hem_y), (0.85*w, hem_y)]

    return {
        "valid": True,
        "rows": raw,
        "max_span_px": float(max_span),
        "waist_width_px": float(waist_width),
        "top_one_row_ratio": top_one_ratio,
        "crotch_found": onset_idx is not None,
        "crotch_y_ratio": crotch_y,
        "crotch_gap_ratio": float(crotch_gap_ratio),
        "lower_two_leg_row_ratio": lower_two_ratio,
        "hem_two_leg_row_ratio": hem_two_ratio,
        "leg_area_balance": leg_balance,
        "left_leg_area_px": left_area,
        "right_leg_area_px": right_area,
        "left_hem_width_px": left_hem_w,
        "right_hem_width_px": right_hem_w,
        "min_hem_width_waist_ratio": float(min_hem_waist_ratio),
        "split_x_px": split_x,
        "geometry_points_can": {
            "waist_img_left": [waist_l, waist_y],
            "waist_center": [waist_c, waist_y],
            "waist_img_right": [waist_r, waist_y],
            "crotch": [crotch_x, crotch_y_px],
            "img_left_hem_outer": list(hem_points[0]),
            "img_left_hem_inner": list(hem_points[1]),
            "img_right_hem_inner": list(hem_points[2]),
            "img_right_hem_outer": list(hem_points[3]),
        },
    }


def _d25_pose_order_score(obs: Any, M: np.ndarray, profile: Dict[str, Any]) -> Dict[str, Any]:
    pose = getattr(obs, "pose", None)
    kp = getattr(pose, "keypoints_px", {}) if pose is not None else {}
    needed = [n for n in d23.BOTTOM_POSE_KPT_NAMES if n in kp]
    if len(needed) < 4:
        return {"score": 0.0, "valid_points": len(needed), "checks": {}}
    can = {n: _d25_transform_points(M, [kp[n]])[0] for n in needed}
    h_est = max(1.0, max(float(p[1]) for p in can.values()) - min(float(p[1]) for p in can.values()))
    checks: Dict[str, bool] = {}
    if "waist_center" in can and "crotch" in can:
        checks["waist_before_crotch"] = float(can["waist_center"][1]) + 0.08*h_est < float(can["crotch"][1])
    hems = [can[n] for n in ("img_left_hem_outer", "img_left_hem_inner", "img_right_hem_inner", "img_right_hem_outer") if n in can]
    if "crotch" in can and hems:
        checks["crotch_before_hems"] = float(can["crotch"][1]) < float(np.median([p[1] for p in hems]))
    if "waist_img_left" in can and "waist_img_right" in can:
        checks["waist_span"] = abs(float(can["waist_img_right"][0] - can["waist_img_left"][0])) >= 0.25 * float(profile["waist_width_px"])
    if len(hems) >= 3:
        checks["hem_longitudinal_consistency"] = float(np.std([p[1] for p in hems])) <= 0.25*h_est
    score = float(np.mean(list(checks.values()))) if checks else 0.0
    return {"score": score, "valid_points": len(needed), "checks": checks}


def _d25_convexity_defects(can: np.ndarray, profile: Dict[str, Any], args) -> Dict[str, Any]:
    contour = _d25_largest_contour(can)
    if contour is None or len(contour) < 8:
        return {"valid": False, "unexpected": [], "all": []}
    eps = max(1.0, 0.003 * cv2.arcLength(contour, True))
    approx = cv2.approxPolyDP(contour, eps, True)
    if len(approx) < 4:
        approx = contour
    hull = cv2.convexHull(approx, returnPoints=False)
    if hull is None or len(hull) < 3:
        return {"valid": False, "unexpected": [], "all": []}
    defects = cv2.convexityDefects(approx, hull)
    h, w = can.shape[:2]
    waist_w = max(1.0, float(profile["waist_width_px"]))
    crotch = np.asarray(profile["geometry_points_can"]["crotch"], np.float32)
    all_items: List[Dict[str, Any]] = []
    unexpected: List[Dict[str, Any]] = []
    if defects is not None:
        for row in defects.reshape(-1, 4):
            s, e, f, depth_raw = [int(x) for x in row]
            if max(s, e, f) >= len(approx):
                continue
            far = np.asarray(approx[f, 0], np.float32)
            depth_px = float(depth_raw) / 256.0
            ratio = depth_px / waist_w
            yn = float(far[1]) / max(1.0, h - 1.0)
            xn = float(far[0]) / max(1.0, w - 1.0)
            expected_crotch = bool(
                np.linalg.norm(far - crotch) <= 0.20 * waist_w
                and 0.15 <= yn <= 0.86
            )
            structural_zone = bool(yn <= 0.21 or yn >= 0.91)
            item = {
                "point_can": [float(far[0]), float(far[1])],
                "depth_px": depth_px,
                "depth_ratio": ratio,
                "expected_crotch": expected_crotch,
                "structural_zone": structural_zone,
                "x_ratio": xn, "y_ratio": yn,
            }
            all_items.append(item)
            if ratio >= float(args.d25_defect_min_depth_ratio) and not expected_crotch and not structural_zone:
                unexpected.append(item)
    # Normal shorts often have a symmetric pair of concavities where the hip
    # body narrows into the two legs. Convex-hull defects report these as dents,
    # but they are expected garment structure. Remove only a geometrically
    # symmetric left/right pair; a one-sided or much deeper indentation remains.
    structural_side_ids = set()
    left_ids = [i for i, x in enumerate(all_items)
                if not x["expected_crotch"] and not x["structural_zone"]
                and 0.08 <= float(x["x_ratio"]) <= 0.38
                and 0.24 <= float(x["y_ratio"]) <= 0.72]
    right_ids = [i for i, x in enumerate(all_items)
                 if not x["expected_crotch"] and not x["structural_zone"]
                 and 0.62 <= float(x["x_ratio"]) <= 0.92
                 and 0.24 <= float(x["y_ratio"]) <= 0.72]
    best_pair = None
    best_pair_error = 1e9
    for li in left_ids:
        for ri in right_ids:
            a, b = all_items[li], all_items[ri]
            yerr = abs(float(a["y_ratio"]) - float(b["y_ratio"]))
            mirror = abs((float(a["x_ratio"]) + float(b["x_ratio"])) - 1.0)
            da, db = float(a["depth_ratio"]), float(b["depth_ratio"])
            derr = abs(da-db) / max(1e-6, max(da,db))
            err = yerr + mirror + 0.5*derr
            if yerr <= 0.075 and mirror <= 0.12 and derr <= 0.34 and err < best_pair_error:
                best_pair = (li,ri); best_pair_error = err
    if best_pair is not None:
        structural_side_ids.update(best_pair)
        for i in best_pair:
            all_items[i]["expected_leg_junction"] = True

    unexpected = []
    for i, item in enumerate(all_items):
        if i in structural_side_ids:
            continue
        if (float(item["depth_ratio"]) >= float(args.d25_defect_min_depth_ratio)
                and not item["expected_crotch"] and not item["structural_zone"]):
            unexpected.append(item)
    max_ratio = max([float(x["depth_ratio"]) for x in unexpected] or [0.0])
    return {
        "valid": True,
        "all": all_items,
        "unexpected": unexpected,
        "expected_leg_junction_count": len(structural_side_ids),
        "unexpected_count": len(unexpected),
        "max_unexpected_depth_ratio": max_ratio,
    }


def _d25_geometry_points_and_fusion(obs: Any, H: Optional[np.ndarray], M_inv: np.ndarray,
                                    profile: Dict[str, Any], args) -> Dict[str, Any]:
    geom_can = profile["geometry_points_can"]
    names = list(d23.BOTTOM_POSE_KPT_NAMES)
    geom_px_arr = _d25_transform_points(M_inv, [geom_can[n] for n in names])
    geom_px = {n: geom_px_arr[i].astype(np.float32) for i, n in enumerate(names)}
    pose = getattr(obs, "pose", None)
    pose_px = getattr(pose, "keypoints_px", {}) if pose is not None else {}
    pose_conf = getattr(pose, "keypoint_conf", {}) if pose is not None else {}

    # Resolve possible left/right semantic reversal by minimizing the distance to
    # the medium pose output. This does not affect topology, only point labels.
    def maybe_swap(group_a: Sequence[str], group_b: Sequence[str]) -> None:
        available = all(n in pose_px for n in list(group_a) + list(group_b))
        if not available:
            return
        direct = sum(float(np.linalg.norm(geom_px[a] - np.asarray(pose_px[a], np.float32))) for a in group_a)
        direct += sum(float(np.linalg.norm(geom_px[b] - np.asarray(pose_px[b], np.float32))) for b in group_b)
        swap = sum(float(np.linalg.norm(geom_px[b] - np.asarray(pose_px[a], np.float32))) for a, b in zip(group_a, group_b))
        swap += sum(float(np.linalg.norm(geom_px[a] - np.asarray(pose_px[b], np.float32))) for a, b in zip(group_a, group_b))
        if swap + 1e-6 < direct:
            old_a = [geom_px[n].copy() for n in group_a]
            old_b = [geom_px[n].copy() for n in group_b]
            for n, p in zip(group_a, old_b): geom_px[n] = p
            for n, p in zip(group_b, old_a): geom_px[n] = p

    maybe_swap(("waist_img_left",), ("waist_img_right",))
    maybe_swap(("img_left_hem_outer", "img_left_hem_inner"),
               ("img_right_hem_outer", "img_right_hem_inner"))

    fused_px: Dict[str, List[float]] = {}
    source: Dict[str, str] = {}
    waist_width_px = max(20.0, float(profile["waist_width_px"]))
    max_dist = max(18.0, float(args.d25_fuse_max_distance_waist_ratio) * waist_width_px)
    gw = float(np.clip(args.d25_geometry_weight, 0.0, 1.0))
    for n in names:
        g = geom_px[n]
        if bool(args.d25_pose_geometry_fuse) and n in pose_px:
            p = np.asarray(pose_px[n], np.float32)
            dist = float(np.linalg.norm(p - g))
            conf = float(pose_conf.get(n, 0.0))
            if dist <= max_dist and conf >= 0.20:
                # High confidence gives pose a little more influence, while the
                # contour keeps boundary points attached to the actual garment.
                geom_weight = float(np.clip(gw - 0.18 * conf, 0.42, 0.78))
                f = geom_weight * g + (1.0 - geom_weight) * p
                source[n] = "FUSED"
            else:
                f = g
                source[n] = "GEOMETRY_RECOVERED"
        else:
            f = g
            source[n] = "GEOMETRY"
        fused_px[n] = [float(f[0]), float(f[1])]

    fused_board: Dict[str, List[float]] = {}
    if H is not None:
        for n, p in fused_px.items():
            try:
                bx, by = d23.pixel_to_board(H, float(p[0]), float(p[1]))
                fused_board[n] = [float(bx), float(by)]
            except Exception:
                pass
    return {
        "geometry_points_px": {n: [float(p[0]), float(p[1])] for n, p in geom_px.items()},
        "fused_points_px": fused_px,
        "fused_points_board": fused_board,
        "point_source": source,
        "geometry_recovered_count": sum(v != "FUSED" for v in source.values()),
    }


def _d25_candidate_report(obs: Any, H: Optional[np.ndarray], clean: np.ndarray,
                          contour: np.ndarray, axis_name: str, axis: np.ndarray,
                          sign: float, args) -> Optional[Dict[str, Any]]:
    canonical = _d25_affine_canonical(clean, axis, sign, args)
    if canonical is None:
        return None
    profile = _d25_profile_analysis(canonical["mask"], args)
    pose_order = _d25_pose_order_score(obs, canonical["M"], profile)
    defects = _d25_convexity_defects(canonical["mask"], profile, args)
    can_contour = _d25_largest_contour(canonical["mask"])
    samples = _d25_resample_closed(can_contour, int(args.d25_contour_resample_points)) if can_contour is not None else np.empty((0,2), np.float32)
    smooth = _d25_circular_smooth(samples, int(args.d25_contour_smooth_window))
    differential = _d25_contour_differential(smooth, float(profile["waist_width_px"]), args)

    waist_s = float(np.clip((profile["top_one_row_ratio"] - 0.30) / 0.55, 0.0, 1.0))
    gap_s = float(np.clip((profile["crotch_gap_ratio"] - 0.02) / 0.15, 0.0, 1.0))
    crotch_pos = float(profile["crotch_y_ratio"])
    pos_s = 1.0 if float(args.d25_crotch_y_min) <= crotch_pos <= float(args.d25_crotch_y_max) else 0.0
    crotch_s = gap_s * pos_s if profile["crotch_found"] else 0.0
    leg_s = float(np.clip((profile["lower_two_leg_row_ratio"] - 0.08) / 0.55, 0.0, 1.0))
    hem_s = float(np.clip((profile["hem_two_leg_row_ratio"] - 0.05) / 0.55, 0.0, 1.0))
    balance_s = float(np.clip((profile["leg_area_balance"] - 0.20) / 0.65, 0.0, 1.0))
    pose_s = float(pose_order["score"])
    orientation_score = (
        0.17 * waist_s + 0.24 * crotch_s + 0.25 * leg_s
        + 0.14 * hem_s + 0.10 * balance_s + 0.10 * pose_s
    )
    points = _d25_geometry_points_and_fusion(obs, H, canonical["M_inv"], profile, args)

    defect_points_px: List[List[float]] = []
    if defects.get("unexpected"):
        arr = _d25_transform_points(canonical["M_inv"], [x["point_can"] for x in defects["unexpected"]])
        defect_points_px = [[float(p[0]), float(p[1])] for p in arr]
    expected_crotch_px = points["geometry_points_px"].get("crotch")
    return {
        "axis_source": axis_name,
        "axis_sign": float(sign),
        "orientation_score": float(orientation_score),
        "profile": profile,
        "pose_order": pose_order,
        "defects": defects,
        "differential": {k: v for k, v in differential.items() if k not in {"curvature_scaled", "high_mask"}},
        "points": points,
        "debug": {
            "axis_px": [float(canonical["axis_px"][0]), float(canonical["axis_px"][1])],
            "unexpected_defect_points_px": defect_points_px,
            "expected_crotch_px": expected_crotch_px,
            "canonical_shape": [int(canonical["mask"].shape[0]), int(canonical["mask"].shape[1])],
        },
    }


def d25_shape_report(obs: Any, args, H: Optional[np.ndarray] = None) -> Dict[str, Any]:
    if obs is None or getattr(obs, "mask", None) is None:
        return {"shape_good": False, "shape_score": 0.0, "remaining": ["mask_missing"]}
    clean = _d25_clean_mask(obs.mask.mask_u8, args)
    contour = _d25_largest_contour(clean)
    if contour is None or cv2.contourArea(contour) < 1000.0:
        return {"shape_good": False, "shape_score": 0.0, "remaining": ["contour_invalid"]}

    candidates: List[Dict[str, Any]] = []
    for name, axis in _d25_pose_axis_candidates(obs, contour):
        for sign in (1.0, -1.0):
            cand = _d25_candidate_report(obs, H, clean, contour, name, axis, sign, args)
            if cand is not None:
                candidates.append(cand)
    if not candidates:
        return {"shape_good": False, "shape_score": 0.0, "remaining": ["orientation_unresolved"]}
    best = max(candidates, key=lambda c: float(c["orientation_score"]))
    p = best["profile"]
    d = best["defects"]
    diff = best["differential"]

    mask_area = float(cv2.countNonZero(clean))
    hull = cv2.convexHull(contour)
    hull_area = max(1.0, float(cv2.contourArea(hull)))
    solidity = mask_area / hull_area
    checks = {
        "waist_continuity": float(p["top_one_row_ratio"]) >= float(args.d25_waist_one_row_ratio_min),
        "crotch_structure": bool(p["crotch_found"])
            and float(args.d25_crotch_y_min) <= float(p["crotch_y_ratio"]) <= float(args.d25_crotch_y_max)
            and float(p["crotch_gap_ratio"]) >= float(args.d25_crotch_gap_ratio_min),
        "two_exposed_legs": float(p["lower_two_leg_row_ratio"]) >= float(args.d25_lower_two_leg_row_ratio_min),
        "two_hems": float(p["hem_two_leg_row_ratio"]) >= float(args.d25_hem_two_leg_row_ratio_min)
            and float(p["min_hem_width_waist_ratio"]) >= float(args.d25_min_hem_width_waist_ratio),
        "leg_area_balance": float(p["leg_area_balance"]) >= float(args.d25_leg_area_balance_min),
        "mask_solidity": float(solidity) >= float(args.d25_min_solidity),
    }
    passed = int(sum(bool(v) for v in checks.values()))
    hard_structure = bool(checks["crotch_structure"] and checks["two_exposed_legs"] and checks["two_hems"])

    unexpected_count = int(d.get("unexpected_count", 0))
    max_defect = float(d.get("max_unexpected_depth_ratio", 0.0))
    high_arc = float(diff.get("high_curvature_arc_ratio", 0.0))
    macro_outer_fold = bool(
        max_defect >= float(args.d25_defect_hard_depth_ratio)
        or unexpected_count > int(args.d25_max_moderate_unexpected_defects)
        or (
            unexpected_count >= 1
            and max_defect >= float(args.d25_defect_min_depth_ratio)
            and high_arc > float(args.d25_high_curvature_arc_ratio_max)
        )
    )

    component_scores = [
        float(best["orientation_score"]),
        float(np.mean(list(checks.values()))),
        float(np.clip(p["leg_area_balance"], 0.0, 1.0)),
        float(np.clip(1.0 - max_defect / max(1e-6, float(args.d25_defect_hard_depth_ratio)), 0.0, 1.0)),
    ]
    shape_score = float(0.42*component_scores[0] + 0.38*component_scores[1] + 0.12*component_scores[2] + 0.08*component_scores[3])
    enough_checks = passed >= int(args.d25_min_topology_checks)
    shape_good = bool(
        hard_structure
        and enough_checks
        and shape_score >= float(args.d25_shape_score_min)
        and not macro_outer_fold
    )
    remaining = [k for k, v in checks.items() if not v]
    if not hard_structure:
        remaining.append("pants_topology")
    if not enough_checks:
        remaining.append("topology_check_count")
    if shape_score < float(args.d25_shape_score_min):
        remaining.append("shape_score")
    if macro_outer_fold:
        remaining.append("outer_macro_fold")

    # Legacy fields remain for existing panel/log consumers, but are no longer
    # hard symmetric-shape conditions.
    return {
        "policy": "D25_REFERENCE_FREE_CONTOUR_TOPOLOGY",
        "shape_good": shape_good,
        "shape_score": shape_score,
        "orientation_score": float(best["orientation_score"]),
        "axis_source": str(best["axis_source"]),
        "axis_sign": float(best["axis_sign"]),
        "hard_checks": checks,
        "passed_check_count": passed,
        "required_check_count": int(args.d25_min_topology_checks),
        "hard_structure_good": hard_structure,
        "remaining": list(dict.fromkeys(remaining)),
        "topology": {
            "waist_one_row_ratio": float(p["top_one_row_ratio"]),
            "crotch_found": bool(p["crotch_found"]),
            "crotch_y_ratio": float(p["crotch_y_ratio"]),
            "crotch_gap_ratio": float(p["crotch_gap_ratio"]),
            "lower_two_leg_row_ratio": float(p["lower_two_leg_row_ratio"]),
            "hem_two_leg_row_ratio": float(p["hem_two_leg_row_ratio"]),
            "leg_area_balance": float(p["leg_area_balance"]),
            "min_hem_width_waist_ratio": float(p["min_hem_width_waist_ratio"]),
            "solidity": float(solidity),
        },
        "contour_differential": best["differential"],
        "convexity_defects": best["defects"],
        "macro_outer_fold": macro_outer_fold,
        "pose_geometry": {
            "pose_order_score": float(best["pose_order"]["score"]),
            "pose_order_checks": best["pose_order"]["checks"],
            **best["points"],
        },
        "debug": best["debug"],
        "candidate_count": len(candidates),
        "candidate_scores": [
            {"axis_source": c["axis_source"], "axis_sign": c["axis_sign"], "score": c["orientation_score"]}
            for c in sorted(candidates, key=lambda c: float(c["orientation_score"]), reverse=True)[:8]
        ],
        # Compatibility diagnostics, not symmetry hard gates.
        "hem_gap_ratio": float(p["min_hem_width_waist_ratio"]),
        "leg_balance": float(p["leg_area_balance"]),
        "crotch_axis_offset_ratio": 0.0,
        "axis_to_waist_ratio": 1.0,
        "symmetry": {
            "lower_reflected_iou": 0.0,
            "lower_width_balance_median": float(p["leg_area_balance"]),
            "lower_width_balance_min": float(p["leg_area_balance"]),
            "symmetry_score": float(p["leg_area_balance"]),
            "good": bool(checks["leg_area_balance"]),
        },
    }


def _d25_fold_report_from_heat(obs: Any, heat: Any, H: np.ndarray, args,
                               shape: Dict[str, Any]) -> Dict[str, Any]:
    if heat is None:
        return {
            "good": False, "reason": "heatmap unavailable", "fold_count": 0,
            "t3_count": 0, "fine_allowed_count": 0, "remaining": ["heatmap"],
        }
    robust = robust_heat_threshold(heat, args)
    waist_ref = _d24v2_waist_reference(obs, heat, H, robust, args)
    candidates = list(getattr(heat, "d21v4_all_candidates", [])) or list(getattr(heat, "candidates", []))
    blockers: List[Dict[str, Any]] = []
    gathers: List[Dict[str, Any]] = []
    seams: List[Dict[str, Any]] = []
    fine: List[Dict[str, Any]] = []
    rejected: List[Dict[str, Any]] = []
    t1: List[Dict[str, Any]] = []
    t2: List[Dict[str, Any]] = []
    for original in candidates:
        cand = dict(original)
        seam = seam_veto_report(cand, heat, obs, H, args)
        cand["d24_seam"] = seam
        if bool(seam.get("matched", False)):
            seams.append(cand); continue
        support = candidate_robust_support(cand, heat, float(robust["threshold"]))
        cand["d24_robust_support"] = support
        if support < float(args.d24_candidate_min_robust_support):
            rejected.append(cand); continue
        metrics = _d24v2_candidate_metrics(cand, heat, obs, waist_ref)
        cand["d24v2_metrics"] = metrics
        gather = _d24v5_waist_gather_report(cand, heat, obs, H, metrics, args)
        cand["d24v5_waist_gather"] = gather
        if bool(gather.get("matched", False)):
            gathers.append(cand); continue
        fold = _d24v6_fold_report(cand, metrics, heat, args)
        cand["d24v6_fold"] = fold
        if bool(fold.get("is_fold", False)):
            blockers.append(cand)
        else:
            fine.append(cand)
        tier = int(metrics.get("tier", 0))
        if tier <= 1: t1.append(cand)
        elif tier == 2: t2.append(cand)

    # A contour-detected outer fold is also a real blocker even when its texture
    # contrast is weak (for example, a uniformly bright folded flap).
    outer_block = bool(shape.get("macro_outer_fold", False))
    mask_area = max(1.0, float(obs.mask.area_px))
    effective_area = max(1.0, float(cv2.countNonZero(heat.inner_mask)))
    blocker_area = float(sum(float(c.get("area_px", 0.0)) for c in blockers))
    checks = {
        "heat_fold_count": len(blockers) <= int(args.finish_max_t3_count),
        "outer_macro_fold": not outer_block,
    }
    good = bool(all(checks.values()))
    return {
        "policy": "D25_MACRO_FOLD_ONLY_HEAT_PLUS_CONTOUR",
        "good": good,
        "checks": checks,
        "fold_count": len(blockers) + int(outer_block),
        "heat_fold_count": len(blockers),
        "outer_macro_fold_count": int(outer_block),
        "t3_count": len(blockers) + int(outer_block),
        "t2_count": len(t2),
        "t1_allowed_count": len(t1),
        "fine_allowed_count": len(fine),
        "waist_gather_ignored_count": len(gathers),
        "center_rise_ignored_px": int((getattr(heat, "center_rise_info", {}) or {}).get("ignored_px", 0)),
        "seam_ignored_count": len(seams),
        "robust_rejected_count": len(rejected),
        "waist_structure_ignored_count": int(getattr(heat, "d23_waist_structure_ignored_count", 0)),
        "fine_ignored_count": len(fine) + int(getattr(heat, "d21v4_fine_ignored_count", 0)),
        "actionable_ratio": blocker_area / effective_area,
        "max_blob_ratio": max([float(c.get("area_px", 0.0))/mask_area for c in blockers] or [0.0]),
        "waist_reference": waist_ref,
        "robust": robust,
        "a119_summary": str(getattr(heat, "summary", "")),
        "remaining": [k for k, v in checks.items() if not v],
        "t1": t1, "t2": t2, "t3": blockers,
        "major_blockers": blockers, "fold_blockers": blockers,
        "waist_gather_ignored": gathers, "fine_allowed": fine,
        "seam_ignored": seams, "robust_rejected": rejected,
        "relative_allowed": [], "non_major_allowed": fine,
    }


def _d25_decide(obs: Any, heat: Any, H: np.ndarray, state: D24State, args,
                snapshot_mode: bool) -> Dict[str, Any]:
    state.evaluation_count += 1
    quality = observation_quality_report(obs, args)
    stability = {
        "good": True, "reason": "single frozen snapshot" if snapshot_mode else "pending",
        "history_count": 1 if snapshot_mode else len(state.history),
        "required": 1 if snapshot_mode else max(2, int(args.history_count)),
        "mask_area_rel": 0.0, "center_shift_mm": 0.0, "pose_axis_spread_deg": 0.0,
    }
    if not quality["good"]:
        state.ready_streak = 0
        return {
            "status": "REJUDGE", "reason": "OBSERVATION_INVALID",
            "evaluation": state.evaluation_count, "quality": quality,
            "stability": stability, "shape": {}, "wrinkle": {},
            "snapshot_mode": snapshot_mode,
        }
    if not snapshot_mode:
        state.history.append(observation_entry(obs))
        stability = temporal_stability_report(state.history, args)
        if not stability["good"]:
            state.ready_streak = 0
            return {
                "status": "REJUDGE", "reason": "OBSERVATION_UNSTABLE",
                "evaluation": state.evaluation_count, "quality": quality,
                "stability": stability, "shape": {}, "wrinkle": {},
                "snapshot_mode": False,
            }

    shape = d25_shape_report(obs, args, H)
    if not bool(shape.get("shape_good", False)):
        state.ready_streak = 0
        return {
            "status": "NOT_READY_SHAPE", "reason": "REFERENCE_FREE_TOPOLOGY_FAILED",
            "evaluation": state.evaluation_count, "quality": quality,
            "stability": stability, "shape": shape, "wrinkle": {},
            "ready_streak": 0, "ready_required": 1 if snapshot_mode else int(args.ready_confirm_count),
            "snapshot_mode": snapshot_mode,
        }
    if heat is None:
        state.ready_streak = 0
        return {
            "status": "REJUDGE", "reason": "MACRO_FOLD_HEATMAP_UNAVAILABLE",
            "evaluation": state.evaluation_count, "quality": quality,
            "stability": stability, "shape": shape, "wrinkle": {},
            "snapshot_mode": snapshot_mode,
        }
    wrinkle = _d25_fold_report_from_heat(obs, heat, H, args, shape)
    if not wrinkle["good"]:
        state.ready_streak = 0
        return {
            "status": "NOT_READY_WRINKLE", "reason": "MACRO_FOLD_OR_OVERLAP_REMAINS",
            "evaluation": state.evaluation_count, "quality": quality,
            "stability": stability, "shape": shape, "wrinkle": wrinkle,
            "ready_streak": 0, "ready_required": 1 if snapshot_mode else int(args.ready_confirm_count),
            "snapshot_mode": snapshot_mode,
        }
    if snapshot_mode:
        state.ready_streak = 1
        status = "READY_GOOD_ENOUGH"
        required = 1
    else:
        state.ready_streak += 1
        required = max(1, int(args.ready_confirm_count))
        status = "READY_GOOD_ENOUGH" if state.ready_streak >= required else "READY_PENDING"
    return {
        "status": status,
        "reason": "PANTS_TOPOLOGY_OK_AND_NO_MACRO_FOLD",
        "evaluation": state.evaluation_count, "quality": quality,
        "stability": stability, "shape": shape, "wrinkle": wrinkle,
        "ready_streak": state.ready_streak, "ready_required": required,
        "snapshot_mode": snapshot_mode,
    }


def evaluate_d24(obs: Any, heat: Any, H: np.ndarray, state: D24State, args) -> Dict[str, Any]:
    return _d25_decide(obs, heat, H, state, args, snapshot_mode=False)


def evaluate_d24_snapshot(obs: Any, heat: Any, H: np.ndarray, state: D24State, args) -> Dict[str, Any]:
    return _d25_decide(obs, heat, H, state, args, snapshot_mode=True)


def append_status_panel(image: np.ndarray, report: Dict[str, Any], args) -> np.ndarray:
    h = max(image.shape[0], 760)
    panel_w = max(560, int(args.panel_width))
    panel = np.zeros((h, panel_w, 3), np.uint8)
    panel[:] = (24, 24, 24)
    y = 33
    def line(text: str, color=(230,230,230), scale=0.45, gap=23):
        nonlocal y
        put_text(panel, str(text)[:112], (17, y), color, scale, 1)
        y += gap
    status = str(report.get("status", "REJUDGE"))
    line(f"D25 FINISH: {status}", STATUS_COLORS.get(status, (255,255,255)), 0.70, 33)
    line(f"reason: {report.get('reason','')}", (210,210,210), 0.43, 26)
    q = report.get("quality", {}) or {}
    line("[1] MEDIUM POSE / MASK", (255,220,120), 0.50, 24)
    line(f"quality {bool_mark(q.get('good'))}  kpts={q.get('valid_keypoints',0)} conf={float(q.get('mean_kpt_conf',0)):.2f}")
    line(f"TTA={q.get('pose_tta_state','-')} score={float(q.get('pose_tta_score',0)):.1f} tested={q.get('pose_tta_tested',0)}")
    line(f"poseValid(advisory)={q.get('pose_valid_advisory',False)} geometryRecover={q.get('geometry_recoverable',False)}", (190,190,190), 0.39)

    sh = report.get("shape", {}) or {}
    top = sh.get("topology", {}) or {}
    line("[2] REFERENCE-FREE TOPOLOGY", (255,220,120), 0.50, 24)
    line(f"shape {bool_mark(sh.get('shape_good'))} score={float(sh.get('shape_score',0)):.2f} axis={sh.get('axis_source','-')} cand={sh.get('candidate_count',0)}")
    line(f"waist1={float(top.get('waist_one_row_ratio',0)):.2f} crotch={top.get('crotch_found',False)} gap={float(top.get('crotch_gap_ratio',0)):.2f} y={float(top.get('crotch_y_ratio',-1)):.2f}")
    line(f"twoLeg={float(top.get('lower_two_leg_row_ratio',0)):.2f} twoHem={float(top.get('hem_two_leg_row_ratio',0)):.2f}")
    line(f"legBalance={float(top.get('leg_area_balance',0)):.2f} hem/waist={float(top.get('min_hem_width_waist_ratio',0)):.2f} solidity={float(top.get('solidity',0)):.2f}")
    pg = sh.get("pose_geometry", {}) or {}
    line(f"poseOrder={float(pg.get('pose_order_score',0)):.2f} geometryRecovered={pg.get('geometry_recovered_count',0)}/8", (190,190,190), 0.40)
    if sh.get("remaining"):
        line("fail: " + ", ".join(str(x) for x in sh["remaining"]), (80,120,255), 0.40)

    defects = sh.get("convexity_defects", {}) or {}
    diff = sh.get("contour_differential", {}) or {}
    line("[3] CONTOUR dP/ds, d2P/ds2", (255,220,120), 0.50, 24)
    line(f"unexpectedDent={defects.get('unexpected_count',0)} maxDepth/W={float(defects.get('max_unexpected_depth_ratio',0)):.3f}")
    line(f"highCurvArc={float(diff.get('high_curvature_arc_ratio',0)):.3f} p90Curv={float(diff.get('p90_abs_curvature_scaled',0)):.3f}")
    line(f"outerMacroFold={sh.get('macro_outer_fold',False)}", (0,0,255) if sh.get('macro_outer_fold') else (0,210,0), 0.44)

    wr = report.get("wrinkle", {}) or {}
    line("[4] MACRO FOLD ONLY", (255,220,120), 0.50, 24)
    line(f"foldGate {bool_mark(wr.get('good'))} total={wr.get('fold_count',0)} heat={wr.get('heat_fold_count',0)} outer={wr.get('outer_macro_fold_count',0)}")
    line(f"fineAllowed={wr.get('fine_allowed_count',0)} waistGather={wr.get('waist_gather_ignored_count',0)} seam={wr.get('seam_ignored_count',0)}")
    line(f"foldRatio={float(wr.get('actionable_ratio',0)):.3f} maxBlob={float(wr.get('max_blob_ratio',0)):.3f}")

    line("[5] DECISION", (255,220,120), 0.50, 24)
    line(f"READY={report.get('ready_streak',0)}/{report.get('ready_required',1)}")
    line("No garment registration/reference is used.", (180,180,180), 0.42)
    line("SPACE force | C reset | S save | L lock H | R reset H | Q quit", (180,180,180), 0.36)
    canvas = np.zeros((h, image.shape[1] + panel_w, 3), np.uint8)
    canvas[:image.shape[0], :image.shape[1]] = image
    canvas[:, image.shape[1]:] = panel
    return canvas


def draw_d24_overlay(frame: np.ndarray, obs: Any, heat: Any, report: Dict[str, Any], H: np.ndarray, cfg, args) -> np.ndarray:
    out = _D25_BASE_DRAW_OVERLAY(frame, obs, heat, report, H, cfg, args)
    if not bool(getattr(args, "d25_draw_geometry", True)):
        return out
    sh = report.get("shape", {}) or {}
    pg = sh.get("pose_geometry", {}) or {}
    points = pg.get("fused_points_px", {}) or {}
    colors = {
        "waist_img_left": (255,220,0), "waist_center": (0,255,255), "waist_img_right": (255,220,0),
        "crotch": (255,0,255),
        "img_left_hem_outer": (255,170,0), "img_left_hem_inner": (255,170,0),
        "img_right_hem_inner": (0,170,255), "img_right_hem_outer": (0,170,255),
    }
    for name, p in points.items():
        x, y = int(round(float(p[0]))), int(round(float(p[1])))
        cv2.circle(out, (x, y), 6, colors.get(name, (255,255,255)), -1)
        put_text(out, name.replace("img_", "")[:12], (x+7, y-5), colors.get(name, (255,255,255)), 0.34, 1)
    dbg = sh.get("debug", {}) or {}
    c = dbg.get("expected_crotch_px")
    axis = dbg.get("axis_px")
    if c is not None and axis is not None:
        cp = np.asarray(c, np.float32); av = np.asarray(axis, np.float32)
        a = tuple(np.round(cp - av*150).astype(int)); b = tuple(np.round(cp + av*150).astype(int))
        cv2.line(out, a, b, (0,255,255), 2)
    for p in dbg.get("unexpected_defect_points_px", []) or []:
        x, y = int(round(float(p[0]))), int(round(float(p[1])))
        cv2.circle(out, (x, y), 10, (0,0,255), 2)
        put_text(out, "UNEXPECTED DENT", (x+8, y+4), (0,0,255), 0.36, 1)
    put_text(out, "D25: MEDIUM POSE + CONTOUR DIFFERENTIAL / NO REFERENCE", (25, out.shape[0]-25), (255,255,255), 0.48, 1)
    return out


def print_report(report: Dict[str, Any]) -> None:
    q = report.get("quality", {}) or {}
    sh = report.get("shape", {}) or {}
    top = sh.get("topology", {}) or {}
    de = sh.get("convexity_defects", {}) or {}
    wr = report.get("wrinkle", {}) or {}
    print(
        f"[D25] status={report.get('status','REJUDGE')} reason={report.get('reason','')} "
        f"kpts={q.get('valid_keypoints',0)} conf={float(q.get('mean_kpt_conf',0)):.2f} "
        f"shape={float(sh.get('shape_score',0)):.2f} axis={sh.get('axis_source','-')} "
        f"waist1={float(top.get('waist_one_row_ratio',0)):.2f} "
        f"crotchGap={float(top.get('crotch_gap_ratio',0)):.2f} "
        f"twoLeg={float(top.get('lower_two_leg_row_ratio',0)):.2f} "
        f"twoHem={float(top.get('hem_two_leg_row_ratio',0)):.2f} "
        f"legBal={float(top.get('leg_area_balance',0)):.2f} "
        f"dent={de.get('unexpected_count',0)}/{float(de.get('max_unexpected_depth_ratio',0)):.3f} "
        f"fold={wr.get('fold_count',0)} ready={report.get('ready_streak',0)}/{report.get('ready_required',1)}"
    )


# =============================================================================
# D25-v2: medium-pose-led shape decision; contour derivatives are veto/support
# =============================================================================
# D25-v1 incorrectly made crotch/two-leg/two-hem scanline topology mandatory.
# Elastic shorts often have a narrow crotch opening that segmentation closing
# fills, so a fully spread garment could fail.  V2 makes the YOLO26m 8-point
# pose the semantic shape authority.  Contour geometry can support/recover the
# pose, but only a conservative, deep outer deformation may veto completion.

STEP_D24_BUILD = "2026-07-17-d25-v2-medium-pose-led-contour-veto"
_D25V2_BASE_BUILD_PARSER = build_parser


def build_parser() -> argparse.ArgumentParser:
    p = _D25V2_BASE_BUILD_PARSER()
    p.description = (
        "D25-v2 pants finish evaluator: YOLO26m pose-led semantic shape + "
        "contour differential support/conservative macro-fold veto; no reference"
    )
    p.set_defaults(
        pose_model="/workspace/project_train/yolo26/bottom_pose8_yolo26m_e40_best.engine",
        d23_pose_tta_fast_first=False,
        min_valid_keypoints=6,
        min_mean_kpt_conf=0.26,
        d25_mask_close_px=3,
        d25_mask_open_px=3,
        output="d25_v2_finish_overlay.jpg",
        snapshot_dir="d25_v2_finish_samples",
        log_jsonl="d25_v2_finish_log.jsonl",
        window_name="D25_V2_POSE_LED_FINISH",
    )
    p.add_argument("--d25v2-pose-score-min", type=float, default=0.56)
    p.add_argument("--d25v2-shape-score-min", type=float, default=0.52)
    p.add_argument("--d25v2-pose-near-mask-min", type=int, default=6)
    p.add_argument("--d25v2-kpt-near-mask-ratio", type=float, default=0.055)
    p.add_argument("--d25v2-geometry-rescue-score-min", type=float, default=0.70)
    p.add_argument("--d25v2-min-mask-solidity", type=float, default=0.32)
    p.add_argument("--d25v2-outer-hard-depth-ratio", type=float, default=0.24)
    p.add_argument("--d25v2-outer-hard-curvature-ratio", type=float, default=0.20)
    p.add_argument("--d25v2-outer-multi-depth-ratio", type=float, default=0.18)
    p.add_argument("--d25v2-outer-multi-count", type=int, default=3)
    p.add_argument("--d25v2-macro-fold-min-length-mm", type=float, default=78.0)
    p.add_argument("--d25v2-macro-fold-min-width-mm", type=float, default=20.0)
    p.add_argument("--d25v2-macro-fold-min-area-ratio", type=float, default=0.010)
    p.add_argument("--d25v2-macro-fold-extreme-width-mm", type=float, default=32.0)
    p.add_argument("--d25v2-macro-fold-extreme-area-ratio", type=float, default=0.024)
    return p


def _d25v2_finite_point(value: Any) -> Optional[np.ndarray]:
    try:
        p = np.asarray(value, np.float32).reshape(2)
    except Exception:
        return None
    return p if np.all(np.isfinite(p)) else None


def _d25v2_pose_structure_report(obs: Any, clean: np.ndarray, args) -> Dict[str, Any]:
    """Evaluate semantic pants structure from the medium 8-point pose.

    This test is rotation invariant because all distances/order checks are
    projected onto a garment-relative body axis (waist center -> hem center).
    It does not require perfect left/right symmetry.
    """
    pose = getattr(obs, "pose", None)
    kp_raw = getattr(pose, "keypoints_px", {}) if pose is not None else {}
    conf_raw = getattr(pose, "keypoint_conf", {}) if pose is not None else {}
    names = list(d23.BOTTOM_POSE_KPT_NAMES)
    pts: Dict[str, np.ndarray] = {}
    confs: Dict[str, float] = {}
    for n in names:
        if n not in kp_raw:
            continue
        p = _d25v2_finite_point(kp_raw[n])
        if p is None:
            continue
        pts[n] = p
        confs[n] = float(conf_raw.get(n, 0.0))

    valid_count = len(pts)
    mean_conf = float(np.mean(list(confs.values()))) if confs else 0.0
    area = max(1.0, float(cv2.countNonZero(clean)))
    near_limit = max(8.0, math.sqrt(area) * float(args.d25v2_kpt_near_mask_ratio))
    outside_dist = cv2.distanceTransform((clean == 0).astype(np.uint8), cv2.DIST_L2, 3)
    near_mask = 0
    h_img, w_img = clean.shape[:2]
    outside_values: Dict[str, float] = {}
    for n, p in pts.items():
        x = int(np.clip(round(float(p[0])), 0, w_img - 1))
        y = int(np.clip(round(float(p[1])), 0, h_img - 1))
        d = float(outside_dist[y, x])
        outside_values[n] = d
        if d <= near_limit:
            near_mask += 1

    def midpoint(a: str, b: str) -> Optional[np.ndarray]:
        if a in pts and b in pts:
            return 0.5 * (pts[a] + pts[b])
        return None

    wc = pts.get("waist_center")
    if wc is None:
        wc = midpoint("waist_img_left", "waist_img_right")
    left_hem = midpoint("img_left_hem_outer", "img_left_hem_inner")
    right_hem = midpoint("img_right_hem_inner", "img_right_hem_outer")
    hem_centers = [p for p in (left_hem, right_hem) if p is not None]
    hem_center = np.mean(hem_centers, axis=0) if hem_centers else None
    crotch = pts.get("crotch")

    checks: Dict[str, bool] = {
        "enough_keypoints": valid_count >= 6,
        "confidence": mean_conf >= float(args.min_mean_kpt_conf),
        "points_near_mask": near_mask >= min(valid_count, int(args.d25v2_pose_near_mask_min)),
    }
    metrics: Dict[str, float] = {
        "valid_keypoints": float(valid_count),
        "mean_conf": mean_conf,
        "near_mask_count": float(near_mask),
        "near_mask_limit_px": near_limit,
    }

    if wc is None or hem_center is None:
        checks.update({
            "body_axis": False,
            "waist_span": False,
            "crotch_order": False,
            "two_leg_centers": False,
            "two_hem_widths": False,
        })
    else:
        body_vec = np.asarray(hem_center - wc, np.float32)
        body_len = float(np.linalg.norm(body_vec))
        body_u = body_vec / max(body_len, 1e-6)
        lat_u = np.asarray([-body_u[1], body_u[0]], np.float32)
        checks["body_axis"] = body_len >= 45.0
        metrics["body_length_px"] = body_len

        wl = pts.get("waist_img_left")
        wr = pts.get("waist_img_right")
        waist_span = float(abs(np.dot((wr - wl), lat_u))) if wl is not None and wr is not None else 0.0
        checks["waist_span"] = waist_span >= max(28.0, 0.25 * body_len)
        metrics["waist_span_px"] = waist_span

        if crotch is not None and body_len > 1e-6:
            crotch_t = float(np.dot(crotch - wc, body_u))
            checks["crotch_order"] = 0.08 * body_len <= crotch_t <= 0.88 * body_len
            metrics["crotch_t_ratio"] = crotch_t / body_len
        else:
            checks["crotch_order"] = False
            metrics["crotch_t_ratio"] = -1.0

        if left_hem is not None and right_hem is not None and waist_span > 1e-6:
            ls = float(np.dot(left_hem - wc, lat_u))
            rs = float(np.dot(right_hem - wc, lat_u))
            separation = abs(ls - rs)
            lt = float(np.dot(left_hem - wc, body_u))
            rt = float(np.dot(right_hem - wc, body_u))
            checks["two_leg_centers"] = (
                separation >= max(24.0, 0.24 * waist_span)
                and min(lt, rt) >= 0.55 * body_len
            )
            metrics["leg_center_separation_ratio"] = separation / waist_span
            metrics["left_hem_t_ratio"] = lt / body_len
            metrics["right_hem_t_ratio"] = rt / body_len
        else:
            checks["two_leg_centers"] = False
            metrics["leg_center_separation_ratio"] = 0.0

        hem_widths: List[float] = []
        for a, b in (
            ("img_left_hem_outer", "img_left_hem_inner"),
            ("img_right_hem_inner", "img_right_hem_outer"),
        ):
            if a in pts and b in pts:
                hem_widths.append(float(abs(np.dot(pts[b] - pts[a], lat_u))))
        min_hem_width = min(hem_widths) if len(hem_widths) == 2 else 0.0
        checks["two_hem_widths"] = (
            len(hem_widths) == 2
            and min_hem_width >= max(8.0, 0.060 * max(waist_span, 1.0))
        )
        metrics["min_hem_width_px"] = min_hem_width
        metrics["min_hem_width_waist_ratio"] = min_hem_width / max(waist_span, 1.0)

    weights = {
        "enough_keypoints": 0.16,
        "confidence": 0.10,
        "points_near_mask": 0.15,
        "body_axis": 0.10,
        "waist_span": 0.12,
        "crotch_order": 0.15,
        "two_leg_centers": 0.14,
        "two_hem_widths": 0.08,
    }
    score = float(sum(weights[k] * float(bool(checks.get(k, False))) for k in weights))
    critical = bool(
        checks.get("enough_keypoints", False)
        and checks.get("points_near_mask", False)
        and checks.get("body_axis", False)
        and checks.get("crotch_order", False)
        and checks.get("two_leg_centers", False)
    )
    good = bool(critical and score >= float(args.d25v2_pose_score_min))
    return {
        "good": good,
        "score": score,
        "critical_good": critical,
        "checks": checks,
        "metrics": metrics,
        "valid_keypoints": valid_count,
        "mean_conf": mean_conf,
        "near_mask_count": near_mask,
        "outside_distance_px": outside_values,
    }


def d25_shape_report(obs: Any, args, H: Optional[np.ndarray] = None) -> Dict[str, Any]:
    """Pose-led, reference-free foldability shape report.

    Scanline crotch/two-leg/two-hem evidence is diagnostic support only.  It is
    intentionally not a mandatory gate, because segmentation morphology can
    close a narrow but valid crotch opening.
    """
    if obs is None or getattr(obs, "mask", None) is None:
        return {"shape_good": False, "shape_score": 0.0, "remaining": ["mask_missing"]}
    clean = _d25_clean_mask(obs.mask.mask_u8, args)
    contour = _d25_largest_contour(clean)
    if contour is None or cv2.contourArea(contour) < 1000.0:
        return {"shape_good": False, "shape_score": 0.0, "remaining": ["contour_invalid"]}

    candidates: List[Dict[str, Any]] = []
    for name, axis in _d25_pose_axis_candidates(obs, contour):
        for sign in (1.0, -1.0):
            cand = _d25_candidate_report(obs, H, clean, contour, name, axis, sign, args)
            if cand is not None:
                # Pose order is deliberately dominant for orientation selection.
                cand["v2_orientation_score"] = (
                    0.72 * float(cand.get("pose_order", {}).get("score", 0.0))
                    + 0.28 * float(cand.get("orientation_score", 0.0))
                )
                candidates.append(cand)
    if not candidates:
        return {"shape_good": False, "shape_score": 0.0, "remaining": ["orientation_unresolved"]}
    best = max(candidates, key=lambda c: float(c.get("v2_orientation_score", 0.0)))
    p = best["profile"]
    d = best["defects"]
    diff = best["differential"]

    pose_report = _d25v2_pose_structure_report(obs, clean, args)
    mask_area = float(cv2.countNonZero(clean))
    hull = cv2.convexHull(contour)
    hull_area = max(1.0, float(cv2.contourArea(hull)))
    solidity = mask_area / hull_area
    mask_good = bool(mask_area >= 1200.0 and solidity >= float(args.d25v2_min_mask_solidity))

    geometry_checks = {
        "waist_continuity": float(p["top_one_row_ratio"]) >= float(args.d25_waist_one_row_ratio_min),
        "crotch_support": bool(p["crotch_found"])
            and float(args.d25_crotch_y_min) <= float(p["crotch_y_ratio"]) <= float(args.d25_crotch_y_max),
        "two_leg_support": float(p["lower_two_leg_row_ratio"]) >= float(args.d25_lower_two_leg_row_ratio_min),
        "two_hem_support": float(p["hem_two_leg_row_ratio"]) >= float(args.d25_hem_two_leg_row_ratio_min),
        "leg_balance_support": float(p["leg_area_balance"]) >= float(args.d25_leg_area_balance_min),
        "mask_solidity": mask_good,
    }
    geometry_score = float(np.mean(list(geometry_checks.values())))
    geometry_rescue = bool(
        geometry_score >= float(args.d25v2_geometry_rescue_score_min)
        and sum(bool(v) for v in geometry_checks.values()) >= 4
    )

    unexpected_count = int(d.get("unexpected_count", 0))
    max_defect = float(d.get("max_unexpected_depth_ratio", 0.0))
    high_arc = float(diff.get("high_curvature_arc_ratio", 0.0))
    # Conservative veto: one ordinary hip/leg junction or segmentation notch
    # cannot fail a spread garment.  Only a deep dent with broad curvature, or
    # several substantial unexpected dents, is treated as a real outer fold.
    hard_outer_veto = bool(
        (
            unexpected_count >= 1
            and max_defect >= float(args.d25v2_outer_hard_depth_ratio)
            and high_arc >= float(args.d25v2_outer_hard_curvature_ratio)
        )
        or (
            unexpected_count >= int(args.d25v2_outer_multi_count)
            and max_defect >= float(args.d25v2_outer_multi_depth_ratio)
        )
    )

    pose_score = float(pose_report["score"])
    evidence_good = bool(pose_report["good"] or geometry_rescue)
    outer_score = 0.0 if hard_outer_veto else 1.0
    shape_score = float(
        0.62 * pose_score
        + 0.20 * geometry_score
        + 0.10 * float(mask_good)
        + 0.08 * outer_score
    )
    shape_good = bool(
        mask_good
        and evidence_good
        and shape_score >= float(args.d25v2_shape_score_min)
        and not hard_outer_veto
    )

    remaining: List[str] = []
    if not mask_good:
        remaining.append("mask_coherence")
    if not pose_report["good"] and not geometry_rescue:
        remaining.append("pose_and_geometry_weak")
    if shape_score < float(args.d25v2_shape_score_min):
        remaining.append("shape_score")
    if hard_outer_veto:
        remaining.append("deep_outer_fold")

    points = best["points"]
    points["pose_order_score"] = float(pose_report["score"])
    points["pose_order_checks"] = pose_report["checks"]
    points["pose_structure"] = pose_report
    return {
        "policy": "D25_V2_MEDIUM_POSE_LED_CONTOUR_VETO",
        "shape_good": shape_good,
        "shape_score": shape_score,
        "orientation_score": float(best.get("v2_orientation_score", 0.0)),
        "axis_source": str(best["axis_source"]),
        "axis_sign": float(best["axis_sign"]),
        "hard_checks": pose_report["checks"],
        "passed_check_count": int(sum(bool(v) for v in pose_report["checks"].values())),
        "required_check_count": 0,
        "hard_structure_good": bool(pose_report["good"]),
        "remaining": remaining,
        "pose_structure": pose_report,
        "geometry_support": {
            "score": geometry_score,
            "rescue": geometry_rescue,
            "checks": geometry_checks,
        },
        "topology": {
            "waist_one_row_ratio": float(p["top_one_row_ratio"]),
            "crotch_found": bool(p["crotch_found"]),
            "crotch_y_ratio": float(p["crotch_y_ratio"]),
            "crotch_gap_ratio": float(p["crotch_gap_ratio"]),
            "lower_two_leg_row_ratio": float(p["lower_two_leg_row_ratio"]),
            "hem_two_leg_row_ratio": float(p["hem_two_leg_row_ratio"]),
            "leg_area_balance": float(p["leg_area_balance"]),
            "min_hem_width_waist_ratio": float(p["min_hem_width_waist_ratio"]),
            "solidity": float(solidity),
        },
        "contour_differential": best["differential"],
        "convexity_defects": best["defects"],
        "macro_outer_fold": hard_outer_veto,
        "pose_geometry": points,
        "debug": best["debug"],
        "candidate_count": len(candidates),
        "candidate_scores": [
            {
                "axis_source": c["axis_source"],
                "axis_sign": c["axis_sign"],
                "score": float(c.get("v2_orientation_score", 0.0)),
                "pose_order": float(c.get("pose_order", {}).get("score", 0.0)),
            }
            for c in sorted(candidates, key=lambda c: float(c.get("v2_orientation_score", 0.0)), reverse=True)[:8]
        ],
        # Compatibility fields; none is an independent hard symmetry gate.
        "hem_gap_ratio": float(p["min_hem_width_waist_ratio"]),
        "leg_balance": float(p["leg_area_balance"]),
        "crotch_axis_offset_ratio": 0.0,
        "axis_to_waist_ratio": 1.0,
        "symmetry": {
            "lower_reflected_iou": 0.0,
            "lower_width_balance_median": float(p["leg_area_balance"]),
            "lower_width_balance_min": float(p["leg_area_balance"]),
            "symmetry_score": float(p["leg_area_balance"]),
            "good": True,
        },
    }


def _d25_fold_report_from_heat(obs: Any, heat: Any, H: np.ndarray, args,
                               shape: Dict[str, Any]) -> Dict[str, Any]:
    """Ignore surface wrinkles; block only a large, broad physical fold."""
    if heat is None:
        return {
            "good": False, "reason": "heatmap unavailable", "fold_count": 0,
            "t3_count": 0, "fine_allowed_count": 0, "remaining": ["heatmap"],
        }
    robust = robust_heat_threshold(heat, args)
    waist_ref = _d24v2_waist_reference(obs, heat, H, robust, args)
    candidates = list(getattr(heat, "d21v4_all_candidates", [])) or list(getattr(heat, "candidates", []))
    blockers: List[Dict[str, Any]] = []
    gathers: List[Dict[str, Any]] = []
    seams: List[Dict[str, Any]] = []
    fine: List[Dict[str, Any]] = []
    rejected: List[Dict[str, Any]] = []
    t1: List[Dict[str, Any]] = []
    t2: List[Dict[str, Any]] = []

    mask_area = max(1.0, float(obs.mask.area_px))
    for original in candidates:
        cand = dict(original)
        seam = seam_veto_report(cand, heat, obs, H, args)
        cand["d24_seam"] = seam
        if bool(seam.get("matched", False)):
            seams.append(cand)
            continue
        support = candidate_robust_support(cand, heat, float(robust["threshold"]))
        cand["d24_robust_support"] = support
        if support < float(args.d24_candidate_min_robust_support):
            rejected.append(cand)
            continue
        metrics = _d24v2_candidate_metrics(cand, heat, obs, waist_ref)
        cand["d24v2_metrics"] = metrics
        gather = _d24v5_waist_gather_report(cand, heat, obs, H, metrics, args)
        cand["d24v5_waist_gather"] = gather
        if bool(gather.get("matched", False)):
            gathers.append(cand)
            continue
        legacy_fold = _d24v6_fold_report(cand, metrics, heat, args)
        cand["d24v6_fold"] = legacy_fold
        length = float(metrics.get("length_mm", 0.0))
        width = float(cand.get("minor_length_mm", 0.0))
        area_ratio = float(metrics.get("area_ratio", float(cand.get("area_px", 0.0)) / mask_area))
        macro_geometry = bool(
            length >= float(args.d25v2_macro_fold_min_length_mm)
            and width >= float(args.d25v2_macro_fold_min_width_mm)
            and area_ratio >= float(args.d25v2_macro_fold_min_area_ratio)
        )
        extreme_geometry = bool(
            width >= float(args.d25v2_macro_fold_extreme_width_mm)
            and area_ratio >= float(args.d25v2_macro_fold_extreme_area_ratio)
        )
        is_blocker = bool(legacy_fold.get("is_fold", False) and (macro_geometry or extreme_geometry))
        cand["d25v2_macro_fold"] = {
            "is_fold": is_blocker,
            "macro_geometry": macro_geometry,
            "extreme_geometry": extreme_geometry,
            "length_mm": length,
            "width_mm": width,
            "area_ratio": area_ratio,
        }
        if is_blocker:
            blockers.append(cand)
        else:
            fine.append(cand)
        tier = int(metrics.get("tier", 0))
        if tier <= 1:
            t1.append(cand)
        elif tier == 2:
            t2.append(cand)

    outer_block = bool(shape.get("macro_outer_fold", False))
    effective_area = max(1.0, float(cv2.countNonZero(heat.inner_mask)))
    blocker_area = float(sum(float(c.get("area_px", 0.0)) for c in blockers))
    checks = {
        "heat_macro_fold_count": len(blockers) <= int(args.finish_max_t3_count),
        "deep_outer_fold": not outer_block,
    }
    good = bool(all(checks.values()))
    return {
        "policy": "D25_V2_TRUE_MACRO_FOLD_ONLY",
        "good": good,
        "checks": checks,
        "fold_count": len(blockers) + int(outer_block),
        "heat_fold_count": len(blockers),
        "outer_macro_fold_count": int(outer_block),
        "t3_count": len(blockers) + int(outer_block),
        "t2_count": len(t2),
        "t1_allowed_count": len(t1),
        "fine_allowed_count": len(fine),
        "waist_gather_ignored_count": len(gathers),
        "center_rise_ignored_px": int((getattr(heat, "center_rise_info", {}) or {}).get("ignored_px", 0)),
        "seam_ignored_count": len(seams),
        "robust_rejected_count": len(rejected),
        "waist_structure_ignored_count": int(getattr(heat, "d23_waist_structure_ignored_count", 0)),
        "fine_ignored_count": len(fine) + int(getattr(heat, "d21v4_fine_ignored_count", 0)),
        "actionable_ratio": blocker_area / effective_area,
        "max_blob_ratio": max([float(c.get("area_px", 0.0)) / mask_area for c in blockers] or [0.0]),
        "waist_reference": waist_ref,
        "robust": robust,
        "a119_summary": str(getattr(heat, "summary", "")),
        "remaining": [k for k, v in checks.items() if not v],
        "t1": t1,
        "t2": t2,
        "t3": blockers,
        "major_blockers": blockers,
        "fold_blockers": blockers,
        "waist_gather_ignored": gathers,
        "fine_allowed": fine,
        "seam_ignored": seams,
        "robust_rejected": rejected,
        "relative_allowed": [],
        "non_major_allowed": fine,
    }


def _d25_decide(obs: Any, heat: Any, H: np.ndarray, state: D24State, args,
                snapshot_mode: bool) -> Dict[str, Any]:
    state.evaluation_count += 1
    quality = observation_quality_report(obs, args)
    stability = {
        "good": True,
        "reason": "single frozen snapshot" if snapshot_mode else "pending",
        "history_count": 1 if snapshot_mode else len(state.history),
        "required": 1 if snapshot_mode else max(2, int(args.history_count)),
        "mask_area_rel": 0.0,
        "center_shift_mm": 0.0,
        "pose_axis_spread_deg": 0.0,
    }
    if not quality["good"]:
        state.ready_streak = 0
        return {
            "status": "REJUDGE", "reason": "OBSERVATION_INVALID",
            "evaluation": state.evaluation_count, "quality": quality,
            "stability": stability, "shape": {}, "wrinkle": {},
            "snapshot_mode": snapshot_mode,
        }
    if not snapshot_mode:
        state.history.append(observation_entry(obs))
        stability = temporal_stability_report(state.history, args)
        if not stability["good"]:
            state.ready_streak = 0
            return {
                "status": "REJUDGE", "reason": "OBSERVATION_UNSTABLE",
                "evaluation": state.evaluation_count, "quality": quality,
                "stability": stability, "shape": {}, "wrinkle": {},
                "snapshot_mode": False,
            }

    shape = d25_shape_report(obs, args, H)
    if not bool(shape.get("shape_good", False)):
        state.ready_streak = 0
        return {
            "status": "NOT_READY_SHAPE", "reason": "POSE_LED_SHAPE_FAILED",
            "evaluation": state.evaluation_count, "quality": quality,
            "stability": stability, "shape": shape, "wrinkle": {},
            "ready_streak": 0,
            "ready_required": 1 if snapshot_mode else int(args.ready_confirm_count),
            "snapshot_mode": snapshot_mode,
        }
    if heat is None:
        state.ready_streak = 0
        return {
            "status": "REJUDGE", "reason": "MACRO_FOLD_HEATMAP_UNAVAILABLE",
            "evaluation": state.evaluation_count, "quality": quality,
            "stability": stability, "shape": shape, "wrinkle": {},
            "snapshot_mode": snapshot_mode,
        }
    wrinkle = _d25_fold_report_from_heat(obs, heat, H, args, shape)
    if not wrinkle["good"]:
        state.ready_streak = 0
        return {
            "status": "NOT_READY_WRINKLE", "reason": "TRUE_MACRO_FOLD_REMAINS",
            "evaluation": state.evaluation_count, "quality": quality,
            "stability": stability, "shape": shape, "wrinkle": wrinkle,
            "ready_streak": 0,
            "ready_required": 1 if snapshot_mode else int(args.ready_confirm_count),
            "snapshot_mode": snapshot_mode,
        }
    if snapshot_mode:
        state.ready_streak = 1
        status = "READY_GOOD_ENOUGH"
        required = 1
    else:
        state.ready_streak += 1
        required = max(1, int(args.ready_confirm_count))
        status = "READY_GOOD_ENOUGH" if state.ready_streak >= required else "READY_PENDING"
    return {
        "status": status,
        "reason": "POSE_STRUCTURE_OK_AND_NO_TRUE_MACRO_FOLD",
        "evaluation": state.evaluation_count,
        "quality": quality,
        "stability": stability,
        "shape": shape,
        "wrinkle": wrinkle,
        "ready_streak": state.ready_streak,
        "ready_required": required,
        "snapshot_mode": snapshot_mode,
    }


def evaluate_d24(obs: Any, heat: Any, H: np.ndarray, state: D24State, args) -> Dict[str, Any]:
    return _d25_decide(obs, heat, H, state, args, snapshot_mode=False)


def evaluate_d24_snapshot(obs: Any, heat: Any, H: np.ndarray, state: D24State, args) -> Dict[str, Any]:
    return _d25_decide(obs, heat, H, state, args, snapshot_mode=True)


if __name__ == "__main__":
    raise SystemExit(main_event_snapshot())
