#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class EmptyBoardState:
    """Shared empty-board observation for upper/lower garment runtimes.

    corrected_bgr is the authoritative undistorted perception frame. raw_bgr is
    retained for raw-coordinate stale/change checks. homography and metadata are
    snapshots of the capture environment; consumers should validate compatibility
    instead of silently replacing their runtime calibration.
    """

    captured_at: float
    source: str
    baseline_id: str
    corrected_bgr: np.ndarray
    raw_bgr: np.ndarray
    homography: Optional[np.ndarray]
    camera_geometry: Dict[str, Any]
    camera_capture: Dict[str, Any]

    def clone(self) -> "EmptyBoardState":
        return EmptyBoardState(
            captured_at=float(self.captured_at),
            source=str(self.source),
            baseline_id=str(self.baseline_id),
            corrected_bgr=np.asarray(self.corrected_bgr, dtype=np.uint8).copy(),
            raw_bgr=np.asarray(self.raw_bgr, dtype=np.uint8).copy(),
            homography=None if self.homography is None else np.asarray(self.homography, dtype=np.float64).copy(),
            camera_geometry=copy.deepcopy(dict(self.camera_geometry or {})),
            camera_capture=copy.deepcopy(dict(self.camera_capture or {})),
        )

    @property
    def image_size(self) -> Tuple[int, int]:
        arr = np.asarray(self.corrected_bgr)
        if arr.ndim < 2:
            return 0, 0
        return int(arr.shape[1]), int(arr.shape[0])

    def metadata(self) -> Dict[str, Any]:
        return {
            "captured_at": float(self.captured_at),
            "source": str(self.source),
            "baseline_id": str(self.baseline_id),
            "image_size": list(self.image_size),
            "has_homography": bool(self.homography is not None),
            "camera_geometry": copy.deepcopy(dict(self.camera_geometry or {})),
            "camera_capture": copy.deepcopy(dict(self.camera_capture or {})),
        }
