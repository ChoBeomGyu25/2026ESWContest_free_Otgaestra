#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Reusable OpenCV camera-undistortion helper.

Designed for snapshot-based robot vision:
- calibration is loaded once;
- remap tables are built once per input resolution;
- ``correct()`` performs only ``cv2.remap``;
- output size always equals input size (no ROI crop/rescale);
- calibration identity metadata can be embedded in Homography caches.

Supported calibration files
---------------------------
NPZ keys (aliases accepted):
    camera_matrix / K / mtx
    dist_coeffs / distortion_coefficients / dist / D
    image_size / calibration_size / frame_size  (optional but strongly recommended)
    new_camera_matrix / new_K                    (optional)

JSON uses the same key names. ``image_size`` must be ``[width, height]``.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Optional, Sequence, Tuple

import cv2
import numpy as np


class CameraCalibrationError(RuntimeError):
    """Raised when a camera calibration is missing, malformed, or incompatible."""


def _first(mapping: Dict[str, Any], names: Iterable[str]) -> Any:
    for name in names:
        if name in mapping:
            return mapping[name]
    return None


def _as_matrix3(value: Any, field: str) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float64)
    if arr.shape != (3, 3):
        raise CameraCalibrationError(f"{field} must have shape (3,3); got {arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise CameraCalibrationError(f"{field} contains non-finite values")
    if abs(float(arr[2, 2])) < 1e-12:
        raise CameraCalibrationError(f"{field}[2,2] must be non-zero")
    return arr


def _as_dist(value: Any) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float64).reshape(-1, 1)
    if arr.size < 4:
        raise CameraCalibrationError(
            f"dist_coeffs must contain at least 4 values; got {arr.size}"
        )
    if not np.all(np.isfinite(arr)):
        raise CameraCalibrationError("dist_coeffs contains non-finite values")
    return arr


def _as_size(value: Any, field: str) -> Optional[Tuple[int, int]]:
    if value is None:
        return None
    arr = np.asarray(value).reshape(-1)
    if arr.size != 2:
        raise CameraCalibrationError(f"{field} must be [width,height]; got {value!r}")
    w, h = int(arr[0]), int(arr[1])
    if w <= 0 or h <= 0:
        raise CameraCalibrationError(f"{field} must contain positive integers; got {(w, h)}")
    return w, h


def _load_payload(path: str) -> Dict[str, Any]:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".npz":
        try:
            with np.load(path, allow_pickle=False) as data:
                return {key: data[key] for key in data.files}
        except Exception as exc:
            raise CameraCalibrationError(f"failed to read NPZ calibration: {exc}") from exc
    if ext == ".json":
        try:
            with open(path, "r", encoding="utf-8") as f:
                payload = json.load(f)
        except Exception as exc:
            raise CameraCalibrationError(f"failed to read JSON calibration: {exc}") from exc
        if not isinstance(payload, dict):
            raise CameraCalibrationError("JSON calibration root must be an object")
        return payload
    raise CameraCalibrationError(
        f"unsupported calibration extension {ext!r}; use .npz or .json"
    )


@dataclass(frozen=True)
class CalibrationInfo:
    path: str
    calibration_size: Optional[Tuple[int, int]]
    input_size: Tuple[int, int]
    output_size: Tuple[int, int]
    alpha: float
    calibration_id: str
    scaled_intrinsics: bool
    map_type: str

    def to_metadata(self) -> Dict[str, Any]:
        return {
            "undistort_enabled": True,
            "calibration_path": self.path,
            "calibration_size": list(self.calibration_size) if self.calibration_size else None,
            "input_size": list(self.input_size),
            "output_size": list(self.output_size),
            "alpha": float(self.alpha),
            "calibration_id": self.calibration_id,
            "scaled_intrinsics": bool(self.scaled_intrinsics),
            "map_type": self.map_type,
        }


class CameraUndistorter:
    """Load calibration and apply an efficient, same-size OpenCV remap."""

    def __init__(
        self,
        calibration_path: str,
        *,
        alpha: float = 1.0,
        strict_size: bool = True,
        map_type: int = cv2.CV_16SC2,
    ) -> None:
        self.calibration_path = os.path.abspath(os.path.expanduser(str(calibration_path)))
        self.alpha = float(alpha)
        self.strict_size = bool(strict_size)
        self.map_type = int(map_type)

        if not math.isfinite(self.alpha) or not (0.0 <= self.alpha <= 1.0):
            raise CameraCalibrationError(f"alpha must be within [0,1]; got {alpha!r}")
        if not os.path.isfile(self.calibration_path):
            raise CameraCalibrationError(
                f"calibration file not found: {self.calibration_path}"
            )

        payload = _load_payload(self.calibration_path)
        self.camera_matrix = _as_matrix3(
            _first(payload, ("camera_matrix", "K", "mtx")), "camera_matrix"
        )
        dist_raw = _first(
            payload,
            ("dist_coeffs", "distortion_coefficients", "dist", "D"),
        )
        if dist_raw is None:
            raise CameraCalibrationError(
                "calibration does not contain dist_coeffs/distortion_coefficients/dist/D"
            )
        self.dist_coeffs = _as_dist(dist_raw)
        self.calibration_size = _as_size(
            _first(payload, ("image_size", "calibration_size", "frame_size")),
            "image_size",
        )
        new_raw = _first(payload, ("new_camera_matrix", "new_K"))
        self.file_new_camera_matrix = (
            None if new_raw is None else _as_matrix3(new_raw, "new_camera_matrix")
        )

        self.input_size: Optional[Tuple[int, int]] = None
        self.output_size: Optional[Tuple[int, int]] = None
        self.effective_camera_matrix: Optional[np.ndarray] = None
        self.new_camera_matrix: Optional[np.ndarray] = None
        self.valid_roi: Optional[Tuple[int, int, int, int]] = None
        self.map1: Optional[np.ndarray] = None
        self.map2: Optional[np.ndarray] = None
        self.scaled_intrinsics = False
        self._calibration_id: Optional[str] = None

    @staticmethod
    def _same_aspect(a: Tuple[int, int], b: Tuple[int, int], tolerance: float = 1e-4) -> bool:
        return abs((float(a[0]) / float(a[1])) - (float(b[0]) / float(b[1]))) <= tolerance

    @staticmethod
    def _scale_intrinsics(
        matrix: np.ndarray,
        source_size: Tuple[int, int],
        target_size: Tuple[int, int],
    ) -> np.ndarray:
        sx = float(target_size[0]) / float(source_size[0])
        sy = float(target_size[1]) / float(source_size[1])
        out = np.asarray(matrix, dtype=np.float64).copy()
        out[0, 0] *= sx
        out[0, 1] *= sx
        out[0, 2] *= sx
        out[1, 0] *= sy
        out[1, 1] *= sy
        out[1, 2] *= sy
        return out

    def prepare(self, input_size: Sequence[int]) -> CalibrationInfo:
        size = _as_size(input_size, "input_size")
        assert size is not None

        effective_k = self.camera_matrix.copy()
        scaled = False
        if self.calibration_size is not None and size != self.calibration_size:
            if self.strict_size:
                raise CameraCalibrationError(
                    "camera frame size does not match calibration: "
                    f"camera={size}, calibration={self.calibration_size}. "
                    "Use a calibration made at the exact runtime resolution."
                )
            if not self._same_aspect(size, self.calibration_size):
                raise CameraCalibrationError(
                    "cannot scale intrinsics across a different aspect ratio: "
                    f"camera={size}, calibration={self.calibration_size}"
                )
            effective_k = self._scale_intrinsics(
                self.camera_matrix, self.calibration_size, size
            )
            scaled = True

        # Always derive the runtime matrix from the requested alpha.  A stored
        # new_camera_matrix may have been generated with an unknown alpha/crop,
        # so reusing it would make the runtime metadata misleading.
        new_k, roi = cv2.getOptimalNewCameraMatrix(
            effective_k,
            self.dist_coeffs,
            size,
            self.alpha,
            size,
            centerPrincipalPoint=False,
        )

        map1, map2 = cv2.initUndistortRectifyMap(
            effective_k,
            self.dist_coeffs,
            None,
            new_k,
            size,
            self.map_type,
        )
        if map1 is None or map2 is None:
            raise CameraCalibrationError("OpenCV failed to create undistortion maps")

        self.input_size = size
        self.output_size = size
        self.effective_camera_matrix = effective_k
        self.new_camera_matrix = np.asarray(new_k, dtype=np.float64)
        self.valid_roi = tuple(int(v) for v in roi)
        self.map1 = map1
        self.map2 = map2
        self.scaled_intrinsics = scaled
        self._calibration_id = self._build_calibration_id()
        return self.info()

    def _build_calibration_id(self) -> str:
        if self.input_size is None or self.new_camera_matrix is None:
            raise CameraCalibrationError("prepare() must be called first")
        h = hashlib.sha256()
        h.update(np.asarray(self.effective_camera_matrix, dtype=np.float64).tobytes())
        h.update(np.asarray(self.dist_coeffs, dtype=np.float64).tobytes())
        h.update(np.asarray(self.new_camera_matrix, dtype=np.float64).tobytes())
        h.update(repr((self.input_size, self.alpha, self.scaled_intrinsics)).encode("utf-8"))
        return h.hexdigest()[:20]

    def info(self) -> CalibrationInfo:
        if self.input_size is None or self.output_size is None or self._calibration_id is None:
            raise CameraCalibrationError("prepare() must be called before info()")
        map_name = "CV_16SC2" if self.map_type == cv2.CV_16SC2 else str(self.map_type)
        return CalibrationInfo(
            path=self.calibration_path,
            calibration_size=self.calibration_size,
            input_size=self.input_size,
            output_size=self.output_size,
            alpha=self.alpha,
            calibration_id=self._calibration_id,
            scaled_intrinsics=self.scaled_intrinsics,
            map_type=map_name,
        )

    def correct(self, frame: np.ndarray) -> np.ndarray:
        if self.map1 is None or self.map2 is None or self.input_size is None:
            raise CameraCalibrationError("prepare() must be called before correct()")
        if frame is None or not isinstance(frame, np.ndarray) or frame.ndim < 2:
            raise CameraCalibrationError("correct() received an invalid frame")
        actual = (int(frame.shape[1]), int(frame.shape[0]))
        if actual != self.input_size:
            raise CameraCalibrationError(
                f"frame size changed after map creation: frame={actual}, expected={self.input_size}"
            )
        return cv2.remap(
            frame,
            self.map1,
            self.map2,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
        )

    def raw_points_to_corrected(self, points: Any) -> np.ndarray:
        """Map raw/distorted pixel points to corrected pixel coordinates."""
        if self.new_camera_matrix is None or self.effective_camera_matrix is None:
            raise CameraCalibrationError("prepare() must be called first")
        pts = np.asarray(points, dtype=np.float64).reshape(-1, 1, 2)
        out = cv2.undistortPoints(
            pts,
            self.effective_camera_matrix,
            self.dist_coeffs,
            P=self.new_camera_matrix,
        )
        return out.reshape(-1, 2).astype(np.float32)

    def corrected_points_to_raw(self, points: Any) -> np.ndarray:
        """Map corrected pixels back to raw/distorted pixel coordinates."""
        if self.new_camera_matrix is None or self.effective_camera_matrix is None:
            raise CameraCalibrationError("prepare() must be called first")
        pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
        inv_new = np.linalg.inv(self.new_camera_matrix)
        homog = np.column_stack([pts, np.ones(len(pts), dtype=np.float64)])
        rays = (inv_new @ homog.T).T
        object_points = np.column_stack(
            [rays[:, 0] / rays[:, 2], rays[:, 1] / rays[:, 2], np.ones(len(pts))]
        ).reshape(-1, 1, 3)
        projected, _ = cv2.projectPoints(
            object_points,
            np.zeros((3, 1), dtype=np.float64),
            np.zeros((3, 1), dtype=np.float64),
            self.effective_camera_matrix,
            self.dist_coeffs,
        )
        return projected.reshape(-1, 2).astype(np.float32)

    def status_line(self) -> str:
        info = self.info()
        return (
            f"calibration={os.path.basename(info.path)} id={info.calibration_id} "
            f"size={info.input_size[0]}x{info.input_size[1]} alpha={info.alpha:.2f} "
            f"scaled_intrinsics={info.scaled_intrinsics} map={info.map_type}"
        )
