#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Jerber folding-machine UI v24 (training-matched software exposure auto-fit + one-shot RoArm standby gate).

Integrated maintenance functions:
  - On each folding-board page entry, plate motion stays locked until the operator explicitly
    moves both RoArms to the verified user-taught standby poses; repeated plate operations then stay enabled
  - 1280x720 YUYV full-frame camera input matching the colleague runtime
  - corrected QMessageBox layout/readability through QSS
  - Settings -> Calibration section
  - ArUco Marker calibration page
      * guide image before start
      * R-equivalent in-memory Homography reset
      * live camera marker detection after start
      * missing ID reporting for IDs 0,1,2,3
      * stable multi-frame Homography lock/save on corrected and raw frames
  - Dual RoArm affine calibration page
      * C-equivalent six-point sequence
      * ARM2: ID0 -> ID1 -> RED_EXTRA
      * ARM1: ID2 -> ID3 -> BLUE_EXTRA
      * active target highlighted in real time on the supplied board guide
      * T:105 feedback capture, undo, affine calculation, config/cache save

Safety:
  - Do not run the main robot runtime simultaneously with this UI. Camera,
    /dev/roarm_1, /dev/roarm_2 and /dev/ttyACM0 are exclusive resources.
  - RoArm actual transmission is enabled by default; use --no-roarm-send to disable it.
  - Arduino actual transmission is enabled by default; use --no-arduino-send to disable it.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from PySide6.QtCore import QObject, QPointF, QRectF, QSize, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QFont, QImage, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QSlider,
    QSizePolicy,
    QSpinBox,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

import folding_ui_base as base
from camera_v4l2_controls import (
    AUTO_GAIN_ALIASES,
    AUTO_WB_ALIASES,
    BACKLIGHT_COMPENSATION_ALIASES,
    EXPOSURE_DYNAMIC_FRAMERATE_ALIASES,
    EXPOSURE_MODE_ALIASES,
    EXPOSURE_VALUE_ALIASES,
    GAIN_ALIASES,
    POWER_LINE_ALIASES,
    WB_TEMPERATURE_ALIASES,
    V4L2CameraControls,
    V4L2ControlError,
)

try:
    from camera_undistort import CameraCalibrationError, CameraUndistorter
    CAMERA_UNDISTORT_IMPORT_ERROR = ""
except Exception as exc:
    CameraCalibrationError = RuntimeError
    CameraUndistorter = None  # type: ignore[assignment]
    CAMERA_UNDISTORT_IMPORT_ERROR = repr(exc)

try:
    import serial
except Exception:
    serial = None


BASE_DIR = Path(__file__).resolve().parent
GUIDE_IMAGE_PATH = BASE_DIR / "calibration_board_guide.png"
ROBOT_IMAGE_PATH = BASE_DIR / "robot_top_view.png"
COMPACT_7EP_QSS_PATH = BASE_DIR / "folding_machine_7ep.qss"
DEFAULT_PROJECT_DIR = Path(os.environ.get("JERBER_PROJECT_DIR", "/workspace/project_train/aruco_test/dual"))


DEFAULT_CONFIG: Dict[str, Any] = {
    "camera_index": 0,
    "camera_width": 1280,
    "camera_height": 720,
    "camera_backend": "v4l2",
    "aruco": {
        "dictionary": "DICT_4X4_50",
        "required_ids": [0, 1, 2, 3],
        "marker_board_mm": {
            "0": [0.0, 0.0],
            "1": [0.0, -496.0],
            "2": [492.0, 5.0],
            "3": [494.0, -490.0],
        },
    },
    "dual_roarm": {
        "split_board_x": 247.0,
        "arm1": {
            "name": "ARM1_ID2_ID3_SIDE",
            "side": "ID2_ID3 side",
            "port": "/dev/roarm_1",
            "baudrate": 115200,
            "calib_points": [
                {"label": "ID2", "board_xy": [492.0, 5.0]},
                {"label": "ID3", "board_xy": [494.0, -490.0]},
                {"label": "BLUE_EXTRA", "board_xy": [370.0, -249.0]},
            ],
            "calib_roarm_points": [],
            "calib_roarm_z": [],
            "board_to_roarm_affine_2x3": None,
        },
        "arm2": {
            "name": "ARM2_ID0_ID1_SIDE",
            "side": "ID0_ID1 side",
            "port": "/dev/roarm_2",
            "baudrate": 115200,
            "calib_points": [
                {"label": "ID0", "board_xy": [0.0, 0.0]},
                {"label": "ID1", "board_xy": [0.0, -496.0]},
                {"label": "RED_EXTRA", "board_xy": [115.0, -249.0]},
            ],
            "calib_roarm_points": [],
            "calib_roarm_z": [],
            "board_to_roarm_affine_2x3": None,
        },
    },
}


def deep_update(base_dict: Dict[str, Any], incoming: Dict[str, Any]) -> Dict[str, Any]:
    out = copy.deepcopy(base_dict)

    def rec(dst: Dict[str, Any], src: Dict[str, Any]) -> None:
        for key, value in src.items():
            if isinstance(value, dict) and isinstance(dst.get(key), dict):
                rec(dst[key], value)
            else:
                dst[key] = copy.deepcopy(value)

    rec(out, incoming)
    return out


def load_or_create_config(path: Path) -> Dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_text(
            json.dumps(DEFAULT_CONFIG, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return copy.deepcopy(DEFAULT_CONFIG)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"설정 파일 루트가 JSON object가 아님: {path}")
    return deep_update(DEFAULT_CONFIG, payload)


def save_config(path: Path, cfg: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def get_dictionary(name: str):
    if not hasattr(cv2, "aruco"):
        raise RuntimeError("현재 OpenCV에 cv2.aruco가 없습니다. opencv-contrib-python이 필요합니다.")
    if not hasattr(cv2.aruco, name):
        raise ValueError(f"지원하지 않는 ArUco dictionary: {name}")
    return cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, name))


def make_detector(dictionary):
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


def compute_homography(
    corners,
    ids,
    marker_board_mm: Dict[str, Sequence[float]],
    required_ids: Sequence[int],
) -> Tuple[Optional[np.ndarray], Dict[int, np.ndarray]]:
    if ids is None:
        return None, {}
    centers: Dict[int, np.ndarray] = {}
    for corner, marker_id in zip(corners, ids.flatten()):
        points = np.asarray(corner, dtype=np.float32).reshape(4, 2)
        centers[int(marker_id)] = points.mean(axis=0)
    if not all(int(marker_id) in centers for marker_id in required_ids):
        return None, centers
    image_points = np.asarray(
        [centers[int(marker_id)] for marker_id in required_ids],
        dtype=np.float32,
    )
    board_points = np.asarray(
        [marker_board_mm[str(int(marker_id))] for marker_id in required_ids],
        dtype=np.float32,
    )
    H, _ = cv2.findHomography(image_points, board_points)
    return H, centers


def compute_homography_from_centers(
    centers: Dict[int, np.ndarray],
    marker_board_mm: Dict[str, Sequence[float]],
    required_ids: Sequence[int],
) -> Optional[np.ndarray]:
    if not all(int(marker_id) in centers for marker_id in required_ids):
        return None
    image_points = np.asarray(
        [centers[int(marker_id)] for marker_id in required_ids], dtype=np.float32
    )
    board_points = np.asarray(
        [marker_board_mm[str(int(marker_id))] for marker_id in required_ids],
        dtype=np.float32,
    )
    H, _ = cv2.findHomography(image_points, board_points)
    return None if H is None else np.asarray(H, dtype=np.float32)


def save_homography_bundle(
    path: Path,
    H: np.ndarray,
    raw_H: np.ndarray,
    camera_geometry: Dict[str, Any],
) -> None:
    payload = {
        "schema_version": 2,
        "H": np.asarray(H, dtype=float).tolist(),
        "raw_H": np.asarray(raw_H, dtype=float).tolist(),
        "camera_geometry": copy.deepcopy(camera_geometry),
        "created_at_unix": time.time(),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    tmp.replace(path)

def compute_affine(
    board_points: np.ndarray,
    roarm_points: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    A = np.c_[board_points, np.ones(len(board_points))]
    M, _, _, _ = np.linalg.lstsq(A, roarm_points, rcond=None)
    M_2x3 = M.T
    predicted = A @ M
    error = np.linalg.norm(predicted - roarm_points, axis=1)
    return M_2x3, predicted, error


def board_to_roarm_xy(cfg: Dict[str, Any], arm_key: str, bx: float, by: float) -> Tuple[float, float]:
    arm_cfg = cfg.get("dual_roarm", {}).get(str(arm_key), {})
    raw = arm_cfg.get("board_to_roarm_affine_2x3")
    if raw is None:
        raise RuntimeError(f"{arm_key.upper()} affine 보정값이 없습니다. 설정 → 보정 → 로봇팔 보정을 먼저 수행하세요.")
    matrix = np.asarray(raw, dtype=float).reshape(2, 3)
    out = matrix @ np.asarray([float(bx), float(by), 1.0], dtype=float)
    offset = arm_cfg.get("roarm_xy_offset", [0.0, 0.0])
    ox = float(offset[0]) if isinstance(offset, (list, tuple)) and len(offset) > 0 else 0.0
    oy = float(offset[1]) if isinstance(offset, (list, tuple)) and len(offset) > 1 else 0.0
    return float(out[0] + ox), float(out[1] + oy)


def roarm_to_board_xy(cfg: Dict[str, Any], arm_key: str, rx: float, ry: float) -> Tuple[float, float]:
    arm_cfg = cfg.get("dual_roarm", {}).get(str(arm_key), {})
    raw = arm_cfg.get("board_to_roarm_affine_2x3")
    if raw is None:
        raise RuntimeError(f"{arm_key.upper()} affine 보정값이 없습니다. 설정 → 보정 → 로봇팔 보정을 먼저 수행하세요.")
    matrix = np.asarray(raw, dtype=float).reshape(2, 3)
    offset = arm_cfg.get("roarm_xy_offset", [0.0, 0.0])
    ox = float(offset[0]) if isinstance(offset, (list, tuple)) and len(offset) > 0 else 0.0
    oy = float(offset[1]) if isinstance(offset, (list, tuple)) and len(offset) > 1 else 0.0
    linear = matrix[:, :2]
    translation = matrix[:, 2]
    if abs(float(np.linalg.det(linear))) < 1e-9:
        raise RuntimeError(f"{arm_key.upper()} affine 역변환을 계산할 수 없습니다.")
    board = np.linalg.inv(linear) @ (np.asarray([float(rx) - ox, float(ry) - oy]) - translation)
    return float(board[0]), float(board[1])


def board_limits(cfg: Dict[str, Any]) -> Tuple[float, float, float, float]:
    marker = cfg.get("aruco", {}).get("marker_board_mm", {})
    points = np.asarray(list(marker.values()), dtype=float).reshape(-1, 2)
    return (float(points[:, 0].min()), float(points[:, 0].max()),
            float(points[:, 1].min()), float(points[:, 1].max()))


class UiRoArmSerial:
    """Small RoArm JSON serial wrapper with diagnostic T:105 retries."""

    def __init__(self, port: str, baud: int = 115200, label: str = "arm") -> None:
        if serial is None:
            raise RuntimeError("pyserial이 없습니다. python3 -m pip install pyserial")
        self.port = str(port)
        self.label = str(label)
        self.io_lock = threading.RLock()
        self.last_rx_lines: List[str] = []
        self.last_feedback_error = ""
        self.ser = serial.Serial(self.port, int(baud), timeout=0.15, write_timeout=1.0)
        try:
            self.ser.setDTR(False)
            self.ser.setRTS(False)
        except Exception:
            pass
        # Give the USB serial bridge a short settling window.  This does not send
        # any robot command and prevents the first feedback request from racing the port.
        time.sleep(0.18)
        self.flush()

    def flush(self) -> None:
        try:
            self.ser.reset_input_buffer()
            self.ser.reset_output_buffer()
        except Exception:
            pass

    def _remember_rx(self, text: str) -> None:
        self.last_rx_lines.append(str(text))
        if len(self.last_rx_lines) > 24:
            del self.last_rx_lines[:-24]

    def send(self, command: Dict[str, Any], delay: float = 0.15) -> None:
        line = json.dumps(command, separators=(",", ":"))
        with self.io_lock:
            self.ser.write(line.encode("utf-8") + b"\n")
            self.ser.flush()
            time.sleep(max(0.0, float(delay)))

    def read_json_until(self, predicate, timeout: float = 2.5):
        start = time.monotonic()
        while time.monotonic() - start < float(timeout):
            raw = self.ser.readline()
            if not raw:
                continue
            text = raw.decode("utf-8", errors="replace").strip()
            if not text:
                continue
            self._remember_rx(text)
            try:
                payload = json.loads(text)
            except Exception:
                continue
            try:
                if predicate(payload):
                    return payload
            except Exception:
                continue
        return None

    def torque_off(self) -> None:
        self.send({"T": 210, "cmd": 0}, delay=0.25)

    def torque_on(self) -> None:
        self.send({"T": 210, "cmd": 1}, delay=0.25)

    def feedback(self, timeout: float = 2.5, retries: int = 3):
        """Request T:1051, skipping servo-status/noise lines and retrying."""
        attempts = max(1, int(retries))
        self.last_feedback_error = ""
        with self.io_lock:
            for attempt in range(1, attempts + 1):
                try:
                    # Drop stale status lines before each new T:105 transaction.
                    self.ser.reset_input_buffer()
                except Exception:
                    pass
                self.send({"T": 105}, delay=0.06)
                payload = self.read_json_until(
                    lambda obj: isinstance(obj, dict) and int(obj.get("T", -1)) == 1051,
                    timeout=max(0.2, float(timeout)),
                )
                if payload is not None:
                    return payload
                if attempt < attempts:
                    time.sleep(0.12)
        tail = self.last_rx_lines[-8:]
        self.last_feedback_error = (
            f"{self.label}({self.port}) T:1051 응답 없음 · "
            f"시도={attempts} · 마지막 RX={tail if tail else '없음'}"
        )
        return None

    def move_goal(
        self,
        x: float,
        y: float,
        z: float,
        t: float,
        speed: float,
        move_command: int = 104,
    ) -> None:
        self.send(
            {
                "T": int(move_command),
                "x": float(x),
                "y": float(y),
                "z": float(z),
                "t": float(t),
                "spd": float(speed),
            },
            delay=0.18,
        )

    def set_gripper(self, angle_rad: float) -> None:
        self.send(
            {"T": 106, "cmd": float(angle_rad), "spd": 0, "acc": 0},
            delay=0.18,
        )

    def close(self) -> None:
        try:
            self.ser.close()
        except Exception:
            pass


class CalibrationSignals(QObject):
    robot_connected = Signal(bool, object, str)
    robot_capture_finished = Signal(bool, object, str)


class CalibrationGuideWidget(QWidget):
    """Show the supplied folding-board guide and calibration targets."""

    # Coordinates in the supplied 1024x1024 image.
    TARGET_POINTS: Dict[str, QPointF] = {
        # User-specified on-screen orientation.
        "ID0": QPointF(978.0, 218.0),   # upper-right
        "ID1": QPointF(978.0, 1007.0),  # lower-right
        "ID2": QPointF(268.0, 218.0),   # upper-left
        "ID3": QPointF(268.0, 1007.0),  # lower-left
        "BLUE_EXTRA": QPointF(541.0, 596.0),
        "RED_EXTRA": QPointF(704.0, 596.0),
    }

    def __init__(
        self,
        image_path: Path,
        mode: str,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("calibrationGuideWidget")
        self.setMinimumSize(500, 500)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self._pixmap = QPixmap(str(image_path))
        self._image_size = self._pixmap.size()
        self._draw_rect = QRectF()
        self._mode = str(mode)
        self._current_target: Optional[str] = None
        self._active_arm = ""

    def set_current_target(self, target: Optional[str], active_arm: str = "") -> None:
        self._current_target = str(target) if target else None
        self._active_arm = str(active_arm)
        self.update()

    def _fit_rect(self) -> QRectF:
        if self._pixmap.isNull() or self._image_size.width() <= 0:
            return QRectF(self.rect())
        available = QRectF(self.rect()).adjusted(12.0, 12.0, -12.0, -52.0)
        scale = min(
            available.width() / float(self._image_size.width()),
            available.height() / float(self._image_size.height()),
        )
        width = self._image_size.width() * scale
        height = self._image_size.height() * scale
        return QRectF(
            available.center().x() - width / 2.0,
            available.center().y() - height / 2.0,
            width,
            height,
        )

    def _map_point(self, point: QPointF) -> QPointF:
        sx = self._draw_rect.width() / max(1.0, float(self._image_size.width()))
        sy = self._draw_rect.height() / max(1.0, float(self._image_size.height()))
        return QPointF(
            self._draw_rect.left() + point.x() * sx,
            self._draw_rect.top() + point.y() * sy,
        )

    def paintEvent(self, event) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.fillRect(self.rect(), QColor("#07101C"))
        self._draw_rect = self._fit_rect()

        if self._pixmap.isNull():
            painter.setPen(QColor("#FF8A8A"))
            painter.drawText(
                self.rect(),
                Qt.AlignmentFlag.AlignCenter,
                "calibration_board_guide.png 파일이 없습니다",
            )
            return

        painter.drawPixmap(self._draw_rect, self._pixmap, QRectF(self._pixmap.rect()))

        targets = ["ID0", "ID1", "ID2", "ID3"]
        if self._mode == "robot":
            targets += ["BLUE_EXTRA", "RED_EXTRA"]

        for name in targets:
            center = self._map_point(self.TARGET_POINTS[name])
            selected = name == self._current_target
            if name == "BLUE_EXTRA":
                base_color = QColor("#4169FF")
            elif name == "RED_EXTRA":
                base_color = QColor("#FF3B4D")
            else:
                base_color = QColor("#39D98A")

            radius = 28.0 if selected else 20.0
            if selected:
                painter.setPen(QPen(QColor("#FFD166"), 7))
                painter.setBrush(QColor(255, 209, 102, 50))
                painter.drawEllipse(center, radius + 12.0, radius + 12.0)

            painter.setPen(QPen(QColor("#FFFFFF"), 3))
            painter.setBrush(base_color)
            painter.drawEllipse(center, radius, radius)

            label_rect = QRectF(center.x() - 72.0, center.y() - 15.0, 144.0, 30.0)
            painter.setPen(QColor("#FFFFFF"))
            painter.setFont(QFont("Noto Sans CJK KR", 11, QFont.Weight.Bold))
            painter.drawText(label_rect, Qt.AlignmentFlag.AlignCenter, name)

        painter.setPen(QColor("#D5E0EF"))
        painter.setFont(QFont("Noto Sans CJK KR", 11, QFont.Weight.DemiBold))
        if self._mode == "aruco":
            note = "화면 기준: 오른쪽 위 ID0 · 오른쪽 아래 ID1 · 왼쪽 위 ID2 · 왼쪽 아래 ID3"
        elif self._current_target:
            note = f"현재 위치: {self._active_arm.upper()} → {self._current_target}"
        else:
            note = "ARM2: ID0 → ID1 → 빨간 스티커 / ARM1: ID2 → ID3 → 파란 스티커"
        painter.drawText(
            QRectF(10.0, self.height() - 42.0, self.width() - 20.0, 30.0),
            Qt.AlignmentFlag.AlignCenter,
            note,
        )


class FitVideoWidget(QWidget):
    """Full-frame camera view that never crops the source image.

    The old QLabel + setPixmap() path scaled the pixmap to the QLabel's outer
    size. A stylesheet border reduces QLabel.contentsRect(), so a pixmap sized
    to the outer rectangle can be clipped at the edges. A QLabel also derives
    its sizeHint from the current pixmap, which can disturb layouts while pages
    are being resized.

    This widget keeps an owned QImage and paints it directly into the actual
    contents rectangle using a strict FIT policy. The complete camera frame is
    always visible; unused space is letterboxed.
    """

    def __init__(
        self,
        placeholder: str = "카메라 연결 중...",
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._image = QImage()
        self._placeholder = str(placeholder)
        self._alignment = Qt.AlignmentFlag.AlignCenter
        self._last_frame_size: Tuple[int, int] = (0, 0)
        self._processing_label = "원본 영상"
        self.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Expanding,
        )
        self.setMinimumSize(320, 180)
        self.setAttribute(Qt.WidgetAttribute.WA_OpaquePaintEvent, True)

    def sizeHint(self) -> QSize:  # noqa: N802
        return QSize(640, 360)

    def minimumSizeHint(self) -> QSize:  # noqa: N802
        return QSize(320, 180)

    def setAlignment(self, alignment) -> None:  # QLabel compatibility
        self._alignment = alignment
        self.update()

    def set_processing_label(self, label: str) -> None:
        self._processing_label = str(label)
        self.update()

    def setText(self, text: str) -> None:  # QLabel compatibility
        self._placeholder = str(text)
        self._image = QImage()
        self._last_frame_size = (0, 0)
        self.update()

    def text(self) -> str:
        return self._placeholder

    def clear(self) -> None:
        self._image = QImage()
        self._last_frame_size = (0, 0)
        self.update()

    def set_bgr_frame(self, frame: np.ndarray) -> None:
        if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
            self.setText("카메라 프레임이 비어 있습니다.")
            return
        if frame.ndim == 2:
            rgb = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
        elif frame.ndim == 3 and frame.shape[2] == 4:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGRA2RGBA)
        else:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb = np.ascontiguousarray(rgb)
        height, width = rgb.shape[:2]
        bytes_per_line = int(rgb.strides[0])
        fmt = (
            QImage.Format.Format_RGBA8888
            if rgb.ndim == 3 and rgb.shape[2] == 4
            else QImage.Format.Format_RGB888
        )
        self._image = QImage(
            rgb.data,
            int(width),
            int(height),
            bytes_per_line,
            fmt,
        ).copy()
        self._last_frame_size = (int(width), int(height))
        self.update()

    def _fit_rect(self, source_width: int, source_height: int) -> QRectF:
        # Leave a few pixels inside the styled border. This uses the real
        # contents rectangle, not the QWidget outer size.
        available = QRectF(self.contentsRect()).adjusted(5.0, 5.0, -5.0, -5.0)
        if (
            source_width <= 0
            or source_height <= 0
            or available.width() <= 0.0
            or available.height() <= 0.0
        ):
            return QRectF()
        scale = min(
            available.width() / float(source_width),
            available.height() / float(source_height),
        )
        draw_width = float(source_width) * scale
        draw_height = float(source_height) * scale
        return QRectF(
            available.center().x() - draw_width / 2.0,
            available.center().y() - draw_height / 2.0,
            draw_width,
            draw_height,
        )

    def paintEvent(self, event) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
        painter.fillRect(self.rect(), QColor("#050B14"))

        if self._image.isNull():
            painter.setPen(QColor("#8093AE"))
            painter.setFont(QFont("Noto Sans CJK KR", 15, QFont.Weight.DemiBold))
            painter.drawText(
                self.contentsRect().adjusted(20, 20, -20, -20),
                Qt.AlignmentFlag.AlignCenter | Qt.TextFlag.TextWordWrap,
                self._placeholder,
            )
            return

        draw_rect = self._fit_rect(self._image.width(), self._image.height())
        if draw_rect.isEmpty():
            return

        # Strict FIT: draw the whole source image into the destination rectangle.
        painter.drawImage(draw_rect, self._image, QRectF(self._image.rect()))

        # Thin outline makes the exact full-frame boundary visible.
        painter.setPen(QPen(QColor("#6FA5FF"), 2.0))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawRect(draw_rect)

        frame_width, frame_height = self._last_frame_size
        badge_text = (
            f"전체 프레임 · FIT · {frame_width}×{frame_height} · "
            f"{self._processing_label}"
        )
        painter.setFont(QFont("Noto Sans CJK KR", 10, QFont.Weight.Bold))
        metrics = painter.fontMetrics()
        badge_width = metrics.horizontalAdvance(badge_text) + 22
        badge_height = metrics.height() + 12
        badge = QRectF(
            draw_rect.left() + 10.0,
            draw_rect.top() + 10.0,
            float(badge_width),
            float(badge_height),
        )
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(5, 11, 20, 190))
        painter.drawRoundedRect(badge, 7.0, 7.0)
        painter.setPen(QColor("#FFFFFF"))
        painter.drawText(badge, Qt.AlignmentFlag.AlignCenter, badge_text)



class RobotTestSignals(QObject):
    action_finished = Signal(str, bool, str)
    feedback_ready = Signal(bool, object, str)
    log = Signal(str)


class RobotTopViewWidget(QWidget):
    """User-view top drawing with the physical RoArm mapping.

    The supplied robot_top_view.png is drawn from the operator/user side, not
    from the overhead camera image.  In this image the physical mapping is:
      image-left  = Robot 1 = ARM1 = /dev/roarm_1
      image-right = Robot 2 = ARM2 = /dev/roarm_2

    The live camera image is horizontally opposite (camera-left=ARM2), but that
    camera convention must not be applied to this maintenance illustration.
    """

    armClicked = Signal(str)
    ARM_RECTS = {
        "arm1": QRectF(105.0, 390.0, 375.0, 190.0),  # user-image left
        "arm2": QRectF(545.0, 390.0, 375.0, 190.0),  # user-image right
    }

    def __init__(self, image_path: Path, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setObjectName("robotTopViewWidget")
        self.setMinimumSize(560, 500)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMouseTracking(True)
        self._pixmap = QPixmap(str(image_path))
        self._image_size = self._pixmap.size()
        self._draw_rect = QRectF()
        self._selected_arm = "arm1"
        self._hover_arm = ""

    def set_selected_arm(self, arm_key: str) -> None:
        if str(arm_key) in self.ARM_RECTS:
            self._selected_arm = str(arm_key)
            self.update()

    def _fit_rect(self) -> QRectF:
        if self._pixmap.isNull() or self._image_size.width() <= 0:
            return QRectF(self.rect())
        available = QRectF(self.rect()).adjusted(10.0, 10.0, -10.0, -52.0)
        scale = min(available.width() / self._image_size.width(),
                    available.height() / self._image_size.height())
        width = self._image_size.width() * scale
        height = self._image_size.height() * scale
        return QRectF(available.center().x() - width / 2.0,
                      available.center().y() - height / 2.0, width, height)

    def _map_rect(self, source: QRectF) -> QRectF:
        if self._image_size.width() <= 0:
            return QRectF()
        sx = self._draw_rect.width() / self._image_size.width()
        sy = self._draw_rect.height() / self._image_size.height()
        return QRectF(self._draw_rect.left() + source.left() * sx,
                      self._draw_rect.top() + source.top() * sy,
                      source.width() * sx, source.height() * sy)

    def _arm_at(self, point: QPointF) -> str:
        for arm_key, rect in self.ARM_RECTS.items():
            if self._map_rect(rect).contains(point):
                return arm_key
        return ""

    def paintEvent(self, event) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.fillRect(self.rect(), QColor("#07101C"))
        self._draw_rect = self._fit_rect()
        if self._pixmap.isNull():
            painter.setPen(QColor("#FF8A8A"))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "robot_top_view.png 파일이 없습니다")
            return
        painter.drawPixmap(self._draw_rect, self._pixmap, QRectF(self._pixmap.rect()))

        labels = {"arm1": "로봇팔 1 · ARM1", "arm2": "로봇팔 2 · ARM2"}
        for arm_key, source_rect in self.ARM_RECTS.items():
            rect = self._map_rect(source_rect)
            selected = arm_key == self._selected_arm
            hovered = arm_key == self._hover_arm
            if selected:
                fill = QColor(57, 123, 255, 65)
                border = QColor("#77A9FF")
                width = 5
            elif hovered:
                fill = QColor(53, 208, 127, 45)
                border = QColor("#55E0A0")
                width = 3
            else:
                fill = QColor(4, 10, 18, 100)
                border = QColor("#354863")
                width = 2
            painter.fillRect(rect, fill)
            painter.setPen(QPen(border, width))
            painter.drawRoundedRect(rect, 18.0, 18.0)
            badge = QRectF(rect.center().x() - 92.0, rect.top() + 8.0, 184.0, 38.0)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QColor(8, 20, 36, 220))
            painter.drawRoundedRect(badge, 11.0, 11.0)
            painter.setPen(QColor("#FFFFFF" if selected else "#AEBFD3"))
            painter.setFont(QFont("Noto Sans CJK KR", 12, QFont.Weight.Bold))
            painter.drawText(badge, Qt.AlignmentFlag.AlignCenter, labels[arm_key])

        painter.setPen(QColor("#9FB0C6"))
        painter.setFont(QFont("Noto Sans CJK KR", 10, QFont.Weight.DemiBold))
        painter.drawText(QRectF(10.0, self.height() - 40.0, self.width() - 20.0, 28.0),
                         Qt.AlignmentFlag.AlignCenter,
                         "그림은 사용자 시점임 · 왼쪽=로봇팔 1/ARM1 · 오른쪽=로봇팔 2/ARM2")

    def mouseMoveEvent(self, event) -> None:  # noqa: N802
        arm = self._arm_at(event.position())
        if arm != self._hover_arm:
            self._hover_arm = arm
            self.update()
        self.setCursor(Qt.CursorShape.PointingHandCursor if arm else Qt.CursorShape.ArrowCursor)

    def leaveEvent(self, event) -> None:  # noqa: N802
        self._hover_arm = ""
        self.update()

    def mousePressEvent(self, event) -> None:  # noqa: N802
        if event.button() == Qt.MouseButton.LeftButton:
            arm = self._arm_at(event.position())
            if arm:
                self.set_selected_arm(arm)
                self.armClicked.emit(arm)
                event.accept()
                return
        super().mousePressEvent(event)


class MaintenanceUiController(base.FoldingBoardUiController):
    def __init__(self, args: argparse.Namespace, app: QApplication) -> None:
        self.calibration_signals = CalibrationSignals()
        self.robot_test_signals = RobotTestSignals()
        self.robot_test_selected_arm = "arm1"
        self.robot_test_serial: Optional[UiRoArmSerial] = None
        self.robot_test_serial_arm = ""
        self._robot_test_busy = False
        self._robot_test_torque_enabled: Dict[str, bool] = {"arm1": True, "arm2": True}
        self.machine_config_path = Path(args.robot_config).expanduser().resolve()
        self.hfile_path = Path(args.hfile).expanduser().resolve()
        self.calib_cache_path = Path(args.calib_cache).expanduser().resolve()
        self.machine_cfg = load_or_create_config(self.machine_config_path)

        self.camera_control_config_path = Path(args.camera_control_config).expanduser().resolve()
        self.v4l2_controls = V4L2CameraControls(str(args.camera))
        self.camera_control_names: Dict[str, Optional[str]] = {
            "exposure_mode": None,
            "exposure": None,
            "gain": None,
            "auto_gain": None,
            "power_line": None,
            "auto_wb": None,
            "wb_temperature": None,
            "backlight_compensation": None,
            "exposure_dynamic_framerate": None,
        }
        self._camera_ui_syncing = False
        self._camera_autofit_running = False
        # v23: OV2710 exposes exposure_time_absolute as inactive while hardware AE is on,
        # so the old "hardware auto -> read value -> manual" path could only recover the
        # previous manual value (for example 151).  Auto-fit now meters live pixels while
        # staying in MANUAL mode and searches exposure_time_absolute directly.
        self._camera_autofit_step = 0
        self._camera_autofit_low_exp: Optional[int] = None
        self._camera_autofit_high_exp: Optional[int] = None
        self._camera_autofit_last_p20: Optional[float] = None
        self._camera_autofit_last_p30: Optional[float] = None
        self._camera_autofit_last_p98: Optional[float] = None
        self._camera_autofit_last_p995: Optional[float] = None
        self._camera_autofit_last_mean: Optional[float] = None
        self._camera_autofit_last_clip: Optional[float] = None
        self._camera_autofit_meter_H: Optional[np.ndarray] = None
        self._camera_autofit_meter_source = ""
        self._camera_autofit_best_exp: Optional[int] = None
        self._camera_autofit_best_score = float("inf")

        # v24 training-match meter.  These bands were measured from the supplied
        # 2026-09-17 top-reference board_512 images (17 observations) captured
        # with the production camera profile (manual exposure=151, gain=0,
        # AWB off, WB=4600).  P20/P30 are dominated by the black folding board
        # and are therefore much less sensitive to garment colour than the old
        # full-frame P98-only meter.  Highlights are still guarded separately.
        self._camera_autofit_p20_low = 33.0
        self._camera_autofit_p20_high = 45.0
        self._camera_autofit_p30_low = 38.0
        self._camera_autofit_p30_high = 65.0
        self._camera_autofit_p98_high = 244.0
        self._camera_autofit_hard_clip_p995 = 252.0
        self._camera_autofit_clip_max = 0.010
        self._camera_autofit_max_steps = 12
        self._camera_autofit_settle_ms = 650

        aruco_cfg = self.machine_cfg["aruco"]
        self.required_marker_ids = [int(value) for value in aruco_cfg["required_ids"]]
        self.aruco_detector = make_detector(get_dictionary(str(aruco_cfg["dictionary"])))

        self.latest_raw_camera_frame: Optional[np.ndarray] = None
        self.latest_camera_frame: Optional[np.ndarray] = None
        self.camera_undistorter: Optional[CameraUndistorter] = None
        self.camera_geometry: Dict[str, Any] = {
            "camera_model": "ELP_OV2710",
            "undistort_enabled": False,
            "calibration_id": "NOT_PREPARED",
            "output_size": [int(args.width), int(args.height)],
            "alpha": float(args.camera_undistort_alpha),
        }
        self.camera_undistort_error = ""
        self.aruco_sample_target = max(3, int(args.aruco_sample_count))
        self.aruco_stable_max_shift_px = max(0.1, float(args.aruco_stable_max_shift_px))
        self._last_aruco_stability_log_at = 0.0
        self._aruco_corrected_samples: List[Dict[int, np.ndarray]] = []
        self._aruco_raw_samples: List[Dict[int, np.ndarray]] = []
        self.aruco_started = False
        self.aruco_scanning = False
        self.aruco_locked = False
        self.aruco_latest_H: Optional[np.ndarray] = None
        self.aruco_latest_raw_H: Optional[np.ndarray] = None
        self._last_missing_ids: Optional[Tuple[int, ...]] = None

        self.roarms: Dict[str, UiRoArmSerial] = {}
        self.robot_calibration_active = False
        self.robot_calib_sequence: List[Tuple[str, Dict[str, Any]]] = []
        self.robot_calib_index = 0
        self.temp_roarm_xy: Dict[str, List[List[float]]] = {"arm1": [], "arm2": []}
        self.temp_roarm_z: Dict[str, List[float]] = {"arm1": [], "arm2": []}
        self._robot_capture_busy = False
        # Folding-board page safety gate.  It becomes True only after the operator
        # explicitly moves both RoArms to the verified standby poses.  The gate is
        # kept for repeated plate operations within the same page session and is
        # reset whenever the operator leaves/re-enters the page or loses Arduino readiness.
        self.plate_roarms_ready = False

        # The Waveshare 7EP-CAPLCD is 1280x800.  Several maintenance pages
        # contain many touch controls, so the desktop-sized minimum heights can
        # over-constrain Qt's layout and make widgets overlap/crop.  Resolve a
        # display profile before building the pages, then compact only on the
        # 7-inch profile.  Larger monitors keep the original roomy layout.
        self.display_profile = self._resolve_display_profile(args, app)

        super().__init__(args, app)

        if self.display_profile == "7ep":
            self._apply_7ep_compact_layout()

        title_label = self.window.findChild(QLabel, "titleLabel")
        subtitle_label = self.window.findChild(QLabel, "subtitleLabel")
        if title_label is not None:
            title_label.setText("Jerber")
        if subtitle_label is not None:
            subtitle_label.setText("의류 정리 시스템")
        self.window.setWindowTitle("Jerber · 의류 정리 시스템")

        self.calibration_signals.robot_connected.connect(self._on_robot_connected)
        self.calibration_signals.robot_capture_finished.connect(self._on_robot_capture_finished)
        self.robot_test_signals.action_finished.connect(self._on_robot_test_action_finished)
        self.robot_test_signals.feedback_ready.connect(self._on_robot_test_feedback_ready)
        self.robot_test_signals.log.connect(self._append_robot_test_log_ui)

    @staticmethod
    def _resolve_display_profile(args: argparse.Namespace, app: QApplication) -> str:
        requested = str(getattr(args, "display_profile", "auto")).strip().lower()
        if requested in {"7ep", "desktop"}:
            return requested

        screen = app.primaryScreen()
        if screen is None:
            return "desktop"
        size = screen.size()
        width = int(size.width())
        height = int(size.height())

        # Exact target is 1280x800.  A small tolerance also covers desktop
        # panels/resolution reporting around the same class of 7-inch screens.
        if width <= 1366 and height <= 850:
            return "7ep"
        return "desktop"

    @staticmethod
    def _compact_layout(layout, margins=(12, 9, 12, 9), spacing=7) -> None:
        if layout is None:
            return
        layout.setContentsMargins(*margins)
        layout.setSpacing(int(spacing))

    def _apply_7ep_compact_layout(self) -> None:
        """Fit all maintenance pages inside a 1280x800 7EP-CAPLCD.

        The key fix is not merely smaller fonts: the old code had cumulative
        minimum heights larger than the right-hand control card itself.  Qt then
        had no valid geometry and controls visually collided.  This profile
        lowers those minimums while preserving >=36~44 px touch targets and
        leaves desktop sizing unchanged.
        """
        if COMPACT_7EP_QSS_PATH.exists():
            self.window.setStyleSheet(COMPACT_7EP_QSS_PATH.read_text(encoding="utf-8"))

        # Page roots and headers.
        for page_name in (
            "settings_page",
            "plate_test_page",
            "camera_test_page",
            "robot_test_page",
            "aruco_page",
            "robot_calib_page",
        ):
            page = getattr(self, page_name, None)
            if page is not None:
                self._compact_layout(page.layout(), (14, 10, 14, 10), 10)

        for back_name in (
            "settings_back_button",
            "plate_back_button",
            "camera_test_back_button",
            "robot_test_back_button",
            "aruco_back_button",
            "robot_calib_back_button",
        ):
            button = getattr(self, back_name, None)
            if button is not None:
                button.setMinimumSize(92, 44)

        for badge_name in (
            "arduino_status_badge",
            "camera_control_badge",
            "robot_test_header_badge",
            "aruco_header_badge",
            "robot_header_badge",
        ):
            badge = getattr(self, badge_name, None)
            if badge is not None:
                badge.setMinimumSize(170, 42)

        # Compact all card layouts by object name, including calibration cards.
        for frame in self.window.findChildren(QFrame):
            if frame.objectName() in {
                "settingsHeader", "settingsCard", "calibrationCard",
                "plateBoardCard", "plateControlCard",
                "cameraTestDisplayCard", "cameraControlCard",
                "robotVisualCard", "robotControlCard",
                "calibrationDisplayCard", "calibrationControlCard",
            }:
                self._compact_layout(frame.layout(), (12, 9, 12, 9), 7)

        # Folding-board page.
        self.plate_board_widget.setMinimumSize(360, 360)
        self.selected_plate_label.setMinimumHeight(40)
        self.plate_roarm_standby_button.setMinimumHeight(58)
        self.auto_test_button.setMinimumHeight(66)
        self.stop_button.setMinimumHeight(46)
        self.home_button.setMinimumHeight(46)
        self.reconnect_button.setMinimumHeight(42)
        self.rearm_button.setMinimumHeight(42)
        self.log_view.setMinimumHeight(76)
        plate_warning = self.window.findChild(QLabel, "plateWarningLabel")
        if plate_warning is not None:
            plate_warning.setMinimumHeight(46)

        # Camera page: keep the full 16:9 frame visible but allow the card to
        # shrink enough for 800 px total screen height.
        self.camera_test_view.setMinimumSize(420, 236)
        self.camera_control_status.setMinimumHeight(44)
        for button in (
            self.camera_continuous_auto_button,
            self.camera_autofit_button,
            self.camera_manual_button,
        ):
            button.setMinimumHeight(38)
        for spin in (self.camera_exposure_spin, self.camera_gain_spin, self.camera_wb_spin):
            spin.setMinimumWidth(78)
        for button in (
            self.camera_refresh_button,
            self.camera_save_button,
            self.camera_defaults_button,
        ):
            button.setMinimumHeight(38)
        camera_warning = self.window.findChild(QLabel, "cameraControlWarning")
        if camera_warning is not None:
            camera_warning.setMinimumHeight(48)

        # Robot test page was the most constrained page in v19.  The old sum of
        # minimum heights exceeded the available control-card height, which is
        # why jog/auxiliary buttons overlapped in the 1280x800 screenshot.
        self.robot_top_view.setMinimumSize(380, 300)
        for button in (self.robot_select_arm1_button, self.robot_select_arm2_button):
            button.setMinimumHeight(48)
        self.robot_test_status.setMinimumHeight(42)
        self.robot_auto_test_button.setMinimumHeight(58)
        for button in (
            self.robot_move_forward, self.robot_move_back,
            self.robot_move_left, self.robot_move_right,
            self.robot_move_up, self.robot_move_down,
        ):
            button.setMinimumHeight(38)
        for button in (
            self.robot_wrist_ccw, self.robot_wrist_cw,
            self.robot_gripper_open, self.robot_gripper_close,
            self.robot_torque_on, self.robot_torque_off,
            self.robot_feedback_button, self.robot_standby_button,
        ):
            button.setMinimumHeight(36)
        self.robot_test_log.setMinimumHeight(62)
        robot_warning = self.window.findChild(QLabel, "robotTestWarning")
        if robot_warning is not None:
            robot_warning.setMinimumHeight(44)

        # Calibration pages also use the same 3:2 split and had desktop-sized
        # minimums, so make them safe on the same panel even though they were not
        # the pages shown in the user's screenshots.
        self.aruco_guide_widget.setMinimumSize(360, 300)
        self.robot_guide_widget.setMinimumSize(360, 300)
        self.aruco_camera_label.setMinimumSize(400, 225)
        self.aruco_status_label.setMinimumHeight(48)
        self.aruco_detection_label.setMinimumHeight(58)
        self.aruco_start_button.setMinimumHeight(58)
        for calibration_warning in self.window.findChildren(QLabel, "calibrationWarningLabel"):
            calibration_warning.setMinimumHeight(54)

        self.robot_step_label.setMinimumHeight(66)
        self.robot_progress.setMinimumHeight(28)
        self.robot_start_button.setMinimumHeight(56)
        self.robot_capture_button.setMinimumHeight(46)
        self.robot_undo_button.setMinimumHeight(40)
        self.robot_stop_button.setMinimumHeight(40)

        # Main page was already designed for 1280x800, but the replacement FIT
        # camera widget inherited a 300 px minimum.  Lower it slightly so system
        # font/DPI differences cannot push the bottom action bar off-screen.
        if hasattr(self, "camera_label"):
            self.camera_label.setMinimumHeight(260)

    def _build_pages(self) -> None:
        super()._build_pages()
        self._append_calibration_section_to_settings()
        self.camera_test_page = self._create_camera_test_page()
        self.robot_test_page = self._create_robot_test_page()
        self.aruco_page = self._create_aruco_page()
        self.robot_calib_page = self._create_robot_calibration_page()
        self.page_stack.addWidget(self.camera_test_page)
        self.page_stack.addWidget(self.robot_test_page)
        self.page_stack.addWidget(self.aruco_page)
        self.page_stack.addWidget(self.robot_calib_page)
        self.robot_test_entry_button.clicked.connect(self._enter_robot_test_page)
        self.robot_test_back_button.clicked.connect(self._leave_robot_test_page)

    def _append_calibration_section_to_settings(self) -> None:
        root = self.settings_page.layout()
        if root is None:
            raise RuntimeError("settingsPage layout 없음")

        calibration_card = QFrame()
        calibration_card.setObjectName("calibrationCard")
        card_layout = QVBoxLayout(calibration_card)
        card_layout.setContentsMargins(24, 20, 24, 22)
        card_layout.setSpacing(12)

        section = QLabel("보정")
        section.setObjectName("sectionTitle")
        note = QLabel("카메라 좌표계와 로봇팔 좌표계를 실제 폴딩 보드에 맞춥니다.")
        note.setObjectName("sectionNote")
        card_layout.addWidget(section)
        card_layout.addWidget(note)

        row = QHBoxLayout()
        row.setSpacing(14)
        self.aruco_entry_button = QPushButton(
            "AruCo Marker 보정\n4개 마커 감지 및 Homography Lock"
        )
        self.aruco_entry_button.setObjectName("calibrationEntryButton")
        self.robot_calib_entry_button = QPushButton(
            "로봇팔 보정\n마커·스티커 6점으로 양팔 affine 계산"
        )
        self.robot_calib_entry_button.setObjectName("calibrationEntryButton")
        self.aruco_entry_button.setMinimumHeight(104)
        self.robot_calib_entry_button.setMinimumHeight(104)
        row.addWidget(self.aruco_entry_button)
        row.addWidget(self.robot_calib_entry_button)
        card_layout.addLayout(row)

        insert_index = max(0, root.count() - 1)
        root.insertWidget(insert_index, calibration_card)

    def _create_camera_test_page(self) -> QWidget:
        page = QWidget()
        page.setObjectName("cameraTestPage")
        root = QVBoxLayout(page)
        root.setContentsMargins(24, 20, 24, 20)
        root.setSpacing(16)

        header, self.camera_test_back_button = self._create_page_header(
            "카메라 점검",
            "실시간 영상을 보며 노출·게인·플리커 설정을 조절합니다.",
            "cameraTestBackButton",
        )
        self.camera_control_badge = QLabel("제어값 확인 전")
        self.camera_control_badge.setObjectName("cameraControlBadge")
        self.camera_control_badge.setMinimumSize(210, 52)
        self.camera_control_badge.setAlignment(Qt.AlignmentFlag.AlignCenter)
        header.layout().addWidget(self.camera_control_badge)
        root.addWidget(header)

        content = QHBoxLayout()
        content.setSpacing(18)

        display_card = QFrame()
        display_card.setObjectName("cameraTestDisplayCard")
        display_layout = QVBoxLayout(display_card)
        display_layout.setContentsMargins(16, 14, 16, 14)
        display_layout.setSpacing(8)
        display_title = QLabel("실시간 카메라 영상")
        display_title.setObjectName("cardTitle")
        display_layout.addWidget(display_title)
        self.camera_test_view = FitVideoWidget("카메라 연결 중...")
        self.camera_test_view.setObjectName("cameraTestView")
        self.camera_test_view.setMinimumSize(500, 300)
        display_layout.addWidget(self.camera_test_view, 1)
        self.camera_test_note = QLabel(
            "권장 순서: 플리커 60Hz → 대표 옷을 보드에 놓고 자동 맞춤 후 고정 → 필요하면 수동 미세 조절"
        )
        self.camera_test_note.setObjectName("cameraTestNote")
        self.camera_test_note.setWordWrap(True)
        display_layout.addWidget(self.camera_test_note)
        content.addWidget(display_card, 3)

        control_card = QFrame()
        control_card.setObjectName("cameraControlCard")
        control_layout = QVBoxLayout(control_card)
        control_layout.setContentsMargins(20, 18, 20, 18)
        control_layout.setSpacing(11)

        control_title = QLabel("노출 및 플리커 설정")
        control_title.setObjectName("cardTitle")
        control_layout.addWidget(control_title)

        self.camera_control_status = QLabel(
            "카메라가 제공하는 V4L2 제어값을 확인하고 있습니다."
        )
        self.camera_control_status.setObjectName("cameraControlStatus")
        self.camera_control_status.setWordWrap(True)
        self.camera_control_status.setMinimumHeight(58)
        control_layout.addWidget(self.camera_control_status)

        mode_row = QHBoxLayout()
        mode_row.setSpacing(8)
        self.camera_continuous_auto_button = QPushButton("연속 자동")
        self.camera_autofit_button = QPushButton("밝기 자동 맞춤 후 고정")
        self.camera_manual_button = QPushButton("현재값 수동 고정")
        for button in (
            self.camera_continuous_auto_button,
            self.camera_autofit_button,
            self.camera_manual_button,
        ):
            button.setObjectName("cameraModeButton")
            button.setMinimumHeight(50)
            mode_row.addWidget(button)
        control_layout.addLayout(mode_row)

        grid = QGridLayout()
        grid.setHorizontalSpacing(10)
        grid.setVerticalSpacing(10)
        grid.setColumnStretch(1, 1)

        exposure_label = QLabel("셔터/노출")
        exposure_label.setObjectName("cameraControlLabel")
        self.camera_exposure_slider = QSlider(Qt.Orientation.Horizontal)
        self.camera_exposure_slider.setObjectName("cameraControlSlider")
        self.camera_exposure_spin = QSpinBox()
        self.camera_exposure_spin.setObjectName("cameraControlSpin")
        self.camera_exposure_spin.setMinimumWidth(100)
        self.camera_exposure_hint = QLabel("장치 미지원")
        self.camera_exposure_hint.setObjectName("cameraControlHint")
        grid.addWidget(exposure_label, 0, 0)
        grid.addWidget(self.camera_exposure_slider, 0, 1)
        grid.addWidget(self.camera_exposure_spin, 0, 2)
        grid.addWidget(self.camera_exposure_hint, 1, 1, 1, 2)

        gain_label = QLabel("게인(ISO 역할)")
        gain_label.setObjectName("cameraControlLabel")
        self.camera_gain_slider = QSlider(Qt.Orientation.Horizontal)
        self.camera_gain_slider.setObjectName("cameraControlSlider")
        self.camera_gain_spin = QSpinBox()
        self.camera_gain_spin.setObjectName("cameraControlSpin")
        self.camera_gain_spin.setMinimumWidth(100)
        self.camera_gain_hint = QLabel("장치 미지원")
        self.camera_gain_hint.setObjectName("cameraControlHint")
        grid.addWidget(gain_label, 2, 0)
        grid.addWidget(self.camera_gain_slider, 2, 1)
        grid.addWidget(self.camera_gain_spin, 2, 2)
        grid.addWidget(self.camera_gain_hint, 3, 1, 1, 2)

        flicker_label = QLabel("플리커 방지")
        flicker_label.setObjectName("cameraControlLabel")
        self.camera_flicker_combo = QComboBox()
        self.camera_flicker_combo.setObjectName("cameraControlCombo")
        grid.addWidget(flicker_label, 4, 0)
        grid.addWidget(self.camera_flicker_combo, 4, 1, 1, 2)

        self.camera_awb_checkbox = QCheckBox("화이트밸런스 자동")
        self.camera_awb_checkbox.setObjectName("cameraAwbCheck")
        grid.addWidget(self.camera_awb_checkbox, 5, 0, 1, 3)

        wb_label = QLabel("색온도")
        wb_label.setObjectName("cameraControlLabel")
        self.camera_wb_slider = QSlider(Qt.Orientation.Horizontal)
        self.camera_wb_slider.setObjectName("cameraControlSlider")
        self.camera_wb_spin = QSpinBox()
        self.camera_wb_spin.setObjectName("cameraControlSpin")
        self.camera_wb_spin.setMinimumWidth(100)
        grid.addWidget(wb_label, 6, 0)
        grid.addWidget(self.camera_wb_slider, 6, 1)
        grid.addWidget(self.camera_wb_spin, 6, 2)
        control_layout.addLayout(grid)

        action_row = QHBoxLayout()
        action_row.setSpacing(8)
        self.camera_refresh_button = QPushButton("현재값 새로고침")
        self.camera_save_button = QPushButton("현재 설정 저장")
        self.camera_defaults_button = QPushButton("카메라 기본값")
        for button in (
            self.camera_refresh_button,
            self.camera_save_button,
            self.camera_defaults_button,
        ):
            button.setObjectName("cameraAuxButton")
            button.setMinimumHeight(48)
            action_row.addWidget(button)
        control_layout.addLayout(action_row)

        self.camera_control_path_label = QLabel(
            f"저장 파일\n{self.camera_control_config_path}"
        )
        self.camera_control_path_label.setObjectName("cameraControlPath")
        self.camera_control_path_label.setWordWrap(True)
        control_layout.addWidget(self.camera_control_path_label)

        warning = QLabel(
            "OV2710의 하드웨어 자동노출 값은 V4L2에서 읽어 고정할 수 없으므로, "
            "'밝기 자동 맞춤 후 고정'은 Action Model 학습 당시 board_512 밝기에 맞춰 수동 exposure를 직접 탐색합니다. "
            "가장 정확한 결과를 위해 실제 운용할 대표 옷을 보드에 놓고 실행하세요."
        )
        warning.setObjectName("cameraControlWarning")
        warning.setWordWrap(True)
        warning.setMinimumHeight(72)
        control_layout.addWidget(warning)
        control_layout.addStretch(1)

        content.addWidget(control_card, 2)
        root.addLayout(content, 1)

        self._set_camera_control_widgets_enabled(False)
        return page

    def _set_camera_control_widgets_enabled(self, enabled: bool) -> None:
        for widget in (
            self.camera_continuous_auto_button,
            self.camera_autofit_button,
            self.camera_manual_button,
            self.camera_exposure_slider,
            self.camera_exposure_spin,
            self.camera_gain_slider,
            self.camera_gain_spin,
            self.camera_flicker_combo,
            self.camera_awb_checkbox,
            self.camera_wb_slider,
            self.camera_wb_spin,
            self.camera_refresh_button,
            self.camera_save_button,
            self.camera_defaults_button,
        ):
            widget.setEnabled(enabled)

    def _relevant_camera_control_names(self) -> List[str]:
        return [
            name
            for name in self.camera_control_names.values()
            if name is not None and name in self.v4l2_controls.controls
        ]

    def _configure_numeric_camera_control(
        self,
        name: Optional[str],
        slider: QSlider,
        spin: QSpinBox,
        hint: Optional[QLabel] = None,
    ) -> None:
        if name is None:
            slider.setEnabled(False)
            spin.setEnabled(False)
            if hint is not None:
                hint.setText("이 카메라에서 지원하지 않음")
            return
        info = self.v4l2_controls.controls[name]
        minimum = int(info.minimum if info.minimum is not None else 0)
        maximum = int(info.maximum if info.maximum is not None else max(minimum + 1, 255))
        step = max(1, int(info.step if info.step is not None else 1))
        value = int(info.value if info.value is not None else info.default or minimum)
        self._camera_ui_syncing = True
        try:
            slider.setRange(minimum, maximum)
            slider.setSingleStep(step)
            slider.setPageStep(max(step, (maximum - minimum) // 20 or 1))
            slider.setValue(value)
            spin.setRange(minimum, maximum)
            spin.setSingleStep(step)
            spin.setValue(value)
            editable = "read-only" not in info.flags.lower()
            slider.setEnabled(editable)
            spin.setEnabled(editable)
            if hint is not None and name != self.camera_control_names.get("exposure"):
                hint.setText(f"범위 {minimum}~{maximum} · 단계 {step}")
        finally:
            self._camera_ui_syncing = False

    def _discover_camera_controls(self) -> None:
        if not self.v4l2_controls.available:
            self.camera_control_badge.setText("V4L2 제어 불가")
            self.camera_control_status.setText(
                "v4l2-ctl이 없거나 카메라 장치 경로를 찾을 수 없습니다. "
                "영상 표시는 계속되지만 노출 설정은 사용할 수 없습니다."
            )
            self._set_camera_control_widgets_enabled(False)
            return
        try:
            self.v4l2_controls.discover()
        except V4L2ControlError as exc:
            self.camera_control_badge.setText("제어값 조회 실패")
            self.camera_control_status.setText(f"카메라 제어값 조회 실패: {exc}")
            self._set_camera_control_widgets_enabled(False)
            self._append_log(f"카메라 V4L2 제어값 조회 실패: {exc}")
            return

        self.camera_control_names = {
            "exposure_mode": self.v4l2_controls.first_existing(EXPOSURE_MODE_ALIASES),
            "exposure": self.v4l2_controls.first_existing(EXPOSURE_VALUE_ALIASES),
            "gain": self.v4l2_controls.first_existing(GAIN_ALIASES),
            "auto_gain": self.v4l2_controls.first_existing(AUTO_GAIN_ALIASES),
            "power_line": self.v4l2_controls.first_existing(POWER_LINE_ALIASES),
            "auto_wb": self.v4l2_controls.first_existing(AUTO_WB_ALIASES),
            "wb_temperature": self.v4l2_controls.first_existing(WB_TEMPERATURE_ALIASES),
            "backlight_compensation": self.v4l2_controls.first_existing(BACKLIGHT_COMPENSATION_ALIASES),
            "exposure_dynamic_framerate": self.v4l2_controls.first_existing(EXPOSURE_DYNAMIC_FRAMERATE_ALIASES),
        }

        self._set_camera_control_widgets_enabled(True)
        self._configure_numeric_camera_control(
            self.camera_control_names["exposure"],
            self.camera_exposure_slider,
            self.camera_exposure_spin,
            self.camera_exposure_hint,
        )
        self._configure_numeric_camera_control(
            self.camera_control_names["gain"],
            self.camera_gain_slider,
            self.camera_gain_spin,
            self.camera_gain_hint,
        )
        self._configure_numeric_camera_control(
            self.camera_control_names["wb_temperature"],
            self.camera_wb_slider,
            self.camera_wb_spin,
        )

        power_name = self.camera_control_names["power_line"]
        self._camera_ui_syncing = True
        try:
            self.camera_flicker_combo.clear()
            if power_name is not None:
                info = self.v4l2_controls.controls[power_name]
                for value, label in sorted(info.menu.items()):
                    self.camera_flicker_combo.addItem(label, value)
                self.camera_flicker_combo.setEnabled(bool(info.menu) and info.writable)
            else:
                self.camera_flicker_combo.addItem("장치 미지원", None)
                self.camera_flicker_combo.setEnabled(False)
        finally:
            self._camera_ui_syncing = False

        self.camera_control_badge.setText("실시간 조절 가능")
        detected = [key for key, value in self.camera_control_names.items() if value]
        self.camera_control_status.setText(
            "지원 제어값: " + (", ".join(detected) if detected else "표준 제어값 없음")
        )
        self._append_log(
            "카메라 제어값 감지: "
            + ", ".join(f"{key}={value}" for key, value in self.camera_control_names.items() if value)
        )
        self._refresh_camera_control_values(show_error=False)

    def _camera_set_control(self, name: Optional[str], value: int, *, quiet: bool = False) -> bool:
        if name is None:
            return False
        try:
            self.v4l2_controls.set(name, int(value))
            return True
        except V4L2ControlError as exc:
            if not quiet:
                self.camera_control_status.setText(f"{name} 적용 실패: {exc}")
                self._append_log(f"카메라 제어 실패 {name}={value}: {exc}")
            return False

    def _menu_mode_value(self, automatic: bool) -> Optional[int]:
        name = self.camera_control_names.get("exposure_mode")
        if name is None:
            return None
        info = self.v4l2_controls.controls.get(name)
        if info is None:
            return None
        if "menu" not in info.kind:
            return 1 if automatic else 0
        if automatic:
            return self.v4l2_controls.find_any_menu_value(
                name,
                (("aperture", "priority"), ("auto", "mode"), ("automatic",), ("auto",)),
            )
        return self.v4l2_controls.find_any_menu_value(name, (("manual",),))

    def _set_exposure_automatic(self, automatic: bool, *, quiet: bool = False) -> None:
        mode_name = self.camera_control_names.get("exposure_mode")
        mode_value = self._menu_mode_value(automatic)
        if mode_name is not None and mode_value is not None:
            self._camera_set_control(mode_name, mode_value, quiet=quiet)
        auto_gain_name = self.camera_control_names.get("auto_gain")
        if auto_gain_name is not None:
            self._camera_set_control(auto_gain_name, 1 if automatic else 0, quiet=quiet)

    def _set_awb_automatic(self, automatic: bool, *, quiet: bool = False) -> None:
        name = self.camera_control_names.get("auto_wb")
        if name is not None:
            self._camera_set_control(name, 1 if automatic else 0, quiet=quiet)

    def _refresh_camera_control_values(self, *, show_error: bool = True) -> None:
        if not self.v4l2_controls.controls:
            return
        try:
            self.v4l2_controls.refresh()
        except V4L2ControlError as exc:
            if show_error:
                self.camera_control_status.setText(f"현재값 조회 실패: {exc}")
            return

        self._camera_ui_syncing = True
        try:
            exposure_name = self.camera_control_names.get("exposure")
            gain_name = self.camera_control_names.get("gain")
            wb_name = self.camera_control_names.get("wb_temperature")
            for name, slider, spin in (
                (exposure_name, self.camera_exposure_slider, self.camera_exposure_spin),
                (gain_name, self.camera_gain_slider, self.camera_gain_spin),
                (wb_name, self.camera_wb_slider, self.camera_wb_spin),
            ):
                if name is None or name not in self.v4l2_controls.controls:
                    continue
                info = self.v4l2_controls.controls[name]
                if info.value is not None:
                    slider.setValue(int(info.value))
                    spin.setValue(int(info.value))

            power_name = self.camera_control_names.get("power_line")
            if power_name is not None and power_name in self.v4l2_controls.controls:
                value = self.v4l2_controls.controls[power_name].value
                index = self.camera_flicker_combo.findData(value)
                if index >= 0:
                    self.camera_flicker_combo.setCurrentIndex(index)

            awb_name = self.camera_control_names.get("auto_wb")
            if awb_name is not None and awb_name in self.v4l2_controls.controls:
                awb_value = self.v4l2_controls.controls[awb_name].value
                self.camera_awb_checkbox.setChecked(bool(awb_value))
                wb_enabled = not bool(awb_value)
                self.camera_wb_slider.setEnabled(wb_enabled)
                self.camera_wb_spin.setEnabled(wb_enabled)
        finally:
            self._camera_ui_syncing = False

        self._update_camera_control_hints()
        if show_error:
            self.camera_control_status.setText("현재 카메라 제어값을 새로고침했습니다.")

    def _update_camera_control_hints(self) -> None:
        exposure_name = self.camera_control_names.get("exposure")
        value = self.camera_exposure_spin.value()
        if exposure_name in {"exposure_absolute", "exposure_time_absolute"} and value > 0:
            seconds = value / 10000.0
            if seconds < 1.0:
                denominator = max(1, round(1.0 / seconds))
                self.camera_exposure_hint.setText(
                    f"{value} 장치 단위 · 약 1/{denominator}초"
                )
            else:
                self.camera_exposure_hint.setText(f"{value} 장치 단위 · 약 {seconds:.2f}초")
        elif exposure_name is not None:
            info = self.v4l2_controls.controls.get(exposure_name)
            if info is not None:
                self.camera_exposure_hint.setText(
                    f"장치 단위 · 범위 {info.minimum}~{info.maximum}"
                )

    def _apply_saved_camera_controls(self) -> None:
        path = self.camera_control_config_path
        if not path.exists() or not self.v4l2_controls.controls:
            return
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            values = payload.get("controls", {}) if isinstance(payload, dict) else {}
            if not isinstance(values, dict):
                raise ValueError("controls 항목이 object가 아님")
            ordered_names = [
                self.camera_control_names.get("exposure_mode"),
                self.camera_control_names.get("auto_gain"),
                self.camera_control_names.get("power_line"),
                self.camera_control_names.get("auto_wb"),
                self.camera_control_names.get("backlight_compensation"),
                self.camera_control_names.get("exposure_dynamic_framerate"),
                self.camera_control_names.get("exposure"),
                self.camera_control_names.get("gain"),
                self.camera_control_names.get("wb_temperature"),
            ]
            applied = []
            for name in ordered_names:
                if name is None or name not in values or name not in self.v4l2_controls.controls:
                    continue
                if self._camera_set_control(name, int(values[name]), quiet=True):
                    applied.append(name)
            if applied:
                self._append_log("저장된 카메라 설정 적용: " + ", ".join(applied))
        except Exception as exc:
            self._append_log(f"저장된 카메라 설정 적용 실패: {exc}")

    def _save_current_camera_controls(self) -> None:
        names = self._relevant_camera_control_names()
        values = self.v4l2_controls.snapshot(names)
        payload = {
            "device": self.v4l2_controls.device,
            "saved_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "controls": values,
        }
        try:
            self.camera_control_config_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.camera_control_config_path.with_suffix(
                self.camera_control_config_path.suffix + ".tmp"
            )
            tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            tmp.replace(self.camera_control_config_path)
            self.camera_control_status.setText("현재 카메라 설정을 저장했습니다.")
            self._append_log(f"카메라 설정 저장 완료: {self.camera_control_config_path}")
        except Exception as exc:
            QMessageBox.warning(self.window, "저장 실패", f"카메라 설정 저장 실패:\n{exc}")

    def _restore_camera_defaults(self) -> None:
        defaults = self.v4l2_controls.defaults(self._relevant_camera_control_names())
        if not defaults:
            self.camera_control_status.setText("복원할 기본 제어값이 없습니다.")
            return
        for name, value in defaults.items():
            self._camera_set_control(name, value, quiet=True)
        self._refresh_camera_control_values(show_error=False)
        self.camera_control_status.setText("카메라 드라이버 기본값으로 복원했습니다.")
        self._append_log("카메라 제어값을 드라이버 기본값으로 복원함")

    def _set_continuous_camera_auto(self) -> None:
        self._set_exposure_automatic(True)
        self._set_awb_automatic(True, quiet=True)
        self.camera_control_status.setText(
            "연속 자동 모드입니다. 장면 변화에 따라 밝기와 색이 계속 바뀔 수 있습니다."
        )
        self._refresh_camera_control_values(show_error=False)

    def _set_manual_camera_mode(self) -> None:
        self._set_exposure_automatic(False)
        self._set_awb_automatic(self.camera_awb_checkbox.isChecked(), quiet=True)
        self._apply_exposure_value()
        self._apply_gain_value()
        if not self.camera_awb_checkbox.isChecked():
            self._apply_wb_value()
        self.camera_control_status.setText("현재 표시값을 수동으로 고정했습니다.")
        self._refresh_camera_control_values(show_error=False)

    @staticmethod
    def _camera_autofit_validate_h(value: Any) -> Optional[np.ndarray]:
        try:
            H = np.asarray(value, dtype=np.float64)
        except Exception:
            return None
        if H.shape != (3, 3) or not np.all(np.isfinite(H)):
            return None
        if abs(float(np.linalg.det(H))) < 1e-12:
            return None
        return H

    def _camera_autofit_resolve_meter_h(self) -> Tuple[np.ndarray, str]:
        """Resolve image->board homography for the *corrected* camera frame.

        The action model is trained from rectified ``board_512`` images, so v24
        meters the same board view instead of the entire 1280x720 scene.  Prefer
        the H locked in this UI session, then a live four-marker estimate, then
        the saved corrected-frame H from the homography cache.
        """
        H = self._camera_autofit_validate_h(self.aruco_latest_H)
        if H is not None:
            return H, "session_locked_H"

        frame = self.latest_camera_frame
        if frame is not None and isinstance(frame, np.ndarray) and frame.size:
            try:
                corners, ids, _ = detect_markers(frame, self.aruco_detector)
                live_H, _ = compute_homography(
                    corners,
                    ids,
                    self.machine_cfg["aruco"]["marker_board_mm"],
                    self.required_marker_ids,
                )
                H = self._camera_autofit_validate_h(live_H)
                if H is not None:
                    return H, "live_aruco_H"
            except Exception as exc:
                self._append_log(f"카메라 SW-AE live H 감지 실패(캐시 시도): {exc!r}")

        if self.hfile_path.exists():
            try:
                payload = json.loads(self.hfile_path.read_text(encoding="utf-8"))
                candidate = payload.get("H") if isinstance(payload, dict) else payload
                if isinstance(payload, dict):
                    saved_geometry = payload.get("camera_geometry")
                    if isinstance(saved_geometry, dict):
                        saved_id = str(saved_geometry.get("calibration_id") or "")
                        current_id = str(self.camera_geometry.get("calibration_id") or "")
                        saved_size = saved_geometry.get("output_size")
                        current_size = self.camera_geometry.get("output_size")
                        if saved_id and current_id and saved_id != current_id:
                            self._append_log(
                                "카메라 SW-AE 저장 H 무시: calibration_id mismatch "
                                f"saved={saved_id} current={current_id}"
                            )
                            candidate = None
                        elif (
                            isinstance(saved_size, (list, tuple))
                            and isinstance(current_size, (list, tuple))
                            and list(saved_size) != list(current_size)
                        ):
                            self._append_log(
                                "카메라 SW-AE 저장 H 무시: output_size mismatch "
                                f"saved={saved_size} current={current_size}"
                            )
                            candidate = None
                H = self._camera_autofit_validate_h(candidate)
                if H is not None:
                    return H, "saved_corrected_H"
            except Exception as exc:
                self._append_log(f"카메라 SW-AE H 캐시 읽기 실패: {exc!r}")

        raise RuntimeError(
            "학습환경 기준 자동노출에는 보드 Homography가 필요합니다. "
            "ArUco ID0~ID3이 보이게 하거나 Homography 보정을 먼저 완료하세요."
        )

    def _camera_autofit_rectified_board(self) -> np.ndarray:
        frame = self.latest_camera_frame
        H = self._camera_autofit_meter_H
        if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
            raise RuntimeError("보정 카메라 프레임이 아직 준비되지 않았습니다.")
        if H is None:
            raise RuntimeError("자동노출 보드 Homography가 준비되지 않았습니다.")

        marker_points = np.asarray(
            list(self.machine_cfg["aruco"]["marker_board_mm"].values()), dtype=np.float64
        )
        if marker_points.ndim != 2 or marker_points.shape[0] < 4 or marker_points.shape[1] != 2:
            raise RuntimeError("aruco.marker_board_mm 설정이 올바르지 않습니다.")
        min_x = float(np.min(marker_points[:, 0])); max_x = float(np.max(marker_points[:, 0]))
        min_y = float(np.min(marker_points[:, 1])); max_y = float(np.max(marker_points[:, 1]))

        try:
            inv_H = np.linalg.inv(np.asarray(H, dtype=np.float64))
        except Exception as exc:
            raise RuntimeError(f"Homography 역행렬 계산 실패: {exc}") from exc

        def board_to_pixel(bx: float, by: float) -> Tuple[float, float]:
            q = inv_H @ np.asarray([bx, by, 1.0], dtype=np.float64)
            if abs(float(q[2])) < 1e-9:
                raise RuntimeError("Homography 투영 결과가 유효하지 않습니다.")
            return float(q[0] / q[2]), float(q[1] / q[2])

        src = np.asarray(
            [
                board_to_pixel(min_x, max_y),
                board_to_pixel(max_x, max_y),
                board_to_pixel(max_x, min_y),
                board_to_pixel(min_x, min_y),
            ],
            dtype=np.float32,
        )
        dst = np.asarray([[0, 0], [511, 0], [511, 511], [0, 511]], dtype=np.float32)
        M = cv2.getPerspectiveTransform(src, dst)
        board = cv2.warpPerspective(frame, M, (512, 512), flags=cv2.INTER_AREA)
        if board is None or board.size == 0:
            raise RuntimeError("보드 512x512 rectification에 실패했습니다.")
        return board

    def _camera_autofit_frame_stats(self) -> Tuple[float, float, float, float, float, float]:
        """Return training-matched brightness stats from rectified board_512.

        P20/P30 meter the dark board background and are the primary exposure
        signal.  P98/P99.5/clip250 are highlight guards so a light garment cannot
        be blown out while the black board is being matched.
        """
        board = self._camera_autofit_rectified_board()
        gray = cv2.cvtColor(board, cv2.COLOR_BGR2GRAY)
        values = gray.reshape(-1).astype(np.float32)
        if values.size < 1000:
            raise RuntimeError("자동 노출 측정용 board_512 픽셀이 부족합니다.")
        mean_v = float(np.mean(values))
        p20_v = float(np.percentile(values, 20.0))
        p30_v = float(np.percentile(values, 30.0))
        p98_v = float(np.percentile(values, 98.0))
        p995_v = float(np.percentile(values, 99.5))
        clip_frac = float(np.mean(values >= 250.0))
        return mean_v, p20_v, p30_v, p98_v, p995_v, clip_frac

    def _camera_autofit_score(
        self, p20_v: float, p30_v: float, p98_v: float, p995_v: float, clip_frac: float
    ) -> float:
        def band_penalty(value: float, low: float, high: float) -> float:
            width = max(1.0, float(high - low))
            if value < low:
                return float(low - value) / width
            if value > high:
                return float(value - high) / width
            return 0.0

        score = 2.0 * band_penalty(
            p20_v, self._camera_autofit_p20_low, self._camera_autofit_p20_high
        )
        score += 1.0 * band_penalty(
            p30_v, self._camera_autofit_p30_low, self._camera_autofit_p30_high
        )
        if p98_v > self._camera_autofit_p98_high:
            score += (p98_v - self._camera_autofit_p98_high) / 8.0
        if p995_v >= self._camera_autofit_hard_clip_p995:
            score += 2.0 + (p995_v - self._camera_autofit_hard_clip_p995) / 4.0
        if clip_frac > self._camera_autofit_clip_max:
            score += 4.0 * (clip_frac - self._camera_autofit_clip_max) / max(
                1e-6, self._camera_autofit_clip_max
            )
        return float(score)

    def _camera_autofit_set_exposure(self, value: int) -> bool:
        name = self.camera_control_names.get("exposure")
        if name is None:
            return False
        info = self.v4l2_controls.controls.get(name)
        lo = int(info.minimum if info is not None and info.minimum is not None else 1)
        hi = int(info.maximum if info is not None and info.maximum is not None else 5000)
        step = max(1, int(info.step if info is not None and info.step is not None else 1))
        value = int(np.clip(int(value), lo, hi))
        value = lo + int(round((value - lo) / step)) * step
        value = int(np.clip(value, lo, hi))
        if not self._camera_set_control(name, value, quiet=True):
            return False
        self._camera_ui_syncing = True
        try:
            self.camera_exposure_slider.setValue(value)
            self.camera_exposure_spin.setValue(value)
        finally:
            self._camera_ui_syncing = False
        self._update_camera_control_hints()
        return True

    def _camera_autofit_finish(self, reason: str, *, success: bool = True) -> None:
        try:
            self._set_exposure_automatic(False, quiet=True)
            self._set_awb_automatic(False, quiet=True)
            wb_name = self.camera_control_names.get("wb_temperature")
            if wb_name is not None:
                self._camera_set_control(wb_name, int(self.camera_wb_spin.value()), quiet=True)
            gain_name = self.camera_control_names.get("gain")
            if gain_name is not None:
                info = self.v4l2_controls.controls.get(gain_name)
                gain_min = int(info.minimum if info is not None and info.minimum is not None else 0)
                self._camera_set_control(gain_name, gain_min, quiet=True)
            self._camera_set_control(self.camera_control_names.get("backlight_compensation"), 0, quiet=True)
            self._camera_set_control(self.camera_control_names.get("exposure_dynamic_framerate"), 0, quiet=True)
            self._refresh_camera_control_values(show_error=False)
        finally:
            self._camera_autofit_running = False
            self.camera_autofit_button.setEnabled(True)

        exp = self.camera_exposure_spin.value()
        if success:
            self.camera_control_badge.setText("학습 밝기 맞춤 고정됨")
            self.camera_control_status.setText(
                f"학습환경 밝기 맞춤 완료: exposure={exp}. {reason} "
                "확인 후 '현재 설정 저장'을 눌러 JSON에 저장하세요."
            )
        else:
            self.camera_control_badge.setText("자동 맞춤 확인 필요")
            self.camera_control_status.setText(f"학습환경 밝기 맞춤 중지/불완전: {reason}")
        self._append_log(
            f"카메라 training-match SW-AE {'완료' if success else '중지'}: exposure={exp} "
            f"meter={self._camera_autofit_meter_source} mean={self._camera_autofit_last_mean} "
            f"p20={self._camera_autofit_last_p20} p30={self._camera_autofit_last_p30} "
            f"p98={self._camera_autofit_last_p98} p995={self._camera_autofit_last_p995} "
            f"clip250={self._camera_autofit_last_clip} reason={reason}"
        )

    def _camera_autofit_iteration(self) -> None:
        if not self._camera_autofit_running:
            return
        try:
            mean_v, p20_v, p30_v, p98_v, p995_v, clip_frac = self._camera_autofit_frame_stats()
            self._camera_autofit_last_mean = mean_v
            self._camera_autofit_last_p20 = p20_v
            self._camera_autofit_last_p30 = p30_v
            self._camera_autofit_last_p98 = p98_v
            self._camera_autofit_last_p995 = p995_v
            self._camera_autofit_last_clip = clip_frac
            self._camera_autofit_step += 1

            current = int(self.camera_exposure_spin.value())
            p20_low = float(self._camera_autofit_p20_low)
            p20_high = float(self._camera_autofit_p20_high)
            p30_low = float(self._camera_autofit_p30_low)
            p30_high = float(self._camera_autofit_p30_high)
            p98_high = float(self._camera_autofit_p98_high)
            hard_clip = float(self._camera_autofit_hard_clip_p995)
            clip_max = float(self._camera_autofit_clip_max)

            highlight_unsafe = bool(
                p995_v >= hard_clip or p98_v > p98_high or clip_frac > clip_max
            )
            if highlight_unsafe:
                direction = "bright"
            elif p20_v > p20_high:
                direction = "bright"
            elif p20_v < p20_low:
                direction = "dark"
            elif p30_v > p30_high:
                direction = "bright"
            elif p30_v < p30_low:
                direction = "dark"
            else:
                direction = "ok"

            score = self._camera_autofit_score(p20_v, p30_v, p98_v, p995_v, clip_frac)
            if score < float(self._camera_autofit_best_score):
                self._camera_autofit_best_score = float(score)
                self._camera_autofit_best_exp = int(current)

            self.camera_control_status.setText(
                f"학습 밝기 맞춤 {self._camera_autofit_step}/{self._camera_autofit_max_steps} · "
                f"exposure={current} · P20={p20_v:.1f} (목표 {p20_low:.0f}~{p20_high:.0f}) · "
                f"P30={p30_v:.1f} (목표 {p30_low:.0f}~{p30_high:.0f}) · "
                f"P98={p98_v:.1f} · clip={clip_frac*100:.1f}%"
            )
            self._append_log(
                f"카메라 training-match SW-AE step={self._camera_autofit_step} exposure={current} "
                f"meter={self._camera_autofit_meter_source} mean={mean_v:.2f} "
                f"p20={p20_v:.2f} p30={p30_v:.2f} p98={p98_v:.2f} "
                f"p995={p995_v:.2f} clip250={clip_frac:.4f} score={score:.4f} dir={direction}"
            )

            if direction == "ok":
                self._camera_autofit_finish(
                    f"board_512 P20={p20_v:.1f}, P30={p30_v:.1f}, "
                    f"P98={p98_v:.1f}, clip={clip_frac*100:.1f}%가 학습 범위에 들어왔습니다."
                )
                return

            if self._camera_autofit_step >= int(self._camera_autofit_max_steps):
                best = self._camera_autofit_best_exp
                if best is not None and best != current:
                    self._camera_autofit_set_exposure(best)
                self._camera_autofit_finish(
                    f"최대 {self._camera_autofit_max_steps}회 탐색. 가장 가까운 exposure="
                    f"{self.camera_exposure_spin.value()}를 복원했습니다. 화면을 확인하세요.",
                    success=False,
                )
                return

            if direction == "bright":
                self._camera_autofit_high_exp = current if self._camera_autofit_high_exp is None else min(
                    int(self._camera_autofit_high_exp), current
                )
                if self._camera_autofit_low_exp is not None:
                    next_exp = (int(self._camera_autofit_low_exp) + int(self._camera_autofit_high_exp)) // 2
                else:
                    severe = bool(clip_frac > 0.05 or p995_v >= 254.0)
                    factor = 0.72 if severe else 0.85
                    next_exp = max(1, int(round(current * factor)))
            else:
                self._camera_autofit_low_exp = current if self._camera_autofit_low_exp is None else max(
                    int(self._camera_autofit_low_exp), current
                )
                if self._camera_autofit_high_exp is not None:
                    next_exp = (int(self._camera_autofit_low_exp) + int(self._camera_autofit_high_exp)) // 2
                else:
                    # 151 -> ~181 on the first under-exposed correction, matching
                    # the manually verified dark-room neighbourhood without hard-coding 180.
                    next_exp = max(current + 1, int(round(current * 1.20)))

            if next_exp == current:
                best = self._camera_autofit_best_exp
                if best is not None and best != current:
                    self._camera_autofit_set_exposure(best)
                self._camera_autofit_finish(
                    f"더 이상 exposure 단위를 좁힐 수 없습니다. 가장 가까운 exposure="
                    f"{self.camera_exposure_spin.value()}를 유지합니다.",
                    success=False,
                )
                return
            if not self._camera_autofit_set_exposure(next_exp):
                raise RuntimeError(f"exposure={next_exp} 적용에 실패했습니다.")
            QTimer.singleShot(int(self._camera_autofit_settle_ms), self._camera_autofit_iteration)
        except Exception as exc:
            self._camera_autofit_finish(str(exc), success=False)

    def _start_camera_autofit(self) -> None:
        if self._camera_autofit_running:
            return
        exposure_name = self.camera_control_names.get("exposure")
        if exposure_name is None:
            self.camera_control_status.setText("이 카메라에서는 수동 exposure 제어를 찾지 못했습니다.")
            return
        if self.latest_camera_frame is None:
            self.camera_control_status.setText("카메라 프레임이 준비된 뒤 다시 시도하세요.")
            return

        # Resolve the same corrected-frame board geometry used for the model's
        # board_512 input *before* changing camera controls.  If this cannot be
        # established, do not guess using full-frame statistics.
        try:
            meter_H, meter_source = self._camera_autofit_resolve_meter_h()
        except Exception as exc:
            self.camera_control_badge.setText("자동 맞춤 불가")
            self.camera_control_status.setText(str(exc))
            self._append_log(f"카메라 training-match SW-AE 시작 실패: {exc!r}")
            return

        self._camera_autofit_running = True
        self._camera_autofit_step = 0
        self._camera_autofit_low_exp = None
        self._camera_autofit_high_exp = None
        self._camera_autofit_last_p20 = None
        self._camera_autofit_last_p30 = None
        self._camera_autofit_last_p98 = None
        self._camera_autofit_last_p995 = None
        self._camera_autofit_last_mean = None
        self._camera_autofit_last_clip = None
        self._camera_autofit_meter_H = meter_H.copy()
        self._camera_autofit_meter_source = str(meter_source)
        self._camera_autofit_best_exp = None
        self._camera_autofit_best_score = float("inf")
        self.camera_autofit_button.setEnabled(False)

        # Critical OV2710 policy: hardware AE stays OFF.  On this camera
        # exposure_time_absolute becomes flags=inactive in Aperture Priority mode
        # and the AE-selected shutter cannot be read back/frozen.
        self._set_exposure_automatic(False, quiet=True)
        self._set_awb_automatic(False, quiet=True)
        wb_name = self.camera_control_names.get("wb_temperature")
        if wb_name is not None:
            self._camera_set_control(wb_name, int(self.camera_wb_spin.value()), quiet=True)
        self._camera_set_control(self.camera_control_names.get("backlight_compensation"), 0, quiet=True)
        self._camera_set_control(self.camera_control_names.get("exposure_dynamic_framerate"), 0, quiet=True)

        gain_name = self.camera_control_names.get("gain")
        if gain_name is not None:
            info = self.v4l2_controls.controls.get(gain_name)
            gain_min = int(info.minimum if info is not None and info.minimum is not None else 0)
            self._camera_set_control(gain_name, gain_min, quiet=True)
            self._camera_ui_syncing = True
            try:
                self.camera_gain_slider.setValue(gain_min)
                self.camera_gain_spin.setValue(gain_min)
            finally:
                self._camera_ui_syncing = False

        self.camera_control_badge.setText("학습 밝기 맞춤 중")
        self.camera_control_status.setText(
            "Action Model 학습 당시 board_512 밝기에 맞춰 manual exposure를 탐색합니다. "
            "옷/조명을 움직이지 마세요."
        )
        self._append_log(
            "카메라 training-match SW-AE 시작: hardware AE=OFF, gain=min, AWB=OFF, "
            "backlight=0, dynamic_fps=0, "
            f"meter={self._camera_autofit_meter_source}, "
            f"target_p20={self._camera_autofit_p20_low:.0f}~{self._camera_autofit_p20_high:.0f}, "
            f"target_p30={self._camera_autofit_p30_low:.0f}~{self._camera_autofit_p30_high:.0f}, "
            f"p98_guard<={self._camera_autofit_p98_high:.0f}, clip_max={self._camera_autofit_clip_max*100.0:.1f}%"
        )
        QTimer.singleShot(int(self._camera_autofit_settle_ms), self._camera_autofit_iteration)

    def _apply_exposure_value(self) -> None:
        if self._camera_ui_syncing:
            return
        self._set_exposure_automatic(False, quiet=True)
        value = self.camera_exposure_spin.value()
        self._camera_set_control(self.camera_control_names.get("exposure"), value)
        self._update_camera_control_hints()

    def _apply_gain_value(self) -> None:
        if self._camera_ui_syncing:
            return
        # Gain is commonly inactive while UVC auto exposure is enabled, even
        # when the camera exposes no separate autogain control.
        self._set_exposure_automatic(False, quiet=True)
        auto_gain_name = self.camera_control_names.get("auto_gain")
        if auto_gain_name is not None:
            self._camera_set_control(auto_gain_name, 0, quiet=True)
        self._camera_set_control(
            self.camera_control_names.get("gain"), self.camera_gain_spin.value()
        )

    def _apply_wb_value(self) -> None:
        if self._camera_ui_syncing:
            return
        self._set_awb_automatic(False, quiet=True)
        self._camera_set_control(
            self.camera_control_names.get("wb_temperature"), self.camera_wb_spin.value()
        )

    def _on_exposure_slider_changed(self, value: int) -> None:
        if self._camera_ui_syncing:
            return
        self._camera_ui_syncing = True
        self.camera_exposure_spin.setValue(value)
        self._camera_ui_syncing = False
        self._update_camera_control_hints()

    def _on_gain_slider_changed(self, value: int) -> None:
        if self._camera_ui_syncing:
            return
        self._camera_ui_syncing = True
        self.camera_gain_spin.setValue(value)
        self._camera_ui_syncing = False

    def _on_wb_slider_changed(self, value: int) -> None:
        if self._camera_ui_syncing:
            return
        self._camera_ui_syncing = True
        self.camera_wb_spin.setValue(value)
        self._camera_ui_syncing = False

    def _on_exposure_spin_changed(self, value: int) -> None:
        if self._camera_ui_syncing:
            return
        self._camera_ui_syncing = True
        self.camera_exposure_slider.setValue(value)
        self._camera_ui_syncing = False
        self._update_camera_control_hints()

    def _on_gain_spin_changed(self, value: int) -> None:
        if self._camera_ui_syncing:
            return
        self._camera_ui_syncing = True
        self.camera_gain_slider.setValue(value)
        self._camera_ui_syncing = False

    def _on_wb_spin_changed(self, value: int) -> None:
        if self._camera_ui_syncing:
            return
        self._camera_ui_syncing = True
        self.camera_wb_slider.setValue(value)
        self._camera_ui_syncing = False

    def _on_flicker_changed(self, index: int) -> None:
        if self._camera_ui_syncing or index < 0:
            return
        value = self.camera_flicker_combo.itemData(index)
        if value is None:
            return
        if self._camera_set_control(self.camera_control_names.get("power_line"), int(value)):
            self.camera_control_status.setText(
                f"플리커 방지를 {self.camera_flicker_combo.currentText()}로 적용했습니다."
            )

    def _on_awb_toggled(self, checked: bool) -> None:
        if self._camera_ui_syncing:
            return
        self._set_awb_automatic(checked)
        self.camera_wb_slider.setEnabled(not checked)
        self.camera_wb_spin.setEnabled(not checked)
        self.camera_control_status.setText(
            "화이트밸런스 자동 모드입니다." if checked else "화이트밸런스를 수동으로 조절할 수 있습니다."
        )

    def _enter_camera_test(self) -> None:
        self.page_stack.setCurrentWidget(self.camera_test_page)
        if self.latest_camera_frame is not None:
            self.camera_test_view.set_bgr_frame(self.latest_camera_frame)
        self._refresh_camera_control_values(show_error=False)

    def _leave_camera_test(self) -> None:
        self.page_stack.setCurrentWidget(self.settings_page)


    def _create_aruco_page(self) -> QWidget:
        page = QWidget()
        page.setObjectName("arucoCalibrationPage")
        root = QVBoxLayout(page)
        root.setContentsMargins(24, 20, 24, 20)
        root.setSpacing(16)

        header, self.aruco_back_button = self._create_page_header(
            "AruCo Marker 보정",
            "4개 마커를 감지해 카메라 픽셀 좌표를 보드 좌표로 고정합니다.",
            "arucoBackButton",
        )
        self.aruco_header_badge = QLabel("보정 시작 전")
        self.aruco_header_badge.setObjectName("calibrationStatusBadge")
        self.aruco_header_badge.setMinimumSize(220, 52)
        self.aruco_header_badge.setAlignment(Qt.AlignmentFlag.AlignCenter)
        header.layout().addWidget(self.aruco_header_badge)
        root.addWidget(header)

        content = QHBoxLayout()
        content.setSpacing(18)

        display_card = QFrame()
        display_card.setObjectName("calibrationDisplayCard")
        display_layout = QVBoxLayout(display_card)
        display_layout.setContentsMargins(16, 14, 16, 14)
        display_layout.setSpacing(8)
        title = QLabel("마커 위치 및 카메라 영상")
        title.setObjectName("cardTitle")
        display_layout.addWidget(title)

        self.aruco_display_stack = QStackedWidget()
        self.aruco_display_stack.setObjectName("arucoDisplayStack")
        self.aruco_guide_widget = CalibrationGuideWidget(GUIDE_IMAGE_PATH, mode="aruco")
        self.aruco_camera_label = FitVideoWidget(
            "보정 시작을 누르면 카메라 영상이 표시됩니다."
        )
        self.aruco_camera_label.setObjectName("calibrationCameraLabel")
        self.aruco_camera_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.aruco_camera_label.setMinimumSize(480, 270)
        self.aruco_display_stack.addWidget(self.aruco_guide_widget)
        self.aruco_display_stack.addWidget(self.aruco_camera_label)
        self.aruco_display_stack.setCurrentWidget(self.aruco_guide_widget)
        display_layout.addWidget(self.aruco_display_stack, 1)
        content.addWidget(display_card, 3)

        control_card = QFrame()
        control_card.setObjectName("calibrationControlCard")
        control_layout = QVBoxLayout(control_card)
        control_layout.setContentsMargins(20, 18, 20, 18)
        control_layout.setSpacing(12)

        control_title = QLabel("Homography Lock")
        control_title.setObjectName("cardTitle")
        control_layout.addWidget(control_title)

        self.aruco_status_label = QLabel("보정 시작 전 · 폴딩 보드의 마커 위치를 확인하세요.")
        self.aruco_status_label.setObjectName("calibrationMainStatus")
        self.aruco_status_label.setWordWrap(True)
        self.aruco_status_label.setMinimumHeight(70)
        control_layout.addWidget(self.aruco_status_label)

        self.aruco_detection_label = QLabel(
            "필요 마커: ID0, ID1, ID2, ID3\n감지 상태: 대기"
        )
        self.aruco_detection_label.setObjectName("markerDetectionLabel")
        self.aruco_detection_label.setWordWrap(True)
        self.aruco_detection_label.setMinimumHeight(92)
        control_layout.addWidget(self.aruco_detection_label)

        self.aruco_start_button = QPushButton("보정 시작\nHomography Reset → 4개 마커 자동 Lock")
        self.aruco_start_button.setObjectName("calibrationStartButton")
        self.aruco_start_button.setMinimumHeight(92)
        control_layout.addWidget(self.aruco_start_button)

        path_label = QLabel(f"저장 파일\n{self.hfile_path}")
        path_label.setObjectName("calibrationPathLabel")
        path_label.setWordWrap(True)
        control_layout.addWidget(path_label)

        log_title = QLabel("보정 로그")
        log_title.setObjectName("logTitle")
        control_layout.addWidget(log_title)
        self.aruco_log_view = QPlainTextEdit()
        self.aruco_log_view.setObjectName("calibrationLogView")
        self.aruco_log_view.setReadOnly(True)
        self.aruco_log_view.setMaximumBlockCount(100)
        control_layout.addWidget(self.aruco_log_view, 1)

        warning = QLabel(
            "마커 배치: 오른쪽 위 ID0, 오른쪽 아래 ID1, 왼쪽 위 ID2, 왼쪽 아래 ID3. "
            "보정 완료 후에는 카메라와 폴딩 보드의 상대 위치를 움직이지 마세요."
        )
        warning.setObjectName("calibrationWarningLabel")
        warning.setWordWrap(True)
        warning.setMinimumHeight(84)
        control_layout.addWidget(warning)

        content.addWidget(control_card, 2)
        root.addLayout(content, 1)
        return page

    def _create_robot_test_page(self) -> QWidget:
        page = QWidget()
        page.setObjectName("robotTestPage")
        root = QVBoxLayout(page)
        root.setContentsMargins(24, 20, 24, 20)
        root.setSpacing(16)

        header, self.robot_test_back_button = self._create_page_header(
            "로봇팔 작동 점검",
            "사용자 기준 로봇팔을 선택하고 자동 경로 또는 수동 미세 조작을 실행합니다.",
            "robotTestBackButton",
        )
        self.robot_test_header_badge = QLabel("로봇팔 1 · ARM1")
        self.robot_test_header_badge.setObjectName("robotTestStatusBadge")
        self.robot_test_header_badge.setMinimumSize(220, 52)
        self.robot_test_header_badge.setAlignment(Qt.AlignmentFlag.AlignCenter)
        header.layout().addWidget(self.robot_test_header_badge)
        root.addWidget(header)

        content = QHBoxLayout()
        content.setSpacing(18)

        visual_card = QFrame()
        visual_card.setObjectName("robotVisualCard")
        visual_layout = QVBoxLayout(visual_card)
        visual_layout.setContentsMargins(16, 14, 16, 14)
        visual_layout.setSpacing(10)
        visual_title = QLabel("로봇팔 선택 · 사용자 시점")
        visual_title.setObjectName("cardTitle")
        visual_layout.addWidget(visual_title)
        self.robot_top_view = RobotTopViewWidget(ROBOT_IMAGE_PATH)
        visual_layout.addWidget(self.robot_top_view, 1)

        select_row = QHBoxLayout()
        select_row.setSpacing(10)
        self.robot_select_arm1_button = QPushButton("로봇팔 1\nARM1 · /dev/roarm_1")
        self.robot_select_arm2_button = QPushButton("로봇팔 2\nARM2 · /dev/roarm_2")
        for button in (self.robot_select_arm1_button, self.robot_select_arm2_button):
            button.setObjectName("robotSelectButton")
            button.setMinimumHeight(66)
            button.setCheckable(True)
            select_row.addWidget(button)
        self.robot_select_arm1_button.setChecked(True)
        visual_layout.addLayout(select_row)
        content.addWidget(visual_card, 3)

        control_card = QFrame()
        control_card.setObjectName("robotControlCard")
        control_layout = QVBoxLayout(control_card)
        control_layout.setContentsMargins(20, 18, 20, 18)
        control_layout.setSpacing(10)
        control_title = QLabel("점검 제어")
        control_title.setObjectName("cardTitle")
        control_layout.addWidget(control_title)

        self.robot_test_status = QLabel("선택됨: 로봇팔 1 · ARM1 · 연결 대기")
        self.robot_test_status.setObjectName("robotTestMainStatus")
        self.robot_test_status.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.robot_test_status.setWordWrap(True)
        self.robot_test_status.setMinimumHeight(58)
        control_layout.addWidget(self.robot_test_status)

        self.robot_auto_test_button = QPushButton("◇  자동 점검\n안전 높이에서 전용 작업 영역 한 바퀴 + 그리퍼 점검")
        self.robot_auto_test_button.setObjectName("robotAutoTestButton")
        self.robot_auto_test_button.setMinimumHeight(88)
        control_layout.addWidget(self.robot_auto_test_button)

        manual_title = QLabel("수동 점검 · 한 번 누를 때마다 조금씩 이동")
        manual_title.setObjectName("robotManualTitle")
        control_layout.addWidget(manual_title)

        step_row = QHBoxLayout()
        step_label = QLabel("이동 단위")
        step_label.setObjectName("robotControlLabel")
        self.robot_step_combo = QComboBox()
        self.robot_step_combo.setObjectName("robotStepCombo")
        self.robot_step_combo.addItems(["5 mm", "10 mm", "20 mm"])
        self.robot_step_combo.setCurrentIndex(1)
        step_row.addWidget(step_label)
        step_row.addWidget(self.robot_step_combo, 1)
        control_layout.addLayout(step_row)

        motion_grid = QGridLayout()
        motion_grid.setHorizontalSpacing(8)
        motion_grid.setVerticalSpacing(8)
        self.robot_move_forward = QPushButton("↑  앞으로\n보드 안쪽")
        self.robot_move_back = QPushButton("↓  뒤로\n보드 바깥쪽")
        self.robot_move_left = QPushButton("←  사용자 왼쪽")
        self.robot_move_right = QPushButton("→  사용자 오른쪽")
        self.robot_move_up = QPushButton("Z+  위로")
        self.robot_move_down = QPushButton("Z−  아래로")
        motion_grid.addWidget(self.robot_move_forward, 0, 1)
        motion_grid.addWidget(self.robot_move_left, 1, 0)
        motion_grid.addWidget(self.robot_move_back, 1, 1)
        motion_grid.addWidget(self.robot_move_right, 1, 2)
        motion_grid.addWidget(self.robot_move_up, 2, 0)
        motion_grid.addWidget(self.robot_move_down, 2, 2)
        for button in (self.robot_move_forward, self.robot_move_back, self.robot_move_left,
                       self.robot_move_right, self.robot_move_up, self.robot_move_down):
            button.setObjectName("robotJogButton")
            button.setMinimumHeight(54)
        control_layout.addLayout(motion_grid)

        wrist_row = QHBoxLayout()
        self.robot_wrist_ccw = QPushButton("↶  손목 좌회전")
        self.robot_wrist_cw = QPushButton("↷  손목 우회전")
        self.robot_gripper_open = QPushButton("그리퍼 열기")
        self.robot_gripper_close = QPushButton("그리퍼 닫기")
        for button in (self.robot_wrist_ccw, self.robot_wrist_cw,
                       self.robot_gripper_open, self.robot_gripper_close):
            button.setObjectName("robotAuxButton")
            button.setMinimumHeight(50)
            wrist_row.addWidget(button)
        control_layout.addLayout(wrist_row)

        utility_row = QHBoxLayout()
        self.robot_torque_on = QPushButton("토크 ON")
        self.robot_torque_off = QPushButton("토크 OFF")
        self.robot_feedback_button = QPushButton("현재 좌표")
        self.robot_standby_button = QPushButton("대기 위치")
        for button in (self.robot_torque_on, self.robot_torque_off,
                       self.robot_feedback_button, self.robot_standby_button):
            button.setObjectName("robotUtilityButton")
            button.setMinimumHeight(48)
            utility_row.addWidget(button)
        control_layout.addLayout(utility_row)

        log_title = QLabel("상태 로그")
        log_title.setObjectName("logTitle")
        control_layout.addWidget(log_title)
        self.robot_test_log = QPlainTextEdit()
        self.robot_test_log.setObjectName("robotTestLog")
        self.robot_test_log.setReadOnly(True)
        self.robot_test_log.setMaximumBlockCount(160)
        self.robot_test_log.setMinimumHeight(120)
        control_layout.addWidget(self.robot_test_log, 1)

        warning = QLabel(
            "주의: 자동 점검은 높은 Z에서만 수행합니다. 수동 Z−는 보드 충돌 위험이 있으므로 "
            "5mm 단위부터 사용하고, 물리 비상정지를 바로 누를 수 있는 상태에서 실행하세요."
        )
        warning.setObjectName("robotTestWarning")
        warning.setWordWrap(True)
        warning.setMinimumHeight(62)
        control_layout.addWidget(warning)

        content.addWidget(control_card, 2)
        root.addLayout(content, 1)

        self.robot_top_view.armClicked.connect(self._select_robot_test_arm)
        self.robot_select_arm1_button.clicked.connect(lambda: self._select_robot_test_arm("arm1"))
        self.robot_select_arm2_button.clicked.connect(lambda: self._select_robot_test_arm("arm2"))
        self.robot_auto_test_button.clicked.connect(self._request_robot_auto_test)
        self.robot_move_forward.clicked.connect(lambda: self._request_robot_jog("forward"))
        self.robot_move_back.clicked.connect(lambda: self._request_robot_jog("back"))
        self.robot_move_left.clicked.connect(lambda: self._request_robot_jog("left"))
        self.robot_move_right.clicked.connect(lambda: self._request_robot_jog("right"))
        self.robot_move_up.clicked.connect(lambda: self._request_robot_jog("up"))
        self.robot_move_down.clicked.connect(lambda: self._request_robot_jog("down"))
        self.robot_wrist_ccw.clicked.connect(lambda: self._request_robot_jog("wrist_ccw"))
        self.robot_wrist_cw.clicked.connect(lambda: self._request_robot_jog("wrist_cw"))
        self.robot_gripper_open.clicked.connect(lambda: self._request_robot_gripper(True))
        self.robot_gripper_close.clicked.connect(lambda: self._request_robot_gripper(False))
        self.robot_torque_on.clicked.connect(lambda: self._request_robot_torque(True))
        self.robot_torque_off.clicked.connect(lambda: self._request_robot_torque(False))
        self.robot_feedback_button.clicked.connect(self._request_robot_feedback)
        self.robot_standby_button.clicked.connect(self._request_selected_robot_standby)
        return page

    def _create_robot_calibration_page(self) -> QWidget:
        page = QWidget()
        page.setObjectName("robotCalibrationPage")
        root = QVBoxLayout(page)
        root.setContentsMargins(24, 20, 24, 20)
        root.setSpacing(16)

        header, self.robot_calib_back_button = self._create_page_header(
            "로봇팔 보정",
            "각 로봇팔 끝점을 마커와 스티커에 맞추고 현재 좌표를 기록합니다.",
            "robotCalibBackButton",
        )
        self.robot_header_badge = QLabel("보정 시작 전")
        self.robot_header_badge.setObjectName("calibrationStatusBadge")
        self.robot_header_badge.setMinimumSize(220, 52)
        self.robot_header_badge.setAlignment(Qt.AlignmentFlag.AlignCenter)
        header.layout().addWidget(self.robot_header_badge)
        root.addWidget(header)

        content = QHBoxLayout()
        content.setSpacing(18)

        guide_card = QFrame()
        guide_card.setObjectName("calibrationDisplayCard")
        guide_layout = QVBoxLayout(guide_card)
        guide_layout.setContentsMargins(16, 14, 16, 14)
        guide_layout.setSpacing(8)
        title = QLabel("현재 로봇팔을 놓을 위치")
        title.setObjectName("cardTitle")
        guide_layout.addWidget(title)
        self.robot_guide_widget = CalibrationGuideWidget(GUIDE_IMAGE_PATH, mode="robot")
        guide_layout.addWidget(self.robot_guide_widget, 1)
        content.addWidget(guide_card, 3)

        control_card = QFrame()
        control_card.setObjectName("calibrationControlCard")
        control_layout = QVBoxLayout(control_card)
        control_layout.setContentsMargins(20, 18, 20, 18)
        control_layout.setSpacing(11)

        control_title = QLabel("양팔 affine 보정")
        control_title.setObjectName("cardTitle")
        control_layout.addWidget(control_title)

        self.robot_step_label = QLabel(
            "보정 시작 전\nARM2: ID0 → ID1 → 빨간 스티커\nARM1: ID2 → ID3 → 파란 스티커"
        )
        self.robot_step_label.setObjectName("calibrationMainStatus")
        self.robot_step_label.setWordWrap(True)
        self.robot_step_label.setMinimumHeight(102)
        control_layout.addWidget(self.robot_step_label)

        self.robot_progress = QProgressBar()
        self.robot_progress.setObjectName("robotCalibrationProgress")
        self.robot_progress.setRange(0, 6)
        self.robot_progress.setValue(0)
        self.robot_progress.setFormat("0 / 6")
        self.robot_progress.setTextVisible(True)
        self.robot_progress.setMinimumHeight(34)
        control_layout.addWidget(self.robot_progress)

        self.robot_start_button = QPushButton("보정 시작\n두 로봇팔 연결 후 6점 기록")
        self.robot_start_button.setObjectName("calibrationStartButton")
        self.robot_start_button.setMinimumHeight(82)
        control_layout.addWidget(self.robot_start_button)

        self.robot_capture_button = QPushButton("현재 위치 기록  (Enter 동작)")
        self.robot_capture_button.setObjectName("calibrationCaptureButton")
        self.robot_capture_button.setMinimumHeight(64)
        self.robot_capture_button.setEnabled(False)
        control_layout.addWidget(self.robot_capture_button)

        row = QHBoxLayout()
        row.setSpacing(10)
        self.robot_undo_button = QPushButton("한 단계 취소")
        self.robot_undo_button.setObjectName("calibrationAuxButton")
        self.robot_stop_button = QPushButton("보정 중지")
        self.robot_stop_button.setObjectName("calibrationStopButton")
        self.robot_undo_button.setMinimumHeight(54)
        self.robot_stop_button.setMinimumHeight(54)
        self.robot_undo_button.setEnabled(False)
        self.robot_stop_button.setEnabled(False)
        row.addWidget(self.robot_undo_button)
        row.addWidget(self.robot_stop_button)
        control_layout.addLayout(row)

        log_title = QLabel("보정 로그")
        log_title.setObjectName("logTitle")
        control_layout.addWidget(log_title)
        self.robot_log_view = QPlainTextEdit()
        self.robot_log_view.setObjectName("calibrationLogView")
        self.robot_log_view.setReadOnly(True)
        self.robot_log_view.setMaximumBlockCount(140)
        control_layout.addWidget(self.robot_log_view, 1)

        warning = QLabel(
            "활성 팔의 토크가 해제되면 손으로 천천히 움직이세요. 그리퍼 끝점을 표시된 점의 정확한 중심에 놓은 뒤 "
            "'현재 위치 기록'을 누릅니다. 비활성 팔의 토크는 유지됩니다."
        )
        warning.setObjectName("calibrationWarningLabel")
        warning.setWordWrap(True)
        warning.setMinimumHeight(90)
        control_layout.addWidget(warning)

        content.addWidget(control_card, 2)
        root.addLayout(content, 1)
        return page

    def _replace_main_camera_label_with_fit_view(self) -> None:
        old_label = self.camera_label
        parent = old_label.parentWidget()
        layout = parent.layout() if parent is not None else None
        if layout is None:
            raise RuntimeError("cameraLabel parent layout을 찾을 수 없음")

        view = FitVideoWidget("카메라 연결 중...", parent)
        view.setObjectName("cameraLabel")
        view.setMinimumHeight(max(300, old_label.minimumHeight()))
        view.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.replaceWidget(old_label, view)
        old_label.hide()
        old_label.deleteLater()
        self.camera_label = view

    def _connect_main_page(self) -> None:
        super()._connect_main_page()
        self._replace_main_camera_label_with_fit_view()

        self.camera_test_entry_button.clicked.connect(self._enter_camera_test)
        self.camera_test_back_button.clicked.connect(self._leave_camera_test)
        self.camera_continuous_auto_button.clicked.connect(self._set_continuous_camera_auto)
        self.camera_autofit_button.clicked.connect(self._start_camera_autofit)
        self.camera_manual_button.clicked.connect(self._set_manual_camera_mode)
        self.camera_refresh_button.clicked.connect(
            lambda: self._refresh_camera_control_values(show_error=True)
        )
        self.camera_save_button.clicked.connect(self._save_current_camera_controls)
        self.camera_defaults_button.clicked.connect(self._restore_camera_defaults)
        self.camera_exposure_slider.valueChanged.connect(self._on_exposure_slider_changed)
        self.camera_exposure_slider.sliderReleased.connect(self._apply_exposure_value)
        self.camera_exposure_spin.valueChanged.connect(self._on_exposure_spin_changed)
        self.camera_exposure_spin.editingFinished.connect(self._apply_exposure_value)
        self.camera_gain_slider.valueChanged.connect(self._on_gain_slider_changed)
        self.camera_gain_slider.sliderReleased.connect(self._apply_gain_value)
        self.camera_gain_spin.valueChanged.connect(self._on_gain_spin_changed)
        self.camera_gain_spin.editingFinished.connect(self._apply_gain_value)
        self.camera_wb_slider.valueChanged.connect(self._on_wb_slider_changed)
        self.camera_wb_slider.sliderReleased.connect(self._apply_wb_value)
        self.camera_wb_spin.valueChanged.connect(self._on_wb_spin_changed)
        self.camera_wb_spin.editingFinished.connect(self._apply_wb_value)
        self.camera_flicker_combo.currentIndexChanged.connect(self._on_flicker_changed)
        self.camera_awb_checkbox.toggled.connect(self._on_awb_toggled)

        self.aruco_entry_button.clicked.connect(
            lambda: self.page_stack.setCurrentWidget(self.aruco_page)
        )
        self.robot_calib_entry_button.clicked.connect(
            lambda: self.page_stack.setCurrentWidget(self.robot_calib_page)
        )
        self.aruco_back_button.clicked.connect(self._leave_aruco_page)
        self.robot_calib_back_button.clicked.connect(self._leave_robot_calibration_page)
        self.aruco_start_button.clicked.connect(self._request_aruco_calibration)
        self.robot_start_button.clicked.connect(self._request_robot_calibration)
        self.robot_capture_button.clicked.connect(self._capture_robot_calibration_point)
        self.robot_undo_button.clicked.connect(self._undo_robot_calibration_point)
        self.robot_stop_button.clicked.connect(
            lambda: self._stop_robot_calibration("사용자 중지")
        )

    def _setup_camera(self) -> None:
        self.camera_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.camera_label.setText("카메라 연결 중...")

        source = int(self.args.camera) if str(self.args.camera).isdigit() else self.args.camera
        cap = cv2.VideoCapture(source, cv2.CAP_V4L2)
        if not cap.isOpened():
            cap.release()
            self.cap = None
            self.connection_label.setText("카메라 연결 실패")
            self.camera_live_badge.setText("● OFF")
            self.camera_label.setText(f"카메라를 열 수 없음: {self.args.camera}")
            return

        # Match d55_v13 exactly: YUYV is selected before the 1280x720 mode.
        # MJPG 1280x720 used a narrower sensor crop on this ELP camera.
        fourcc_token = str(self.args.camera_fourcc or "").strip().upper()
        if fourcc_token and fourcc_token.lower() not in {"auto", "none"}:
            if len(fourcc_token) != 4:
                cap.release()
                raise ValueError("--camera-fourcc는 정확히 4글자여야 합니다.")
            cap.set(
                cv2.CAP_PROP_FOURCC,
                cv2.VideoWriter_fourcc(*fourcc_token),
            )
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(self.args.width))
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(self.args.height))
        cap.set(cv2.CAP_PROP_FPS, float(self.args.fps))
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        # V4L2 controls can be changed while the stream is open.  Discover the
        # actual OV2710 controls and apply the user's saved values before the
        # first displayed frame.
        self._discover_camera_controls()
        self._apply_saved_camera_controls()
        self._refresh_camera_control_values(show_error=False)

        probe_ok, probe_raw = cap.read()
        if not probe_ok or probe_raw is None:
            cap.release()
            self.cap = None
            self.connection_label.setText("카메라 프레임 실패")
            self.camera_live_badge.setText("● OFF")
            self.camera_label.setText("카메라는 열렸지만 첫 프레임을 받지 못했습니다.")
            return

        actual_size = (int(probe_raw.shape[1]), int(probe_raw.shape[0]))
        self.cap = cap
        self.latest_raw_camera_frame = probe_raw.copy()

        if bool(self.args.camera_undistort):
            try:
                if CameraUndistorter is None:
                    raise RuntimeError(
                        f"camera_undistort.py import 실패: {CAMERA_UNDISTORT_IMPORT_ERROR}"
                    )
                self.camera_undistorter = CameraUndistorter(
                    self.args.camera_calibration,
                    alpha=float(self.args.camera_undistort_alpha),
                    strict_size=bool(self.args.camera_undistort_strict_size),
                )
                info = self.camera_undistorter.prepare(actual_size)
                self.camera_geometry = info.to_metadata()
                self.camera_geometry["camera_model"] = "ELP_OV2710"
                corrected = self.camera_undistorter.correct(probe_raw)
                self.camera_label.set_processing_label("왜곡 보정 ON")
                self.camera_test_view.set_processing_label("왜곡 보정 ON · 카메라 점검")
                self.aruco_camera_label.set_processing_label("왜곡 보정 ON · ArUco")
                self._append_log("왜곡 보정 준비 완료: " + self.camera_undistorter.status_line())
            except Exception as exc:
                self.camera_undistorter = None
                self.camera_undistort_error = repr(exc)
                corrected = probe_raw.copy()
                self.camera_geometry = {
                    "camera_model": "ELP_OV2710",
                    "undistort_enabled": False,
                    "calibration_id": "PREPARE_FAILED",
                    "input_size": list(actual_size),
                    "output_size": list(actual_size),
                    "alpha": float(self.args.camera_undistort_alpha),
                    "error": self.camera_undistort_error,
                }
                self.camera_label.set_processing_label("왜곡 보정 실패 · RAW")
                self.camera_test_view.set_processing_label("왜곡 보정 실패 · RAW")
                self.aruco_camera_label.set_processing_label("보정 실패 · 저장 차단")
                self._append_log("왜곡 보정 실패: " + self.camera_undistort_error)
        else:
            corrected = probe_raw.copy()
            self.camera_geometry = {
                "camera_model": "ELP_OV2710",
                "undistort_enabled": False,
                "calibration_id": "DISABLED",
                "input_size": list(actual_size),
                "output_size": list(actual_size),
                "alpha": 0.0,
            }
            self.camera_label.set_processing_label("왜곡 보정 OFF · RAW")
            self.camera_test_view.set_processing_label("왜곡 보정 OFF · RAW")
            self.aruco_camera_label.set_processing_label("RAW ArUco")

        self.latest_camera_frame = corrected.copy()
        self.camera_label.set_bgr_frame(corrected)
        self.camera_test_view.set_bgr_frame(corrected)
        self.connection_label.setText("카메라 연결됨")
        self.camera_live_badge.setText("● LIVE")

        actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = float(cap.get(cv2.CAP_PROP_FPS))
        fourcc_value = int(cap.get(cv2.CAP_PROP_FOURCC))
        fourcc = "".join(chr((fourcc_value >> (8 * i)) & 0xFF) for i in range(4))
        self._append_log(
            f"카메라 실제 모드: {actual_width}x{actual_height} {fourcc!r} "
            f"{actual_fps:.1f}fps · 요청={self.args.width}x{self.args.height} "
            f"FOURCC={self.args.camera_fourcc!r}"
        )

        self.camera_timer = QTimer(self.window)
        self.camera_timer.setInterval(max(1, round(1000 / max(float(self.args.fps), 1.0))))
        self.camera_timer.timeout.connect(self._update_camera)
        self.camera_timer.start()

    def _handle_escape(self) -> None:
        current = self.page_stack.currentWidget()
        if current is self.camera_test_page:
            self._leave_camera_test()
        elif current is self.robot_test_page:
            self._leave_robot_test_page()
        elif current is self.aruco_page:
            self._leave_aruco_page()
        elif current is self.robot_calib_page:
            self._leave_robot_calibration_page()
        elif current is self.plate_test_page:
            self._leave_plate_test()
        elif current is self.settings_page:
            self.page_stack.setCurrentIndex(0)
        else:
            self.window.close()

    def _leave_aruco_page(self) -> None:
        self.aruco_scanning = False
        self.page_stack.setCurrentWidget(self.settings_page)

    def _leave_robot_calibration_page(self) -> None:
        if self.robot_calibration_active:
            answer = QMessageBox.question(
                self.window,
                "로봇팔 보정 중",
                "현재 보정을 중지하고 설정 화면으로 돌아가시겠습니까?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if answer != QMessageBox.StandardButton.Yes:
                return
            self._stop_robot_calibration("화면 나가기")
        self.page_stack.setCurrentWidget(self.settings_page)

    def _update_camera(self) -> None:
        # MAIN-LATENCY-DIAG-V1: diagnostic-only stage timings.  No camera/UI
        # behavior is changed; the integration wrapper reads this snapshot after
        # super()._update_camera() returns.
        tick_start = time.monotonic()
        timing = {
            "read_s": 0.0,
            "raw_copy_s": 0.0,
            "undistort_s": 0.0,
            "corrected_copy_s": 0.0,
            "ui_s": 0.0,
            "aruco_s": 0.0,
            "total_s": 0.0,
            "ok": False,
        }
        self._main_latency_camera_timing = timing
        stage = None
        try:
            if self.cap is None or not self.cap.isOpened():
                return

            t = time.monotonic()
            stage = "read_s"
            ok, raw_frame = self.cap.read()
            timing["read_s"] = max(0.0, time.monotonic() - t)
            if not ok or raw_frame is None:
                self.camera_label.setText("카메라 프레임 수신 실패")
                self.camera_live_badge.setText("● OFF")
                if self.aruco_started:
                    self.aruco_camera_label.setText("카메라 프레임 수신 실패")
                return

            t = time.monotonic()
            stage = "raw_copy_s"
            self.latest_raw_camera_frame = raw_frame.copy()
            corrected_frame = raw_frame.copy()
            timing["raw_copy_s"] = max(0.0, time.monotonic() - t)

            publish_raw = getattr(self, "_publish_main_raw_frame_early", None)
            if callable(publish_raw):
                try:
                    publish_raw()
                except Exception as exc:
                    timing["publish_error"] = type(exc).__name__
                    try:
                        print(f"[MAIN-EARLY-RAW-WARN] publish_state={timing.get('publish_state', 'unknown_on_error')} error={timing['publish_error']}", flush=True)
                    except Exception:
                        pass

            t = time.monotonic()
            stage = "undistort_s"
            if self.camera_undistorter is not None:
                try:
                    corrected_frame = self.camera_undistorter.correct(raw_frame)
                except Exception as exc:
                    # Never continue collecting a raw-frame Homography under a
                    # misleading "corrected" state after remap failure.
                    self.camera_undistort_error = repr(exc)
                    self.camera_undistorter = None
                    self.camera_geometry = {
                        "camera_model": "ELP_OV2710",
                        "undistort_enabled": False,
                        "calibration_id": "RUNTIME_REMAP_FAILED",
                        "input_size": [int(raw_frame.shape[1]), int(raw_frame.shape[0])],
                        "output_size": [int(raw_frame.shape[1]), int(raw_frame.shape[0])],
                        "alpha": float(self.args.camera_undistort_alpha),
                        "error": self.camera_undistort_error,
                    }
                    self.aruco_scanning = False
                    self.camera_label.set_processing_label("왜곡 보정 실패 · RAW")
                    self.camera_test_view.set_processing_label("왜곡 보정 실패 · RAW")
                    self.aruco_camera_label.set_processing_label("보정 실패 · 저장 차단")
                    self._append_log("실시간 왜곡 보정 실패: " + self.camera_undistort_error)
            timing["undistort_s"] = max(0.0, time.monotonic() - t)

            t = time.monotonic()
            stage = "corrected_copy_s"
            self.latest_camera_frame = corrected_frame.copy()
            timing["corrected_copy_s"] = max(0.0, time.monotonic() - t)

            t = time.monotonic()
            stage = "ui_s"
            self.camera_label.set_bgr_frame(corrected_frame)
            self.camera_test_view.set_bgr_frame(corrected_frame)
            self.connection_label.setText("카메라 연결됨")
            self.camera_live_badge.setText("● LIVE")
            timing["ui_s"] = max(0.0, time.monotonic() - t)

            if self.aruco_started:
                t = time.monotonic()
                stage = "aruco_s"
                annotated = self._process_aruco_frame(raw_frame, corrected_frame)
                self.aruco_camera_label.set_bgr_frame(annotated)
                timing["aruco_s"] = max(0.0, time.monotonic() - t)

            timing["ok"] = True
        except Exception as exc:
            timing["partial"] = True
            timing["error_stage"] = stage or "camera_check"
            timing["error_type"] = type(exc).__name__
            if stage is not None:
                timing[stage] = max(0.0, time.monotonic() - t)
            raise
        finally:
            timing["total_s"] = max(0.0, time.monotonic() - tick_start)

    def _append_aruco_log(self, message: str) -> None:
        timestamp = time.strftime("%H:%M:%S")
        self.aruco_log_view.appendPlainText(f"[{timestamp}] {message}")
        bar = self.aruco_log_view.verticalScrollBar()
        bar.setValue(bar.maximum())

    def _append_robot_log(self, message: str) -> None:
        timestamp = time.strftime("%H:%M:%S")
        self.robot_log_view.appendPlainText(f"[{timestamp}] {message}")
        bar = self.robot_log_view.verticalScrollBar()
        bar.setValue(bar.maximum())

    def _request_aruco_calibration(self) -> None:
        answer = QMessageBox.question(
            self.window,
            "AruCo Marker 보정 시작",
            "기존 Homography Lock을 메모리에서 초기화하고 새로 계산합니다.\n\n"
            "카메라와 폴딩 보드를 고정하고 ID0, ID1, ID2, ID3이 모두 보이게 해주세요.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if answer != QMessageBox.StandardButton.Yes:
            return

        if self.cap is None or not self.cap.isOpened():
            QMessageBox.warning(self.window, "카메라 오류", "카메라가 연결되어 있지 않습니다.")
            return

        if bool(self.args.camera_undistort) and self.camera_undistorter is None:
            QMessageBox.critical(
                self.window,
                "왜곡 보정 준비 실패",
                "보정된 스냅샷으로 Homography를 만들 수 없어 저장을 차단했습니다.\n\n"
                + (self.camera_undistort_error or "camera calibration unavailable"),
            )
            return

        # Same state reset as the uploaded runtime's R key.
        self.aruco_latest_H = None
        self.aruco_latest_raw_H = None
        self._aruco_corrected_samples.clear()
        self._aruco_raw_samples.clear()
        self.aruco_locked = False
        self.aruco_started = True
        self.aruco_scanning = True
        self._last_missing_ids = None
        self.aruco_display_stack.setCurrentWidget(self.aruco_camera_label)
        self.aruco_header_badge.setText("마커 탐색 중")
        self.aruco_status_label.setText(
            f"Homography reset 완료 · 4개 마커를 {self.aruco_sample_target}프레임 "
            "연속 확인한 뒤 보정 H와 raw_H를 저장합니다."
        )
        self.aruco_detection_label.setText("감지 상태: 프레임 대기 중")
        self.aruco_start_button.setText("보정 다시 시작\nHomography Reset 후 재탐색")
        self._append_aruco_log("[H] reset · 새 Homography 탐색 시작")

    @staticmethod
    def _median_marker_centers(
        samples: Sequence[Dict[int, np.ndarray]],
        required_ids: Sequence[int],
    ) -> Dict[int, np.ndarray]:
        result: Dict[int, np.ndarray] = {}
        for marker_id in required_ids:
            points = np.asarray(
                [sample[int(marker_id)] for sample in samples], dtype=np.float32
            )
            result[int(marker_id)] = np.median(points, axis=0).astype(np.float32)
        return result

    @staticmethod
    def _marker_center_max_shift(
        previous: Dict[int, np.ndarray],
        current: Dict[int, np.ndarray],
        required_ids: Sequence[int],
    ) -> float:
        shifts = [
            float(np.linalg.norm(
                np.asarray(current[int(marker_id)], dtype=np.float32)
                - np.asarray(previous[int(marker_id)], dtype=np.float32)
            ))
            for marker_id in required_ids
        ]
        return max(shifts, default=0.0)

    def _process_aruco_frame(
        self, raw_frame: np.ndarray, corrected_frame: np.ndarray
    ) -> np.ndarray:
        canvas = corrected_frame.copy()
        try:
            corrected_corners, corrected_ids, _ = detect_markers(
                corrected_frame, self.aruco_detector
            )
            raw_corners, raw_ids, _ = detect_markers(raw_frame, self.aruco_detector)
        except Exception as exc:
            self.aruco_status_label.setText(f"ArUco 감지 오류: {exc!r}")
            return canvas

        if corrected_ids is not None:
            cv2.aruco.drawDetectedMarkers(canvas, corrected_corners, corrected_ids)

        corrected_H_live, corrected_centers = compute_homography(
            corrected_corners, corrected_ids,
            self.machine_cfg["aruco"]["marker_board_mm"],
            self.required_marker_ids,
        )
        raw_H_live, raw_centers = compute_homography(
            raw_corners, raw_ids,
            self.machine_cfg["aruco"]["marker_board_mm"],
            self.required_marker_ids,
        )
        detected = sorted(int(marker_id) for marker_id in corrected_centers)
        missing = tuple(
            marker_id for marker_id in self.required_marker_ids
            if marker_id not in corrected_centers or marker_id not in raw_centers
        )
        sample_count = len(self._aruco_corrected_samples)
        self.aruco_detection_label.setText(
            "보정 영상 감지: "
            + (", ".join(f"ID{marker_id}" for marker_id in detected) if detected else "없음")
            + "\n보정/원본 공통 미감지: "
            + (", ".join(f"ID{marker_id}" for marker_id in missing) if missing else "없음")
            + f"\n안정 스냅샷: {sample_count} / {self.aruco_sample_target}"
        )

        if missing != self._last_missing_ids:
            self._last_missing_ids = missing
            if missing:
                self._append_aruco_log(
                    "미감지 마커: " + ", ".join(f"ID{marker_id}" for marker_id in missing)
                )
            else:
                self._append_aruco_log("보정/원본 영상에서 ID0~ID3 모두 감지됨")

        all_ready = not missing and corrected_H_live is not None and raw_H_live is not None
        if self.aruco_scanning and all_ready:
            corrected_sample = {
                key: value.copy() for key, value in corrected_centers.items()
            }
            raw_sample = {key: value.copy() for key, value in raw_centers.items()}

            max_shift = 0.0
            if self._aruco_corrected_samples and self._aruco_raw_samples:
                max_shift = max(
                    self._marker_center_max_shift(
                        self._aruco_corrected_samples[-1], corrected_sample,
                        self.required_marker_ids,
                    ),
                    self._marker_center_max_shift(
                        self._aruco_raw_samples[-1], raw_sample,
                        self.required_marker_ids,
                    ),
                )
            if max_shift > self.aruco_stable_max_shift_px:
                self._aruco_corrected_samples.clear()
                self._aruco_raw_samples.clear()
                now = time.monotonic()
                if now - self._last_aruco_stability_log_at >= 0.7:
                    self._append_aruco_log(
                        f"마커 이동 {max_shift:.2f}px 감지 · 안정 스냅샷 누적 재시작"
                    )
                    self._last_aruco_stability_log_at = now

            self._aruco_corrected_samples.append(corrected_sample)
            self._aruco_raw_samples.append(raw_sample)
            self._aruco_corrected_samples = self._aruco_corrected_samples[-self.aruco_sample_target:]
            self._aruco_raw_samples = self._aruco_raw_samples[-self.aruco_sample_target:]
            sample_count = len(self._aruco_corrected_samples)
            self.aruco_status_label.setText(
                f"마커 안정화 중 · {sample_count}/{self.aruco_sample_target} 프레임"
            )
            if sample_count >= self.aruco_sample_target:
                corrected_median = self._median_marker_centers(
                    self._aruco_corrected_samples, self.required_marker_ids
                )
                raw_median = self._median_marker_centers(
                    self._aruco_raw_samples, self.required_marker_ids
                )
                H = compute_homography_from_centers(
                    corrected_median, self.machine_cfg["aruco"]["marker_board_mm"],
                    self.required_marker_ids,
                )
                raw_H = compute_homography_from_centers(
                    raw_median, self.machine_cfg["aruco"]["marker_board_mm"],
                    self.required_marker_ids,
                )
                if H is not None and raw_H is not None:
                    try:
                        save_homography_bundle(
                            self.hfile_path, H, raw_H, self.camera_geometry
                        )
                    except Exception as exc:
                        self.aruco_header_badge.setText("저장 실패")
                        self.aruco_status_label.setText(
                            f"H/raw_H 계산 성공, 저장 실패: {exc!r}"
                        )
                        self._append_aruco_log(f"[H] save failed: {exc!r}")
                    else:
                        self.aruco_latest_H = H.copy()
                        self.aruco_latest_raw_H = raw_H.copy()
                        self.aruco_locked = True
                        self.aruco_scanning = False
                        self.aruco_header_badge.setText("H + raw_H LOCKED")
                        self.aruco_status_label.setText(
                            "보정 완료 · 왜곡 보정 영상 기준 H와 원본 기준 raw_H를 "
                            "함께 저장했습니다."
                        )
                        self._append_aruco_log(
                            f"[H] corrected/raw bundle saved: {self.hfile_path}"
                        )
        elif self.aruco_scanning and not all_ready:
            if self._aruco_corrected_samples or self._aruco_raw_samples:
                self._append_aruco_log("마커 누락으로 안정 스냅샷 누적을 초기화함")
            self._aruco_corrected_samples.clear()
            self._aruco_raw_samples.clear()

        if self.aruco_locked:
            text = "H + raw_H LOCKED"
            color = (0, 255, 0)
        elif all_ready:
            text = f"Stable snapshots {len(self._aruco_corrected_samples)}/{self.aruco_sample_target}"
            color = (0, 255, 255)
        else:
            text = "Need IDs " + ",".join(str(value) for value in missing)
            color = (0, 0, 255)
        cv2.putText(canvas, text, (24, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 0, 0), 5)
        cv2.putText(canvas, text, (24, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.85, color, 2)
        return canvas

    def _request_robot_calibration(self) -> None:
        if self.robot_calibration_active or self._robot_capture_busy:
            QMessageBox.information(self.window, "보정 진행 중", "이미 로봇팔 보정이 진행 중입니다.")
            return
        if not self.args.roarm_send:
            QMessageBox.warning(
                self.window,
                "실제 로봇팔 연결 꺼짐",
                "현재 RoArm 실제 전송이 비활성화되어 있습니다. --no-roarm-send 없이 다시 실행하세요.",
            )
            return

        answer = QMessageBox.question(
            self.window,
            "로봇팔 보정 시작",
            "활성 로봇팔의 토크가 단계별로 해제됩니다.\n\n"
            "로봇팔을 손으로 천천히 움직이고, 다른 팔과 폴딩 보드에 충돌하지 않도록 확인하세요.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if answer != QMessageBox.StandardButton.Yes:
            return

        # Release any serial handles left from a completed/aborted previous session.
        self._close_roarms()
        self.robot_start_button.setEnabled(False)
        self.robot_header_badge.setText("로봇팔 연결 중")
        self.robot_step_label.setText("/dev/roarm_1, /dev/roarm_2 연결 중...")
        self._append_robot_log("양팔 serial 연결 시작")

        def worker() -> None:
            arms: Dict[str, UiRoArmSerial] = {}
            try:
                for arm_key, cli_port in (
                    ("arm1", self.args.arm1_port),
                    ("arm2", self.args.arm2_port),
                ):
                    arm_cfg = self.machine_cfg["dual_roarm"][arm_key]
                    port = str(cli_port or arm_cfg.get("port"))
                    baud = int(self.args.roarm_baud or arm_cfg.get("baudrate", 115200))
                    arms[arm_key] = UiRoArmSerial(port, baud, label=arm_key)
                self.calibration_signals.robot_connected.emit(True, arms, "양팔 연결 성공")
            except Exception as exc:
                for arm in arms.values():
                    arm.close()
                self.calibration_signals.robot_connected.emit(False, {}, repr(exc))

        threading.Thread(target=worker, name="ui-roarm-calibration-connect", daemon=True).start()

    def _on_robot_connected(self, ok: bool, arms_obj: object, detail: str) -> None:
        if not ok:
            self.robot_start_button.setEnabled(True)
            self.robot_header_badge.setText("연결 실패")
            self.robot_step_label.setText(f"로봇팔 연결 실패\n{detail}")
            self._append_robot_log(f"연결 실패: {detail}")
            return

        self._close_roarms()
        self.roarms = dict(arms_obj) if isinstance(arms_obj, dict) else {}
        self.robot_calibration_active = True
        self.robot_calib_index = 0
        self.temp_roarm_xy = {"arm1": [], "arm2": []}
        self.temp_roarm_z = {"arm1": [], "arm2": []}
        self.robot_calib_sequence = []
        # Same order as the uploaded runtime's C flow: left side/ARM2 first.
        for arm_key in ("arm2", "arm1"):
            for point in self.machine_cfg["dual_roarm"][arm_key]["calib_points"]:
                self.robot_calib_sequence.append((arm_key, copy.deepcopy(point)))

        self.robot_header_badge.setText("보정 진행 중")
        self.robot_capture_button.setEnabled(True)
        self.robot_undo_button.setEnabled(False)
        self.robot_stop_button.setEnabled(True)
        self.robot_progress.setValue(0)
        self.robot_progress.setFormat("0 / 6")
        self._append_robot_log("DUAL-ARM calibration 시작")
        self._append_robot_log("ARM2: ID0 → ID1 → RED_EXTRA")
        self._append_robot_log("ARM1: ID2 → ID3 → BLUE_EXTRA")
        self._show_robot_calibration_step()

    def _show_robot_calibration_step(self) -> None:
        if not self.robot_calibration_active:
            return
        if self.robot_calib_index >= len(self.robot_calib_sequence):
            self._finish_robot_calibration()
            return

        arm_key, point = self.robot_calib_sequence[self.robot_calib_index]
        label = str(point["label"])
        board_xy = point["board_xy"]
        self.robot_guide_widget.set_current_target(label, arm_key)
        self.robot_step_label.setText(
            f"STEP {self.robot_calib_index + 1} / {len(self.robot_calib_sequence)}\n"
            f"{arm_key.upper()} 그리퍼 끝점을 {label} 중심에 놓으세요.\n"
            f"보드 좌표: ({float(board_xy[0]):.1f}, {float(board_xy[1]):.1f})"
        )
        self.robot_progress.setValue(self.robot_calib_index)
        self.robot_progress.setFormat(
            f"{self.robot_calib_index} / {len(self.robot_calib_sequence)}"
        )
        self.robot_undo_button.setEnabled(self.robot_calib_index > 0)

        try:
            for other_key, other_arm in self.roarms.items():
                if other_key != arm_key:
                    other_arm.torque_on()
            active_arm = self.roarms.get(arm_key)
            if active_arm is None:
                raise RuntimeError(f"{arm_key} serial 없음")
            active_arm.torque_off()
        except Exception as exc:
            self._append_robot_log(f"토크 전환 오류: {exc!r}")
            self.robot_step_label.setText(self.robot_step_label.text() + f"\n토크 전환 오류: {exc!r}")
            return

        self._append_robot_log(
            f"STEP {self.robot_calib_index + 1}: {arm_key.upper()} → {label} · active torque OFF"
        )

    def _capture_robot_calibration_point(self) -> None:
        if not self.robot_calibration_active or self._robot_capture_busy:
            return
        if self.robot_calib_index >= len(self.robot_calib_sequence):
            return
        arm_key, point = self.robot_calib_sequence[self.robot_calib_index]
        arm = self.roarms.get(arm_key)
        if arm is None:
            QMessageBox.warning(self.window, "기록 실패", f"{arm_key} serial이 연결되지 않았습니다.")
            return

        self._robot_capture_busy = True
        self.robot_capture_button.setEnabled(False)
        self.robot_step_label.setText(
            self.robot_step_label.text().split("\n좌표 요청 중")[0] + "\n좌표 요청 중..."
        )
        self._append_robot_log(f"{arm_key.upper()} {point['label']} T:105 feedback 요청")

        def worker() -> None:
            try:
                feedback = arm.feedback(timeout=float(self.args.roarm_feedback_timeout_s))
                if feedback is None or "x" not in feedback or "y" not in feedback:
                    raise RuntimeError("T:105 feedback 실패")
                self.calibration_signals.robot_capture_finished.emit(
                    True,
                    feedback,
                    f"{arm_key}:{point['label']}",
                )
            except Exception as exc:
                self.calibration_signals.robot_capture_finished.emit(
                    False,
                    {},
                    repr(exc),
                )

        threading.Thread(target=worker, name="ui-roarm-feedback", daemon=True).start()

    def _on_robot_capture_finished(self, ok: bool, payload_obj: object, detail: str) -> None:
        self._robot_capture_busy = False
        if not self.robot_calibration_active:
            return
        self.robot_capture_button.setEnabled(True)
        if not ok:
            self._append_robot_log(f"좌표 기록 실패: {detail}")
            QMessageBox.warning(
                self.window,
                "좌표 기록 실패",
                "로봇팔 피드백을 받지 못했습니다. serial 연결을 확인하고 다시 눌러주세요.\n\n" + detail,
            )
            self._show_robot_calibration_step()
            return

        feedback = dict(payload_obj) if isinstance(payload_obj, dict) else {}
        arm_key, point = self.robot_calib_sequence[self.robot_calib_index]
        x = float(feedback["x"])
        y = float(feedback["y"])
        z = float(feedback.get("z", float("nan")))
        self.temp_roarm_xy[arm_key].append([x, y])
        self.temp_roarm_z[arm_key].append(z)
        self._append_robot_log(
            f"저장: {arm_key.upper()} {point['label']} · roarm=({x:.2f}, {y:.2f}, {z:.2f})"
        )
        self.robot_calib_index += 1
        self.robot_progress.setValue(self.robot_calib_index)
        self.robot_progress.setFormat(
            f"{self.robot_calib_index} / {len(self.robot_calib_sequence)}"
        )
        if self.robot_calib_index >= len(self.robot_calib_sequence):
            self._finish_robot_calibration()
        else:
            self._show_robot_calibration_step()

    def _undo_robot_calibration_point(self) -> None:
        if not self.robot_calibration_active or self._robot_capture_busy:
            return
        if self.robot_calib_index <= 0:
            self._append_robot_log("취소할 보정점이 없음")
            return
        self.robot_calib_index -= 1
        arm_key, point = self.robot_calib_sequence[self.robot_calib_index]
        if self.temp_roarm_xy[arm_key]:
            self.temp_roarm_xy[arm_key].pop()
        if self.temp_roarm_z[arm_key]:
            self.temp_roarm_z[arm_key].pop()
        self._append_robot_log(f"한 단계 취소: {arm_key.upper()} {point['label']}")
        self._show_robot_calibration_step()

    def _finish_robot_calibration(self) -> None:
        try:
            result_lines: List[str] = []
            for arm_key in ("arm2", "arm1"):
                points = self.machine_cfg["dual_roarm"][arm_key]["calib_points"]
                board_points = np.asarray([point["board_xy"] for point in points], dtype=float)
                roarm_points = np.asarray(self.temp_roarm_xy[arm_key], dtype=float)
                if len(roarm_points) != len(board_points):
                    raise RuntimeError(f"{arm_key}: 기록점 부족")
                M, predicted, error = compute_affine(board_points, roarm_points)
                arm_cfg = self.machine_cfg["dual_roarm"][arm_key]
                arm_cfg["calib_roarm_points"] = roarm_points.tolist()
                arm_cfg["calib_roarm_z"] = list(self.temp_roarm_z[arm_key])
                arm_cfg["board_to_roarm_affine_2x3"] = M.tolist()
                z_values = np.asarray(self.temp_roarm_z[arm_key], dtype=float)
                finite_z = z_values[np.isfinite(z_values)]
                if finite_z.size:
                    arm_cfg["grip_z"] = float(finite_z.mean())
                result_lines.append(
                    f"{arm_key.upper()}: mean={float(error.mean()):.2f}mm · max={float(error.max()):.2f}mm"
                )
                self._append_robot_log(result_lines[-1])
                for point, measured, pred, err in zip(points, roarm_points, predicted, error):
                    self._append_robot_log(
                        f"  {point['label']}: measured=({measured[0]:.2f},{measured[1]:.2f}) "
                        f"pred=({pred[0]:.2f},{pred[1]:.2f}) err={float(err):.2f}mm"
                    )

            save_config(self.machine_config_path, self.machine_cfg)
            self.calib_cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_payload = {
                "created_at_unix": time.time(),
                "dual_roarm": self.machine_cfg["dual_roarm"],
            }
            tmp = self.calib_cache_path.with_suffix(self.calib_cache_path.suffix + ".tmp")
            tmp.write_text(
                json.dumps(cache_payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            tmp.replace(self.calib_cache_path)

            for arm in self.roarms.values():
                arm.torque_on()

            self.robot_calibration_active = False
            self.robot_header_badge.setText("보정 완료")
            self.robot_step_label.setText(
                "양팔 보정 완료\n"
                + "\n".join(result_lines)
                + f"\n저장: {self.calib_cache_path.name}"
            )
            self.robot_guide_widget.set_current_target(None)
            self.robot_progress.setValue(6)
            self.robot_progress.setFormat("6 / 6 · 완료")
            self.robot_capture_button.setEnabled(False)
            self.robot_undo_button.setEnabled(False)
            self.robot_stop_button.setEnabled(False)
            self.robot_start_button.setEnabled(True)
            self._append_robot_log(f"config 저장: {self.machine_config_path}")
            self._append_robot_log(f"cache 저장: {self.calib_cache_path}")
            self._close_roarms()
            QMessageBox.information(
                self.window,
                "로봇팔 보정 완료",
                "양팔 affine 보정을 완료하고 설정 파일과 캐시를 저장했습니다.\n\n"
                + "\n".join(result_lines),
            )
        except Exception as exc:
            self._append_robot_log(f"보정 계산/저장 실패: {exc!r}")
            QMessageBox.critical(self.window, "보정 실패", repr(exc))
            self._stop_robot_calibration("계산 또는 저장 실패")

    def _stop_robot_calibration(self, reason: str) -> None:
        for arm_key, arm in list(self.roarms.items()):
            try:
                arm.torque_on()
                self._append_robot_log(f"{arm_key.upper()} torque ON")
            except Exception as exc:
                self._append_robot_log(f"{arm_key.upper()} torque ON 실패: {exc!r}")
        self.robot_calibration_active = False
        self._robot_capture_busy = False
        self.robot_header_badge.setText("보정 중지")
        self.robot_step_label.setText(f"보정 중지됨\n사유: {reason}")
        self.robot_guide_widget.set_current_target(None)
        self.robot_capture_button.setEnabled(False)
        self.robot_undo_button.setEnabled(False)
        self.robot_stop_button.setEnabled(False)
        self.robot_start_button.setEnabled(True)
        self._append_robot_log(f"보정 중지: {reason}")
        self._close_roarms()

    def _append_robot_test_log_ui(self, text: str) -> None:
        timestamp = time.strftime("%H:%M:%S")
        line = f"[{timestamp}] {text}"
        if hasattr(self, "robot_test_log"):
            self.robot_test_log.appendPlainText(line)
        print(f"[ROBOT-TEST] {text}")

    def _robot_test_log_message(self, text: str) -> None:
        self.robot_test_signals.log.emit(str(text))

    @staticmethod
    def _robot_ui_name(arm_key: str) -> str:
        return "로봇팔 1 · ARM1" if str(arm_key) == "arm1" else "로봇팔 2 · ARM2"

    def _robot_port(self, arm_key: str) -> str:
        return str(self.args.arm1_port if str(arm_key) == "arm1" else self.args.arm2_port)

    def _enter_robot_test_page(self) -> None:
        if self.robot_calibration_active:
            QMessageBox.warning(self.window, "진입 불가", "로봇팔 보정이 진행 중입니다.")
            return
        self.page_stack.setCurrentWidget(self.robot_test_page)
        self._select_robot_test_arm(self.robot_test_selected_arm)
        self._robot_test_log_message(
            "카메라 시점 좌우와 사용자 시점 좌우가 반대임 · 사용자 1번=ARM1, 2번=ARM2"
        )

    def _leave_robot_test_page(self) -> None:
        if self._robot_test_busy:
            QMessageBox.warning(self.window, "동작 진행 중", "로봇팔 점검 동작이 끝난 뒤 돌아가세요.")
            return
        self._close_robot_test_serial()
        self.page_stack.setCurrentWidget(self.settings_page)

    def _select_robot_test_arm(self, arm_key: str) -> None:
        arm_key = str(arm_key)
        if arm_key not in ("arm1", "arm2"):
            return
        if self._robot_test_busy:
            return
        if self.robot_test_serial_arm and self.robot_test_serial_arm != arm_key:
            self._close_robot_test_serial()
        self.robot_test_selected_arm = arm_key
        self.robot_top_view.set_selected_arm(arm_key)
        self.robot_select_arm1_button.setChecked(arm_key == "arm1")
        self.robot_select_arm2_button.setChecked(arm_key == "arm2")
        name = self._robot_ui_name(arm_key)
        self.robot_test_header_badge.setText(name)
        self.robot_test_status.setText(f"선택됨: {name} · {self._robot_port(arm_key)}")
        self._robot_test_log_message(f"{name} 선택 · 그림에서는 {'오른쪽' if arm_key == 'arm1' else '왼쪽'} 팔")

    def _close_robot_test_serial(self) -> None:
        arm = self.robot_test_serial
        self.robot_test_serial = None
        self.robot_test_serial_arm = ""
        if arm is not None:
            try:
                arm.close()
            except Exception:
                pass

    def _robot_test_get_serial(self, arm_key: Optional[str] = None) -> UiRoArmSerial:
        if not bool(self.args.roarm_send):
            raise RuntimeError("로봇팔 실제 전송이 꺼져 있습니다. --roarm-send로 실행하세요.")
        if serial is None:
            raise RuntimeError("pyserial이 없습니다.")
        key = str(arm_key or self.robot_test_selected_arm)
        if key not in ("arm1", "arm2"):
            raise RuntimeError(f"지원하지 않는 로봇팔: {key}")
        if self.robot_test_serial is not None and self.robot_test_serial_arm == key:
            return self.robot_test_serial
        self._close_robot_test_serial()
        port = self._robot_port(key)
        arm = UiRoArmSerial(port, int(self.args.roarm_baud), label=f"robot-test-{key}")
        self.robot_test_serial = arm
        self.robot_test_serial_arm = key
        self._robot_test_log_message(f"{self._robot_ui_name(key)} 연결 완료 · {port}")
        return arm

    def _robot_test_step_mm(self) -> float:
        text = str(self.robot_step_combo.currentText()).split()[0]
        try:
            return float(text)
        except Exception:
            return 10.0

    def _set_robot_test_controls_enabled(self, enabled: bool) -> None:
        widgets = [
            self.robot_select_arm1_button, self.robot_select_arm2_button,
            self.robot_auto_test_button, self.robot_move_forward, self.robot_move_back,
            self.robot_move_left, self.robot_move_right, self.robot_move_up,
            self.robot_move_down, self.robot_wrist_ccw, self.robot_wrist_cw,
            self.robot_gripper_open, self.robot_gripper_close, self.robot_torque_on,
            self.robot_torque_off, self.robot_feedback_button, self.robot_standby_button,
            self.robot_step_combo,
        ]
        for widget in widgets:
            widget.setEnabled(bool(enabled))

    def _run_robot_test_async(self, label: str, task) -> None:
        if self._robot_test_busy:
            self._robot_test_log_message("다른 로봇팔 점검 동작이 진행 중임")
            return
        self._robot_test_busy = True
        self._set_robot_test_controls_enabled(False)
        self.robot_test_status.setText(f"{self._robot_ui_name(self.robot_test_selected_arm)} · {label} 진행 중")
        self._robot_test_log_message(f"{label} 시작")

        def worker() -> None:
            try:
                result = str(task())
                self.robot_test_signals.action_finished.emit(label, True, result)
            except Exception as exc:
                self.robot_test_signals.action_finished.emit(label, False, repr(exc))

        threading.Thread(target=worker, name=f"robot-test-{label}", daemon=True).start()

    def _on_robot_test_action_finished(self, label: str, success: bool, result: str) -> None:
        self._robot_test_busy = False
        self._set_robot_test_controls_enabled(True)
        state = "완료" if success else "실패"
        self.robot_test_status.setText(
            f"{self._robot_ui_name(self.robot_test_selected_arm)} · {label} {state}"
        )
        self._robot_test_log_message(f"{label} {state} · {result}")
        if not success:
            QMessageBox.critical(self.window, "로봇팔 점검 실패", f"{label}\n\n{result}")

    def _on_robot_test_feedback_ready(self, success: bool, payload: Any, detail: str) -> None:
        _ = success, payload, detail

    def _feedback_with_diagnostics(self, arm_key: str, arm: UiRoArmSerial) -> Dict[str, Any]:
        feedback = arm.feedback(
            timeout=float(self.args.roarm_feedback_timeout_s),
            retries=int(self.args.roarm_feedback_retries),
        )
        if feedback is None:
            raw = arm.last_rx_lines[-8:]
            raise RuntimeError(
                f"{self._robot_ui_name(arm_key)} T:1051 응답 없음 · port={arm.port} · "
                f"마지막 RX={raw if raw else '없음'}"
            )
        x, y, z, t = self._feedback_xyz(feedback)
        self._robot_test_log_message(
            f"{self._robot_ui_name(arm_key)} RX T:1051 · "
            f"x={x:.2f}, y={y:.2f}, z={z:.2f}, t={t:.3f}"
        )
        return feedback

    def _request_robot_feedback(self) -> None:
        arm_key = self.robot_test_selected_arm

        def task() -> str:
            arm = self._robot_test_get_serial(arm_key)
            payload = self._feedback_with_diagnostics(arm_key, arm)
            x, y, z, t = self._feedback_xyz(payload)
            try:
                bx, by = roarm_to_board_xy(self.machine_cfg, arm_key, x, y)
                board_text = f" · board=({bx:.1f}, {by:.1f})"
            except Exception as exc:
                board_text = f" · board 변환 불가({exc})"
            return f"RoArm=({x:.1f}, {y:.1f}, {z:.1f}, t={t:.3f}){board_text}"

        self._run_robot_test_async("현재 좌표 확인", task)

    def _request_robot_torque(self, enabled: bool) -> None:
        arm_key = self.robot_test_selected_arm

        def task() -> str:
            arm = self._robot_test_get_serial(arm_key)
            if enabled:
                arm.torque_on()
            else:
                arm.torque_off()
            self._robot_test_torque_enabled[arm_key] = bool(enabled)
            return "토크 ON" if enabled else "토크 OFF · 수동 이동 명령 차단"

        self._run_robot_test_async("토크 ON" if enabled else "토크 OFF", task)

    def _request_robot_gripper(self, open_gripper: bool) -> None:
        arm_key = self.robot_test_selected_arm

        def task() -> str:
            arm = self._robot_test_get_serial(arm_key)
            if not self._robot_test_torque_enabled.get(arm_key, True):
                raise RuntimeError("토크가 OFF입니다. 토크 ON 후 실행하세요.")
            angle = float(self.args.robot_test_gripper_open if open_gripper
                          else self.args.robot_test_gripper_close)
            arm.set_gripper(angle)
            return f"그리퍼 {'열기' if open_gripper else '닫기'} · {angle:.2f}rad"

        self._run_robot_test_async("그리퍼 열기" if open_gripper else "그리퍼 닫기", task)

    def _validate_robot_board_target(self, arm_key: str, bx: float, by: float) -> None:
        min_x, max_x, min_y, max_y = board_limits(self.machine_cfg)
        margin = float(self.args.robot_test_board_margin_mm)
        split = float(self.machine_cfg.get("dual_roarm", {}).get("split_board_x", 247.0))
        dead = float(self.args.robot_test_dead_half_width_mm)
        if not (min_y + margin <= by <= max_y - margin):
            raise RuntimeError(f"보드 Y 안전 범위 이탈: y={by:.1f}")
        if arm_key == "arm1":
            if not (split + dead <= bx <= max_x - margin):
                raise RuntimeError(f"ARM1 전용 작업 영역 이탈: x={bx:.1f}")
        else:
            if not (min_x + margin <= bx <= split - dead):
                raise RuntimeError(f"ARM2 전용 작업 영역 이탈: x={bx:.1f}")

    def _wait_robot_test_arrival(
        self, arm_key: str, arm: UiRoArmSerial, target: Dict[str, float], stage: str
    ) -> Dict[str, Any]:
        deadline = time.monotonic() + max(1.0, float(self.args.robot_test_arrival_timeout_s))
        consecutive = 0
        required = max(1, int(self.args.robot_test_arrival_consecutive))
        last: Optional[Dict[str, Any]] = None
        while time.monotonic() < deadline:
            feedback = arm.feedback(
                timeout=float(self.args.roarm_feedback_timeout_s),
                retries=int(self.args.roarm_feedback_retries),
            )
            if feedback is None:
                consecutive = 0
                continue
            last = dict(feedback)
            x, y, z, _ = self._feedback_xyz(feedback)
            xy_error = math.hypot(x - target["x"], y - target["y"])
            z_error = abs(z - target["z"])
            if xy_error <= float(self.args.robot_test_xy_tolerance_mm) and z_error <= float(self.args.robot_test_z_tolerance_mm):
                consecutive += 1
                if consecutive >= required:
                    self._robot_test_log_message(
                        f"{stage} 도착 · XY오차={xy_error:.1f}mm · Z오차={z_error:.1f}mm"
                    )
                    return last
            else:
                consecutive = 0
            time.sleep(max(0.04, float(self.args.robot_test_poll_interval_s)))
        raise RuntimeError(f"{stage} 도착 확인 실패 · 마지막 피드백={last} · RX={arm.last_rx_lines[-8:]}")

    def _request_robot_jog(self, direction: str) -> None:
        arm_key = self.robot_test_selected_arm
        step = self._robot_test_step_mm()

        def task() -> str:
            if not self._robot_test_torque_enabled.get(arm_key, True):
                raise RuntimeError("토크가 OFF입니다. 토크 ON 후 실행하세요.")
            arm = self._robot_test_get_serial(arm_key)
            feedback = self._feedback_with_diagnostics(arm_key, arm)
            x, y, z, t = self._feedback_xyz(feedback)
            target = {"x": x, "y": y, "z": z, "t": t}

            if direction in ("up", "down"):
                target["z"] = z + (step if direction == "up" else -step)
                if not (float(self.args.robot_test_min_z) <= target["z"] <= float(self.args.robot_test_max_z)):
                    raise RuntimeError(f"Z 안전 범위 이탈: {target['z']:.1f}")
            elif direction in ("wrist_ccw", "wrist_cw"):
                delta = float(self.args.robot_test_wrist_step_rad)
                target["t"] = t + (delta if direction == "wrist_ccw" else -delta)
            else:
                bx, by = roarm_to_board_xy(self.machine_cfg, arm_key, x, y)
                # User stands at the display/bottom side.  Compared with the camera
                # image, user-left is +board-X and forward is +board-Y.
                if direction == "forward":
                    by += step
                elif direction == "back":
                    by -= step
                elif direction == "left":
                    bx += step
                elif direction == "right":
                    bx -= step
                self._validate_robot_board_target(arm_key, bx, by)
                target["x"], target["y"] = board_to_roarm_xy(self.machine_cfg, arm_key, bx, by)

            arm.move_goal(target["x"], target["y"], target["z"], target["t"],
                          float(self.args.robot_test_manual_speed))
            self._wait_robot_test_arrival(arm_key, arm, target, f"수동 {direction}")
            return (
                f"{direction} {step:.0f}mm · target=({target['x']:.1f}, {target['y']:.1f}, "
                f"{target['z']:.1f}, t={target['t']:.3f})"
            )

        self._run_robot_test_async(f"수동 {direction}", task)

    def _move_one_robot_to_standby(self, arm_key: str, arm: UiRoArmSerial) -> None:
        target = self._plate_standby_targets()[arm_key]
        feedback = self._feedback_with_diagnostics(arm_key, arm)
        x, y, z, t = self._feedback_xyz(feedback)
        clear = {"x": x, "y": y, "z": max(z, float(self.args.plate_standby_clear_z)), "t": t}
        arm.move_goal(clear["x"], clear["y"], clear["z"], clear["t"],
                      float(self.args.plate_standby_speed))
        self._wait_robot_test_arrival(arm_key, arm, clear, "수직 안전 상승")
        arm.set_gripper(float(self.args.plate_standby_gripper_angle))
        # Move horizontally at clear Z first, then descend vertically at standby XY.
        above = {"x": target["x"], "y": target["y"], "z": clear["z"], "t": target["t"]}
        arm.move_goal(above["x"], above["y"], above["z"], above["t"],
                      float(self.args.plate_standby_speed))
        self._wait_robot_test_arrival(arm_key, arm, above, "대기 위치 상부")
        arm.move_goal(target["x"], target["y"], target["z"], target["t"],
                      float(self.args.plate_standby_speed))
        self._wait_robot_test_arrival(arm_key, arm, target, "최종 대기 위치")

    def _request_selected_robot_standby(self) -> None:
        arm_key = self.robot_test_selected_arm

        def task() -> str:
            arm = self._robot_test_get_serial(arm_key)
            self._move_one_robot_to_standby(arm_key, arm)
            return "직접 티칭 대기 위치 도착 확인"

        self._run_robot_test_async("대기 위치 복귀", task)

    def _robot_auto_board_path(self, arm_key: str) -> List[Tuple[float, float]]:
        if arm_key == "arm1":
            return [(325.0, -115.0), (445.0, -115.0), (445.0, -380.0),
                    (325.0, -380.0), (325.0, -115.0)]
        return [(49.0, -115.0), (169.0, -115.0), (169.0, -380.0),
                (49.0, -380.0), (49.0, -115.0)]

    def _request_robot_auto_test(self) -> None:
        arm_key = self.robot_test_selected_arm
        if not self._confirm_motion(
            "로봇팔 자동 점검",
            f"{self._robot_ui_name(arm_key)}을 안전 높이에서 전용 작업 영역 한 바퀴 이동하고 "
            "그리퍼를 열고 닫은 뒤 대기 위치로 복귀하시겠습니까?",
        ):
            return

        def task() -> str:
            # Keep the non-selected arm outside the board before the selected arm
            # traverses its workspace.  Reuse the same verified dual-standby gate
            # used by the folding-board test.
            self._close_robot_test_serial()
            self._robot_test_log_message("자동 점검 전 양팔 안전 대기 위치 확인")
            self._prepare_roarms_for_plate_motion()
            arm = self._robot_test_get_serial(arm_key)
            arm.torque_on()
            self._robot_test_torque_enabled[arm_key] = True
            feedback = self._feedback_with_diagnostics(arm_key, arm)
            x, y, z, t = self._feedback_xyz(feedback)
            clear_z = max(z, float(self.args.robot_test_auto_z))
            clear = {"x": x, "y": y, "z": clear_z, "t": t}
            arm.move_goal(x, y, clear_z, t, float(self.args.robot_test_auto_speed))
            self._wait_robot_test_arrival(arm_key, arm, clear, "자동 점검 안전 상승")

            for index, (bx, by) in enumerate(self._robot_auto_board_path(arm_key), 1):
                self._validate_robot_board_target(arm_key, bx, by)
                rx, ry = board_to_roarm_xy(self.machine_cfg, arm_key, bx, by)
                target = {"x": rx, "y": ry, "z": clear_z,
                          "t": self._plate_standby_targets()[arm_key]["t"]}
                arm.move_goal(target["x"], target["y"], target["z"], target["t"],
                              float(self.args.robot_test_auto_speed))
                self._wait_robot_test_arrival(arm_key, arm, target, f"자동 경로 {index}/5")

            arm.set_gripper(float(self.args.robot_test_gripper_open))
            time.sleep(0.55)
            arm.set_gripper(float(self.args.robot_test_gripper_close))
            time.sleep(0.55)
            arm.set_gripper(float(self.args.plate_standby_gripper_angle))
            time.sleep(0.35)
            self._move_one_robot_to_standby(arm_key, arm)
            return "전용 작업 영역 1회 순회 + 그리퍼 열기/닫기 + 대기 복귀 완료"

        self._run_robot_test_async("자동 점검", task)

    def _enter_plate_test(self) -> None:
        # One explicit standby verification is required each time the operator
        # opens the folding-board page.  After success, all plate actions can be
        # repeated without moving the arms again until this page is left.
        self.plate_roarms_ready = False
        self.plate_board_widget.set_selected_plate(None)
        self.plate_board_widget.set_active_plates([])
        self.selected_plate_label.setText("먼저 로봇팔을 대기 위치로 이동하세요.")
        self.page_stack.setCurrentIndex(2)
        self._refresh_controller_status()

    def _leave_plate_test(self) -> None:
        previous_index = self.page_stack.currentIndex()
        super()._leave_plate_test()
        if previous_index == 2 and self.page_stack.currentIndex() != 2:
            self.plate_roarms_ready = False
            self.plate_board_widget.set_selected_plate(None)
            self.plate_board_widget.set_active_plates([])
            self.selected_plate_label.setText("먼저 로봇팔을 대기 위치로 이동하세요.")
            self._refresh_controller_status()

    def _apply_plate_motion_gate(self, controller_enabled: bool) -> None:
        controller = self.controller
        connection_valid = bool(
            controller is not None
            and controller.connected
            and controller.ready
            and bool(getattr(controller, "protocol_ok", False))
            and controller.send_enabled
        )
        if not connection_valid:
            self.plate_roarms_ready = False

        operation_enabled = bool(controller_enabled and self.plate_roarms_ready)
        self.plate_board_widget.set_motion_enabled(operation_enabled)
        self.auto_test_button.setEnabled(operation_enabled)

        # The standby button remains the only motion entry point before the gate
        # is satisfied.  It can also be pressed again to re-verify the poses.
        self.plate_roarm_standby_button.setEnabled(bool(controller_enabled))
        if self.plate_roarms_ready:
            self.plate_roarm_standby_button.setText(
                "✓ 로봇팔 대기 위치 확인 완료\n다시 이동하려면 클릭"
            )
        else:
            self.plate_roarm_standby_button.setText(
                "로봇팔 대기 위치로 이동\n완료 후 폴딩판 작동이 활성화됩니다"
            )

    def _request_plate_roarm_standby(self) -> None:
        ready, reason = self._controller_ready_for_motion()
        if not ready:
            QMessageBox.warning(self.window, "실행 불가", reason)
            return
        standby_ok, standby_reason = self._plate_standby_precheck()
        if not standby_ok:
            QMessageBox.warning(self.window, "로봇팔 안전 이동 불가", standby_reason)
            return
        if not self._confirm_motion(
            "로봇팔 대기 위치로 이동",
            "양쪽 로봇팔을 안전 높이로 올린 뒤 폴딩판 외측 대기 위치로 이동하시겠습니까?\n\n"
            "도착 확인이 완료되면 이 화면을 나가기 전까지 폴딩판 작동을 반복할 수 있습니다.",
        ):
            return

        self.plate_roarms_ready = False
        self.plate_board_widget.set_motion_enabled(False)
        self.auto_test_button.setEnabled(False)
        self.selected_plate_label.setText("로봇팔 대기 위치 이동 및 도착 확인 중")

        def task() -> str:
            self._prepare_roarms_for_plate_motion()
            return "대기 위치 확인 완료"

        self._run_action_async("로봇팔 대기 위치로 이동", task)

    def _plate_standby_targets(self) -> Dict[str, Dict[str, float]]:
        return {
            "arm1": {
                "x": float(self.args.plate_arm1_standby_x),
                "y": float(self.args.plate_arm1_standby_y),
                "z": float(self.args.plate_arm1_standby_z),
                "t": float(self.args.plate_arm1_standby_t),
            },
            "arm2": {
                "x": float(self.args.plate_arm2_standby_x),
                "y": float(self.args.plate_arm2_standby_y),
                "z": float(self.args.plate_arm2_standby_z),
                "t": float(self.args.plate_arm2_standby_t),
            },
        }

    def _plate_standby_precheck(self) -> Tuple[bool, str]:
        if not bool(self.args.plate_standby_safety):
            return True, "로봇팔 대기 이동 안전 기능이 명시적으로 비활성화됨"
        if not bool(self.args.roarm_send):
            return False, "로봇팔 실제 전송이 꺼져 있어 폴딩판 안전 점검을 수행할 수 없습니다."
        if self.robot_calibration_active or self.roarms:
            return False, "로봇팔 보정이 진행 중이므로 폴딩판 테스트를 실행할 수 없습니다."
        if self._robot_test_busy or self.robot_test_serial is not None:
            return False, "로봇팔 작동 점검 포트가 사용 중입니다. 점검 화면에서 뒤로 이동한 뒤 실행하세요."
        if serial is None:
            return False, "pyserial이 없어 로봇팔 대기 위치를 확인할 수 없습니다."
        return True, "READY"

    @staticmethod
    def _feedback_xyz(payload: Any) -> Tuple[float, float, float, float]:
        if payload is None:
            raise RuntimeError("T:1051 응답을 받지 못했습니다.")
        if not isinstance(payload, dict):
            raise RuntimeError(f"T:1051 JSON object가 아님: {type(payload).__name__}")
        source: Dict[str, Any] = dict(payload)
        nested = payload.get("data")
        if isinstance(nested, dict):
            source.update(nested)
        # X/Y/Z may be uppercase on some firmware builds.  Wrist t must stay
        # case-sensitive because uppercase T is the protocol command ID (1051).
        coords: Dict[str, Any] = {}
        for key in ("x", "y", "z"):
            if key in source:
                coords[key] = source[key]
            elif key.upper() in source:
                coords[key] = source[key.upper()]
        missing = [key for key in ("x", "y", "z") if key not in coords]
        if missing:
            raise RuntimeError(
                f"T:1051 좌표 키 누락={missing} · 수신 keys={sorted(map(str, source.keys()))}"
            )
        wrist_t = source.get("t", source.get("theta", 3.14))
        try:
            values = (
                float(coords["x"]),
                float(coords["y"]),
                float(coords["z"]),
                float(wrist_t),
            )
        except Exception as exc:
            raise RuntimeError(f"T:1051 좌표를 숫자로 변환하지 못함: {source}") from exc
        if not all(math.isfinite(value) for value in values):
            raise RuntimeError(f"T:1051 좌표에 유한하지 않은 값이 있음: {values}")
        return values

    def _wait_plate_roarm_arrival(
        self,
        arm_key: str,
        arm: UiRoArmSerial,
        target: Dict[str, float],
        *,
        stage: str,
    ) -> Dict[str, Any]:
        deadline = time.monotonic() + max(0.5, float(self.args.plate_standby_arrival_timeout_s))
        consecutive_required = max(1, int(self.args.plate_standby_arrival_consecutive))
        consecutive = 0
        last_feedback: Optional[Dict[str, Any]] = None
        while time.monotonic() < deadline:
            remaining = max(0.1, deadline - time.monotonic())
            feedback = arm.feedback(
                timeout=min(float(self.args.roarm_feedback_timeout_s), remaining),
                retries=int(self.args.roarm_feedback_retries),
            )
            if feedback is None:
                consecutive = 0
                continue
            last_feedback = dict(feedback)
            x, y, z, _ = self._feedback_xyz(feedback)
            xy_error = math.hypot(x - float(target["x"]), y - float(target["y"]))
            z_error = abs(z - float(target["z"]))
            if (
                xy_error <= float(self.args.plate_standby_xy_tolerance_mm)
                and z_error <= float(self.args.plate_standby_z_tolerance_mm)
            ):
                consecutive += 1
                if consecutive >= consecutive_required:
                    message = (
                        f"{arm_key.upper()} {stage} 도착 확인 · "
                        f"XY 오차={xy_error:.1f}mm · Z 오차={z_error:.1f}mm"
                    )
                    self.signals.log.emit(message)
                    if self._robot_test_busy:
                        self._robot_test_log_message(message)
                    return last_feedback
            else:
                consecutive = 0
            time.sleep(max(0.02, float(self.args.plate_standby_poll_interval_s)))
        raise RuntimeError(
            f"{arm_key.upper()} {stage} 도착 확인 실패 · 마지막 피드백={last_feedback} · "
            f"마지막 RX={arm.last_rx_lines[-8:] if arm.last_rx_lines else '없음'}"
        )

    def _prepare_roarms_for_plate_motion(self) -> None:
        def emit_log(message: str) -> None:
            self.signals.log.emit(str(message))
            if self._robot_test_busy:
                self._robot_test_log_message(str(message))

        if not bool(self.args.plate_standby_safety):
            emit_log("경고: 폴딩판 테스트 전 로봇팔 대기 이동이 비활성화되어 있음")
            return

        targets = self._plate_standby_targets()
        arms: Dict[str, UiRoArmSerial] = {}
        current: Dict[str, Dict[str, float]] = {}
        try:
            emit_log("폴딩판 안전 준비: 양팔 serial 연결 및 현재 좌표 확인")
            for arm_key in ("arm1", "arm2"):
                port = self.args.arm1_port if arm_key == "arm1" else self.args.arm2_port
                arms[arm_key] = UiRoArmSerial(
                    port,
                    int(self.args.roarm_baud),
                    label=f"plate-{arm_key}",
                )

            for arm_key, arm in arms.items():
                feedback = arm.feedback(
                    timeout=float(self.args.roarm_feedback_timeout_s),
                    retries=int(self.args.roarm_feedback_retries),
                )
                if feedback is None:
                    raise RuntimeError(
                        f"{arm_key.upper()} T:1051 응답 없음 · port={arm.port} · "
                        f"마지막 RX={arm.last_rx_lines[-8:] if arm.last_rx_lines else '없음'}"
                    )
                emit_log(
                    f"{arm_key.upper()} T:1051 RX 원문={json.dumps(feedback, ensure_ascii=False)}"
                )
                x, y, z, t = self._feedback_xyz(feedback)
                current[arm_key] = {"x": x, "y": y, "z": z, "t": t}
                emit_log(
                    f"{arm_key.upper()} 현재 좌표=({x:.1f}, {y:.1f}, {z:.1f}, t={t:.3f})"
                )

            already_at_target: Dict[str, bool] = {}
            for arm_key in ("arm1", "arm2"):
                now = current[arm_key]
                target = targets[arm_key]
                already_at_target[arm_key] = bool(
                    math.hypot(now["x"] - target["x"], now["y"] - target["y"])
                    <= float(self.args.plate_standby_xy_tolerance_mm)
                    and abs(now["z"] - target["z"])
                    <= float(self.args.plate_standby_z_tolerance_mm)
                )

            # Stage 1: both arms first rise vertically at their current XY.  An arm
            # already verified at standby is left untouched.
            clear_targets: Dict[str, Dict[str, float]] = {}
            for arm_key, arm in arms.items():
                if already_at_target[arm_key]:
                    continue
                now = current[arm_key]
                clear_target = {
                    "x": now["x"],
                    "y": now["y"],
                    "z": max(now["z"], float(self.args.plate_standby_clear_z)),
                    "t": now["t"],
                }
                clear_targets[arm_key] = clear_target
                arm.move_goal(
                    clear_target["x"], clear_target["y"], clear_target["z"],
                    clear_target["t"], float(self.args.plate_standby_speed),
                )
                emit_log(
                    f"{arm_key.upper()} 현재 XY 유지 후 안전 높이 Z={clear_target['z']:.1f} 이동 명령"
                )
            for arm_key, clear_target in clear_targets.items():
                self._wait_plate_roarm_arrival(
                    arm_key, arms[arm_key], clear_target, stage="수직 상승"
                )

            # Stage 2: open to the source code's imaging-standby gripper angle,
            # then move both arms to the directly taught standby poses.
            for arm_key, arm in arms.items():
                arm.set_gripper(float(self.args.plate_standby_gripper_angle))
            emit_log(
                f"양팔 그리퍼 대기 각도 {float(self.args.plate_standby_gripper_angle):.2f}rad 적용"
            )

            for arm_key, arm in arms.items():
                if already_at_target[arm_key]:
                    continue
                target = targets[arm_key]
                arm.move_goal(
                    target["x"], target["y"], target["z"], target["t"],
                    float(self.args.plate_standby_speed),
                )
                emit_log(
                    f"{arm_key.upper()} 폴딩판 외측 대기 위치 이동 명령 · "
                    f"({target['x']:.1f}, {target['y']:.1f}, {target['z']:.1f}, t={target['t']:.3f})"
                )

            for arm_key, target in targets.items():
                self._wait_plate_roarm_arrival(
                    arm_key, arms[arm_key], target, stage="최종 대기 위치"
                )
            emit_log("양팔 대기 위치 확인 완료 · 폴딩판 명령 허용")
        except Exception as exc:
            raise RuntimeError(f"ROARM_STANDBY_FAILED: {exc}") from exc
        finally:
            for arm in arms.values():
                try:
                    arm.close()
                except Exception:
                    pass

    def _request_plate_test(self, plate: int) -> None:
        ready, reason = self._controller_ready_for_motion()
        if not ready:
            QMessageBox.warning(self.window, "실행 불가", reason)
            return
        if not self.plate_roarms_ready:
            QMessageBox.warning(
                self.window,
                "로봇팔 대기 위치 확인 필요",
                "먼저 '로봇팔 대기 위치로 이동' 버튼을 눌러 양팔 도착 확인을 완료하세요.",
            )
            return
        if plate not in (1, 2, 3, 4, 5):
            QMessageBox.warning(
                self.window,
                "지원하지 않는 판",
                f"현재 runtime은 {plate}번 명령을 지원하지 않습니다.",
            )
            return
        self.plate_board_widget.set_selected_plate(plate)
        self.selected_plate_label.setText(f"선택된 판: {plate}번 · 실행 중")

        def task() -> str:
            assert self.controller is not None
            ok = bool(self.controller.send_manual(str(plate), dry_run=False))
            return "e2 방식 판 동작 완료" if ok else "판 동작 실패"

        self._run_action_async(f"{plate}번 판", task)

    def _request_auto_test(self) -> None:
        ready, reason = self._controller_ready_for_motion()
        if not ready:
            QMessageBox.warning(self.window, "실행 불가", reason)
            return
        if not self.plate_roarms_ready:
            QMessageBox.warning(
                self.window,
                "로봇팔 대기 위치 확인 필요",
                "먼저 '로봇팔 대기 위치로 이동' 버튼을 눌러 양팔 도착 확인을 완료하세요.",
            )
            return
        if not self._confirm_motion(
            "폴딩판 작동",
            "다음 동기화 시나리오를 실행하시겠습니까?\n\n"
            "1) 1·2번 동시 완전 상승\n"
            "2) 1·2번 50% 하강 → 0.5초 대기 → 3번 완전 상승\n"
            "3) 3번 복귀 → 0.5초 대기 → 1·2번 완전 재상승\n"
            "4) 1·2번 50% 하강 → 0.5초 대기 → 4번 완전 상승\n"
            "5) 4번 복귀 → 0.5초 대기 → 1·2번 완전 재상승\n"
            "6) 1·2번 유지 상태에서 5번 완전 동작\n"
            "7) 1·2·5번 동시 원위치 복귀",
        ):
            return

        self.plate_board_widget.set_active_plates([])
        self.selected_plate_label.setText("폴딩판 작동: 동기화 시나리오 실행 중")

        def task() -> str:
            assert self.controller is not None
            self.controller.rearm("UI folding-board operation", force=True)
            return str(
                self.controller.trigger_fold_and_wait(
                    source="ui-folding-board-operation",
                    dry_run=False,
                    command="G",
                )
            )

        self._run_action_async("폴딩판 작동 G", task)

    def _handle_plate_runtime_line(self, line: str) -> None:
        upper = str(line).upper()
        # Specific tokens must be checked before P12_FULL because
        # P12_FULL_AFTER_P3/P4 contain the shorter token as a substring.
        stage_map = {
            "P12_HALF_WAIT_P3": ([1, 2], "1·2번 절반 하강 완료 · 0.5초 대기"),
            "P3_UP_AFTER_DELAY": ([1, 2, 3], "대기 후 3번 상승"),
            "P3_DOWN_WAIT_P12": ([3], "3번 복귀 완료 · 0.5초 대기"),
            "P12_FULL_AFTER_P3": ([1, 2], "대기 후 1·2번 재상승"),
            "P12_HALF_WAIT_P4": ([1, 2], "1·2번 절반 하강 완료 · 0.5초 대기"),
            "P4_UP_AFTER_DELAY": ([1, 2, 4], "대기 후 4번 상승"),
            "P4_DOWN_WAIT_P12": ([4], "4번 복귀 완료 · 0.5초 대기"),
            "P12_FULL_AFTER_P4": ([1, 2], "대기 후 1·2번 재상승"),
            "P5_UP_WITH_P12": ([1, 2, 5], "1·2번 유지 + 5번 동작"),
            "ALL_HOME": ([1, 2, 5], "1·2·5번 전체 복귀"),
            "P12_FULL": ([1, 2], "1·2번 메인판 동시 상승"),
        }
        for token, (plates, description) in stage_map.items():
            if token in upper:
                self.plate_board_widget.set_active_plates(plates)
                self.selected_plate_label.setText(f"폴딩판 작동 · {description}")
                return
        if "DONE" in upper:
            self.plate_board_widget.set_active_plates([])
            self.selected_plate_label.setText("폴딩판 작동 완료")
        elif "ABORT" in upper or upper.startswith("[STOP]"):
            self.plate_board_widget.set_active_plates([])
            self.selected_plate_label.setText("폴딩판 작동 중지됨")

    def _on_action_finished(self, label: str, success: bool, result: str) -> None:
        super()._on_action_finished(label, success, result)
        if label == "로봇팔 대기 위치로 이동":
            self.plate_roarms_ready = bool(success)
            if success:
                self.selected_plate_label.setText(
                    "로봇팔 대기 위치 확인 완료 · 폴딩판 작동 가능"
                )
                self._append_log(
                    "안전 게이트 해제: 이 화면을 나가기 전까지 폴딩판 작동을 반복할 수 있습니다."
                )
            else:
                self.selected_plate_label.setText(
                    "대기 위치 확인 실패 · 폴딩판 작동 차단"
                )
            self._refresh_controller_status()
        elif self.plate_roarms_ready and not self._busy:
            self.selected_plate_label.setText(
                "로봇팔 대기 위치 확인 완료 · 다음 폴딩판 작동 가능"
            )

        if not success and "ROARM_STANDBY_FAILED" in str(result):
            self.plate_roarms_ready = False
            self._refresh_controller_status()
            QMessageBox.critical(
                self.window,
                "폴딩판 동작 차단",
                "로봇팔이 안전 대기 위치에 도착한 것을 확인하지 못해 "
                "폴딩판 작동을 활성화하지 않았습니다.\n\n" + str(result),
            )

    def _close_roarms(self) -> None:
        for arm in self.roarms.values():
            try:
                arm.close()
            except Exception:
                pass
        self.roarms = {}

    def cleanup(self) -> None:
        try:
            if self.robot_calibration_active:
                self._stop_robot_calibration("프로그램 종료")
            self._close_roarms()
            self._close_robot_test_serial()
        finally:
            super().cleanup()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Jerber v19 UI with one-shot RoArm standby gate and delayed e4 G scenario"
    )
    parser.add_argument("--camera", default="/dev/video0")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=float, default=9.0)
    parser.add_argument("--camera-fourcc", default="YUYV", help="기본 YUYV: 1280x720 전체 화각 유지")
    parser.add_argument(
        "--camera-control-config",
        default=str(DEFAULT_PROJECT_DIR / "elp_ov2710_camera_controls.json"),
        help="카메라 노출·게인·플리커 설정 저장 JSON",
    )
    parser.add_argument("--camera-undistort", dest="camera_undistort", action="store_true", default=True)
    parser.add_argument("--no-camera-undistort", dest="camera_undistort", action="store_false")
    parser.add_argument(
        "--camera-calibration",
        default=str(DEFAULT_PROJECT_DIR / "undistort" / "elp_ov2710_1280x720_calibration.npz"),
    )
    parser.add_argument("--camera-undistort-alpha", type=float, default=0.0)
    parser.add_argument(
        "--camera-undistort-strict-size",
        dest="camera_undistort_strict_size", action="store_true", default=True,
    )
    parser.add_argument(
        "--no-camera-undistort-strict-size",
        dest="camera_undistort_strict_size", action="store_false",
    )
    parser.add_argument("--aruco-sample-count", type=int, default=5)
    parser.add_argument("--aruco-stable-max-shift-px", type=float, default=1.5)
    parser.add_argument("--windowed", action="store_true")
    parser.add_argument(
        "--display-profile",
        choices=("auto", "7ep", "desktop"),
        default="auto",
        help="UI 크기 프로필. auto는 1280x800급 화면에서 Waveshare 7EP compact 모드를 자동 적용",
    )

    parser.add_argument("--arduino-enable", dest="arduino_enable", action="store_true", default=True)
    parser.add_argument("--no-arduino-enable", dest="arduino_enable", action="store_false")
    parser.add_argument("--arduino-send", dest="arduino_send", action="store_true", default=True)
    parser.add_argument("--no-arduino-send", dest="arduino_send", action="store_false")
    parser.add_argument("--arduino-port", default="auto")
    parser.add_argument("--arduino-baud", type=int, default=9600)
    parser.add_argument("--arduino-ready-timeout-s", type=float, default=20.0)
    parser.add_argument("--arduino-done-timeout-s", type=float, default=60.0)
    parser.add_argument("--no-confirm", action="store_true", default=False)

    parser.add_argument(
        "--robot-config",
        default=str(DEFAULT_PROJECT_DIR / "dual_roarm_folding_board_config.json"),
    )
    parser.add_argument(
        "--hfile",
        default=str(DEFAULT_PROJECT_DIR / "elp_ov2710_folding_board_homography_cache.json"),
    )
    parser.add_argument(
        "--calib-cache",
        default=str(DEFAULT_PROJECT_DIR / "dual_roarm_affine_cache.json"),
    )
    parser.add_argument("--roarm-send", dest="roarm_send", action="store_true", default=True)
    parser.add_argument("--no-roarm-send", dest="roarm_send", action="store_false")
    parser.add_argument("--arm1-port", default="/dev/roarm_1")
    parser.add_argument("--arm2-port", default="/dev/roarm_2")
    parser.add_argument("--roarm-baud", type=int, default=115200)
    parser.add_argument("--roarm-feedback-timeout-s", type=float, default=2.5)
    parser.add_argument("--roarm-feedback-retries", type=int, default=3)

    # Folding-board collision prevention. Defaults are copied from the user's
    # step_a150-3-18 directly taught imaging-standby poses.
    parser.add_argument("--plate-standby-safety", dest="plate_standby_safety", action="store_true", default=True)
    parser.add_argument("--no-plate-standby-safety", dest="plate_standby_safety", action="store_false")
    parser.add_argument("--plate-standby-clear-z", type=float, default=180.0)
    parser.add_argument("--plate-standby-speed", type=float, default=0.45)
    parser.add_argument("--plate-standby-gripper-angle", type=float, default=1.57)
    parser.add_argument("--plate-standby-arrival-timeout-s", type=float, default=10.0)
    parser.add_argument("--plate-standby-poll-interval-s", type=float, default=0.15)
    parser.add_argument("--plate-standby-arrival-consecutive", type=int, default=2)
    parser.add_argument("--plate-standby-xy-tolerance-mm", type=float, default=25.0)
    parser.add_argument("--plate-standby-z-tolerance-mm", type=float, default=20.0)
    parser.add_argument("--plate-arm1-standby-x", type=float, default=9.981603127)
    parser.add_argument("--plate-arm1-standby-y", type=float, default=-127.327706)
    parser.add_argument("--plate-arm1-standby-z", type=float, default=24.8031204)
    parser.add_argument("--plate-arm1-standby-t", type=float, default=2.899223689)
    parser.add_argument("--plate-arm2-standby-x", type=float, default=-3.342226292)
    parser.add_argument("--plate-arm2-standby-y", type=float, default=-108.9054583)
    parser.add_argument("--plate-arm2-standby-z", type=float, default=21.07581154)
    parser.add_argument("--plate-arm2-standby-t", type=float, default=2.816388727)

    # Robot operation-check page.  Automatic motion stays at a high Cartesian Z
    # and board XY targets are converted with the calibrated affine matrix.
    parser.add_argument("--robot-test-auto-z", type=float, default=180.0)
    parser.add_argument("--robot-test-auto-speed", type=float, default=0.45)
    parser.add_argument("--robot-test-manual-speed", type=float, default=0.25)
    parser.add_argument("--robot-test-arrival-timeout-s", type=float, default=10.0)
    parser.add_argument("--robot-test-arrival-consecutive", type=int, default=2)
    parser.add_argument("--robot-test-poll-interval-s", type=float, default=0.15)
    parser.add_argument("--robot-test-xy-tolerance-mm", type=float, default=25.0)
    parser.add_argument("--robot-test-z-tolerance-mm", type=float, default=20.0)
    parser.add_argument("--robot-test-board-margin-mm", type=float, default=24.0)
    parser.add_argument("--robot-test-dead-half-width-mm", type=float, default=40.0)
    parser.add_argument("--robot-test-min-z", type=float, default=-140.0)
    parser.add_argument("--robot-test-max-z", type=float, default=260.0)
    parser.add_argument("--robot-test-wrist-step-rad", type=float, default=0.10)
    parser.add_argument("--robot-test-gripper-open", type=float, default=1.35)
    parser.add_argument("--robot-test-gripper-close", type=float, default=3.05)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    app.setFont(QFont("Noto Sans CJK KR", 12))

    qss_path = BASE_DIR / "folding_machine.qss"
    if qss_path.exists():
        app.setStyleSheet(qss_path.read_text(encoding="utf-8"))

    controller = MaintenanceUiController(args, app)
    app.aboutToQuit.connect(controller.cleanup)
    controller.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
