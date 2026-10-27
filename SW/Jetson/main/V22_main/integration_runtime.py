#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Jerber V22 integrated runtime
============================

Purpose
-------
1. Reuse the existing V22 LCD/PySide6 UI and camera pipeline.
2. Segment the garment, then classify the crop with the dedicated TOP/BOTTOM TensorRT engine before choosing a VLA.
3. Fail closed when the class is ambiguous; never route on a single weak frame.
4. Keep a persistent completed count: upper / bottom / total.
5. After upper/lower FINISH, verify both RoArms are safe, run the garment-specific folding sequence, then refresh the active empty-board baseline.

Important source-specific behavior
----------------------------------
- upper.py can be started in its optional continuous AUTO-X mode. This main watches
  [VLA0855-FULLAUTO-FINISH] and then releases that process before folding.
- The lower robot pipeline is intentionally disconnected in this clean baseline.
  BOTTOM classification ends in BOTTOM HOLD with no lower robot motion.

Recommended first test
----------------------
    python3 main.py --routing-mode classify --windowed

Real dispatch is intentionally explicit:
    python3 main.py --routing-mode dispatch --physical --arduino-send --roarm-send

Place this file inside the V22 folder, beside app_folding_board_test.py.
"""

from __future__ import annotations

import argparse
import collections
import copy
import dataclasses
import hashlib
import gc
import importlib.util
import json
import math
import os
import queue
import selectors
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Deque, Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np

# V22 local module. main.py is intended to live inside the V22 directory.
HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

try:
    import app_folding_board_test as v22
except Exception as exc:  # pragma: no cover - runtime environment diagnostic
    raise RuntimeError(
        "main.py must be placed in the V22 folder beside app_folding_board_test.py"
    ) from exc

from PySide6.QtCore import QObject, Signal, Slot, QTimer
from PySide6.QtGui import QFont
from PySide6.QtWidgets import QApplication, QLabel, QPushButton, QSizePolicy

try:
    from ultralytics import YOLO
except Exception:
    YOLO = None


BUILD = "2026-10-01-v38-small-display-compact-ui"
DEFAULT_SEG_MODEL = str(Path(__file__).resolve().parents[2] / "models" / "segmentation" / "kfashion_yolo26s_seg3_e100_best.engine")
DEFAULT_TOP_BOTTOM_MODEL = str(Path(__file__).resolve().parents[2] / "models" / "classification" / "top_bottom" / "top_bottom_vla_finetune_mix5_e8_v1_best.pt")


# -----------------------------------------------------------------------------
# Persistent completed counts
# -----------------------------------------------------------------------------
class GarmentCountStore:
    def __init__(self, path: Path) -> None:
        self.path = Path(path).expanduser().resolve()
        self.lock = threading.RLock()
        self.data = self._load()

    @staticmethod
    def _default() -> Dict[str, Any]:
        return {
            "schema_version": 1,
            "upper": 0,
            "bottom": 0,
            "total": 0,
            "updated_at_unix": None,
        }

    def _load(self) -> Dict[str, Any]:
        if not self.path.is_file():
            return self._default()
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
            if not isinstance(payload, dict):
                raise ValueError("root is not object")
            out = self._default()
            for key in ("upper", "bottom"):
                out[key] = max(0, int(payload.get(key, 0)))
            out["total"] = int(out["upper"] + out["bottom"])
            out["updated_at_unix"] = payload.get("updated_at_unix")
            return out
        except Exception as exc:
            print(f"[COUNT-WARN] invalid count file {self.path}: {exc!r}; starting at zero")
            return self._default()

    def snapshot(self) -> Dict[str, Any]:
        with self.lock:
            return dict(self.data)

    def increment(self, garment: str) -> Dict[str, Any]:
        key = str(garment).lower()
        if key not in {"upper", "bottom"}:
            raise ValueError(f"unsupported garment count key: {garment}")
        with self.lock:
            self.data[key] = int(self.data.get(key, 0)) + 1
            self.data["total"] = int(self.data.get("upper", 0)) + int(self.data.get("bottom", 0))
            self.data["updated_at_unix"] = time.time()
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix(self.path.suffix + ".tmp")
            tmp.write_text(json.dumps(self.data, ensure_ascii=False, indent=2), encoding="utf-8")
            tmp.replace(self.path)
            return dict(self.data)


# -----------------------------------------------------------------------------
# Board ROI + temporal garment classification
# -----------------------------------------------------------------------------
@dataclasses.dataclass
class FrameEvidence:
    upper_score: float = 0.0
    bottom_score: float = 0.0
    upper_conf: float = 0.0
    bottom_conf: float = 0.0
    garment_score: float = 0.0
    winner: str = "AMBIGUOUS"  # UPPER / BOTTOM / AMBIGUOUS / EMPTY / SETTLING
    detail: str = ""


@dataclasses.dataclass
class ClassificationResult:
    garment: str  # UPPER / BOTTOM / EMPTY / AMBIGUOUS
    confidence: float
    upper_votes: int
    bottom_votes: int
    sample_count: int
    detail: str


class BoardRoiProjector:
    """Projects the calibrated board rectangle into the corrected camera frame."""

    def __init__(self, hfile: Path, machine_cfg: Dict[str, Any], allow_full_frame: bool = False) -> None:
        self.hfile = Path(hfile).expanduser().resolve()
        self.machine_cfg = machine_cfg
        self.allow_full_frame = bool(allow_full_frame)
        self.H: Optional[np.ndarray] = None
        self._mask_cache: Dict[Tuple[int, int], np.ndarray] = {}
        self._load()

    def _load(self) -> None:
        try:
            payload = json.loads(self.hfile.read_text(encoding="utf-8"))
            H = np.asarray(payload["H"], dtype=np.float64).reshape(3, 3)
            if not np.all(np.isfinite(H)) or abs(float(np.linalg.det(H))) < 1e-12:
                raise ValueError("H is invalid/singular")
            self.H = H
        except Exception as exc:
            if self.allow_full_frame:
                self.H = None
                print(f"[CLASS-ROI-WARN] board H unavailable; full-frame fallback enabled: {exc!r}")
                return
            raise RuntimeError(
                f"classification requires a valid corrected-frame Homography H: {self.hfile} | {exc!r}"
            ) from exc

    def mask(self, shape: Sequence[int]) -> np.ndarray:
        h, w = int(shape[0]), int(shape[1])
        key = (h, w)
        cached = self._mask_cache.get(key)
        if cached is not None:
            return cached
        if self.H is None:
            mask = np.full((h, w), 255, np.uint8)
            self._mask_cache[key] = mask
            return mask

        marker = self.machine_cfg.get("aruco", {}).get("marker_board_mm", {})
        pts = np.asarray(list(marker.values()), dtype=np.float64).reshape(-1, 2)
        if len(pts) < 4:
            raise RuntimeError("machine config has fewer than four board marker points")
        x0, x1 = float(np.min(pts[:, 0])), float(np.max(pts[:, 0]))
        y0, y1 = float(np.min(pts[:, 1])), float(np.max(pts[:, 1]))
        board = np.asarray([[[x0, y0], [x1, y0], [x1, y1], [x0, y1]]], np.float32)
        Hinv = np.linalg.inv(self.H).astype(np.float32)
        px = cv2.perspectiveTransform(board, Hinv)[0]
        if not np.all(np.isfinite(px)):
            raise RuntimeError("board ROI projection produced non-finite pixels")
        poly = np.rint(px).astype(np.int32)
        mask = np.zeros((h, w), np.uint8)
        cv2.fillConvexPoly(mask, poly, 255)
        area = int(cv2.countNonZero(mask))
        if area < int(0.08 * h * w):
            raise RuntimeError(f"projected board ROI suspiciously small: {area}/{h*w}")
        self._mask_cache[key] = mask
        print(f"[CLASS-ROI] corrected board ROI ready area={area}px ratio={area/max(1,h*w):.3f}")
        return mask


class TemporalGarmentClassifier:
    """Two-stage garment classifier: segmentation crop -> dedicated TOP/BOTTOM classifier.

    The segmentation model is used only to isolate the garment on the calibrated board.
    Routing is decided exclusively by the dedicated TOP/BOTTOM classification model.
V8 adds orientation-aware segmentation: 0-degree inference is preferred, while repeated misses trigger 90-degree clockwise/counter-clockwise rescue. A successful rescue orientation is latched until the garment disappears. Classification still uses the frozen original-camera RGB crop.
    """

    SEG_ALLOWED_NAMES = {"outer", "top", "bottoms", "bottom", "pants", "trousers", "shirt", "tshirt", "t-shirt", "dress"}
    UPPER_LABEL_TOKENS = ("top", "upper", "shirt", "tshirt", "t-shirt")
    BOTTOM_LABEL_TOKENS = ("bottom", "lower", "pants", "pant", "trouser", "jeans")

    def __init__(
        self,
        segment_model_path: Path,
        classify_model_path: Path,
        roi: BoardRoiProjector,
        *,
        segment_imgsz: int = 640,
        classify_imgsz: int = 224,
        det_conf: float = 0.12,
        window: int = 9,
        min_votes: int = 6,
        min_vote_share: float = 0.70,
        min_mean_conf: float = 0.70,
        min_score_margin: float = 1.35,
        min_roi_overlap: float = 0.15,
        min_area_ratio: float = 0.003,
        per_frame_min_conf: float = 0.55,
        per_frame_min_ratio: float = 1.15,
        label_map_mode: str = "swapped",
        settle_frames: int = 6,
        settle_area_tol: float = 0.045,
        settle_center_tol_px: float = 10.0,
        settle_bbox_tol: float = 0.055,
        tta_enabled: bool = True,
        geometry_enabled: bool = True,
        rotation_rescue_enabled: bool = True,
        rotation_rescue_after: int = 3,
        rotation_rescue_period: int = 4,
        rescue_conf: float = 0.03,
        rotation_release_misses: int = 6,
    ) -> None:
        self.segment_model_path = Path(segment_model_path).expanduser().resolve()
        self.classify_model_path = Path(classify_model_path).expanduser().resolve()
        self.roi = roi
        self.segment_imgsz = int(segment_imgsz)
        self.classify_imgsz = int(classify_imgsz)
        self.det_conf = float(det_conf)
        self.window = max(5, int(window))
        self.min_votes = max(3, min(int(min_votes), self.window))
        self.min_vote_share = float(min_vote_share)
        self.min_mean_conf = float(min_mean_conf)
        self.min_score_margin = float(min_score_margin)
        self.min_roi_overlap = float(min_roi_overlap)
        self.min_area_ratio = float(min_area_ratio)
        self.per_frame_min_conf = float(per_frame_min_conf)
        self.per_frame_min_ratio = float(per_frame_min_ratio)

        mode = str(label_map_mode).strip().lower()
        if mode not in {"native", "swapped"}:
            raise ValueError(f"unsupported label_map_mode={label_map_mode!r}")
        self.label_map_mode = mode

        self.settle_frames = max(3, int(settle_frames))
        self.settle_area_tol = max(0.005, float(settle_area_tol))
        self.settle_center_tol_px = max(1.0, float(settle_center_tol_px))
        self.settle_bbox_tol = max(0.005, float(settle_bbox_tol))
        self.tta_enabled = bool(tta_enabled)
        self.geometry_enabled = bool(geometry_enabled)

        self.rotation_rescue_enabled = bool(rotation_rescue_enabled)
        self.rotation_rescue_after = max(1, int(rotation_rescue_after))
        self.rotation_rescue_period = max(1, int(rotation_rescue_period))
        self.rescue_conf = max(0.001, min(float(rescue_conf), self.det_conf))
        self.rotation_release_misses = max(2, int(rotation_release_misses))
        self._seg_orientation = 0  # 0, +90(CW), -90(CCW)
        self._seg_miss_streak = 0

        self.history: Deque[FrameEvidence] = collections.deque(maxlen=self.window)
        self._settle_history: Deque[Dict[str, Any]] = collections.deque(maxlen=self.settle_frames)
        self._stable_bbox: Optional[Tuple[int, int, int, int]] = None
        self._stable_reference: Optional[Dict[str, Any]] = None
        self._stable_generation = 0

        self.segment_model = None
        self.classify_model = None
        self._warmed = False
        self._infer_seq = 0
        self._presence_seq = 0
        self._last_select_detail = "not evaluated"

    @staticmethod
    def _name(names: Any, cls_id: int) -> str:
        if isinstance(names, dict):
            return str(names.get(int(cls_id), cls_id)).strip().lower()
        if isinstance(names, (list, tuple)) and 0 <= int(cls_id) < len(names):
            return str(names[int(cls_id)]).strip().lower()
        return str(cls_id).strip().lower()

    @staticmethod
    def _to_numpy(value: Any) -> np.ndarray:
        if hasattr(value, "detach"):
            value = value.detach()
        if hasattr(value, "cpu"):
            value = value.cpu()
        if hasattr(value, "numpy"):
            value = value.numpy()
        return np.asarray(value)

    def _semantic_from_label(self, label: str) -> Optional[str]:
        """Convert dedicated-classifier metadata into physical UPPER/BOTTOM.

        The current e30 engine reports metadata names {BOTTOM, TOP}, but real-camera
        experiments showed that the learned decision can behave as if that pair is
        reversed for some samples.  V6 therefore makes the mapping explicit instead
        of baking the assumption into the code.  V7 defaults to ``native`` because top_bottom_vla_finetune_mix5_e8_v1_best.pt stores the intended physical class semantics: TOP=>UPPER and BOTTOM=>BOTTOM.
        """
        value = str(label).strip().lower().replace("_", "-")

        if value in {"top", "upper"}:
            return "BOTTOM" if self.label_map_mode == "swapped" else "UPPER"
        if value in {"bottom", "lower"}:
            return "UPPER" if self.label_map_mode == "swapped" else "BOTTOM"

        # Explicit garment nouns are not swapped.
        if any(token in value for token in ("shirt", "tshirt", "t-shirt")):
            return "UPPER"
        if any(token in value for token in ("pants", "pant", "trouser", "jeans")):
            return "BOTTOM"
        return None

    @staticmethod
    def _letterbox_square(image: np.ndarray, size: int, pad_value: int = 114) -> np.ndarray:
        size = max(32, int(size))
        if image is None or image.size == 0:
            raise ValueError("classifier crop is empty")
        h, w = image.shape[:2]
        scale = min(float(size) / max(1, w), float(size) / max(1, h))
        nw = max(1, min(size, int(round(w * scale))))
        nh = max(1, min(size, int(round(h * scale))))
        resized = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_LINEAR)
        canvas = np.full((size, size, 3), int(pad_value), dtype=np.uint8)
        x0 = (size - nw) // 2
        y0 = (size - nh) // 2
        canvas[y0:y0 + nh, x0:x0 + nw] = resized
        return canvas

    def ensure_model(self) -> None:
        if self.segment_model is not None and self.classify_model is not None:
            return
        if YOLO is None:
            raise RuntimeError("ultralytics import failed; classifier cannot start")
        if not self.segment_model_path.is_file():
            raise FileNotFoundError(f"segmentation TensorRT not found: {self.segment_model_path}")
        if not self.classify_model_path.is_file():
            raise FileNotFoundError(
                "TOP/BOTTOM TensorRT not found: " + str(self.classify_model_path)
                + " | expected from test_top_bottom_trt_camera.py"
            )
        # The fine-tuned .pt checkpoint references train_top_bottom_cls_letterbox.py.
        # Make its own directory importable automatically so no shell PYTHONPATH is required.
        model_support_dir = str(self.classify_model_path.parent)
        if model_support_dir not in sys.path:
            sys.path.insert(0, model_support_dir)
            print(f"[CLASS-MODEL] support path added: {model_support_dir}", flush=True)

        print(f"[CLASS-MODEL] segmentation loading: {self.segment_model_path}")
        self.segment_model = YOLO(str(self.segment_model_path), task="segment")
        print(f"[CLASS-MODEL] TOP/BOTTOM loading: {self.classify_model_path}")
        self.classify_model = YOLO(str(self.classify_model_path), task="classify")
        print("[CLASS-MODEL] both model wrappers created")

    def warmup(self) -> None:
        if self._warmed:
            return
        self.ensure_model()
        seg_dummy = np.zeros((self.segment_imgsz, self.segment_imgsz, 3), dtype=np.uint8)
        cls_dummy = np.full((self.classify_imgsz, self.classify_imgsz, 3), 114, dtype=np.uint8)
        t0 = time.monotonic()
        print(f"[CLASS-WARMUP] SEG START imgsz={self.segment_imgsz}", flush=True)
        self.segment_model.predict(
            source=seg_dummy,
            imgsz=self.segment_imgsz,
            conf=self.det_conf,
            verbose=False,
        )
        t1 = time.monotonic()
        print(f"[CLASS-WARMUP] SEG DONE {t1-t0:.3f}s", flush=True)
        print(f"[CLASS-WARMUP] TOP/BOTTOM START imgsz={self.classify_imgsz}", flush=True)
        warm_cls = self.classify_model.predict(
            source=cls_dummy,
            imgsz=self.classify_imgsz,
            verbose=False,
        )[0]
        t2 = time.monotonic()
        print(f"[CLASS-WARMUP] TOP/BOTTOM DONE {t2-t1:.3f}s", flush=True)
        if self.label_map_mode == "swapped":
            mapping_text = "raw TOP=>physical BOTTOM, raw BOTTOM=>physical UPPER"
        else:
            mapping_text = "raw TOP=>physical UPPER, raw BOTTOM=>physical BOTTOM"
        print(
            f"[CLASS-LABEL-MAP] engine_names={getattr(warm_cls, 'names', {})!r} | "
            f"mode={self.label_map_mode} | {mapping_text}",
            flush=True,
        )
        print(
            f"[CLASS-V8] settle={self.settle_frames}frames "
            f"areaTol={self.settle_area_tol:.3f} centerTol={self.settle_center_tol_px:.1f}px "
            f"bboxTol={self.settle_bbox_tol:.3f} inference=SINGLE_RGB_224_LETTERBOX114 "
            f"rotationRescue={self.rotation_rescue_enabled} rescueAfter={self.rotation_rescue_after} "
            f"rescueConf={self.rescue_conf:.3f} heuristicOverride=OFF",
            flush=True,
        )
        self._warmed = True
        print(f"[CLASS-WARMUP] ALL PASS total={t2-t0:.3f}s", flush=True)

    def unload(self) -> None:
        self.history.clear()
        self.segment_model = None
        self.classify_model = None
        self._warmed = False
        self._seg_orientation = 0
        self._seg_miss_streak = 0
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    def _instance_mask(
        self,
        result: Any,
        index: int,
        frame_shape: Sequence[int],
        board_mask: np.ndarray,
    ) -> Tuple[float, float, Optional[np.ndarray]]:
        h, w = int(frame_shape[0]), int(frame_shape[1])
        board_area = max(1, int(cv2.countNonZero(board_mask)))
        masks = getattr(result, "masks", None)
        data = getattr(masks, "data", None) if masks is not None else None
        if data is not None:
            try:
                raw = self._to_numpy(data[index]).squeeze()
                m = (raw > 0.5).astype(np.uint8)
                if m.shape != (h, w):
                    m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
                total = int(np.count_nonzero(m))
                if total <= 0:
                    return 0.0, 0.0, None
                inside_mask = ((m > 0) & (board_mask > 0)).astype(np.uint8)
                inside = int(np.count_nonzero(inside_mask))
                return float(inside) / float(board_area), float(inside) / float(total), inside_mask
            except Exception:
                pass

        boxes = getattr(result, "boxes", None)
        xyxy = getattr(boxes, "xyxy", None) if boxes is not None else None
        if xyxy is None:
            return 0.0, 0.0, None
        try:
            b = self._to_numpy(xyxy[index]).reshape(-1)[:4]
            x0, y0, x1, y1 = [int(round(float(v))) for v in b]
            x0, x1 = max(0, min(w - 1, x0)), max(0, min(w, x1))
            y0, y1 = max(0, min(h - 1, y0)), max(0, min(h, y1))
            if x1 <= x0 or y1 <= y0:
                return 0.0, 0.0, None
            box_mask = np.zeros((h, w), np.uint8)
            box_mask[y0:y1, x0:x1] = 1
            total = int(np.count_nonzero(box_mask))
            inside_mask = ((box_mask > 0) & (board_mask > 0)).astype(np.uint8)
            inside = int(np.count_nonzero(inside_mask))
            return float(inside) / float(board_area), float(inside) / float(max(1, total)), inside_mask
        except Exception:
            return 0.0, 0.0, None

    def _select_garment(self, result: Any, frame: np.ndarray, board_mask: np.ndarray):
        """Select one board-supported garment instance.

        V2 used a 60% instance-inside-board hard gate.  On the live ELP view that
        rejected every otherwise-valid garment mask, so the dedicated classifier
        never ran.  V3 keeps the board intersection as the anti-robot/background
        guard, but uses a tiered gate: preferred garment classes first, then a
        conservative generic-mask rescue if class naming differs from expectation.
        """
        boxes = getattr(result, "boxes", None)
        if boxes is None or len(boxes) == 0:
            self._last_select_detail = "boxes=0"
            return None

        cls_values = self._to_numpy(boxes.cls).reshape(-1)
        conf_values = self._to_numpy(boxes.conf).reshape(-1)
        strict = []
        relaxed = []
        generic = []
        debug = []

        soft_area = max(0.0015, 0.50 * float(self.min_area_ratio))
        soft_overlap = max(0.08, 0.50 * float(self.min_roi_overlap))

        for i, (cls_raw, conf_raw) in enumerate(zip(cls_values, conf_values)):
            cls_id = int(cls_raw)
            conf = float(conf_raw)
            name = self._name(getattr(result, "names", {}), cls_id)
            area_ratio, overlap, mask = self._instance_mask(result, i, frame.shape, board_mask)
            inside_px = int(np.count_nonzero(mask)) if mask is not None else 0
            debug.append(f"{i}:{name}@{conf:.2f}/A{area_ratio:.4f}/R{overlap:.2f}/px{inside_px}")
            if mask is None or inside_px < 350:
                continue

            score = float(conf) * math.sqrt(max(area_ratio, 1e-9))
            item = (score, conf, area_ratio, overlap, name, mask)

            if name in self.SEG_ALLOWED_NAMES:
                if area_ratio >= self.min_area_ratio and overlap >= self.min_roi_overlap:
                    strict.append(item)
                elif area_ratio >= soft_area and overlap >= soft_overlap:
                    relaxed.append(item)
            else:
                # Safe fallback only for a reasonably large mask substantially
                # supported by the calibrated board.  This handles unexpected
                # K-Fashion class names without opening the door to side robot arms.
                if area_ratio >= max(0.006, self.min_area_ratio) and overlap >= max(0.25, self.min_roi_overlap):
                    generic.append(item)

        pool = strict or relaxed or generic
        tier = "STRICT" if strict else ("RELAXED" if relaxed else ("GENERIC" if generic else "NONE"))
        if not pool:
            self._last_select_detail = f"tier=NONE boxes={len(cls_values)} " + " | ".join(debug[:8])
            return None
        pool.sort(key=lambda x: (x[0], x[2], x[1]), reverse=True)
        best = pool[0]
        self._last_select_detail = (
            f"tier={tier} selected={best[4]} conf={best[1]:.2f} "
            f"A={best[2]:.4f} R={best[3]:.2f}; boxes={len(cls_values)}; "
            + " | ".join(debug[:8])
        )
        return best

    @staticmethod
    def _orientation_name(deg: int) -> str:
        return {0: "0", 90: "CW90", -90: "CCW90"}.get(int(deg), str(deg))

    @staticmethod
    def _rotate_image(image: np.ndarray, deg: int) -> np.ndarray:
        if int(deg) == 0:
            return image
        if int(deg) == 90:
            return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        if int(deg) == -90:
            return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
        raise ValueError(f"unsupported segmentation orientation: {deg}")

    @staticmethod
    def _mask_to_original(mask: np.ndarray, deg: int, original_shape: Sequence[int]) -> np.ndarray:
        if int(deg) == 0:
            out = mask
        elif int(deg) == 90:
            out = cv2.rotate(mask, cv2.ROTATE_90_COUNTERCLOCKWISE)
        elif int(deg) == -90:
            out = cv2.rotate(mask, cv2.ROTATE_90_CLOCKWISE)
        else:
            raise ValueError(f"unsupported segmentation orientation: {deg}")
        h, w = int(original_shape[0]), int(original_shape[1])
        if out.shape[:2] != (h, w):
            out = cv2.resize(out.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
        return (out > 0).astype(np.uint8)

    def _segment_at_orientation(
        self,
        frame: np.ndarray,
        board_mask: np.ndarray,
        deg: int,
        conf: float,
    ):
        """Run segmentation in one orientation and return selection in ORIGINAL camera coordinates."""
        rframe = self._rotate_image(frame, deg)
        rboard = self._rotate_image(board_mask, deg)
        t0 = time.monotonic()
        result = self.segment_model.predict(
            source=rframe,
            imgsz=self.segment_imgsz,
            conf=float(conf),
            verbose=False,
        )[0]
        elapsed = time.monotonic() - t0
        selected = self._select_garment(result, rframe, rboard)
        detail = self._last_select_detail
        if selected is None:
            return None, elapsed, detail

        score, seg_conf, area_ratio, overlap, seg_name, rmask = selected
        omask = self._mask_to_original(rmask, deg, frame.shape)
        return (
            score, seg_conf, area_ratio, overlap, seg_name, omask
        ), elapsed, detail

    def _segment_with_rotation_rescue(
        self,
        frame: np.ndarray,
        board_mask: np.ndarray,
        seq: int,
    ):
        """Prefer the current orientation; rescue persistent misses with ±90° inference.

        Once a rotated orientation succeeds, it is latched so a horizontal garment
        can remain detectable on subsequent frames instead of dropping back to 0°.
        """
        preferred = int(self._seg_orientation)
        conf = self.rescue_conf if preferred != 0 else self.det_conf
        selected, elapsed, detail = self._segment_at_orientation(
            frame, board_mask, preferred, conf
        )
        total_elapsed = float(elapsed)

        if selected is not None:
            self._seg_miss_streak = 0
            self._last_select_detail = (
                f"ori={self._orientation_name(preferred)} confThr={conf:.3f} | {detail}"
            )
            return selected, total_elapsed

        self._seg_miss_streak += 1
        miss = int(self._seg_miss_streak)
        primary_detail = detail

        # While a rotated orientation is latched, recover quickly if the garment
        # changes orientation. On an empty board, release the latch after N misses.
        if preferred != 0 and miss >= self.rotation_release_misses:
            print(
                f"[CLASS-SEG-ROTATE] release {self._orientation_name(preferred)} "
                f"after {miss} misses",
                flush=True,
            )
            self._seg_orientation = 0
            preferred = 0

        if not self.rotation_rescue_enabled:
            self._last_select_detail = (
                f"ori={self._orientation_name(preferred)} miss={miss} | {primary_detail}"
            )
            return None, total_elapsed

        # Do not triple inference continuously while the board is empty.
        # Rescue on the first persistent-miss threshold, then periodically.
        rescue_due = (
            miss == self.rotation_rescue_after
            or (
                miss > self.rotation_rescue_after
                and (miss - self.rotation_rescue_after) % self.rotation_rescue_period == 0
            )
            or int(self._seg_orientation) != 0
        )
        if not rescue_due:
            self._last_select_detail = (
                f"ori={self._orientation_name(preferred)} miss={miss} rescue=pending | "
                f"{primary_detail}"
            )
            return None, total_elapsed

        orientations = [0, 90, -90]
        # Exclude whichever orientation was already evaluated.
        orientations = [d for d in orientations if d != int(self._seg_orientation)]
        candidates = []
        debug = []
        for deg in orientations:
            rescue_selected, dt, rescue_detail = self._segment_at_orientation(
                frame, board_mask, deg, self.rescue_conf
            )
            total_elapsed += float(dt)
            debug.append(
                f"{self._orientation_name(deg)}:{'HIT' if rescue_selected is not None else 'MISS'}"
            )
            if rescue_selected is not None:
                candidates.append((rescue_selected, deg, rescue_detail))

        if not candidates:
            self._last_select_detail = (
                f"ori={self._orientation_name(self._seg_orientation)} miss={miss} "
                f"rotationRescue={'/'.join(debug)} | {primary_detail}"
            )
            return None, total_elapsed

        candidates.sort(
            key=lambda item: (float(item[0][0]), float(item[0][2]), float(item[0][1])),
            reverse=True,
        )
        best, best_deg, best_detail = candidates[0]
        old = int(self._seg_orientation)
        self._seg_orientation = int(best_deg)
        self._seg_miss_streak = 0
        self._last_select_detail = (
            f"ROTATION-RESCUE {self._orientation_name(old)}=>{self._orientation_name(best_deg)} "
            f"confThr={self.rescue_conf:.3f} | {best_detail}"
        )
        print(
            f"[CLASS-SEG-ROTATE-RESCUE] #{seq} "
            f"{self._orientation_name(old)}=>{self._orientation_name(best_deg)} "
            f"{best_detail}",
            flush=True,
        )
        return best, total_elapsed

    @staticmethod
    def _mask_descriptor(mask: np.ndarray) -> Dict[str, Any]:
        ys, xs = np.nonzero(mask > 0)
        if len(xs) < 16:
            raise ValueError("selected garment mask has too few pixels")
        x0, x1 = int(xs.min()), int(xs.max()) + 1
        y0, y1 = int(ys.min()), int(ys.max()) + 1
        area = float(len(xs))
        cx = float(np.mean(xs))
        cy = float(np.mean(ys))
        return {
            "bbox": (x0, y0, x1, y1),
            "area": area,
            "center": (cx, cy),
            "width": float(max(1, x1 - x0)),
            "height": float(max(1, y1 - y0)),
        }

    def _reset_stability(self, reason: str = "") -> None:
        self._settle_history.clear()
        self._stable_bbox = None
        self._stable_reference = None
        if reason:
            print(f"[CLASS-SETTLE-RESET] {reason}", flush=True)

    def _observe_stability(self, mask: np.ndarray, frame_shape: Sequence[int]) -> Tuple[bool, str]:
        desc = self._mask_descriptor(mask)

        # If a frozen garment suddenly changes shape/position, do not keep classifying
        # the stale crop.  Return to the settling phase.
        if self._stable_reference is not None:
            ref = self._stable_reference
            area_rel = abs(desc["area"] - ref["area"]) / max(ref["area"], 1.0)
            dc = math.hypot(
                desc["center"][0] - ref["center"][0],
                desc["center"][1] - ref["center"][1],
            )
            wr = abs(desc["width"] - ref["width"]) / max(ref["width"], 1.0)
            hr = abs(desc["height"] - ref["height"]) / max(ref["height"], 1.0)
            if (
                area_rel > self.settle_area_tol * 2.5
                or dc > self.settle_center_tol_px * 2.5
                or max(wr, hr) > self.settle_bbox_tol * 2.5
            ):
                self.history.clear()
                self._reset_stability(
                    f"garment moved area={area_rel:.3f} center={dc:.1f}px size={max(wr,hr):.3f}"
                )
            else:
                return True, (
                    f"stable generation={self._stable_generation} areaΔ={area_rel:.3f} "
                    f"centerΔ={dc:.1f}px sizeΔ={max(wr,hr):.3f}"
                )

        self._settle_history.append(desc)
        if len(self._settle_history) < self.settle_frames:
            return False, f"collect {len(self._settle_history)}/{self.settle_frames}"

        items = list(self._settle_history)
        areas = np.asarray([x["area"] for x in items], np.float64)
        centers = np.asarray([x["center"] for x in items], np.float64)
        widths = np.asarray([x["width"] for x in items], np.float64)
        heights = np.asarray([x["height"] for x in items], np.float64)

        area_span = float((np.max(areas) - np.min(areas)) / max(float(np.mean(areas)), 1.0))
        med_center = np.median(centers, axis=0)
        center_spread = float(np.max(np.linalg.norm(centers - med_center[None, :], axis=1)))
        width_span = float((np.max(widths) - np.min(widths)) / max(float(np.mean(widths)), 1.0))
        height_span = float((np.max(heights) - np.min(heights)) / max(float(np.mean(heights)), 1.0))
        bbox_span = max(width_span, height_span)

        stable = bool(
            area_span <= self.settle_area_tol
            and center_spread <= self.settle_center_tol_px
            and bbox_span <= self.settle_bbox_tol
        )
        detail = (
            f"areaSpan={area_span:.3f}/{self.settle_area_tol:.3f} "
            f"centerSpread={center_spread:.1f}/{self.settle_center_tol_px:.1f}px "
            f"bboxSpan={bbox_span:.3f}/{self.settle_bbox_tol:.3f}"
        )
        if not stable:
            return False, detail

        # Freeze the crop rectangle using the median of the stable descriptors.
        boxes = np.asarray([x["bbox"] for x in items], np.float64)
        x0, y0, x1, y1 = np.rint(np.median(boxes, axis=0)).astype(int).tolist()
        h, w = int(frame_shape[0]), int(frame_shape[1])
        pad = max(10, int(round(0.07 * max(1, x1 - x0, y1 - y0))))
        x0, x1 = max(0, x0 - pad), min(w, x1 + pad)
        y0, y1 = max(0, y0 - pad), min(h, y1 + pad)
        self._stable_bbox = (x0, y0, x1, y1)
        self._stable_reference = {
            "area": float(np.median(areas)),
            "center": (float(med_center[0]), float(med_center[1])),
            "width": float(np.median(widths)),
            "height": float(np.median(heights)),
        }
        self._stable_generation += 1
        self.history.clear()
        print(
            f"[CLASS-SETTLE] STABLE generation={self._stable_generation} "
            f"bbox={self._stable_bbox} {detail}",
            flush=True,
        )
        return True, f"STABLE {detail}"

    def _fixed_crop_pair(self, frame: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if self._stable_bbox is None:
            raise RuntimeError("stable crop requested before settling")
        x0, y0, x1, y1 = self._stable_bbox
        raw = frame[y0:y1, x0:x1].copy()
        if raw.size == 0:
            raise RuntimeError("stable crop is empty")
        local_mask = (mask[y0:y1, x0:x1] > 0)
        masked = np.full_like(raw, 114)
        masked[local_mask] = raw[local_mask]
        return raw, masked

    def _snapshot_crop_pair(
        self,
        frame: np.ndarray,
        mask: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Build classifier crops directly from one frozen snapshot mask.

        Mode B snapshot classification intentionally skips the live temporal
        settling state machine, so this path must not depend on _stable_bbox.
        """
        if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
            raise RuntimeError("snapshot crop requested from invalid frame")
        if mask is None or not isinstance(mask, np.ndarray) or mask.size == 0:
            raise RuntimeError("snapshot crop requested from invalid mask")
        if mask.shape[:2] != frame.shape[:2]:
            raise RuntimeError(
                f"snapshot crop frame/mask shape mismatch: "
                f"frame={frame.shape[:2]} mask={mask.shape[:2]}"
            )

        desc = self._mask_descriptor(mask)
        x0, y0, x1, y1 = [int(v) for v in desc["bbox"]]

        h, w = int(frame.shape[0]), int(frame.shape[1])
        pad = max(10, int(round(0.07 * max(1, x1 - x0, y1 - y0))))
        x0, x1 = max(0, x0 - pad), min(w, x1 + pad)
        y0, y1 = max(0, y0 - pad), min(h, y1 + pad)

        if x1 <= x0 or y1 <= y0:
            raise RuntimeError(
                f"snapshot crop bbox invalid after padding: {(x0, y0, x1, y1)}"
            )

        raw = frame[y0:y1, x0:x1].copy()
        if raw.size == 0:
            raise RuntimeError("snapshot crop is empty")

        local_mask = mask[y0:y1, x0:x1] > 0
        if not bool(np.any(local_mask)):
            raise RuntimeError("snapshot crop contains no garment-mask pixels")

        masked = np.full_like(raw, 114)
        masked[local_mask] = raw[local_mask]

        print(
            f"[CLASS-SNAPSHOT-CROP] bbox={(x0, y0, x1, y1)} "
            f"maskArea={int(np.count_nonzero(local_mask))} "
            f"crop={raw.shape[1]}x{raw.shape[0]}",
            flush=True,
        )
        return raw, masked

    @staticmethod
    def _row_run_metrics(binary: np.ndarray) -> Tuple[float, float]:
        """Return (split-row fraction, longest consecutive split-row fraction)."""
        h, w = binary.shape[:2]
        if h < 8 or w < 8:
            return 0.0, 0.0
        min_run = max(3, int(round(0.055 * w)))
        min_gap = max(3, int(round(0.055 * w)))
        split_flags = []
        for y in range(h):
            xs = np.flatnonzero(binary[y] > 0)
            if xs.size < 2 * min_run:
                split_flags.append(False)
                continue
            gaps = np.flatnonzero(np.diff(xs) > 1)
            starts = np.r_[0, gaps + 1]
            ends = np.r_[gaps, xs.size - 1]
            runs = [(int(xs[s]), int(xs[e])) for s, e in zip(starts, ends)]
            runs = [r for r in runs if (r[1] - r[0] + 1) >= min_run]
            good = False
            if len(runs) >= 2:
                runs.sort()
                for a, b in zip(runs[:-1], runs[1:]):
                    if (b[0] - a[1] - 1) >= min_gap:
                        good = True
                        break
            split_flags.append(good)
        flags = np.asarray(split_flags, np.uint8)
        frac = float(np.mean(flags)) if flags.size else 0.0
        longest = 0
        current = 0
        for value in flags.tolist():
            if value:
                current += 1
                longest = max(longest, current)
            else:
                current = 0
        return frac, float(longest) / float(max(1, h))

    def _geometry_evidence(self, mask: np.ndarray) -> Tuple[Optional[str], float, Dict[str, float]]:
        """Conservative shape cue; it may veto/confirm, but weak geometry never routes."""
        if not self.geometry_enabled:
            return None, 0.0, {}

        ys, xs = np.nonzero(mask > 0)
        if len(xs) < 500:
            return None, 0.0, {}

        pts = np.column_stack([xs.astype(np.float64), ys.astype(np.float64)])
        center = np.mean(pts, axis=0)
        centered = pts - center[None, :]
        cov = centered.T @ centered / max(1, len(centered) - 1)
        vals, vecs = np.linalg.eigh(cov)
        major = vecs[:, int(np.argmax(vals))]
        # Rotate the PCA major axis to vertical so split-leg evidence is orientation robust.
        angle = math.degrees(math.atan2(float(major[1]), float(major[0])))
        rotate_deg = 90.0 - angle
        h0, w0 = mask.shape[:2]
        M = cv2.getRotationMatrix2D((float(center[0]), float(center[1])), rotate_deg, 1.0)
        rot = cv2.warpAffine(
            (mask > 0).astype(np.uint8) * 255,
            M,
            (w0, h0),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        yy, xx = np.nonzero(rot > 0)
        if len(xx) < 500:
            return None, 0.0, {}
        x0, x1 = int(xx.min()), int(xx.max()) + 1
        y0, y1 = int(yy.min()), int(yy.max()) + 1
        roi = (rot[y0:y1, x0:x1] > 0).astype(np.uint8)
        rh, rw = roi.shape[:2]
        aspect = float(rh) / float(max(1, rw))

        split_frac, split_run = self._row_run_metrics(roi)

        # Width profile for a weak shirt cue.  This never overrides strong pants evidence.
        spans = []
        for row in roi:
            ids = np.flatnonzero(row > 0)
            spans.append(float(ids[-1] - ids[0] + 1) if ids.size else 0.0)
        spans = np.asarray(spans, np.float64)
        n = len(spans)
        if n >= 10:
            k = max(2, int(round(0.28 * n)))
            end1 = float(np.percentile(spans[:k], 75))
            end2 = float(np.percentile(spans[-k:], 75))
            m0, m1 = int(round(0.35 * n)), int(round(0.70 * n))
            middle = float(np.median(spans[m0:max(m0 + 1, m1)]))
            end_bulge = max(end1, end2) / max(middle, 1.0)
        else:
            end_bulge = 1.0

        metrics = {
            "aspect": aspect,
            "split_frac": split_frac,
            "split_run": split_run,
            "end_bulge": end_bulge,
        }

        # Strong two-leg evidence: many rows contain two substantial separated runs,
        # and the split persists over a meaningful portion of the garment length.
        bottom_strength = float(np.clip(
            0.55 * (split_frac / 0.24) + 0.45 * (split_run / 0.20),
            0.0,
            1.0,
        ))
        if split_frac >= 0.16 and split_run >= 0.14 and bottom_strength >= 0.78:
            return "BOTTOM", bottom_strength, metrics

        # Shirt cue is deliberately weaker because sleeves/torso geometry varies a lot.
        # It is used as a consistency check rather than an aggressive override.
        upper_strength = 0.0
        if split_frac < 0.06 and 0.70 <= aspect <= 1.75 and end_bulge >= 1.42:
            upper_strength = float(np.clip((end_bulge - 1.20) / 0.45, 0.0, 0.88))
        if upper_strength >= 0.72:
            return "UPPER", upper_strength, metrics

        return None, max(bottom_strength * 0.45, upper_strength), metrics

    def _classify_ensemble(
        self,
        raw_crop: np.ndarray,
        masked_crop: np.ndarray,
    ) -> Tuple[float, float, str]:
        """V7 clean test path for the newly fine-tuned classifier.

        Use one frozen RGB garment crop with the exact 224x224 / 114 letterbox
        preprocessing.  Do not mix in masked crops, flips, rotations, or geometry
        overrides while validating the new model; those can be re-enabled later
        only if real-camera testing proves they help.
        """
        inp = self._letterbox_square(raw_crop, self.classify_imgsz, pad_value=114)
        out = self.classify_model.predict(
            source=inp,
            imgsz=self.classify_imgsz,
            verbose=False,
        )[0]
        up, bot, raw_label, raw_conf, _ = self._classification_probs(out)
        return float(up), float(bot), (
            f"V7_SINGLE_RGB raw={raw_label} conf={raw_conf:.3f} "
            f"U={up:.3f} B={bot:.3f}"
        )

    def _classification_probs(self, result: Any) -> Tuple[float, float, str, float, Dict[str, float]]:
        probs = getattr(result, "probs", None)
        data = getattr(probs, "data", None) if probs is not None else None
        if data is None:
            raise RuntimeError("TOP/BOTTOM classifier returned no probs")
        scores = self._to_numpy(data).astype(np.float64).reshape(-1)
        names = getattr(result, "names", {})
        upper = 0.0
        bottom = 0.0
        score_map: Dict[str, float] = {}
        for i, score in enumerate(scores):
            label = self._name(names, i)
            value = float(score)
            score_map[label] = value
            semantic = self._semantic_from_label(label)
            if semantic == "UPPER":
                upper += value
            elif semantic == "BOTTOM":
                bottom += value
        if upper <= 0.0 and bottom <= 0.0:
            raise RuntimeError(f"TOP/BOTTOM class names unsupported: {names!r}")
        top_idx = int(np.argmax(scores))
        top_label = self._name(names, top_idx)
        top_conf = float(scores[top_idx])
        return upper, bottom, top_label, top_conf, score_map

    def infer_presence(self, frame: np.ndarray) -> Tuple[bool, str]:
        """Segmentation-only garment presence check used after classification lock.

        This deliberately does NOT call the dedicated TOP/BOTTOM classifier.  Once a
        garment has already been locked as UPPER/BOTTOM, the only question is whether
        that same garment has been removed from the board so the next cycle may start.
        """
        self.ensure_model()
        self._presence_seq += 1
        seq = int(self._presence_seq)
        board_mask = self.roi.mask(frame.shape)
        t0 = time.monotonic()
        seg_result = self.segment_model.predict(
            source=frame,
            imgsz=self.segment_imgsz,
            conf=self.det_conf,
            verbose=False,
        )[0]
        t1 = time.monotonic()
        selected = self._select_garment(seg_result, frame, board_mask)
        present = selected is not None
        if seq <= 2 or seq % 10 == 0:
            print(
                f"[CLASS-PRESENCE] #{seq} present={present} seg={t1-t0:.3f}s {self._last_select_detail}",
                flush=True,
            )
        return present, f"present={present} seg={t1-t0:.3f}s {self._last_select_detail}"

    def infer_frame(self, frame: np.ndarray) -> FrameEvidence:
        self.ensure_model()
        self._infer_seq += 1
        seq = int(self._infer_seq)
        board_mask = self.roi.mask(frame.shape)

        t0 = time.monotonic()
        verbose_seg = bool(seq <= 3 or seq % 10 == 0)
        if verbose_seg:
            print(
                f"[CLASS-INFER] #{seq} SEG START preferred={self._orientation_name(self._seg_orientation)}",
                flush=True,
            )
        selected, seg_elapsed = self._segment_with_rotation_rescue(
            frame, board_mask, seq
        )
        t1 = time.monotonic()
        if verbose_seg:
            print(
                f"[CLASS-INFER] #{seq} SEG DONE {seg_elapsed:.3f}s "
                f"active={self._orientation_name(self._seg_orientation)}",
                flush=True,
            )

        if selected is None:
            if seq <= 3 or seq % 5 == 0:
                print(f"[CLASS-SEG-NO-GARMENT] #{seq} {self._last_select_detail}", flush=True)
            self.history.clear()
            self._reset_stability()
            return FrameEvidence(
                winner="EMPTY",
                detail=f"seg=no board-supported garment | {self._last_select_detail} | seg={seg_elapsed:.3f}s",
            )

        seg_score, seg_conf, area_ratio, overlap, seg_name, mask = selected

        stable, settle_detail = self._observe_stability(mask, frame.shape)
        if not stable:
            if seq <= 3 or seq % 5 == 0:
                print(f"[CLASS-SETTLE] #{seq} WAIT {settle_detail}", flush=True)
            return FrameEvidence(
                garment_score=float(seg_score),
                winner="SETTLING",
                detail=(
                    f"SETTLING {settle_detail} "
                    f"SEG={seg_name}:{seg_conf:.2f}/A{area_ratio:.3f}/R{overlap:.2f}"
                ),
            )

        raw_crop, masked_crop = self._fixed_crop_pair(frame, mask)

        t_cls0 = time.monotonic()
        upper_prob, bottom_prob, ensemble_detail = self._classify_ensemble(raw_crop, masked_crop)
        t2 = time.monotonic()

        # V7 intentionally evaluates the new classifier without heuristic overrides.
        geo_sem, geo_strength, geo_metrics = None, 0.0, {}
        best = max(upper_prob, bottom_prob)
        other = min(upper_prob, bottom_prob)
        ratio = best / max(other, 1e-6)
        winner = (
            "UPPER" if upper_prob > bottom_prob else "BOTTOM"
        ) if (best >= self.per_frame_min_conf and ratio >= self.per_frame_min_ratio) else "AMBIGUOUS"
        decision_note = "NEW_MODEL_NATIVE" if winner != "AMBIGUOUS" else "AMBIGUOUS"
        gm = ""
        print(
            f"[CLASS-V8] #{seq} stableGen={self._stable_generation} "
            f"U={upper_prob:.3f} B={bottom_prob:.3f} ratio={ratio:.2f} "
            f"geo={geo_sem}:{geo_strength:.2f} decision={winner}/{decision_note} "
            f"cls={t2-t_cls0:.3f}s",
            flush=True,
        )

        return FrameEvidence(
            upper_score=float(upper_prob),
            bottom_score=float(bottom_prob),
            upper_conf=float(upper_prob),
            bottom_conf=float(bottom_prob),
            garment_score=float(seg_score),
            winner=winner,
            detail=(
                f"V8 U={upper_prob:.3f} B={bottom_prob:.3f} ratio={ratio:.2f} "
                f"geo={geo_sem}:{geo_strength:.2f}({gm}) decision={decision_note} "
                f"SEG={seg_name}:{seg_conf:.2f}/A{area_ratio:.3f}/R{overlap:.2f} "
                f"SETTLE={settle_detail} TTA={ensemble_detail} "
                f"TIMING seg={seg_elapsed:.3f}s cls={t2-t_cls0:.3f}s"
            ),
        )

    def _snapshot_empty_board_diff_fallback(
        self,
        frame: np.ndarray,
        empty_board: Optional[np.ndarray],
        board_mask: np.ndarray,
        seq: int,
    ):
        """Mode-B-only fallback when semantic segmentation returns no garment.

        The current frozen corrected snapshot is compared against the active
        corrected empty-board baseline.  Only a sufficiently large connected
        foreground region inside the calibrated board ROI is accepted.

        This never runs in Mode A and never replaces a valid YOLO mask.
        """
        if empty_board is None or not isinstance(empty_board, np.ndarray) or empty_board.size == 0:
            print(
                f"[CLASS-SNAPSHOT-DIFF-SKIP] #{seq} no active empty-board baseline",
                flush=True,
            )
            return None

        if frame.shape[:2] != empty_board.shape[:2]:
            print(
                f"[CLASS-SNAPSHOT-DIFF-SKIP] #{seq} shape mismatch "
                f"frame={frame.shape[:2]} empty={empty_board.shape[:2]}",
                flush=True,
            )
            return None

        current = np.asarray(frame, np.uint8)
        empty = np.asarray(empty_board, np.uint8)
        roi = (np.asarray(board_mask, np.uint8) > 0).astype(np.uint8) * 255

        # Shrink the calibrated board ROI slightly so ArUco/board edges and
        # tiny camera-registration differences cannot dominate the fallback.
        erode_kernel = np.ones((11, 11), np.uint8)
        inner_roi = cv2.erode(roi, erode_kernel, iterations=1)
        board_area = max(1, int(cv2.countNonZero(inner_roi)))

        # Use maximum per-channel difference rather than grayscale only.
        # This keeps colored garments visible even when their luminance happens
        # to be similar to the empty board.
        diff_bgr = cv2.absdiff(current, empty)
        diff_max = np.max(diff_bgr, axis=2).astype(np.uint8)

        # The threshold deliberately matches the established V51 fallback
        # family (18) and is high enough to suppress normal sensor noise.
        diff_threshold = 18
        raw_mask = ((diff_max >= diff_threshold) & (inner_roi > 0)).astype(np.uint8) * 255

        # Suppress isolated sensor/reflection speckles, then join nearby cloth
        # regions.  All operations stay inside the calibrated board ROI.
        raw_mask = cv2.medianBlur(raw_mask, 5)
        open_kernel = np.ones((5, 5), np.uint8)
        close_kernel = np.ones((11, 11), np.uint8)
        clean = cv2.morphologyEx(raw_mask, cv2.MORPH_OPEN, open_kernel, iterations=1)
        clean = cv2.morphologyEx(clean, cv2.MORPH_CLOSE, close_kernel, iterations=2)
        clean = cv2.bitwise_and(clean, inner_roi)

        contours, _ = cv2.findContours(clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            mean_diff = float(cv2.mean(diff_max, mask=inner_roi)[0])
            print(
                f"[CLASS-SNAPSHOT-DIFF-MISS] #{seq} no contours "
                f"meanDiff={mean_diff:.2f} thr={diff_threshold}",
                flush=True,
            )
            return None

        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        largest = contours[0]
        contour_area = float(cv2.contourArea(largest))
        area_ratio = contour_area / float(board_area)

        # Mode B manually placed garments are expected to occupy a meaningful
        # portion of the board.  Reject tiny glare/shadow changes and nearly
        # whole-board illumination shifts.
        min_ratio = max(0.012, float(self.min_area_ratio))
        max_ratio = 0.72
        if area_ratio < min_ratio or area_ratio > max_ratio:
            mean_diff = float(cv2.mean(diff_max, mask=inner_roi)[0])
            print(
                f"[CLASS-SNAPSHOT-DIFF-REJECT] #{seq} areaRatio={area_ratio:.4f} "
                f"allowed={min_ratio:.4f}~{max_ratio:.2f} "
                f"meanDiff={mean_diff:.2f}",
                flush=True,
            )
            return None

        mask = np.zeros(frame.shape[:2], np.uint8)
        cv2.drawContours(mask, [largest], -1, 255, thickness=-1)
        mask = cv2.bitwise_and(mask, inner_roi)

        inside = mask > 0
        if int(np.count_nonzero(inside)) < 350:
            print(
                f"[CLASS-SNAPSHOT-DIFF-REJECT] #{seq} mask too small after fill",
                flush=True,
            )
            return None

        candidate_mean = float(np.mean(diff_max[inside]))
        if candidate_mean < float(diff_threshold):
            print(
                f"[CLASS-SNAPSHOT-DIFF-REJECT] #{seq} candidateMean={candidate_mean:.2f} "
                f"< threshold={diff_threshold}",
                flush=True,
            )
            return None

        # Keep the tuple compatible with the semantic selection path.
        score = math.sqrt(max(area_ratio, 1e-9))
        overlap = 1.0
        print(
            f"[CLASS-SNAPSHOT-DIFF-HIT] #{seq} areaRatio={area_ratio:.4f} "
            f"meanDiff={candidate_mean:.2f} thr={diff_threshold} "
            f"pixels={int(np.count_nonzero(mask))}",
            flush=True,
        )
        return (
            float(score),
            0.0,
            float(area_ratio),
            float(overlap),
            "fallback_diff",
            mask,
        )

    def infer_snapshot(
        self,
        frame: np.ndarray,
        *,
        empty_board: Optional[np.ndarray] = None,
    ) -> FrameEvidence:
        """Classify one frozen/median snapshot without temporal geometry settling.

        Mode B uses a camera-stillness gate *before* this call, so re-running the
        live-mask settle state machine here would reintroduce the same oscillation
        this path is meant to avoid.  Segmentation and TOP/BOTTOM classification
        therefore consume this exact same frozen image.
        """
        self.ensure_model()
        self._infer_seq += 1
        seq = int(self._infer_seq)
        board_mask = self.roi.mask(frame.shape)

        t0 = time.monotonic()
        print(
            f"[CLASS-SNAPSHOT-INFER] #{seq} SEG START preferred={self._orientation_name(self._seg_orientation)}",
            flush=True,
        )
        selected, seg_elapsed = self._segment_with_rotation_rescue(
            frame, board_mask, seq
        )
        if selected is None:
            semantic_detail = str(self._last_select_detail)
            print(
                f"[CLASS-SNAPSHOT-NO-GARMENT] #{seq} {semantic_detail}",
                flush=True,
            )
            selected = self._snapshot_empty_board_diff_fallback(
                frame,
                empty_board,
                board_mask,
                seq,
            )
            if selected is None:
                return FrameEvidence(
                    winner="EMPTY",
                    detail=(
                        "snapshot seg=no board-supported garment; "
                        "fallback_diff=MISS | "
                        f"{semantic_detail} | seg={seg_elapsed:.3f}s"
                    ),
                )
            print(
                f"[CLASS-SNAPSHOT-FALLBACK-DIFF] #{seq} "
                "semantic segmentation failed; active empty-board diff accepted",
                flush=True,
            )

        seg_score, seg_conf, area_ratio, overlap, seg_name, mask = selected
        raw_crop, masked_crop = self._snapshot_crop_pair(frame, mask)
        t_cls0 = time.monotonic()
        upper_prob, bottom_prob, ensemble_detail = self._classify_ensemble(
            raw_crop, masked_crop
        )
        t2 = time.monotonic()

        best = max(upper_prob, bottom_prob)
        other = min(upper_prob, bottom_prob)
        ratio = best / max(other, 1e-6)
        winner = (
            "UPPER" if upper_prob > bottom_prob else "BOTTOM"
        ) if (
            best >= self.per_frame_min_conf
            and ratio >= self.per_frame_min_ratio
        ) else "AMBIGUOUS"

        print(
            f"[CLASS-SNAPSHOT-V8] #{seq} U={upper_prob:.3f} B={bottom_prob:.3f} "
            f"ratio={ratio:.2f} decision={winner} seg={seg_elapsed:.3f}s "
            f"cls={t2-t_cls0:.3f}s",
            flush=True,
        )
        return FrameEvidence(
            upper_score=float(upper_prob),
            bottom_score=float(bottom_prob),
            upper_conf=float(upper_prob),
            bottom_conf=float(bottom_prob),
            garment_score=float(seg_score),
            winner=winner,
            detail=(
                f"SNAPSHOT U={upper_prob:.3f} B={bottom_prob:.3f} ratio={ratio:.2f} "
                f"SEG={seg_name}:{seg_conf:.2f}/A{area_ratio:.3f}/R{overlap:.2f} "
                f"TTA={ensemble_detail} TIMING seg={seg_elapsed:.3f}s cls={t2-t_cls0:.3f}s"
            ),
        )

    def push(self, evidence: FrameEvidence) -> ClassificationResult:
        # Empty or still-moving garments must not poison temporal consensus.
        if evidence.winner == "EMPTY":
            self.history.clear()
            return ClassificationResult(
                "AMBIGUOUS", 0.0, 0, 0, 0, "board empty / waiting for garment"
            )
        if evidence.winner == "SETTLING":
            return ClassificationResult(
                "AMBIGUOUS", 0.0, 0, 0, 0, "waiting for stable garment geometry"
            )

        self.history.append(evidence)
        hist = list(self.history)
        upper_votes = sum(1 for x in hist if x.winner == "UPPER")
        bottom_votes = sum(1 for x in hist if x.winner == "BOTTOM")
        semantic_votes = upper_votes + bottom_votes
        samples = len(hist)

        if samples < self.min_votes:
            return ClassificationResult(
                "AMBIGUOUS", 0.0, upper_votes, bottom_votes, samples,
                f"stable probability warmup {samples}/{self.min_votes}",
            )

        mean_upper = float(np.mean([x.upper_score for x in hist]))
        mean_bottom = float(np.mean([x.bottom_score for x in hist]))
        winner = "UPPER" if mean_upper >= mean_bottom else "BOTTOM"
        winner_prob = max(mean_upper, mean_bottom)
        other_prob = min(mean_upper, mean_bottom)
        probability_margin = winner_prob / max(other_prob, 1e-6)

        winner_votes = upper_votes if winner == "UPPER" else bottom_votes
        vote_share = float(winner_votes) / float(max(1, semantic_votes))
        last3 = [x.winner for x in hist[-3:]]
        consecutive_ok = len(last3) == 3 and all(x == winner for x in last3)

        # Probability mean is primary in V6.  Vote count remains a consistency gate.
        accepted = bool(
            semantic_votes >= self.min_votes
            and winner_votes >= max(3, self.min_votes - 1)
            and vote_share >= self.min_vote_share
            and winner_prob >= self.min_mean_conf
            and probability_margin >= self.min_score_margin
            and consecutive_ok
        )

        confidence = float(np.clip(
            0.68 * winner_prob + 0.32 * vote_share,
            0.0,
            1.0,
        ))
        detail = (
            f"V8 meanProb U={mean_upper:.3f} B={mean_bottom:.3f} "
            f"votes U={upper_votes} B={bottom_votes} share={vote_share:.2f} "
            f"margin={probability_margin:.2f} last3={','.join(last3)}"
        )
        return ClassificationResult(
            winner if accepted else "AMBIGUOUS",
            confidence if accepted else 0.0,
            upper_votes,
            bottom_votes,
            samples,
            detail,
        )

    def reset_history(self) -> None:
        self.history.clear()
        self._reset_stability()


# -----------------------------------------------------------------------------
# V22 shared-camera integration runtime
# -----------------------------------------------------------------------------
DUAL_ROOT = HERE.parent
JETSON_ROOT = DUAL_ROOT.parent
MODEL_ROOT = JETSON_ROOT / "models"
COMMON_ROOT = JETSON_ROOT / "common"
if str(DUAL_ROOT) not in sys.path:
    sys.path.insert(0, str(DUAL_ROOT))

from shared_empty_board_state import EmptyBoardState

DEFAULT_TRANSFER_MODULE = DUAL_ROOT / "garment_transfer.py"
DEFAULT_UPPER_PROCESS_MODULE = DUAL_ROOT / "upper_process.py"
DEFAULT_UPPER_VLA_PROCESS_MODULE = DUAL_ROOT / "upper_vla_process.py"
DEFAULT_BOTTOM_PROCESS_MODULE = DUAL_ROOT / "bottom_process.py"
DEFAULT_BOTTOM_RUNTIME = DUAL_ROOT / "bottom" / "bottom_vla_runtime.py"
DEFAULT_SHARED_EMPTY_BOARD_STATE = DUAL_ROOT / "shared_empty_board_state.py"
DEFAULT_EMPTY_BOARD = DUAL_ROOT / "d34_empty_board.png"
# Final upper V51 uses the MICRO 7-class Action Model.  The deployed filename below
# currently contains the validated V3 light-augmentation FP32 TensorRT engine.
# CLI --integration-upper-vla-model / --integration-upper-vla-state-norm still override these defaults.
DEFAULT_UPPER_VLA_ENGINE = Path(
    MODEL_ROOT / "decision/upper/top_board_state_micro_v2/top_board_state_micro_v2_fp32.engine"
)
DEFAULT_UPPER_VLA_STATE_NORM = Path(
    MODEL_ROOT / "decision/upper/top_board_state_micro_v2/state_normalization.npz"
)
DEFAULT_MACHINE_CONFIG = Path(
    COMMON_ROOT / "calibration/dual_roarm_folding_board_config.json"
)
DEFAULT_HFILE = (
    COMMON_ROOT / "calibration/elp_ov2710_folding_board_homography_cache.json"
)
DEFAULT_BASKET_CALIB = Path(
    COMMON_ROOT / "calibration/basket_arm2_5point_affine.json"
)
DEFAULT_CAMERA_CALIB = Path(
    COMMON_ROOT / "camera/elp_ov2710_1280x720_calibration.npz"
)
DEFAULT_SEG_ENGINE_INTEGRATION = Path(
    MODEL_ROOT / "segmentation/kfashion_yolo26s_seg3_e100_best.engine"
)
DEFAULT_UPPER_POSE_ENGINE = Path(
    MODEL_ROOT / "pose/upper/tshirt_pose_yolo26m_synth_artf_board_v1_best.engine"
)
DEFAULT_BOTTOM_POSE_ENGINE = Path(
    MODEL_ROOT / "pose/lower/bottom_pose8_beige_finetune_v2_best.engine"
)

def _load_integration_ui_camera_session(owner: Any) -> Dict[str, Any]:
    """Read V22's existing Saved button JSON without changing the camera.

    This is run on the UI thread when Integrated Start is pressed.  The immutable
    snapshot is consumed by the FIX46 model-warmup reassertions.  No second
    camera instance or control writer is constructed here.
    """
    path = getattr(owner, "camera_control_config_path", None)
    if path is None:
        return {"source": "EMBEDDED_DEFAULT", "controls": {}, "extra_controls": {},
                "profile_sha256": None, "profile_path": None}
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        return {"source": "EMBEDDED_DEFAULT", "controls": {}, "extra_controls": {},
                "profile_sha256": None, "profile_path": str(path)}
    data = path.read_bytes()
    try:
        payload = json.loads(data.decode("utf-8"))
    except Exception as exc:
        raise RuntimeError(f"saved V24 camera profile is invalid ({path}): {exc}") from exc
    if not isinstance(payload, dict) or not isinstance(payload.get("controls"), dict):
        raise RuntimeError(f"saved V24 camera profile has no valid controls object: {path}")
    values = payload["controls"]
    names = dict(getattr(owner, "camera_control_names", {}) or {})
    names_by_target = {
        "auto_exposure": names.get("exposure_mode"),
        "exposure_time_absolute": names.get("exposure"),
        "gain": names.get("gain"),
        "power_line_frequency": names.get("power_line"),
        "white_balance_automatic": names.get("auto_wb"),
        "white_balance_temperature": names.get("wb_temperature"),
    }
    aliases = {
        "auto_exposure": ("auto_exposure", "exposure_auto"),
        "exposure_time_absolute": ("exposure_time_absolute", "exposure_absolute"),
        "gain": ("gain",),
        "power_line_frequency": ("power_line_frequency",),
        "white_balance_automatic": ("white_balance_automatic",),
        "white_balance_temperature": ("white_balance_temperature",),
    }
    normalized: Dict[str, int] = {}
    missing = []
    for target, detected_name in names_by_target.items():
        candidates = [detected_name] if detected_name else []
        candidates.extend(aliases[target])
        found = next((candidate for candidate in candidates if candidate in values), None)
        if found is None:
            missing.append(target)
            continue
        value = values[found]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or int(value) != value:
            raise RuntimeError(f"saved camera {target} is not an integer: {value!r}")
        normalized[target] = int(value)
    if missing:
        raise RuntimeError(f"saved V24 camera profile is incomplete ({path}); "
                           f"missing={missing}; open Camera Settings and Save again")
    if normalized["auto_exposure"] != 1 or normalized["white_balance_automatic"] != 0:
        raise RuntimeError("saved profile requires fixed/manual exposure (1) and AWB OFF (0). "
                           "Run Auto Fit, wait until fixed, and press Save")
    if normalized["exposure_time_absolute"] <= 0 or normalized["gain"] < 0 or normalized["white_balance_temperature"] <= 0:
        raise RuntimeError("saved exposure/gain/white balance has invalid values")
    extra: Dict[str, int] = {}
    auto_gain_name = names.get("auto_gain")
    if auto_gain_name and auto_gain_name in values:
        auto_gain_value = values[auto_gain_name]
        if isinstance(auto_gain_value, bool) or auto_gain_value != 0:
            raise RuntimeError("saved V24 auto_gain must be OFF before integrated run")
        if auto_gain_name not in normalized:
            extra[auto_gain_name] = 0

    # V24 training-match SW-AE explicitly fixes these two OV2710 controls to 0.
    # Do not synthesize values: if the active device exposes a control, require
    # the exact value saved by V24's "현재 설정 저장" flow.
    for key, detected_name, aliases_for_key in (
        ("backlight_compensation", names.get("backlight_compensation"), ("backlight_compensation",)),
        ("exposure_dynamic_framerate", names.get("exposure_dynamic_framerate"), ("exposure_dynamic_framerate",)),
    ):
        if not detected_name:
            continue
        candidates = [detected_name]
        candidates.extend(aliases_for_key)
        found = next((candidate for candidate in candidates if candidate in values), None)
        if found is None:
            raise RuntimeError(
                f"saved V24 camera profile is incomplete ({path}); missing={key}; "
                "open Camera Settings, run brightness auto-fit, then Save again"
            )
        value = values[found]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or int(value) != value:
            raise RuntimeError(f"saved camera {key} is not an integer: {value!r}")
        value_i = int(value)
        if value_i != 0:
            raise RuntimeError(
                f"saved V24 {key} must remain 0 for training-match exposure; got {value_i}"
            )
        extra[found] = value_i
    saved_device = str(payload.get("device") or "").strip()
    active_device = str(getattr(getattr(owner, "v4l2_controls", None), "device", "")).strip()
    def canonical_dev(value: str) -> str:
        return f"/dev/video{value}" if value.isdigit() else value
    if saved_device and active_device and canonical_dev(saved_device) != canonical_dev(active_device):
        raise RuntimeError(f"camera profile device mismatch saved={saved_device} active={active_device}")
    return {"source": "V24_UI_SAVED", "controls": normalized, "extra_controls": extra,
            "profile_sha256": hashlib.sha256(data).hexdigest(), "profile_path": str(path)}


class V22SharedRawCameraProvider:
    """Raw-frame bridge over V22's one and only physical VideoCapture.

    The PySide6/V22 camera timer remains the sole reader of /dev/video0.  Robot
    modules receive immutable RAW frame copies through the historical FIX11
    RawPreviewController-compatible interface.  Overlay calls are intentionally
    ignored so the visible V22 camera page never changes to grasp/mask overlays.
    """

    def __init__(self, owner: "JerberMainController") -> None:
        self.owner = owner
        # Match the FIX11 RawPreviewController object shape because the existing
        # validated transfer/D50 code accesses several attributes directly.
        self.args = owner.args
        self.window_name = "V22 UI Shared Camera"
        self._lock = threading.RLock()
        self._condition = threading.Condition(self._lock)
        self._stop = threading.Event()
        self._frame: Optional[np.ndarray] = None
        self._frame_seq = 0
        self._frame_time = 0.0
        self._keys: Deque[int] = collections.deque(maxlen=128)
        self._status = "V22 SHARED CAMERA"
        self._override = None
        self._override_status = ""
        self._error: Optional[BaseException] = None
        self._camera_info: Dict[str, Any] = {}
        self._started = threading.Event()
        self._started.set()
        # CameraPumpLease checks preview._thread.is_alive().  The provider itself
        # supplies that tiny compatibility contract; there is no second thread.
        self._thread = self
        self._shutdown = False

        # MAIN-LATENCY-DIAG-V1: integration-only timing telemetry.  These fields
        # are read/written under _lock and never influence camera ownership, frame
        # delivery, robot motion, planner decisions, or safety gates.
        self._latency_trace_active = False
        self._latency_trace_label = ""
        self._latency_trace_started_mono = 0.0
        self._latency_trace_last_publish_mono = 0.0
        self._latency_trace_publish_count = 0
        self._latency_trace_publish_gap_sum_s = 0.0
        self._latency_trace_publish_gap_max_s = 0.0
        self._latency_trace_read_count = 0
        self._latency_trace_read_wait_sum_s = 0.0
        self._latency_trace_read_wait_max_s = 0.0
        self._latency_trace_update_count = 0
        self._latency_trace_partial_update_count = 0
        self._latency_trace_stage_sum = collections.defaultdict(float)
        self._latency_trace_stage_max = collections.defaultdict(float)

    def is_alive(self) -> bool:
        if self._shutdown:
            return False
        cap = getattr(self.owner, "cap", None)
        try:
            return cap is not None and bool(cap.isOpened())
        except Exception:
            return cap is not None

    def publish(self, raw_frame: Optional[np.ndarray]) -> None:
        if raw_frame is None or not isinstance(raw_frame, np.ndarray) or raw_frame.size == 0:
            return
        now_mono = time.monotonic()
        slow_gap = None
        with self._condition:
            self._frame = np.asarray(raw_frame, dtype=np.uint8).copy()
            self._frame_seq += 1
            self._frame_time = time.time()
            if self._latency_trace_active:
                previous = float(self._latency_trace_last_publish_mono or 0.0)
                if previous > 0.0:
                    gap = max(0.0, now_mono - previous)
                    self._latency_trace_publish_gap_sum_s += gap
                    self._latency_trace_publish_gap_max_s = max(
                        float(self._latency_trace_publish_gap_max_s), gap
                    )
                    if gap >= 0.55:
                        slow_gap = gap
                self._latency_trace_last_publish_mono = now_mono
                self._latency_trace_publish_count += 1
            self._condition.notify_all()
        if slow_gap is not None:
            self.latency_event(
                "PUBLISH_GAP",
                gap_s=slow_gap,
                frame_seq=int(self._frame_seq),
            )

    @staticmethod
    def _latency_fmt(value: Any) -> str:
        if isinstance(value, float):
            return f"{value:.4f}"
        return str(value).replace(" ", "_")

    def begin_latency_trace(self, label: str) -> None:
        now = time.monotonic()
        with self._lock:
            self._latency_trace_active = True
            self._latency_trace_label = str(label)
            self._latency_trace_started_mono = now
            self._latency_trace_last_publish_mono = 0.0
            self._latency_trace_publish_count = 0
            self._latency_trace_publish_gap_sum_s = 0.0
            self._latency_trace_publish_gap_max_s = 0.0
            self._latency_trace_read_count = 0
            self._latency_trace_read_wait_sum_s = 0.0
            self._latency_trace_read_wait_max_s = 0.0
            self._latency_trace_update_count = 0
            self._latency_trace_partial_update_count = 0
            self._latency_trace_stage_sum = collections.defaultdict(float)
            self._latency_trace_stage_max = collections.defaultdict(float)
        print(f"[MAIN-LATENCY] event=TRACE_BEGIN label={str(label)}", flush=True)

    def latency_event(self, event: str, **fields: Any) -> None:
        checkpoint: Dict[str, Any] = {}
        with self._lock:
            if not self._latency_trace_active:
                return
            elapsed = max(0.0, time.monotonic() - float(self._latency_trace_started_mono))
            label = str(self._latency_trace_label)
            if str(event) == "V51_BURST_COMPLETE":
                checkpoint = {
                    "main_publish_gap_max_s": float(self._latency_trace_publish_gap_max_s),
                    "main_shared_wait_max_s": float(self._latency_trace_read_wait_max_s),
                    "main_camera_update_max_s": float(self._latency_trace_stage_max.get("main_update_s", 0.0)),
                    "main_camera_updates": int(self._latency_trace_update_count),
                }
        parts = [
            "[MAIN-LATENCY]",
            f"event={str(event)}",
            f"trace={label}",
            f"t_rel_s={elapsed:.4f}",
        ]
        merged_fields = dict(fields)
        merged_fields.update(checkpoint)
        for key, value in merged_fields.items():
            parts.append(f"{key}={self._latency_fmt(value)}")
        print(" ".join(parts), flush=True)

    def note_shared_frame_read(self, *, wait_s: float, seq: int, source_age_s: float) -> None:
        wait_s = max(0.0, float(wait_s))
        source_age_s = max(0.0, float(source_age_s))
        with self._lock:
            if not self._latency_trace_active:
                return
            self._latency_trace_read_count += 1
            self._latency_trace_read_wait_sum_s += wait_s
            self._latency_trace_read_wait_max_s = max(self._latency_trace_read_wait_max_s, wait_s)
        # Normal ~3 FPS reads are expected around 0.3 s.  Log only abnormal waits
        # individually; the complete mean/max is printed in TRACE_END.
        if wait_s >= 0.45 or source_age_s >= 0.20:
            self.latency_event(
                "SHARED_FRAME_WAIT",
                wait_s=wait_s,
                frame_seq=int(seq),
                source_age_s=source_age_s,
            )

    def note_camera_update(self, timing: Dict[str, Any], publish_s: float) -> None:
        fields = {
            "read_s": float(timing.get("read_s", 0.0) or 0.0),
            "raw_copy_s": float(timing.get("raw_copy_s", 0.0) or 0.0),
            "undistort_s": float(timing.get("undistort_s", 0.0) or 0.0),
            "corrected_copy_s": float(timing.get("corrected_copy_s", 0.0) or 0.0),
            "ui_s": float(timing.get("ui_s", 0.0) or 0.0),
            "aruco_s": float(timing.get("aruco_s", 0.0) or 0.0),
            "base_total_s": float(timing.get("total_s", 0.0) or 0.0),
            "publish_s": max(0.0, float(publish_s)),
        }
        total_s = fields["base_total_s"] + fields["publish_s"]
        # Failed/partial ticks have their own diagnostics, never normal means/maxima.
        partial = not timing.get("ok", False) or bool(timing.get("partial") or timing.get("publish_error"))
        with self._lock:
            if not self._latency_trace_active:
                return
            if partial:
                self._latency_trace_partial_update_count += 1
            else:
                self._latency_trace_update_count += 1
                for key, value in fields.items():
                    self._latency_trace_stage_sum[key] += value
                    self._latency_trace_stage_max[key] = max(self._latency_trace_stage_max[key], value)
                self._latency_trace_stage_sum["main_update_s"] += total_s
                self._latency_trace_stage_max["main_update_s"] = max(
                    self._latency_trace_stage_max["main_update_s"], total_s
                )
        if partial:
            self.latency_event(
                "PARTIAL_CAMERA_UPDATE", total_s=total_s, ok=bool(timing.get("ok")),
                partial=bool(timing.get("partial")), error_stage=timing.get("error_stage", "none"),
                error_type=timing.get("error_type", "none"), publish_state=timing.get("publish_state", "not_attempted"),
                publish_error=timing.get("publish_error", "none"), **fields,
            )
            return
        if total_s >= 0.55:
            self.latency_event("SLOW_CAMERA_UPDATE", total_s=total_s, **fields)

    def end_latency_trace(self, label: str = "") -> None:
        with self._lock:
            if not self._latency_trace_active:
                return
            elapsed = max(0.0, time.monotonic() - float(self._latency_trace_started_mono))
            trace_label = str(label or self._latency_trace_label)
            publish_count = int(self._latency_trace_publish_count)
            publish_intervals = max(0, publish_count - 1)
            publish_gap_mean = (
                float(self._latency_trace_publish_gap_sum_s) / float(publish_intervals)
                if publish_intervals > 0 else 0.0
            )
            read_count = int(self._latency_trace_read_count)
            read_mean = (
                float(self._latency_trace_read_wait_sum_s) / float(read_count)
                if read_count > 0 else 0.0
            )
            update_count = int(self._latency_trace_update_count)
            partial_update_count = int(self._latency_trace_partial_update_count)
            stage_sum = dict(self._latency_trace_stage_sum)
            stage_max = dict(self._latency_trace_stage_max)
            publish_gap_max = float(self._latency_trace_publish_gap_max_s)
            shared_wait_max = float(self._latency_trace_read_wait_max_s)
            self._latency_trace_active = False
        def mean_stage(name: str) -> float:
            return float(stage_sum.get(name, 0.0)) / float(update_count) if update_count > 0 else 0.0
        print(
            "[MAIN-LATENCY-SUMMARY] "
            f"trace={trace_label} elapsed_s={elapsed:.3f} "
            f"published={publish_count} publish_gap_mean_s={publish_gap_mean:.3f} "
            f"publish_gap_max_s={publish_gap_max:.3f} "
            f"shared_reads={read_count} shared_wait_mean_s={read_mean:.3f} "
            f"shared_wait_max_s={shared_wait_max:.3f} "
            f"camera_updates={update_count} "
            f"partial_camera_updates={partial_update_count} "
            f"read_mean_s={mean_stage('read_s'):.3f} read_max_s={float(stage_max.get('read_s', 0.0)):.3f} "
            f"undistort_mean_s={mean_stage('undistort_s'):.3f} undistort_max_s={float(stage_max.get('undistort_s', 0.0)):.3f} "
            f"ui_mean_s={mean_stage('ui_s'):.3f} ui_max_s={float(stage_max.get('ui_s', 0.0)):.3f} "
            f"publish_mean_s={mean_stage('publish_s'):.3f} publish_max_s={float(stage_max.get('publish_s', 0.0)):.3f} "
            f"main_update_mean_s={mean_stage('main_update_s'):.3f} main_update_max_s={float(stage_max.get('main_update_s', 0.0)):.3f}",
            flush=True,
        )

    def start(self, timeout: float = 12.0) -> None:
        if not self.is_alive():
            raise RuntimeError("V22 shared camera provider start requested while V22 camera is not open")
        print(
            "[MAIN-SHARED-CAMERA] start=no-op; V22 already owns and reads /dev/video0",
            flush=True,
        )

    def stop(self, timeout: float = 3.0) -> None:
        # Intentionally no-op.  Only application cleanup may close V22's camera.
        print(
            "[MAIN-SHARED-CAMERA] stop=no-op; V22 camera remains OPEN/LIVE",
            flush=True,
        )

    def shutdown(self) -> None:
        with self._condition:
            self._shutdown = True
            self._stop.set()
            self._condition.notify_all()

    def current_sequence(self) -> int:
        with self._lock:
            return int(self._frame_seq)

    def wait_for_new_frame(self, last_sequence: int, timeout: float = 2.0):
        deadline = time.time() + max(0.05, float(timeout))
        with self._condition:
            while True:
                if self._shutdown:
                    raise RuntimeError("V22 shared camera provider is shutting down")
                if self._error is not None:
                    raise RuntimeError(f"V22 shared camera error: {self._error!r}")
                if self._frame is not None and int(self._frame_seq) > int(last_sequence):
                    return self._frame.copy(), int(self._frame_seq), float(self._frame_time)
                remain = deadline - time.time()
                if remain <= 0:
                    raise TimeoutError(
                        f"V22 shared camera newer-frame timeout last={last_sequence} current={self._frame_seq}"
                    )
                self._condition.wait(timeout=min(0.10, remain))

    def camera_info(self) -> Dict[str, Any]:
        frame = None
        with self._lock:
            if self._frame is not None:
                frame = self._frame
        if frame is not None:
            h, w = frame.shape[:2]
        else:
            w = int(getattr(self.owner.args, "camera_width", 1280))
            h = int(getattr(self.owner.args, "camera_height", 720))
        info = {
            "actual_size": [int(w), int(h)],
            "fps": float(getattr(self.owner.args, "fps", 9.0)),
            "fourcc": str(getattr(self.owner.args, "camera_fourcc", "YUYV")),
            "source": "V22_UI_SOLE_CAMERA_OWNER",
        }
        with self._lock:
            self._camera_info = dict(info)
            return dict(self._camera_info)

    def integration_camera_controls_metadata(self) -> Dict[str, Any]:
        """Read-only camera-control provenance for integrated consumers.

        This never writes V4L2 controls.  V22 remains the only camera/config owner.
        """
        actual: Dict[str, Any] = {}
        owner = self.owner
        try:
            names_fn = getattr(owner, "_relevant_camera_control_names", None)
            controls = getattr(owner, "v4l2_controls", None)
            if callable(names_fn) and controls is not None and hasattr(controls, "snapshot"):
                actual = dict(controls.snapshot(names_fn()) or {})
        except Exception as exc:
            actual = {"snapshot_error": repr(exc)}
        profile = getattr(owner, "camera_control_config_path", None)
        session = dict(getattr(owner, "_integration_ui_camera_session", None) or {})
        return {
            "enabled": True,
            "source": session.get("source", "V22_UI_SOLE_CAMERA_OWNER"),
            "device": str(getattr(getattr(owner, "v4l2_controls", None), "device", self.owner.args.camera)),
            "profile_path": session.get("profile_path", None if profile is None else str(profile)),
            "profile_sha256": session.get("profile_sha256"),
            "declared_controls": dict(session.get("controls") or {}),
            "extra_controls": dict(session.get("extra_controls") or {}),
            "actual_controls": actual,
            "mismatches": [],
        }

    def set_status(self, status: str) -> None:
        with self._lock:
            self._status = str(status)

    def show_override(self, frame: np.ndarray, status: str = "") -> None:
        with self._lock:
            if isinstance(frame, np.ndarray) and frame.size > 0:
                self._override = frame.copy()
            else:
                self._override = None
            self._override_status = str(status)
            if status:
                self._status = str(status)

    def override_frame_copy(self) -> Optional[np.ndarray]:
        with self._lock:
            if isinstance(self._override, np.ndarray) and self._override.size > 0:
                return self._override.copy()
        return None

    def clear_override(self) -> None:
        with self._lock:
            self._override = None
            self._override_status = ""

    def pop_key(self) -> int:
        with self._lock:
            if self._keys:
                return int(self._keys.popleft())
        return 255


class WorkflowSignals(QObject):
    ui = Signal(object)
    log = Signal(str)


class JerberMainController(v22.MaintenanceUiController):
    def __init__(self, args: argparse.Namespace, app: QApplication) -> None:
        self.workflow_signals = WorkflowSignals()
        self.workflow_stop = threading.Event()
        self.workflow_thread: Optional[threading.Thread] = None
        self.current_garment: Optional[str] = None
        self._main_closing = False
        self._classifier: Optional[TemporalGarmentClassifier] = None
        self._classifier_lock = threading.RLock()
        self._active_cycle = False
        self._runtime_modules: Dict[str, Any] = {}
        self._active_upper_vla_lock = threading.RLock()
        self._active_upper_vla_process: Optional[Any] = None
        self._active_bottom_lock = threading.RLock()
        self._active_bottom_process: Optional[Any] = None

        # V35 display-only macro telemetry + 7-minute optional force-fold watch.
        # This is separate from the normal workflow_stop button: choosing
        # force-fold cooperatively stops only the active VLA and, after its safe
        # cleanup/standby contract succeeds, continues into the existing garment-
        # specific post-FINISH folding sequence without fabricating a FINISH label.
        self._vla_watch_lock = threading.RLock()
        self._vla_watch_kind: Optional[str] = None
        self._vla_watch_started_monotonic: Optional[float] = None
        self._vla_timeout_prompt_shown = False
        self._force_fold_requested = threading.Event()
        self._force_fold_kind: Optional[str] = None
        self._force_fold_stop_completed_kind: Optional[str] = None
        self._last_ui_action_text = ""

        # Starts empty: camera settings are chosen when the operator presses
        # Integrated Start, AFTER any UI Auto Fit / Save on this application.
        self._integration_ui_camera_session: Optional[Dict[str, Any]] = None

        # One-shot startup empty-board RAW reference.
        #
        # This frame is captured only once, after the V22 camera has already been
        # opened/configured and after a short fresh-frame stabilization period.
        # It is intentionally kept as RAW BGR and is never overwritten during the
        # application lifetime.  Future VLA integration will consume a COPY of this
        # frame and reproduce the colleague upper.py E-key baseline construction.
        self._initial_empty_board_lock = threading.RLock()
        self._initial_empty_board_ready = threading.Event()
        self._initial_empty_board_capture_thread: Optional[threading.Thread] = None
        self._initial_empty_board_capture_error: Optional[str] = None
        self.initial_empty_board_raw_frame: Optional[np.ndarray] = None
        self.initial_empty_board_raw_sequence: Optional[int] = None
        self.initial_empty_board_raw_time: Optional[float] = None
        self.initial_empty_board_state: Optional[EmptyBoardState] = None
        # Startup baseline remains immutable for diagnostics.  The active baseline
        # is cycle-scoped and may be refreshed after folding or manually from the
        # main screen.  _final_shared_empty_board_state is kept as the compatibility
        # field consumed by existing upper/lower handoff code.
        self._final_shared_empty_board_state: Optional[EmptyBoardState] = None
        self._empty_board_refresh_lock = threading.RLock()
        self._integration_plate_lock = threading.RLock()
        self._integration_plate_busy = False

        # Demo workflow mode.  Mode A keeps the original basket/transfer pipeline.
        # Mode B assumes the operator has already placed/crumpled one garment on
        # the board and therefore starts at TOP/BOTTOM classification.
        self.demo_mode = "A"
        self._cycle_demo_mode = "A"

        super().__init__(args, app)
        self.args = args

        # Create the bridge only after V22 has opened/configured the real camera.
        self.shared_raw_camera = V22SharedRawCameraProvider(self)
        if isinstance(getattr(self, "latest_raw_camera_frame", None), np.ndarray):
            self.shared_raw_camera.publish(self.latest_raw_camera_frame)

        self.transfer_module_path = Path(args.garment_transfer_module).expanduser().resolve()
        self.upper_process_module_path = Path(args.upper_process_module).expanduser().resolve()
        self.upper_vla_process_module_path = Path(args.upper_vla_process_module).expanduser().resolve()
        self.bottom_process_module_path = Path(args.bottom_process_module).expanduser().resolve()
        self.roi_projector = BoardRoiProjector(
            Path(self.args.hfile),
            self.machine_cfg,
            allow_full_frame=bool(args.allow_full_frame_classification),
        )

        self.detected_caption = self.window.findChild(QLabel, "detectedCaption")
        self.detected_value = self.window.findChild(QLabel, "detectedValue")
        self.time_caption = self.window.findChild(QLabel, "timeCaption")
        self.time_value = self.window.findChild(QLabel, "timeValue")
        if self.detected_caption is not None:
            self.detected_caption.setText("현재 작업 의류")
        if self.time_caption is not None:
            self.time_caption.setText("통합 상태")
        if self.time_value is not None:
            self.time_value.setText("카메라 LIVE · 대기")

        # Keep the demo UI compact: one current-action line, never an action log.
        self.action_caption = QLabel("현재 선택 동작")
        self.action_caption.setObjectName("currentActionCaption")
        self.action_value = QLabel("-")
        self.action_value.setObjectName("currentActionValue")
        self.action_value.setWordWrap(False)
        self.timeout_prompt_label = QLabel(
            "7분이 넘게 소요되고 있습니다. 중단하고 옷을 접을까요?"
        )
        self.timeout_prompt_label.setObjectName("vlaTimeoutPromptLabel")
        self.timeout_prompt_label.setWordWrap(True)
        self.timeout_prompt_label.setVisible(False)
        self.timeout_fold_button = QPushButton("중단 후 접기")
        self.timeout_fold_button.setObjectName("vlaTimeoutFoldButton")
        self.timeout_fold_button.setMinimumHeight(
            42 if str(getattr(self, "display_profile", "desktop")) == "7ep" else 52
        )
        self.timeout_fold_button.setVisible(False)
        self.timeout_fold_button.setEnabled(False)
        self.timeout_fold_button.clicked.connect(self._request_timeout_force_fold)

        info_parent = self.detected_caption.parentWidget() if self.detected_caption is not None else None
        info_layout = None if info_parent is None else info_parent.layout()
        if info_layout is None:
            raise RuntimeError("main info layout is unavailable for current-action/7-minute controls")
        time_caption_index = info_layout.indexOf(self.time_caption) if self.time_caption is not None else -1
        if time_caption_index < 0:
            time_caption_index = info_layout.count()
        info_layout.insertWidget(time_caption_index, self.action_caption)
        info_layout.insertWidget(time_caption_index + 1, self.action_value)
        status_value_index = info_layout.indexOf(self.time_value) if self.time_value is not None else -1
        if status_value_index < 0:
            status_value_index = info_layout.count() - 1
        info_layout.insertWidget(status_value_index + 1, self.timeout_prompt_label)
        info_layout.insertWidget(status_value_index + 2, self.timeout_fold_button)

        self._vla_ui_timer = QTimer(self.window)
        self._vla_ui_timer.setInterval(200)
        self._vla_ui_timer.timeout.connect(self._poll_vla_watch_ui)
        self._vla_ui_timer.start()

        try:
            self.start_button.clicked.disconnect()
        except Exception:
            pass
        try:
            self.pause_button.clicked.disconnect()
        except Exception:
            pass
        try:
            self.reset_button.clicked.disconnect()
        except Exception:
            pass
        self.start_button.clicked.connect(self._start_workflow)
        self.pause_button.clicked.connect(self._request_stop_after_current)
        self.reset_button.clicked.connect(self._reset_main_idle)
        self.start_button.setText("▶  상의·하의 통합 시작")
        self.pause_button.setText("■  운전 종료 요청")
        self.pause_button.setEnabled(False)

        # Competition/demo workflow selector.  One checkable button behaves like
        # a two-position switch: unchecked=A (automatic basket transfer),
        # checked=B (operator places the garment directly on the board).
        self.mode_toggle_button = QPushButton("Mode A · 자동 투입")
        self.mode_toggle_button.setObjectName("modeToggleButton")
        self.mode_toggle_button.setCheckable(True)
        self.mode_toggle_button.setChecked(False)
        self.mode_toggle_button.setMinimumHeight(68)
        self.mode_toggle_button.setEnabled(False)
        self.mode_toggle_button.clicked.connect(self._on_mode_toggle_clicked)

        # Outer/main screen manual active-baseline refresh.  No emoji by design.
        self.empty_board_refresh_button = QPushButton("빈 판 기준 다시 촬영")
        self.empty_board_refresh_button.setObjectName("emptyBoardRefreshButton")
        self.empty_board_refresh_button.setMinimumHeight(68)
        self.empty_board_refresh_button.setEnabled(False)
        self.empty_board_refresh_button.clicked.connect(
            self._request_manual_empty_board_refresh
        )
        control_parent = self.start_button.parentWidget()
        control_layout = None if control_parent is None else control_parent.layout()
        if control_layout is None:
            raise RuntimeError("main controlLayout is unavailable for empty-board refresh button")
        # Mode / Start / Stop / Empty-board refresh / Reset
        try:
            control_layout.insertWidget(0, self.mode_toggle_button)
        except Exception:
            control_layout.addWidget(self.mode_toggle_button)
        try:
            control_layout.insertWidget(3, self.empty_board_refresh_button)
        except Exception:
            control_layout.addWidget(self.empty_board_refresh_button)

        # V22 integrated-main 7EP compact pass.
        # The base 7EP compactor runs before this integration layer inserts
        # current-action and 7-minute controls, so re-compact the main page here.
        if str(getattr(self, "display_profile", "desktop")) == "7ep":
            self._apply_integrated_main_7ep_compact_layout(info_layout, control_layout)

        self.workflow_signals.ui.connect(self._apply_ui_update)
        self.workflow_signals.log.connect(self._append_log)

        self._apply_ui_update({
            "state": "초기 빈 판 기준 준비",
            "step": "0 / 4 · EMPTY BOARD RAW 캡처",
            "progress": 0,
            "detected": "빈 판을 그대로 유지해 주세요 · 기준 RAW 1회 캡처 중",
            "start_enabled": False,
            "stop_enabled": False,
            "reset_enabled": False,
            "mode_enabled": False,
            "integration_status": "카메라 LIVE · EMPTY BOARD RAW 준비",
        })
        self._begin_initial_empty_board_capture()
        print(f"[MAIN] build={BUILD} physical={bool(args.physical)}")
        print(
            "[MAIN-FLOW] Mode A: garment_transfer -> TOP/BOTTOM classifier; "
            "Mode B: operator board placement -> TOP/BOTTOM classifier; "
            "then TOP:upper_process->V51 / BOTTOM:V114 -> auto fold"
        )
        print(
            "[MAIN-CAMERA-INVARIANT] V22 is sole VideoCapture owner from startup to application exit; "
            "robot modules consume latest_raw_camera_frame copies only"
        )
        print(
            f"[MAIN-INTEGRATION] transfer={self.transfer_module_path} upper={self.upper_process_module_path} "
            f"upper_vla={self.upper_vla_process_module_path} bottom={self.bottom_process_module_path}"
        )
        print(
            f"[MAIN-INTEGRATION] transfer={self.transfer_module_path} upper={self.upper_process_module_path} "
            f"upper_vla={self.upper_vla_process_module_path}"
        )

    # MaintenanceUiController's QTimer is still the only physical cap.read loop.
    # Default publication follows the inherited update; early publication is opt-in.
    def _update_camera(self) -> None:
        # Freeze the route for this tick, even if the workflow changes during UI work.
        self._main_early_raw_tick = False
        self._main_early_raw_publish_s = None
        if (os.environ.get("TONGSPAM_MAIN_EARLY_RAW", "0") == "1"
                and not self._main_closing and not self.workflow_stop.is_set()):
            with self._vla_watch_lock:
                self._main_early_raw_tick = self._vla_watch_kind == "upper"
        # The parent replaces this copy only after a successful RAW capture.
        previous_raw = getattr(self, "latest_raw_camera_frame", None)
        completed = False
        try:
            super()._update_camera()
            completed = True
        finally:
            provider = getattr(self, "shared_raw_camera", None)
            raw = getattr(self, "latest_raw_camera_frame", None)
            if provider is not None:
                publish_s = self._main_early_raw_publish_s or 0.0
                if completed and isinstance(raw, np.ndarray) and raw.size > 0 and raw is not previous_raw and self._main_early_raw_publish_s is None:
                    publish_started = time.monotonic()
                    provider.publish(raw)
                    publish_s = max(0.0, time.monotonic() - publish_started)
                try:
                    timing = dict(getattr(self, "_main_latency_camera_timing", None) or {})
                    if self._main_early_raw_publish_s is not None:
                        # The parent total already includes the inline publish duration.
                        timing["total_s"] = max(0.0, float(timing.get("total_s", 0.0)) - publish_s)
                    provider.note_camera_update(timing, publish_s)
                except Exception as exc:
                    try:
                        print(f"[MAIN-LATENCY-WARN] camera timing telemetry failed: {type(exc).__name__}", flush=True)
                    except Exception:
                        pass
                if completed and isinstance(raw, np.ndarray) and raw.size > 0:
                    override = provider.override_frame_copy()
                    if isinstance(override, np.ndarray) and override.size > 0:
                        self.camera_label.set_bgr_frame(override)
                        self.camera_test_view.set_bgr_frame(override)

    def _publish_main_raw_frame_early(self) -> None:
        if not self._main_early_raw_tick:
            return
        # A failed or ambiguous attempt must never trigger a second, late publish.
        self._main_early_raw_publish_s = 0.0
        provider = getattr(self, "shared_raw_camera", None)
        raw = self.latest_raw_camera_frame
        if provider is not None and isinstance(raw, np.ndarray) and raw.size > 0:
            timing = self._main_latency_camera_timing
            before = None
            try:
                before = provider.current_sequence()
            except Exception:
                pass
            started = time.monotonic()
            try:
                provider.publish(raw)
                timing["publish_state"] = "returned"
            except Exception:
                timing["publish_state"] = "unknown_on_error"
                try:
                    if before is not None:
                        after = provider.current_sequence()
                        timing["publish_state"] = "sequence_advanced_on_error" if after > before else "sequence_unchanged_on_error"
                except Exception:
                    pass
                raise
            finally:
                self._main_early_raw_publish_s = max(0.0, time.monotonic() - started)

    def _begin_initial_empty_board_capture(self) -> None:
        """Start the one-shot startup RAW empty-board capture worker.

        The V22 QTimer remains the only physical camera reader.  This worker only
        waits for immutable RAW copies published by V22SharedRawCameraProvider.
        """
        with self._initial_empty_board_lock:
            if self._initial_empty_board_ready.is_set():
                return
            if (
                self._initial_empty_board_capture_thread is not None
                and self._initial_empty_board_capture_thread.is_alive()
            ):
                return
            self._initial_empty_board_capture_error = None
            worker = threading.Thread(
                target=self._capture_initial_empty_board_worker,
                name="v22-initial-empty-board-raw",
                daemon=True,
            )
            self._initial_empty_board_capture_thread = worker
            worker.start()

    def _capture_initial_empty_board_worker(self) -> None:
        """Lock exactly one stabilized RAW frame and never overwrite it."""
        try:
            warmup_frames = max(
                1, int(getattr(self.args, "initial_empty_board_warmup_frames", 12))
            )
            frame_timeout = max(
                0.5, float(getattr(self.args, "initial_empty_board_frame_timeout_s", 3.0))
            )

            last_seq = int(self.shared_raw_camera.current_sequence())
            print(
                "[MAIN-EMPTY-BOARD-RAW] waiting for startup stabilization "
                f"fresh_frames={warmup_frames} start_seq={last_seq}",
                flush=True,
            )

            latest_frame: Optional[np.ndarray] = None
            latest_seq = last_seq
            latest_time = 0.0

            # The camera controls were already applied by V22 before the first
            # displayed frame.  Consume fresh RAW frames only to allow sensor
            # exposure/WB settling; do not average or alter the reference image.
            for index in range(warmup_frames):
                if self._main_closing:
                    return
                latest_frame, latest_seq, latest_time = (
                    self.shared_raw_camera.wait_for_new_frame(
                        latest_seq, timeout=frame_timeout
                    )
                )
                if index in {0, warmup_frames - 1}:
                    print(
                        "[MAIN-EMPTY-BOARD-RAW-STABILIZE] "
                        f"{index + 1}/{warmup_frames} seq={latest_seq}",
                        flush=True,
                    )

            # Capture ONE additional fresh RAW frame after stabilization.  This is
            # the exact frame object that will later be transformed by the VLA's
            # original E-key-equivalent baseline routine.
            latest_frame, latest_seq, latest_time = (
                self.shared_raw_camera.wait_for_new_frame(
                    latest_seq, timeout=frame_timeout
                )
            )
            if (
                latest_frame is None
                or not isinstance(latest_frame, np.ndarray)
                or latest_frame.size == 0
            ):
                raise RuntimeError("startup empty-board RAW frame is empty")

            locked = np.asarray(latest_frame, dtype=np.uint8).copy()
            if getattr(self, "camera_undistorter", None) is not None:
                corrected = np.asarray(self.camera_undistorter.correct(locked), dtype=np.uint8).copy()
            else:
                corrected = locked.copy()

            homography = None
            roi = getattr(self, "roi_projector", None)
            if roi is not None and getattr(roi, "H", None) is not None:
                homography = np.asarray(roi.H, dtype=np.float64).copy()

            camera_geometry = copy.deepcopy(dict(getattr(self, "camera_geometry", {}) or {}))
            camera_capture = dict(self.shared_raw_camera.camera_info())
            camera_capture.update({
                "raw_sequence": int(latest_seq),
                "raw_frame_time": float(latest_time),
                "source": "V22_UI_SOLE_CAMERA_OWNER",
                "controls": self.shared_raw_camera.integration_camera_controls_metadata(),
            })
            digest = hashlib.sha256()
            digest.update(np.ascontiguousarray(corrected, dtype=np.uint8).tobytes())
            if homography is not None:
                digest.update(np.ascontiguousarray(homography, dtype=np.float64).tobytes())
            digest.update(str(camera_geometry.get("calibration_id", "UNKNOWN")).encode("utf-8"))
            digest.update(str(int(latest_seq)).encode("ascii"))
            empty_state = EmptyBoardState(
                captured_at=float(latest_time or time.time()),
                source="integration_runtime_empty_board",
                baseline_id=digest.hexdigest()[:20],
                corrected_bgr=corrected,
                raw_bgr=locked,
                homography=homography,
                camera_geometry=camera_geometry,
                camera_capture=camera_capture,
            )

            with self._initial_empty_board_lock:
                # One-shot invariant: never overwrite a previously locked baseline.
                if self.initial_empty_board_raw_frame is None:
                    self.initial_empty_board_raw_frame = locked
                    self.initial_empty_board_raw_sequence = int(latest_seq)
                    self.initial_empty_board_raw_time = float(latest_time)
                    self.initial_empty_board_state = empty_state.clone()
                self._initial_empty_board_ready.set()

            h, w = locked.shape[:2]
            print(
                "[MAIN-EMPTY-BOARD-LOCK] "
                f"baseline_id={empty_state.baseline_id} seq={latest_seq} size={w}x{h} "
                f"calibration_id={camera_geometry.get('calibration_id')} "
                f"homography={'yes' if homography is not None else 'no'} "
                "lifetime=APPLICATION source=integration_runtime_empty_board",
                flush=True,
            )
            self.workflow_signals.log.emit(
                f"초기 빈 판 RAW 기준 고정 완료 · seq={latest_seq} · {w}x{h}"
            )
            self.workflow_signals.ui.emit({
                "state": "대기 중",
                "step": "0 / 4 · 시작 대기",
                "progress": 0,
                "detected": "초기 빈 판 RAW 기준 고정 완료 · 바구니에 의류를 준비한 뒤 시작",
                "start_enabled": True,
                "stop_enabled": False,
                "reset_enabled": True,
                "empty_board_refresh_enabled": True,
                "mode_enabled": True,
                "integration_status": "카메라 LIVE · EMPTY BOARD RAW LOCKED · MODE A",
            })
        except Exception as exc:
            message = repr(exc)
            with self._initial_empty_board_lock:
                self._initial_empty_board_capture_error = message
            print(f"[MAIN-EMPTY-BOARD-RAW-ERROR] {message}", flush=True)
            self.workflow_signals.log.emit(
                "초기 빈 판 RAW 기준 캡처 실패: " + message
            )
            self.workflow_signals.ui.emit({
                "state": "빈 판 기준 캡처 실패",
                "step": "0 / 4 · 로그 확인",
                "progress": 0,
                "detected": message[:100],
                "start_enabled": False,
                "stop_enabled": False,
                "reset_enabled": True,
                "integration_status": "카메라 LIVE · EMPTY BOARD RAW ERROR",
            })

    def _build_empty_board_state(
        self,
        raw: np.ndarray,
        seq: int,
        captured_at: float,
        *,
        source: str,
    ) -> EmptyBoardState:
        """Build one EmptyBoardState from the shared V22 camera without reopening it."""
        frame = np.asarray(raw, np.uint8).copy()
        if frame.ndim != 3 or frame.shape[2] != 3 or not frame.size:
            raise RuntimeError(f"{source} shared RAW empty-board frame is invalid")

        corrected = (
            np.asarray(self.camera_undistorter.correct(frame), np.uint8).copy()
            if getattr(self, "camera_undistorter", None) is not None
            else frame.copy()
        )
        projector = getattr(self, "roi_projector", None)
        H = (
            None
            if projector is None or getattr(projector, "H", None) is None
            else np.asarray(projector.H, np.float64).copy()
        )
        geometry = copy.deepcopy(dict(getattr(self, "camera_geometry", {}) or {}))
        capture = dict(self.shared_raw_camera.camera_info())
        capture.update({
            "raw_sequence": int(seq),
            "raw_frame_time": float(captured_at),
            "source": str(source),
            "controls": self.shared_raw_camera.integration_camera_controls_metadata(),
        })

        digest = hashlib.sha256()
        digest.update(np.ascontiguousarray(corrected).tobytes())
        if H is not None:
            digest.update(np.ascontiguousarray(H).tobytes())
        digest.update(str(geometry.get("calibration_id", "UNKNOWN")).encode("utf-8"))
        digest.update(str(int(seq)).encode("ascii"))
        digest.update(str(source).encode("utf-8"))
        return EmptyBoardState(
            captured_at=float(captured_at or time.time()),
            source=str(source),
            baseline_id=digest.hexdigest()[:20],
            raw_bgr=frame,
            corrected_bgr=corrected,
            homography=H,
            camera_geometry=geometry,
            camera_capture=capture,
        )

    def _set_active_empty_board_state(
        self,
        state: EmptyBoardState,
        *,
        reason: str,
    ) -> EmptyBoardState:
        self._validate_empty_board_state_for_vla(state)
        with self._initial_empty_board_lock:
            previous = self._final_shared_empty_board_state
            self._final_shared_empty_board_state = state.clone()
        print(
            f"[MAIN-EMPTY-BOARD-ACTIVE] reason={reason} "
            f"baseline_id={state.baseline_id} "
            f"previous={getattr(previous, 'baseline_id', None)} "
            f"source={state.source}",
            flush=True,
        )
        return state.clone()

    def _capture_active_empty_board_baseline(
        self,
        *,
        source: str,
        settle_s: float = 0.0,
    ) -> EmptyBoardState:
        """Capture one fresh active baseline from the sole shared camera owner."""
        with self._empty_board_refresh_lock:
            if settle_s > 0.0:
                deadline = time.monotonic() + float(settle_s)
                while time.monotonic() < deadline:
                    if self._main_closing:
                        raise RuntimeError("application closing during empty-board settle")
                    time.sleep(min(0.05, max(0.0, deadline - time.monotonic())))

            last_seq = int(self.shared_raw_camera.current_sequence())
            frame, seq, captured_at = self.shared_raw_camera.wait_for_new_frame(
                last_seq,
                timeout=max(
                    0.5,
                    float(getattr(self.args, "initial_empty_board_frame_timeout_s", 3.0)),
                ),
            )
            state = self._build_empty_board_state(
                frame,
                int(seq),
                float(captured_at),
                source=str(source),
            )
            return self._set_active_empty_board_state(
                state,
                reason=str(source),
            )

    def _receive_final_transfer_empty_board(
        self,
        raw: np.ndarray,
        seq: int,
        captured_at: float,
    ) -> None:
        """Publish FIX46 pre-basket image as the active baseline for this cycle."""
        state = self._build_empty_board_state(
            raw,
            int(seq),
            float(captured_at),
            source="TRANSFER_FIX46_FINAL_PRE_BASKET",
        )
        state = self._set_active_empty_board_state(
            state,
            reason="TRANSFER_FIX46_FINAL_PRE_BASKET",
        )
        print(
            f"[MAIN-EMPTY-BOARD-FINAL-SHARED] baseline_id={state.baseline_id} "
            f"seq={seq} source=TRANSFER_FIX46_FINAL_PRE_BASKET "
            "lifetime=CYCLE consumers=UPPER,BOTTOM",
            flush=True,
        )

    def _request_manual_empty_board_refresh(self) -> None:
        """Main-screen idle-only active baseline refresh."""
        controller = self.controller
        maintenance_busy = bool(
            self._busy
            or self._robot_test_busy
            or self.robot_calibration_active
            or (
                controller is not None
                and bool(getattr(controller, "fold_in_progress", False))
            )
        )
        if self._active_cycle or self._integration_plate_busy or maintenance_busy:
            self._append_log(
                "빈 판 기준 다시 촬영 차단: 로봇/폴딩판/점검 동작 진행 중"
            )
            return
        if self.workflow_thread is not None and self.workflow_thread.is_alive():
            self._append_log("빈 판 기준 다시 촬영 차단: workflow thread 진행 중")
            return
        if not self._initial_empty_board_ready.is_set():
            self._append_log("빈 판 기준 다시 촬영 차단: 초기 카메라 기준 준비 전")
            return

        self.empty_board_refresh_button.setEnabled(False)
        self._apply_ui_update({
            "state": "빈 판 기준 촬영 중",
            "step": "빈 판을 비운 상태로 유지",
            "progress": 0,
            "detected": "현재 빈 폴딩판을 다음 작업 기준으로 촬영합니다.",
            "start_enabled": False,
            "stop_enabled": False,
            "reset_enabled": False,
            "empty_board_refresh_enabled": False,
            "mode_enabled": False,
            "integration_status": "카메라 LIVE · EMPTY BOARD REFRESH",
        })

        def worker() -> None:
            try:
                # If the operator adjusted/saved V24 exposure controls while idle,
                # bind the newly saved profile to this newly captured baseline.
                self._integration_ui_camera_session = _load_integration_ui_camera_session(self)
                state = self._capture_active_empty_board_baseline(
                    source="MANUAL_MAIN_EMPTY_BOARD_REFRESH",
                    settle_s=0.25,
                )
                self.workflow_signals.log.emit(
                    f"빈 판 기준 갱신 완료 · baseline_id={state.baseline_id}"
                )
                self.workflow_signals.ui.emit({
                    "state": "대기 중",
                    "step": "0 / 4 · 시작 대기",
                    "progress": 0,
                    "detected": "빈 판 기준 갱신 완료 · 다음 작업 준비",
                    "start_enabled": True,
                    "stop_enabled": False,
                    "reset_enabled": True,
                    "empty_board_refresh_enabled": True,
                    "mode_enabled": True,
                    "integration_status": f"카메라 LIVE · ACTIVE EMPTY BOARD UPDATED · MODE {self.demo_mode}",
                })
            except Exception as exc:
                message = repr(exc)
                print(f"[MAIN-EMPTY-BOARD-MANUAL-ERROR] {message}", flush=True)
                self.workflow_signals.log.emit("빈 판 기준 갱신 실패: " + message)
                self.workflow_signals.ui.emit({
                    "state": "빈 판 기준 갱신 실패",
                    "step": "로그 확인",
                    "progress": 0,
                    "detected": message[:100],
                    "start_enabled": True,
                    "stop_enabled": False,
                    "reset_enabled": True,
                    "empty_board_refresh_enabled": True,
                    "mode_enabled": True,
                    "integration_status": f"카메라 LIVE · EMPTY BOARD REFRESH ERROR · MODE {self.demo_mode}",
                })

        threading.Thread(
            target=worker,
            name="main-manual-empty-board-refresh",
            daemon=True,
        ).start()

    def initial_empty_board_raw_copy(self) -> np.ndarray:
        """Return an immutable-use copy of the one-shot startup empty-board RAW."""
        with self._initial_empty_board_lock:
            frame = self.initial_empty_board_raw_frame
            if frame is None or not self._initial_empty_board_ready.is_set():
                detail = self._initial_empty_board_capture_error or "not captured yet"
                raise RuntimeError(
                    f"initial empty-board RAW is not ready: {detail}"
                )
            return frame.copy()

    def initial_empty_board_state_copy(self) -> EmptyBoardState:
        """Return the current active cycle baseline; fall back to startup baseline."""
        with self._initial_empty_board_lock:
            state = self._final_shared_empty_board_state or self.initial_empty_board_state
            if state is None or not self._initial_empty_board_ready.is_set():
                detail = self._initial_empty_board_capture_error or "not captured yet"
                raise RuntimeError(f"initial EmptyBoardState is not ready: {detail}")
            return state.clone()

    def _validate_empty_board_state_for_vla(self, state: EmptyBoardState) -> None:
        active_capture_sources = {
            "integration_runtime_empty_board",
            "TRANSFER_FIX46_FINAL_PRE_BASKET",
            "POST_FOLD_UPPER_EMPTY_BOARD",
            "POST_FOLD_BOTTOM_EMPTY_BOARD",
            "MANUAL_MAIN_EMPTY_BOARD_REFRESH",
        }
        if str(getattr(state, "source", "")) in active_capture_sources:
            baseline_meta = dict((state.camera_capture or {}).get("controls", {}) or {})
            baseline_actual = dict(baseline_meta.get("actual_controls", {}) or {})
            current_meta = self.shared_raw_camera.integration_camera_controls_metadata()
            current_actual = dict(current_meta.get("actual_controls", {}) or {})
            if (not baseline_actual or not current_actual
                    or "snapshot_error" in baseline_actual or "snapshot_error" in current_actual):
                raise RuntimeError(
                    "active empty-board camera readback unavailable; VLA handoff blocked"
                )
            keys = set(baseline_actual) | set(current_actual)
            mismatches = [
                key for key in sorted(keys)
                if baseline_actual.get(key) != current_actual.get(key)
            ]
            if mismatches:
                raise RuntimeError(
                    f"camera controls changed during empty-board capture: {mismatches}"
                )
        if not isinstance(state, EmptyBoardState):
            raise TypeError("VLA requires EmptyBoardState")
        raw = np.asarray(state.raw_bgr)
        corrected = np.asarray(state.corrected_bgr)
        if raw.ndim != 3 or corrected.ndim != 3 or raw.size == 0 or corrected.size == 0:
            raise RuntimeError("EmptyBoardState contains invalid raw/corrected frames")
        if raw.shape[:2] != corrected.shape[:2]:
            raise RuntimeError(f"EmptyBoardState raw/corrected shape mismatch: {raw.shape} vs {corrected.shape}")
        h = state.homography
        if h is None:
            raise RuntimeError("EmptyBoardState Homography is missing; VLA handoff blocked")
        H = np.asarray(h, dtype=np.float64)
        if H.shape != (3, 3) or not np.all(np.isfinite(H)) or abs(float(np.linalg.det(H))) < 1e-12:
            raise RuntimeError("EmptyBoardState Homography is invalid/singular")
        current_cal = str((getattr(self, "camera_geometry", {}) or {}).get("calibration_id", ""))
        state_cal = str((state.camera_geometry or {}).get("calibration_id", ""))
        if current_cal and state_cal and current_cal != state_cal:
            raise RuntimeError(
                f"EmptyBoardState calibration mismatch: state={state_cal} current={current_cal}"
            )
        roi_H = getattr(getattr(self, "roi_projector", None), "H", None)
        if roi_H is not None:
            A = np.asarray(roi_H, dtype=np.float64)
            def _norm_h(x: np.ndarray) -> np.ndarray:
                scale = float(x[2, 2]) if abs(float(x[2, 2])) > 1e-12 else float(np.linalg.norm(x))
                return x / max(abs(scale), 1e-12)
            if not np.allclose(_norm_h(H), _norm_h(A), rtol=1e-6, atol=1e-6):
                raise RuntimeError("EmptyBoardState Homography differs from current integration Homography")
        print(
            f"[MAIN-EMPTY-BOARD-VALID] baseline_id={state.baseline_id} "
            f"calibration_id={state_cal or current_cal or 'UNKNOWN'} homography=yes size={state.image_size}"
        )

    def camera_index_for_robot(self) -> int:
        raw = str(self.args.camera)
        if raw.isdigit():
            return int(raw)
        name = Path(raw).name
        if name.startswith("video") and name[5:].isdigit():
            return int(name[5:])
        return 0

    @staticmethod
    def _load_runtime_module(path: Path, module_name: str):
        if not path.is_file():
            raise FileNotFoundError(f"integration module missing: {path}")
        spec = importlib.util.spec_from_file_location(module_name, str(path))
        if spec is None or spec.loader is None:
            raise RuntimeError(f"could not create module spec: {path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        return module

    def _runtime_module(self, key: str, path: Path):
        module = self._runtime_modules.get(key)
        if module is None:
            module = self._load_runtime_module(path, f"v22_integrated_{key}")
            self._runtime_modules[key] = module
        return module

    def _new_classifier(self) -> TemporalGarmentClassifier:
        return TemporalGarmentClassifier(
            Path(self.args.classifier_model),
            Path(self.args.garment_classify_model),
            self.roi_projector,
            segment_imgsz=int(self.args.classifier_imgsz),
            classify_imgsz=int(self.args.garment_classify_imgsz),
            det_conf=float(self.args.classifier_det_conf),
            window=int(self.args.classifier_window),
            min_votes=int(self.args.classifier_min_votes),
            min_vote_share=float(self.args.classifier_min_vote_share),
            min_mean_conf=float(self.args.classifier_min_mean_conf),
            min_score_margin=float(self.args.classifier_min_score_margin),
            min_roi_overlap=float(self.args.classifier_min_roi_overlap),
            min_area_ratio=float(self.args.classifier_min_area_ratio),
            per_frame_min_conf=float(self.args.classifier_frame_min_conf),
            per_frame_min_ratio=float(self.args.classifier_frame_min_ratio),
            label_map_mode=str(self.args.classifier_label_map),
            settle_frames=int(self.args.classifier_settle_frames),
            settle_area_tol=float(self.args.classifier_settle_area_tol),
            settle_center_tol_px=float(self.args.classifier_settle_center_tol_px),
            settle_bbox_tol=float(self.args.classifier_settle_bbox_tol),
            tta_enabled=False,
            geometry_enabled=False,
            rotation_rescue_enabled=not bool(self.args.classifier_no_rotation_rescue),
            rotation_rescue_after=int(self.args.classifier_rotation_rescue_after),
            rotation_rescue_period=int(self.args.classifier_rotation_rescue_period),
            rescue_conf=float(self.args.classifier_rescue_conf),
            rotation_release_misses=int(self.args.classifier_rotation_release_misses),
        )

    def _get_classifier(self) -> TemporalGarmentClassifier:
        with self._classifier_lock:
            if self._classifier is None:
                self._classifier = self._new_classifier()
            return self._classifier

    def _unload_classifier(self) -> None:
        with self._classifier_lock:
            if self._classifier is not None:
                self._classifier.unload()
            self._classifier = None

    def _frame_copy(self) -> Optional[np.ndarray]:
        # Classifier intentionally keeps the existing corrected-frame V22 path.
        frame = self.latest_camera_frame
        if not isinstance(frame, np.ndarray) or frame.size == 0:
            return None
        return frame.copy()

    def _apply_integrated_main_7ep_compact_layout(self, info_layout, control_layout) -> None:
        """Compact only the integrated MAIN page for 1280x800 / 7EP.

        Geometry only: no workflow, VLA, STOP, timeout, or folding behavior changes.
        """
        if info_layout is not None:
            info_layout.setContentsMargins(14, 12, 14, 12)
            info_layout.setSpacing(7)

            # Collapse the original vertical spacer in statusLayout.
            for idx in range(info_layout.count()):
                item = info_layout.itemAt(idx)
                spacer = None if item is None else item.spacerItem()
                if spacer is not None:
                    spacer.changeSize(
                        0,
                        2,
                        QSizePolicy.Policy.Minimum,
                        QSizePolicy.Policy.Minimum,
                    )
            info_layout.invalidate()

        # Keep the action display to one line.
        self.action_value.setWordWrap(False)
        self.action_value.setMinimumHeight(20)

        # Original main status widgets.
        if getattr(self, "state_badge", None) is not None:
            self.state_badge.setMinimumHeight(42)
            self.state_badge.setMaximumHeight(46)
        if getattr(self, "step_label", None) is not None:
            self.step_label.setMinimumHeight(22)
        if getattr(self, "progress_bar", None) is not None:
            self.progress_bar.setMinimumHeight(14)
            self.progress_bar.setMaximumHeight(16)

        notice_label = self.window.findChild(QLabel, "noticeLabel")
        if notice_label is not None:
            notice_label.setMinimumHeight(50)
            notice_label.setMaximumHeight(58)

        # 7-minute prompt and button.
        self.timeout_prompt_label.setMinimumHeight(38)
        self.timeout_prompt_label.setMaximumHeight(58)
        self.timeout_fold_button.setMinimumHeight(42)
        self.timeout_fold_button.setMaximumHeight(44)

        # Five main control buttons share one row on 1280 px width.
        if control_layout is not None:
            control_layout.setContentsMargins(12, 9, 12, 9)
            control_layout.setSpacing(8)

        for button in (
            self.mode_toggle_button,
            self.start_button,
            self.pause_button,
            self.empty_board_refresh_button,
            self.reset_button,
        ):
            if button is not None:
                button.setMinimumHeight(52)
                button.setMaximumHeight(56)

        control_parent = self.start_button.parentWidget()
        if control_parent is not None:
            control_parent.setMinimumHeight(78)
            control_parent.setMaximumHeight(86)

        print(
            "[UI-7EP-INTEGRATION-COMPACT] "
            "statusMargins=14/12 spacing=7 actionWrap=OFF "
            "timeoutButton=42~44 controlButtons=52~56",
            flush=True,
        )

    def _apply_ui_update(self, payload: object) -> None:
        data = dict(payload or {}) if isinstance(payload, dict) else {}
        if "state" in data:
            self.state_badge.setText(str(data["state"]))
        if "step" in data:
            self.step_label.setText(str(data["step"]))
        if "progress" in data:
            value = int(np.clip(int(data["progress"]), 0, 100))
            self.progress_bar.setValue(value)
            self.progress_text.setText(f"{value}%")
        if "detected" in data and self.detected_value is not None:
            self.detected_value.setText(str(data["detected"]))
        if "start_enabled" in data:
            self.start_button.setEnabled(bool(data["start_enabled"]))
        if "stop_enabled" in data:
            self.pause_button.setEnabled(bool(data["stop_enabled"]))
        if "reset_enabled" in data:
            self.reset_button.setEnabled(bool(data["reset_enabled"]))
        if "empty_board_refresh_enabled" in data:
            self.empty_board_refresh_button.setEnabled(
                bool(data["empty_board_refresh_enabled"])
            )
        if "mode_enabled" in data:
            self.mode_toggle_button.setEnabled(bool(data["mode_enabled"]))
        if "integration_status" in data and self.time_value is not None:
            self.time_value.setText(str(data["integration_status"]))
        if "current_action" in data and self.action_value is not None:
            value = str(data["current_action"] or "-")
            if value != self._last_ui_action_text:
                self.action_value.setText(value)
                self._last_ui_action_text = value
        if "timeout_prompt_text" in data:
            self.timeout_prompt_label.setText(str(data["timeout_prompt_text"]))
        if "timeout_prompt_visible" in data:
            visible = bool(data["timeout_prompt_visible"])
            self.timeout_prompt_label.setVisible(visible)
            self.timeout_fold_button.setVisible(visible)
        if "timeout_button_enabled" in data:
            self.timeout_fold_button.setEnabled(bool(data["timeout_button_enabled"]))

    def _begin_vla_watch(self, kind: str) -> None:
        kind = str(kind).strip().lower()
        if kind not in {"upper", "bottom"}:
            raise ValueError(f"invalid VLA watch kind={kind!r}")
        with self._vla_watch_lock:
            self._vla_watch_kind = kind
            self._vla_watch_started_monotonic = time.monotonic()
            self._vla_timeout_prompt_shown = False
            self._force_fold_requested.clear()
            self._force_fold_kind = None
            self._force_fold_stop_completed_kind = None
        self.workflow_signals.ui.emit({
            "current_action": "판단 중...",
            "timeout_prompt_visible": False,
            "timeout_button_enabled": False,
        })
        print(f"[MAIN-VLA-WATCH-START] garment={kind} limit={float(self.args.vla_force_fold_prompt_s):.1f}s", flush=True)

    def _end_vla_watch(self) -> None:
        with self._vla_watch_lock:
            self._vla_watch_kind = None
            self._vla_watch_started_monotonic = None
            self._vla_timeout_prompt_shown = False
        self.workflow_signals.ui.emit({
            "timeout_prompt_visible": False,
            "timeout_button_enabled": False,
        })

    def _force_fold_for(self, kind: str) -> bool:
        with self._vla_watch_lock:
            return bool(
                self._force_fold_requested.is_set()
                and self._force_fold_kind == str(kind).strip().lower()
            )

    def _force_fold_stop_completed_for(self, kind: str) -> bool:
        with self._vla_watch_lock:
            return self._force_fold_stop_completed_kind == str(kind).strip().lower()

    def _mark_force_fold_stop_completed(self, kind: str) -> None:
        with self._vla_watch_lock:
            self._force_fold_stop_completed_kind = str(kind).strip().lower()

    def _clear_force_fold_state(self) -> None:
        with self._vla_watch_lock:
            self._vla_watch_kind = None
            self._vla_watch_started_monotonic = None
            self._vla_timeout_prompt_shown = False
            self._force_fold_requested.clear()
            self._force_fold_kind = None
            self._force_fold_stop_completed_kind = None
        self.workflow_signals.ui.emit({
            "timeout_prompt_visible": False,
            "timeout_button_enabled": False,
        })

    def _poll_vla_watch_ui(self) -> None:
        with self._vla_watch_lock:
            kind = self._vla_watch_kind
            started = self._vla_watch_started_monotonic
            prompt_shown = bool(self._vla_timeout_prompt_shown)
            forced = bool(self._force_fold_requested.is_set())

        if kind not in {"upper", "bottom"} or started is None:
            return

        # Read-only action telemetry; no log parsing and no effect on planning.
        process = None
        if kind == "upper":
            with self._active_upper_vla_lock:
                process = self._active_upper_vla_process
        else:
            with self._active_bottom_lock:
                process = self._active_bottom_process
        action = None
        if process is not None:
            getter = getattr(process, "get_current_action", None)
            if callable(getter):
                try:
                    action = getter()
                except Exception as exc:
                    print(f"[MAIN-ACTION-TELEMETRY-WARN] garment={kind} error={exc!r}", flush=True)

        action_text = "중단 후 접기" if forced else (str(action) if action else "판단 중...")
        if action_text != self._last_ui_action_text:
            self._apply_ui_update({"current_action": action_text})

        elapsed = max(0.0, time.monotonic() - float(started))
        threshold = max(1.0, float(self.args.vla_force_fold_prompt_s))
        if elapsed >= threshold and not prompt_shown and not forced and not self.workflow_stop.is_set():
            with self._vla_watch_lock:
                if self._vla_watch_kind == kind and not self._vla_timeout_prompt_shown:
                    self._vla_timeout_prompt_shown = True
                    prompt_shown = True
            if prompt_shown:
                print(f"[MAIN-VLA-7MIN-PROMPT] garment={kind} elapsed={elapsed:.1f}s", flush=True)
                self._apply_ui_update({
                    "timeout_prompt_text": "7분이 넘게 소요되고 있습니다. 중단하고 옷을 접을까요?",
                    "timeout_prompt_visible": True,
                    "timeout_button_enabled": True,
                })

    def _request_timeout_force_fold(self) -> None:
        with self._vla_watch_lock:
            kind = self._vla_watch_kind
            started = self._vla_watch_started_monotonic
            if kind not in {"upper", "bottom"} or started is None:
                return
            if self._force_fold_requested.is_set():
                return
            elapsed = max(0.0, time.monotonic() - float(started))
            threshold = max(1.0, float(self.args.vla_force_fold_prompt_s))
            if elapsed < threshold:
                return
            self._force_fold_requested.set()
            self._force_fold_kind = kind

        self._apply_ui_update({
            "current_action": "중단 후 접기",
            "timeout_prompt_text": "현재 동작을 안전하게 마무리한 뒤 자동으로 접습니다.",
            "timeout_prompt_visible": True,
            "timeout_button_enabled": False,
            "integration_status": "카메라 LIVE · VLA 안전 중단 후 자동 폴딩 준비",
        })
        self._append_log(f"{kind.upper()} VLA 7분 초과 · 사용자가 '중단 후 접기'를 선택했습니다.")
        print(f"[MAIN-FORCE-FOLD-REQUEST] garment={kind} elapsed={elapsed:.1f}s", flush=True)

        process = None
        if kind == "upper":
            with self._active_upper_vla_lock:
                process = self._active_upper_vla_process
        else:
            with self._active_bottom_lock:
                process = self._active_bottom_process
        if process is None:
            self._append_log("중단 후 접기 요청 시 활성 VLA 프로세스를 찾지 못했습니다.")
            print(f"[MAIN-FORCE-FOLD-WARN] garment={kind} active_process=NONE", flush=True)
            return
        try:
            process.request_stop("7-minute force-fold requested by display UI")
        except Exception as exc:
            self._append_log(f"중단 후 접기 요청 전달 실패: {exc!r}")
            print(f"[MAIN-FORCE-FOLD-STOP-WARN] garment={kind} error={exc!r}", flush=True)

    def _mode_instruction(self, mode: Optional[str] = None) -> str:
        selected = str(mode or self.demo_mode).strip().upper()
        if selected == "B":
            return "Mode B · 빈 판 기준 확보 후 사람이 옷을 보드에 올려 구긴 뒤 시작"
        return "Mode A · 바구니에 의류를 준비한 뒤 시작"

    def _on_mode_toggle_clicked(self, checked: bool) -> None:
        # The button is disabled while any workflow/maintenance action is active,
        # but keep a defensive guard so programmatic clicks cannot change a cycle.
        busy = bool(
            self._active_cycle
            or self._integration_plate_busy
            or self._busy
            or self._robot_test_busy
            or self.robot_calibration_active
            or (self.workflow_thread is not None and self.workflow_thread.is_alive())
        )
        if busy:
            self.mode_toggle_button.blockSignals(True)
            self.mode_toggle_button.setChecked(self.demo_mode == "B")
            self.mode_toggle_button.blockSignals(False)
            self._append_log("Mode 변경 차단: 현재 로봇/통합 작업이 진행 중입니다.")
            return

        self.demo_mode = "B" if bool(checked) else "A"
        self._cycle_demo_mode = self.demo_mode
        self.mode_toggle_button.setText(
            "Mode B · 수동 배치" if self.demo_mode == "B" else "Mode A · 자동 투입"
        )
        instruction = self._mode_instruction()
        self._append_log(f"시연 모드 변경: {instruction}")
        print(f"[MAIN-DEMO-MODE] selected={self.demo_mode}", flush=True)
        if self._initial_empty_board_ready.is_set():
            self._apply_ui_update({
                "state": "대기 중",
                "step": "0 / 4 · 시작 대기",
                "progress": 0,
                "detected": instruction,
                "integration_status": f"카메라 LIVE · READY · MODE {self.demo_mode}",
            })

    def _cycle_mode(self) -> str:
        mode = str(getattr(self, "_cycle_demo_mode", self.demo_mode)).strip().upper()
        return "B" if mode == "B" else "A"

    def _validate_cycle_empty_board_state(
        self, state: EmptyBoardState, consumer: str
    ) -> None:
        self._validate_empty_board_state_for_vla(state)
        mode = self._cycle_mode()
        if mode == "A" and state.source != "TRANSFER_FIX46_FINAL_PRE_BASKET":
            raise RuntimeError(
                f"{consumer} blocked in Mode A: no shared final FIX46 baseline "
                f"(source={state.source})"
            )
        # Mode B intentionally uses the latest active empty-board state captured
        # before the operator placed the garment.  Never capture a new baseline
        # after Mode B Start because the board already contains the garment.
        if mode == "B":
            print(
                f"[MAIN-MODE-B-BASELINE] consumer={consumer} "
                f"baseline_id={state.baseline_id} source={state.source}",
                flush=True,
            )

    def _wait_camera_ready(self) -> None:
        deadline = time.monotonic() + 15.0
        while time.monotonic() < deadline and not self.workflow_stop.is_set():
            corrected = self._frame_copy()
            try:
                raw_ready = self.shared_raw_camera.current_sequence() > 0
            except Exception:
                raw_ready = False
            if corrected is not None and raw_ready and self.shared_raw_camera.is_alive():
                return
            time.sleep(0.10)
        raise RuntimeError("V22 shared camera frame not ready")

    def _start_workflow(self) -> None:
        if self.workflow_thread is not None and self.workflow_thread.is_alive():
            return
        if not self._initial_empty_board_ready.is_set():
            detail = self._initial_empty_board_capture_error or "startup capture still in progress"
            self._apply_ui_update({
                "state": "실행 대기",
                "step": "0 / 4 · EMPTY BOARD RAW 필요",
                "progress": 0,
                "detected": "초기 빈 판 RAW 기준이 아직 준비되지 않았습니다.",
                "start_enabled": False,
                "stop_enabled": False,
                "reset_enabled": True,
                "integration_status": "카메라 LIVE · EMPTY BOARD RAW WAIT",
            })
            print(f"[MAIN-EMPTY-BOARD-RAW-BLOCK] workflow start blocked: {detail}")
            return
        if not bool(self.args.physical):
            self._apply_ui_update({
                "state": "실행 차단",
                "step": "--physical 필요",
                "progress": 0,
                "detected": "통합 로봇 테스트는 --physical 옵션이 필요합니다.",
                "start_enabled": True,
                "stop_enabled": False,
                "reset_enabled": True,
                "integration_status": "카메라 LIVE · 물리 실행 차단",
            })
            print("[MAIN-PHYSICAL-BLOCKED] --physical is required; no robot module started")
            return

        # Freeze the Saved button JSON at Integrated Start. If the operator changed
        # V24 camera settings while idle, the active empty-board baseline must be
        # refreshed with the main-screen button before the next cycle.
        try:
            new_camera_session = _load_integration_ui_camera_session(self)
            previous = self._integration_ui_camera_session
            if getattr(self, "_final_shared_empty_board_state", None) is not None and previous is not None:
                fingerprint_keys = ("controls", "extra_controls")
                if any(new_camera_session.get(key) != previous.get(key) for key in fingerprint_keys):
                    raise RuntimeError(
                        "saved camera settings changed after the active empty-board baseline; "
                        "return to an empty board and press '빈 판 기준 다시 촬영' before the next run"
                    )
            self._integration_ui_camera_session = new_camera_session
            print("[MAIN-UI-CAMERA-SESSION] "
                  f"source={new_camera_session['source']} "
                  f"profile={new_camera_session['profile_path']} "
                  f"sha256={new_camera_session['profile_sha256']} "
                  f"controls={new_camera_session['controls']} "
                  f"extras={new_camera_session['extra_controls']}", flush=True)
        except Exception as exc:
            print(f"[MAIN-UI-CAMERA-SESSION-BLOCK] {exc}", flush=True)
            self._apply_ui_update({
                "state": "카메라 설정 확인 필요", "step": "0 / 4 · 저장값 확인",
                "progress": 0, "detected": str(exc)[:100],
                "start_enabled": True, "stop_enabled": False, "reset_enabled": True,
                "integration_status": "카메라 LIVE · 저장 설정 오류",
            })
            return
        self.workflow_stop.clear()
        self._cycle_demo_mode = self.demo_mode
        self._active_cycle = True
        self.current_garment = None
        self._unload_classifier()
        mode_b = self._cycle_demo_mode == "B"
        self._apply_ui_update({
            "state": "Mode B 시작 준비" if mode_b else "바구니 파지 준비",
            "step": (
                "1 / 3 · 수동 배치 상태 확인"
                if mode_b else "1 / 4 · garment_transfer 시작"
            ),
            "progress": 8 if mode_b else 5,
            "detected": (
                "수동 배치 의류 · 로봇팔 제어 없이 바로 상의/하의 분류"
                if mode_b else "상의/하의 분류 전 · 바구니 파지 및 보드 배치"
            ),
            "start_enabled": False,
            "stop_enabled": True,
            "reset_enabled": False,
            "empty_board_refresh_enabled": False,
            "mode_enabled": False,
            "current_action": "-",
            "timeout_prompt_visible": False,
            "timeout_button_enabled": False,
            "integration_status": (
                "카메라 LIVE · MODE B DIRECT CLASSIFIER"
                if mode_b else "카메라 LIVE · MODE A TRANSFER"
            ),
        })
        print(f"[MAIN-CYCLE-MODE] mode={self._cycle_demo_mode}", flush=True)
        self.workflow_thread = threading.Thread(
            target=self._workflow_loop,
            name="v22-shared-camera-workflow",
            daemon=True,
        )
        self.workflow_thread.start()

    def _request_stop_after_current(self) -> None:
        self.workflow_stop.set()
        self._apply_ui_update({
            "timeout_prompt_visible": False,
            "timeout_button_enabled": False,
        })
        self.pause_button.setEnabled(False)
        self.state_badge.setText("종료 요청")
        with self._active_bottom_lock:
            bottom_process = self._active_bottom_process
        if bottom_process is not None:
            try:
                active = bool(bottom_process.request_stop("display UI stop requested"))
                self._append_log("하의 V114에 안전 중단을 요청했습니다.")
                print(f"[MAIN-BOTTOM-STOP-REQUEST] active_cycle={active}")
            except Exception as exc:
                self._append_log(f"하의 V114 중단 요청 전달 실패: {exc!r}")
                print(f"[MAIN-BOTTOM-STOP-WARN] {exc!r}")
        with self._active_upper_vla_lock:
            process = self._active_upper_vla_process
        if process is None and bottom_process is None and self._active_cycle:
            self._append_log(
                "현재 통합 단계에 중단 요청을 전달했습니다. garment_transfer 포함 현재 물리 구간을 안전하게 마친 뒤 새 동작을 막고 종료합니다."
            )
            print("[MAIN-WORKFLOW-STOP-REQUEST] shared stop_event=SET garment_transfer=CONNECTED")
        if process is not None:
            try:
                active = bool(process.request_stop("display UI stop requested"))
                self._append_log(
                    "주름 펴기 안전 중단 요청됨; 새 판단/동작은 차단하고 현재 macro의 release/clear 후 종료합니다."
                )
                print(f"[MAIN-UPPER-VLA-STOP-REQUEST] active_cycle={active}")
            except Exception as exc:
                self._append_log(f"주름 펴기 중단 요청 전달 실패: {exc!r}")
                print(f"[MAIN-UPPER-VLA-STOP-WARN] {exc!r}")
        elif bottom_process is None:
            self._append_log("종료 요청됨; 진행 중 로봇 동작은 강제 정지하지 않고 현재 단계 완료 후 종료합니다.")

    def _reset_main_idle(self) -> None:
        if self.workflow_thread is not None and self.workflow_thread.is_alive():
            return
        self.current_garment = None
        self._clear_force_fold_state()
        self._unload_classifier()
        self._apply_ui_update({
            "state": "대기 중",
            "step": "0 / 4 · 시작 대기",
            "progress": 0,
            "detected": self._mode_instruction(),
            "start_enabled": bool(self._initial_empty_board_ready.is_set()),
            "stop_enabled": False,
            "reset_enabled": True,
            "empty_board_refresh_enabled": bool(self._initial_empty_board_ready.is_set()),
            "mode_enabled": bool(self._initial_empty_board_ready.is_set()),
            "current_action": "-",
            "timeout_prompt_visible": False,
            "timeout_button_enabled": False,
            "integration_status": (
                f"카메라 LIVE · READY · MODE {self.demo_mode}"
                if self._initial_empty_board_ready.is_set()
                else "카메라 LIVE · EMPTY BOARD RAW WAIT"
            ),
        })

    def _classify_until_locked(self) -> Optional[ClassificationResult]:
        classifier = self._get_classifier()
        classifier.reset_history()
        self._wait_camera_ready()
        while not self.workflow_stop.is_set():
            frame = self._frame_copy()
            if frame is None:
                time.sleep(0.10)
                continue
            try:
                ev = classifier.infer_frame(frame)
                result = classifier.push(ev)
            except Exception as exc:
                raise RuntimeError(f"garment classification failed: {exc!r}") from exc

            if result.garment == "UPPER":
                text = f"상의 · 확정 신뢰도 {result.confidence*100:.0f}%"
            elif result.garment == "BOTTOM":
                text = f"하의 · 확정 신뢰도 {result.confidence*100:.0f}%"
            else:
                if ev.winner == "EMPTY":
                    text = "보드 위 의류 확인 중"
                elif ev.winner == "SETTLING":
                    text = "의류 안정화 확인 중"
                else:
                    text = f"판단 중 · 상의표 {result.upper_votes} / 하의표 {result.bottom_votes}"
            self.workflow_signals.ui.emit({
                "state": "상의/하의 분류 중",
                "step": "2 / 4 · BOARD_READY 후 분류",
                "progress": 48,
                "detected": text,
                "integration_status": "카메라 LIVE · CLASSIFY",
            })
            self.workflow_signals.log.emit(
                f"CLASS frame={ev.winner} {ev.detail} | {result.detail}"
            )
            if result.garment in {"UPPER", "BOTTOM"}:
                return result
            time.sleep(max(0.05, float(self.args.classifier_interval_s)))
        return None

    def _capture_mode_b_median_snapshot(self, attempt: int) -> Optional[np.ndarray]:
        """Mode B snapshot: flush stale frames -> immediate fresh burst -> median -> correct once.

        Deliberately does NOT use scene/stillness/motion gating.  Competition
        spectators or background motion must never delay the snapshot.  Only
        fresh-frame availability and workflow STOP can block this path.
        """
        self._wait_camera_ready()
        burst_need = int(np.clip(int(self.args.mode_b_snapshot_burst_frames), 3, 5))
        flush_need = max(0, int(self.args.mode_b_snapshot_flush_frames))
        timeout_s = max(2.0, float(self.args.mode_b_snapshot_timeout_s))
        frame_timeout_s = max(0.25, float(self.args.mode_b_snapshot_frame_timeout_s))

        last_seq = int(self.shared_raw_camera.current_sequence())

        # Discard a few already-buffered frames so the burst reflects the
        # current board state, but do not inspect motion in any of them.
        for idx in range(flush_need):
            if self.workflow_stop.is_set():
                return None
            raw, last_seq, _ = self.shared_raw_camera.wait_for_new_frame(
                last_seq, timeout=frame_timeout_s
            )
            if idx == flush_need - 1:
                print(
                    f"[MODE-B-SNAPSHOT-FLUSH] attempt={attempt} "
                    f"fresh={flush_need}/{flush_need} seq={last_seq}",
                    flush=True,
                )

        self.workflow_signals.ui.emit({
            "state": "Mode B snapshot 촬영",
            "step": f"즉시 촬영 · 시도 {attempt}",
            "progress": 42,
            "detected": f"주변 움직임 무시 · fresh {burst_need}프레임 median",
            "integration_status": "카메라 LIVE · MODE B SNAPSHOT BURST",
        })
        print(
            f"[MODE-B-SNAPSHOT-NO-STILLNESS] attempt={attempt} "
            f"motion_gate=DISABLED burst={burst_need}",
            flush=True,
        )

        burst: List[np.ndarray] = []
        deadline = time.monotonic() + timeout_s

        while len(burst) < burst_need and time.monotonic() < deadline:
            if self.workflow_stop.is_set():
                return None

            remain = max(0.05, deadline - time.monotonic())
            try:
                raw, last_seq, _ = self.shared_raw_camera.wait_for_new_frame(
                    last_seq,
                    timeout=min(frame_timeout_s, remain),
                )
            except TimeoutError:
                continue

            burst.append(raw.copy())
            print(
                f"[MODE-B-SNAPSHOT-BURST] attempt={attempt} "
                f"{len(burst)}/{burst_need} seq={last_seq}",
                flush=True,
            )

        if len(burst) < burst_need:
            raise RuntimeError(
                "MODE_B_SNAPSHOT_TIMEOUT: "
                f"attempt={attempt} timeout={timeout_s:.1f}s "
                f"fresh_frames={len(burst)}/{burst_need}"
            )

        stack = np.stack(burst[-burst_need:], axis=0).astype(np.float32)
        raw_median = np.median(stack, axis=0).astype(np.uint8)
        corrected = (
            np.asarray(self.camera_undistorter.correct(raw_median), np.uint8).copy()
            if getattr(self, "camera_undistorter", None) is not None
            else raw_median.copy()
        )
        print(
            f"[MODE-B-SNAPSHOT-READY] attempt={attempt} frames={burst_need} "
            f"seq={last_seq} corrected={getattr(self, 'camera_undistorter', None) is not None}",
            flush=True,
        )
        return corrected

    def _classify_mode_b_snapshot(self) -> Optional[ClassificationResult]:
        """Classify Mode B from one F24-style frozen median snapshot per attempt."""
        classifier = self._get_classifier()
        classifier.reset_history()
        self._wait_camera_ready()

        # Reuse the active baseline captured while the board was empty.
        # Never recapture after Mode B Start because the garment is already on-board.
        empty_state = self.initial_empty_board_state_copy()
        self._validate_cycle_empty_board_state(empty_state, "mode_b_snapshot_diff")
        empty_corrected = np.asarray(empty_state.corrected_bgr, np.uint8).copy()
        print(
            f"[MODE-B-SNAPSHOT-DIFF-BASELINE] baseline_id={empty_state.baseline_id} "
            f"source={empty_state.source} shape={empty_corrected.shape[1]}x{empty_corrected.shape[0]}",
            flush=True,
        )

        max_attempts = max(1, int(self.args.mode_b_snapshot_max_attempts))
        last_detail = "not attempted"

        for attempt in range(1, max_attempts + 1):
            if self.workflow_stop.is_set():
                return None
            snapshot = self._capture_mode_b_median_snapshot(attempt)
            if snapshot is None:
                return None
            try:
                ev = classifier.infer_snapshot(
                    snapshot,
                    empty_board=empty_corrected,
                )
            except Exception as exc:
                raise RuntimeError(
                    f"Mode B snapshot classification failed: {exc!r}"
                ) from exc

            best = max(float(ev.upper_score), float(ev.bottom_score))
            other = min(float(ev.upper_score), float(ev.bottom_score))
            ratio = best / max(other, 1e-6)
            accepted = bool(
                ev.winner in {"UPPER", "BOTTOM"}
                and best >= float(classifier.min_mean_conf)
                and ratio >= float(classifier.min_score_margin)
            )
            last_detail = (
                f"attempt={attempt}/{max_attempts} winner={ev.winner} "
                f"U={ev.upper_score:.3f} B={ev.bottom_score:.3f} "
                f"conf={best:.3f}/{classifier.min_mean_conf:.3f} "
                f"ratio={ratio:.2f}/{classifier.min_score_margin:.2f} | {ev.detail}"
            )
            self.workflow_signals.log.emit("MODE B SNAPSHOT CLASS " + last_detail)
            print(f"[MODE-B-SNAPSHOT-CLASS] {last_detail}", flush=True)

            if accepted:
                garment = str(ev.winner)
                self.workflow_signals.ui.emit({
                    "state": "상의/하의 분류 완료",
                    "step": "Mode B snapshot 분류 확정",
                    "progress": 55,
                    "detected": (
                        f"상의 · snapshot 신뢰도 {best*100:.0f}%"
                        if garment == "UPPER"
                        else f"하의 · snapshot 신뢰도 {best*100:.0f}%"
                    ),
                    "integration_status": "카메라 LIVE · MODE B SNAPSHOT LOCKED",
                })
                return ClassificationResult(
                    garment=garment,
                    confidence=float(best),
                    upper_votes=1 if garment == "UPPER" else 0,
                    bottom_votes=1 if garment == "BOTTOM" else 0,
                    sample_count=1,
                    detail="MODE_B_SNAPSHOT " + last_detail,
                )

            if attempt < max_attempts:
                self.workflow_signals.ui.emit({
                    "state": "Mode B snapshot 재촬영",
                    "step": f"분류 재시도 {attempt + 1}/{max_attempts}",
                    "progress": 46,
                    "detected": "snapshot에서 의류/분류 확정 실패 · fresh snapshot 재시도",
                    "integration_status": "카메라 LIVE · MODE B SNAPSHOT RETRY",
                })
                time.sleep(0.10)

        raise RuntimeError(
            "MODE_B_SNAPSHOT_CLASSIFICATION_NOT_LOCKED: " + last_detail
        )

    def _robot_common_argv(self) -> List[str]:
        return [
            "--physical-auto",
            "--config", str(Path(self.args.integration_config).expanduser().resolve()),
            "--hfile", str(Path(self.args.hfile).expanduser().resolve()),
            "--camera-calibration", str(Path(self.args.integration_camera_calibration).expanduser().resolve()),
            "--seg-model", str(Path(self.args.integration_seg_model).expanduser().resolve()),
            "--tshirt-pose-model", str(Path(self.args.integration_upper_pose_model).expanduser().resolve()),
            "--camera", str(self.camera_index_for_robot()),
            "--backend", "v4l2",
            "--width", str(int(getattr(self.args, "camera_width", 1280))),
            "--height", str(int(getattr(self.args, "camera_height", 720))),
            "--arm1-port", "/dev/roarm_1",
            "--arm2-port", "/dev/roarm_2",
            "--speed", "1.12",
        ]

    def _fix134c_capture_h_bundle(self):
        path = Path(self.args.hfile).expanduser().resolve()
        try:
            raw = path.read_bytes()
            payload = json.loads(raw.decode('utf-8'))
            H = np.asarray(payload.get('H'), dtype=np.float64)
            raw_H = np.asarray(payload.get('raw_H'), dtype=np.float64)
            geometry = payload.get('camera_geometry')
            if H.shape != (3, 3) or raw_H.shape != (3, 3):
                raise ValueError('H/raw_H must both be 3x3')
            if not np.all(np.isfinite(H)) or not np.all(np.isfinite(raw_H)):
                raise ValueError('H/raw_H contain non-finite values')
            if not isinstance(geometry, dict) or not geometry:
                raise ValueError('camera_geometry missing')
        except Exception as exc:
            print(f'[FIX134C-H-BUNDLE-GUARD-WARN] bundle incomplete path={path} error={exc!r}; recalibrate once')
            return None
        digest = hashlib.sha256(raw).hexdigest()
        print(f'[FIX134C-H-BUNDLE-GUARD] captured full bundle sha256={digest}')
        return {'path': path, 'raw': raw, 'sha256': digest}

    def _fix134c_restore_h_bundle(self, state):
        if not isinstance(state, dict):
            return
        raw = state.get('raw')
        if not isinstance(raw, (bytes, bytearray)):
            return
        authoritative = Path(state['path']).expanduser().resolve()
        targets = [authoritative]
        restored = 0
        for path in dict.fromkeys(targets):
            try:
                path.parent.mkdir(parents=True, exist_ok=True)
                tmp = path.with_name(path.name + f'.fix134c.{os.getpid()}.tmp')
                tmp.write_bytes(bytes(raw))
                os.replace(tmp, path)
                restored += 1
            except Exception as exc:
                print(f'[FIX134C-H-BUNDLE-GUARD-WARN] restore failed path={path} error={exc!r}')
        print(f'[FIX134C-H-BUNDLE-GUARD] restored full H+raw_H+camera_geometry targets={restored}')

    def _run_garment_transfer(self) -> None:
        self._wait_camera_ready()
        module = self._runtime_module("garment_transfer", self.transfer_module_path)
        if not hasattr(module, "run_integrated"):
            raise RuntimeError("garment transfer module has no run_integrated() shared-camera entrypoint")
        argv = self._robot_common_argv()
        argv += ["--calib-file", str(Path(self.args.integration_basket_calib).expanduser().resolve())]
        self.workflow_signals.ui.emit({
            "state": "바구니 파지/보드 배치 중",
            "step": "1 / 4 · garment_transfer",
            "progress": 18,
            "detected": "분류 전 · 바구니 파지 코드 실행 중",
            "integration_status": "카메라 LIVE · TRANSFER",
        })
        print("[MAIN-TRANSFER-START] V22 camera stays OPEN; running garment_transfer in-process")
        with self._initial_empty_board_lock:
            reuse = (
                None
                if self._final_shared_empty_board_state is None
                else self._final_shared_empty_board_state.clone()
            )
        fix134c_h_bundle = self._fix134c_capture_h_bundle()
        try:
            rc = int(module.run_integrated(
                self.shared_raw_camera,
                argv,
                on_final_empty_board=self._receive_final_transfer_empty_board,
                reuse_empty_board_state=reuse,
                stop_event=self.workflow_stop,
            ))
        finally:
            self._fix134c_restore_h_bundle(fix134c_h_bundle)
        print(f"[MAIN-TRANSFER-RETURN] rc={rc}")
        if rc == 130 and self.workflow_stop.is_set():
            self.workflow_signals.log.emit(
                "바구니/보드 배치 단계가 안전 중단되었습니다. 다음 동작은 시작하지 않습니다."
            )
            print(
                "[MAIN-TRANSFER-STOPPED] cooperative STOP completed; "
                "classifier/upper/lower/auto-fold stages are blocked"
            )
            return
        if rc != 0:
            raise RuntimeError(f"garment_transfer integrated run failed rc={rc}")
        with self._initial_empty_board_lock:
            if self._final_shared_empty_board_state is None:
                raise RuntimeError("FIX46 pre-basket baseline not handed to MAIN; classifier blocked")
        print("[MAIN-BOARD-READY] garment_transfer returned rc=0; enabling TOP/BOTTOM classifier")

    def _run_upper_process(self) -> None:
        self._wait_camera_ready()
        module = self._runtime_module("upper_process", self.upper_process_module_path)
        if not hasattr(module, "run_integrated"):
            raise RuntimeError("upper process module has no run_integrated() shared-camera entrypoint")
        argv = self._robot_common_argv()
        state = self.initial_empty_board_state_copy()
        self._validate_cycle_empty_board_state(state, "upper_process")
        self.workflow_signals.ui.emit({
            "state": "상의 후반 처리 중",
            "step": "3 / 4 · ARM1 recenter 이후 상의 처리",
            "progress": 70,
            "detected": "상의 · 91 mm 위치보정부터 실행",
            "stop_enabled": True,
            "integration_status": "카메라 LIVE · UPPER",
        })
        print("[MAIN-UPPER-START] classifier unloaded; V22 camera stays OPEN; running upper_process in-process")
        rc = int(module.run_integrated(
            self.shared_raw_camera,
            argv,
            empty_board_state=state.clone(),
            stop_event=self.workflow_stop,
        ))
        print(f"[MAIN-UPPER-RETURN] rc={rc}")
        if rc == 130 and self.workflow_stop.is_set():
            print(
                "[MAIN-UPPER-STOPPED] upper_process cooperative stop complete; "
                "V51 will not start"
            )
            return
        if rc != 0:
            raise RuntimeError(f"upper_process integrated run failed rc={rc}")

    def _upper_vla_argv(self) -> List[str]:
        return [
            "--config", str(Path(self.args.integration_config).expanduser().resolve()),
            "--hfile", str(Path(self.args.hfile).expanduser().resolve()),
            "--camera-calibration", str(Path(self.args.integration_camera_calibration).expanduser().resolve()),
            "--grasp-model", str(Path(self.args.integration_seg_model).expanduser().resolve()),
            "--pose-model", str(Path(self.args.integration_upper_pose_model).expanduser().resolve()),
            "--arm1-port", "/dev/roarm_1",
            "--arm2-port", "/dev/roarm_2",
            "--speed", "1.12",
            "--vla-shadow-engine", str(Path(self.args.integration_upper_vla_model).expanduser().resolve()),
            "--vla-shadow-state-norm", str(Path(self.args.integration_upper_vla_state_norm).expanduser().resolve()),
            "--no-window",
            "--no-terminal-control",
            "--no-camera-controls",
        ]

    def _run_upper_vla_process(self) -> Any:
        self._wait_camera_ready()
        state = self.initial_empty_board_state_copy()
        self._validate_cycle_empty_board_state(state, "upper_v51")
        module = self._runtime_module("upper_vla_process", self.upper_vla_process_module_path)
        process_cls = getattr(module, "UpperProcess", None)
        if process_cls is None:
            raise RuntimeError("upper_vla_process module has no UpperProcess public API")
        argv = self._upper_vla_argv()
        self.workflow_signals.ui.emit({
            "state": "VLA 주름 펴기 중",
            "step": "4 / 5 · V51 모델 판단/로봇 동작",
            "progress": 84,
            "detected": f"상의 · VLA 자동 판단 시작 · baseline={state.baseline_id[:10]}",
            "stop_enabled": True,
            "integration_status": "카메라 LIVE · UPPER VLA",
        })
        print(
            f"[MAIN-UPPER-VLA-START] baseline_id={state.baseline_id} "
            "V22 camera remains sole VideoCapture owner"
        )
        process = process_cls(
            argv=argv,
            debug=False,
            external_frame_provider=self.shared_raw_camera,
        )
        with self._active_upper_vla_lock:
            self._active_upper_vla_process = process
        start_timeout = max(5.0, float(self.args.upper_vla_start_timeout_s))
        cycle_timeout_raw = float(self.args.upper_vla_cycle_timeout_s)
        cycle_timeout = None if cycle_timeout_raw <= 0.0 else cycle_timeout_raw
        try:
            if self.workflow_stop.is_set():
                print("[MAIN-UPPER-VLA-START-SKIP] stop was requested before v51 initialization")
                return None
            process.start(timeout=start_timeout)
            if self.workflow_stop.is_set():
                print("[MAIN-UPPER-VLA-START-SKIP] stop was requested while v51 initialized; cycle not started")
                return None
            applied = process.set_empty_board_state(state.clone(), timeout=15.0)
            print(
                f"[MAIN-UPPER-VLA-EMPTY] garment_type=TOP "
                f"empty_board_baseline_id={applied.baseline_id}"
            )
            if self.workflow_stop.is_set():
                print("[MAIN-UPPER-VLA-START-SKIP] stop was requested while applying empty-board state; cycle not started")
                return None
            self.shared_raw_camera.begin_latency_trace("upper_v51_run_cycle")
            self.shared_raw_camera.latency_event("RUN_CYCLE_CALL")
            self._begin_vla_watch("upper")
            result = process.run_cycle(timeout=cycle_timeout)
            self.shared_raw_camera.latency_event(
                "RUN_CYCLE_RETURN",
                status=str(getattr(result, "status", "")),
                reason=str(getattr(result, "reason", "")),
            )
            status = str(getattr(result, "status", ""))
            reason = str(getattr(result, "reason", ""))
            recovery = bool(getattr(result, "recovery_required", False))
            print(
                f"[MAIN-UPPER-VLA-RETURN] status={status} recovery_required={recovery} reason={reason}"
            )
            if self._force_fold_for("upper") and status in {"ABORTED", "STOPPED", "ERROR_STOP"}:
                if recovery:
                    raise RuntimeError(
                        "FORCE_FOLD_UPPER_STOP_UNSAFE: V51 cooperative stop reported recovery_required=True; "
                        "automatic folding is blocked to avoid plate/arm collision"
                    )
                self._mark_force_fold_stop_completed("upper")
                self.workflow_signals.log.emit(
                    "상의 V51 안전 중단 및 STANDBY 확인 완료 · FINISH 라벨을 만들지 않고 기존 자동 폴딩 경로로 이동"
                )
                print(f"[MAIN-UPPER-FORCE-FOLD-STOPPED] status={status} standby=CONFIRMED", flush=True)
                return result
            if status == "ABORTED" and self.workflow_stop.is_set():
                if recovery:
                    print(
                        "[MAIN-UPPER-VLA-ABORT-WARNING] recovery required: "
                        "robot state check/manual recovery may be needed"
                    )
                    try:
                        self.workflow_signals.ui.emit({
                            "state": "작업 중단됨 - 로봇 상태 확인 필요",
                            "step": "ABORTED · recovery required",
                            "progress": 0,
                        })
                    except Exception:
                        pass
                print(
                    f"[MAIN-UPPER-VLA-ABORTED] cooperative stop complete "
                    f"recovery_required={recovery} reason={reason}"
                )
                return result
            if status != "SUCCESS":
                raise RuntimeError(
                    f"upper VLA v51 did not finish successfully: status={status} "
                    f"recovery_required={recovery} reason={reason}"
                )
            self.workflow_signals.ui.emit({"current_action": "FINISH"})
            return result
        finally:
            try:
                try:
                    self.shared_raw_camera.end_latency_trace("upper_v51_run_cycle")
                except Exception as exc:
                    print(f"[MAIN-LATENCY-WARN] trace finalization failed: {exc!r}", flush=True)
                process.close(timeout=65.0)
            finally:
                self._end_vla_watch()
                with self._active_upper_vla_lock:
                    if self._active_upper_vla_process is process:
                        self._active_upper_vla_process = None

    def _bottom_vla_argv(self) -> List[str]:
        return [
            "--config", str(Path(self.args.integration_config).expanduser().resolve()),
            "--hfile", str(Path(self.args.hfile).expanduser().resolve()),
            "--camera-calibration", str(Path(self.args.integration_camera_calibration).expanduser().resolve()),
            "--seg-model", str(Path(self.args.integration_seg_model).expanduser().resolve()),
            "--pose-model", str(Path(self.args.integration_bottom_pose_model).expanduser().resolve()),
            "--d50-basket-calib", str(Path(self.args.integration_basket_calib).expanduser().resolve()),
            "--width", str(int(self.args.width)),
            "--height", str(int(self.args.height)),
            "--arm1-port", "/dev/roarm_1",
            "--arm2-port", "/dev/roarm_2",
            "--no-camera-controls-enable",
        ]

    def _run_bottom_process(self) -> Any:
        self._wait_camera_ready()
        state = self.initial_empty_board_state_copy()
        self._validate_cycle_empty_board_state(state, "bottom_v114")

        module = self._runtime_module("bottom_process", self.bottom_process_module_path)
        process_cls = getattr(module, "BottomProcess", None)
        if process_cls is None:
            raise RuntimeError("bottom_process module has no BottomProcess public API")

        self.workflow_signals.ui.emit({
            "state": "하의 V114 시작 준비",
            "step": "4 / 4 · STANDBY 복귀 → V114",
            "progress": 72,
            "detected": f"하의 · 안전 높이 경유 후 STANDBY 정규화 · baseline={state.baseline_id[:10]}",
            "stop_enabled": True,
            "integration_status": "카메라 LIVE · BOTTOM STARTUP",
        })
        print(
            f"[MAIN-BOTTOM-START] baseline_id={state.baseline_id} "
            "shared camera retained; startup standby normalization enabled"
        )

        process = process_cls(
            argv=self._bottom_vla_argv(),
            debug=False,
            external_frame_provider=self.shared_raw_camera,
        )
        with self._active_bottom_lock:
            self._active_bottom_process = process

        try:
            if self.workflow_stop.is_set():
                print("[MAIN-BOTTOM-SKIP] stop requested before V114 start")
                return None

            process.start(timeout=max(5.0, float(self.args.bottom_start_timeout_s)))

            if self.workflow_stop.is_set():
                process.request_stop("stop requested during V114 startup")
                return None

            applied = process.set_empty_board_state(state.clone(), timeout=15.0)
            if applied is not True:
                raise RuntimeError("V114 rejected V22 EmptyBoardState")
            print(f"[MAIN-BOTTOM-EMPTY] baseline_id={state.baseline_id} installed")

            if self.workflow_stop.is_set():
                process.request_stop("stop requested before V114 cycle")
                return None

            self._begin_vla_watch("bottom")
            result = process.run_cycle(timeout=max(10.0, float(self.args.bottom_cycle_timeout_s)))
            status = str(getattr(result, "status", ""))
            reason = str(getattr(result, "reason", ""))
            recovery = bool(getattr(result, "recovery_required", False))
            print(
                f"[MAIN-BOTTOM-RETURN] status={status} "
                f"recovery_required={recovery} reason={reason}"
            )

            if self._force_fold_for("bottom") and status in {"STOPPED", "ABORTED", "ERROR_STOP"}:
                # V114 STOP returns before all worker cleanup has necessarily finished.
                # Wait for quiescence, then reuse its proven pure-Z -> camera-clear
                # standby routine before allowing any folding plate to move.
                standby_ok = bool(process.prepare_force_fold_standby(timeout=35.0))
                if not standby_ok:
                    raise RuntimeError(
                        "FORCE_FOLD_BOTTOM_STANDBY_UNSAFE: cooperative stop completed but verified "
                        "camera-clear standby could not be established; automatic folding blocked"
                    )
                self._mark_force_fold_stop_completed("bottom")
                self.workflow_signals.log.emit(
                    "하의 V114 안전 중단 + camera-clear STANDBY 확인 완료 · FINISH 라벨 없이 기존 자동 폴딩 경로로 이동"
                )
                print(f"[MAIN-BOTTOM-FORCE-FOLD-STOPPED] status={status} standby=CONFIRMED", flush=True)
                return result
            if status == "STOPPED" and self.workflow_stop.is_set():
                return result
            if status != "SUCCESS":
                raise RuntimeError(
                    f"bottom V114 did not finish: status={status} "
                    f"recovery_required={recovery} reason={reason}"
                )
            self.workflow_signals.ui.emit({"current_action": "FINISH"})
            return result
        finally:
            try:
                process.close(timeout=65.0)
            finally:
                self._end_vla_watch()
                with self._active_bottom_lock:
                    if self._active_bottom_process is process:
                        self._active_bottom_process = None


    def _verify_roarms_safe_for_auto_fold(self) -> None:
        """Check-only T:105 gate before plate motion; never moves either arm."""
        ready, reason = self._plate_standby_precheck()
        if not ready:
            raise RuntimeError(f"AUTO_FOLD_ROARM_PRECHECK_FAILED: {reason}")

        targets = self._plate_standby_targets()
        arms: Dict[str, Any] = {}
        try:
            for arm_key in ("arm1", "arm2"):
                port = self.args.arm1_port if arm_key == "arm1" else self.args.arm2_port
                arms[arm_key] = v22.UiRoArmSerial(
                    port,
                    int(self.args.roarm_baud),
                    label=f"auto-fold-check-{arm_key}",
                )

            consecutive_required = max(
                1, int(self.args.plate_standby_arrival_consecutive)
            )
            for arm_key in ("arm1", "arm2"):
                arm = arms[arm_key]
                target = targets[arm_key]
                consecutive = 0
                last_diag: Optional[str] = None
                deadline = time.monotonic() + max(
                    0.5, float(self.args.plate_standby_arrival_timeout_s)
                )
                while time.monotonic() < deadline:
                    feedback = arm.feedback(
                        timeout=float(self.args.roarm_feedback_timeout_s),
                        retries=int(self.args.roarm_feedback_retries),
                    )
                    if feedback is None:
                        consecutive = 0
                        last_diag = "no T:105 feedback"
                        continue
                    x, y, z, _ = self._feedback_xyz(feedback)
                    xy_error = math.hypot(
                        x - float(target["x"]),
                        y - float(target["y"]),
                    )
                    z_error = abs(z - float(target["z"]))
                    last_diag = (
                        f"actual=({x:.1f},{y:.1f},{z:.1f}) "
                        f"target=({target['x']:.1f},{target['y']:.1f},{target['z']:.1f}) "
                        f"xy_error={xy_error:.1f} z_error={z_error:.1f}"
                    )
                    if (
                        xy_error <= float(self.args.plate_standby_xy_tolerance_mm)
                        and z_error <= float(self.args.plate_standby_z_tolerance_mm)
                    ):
                        consecutive += 1
                        if consecutive >= consecutive_required:
                            self.workflow_signals.log.emit(
                                f"자동 폴딩 안전 확인 {arm_key.upper()} PASS · {last_diag}"
                            )
                            break
                    else:
                        consecutive = 0
                    time.sleep(max(0.02, float(self.args.plate_standby_poll_interval_s)))
                else:
                    raise RuntimeError(
                        f"AUTO_FOLD_ROARM_UNSAFE:{arm_key.upper()}:{last_diag}"
                    )
        finally:
            for arm in arms.values():
                try:
                    arm.close()
                except Exception:
                    pass

    def _plate_controller_for_auto_fold(self):
        controller = self.controller
        if controller is None:
            raise RuntimeError("AUTO_FOLD_ARDUINO_CONTROLLER_MISSING")
        if not controller.connected:
            raise RuntimeError("AUTO_FOLD_ARDUINO_NOT_CONNECTED")
        if not controller.ready:
            raise RuntimeError("AUTO_FOLD_ARDUINO_NOT_READY")
        if not bool(getattr(controller, "protocol_ok", False)):
            raise RuntimeError(
                "AUTO_FOLD_ARDUINO_PROTOCOL_MISMATCH:"
                f"{getattr(controller, 'firmware_id', '')}"
            )
        if not bool(controller.send_enabled):
            raise RuntimeError("AUTO_FOLD_ARDUINO_SEND_DISABLED")
        return controller

    def _auto_fold_sleep(self, seconds: float, *, label: str) -> None:
        """Interruptible physical hold/settle delay for plate motion."""
        seconds = max(0.0, float(seconds))
        if seconds <= 0.0:
            return
        self.workflow_signals.log.emit(
            f"자동 폴딩 {label} · {seconds:.1f}초 대기"
        )
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            if self.workflow_stop.is_set():
                raise RuntimeError(f"AUTO_FOLD_CANCELLED_DURING_{label}")
            time.sleep(min(0.05, max(0.0, deadline - time.monotonic())))

    def _auto_fold_plate(
        self,
        controller,
        plate: int,
        *,
        wait_for: str,
        label: str,
        hold_after_s: float = 0.0,
    ) -> None:
        result = str(
            controller.operate_plate_and_wait(
                int(plate),
                wait_for=str(wait_for),
                dry_run=False,
                timeout_s=10.0,
            )
        )
        expected = "OPERATED" if str(wait_for) == "operated" else "HOME"
        self.workflow_signals.log.emit(
            f"자동 폴딩 {label} · plate={plate} · result={result}"
        )
        print(
            f"[MAIN-AUTO-FOLD-PLATE] label={label} plate={plate} "
            f"wait_for={wait_for} result={result}",
            flush=True,
        )
        if result != expected:
            raise RuntimeError(
                f"AUTO_FOLD_PLATE_FAILED:plate={plate}:wait={wait_for}:"
                f"result={result}:detail={controller.last_error}"
            )
        if float(hold_after_s) > 0.0:
            self._auto_fold_sleep(
                float(hold_after_s),
                label=f"{label} 물리 HOLD",
            )

    def _auto_fold_home(
        self,
        controller,
        required_plates,
        *,
        label: str,
        settle_after_s: float = 1.0,
    ) -> None:
        required = tuple(int(value) for value in required_plates)
        result = str(
            controller.home_and_wait(
                required,
                dry_run=False,
                timeout_s=4.0,
            )
        )
        self.workflow_signals.log.emit(
            f"자동 폴딩 {label} · HOME plates={required} · result={result}"
        )
        print(
            f"[MAIN-AUTO-FOLD-HOME] label={label} required={required} "
            f"result={result}",
            flush=True,
        )
        if result != "HOME":
            raise RuntimeError(
                f"AUTO_FOLD_HOME_FAILED:required={required}:"
                f"result={result}:detail={controller.last_error}"
            )
        # Firmware ACK is emitted immediately after servo.write(startAngle), not from
        # physical position feedback.  Give the mechanisms time to reach HOME.
        self._auto_fold_sleep(
            float(settle_after_s),
            label=f"{label} 물리 안정화",
        )

    def _auto_fold_plate3_cycle(self, controller) -> None:
        """Firmware plate commands are toggles: first 3=UP, second 3=slow HOME."""
        self._auto_fold_plate(
            controller,
            3,
            wait_for="operated",
            label="3번 판 상승",
            hold_after_s=1.0,
        )
        self._auto_fold_plate(
            controller,
            3,
            wait_for="home",
            label="3번 판 HOME 복귀",
            hold_after_s=0.0,
        )
        self._auto_fold_sleep(1.0, label="3번 판 HOME 물리 안정화")

    def _run_post_finish_folding(self, garment: str) -> EmptyBoardState:
        """FINISH -> 1 s -> requested plate sequence -> refresh baseline.

        The upper/bottom runtimes already return only after their own camera-clear/
        standby handling.  The automatic FINISH path therefore does not command or
        verify a second, unrelated folding-plate standby pose.  Manual plate-control
        paths keep the existing RoArm preparation/safety helpers unchanged.
        """
        kind = str(garment).strip().lower()
        if kind not in {"upper", "bottom"}:
            raise ValueError(f"unsupported folding garment={garment!r}")

        with self._integration_plate_lock:
            if self._integration_plate_busy:
                raise RuntimeError("AUTO_FOLD_ALREADY_RUNNING")
            self._integration_plate_busy = True

        try:
            self.workflow_signals.ui.emit({
                "state": "폴딩 준비",
                "step": "FINISH 확인 · 1초 안전 대기",
                "progress": 90,
                "detected": (
                    "상의 FINISH · 자동 폴딩 준비"
                    if kind == "upper"
                    else "하의 FINISH · 자동 폴딩 준비"
                ),
                "start_enabled": False,
                "stop_enabled": False,
                "reset_enabled": False,
                "empty_board_refresh_enabled": False,
                "integration_status": "카메라 LIVE · AUTO FOLD READY",
            })

            # User-requested delay after FINISH.  Upper/lower runtimes already
            # complete their own post-action standby/camera-clear handling before
            # returning SUCCESS.  Do not impose the separate manual folding-plate
            # standby pose here: that second coordinate contract caused a false
            # AUTO_FOLD_ROARM_UNSAFE stop after a valid VLA FINISH.
            deadline = time.monotonic() + 1.0
            while time.monotonic() < deadline:
                if self.workflow_stop.is_set():
                    raise RuntimeError("AUTO_FOLD_CANCELLED_BEFORE_START")
                time.sleep(min(0.05, max(0.0, deadline - time.monotonic())))

            print(
                "[MAIN-AUTO-FOLD-DIRECT] VLA standby accepted; "
                "dedicated plate-standby move/check bypassed",
                flush=True,
            )
            self.workflow_signals.log.emit(
                "VLA 종료 대기 위치를 유지하고 자동 폴딩판 시퀀스를 시작합니다."
            )

            controller = self._plate_controller_for_auto_fold()
            controller.rearm("main FINISH automatic folding", force=True)

            self.workflow_signals.ui.emit({
                "state": "자동 접기",
                "step": "폴딩판 시퀀스 실행 중",
                "progress": 94,
                "detected": "폴딩판 자동 동작 중",
                "integration_status": "카메라 LIVE · AUTO FOLD",
            })

            # V21: upper and bottom use the same large-plate sequence.
            #   1 up -> 2 up -> both HOME
            #
            # This common helper is used by both normal FINISH and the 7-minute
            # cooperative force-fold path, so the sequence is identical in both cases.
            self._auto_fold_plate(
                controller,
                1,
                wait_for="operated",
                label="1번 판 상승",
                hold_after_s=1.0,
            )
            self._auto_fold_plate(
                controller,
                2,
                wait_for="operated",
                label="2번 판 상승",
                hold_after_s=1.0,
            )
            self._auto_fold_home(
                controller, (1, 2), label="1·2번 동시 HOME 복귀"
            )

            # Firmware individual plate commands are toggles.  Plate 3 therefore
            # needs two commands: UP, hold, then the second 3 for slow HOME.
            self._auto_fold_plate3_cycle(controller)

            # User-verified display behavior:
            #   plate 4 reaches/holds operated position
            #   plate 5 reaches/holds operated position while 4 stays up
            #   one global H sends 4 and 5 HOME together.
            self._auto_fold_plate(
                controller,
                4,
                wait_for="operated",
                label="4번 상승/HOLD",
                hold_after_s=1.0,
            )
            self._auto_fold_plate(
                controller,
                5,
                wait_for="operated",
                label="5번 상승/HOLD",
                hold_after_s=1.0,
            )
            self._auto_fold_home(
                controller, (4, 5), label="4·5번 동시 HOME 복귀"
            )

            self.workflow_signals.ui.emit({
                "state": "폴딩 완료",
                "step": "최종 HOME 확인 완료 · 빈 판 기준 갱신 대기",
                "progress": 98,
                "detected": "폴딩판 HOME 완료 · 1초 후 빈 판 기준 자동 촬영",
                "integration_status": "카메라 LIVE · POST-FOLD EMPTY BOARD",
            })

            # User-requested 1 s stabilization after the final HOME.
            state = self._capture_active_empty_board_baseline(
                source=(
                    "POST_FOLD_UPPER_EMPTY_BOARD"
                    if kind == "upper"
                    else "POST_FOLD_BOTTOM_EMPTY_BOARD"
                ),
                settle_s=1.0,
            )
            self.workflow_signals.log.emit(
                f"폴딩 완료 후 빈 판 기준 자동 갱신 완료 · baseline_id={state.baseline_id}"
            )
            print(
                f"[MAIN-AUTO-FOLD-COMPLETE] garment={kind} "
                f"baseline_id={state.baseline_id}",
                flush=True,
            )
            return state
        finally:
            self._integration_plate_busy = False

    def _workflow_loop(self) -> None:
        error: Optional[str] = None
        final_kind: Optional[str] = None
        try:
            mode = self._cycle_mode()
            if mode == "A":
                # Original pipeline: basket -> preprocessing/board placement first.
                self._run_garment_transfer()
                if self.workflow_stop.is_set():
                    return
                classify_step = "2 / 4 · BOARD_READY 후 모델 워밍업"
                classify_progress = 38
                classify_detected = "보드 배치 완료 · 상의/하의 분류 준비"
            else:
                # Competition/demo Mode B: the operator already placed/crumpled
                # the garment.  Do NOT run garment_transfer and do NOT recapture
                # the empty-board baseline now because the board contains clothing.
                state = self.initial_empty_board_state_copy()
                self._validate_cycle_empty_board_state(state, "mode_b_start")
                # Mode B must preserve the operator-selected arm pose during
                # classification.  Do not open RoArm serial ports or request
                # any robot feedback/motion here.  The selected upper/bottom
                # module keeps its existing startup/standby behavior after the
                # classifier has finished.
                if self.workflow_stop.is_set():
                    return
                print(
                    f"[MAIN-MODE-B-SKIP-TRANSFER] baseline_id={state.baseline_id} "
                    f"source={state.source}; entering classifier directly",
                    flush=True,
                )
                self.workflow_signals.log.emit(
                    "Mode B · garment_transfer 생략 · 로봇팔 제어 없이 분류부터 시작"
                )
                classify_step = "1 / 3 · 수동 배치 의류 분류"
                classify_progress = 30
                classify_detected = "수동 배치 완료 · 상의/하의 분류 준비"

            # Both modes share the same classifier.  After classification, Mode B
            # skips only the TOP upper_process stage; V51/BOTTOM V114 and later
            # FINISH -> auto-fold -> empty-board refresh remain shared.
            self.workflow_signals.ui.emit({
                "state": "분류 모델 준비 중",
                "step": classify_step,
                "progress": classify_progress,
                "detected": classify_detected,
                "integration_status": f"카메라 LIVE · MODE {mode} · CLASSIFIER WARMUP",
            })
            classifier = self._get_classifier()
            classifier.warmup()
            if mode == "B":
                print(
                    "[MAIN-MODE-B-SNAPSHOT-CLASSIFIER] "
                    "fresh-frame median snapshot path enabled; stillness/motion gate disabled",
                    flush=True,
                )
                result = self._classify_mode_b_snapshot()
            else:
                result = self._classify_until_locked()
            if result is None:
                return

            garment = "upper" if result.garment == "UPPER" else "bottom"
            self.current_garment = garment
            print(
                f"[MAIN-ROUTE] LOCK garment={garment} confidence={result.confidence:.3f} {result.detail}"
            )

            if garment == "bottom":
                self._unload_classifier()
                print(
                    "[MAIN-BOTTOM-HANDOFF] BOTTOM classified; "
                    "starting clean V114 module without BASKET_GRASP"
                )
                self._run_bottom_process()
                if self.workflow_stop.is_set():
                    print("[MAIN-STOPPED] V114 cooperative stop; no bottom FINISH success UI")
                    return
                forced_fold = self._force_fold_stop_completed_for("bottom")
                self._run_post_finish_folding("bottom")
                final_kind = "bottom"
                self.workflow_signals.ui.emit({
                    "state": "하의 통합 완료",
                    "step": "자동 폴딩 + 빈 판 기준 갱신 완료",
                    "progress": 100,
                    "detected": (
                        "하의 · 7분 초과 안전 중단 후 자동 폴딩 완료 · 빈 판 기준 갱신 완료"
                        if forced_fold else
                        "하의 FINISH · 자동 폴딩 완료 · 다음 빈 판 기준 갱신 완료"
                    ),
                    "start_enabled": False,
                    "stop_enabled": False,
                    "reset_enabled": True,
                    "empty_board_refresh_enabled": False,
                    "current_action": "중단 후 접기" if forced_fold else "FINISH",
                    "timeout_prompt_visible": False,
                    "timeout_button_enabled": False,
                    "integration_status": "카메라 LIVE · BOTTOM COMPLETE",
                })
                print(
                    "[MAIN-INTEGRATION-COMPLETE] BOTTOM "
                    + ("FORCE_FOLD_AFTER_7MIN" if forced_fold else "FINISH")
                    + " + AUTO FOLD + EMPTY BOARD REFRESH; V22 camera still LIVE"
                )
                self._clear_force_fold_state()
                return

            # 3) TOP: unload classifier GPU objects but NEVER release the camera.
            self._unload_classifier()
            if mode == "A":
                # Original pipeline: classifier -> upper_process preprocessing/post-processing
                # -> V51 smoothing.  Preserve this path exactly for Mode A.
                print(
                    "[MAIN-TOP-HANDOFF] UPPER classified; camera remains LIVE; "
                    "Mode A entering upper_process"
                )
                self._run_upper_process()
                if self.workflow_stop.is_set():
                    print(
                        "[MAIN-STOPPED] stop requested after upper_process; "
                        "V51 cycle will not start"
                    )
                    return
                print(
                    "[MAIN-UPPER-TO-VLA-HANDOFF] upper_process complete; "
                    "entering headless V51 through public wrapper"
                )
            else:
                # Demo Mode B: the operator manually places the garment in the
                # board state that would normally exist AFTER upper_process.
                # Therefore upper_process must not run here.  Start V51 directly.
                print(
                    "[MAIN-MODE-B-SKIP-UPPER-PROCESS] UPPER classified; "
                    "operator placement accepted as post-upper_process state; "
                    "entering V51 directly"
                )
                self.workflow_signals.log.emit(
                    "Mode B · 상의 upper_process 생략 · V51 평활화 바로 시작"
                )

            self._run_upper_vla_process()
            if self.workflow_stop.is_set():
                print("[MAIN-STOPPED] v44 cooperative abort completed; workflow ends without FINISH success UI")
                return
            forced_fold = self._force_fold_stop_completed_for("upper")
            self._run_post_finish_folding("upper")
            final_kind = "upper"
            self.workflow_signals.ui.emit({
                "state": "상의 통합 완료",
                "step": "자동 폴딩 + 빈 판 기준 갱신 완료",
                "progress": 100,
                "detected": (
                    "상의 · 7분 초과 안전 중단 후 자동 폴딩 완료 · 빈 판 기준 갱신 완료"
                    if forced_fold else
                    "상의 FINISH · 자동 폴딩 완료 · 다음 빈 판 기준 갱신 완료"
                ),
                "start_enabled": False,
                "stop_enabled": False,
                "reset_enabled": True,
                "empty_board_refresh_enabled": False,
                "current_action": "중단 후 접기" if forced_fold else "FINISH",
                "timeout_prompt_visible": False,
                "timeout_button_enabled": False,
                "integration_status": "카메라 LIVE · UPPER COMPLETE",
            })
            print(
                "[MAIN-INTEGRATION-COMPLETE] UPPER "
                + ("FORCE_FOLD_AFTER_7MIN" if forced_fold else "FINISH")
                + " + AUTO FOLD + EMPTY BOARD REFRESH; V22 camera still LIVE"
            )
            self._clear_force_fold_state()
        except Exception as exc:
            error = repr(exc)
            print(f"[MAIN-ERROR] {error}")
            self.workflow_signals.log.emit(f"MAIN ERROR: {error}")
        finally:
            self._unload_classifier()
            self._end_vla_watch()
            self._active_cycle = False
            self.current_garment = final_kind
            self.workflow_signals.ui.emit({
                "timeout_prompt_visible": False,
                "timeout_button_enabled": False,
                "empty_board_refresh_enabled": bool(
                    self._initial_empty_board_ready.is_set()
                    and not self._integration_plate_busy
                ),
                "mode_enabled": bool(
                    self._initial_empty_board_ready.is_set()
                    and not self._integration_plate_busy
                ),
            })
            if error:
                self.workflow_signals.ui.emit({
                    "state": "확인 필요",
                    "step": "통합 작업 중단 · 로그 확인",
                    "progress": 0,
                    "detected": error[:90],
                    "start_enabled": False,
                    "stop_enabled": False,
                    "reset_enabled": True,
                    "integration_status": "카메라 LIVE · ERROR",
                })
            elif final_kind is None:
                self.workflow_signals.ui.emit({
                    "state": "대기 중",
                    "step": "0 / 4 · 운전 종료",
                    "progress": 0,
                    "detected": "작업이 종료되었습니다.",
                    "start_enabled": False,
                    "stop_enabled": False,
                    "reset_enabled": True,
                    "integration_status": "카메라 LIVE · STOPPED",
                })

    def cleanup(self) -> None:
        if self._main_closing:
            return
        self._main_closing = True
        self.workflow_stop.set()
        with self._active_bottom_lock:
            active_bottom = self._active_bottom_process
        if active_bottom is not None:
            try:
                active_bottom.request_stop("main UI/application closing")
                active_bottom.close(timeout=65.0)
            except Exception as exc:
                print(f"[MAIN-BOTTOM-CLEANUP-WARN] {exc!r}")
            finally:
                with self._active_bottom_lock:
                    if self._active_bottom_process is active_bottom:
                        self._active_bottom_process = None
        with self._active_upper_vla_lock:
            active_vla = self._active_upper_vla_process
        if active_vla is not None:
            try:
                active_vla.request_stop("main UI/application closing")
                active_vla.close(timeout=65.0)
            except Exception as exc:
                print(f"[MAIN-UPPER-VLA-CLEANUP-WARN] {exc!r}")
            finally:
                with self._active_upper_vla_lock:
                    if self._active_upper_vla_process is active_vla:
                        self._active_upper_vla_process = None
        # V10 transfer-stop safety: workflow_stop is also consumed by
        # garment_transfer.  Wait for that worker to leave FIX11/D50 and release
        # its copied-frame lease BEFORE the sole V22 camera owner is shut down.
        workflow = self.workflow_thread
        if (
            workflow is not None
            and workflow.is_alive()
            and threading.current_thread() is not workflow
        ):
            print(
                "[MAIN-CLEANUP-WAIT] waiting for workflow/garment_transfer cooperative stop "
                "before shared-camera shutdown"
            )
            workflow.join(timeout=65.0)
        if workflow is not None and workflow.is_alive():
            print(
                "[MAIN-CLEANUP-CAMERA-DEFERRED] workflow still active after 65s; "
                "physical camera release is intentionally skipped to avoid provider-shutdown/core-dump race"
            )
            self._unload_classifier()
            return
        try:
            self.shared_raw_camera.shutdown()
        except Exception:
            pass
        self._unload_classifier()
        # The inherited cleanup is the ONE place where V22 may finally release
        # its physical VideoCapture, because the application itself is exiting.
        super().cleanup()


# -----------------------------------------------------------------------------
# Combined argument parser: strip main-only flags, then reuse V22's own parser.
# -----------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(add_help=False)
    # Compatibility only: legacy wrappers may still pass --routing-mode classify.
    p.add_argument("--routing-mode", choices=("classify", "dispatch", "integrated"), default="integrated")
    p.add_argument("--physical", action="store_true", default=False)
    p.add_argument("--integration-self-test", action="store_true", default=False)
    p.add_argument("--integration-self-test-core-only", action="store_true", default=False)
    p.add_argument("--initial-empty-board-warmup-frames", type=int, default=12)
    p.add_argument("--initial-empty-board-frame-timeout-s", type=float, default=3.0)

    p.add_argument("--classifier-model", default=DEFAULT_SEG_MODEL)
    p.add_argument("--garment-classify-model", default=DEFAULT_TOP_BOTTOM_MODEL)
    p.add_argument("--classifier-imgsz", type=int, default=640)
    p.add_argument("--garment-classify-imgsz", type=int, default=224)
    p.add_argument("--classifier-det-conf", type=float, default=0.12)
    p.add_argument("--classifier-window", type=int, default=9)
    p.add_argument("--classifier-min-votes", type=int, default=6)
    p.add_argument("--classifier-min-vote-share", type=float, default=0.70)
    p.add_argument("--classifier-min-mean-conf", type=float, default=0.70)
    p.add_argument("--classifier-min-score-margin", type=float, default=1.35)
    p.add_argument("--classifier-min-roi-overlap", type=float, default=0.15)
    p.add_argument("--classifier-min-area-ratio", type=float, default=0.003)
    p.add_argument("--classifier-frame-min-conf", type=float, default=0.55)
    p.add_argument("--classifier-frame-min-ratio", type=float, default=1.15)
    p.add_argument("--classifier-label-map", choices=("swapped", "native"), default="native")
    p.add_argument("--classifier-settle-frames", type=int, default=6)
    p.add_argument("--classifier-settle-area-tol", type=float, default=0.045)
    p.add_argument("--classifier-settle-center-tol-px", type=float, default=10.0)
    p.add_argument("--classifier-settle-bbox-tol", type=float, default=0.055)
    p.add_argument("--classifier-no-tta", action="store_true", default=False)
    p.add_argument("--classifier-no-geometry", action="store_true", default=False)
    p.add_argument("--classifier-no-rotation-rescue", action="store_true", default=False)
    p.add_argument("--classifier-rotation-rescue-after", type=int, default=3)
    p.add_argument("--classifier-rotation-rescue-period", type=int, default=4)
    p.add_argument("--classifier-rescue-conf", type=float, default=0.03)
    p.add_argument("--classifier-rotation-release-misses", type=int, default=6)
    p.add_argument("--mode-b-snapshot-settle-s", type=float, default=0.5,
                   help="Deprecated compatibility option; ignored because Mode B stillness gating is disabled.")
    p.add_argument("--mode-b-snapshot-motion-threshold", type=float, default=3.0,
                   help="Deprecated compatibility option; ignored because Mode B motion gating is disabled.")
    p.add_argument("--mode-b-snapshot-min-frames", type=int, default=5,
                   help="Deprecated compatibility option; ignored because Mode B motion gating is disabled.")
    p.add_argument("--mode-b-snapshot-burst-frames", type=int, default=5,
                   help="Mode B pixel-median burst size; clamped to 3~5 fresh raw frames.")
    p.add_argument("--mode-b-snapshot-flush-frames", type=int, default=3,
                   help="Mode B fresh frames discarded before stillness measurement.")
    p.add_argument("--mode-b-snapshot-timeout-s", type=float, default=8.0,
                   help="Maximum fresh-burst wait per Mode B snapshot attempt.")
    p.add_argument("--mode-b-snapshot-frame-timeout-s", type=float, default=2.0,
                   help="Fresh-frame wait timeout while building Mode B snapshot.")
    p.add_argument("--mode-b-snapshot-max-attempts", type=int, default=3,
                   help="Fresh median snapshots tried before Mode B classification fails closed.")
    p.add_argument("--classifier-interval-s", type=float, default=0.20)
    p.add_argument("--allow-full-frame-classification", action="store_true", default=False)

    p.add_argument("--garment-transfer-module", default=str(DEFAULT_TRANSFER_MODULE))
    p.add_argument("--upper-process-module", default=str(DEFAULT_UPPER_PROCESS_MODULE))
    p.add_argument("--upper-vla-process-module", default=str(DEFAULT_UPPER_VLA_PROCESS_MODULE))
    p.add_argument("--bottom-process-module", default=str(DEFAULT_BOTTOM_PROCESS_MODULE))
    p.add_argument("--integration-empty-board", default=str(DEFAULT_EMPTY_BOARD))
    p.add_argument("--integration-upper-vla-model", default=str(DEFAULT_UPPER_VLA_ENGINE))
    p.add_argument("--integration-upper-vla-state-norm", default=str(DEFAULT_UPPER_VLA_STATE_NORM))
    p.add_argument("--upper-vla-start-timeout-s", type=float, default=120.0)
    p.add_argument("--upper-vla-cycle-timeout-s", type=float, default=0.0)
    p.add_argument(
        "--vla-force-fold-prompt-s", type=float, default=420.0,
        help="Show the optional cooperative stop -> auto-fold prompt after this many seconds of active upper/lower VLA smoothing (default: 420 = 7 minutes).",
    )
    p.add_argument("--bottom-start-timeout-s", type=float, default=120.0)
    p.add_argument("--bottom-cycle-timeout-s", type=float, default=900.0)
    p.add_argument("--integration-config", default=str(DEFAULT_MACHINE_CONFIG))
    p.add_argument("--integration-basket-calib", default=str(DEFAULT_BASKET_CALIB))
    p.add_argument("--integration-camera-calibration", default=str(DEFAULT_CAMERA_CALIB))
    p.add_argument("--integration-seg-model", default=str(DEFAULT_SEG_ENGINE_INTEGRATION))
    p.add_argument("--integration-upper-pose-model", default=str(DEFAULT_UPPER_POSE_ENGINE))
    p.add_argument("--integration-bottom-pose-model", default=str(DEFAULT_BOTTOM_POSE_ENGINE))

    main_args, remaining = p.parse_known_args(sys.argv[1:])
    old_argv = sys.argv[:]
    try:
        sys.argv = [old_argv[0]] + remaining
        args = v22.parse_args()
    finally:
        sys.argv = old_argv
    for key, value in vars(main_args).items():
        setattr(args, key, value)
    return args


def _self_test_plate3_full_cycle(runtime_body: str) -> bool:
    import ast

    controller = next(
        (node for node in ast.parse(runtime_body).body
         if isinstance(node, ast.ClassDef) and node.name == "JerberMainController"),
        None,
    )
    if controller is None:
        return False
    cycle = next(
        (node for node in controller.body
         if isinstance(node, ast.FunctionDef) and node.name == "_auto_fold_plate3_cycle"),
        None,
    )
    if cycle is None:
        return False
    calls = [
        node.value for node in cycle.body
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)
        and isinstance(node.value.func, ast.Attribute)
        and isinstance(node.value.func.value, ast.Name)
        and node.value.func.value.id == "self"
        and node.value.func.attr == "_auto_fold_plate"
    ]
    return len(calls) == 2 and all(
        len(call.args) == 2
        and isinstance(call.args[0], ast.Name) and call.args[0].id == "controller"
        and isinstance(call.args[1], ast.Constant) and call.args[1].value == 3
        and any(keyword.arg == "wait_for" and isinstance(keyword.value, ast.Constant)
                and keyword.value.value == wait_for for keyword in call.keywords)
        for call, wait_for in zip(calls, ("operated", "home"))
    )


def _integration_self_test(args: argparse.Namespace) -> int:
    print(f"[BUILD] {BUILD}")
    print("[MAIN-INTEGRATION-SELF-TEST] camera=NOT_OPENED robot=NOT_STARTED")

    required = {
        "v24_app": HERE / "app_folding_board_test.py",
        "v24_camera_controls": HERE / "camera_v4l2_controls.py",
        "v24_ui_layout": HERE / "folding_machine.ui",
        "v24_ui_qss": HERE / "folding_machine.qss",
        "v24_ui_7ep_qss": HERE / "folding_machine_7ep.qss",
        "v24_folding_plate_image": HERE / "folding_plate.png",
        "garment_transfer_module": Path(args.garment_transfer_module).expanduser().resolve(),
        "upper_process_module": Path(args.upper_process_module).expanduser().resolve(),
        "upper_vla_process_module": Path(args.upper_vla_process_module).expanduser().resolve(),
        "bottom_process_module": Path(args.bottom_process_module).expanduser().resolve(),
        "bottom_v114_runtime": DUAL_ROOT / "bottom" / "bottom_vla_runtime.py",
        "bottom_v23": DUAL_ROOT / "bottom" / "bottom_vla-23.py",
        "bottom_main33": DUAL_ROOT / "bottom" / "main-33.py",
        "bottom_50": DUAL_ROOT / "bottom" / "50-1.py",
        "bottom_54": DUAL_ROOT / "bottom" / "54-3.py",
        "bottom_55": DUAL_ROOT / "bottom" / "55-5.py",
        "bottom_58": DUAL_ROOT / "bottom" / "58-3.py",
        "bottom_60": DUAL_ROOT / "bottom" / "60-15.py",
        "bottom_align": DUAL_ROOT / "bottom" / "align-11.py",
        "bottom_e49": DUAL_ROOT / "bottom" / "step_e49_bottom_perception.py",
        "bottom_e62": DUAL_ROOT / "bottom" / "step_e62_bottom_perception.py",
        "bottom_d25": DUAL_ROOT / "bottom" / "step_d25_v2.py",
        "bottom_camera_undistort": DUAL_ROOT / "bottom" / "camera_undistort.py",
        "shared_empty_board_state": DEFAULT_SHARED_EMPTY_BOARD_STATE,
        "upper_vla_runtime": DUAL_ROOT / "upper_runtime_v51.py",
        "upper_vla_micro_core": DUAL_ROOT / "micro_unfold_e156_core_v51.py",
        "basket_runtime": DUAL_ROOT / "upper_basket_runtime.py",
        "board_runtime": DUAL_ROOT / "upper_board_runtime.py",
    }
    runtime_assets = {
        "config": Path(args.integration_config).expanduser().resolve(),
        "hfile": Path(args.hfile).expanduser().resolve(),
        "basket_calib": Path(args.integration_basket_calib).expanduser().resolve(),
        "camera_calibration": Path(args.integration_camera_calibration).expanduser().resolve(),
        "seg_model": Path(args.integration_seg_model).expanduser().resolve(),
        "upper_pose_model": Path(args.integration_upper_pose_model).expanduser().resolve(),
        "classifier_seg_model": Path(args.classifier_model).expanduser().resolve(),
        "top_bottom_model": Path(args.garment_classify_model).expanduser().resolve(),
        "upper_vla_model": Path(args.integration_upper_vla_model).expanduser().resolve(),
        "upper_vla_state_norm": Path(args.integration_upper_vla_state_norm).expanduser().resolve(),
        "bottom_pose_model": Path(args.integration_bottom_pose_model).expanduser().resolve(),
    }

    ok = True
    for name, path in required.items():
        exists = path.is_file()
        print(f"[MAIN-INTEGRATION-CHECK] file.{name}={exists} path={path}")
        ok = ok and exists
    if not ok:
        return 2

    core_only = bool(getattr(args, "integration_self_test_core_only", False))
    assets_ok = True
    for name, path in runtime_assets.items():
        exists = path.is_file()
        print(f"[MAIN-INTEGRATION-CHECK] runtime.{name}={exists} path={path}")
        assets_ok = assets_ok and exists
    frozen_root = Path("/workspace/project_train/models/bottom_classifier_frozen_final824")
    robust_root = MODEL_ROOT / "decision/lower/robust30"
    bottom_active = str(os.environ.get("BOTTOM_ACTION_MODEL", "robust30")).strip().lower()
    bottom_specs = (
        ("robust30", robust_root, "robust30_aug", "BOTTOM_ROBUST30_NORM", "ROBUST30-AUG"),
        ("v2", frozen_root, "v2_30d", "BOTTOM_V2_NORM", "V2"),
        ("v6", frozen_root, "v6_46d", "BOTTOM_V6_NORM", "V6"),
    )
    for tag, root, stem, norm_env, label in bottom_specs:
        override = str(os.environ.get(f"BOTTOM_{tag.upper()}_MODEL", "")).strip()
        engine = root / f"{stem}_fp16.engine"
        candidate = Path(override) if override else (
            engine if engine.is_file() else root / f"{stem}_normalized_input.onnx"
        )
        norm_override = str(os.environ.get(norm_env, "")).strip()
        normalization = Path(norm_override) if norm_override else root / f"{stem}_normalization.npz"
        both = candidate.is_file() and normalization.is_file()
        role = "ACTIVE" if tag == bottom_active else "SHADOW"
        print(
            f"[MAIN-INTEGRATION-CHECK] bottomAction.{tag} label={label} role={role} ready={both} "
            f"model={candidate} normalization={normalization}"
        )
        if tag == bottom_active:
            assets_ok = assets_ok and both
        elif not both:
            print(f"[MAIN-INTEGRATION-CHECK] WARN optional bottom {label} SHADOW not available")

    if not core_only:
        ok = ok and assets_ok
    else:
        print(f"[MAIN-INTEGRATION-CHECK] coreOnly=True runtimeAssetsReady={assets_ok}")

    own = Path(__file__).read_text(encoding="utf-8", errors="replace")
    runtime_body = own.split("def _integration_self_test(", 1)[0]
    try:
        v24_app_text = (HERE / "app_folding_board_test.py").read_text(encoding="utf-8", errors="replace")
    except Exception:
        v24_app_text = ""
    try:
        v24_controls_text = (HERE / "camera_v4l2_controls.py").read_text(encoding="utf-8", errors="replace")
    except Exception:
        v24_controls_text = ""
    try:
        bottom_runtime_text = Path(DEFAULT_BOTTOM_RUNTIME).read_text(encoding="utf-8", errors="replace")
    except Exception:
        bottom_runtime_text = ""
    try:
        bottom_process_text = Path(DEFAULT_BOTTOM_PROCESS_MODULE).read_text(encoding="utf-8", errors="replace")
    except Exception:
        bottom_process_text = ""
    try:
        plate_controller_text = (HERE / "arduino_plate_controller.py").read_text(encoding="utf-8", errors="replace")
    except Exception:
        plate_controller_text = ""
    checks = {
        "v24_training_match_autofit_present": all(token in v24_app_text for token in (
            "self._camera_autofit_p20_low = 33.0",
            "self._camera_autofit_p20_high = 45.0",
            "self._camera_autofit_p30_low = 38.0",
            "self._camera_autofit_p30_high = 65.0",
            "self._camera_autofit_p98_high = 244.0",
            "self._camera_autofit_hard_clip_p995 = 252.0",
            "self._camera_autofit_clip_max = 0.010",
            "self._camera_autofit_max_steps = 12",
            "self._camera_autofit_settle_ms = 650",
            "def _camera_autofit_rectified_board",
            "def _camera_autofit_iteration",
            "backlight_compensation",
            "exposure_dynamic_framerate",
        )),
        "v24_camera_aliases_present": all(token in v24_controls_text for token in (
            "BACKLIGHT_COMPENSATION_ALIASES",
            "EXPOSURE_DYNAMIC_FRAMERATE_ALIASES",
        )),
        "v24_main_session_tracks_fixed_extras": all(token in own for token in (
            'names.get("backlight_compensation")',
            'names.get("exposure_dynamic_framerate")',
            'must remain 0 for training-match exposure',
        )),
        "vla_runner_deleted": ("class " + "VlaProcessRunner") not in own,
        "camera_release_route_deleted": ("[MAIN-CAMERA] V22 camera " + "released") not in own,
        "shared_raw_provider_present": "class V22SharedRawCameraProvider" in own,
        "shared_provider_fix11_shape": all(token in own for token in (
            'self.window_name = "V22 UI Shared Camera"',
            "self._stop = threading.Event()",
            'self._override_status = ""',
            "self._camera_info: Dict[str, Any] = {}",
        )),
        "robot_uses_raw_provider": "module.run_integrated(self.shared_raw_camera" in own,
        "classifier_uses_corrected_frame": "frame = self.latest_camera_frame" in own,
        "bottom_v114_route_enabled": (
            "[MAIN-BOTTOM-HANDOFF]" in runtime_body
            and "def _run_bottom_process(" in runtime_body
            and "[MAIN-BOTTOM-HOLD]" not in runtime_body
        ),
        "bottom_after_classifier": (
            runtime_body.find("self._classify_until_locked()")
            < runtime_body.find("self._run_bottom_process()")
        ),
        "bottom_v114_camera_shared": "external_frame_provider=self.shared_raw_camera" in runtime_body,
        "bottom_v114_baseline_shared": "process.set_empty_board_state(state.clone()" in runtime_body,
        "bottom_startup_standby_wired": "process.start(timeout=max(5.0, float(self.args.bottom_start_timeout_s)))" in runtime_body,
        "bottom_stop_wired": (
            'bottom_process.request_stop("display UI stop requested")' in runtime_body
            and "self._active_bottom_process" in runtime_body
        ),
        "bottom_frame_none_guard_wired": (
            "def _integration_live_frame_pair" in bottom_runtime_text
            and "frame_for_action_returned_none" in bottom_runtime_text
            and "[BOTTOM-INTEGRATION-DISPLAY-WARN]" in bottom_runtime_text
        ),
        "bottom_worker_quiesce_wired": (
            "def wait_for_integrated_idle" in bottom_runtime_text
            and "close-before-serial" in bottom_process_text
            and "cycle-exception" in bottom_process_text
        ),
        "bottom_plan_overlay_wired": (
            "provider.show_override(shown, status)" in bottom_runtime_text
            and "override = provider.override_frame_copy()" in runtime_body
        ),
        "bottom_noplan_rejudge_wired": (
            "[BOTTOM-INTEGRATION-NOPLAN-REJUDGE]" in bottom_runtime_text
            and 'self._queue("REJUDGE")' in bottom_runtime_text
        ),
        "bottom_action_mask_rejudge_wired": (
            "[BOTTOM-INTEGRATION-AUTO-REJUDGE]" in bottom_runtime_text
            and 'status.upper().startswith("AUTO JUDGE: REJUDGE REQUIRED")' in bottom_runtime_text
            and "auto_rejudge_due" in bottom_runtime_text
            and "self._rejudge()" in bottom_runtime_text
        ),
        "bottom_display_cached_raw_guard_wired": (
            "live_raw_now = (" in bottom_runtime_text
            and "shown = display_now if display_now is not None else live_raw_now.copy()" in bottom_runtime_text
            and "provider.show_override(live_raw_now, status)" in bottom_runtime_text
        ),
        "bottom_robust30_active_loader_wired": (
            'os.environ.get("BOTTOM_ACTION_MODEL", "robust30")' in bottom_runtime_text
            and '"robust30_aug_fp16.engine"' in bottom_runtime_text
            and '"robust30_aug_normalization.npz"' in bottom_runtime_text
            and 'preds["robust30"] = self._v110_models["robust30"].predict(vectors["v2"])' in bottom_runtime_text
        ),
        "bottom_finish_model_owned_wired": (
            "[FINISH-V116-MODEL-OWNED]" in bottom_runtime_text
            and "legacy override disabled" in bottom_runtime_text
            and "self._v110_finish_rule_gate = False" in bottom_runtime_text
            and "[FINISH-V110-GATE-BLOCK] choose model non-FINISH" not in bottom_runtime_text
        ),
        "bottom_finish_terminal_latch_wired": (
            "[FINISH-V116-TERMINAL-LATCH]" in bottom_runtime_text
            and "[FINISH-V116-TERMINAL-BLOCK]" in bottom_runtime_text
            and "[FINISH-V116-TERMINAL-SKIP]" in bottom_runtime_text
        ),
        "transfer_before_classify": own.find("self._run_garment_transfer()") < own.find("self._classify_until_locked()"),
        "transfer_stop_wired": (
            "on_final_empty_board=self._receive_final_transfer_empty_board" in own
            and "reuse_empty_board_state=reuse" in own
            and "stop_event=self.workflow_stop" in own
            and "[MAIN-TRANSFER-STOPPED]" in own
            and "rc == 130 and self.workflow_stop.is_set()" in own
        ),
        "transfer_cleanup_waits_before_camera_shutdown": (
            "[MAIN-CLEANUP-WAIT]" in own
            and "workflow.join(timeout=65.0)" in own
            and "[MAIN-CLEANUP-CAMERA-DEFERRED]" in own
        ),
        "classify_before_upper": own.find("self._classify_until_locked()") < own.find("self._run_upper_process()"),
        "no_old_source_cli": (("--" + "fix11-source") not in own and ("--" + "d50-source") not in own),
        "clean_module_defaults": (
            'DEFAULT_TRANSFER_MODULE = DUAL_ROOT / "garment_transfer.py"' in own
            and 'DEFAULT_UPPER_PROCESS_MODULE = DUAL_ROOT / "upper_process.py"' in own
        ),
        "upper_return_log_contract": "[MAIN-UPPER-RETURN] rc={rc}" in own,
        "upper_vla_public_wrapper_path": 'DEFAULT_UPPER_VLA_PROCESS_MODULE = DUAL_ROOT / "upper_vla_process.py"' in own,
        "upper_before_vla": own.find("self._run_upper_process()") < own.find("self._run_upper_vla_process()"),
        "vla_uses_shared_provider": "external_frame_provider=self.shared_raw_camera" in own,
        "vla_stop_api_wired": (
            'process.request_stop("display UI stop requested")' in own
            and "self._active_upper_vla_process" in own
            and 'status == "ABORTED" and self.workflow_stop.is_set()' in own
        ),
        "vla_stop_enabled_in_ui": '"integration_status": "카메라 LIVE · UPPER VLA"' in own and '"stop_enabled": True' in own,
        "upper_process_stop_wired": (
            '"integration_status": "카메라 LIVE · UPPER"' in own
            and "stop_event=self.workflow_stop" in own
            and "rc == 130 and self.workflow_stop.is_set()" in own
            and "[MAIN-WORKFLOW-STOP-REQUEST]" in own
        ),
        "shared_empty_state_type": "from shared_empty_board_state import EmptyBoardState" in own,
        "shared_empty_state_copy_api": "def initial_empty_board_state_copy(self) -> EmptyBoardState:" in own,
        "initial_empty_board_raw_state": "self.initial_empty_board_raw_frame: Optional[np.ndarray] = None" in own,
        "initial_empty_board_one_shot_lock": (
            "if self.initial_empty_board_raw_frame is None:" in own
            and "self.initial_empty_board_raw_frame = locked" in own
        ),
        "initial_empty_board_copy_api": "def initial_empty_board_raw_copy(self) -> np.ndarray:" in own,
        "initial_empty_board_start_gate": "[MAIN-EMPTY-BOARD-RAW-BLOCK]" in own,
        "initial_empty_board_no_disk_dependency": (
            ("initial_empty_board_raw_frame" + " = cv2." + "imread") not in own
        ),
        "final_fix46_baseline_preserved": (
            "def _receive_final_transfer_empty_board(" in own
            and 'source="TRANSFER_FIX46_FINAL_PRE_BASKET"' in own
            and "on_final_empty_board=self._receive_final_transfer_empty_board" in own
        ),
        "auto_fold_after_both_finish": (
            'self._run_post_finish_folding("upper")' in runtime_body
            and 'self._run_post_finish_folding("bottom")' in runtime_body
        ),
        "auto_fold_direct_after_vla_standby": (
            "[MAIN-AUTO-FOLD-DIRECT]" in runtime_body
            and "AUTO_FOLD_REQUIRES_PLATE_STANDBY_SAFETY" not in runtime_body
            and "self._prepare_roarms_for_plate_motion()" not in runtime_body
            and "self._verify_roarms_safe_for_auto_fold()" not in runtime_body
        ),
        "auto_fold_45_shared_home": (
            'controller, (4, 5), label="4·5번 동시 HOME 복귀"' in runtime_body
            and "def home_and_wait(" in plate_controller_text
        ),
        "auto_fold_plate3_full_cycle": _self_test_plate3_full_cycle(runtime_body),
        "post_fold_empty_board_refresh": (
            "POST_FOLD_UPPER_EMPTY_BOARD" in runtime_body
            and "POST_FOLD_BOTTOM_EMPTY_BOARD" in runtime_body
            and "settle_s=1.0" in runtime_body
        ),
        "manual_empty_board_main_button": (
            'QPushButton("빈 판 기준 다시 촬영")' in runtime_body
            and "def _request_manual_empty_board_refresh" in runtime_body
        ),
    }
    for name, value in checks.items():
        print(f"[MAIN-INTEGRATION-CHECK] contract.{name}={bool(value)}")
        ok = ok and bool(value)

    try:
        transfer = JerberMainController._load_runtime_module(
            required["garment_transfer_module"], "selftest_garment_transfer_v8"
        )
        if not hasattr(transfer, "self_test"):
            raise RuntimeError("garment_transfer module has no self_test()")
        transfer_rc = int(transfer.self_test())
    except Exception as exc:
        print(f"[MAIN-INTEGRATION-CHECK] garment_transfer_selftest_exception={exc!r}")
        transfer_rc = 2
    print(f"[MAIN-INTEGRATION-CHECK] garment_transfer_selftest={transfer_rc == 0} rc={transfer_rc}")
    ok = ok and transfer_rc == 0

    try:
        upper = JerberMainController._load_runtime_module(
            required["upper_process_module"], "selftest_upper_process_v4"
        )
        if not hasattr(upper, "self_test"):
            raise RuntimeError("upper_process module has no self_test()")
        upper_rc = int(upper.self_test())
    except Exception as exc:
        print(f"[MAIN-INTEGRATION-CHECK] upper_process_selftest_exception={exc!r}")
        upper_rc = 2
    print(f"[MAIN-INTEGRATION-CHECK] upper_process_selftest={upper_rc == 0} rc={upper_rc}")
    ok = ok and upper_rc == 0

    try:
        upper_vla = JerberMainController._load_runtime_module(
            required["upper_vla_process_module"], "selftest_upper_vla_process_v51"
        )
        if not hasattr(upper_vla, "self_test"):
            raise RuntimeError("upper_vla_process module has no self_test()")
        upper_vla_rc = int(upper_vla.self_test())
    except Exception as exc:
        print(f"[MAIN-INTEGRATION-CHECK] upper_vla_selftest_exception={exc!r}")
        upper_vla_rc = 2
    print(f"[MAIN-INTEGRATION-CHECK] upper_vla_selftest={upper_vla_rc == 0} rc={upper_vla_rc}")
    ok = ok and upper_vla_rc == 0

    print(f"[MAIN-INTEGRATION-SELF-TEST] ALL_PASS={bool(ok)}")
    return 0 if ok else 2


def main() -> int:
    args = parse_args()
    if bool(args.integration_self_test) or bool(args.integration_self_test_core_only):
        return _integration_self_test(args)

    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    app.setFont(QFont("Noto Sans CJK KR", 12))

    qss_path = HERE / "folding_machine.qss"
    if qss_path.exists():
        app.setStyleSheet(qss_path.read_text(encoding="utf-8"))

    controller = JerberMainController(args, app)
    app.aboutToQuit.connect(controller.cleanup)
    controller.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
