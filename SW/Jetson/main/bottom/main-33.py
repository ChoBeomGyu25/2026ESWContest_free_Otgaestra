#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Action-only bottom-garment VLA collector with read-only D54/D55/D56 adapters.

Runtime authority
-----------------
* Camera, Homography, models, robot serial connections and the
  authoritative folding-board config are created once by this process.
* The three source files are imported read-only and are never rewritten.
* D54/D55 keep their original corrected ELP frame + corrected H internally.
* D56 keeps its original raw cap.read() frame + raw_H path.
* The live operator view always stays on the D56-45 RAW camera geometry.
* Each action calls its own source module's perception, planner, overlay and
  executor.  The collector adds only the manual VLA latch and records.
* E is mandatory once per session on an empty board.  One camera instant is
  installed as corrected baseline for D54/D55 and raw baseline for D56.

Physical motion is always an exact I-locked plan and always requires Enter.
There is no six-state label.  The operator chooses only 54/55/56, judges the
result, then explicitly keeps or discards that transition for training.
P marks an I-locked but incorrect grasp/target/direction plan before Enter; it is
recorded for debugging and automatically excluded from training without motion.
"""

from __future__ import annotations

import argparse
import atexit
import copy
import dataclasses
import hashlib
import importlib
import importlib.util
import json
import math
import os
from pathlib import Path
import select
import shutil
import subprocess
import sys
import threading
import time
from types import ModuleType, SimpleNamespace
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


SCHEMA_VERSION = 6
MOTION_VERSION = "bottom_vla_manual_merged_v20_main33"
MOTION_POLICY_VERSION = "main-33_d56-real-fix55-fix56-2d-master_fix59-grip_d58-strongest-safe-correction"
D56_RECOGNITION_VERSION = "d56-45_pose_mask_waist_semantic_hardening"
D54_D55_EXTRA_GRIP_INSET_MM = 15.0
D55_EXTRA_GRIP_INSET_MM = 25.0
D54_PULL_GAIN = 0.85
D54_PULL_MIN_MM = 110.0
D54_PULL_MAX_MM = 180.0
D54_PULL_HARD_MAX_MM = 200.0
D54_PULL_EXTRA_MM = 50.0
D56_TEMPORAL_FRAMES = 1
D56_TEMPORAL_INTERVAL_S = 0.12


# Taught laydown trajectory profile.
# Only the normalized longitudinal timing/reversal shape is embedded here.
# Upper-garment perception, grasp selection, canonical XYZ, and training logic
# are not imported; lower-garment perception and waist grasp planning remain
# authoritative.
MAIN28_D56_TAUGHT_PROFILE = (
    (0.000000, -0.000000, 0.000000000),
    (0.243636, -3.921540, -0.000892114),
    (0.487273, -4.551040, -0.001795960),
    (0.730909, -6.168245, -0.001844065),
    (0.974545, -21.554600, -0.000007585),
    (1.218180, -39.034700, -0.000055918),
    (1.461820, -56.359600, -0.000053164),
    (1.705450, -76.974550, 0.000986806),
    (1.949090, -96.924350, 0.002984520),
    (2.192730, -115.539500, 0.013207170),
    (2.436360, -134.473000, 0.001434470),
    (2.680000, -150.187500, -0.001679800),
    (2.923640, -175.905000, 0.007579168),
    (3.167270, -203.117000, 0.002854990),
    (3.410910, -235.987500, 0.032011865),
    (3.654550, -255.731000, 0.039346300),
    (3.898180, -271.941000, 0.060063450),
    (4.141820, -285.752500, 0.090576450),
    (4.385450, -292.526500, 0.109567500),
    (4.629090, -293.905500, 0.112792000),
    (4.872730, -294.253000, 0.113179000),
    (5.116360, -294.593500, 0.113113500),
    (5.360000, -299.590000, 0.114407000),
    (5.603640, -305.008500, 0.114571500),
    (5.847270, -307.709500, 0.116884000),
    (6.090910, -308.013000, 0.118297500),
    (6.334550, -308.383000, 0.118190500),
    (6.578180, -308.332500, 0.116033000),
    (6.821820, -308.588000, 0.116392500),
    (7.065450, -308.549000, 0.116129000),
    (7.309090, -307.473000, 0.111882500),
    (7.552730, -305.817500, 0.107472500),
    (7.796360, -305.108000, 0.107628500),
    (8.040000, -302.140500, 0.101588650),
    (8.283640, -298.265500, 0.100102600),
    (8.527270, -292.083000, 0.107656000),
    (8.770910, -277.273000, 0.107544800),
    (9.014550, -258.986500, 0.114655000),
    (9.258180, -233.308000, 0.113466000),
    (9.501820, -195.567500, 0.118603000),
    (9.745450, -149.480000, 0.142393500),
    (9.989090, -88.405650, 0.187599500),
    (10.232700, -29.338600, 0.255656500),
    (10.476400, 23.498600, 0.344179500),
    (10.720000, 71.439950, 0.419880500),
    (10.963600, 112.655000, 0.516758000),
    (11.207300, 156.421500, 0.595766500),
    (11.450900, 190.823000, 0.691970000),
    (11.694500, 225.191500, 0.784209500),
    (11.938200, 258.247500, 0.865133500),
    (12.181800, 281.745000, 0.934837500),
    (12.425500, 294.818000, 0.973986500),
    (12.669100, 296.719500, 0.997815500),
    (12.912700, 302.844000, 0.995617000),
    (13.156400, 303.236500, 0.999582500),
    (13.400000, 303.095500, 1.000000000),
)
MAIN28_D56_TAUGHT_PROFILE_SHA16 = "2645f613bc21f1a5"
MAIN28_D56_SOURCE_DURATION_S = 13.400000
MAIN28_D56_SOURCE_BACK_MM = 308.588000
MAIN28_D56_SOURCE_FORWARD_MM = 303.236500


# Full per-arm 2-D taught laydown trajectory.
# The source motion is reconstructed as a board-space symmetric trajectory and
# rigidly rotated onto the detected pants body direction. Lower-garment
# segmentation, pose, waist, and grasp-point recognition remain authoritative.
MAIN33_D56_SOURCE_SHA256 = '8345bd50917eb3de51711994c462cc10bf13a532040fde7ca55642d9360f66fa'
MAIN33_D56_CANONICAL_LOCAL = {
    "arm1": (246.1141701, 28.1145506, 338.7358978),
    "arm2": (257.3242225, -36.155616, 319.4688398),
}
MAIN33_D56_SOURCE_TRAJECTORY = ((0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0), (0.243636, -0.113793, -3.75542, -0.000803518, 0.0471378, 4.08766, -0.00098071), (0.487273, -0.77219, -4.34194, -0.00261121, 0.130048, 4.76014, -0.00098071), (0.730909, -0.604324, -6.22098, -0.00270742, 0.2983, 6.11551, -0.00098071), (0.974545, 2.39755, -20.3291, 0.00096554, 1.70576, 22.7801, -0.00098071), (1.21818, 2.24101, -38.4607, 0.000868874, 2.0008, 39.6087, -0.00098071), (1.46182, 1.0853, -54.2023, 0.000874382, 1.06638, 58.5169, -0.00098071), (1.70545, -1.03598, -75.5446, 0.0026784, -1.27111, 78.4045, -0.000704789), (1.94909, -5.49367, -94.7303, 0.00278768, -3.61108, 99.1184, 0.00318136), (2.19273, -1.65627, -113.287, 0.0189248, -6.74667, 117.792, 0.00748954), (2.43636, 13.6354, -134.914, -0.00501128, -12.4153, 134.032, 0.00788022), (2.68, 16.6481, -143.286, -0.0134658, -21.7795, 157.089, 0.0101062), (2.92364, 6.13851, -175.831, -0.000541164, -19.5156, 175.979, 0.0156995), (3.16727, 1.43376, -197.647, 0.0123114, -0.375874, 208.587, -0.00660142), (3.41091, 0.897959, -233.088, 0.0548157, -6.84229, 238.887, 0.00920803), (3.65455, 9.8432, -257.638, 0.0605592, -17.5511, 253.824, 0.0181334), (3.89818, 12.6047, -270.08, 0.0697406, -18.9339, 273.802, 0.0503863), (4.14182, 8.53176, -278.197, 0.0817431, -11.8823, 293.308, 0.0994098), (4.38545, 15.7941, -291.024, 0.116166, -11.0532, 294.029, 0.102969), (4.62909, 14.7719, -293.344, 0.119962, -10.635, 294.467, 0.105622), (4.87273, 14.5134, -293.584, 0.119876, -10.7166, 294.922, 0.106482), (5.11636, 14.3011, -293.793, 0.119876, -11.2478, 295.394, 0.106351), (5.36, 12.7333, -295.304, 0.119792, -19.6774, 303.876, 0.109022), (5.60364, 1.62705, -305.975, 0.122021, -20.3239, 304.042, 0.107122), (5.84727, 1.98838, -306.109, 0.125166, -26.0197, 309.31, 0.108602), (6.09091, 2.17505, -306.318, 0.127767, -26.4709, 309.708, 0.108828), (6.33455, 2.07473, -306.206, 0.126373, -26.4825, 310.56, 0.110008), (6.57818, 1.695, -305.78, 0.121283, -26.1598, 310.885, 0.110783), (6.82182, 1.84687, -305.95, 0.120564, -25.8729, 311.226, 0.112221), (7.06545, 1.82422, -305.925, 0.120258, -25.9171, 311.173, 0.112), (7.30909, 0.625154, -304.582, 0.114797, -26.4186, 310.364, 0.108968), (7.55273, -0.152674, -302.995, 0.111562, -27.0949, 308.64, 0.103383), (7.79636, -0.123968, -302.737, 0.114609, -27.317, 307.479, 0.100648), (8.04, -1.39131, -298.175, 0.103971, -26.4857, 306.106, 0.0992063), (8.28364, -4.1678, -291.051, 0.0965352, -25.7553, 305.48, 0.10367), (8.52727, -13.631, -279.859, 0.111109, -25.833, 304.307, 0.104203), (8.77091, -23.7403, -268.226, 0.126812, -19.5774, 286.32, 0.0882776), (9.01455, -22.8117, -257.714, 0.12432, -33.6441, 260.259, 0.10499), (9.25818, -16.3989, -228.703, 0.121583, -32.6031, 237.913, 0.105349), (9.50182, -35.0878, -189.799, 0.136493, -25.6784, 201.336, 0.100713), (9.74545, -31.0159, -145.533, 0.153439, -30.6861, 153.427, 0.131348), (9.98909, -34.0764, -82.4938, 0.204064, -27.5938, 94.3175, 0.171135), (10.2327, -34.7935, -24.2842, 0.267999, -23.4123, 34.393, 0.243314), (10.4764, -43.7734, 30.2581, 0.356526, -30.4645, -16.7391, 0.331833), (10.72, -51.2346, 79.4929, 0.441814, -38.2522, -63.387, 0.397947), (10.9636, -52.0062, 121.989, 0.526172, -31.5926, -103.321, 0.507344), (11.2073, -48.5363, 165.967, 0.60677, -33.4364, -146.876, 0.584763), (11.4509, -35.005, 204.914, 0.69343, -44.9401, -176.732, 0.69051), (11.6945, -35.6794, 237.469, 0.783072, -37.9573, -212.914, 0.785347), (11.9382, -39.2221, 267.704, 0.85401, -33.5211, -248.791, 0.876257), (12.1818, -44.4132, 289.596, 0.919927, -26.1592, -273.894, 0.949748), (12.4255, -53.1687, 302.099, 0.955227, -22.2905, -287.537, 0.992746), (12.6691, -55.2485, 302.811, 0.986981, -21.4951, -290.628, 1.00865), (12.9127, -53.2343, 307.641, 0.991054, -21.4554, -298.047, 1.00018), (13.1564, -53.4845, 307.111, 0.999914, -22.6181, -299.362, 0.999251), (13.4, -53.4886, 307.104, 1.0, -22.7551, -299.087, 1.0))
MAIN33_D56_TARGET_DURATION_S = 4.07
MAIN33_D56_T104_SPEED = 0.95
MAIN33_D56_FINAL_CLEARANCE_MM = 12.0
MAIN33_D56_SOURCE_MIN_SCALE = 0.35

PHYSICAL_ACTIONS = (
    "D50_BASKET_SWING_LAYDOWN",
    "D54_OUTER_PULL",
    "D55_PRESS_SWEEP",
    "D56_WAIST_LIFT_LAYDOWN",
    "D58_CIRC_POSITION",
)

RESULT_LABELS = ("GOOD", "BAD", "SKIP")

DEFAULT_CONFIG = "/workspace/project_train/aruco_test/dual/dual_roarm_folding_board_config.json"
DEFAULT_HFILE = "/workspace/project_train/aruco_test/dual/undistort/elp_ov2710_folding_board_homography_cache.json"
DEFAULT_CALIBRATION = "/workspace/project_train/aruco_test/dual/undistort/elp_ov2710_1280x720_calibration.npz"
DEFAULT_CAMERA_CONTROLS = "/workspace/project_train/aruco_test/dual/elp_ov2710_camera_controls.json"
DEFAULT_DATASET = "/workspace/project_train/aruco_test/dual/vla/vla_training_data"
DEFAULT_SEG_MODEL = "/workspace/project_train/aruco_test/dual/models/kfashion_yolo26s_seg3_e100_best.engine"
DEFAULT_POSE_MODEL = "/workspace/project_train/yolo26/bottom_pose8_yolo26m_e40_best.engine"
DEFAULT_SOURCE_DIR = "/workspace/project_train/aruco_test/dual/undistort"
DEFAULT_D56_SOURCE = f"{DEFAULT_SOURCE_DIR}/56-45.py"
DEFAULT_D58_SOURCE = f"{DEFAULT_SOURCE_DIR}/58-2.py"
DEFAULT_D54_SOURCE = f"{DEFAULT_SOURCE_DIR}/54-3.py"
DEFAULT_D55_SOURCE = f"{DEFAULT_SOURCE_DIR}/55-5.py"
DEFAULT_D50_SOURCE = f"{DEFAULT_SOURCE_DIR}/50-1.py"


def _now_text() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def _new_id(prefix: str, counter: int) -> str:
    now = time.time()
    return f"{prefix}_{counter:06d}_{int(now * 1000)}"


_SHA256_CACHE: Dict[str, Tuple[Tuple[int, int, int], str]] = {}
_SHA256_CACHE_LOCK = threading.RLock()


def _sha256_file(path: Any) -> Optional[str]:
    """Hash immutable runtime files once, then reuse while stat metadata is unchanged.

    main-27 re-read both TensorRT engines plus source/config files for every I/observation.
    The source-integrity guarantee is preserved by keying the cache on size + mtime + ctime.
    """
    try:
        p = Path(str(path)).expanduser().resolve()
        st = p.stat()
        signature = (int(st.st_size), int(st.st_mtime_ns), int(st.st_ctime_ns))
        key = str(p)
        with _SHA256_CACHE_LOCK:
            cached = _SHA256_CACHE.get(key)
            if cached is not None and cached[0] == signature:
                return cached[1]
        h = hashlib.sha256()
        with open(p, "rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                h.update(chunk)
        digest = h.hexdigest()
        with _SHA256_CACHE_LOCK:
            _SHA256_CACHE[key] = (signature, digest)
        return digest
    except Exception:
        return None


def _json_safe(value: Any) -> Any:
    if dataclasses.is_dataclass(value):
        return _json_safe(dataclasses.asdict(value))
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items() if not str(k).startswith("_")}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if hasattr(value, "__dict__"):
        return _json_safe(vars(value))
    return str(value)


def _atomic_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(_json_safe(payload), f, ensure_ascii=False, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def _model_metadata(path: str) -> Dict[str, Any]:
    p = Path(path).expanduser().resolve()
    out: Dict[str, Any] = {"path": str(p), "exists": p.is_file()}
    if p.is_file():
        st = p.stat()
        out.update(size_bytes=int(st.st_size), mtime_ns=int(st.st_mtime_ns), sha256=_sha256_file(p))
    return out


def _require_file(path: str, label: str) -> str:
    p = Path(path).expanduser().resolve()
    if not p.is_file():
        raise FileNotFoundError(f"{label} not found: {p}")
    return str(p)


def _resolve_source(requested: str, candidates: Sequence[str], label: str) -> str:
    base = Path(__file__).resolve().parent
    tried: List[Path] = []
    for raw in (requested, *candidates):
        if not raw:
            continue
        p = Path(raw).expanduser()
        if not p.is_absolute():
            p = base / p
        p = p.resolve()
        if p in tried:
            continue
        tried.append(p)
        if p.is_file():
            print(f"[SOURCE] {label}={p}")
            return str(p)
    raise FileNotFoundError(f"{label} source unavailable; tried: {', '.join(map(str, tried))}")


def _load_module(name: str, path: str) -> ModuleType:
    source_path = Path(path).expanduser().resolve()
    source_parent = source_path.parent
    # The deployed layout keeps d54/d55/d56 in dual/undistort while the
    # authoritative E49/E62/D25 modules live one directory higher in dual/.
    # Running from undistort therefore needs BOTH directories in sys.path
    # before the legacy source executes its top-level imports.
    runtime_dirs = [
        source_parent,
        source_parent.parent,
        source_parent / "undistort",
        source_parent.parent / "undistort",
    ]
    for runtime_dir in reversed(runtime_dirs):
        if runtime_dir.is_dir():
            runtime_text = str(runtime_dir)
            if runtime_text not in sys.path:
                sys.path.insert(0, runtime_text)
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {name}: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _require_perception_imports(module: ModuleType, label: str) -> None:
    """Fail before camera/robot setup when E49 or E62 was not imported."""
    failures: List[str] = []
    loaded: List[str] = []
    for short_name, attr_name, error_attr in (
        ("E49", "e49_bottom", "_E49_IMPORT_ERROR"),
        ("E62", "e62_bottom", "_E62_IMPORT_ERROR"),
    ):
        imported = getattr(module, attr_name, None)
        if imported is None:
            failures.append(
                f"{short_name}={getattr(module, error_attr, 'unknown import error')!r}"
            )
        else:
            loaded.append(
                f"{short_name}={Path(getattr(imported, '__file__', short_name)).resolve()}"
            )
    if failures:
        source_dir = Path(getattr(module, "__file__", ".")).resolve().parent
        expected_parent = source_dir.parent
        raise RuntimeError(
            f"{label} perception import failed: {' | '.join(failures)}; "
            f"expected step_e49_bottom_perception.py and step_e62_bottom_perception.py "
            f"under {source_dir} or {expected_parent}"
        )
    print(f"[PERCEPTION-MODULE] {label} " + " ".join(loaded))


class _ParserCaptured(BaseException):
    def __init__(self, parser: argparse.ArgumentParser):
        self.parser = parser


def _module_default_args(module: ModuleType, preset: str = "physical-manual") -> argparse.Namespace:
    """Capture a legacy module's complete parser without entering hardware setup."""
    original_parse = argparse.ArgumentParser.parse_args
    original_argv = list(sys.argv)

    def capture(parser_self, *unused_args, **unused_kwargs):
        raise _ParserCaptured(parser_self)

    try:
        argparse.ArgumentParser.parse_args = capture
        sys.argv = [str(getattr(module, "__file__", "legacy.py")), f"--{preset}"]
        try:
            module.main()
        except _ParserCaptured as caught:
            parser = caught.parser
        else:
            raise RuntimeError(f"{module.__name__}.main did not reach argparse")
    finally:
        argparse.ArgumentParser.parse_args = original_parse
        sys.argv = original_argv
    return parser.parse_args([f"--{preset}"])


def _copy_shared_args(target: argparse.Namespace, common: argparse.Namespace, mode: str) -> None:
    mapping = {
        "config": common.config,
        "hfile": common.hfile,
        "seg_model": common.seg_model,
        "pose_model": common.pose_model,
        "arm1_port": common.arm1_port,
        "arm2_port": common.arm2_port,
        "move_command": common.move_command,
        "camera": common.camera,
        "width": common.width,
        "height": common.height,
        "backend": common.backend,
    }
    for key, value in mapping.items():
        if hasattr(target, key):
            setattr(target, key, value)
    target.send = mode in {"hover", "physical"}
    target.dry_run = mode == "dry-run"
    target.hover_only = mode == "hover"
    target.enter_confirm = False
    # Match the original physical/hover presets.  The selected source module's
    # outer action flow performs the one camera-clear return used for AFTER.
    target.standby_after_motion = False
    target.auto_reinfer_after_motion = False
    target.cam_auto_adjust = False
    if hasattr(target, "calib_cache"):
        target.calib_cache = ""
    if hasattr(target, "physical_auto"):
        target.physical_auto = False
    if hasattr(target, "physical_manual"):
        target.physical_manual = mode == "physical"
    if hasattr(target, "hover_check"):
        target.hover_check = mode == "hover"


def _parse_v4l2_value(text: str) -> Any:
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


def apply_camera_controls(path: str, device: str, strict: bool) -> Dict[str, Any]:
    profile = Path(_require_file(path, "camera controls JSON"))
    with open(profile, "r", encoding="utf-8") as f:
        payload = json.load(f)
    controls = payload.get("controls") if isinstance(payload, dict) else None
    if not isinstance(controls, dict):
        raise RuntimeError("camera controls JSON must contain a numeric 'controls' object")
    declared_device = str(payload.get("device") or device)
    if strict and declared_device != device:
        raise RuntimeError(f"camera device mismatch: profile={declared_device}, capture={device}")
    if shutil.which("v4l2-ctl") is None:
        raise RuntimeError("v4l2-ctl is required to apply the fixed camera profile")
    actual: Dict[str, Any] = {}
    mismatches: List[Dict[str, Any]] = []
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
    print(f"[CAMERA-CONTROL] fixed profile applied: {actual}")
    return {
        "enabled": True,
        "profile_path": str(profile),
        "profile_sha256": _sha256_file(profile),
        "device": declared_device,
        "declared_controls": controls,
        "actual_controls": actual,
        "mismatches": mismatches,
        "applied_at": _now_text(),
    }


@dataclasses.dataclass
class FrameBundle:
    raw: np.ndarray
    corrected: np.ndarray
    captured_at: str


@dataclasses.dataclass
class LockedPlan:
    state_label: Optional[str]
    action: str
    frame: FrameBundle
    observation: Any
    plan: Any
    heatmap: Any
    overlay: np.ndarray
    H: np.ndarray
    created_at: float
    plan_ok: bool
    reason: str
    planner_failure: Optional[str]
    diagnostics: Dict[str, Any]
    observation_record: Optional[Dict[str, Any]] = None


class DatasetRecorder:
    def __init__(self, root: str, cv2_module: ModuleType, runtime_metadata: Dict[str, Any],
                 board_bounds: Tuple[float, float, float, float], board_size: int,
                 official_collection: bool):
        self.root = Path(root).expanduser().resolve()
        self.cv2 = cv2_module
        self.runtime_metadata = copy.deepcopy(runtime_metadata)
        self.board_bounds = tuple(map(float, board_bounds))
        self.board_size = max(128, int(board_size))
        self.official_collection = bool(official_collection)
        self.lock = threading.RLock()
        self.obs_counter = 0
        self.decision_counter = 0
        self.transition_counter = 0
        self.garment_motion_count = 0
        self.cycle_id = time.strftime("cycle_%Y%m%d_%H%M%S") + f"_{int((time.time()%1)*1000):03d}"
        self.episode_dir = self.root / "bottom" / "episodes" / self.cycle_id
        self.current_observation: Optional[Dict[str, Any]] = None
        self.current_decision: Optional[Dict[str, Any]] = None
        self.latest_transition: Optional[Dict[str, Any]] = None
        self.terminal_status = ""
        self.initialized = False
        self.dataset_stats: Dict[str, Any] = self._empty_dataset_stats()

    def _empty_dataset_stats(self) -> Dict[str, Any]:
        return {
            "total_transitions": 0,
            "reviewed": 0,
            "eligible": 0,
            "skip": 0,
            "plan_inaccurate": 0,
            "keep": 0,
            "discard": 0,
            "by_action": {
                action: {"eligible": 0, "good": 0, "bad": 0}
                for action in PHYSICAL_ACTIONS
            },
        }

    def _iter_bottom_transition_files(self) -> List[Path]:
        """Find current and legacy bottom transition layouts without double-counting."""
        candidates: List[Path] = []
        roots = [
            self.root / "bottom" / "episodes",   # current layout
            self.root / "episodes",              # legacy layout
        ]
        seen = set()
        for transitions_root in roots:
            if not transitions_root.is_dir():
                continue
            for path in sorted(transitions_root.glob("*/transitions/*.json")):
                try:
                    key = str(path.resolve())
                except Exception:
                    key = str(path)
                if key in seen:
                    continue
                seen.add(key)
                candidates.append(path)
        return candidates

    def refresh_dataset_stats(self) -> Dict[str, Any]:
        """Rebuild persistent KEEP/eligible counts from all saved bottom transitions."""
        with self.lock:
            stats = self._empty_dataset_stats()
            stats["scan_paths"] = []
            stats["by_action_keep"] = {action: 0 for action in PHYSICAL_ACTIONS}
            files = self._iter_bottom_transition_files()
            for path in files:
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        tr = json.load(f)
                except Exception as exc:
                    print(f"[DATASET-STATS] skip unreadable transition {path}: {exc}")
                    continue

                action = str(
                    tr.get("human_selected_action")
                    or tr.get("executed_action")
                    or ""
                )
                # Legacy root/episodes can contain other garment data. Only accept
                # the three bottom D-series actions there.
                if action not in PHYSICAL_ACTIONS:
                    continue

                stats["total_transitions"] += 1
                result = str(tr.get("human_result") or "").upper()
                if result in RESULT_LABELS:
                    stats["reviewed"] += 1
                plan_review = str(tr.get("plan_review") or "").upper()
                is_plan_inaccurate = bool(
                    plan_review == "PLAN_INACCURATE"
                    or str(tr.get("status") or "").upper() == "DISCARDED_PLAN_INACCURATE"
                )
                if is_plan_inaccurate:
                    stats["plan_inaccurate"] += 1
                elif result == "SKIP":
                    stats["skip"] += 1
                decision = str(tr.get("collection_decision") or "").upper()
                if decision == "KEEP":
                    stats["keep"] += 1
                    stats["by_action_keep"][action] = stats["by_action_keep"].get(action, 0) + 1
                elif decision == "DISCARD":
                    stats["discard"] += 1

                # Historical JSON may contain a real bool or a string. Normalize it.
                raw_eligible = tr.get("training_eligible", False)
                if isinstance(raw_eligible, str):
                    is_eligible = raw_eligible.strip().lower() in {"1", "true", "yes", "y"}
                else:
                    is_eligible = bool(raw_eligible)
                if not is_eligible:
                    continue

                stats["eligible"] += 1
                if action not in stats["by_action"]:
                    stats["by_action"][action] = {"eligible": 0, "good": 0, "bad": 0}
                bucket = stats["by_action"][action]
                bucket["eligible"] += 1
                if result == "GOOD":
                    bucket["good"] += 1
                elif result == "BAD":
                    bucket["bad"] += 1

            stats["scanned_files"] = len(files)
            self.dataset_stats = stats
            return copy.deepcopy(stats)

    def dataset_stats_snapshot(self) -> Dict[str, Any]:
        with self.lock:
            return copy.deepcopy(self.dataset_stats)

    def initialize(self) -> None:
        with self.lock:
            if self.initialized:
                return
            for name in ("observations", "decisions", "transitions"):
                (self.episode_dir / name).mkdir(parents=True, exist_ok=True)
            self._update_manifest()
            episode = {
                **copy.deepcopy(self.runtime_metadata),
                "schema_version": SCHEMA_VERSION,
                "dataset_type": "macro_action_imitation_with_outcome_labels",
                "garment_type": "BOTTOM",
                "cycle_id": self.cycle_id,
                "created_at": _now_text(),
                "status": "OPEN",
                "episode_result": None,
                "planner_domain": "BOTTOM_ACTION_ONLY_D_SERIES_EXECUTORS",
                "state_labels": [],
                "physical_actions": list(PHYSICAL_ACTIONS),
                "label_policy": {
                    "human_state_labeling": "DISABLED_BY_OPERATOR_REQUEST",
                    "physical_approval": "ENTER_ONLY_EXACT_I_LOCK",
                    "post_action": list(RESULT_LABELS),
                    "collection_decision": ["KEEP", "DISCARD"],
                    "program_exit": "INCOMPLETE",
                },
            }
            _atomic_json(self.episode_dir / "episode.json", episode)
            self.initialized = True
            stats = self.refresh_dataset_stats()
            print(f"[VLA-EPISODE] OPEN {self.episode_dir}")
            print(
                f"[DATASET-STATS] persistent eligible={stats['eligible']} "
                f"D54={stats['by_action'].get('D54_OUTER_PULL', {}).get('eligible', 0)} "
                f"D55={stats['by_action'].get('D55_PRESS_SWEEP', {}).get('eligible', 0)} "
                f"D56={stats['by_action'].get('D56_WAIST_LIFT_LAYDOWN', {}).get('eligible', 0)} "
                f"skip={stats['skip']}"
            )

    def _update_manifest(self) -> None:
        path = self.root / "dataset_manifest.json"
        if path.is_file():
            try:
                with open(path, "r", encoding="utf-8") as f:
                    payload = json.load(f)
            except Exception:
                payload = {}
        else:
            payload = {}
        payload.update({
            "schema_version": SCHEMA_VERSION,
            "dataset_type": "macro_action_imitation_with_outcome_labels",
            "record_layout": "garment/episodes/cycle/{observations,decisions,transitions}",
            "bottom_state_labels": [],
            "bottom_physical_actions": list(PHYSICAL_ACTIONS),
            "human_result_labels": list(RESULT_LABELS),
            "observation_quality_labels": ["UNREVIEWED", "MASK_INACCURATE"],
            "plan_quality_labels": ["UNREVIEWED", "PLAN_INACCURATE"],
            "episode_result_labels": ["SUCCESS", "FAILED", "INCOMPLETE"],
            "motion_parameter_policy": "store exact frozen D-series plan and executor variant",
            "camera_recording": {
                "camera.png": "action-native inference frame",
                "camera_raw.png": "raw camera snapshot",
                "board.png": "512px rectified board",
                "overlay.png": "exact displayed planner overlay",
                "garment_mask.png": "planner garment mask",
            },
        })
        domains = list(payload.get("garment_domains", []) or [])
        payload["garment_domains"] = list(dict.fromkeys(domains + ["BOTTOM"]))
        payload.setdefault("created_at", _now_text())
        _atomic_json(path, payload)

    def _warp_board(self, image: np.ndarray, H: np.ndarray, nearest: bool = False) -> np.ndarray:
        x0, x1, y0, y1 = self.board_bounds
        sx = (self.board_size - 1) / max(1e-6, x1 - x0)
        sy = (self.board_size - 1) / max(1e-6, y1 - y0)
        scale = np.asarray([[sx, 0.0, -sx * x0], [0.0, sy, -sy * y0], [0.0, 0.0, 1.0]], np.float32)
        M = scale @ np.asarray(H, np.float32)
        interpolation = self.cv2.INTER_NEAREST if nearest else self.cv2.INTER_LINEAR
        return self.cv2.warpPerspective(image, M, (self.board_size, self.board_size), flags=interpolation)

    def save_observation(self, locked: LockedPlan, label: str, environment: Dict[str, Any]) -> Dict[str, Any]:
        self.initialize()
        with self.lock:
            self.obs_counter += 1
            obs_id = _new_id("obs", self.obs_counter)
            rel_base = Path("observations") / obs_id
            fs_base = self.episode_dir / rel_base
            files = {
                "camera": str(rel_base.with_name(obs_id + "_camera.png")),
                "camera_raw": str(rel_base.with_name(obs_id + "_camera_raw.png")),
                "overlay": str(rel_base.with_name(obs_id + "_overlay.png")),
                "board": str(rel_base.with_name(obs_id + "_board.png")),
                "garment_mask": str(rel_base.with_name(obs_id + "_garment_mask.png")),
            }
            mask = None
            if locked.observation is not None and getattr(locked.observation, "mask", None) is not None:
                mask = np.asarray(locked.observation.mask.mask_u8, np.uint8)
            if mask is None:
                mask = np.zeros(locked.frame.corrected.shape[:2], np.uint8)
            io_t0 = time.monotonic()
            board = self._warp_board(locked.frame.corrected, locked.H)
            warp_dt = time.monotonic() - io_t0
            # PNG level 1 is still lossless but avoids spending Jetson CPU on the
            # default high-compression path during an interactive I-lock.
            png_fast = [int(self.cv2.IMWRITE_PNG_COMPRESSION), 1]
            png_t0 = time.monotonic()
            required = {
                "camera": self.cv2.imwrite(str(self.episode_dir / files["camera"]), locked.frame.corrected, png_fast),
                "camera_raw": self.cv2.imwrite(str(self.episode_dir / files["camera_raw"]), locked.frame.raw, png_fast),
                "overlay": self.cv2.imwrite(str(self.episode_dir / files["overlay"]), locked.overlay, png_fast),
                "board": self.cv2.imwrite(str(self.episode_dir / files["board"]), board, png_fast),
                "garment_mask": self.cv2.imwrite(str(self.episode_dir / files["garment_mask"]), mask, png_fast),
            }
            if locked.heatmap is not None:
                for key, attr in (("heatmap", "heatmap"), ("binary", "binary"), ("inner_mask", "inner_mask")):
                    value = getattr(locked.heatmap, attr, None)
                    if value is None:
                        continue
                    files[key] = str(rel_base.with_name(obs_id + f"_{key}.png"))
                    required[key] = self.cv2.imwrite(
                        str(self.episode_dir / files[key]), np.asarray(value, np.uint8), png_fast
                    )
            png_dt = time.monotonic() - png_t0
            payload = {
                "schema_version": SCHEMA_VERSION,
                "garment_type": "BOTTOM",
                "cycle_id": self.cycle_id,
                "observation_id": obs_id,
                "observed_at": locked.frame.captured_at,
                "observe_label": str(label),
                "inference_serial": int(getattr(locked.observation, "bottom_vla_inference_serial", self.obs_counter)),
                "garment_motion_count": int(self.garment_motion_count),
                "human_mask_quality": "UNREVIEWED",
                "human_plan_quality": "UNREVIEWED",
                "training_input_eligible": bool(self.official_collection and all(required.values())),
                "human_state_label": locked.state_label,
                "human_selected_action": locked.action,
                "planner_state": "UNAVAILABLE",
                "planner_state_reason": "ACTION_ONLY_MANUAL_SELECTION",
                "planner_action": locked.action if locked.plan_ok else "NO_PLAN",
                "planner_failure": locked.planner_failure,
                "planner_reason": locked.reason,
                "segmentation": {
                    "source": str(getattr(locked.observation, "d38_mask_source", "UNKNOWN")),
                    "class_name": getattr(getattr(locked.observation, "mask", None), "class_name", None),
                    "confidence": getattr(getattr(locked.observation, "mask", None), "confidence", None),
                    "area_px": getattr(getattr(locked.observation, "mask", None), "area_px", None),
                },
                "environment": copy.deepcopy(environment),
                "homography": {
                    "path": self.runtime_metadata.get("source_files", {}).get("homography", {}).get("path"),
                    "sha256": self.runtime_metadata.get("homography_sha256"),
                    "matrix": _json_safe(locked.H),
                },
                "metrics": copy.deepcopy(locked.diagnostics),
                "state_summary": {
                    "human_state": locked.state_label,
                    "selected_action": locked.action,
                    "safe_frozen_plan": bool(locked.plan_ok),
                },
                "bottom_state_features": copy.deepcopy(locked.diagnostics),
                "plan": _json_safe(locked.plan),
                "files": files,
                "required_file_status": required,
            }
            json_rel = str(Path("observations") / f"{obs_id}.json")
            json_path = self.episode_dir / json_rel
            json_t0 = time.monotonic()
            _atomic_json(json_path, payload)
            json_dt = time.monotonic() - json_t0
            record = {"id": obs_id, "relative_path": json_rel, "fs_path": str(json_path), "payload": payload}
            self.current_observation = copy.deepcopy(record)
            print(f"[VLA-OBS] saved={obs_id} action={locked.action}")
            print(
                f"[PERF-IO] obs={obs_id} warp={warp_dt:.3f}s png={png_dt:.3f}s "
                f"json={json_dt:.3f}s total={(time.monotonic()-io_t0):.3f}s"
            )
            return record

    def note_motion_committed(self) -> int:
        with self.lock:
            self.garment_motion_count += 1
            return int(self.garment_motion_count)

    def save_decision(self, locked: LockedPlan, obs_record: Dict[str, Any]) -> Dict[str, Any]:
        with self.lock:
            self.decision_counter += 1
            decision_id = _new_id("decision", self.decision_counter)
            payload = {
                "schema_version": SCHEMA_VERSION,
                "cycle_id": self.cycle_id,
                "decision_id": decision_id,
                "observation_id": obs_record["id"],
                "observation_path": obs_record["relative_path"],
                "decided_at": _now_text(),
                "planner_state": "UNAVAILABLE",
                "planner_reason": "ACTION_ONLY_MANUAL_SELECTION",
                "human_state": locked.state_label,
                "human_selected_action": locked.action,
                "human_override": None,
                "planner_action": locked.action if locked.plan_ok else "NO_PLAN",
                "pending_executable": bool(locked.plan_ok),
                "planner_failure": locked.planner_failure,
                "input_mask_quality": obs_record["payload"].get("human_mask_quality", "UNREVIEWED"),
                "frozen_plan_created_at": locked.created_at,
                "frozen_plan": _json_safe(locked.plan),
            }
            rel = str(Path("decisions") / f"{decision_id}.json")
            path = self.episode_dir / rel
            _atomic_json(path, payload)
            record = {"id": decision_id, "relative_path": rel, "fs_path": str(path), "payload": payload}
            self.current_decision = copy.deepcopy(record)
            return record

    def open_transition(self, locked: LockedPlan) -> Dict[str, Any]:
        if locked.observation_record is None or self.current_decision is None:
            raise RuntimeError("observation/decision must be saved before transition")
        with self.lock:
            if self.latest_transition is not None:
                previous = self.latest_transition["payload"]
                if not previous.get("human_result"):
                    raise RuntimeError("previous transition requires GOOD/BAD/SKIP")
                if previous.get("collection_decision") is None:
                    raise RuntimeError("previous transition requires KEEP/DISCARD")
            self.transition_counter += 1
            transition_id = _new_id("transition", self.transition_counter)
            before = locked.observation_record
            payload = {
                "schema_version": SCHEMA_VERSION,
                "cycle_id": self.cycle_id,
                "transition_id": transition_id,
                "decision_id": self.current_decision["id"],
                "opened_at": _now_text(),
                "before_observation_id": before["id"],
                "before_observation_path": before["relative_path"],
                "before_metrics": copy.deepcopy(locked.diagnostics),
                "before_mask_quality": before["payload"].get("human_mask_quality", "UNREVIEWED"),
                "planner_state": "UNAVAILABLE",
                "human_state": locked.state_label,
                "human_selected_action": locked.action,
                "human_override": None,
                "executed_action": locked.action,
                "executed_macro_action": "SPREAD",
                "action_variant": str(getattr(locked.plan, "action", locked.action)),
                "motion_parameters": _json_safe(locked.plan),
                "execution_success": False,
                "execution_sent": False,
                "garment_motion_committed": False,
                "execution_detail": "PENDING",
                "after_observation_id": None,
                "after_observation_path": None,
                "after_metrics": {},
                "after_mask_quality": None,
                "automatic_result": None,
                "human_result": None,
                "collection_decision": None,
                "user_selected_for_collection": None,
                "training_eligible": False,
                "status": "OPEN",
                "collection_phase": "official" if self.official_collection else "motion_validation",
            }
            rel = str(Path("transitions") / f"{transition_id}.json")
            path = self.episode_dir / rel
            _atomic_json(path, payload)
            record = {"id": transition_id, "relative_path": rel, "fs_path": str(path), "payload": payload}
            self.latest_transition = record
            return record

    def complete_transition(self, execution: Dict[str, Any], after: Optional[Dict[str, Any]]) -> None:
        with self.lock:
            if self.latest_transition is None:
                return
            tr = self.latest_transition["payload"]
            tr.update(_json_safe(execution))
            if after is not None:
                tr.update({
                    "after_observation_id": after["id"],
                    "after_observation_path": after["relative_path"],
                    "after_metrics": after["payload"].get("bottom_state_features", {}),
                    "after_mask_quality": after["payload"].get("human_mask_quality", "UNREVIEWED"),
                })
            tr["completed_at"] = _now_text()
            tr["status"] = "AWAITING_HUMAN_RESULT"
            _atomic_json(Path(self.latest_transition["fs_path"]), tr)

    def require_result(self) -> bool:
        with self.lock:
            return bool(self.latest_transition is not None and not self.latest_transition["payload"].get("human_result"))

    def require_collection_decision(self) -> bool:
        with self.lock:
            if self.latest_transition is None:
                return False
            tr = self.latest_transition["payload"]
            return bool(tr.get("human_result") and tr.get("collection_decision") is None)

    def review_pending(self) -> bool:
        return bool(self.require_result() or self.require_collection_decision())

    def label_result(self, label: str) -> bool:
        label = str(label).upper()
        if label not in RESULT_LABELS:
            return False
        with self.lock:
            if self.latest_transition is None or self.latest_transition["payload"].get("human_result"):
                print("[VLA-LABEL] no unlabeled transition")
                return False
            tr = self.latest_transition["payload"]
            tr["human_result"] = label
            tr["human_result_at"] = _now_text()
            tr["training_eligible"] = False
            tr["status"] = "AWAITING_COLLECTION_DECISION"
            _atomic_json(Path(self.latest_transition["fs_path"]), tr)
            print(f"[VLA-LABEL] {label} transition={tr['transition_id']}")
            return True

    def decide_collection(self, keep: bool) -> bool:
        with self.lock:
            if self.latest_transition is None:
                print("[VLA-COLLECT] no reviewed transition")
                return False
            tr = self.latest_transition["payload"]
            if not tr.get("human_result"):
                print("[VLA-COLLECT] judge G/B/K first")
                return False
            if tr.get("collection_decision") is not None:
                print("[VLA-COLLECT] collection decision already saved")
                return False
            eligible = bool(
                keep
                and self.official_collection
                and tr.get("human_result") in {"GOOD", "BAD"}
                and tr.get("execution_success")
                and tr.get("execution_sent")
                and tr.get("garment_motion_committed")
                and tr.get("before_observation_id")
                and tr.get("after_observation_id")
                and tr.get("before_mask_quality") != "MASK_INACCURATE"
                and tr.get("after_mask_quality") != "MASK_INACCURATE"
            )
            eligibility_missing = []
            if keep and not self.official_collection:
                eligibility_missing.append("collection_phase_not_official")
            if keep and tr.get("human_result") not in {"GOOD", "BAD"}:
                eligibility_missing.append("result_not_good_or_bad")
            if keep and not tr.get("execution_success"):
                eligibility_missing.append("execution_success_false")
            if keep and not tr.get("execution_sent"):
                eligibility_missing.append("execution_sent_false")
            if keep and not tr.get("garment_motion_committed"):
                eligibility_missing.append("garment_motion_committed_false")
            if keep and not tr.get("before_observation_id"):
                eligibility_missing.append("before_missing")
            if keep and not tr.get("after_observation_id"):
                eligibility_missing.append("after_missing")
            if keep and tr.get("before_mask_quality") == "MASK_INACCURATE":
                eligibility_missing.append("before_mask_inaccurate")
            if keep and tr.get("after_mask_quality") == "MASK_INACCURATE":
                eligibility_missing.append("after_mask_inaccurate")
            tr["training_eligibility_reason"] = "OK" if eligible else ("|".join(eligibility_missing) or "not_kept")
            tr["collection_decision"] = "KEEP" if keep else "DISCARD"
            tr["collection_decision_at"] = _now_text()
            tr["user_selected_for_collection"] = bool(keep)
            tr["training_eligible"] = eligible
            tr["status"] = "SELECTED_FOR_COLLECTION" if keep else "DISCARDED_BY_USER"
            _atomic_json(Path(self.latest_transition["fs_path"]), tr)
            # main-28: the current transition was created after the startup full
            # scan, so update only its counters. A full authoritative rebuild still
            # happens once on the next program start. This prevents Y/N latency from
            # growing linearly with the entire historical dataset.
            stats = self.dataset_stats
            stats.setdefault("by_action_keep", {action: 0 for action in PHYSICAL_ACTIONS})
            stats["total_transitions"] = int(stats.get("total_transitions", 0)) + 1
            stats["reviewed"] = int(stats.get("reviewed", 0)) + 1
            result_now = str(tr.get("human_result") or "").upper()
            action_now = str(tr.get("human_selected_action") or tr.get("executed_action") or "")
            if result_now == "SKIP":
                stats["skip"] = int(stats.get("skip", 0)) + 1
            if keep:
                stats["keep"] = int(stats.get("keep", 0)) + 1
                stats["by_action_keep"][action_now] = int(stats["by_action_keep"].get(action_now, 0)) + 1
            else:
                stats["discard"] = int(stats.get("discard", 0)) + 1
            if eligible:
                stats["eligible"] = int(stats.get("eligible", 0)) + 1
                bucket = stats.setdefault("by_action", {}).setdefault(
                    action_now, {"eligible": 0, "good": 0, "bad": 0}
                )
                bucket["eligible"] = int(bucket.get("eligible", 0)) + 1
                if result_now == "GOOD":
                    bucket["good"] = int(bucket.get("good", 0)) + 1
                elif result_now == "BAD":
                    bucket["bad"] = int(bucket.get("bad", 0)) + 1
            self.dataset_stats = stats
            stats = copy.deepcopy(stats)
            print(
                f"[VLA-COLLECT] {tr['collection_decision']} "
                f"transition={tr['transition_id']} trainingEligible={eligible}"
            )
            print(
                f"[DATASET-STATS] keep={stats['keep']} eligible={stats['eligible']} "
                f"D54={stats['by_action'].get('D54_OUTER_PULL', {}).get('eligible', 0)} "
                f"D55={stats['by_action'].get('D55_PRESS_SWEEP', {}).get('eligible', 0)} "
                f"D56={stats['by_action'].get('D56_WAIST_LIFT_LAYDOWN', {}).get('eligible', 0)}"
            )
            if keep and not eligible:
                print(f"[VLA-ELIGIBILITY] NOT_ELIGIBLE reason={tr.get('training_eligibility_reason')}")
            return True

    def mark_plan_inaccurate(self, locked: LockedPlan) -> bool:
        """Record an I-locked plan as operator-rejected before Enter.

        The visual observation, selected action and exact frozen plan are preserved
        for debugging, but no robot command is sent and the transition is forced out
        of training.  `human_result=SKIP` is retained only for old-schema lifecycle
        compatibility; `plan_review=PLAN_INACCURATE` keeps it distinct from K/SKIP.
        """
        with self.lock:
            if locked is None or locked.observation_record is None:
                print("[VLA-PLAN] no I-locked observation")
                return False
            if self.current_decision is None:
                print("[VLA-PLAN] no current decision")
                return False
            if self.review_pending():
                print("[VLA-PLAN] previous transition still requires review")
                return False

            # main-5 P fast path: the full observation and decision were already
            # saved at I-lock.  Do not read/rewrite those large JSON files and do
            # not serialize the full plan/diagnostics again.  One compact,
            # authoritative rejection transition is sufficient.
            self.transition_counter += 1
            transition_id = _new_id("transition", self.transition_counter)
            now = _now_text()
            before = locked.observation_record
            tr = {
                "schema_version": SCHEMA_VERSION,
                "cycle_id": self.cycle_id,
                "transition_id": transition_id,
                "decision_id": self.current_decision["id"],
                "opened_at": now,
                "before_observation_id": before["id"],
                "before_observation_path": before["relative_path"],
                "human_state": locked.state_label,
                "human_selected_action": locked.action,
                "executed_action": locked.action,
                "action_variant": str(getattr(locked.plan, "action", locked.action)),
                "plan_review": "PLAN_INACCURATE",
                "plan_review_at": now,
                "plan_inaccurate_stage": "I_LOCK_BEFORE_ENTER",
                "execution_success": False,
                "execution_sent": False,
                "garment_motion_committed": False,
                "execution_detail": "NOT_EXECUTED_PLAN_INACCURATE",
                "human_result": "SKIP",
                "human_result_at": now,
                "collection_decision": "DISCARD",
                "collection_decision_at": now,
                "user_selected_for_collection": False,
                "training_eligible": False,
                "training_eligibility_reason": "plan_inaccurate_before_execution",
                "status": "DISCARDED_PLAN_INACCURATE",
                "completed_at": now,
                "collection_phase": "official" if self.official_collection else "motion_validation",
            }
            rel = str(Path("transitions") / f"{transition_id}.json")
            path = self.episode_dir / rel
            _atomic_json(path, tr)
            record = {"id": transition_id, "relative_path": rel, "fs_path": str(path), "payload": tr}
            self.latest_transition = record
            # V13 fast path: do not rescan every historical transition on each P.
            # The just-written JSON is authoritative and a full rebuild still occurs
            # on the next program start.  Update the in-memory counters immediately.
            self.dataset_stats.setdefault("by_action_keep", {action: 0 for action in PHYSICAL_ACTIONS})
            self.dataset_stats["total_transitions"] = int(self.dataset_stats.get("total_transitions", 0)) + 1
            self.dataset_stats["reviewed"] = int(self.dataset_stats.get("reviewed", 0)) + 1
            self.dataset_stats["plan_inaccurate"] = int(self.dataset_stats.get("plan_inaccurate", 0)) + 1
            self.dataset_stats["discard"] = int(self.dataset_stats.get("discard", 0)) + 1
            stats = copy.deepcopy(self.dataset_stats)
            print(
                f"[VLA-PLAN] PLAN_INACCURATE transition={tr['transition_id']} "
                f"action={tr.get('human_selected_action')} no_execution trainingEligible=False"
            )
            print(
                f"[DATASET-STATS] keep={stats['keep']} eligible={stats['eligible']} "
                f"planBad={stats.get('plan_inaccurate', 0)} skip={stats['skip']}"
            )
            return True

    def mark_mask_inaccurate(self) -> bool:
        with self.lock:
            record = self.current_observation
            if record is None:
                print("[VLA-MASK] no current observation")
                return False
            with open(record["fs_path"], "r", encoding="utf-8") as f:
                payload = json.load(f)
            payload["human_mask_quality"] = "MASK_INACCURATE"
            payload["human_mask_quality_at"] = _now_text()
            payload["training_input_eligible"] = False
            _atomic_json(Path(record["fs_path"]), payload)
            record["payload"] = payload
            self.current_observation = record
            transitions_dir = self.episode_dir / "transitions"
            changed: List[str] = []
            for path in sorted(transitions_dir.glob("*.json")):
                with open(path, "r", encoding="utf-8") as f:
                    tr = json.load(f)
                roles = []
                if tr.get("before_observation_id") == record["id"]:
                    tr["before_mask_quality"] = "MASK_INACCURATE"
                    roles.append("before")
                if tr.get("after_observation_id") == record["id"]:
                    tr["after_mask_quality"] = "MASK_INACCURATE"
                    roles.append("after")
                if not roles:
                    continue
                if tr.get("human_result") not in (None, "", "SKIP"):
                    tr["human_result_before_mask_inaccurate"] = tr.get("human_result")
                tr.update({
                    "human_result": "SKIP",
                    "human_result_at": _now_text(),
                    "collection_decision": "DISCARD",
                    "collection_decision_at": _now_text(),
                    "user_selected_for_collection": False,
                    "training_eligible": False,
                    "status": "DISCARDED_MASK_INACCURATE_SKIP",
                    "mask_inaccurate_roles": roles,
                    "mask_inaccurate_observation_id": record["id"],
                })
                _atomic_json(path, tr)
                changed.append(str(tr.get("transition_id")))
                if self.latest_transition is not None and tr.get("transition_id") == self.latest_transition["id"]:
                    self.latest_transition["payload"] = tr
            self.refresh_dataset_stats()
            print(f"[VLA-MASK] MASK_INACCURATE obs={record['id']} transitions={changed}")
            return True

    def finish_success(self) -> bool:
        return self._terminal("COMPLETE", "SUCCESS", "OPERATOR_FINISH")

    def fail_episode(self, reason: str = "NOT_READY_AFTER_ACTIONS") -> bool:
        return self._terminal("FAILED", "FAILED", reason)

    def mark_incomplete(self) -> bool:
        return self._terminal("INCOMPLETE", "INCOMPLETE", "PROGRAM_EXIT")

    def _terminal(self, status: str, result: str, reason: str) -> bool:
        with self.lock:
            if not self.initialized or self.terminal_status:
                return False
            if self.review_pending() and status == "COMPLETE":
                print("[VLA-EPISODE] G/B/K and Y/N required before SUCCESS")
                return False
            path = self.episode_dir / "episode.json"
            with open(path, "r", encoding="utf-8") as f:
                payload = json.load(f)
            payload.update({
                "status": status,
                "episode_result": result,
                "status_reason": reason,
                "terminal_at": _now_text(),
            })
            _atomic_json(path, payload)
            self.terminal_status = status
            print(f"[VLA-EPISODE] {status} result={result} reason={reason}")
            return True


class WaistROIRecognizer:
    def __init__(self, cv2_module: ModuleType, d56: ModuleType, args: argparse.Namespace):
        self.cv2 = cv2_module
        self.d56 = d56
        self.args = args

    @staticmethod
    def _unit(v: np.ndarray, fallback=(0.0, 1.0)) -> np.ndarray:
        a = np.asarray(v, np.float32).reshape(2)
        n = float(np.linalg.norm(a))
        if not np.isfinite(n) or n <= 1e-6:
            a = np.asarray(fallback, np.float32)
            n = float(np.linalg.norm(a))
        return a / max(n, 1e-6)

    def _reference(self, obs: Any, H: np.ndarray, d56_args: argparse.Namespace) -> Tuple[Optional[Dict[str, Any]], str]:
        body = None
        direction_meta: Dict[str, Any] = {}
        try:
            body, direction_meta, reason = self.d56._d45v6_directed_body_axis(obs, d56_args)
        except Exception as exc:
            reason = f"directed body axis error:{exc!r}"
        pose = getattr(obs, "pose", None)
        if body is None and pose is not None:
            try:
                waist = np.asarray(pose.waist_center, np.float32).reshape(2)
                lower = np.asarray(pose.lower_center, np.float32).reshape(2)
                body = self._unit(lower - waist)
                mask_center = np.asarray(obs.mask.center_board, np.float32).reshape(2)
                if float(np.dot(mask_center - waist, body)) < 0.0:
                    body = -body
                direction_meta = {"source": "POSE_WAIST_TO_LOWER", "body_axis_board": body.tolist()}
                reason = "POSE_FALLBACK_OK"
            except Exception:
                body = None
        if body is None:
            return None, f"WAIST_ROI_BODY_AXIS_UNAVAILABLE:{reason}"
        body = self._unit(body)

        try:
            curve, curve_reason = self.d56._d45v6_extract_waist_curve(obs, H, body, d56_args)
        except Exception as exc:
            curve, curve_reason = None, f"curve error:{exc!r}"
        if curve is not None:
            return {
                "points_board": np.asarray(curve["points_board"], np.float32),
                "points_px": [p for p in curve.get("points_px", []) if p is not None],
                "body_axis": body,
                "source": "POSE_DIRECTED_MASK_WAIST_CURVE",
                "direction": direction_meta,
                "curve": _json_safe(curve),
            }, "OK"

        if pose is None:
            return None, f"WAIST_ROI_MASK_CURVE_FAILED:{curve_reason}"
        try:
            left = np.asarray(pose.waist_left, np.float32).reshape(2)
            right = np.asarray(pose.waist_right, np.float32).reshape(2)
            width = float(np.linalg.norm(right - left))
            if width < float(self.args.d56_roi_pose_min_width_mm):
                raise ValueError(f"pose waist width {width:.1f}mm")
            points = np.linspace(left, right, 32).astype(np.float32)
            points_px = []
            for p in points:
                q = self.d56.board_to_pixel(H, float(p[0]), float(p[1]))
                if q is not None:
                    points_px.append((float(q[0]), float(q[1])))
            return {
                "points_board": points,
                "points_px": points_px,
                "body_axis": body,
                "source": "POSE_WAIST_SEGMENT_MASK_RESTRICTED",
                "direction": direction_meta,
                "curve_failure": curve_reason,
            }, "OK"
        except Exception as exc:
            return None, f"WAIST_ROI_REFERENCE_FAILED:{curve_reason}|{exc!r}"

    def _point_metrics(self, point: np.ndarray, reference: Dict[str, Any]) -> Dict[str, float]:
        curve = np.asarray(reference["points_board"], np.float32).reshape(-1, 2)
        p = np.asarray(point, np.float32).reshape(2)
        i = int(np.argmin(np.linalg.norm(curve - p.reshape(1, 2), axis=1)))
        delta = p - curve[i]
        body = self._unit(np.asarray(reference["body_axis"], np.float32))
        signed = float(np.dot(delta, body))
        lateral = float(abs(np.dot(delta, np.asarray([-body[1], body[0]], np.float32))))
        return {"distance_mm": float(np.linalg.norm(delta)), "signed_body_mm": signed, "lateral_mm": lateral}

    def _filter_dense(self, report: Dict[str, Any], reference: Dict[str, Any], H: np.ndarray) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        original = report.get("selected") if isinstance(report.get("selected"), dict) else None
        accepted: List[Dict[str, Any]] = []
        rows: List[Dict[str, Any]] = []
        depth = float(self.args.d56_roi_depth_mm)
        outward_slack = float(self.args.d56_roi_outward_slack_mm)
        for candidate in list(report.get("bundle_candidates", []) or []):
            path_px = np.asarray(candidate.get("path_px", []), np.float32).reshape(-1, 2)
            metrics = []
            for px in path_px:
                try:
                    board = np.asarray(self.d56.pixel_to_board(H, float(px[0]), float(px[1])), np.float32)
                    metrics.append(self._point_metrics(board, reference))
                except Exception:
                    pass
            signed = np.asarray([m["signed_body_mm"] for m in metrics], np.float32)
            distance = np.asarray([m["distance_mm"] for m in metrics], np.float32)
            inside = bool(
                len(metrics) >= 2
                and float(np.mean((signed >= -outward_slack) & (signed <= depth))) >= float(self.args.d56_roi_dense_inside_ratio)
                and float(np.median(distance)) <= depth
            )
            row = {
                "line_count": int(candidate.get("line_count", 0)),
                "span_mm": float(candidate.get("ribbon_span_mm", 0.0)),
                "score": float(candidate.get("score", 0.0)),
                "median_curve_distance_mm": float(np.median(distance)) if len(distance) else None,
                "signed_body_median_mm": float(np.median(signed)) if len(signed) else None,
                "inside_roi": inside,
            }
            rows.append(row)
            if inside:
                c = copy.deepcopy(candidate)
                c["d56_roi_verified"] = True
                c["d56_roi_metrics"] = row
                accepted.append(c)
        accepted.sort(
            key=lambda c: (
                float(c.get("d56v18_rank_score", c.get("score", 0.0))),
                float(c.get("ribbon_span_mm", 0.0)),
                int(c.get("line_count", 0)),
            ), reverse=True,
        )
        new_report = copy.deepcopy(report)
        new_report["selected_before_roi"] = copy.deepcopy(original)
        new_report["selected"] = copy.deepcopy(accepted[0]) if accepted else None
        new_report["reason"] = "DENSE_INSIDE_WAIST_ROI" if accepted else "DENSE_OUTSIDE_OR_MISSING_WAIST_ROI"
        outside_fp = bool(original is not None and not accepted)
        return new_report, {
            "dense_candidate_rows": rows,
            "dense_inside_count": len(accepted),
            "dense_selected_outside_roi": outside_fp,
        }

    def _weak_lines(self, frame: np.ndarray, obs: Any, H: np.ndarray, reference: Dict[str, Any]) -> Dict[str, Any]:
        mask = (np.asarray(obs.mask.mask_u8, np.uint8) > 0).astype(np.uint8) * 255
        if mask.shape[:2] != frame.shape[:2]:
            mask = self.cv2.resize(mask, (frame.shape[1], frame.shape[0]), interpolation=self.cv2.INTER_NEAREST)
        lab = self.cv2.cvtColor(frame, self.cv2.COLOR_BGR2LAB)
        gray = lab[:, :, 0]
        clahe = self.cv2.createCLAHE(clipLimit=float(self.args.d56_weak_clahe_clip), tileGridSize=(8, 8)).apply(gray)
        low = self.cv2.GaussianBlur(clahe, (0, 0), float(self.args.d56_weak_highpass_sigma))
        high = self.cv2.absdiff(clahe, low).astype(np.float32)
        detected = self.cv2.createLineSegmentDetector(self.cv2.LSD_REFINE_STD).detect(clahe)[0]
        accepted: List[Dict[str, Any]] = []
        if detected is None:
            return {"lines": [], "reason": "NO_LSD_LINES"}
        h, w = mask.shape[:2]
        body = self._unit(np.asarray(reference["body_axis"], np.float32))
        body_angle = float(np.degrees(math.atan2(float(body[1]), float(body[0]))) % 180.0)
        for raw in detected.reshape(-1, 4):
            p0 = np.asarray(raw[:2], np.float32)
            p1 = np.asarray(raw[2:], np.float32)
            if float(np.linalg.norm(p1 - p0)) < 4.0:
                continue
            try:
                b0 = np.asarray(self.d56.pixel_to_board(H, float(p0[0]), float(p0[1])), np.float32)
                b1 = np.asarray(self.d56.pixel_to_board(H, float(p1[0]), float(p1[1])), np.float32)
            except Exception:
                continue
            vec = b1 - b0
            length_mm = float(np.linalg.norm(vec))
            if not (float(self.args.d56_weak_line_min_mm) <= length_mm <= float(self.args.d56_weak_line_max_mm)):
                continue
            center_px = 0.5 * (p0 + p1)
            center_board = 0.5 * (b0 + b1)
            roi = self._point_metrics(center_board, reference)
            if not (-float(self.args.d56_roi_outward_slack_mm) <= roi["signed_body_mm"] <= float(self.args.d56_roi_depth_mm)):
                continue
            samples = []
            mask_hits = 0
            for t in np.linspace(0.0, 1.0, 9):
                p = (1.0 - float(t)) * p0 + float(t) * p1
                x = int(np.clip(round(float(p[0])), 0, w - 1))
                y = int(np.clip(round(float(p[1])), 0, h - 1))
                mask_hits += int(mask[y, x] > 0)
                samples.append(float(high[y, x]))
            support = float(mask_hits / 9.0)
            contrast = float(np.mean(samples)) if samples else 0.0
            if support < float(self.args.d56_weak_mask_support_min) or contrast < float(self.args.d56_weak_contrast_min):
                continue
            angle = float(np.degrees(math.atan2(float(vec[1]), float(vec[0]))) % 180.0)
            angle_diff = abs(float((angle - body_angle + 90.0) % 180.0 - 90.0))
            if angle_diff > float(self.args.d56_weak_body_angle_max_deg):
                continue
            accepted.append({
                "p0_px": p0.tolist(), "p1_px": p1.tolist(), "center_px": center_px.tolist(),
                "center_board": center_board.tolist(), "length_mm": length_mm,
                "length_px": float(np.linalg.norm(p1 - p0)), "angle_deg": angle,
                "mask_support": support, "contrast": contrast, "roi": roi,
            })
        accepted.sort(key=lambda x: (float(x["contrast"]), float(x["mask_support"])), reverse=True)
        accepted = accepted[:int(self.args.d56_weak_max_lines)]
        left_limit = float(self.args.d56_weak_arm2_coverage_x_max_mm)
        right_limit = float(self.args.d56_weak_arm1_coverage_x_min_mm)
        left = [x for x in accepted if float(x["center_board"][0]) <= left_limit]
        right = [x for x in accepted if float(x["center_board"][0]) >= right_limit]
        min_side = int(self.args.d56_weak_min_lines_per_arm)
        if len(left) < min_side or len(right) < min_side:
            return {
                "lines": accepted, "left_count": len(left), "right_count": len(right),
                "reason": f"WEAK_SIDE_COVERAGE_{len(left)}/{len(right)}_LT_{min_side}",
            }
        lateral = np.asarray([-body[1], body[0]], np.float32)
        if float(lateral[0]) < 0.0:
            lateral = -lateral
        selected_ids = list(range(len(accepted)))
        selected_ids.sort(key=lambda i: float(np.dot(np.asarray(accepted[i]["center_board"], np.float32), lateral)))
        centers = np.asarray([accepted[i]["center_board"] for i in selected_ids], np.float32)
        span = float(np.sum(np.linalg.norm(np.diff(centers, axis=0), axis=1))) if len(centers) >= 2 else 0.0
        if span < float(self.args.d56_weak_min_span_mm):
            return {"lines": accepted, "left_count": len(left), "right_count": len(right), "reason": f"WEAK_SPAN_{span:.1f}MM"}
        lengths = np.asarray([accepted[i]["length_mm"] for i in selected_ids], np.float32)
        path_px = [accepted[i]["center_px"] for i in selected_ids]
        selected = {
            "line_indices": selected_ids,
            "path_line_indices": selected_ids,
            "path_px": path_px,
            "path_center_board": np.mean(centers, axis=0).tolist(),
            "line_count": len(selected_ids),
            "dense_core_line_count": len(selected_ids),
            "median_length_mm": float(np.median(lengths)),
            "length_cv": float(np.std(lengths) / max(1e-6, float(np.mean(lengths)))),
            "orientation_consistency": 1.0,
            "ribbon_span_mm": span,
            "bundle_mst_span_mm": span,
            "body_support": 1.0,
            "body_sign": 1,
            "score": float(np.clip(0.58 + 0.02 * min(10, len(selected_ids)), 0.0, 0.88)),
            "d56v18_rank_score": float(np.clip(0.60 + 0.02 * min(10, len(selected_ids)), 0.0, 0.90)),
            "d56v18_temporal_score": 0.0,
            "d56v22_sequence_score": 0.0,
            "d56_weak_disconnected_actual_lines": True,
            "d56_roi_verified": True,
        }
        return {
            "lines": accepted, "left_count": len(left), "right_count": len(right),
            "selected": selected, "reason": "ROI_INTERNAL_WEAK_WRINKLE_PAIR",
        }

    def plan(self, frame: np.ndarray, obs: Any, H: np.ndarray, config: Dict[str, Any],
             cfg: Any, d56_args: argparse.Namespace) -> Tuple[Any, Dict[str, Any]]:
        reference, reference_reason = self._reference(obs, H, d56_args)
        diagnostics: Dict[str, Any] = {
            "recognition_version": D56_RECOGNITION_VERSION,
            "roi_reference_reason": reference_reason,
            "roi_reference_source": None if reference is None else reference.get("source"),
            "waist_failure_flags": [],
        }
        if reference is None:
            diagnostics["waist_failure_flags"] = ["WAIST_MISSED"]
            plan = self.d56.D31DualGraspPlan(
                False, "D56 WAIST_MISSED: Pose+mask waist ROI unavailable",
                metrics={"d42_plan_mode": "D56_ROI_NO_PLAN", "planner_failure": "WAIST_MISSED"},
            )
            return plan, diagnostics

        report = self.d56._d56v7_build_waist_observer(frame, obs, H, d56_args)
        report, dense_meta = self._filter_dense(report, reference, H)
        diagnostics.update(dense_meta)
        selected_source = "DENSE"
        weak_meta: Dict[str, Any] = {}
        if not isinstance(report.get("selected"), dict):
            weak_meta = self._weak_lines(frame, obs, H, reference)
            diagnostics["weak_wrinkle"] = _json_safe(weak_meta)
            if isinstance(weak_meta.get("selected"), dict):
                report["short_lines"] = copy.deepcopy(weak_meta["lines"])
                report["selected"] = copy.deepcopy(weak_meta["selected"])
                report["reason"] = "ROI_INTERNAL_WEAK_WRINKLE_PAIR"
                selected_source = "WEAK_WRINKLE"

        try:
            obs.d56v7_waist_observer = report
            obs.bottom_vla_waist_roi = reference
        except Exception:
            pass

        if not isinstance(report.get("selected"), dict):
            flags = []
            if dense_meta.get("dense_selected_outside_roi"):
                flags.append("WAIST_FALSE_POSITIVE")
            flags.append("WAIST_MISSED")
            diagnostics["waist_failure_flags"] = list(dict.fromkeys(flags))
            primary = flags[0]
            plan = self.d56.D31DualGraspPlan(
                False,
                f"D56 {primary}: ROI-contained actual waist wrinkles unavailable; torso wrinkles are not substituted",
                metrics={
                    "d42_plan_mode": "D56_ROI_NO_PLAN",
                    "planner_failure": primary,
                    "waist_failure_flags": diagnostics["waist_failure_flags"],
                    "mask_curve_waist_suppressed": True,
                },
            )
            return plan, diagnostics

        plan = self.d56._d56v15_build_waist_ribbon_plan(obs, H, config, cfg, d56_args)
        plan.metrics = dict(getattr(plan, "metrics", {}) or {})
        plan.metrics.update({
            "d56_vla_recognition_source": selected_source,
            "d56_vla_roi_reference_source": reference.get("source"),
            "d56_vla_no_d45_fallback": True,
            "mask_curve_waist_suppressed": True,
        })
        diagnostics["selected_source"] = selected_source
        diagnostics["waist_failure_flags"] = [] if bool(plan.ok) else ["WAIST_MISSED"]
        if not plan.ok:
            plan.metrics["planner_failure"] = "WAIST_MISSED"
        return plan, diagnostics

    def draw(self, image: np.ndarray, reference: Optional[Dict[str, Any]], diagnostics: Dict[str, Any]) -> np.ndarray:
        out = image
        if isinstance(reference, dict):
            pts = [tuple(map(int, map(round, p))) for p in reference.get("points_px", [])]
            if len(pts) >= 2:
                self.cv2.polylines(out, [np.asarray(pts, np.int32).reshape(-1, 1, 2)], False, (255, 255, 0), 3)
                self.cv2.putText(out, "D56 WAIST ROI GUIDE (NOT A GRASP LINE)", (20, 118),
                                 self.cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 2)
        weak = diagnostics.get("weak_wrinkle", {}) if isinstance(diagnostics, dict) else {}
        for line in list(weak.get("lines", []) or [])[:160]:
            p0 = tuple(map(int, map(round, line["p0_px"])))
            p1 = tuple(map(int, map(round, line["p1_px"])))
            self.cv2.line(out, p0, p1, (255, 0, 255), 2, self.cv2.LINE_AA)
        return out


class BottomManualVLAApp:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        try:
            import cv2
        except Exception as exc:
            raise RuntimeError(f"OpenCV is required on the Jetson runtime: {exc}") from exc
        self.cv2 = cv2
        self.base = Path(__file__).resolve().parent
        self.d56_path = _resolve_source(
            args.d56_source, (f"{DEFAULT_SOURCE_DIR}/56-45.py", "56-45.py"), "D56-45"
        )
        self.d54_path = _resolve_source(
            args.d54_source,
            (f"{DEFAULT_SOURCE_DIR}/54-3.py", "54-3.py"),
            "D54-3",
        )
        self.d55_path = _resolve_source(
            args.d55_source,
            (f"{DEFAULT_SOURCE_DIR}/55-5.py", "55-5.py"),
            "D55-2",
        )
        self.d58_path = _resolve_source(
            args.d58_source,
            (f"{DEFAULT_SOURCE_DIR}/58-2.py", "58-2.py"),
            "D58-2",
        )
        self.d50_path = _resolve_source(
            args.d50_source,
            (f"{DEFAULT_SOURCE_DIR}/50-1.py", "50-1.py"),
            "D50-BASKET-DROP",
        )
        self.d56 = _load_module("bottom_vla_d56", self.d56_path)
        self.d54 = _load_module("bottom_vla_d54", self.d54_path)
        self.d55 = _load_module("bottom_vla_d55", self.d55_path)
        self.d58 = _load_module("bottom_vla_d58", self.d58_path)
        self.source_sha256 = {
            "D50_BASKET_SWING_LAYDOWN": _sha256_file(self.d50_path),
            "D54_OUTER_PULL": _sha256_file(self.d54_path),
            "D55_PRESS_SWEEP": _sha256_file(self.d55_path),
            "D56_WAIST_LIFT_LAYDOWN": _sha256_file(self.d56_path),
            "D58_CIRC_POSITION": _sha256_file(self.d58_path),
        }
        for action_name, digest in self.source_sha256.items():
            print(f"[SOURCE-READONLY] {action_name} sha256={digest}")
        # Never convert a broken perception import into a misleading
        # WAIST_MISSED/NO_SAFE_PLAN observation.  The merged collector must use
        # the exact E49 mask and E62 pose dependencies used by the source files.
        _require_perception_imports(self.d56, "D56-45")
        _require_perception_imports(self.d54, "D54-V12")
        _require_perception_imports(self.d55, "D55-2")

        # D56-45 exposes physical-auto (not physical-manual).  We still turn
        # autonomous looping off below; the wrapper owns every Enter latch.
        source_preset = "hover-check" if args.mode == "hover" else "physical-manual"
        d56_preset = "hover-check" if args.mode == "hover" else "physical-auto"
        self.d56_args = _module_default_args(self.d56, preset=d56_preset)
        self.d54_args = _module_default_args(self.d54, preset=source_preset)
        self.d55_args = _module_default_args(self.d55, preset=source_preset)
        self.d58_args = self.d58.parser58().parse_args([])
        for namespace in (self.d56_args, self.d54_args, self.d55_args, self.d58_args):
            _copy_shared_args(namespace, args, args.mode)
        # The original mains add the inherited D25 evaluator defaults before
        # building perception/plans.  Reproduce that initialization without
        # entering any of the legacy main loops.
        if hasattr(self.d56, "_d26_prepare_d25_args"):
            self.d56_args = self.d56._d26_prepare_d25_args(self.d56_args)
        if hasattr(self.d54, "_d26_prepare_d25_args"):
            self.d54_args = self.d54._d26_prepare_d25_args(self.d54_args)
        if hasattr(self.d55, "_d26_prepare_d25_args"):
            self.d55_args = self.d55._d26_prepare_d25_args(self.d55_args)
        self._configure_action_args()
        self.d58_args.config = self.args.config
        self.d58_args.hfile = self.args.hfile
        self.d58_args.seg_model = self.args.seg_model
        self.d58_args.pose_model = self.args.pose_model
        self.d58_args.camera_calibration = self.args.camera_calibration
        self.d58_args.send = self.args.mode == "physical"
        self.d58_args.dry_run = self.args.mode == "dry-run"
        self.d58_args.hover_only = self.args.mode == "hover"

        with open(_require_file(args.config, "folding-board config"), "r", encoding="utf-8") as f:
            self.config = json.load(f)
        if not isinstance(self.config, dict) or not isinstance(self.config.get("dual_roarm"), dict):
            raise RuntimeError("authoritative config lacks dual_roarm")
        dictionary_name = self.config.get("aruco", {}).get("dictionary", "DICT_4X4_50")
        self.aruco_detector = self.d54.make_aruco_detector(
            self.d54.get_dictionary(dictionary_name)
        )
        self.H_corrected, self.H_raw, self.h_bundle = self._load_homography(args.hfile)
        self.cfg56 = self.d56.make_safety_config_from_args(self.d56_args, self.config)
        self.cfg54 = self.d54.make_safety_config_from_args(self.d54_args, self.config)
        self.cfg55 = self.d55.make_safety_config_from_args(self.d55_args, self.config)
        self._sync_module_runtime()

        self.cap_lock = threading.RLock()
        self.state_lock = threading.RLock()
        self.event_lock = threading.Lock()
        self.events: List[str] = []
        self.motion_busy = False
        self.inference_serial = 0
        self.d56_prespread_attempt_count = 0
        self.status = "EMPTY BOARD REQUIRED: PRESS E ONCE"
        self.selected_action: Optional[str] = None
        self.locked: Optional[LockedPlan] = None
        self.display_image: Optional[np.ndarray] = None
        self.latest_live_raw: Optional[np.ndarray] = None
        self.latest_live_corrected: Optional[np.ndarray] = None
        # The OpenCV main loop is the only continuous camera reader.
        # Inference workers consume a copied frozen RAW frame and perform
        # undistortion exactly once for each decision.
        self.live_frame_lock = threading.RLock()
        self.latest_live_monotonic = 0.0
        self.empty_baseline_ready = False
        self.closed = False
        self.arms: Dict[str, Any] = {}
        self.worker: Optional[threading.Thread] = None
        # main-26: D56 and D58 I-perception run off the OpenCV UI thread.
        # D58 also remembers the most recent successful TTA transform and tries
        # that single transform first on the next I; the original full TTA is
        # retained as an automatic fallback.
        self.inference_busy = False
        self.inference_action: Optional[str] = None
        self.inference_worker: Optional[threading.Thread] = None
        self.d58_pose_tta_hint: Optional[Tuple[float, str]] = None
        # main-27: prevent a stale/accidental GUI or terminal ABORT event from
        # immediately terminating a freshly opened dataset episode.  ABORT must
        # be explicitly confirmed by a second ABORT event after 0.6 s and within
        # a 4 s confirmation window.
        self.abort_armed_at = 0.0
        self.abort_confirm_until = 0.0

        self.cap = self._open_camera()
        self.camera_controls_meta = apply_camera_controls(
            args.camera_controls_json, args.camera_device, args.camera_controls_strict
        ) if args.camera_controls_enable else {"enabled": False, "profile_path": args.camera_controls_json}
        for _ in range(max(0, int(args.camera_controls_stabilization_frames))):
            with self.cap_lock:
                self.cap.read()
        ok, probe = self._read_raw(flush=0)
        if not ok or probe is None:
            raise RuntimeError("camera opened but initial frame read failed")
        actual_size = (int(probe.shape[1]), int(probe.shape[0]))
        if actual_size != (int(args.width), int(args.height)):
            raise RuntimeError(f"camera size {actual_size} != requested {(args.width, args.height)}")
        if getattr(self.d54, "CameraUndistorter", None) is None:
            raise RuntimeError(
                f"D54/D55 original camera_undistort.py unavailable: "
                f"{getattr(self.d54, '_CAMERA_UNDISTORT_IMPORT_ERROR', None)!r}"
            )
        self.undistorter = self.d54.CameraUndistorter(
            self.args.camera_calibration,
            alpha=float(getattr(self.d54_args, "camera_undistort_alpha", 0.0)),
            strict_size=bool(getattr(self.d54_args, "camera_undistort_strict_size", True)),
        )
        self.undistorter.prepare(actual_size)
        self.corrected_camera_geometry = self.undistorter.info().to_metadata()
        self.raw_camera_geometry = {
            "camera_model": "ELP_OV2710",
            "undistort_enabled": False,
            "implementation": "d56-45_raw_cap_read",
            "input_size": list(actual_size),
            "output_size": list(actual_size),
        }
        self.camera_geometry_meta = {
            "D54_OUTER_PULL": copy.deepcopy(self.corrected_camera_geometry),
            "D55_PRESS_SWEEP": copy.deepcopy(self.corrected_camera_geometry),
            "D56_WAIST_LIFT_LAYDOWN": copy.deepcopy(self.raw_camera_geometry),
            "D58_CIRC_POSITION": copy.deepcopy(self.corrected_camera_geometry),
        }
        print("[ELP-UNDISTORT-D54-D55] " + self.undistorter.status_line())
        print("[CAMERA-GEOMETRY-D56] RAW cap.read() + raw_H")
        self._ensure_startup_homography()
        self._validate_h_geometry()

        self.seg_model, self.pose_model = self.d56.load_models(self.d56_args)
        # main-25: 56-45 already provides this exact startup warmup specifically
        # so the first I key does not pay TensorRT engine/context creation cost.
        # Outputs are discarded; this performs NO robot motion and changes no
        # perception/planner thresholds.
        if hasattr(self.d56, "_d56v39_warm_inference_models_once"):
            with self.cap_lock:
                self.d56._d56v39_warm_inference_models_once(
                    self.cap, self.seg_model, self.pose_model, self.d56_args
                )
        print("[MAIN32-FAST-RUNTIME] TOP-VLA pattern enabled: RAW live loop + snapshot-only correction + unified I worker")
        print("[MAIN32-SERIAL] main-27 persistent serial/startup path preserved; no main-30/31 boot-sentinel gate")
        print("[MAIN33-D56] D56-45 perception + FIX59 grasp timing + real D50 FIX55/FIX56 2-D master replay")
        print("[MAIN33-D58] strongest-safe correction priority; target request 120~180mm, no 30mm-first scoring")
        self.arms = self._connect_arms_no_motion()
        self._set_authoritative_board_geometry()
        self.recorder = DatasetRecorder(
            args.dataset_root, self.cv2, self._runtime_metadata(actual_size),
            self.d56.board_bounds_from_config(self.config), args.board_image_size,
            official_collection=args.collection_phase == "official",
        )
        self.recorder.initialize()
        atexit.register(self._atexit)
        if not self._ensure_camera_clear("STARTUP", allow_move=True):
            self.status = "STARTUP STANDBY FAILED: CHECK ARMS BEFORE E"
            print("[STARTUP] automatic standby move/verification failed; check both arms before E")
        else:
            self.status = "EMPTY BOARD READY: PRESS E ONCE"
            print("[STARTUP] ARM1/ARM2 automatic standby verified")
        print("[E-REQUIRED] I/ENTER are blocked until this session captures an empty-board baseline")

    def _configure_action_args(self) -> None:
        self.d56_args.config = self.args.config
        self.d56_args.hfile = self.args.hfile
        self.d56_args.seg_model = self.args.seg_model
        self.d56_args.pose_model = self.args.pose_model
        self.d56_args.d34_empty_board_path = self.args.empty_board_raw_path
        self.d56_args.cam_auto_adjust = False
        self.d56_args.d38_live_perception = False
        self.d56_args.d17_auto_loop = False
        # main-24: preserve D56-45's own perception/grasp/motion defaults.
        # Do not overwrite its Pose+Mask waist estimate, 35 mm whole-circle-safe
        # grasp search, asymmetric ARM1/ARM2 curve insets, adaptive body-inset
        # retries, +150 mm initial-lift policy, or second-lift=OFF policy.
        # Only collector/runtime controls below are forced.
        self.d56_args.d34_crumpled_mask = True
        self.d56_args.d34_mode = "auto"
        self.d56_args.d22_hybrid_wrinkle_policy = True
        self.d56_args.d23v2_allow_legacy_fallback = False
        self.d56_args.d21_geometry_grip_pull = False
        if bool(getattr(self.d56_args, "d26v4_full_tta", True)):
            self.d56_args.d23_pose_tta_fast_first = False
        self.d56_args.d26_keep_gripper_closed = True
        self.d56_args.d23v2_press_gripper = float(self.d56_args.d26_press_gripper)
        self.d56_args.d22_press_sweep_min_mm = float(self.d56_args.d26v3_press_min_mm)
        self.d56_args.d22_press_sweep_max_mm = float(self.d56_args.d26v3_press_max_mm)
        self.d56_args.d26_shape_press_min_mm = float(self.d56_args.d26v3_shape_press_min_mm)
        self.d56_args.d26_shape_press_max_mm = float(self.d56_args.d26v3_shape_press_max_mm)
        self.d56_args.pose_tta_angles_list = self.d56.parse_pose_tta_angles(self.d56_args.angles)
        self.d56_args.pose_tta_flip_modes_list = self.d56.parse_pose_tta_flip_modes(self.d56_args.flip_modes)
        print(
            "[D56-45-NATIVE] Pose+Mask waist hardening + 35mm whole-circle safe grasp "
            "+ native D56-45 lift/tension/swing defaults"
        )

        self.d54_args.camera_calibration = self.args.camera_calibration
        self.d54_args.d34_empty_board_path = self.args.empty_board_corrected_path
        self.d54_args.d54_pull_extra_mm = float(self.args.d54_pull_extra_mm)
        # V14 D54 distance-only change: preserve direction/planner/executor,
        # but request a materially longer outward pull.
        self.d54_args.d54_pull_gain = float(D54_PULL_GAIN)
        self.d54_args.d54_pull_min_mm = float(D54_PULL_MIN_MM)
        self.d54_args.d54_pull_max_mm = float(D54_PULL_MAX_MM)
        self.d54_args.d54_pull_hard_max_mm = float(D54_PULL_HARD_MAX_MM)
        # V10.2: keep the original D54 planner/direction, but request an additional
        # 15 mm physical inward correction from the I-locked grip point.  The
        # original D54 safety fitter still owns workspace/board clipping.
        self.d54_extra_grip_inset_mm = float(D54_D55_EXTRA_GRIP_INSET_MM)
        self.d54_args.d51_diagonal_pull_enabled = True
        self.d54_args.d51_diag_second_axis_if_weak = False
        self.d54_args.d47_global_prespread = False
        self.d54_args.d50_large_swing_enabled = False
        self.d54_args.d50_motion_test = False
        self.d54_args.d50_require_waist_plan = False
        self.d54_args.d51_mask_core_recovery = False
        self.d54_args.d51_snapshot_preserve_on_transient_failure = False
        self.d54_args.d17_auto_loop = False
        # main-24: restore 54-3's native D34/E49 mask refinement.  This uses
        # the session E baseline to reject board/background leakage and avoids
        # planning from the jagged raw/fallback outer envelope.
        self.d54_args.d34_crumpled_mask = True
        self.d54_args.d34_mode = "auto"
        self.d54_args.d22_hybrid_wrinkle_policy = True
        self.d54_args.d23v2_allow_legacy_fallback = False
        self.d54_args.d21_geometry_grip_pull = False
        if bool(getattr(self.d54_args, "d26v4_full_tta", True)):
            self.d54_args.d23_pose_tta_fast_first = False
        self.d54_args.d26_keep_gripper_closed = True
        self.d54_args.d23v2_press_gripper = float(self.d54_args.d26_press_gripper)
        self.d54_args.d22_press_sweep_min_mm = float(self.d54_args.d26v3_press_min_mm)
        self.d54_args.d22_press_sweep_max_mm = float(self.d54_args.d26v3_press_max_mm)
        self.d54_args.d26_shape_press_min_mm = float(self.d54_args.d26v3_shape_press_min_mm)
        self.d54_args.d26_shape_press_max_mm = float(self.d54_args.d26v3_shape_press_max_mm)
        self.d54_args.pose_tta_angles_list = self.d54.parse_pose_tta_angles(self.d54_args.angles)
        self.d54_args.pose_tta_flip_modes_list = self.d54.parse_pose_tta_flip_modes(self.d54_args.flip_modes)

        self.d55_args.camera_calibration = self.args.camera_calibration
        self.d55_args.d34_empty_board_path = self.args.empty_board_corrected_path
        self.d55_args.d47_global_prespread = False
        self.d55_args.d50_large_swing_enabled = False
        self.d55_args.d50_motion_test = False
        self.d55_args.d50_require_waist_plan = False
        self.d55_args.d51_mask_core_recovery = False
        self.d55_args.d51_diagonal_pull_enabled = False
        self.d55_args.d51_diag_second_axis_if_weak = False
        self.d55_args.d51_snapshot_preserve_on_transient_failure = False

        self.d55_args.d17_auto_loop = False
        self.d55_args.d38_live_perception = False
        self.d55_args.d21_geometry_grip_pull = False
        self.d55_args.d23v2_allow_legacy_fallback = False
        self.d55_args.d23v2_shape_asymmetry_press_first = False
        self.d55_args.d55_allow_shape_correction = False
        self.d55_args.d26v3_prefer_dual_press = True
        self.d55_args.d26v3_parallel_outward_sweep = True
        self.d55_args.d55v5_use_assist_arm = True
        # 55-1: keep both contact centers materially inside the garment and
        # validate a gripper-sized neighborhood, not just one mask pixel.
        self.d55_args.d55v8_contact_outer_min_mm = 0.0
        self.d55_args.d55v8_contact_outer_max_mm = 15.0
        self.d55_args.d55v8_contact_outer_fraction = 0.15
        self.d55_args.d55v11_xy_guard_radius_mm = 10.0
        self.d55_args.d55v11_xy_guard_min_ratio = 0.80
        self.d55_args.d55v11_xy_guard_max_inward_mm = 60.0
        self.d55_args.d55v14_mask_core_inset_mm = 20.0
        self.d55_args.d27_track_terminal_lock = False
        self.d55_args.d26v7_consensus_finish = False
        self.d55_args.d21v4_good_enough_finish = False
        self.d55_args.d55_candidate_consensus = True
        self.d55_args.d55_candidate_consensus_frames = int(self.args.d55_consensus_frames)
        self.d55_args.d55_candidate_consensus_required = min(2, int(self.args.d55_consensus_frames))
        self.d55_args.d55_candidate_consensus_interval_s = float(self.args.d55_consensus_interval_s)
        # V24: move both D55 press-contact points 25 mm farther inward relative
        # to the V13 baseline by increasing the existing contour edge inset.
        # ARM2's V13-specific extra inset remains intact, so both arms receive
        # exactly +25 mm compared with their previous V13 contact policy.
        self.d55_args.d55v8_contact_edge_inset_mm = (
            float(self.d55_args.d55v8_contact_edge_inset_mm)
            + float(D55_EXTRA_GRIP_INSET_MM)
        )
        self.d55_args.d34_crumpled_mask = False
        self.d55_args.d34_mode = "off"
        self.d55_args.d22_hybrid_wrinkle_policy = True
        if bool(getattr(self.d55_args, "d26v4_full_tta", True)):
            self.d55_args.d23_pose_tta_fast_first = False
        self.d55_args.d26_keep_gripper_closed = True
        self.d55_args.d23v2_press_gripper = float(self.d55_args.d26_press_gripper)
        self.d55_args.d22_press_sweep_min_mm = float(self.d55_args.d26v3_press_min_mm)
        self.d55_args.d22_press_sweep_max_mm = float(self.d55_args.d26v3_press_max_mm)
        self.d55_args.d26_shape_press_min_mm = float(self.d55_args.d26v3_shape_press_min_mm)
        self.d55_args.d26_shape_press_max_mm = float(self.d55_args.d26v3_shape_press_max_mm)
        self.d55_args.pose_tta_angles_list = self.d55.parse_pose_tta_angles(self.d55_args.angles)
        self.d55_args.pose_tta_flip_modes_list = self.d55.parse_pose_tta_flip_modes(self.d55_args.flip_modes)

        # main-28 D58: retain the original circumcenter/grasp/low-Z drag planner,
        # but remove the timid 85mm cap. Re-perception after every action remains
        # unchanged, so a large correction is still split across closed-loop cycles.
        self.d58_args.d58_max_move_mm = float(self.args.main33_d58_max_move_mm)
        print(
            f"[D58-MAIN28-MOVE] original planner/executor preserved; "
            f"maxMove={float(self.d58_args.d58_max_move_mm):.1f}mm strongest-safe priority"
        )
        print(
            f"[VLA-GRIP-INSET] D54=+{self.d54_extra_grip_inset_mm:.1f}mm physical inward; "
            f"D55=+{D55_EXTRA_GRIP_INSET_MM:.1f}mm edge inset "
            f"(base={float(self.d55_args.d55v8_contact_edge_inset_mm):.1f}mm)"
        )
        print(
            f"[D54-PULL-V14] gain={self.d54_args.d54_pull_gain:.2f} "
            f"range={self.d54_args.d54_pull_min_mm:.0f}~{self.d54_args.d54_pull_max_mm:.0f}mm "
            f"hardMax={self.d54_args.d54_pull_hard_max_mm:.0f}mm"
        )

        # These post-parse assignments are performed by each original main().
        # They were missing from V6 and can change the E49 board ROI/mask.
        for namespace in (self.d56_args, self.d54_args, self.d55_args):
            if bool(getattr(namespace, "d35_strict_aruco_roi", True)):
                namespace.d32_board_roi = True
                namespace.d26v4_board_clip_expand_px = 0

    def _connect_arms_no_motion(self) -> Dict[str, Any]:
        """Open the two serial ports without issuing gripper or pose commands."""
        if not bool(getattr(self.d56_args, "send", False)):
            return {}
        arms: Dict[str, Any] = {}
        try:
            arms["arm1"] = self.d56.RoArmSerial(
                self.d56_args.arm1_port, baudrate=115200, label="ARM1"
            )
            arms["arm2"] = self.d56.RoArmSerial(
                self.d56_args.arm2_port, baudrate=115200, label="ARM2"
            )
        except Exception:
            for arm in arms.values():
                try:
                    arm.close()
                except Exception:
                    pass
            raise
        print("[ARMS] ARM1/ARM2 serial connected; no startup gripper or pose command sent")
        return arms

    def _sync_module_runtime(self) -> None:
        marker_map = copy.deepcopy(self.config.get("aruco", {}).get("marker_board_mm", {}))
        ids = [int(v) for v in self.config.get("aruco", {}).get("required_ids", [0, 1, 2, 3])]
        for module, namespace in ((self.d56, self.d56_args), (self.d54, self.d54_args), (self.d55, self.d55_args)):
            module.D32_RUNTIME_MARKER_BOARD_MM = copy.deepcopy(marker_map)
            module.D32_RUNTIME_REQUIRED_IDS = list(ids)
            namespace.d32_runtime_marker_board_mm = copy.deepcopy(marker_map)
            namespace.d32_runtime_required_ids = list(ids)
            namespace._board_marker_map = copy.deepcopy(marker_map)
            namespace._board_required_ids = list(ids)
            namespace._board_roi_source = "dual_roarm_folding_board_config"
            namespace.board_roi = bool(getattr(namespace, "d32_board_roi", True))
            namespace.board_roi_strict = bool(getattr(namespace, "d35_strict_aruco_roi", True))
            if hasattr(module, "_D34_RUNTIME_ARGS"):
                module._D34_RUNTIME_ARGS = namespace
            if hasattr(module, "_D26V3_RUNTIME_ARGS"):
                module._D26V3_RUNTIME_ARGS = namespace

    def _set_authoritative_board_geometry(self) -> None:
        for module, namespace in ((self.d56, self.d56_args), (self.d54, self.d54_args), (self.d55, self.d55_args)):
            action = (
                "D56_WAIST_LIFT_LAYDOWN" if module is self.d56 else
                "D54_OUTER_PULL" if module is self.d54 else
                "D55_PRESS_SWEEP"
            )
            action_h = self._H_for_action(action)
            if hasattr(module, "_d38_sync_e49_runtime"):
                module._d38_sync_e49_runtime(namespace, action_h)
            empty_path = getattr(namespace, "d34_empty_board_path", None)
            if module is self.d56 and empty_path and hasattr(module, "_d56v20_prepare_persistent_empty_board"):
                try:
                    module._d56v20_prepare_persistent_empty_board(empty_path)
                except Exception as exc:
                    print(f"[EMPTY-BOARD-WARN] {module.__name__} persistent: {exc!r}")
            if empty_path and hasattr(module, "_d34_load_empty_board"):
                try:
                    module._d34_load_empty_board(empty_path)
                except Exception as exc:
                    print(f"[EMPTY-BOARD-WARN] {module.__name__}: {exc!r}")

    def _load_homography(self, path: str) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Dict[str, Any]]:
        """Load whatever is usable from the cache; startup repairs missing geometry automatically."""
        p = Path(_require_file(path, "Homography cache"))
        with open(p, "r", encoding="utf-8") as f:
            payload = json.load(f)

        def valid_matrix(value: Any) -> Optional[np.ndarray]:
            try:
                matrix = np.asarray(value, np.float32)
            except Exception:
                return None
            if matrix.shape != (3, 3) or not np.all(np.isfinite(matrix)):
                return None
            return matrix.copy()

        H = valid_matrix(payload.get("H"))
        raw_H = valid_matrix(payload.get("raw_H"))
        if H is None:
            print(f"[H-STARTUP] corrected H missing/invalid in {p}; automatic camera recovery will run")
        if raw_H is None:
            print(f"[H-STARTUP] raw_H missing/invalid in {p}; automatic camera recovery will run")
        return H, raw_H, payload

    def _H_for_action(self, action: Optional[str]) -> np.ndarray:
        if action == "D56_WAIST_LIFT_LAYDOWN":
            return self.H_raw
        return self.H_corrected

    def _h_from_marker_centers(self, centers: Dict[int, np.ndarray], marker_map: Dict[str, Any], required_ids: Sequence[int]) -> Optional[np.ndarray]:
        if not all(int(mid) in centers and str(int(mid)) in marker_map for mid in required_ids):
            return None
        img_pts = np.asarray([centers[int(mid)] for mid in required_ids], np.float32)
        board_pts = np.asarray([marker_map[str(int(mid))] for mid in required_ids], np.float32)
        H, _ = self.cv2.findHomography(img_pts, board_pts)
        if H is None:
            return None
        H = np.asarray(H, np.float32)
        return H if H.shape == (3, 3) and np.all(np.isfinite(H)) else None

    @staticmethod
    def _merge_marker_centers(corners: Any, ids: Any, store: Dict[int, np.ndarray]) -> None:
        if ids is None:
            return
        try:
            for corner, marker_id in zip(corners, ids.flatten()):
                pts = np.asarray(corner, np.float32).reshape(4, 2)
                store[int(marker_id)] = pts.mean(axis=0).astype(np.float32)
        except Exception:
            return

    def _marker_pixels_from_h(
        self,
        H: np.ndarray,
        marker_map: Dict[str, Any],
        required_ids: Sequence[int],
    ) -> Optional[np.ndarray]:
        """Project board-marker centers into the image geometry owned by H."""
        try:
            inv = np.linalg.inv(np.asarray(H, np.float64))
            board_pts = np.asarray(
                [marker_map[str(int(mid))] for mid in required_ids], np.float32
            ).reshape(-1, 1, 2)
            pixels = self.cv2.perspectiveTransform(
                board_pts, np.asarray(inv, np.float32)
            ).reshape(-1, 2)
        except Exception:
            return None
        if len(pixels) != len(required_ids) or not np.all(np.isfinite(pixels)):
            return None
        return np.asarray(pixels, np.float32)

    def _cached_corrected_h_sanity(
        self,
        H: np.ndarray,
        marker_map: Dict[str, Any],
        required_ids: Sequence[int],
    ) -> Tuple[bool, str]:
        """Geometry-only guard used when an old cache lost camera metadata.

        This does not recalibrate the board.  It only verifies that the existing
        numeric corrected H still projects the authoritative board-marker centers
        to a large, finite quadrilateral inside the current 1280x720 image.
        """
        pixels = self._marker_pixels_from_h(H, marker_map, required_ids)
        if pixels is None:
            return False, "cached corrected H cannot project board marker centers"
        w = float(self.args.width)
        h = float(self.args.height)
        margin_x = 0.18 * w
        margin_y = 0.18 * h
        if np.any(pixels[:, 0] < -margin_x) or np.any(pixels[:, 0] > w + margin_x):
            return False, f"projected marker X outside camera: {np.round(pixels,1).tolist()}"
        if np.any(pixels[:, 1] < -margin_y) or np.any(pixels[:, 1] > h + margin_y):
            return False, f"projected marker Y outside camera: {np.round(pixels,1).tolist()}"
        try:
            hull = self.cv2.convexHull(pixels.reshape(-1, 1, 2))
            area = float(abs(self.cv2.contourArea(hull)))
        except Exception:
            area = 0.0
        min_area = 0.06 * w * h
        if area < min_area:
            return False, f"projected board area too small: {area:.0f}px2"
        return True, f"numeric corrected H sane (projected board area={area:.0f}px2)"

    @staticmethod
    def _bilinear_sample_map(map_f32: np.ndarray, pixels: np.ndarray) -> np.ndarray:
        """Bilinear sample a single-channel float map at Nx2 pixel coordinates."""
        m = np.asarray(map_f32, np.float32)
        pts = np.asarray(pixels, np.float32).reshape(-1, 2)
        h, w = m.shape[:2]
        out = np.full((len(pts),), np.nan, np.float32)
        for i, (x, y) in enumerate(pts):
            if not np.isfinite(x + y) or x < 0.0 or y < 0.0 or x > w - 1 or y > h - 1:
                continue
            x0 = int(np.floor(x)); y0 = int(np.floor(y))
            x1 = min(w - 1, x0 + 1); y1 = min(h - 1, y0 + 1)
            dx = float(x - x0); dy = float(y - y0)
            out[i] = (
                (1.0 - dx) * (1.0 - dy) * float(m[y0, x0])
                + dx * (1.0 - dy) * float(m[y0, x1])
                + (1.0 - dx) * dy * float(m[y1, x0])
                + dx * dy * float(m[y1, x1])
            )
        return out

    def _undistort_source_maps(self) -> Tuple[np.ndarray, np.ndarray, str]:
        """Return corrected-pixel -> raw-pixel maps from the active undistorter.

        Primary path uses the exact public correct() operation on a float coordinate
        image.  Therefore raw_H reconstruction follows the same remap that D54/D55
        already use and does not require ArUco markers to be visible at startup.
        """
        h = int(self.args.height)
        w = int(self.args.width)

        # Generic/public path: remap an XY coordinate image through the exact same
        # CameraUndistorter.correct() call used for real camera frames.
        try:
            yy, xx = np.indices((h, w), dtype=np.float32)
            probe = np.zeros((h, w, 3), np.float32)
            probe[:, :, 0] = xx
            probe[:, :, 1] = yy
            mapped = np.asarray(self.undistorter.correct(probe))
            if mapped.shape[:2] == (h, w) and mapped.ndim == 3 and mapped.shape[2] >= 2:
                map_x = np.asarray(mapped[:, :, 0], np.float32)
                map_y = np.asarray(mapped[:, :, 1], np.float32)
                finite = np.isfinite(map_x) & np.isfinite(map_y)
                if (float(np.mean(finite)) > 0.95 and
                    float(np.nanmax(map_x)) > 0.55 * w and
                    float(np.nanmax(map_y)) > 0.55 * h):
                    return map_x, map_y, "CameraUndistorter.correct(XY probe)"
        except Exception as exc:
            print(f"[H-AUTO-WARN] XY-probe undistort map extraction failed: {exc!r}")

        # Compatibility path for camera_undistort.py implementations that reject
        # float images but expose their OpenCV remap arrays internally.
        pairs = (
            ("map1", "map2"), ("_map1", "_map2"),
            ("map_x", "map_y"), ("_map_x", "_map_y"),
            ("mapx", "mapy"), ("_mapx", "_mapy"),
        )
        for n1, n2 in pairs:
            m1 = getattr(self.undistorter, n1, None)
            m2 = getattr(self.undistorter, n2, None)
            if not isinstance(m1, np.ndarray):
                continue
            try:
                if (m1.ndim == 3 and m1.shape[2] == 2 and
                    np.issubdtype(m1.dtype, np.floating)):
                    mx = np.asarray(m1[:, :, 0], np.float32)
                    my = np.asarray(m1[:, :, 1], np.float32)
                elif isinstance(m2, np.ndarray):
                    # Handles the common CV_16SC2 + CV_16UC1 representation.
                    mx, my = self.cv2.convertMaps(m1, m2, self.cv2.CV_32FC1)
                    mx = np.asarray(mx, np.float32)
                    my = np.asarray(my, np.float32)
                else:
                    continue
                if mx.shape == (h, w) and my.shape == (h, w):
                    return mx, my, f"undistorter.{n1}/{n2}"
            except Exception:
                continue
        raise RuntimeError("active CameraUndistorter remap could not be extracted")

    def _derive_raw_h_from_corrected_h(
        self,
        corrected_H: np.ndarray,
        marker_map: Dict[str, Any],
        required_ids: Sequence[int],
    ) -> Tuple[np.ndarray, str]:
        """Reconstruct D56 raw_H from corrected H + the active lens remap.

        The four authoritative board marker CENTER coordinates are projected into
        corrected pixels using the existing corrected H.  The exact undistortion
        source map then converts those corrected pixels back to their raw-camera
        pixel positions.  A raw-pixel -> board homography is fitted from those same
        four physical references.  No visible ArUco marker or operator key is needed.
        """
        corr_px = self._marker_pixels_from_h(corrected_H, marker_map, required_ids)
        if corr_px is None:
            raise RuntimeError("corrected H cannot project authoritative marker centers")
        map_x, map_y, map_source = self._undistort_source_maps()
        raw_x = self._bilinear_sample_map(map_x, corr_px)
        raw_y = self._bilinear_sample_map(map_y, corr_px)
        raw_px = np.stack([raw_x, raw_y], axis=1).astype(np.float32)
        if not np.all(np.isfinite(raw_px)):
            raise RuntimeError(
                f"corrected board references fall outside undistort map: {np.round(corr_px,1).tolist()}"
            )
        w = float(self.args.width); h = float(self.args.height)
        if (np.any(raw_px[:, 0] < -5.0) or np.any(raw_px[:, 0] > w + 5.0) or
            np.any(raw_px[:, 1] < -5.0) or np.any(raw_px[:, 1] > h + 5.0)):
            raise RuntimeError(f"derived raw marker pixels outside camera: {np.round(raw_px,1).tolist()}")
        board_pts = np.asarray(
            [marker_map[str(int(mid))] for mid in required_ids], np.float32
        )
        raw_H, _ = self.cv2.findHomography(raw_px, board_pts, method=0)
        if raw_H is None:
            raise RuntimeError("cv2.findHomography failed while deriving raw_H")
        raw_H = np.asarray(raw_H, np.float32)
        if raw_H.shape != (3, 3) or not np.all(np.isfinite(raw_H)):
            raise RuntimeError("derived raw_H is invalid")

        # Numerical self-check at the same four physical references.
        projected = self.cv2.perspectiveTransform(
            raw_px.reshape(-1, 1, 2), raw_H
        ).reshape(-1, 2)
        err = np.linalg.norm(projected - board_pts, axis=1)
        max_err = float(np.max(err)) if len(err) else 999.0
        if not np.isfinite(max_err) or max_err > 2.0:
            raise RuntimeError(f"derived raw_H self-check failed: maxErr={max_err:.2f}mm")
        return raw_H, f"{map_source}; marker-center fit maxErr={max_err:.3f}mm"

    def _ensure_startup_homography(self) -> None:
        """Repair H/raw_H automatically without requiring L or visible ArUco markers.

        Normal deployed case:
          * numeric corrected H already exists from the board calibration;
          * if its metadata was lost, retain that numeric H after a board-projection
            sanity check and attach the current camera metadata;
          * if raw_H is missing, derive it deterministically from corrected H and the
            exact active CameraUndistorter remap.

        Live ArUco recovery is kept only as a last-resort for a genuinely missing or
        numerically invalid corrected H.  It is not part of normal startup anymore.
        """
        marker_cfg = self.config.get("aruco", {})
        marker_map = copy.deepcopy(marker_cfg.get("marker_board_mm", {}))
        required_ids = [int(v) for v in marker_cfg.get("required_ids", [0, 1, 2, 3])]
        if not marker_map or not required_ids:
            raise RuntimeError("authoritative ArUco marker map/required_ids unavailable")

        matcher = getattr(self.d54, "_d53_camera_geometry_matches", None)
        corrected_numeric = bool(
            self.H_corrected is not None
            and np.asarray(self.H_corrected).shape == (3, 3)
            and np.all(np.isfinite(self.H_corrected))
        )
        raw_ok = bool(
            self.H_raw is not None
            and np.asarray(self.H_raw).shape == (3, 3)
            and np.all(np.isfinite(self.H_raw))
        )

        corrected_ok = False
        corrected_reason = "corrected H unavailable"
        metadata_needs_refresh = False
        if corrected_numeric:
            cached_geometry = dict(self.h_bundle.get("camera_geometry", {}) or {})
            if matcher is not None:
                corrected_ok, corrected_reason = matcher(
                    cached_geometry, self.corrected_camera_geometry, False
                )
            if not corrected_ok:
                sane, sane_reason = self._cached_corrected_h_sanity(
                    self.H_corrected, marker_map, required_ids
                )
                if sane:
                    print(
                        f"[H-AUTO] corrected H metadata stale/missing ({corrected_reason}); "
                        f"preserving calibrated numeric H: {sane_reason}"
                    )
                    corrected_ok = True
                    corrected_reason = "numeric corrected H preserved; camera metadata refreshed"
                    metadata_needs_refresh = True

        # Genuine missing/invalid corrected H: automatic live ArUco is the only
        # physically grounded recovery available. This branch should be rare.
        if not corrected_ok:
            print(f"[H-AUTO] corrected H unavailable ({corrected_reason}); trying automatic live recovery")
            corrected_centers: Dict[int, np.ndarray] = {}
            for attempt in range(1, 37):
                ok, raw = self._read_raw(flush=0)
                if not ok or raw is None:
                    time.sleep(0.04); continue
                corrected = self.undistorter.correct(raw)
                corners, ids, _ = self.d54.detect_markers(corrected, self.aruco_detector)
                self._merge_marker_centers(corners, ids, corrected_centers)
                candidate = self._h_from_marker_centers(corrected_centers, marker_map, required_ids)
                if candidate is not None:
                    self.H_corrected = candidate
                    corrected_ok = True
                    corrected_reason = f"live corrected ArUco recovery frame={attempt} IDs={sorted(corrected_centers)}"
                    metadata_needs_refresh = True
                    print(f"[H-AUTO] {corrected_reason}")
                    break
                time.sleep(0.04)
            if not corrected_ok:
                raise RuntimeError(
                    "corrected Homography is genuinely missing/invalid and cannot be recovered automatically; "
                    f"seen corrected ArUco IDs={sorted(corrected_centers)}"
                )

        # This is the normal repair for the user's current cache. It requires no
        # marker visibility and produces the raw geometry expected by unchanged D56.
        if not raw_ok:
            self.H_raw, derive_reason = self._derive_raw_h_from_corrected_h(
                self.H_corrected, marker_map, required_ids
            )
            raw_ok = True
            metadata_needs_refresh = True
            print(f"[H-AUTO] raw_H derived without ArUco visibility: {derive_reason}")

        if not corrected_ok or not raw_ok:
            raise RuntimeError("startup Homography repair did not produce corrected H + raw_H")

        if metadata_needs_refresh:
            self.d54.save_homography(
                self.args.hfile, self.H_corrected, self.H_raw, self.corrected_camera_geometry
            )
            with open(self.args.hfile, "r", encoding="utf-8") as stream:
                self.h_bundle = json.load(stream)
            self._sync_module_runtime()
            print(f"[H-AUTO] corrected H + raw_H saved automatically: {self.args.hfile}")
        else:
            print(f"[H-STARTUP] cached corrected H + raw_H ready: {corrected_reason}")

    def _validate_h_geometry(self) -> None:
        if self.H_corrected is None or self.H_raw is None:
            raise RuntimeError("startup Homography recovery did not produce both corrected H and raw_H")
        cached_geometry = dict(self.h_bundle.get("camera_geometry", {}) or {})
        matcher = getattr(self.d54, "_d53_camera_geometry_matches", None)
        if matcher is None:
            raise RuntimeError("D54 original camera-geometry validator unavailable")
        ok, reason = matcher(cached_geometry, self.corrected_camera_geometry, False)
        if not ok:
            raise RuntimeError(f"D54/D55 corrected Homography rejected: {reason}")
        print(f"[H-D54-D55] corrected H accepted: {reason}")
        print("[H-D56] raw_H selected for D56-45 RAW frame")

    def _runtime_metadata(self, actual_size: Tuple[int, int]) -> Dict[str, Any]:
        paths = {
            "merged_source": str(Path(__file__).resolve()),
            "d56_source": self.d56_path,
            "d54_source": self.d54_path,
            "d55_source": self.d55_path,
            "d58_source": self.d58_path,
            "config": self.args.config,
            "homography": self.args.hfile,
            "camera_controls": self.args.camera_controls_json,
        }
        return {
            "motion_version": MOTION_VERSION,
            "motion_policy_version": MOTION_POLICY_VERSION,
            "d56_recognition_version": D56_RECOGNITION_VERSION,
            "collection_phase": self.args.collection_phase,
            "source_files": {k: {"path": v, "sha256": _sha256_file(v)} for k, v in paths.items()},
            "models": {
                "segmentation": _model_metadata(self.args.seg_model),
                "pose": _model_metadata(self.args.pose_model),
            },
            "camera": {
                "index": int(self.args.camera), "device": self.args.camera_device,
                "actual_size": list(actual_size), "geometry": self.camera_geometry_meta,
                "controls": self.camera_controls_meta,
            },
            "homography_sha256": _sha256_file(self.args.hfile),
            "homography_matrix_corrected": self.H_corrected.tolist(),
            "homography_matrix_raw": self.H_raw.tolist(),
            "effective_args": _json_safe(vars(self.args)),
            "mode": self.args.mode,
        }

    def _open_camera(self):
        # Call the D56-45 camera opener itself.  This intentionally avoids the
        # merged collector owning a subtly different FOURCC/remap pipeline.
        cap = self.d56.open_camera(self.d56_args)
        print("[CAMERA-OPEN] delegated to d56-45.open_camera()")
        return cap

    def _read_raw(self, flush: int = 0) -> Tuple[bool, Optional[np.ndarray]]:
        with self.cap_lock:
            frame = None
            ok = False
            for _ in range(max(0, int(flush)) + 1):
                ok, frame = self.cap.read()
                if not ok:
                    break
            return bool(ok), None if frame is None else frame.copy()

    def _frame_for_action(self, raw: np.ndarray, action: Optional[str]) -> np.ndarray:
        """Use exactly the camera geometry selected by the original source."""
        if action in {"D54_OUTER_PULL", "D55_PRESS_SWEEP", "D58_CIRC_POSITION"}:
            return self.undistorter.correct(raw)
        return raw.copy()

    def _capture_action_frame(self, action: Optional[str], flush: int = 0) -> FrameBundle:
        # Original manual-I paths consume the newest available frame.  Do not
        # add a new global stillness veto that the source code does not have.
        ok, raw = self._read_raw(flush=flush)
        if not ok or raw is None:
            raise RuntimeError("fresh camera frame unavailable")
        return FrameBundle(raw, self._frame_for_action(raw, action), _now_text())

    def _capture_i_frame_from_live(self, action: Optional[str], timeout_s: float = 0.50) -> FrameBundle:
        """Freeze the newest RAW frame already owned by the main camera loop.

        main-32 ports the TOP-VLA snapshot discipline: background inference never
        calls VideoCapture.read().  RAW remains the live preview geometry; only a
        finalized I snapshot is corrected once when the selected action requires it.
        """
        deadline = time.monotonic() + max(0.05, float(timeout_s))
        raw = None
        captured_mono = 0.0
        while time.monotonic() < deadline:
            with self.live_frame_lock:
                if self.latest_live_raw is not None:
                    raw = self.latest_live_raw.copy()
                    captured_mono = float(self.latest_live_monotonic)
            if raw is not None:
                break
            time.sleep(0.01)
        if raw is None:
            raise RuntimeError("main-loop live RAW snapshot unavailable")
        age_ms = max(0.0, (time.monotonic() - captured_mono) * 1000.0) if captured_mono > 0 else -1.0
        t0 = time.monotonic()
        corrected = self._frame_for_action(raw, action)
        correction_ms = (time.monotonic() - t0) * 1000.0
        print(
            f"[MAIN32-SNAPSHOT] action={action} source=MAIN_LOOP_RAW "
            f"age={age_ms:.1f}ms correction={correction_ms:.1f}ms "
            f"geometry={'CORRECTED_ONCE' if action in {'D54_OUTER_PULL','D55_PRESS_SWEEP','D58_CIRC_POSITION'} else 'RAW'}"
        )
        return FrameBundle(raw, corrected, _now_text())

    def _verify_sources_unchanged(self) -> None:
        current = {
            "D50_BASKET_SWING_LAYDOWN": _sha256_file(self.d50_path),
            "D54_OUTER_PULL": _sha256_file(self.d54_path),
            "D55_PRESS_SWEEP": _sha256_file(self.d55_path),
            "D56_WAIST_LIFT_LAYDOWN": _sha256_file(self.d56_path),
            "D58_CIRC_POSITION": _sha256_file(self.d58_path),
        }
        changed = [name for name, digest in current.items() if digest != self.source_sha256.get(name)]
        if changed:
            raise RuntimeError(f"original source changed during session: {changed}")

    def _lock_and_save_homography(self) -> None:
        """Original L flow: lock corrected H and raw_H from one camera moment."""
        with self.state_lock:
            if self.motion_busy:
                print("[H] lock blocked during robot motion")
                return
            if self.recorder.review_pending():
                print("[H] lock blocked: finish G/B/K and Y/N for the previous action")
                return
        try:
            self._verify_sources_unchanged()
            ok, raw = self._read_raw(flush=max(1, int(self.args.imaging_frame_flush_count)))
            if not ok or raw is None:
                raise RuntimeError("fresh camera frame unavailable")
            corrected = self.undistorter.correct(raw)
            marker_cfg = self.config.get("aruco", {})
            marker_map = copy.deepcopy(marker_cfg.get("marker_board_mm", {}))
            required_ids = [int(v) for v in marker_cfg.get("required_ids", [0, 1, 2, 3])]

            raw_corners, raw_ids, _ = self.d54.detect_markers(raw, self.aruco_detector)
            raw_h_live, raw_centers = self.d54.compute_homography(
                raw_corners, raw_ids, marker_map, required_ids
            )
            corners, ids, _ = self.d54.detect_markers(corrected, self.aruco_detector)
            h_live, centers = self.d54.compute_homography(
                corners, ids, marker_map, required_ids
            )
            if h_live is None or raw_h_live is None:
                corrected_seen = sorted(int(v) for v in centers.keys())
                raw_seen = sorted(int(v) for v in raw_centers.keys())
                self.status = "H LOCK FAILED: SHOW ARUCO ID0-ID3"
                print(
                    "[H] cannot lock: need ArUco ID0~ID3 in both corrected and raw "
                    f"ELP frames; corrected={corrected_seen} raw={raw_seen}"
                )
                return

            h_new = np.asarray(h_live, dtype=np.float32).copy()
            raw_h_new = np.asarray(raw_h_live, dtype=np.float32).copy()
            if not np.all(np.isfinite(h_new)) or not np.all(np.isfinite(raw_h_new)):
                raise RuntimeError("detected Homography contains non-finite values")

            # Keep each original module's strict inner-marker ROI in its native
            # camera geometry: D54/D55 corrected, D56 raw.
            for module, namespace in ((self.d54, self.d54_args), (self.d55, self.d55_args)):
                module._d38_update_live_inner_marker_roi(
                    corners, ids, required_ids, namespace, corrected.shape
                )
            self.d56._d38_update_live_inner_marker_roi(
                raw_corners, raw_ids, required_ids, self.d56_args, raw.shape
            )

            self.d54.save_homography(
                self.args.hfile, h_new, raw_h_new, self.corrected_camera_geometry
            )
            self.H_corrected = h_new
            self.H_raw = raw_h_new
            with open(self.args.hfile, "r", encoding="utf-8") as stream:
                self.h_bundle = json.load(stream)
            self._sync_module_runtime()
            self._set_authoritative_board_geometry()
            with self.state_lock:
                self.locked = None
                self.display_image = None
                self.empty_baseline_ready = False
                self.status = "H SAVED: EMPTY BOARD REQUIRED, PRESS E ONCE"
            print(f"[H] ELP corrected H + raw_H locked and saved: {self.args.hfile}")
            print("[NEXT] clear arms, show empty board, then press E once")
        except Exception as exc:
            self.status = f"H LOCK ERROR: {type(exc).__name__}"
            print(f"[H-ERROR] {exc!r}")

    def _capture_empty_board(self) -> None:
        with self.state_lock:
            if self.motion_busy:
                print("[E] blocked: robot motion is running")
                return
            if self.recorder.review_pending():
                print("[E] blocked: finish G/B/K and Y/N for the previous action")
                return
        if not self._ensure_camera_clear("BEFORE_E", allow_move=False):
            self.status = "E BLOCKED: ARMS ARE NOT AT STANDBY"
            print("[E] blocked: arms are not at the verified standby position")
            return
        try:
            self._verify_sources_unchanged()
            ok, raw = self._read_raw(flush=max(1, int(self.args.imaging_frame_flush_count)))
            if not ok or raw is None:
                raise RuntimeError("empty-board camera frame unavailable")
            corrected = self.undistorter.correct(raw)
            jobs = (
                (self.d54, self.d54_args, corrected, self.H_corrected, True, "D54-CORRECTED"),
                (self.d55, self.d55_args, corrected, self.H_corrected, False, "D55-CORRECTED"),
                (self.d56, self.d56_args, raw, self.H_raw, True, "D56-RAW"),
            )
            completed: List[str] = []
            for module, namespace, frame, action_h, save, label in jobs:
                setter = getattr(module, "_d34_set_empty_board", None)
                sync = getattr(module, "_d38_sync_e49_runtime", None)
                if setter is None or sync is None:
                    raise RuntimeError(f"{label} original E handler dependency unavailable")
                if not setter(frame.copy(), namespace.d34_empty_board_path, save=save):
                    raise RuntimeError(f"{label} empty-board baseline failed")
                sync(namespace, action_h)
                if hasattr(namespace, "_d56v20_fallback_last_trusted_area_px"):
                    namespace._d56v20_fallback_last_trusted_area_px = 0.0
                completed.append(label)
            self.empty_baseline_ready = True
            self.locked = None
            self.display_image = None
            self.status = "E READY: PLACE GARMENT, TYPE 54/55/56/58, THEN I"
            print("[E-READY] one empty-board instant installed: " + " + ".join(completed))
            print("[NEXT] place garment -> action 54/55/56/58 -> i")
        except Exception as exc:
            self.empty_baseline_ready = False
            self.status = f"E FAILED: {type(exc).__name__}"
            print(f"[E-ERROR] {exc!r}")

    def _query_arm_xyz(self, arm_key: str, retries: int = 2) -> Optional[np.ndarray]:
        if self.args.mode == "dry-run":
            return None
        arm = self.arms.get(arm_key)
        if arm is None:
            return None
        for attempt in range(1, max(1, int(retries)) + 1):
            report = self.d54._d31_query_feedback(
                arm, timeout_s=float(self.args.standby_feedback_timeout_s)
            )
            xyz = self.d54._d31_feedback_xyz(report)
            if xyz is not None:
                return np.asarray(xyz, np.float32)
            if attempt < max(1, int(retries)):
                print(f"[CAMERA-CLEAR] {arm_key} feedback retry {attempt}/{int(retries)}")
                time.sleep(0.12)
        return None

    def _feedback_camera_clear(self) -> bool:
        if self.args.mode == "dry-run":
            return True
        targets = {k: np.asarray(self.d56.standby_roarm_pose(self.cfg56, k)[:3], np.float32) for k in ("arm1", "arm2")}
        success = True
        for arm_key in ("arm1", "arm2"):
            xyz = self._query_arm_xyz(arm_key, retries=2)
            if xyz is None:
                success = False
                print(f"[CAMERA-CLEAR] {arm_key} feedback unavailable")
                continue
            error = float(np.linalg.norm(xyz - targets[arm_key]))
            z_ok = float(xyz[2]) >= float(targets[arm_key][2]) - float(self.args.standby_z_tolerance_mm)
            arm_ok = bool(error <= float(self.args.standby_xyz_tolerance_mm) and z_ok)
            print(f"[CAMERA-CLEAR] {arm_key} error={error:.1f}mm zOK={z_ok} verified={arm_ok}")
            success = success and arm_ok
        return success

    def _ensure_camera_clear(self, reason: str, allow_move: bool = False) -> bool:
        if self.args.mode == "dry-run":
            return True
        if allow_move:
            moved = self.d56.move_arms_to_standby(
                self.arms, self.cfg56, move_command=int(self.args.move_command), reason=f"BOTTOM_VLA_{reason}"
            )
            if not moved:
                return False
            time.sleep(max(0.0, float(self.args.imaging_standby_wait_s)))
        attempts = int(self.args.standby_verify_retries) if allow_move else 2
        for attempt in range(1, max(1, attempts) + 1):
            if self._feedback_camera_clear():
                for _ in range(max(1, int(self.args.imaging_frame_flush_count))):
                    self._read_raw(flush=0)
                return True
            print(f"[CAMERA-CLEAR] verify retry {attempt}/{max(1, attempts)}")
            time.sleep(0.25)
        return False

    def _adapter_for_action(self, action: Optional[str]) -> Tuple[ModuleType, argparse.Namespace, Any, np.ndarray]:
        if action == "D54_OUTER_PULL":
            return self.d54, self.d54_args, self.cfg54, self.H_corrected
        if action == "D55_PRESS_SWEEP":
            return self.d55, self.d55_args, self.cfg55, self.H_corrected
        if action == "D58_CIRC_POSITION":
            return self.d58, self.d58_args, None, self.H_corrected
        return self.d56, self.d56_args, self.cfg56, self.H_raw

    def _module_cfg_for_action(self, action: Optional[str]) -> Tuple[ModuleType, Any]:
        if action == "D58_CIRC_POSITION":
            return self.d56, self.cfg56
        module, _namespace, cfg, _H = self._adapter_for_action(action)
        return module, cfg

    def _infer_d58_pose_fast(self, frame: np.ndarray, mask: Any) -> Tuple[Any, str]:
        """Try one cheap D58 pose transform first, then preserve the full original TTA fallback.

        Priority is the most recently successful D58 transform.  Before a hint
        exists, 0deg/no-flip is tried once.  Any failure immediately restores
        the exact original D58 TTA lists and runs the full search.
        """
        original_angles = str(self.d58_args.pose_tta_angles)
        original_flips = str(self.d58_args.pose_tta_flips)
        hint = self.d58_pose_tta_hint or (0.0, "none")
        try:
            self.d58_args.pose_tta_angles = f"{float(hint[0]):g}"
            self.d58_args.pose_tta_flips = str(hint[1])
            t0 = time.monotonic()
            pose, status = self.d58.infer_pose(
                self.pose_model, frame, self.H_corrected, mask, self.d58_args
            )
            fast_dt = time.monotonic() - t0
            if pose is not None:
                source = str(getattr(pose, "source", ""))
                try:
                    token = source.replace("TTA", "", 1).strip()
                    deg_text, flip_text = token.split("/", 1)
                    self.d58_pose_tta_hint = (float(deg_text), str(flip_text).strip())
                except Exception:
                    self.d58_pose_tta_hint = (float(hint[0]), str(hint[1]))
                print(
                    f"[D58-POSE-FAST] PASS hint={hint[0]:+g}/{hint[1]} "
                    f"time={fast_dt*1000:.0f}ms; full TTA skipped"
                )
                return pose, status + " FAST_HINT"
            print(
                f"[D58-POSE-FAST] miss hint={hint[0]:+g}/{hint[1]} "
                f"time={fast_dt*1000:.0f}ms -> original full TTA fallback"
            )
        finally:
            self.d58_args.pose_tta_angles = original_angles
            self.d58_args.pose_tta_flips = original_flips

        t1 = time.monotonic()
        pose, status = self.d58.infer_pose(
            self.pose_model, frame, self.H_corrected, mask, self.d58_args
        )
        print(f"[D58-POSE-FULL] time={(time.monotonic()-t1):.3f}s")
        if pose is not None:
            source = str(getattr(pose, "source", ""))
            try:
                token = source.replace("TTA", "", 1).strip()
                deg_text, flip_text = token.split("/", 1)
                self.d58_pose_tta_hint = (float(deg_text), str(flip_text).strip())
                print(f"[D58-POSE-HINT] cached={self.d58_pose_tta_hint[0]:+g}/{self.d58_pose_tta_hint[1]}")
            except Exception:
                pass
        return pose, status

    def _infer_for_action(self, action: Optional[str], frame: np.ndarray) -> Any:
        if action == "D58_CIRC_POSITION":
            mask, mask_status = self.d58.infer_mask(self.seg_model, frame, self.H_corrected, self.config, self.d58_args)
            pose = None
            pose_status = "SKIP(no mask)"
            if mask is not None:
                pose, pose_status = self._infer_d58_pose_fast(frame, mask)
            print(f"[D58-MASK] {mask_status}")
            print(f"[D58-POSE] {pose_status}")
            obs = SimpleNamespace(mask=mask, pose=pose, valid=mask is not None, reason=mask_status, d38_mask_source="D58_SEG")
        else:
            module, namespace, cfg, action_h = self._adapter_for_action(action)
            obs = module.infer_bottom_observation(
                self.seg_model, self.pose_model, frame, action_h, namespace, cfg
            )
        self.inference_serial += 1
        try:
            obs.bottom_vla_inference_serial = int(self.inference_serial)
        except Exception:
            pass
        return obs

    def _base_overlay(self, action: Optional[str], frame: np.ndarray, obs: Any) -> np.ndarray:
        module, namespace, cfg, action_h = self._adapter_for_action(action)
        return module.draw_bottom_overlay_safe(
            frame, action_h, obs, cfg, None, None, namespace, False, ""
        )

    def _d56_original_overlay(self, frame: np.ndarray, obs: Any, plan: Any) -> np.ndarray:
        """Reproduce D56-45's visible waist-ribbon + locked-plan layers."""
        ribbon_report = getattr(obs, "d56v7_waist_observer", None) if obs is not None else None
        ribbon_selected = (
            ribbon_report.get("selected")
            if isinstance(ribbon_report, dict) and isinstance(ribbon_report.get("selected"), dict)
            else None
        )
        plan_mode = str((getattr(plan, "metrics", {}) or {}).get("d42_plan_mode", ""))
        ribbon_active = bool(ribbon_selected is not None or plan_mode == "D56_15_WAIST_RIBBON")
        canvas = self.d56.draw_bottom_overlay_safe(
            frame, self.H_raw, obs, self.cfg56, None, None, self.d56_args, False, ""
        )
        if not ribbon_active:
            dual_plan = plan if str(getattr(plan, "action", "")).startswith("D47_") else None
            canvas = self.d56.draw_dual_wrinkle_plan_overlay(canvas, self.H_raw, dual_plan)
        canvas = self.d56._d56v7_draw_waist_observer(canvas, obs, self.d56_args)
        state = "LIVE_PREVIEW" if bool(getattr(plan, "ok", False)) else "LIVE_NO_PLAN"
        summary = str(getattr(plan, "reason", ""))
        return self.d56._d30_draw_overlay(canvas, self.H_raw, plan, state, summary)

    def _d54_d55_original_overlay(
        self,
        bundle: FrameBundle,
        action: str,
        obs: Any,
        plan: Any,
        heat: Any,
        effect_state: str = "WAITING",
        effect_summary: str = "No physical action evaluated yet",
    ) -> np.ndarray:
        """Use the original D54/D55 display geometry and complete layer order."""
        module, namespace, cfg, action_h = self._adapter_for_action(action)
        dual_view = plan if action == "D55_PRESS_SWEEP" else None
        d30_plan_view = plan if action == "D54_OUTER_PULL" else None
        canvas = module.draw_bottom_overlay_safe(
            bundle.corrected, action_h, obs, cfg, None, heat, namespace, False, ""
        )
        canvas = module.draw_dual_wrinkle_plan_overlay(canvas, action_h, dual_view)

        metrics = dict(getattr(d30_plan_view, "metrics", {}) or {})
        if str(metrics.get("d42_plan_mode", "")) in {
            "D45_V6_MASK_CURVE_WAIST", "D45_V6_MASK_CURVE_FAILED"
        }:
            curve_px = list(metrics.get("waist_curve_points_px", []) or [])
            valid_curve = np.asarray([
                [int(round(float(p[0]))), int(round(float(p[1])))]
                for p in curve_px if p is not None and len(p) >= 2
            ], dtype=np.int32)
            if len(valid_curve) >= 2:
                self.cv2.polylines(
                    canvas, [valid_curve.reshape(-1, 1, 2)], False,
                    (255, 0, 255), 3, self.cv2.LINE_AA,
                )
                self.cv2.putText(
                    canvas, "MASK_CURVE_WAIST", tuple(valid_curve[len(valid_curve) // 2]),
                    self.cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 0, 255), 2,
                    self.cv2.LINE_AA,
                )

        canvas = module._d22_draw_symmetry_overlay(canvas, action_h, obs)
        canvas = module._d28_draw_status_overlay(
            canvas, 0, 0, effect_state, effect_summary,
            enabled=bool(getattr(namespace, "d28_status_overlay", True)),
        )
        d30_state = "I_LOCKED" if bool(getattr(plan, "ok", False)) else "NO_SAFE_PLAN"
        d30_summary = str(getattr(plan, "reason", effect_summary))
        canvas = module._d30_draw_overlay(
            canvas, action_h, d30_plan_view, d30_state, d30_summary
        )
        canvas = module._d34_draw_status(canvas)
        canvas = module._d47v11_draw_next_action_bar(
            canvas, dual_view, d30_plan_view, None, False, False, ""
        )
        return canvas

    def _operator_overlay(
        self, bundle: FrameBundle, action: str, obs: Any, plan: Any, heat: Any = None
    ) -> np.ndarray:
        if action in {"D54_OUTER_PULL", "D55_PRESS_SWEEP"}:
            return self._d54_d55_original_overlay(bundle, action, obs, plan, heat)
        if action == "D58_CIRC_POSITION":
            mask = getattr(obs, "mask", None) if obs is not None else None
            if mask is not None:
                return self.d58.make_d58_overlay(bundle.corrected.copy(), mask, plan, self.H_corrected, self.config)
            return self._banner(bundle.corrected.copy(), None, action, False, str(getattr(plan, "reason", "NO MASK")))
        return self._d56_original_overlay(bundle.raw.copy(), obs, plan)

    def _draw_arm_plan(self, image: np.ndarray, plan: Any, module: ModuleType,
                       cfg: Any, action: str, action_h: np.ndarray) -> np.ndarray:
        out = image
        colors = {"arm2": (255, 120, 0), "arm1": (0, 255, 120)}
        for arm_key, item in dict(getattr(plan, "arm_points", {}) or {}).items():
            grip = item.get("grip_board", item.get("source_board"))
            target = item.get("target_board")
            if action == "D54_OUTER_PULL" and grip is not None and target is not None:
                try:
                    grip, target, _n, _mm, _rep = module._d54_v12_physical_board_points(
                        plan, arm_key, self.d54_args, cfg, self.config
                    )
                except Exception:
                    pass
            if grip is None:
                continue
            gp = module.board_to_pixel(action_h, float(grip[0]), float(grip[1]))
            tp = None if target is None else module.board_to_pixel(action_h, float(target[0]), float(target[1]))
            if gp is None:
                continue
            g = tuple(map(int, map(round, gp)))
            color = colors.get(arm_key, (255, 255, 255))
            self.cv2.drawMarker(out, g, color, self.cv2.MARKER_CROSS, 24, 3)
            self.cv2.putText(out, f"{arm_key.upper()} GRIP", (g[0] + 7, g[1] - 7), self.cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 2)
            if tp is not None:
                t = tuple(map(int, map(round, tp)))
                self.cv2.arrowedLine(out, g, t, color, 3, tipLength=0.18)
        return out

    def _banner(self, image: np.ndarray, state: Optional[str], action: str, ok: bool, reason: str) -> np.ndarray:
        out = image
        self.cv2.rectangle(out, (0, 0), (out.shape[1] - 1, 90), (0, 0, 0), -1)
        color = (0, 255, 0) if ok else (0, 0, 255)
        self.cv2.putText(out, f"ACTION={action} | PLAN={'OK' if ok else 'NO_PLAN'}",
                         (18, 32), self.cv2.FONT_HERSHEY_SIMPLEX, 0.70, color, 2)
        self.cv2.putText(out, str(reason)[:145], (18, 67), self.cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 1)
        return out

    def _apply_d54_extra_grip_inset(self, plan: Any) -> Dict[str, Any]:
        """Request +15 mm inward physical grip correction without changing D54 motion geometry."""
        report: Dict[str, Any] = {"extra_inset_mm": float(self.d54_extra_grip_inset_mm), "arms": {}}
        if plan is None or not bool(getattr(plan, "ok", False)):
            report["applied"] = False
            report["reason"] = "PLAN_NOT_OK"
            return report
        arm_points = dict(getattr(plan, "arm_points", {}) or {})
        changed = 0
        for arm_key in ("arm2", "arm1"):
            item = arm_points.get(arm_key)
            if not isinstance(item, dict) or "grip_board" not in item:
                continue
            old_requested = float(item.get(
                "physical_correction_requested_mm",
                item.get("physical_correction_mm", 0.0),
            ))
            new_requested = max(0.0, old_requested + float(self.d54_extra_grip_inset_mm))
            item["physical_correction_requested_mm"] = float(new_requested)
            item["vla_extra_grip_inset_mm"] = float(self.d54_extra_grip_inset_mm)
            report["arms"][arm_key] = {
                "before_requested_mm": float(old_requested),
                "after_requested_mm": float(new_requested),
            }
            changed += 1
        metrics = dict(getattr(plan, "metrics", {}) or {})
        metrics["vla_extra_grip_inset_mm"] = float(self.d54_extra_grip_inset_mm)
        metrics["vla_extra_grip_inset_policy"] = "D54_PHYSICAL_INWARD_REQUEST_PLUS_15MM"
        plan.metrics = metrics
        report["applied"] = bool(changed)
        report["reason"] = "OK" if changed else "NO_ARM_POINTS"
        if changed:
            print(
                "[VLA-D54-GRIP-INSET] +"
                f"{float(self.d54_extra_grip_inset_mm):.1f}mm inward "
                + " ".join(
                    f"{arm}={vals['before_requested_mm']:.1f}->{vals['after_requested_mm']:.1f}mm"
                    for arm, vals in report["arms"].items()
                )
            )
        return report

    def _enforce_d54_mask_core_20mm(self, plan: Any, obs: Any) -> Dict[str, Any]:
        """Move D54's grasp pair into the mask's 20 mm eroded core.

        The same XY delta is applied to each arm's pull target, preserving the
        original pull vector and distance.  If no core-supported point exists,
        the I-lock becomes NO_PLAN instead of grasping an inflated mask edge.
        """
        report: Dict[str, Any] = {"required_depth_mm": 20.0, "arms": {}}
        if plan is None or not bool(getattr(plan, "ok", False)):
            report.update(applied=False, reason="PLAN_NOT_OK")
            return report
        mask_obj = getattr(obs, "mask", None)
        if mask_obj is None:
            plan.ok = False
            plan.reason = "D54 mask-core 20mm unavailable: mask missing"
            report.update(applied=False, reason="MASK_MISSING")
            return report
        mask_u8 = (np.asarray(mask_obj.mask_u8, np.uint8) > 0).astype(np.uint8)
        inside_dt = self.cv2.distanceTransform(mask_u8, self.cv2.DIST_L2, 5)
        mask_center = np.asarray(mask_obj.center_board, np.float32).reshape(2)

        def depth_mm(point: np.ndarray) -> float:
            p = np.asarray(point, np.float32).reshape(2)
            px = self.d54.board_to_pixel(self.H_corrected, float(p[0]), float(p[1]))
            px_x = self.d54.board_to_pixel(self.H_corrected, float(p[0] + 1.0), float(p[1]))
            px_y = self.d54.board_to_pixel(self.H_corrected, float(p[0]), float(p[1] + 1.0))
            if px is None or px_x is None or px_y is None:
                return 0.0
            scale = 0.5 * (
                float(np.linalg.norm(np.asarray(px_x) - np.asarray(px)))
                + float(np.linalg.norm(np.asarray(px_y) - np.asarray(px)))
            )
            x, y = int(round(float(px[0]))), int(round(float(px[1])))
            if not (0 <= y < inside_dt.shape[0] and 0 <= x < inside_dt.shape[1]):
                return 0.0
            return float(inside_dt[y, x]) / max(scale, 1e-6)

        for arm_key in ("arm2", "arm1"):
            item = dict(getattr(plan, "arm_points", {}) or {}).get(arm_key)
            if not isinstance(item, dict) or "grip_board" not in item or "target_board" not in item:
                plan.ok = False
                plan.reason = f"D54 mask-core 20mm: {arm_key} point missing"
                report.update(applied=False, reason=f"{arm_key}_POINT_MISSING")
                return report
            grip = np.asarray(item["grip_board"], np.float32).reshape(2)
            target = np.asarray(item["target_board"], np.float32).reshape(2)
            inward_raw = item.get("physical_inward_board")
            inward = (
                np.asarray(inward_raw, np.float32).reshape(2)
                if inward_raw is not None else mask_center - grip
            )
            n = float(np.linalg.norm(inward))
            if n <= 1e-6:
                inward = mask_center - grip
                n = float(np.linalg.norm(inward))
            inward = inward / max(n, 1e-6)
            chosen = None
            for shift_mm in np.arange(0.0, 82.0, 2.0):
                q = grip + inward * float(shift_mm)
                qt = target + inward * float(shift_mm)
                if self.d54.select_arm_for_board_x(self.cfg54, float(q[0])) != arm_key:
                    continue
                if self.d54.select_arm_for_board_x(self.cfg54, float(qt[0])) != arm_key:
                    continue
                q_depth = depth_mm(q)
                if q_depth >= 20.0:
                    chosen = (q, qt, float(shift_mm), float(q_depth))
                    break
            if chosen is None:
                plan.ok = False
                plan.reason = f"D54 {arm_key} has no grasp in mask inner 20mm core"
                report["arms"][arm_key] = {"ok": False, "original_depth_mm": depth_mm(grip)}
                report.update(applied=False, reason=f"{arm_key}_NO_20MM_CORE")
                return report
            q, qt, shift_mm, q_depth = chosen
            item["grip_board"] = (float(q[0]), float(q[1]))
            item["target_board"] = (float(qt[0]), float(qt[1]))
            item["vla_mask_core_inset_mm"] = 20.0
            report["arms"][arm_key] = {
                "ok": True, "shift_mm": shift_mm, "final_depth_mm": q_depth,
            }
        report.update(applied=True, reason="OK")
        print("[54-3-MASK-CORE] both grasp points inside mask by >=20mm")
        return report

    def _pump_d55_i_preview(self, bundle: FrameBundle, obs: Any, heat: Any, index: int, count: int) -> None:
        """Publish D55 consensus progress without touching HighGUI from a worker thread."""
        try:
            preview = self.d55.draw_bottom_overlay_safe(
                bundle.corrected, self.H_corrected, obs, self.cfg55, None, heat,
                self.d55_args, False, ""
            )
            self.cv2.rectangle(preview, (0, 0), (preview.shape[1] - 1, 42), (0, 0, 0), -1)
            self.cv2.putText(
                preview, f"D55 I FRESH CONSENSUS {int(index)}/{int(count)}", (18, 29),
                self.cv2.FONT_HERSHEY_SIMPLEX, 0.72, (0, 255, 255), 2, self.cv2.LINE_AA
            )
            with self.state_lock:
                self.display_image = preview.copy()
                self.status = f"D55 INFERENCE {int(index)}/{int(count)} - CAMERA LOOP ALIVE"
        except Exception as exc:
            print(f"[D55-I-PREVIEW] publish skipped: {exc}")

    def _start_prepare_action(self) -> None:
        """Run every I-lock off the OpenCV/UI thread.

        Ported from the TOP-VLA pattern: the main loop keeps reading/displaying the
        camera while one frozen decision is processed in a worker.  Models remain
        persistent and no physical motion is sent from this worker.
        """
        with self.state_lock:
            if self.motion_busy:
                print("[I] blocked during motion")
                return
            if self.inference_busy:
                print(f"[I] {self.inference_action or 'inference'} already running")
                return
            action = self.selected_action

        if action is None:
            print("[I] select D50/D54/D55/D56/D58 first")
            return

        short = {
            "D50_BASKET_SWING_LAYDOWN": "D50",
            "D54_OUTER_PULL": "D54",
            "D55_PRESS_SWEEP": "D55",
            "D56_WAIST_LIFT_LAYDOWN": "D56",
            "D58_CIRC_POSITION": "D58",
        }.get(action, str(action))
        with self.state_lock:
            self.inference_busy = True
            self.inference_action = action
            self.status = f"{short} I-LOCK RUNNING - CAMERA LOOP ALIVE"
        print(f"[{short}-I] unified inference worker started; main camera/UI remains responsive")

        def infer_worker() -> None:
            t0 = time.monotonic()
            try:
                self._prepare_action()
            finally:
                total = time.monotonic() - t0
                with self.state_lock:
                    self.inference_busy = False
                    self.inference_action = None
                print(f"[VLA-I-LATENCY] action={short} worker_total={total:.3f}s")

        self.inference_worker = threading.Thread(
            target=infer_worker, name=f"bottom-vla-{short.lower()}-inference", daemon=True
        )
        self.inference_worker.start()

    def _prepare_action(self) -> None:
        with self.state_lock:
            if self.motion_busy:
                print("[I] blocked during motion")
                return
            if self.recorder.review_pending():
                print("[I] blocked: finish G/B/K and Y/N for the previous action")
                return
            action = self.selected_action
        perf_i0 = time.monotonic()
        perf_source_dt = perf_camera_dt = 0.0
        if not self.empty_baseline_ready:
            self.status = "I BLOCKED: EMPTY BOARD E REQUIRED"
            print("[I] blocked: remove garment, clear arms, and press E once on the empty board")
            return
        try:
            _perf_t = time.monotonic()
            self._verify_sources_unchanged()
            perf_source_dt = time.monotonic() - _perf_t
        except Exception as exc:
            self.status = "I BLOCKED: ORIGINAL SOURCE CHANGED"
            print(f"[I] {exc}")
            return
        if action is None:
            print("[I] select D54/D55/D56/D58 first")
            return
        state = None
        self.status = "CAMERA-CLEAR + FROZEN INFERENCE"
        # I is perception/plan lock only.  It may verify arm feedback but never
        # sends a robot command; every commanded motion remains Enter-only.
        _perf_t = time.monotonic()
        camera_clear_ok = self._ensure_camera_clear("BEFORE_I", allow_move=False)
        perf_camera_dt = time.monotonic() - _perf_t
        if not camera_clear_ok:
            self.status = "I BLOCKED: CAMERA-CLEAR NOT VERIFIED"
            print(f"[I] {self.status}")
            return

        try:
            if action == "D50_BASKET_SWING_LAYDOWN":
                bundle = self._capture_i_frame_from_live(action)
                plan = SimpleNamespace(
                    ok=True, reason="ARM2 basket grasp + initial V24 direct center air drop locked; Enter executes once",
                    action="D50_BASKET_SWING_LAYDOWN", metrics={
                        "executor": "50-1.py --auto-run-once",
                        "arm_usage": "ARM2_ONLY", "i_sends_motion": False,
                    }, arm_points={},
                )
                canvas = self._banner(bundle.raw.copy(), state, action, True, plan.reason)
                locked = LockedPlan(
                    state, action, bundle, None, plan, None, canvas,
                    self.H_raw.copy(), time.time(), True, plan.reason, None,
                    {"d50_source_sha256": self.source_sha256.get(action), "camera_geometry_path": "RAW"},
                )
                obs_record = self.recorder.save_observation(locked, "BEFORE_ACTION", self._environment_metadata())
                locked.observation_record = obs_record
                self.recorder.save_decision(locked, obs_record)
                with self.state_lock:
                    self.locked = locked
                    self.display_image = canvas.copy()
                    self.status = "D50 I-LOCKED: ENTER RUNS ARM2 BASKET + DIRECT AIR DROP"
                print("[D50-I-LOCK] robot motion=NONE; Enter runs 50-1.py")
                return
            if action == "D58_CIRC_POSITION":
                _t = time.monotonic(); bundle = self._capture_i_frame_from_live(action); capture_dt = time.monotonic()-_t
                _t = time.monotonic(); obs = self._infer_for_action(action, bundle.corrected); infer_dt = time.monotonic()-_t
                mask = getattr(obs, "mask", None)
                pose = getattr(obs, "pose", None)
                _t = time.monotonic()
                if mask is None:
                    plan = self.d58.D58Plan(False, "D58 mask unavailable")
                else:
                    plan = self.d58.build_d58_plan(
                        bundle.corrected, mask, pose, self.H_corrected, self.config, self.d58_args
                    )
                    plan = self._main33_strengthen_d58_plan(bundle.corrected, mask, pose, plan)
                plan_dt = time.monotonic()-_t
                _t = time.monotonic(); canvas = self._operator_overlay(bundle, action, obs, plan, None); overlay_dt = time.monotonic()-_t
                plan_ok = bool(getattr(plan, "ok", False))
                reason = str(getattr(plan, "reason", ""))
                diagnostics = {
                    "d58_target_source": str(getattr(plan, "target_source", "NONE")),
                    "d58_move_mm": float(getattr(plan, "move_mm", 0.0) or 0.0),
                    "d58_selected_arm": str(getattr(plan, "selected_arm", "")),
                    "original_source_sha256": self.source_sha256.get(action),
                    "camera_geometry_path": "CORRECTED+H",
                }
                locked = LockedPlan(
                    state, action, bundle, obs, plan, None, canvas, self.H_corrected.copy(),
                    time.time(), plan_ok, reason, None if plan_ok else "NO_SAFE_PLAN", diagnostics,
                )
                _t = time.monotonic(); env58 = self._environment_metadata(); env_dt = time.monotonic()-_t
                _t = time.monotonic(); obs_record = self.recorder.save_observation(locked, "BEFORE_ACTION", env58); save_dt = time.monotonic()-_t
                locked.observation_record = obs_record
                _t = time.monotonic(); self.recorder.save_decision(locked, obs_record); decision_dt = time.monotonic()-_t
                with self.state_lock:
                    self.locked = locked
                    self.display_image = canvas.copy()
                    self.status = "D58 I-LOCKED: VERIFY CIRC/GRIP/END, THEN ENTER" if plan_ok else f"NO_PLAN: {reason}"
                print(f"[I-LOCK] action={action} ok={plan_ok} reason={reason}")
                print(
                    f"[PERF-I-D58] source={perf_source_dt:.3f}s camera={perf_camera_dt:.3f}s "
                    f"capture={capture_dt:.3f}s infer={infer_dt:.3f}s plan={plan_dt:.3f}s "
                    f"overlay={overlay_dt:.3f}s env={env_dt:.3f}s save={save_dt:.3f}s decision={decision_dt:.3f}s "
                    f"TOTAL={(time.monotonic()-perf_i0):.3f}s"
                )
                return
            samples: List[Tuple[FrameBundle, Any, Any]] = []
            if action == "D55_PRESS_SWEEP":
                count = int(self.args.d55_consensus_frames)
            elif action == "D56_WAIST_LIFT_LAYDOWN":
                count = int(D56_TEMPORAL_FRAMES)
            else:
                count = 1
            d56_temporal_rows: List[Dict[str, Any]] = []
            for index in range(count):
                _sample_capture_t = time.monotonic()
                bundle = self._capture_i_frame_from_live(action)
                _sample_capture_dt = time.monotonic() - _sample_capture_t
                _sample_infer_t = time.monotonic()
                obs = self._infer_for_action(action, bundle.corrected)
                _sample_infer_dt = time.monotonic() - _sample_infer_t
                if action == "D56_WAIST_LIFT_LAYDOWN":
                    print(f"[PERF-I-D56] capture={_sample_capture_dt:.3f}s infer={_sample_infer_dt:.3f}s")
                heat = None
                if action in {"D54_OUTER_PULL", "D55_PRESS_SWEEP"} and getattr(obs, "mask", None) is not None:
                    heat_module = self.d54 if action == "D54_OUTER_PULL" else self.d55
                    heat_args = self.d54_args if action == "D54_OUTER_PULL" else self.d55_args
                    heat = heat_module.build_d21_wrinkle_heatmap(
                        bundle.corrected, obs, self.H_corrected, heat_args
                    )
                if action == "D56_WAIST_LIFT_LAYDOWN" and obs is not None and getattr(obs, "mask", None) is not None:
                    try:
                        _observer_t = time.monotonic()
                        report56 = self.d56._d56v7_build_waist_observer(
                            bundle.corrected, obs, self.H_raw, self.d56_args
                        )
                        print(f"[PERF-I-D56] waist_observer={(time.monotonic()-_observer_t):.3f}s")
                        obs.d56v7_waist_observer = report56
                        selected56 = report56.get("selected") if isinstance(report56, dict) else None
                        d56_temporal_rows.append({
                            "index": int(index),
                            "selected": bool(isinstance(selected56, dict)),
                            "score": float((selected56 or {}).get("score", 0.0)),
                            "rank": float((selected56 or {}).get("d56v18_rank_score", (selected56 or {}).get("score", 0.0))),
                            "source": ("POSE_MASK_ESTIMATED" if bool((selected56 or {}).get("d56v55_pose_mask_estimated", False)) else ("RIBBON" if isinstance(selected56, dict) else "NONE")),
                            "reason": str((report56 or {}).get("reason", "")),
                        })
                    except Exception as exc:
                        d56_temporal_rows.append({"index": int(index), "selected": False, "score": 0.0, "rank": 0.0, "source": "ERROR", "reason": repr(exc)})
                samples.append((bundle, obs, heat))
                if action == "D55_PRESS_SWEEP":
                    self._pump_d55_i_preview(bundle, obs, heat, index + 1, count)
                if index + 1 < count:
                    if action == "D55_PRESS_SWEEP":
                        time.sleep(float(self.args.d55_consensus_interval_s))
                    elif action == "D56_WAIST_LIFT_LAYDOWN":
                        time.sleep(float(D56_TEMPORAL_INTERVAL_S))

            diagnostics: Dict[str, Any] = {}
            heat = None
            d56_best_index: Optional[int] = None
            if action == "D56_WAIST_LIFT_LAYDOWN" and samples:
                usable = [r for r in d56_temporal_rows if bool(r.get("selected", False))]
                if usable:
                    # Prefer a real ribbon over Pose+Mask estimate; within the
                    # same source use the highest temporal/rank score.
                    usable.sort(key=lambda r: (1 if r.get("source") == "RIBBON" else 0, float(r.get("rank", 0.0)), float(r.get("score", 0.0))), reverse=True)
                    d56_best_index = int(usable[0]["index"])
                else:
                    d56_best_index = len(samples) - 1
                diagnostics["d56_temporal_fusion"] = {
                    "frames": int(len(samples)),
                    "selected_index": int(d56_best_index),
                    "rows": _json_safe(d56_temporal_rows),
                }
                print(
                    f"[D56-SINGLE-FRAME] source="
                    f"{(d56_temporal_rows[d56_best_index].get('source') if d56_best_index < len(d56_temporal_rows) else 'NONE')}"
                )
            if action == "D55_PRESS_SWEEP":
                # Exact D55-v13 sample schema: frame + observation + heatmap.
                raw_samples = [
                    {"frame": x[0].corrected, "obs": x[1], "heat": x[2]}
                    for x in samples
                ]
                rep_index, heat, consensus = self.d55._d55v2_filter_persistent_samples(raw_samples, self.d55_args)
                rep_index = int(np.clip(rep_index, 0, len(samples) - 1))
                bundle, obs, _ = samples[rep_index]
                plan = self.d55.build_d22_hybrid_wrinkle_plan(
                    obs, heat, self.H_corrected, self.config, self.cfg55, self.d55_args
                )
                if plan is None:
                    valid, validation = False, {"reason": "NO_PLAN"}
                    plan = self.d55.DualWrinkleStretchPlan(False, "D55 original planner returned NO_PLAN")
                else:
                    d55_arm_count = len(dict(getattr(plan, "arm_points", {}) or {}))
                    if bool(getattr(plan, "ok", False)) and d55_arm_count != 2:
                        valid = False
                        validation = {
                            "reason": "D55_DUAL_ARMS_REQUIRED",
                            "arm_count": int(d55_arm_count),
                            "assist_enabled": True,
                        }
                        plan.ok = False
                        plan.reason = (
                            "D55 dual-arm required: no safe compatible second "
                            "wrinkle/assist plan"
                        )
                        print(
                            f"[D55-DUAL-ONLY] blocked single-arm plan "
                            f"action={getattr(plan, 'action', 'UNKNOWN')}"
                        )
                    else:
                        valid, validation = self.d55._d55v8_validate_plan(
                            plan, obs, self.d55_args
                        )
                    if not valid and bool(getattr(plan, "ok", False)):
                        plan.ok = False
                        plan.reason = f"D55 validation blocked: {validation.get('reason')}"
                diagnostics.update(d55_consensus=consensus, d55_validation=validation)
                canvas = self.d55.draw_bottom_overlay_safe(
                    bundle.corrected, self.H_corrected, obs, self.cfg55, None, heat, self.d55_args, False, ""
                )
                canvas = self.d55.draw_dual_wrinkle_plan_overlay(canvas, self.H_corrected, plan)
            else:
                if action == "D56_WAIST_LIFT_LAYDOWN" and d56_best_index is not None:
                    bundle, obs, heat = samples[int(d56_best_index)]
                else:
                    bundle, obs, heat = samples[-1]
                if action == "D54_OUTER_PULL":
                    occ_obs, occ_report = self.d54._d47_build_occupancy_observation(
                        bundle.corrected, obs, self.H_corrected, self.d54_args,
                        seg_model=self.seg_model, pose_model=self.pose_model, cfg=self.cfg54,
                    )
                    diagnostics["d54_occupancy"] = {
                        k: v for k, v in dict(occ_report or {}).items() if not str(k).startswith("_")
                    }
                    diagonal_obs = (
                        occ_obs if occ_obs is not None and getattr(occ_obs, "mask", None) is not None
                        else obs
                    )
                    gross = dict(getattr(obs, "d45_gross_mask_validation", {}) or {})
                    if bool(gross.get("rejected", False)):
                        plan = self.d54.DualWrinkleStretchPlan(
                            False, f"D54 original gross-mask veto: {gross.get('reason', 'unsafe mask')}"
                        )
                    else:
                        plan = self.d54._d51v4_build_diagonal_pull_plan(
                            diagonal_obs, self.H_corrected, self.config,
                            self.cfg54, self.d54_args, stage=0,
                        )
                    if plan is None:
                        plan = self.d54.DualWrinkleStretchPlan(False, "D54 original planner returned NO_PLAN")
                    diagnostics["d54_mask_core_20mm"] = self._enforce_d54_mask_core_20mm(
                        plan, diagonal_obs
                    )
                    diagnostics["d54_extra_grip_inset"] = self._apply_d54_extra_grip_inset(plan)
                    # Match the original manual-I behavior: display/lock the
                    # planner result.  The original executor owns preflight at
                    # Enter; the collector must not add an earlier veto.
                    diagnostics["d54_preflight"] = {"deferred_to_original_executor": True}
                    canvas = self._draw_arm_plan(
                        self._base_overlay(action, bundle.corrected, obs), plan,
                        self.d54, self.cfg54, action, self.H_corrected,
                    )
                elif action == "D56_WAIST_LIFT_LAYDOWN":
                    # D56-45 manual-I order: E49/E62 -> D47 occupancy rescue ->
                    # D42 waist plan -> optional D47 pre-spread selection.
                    _d56_occ_t = time.monotonic()
                    occ_obs, occ_report = self.d56._d47_build_occupancy_observation(
                        bundle.corrected, obs, self.H_raw, self.d56_args,
                        seg_model=self.seg_model, pose_model=self.pose_model, cfg=self.cfg56,
                    )
                    print(f"[PERF-I-D56] occupancy={(time.monotonic()-_d56_occ_t):.3f}s")
                    diagnostics["d56_occupancy"] = {
                        k: v for k, v in dict(occ_report or {}).items() if not str(k).startswith("_")
                    }
                    planner_obs = obs
                    gross = dict(getattr(obs, "d45_gross_mask_validation", {}) or {})
                    if (
                        obs is not None and getattr(obs, "mask", None) is not None
                        and bool(gross.get("rejected", False))
                    ):
                        planner_obs = copy.copy(obs)
                        planner_obs.mask = None
                        planner_obs.valid = False
                        planner_obs.reason = (
                            "D45 gross board-mask veto: "
                            + str(gross.get("reason", "unsafe mask"))
                        )
                        diagnostics["d56_gross_mask_veto"] = gross
                    plan = None
                    if planner_obs is not None and getattr(planner_obs, "mask", None) is not None:
                        _d56_plan_t = time.monotonic()
                        plan = self.d56._d42_build_hybrid_grasp_plan(
                            planner_obs, self.H_raw, self.config, self.cfg56, self.d56_args
                        )
                        plan = self.d56._d56_apply_arm1_waistward_correction(
                            plan, planner_obs, self.H_raw, self.config, self.cfg56, self.d56_args
                        )
                        print(f"[PERF-I-D56] waist_plan={(time.monotonic()-_d56_plan_t):.3f}s")
                    initial_source = str(
                        getattr(planner_obs, "d38_segmentation_source", "")
                        if planner_obs is not None else ""
                    )
                    initial_ribbon = (
                        getattr(planner_obs, "d56v7_waist_observer", None)
                        if planner_obs is not None else None
                    )
                    initial_selected = (
                        initial_ribbon.get("selected")
                        if isinstance(initial_ribbon, dict) else None
                    )
                    if initial_source == "FALLBACK_BGDIFF" and not isinstance(initial_selected, dict):
                        retry_count = max(0, int(getattr(self.d56_args, "d56v24_bgdiff_waist_retry_count", 2)))
                        retry_delay = max(0.05, float(getattr(self.d56_args, "d56v24_bgdiff_waist_retry_delay_s", 0.18)))
                        print(f"[D56-24-BGDIFF-REJUDGE] trigger: retries={retry_count}")
                        for retry_no in range(1, retry_count + 1):
                            time.sleep(retry_delay)
                            retry_bundle = self._capture_i_frame_from_live(action)
                            retry_obs = self._infer_for_action(action, retry_bundle.corrected)
                            retry_occ_obs, retry_occ_report = self.d56._d47_build_occupancy_observation(
                                retry_bundle.corrected, retry_obs, self.H_raw, self.d56_args,
                                seg_model=self.seg_model, pose_model=self.pose_model, cfg=self.cfg56,
                            )
                            retry_planner_obs = retry_obs
                            retry_gross = dict(getattr(retry_obs, "d45_gross_mask_validation", {}) or {})
                            if (
                                retry_obs is not None and getattr(retry_obs, "mask", None) is not None
                                and bool(retry_gross.get("rejected", False))
                            ):
                                retry_planner_obs = copy.copy(retry_obs)
                                retry_planner_obs.mask = None
                                retry_planner_obs.valid = False
                                retry_planner_obs.reason = (
                                    "D45 gross board-mask veto: "
                                    + str(retry_gross.get("reason", "unsafe mask"))
                                )
                            retry_plan = None
                            if retry_planner_obs is not None and getattr(retry_planner_obs, "mask", None) is not None:
                                retry_plan = self.d56._d42_build_hybrid_grasp_plan(
                                    retry_planner_obs, self.H_raw, self.config, self.cfg56, self.d56_args
                                )
                                retry_plan = self.d56._d56_apply_arm1_waistward_correction(
                                    retry_plan, retry_planner_obs, self.H_raw,
                                    self.config, self.cfg56, self.d56_args,
                                )
                            retry_source = str(
                                getattr(retry_planner_obs, "d38_segmentation_source", "")
                                if retry_planner_obs is not None else ""
                            )
                            retry_ribbon = (
                                getattr(retry_planner_obs, "d56v7_waist_observer", None)
                                if retry_planner_obs is not None else None
                            )
                            retry_has_ribbon = bool(
                                isinstance(retry_ribbon, dict)
                                and isinstance(retry_ribbon.get("selected"), dict)
                            )
                            print(
                                f"[D56-24-BGDIFF-REJUDGE] {retry_no}/{retry_count} "
                                f"source={retry_source or 'NONE'} ribbon={retry_has_ribbon}"
                            )
                            if retry_source == "YOLO" or retry_has_ribbon:
                                bundle = retry_bundle
                                obs = retry_obs
                                planner_obs = retry_planner_obs
                                plan = retry_plan
                                occ_obs = retry_occ_obs
                                occ_report = retry_occ_report
                                diagnostics["d56_bgdiff_retry_accepted"] = retry_no
                                break
                    diagnostics["d56_occupancy"] = {
                        k: v for k, v in dict(occ_report or {}).items() if not str(k).startswith("_")
                    }
                    if plan is None:
                        plan = self.d56.D31DualGraspPlan(False, "D56-45 original planner returned NO_PLAN")
                    d47_report = self.d56._d47_analyze_global_prespread(
                        bundle.corrected, occ_obs, self.H_raw, self.d56_args,
                        base_plan=plan, occupancy_report=occ_report,
                    )
                    diagnostics["d56_d47"] = self.d56._d47_public_crumple_report(d47_report)
                    if bool(d47_report.get("available", False) and d47_report.get("trigger", False)):
                        attempt = int(self.d56_prespread_attempt_count) + 1
                        max_attempts = max(0, int(getattr(self.d56_args, "d47_prespread_max_attempts", 2)))
                        if attempt <= max_attempts:
                            press_plan = self.d56._d47_build_global_prespread_plan(
                                occ_obs, self.H_raw, self.config, self.cfg56,
                                self.d56_args, d47_report, attempt,
                            )
                            if bool(getattr(press_plan, "ok", False)):
                                plan = press_plan
                                planner_obs = occ_obs
                                diagnostics["d56_selected_executor"] = "D47_OCCUPANCY_PRESPREAD"
                    diagnostics["d56_original_plan_mode"] = str(
                        (getattr(plan, "metrics", {}) or {}).get("d42_plan_mode", "")
                    )
                    if bool(getattr(plan, "ok", False)) and not str(getattr(plan, "action", "")).startswith("D47_"):
                        _taught_t = time.monotonic()
                        taught_spec = self._main28_build_d56_taught_spec(plan)
                        diagnostics["main28_d56_taught_laydown"] = _json_safe(taught_spec)
                        if bool(taught_spec.get("ok", False)):
                            plan.metrics["main28_taught_laydown"] = taught_spec
                            print(
                                f"[D56-MAIN33-I-LOCK] real FIX55/FIX56 2D PASS "
                                f"scale={float(taught_spec['trajectory_scale']):.2f} "
                                f"rot={float(taught_spec['rotation_deg']):+.1f}deg "
                                f"span={float(taught_spec['forward_span_mm']):.0f}mm "
                                f"duration={float(taught_spec['duration_s']):.2f}s"
                            )
                        else:
                            # No semantic NO_ACTION conversion. Keep the D56 plan alive;
                            # execution will report only a genuine physical representation failure.
                            plan.metrics["main28_taught_laydown"] = taught_spec
                            print(f"[D56-MAIN33-PREFLIGHT-WARN] {taught_spec.get('reason')} recent={taught_spec.get('recent_rejects', [])}")
                        print(f"[PERF-I-D56] taught_preflight={(time.monotonic()-_taught_t):.3f}s")
                    obs = planner_obs
                    canvas = self._base_overlay(action, bundle.corrected, obs)
                    canvas = self._draw_arm_plan(
                        canvas, plan, self.d56, self.cfg56, action, self.H_raw
                    )
                    _taught_overlay_spec = dict((getattr(plan, "metrics", {}) or {}).get("main28_taught_laydown", {}) or {})
                    if bool(_taught_overlay_spec.get("ok", False)):
                        canvas = self._main28_draw_d56_taught_overlay(canvas, _taught_overlay_spec)
                    diagnostics["d56_preflight"] = {"main28_taught_laydown_frozen_at_I": True}
            plan_ok = bool(getattr(plan, "ok", False))
            reason = str(getattr(plan, "reason", ""))
            metrics = dict(getattr(plan, "metrics", {}) or {})
            planner_failure = metrics.get("planner_failure")
            if not plan_ok and planner_failure is None:
                if action == "D56_WAIST_LIFT_LAYDOWN":
                    flags = diagnostics.get("waist_failure_flags", [])
                    planner_failure = flags[0] if flags else "WAIST_MISSED"
                elif action != "NONE":
                    planner_failure = "NO_SAFE_PLAN"
            canvas = self._operator_overlay(bundle, action, obs, plan, heat)
            if action == "D54_OUTER_PULL":
                pm = dict(getattr(plan, "metrics", {}) or {})
                desired54 = float(pm.get("d54_desired_pull_mm", pm.get("desired_pull", 0.0)) or 0.0)
                actual54 = float(pm.get("d54_common_safe_pull_mm", pm.get("common_safe_pull", pm.get("common_pull", 0.0))) or 0.0)
                self.cv2.putText(
                    canvas, f"D54 PULL desired={desired54:.0f}mm actual={actual54:.0f}mm",
                    (18, 112), self.cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 255, 255), 2, self.cv2.LINE_AA
                )
                diagnostics["d54_pull_display"] = {"desired_mm": desired54, "actual_mm": actual54}
            # D54/D55 already include their original top status panels and
            # bottom next-action bar.  Do not cover those layers with V7's
            # simplified merged banner.
            if action == "D56_WAIST_LIFT_LAYDOWN":
                canvas = self._banner(canvas, state, action, plan_ok, reason)
            elif action == "D55_PRESS_SWEEP":
                self.cv2.putText(
                    canvas, f"D55 FRESH I-LOCK #{int(self.inference_serial)} {time.strftime('%H:%M:%S')}",
                    (18, 112), self.cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 255, 255), 2, self.cv2.LINE_AA
                )
            action_h = self._H_for_action(action)
            diagnostics["original_source_sha256"] = self.source_sha256.get(action)
            diagnostics["camera_geometry_path"] = (
                "RAW+raw_H" if action == "D56_WAIST_LIFT_LAYDOWN" else "CORRECTED+H"
            )
            diagnostics["operator_display_geometry"] = (
                "RAW+raw_H" if action == "D56_WAIST_LIFT_LAYDOWN" else "CORRECTED+H"
            )
            locked = LockedPlan(
                state, action, bundle, obs, plan, heat, canvas, action_h.copy(), time.time(),
                plan_ok, reason, planner_failure, diagnostics,
            )
            environment = self._environment_metadata()
            obs_record = self.recorder.save_observation(locked, "BEFORE_ACTION", environment)
            locked.observation_record = obs_record
            self.recorder.save_decision(locked, obs_record)
            with self.state_lock:
                self.locked = locked
                self.display_image = canvas.copy()
                self.status = "I-LOCKED: VERIFY OVERLAY, THEN ENTER" if plan_ok else f"NO_PLAN: {planner_failure}"
            print(f"[I-LOCK] action={action} ok={plan_ok} failure={planner_failure}")
            if action == "D56_WAIST_LIFT_LAYDOWN":
                print(
                    f"[PERF-I-D56] source={perf_source_dt:.3f}s camera={perf_camera_dt:.3f}s "
                    f"TOTAL={(time.monotonic()-perf_i0):.3f}s (see PERF-IO for save breakdown)"
                )
        except Exception as exc:
            self.status = f"I ERROR: {type(exc).__name__}"
            print(f"[I-ERROR] {exc!r}")

    def _environment_metadata(self) -> Dict[str, Any]:
        payload = {
            "config_sha256": _sha256_file(self.args.config),
            "homography_sha256": _sha256_file(self.args.hfile),
            "camera_geometry": self.camera_geometry_meta,
            "camera_controls": self.camera_controls_meta,
            "models": {
                "segmentation_sha256": _sha256_file(self.args.seg_model),
                "pose_sha256": _sha256_file(self.args.pose_model),
            },
        }
        payload["environment_fingerprint"] = hashlib.sha256(
            json.dumps(_json_safe(payload), sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        return payload

    @staticmethod
    def _main28_ease_linear_ends(u: float, edge: float = 0.05) -> float:
        """Linear-middle progress with cubic zero-slope easing only at both ends."""
        u = float(np.clip(float(u), 0.0, 1.0))
        f = float(np.clip(float(edge), 1e-4, 0.20))
        if u < f:
            r = u / f
            return float(f * (2.0 * r * r - r * r * r))
        if u > 1.0 - f:
            r = (1.0 - u) / f
            return float(1.0 - f * (2.0 * r * r - r * r * r))
        return u

    def _main33_local_xy_to_board(self, arm_key: str, local_xy: np.ndarray) -> np.ndarray:
        """Invert the authoritative board->RoArm affine for one XY point."""
        arm_cfg = self.config.get("dual_roarm", {}).get(arm_key, {})
        M = np.asarray(arm_cfg.get("board_to_roarm_affine_2x3"), np.float64).reshape(2, 3)
        A = M[:, :2]
        off = arm_cfg.get("roarm_xy_offset", [0.0, 0.0])
        ox = float(off[0]) if isinstance(off, (list, tuple)) and len(off) >= 1 else 0.0
        oy = float(off[1]) if isinstance(off, (list, tuple)) and len(off) >= 2 else 0.0
        q = np.asarray(local_xy, np.float64).reshape(2) - np.asarray([ox, oy], np.float64)
        out = np.linalg.solve(A, q - M[:, 2])
        if not np.all(np.isfinite(out)):
            raise RuntimeError(f"{arm_key} inverse affine produced nonfinite board XY")
        return out.astype(np.float64)

    def _main33_source_fix55_displacements(self) -> List[Dict[str, Any]]:
        """Rebuild the teammate FIX55 master exactly from the original 2-D teaching."""
        start_local = {
            k: np.asarray(MAIN33_D56_CANONICAL_LOCAL[k][:2], np.float64)
            for k in ("arm2", "arm1")
        }
        start_board = {k: self._main33_local_xy_to_board(k, start_local[k]) for k in start_local}
        samples: List[Dict[str, Any]] = []
        for rec in MAIN33_D56_SOURCE_TRAJECTORY:
            t_rel, a1dx, a1dy, a1zf, a2dx, a2dy, a2zf = rec
            src1 = start_local["arm1"] + np.asarray([a1dx, a1dy], np.float64)
            src2 = start_local["arm2"] + np.asarray([a2dx, a2dy], np.float64)
            b1 = self._main33_local_xy_to_board("arm1", src1)
            b2 = self._main33_local_xy_to_board("arm2", src2)
            d1 = b1 - start_board["arm1"]
            d2 = b2 - start_board["arm2"]
            d1_as_arm2 = np.asarray([-float(d1[0]), float(d1[1])], np.float64)
            master = 0.5 * (d2 + d1_as_arm2)
            samples.append({
                "t_src": float(t_rel),
                "master": np.asarray(master, np.float64),
                "arm2_src": np.asarray(master, np.float64),
                "arm1_src": np.asarray([-float(master[0]), float(master[1])], np.float64),
                "source_zf": 0.5 * (float(a1zf) + float(a2zf)),
            })
        return samples

    def _main28_build_d56_taught_spec(self, plan: Any) -> Dict[str, Any]:
        """main-33: current D56 grasp + actual teammate FIX55/FIX56 2-D master."""
        if not bool(getattr(plan, "ok", False)):
            return {"ok": False, "reason": "D56_PLAN_NOT_OK"}
        points = dict(getattr(plan, "arm_points", {}) or {})
        if not all(k in points and points[k].get("grip_board") is not None for k in ("arm2", "arm1")):
            return {"ok": False, "reason": "D56_GRIP_PAIR_MISSING"}
        grips = {k: np.asarray(points[k]["grip_board"], np.float64).reshape(2) for k in ("arm2", "arm1")}

        # Preserve D56-45 semantics, but use the full 2-D body vector instead of only its Y sign.
        metrics = dict(getattr(plan, "metrics", {}) or {})
        body = metrics.get("d42_body_normal_board")
        try:
            body = np.asarray(body, np.float64).reshape(2)
            bn = float(np.linalg.norm(body))
            if not np.isfinite(bn) or bn < 1e-6:
                raise ValueError("invalid body vector")
            target_dir = body / bn
        except Exception:
            waist_y = 0.5 * (float(grips["arm2"][1]) + float(grips["arm1"][1]))
            board_mid_y = 0.5 * (float(self.cfg56.board_y_min) + float(self.cfg56.board_y_max))
            target_dir = np.asarray([0.0, -1.0 if waist_y >= board_mid_y else 1.0], np.float64)

        source = self._main33_source_fix55_displacements()
        if len(source) < 8:
            return {"ok": False, "reason": "MAIN33_SOURCE_TRAJECTORY_TOO_SHORT"}
        source_final = np.asarray(source[-1]["master"], np.float64)
        sn = float(np.linalg.norm(source_final))
        if sn < 1e-6:
            return {"ok": False, "reason": "MAIN33_SOURCE_FINAL_VECTOR_ZERO"}
        sdir = source_final / sn
        cross = float(sdir[0] * target_dir[1] - sdir[1] * target_dir[0])
        dot = float(np.clip(np.dot(sdir, target_dir), -1.0, 1.0))
        theta = math.atan2(cross, dot)
        c, s = math.cos(theta), math.sin(theta)
        R = np.asarray([[c, -s], [s, c]], np.float64)

        source_duration = max(1e-6, float(source[-1]["t_src"]))
        duration = max(2.4, float(getattr(self.args, "main33_d56_duration_s", MAIN33_D56_TARGET_DURATION_S)))
        margin = max(10.0, float(getattr(self.d56_args, "d50_final_board_margin_mm", 18.0)))
        radius_limit = min(
            float(getattr(self.d56_args, "d31_roarm_xy_radius_max_mm", 420.0)),
            float(getattr(self.d56_args, "d50_roarm_xy_radius_max_mm", 390.0)),
        )
        left_max = float(self.cfg56.split_board_x - self.cfg56.center_dead_half_width)
        right_min = float(self.cfg56.split_board_x + self.cfg56.center_dead_half_width)

        # Detect the exact FIX56 longitudinal reversal from the rotated master trajectory.
        rotated_master = [R @ np.asarray(sample["master"], np.float64) for sample in source]
        final_vec = np.asarray(rotated_master[-1], np.float64)
        final_axis = final_vec / max(1e-9, float(np.linalg.norm(final_vec)))
        longitudinal = np.asarray([float(np.dot(v, final_axis)) for v in rotated_master], np.float64)
        final_direction = 1.0 if float(longitudinal[-1]) >= float(longitudinal[0]) else -1.0
        reversal_idx = int(np.argmin(longitudinal) if final_direction > 0.0 else np.argmax(longitudinal))
        reversal_q = float(longitudinal[reversal_idx])
        forward_span = float(final_direction * (float(longitudinal[-1]) - reversal_q))
        reversal_zf = float(np.clip(float(source[reversal_idx]["source_zf"]), -0.03, 1.0))
        use_escalator = bool(0 < reversal_idx < len(source)-2 and forward_span >= 120.0 and reversal_zf < 0.98)

        def esc_progress(u: float, e: float = 0.05) -> float:
            u = float(np.clip(u, 0.0, 1.0)); e = float(np.clip(e, 0.0, 0.25))
            if e <= 1e-9: return u
            area_total = 1.0 - e
            if u < e: area = 0.5 * u * u / e
            elif u <= 1.0-e: area = 0.5*e + (u-e)
            else:
                tail = 1.0-u; area = area_total - 0.5*tail*tail/e
            return float(np.clip(area / area_total, 0.0, 1.0))

        def build_rows(scale: float) -> Tuple[Optional[List[Dict[str, Any]]], str]:
            rows: List[Dict[str, Any]] = []
            monotonic_u = 0.0
            for idx, sample in enumerate(source):
                d2 = R @ np.asarray(sample["arm2_src"], np.float64) * float(scale)
                d1 = R @ np.asarray(sample["arm1_src"], np.float64) * float(scale)
                b2 = grips["arm2"] + d2
                b1 = grips["arm1"] + d1
                for k, bp in (("arm2", b2), ("arm1", b1)):
                    if not self.d56.point_in_board(self.cfg56, bp, margin=margin):
                        return None, f"{k}_BOARD_MARGIN@{idx}"
                    if k == "arm2" and float(bp[0]) > left_max + 1e-6:
                        return None, f"ARM2_WORKSPACE@{idx}:{float(bp[0]):.1f}>{left_max:.1f}"
                    if k == "arm1" and float(bp[0]) < right_min - 1e-6:
                        return None, f"ARM1_WORKSPACE@{idx}:{float(bp[0]):.1f}<{right_min:.1f}"
                    rx, ry = self.d56.board_to_arm_xy(self.config, k, float(bp[0]), float(bp[1]))
                    rr = float(math.hypot(rx, ry))
                    if rr > radius_limit + 1e-6:
                        return None, f"{k}_REACH@{idx}:{rr:.1f}>{radius_limit:.1f}"
                src_zf = float(sample["source_zf"])
                if use_escalator and idx > reversal_idx:
                    q = float(np.dot(rotated_master[idx] * float(scale), final_axis))
                    rq = reversal_q * float(scale)
                    fs = max(1e-9, forward_span * float(scale))
                    raw_u = float(np.clip(final_direction * (q-rq) / fs, 0.0, 1.0))
                    monotonic_u = max(monotonic_u, raw_u)
                    zf = reversal_zf + esc_progress(monotonic_u) * (1.0-reversal_zf)
                elif use_escalator and idx == reversal_idx:
                    monotonic_u = 0.0; zf = reversal_zf
                else:
                    zf = src_zf
                rows.append({
                    "t": float(sample["t_src"]) / source_duration * duration,
                    "zf": float(np.clip(zf, -0.03, 1.0)),
                    "arm2_board": b2.astype(float).tolist(),
                    "arm1_board": b1.astype(float).tolist(),
                    "master_disp": (R @ np.asarray(sample["master"], np.float64) * float(scale)).astype(float).tolist(),
                })
            return rows, "OK"

        selected_rows = None; selected_scale = None; rejects: List[str] = []
        scale = 1.0
        while scale >= MAIN33_D56_SOURCE_MIN_SCALE - 1e-9:
            rows, why = build_rows(scale)
            if rows is not None:
                selected_rows = rows; selected_scale = float(scale); break
            rejects.append(f"S{scale:.2f}:{why}")
            scale -= 0.05
        # Do not semantic-block the action. If the full teaching cannot fit, use the
        # largest conservative partial replay that still preserves the actual 2-D source shape.
        if selected_rows is None:
            scale = 0.30
            while scale >= 0.10 - 1e-9:
                rows, why = build_rows(scale)
                if rows is not None:
                    selected_rows = rows; selected_scale = float(scale); break
                rejects.append(f"S{scale:.2f}:{why}")
                scale -= 0.05
        if selected_rows is None:
            # Physical coordinate conversion itself is impossible; this is not a semantic/no-action gate.
            return {"ok": False, "reason": "MAIN33_D56_PHYSICAL_PATH_UNREPRESENTABLE", "recent_rejects": rejects[-10:]}

        contact_ref = {k: float(self.d56.arm_contact_z(self.cfg56, k)) for k in ("arm2", "arm1")}
        contact_cmd = dict(contact_ref)
        contact_cmd["arm2"] -= max(0.0, float(points["arm2"].get("d56v25_contact_z_lower_mm", 5.0) or 0.0))
        # FIX59: first prove the cloth is held with a pure +18 mm vertical test lift.
        test_lift_z = {k: float(contact_cmd[k] + 18.0) for k in ("arm2", "arm1")}
        # Use the source FIX49 canonical Zs as the aerial start, clipped only by D56's hard Z limit.
        max_z = float(getattr(self.d56_args, "max_z", 420.0))
        aerial_z = {
            "arm2": min(float(MAIN33_D56_CANONICAL_LOCAL["arm2"][2]), max_z - 10.0),
            "arm1": min(float(MAIN33_D56_CANONICAL_LOCAL["arm1"][2]), max_z - 10.0),
        }
        final_z = {k: float(contact_ref[k] + MAIN33_D56_FINAL_CLEARANCE_MM) for k in ("arm2", "arm1")}
        return {
            "ok": True,
            "version": "MAIN33_D56_REAL_FIX55_FIX56_2D_V1",
            "source_sha256": MAIN33_D56_SOURCE_SHA256,
            "target_dir_board": target_dir.astype(float).tolist(),
            "source_final_dir_board": sdir.astype(float).tolist(),
            "rotation_deg": float(math.degrees(theta)),
            "trajectory_scale": float(selected_scale),
            "duration_s": float(duration),
            "t104_speed": float(MAIN33_D56_T104_SPEED),
            "reversal_index": int(reversal_idx),
            "forward_span_mm": float(forward_span * selected_scale),
            "grip_board": {k: grips[k].astype(float).tolist() for k in grips},
            "contact_ref_z": contact_ref,
            "contact_cmd_z": contact_cmd,
            "test_lift_z": test_lift_z,
            "aerial_z": aerial_z,
            "final_z": final_z,
            "waypoints": selected_rows,
            "recent_rejects": rejects[-10:],
        }

    def _main28_draw_d56_taught_overlay(self, canvas: np.ndarray, spec: Dict[str, Any]) -> np.ndarray:
        out = canvas.copy()
        if not bool(spec.get("ok", False)):
            return out
        rows = list(spec.get("waypoints", []) or [])
        if not rows:
            return out
        try:
            for arm_key, color in (("arm2", (255, 255, 0)), ("arm1", (0, 255, 255))):
                pts = []
                key = f"{arm_key}_board"
                for row in rows:
                    bp = row.get(key)
                    if bp is None:
                        continue
                    px = self.d56.board_to_pixel(self.H_raw, float(bp[0]), float(bp[1]))
                    if px is not None:
                        pts.append((int(round(px[0])), int(round(px[1]))))
                if len(pts) >= 2:
                    self.cv2.polylines(out, [np.asarray(pts, np.int32).reshape(-1, 1, 2)], False, color, 2, self.cv2.LINE_AA)
                    self.cv2.circle(out, pts[-1], 7, color, 2, self.cv2.LINE_AA)
            self.cv2.putText(
                out,
                f"MAIN33 FIX55/56 2D: scale={float(spec['trajectory_scale']):.2f} rot={float(spec['rotation_deg']):+.1f}deg {float(spec['duration_s']):.2f}s",
                (18, 142), self.cv2.FONT_HERSHEY_SIMPLEX, 0.56, (0, 255, 255), 2, self.cv2.LINE_AA,
            )
        except Exception as exc:
            print(f"[D56-MAIN33-OVERLAY-WARN] {exc!r}")
        return out

    def _main28_execute_d56_taught_laydown(self, plan: Any, mark_sent) -> Tuple[bool, str]:
        """main-33 D56: FIX59 grasp choreography + real FIX55/FIX56 2-D replay."""
        if self.args.mode != "physical":
            print("[D56-MAIN33-HOVER] real FIX55/FIX56 path frozen; no robot command sent")
            return True, "D56_MAIN33_HOVER_PREVIEW"
        spec = dict((getattr(plan, "metrics", {}) or {}).get("main28_taught_laydown", {}) or {})
        if not bool(spec.get("ok", False)):
            return False, "D56_MAIN33_TAUGHT_SPEC_MISSING"
        if not all(k in self.arms for k in ("arm2", "arm1")):
            return False, "D56_MAIN33_BOTH_ARMS_REQUIRED"
        rows = list(spec.get("waypoints", []) or [])
        if len(rows) < 8:
            return False, "D56_MAIN33_WAYPOINTS_TOO_FEW"
        arms = {k: self.arms[k] for k in ("arm2", "arm1")}
        tool_t = float(getattr(self.cfg56, "tool_t_rad", 3.14))
        move_command = int(self.args.move_command)
        # Keep the bottom hardware's proven full-open angle, but use teammate FIX59 timing.
        open_angle = float(getattr(self.d56_args, "d31_grip_open", 1.35))
        close_angle = 3.05
        hold_angle = 3.14
        approach_speed = float(getattr(self.d56_args, "d31_approach_speed", 1.12))
        vertical_speed = float(getattr(self.d56_args, "d31_vertical_speed", 1.12))
        grip_board = {k: np.asarray(spec["grip_board"][k], np.float64) for k in ("arm2", "arm1")}
        hover_z = float(getattr(self.cfg56, "hover_z", 180.0))

        def pair_gripper(angle: float, label: str, settle_s: float, serial_delay: float = 0.25) -> Tuple[bool, str]:
            errors: List[str] = []; lock = threading.Lock()
            def one(k: str):
                try:
                    arms[k].set_gripper(float(angle), delay=float(serial_delay), stage=label)
                except Exception as exc:
                    with lock: errors.append(f"{k}:{exc!r}")
            ths = [threading.Thread(target=one, args=(k,), name=f"main33-{label}-{k}") for k in ("arm2", "arm1")]
            for th in ths: th.start()
            for th in ths: th.join(timeout=4.0)
            if any(th.is_alive() for th in ths): return False, f"{label}_THREAD_TIMEOUT"
            if errors: return False, f"{label}_SEND:{errors}"
            time.sleep(max(0.0, float(settle_s)))
            return True, "OK"

        def pair_move(targets: Dict[str, np.ndarray], prev: Dict[str, np.ndarray], speed: float, label: str) -> Tuple[bool, str]:
            results: Dict[str, Tuple[bool, str, Any]] = {}
            def one(k: str):
                results[k] = self.d56._d31_move_verified(
                    arms[k], targets[k], self.cfg56, float(speed), move_command,
                    self.d56_args, label, wait_from=prev[k],
                )
            ths=[threading.Thread(target=one,args=(k,),name=f"main33-{label}-{k}") for k in ("arm2","arm1")]
            for th in ths: th.start()
            for th in ths: th.join(timeout=14.0)
            if any(th.is_alive() for th in ths): return False, f"{label}_THREAD_TIMEOUT"
            for k in ("arm2","arm1"):
                ok, why, _ = results.get(k,(False,"NO_RESULT",None))
                if not ok: return False, f"{label}_{k}:{why}"
            return True, "OK"

        standby={k:np.asarray(self.d56.standby_roarm_pose(self.cfg56,k)[:3],np.float64) for k in ("arm2","arm1")}
        hover={}; contact={}; test_lift={}; aerial={}
        for k in ("arm2","arm1"):
            gx,gy=self.d56.board_to_arm_xy(self.config,k,float(grip_board[k][0]),float(grip_board[k][1]))
            hover[k]=np.asarray([gx,gy,hover_z],np.float64)
            contact[k]=np.asarray([gx,gy,float(spec["contact_cmd_z"][k])],np.float64)
            test_lift[k]=np.asarray([gx,gy,float(spec["test_lift_z"][k])],np.float64)
            aerial[k]=np.asarray([gx,gy,float(spec["aerial_z"][k])],np.float64)

        print(
            f"[D56-MAIN33-SEQUENCE] D56-45 GRIP POINTS -> FIX59 OPEN/CONTACT/3.05/3.14/REASSERT/+18mm "
            f"-> AERIAL -> REAL FIX55/FIX56 2D -> RELEASE | scale={float(spec['trajectory_scale']):.2f} "
            f"rot={float(spec['rotation_deg']):+.1f}deg duration={float(spec['duration_s']):.2f}s"
        )
        mark_sent()
        # FIX59-like: ensure jaws are physically open before the arm starts travelling toward cloth.
        ok,why=pair_gripper(open_angle,"MAIN33_D56_OPEN_PRE_APPROACH",0.30,0.25)
        if not ok:return False,why
        ok,why=pair_move(hover,standby,approach_speed,"MAIN33_D56_APPROACH_OPEN")
        if not ok:return False,why
        # Reassert OPEN at the approach pose and give the servo enough time before descent.
        ok,why=pair_gripper(open_angle,"MAIN33_D56_OPEN_AT_APPROACH",0.30,0.25)
        if not ok:return False,why
        ok,why=pair_move(contact,hover,vertical_speed,"MAIN33_D56_CONTACT_WHILE_OPEN")
        if not ok:return False,why
        time.sleep(0.10)
        ok,why=pair_gripper(close_angle,"MAIN33_D56_LIMITED_CLOSE_305",0.25,0.25)
        if not ok:return False,why
        ok,why=pair_gripper(hold_angle,"MAIN33_D56_HOLD_314",0.08,0.25)
        if not ok:return False,why
        ok,why=pair_gripper(hold_angle,"MAIN33_D56_HOLD_REASSERT_314",0.30,0.25)
        if not ok:return False,why
        # Do not use T:105 `t` as gripper-servo position here; teammate FIX59 explicitly treats it as tool attitude.
        ok,why=pair_move(test_lift,contact,0.28,"MAIN33_D56_VERTICAL_TEST_LIFT_18MM")
        if not ok:
            print(f"[D56-MAIN33-TEST-LIFT-WARN] {why}; HOLD remains latched, continuing to aerial stage")
        ok,why=pair_move(aerial,test_lift,0.45,"MAIN33_D56_AERIAL_START")
        if not ok:return False,why

        final_z={k:float(spec["final_z"][k]) for k in ("arm2","arm1")}
        duration=float(spec["duration_s"]); speed=float(spec.get("t104_speed",MAIN33_D56_T104_SPEED))
        t0=time.monotonic(); max_lag=0.0; last_targets=aerial
        for idx,row in enumerate(rows):
            scheduled=t0+float(row["t"]); remain=scheduled-time.monotonic()
            if remain>0: time.sleep(remain)
            else: max_lag=max(max_lag,-remain)
            zf=float(np.clip(row["zf"],-0.03,1.0)); targets={}
            for k in ("arm2","arm1"):
                bp=np.asarray(row[f"{k}_board"],np.float64)
                rx,ry=self.d56.board_to_arm_xy(self.config,k,float(bp[0]),float(bp[1]))
                z=float(aerial[k][2])+zf*(float(final_z[k])-float(aerial[k][2]))
                z=max(float(final_z[k]),z)
                targets[k]=np.asarray([rx,ry,z],np.float64)
            errors=[]; lock=threading.Lock()
            def send_one(k:str):
                try:
                    p=targets[k]
                    arms[k].move_goal(float(p[0]),float(p[1]),float(p[2]),hold_angle,speed,
                                      move_command=104,stage=f"MAIN33_D56_FIX56_{idx:02d}",delay=0.0,log_command=False)
                except Exception as exc:
                    with lock: errors.append(f"{k}:{exc!r}")
            th2=threading.Thread(target=send_one,args=("arm2",),name="main33-d56-stream-a2")
            th1=threading.Thread(target=send_one,args=("arm1",),name="main33-d56-stream-a1")
            th2.start();th1.start();th2.join();th1.join()
            if errors:return False,f"MAIN33_D56_STREAM_SEND:{errors}"
            last_targets=targets
            if idx in (0,len(rows)-1) or idx%7==0:
                md=np.asarray(row.get("master_disp",[0,0]),np.float64)
                print(f"[D56-MAIN33-STREAM] {idx+1:02d}/{len(rows)} t={float(row['t']):.2f}/{duration:.2f}s master=({md[0]:+.0f},{md[1]:+.0f}) zf={zf:.3f}")
        print(f"[D56-MAIN33-STREAM-END] maxLag={max_lag*1000.0:.1f}ms")
        time.sleep(0.35)

        # FIX72-inspired support policy: XY miss is diagnostic; support height controls release.
        endpoint=last_targets
        for k in ("arm2","arm1"):
            try:
                p=endpoint[k]
                arms[k].move_goal(float(p[0]),float(p[1]),float(p[2]),hold_angle,0.35,
                                  move_command=104,stage="MAIN33_D56_FINAL_CLOSED_SETTLE",delay=0.0)
            except Exception as exc:
                print(f"[D56-MAIN33-SUPPORT-WARN] {k} final settle send failed: {exc!r}")
        time.sleep(0.55)
        for k in ("arm2","arm1"):
            try:
                fb=self.d56._d31_query_feedback(arms[k],float(self.args.standby_feedback_timeout_s))
                xyz=self.d56._d31_feedback_xyz(fb)
                if xyz is not None:
                    xyz=np.asarray(xyz,np.float64); xyerr=float(np.linalg.norm(xyz[:2]-endpoint[k][:2]))
                    zhigh=float(xyz[2]-final_z[k])
                    print(f"[D56-MAIN33-SUPPORT] {k.upper()} xyErr={xyerr:.1f}mm zAboveTarget={zhigh:+.1f}mm (XY diagnostic only)")
                    if zhigh>30.0:
                        # FIX71-style vertical support recovery at ACTUAL current XY; do not abort the action.
                        arms[k].move_goal(float(xyz[0]),float(xyz[1]),float(final_z[k]),hold_angle,0.25,
                                          move_command=104,stage="MAIN33_D56_SUPPORT_VERTICAL_RECOVERY",delay=0.0)
            except Exception as exc:
                print(f"[D56-MAIN33-SUPPORT-WARN] {k} feedback unavailable: {exc!r}; release continues")
        time.sleep(0.55)
        ok,why=pair_gripper(open_angle,"MAIN33_D56_RELEASE",0.65,0.25)
        if not ok:return False,why
        clear={k:endpoint[k].copy() for k in ("arm2","arm1")}
        for k in clear:clear[k][2]=hover_z
        ok,why=pair_move(clear,endpoint,vertical_speed,"MAIN33_D56_CLEAR_OPEN")
        if not ok:return False,why
        print("[D56-MAIN33-COMPLETE] FIX59 grasp + real FIX55/FIX56 2D laydown + release complete")
        return True,"D56_MAIN33_REAL_FIX55_FIX56_OK"

    def _main33_d58_contact_candidates_strong(self, mask: Any, H: np.ndarray, center: np.ndarray,
                                              pull_u: np.ndarray, requested_move: float,
                                              preferred_vertex: Optional[np.ndarray] = None) -> Tuple[Optional[Dict[str, Any]], str]:
        """D58 contact search that ranks maximum safe correction distance first."""
        a=self.d58_args; contour=self.d58._d58_contour_board(mask,H)
        if contour is None or len(contour)<4:return None,"contour unavailable"
        proj=(contour-center.reshape(1,2))@pull_u.reshape(2); mx=float(np.max(proj))
        band=max(25.0,float(a.d58_contact_edge_band_mm)); ids=np.where(proj>=mx-band)[0]
        if len(ids)==0:ids=np.asarray([int(np.argmax(proj))])
        ids=ids[np.argsort(proj[ids])[::-1]]
        desired_anchor=None
        if preferred_vertex is not None:
            pv=np.asarray(preferred_vertex,np.float32).reshape(2); pin=self.d58._d58_unit(center-pv)
            if pin is not None:
                desired_anchor=(pv+pin*float(a.d58_contact_inset_mm)).astype(np.float32)
                ni=int(np.argmin(np.linalg.norm(contour-pv.reshape(1,2),axis=1)))
                ids=np.concatenate([np.asarray([ni],dtype=int),ids[ids!=ni]])
        candidates=[]; max_eval=min(max(1,int(a.d58_contact_max_candidates)*3),len(ids))
        preferred_arm='arm1' if float(pull_u[0])>float(a.d58_horizontal_arm_deadband) else ('arm2' if float(pull_u[0])<-float(a.d58_horizontal_arm_deadband) else None)
        for ii in ids[:max_eval]:
            raw=np.asarray(contour[int(ii)],np.float32); inward=self.d58._d58_unit(center-raw)
            if inward is None:continue
            for inset in (float(a.d58_contact_inset_mm),float(a.d58_contact_inset_mm)+12.0,float(a.d58_contact_inset_mm)+24.0,
                          float(a.d58_contact_inset_mm)+36.0):
                grip=(raw+inward*inset).astype(np.float32)
                okmask,signed,local=self.d58._d58_mask_support(mask,H,grip,a)
                if not okmask:continue
                arm=self.d58.arm_for_x(self.config,float(grip[0]),float(a.center_dead_half_width))
                if arm not in ('arm1','arm2'):continue
                cz=self.d58.contact_z(self.config,arm,a); hz=self.d58.hover_z(self.config,arm,a)
                clear_z=min(float(hz),float(cz)+float(a.d58_post_drag_clear_mm))
                if not self.d58._d58_arm_point_safe(self.config,arm,grip,cz,a):continue
                chosen=0.0;end=None;d=float(requested_move)
                while d>=float(a.d58_min_move_mm)-1e-6:
                    q=(grip+pull_u*d).astype(np.float32)
                    if (self.d58._d58_arm_point_safe(self.config,arm,q,cz,a) and
                        self.d58._d58_arm_point_safe(self.config,arm,q,clear_z,a)):
                        chosen=float(d);end=q;break
                    d-=5.0
                if end is None:continue
                try:
                    rx,ry=self.d58.board_to_arm_xy(self.config,arm,float(grip[0]),float(grip[1]));r0=math.hypot(rx,ry)
                    ex,ey=self.d58.board_to_arm_xy(self.config,arm,float(end[0]),float(end[1]));r1=math.hypot(ex,ey)
                except Exception:continue
                arm_bonus=1.0 if preferred_arm==arm else 0.0
                anchor_pen=0.0 if desired_anchor is None else float(np.linalg.norm(grip-desired_anchor))
                item={'arm':arm,'raw':raw,'grip':grip,'end':end,'move_mm':chosen,'inset_mm':float(inset),
                      'local':float(local),'inside_px':float(signed),'r0':float(r0),'r1':float(r1),
                      'rank':(chosen,arm_bonus,float(local),float(signed),-max(r0,r1),-anchor_pen,-inset)}
                candidates.append(item)
        if not candidates:return None,"no safe contour-inset contact"
        candidates.sort(key=lambda x:x['rank'],reverse=True)
        return candidates[0],"OK"

    def _main33_strengthen_d58_plan(self, frame: np.ndarray, mask: Any, pose: Any, plan: Any) -> Any:
        """Always prefer the strongest safe position-correction candidate; never semantic-no-op."""
        a=self.d58_args
        try:
            samples=self.d58._d58_robust_board_points(mask,self.H_corrected,int(a.mask_sample_stride_px))
            if len(samples)<30:return plan
            center=np.asarray(getattr(plan,"center_board",None),np.float32).reshape(2) if getattr(plan,"center_board",None) is not None else np.median(samples,axis=0).astype(np.float32)
            target=np.asarray(getattr(plan,"target_board",None),np.float32).reshape(2) if getattr(plan,"target_board",None) is not None else None
            vertex=np.asarray(getattr(plan,"contour_vertex",None),np.float32).reshape(2) if getattr(plan,"contour_vertex",None) is not None else None
            source=str(getattr(plan,"target_source","") or "")
            if target is None:
                gapinfo=self.d58._d58_gap_info(samples,self.config,float(a.d58_gap_percentile)); sa,sb=self.d58._d58_choose_two_sides(gapinfo)
                if sa is not None and sb is not None:
                    target=self.d58._d58_weighted_fallback_target(gapinfo['gaps'],sa,sb,self.config,float(a.d58_circumcenter_safe_margin_mm)); source="MAIN33_WEIGHTED_FORCE"
                    try: plan.gaps={k:float(v) for k,v in gapinfo['gaps'].items()};plan.side_a=sa;plan.side_b=sb
                    except Exception:pass
                else:
                    xmin,xmax,ymin,ymax=self.d58.board_bounds(self.config);target=np.asarray([(xmin+xmax)/2,(ymin+ymax)/2],np.float32);source="MAIN33_BOARD_CENTER_FORCE"
            delta=target-center;dist=float(np.linalg.norm(delta));u=self.d58._d58_unit(delta)
            if u is None or dist<1.0:
                # Already centered by geometry: still produce a deterministic correction toward the largest vacant side.
                gapinfo=self.d58._d58_gap_info(samples,self.config,float(a.d58_gap_percentile)); side=max(gapinfo['gaps'],key=gapinfo['gaps'].get)
                side_mid=self.d58._d58_side_midpoint(side,self.config);u=self.d58._d58_unit(side_mid-center)
                if u is None:u=np.asarray([0.0,-1.0],np.float32)
                target=(center+u*float(getattr(self.args,"main33_d58_min_effective_move_mm",120.0))).astype(np.float32);dist=float(np.linalg.norm(target-center));source=f"MAIN33_FORCE_{side}"
            request=min(float(getattr(self.args,"main33_d58_max_move_mm",180.0)),
                        max(float(getattr(self.args,"main33_d58_min_effective_move_mm",120.0)),dist*float(a.d58_center_target_gain)))
            contact,why=self._main33_d58_contact_candidates_strong(mask,self.H_corrected,center,u,request,preferred_vertex=vertex)
            if contact is None:
                print(f"[D58-MAIN33-STRONG-WARN] {why}; retaining original plan")
                return plan
            old_move=float(getattr(plan,"move_mm",0.0) or 0.0)
            plan.center_board=center.astype(np.float32);plan.target_board=np.asarray(target,np.float32);plan.target_source=source or "MAIN33_STRONG"
            plan.pull_unit=np.asarray(u,np.float32);plan.selected_arm=str(contact['arm']);plan.grip_board=np.asarray(contact['grip'],np.float32)
            plan.end_board=np.asarray(contact['end'],np.float32);plan.move_mm=float(contact['move_mm']);plan.grip_inset_mm=float(contact['inset_mm'])
            plan.ok=True;plan.reason="OK_MAIN33_STRONGEST_SAFE"
            plan.overlay=self.d58.make_d58_overlay(frame,mask,plan,self.H_corrected,self.config)
            print(f"[D58-MAIN33-STRONG] oldMove={old_move:.1f}mm -> move={plan.move_mm:.1f}mm request={request:.1f}mm arm={plan.selected_arm.upper()} reachR={contact['r0']:.0f}->{contact['r1']:.0f}")
            return plan
        except Exception as exc:
            print(f"[D58-MAIN33-STRONG-WARN] {exc!r}; retaining original plan")
            return plan

    def _execute_d58_plan_merged(self, plan: Any, mark_sent) -> Tuple[bool, str]:
        if self.args.mode != "physical":
            print("[D58-HOVER-SAFE] frozen plan only; no T104/T106 command sent")
            return True, "D58_HOVER_PREVIEW_NO_ROBOT_COMMAND"
        key = str(getattr(plan, "selected_arm", ""))
        if key not in {"arm1", "arm2"} or getattr(plan, "grip_board", None) is None or getattr(plan, "end_board", None) is None:
            return False, "D58_INVALID_LOCKED_PLAN"
        arm = self.arms.get(key)
        if arm is None:
            return False, f"D58_{key.upper()}_SERIAL_MISSING"
        a = self.d58_args
        gp = np.asarray(plan.grip_board, np.float32); ep = np.asarray(plan.end_board, np.float32)
        gx, gy = self.d58.board_to_arm_xy(self.config, key, float(gp[0]), float(gp[1]))
        ex, ey = self.d58.board_to_arm_xy(self.config, key, float(ep[0]), float(ep[1]))
        cz = self.d58.contact_z(self.config, key, a); hz = self.d58.hover_z(self.config, key, a); tt = self.d58.tool_t(self.config, key)
        mb = gp + (ep-gp) * float(np.clip(a.d58_drag_mid_ratio, 0.20, 0.80))
        mx, my = self.d58.board_to_arm_xy(self.config, key, float(mb[0]), float(mb[1]))
        clear_z = min(float(hz), float(cz) + float(a.d58_post_drag_clear_mm))
        label = key.upper()
        print(f"[D58-2-MERGED-SEQUENCE] {label} OPEN -> HOVER -> CONTACT -> CLOSE/HOLD -> LOW-Z MID -> LOW-Z SWEEP -> CLEAR -> OPEN -> RETRACT")
        print(f"[D58-2-MERGED-DRAG-Z] contact={cz:.1f} mid={cz:.1f} sweep={cz:.1f} prePullLift=0.0mm")
        mark_sent(); arm.set_gripper(float(a.grip_open), delay=0.0, stage="D58_OPEN"); time.sleep(float(a.pre_open_wait_s))
        arm.move_goal(gx,gy,hz,tt,float(a.free_speed),move_command=int(self.args.move_command),stage="D58_HOVER_OPEN",delay=0.0); time.sleep(float(a.hover_wait_s))
        arm.set_gripper(float(a.grip_open), delay=0.0, stage="D58_OPEN_BEFORE_DESCENT"); time.sleep(float(a.open_before_descent_wait_s))
        arm.move_goal(gx,gy,cz,tt,float(a.near_speed),move_command=int(self.args.move_command),stage="D58_CONTACT_OPEN",delay=0.0); time.sleep(float(a.contact_move_wait_s))
        arm.set_gripper(float(a.grip_close), delay=0.0, stage="D58_LIMITED_CLOSE"); time.sleep(float(a.close_limited_wait_s))
        for i in range(max(1,int(a.close_repeat))):
            arm.set_gripper(float(a.grip_hold), delay=0.0, stage=f"D58_HOLD_{i+1}")
            if i+1 < int(a.close_repeat): time.sleep(float(a.close_repeat_gap_s))
        time.sleep(float(a.close_final_hold_s))
        rep = self.d56._d31_query_feedback(arm, timeout_s=1.2)
        tv = None if not isinstance(rep, dict) else rep.get("t")
        print(f"[D58-2-MERGED-GRIP-FEEDBACK] {label} t={tv}")
        if tv is not None and float(tv) < float(a.obvious_open_feedback_rad):
            grip_ok = False
            for j in range(int(a.close_extra_retries)):
                arm.set_gripper(float(a.grip_hold), delay=0.0, stage=f"D58_HOLD_RETRY_{j+1}")
                time.sleep(float(a.close_retry_wait_s))
                rep = self.d56._d31_query_feedback(arm, timeout_s=1.2)
                tv = None if not isinstance(rep, dict) else rep.get("t")
                print(f"[D58-2-MERGED-GRIP-RETRY] {label} retry={j+1} t={tv}")
                if tv is None or float(tv) >= float(a.obvious_open_feedback_rad):
                    grip_ok = True
                    break
            if not grip_ok:
                print("[D58-2-MERGED-GRIP-BLOCK] gripper still obviously open")
                arm.set_gripper(float(a.grip_open), delay=0.0, stage="D58_FAIL_OPEN")
                time.sleep(0.3)
                arm.move_goal(gx,gy,hz,tt,float(a.free_speed),move_command=int(self.args.move_command),stage="D58_FAIL_RETRACT",delay=0.0)
                time.sleep(1.0)
                return False, "D58_GRIPPER_OPEN_BLOCK"
        arm.move_goal(mx,my,cz,tt,float(a.d58_translate_speed),move_command=int(self.args.move_command),stage="D58_LOW_Z_MID_CLOSED",delay=0.0); time.sleep(float(a.d58_mid_wait_s))
        arm.move_goal(ex,ey,cz,tt,float(a.d58_translate_speed),move_command=int(self.args.move_command),stage="D58_LOW_Z_SWEEP_CLOSED",delay=0.0); time.sleep(float(a.d58_translate_wait_s))
        arm.move_goal(ex,ey,clear_z,tt,float(a.d58_post_drag_clear_speed),move_command=int(self.args.move_command),stage="D58_POST_DRAG_CLEAR_CLOSED",delay=0.0); time.sleep(float(a.d58_post_drag_clear_wait_s))
        arm.set_gripper(float(a.release_open), delay=0.0, stage="D58_RELEASE_AFTER_CLEAR"); time.sleep(float(a.release_wait_s))
        arm.move_goal(ex,ey,hz,tt,float(a.vertical_retract_speed),move_command=int(self.args.move_command),stage="D58_VERTICAL_RETRACT_OPEN",delay=0.0); time.sleep(float(a.vertical_retract_wait_s))
        return True, "D58_ACTION_OK"

    def _d50_main_safe_release_and_standby(self) -> Tuple[bool, str]:
        """After 50-1.py returns, safely release ARM2 and restore main imaging standby.

        D50 owns a separate serial session and may terminate before its own A150
        standby (for example on release-feedback failure).  Main therefore
        reasserts an OPEN command after reconnecting, verifies the gripper is not
        still closed, and only then moves both arms to the normal main standby.
        """
        if self.args.mode != "physical":
            return True, "D50_POST_NO_PHYSICAL"
        arm2 = self.arms.get("arm2")
        if arm2 is None:
            print("[D50-POST] ARM2 serial missing; standby blocked")
            return False, "D50_POST_ARM2_MISSING"

        open_target = 1.35
        released = False
        last_t = None
        for attempt in range(1, 4):
            try:
                arm2.set_gripper(open_target, delay=0.0, stage=f"D50_MAIN_SAFE_OPEN_{attempt}")
                time.sleep(0.45)
                rep = self.d56._d31_query_feedback(arm2, timeout_s=1.2)
                if isinstance(rep, dict) and rep.get("t") is not None:
                    last_t = float(rep.get("t"))
                    # Main only needs proof that the jaw is no longer in the
                    # cloth-holding/closed range before carrying the arm home.
                    released = last_t <= 2.15
                    print(
                        f"[D50-POST-OPEN] attempt={attempt}/3 target={open_target:.2f} "
                        f"feedback={last_t:.3f} released={released}"
                    )
                    if released:
                        break
                else:
                    print(f"[D50-POST-OPEN] attempt={attempt}/3 feedback unavailable")
            except Exception as exc:
                print(f"[D50-POST-OPEN] attempt={attempt}/3 error={exc!r}")

        if not released:
            print(
                f"[D50-POST-BLOCK] ARM2 open could not be verified (last_t={last_t}); "
                "automatic standby is intentionally blocked to avoid dragging cloth"
            )
            return False, "D50_POST_RELEASE_NOT_VERIFIED"

        if not self._ensure_camera_clear("D50_POST_ACTION", allow_move=True):
            print("[D50-POST-BLOCK] main standby move/verification failed")
            return False, "D50_POST_STANDBY_FAILED"
        print("[D50-POST-STANDBY] ARM1/ARM2 main standby verified")
        return True, "D50_POST_STANDBY_OK"

    def _start_execution(self) -> None:
        with self.state_lock:
            if self.motion_busy:
                return
            # The I-locked plan remains immutable until this Enter handler clears it.
            # Reuse that exact object instead of deep-copying full-resolution frames,
            # masks, and diagnostics immediately before robot execution.
            locked = self.locked
        if locked is None or not locked.plan_ok or locked.action == "NONE":
            print("[ENTER] blocked: no exact valid I-locked physical plan")
            return
        if not self.empty_baseline_ready:
            print("[ENTER] blocked: E empty-board baseline is not ready")
            return
        try:
            self._verify_sources_unchanged()
        except Exception as exc:
            print(f"[ENTER] blocked: {exc}")
            return
        age = time.time() - locked.created_at
        if age > float(self.args.locked_plan_max_age_s):
            print(f"[ENTER] blocked: plan stale by age {age:.1f}s")
            self._invalidate_lock("PLAN_AGE_STALE")
            return
        # The original D54/D55/D56 manual Enter flows approve the exact visible
        # I-lock by age/H, without V6's extra mean-pixel-difference veto.
        try:
            self.recorder.open_transition(locked)
        except Exception as exc:
            print(f"[ENTER] transition open blocked: {exc}")
            return
        with self.state_lock:
            self.motion_busy = True
            self.status = f"ENTER ACCEPTED - STARTING {locked.action}"
            self.locked = None
        print(f"[ENTER-ACCEPTED] action={locked.action}; robot worker starting now")

        def worker() -> None:
            sent = False

            def mark_sent():
                nonlocal sent
                sent = self.args.mode == "physical"

            success = False
            detail = "NOT_STARTED"
            after_record = None
            try:
                if locked.action == "D50_BASKET_SWING_LAYDOWN":
                    if self.args.mode != "physical":
                        success, detail = True, "D50_HOVER_PREVIEW_NO_ROBOT_COMMAND"
                    else:
                        # 50-1.py owns its ARM2 serial session and camera. Hold the
                        # camera lock for the complete hand-off so the UI loop
                        # cannot interpret the intentional release as a fatal
                        # camera read failure and terminate the episode.
                        for arm in list(self.arms.values()):
                            try: arm.close()
                            except Exception: pass
                        self.arms = {}
                        command = [
                            sys.executable, self.d50_path, "--send", "--load-calib",
                            "--auto-run-once", "--no-window", "--no-terminal-control",
                            "--calib-file", self.args.d50_basket_calib,
                            "--board-config", self.args.config,
                        ]
                        print("[D50-ENTER] collector camera/serial released; starting 50-1.py once")
                        mark_sent()
                        with self.cap_lock:
                            try: self.cap.release()
                            except Exception: pass
                            completed = subprocess.run(command, check=False)
                            success = completed.returncode == 0
                            detail = f"D50_SUBPROCESS_EXIT_{completed.returncode}"
                            self.cap = self._open_camera()
                            # Opening the USB camera again can restore driver
                            # defaults.  Reapply the exact episode exposure/
                            # white-balance profile used at initial startup,
                            # then discard unstable post-control frames.
                            self.camera_controls_meta = apply_camera_controls(
                                self.args.camera_controls_json,
                                self.args.camera_device,
                                self.args.camera_controls_strict,
                            ) if self.args.camera_controls_enable else {
                                "enabled": False,
                                "profile_path": self.args.camera_controls_json,
                            }
                            for _ in range(max(
                                0, int(self.args.camera_controls_stabilization_frames)
                            )):
                                self.cap.read()
                            print(
                                "[D50-CAMERA-CONTROLS] startup JSON profile "
                                "reapplied after camera reopen"
                            )
                        self.arms = self._connect_arms_no_motion()
                        print("[D50-RETURN] collector camera + ARM1/ARM2 serial reconnected")
                        d50_post_ok, d50_post_detail = self._d50_main_safe_release_and_standby()
                        detail = f"{detail}|{d50_post_detail}"
                        if success and not d50_post_ok:
                            success = False
                elif locked.action == "D54_OUTER_PULL":
                    success, detail = self.d54._d51v4_execute_diagonal_pull(
                        locked.plan, self.arms, self.config, self.cfg54, self.d54_args,
                        on_verified_start=mark_sent,
                    )
                elif locked.action == "D55_PRESS_SWEEP":
                    if self.args.mode != "physical":
                        # D55-v13's low-level executor does not honor
                        # cfg.hover_only and would descend to press Z even in a
                        # hover session.  Keep hover strictly non-contact here.
                        success = True
                        detail = "D55_HOVER_PREVIEW_NO_ROBOT_COMMAND"
                        print("[D55-HOVER-SAFE] plan kept on screen; no T104/T106 command sent")
                    else:
                        original_sends: Dict[str, Any] = {}
                        try:
                            # D55 has no verified-start callback.  Observe its first
                            # actual serial send so partial failures are recorded.
                            for arm_key, arm in self.arms.items():
                                original = getattr(arm, "send", None)
                                if original is None:
                                    continue
                                original_sends[arm_key] = original

                                def tracked_send(*send_args, _original=original, **send_kwargs):
                                    mark_sent()
                                    return _original(*send_args, **send_kwargs)

                                arm.send = tracked_send
                            success = bool(self.d55.execute_dual_wrinkle_stretch_plan(
                                locked.plan, self.arms, self.cfg55, move_command=int(self.args.move_command),
                                args=self.d55_args, skip_confirm=True,
                            ))
                        finally:
                            for arm_key, original in original_sends.items():
                                self.arms[arm_key].send = original
                        detail = "D55_ACTION_OK" if success else "D55_EXEC_FAILED"
                elif locked.action == "D58_CIRC_POSITION":
                    success, detail = self._execute_d58_plan_merged(locked.plan, mark_sent)
                elif locked.action == "D56_WAIST_LIFT_LAYDOWN":
                    if str(getattr(locked.plan, "action", "")).startswith("D47_"):
                        if self.args.mode != "physical":
                            success = True
                            detail = "D56_D47_HOVER_PREVIEW_NO_ROBOT_COMMAND"
                            print("[D56-D47-HOVER-SAFE] plan kept on screen; no T104/T106 command sent")
                        else:
                            original_sends: Dict[str, Any] = {}
                            try:
                                for arm_key, arm in self.arms.items():
                                    original = getattr(arm, "send", None)
                                    if original is None:
                                        continue
                                    original_sends[arm_key] = original

                                    def tracked_send(*send_args, _original=original, **send_kwargs):
                                        mark_sent()
                                        return _original(*send_args, **send_kwargs)

                                    arm.send = tracked_send
                                success = bool(self.d56.execute_dual_wrinkle_stretch_plan(
                                    locked.plan, self.arms, self.cfg56,
                                    move_command=int(self.args.move_command),
                                    args=self.d56_args, skip_confirm=True,
                                ))
                            finally:
                                for arm_key, original in original_sends.items():
                                    self.arms[arm_key].send = original
                            detail = "D56_D47_ACTION_OK" if success else "D56_D47_EXEC_FAILED"
                            if success:
                                self.d56_prespread_attempt_count += 1
                    else:
                        # main-33: D56-45 stays authoritative for perception/grasp planning.
                        # Prefer the transplanted real FIX55/FIX56 2-D replay.  If that
                        # adapter cannot build a source-faithful path for this frozen plan,
                        # do NOT turn the semantic decision into NO_ACTION: execute the
                        # original D56-45 physical plan instead.  Hard serial/physical
                        # safety failures inside either executor are still respected.
                        taught_spec = dict(
                            (getattr(locked.plan, "metrics", {}) or {}).get(
                                "main28_taught_laydown", {}
                            ) or {}
                        )
                        if bool(taught_spec.get("ok", False)):
                            success, detail = self._main28_execute_d56_taught_laydown(
                                locked.plan, mark_sent
                            )
                        else:
                            print(
                                "[D56-MAIN33-FALLBACK] real FIX55/FIX56 adapter unavailable "
                                "-> executing native D56-45 plan instead of semantic NO_ACTION"
                            )
                            success, detail = self.d56._d31_execute_dual_grasp_plan(
                                locked.plan, self.arms, self.config, self.cfg56, self.d56_args,
                                on_verified_start=mark_sent,
                            )
                            detail = f"D56_NATIVE_FALLBACK:{detail}"
                        if success and self.args.mode == "physical":
                            self.d56_prespread_attempt_count = 0
                committed = bool(success and self.args.mode == "physical" and sent)
                if committed:
                    self.recorder.note_motion_committed()
                # Every completed physical action must leave both arms at standby.
                # Executors that already return to standby are only verified here;
                # an additional move is sent only when that verification fails.
                after_ready = bool(success)
                module, namespace, cfg, action_h = self._adapter_for_action(locked.action)
                if after_ready and self.args.mode == "physical":
                    if not self._ensure_camera_clear("POST_ACTION_CHECK", allow_move=False):
                        standby_module, standby_cfg = self._module_cfg_for_action(locked.action)
                        moved = standby_module.move_arms_to_standby(
                            self.arms, standby_cfg, move_command=int(self.args.move_command),
                            reason=f"automatic {locked.action} post-action standby",
                        )
                        if moved:
                            time.sleep(max(0.0, float(self.args.imaging_standby_wait_s)))
                            after_ready = bool(self._ensure_camera_clear("POST_ACTION_VERIFY", allow_move=False))
                        else:
                            after_ready = False
                if after_ready and self.args.mode == "physical":
                    time.sleep(max(
                        0.0,
                        float(getattr(namespace, "wrinkle_after_settle_s", 0.45)),
                        float(getattr(namespace, "d30_after_settle_s", 0.0)),
                    ))
                if after_ready:
                    after_bundle = self._capture_action_frame(locked.action, flush=1)
                    after_obs = self._infer_for_action(locked.action, after_bundle.corrected)
                    after_heat = None
                    if (
                        locked.action in {"D54_OUTER_PULL", "D55_PRESS_SWEEP"}
                        and getattr(after_obs, "mask", None) is not None
                    ):
                        heat_module = self.d54 if locked.action == "D54_OUTER_PULL" else self.d55
                        heat_args = self.d54_args if locked.action == "D54_OUTER_PULL" else self.d55_args
                        after_heat = heat_module.build_d21_wrinkle_heatmap(
                            after_bundle.corrected, after_obs, self.H_corrected, heat_args
                        )
                    # Use the source module's real empty-plan schema.  D55's
                    # original overlay reads center_px directly (not through
                    # getattr), so a generic SimpleNamespace aborts AFTER image
                    # creation and makes an otherwise successful KEEP record
                    # ineligible with after_missing.
                    if locked.action == "D58_CIRC_POSITION":
                        after_plan = self.d58.D58Plan(False, "AFTER_OBSERVATION")
                    else:
                        after_plan = module.DualWrinkleStretchPlan(False, "AFTER_OBSERVATION")
                    if locked.action in {"D54_OUTER_PULL", "D55_PRESS_SWEEP"}:
                        after_canvas = self._d54_d55_original_overlay(
                            after_bundle, locked.action, after_obs, after_plan, after_heat,
                            effect_state="SUCCESS" if success else "EXEC_FAILED",
                            effect_summary=str(detail),
                        )
                    else:
                        after_canvas = self._operator_overlay(
                            after_bundle, locked.action, after_obs, after_plan, after_heat
                        )
                        after_canvas = self._banner(
                            after_canvas, locked.state_label,
                            "AFTER_" + locked.action, bool(success), detail,
                        )
                    after_locked = LockedPlan(
                        locked.state_label, "AFTER_" + locked.action, after_bundle, after_obs,
                        after_plan,
                        after_heat, after_canvas, action_h.copy(), time.time(), False, "AFTER_OBSERVATION", None,
                        {"after_action": locked.action},
                    )
                    after_record = self.recorder.save_observation(after_locked, "AFTER_ACTION", self._environment_metadata())
                    with self.state_lock:
                        self.display_image = after_canvas.copy()
                else:
                    detail = str(detail) + "|ORIGINAL_POST_CAMERA_CLEAR_FAILED"
            except Exception as exc:
                detail = f"EXEC_EXCEPTION:{type(exc).__name__}:{exc}"
                print(f"[EXEC-ERROR] {exc!r}")
            finally:
                execution_sent = bool(sent)
                committed = bool(success and self.args.mode == "physical" and execution_sent)
                self.recorder.complete_transition({
                    "execution_success": bool(success),
                    "execution_sent": execution_sent,
                    "garment_motion_committed": committed,
                    "execution_detail": str(detail),
                    "motion_parameters": _json_safe(locked.plan),
                    "automatic_result": None,
                }, after_record)
                with self.state_lock:
                    self.motion_busy = False
                    self.status = "RESULT READY: TERMINAL g=GOOD / b=BAD / k=SKIP"
                print(f"[EXEC-DONE] action={locked.action} success={success} sent={execution_sent} committed={committed}")
                print("[RESULT-INPUT-NOW] type g (GOOD), b (BAD), or k (SKIP), then press Enter")

        self.worker = threading.Thread(target=worker, name=f"bottom-vla-{locked.action}", daemon=True)
        self.worker.start()

    def _invalidate_lock(self, reason: str) -> None:
        with self.state_lock:
            self.locked = None
            self.display_image = None
            self.status = f"LOCK INVALIDATED: {reason}; PRESS I"

    def _queue(self, event: str) -> None:
        with self.event_lock:
            self.events.append(str(event))

    def _drain_events(self) -> List[str]:
        with self.event_lock:
            out = list(self.events)
            self.events.clear()
            return out

    def _handle(self, event: str) -> None:
        with self.state_lock:
            infer_busy_now = bool(self.inference_busy)
        if infer_busy_now and event != "QUIT":
            print(f"[INPUT] blocked while {self.inference_action or 'inference'} is running: {event}")
            return
        if self.recorder.terminal_status and event != "QUIT":
            print(f"[EPISODE] already terminal={self.recorder.terminal_status}; press Q")
            return
        if event.startswith("ACTION:"):
            if self.motion_busy or self.recorder.review_pending():
                print("[ACTION] blocked: finish G/B/K and Y/N for the previous action")
                return
            self.selected_action = event.split(":", 1)[1]
            self._invalidate_lock("ACTION_CHANGED")
            print(f"[ACTION] {self.selected_action}")
        elif event == "I":
            self._start_prepare_action()
        elif event == "EMPTY_BASELINE":
            self._capture_empty_board()
        elif event == "LOCK_H":
            self._lock_and_save_homography()
        elif event == "ENTER":
            self._start_execution()
        elif event.startswith("RESULT:"):
            if self.motion_busy:
                print("[RESULT] blocked until motion/after observation finishes")
                return
            if not self.recorder.require_result():
                print("[RESULT] not available yet; wait for [RESULT-INPUT-NOW] after EXEC-DONE")
                return
            if self.recorder.label_result(event.split(":", 1)[1]):
                self.status = "RESULT JUDGED: y=KEEP DATA / n=DISCARD DATA"
                print("[COLLECT-INPUT-NOW] type y (KEEP) or n (DISCARD), then press Enter")
        elif event.startswith("COLLECT:"):
            if self.motion_busy:
                print("[COLLECT] blocked until motion/after observation finishes")
                return
            keep = event.split(":", 1)[1] == "KEEP"
            if self.recorder.decide_collection(keep):
                self.status = "DONE: TYPE NEXT ACTION 54/55/56/58, THEN I"
                self.display_image = None
                print("[NEXT] action 54/55/56/58 -> i -> Enter")
        elif event == "PLAN_INACCURATE":
            if self.motion_busy:
                print("[VLA-PLAN] blocked while robot motion is running")
                return
            with self.state_lock:
                locked = copy.deepcopy(self.locked)
            if locked is None:
                print("[VLA-PLAN] press I first; there is no frozen plan to reject")
                return
            if self.recorder.mark_plan_inaccurate(locked):
                self._invalidate_lock("PLAN_INACCURATE")
                self.status = "PLAN INACCURATE RECORDED: CHOOSE 54/55/56 OR PRESS I AGAIN"
        elif event == "MASK_INACCURATE":
            if self.motion_busy:
                return
            if self.recorder.mark_mask_inaccurate():
                self._invalidate_lock("MASK_INACCURATE")
                self.status = "MASK INACCURATE: PLAN DISCARDED; PRESS I AGAIN"
                print("[VLA-MASK] current I-lock intentionally discarded; Enter remains blocked until a fresh I-lock")
        elif event == "FINISH_SUCCESS":
            if self.recorder.review_pending():
                print("[FINISH] finish G/B/K and Y/N first")
                return
            if self.recorder.finish_success():
                self.status = "EPISODE COMPLETE / SUCCESS; Q TO EXIT"
        elif event == "ABORT_FAILED":
            if self.motion_busy:
                print("[ABORT] blocked while robot motion is running")
                return
            now = time.monotonic()
            elapsed = now - float(self.abort_armed_at)
            if (
                float(self.abort_armed_at) > 0.0
                and 0.6 <= elapsed <= 4.0
                and now <= float(self.abort_confirm_until)
            ):
                self.abort_armed_at = 0.0
                self.abort_confirm_until = 0.0
                self.recorder.fail_episode()
            else:
                self.abort_armed_at = now
                self.abort_confirm_until = now + 4.0
                self.status = "ABORT ARMED: PRESS ABORT AGAIN IN 0.6~4.0s TO CONFIRM"
                print("[ABORT-CONFIRM] first ABORT ignored; press ABORT again in 0.6~4.0s only if FAILED is intended")
        elif event == "QUIT":
            if self.motion_busy:
                print("[QUIT] blocked while robot motion is running")
            else:
                if self.args.mode == "physical":
                    print("[QUIT] returning ARM1/ARM2 to standby before normal shutdown")
                    if not self._ensure_camera_clear("NORMAL_QUIT", allow_move=True):
                        print("[QUIT-WARN] standby move/verification failed; closing serial without further motion")
                self.closed = True

    def _draw_panel(self) -> np.ndarray:
        w, h = 1060, 540
        panel = np.full((h, w, 3), 28, np.uint8)
        self.cv2.putText(panel, "VLA DATASET CONTROLS - BOTTOM", (24, 36), self.cv2.FONT_HERSHEY_SIMPLEX, 0.82, (255, 255, 255), 2)
        rects: List[Tuple[str, Tuple[int, int, int, int], Tuple[int, int, int]]] = []

        def row(items: Sequence[Tuple[str, str]], y: int, color: Tuple[int, int, int], selected: Optional[str] = None):
            margin, gap = 22, 8
            width = int((w - 2 * margin - gap * (len(items) - 1)) / len(items))
            for i, (label, event) in enumerate(items):
                x0 = margin + i * (width + gap)
                rect = (x0, y, x0 + width, y + 54)
                value = event.split(":", 1)[-1]
                fill = tuple(min(255, int(c * 1.45)) for c in color) if selected == value else color
                self.cv2.rectangle(panel, rect[:2], rect[2:], fill, -1)
                self.cv2.rectangle(panel, rect[:2], rect[2:], (230, 230, 230), 1)
                self.cv2.putText(panel, label, (x0 + 8, y + 34), self.cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 0, 0), 2)
                rects.append((event, rect, color))

        row([("D50 BASKET SWING", "ACTION:D50_BASKET_SWING_LAYDOWN"), ("D54 OUTER PULL", "ACTION:D54_OUTER_PULL"), ("D55 PRESS SWEEP", "ACTION:D55_PRESS_SWEEP"),
             ("D56 WAIST LIFT", "ACTION:D56_WAIST_LIFT_LAYDOWN"), ("D58 CIRC POS", "ACTION:D58_CIRC_POSITION")], 58, (100, 220, 150), self.selected_action)
        row([("L: LOCK+SAVE H", "LOCK_H"), ("E: EMPTY BOARD", "EMPTY_BASELINE"),
             ("I: LOCK PLAN", "I"), ("ENTER: EXECUTE", "ENTER")], 128, (100, 210, 250))
        row([("GOOD", "RESULT:GOOD"), ("BAD", "RESULT:BAD"), ("SKIP", "RESULT:SKIP")], 198, (90, 190, 250))
        row([("KEEP DATA (Y)", "COLLECT:KEEP"), ("DISCARD DATA (N)", "COLLECT:DISCARD")], 268, (100, 210, 170))
        row([("MASK INACCURATE (M)", "MASK_INACCURATE"), ("PLAN INACCURATE (P)", "PLAN_INACCURATE"),
             ("FINISH EPISODE", "FINISH_SUCCESS"), ("ABORT FAILED (2X)", "ABORT_FAILED")], 338, (110, 140, 235))
        row([("QUIT", "QUIT")], 408, (120, 120, 120))
        self.cv2.putText(panel, f"STATUS: {self.status}"[:125], (24, 500), self.cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 0), 2)
        self._panel_rects = rects
        return panel

    def _draw_dataset_counter(self, image: np.ndarray) -> np.ndarray:
        stats = self.recorder.dataset_stats_snapshot()
        out = image
        h, w = out.shape[:2]
        box_w, box_h = 390, 199
        x0 = max(8, w - box_w - 12)
        y0 = 12
        x1 = min(w - 8, x0 + box_w)
        y1 = min(h - 8, y0 + box_h)
        overlay = out.copy()
        self.cv2.rectangle(overlay, (x0, y0), (x1, y1), (20, 20, 20), -1)
        self.cv2.addWeighted(overlay, 0.72, out, 0.28, 0, out)
        lines = [
            f"DATASET KEEP(Y): {int(stats.get('keep', 0))}",
            f"TRAIN ELIGIBLE: {int(stats.get('eligible', 0))}",
        ]
        keep_by_action = stats.get("by_action_keep", {}) or {}
        for action, short in (("D54_OUTER_PULL", "D54"), ("D55_PRESS_SWEEP", "D55"), ("D56_WAIST_LIFT_LAYDOWN", "D56"), ("D58_CIRC_POSITION", "D58")):
            bucket = (stats.get("by_action", {}) or {}).get(action, {}) or {}
            lines.append(
                f"{short}: K {int(keep_by_action.get(action, 0))} / E {int(bucket.get('eligible', 0))} "
                f"(G {int(bucket.get('good', 0))}/B {int(bucket.get('bad', 0))})"
            )
        lines.append(
            f"PLAN BAD(P): {int(stats.get('plan_inaccurate', 0))}  SKIP(K): {int(stats.get('skip', 0))}"
        )
        lines.append(
            f"DISCARD: {int(stats.get('discard', 0))}"
        )
        for i, text in enumerate(lines):
            y = y0 + 25 + i * 25
            self.cv2.putText(out, text, (x0 + 10, y), self.cv2.FONT_HERSHEY_SIMPLEX, 0.53, (255, 255, 255), 1, self.cv2.LINE_AA)
        return out

    def _mouse(self, event, x, y, flags, param) -> None:
        del flags, param
        if event != self.cv2.EVENT_LBUTTONUP:
            return
        for name, rect, _color in getattr(self, "_panel_rects", []):
            x0, y0, x1, y1 = rect
            if x0 <= x <= x1 and y0 <= y <= y1:
                self._queue(name)
                return

    def _terminal_event(self) -> Optional[str]:
        try:
            ready, _, _ = select.select([sys.stdin], [], [], 0.0)
        except Exception:
            return None
        if not ready:
            return None
        line = sys.stdin.readline().strip().lower()
        mapping = {
            "50": "ACTION:D50_BASKET_SWING_LAYDOWN", "54": "ACTION:D54_OUTER_PULL", "55": "ACTION:D55_PRESS_SWEEP", "56": "ACTION:D56_WAIST_LIFT_LAYDOWN", "58": "ACTION:D58_CIRC_POSITION",
            "l": "LOCK_H", "lock": "LOCK_H",
            "e": "EMPTY_BASELINE", "empty": "EMPTY_BASELINE",
            "i": "I", "enter": "ENTER", "": "ENTER",
            "good": "RESULT:GOOD", "g": "RESULT:GOOD",
            "bad": "RESULT:BAD", "b": "RESULT:BAD", "skip": "RESULT:SKIP", "k": "RESULT:SKIP",
            "y": "COLLECT:KEEP", "yes": "COLLECT:KEEP", "keep": "COLLECT:KEEP",
            "n": "COLLECT:DISCARD", "no": "COLLECT:DISCARD", "discard": "COLLECT:DISCARD",
            "mask": "MASK_INACCURATE", "m": "MASK_INACCURATE",
            "plan": "PLAN_INACCURATE", "p": "PLAN_INACCURATE", "finish": "FINISH_SUCCESS",
            "abort": "ABORT_FAILED", "failed": "ABORT_FAILED", "q": "QUIT", "quit": "QUIT",
        }
        return mapping.get(line)

    def run(self) -> int:
        window = "BOTTOM VLA MANUAL - D54 D55 D56"
        panel_name = "VLA DATASET CONTROLS - BOTTOM"
        self.cv2.namedWindow(window, self.cv2.WINDOW_NORMAL)
        self.cv2.namedWindow(panel_name, self.cv2.WINDOW_NORMAL)
        self.cv2.resizeWindow(window, int(self.args.width), int(self.args.height))
        self.cv2.resizeWindow(panel_name, 1060, 540)
        self.cv2.setMouseCallback(panel_name, self._mouse)
        print("\n[KEYS / TERMINAL]")
        print("  terminal 50 / 54 / 55 / 56 : choose basket / D54 / D55 / D56")
        print("  L : OPTIONAL manual Homography recalibration (startup repairs H/raw_H automatically)")
        print("  E : mandatory once on EMPTY BOARD; installs D54/D55 corrected + D56 raw baselines")
        print("  I : verify camera-clear + frozen plan | ENTER : execute exact displayed plan")
        print("  AFTER [RESULT-INPUT-NOW] only: g=GOOD / b=BAD / k=SKIP")
        print("  AFTER [COLLECT-INPUT-NOW] only: y=KEEP DATA / n=DISCARD DATA")
        print("  M : MASK_INACCURATE | P : PLAN_INACCURATE (after I, before Enter) | Q : quit")
        print(f"[MODE] {self.args.mode} collection_phase={self.args.collection_phase}")
        try:
            while not self.closed:
                ok, raw = self._read_raw(flush=0)
                if not ok or raw is None:
                    print("[CAM] read failed")
                    break
                # main-32 TOP-VLA port: keep live preview RAW and cheap.  Do NOT
                # undistort every camera frame.  The finalized I snapshot is corrected
                # exactly once inside _capture_i_frame_from_live().
                with self.live_frame_lock:
                    self.latest_live_raw = raw
                    self.latest_live_corrected = None
                    self.latest_live_monotonic = time.monotonic()
                with self.state_lock:
                    shown = self.display_image.copy() if self.display_image is not None else raw.copy()
                    busy = self.motion_busy
                    infer_busy = self.inference_busy
                    infer_action = self.inference_action
                    status = self.status
                if busy:
                    self.cv2.putText(shown, "ROBOT MOTION RUNNING - INPUT BLOCKED", (20, 40), self.cv2.FONT_HERSHEY_SIMPLEX, 0.78, (0, 0, 255), 3)
                elif infer_busy:
                    infer_short = {
                        "D50_BASKET_SWING_LAYDOWN": "D50", "D54_OUTER_PULL": "D54",
                        "D55_PRESS_SWEEP": "D55", "D56_WAIST_LIFT_LAYDOWN": "D56",
                        "D58_CIRC_POSITION": "D58",
                    }.get(infer_action, "I")
                    self.cv2.putText(shown, f"{infer_short} I-LOCK RUNNING - CAMERA LOOP ALIVE", (20, 40), self.cv2.FONT_HERSHEY_SIMPLEX, 0.78, (0, 200, 255), 3)
                self.cv2.putText(shown, status[:120], (20, shown.shape[0] - 24), self.cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
                shown = self._draw_dataset_counter(shown)
                self.cv2.imshow(window, shown)
                self.cv2.imshow(panel_name, self._draw_panel())
                key = self.cv2.waitKey(1) & 0xFF
                keymap = {
                    ord("l"): "LOCK_H", ord("e"): "EMPTY_BASELINE",
                    ord("i"): "I", 10: "ENTER", 13: "ENTER",
                    ord("g"): "RESULT:GOOD", ord("b"): "RESULT:BAD",
                    ord("k"): "RESULT:SKIP", ord("y"): "COLLECT:KEEP", ord("n"): "COLLECT:DISCARD",
                    ord("m"): "MASK_INACCURATE", ord("p"): "PLAN_INACCURATE",
                    ord("q"): "QUIT", 27: "QUIT",
                }
                if key in keymap:
                    self._queue(keymap[key])
                terminal = self._terminal_event()
                if terminal:
                    self._queue(terminal)
                for event in self._drain_events():
                    self._handle(event)
                if self.recorder.terminal_status:
                    self.status = f"EPISODE {self.recorder.terminal_status}; Q TO EXIT"
            return 0
        finally:
            self.close()

    def close(self) -> None:
        if self.closed and not hasattr(self, "cap"):
            return
        self.closed = True
        if self.inference_worker is not None and self.inference_worker.is_alive():
            print("[SHUTDOWN] waiting for the active I-lock inference worker")
            self.inference_worker.join()
        if self.worker is not None and self.worker.is_alive():
            print("[SHUTDOWN] waiting for the active motion/observation worker")
            self.worker.join()
        for arm in list(self.arms.values()):
            try:
                arm.close()
            except Exception:
                pass
        try:
            self.cap.release()
        except Exception:
            pass
        try:
            self.cv2.destroyAllWindows()
        except Exception:
            pass

    def _atexit(self) -> None:
        try:
            if hasattr(self, "recorder"):
                self.recorder.mark_incomplete()
        except Exception:
            pass


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Main-32 fast bottom VLA manual runtime: D50/D54/D55/D56/D58",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--mode", choices=("dry-run", "hover", "physical"), default="dry-run")
    p.add_argument("--collection-phase", choices=("motion_validation", "official"), default="motion_validation")
    p.add_argument("--d56-source", default=DEFAULT_D56_SOURCE)
    p.add_argument("--d50-source", default=DEFAULT_D50_SOURCE)
    p.add_argument("--d58-source", default=DEFAULT_D58_SOURCE)
    p.add_argument("--d50-basket-calib", default="/workspace/project_train/aruco_test/dual/basket_arm2_5point_affine.json")
    p.add_argument("--d54-source", default=DEFAULT_D54_SOURCE)
    p.add_argument("--d55-source", default=DEFAULT_D55_SOURCE)
    p.add_argument("--config", default=DEFAULT_CONFIG)
    p.add_argument("--hfile", default=DEFAULT_HFILE)
    p.add_argument(
        "--camera-calibration", default=DEFAULT_CALIBRATION,
        help="Original D54/D55 ELP calibration; D56 continues to use RAW frames",
    )
    p.add_argument(
        "--empty-board-raw-path",
        default="/workspace/project_train/aruco_test/dual/d34_empty_board.png",
        help="D56-45 RAW empty-board baseline written by E",
    )
    p.add_argument(
        "--empty-board-corrected-path",
        default="/workspace/project_train/aruco_test/dual/undistort/d34_empty_board_corrected.png",
        help="D54/D55 corrected empty-board baseline written by E",
    )
    p.add_argument("--camera-controls-json", default=DEFAULT_CAMERA_CONTROLS)
    p.add_argument("--camera-controls-enable", action="store_true", default=True)
    p.add_argument("--no-camera-controls-enable", dest="camera_controls_enable", action="store_false")
    p.add_argument("--camera-controls-strict", action="store_true", default=True)
    p.add_argument("--no-camera-controls-strict", dest="camera_controls_strict", action="store_false")
    p.add_argument("--camera-controls-stabilization-frames", type=int, default=12)
    p.add_argument("--dataset-root", default=DEFAULT_DATASET)
    p.add_argument("--board-image-size", type=int, default=512)
    p.add_argument("--seg-model", default=DEFAULT_SEG_MODEL)
    p.add_argument("--pose-model", default=DEFAULT_POSE_MODEL)
    p.add_argument("--camera", type=int, default=0)
    p.add_argument("--camera-device", default="/dev/video0")
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--backend", choices=("v4l2", "any"), default="v4l2")
    p.add_argument("--arm1-port", default="/dev/roarm_1")
    p.add_argument("--arm2-port", default="/dev/roarm_2")
    p.add_argument("--move-command", type=int, choices=(104, 1041), default=104)
    p.add_argument("--d54-pull-extra-mm", type=float, default=D54_PULL_EXTRA_MM)
    p.add_argument("--main28-d58-max-move-mm", type=float, default=160.0,
                   help="D58 maximum one-cycle low-Z circumcenter correction; original was 85mm")
    p.add_argument("--main33-d58-max-move-mm", type=float, default=180.0,
                   help="main-33 D58 strongest-safe one-cycle correction cap")
    p.add_argument("--main33-d58-min-effective-move-mm", type=float, default=120.0,
                   help="main-33 D58 requests at least this much correction before safety backoff")
    p.add_argument("--main33-d56-duration-s", type=float, default=4.07,
                   help="Exact FIX55/FIX56 2-D taught replay deadline")
    p.add_argument("--main28-d56-back-mm", type=float, default=110.0,
                   help="compatibility-only in main-33; real FIX55/FIX56 2-D trajectory is used")
    p.add_argument("--main28-d56-forward-mm", type=float, default=360.0,
                   help="compatibility-only in main-33; real FIX55/FIX56 2-D trajectory is used")
    p.add_argument("--main28-d56-min-forward-mm", type=float, default=160.0,
                   help="compatibility-only in main-33")
    p.add_argument("--main28-d56-duration-s", type=float, default=4.07,
                   help="Taught D56 reversal + forward laydown stream duration")
    p.add_argument("--main28-d56-lift-mm", type=float, default=380.0,
                   help="compatibility-only in main-33; FIX59 +18mm test lift and source canonical aerial Z are used")
    p.add_argument("--main28-d56-final-tolerance-mm", type=float, default=60.0,
                   help="Maximum 3-D support-endpoint error allowed before release")
    p.add_argument("--d55-consensus-frames", type=int, default=3)
    p.add_argument("--d55-consensus-interval-s", type=float, default=0.16)
    p.add_argument("--locked-plan-max-age-s", type=float, default=600.0)
    p.add_argument("--imaging-standby-wait-s", type=float, default=1.10)
    p.add_argument("--imaging-frame-flush-count", type=int, default=6)
    p.add_argument("--standby-feedback-timeout-s", type=float, default=1.4)
    p.add_argument("--standby-xyz-tolerance-mm", type=float, default=45.0)
    p.add_argument("--standby-z-tolerance-mm", type=float, default=35.0)
    p.add_argument("--standby-verify-retries", type=int, default=4)
    return p


def main() -> int:
    args = build_parser().parse_args()
    # All runtime files are fail-fast; no config/H is created or overwritten.
    args.config = _require_file(args.config, "folding-board config")
    args.hfile = _require_file(args.hfile, "Homography cache")
    args.camera_calibration = _require_file(args.camera_calibration, "D54/D55 camera calibration")
    args.camera_controls_json = _require_file(args.camera_controls_json, "camera controls JSON")
    args.seg_model = _require_file(args.seg_model, "bottom segmentation model")
    args.pose_model = _require_file(args.pose_model, "bottom pose model")
    args.d50_source = _require_file(args.d50_source, "D50 basket drop source")
    args.d50_basket_calib = _require_file(args.d50_basket_calib, "D50 basket five-point calibration")
    args.empty_board_raw_path = str(Path(args.empty_board_raw_path).expanduser().resolve())
    args.empty_board_corrected_path = str(Path(args.empty_board_corrected_path).expanduser().resolve())
    if args.empty_board_raw_path == args.empty_board_corrected_path:
        raise RuntimeError("raw and corrected empty-board paths must be different")
    app = BottomManualVLAApp(args)
    return app.run()


if __name__ == "__main__":
    raise SystemExit(main())
