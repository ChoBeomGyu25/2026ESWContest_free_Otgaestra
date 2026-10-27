import argparse
import copy
import hashlib
import importlib.util
import json
import math
import os
import select
import statistics
import sys
import threading
import time
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

BUILD = "2026-08-31-bottom-vla-v23"
SEMANTIC_ACTIONS = (
    "BASKET_GRASP",
    "POSITION_ADJUST",
    "OUTER_PULL",
    "PRESS_SWEEP",
    "WAIST_PULL_LAYDOWN",
    "ALIGN",
    "FINISH",
    "REJUDGE",
)
PHYSICAL_SEMANTICS = (
    "BASKET_GRASP",
    "POSITION_ADJUST",
    "OUTER_PULL",
    "PRESS_SWEEP",
    "WAIST_PULL_LAYDOWN",
    "ALIGN",
)
SEMANTIC_TO_INTERNAL = {
    "BASKET_GRASP": "BASKET_GRASP",
    "POSITION_ADJUST": "D58_CIRC_POSITION",
    "OUTER_PULL": "D54_OUTER_PULL",
    "PRESS_SWEEP": "D55_PRESS_SWEEP",
    "WAIST_PULL_LAYDOWN": "WAIST_PULL_LAYDOWN",
    "ALIGN": "ALIGN",
    "FINISH": "FINISH",
}
INTERNAL_TO_SEMANTIC = {v: k for k, v in SEMANTIC_TO_INTERNAL.items()}
BOTTOM_CLASSIFIER_ACTIONS = (
    "POSITION_ADJUST",
    "OUTER_PULL",
    "PRESS_SWEEP",
    "WAIST_PULL_LAYDOWN",
    "ALIGN",
    "FINISH",
)
BOTTOM_STATE_V2_NAMES = (
    "mask_valid",
    "mask_area_ratio",
    "mask_center_x_norm",
    "mask_center_y_norm",
    "board_center_error_norm",
    "mask_width_norm",
    "mask_height_norm",
    "mask_solidity",
    "mask_extent",
    "mask_pca_axis_angle_norm",
    "left_board_margin_norm",
    "right_board_margin_norm",
    "top_board_margin_norm",
    "bottom_board_margin_norm",
    "pose_valid",
    "pose_valid_keypoint_ratio",
    "waist_width_norm",
    "waist_axis_angle_norm",
    "body_axis_angle_norm",
    "waist_center_x_norm",
    "waist_center_y_norm",
    "leg_length_balance",
    "hem_width_balance",
    "pose_left_right_symmetry_error_norm",
    "waist_target_y_error_norm",
    "wrinkle_edge_ratio",
    "wrinkle_component_count_norm",
    "dominant_wrinkle_length_norm",
    "dominant_wrinkle_linearity_norm",
    "wrinkle_vs_body_axis_angle_norm",
)
DEFAULT_SEG_MODEL = "/workspace/project_train/aruco_test/dual/models/kfashion_yolo26s_seg3_e100_best.engine"
DEFAULT_POSE_MODEL = "/workspace/project_train/yolo26/bottom_pose8_beige_finetune_v2_best.engine"


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _resolve(explicit: str, candidates: List[str], label: str) -> str:
    tried: List[str] = []
    if explicit:
        candidates = [explicit] + candidates
    for raw in candidates:
        p = Path(raw).expanduser()
        if not p.is_absolute():
            p = Path(__file__).resolve().parent / p
        p = p.resolve()
        tried.append(str(p))
        if p.is_file():
            return str(p)
    raise FileNotFoundError(f"{label} not found: {tried}")


def _load_module(name: str, path: str) -> ModuleType:
    p = Path(path).resolve()
    for d in (p.parent, p.parent.parent, p.parent / "undistort", p.parent.parent / "undistort"):
        if d.is_dir() and str(d) not in sys.path:
            sys.path.insert(0, str(d))
    spec = importlib.util.spec_from_file_location(name, str(p))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {name}: {p}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        if isinstance(value, float) and not np.isfinite(value):
            return None
        return value
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items() if not str(k).startswith("_runtime")}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(v) for v in value]
    if hasattr(value, "__dict__"):
        return _json_safe(vars(value))
    return str(value)


def _atomic_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    with open(temp, "w", encoding="utf-8") as f:
        json.dump(_json_safe(payload), f, ensure_ascii=False, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(temp, path)


def _patch_json_record(record: Optional[Dict[str, Any]], fields: Dict[str, Any]) -> None:
    if not isinstance(record, dict):
        return
    raw = record.get("fs_path") or record.get("path") or record.get("json_path")
    if not raw:
        return
    p = Path(str(raw))
    if p.is_dir():
        p = p / "data.json"
    if not p.is_file():
        return
    try:
        with open(p, "r", encoding="utf-8") as f:
            payload = json.load(f)
        payload.update(_json_safe(fields))
        _atomic_json(p, payload)
    except Exception as exc:
        print(f"[DATA-PATCH-WARN] {p}: {exc!r}")


def _existing_user_option(argv: List[str], name: str) -> bool:
    return any(x == name or x.startswith(name + "=") for x in argv)


def _set_if_present(ns: argparse.Namespace, name: str, value: Any) -> None:
    if hasattr(ns, name):
        setattr(ns, name, value)


def _get_plan_reason(plan: Any, fallback: str = "") -> str:
    return str(getattr(plan, "reason", fallback) or fallback)


def _compact_plan(plan: Any) -> Dict[str, Any]:
    if plan is None:
        return {}
    if isinstance(plan, dict):
        return _json_safe(plan)
    out: Dict[str, Any] = {}
    for name in ("ok", "reason", "action", "metrics", "arm_points"):
        if hasattr(plan, name):
            out[name] = _json_safe(getattr(plan, name))
    return out


def _parse_front(argv: List[str]) -> Tuple[argparse.Namespace, List[str]]:
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--base-main", default="")
    p.add_argument("--d60-source", default="")
    p.add_argument("--position-source", default="")
    p.add_argument("--align-source", default="")
    p.add_argument("--auto-prepare-next", dest="auto_prepare_next", action="store_true", default=True)
    p.add_argument("--no-auto-prepare-next", dest="auto_prepare_next", action="store_false")
    p.add_argument("--align-finish-angle-deg", type=float, default=7.0)
    p.add_argument("--align-waist-target-y-mm", type=float, default=-110.0)
    p.add_argument("--align-dual-waist-min-pull-mm", type=float, default=55.0)
    p.add_argument("--press-sweep-max-normal-error-deg", type=float, default=25.0)
    p.add_argument("--d60-prepare-attempts", type=int, default=3)
    p.add_argument("--d60-waist-grip-body-min-mm", type=float, default=12.0)
    p.add_argument("--d60-waist-grip-body-max-mm", type=float, default=60.0)
    p.add_argument("--d60-waist-grip-tangent-max-mm", type=float, default=45.0)
    p.add_argument("--d60-waist-grip-endpoint-radius-max-mm", type=float, default=80.0)
    p.add_argument("--d60-pose-waist-min-conf", type=float, default=0.18)
    p.add_argument("--d60-pose-waist-min-width-mm", type=float, default=80.0)
    p.add_argument("--d60-pose-waist-max-width-mm", type=float, default=360.0)
    p.add_argument("--d60-waist-hard-max-width-mm", type=float, default=360.0)
    p.add_argument("--d60-pose-waist-mask-near-px", type=float, default=55.0)
    p.add_argument("--arm1-standby-z-offset-mm", type=float, default=-20.0)
    p.add_argument("--basket-calib", default="")
    p.add_argument("--basket-hover-offset-mm", type=float, default=30.0)
    p.add_argument("--basket-rim-clearance-mm", type=float, default=30.0)
    p.add_argument("--basket-transit-clearance-mm", type=float, default=100.0)
    p.add_argument("--basket-floor-z", type=float, default=-325.8368564)
    p.add_argument("--basket-floor-clearance-mm", type=float, default=15.0)
    p.add_argument("--basket-fast-step-mm", type=float, default=20.0)
    p.add_argument("--basket-fast-speed", type=float, default=0.90)
    p.add_argument("--basket-slow-step-mm", type=float, default=5.0)
    p.add_argument("--basket-slow-speed", type=float, default=0.40)
    p.add_argument("--basket-move-speed", type=float, default=1.12)
    p.add_argument("--basket-descent-speed", type=float, default=0.35)
    p.add_argument("--basket-move-timeout-s", type=float, default=15.0)
    p.add_argument("--basket-move-tolerance-mm", type=float, default=25.0)
    p.add_argument("--basket-move-poll-s", type=float, default=0.30)
    p.add_argument("--basket-feedback-timeout-s", type=float, default=2.5)
    p.add_argument("--basket-probe-step-timeout-s", type=float, default=5.0)
    p.add_argument("--basket-probe-z-tolerance-mm", type=float, default=12.0)
    p.add_argument("--basket-probe-z-stable-span-mm", type=float, default=2.0)
    p.add_argument("--basket-lift-z-tolerance-mm", type=float, default=25.0)
    p.add_argument("--basket-lift-stall-polls", type=int, default=5)
    p.add_argument("--basket-lift-stall-span-mm", type=float, default=1.0)
    p.add_argument("--basket-grip-open-percent", type=float, default=30.0)
    p.add_argument("--basket-post-contact-open-percent", type=float, default=90.0)
    p.add_argument("--basket-grip-fully-open", type=float, default=1.35)
    p.add_argument("--basket-close-target", type=float, default=3.05)
    p.add_argument("--basket-final-close-target", type=float, default=3.32)
    p.add_argument("--basket-final-latch-torque", type=int, default=1000)
    p.add_argument("--basket-final-latch-settle-s", type=float, default=0.35)
    p.add_argument("--basket-close-tolerance-rad", type=float, default=0.18)
    p.add_argument("--basket-close-attempts", type=int, default=5)
    p.add_argument("--basket-release-target", type=float, default=1.35)
    p.add_argument("--basket-release-tolerance-rad", type=float, default=0.22)
    p.add_argument("--basket-release-attempts", type=int, default=5)
    p.add_argument("--basket-gripper-settle-s", type=float, default=1.2)
    p.add_argument("--basket-post-contact-open-settle-s", type=float, default=0.8)
    p.add_argument("--basket-baseline-samples", type=int, default=7)
    p.add_argument("--basket-baseline-interval-s", type=float, default=0.10)
    p.add_argument("--basket-contact-shoulder-delta", type=float, default=40.0)
    p.add_argument("--basket-contact-elbow-delta", type=float, default=20.0)
    p.add_argument("--basket-contact-z-lag-mm", type=float, default=2.5)
    p.add_argument("--basket-contact-confirm-steps", type=int, default=2)
    p.add_argument("--basket-hard-axis-delta", type=float, default=300.0)
    p.add_argument("--basket-fast-hard-se-delta", type=float, default=140.0)
    p.add_argument("--basket-stall-min-command-mm", type=float, default=4.0)
    p.add_argument("--basket-stall-max-actual-mm", type=float, default=1.5)
    p.add_argument("--basket-stall-confirm-steps", type=int, default=2)
    p.add_argument("--basket-pickup-lift-z", type=float, default=180.0)
    p.add_argument("--basket-lift-speed", type=float, default=0.95)
    p.add_argument("--basket-board-transit-speed", type=float, default=0.95)
    p.add_argument("--basket-placement-rotate-speed", type=float, default=0.75)
    p.add_argument("--basket-board-center-blend-mm", type=float, default=70.0)
    p.add_argument("--basket-placement-extra-deg", type=float, default=30.0)
    p.add_argument("--basket-placement-lower-mm", type=float, default=180.0)
    p.add_argument("--basket-placement-lower-speed", type=float, default=0.25)
    p.add_argument("--basket-placement-min-z", type=float, default=-80.0)
    p.add_argument("--basket-retention-threshold", type=float, default=220.0)
    p.add_argument("--basket-retention-samples", type=int, default=3)
    p.add_argument("--basket-retention-interval-s", type=float, default=0.12)
    p.add_argument("--basket-arm2-standby-x", type=float, default=2.870034)
    p.add_argument("--basket-arm2-standby-y", type=float, default=-233.859636)
    p.add_argument("--basket-arm2-standby-z", type=float, default=102.23829)
    p.add_argument("--basket-arm2-standby-t", type=float, default=1.356039)
    p.add_argument("--basket-standby-speed", type=float, default=0.24)
    return p.parse_known_args(argv)


def _resolve_sources(front: argparse.Namespace) -> Dict[str, str]:
    base = _resolve(
        front.base_main,
        [
            "main-33.py",
            "/workspace/project_train/aruco_test/dual/undistort/main-33.py",
            "/workspace/project_train/aruco_test/dual/vla/v2/main-33.py",
            "/workspace/project_train/aruco_test/dual/vla/main-33.py",
        ],
        "main-33.py",
    )
    d60 = _resolve(
        front.d60_source,
        [
            "60-15.py",
            "/workspace/project_train/aruco_test/dual/undistort/60-15.py",
        ],
        "60-15.py",
    )
    position = _resolve(
        front.position_source,
        [
            "58-3.py",
            "/workspace/project_train/aruco_test/dual/undistort/58-3.py",
        ],
        "58-3.py",
    )
    align = _resolve(
        front.align_source,
        [
            "align-11.py",
            "/workspace/project_train/aruco_test/dual/undistort/align-11.py",
        ],
        "align-11.py",
    )
    return {"base": base, "d60": d60, "position": position, "align": align}


def _build_runtime(front: argparse.Namespace, remaining: List[str], source_paths: Dict[str, str]):
    base = _load_module("bottom_vla_base_main33", source_paths["base"])
    base.PHYSICAL_ACTIONS = (
        "BASKET_GRASP",
        "D58_CIRC_POSITION",
        "D54_OUTER_PULL",
        "D55_PRESS_SWEEP",
        "WAIST_PULL_LAYDOWN",
        "ALIGN",
    )
    if hasattr(base, "MOTION_VERSION"):
        base.MOTION_VERSION = BUILD
    if hasattr(base, "MOTION_POLICY_VERSION"):
        base.MOTION_POLICY_VERSION = BUILD + "-frozen-plan"
    parser = base.build_parser()
    args = parser.parse_args(remaining)
    args.d56_source = source_paths["d60"]
    if hasattr(args, "d58_source"):
        args.d58_source = source_paths["position"]
    if not _existing_user_option(remaining, "--seg-model"):
        args.seg_model = DEFAULT_SEG_MODEL
    if not _existing_user_option(remaining, "--pose-model"):
        args.pose_model = DEFAULT_POSE_MODEL
    if hasattr(args, "d55_consensus_frames"):
        args.d55_consensus_frames = 1
    align_mod = _load_module("bottom_vla_align11", source_paths["align"])
    return base, align_mod, args


def _module_args(base: ModuleType, module: ModuleType, common: argparse.Namespace, mode: str) -> argparse.Namespace:
    preset = "hover-check" if mode == "hover" else "physical-auto"
    ns = base._module_default_args(module, preset=preset)
    base._copy_shared_args(ns, common, mode)
    if hasattr(module, "_d26_prepare_d25_args"):
        ns = module._d26_prepare_d25_args(ns)
    ns.send = mode == "physical"
    ns.dry_run = mode != "physical"
    ns.hover_only = mode == "hover"
    ns.enter_confirm = False
    ns.auto_reinfer_after_motion = False
    if hasattr(ns, "d17_auto_loop"):
        ns.d17_auto_loop = False
    return ns


class _GentleGripperProxy:
    def __init__(self, arm: Any, approach_min: float, release_min: float, label: str, start_angle: Optional[float] = None, max_open_step: float = 0.20, step_wait: float = 0.12):
        self._arm = arm
        self._approach_min = float(approach_min)
        self._release_min = float(release_min)
        self._label = str(label)
        self._last_angle = None if start_angle is None else float(start_angle)
        self._max_open_step = max(0.05, float(max_open_step))
        self._step_wait = max(0.02, float(step_wait))

    def __getattr__(self, name: str) -> Any:
        return getattr(self._arm, name)

    def _tag(self, args: Tuple[Any, ...], kwargs: Dict[str, Any]) -> str:
        parts = [str(kwargs.get("stage", "")), str(kwargs.get("caller", ""))]
        parts.extend(str(x) for x in args if isinstance(x, str))
        return " ".join(parts).upper()

    def _target(self, value: Any, tag: str) -> float:
        v = float(value)
        target = v
        if "RELEASE" in tag:
            target = max(v, self._release_min)
        elif "OPEN" in tag:
            target = max(v, self._approach_min)
        if abs(target - v) > 1e-9:
            print(f"[GRIP-OPEN-LIMIT] {self._label} {tag[:68]} {v:.3f}->{target:.3f}")
        return target

    def _stages(self, target: float, tag: str) -> List[float]:
        current = self._last_angle
        if current is None:
            current = 3.05
        if "OPEN" not in tag and "RELEASE" not in tag:
            return [float(target)]
        if target >= current - 1e-6:
            return [float(target)]
        values: List[float] = []
        value = float(current)
        while value - target > self._max_open_step + 1e-9:
            value -= self._max_open_step
            values.append(float(value))
        if not values or abs(values[-1] - target) > 1e-9:
            values.append(float(target))
        return values

    def set_gripper(self, angle_rad: float, *args: Any, **kwargs: Any) -> Any:
        tag = self._tag(args, kwargs)
        target = self._target(angle_rad, tag)
        stages = self._stages(target, tag)
        if len(stages) > 1:
            print(f"[GRIP-OPEN-GENTLE] {self._label} {self._last_angle if self._last_angle is not None else 3.05:.3f}->{target:.3f} steps={len(stages)} step<={self._max_open_step:.2f}rad wait={self._step_wait:.2f}s")
        result = None
        original_delay = kwargs.get("delay", None)
        for index, value in enumerate(stages):
            call_kwargs = dict(kwargs)
            if index + 1 < len(stages):
                call_kwargs["delay"] = 0.0
            elif original_delay is not None:
                call_kwargs["delay"] = original_delay
            result = self._arm.set_gripper(float(value), *args, **call_kwargs)
            self._last_angle = float(value)
            if index + 1 < len(stages):
                time.sleep(self._step_wait)
        return result

    def send(self, cmd: Dict[str, Any], *args: Any, **kwargs: Any) -> Any:
        payload = dict(cmd) if isinstance(cmd, dict) else cmd
        if not isinstance(payload, dict) or int(payload.get("T", -1)) != 106 or "cmd" not in payload:
            return self._arm.send(payload, *args, **kwargs)
        tag = self._tag(args, kwargs)
        target = self._target(payload["cmd"], tag)
        stages = self._stages(target, tag)
        if len(stages) > 1:
            print(f"[GRIP-OPEN-GENTLE] {self._label} {self._last_angle if self._last_angle is not None else 3.05:.3f}->{target:.3f} steps={len(stages)} step<={self._max_open_step:.2f}rad wait={self._step_wait:.2f}s")
        result = None
        original_delay = kwargs.get("delay", None)
        for index, value in enumerate(stages):
            call_kwargs = dict(kwargs)
            if index + 1 < len(stages):
                call_kwargs["delay"] = 0.0
            elif original_delay is not None:
                call_kwargs["delay"] = original_delay
            stage_payload = dict(payload)
            stage_payload["cmd"] = float(value)
            result = self._arm.send(stage_payload, *args, **call_kwargs)
            self._last_angle = float(value)
            if index + 1 < len(stages):
                time.sleep(self._step_wait)
        return result


def _make_app_class(base: ModuleType, align_mod: ModuleType, front: argparse.Namespace, source_paths: Dict[str, str]):
    class BottomVLAApp(base.BottomManualVLAApp):
        def __init__(self, args: argparse.Namespace):
            self.bottom_vla_build = BUILD
            self.bottom_vla_sources = copy.deepcopy(source_paths)
            self.semantic_selected: Optional[str] = None
            self.auto_recommended = "BASKET_GRASP"
            self.plan_origin = "HUMAN"
            self.pending_semantic: Optional[str] = None
            self.last_executed_semantic: Optional[str] = None
            self.last_result: Optional[str] = None
            self.align_phase = "DUAL_WAIST_TOP"
            self.align_correction_count = 0
            self.align_last_decision: Dict[str, Any] = {}
            self.align_runtime: Dict[str, Dict[str, Any]] = {}
            self.align_runtime_lock = threading.RLock()
            self.d60_runtime: Dict[str, Dict[str, Any]] = {}
            self.d60_runtime_lock = threading.RLock()
            self.front = front
            self._panel_rects: List[Tuple[str, Tuple[int, int, int, int], Tuple[int, int, int]]] = []
            self._prepare_origin = "HUMAN"
            self._human_selected = None
            self._auto_at_selection = "BASKET_GRASP"
            self._prepare_generation = 0
            self._active_prepare_generation = 0
            self._queued_human_action: Optional[str] = None
            self._queued_human_origin = "HUMAN"
            self._classifier_manifest_lock = threading.RLock()
            super().__init__(args)
            self.align = align_mod
            self.align_args = _module_args(base, self.align, self.args, self.args.mode)
            self.align_cfg = self.align.make_safety_config_from_args(self.align_args, self.config)
            self.arm1_standby_z_offset_mm = float(self.front.arm1_standby_z_offset_mm)
            self._apply_arm1_standby_z_offset()
            self.align_source_sha256 = _sha256(source_paths["align"])
            self.d60_source_sha256 = _sha256(source_paths["d60"])
            self.position_source_sha256 = _sha256(source_paths["position"])
            self.d54_args.d54_approach_open = 2.35
            self.d54_args.d54_release_open = 1.90
            self.d56_args.d31_grip_open = 2.45
            self.d56_args.step68_final_release_angle = 1.90
            self.d56_args.step60_pullup_speed = 1.00
            self.align_grip_approach_min = 2.30
            self.align_grip_release_min = 1.90
            self.gentle_open_step_rad = 0.20
            self.gentle_open_wait_s = 0.12
            required_d60 = ("_63_step120_motion_plan_from_d56_plan", "_63_execute_step120_motion_from_d56_plan")
            missing_d60 = [name for name in required_d60 if not callable(getattr(self.d56, name, None))]
            if missing_d60:
                raise RuntimeError(f"60-15 direct executor missing: {missing_d60}")
            self._ensure_d60_arm_api()
            self._install_d60_waist_width_guard()
            self._install_d58_no_grip_standby()
            # V22 D58 release policy:
            # - do NOT fully open to 1.35 after the drag
            # - release only to 1.90
            # - wait for jaw motion to finish before retract/standby
            if hasattr(self, "d58_args"):
                if hasattr(self.d58_args, "release_open"):
                    self.d58_args.release_open = 1.90
                if hasattr(self.d58_args, "release_wait_s"):
                    self.d58_args.release_wait_s = max(1.20, float(self.d58_args.release_wait_s))
            self._install_d54_gentle_gripper()
            self._install_d55_perpendicular_policy()
            self._write_bottom_vla_manifest()
            print(f"[BOTTOM-VLA] build={BUILD}")
            print(f"[BOTTOM-VLA] D60={source_paths['d60']} sha256={self.d60_source_sha256}")
            print(f"[BOTTOM-VLA] POSITION_ADJUST={source_paths['position']} sha256={self.position_source_sha256}")
            print(f"[BOTTOM-VLA] ALIGN={source_paths['align']} sha256={self.align_source_sha256}")
            print("[GRIP-OPEN-LIMIT] D54 approach=2.35 release=1.90 | D60 approach=2.45 release=1.90 | ALIGN approach=2.30 release=1.90")
            print(f"[GRIP-OPEN-GENTLE] D54/D60/ALIGN max_step={self.gentle_open_step_rad:.2f}rad wait={self.gentle_open_wait_s:.2f}s; BASKET unchanged")
            print("[D60-PULLUP] outbound/gravity-settle -> +50mm pure-Z closed lift -> same-grip pull-up speed=1.00")
            print(f"[D55-NORMAL-POLICY] final clipped sweep angle=90+/-{float(self.front.press_sweep_max_normal_error_deg):.0f}deg; generic outward partner disabled")
            print(f"[ALIGN-CURRENT-STATE] waist targetY={float(self.front.align_waist_target_y_mm):.1f}mm dual trigger pull>={float(self.front.align_dual_waist_min_pull_mm):.1f}mm")
            print(f"[BASKET-CENTER-BLEND] switch to final {abs(float(self.front.basket_placement_extra_deg)):.0f}deg target when center error<={float(self.front.basket_board_center_blend_mm):.0f}mm")
            print(f"[BASKET-ROTATE-SPEED] final {abs(float(self.front.basket_placement_extra_deg)):.0f}deg segment speed={float(self.front.basket_placement_rotate_speed):.2f}")
            print(f"[BASKET-PLACE-LOWER] after rotation lower={float(self.front.basket_placement_lower_mm):.0f}mm speed={float(self.front.basket_placement_lower_speed):.2f} minZ={float(self.front.basket_placement_min_z):.1f} before release")
            print(f"[D60-WAIST-END-GATE] attempts={int(max(1, self.front.d60_prepare_attempts))} body={float(self.front.d60_waist_grip_body_min_mm):.0f}..{float(self.front.d60_waist_grip_body_max_mm):.0f}mm tangent<={float(self.front.d60_waist_grip_tangent_max_mm):.0f}mm radius<={float(self.front.d60_waist_grip_endpoint_radius_max_mm):.0f}mm")
            print(f"[D60-POSE-WAIST-FIRST] minConf={float(self.front.d60_pose_waist_min_conf):.2f} width={float(self.front.d60_pose_waist_min_width_mm):.0f}..{float(self.front.d60_pose_waist_max_width_mm):.0f}mm maskNear<={float(self.front.d60_pose_waist_mask_near_px):.0f}px")
            print(f"[D60-WAIST-LENGTH-GUARD] pose/ribbon/fallback hardMax={float(self.front.d60_waist_hard_max_width_mm):.0f}mm; overlong hypothesis -> NO PLAN/REJUDGE")
            print(f"[BOTTOM-STATE-V2] dim={len(BOTTOM_STATE_V2_NAMES)} action-independent common state enabled; classifier labels={list(BOTTOM_CLASSIFIER_ACTIONS)}")
            print("[D58-STANDBY] post-action standby is T104-only; no gripper command is issued")
            print("[D58-RELEASE-V22] after drag release=1.90rad, settle>=1.20s, then retract/standby; no T106 during retract/standby")

        def _install_d60_waist_width_guard(self) -> None:
            original = getattr(self, "_d60_plan_from_obs", None)
            if not callable(original):
                raise RuntimeError("D60 waist planner hook unavailable")
            if bool(getattr(original, "_bottom_vla_v23_waist_guard", False)):
                return

            hard_max = float(self.front.d60_waist_hard_max_width_mm)

            def finite_number(value):
                try:
                    v = float(value)
                except Exception:
                    return None
                return v if math.isfinite(v) else None

            def collect_widths(obj, prefix=""):
                rows = []
                if isinstance(obj, dict):
                    for k, v in obj.items():
                        key = str(k)
                        low = key.lower()
                        path = f"{prefix}.{key}" if prefix else key
                        num = finite_number(v)
                        if num is not None and (
                            ("waist" in low and "width" in low)
                            or ("waist" in low and "chord" in low)
                            or ("waist" in low and "arc" in low)
                        ):
                            rows.append((path, abs(num)))
                        elif isinstance(v, dict):
                            rows.extend(collect_widths(v, path))
                return rows

            def endpoint_width(metrics):
                pairs = (
                    ("waist_left_board", "waist_right_board"),
                    ("waist_left", "waist_right"),
                    ("left_waist_board", "right_waist_board"),
                )
                for lk, rk in pairs:
                    if lk not in metrics or rk not in metrics:
                        continue
                    try:
                        a = np.asarray(metrics[lk], dtype=np.float64).reshape(-1)[:2]
                        b = np.asarray(metrics[rk], dtype=np.float64).reshape(-1)[:2]
                        if a.size == 2 and b.size == 2 and np.all(np.isfinite(a)) and np.all(np.isfinite(b)):
                            return float(np.linalg.norm(b - a)), f"{lk}<->{rk}"
                    except Exception:
                        pass
                return None, ""

            def wrapped(obs):
                plan = original(obs)
                if plan is None or not bool(getattr(plan, "ok", False)):
                    return plan

                metrics = dict(getattr(plan, "metrics", {}) or {})
                widths = collect_widths(metrics)
                ew, esrc = endpoint_width(metrics)
                if ew is not None:
                    widths.append((esrc, ew))

                reason_text = str(getattr(plan, "reason", "") or "")
                import re
                for m in re.finditer(r"(?:waist\s*)?width\s*[=:]\s*([0-9]+(?:\.[0-9]+)?)\s*mm", reason_text, re.I):
                    try:
                        widths.append(("reason.width", float(m.group(1))))
                    except Exception:
                        pass

                if not widths:
                    return plan

                source, width = max(widths, key=lambda item: float(item[1]))
                if float(width) <= hard_max:
                    return plan

                print(
                    f"[D60-WAIST-LENGTH-REJECT] source={source} width={float(width):.1f}mm "
                    f"> hardMax={hard_max:.1f}mm; reject abnormal waist hypothesis"
                )
                cls = type(plan)
                try:
                    blocked = cls(
                        False,
                        f"D60 abnormal waist width {float(width):.1f}mm > {hard_max:.1f}mm",
                    )
                except Exception:
                    try:
                        blocked = copy.deepcopy(plan)
                        blocked.ok = False
                        blocked.reason = f"D60 abnormal waist width {float(width):.1f}mm > {hard_max:.1f}mm"
                    except Exception:
                        return None

                try:
                    blocked.metrics = dict(getattr(blocked, "metrics", {}) or {})
                    blocked.metrics.update({
                        "d60_v23_abnormal_waist_rejected": True,
                        "d60_v23_waist_width_mm": float(width),
                        "d60_v23_waist_width_source": str(source),
                        "d60_v23_waist_hard_max_mm": float(hard_max),
                    })
                except Exception:
                    pass
                return blocked

            wrapped._bottom_vla_v23_waist_guard = True
            self._d60_plan_from_obs = wrapped
            print(
                f"[D60-WAIST-LENGTH-GUARD] hardMax={hard_max:.0f}mm; "
                "Pose/ribbon observer preserved, abnormally long waist rejected before freeze"
            )

        def _apply_arm1_standby_z_offset(self) -> None:
            offset = float(self.arm1_standby_z_offset_mm)
            seen = set()
            values = []
            for obj in (getattr(self, "cfg56", None), getattr(self, "cfg54", None), getattr(self, "cfg55", None), getattr(self, "align_cfg", None)):
                if obj is None or id(obj) in seen or not hasattr(obj, "arm1_standby_roarm_z"):
                    continue
                seen.add(id(obj))
                old = float(getattr(obj, "arm1_standby_roarm_z"))
                new = old + offset
                setattr(obj, "arm1_standby_roarm_z", new)
                values.append((old, new))
            for obj in (getattr(self, "d56_args", None), getattr(self, "d54_args", None), getattr(self, "d55_args", None), getattr(self, "align_args", None)):
                if obj is None or not hasattr(obj, "arm1_standby_roarm_z"):
                    continue
                try:
                    old = float(getattr(obj, "arm1_standby_roarm_z"))
                    setattr(obj, "arm1_standby_roarm_z", old + offset)
                except Exception:
                    pass
            if values:
                old, new = values[0]
                print(f"[ARM1-STANDBY-Z] {old:.1f}->{new:.1f}mm offset={offset:+.1f}mm")

        def _install_d58_no_grip_standby(self) -> None:
            original = getattr(self.d56, "move_arms_to_standby", None)
            pose_fn = getattr(self.d56, "standby_roarm_pose", None)
            if not callable(original) or not callable(pose_fn):
                print("[D58-STANDBY-NO-GRIP] hook unavailable")
                return
            if bool(getattr(original, "_bottom_vla_d58_no_grip", False)):
                return

            def wrapped(arms, cfg, move_command=104, reason="standby"):
                reason_text = str(reason).upper()
                is_d58_standby = (
                    "D58_CIRC_POSITION" in reason_text
                    or "D58" in reason_text
                    or "POSITION_ADJUST" in reason_text
                )
                if not is_d58_standby:
                    return original(arms, cfg, move_command=move_command, reason=reason)
                if bool(getattr(cfg, "dry_run", False)) or not bool(getattr(cfg, "send", True)):
                    return False
                speed = float(getattr(self.d56_args, "d26v3_return_speed", 0.24))
                sent = False
                for key in ("arm2", "arm1"):
                    arm = arms.get(key) if isinstance(arms, dict) else None
                    if arm is None:
                        continue
                    x, y, z, t = pose_fn(cfg, key)
                    print(f"[D58-STANDBY-NO-GRIP] {key.upper()} T104-only target=({x:.1f},{y:.1f},{z:.1f}) speed={speed:.2f}; no T106")
                    try:
                        arm.move_goal(float(x), float(y), float(z), float(t), speed, move_command=int(move_command), stage="D58_STANDBY_T104_ONLY")
                    except TypeError:
                        arm.move_goal(float(x), float(y), float(z), float(t), speed, move_command=int(move_command))
                    sent = True
                return sent

            wrapped._bottom_vla_d58_no_grip = True
            self.d56.move_arms_to_standby = wrapped

        def _read_gripper_angle(self, module: ModuleType, arm: Any) -> Optional[float]:
            query = getattr(module, "_d31_query_feedback", None)
            if not callable(query) or arm is None:
                return None
            try:
                fb = query(arm, timeout_s=0.75)
            except TypeError:
                try:
                    fb = query(arm)
                except Exception:
                    return None
            except Exception:
                return None
            if not isinstance(fb, dict):
                return None
            try:
                value = float(fb.get("t"))
            except Exception:
                return None
            return value if math.isfinite(value) else None

        def _gentle_arms(self, arms: Dict[str, Any], module: ModuleType, approach_min: float, release_min: float, prefix: str) -> Dict[str, Any]:
            wrapped: Dict[str, Any] = {}
            for key, arm in arms.items():
                if arm is None:
                    wrapped[key] = None
                    continue
                start = self._read_gripper_angle(module, arm)
                if start is not None:
                    print(f"[GRIP-GENTLE-START] {prefix}-{key.upper()} actual={start:.3f}")
                wrapped[key] = _GentleGripperProxy(
                    arm, approach_min, release_min, f"{prefix}-{key.upper()}",
                    start_angle=start, max_open_step=self.gentle_open_step_rad,
                    step_wait=self.gentle_open_wait_s,
                )
            return wrapped

        def _install_d54_gentle_gripper(self) -> None:
            original = getattr(self.d54, "_d51v4_execute_diagonal_pull", None)
            if not callable(original):
                print("[GRIP-OPEN-GENTLE] D54 executor hook unavailable")
                return
            if bool(getattr(original, "_bottom_vla_gentle_open", False)):
                return

            def wrapped(plan, arms, config, cfg, args, on_verified_start=None):
                gentle = self._gentle_arms(arms, self.d54, 2.35, 1.90, "D54")
                return original(plan, gentle, config, cfg, args, on_verified_start=on_verified_start)

            wrapped._bottom_vla_gentle_open = True
            self.d54._d51v4_execute_diagonal_pull = wrapped

        def _d55_perpendicular_report(self, plan: Any) -> Tuple[bool, Dict[str, Any]]:
            max_error = float(np.clip(float(self.front.press_sweep_max_normal_error_deg), 0.0, 45.0))
            rows: List[Dict[str, Any]] = []
            ok = True
            moving = 0
            arm_points = dict(getattr(plan, "arm_points", {}) or {}) if plan is not None else {}
            for arm_key in ("arm2", "arm1"):
                points = arm_points.get(arm_key)
                if not isinstance(points, dict):
                    rows.append({"arm": arm_key, "ok": False, "reason": "MISSING_ARM"})
                    ok = False
                    continue
                try:
                    src = np.asarray(points.get("source_board"), np.float64).reshape(2)
                    dst = np.asarray(points.get("target_board"), np.float64).reshape(2)
                except Exception:
                    rows.append({"arm": arm_key, "ok": False, "reason": "INVALID_SOURCE_TARGET"})
                    ok = False
                    continue
                move = dst - src
                move_mm = float(np.linalg.norm(move))
                role = str(points.get("role", ""))
                stationary = bool(move_mm <= 1.0 and ("support" in role.lower() or "anchor" in role.lower()))
                if stationary:
                    rows.append({"arm": arm_key, "ok": True, "stationary_support": True, "move_mm": move_mm, "role": role})
                    continue
                moving += 1
                try:
                    tangent = np.asarray(points.get("wrinkle_tangent_board"), np.float64).reshape(2)
                    tangent_n = float(np.linalg.norm(tangent))
                except Exception:
                    tangent_n = 0.0
                    tangent = np.zeros(2, dtype=np.float64)
                if move_mm <= 1e-6 or tangent_n <= 1e-6:
                    rows.append({"arm": arm_key, "ok": False, "reason": "NO_VALID_WRINKLE_TANGENT", "move_mm": move_mm, "role": role})
                    ok = False
                    continue
                move_u = move / move_mm
                tangent_u = tangent / tangent_n
                dot = float(np.clip(np.dot(tangent_u, move_u), -1.0, 1.0))
                angle = float(math.degrees(math.acos(dot)))
                normal_error = abs(90.0 - angle)
                arm_ok = bool(normal_error <= max_error)
                rows.append({
                    "arm": arm_key,
                    "ok": arm_ok,
                    "stationary_support": False,
                    "move_mm": move_mm,
                    "role": role,
                    "angle_to_wrinkle_tangent_deg": angle,
                    "normal_error_deg": normal_error,
                    "allowed_angle_min_deg": 90.0 - max_error,
                    "allowed_angle_max_deg": 90.0 + max_error,
                })
                ok = ok and arm_ok
            ok = bool(ok and moving >= 1 and len(arm_points) == 2)
            return ok, {
                "ok": ok,
                "policy": "BOTTOM_VLA_V14_POST_CLIP_WRINKLE_NORMAL",
                "max_normal_error_deg": max_error,
                "allowed_angle_deg": [90.0 - max_error, 90.0 + max_error],
                "moving_arm_count": moving,
                "arms": rows,
            }

        def _d55_build_perpendicular_plan(self, obs: Any, heat: Any, H: Any, config: Any, cfg: Any, args: Any) -> Any:
            module = self.d55
            if heat is None:
                return module.DualWrinkleStretchPlan(False, "D55-V14 NO_WRINKLE: heatmap unavailable", metrics={"d55_failure_category": "NO_WRINKLE"})
            candidates = [
                c for c in list(getattr(heat, "candidates", []) or [])
                if str(c.get("d55v5_class", "DETACHED_WRINKLE")) == "DETACHED_WRINKLE"
            ]
            if not candidates:
                waist_n = len(getattr(heat, "d55v5_waist_ignored_candidates", []) or [])
                category = "ONLY_WAIST_CONNECTED_CCA" if waist_n > 0 else "NO_DETACHED_WRINKLE"
                return module.DualWrinkleStretchPlan(False, f"D55-V14 {category}", metrics={"d55_failure_category": category, "waist_connected_count": waist_n})
            failures: List[Dict[str, Any]] = []
            dual_normal: List[Tuple[float, Any]] = []
            support: List[Tuple[float, Any]] = []
            max_trials = max(1, int(getattr(args, "d55_max_candidate_trials", 10)))
            for ci, cand in enumerate(candidates[:max_trials]):
                primary = module._d55v8_make_perpendicular_press_plan(obs, heat, cand, ci, H, config, cfg, args)
                primary, primary_guard = module._d55v11_apply_xy_only_mask_guard(primary, obs, H, config, cfg, args)
                primary_arm = module._d26v3_arm_of_plan(primary)
                if primary is None or primary_arm is None:
                    failures.append({"candidate": ci + 1, "reason": str(cand.get("_d55_plan_failure", "PRIMARY_UNAVAILABLE")), "guard": _json_safe(primary_guard)})
                    continue
                primary.arm_points[primary_arm]["role"] = "same_wrinkle_primary_normal_sweep"
                major_px = max(0.0, float(cand.get("major_length_px", 0.0) or 0.0))
                linearity = max(1.0, float(cand.get("linearity", 1.0) or 1.0))
                length_norm = float(np.clip(major_px / 150.0, 0.0, 1.65))
                line_norm = float(np.clip((linearity - 1.0) / 3.0, 0.0, 1.0))
                long_bonus = 2_400_000.0 * length_norm * (0.70 + 0.30 * line_norm)
                score = (
                    2_000_000.0 * int(cand.get("d20_priority_tier", 0))
                    + 700_000.0 * float(cand.get("d21_severity", 0.0))
                    + 180_000.0 * int(cand.get("d55_persistence_count", 1))
                    + float(cand.get("priority_score", 0.0))
                    + long_bonus
                )
                cand["bottom_vla_v16_long_wrinkle_bonus"] = float(long_bonus)
                cand["bottom_vla_v16_major_length_px"] = float(major_px)
                cand["bottom_vla_v16_linearity"] = float(linearity)
                missing = "arm1" if primary_arm == "arm2" else "arm2"
                secondary = module._d55v14_make_same_wrinkle_sweep(primary, cand, heat, obs, H, config, cfg, args, missing)
                if secondary is not None:
                    combined = module._d55v14_combine_same_candidate(primary, secondary, cand, score, "SAME_WRINKLE_NORMAL_SWEEP_SWEEP", args)
                    if combined is not None:
                        valid, report = self._d55_perpendicular_report(combined)
                        if valid:
                            combined.metrics["d55v15_partner_policy"] = "BOTH_MOVE_SAME_WRINKLE_NORMAL"
                            combined.metrics["bottom_vla_v14_perpendicular_guard"] = report
                            combined.metrics["bottom_vla_v14_generic_outward_assist"] = False
                            dual_normal.append((score, combined))
                            continue
                        failures.append({"candidate": ci + 1, "reason": "POST_CLIP_NORMAL_GUARD_SECONDARY", "report": report})
                anchor = module._d55v14_make_anchor(primary, obs, heat, H, config, cfg, args, missing)
                if anchor is not None:
                    combined = module._d55v14_combine_same_candidate(primary, anchor, cand, score, "SAME_WRINKLE_NORMAL_SWEEP_SUPPORT", args)
                    if combined is not None:
                        valid, report = self._d55_perpendicular_report(combined)
                        if valid:
                            combined.metrics["d55v15_partner_policy"] = "STATIONARY_SUPPORT_LAST_RESORT"
                            combined.metrics["bottom_vla_v14_perpendicular_guard"] = report
                            combined.metrics["bottom_vla_v14_generic_outward_assist"] = False
                            support.append((score, combined))
                            continue
                        failures.append({"candidate": ci + 1, "reason": "POST_CLIP_NORMAL_GUARD_SUPPORT", "report": report})
                failures.append({"candidate": ci + 1, "reason": "NO_SAFE_SAME_WRINKLE_NORMAL_TWO_ARM_PLAN"})
            if dual_normal:
                bucket = dual_normal
                bucket_name = "PREFER_SAME_WRINKLE_NORMAL_SWEEP_SWEEP"
            elif support:
                bucket = support
                bucket_name = "NORMAL_SWEEP_WITH_STATIONARY_SUPPORT"
            else:
                return module.DualWrinkleStretchPlan(
                    False,
                    "D55-V14 NO_SAFE_NORMAL_PLAN: no two-arm plan survives final perpendicular guard",
                    metrics={"d55_failure_category": "NO_SAFE_SAME_WRINKLE_NORMAL_PLAN", "d55_candidate_failures": failures, "bottom_vla_v14_generic_outward_assist": False},
                )
            bucket.sort(key=lambda item: item[0], reverse=True)
            chosen = bucket[0][1]
            chosen.metrics["d55_candidate_failures"] = failures
            chosen.metrics["d55v15_selection_bucket"] = bucket_name
            chosen.metrics["d55v15_both_move_available"] = bool(dual_normal)
            chosen.metrics["bottom_vla_v14_allowed_sweep_angle_deg"] = [90.0 - float(self.front.press_sweep_max_normal_error_deg), 90.0 + float(self.front.press_sweep_max_normal_error_deg)]
            ci = int(getattr(chosen, "candidate_index", -1))
            selected_cand = candidates[ci] if 0 <= ci < len(candidates) else {}
            chosen.metrics["bottom_vla_v16_long_wrinkle_rank"] = {
                "major_length_px": float(selected_cand.get("bottom_vla_v16_major_length_px", selected_cand.get("major_length_px", 0.0)) or 0.0),
                "linearity": float(selected_cand.get("bottom_vla_v16_linearity", selected_cand.get("linearity", 1.0)) or 1.0),
                "long_bonus": float(selected_cand.get("bottom_vla_v16_long_wrinkle_bonus", 0.0) or 0.0),
            }
            print(
                f"[D55-V16-LONG-NORMAL-SELECT] bucket={bucket_name} candidate={ci + 1} "
                f"major={float(selected_cand.get('major_length_px',0.0) or 0.0):.1f}px linearity={float(selected_cand.get('linearity',1.0) or 1.0):.2f} "
                f"ARM2={str(chosen.arm_points.get('arm2', {}).get('role', '?'))} "
                f"ARM1={str(chosen.arm_points.get('arm1', {}).get('role', '?'))}"
            )
            return chosen

        def _install_d55_perpendicular_policy(self) -> None:
            required = (
                "DualWrinkleStretchPlan",
                "_d55v8_make_perpendicular_press_plan",
                "_d55v11_apply_xy_only_mask_guard",
                "_d26v3_arm_of_plan",
                "_d55v14_make_same_wrinkle_sweep",
                "_d55v14_combine_same_candidate",
                "_d55v14_make_anchor",
            )
            missing = [name for name in required if not hasattr(self.d55, name)]
            if missing:
                raise RuntimeError(f"55-5 perpendicular policy API missing: {missing}")
            self._d55_original_planner = self.d55.build_d22_hybrid_wrinkle_plan
            self.d55.build_d22_hybrid_wrinkle_plan = self._d55_build_perpendicular_plan

        def _d60_feedback_xyz(self, arm: Any) -> Optional[np.ndarray]:
            query = getattr(self.d56, "_d31_query_feedback", None)
            xyz_fn = getattr(self.d56, "_d31_feedback_xyz", None)
            if not callable(query):
                return None
            try:
                fb = query(arm, timeout_s=0.75)
            except TypeError:
                try:
                    fb = query(arm)
                except Exception:
                    return None
            except Exception:
                return None
            if callable(xyz_fn):
                try:
                    xyz = xyz_fn(fb)
                    if xyz is not None:
                        arr = np.asarray(xyz, np.float32).reshape(3)
                        if np.all(np.isfinite(arr)):
                            return arr
                except Exception:
                    pass
            if isinstance(fb, dict):
                try:
                    arr = np.asarray([float(fb["x"]), float(fb["y"]), float(fb["z"])], np.float32)
                    if np.all(np.isfinite(arr)):
                        return arr
                except Exception:
                    pass
            return None

        def _runtime_metadata(self, actual_size: Tuple[int, int]) -> Dict[str, Any]:
            meta = super()._runtime_metadata(actual_size)
            meta["bottom_vla"] = {
                "build": BUILD,
                "semantic_actions": list(SEMANTIC_ACTIONS),
                "frozen_plan_policy": True,
                "enter_time_reinference": False,
                "d60_source": source_paths["d60"],
                "position_adjust_source": source_paths["position"],
                "align_source": source_paths["align"],
                "base_source": source_paths["base"],
                "basket_executor": "integrated_persistent_arm2",
                "basket_external_process": False,
                "basket_camera_reopen": False,
                "basket_serial_reopen": False,
            }
            return meta

        def _write_bottom_vla_manifest(self) -> None:
            root = Path(self.args.dataset_root).expanduser().resolve() / "bottom"
            payload = {
                "schema": "bottom_vla_v2",
                "build": BUILD,
                "semantic_actions": list(SEMANTIC_ACTIONS),
                "physical_actions": list(PHYSICAL_SEMANTICS),
                "sources": {
                    "base": {"path": source_paths["base"], "sha256": _sha256(source_paths["base"])},
                    "d60": {"path": source_paths["d60"], "sha256": _sha256(source_paths["d60"])},
                    "position_adjust": {"path": source_paths["position"], "sha256": _sha256(source_paths["position"])},
                    "align": {"path": source_paths["align"], "sha256": _sha256(source_paths["align"])},
                },
                "models": {
                    "segmentation": self.args.seg_model,
                    "pose": self.args.pose_model,
                },
                "classifier": {
                    "label_space": list(BOTTOM_CLASSIFIER_ACTIONS),
                    "excluded_actions": ["BASKET_GRASP", "REJUDGE"],
                    "state_schema": "bottom_state_v2",
                    "state_dim": len(BOTTOM_STATE_V2_NAMES),
                    "state_names": list(BOTTOM_STATE_V2_NAMES),
                    "sample_manifest": "classifier_samples_bottom_v2.jsonl",
                    "planner_diagnostics_are_classifier_input": False,
                    "finish_is_classifier_label": True,
                },
                "policy": {
                    "one_snapshot_one_frozen_plan": True,
                    "enter_executes_exact_frozen_plan": True,
                    "enter_time_reinference": False,
                    "result_labels": ["GOOD", "BAD", "SKIP"],
                    "collection_decision_labels": ["KEEP", "DISCARD"],
                    "basket_grasp_requires_result_label": False,
                    "basket_grasp_requires_collection_decision": False,
                    "basket_grasp_persistent_arm2_session": True,
                    "basket_grasp_arm1_motion_commands": False,
                    "basket_close_verified_before_lift": True,
                    "basket_release_verified_before_standby": True,
                    "mask_labels": ["MASK_ACCURATE", "MASK_INACCURATE"],
                    "finish_is_no_motion": True,
                    "finish_auto_new_episode": True,
                    "d60_rich_frozen_overlay": True,
                    "d60_pullup_profile": {
                        "enabled": True,
                        "pre_pullup_vertical_lift_mm": 50.0,
                        "high_z_horizontal_first_mm": 0.0,
                    },
                    "position_adjust_uses_58_3": True,
                    "basket_placement_extra_deg": float(self.front.basket_placement_extra_deg),
                    "basket_placement_rotate_speed": float(self.front.basket_placement_rotate_speed),
                    "basket_placement_lower_mm": float(self.front.basket_placement_lower_mm),
                    "basket_placement_lower_speed": float(self.front.basket_placement_lower_speed),
                    "basket_placement_min_z": float(self.front.basket_placement_min_z),
                    "d60_pre_freeze_retry_attempts": int(max(1, self.front.d60_prepare_attempts)),
                    "d60_pose_waist_priority": {
                        "enabled": True,
                        "min_conf": float(self.front.d60_pose_waist_min_conf),
                        "min_width_mm": float(self.front.d60_pose_waist_min_width_mm),
                        "max_width_mm": float(self.front.d60_pose_waist_max_width_mm),
                        "mask_near_px": float(self.front.d60_pose_waist_mask_near_px),
                    },
                    "d60_waist_endpoint_gate": {
                        "body_min_mm": float(self.front.d60_waist_grip_body_min_mm),
                        "body_max_mm": float(self.front.d60_waist_grip_body_max_mm),
                        "tangent_max_mm": float(self.front.d60_waist_grip_tangent_max_mm),
                        "endpoint_radius_max_mm": float(self.front.d60_waist_grip_endpoint_radius_max_mm),
                    },
                    "gripper_open_limits": {
                        "OUTER_PULL": {"approach": 2.35, "release": 1.90},
                        "WAIST_PULL_LAYDOWN": {"approach": 2.45, "release": 1.90},
                        "ALIGN": {"approach": 2.30, "release": 1.90},
                    },
                    "rejudge_is_no_motion": True,
                },
            }
            _atomic_json(root / "bottom_vla_manifest.json", payload)

        def _verify_sources_unchanged(self) -> None:
            super()._verify_sources_unchanged()
            if hasattr(self, "align_source_sha256"):
                now = _sha256(source_paths["align"])
                if now != self.align_source_sha256:
                    raise RuntimeError("align-11 source changed during session")
            if hasattr(self, "d60_source_sha256"):
                now = _sha256(source_paths["d60"])
                if now != self.d60_source_sha256:
                    raise RuntimeError("60-14 source changed during session")
            if hasattr(self, "position_source_sha256"):
                now = _sha256(source_paths["position"])
                if now != self.position_source_sha256:
                    raise RuntimeError("58-3 POSITION_ADJUST source changed during session")

        def _main28_build_d56_taught_spec(self, locked):
            return {"ok": False, "reason": "BOTTOM_VLA_USES_NATIVE_60_13_EXECUTOR"}

        def _semantic_metadata(self, executed: Optional[str] = None) -> Dict[str, Any]:
            semantic = self.pending_semantic or self.semantic_selected
            return {
                "bottom_vla_schema": "bottom_vla_v2",
                "bottom_vla_build": BUILD,
                "action_label_space": list(SEMANTIC_ACTIONS),
                "auto_recommended_action": self._auto_at_selection,
                "human_selected_action": self._human_selected,
                "selected_action": semantic,
                "executed_action": executed,
                "plan_origin": self._prepare_origin,
                "frozen_plan": True,
                "enter_time_reinference": False,
                "source_motion": {
                    "BASKET_GRASP": "50-1-integrated-persistent-arm2",
                    "POSITION_ADJUST": "58-3",
                    "OUTER_PULL": "54-3",
                    "PRESS_SWEEP": "55-5",
                    "WAIST_PULL_LAYDOWN": "60-14",
                    "ALIGN": "align-11",
                    "FINISH": "NO_MOTION",
                }.get(semantic),
            }

        def _bottom_state_v2_parts(self, locked: Any) -> Tuple[Any, Any, np.ndarray]:
            obs = None
            bundle = None
            H = None
            try:
                values = list(vars(locked).values())
            except Exception:
                values = []
            for value in values:
                if obs is None and hasattr(value, "mask") and hasattr(value, "pose"):
                    obs = value
                if bundle is None and hasattr(value, "raw") and hasattr(value, "corrected"):
                    bundle = value
                if H is None and isinstance(value, np.ndarray) and value.shape == (3, 3):
                    H = value
            if H is None:
                H = np.asarray(self.H_raw, dtype=np.float64).reshape(3, 3)
            return obs, bundle, np.asarray(H, dtype=np.float64).reshape(3, 3)

        def _bottom_state_v2_board_points(self, contour: Any, H: np.ndarray) -> Optional[np.ndarray]:
            try:
                pts = np.asarray(contour, dtype=np.float64).reshape(-1, 2)
            except Exception:
                return None
            if len(pts) < 3:
                return None
            if len(pts) > 800:
                step = max(1, len(pts) // 800)
                pts = pts[::step]
            ones = np.ones((len(pts), 1), dtype=np.float64)
            hp = np.concatenate([pts, ones], axis=1)
            out = hp @ H.T
            den = out[:, 2]
            good = np.isfinite(den) & (np.abs(den) > 1e-9)
            if int(np.count_nonzero(good)) < 3:
                return None
            board = out[good, :2] / den[good, None]
            board = board[np.all(np.isfinite(board), axis=1)]
            return board.astype(np.float32) if len(board) >= 3 else None

        def _bottom_state_v2_pose_point(self, pose: Any, key: str, attr: str = "") -> Optional[np.ndarray]:
            if pose is None:
                return None
            raw = None
            if attr:
                raw = getattr(pose, attr, None)
            if raw is None:
                try:
                    raw = dict(getattr(pose, "keypoints_board", {}) or {}).get(key)
                except Exception:
                    raw = None
            try:
                q = np.asarray(raw, dtype=np.float32).reshape(2)
            except Exception:
                return None
            return q if np.all(np.isfinite(q)) else None

        def _bottom_state_v2_axis_norm(self, angle_deg: float, directed: bool) -> float:
            try:
                angle = float(angle_deg)
            except Exception:
                return 0.0
            if not math.isfinite(angle):
                return 0.0
            if directed:
                angle = ((angle + 180.0) % 360.0) - 180.0
                return float(np.clip(angle / 180.0, -1.0, 1.0))
            angle = ((angle + 90.0) % 180.0) - 90.0
            return float(np.clip(angle / 90.0, -1.0, 1.0))

        def _bottom_state_v2_wrinkle(self, frame: Any, mask_u8: Any, H: np.ndarray, board_diag: float, body_angle_deg: Optional[float]) -> Tuple[Dict[str, float], Dict[str, Any]]:
            zero = {
                "wrinkle_edge_ratio": 0.0,
                "wrinkle_component_count_norm": 0.0,
                "dominant_wrinkle_length_norm": 0.0,
                "dominant_wrinkle_linearity_norm": 0.0,
                "wrinkle_vs_body_axis_angle_norm": 0.0,
            }
            validity = {"valid": False, "component_count": 0, "reason": "UNAVAILABLE"}
            if not isinstance(frame, np.ndarray) or mask_u8 is None:
                return zero, validity
            try:
                mask = (np.asarray(mask_u8) > 0).astype(np.uint8) * 255
                if mask.shape[:2] != frame.shape[:2]:
                    mask = self.cv2.resize(mask, (frame.shape[1], frame.shape[0]), interpolation=self.cv2.INTER_NEAREST)
                inner = self.cv2.erode(mask, np.ones((7, 7), np.uint8), iterations=1)
                if int(np.count_nonzero(inner)) < 100:
                    validity["reason"] = "INNER_MASK_TOO_SMALL"
                    return zero, validity
                gray = self.cv2.cvtColor(frame, self.cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame.astype(np.uint8)
                blur = self.cv2.GaussianBlur(gray, (5, 5), 0)
                lap = np.abs(self.cv2.Laplacian(blur, self.cv2.CV_32F, ksize=3))
                vals = lap[inner > 0]
                if vals.size < 100:
                    validity["reason"] = "NO_INNER_PIXELS"
                    return zero, validity
                threshold = max(float(np.percentile(vals, 88.0)), float(np.mean(vals) + 0.65 * np.std(vals)))
                binary = ((lap >= threshold) & (inner > 0)).astype(np.uint8) * 255
                binary = self.cv2.morphologyEx(binary, self.cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
                contours, _ = self.cv2.findContours(binary, self.cv2.RETR_EXTERNAL, self.cv2.CHAIN_APPROX_NONE)
                min_area = max(8.0, 0.00018 * float(np.count_nonzero(mask)))
                kept = np.zeros_like(binary)
                components: List[Dict[str, Any]] = []
                for contour in contours:
                    area = float(self.cv2.contourArea(contour))
                    if area < min_area:
                        continue
                    pts = np.asarray(contour, dtype=np.float64).reshape(-1, 2)
                    if len(pts) < 5:
                        continue
                    center = np.mean(pts, axis=0)
                    cov = np.cov((pts - center).T)
                    eigval, eigvec = np.linalg.eigh(cov)
                    order = np.argsort(eigval)[::-1]
                    major_var = max(float(eigval[order[0]]), 1e-6)
                    minor_var = max(float(eigval[order[1]]), 1e-6)
                    axis = eigvec[:, order[0]]
                    sigma = math.sqrt(major_var)
                    p0 = center - axis * (2.3 * sigma)
                    p1 = center + axis * (2.3 * sigma)
                    hp = np.asarray([[p0[0], p0[1], 1.0], [p1[0], p1[1], 1.0]], dtype=np.float64) @ H.T
                    if np.all(np.isfinite(hp)) and np.all(np.abs(hp[:, 2]) > 1e-9):
                        b = hp[:, :2] / hp[:, 2:3]
                        length_board = float(np.linalg.norm(b[1] - b[0]))
                        angle_board = float(math.degrees(math.atan2(float(b[1, 1] - b[0, 1]), float(b[1, 0] - b[0, 0]))))
                    else:
                        length_board = 0.0
                        angle_board = 0.0
                    linearity = float(math.sqrt(major_var / minor_var))
                    score = float(length_board * min(linearity, 12.0))
                    components.append({"length_board": length_board, "linearity": linearity, "angle_board": angle_board, "score": score})
                    self.cv2.drawContours(kept, [contour], -1, 255, thickness=-1)
                inner_count = max(1, int(np.count_nonzero(inner)))
                edge_ratio = float(np.count_nonzero(kept) / inner_count)
                count = len(components)
                if not components:
                    validity.update({"valid": True, "component_count": 0, "reason": "NO_DOMINANT_COMPONENT"})
                    zero["wrinkle_edge_ratio"] = edge_ratio
                    return zero, validity
                dominant = max(components, key=lambda x: x["score"])
                if body_angle_deg is None or not math.isfinite(float(body_angle_deg)):
                    angle_diff_norm = 0.0
                else:
                    diff = abs(((float(dominant["angle_board"]) - float(body_angle_deg) + 90.0) % 180.0) - 90.0)
                    angle_diff_norm = float(np.clip(diff / 90.0, 0.0, 1.0))
                values = {
                    "wrinkle_edge_ratio": float(np.clip(edge_ratio, 0.0, 1.0)),
                    "wrinkle_component_count_norm": float(np.clip(count / 12.0, 0.0, 1.0)),
                    "dominant_wrinkle_length_norm": float(np.clip(float(dominant["length_board"]) / max(board_diag, 1.0), 0.0, 1.0)),
                    "dominant_wrinkle_linearity_norm": float(np.clip((float(dominant["linearity"]) - 1.0) / 9.0, 0.0, 1.0)),
                    "wrinkle_vs_body_axis_angle_norm": angle_diff_norm,
                }
                validity.update({"valid": True, "component_count": int(count), "reason": "OK", "threshold": float(threshold)})
                return values, validity
            except Exception as exc:
                validity["reason"] = repr(exc)
                return zero, validity

        def _build_bottom_state_v2(self, locked: Any) -> Dict[str, Any]:
            obs, bundle, H = self._bottom_state_v2_parts(locked)
            cfg = self.cfg56
            x0 = float(getattr(cfg, "board_x_min", 0.0))
            x1 = float(getattr(cfg, "board_x_max", 494.0))
            y0 = float(getattr(cfg, "board_y_min", -496.0))
            y1 = float(getattr(cfg, "board_y_max", 5.0))
            bw = max(1.0, x1 - x0)
            bh = max(1.0, y1 - y0)
            bdiag = math.hypot(bw, bh)
            bc = np.asarray([(x0 + x1) * 0.5, (y0 + y1) * 0.5], dtype=np.float32)
            values = {name: 0.0 for name in BOTTOM_STATE_V2_NAMES}
            validity: Dict[str, Any] = {"mask": False, "pose": False, "wrinkle": False, "homography": bool(np.all(np.isfinite(H)))}
            mask = getattr(obs, "mask", None) if obs is not None else None
            board_pts = None
            if mask is not None:
                values["mask_valid"] = 1.0
                validity["mask"] = True
                board_pts = self._bottom_state_v2_board_points(getattr(mask, "contour", None), H)
                try:
                    center = np.asarray(getattr(mask, "center_board"), dtype=np.float32).reshape(2)
                except Exception:
                    center = bc.copy()
                values["mask_center_x_norm"] = float((center[0] - x0) / bw)
                values["mask_center_y_norm"] = float((center[1] - y0) / bh)
                values["board_center_error_norm"] = float(np.linalg.norm(center - bc) / bdiag)
                values["mask_solidity"] = float(np.clip(float(getattr(mask, "solidity", 0.0) or 0.0), 0.0, 1.0))
                if board_pts is not None and len(board_pts) >= 3:
                    minx, miny = np.min(board_pts, axis=0)
                    maxx, maxy = np.max(board_pts, axis=0)
                    mw = max(1e-6, float(maxx - minx))
                    mh = max(1e-6, float(maxy - miny))
                    try:
                        area_board = abs(float(self.cv2.contourArea(board_pts.reshape(-1, 1, 2).astype(np.float32))))
                    except Exception:
                        area_board = 0.0
                    values["mask_area_ratio"] = float(np.clip(area_board / (bw * bh), 0.0, 2.0))
                    values["mask_width_norm"] = float(mw / bw)
                    values["mask_height_norm"] = float(mh / bh)
                    values["mask_extent"] = float(np.clip(area_board / max(mw * mh, 1e-6), 0.0, 1.0))
                    values["left_board_margin_norm"] = float((float(minx) - x0) / bw)
                    values["right_board_margin_norm"] = float((x1 - float(maxx)) / bw)
                    values["top_board_margin_norm"] = float((y1 - float(maxy)) / bh)
                    values["bottom_board_margin_norm"] = float((float(miny) - y0) / bh)
                    centered = board_pts.astype(np.float64) - np.mean(board_pts.astype(np.float64), axis=0)
                    if len(centered) >= 3:
                        cov = np.cov(centered.T)
                        eigval, eigvec = np.linalg.eigh(cov)
                        axis = eigvec[:, int(np.argmax(eigval))]
                        angle = math.degrees(math.atan2(float(axis[1]), float(axis[0])))
                        values["mask_pca_axis_angle_norm"] = self._bottom_state_v2_axis_norm(angle, False)
                else:
                    try:
                        area_px = float(getattr(mask, "area_px", 0.0) or 0.0)
                        frame_area = float(np.prod(np.asarray(getattr(mask, "mask_u8")).shape[:2]))
                        values["mask_area_ratio"] = float(np.clip(area_px / max(frame_area, 1.0), 0.0, 1.0))
                    except Exception:
                        pass
            pose = getattr(obs, "pose", None) if obs is not None else None
            pose_valid = bool(pose is not None and bool(getattr(pose, "valid", True)))
            values["pose_valid"] = 1.0 if pose_valid else 0.0
            validity["pose"] = pose_valid
            key_board = dict(getattr(pose, "keypoints_board", {}) or {}) if pose is not None else {}
            values["pose_valid_keypoint_ratio"] = float(np.clip(len(key_board) / 8.0, 0.0, 1.0))
            wl = self._bottom_state_v2_pose_point(pose, "waist_img_left", "waist_left")
            wc = self._bottom_state_v2_pose_point(pose, "waist_center", "waist_center")
            wr = self._bottom_state_v2_pose_point(pose, "waist_img_right", "waist_right")
            crotch = self._bottom_state_v2_pose_point(pose, "crotch", "crotch")
            lho = self._bottom_state_v2_pose_point(pose, "img_left_hem_outer")
            lhi = self._bottom_state_v2_pose_point(pose, "img_left_hem_inner")
            rhi = self._bottom_state_v2_pose_point(pose, "img_right_hem_inner")
            rho = self._bottom_state_v2_pose_point(pose, "img_right_hem_outer")
            lc = self._bottom_state_v2_pose_point(pose, "", "left_hem_center")
            rc = self._bottom_state_v2_pose_point(pose, "", "right_hem_center")
            lower = self._bottom_state_v2_pose_point(pose, "", "lower_center")
            if lc is None and lho is not None and lhi is not None:
                lc = 0.5 * (lho + lhi)
            if rc is None and rhi is not None and rho is not None:
                rc = 0.5 * (rhi + rho)
            if wc is None and wl is not None and wr is not None:
                wc = 0.5 * (wl + wr)
            if lower is None and lc is not None and rc is not None:
                lower = 0.5 * (lc + rc)
            waist_angle = None
            if wl is not None and wr is not None:
                waist_vec = wr - wl
                waist_width = float(np.linalg.norm(waist_vec))
                values["waist_width_norm"] = float(waist_width / bw)
                if waist_width > 1e-6:
                    waist_angle = float(math.degrees(math.atan2(float(waist_vec[1]), float(waist_vec[0]))))
                    values["waist_axis_angle_norm"] = self._bottom_state_v2_axis_norm(waist_angle, False)
            elif pose is not None:
                try:
                    waist_width = float(getattr(pose, "waist_width_mm", 0.0) or 0.0)
                    values["waist_width_norm"] = float(waist_width / bw)
                    waist_angle = float(getattr(pose, "waist_angle_deg", 0.0) or 0.0)
                    values["waist_axis_angle_norm"] = self._bottom_state_v2_axis_norm(waist_angle, False)
                except Exception:
                    waist_angle = None
            body_angle = None
            if wc is not None:
                values["waist_center_x_norm"] = float((wc[0] - x0) / bw)
                values["waist_center_y_norm"] = float((wc[1] - y0) / bh)
                values["waist_target_y_error_norm"] = float((wc[1] - float(self.front.align_waist_target_y_mm)) / bh)
                body_target = lower if lower is not None else crotch
                if body_target is not None:
                    body_vec = body_target - wc
                    if float(np.linalg.norm(body_vec)) > 1e-6:
                        body_angle = float(math.degrees(math.atan2(float(body_vec[1]), float(body_vec[0]))))
                        values["body_axis_angle_norm"] = self._bottom_state_v2_axis_norm(body_angle, True)
            if body_angle is None and pose is not None:
                try:
                    body_angle = float(getattr(pose, "pants_axis_angle_deg"))
                    values["body_axis_angle_norm"] = self._bottom_state_v2_axis_norm(body_angle, True)
                except Exception:
                    body_angle = None
            if crotch is not None and lc is not None and rc is not None:
                ll = float(np.linalg.norm(lc - crotch))
                rl = float(np.linalg.norm(rc - crotch))
                if max(ll, rl) > 1e-6:
                    values["leg_length_balance"] = float(min(ll, rl) / max(ll, rl))
                    values["pose_left_right_symmetry_error_norm"] = float(abs(ll - rl) / max(0.5 * (ll + rl), 1.0))
            if lho is not None and lhi is not None and rhi is not None and rho is not None:
                lw = float(np.linalg.norm(lho - lhi))
                rw = float(np.linalg.norm(rho - rhi))
                if max(lw, rw) > 1e-6:
                    values["hem_width_balance"] = float(min(lw, rw) / max(lw, rw))
            frame = None
            if bundle is not None:
                frame = getattr(bundle, "corrected", None)
                if not isinstance(frame, np.ndarray):
                    frame = getattr(bundle, "raw", None)
            if frame is None and obs is not None:
                frame = getattr(obs, "frame", None)
            wrinkle, wrinkle_validity = self._bottom_state_v2_wrinkle(frame, getattr(mask, "mask_u8", None) if mask is not None else None, H, bdiag, body_angle)
            values.update(wrinkle)
            validity["wrinkle"] = bool(wrinkle_validity.get("valid", False))
            validity["wrinkle_detail"] = wrinkle_validity
            ordered = [float(values[name]) if math.isfinite(float(values[name])) else 0.0 for name in BOTTOM_STATE_V2_NAMES]
            return {
                "schema": "bottom_state_v2",
                "version": 2,
                "dim": len(BOTTOM_STATE_V2_NAMES),
                "names": list(BOTTOM_STATE_V2_NAMES),
                "values": ordered,
                "validity": _json_safe(validity),
                "action_independent": True,
                "planner_features_used": False,
                "board_bounds_mm": [x0, x1, y0, y1],
            }

        def _classifier_manifest_path(self) -> Path:
            root = Path(self.args.dataset_root).expanduser().resolve() / "bottom"
            root.mkdir(parents=True, exist_ok=True)
            return root / "classifier_samples_bottom_v2.jsonl"

        def _record_json_path(self, record: Any) -> Optional[Path]:
            if not isinstance(record, dict):
                return None
            raw = record.get("fs_path") or record.get("path") or record.get("json_path")
            if not raw:
                return None
            p = Path(str(raw))
            if p.is_dir():
                p = p / "data.json"
            return p if p.is_file() else None

        def _append_classifier_transition_sample(self, semantic: Optional[str], keep: bool) -> None:
            semantic = str(semantic or "")
            if semantic not in BOTTOM_CLASSIFIER_ACTIONS or semantic == "FINISH":
                return
            transition_record = getattr(self.recorder, "latest_transition", None)
            transition_path = self._record_json_path(transition_record)
            payload: Dict[str, Any] = {}
            if transition_path is not None:
                try:
                    with open(transition_path, "r", encoding="utf-8") as f:
                        payload = json.load(f)
                except Exception:
                    payload = {}
            common = payload.get("common_state") if isinstance(payload, dict) else None
            state = common.get("bottom_state_v2") if isinstance(common, dict) else None
            if not isinstance(state, dict):
                with self.state_lock:
                    locked = self.locked
                if locked is not None:
                    state = (dict(getattr(locked, "diagnostics", {}) or {}).get("common_state") or {}).get("bottom_state_v2")
            if not isinstance(state, dict):
                print(f"[BOTTOM-CLASSIFIER-WARN] state missing for {semantic}; manifest sample skipped")
                return
            record = {
                "schema": "bottom_classifier_sample_v2",
                "build": BUILD,
                "created_at": time.time(),
                "cycle_id": getattr(self.recorder, "cycle_id", None),
                "action": semantic,
                "label": {"human_selected_action": semantic},
                "inputs": {"bottom_state_v2": state, "board_source": "BEFORE_ACTION_OBSERVATION"},
                "quality": {
                    "human_result": payload.get("human_result", self.last_result),
                    "collection_decision": "KEEP" if keep else "DISCARD",
                    "training_eligible": bool(payload.get("training_eligible", False)),
                    "mask_quality": payload.get("mask_quality"),
                },
                "execution": {
                    "executed_action": payload.get("executed_action", semantic),
                    "execution_success": payload.get("execution_success"),
                    "garment_motion_committed": payload.get("garment_motion_committed"),
                },
                "transition_json": None if transition_path is None else str(transition_path),
                "observation_record": _json_safe(payload.get("before_observation", payload.get("observation", None))),
                "planner_diagnostics_are_input": False,
            }
            line = json.dumps(_json_safe(record), ensure_ascii=False, separators=(",", ":")) + "\n"
            with self._classifier_manifest_lock:
                with open(self._classifier_manifest_path(), "a", encoding="utf-8") as f:
                    f.write(line)
                    f.flush()
                    os.fsync(f.fileno())
            print(f"[BOTTOM-CLASSIFIER-SAMPLE] action={semantic} keep={keep} eligible={record['quality']['training_eligible']}")

        def _append_finish_classifier_sample(self, locked: Any, finish_json: Path) -> None:
            common = dict(getattr(locked, "diagnostics", {}) or {}).get("common_state") or {}
            state = common.get("bottom_state_v2") if isinstance(common, dict) else None
            if not isinstance(state, dict):
                state = self._build_bottom_state_v2(locked)
            record = {
                "schema": "bottom_classifier_sample_v2",
                "build": BUILD,
                "created_at": time.time(),
                "cycle_id": getattr(self.recorder, "cycle_id", None),
                "action": "FINISH",
                "label": {"human_selected_action": "FINISH"},
                "inputs": {"bottom_state_v2": state, "board_source": "FINISH_STATE_OBSERVATION"},
                "quality": {"human_result": "NOT_APPLICABLE", "collection_decision": "KEEP", "training_eligible": True, "mask_quality": None},
                "execution": {"executed_action": "FINISH", "execution_success": True, "garment_motion_committed": False, "no_motion": True},
                "finish_json": str(finish_json),
                "observation_record": _json_safe(getattr(locked, "observation_record", None)),
                "planner_diagnostics_are_input": False,
            }
            line = json.dumps(_json_safe(record), ensure_ascii=False, separators=(",", ":")) + "\n"
            with self._classifier_manifest_lock:
                with open(self._classifier_manifest_path(), "a", encoding="utf-8") as f:
                    f.write(line)
                    f.flush()
                    os.fsync(f.fileno())
            print("[BOTTOM-CLASSIFIER-SAMPLE] action=FINISH keep=True eligible=True")

        def _decorate_current_records(self, executed: Optional[str] = None) -> None:
            fields = self._semantic_metadata(executed)
            with self.state_lock:
                locked = self.locked
            if locked is not None:
                diagnostics = dict(getattr(locked, "diagnostics", {}) or {})
                common = diagnostics.get("common_state")
                state = common.get("bottom_state_v2") if isinstance(common, dict) else None
                if not isinstance(state, dict):
                    state = self._build_bottom_state_v2(locked)
                    diagnostics["common_state"] = {"bottom_state_v2": state}
                    locked.diagnostics = diagnostics
                fields["common_state"] = {"bottom_state_v2": state}
                fields["bottom_state_v2"] = state
                fields["classifier_input_policy"] = {
                    "label_space": list(BOTTOM_CLASSIFIER_ACTIONS),
                    "planner_diagnostics_are_input": False,
                    "state_schema": "bottom_state_v2",
                }
            _patch_json_record(getattr(self.recorder, "current_observation", None), fields)
            _patch_json_record(getattr(self.recorder, "current_decision", None), fields)
            _patch_json_record(getattr(self.recorder, "latest_transition", None), fields)

        def _invalidate_for_new_action(self, reason: str) -> None:
            try:
                self._invalidate_lock(reason)
            except Exception:
                with self.state_lock:
                    self.locked = None
                    self.display_image = None
            with self.align_runtime_lock:
                self.align_runtime.clear()
            with self.d60_runtime_lock:
                self.d60_runtime.clear()

        def _ensure_d60_arm_api(self) -> None:
            for arm in self.arms.values():
                if arm is None or hasattr(arm, "set_gripper_torque"):
                    continue
                def compat(torque: int, delay: float = 0.18, stage: Optional[str] = None, caller: Optional[str] = None, _arm=arm) -> None:
                    cmd = {"T": 107, "tor": int(torque)}
                    try:
                        _arm.send(cmd, delay=delay, stage=stage, caller=caller, log_command=True)
                    except TypeError:
                        try:
                            _arm.send(cmd, delay=delay, stage=stage, caller=caller)
                        except TypeError:
                            _arm.send(cmd, delay=delay)
                setattr(arm, "set_gripper_torque", compat)

        def _frame_for_action(self, raw: np.ndarray, action: Optional[str]) -> np.ndarray:
            if action == "BASKET_GRASP":
                return raw.copy()
            if action == "WAIST_PULL_LAYDOWN":
                return super()._frame_for_action(raw, "D56_WAIST_LIFT_LAYDOWN")
            return super()._frame_for_action(raw, action)

        def _basket_calib_path(self) -> Path:
            requested = str(self.front.basket_calib or getattr(self.args, "d50_basket_calib", "") or "basket_arm2_5point_affine.json")
            raw = Path(requested).expanduser()
            candidates = [raw]
            if not raw.is_absolute():
                candidates.extend([
                    Path(__file__).resolve().parent / raw,
                    Path(source_paths["base"]).resolve().parent / raw,
                    Path("/workspace/project_train/aruco_test/dual/undistort") / raw,
                    Path("/workspace/project_train/aruco_test/dual") / raw,
                ])
            seen = set()
            for candidate in candidates:
                try:
                    p = candidate.resolve()
                except Exception:
                    p = candidate
                key = str(p)
                if key in seen:
                    continue
                seen.add(key)
                if p.is_file():
                    return p
            raise FileNotFoundError(f"basket calibration not found: {requested}")

        def _basket_board_target(self) -> Dict[str, Any]:
            marker_map = self.config.get("aruco", {}).get("marker_board_mm", {})
            pts = {}
            for key in ("0", "1", "2", "3"):
                value = marker_map.get(key)
                if not isinstance(value, (list, tuple)) or len(value) < 2:
                    raise RuntimeError("board marker coordinates 0/1/2/3 are required")
                pts[key] = np.asarray([float(value[0]), float(value[1])], dtype=np.float64)

            def cross2(a: np.ndarray, b: np.ndarray) -> float:
                return float(a[0] * b[1] - a[1] * b[0])

            p0, p1, p2, p3 = pts["0"], pts["1"], pts["2"], pts["3"]
            r = p3 - p0
            v = p2 - p1
            den = cross2(r, v)
            if abs(den) < 1e-9:
                raise RuntimeError("board diagonals are degenerate")
            t = cross2(p1 - p0, v) / den
            center_board = p0 + t * r
            arm_cfg = self.config.get("dual_roarm", {}).get("arm2", {})
            affine = np.asarray(arm_cfg.get("board_to_roarm_affine_2x3"), dtype=np.float64)
            if affine.shape != (2, 3) or not np.all(np.isfinite(affine)):
                raise RuntimeError("ARM2 board affine is invalid")
            center_arm = affine[:, :2] @ center_board + affine[:, 2]
            inner = None
            labels = arm_cfg.get("calib_points", [])
            roarm = arm_cfg.get("calib_roarm_points", [])
            if isinstance(labels, list) and isinstance(roarm, list):
                for index, item in enumerate(labels):
                    if not isinstance(item, dict) or str(item.get("label", "")).upper() != "RED_EXTRA":
                        continue
                    if index >= len(roarm):
                        continue
                    candidate = np.asarray(roarm[index], dtype=np.float64)
                    if candidate.shape == (2,) and np.all(np.isfinite(candidate)):
                        inner = candidate
                        break
            if inner is None:
                inner = np.asarray(center_arm, dtype=np.float64)
            safe_hover_z = float(arm_cfg.get("safe_hover_z", 180.0))
            if not np.isfinite(safe_hover_z):
                safe_hover_z = 180.0
            return {
                "board_center_xy": center_board.astype(float).tolist(),
                "arm2_center_xy": center_arm.astype(float).tolist(),
                "arm2_inner_xy": inner.astype(float).tolist(),
                "safe_hover_z": safe_hover_z,
            }

        def _basket_arm2(self):
            arm = self.arms.get("arm2") if isinstance(self.arms, dict) else None
            if arm is None:
                raise RuntimeError("ARM2 persistent session is unavailable")
            return arm

        def _basket_feedback(self, quiet: bool = False) -> Optional[Dict[str, Any]]:
            arm = self._basket_arm2()
            timeout = float(self.front.basket_feedback_timeout_s)
            if hasattr(self.d54, "_d31_query_feedback"):
                try:
                    value = self.d54._d31_query_feedback(arm, timeout_s=timeout)
                    if isinstance(value, dict) and all(k in value for k in ("x", "y", "z")):
                        return value
                except Exception as exc:
                    if not quiet:
                        print(f"[BASKET-FEEDBACK-WARN] {exc!r}")
            methods = []
            if hasattr(arm, "feedback_retry"):
                methods.append(lambda: arm.feedback_retry(timeout, attempts=2, retry_delay=0.15, quiet=quiet))
                methods.append(lambda: arm.feedback_retry(timeout, attempts=2, retry_delay=0.15))
            if hasattr(arm, "feedback"):
                methods.append(lambda: arm.feedback(timeout=timeout, quiet=quiet))
                methods.append(lambda: arm.feedback(timeout))
            for method in methods:
                try:
                    value = method()
                    if isinstance(value, dict) and all(k in value for k in ("x", "y", "z")):
                        return value
                except TypeError:
                    continue
                except Exception as exc:
                    if not quiet:
                        print(f"[BASKET-FEEDBACK-WARN] {exc!r}")
            return None

        def _basket_torque_on(self) -> None:
            arm = self._basket_arm2()
            if hasattr(arm, "torque_on"):
                arm.torque_on()
                return
            if hasattr(arm, "send"):
                attempts = [
                    lambda: arm.send({"T": 210, "cmd": 1}, delay=0.15, stage="BASKET_TORQUE_ON", caller="bottom_vla_basket"),
                    lambda: arm.send({"T": 210, "cmd": 1}, delay=0.15),
                    lambda: arm.send({"T": 210, "cmd": 1}),
                ]
                for method in attempts:
                    try:
                        method()
                        return
                    except TypeError:
                        continue
            raise RuntimeError("ARM2 torque-on command unavailable")

        def _basket_set_gripper(self, angle: float, delay: float, stage: str) -> None:
            arm = self._basket_arm2()
            value = float(angle)
            if hasattr(arm, "set_gripper"):
                attempts = [
                    lambda: arm.set_gripper(value, delay=float(delay), stage=stage, caller="bottom_vla_basket"),
                    lambda: arm.set_gripper(value, delay=float(delay)),
                    lambda: arm.set_gripper(value, spd=0.0, acc=0.0, delay=float(delay)),
                    lambda: arm.set_gripper(value),
                ]
            elif hasattr(arm, "gripper_open"):
                attempts = [lambda: arm.gripper_open(value, 0.0, 0.0)]
            else:
                raise RuntimeError("ARM2 gripper API unavailable")
            last = None
            for method in attempts:
                try:
                    method()
                    return
                except TypeError as exc:
                    last = exc
                    continue
            raise RuntimeError(f"ARM2 gripper command signature mismatch: {last!r}")

        def _basket_move_goal(self, x: float, y: float, z: float, t: float, speed: float, stage: str) -> None:
            arm = self._basket_arm2()
            command = int(getattr(self.args, "move_command", 104))
            methods = [
                lambda: arm.move_goal(float(x), float(y), float(z), float(t), float(speed), move_command=command, stage=stage, caller="bottom_vla_basket", delay=0.05, log_command=True),
                lambda: arm.move_goal(float(x), float(y), float(z), float(t), float(speed), move_command=command),
                lambda: arm.move_goal(command, float(x), float(y), float(z), float(t), float(speed)),
            ]
            last = None
            for method in methods:
                try:
                    method()
                    return
                except TypeError as exc:
                    last = exc
                    continue
            raise RuntimeError(f"ARM2 move_goal signature mismatch: {last!r}")

        def _basket_wait_waypoint(self, label: str, target: Tuple[float, float, float], xy_only: bool = False) -> np.ndarray:
            deadline = time.time() + max(0.5, float(self.front.basket_move_timeout_s))
            target_arr = np.asarray(target, dtype=np.float64)
            recent_xy: List[np.ndarray] = []
            last = None
            while time.time() < deadline:
                time.sleep(max(0.05, float(self.front.basket_move_poll_s)))
                fb = self._basket_feedback(quiet=True)
                if fb is None or not all(k in fb for k in ("x", "y", "z")):
                    continue
                actual = np.asarray([float(fb["x"]), float(fb["y"]), float(fb["z"])], dtype=np.float64)
                last = actual
                if xy_only:
                    error = float(np.linalg.norm(actual[:2] - target_arr[:2]))
                    recent_xy.append(actual[:2].copy())
                    if len(recent_xy) > 3:
                        recent_xy = recent_xy[-3:]
                    settled = False
                    if len(recent_xy) == 3:
                        anchor = recent_xy[0]
                        settled = max(float(np.linalg.norm(p - anchor)) for p in recent_xy[1:]) <= 2.0
                    print(f"[{label}] xy_error={error:.1f}mm z={actual[2]:.1f} settled={settled}")
                    if error <= float(self.front.basket_move_tolerance_mm) and settled:
                        return actual
                else:
                    error = float(np.linalg.norm(actual - target_arr))
                    print(f"[{label}] error={error:.1f}mm actual=({actual[0]:.1f},{actual[1]:.1f},{actual[2]:.1f})")
                    if error <= float(self.front.basket_move_tolerance_mm):
                        return actual
            raise RuntimeError(f"{label} arrival timeout last={None if last is None else last.tolist()}")

        def _basket_wait_transit_near(self, label: str, target: Tuple[float, float, float], near_mm: float) -> Tuple[np.ndarray, float]:
            deadline = time.time() + max(0.5, float(self.front.basket_move_timeout_s))
            target_arr = np.asarray(target, dtype=np.float64)
            threshold = max(float(self.front.basket_move_tolerance_mm), float(near_mm))
            last = None
            best_error = float("inf")
            while time.time() < deadline:
                time.sleep(max(0.04, min(0.10, float(self.front.basket_move_poll_s))))
                fb = self._basket_feedback(quiet=True)
                if fb is None or not all(k in fb for k in ("x", "y", "z")):
                    continue
                actual = np.asarray([float(fb["x"]), float(fb["y"]), float(fb["z"])], dtype=np.float64)
                last = actual
                error = float(np.linalg.norm(actual[:2] - target_arr[:2]))
                best_error = min(best_error, error)
                print(f"[{label}] transit_xy_error={error:.1f}mm switch_at<={threshold:.1f}mm z={actual[2]:.1f}")
                if error <= threshold:
                    return actual, error
            raise RuntimeError(f"{label} transit-near timeout best_error={best_error:.1f} last={None if last is None else last.tolist()}")

        def _basket_probe_feedback(self, label: str, target_z: float) -> Dict[str, Any]:
            deadline = time.time() + max(0.5, float(self.front.basket_probe_step_timeout_s))
            recent: List[float] = []
            latest = None
            while time.time() < deadline:
                time.sleep(max(0.05, min(0.20, float(self.front.basket_move_poll_s))))
                fb = self._basket_feedback(quiet=True)
                if fb is None or not all(k in fb for k in ("x", "y", "z")):
                    continue
                latest = fb
                z = float(fb["z"])
                recent.append(z)
                if len(recent) > 3:
                    recent = recent[-3:]
                close = abs(z - float(target_z)) <= float(self.front.basket_probe_z_tolerance_mm)
                stable = len(recent) == 3 and max(recent) - min(recent) <= float(self.front.basket_probe_z_stable_span_mm)
                print(f"[{label}] target_z={float(target_z):.1f} actual_z={z:.1f} close={close} stable={stable}")
                if close or stable:
                    return fb
            if latest is not None:
                return latest
            raise RuntimeError(f"{label} feedback unavailable")

        def _basket_lift_adaptive(self, x: float, y: float, start_z: float, target_z: float, tool_t: float) -> Dict[str, Any]:
            if float(target_z) <= float(start_z) + 1.0:
                raise RuntimeError(f"basket lift target {float(target_z):.1f} is not above grasp Z {float(start_z):.1f}")
            self._basket_move_goal(x, y, target_z, tool_t, self.front.basket_lift_speed, "BASKET-LIFT")
            deadline = time.time() + max(0.5, float(self.front.basket_move_timeout_s))
            recent: List[float] = []
            latest = None
            started = time.monotonic()
            while time.time() < deadline:
                time.sleep(max(0.05, float(self.front.basket_move_poll_s)))
                fb = self._basket_feedback(quiet=True)
                if fb is None or not all(k in fb for k in ("x", "y", "z")):
                    continue
                latest = fb
                z = float(fb["z"])
                err = abs(z - float(target_z))
                print(f"[BASKET-LIFT] target_z={float(target_z):.1f} actual_z={z:.1f} error={err:.1f}mm")
                if err <= float(self.front.basket_lift_z_tolerance_mm):
                    return {"reached": True, "stalled": False, "feedback": fb}
                recent.append(z)
                n = max(3, int(self.front.basket_lift_stall_polls))
                if len(recent) > n:
                    recent = recent[-n:]
                if time.monotonic() - started >= 1.0 and len(recent) == n and max(recent) - min(recent) <= float(self.front.basket_lift_stall_span_mm):
                    return {"reached": False, "stalled": True, "feedback": fb}
            if latest is None:
                raise RuntimeError("basket lift feedback unavailable")
            return {"reached": False, "stalled": False, "feedback": latest}

        def _basket_verify_gripper(self, target: float, tolerance: float, attempts: int, settle: float, stage: str) -> Dict[str, Any]:
            last_fb = None
            last_error = float("inf")
            for attempt in range(1, max(1, int(attempts)) + 1):
                self._basket_set_gripper(float(target), 0.0, f"{stage}_{attempt}")
                time.sleep(max(0.0, float(settle)))
                fb = self._basket_feedback(quiet=True)
                if fb is None or "t" not in fb:
                    print(f"[{stage}] attempt={attempt} feedback unavailable")
                    continue
                last_fb = fb
                last_error = abs(float(fb["t"]) - float(target))
                print(f"[{stage}] attempt={attempt} target={float(target):.3f} actual={float(fb['t']):.3f} error={last_error:.3f}rad")
                if last_error <= float(tolerance):
                    return fb
            raise RuntimeError(f"{stage} failed target={float(target):.3f} error={last_error:.3f}rad feedback={last_fb}")

        def _basket_gripper_angle(self, percent: float) -> float:
            ratio = min(100.0, max(0.0, float(percent))) / 100.0
            return float(self.front.basket_close_target) + ratio * (float(self.front.basket_grip_fully_open) - float(self.front.basket_close_target))

        def _basket_descending_targets(self, start_z: float, end_z: float, step: float) -> List[float]:
            current = float(start_z)
            end = float(end_z)
            step = max(1.0, abs(float(step)))
            out: List[float] = []
            while current - step > end:
                current -= step
                out.append(current)
            if current > end + 1e-6:
                out.append(end)
            return out

        def _basket_collect_baseline(self, commanded_z: float) -> Tuple[float, float, float, float]:
            samples: List[Tuple[float, float, float, float]] = []
            for _ in range(max(3, int(self.front.basket_baseline_samples))):
                fb = self._basket_feedback(quiet=True)
                if fb is not None:
                    samples.append(tuple(float(fb.get(k, 0.0)) for k in ("torB", "torS", "torE", "torH")))
                time.sleep(max(0.03, float(self.front.basket_baseline_interval_s)))
            if len(samples) < 3:
                raise RuntimeError(f"basket torque baseline unavailable at z={commanded_z:.1f}")
            cols = list(zip(*samples))
            return tuple(float(statistics.median(col)) for col in cols)

        def _basket_build_plan(self, start_fb: Optional[Dict[str, Any]]) -> Dict[str, Any]:
            path = self._basket_calib_path()
            with open(path, "r", encoding="utf-8") as f:
                calib = json.load(f)
            points = calib.get("points", [])
            geometry = calib.get("geometry", {})
            grasp_xy = geometry.get("temporary_grasp_arm2_xy_direct")
            if not isinstance(points, list) or len(points) != 5:
                raise RuntimeError("basket calibration must contain five points")
            if not isinstance(grasp_xy, (list, tuple)) or len(grasp_xy) != 2:
                raise RuntimeError("basket temporary grasp XY missing")
            z_values = np.asarray([float(item["arm2_xyz"][2]) for item in points], dtype=np.float64)
            if z_values.size != 5 or not np.all(np.isfinite(z_values)):
                raise RuntimeError("basket calibration Z values invalid")
            board = self._basket_board_target()
            sx = float(start_fb.get("x", self.front.basket_arm2_standby_x)) if start_fb else float(self.front.basket_arm2_standby_x)
            sy = float(start_fb.get("y", self.front.basket_arm2_standby_y)) if start_fb else float(self.front.basket_arm2_standby_y)
            sz = float(start_fb.get("z", self.front.basket_arm2_standby_z)) if start_fb else float(self.front.basket_arm2_standby_z)
            tool_t = float(start_fb.get("t", self.front.basket_arm2_standby_t)) if start_fb else float(self.front.basket_arm2_standby_t)
            rim_mean = float(np.mean(z_values))
            rim_max = float(np.max(z_values))
            hover_z = max(rim_mean + float(self.front.basket_hover_offset_mm), rim_max + float(self.front.basket_rim_clearance_mm))
            safe_z = max(sz, hover_z + float(self.front.basket_transit_clearance_mm), rim_max + float(self.front.basket_transit_clearance_mm), float(board["safe_hover_z"]))
            min_safe_z = float(self.front.basket_floor_z) + float(self.front.basket_floor_clearance_mm)
            if min_safe_z >= rim_mean:
                raise RuntimeError("basket floor safety limit is not below rim mean")
            center_x, center_y = map(float, board["arm2_center_xy"])
            grasp_angle = math.atan2(float(grasp_xy[1]), float(grasp_xy[0]))
            center_angle = math.atan2(center_y, center_x)
            travel_angle = math.atan2(math.sin(center_angle - grasp_angle), math.cos(center_angle - grasp_angle))
            direction = 1.0 if travel_angle > 0.0 else -1.0
            if abs(travel_angle) < math.radians(5.0):
                direction = -1.0
            signed_extra_deg = direction * abs(float(self.front.basket_placement_extra_deg))
            theta = math.radians(signed_extra_deg)
            placement_x = math.cos(theta) * center_x - math.sin(theta) * center_y
            placement_y = math.sin(theta) * center_x + math.cos(theta) * center_y
            return {
                "calibration_path": str(path),
                "calibration_sha256": _sha256(str(path)),
                "start_xyz": [sx, sy, sz],
                "tool_t": tool_t,
                "grasp_xy": [float(grasp_xy[0]), float(grasp_xy[1])],
                "grasp_pixel_uv": geometry.get("temporary_grasp_pixel_uv"),
                "rim_mean_z": rim_mean,
                "rim_max_z": rim_max,
                "hover_z": hover_z,
                "safe_z": safe_z,
                "min_safe_z": min_safe_z,
                "board_inner_xy": list(map(float, board["arm2_inner_xy"])),
                "board_center_xy": [center_x, center_y],
                "placement_xy": [float(placement_x), float(placement_y)],
                "placement_extra_deg_signed": float(signed_extra_deg),
                "placement_reference_travel_deg": float(math.degrees(travel_angle)),
                "open_descent_angle": self._basket_gripper_angle(self.front.basket_grip_open_percent),
                "post_contact_open_angle": self._basket_gripper_angle(self.front.basket_post_contact_open_percent),
                "close_target": float(self.front.basket_close_target),
                "final_close_target": float(self.front.basket_final_close_target),
                "final_latch_torque": int(self.front.basket_final_latch_torque),
                "release_target": float(self.front.basket_release_target),
                "pickup_lift_z": min(float(self.front.basket_pickup_lift_z), float(board["safe_hover_z"])),
                "standby": [
                    float(self.front.basket_arm2_standby_x),
                    float(self.front.basket_arm2_standby_y),
                    float(self.front.basket_arm2_standby_z),
                    float(self.front.basket_arm2_standby_t),
                ],
                "arm1_motion_commands": False,
                "camera_reopen": False,
                "serial_reopen": False,
                "external_process": False,
            }

        def _prepare_basket(self) -> None:
            if not self.empty_baseline_ready:
                self.status = "BASKET BLOCKED: EMPTY BOARD E REQUIRED"
                return
            self._verify_sources_unchanged()
            if not self._ensure_camera_clear("BASKET_BEFORE_PLAN", allow_move=False):
                self.status = "BASKET BLOCKED: CAMERA-CLEAR NOT VERIFIED"
                return
            bundle = self._capture_i_frame_from_live("D56_WAIST_LIFT_LAYDOWN")
            obs = self._infer_for_action("D56_WAIST_LIFT_LAYDOWN", bundle.corrected)
            start_fb = self._basket_feedback(quiet=True) if self.args.mode != "dry-run" else None
            plan_data = self._basket_build_plan(start_fb)
            canvas = bundle.raw.copy()
            pixel = plan_data.get("grasp_pixel_uv")
            if isinstance(pixel, (list, tuple)) and len(pixel) >= 2:
                px = (int(round(float(pixel[0]))), int(round(float(pixel[1]))))
                self.cv2.drawMarker(canvas, px, (0, 255, 255), self.cv2.MARKER_CROSS, 28, 3)
                self.cv2.putText(canvas, "BASKET GRASP", (px[0] + 12, px[1] - 12), self.cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
            self.cv2.putText(canvas, "BASKET_GRASP FROZEN - ARM2 ONLY", (20, 45), self.cv2.FONT_HERSHEY_SIMPLEX, 0.72, (0, 255, 255), 2)
            proxy = SimpleNamespace(ok=True, reason="BASKET_EXACT_PLAN_FROZEN", action="BASKET_GRASP", metrics={"basket_plan": _json_safe(plan_data)}, arm_points={"arm2": {"grasp_xy": plan_data["grasp_xy"]}})
            diagnostics = {"basket_plan": _json_safe(plan_data), **self._semantic_metadata()}
            locked = base.LockedPlan(None, "BASKET_GRASP", bundle, obs, proxy, None, canvas, self.H_raw.copy(), time.time(), True, "BASKET_EXACT_PLAN_FROZEN", None, diagnostics)
            obs_record = self.recorder.save_observation(locked, "BEFORE_ACTION", self._environment_metadata())
            locked.observation_record = obs_record
            self.recorder.save_decision(locked, obs_record)
            with self.state_lock:
                self.locked = locked
                self.display_image = canvas.copy()
                self.status = "BASKET_GRASP FROZEN: ENTER EXECUTES EXACT PLAN"
            self._decorate_current_records()

        def _basket_execute_hover(self, plan: Dict[str, Any]) -> None:
            start = np.asarray(plan["start_xyz"], dtype=np.float64)
            grasp_x, grasp_y = map(float, plan["grasp_xy"])
            inner_x, inner_y = map(float, plan["board_inner_xy"])
            safe_z = float(plan["safe_z"])
            hover_z = float(plan["hover_z"])
            tool_t = float(plan["tool_t"])
            if start[2] < safe_z - 1.0:
                self._basket_move_goal(start[0], start[1], safe_z, tool_t, self.front.basket_move_speed, "BASKET_RAISE")
                self._basket_wait_waypoint("BASKET-RAISE", (start[0], start[1], safe_z))
            self._basket_move_goal(inner_x, inner_y, safe_z, tool_t, self.front.basket_move_speed, "BASKET_INNER")
            self._basket_wait_waypoint("BASKET-INNER", (inner_x, inner_y, safe_z))
            self._basket_move_goal(grasp_x, grasp_y, safe_z, tool_t, self.front.basket_move_speed, "BASKET_TRANSIT")
            self._basket_wait_waypoint("BASKET-TRANSIT", (grasp_x, grasp_y, safe_z), xy_only=True)
            self._basket_move_goal(grasp_x, grasp_y, hover_z, tool_t, self.front.basket_descent_speed, "BASKET_HOVER")
            self._basket_wait_waypoint("BASKET-HOVER", (grasp_x, grasp_y, hover_z))

        def _basket_execute_contact_descent(self, plan: Dict[str, Any]) -> Dict[str, Any]:
            grasp_x, grasp_y = map(float, plan["grasp_xy"])
            tool_t = float(plan["tool_t"])
            open_angle = float(plan["open_descent_angle"])
            open_fb = self._basket_verify_gripper(open_angle, max(0.18, float(self.front.basket_close_tolerance_rad)), 4, float(self.front.basket_gripper_settle_s), "BASKET-DESCENT-OPEN")
            current_z = float(open_fb.get("z", plan["hover_z"]))
            baseline = self._basket_collect_baseline(current_z)
            previous = baseline
            fast_targets = self._basket_descending_targets(current_z, max(float(plan["rim_mean_z"]), float(plan["min_safe_z"])), self.front.basket_fast_step_mm)
            for index, target_z in enumerate(fast_targets, 1):
                self._basket_move_goal(grasp_x, grasp_y, target_z, open_angle, self.front.basket_fast_speed, f"BASKET_FAST_{index}")
                fb = self._basket_probe_feedback(f"BASKET-FAST-{index}", target_z)
                torque = tuple(float(fb.get(k, 0.0)) for k in ("torB", "torS", "torE", "torH"))
                if max(abs(torque[1] - previous[1]), abs(torque[2] - previous[2])) >= float(self.front.basket_fast_hard_se_delta):
                    raise RuntimeError("basket fast approach shoulder/elbow hard stop")
                previous = torque
            baseline = self._basket_collect_baseline(float(plan["rim_mean_z"]))
            fb0 = self._basket_feedback(quiet=True)
            if fb0 is None:
                raise RuntimeError("basket feedback unavailable before slow descent")
            previous_z = float(fb0["z"])
            torque_count = 0
            stall_count = 0
            slow_targets = self._basket_descending_targets(min(previous_z, float(plan["rim_mean_z"])), float(plan["min_safe_z"]), self.front.basket_slow_step_mm)
            for index, target_z in enumerate(slow_targets, 1):
                before_z = previous_z
                self._basket_move_goal(grasp_x, grasp_y, target_z, open_angle, self.front.basket_slow_speed, f"BASKET_SLOW_{index}")
                fb = self._basket_probe_feedback(f"BASKET-SLOW-{index}", target_z)
                shoulder_load = float(baseline[1]) - float(fb.get("torS", 0.0))
                elbow_change = abs(float(fb.get("torE", 0.0)) - float(baseline[2]))
                z_lag = float(fb.get("z", target_z)) - float(target_z)
                candidate = shoulder_load >= float(self.front.basket_contact_shoulder_delta) and (elbow_change >= float(self.front.basket_contact_elbow_delta) or z_lag >= float(self.front.basket_contact_z_lag_mm))
                torque_count = torque_count + 1 if candidate else 0
                commanded_drop = max(0.0, before_z - float(target_z))
                actual_drop = max(0.0, before_z - float(fb.get("z", target_z)))
                stall = commanded_drop >= float(self.front.basket_stall_min_command_mm) and actual_drop <= float(self.front.basket_stall_max_actual_mm)
                stall_count = stall_count + 1 if stall else 0
                hard = max(shoulder_load, elbow_change)
                print(f"[BASKET-CONTACT] step={index} shoulder={shoulder_load:.1f} elbow={elbow_change:.1f} zlag={z_lag:.1f} confirm={torque_count}/{int(self.front.basket_contact_confirm_steps)}")
                previous_z = float(fb.get("z", target_z))
                if hard >= float(self.front.basket_hard_axis_delta):
                    raise RuntimeError("basket hard shoulder/elbow torque stop")
                if torque_count >= int(self.front.basket_contact_confirm_steps):
                    return {"contact": True, "feedback": fb, "reason": "TORQUE_CONTACT"}
                if stall_count >= int(self.front.basket_stall_confirm_steps):
                    raise RuntimeError("basket Z stall before confirmed clothing contact")
            raise RuntimeError("basket floor limit reached without confirmed clothing contact")

        def _basket_retention(self) -> Dict[str, Any]:
            scores: List[float] = []
            for index in range(max(1, int(self.front.basket_retention_samples))):
                fb = self._basket_feedback(quiet=True)
                if fb is not None and "torS" in fb and "torE" in fb:
                    scores.append(float(fb["torS"]) + float(fb["torE"]))
                if index + 1 < int(self.front.basket_retention_samples):
                    time.sleep(max(0.0, float(self.front.basket_retention_interval_s)))
            if len(scores) != max(1, int(self.front.basket_retention_samples)):
                return {"success": False, "scores": scores, "median": None, "reason": "RETENTION_FEEDBACK_INCOMPLETE"}
            median = float(statistics.median(scores))
            return {"success": median > float(self.front.basket_retention_threshold), "scores": scores, "median": median, "threshold": float(self.front.basket_retention_threshold), "reason": "OK" if median > float(self.front.basket_retention_threshold) else "RETENTION_LOW"}

        def _basket_final_grip_latch(self, plan: Dict[str, Any]) -> Dict[str, Any]:
            final_close = float(plan.get("final_close_target", self.front.basket_final_close_target))
            torque = int(plan.get("final_latch_torque", self.front.basket_final_latch_torque))
            self._basket_set_gripper(final_close, float(self.front.basket_final_latch_settle_s), "BASKET-FINAL-CLOSE-3_32")
            arm = self._basket_arm2()
            sent = False
            last = None
            if hasattr(arm, "send"):
                attempts = [
                    lambda: arm.send({"T": 107, "tor": torque}, delay=0.20, stage="BASKET-FINAL-TORQUE-LATCH", caller="bottom_vla_basket"),
                    lambda: arm.send({"T": 107, "tor": torque}, delay=0.20),
                    lambda: arm.send({"T": 107, "tor": torque}),
                ]
                for method in attempts:
                    try:
                        method()
                        sent = True
                        break
                    except TypeError as exc:
                        last = exc
            if not sent:
                raise RuntimeError(f"ARM2 T107 torque latch unavailable: {last!r}")
            time.sleep(max(0.0, float(self.front.basket_final_latch_settle_s)))
            fb = self._basket_feedback(quiet=True)
            actual = None if fb is None or "t" not in fb else float(fb["t"])
            print(f"[BASKET-FINAL-GRIP-LATCH] close={final_close:.2f}rad torque={torque} feedback={actual}")
            return {"final_close_target": final_close, "torque_latch": torque, "feedback_angle": actual}

        def _basket_release_verified(self, target: Optional[float] = None, stage: str = "BASKET-RELEASE") -> Dict[str, Any]:
            value = float(self.front.basket_release_target if target is None else target)
            return self._basket_verify_gripper(value, float(self.front.basket_release_tolerance_rad), int(self.front.basket_release_attempts), 0.45, stage)

        def _basket_return_arm2_standby(self, release_verified: bool) -> np.ndarray:
            if not release_verified:
                raise RuntimeError("ARM2 standby blocked until release is verified")
            x = float(self.front.basket_arm2_standby_x)
            y = float(self.front.basket_arm2_standby_y)
            z = float(self.front.basket_arm2_standby_z)
            t = float(self.front.basket_arm2_standby_t)
            self._basket_move_goal(x, y, z, t, self.front.basket_standby_speed, "BASKET_ARM2_STANDBY")
            return self._basket_wait_waypoint("BASKET-ARM2-STANDBY", (x, y, z))

        def _basket_capture_after(self) -> Optional[Dict[str, Any]]:
            if not self._ensure_camera_clear("BASKET_AFTER_STANDBY", allow_move=False):
                raise RuntimeError("camera-clear verification failed after ARM2-only standby return")
            bundle = self._capture_i_frame_from_live("D56_WAIST_LIFT_LAYDOWN")
            obs = self._infer_for_action("D56_WAIST_LIFT_LAYDOWN", bundle.corrected)
            proxy = SimpleNamespace(ok=True, reason="AFTER_ACTION", action="BASKET_GRASP", metrics={}, arm_points={})
            locked = base.LockedPlan(None, "BASKET_GRASP", bundle, obs, proxy, None, bundle.raw.copy(), self.H_raw.copy(), time.time(), True, "AFTER_ACTION", None, {**self._semantic_metadata("BASKET_GRASP")})
            return self.recorder.save_observation(locked, "AFTER_ACTION", self._environment_metadata())

        def _execute_basket(self, locked: Any) -> None:
            age = time.time() - float(locked.created_at)
            if age > float(self.args.locked_plan_max_age_s):
                print(f"[ENTER] BASKET blocked: frozen plan age={age:.1f}s")
                self._invalidate_for_new_action("BASKET_PLAN_STALE")
                return
            plan = copy.deepcopy((locked.diagnostics or {}).get("basket_plan"))
            if not isinstance(plan, dict):
                print("[ENTER] BASKET blocked: frozen basket plan missing")
                return
            try:
                self.recorder.open_transition(locked)
            except Exception as exc:
                print(f"[ENTER] BASKET transition blocked: {exc}")
                return
            with self.state_lock:
                self.motion_busy = True
                self.locked = None
                self.status = "BASKET_GRASP EXECUTING EXACT FROZEN PLAN"
            self.last_executed_semantic = "BASKET_GRASP"
            self.pending_semantic = "BASKET_GRASP"

            def worker() -> None:
                sent = False
                success = False
                after_record = None
                report: Dict[str, Any] = {
                    "arm1_motion_commands": 0,
                    "persistent_arm2_session": True,
                    "serial_reopen": False,
                    "camera_reopen": False,
                    "external_process": False,
                    "grasp_success": False,
                    "close_achieved_angle": None,
                    "lift_achieved_z": None,
                    "release_achieved_angle": None,
                    "standby_reached": False,
                    "camera_clear_verified": False,
                }
                detail = "NOT_STARTED"
                release_verified = False
                try:
                    if self.args.mode == "dry-run":
                        report.update({"grasp_success": True, "close_achieved_angle": float(plan["close_target"]), "lift_achieved_z": float(plan["pickup_lift_z"]), "release_achieved_angle": float(plan["release_target"]), "standby_reached": True, "camera_clear_verified": True})
                        success = True
                        detail = "DRY_RUN"
                    else:
                        sent = self.args.mode == "physical"
                        self._basket_torque_on()
                        self._basket_execute_hover(plan)
                        contact = self._basket_execute_contact_descent(plan)
                        report["contact"] = _json_safe(contact.get("reason"))
                        wider_fb = self._basket_verify_gripper(float(plan["post_contact_open_angle"]), float(self.front.basket_close_tolerance_rad), 4, float(self.front.basket_post_contact_open_settle_s), "BASKET-POST-CONTACT-OPEN")
                        report["post_contact_open_angle"] = float(wider_fb["t"])
                        close_fb = self._basket_verify_gripper(float(plan["close_target"]), float(self.front.basket_close_tolerance_rad), int(self.front.basket_close_attempts), float(self.front.basket_gripper_settle_s), "BASKET-CLOSE-VERIFY")
                        close_t = float(close_fb["t"])
                        report["close_achieved_angle"] = close_t
                        report["close_verified"] = True
                        final_latch = self._basket_final_grip_latch(plan)
                        report["final_grip_latch"] = _json_safe(final_latch)
                        report["final_close_target"] = float(final_latch["final_close_target"])
                        report["final_latch_torque"] = int(final_latch["torque_latch"])
                        report["final_close_feedback_angle"] = final_latch.get("feedback_angle")
                        lift_x = float(close_fb["x"])
                        lift_y = float(close_fb["y"])
                        lift_target = float(plan["pickup_lift_z"])
                        lift_result = self._basket_lift_adaptive(lift_x, lift_y, float(close_fb["z"]), lift_target, float(plan["close_target"]))
                        lift_fb = lift_result["feedback"]
                        if not bool(lift_result.get("reached")) and not bool(lift_result.get("stalled")):
                            raise RuntimeError("basket lift neither reached target nor confirmed saturation")
                        lift_pose = np.asarray([float(lift_fb["x"]), float(lift_fb["y"]), float(lift_fb["z"])], dtype=np.float64)
                        report["lift_achieved_z"] = float(lift_pose[2])
                        report["lift_reached_target"] = bool(lift_result.get("reached"))
                        report["lift_saturated"] = bool(lift_result.get("stalled"))
                        retention = self._basket_retention()
                        report["retention"] = _json_safe(retention)
                        if not bool(retention.get("success", False)):
                            retention_release_fb = self._basket_release_verified(float(self.front.basket_grip_fully_open), "BASKET-RETENTION-FAIL-RELEASE")
                            release_verified = True
                            report["release_achieved_angle"] = float(retention_release_fb["t"])
                            self._basket_return_arm2_standby(True)
                            report["standby_reached"] = True
                            raise RuntimeError(f"basket grasp retention failed: {retention.get('reason')}")
                        report["grasp_success"] = True
                        center_x, center_y = map(float, plan["board_center_xy"])
                        transit_z = float(lift_pose[2])
                        self._basket_move_goal(center_x, center_y, transit_z, float(plan["close_target"]), self.front.basket_board_transit_speed, "BASKET-BOARD-CENTER")
                        placement_x, placement_y = map(float, plan.get("placement_xy", plan["board_center_xy"]))
                        placement_deg = float(plan.get("placement_extra_deg_signed", 0.0))
                        placement_distance = float(np.linalg.norm(np.asarray([placement_x - center_x, placement_y - center_y], dtype=np.float64)))
                        if placement_distance > 1.0:
                            center_pose, center_error = self._basket_wait_transit_near(
                                "BASKET-BOARD-CENTER-BLEND",
                                (center_x, center_y, transit_z),
                                float(self.front.basket_board_center_blend_mm),
                            )
                            report["board_center_actual"] = center_pose.astype(float).tolist()
                            report["board_center_blend_error_mm"] = float(center_error)
                            report["board_center_blend_threshold_mm"] = float(max(float(self.front.basket_move_tolerance_mm), float(self.front.basket_board_center_blend_mm)))
                            stage = f"BASKET-PLACEMENT-ROTATE-{abs(placement_deg):.0f}"
                            print(f"[BASKET-CENTER-BLEND] center_error={center_error:.1f}mm -> continuous final target {placement_deg:+.1f}deg")
                            self._basket_move_goal(placement_x, placement_y, transit_z, float(plan["close_target"]), self.front.basket_placement_rotate_speed, stage)
                            placement_pose = self._basket_wait_waypoint(stage, (placement_x, placement_y, transit_z), xy_only=True)
                        else:
                            center_pose = self._basket_wait_waypoint("BASKET-BOARD-CENTER", (center_x, center_y, transit_z), xy_only=True)
                            report["board_center_actual"] = center_pose.astype(float).tolist()
                            placement_pose = center_pose
                        report["placement_extra_deg_signed"] = placement_deg
                        report["placement_target_xy"] = [placement_x, placement_y]
                        report["placement_actual"] = placement_pose.astype(float).tolist()
                        print(f"[BASKET-PLACEMENT-ROTATE] extra={placement_deg:+.1f}deg target=({placement_x:.1f},{placement_y:.1f})")
                        lower_x = float(placement_pose[0])
                        lower_y = float(placement_pose[1])
                        lower_start_z = float(placement_pose[2])
                        lower_target_z = max(float(self.front.basket_placement_min_z), lower_start_z - max(0.0, float(self.front.basket_placement_lower_mm)))
                        if lower_target_z < lower_start_z - 1.0:
                            print(f"[BASKET-PLACE-LOWER] sameXY=({lower_x:.1f},{lower_y:.1f}) z={lower_start_z:.1f}->{lower_target_z:.1f} speed={float(self.front.basket_placement_lower_speed):.2f} gripper=CLOSED")
                            self._basket_move_goal(lower_x, lower_y, lower_target_z, float(plan["close_target"]), float(self.front.basket_placement_lower_speed), "BASKET-PLACE-SLOW-DESCENT-CLOSED")
                            lower_pose = self._basket_wait_waypoint("BASKET-PLACE-SLOW-DESCENT-CLOSED", (lower_x, lower_y, lower_target_z))
                        else:
                            lower_pose = placement_pose
                        report["placement_lower_start_z"] = lower_start_z
                        report["placement_lower_target_z"] = lower_target_z
                        report["placement_lower_actual"] = lower_pose.astype(float).tolist()
                        release_fb = self._basket_release_verified(float(plan["release_target"]), "BASKET-RELEASE-VERIFY")
                        release_verified = True
                        report["release_achieved_angle"] = float(release_fb["t"])
                        standby_pose = self._basket_return_arm2_standby(True)
                        report["standby_reached"] = True
                        report["standby_actual"] = standby_pose.astype(float).tolist()
                        after_record = self._basket_capture_after()
                        report["camera_clear_verified"] = True
                        success = True
                        detail = "BASKET_GRASP_ARM2_ONLY_COMPLETE"
                except Exception as exc:
                    detail = repr(exc)
                    print(f"[BASKET-ERROR] {exc!r}")
                    if self.args.mode != "dry-run" and not release_verified:
                        try:
                            fb = self._basket_release_verified(float(self.front.basket_grip_fully_open), "BASKET-FAILSAFE-RELEASE")
                            release_verified = True
                            report["failsafe_release_angle"] = float(fb["t"])
                            report["release_achieved_angle"] = float(fb["t"])
                        except Exception as release_exc:
                            report["failsafe_release_error"] = repr(release_exc)
                    if self.args.mode != "dry-run" and release_verified and not report.get("standby_reached"):
                        try:
                            standby_pose = self._basket_return_arm2_standby(True)
                            report["standby_reached"] = True
                            report["standby_actual"] = standby_pose.astype(float).tolist()
                        except Exception as standby_exc:
                            report["standby_error"] = repr(standby_exc)
                    if self.args.mode != "dry-run" and report.get("standby_reached") and after_record is None:
                        try:
                            after_record = self._basket_capture_after()
                            report["camera_clear_verified"] = True
                        except Exception as after_exc:
                            report["after_capture_error"] = repr(after_exc)
                finally:
                    committed = bool(sent and report.get("lift_achieved_z") is not None and self.args.mode == "physical")
                    if committed:
                        try:
                            self.recorder.note_motion_committed()
                        except Exception:
                            pass
                    self.recorder.complete_transition({
                        "execution_success": bool(success),
                        "execution_sent": bool(sent),
                        "garment_motion_committed": committed,
                        "execution_detail": str(detail),
                        "motion_parameters": {
                            "semantic_action": "BASKET_GRASP",
                            "source": "50-1-integrated-persistent-arm2",
                            "frozen_plan": _json_safe(plan),
                            "execution_report": _json_safe(report),
                        },
                        "automatic_result": None,
                    }, after_record)
                    self._decorate_current_records("BASKET_GRASP")
                    self._finalize_basket_without_review(bool(success), report)
                    with self.state_lock:
                        self.motion_busy = False
                        self.locked = None
                        self.semantic_selected = None
                        self.pending_semantic = None
                        self.last_executed_semantic = None
                        self.selected_action = None
                        self.status = "BASKET_GRASP COMPLETE: SELECT NEXT ACTION"
                    print(f"[BASKET-DONE] success={success} detail={detail}; no G/B/K or Y/N review")

            self.worker = threading.Thread(target=worker, name="bottom-vla-basket-exec", daemon=True)
            self.worker.start()

        def _record_id(self, record: Any) -> Optional[str]:
            if not isinstance(record, dict):
                return None
            value = record.get("id") or record.get("observation_id") or record.get("decision_id")
            return None if value is None else str(value)

        def _discard_interrupted_prepare_records(self, generation: int, action: str, before_obs_id: Optional[str], before_decision_id: Optional[str]) -> None:
            fields = {
                "operator_interrupt_discarded": True,
                "operator_interrupt_reason": "HUMAN_ACTION_OVERRIDE_DURING_INFERENCE",
                "inference_generation": int(generation),
                "interrupted_internal_action": str(action),
                "pending_executable": False,
                "training_eligible": False,
            }
            with self.recorder.lock:
                obs = getattr(self.recorder, "current_observation", None)
                dec = getattr(self.recorder, "current_decision", None)
                obs_changed = self._record_id(obs) is not None and self._record_id(obs) != before_obs_id
                dec_changed = self._record_id(dec) is not None and self._record_id(dec) != before_decision_id
                if obs_changed:
                    _patch_json_record(obs, fields)
                    self.recorder.current_observation = None
                if dec_changed:
                    _patch_json_record(dec, fields)
                    self.recorder.current_decision = None
            print(f"[HUMAN-INTERRUPT-DISCARD] generation={generation} action={action} obs={obs_changed} decision={dec_changed}")

        def _launch_prepare_worker(self, action: str, generation: int, before_obs_id: Optional[str], before_decision_id: Optional[str]) -> None:
            semantic = INTERNAL_TO_SEMANTIC.get(str(action), str(self.pending_semantic or action))

            def infer_worker() -> None:
                t0 = time.monotonic()
                with self.state_lock:
                    pre_stale = int(generation) != int(self._prepare_generation)
                if pre_stale:
                    print(f"[I-SKIP-STALE] action={semantic} generation={generation}")
                else:
                    try:
                        self._prepare_action()
                    except Exception as exc:
                        print(f"[I-ERROR] action={semantic} generation={generation} error={exc!r}")
                total = time.monotonic() - t0
                next_semantic = None
                next_generation = None
                next_action = None
                with self.state_lock:
                    stale = int(generation) != int(self._prepare_generation)
                    if stale:
                        next_semantic = self._queued_human_action
                        self._queued_human_action = None
                        self.locked = None
                        self.display_image = None
                    if stale and next_semantic in SEMANTIC_TO_INTERNAL:
                        self.semantic_selected = next_semantic
                        self.pending_semantic = next_semantic
                        self._prepare_origin = "HUMAN"
                        self._auto_at_selection = str(self.auto_recommended)
                        self._human_selected = next_semantic
                        self.plan_origin = "HUMAN"
                        self.selected_action = SEMANTIC_TO_INTERNAL[next_semantic]
                        self.status = f"{next_semantic} PREPARING ONE FROZEN PLAN"
                        self._prepare_generation += 1
                        next_generation = int(self._prepare_generation)
                        self._active_prepare_generation = next_generation
                        next_action = str(self.selected_action)
                        self.inference_busy = True
                        self.inference_action = next_action
                    else:
                        self.inference_busy = False
                        self.inference_action = None
                        self._active_prepare_generation = 0
                print(f"[VLA-I-LATENCY] action={semantic} generation={generation} worker_total={total:.3f}s stale={stale}")
                if stale:
                    self._discard_interrupted_prepare_records(generation, action, before_obs_id, before_decision_id)
                    with self.align_runtime_lock:
                        self.align_runtime.clear()
                    with self.d60_runtime_lock:
                        self.d60_runtime.clear()
                if next_semantic in SEMANTIC_TO_INTERNAL and next_generation is not None and next_action is not None:
                    with self.recorder.lock:
                        next_obs_id = self._record_id(getattr(self.recorder, "current_observation", None))
                        next_decision_id = self._record_id(getattr(self.recorder, "current_decision", None))
                    print(f"[HUMAN-INTERRUPT-START] semantic={next_semantic} generation={next_generation} auto={self._auto_at_selection}")
                    self._launch_prepare_worker(next_action, next_generation, next_obs_id, next_decision_id)

            self.inference_worker = threading.Thread(
                target=infer_worker, name=f"bottom-vla-{str(semantic).lower()}-inference-g{generation}", daemon=True
            )
            self.inference_worker.start()

        def _start_prepare_action(self) -> None:
            with self.state_lock:
                if self.motion_busy:
                    print("[I] blocked during motion")
                    return
                if self.inference_busy:
                    print(f"[I] {self.inference_action or 'inference'} already running")
                    return
                action = self.selected_action
                if action is None:
                    print("[I] select an action first")
                    return
                self._prepare_generation += 1
                generation = int(self._prepare_generation)
                self._active_prepare_generation = generation
                self.inference_busy = True
                self.inference_action = str(action)
                self.status = f"{self.pending_semantic or action} I-LOCK RUNNING - CAMERA LOOP ALIVE"
            with self.recorder.lock:
                before_obs_id = self._record_id(getattr(self.recorder, "current_observation", None))
                before_decision_id = self._record_id(getattr(self.recorder, "current_decision", None))
            print(f"[I-START] semantic={self.pending_semantic or action} generation={generation} origin={self._prepare_origin}")
            self._launch_prepare_worker(str(action), generation, before_obs_id, before_decision_id)

        def _select_semantic(self, semantic: str, origin: str = "HUMAN") -> None:
            semantic = str(semantic).upper()
            origin = str(origin).upper()
            if semantic == "REJUDGE":
                self._rejudge()
                return
            if semantic not in SEMANTIC_TO_INTERNAL:
                print(f"[ACTION] unsupported semantic={semantic}")
                return
            with self.state_lock:
                if self.motion_busy:
                    print("[ACTION] blocked during motion")
                    return
                if self.recorder.review_pending():
                    print("[ACTION] blocked: complete review and Y/N first")
                    return
                if self.inference_busy:
                    if origin == "HUMAN":
                        current = str(self.pending_semantic or INTERNAL_TO_SEMANTIC.get(str(self.inference_action), self.inference_action or "INFERENCE"))
                        if semantic == current and self._queued_human_action is None:
                            print(f"[ACTION] {semantic} already being prepared")
                            return
                        self._prepare_generation += 1
                        self._queued_human_action = semantic
                        self._queued_human_origin = "HUMAN"
                        self.status = f"HUMAN OVERRIDE QUEUED: {semantic}; DISCARDING {current}"
                        print(f"[HUMAN-INTERRUPT] current={current} -> selected={semantic}; old result will be discarded")
                        return
                    print("[ACTION] blocked while a frozen plan is being prepared")
                    return
            self._invalidate_for_new_action("SEMANTIC_ACTION_CHANGED")
            self.semantic_selected = semantic
            self.pending_semantic = semantic
            self._prepare_origin = origin
            self._auto_at_selection = str(self.auto_recommended)
            self._human_selected = semantic
            self.plan_origin = self._prepare_origin
            self.selected_action = SEMANTIC_TO_INTERNAL[semantic]
            self.status = f"{semantic} PREPARING ONE FROZEN PLAN"
            print(f"[ACTION] semantic={semantic} origin={self._prepare_origin} auto={self._auto_at_selection}")
            self._start_prepare_action()

        def _rejudge(self) -> None:
            with self.state_lock:
                if self.motion_busy or self.inference_busy:
                    print("[REJUDGE] blocked while worker is active")
                    return
                if self.recorder.review_pending():
                    print("[REJUDGE] blocked: complete review and Y/N first")
                    return
            target = self.semantic_selected or self.auto_recommended
            if target == "REJUDGE":
                target = self.auto_recommended
            self._invalidate_for_new_action("REJUDGE")
            print(f"[REJUDGE] one fresh frozen plan for {target}")
            self._select_semantic(target, origin="REJUDGE")

        def _prepare_position_adjust(self) -> None:
            with self.state_lock:
                if self.motion_busy:
                    print("[POSITION_ADJUST] blocked during motion")
                    return
                if self.recorder.review_pending():
                    print("[POSITION_ADJUST] blocked: complete review and Y/N first")
                    return
            if not self.empty_baseline_ready:
                self.status = "POSITION_ADJUST BLOCKED: EMPTY BOARD E REQUIRED"
                print("[POSITION_ADJUST] press E on empty board first")
                return
            perf0 = time.monotonic()
            self._verify_sources_unchanged()
            if not self._ensure_camera_clear("POSITION_ADJUST_BEFORE_PLAN", allow_move=False):
                self.status = "POSITION_ADJUST BLOCKED: CAMERA-CLEAR NOT VERIFIED"
                return
            t = time.monotonic()
            bundle = self._capture_i_frame_from_live("D58_CIRC_POSITION")
            capture_dt = time.monotonic() - t
            t = time.monotonic()
            obs = self._infer_for_action("D58_CIRC_POSITION", bundle.corrected)
            infer_dt = time.monotonic() - t
            mask = getattr(obs, "mask", None)
            pose = getattr(obs, "pose", None)
            t = time.monotonic()
            if mask is None:
                plan = self.d58.D58Plan(False, "D58 mask unavailable")
            else:
                plan = self.d58.build_d58_plan(
                    bundle.corrected, mask, pose, self.H_corrected, self.config, self.d58_args
                )
                strengthen = getattr(self, "_main33_strengthen_d58_plan", None)
                if callable(strengthen):
                    plan = strengthen(bundle.corrected, mask, pose, plan)
            plan_dt = time.monotonic() - t
            t = time.monotonic()
            canvas = self._operator_overlay(bundle, "D58_CIRC_POSITION", obs, plan, None)
            overlay_dt = time.monotonic() - t
            had_overlay = isinstance(getattr(plan, "overlay", None), np.ndarray)
            if hasattr(plan, "overlay"):
                plan.overlay = None
            plan_ok = bool(getattr(plan, "ok", False))
            reason = str(getattr(plan, "reason", ""))
            diagnostics = {
                "d58_target_source": str(getattr(plan, "target_source", "NONE")),
                "d58_move_mm": float(getattr(plan, "move_mm", 0.0) or 0.0),
                "d58_selected_arm": str(getattr(plan, "selected_arm", "")),
                "position_plan_overlay_json_excluded": True,
                "position_plan_overlay_was_present": bool(had_overlay),
                "original_source_sha256": self.position_source_sha256,
                "camera_geometry_path": "CORRECTED+H",
                **self._semantic_metadata(),
            }
            locked = base.LockedPlan(
                None, "D58_CIRC_POSITION", bundle, obs, plan, None, canvas, self.H_corrected.copy(),
                time.time(), plan_ok, reason, None if plan_ok else "NO_SAFE_PLAN", diagnostics
            )
            t = time.monotonic()
            obs_record = self.recorder.save_observation(locked, "BEFORE_ACTION", self._environment_metadata())
            save_dt = time.monotonic() - t
            locked.observation_record = obs_record
            t = time.monotonic()
            self.recorder.save_decision(locked, obs_record)
            decision_dt = time.monotonic() - t
            with self.state_lock:
                self.locked = locked
                self.display_image = canvas.copy()
                self.status = "POSITION_ADJUST FROZEN: ENTER EXECUTES EXACT DISPLAYED PLAN" if plan_ok else f"POSITION_ADJUST NO PLAN: {reason}"
            self._decorate_current_records()
            print(
                f"[PERF-I-POSITION] capture={capture_dt:.3f}s infer={infer_dt:.3f}s plan={plan_dt:.3f}s "
                f"overlay={overlay_dt:.3f}s save={save_dt:.3f}s decision={decision_dt:.3f}s "
                f"total={(time.monotonic()-perf0):.3f}s overlay_json=EXCLUDED"
            )

        def _prepare_action(self) -> None:
            if self.selected_action == "BASKET_GRASP":
                self._prepare_basket()
                return
            if self.selected_action == "D58_CIRC_POSITION":
                self._prepare_position_adjust()
                return
            if self.selected_action == "WAIST_PULL_LAYDOWN":
                self._prepare_waist_pull_laydown()
                return
            if self.selected_action == "ALIGN":
                self._prepare_align()
                return
            if self.selected_action == "FINISH":
                self._prepare_finish()
                return
            super()._prepare_action()
            with self.state_lock:
                locked = self.locked
            if locked is not None:
                locked.diagnostics = dict(getattr(locked, "diagnostics", {}) or {})
                locked.diagnostics.update(self._semantic_metadata())
                self._decorate_current_records()
                self.status = f"{self.pending_semantic} FROZEN: ENTER EXECUTES EXACT DISPLAYED PLAN"

        def _d60_build_source_plan(self, planner_obs: Any) -> Any:
            if planner_obs is None or getattr(planner_obs, "mask", None) is None:
                return None
            plan = self.d56._d42_build_hybrid_grasp_plan(
                planner_obs, self.H_raw, self.config, self.cfg56, self.d56_args
            )
            if hasattr(self.d56, "_d56_apply_arm1_waistward_correction"):
                plan = self.d56._d56_apply_arm1_waistward_correction(
                    plan, planner_obs, self.H_raw, self.config, self.cfg56, self.d56_args
                )
            return plan

        def _d60_pathological_ribbon(self, plan: Any) -> Tuple[bool, Dict[str, Any]]:
            metrics = dict(getattr(plan, "metrics", {}) or {}) if plan is not None else {}
            warning = bool(metrics.get("d56v60_5_pair_sanity_warning_only", False))
            rescue = float(metrics.get("d56v60_5_ribbon_max_rescue_mm", 0.0) or 0.0)
            depths = metrics.get("d56v60_5_ribbon_depths_mm", ())
            try:
                a2 = float(depths[0])
                a1 = float(depths[1])
            except Exception:
                a2 = float(metrics.get("d56v47_arm2_body_offset_mm", 0.0) or 0.0)
                a1 = float(metrics.get("d56v47_arm1_body_offset_mm", 0.0) or 0.0)
            ratio = max(a2, a1) / max(1.0, min(a2, a1)) if max(a2, a1) > 0.0 else 1.0
            diff = abs(a2 - a1)
            bad = bool(warning or rescue > 45.0 + 1e-6)
            return bad, {
                "warning": warning,
                "max_rescue_mm": rescue,
                "arm2_body_depth_mm": a2,
                "arm1_body_depth_mm": a1,
                "depth_diff_mm": diff,
                "depth_ratio": ratio,
            }

        def _d60_without_selected_ribbon(self, planner_obs: Any, safety: Dict[str, Any]) -> Any:
            fallback_obs = copy.copy(planner_obs)
            report = copy.deepcopy(getattr(planner_obs, "d56v7_waist_observer", None))
            if not isinstance(report, dict):
                return fallback_obs
            selected = report.get("selected")
            if isinstance(selected, dict):
                report["selected_rejected_for_grasp"] = selected
            report["selected"] = None
            report["reason"] = "BOTTOM_VLA_D60_UNSAFE_RIBBON_REJECTED"
            report["bottom_vla_d60_safety"] = copy.deepcopy(safety)
            try:
                fallback_obs.d56v7_waist_observer = report
            except Exception:
                pass
            return fallback_obs

        def _d60_plan_from_obs(self, obs: Any) -> Tuple[Any, Dict[str, Any]]:
            planner_obs = obs
            diagnostics: Dict[str, Any] = {"source_pipeline": "60-14_D42_SINGLE_FROZEN_OBSERVATION"}
            gross = dict(getattr(obs, "d45_gross_mask_validation", {}) or {}) if obs is not None else {}
            diagnostics["gross_mask_validation"] = _json_safe(gross)
            if obs is not None and getattr(obs, "mask", None) is not None and bool(gross.get("rejected", False)):
                planner_obs = copy.copy(obs)
                planner_obs.mask = None
                planner_obs.valid = False
                planner_obs.reason = "D45 gross board-mask veto: " + str(gross.get("reason", "unsafe mask"))
            plan = self._d60_build_source_plan(planner_obs)
            bad, safety = self._d60_pathological_ribbon(plan)
            diagnostics["ribbon_safety"] = _json_safe(safety)
            diagnostics["ribbon_rejected"] = bool(bad)
            if bad and planner_obs is not None and getattr(planner_obs, "mask", None) is not None:
                print(
                    "[D60-RIBBON-SAFETY] reject source ribbon "
                    f"warning={safety['warning']} rescue={safety['max_rescue_mm']:.1f}mm "
                    f"depths={safety['arm2_body_depth_mm']:.1f}/{safety['arm1_body_depth_mm']:.1f}mm "
                    "-> rerun 60-14 D42 on SAME frozen observation with ribbon disabled"
                )
                fallback_obs = self._d60_without_selected_ribbon(planner_obs, safety)
                fallback_plan = self._d60_build_source_plan(fallback_obs)
                if fallback_plan is not None and bool(getattr(fallback_plan, "ok", False)):
                    plan = fallback_plan
                    diagnostics["fallback_used"] = True
                    diagnostics["fallback_reason"] = "UNSAFE_RIBBON_TO_EXISTING_60_13_SAFE_FALLBACK"
                    print(f"[D60-RIBBON-SAFETY] fallback accepted mode={str(getattr(fallback_plan, 'metrics', {}).get('d42_plan_mode', '-'))}")
                else:
                    plan = fallback_plan
                    diagnostics["fallback_used"] = True
                    diagnostics["fallback_reason"] = "UNSAFE_RIBBON_FALLBACK_FAILED_BLOCK"
                    print(f"[D60-RIBBON-SAFETY] fallback failed; WAIST_PULL_LAYDOWN blocked reason={_get_plan_reason(fallback_plan, 'NO_SAFE_FALLBACK')}")
            else:
                diagnostics["fallback_used"] = False
            return plan, diagnostics

        def _d60_pose_waist_frame(self, obs: Any) -> Dict[str, Any]:
            report: Dict[str, Any] = {"ok": False, "source": "POSE_WAIST_LOCAL_GATE"}
            if obs is None or getattr(obs, "pose", None) is None or getattr(obs, "mask", None) is None:
                report["reason"] = "POSE_OR_MASK_UNAVAILABLE"
                return report
            pose = obs.pose
            key_board = dict(getattr(pose, "keypoints_board", {}) or {})
            key_conf = dict(getattr(pose, "keypoint_conf", {}) or {})
            def pick(key: str, attr: str) -> Tuple[Optional[np.ndarray], Optional[float]]:
                raw = key_board.get(key)
                if raw is None:
                    raw = getattr(pose, attr, None)
                try:
                    q = np.asarray(raw, dtype=np.float32).reshape(2)
                except Exception:
                    return None, None
                if not np.all(np.isfinite(q)):
                    return None, None
                conf = None
                try:
                    if key in key_conf:
                        conf = float(key_conf[key])
                except Exception:
                    conf = None
                return q, conf
            wl, cl = pick("waist_img_left", "waist_left")
            wc, cc = pick("waist_center", "waist_center")
            wr, cr = pick("waist_img_right", "waist_right")
            if wl is None or wr is None:
                report["reason"] = "RAW_POSE_WAIST_ENDPOINTS_UNAVAILABLE"
                return report
            conf_rows = [v for v in (cl, cc, cr) if v is not None and math.isfinite(v)]
            min_conf = float(self.front.d60_pose_waist_min_conf)
            if conf_rows and min(conf_rows) < min_conf:
                report.update({"reason": "RAW_POSE_WAIST_CONF_LOW", "conf": {"left": cl, "center": cc, "right": cr}, "min_conf": min_conf})
                return report
            waist_vec = wr - wl
            width = float(np.linalg.norm(waist_vec))
            min_width = float(self.front.d60_pose_waist_min_width_mm)
            max_width = float(self.front.d60_pose_waist_max_width_mm)
            if not (min_width <= width <= max_width):
                report.update({"reason": "RAW_POSE_WAIST_WIDTH_INVALID", "width_mm": width, "limits": [min_width, max_width]})
                return report
            waist_u = waist_vec / max(width, 1e-6)
            midpoint = 0.5 * (wl + wr)
            center_rebuilt = False
            if wc is None:
                wc = midpoint.copy()
                center_rebuilt = True
            else:
                denom = max(float(np.dot(waist_vec, waist_vec)), 1e-6)
                projection = float(np.dot(wc - wl, waist_vec) / denom)
                projected = wl + np.clip(projection, 0.0, 1.0) * waist_vec
                line_dist = float(np.linalg.norm(wc - projected))
                if projection < -0.15 or projection > 1.15 or line_dist > 65.0:
                    wc = midpoint.copy()
                    center_rebuilt = True
            mask_u8 = np.asarray(obs.mask.mask_u8, dtype=np.uint8)
            near_limit = float(self.front.d60_pose_waist_mask_near_px)
            inv = (mask_u8 == 0).astype(np.uint8)
            outside_dist = self.cv2.distanceTransform(inv, self.cv2.DIST_L2, 5)
            support: Dict[str, Any] = {}
            for name, q in (("left", wl), ("center", wc), ("right", wr)):
                px = self.d56.board_to_pixel(self.H_raw, float(q[0]), float(q[1])) if callable(getattr(self.d56, "board_to_pixel", None)) else None
                if px is None:
                    support[name] = {"ok": True, "outside_px": None}
                    continue
                u = int(round(float(px[0])))
                v = int(round(float(px[1])))
                if not (0 <= u < mask_u8.shape[1] and 0 <= v < mask_u8.shape[0]):
                    support[name] = {"ok": False, "outside_px": None}
                    continue
                d = float(outside_dist[v, u])
                support[name] = {"ok": bool(d <= near_limit), "outside_px": d}
            if not all(bool(row.get("ok", False)) for row in support.values()):
                report.update({"reason": "RAW_POSE_WAIST_TOO_FAR_FROM_MASK", "mask_support": support, "mask_near_px": near_limit})
                return report
            def body_point(key: str, attr: str) -> Optional[np.ndarray]:
                raw = key_board.get(key)
                if raw is None:
                    raw = getattr(pose, attr, None)
                try:
                    q = np.asarray(raw, dtype=np.float32).reshape(2)
                except Exception:
                    return None
                return q if np.all(np.isfinite(q)) else None
            body_target = body_point("crotch", "crotch")
            if body_target is None:
                body_target = body_point("", "lower_center")
            if body_target is None:
                try:
                    body_target = np.asarray(obs.mask.center_board, dtype=np.float32).reshape(2)
                except Exception:
                    body_target = None
            if body_target is None:
                report["reason"] = "RAW_POSE_BODY_DIRECTION_UNAVAILABLE"
                return report
            body_vec = body_target - wc
            body_vec = body_vec - waist_u * float(np.dot(body_vec, waist_u))
            body_norm = float(np.linalg.norm(body_vec))
            if body_norm <= 1e-6:
                report["reason"] = "RAW_POSE_BODY_DIRECTION_DEGENERATE"
                return report
            body_u = body_vec / body_norm
            try:
                mask_center = np.asarray(obs.mask.center_board, dtype=np.float32).reshape(2)
                if float(np.dot(mask_center - wc, body_u)) < 0.0:
                    body_u = -body_u
            except Exception:
                pass
            report.update({
                "ok": True,
                "source": "POSE_WAIST_LOCAL_GATE",
                "center": wc,
                "waist_u": waist_u,
                "body_u": body_u,
                "width_mm": width,
                "endpoints": [wl, wr],
                "conf": {"left": cl, "center": cc, "right": cr},
                "center_rebuilt": bool(center_rebuilt),
                "mask_support": support,
                "reason": "OK",
            })
            return report

        def _d60_waist_frame(self, obs: Any) -> Dict[str, Any]:
            report: Dict[str, Any] = {"ok": False, "source": "NONE"}
            if obs is None or getattr(obs, "mask", None) is None:
                report["reason"] = "MASK_UNAVAILABLE"
                return report
            pose_frame = self._d60_pose_waist_frame(obs)
            if bool(pose_frame.get("ok", False)):
                print(
                    f"[D60-POSE-WAIST-FIRST] ACCEPT width={float(pose_frame.get('width_mm',0.0)):.1f}mm "
                    f"conf={pose_frame.get('conf')} centerRebuilt={bool(pose_frame.get('center_rebuilt',False))}"
                )
                return pose_frame
            print(f"[D60-POSE-WAIST-FIRST] REJECT reason={pose_frame.get('reason','-')} -> mask recovery")
            mask_u8 = np.asarray(obs.mask.mask_u8, dtype=np.uint8)
            recover = getattr(self.d56, "_d56v62_mask_waist_prior", None)
            if callable(recover):
                try:
                    info = recover(obs, self.H_raw, mask_u8.shape, self.d56_args)
                except Exception as exc:
                    info = {"available": False, "reason": repr(exc)}
                if isinstance(info, dict) and bool(info.get("available", False)):
                    try:
                        center = np.asarray(info["center_board"], dtype=np.float32).reshape(2)
                        waist_u = np.asarray(info["waist_axis_board"], dtype=np.float32).reshape(2)
                        body_u = np.asarray(info["body_axis_board"], dtype=np.float32).reshape(2)
                        width = float(info["waist_width_mm"])
                        waist_u = waist_u / max(float(np.linalg.norm(waist_u)), 1e-6)
                        body_u = body_u - waist_u * float(np.dot(body_u, waist_u))
                        body_u = body_u / max(float(np.linalg.norm(body_u)), 1e-6)
                        e0 = center - waist_u * (0.5 * width)
                        e1 = center + waist_u * (0.5 * width)
                        report.update({
                            "ok": True,
                            "source": "OUTER_MASK_D56_62",
                            "center": center,
                            "waist_u": waist_u,
                            "body_u": body_u,
                            "width_mm": width,
                            "endpoints": [e0, e1],
                            "recovery": _json_safe(info),
                            "pose_reject": _json_safe(pose_frame),
                            "reason": "OK",
                        })
                        return report
                    except Exception:
                        pass
            report.update({"reason": "WAIST_FRAME_UNAVAILABLE", "pose_reject": _json_safe(pose_frame)})
            return report

        def _d60_waist_endpoint_gate(self, obs: Any, plan: Any) -> Tuple[bool, Dict[str, Any]]:
            report: Dict[str, Any] = {"ok": False}
            if plan is None or not bool(getattr(plan, "ok", False)):
                report["reason"] = "PLAN_NOT_OK"
                return False, report
            frame = self._d60_waist_frame(obs)
            if not bool(frame.get("ok", False)):
                report.update({"reason": str(frame.get("reason", "WAIST_FRAME_UNAVAILABLE")), "waist_frame": _json_safe(frame)})
                return False, report
            endpoints_raw = frame.get("endpoints", [])
            if len(endpoints_raw) != 2:
                report["reason"] = "WAIST_ENDPOINTS_UNAVAILABLE"
                return False, report
            endpoints_list = [np.asarray(q, dtype=np.float32).reshape(2) for q in endpoints_raw]
            waist_u = np.asarray(frame["waist_u"], dtype=np.float32).reshape(2)
            body_u = np.asarray(frame["body_u"], dtype=np.float32).reshape(2)
            arm_points = dict(getattr(plan, "arm_points", {}) or {})
            grips: Dict[str, np.ndarray] = {}
            for arm_key in ("arm2", "arm1"):
                item = arm_points.get(arm_key)
                if not isinstance(item, dict):
                    report["reason"] = f"{arm_key.upper()}_GRIP_UNAVAILABLE"
                    return False, report
                raw = item.get("grip_board", item.get("source_board"))
                try:
                    g = np.asarray(raw, dtype=np.float32).reshape(2)
                except Exception:
                    report["reason"] = f"{arm_key.upper()}_GRIP_INVALID"
                    return False, report
                if not np.all(np.isfinite(g)):
                    report["reason"] = f"{arm_key.upper()}_GRIP_INVALID"
                    return False, report
                grips[arm_key] = g
            direct_cost = float(np.linalg.norm(grips["arm2"] - endpoints_list[0]) + np.linalg.norm(grips["arm1"] - endpoints_list[1]))
            swap_cost = float(np.linalg.norm(grips["arm2"] - endpoints_list[1]) + np.linalg.norm(grips["arm1"] - endpoints_list[0]))
            endpoints = {"arm2": endpoints_list[0], "arm1": endpoints_list[1]} if direct_cost <= swap_cost else {"arm2": endpoints_list[1], "arm1": endpoints_list[0]}
            body_min = float(self.front.d60_waist_grip_body_min_mm)
            body_max = float(self.front.d60_waist_grip_body_max_mm)
            tangent_max = float(self.front.d60_waist_grip_tangent_max_mm)
            radius_max = float(self.front.d60_waist_grip_endpoint_radius_max_mm)
            arm_report: Dict[str, Any] = {}
            all_ok = True
            for arm_key in ("arm2", "arm1"):
                delta = grips[arm_key] - endpoints[arm_key]
                body_mm = float(np.dot(delta, body_u))
                tangent_mm = abs(float(np.dot(delta, waist_u)))
                radius_mm = float(np.linalg.norm(delta))
                arm_ok = bool(body_min <= body_mm <= body_max and tangent_mm <= tangent_max and radius_mm <= radius_max)
                arm_report[arm_key] = {
                    "grip_board": grips[arm_key].astype(float).tolist(),
                    "waist_endpoint_board": endpoints[arm_key].astype(float).tolist(),
                    "bodyward_below_mm": body_mm,
                    "waist_tangent_offset_mm": tangent_mm,
                    "endpoint_radius_mm": radius_mm,
                    "ok": arm_ok,
                }
                all_ok = all_ok and arm_ok
            metrics = dict(getattr(plan, "metrics", {}) or {})
            report.update({
                "ok": bool(all_ok),
                "reason": "OK" if all_ok else "GRIP_NOT_BELOW_WAIST_ENDPOINTS",
                "mode": str(metrics.get("d42_plan_mode", "")),
                "waist_source": str(metrics.get("waist_source", frame.get("source", ""))),
                "waist_frame_source": str(frame.get("source", "")),
                "waist_left": endpoints_list[0].astype(float).tolist(),
                "waist_right": endpoints_list[1].astype(float).tolist(),
                "waist_mid": np.asarray(frame["center"], dtype=np.float32).astype(float).tolist(),
                "body_u": body_u.astype(float).tolist(),
                "waist_u": waist_u.astype(float).tolist(),
                "assignment": "DIRECT" if direct_cost <= swap_cost else "SWAPPED",
                "arms": arm_report,
                "limits": {
                    "body_min_mm": body_min,
                    "body_max_mm": body_max,
                    "tangent_max_mm": tangent_max,
                    "endpoint_radius_max_mm": radius_max,
                },
            })
            return bool(all_ok), report

        def _d60_force_waist_endpoint_plan(self, obs: Any, source_plan: Any) -> Tuple[Any, Any, Dict[str, Any]]:
            result: Dict[str, Any] = {"ok": False, "reason": "NOT_EVALUATED"}
            if obs is None or getattr(obs, "mask", None) is None:
                result["reason"] = "OUTER_MASK_UNAVAILABLE"
                return None, None, result
            frame = self._d60_waist_frame(obs)
            if not bool(frame.get("ok", False)):
                result.update({"reason": str(frame.get("reason", "WAIST_FRAME_UNAVAILABLE")), "waist_frame": _json_safe(frame)})
                return None, None, result
            mask_u8 = np.asarray(obs.mask.mask_u8, dtype=np.uint8)
            ys, xs = np.where(mask_u8 > 0)
            if len(xs) < 100:
                result["reason"] = "OUTER_MASK_TOO_SMALL"
                return None, None, result
            max_samples = 42000
            if len(xs) > max_samples:
                step = max(1, int(math.ceil(len(xs) / float(max_samples))))
                xs = xs[::step]
                ys = ys[::step]
            pts_px = np.column_stack([xs, ys]).astype(np.float32)
            try:
                pts_board = self.cv2.perspectiveTransform(pts_px.reshape(-1, 1, 2), self.H_raw).reshape(-1, 2)
            except Exception as exc:
                result["reason"] = f"MASK_BOARD_TRANSFORM_FAILED:{exc!r}"
                return None, None, result
            dist_map = self.cv2.distanceTransform((mask_u8 > 0).astype(np.uint8), self.cv2.DIST_L2, 5)
            finite = np.all(np.isfinite(pts_board), axis=1)
            cfg = self.cfg56
            board_ok = finite & (pts_board[:, 0] >= float(cfg.board_x_min) + 6.0) & (pts_board[:, 0] <= float(cfg.board_x_max) - 6.0) & (pts_board[:, 1] >= float(cfg.board_y_min) + 6.0) & (pts_board[:, 1] <= float(cfg.board_y_max) - 6.0)
            inside_px = dist_map[ys.astype(np.int32), xs.astype(np.int32)].astype(np.float32)
            center = np.asarray(frame["center"], dtype=np.float32).reshape(2)
            waist_u = np.asarray(frame["waist_u"], dtype=np.float32).reshape(2)
            body_u = np.asarray(frame["body_u"], dtype=np.float32).reshape(2)
            endpoints = [np.asarray(q, dtype=np.float32).reshape(2) for q in frame["endpoints"]]
            body_min = float(self.front.d60_waist_grip_body_min_mm)
            body_max = float(self.front.d60_waist_grip_body_max_mm)
            tangent_max = float(self.front.d60_waist_grip_tangent_max_mm)
            radius_max = float(self.front.d60_waist_grip_endpoint_radius_max_mm)
            left_max = float(cfg.split_board_x - cfg.center_dead_half_width)
            right_min = float(cfg.split_board_x + cfg.center_dead_half_width)
            local_fn = getattr(self.d56, "_d13_local_mask_ratio", None)

            def candidates_for(arm_key: str, endpoint: np.ndarray) -> List[Dict[str, Any]]:
                rel = pts_board - endpoint.reshape(1, 2)
                body = rel @ body_u
                tangent = np.abs(rel @ waist_u)
                radius = np.linalg.norm(rel, axis=1)
                own = pts_board[:, 0] <= left_max if arm_key == "arm2" else pts_board[:, 0] >= right_min
                valid = board_ok & own & (body >= body_min) & (body <= body_max) & (tangent <= tangent_max) & (radius <= radius_max) & (inside_px >= 1.0)
                ids = np.flatnonzero(valid)
                if len(ids) == 0:
                    return []
                target_body = float(np.clip(55.0, body_min + 4.0, body_max - 4.0))
                pre = 1.8 * np.abs(body[ids] - target_body) + 1.15 * tangent[ids] + 0.12 * np.abs(radius[ids] - target_body) - 3.2 * inside_px[ids]
                order = ids[np.argsort(pre)[: min(100, len(ids))]]
                rows: List[Dict[str, Any]] = []
                for idx in order:
                    q = np.asarray(pts_board[int(idx)], dtype=np.float32).reshape(2)
                    local = float(local_fn(mask_u8, self.H_raw, q, 18)) if callable(local_fn) else 1.0
                    if local < 0.18 and float(inside_px[int(idx)]) < 2.5:
                        continue
                    score = float(1.8 * abs(float(body[int(idx)]) - target_body) + 1.15 * float(tangent[int(idx)]) + 0.12 * abs(float(radius[int(idx)]) - target_body) - 3.2 * float(inside_px[int(idx)]) - 42.0 * local)
                    rows.append({
                        "board": q,
                        "score": score,
                        "body_mm": float(body[int(idx)]),
                        "tangent_mm": float(tangent[int(idx)]),
                        "radius_mm": float(radius[int(idx)]),
                        "inside_px": float(inside_px[int(idx)]),
                        "local": local,
                    })
                rows.sort(key=lambda x: x["score"])
                return rows[:18]

            assignment_reports: List[Dict[str, Any]] = []
            pair_trials: List[Tuple[float, int, Dict[str, Any], Dict[str, Any]]] = []
            for assignment_index, pair in enumerate(((endpoints[0], endpoints[1]), (endpoints[1], endpoints[0]))):
                a2 = candidates_for("arm2", pair[0])
                a1 = candidates_for("arm1", pair[1])
                assignment_reports.append({"assignment": assignment_index, "arm2_candidates": len(a2), "arm1_candidates": len(a1)})
                for c2 in a2:
                    for c1 in a1:
                        sep = float(np.linalg.norm(c1["board"] - c2["board"]))
                        if not 82.0 <= sep <= 425.0:
                            continue
                        pair_score = float(c2["score"] + c1["score"] + 0.08 * abs(sep - max(120.0, min(330.0, float(frame.get("width_mm", sep))))))
                        pair_trials.append((pair_score, assignment_index, c2, c1))
            pair_trials.sort(key=lambda row: row[0])
            failures: List[str] = []
            for pair_score, assignment_index, c2, c1 in pair_trials[:120]:
                if source_plan is not None:
                    try:
                        forced = copy.deepcopy(source_plan)
                    except Exception:
                        forced = None
                else:
                    forced = None
                if forced is None or not hasattr(forced, "arm_points"):
                    cls = getattr(self.d56, "D31DualGraspPlan", None)
                    if cls is None:
                        result["reason"] = "D31_PLAN_CLASS_UNAVAILABLE"
                        return None, None, result
                    forced = cls(True, "BOTTOM_VLA_V19_POSE_OR_MASK_WAIST_ENDPOINT_PAIR")
                forced.ok = True
                forced.reason = "BOTTOM_VLA_V19 actual outer-mask pair just below authoritative waist endpoints"
                forced.arm_points = {}
                for arm_key, cand in (("arm2", c2), ("arm1", c1)):
                    q = np.asarray(cand["board"], dtype=np.float32).reshape(2)
                    px = self.d56.board_to_pixel(self.H_raw, float(q[0]), float(q[1])) if callable(getattr(self.d56, "board_to_pixel", None)) else None
                    forced.arm_points[arm_key] = {
                        "role": "waist_endpoint_outer_mask_forced",
                        "grip_board": q.copy(),
                        "source_board": q.copy(),
                        "target_board": q.copy(),
                        "grip_px": None if px is None else [float(px[0]), float(px[1])],
                        "local_mask_ratio": float(cand["local"]),
                        "safe_local_mask_ratio": float(cand["local"]),
                        "mask_core_depth_px": float(cand["inside_px"]),
                        "effective_inset_mm": float(cand["body_mm"]),
                        "approx_total_clearance_mm": float(cand["body_mm"]),
                        "d56v35_grasp_circle_radius_mm": 8.0,
                        "bottom_vla_v19_body_below_endpoint_mm": float(cand["body_mm"]),
                        "bottom_vla_v19_tangent_from_endpoint_mm": float(cand["tangent_mm"]),
                    }
                metrics = copy.deepcopy(dict(getattr(forced, "metrics", {}) or {}))
                eps = [np.asarray(q, dtype=np.float32).reshape(2) for q in endpoints]
                ep_px = [self.d56.board_to_pixel(self.H_raw, float(q[0]), float(q[1])) for q in eps]
                center_px = self.d56.board_to_pixel(self.H_raw, float(center[0]), float(center[1]))
                metrics.update({
                    "d42_plan_mode": "D56_15_WAIST_RIBBON",
                    "execution_mode": "D32_WAIST_LIFT",
                    "waist_source": "BOTTOM_VLA_V19_POSE_WAIST_DIRECT" if str(frame.get("source", "")) == "POSE_WAIST_LOCAL_GATE" else "BOTTOM_VLA_V19_MASK_WAIST_FALLBACK",
                    "waist_width_mm": float(frame.get("width_mm", 0.0)),
                    "waist_angle_deg": float(math.degrees(math.atan2(float(waist_u[1]), float(waist_u[0])))),
                    "waist_left_px": None if ep_px[0] is None else [float(ep_px[0][0]), float(ep_px[0][1])],
                    "waist_center_px": None if center_px is None else [float(center_px[0]), float(center_px[1])],
                    "waist_right_px": None if ep_px[1] is None else [float(ep_px[1][0]), float(ep_px[1][1])],
                    "bottom_vla_v19_forced_endpoint_pair": True,
                    "bottom_vla_v19_waist_frame_source": str(frame.get("source", "")),
                    "bottom_vla_v19_pair_score": float(pair_score),
                    "bottom_vla_v19_assignment": int(assignment_index),
                })
                forced.metrics = metrics
                gate_ok, gate_report = self._d60_waist_endpoint_gate(obs, forced)
                if not gate_ok:
                    failures.append(f"gate:{gate_report.get('reason','FAIL')}")
                    continue
                motion, why = self.d56._63_step120_motion_plan_from_d56_plan(forced, self.config, self.cfg56, self.d56_args)
                if motion is None:
                    failures.append(str(why))
                    continue
                result.update({
                    "ok": True,
                    "reason": "FORCED_OUTER_MASK_WAIST_ENDPOINT_PAIR_OK",
                    "waist_frame_source": str(frame.get("source", "")),
                    "assignment": int(assignment_index),
                    "pair_score": float(pair_score),
                    "arm2": _json_safe(c2),
                    "arm1": _json_safe(c1),
                    "gate": _json_safe(gate_report),
                    "assignment_reports": assignment_reports,
                    "pair_trials": len(pair_trials),
                })
                print(
                    f"[D60-WAIST-FORCE] OK frame={frame.get('source','-')} pairTrials={len(pair_trials)} "
                    f"A2=({c2['board'][0]:.1f},{c2['board'][1]:.1f}) below={c2['body_mm']:.1f}mm tan={c2['tangent_mm']:.1f}mm "
                    f"A1=({c1['board'][0]:.1f},{c1['board'][1]:.1f}) below={c1['body_mm']:.1f}mm tan={c1['tangent_mm']:.1f}mm"
                )
                return forced, motion, result
            result.update({
                "reason": "NO_KINEMATICALLY_SAFE_MASK_PAIR_BELOW_WAIST_ENDPOINTS",
                "waist_frame_source": str(frame.get("source", "")),
                "assignment_reports": assignment_reports,
                "pair_trials": len(pair_trials),
                "failures": failures[-20:],
            })
            print(f"[D60-WAIST-FORCE] FAILED frame={frame.get('source','-')} assignments={assignment_reports} pairTrials={len(pair_trials)} failures={failures[-5:]}")
            return None, None, result

        def _store_d60_runtime(self, payload: Dict[str, Any]) -> str:
            token = self._new_runtime_token()
            with self.d60_runtime_lock:
                self.d60_runtime = {token: payload}
            return token

        def _draw_d60_frozen_overlay(self, bundle: Any, obs: Any, d60_plan: Any, motion60: Any) -> np.ndarray:
            canvas = bundle.raw.copy()
            draw_base = getattr(self.d56, "draw_bottom_overlay_safe", None)
            if callable(draw_base):
                try:
                    canvas = draw_base(
                        canvas, self.H_raw, obs, self.cfg56, plan=None, wrinkle_plan=None,
                        args=self.d56_args, motion_busy=False, motion_name="WAIST_PULL_LAYDOWN"
                    )
                except Exception as exc:
                    print(f"[D60-OVERLAY-WARN] base={exc!r}")
            metrics = dict(getattr(d60_plan, "metrics", {}) or {}) if d60_plan is not None else {}
            mode = str(metrics.get("d42_plan_mode", ""))
            draw_safe = getattr(self.d56, "_d56v28_draw_safe_mask_overlay", None)
            if callable(draw_safe):
                try:
                    active = mode in ("D56_15_WAIST_RIBBON", "D45_V6_MASK_CURVE_WAIST", "D45_V6_MASK_CURVE_FAILED")
                    canvas = draw_safe(canvas, self.H_raw, obs, self.cfg56, self.d56_args, active=active)
                except Exception as exc:
                    print(f"[D60-OVERLAY-WARN] safe={exc!r}")
            waist_source = str(metrics.get("waist_source", ""))
            pose_direct = waist_source == "BOTTOM_VLA_V19_POSE_WAIST_DIRECT"
            if not pose_direct:
                draw_waist = getattr(self.d56, "_d56v7_draw_waist_observer", None)
                if callable(draw_waist):
                    try:
                        canvas = draw_waist(canvas, obs, self.d56_args)
                    except Exception as exc:
                        print(f"[D60-OVERLAY-WARN] waist={exc!r}")
            else:
                try:
                    frame = self._d60_pose_waist_frame(obs)
                    eps = [np.asarray(q, dtype=np.float32).reshape(2) for q in frame.get("endpoints", [])]
                    if len(eps) == 2:
                        p0 = self.d56.board_to_pixel(self.H_raw, float(eps[0][0]), float(eps[0][1]))
                        p1 = self.d56.board_to_pixel(self.H_raw, float(eps[1][0]), float(eps[1][1]))
                        pc = self.d56.board_to_pixel(self.H_raw, float(frame["center"][0]), float(frame["center"][1]))
                        if p0 is not None and p1 is not None:
                            self.cv2.line(canvas, (int(round(p0[0])), int(round(p0[1]))), (int(round(p1[0])), int(round(p1[1]))), (255, 0, 255), 4, self.cv2.LINE_AA)
                        if pc is not None:
                            qx, qy = int(round(pc[0])), int(round(pc[1]))
                            self.cv2.circle(canvas, (qx, qy), 8, (255, 0, 255), -1, self.cv2.LINE_AA)
                            self.cv2.putText(canvas, "POSE WAIST USED", (qx + 10, qy - 10), self.cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 0, 255), 2, self.cv2.LINE_AA)
                except Exception as exc:
                    print(f"[D60-OVERLAY-WARN] pose-waist={exc!r}")
            draw_plan = getattr(self.d56, "_d30_draw_overlay", None)
            if callable(draw_plan):
                try:
                    summary = "60-14 FROZEN WAIST PLAN" if d60_plan is not None and bool(getattr(d60_plan, "ok", False)) else _get_plan_reason(d60_plan, "NO SAFE WAIST PLAN")
                    canvas = draw_plan(canvas, self.H_raw, d60_plan, "WAIST_PLAN_LOCKED", summary)
                except Exception as exc:
                    print(f"[D60-OVERLAY-WARN] plan={exc!r}")
            return canvas

        def _prepare_waist_pull_laydown(self) -> None:
            with self.state_lock:
                if self.motion_busy:
                    print("[WAIST-PULL-LAYDOWN] blocked during motion")
                    return
                if self.recorder.review_pending():
                    print("[WAIST-PULL-LAYDOWN] blocked: complete previous G/B/K and Y/N")
                    return
            if not self.empty_baseline_ready:
                self.status = "WAIST_PULL_LAYDOWN BLOCKED: EMPTY BOARD E REQUIRED"
                return
            self._verify_sources_unchanged()
            if not self._ensure_camera_clear("WAIST_PULL_LAYDOWN_BEFORE_PLAN", allow_move=False):
                self.status = "WAIST_PULL_LAYDOWN BLOCKED: CAMERA-CLEAR NOT VERIFIED"
                return
            attempts = int(max(1, self.front.d60_prepare_attempts))
            bundle = None
            obs = None
            d60_plan = None
            motion60 = None
            reason = "WAIST_PAIR_UNAVAILABLE"
            plan_ok = False
            selected_attempt = 0
            attempt_reports: List[Dict[str, Any]] = []
            for attempt in range(1, attempts + 1):
                candidate_bundle = self._capture_i_frame_from_live("D56_WAIST_LIFT_LAYDOWN")
                candidate_obs = self._infer_for_action("D56_WAIST_LIFT_LAYDOWN", candidate_bundle.corrected)
                pose_frame = self._d60_pose_waist_frame(candidate_obs)
                candidate_plan = None
                candidate_motion = None
                candidate_diag: Dict[str, Any] = {}
                force_report = None
                endpoint_report: Dict[str, Any] = {"ok": False, "reason": "NOT_EVALUATED"}
                candidate_reason = "WAIST_PAIR_UNAVAILABLE"
                candidate_ok = False
                if bool(pose_frame.get("ok", False)):
                    print(
                        f"[D60-POSE-WAIST-AUTHORITATIVE] attempt={attempt}/{attempts} "
                        f"width={float(pose_frame.get('width_mm', 0.0)):.1f}mm; bypass D56 waist-ribbon selection"
                    )
                    forced_plan, forced_motion, force_report = self._d60_force_waist_endpoint_plan(candidate_obs, None)
                    candidate_diag = {
                        "source_pipeline": "POSE_WAIST_DIRECT_TO_OUTER_MASK_GRIPS",
                        "pose_waist": _json_safe(pose_frame),
                        "legacy_d56_waist_ribbon_bypassed": True,
                    }
                    if forced_plan is not None and forced_motion is not None:
                        candidate_plan = forced_plan
                        candidate_motion = forced_motion
                        endpoint_ok, endpoint_report = self._d60_waist_endpoint_gate(candidate_obs, candidate_plan)
                        candidate_ok = bool(endpoint_ok)
                        candidate_reason = "POSE_WAIST_DIRECT_ENDPOINT_PAIR_OK" if candidate_ok else "POSE_WAIST_DIRECT_ENDPOINT_GATE_FAILED"
                    else:
                        candidate_reason = str((force_report or {}).get("reason", "POSE_WAIST_DIRECT_PAIR_FAILED"))
                else:
                    print(f"[D60-POSE-WAIST-AUTHORITATIVE] attempt={attempt}/{attempts} unavailable reason={pose_frame.get('reason','-')} -> legacy/mask fallback")
                    candidate_plan, candidate_diag = self._d60_plan_from_obs(candidate_obs)
                    endpoint_ok, endpoint_report = self._d60_waist_endpoint_gate(candidate_obs, candidate_plan)
                    if candidate_plan is not None and bool(getattr(candidate_plan, "ok", False)) and endpoint_ok:
                        candidate_motion, why = self.d56._63_step120_motion_plan_from_d56_plan(
                            candidate_plan, self.config, self.cfg56, self.d56_args
                        )
                        candidate_reason = str(why)
                        candidate_ok = candidate_motion is not None
                    else:
                        candidate_reason = _get_plan_reason(candidate_plan, "WAIST_PAIR_UNAVAILABLE")
                    if not candidate_ok and getattr(candidate_obs, "mask", None) is not None:
                        forced_plan, forced_motion, force_report = self._d60_force_waist_endpoint_plan(candidate_obs, candidate_plan)
                        if forced_plan is not None and forced_motion is not None:
                            candidate_plan = forced_plan
                            candidate_motion = forced_motion
                            endpoint_ok, endpoint_report = self._d60_waist_endpoint_gate(candidate_obs, candidate_plan)
                            candidate_reason = "MASK_WAIST_FALLBACK_ENDPOINT_PAIR_OK"
                            candidate_ok = bool(endpoint_ok)
                mode = str((getattr(candidate_plan, "metrics", {}) or {}).get("d42_plan_mode", "")) if candidate_plan is not None else ""
                arm_gate = endpoint_report.get("arms", {}) if isinstance(endpoint_report, dict) else {}
                a2_gate = arm_gate.get("arm2", {}) if isinstance(arm_gate, dict) else {}
                a1_gate = arm_gate.get("arm1", {}) if isinstance(arm_gate, dict) else {}
                print(
                    f"[D60-WAIST-END-GATE] attempt={attempt}/{attempts} ok={bool(endpoint_report.get('ok', False))} mode={mode or '-'} "
                    f"A2below={float(a2_gate.get('bodyward_below_mm', -999.0)):.1f}mm A2tan={float(a2_gate.get('waist_tangent_offset_mm', -999.0)):.1f}mm "
                    f"A1below={float(a1_gate.get('bodyward_below_mm', -999.0)):.1f}mm A1tan={float(a1_gate.get('waist_tangent_offset_mm', -999.0)):.1f}mm "
                    f"reason={endpoint_report.get('reason', '-') if isinstance(endpoint_report, dict) else '-'}"
                )
                attempt_entry = {
                    "attempt": attempt,
                    "planner": _json_safe(candidate_diag),
                    "pose_waist_authoritative": bool(pose_frame.get("ok", False)),
                    "pose_waist_frame": _json_safe(pose_frame),
                    "waist_endpoint_gate": _json_safe(endpoint_report),
                    "forced_endpoint_pair": _json_safe(force_report),
                    "source_plan_ok": bool(candidate_plan is not None and bool(getattr(candidate_plan, "ok", False))),
                    "motion_plan_ok": bool(candidate_ok),
                    "reason": str(candidate_reason),
                }
                attempt_reports.append(attempt_entry)
                bundle = candidate_bundle
                obs = candidate_obs
                d60_plan = candidate_plan
                motion60 = candidate_motion
                reason = str(candidate_reason)
                if candidate_ok:
                    plan_ok = True
                    selected_attempt = attempt
                    break
                if attempt < attempts:
                    print(f"[D60-FRESH-RETRY] attempt={attempt}/{attempts} rejected -> capture a fresh snapshot before freeze")
            if isinstance(motion60, dict) and plan_ok:
                motion60["bottom_vla_pullup_profile"] = {
                    "enabled": True,
                    "pre_pullup_vertical_lift_mm": 50.0,
                    "high_z_horizontal_first_mm": 0.0,
                }
            if bundle is None or obs is None:
                self.status = "WAIST_PULL_LAYDOWN NO PLAN: SNAPSHOT_UNAVAILABLE"
                return
            display_plan = d60_plan if plan_ok else None
            canvas = self._draw_d60_frozen_overlay(bundle, obs, display_plan, motion60)
            title = "WAIST_PULL_LAYDOWN FROZEN - 60-14" if plan_ok else f"WAIST_PULL_LAYDOWN NO PLAN: {reason}"
            self.cv2.putText(canvas, title[:110], (20, 45), self.cv2.FONT_HERSHEY_SIMPLEX, 0.68, (0, 255, 255) if plan_ok else (0, 0, 255), 2, self.cv2.LINE_AA)
            proxy = SimpleNamespace(
                ok=bool(plan_ok),
                reason=str(reason),
                action="WAIST_PULL_LAYDOWN",
                metrics={"source": "60-14", "motion60": _json_safe(motion60)},
                arm_points=_json_safe(getattr(d60_plan, "arm_points", {})) if plan_ok else {},
            )
            token = self._store_d60_runtime({"plan": d60_plan, "motion": motion60}) if plan_ok else ""
            diagnostics = {
                "d60_runtime_token": token,
                "source_motion": "60-15",
                "d60_direct_executor": True,
                "d60_pre_freeze_attempts": int(attempts),
                "d60_selected_attempt": int(selected_attempt),
                "d60_waist_endpoint_gate_required": True,
                "d60_planner_diagnostics": {"attempts": _json_safe(attempt_reports)},
                **self._semantic_metadata(),
            }
            locked = base.LockedPlan(
                None, "WAIST_PULL_LAYDOWN", bundle, obs, proxy, None, canvas,
                self.H_raw.copy(), time.time(), bool(plan_ok), str(reason),
                None if plan_ok else str(reason), diagnostics
            )
            obs_record = self.recorder.save_observation(locked, "BEFORE_ACTION", self._environment_metadata())
            locked.observation_record = obs_record
            self.recorder.save_decision(locked, obs_record)
            with self.state_lock:
                self.locked = locked
                self.display_image = canvas.copy()
                self.status = (
                    "WAIST_PULL_LAYDOWN FROZEN: ENTER RUNS 60-15 EXACT PLAN"
                    if plan_ok else f"WAIST_PULL_LAYDOWN NO PLAN: {reason}"
                )
            self._decorate_current_records()

        def _execute_waist_pull_laydown(self, locked: Any) -> None:
            if not self.empty_baseline_ready:
                print("[ENTER] WAIST_PULL_LAYDOWN blocked: empty-board baseline missing")
                return
            try:
                self._verify_sources_unchanged()
            except Exception as exc:
                print(f"[ENTER] WAIST_PULL_LAYDOWN blocked: {exc}")
                return
            age = time.time() - float(locked.created_at)
            if age > float(self.args.locked_plan_max_age_s):
                print(f"[ENTER] WAIST_PULL_LAYDOWN blocked: frozen plan age={age:.1f}s")
                self._invalidate_for_new_action("WAIST_PULL_LAYDOWN_PLAN_STALE")
                return
            token = str((locked.diagnostics or {}).get("d60_runtime_token", ""))
            with self.d60_runtime_lock:
                runtime = self.d60_runtime.get(token)
            if not isinstance(runtime, dict) or runtime.get("plan") is None or runtime.get("motion") is None:
                print("[ENTER] WAIST_PULL_LAYDOWN blocked: exact 60-15 frozen runtime missing")
                return
            try:
                self.recorder.open_transition(locked)
            except Exception as exc:
                print(f"[ENTER] WAIST_PULL_LAYDOWN transition blocked: {exc}")
                return
            with self.state_lock:
                self.motion_busy = True
                self.locked = None
                self.status = "WAIST_PULL_LAYDOWN EXECUTING EXACT 60-15 FROZEN PLAN"
            semantic = "WAIST_PULL_LAYDOWN"

            def worker() -> None:
                sent = False
                success = False
                detail = "NOT_STARTED"
                after_record = None

                def mark_sent() -> None:
                    nonlocal sent
                    sent = self.args.mode == "physical"

                try:
                    d60_arms = self._gentle_arms(self.arms, self.d56, 2.45, 1.90, "D60")
                    success, detail = self.d56._63_execute_step120_motion_from_d56_plan(
                        runtime["plan"], runtime["motion"], d60_arms, self.config,
                        self.cfg56, self.d56_args, on_verified_start=mark_sent
                    )
                    try:
                        if self._ensure_camera_clear("WAIST_PULL_LAYDOWN_AFTER", allow_move=False):
                            bundle_after = self._capture_i_frame_from_live("D56_WAIST_LIFT_LAYDOWN")
                            obs_after = self._infer_for_action("D56_WAIST_LIFT_LAYDOWN", bundle_after.corrected)
                            after_proxy = SimpleNamespace(ok=True, reason="AFTER_ACTION", action=semantic, metrics={}, arm_points={})
                            after_locked = base.LockedPlan(
                                None, semantic, bundle_after, obs_after, after_proxy, None,
                                bundle_after.raw.copy(), self.H_raw.copy(), time.time(), True,
                                "AFTER_ACTION", None, {**self._semantic_metadata(semantic)}
                            )
                            after_record = self.recorder.save_observation(
                                after_locked, "AFTER_ACTION", self._environment_metadata()
                            )
                    except Exception as exc:
                        print(f"[WAIST-PULL-LAYDOWN-AFTER-WARN] {exc!r}")
                    committed = bool(sent and self.args.mode == "physical")
                    if committed:
                        try:
                            self.recorder.note_motion_committed()
                        except Exception:
                            pass
                    self.recorder.complete_transition({
                        "execution_success": bool(success),
                        "execution_sent": bool(sent),
                        "garment_motion_committed": committed,
                        "execution_detail": str(detail),
                        "motion_parameters": {
                            "semantic_action": semantic,
                            "source": "60-14",
                            "frozen_plan": _compact_plan(locked.plan),
                            "motion60": _json_safe(runtime["motion"]),
                        },
                        "automatic_result": None,
                    }, after_record)
                    self.last_executed_semantic = semantic
                    self.pending_semantic = semantic
                    self._decorate_current_records(semantic)
                    self.status = "WAIST_PULL_LAYDOWN DONE: GOOD / BAD / SKIP"
                    print(f"[WAIST-PULL-LAYDOWN-DONE] success={success} detail={detail}")
                except Exception as exc:
                    print(f"[WAIST-PULL-LAYDOWN-ERROR] {exc!r}")
                    try:
                        self.recorder.complete_transition({
                            "execution_success": False,
                            "execution_sent": bool(sent),
                            "garment_motion_committed": bool(sent and self.args.mode == "physical"),
                            "execution_detail": repr(exc),
                            "motion_parameters": {"semantic_action": semantic, "source": "60-14"},
                            "automatic_result": None,
                        }, after_record)
                    except Exception:
                        pass
                    self.last_executed_semantic = semantic
                    self.pending_semantic = semantic
                    self.status = f"WAIST_PULL_LAYDOWN ERROR: {exc!r}"
                finally:
                    with self.state_lock:
                        self.motion_busy = False
                    with self.d60_runtime_lock:
                        self.d60_runtime.clear()

            self.worker = threading.Thread(target=worker, name="bottom-vla-waist-pull-laydown", daemon=True)
            self.worker.start()

        def _new_runtime_token(self) -> str:
            return f"rt_{time.time_ns()}"

        def _store_align_runtime(self, payload: Dict[str, Any]) -> str:
            token = self._new_runtime_token()
            with self.align_runtime_lock:
                self.align_runtime = {token: payload}
            return token

        def _align_waist_y_from_plan(self, d56_plan: Any) -> Optional[float]:
            if d56_plan is None or not bool(getattr(d56_plan, "ok", False)):
                return None
            metrics = dict(getattr(d56_plan, "metrics", {}) or {})
            center = metrics.get("waist_center_board")
            if isinstance(center, (list, tuple, np.ndarray)) and len(center) >= 2:
                try:
                    y = float(center[1])
                    if math.isfinite(y):
                        return y
                except Exception:
                    pass
            ys: List[float] = []
            for points in dict(getattr(d56_plan, "arm_points", {}) or {}).values():
                if not isinstance(points, dict):
                    continue
                grip = points.get("grip_board")
                if isinstance(grip, (list, tuple, np.ndarray)) and len(grip) >= 2:
                    try:
                        y = float(grip[1])
                        if math.isfinite(y):
                            ys.append(y)
                    except Exception:
                        pass
            return float(sum(ys) / len(ys)) if ys else None

        def _draw_align_white_reference(self, canvas: np.ndarray) -> np.ndarray:
            try:
                cal = self.align._c93_taught_trace_calibration(self.config, self.align_args)
                seam_x = float(cal["boundary_x_mm"])
                y0 = float(getattr(self.align_cfg, "board_y_min", getattr(self.cfg56, "board_y_min", -496.0)))
                y1 = float(getattr(self.align_cfg, "board_y_max", getattr(self.cfg56, "board_y_max", 5.0)))
                margin = max(8.0, float(getattr(self.align_cfg, "board_margin_mm", 18.0)))
                p0 = self.d56.board_to_pixel(self.H_raw, seam_x, y0 + margin)
                p1 = self.d56.board_to_pixel(self.H_raw, seam_x, y1 - margin)
                if p0 is not None and p1 is not None:
                    a = (int(round(float(p0[0]))), int(round(float(p0[1]))))
                    b = (int(round(float(p1[0]))), int(round(float(p1[1]))))
                    self.cv2.line(canvas, a, b, (0, 0, 0), 6, self.cv2.LINE_AA)
                    self.cv2.line(canvas, a, b, (255, 255, 255), 3, self.cv2.LINE_AA)
                    self.cv2.putText(
                        canvas, "ALIGN REF", (b[0] + 7, max(22, b[1] - 8)),
                        self.cv2.FONT_HERSHEY_SIMPLEX, 0.48,
                        (0, 0, 0), 4, self.cv2.LINE_AA
                    )
                    self.cv2.putText(
                        canvas, "ALIGN REF", (b[0] + 7, max(22, b[1] - 8)),
                        self.cv2.FONT_HERSHEY_SIMPLEX, 0.48,
                        (255, 255, 255), 2, self.cv2.LINE_AA
                    )
            except Exception as exc:
                print(f"[ALIGN-WHITE-REF-WARN] {exc!r}")
            return canvas

        def _prepare_align(self) -> None:
            with self.state_lock:
                if self.motion_busy:
                    print("[ALIGN] blocked during motion")
                    return
                if self.recorder.review_pending():
                    print("[ALIGN] blocked: enter GOOD/BAD/SKIP first")
                    return
            if not self.empty_baseline_ready:
                self.status = "ALIGN BLOCKED: EMPTY BOARD E REQUIRED"
                print("[ALIGN] press E on empty board first")
                return
            self._verify_sources_unchanged()
            if not self._ensure_camera_clear("ALIGN_BEFORE_PLAN", allow_move=False):
                self.status = "ALIGN BLOCKED: CAMERA-CLEAR NOT VERIFIED"
                return
            t0 = time.monotonic()
            bundle = self._capture_i_frame_from_live("D56_WAIST_LIFT_LAYDOWN")
            obs = self._infer_for_action("D56_WAIST_LIFT_LAYDOWN", bundle.corrected)
            infer_ms = (time.monotonic() - t0) * 1000.0
            plan_ok = False
            reason = ""
            runtime: Dict[str, Any] = {}
            proxy: Any = None
            decision: Dict[str, Any] = {}
            canvas = bundle.raw.copy()
            used_center_fallback = False
            dual_failure_reason = ""
            d56_plan = self._d60_plan_from_obs(obs)
            waist_y = self._align_waist_y_from_plan(d56_plan)
            target_y = float(self.front.align_waist_target_y_mm)
            min_pull = max(0.0, float(self.front.align_dual_waist_min_pull_mm))
            required_pull = None if waist_y is None else float(target_y - waist_y)
            choose_dual = bool(required_pull is not None and required_pull >= min_pull)
            current_state = {
                "waist_y_mm": waist_y,
                "waist_target_y_mm": target_y,
                "required_upward_pull_mm": required_pull,
                "dual_trigger_mm": min_pull,
                "selected_submode": "DUAL_WAIST_TOP" if choose_dual else "CENTER_VECTOR",
                "d60_plan_ok": bool(d56_plan is not None and bool(getattr(d56_plan, "ok", False))),
            }
            self.align_phase = str(current_state["selected_submode"])
            if waist_y is None:
                print("[ALIGN-CURRENT-STATE] waistY unavailable -> CENTER_VECTOR; D60 waist plan is not used as stale phase authority")
            else:
                print(
                    f"[ALIGN-CURRENT-STATE] waistY={waist_y:.1f} target={target_y:.1f} "
                    f"needUp={required_pull:.1f}mm trigger={min_pull:.1f} -> {self.align_phase}"
                )

            def build_center_vector() -> Tuple[Any, Dict[str, Any], float]:
                seam_x = float(self.align._c93_taught_trace_calibration(self.config, self.align_args)["boundary_x_mm"])
                tp = time.monotonic()
                action_plan, center_decision = self.align._align7_build_auto_plan(
                    obs, self.H_raw, None, seam_x, self.align_correction_count,
                    self.config, self.align_cfg, self.align_args
                )
                if not isinstance(center_decision, dict):
                    center_decision = {"reason": str(center_decision)}
                return action_plan, center_decision, (time.monotonic() - tp) * 1000.0

            if choose_dual:
                motion = None
                why = ""
                if d56_plan is not None and bool(getattr(d56_plan, "ok", False)):
                    motion, why = self.align._align3_build_dual_waist_top_locked(
                        d56_plan, self.config, self.align_cfg, self.align_args
                    )
                else:
                    why = _get_plan_reason(d56_plan, "D60 waist pair unavailable")
                if motion is not None:
                    plan_ok = True
                    reason = str(why or "ALIGN_DUAL_WAIST_TOP_READY")
                    runtime = {"kind": "DUAL_WAIST_TOP", "d56_plan": d56_plan, "motion": motion, "current_state": copy.deepcopy(current_state)}
                    proxy = SimpleNamespace(
                        ok=True,
                        reason=reason,
                        action="ALIGN_DUAL_WAIST_TOP",
                        metrics={"phase": "DUAL_WAIST_TOP", "motion": _json_safe(motion), "current_state": _json_safe(current_state)},
                        arm_points=_json_safe(getattr(d56_plan, "arm_points", {})),
                    )
                    try:
                        canvas = self._draw_arm_plan(canvas, d56_plan, self.d56, self.cfg56, "D56_WAIST_LIFT_LAYDOWN", self.H_raw)
                    except Exception:
                        pass
                    try:
                        canvas = self.align._align3_draw_overlay(canvas, self.H_raw, obs, self.config, self.align_args)
                    except Exception:
                        pass
                else:
                    dual_failure_reason = str(why or "D60 waist pair unavailable")
                    used_center_fallback = True
                    print(f"[ALIGN-CURRENT-STATE-FALLBACK] low waist but dual plan unavailable: {dual_failure_reason} -> CENTER_VECTOR")
            if not plan_ok:
                action_plan, decision, plan_ms = build_center_vector()
                self.align_last_decision = copy.deepcopy(decision)
                angle_abs = None
                try:
                    angle_abs = abs(float(decision.get("center_angle_error_deg")))
                except Exception:
                    try:
                        angle_abs = abs(float(action_plan.get("angle_error_deg"))) if action_plan is not None else None
                    except Exception:
                        angle_abs = None
                if (
                    self._prepare_origin == "AUTO"
                    and self.align_correction_count > 0
                    and angle_abs is not None
                    and angle_abs <= float(self.front.align_finish_angle_deg)
                ):
                    self.semantic_selected = "FINISH"
                    self.pending_semantic = "FINISH"
                    self.selected_action = "FINISH"
                    self._human_selected = "FINISH"
                    self._prepare_finish_from_existing(bundle, obs, angle_abs)
                    print(f"[AUTO-FINISH] center-vector angle={angle_abs:.1f}deg <= {float(self.front.align_finish_angle_deg):.1f}deg")
                    return
                if action_plan is not None:
                    plan_ok = True
                    reason = str(decision.get("decision") or decision.get("reason") or "ALIGN_READY")
                    runtime = {"kind": "ARM2_CORRECTION", "action_plan": copy.deepcopy(action_plan), "decision": copy.deepcopy(decision), "current_state": copy.deepcopy(current_state)}
                    proxy = SimpleNamespace(
                        ok=True,
                        reason=reason,
                        action="ALIGN_ARM2_CORRECTION",
                        metrics={
                            "phase": "ARM2_ALIGN_FALLBACK" if used_center_fallback else "ARM2_ALIGN",
                            "dual_waist_failure": dual_failure_reason,
                            "action_plan": _json_safe(action_plan),
                            "decision": _json_safe(decision),
                            "current_state": _json_safe(current_state),
                        },
                        arm_points={},
                    )
                else:
                    center_reason = str(decision.get("reason", "CENTER_VECTOR_ALIGN_NO_PLAN"))
                    reason = f"center-vector unavailable ({center_reason})"
                    if used_center_fallback and dual_failure_reason:
                        reason = f"DUAL_WAIST_TOP unavailable ({dual_failure_reason}); {reason}"
                    proxy = SimpleNamespace(
                        ok=False, reason=reason, action="ALIGN",
                        metrics={"phase": "ARM2_ALIGN", "dual_waist_failure": dual_failure_reason, "decision": _json_safe(decision), "current_state": _json_safe(current_state)},
                        arm_points={}
                    )
                try:
                    canvas = self.align._align4_draw_midpoint_overlay(
                        canvas, self.H_raw, obs, None, "ARM2_ALIGN", self.config,
                        self.align_cfg, self.align_args, self.align_correction_count,
                        cached_action=action_plan, cached_decision=decision
                    )
                except Exception:
                    try:
                        canvas = self.align._align3_draw_overlay(canvas, self.H_raw, obs, self.config, self.align_args)
                    except Exception:
                        pass
                print(f"[ALIGN-CENTER-VECTOR] infer={infer_ms:.0f}ms plan={plan_ms:.0f}ms ready={action_plan is not None} fallback={used_center_fallback} reason={reason}")
            if proxy is None:
                proxy = SimpleNamespace(ok=False, reason=reason, action="ALIGN", metrics={"current_state": _json_safe(current_state)}, arm_points={})
            canvas = self._draw_align_white_reference(canvas)
            token = self._store_align_runtime(runtime) if plan_ok else ""
            diagnostics = {
                "align_runtime_token": token,
                "align_phase": self.align_phase,
                "align_correction_count": self.align_correction_count,
                "align_decision": _json_safe(decision),
                "align_current_state": _json_safe(current_state),
                "align_center_vector_fallback": bool(used_center_fallback),
                "align_dual_waist_failure": dual_failure_reason,
                "inference_ms": infer_ms,
                **self._semantic_metadata(),
            }
            locked = base.LockedPlan(
                None, "ALIGN", bundle, obs, proxy, None, canvas, self.H_raw.copy(), time.time(),
                bool(plan_ok), str(reason), None if plan_ok else str(reason), diagnostics,
            )
            obs_record = self.recorder.save_observation(locked, "BEFORE_ACTION", self._environment_metadata())
            locked.observation_record = obs_record
            self.recorder.save_decision(locked, obs_record)
            with self.state_lock:
                self.locked = locked
                self.display_image = canvas.copy()
                self.status = "ALIGN FROZEN: ENTER EXECUTES EXACT DISPLAYED PLAN" if plan_ok else f"ALIGN NO PLAN: {reason}"
            self._decorate_current_records()

        def _prepare_finish_from_existing(self, bundle: Any, obs: Any, angle_abs: Optional[float] = None) -> None:
            canvas = bundle.raw.copy()
            try:
                canvas = self.align._align3_draw_overlay(
                    canvas, self.H_raw, obs, self.config, self.align_args
                )
            except Exception:
                pass
            self.cv2.putText(
                canvas,
                "FINISH FROZEN - ENTER CONFIRMS NO MOTION",
                (20, 45),
                self.cv2.FONT_HERSHEY_SIMPLEX,
                0.72,
                (0, 255, 255),
                2,
                self.cv2.LINE_AA,
            )
            proxy = SimpleNamespace(
                ok=True,
                reason="FINISH_STATE_FROZEN",
                action="FINISH",
                metrics={"center_angle_error_abs_deg": angle_abs},
                arm_points={},
            )
            diagnostics = {"finish_no_motion": True, **self._semantic_metadata()}
            locked = base.LockedPlan(
                None, "FINISH", bundle, obs, proxy, None, canvas,
                self.H_raw.copy(), time.time(), True, "FINISH_STATE_FROZEN", None, diagnostics
            )
            obs_record = self.recorder.save_observation(locked, "FINISH_STATE", self._environment_metadata())
            locked.observation_record = obs_record
            with self.state_lock:
                self.locked = locked
                self.display_image = canvas.copy()
                self.status = "FINISH FROZEN: ENTER CONFIRMS EPISODE SUCCESS"
            self._decorate_current_records()

        def _prepare_finish(self) -> None:
            with self.state_lock:
                if self.motion_busy or self.recorder.review_pending():
                    print("[FINISH] blocked while motion/review is pending")
                    return
            if not self.empty_baseline_ready:
                print("[FINISH] press E on empty board before episode collection")
                return
            self._verify_sources_unchanged()
            if not self._ensure_camera_clear("FINISH_BEFORE_FREEZE", allow_move=False):
                self.status = "FINISH BLOCKED: CAMERA-CLEAR NOT VERIFIED"
                return
            bundle = self._capture_i_frame_from_live("D56_WAIST_LIFT_LAYDOWN")
            obs = self._infer_for_action("D56_WAIST_LIFT_LAYDOWN", bundle.corrected)
            self._prepare_finish_from_existing(bundle, obs, None)

        def _finish_state_dir(self) -> Path:
            root = Path(self.args.dataset_root).expanduser().resolve() / "bottom" / "finish_states"
            root.mkdir(parents=True, exist_ok=True)
            return root

        def _open_next_episode(self) -> bool:
            old = self.recorder
            try:
                recorder_type = type(old)
                new_recorder = recorder_type(
                    str(old.root), old.cv2, copy.deepcopy(old.runtime_metadata),
                    tuple(old.board_bounds), int(old.board_size), bool(old.official_collection)
                )
                new_recorder.initialize()
            except Exception as exc:
                print(f"[VLA-EPISODE-NEW-ERROR] {exc!r}")
                return False
            self.recorder = new_recorder
            with self.align_runtime_lock:
                self.align_runtime.clear()
            with self.d60_runtime_lock:
                self.d60_runtime.clear()
            with self.state_lock:
                self.locked = None
                self.display_image = None
                self.semantic_selected = None
                self.pending_semantic = None
                self.last_executed_semantic = None
                self.last_result = None
                self.selected_action = None
                self.auto_recommended = "BASKET_GRASP"
                self.plan_origin = "HUMAN"
                self._prepare_origin = "HUMAN"
                self._human_selected = None
                self._auto_at_selection = "BASKET_GRASP"
                self.align_phase = "DUAL_WAIST_TOP"
                self.align_correction_count = 0
                self.align_last_decision = {}
                if hasattr(self, "d56_prespread_attempt_count"):
                    self.d56_prespread_attempt_count = 0
                self.status = "NEW EPISODE READY: SELECT NEXT ACTION"
            print(f"[VLA-EPISODE] NEXT OPEN {new_recorder.cycle_id}")
            return True

        def _execute_finish(self, locked: Any) -> None:
            stamp = time.strftime("%Y%m%d_%H%M%S") + f"_{int((time.time()%1)*1000):03d}"
            root = self._finish_state_dir()
            image_path = root / f"finish_{stamp}.jpg"
            json_path = root / f"finish_{stamp}.json"
            self.cv2.imwrite(str(image_path), locked.frame.raw)
            common_state = (dict(getattr(locked, "diagnostics", {}) or {}).get("common_state") or {}).get("bottom_state_v2")
            if not isinstance(common_state, dict):
                common_state = self._build_bottom_state_v2(locked)
            payload = {
                "schema": "bottom_vla_finish_v2",
                "build": BUILD,
                "cycle_id": getattr(self.recorder, "cycle_id", None),
                "action": "FINISH",
                "no_motion": True,
                "created_at": time.time(),
                "image": str(image_path),
                "observation": _json_safe(getattr(locked, "observation_record", None)),
                "common_state": {"bottom_state_v2": common_state},
                "classifier_sample": {
                    "action": "FINISH",
                    "state_schema": "bottom_state_v2",
                    "training_eligible": True,
                    "planner_diagnostics_are_input": False,
                },
                "plan": _compact_plan(locked.plan),
                **self._semantic_metadata("FINISH"),
            }
            _atomic_json(json_path, payload)
            self._append_finish_classifier_sample(locked, json_path)
            ok = self.recorder.finish_success()
            finished_cycle = getattr(self.recorder, "cycle_id", None)
            self.last_executed_semantic = "FINISH"
            print(f"[FINISH] saved {image_path}")
            if ok:
                if self._open_next_episode():
                    print(f"[FINISH] episode={finished_cycle} SUCCESS; next episode ready without restart")
                else:
                    with self.state_lock:
                        self.locked = None
                        self.display_image = locked.frame.raw.copy()
                        self.status = "EPISODE COMPLETE / SUCCESS; AUTO NEW EPISODE FAILED"
            else:
                with self.state_lock:
                    self.locked = None
                    self.display_image = locked.frame.raw.copy()
                    self.status = "FINISH SAVED"

        def _execute_align(self, locked: Any) -> None:
            if not self.empty_baseline_ready:
                print("[ENTER] ALIGN blocked: empty-board baseline missing")
                return
            try:
                self._verify_sources_unchanged()
            except Exception as exc:
                print(f"[ENTER] ALIGN blocked: {exc}")
                return
            age = time.time() - float(locked.created_at)
            if age > float(self.args.locked_plan_max_age_s):
                print(f"[ENTER] ALIGN blocked: frozen plan age={age:.1f}s")
                self._invalidate_for_new_action("ALIGN_PLAN_STALE")
                return
            token = str((locked.diagnostics or {}).get("align_runtime_token", ""))
            with self.align_runtime_lock:
                runtime = self.align_runtime.get(token)
            if not isinstance(runtime, dict):
                print("[ENTER] ALIGN blocked: exact frozen runtime plan missing")
                return
            try:
                self.recorder.open_transition(locked)
            except Exception as exc:
                print(f"[ENTER] ALIGN transition blocked: {exc}")
                return
            with self.state_lock:
                self.motion_busy = True
                self.locked = None
                self.status = "ALIGN EXECUTING EXACT FROZEN PLAN"
            semantic = "ALIGN"

            def worker() -> None:
                sent = False
                success = False
                detail = "NOT_STARTED"
                report: Dict[str, Any] = {}

                def mark_sent():
                    nonlocal sent
                    sent = self.args.mode == "physical"

                try:
                    kind = str(runtime.get("kind", ""))
                    align_arms = self._gentle_arms(
                        self.arms, self.align, self.align_grip_approach_min,
                        self.align_grip_release_min, "ALIGN"
                    )
                    if kind == "DUAL_WAIST_TOP":
                        success, detail = self.align._align3_execute_dual_waist_top_locked(
                            runtime["d56_plan"], runtime["motion"], align_arms,
                            self.align_cfg, self.align_args, on_verified_start=mark_sent
                        )
                        report = {"motion": _json_safe(runtime.get("motion")), "detail": str(detail)}
                        if success:
                            self.align_phase = "ARM2_ALIGN"
                            self.align_correction_count = 0
                    elif kind == "ARM2_CORRECTION":
                        success, detail, report = self.align._align7_execute_arm2_angle_pull_60style(
                            runtime["action_plan"], align_arms, self.config,
                            self.align_cfg, self.align_args, on_verified_start=mark_sent
                        )
                        if success:
                            self.align_correction_count += 1
                    else:
                        detail = "ALIGN_RUNTIME_KIND_INVALID"
                    after_record = None
                    try:
                        bundle_after = self._capture_i_frame_from_live("D56_WAIST_LIFT_LAYDOWN")
                        obs_after = self._infer_for_action("D56_WAIST_LIFT_LAYDOWN", bundle_after.corrected)
                        after_proxy = SimpleNamespace(ok=True, reason="AFTER_ACTION", action="ALIGN", metrics={}, arm_points={})
                        after_locked = base.LockedPlan(
                            None, "ALIGN", bundle_after, obs_after, after_proxy, None,
                            bundle_after.raw.copy(), self.H_raw.copy(), time.time(), True,
                            "AFTER_ACTION", None, {**self._semantic_metadata("ALIGN")}
                        )
                        after_record = self.recorder.save_observation(
                            after_locked, "AFTER_ACTION", self._environment_metadata()
                        )
                    except Exception as exc:
                        print(f"[ALIGN-AFTER-WARN] {exc!r}")
                    committed = bool(sent and success and self.args.mode == "physical")
                    if committed:
                        try:
                            self.recorder.note_motion_committed()
                        except Exception:
                            pass
                    self.recorder.complete_transition({
                        "execution_success": bool(success),
                        "execution_sent": bool(sent),
                        "garment_motion_committed": committed,
                        "execution_detail": str(detail),
                        "motion_parameters": {
                            "semantic_action": semantic,
                            "align_phase": runtime.get("kind"),
                            "frozen_plan": _compact_plan(locked.plan),
                            "execution_report": _json_safe(report),
                            "source": "align-11",
                        },
                        "automatic_result": None,
                    }, after_record)
                    self.last_executed_semantic = semantic
                    self.pending_semantic = semantic
                    self._decorate_current_records(semantic)
                    self.status = "ALIGN DONE: GOOD / BAD / SKIP"
                    print(f"[ALIGN-DONE] success={success} detail={detail}")
                except Exception as exc:
                    print(f"[ALIGN-ERROR] {exc!r}")
                    try:
                        self.recorder.complete_transition({
                            "execution_success": False,
                            "execution_sent": bool(sent),
                            "garment_motion_committed": bool(sent and self.args.mode == "physical"),
                            "execution_detail": repr(exc),
                            "motion_parameters": {"semantic_action": semantic, "source": "align-11"},
                            "automatic_result": None,
                        }, None)
                    except Exception:
                        pass
                    self.status = f"ALIGN ERROR: {exc!r}"
                finally:
                    with self.state_lock:
                        self.motion_busy = False
                    with self.align_runtime_lock:
                        self.align_runtime.clear()

            self.worker = threading.Thread(target=worker, name="bottom-vla-align-exec", daemon=True)
            self.worker.start()

        def _start_execution(self) -> None:
            with self.state_lock:
                locked = self.locked
            if locked is None or not bool(getattr(locked, "plan_ok", False)):
                print("[ENTER] blocked: no exact valid frozen plan")
                return
            if locked.action == "FINISH":
                self._execute_finish(locked)
                return
            if locked.action == "ALIGN":
                self._execute_align(locked)
                return
            if locked.action == "BASKET_GRASP":
                self._execute_basket(locked)
                return
            if locked.action == "WAIST_PULL_LAYDOWN":
                self._execute_waist_pull_laydown(locked)
                return
            self.last_executed_semantic = INTERNAL_TO_SEMANTIC.get(locked.action, self.pending_semantic)
            self.pending_semantic = self.last_executed_semantic
            super()._start_execution()

        def _mask_accurate(self) -> None:
            fields = {
                "human_mask_quality": "MASK_ACCURATE",
                "mask_quality_labeled_at": time.time(),
            }
            _patch_json_record(getattr(self.recorder, "current_observation", None), fields)
            _patch_json_record(getattr(self.recorder, "current_decision", None), fields)
            with self.state_lock:
                if self.locked is not None:
                    self.locked.diagnostics = dict(self.locked.diagnostics or {})
                    self.locked.diagnostics.update(fields)
            self.status = "MASK ACCURATE"
            print("[MASK] ACCURATE")

        def _current_review_semantic(self) -> Optional[str]:
            return self.last_executed_semantic or self.pending_semantic or self.semantic_selected

        def _finalize_basket_without_review(self, success: bool, report: Dict[str, Any]) -> None:
            record = getattr(self.recorder, "latest_transition", None)
            if not isinstance(record, dict):
                print("[BASKET-AUTO-FINALIZE] no transition")
                return
            payload = record.get("payload")
            if not isinstance(payload, dict):
                print("[BASKET-AUTO-FINALIZE] transition payload missing")
                return
            payload["human_result"] = "NOT_APPLICABLE"
            payload["human_result_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
            payload["result_label_required"] = False
            payload["basket_review_policy"] = "NO_HUMAN_REVIEW"
            payload["training_eligible"] = False
            payload["basket_auto_collection"] = True
            payload["basket_auto_collection_rule"] = "KEEP_ON_SUCCESS_DISCARD_ON_FAILURE"
            record["payload"] = payload
            self.recorder.latest_transition = record
            raw = record.get("fs_path") or record.get("path") or record.get("json_path")
            if raw:
                p = Path(str(raw))
                if p.is_dir():
                    p = p / "data.json"
                if p.is_file():
                    _atomic_json(p, payload)
            decided = False
            try:
                decided = bool(self.recorder.decide_collection(bool(success)))
            except Exception as exc:
                print(f"[BASKET-AUTO-FINALIZE-WARN] recorder decision failed: {exc!r}")
            record = getattr(self.recorder, "latest_transition", record)
            payload = record.get("payload") if isinstance(record, dict) else None
            if not isinstance(payload, dict):
                payload = {}
            payload.update({
                "human_result": "NOT_APPLICABLE",
                "result_label_required": False,
                "basket_review_policy": "NO_HUMAN_REVIEW",
                "basket_auto_collection": True,
                "basket_auto_collection_rule": "KEEP_ON_SUCCESS_DISCARD_ON_FAILURE",
                "collection_decision": "KEEP" if success else "DISCARD",
                "user_selected_for_collection": False,
                "training_eligible": False,
                "basket_execution": _json_safe(report),
                "grasp_success": bool(report.get("grasp_success")),
                "close_achieved_angle": report.get("close_achieved_angle"),
                "lift_achieved_z": report.get("lift_achieved_z"),
                "release_achieved_angle": report.get("release_achieved_angle"),
                "standby_reached": bool(report.get("standby_reached")),
                "status": "COLLECTED" if success else "DISCARDED",
            })
            if isinstance(record, dict):
                record["payload"] = payload
                self.recorder.latest_transition = record
                raw = record.get("fs_path") or record.get("path") or record.get("json_path")
                if raw:
                    p = Path(str(raw))
                    if p.is_dir():
                        p = p / "data.json"
                    if p.is_file():
                        _atomic_json(p, payload)
            if not decided:
                try:
                    self.recorder.refresh_dataset_stats()
                except Exception:
                    pass
            self.auto_recommended = "POSITION_ADJUST" if success else "BASKET_GRASP"
            self.last_result = None
            print(f"[BASKET-AUTO-FINALIZE] {'KEEP' if success else 'DISCARD'}; next action is immediately selectable")

        def _label_result_only(self, label: str) -> None:
            label = str(label).upper()
            if self.motion_busy:
                print("[RESULT] blocked during motion")
                return
            semantic = self._current_review_semantic()
            if semantic == "BASKET_GRASP":
                print("[RESULT] BASKET_GRASP has no manual review")
                self.status = "BASKET_GRASP COMPLETE: SELECT NEXT ACTION"
                return
            if not self.recorder.require_result():
                print("[RESULT] unavailable")
                return
            if not self.recorder.label_result(label):
                return
            self.last_result = label
            fields = {
                **self._semantic_metadata(semantic),
                "result_label": label,
                "collection_decision": None,
                "training_eligible_policy": {
                    "requires_good_or_bad": True,
                    "requires_execution_success": True,
                    "requires_motion_committed": True,
                    "requires_valid_before_after": True,
                    "requires_mask_not_inaccurate": True,
                    "requires_action_binding_match": True,
                },
            }
            _patch_json_record(getattr(self.recorder, "latest_transition", None), fields)
            self.status = f"{label} RECORDED: Y KEEP / N DISCARD"
            print(f"[RESULT] {label} recorded; choose Y/N")

        def _advance_after_collection(self, keep: bool, semantic: Optional[str]) -> None:
            semantic = str(semantic or self.auto_recommended)
            if keep and semantic == "BASKET_GRASP":
                self.auto_recommended = "OUTER_PULL"
            elif keep and self.last_result == "GOOD":
                next_map = {
                    "POSITION_ADJUST": "OUTER_PULL",
                    "OUTER_PULL": "PRESS_SWEEP",
                    "PRESS_SWEEP": "WAIST_PULL_LAYDOWN",
                    "WAIST_PULL_LAYDOWN": "ALIGN",
                    "ALIGN": "ALIGN",
                }
                self.auto_recommended = next_map.get(semantic, semantic)
            else:
                self.auto_recommended = semantic
            self.status = f"{'KEEP' if keep else 'DISCARD'} SAVED: NEXT AUTO={self.auto_recommended}"
            print(f"[COLLECT] {'KEEP' if keep else 'DISCARD'} auto_next={self.auto_recommended}")
            if bool(self.front.auto_prepare_next):
                self.semantic_selected = self.auto_recommended
                self.pending_semantic = self.auto_recommended
                self._human_selected = self.auto_recommended
                self._prepare_origin = "AUTO"
                self._auto_at_selection = self.auto_recommended
                self.selected_action = SEMANTIC_TO_INTERNAL[self.auto_recommended]
                self._invalidate_for_new_action("POST_COLLECTION_AUTO_NEXT")
                self._start_prepare_action()

        def _save_collection(self, keep: bool) -> None:
            if self.motion_busy:
                print("[COLLECT] blocked during motion")
                return
            semantic = self._current_review_semantic()
            if semantic == "BASKET_GRASP":
                print("[COLLECT] BASKET_GRASP is auto-finalized; Y/N is not used")
                self.status = "BASKET_GRASP COMPLETE: SELECT NEXT ACTION"
                return
            else:
                if self.recorder.require_result():
                    print("[COLLECT] choose GOOD/BAD/SKIP first")
                    self.status = "RESULT REQUIRED: G GOOD / B BAD / K SKIP"
                    return
                if not self.recorder.require_collection_decision():
                    print("[COLLECT] unavailable")
                    return
            if not self.recorder.decide_collection(bool(keep)):
                return
            fields = {
                **self._semantic_metadata(semantic),
                "collection_decision": "KEEP" if keep else "DISCARD",
                "basket_review_policy": "G_B_K_THEN_Y_N",
            }
            _patch_json_record(getattr(self.recorder, "latest_transition", None), fields)
            self._append_classifier_transition_sample(semantic, bool(keep))
            if semantic == "BASKET_GRASP":
                self.last_result = None
            self._advance_after_collection(bool(keep), semantic)

        def _handle(self, event: str) -> None:
            if event.startswith("ACTION:"):
                self._select_semantic(event.split(":", 1)[1], origin="HUMAN")
                return
            if event == "REJUDGE":
                self._rejudge()
                return
            if event == "ENTER":
                self._start_execution()
                return
            if event == "MASK_ACCURATE":
                self._mask_accurate()
                return
            if event == "MASK_INACCURATE":
                if self.motion_busy:
                    return
                if self.recorder.mark_mask_inaccurate():
                    self._invalidate_for_new_action("MASK_INACCURATE")
                    self.status = "MASK INACCURATE: FROZEN PLAN DISCARDED"
                return
            if event.startswith("RESULT:"):
                self._label_result_only(event.split(":", 1)[1])
                return
            if event == "COLLECT:KEEP":
                self._save_collection(True)
                return
            if event == "COLLECT:DISCARD":
                self._save_collection(False)
                return
            if event in {"I", "PLAN_INACCURATE"}:
                print(f"[CONTROL] {event} is not used in bottom_vla workflow")
                return
            if event == "FINISH_SUCCESS":
                self._select_semantic("FINISH", origin="HUMAN")
                return
            super()._handle(event)

        def _terminal_event(self) -> Optional[str]:
            try:
                ready, _, _ = select.select([sys.stdin], [], [], 0.0)
            except Exception:
                return None
            if not ready:
                return None
            line = sys.stdin.readline().strip().lower()
            mapping = {
                "1": "ACTION:BASKET_GRASP",
                "basket": "ACTION:BASKET_GRASP",
                "basket_grasp": "ACTION:BASKET_GRASP",
                "2": "ACTION:OUTER_PULL",
                "outer": "ACTION:OUTER_PULL",
                "outer_pull": "ACTION:OUTER_PULL",
                "3": "ACTION:PRESS_SWEEP",
                "press": "ACTION:PRESS_SWEEP",
                "press_sweep": "ACTION:PRESS_SWEEP",
                "4": "ACTION:WAIST_PULL_LAYDOWN",
                "8": "ACTION:POSITION_ADJUST",
                "position": "ACTION:POSITION_ADJUST",
                "position_adjust": "ACTION:POSITION_ADJUST",
                "waist": "ACTION:WAIST_PULL_LAYDOWN",
                "waist_pull_laydown": "ACTION:WAIST_PULL_LAYDOWN",
                "5": "ACTION:ALIGN",
                "align": "ACTION:ALIGN",
                "6": "ACTION:FINISH",
                "finish": "ACTION:FINISH",
                "7": "REJUDGE",
                "rejudge": "REJUDGE",
                "l": "LOCK_H",
                "lock": "LOCK_H",
                "e": "EMPTY_BASELINE",
                "empty": "EMPTY_BASELINE",
                "enter": "ENTER",
                "": "ENTER",
                "a": "MASK_ACCURATE",
                "accurate": "MASK_ACCURATE",
                "m": "MASK_INACCURATE",
                "mask": "MASK_INACCURATE",
                "g": "RESULT:GOOD",
                "good": "RESULT:GOOD",
                "b": "RESULT:BAD",
                "bad": "RESULT:BAD",
                "k": "RESULT:SKIP",
                "skip": "RESULT:SKIP",
                "y": "COLLECT:KEEP",
                "keep": "COLLECT:KEEP",
                "n": "COLLECT:DISCARD",
                "discard": "COLLECT:DISCARD",
                "abort": "ABORT_FAILED",
                "failed": "ABORT_FAILED",
                "q": "QUIT",
                "quit": "QUIT",
            }
            return mapping.get(line)

        def _draw_panel(self) -> np.ndarray:
            w, h = 1240, 410
            panel = np.full((h, w, 3), 28, np.uint8)
            cv2 = self.cv2
            cv2.putText(panel, "BOTTOM VLA", (24, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.90, (255, 255, 255), 2)
            rects: List[Tuple[str, Tuple[int, int, int, int], Tuple[int, int, int]]] = []

            def row(items, y, color):
                margin, gap = 22, 8
                width = int((w - 2 * margin - gap * (len(items) - 1)) / len(items))
                for i, (label, event) in enumerate(items):
                    x0 = margin + i * (width + gap)
                    rect = (x0, y, x0 + width, y + 54)
                    selected = event.startswith("ACTION:") and event.split(":", 1)[1] == self.semantic_selected
                    fill = tuple(min(255, int(c * 1.35)) for c in color) if selected else color
                    cv2.rectangle(panel, rect[:2], rect[2:], fill, -1)
                    cv2.rectangle(panel, rect[:2], rect[2:], (230, 230, 230), 1)
                    cv2.putText(panel, label, (x0 + 7, y + 34), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 2)
                    rects.append((event, rect, color))

            row([
                ("1 BASKET_GRASP", "ACTION:BASKET_GRASP"),
                ("2 OUTER_PULL", "ACTION:OUTER_PULL"),
                ("3 PRESS_SWEEP", "ACTION:PRESS_SWEEP"),
                ("4 WAIST_PULL_LAYDOWN", "ACTION:WAIST_PULL_LAYDOWN"),
            ], 68, (110, 180, 235))
            row([
                ("5 ALIGN", "ACTION:ALIGN"),
                ("6 FINISH", "ACTION:FINISH"),
                ("7 REJUDGE", "REJUDGE"),
                ("8 POSITION_ADJUST", "ACTION:POSITION_ADJUST"),
            ], 130, (120, 210, 150))
            row([
                ("ENTER EXECUTE", "ENTER"),
                ("A MASK_ACCURATE", "MASK_ACCURATE"),
                ("M MASK_INACCURATE", "MASK_INACCURATE"),
                ("E EMPTY", "EMPTY_BASELINE"),
                ("L LOCK_H", "LOCK_H"),
            ], 208, (210, 190, 100))
            row([
                ("G GOOD", "RESULT:GOOD"),
                ("B BAD", "RESULT:BAD"),
                ("K SKIP", "RESULT:SKIP"),
                ("ABORT FAILED", "ABORT_FAILED"),
            ], 270, (190, 140, 210))
            row([
                ("Y KEEP", "COLLECT:KEEP"),
                ("N DISCARD", "COLLECT:DISCARD"),
            ], 332, (135, 205, 205))
            self._panel_rects = rects
            return panel

        def _draw_dataset_counter(self, image: np.ndarray) -> np.ndarray:
            stats = self.recorder.dataset_stats_snapshot()
            out = image
            h, w = out.shape[:2]
            box_w, box_h = 420, 198
            x0 = max(8, w - box_w - 12)
            y0 = 12
            x1 = min(w - 8, x0 + box_w)
            y1 = min(h - 8, y0 + box_h)
            overlay = out.copy()
            self.cv2.rectangle(overlay, (x0, y0), (x1, y1), (20, 20, 20), -1)
            self.cv2.addWeighted(overlay, 0.72, out, 0.28, 0, out)
            keep_by_action = stats.get("by_action_keep", {}) or {}
            by_action = stats.get("by_action", {}) or {}
            lines = [
                f"DATASET KEEP: {int(stats.get('keep', 0))}   ELIGIBLE: {int(stats.get('eligible', 0))}",
            ]
            for action, short in (("D58_CIRC_POSITION", "POSITION_ADJ"), ("D54_OUTER_PULL", "OUTER_PULL"), ("D55_PRESS_SWEEP", "PRESS_SWEEP"), ("WAIST_PULL_LAYDOWN", "WAIST_PULL"), ("ALIGN", "ALIGN")):
                bucket = by_action.get(action, {}) or {}
                lines.append(
                    f"{short}: K {int(keep_by_action.get(action, 0))} / E {int(bucket.get('eligible', 0))} "
                    f"(G {int(bucket.get('good', 0))}/B {int(bucket.get('bad', 0))})"
                )
            lines.append(f"SKIP: {int(stats.get('skip', 0))}   DISCARD: {int(stats.get('discard', 0))}")
            for i, text in enumerate(lines):
                y = y0 + 25 + i * 24
                self.cv2.putText(out, text, (x0 + 10, y), self.cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 255), 1, self.cv2.LINE_AA)
            return out

        def _mouse(self, event, x, y, flags, param) -> None:
            del flags, param
            if event != self.cv2.EVENT_LBUTTONUP:
                return
            for name, rect, _ in self._panel_rects:
                x0, y0, x1, y1 = rect
                if x0 <= x <= x1 and y0 <= y <= y1:
                    self._queue(name)
                    return

        def run(self) -> int:
            window = "BOTTOM VLA"
            panel_name = "BOTTOM VLA CONTROLS"
            self.cv2.namedWindow(window, self.cv2.WINDOW_NORMAL)
            self.cv2.namedWindow(panel_name, self.cv2.WINDOW_NORMAL)
            self.cv2.resizeWindow(window, int(self.args.width), int(self.args.height))
            self.cv2.resizeWindow(panel_name, 1240, 410)
            self.cv2.setMouseCallback(panel_name, self._mouse)
            print("[BOTTOM-VLA CONTROLS]")
            print("1 BASKET_GRASP | 2 OUTER_PULL | 3 PRESS_SWEEP | 4 WAIST_PULL_LAYDOWN")
            print("5 ALIGN | 6 FINISH | 7 REJUDGE | 8 POSITION_ADJUST")
            print("ENTER execute | A/M mask | G/B/K result | Y/N save | E empty | L lock | Q quit")
            print("ACTION -> FROZEN PLAN -> ENTER -> REVIEW -> Y/N SAVE")
            print("BASKET_GRASP: auto finalize | OTHER ACTIONS: G/B/K -> Y/N")
            print("One preparation creates one frozen plan. ENTER never reinfers.")
            try:
                while not self.closed:
                    ok, raw = self._read_raw(flush=0)
                    if not ok or raw is None:
                        print("[CAM] read failed")
                        break
                    corrected = self._frame_for_action(raw, self.selected_action)
                    with self.live_frame_lock:
                        self.latest_live_raw = raw.copy()
                        self.latest_live_corrected = corrected.copy()
                        self.latest_live_monotonic = time.monotonic()
                    with self.state_lock:
                        shown = self.display_image.copy() if self.display_image is not None else raw.copy()
                        busy = self.motion_busy
                        infer_busy = self.inference_busy
                        status = str(self.status)
                    if busy:
                        self.cv2.putText(shown, "ROBOT MOTION RUNNING", (20, 40), self.cv2.FONT_HERSHEY_SIMPLEX, 0.78, (0, 0, 255), 3)
                    elif infer_busy:
                        self.cv2.putText(shown, "PREPARING ONE FROZEN PLAN", (20, 40), self.cv2.FONT_HERSHEY_SIMPLEX, 0.72, (0, 200, 255), 2)
                    self.cv2.putText(shown, status[:130], (20, shown.shape[0] - 24), self.cv2.FONT_HERSHEY_SIMPLEX, 0.53, (0, 255, 255), 2)
                    try:
                        shown = self._draw_dataset_counter(shown)
                    except Exception:
                        pass
                    self.cv2.imshow(window, shown)
                    self.cv2.imshow(panel_name, self._draw_panel())
                    key = self.cv2.waitKey(1) & 0xFF
                    keymap = {
                        ord("1"): "ACTION:BASKET_GRASP",
                        ord("2"): "ACTION:OUTER_PULL",
                        ord("3"): "ACTION:PRESS_SWEEP",
                        ord("4"): "ACTION:WAIST_PULL_LAYDOWN",
                        ord("5"): "ACTION:ALIGN",
                        ord("8"): "ACTION:POSITION_ADJUST",
                        ord("6"): "ACTION:FINISH",
                        ord("7"): "REJUDGE",
                        ord("e"): "EMPTY_BASELINE",
                        ord("l"): "LOCK_H",
                        ord("a"): "MASK_ACCURATE",
                        ord("m"): "MASK_INACCURATE",
                        ord("g"): "RESULT:GOOD",
                        ord("b"): "RESULT:BAD",
                        ord("k"): "RESULT:SKIP",
                        ord("y"): "COLLECT:KEEP",
                        ord("n"): "COLLECT:DISCARD",
                        10: "ENTER",
                        13: "ENTER",
                        ord("q"): "QUIT",
                        27: "QUIT",
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

    return BottomVLAApp


def main() -> int:
    front, remaining = _parse_front(sys.argv[1:])
    sources = _resolve_sources(front)
    base, align_mod, args = _build_runtime(front, remaining, sources)
    cls = _make_app_class(base, align_mod, front, sources)
    app = cls(args)
    return app.run()


if __name__ == "__main__":
    raise SystemExit(main())
