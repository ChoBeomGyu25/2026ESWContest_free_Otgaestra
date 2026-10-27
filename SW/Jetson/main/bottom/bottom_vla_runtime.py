import copy
import faulthandler
import hashlib
import importlib.util
import inspect
import json
import textwrap
import math
import os
import platform
import select
import sys
import threading
import time
import traceback
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np

BUILD = "2026-09-29-bottom-vla-v116-robust30-active-finish-model-owned-v22-integration-v6"


_RUNTIME_LOG_DIRNAME = "runtime_logs/bottom_vla"
_RUNTIME_LOG_LOCK = threading.RLock()


def _runtime_log_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _runtime_log_host_mirror(path):
    text = str(path)
    prefix = "/workspace/project_train/"
    if text.startswith(prefix):
        return "/home/deca/project_train/" + text[len(prefix):]
    return None


def _runtime_log_path():
    base = Path(__file__).resolve().parent
    override = str(os.environ.get("BOTTOM_VLA_LOG_DIR", "")).strip()
    log_dir = Path(override) if override else base / _RUNTIME_LOG_DIRNAME
    log_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    return log_dir / f"bottom_vla-112_{stamp}_pid{os.getpid()}.log"


def _runtime_log_reader(read_fd, echo_fd, log_fp):
    try:
        while True:
            try:
                data = os.read(read_fd, 65536)
            except OSError:
                break
            if not data:
                break
            try:
                os.write(echo_fd, data)
            except OSError:
                pass
            try:
                with _RUNTIME_LOG_LOCK:
                    log_fp.write(data)
                    log_fp.flush()
            except Exception:
                pass
    finally:
        try:
            os.close(read_fd)
        except OSError:
            pass


def _runtime_log_header(log_path):
    script = Path(__file__).resolve()
    try:
        script_hash = _runtime_log_sha256(script)
    except Exception as exc:
        script_hash = f"ERROR:{type(exc).__name__}:{exc}"
    print("=" * 110)
    print("[RUNTIME-LOG] BOTTOM VLA SESSION START")
    print(f"[RUNTIME-LOG] started_at={datetime.now().astimezone().isoformat(timespec='milliseconds')}")
    print(f"[RUNTIME-LOG] build={BUILD}")
    print(f"[RUNTIME-LOG] script={script}")
    print(f"[RUNTIME-LOG] script_sha256={script_hash}")
    print(f"[RUNTIME-LOG] log_file={log_path}")
    host_path = _runtime_log_host_mirror(log_path)
    if host_path:
        print(f"[RUNTIME-LOG] host_mirror={host_path}")
    print(f"[RUNTIME-LOG] argv={sys.argv!r}")
    print(f"[RUNTIME-LOG] cwd={os.getcwd()}")
    print(f"[RUNTIME-LOG] pid={os.getpid()} ppid={os.getppid()}")
    print(f"[RUNTIME-LOG] hostname={platform.node()}")
    print(f"[RUNTIME-LOG] python={sys.version.replace(chr(10), ' ')}")
    print(f"[RUNTIME-LOG] platform={platform.platform()}")
    print(f"[RUNTIME-LOG] DISPLAY={os.environ.get('DISPLAY', '')!r}")
    print(f"[RUNTIME-LOG] CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '')!r}")
    print("[RUNTIME-LOG] capture=FD1+FD2 (Python stdout/stderr + background-thread errors + native/TensorRT console output)")
    print("=" * 110)


def _run_with_runtime_log():
    log_path = _runtime_log_path()
    log_fp = open(log_path, "ab", buffering=0)
    original_stdout_fd = os.dup(1)
    original_stderr_fd = os.dup(2)
    stdout_read, stdout_write = os.pipe()
    stderr_read, stderr_write = os.pipe()
    stdout_thread = threading.Thread(
        target=_runtime_log_reader,
        args=(stdout_read, original_stdout_fd, log_fp),
        name="bottom-vla-runtime-log-stdout",
        daemon=True,
    )
    stderr_thread = threading.Thread(
        target=_runtime_log_reader,
        args=(stderr_read, original_stderr_fd, log_fp),
        name="bottom-vla-runtime-log-stderr",
        daemon=True,
    )
    exit_code = 1
    fatal_handler_enabled = False
    try:
        try:
            sys.stdout.flush()
            sys.stderr.flush()
        except Exception:
            pass
        stdout_thread.start()
        stderr_thread.start()
        os.dup2(stdout_write, 1)
        os.dup2(stderr_write, 2)
        os.close(stdout_write)
        os.close(stderr_write)
        try:
            if not faulthandler.is_enabled():
                faulthandler.enable(file=log_fp, all_threads=True)
                fatal_handler_enabled = True
        except Exception as exc:
            print(f"[RUNTIME-LOG-WARN] faulthandler enable failed: {exc!r}")
        _runtime_log_header(log_path)
        try:
            result = main()
            exit_code = int(result or 0)
        except KeyboardInterrupt:
            exit_code = 130
            print("[RUNTIME-LOG-FATAL] KeyboardInterrupt")
            traceback.print_exc()
        except BaseException as exc:
            exit_code = 1
            print(f"[RUNTIME-LOG-FATAL] unhandled={type(exc).__name__}: {exc!r}")
            traceback.print_exc()
        finally:
            print("-" * 110)
            print(f"[RUNTIME-LOG] finished_at={datetime.now().astimezone().isoformat(timespec='milliseconds')} exit_code={exit_code}")
            print(f"[RUNTIME-LOG] log_file={log_path}")
            print("[RUNTIME-LOG] BOTTOM VLA SESSION END")
            print("=" * 110)
            try:
                sys.stdout.flush()
                sys.stderr.flush()
            except Exception:
                pass
    finally:
        if fatal_handler_enabled:
            try:
                faulthandler.disable()
            except Exception:
                pass
        try:
            os.dup2(original_stdout_fd, 1)
        except OSError:
            pass
        try:
            os.dup2(original_stderr_fd, 2)
        except OSError:
            pass
        stdout_thread.join(timeout=2.0)
        stderr_thread.join(timeout=2.0)
        try:
            os.close(original_stdout_fd)
        except OSError:
            pass
        try:
            os.close(original_stderr_fd)
        except OSError:
            pass
        try:
            log_fp.flush()
        except Exception:
            pass
        try:
            log_fp.close()
        except Exception:
            pass
    return exit_code


def _load_v23():
    candidates = [
        Path(__file__).resolve().parent / "bottom_vla-23.py",
        Path("/workspace/project_train/aruco_test/dual/undistort/bottom_vla-23.py"),
    ]
    for path in candidates:
        if path.is_file():
            spec = importlib.util.spec_from_file_location("bottom_vla_v23_runtime", str(path))
            if spec is None or spec.loader is None:
                continue
            mod = importlib.util.module_from_spec(spec)
            sys.modules["bottom_vla_v23_runtime"] = mod
            spec.loader.exec_module(mod)
            return mod
    raise FileNotFoundError("bottom_vla-23.py not found beside bottom_vla-34.py or in the undistort directory")


v23 = _load_v23()
v23.BUILD = BUILD
v23.BOTTOM_CLASSIFIER_ACTIONS = ("POSITION_ADJUST", "PRESS_SWEEP", "WAIST_PULL_LAYDOWN", "ALIGN", "FINISH")


class _D60FeedbackProxy:
    def __init__(self, delegate, app, module, real_arm, arm_key):
        object.__setattr__(self, "_delegate", delegate)
        object.__setattr__(self, "_app", app)
        object.__setattr__(self, "_module", module)
        object.__setattr__(self, "_real_arm", real_arm)
        object.__setattr__(self, "_arm_key", str(arm_key).upper())

    def __getattr__(self, name):
        return getattr(self._delegate, name)

    def __setattr__(self, name, value):
        if name in {"_delegate", "_app", "_module", "_real_arm", "_arm_key"}:
            object.__setattr__(self, name, value)
        else:
            setattr(self._delegate, name, value)

    def _feedback_rule(self, angle, stage):
        text = str(stage or "").upper()
        value = float(angle)
        if "FINAL_DUAL_CLOSE" in text or value >= 3.20:
            return "FINAL_CLOSE", 2.80, None
        if "LIMITED_CLOSE" in text or value >= 2.95:
            return "CLOSE", 2.70, None
        return None

    def move_goal(self, x, y, z, t, spd, move_command=104, stage=None, caller=None, delay=0.18, log_command=True):
        stage_text = str(stage or "").upper()
        if stage_text == "STEP63_CAMERA_CLEAR_STANDBY":
            speed = float(getattr(self._app, "_v87_waist_standby_speed", 0.50))
            result = self._delegate.move_goal(
                x, y, z, t, speed, move_command=move_command, stage=stage, caller=caller, delay=delay, log_command=log_command
            )
            target = np.asarray([float(x), float(y), float(z)], np.float64)
            xyz_tol = float(getattr(self._app.args, "standby_xyz_tolerance_mm", 45.0))
            z_tol = float(getattr(self._app.args, "standby_z_tolerance_mm", 35.0))
            deadline = time.monotonic() + float(getattr(self._app, "_v87_waist_standby_timeout_s", 2.80))
            sample = 0
            while time.monotonic() < deadline:
                sample += 1
                try:
                    fb = self._module._d31_query_feedback(self._real_arm, timeout_s=0.65)
                    xyz = self._module._d31_feedback_xyz(fb)
                except Exception:
                    xyz = None
                if xyz is not None:
                    xyz = np.asarray(xyz, np.float64).reshape(3)
                    err = float(np.linalg.norm(xyz - target))
                    z_ok = bool(float(xyz[2]) >= float(target[2]) - z_tol)
                    if err <= xyz_tol and z_ok:
                        print(
                            f"[WAIST-V87-STANDBY] {self._arm_key} verified sample={sample} "
                            f"error={err:.1f}mm speed={speed:.2f}"
                        )
                        return result
                time.sleep(float(getattr(self._app, "_v87_waist_standby_poll_s", 0.08)))
            print(f"[WAIST-V87-STANDBY-WARN] {self._arm_key} adaptive verify timeout; post-action camera-clear check will decide")
            return result
        return self._delegate.move_goal(
            x, y, z, t, spd, move_command=move_command, stage=stage, caller=caller, delay=delay, log_command=log_command
        )

    def set_gripper(self, angle_rad, *args, **kwargs):
        rule = self._feedback_rule(angle_rad, kwargs.get("stage"))
        if rule is None:
            return self._delegate.set_gripper(angle_rad, *args, **kwargs)
        kind, _, _ = rule
        fast_kwargs = dict(kwargs)
        fast_kwargs["delay"] = min(float(fast_kwargs.get("delay", 0.25)), 0.03)
        result = self._delegate.set_gripper(angle_rad, *args, **fast_kwargs)
        print(f"[D60-V96-INSTANT-GRIP] {self._arm_key} {kind} cmd={float(angle_rad):.3f}; feedback retry wait skipped")
        return result


class _AlignV54LevelCarryProxy:
    def __init__(self, delegate, arm_key, level_xyz, coordinator, align_module):
        self._delegate = delegate
        self._arm_key = str(arm_key).lower()
        self._level_xyz = np.asarray(level_xyz, np.float64).reshape(3)
        self._coordinator = coordinator
        self._align = align_module
        self._used = False

    def __getattr__(self, name):
        return getattr(self._delegate, name)

    def _wait_level_arrival(self, tolerance_mm=20.0, timeout_s=7.0):
        query = getattr(self._align, "_d31_query_feedback", None)
        xyz_fn = getattr(self._align, "_d31_feedback_xyz", None)
        if not callable(query) or not callable(xyz_fn):
            time.sleep(0.70)
            return True
        deadline = time.monotonic() + float(timeout_s)
        while time.monotonic() < deadline:
            try:
                fb = query(self._delegate, timeout_s=0.65)
                xyz = xyz_fn(fb)
                if xyz is not None:
                    q = np.asarray(xyz, np.float64).reshape(3)
                    if float(np.linalg.norm(q - self._level_xyz)) <= float(tolerance_mm):
                        return True
            except Exception:
                pass
            time.sleep(0.10)
        return False

    def move_goal(
        self, x, y, z, t, spd, move_command=104, stage=None,
        caller=None, delay=0.18, log_command=True,
    ):
        stage_text = str(stage or "").upper()
        intercept = bool(
            (not self._used)
            and "WAIST_TOP_CARRY_CLOSED" in stage_text
            and bool(self._coordinator.get("enabled", False))
        )
        if not intercept:
            return self._delegate.move_goal(
                x, y, z, t, spd, move_command=move_command, stage=stage,
                caller=caller, delay=delay, log_command=log_command,
            )

        self._used = True
        lx, ly, lz = (float(v) for v in self._level_xyz)
        level_speed = float(self._coordinator.get("level_speed", spd))
        print(
            f"[ALIGN-V54-LEVEL-EXEC] {self._arm_key.upper()} level first "
            f"target=({lx:.1f},{ly:.1f},{lz:.1f}) speed={level_speed:.2f}"
        )
        self._delegate.move_goal(
            lx, ly, lz, t, level_speed, move_command=move_command,
            stage="ALIGN_V54_WAIST_LEVEL_CLOSED", caller="_AlignV54LevelCarryProxy",
            delay=delay, log_command=log_command,
        )
        arrived = self._wait_level_arrival(
            tolerance_mm=float(self._coordinator.get("level_tolerance_mm", 22.0)),
            timeout_s=float(self._coordinator.get("level_timeout_s", 12.0)),
        )
        if not arrived:
            retry_speed = max(0.30, level_speed * float(self._coordinator.get("level_retry_speed_scale", 0.75)))
            print(
                f"[ALIGN-V55-LEVEL-RETRY] {self._arm_key.upper()} first arrival timeout; "
                f"re-send SAME level target at speed={retry_speed:.2f} instead of aborting"
            )
            self._delegate.move_goal(
                lx, ly, lz, t, retry_speed, move_command=move_command,
                stage="ALIGN_V55_WAIST_LEVEL_RETRY_CLOSED", caller="_AlignV54LevelCarryProxy",
                delay=delay, log_command=log_command,
            )
            arrived = self._wait_level_arrival(
                tolerance_mm=float(self._coordinator.get("level_tolerance_mm", 22.0)) + 6.0,
                timeout_s=float(self._coordinator.get("level_retry_timeout_s", 8.0)),
            )
        if not arrived:
            raise RuntimeError(f"ALIGN_V55_{self._arm_key.upper()}_WAIST_LEVEL_RETRY_EXHAUSTED")
        try:
            self._coordinator["barrier"].wait(timeout=float(self._coordinator.get("barrier_timeout_s", 18.0)))
        except threading.BrokenBarrierError as exc:
            raise RuntimeError("ALIGN_V55_DUAL_LEVEL_BARRIER_FAILED") from exc
        time.sleep(float(self._coordinator.get("level_settle_s", 0.12)))
        print(f"[ALIGN-V54-LEVEL-PASS] {self._arm_key.upper()} level verified; now pull both sides upward")
        return self._delegate.move_goal(
            x, y, z, t, spd, move_command=move_command, stage=stage,
            caller=caller, delay=delay, log_command=log_command,
        )


class _V80PullupRecenterCoordinator:
    def __init__(self, app, arms):
        self.app = app
        self.arms = arms
        self.cond = threading.Condition()
        self.calls = {}
        self.ready = False
        self.computing = False
        self.dx_board = 0.0

    def _board_to_arm(self, arm_key, point):
        p = np.asarray(point, np.float64).reshape(2)
        rx, ry = self.app.align.board_to_arm_xy(self.app.config, arm_key, float(p[0]), float(p[1]))
        return np.asarray([float(rx), float(ry)], np.float64)

    def _arm_to_board(self, arm_key, xy):
        q = np.asarray(xy, np.float64).reshape(2)
        o = self._board_to_arm(arm_key, (0.0, 0.0))
        ex = self._board_to_arm(arm_key, (1.0, 0.0)) - o
        ey = self._board_to_arm(arm_key, (0.0, 1.0)) - o
        a = np.column_stack((ex, ey))
        if abs(float(np.linalg.det(a))) < 1e-9:
            raise RuntimeError(f"V80 {arm_key} board/arm affine is singular")
        return np.linalg.solve(a, q - o)

    def _feedback_xyz(self, arm_key):
        arm = self.arms[arm_key]
        query = getattr(self.app.align, '_d31_query_feedback', None)
        xyz_fn = getattr(self.app.align, '_d31_feedback_xyz', None)
        if not callable(query) or not callable(xyz_fn):
            return None
        fb = query(arm, timeout_s=0.85)
        xyz = xyz_fn(fb)
        if xyz is None:
            return None
        q = np.asarray(xyz, np.float64).reshape(3)
        return q if np.all(np.isfinite(q)) else None

    def _board_safe(self, arm_key, p):
        p = np.asarray(p, np.float64).reshape(2)
        try:
            if not self.app._v56_align_board_safe(p, 8.0):
                return False
            ok, _, _ = self.app._v56_align_arm_reach(arm_key, p, final_target=True)
            return bool(ok)
        except Exception:
            try:
                return bool(self.app.d56.point_in_board(self.app.cfg56, p, margin=8.0))
            except Exception:
                return True

    def _wait_xy(self, arm_key, target_xy, timeout_s=2.6, tol_mm=25.0):
        deadline = time.monotonic() + float(timeout_s)
        while time.monotonic() < deadline:
            xyz = self._feedback_xyz(arm_key)
            if xyz is not None and float(np.linalg.norm(xyz[:2] - np.asarray(target_xy, np.float64))) <= float(tol_mm):
                return True
            time.sleep(0.08)
        return False

    def _compute_and_recenter(self):
        current_xyz = {k: self._feedback_xyz(k) for k in ('arm2', 'arm1')}
        if any(current_xyz[k] is None for k in ('arm2', 'arm1')):
            print('[D60-V80-PRE-PULLUP-CENTER] feedback unavailable; skip pair-center translation')
            return 0.0
        current_board = {k: self._arm_to_board(k, current_xyz[k][:2]) for k in ('arm2', 'arm1')}
        pullup_board = {}
        for k in ('arm2', 'arm1'):
            call = self.calls[k]
            pullup_board[k] = self._arm_to_board(k, (call['x'], call['y']))
        midpoint = 0.5 * (current_board['arm2'] + current_board['arm1'])
        center_x = float(getattr(self.app.cfg56, 'split_board_x', 324.5))
        requested = float(np.clip(center_x - float(midpoint[0]), -float(self.app._v80_pullup_recenter_max_mm), float(self.app._v80_pullup_recenter_max_mm)))
        if abs(requested) < float(self.app._v80_pullup_recenter_deadband_mm):
            print(f'[D60-V80-PRE-PULLUP-CENTER] midpointX={midpoint[0]:.1f} boardCenterX={center_x:.1f} correction={requested:+.1f}mm inside deadband -> keep pair')
            return 0.0
        selected = 0.0
        for scale in (1.0, 0.75, 0.50, 0.25):
            dx = requested * scale
            ok = True
            for k in ('arm2', 'arm1'):
                if not self._board_safe(k, current_board[k] + np.asarray([dx, 0.0])):
                    ok = False
                    break
                if not self._board_safe(k, pullup_board[k] + np.asarray([dx, 0.0])):
                    ok = False
                    break
            if ok:
                selected = float(dx)
                break
        if abs(selected) < 1e-6:
            print(f'[D60-V80-PRE-PULLUP-CENTER] requested={requested:+.1f}mm but no common safe rigid translation; keep current pair')
            return 0.0
        targets = {}
        threads = []
        for k in ('arm2', 'arm1'):
            board_target = current_board[k] + np.asarray([selected, 0.0])
            arm_xy = self._board_to_arm(k, board_target)
            targets[k] = arm_xy
            call = self.calls[k]
            def runner(key=k, xy=arm_xy, z=float(current_xyz[k][2]), tool=float(call['t'])):
                self.arms[key].move_goal(
                    float(xy[0]), float(xy[1]), float(z), float(tool),
                    float(self.app._v80_pullup_recenter_speed),
                    move_command=104, stage='STEP80_PRE_PULLUP_PAIR_CENTER_CLOSED',
                    caller='_V80PullupRecenterCoordinator', delay=0.18, log_command=True,
                )
            th = threading.Thread(target=runner, daemon=True)
            threads.append(th)
            th.start()
        for th in threads:
            th.join(timeout=3.0)
        reached = {k: self._wait_xy(k, targets[k], tol_mm=float(self.app._v80_pullup_recenter_tol_mm)) for k in ('arm2', 'arm1')}
        print(
            f'[D60-V80-PRE-PULLUP-CENTER] actualPairMidX={midpoint[0]:.1f} -> targetCenterX={center_x:.1f} '
            f'rigidDx={selected:+.1f}mm reachA2/A1={reached["arm2"]}/{reached["arm1"]}; pair separation/orientation preserved'
        )
        return selected

    def submit_pullup(self, arm_key, call):
        leader = False
        with self.cond:
            self.calls[arm_key] = dict(call)
            if len(self.calls) == 2 and not self.ready and not self.computing:
                self.computing = True
                leader = True
            elif not self.ready:
                deadline = time.monotonic() + 4.0
                while not self.ready and time.monotonic() < deadline:
                    self.cond.wait(timeout=0.15)
                return float(self.dx_board)
        if leader:
            dx = 0.0
            try:
                dx = float(self._compute_and_recenter())
            except Exception as exc:
                print(f'[D60-V80-PRE-PULLUP-CENTER-WARN] {type(exc).__name__}: {exc}; use original pair')
            with self.cond:
                self.dx_board = float(dx)
                self.ready = True
                self.computing = False
                self.cond.notify_all()
            return float(dx)
        return 0.0

    def shifted_arm_xy(self, arm_key, x, y):
        dx = float(self.dx_board)
        if abs(dx) < 1e-9:
            return float(x), float(y)
        board = self._arm_to_board(arm_key, (x, y)) + np.asarray([dx, 0.0], np.float64)
        arm = self._board_to_arm(arm_key, board)
        return float(arm[0]), float(arm[1])


class _V80PullupArmProxy:
    def __init__(self, delegate, arm_key, coordinator):
        self._delegate = delegate
        self._arm_key = str(arm_key).lower()
        self._coord = coordinator

    def __getattr__(self, name):
        return getattr(self._delegate, name)

    def move_goal(self, x, y, z, t, spd, move_command=104, stage=None, caller=None, delay=0.18, log_command=True):
        stage_text = str(stage or '').upper()
        if stage_text == 'STEP67_FAST_PULLUP_CLOSED':
            self._coord.submit_pullup(self._arm_key, {'x': float(x), 'y': float(y), 'z': float(z), 't': float(t), 'spd': float(spd)})
            x, y = self._coord.shifted_arm_xy(self._arm_key, x, y)
        elif self._coord.ready and stage_text in {'STEP70_PRE_RELEASE_MICRO_LIFT_CLOSED', 'STEP63_POST_RELEASE_PURE_Z_CLEAR'}:
            x, y = self._coord.shifted_arm_xy(self._arm_key, x, y)
        return self._delegate.move_goal(
            x, y, z, t, spd, move_command=move_command, stage=stage,
            caller=caller, delay=delay, log_command=log_command,
        )



# ============================================================================
# V110 inference-only next-action classifier runtime
# ============================================================================
_V110_ACTIONS = (
    "POSITION_ADJUST",
    "PRESS_SWEEP",
    "WAIST_PULL_LAYDOWN",
    "ALIGN",
    "FINISH",
)

_V110_V6_EXTRA_NAMES = tuple(
    f"wrinkle_grid_r{r}_c{c}_density" for r in range(4) for c in range(4)
)


def _v110_softmax(logits):
    x = np.asarray(logits, np.float64).reshape(-1)
    if x.size != len(_V110_ACTIONS):
        raise RuntimeError(f"ACTION_MODEL_OUTPUT_DIM:{x.size}!={len(_V110_ACTIONS)}")
    x = x - np.max(x)
    e = np.exp(x)
    return (e / max(float(np.sum(e)), 1e-12)).astype(np.float32)


def _v110_sampled_nonzero_xy(mask, max_points=20000):
    yy, xx = np.nonzero(np.asarray(mask) > 0)
    n = len(xx)
    if n == 0:
        return np.empty((0, 2), dtype=np.float64)
    if n > int(max_points):
        step = int(math.ceil(n / float(max_points)))
        xx = xx[::step]
        yy = yy[::step]
    return np.column_stack([xx, yy]).astype(np.float64)


def _v110_project_xy(xy, H):
    pts = np.asarray(xy, dtype=np.float64).reshape(-1, 2)
    if len(pts) == 0:
        return np.empty((0, 2), dtype=np.float64)
    H = np.asarray(H, dtype=np.float64).reshape(3, 3)
    hp = np.c_[pts, np.ones(len(pts), dtype=np.float64)] @ H.T
    den = hp[:, 2]
    good = np.isfinite(den) & (np.abs(den) > 1e-9)
    out = np.full((len(pts), 2), np.nan, dtype=np.float64)
    out[good] = hp[good, :2] / den[good, None]
    good2 = good & np.all(np.isfinite(out), axis=1)
    return out[good2]


def _v110_body_frame_angle_deg(v2_names, v2_values):
    idx = {str(n): i for i, n in enumerate(v2_names)}
    for name in ("pose_valid", "body_axis_angle_norm", "mask_pca_axis_angle_norm"):
        if name not in idx:
            raise RuntimeError(f"V2_STATE_MISSING:{name}")
    pose_valid = float(v2_values[idx["pose_valid"]]) >= 0.5
    if pose_valid:
        angle = float(v2_values[idx["body_axis_angle_norm"]]) * 180.0
        if math.isfinite(angle):
            return angle, "body_axis"
    angle = float(v2_values[idx["mask_pca_axis_angle_norm"]]) * 90.0
    if not math.isfinite(angle):
        angle = -90.0
    ux = math.cos(math.radians(angle))
    uy = math.sin(math.radians(angle))
    if uy > 0:
        angle = ((angle + 180.0 + 180.0) % 360.0) - 180.0
    return angle, "mask_pca_fallback"


def _v110_build_v6_grid16(frame, mask_u8, H, v2_names, v2_values, cv2):
    """Exact runtime port of the frozen V6 4x4 builder used for training."""
    frame = np.asarray(frame)
    mask = (np.asarray(mask_u8) > 0).astype(np.uint8) * 255
    if mask.shape[:2] != frame.shape[:2]:
        mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)

    inner = cv2.erode(mask, np.ones((7, 7), np.uint8), iterations=1)
    inner_count = int(np.count_nonzero(inner))
    if inner_count < 100:
        raise RuntimeError("V6_INNER_MASK_TOO_SMALL")

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame.astype(np.uint8)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    lap = np.abs(cv2.Laplacian(blur, cv2.CV_32F, ksize=3))
    vals = lap[inner > 0]
    if vals.size < 100:
        raise RuntimeError("V6_NO_INNER_PIXELS")
    threshold = max(float(np.percentile(vals, 88.0)), float(np.mean(vals) + 0.65 * np.std(vals)))
    binary = ((lap >= threshold) & (inner > 0)).astype(np.uint8) * 255
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    min_area = max(8.0, 0.00018 * float(np.count_nonzero(mask)))
    kept = np.zeros_like(binary)
    components = []
    for contour in contours:
        area = float(cv2.contourArea(contour))
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
        board = _v110_project_xy(np.asarray([p0, p1]), H)
        length_board = float(np.linalg.norm(board[1] - board[0])) if len(board) == 2 else 0.0
        linearity = float(math.sqrt(major_var / minor_var))
        score = float(length_board * min(linearity, 12.0))
        components.append((score, contour))
        cv2.drawContours(kept, [contour], -1, 255, thickness=-1)

    if not components:
        return np.zeros(16, np.float32), {
            "valid": True, "reason": "NO_DOMINANT_COMPONENT", "component_count": 0,
            "threshold": float(threshold),
        }

    wrinkle_xy = _v110_sampled_nonzero_xy(kept, max_points=16000)
    garment_xy = _v110_sampled_nonzero_xy(inner, max_points=24000)
    wrinkle_board = _v110_project_xy(wrinkle_xy, H)
    garment_board = _v110_project_xy(garment_xy, H)
    if len(wrinkle_board) < 10:
        raise RuntimeError(f"V6_TOO_FEW_PROJECTED_WRINKLE_PIXELS:{len(wrinkle_board)}")
    if len(garment_board) < 50:
        raise RuntimeError(f"V6_TOO_FEW_PROJECTED_GARMENT_PIXELS:{len(garment_board)}")

    angle_deg, angle_source = _v110_body_frame_angle_deg(v2_names, v2_values)
    th = math.radians(angle_deg)
    body = np.asarray([math.cos(th), math.sin(th)], dtype=np.float64)
    lateral = np.asarray([-body[1], body[0]], dtype=np.float64)
    origin = np.mean(garment_board, axis=0)
    gd = garment_board - origin
    wd = wrinkle_board - origin
    gu = gd @ lateral
    gv = gd @ body
    wu = wd @ lateral
    wv = wd @ body
    u0, u1 = float(np.min(gu)), float(np.max(gu))
    v0, v1 = float(np.min(gv)), float(np.max(gv))
    du = max(u1 - u0, 1e-6)
    dv = max(v1 - v0, 1e-6)
    un = np.clip((wu - u0) / du, 0.0, 1.0)
    vn = np.clip((wv - v0) / dv, 0.0, 1.0)
    gun = np.clip((gu - u0) / du, 0.0, 1.0)
    gvn = np.clip((gv - v0) / dv, 0.0, 1.0)
    total_wrinkle_px = max(int(np.count_nonzero(kept)), 1)
    total_garment_px = max(int(inner_count), 1)

    def density(wmask, gmask):
        wfrac = float(np.count_nonzero(wmask)) / max(len(un), 1)
        gfrac = float(np.count_nonzero(gmask)) / max(len(gun), 1)
        west = wfrac * total_wrinkle_px
        gest = gfrac * total_garment_px
        if gest < 1.0:
            return 0.0
        return float(np.clip(west / gest, 0.0, 1.0))

    values = []
    for r in range(4):
        vlo, vhi = r / 4.0, (r + 1) / 4.0
        wvm = (vn >= vlo) & ((vn < vhi) if r < 3 else (vn <= vhi))
        gvm = (gvn >= vlo) & ((gvn < vhi) if r < 3 else (gvn <= vhi))
        for c in range(4):
            ulo, uhi = c / 4.0, (c + 1) / 4.0
            wum = (un >= ulo) & ((un < uhi) if c < 3 else (un <= uhi))
            gum = (gun >= ulo) & ((gun < uhi) if c < 3 else (gun <= uhi))
            values.append(density(wvm & wum, gvm & gum))

    return np.asarray(values, np.float32), {
        "valid": True,
        "reason": "OK",
        "component_count": int(len(components)),
        "threshold": float(threshold),
        "body_frame_angle_deg": float(angle_deg),
        "body_frame_angle_source": str(angle_source),
        "frame_wrinkle_pixels_sampled": int(len(wrinkle_xy)),
        "board_wrinkle_pixels_projected": int(len(wrinkle_board)),
        "board_garment_pixels_projected": int(len(garment_board)),
    }



class _V110InferenceRecorderProxy:
    """Legacy-recorder compatibility shim for physical inference-only runs.

    The robot executors still expect the recorder transaction API even when
    dataset collection is intentionally disabled.  Keep that API entirely in
    memory so open_transition()/complete_transition() can never block motion
    or touch the on-disk dataset.
    """
    _BLOCK_PREFIXES = (
        "save", "append", "write", "record", "collect", "commit",
        "finalize", "close_episode", "finish_episode", "abort_failed",
        "refresh_dataset", "mark_", "label_", "decide_", "fail_episode",
        "finish_success",
    )

    def __init__(self, delegate):
        object.__setattr__(self, "_delegate", delegate)
        object.__setattr__(self, "lock", threading.RLock())
        object.__setattr__(self, "current_observation", None)
        object.__setattr__(self, "current_decision", None)
        object.__setattr__(self, "latest_transition", None)
        object.__setattr__(self, "garment_motion_count", 0)
        object.__setattr__(self, "_virtual_counter", 0)
        object.__setattr__(self, "_warned", set())

    def __setattr__(self, name, value):
        local = {
            "_delegate", "lock", "current_observation", "current_decision",
            "latest_transition", "garment_motion_count", "_virtual_counter",
            "_warned",
        }
        if name in local:
            object.__setattr__(self, name, value)
            return
        delegate = object.__getattribute__(self, "_delegate")
        try:
            setattr(delegate, name, value)
        except Exception:
            object.__setattr__(self, name, value)

    def _warn_once(self, name):
        warned = object.__getattribute__(self, "_warned")
        if name not in warned:
            warned.add(name)
            print(f"[INFERENCE-ONLY-V110-NOSAVE] recorder.{name} suppressed")

    def _next_virtual_id(self, kind):
        n = int(object.__getattribute__(self, "_virtual_counter")) + 1
        object.__setattr__(self, "_virtual_counter", n)
        return f"inference_only_{kind}_{n:06d}"

    def review_pending(self):
        return False

    def require_result(self):
        return False

    def require_collection_decision(self):
        return False

    def save_observation(self, locked, phase, environment=None):
        self._warn_once("save_observation")
        rec = {
            "id": self._next_virtual_id("observation"),
            "relative_path": "",
            "fs_path": "",
            "payload": {
                "inference_only": True,
                "saved": False,
                "phase": str(phase),
                "human_mask_quality": "UNREVIEWED",
                "bottom_state_features": copy.deepcopy(getattr(locked, "diagnostics", {}) or {}),
            },
        }
        object.__setattr__(self, "current_observation", rec)
        return rec

    def save_decision(self, locked, obs_record):
        self._warn_once("save_decision")
        rec = {
            "id": self._next_virtual_id("decision"),
            "relative_path": "",
            "fs_path": "",
            "payload": {
                "inference_only": True,
                "saved": False,
                "observation_id": None if not isinstance(obs_record, dict) else obs_record.get("id"),
                "executed_action": str(getattr(locked, "action", "")),
            },
        }
        object.__setattr__(self, "current_decision", rec)
        return rec

    def open_transition(self, locked):
        self._warn_once("open_transition")
        rec = {
            "id": self._next_virtual_id("transition"),
            "relative_path": "",
            "fs_path": "",
            "payload": {
                "inference_only": True,
                "saved": False,
                "executed_action": str(getattr(locked, "action", "")),
                "execution_success": False,
                "execution_sent": False,
                "garment_motion_committed": False,
                "status": "VIRTUAL_OPEN",
            },
        }
        object.__setattr__(self, "latest_transition", rec)
        return rec

    def note_motion_committed(self):
        self._warn_once("note_motion_committed")
        n = int(object.__getattribute__(self, "garment_motion_count")) + 1
        object.__setattr__(self, "garment_motion_count", n)
        return n

    def complete_transition(self, execution, after=None):
        self._warn_once("complete_transition")
        rec = object.__getattribute__(self, "latest_transition")
        if isinstance(rec, dict):
            payload = rec.setdefault("payload", {})
            if isinstance(execution, dict):
                payload.update(copy.deepcopy(execution))
            if isinstance(after, dict):
                payload["after_observation_id"] = after.get("id")
            payload["status"] = "VIRTUAL_COMPLETE"
        return None

    def label_result(self, *args, **kwargs):
        self._warn_once("label_result")
        return False

    def decide_collection(self, *args, **kwargs):
        self._warn_once("decide_collection")
        return False

    def mark_plan_inaccurate(self, *args, **kwargs):
        self._warn_once("mark_plan_inaccurate")
        return False

    def mark_mask_inaccurate(self, *args, **kwargs):
        self._warn_once("mark_mask_inaccurate")
        return False

    def _noop(self, name, *args, **kwargs):
        self._warn_once(name)
        if str(name).startswith("refresh_dataset_stats"):
            return None
        return None

    def __getattr__(self, name):
        if name in {"review_pending", "require_result", "require_collection_decision"}:
            return object.__getattribute__(self, name)
        delegate = object.__getattribute__(self, "_delegate")
        attr = getattr(delegate, name)
        if not callable(attr):
            return attr
        low = str(name).lower()
        if low.startswith(self._BLOCK_PREFIXES):
            return lambda *a, **kw: self._noop(name, *a, **kw)
        return attr


class _V110ActionModel:
    def __init__(self, model_path, norm_path, input_dim, label):
        self.model_path = Path(str(model_path)).expanduser()
        self.norm_path = Path(str(norm_path)).expanduser()
        self.input_dim = int(input_dim)
        self.label = str(label)
        if not self.model_path.is_file():
            raise FileNotFoundError(f"{self.label} model not found: {self.model_path}")
        if not self.norm_path.is_file():
            raise FileNotFoundError(f"{self.label} normalization not found: {self.norm_path}")
        norm = np.load(str(self.norm_path))
        self.mean = np.asarray(norm["mean"], np.float32).reshape(-1)
        self.std = np.asarray(norm["std"], np.float32).reshape(-1)
        if self.mean.size != self.input_dim or self.std.size != self.input_dim:
            raise RuntimeError(
                f"{self.label} normalization dim mean/std={self.mean.size}/{self.std.size} expected={self.input_dim}"
            )
        self.std = np.where(np.abs(self.std) < 1e-6, 1.0, self.std).astype(np.float32)
        self.kind = self.model_path.suffix.lower()
        self.session = None
        self.input_name = None
        self.output_name = None
        self.trt = None
        self.torch = None
        self.engine = None
        self.context = None
        self.trt_input_name = None
        self.trt_output_name = None
        self._load_backend()

    def _load_backend(self):
        if self.kind == ".onnx":
            import onnxruntime as ort
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            available = set(ort.get_available_providers())
            providers = [p for p in providers if p in available] or ["CPUExecutionProvider"]
            self.session = ort.InferenceSession(str(self.model_path), providers=providers)
            self.input_name = self.session.get_inputs()[0].name
            self.output_name = self.session.get_outputs()[0].name
            print(f"[ACTION-MODEL-V110] {self.label} ONNX loaded providers={self.session.get_providers()}")
            return
        if self.kind == ".engine":
            import tensorrt as trt
            import torch
            if not torch.cuda.is_available():
                raise RuntimeError(f"{self.label} TensorRT requested but torch.cuda is unavailable")
            logger = trt.Logger(trt.Logger.WARNING)
            runtime = trt.Runtime(logger)
            with open(self.model_path, "rb") as f:
                engine = runtime.deserialize_cuda_engine(f.read())
            if engine is None:
                raise RuntimeError(f"{self.label} TensorRT engine deserialize failed")
            context = engine.create_execution_context()
            if context is None:
                raise RuntimeError(f"{self.label} TensorRT execution context failed")
            self.trt = trt
            self.torch = torch
            self.engine = engine
            self.context = context
            if hasattr(engine, "num_io_tensors"):
                names = [engine.get_tensor_name(i) for i in range(engine.num_io_tensors)]
                ins = [n for n in names if engine.get_tensor_mode(n) == trt.TensorIOMode.INPUT]
                outs = [n for n in names if engine.get_tensor_mode(n) == trt.TensorIOMode.OUTPUT]
                if len(ins) != 1 or len(outs) < 1:
                    raise RuntimeError(f"{self.label} unexpected TRT IO inputs={ins} outputs={outs}")
                self.trt_input_name, self.trt_output_name = ins[0], outs[0]
            print(f"[ACTION-MODEL-V110] {self.label} TensorRT loaded path={self.model_path}")
            return
        raise RuntimeError(f"Unsupported action-model extension: {self.model_path}")

    def predict(self, state):
        x = np.asarray(state, np.float32).reshape(-1)
        if x.size != self.input_dim:
            raise RuntimeError(f"{self.label} state dim={x.size} expected={self.input_dim}")
        if not np.all(np.isfinite(x)):
            raise RuntimeError(f"{self.label} state contains non-finite values")
        x = ((x - self.mean) / self.std).astype(np.float32).reshape(1, self.input_dim)
        if self.kind == ".onnx":
            logits = self.session.run([self.output_name], {self.input_name: x})[0]
        else:
            logits = self._predict_trt(x)
        probs = _v110_softmax(logits)
        order = np.argsort(probs)[::-1]
        i0, i1 = int(order[0]), int(order[1])
        return {
            "action": _V110_ACTIONS[i0],
            "confidence": float(probs[i0]),
            "margin": float(probs[i0] - probs[i1]),
            "probs": probs,
            "logits": np.asarray(logits, np.float32).reshape(-1),
        }

    def _predict_trt(self, x):
        torch = self.torch
        trt = self.trt
        stream = torch.cuda.current_stream()
        inp = torch.as_tensor(x, dtype=torch.float32, device="cuda").contiguous()
        if hasattr(self.engine, "num_io_tensors"):
            name_in = self.trt_input_name
            name_out = self.trt_output_name
            self.context.set_input_shape(name_in, tuple(inp.shape))
            out_shape = tuple(int(v) for v in self.context.get_tensor_shape(name_out))
            if any(v < 0 for v in out_shape):
                out_shape = (1, len(_V110_ACTIONS))
            out = torch.empty(out_shape, dtype=torch.float32, device="cuda")
            self.context.set_tensor_address(name_in, int(inp.data_ptr()))
            self.context.set_tensor_address(name_out, int(out.data_ptr()))
            ok = self.context.execute_async_v3(stream.cuda_stream)
            if ok is False:
                raise RuntimeError(f"{self.label} execute_async_v3 failed")
            stream.synchronize()
            return out.detach().cpu().numpy()

        n = int(self.engine.num_bindings)
        bindings = [0] * n
        output = None
        for i in range(n):
            if self.engine.binding_is_input(i):
                self.context.set_binding_shape(i, tuple(inp.shape))
                bindings[i] = int(inp.data_ptr())
            else:
                shape = tuple(int(v) for v in self.context.get_binding_shape(i))
                if any(v < 0 for v in shape):
                    shape = (1, len(_V110_ACTIONS))
                output = torch.empty(shape, dtype=torch.float32, device="cuda")
                bindings[i] = int(output.data_ptr())
        if output is None:
            raise RuntimeError(f"{self.label} TRT output binding missing")
        ok = self.context.execute_async_v2(bindings=bindings, stream_handle=stream.cuda_stream)
        if ok is False:
            raise RuntimeError(f"{self.label} execute_async_v2 failed")
        stream.synchronize()
        return output.detach().cpu().numpy()


def _make_app_class(base, align_mod, front, source_paths):
    front.d60_waist_grip_body_min_mm = 12.0
    front.d60_waist_grip_body_max_mm = 60.0
    front.d60_waist_grip_tangent_max_mm = 45.0
    front.d60_waist_grip_endpoint_radius_max_mm = 80.0
    front.d60_pose_waist_min_width_mm = 80.0
    front.d60_pose_waist_max_width_mm = 470.0
    front.d60_waist_hard_max_width_mm = 9999.0
    front.basket_placement_extra_deg = 20.0
    front.basket_placement_lower_mm = 140.0
    front.basket_standby_speed = 0.24
    front.align_waist_target_y_mm = -90.0

    Base = v23._make_app_class(base, align_mod, front, source_paths)

    class BottomVLAApp(Base):
        def _connect_arms_no_motion(self):
            cached = getattr(self, '_fix134c_early_arms', None)
            if isinstance(cached, dict) and cached:
                self.arms = cached
                print('[FIX134C-FAST-STARTUP] reusing early ARM serial session; duplicate OPEN suppressed')
                return cached
            arms = super()._connect_arms_no_motion()
            if isinstance(arms, dict) and arms:
                self._fix134c_early_arms = arms
            return arms

        def _fix134c_connect_and_send_standby_early(self):
            if not bool(getattr(self, '_integration_mode', False)):
                return False
            if bool(getattr(self, '_fix134c_early_standby_attempted', False)):
                return bool(getattr(self, '_fix134c_early_standby_sent', False))
            self._fix134c_early_standby_attempted = True
            arms = self._connect_arms_no_motion()
            self.arms = arms
            if not isinstance(arms, dict) or not arms:
                return False
            mover = getattr(self, '_v84_fast_move_arms_to_standby', None)
            if not callable(mover):
                print('[FIX134C-FAST-STARTUP-WARN] standby helper unavailable; legacy order retained')
                return False
            sent = bool(mover(
                self.d56,
                arms,
                self.cfg56,
                move_command=int(getattr(self.args, 'move_command', 104)),
                reason='BOTTOM_VLA_STARTUP_FIX134C_EARLY',
            ))
            self._fix134c_early_standby_sent = sent
            print(f'[FIX134C-FAST-STARTUP] serial connected -> immediate dual standby sent={sent}; model warmup overlaps return')
            return sent

        def _open_camera(self):
            self._fix134c_connect_and_send_standby_early()
            cap = getattr(self, "_integration_external_camera_cap", None)
            if cap is not None:
                print("[BOTTOM-INTEGRATION-CAMERA] using V22 shared RAW provider; local VideoCapture disabled")
                return cap
            return super()._open_camera()

        def __init__(self, args):
            self._integration_external_frame_provider = getattr(args, "_external_frame_provider", None)
            self._integration_external_camera_cap = getattr(args, "_external_camera_cap", None)
            self._integration_mode = bool(getattr(args, "_integration_mode", False))
            self._integration_stop_event = threading.Event()
            # V20 display-only telemetry: latched only after the real executor
            # accepts a locked action.  It never participates in planning.
            self._integration_executing_action = None
            self._integration_empty_board_baseline_id = None
            self._integration_frame_fallback_last_signature = None
            self._integration_frame_fallback_last_time = 0.0
            if self._integration_mode:
                if self._integration_external_frame_provider is None or self._integration_external_camera_cap is None:
                    raise RuntimeError("integration mode requires external_frame_provider and external_camera_cap")
                if hasattr(args, "camera_controls_enable"):
                    args.camera_controls_enable = False
                if hasattr(args, "camera_controls_stabilization_frames"):
                    args.camera_controls_stabilization_frames = 0
            self._auto_judge_lock = threading.RLock()
            self._v33_d60_rescue_lock = threading.RLock()
            self._auto_judge_busy = False
            self._auto_after_basket = False
            self._v51_post_basket_position_pending = False
            self._v30_source_warning_signature = None
            self._v30_last_trusted_mask_area_px = None
            self._v30_fallback_min_ratio = 0.45
            self._v52_basket_active_plan = None
            self._v52_basket_contact_feedback = None
            self._v52_basket_regrasp_count = 0
            self._v52_basket_regrasp_max = 2
            self._v52_basket_retention_exhausted = False
            self._v52_basket_open_step_rad = 0.20
            self._v52_basket_open_step_wait_s = 0.12
            self._v57_basket_open_relief_mm = 18.0
            self._v57_basket_emergency_raise_mm = 80.0
            self._v57_basket_failure_safe_raised = False
            self._v53_position_lateral_dominance = 1.40
            self._v53_position_lateral_min_mm = 35.0
            self._v53_waist_hard_inset_mm = 35.0
            self._v53_waist_inset_tolerance_mm = 1.0
            self._v53_waist_pose_edge_max_mm = 55.0
            self._v54_waist_corner_band_mm = 34.0
            self._v54_waist_corner_min_turn_deg = 10.0
            self._v54_waist_corner_max_waistward_shift_mm = 120.0
            self._v54_waist_corner_max_bodyward_shift_mm = 12.0
            self._v54_align_waist_recheck_mm = 18.0
            self._v54_align_guard_fraction = 0.42
            self._v54_align_level_y_tolerance_mm = 8.0
            self._v55_waist_outer_band_mm = 58.0
            self._v55_waist_outer_max_bodyward_shift_mm = 30.0
            self._v55_post_basket_lead_quantile = 0.90
            self._v56_press_reach_trigger_mm = 540.0
            self._v56_press_reach_target_mm = 510.0
            self._v56_press_horizontal_shift_max_mm = 120.0
            self._v56_press_horizontal_shift_step_mm = 10.0
            self._v56_rejected_plans = []
            self._v56_rejected_max = 6
            self._v56_reject_source_tol_mm = 28.0
            self._v56_reject_target_tol_mm = 28.0
            self._v56_reject_angle_tol_deg = 12.0
            self._v56_reject_move_tol_mm = 24.0
            self._v56_align_press_clearance_mm = 2.0
            self._v56_align_drag_speed = 0.55
            self._v56_align_center_contact_offsets_mm = (0.0, -8.0, 8.0, -16.0, 16.0, -24.0, 24.0)
            self._v59_align_center_min_action_mm = 40.0
            self._v59_align_center_t_values = (0.22, 0.30, 0.38, 0.46, 0.54, 0.62, 0.70, 0.78)
            self._v58_align_red_outer_inset_mm = 70.0
            self._v58_align_red_outer_row_band_mm = 8.0
            self._v58_align_red_outer_z_offset_mm = 4.0
            self._v59_align_red_outer_extra_drag_mm = 20.0
            self._v59_align_stage1_body_inset_mm = 40.0
            self._v59_align_stage1_press_clearance_mm = 2.0
            self._v59_refined_obs_cache_id = None
            self._v59_refined_obs_cache_value = None
            self._v53_align_capture_active = False
            self._v53_align_current_obs = None
            self._v53_align_initial_dual_done = False
            self._v60_arm1_standby_offset_mm = float(getattr(front, "arm1_standby_z_offset_mm", -20.0))
            self._v60_arm1_standby_offset_ids = set()
            self._v60_startup_camera_clear_handled = False
            self._v60_align_standby_attempts = 4
            self._v60_align_standby_poll_timeout_s = 3.0
            self._v61_waist_outer_pose_max_width_mm = 480.0
            self._v61_waist_outer_max_width_mm = 410.0
            self._v61_waist_outer_max_fit_rms_mm = 38.0
            self._v61_waist_outer_max_perp_error_deg = 30.0
            self._v61_press_crease_min_score = 0.18
            self._v61_press_shape_gain_floor = -0.005
            self._v61_press_irregular_long_reject_mm = 165.0
            self._v61_align_center_hard_inset_mm = 35.0
            self._v61_align_center_near_max_gap_mm = 45.0
            self._v61_align_center_contact_offsets_mm = (0.0,)
            self._v61_align_axis_support_min_ratio = 0.54
            self._v61_align_axis_support_required = 3
            self._v61_align_red_hard_inset_mm = 55.0
            self._v61_align_level_settle_s = 0.30
            self._v62_basket_close_probe_min_rad = 2.55
            self._v62_basket_probe_lift_mm = 12.0
            self._v62_basket_probe_stable_span_rad = 0.12
            self._v62_basket_regrasp_attempts = 2
            self._v62_press_cluster_radius_mm = 95.0
            self._v62_press_cluster_normal_margin_mm = 22.0
            self._v62_align_center_near_max_gap_mm = 45.0
            self._v62_align_center_small_gap_extra_mm = 50.0
            self._v62_align_center_small_gap_threshold_mm = 80.0
            self._v62_align_red_outer_inset_mm = 70.0
            self._v62_align_red_hard_inset_mm = 55.0
            self._v62_align_drag_speed = 0.55
            self._v63_waist_outer_max_angle_delta_deg = 18.0
            self._v63_waist_outer_max_center_delta_mm = 45.0
            self._v63_waist_required_edge_points = 3
            self._v63_align_center_t_values = (0.50, 0.44, 0.56, 0.38, 0.62, 0.32, 0.66)
            self._v63_align_center_preferred_t = 0.50
            self._v63_align_center_move_slack_mm = 20.0
            self._v63_align_center_arm2_shift_max_mm = 120.0
            self._v63_align_center_arm2_shift_step_mm = 10.0
            self._v63_align_center_shift_min_improve_mm = 5.0
            # V66 PRESS: reject only high-confidence inter-leg exterior gap boundaries.
            # Internal overlap/fold creases remain candidates because they do not satisfy
            # the full topology + contour + inter-leg-corridor gate.
            self._v66_press_gap_boundary_near_px = 10
            self._v66_press_gap_boundary_fraction_min = 0.55
            self._v66_press_gap_side_distances_mm = (10.0, 20.0, 35.0)
            self._v66_press_gap_garment_side_min = 0.62
            self._v66_press_gap_background_side_max = 0.38
            self._v66_press_gap_external_ratio_min = 0.60
            self._v66_press_gap_tangent_body_min = 0.65
            self._v66_press_gap_corridor_ratio = 0.32
            self._v66_press_gap_corridor_min_mm = 32.0
            self._v66_press_gap_corridor_max_mm = 105.0
            self._v66_press_gap_bg_cache_id = None
            self._v66_press_gap_bg_cache = None
            # V66 WAIST: 3/3 real-edge support is strong PASS; 2/3 can pass only
            # when the third point stays inside/near the real Mask and the independent
            # outer-edge reference agrees. Physical grasp inset remains the old hard gate.
            self._v66_waist_min_edge_points = 2
            self._v66_waist_conditional_max_edge_mm = 90.0
            # V66 ALIGN: pivot HOLD is fine-correction only. Large CENTER moves keep V64.
            self._v66_align_pivot_angle_min_deg = 5.0
            self._v66_align_pivot_move_max_mm = 80.0
            self._v66_align_pivot_preferred_t = 0.30
            self._v66_align_pivot_t_values = (0.24, 0.30, 0.36, 0.42)
            self._v66_align_pivot_lateral_offsets_mm = (55.0, 70.0, 85.0, 100.0, 115.0)
            self._v66_align_pivot_hard_inset_mm = 35.0
            self._v66_align_pivot_min_separation_mm = 100.0
            self._v67_waist_body_min_mm = 34.0
            self._v67_waist_body_max_mm = 50.0
            self._v67_waist_tangent_max_mm = 90.0
            self._v67_waist_endpoint_radius_max_mm = 125.0
            self._v67_contour_cache_id = None
            self._v67_contour_cache_value = None
            self._v67_mask_report_cache_id = None
            self._v67_mask_report_cache = {}
            self._v67_pose_waist_cache_id = None
            self._v67_pose_waist_cache_value = None
            self._v67_align_extra_move_mm = 0.0
            self._v67_align_center_z_extra_mm = -2.0
            self._v67_align_intersection_t_min = 0.12
            self._v67_align_intersection_t_max = 0.82
            self._v67_align_intersection_midgap_max_mm = 95.0
            self._v67_align_intersection_angle_min_deg = 5.0
            self._v67_align_intersection_angle_max_deg = 28.0
            self._v67_align_pivot_hard_inset_mm = 35.0
            self._v67_align_pivot_min_separation_mm = 100.0
            self._v67_align_arm1_move_min_mm = 20.0
            self._v67_align_arm1_move_max_mm = 155.0
            self._v70_align_center_z_extra_mm = -1.0
            self._v70_align_stage1_press_clearance_mm = 0.0
            self._v70_align_drag_speed = 0.48
            self._v70_align_center_move_slack_mm = 8.0
            self._v70_align_pivot_preferred_sep_mm = 210.0
            self._v70_align_pivot_move_reward = 0.30
            self._v70_press_source_normal_min_mm = 24.0
            self._v70_press_source_normal_max_mm = 115.0
            self._v70_press_source_normal_preferred_mm = 58.0
            self._v70_press_source_crotch_min_mm = 45.0
            self._v70_press_target_crotch_min_mm = 40.0
            self._v71_waist_search_tiers = (
                (34.0, 50.0, 90.0, 125.0),
                (42.0, 64.0, 125.0, 160.0),
                (52.0, 78.0, 165.0, 205.0),
            )
            self._v71_press_leg_t_min = -0.05
            self._v71_press_leg_t_max = 1.45
            self._v71_press_ridge_t_min = 0.02
            self._v71_press_leg_mask_ratio = 0.52
            self._v71_press_ridge_crotch_min_mm = 32.0
            self._v72_waist_preferred_inset_mm = 50.0
            self._v72_waist_fast_min_inset_mm = 40.0
            self._v72_waist_fast_max_inset_mm = 62.0
            self._v72_waist_fast_max_balance_mm = 18.0
            self._v72_waist_outer_max_bodyward_shift_mm = 45.0
            self._v72_press_right_prior_ratio = 0.10
            self._v72_press_inner_margin_prefer_mm = 24.0
            self._v73_waist_arm1_outward_mm = 30.0
            self._v73_waist_arm2_inward_mm = 20.0
            self._v73_align_arm1_z_extra_mm = -10.0
            self._v73_align_pivot_min_move_mm = 80.0
            self._v73_align_pivot_max_move_mm = 180.0
            self._v73_align_pivot_preferred_sep_mm = 280.0
            self._v73_align_pivot_move_reward = 0.50
            self._v73_align_anchor_shift_mm = 60.0
            self._v75_align_pivot_distance_gate_mm = 55.0
            self._v75_align_pivot_angle_min_deg = 7.0
            self._v75_align_pivot_pull_min_mm = 105.0
            self._v75_align_pivot_pull_max_mm = 150.0
            self._v75_align_pivot_gap_gain = 0.70
            self._v75_align_pivot_extra_mm = 20.0
            self._v75_align_pivot_min_horizontal_lever_mm = 120.0
            self._v75_align_pivot_min_applied_mm = 90.0
            self._v76_align_arm2_z_extra_mm = 0.0
            self._v76_finish_commit_queued = False
            self._v74_finish_processing = False
            self._v74_finish_complete_banner = False
            self._v74_post_finish_standby_pending = False
            self._v101_skip_align_autojudge_once = False
            self._v101_waist_arm2_target_inset_mm = 70.0
            self._v101_waist_arm2_preferred_inset_min_mm = 65.0
            self._v101_waist_arm2_preferred_inset_max_mm = 75.0
            self._v101_waist_arm2_relaxed_inset_min_mm = 55.0
            self._v101_waist_arm2_relaxed_inset_max_mm = 85.0
            self._v102_align_center_t_values = (0.28, 0.34, 0.40, 0.46, 0.50)
            self._v102_align_center_preferred_t = 0.38
            self._v102_align_crotch_margin_t = 0.08
            self._v102_align_display_t_max = 0.52
            self._v102_waist_relocate_steps_mm = (5.0, -5.0, 10.0, -10.0, 15.0, -15.0, 20.0, -20.0, 25.0, -25.0, 30.0, -30.0, 40.0, -40.0, 50.0, -50.0)
            self._v102_waist_relocate_soft_tokens = ("PRIVATE_WORKSPACE", "_REACH:")
            self._v103_waist_relocate_normal_steps_mm = (10.0, 20.0, 30.0, 40.0, -10.0, -20.0, -30.0)
            self._v103_waist_relocate_combo_tangent_mm = (15.0, -15.0, 30.0, -30.0)
            self._v103_waist_relocate_combo_normal_mm = (15.0, 30.0, -15.0)
            self._v103_waist_perception_attempts = 3
            self._v103_waist_semantic_inversion_ratio = 1.35
            self._v103_waist_semantic_inversion_margin_mm = 80.0
            self._v103_position_seg_retry_count = 2
            self._v103_position_seg_retry_wait_s = 0.08
            self._v103_pre_pullup_min_lift_mm = 15.0
            self._v104_align_asym_width_trigger_mm = 65.0
            self._v104_align_asym_width_release_mm = 45.0
            self._v104_align_asym_length_trigger_mm = 80.0
            self._v104_align_asym_length_release_mm = 55.0
            self._v104_align_asym_narrow_ratio = 0.55
            self._v104_align_asym_move_min_mm = 30.0
            self._v104_align_asym_move_max_mm = 85.0
            self._v104_align_asym_source_inset_mm = 15.0
            self._v104_press_pure_z_clear_mm = 180.0
            self._v104_press_pure_z_verify_mm = 145.0
            self._v104_press_pure_z_speed = 0.70
            self._v105_position_relaxed_board_margin_mm = 0.0
            self._v105_position_relaxed_max_candidates = 220
            self._v106_position_grip_inset_offset_mm = 70.0
            self._v106_position_contact_z_extra_mm = -3.0
            self._v107_position_rotation_guard_max_probes = 5
            self._v107_position_rotation_contact_max_candidates = 80
            self._v107_waist_arm2_min_preferred_actual_inset_mm = 50.0
            self._v107_waist_arm2_bodyward_steps_mm = (20.0, 40.0, 60.0)
            self._v105_waist_relaxed_reach_max_mm = 430.0
            self._v105_waist_reach_clamp_target_mm = 424.0
            self._v105_waist_reach_clamp_max_shift_mm = 40.0
            self._v105_waist_min_pair_sep_mm = 90.0
            self._v102_finish_stats_refresh_state = None
            self._v74_post_finish_safe_z_mm = 180.0
            self._v74_post_finish_raise_speed = 0.45
            self._v78_initial_spread_busy = False
            self._v78_initial_spread_done = False
            self._v78_initial_spread_pull_mm = 110.0
            self._v78_initial_spread_min_pull_mm = 45.0
            self._v78_initial_spread_target_inset_mm = 55.0
            self._v79_initial_spread_inset_tiers_mm = (55.0, 50.0, 45.0)
            self._v79_initial_spread_support_radius_px = 7
            self._v79_initial_spread_support_min_ratio = 0.82
            self._v78_initial_spread_top_band_mm = 135.0
            self._v78_initial_spread_min_pair_sep_mm = 140.0
            self._v80_pullup_recenter_max_mm = 35.0
            self._v80_pullup_recenter_deadband_mm = 4.0
            self._v80_pullup_recenter_speed = 0.30
            self._v80_pullup_recenter_tol_mm = 25.0
            self._v84_standby_speed = 0.50
            self._v85_basket_standby_speed = 0.24
            self._v85_initial_spread_hover_speed = 1.00
            self._v85_initial_spread_down_speed = 0.60
            self._v85_initial_spread_sweep_speed = 0.55
            self._v85_initial_spread_lift_speed = 0.80
            self._v85_align_long_move_threshold_mm = 100.0
            self._v85_align_long_move_z_extra_mm = 0.0
            self._v86_align_arm2_long_extra_mm = 20.0
            self._v86_align_arm2_segment1_speed = 0.65
            self._v86_align_arm2_segment2_speed = 0.60
            self._v87_waist_standby_speed = 0.50
            self._v87_waist_standby_poll_s = 0.08
            self._v87_waist_standby_timeout_s = 2.80
            self._v87_finish_cache = None
            self._v87_finish_cache_max_age_s = 6.0
            self._v87_finish_requested = False
            self._v87_finish_wait_queued = False
            self._v87_position_rotation_preferred_ratio = 0.28
            self._v87_position_rotation_hard_ratio = 0.48
            self._v87_position_rotation_limited_move_mm = 105.0
            self._v87_align_arm1_center_z_extra_mm = -4.0
            self._v88_last_executed_align_signature = None
            self._v88_align_pivot_arm1_body_t_min = 0.30
            self._v88_align_pivot_arm1_body_t_max = 0.78
            self._v88_align_pivot_arm1_body_t_preferred = 0.58
            self._v88_press_pair_max_along_delta_mm = 58.0
            self._v88_press_pair_mid_span_gain = 0.58
            self._v88_waist_curve_outward_limit_mm = 42.0
            self._v88_waist_curve_bodyward_limit_mm = 24.0
            self._v88_waist_grip_body_max_mm = 68.0
            self._v88_waist_grip_tangent_max_mm = 32.0
            self._v90_waist_curve_outward_limit_mm = 190.0
            self._v90_waist_curve_bodyward_limit_mm = 75.0
            self._v90_waist_grip_body_max_mm = 72.0
            self._v90_waist_grip_tangent_max_mm = 150.0
            self._v90_waist_inset_preferred_min_mm = 40.0
            self._v90_waist_inset_preferred_max_mm = 68.0
            self._v90_waist_inset_hard_max_mm = 85.0
            self._v90_waist_pair_shift_penalty = 0.40
            self._v90_waist_pair_sep_change_penalty = 0.08
            self._v88_motion_epoch = 0
            self._v110_next_scheduled_epoch = -1
            self._v111_action_mask_retry_count = 2
            self._v111_action_mask_retry_wait_s = 0.10
            self._v111_auto_execute = True
            self._v111_autoexec_lock = threading.RLock()
            self._v111_autoexec_token = None
            self._v111_align_internal_finish_disabled = True
            self._v111_pivot_soft_poll_s = 0.06
            self._v111_pivot_soft_timeout_approach_s = 2.40
            self._v111_pivot_soft_timeout_contact_s = 1.90
            self._v111_pivot_soft_timeout_drag_s = 1.55
            self._v111_pivot_soft_timeout_clear_s = 2.40
            self._v111_pivot_clear_z_pass_mm = 150.0
            self._v111_pivot_drag_progress_pass = 0.90
            self._v85_align_fast_mode = False
            self._v84_standby_first_poll_s = 0.10
            self._v84_standby_retry_poll_s = 0.15
            self._v94_startup_standby_speed = 0.75
            self._v94_startup_first_poll_s = 0.04
            self._v94_startup_retry_poll_s = 0.08
            self._v94_arm2_waist_tangent_inset_mm = 70.0
            self._v95_press_force_source_offset_mm = 80.0
            self._v95_press_force_sweep_mm = 72.0
            self._v96_press_force_min_pair_sep_mm = 140.0
            self._v96_waist_pullup_extra_mm = 50.0
            self._v96_waist_gravity_settle_s = 0.35
            self._v95_finish_commit_worker_active = False
            self._v98_human_finish_snapshot = None
            self._v98_human_finish_snapshot_lock = threading.RLock()
            self._v100_finish_queued_action = None
            self._v100_finish_queued_origin = "HUMAN"
            self._v100_finish_queue_lock = threading.RLock()
            self._v98_press_mask_inset_px = 38.0
            self._v98_press_strong_crease_min_score = 0.28
            self._v98_press_strong_severity_min = 0.55
            self._v98_align_mask_retry_count = 2
            self._v98_waist_top_y_threshold_mm = -220.0
            self._v98_waist_top_motion_options = 8
            self._v94_waist_depth_balance_max_mm = 10.0
            self._v94_waist_depth_balance_penalty = 6.0
            if hasattr(args, "imaging_standby_wait_s"):
                args.imaging_standby_wait_s = min(
                    float(getattr(args, "imaging_standby_wait_s", 0.10)), 0.10
                )
            fix134c_recorder_cls = None
            fix134c_recorder_initialize = None
            fix134c_inference_only = (
                self._integration_mode
                and str(os.environ.get('BOTTOM_INFERENCE_ONLY', '1')).strip().lower()
                not in {'0', 'false', 'no'}
            )
            if fix134c_inference_only:
                candidate = getattr(base, 'DatasetRecorder', None)
                original = getattr(candidate, 'initialize', None) if candidate is not None else None
                if callable(original):
                    fix134c_recorder_cls = candidate
                    fix134c_recorder_initialize = original
                    def _fix134c_fast_recorder_initialize(recorder_self):
                        with recorder_self.lock:
                            if recorder_self.initialized:
                                return
                            recorder_self.initialized = True
                            recorder_self.dataset_stats = recorder_self._empty_dataset_stats()
                        print('[FIX134C-FAST-STARTUP] inference-only dataset full scan skipped before AUTO-JUDGE')
                    candidate.initialize = _fix134c_fast_recorder_initialize
            try:
                super().__init__(args)
            except BaseException:
                early = getattr(self, '_fix134c_early_arms', None)
                if isinstance(early, dict):
                    for arm in early.values():
                        try:
                            arm.close()
                        except Exception:
                            pass
                raise
            finally:
                if fix134c_recorder_cls is not None and fix134c_recorder_initialize is not None:
                    fix134c_recorder_cls.initialize = fix134c_recorder_initialize
            self.bottom_vla_build = BUILD
            self.front.basket_placement_extra_deg = 20.0
            self.front.basket_placement_lower_mm = 140.0
            self.front.basket_standby_speed = float(self._v85_basket_standby_speed)
            self._v84_install_fast_standby_helpers()
            self._v87_install_d60_fast_standby_wait()
            self.front.basket_standby_speed = float(self._v85_basket_standby_speed)
            self.front.d60_waist_grip_body_min_mm = 12.0
            self.front.d60_waist_grip_body_max_mm = 60.0
            self.front.d60_waist_grip_tangent_max_mm = 45.0
            self.front.d60_waist_grip_endpoint_radius_max_mm = 80.0
            self.front.d60_pose_waist_min_width_mm = 80.0
            self.front.d60_pose_waist_max_width_mm = 470.0
            self.front.d60_waist_hard_max_width_mm = 9999.0
            self.front.align_waist_target_y_mm = -90.0
            self.d56_args.d31_grip_open = 1.35
            self.d56_args.step60_pullup_speed = 0.50
            self.d56_args.step62_v39_gravity_settle_s = float(self._v96_waist_gravity_settle_s)
            self.d56_args.step60_3_pre_pullup_plateau_polls = 0
            self.d56_args.step60_3_pre_pullup_min_plateau_lift_mm = float(self._v103_pre_pullup_min_lift_mm)
            self.d56_args.step65_pre_tension_per_arm_mm = 0.0
            self.d56_args.step67_outbound_edge_mm_s = 300.0
            self.d56_args.step67_outbound_deep_mm_s = 260.0
            self.d56_args.step65_outbound_verify_tol_mm = 20.0
            self.d56_args.step68_outbound_accept_without_correction_mm = float("nan")
            self._v96_install_d60_fast_prelift()
            self._v80_install_d60_pullup_recenter()
            self._v33_d54_old_arm2_inward = float(getattr(self.d54_args, "d54_arm2_physical_inward_mm", 12.0))
            self._v33_d54_old_arm1_inward = float(getattr(self.d54_args, "d54_arm1_physical_inward_mm", 5.0))
            self._v33_d54_old_target_depth = float(getattr(self.d54_args, "d54_physical_target_depth_mm", 22.0))
            self.d54_args.d54_arm2_physical_inward_mm = max(self._v33_d54_old_arm2_inward, 35.0)
            self.d54_args.d54_arm1_physical_inward_mm = max(self._v33_d54_old_arm1_inward, 35.0)
            self.d54_args.d54_physical_target_depth_mm = max(self._v33_d54_old_target_depth, 32.0)
            self.d55_args.d55v11_xy_guard_radius_mm = 10.0
            self.d55_args.d55v11_xy_guard_min_ratio = 0.85
            self.d55_args.d55v11_xy_guard_max_inward_mm = 60.0
            self.d55_args.d55v14_mask_core_inset_mm = 25.0
            self._v30_d58_arm1_z_offset_mm = -10.0
            self._v30_align_surface_z_offset_mm = -8.0
            self._v30_align_line_done_mm = 28.0
            self._v30_align_large_endpoint_error_mm = 70.0
            self._v30_align_angle_done_deg = 5.0
            self._v30_align_move_min_mm = 35.0
            self._v30_align_move_max_mm = 155.0
            self._v50_d60_shallow_body_max_mm = 44.0
            self._v50_d60_reach_tangent_max_mm = 90.0
            self._v50_d60_reach_radius_max_mm = 125.0
            self._v50_align_axis_good_deg = 5.0
            self._v50_align_full_gap_max_angle_deg = 12.0
            self._v50_align_full_gap_endpoint_gap_mm = 70.0
            self._v50_align_left_overflow_trigger_mm = 15.0
            self._v51_basket_close_soft_min_rad = 2.68
            self._v51_basket_close_stable_span_rad = 0.04
            self._v51_basket_soft_depth_below_rim_mm = 90.0
            self._install_full_board_mask_roi()
            self._install_d58_arm1_contact_z_offset()
            self._install_post_basket_position_policy()
            self._install_align_surface_z_offset()
            self._install_align_waist_level_policy()
            self._install_align_centerline_policy()
            self._install_align_press_drag_executor()
            self._install_align_v56_overlay()
            self._install_v48_d55_validator()
            d55_arm2_base = getattr(self.d55_args, "arm2_press_z", None)
            if d55_arm2_base is None:
                d55_arm2_base = getattr(self.d55_args, "press_z", None)
            if d55_arm2_base is not None:
                self.d55_args.arm2_press_z = float(d55_arm2_base) - 5.0
            print(f"[BOTTOM-VLA] build={BUILD}")
            print(
                f"[V84-OPT] non-BASKET standby returns target speed={float(self._v84_standby_speed):.2f}; "
                "dual-arm returns remain parallel; generic/startup/INITIAL_SPREAD/post-FINISH standby waits are shortened and still feedback-verified"
            )
            print(f"[BASKET-V85-STANDBY] post-BASKET ARM2 standby restored to original speed={float(self._v85_basket_standby_speed):.2f}")
            print(
                f"[INITIAL-SPREAD-V85-SYNC] both arms HOVER/DOWN/MID/SWEEP/LIFT parallel; "
                f"speeds hover={self._v85_initial_spread_hover_speed:.2f} down={self._v85_initial_spread_down_speed:.2f} "
                f"sweep={self._v85_initial_spread_sweep_speed:.2f} lift={self._v85_initial_spread_lift_speed:.2f}"
            )
            print("[ALIGN-V85-FAST] one fresh Seg+Pose observation -> Mask outer + center/mid-vector planner; D60 waist/ribbon planning skipped")
            print(
                f"[ALIGN-V85-LONG-Z] CENTER/PIVOT move>={self._v85_align_long_move_threshold_mm:.0f}mm "
                f"adds {self._v85_align_long_move_z_extra_mm:+.1f}mm contact/drag Z on the moving arm"
            )
            print(
                f"[ALIGN-V86-ARM2-LONG] ARM2-only CENTER move>={self._v85_align_long_move_threshold_mm:.0f}mm "
                f"requests +{self._v86_align_arm2_long_extra_mm:.0f}mm safe extension and uses 2 low-Z drag segments "
                f"at {self._v86_align_arm2_segment1_speed:.2f}/{self._v86_align_arm2_segment2_speed:.2f}"
            )
            print(f"[WAIST-V87-STANDBY] D60 internal standby speed={self._v87_waist_standby_speed:.2f}; fixed conservative wait replaced by feedback polling")
            print("[WAIST-V87-CURVE] Pose waist points are anchors only; actual waist line/grips follow the real Mask outer curve with hard 35mm inset")
            print("[POSITION-V87-VACANCY] largest single board vacancy is primary; contact is reselected to minimize rotation torque before drag")
            print(f"[ALIGN-V87-ARM1-Z] ARM1 CENTER extra={self._v87_align_arm1_center_z_extra_mm:+.1f}mm; ARM1 PIVOT base extra={self._v73_align_arm1_z_extra_mm:+.1f}mm")
            print("[FINISH-V98-HUMAN] HUMAN FINISH freezes the exact button-press live frame; auto-judge cache is never reused for human FINISH")
            print("[FINISH-V88-MOTION-CACHE] AUTO FINISH cache remains available; HUMAN FINISH bypasses it in V98")
            print(
                f"[ALIGN-V88-REPEAT] last executed ALIGN is blocked if the next fresh plan is near-identical; "
                f"PIVOT ARM1 bodyT prefers {self._v88_align_pivot_arm1_body_t_preferred:.2f} (legward)"
            )
            print("[ALIGN-V88-PIVOT-SYNC] HOVER parallel -> ARM2 press/hold first -> ARM1 press+drag -> dual CLEAR/STANDBY parallel")
            print(
                f"[WAIST-V90-REACHABLE-PAIR] Pose anchors the waist; real Mask outer curve may move outward up to {self._v90_waist_curve_outward_limit_mm:.0f}mm "
                f"but only {self._v90_waist_curve_bodyward_limit_mm:.0f}mm toward the legs; grasp search stays 35..{self._v90_waist_inset_hard_max_mm:.0f}mm inside the real Mask and slides along the waist for reach"
            )
            print("[WAIST-V90-RIGID-SHIFT] if one side is unreachable, pair search prefers a common waist-tangent translation before asymmetric fallback; deterministic pair is motion-build checked")
            print("[WAIST-V91-SEPARATION] pair waist-tangent separation is direction-invariant; ARM1/ARM2 ordering can no longer reject a physically separated safe pair")
            print("[WAIST-V91-GUARANTEE] if V90 relocation fails but the already-built source pair passes 35mm real-Mask inset + board + reach + dual spacing + motion-build, keep that pair instead of returning NO PLAN")
            print("[WAIST-V92-BASE-PAIR-GUARANTEE] every base WAIST pair is safety-validated before V90 replacement; if V90 relocation fails, the safe base pair is retained and endpoint-gate accepted")
            print("[WAIST-V101-DIRECT] real Mask waist drives grips; ARM2 targets 70mm actual inset with 65..75 preferred, relaxed fallback, and no inset-only NO-PLAN veto")
            print("[WAIST-V96-GRIP] contact close -> final close -> torque latch runs without T105 feedback retry dwell")
            print("[WAIST-V96-PULLUP] 50mm pre-lift keeps motion verification, plateau dwell is removed, and pull-up destination extends +50mm toward board top")
            print(
                f"[PRESS-V88-RIDGE-SIDES] PRESS uses opposite sides of the same validated red wrinkle; "
                f"along-pair mismatch<={self._v88_press_pair_max_along_delta_mm:.0f}mm; stationary/legacy fallback disabled"
            )
            print("[V89-STARTUP-ROLLBACK] v88 splash/background warmup removed; stable synchronous v87 startup restored")
            print(
                "[V94-STARTUP] redundant startup T104 is skipped when feedback already proves camera-clear; "
                "TensorRT warmup/model readiness remains preserved before controls become READY to avoid first-inference regressions"
            )
            print("[BASKET-V33] placement extra rotation=20.0deg; final closed descent 180->140mm, then release")
            print(f"[BASKET-V51-CLOSE] primary>=2.80rad; stable cloth-contact fallback>={self._v51_basket_close_soft_min_rad:.2f}rad; final 3.32rad + T107=1000 latch preserved")
            print(f"[BASKET-V52-REGRASP] retention is checked at the existing pre-rotation lift pause; on failure ARM2 lowers vertically at the same XY and re-grasps up to {self._v52_basket_regrasp_max} times")
            print(f"[BASKET-V52-RELEASE] successful placement opens in <= {self._v52_basket_open_step_rad:.2f}rad steps with {self._v52_basket_open_step_wait_s:.2f}s gaps before final verified release")
            print("[BASKET-V52-FAIL-RETURN] after exhausted re-grasp attempts, release occurs low at the basket; ARM2 then raises vertically before any standby transit")
            print(f"[BASKET-V57-OPEN-RELIEF] post-contact OPEN jam -> same-XY +{self._v57_basket_open_relief_mm:.0f}mm vertical relief -> retry; persistent jam -> vertical raise before failure finalization")
            print("[BASKET-V33-STANDBY] ARM2 standby T104 is re-sent up to 4 times if feedback does not reach standby")
            print("[D60-V78-RIGID-OUTBOUND] WAIST EDGE/DEEP=300/260mm/s; final pair tracking tolerance=20mm and asymmetric no-correction acceptance disabled")
            print("[PERCEPTION-V33] small CRUMPLED/BGDIFF fallback under 45% of last trusted mask gets one fresh retry, then execution is blocked if still undersized")
            print("[SOURCE-V33] on-disk source hash changes warn but do not brick the already-loaded runtime")
            print("[UI-V33] mouse callback and original 1240x410 panel / camera resize preserved")
            print("[MASK-UI] manual MASK ACCURATE / MASK INACCURATE controls remain removed")
            print("[AUTO-JUDGE] fresh current-state semantic selection preserved")
            print("[D60-V33] unchanged v49 waist gate remains available: body=12..60mm tangent<=45mm radius<=80mm poseWidth=80..470mm")
            print("[D60-V50-WAISTLINE] first search keeps the 35mm garment grasp-safe inset but prefers body depth<=44mm; old 12..60mm planner remains the next fallback")
            print("[D60-V50-REACH] if the detected waist is valid but the pair is unreachable, retry farther ALONG the waist (tangent<=90mm/radius<=125mm) without increasing the robot reach limit")
            print(f"[D60-V66-WAIST] formed waist validation: 3/3 edge support=strong PASS; 2/3 allowed only with third point <= {float(self._v66_waist_conditional_max_edge_mm):.0f}mm from real edge + independent outer agreement; final 35mm grasp inset unchanged")
            print("[D58-V55-POST-BASKET] first POSITION after Basket preserves BOTTOM+RIGHT travel and reselects the stable leading-side interior grip using the D58 safe contact helper")
            print(f"[ALIGN-V64] preferred order=WAIST LEVEL ONLY -> fresh recheck -> optional WAIST UP ONLY(targetY={float(self.front.align_waist_target_y_mm):.0f}mm) -> CENTER -> RED; if preferred waist stage has no executable pair, an already-safe CENTER/RED plan remains executable")
            print(f"[ALIGN-V63-CENTER] contact prefers mid-vector (not crotch), maximizes actual executable drag, and may parallel-shift source+target toward ARM2 while preserving requested move; skip <{float(self._v59_align_center_min_action_mm):.0f}mm")
            print(f"[D55-V56-REACH] PRESS arm beyond {float(self._v56_press_reach_trigger_mm):.0f}mm robot-XY radius is horizontally translated toward the closer safe side; source/mid/target move together")
            print("[REJUDGE-V56] REJUDGE blacklists the rejected frozen plan; near-identical source/target/vector plans are diversified or blocked")
            print(f"[D60-V53-INSET] final WAIST grasp centers must stay >= {self._v53_waist_hard_inset_mm:.0f}mm inside the real garment outer Mask (1mm raster/calibration tolerance); v33 deep/overlap rescue is disabled")
            print("[D60-V34-GRIP] pre-contact OPEN restored to 1.35rad and is never T105-retried; CLOSE 3.05 / FINAL 3.32 keep feedback verification")
            print("[D60-V51-GRIP] CLOSE/FINAL_CLOSE feedback is diagnostic/retry only; a low cloth-blocked angle no longer aborts an already-started WAIST motion")
            print(f"[D54-V33-INSET] adaptive physical inward cap A2={self._v33_d54_old_arm2_inward:.1f}->{float(self.d54_args.d54_arm2_physical_inward_mm):.1f}mm A1={self._v33_d54_old_arm1_inward:.1f}->{float(self.d54_args.d54_arm1_physical_inward_mm):.1f}mm targetDepth={self._v33_d54_old_target_depth:.1f}->{float(self.d54_args.d54_physical_target_depth_mm):.1f}mm")
            print("[D55-V33-MASK-GUARD] contact support radius=10mm ratio>=0.85 mask-core>=25mm; inward correction<=60mm while preserving sweep vector and Z")
            print("[D58-V33] ARM1 effective contact/mid/sweep surface Z offset=-10.0mm")
            print("[D58-V73-VACANCY] POSITION keeps the original widest-free-board vacancy direction as the actual drag direction; board-center vector is fallback only")
            print("[D58-V51-POST-BASKET] first POSITION after any BASKET attempt keeps its vacancy/contact context and V73 preserves that travel direction")
            print(f"[BASKET-V51-DEPTH] slow descent soft limit=rim-{self._v51_basket_soft_depth_below_rim_mm:.0f}mm; if torque contact is still absent, close is attempted there instead of descending deeper")
            print("[MASK-V39] D54/D55/D60/D58 use the full four-marker quadrilateral with no extra Mask inset/marker holes; D60 35mm garment grasp-safe inset unchanged")
            print(f"[ALIGN-V50-WAIST-Y] initial dual-waist recenter restored: targetY={float(self.front.align_waist_target_y_mm):.1f}mm, native trigger={float(self.front.align_dual_waist_min_pull_mm):.1f}mm")
            print("[ALIGN-V50-CENTER] WAIST/THIGH native safe source is preserved, but moderate-angle centerline correction requests the measured body-vector-to-white-line gap")
            print(f"[ALIGN-V50-CENTER] full-gap candidate angle<={self._v50_align_full_gap_max_angle_deg:.1f}deg endpointGap<={self._v50_align_full_gap_endpoint_gap_mm:.0f}mm; unsafe full target is clipped to the furthest safe fraction")
            print(f"[ALIGN-V53-FIRST] first ALIGN requires the current dual-waist Y move toward {float(self.front.align_waist_target_y_mm):.1f}mm when upward recenter is needed; unsafe/missing dual pair blocks instead of falling into single-arm correction")
            print(f"[ALIGN-V53-OUTER] red guard is placed on the visual-left side of the white line; once axis is good, visual-left overflow requests a visual-right correction and blocks if that direction is unsafe")
            print("[ALIGN-V64-PHYSICAL] LEVEL and UP stay separate; stage priority never invalidates an already-generated CENTER/RED plan that passed the planner safety checks")
            try:
                _v60_a1_standby = self.d56.standby_roarm_pose(self.cfg56, "arm1")
                print(f"[CAMERA-CLEAR-V60] startup command keeps original high standby, then verifies with ARM1 offset target Z={float(_v60_a1_standby[2]):.1f}mm; ALIGN standby is feedback-verified/retried")
            except Exception:
                pass
            print(f"[ALIGN-V62-RED] contact={self._v58_align_red_outer_inset_mm:.0f}mm inside local Mask edge; requested drag=measured overflow+{self._v59_align_red_outer_extra_drag_mm:.0f}mm; press Z={self._v58_align_red_outer_z_offset_mm:+.0f}mm vs V56")
            print("[ALIGN-V63-PERF] after waist LEVEL/UP is verified, forced D60 waist-pair search is not re-enabled; Seg/Pose/thresholds unchanged")
            print("[ALIGN-V64-DEADLOCK-FIX] a safe CENTER/RED plan is never rewritten to NO_SAFE_PLAN only because LEVEL/UP is still preferred; hard reach/board/collision safety remains active")
            print("[D55-V66-GAP] PRESS rejects only high-confidence exterior-connected inter-leg gap boundaries; internal overlap/fold creases are preserved")
            print(f"[D60-V67-FAST] accepted waist geometry is cached; grasp search uses one body-side band {self._v67_waist_body_min_mm:.0f}..{self._v67_waist_body_max_mm:.0f}mm with final real-Mask 35mm safety unchanged")
            print(f"[ALIGN-V74-CENTER] normal CENTER uses its native safe distance with no extra extension; pivot-only long-move policy remains active; CENTER Z extra={self._v70_align_center_z_extra_mm:+.0f}mm, drag={self._v70_align_drag_speed:.2f}; RED unchanged")
            print("[ALIGN-V67-PIVOT] if the garment center-vector actually intersects the white line, ARM2 holds near that intersection and ARM1 rotates/translates the cloth so both angle and line distance improve")
            print(f"[ALIGN-V70-CONTACT] CENTER drag total surface offset={float(self._v56_align_press_clearance_mm)+float(self._v70_align_center_z_extra_mm):+.1f}mm; stage1={float(self._v70_align_stage1_press_clearance_mm):+.1f}mm; speed={float(self._v70_align_drag_speed):.2f}; RED unchanged")
            print(f"[D55-V71-LEG] PRESS source band={float(self._v70_press_source_normal_min_mm):.0f}..{float(self._v70_press_source_normal_max_mm):.0f}mm from selected ridge; both moving and stationary support contacts must stay on the selected leg, with no forced inner-side preference")
            print(f"[D55-V72-LEG-SELECT] PRESS chooses LEFT/RIGHT leg before pair search; RIGHT gets only a {100.0*self._v72_press_right_prior_ratio:.0f}% soft prior, then the selected leg stays locked through all fallbacks")
            print(f"[D55-V72-INNER-PREF] same-leg contacts prefer >= {self._v72_press_inner_margin_prefer_mm:.0f}mm separation from the inter-leg decision boundary; inner contacts remain fallback-only when no better same-leg point exists")
            print("[D60-V70-SPACING] WAIST outward pre-tension disabled; inherited planner preserves the frozen source-pair separation through align/outbound whenever safe")
            print(f"[D60-V72-SAFE-SEARCH] WAIST tries {len(self._v71_waist_search_tiers)} endpoint bands, prefers both real-Mask insets near {self._v72_waist_preferred_inset_mm:.0f}mm and balanced within {self._v72_waist_fast_max_balance_mm:.0f}mm, while keeping the hard 35mm minimum")
            print(f"[D60-V72-WAIST-OUTER] Pose locates the waist end; accepted waist geometry uses the real Mask outer envelope, with bodyward refinement allowance <= {self._v72_waist_outer_max_bodyward_shift_mm:.0f}mm")
            print(f"[D60-V74-ARM-BIAS] WAIST final source bias ARM1={self._v73_waist_arm1_outward_mm:.0f}mm toward waist outer edge, ARM2={self._v73_waist_arm2_inward_mm:.0f}mm toward garment body; hard 35mm real-Mask inset is rechecked")
            print(f"[ALIGN-V75-PIVOT] if |midGap|>{self._v75_align_pivot_distance_gate_mm:.0f}mm use one-arm CENTER first; once close and |angle|>={self._v75_align_pivot_angle_min_deg:.0f}deg, ARM2 holds and ARM1 pulls mainly vertical with distance computed from angle+remaining gap ({self._v75_align_pivot_pull_min_mm:.0f}..{self._v75_align_pivot_pull_max_mm:.0f}mm safe request)")
            print("[FINISH-V76] FINISH plan auto-commits after inference; ENTER is not required; lock always releases in finally")
            print("[POST-FINISH-V95] FINISH terminal save runs off the UI thread; SUCCESS unlocks next action immediately and the banner clears on the next action")
            print("[PRESS-V101-STRUCTURAL] large/severe shape-changing wrinkles outrank tiny high-contrast/waist-line wrinkles; forced grasp points keep the ~38px Mask interior requirement")
            print("[ALIGN-V99] v98 mask-gate call fixed so normal YOLO passes; fallback remains blocked; done angle=5deg and Z/invalid-axis protections preserved")
            print("[D58-V101-FALLBACK] V73 vacancy-contact failure preserves an already-safe native D58 plan")
            print("[D60-V100-TOP] high-waist direct grips retry STEP63 with dynamic deeper pull-down targets before blocking")
            print("[ALIGN-V100-MASK-CENTERLINE] body axis uses fitted Mask width-centers first; Pose axis is fallback")
            print("[ALIGN-V101-STANDBY-GRIP] ARM2 gripper ramps slowly all the way to the standby tool angle before Cartesian return")
            print("[ALIGN-V102-UPPER-CONTACT] long Mask axis remains for geometry; CENTER press contacts are capped before crotch/lower-leg region")
            print("[WAIST-V102-RELOCATE] private-workspace/small-reach failures retry rigid waist-tangent pair shifts before blocking")
            print("[WAIST-V103-2D-RELOCATE] soft workspace/reach failures retry rigid body-normal and tangent+normal pair shifts before blocking")
            print("[WAIST-V104-POSE-SANITY-SOFT] v103 waist/hem inversion veto is diagnostic only; valid Pose8+Mask still generates grips")
            print(f"[ALIGN-V104-ASYMMETRY] loose trigger widthDiff>={self._v104_align_asym_width_trigger_mm:.0f}mm or legLenDiff>={self._v104_align_asym_length_trigger_mm:.0f}mm; minor asymmetry is accepted")
            print(f"[PRESS-V104-PURE-Z-CLEAR] after PRESS_SWEEP both arms rise vertically to z={self._v104_press_pure_z_clear_mm:.0f}mm before XY standby")
            print("[POSITION-V105-RELAXED] if directed vacancy is valid but the strict same-arm low-lift contact gate fails, retry a mask-supported board-inside/reachable contact without the private-workspace veto")
            print(f"[POSITION-V107-GRIP-INSET] final POSITION grasp targets {self._v106_position_grip_inset_offset_mm:.0f}mm toward the garment center with 5mm relaxed fallback and no inset-only NO-PLAN veto")
            print(f"[POSITION-V107-FAST] rotation-guard probes<={self._v107_position_rotation_guard_max_probes}, repeated contact candidates<={self._v107_position_rotation_contact_max_candidates}, relaxed fallback candidates<={self._v105_position_relaxed_max_candidates}")
            print(f"[WAIST-V107-ARM2-BODYWARD] ARM2 keeps >=~{self._v107_waist_arm2_min_preferred_actual_inset_mm:.0f}mm actual waist-end inset when possible; if the waist-line point is too outer/unreachable, retry bodyward +20/+40/+60mm before legacy fallback")
            print(f"[POSITION-V106-Z] POSITION contact/drag Z is {abs(self._v106_position_contact_z_extra_mm):.0f}mm lower for both arms; ARM1 keeps its existing extra offset")
            print(f"[WAIST-V105-BOARD-GATE] valid Mask-inside/board-inside grasp pairs retry with center dead-zone removed and reach clamped toward {self._v105_waist_reach_clamp_target_mm:.0f}mm; pair/collision spacing remains enforced")
            print("[FINISH-V105-QUALITY] HUMAN FINISH keeps exact frame, but fallback/BGDIFF Mask geometry cannot overwrite BODY AXIS and is excluded from classifier training")
            print(f"[WAIST-V103-PRELIFT] >= {self._v103_pre_pullup_min_lift_mm:.0f}mm measured rise per arm is enough to continue to pull-up")
            print(f"[POSITION-V103-SEG-RETRY] no-Mask POSITION automatically retries up to {self._v103_position_seg_retry_count} fresh frame(s)")
            print("[FINISH-V102-FAST-NEXT] next action starts before deferred full dataset-stats rescan")
            print("[FINISH-V101-NEXT] exact-frame FINISH save remains backgrounded; closed-episode Q-to-exit status no longer blocks the continuing UI")
            print("[FINISH-V98] HUMAN FINISH saves the exact button-press frame; stale auto-judge cache is ignored")
            print("[D60-V77-WAIST-ROLLBACK] WAIST restored to v74 behavior; v76 Pose8 inset rescue removed; v72 safe-pair + v74 ARM1/ARM2 bias preserved")
            print("[D60-V78-PAIR-LOCK] outbound planned rigid pair is tracked more tightly; if either arm misses >20mm, synchronized slow final correction is mandatory before gravity settle")
            print("[INITIAL-SPREAD-V78] after successful BASKET standby, detect upper Mask outer corners and run one automatic dual press-drag outward before normal auto-judge")
            print("[INITIAL-SPREAD-V79-INSET] contact depth targets=55->50->45mm using each candidate nearest real-Mask edge local inward normal; stronger local Mask support required")
            print("[INITIAL-SPREAD-V82-PRESS] INITIAL_SPREAD execution uses D55 PRESS_SWEEP closed-press motion; ALIGN executor/private-workspace precheck is not used")
            print("[INITIAL-SPREAD-V83-STANDBY] after successful spread, both arms return to verified standby before AUTO-JUDGE")
            print("[BASKET-V80-MOTION-ONLY] key/button 1 runs the fixed Basket physical sequence immediately; no YOLO/Pose, no before/after observation, no ENTER")
            print("[D60-V80-WAIST-TOP] Pose anchors the semantic waist neighborhood; strict/local real-Mask top outer edge is used before any lower body-width fallback")
            print("[D60-V80-ARM1-TANGENT] ARM1 outward 30mm follows the waist tangent toward its real outer endpoint; ARM2 inward 20mm remains body-normal")
            print("[D60-V81-WAIST] Pose-constrained real outer waist + deterministic inward pair; normal Pose+Mask never NO PLAN for candidate/inset/endpoint gates")
            print("[D60-V81-VETO] WAIST final veto policy=board boundary or dual-arm collision/compatibility only")
            print(f"[D60-V80-PRE-PULLUP-CENTER] after the existing +50mm lift, translate the rigid pair toward board center by up to {self._v80_pullup_recenter_max_mm:.0f}mm before pull-up")
            print(f"[AUTO-JUDGE-V80-FIX] large endpoint error threshold initialized to {self._v30_align_large_endpoint_error_mm:.0f}mm")
            print(f"[ALIGN-V76-ARM2-Z] CENTER/PIVOT ARM2 contact is {abs(self._v76_align_arm2_z_extra_mm):.0f}mm lower than v75; ARM1 and RED unchanged")
            print("[AUTO-JUDGE-V74] order=POSITION -> WAIST -> ALIGN -> PRESS -> FINISH; OUTER_PULL runtime selection disabled")
            print("[HUMAN-INTERRUPT-V69] human override invalidates the active generation; PRESS/WAIST/ALIGN local planning checks cancellation cooperatively before heavy loops")
            print("[D55-V33] folded-leg geometry separates FOLD_BOUNDARY from normal wrinkles and enables FOLDED_LEG_RECOVERY")
            print("[D55-V48] center-vector shape symmetry is evaluated first; symmetric pants use wrinkle-cleanup ranking, asymmetric pants prioritize shape-causing wrinkles")
            print("[D55-V48] PRESS sweep stays wrinkle-normal; source points come from the two cloth sides and each side may move a different distance")
            print("[D55-V48-VALIDATOR] legacy pants-center radial/inward guard is bypassed only for V48 PRESS plans; board rectangle + dual collision compatibility remain mandatory")
            print("[D55-V52-MASK-FALLBACK] only after all V48/V43/inherited wrinkle plans fail, a trustworthy garment Mask can generate multiple interior dual-source candidates; board/reach/collision compatibility stays mandatory")
            if d55_arm2_base is not None:
                print(f"[D55-V33] ARM2 press Z {float(d55_arm2_base):.1f}->{float(self.d55_args.arm2_press_z):.1f}mm preserved from v27; ARM1 unchanged")
            self._v110_init_action_models()


        def _v88_mark_motion_started(self, action):
            action_u = str(action or "").upper()
            if action_u == "FINISH":
                return
            self._v88_motion_epoch = int(getattr(self, "_v88_motion_epoch", 0)) + 1
            self._v87_finish_cache = None
            self._v87_finish_requested = False
            if action_u not in {"ALIGN", "ALIGN_ARM2_CORRECTION"}:
                self._v88_last_executed_align_signature = None
            print(f"[V88-MOTION-EPOCH] epoch={self._v88_motion_epoch} action={action_u or '-'}")

        def _v88_remember_executed_align(self, action_plan):
            sig = self._v56_signature_from_obj("ALIGN", action_plan, "ARM2_CORRECTION")
            if sig is None:
                return
            self._v88_last_executed_align_signature = copy.deepcopy(sig)
            print(
                f"[ALIGN-V88-LAST-PLAN] remembered executed ALIGN for one-step repeat guard "
                f"paths={list(sig.get('paths', {}))}"
            )

        def _v88_align_repeat_guard(self, obs, seam_x, action_plan, decision):
            last = self._v88_last_executed_align_signature
            if last is None or action_plan is None:
                return action_plan, decision
            sig = self._v56_signature_from_obj("ALIGN", action_plan, "ARM2_CORRECTION")
            if sig is None or not self._v56_signatures_similar(sig, last):
                return action_plan, decision
            print("[ALIGN-V88-REPEAT-BLOCK] fresh ALIGN reproduced the last executed plan; request one distinct safe alternative")
            old_origin = str(getattr(self, "_prepare_origin", ""))
            old_rejected = copy.deepcopy(self._v56_rejected_plans)
            try:
                if not any(self._v56_signatures_similar(last, x) for x in self._v56_rejected_plans):
                    self._v56_rejected_plans.append(copy.deepcopy(last))
                    self._v56_rejected_plans = self._v56_rejected_plans[-int(self._v56_rejected_max):]
                self._prepare_origin = "REJUDGE"
                alt_plan, alt_decision = self.align._align7_build_auto_plan(
                    obs, self.H_raw, None, seam_x, self.align_correction_count,
                    self.config, self.align_cfg, self.align_args,
                )
            finally:
                self._prepare_origin = old_origin
                self._v56_rejected_plans = old_rejected
            alt_sig = self._v56_signature_from_obj("ALIGN", alt_plan, "ARM2_CORRECTION") if alt_plan is not None else None
            if alt_plan is not None and alt_sig is not None and not self._v56_signatures_similar(alt_sig, last):
                print("[ALIGN-V88-REPEAT-DIVERSIFY] distinct safe ALIGN alternative selected without new inference")
                return alt_plan, alt_decision
            out_decision = dict(decision or {}) if isinstance(decision, dict) else {"reason": str(decision)}
            out_decision["decision"] = "V88_REPEAT_ALIGN_BLOCKED_NO_DISTINCT_SAFE_PLAN"
            out_decision["reason"] = "last executed ALIGN would repeat with no distinct safe alternative"
            print("[ALIGN-V88-REPEAT-NOPLAN] repeated plan blocked; no distinct safe alternative")
            return None, out_decision


        def _v60_apply_arm1_standby_offset_once(self):
            offset = float(self._v60_arm1_standby_offset_mm)
            changed = []
            for name in ("cfg56", "cfg54", "cfg55", "align_cfg", "d56_args", "d54_args", "d55_args", "align_args"):
                obj = getattr(self, name, None)
                if obj is None or not hasattr(obj, "arm1_standby_roarm_z"):
                    continue
                oid = id(obj)
                if oid in self._v60_arm1_standby_offset_ids:
                    continue
                old = float(getattr(obj, "arm1_standby_roarm_z"))
                new = old + offset
                setattr(obj, "arm1_standby_roarm_z", new)
                self._v60_arm1_standby_offset_ids.add(oid)
                changed.append((name, old, new))
            if changed:
                summary = ", ".join(f"{name}:{old:.1f}->{new:.1f}" for name, old, new in changed)
                print(f"[ARM1-STANDBY-Z-V60] offset={offset:+.1f}mm applied once: {summary}")
            return bool(changed)

        def _apply_arm1_standby_z_offset(self):
            # v23 calls this again after ALIGN args/cfg are created. Object-id tracking
            # prevents a second -20 mm shift on cfg56/d56_args while still updating
            # the late-created ALIGN objects.
            self.arm1_standby_z_offset_mm = float(self._v60_arm1_standby_offset_mm)
            return self._v60_apply_arm1_standby_offset_once()

        def _v84_fast_move_arm_to_standby(self, module, arm_key, arms, cfg, move_command=104, reason="standby"):
            arm_key = str(arm_key).lower()
            arm = arms.get(arm_key) if isinstance(arms, dict) else None
            if arm is None:
                print(f"[V84-STANDBY] {arm_key.upper()} skipped: serial arm not connected")
                return False
            if not bool(getattr(cfg, "imaging_standby", True)):
                print(f"[V84-STANDBY] {arm_key.upper()} skipped: imaging_standby=False")
                return False
            if bool(getattr(cfg, "dry_run", False)) or not bool(getattr(cfg, "send", True)):
                print(
                    f"[V84-STANDBY] {arm_key.upper()} dry-run/send guard: "
                    f"send={getattr(cfg, 'send', None)} dry_run={getattr(cfg, 'dry_run', None)}"
                )
                return False
            try:
                x, y, z, tool_t = module.standby_roarm_pose(cfg, arm_key)
            except Exception as exc:
                print(f"[V84-STANDBY] {arm_key.upper()} target unavailable: {exc!r}")
                return False
            reason_u = str(reason).upper()
            speed = float(self._v94_startup_standby_speed) if "BOTTOM_VLA_STARTUP" in reason_u else float(self._v84_standby_speed)
            print(
                f"[V84-STANDBY] {arm_key.upper()} {reason}: "
                f"roarm=({float(x):.1f},{float(y):.1f},{float(z):.1f}) speed={speed:.2f}; gripper unchanged"
            )
            arm.move_goal(
                float(x), float(y), float(z), float(tool_t), speed,
                move_command=int(move_command),
            )
            return True

        def _v84_fast_move_arms_to_standby(self, module, arms, cfg, move_command=104, reason="standby"):
            if not bool(getattr(cfg, "imaging_standby", True)):
                print("[V84-STANDBY] skipped: imaging_standby=False")
                return False
            if bool(getattr(cfg, "dry_run", False)) or not bool(getattr(cfg, "send", True)):
                print(
                    f"[V84-STANDBY] dry-run/send guard: "
                    f"send={getattr(cfg, 'send', None)} dry_run={getattr(cfg, 'dry_run', None)}"
                )
                return False
            keys = [key for key in ("arm2", "arm1") if isinstance(arms, dict) and key in arms]
            if not keys:
                return False
            results = {}
            errors = []
            result_lock = threading.Lock()

            def runner(key):
                try:
                    ok = self._v84_fast_move_arm_to_standby(
                        module, key, arms, cfg, move_command=move_command, reason=reason,
                    )
                    with result_lock:
                        results[key] = bool(ok)
                except Exception as exc:
                    with result_lock:
                        errors.append((key, repr(exc)))
                        results[key] = False

            threads = [threading.Thread(target=runner, args=(key,), daemon=True) for key in keys]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join(timeout=2.0)
            if errors:
                print(f"[V84-STANDBY] parallel command errors={errors}")
            return bool(results) and all(bool(results.get(key, False)) for key in keys)

        def _v84_startup_already_camera_clear(self):
            if str(getattr(self.args, "mode", "physical")) == "dry-run":
                return True
            xyz_tol = float(getattr(self.args, "standby_xyz_tolerance_mm", 45.0))
            z_tol = float(getattr(self.args, "standby_z_tolerance_mm", 35.0))
            targets = {}
            try:
                for key in ("arm1", "arm2"):
                    targets[key] = np.asarray(
                        self.d56.standby_roarm_pose(self.cfg56, key)[:3], np.float64
                    ).reshape(3)
                targets["arm1"] = targets["arm1"].copy()
                targets["arm1"][2] += float(self._v60_arm1_standby_offset_mm)
            except Exception as exc:
                print(f"[V84-STARTUP-SKIP] target unavailable: {exc!r}")
                return False

            ready = True
            for key in ("arm1", "arm2"):
                try:
                    xyz = self._query_arm_xyz(key, retries=1)
                except Exception as exc:
                    print(f"[V84-STARTUP-SKIP] {key.upper()} feedback error={exc!r}")
                    return False
                if xyz is None:
                    return False
                xyz = np.asarray(xyz, np.float64).reshape(3)
                target = targets[key]
                error = float(np.linalg.norm(xyz - target))
                z_ok = bool(float(xyz[2]) >= float(target[2]) - z_tol)
                arm_ok = bool(error <= xyz_tol and z_ok)
                print(
                    f"[V84-STARTUP-SKIP] {key.upper()} error={error:.1f}mm "
                    f"zOK={z_ok} cameraClear={arm_ok}"
                )
                ready = ready and arm_ok
            return bool(ready)

        def _v84_install_fast_standby_helpers(self):
            speed = float(self._v84_standby_speed)
            patched = []
            seen = set()
            for label, module in (
                ("D56/WAIST", getattr(self, "d56", None)),
                ("D55/PRESS", getattr(self, "d55", None)),
                ("D54", getattr(self, "d54", None)),
                ("D58/POSITION", getattr(self, "d58", None)),
                ("ALIGN", getattr(self, "align", None)),
            ):
                if module is None or id(module) in seen:
                    continue
                seen.add(id(module))
                original = getattr(module, "move_arm_to_standby", None)
                if not callable(original):
                    continue
                if bool(getattr(original, "_bottom_vla_v84_fast_standby", False)):
                    patched.append(label)
                    continue
                owner = self

                def fast_move_arm_to_standby(arm_key, arms, cfg, move_command=104, reason="standby", _module=module):
                    return owner._v84_fast_move_arm_to_standby(
                        _module, arm_key, arms, cfg, move_command=move_command, reason=reason,
                    )

                fast_move_arm_to_standby._bottom_vla_v84_fast_standby = True
                fast_move_arm_to_standby._bottom_vla_v84_original = original
                module.move_arm_to_standby = fast_move_arm_to_standby
                runtime = getattr(module, "_D26V3_RUNTIME_ARGS", None)
                if runtime is not None:
                    try:
                        setattr(runtime, "d26v3_return_speed", speed)
                        setattr(runtime, "d26v3_parallel_return", True)
                    except Exception:
                        pass
                patched.append(label)

            for name in ("d56_args", "d55_args", "d54_args", "d58_args", "align_args"):
                ns = getattr(self, name, None)
                if ns is None:
                    continue
                try:
                    setattr(ns, "d26v3_return_speed", speed)
                    setattr(ns, "d26v3_parallel_return", True)
                except Exception:
                    pass
            try:
                self.front.basket_standby_speed = float(self._v85_basket_standby_speed)
            except Exception:
                pass
            print(
                f"[V84-STANDBY-FAST] speed={speed:.2f}; parallel dual return preserved; "
                f"patched={patched}; Basket ARM2 standby={float(getattr(self.front, 'basket_standby_speed', self._v85_basket_standby_speed)):.2f}"
            )

        def _v104_press_pure_z_clear(self):
            if not bool(getattr(self.cfg56, "send", True)) or bool(getattr(self.cfg56, "dry_run", False)):
                return True
            targets = {}
            errors = []
            for key in ("arm2", "arm1"):
                arm = self.arms.get(key)
                if arm is None:
                    errors.append((key, "ARM_MISSING"))
                    continue
                try:
                    fb = self.align._d31_query_feedback(arm, timeout_s=0.9)
                    xyz = self.align._d31_feedback_xyz(fb)
                    if xyz is None:
                        errors.append((key, "FEEDBACK_XYZ_MISSING"))
                        continue
                    xyz = np.asarray(xyz, np.float64).reshape(3)
                    hand = self._v100_feedback_gripper_angle(fb)
                    if hand is None:
                        hand = float(getattr(self.cfg56, "tool_t_rad", 3.14))
                    targets[key] = (xyz, float(hand))
                except Exception as exc:
                    errors.append((key, repr(exc)))
            if errors or len(targets) != 2:
                print(f"[PRESS-V104-PURE-Z-CLEAR-BLOCK] feedback errors={errors}")
                return False

            move_command = int(getattr(self.args, "move_command", 104))
            target_z = float(self._v104_press_pure_z_clear_mm)
            speed = float(self._v104_press_pure_z_speed)

            def send_one(key):
                xyz, hand = targets[key]
                arm = self.arms[key]
                try:
                    arm.move_goal(
                        float(xyz[0]), float(xyz[1]), target_z, float(hand), speed,
                        move_command=move_command,
                        stage="PRESS_V104_PURE_Z_CLEAR",
                        caller="_v104_press_pure_z_clear",
                    )
                except TypeError:
                    arm.move_goal(float(xyz[0]), float(xyz[1]), target_z, float(hand), speed, move_command=move_command)
                except Exception as exc:
                    errors.append((key, repr(exc)))

            threads = [threading.Thread(target=send_one, args=(key,), daemon=True) for key in ("arm2", "arm1")]
            for th in threads:
                th.start()
            for th in threads:
                th.join(timeout=1.5)
            if errors:
                print(f"[PRESS-V104-PURE-Z-CLEAR-BLOCK] command errors={errors}")
                return False

            verify_z = float(self._v104_press_pure_z_verify_mm)
            for attempt in range(1, 7):
                all_high = True
                rows = []
                for key in ("arm2", "arm1"):
                    try:
                        fb = self.align._d31_query_feedback(self.arms[key], timeout_s=0.7)
                        xyz = self.align._d31_feedback_xyz(fb)
                        z = None if xyz is None else float(np.asarray(xyz, np.float64).reshape(3)[2])
                    except Exception:
                        z = None
                    rows.append((key, z))
                    if z is None or z < verify_z:
                        all_high = False
                print(f"[PRESS-V104-PURE-Z-CLEAR] verify {attempt}/6 " + " ".join(f"{k.upper()}z={'NA' if z is None else f'{z:.1f}'}" for k,z in rows))
                if all_high:
                    return True
                time.sleep(0.22)
            return False


        def prepare_integrated_startup_standby(self):
            if not bool(getattr(self, "_integration_mode", False)):
                raise RuntimeError("prepare_integrated_startup_standby is integration-only")
            if str(getattr(self.args, "mode", "physical")) == "dry-run":
                print("[BOTTOM-INTEGRATION-STANDBY] dry-run: startup standby accepted")
                return True
            if self._v84_startup_already_camera_clear():
                ok = bool(self._ensure_camera_clear("STARTUP", allow_move=True))
                print(f"[BOTTOM-INTEGRATION-STANDBY] already-clear fast path verified={ok}")
                return ok

            move_command = int(getattr(self.args, "move_command", 104))
            clear_z = float(getattr(self, "_v104_press_pure_z_clear_mm", 180.0))
            clear_speed = min(0.45, float(getattr(self, "_v94_startup_standby_speed", 0.75)))
            targets = {}
            errors = []

            for key in ("arm2", "arm1"):
                arm = self.arms.get(key)
                if arm is None:
                    errors.append((key, "ARM_MISSING"))
                    continue
                try:
                    fb = self.align._d31_query_feedback(arm, timeout_s=0.9)
                    xyz = self.align._d31_feedback_xyz(fb)
                    if xyz is None:
                        errors.append((key, "FEEDBACK_XYZ_MISSING"))
                        continue
                    xyz = np.asarray(xyz, np.float64).reshape(3)
                    hand = self._v100_feedback_gripper_angle(fb)
                    if hand is None:
                        hand = float(getattr(self.cfg56, "tool_t_rad", 3.14))
                    target_z = max(float(xyz[2]), clear_z)
                    targets[key] = (xyz, float(hand), target_z)
                except Exception as exc:
                    errors.append((key, repr(exc)))

            if errors or len(targets) != 2:
                print(f"[BOTTOM-INTEGRATION-STANDBY-BLOCK] feedback unavailable errors={errors}")
                return False

            send_errors = []
            send_lock = threading.Lock()

            def raise_one(key):
                xyz, hand, target_z = targets[key]
                try:
                    arm = self.arms[key]
                    try:
                        arm.move_goal(
                            float(xyz[0]), float(xyz[1]), float(target_z), float(hand), clear_speed,
                            move_command=move_command,
                            stage="BOTTOM_INTEGRATION_STARTUP_Z_CLEAR",
                            caller="prepare_integrated_startup_standby",
                        )
                    except TypeError:
                        arm.move_goal(
                            float(xyz[0]), float(xyz[1]), float(target_z), float(hand), clear_speed,
                            move_command=move_command,
                        )
                except Exception as exc:
                    with send_lock:
                        send_errors.append((key, repr(exc)))

            print(
                "[BOTTOM-INTEGRATION-STANDBY] arbitrary pose detected -> "
                f"pure-Z clear first (z>={clear_z:.1f}) -> verified standby"
            )
            threads = [
                threading.Thread(target=raise_one, args=(key,), daemon=True)
                for key in ("arm2", "arm1")
            ]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join(timeout=2.0)

            if send_errors:
                print(f"[BOTTOM-INTEGRATION-STANDBY-BLOCK] pure-Z command errors={send_errors}")
                return False

            for attempt in range(1, 8):
                ready = True
                rows = []
                for key in ("arm2", "arm1"):
                    try:
                        fb = self.align._d31_query_feedback(self.arms[key], timeout_s=0.7)
                        xyz = self.align._d31_feedback_xyz(fb)
                        z = None if xyz is None else float(np.asarray(xyz, np.float64).reshape(3)[2])
                    except Exception:
                        z = None
                    required_z = float(targets[key][2]) - 15.0
                    rows.append((key, z, required_z))
                    if z is None or z < required_z:
                        ready = False
                print(
                    f"[BOTTOM-INTEGRATION-Z-CLEAR] verify {attempt}/7 "
                    + " ".join(
                        f"{key.upper()}z={'NA' if z is None else f'{z:.1f}'}/{need:.1f}"
                        for key, z, need in rows
                    )
                )
                if ready:
                    break
                time.sleep(0.18)
            else:
                print("[BOTTOM-INTEGRATION-STANDBY-BLOCK] pure-Z clearance not verified")
                return False

            self._v60_startup_camera_clear_handled = False
            ok = bool(self._ensure_camera_clear("STARTUP", allow_move=True))
            if not ok:
                print("[BOTTOM-INTEGRATION-STANDBY-BLOCK] standby command/verification failed")
                return False
            verified = bool(self._feedback_camera_clear())
            print(f"[BOTTOM-INTEGRATION-STANDBY] final camera-clear verified={verified}")
            return verified

        def _v87_install_d60_fast_standby_wait(self):
            original = getattr(self.d56, "_d31_conservative_wait_seconds", None)
            if not callable(original) or bool(getattr(original, "_bottom_vla_v87_fast_d60_standby", False)):
                return
            targets = []
            try:
                for key in ("arm2", "arm1"):
                    sx, sy, sz, _ = self.d56.standby_roarm_pose(self.cfg56, key)
                    targets.append(np.asarray([float(sx), float(sy), float(sz)], np.float64))
            except Exception:
                targets = []

            def wrapped(start_xyz, target_xyz, speed, args, extra=0.0):
                value = float(original(start_xyz, target_xyz, speed, args, extra=extra))
                try:
                    target = np.asarray(target_xyz, np.float64).reshape(3)
                    if any(float(np.linalg.norm(target[:2] - q[:2])) <= 4.0 and abs(float(target[2] - q[2])) <= 30.0 for q in targets):
                        return min(value, 0.05)
                except Exception:
                    pass
                return value

            wrapped._bottom_vla_v87_fast_d60_standby = True
            wrapped._bottom_vla_v87_original = original
            self.d56._d31_conservative_wait_seconds = wrapped

        def _ensure_camera_clear(self, reason, allow_move=False):
            reason_text = str(reason).upper()
            if bool(allow_move) and "D55_PRESS_SWEEP" in reason_text:
                if not self._v104_press_pure_z_clear():
                    print(f"[PRESS-V104-PURE-Z-CLEAR-BLOCK] reason={reason}; diagonal standby suppressed")
                    return False
            if reason_text == "STARTUP" and bool(allow_move) and not self._v60_startup_camera_clear_handled:
                # Fast path: a normal previous shutdown already leaves both arms at
                # camera-clear standby. Verify the final effective targets first and
                # skip a redundant T104 move when no motion is needed.
                if self._v84_startup_already_camera_clear():
                    self._v60_apply_arm1_standby_offset_once()
                    self._read_raw(flush=0)
                    self._v60_startup_camera_clear_handled = True
                    print("[CAMERA-CLEAR-V84] STARTUP move skipped: both arms already camera-clear")
                    return True
                # Preserve the long-used startup move itself (ARM1 z=170.9). After the
                # move is sent, install v23's -20 mm effective imaging target BEFORE
                # the first feedback verification. This removes the 170.9-vs-150.9
                # startup mismatch without lowering the startup move command.
                moved = self._v84_fast_move_arms_to_standby(
                    self.d56, self.arms, self.cfg56, move_command=int(self.args.move_command),
                    reason="BOTTOM_VLA_STARTUP_V84",
                )
                if not moved:
                    return False
                self._v60_apply_arm1_standby_offset_once()
                attempts = max(1, int(self.args.standby_verify_retries))
                time.sleep(float(self._v94_startup_first_poll_s))
                for attempt in range(1, attempts + 1):
                    if self._feedback_camera_clear():
                        self._read_raw(flush=0)
                        self._v60_startup_camera_clear_handled = True
                        print(
                            f"[CAMERA-CLEAR-V84] STARTUP verified with adaptive polling; "
                            f"fixed imaging standby wait skipped, startup return speed={float(self._v94_startup_standby_speed):.2f}"
                        )
                        return True
                    print(f"[CAMERA-CLEAR-V84] verify retry {attempt}/{attempts}")
                    if attempt < attempts:
                        time.sleep(float(self._v94_startup_retry_poll_s))
                self._v60_startup_camera_clear_handled = True
                return False

            self._v60_apply_arm1_standby_offset_once()
            return super()._ensure_camera_clear(reason, allow_move=allow_move)

        def _v100_feedback_gripper_angle(self, payload):
            keys = {"hand", "gripper", "gripper_angle", "hand_angle", "hand_rad", "gripper_rad"}
            def walk(obj):
                if isinstance(obj, dict):
                    for key, value in obj.items():
                        if str(key).lower() in keys:
                            try:
                                number = float(value)
                                if math.isfinite(number) and -0.2 <= number <= 3.8:
                                    return number
                            except Exception:
                                pass
                    for value in obj.values():
                        found = walk(value)
                        if found is not None:
                            return found
                elif isinstance(obj, (list, tuple)):
                    for value in obj:
                        found = walk(value)
                        if found is not None:
                            return found
                return None
            return walk(payload)

        def _v100_align_arm2_standby_gripper_ramp(self, arm, cfg):
            try:
                standby_pose = self.align.standby_roarm_pose(cfg, "arm2")
                target = float(standby_pose[3]) if len(standby_pose) >= 4 else 1.356039
            except Exception as exc:
                target = 1.356039
                print(f"[ALIGN-V101-ARM2-GRIP-RAMP] standby tool target fallback={target:.3f}: {exc!r}")
            current = None
            try:
                fb = self.align._d31_query_feedback(arm, timeout_s=0.75)
                current = self._v100_feedback_gripper_angle(fb)
            except Exception as exc:
                print(f"[ALIGN-V101-ARM2-GRIP-RAMP] feedback unavailable: {exc!r}")
            if current is None:
                current = 3.05
                print("[ALIGN-V101-ARM2-GRIP-RAMP] gripper feedback unavailable; use ALIGN closed-state 3.05rad")
            if current <= target + 0.05:
                print(f"[ALIGN-V101-ARM2-GRIP-RAMP] already open enough current={current:.3f} target={target:.3f}")
                return
            max_step = 0.10
            wait_s = 0.18
            values = []
            value = float(current)
            while value - target > max_step + 1e-9:
                value -= max_step
                values.append(float(value))
            if not values or abs(values[-1] - target) > 1e-9:
                values.append(float(target))
            print(f"[ALIGN-V101-ARM2-GRIP-RAMP] current={current:.3f} -> standby={target:.3f} steps={len(values)}")
            for index, value in enumerate(values):
                arm.set_gripper(
                    float(value), delay=0.0,
                    stage="ALIGN_V101_ARM2_STANDBY_GRIP_RAMP",
                    caller="_v100_align_arm2_standby_gripper_ramp",
                )
                if index + 1 < len(values):
                    time.sleep(wait_s)
            time.sleep(0.05)

        def _v60_align_return_standby_verified(self, arm_key, arms, cfg, move_command, reason):
            arm_key = str(arm_key).lower()
            arm = arms.get(arm_key) if isinstance(arms, dict) else None
            if arm is None:
                print(f"[ALIGN-V60-STANDBY] {arm_key.upper()} missing arm")
                return False
            try:
                target = np.asarray(self.align.standby_roarm_pose(cfg, arm_key)[:3], np.float64)
            except Exception as exc:
                print(f"[ALIGN-V60-STANDBY] {arm_key.upper()} target unavailable: {exc!r}")
                return False

            xyz_tol = float(getattr(self.args, "standby_xyz_tolerance_mm", 45.0))
            z_tol = float(getattr(self.args, "standby_z_tolerance_mm", 35.0))
            attempts = max(1, int(self._v60_align_standby_attempts))
            poll_timeout = max(0.8, float(self._v60_align_standby_poll_timeout_s))

            if arm_key == "arm2":
                try:
                    self._v100_align_arm2_standby_gripper_ramp(arm, cfg)
                except Exception as exc:
                    print(f"[ALIGN-V101-ARM2-GRIP-RAMP-WARN] {type(exc).__name__}: {exc}")

            for attempt in range(1, attempts + 1):
                command_cfg = cfg
                # If ARM1 does not clear at the normal offset standby, re-command the
                # same x/y/t at the old higher startup Z. Verification still uses the
                # normal offset target, so this is only an upward camera-clear recovery.
                if arm_key == "arm1" and attempt >= 2 and self._v60_arm1_standby_offset_mm < 0.0:
                    try:
                        command_cfg = copy.copy(cfg)
                        recovery_z = float(target[2] - self._v60_arm1_standby_offset_mm)
                        setattr(command_cfg, "arm1_standby_roarm_z", recovery_z)
                        print(f"[ALIGN-V60-STANDBY] ARM1 upward recovery target z={recovery_z:.1f}mm on attempt={attempt}/{attempts}")
                    except Exception:
                        command_cfg = cfg
                try:
                    sent = self.align.move_arm_to_standby(
                        arm_key, arms, command_cfg, move_command=move_command,
                        reason=f"{reason}; verified attempt {attempt}/{attempts}",
                    )
                except Exception as exc:
                    sent = False
                    print(f"[ALIGN-V60-STANDBY] {arm_key.upper()} send attempt={attempt}/{attempts} error={exc!r}")
                if sent is False:
                    print(f"[ALIGN-V60-STANDBY] {arm_key.upper()} send attempt={attempt}/{attempts} returned False")

                deadline = time.monotonic() + poll_timeout
                sample = 0
                while time.monotonic() < deadline:
                    sample += 1
                    time.sleep(float(self._v84_standby_retry_poll_s))
                    try:
                        fb = self.align._d31_query_feedback(arm, timeout_s=0.75)
                        xyz = self.align._d31_feedback_xyz(fb)
                    except Exception as exc:
                        print(f"[ALIGN-V60-STANDBY] {arm_key.upper()} feedback error={exc!r}")
                        xyz = None
                    if xyz is None:
                        continue
                    xyz = np.asarray(xyz, np.float64).reshape(3)
                    err = float(np.linalg.norm(xyz - target))
                    z_ok = bool(float(xyz[2]) >= float(target[2]) - z_tol)
                    ok = bool(err <= xyz_tol and z_ok)
                    print(
                        f"[ALIGN-V60-STANDBY] {arm_key.upper()} attempt={attempt}/{attempts} "
                        f"sample={sample} xyz=({xyz[0]:.1f},{xyz[1]:.1f},{xyz[2]:.1f}) "
                        f"verifyTarget=({target[0]:.1f},{target[1]:.1f},{target[2]:.1f}) "
                        f"error={err:.1f}mm zOK={z_ok} verified={ok}"
                    )
                    if ok:
                        return True
                print(f"[ALIGN-V60-STANDBY] {arm_key.upper()} retrying standby return")

            print(f"[ALIGN-V60-STANDBY-FAILED] {arm_key.upper()} not verified after {attempts} attempts")
            return False

        def _gentle_arms(self, arms, module, approach_min, release_min, prefix):
            is_d60 = str(prefix).upper() == "D60"
            effective_approach = 1.35 if is_d60 else approach_min
            wrapped = super()._gentle_arms(arms, module, effective_approach, release_min, prefix)
            if not is_d60:
                return wrapped
            out = {}
            for key, proxy in wrapped.items():
                real_arm = arms.get(key) if isinstance(arms, dict) else None
                if proxy is None or real_arm is None:
                    out[key] = proxy
                else:
                    out[key] = _D60FeedbackProxy(proxy, self, module, real_arm, key)
            return out

        def _v50_d60_shallow_limits(self):
            return {
                "body_min_mm": 12.0,
                "body_max_mm": float(self._v50_d60_shallow_body_max_mm),
                "tangent_max_mm": 45.0,
                "radius_max_mm": 80.0,
            }

        def _v50_d60_reach_limits(self):
            return {
                "body_min_mm": 12.0,
                "body_max_mm": float(self._v50_d60_shallow_body_max_mm),
                "tangent_max_mm": float(self._v50_d60_reach_tangent_max_mm),
                "radius_max_mm": float(self._v50_d60_reach_radius_max_mm),
            }

        def _v33_d60_rescue_limits(self):
            return {
                "body_min_mm": 8.0,
                "body_max_mm": 82.0,
                "tangent_max_mm": 65.0,
                "radius_max_mm": 110.0,
            }

        def _v67_d60_fast_limits(self):
            return {
                "body_min_mm": float(self._v67_waist_body_min_mm),
                "body_max_mm": float(self._v67_waist_body_max_mm),
                "tangent_max_mm": float(self._v67_waist_tangent_max_mm),
                "radius_max_mm": float(self._v67_waist_endpoint_radius_max_mm),
            }

        def _v71_d60_safe_search_limits(self):
            return [
                {
                    "body_min_mm": float(body_min),
                    "body_max_mm": float(body_max),
                    "tangent_max_mm": float(tangent_max),
                    "radius_max_mm": float(radius_max),
                }
                for body_min, body_max, tangent_max, radius_max in self._v71_waist_search_tiers
            ]

        def _v50_d60_force_with_limits(self, obs, source_plan, limits, tag, overlap_mm=None):
            with self._v33_d60_rescue_lock:
                old = (
                    float(self.front.d60_waist_grip_body_min_mm),
                    float(self.front.d60_waist_grip_body_max_mm),
                    float(self.front.d60_waist_grip_tangent_max_mm),
                    float(self.front.d60_waist_grip_endpoint_radius_max_mm),
                    float(self.cfg56.center_dead_half_width),
                )
                try:
                    self.front.d60_waist_grip_body_min_mm = float(limits["body_min_mm"])
                    self.front.d60_waist_grip_body_max_mm = float(limits["body_max_mm"])
                    self.front.d60_waist_grip_tangent_max_mm = float(limits["tangent_max_mm"])
                    self.front.d60_waist_grip_endpoint_radius_max_mm = float(limits["radius_max_mm"])
                    if overlap_mm is not None:
                        self.cfg56.center_dead_half_width = -abs(float(overlap_mm)) if float(overlap_mm) > 0.0 else 0.0
                    plan, motion, report = super()._d60_force_waist_endpoint_plan(obs, source_plan)
                finally:
                    self.front.d60_waist_grip_body_min_mm = old[0]
                    self.front.d60_waist_grip_body_max_mm = old[1]
                    self.front.d60_waist_grip_tangent_max_mm = old[2]
                    self.front.d60_waist_grip_endpoint_radius_max_mm = old[3]
                    self.cfg56.center_dead_half_width = old[4]
            report = dict(report or {})
            report[f"bottom_vla_{tag}"] = True
            report[f"bottom_vla_{tag}_limits"] = dict(limits)
            report[f"bottom_vla_{tag}_overlap_mm"] = None if overlap_mm is None else float(overlap_mm)
            if plan is not None and motion is not None:
                metrics = dict(getattr(plan, "metrics", {}) or {})
                metrics[f"bottom_vla_{tag}"] = True
                metrics[f"bottom_vla_{tag}_limits"] = dict(limits)
                metrics[f"bottom_vla_{tag}_overlap_mm"] = None if overlap_mm is None else float(overlap_mm)
                plan.metrics = metrics
                print(
                    f"[D60-V50-PLAN] mode={tag} body={float(limits['body_min_mm']):.0f}..{float(limits['body_max_mm']):.0f}mm "
                    f"tangent<={float(limits['tangent_max_mm']):.0f} radius<={float(limits['radius_max_mm']):.0f} "
                    f"deadZone={'PRESERVE' if overlap_mm is None else f'OVERLAP_{float(overlap_mm):.0f}mm'} pairTrials={int(report.get('pair_trials', 0) or 0)}"
                )
            return plan, motion, report

        def _v33_d60_force_rescue_once(self, obs, source_plan, overlap_mm):
            return self._v50_d60_force_with_limits(
                obs, source_plan, self._v33_d60_rescue_limits(), "v33_d60_rescue", overlap_mm
            )

        def _v53_board_contour(self, obs):
            key = id(obs)
            if key == self._v67_contour_cache_id and self._v67_contour_cache_value is not None:
                px, board = self._v67_contour_cache_value
                return px, board
            mask_obj = getattr(obs, "mask", None)
            contour = getattr(mask_obj, "contour", None) if mask_obj is not None else None
            if contour is None:
                self._v67_contour_cache_id = key
                self._v67_contour_cache_value = (None, None)
                return None, None
            try:
                px = np.asarray(contour, np.float64).reshape(-1, 2)
                if len(px) < 20:
                    value = (None, None)
                else:
                    Hm = np.asarray(self.H_raw, np.float64).reshape(3, 3)
                    homo = np.column_stack((px[:, 0], px[:, 1], np.ones(len(px), np.float64)))
                    out = homo @ Hm.T
                    good = np.isfinite(out).all(axis=1) & (np.abs(out[:, 2]) > 1e-8)
                    if int(np.count_nonzero(good)) < 20:
                        value = (None, None)
                    else:
                        value = (px[good], out[good, :2] / out[good, 2:3])
            except Exception:
                value = (None, None)
            self._v67_contour_cache_id = key
            self._v67_contour_cache_value = value
            self._v67_mask_report_cache_id = key
            self._v67_mask_report_cache = {}
            return value

        def _v53_mask_point_report(self, obs, point_board):
            p = self._finite_point2(point_board)
            if p is None:
                return {"ok": False, "reason": "POINT_UNAVAILABLE", "edge_depth_mm": None}
            obs_id = id(obs)
            if self._v67_mask_report_cache_id != obs_id:
                self._v67_mask_report_cache_id = obs_id
                self._v67_mask_report_cache = {}
            cache_key = (round(float(p[0]), 2), round(float(p[1]), 2))
            cached = self._v67_mask_report_cache.get(cache_key)
            if cached is not None:
                return copy.deepcopy(cached)
            contour_px, contour_board = self._v53_board_contour(obs)
            if contour_px is None or contour_board is None:
                result = {"ok": False, "reason": "MASK_CONTOUR_UNAVAILABLE", "edge_depth_mm": None}
                self._v67_mask_report_cache[cache_key] = result
                return copy.deepcopy(result)
            a = np.asarray(contour_board, np.float64)
            b = np.roll(a, -1, axis=0)
            ab = b - a
            denom = np.sum(ab * ab, axis=1)
            t = np.zeros(len(a), np.float64)
            valid = denom > 1e-9
            t[valid] = np.clip(np.sum((p - a[valid]) * ab[valid], axis=1) / denom[valid], 0.0, 1.0)
            q = a + ab * t[:, None]
            edge_depth = float(np.min(np.linalg.norm(q - p[None, :], axis=1)))
            pixel = None
            try:
                pixel = self.d56.board_to_pixel(self.H_raw, float(p[0]), float(p[1]))
            except Exception:
                try:
                    inv = np.linalg.inv(np.asarray(self.H_raw, np.float64).reshape(3, 3))
                    hp = inv @ np.asarray([float(p[0]), float(p[1]), 1.0], np.float64)
                    if abs(float(hp[2])) > 1e-8:
                        pixel = hp[:2] / hp[2]
                except Exception:
                    pixel = None
            if pixel is None:
                result = {"ok": False, "reason": "BOARD_TO_PIXEL_FAILED", "edge_depth_mm": edge_depth}
            else:
                u, v = float(pixel[0]), float(pixel[1])
                try:
                    inside = float(self.cv2.pointPolygonTest(np.asarray(contour_px, np.float32).reshape(-1, 1, 2), (u, v), False)) >= 0.0
                except Exception:
                    inside = False
                if not inside:
                    result = {"ok": False, "reason": "OUTSIDE_REAL_GARMENT_MASK", "edge_depth_mm": edge_depth, "pixel": [u, v]}
                else:
                    result = {"ok": True, "reason": "OK", "edge_depth_mm": edge_depth, "pixel": [u, v]}
            self._v67_mask_report_cache[cache_key] = copy.deepcopy(result)
            return result

        def _v54_waist_corner_refinement(self, obs):
            result = {"ok": False, "reason": "UNAVAILABLE", "v55_outer_edge": True}
            pts = self._pose_board_points(obs)
            wl = self._finite_point2(pts.get("waist_img_left"))
            wc = self._finite_point2(pts.get("waist_center"))
            wr = self._finite_point2(pts.get("waist_img_right"))
            if wl is None or wr is None:
                result["reason"] = "POSE_WAIST_ENDPOINTS_UNAVAILABLE"
                return result
            if wc is None:
                wc = 0.5 * (wl + wr)
            crotch = self._finite_point2(pts.get("crotch"))
            lower = crotch if crotch is not None else self._finite_point2(pts.get("lower_center"))
            if lower is None:
                result["reason"] = "BODY_DIRECTION_UNAVAILABLE"
                return result

            pose_waist_vec = np.asarray(wr - wl, np.float64)
            pose_waist_width = float(np.linalg.norm(pose_waist_vec))
            if not math.isfinite(pose_waist_width) or pose_waist_width < 70.0:
                result["reason"] = "POSE_WAIST_WIDTH_INVALID"
                return result
            if pose_waist_width > float(self._v61_waist_outer_pose_max_width_mm):
                result.update({
                    "reason": "V61_POSE_WAIST_TOO_WIDE_FOR_OUTER_REFINE",
                    "pose_width_mm": float(pose_waist_width),
                })
                print(
                    f"[D60-V61-WAIST-GATE-REJECT] raw Pose waist width={pose_waist_width:.1f}mm "
                    f"> {float(self._v61_waist_outer_pose_max_width_mm):.1f}mm; do not let a spread leg become the waistband"
                )
                return result

            # Pose decides only which end of the garment is the waist.  The real waist line
            # is taken from the outer Mask edge so a tilted/misplaced Pose waist cannot
            # drag the planner down the trouser body.
            body_raw = np.asarray(lower - wc, np.float64)
            body_len = float(np.linalg.norm(body_raw))
            if body_len < 20.0:
                result["reason"] = "BODY_AXIS_DEGENERATE"
                return result
            body_u = body_raw / body_len
            waist_u = np.asarray([body_u[1], -body_u[0]], np.float64)
            if float(np.dot(waist_u, pose_waist_vec)) < 0.0:
                waist_u = -waist_u

            _, contour = self._v53_board_contour(obs)
            if contour is None or len(contour) < 30:
                result["reason"] = "REAL_MASK_CONTOUR_UNAVAILABLE"
                return result
            c = np.asarray(contour, np.float64).reshape(-1, 2)
            rel = c - wc[None, :]
            axial = rel @ body_u
            lateral = rel @ waist_u

            side_limit = float(np.clip(0.78 * pose_waist_width + 45.0, 125.0, 360.0))
            semantic = np.abs(lateral) <= side_limit
            if int(np.count_nonzero(semantic)) < 30:
                result["reason"] = "WAIST_SIDE_CONTOUR_TOO_SPARSE"
                return result

            lat_sem = lateral[semantic]
            lo = float(np.percentile(lat_sem, 4.0))
            hi = float(np.percentile(lat_sem, 96.0))
            if hi - lo < 90.0:
                result["reason"] = "WAIST_OUTER_SPAN_TOO_SMALL"
                return result

            # Build the waist-side outer envelope: in each lateral slice keep the most
            # waistward contour point.  This explicitly follows the garment exterior and
            # cannot jump to the hem because body_u fixes the waist polarity first.
            envelope_idx = []
            edges = np.linspace(lo, hi, 23)
            for a, b in zip(edges[:-1], edges[1:]):
                idx = np.flatnonzero(semantic & (lateral >= a) & (lateral < b))
                if len(idx) == 0:
                    continue
                j = int(idx[int(np.argmin(axial[idx]))])
                envelope_idx.append(j)
            envelope_idx = np.asarray(sorted(set(envelope_idx)), np.int32)
            if len(envelope_idx) < 8:
                result["reason"] = "WAIST_OUTER_ENVELOPE_TOO_SPARSE"
                return result
            env = c[envelope_idx]

            def fit_line(points):
                center = np.mean(points, axis=0)
                q = points - center[None, :]
                _, _, vh = np.linalg.svd(q, full_matrices=False)
                tangent = np.asarray(vh[0], np.float64)
                tangent /= max(float(np.linalg.norm(tangent)), 1e-9)
                if float(np.dot(tangent, waist_u)) < 0.0:
                    tangent = -tangent
                normal = np.asarray([-tangent[1], tangent[0]], np.float64)
                return center, tangent, normal

            line_c, line_u, line_n = fit_line(env)
            residual = np.abs((env - line_c[None, :]) @ line_n)
            med = float(np.median(residual))
            mad = float(np.median(np.abs(residual - med)))
            keep_thr = max(12.0, med + 2.8 * max(mad, 1.0))
            keep = residual <= keep_thr
            if int(np.count_nonzero(keep)) >= 6:
                env = env[keep]
                envelope_idx = envelope_idx[keep]
                line_c, line_u, line_n = fit_line(env)
                residual = np.abs((env - line_c[None, :]) @ line_n)

            along = (env - line_c[None, :]) @ line_u
            order = np.argsort(along)
            env = env[order]
            along = along[order]
            if len(env) < 6:
                result["reason"] = "WAIST_OUTER_LINE_TOO_SPARSE"
                return result
            li = int(np.clip(round(0.06 * (len(env) - 1)), 0, len(env) - 1))
            ri = int(np.clip(round(0.94 * (len(env) - 1)), 0, len(env) - 1))
            refined_left = np.asarray(env[li], np.float64)
            refined_right = np.asarray(env[ri], np.float64)
            if float(np.dot(refined_right - refined_left, waist_u)) < 0.0:
                refined_left, refined_right = refined_right, refined_left

            refined_width = float(np.linalg.norm(refined_right - refined_left))
            min_width = max(100.0, 0.42 * pose_waist_width)
            max_width = min(560.0, 1.70 * pose_waist_width)
            if refined_width < min_width or refined_width > max_width:
                result.update({"reason": "OUTER_WAIST_WIDTH_IMPLAUSIBLE", "refined_width_mm": refined_width})
                return result
            if refined_width > float(self._v61_waist_outer_max_width_mm):
                result.update({
                    "reason": "V61_OUTER_WAIST_TOO_WIDE",
                    "refined_width_mm": float(refined_width),
                })
                print(
                    f"[D60-V61-WAIST-GATE-REJECT] refined waist width={refined_width:.1f}mm "
                    f"> {float(self._v61_waist_outer_max_width_mm):.1f}mm; reject side/leg-edge waistband hypothesis"
                )
                return result

            refined_vec_gate = np.asarray(refined_right - refined_left, np.float64)
            refined_u_gate = refined_vec_gate / max(refined_width, 1e-9)
            parallel_body = float(np.clip(abs(float(np.dot(refined_u_gate, body_u))), 0.0, 1.0))
            perp_error_deg = float(np.degrees(np.arcsin(parallel_body)))
            if perp_error_deg > float(self._v61_waist_outer_max_perp_error_deg):
                result.update({
                    "reason": "V61_OUTER_WAIST_NOT_PERPENDICULAR_TO_BODY",
                    "refined_width_mm": float(refined_width),
                    "perp_error_deg": float(perp_error_deg),
                })
                print(
                    f"[D60-V61-WAIST-GATE-REJECT] waist/body perpendicular error={perp_error_deg:.1f}deg "
                    f"> {float(self._v61_waist_outer_max_perp_error_deg):.1f}deg; reject diagonal leg-edge hypothesis"
                )
                return result

            center_idx = int(np.clip(round(0.50 * (len(env) - 1)), 0, len(env) - 1))
            refined_center = np.asarray(env[center_idx], np.float64)
            shift_vec = refined_center - wc
            body_shift = float(np.dot(shift_vec, body_u))
            tangent_shift = float(np.dot(shift_vec, waist_u))
            if body_shift > float(self._v72_waist_outer_max_bodyward_shift_mm):
                result.update({"reason": "OUTER_WAIST_WOULD_MOVE_DOWN_BODY", "body_shift_mm": body_shift})
                return result
            if abs(tangent_shift) > 120.0:
                result.update({"reason": "OUTER_WAIST_TANGENT_DRIFT", "tangent_shift_mm": tangent_shift})
                return result

            edge_rms = float(np.sqrt(np.mean(np.square(residual)))) if len(residual) else 999.0
            if edge_rms > float(self._v61_waist_outer_max_fit_rms_mm):
                result.update({
                    "reason": "V61_OUTER_WAIST_FIT_TOO_ROUGH",
                    "refined_width_mm": float(refined_width),
                    "edge_fit_rms_mm": float(edge_rms),
                })
                print(
                    f"[D60-V61-WAIST-GATE-REJECT] outer waist fitRMS={edge_rms:.1f}mm "
                    f"> {float(self._v61_waist_outer_max_fit_rms_mm):.1f}mm; reject mixed waist/leg contour"
                )
                return result
            refined_vec = refined_right - refined_left
            refined_angle_deg = float(np.degrees(np.arctan2(refined_vec[1], refined_vec[0])))
            result.update({
                "ok": True,
                "reason": "V55_REAL_MASK_WAIST_OUTER_EDGE",
                "waist_left": refined_left,
                "waist_center": refined_center,
                "waist_right": refined_right,
                "pose_waist_left": wl,
                "pose_waist_center": wc,
                "pose_waist_right": wr,
                "waist_u": line_u,
                "body_u": body_u,
                "pose_width_mm": pose_waist_width,
                "refined_width_mm": refined_width,
                "body_shift_mm": body_shift,
                "tangent_shift_mm": tangent_shift,
                "edge_fit_rms_mm": edge_rms,
                "refined_angle_deg": refined_angle_deg,
                "envelope_points": int(len(env)),
                "left_corner": {"point": refined_left.copy()},
                "right_corner": {"point": refined_right.copy()},
            })
            print(
                f"[D60-V55-WAIST-OUTER] poseWidth={pose_waist_width:.1f}mm -> outerWidth={refined_width:.1f}mm "
                f"angle={refined_angle_deg:+.1f}deg bodyShift={body_shift:+.1f}mm fitRMS={edge_rms:.1f}mm n={len(env)}"
            )
            return result

        def _v54_obs_with_refined_waist_uncached(self, obs):
            ref = self._v54_waist_corner_refinement(obs)
            if not bool(ref.get("ok", False)):
                print(f"[D60-V55-WAIST-OUTER-SKIP] reason={ref.get('reason','-')}; keep existing Pose waist frame")
                return obs, ref
            try:
                obs2 = copy.copy(obs)
                pose = copy.copy(getattr(obs, "pose"))
                kb = dict(getattr(pose, "keypoints_board", {}) or {})
                kp = dict(getattr(pose, "keypoints_px", {}) or {})
                mapping = {
                    "waist_img_left": np.asarray(ref["waist_left"], np.float64),
                    "waist_center": np.asarray(ref["waist_center"], np.float64),
                    "waist_img_right": np.asarray(ref["waist_right"], np.float64),
                }
                for name, point in mapping.items():
                    kb[name] = (float(point[0]), float(point[1]))
                    try:
                        px = self.d56.board_to_pixel(self.H_raw, float(point[0]), float(point[1]))
                        if px is not None:
                            kp[name] = (float(px[0]), float(px[1]))
                    except Exception:
                        pass
                pose.keypoints_board = kb
                if kp:
                    pose.keypoints_px = kp
                pose.waist_left = np.asarray(mapping["waist_img_left"], np.float32)
                pose.waist_center = np.asarray(mapping["waist_center"], np.float32)
                pose.waist_right = np.asarray(mapping["waist_img_right"], np.float32)
                if hasattr(pose, "waist_width_mm"):
                    pose.waist_width_mm = float(ref["refined_width_mm"])
                if hasattr(pose, "waist_angle_deg"):
                    dv = mapping["waist_img_right"] - mapping["waist_img_left"]
                    pose.waist_angle_deg = float(math.degrees(math.atan2(float(dv[1]), float(dv[0]))))
                obs2.pose = pose
                setattr(obs2, "bottom_vla_v54_waist_corner_refinement", copy.deepcopy(ref))
                print(
                    f"[D60-V55-WAIST-OUTER-USE] poseWidth={float(ref['pose_width_mm']):.1f}mm "
                    f"outerWidth={float(ref['refined_width_mm']):.1f}mm angle={float(ref.get('refined_angle_deg', 0.0)):+.1f}deg "
                    f"shiftBody={float(ref['body_shift_mm']):+.1f}mm fitRMS={float(ref.get('edge_fit_rms_mm', 999.0)):.1f}mm; "
                    "waist line follows the real outer Mask edge"
                )
                return obs2, ref
            except Exception as exc:
                ref = dict(ref)
                ref.update({"ok": False, "reason": f"REFINED_OBS_BUILD_FAILED:{exc!r}"})
                print(f"[D60-V55-WAIST-OUTER-SKIP] {ref['reason']}")
                return obs, ref

        def _v54_obs_with_refined_waist(self, obs):
            key = id(obs)
            if key == self._v59_refined_obs_cache_id and self._v59_refined_obs_cache_value is not None:
                return self._v59_refined_obs_cache_value
            value = self._v54_obs_with_refined_waist_uncached(obs)
            self._v59_refined_obs_cache_id = key
            self._v59_refined_obs_cache_value = value
            return value

        def _v53_pose_waist_mask_consistency(self, obs):
            pts = self._pose_board_points(obs)
            names = ("waist_img_left", "waist_center", "waist_img_right")
            waist = [self._finite_point2(pts.get(name)) for name in names]
            if any(p is None for p in waist):
                return {"ok": False, "reason": "WAIST_3PTS_UNAVAILABLE"}
            reports = [self._v53_mask_point_report(obs, p) for p in waist]
            near = sum(1 for r in reports if r.get("edge_depth_mm") is not None and float(r["edge_depth_mm"]) <= float(self._v53_waist_pose_edge_max_mm))
            if near < 2:
                return {"ok": False, "reason": "WAIST_NOT_NEAR_REAL_MASK_EDGE", "points": reports, "near_count": near}
            crotch = self._finite_point2(pts.get("crotch"))
            center = waist[1]
            if crotch is not None:
                body = crotch - center
                norm = float(np.linalg.norm(body))
                if norm > 1e-6:
                    body = body / norm
                    probe = center + body * 25.0
                    probe_report = self._v53_mask_point_report(obs, probe)
                    if not bool(probe_report.get("ok", False)):
                        return {"ok": False, "reason": "WAIST_BODY_DIRECTION_DOES_NOT_ENTER_MASK", "points": reports, "body_probe": probe_report}
            return {"ok": True, "reason": "OK", "points": reports, "near_count": near}

        def _d60_pose_waist_frame(self, obs):
            key = id(obs)
            if key == self._v67_pose_waist_cache_id and self._v67_pose_waist_cache_value is not None:
                return copy.deepcopy(self._v67_pose_waist_cache_value)

            def finish(frame):
                self._v67_pose_waist_cache_id = key
                self._v67_pose_waist_cache_value = copy.deepcopy(frame)
                return frame

            planning_obs, refinement = self._v54_obs_with_refined_waist(obs)
            frame = dict(super()._d60_pose_waist_frame(planning_obs) or {})
            frame["v54_waist_corner_refinement"] = copy.deepcopy(refinement)
            refinement_reason = str((refinement or {}).get("reason", ""))
            if (not bool((refinement or {}).get("ok", False))) and refinement_reason.startswith(("V61_", "V62_", "V63_", "V66_", "V67_")):
                frame["ok"] = False
                frame["reason"] = "V67_WAIST_FINAL_VALIDATION_REJECTED:" + refinement_reason
                print(f"[D60-V67-WAIST-GATE-BLOCK] {refinement_reason}")
                return finish(frame)
            if not bool(frame.get("ok", False)):
                return finish(frame)
            check = self._v53_pose_waist_mask_consistency(planning_obs)
            frame["v53_mask_waist_consistency"] = copy.deepcopy(check)
            if not bool(check.get("ok", False)):
                frame["ok"] = False
                frame["reason"] = "V67_" + str(check.get("reason", "WAIST_MASK_MISMATCH"))
                print(f"[D60-V67-POSE-REJECT] {frame['reason']}")
            else:
                if bool((refinement or {}).get("ok", False)):
                    frame["waist_geometry_source"] = "V72_OUTER_MASK_WAIST_REFINED"
                    print(
                        f"[D60-V72-WAIST-OUTER-USED] source=V72_OUTER_MASK_WAIST_REFINED "
                        f"edgeSupport={int(check.get('near_count', 0))}/3; Pose selects the waist end and the real Mask envelope supplies left/center/right waist geometry"
                    )
                else:
                    frame["waist_geometry_source"] = "POSE_WAIST_VALIDATED_BY_OUTER"
                    print(
                        f"[D60-V67-POSE-PASS] source=POSE_WAIST_VALIDATED_BY_OUTER "
                        f"edgeSupport={int(check.get('near_count', 0))}/3; outer refinement unavailable, keep validated Pose waist"
                    )
            return finish(frame)

        def _d60_plan_from_obs(self, obs):
            planning_obs, refinement = self._v54_obs_with_refined_waist(obs)
            refinement_reason = str((refinement or {}).get("reason", ""))
            plan = super()._d60_plan_from_obs(planning_obs)
            if (not bool((refinement or {}).get("ok", False))) and refinement_reason.startswith(("V61_", "V62_", "V63_", "V66_")):
                print(f"[D60-V66-WAIST-PLAN-BLOCK] final waist validation rejected: {refinement_reason}; no forced endpoint fallback")
                return plan
            if plan is not None:
                try:
                    metrics = dict(getattr(plan, "metrics", {}) or {})
                    metrics["bottom_vla_v54_waist_corner_refinement"] = copy.deepcopy(refinement)
                    plan.metrics = metrics
                except Exception:
                    pass
            if plan is not None and bool(getattr(plan, "ok", False)):
                if not bool(getattr(self, "_v53_align_capture_active", False)):
                    biased_plan, bias_report = self._v73_apply_waist_grasp_bias(planning_obs, plan)
                    if bool((bias_report or {}).get("ok", False)):
                        plan = biased_plan
                return plan
            if not bool(getattr(self, "_v53_align_capture_active", False)):
                return plan
            forced_plan, forced_motion, forced_report = self._d60_force_waist_endpoint_plan(planning_obs, plan)
            if forced_plan is None or forced_motion is None or not bool(getattr(forced_plan, "ok", False)):
                print(
                    "[ALIGN-V54-DUAL-PAIR-BLOCK] current refined-waist source and endpoint fallback "
                    "could not make a safe dual pair"
                )
                return plan
            try:
                metrics = dict(getattr(forced_plan, "metrics", {}) or {})
                metrics.update({
                    "bottom_vla_v54_align_dual_pair_recovered": True,
                    "bottom_vla_v54_align_dual_pair_report": copy.deepcopy(forced_report or {}),
                    "bottom_vla_v54_waist_corner_refinement": copy.deepcopy(refinement),
                })
                forced_plan.metrics = metrics
            except Exception:
                pass
            print(
                "[ALIGN-V54-DUAL-PAIR] D60 source unavailable; recovered a safe current-waist "
                "dual pair for ALIGN stage-1 LEVEL+UP"
            )
            return forced_plan

        def _v53_validate_waist_safe_pair(self, obs, plan, label):
            required = float(self._v53_waist_hard_inset_mm)
            tolerance = float(self._v53_waist_inset_tolerance_mm)
            points = plan.get("arm_points", {}) if isinstance(plan, dict) else getattr(plan, "arm_points", {})
            reports = {}
            ok = True
            for arm in ("arm2", "arm1"):
                item = points.get(arm, {}) if isinstance(points, dict) else {}
                grip = None
                if isinstance(item, dict):
                    for key in ("grip_board", "source_board", "grasp_board"):
                        grip = self._finite_point2(item.get(key))
                        if grip is not None:
                            break
                if grip is None:
                    reports[arm] = {"ok": False, "reason": "GRIP_BOARD_UNAVAILABLE", "edge_depth_mm": None}
                    ok = False
                    continue
                rep = self._v53_mask_point_report(obs, grip)
                depth = rep.get("edge_depth_mm")
                pass_depth = bool(rep.get("ok", False) and depth is not None and float(depth) + tolerance >= required)
                rep = dict(rep)
                rep.update({"grip_board": [float(grip[0]), float(grip[1])], "required_inset_mm": required, "tolerance_mm": tolerance, "pass_inset": pass_depth})
                reports[arm] = rep
                ok = ok and pass_depth
            a2 = reports.get("arm2", {}).get("edge_depth_mm")
            a1 = reports.get("arm1", {}).get("edge_depth_mm")
            print(
                f"[D60-V53-INSET] mode={label} A2={'NA' if a2 is None else f'{float(a2):.1f}'}mm "
                f"A1={'NA' if a1 is None else f'{float(a1):.1f}'}mm required>={required:.1f}mm "
                f"tol={tolerance:.1f} -> {'PASS' if ok else 'REJECT'}"
            )
            if plan is not None:
                try:
                    metrics = dict(getattr(plan, "metrics", {}) or {})
                    metrics["bottom_vla_v53_waist_hard_inset_mm"] = required
                    metrics["bottom_vla_v53_waist_inset_report"] = copy.deepcopy(reports)
                    metrics["bottom_vla_v53_waist_inset_ok"] = bool(ok)
                    plan.metrics = metrics
                except Exception:
                    pass
            return bool(ok), {"ok": bool(ok), "reason": "OK" if ok else "V53_GRIP_OUTSIDE_35MM_GARMENT_INSET", "arms": reports, "required_inset_mm": required, "tolerance_mm": tolerance}

        def _v72_waist_balance_report(self, inset_report, tier):
            arms = dict((inset_report or {}).get("arms", {}) or {})
            a2 = (arms.get("arm2", {}) or {}).get("edge_depth_mm")
            a1 = (arms.get("arm1", {}) or {}).get("edge_depth_mm")
            if a2 is None or a1 is None:
                return {"ok": False, "score": float("inf"), "preferred": False, "a2_mm": a2, "a1_mm": a1}
            a2 = float(a2)
            a1 = float(a1)
            diff = abs(a1 - a2)
            target = float(self._v72_waist_preferred_inset_mm)
            deviation = abs(a1 - target) + abs(a2 - target)
            near_edge = max(0.0, float(self._v72_waist_fast_min_inset_mm) - a1) + max(0.0, float(self._v72_waist_fast_min_inset_mm) - a2)
            too_deep = max(0.0, a1 - 65.0) + max(0.0, a2 - 65.0)
            score = 1.8 * diff + 0.55 * deviation + 2.0 * near_edge + 1.1 * too_deep + 3.0 * max(0, int(tier) - 1)
            preferred = bool(
                float(self._v72_waist_fast_min_inset_mm) <= a1 <= float(self._v72_waist_fast_max_inset_mm)
                and float(self._v72_waist_fast_min_inset_mm) <= a2 <= float(self._v72_waist_fast_max_inset_mm)
                and diff <= float(self._v72_waist_fast_max_balance_mm)
            )
            return {"ok": True, "score": float(score), "preferred": preferred, "a2_mm": a2, "a1_mm": a1, "difference_mm": float(diff), "target_mm": target}

        def _v73_waist_body_unit(self, obs):
            pts = self._pose_board_points(obs)
            wc = self._finite_point2(pts.get("waist_center"))
            crotch = self._finite_point2(pts.get("crotch"))
            if wc is None or crotch is None:
                return None
            v = np.asarray(crotch - wc, np.float64)
            n = float(np.linalg.norm(v))
            if n < 1e-6:
                return None
            return v / n

        def _v73_apply_waist_grasp_bias(self, obs, plan):
            if plan is None or not bool(getattr(plan, "ok", False)):
                return plan, {"ok": False, "reason": "PLAN_UNAVAILABLE"}
            if bool(getattr(self, "_v53_align_capture_active", False)):
                return plan, {"ok": True, "reason": "ALIGN_CAPTURE_UNCHANGED", "skipped": True}
            body_u = self._v73_waist_body_unit(obs)
            pts = self._pose_board_points(obs)
            wl = self._finite_point2(pts.get("waist_img_left"))
            wc = self._finite_point2(pts.get("waist_center"))
            wr = self._finite_point2(pts.get("waist_img_right"))
            if body_u is None or wl is None or wr is None:
                return plan, {"ok": False, "reason": "WAIST_DIRECTION_UNAVAILABLE"}
            if wc is None:
                wc = 0.5 * (wl + wr)
            waist_vec = np.asarray(wr - wl, np.float64)
            waist_len = float(np.linalg.norm(waist_vec))
            if waist_len < 1e-6:
                return plan, {"ok": False, "reason": "WAIST_TANGENT_UNAVAILABLE"}
            waist_u = waist_vec / waist_len
            out = copy.deepcopy(plan)
            points = getattr(out, "arm_points", None)
            if not isinstance(points, dict):
                return plan, {"ok": False, "reason": "ARM_POINTS_UNAVAILABLE"}
            required = float(self._v53_waist_hard_inset_mm)
            tolerance = float(self._v53_waist_inset_tolerance_mm)
            report = {"ok": True, "reason": "OK", "body_u": body_u.astype(float).tolist(), "waist_u": waist_u.astype(float).tolist(), "arms": {}}
            for arm in ("arm1", "arm2"):
                item = points.get(arm, {})
                if not isinstance(item, dict):
                    report["ok"] = False
                    report["arms"][arm] = {"ok": False, "reason": "ARM_ITEM_UNAVAILABLE"}
                    continue
                old = None
                for key in ("grip_board", "source_board", "grasp_board", "target_board"):
                    old = self._finite_point2(item.get(key))
                    if old is not None:
                        break
                if old is None:
                    report["ok"] = False
                    report["arms"][arm] = {"ok": False, "reason": "GRIP_BOARD_UNAVAILABLE"}
                    continue
                old_rep = self._v53_mask_point_report(obs, old)
                if arm == "arm1":
                    direction = waist_u.copy()
                    if float(np.dot(old - wc, direction)) < 0.0:
                        direction = -direction
                    requested = float(self._v73_waist_arm1_outward_mm)
                    direction_name = "OUTWARD_ALONG_WAIST_TANGENT"
                else:
                    direction = body_u.copy()
                    requested = float(self._v73_waist_arm2_inward_mm)
                    direction_name = "INWARD_TO_BODY"
                selected = old.copy()
                selected_rep = old_rep
                applied = 0.0
                for amount in np.linspace(float(requested), 0.0, int(round(float(requested))) + 1):
                    cand = old + direction * float(amount)
                    rep = self._v53_mask_point_report(obs, cand)
                    depth = rep.get("edge_depth_mm") if isinstance(rep, dict) else None
                    if bool(rep.get("ok", False)) and depth is not None and float(depth) + tolerance >= required:
                        selected = np.asarray(cand, np.float64)
                        selected_rep = rep
                        applied = float(amount)
                        break
                for key in ("grip_board", "source_board", "grasp_board", "target_board"):
                    cur = self._finite_point2(item.get(key))
                    if cur is not None and float(np.linalg.norm(cur - old)) <= 2.0:
                        item[key] = (float(selected[0]), float(selected[1]))
                try:
                    px = self.d56.board_to_pixel(self.H_raw, float(selected[0]), float(selected[1]))
                    if px is not None and "grip_px" in item:
                        item["grip_px"] = (float(px[0]), float(px[1]))
                except Exception:
                    pass
                points[arm] = item
                report["arms"][arm] = {
                    "ok": True,
                    "requested_mm": float(requested),
                    "applied_mm": float(applied),
                    "direction": direction_name,
                    "direction_unit": direction.astype(float).tolist(),
                    "before_board": old.astype(float).tolist(),
                    "after_board": selected.astype(float).tolist(),
                    "before_inset_mm": old_rep.get("edge_depth_mm") if isinstance(old_rep, dict) else None,
                    "after_inset_mm": selected_rep.get("edge_depth_mm") if isinstance(selected_rep, dict) else None,
                }
            out.arm_points = points
            metrics = dict(getattr(out, "metrics", {}) or {})
            metrics["bottom_vla_v80_waist_asym_bias"] = True
            metrics["bottom_vla_v80_waist_asym_bias_report"] = copy.deepcopy(report)
            out.metrics = metrics
            final_ok, final_rep = self._v53_validate_waist_safe_pair(obs, out, "V80_ARM_BIAS")
            report["final_inset"] = copy.deepcopy(final_rep)
            report["ok"] = bool(report.get("ok", False) and final_ok)
            if not report["ok"]:
                return plan, report
            a1 = report["arms"].get("arm1", {})
            a2 = report["arms"].get("arm2", {})
            print(
                f"[D60-V80-ARM-BIAS] ARM1 tangent-outward={float(a1.get('applied_mm',0.0)):.1f}/{float(a1.get('requested_mm',0.0)):.1f}mm "
                f"inset={a1.get('before_inset_mm')}->{a1.get('after_inset_mm')} | "
                f"ARM2 inward={float(a2.get('applied_mm',0.0)):.1f}/{float(a2.get('requested_mm',0.0)):.1f}mm "
                f"inset={a2.get('before_inset_mm')}->{a2.get('after_inset_mm')}"
            )
            return out, report

        def _v73_finalize_forced_waist_bias(self, obs, plan, motion, report):
            if plan is None or motion is None or bool(getattr(self, "_v53_align_capture_active", False)):
                return plan, motion, report
            biased, bias_report = self._v73_apply_waist_grasp_bias(obs, plan)
            if biased is plan or not bool((bias_report or {}).get("ok", False)):
                rep = dict(report or {})
                rep["bottom_vla_v73_waist_bias"] = copy.deepcopy(bias_report)
                return plan, motion, rep
            rebuilt, why = self.d56._63_step120_motion_plan_from_d56_plan(
                biased, self.config, self.cfg56, self.d56_args
            )
            if rebuilt is None:
                rep = dict(report or {})
                rep["bottom_vla_v73_waist_bias"] = copy.deepcopy(bias_report)
                rep["bottom_vla_v73_waist_bias_motion_reason"] = str(why)
                print(f"[D60-V73-ARM-BIAS-FALLBACK] biased motion rebuild failed: {why}; keep pre-bias safe pair")
                return plan, motion, rep
            rep = dict(report or {})
            rep["bottom_vla_v73_waist_bias"] = copy.deepcopy(bias_report)
            rep["bottom_vla_v73_waist_bias_motion_reason"] = str(why)
            return biased, rebuilt, rep

        def _d60_force_waist_endpoint_plan(self, obs, source_plan):
            planning_obs, refinement = self._v54_obs_with_refined_waist(obs)
            reports = []
            safe_candidates = []
            for tier, limits in enumerate(self._v71_d60_safe_search_limits(), 1):
                if self._v67_prepare_cancel_requested():
                    return None, None, {"reason": "V72_CANCELLED_BY_HUMAN_OVERRIDE", "pair_trials": 0}
                plan, motion, report = self._v50_d60_force_with_limits(
                    planning_obs, source_plan, limits, "v72_d60_balanced_safe_pair_search", None
                )
                report = dict(report or {})
                report["bottom_vla_v72_search_tier"] = int(tier)
                reports.append(copy.deepcopy(report))
                if self._v67_prepare_cancel_requested():
                    return None, None, {"reason": "V72_CANCELLED_BY_HUMAN_OVERRIDE", "pair_trials": int(report.get("pair_trials", 0) or 0)}
                if plan is None or motion is None:
                    print(
                        f"[D60-V72-SAFE-SEARCH] tier={tier} no pair "
                        f"body={limits['body_min_mm']:.0f}..{limits['body_max_mm']:.0f}mm "
                        f"tangent<={limits['tangent_max_mm']:.0f} radius<={limits['radius_max_mm']:.0f}"
                    )
                    continue
                inset_ok, inset_report = self._v53_validate_waist_safe_pair(planning_obs, plan, f"V72_SAFE_TIER_{tier}")
                report["bottom_vla_v72_waist_inset"] = copy.deepcopy(inset_report)
                metrics = dict(getattr(plan, "metrics", {}) or {})
                metrics["bottom_vla_v72_search_tier"] = int(tier)
                metrics["bottom_vla_v72_waist_inset"] = copy.deepcopy(inset_report)
                plan.metrics = metrics
                if inset_ok:
                    balance = self._v72_waist_balance_report(inset_report, tier)
                    report["bottom_vla_v72_waist_balance"] = copy.deepcopy(balance)
                    plan.metrics["bottom_vla_v72_waist_balance"] = copy.deepcopy(balance)
                    safe_candidates.append((float(balance.get("score", float("inf"))), int(tier), plan, motion, copy.deepcopy(report)))
                    print(
                        f"[D60-V72-SAFE-PAIR] tier={tier} A2={float(balance.get('a2_mm',0.0)):.1f}mm "
                        f"A1={float(balance.get('a1_mm',0.0)):.1f}mm diff={float(balance.get('difference_mm',999.0)):.1f}mm "
                        f"preferred={bool(balance.get('preferred',False))}"
                    )
                    if bool(balance.get("preferred", False)):
                        return self._v73_finalize_forced_waist_bias(planning_obs, plan, motion, report)
                    continue
                print(
                    f"[D60-V72-SAFE-SEARCH] tier={tier} pair rejected by final 35mm real-Mask inset; trying deeper/farther safe endpoint band"
                )
            if safe_candidates:
                safe_candidates.sort(key=lambda row: (row[0], row[1]))
                _, tier, plan, motion, report = safe_candidates[0]
                bal = dict(report.get("bottom_vla_v72_waist_balance", {}) or {})
                print(
                    f"[D60-V72-BALANCE-FALLBACK] no ideal balanced pair; use safest available tier={tier} "
                    f"A2={float(bal.get('a2_mm',0.0)):.1f}mm A1={float(bal.get('a1_mm',0.0)):.1f}mm "
                    f"diff={float(bal.get('difference_mm',999.0)):.1f}mm; hard 35mm inset preserved"
                )
                return self._v73_finalize_forced_waist_bias(planning_obs, plan, motion, report)
            total_trials = int(sum(int(r.get("pair_trials", 0) or 0) for r in reports))
            final_report = {
                "reason": "V72_NO_SAFE_WAIST_PAIR_INSIDE_35MM_INSET",
                "pair_trials": total_trials,
                "bottom_vla_v72_multi_band_search": True,
                "bottom_vla_v72_search_reports": reports,
                "bottom_vla_v72_waist_corner_refinement": copy.deepcopy(refinement),
            }
            print(
                "[D60-V72-SAFE-BLOCK] all WAIST endpoint bands exhausted; keep hard 35mm real-Mask inset and return NO PLAN"
            )
            return None, None, final_report

        def _d60_waist_endpoint_gate(self, obs, plan):
            if plan is None:
                return False, {"ok": False, "reason": "PLAN_UNAVAILABLE"}
            metrics = dict(getattr(plan, "metrics", {}) or {})
            tag = None
            limits = None
            if bool(metrics.get("bottom_vla_v73_waist_asym_bias", False)):
                source_limits = None
                for source_tag, default_limits in (
                    ("v72_d60_balanced_safe_pair_search", self._v67_d60_fast_limits()),
                    ("v71_d60_safe_pair_search", self._v67_d60_fast_limits()),
                    ("v67_d60_body_inset_fast", self._v67_d60_fast_limits()),
                    ("v50_d60_waistline_reach", self._v50_d60_reach_limits()),
                    ("v50_d60_shallow", self._v50_d60_shallow_limits()),
                ):
                    source_key = f"bottom_vla_{source_tag}"
                    if bool(metrics.get(source_key, False)):
                        source_limits = dict(metrics.get(f"{source_key}_limits", default_limits))
                        break
                if source_limits is None:
                    source_limits = {
                        "body_min_mm": float(self.front.d60_waist_grip_body_min_mm),
                        "body_max_mm": float(self.front.d60_waist_grip_body_max_mm),
                        "tangent_max_mm": float(self.front.d60_waist_grip_tangent_max_mm),
                        "radius_max_mm": float(self.front.d60_waist_grip_endpoint_radius_max_mm),
                    }
                limits = dict(source_limits)
                extra = 22.0
                limits["body_min_mm"] = max(0.0, float(limits["body_min_mm"]) - 22.0)
                limits["body_max_mm"] = float(limits["body_max_mm"]) + extra
                limits["radius_max_mm"] = float(limits["radius_max_mm"]) + extra
                tag = "v73_d60_asym_waist_bias"
            for candidate_tag, default_limits in (
                ("v72_d60_balanced_safe_pair_search", self._v67_d60_fast_limits()),
                ("v71_d60_safe_pair_search", self._v67_d60_fast_limits()),
                ("v67_d60_body_inset_fast", self._v67_d60_fast_limits()),
                ("v50_d60_waistline_reach", self._v50_d60_reach_limits()),
                ("v50_d60_shallow", self._v50_d60_shallow_limits()),
            ):
                if tag is not None:
                    break
                key = f"bottom_vla_{candidate_tag}"
                if bool(metrics.get(key, False)):
                    tag = candidate_tag
                    limits = dict(metrics.get(f"{key}_limits", default_limits))
                    break
            if tag is None or limits is None:
                ok, gate_report = super()._d60_waist_endpoint_gate(obs, plan)
            else:
                with self._v33_d60_rescue_lock:
                    old = (
                        float(self.front.d60_waist_grip_body_min_mm),
                        float(self.front.d60_waist_grip_body_max_mm),
                        float(self.front.d60_waist_grip_tangent_max_mm),
                        float(self.front.d60_waist_grip_endpoint_radius_max_mm),
                    )
                    try:
                        self.front.d60_waist_grip_body_min_mm = float(limits["body_min_mm"])
                        self.front.d60_waist_grip_body_max_mm = float(limits["body_max_mm"])
                        self.front.d60_waist_grip_tangent_max_mm = float(limits["tangent_max_mm"])
                        self.front.d60_waist_grip_endpoint_radius_max_mm = float(limits["radius_max_mm"])
                        ok, gate_report = super()._d60_waist_endpoint_gate(obs, plan)
                    finally:
                        self.front.d60_waist_grip_body_min_mm = old[0]
                        self.front.d60_waist_grip_body_max_mm = old[1]
                        self.front.d60_waist_grip_tangent_max_mm = old[2]
                        self.front.d60_waist_grip_endpoint_radius_max_mm = old[3]
            gate_report = dict(gate_report or {})
            if tag is not None and limits is not None:
                gate_report[f"bottom_vla_{tag}"] = True
                gate_report[f"bottom_vla_{tag}_limits"] = dict(limits)
            if not bool(ok):
                return False, gate_report
            inset_ok, inset_report = self._v53_validate_waist_safe_pair(obs, plan, "FINAL_GATE")
            gate_report["bottom_vla_v53_waist_inset"] = copy.deepcopy(inset_report)
            if not inset_ok:
                gate_report["ok"] = False
                gate_report["reason"] = str(inset_report.get("reason", "V53_WAIST_INSET_REJECT"))
                return False, gate_report
            return True, gate_report

        def _verify_sources_unchanged(self):
            changed = []
            base_checks = (
                ("D50_BASKET_SWING_LAYDOWN", getattr(self, "d50_path", None)),
                ("D54_OUTER_PULL", getattr(self, "d54_path", None)),
                ("D55_PRESS_SWEEP", getattr(self, "d55_path", None)),
                ("D56_WAIST_LIFT_LAYDOWN", getattr(self, "d56_path", None)),
                ("D58_CIRC_POSITION", getattr(self, "d58_path", None)),
            )
            expected_map = getattr(self, "source_sha256", {}) or {}
            for label, path in base_checks:
                expected = expected_map.get(label)
                if not path or not expected:
                    continue
                try:
                    current = v23._sha256(str(path))
                except Exception as exc:
                    changed.append(f"{label}:hash-error:{type(exc).__name__}")
                    continue
                if current != expected:
                    changed.append(label)
            extra_checks = (
                ("ALIGN", source_paths.get("align"), getattr(self, "align_source_sha256", None)),
                ("D60_DIRECT", source_paths.get("d60"), getattr(self, "d60_source_sha256", None)),
                ("POSITION_ADJUST", source_paths.get("position"), getattr(self, "position_source_sha256", None)),
            )
            for label, path, expected in extra_checks:
                if not path or not expected:
                    continue
                try:
                    current = v23._sha256(str(path))
                except Exception as exc:
                    changed.append(f"{label}:hash-error:{type(exc).__name__}")
                    continue
                if current != expected:
                    changed.append(label)
            signature = tuple(changed)
            if changed and signature != self._v30_source_warning_signature:
                self._v30_source_warning_signature = signature
                print(f"[SOURCE-V30-WARN] on-disk source changed after startup: {changed}; current in-memory loaded modules remain fixed, so execution continues")
            return None

        def _v30_mask_info(self, locked):
            obs = getattr(locked, "observation", None)
            mask = getattr(obs, "mask", None) if obs is not None else None
            if mask is None:
                return {"valid": False, "area_px": 0.0, "source": "NONE", "fallback": False}
            area = None
            for value in (getattr(mask, "area_px", None), getattr(mask, "area", None)):
                try:
                    candidate = float(value)
                    if math.isfinite(candidate) and candidate > 0.0:
                        area = candidate
                        break
                except Exception:
                    pass
            if area is None:
                try:
                    mask_u8 = getattr(mask, "mask_u8", None)
                    if mask_u8 is not None:
                        area = float(np.count_nonzero(np.asarray(mask_u8) > 0))
                except Exception:
                    area = None
            parts = []
            for obj in (mask, obs):
                for name in ("source", "mask_source", "class_name", "reason", "status", "raw_status"):
                    value = getattr(obj, name, None)
                    if value is not None:
                        parts.append(str(value))
                for name in ("d30_mask_info", "mask_info"):
                    value = getattr(obj, name, None)
                    if isinstance(value, dict):
                        for key in ("source", "fallback", "reason", "raw_status", "status"):
                            if value.get(key) is not None:
                                parts.append(str(value.get(key)))
            source = " | ".join(parts)
            upper = source.upper()
            fallback = any(token in upper for token in ("FALLBACK_BGDIFF", "CRUMPLED-FG", "CRUMPLED", "EMPTY-BOARD FOREGROUND", "BOARD_BLOB_FALLBACK"))
            return {
                "valid": True,
                "area_px": float(area or 0.0),
                "source": source or "MASK",
                "fallback": bool(fallback),
            }

        def _v30_patch_retry_records(self, fields):
            with self.recorder.lock:
                obs_record = getattr(self.recorder, "current_observation", None)
                dec_record = getattr(self.recorder, "current_decision", None)
                if isinstance(obs_record, dict):
                    v23._patch_json_record(obs_record, fields)
                if isinstance(dec_record, dict):
                    v23._patch_json_record(dec_record, fields)
                self.recorder.current_observation = None
                self.recorder.current_decision = None

        def _v30_finalize_perception_status(self):
            with self.state_lock:
                locked = self.locked
            if locked is None:
                return
            info = self._v30_mask_info(locked)
            semantic = str(self.pending_semantic or self.semantic_selected or locked.action)
            if bool(info["valid"]) and not bool(info["fallback"]) and float(info["area_px"]) > 0.0:
                self._v30_last_trusted_mask_area_px = float(info["area_px"])
            if bool(getattr(locked, "plan_ok", False)):
                return
            reason = str(getattr(locked, "reason", "") or getattr(locked, "planner_failure", "") or "NO_SAFE_PLAN")
            if not bool(info["valid"]):
                self.status = f"{semantic} RECOGNITION FAILED: {reason}"
                print(f"[PERCEPTION-V30] action={semantic} recognition=FAILED mask=NONE reason={reason}")
            else:
                self.status = f"{semantic} RECOGNITION OK / NO SAFE PLAN: {reason}"
                print(f"[PERCEPTION-V30] action={semantic} recognition=OK maskArea={float(info['area_px']):.0f}px planner=NO_SAFE_PLAN reason={reason}")

        def _prepare_action(self):
            action = str(self.selected_action or "")
            super()._prepare_action()
            with self.state_lock:
                locked = self.locked
            if str(action).upper() == "FINISH":
                if locked is None:
                    self._v74_finish_processing = False
                    self._v74_finish_complete_banner = False
                    self._v76_finish_commit_queued = False
                    with self.state_lock:
                        self.status = "FINISH PREP FAILED - RETRY FINISH"
                    print("[FINISH-V76-PREP-FAIL] FINISH inference completed without a locked terminal plan; lock released")
                    return
                if not bool(getattr(self, "_v76_finish_commit_queued", False)):
                    self._v76_finish_commit_queued = True
                    with self.state_lock:
                        self.status = "FINISH PLAN READY - AUTO SAVING / CLOSING EPISODE"
                    print("[FINISH-V76-AUTO-COMMIT] FINISH plan ready; queue internal commit now; ENTER not required")
                    self._queue("FINISH_COMMIT")
                return
            if locked is None:
                return
            info = self._v30_mask_info(locked)
            if action == "D58_CIRC_POSITION" and not bool(info.get("valid", False)):
                for retry_index in range(1, int(self._v103_position_seg_retry_count) + 1):
                    if getattr(self, "_queued_human_action", None) is not None:
                        break
                    self._v30_patch_retry_records({
                        "perception_retry_discarded": True,
                        "perception_retry_reason": "POSITION_NO_MASK_FRESH_RETRY",
                        "perception_retry_index": int(retry_index),
                        "pending_executable": False,
                        "training_eligible": False,
                    })
                    with self.state_lock:
                        self.locked = None
                        self.display_image = None
                        self.status = f"POSITION_ADJUST MASK MISS: FRESH RETRY {retry_index}/{int(self._v103_position_seg_retry_count)}"
                    print(
                        f"[POSITION-V103-SEG-RETRY] no valid D58 Mask; fresh retry "
                        f"{retry_index}/{int(self._v103_position_seg_retry_count)}"
                    )
                    time.sleep(float(self._v103_position_seg_retry_wait_s))
                    super()._prepare_action()
                    with self.state_lock:
                        locked = self.locked
                    if locked is None:
                        break
                    info = self._v30_mask_info(locked)
                    if bool(info.get("valid", False)):
                        print(
                            f"[POSITION-V103-SEG-RECOVERED] retry={retry_index} "
                            f"maskArea={float(info.get('area_px',0.0)):.0f}px"
                        )
                        break
                if locked is None:
                    return
                info = self._v30_mask_info(locked)
            reference = self._v30_last_trusted_mask_area_px
            retry_actions = {"D54_OUTER_PULL", "D55_PRESS_SWEEP"}
            undersized = bool(
                action in retry_actions
                and info["valid"]
                and info["fallback"]
                and reference is not None
                and float(reference) >= 30000.0
                and float(info["area_px"]) > 0.0
                and float(info["area_px"]) < float(reference) * float(self._v30_fallback_min_ratio)
            )
            if undersized and getattr(self, "_queued_human_action", None) is None:
                ratio = float(info["area_px"]) / max(float(reference), 1.0)
                fields = {
                    "perception_retry_discarded": True,
                    "perception_retry_reason": "SMALL_FALLBACK_MASK",
                    "perception_fallback_area_px": float(info["area_px"]),
                    "perception_last_trusted_area_px": float(reference),
                    "perception_area_ratio": float(ratio),
                    "pending_executable": False,
                    "training_eligible": False,
                }
                print(f"[PERCEPTION-V30-RETRY] action={action} fallbackArea={float(info['area_px']):.0f}px trustedRef={float(reference):.0f}px ratio={ratio:.3f}<0.450; discard candidate and take one fresh snapshot")
                self._v30_patch_retry_records(fields)
                with self.state_lock:
                    self.locked = None
                    self.display_image = None
                    self.status = f"{action} SMALL FALLBACK MASK: FRESH RETRY"
                time.sleep(0.08)
                super()._prepare_action()
                with self.state_lock:
                    locked = self.locked
                if locked is None:
                    return
                info = self._v30_mask_info(locked)
                still_small = bool(
                    info["valid"]
                    and info["fallback"]
                    and float(info["area_px"]) > 0.0
                    and float(info["area_px"]) < float(reference) * float(self._v30_fallback_min_ratio)
                )
                if still_small:
                    ratio = float(info["area_px"]) / max(float(reference), 1.0)
                    locked.plan_ok = False
                    locked.planner_failure = "UNTRUSTED_SMALL_FALLBACK_MASK"
                    locked.reason = f"fallback mask too small {float(info['area_px']):.0f}px vs trusted {float(reference):.0f}px ratio={ratio:.3f}"
                    locked.diagnostics = dict(getattr(locked, "diagnostics", {}) or {})
                    locked.diagnostics.update({
                        "v30_small_fallback_blocked": True,
                        "fallback_area_px": float(info["area_px"]),
                        "last_trusted_area_px": float(reference),
                        "fallback_area_ratio": float(ratio),
                    })
                    fields = {
                        "pending_executable": False,
                        "training_eligible": False,
                        "planner_failure": "UNTRUSTED_SMALL_FALLBACK_MASK",
                        "v30_small_fallback_blocked": True,
                        "fallback_area_px": float(info["area_px"]),
                        "last_trusted_area_px": float(reference),
                        "fallback_area_ratio": float(ratio),
                    }
                    with self.recorder.lock:
                        if isinstance(getattr(self.recorder, "current_observation", None), dict):
                            v23._patch_json_record(self.recorder.current_observation, fields)
                        if isinstance(getattr(self.recorder, "current_decision", None), dict):
                            v23._patch_json_record(self.recorder.current_decision, fields)
                    with self.state_lock:
                        self.locked = locked
                        self.status = f"{self.pending_semantic or action} RECOGNITION FAILED: FALLBACK MASK TOO SMALL"
                    print(f"[PERCEPTION-V30-BLOCK] action={action} second fallback still too small ratio={ratio:.3f}; ENTER disabled")
                    return
            self._v30_finalize_perception_status()
            self._v56_enforce_rejudge_backstop()
            self._v111_schedule_auto_execute_locked("PREPARE_COMPLETE")

        def _infer_for_action(self, action, frame):
            obs = super()._infer_for_action(action, frame)
            if bool(getattr(self, "_v53_align_capture_active", False)) and str(action) == "D56_WAIST_LIFT_LAYDOWN":
                self._v53_align_current_obs = obs
            return obs

        def _finite_point2(self, value):
            try:
                arr = np.asarray(value, np.float64).reshape(-1)[:2]
                if arr.size == 2 and np.all(np.isfinite(arr)):
                    return arr.astype(np.float64)
            except Exception:
                pass
            return None

        def _pose_board_points(self, obs):
            pose = getattr(obs, "pose", None)
            if pose is None:
                return {}
            out = {}
            aliases = {
                "waist_img_left": ("waist_img_left", "waist_left"),
                "waist_center": ("waist_center",),
                "waist_img_right": ("waist_img_right", "waist_right"),
                "crotch": ("crotch", "crotch_center"),
                "img_left_hem_outer": ("img_left_hem_outer", "left_hem_outer"),
                "img_left_hem_inner": ("img_left_hem_inner", "left_hem_inner"),
                "img_right_hem_inner": ("img_right_hem_inner", "right_hem_inner"),
                "img_right_hem_outer": ("img_right_hem_outer", "right_hem_outer"),
            }
            for dict_name in ("keypoints_board", "points_board", "landmarks_board", "refined_board"):
                d = getattr(pose, dict_name, None)
                if not isinstance(d, dict):
                    continue
                for canonical, names in aliases.items():
                    if canonical in out:
                        continue
                    for name in names:
                        p = self._finite_point2(d.get(name))
                        if p is not None:
                            out[canonical] = p
                            break
            for canonical, names in aliases.items():
                if canonical in out:
                    continue
                for name in names:
                    p = self._finite_point2(getattr(pose, name, None))
                    if p is not None:
                        out[canonical] = p
                        break
            for name in ("left_hem_center", "right_hem_center", "lower_center"):
                p = self._finite_point2(getattr(pose, name, None))
                if p is not None:
                    out[name] = p
            if "left_hem_center" not in out and "img_left_hem_outer" in out and "img_left_hem_inner" in out:
                out["left_hem_center"] = 0.5 * (out["img_left_hem_outer"] + out["img_left_hem_inner"])
            if "right_hem_center" not in out and "img_right_hem_inner" in out and "img_right_hem_outer" in out:
                out["right_hem_center"] = 0.5 * (out["img_right_hem_inner"] + out["img_right_hem_outer"])
            if "lower_center" not in out and "left_hem_center" in out and "right_hem_center" in out:
                out["lower_center"] = 0.5 * (out["left_hem_center"] + out["right_hem_center"])
            return out

        def _v100_mask_centerline_geometry(self, obs, seam_x=None):
            samples = self._v56_mask_board_samples(obs, self.H_raw, max_points=12000)
            if samples is None:
                return None
            pts = np.asarray(samples, np.float64).reshape(-1, 2)
            pts = pts[np.isfinite(pts).all(axis=1)]
            if len(pts) < 120:
                return None
            center = np.median(pts, axis=0)
            centered = pts - center
            try:
                cov = np.cov(centered.T)
                eigvals, eigvecs = np.linalg.eigh(cov)
                major = np.asarray(eigvecs[:, int(np.argmax(eigvals))], np.float64).reshape(2)
            except Exception:
                return None
            norm = float(np.linalg.norm(major))
            if norm < 1e-6:
                return None
            major /= norm
            pose_pts = self._pose_board_points(obs)
            pose_waist = pose_pts.get("waist_center")
            if pose_waist is None and pose_pts.get("waist_img_left") is not None and pose_pts.get("waist_img_right") is not None:
                pose_waist = 0.5 * (pose_pts["waist_img_left"] + pose_pts["waist_img_right"])
            pose_lower = pose_pts.get("crotch")
            if pose_lower is None:
                pose_lower = pose_pts.get("lower_center")
            if pose_waist is not None and pose_lower is not None:
                guide = np.asarray(pose_lower - pose_waist, np.float64)
                if float(np.linalg.norm(guide)) > 20.0 and float(np.dot(major, guide)) < 0.0:
                    major = -major
            elif float(major[1]) > 0.0:
                major = -major
            lateral = np.asarray([-major[1], major[0]], np.float64)
            longitudinal = centered @ major
            transverse = centered @ lateral
            try:
                s_lo, s_hi = np.percentile(longitudinal, [8.0, 82.0])
            except Exception:
                return None
            span = float(s_hi - s_lo)
            if not math.isfinite(span) or span < 120.0:
                return None
            rows = []
            half_window = max(14.0, span / 18.0)
            for s0 in np.linspace(float(s_lo), float(s_hi), 9):
                take = np.abs(longitudinal - float(s0)) <= half_window
                values = transverse[take]
                if len(values) < 24:
                    continue
                lo, hi = np.percentile(values, [8.0, 92.0])
                width = float(hi - lo)
                if not math.isfinite(width) or width < 45.0:
                    continue
                rows.append((float(s0), float(0.5 * (lo + hi)), width, int(len(values))))
            if len(rows) < 5:
                return None
            ss = np.asarray([row[0] for row in rows], np.float64)
            cc = np.asarray([row[1] for row in rows], np.float64)
            try:
                coef = np.polyfit(ss, cc, 1)
                fitted = np.polyval(coef, ss)
            except Exception:
                return None
            rms = float(np.sqrt(np.mean((fitted - cc) ** 2)))
            if not math.isfinite(rms) or rms > 35.0:
                return None
            s0 = float(min(row[0] for row in rows))
            s1 = float(max(row[0] for row in rows))
            c0 = float(np.polyval(coef, s0))
            c1 = float(np.polyval(coef, s1))
            waist = center + major * s0 + lateral * c0
            lower = center + major * s1 + lateral * c1
            axis = lower - waist
            axis_len = float(np.linalg.norm(axis))
            if not math.isfinite(axis_len) or axis_len < 100.0:
                return None
            if seam_x is None:
                try:
                    seam_x = float(self.align._c93_taught_trace_calibration(self.config, self.align_args)["boundary_x_mm"])
                except Exception:
                    return None
            raw_angle = math.degrees(math.atan2(float(axis[0]), float(-axis[1])))
            angle_error = float((raw_angle + 90.0) % 180.0 - 90.0)
            mid = 0.5 * (waist + lower)
            waist_err = float(waist[0] - seam_x)
            lower_err = float(lower[0] - seam_x)
            mid_err = float(mid[0] - seam_x)
            return {
                "seam_x_mm": float(seam_x),
                "waist_center": np.asarray(waist, np.float64),
                "lower_point": np.asarray(lower, np.float64),
                "lower_source": "MASK_WIDTH_CENTERLINE",
                "axis_source": "MASK_WIDTH_CENTERLINE",
                "midpoint": np.asarray(mid, np.float64),
                "axis_length_mm": axis_len,
                "angle_error_deg": angle_error,
                "waist_error_mm": waist_err,
                "lower_error_mm": lower_err,
                "mid_error_mm": mid_err,
                "max_endpoint_error_mm": float(max(abs(waist_err), abs(lower_err))),
                "mask_centerline_bins": int(len(rows)),
                "mask_centerline_fit_rms_mm": float(rms),
                "mask_centerline_median_width_mm": float(np.median([row[2] for row in rows])),
            }

        def _align_vector_geometry(self, obs, seam_x=None):
            mask_geom = self._v100_mask_centerline_geometry(obs, seam_x)
            if mask_geom is not None:
                return mask_geom
            pts = self._pose_board_points(obs)
            waist = pts.get("waist_center")
            if waist is None and pts.get("waist_img_left") is not None and pts.get("waist_img_right") is not None:
                waist = 0.5 * (pts["waist_img_left"] + pts["waist_img_right"])
            lower = pts.get("crotch")
            lower_source = "CROTCH"
            if lower is None:
                lower = pts.get("lower_center")
                lower_source = "LOWER_CENTER"
            if waist is None or lower is None:
                return None
            if seam_x is None:
                try:
                    seam_x = float(self.align._c93_taught_trace_calibration(self.config, self.align_args)["boundary_x_mm"])
                except Exception:
                    return None
            axis = lower - waist
            axis_len = float(np.linalg.norm(axis))
            if not math.isfinite(axis_len) or axis_len < 20.0:
                return None
            raw_angle = math.degrees(math.atan2(float(axis[0]), float(-axis[1])))
            angle_error = float((raw_angle + 90.0) % 180.0 - 90.0)
            mid = 0.5 * (waist + lower)
            waist_err = float(waist[0] - seam_x)
            lower_err = float(lower[0] - seam_x)
            mid_err = float(mid[0] - seam_x)
            return {
                "seam_x_mm": float(seam_x),
                "waist_center": waist,
                "lower_point": lower,
                "lower_source": lower_source,
                "axis_source": "POSE_WAIST_TO_" + str(lower_source),
                "midpoint": mid,
                "axis_length_mm": axis_len,
                "angle_error_deg": angle_error,
                "waist_error_mm": waist_err,
                "lower_error_mm": lower_err,
                "mid_error_mm": mid_err,
                "max_endpoint_error_mm": float(max(abs(waist_err), abs(lower_err))),
            }

        def _install_full_board_mask_roi(self):
            namespace_specs = (
                ("MAIN", "args"),
                ("D60", "d56_args"),
                ("D54", "d54_args"),
                ("D55", "d55_args"),
                ("D58", "d58_args"),
                ("ALIGN", "align_args"),
            )
            namespaces = []
            for label, name in namespace_specs:
                ns = getattr(self, name, None)
                if ns is None or any(id(ns) == id(existing) for _, existing in namespaces):
                    continue
                namespaces.append((label, ns))

            zero_fields = (
                "d35_aruco_roi_inset_px",
                "perception_roi_board_shrink_px",
                "planning_mask_board_shrink_px",
                "perception_board_shrink_px",
                "board_roi_shrink_px",
                "board_roi_expand_px",
                "board_marker_exclusion_radius_px",
                "board_frame_border_exclude_px",
                "d26v4_board_clip_expand_px",
                "d26v4_marker_exclusion_radius_px",
                "d26v4_frame_border_exclude_px",
                "d38_inner_marker_roi_extra_inset_px",
            )
            true_fields = (
                "d35_strict_aruco_roi",
                "d32_board_roi",
                "board_roi",
                "board_roi_strict",
                "perception_roi_clip_to_aruco_board",
                "planning_mask_clip_to_board",
            )

            for _label, ns in namespaces:
                for key in zero_fields:
                    setattr(ns, key, 0)
                for key in true_fields:
                    setattr(ns, key, True)

            sync_jobs = (
                ("D60", getattr(self, "d56", None), getattr(self, "d56_args", None), getattr(self, "H_raw", None)),
                ("D54", getattr(self, "d54", None), getattr(self, "d54_args", None), getattr(self, "H_corrected", None)),
                ("D55", getattr(self, "d55", None), getattr(self, "d55_args", None), getattr(self, "H_corrected", None)),
            )
            for label, module, ns, action_h in sync_jobs:
                sync = getattr(module, "_d38_sync_e49_runtime", None) if module is not None else None
                if callable(sync) and ns is not None:
                    try:
                        sync(ns, action_h)
                    except Exception as exc:
                        print(f"[MASK-V39-SYNC-WARN] {label} {type(exc).__name__}: {exc}")

            print(
                "[MASK-V39] exact four-marker quadrilateral for perception/planning; "
                "board shrink=0px, marker exclusion=0px, frame-border exclusion=0px, "
                "inner-marker extra inset=0px; outside quadrilateral remains clipped"
            )
            for label, ns in namespaces:
                print(
                    f"[MASK-V39-CONFIG] {label} "
                    f"strict={bool(getattr(ns, 'board_roi_strict', getattr(ns, 'd35_strict_aruco_roi', True)))} "
                    f"boardShrink={int(getattr(ns, 'board_roi_shrink_px', 0))} "
                    f"e49MarkerHole={int(getattr(ns, 'board_marker_exclusion_radius_px', 0))} "
                    f"legacyMarkerHole={int(getattr(ns, 'd26v4_marker_exclusion_radius_px', 0))} "
                    f"innerInset={int(getattr(ns, 'd38_inner_marker_roi_extra_inset_px', 0))}"
                )


        def _install_d58_arm1_contact_z_offset(self):
            original = getattr(self.d58, "contact_z", None)
            if not callable(original) or bool(getattr(original, "_bottom_vla_v106_position_z", False)):
                return
            arm1_offset = float(self._v30_d58_arm1_z_offset_mm)
            common_offset = float(self._v106_position_contact_z_extra_mm)

            def wrapped(config, arm_key, args):
                value = float(original(config, arm_key, args)) + common_offset
                if str(arm_key).lower() == "arm1":
                    value += arm1_offset
                return value

            wrapped._bottom_vla_v30_arm1_z = True
            wrapped._bottom_vla_v106_position_z = True
            self.d58.contact_z = wrapped

        def _v53_board_center(self, config):
            try:
                fn = getattr(self.d58, "board_bounds", None)
                if callable(fn):
                    xmin, xmax, ymin, ymax = fn(config)
                else:
                    marker_map = config.get("aruco", {}).get("marker_board_mm", {})
                    board = np.asarray(list(marker_map.values()), np.float64)
                    xmin, xmax = float(np.min(board[:, 0])), float(np.max(board[:, 0]))
                    ymin, ymax = float(np.min(board[:, 1])), float(np.max(board[:, 1]))
                return np.asarray([0.5 * (float(xmin) + float(xmax)), 0.5 * (float(ymin) + float(ymax))], np.float64)
            except Exception:
                return None

        def _v87_largest_vacancy(self, mask, H, config, args, center):
            samples_fn = getattr(self.d58, "_d58_robust_board_points", None)
            if not callable(samples_fn):
                return None
            try:
                samples = samples_fn(mask, H, int(getattr(args, "mask_sample_stride_px", 3)))
                samples = np.asarray(samples, np.float64).reshape(-1, 2)
            except Exception:
                return None
            if len(samples) < 30:
                return None
            try:
                xmin, xmax, ymin, ymax = self.d58.board_bounds(config)
            except Exception:
                board_center = self._v53_board_center(config)
                if board_center is None:
                    return None
                marker = np.asarray(list(config.get("aruco", {}).get("marker_board_mm", {}).values()), np.float64)
                if len(marker) < 4:
                    return None
                xmin, xmax = float(np.min(marker[:, 0])), float(np.max(marker[:, 0]))
                ymin, ymax = float(np.min(marker[:, 1])), float(np.max(marker[:, 1]))
            minx, maxx = float(np.min(samples[:, 0])), float(np.max(samples[:, 0]))
            miny, maxy = float(np.min(samples[:, 1])), float(np.max(samples[:, 1]))
            gaps = {
                "LEFT": max(0.0, minx - float(xmin)),
                "RIGHT": max(0.0, float(xmax) - maxx),
                "BOTTOM": max(0.0, miny - float(ymin)),
                "TOP": max(0.0, float(ymax) - maxy),
            }
            side = max(gaps, key=gaps.get)
            direction = {
                "LEFT": np.asarray([-1.0, 0.0], np.float64),
                "RIGHT": np.asarray([1.0, 0.0], np.float64),
                "BOTTOM": np.asarray([0.0, -1.0], np.float64),
                "TOP": np.asarray([0.0, 1.0], np.float64),
            }[side]
            amount = max(float(gaps[side]), float(getattr(args, "d58_min_move_mm", 30.0)))
            delta = direction * amount
            return {"side": side, "gaps": gaps, "delta": delta, "samples": samples}

        def _v87_position_rotation_ratio(self, center, grip, pull_u):
            c = np.asarray(center, np.float64).reshape(2)
            g = np.asarray(grip, np.float64).reshape(2)
            u = np.asarray(pull_u, np.float64).reshape(2)
            un = float(np.linalg.norm(u))
            if un < 1e-6:
                return 1.0, float("inf")
            u = u / un
            r = g - c
            rn = float(np.linalg.norm(r))
            if rn < 1e-6:
                return 1.0, 0.0
            n = np.asarray([-u[1], u[0]], np.float64)
            lever = abs(float(np.dot(r, n)))
            return float(lever / rn), float(lever)

        def _v87_rotation_guard_contact(self, mask, H, config, args, center, u, request, contact):
            contact_fn = getattr(self.d58, "_d58_contact_candidates", None)
            contour_fn = getattr(self.d58, "_d58_contour_board", None)
            if contact is None or not callable(contact_fn):
                return contact, {"reason": "NO_CONTACT"}
            best = contact
            best_ratio, best_lever = self._v87_position_rotation_ratio(center, contact["grip"], u)
            original_ratio, original_lever = best_ratio, best_lever
            try:
                contour = contour_fn(mask, H) if callable(contour_fn) else None
                if contour is None:
                    _, contour = self._v53_board_contour(SimpleNamespace(mask=mask))
                contour = np.asarray(contour, np.float64).reshape(-1, 2)
            except Exception:
                contour = None
            if contour is not None and len(contour) >= 12 and best_ratio > float(self._v87_position_rotation_preferred_ratio):
                center_v = np.asarray(center, np.float64).reshape(2)
                u_v = np.asarray(u, np.float64).reshape(2)
                u_v /= max(float(np.linalg.norm(u_v)), 1e-9)
                n_v = np.asarray([-u_v[1], u_v[0]], np.float64)
                rel = contour - center_v[None, :]
                forward = rel @ u_v
                lateral = np.abs(rel @ n_v)
                mx = float(np.max(forward))
                ids = np.flatnonzero(forward >= mx - 110.0)
                if len(ids) == 0:
                    ids = np.arange(len(contour))
                order = ids[np.argsort(lateral[ids] - 0.10 * forward[ids])]
                guard_args = copy.copy(args)
                try:
                    guard_args.d58_contact_max_candidates = min(
                        int(getattr(args, "d58_contact_max_candidates", self._v107_position_rotation_contact_max_candidates)),
                        int(self._v107_position_rotation_contact_max_candidates),
                    )
                except Exception:
                    pass
                for ii in order[:int(self._v107_position_rotation_guard_max_probes)]:
                    preferred = np.asarray(contour[int(ii)], np.float64)
                    cand, why = contact_fn(mask, H, config, center_v, u_v, float(request), guard_args, preferred_vertex=preferred)
                    if cand is None:
                        continue
                    ratio, lever = self._v87_position_rotation_ratio(center_v, cand["grip"], u_v)
                    move = float(cand.get("move_mm", 0.0))
                    best_move = float(best.get("move_mm", 0.0))
                    score = 120.0 * ratio + 0.08 * abs(float(request) - move)
                    best_score = 120.0 * best_ratio + 0.08 * abs(float(request) - best_move)
                    if score + 1e-6 < best_score:
                        best, best_ratio, best_lever = cand, float(ratio), float(lever)
                        if best_ratio <= float(self._v87_position_rotation_preferred_ratio):
                            break
            limited = False
            if best_ratio > float(self._v87_position_rotation_hard_ratio) and float(best.get("move_mm", 0.0)) > float(self._v87_position_rotation_limited_move_mm):
                limited_request = float(self._v87_position_rotation_limited_move_mm)
                preferred = np.asarray(best.get("raw", best.get("grip")), np.float64).reshape(2)
                cand, _ = contact_fn(mask, H, config, center, u, limited_request, locals().get("guard_args", args), preferred_vertex=preferred)
                if cand is not None:
                    ratio, lever = self._v87_position_rotation_ratio(center, cand["grip"], u)
                    best, best_ratio, best_lever = cand, float(ratio), float(lever)
                    limited = True
            report = {
                "original_ratio": float(original_ratio),
                "original_lever_mm": float(original_lever),
                "final_ratio": float(best_ratio),
                "final_lever_mm": float(best_lever),
                "limited_move": bool(limited),
            }
            return best, report

        def _v105_position_relaxed_contact(self, mask, H, config, args, center, pull_u, requested_move):
            samples_fn = getattr(self.d58, "_d58_robust_board_points", None)
            mask_support_fn = getattr(self.d58, "_d58_mask_support", None)
            board_inside_fn = getattr(self.d58, "board_inside", None)
            board_to_arm_fn = getattr(self.d58, "board_to_arm_xy", None)
            arm_for_x_fn = getattr(self.d58, "arm_for_x", None)
            if not all(callable(fn) for fn in (samples_fn, mask_support_fn, board_inside_fn, board_to_arm_fn, arm_for_x_fn)):
                return None, "V105_REQUIRED_D58_HELPER_MISSING"
            try:
                samples = samples_fn(mask, H, int(getattr(args, "mask_sample_stride_px", 3)))
                samples = np.asarray(samples, np.float64).reshape(-1, 2)
                samples = samples[np.isfinite(samples).all(axis=1)]
            except Exception as exc:
                return None, f"V105_SAMPLE_FAILED:{type(exc).__name__}"
            if len(samples) < 30:
                return None, "V105_MASK_BOARD_SAMPLES_TOO_FEW"
            center = np.asarray(center, np.float64).reshape(2)
            u = np.asarray(pull_u, np.float64).reshape(2)
            un = float(np.linalg.norm(u))
            if un < 1e-6:
                return None, "V105_ZERO_DIRECTION"
            u /= un
            proj = (samples - center.reshape(1, 2)) @ u
            order = np.argsort(proj)[::-1]
            max_candidates = min(len(order), int(self._v105_position_relaxed_max_candidates))
            min_move = float(getattr(args, "d58_min_move_mm", 30.0))
            radius_max = float(getattr(args, "d58_roarm_xy_radius_max_mm", 420.0))
            board_margin = float(self._v105_position_relaxed_board_margin_mm)
            preferred_arm = "arm1" if float(u[0]) > 0.05 else ("arm2" if float(u[0]) < -0.05 else None)
            best = None
            best_score = -1e18
            for ii in order[:max_candidates]:
                grip = np.asarray(samples[int(ii)], np.float64).reshape(2)
                try:
                    mask_ok, signed_px, local = mask_support_fn(mask, H, grip, args)
                except Exception:
                    continue
                if not bool(mask_ok):
                    continue
                if not bool(board_inside_fn(config, grip, board_margin)):
                    continue
                try:
                    arm = arm_for_x_fn(config, float(grip[0]), 0.0)
                except Exception:
                    arm = None
                arm_candidates = [arm] if arm in ("arm1", "arm2") else ["arm2", "arm1"]
                for arm_key in arm_candidates:
                    try:
                        gx, gy = board_to_arm_fn(config, arm_key, float(grip[0]), float(grip[1]))
                        r0 = float(math.hypot(gx, gy))
                    except Exception:
                        continue
                    if r0 > radius_max + 1e-6:
                        continue
                    chosen = None
                    end = None
                    r1 = None
                    distance = float(requested_move)
                    while distance >= min_move - 1e-6:
                        q = np.asarray(grip + u * distance, np.float64)
                        if bool(board_inside_fn(config, q, board_margin)):
                            try:
                                ex, ey = board_to_arm_fn(config, arm_key, float(q[0]), float(q[1]))
                                rr = float(math.hypot(ex, ey))
                            except Exception:
                                rr = float("inf")
                            if rr <= radius_max + 1e-6:
                                chosen = float(distance)
                                end = q
                                r1 = rr
                                break
                        distance -= 5.0
                    if end is None:
                        continue
                    lead = float(np.dot(grip - center, u))
                    arm_bonus = 22.0 if preferred_arm == arm_key else 0.0
                    score = (
                        1.25 * lead
                        + 0.55 * float(chosen)
                        + 55.0 * float(local)
                        + 1.8 * float(signed_px)
                        + arm_bonus
                        - 0.035 * max(float(r0), float(r1))
                    )
                    if score <= best_score:
                        continue
                    best_score = score
                    best = {
                        "arm": str(arm_key),
                        "raw": grip.astype(np.float32),
                        "grip": grip.astype(np.float32),
                        "end": end.astype(np.float32),
                        "move_mm": float(chosen),
                        "inset_mm": float(max(0.0, signed_px)),
                        "local": float(local),
                        "inside_px": float(signed_px),
                        "score": float(score),
                        "r0": float(r0),
                        "r1": float(r1),
                        "v105_private_workspace_relaxed": True,
                    }
            if best is None:
                return None, "V105_NO_MASK_SUPPORTED_BOARD_REACHABLE_CONTACT"
            return best, "V105_RELAXED_PRIVATE_WORKSPACE_CONTACT"

        def _v106_position_apply_grip_inset(self, frame, mask, H, config, args, plan):
            if plan is None or not bool(getattr(plan, "ok", False)):
                return plan
            grip = self._finite_point2(getattr(plan, "grip_board", None))
            end = self._finite_point2(getattr(plan, "end_board", None))
            center = self._finite_point2(getattr(plan, "center_board", None))
            if grip is None or end is None or center is None:
                return plan
            inward = np.asarray(center - grip, np.float64)
            norm = float(np.linalg.norm(inward))
            if norm < 1e-6:
                return plan
            inward /= norm
            board_inside_fn = getattr(self.d58, "board_inside", None)
            board_to_arm_fn = getattr(self.d58, "board_to_arm_xy", None)
            mask_support_fn = getattr(self.d58, "_d58_mask_support", None)
            if not all(callable(fn) for fn in (board_inside_fn, board_to_arm_fn, mask_support_fn)):
                return plan
            arm_key = str(getattr(plan, "selected_arm", ""))
            if arm_key not in ("arm1", "arm2"):
                return plan
            radius_max = float(getattr(args, "d58_roarm_xy_radius_max_mm", 420.0))
            target_offset = float(self._v106_position_grip_inset_offset_mm)
            selected = None
            offsets = []
            offset_now = float(target_offset)
            while offset_now >= 5.0 - 1e-6:
                offsets.append(float(offset_now))
                offset_now -= 5.0
            for offset in offsets:
                cand_grip = np.asarray(grip + inward * offset, np.float64)
                cand_end = np.asarray(end + inward * offset, np.float64)
                try:
                    mask_ok, signed_px, local = mask_support_fn(mask, H, cand_grip, args)
                except Exception:
                    continue
                if not bool(mask_ok):
                    continue
                if not bool(board_inside_fn(config, cand_grip, 0.0)) or not bool(board_inside_fn(config, cand_end, 0.0)):
                    continue
                try:
                    gx, gy = board_to_arm_fn(config, arm_key, float(cand_grip[0]), float(cand_grip[1]))
                    ex, ey = board_to_arm_fn(config, arm_key, float(cand_end[0]), float(cand_end[1]))
                    r0 = float(math.hypot(gx, gy))
                    r1 = float(math.hypot(ex, ey))
                except Exception:
                    continue
                if r0 > radius_max + 1e-6 or r1 > radius_max + 1e-6:
                    continue
                selected = (offset, cand_grip, cand_end, float(signed_px), float(local), r0, r1)
                break
            if selected is None:
                print(f"[D58-V107-INSET-WARN] target {target_offset:.0f}mm inward offset unavailable; keep original valid POSITION plan")
                return plan
            offset, cand_grip, cand_end, signed_px, local, r0, r1 = selected
            plan.grip_board = cand_grip.astype(np.float32)
            plan.end_board = cand_end.astype(np.float32)
            try:
                plan.v106_grip_inset_offset_mm = float(offset)
                plan.v106_grip_inset_target_mm = float(target_offset)
                plan.v106_grip_inset_signed_px = float(signed_px)
                plan.v106_grip_inset_local = float(local)
            except Exception:
                pass
            overlay_fn = getattr(self.d58, "make_d58_overlay", None)
            if callable(overlay_fn):
                try:
                    plan.overlay = overlay_fn(frame.copy(), mask, plan, H, config)
                except Exception:
                    pass
            print(
                f"[D58-V107-GRIP-INSET] arm={arm_key.upper()} offset={float(offset):.1f}/{target_offset:.1f}mm "
                f"grip={np.round(cand_grip,1).tolist()} end={np.round(cand_end,1).tolist()} "
                f"reach={r0:.0f}->{r1:.0f}mm"
            )
            return plan

        def _v53_recenter_d58_plan(self, frame, mask, H, config, args, plan):
            if plan is None:
                return None
            original_safe_plan = copy.deepcopy(plan) if bool(getattr(plan, "ok", False)) else None
            center = self._finite_point2(getattr(plan, "center_board", None))
            original_target = self._finite_point2(getattr(plan, "target_board", None))
            board_center = self._v53_board_center(config)
            if center is None:
                samples_fn = getattr(self.d58, "_d58_robust_board_points", None)
                if callable(samples_fn):
                    try:
                        samples = samples_fn(mask, H, int(getattr(args, "mask_sample_stride_px", 3)))
                        if samples is not None and len(samples) >= 30:
                            center = np.median(np.asarray(samples, np.float64), axis=0)
                            plan.center_board = np.asarray(center, np.float32)
                            plan.mask_center_board = np.asarray(center, np.float32)
                    except Exception:
                        center = None
            if center is None or board_center is None:
                return plan
            center_delta = np.asarray(board_center - center, np.float64)
            vacancy_report = self._v87_largest_vacancy(mask, H, config, args, center)
            if isinstance(vacancy_report, dict):
                delta = np.asarray(vacancy_report["delta"], np.float64)
                mode = f"LARGEST_VACANCY_{vacancy_report['side']}"
                gaps_text = ",".join(f"{k}:{float(v):.1f}" for k, v in vacancy_report["gaps"].items())
                print(f"[D58-V87-LARGEST-VACANCY] gaps={gaps_text} selected={vacancy_report['side']}")
            else:
                vacancy_delta = None if original_target is None else np.asarray(original_target - center, np.float64)
                vacancy_dist = 0.0 if vacancy_delta is None else float(np.linalg.norm(vacancy_delta))
                if vacancy_delta is not None and vacancy_dist >= 20.0:
                    delta = vacancy_delta.copy()
                    mode = "VACANCY_DIRECTION_FALLBACK"
                else:
                    delta = center_delta.copy()
                    dx, dy = float(delta[0]), float(delta[1])
                    mode = "FULL_CENTER_ERROR_FALLBACK"
                    if abs(dx) >= float(self._v53_position_lateral_min_mm) and abs(dx) >= float(self._v53_position_lateral_dominance) * max(abs(dy), 1.0):
                        delta[1] = 0.0
                        mode = "HORIZONTAL_DOMINANT_FALLBACK"
            dist = float(np.linalg.norm(delta))
            if dist < 5.0:
                try:
                    plan.ok = False
                    plan.reason = f"NO_POSITION_CORRECTION targetError={dist:.1f}mm"
                except Exception:
                    pass
                return plan
            unit_fn = getattr(self.d58, "_d58_unit", None)
            contact_fn = getattr(self.d58, "_d58_contact_candidates", None)
            if not callable(unit_fn) or not callable(contact_fn):
                return plan
            u = unit_fn(delta)
            if u is None:
                return plan
            move_cap = float(getattr(args, "d58_max_move_mm", 150.0))
            samples_fn = getattr(self.d58, "_d58_robust_board_points", None)
            if callable(samples_fn):
                try:
                    samples = samples_fn(mask, H, int(getattr(args, "mask_sample_stride_px", 3)))
                    if samples is not None and len(samples) >= 30:
                        span_x = float(np.ptp(np.asarray(samples)[:, 0]))
                        span_y = float(np.ptp(np.asarray(samples)[:, 1]))
                        garment_span = max(span_x, span_y)
                        if garment_span >= float(getattr(args, "d58_long_garment_span_mm", 480.0)):
                            move_cap = max(move_cap, float(getattr(args, "d58_long_garment_max_move_mm", 165.0)))
                except Exception:
                    pass
            request = min(move_cap, max(float(getattr(args, "d58_min_move_mm", 30.0)), dist * float(getattr(args, "d58_center_target_gain", 1.0))))
            vertex = self._finite_point2(getattr(plan, "contour_vertex", None))
            contact, why = contact_fn(mask, H, config, center, u, request, args, preferred_vertex=vertex)
            if contact is None:
                relaxed_contact, relaxed_why = self._v105_position_relaxed_contact(
                    mask, H, config, args, center, u, request
                )
                if relaxed_contact is not None:
                    contact = relaxed_contact
                    why = relaxed_why
                    print(
                        f"[D58-V105-RELAXED-CONTACT] strict directed contact failed; "
                        f"arm={str(contact.get('arm','?')).upper()} "
                        f"grip={np.round(np.asarray(contact.get('grip'),np.float64),1).tolist()} "
                        f"end={np.round(np.asarray(contact.get('end'),np.float64),1).tolist()} "
                        f"move={float(contact.get('move_mm',0.0)):.1f}mm "
                        f"reach={float(contact.get('r0',0.0)):.0f}->{float(contact.get('r1',0.0)):.0f}mm"
                    )
                else:
                    why = f"{why}; {relaxed_why}"
            rotation_report = None
            if contact is not None:
                contact, rotation_report = self._v87_rotation_guard_contact(mask, H, config, args, center, u, request, contact)
                if isinstance(rotation_report, dict):
                    print(
                        f"[D58-V87-ROTATION-GUARD] ratio {float(rotation_report.get('original_ratio', 0.0)):.2f}"
                        f"->{float(rotation_report.get('final_ratio', 0.0)):.2f} lever "
                        f"{float(rotation_report.get('original_lever_mm', 0.0)):.1f}"
                        f"->{float(rotation_report.get('final_lever_mm', 0.0)):.1f}mm "
                        f"limitedMove={bool(rotation_report.get('limited_move', False))}"
                    )
            target = center + delta
            old_source = str(getattr(plan, "target_source", "NONE") or "NONE")
            plan.target_board = np.asarray(target, np.float32)
            plan.target_source = f"V73_{mode}::{old_source}"
            plan.pull_unit = np.asarray(u, np.float32)
            if contact is None:
                if original_safe_plan is not None:
                    try:
                        original_safe_plan.v100_v73_fallback = True
                        original_safe_plan.v100_v73_failure_reason = str(why)
                    except Exception:
                        pass
                    print(
                        f"[D58-V101-FALLBACK] V73 directed contact unavailable ({why}); "
                        f"keep original safe D58 plan arm={str(getattr(original_safe_plan,'selected_arm','?')).upper()} "
                        f"move={float(getattr(original_safe_plan,'move_mm',0.0)):.1f}mm"
                    )
                    return original_safe_plan
                plan.ok = False
                plan.reason = "V73_POSITION_DIRECTED_CONTACT_UNAVAILABLE: " + str(why)
                plan.selected_arm = None
                plan.grip_board = None
                plan.end_board = None
                plan.move_mm = 0.0
                print(f"[D58-V73-BLOCK] center={np.round(center,1).tolist()} target={np.round(target,1).tolist()} delta={np.round(delta,1).tolist()} mode={mode} reason={why}")
            else:
                plan.selected_arm = str(contact["arm"])
                plan.grip_board = np.asarray(contact["grip"], np.float32)
                plan.end_board = np.asarray(contact["end"], np.float32)
                plan.move_mm = float(contact["move_mm"])
                plan.grip_inset_mm = float(contact["inset_mm"])
                plan.ok = True
                plan.reason = "OK"
                try:
                    plan.v87_rotation_guard = copy.deepcopy(rotation_report)
                    plan.v87_vacancy_mode = str(mode)
                    plan.v105_relaxed_contact = bool(contact.get("v105_private_workspace_relaxed", False))
                except Exception:
                    pass
                print(f"[D58-V73-VACANCY] center={np.round(center,1).tolist()} boardCenter={np.round(board_center,1).tolist()} originalTarget={None if original_target is None else np.round(original_target,1).tolist()} delta={np.round(delta,1).tolist()} mode={mode} request={request:.1f}mm arm={plan.selected_arm.upper()} move={plan.move_mm:.1f}mm")
            overlay_fn = getattr(self.d58, "make_d58_overlay", None)
            if callable(overlay_fn):
                try:
                    plan.overlay = overlay_fn(frame.copy(), mask, plan, H, config)
                except Exception as exc:
                    print(f"[D58-V73-OVERLAY-WARN] {exc!r}")
            return plan

        def _v55_post_basket_reselect_grip(self, frame, mask, H, config, args, plan):
            if plan is None or not bool(getattr(plan, "ok", False)):
                return plan
            center = self._finite_point2(getattr(plan, "center_board", None))
            grip0 = self._finite_point2(getattr(plan, "grip_board", None))
            end0 = self._finite_point2(getattr(plan, "end_board", None))
            u = self._finite_point2(getattr(plan, "pull_unit", None))
            if u is None and grip0 is not None and end0 is not None:
                u = end0 - grip0
            if center is None or u is None:
                return plan
            un = float(np.linalg.norm(u))
            if un < 1e-6:
                return plan
            u = np.asarray(u, np.float64) / un
            samples_fn = getattr(self.d58, "_d58_robust_board_points", None)
            contact_fn = getattr(self.d58, "_d58_contact_candidates", None)
            if not callable(samples_fn) or not callable(contact_fn):
                return plan
            try:
                samples = samples_fn(mask, H, int(getattr(args, "mask_sample_stride_px", 3)))
                samples = np.asarray(samples, np.float64).reshape(-1, 2)
            except Exception:
                return plan
            if len(samples) < 40:
                return plan
            pool = samples
            try:
                contour_px = np.asarray(getattr(mask, "contour", None), np.float64).reshape(-1, 2)
                if len(contour_px) >= 20:
                    Hm = np.asarray(H, np.float64).reshape(3, 3)
                    homo = np.column_stack((contour_px[:, 0], contour_px[:, 1], np.ones(len(contour_px), np.float64)))
                    hb = homo @ Hm.T
                    good = np.isfinite(hb).all(axis=1) & (np.abs(hb[:, 2]) > 1e-8)
                    contour_board = hb[good, :2] / hb[good, 2:3]
                    if len(contour_board) >= 20:
                        pool = contour_board
            except Exception:
                pool = samples
            rel = pool - center[None, :]
            lead = rel @ u
            side = np.abs(rel[:, 0] * (-u[1]) + rel[:, 1] * u[0])
            q = float(np.quantile(lead, float(self._v55_post_basket_lead_quantile)))
            idxs = np.flatnonzero(lead >= q)
            if len(idxs) < 6:
                idxs = np.argsort(lead)[-max(6, min(30, len(pool))):]
            score = lead[idxs] - 0.35 * side[idxs]
            preferred = np.asarray(pool[int(idxs[int(np.argmax(score))])], np.float64)
            request = float(getattr(plan, "move_mm", 0.0) or 0.0)
            request = max(request, float(getattr(args, "d58_min_move_mm", 30.0)))
            try:
                contact, why = contact_fn(mask, H, config, center, u, request, args, preferred_vertex=preferred)
            except Exception as exc:
                print(f"[D58-V55-POST-BASKET-GRIP-WARN] contact helper failed: {exc!r}; keep original safe plan")
                return plan
            if contact is None:
                print(f"[D58-V55-POST-BASKET-GRIP-FALLBACK] preferred={np.round(preferred,1).tolist()} unavailable reason={why}; keep original safe grip")
                return plan
            old_grip = grip0.copy() if grip0 is not None else None
            plan.selected_arm = str(contact["arm"])
            plan.grip_board = np.asarray(contact["grip"], np.float32)
            plan.end_board = np.asarray(contact["end"], np.float32)
            plan.move_mm = float(contact["move_mm"])
            plan.grip_inset_mm = float(contact["inset_mm"])
            plan.contour_vertex = np.asarray(preferred, np.float32)
            plan.target_source = f"V55_POST_BASKET_LEADING_INTERIOR::{getattr(plan, 'target_source', 'NONE')}"
            print(
                f"[D58-V55-POST-BASKET-GRIP] preferredLead={np.round(preferred,1).tolist()} "
                f"oldGrip={None if old_grip is None else np.round(old_grip,1).tolist()} -> "
                f"newGrip={np.round(np.asarray(plan.grip_board),1).tolist()} arm={plan.selected_arm.upper()} move={plan.move_mm:.1f}mm"
            )
            overlay_fn = getattr(self.d58, "make_d58_overlay", None)
            if callable(overlay_fn):
                try:
                    plan.overlay = overlay_fn(frame.copy(), mask, plan, H, config)
                except Exception as exc:
                    print(f"[D58-V55-POST-BASKET-OVERLAY-WARN] {exc!r}")
            return plan

        def _install_post_basket_position_policy(self):
            original = getattr(self.d58, "build_d58_plan", None)
            choose_original = getattr(self.d58, "_d58_choose_two_sides", None)
            if not callable(original) or not callable(choose_original) or bool(getattr(original, "_bottom_vla_v54_center_position", False)):
                return

            def wrapped(frame, mask, pose, H, config, args):
                post_basket = bool(getattr(self, "_v51_post_basket_position_pending", False))
                if post_basket:
                    old_choose = getattr(self.d58, "_d58_choose_two_sides")
                    old_min_gap = float(getattr(args, "d58_min_vacancy_gap_mm", 30.0))

                    def force_bottom_right(_info):
                        return "BOTTOM", "RIGHT"

                    try:
                        self.d58._d58_choose_two_sides = force_bottom_right
                        args.d58_min_vacancy_gap_mm = 0.0
                        print("[D58-V51-POST-BASKET] BOTTOM+RIGHT retained as the one-shot vacancy/contact context")
                        plan = original(frame, mask, pose, H, config, args)
                    finally:
                        self.d58._d58_choose_two_sides = old_choose
                        args.d58_min_vacancy_gap_mm = old_min_gap
                    if plan is not None:
                        plan.side_a = "BOTTOM"
                        plan.side_b = "RIGHT"
                else:
                    plan = original(frame, mask, pose, H, config, args)

                if post_basket:
                    if plan is not None:
                        plan.target_source = f"POST_BASKET_BOTTOM_RIGHT_EXACT::{getattr(plan, 'target_source', 'NONE')}"
                        plan = self._v55_post_basket_reselect_grip(frame, mask, H, config, args, plan)
                        print(
                            "[D58-V55-POST-BASKET] BOTTOM+RIGHT travel preserved; grip is reselected on the leading-side stable interior region"
                        )
                        plan = self._v56_position_rejudge_alternative(frame, mask, H, config, args, plan)
                    plan = self._v106_position_apply_grip_inset(frame, mask, H, config, args, plan)
                    return plan

                plan = self._v53_recenter_d58_plan(frame, mask, H, config, args, plan)
                plan = self._v56_position_rejudge_alternative(frame, mask, H, config, args, plan)
                plan = self._v106_position_apply_grip_inset(frame, mask, H, config, args, plan)
                return plan

            wrapped._bottom_vla_v54_center_position = True
            self.d58.build_d58_plan = wrapped

        def _install_align_surface_z_offset(self):
            offset = float(self._v30_align_surface_z_offset_mm)
            surface = getattr(self.align, "_c93_surface_z_at_board", None)
            if callable(surface) and not bool(getattr(surface, "_bottom_vla_v30_z", False)):
                def surface_wrapped(*args, **kwargs):
                    return float(surface(*args, **kwargs)) + offset
                surface_wrapped._bottom_vla_v30_z = True
                self.align._c93_surface_z_at_board = surface_wrapped
                return
            contact = getattr(self.align, "arm_contact_z", None)
            if callable(contact) and not bool(getattr(contact, "_bottom_vla_v30_z", False)):
                def contact_wrapped(*args, **kwargs):
                    return float(contact(*args, **kwargs)) + offset
                contact_wrapped._bottom_vla_v30_z = True
                self.align.arm_contact_z = contact_wrapped

        def _align_waist_y_from_plan(self, d56_plan):
            if bool(getattr(self, "_v53_align_capture_active", False)):
                current = self._v54_current_waist_geometry()
                if current is not None:
                    y = float(current["center"][1])
                    print(f"[ALIGN-V54-WAIST-Y] current actual waistband centerY={y:.1f}mm has priority over inset grasp Y")
                    return y
            try:
                y = super()._align_waist_y_from_plan(d56_plan)
            except Exception:
                y = None
            try:
                if y is not None and math.isfinite(float(y)):
                    return float(y)
            except Exception:
                pass
            if d56_plan is not None:
                metrics = d56_plan.get("metrics", {}) if isinstance(d56_plan, dict) else getattr(d56_plan, "metrics", {})
                metrics = dict(metrics or {}) if isinstance(metrics, dict) else {}
                for key in ("waist_center_board", "pose_waist_center_board", "waist_center", "waist_mid_board"):
                    p = self._finite_point2(metrics.get(key))
                    if p is not None:
                        print(f"[ALIGN-V53-WAIST-Y] recovered semantic waistY={float(p[1]):.1f}mm from {key}")
                        return float(p[1])
                points = d56_plan.get("arm_points", {}) if isinstance(d56_plan, dict) else getattr(d56_plan, "arm_points", {})
                ys = []
                if isinstance(points, dict):
                    for item in points.values():
                        if not isinstance(item, dict):
                            continue
                        for key in ("grip_board", "source_board", "grasp_board"):
                            p = self._finite_point2(item.get(key))
                            if p is not None:
                                ys.append(float(p[1]))
                                break
                if ys:
                    value = float(sum(ys) / len(ys))
                    print(f"[ALIGN-V53-WAIST-Y] recovered waistY={value:.1f}mm from dual D60 grip pair")
                    return value
            obs = getattr(self, "_v53_align_current_obs", None)
            if obs is not None:
                planning_obs, refinement = self._v54_obs_with_refined_waist(obs)
                consistency = self._v53_pose_waist_mask_consistency(planning_obs)
                if bool(consistency.get("ok", False)):
                    pts = self._pose_board_points(planning_obs)
                    center = self._finite_point2(pts.get("waist_center"))
                    if center is None:
                        left = self._finite_point2(pts.get("waist_img_left"))
                        right = self._finite_point2(pts.get("waist_img_right"))
                        if left is not None and right is not None:
                            center = 0.5 * (left + right)
                    if center is not None:
                        print(f"[ALIGN-V54-WAIST-Y] recovered current-snapshot refined waistY={float(center[1]):.1f}mm; no stale D60-plan dependency")
                        return float(center[1])
                else:
                    print(f"[ALIGN-V53-WAIST-Y] current Pose waist rejected by real-mask consistency: {consistency.get('reason','-')}")
            return None

        def _v50_align_left_guard_x(self, seam_x):
            try:
                marker_map = self.config.get("aruco", {}).get("marker_board_mm", {})
                board = np.asarray(list(marker_map.values()), np.float64)
                if board.ndim != 2 or board.shape[0] < 3 or board.shape[1] < 2:
                    return None
                xmin = float(np.min(board[:, 0]))
                xmax = float(np.max(board[:, 0]))
                width = float(xmax - xmin)
                if not math.isfinite(width) or width < 120.0:
                    return None
                guard = float(seam_x) + width * float(self._v54_align_guard_fraction)
                guard = min(xmax - 15.0, max(float(seam_x) + 20.0, guard))
                if guard <= float(seam_x) + 10.0 or guard >= xmax - 10.0:
                    return None
                return float(guard)
            except Exception:
                return None

        def _v50_mask_left_x(self, obs, H):
            try:
                mask_obj = getattr(obs, "mask", None)
                if mask_obj is None:
                    return None
                contour = getattr(mask_obj, "contour", None)
                px = None
                if contour is not None:
                    arr = np.asarray(contour, np.float64).reshape(-1, 2)
                    if len(arr) >= 20:
                        px = arr
                if px is None:
                    mask = getattr(mask_obj, "mask_u8", None)
                    if mask is None:
                        return None
                    mask = np.asarray(mask)
                    yy, xx = np.nonzero(mask > 0)
                    if len(xx) < 50:
                        return None
                    step = max(1, len(xx) // 5000)
                    px = np.column_stack((xx[::step], yy[::step])).astype(np.float64)
                Hm = np.asarray(H, np.float64).reshape(3, 3)
                homo = np.column_stack((px[:, 0], px[:, 1], np.ones(len(px), np.float64)))
                out = homo @ Hm.T
                good = np.isfinite(out).all(axis=1) & (np.abs(out[:, 2]) > 1e-8)
                if int(np.count_nonzero(good)) < 20:
                    return None
                board = out[good, :2] / out[good, 2:3]
                xs = board[:, 0]
                marker_map = self.config.get("aruco", {}).get("marker_board_mm", {})
                marker = np.asarray(list(marker_map.values()), np.float64)
                if marker.ndim == 2 and marker.shape[0] >= 3:
                    xmin = float(np.min(marker[:, 0])) - 15.0
                    xmax = float(np.max(marker[:, 0])) + 15.0
                    xs = xs[(xs >= xmin) & (xs <= xmax)]
                if len(xs) < 20:
                    return None
                return float(np.percentile(xs, 98.0))
            except Exception as exc:
                print(f"[ALIGN-V53-OUTER-WARN] mask boundary conversion failed: {exc!r}")
                return None

        def _draw_align_white_reference(self, canvas):
            canvas = super()._draw_align_white_reference(canvas)
            try:
                cal = self.align._c93_taught_trace_calibration(self.config, self.align_args)
                seam_x = float(cal["boundary_x_mm"])
                guard_x = self._v50_align_left_guard_x(seam_x)
                if guard_x is None:
                    return canvas
                y0 = float(getattr(self.align_cfg, "board_y_min", getattr(self.cfg56, "board_y_min", -780.0)))
                y1 = float(getattr(self.align_cfg, "board_y_max", getattr(self.cfg56, "board_y_max", 0.0)))
                margin = max(8.0, float(getattr(self.align_cfg, "board_margin_mm", 18.0)))
                p0 = self.d56.board_to_pixel(self.H_raw, guard_x, y0 + margin)
                p1 = self.d56.board_to_pixel(self.H_raw, guard_x, y1 - margin)
                if p0 is not None and p1 is not None:
                    a = (int(round(float(p0[0]))), int(round(float(p0[1]))))
                    b = (int(round(float(p1[0]))), int(round(float(p1[1]))))
                    self.cv2.line(canvas, a, b, (0, 0, 0), 6, self.cv2.LINE_AA)
                    self.cv2.line(canvas, a, b, (0, 0, 255), 3, self.cv2.LINE_AA)
                    self.cv2.putText(canvas, "LEFT OUTER LIMIT", (a[0] + 6, max(24, a[1] + 24)), self.cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 4, self.cv2.LINE_AA)
                    self.cv2.putText(canvas, "LEFT OUTER LIMIT", (a[0] + 6, max(24, a[1] + 24)), self.cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2, self.cv2.LINE_AA)
            except Exception as exc:
                print(f"[ALIGN-V53-OUTER-LINE-WARN] {exc!r}")
            return canvas

        def _v54_current_waist_geometry(self):
            obs = getattr(self, "_v53_align_current_obs", None)
            if obs is None:
                return None
            planning_obs, refinement = self._v54_obs_with_refined_waist(obs)
            consistency = self._v53_pose_waist_mask_consistency(planning_obs)
            if not bool(consistency.get("ok", False)):
                return None
            pts = self._pose_board_points(planning_obs)
            left = self._finite_point2(pts.get("waist_img_left"))
            center = self._finite_point2(pts.get("waist_center"))
            right = self._finite_point2(pts.get("waist_img_right"))
            if left is None or right is None:
                return None
            if center is None:
                center = 0.5 * (left + right)
            return {
                "left": left,
                "center": center,
                "right": right,
                "refinement": copy.deepcopy(refinement),
                "consistency": copy.deepcopy(consistency),
            }

        def _v59_align_stage1_reselect_sources(self, src, config, cfg, args):
            obs = getattr(self, "_v53_align_current_obs", None)
            semantic = self._v54_current_waist_geometry()
            if obs is None or semantic is None:
                return src, "D60_SOURCE_FALLBACK"
            planning_obs, _ = self._v54_obs_with_refined_waist(obs)
            pts = self._pose_board_points(planning_obs)
            crotch = self._finite_point2(pts.get("crotch"))
            center = self._finite_point2(semantic.get("center"))
            if crotch is None or center is None:
                return src, "D60_SOURCE_FALLBACK"
            body = np.asarray(crotch-center, np.float64)
            bn = float(np.linalg.norm(body))
            if bn < 1e-6:
                return src, "D60_SOURCE_FALLBACK"
            body /= bn
            point_ok = getattr(self.align, "_align3_dual_point_ok", None)
            if not callable(point_ok):
                return src, "D60_SOURCE_FALLBACK"

            selected = {}
            for arm_key, edge_key in (("arm2", "left"), ("arm1", "right")):
                edge = self._finite_point2(semantic.get(edge_key))
                if edge is None:
                    return src, "D60_SOURCE_FALLBACK"
                inward = np.asarray(center-edge, np.float64)
                inn = float(np.linalg.norm(inward))
                if inn < 1e-6:
                    return src, "D60_SOURCE_FALLBACK"
                inward /= inn
                trials = []
                for depth in (36.0, 40.0, 44.0):
                    for tang in (14.0, 20.0, 26.0, 32.0):
                        q = edge + body*depth + inward*tang
                        rep = self._v53_mask_point_report(planning_obs, q)
                        if not bool(rep.get("ok", False)):
                            continue
                        edge_depth = rep.get("edge_depth_mm")
                        if edge_depth is None or float(edge_depth) < 34.0 or float(edge_depth) > 58.0:
                            continue
                        support_ok, support_ratio = self._v56_mask_support_at_board(planning_obs, self.H_raw, q, radius_px=6, min_ratio=0.72)
                        if not support_ok:
                            continue
                        try:
                            ok, why, radius = point_ok(config, cfg, args, arm_key, q)
                        except Exception:
                            continue
                        if not bool(ok):
                            continue
                        score = (
                            2.0*abs(float(depth)-float(self._v59_align_stage1_body_inset_mm))
                            + 0.45*float(tang)
                            + 1.2*abs(float(edge_depth)-42.0)
                            + 0.03*float(radius)
                            + 18.0*(1.0-float(support_ratio))
                        )
                        trials.append((score, np.asarray(q, np.float64), float(edge_depth), float(depth), float(tang)))
                if not trials:
                    return src, "D60_SOURCE_FALLBACK"
                trials.sort(key=lambda row: row[0])
                selected[arm_key] = trials[0]

            out = {k: selected[k][1] for k in ("arm2", "arm1")}
            sep = float(np.linalg.norm(out["arm1"]-out["arm2"]))
            if sep < 120.0:
                return src, "D60_SOURCE_FALLBACK"
            print(
                f"[ALIGN-V61-WAIST-SOURCE] A2={np.round(out['arm2'],1).tolist()} edgeDepth={selected['arm2'][2]:.1f}mm "
                f"A1={np.round(out['arm1'],1).tolist()} edgeDepth={selected['arm1'][2]:.1f}mm sep={sep:.1f}mm; "
                "current Mask-waist interior points replace low D60 grips for ALIGN only"
            )
            return out, "CURRENT_MASK_WAIST_INTERIOR_V59"

        def _v54_level_waist_top_motion(self, motion, config, cfg, args):
            if motion is None:
                return None, "ALIGN_V63_MOTION_MISSING"
            if bool(motion.get("skip", False)):
                return motion, "SKIP"
            try:
                out = copy.deepcopy(motion)
                src_raw = dict(out.get("source_board", {}) or {})
                src = {k: np.asarray(src_raw[k], np.float64).reshape(2) for k in ("arm2", "arm1")}
            except Exception as exc:
                return None, f"ALIGN_V63_SOURCE_INVALID:{exc!r}"

            src, source_mode = self._v59_align_stage1_reselect_sources(src, config, cfg, args)
            if str(source_mode).startswith("D60_SOURCE_FALLBACK"):
                return None, "ALIGN_V63_REAL_WAIST_SOURCE_REQUIRED_NO_LEG_FALLBACK"
            out["source_board"] = {k: np.asarray(src[k], np.float64).astype(float).tolist() for k in ("arm2", "arm1")}
            out["v59_align_stage1_source_mode"] = str(source_mode)

            semantic = self._v54_current_waist_geometry()
            if semantic is None:
                return None, "ALIGN_V63_REAL_WAIST_GEOMETRY_REQUIRED"
            left_edge = np.asarray(semantic["left"], np.float64).reshape(2)
            right_edge = np.asarray(semantic["right"], np.float64).reshape(2)
            direct_cost = float(np.linalg.norm(src["arm2"]-left_edge) + np.linalg.norm(src["arm1"]-right_edge))
            swapped_cost = float(np.linalg.norm(src["arm2"]-right_edge) + np.linalg.norm(src["arm1"]-left_edge))
            if swapped_cost + 5.0 < direct_cost:
                matched = {"arm2": right_edge, "arm1": left_edge}
                waist_source = "CURRENT_MASK_OUTER_WAIST_ENDPOINTS_V63_SWAPPED_BY_SOURCE"
            else:
                matched = {"arm2": left_edge, "arm1": right_edge}
                waist_source = "CURRENT_MASK_OUTER_WAIST_ENDPOINTS_V63_NEAREST_SOURCE"
            waist_y = {k: float(matched[k][1]) for k in ("arm2", "arm1")}
            print(
                f"[ALIGN-V63-WAIST-MAP] directCost={direct_cost:.1f}mm swappedCost={swapped_cost:.1f}mm "
                f"mapping={waist_source}; only actual waist endpoints are allowed for LEVEL/UP"
            )

            source_gap = abs(float(waist_y["arm2"])-float(waist_y["arm1"]))
            semantic_level_y = max(float(waist_y["arm2"]), float(waist_y["arm1"]))
            requested_waist_y = float(self.front.align_waist_target_y_mm)
            requested_up_dy = max(0.0, requested_waist_y-semantic_level_y)
            level_required = bool(source_gap > float(self._v54_align_level_y_tolerance_mm))
            up_required_after_level = bool(requested_up_dy >= float(self._v54_align_waist_recheck_mm))

            level_points = {}
            for k in ("arm2", "arm1"):
                level_dy = max(0.0, semantic_level_y-float(waist_y[k]))
                level_points[k] = src[k] + np.asarray([0.0, level_dy], np.float64)

            point_ok = getattr(self.align, "_align3_dual_point_ok", None)
            if not callable(point_ok):
                return None, "ALIGN_V63_POINT_GUARD_MISSING"

            def pair_safe(points):
                checks = {}
                for k in ("arm2", "arm1"):
                    try:
                        ok, why, radius = point_ok(config, cfg, args, k, points[k])
                    except Exception as exc:
                        return False, {k:{"ok":False,"reason":f"GUARD_ERROR:{exc!r}"}}
                    checks[k] = {"ok":bool(ok),"reason":str(why),"radius_mm":float(radius)}
                    if not bool(ok):
                        return False, checks
                return True, checks

            level_ok, level_checks = pair_safe(level_points)
            if not level_ok:
                return None, f"ALIGN_V63_LEVEL_UNSAFE:{level_checks}"

            final_fraction = 1.0
            final_checks = copy.deepcopy(level_checks)
            selected_up_dy = 0.0
            if level_required:
                stage_mode = "LEVEL_ONLY"
                final_points = {k: np.asarray(level_points[k],np.float64).copy() for k in ("arm2","arm1")}
                selected_waist_y = float(semantic_level_y)
            elif up_required_after_level:
                stage_mode = "UP_ONLY"
                desired_final = {k: level_points[k] + np.asarray([0.0,requested_up_dy],np.float64) for k in ("arm2","arm1")}
                final_points = desired_final
                final_ok, final_checks = pair_safe(final_points)
                if not final_ok:
                    original_failure = copy.deepcopy(final_checks)
                    final_points = None
                    for frac in np.linspace(0.95,0.10,18):
                        cand = {k: level_points[k] + np.asarray([0.0,requested_up_dy*float(frac)],np.float64) for k in ("arm2","arm1")}
                        cand_ok, cand_checks = pair_safe(cand)
                        if cand_ok:
                            final_points = cand
                            final_checks = cand_checks
                            final_fraction = float(frac)
                            break
                    if final_points is None:
                        return None, f"ALIGN_V63_UP_UNSAFE:{original_failure}"
                selected_up_dy = float(requested_up_dy*final_fraction)
                selected_waist_y = float(semantic_level_y+selected_up_dy)
            else:
                stage_mode = "ALREADY_LEVEL_AND_HIGH"
                final_points = {k: np.asarray(level_points[k],np.float64).copy() for k in ("arm2","arm1")}
                selected_waist_y = float(semantic_level_y)

            robot = dict(out.get("robot", {}) or {})
            if any(k not in robot for k in ("arm2","arm1")):
                return None, "ALIGN_V63_ROBOT_POINTS_MISSING"
            for k in ("arm2","arm1"):
                r = dict(robot[k])
                try:
                    lrx,lry = self.align.board_to_arm_xy(config,k,float(level_points[k][0]),float(level_points[k][1]))
                    frx,fry = self.align.board_to_arm_xy(config,k,float(final_points[k][0]),float(final_points[k][1]))
                    travel=np.asarray(r["travel"],np.float64).reshape(3)
                    target_travel=np.asarray(r["target_travel"],np.float64).reshape(3)
                    place=np.asarray(r["place"],np.float64).reshape(3)
                    clear=np.asarray(r["clear"],np.float64).reshape(3)
                except Exception as exc:
                    return None, f"ALIGN_V63_ROBOT_REMAP_FAILED:{k}:{exc!r}"
                r["v54_level_travel"] = np.asarray([lrx,lry,float(travel[2])],np.float32)
                r["target_travel"] = np.asarray([frx,fry,float(target_travel[2])],np.float32)
                r["place"] = np.asarray([frx,fry,float(place[2])],np.float32)
                r["clear"] = np.asarray([frx,fry,float(clear[2])],np.float32)
                robot[k]=r

            out["robot"] = robot
            out["target_board"] = {k:np.asarray(final_points[k],np.float64).astype(float).tolist() for k in ("arm2","arm1")}
            out["v54_level_board"] = {k:np.asarray(level_points[k],np.float64).astype(float).tolist() for k in ("arm2","arm1")}
            out["v54_waist_y_source"] = waist_source
            out["v54_waist_source_y_mm"] = {k:float(v) for k,v in waist_y.items()}
            out["v54_source_y_gap_mm"] = float(source_gap)
            out["v54_semantic_level_y_mm"] = float(semantic_level_y)
            out["v54_requested_waist_target_y_mm"] = float(requested_waist_y)
            out["v54_selected_waist_target_y_mm"] = float(selected_waist_y)
            out["v54_requested_up_dy_mm"] = float(requested_up_dy)
            out["v54_selected_up_dy_mm"] = float(selected_up_dy)
            out["v54_level_required"] = bool(level_required)
            out["v54_level_checks"] = copy.deepcopy(level_checks)
            out["v54_final_checks"] = copy.deepcopy(final_checks)
            out["v54_final_fraction"] = float(final_fraction)
            out["v54_no_downward_waist_move"] = True
            out["v63_align_stage_mode"] = str(stage_mode)
            out["v63_up_required_after_level"] = bool(up_required_after_level)
            out["v63_up_remaining_mm"] = float(max(0.0, requested_waist_y-selected_waist_y))
            out["selected_dy_mm"] = float(np.median([float(final_points[k][1]-src[k][1]) for k in ("arm2","arm1")]))
            print(
                f"[ALIGN-V63-FIRST-PLAN] mode={stage_mode} waistY A2/A1={waist_y['arm2']:.1f}/{waist_y['arm1']:.1f}mm "
                f"gap={source_gap:.1f}mm levelY={semantic_level_y:.1f}mm targetY={requested_waist_y:.1f}mm "
                f"selectedY={selected_waist_y:.1f}mm remaining={float(out['v63_up_remaining_mm']):.1f}mm"
            )
            return out, "OK"

        def _v59_execute_dual_waist_press_drag(self, motion, arms, cfg, args, on_verified_start=None):
            if motion is None or bool(motion.get("skip", False)):
                return False, "ALIGN_V59_STAGE1_MOTION_MISSING"
            try:
                source = {k: np.asarray(motion["source_board"][k], np.float64).reshape(2) for k in ("arm2", "arm1")}
                level_raw = motion.get("v54_level_board", motion["source_board"])
                level = {k: np.asarray(level_raw[k], np.float64).reshape(2) for k in ("arm2", "arm1")}
                final = {k: np.asarray(motion["target_board"][k], np.float64).reshape(2) for k in ("arm2", "arm1")}
            except Exception as exc:
                return False, f"ALIGN_V59_STAGE1_POINTS_INVALID:{exc!r}"
            stage_mode = str(motion.get("v63_align_stage_mode", "LEVEL_ONLY")).upper()
            do_level = stage_mode == "LEVEL_ONLY"
            do_up = stage_mode == "UP_ONLY"
            if stage_mode == "ALREADY_LEVEL_AND_HIGH":
                self._v53_align_initial_dual_done = True
                print("[ALIGN-V63-STAGE1-NOOP] waistband already level and high enough; no LEVEL/UP motion")
                return True, "ACTION_OK"
            if cfg.dry_run or not cfg.send:
                return True, "PREVIEW"
            if any(k not in arms for k in ("arm2", "arm1")):
                return False, "BOTH_ARMS_REQUIRED"

            config = self.config
            move_command = int(getattr(args, "move_command", 104))
            hover_z = float(max(float(getattr(cfg, "clear_z", 180.0)), float(getattr(args, "d31_hover_z", 180.0))))
            approach_speed = float(np.clip(getattr(args, "d31_approach_speed", 1.12), 0.45, 1.20))
            vertical_speed = float(np.clip(getattr(args, "d31_vertical_speed", 1.12), 0.35, 1.20))
            drag_speed = float(np.clip(self._v70_align_drag_speed, 0.25, 0.55))
            clear_speed = float(np.clip(getattr(args, "align6_arm2_clear_speed", 0.50), 0.25, 0.70))
            close_angle = float(np.clip(getattr(args, "d31_grip_close", 3.05), 2.90, 3.14))
            surface_fn = getattr(self.align, "_c93_surface_z_at_board", None)
            if not callable(surface_fn):
                return False, "ALIGN_V59_SURFACE_API_MISSING"
            point_ok = getattr(self.align, "_align3_dual_point_ok", None)
            if callable(point_ok):
                max_dist = max(float(np.linalg.norm(final[k]-level[k])) for k in ("arm2", "arm1"))
                precheck_segments = max(1, int(math.ceil(max_dist/25.0)))
                for k in ("arm2", "arm1"):
                    check_points = [source[k], level[k]]
                    check_points.extend(level[k] + (final[k]-level[k])*(float(i)/float(precheck_segments)) for i in range(1, precheck_segments+1))
                    for q in check_points:
                        try:
                            ok, why, _ = point_ok(config, cfg, args, k, q)
                        except Exception as exc:
                            return False, f"ALIGN_V59_STAGE1_PRECHECK_ERROR:{k}:{exc!r}"
                        if not bool(ok):
                            return False, f"ALIGN_V59_STAGE1_PRECHECK_UNSAFE:{k}:{why}"
            if callable(on_verified_start):
                on_verified_start()

            def xyz_at(arm_key, board_point, z):
                rx, ry = self.align.board_to_arm_xy(config, arm_key, float(board_point[0]), float(board_point[1]))
                return np.asarray([float(rx), float(ry), float(z)], np.float32)

            def run_parallel(targets, speed, stage, wait_from):
                results = {}
                lock = threading.Lock()
                def runner(arm_key):
                    try:
                        res = self.align._d31_move_verified(
                            arms[arm_key], targets[arm_key], cfg, speed, move_command, args,
                            f"ALIGN_V59_STAGE1_{stage}_{arm_key.upper()}",
                            wait_from=None if wait_from is None else wait_from.get(arm_key),
                        )
                    except Exception as exc:
                        res = (False, f"EXCEPTION:{exc!r}", None)
                    with lock:
                        results[arm_key] = res
                threads = [threading.Thread(target=runner, args=(k,), daemon=True) for k in ("arm2", "arm1")]
                for th in threads:
                    th.start()
                for th in threads:
                    th.join()
                for k in ("arm2", "arm1"):
                    res = results.get(k, (False, "NO_RESULT", None))
                    if not bool(res[0]):
                        return False, f"{k.upper()}_{res[1]}"
                return True, "OK"

            current = {}
            try:
                close_threads = []
                for k in ("arm2", "arm1"):
                    th = threading.Thread(
                        target=arms[k].set_gripper,
                        args=(close_angle,),
                        kwargs={"delay": 0.25, "stage": "ALIGN_V59_STAGE1_CLOSED_BEFORE_APPROACH", "caller": "_v59_execute_dual_waist_press_drag"},
                        daemon=True,
                    )
                    close_threads.append(th)
                    th.start()
                for th in close_threads:
                    th.join()

                approach = {k: xyz_at(k, source[k], hover_z) for k in ("arm2", "arm1")}
                ok, why = run_parallel(approach, approach_speed, "APPROACH_CLOSED", None)
                if not ok:
                    return False, why
                current = approach

                contact = {}
                for k in ("arm2", "arm1"):
                    fallback_floor = float(self.align.arm_contact_z(cfg, k))
                    floor = float(surface_fn(config, k, source[k], fallback_floor))
                    contact[k] = xyz_at(k, source[k], floor + float(self._v70_align_stage1_press_clearance_mm))
                ok, why = run_parallel(contact, vertical_speed, "PRESS_DOWN_CLOSED", current)
                if not ok:
                    return False, why
                current = contact

                if do_level and max(float(np.linalg.norm(level[k]-source[k])) for k in ("arm2", "arm1")) > 2.0:
                    level_xyz = {}
                    for k in ("arm2", "arm1"):
                        fallback_floor = float(self.align.arm_contact_z(cfg, k))
                        floor = float(surface_fn(config, k, level[k], fallback_floor))
                        level_xyz[k] = xyz_at(k, level[k], floor + float(self._v70_align_stage1_press_clearance_mm))
                    ok, why = run_parallel(level_xyz, drag_speed, "LEVEL_PRESS_DRAG", current)
                    if not ok:
                        return False, why
                    current = level_xyz
                    print("[ALIGN-V63-LEVEL-DONE] waistband LEVEL-only drag completed; UP is intentionally deferred until a fresh snapshot")
                    time.sleep(float(self._v61_align_level_settle_s))

                if do_up:
                    targets = {}
                    for k in ("arm2", "arm1"):
                        q = final[k]
                        fallback_floor = float(self.align.arm_contact_z(cfg, k))
                        floor = float(surface_fn(config, k, q, fallback_floor))
                        targets[k] = xyz_at(k, q, floor + float(self._v70_align_stage1_press_clearance_mm))
                    ok, why = run_parallel(targets, drag_speed, "UP_PRESS_DRAG_CONTINUOUS", current)
                    if not ok:
                        return False, why
                    current = targets
                    print(f"[ALIGN-V63-UP-CONTINUOUS] fresh recheck requested UP-only; both arms move once toward targetY={float(self.front.align_waist_target_y_mm):.1f}mm")

                clear = {k: np.asarray([float(current[k][0]), float(current[k][1]), hover_z], np.float32) for k in ("arm2", "arm1")}
                ok, why = run_parallel(clear, clear_speed, "PURE_Z_CLEAR_CLOSED", current)
                if not ok:
                    return False, why
                for k in ("arm2", "arm1"):
                    if not self._v60_align_return_standby_verified(
                        k, arms, cfg, move_command,
                        "ALIGN-V60 stage1 dual waist press-drag; gripper remains closed",
                    ):
                        return False, f"ALIGN_V60_{k.upper()}_STANDBY_NOT_VERIFIED"
                move_a2 = float(np.linalg.norm(final["arm2"]-source["arm2"]))
                move_a1 = float(np.linalg.norm(final["arm1"]-source["arm1"]))
                if stage_mode == "LEVEL_ONLY":
                    self._v53_align_initial_dual_done = not bool(motion.get("v63_up_required_after_level", False))
                elif stage_mode == "UP_ONLY":
                    self._v53_align_initial_dual_done = bool(float(motion.get("v63_up_remaining_mm", 0.0)) < float(self._v54_align_waist_recheck_mm))
                print(
                    f"[ALIGN-V63-STAGE1-PRESS] mode={stage_mode} sourceA2={np.round(source['arm2'],1).tolist()} "
                    f"sourceA1={np.round(source['arm1'],1).tolist()} targetY={float(self.front.align_waist_target_y_mm):.1f}mm "
                    f"moves={move_a2:.1f}/{move_a1:.1f}mm; LEVEL and UP are separate actions"
                )
                return True, "ACTION_OK"
            except Exception as exc:
                print(f"[ALIGN-V61-STAGE1-PRESS-ERROR] {exc!r}")
                try:
                    for k in ("arm2", "arm1"):
                        fb = self.align._d31_query_feedback(arms[k], timeout_s=0.8)
                        xyz = self.align._d31_feedback_xyz(fb)
                        if xyz is not None:
                            xyz = np.asarray(xyz, np.float32).reshape(3)
                            arms[k].move_goal(
                                float(xyz[0]), float(xyz[1]), hover_z, float(getattr(cfg, "tool_t_rad", 3.14)), 0.35,
                                move_command=move_command, stage="ALIGN_V59_STAGE1_EXCEPTION_PURE_Z_CLEAR_CLOSED",
                                caller="_v59_execute_dual_waist_press_drag",
                            )
                    time.sleep(0.8)
                    for k in ("arm2", "arm1"):
                        self._v60_align_return_standby_verified(
                            k, arms, cfg, move_command,
                            "ALIGN-V60 stage1 exception return; gripper remains closed",
                        )
                except Exception as exc2:
                    print(f"[ALIGN-V61-STAGE1-RECOVERY-WARN] {exc2!r}")
                return False, f"ALIGN_V59_STAGE1_PRESS_EXCEPTION:{exc!r}"

        def _install_align_waist_level_policy(self):
            builder = getattr(self.align, "_align3_build_dual_waist_top_locked", None)
            executor = getattr(self.align, "_align3_execute_dual_waist_top_locked", None)
            if not callable(builder) or not callable(executor):
                print("[ALIGN-V61-LEVEL-WARN] align-11 dual waist builder/executor unavailable")
                return

            if not bool(getattr(builder, "_bottom_vla_v59_waist_level", False)):
                original_builder = builder

                def builder_wrapped(d56_plan, config, cfg, args):
                    motion, why = original_builder(d56_plan, config, cfg, args)
                    if motion is None or bool(motion.get("skip", False)):
                        return motion, why
                    leveled, level_why = self._v54_level_waist_top_motion(motion, config, cfg, args)
                    if leveled is None:
                        print(f"[ALIGN-V61-FIRST-BLOCK] LEVEL+UP press plan unavailable reason={level_why}")
                        return None, str(level_why)
                    return leveled, why

                builder_wrapped._bottom_vla_v59_waist_level = True
                self.align._align3_build_dual_waist_top_locked = builder_wrapped

            executor = getattr(self.align, "_align3_execute_dual_waist_top_locked", None)
            if callable(executor) and not bool(getattr(executor, "_bottom_vla_v59_waist_press", False)):
                original_executor = executor

                def executor_wrapped(d56_plan, motion, arms, cfg, args, on_verified_start=None):
                    if motion is None or bool(motion.get("skip", False)):
                        return original_executor(
                            d56_plan, motion, arms, cfg, args,
                            on_verified_start=on_verified_start,
                        )
                    return self._v59_execute_dual_waist_press_drag(
                        motion, arms, cfg, args, on_verified_start=on_verified_start
                    )

                executor_wrapped._bottom_vla_v59_waist_press = True
                self.align._align3_execute_dual_waist_top_locked = executor_wrapped

        def _v56_mask_support_at_board(self, obs, H, point, radius_px=5, min_ratio=0.62):
            try:
                mask_obj = getattr(obs, "mask", None)
                mask = getattr(mask_obj, "mask_u8", None) if mask_obj is not None else None
                if mask is None or H is None:
                    return False, 0.0
                q = self._finite_point2(point)
                if q is None:
                    return False, 0.0
                px = self.d56.board_to_pixel(H, float(q[0]), float(q[1]))
                if px is None:
                    return False, 0.0
                arr = np.asarray(mask)
                h, w = arr.shape[:2]
                u = int(round(float(px[0])))
                v = int(round(float(px[1])))
                r = max(1, int(radius_px))
                x0, x1 = max(0, u-r), min(w, u+r+1)
                y0, y1 = max(0, v-r), min(h, v+r+1)
                if x1 <= x0 or y1 <= y0:
                    return False, 0.0
                ratio = float(np.mean(arr[y0:y1, x0:x1] > 0))
                return bool(ratio >= float(min_ratio)), ratio
            except Exception:
                return False, 0.0

        def _v56_signature_from_obj(self, semantic, plan, kind=None):
            if plan is None:
                return None
            paths = {}
            plan_kind = str(kind or "").upper()
            try:
                if isinstance(plan, dict):
                    plan_kind = str(plan.get("v56_mode") or plan.get("kind") or plan_kind).upper()
                    arm_points = plan.get("arm_points")
                else:
                    plan_kind = str(getattr(plan, "v56_mode", None) or getattr(plan, "kind", None) or getattr(plan, "action", None) or plan_kind).upper()
                    arm_points = getattr(plan, "arm_points", None)
                if isinstance(arm_points, dict):
                    for arm_key in sorted(arm_points):
                        item = arm_points.get(arm_key)
                        if not isinstance(item, dict):
                            continue
                        src = self._finite_point2(item.get("source_board", item.get("grip_board")))
                        dst = self._finite_point2(item.get("target_board", item.get("end_board")))
                        if src is not None and dst is not None:
                            paths[str(arm_key)] = (src, dst)
                if not paths:
                    if isinstance(plan, dict):
                        src_raw = plan.get("source_board", plan.get("grip_board"))
                        dst_raw = plan.get("target_board", plan.get("end_board"))
                    else:
                        src_raw = getattr(plan, "source_board", None)
                        if src_raw is None:
                            src_raw = getattr(plan, "grip_board", None)
                        dst_raw = getattr(plan, "target_board", None)
                        if dst_raw is None:
                            dst_raw = getattr(plan, "end_board", None)
                    if isinstance(src_raw, dict) and isinstance(dst_raw, dict):
                        for arm_key in sorted(set(src_raw) & set(dst_raw)):
                            src = self._finite_point2(src_raw.get(arm_key))
                            dst = self._finite_point2(dst_raw.get(arm_key))
                            if src is not None and dst is not None:
                                paths[str(arm_key)] = (src, dst)
                    else:
                        src = self._finite_point2(src_raw)
                        dst = self._finite_point2(dst_raw)
                        if src is not None and dst is not None:
                            paths["main"] = (src, dst)
                if not paths:
                    if isinstance(plan, dict):
                        src = self._finite_point2(plan.get("grip_board_xy"))
                        dst = self._finite_point2(plan.get("target_board_xy"))
                    else:
                        src = self._finite_point2(getattr(plan, "grip_board_xy", None))
                        dst = self._finite_point2(getattr(plan, "target_board_xy", None))
                    if src is not None and dst is not None:
                        paths["main"] = (src, dst)
            except Exception:
                return None
            if not paths:
                return None
            return {
                "semantic": str(semantic or "").upper(),
                "kind": plan_kind,
                "paths": {
                    k: {
                        "source": [float(v[0][0]), float(v[0][1])],
                        "target": [float(v[1][0]), float(v[1][1])],
                    }
                    for k, v in paths.items()
                },
            }

        def _v56_signature_from_locked(self, locked):
            if locked is None:
                return None
            semantic = str(self.pending_semantic or self.semantic_selected or getattr(locked, "action", "") or "").upper()
            if semantic == "ALIGN" or str(getattr(locked, "action", "")).upper() == "ALIGN":
                try:
                    diagnostics = dict(getattr(locked, "diagnostics", {}) or {})
                    token = str(diagnostics.get("align_runtime_token", ""))
                    with self.align_runtime_lock:
                        runtime = copy.deepcopy(self.align_runtime.get(token)) if token else None
                    if isinstance(runtime, dict):
                        if str(runtime.get("kind", "")) == "ARM2_CORRECTION":
                            return self._v56_signature_from_obj("ALIGN", runtime.get("action_plan"), runtime.get("kind"))
                        if str(runtime.get("kind", "")) == "DUAL_WAIST_TOP":
                            return self._v56_signature_from_obj("ALIGN", runtime.get("motion"), runtime.get("kind"))
                except Exception:
                    pass
            action = str(getattr(locked, "action", "") or "").upper()
            semantic_map = {
                "D55_PRESS_SWEEP": "PRESS_SWEEP",
                "D58_CIRC_POSITION": "POSITION_ADJUST",
                "D54_OUTER_PULL": "OUTER_PULL",
                "D56_WAIST_LIFT_LAYDOWN": "WAIST_PULL_LAYDOWN",
                "WAIST_PULL_LAYDOWN": "WAIST_PULL_LAYDOWN",
                "BASKET_GRASP": "BASKET_GRASP",
            }
            semantic = semantic_map.get(action, semantic)
            return self._v56_signature_from_obj(semantic, getattr(locked, "plan", None))

        def _v56_signatures_similar(self, a, b):
            if not isinstance(a, dict) or not isinstance(b, dict):
                return False
            if str(a.get("semantic", "")) != str(b.get("semantic", "")):
                return False
            ka, kb = str(a.get("kind", "")), str(b.get("kind", ""))
            if ka and kb and ka != kb:
                return False
            pa, pb = dict(a.get("paths", {}) or {}), dict(b.get("paths", {}) or {})
            if set(pa) != set(pb) or not pa:
                return False
            for key in pa:
                sa = self._finite_point2(pa[key].get("source"))
                ta = self._finite_point2(pa[key].get("target"))
                sb = self._finite_point2(pb[key].get("source"))
                tb = self._finite_point2(pb[key].get("target"))
                if any(x is None for x in (sa, ta, sb, tb)):
                    return False
                if float(np.linalg.norm(sa - sb)) > float(self._v56_reject_source_tol_mm):
                    return False
                if float(np.linalg.norm(ta - tb)) > float(self._v56_reject_target_tol_mm):
                    return False
                va, vb = ta - sa, tb - sb
                la, lb = float(np.linalg.norm(va)), float(np.linalg.norm(vb))
                if abs(la - lb) > float(self._v56_reject_move_tol_mm):
                    return False
                if la >= 3.0 and lb >= 3.0:
                    dot = float(np.clip(np.dot(va/la, vb/lb), -1.0, 1.0))
                    angle = float(math.degrees(math.acos(dot)))
                    if angle > float(self._v56_reject_angle_tol_deg):
                        return False
                elif max(la, lb) >= 3.0:
                    return False
            return True

        def _v56_plan_obj_rejected(self, semantic, plan, kind=None):
            if str(getattr(self, "_prepare_origin", "")).upper() != "REJUDGE":
                return False
            sig = self._v56_signature_from_obj(semantic, plan, kind)
            if sig is None:
                return False
            return any(self._v56_signatures_similar(sig, old) for old in list(self._v56_rejected_plans))

        def _v56_capture_rejected_plan(self):
            with self.state_lock:
                locked = self.locked
            if locked is None or not bool(getattr(locked, "plan_ok", False)):
                return
            sig = self._v56_signature_from_locked(locked)
            if sig is None:
                return
            if not any(self._v56_signatures_similar(sig, old) for old in self._v56_rejected_plans):
                self._v56_rejected_plans.append(copy.deepcopy(sig))
                self._v56_rejected_plans = self._v56_rejected_plans[-int(self._v56_rejected_max):]
            print(f"[REJUDGE-V56-BLACKLIST] semantic={sig.get('semantic')} kind={sig.get('kind') or '-'} paths={list(sig.get('paths',{}))} count={len(self._v56_rejected_plans)}")

        def _rejudge(self):
            with self.state_lock:
                if self.motion_busy or self.inference_busy:
                    print("[REJUDGE] blocked while worker is active")
                    return
                if self.recorder.review_pending():
                    print("[REJUDGE] blocked: complete review and Y/N first")
                    return
            with self._auto_judge_lock:
                if bool(self._auto_judge_busy):
                    print("[REJUDGE] blocked while AUTO-JUDGE is active")
                    return
            self._v56_capture_rejected_plan()
            if bool(getattr(self, "_v110_inference_only", False)):
                try:
                    self._invalidate_for_new_action("V110_MODEL_REJUDGE")
                except Exception as exc:
                    print(f"[MODEL-REJUDGE-V110-WARN] invalidate failed: {exc!r}")
                with self.state_lock:
                    self.locked = None
                    self.display_image = None
                    self.semantic_selected = None
                    self.pending_semantic = None
                    self.selected_action = None
                    self._human_selected = None
                    self.status = "MODEL REJUDGE: FRESH SNAPSHOT / MODEL JUDGE"
                self._prepare_origin = "REJUDGE"
                print("[MODEL-REJUDGE-V110] fresh snapshot -> active model judgement")
                threading.Thread(
                    target=self._auto_judge_next,
                    args=("manual_rejudge",),
                    name="bottom-vla-v110-model-rejudge",
                    daemon=True,
                ).start()
                return
            target = str(self.semantic_selected or self.auto_recommended or "").upper()
            if target in {"", "REJUDGE"}:
                with self.state_lock:
                    self.status = "REJUDGE BLOCKED: NO CONCRETE ACTION TARGET"
                print("[REJUDGE] blocked: no concrete action target")
                return
            return super()._rejudge()

        def _v56_enforce_rejudge_backstop(self):
            if str(getattr(self, "_prepare_origin", "")).upper() != "REJUDGE":
                return
            with self.state_lock:
                locked = self.locked
            if locked is None or not bool(getattr(locked, "plan_ok", False)):
                return
            sig = self._v56_signature_from_locked(locked)
            if sig is None or not any(self._v56_signatures_similar(sig, old) for old in self._v56_rejected_plans):
                return
            reason = "V56_REJUDGE_REJECTED_PLAN_REPEATED"
            try:
                locked.plan_ok = False
                locked.reason = reason
                locked.planner_failure = reason
                locked.diagnostics = dict(getattr(locked, "diagnostics", {}) or {})
                locked.diagnostics["v56_rejudge_same_plan_blocked"] = True
                token = str(locked.diagnostics.get("align_runtime_token", ""))
                if token:
                    with self.align_runtime_lock:
                        self.align_runtime.pop(token, None)
            except Exception:
                pass
            with self.state_lock:
                self.locked = locked
                self.status = "REJUDGE: PREVIOUSLY REJECTED PLAN BLOCKED"
            fields = {"pending_executable": False, "training_eligible": False, "planner_failure": reason, "v56_rejudge_same_plan_blocked": True}
            try:
                with self.recorder.lock:
                    if isinstance(getattr(self.recorder, "current_observation", None), dict):
                        v23._patch_json_record(self.recorder.current_observation, fields)
                    if isinstance(getattr(self.recorder, "current_decision", None), dict):
                        v23._patch_json_record(self.recorder.current_decision, fields)
            except Exception:
                pass
            print("[REJUDGE-V56-BLOCK] generated plan is effectively the same as a rejected frozen plan; ENTER disabled")

        def _v56_position_rejudge_alternative(self, frame, mask, H, config, args, plan):
            if plan is None or not bool(getattr(plan, "ok", False)) or not self._v56_plan_obj_rejected("POSITION_ADJUST", plan):
                return plan
            center = self._finite_point2(getattr(plan, "center_board", None))
            u = self._finite_point2(getattr(plan, "pull_unit", None))
            grip0 = self._finite_point2(getattr(plan, "grip_board", None))
            if center is None or u is None or grip0 is None:
                return plan
            un = float(np.linalg.norm(u))
            if un < 1e-6:
                return plan
            u = u / un
            samples_fn = getattr(self.d58, "_d58_robust_board_points", None)
            contact_fn = getattr(self.d58, "_d58_contact_candidates", None)
            if not callable(samples_fn) or not callable(contact_fn):
                return plan
            try:
                samples = np.asarray(samples_fn(mask, H, int(getattr(args, "mask_sample_stride_px", 3))), np.float64).reshape(-1, 2)
            except Exception:
                return plan
            if len(samples) < 40:
                return plan
            rel = samples - center[None, :]
            lead = rel @ u
            lateral = np.abs(rel[:, 0] * (-u[1]) + rel[:, 1] * u[0])
            order = np.argsort(-(lead - 0.18*lateral))
            request = max(float(getattr(plan, "move_mm", 0.0) or 0.0), float(getattr(args, "d58_min_move_mm", 30.0)))
            tested = 0
            for idx in order[::max(1, len(order)//80)]:
                preferred = np.asarray(samples[int(idx)], np.float64)
                if float(np.linalg.norm(preferred - grip0)) < 36.0:
                    continue
                try:
                    contact, why = contact_fn(mask, H, config, center, u, request, args, preferred_vertex=preferred)
                except Exception:
                    continue
                tested += 1
                if contact is None:
                    continue
                candidate = copy.deepcopy(plan)
                candidate.selected_arm = str(contact["arm"])
                candidate.grip_board = np.asarray(contact["grip"], np.float32)
                candidate.end_board = np.asarray(contact["end"], np.float32)
                candidate.move_mm = float(contact["move_mm"])
                candidate.grip_inset_mm = float(contact["inset_mm"])
                candidate.contour_vertex = preferred.astype(np.float32)
                candidate.target_source = f"V56_REJUDGE_ALTERNATE::{getattr(candidate,'target_source','NONE')}"
                if self._v56_plan_obj_rejected("POSITION_ADJUST", candidate):
                    continue
                overlay_fn = getattr(self.d58, "make_d58_overlay", None)
                if callable(overlay_fn):
                    try:
                        candidate.overlay = overlay_fn(frame.copy(), mask, candidate, H, config)
                    except Exception:
                        pass
                print(f"[REJUDGE-V56-POSITION] rejected grip={np.round(grip0,1).tolist()} -> alternate={np.round(np.asarray(candidate.grip_board),1).tolist()} tested={tested}")
                return candidate
            print(f"[REJUDGE-V56-POSITION] no distinct safe alternative after {tested} contact trials; generic same-plan backstop will block")
            return plan

        def _v56_press_arm_radius(self, plan, arm_key):
            try:
                item = dict(getattr(plan, "arm_points", {}) or {}).get(arm_key, {})
                radii = []
                for key in ("hover", "down", "sweep", "lift"):
                    q = np.asarray(item.get(key), np.float64).reshape(-1)
                    if q.size >= 2 and np.all(np.isfinite(q[:2])):
                        radii.append(float(math.hypot(float(q[0]), float(q[1]))))
                return max(radii) if radii else float("inf")
            except Exception:
                return float("inf")

        def _v56_shift_press_arm_horizontal(self, plan, obs, H, config, cfg, args, arm_key, dx):
            try:
                candidate = copy.deepcopy(plan)
                item = dict(candidate.arm_points[arm_key])
                src = self._finite_point2(item.get("source_board"))
                dst = self._finite_point2(item.get("target_board"))
                if src is None or dst is None:
                    return None
                src2 = src + np.asarray([float(dx), 0.0], np.float64)
                dst2 = dst + np.asarray([float(dx), 0.0], np.float64)
                support_ok, support_ratio = self._v56_mask_support_at_board(obs, H, src2, radius_px=5, min_ratio=0.62)
                if not support_ok:
                    return None
                pose_fn = getattr(self.d55, "_roarm_pose_for_board", None)
                if not callable(pose_fn):
                    return None
                clear_z = float(getattr(args, "b34_clear_z", -25.0))
                press_z = self._v47_press_z_for_arm(args, arm_key)
                mid_ratio = float(np.clip(getattr(args, "b34_micro_mid_ratio", 0.55), 0.30, 0.75))
                mid2 = src2 + mid_ratio * (dst2 - src2)
                hover = pose_fn(config, cfg, arm_key, src2, clear_z)
                down = pose_fn(config, cfg, arm_key, src2, press_z)
                mid = pose_fn(config, cfg, arm_key, mid2, press_z)
                sweep = pose_fn(config, cfg, arm_key, dst2, press_z)
                lift = pose_fn(config, cfg, arm_key, dst2, clear_z)
                flat = np.asarray(list(hover[:3]) + list(down[:3]) + list(mid[:3]) + list(sweep[:3]) + list(lift[:3]), np.float64)
                if not np.all(np.isfinite(flat)):
                    return None
                item.update({
                    "source_board": (float(src2[0]), float(src2[1])),
                    "target_board": (float(dst2[0]), float(dst2[1])),
                    "hover": hover,
                    "down": down,
                    "mid": mid,
                    "sweep": sweep,
                    "lift": lift,
                    "v56_horizontal_reach_shift_mm": float(dx),
                    "v56_shifted_source_mask_ratio": float(support_ratio),
                })
                candidate.arm_points[arm_key] = item
                metrics = dict(getattr(candidate, "metrics", {}) or {})
                metrics[f"{arm_key}_v56_horizontal_shift_mm"] = float(dx)
                metrics[f"{arm_key}_press_distance_mm"] = float(np.linalg.norm(dst2-src2))
                candidate.metrics = metrics
                leg_lock = self._v70_press_leg_lock_report(candidate, obs)
                candidate.metrics["d55v70_leg_lock_after_reach_shift"] = copy.deepcopy(leg_lock)
                if bool(leg_lock.get("enforced", False)) and not bool(leg_lock.get("ok", False)):
                    return None
                ok, _ = self._v48_validate_press_plan(candidate, obs, args)
                if not bool(ok):
                    return None
                return candidate
            except Exception:
                return None

        def _v88_press_same_wrinkle_pair_ok(self, plan):
            if plan is None or not bool(getattr(plan, "ok", False)):
                return False, "PLAN_INVALID"
            metrics = dict(getattr(plan, "metrics", {}) or {})
            ridge = metrics.get("d55v47_ridge", {}) if isinstance(metrics.get("d55v47_ridge", {}), dict) else {}
            center = self._finite_point2(ridge.get("center_board"))
            tangent = self._finite_point2(ridge.get("tangent_board"))
            normal = self._finite_point2(ridge.get("normal_board"))
            span = float(ridge.get("span_mm", 0.0) or 0.0)
            if center is None or tangent is None or normal is None:
                return False, "RIDGE_GEOMETRY_MISSING"
            tn = float(np.linalg.norm(tangent))
            nn = float(np.linalg.norm(normal))
            if tn < 1e-6 or nn < 1e-6:
                return False, "RIDGE_AXIS_INVALID"
            tangent = tangent / tn
            normal = normal / nn
            points = dict(getattr(plan, "arm_points", {}) or {})
            sources = []
            for key in ("arm1", "arm2"):
                src = self._finite_point2((points.get(key, {}) or {}).get("source_board"))
                if src is None:
                    return False, f"{key.upper()}_SOURCE_MISSING"
                rel = src - center
                sources.append((key, src, float(np.dot(rel, normal)), float(np.dot(rel, tangent))))
            if sources[0][2] * sources[1][2] >= 0.0:
                return False, "SOURCES_NOT_OPPOSITE_RIDGE_SIDES"
            along_delta = abs(float(sources[0][3] - sources[1][3]))
            along_mid = 0.5 * float(sources[0][3] + sources[1][3])
            max_mid = float(self._v88_press_pair_mid_span_gain) * max(1.0, span) + 18.0
            if along_delta > float(self._v88_press_pair_max_along_delta_mm):
                return False, f"PAIR_ALONG_MISMATCH_{along_delta:.1f}"
            if abs(along_mid) > max_mid:
                return False, f"PAIR_OUTSIDE_SELECTED_RIDGE_SPAN_{along_mid:.1f}"
            return True, {"along_delta_mm": float(along_delta), "along_mid_mm": float(along_mid)}

        def _v56_finalize_press_plan(self, plan, obs, H, config, cfg, args):
            if plan is None or not bool(getattr(plan, "ok", False)):
                return plan
            current = copy.deepcopy(plan)
            initial_leg_lock = self._v70_press_leg_lock_report(current, obs)
            current.metrics = dict(getattr(current, "metrics", {}) or {})
            current.metrics["d55v70_leg_lock_initial"] = copy.deepcopy(initial_leg_lock)
            if bool(initial_leg_lock.get("enforced", False)) and not bool(initial_leg_lock.get("ok", False)):
                print(
                    f"[D55-V70-LEG-FINALIZE-REJECT] leg={initial_leg_lock.get('ridge_leg')} "
                    f"reason={initial_leg_lock.get('reason')}"
                )
                return None
            before = {k: self._v56_press_arm_radius(current, k) for k in ("arm1", "arm2")}
            for arm_key in sorted(before, key=lambda k: before[k], reverse=True):
                base_radius = self._v56_press_arm_radius(current, arm_key)
                if not math.isfinite(base_radius) or base_radius <= float(self._v56_press_reach_trigger_mm):
                    continue
                best = None
                best_radius = base_radius
                max_shift = int(round(float(self._v56_press_horizontal_shift_max_mm)))
                step = max(5, int(round(float(self._v56_press_horizontal_shift_step_mm))))
                for mag in range(step, max_shift + step, step):
                    for sign in (-1.0, 1.0):
                        dx = float(sign * mag)
                        cand = self._v56_shift_press_arm_horizontal(current, obs, H, config, cfg, args, arm_key, dx)
                        if cand is None:
                            continue
                        radius = self._v56_press_arm_radius(cand, arm_key)
                        if radius < best_radius - 1e-6:
                            best = cand
                            best_radius = radius
                    if best is not None and best_radius <= float(self._v56_press_reach_target_mm):
                        break
                if best is not None and best_radius <= base_radius - 15.0:
                    current = best
                    print(f"[D55-V56-REACH-SHIFT] {arm_key.upper()} radius={base_radius:.1f}->{best_radius:.1f}mm horizontalShift={float(current.metrics.get(f'{arm_key}_v56_horizontal_shift_mm',0.0)):+.1f}mm; other arm unchanged")
                else:
                    print(f"[D55-V56-REACH-WARN] {arm_key.upper()} radius={base_radius:.1f}mm but no safer horizontal translation passed Mask/board/dual validation; keep existing validated plan")

            final_leg_lock = self._v70_press_leg_lock_report(current, obs)
            current.metrics = dict(getattr(current, "metrics", {}) or {})
            current.metrics["d55v70_leg_lock_final"] = copy.deepcopy(final_leg_lock)
            if bool(final_leg_lock.get("enforced", False)) and not bool(final_leg_lock.get("ok", False)):
                print(
                    f"[D55-V70-LEG-FINALIZE-REJECT] post-shift leg={final_leg_lock.get('ridge_leg')} "
                    f"reason={final_leg_lock.get('reason')}"
                )
                return None
            same_ok, same_report = self._v88_press_same_wrinkle_pair_ok(current)
            if not same_ok:
                print(f"[D55-V88-SAME-WRINKLE-REJECT] reason={same_report}")
                return None
            current.metrics["d55v88_same_wrinkle_final_check"] = copy.deepcopy(same_report)
            if not self._v56_plan_obj_rejected("PRESS_SWEEP", current):
                return current

            order = sorted(("arm1", "arm2"), key=lambda k: self._v56_press_arm_radius(current, k), reverse=True)
            alternatives = []
            for arm_key in order:
                for mag in (30.0, 45.0, 60.0, 75.0, 90.0, 110.0):
                    for sign in (-1.0, 1.0):
                        cand = self._v56_shift_press_arm_horizontal(current, obs, H, config, cfg, args, arm_key, sign*mag)
                        if cand is None or self._v56_plan_obj_rejected("PRESS_SWEEP", cand):
                            continue
                        score = self._v56_press_arm_radius(cand, arm_key) + 0.05*mag
                        alternatives.append((score, cand, arm_key, sign*mag))
            if alternatives:
                alternatives.sort(key=lambda row: row[0])
                _, chosen, arm_key, dx = alternatives[0]
                chosen.metrics = dict(getattr(chosen, "metrics", {}) or {})
                chosen.metrics["v56_rejudge_diversified"] = True
                print(f"[REJUDGE-V56-PRESS] previous plan rejected -> distinct safe horizontal alternative {arm_key.upper()} shift={dx:+.1f}mm")
                return chosen
            print("[REJUDGE-V56-PRESS] previous PRESS plan rejected and no distinct safe horizontal alternative exists; continue to next planner fallback")
            return None

        def _v56_align_arm_reach(self, arm_key, point, final_target=False):
            try:
                fn = getattr(self.align, "_c28_arm_point_ok", None)
                if not callable(fn):
                    return False, "REACH_API_MISSING", float("inf")
                return fn(self.config, arm_key, np.asarray(point, np.float32), self.align_args, final_target=bool(final_target))
            except Exception as exc:
                return False, f"REACH_ERROR:{exc!r}", float("inf")

        def _v56_align_board_safe(self, point, margin=8.0):
            q = self._finite_point2(point)
            if q is None:
                return False
            try:
                return bool(
                    float(self.align_cfg.board_x_min)+margin <= q[0] <= float(self.align_cfg.board_x_max)-margin
                    and float(self.align_cfg.board_y_min)+margin <= q[1] <= float(self.align_cfg.board_y_max)-margin
                )
            except Exception:
                return False

        def _v56_align_safe_progress(self, arm_key, source, desired):
            source = np.asarray(source, np.float64).reshape(2)
            desired = np.asarray(desired, np.float64).reshape(2)
            for frac in np.linspace(1.0, 0.10, 19):
                q = source + (desired-source)*float(frac)
                if not self._v56_align_board_safe(q, 8.0):
                    continue
                ok, _, _ = self._v56_align_arm_reach(arm_key, q, final_target=True)
                if ok:
                    return q, float(frac)
            return None, 0.0

        def _v66_align_select_arm1_pivot(self, obs, H, geom, moving_source, moving_target, move_mm):
            if not isinstance(geom, dict):
                return None
            angle = abs(float(geom.get("angle_error_deg", 0.0)))
            move_mm = float(move_mm)
            if angle < float(self._v66_align_pivot_angle_min_deg):
                return None
            if move_mm > float(self._v66_align_pivot_move_max_mm):
                print(
                    f"[ALIGN-V66-PIVOT-SKIP-LARGE] move={move_mm:.1f}mm > "
                    f"{float(self._v66_align_pivot_move_max_mm):.1f}mm; keep V64 single-arm CENTER"
                )
                return None
            waist = self._finite_point2(geom.get("waist_center"))
            lower = self._finite_point2(geom.get("lower_point"))
            src = self._finite_point2(moving_source)
            tgt = self._finite_point2(moving_target)
            if waist is None or lower is None or src is None or tgt is None:
                return None
            axis = np.asarray(lower - waist, np.float64)
            axis_len = float(np.linalg.norm(axis))
            if not math.isfinite(axis_len) or axis_len < 40.0:
                return None
            axis_u = axis / axis_len
            lateral_u = np.asarray([-axis_u[1], axis_u[0]], np.float64)
            if float(lateral_u[0]) < 0.0:
                lateral_u = -lateral_u

            trials = []
            for t in self._v66_align_pivot_t_values:
                center = waist + axis * float(t)
                for lateral in self._v66_align_pivot_lateral_offsets_mm:
                    q = center + lateral_u * float(lateral)
                    if not self._v56_align_board_safe(q, 8.0):
                        continue
                    seg = np.asarray(tgt - src, np.float64)
                    seg_den = float(np.dot(seg, seg))
                    if seg_den > 1e-9:
                        seg_t = float(np.clip(np.dot(q - src, seg) / seg_den, 0.0, 1.0))
                        nearest = src + seg * seg_t
                    else:
                        nearest = src
                    if float(np.linalg.norm(q - nearest)) < float(self._v66_align_pivot_min_separation_mm):
                        continue
                    support_ok, support_ratio = self._v56_mask_support_at_board(
                        obs, H, q, radius_px=6, min_ratio=0.82
                    )
                    if not support_ok:
                        continue
                    rep = self._v53_mask_point_report(obs, q)
                    edge_depth = rep.get("edge_depth_mm") if isinstance(rep, dict) else None
                    if (
                        not bool(rep.get("ok", False))
                        or edge_depth is None
                        or float(edge_depth) < float(self._v66_align_pivot_hard_inset_mm)
                    ):
                        continue
                    reach_ok, reach_why, reach_r = self._v56_align_arm_reach("arm1", q, final_target=False)
                    if not bool(reach_ok):
                        continue
                    point_ok = getattr(self.align, "_align3_dual_point_ok", None)
                    if callable(point_ok):
                        try:
                            pivot_ok, pivot_why, pivot_r = point_ok(
                                self.config, self.align_cfg, self.align_args, "arm1", q
                            )
                        except Exception:
                            pivot_ok = False
                            pivot_why = "PIVOT_GUARD_ERROR"
                            pivot_r = reach_r
                        if not bool(pivot_ok):
                            continue
                        reach_r = max(float(reach_r), float(pivot_r))
                    score = (
                        120.0 * abs(float(t) - float(self._v66_align_pivot_preferred_t))
                        + 0.22 * abs(float(lateral) - 75.0)
                        + 0.04 * float(reach_r)
                        + 20.0 * (1.0 - float(support_ratio))
                        + 0.20 * max(0.0, 55.0 - float(edge_depth))
                    )
                    trials.append((
                        float(score), np.asarray(q, np.float64), float(t), float(lateral),
                        float(support_ratio), float(edge_depth), float(reach_r), str(reach_why),
                    ))
            if not trials:
                print(
                    f"[ALIGN-V66-PIVOT-SKIP] move={move_mm:.1f}mm angle={angle:.1f}deg; "
                    f"no ARM1 upper-body interior anchor passed Mask>={float(self._v66_align_pivot_hard_inset_mm):.0f}mm/reach/separation guards; "
                    "keep V64 single-arm CENTER"
                )
                return None
            trials.sort(key=lambda row: row[0])
            _, q, t, lateral, support_ratio, edge_depth, reach_r, _ = trials[0]
            print(
                f"[ALIGN-V66-PIVOT-SELECT] fine move={move_mm:.1f}mm angle={angle:.1f}deg "
                f"ARM1 anchor={np.round(q,1).tolist()} t={t:.2f} lateral={lateral:.1f}mm "
                f"edgeDepth={edge_depth:.1f}mm mask={support_ratio:.2f} reachR={reach_r:.1f}mm"
            )
            return {
                "arm_key": "arm1",
                "source_board": [float(q[0]), float(q[1])],
                "t": float(t),
                "lateral_mm": float(lateral),
                "mask_ratio": float(support_ratio),
                "edge_depth_mm": float(edge_depth),
                "reach_radius_mm": float(reach_r),
            }

        def _v67_align_intersection_pivot_plan(self, obs, H, seam_x, geom, base_plan):
            if not isinstance(geom, dict):
                return None
            waist = self._finite_point2(geom.get("waist_center"))
            lower = self._finite_point2(geom.get("lower_point"))
            if waist is None or lower is None:
                return None
            axis = lower - waist
            axis_len2 = float(np.dot(axis, axis))
            axis_len = math.sqrt(axis_len2) if axis_len2 > 0.0 else 0.0
            if axis_len2 < 1e-6 or axis_len < 1e-6:
                return None

            angle_signed = float(geom.get("angle_error_deg", 0.0))
            angle = abs(angle_signed)
            mid_gap = abs(float(geom.get("mid_error_mm", 0.0)))
            if mid_gap > float(self._v75_align_pivot_distance_gate_mm):
                print(
                    f"[ALIGN-V75-PIVOT-DEFER] midGap={mid_gap:.1f}mm > {self._v75_align_pivot_distance_gate_mm:.1f}mm; "
                    "distance is still large, keep one-arm CENTER first"
                )
                return None
            if angle < float(self._v75_align_pivot_angle_min_deg):
                return None

            axis_u = axis / axis_len
            if abs(float(axis[0])) >= 1e-6:
                raw_t = (float(seam_x) - float(waist[0])) / float(axis[0])
            else:
                raw_t = 0.40
            raw_t = float(np.clip(raw_t, 0.12, 0.82))
            raw_anchor = waist + axis * raw_t
            pull_sign = -1.0 if angle_signed >= 0.0 else 1.0

            samples = self._v56_mask_board_samples(obs, H, max_points=4200)
            if samples is None:
                return None
            trials = []
            shift_values = (float(self._v73_align_anchor_shift_mm), 45.0, 30.0)
            for q in np.asarray(samples, np.float64).reshape(-1, 2):
                if self._v67_prepare_cancel_requested():
                    return None
                proj = float(np.dot(q - waist, axis) / axis_len2)
                if proj < float(self._v88_align_pivot_arm1_body_t_min) or proj > float(self._v88_align_pivot_arm1_body_t_max):
                    continue
                if not self._v56_align_board_safe(q, 8.0):
                    continue
                src_rep = self._v53_mask_point_report(obs, q)
                src_depth = src_rep.get("edge_depth_mm") if isinstance(src_rep, dict) else None
                if not bool((src_rep or {}).get("ok", False)) or src_depth is None or float(src_depth) < float(self._v67_align_pivot_hard_inset_mm):
                    continue
                src_support_ok, src_ratio = self._v56_mask_support_at_board(obs, H, q, radius_px=7, min_ratio=0.80)
                if not src_support_ok:
                    continue
                src_ok, _, src_r = self._v56_align_arm_reach("arm1", q, final_target=False)
                if not src_ok:
                    continue

                direction = -1.0 if raw_t <= proj else 1.0
                semantic = "WAIST" if direction < 0.0 else "CROTCH"
                for shift_mm in shift_values:
                    anchor_t = float(np.clip(raw_t + direction * float(shift_mm) / axis_len, 0.04, 0.92))
                    anchor = waist + axis * anchor_t
                    support_ok, support_ratio = self._v56_mask_support_at_board(obs, H, anchor, radius_px=7, min_ratio=0.80)
                    rep = self._v53_mask_point_report(obs, anchor)
                    edge_depth = rep.get("edge_depth_mm") if isinstance(rep, dict) else None
                    if (
                        not support_ok
                        or not bool((rep or {}).get("ok", False))
                        or edge_depth is None
                        or float(edge_depth) < float(self._v67_align_pivot_hard_inset_mm)
                        or not self._v56_align_board_safe(anchor, 8.0)
                    ):
                        continue
                    arm2_ok, _, arm2_r = self._v56_align_arm_reach("arm2", anchor, final_target=False)
                    if not arm2_ok:
                        continue
                    sep = float(np.linalg.norm(q - anchor))
                    horizontal_lever = abs(float(q[0]) - float(anchor[0]))
                    if sep < float(self._v67_align_pivot_min_separation_mm) or sep > 340.0:
                        continue
                    if horizontal_lever < float(self._v75_align_pivot_min_horizontal_lever_mm):
                        continue

                    angle_pull = horizontal_lever * abs(math.tan(math.radians(angle)))
                    desired_pull = float(np.clip(
                        angle_pull
                        + float(self._v75_align_pivot_gap_gain) * mid_gap
                        + float(self._v75_align_pivot_extra_mm),
                        float(self._v75_align_pivot_pull_min_mm),
                        float(self._v75_align_pivot_pull_max_mm),
                    ))
                    desired_target = q + np.asarray([0.0, pull_sign * desired_pull], np.float64)
                    target, frac = self._v56_align_safe_progress("arm1", q, desired_target)
                    if target is None:
                        continue
                    move = float(np.linalg.norm(target - q))
                    if move < float(self._v75_align_pivot_min_applied_mm):
                        continue
                    if not self._v56_align_board_safe(target, 8.0):
                        continue
                    tgt_ok, _, tgt_r = self._v56_align_arm_reach("arm1", target, final_target=True)
                    if not tgt_ok:
                        continue

                    score = (
                        1.8 * abs(desired_pull - move)
                        + 0.55 * abs(sep - float(self._v73_align_pivot_preferred_sep_mm))
                        + 55.0 * abs(proj - float(self._v88_align_pivot_arm1_body_t_preferred))
                        + 0.35 * abs(float(anchor[0]) - float(seam_x))
                        + 0.18 * float(max(src_r, tgt_r, arm2_r))
                        - 0.15 * min(float(src_depth), 80.0)
                        + 18.0 * (1.0 - float(src_ratio))
                        + 8.0 * (1.0 - float(support_ratio))
                    )
                    trials.append((
                        score, q.copy(), target.copy(), desired_target.copy(), anchor.copy(), anchor_t, semantic,
                        shift_mm, sep, move, desired_pull, angle_pull, float(frac), horizontal_lever,
                        float(src_depth), float(src_ratio), float(max(src_r, tgt_r)), float(edge_depth),
                        float(support_ratio), float(arm2_r),
                    ))
                    break

            if not trials:
                print(
                    f"[ALIGN-V75-PIVOT-FALLBACK] close gap={mid_gap:.1f}mm angle={angle:.1f}deg, "
                    "but no safe ARM2-hold/ARM1-vertical pair; keep one-arm CENTER"
                )
                return None
            trials.sort(key=lambda row: row[0])
            (
                _, source, target, requested_target, anchor, anchor_t, anchor_semantic, anchor_shift, sep, move,
                desired_pull, angle_pull, target_fraction, horizontal_lever, src_depth, src_ratio, arm1_r,
                edge_depth, support_ratio, arm2_r,
            ) = trials[0]
            plan = copy.deepcopy(base_plan) if isinstance(base_plan, dict) else {}
            plan.update({
                "kind": "CENTER",
                "v56_mode": "CENTER_VECTOR_PRESS",
                "arm_key": "arm1",
                "source_board": [float(source[0]), float(source[1])],
                "target_board": [float(target[0]), float(target[1])],
                "requested_target_board": [float(requested_target[0]), float(requested_target[1])],
                "move_mm": float(move),
                "target_fraction": float(target_fraction),
                "angle_error_deg": float(angle_signed),
                "v67_intersection_pivot": True,
                "v67_pivot_arm_key": "arm2",
                "v67_pivot_source_board": [float(anchor[0]), float(anchor[1])],
                "v67_intersection_t": float(raw_t),
                "v67_rotation_deg": float(angle_signed),
                "v67_pivot_edge_depth_mm": float(edge_depth),
                "v67_pivot_mask_ratio": float(support_ratio),
                "v67_pivot_reach_radius_mm": float(arm2_r),
                "v67_arm1_source_edge_depth_mm": float(src_depth),
                "v67_arm1_source_mask_ratio": float(src_ratio),
                "v67_arm1_reach_radius_mm": float(arm1_r),
                "v67_pivot_separation_mm": float(sep),
                "v67_corrects_angle_and_distance": True,
                "v73_raw_intersection_board": [float(raw_anchor[0]), float(raw_anchor[1])],
                "v73_anchor_t": float(anchor_t),
                "v73_anchor_semantic": str(anchor_semantic),
                "v73_anchor_shift_mm": float(anchor_shift),
                "v75_distance_gated_pivot": True,
                "v75_mid_gap_mm": float(mid_gap),
                "v75_distance_gate_mm": float(self._v75_align_pivot_distance_gate_mm),
                "v75_vertical_pull": True,
                "v75_pull_sign": float(pull_sign),
                "v75_desired_pull_mm": float(desired_pull),
                "v75_angle_pull_mm": float(angle_pull),
                "v75_horizontal_lever_mm": float(horizontal_lever),
            })
            if self._v56_plan_obj_rejected("ALIGN", plan, "ARM2_CORRECTION"):
                return None
            direction_text = "DOWN" if pull_sign < 0.0 else "UP"
            print(
                f"[ALIGN-V75-PIVOT] midGap={mid_gap:.1f}mm angle={angle_signed:+.1f}deg -> ARM2 HOLD={np.round(anchor,1).tolist()} "
                f"({anchor_shift:.0f}mm toward {anchor_semantic}); ARM1 {np.round(source,1).tolist()} -> "
                f"{np.round(target,1).tolist()} {direction_text} move={move:.1f}/{desired_pull:.1f}mm "
                f"anglePart={angle_pull:.1f}mm leverX={horizontal_lever:.1f}mm sep={sep:.1f}mm "
                f"arm1BodyT={float(np.dot(source-waist, axis)/axis_len2):.2f}"
            )
            return plan

        def _v102_align_center_t_policy(self, obs, geom):
            values = [float(x) for x in self._v102_align_center_t_values]
            cap = 0.50
            crotch_t = None
            try:
                waist = self._finite_point2(geom.get("waist_center")) if isinstance(geom, dict) else None
                lower = self._finite_point2(geom.get("lower_point")) if isinstance(geom, dict) else None
                crotch = self._finite_point2(self._pose_board_points(obs).get("crotch"))
                if waist is not None and lower is not None and crotch is not None:
                    axis = lower - waist
                    den = float(np.dot(axis, axis))
                    if den > 1e-6:
                        raw_t = float(np.dot(crotch - waist, axis) / den)
                        if 0.38 <= raw_t <= 0.85:
                            crotch_t = raw_t
                            cap = min(cap, max(0.30, raw_t - float(self._v102_align_crotch_margin_t)))
            except Exception:
                pass
            filtered = [t for t in values if t <= cap + 1e-9]
            if not filtered:
                filtered = [min(0.30, cap)]
            preferred = min(float(self._v102_align_center_preferred_t), max(filtered))
            return tuple(filtered), float(cap), crotch_t, float(preferred)

        def _v104_align_asymmetry_report(self, obs, geom):
            report = {"ok": False, "trigger": False, "mode": "NONE", "reason": "UNAVAILABLE"}
            if not isinstance(geom, dict):
                return report
            pts = self._pose_board_points(obs)
            waist = self._finite_point2(geom.get("waist_center"))
            lower = self._finite_point2(geom.get("lower_point"))
            crotch = self._finite_point2(pts.get("crotch"))

            def hem_center(prefix):
                q = self._finite_point2(pts.get(prefix + "_hem_center"))
                if q is not None:
                    return q
                outer = self._finite_point2(pts.get("img_" + prefix + "_hem_outer"))
                inner = self._finite_point2(pts.get("img_" + prefix + "_hem_inner"))
                if outer is not None and inner is not None:
                    return 0.5 * (outer + inner)
                return outer if outer is not None else inner

            left = hem_center("left")
            right = hem_center("right")
            if waist is None or lower is None or crotch is None or left is None or right is None:
                report["reason"] = "POSE_HEM_OR_AXIS_MISSING"
                return report
            axis = lower - waist
            axis_n = float(np.linalg.norm(axis))
            if not math.isfinite(axis_n) or axis_n < 80.0:
                report["reason"] = "AXIS_TOO_SHORT"
                return report
            u = axis / axis_n
            nvec = np.asarray([-u[1], u[0]], np.float64)
            dl = float(np.dot(left - waist, nvec))
            dr = float(np.dot(right - waist, nvec))
            wl = abs(dl)
            wr = abs(dr)
            if min(wl, wr) < 20.0 or dl * dr >= 0.0:
                report["reason"] = "HEMS_NOT_RELIABLY_OPPOSITE"
                return report
            width_diff = abs(wl - wr)
            width_ratio = min(wl, wr) / max(wl, wr, 1e-6)
            len_l = float(np.linalg.norm(left - crotch))
            len_r = float(np.linalg.norm(right - crotch))
            len_diff = abs(len_l - len_r)
            len_ratio = min(len_l, len_r) / max(len_l, len_r, 1e-6)
            width_trigger = bool(
                width_diff >= float(self._v104_align_asym_width_trigger_mm)
                and (width_ratio <= 0.72 or width_diff >= float(self._v104_align_asym_width_trigger_mm) + 20.0)
            )
            length_trigger = bool(
                len_diff >= float(self._v104_align_asym_length_trigger_mm)
                and len_ratio <= 0.75
            )
            mode = "WIDTH" if width_trigger else ("LENGTH" if length_trigger else "NONE")
            report.update({
                "ok": True, "trigger": bool(width_trigger or length_trigger), "mode": mode, "reason": "OK",
                "waist": waist, "crotch": crotch, "left_hem": left, "right_hem": right,
                "axis_u": u, "axis_n": nvec, "left_signed_width_mm": dl, "right_signed_width_mm": dr,
                "left_half_width_mm": wl, "right_half_width_mm": wr, "width_diff_mm": width_diff,
                "width_ratio": width_ratio, "left_leg_len_mm": len_l, "right_leg_len_mm": len_r,
                "leg_len_diff_mm": len_diff, "leg_len_ratio": len_ratio,
            })
            return report

        def _v104_align_asymmetry_plan(self, obs, H, seam_x, geom, native_plan):
            report = self._v104_align_asymmetry_report(obs, geom)
            if not bool(report.get("ok", False)) or not bool(report.get("trigger", False)):
                if bool(report.get("ok", False)):
                    print(
                        f"[ALIGN-V104-ASYM-ACCEPT] widthDiff={float(report.get('width_diff_mm',0.0)):.1f}mm "
                        f"legLenDiff={float(report.get('leg_len_diff_mm',0.0)):.1f}mm -> minor asymmetry accepted"
                    )
                return None, report

            mode = str(report.get("mode", "NONE"))
            crotch = np.asarray(report["crotch"], np.float64)
            left = np.asarray(report["left_hem"], np.float64)
            right = np.asarray(report["right_hem"], np.float64)
            nvec = np.asarray(report["axis_n"], np.float64)
            u = np.asarray(report["axis_u"], np.float64)
            wl = float(report["left_half_width_mm"])
            wr = float(report["right_half_width_mm"])
            dl = float(report["left_signed_width_mm"])
            dr = float(report["right_signed_width_mm"])
            len_l = float(report["left_leg_len_mm"])
            len_r = float(report["right_leg_len_mm"])

            if mode == "WIDTH":
                ratio = float(report["width_ratio"])
                if ratio < float(self._v104_align_asym_narrow_ratio):
                    side = "left" if wl < wr else "right"
                    outward = True
                else:
                    side = "left" if wl > wr else "right"
                    outward = False
                hem = left if side == "left" else right
                signed = dl if side == "left" else dr
                sign = 1.0 if signed >= 0.0 else -1.0
                direction = sign * nvec if outward else -sign * nvec
                raw_move = float(report["width_diff_mm"]) - float(self._v104_align_asym_width_release_mm)
            else:
                side = "left" if len_l < len_r else "right"
                hem = left if side == "left" else right
                direction = u
                outward = True
                raw_move = float(report["leg_len_diff_mm"]) - float(self._v104_align_asym_length_release_mm)

            move_request = float(np.clip(raw_move, self._v104_align_asym_move_min_mm, self._v104_align_asym_move_max_mm))
            signed_side = float(np.dot(hem - np.asarray(report["waist"], np.float64), nvec))
            inward = -1.0 * (1.0 if signed_side >= 0.0 else -1.0) * nvec
            trials = []
            for frac in (0.58, 0.50, 0.66, 0.44):
                base = crotch + (hem - crotch) * float(frac)
                for inset in (0.0, 15.0, 28.0, 40.0):
                    source = base + inward * float(inset)
                    rep = self._v53_mask_point_report(obs, source)
                    edge_depth = rep.get("edge_depth_mm") if isinstance(rep, dict) else None
                    if not bool((rep or {}).get("ok", False)) or edge_depth is None or float(edge_depth) < float(self._v104_align_asym_source_inset_mm):
                        continue
                    support_ok, support = self._v56_mask_support_at_board(obs, H, source, radius_px=6, min_ratio=0.76)
                    if not support_ok or not self._v56_align_board_safe(source, 8.0):
                        continue
                    arm_key = "arm2" if float(source[0]) < float(seam_x) else "arm1"
                    src_ok, _, _ = self._v56_align_arm_reach(arm_key, source, final_target=False)
                    if not src_ok:
                        continue
                    requested_target = source + direction * move_request
                    target, frac_safe = self._v56_align_safe_progress(arm_key, source, requested_target)
                    if target is None:
                        continue
                    move = float(np.linalg.norm(np.asarray(target, np.float64) - source))
                    if move < float(self._v104_align_asym_move_min_mm):
                        continue
                    plan = copy.deepcopy(native_plan) if isinstance(native_plan, dict) else {}
                    plan.update({
                        "kind": "CENTER", "v56_mode": "CENTER_VECTOR_PRESS", "arm_key": arm_key,
                        "source_board": source.astype(float).tolist(),
                        "target_board": np.asarray(target, np.float64).astype(float).tolist(),
                        "requested_target_board": requested_target.astype(float).tolist(),
                        "move_mm": move, "target_fraction": float(frac_safe),
                        "angle_error_deg": float(geom.get("angle_error_deg", 0.0)),
                        "predicted_angle_residual_deg": float(geom.get("angle_error_deg", 0.0)),
                        "predicted_seam_residual_mm": float(geom.get("mid_error_mm", 0.0)),
                        "v104_asymmetry_correction": True,
                        "v104_asymmetry_mode": mode,
                        "v104_asymmetry_side": side.upper(),
                        "v104_asymmetry_outward": bool(outward),
                        "v104_width_diff_mm": float(report.get("width_diff_mm", 0.0)),
                        "v104_leg_len_diff_mm": float(report.get("leg_len_diff_mm", 0.0)),
                        "v104_source_edge_depth_mm": float(edge_depth),
                        "v104_source_mask_ratio": float(support),
                    })
                    trials.append((plan, abs(float(inset) - 15.0), -move))
            if not trials:
                print(
                    f"[ALIGN-V104-ASYM-NO-SAFE-PLAN] mode={mode} widthDiff={float(report.get('width_diff_mm',0.0)):.1f}mm "
                    f"legLenDiff={float(report.get('leg_len_diff_mm',0.0)):.1f}mm; fall through to existing next-action logic"
                )
                return None, report
            trials.sort(key=lambda row: (row[1], row[2]))
            plan = trials[0][0]
            print(
                f"[ALIGN-V104-ASYM-PLAN] mode={mode} side={str(plan['v104_asymmetry_side'])} "
                f"widthDiff={float(report.get('width_diff_mm',0.0)):.1f}mm legLenDiff={float(report.get('leg_len_diff_mm',0.0)):.1f}mm "
                f"move={float(plan.get('move_mm',0.0)):.1f}mm arm={str(plan.get('arm_key','')).upper()} "
                f"source={np.round(np.asarray(plan['source_board']),1).tolist()} target={np.round(np.asarray(plan['target_board']),1).tolist()}"
            )
            return plan, report

        def _v56_align_center_plan(self, obs, H, seam_x, geom, native_plan, native_decision):
            if self._v67_prepare_cancel_requested():
                return None, "V67_CANCELLED_BY_HUMAN_OVERRIDE"
            waist = np.asarray(geom["waist_center"], np.float64)
            lower = np.asarray(geom["lower_point"], np.float64)
            axis = lower - waist
            trials = []
            rejected_small = []

            def source_ok(q):
                if not self._v56_align_board_safe(q, 8.0):
                    return None
                support_ok, support_ratio = self._v56_mask_support_at_board(obs, H, q, radius_px=6, min_ratio=0.78)
                if not support_ok:
                    return None
                rep = self._v53_mask_point_report(obs, q)
                edge_depth = rep.get("edge_depth_mm") if isinstance(rep, dict) else None
                if not bool(rep.get("ok", False)) or edge_depth is None or float(edge_depth) < float(self._v61_align_center_hard_inset_mm):
                    return None
                return float(support_ratio), float(edge_depth)

            center_t_values, center_t_cap, crotch_t, center_preferred_t = self._v102_align_center_t_policy(obs, geom)
            print(
                f"[ALIGN-V102-UPPER-CONTACT] tValues={[round(x,2) for x in center_t_values]} "
                f"cap={center_t_cap:.2f} crotchT={'NA' if crotch_t is None else f'{crotch_t:.2f}'}; crotch/lower-leg contacts excluded"
            )
            for t in center_t_values:
                if self._v67_prepare_cancel_requested():
                    return None, "V67_CANCELLED_BY_HUMAN_OVERRIDE"
                base = waist + axis * float(t)
                base_gap = abs(float(base[0]) - float(seam_x))
                correction_sign = 1.0 if float(seam_x) >= float(base[0]) else -1.0
                if base_gap < float(self._v62_align_center_small_gap_threshold_mm):
                    requested_dx = float(seam_x) + correction_sign * float(self._v62_align_center_small_gap_extra_mm) - float(base[0])
                else:
                    requested_dx = float(seam_x) - float(base[0])
                if abs(requested_dx) < 1e-6:
                    continue

                _, _, base_src_r = self._v56_align_arm_reach("arm2", base, final_target=False)
                _, _, base_tgt_r = self._v56_align_arm_reach("arm2", base + np.asarray([requested_dx, 0.0]), final_target=True)
                base_r = max(float(base_src_r), float(base_tgt_r))
                shift_values = [0.0]
                step = float(self._v63_align_center_arm2_shift_step_mm)
                max_shift = float(self._v63_align_center_arm2_shift_max_mm)
                for mag in np.arange(step, max_shift + 0.1, step):
                    shift_values.extend((-float(mag), float(mag)))

                for arm_key in ("arm2", "arm1"):
                    arm_shifts = shift_values if arm_key == "arm2" else [0.0]
                    for shift_x in arm_shifts:
                        if self._v67_prepare_cancel_requested():
                            return None, "V67_CANCELLED_BY_HUMAN_OVERRIDE"
                        source = base + np.asarray([float(shift_x), 0.0], np.float64)
                        requested_target = source + np.asarray([requested_dx, 0.0], np.float64)
                        src_info = source_ok(source)
                        if src_info is None:
                            continue
                        support_ratio, edge_depth = src_info
                        src_reach, _, src_r = self._v56_align_arm_reach(arm_key, source, final_target=False)
                        if not src_reach:
                            continue
                        direct_ok, _, _ = self._v56_align_arm_reach(arm_key, requested_target, final_target=True)
                        if direct_ok and self._v56_align_board_safe(requested_target, 8.0):
                            target = np.asarray(requested_target, np.float64)
                            frac = 1.0
                        else:
                            target, frac = self._v56_align_safe_progress(arm_key, source, requested_target)
                            if target is None:
                                continue
                        move = float(np.linalg.norm(target - source))
                        if move < float(self._v59_align_center_min_action_mm):
                            rejected_small.append((base_gap, move))
                            continue
                        _, _, tgt_r = self._v56_align_arm_reach(arm_key, target, final_target=True)
                        max_r = max(float(src_r), float(tgt_r))
                        if arm_key == "arm2" and abs(float(shift_x)) > 1e-6 and math.isfinite(base_r):
                            improves_radius = max_r <= base_r - float(self._v63_align_center_shift_min_improve_mm)
                            improves_fraction = float(frac) >= 0.95 and not bool(direct_ok and abs(float(shift_x)) < 1e-6)
                            if not improves_radius and not improves_fraction:
                                continue
                        plan = copy.deepcopy(native_plan) if isinstance(native_plan, dict) else {}
                        plan.update({
                            "kind": "CENTER", "v56_mode": "CENTER_VECTOR_PRESS", "arm_key": arm_key,
                            "source_board": [float(source[0]), float(source[1])],
                            "target_board": [float(target[0]), float(target[1])],
                            "requested_target_board": [float(requested_target[0]), float(requested_target[1])],
                            "move_mm": move, "target_fraction": float(frac),
                            "angle_error_deg": float(geom["angle_error_deg"]),
                            "predicted_angle_residual_deg": float(geom["angle_error_deg"]),
                            "predicted_seam_residual_mm": float(target[0] - seam_x),
                            "v56_center_t": float(t), "v56_center_axis_offset_mm": float(shift_x),
                            "v56_source_mask_ratio": float(support_ratio),
                            "v61_source_edge_depth_mm": float(edge_depth),
                            "v61_center_hard_inset_mm": float(self._v61_align_center_hard_inset_mm),
                            "v59_center_base_gap_mm": float(base_gap),
                            "v59_center_min_action_mm": float(self._v59_align_center_min_action_mm),
                            "v63_parallel_shift_mm": float(shift_x),
                            "v63_parallel_shift_preserves_delta_mm": float(abs(requested_dx)),
                            "v63_actual_executable_move_mm": float(move),
                            "v63_arm_reach_radius_mm": float(max_r),
                            "v102_center_t_cap": float(center_t_cap),
                            "v102_crotch_t": None if crotch_t is None else float(crotch_t),
                        })
                        if self._v56_plan_obj_rejected("ALIGN", plan, "ARM2_CORRECTION"):
                            continue
                        trials.append((plan, max_r))

            if not trials:
                if rejected_small:
                    max_gap = max(float(x[0]) for x in rejected_small)
                    max_move = max(float(x[1]) for x in rejected_small)
                    print(f"[ALIGN-V67-CENTER-SKIP] maxAxisGap={max_gap:.1f}mm maxSafeMove={max_move:.1f}mm")
                    return None, "V67_CENTER_VECTOR_NO_MEANINGFUL_SAFE_MOVE"
                return None, "V67_CENTER_VECTOR_SAFE_INSET_CONTACT_UNAVAILABLE"

            max_move = max(float(p.get("move_mm", 0.0)) for p, _ in trials)
            floor = max(float(self._v59_align_center_min_action_mm), max_move - float(self._v70_align_center_move_slack_mm))
            pool = [row for row in trials if float(row[0].get("move_mm", 0.0)) >= floor]
            pool.sort(key=lambda row: (
                -float(row[0].get("move_mm", 0.0)),
                abs(float(row[0].get("v56_center_t", 0.5)) - float(center_preferred_t)),
                0 if str(row[0].get("arm_key", "")) == "arm2" else 1,
                1.0 - float(row[0].get("target_fraction", 0.0)),
                abs(float(row[0].get("v63_parallel_shift_mm", 0.0))),
                float(row[1]),
            ))
            plan = pool[0][0]

            pivot_plan = self._v67_align_intersection_pivot_plan(obs, H, seam_x, geom, plan)
            if pivot_plan is not None:
                return pivot_plan, "V75_DISTANCE_GATED_ARM2_HOLD_ARM1_VERTICAL_PIVOT"

            source = np.asarray(plan["source_board"], np.float64).reshape(2)
            target = np.asarray(plan["target_board"], np.float64).reshape(2)
            old_move = float(np.linalg.norm(target - source))
            vec = target - source
            n = float(np.linalg.norm(vec))
            arm_key_selected = str(plan.get("arm_key", "arm2")).lower()
            requested_extra_mm = 0.0
            if arm_key_selected == "arm2" and old_move >= float(self._v85_align_long_move_threshold_mm):
                requested_extra_mm = float(self._v86_align_arm2_long_extra_mm)
            elif float(self._v67_align_extra_move_mm) > 0.0:
                requested_extra_mm = float(self._v67_align_extra_move_mm)
            if n > 1e-6 and requested_extra_mm > 0.0:
                extended_requested = target + (vec / n) * requested_extra_mm
                extended_target, _ = self._v56_align_safe_progress(arm_key_selected, source, extended_requested)
                if extended_target is not None:
                    extended_move = float(np.linalg.norm(extended_target - source))
                    if extended_move > old_move + 1.0:
                        plan["target_board"] = [float(extended_target[0]), float(extended_target[1])]
                        plan["requested_target_board"] = [float(extended_requested[0]), float(extended_requested[1])]
                        plan["move_mm"] = float(extended_move)
                        plan["v67_center_old_move_mm"] = float(old_move)
                        plan["v67_center_extra_requested_mm"] = float(requested_extra_mm)
                        plan["v67_center_extra_applied_mm"] = float(extended_move - old_move)
                        plan["v67_center_z_extra_mm"] = float(self._v70_align_center_z_extra_mm)
                        plan["v70_center_z_extra_mm"] = float(self._v70_align_center_z_extra_mm)
                        plan["v70_center_drag_speed"] = float(self._v70_align_drag_speed)
                        if arm_key_selected == "arm2" and old_move >= float(self._v85_align_long_move_threshold_mm):
                            plan["v86_arm2_long_extra_requested_mm"] = float(requested_extra_mm)
                            plan["v86_arm2_long_extra_applied_mm"] = float(extended_move - old_move)
                            print(
                                f"[ALIGN-V86-ARM2-EXTRA] old={old_move:.1f}mm request=+{requested_extra_mm:.0f}mm "
                                f"applied=+{extended_move-old_move:.1f}mm new={extended_move:.1f}mm"
                            )
                        else:
                            print(
                                f"[ALIGN-V67-CENTER-EXTRA] old={old_move:.1f}mm request=+{requested_extra_mm:.0f}mm "
                                f"applied=+{extended_move-old_move:.1f}mm new={extended_move:.1f}mm"
                            )

            shift = float(plan.get("v63_parallel_shift_mm", 0.0))
            print(
                f"[ALIGN-V67-CENTER-SELECT] move={float(plan.get('move_mm',0.0)):.1f}mm "
                f"t={float(plan.get('v56_center_t',0.0)):.2f} arm={str(plan.get('arm_key','')).upper()} "
                f"shift={shift:+.1f}mm"
            )
            return plan, "V74_CENTER_NATIVE_DISTANCE"

        def _v56_mask_board_samples(self, obs, H, max_points=9000):
            try:
                mask_obj = getattr(obs, "mask", None)
                mask = getattr(mask_obj, "mask_u8", None) if mask_obj is not None else None
                if mask is None:
                    return None
                arr = np.asarray(mask)
                yy, xx = np.nonzero(arr > 0)
                if len(xx) < 50:
                    return None
                step = max(1, int(math.ceil(len(xx)/float(max_points))))
                px = np.column_stack((xx[::step], yy[::step])).astype(np.float64)
                Hm = np.asarray(H, np.float64).reshape(3, 3)
                homo = np.column_stack((px[:,0], px[:,1], np.ones(len(px), np.float64)))
                out = homo @ Hm.T
                good = np.isfinite(out).all(axis=1) & (np.abs(out[:,2]) > 1e-8)
                board = out[good,:2] / out[good,2:3]
                return board if len(board) >= 30 else None
            except Exception:
                return None

        def _v61_align_axis_mask_gate(self, obs, H, geom):
            report = {"ok": False, "supported": 0, "tested": 0, "ratios": [], "reason": "UNAVAILABLE"}
            if not isinstance(geom, dict):
                return report
            if str(geom.get("axis_source", "")).upper() == "MASK_WIDTH_CENTERLINE":
                bins = int(geom.get("mask_centerline_bins", 0))
                rms = float(geom.get("mask_centerline_fit_rms_mm", 999.0))
                report.update({"supported": bins, "tested": bins, "ratios": [], "fit_rms_mm": rms})
                report["ok"] = bool(bins >= 5 and math.isfinite(rms) and rms <= 35.0)
                report["reason"] = "OK_MASK_WIDTH_CENTERLINE" if report["ok"] else "MASK_CENTERLINE_FIT_UNSTABLE"
                return report
            waist = self._finite_point2(geom.get("waist_center"))
            lower = self._finite_point2(geom.get("lower_point"))
            if waist is None or lower is None:
                report["reason"] = "AXIS_POINTS_UNAVAILABLE"
                return report
            axis = lower-waist
            if float(np.linalg.norm(axis)) < 20.0:
                report["reason"] = "AXIS_DEGENERATE"
                return report
            ratios = []
            supported = 0
            for t in (0.12, 0.24, 0.36, 0.48, 0.60):
                q = waist + axis*float(t)
                ok, ratio = self._v56_mask_support_at_board(
                    obs, H, q, radius_px=5,
                    min_ratio=float(self._v61_align_axis_support_min_ratio),
                )
                ratios.append(float(ratio))
                supported += int(bool(ok))
            report.update({
                "supported": int(supported),
                "tested": int(len(ratios)),
                "ratios": ratios,
            })
            report["ok"] = bool(supported >= int(self._v61_align_axis_support_required))
            report["reason"] = "OK" if report["ok"] else "BODY_AXIS_DOES_NOT_FOLLOW_GARMENT_MASK"
            return report

        def _v56_align_outer_plan(self, obs, H, seam_x, guard_x, left_x, geom, native_plan):
            samples = self._v56_mask_board_samples(obs, H)
            if samples is None:
                return None, "V61_RED_OVERFLOW_MASK_SAMPLES_UNAVAILABLE"
            overflow = samples[samples[:,0] > float(guard_x)+4.0]
            if len(overflow) < 12:
                return None, "V61_RED_OVERFLOW_REGION_TOO_SMALL"

            axis_valid = bool((geom or {}).get("v61_axis_mask_valid", True)) if isinstance(geom, dict) else False
            pts = self._pose_board_points(obs)
            waist = self._finite_point2((geom or {}).get("waist_center")) if isinstance(geom, dict) else None
            hem = self._finite_point2(pts.get("left_hem_center"))
            if axis_valid and waist is not None and hem is not None:
                y_pref = float((0.35*waist + 0.65*hem)[1])
                y_source = "VALID_AXIS_WAIST_HEM"
            else:
                y_pref = float(np.median(overflow[:,1]))
                y_source = "MASK_OVERFLOW_MEDIAN"

            requested_overflow = max(0.0, float(left_x)-float(guard_x))
            order = np.argsort(np.abs(overflow[:,1]-y_pref))
            trials = []
            seen_rows = []
            band_mm = float(self._v58_align_red_outer_row_band_mm)
            inset_mm = float(self._v62_align_red_outer_inset_mm)
            for idx in order[:350]:
                y0 = float(overflow[int(idx),1])
                if any(abs(y0-y_prev) < 4.0 for y_prev in seen_rows):
                    continue
                seen_rows.append(y0)
                row = samples[np.abs(samples[:,1]-y0) <= band_mm]
                if len(row) < 12:
                    continue
                local_outer_x = float(np.max(row[:,0]))
                local_overflow = max(0.0, local_outer_x-float(guard_x))
                if local_overflow < 8.0:
                    continue
                wanted_source = np.asarray([local_outer_x-inset_mm, y0], np.float64)
                row_order = np.argsort(np.abs(row[:,0]-wanted_source[0]) + 0.20*np.abs(row[:,1]-y0))
                source = np.asarray(row[int(row_order[0])], np.float64)
                measured_inset = float(local_outer_x-source[0])
                if measured_inset < 38.0 or measured_inset > 70.0:
                    continue
                support_ok, support_ratio = self._v56_mask_support_at_board(obs, H, source, radius_px=6, min_ratio=0.78)
                if not support_ok:
                    continue
                inset_report = self._v53_mask_point_report(obs, source)
                edge_depth = inset_report.get("edge_depth_mm") if isinstance(inset_report, dict) else None
                if (
                    not bool((inset_report or {}).get("ok", False))
                    or edge_depth is None
                    or float(edge_depth) < float(self._v62_align_red_hard_inset_mm)
                ):
                    continue

                drag_request = max(local_overflow, requested_overflow) + float(self._v59_align_red_outer_extra_drag_mm)
                desired = np.asarray([float(source[0])-float(drag_request), float(source[1])], np.float64)
                if desired[0] >= source[0]-8.0:
                    continue
                for arm_key in ("arm1", "arm2"):
                    src_ok, _, src_r = self._v56_align_arm_reach(arm_key, source, final_target=False)
                    if not src_ok:
                        continue
                    target, frac = self._v56_align_safe_progress(arm_key, source, desired)
                    if target is None:
                        continue
                    move = float(np.linalg.norm(target-source))
                    if move < 8.0:
                        continue
                    angle_value = float((geom or {}).get("angle_error_deg", 0.0)) if isinstance(geom, dict) else 0.0
                    mid_value = float((geom or {}).get("mid_error_mm", 0.0)) if isinstance(geom, dict) else 0.0
                    plan = copy.deepcopy(native_plan) if isinstance(native_plan, dict) else {}
                    plan.update({
                        "kind": "OUTER",
                        "v56_mode": "RED_OUTER_PRESS",
                        "arm_key": arm_key,
                        "source_board": [float(source[0]), float(source[1])],
                        "target_board": [float(target[0]), float(target[1])],
                        "requested_target_board": [float(desired[0]), float(desired[1])],
                        "move_mm": move,
                        "target_fraction": float(frac),
                        "angle_error_deg": angle_value,
                        "predicted_angle_residual_deg": angle_value,
                        "predicted_seam_residual_mm": mid_value,
                        "v56_red_guard_x_mm": float(guard_x),
                        "v56_left_outer_x_mm": float(left_x),
                        "v56_source_mask_ratio": float(support_ratio),
                        "v58_red_local_outer_x_mm": float(local_outer_x),
                        "v58_red_local_overflow_mm": float(local_overflow),
                        "v58_red_requested_overflow_mm": float(drag_request),
                        "v58_red_source_inset_mm": float(measured_inset),
                        "v61_source_edge_depth_mm": float(edge_depth),
                        "v61_red_hard_inset_mm": float(self._v62_align_red_hard_inset_mm),
                        "v61_axis_mask_valid": bool(axis_valid),
                        "v61_red_y_source": str(y_source),
                    })
                    if self._v56_plan_obj_rejected("ALIGN", plan, "ARM2_CORRECTION"):
                        continue
                    _, _, tgt_r = self._v56_align_arm_reach(arm_key, target, final_target=True)
                    score = (
                        abs(float(source[1])-y_pref)
                        + 2.5*abs(float(measured_inset)-inset_mm)
                        + 0.10*max(float(src_r), float(tgt_r))
                        + 20.0*(1.0-float(frac))
                        - 0.05*float(local_overflow)
                    )
                    trials.append((score, plan))
            if not trials:
                return None, "V61_RED_OVERFLOW_SAFE_INSET_CONTACT_UNAVAILABLE"
            trials.sort(key=lambda row: row[0])
            plan = trials[0][1]
            print(
                f"[ALIGN-V61-RED-SELECT] source={np.round(np.asarray(plan['source_board']),1).tolist()} "
                f"edgeDepth={float(plan.get('v61_source_edge_depth_mm',0.0)):.1f}mm "
                f"drag={float(plan.get('move_mm',0.0)):.1f}mm ySource={plan.get('v61_red_y_source')}"
            )
            return plan, "V61_RED_OVERFLOW_SAFE_INSET_CONTACT"

        def _install_align_centerline_policy(self):
            original = getattr(self.align, "_align7_build_auto_plan", None)
            if not callable(original) or bool(getattr(original, "_bottom_vla_v61_centerline", False)):
                return

            def wrapped(obs, H, waist, seam_x, correction_count, config, cfg, args):
                if bool(getattr(self, "_v85_align_fast_mode", False)):
                    geom_obs = obs
                else:
                    geom_obs, _ = self._v54_obs_with_refined_waist(obs)
                geom = self._align_vector_geometry(geom_obs, seam_x)
                native_plan = None
                native_decision = {"reason": "V61_NATIVE_ALIGN_PLAN_SKIPPED_DUPLICATE_WORK"}
                decision = dict(native_decision)

                guard_x = self._v50_align_left_guard_x(float(seam_x))
                left_x = self._v50_mask_left_x(geom_obs, H) if guard_x is not None else None
                left_overflow = None if left_x is None or guard_x is None else float(left_x-guard_x)
                overflow_active = bool(
                    left_overflow is not None
                    and left_overflow > float(self._v50_align_left_overflow_trigger_mm)
                )

                axis_report = self._v61_align_axis_mask_gate(geom_obs, H, geom) if geom is not None else {
                    "ok": False, "supported": 0, "tested": 0, "ratios": [], "reason": "GEOMETRY_UNAVAILABLE"
                }
                axis_valid = bool(axis_report.get("ok", False))
                if isinstance(geom, dict):
                    geom = copy.deepcopy(geom)
                    geom["v61_axis_mask_valid"] = bool(axis_valid)
                    geom["v61_axis_mask_report"] = copy.deepcopy(axis_report)

                if not axis_valid:
                    print(
                        f"[ALIGN-V61-AXIS-REJECT] reason={axis_report.get('reason')} "
                        f"support={int(axis_report.get('supported',0))}/{int(axis_report.get('tested',0))}; "
                        "CENTER vector is not trusted or drawn"
                    )

                mid_err = float(geom.get("mid_error_mm", 0.0)) if isinstance(geom, dict) else 0.0
                angle_err = float(geom.get("angle_error_deg", 0.0)) if isinstance(geom, dict) else 0.0
                axis_good = bool(
                    axis_valid
                    and abs(mid_err) <= float(self._v30_align_line_done_mm)
                    and abs(angle_err) <= float(self._v50_align_axis_good_deg)
                )

                plan = None
                reason = ""

                # If the green body axis is not trustworthy, never invent a CENTER action.
                # A real Mask overflow beyond the red guard remains independently usable.
                if not axis_valid:
                    plan = None
                    reason = "V98_CENTER_AXIS_INVALID_BLOCK_OUTER_AND_REJUDGE"
                    print(
                        f"[ALIGN-V98-AXIS-BLOCK] invalid CENTER axis reason={axis_report.get('reason')} "
                        "-> OUTER correction blocked; fresh trusted Mask/axis required"
                    )
                elif axis_good:
                    if overflow_active:
                        plan, reason = self._v56_align_outer_plan(
                            geom_obs, H, seam_x, float(guard_x), float(left_x), geom, native_plan
                        )
                        if plan is not None:
                            print(
                                f"[ALIGN-V62-RED] axisGood=True overflow={left_overflow:.1f}mm "
                                f"source={np.round(np.asarray(plan['source_board']),1).tolist()} -> "
                                f"target={np.round(np.asarray(plan['target_board']),1).tolist()} arm={str(plan.get('arm_key','arm2')).upper()}"
                            )
                    else:
                        asym_plan, asym_report = self._v104_align_asymmetry_plan(geom_obs, H, seam_x, geom, native_plan)
                        if asym_plan is not None:
                            plan = asym_plan
                            reason = "V104_ALIGN_ASYMMETRY_CORRECTION"
                        elif left_overflow is None:
                            reason = "V61_AXIS_GOOD_RED_OUTER_UNAVAILABLE"
                        else:
                            reason = "V61_ALIGN_COMPLETE_AXIS_AND_RED_OUTER"
                            print(
                                f"[ALIGN-V61-DONE] midGap={mid_err:+.1f}mm angle={angle_err:+.1f}deg "
                                f"leftOverflow={left_overflow:+.1f}mm -> axis acceptable and no large asymmetry correction required"
                            )
                else:
                    plan, reason = self._v56_align_center_plan(
                        geom_obs, H, seam_x, geom, native_plan, decision
                    )
                    if plan is not None:
                        print(
                            f"[ALIGN-V62-CENTER] midGap={mid_err:+.1f}mm angle={angle_err:+.1f}deg "
                            f"source={np.round(np.asarray(plan['source_board']),1).tolist()} -> "
                            f"target={np.round(np.asarray(plan['target_board']),1).tolist()} arm={str(plan.get('arm_key','arm2')).upper()}"
                        )
                    elif overflow_active:
                        # If every center-axis source fails the 35 mm garment inset/reach gate,
                        # prefer the real red-overflow correction instead of an outside contact.
                        plan, red_reason = self._v56_align_outer_plan(
                            geom_obs, H, seam_x, float(guard_x), float(left_x), geom, native_plan
                        )
                        if plan is not None:
                            reason = red_reason
                            print(
                                f"[ALIGN-V61-CENTER-TO-RED] CENTER contact unavailable; overflow={left_overflow:.1f}mm -> RED OUTER"
                            )

                if plan is not None:
                    chosen = str(plan.get("kind", "CENTER")).upper()
                    decision.update({
                        "chosen": chosen,
                        "preferred": chosen,
                        "reason": str(reason),
                        "decision": str(reason),
                        "v56_press_drag": True,
                        "v56_mode": str(plan.get("v56_mode", "")),
                        "v56_arm_key": str(plan.get("arm_key", "arm2")),
                        "v56_source_on_center_vector_preferred": bool(str(plan.get("v56_mode", "")).startswith("CENTER_VECTOR")),
                    })
                else:
                    decision.update({
                        "chosen": None,
                        "preferred": "NONE",
                        "reason": str(reason or "V61_ALIGN_NO_SAFE_PLAN"),
                        "decision": str(reason or "V61_ALIGN_NO_SAFE_PLAN"),
                    })

                decision.update({
                    "v56_objective": "PREFER_WAIST_LEVEL_UP_THEN_SAFE_CENTER_OR_RED_FALLBACK_WITHOUT_STAGE_VETO",
                    "v56_geometry_available": bool(geom is not None),
                    "v61_axis_mask_valid": bool(axis_valid),
                    "v61_axis_mask_report": copy.deepcopy(axis_report),
                    "v56_axis_good": bool(axis_good),
                    "v56_line_done_mm": float(self._v30_align_line_done_mm),
                    "v56_angle_done_deg": float(self._v50_align_axis_good_deg),
                    "v50_left_guard_x_mm": None if guard_x is None else float(guard_x),
                    "v50_mask_left_x_mm": None if left_x is None else float(left_x),
                    "v50_left_overflow_mm": None if left_overflow is None else float(left_overflow),
                    "white_line_x_mm": float(seam_x),
                })

                # The overlay in align-11 draws the green vector whenever these keys exist.
                # Only expose them when the body axis actually follows the garment Mask.
                if axis_valid and isinstance(geom, dict):
                    display_waist = np.asarray(geom["waist_center"], np.float64).reshape(2)
                    display_lower_full = np.asarray(geom["lower_point"], np.float64).reshape(2)
                    display_t = float(self._v102_align_display_t_max)
                    try:
                        _, display_cap, display_crotch_t, _ = self._v102_align_center_t_policy(geom_obs, geom)
                        if display_crotch_t is not None:
                            display_t = min(display_t, max(0.38, float(display_crotch_t)))
                        else:
                            display_t = min(display_t, max(0.38, float(display_cap) + 0.06))
                    except Exception:
                        pass
                    display_lower = display_waist + (display_lower_full - display_waist) * display_t
                    display_mid = 0.5 * (display_waist + display_lower)
                    decision.update({
                        "vector_mid_x_mm": float(display_mid[0]),
                        "white_line_mid_error_mm": float(mid_err),
                        "white_line_waist_error_mm": float(geom["waist_error_mm"]),
                        "white_line_lower_error_mm": float(geom["lower_error_mm"]),
                        "white_line_max_endpoint_error_mm": float(geom["max_endpoint_error_mm"]),
                        "center_angle_error_deg": float(angle_err),
                        "waist_center": display_waist.astype(float).tolist(),
                        "crotch_center": display_lower.astype(float).tolist(),
                        "leg_center": display_lower.astype(float).tolist(),
                        "v102_display_axis_t": float(display_t),
                    })
                return plan, decision

            wrapped._bottom_vla_v61_centerline = True
            wrapped._bottom_vla_v56_centerline = True
            self.align._align7_build_auto_plan = wrapped

        def _v67_execute_intersection_pivot_press_drag(self, action_plan, arms, config, cfg, args, on_verified_start=None):
            report = copy.deepcopy(action_plan)
            moving_key = "arm1"
            anchor_key = "arm2"
            if str(action_plan.get("arm_key", "")).lower() != moving_key or str(action_plan.get("v67_pivot_arm_key", "")).lower() != anchor_key:
                return False, "ALIGN_V67_PIVOT_ARM_MAPPING_INVALID", report
            moving_arm = arms.get(moving_key)
            anchor_arm = arms.get(anchor_key)
            if (moving_arm is None or anchor_arm is None) and cfg.send and not cfg.dry_run:
                return False, "ALIGN_V67_PIVOT_ARM_MISSING", report
            source = np.asarray(action_plan["source_board"], np.float64).reshape(2)
            target = np.asarray(action_plan["target_board"], np.float64).reshape(2)
            anchor_source = np.asarray(action_plan["v67_pivot_source_board"], np.float64).reshape(2)
            if cfg.dry_run or not cfg.send:
                return True, "PREVIEW", report
            if callable(on_verified_start):
                on_verified_start()

            move_command = int(getattr(args, "move_command", 104))
            tool_t = float(getattr(cfg, "tool_t_rad", 3.14))
            hover_z = float(max(float(getattr(cfg, "clear_z", 180.0)), float(getattr(args, "d31_hover_z", 180.0))))
            approach_speed = float(np.clip(getattr(args, "d31_approach_speed", 1.12), 0.45, 1.20))
            vertical_speed = float(np.clip(getattr(args, "d31_vertical_speed", 1.12), 0.35, 1.20))
            drag_speed = 0.55
            clear_speed = max(0.65, float(np.clip(getattr(args, "align6_arm2_clear_speed", 0.50), 0.25, 0.70)))
            close_angle = float(np.clip(getattr(args, "d31_grip_close", 3.05), 2.90, 3.14))
            anchor_press_clearance_mm = (
                float(self._v56_align_press_clearance_mm)
                + float(self._v70_align_center_z_extra_mm)
                + float(self._v76_align_arm2_z_extra_mm)
            )
            moving_press_clearance_mm = float(self._v56_align_press_clearance_mm) + float(self._v73_align_arm1_z_extra_mm)
            pivot_move_mm = float(np.linalg.norm(target - source))
            pivot_long_z_extra_mm = (
                float(self._v85_align_long_move_z_extra_mm)
                if pivot_move_mm >= float(self._v85_align_long_move_threshold_mm)
                else 0.0
            )
            moving_press_clearance_mm += pivot_long_z_extra_mm
            if pivot_long_z_extra_mm < 0.0:
                print(
                    f"[ALIGN-V85-LONG-Z] PIVOT move={pivot_move_mm:.1f}mm -> "
                    f"ARM1 mover contact/drag extra={pivot_long_z_extra_mm:+.1f}mm"
                )
            report["v85_long_move_mm"] = float(pivot_move_mm)
            report["v85_long_move_z_extra_mm"] = float(pivot_long_z_extra_mm)
            surface_fn = getattr(self.align, "_c93_surface_z_at_board", None)
            if not callable(surface_fn):
                return False, "ALIGN_V67_SURFACE_API_MISSING", report
            current = {"arm1": None, "arm2": None}

            def xyz_at(key, board_point, z):
                rx, ry = self.align.board_to_arm_xy(config, key, float(board_point[0]), float(board_point[1]))
                return np.asarray([float(rx), float(ry), float(z)], np.float32)

            def move(key, arm, xyz, speed, stage):
                target_xyz = np.asarray(xyz, np.float32).reshape(3)
                start_xyz = None if current.get(key) is None else np.asarray(current.get(key), np.float32).reshape(3)
                stage_u = str(stage).upper()
                arm.move_goal(
                    float(target_xyz[0]), float(target_xyz[1]), float(target_xyz[2]), tool_t, float(speed),
                    move_command=move_command, stage=stage, caller="_align_v67_intersection_pivot_v111",
                )
                if "APPROACH" in stage_u:
                    timeout_s = float(self._v111_pivot_soft_timeout_approach_s)
                elif "ROTATE_TRANSLATE" in stage_u:
                    timeout_s = float(self._v111_pivot_soft_timeout_drag_s)
                elif "CLEAR" in stage_u:
                    timeout_s = float(self._v111_pivot_soft_timeout_clear_s)
                else:
                    timeout_s = float(self._v111_pivot_soft_timeout_contact_s)
                deadline = time.monotonic() + max(0.4, timeout_s)
                last_xyz = None
                soft_ok = False
                soft_reason = "NO_FEEDBACK"
                while time.monotonic() < deadline:
                    fb = self.align._d31_query_feedback(arm, timeout_s=0.22)
                    actual = self.align._d31_feedback_xyz(fb)
                    if actual is None:
                        time.sleep(float(self._v111_pivot_soft_poll_s))
                        continue
                    actual = np.asarray(actual, np.float32).reshape(3)
                    last_xyz = actual
                    xy_err = float(np.linalg.norm(actual[:2] - target_xyz[:2]))
                    z_err = abs(float(actual[2] - target_xyz[2]))
                    if "APPROACH" in stage_u:
                        soft_ok = bool(xy_err <= 30.0 and float(actual[2]) >= 150.0)
                        soft_reason = f"approach xy={xy_err:.1f} z={float(actual[2]):.1f}"
                    elif "CLEAR" in stage_u:
                        soft_ok = bool(xy_err <= 30.0 and float(actual[2]) >= float(self._v111_pivot_clear_z_pass_mm))
                        soft_reason = f"clear xy={xy_err:.1f} z={float(actual[2]):.1f}"
                    elif "ROTATE_TRANSLATE" in stage_u and start_xyz is not None:
                        vec = target_xyz[:2] - start_xyz[:2]
                        denom = float(np.dot(vec, vec))
                        if denom > 1e-6:
                            rel = actual[:2] - start_xyz[:2]
                            progress = float(np.dot(rel, vec) / denom)
                            proj = start_xyz[:2] + np.clip(progress, 0.0, 1.0) * vec
                            cross = float(np.linalg.norm(actual[:2] - proj))
                        else:
                            progress = 1.0
                            cross = xy_err
                        soft_ok = bool(
                            progress >= float(self._v111_pivot_drag_progress_pass)
                            and cross <= 22.0
                            and z_err <= 14.0
                        )
                        soft_reason = f"drag progress={progress:.2f} cross={cross:.1f} zerr={z_err:.1f}"
                    else:
                        soft_ok = bool(xy_err <= 22.0 and z_err <= 14.0)
                        soft_reason = f"contact xy={xy_err:.1f} zerr={z_err:.1f}"
                    if soft_ok:
                        print(f"[ALIGN-V111-SOFT-PASS] {stage} {soft_reason}; continue without exact settle")
                        break
                    time.sleep(float(self._v111_pivot_soft_poll_s))
                if not soft_ok:
                    print(
                        f"[ALIGN-V111-SOFT-FALLBACK] {stage} soft-pass timeout ({soft_reason}); "
                        "fall back to original exact verified arrival"
                    )
                    ok, why, _ = self.align._d31_move_verified(
                        arm, target_xyz, cfg, float(speed), move_command, args, stage, wait_from=last_xyz if last_xyz is not None else start_xyz
                    )
                    if not ok:
                        raise RuntimeError(str(why))
                current[key] = target_xyz.copy()

            def parallel_moves(rows):
                errors = []
                threads = []
                def runner(key, arm, xyz, speed, stage):
                    try:
                        move(key, arm, xyz, speed, stage)
                    except Exception as exc:
                        errors.append((key, repr(exc)))
                for key, arm, xyz, speed, stage in rows:
                    t = threading.Thread(target=runner, args=(key, arm, xyz, speed, stage), daemon=True)
                    threads.append(t)
                    t.start()
                for t in threads:
                    t.join(timeout=4.5)
                if errors:
                    raise RuntimeError(f"parallel move failed: {errors}")
                if any(t.is_alive() for t in threads):
                    raise RuntimeError("parallel move timeout")

            def parallel_gripper_close():
                errors = []
                threads = []
                def runner(key, arm, stage):
                    try:
                        arm.set_gripper(close_angle, delay=0.25, stage=stage, caller="_align_v67_intersection_pivot")
                    except Exception as exc:
                        errors.append((key, repr(exc)))
                for key, arm, stage in (
                    (anchor_key, anchor_arm, "ALIGN_V88_ARM2_PIVOT_CLOSED"),
                    (moving_key, moving_arm, "ALIGN_V88_ARM1_MOVER_CLOSED"),
                ):
                    t = threading.Thread(target=runner, args=(key, arm, stage), daemon=True)
                    threads.append(t)
                    t.start()
                for t in threads:
                    t.join(timeout=1.5)
                if errors:
                    raise RuntimeError(f"parallel gripper close failed: {errors}")

            def parallel_standby():
                results = {}
                threads = []
                def runner(key, why):
                    results[key] = bool(self._v60_align_return_standby_verified(key, arms, cfg, move_command, why))
                for key, why in (
                    (moving_key, "ALIGN-V88 ARM1 mover complete"),
                    (anchor_key, "ALIGN-V88 ARM2 pivot complete"),
                ):
                    t = threading.Thread(target=runner, args=(key, why), daemon=True)
                    threads.append(t)
                    t.start()
                for t in threads:
                    t.join(timeout=5.0)
                return all(bool(results.get(k, False)) for k in (moving_key, anchor_key)) and not any(t.is_alive() for t in threads)

            try:
                parallel_gripper_close()

                anchor_approach = xyz_at(anchor_key, anchor_source, hover_z)
                moving_approach = xyz_at(moving_key, source, hover_z)
                print("[ALIGN-V88-PIVOT-SYNC] ARM1+ARM2 HOVER simultaneously")
                parallel_moves([
                    (anchor_key, anchor_arm, anchor_approach, approach_speed, "ALIGN_V88_ARM2_PIVOT_APPROACH_PARALLEL"),
                    (moving_key, moving_arm, moving_approach, approach_speed, "ALIGN_V88_ARM1_MOVER_APPROACH_PARALLEL"),
                ])

                anchor_floor = float(surface_fn(config, anchor_key, anchor_source, float(self.align.arm_contact_z(cfg, anchor_key))))
                anchor_contact = xyz_at(anchor_key, anchor_source, anchor_floor + anchor_press_clearance_mm)
                print("[ALIGN-V88-PIVOT-ORDER] ARM2 press/hold first; ARM1 remains at HOVER")
                move(anchor_key, anchor_arm, anchor_contact, vertical_speed, "ALIGN_V88_ARM2_PIVOT_PRESS_HOLD_FIRST")

                moving_floor = float(surface_fn(config, moving_key, source, float(self.align.arm_contact_z(cfg, moving_key))))
                moving_contact = xyz_at(moving_key, source, moving_floor + moving_press_clearance_mm)
                print("[ALIGN-V88-PIVOT-ORDER] ARM2 hold verified -> ARM1 press")
                move(moving_key, moving_arm, moving_contact, vertical_speed, "ALIGN_V88_ARM1_MOVER_PRESS_AFTER_ARM2_HOLD")

                target_floor = float(surface_fn(config, moving_key, target, float(self.align.arm_contact_z(cfg, moving_key))))
                moving_target = xyz_at(moving_key, target, target_floor + moving_press_clearance_mm)
                move(moving_key, moving_arm, moving_target, drag_speed, "ALIGN_V88_ARM1_ROTATE_TRANSLATE_WITH_ARM2_HOLD")

                moving_clear = np.asarray([float(current[moving_key][0]), float(current[moving_key][1]), hover_z], np.float32)
                anchor_clear = np.asarray([float(current[anchor_key][0]), float(current[anchor_key][1]), hover_z], np.float32)
                print("[ALIGN-V88-PIVOT-SYNC] ARM1+ARM2 CLEAR simultaneously after drag")
                parallel_moves([
                    (moving_key, moving_arm, moving_clear, clear_speed, "ALIGN_V88_ARM1_CLEAR_PARALLEL"),
                    (anchor_key, anchor_arm, anchor_clear, clear_speed, "ALIGN_V88_ARM2_CLEAR_PARALLEL"),
                ])

                if not parallel_standby():
                    raise RuntimeError("ALIGN_V88_DUAL_STANDBY_NOT_VERIFIED")
                report["v67_pivot_executed"] = True
                report["v88_parallel_hover"] = True
                report["v88_arm2_first_hold"] = True
                report["v88_parallel_clear_standby"] = True
                self._v88_remember_executed_align(action_plan)
                print(
                    f"[ALIGN-V88-PIVOT-EXEC] ARM2 HOLD={np.round(anchor_source,1).tolist()} | "
                    f"ARM1 {np.round(source,1).tolist()} -> {np.round(target,1).tolist()} | "
                    f"HOVER parallel, ARM2 hold first, ARM1 drag, CLEAR/STANDBY parallel | "
                    f"ARM1 Z extra={self._v73_align_arm1_z_extra_mm:+.1f}mm "
                    f"ARM2 hold Z extra={float(self._v70_align_center_z_extra_mm)+float(self._v76_align_arm2_z_extra_mm):+.1f}mm speed={drag_speed:.2f}"
                )
                return True, "ACTION_OK", report
            except Exception as exc:
                report["exception"] = repr(exc)
                print(f"[ALIGN-V88-PIVOT-ERROR] {exc!r}")
                for key, arm in ((moving_key, moving_arm), (anchor_key, anchor_arm)):
                    try:
                        fb = self.align._d31_query_feedback(arm, timeout_s=0.8)
                        xyz = self.align._d31_feedback_xyz(fb)
                        if xyz is not None:
                            xyz = np.asarray(xyz, np.float32).reshape(3)
                            arm.move_goal(float(xyz[0]), float(xyz[1]), hover_z, tool_t, 0.35,
                                          move_command=move_command, stage=f"ALIGN_V88_{key.upper()}_RECOVERY_CLEAR",
                                          caller="_align_v67_intersection_pivot")
                            time.sleep(0.6)
                        self._v60_align_return_standby_verified(key, arms, cfg, move_command, "ALIGN-V88 pivot recovery")
                    except Exception as exc2:
                        print(f"[ALIGN-V88-PIVOT-RECOVERY-WARN] {key.upper()} {exc2!r}")
                return False, f"ALIGN_V88_PIVOT_EXCEPTION:{exc!r}", report

        def _install_align_press_drag_executor(self):
            original = getattr(self.align, "_align7_execute_arm2_angle_pull_60style", None)
            if not callable(original) or bool(getattr(original, "_bottom_vla_v56_press_drag", False)):
                return

            def wrapped(action_plan, arms, config, cfg, args, on_verified_start=None):
                if not isinstance(action_plan, dict) or not str(action_plan.get("v56_mode", "")).upper() in {"CENTER_VECTOR_PRESS", "RED_OUTER_PRESS"}:
                    return original(
                        action_plan, arms, config, cfg, args,
                        on_verified_start=on_verified_start,
                    )
                if (
                    str(action_plan.get("v56_mode", "")).upper() == "CENTER_VECTOR_PRESS"
                    and bool(action_plan.get("v67_intersection_pivot", False))
                ):
                    return self._v67_execute_intersection_pivot_press_drag(
                        action_plan, arms, config, cfg, args,
                        on_verified_start=on_verified_start,
                    )
                report = copy.deepcopy(action_plan)
                arm_key = str(action_plan.get("arm_key", "arm2")).lower()
                arm = arms.get(arm_key)
                if arm is None and cfg.send and not cfg.dry_run:
                    return False, f"{arm_key.upper()}_MISSING", report
                source = np.asarray(action_plan["source_board"], np.float64).reshape(2)
                target = np.asarray(action_plan["target_board"], np.float64).reshape(2)
                mode = str(action_plan.get("v56_mode", "ALIGN_PRESS"))
                press_clearance_mm = float(self._v56_align_press_clearance_mm)
                distance = float(np.linalg.norm(target - source))
                long_z_extra_mm = 0.0
                if mode.upper() == "CENTER_VECTOR_PRESS":
                    press_clearance_mm += float(self._v70_align_center_z_extra_mm)
                    if arm_key == "arm2":
                        press_clearance_mm += float(self._v76_align_arm2_z_extra_mm)
                    elif arm_key == "arm1":
                        press_clearance_mm += float(self._v87_align_arm1_center_z_extra_mm)
                        print(f"[ALIGN-V87-ARM1-Z] CENTER ARM1 contact/drag extra={self._v87_align_arm1_center_z_extra_mm:+.1f}mm")
                    if distance >= float(self._v85_align_long_move_threshold_mm):
                        long_z_extra_mm = float(self._v85_align_long_move_z_extra_mm)
                        press_clearance_mm += long_z_extra_mm
                        print(
                            f"[ALIGN-V85-LONG-Z] CENTER move={distance:.1f}mm -> "
                            f"{arm_key.upper()} contact/drag extra={long_z_extra_mm:+.1f}mm"
                        )
                elif mode.upper() == "RED_OUTER_PRESS":
                    press_clearance_mm += float(self._v58_align_red_outer_z_offset_mm)
                report["v85_long_move_mm"] = float(distance)
                report["v85_long_move_z_extra_mm"] = float(long_z_extra_mm)
                if cfg.dry_run or not cfg.send:
                    return True, "PREVIEW", report
                if callable(on_verified_start):
                    on_verified_start()

                move_command = int(getattr(args, "move_command", 104))
                tool_t = float(getattr(cfg, "tool_t_rad", 3.14))
                hover_z = float(max(float(getattr(cfg, "clear_z", 180.0)), float(getattr(args, "d31_hover_z", 180.0))))
                approach_speed = float(np.clip(getattr(args, "d31_approach_speed", 1.12), 0.45, 1.20))
                vertical_speed = float(np.clip(getattr(args, "d31_vertical_speed", 1.12), 0.35, 1.20))
                drag_speed = float(np.clip(
                    self._v70_align_drag_speed if mode.upper() == "CENTER_VECTOR_PRESS" else self._v62_align_drag_speed,
                    0.25, 0.55,
                ))
                clear_speed = float(np.clip(getattr(args, "align6_arm2_clear_speed", 0.50), 0.25, 0.70))
                close_angle = float(np.clip(getattr(args, "d31_grip_close", 3.05), 2.90, 3.14))
                surface_fn = getattr(self.align, "_c93_surface_z_at_board", None)
                if not callable(surface_fn):
                    return False, "ALIGN_V56_SURFACE_API_MISSING", report

                def xyz_at(board_point, z):
                    rx, ry = self.align.board_to_arm_xy(config, arm_key, float(board_point[0]), float(board_point[1]))
                    return np.asarray([float(rx), float(ry), float(z)], np.float32)

                current = None
                try:
                    arm.set_gripper(
                        close_angle, delay=0.25,
                        stage=f"ALIGN_V56_{mode}_CLOSED_BEFORE_APPROACH",
                        caller="_align_v56_press_drag",
                    )
                    approach = xyz_at(source, hover_z)
                    ok, why, _ = self.align._d31_move_verified(
                        arm, approach, cfg, approach_speed, move_command, args,
                        f"ALIGN_V56_{mode}_APPROACH_CLOSED",
                        wait_from=None,
                    )
                    if not ok:
                        return False, why, report
                    current = approach

                    fallback_floor = float(self.align.arm_contact_z(cfg, arm_key))
                    source_floor = float(surface_fn(config, arm_key, source, fallback_floor))
                    contact_z = float(source_floor + press_clearance_mm)
                    contact = xyz_at(source, contact_z)
                    ok, why, _ = self.align._d31_move_verified(
                        arm, contact, cfg, vertical_speed, move_command, args,
                        f"ALIGN_V56_{mode}_PRESS_DOWN_CLOSED",
                        wait_from=current,
                    )
                    if not ok:
                        return False, why, report
                    current = contact

                    floor = float(surface_fn(config, arm_key, target, fallback_floor))
                    z = float(floor + press_clearance_mm)
                    use_v86_two_segment = bool(
                        mode.upper() == "CENTER_VECTOR_PRESS"
                        and arm_key == "arm2"
                        and distance >= float(self._v85_align_long_move_threshold_mm)
                    )
                    if use_v86_two_segment:
                        midpoint = source + 0.5 * (target - source)
                        mid_floor = float(surface_fn(config, arm_key, midpoint, fallback_floor))
                        common_drag_z = float(min(contact_z, mid_floor + press_clearance_mm, z))
                        mid_waypoint = xyz_at(midpoint, common_drag_z)
                        waypoint = xyz_at(target, common_drag_z)
                        seg1_speed = float(self._v86_align_arm2_segment1_speed)
                        seg2_speed = float(self._v86_align_arm2_segment2_speed)
                        print(
                            f"[ALIGN-V86-ARM2-2SEG] move={distance:.1f}mm mid=50% commonZ={common_drag_z:.1f} "
                            f"speed={seg1_speed:.2f}->{seg2_speed:.2f}; no added sleep"
                        )
                        ok, why, _ = self.align._d31_move_verified(
                            arm, mid_waypoint, cfg, seg1_speed, move_command, args,
                            f"ALIGN_V86_{mode}_PRESS_DRAG_SEG1_LOW_Z",
                            wait_from=current,
                        )
                        if not ok:
                            return False, why, report
                        current = mid_waypoint
                        ok, why, _ = self.align._d31_move_verified(
                            arm, waypoint, cfg, seg2_speed, move_command, args,
                            f"ALIGN_V86_{mode}_PRESS_DRAG_SEG2_LOW_Z",
                            wait_from=current,
                        )
                        if not ok:
                            return False, why, report
                        current = waypoint
                        report["v56_press_drag_waypoints"] = [
                            {"board": [float(midpoint[0]), float(midpoint[1])], "roarm": mid_waypoint.astype(float).tolist()},
                            {"board": [float(target[0]), float(target[1])], "roarm": waypoint.astype(float).tolist()},
                        ]
                        report["v86_arm2_two_segment"] = True
                        report["v86_arm2_common_drag_z"] = float(common_drag_z)
                        report["v86_arm2_segment_speeds"] = [float(seg1_speed), float(seg2_speed)]
                    else:
                        waypoint = xyz_at(target, z)
                        ok, why, _ = self.align._d31_move_verified(
                            arm, waypoint, cfg, drag_speed, move_command, args,
                            f"ALIGN_V62_{mode}_PRESS_DRAG_CONTINUOUS",
                            wait_from=current,
                        )
                        if not ok:
                            return False, why, report
                        current = waypoint
                        report["v56_press_drag_waypoints"] = [{"board": [float(target[0]), float(target[1])], "roarm": waypoint.astype(float).tolist()}]
                    report["v62_continuous_drag"] = True

                    clear = np.asarray([float(current[0]), float(current[1]), hover_z], np.float32)
                    ok, why, _ = self.align._d31_move_verified(
                        arm, clear, cfg, clear_speed, move_command, args,
                        f"ALIGN_V56_{mode}_PURE_Z_CLEAR_CLOSED",
                        wait_from=current,
                    )
                    if not ok:
                        return False, why, report
                    if not self._v60_align_return_standby_verified(
                        arm_key, arms, cfg, move_command,
                        f"ALIGN-V60 {mode} press-drag; gripper remains closed",
                    ):
                        return False, f"ALIGN_V60_{arm_key.upper()}_STANDBY_NOT_VERIFIED", report
                    self._v88_remember_executed_align(action_plan)
                    print(
                        f"[ALIGN-V56-PRESS-DRAG] {arm_key.upper()} mode={mode} "
                        f"source=({source[0]:.1f},{source[1]:.1f}) target=({target[0]:.1f},{target[1]:.1f}) "
                        f"move={distance:.1f}mm; NO OPEN / NO GRASP / NO 18mm LIFT / NO RELEASE"
                    )
                    return True, "ACTION_OK", report
                except Exception as exc:
                    report["exception"] = repr(exc)
                    print(f"[ALIGN-V56-PRESS-DRAG-ERROR] {arm_key.upper()} {exc!r}")
                    try:
                        fb = self.align._d31_query_feedback(arm, timeout_s=0.8)
                        xyz = self.align._d31_feedback_xyz(fb)
                        if xyz is not None:
                            xyz = np.asarray(xyz, np.float32).reshape(3)
                            clear = np.asarray([float(xyz[0]), float(xyz[1]), hover_z], np.float32)
                            arm.move_goal(
                                float(clear[0]), float(clear[1]), float(clear[2]), tool_t, 0.35,
                                move_command=move_command,
                                stage="ALIGN_V56_EXCEPTION_PURE_Z_CLEAR_CLOSED",
                                caller="_align_v56_press_drag",
                            )
                            time.sleep(0.8)
                        self._v60_align_return_standby_verified(
                            arm_key, arms, cfg, move_command,
                            "ALIGN-V60 press-drag exception return; gripper remains closed",
                        )
                    except Exception as exc2:
                        print(f"[ALIGN-V56-PRESS-DRAG-RECOVERY-WARN] {exc2!r}")
                    return False, f"ALIGN_V56_PRESS_DRAG_EXCEPTION:{exc!r}", report

            wrapped._bottom_vla_v56_press_drag = True
            self.align._align7_execute_arm2_angle_pull_60style = wrapped

        def _install_align_v56_overlay(self):
            original = getattr(self.align, "_align4_draw_midpoint_overlay", None)
            if not callable(original) or bool(getattr(original, "_bottom_vla_v56_overlay", False)):
                return

            def wrapped(image, H, obs, plan, phase, config, cfg, args, correction_count=0, cached_action=None, cached_decision=None):
                out = original(
                    image, H, obs, plan, phase, config, cfg, args,
                    correction_count=correction_count,
                    cached_action=cached_action,
                    cached_decision=cached_decision,
                )
                action = cached_action
                if out is None or H is None or not isinstance(action, dict) or not str(action.get("v56_mode", "")):
                    return out
                try:
                    src = np.asarray(action["source_board"], np.float64).reshape(2)
                    tgt = np.asarray(action["target_board"], np.float64).reshape(2)
                    ps = self.d56.board_to_pixel(H, float(src[0]), float(src[1]))
                    pt = self.d56.board_to_pixel(H, float(tgt[0]), float(tgt[1]))
                    if ps is None or pt is None:
                        return out
                    a = (int(round(ps[0])), int(round(ps[1])))
                    b = (int(round(pt[0])), int(round(pt[1])))
                    mode = str(action.get("v56_mode", "ALIGN_PRESS"))
                    arm_key = str(action.get("arm_key", "arm2")).upper()
                    self.cv2.circle(out, a, 13, (150, 0, 255), -1, self.cv2.LINE_AA)
                    self.cv2.arrowedLine(out, a, b, (150, 0, 255), 5, self.cv2.LINE_AA, tipLength=0.14)
                    x0, y0 = a[0]+16, max(22, a[1]-34)
                    self.cv2.rectangle(out, (x0-4, y0-18), (min(out.shape[1]-2, x0+350), y0+7), (0,0,0), -1)
                    self.cv2.putText(out, f"{arm_key} {mode} CLOSED PRESS", (x0, y0), self.cv2.FONT_HERSHEY_SIMPLEX, 0.47, (150,0,255), 2, self.cv2.LINE_AA)
                    if mode.upper() == "CENTER_VECTOR_PRESS" and bool(action.get("v67_intersection_pivot", False)):
                        anchor = np.asarray(action.get("v67_pivot_source_board"), np.float64).reshape(2)
                        pa = self.d56.board_to_pixel(H, float(anchor[0]), float(anchor[1]))
                        if pa is not None:
                            ap = (int(round(pa[0])), int(round(pa[1])))
                            self.cv2.circle(out, ap, 13, (255, 0, 0), -1, self.cv2.LINE_AA)
                            ax0, ay0 = ap[0]+16, max(22, ap[1]-20)
                            self.cv2.rectangle(out, (ax0-4, ay0-18), (min(out.shape[1]-2, ax0+220), ay0+7), (0,0,0), -1)
                            self.cv2.putText(out, "ARM2 PIVOT HOLD", (ax0, ay0), self.cv2.FONT_HERSHEY_SIMPLEX, 0.47, (255,0,0), 2, self.cv2.LINE_AA)
                except Exception:
                    pass
                return out

            wrapped._bottom_vla_v56_overlay = True
            self.align._align4_draw_midpoint_overlay = wrapped


        def _prepare_align(self):
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

            self._v53_align_current_obs = None
            self._v53_align_capture_active = True
            self._v85_align_fast_mode = True
            try:
                t0 = time.monotonic()
                bundle = None
                obs = None
                mask_info = None
                for align_attempt in range(1, int(self._v98_align_mask_retry_count) + 1):
                    bundle = self._capture_i_frame_from_live("D56_WAIST_LIFT_LAYDOWN")
                    obs = self._infer_for_action("D56_WAIST_LIFT_LAYDOWN", bundle.corrected)
                    mask_info = self._v30_mask_info(SimpleNamespace(observation=obs))
                    if bool(mask_info.get("valid", False)) and not bool(mask_info.get("fallback", False)):
                        print(
                            f"[ALIGN-V99-MASK-OK] attempt={align_attempt} source={mask_info.get('source','MASK')} "
                            f"area={float(mask_info.get('area_px',0.0)):.0f}px"
                        )
                        break
                    print(
                        f"[ALIGN-V99-MASK-RETRY] attempt={align_attempt}/{int(self._v98_align_mask_retry_count)} "
                        f"valid={bool((mask_info or {}).get('valid',False))} "
                        f"fallback={bool((mask_info or {}).get('fallback',False))} source={(mask_info or {}).get('source','-')}"
                    )
                infer_ms = (time.monotonic() - t0) * 1000.0
                if obs is None or mask_info is None or not bool(mask_info.get("valid", False)) or bool(mask_info.get("fallback", False)):
                    with self.state_lock:
                        self.status = "ALIGN NO PLAN: TRUSTED YOLO MASK REQUIRED - REJUDGE"
                    print("[ALIGN-V99-MASK-BLOCK] only true fallback/BGDIFF Mask is blocked; normal YOLO Mask passes")
                    return
                self._v53_align_current_obs = obs
                self._v53_align_initial_dual_done = True
                self.align_phase = "CENTER_VECTOR"

                seam_x = float(self.align._c93_taught_trace_calibration(self.config, self.align_args)["boundary_x_mm"])
                tp = time.monotonic()
                action_plan, decision = self.align._align7_build_auto_plan(
                    obs, self.H_raw, None, seam_x, self.align_correction_count,
                    self.config, self.align_cfg, self.align_args,
                )
                plan_ms = (time.monotonic() - tp) * 1000.0
                if not isinstance(decision, dict):
                    decision = {"reason": str(decision)}
                action_plan, decision = self._v88_align_repeat_guard(obs, seam_x, action_plan, decision)
                if not isinstance(decision, dict):
                    decision = {"reason": str(decision)}
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
                    print(
                        f"[FINISH-V111-CENTRAL-ONLY] ALIGN angle={angle_abs:.1f}deg <= "
                        f"{float(self.front.align_finish_angle_deg):.1f}deg, but internal ALIGN AUTO-FINISH is disabled; "
                        "keep the current ALIGN plan/decision and let the next fresh MODEL+LEGACY gate own FINISH"
                    )

                plan_ok = action_plan is not None
                reason = str(
                    decision.get("decision")
                    or decision.get("reason")
                    or ("ALIGN_READY" if plan_ok else "CENTER_VECTOR_ALIGN_NO_PLAN")
                )
                current_state = {
                    "waist_y_mm": None,
                    "waist_target_y_mm": float(self.front.align_waist_target_y_mm),
                    "required_upward_pull_mm": None,
                    "dual_trigger_mm": float(self.front.align_dual_waist_min_pull_mm),
                    "selected_submode": "CENTER_VECTOR_FAST",
                    "d60_plan_ok": False,
                    "v85_fast_align": True,
                }
                runtime = {}
                if plan_ok:
                    runtime = {
                        "kind": "ARM2_CORRECTION",
                        "action_plan": copy.deepcopy(action_plan),
                        "decision": copy.deepcopy(decision),
                        "current_state": copy.deepcopy(current_state),
                    }
                    proxy = SimpleNamespace(
                        ok=True,
                        reason=reason,
                        action="ALIGN_ARM2_CORRECTION",
                        metrics={
                            "phase": "ARM2_ALIGN_FAST",
                            "action_plan": v23._json_safe(action_plan),
                            "decision": v23._json_safe(decision),
                            "current_state": copy.deepcopy(current_state),
                        },
                        arm_points={},
                    )
                else:
                    proxy = SimpleNamespace(
                        ok=False, reason=reason, action="ALIGN",
                        metrics={"phase": "ARM2_ALIGN_FAST", "decision": copy.deepcopy(decision), "current_state": copy.deepcopy(current_state)},
                        arm_points={},
                    )

                canvas = bundle.raw.copy()
                try:
                    canvas = self.align._align4_draw_midpoint_overlay(
                        canvas, self.H_raw, obs, None, "ARM2_ALIGN", self.config,
                        self.align_cfg, self.align_args, self.align_correction_count,
                        cached_action=action_plan, cached_decision=decision,
                    )
                except Exception:
                    try:
                        canvas = self.align._align3_draw_overlay(canvas, self.H_raw, obs, self.config, self.align_args)
                    except Exception:
                        pass
                canvas = self._draw_align_white_reference(canvas)
                token = self._store_align_runtime(runtime) if plan_ok else ""
                diagnostics = {
                    "align_runtime_token": token,
                    "align_phase": self.align_phase,
                    "align_correction_count": self.align_correction_count,
                    "align_decision": copy.deepcopy(decision),
                    "align_current_state": copy.deepcopy(current_state),
                    "align_center_vector_fallback": False,
                    "align_dual_waist_failure": "V85_FAST_PATH_D60_SKIPPED",
                    "inference_ms": float(infer_ms),
                    "plan_ms": float(plan_ms),
                    "v85_align_fast_path": True,
                    **self._semantic_metadata(),
                }
                locked = base.LockedPlan(
                    None, "ALIGN", bundle, obs, proxy, None, canvas, self.H_raw.copy(), time.time(),
                    bool(plan_ok), str(reason), None if plan_ok else str(reason), diagnostics,
                )
                obs_record = self.recorder.save_observation(locked, "BEFORE_ACTION", self._environment_metadata())
                locked.observation_record = obs_record
                self.recorder.save_decision(locked, obs_record)
                align_complete_no_plan = bool((not plan_ok) and ("ALIGN_COMPLETE" in str(reason).upper()))
                with self.state_lock:
                    self.locked = locked
                    self.display_image = canvas.copy()
                    self.status = "ALIGN PLAN READY: AUTO EXECUTE" if plan_ok else f"ALIGN NO PLAN: {reason}"
                self._decorate_current_records()
                print(
                    f"[ALIGN-V85-FAST] infer={infer_ms:.0f}ms plan={plan_ms:.0f}ms "
                    f"ready={plan_ok} mode={str((action_plan or {}).get('v56_mode','NONE'))}; D60 waist/ribbon planner skipped"
                )
                if align_complete_no_plan:
                    with self.state_lock:
                        self.locked = None
                        self.semantic_selected = None
                        self.pending_semantic = None
                        self.selected_action = None
                        self.status = "ALIGN COMPLETE - FRESH AUTO JUDGE NEXT ACTION"
                    self._v101_skip_align_autojudge_once = True
                    print(f"[ALIGN-V101-FALLTHROUGH] reason={reason}; ALIGN is excluded once and a fresh next-action decision is scheduled")
                    self._schedule_auto_judge("after_align_complete", 0.06)
            finally:
                self._v85_align_fast_mode = False
                self._v53_align_capture_active = False

        def _candidate_board_center(self, cand, H):
            g = cand.get("d21_geometry", {}) if isinstance(cand, dict) else {}
            if isinstance(g, dict):
                p = self._finite_point2(g.get("center_board"))
                if p is not None:
                    return p
            p = self._finite_point2(cand.get("center_board")) if isinstance(cand, dict) else None
            if p is not None:
                return p
            if isinstance(cand, dict):
                px = self._finite_point2(cand.get("center_px"))
                fn = getattr(self.d55, "pixel_to_board", None)
                if px is not None and callable(fn):
                    try:
                        return self._finite_point2(fn(H, float(px[0]), float(px[1])))
                    except Exception:
                        pass
            return None

        def _candidate_board_normal(self, cand):
            if not isinstance(cand, dict):
                return None
            g = cand.get("d21_geometry", {})
            if isinstance(g, dict):
                n = self._finite_point2(g.get("normal_board"))
                if n is not None:
                    nn = float(np.linalg.norm(n))
                    if nn > 1e-6:
                        return n / nn
                t = self._finite_point2(g.get("tangent_board"))
                if t is not None:
                    tn = float(np.linalg.norm(t))
                    if tn > 1e-6:
                        t = t / tn
                        return np.asarray([-t[1], t[0]], np.float64)
            return None

        def _v35_axis_error_deg(self, a, b):
            aa = self._finite_point2(a)
            bb = self._finite_point2(b)
            if aa is None or bb is None:
                return None
            an = float(np.linalg.norm(aa))
            bn = float(np.linalg.norm(bb))
            if an < 1e-6 or bn < 1e-6:
                return None
            dot = float(np.clip(abs(float(np.dot(aa / an, bb / bn))), 0.0, 1.0))
            return float(math.degrees(math.acos(dot)))

        def _v35_local_wrinkle_axis(self, cand, heat, H):
            report = {
                "checked": False,
                "accepted": False,
                "reason": "UNAVAILABLE",
                "axis_error_deg": None,
                "local_linearity": None,
                "local_point_count": 0,
            }
            if not isinstance(cand, dict) or heat is None or H is None:
                return None, None, report
            geom = cand.get("d21_geometry", {})
            if not isinstance(geom, dict):
                return None, None, report
            shape = str(geom.get("shape_class", cand.get("d21_shape_class", "IRREGULAR"))).upper()
            if shape not in {"LINE", "CURVE"}:
                report.update({"reason": "NON_DIRECTIONAL_SHAPE", "shape_class": shape})
                return None, None, report
            binary = getattr(heat, "binary", None)
            center_px = self._finite_point2(cand.get("center_px"))
            if binary is None or center_px is None:
                report.update({"checked": True, "reason": "BINARY_OR_CENTER_UNAVAILABLE", "shape_class": shape})
                return None, None, report
            try:
                mask = (np.asarray(binary) > 0).astype(np.uint8)
                if mask.ndim != 2 or mask.size == 0:
                    report.update({"checked": True, "reason": "INVALID_BINARY", "shape_class": shape})
                    return None, None, report
                h, w = mask.shape[:2]
                cx = int(np.clip(round(float(center_px[0])), 0, w - 1))
                cy = int(np.clip(round(float(center_px[1])), 0, h - 1))
                nlab, labels = self.cv2.connectedComponents(mask, connectivity=8)
                lid = int(labels[cy, cx])
                if lid <= 0:
                    r = 10
                    x0, x1 = max(0, cx - r), min(w, cx + r + 1)
                    y0, y1 = max(0, cy - r), min(h, cy + r + 1)
                    yy, xx = np.nonzero(labels[y0:y1, x0:x1] > 0)
                    if len(xx) == 0:
                        report.update({"checked": True, "reason": "CENTER_NOT_ON_COMPONENT", "shape_class": shape})
                        return None, None, report
                    xx = xx + x0
                    yy = yy + y0
                    nearest = int(np.argmin((xx - cx) ** 2 + (yy - cy) ** 2))
                    lid = int(labels[int(yy[nearest]), int(xx[nearest])])
                if lid <= 0 or lid >= int(nlab):
                    report.update({"checked": True, "reason": "COMPONENT_NOT_FOUND", "shape_class": shape})
                    return None, None, report
                yy, xx = np.nonzero(labels == lid)
                if len(xx) < 16:
                    report.update({"checked": True, "reason": "COMPONENT_TOO_SMALL", "shape_class": shape, "local_point_count": int(len(xx))})
                    return None, None, report
                major_px = float(cand.get("major_length_px", 0.0) or 0.0)
                radius_px = float(np.clip(0.32 * major_px if major_px > 0.0 else 42.0, 24.0, 58.0))
                keep = (xx.astype(np.float64) - float(center_px[0])) ** 2 + (yy.astype(np.float64) - float(center_px[1])) ** 2 <= radius_px ** 2
                lxx = xx[keep]
                lyy = yy[keep]
                if len(lxx) < 16:
                    lxx, lyy = xx, yy
                if len(lxx) > 900:
                    step = int(math.ceil(len(lxx) / 900.0))
                    lxx = lxx[::step]
                    lyy = lyy[::step]
                pts_px = np.stack([lxx, lyy], axis=1).astype(np.float32).reshape(-1, 1, 2)
                pts_board = self.cv2.perspectiveTransform(pts_px, np.asarray(H, np.float32)).reshape(-1, 2).astype(np.float64)
                finite = np.all(np.isfinite(pts_board), axis=1)
                pts_board = pts_board[finite]
                if len(pts_board) < 12:
                    report.update({"checked": True, "reason": "BOARD_POINTS_TOO_FEW", "shape_class": shape, "local_point_count": int(len(pts_board))})
                    return None, None, report
                centered = pts_board - np.mean(pts_board, axis=0, keepdims=True)
                cov = centered.T @ centered / max(1, len(centered) - 1)
                vals, vecs = np.linalg.eigh(cov)
                order = np.argsort(vals)
                small = max(float(vals[order[0]]), 1e-6)
                large = max(float(vals[order[-1]]), 0.0)
                linearity = float(large / small)
                tangent = np.asarray(vecs[:, order[-1]], np.float64).reshape(2)
                tn = float(np.linalg.norm(tangent))
                if tn < 1e-6 or linearity < 1.35:
                    report.update({
                        "checked": True,
                        "reason": "LOCAL_AXIS_NOT_LINEAR_ENOUGH",
                        "shape_class": shape,
                        "local_linearity": linearity,
                        "local_point_count": int(len(pts_board)),
                        "radius_px": radius_px,
                    })
                    return None, None, report
                tangent = tangent / tn
                global_tangent = self._finite_point2(geom.get("tangent_board"))
                if global_tangent is None:
                    global_normal = self._finite_point2(geom.get("normal_board"))
                    if global_normal is not None:
                        global_tangent = np.asarray([global_normal[1], -global_normal[0]], np.float64)
                axis_error = self._v35_axis_error_deg(tangent, global_tangent)
                if axis_error is None:
                    report.update({"checked": True, "reason": "GLOBAL_AXIS_UNAVAILABLE", "shape_class": shape})
                    return None, None, report
                normal = np.asarray([-tangent[1], tangent[0]], np.float64)
                old_normal = self._finite_point2(geom.get("normal_board"))
                if old_normal is not None and float(np.dot(normal, old_normal)) < 0.0:
                    normal = -normal
                accepted = bool(axis_error <= 20.0)
                report.update({
                    "checked": True,
                    "accepted": accepted,
                    "reason": "LOCAL_GLOBAL_AXIS_CONSENSUS" if accepted else "LOCAL_GLOBAL_AXIS_DISAGREE",
                    "shape_class": shape,
                    "axis_error_deg": float(axis_error),
                    "local_linearity": linearity,
                    "local_point_count": int(len(pts_board)),
                    "radius_px": radius_px,
                    "local_tangent_board": [float(tangent[0]), float(tangent[1])],
                    "local_normal_board": [float(normal[0]), float(normal[1])],
                })
                if not accepted:
                    return None, None, report
                return tangent, normal, report
            except Exception as exc:
                report.update({"checked": True, "reason": f"LOCAL_AXIS_ERROR:{type(exc).__name__}", "shape_class": shape})
                return None, None, report

        def _v35_direction_candidates(self, heat, H):
            candidates = list(getattr(heat, "candidates", []) or []) if heat is not None else []
            refined = []
            rejected = []
            for index, cand in enumerate(candidates):
                if not isinstance(cand, dict):
                    refined.append(cand)
                    continue
                clone = copy.deepcopy(cand)
                geom = clone.get("d21_geometry", {})
                shape = str(geom.get("shape_class", clone.get("d21_shape_class", "IRREGULAR"))).upper() if isinstance(geom, dict) else "IRREGULAR"
                clone["_v35_original_index"] = int(index)
                if shape not in {"LINE", "CURVE"}:
                    clone["d55v35_direction_check"] = {"checked": False, "accepted": True, "reason": "NON_DIRECTIONAL_SHAPE", "shape_class": shape}
                    refined.append(clone)
                    continue
                tangent, normal, report = self._v35_local_wrinkle_axis(clone, heat, H)
                clone["d55v35_direction_check"] = dict(report)
                if tangent is None or normal is None or not bool(report.get("accepted", False)):
                    rejected.append((index, report))
                    print(
                        f"[D55-V35-DIR-REJECT] candidate={index + 1} shape={shape} "
                        f"reason={report.get('reason')} axisErr={report.get('axis_error_deg')} "
                        f"localLinearity={report.get('local_linearity')}"
                    )
                    continue
                geom = dict(geom or {})
                geom["tangent_board"] = np.asarray(tangent, np.float64)
                geom["normal_board"] = np.asarray(normal, np.float64)
                geom["d55v35_local_axis_refined"] = True
                geom["d55v35_axis_error_deg"] = float(report.get("axis_error_deg", 0.0))
                geom["d55v35_local_linearity"] = float(report.get("local_linearity", 0.0))
                clone["d21_geometry"] = geom
                refined.append(clone)
                print(
                    f"[D55-V35-DIR-ACCEPT] candidate={index + 1} shape={shape} "
                    f"axisErr={float(report.get('axis_error_deg', 0.0)):.1f}deg "
                    f"localLinearity={float(report.get('local_linearity', 0.0)):.2f}"
                )
            return refined, rejected

        def _v35_final_direction_report(self, plan, candidates):
            report = {
                "checked": False,
                "passed": True,
                "reason": "NO_DIRECTIONAL_CANDIDATE",
                "max_axis_error_deg": None,
                "arm_axis_error_deg": {},
            }
            if plan is None or not bool(getattr(plan, "ok", False)):
                return report
            ci_value = getattr(plan, "candidate_index", -1)
            ci = int(ci_value) if ci_value is not None else -1
            if ci < 0 or ci >= len(candidates) or not isinstance(candidates[ci], dict):
                return report
            cand = candidates[ci]
            direction_check = cand.get("d55v35_direction_check", {})
            if not bool(direction_check.get("checked", False)) or not bool(direction_check.get("accepted", False)):
                return report
            expected = self._candidate_board_normal(cand)
            if expected is None:
                report.update({"checked": True, "passed": False, "reason": "EXPECTED_NORMAL_UNAVAILABLE"})
                return report
            errors = {}
            arm_points = dict(getattr(plan, "arm_points", {}) or {})
            for arm_key, points in arm_points.items():
                if not isinstance(points, dict):
                    continue
                src = self._finite_point2(points.get("source_board"))
                dst = self._finite_point2(points.get("target_board"))
                if src is None or dst is None:
                    continue
                move = dst - src
                move_n = float(np.linalg.norm(move))
                if move_n <= 5.0:
                    continue
                error = self._v35_axis_error_deg(move / move_n, expected)
                if error is not None:
                    errors[str(arm_key)] = float(error)
            if not errors:
                report.update({"checked": True, "passed": False, "reason": "NO_MOVING_SWEEP_VECTOR"})
                return report
            max_error = float(max(errors.values()))
            passed = bool(max_error <= 20.0)
            report.update({
                "checked": True,
                "passed": passed,
                "reason": "FINAL_SWEEP_PERPENDICULAR_OK" if passed else "FINAL_SWEEP_AXIS_DRIFT",
                "max_axis_error_deg": max_error,
                "arm_axis_error_deg": errors,
                "tolerance_deg": 20.0,
            })
            return report

        def _v35_build_direction_checked_super(self, obs, heat, H, config, cfg, args, candidates):
            pool = list(candidates)
            last_plan = None
            attempts = 0
            while pool and attempts < 6:
                attempts += 1
                local_heat = copy.copy(heat)
                local_heat.candidates = pool
                plan = super()._d55_build_perpendicular_plan(obs, local_heat, H, config, cfg, args)
                last_plan = plan
                if plan is None or not bool(getattr(plan, "ok", False)):
                    return plan
                final_report = self._v35_final_direction_report(plan, pool)
                metrics = dict(getattr(plan, "metrics", {}) or {})
                metrics["d55v35_final_direction"] = dict(final_report)
                plan.metrics = metrics
                if bool(final_report.get("passed", True)):
                    if bool(final_report.get("checked", False)):
                        print(
                            f"[D55-V35-FINAL-DIR] PASS candidate={int(getattr(plan, 'candidate_index', -1)) + 1} "
                            f"maxErr={float(final_report.get('max_axis_error_deg', 0.0)):.1f}deg tol=20.0deg"
                        )
                    return plan
                ci_value = getattr(plan, "candidate_index", -1)
                ci = int(ci_value) if ci_value is not None else -1
                print(
                    f"[D55-V35-FINAL-DIR] REJECT candidate={ci + 1} "
                    f"reason={final_report.get('reason')} maxErr={final_report.get('max_axis_error_deg')}"
                )
                if ci < 0 or ci >= len(pool):
                    plan.ok = False
                    plan.reason = f"D55-V35 direction blocked: {final_report.get('reason')}"
                    return plan
                pool = [cand for i, cand in enumerate(pool) if i != ci]
            if last_plan is not None:
                last_plan.ok = False
                last_plan.reason = "D55-V35 direction blocked: no candidate passed final sweep-axis validation"
                return last_plan
            return self.d55.DualWrinkleStretchPlan(False, "D55-V35 direction blocked: no direction-safe wrinkle candidate", metrics={
                "d55_failure_category": "NO_DIRECTION_SAFE_WRINKLE",
                "d55v35_direction_gate": True,
            })

        def _fold_recovery_geometry(self, obs, H, candidates):
            pts = self._pose_board_points(obs)
            waist = pts.get("waist_center")
            crotch = pts.get("crotch")
            left = pts.get("left_hem_center")
            right = pts.get("right_hem_center")
            if any(x is None for x in (waist, crotch, left, right)):
                return None
            left_len = float(np.linalg.norm(left - crotch))
            right_len = float(np.linalg.norm(right - crotch))
            if min(left_len, right_len) < 25.0 or max(left_len, right_len) < 1e-6:
                return None
            leg_balance = float(min(left_len, right_len) / max(left_len, right_len))
            left_width = None
            right_width = None
            if pts.get("img_left_hem_outer") is not None and pts.get("img_left_hem_inner") is not None:
                left_width = float(np.linalg.norm(pts["img_left_hem_outer"] - pts["img_left_hem_inner"]))
            if pts.get("img_right_hem_inner") is not None and pts.get("img_right_hem_outer") is not None:
                right_width = float(np.linalg.norm(pts["img_right_hem_outer"] - pts["img_right_hem_inner"]))
            hem_balance = 1.0
            if left_width is not None and right_width is not None and max(left_width, right_width) > 1e-6:
                hem_balance = float(min(left_width, right_width) / max(left_width, right_width))
            folded = bool(leg_balance < 0.70 or hem_balance < 0.58)
            if not folded:
                return None
            folded_side = "left" if left_len <= right_len else "right"
            folded_hem = left if folded_side == "left" else right
            other_hem = right if folded_side == "left" else left
            body = crotch - waist
            body_n = float(np.linalg.norm(body))
            leg = folded_hem - crotch
            leg_n = float(np.linalg.norm(leg))
            if body_n < 20.0 or leg_n < 20.0:
                return None
            body_u = body / body_n
            leg_u = leg / leg_n
            proj = waist + body_u * float(np.dot(folded_hem - waist, body_u))
            outward = folded_hem - proj
            outward_n = float(np.linalg.norm(outward))
            if outward_n < 8.0:
                side = np.asarray([-body_u[1], body_u[0]], np.float64)
                if float(np.dot(side, folded_hem - crotch)) < 0.0:
                    side = -side
                outward_u = side
            else:
                outward_u = outward / outward_n
            desired = 0.68 * leg_u + 0.62 * outward_u
            desired_n = float(np.linalg.norm(desired))
            if desired_n < 1e-6:
                return None
            desired = desired / desired_n
            folded_lateral = abs(float(body_u[0] * (folded_hem - waist)[1] - body_u[1] * (folded_hem - waist)[0]))
            folded_cross = float(body_u[0] * (folded_hem - crotch)[1] - body_u[1] * (folded_hem - crotch)[0])
            recovery = []
            boundary_ids = set()
            min_normal_alignment = math.cos(math.radians(30.0))
            for index, cand in enumerate(candidates):
                center = self._candidate_board_center(cand, H)
                normal = self._candidate_board_normal(cand)
                if center is None:
                    continue
                rel = center - crotch
                t = float(np.dot(rel, leg) / max(leg_n * leg_n, 1e-6))
                lateral = abs(float(body_u[0] * (center - waist)[1] - body_u[1] * (center - waist)[0]))
                cross = float(body_u[0] * rel[1] - body_u[1] * rel[0])
                same_side = bool(abs(folded_cross) < 5.0 or abs(cross) < 5.0 or folded_cross * cross > 0.0)
                central_limit = max(14.0, 0.30 * folded_lateral)
                if lateral <= central_limit and 0.0 <= t <= 0.82:
                    boundary_ids.add(index)
                    continue
                if not same_side or t < 0.10 or t > 1.15 or normal is None:
                    continue
                normal_align = float(abs(np.dot(normal, desired)))
                if normal_align < min_normal_alignment:
                    continue
                major = max(0.0, float(cand.get("major_length_px", 0.0) or 0.0))
                linearity = max(1.0, float(cand.get("linearity", 1.0) or 1.0))
                center_bias = 1.0 - min(1.0, abs(t - 0.58) / 0.58)
                score = 110.0 * normal_align + 30.0 * center_bias + min(30.0, major / 6.0) + min(15.0, (linearity - 1.0) * 5.0)
                recovery.append((score, index, cand, center, normal, t))
            recovery.sort(key=lambda x: x[0], reverse=True)
            return {
                "folded_side": folded_side,
                "leg_balance": leg_balance,
                "hem_balance": hem_balance,
                "waist": waist,
                "crotch": crotch,
                "folded_hem": folded_hem,
                "other_hem": other_hem,
                "body_u": body_u,
                "body_len": body_n,
                "desired": desired,
                "recovery": recovery,
                "boundary_ids": boundary_ids,
                "d55v35_min_normal_alignment": float(min_normal_alignment),
            }

        def _v35_d55_build_perpendicular_plan(self, obs, heat, H, config, cfg, args):
            candidates, direction_rejected = self._v35_direction_candidates(heat, H)
            if not candidates:
                print(f"[D55-V35-DIR-BLOCK] no direction-safe candidates; rejected={len(direction_rejected)}")
                return self.d55.DualWrinkleStretchPlan(False, "D55-V35 no direction-safe wrinkle candidate", metrics={
                    "d55_failure_category": "NO_DIRECTION_SAFE_WRINKLE",
                    "d55v35_direction_rejected_count": int(len(direction_rejected)),
                })
            geom = self._fold_recovery_geometry(obs, H, candidates) if candidates else None
            if geom is None:
                return self._v35_build_direction_checked_super(obs, heat, H, config, cfg, args, candidates)
            desired = np.asarray(geom["desired"], np.float64).reshape(2)
            min_fold_direction_alignment = math.cos(math.radians(30.0))
            for _, index, cand, center, normal, t in list(geom["recovery"])[:5]:
                try:
                    local_heat = copy.copy(heat)
                    local_heat.candidates = [copy.deepcopy(cand)]
                    plan = super()._d55_build_perpendicular_plan(obs, local_heat, H, config, cfg, args)
                except Exception:
                    continue
                if plan is None or not bool(getattr(plan, "ok", False)):
                    continue
                arm_points = dict(getattr(plan, "arm_points", {}) or {})
                moving_key = None
                support_key = None
                actual_u = None
                active_t = None
                support_t = None
                for arm_key, points in arm_points.items():
                    if not isinstance(points, dict):
                        continue
                    src = self._finite_point2(points.get("source_board"))
                    dst = self._finite_point2(points.get("target_board"))
                    if src is None or dst is None:
                        continue
                    move = dst - src
                    move_n = float(np.linalg.norm(move))
                    role = str(points.get("role", "")).lower()
                    body_t = float(np.dot(src - geom["waist"], geom["body_u"]) / max(float(geom["body_len"]), 1e-6))
                    if move_n > 5.0:
                        moving_key = arm_key
                        actual_u = move / move_n
                        active_t = body_t
                    elif "support" in role or "anchor" in role or move_n <= 1.0:
                        support_key = arm_key
                        support_t = body_t
                if moving_key is None or support_key is None or actual_u is None:
                    continue
                direction_alignment = float(np.dot(actual_u, desired))
                wrinkle_axis_error_deg = self._v35_axis_error_deg(actual_u, normal)
                upper_support_ok = bool(active_t is not None and support_t is not None and support_t <= active_t - 0.04 and support_t <= 0.78)
                if (
                    direction_alignment < min_fold_direction_alignment
                    or wrinkle_axis_error_deg is None
                    or float(wrinkle_axis_error_deg) > 20.0
                    or not upper_support_ok
                ):
                    print(
                        f"[D55-V35-FOLD-DIR-REJECT] candidate={index + 1} "
                        f"desiredAlign={direction_alignment:.2f} wrinkleNormalErr={wrinkle_axis_error_deg} "
                        f"supportOK={upper_support_ok}"
                    )
                    continue
                metrics = dict(getattr(plan, "metrics", {}) or {})
                metrics.update({
                    "d55v30_internal_mode": "FOLDED_LEG_RECOVERY",
                    "d55v30_folded_side": str(geom["folded_side"]),
                    "d55v30_leg_balance": float(geom["leg_balance"]),
                    "d55v30_hem_balance": float(geom["hem_balance"]),
                    "d55v30_candidate_original_index": int(cand.get("_v35_original_index", index)),
                    "d55v30_candidate_center_board": [float(center[0]), float(center[1])],
                    "d55v30_desired_sweep_unit": [float(desired[0]), float(desired[1])],
                    "d55v30_actual_direction_alignment": direction_alignment,
                    "d55v30_upper_support_ok": upper_support_ok,
                    "d55v30_fold_boundary_excluded_count": int(len(geom["boundary_ids"])),
                    "d55v35_fold_direction_min_alignment": float(min_fold_direction_alignment),
                    "d55v35_final_wrinkle_normal_error_deg": float(wrinkle_axis_error_deg),
                    "d55v35_final_wrinkle_normal_tolerance_deg": 20.0,
                    "d55v35_direction_check": copy.deepcopy(cand.get("d55v35_direction_check", {})),
                })
                plan.metrics = metrics
                plan.arm_points[moving_key]["role"] = "folded_leg_outward_downward_sweep"
                plan.arm_points[support_key]["role"] = "folded_leg_upper_support_anchor"
                print(
                    f"[D55-V35-FOLDED-LEG-RECOVERY] side={geom['folded_side']} leg={geom['leg_balance']:.2f} "
                    f"hem={geom['hem_balance']:.2f} candidate={index + 1} active={str(moving_key).upper()} "
                    f"support={str(support_key).upper()} dirAlign={direction_alignment:.2f} "
                    f"wrinkleNormalErr={float(wrinkle_axis_error_deg):.1f}deg"
                )
                return plan
            safe_candidates = [cand for i, cand in enumerate(candidates) if i not in geom["boundary_ids"]]
            if safe_candidates:
                plan = self._v35_build_direction_checked_super(obs, heat, H, config, cfg, args, safe_candidates)
                if plan is not None:
                    metrics = dict(getattr(plan, "metrics", {}) or {})
                    metrics.update({
                        "d55v30_internal_mode": "NORMAL_WRINKLE_FOLD_BOUNDARY_FILTERED",
                        "d55v30_folded_side": str(geom["folded_side"]),
                        "d55v30_leg_balance": float(geom["leg_balance"]),
                        "d55v30_hem_balance": float(geom["hem_balance"]),
                        "d55v30_fold_boundary_excluded_count": int(len(geom["boundary_ids"])),
                        "d55v35_direction_rejected_count": int(len(direction_rejected)),
                    })
                    plan.metrics = metrics
                print(f"[D55-V35-FOLD-FALLBACK] no direction-safe folded recovery; excluded={len(geom['boundary_ids'])} boundary candidates, direction-checked normal planner used")
                return plan
            module = self.d55
            print(f"[D55-V35-FOLD-BLOCKED] folded state detected but no direction-safe recovery/non-boundary wrinkle candidate; excluded={len(geom['boundary_ids'])}")
            return module.DualWrinkleStretchPlan(False, "D55-V35 folded state: no direction-safe FOLDED_LEG_RECOVERY", metrics={
                "d55_failure_category": "NO_DIRECTION_SAFE_FOLDED_LEG_RECOVERY",
                "d55v30_internal_mode": "FOLDED_LEG_RECOVERY_BLOCKED",
                "d55v30_folded_side": str(geom["folded_side"]),
                "d55v30_leg_balance": float(geom["leg_balance"]),
                "d55v30_hem_balance": float(geom["hem_balance"]),
                "d55v30_fold_boundary_excluded_count": int(len(geom["boundary_ids"])),
                "d55v35_direction_rejected_count": int(len(direction_rejected)),
            })

        def _v40_candidate_component(self, cand, heat):
            if not isinstance(cand, dict) or heat is None:
                return None
            binary = getattr(heat, "binary", None)
            center = self._finite_point2(cand.get("center_px"))
            if binary is None or center is None:
                return None
            try:
                mask = (np.asarray(binary) > 0).astype(np.uint8)
                if mask.ndim != 2 or mask.size == 0:
                    return None
                h, w = mask.shape[:2]
                cx = int(np.clip(round(float(center[0])), 0, w - 1))
                cy = int(np.clip(round(float(center[1])), 0, h - 1))
                nlab, labels = self.cv2.connectedComponents(mask, connectivity=8)
                lid = int(labels[cy, cx])
                if lid <= 0:
                    yy_all, xx_all = np.nonzero(labels > 0)
                    if len(xx_all) == 0:
                        return None
                    dist2 = (xx_all.astype(np.float64) - float(cx)) ** 2 + (yy_all.astype(np.float64) - float(cy)) ** 2
                    jj = int(np.argmin(dist2))
                    if float(dist2[jj]) > 24.0 ** 2:
                        return None
                    lid = int(labels[int(yy_all[jj]), int(xx_all[jj])])
                if lid <= 0 or lid >= int(nlab):
                    return None
                yy, xx = np.nonzero(labels == lid)
                if len(xx) < 18:
                    return None
                return {
                    "labels": labels,
                    "label": int(lid),
                    "xx": xx.astype(np.int32),
                    "yy": yy.astype(np.int32),
                    "center_px": np.asarray([cx, cy], np.float64),
                    "shape": (h, w),
                }
            except Exception:
                return None

        def _v40_zone_segments(self, obs, cand, heat, H, original_index):
            comp = self._v40_candidate_component(cand, heat)
            if comp is None or H is None:
                return []
            xx = comp["xx"]
            yy = comp["yy"]
            center = comp["center_px"]
            h, w = comp["shape"]
            heatmap = getattr(heat, "heatmap", None)
            if heatmap is not None:
                try:
                    hm = np.asarray(heatmap, np.float32)
                    if hm.shape[:2] != (h, w):
                        hm = self.cv2.resize(hm, (w, h), interpolation=self.cv2.INTER_LINEAR)
                    values = hm[yy, xx]
                    max_heat = float(np.max(values)) if len(values) else 0.0
                except Exception:
                    hm = None
                    values = np.zeros((len(xx),), np.float32)
                    max_heat = 0.0
            else:
                hm = None
                values = np.zeros((len(xx),), np.float32)
                max_heat = 0.0

            anchors = [center.copy()]
            if len(xx) > 0:
                if max_heat > 1e-9:
                    order = np.argsort(values)[::-1]
                else:
                    dist2 = (xx.astype(np.float64) - center[0]) ** 2 + (yy.astype(np.float64) - center[1]) ** 2
                    order = np.argsort(dist2)
                for jj in order[: min(len(order), 700)]:
                    p = np.asarray([float(xx[int(jj)]), float(yy[int(jj)])], np.float64)
                    if all(float(np.linalg.norm(p - q)) >= 30.0 for q in anchors):
                        anchors.append(p)
                    if len(anchors) >= 3:
                        break

            raw_segments = []
            major_hint = float(cand.get("major_length_px", 0.0) or 0.0)
            radius = float(np.clip(0.18 * major_hint if major_hint > 0.0 else 34.0, 27.0, 46.0))
            for anchor_i, anchor in enumerate(anchors):
                keep = (xx.astype(np.float64) - anchor[0]) ** 2 + (yy.astype(np.float64) - anchor[1]) ** 2 <= radius ** 2
                lxx = xx[keep]
                lyy = yy[keep]
                if len(lxx) < 14:
                    continue
                if len(lxx) > 800:
                    step = int(math.ceil(len(lxx) / 800.0))
                    lxx = lxx[::step]
                    lyy = lyy[::step]
                pts_px = np.stack([lxx, lyy], axis=1).astype(np.float64)
                cpx = np.mean(pts_px, axis=0)
                px_centered = pts_px - cpx.reshape(1, 2)
                cov_px = px_centered.T @ px_centered / max(1, len(px_centered) - 1)
                vals_px, vecs_px = np.linalg.eigh(cov_px)
                order_px = np.argsort(vals_px)
                small_px = max(float(vals_px[order_px[0]]), 1e-6)
                large_px = max(float(vals_px[order_px[-1]]), 0.0)
                local_linearity = float(large_px / small_px)
                if local_linearity < 1.15:
                    continue
                tangent_px = np.asarray(vecs_px[:, order_px[-1]], np.float64).reshape(2)
                tpn = float(np.linalg.norm(tangent_px))
                if tpn < 1e-6:
                    continue
                tangent_px /= tpn
                proj_px = px_centered @ tangent_px
                span_px = float(np.max(proj_px) - np.min(proj_px)) if len(proj_px) else 0.0
                pts_cv = pts_px.astype(np.float32).reshape(-1, 1, 2)
                pts_board = self.cv2.perspectiveTransform(pts_cv, np.asarray(H, np.float32)).reshape(-1, 2).astype(np.float64)
                pts_board = pts_board[np.all(np.isfinite(pts_board), axis=1)]
                if len(pts_board) < 12:
                    continue
                cb = np.mean(pts_board, axis=0)
                bd_centered = pts_board - cb.reshape(1, 2)
                cov_bd = bd_centered.T @ bd_centered / max(1, len(bd_centered) - 1)
                vals_bd, vecs_bd = np.linalg.eigh(cov_bd)
                order_bd = np.argsort(vals_bd)
                tangent = np.asarray(vecs_bd[:, order_bd[-1]], np.float64).reshape(2)
                tn = float(np.linalg.norm(tangent))
                if tn < 1e-6:
                    continue
                tangent /= tn
                normal = np.asarray([-tangent[1], tangent[0]], np.float64)
                old_normal = self._candidate_board_normal(cand)
                if old_normal is not None and float(np.dot(normal, old_normal)) < 0.0:
                    normal = -normal
                    tangent = -tangent
                    tangent_px = -tangent_px
                proj_bd = bd_centered @ tangent
                span_mm = float(np.max(proj_bd) - np.min(proj_bd)) if len(proj_bd) else 0.0
                heat_strength = 0.0
                if hm is not None and max_heat > 1e-9:
                    ax = int(np.clip(round(float(anchor[0])), 0, w - 1))
                    ay = int(np.clip(round(float(anchor[1])), 0, h - 1))
                    heat_strength = float(np.clip(float(hm[ay, ax]) / max_heat, 0.0, 1.0))
                raw_segments.append({
                    "anchor_index": int(anchor_i),
                    "anchor_px": anchor.astype(np.float64),
                    "center_px": cpx.astype(np.float64),
                    "center_board": cb.astype(np.float64),
                    "tangent_px": tangent_px.astype(np.float64),
                    "tangent_board": tangent.astype(np.float64),
                    "normal_board": normal.astype(np.float64),
                    "local_linearity": float(local_linearity),
                    "span_px": float(span_px),
                    "span_mm": float(span_mm),
                    "point_count": int(len(pts_board)),
                    "radius_px": float(radius),
                    "heat_strength": float(heat_strength),
                })

            if not raw_segments:
                return []
            tangent_spread = 0.0
            for i in range(len(raw_segments)):
                for j in range(i + 1, len(raw_segments)):
                    err = self._v35_axis_error_deg(raw_segments[i]["tangent_board"], raw_segments[j]["tangent_board"])
                    if err is not None:
                        tangent_spread = max(tangent_spread, float(err))
            geom0 = cand.get("d21_geometry", {}) if isinstance(cand, dict) else {}
            original_shape = str((geom0 or {}).get("shape_class", cand.get("d21_shape_class", "IRREGULAR"))).upper()
            if original_shape == "LINE" and tangent_spread <= 14.0:
                zone_shape = "LINE"
            elif original_shape in {"LINE", "CURVE"} and tangent_spread <= 28.0:
                zone_shape = "CURVE"
            else:
                zone_shape = "ZIGZAG"

            out = []
            for seg in raw_segments:
                clone = copy.deepcopy(cand)
                clone["_v40_original_index"] = int(original_index)
                clone["center_px"] = np.asarray(seg["center_px"], np.float64)
                clone["center_board"] = np.asarray(seg["center_board"], np.float64)
                clone["major_axis_px"] = np.asarray(seg["tangent_px"], np.float64)
                clone["major_length_px"] = max(float(clone.get("major_length_px", 0.0) or 0.0), float(seg["span_px"]))
                clone["linearity"] = max(float(clone.get("linearity", 1.0) or 1.0), float(seg["local_linearity"]))
                geom = dict(clone.get("d21_geometry", {}) or {})
                geom["shape_class"] = "LINE" if zone_shape == "LINE" else "CURVE"
                geom["tangent_board"] = np.asarray(seg["tangent_board"], np.float64)
                geom["normal_board"] = np.asarray(seg["normal_board"], np.float64)
                geom["center_board"] = np.asarray(seg["center_board"], np.float64)
                geom["major_length_px"] = float(seg["span_px"])
                geom["major_length_mm"] = float(seg["span_mm"])
                geom["linearity"] = float(seg["local_linearity"])
                geom["d55v40_local_segment"] = True
                geom["d55v40_region_shape"] = str(zone_shape)
                geom["d55v40_tangent_spread_deg"] = float(tangent_spread)
                clone["d21_geometry"] = geom
                clone["d21_shape_class"] = geom["shape_class"]
                clone["d55v40_region_shape"] = str(zone_shape)
                clone["d55v40_local_segment"] = {
                    "anchor_index": int(seg["anchor_index"]),
                    "center_px": [float(seg["center_px"][0]), float(seg["center_px"][1])],
                    "center_board": [float(seg["center_board"][0]), float(seg["center_board"][1])],
                    "local_linearity": float(seg["local_linearity"]),
                    "span_px": float(seg["span_px"]),
                    "span_mm": float(seg["span_mm"]),
                    "point_count": int(seg["point_count"]),
                    "radius_px": float(seg["radius_px"]),
                    "heat_strength": float(seg["heat_strength"]),
                    "tangent_spread_deg": float(tangent_spread),
                }
                out.append(clone)
            print(
                f"[D55-V40-ZONE] candidate={int(original_index)+1} originalShape={original_shape} "
                f"regionShape={zone_shape} segments={len(out)} tangentSpread={tangent_spread:.1f}deg"
            )
            return out

        def _v40_leg_region(self, obs, center_board):
            result = {"region": "UNKNOWN", "crotch_distance_mm": None, "near_crotch": False}
            p = self._finite_point2(center_board)
            if p is None:
                return result
            pts = self._pose_board_points(obs)
            crotch = pts.get("crotch")
            left = pts.get("left_hem_center")
            right = pts.get("right_hem_center")
            if crotch is None:
                return result
            cd = float(np.linalg.norm(p - crotch))
            result["crotch_distance_mm"] = cd
            result["near_crotch"] = bool(cd < 45.0)

            def seg_dist(q, a, b):
                if a is None or b is None:
                    return float("inf")
                v = b - a
                den = float(np.dot(v, v))
                if den < 1e-6:
                    return float(np.linalg.norm(q - a))
                t = float(np.clip(np.dot(q - a, v) / den, 0.0, 1.0))
                return float(np.linalg.norm(q - (a + t * v)))

            dl = seg_dist(p, crotch, left)
            dr = seg_dist(p, crotch, right)
            if math.isfinite(dl) or math.isfinite(dr):
                result["left_leg_distance_mm"] = dl
                result["right_leg_distance_mm"] = dr
                result["region"] = "LEFT_LEG" if dl <= dr else "RIGHT_LEG"
            return result

        def _v70_press_leg_point_report(self, obs, point, expected_leg):
            p = self._finite_point2(point)
            out = {"ok": False, "available": False, "region": "UNKNOWN", "leg_t": None, "lateral_mm": None, "crotch_distance_mm": None, "mask_ok": None, "mask_ratio": None, "left_leg_distance_mm": None, "right_leg_distance_mm": None, "leg_margin_mm": None}
            if p is None or str(expected_leg) not in {"LEFT_LEG", "RIGHT_LEG"}:
                return out
            pts = self._pose_board_points(obs)
            crotch = self._finite_point2(pts.get("crotch"))
            hem_key = "left_hem_center" if str(expected_leg) == "LEFT_LEG" else "right_hem_center"
            hem = self._finite_point2(pts.get(hem_key))
            if crotch is None or hem is None:
                return out
            v = hem - crotch
            den = float(np.dot(v, v))
            if den < 1e-6:
                return out
            rel = p - crotch
            t = float(np.dot(rel, v) / den)
            nearest = crotch + np.clip(t, 0.0, 1.0) * v
            lateral = float(np.linalg.norm(p - nearest))
            base = self._v40_leg_region(obs, p)
            mask_obj = getattr(obs, "mask", None)
            mask_available = bool(mask_obj is not None and getattr(mask_obj, "mask_u8", None) is not None and getattr(self, "H_corrected", None) is not None)
            if mask_available:
                mask_ok, mask_ratio = self._v56_mask_support_at_board(
                    obs, self.H_corrected, p, radius_px=4, min_ratio=float(self._v71_press_leg_mask_ratio)
                )
            else:
                mask_ok, mask_ratio = True, None
            left_dist = base.get("left_leg_distance_mm")
            right_dist = base.get("right_leg_distance_mm")
            own_dist = left_dist if str(expected_leg) == "LEFT_LEG" else right_dist
            other_dist = right_dist if str(expected_leg) == "LEFT_LEG" else left_dist
            leg_margin = None
            if own_dist is not None and other_dist is not None and math.isfinite(float(own_dist)) and math.isfinite(float(other_dist)):
                leg_margin = float(other_dist) - float(own_dist)
            out.update({
                "available": True,
                "region": str(base.get("region", "UNKNOWN")),
                "leg_t": t,
                "lateral_mm": lateral,
                "crotch_distance_mm": base.get("crotch_distance_mm"),
                "mask_ok": bool(mask_ok),
                "mask_ratio": None if mask_ratio is None else float(mask_ratio),
                "left_leg_distance_mm": None if left_dist is None else float(left_dist),
                "right_leg_distance_mm": None if right_dist is None else float(right_dist),
                "leg_margin_mm": leg_margin,
            })
            out["ok"] = bool(
                str(out["region"]) == str(expected_leg)
                and float(self._v71_press_leg_t_min) <= t <= float(self._v71_press_leg_t_max)
                and bool(mask_ok)
            )
            return out

        def _v70_press_leg_lock_report(self, plan, obs, ridge_center=None):
            report = {"ok": True, "enforced": False, "reason": "POSE_LEG_UNAVAILABLE", "ridge_leg": "UNKNOWN", "arms": {}}
            if plan is None or not bool(getattr(plan, "ok", False)):
                report.update({"ok": False, "reason": "PLAN_UNAVAILABLE"})
                return report
            metrics = dict(getattr(plan, "metrics", {}) or {})
            center = self._finite_point2(ridge_center)
            if center is None:
                ridge_meta = metrics.get("d55v47_ridge", {}) if isinstance(metrics.get("d55v47_ridge", {}), dict) else {}
                center = self._finite_point2(ridge_meta.get("center_board"))
            if center is None:
                center = self._finite_point2(getattr(plan, "center_board", None))
            ridge_info = self._v40_leg_region(obs, center)
            ridge_leg = str(ridge_info.get("region", "UNKNOWN"))
            report["ridge_leg"] = ridge_leg
            report["ridge"] = copy.deepcopy(ridge_info)
            if ridge_leg not in {"LEFT_LEG", "RIGHT_LEG"}:
                return report
            ridge_corridor = self._v70_press_leg_point_report(obs, center, ridge_leg)
            report["ridge_corridor"] = copy.deepcopy(ridge_corridor)
            if not bool(ridge_corridor.get("available", False)):
                return report
            ridge_t = ridge_corridor.get("leg_t")
            ridge_cd = ridge_corridor.get("crotch_distance_mm")
            if ridge_t is None or float(ridge_t) < float(self._v71_press_ridge_t_min):
                report["reason"] = "RIDGE_NOT_SINGLE_LEG"
                return report
            report["enforced"] = True
            if not bool(ridge_corridor.get("ok", False)):
                report.update({"ok": False, "reason": "RIDGE_IS_NOT_IN_SELECTED_LEG_MASK"})
                return report
            if ridge_cd is not None and float(ridge_cd) < float(self._v71_press_ridge_crotch_min_mm):
                report.update({"ok": False, "reason": "RIDGE_TOO_CLOSE_TO_CROTCH"})
                return report
            arm_points = dict(getattr(plan, "arm_points", {}) or {})
            moving_count = 0
            contact_count = 0
            for arm_key in ("arm2", "arm1"):
                item = arm_points.get(arm_key, {})
                if not isinstance(item, dict):
                    continue
                src = None
                for key in ("source_board", "detected_source_board", "grip_board"):
                    src = self._finite_point2(item.get(key))
                    if src is not None:
                        break
                if src is None:
                    continue
                dst = self._finite_point2(item.get("target_board"))
                if dst is None:
                    dst = self._finite_point2(item.get("sweep_board"))
                if dst is None:
                    dst = src.copy()
                contact_count += 1
                move_mm = float(np.linalg.norm(dst - src))
                moving = bool(move_mm > 5.0)
                if moving:
                    moving_count += 1
                src_info = self._v70_press_leg_point_report(obs, src, ridge_leg)
                dst_info = self._v70_press_leg_point_report(obs, dst, ridge_leg) if moving else copy.deepcopy(src_info)
                src_cd = src_info.get("crotch_distance_mm")
                dst_cd = dst_info.get("crotch_distance_mm")
                source_ok = bool(
                    bool(src_info.get("ok", False))
                    and src_cd is not None
                    and float(src_cd) >= float(self._v70_press_source_crotch_min_mm)
                )
                target_ok = bool(
                    (not moving)
                    or (
                        bool(dst_info.get("ok", False))
                        and dst_cd is not None
                        and float(dst_cd) >= float(self._v70_press_target_crotch_min_mm)
                    )
                )
                arm_ok = bool(source_ok and target_ok)
                report["arms"][arm_key] = {
                    "ok": arm_ok,
                    "moving": moving,
                    "role": str(item.get("role", "")),
                    "move_mm": move_mm,
                    "source": [float(src[0]), float(src[1])],
                    "target": [float(dst[0]), float(dst[1])],
                    "source_leg": str(src_info.get("region", "UNKNOWN")),
                    "target_leg": str(dst_info.get("region", "UNKNOWN")),
                    "source_leg_t": src_info.get("leg_t"),
                    "target_leg_t": dst_info.get("leg_t"),
                    "source_mask_ratio": src_info.get("mask_ratio"),
                    "target_mask_ratio": dst_info.get("mask_ratio"),
                    "source_crotch_distance_mm": None if src_cd is None else float(src_cd),
                    "target_crotch_distance_mm": None if dst_cd is None else float(dst_cd),
                }
                if not arm_ok:
                    report["ok"] = False
                    report["reason"] = f"{arm_key.upper()}_CONTACT_LEAVES_SELECTED_{ridge_leg}"
            if contact_count <= 0:
                report.update({"ok": False, "reason": "NO_PRESS_CONTACT"})
            elif moving_count <= 0:
                report.update({"ok": False, "reason": "NO_MOVING_PRESS_ARM"})
            elif report["ok"]:
                report["reason"] = "OK"
            return report

        def _v40_plan_geometry_report(self, plan, cand):
            report = {
                "usable": False,
                "reason": "INVALID_PLAN",
                "two_arm_moving": False,
                "moving_arm_count": 0,
                "symmetry_score": 0.0,
                "perpendicular_quality": 0.0,
                "max_normal_error_deg": None,
                "length_balance": 0.0,
                "opposite_score": 0.0,
                "midpoint_preservation": 0.0,
                "radial_outward_score": 0.0,
            }
            if plan is None or not bool(getattr(plan, "ok", False)):
                return report
            expected = self._candidate_board_normal(cand)
            arm_points = dict(getattr(plan, "arm_points", {}) or {})
            rows = []
            for arm_key in ("arm2", "arm1"):
                points = arm_points.get(arm_key)
                if not isinstance(points, dict):
                    continue
                src = self._finite_point2(points.get("source_board"))
                dst = self._finite_point2(points.get("target_board"))
                if src is None or dst is None:
                    continue
                move = dst - src
                length = float(np.linalg.norm(move))
                rows.append({"arm": arm_key, "src": src, "dst": dst, "move": move, "length": length})
            moving = [r for r in rows if r["length"] > 5.0]
            report["moving_arm_count"] = int(len(moving))
            if not moving:
                report["reason"] = "NO_MOVING_SWEEP_VECTOR"
                return report
            normal_errors = []
            if expected is not None:
                for row in moving:
                    err = self._v35_axis_error_deg(row["move"] / max(row["length"], 1e-6), expected)
                    if err is not None:
                        normal_errors.append(float(err))
            if normal_errors:
                max_err = float(max(normal_errors))
                report["max_normal_error_deg"] = max_err
                report["perpendicular_quality"] = float(np.clip(1.0 - max_err / 45.0, 0.0, 1.0))
                if max_err > 45.0:
                    report["reason"] = f"SWEEP_NOT_LOCAL_NORMAL:{max_err:.1f}deg"
                    return report
            else:
                report["perpendicular_quality"] = 0.45

            if len(moving) >= 2:
                a, b = moving[0], moving[1]
                la, lb = a["length"], b["length"]
                length_balance = float(min(la, lb) / max(max(la, lb), 1e-6))
                ua = a["move"] / max(la, 1e-6)
                ub = b["move"] / max(lb, 1e-6)
                opposite = float(np.clip(0.5 * (1.0 - float(np.dot(ua, ub))), 0.0, 1.0))
                midpoint = 0.5 * (a["src"] + b["src"])
                target_mid = 0.5 * (a["dst"] + b["dst"])
                mid_shift = float(np.linalg.norm(target_mid - midpoint))
                midpoint_preservation = float(np.clip(1.0 - mid_shift / 45.0, 0.0, 1.0))
                radial_scores = []
                for row, u in ((a, ua), (b, ub)):
                    radial = row["src"] - midpoint
                    rn = float(np.linalg.norm(radial))
                    radial_scores.append(float(np.clip(np.dot(u, radial / rn), 0.0, 1.0)) if rn > 1e-6 else 0.0)
                radial = float(np.mean(radial_scores)) if radial_scores else 0.0
                symmetry = 0.36 * length_balance + 0.30 * opposite + 0.22 * midpoint_preservation + 0.12 * radial
                report.update({
                    "two_arm_moving": True,
                    "length_balance": length_balance,
                    "opposite_score": opposite,
                    "midpoint_preservation": midpoint_preservation,
                    "midpoint_shift_mm": mid_shift,
                    "radial_outward_score": radial,
                    "symmetry_score": float(np.clip(symmetry, 0.0, 1.0)),
                })
            else:
                report.update({
                    "two_arm_moving": False,
                    "symmetry_score": 0.34,
                    "length_balance": 0.0,
                    "opposite_score": 0.0,
                    "midpoint_preservation": 0.0,
                    "radial_outward_score": 0.0,
                })
            report["usable"] = True
            report["reason"] = "OK"
            return report

        def _v43_wrinkle_size_report(self, cand, heat, H, original_index):
            geom = dict(cand.get("d21_geometry", {}) or {}) if isinstance(cand, dict) else {}
            shape = str(geom.get("shape_class", cand.get("d21_shape_class", "IRREGULAR"))).upper() if isinstance(cand, dict) else "IRREGULAR"
            source = str(cand.get("d55v44_source", "BASE")).upper() if isinstance(cand, dict) else "BASE"
            area_px = max(0.0, float(cand.get("area_px", 0.0) or 0.0)) if isinstance(cand, dict) else 0.0
            comp = self._v40_candidate_component(cand, heat)
            if area_px <= 0.0 and comp is not None:
                area_px = float(len(comp.get("xx", [])))

            major_mm = max(0.0, float(cand.get("major_length_mm", 0.0) or 0.0)) if isinstance(cand, dict) else 0.0
            if shape == "CURVE":
                major_mm = max(major_mm, float(geom.get("arc_length_mm", 0.0) or 0.0))
            major_mm = max(major_mm, float(geom.get("major_length_mm", 0.0) or 0.0))

            component_span_mm = 0.0
            if comp is not None and H is not None:
                try:
                    xx = np.asarray(comp.get("xx", []), np.float64)
                    yy = np.asarray(comp.get("yy", []), np.float64)
                    if len(xx) >= 12:
                        step = max(1, int(math.ceil(len(xx) / 1200.0)))
                        pts = np.stack([xx[::step], yy[::step]], axis=1).astype(np.float32).reshape(-1, 1, 2)
                        board = self.cv2.perspectiveTransform(pts, np.asarray(H, np.float32)).reshape(-1, 2).astype(np.float64)
                        board = board[np.all(np.isfinite(board), axis=1)]
                        if len(board) >= 12:
                            c = np.mean(board, axis=0)
                            q = board - c.reshape(1, 2)
                            cov = q.T @ q / max(1, len(q) - 1)
                            vals, vecs = np.linalg.eigh(cov)
                            axis = np.asarray(vecs[:, int(np.argmax(vals))], np.float64)
                            an = float(np.linalg.norm(axis))
                            if an > 1e-6:
                                axis /= an
                                proj = q @ axis
                                component_span_mm = float(np.max(proj) - np.min(proj))
                except Exception:
                    component_span_mm = 0.0
            if major_mm <= 1e-6:
                major_mm = component_span_mm

            severity = float(np.clip(float(cand.get("d21_severity", 0.0) or 0.0), 0.0, 1.0)) if isinstance(cand, dict) else 0.0
            tier = float(max(0.0, float(cand.get("d20_priority_tier", 0.0) or 0.0))) if isinstance(cand, dict) else 0.0
            max_score = float(max(0.0, float(cand.get("max_score", cand.get("priority_score", 0.0)) or 0.0))) if isinstance(cand, dict) else 0.0
            heat_quality = float(np.clip(max_score / 255.0 if max_score > 1.5 else max_score, 0.0, 1.0))

            length_quality = float(np.clip(major_mm / 180.0, 0.0, 1.0))
            area_quality = float(np.clip(area_px / 1800.0, 0.0, 1.0))
            tier_quality = float(np.clip(tier / 3.0, 0.0, 1.0))
            shape_bonus = 6.0 if shape == "LINE" else 3.0 if shape == "CURVE" else -8.0 if shape == "IRREGULAR" else 0.0
            source_bonus = 4.0 if source == "BASE" and shape in {"LINE", "CURVE"} else 0.0
            if source == "ADAPTIVE_LIGHT" and shape == "IRREGULAR":
                source_bonus -= 5.0
            size_score = (
                68.0 * length_quality
                + 20.0 * severity
                + 6.0 * heat_quality
                + 4.0 * area_quality
                + 2.0 * tier_quality
                + shape_bonus
                + source_bonus
            )
            return {
                "original_index": int(original_index),
                "shape": shape,
                "source": source,
                "major_length_mm": float(major_mm),
                "component_span_mm": float(component_span_mm),
                "area_px": float(area_px),
                "severity": float(severity),
                "heat_quality": float(heat_quality),
                "priority_tier": float(tier),
                "shape_bonus": float(shape_bonus),
                "source_bonus": float(source_bonus),
                "size_score": float(size_score),
            }

        def _v46_support_proximity_report(self, plan, cand):
            result = {"available": False, "distance_mm": None, "quality": 0.0, "support_arm": None}
            if plan is None or not isinstance(cand, dict):
                return result
            seg = dict(cand.get("d55v40_local_segment", {}) or {})
            center = self._finite_point2(seg.get("center_board"))
            geom = dict(cand.get("d21_geometry", {}) or {})
            tangent = self._finite_point2(geom.get("tangent_board"))
            if center is None or tangent is None:
                return result
            tn = float(np.linalg.norm(tangent))
            if tn < 1e-6:
                return result
            tangent = tangent / tn
            half_span = max(20.0, 0.5 * float(seg.get("span_mm", 0.0) or 0.0))
            arm_points = dict(getattr(plan, "arm_points", {}) or {})
            best = None
            for arm_key, points in arm_points.items():
                if not isinstance(points, dict):
                    continue
                src = self._finite_point2(points.get("source_board"))
                dst = self._finite_point2(points.get("target_board"))
                if src is None or dst is None:
                    continue
                role = str(points.get("role", "")).lower()
                move_mm = float(np.linalg.norm(dst - src))
                if move_mm > 5.0 and "support" not in role and "anchor" not in role:
                    continue
                along = float(np.dot(src - center, tangent))
                along = float(np.clip(along, -half_span, half_span))
                nearest = center + along * tangent
                dist = float(np.linalg.norm(src - nearest))
                row = (dist, str(arm_key), role)
                if best is None or row[0] < best[0]:
                    best = row
            if best is None:
                return result
            dist, arm_key, role = best
            quality = float(np.clip(1.0 - dist / 240.0, 0.0, 1.0))
            far_penalty = float(np.clip((dist - 170.0) / 150.0, 0.0, 1.0))
            result.update({
                "available": True,
                "distance_mm": float(dist),
                "quality": float(quality),
                "far_penalty": float(far_penalty),
                "support_arm": str(arm_key),
                "support_role": str(role),
            })
            return result

        def _v43_midvector_shape_geometry(self, obs):
            result = {
                "available": False,
                "reason": "POSE_GEOMETRY_UNAVAILABLE",
                "symmetry_before": None,
                "deficit_side": "NONE",
            }
            pts = self._pose_board_points(obs)
            waist = pts.get("waist_center")
            if waist is None and pts.get("waist_img_left") is not None and pts.get("waist_img_right") is not None:
                waist = 0.5 * (pts["waist_img_left"] + pts["waist_img_right"])
            crotch = pts.get("crotch")
            lower = crotch if crotch is not None else pts.get("lower_center")
            left_hem = pts.get("left_hem_center")
            right_hem = pts.get("right_hem_center")
            if waist is None or lower is None or left_hem is None or right_hem is None:
                return result
            axis = np.asarray(lower - waist, np.float64)
            an = float(np.linalg.norm(axis))
            if not math.isfinite(an) or an < 20.0:
                result["reason"] = "MIDVECTOR_TOO_SHORT"
                return result
            axis /= an
            lateral = np.asarray([-axis[1], axis[0]], np.float64)
            left_lat = float(np.dot(left_hem - waist, lateral))
            right_lat = float(np.dot(right_hem - waist, lateral))
            if left_lat > right_lat:
                lateral = -lateral
                left_lat = -left_lat
                right_lat = -right_lat
            if not (left_lat < -4.0 and right_lat > 4.0):
                result.update({
                    "reason": "HEMS_NOT_ON_OPPOSITE_MIDVECTOR_SIDES",
                    "left_lateral_mm": float(left_lat),
                    "right_lateral_mm": float(right_lat),
                })
                return result
            left_width = abs(left_lat)
            right_width = abs(right_lat)
            symmetry = float(min(left_width, right_width) / max(max(left_width, right_width), 1e-6))
            if left_width + 8.0 < right_width:
                deficit = "LEFT"
            elif right_width + 8.0 < left_width:
                deficit = "RIGHT"
            else:
                deficit = "BALANCED"
            result.update({
                "available": True,
                "reason": "OK",
                "origin_board": [float(waist[0]), float(waist[1])],
                "axis_board": [float(axis[0]), float(axis[1])],
                "lateral_board": [float(lateral[0]), float(lateral[1])],
                "left_lateral_mm": float(left_lat),
                "right_lateral_mm": float(right_lat),
                "left_width_mm": float(left_width),
                "right_width_mm": float(right_width),
                "symmetry_before": float(symmetry),
                "deficit_side": deficit,
            })
            return result

        def _v43_shape_symmetry_report(self, plan, midgeom):
            report = {
                "available": False,
                "reason": "MIDVECTOR_UNAVAILABLE",
                "symmetry_before": midgeom.get("symmetry_before") if isinstance(midgeom, dict) else None,
                "symmetry_after_proxy": None,
                "symmetry_gain": 0.0,
                "direction_score": 0.0,
                "robot_motion_symmetry_used": False,
            }
            if not isinstance(midgeom, dict) or not bool(midgeom.get("available", False)):
                return report
            origin = self._finite_point2(midgeom.get("origin_board"))
            lateral = self._finite_point2(midgeom.get("lateral_board"))
            if origin is None or lateral is None:
                report["reason"] = "MIDVECTOR_FIELDS_MISSING"
                return report
            ln = float(np.linalg.norm(lateral))
            if ln < 1e-6:
                report["reason"] = "MIDVECTOR_LATERAL_INVALID"
                return report
            lateral = lateral / ln
            left_width = float(midgeom.get("left_width_mm", 0.0) or 0.0)
            right_width = float(midgeom.get("right_width_mm", 0.0) or 0.0)
            before = float(midgeom.get("symmetry_before", 0.0) or 0.0)
            rows = []
            for arm_key, arm in (getattr(plan, "arm_points", {}) or {}).items():
                if not isinstance(arm, dict):
                    continue
                src = self._finite_point2(arm.get("source_board"))
                dst = self._finite_point2(arm.get("target_board"))
                if src is None or dst is None:
                    continue
                move = dst - src
                if float(np.linalg.norm(move)) < 2.0:
                    continue
                src_lat = float(np.dot(src - origin, lateral))
                dst_lat = float(np.dot(dst - origin, lateral))
                if abs(src_lat) < 6.0:
                    continue
                side = "LEFT" if src_lat < 0.0 else "RIGHT"
                outward_delta = float(abs(dst_lat) - abs(src_lat))
                rows.append({
                    "arm": str(arm_key),
                    "side": side,
                    "source_lateral_mm": float(src_lat),
                    "target_lateral_mm": float(dst_lat),
                    "outward_delta_mm": float(outward_delta),
                })
                if side == "LEFT":
                    left_width = max(1.0, left_width + 0.35 * outward_delta)
                else:
                    right_width = max(1.0, right_width + 0.35 * outward_delta)
            if not rows:
                report["reason"] = "NO_LATERAL_SWEEP_COMPONENT"
                return report
            after = float(min(left_width, right_width) / max(max(left_width, right_width), 1e-6))
            gain = float(after - before)
            direction_score = float(np.clip(gain / 0.12, -1.0, 1.0))
            report.update({
                "available": True,
                "reason": "OK",
                "symmetry_before": float(before),
                "symmetry_after_proxy": float(after),
                "symmetry_gain": float(gain),
                "direction_score": float(direction_score),
                "predicted_left_width_mm": float(left_width),
                "predicted_right_width_mm": float(right_width),
                "deficit_side": str(midgeom.get("deficit_side", "NONE")),
                "moving_rows": rows,
            })
            return report

        def _v44_adaptive_luminance_frame(self, obs):
            frame = getattr(obs, "frame", None)
            mask_obj = getattr(obs, "mask", None)
            mask_u8 = getattr(mask_obj, "mask_u8", None) if mask_obj is not None else None
            if frame is None or mask_u8 is None:
                return None, {"available": False, "reason": "FRAME_OR_MASK_MISSING"}
            try:
                image = np.asarray(frame).copy()
                mask = np.asarray(mask_u8)
                if mask.shape[:2] != image.shape[:2]:
                    mask = self.cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=self.cv2.INTER_NEAREST)
                valid = mask > 0
                if int(np.count_nonzero(valid)) < 500:
                    return None, {"available": False, "reason": "MASK_TOO_SMALL"}
                lab = self.cv2.cvtColor(image, self.cv2.COLOR_BGR2LAB)
                lch = lab[:, :, 0].astype(np.float32)
                vals = lch[valid]
                p05 = float(np.percentile(vals, 5.0))
                p50 = float(np.percentile(vals, 50.0))
                p95 = float(np.percentile(vals, 95.0))
                span = max(1.0, p95 - p05)
                needs = bool(p50 >= 135.0 or span <= 72.0)
                if not needs:
                    return None, {
                        "available": False,
                        "reason": "NORMAL_CONTRAST",
                        "median_l": p50,
                        "p05_l": p05,
                        "p95_l": p95,
                        "span_l": span,
                    }
                stretched = np.clip((lch - p05) * (255.0 / span), 0.0, 255.0).astype(np.uint8)
                clahe = self.cv2.createCLAHE(clipLimit=2.8, tileGridSize=(8, 8))
                local = clahe.apply(stretched)
                bright_term = float(np.clip((p50 - 125.0) / 80.0, 0.0, 1.0))
                flat_term = float(np.clip((72.0 - span) / 45.0, 0.0, 1.0))
                blend = float(np.clip(0.48 + 0.18 * bright_term + 0.18 * flat_term, 0.48, 0.78))
                out_l = lch.astype(np.uint8)
                mixed = np.clip((1.0 - blend) * lch + blend * local.astype(np.float32), 0.0, 255.0).astype(np.uint8)
                out_l[valid] = mixed[valid]
                out_lab = lab.copy()
                out_lab[:, :, 0] = out_l
                out = self.cv2.cvtColor(out_lab, self.cv2.COLOR_LAB2BGR)
                return out, {
                    "available": True,
                    "reason": "BRIGHT_OR_LOW_CONTRAST",
                    "median_l": p50,
                    "p05_l": p05,
                    "p95_l": p95,
                    "span_l": span,
                    "blend": blend,
                }
            except Exception as exc:
                return None, {"available": False, "reason": f"ERROR:{type(exc).__name__}:{exc}"}

        def _v44_candidate_length_hint(self, cand):
            if not isinstance(cand, dict):
                return 0.0
            geom = dict(cand.get("d21_geometry", {}) or {})
            vals = [
                float(cand.get("major_length_mm", 0.0) or 0.0),
                float(geom.get("major_length_mm", 0.0) or 0.0),
                float(geom.get("arc_length_mm", 0.0) or 0.0),
                1.2 * float(cand.get("major_length_px", 0.0) or 0.0),
            ]
            return float(max(vals))

        def _v44_merge_candidates(self, base_candidates, adaptive_candidates):
            rows = []
            for source, items in (("BASE", base_candidates), ("ADAPTIVE_LIGHT", adaptive_candidates)):
                for cand in list(items or []):
                    if not isinstance(cand, dict):
                        continue
                    item = copy.deepcopy(cand)
                    item["d55v44_source"] = source
                    rows.append(item)
            rows.sort(
                key=lambda c: (
                    self._v44_candidate_length_hint(c),
                    float(c.get("d21_severity", 0.0) or 0.0),
                    float(c.get("area_px", 0.0) or 0.0),
                    float(c.get("priority_score", 0.0) or 0.0),
                ),
                reverse=True,
            )
            kept = []
            for cand in rows:
                center = self._finite_point2(cand.get("center_px"))
                if center is None:
                    continue
                axis = self._finite_point2(cand.get("major_axis_px"))
                duplicate = False
                replace_at = None
                for j, old in enumerate(kept):
                    old_center = self._finite_point2(old.get("center_px"))
                    if old_center is None or float(np.linalg.norm(center - old_center)) > 26.0:
                        continue
                    old_axis = self._finite_point2(old.get("major_axis_px"))
                    axis_close = True
                    if axis is not None and old_axis is not None:
                        an = float(np.linalg.norm(axis))
                        bn = float(np.linalg.norm(old_axis))
                        if an > 1e-6 and bn > 1e-6:
                            dot = float(np.clip(abs(float(np.dot(axis / an, old_axis / bn))), 0.0, 1.0))
                            axis_close = bool(math.degrees(math.acos(dot)) <= 28.0)
                    if not axis_close:
                        continue
                    duplicate = True
                    if self._v44_candidate_length_hint(cand) > self._v44_candidate_length_hint(old) + 4.0:
                        replace_at = j
                    break
                if replace_at is not None:
                    kept[replace_at] = cand
                elif not duplicate:
                    kept.append(cand)
                if len(kept) >= 14:
                    break
            return kept

        def _v44_enhance_heat_for_light_garment(self, obs, heat, H, args):
            if heat is None:
                return heat, {"available": False, "reason": "BASE_HEAT_MISSING"}
            adapted, light_report = self._v44_adaptive_luminance_frame(obs)
            if adapted is None:
                print(
                    f"[D55-V44-LIGHT] adaptive=False reason={light_report.get('reason')} "
                    f"medianL={float(light_report.get('median_l',0.0)):.1f} spanL={float(light_report.get('span_l',0.0)):.1f}"
                )
                return heat, light_report
            builder = getattr(self.d55, "build_d21_wrinkle_heatmap", None)
            if not callable(builder):
                light_report["available"] = False
                light_report["reason"] = "D21_BUILDER_UNAVAILABLE"
                return heat, light_report
            adaptive_args = copy.copy(args)
            adaptive_args.wrinkle_heatmap_clahe_clip = max(2.8, float(getattr(args, "wrinkle_heatmap_clahe_clip", 2.0)))
            adaptive_args.wrinkle_heatmap_clahe_tile = 8
            adaptive_args.wrinkle_heatmap_blur_ksize = 71
            adaptive_args.wrinkle_heatmap_erode_px = min(8, max(0, int(getattr(args, "wrinkle_heatmap_erode_px", 12))))
            adaptive_args.wrinkle_heatmap_percentile = min(86.0, float(getattr(args, "wrinkle_heatmap_percentile", 93.0)))
            adaptive_args.wrinkle_heatmap_min_blob_area = min(28.0, float(getattr(args, "wrinkle_heatmap_min_blob_area", 45.0)))
            adaptive_args.wrinkle_heatmap_max_candidates = max(14, int(getattr(args, "wrinkle_heatmap_max_candidates", 10)))
            adaptive_args.wrinkle_heatmap_gabor = True
            adaptive_args.wrinkle_heatmap_gabor_ksize = 31
            adaptive_args.wrinkle_heatmap_gabor_lambda = 14.0
            adaptive_args.wrinkle_heatmap_gabor_sigma = 6.0
            adaptive_args.wrinkle_heatmap_score_min = min(26.0, float(getattr(args, "wrinkle_heatmap_score_min", 35.0)))
            try:
                adaptive = builder(adapted, obs, H, adaptive_args)
            except Exception as exc:
                light_report["available"] = False
                light_report["reason"] = f"ADAPTIVE_D21_ERROR:{type(exc).__name__}:{exc}"
                print(f"[D55-V44-LIGHT-ERROR] {type(exc).__name__}: {exc}")
                return heat, light_report
            if adaptive is None:
                light_report["available"] = False
                light_report["reason"] = "ADAPTIVE_D21_NONE"
                return heat, light_report
            try:
                base_map = np.asarray(getattr(heat, "heatmap", None), np.uint8)
                ada_map = np.asarray(getattr(adaptive, "heatmap", None), np.uint8)
                if base_map.shape != ada_map.shape:
                    ada_map = self.cv2.resize(ada_map, (base_map.shape[1], base_map.shape[0]), interpolation=self.cv2.INTER_LINEAR)
                merged_map = np.maximum(base_map, ada_map)
                base_bin = np.asarray(getattr(heat, "binary", np.zeros_like(base_map)), np.uint8)
                ada_bin = np.asarray(getattr(adaptive, "binary", np.zeros_like(base_map)), np.uint8)
                if base_bin.shape != ada_bin.shape:
                    ada_bin = self.cv2.resize(ada_bin, (base_bin.shape[1], base_bin.shape[0]), interpolation=self.cv2.INTER_NEAREST)
                merged_bin = self.cv2.bitwise_or(base_bin, ada_bin)
                merged_bin = self.cv2.morphologyEx(merged_bin, self.cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8), iterations=1)
                base_inner = np.asarray(getattr(heat, "inner_mask", np.zeros_like(base_map)), np.uint8)
                ada_inner = np.asarray(getattr(adaptive, "inner_mask", np.zeros_like(base_map)), np.uint8)
                if base_inner.shape != ada_inner.shape:
                    ada_inner = self.cv2.resize(ada_inner, (base_inner.shape[1], base_inner.shape[0]), interpolation=self.cv2.INTER_NEAREST)
                merged_inner = self.cv2.bitwise_or(base_inner, ada_inner)
                merged_map[merged_inner == 0] = 0
                merged_bin[merged_inner == 0] = 0
                base_candidates = list(getattr(heat, "candidates", []) or [])
                adaptive_candidates = list(getattr(adaptive, "candidates", []) or [])
                merged_candidates = self._v44_merge_candidates(base_candidates, adaptive_candidates)
                heat.heatmap = merged_map
                heat.binary = merged_bin
                heat.inner_mask = merged_inner
                heat.candidates = merged_candidates
                heat.threshold = float(min(float(getattr(heat, "threshold", 255.0)), float(getattr(adaptive, "threshold", 255.0))))
                denom = max(1, int(self.cv2.countNonZero(merged_inner)))
                heat.wrinkle_ratio = float(self.cv2.countNonZero(merged_bin)) / float(denom)
                heat.status = "OK-V44-ADAPTIVE"
                heat.summary = (
                    f"V44 adaptive-light base={len(base_candidates)} adaptive={len(adaptive_candidates)} "
                    f"merged={len(merged_candidates)} ratio={float(heat.wrinkle_ratio):.3f}"
                )
                light_report.update({
                    "base_candidates": int(len(base_candidates)),
                    "adaptive_candidates": int(len(adaptive_candidates)),
                    "merged_candidates": int(len(merged_candidates)),
                    "adaptive_threshold": float(getattr(adaptive, "threshold", 0.0)),
                    "merged_ratio": float(heat.wrinkle_ratio),
                })
                print(
                    f"[D55-V44-LIGHT] adaptive=True medianL={float(light_report.get('median_l',0.0)):.1f} "
                    f"spanL={float(light_report.get('span_l',0.0)):.1f} blend={float(light_report.get('blend',0.0)):.2f} "
                    f"base={len(base_candidates)} adaptive={len(adaptive_candidates)} merged={len(merged_candidates)} "
                    f"thr={float(getattr(adaptive,'threshold',0.0)):.1f}"
                )
                return heat, light_report
            except Exception as exc:
                light_report["available"] = False
                light_report["reason"] = f"MERGE_ERROR:{type(exc).__name__}:{exc}"
                print(f"[D55-V44-MERGE-ERROR] {type(exc).__name__}: {exc}")
                return heat, light_report

        def _v47_board_rect(self, config, cfg=None):
            try:
                aruco = config.get("aruco", {}) if isinstance(config, dict) else {}
                marker = aruco.get("marker_board_mm", {}) if isinstance(aruco, dict) else {}
                pts = np.asarray(list(marker.values()), np.float64).reshape(-1, 2)
                pts = pts[np.all(np.isfinite(pts), axis=1)]
                if len(pts) >= 4:
                    return (
                        float(np.min(pts[:, 0])), float(np.max(pts[:, 0])),
                        float(np.min(pts[:, 1])), float(np.max(pts[:, 1])),
                    )
            except Exception:
                pass
            try:
                fn = getattr(self.d55, "board_bounds_from_config", None)
                if callable(fn):
                    vals = fn(config)
                    if vals is not None and len(vals) == 4:
                        return tuple(float(v) for v in vals)
            except Exception:
                pass
            if cfg is not None:
                return (float(cfg.board_x_min), float(cfg.board_x_max),
                        float(cfg.board_y_min), float(cfg.board_y_max))
            return (0.0, 494.0, -496.0, 5.0)

        def _v47_ridge_frame(self, cand, heat, H):
            comp = self._v40_candidate_component(cand, heat)
            if comp is None or H is None:
                return None
            try:
                xx = np.asarray(comp.get("xx", []), np.float64)
                yy = np.asarray(comp.get("yy", []), np.float64)
                if len(xx) < 18:
                    return None
                step = max(1, int(math.ceil(len(xx) / 1400.0)))
                pts_px = np.stack([xx[::step], yy[::step]], axis=1).astype(np.float32)
                pts_board = self.cv2.perspectiveTransform(
                    pts_px.reshape(-1, 1, 2), np.asarray(H, np.float32)
                ).reshape(-1, 2).astype(np.float64)
                pts_board = pts_board[np.all(np.isfinite(pts_board), axis=1)]
                if len(pts_board) < 12:
                    return None
                center = self._finite_point2(cand.get("center_board"))
                if center is None:
                    cpx = self._finite_point2(cand.get("center_px"))
                    if cpx is not None:
                        try:
                            center = self._finite_point2(self.d55.pixel_to_board(H, float(cpx[0]), float(cpx[1])))
                        except Exception:
                            center = None
                if center is None:
                    center = np.mean(pts_board, axis=0)
                q = pts_board - np.mean(pts_board, axis=0).reshape(1, 2)
                cov = q.T @ q / max(1, len(q) - 1)
                vals, vecs = np.linalg.eigh(cov)
                pca_tangent = np.asarray(vecs[:, int(np.argmax(vals))], np.float64).reshape(2)
                pn = float(np.linalg.norm(pca_tangent))
                if pn < 1e-6:
                    return None
                pca_tangent /= pn
                geom = dict(cand.get("d21_geometry", {}) or {}) if isinstance(cand, dict) else {}
                hint = self._finite_point2(geom.get("tangent_board"))
                if hint is not None and float(np.linalg.norm(hint)) > 1e-6:
                    hint = hint / float(np.linalg.norm(hint))
                    if float(np.dot(hint, pca_tangent)) < 0.0:
                        hint = -hint
                    tangent = hint
                else:
                    tangent = pca_tangent
                normal = np.asarray([-tangent[1], tangent[0]], np.float64)
                old_normal = self._candidate_board_normal(cand)
                if old_normal is not None and float(np.dot(normal, old_normal)) < 0.0:
                    normal = -normal
                proj = (pts_board - center.reshape(1, 2)) @ tangent
                if len(proj) >= 20:
                    lo, hi = np.percentile(proj, [5.0, 95.0])
                else:
                    lo, hi = float(np.min(proj)), float(np.max(proj))
                span = max(25.0, float(hi - lo))
                return {
                    "center_board": center.astype(np.float64),
                    "tangent_board": tangent.astype(np.float64),
                    "normal_board": normal.astype(np.float64),
                    "span_mm": float(span),
                    "component_points": int(len(pts_board)),
                }
            except Exception as exc:
                print(f"[D55-V47-RIDGE-ERROR] {type(exc).__name__}: {exc}")
                return None

        def _v47_region_source_candidates(self, obs, cand, heat, H):
            ridge = self._v47_ridge_frame(cand, heat, H)
            mask_obj = getattr(obs, "mask", None)
            mask_u8 = getattr(mask_obj, "mask_u8", None) if mask_obj is not None else None
            if ridge is None or mask_u8 is None or H is None:
                return None
            try:
                mask = (np.asarray(mask_u8) > 0).astype(np.uint8)
                if mask.ndim != 2 or int(np.count_nonzero(mask)) < 600:
                    return None
                dist = self.cv2.distanceTransform(mask, self.cv2.DIST_L2, 5)
                stride = 4
                sy, sx = np.nonzero(mask[::stride, ::stride] > 0)
                if len(sx) < 60:
                    return None
                px = (sx.astype(np.float64) * stride + 0.5 * stride)
                py = (sy.astype(np.float64) * stride + 0.5 * stride)
                px = np.clip(px, 0, mask.shape[1] - 1)
                py = np.clip(py, 0, mask.shape[0] - 1)
                xi = np.asarray(np.rint(px), np.int32)
                yi = np.asarray(np.rint(py), np.int32)
                interior = dist[yi, xi].astype(np.float64)
                keep = interior >= 7.0
                if int(np.count_nonzero(keep)) < 20:
                    keep = interior >= 4.5
                px, py, interior = px[keep], py[keep], interior[keep]
                if len(px) < 16:
                    return None
                pts_px = np.stack([px, py], axis=1).astype(np.float32)
                board = self.cv2.perspectiveTransform(
                    pts_px.reshape(-1, 1, 2), np.asarray(H, np.float32)
                ).reshape(-1, 2).astype(np.float64)
                finite = np.all(np.isfinite(board), axis=1)
                board, pts_px, interior = board[finite], pts_px[finite], interior[finite]
                if len(board) < 16:
                    return None
                center = ridge["center_board"]
                tangent = ridge["tangent_board"]
                normal = ridge["normal_board"]
                ridge_leg_report = self._v40_leg_region(obs, center)
                ridge_leg = str(ridge_leg_report.get("region", "UNKNOWN"))
                if ridge_leg not in {"LEFT_LEG", "RIGHT_LEG"}:
                    print("[D55-V70-LEG-BLOCK] selected wrinkle has no reliable Pose leg assignment")
                    return None
                rel = board - center.reshape(1, 2)
                along = rel @ tangent
                signed = rel @ normal
                abs_normal = np.abs(signed)
                longitudinal_window = float(np.clip(0.45 * ridge["span_mm"] + 55.0, 80.0, 145.0))
                valid = (
                    (np.abs(along) <= longitudinal_window)
                    & (abs_normal >= float(self._v70_press_source_normal_min_mm))
                    & (abs_normal <= float(self._v70_press_source_normal_max_mm))
                )
                board, pts_px, interior, along, signed, abs_normal = (
                    board[valid], pts_px[valid], interior[valid], along[valid], signed[valid], abs_normal[valid]
                )
                if len(board) < 12:
                    return None
                interior_q = np.clip(interior / 28.0, 0.0, 1.0)
                normal_span = max(1.0, float(self._v70_press_source_normal_max_mm) - float(self._v70_press_source_normal_min_mm))
                normal_q = np.clip(1.0 - np.abs(abs_normal - float(self._v70_press_source_normal_preferred_mm)) / normal_span, 0.0, 1.0)
                along_q = np.clip(1.0 - np.abs(along) / max(longitudinal_window, 1.0), 0.0, 1.0)
                score = 0.56 * interior_q + 0.29 * normal_q + 0.15 * along_q
                sides = {}
                for sign, name in ((-1.0, "negative"), (1.0, "positive")):
                    ids = np.nonzero(signed * sign > 0.0)[0]
                    if len(ids) == 0:
                        sides[name] = []
                        continue
                    order = ids[np.argsort(score[ids])[::-1]]
                    chosen = []
                    deferred = []
                    for idx in order[: min(len(order), 300)]:
                        p = board[int(idx)]
                        leg_info = self._v70_press_leg_point_report(obs, p, ridge_leg)
                        if not bool(leg_info.get("ok", False)):
                            continue
                        crotch_dist = leg_info.get("crotch_distance_mm")
                        if crotch_dist is None or float(crotch_dist) < float(self._v70_press_source_crotch_min_mm):
                            continue
                        margin = leg_info.get("leg_margin_mm")
                        row = (int(idx), p, leg_info, float(crotch_dist))
                        if margin is not None and float(margin) < float(self._v72_press_inner_margin_prefer_mm):
                            deferred.append(row)
                            continue
                        if any(float(np.linalg.norm(p - old["source_board"])) < 42.0 for old in chosen):
                            continue
                        chosen.append({
                            "source_board": p.astype(np.float64),
                            "source_px": pts_px[int(idx)].astype(np.float64),
                            "signed_normal_mm": float(signed[int(idx)]),
                            "along_mm": float(along[int(idx)]),
                            "interior_depth_px": float(interior[int(idx)]),
                            "region_score": float(score[int(idx)]),
                            "side_sign": float(sign),
                            "v70_leg_region": str(ridge_leg),
                            "v70_leg_t": float(leg_info.get("leg_t")),
                            "v70_crotch_distance_mm": float(crotch_dist),
                            "v72_leg_margin_mm": leg_info.get("leg_margin_mm"),
                        })
                        if len(chosen) >= 5:
                            break
                    if len(chosen) < 5:
                        for idx, p, leg_info, crotch_dist in deferred:
                            if any(float(np.linalg.norm(p - old["source_board"])) < 42.0 for old in chosen):
                                continue
                            chosen.append({
                                "source_board": p.astype(np.float64),
                                "source_px": pts_px[int(idx)].astype(np.float64),
                                "signed_normal_mm": float(signed[int(idx)]),
                                "along_mm": float(along[int(idx)]),
                                "interior_depth_px": float(interior[int(idx)]),
                                "region_score": float(score[int(idx)]) - 0.12,
                                "side_sign": float(sign),
                                "v70_leg_region": str(ridge_leg),
                                "v70_leg_t": float(leg_info.get("leg_t")),
                                "v70_crotch_distance_mm": float(crotch_dist),
                                "v72_leg_margin_mm": leg_info.get("leg_margin_mm"),
                                "v72_inner_side_fallback": True,
                            })
                            if len(chosen) >= 5:
                                break
                    sides[name] = chosen
                print(
                    f"[D55-V70-LEG-REGION] leg={ridge_leg} span={ridge['span_mm']:.1f}mm longWin={longitudinal_window:.1f}mm "
                    f"normal={float(self._v70_press_source_normal_min_mm):.0f}..{float(self._v70_press_source_normal_max_mm):.0f}mm "
                    f"neg={len(sides.get('negative', []))} pos={len(sides.get('positive', []))}"
                )
                return {"ridge": ridge, "sides": sides, "longitudinal_window_mm": longitudinal_window, "ridge_leg": ridge_leg}
            except Exception as exc:
                print(f"[D55-V47-REGION-ERROR] {type(exc).__name__}: {exc}")
                return None

        def _v47_ray_room_to_board(self, source, direction, bounds):
            src = np.asarray(source, np.float64).reshape(2)
            vec = np.asarray(direction, np.float64).reshape(2)
            n = float(np.linalg.norm(vec))
            if n < 1e-6:
                return 0.0
            vec /= n
            xmin, xmax, ymin, ymax = [float(v) for v in bounds]
            rooms = []
            if vec[0] > 1e-8:
                rooms.append((xmax - src[0]) / vec[0])
            elif vec[0] < -1e-8:
                rooms.append((xmin - src[0]) / vec[0])
            if vec[1] > 1e-8:
                rooms.append((ymax - src[1]) / vec[1])
            elif vec[1] < -1e-8:
                rooms.append((ymin - src[1]) / vec[1])
            rooms = [float(v) for v in rooms if math.isfinite(float(v)) and float(v) >= 0.0]
            return min(rooms) if rooms else 0.0

        def _v47_sweep_distance(self, source, direction, midgeom, bounds):
            requested = 90.0
            if isinstance(midgeom, dict) and bool(midgeom.get("available", False)):
                origin = self._finite_point2(midgeom.get("origin_board"))
                lateral = self._finite_point2(midgeom.get("lateral_board"))
                if origin is not None and lateral is not None and float(np.linalg.norm(lateral)) > 1e-6:
                    lateral = lateral / float(np.linalg.norm(lateral))
                    side = "LEFT" if float(np.dot(np.asarray(source) - origin, lateral)) < 0.0 else "RIGHT"
                    deficit = str(midgeom.get("deficit_side", "BALANCED"))
                    if side == deficit:
                        requested += 15.0
                    elif deficit in {"LEFT", "RIGHT"}:
                        requested -= 5.0
            room = self._v47_ray_room_to_board(source, direction, bounds)
            maximum = min(125.0, max(0.0, room - 1.0))
            actual = min(requested, maximum)
            return float(actual), {"requested_mm": float(requested), "board_room_mm": float(room), "max_mm": float(maximum)}

        def _v47_press_z_for_arm(self, args, arm_key):
            specific = getattr(args, f"{arm_key}_press_z", None)
            if specific is not None:
                return float(specific)
            common = getattr(args, "b34_press_z", None)
            if common is None:
                common = getattr(args, "press_z", -104.0)
            return float(common)

        def _v47_build_two_region_plan(self, obs, heat, H, config, cfg, args, cand, original_index, rank, size, midgeom):
            if self._v67_prepare_cancel_requested():
                return None
            region = self._v47_region_source_candidates(obs, cand, heat, H)
            if region is None:
                return None
            negative = list(region["sides"].get("negative", []) or [])
            positive = list(region["sides"].get("positive", []) or [])
            if not negative or not positive:
                print(f"[D55-V47-PAIR-REJECT] rank={rank} candidate={original_index+1} reason=ONE_RIDGE_SIDE_EMPTY")
                return None
            ridge = region["ridge"]
            normal = np.asarray(ridge["normal_board"], np.float64)
            tangent = np.asarray(ridge["tangent_board"], np.float64)
            bounds = self._v47_board_rect(config, cfg)
            compat_fn = getattr(self.d55, "_d55v2_dual_compatibility_report", None)
            pose_fn = getattr(self.d55, "_roarm_pose_for_board", None)
            plan_cls = getattr(self.d55, "DualWrinkleStretchPlan", None)
            if not callable(compat_fn) or not callable(pose_fn) or plan_cls is None:
                print("[D55-V47-PAIR-REJECT] required D55 plan API unavailable")
                return None
            trials = []
            for neg in negative[:5]:
                if self._v67_prepare_cancel_requested():
                    return None
                for pos in positive[:5]:
                    if self._v67_prepare_cancel_requested():
                        return None
                    src_neg = np.asarray(neg["source_board"], np.float64)
                    src_pos = np.asarray(pos["source_board"], np.float64)
                    start_sep = float(np.linalg.norm(src_pos - src_neg))
                    if start_sep < 82.0:
                        continue
                    neg_along = float(neg.get("along_mm", 0.0))
                    pos_along = float(pos.get("along_mm", 0.0))
                    along_delta = abs(neg_along - pos_along)
                    along_mid = 0.5 * (neg_along + pos_along)
                    max_along_delta = float(self._v88_press_pair_max_along_delta_mm)
                    max_mid = float(self._v88_press_pair_mid_span_gain) * float(ridge.get("span_mm", 0.0)) + 18.0
                    if along_delta > max_along_delta or abs(along_mid) > max_mid:
                        continue
                    dir_neg = -normal
                    dir_pos = normal
                    move_neg, room_neg = self._v47_sweep_distance(src_neg, dir_neg, midgeom, bounds)
                    move_pos, room_pos = self._v47_sweep_distance(src_pos, dir_pos, midgeom, bounds)
                    if move_neg < 38.0 or move_pos < 38.0:
                        continue
                    dst_neg = src_neg + dir_neg * move_neg
                    dst_pos = src_pos + dir_pos * move_pos
                    rows = sorted([
                        {"src": src_neg, "dst": dst_neg, "dir": dir_neg, "move": move_neg, "meta": neg, "room": room_neg, "ridge_side": "negative"},
                        {"src": src_pos, "dst": dst_pos, "dir": dir_pos, "move": move_pos, "meta": pos, "room": room_pos, "ridge_side": "positive"},
                    ], key=lambda row: float(row["src"][0]))
                    assigned = {"arm2": rows[0], "arm1": rows[1]}
                    s1 = assigned["arm1"]["src"].astype(np.float32)
                    s2 = assigned["arm2"]["src"].astype(np.float32)
                    t1 = assigned["arm1"]["dst"].astype(np.float32)
                    t2 = assigned["arm2"]["dst"].astype(np.float32)
                    compat = compat_fn(s1, s2, t1, t2, args)
                    if not bool(compat.get("ok", False)):
                        continue
                    clear_z = float(getattr(args, "b34_clear_z", -25.0))
                    mid_ratio = float(np.clip(getattr(args, "b34_micro_mid_ratio", 0.55), 0.30, 0.75))
                    arm_points = {}
                    failed_pose = False
                    for arm_key in ("arm2", "arm1"):
                        row = assigned[arm_key]
                        srcp = np.asarray(row["src"], np.float64)
                        dstp = np.asarray(row["dst"], np.float64)
                        midp = srcp + mid_ratio * (dstp - srcp)
                        press_z = self._v47_press_z_for_arm(args, arm_key)
                        try:
                            hover = pose_fn(config, cfg, arm_key, srcp, clear_z)
                            down = pose_fn(config, cfg, arm_key, srcp, press_z)
                            mid = pose_fn(config, cfg, arm_key, midp, press_z)
                            sweep = pose_fn(config, cfg, arm_key, dstp, press_z)
                            lift = pose_fn(config, cfg, arm_key, dstp, clear_z)
                            flat = np.asarray(list(hover[:3]) + list(down[:3]) + list(sweep[:3]) + list(lift[:3]), np.float64)
                            if not np.all(np.isfinite(flat)):
                                failed_pose = True
                                break
                        except Exception:
                            failed_pose = True
                            break
                        arm_points[arm_key] = {
                            "role": "v47_ridge_side_region_sweep",
                            "detected_source_board": (float(srcp[0]), float(srcp[1])),
                            "source_board": (float(srcp[0]), float(srcp[1])),
                            "target_board": (float(dstp[0]), float(dstp[1])),
                            "wrinkle_tangent_board": (float(tangent[0]), float(tangent[1])),
                            "wrinkle_normal_board": (float(normal[0]), float(normal[1])),
                            "hover": hover,
                            "down": down,
                            "mid": mid,
                            "sweep": sweep,
                            "lift": lift,
                            "v47_ridge_side": str(row["ridge_side"]),
                            "v47_region_score": float(row["meta"].get("region_score", 0.0)),
                            "v47_interior_depth_px": float(row["meta"].get("interior_depth_px", 0.0)),
                            "v47_signed_normal_mm": float(row["meta"].get("signed_normal_mm", 0.0)),
                            "v47_along_mm": float(row["meta"].get("along_mm", 0.0)),
                            "v47_sweep_distance_mm": float(row["move"]),
                        }
                    if failed_pose or set(arm_points) != {"arm1", "arm2"}:
                        continue
                    plan = plan_cls(
                        True,
                        f"D55-V47 ridge-side two-region dual rank={rank}",
                        action="D55V14_SAME_WRINKLE_DUAL_COOP",
                        candidate_index=int(original_index),
                        center_px=tuple(float(v) for v in np.asarray(cand.get("center_px", [0.0, 0.0])).reshape(-1)[:2]),
                        center_board=(float(ridge["center_board"][0]), float(ridge["center_board"][1])),
                        major_axis_px=tuple(float(v) for v in np.asarray(cand.get("major_axis_px", [0.0, 0.0])).reshape(-1)[:2]),
                        normal_axis_board=(float(normal[0]), float(normal[1])),
                        arm_points=arm_points,
                        metrics={
                            "policy_family": "D55V47_RIDGE_SIDE_TWO_REGION_DUAL",
                            "d55_plan_mode": "V47_RIDGE_SIDE_TWO_REGION_DUAL",
                            "active_arm": "both",
                            "d26_no_grip_open": True,
                            "d55v47_two_cloth_regions": True,
                            "d55v47_stationary_support_used": False,
                            "d55v47_robot_motion_symmetry_not_objective": True,
                            "d55v47_original_candidate_index": int(original_index),
                            "d55v47_locked_wrinkle_rank": int(rank),
                            "d55v47_locked_wrinkle": copy.deepcopy(size),
                            "d55v47_ridge": {
                                "center_board": [float(v) for v in ridge["center_board"]],
                                "tangent_board": [float(v) for v in tangent],
                                "normal_board": [float(v) for v in normal],
                                "span_mm": float(ridge["span_mm"]),
                            },
                            "d55v47_dual_compatibility": copy.deepcopy(compat),
                            "d55v47_board_rect_only": [float(v) for v in bounds],
                            "arm1_press_distance_mm": float(np.linalg.norm(np.asarray(arm_points["arm1"]["target_board"]) - np.asarray(arm_points["arm1"]["source_board"]))),
                            "arm2_press_distance_mm": float(np.linalg.norm(np.asarray(arm_points["arm2"]["target_board"]) - np.asarray(arm_points["arm2"]["source_board"]))),
                        },
                    )
                    leg_lock = self._v70_press_leg_lock_report(plan, obs, ridge_center=ridge["center_board"])
                    if bool(leg_lock.get("enforced", False)) and not bool(leg_lock.get("ok", False)):
                        print(
                            f"[D55-V70-LEG-PAIR-REJECT] rank={rank} candidate={original_index+1} "
                            f"leg={leg_lock.get('ridge_leg')} reason={leg_lock.get('reason')}"
                        )
                        continue
                    plan.metrics["d55v70_leg_lock"] = copy.deepcopy(leg_lock)
                    shape_report = self._v43_shape_symmetry_report(plan, midgeom)
                    plan.metrics["d55v47_midvector_shape_geometry"] = copy.deepcopy(midgeom)
                    plan.metrics["d55v47_shape_symmetry_direction"] = copy.deepcopy(shape_report)
                    region_score = float(neg.get("region_score", 0.0)) + float(pos.get("region_score", 0.0))
                    interior_score = min(float(neg.get("interior_depth_px", 0.0)), float(pos.get("interior_depth_px", 0.0)))
                    direction_score = float(shape_report.get("direction_score", 0.0)) if bool(shape_report.get("available", False)) else 0.0
                    pair_score = (
                        52.0 * region_score
                        + 0.20 * min(interior_score, 35.0)
                        + 0.025 * min(start_sep, 420.0)
                        + 11.0 * direction_score
                        - 0.22 * float(along_delta)
                        - 0.08 * abs(float(along_mid))
                    )
                    plan.metrics["d55v47_pair_score"] = float(pair_score)
                    plan.metrics["d55v47_source_separation_mm"] = float(start_sep)
                    plan.metrics["d55v88_same_wrinkle_opposite_sides"] = True
                    plan.metrics["d55v88_pair_along_delta_mm"] = float(along_delta)
                    plan.metrics["d55v88_pair_along_mid_mm"] = float(along_mid)
                    gain = float(shape_report.get("symmetry_gain", 0.0)) if bool(shape_report.get("available", False)) else 0.0
                    plan.reason = (
                        f"D55-V47 rank={rank} {size.get('shape','RIDGE')} len={float(size.get('major_length_mm',0.0)):.1f}mm; "
                        f"two broad cloth-side points, both move away from ridge; shapeSymGain={gain:+.3f}"
                    )
                    print(
                        f"[D55-V47-PAIR] rank={rank} candidate={original_index+1} sep={start_sep:.1f}mm "
                        f"A2={plan.metrics['arm2_press_distance_mm']:.1f}mm A1={plan.metrics['arm1_press_distance_mm']:.1f}mm "
                        f"sameWrinkleAlongDelta={along_delta:.1f}mm shapeGain={gain:+.3f} score={pair_score:.1f}"
                    )
                    trials.append((float(pair_score), plan))
            if not trials:
                print(f"[D55-V47-PAIR-REJECT] rank={rank} candidate={original_index+1} reason=NO_SAFE_TWO_REGION_PAIR")
                return None
            trials.sort(key=lambda item: item[0], reverse=True)
            return trials[0][1]

        def _v47_build_region_area_plan(self, obs, heat, H, config, cfg, args):
            original_candidates = list(getattr(heat, "candidates", []) or []) if heat is not None else []
            if not original_candidates:
                return None
            ranked = []
            for index, cand in enumerate(original_candidates[:10]):
                if not isinstance(cand, dict):
                    continue
                size = self._v43_wrinkle_size_report(cand, heat, H, index)
                ranked.append((float(size["size_score"]), float(size["major_length_mm"]), float(size["area_px"]), index, cand, size))
            ranked.sort(key=lambda row: (row[0], row[1], row[2]), reverse=True)
            midgeom = self._v43_midvector_shape_geometry(obs)
            for rank, row in enumerate(ranked, 1):
                _, _, _, original_index, original_cand, size = row
                segments = self._v40_zone_segments(obs, original_cand, heat, H, original_index)
                candidates_to_try = segments[:6] if segments else [copy.deepcopy(original_cand)]
                rank_trials = []
                for local_index, cand in enumerate(candidates_to_try, 1):
                    plan = self._v47_build_two_region_plan(
                        obs, heat, H, config, cfg, args, cand, original_index, rank, size, midgeom
                    )
                    if plan is not None and bool(getattr(plan, "ok", False)):
                        rank_trials.append((float((getattr(plan, "metrics", {}) or {}).get("d55v47_pair_score", 0.0)), plan, local_index))
                if rank_trials:
                    rank_trials.sort(key=lambda item: item[0], reverse=True)
                    score, plan, local_index = rank_trials[0]
                    print(
                        f"[D55-V47-SELECT] rank={rank} candidate={original_index+1} local={local_index} "
                        f"shape={size['shape']} len={size['major_length_mm']:.1f}mm pairScore={score:.1f}"
                    )
                    return plan
                print(
                    f"[D55-V47-RANK-NOPLAN] rank={rank} candidate={original_index+1} "
                    f"shape={size['shape']} len={size['major_length_mm']:.1f}mm; trying next wrinkle"
                )
            return None

        def _v43_plan_score(self, obs, cand, plan, geometry_report, midgeom):
            seg = dict(cand.get("d55v40_local_segment", {}) or {}) if isinstance(cand, dict) else {}
            shape = str(cand.get("d55v40_region_shape", "UNKNOWN")) if isinstance(cand, dict) else "UNKNOWN"
            heat_strength = float(np.clip(seg.get("heat_strength", 0.0), 0.0, 1.0))
            linearity = max(1.0, float(seg.get("local_linearity", 1.0) or 1.0))
            local_quality = float(np.clip((linearity - 1.15) / 2.2, 0.0, 1.0))
            span_mm = max(0.0, float(seg.get("span_mm", 0.0) or 0.0))
            span_quality = float(np.clip(span_mm / 130.0, 0.0, 1.0))
            perp = float(geometry_report.get("perpendicular_quality", 0.0))
            shape_report = self._v43_shape_symmetry_report(plan, midgeom)
            direction_score = float(shape_report.get("direction_score", 0.0)) if bool(shape_report.get("available", False)) else 0.0
            score = (
                38.0 * perp
                + 24.0 * heat_strength
                + 16.0 * local_quality
                + 10.0 * span_quality
                + 12.0 * direction_score
            )
            if shape == "ZIGZAG":
                score += 2.0
            elif shape == "LINE":
                score += 3.0
            leg = self._v40_leg_region(obs, cand.get("center_board") if isinstance(cand, dict) else None)
            if bool(leg.get("near_crotch", False)):
                score -= 6.0
            return float(score), leg, shape_report

        def _v43_build_largest_wrinkle_plan(self, obs, heat, H, config, cfg, args):
            original_candidates = list(getattr(heat, "candidates", []) or []) if heat is not None else []
            if not original_candidates:
                return None

            ranked = []
            for index, cand in enumerate(original_candidates[:8]):
                if not isinstance(cand, dict):
                    continue
                size = self._v43_wrinkle_size_report(cand, heat, H, index)
                ranked.append((float(size["size_score"]), float(size["major_length_mm"]), float(size["area_px"]), index, cand, size))
            if not ranked:
                return None
            ranked.sort(key=lambda row: (row[0], row[1], row[2]), reverse=True)
            for rank, row in enumerate(ranked, 1):
                _, _, _, index, _, size = row
                print(
                    f"[D55-V43-WRINKLE-RANK] rank={rank} candidate={index+1} shape={size['shape']} "
                    f"len={size['major_length_mm']:.1f}mm area={size['area_px']:.0f}px "
                    f"severity={size['severity']:.2f} sizeScore={size['size_score']:.1f}"
                )

            midgeom = self._v43_midvector_shape_geometry(obs)
            if bool(midgeom.get("available", False)):
                print(
                    f"[D55-V43-MIDVECTOR] symBefore={float(midgeom.get('symmetry_before',0.0)):.2f} "
                    f"left={float(midgeom.get('left_width_mm',0.0)):.1f}mm right={float(midgeom.get('right_width_mm',0.0)):.1f}mm "
                    f"deficit={midgeom.get('deficit_side')}"
                )
            else:
                print(f"[D55-V43-MIDVECTOR] unavailable reason={midgeom.get('reason')}; largest-wrinkle policy still active")

            for rank, row in enumerate(ranked, 1):
                _, _, _, original_index, original_cand, size = row
                segment_candidates = self._v40_zone_segments(obs, original_cand, heat, H, original_index)
                if not segment_candidates:
                    print(
                        f"[D55-V43-LOCK-NOPLAN] rank={rank} candidate={original_index+1} "
                        f"len={size['major_length_mm']:.1f}mm reason=NO_LOCAL_SEGMENT; trying next-largest wrinkle"
                    )
                    continue
                print(
                    f"[D55-V43-LOCK] rank={rank} candidate={original_index+1} shape={size['shape']} "
                    f"len={size['major_length_mm']:.1f}mm area={size['area_px']:.0f}px segments={len(segment_candidates)}"
                )
                trials = []
                for trial_index, cand in enumerate(segment_candidates[:8]):
                    local_heat = copy.copy(heat)
                    local_heat.candidates = [copy.deepcopy(cand)]
                    try:
                        plan = super()._d55_build_perpendicular_plan(obs, local_heat, H, config, cfg, args)
                    except Exception as exc:
                        print(f"[D55-V43-PLAN-ERROR] trial={trial_index+1} {type(exc).__name__}: {exc}")
                        continue
                    if plan is None or not bool(getattr(plan, "ok", False)):
                        reason = "NONE" if plan is None else str(getattr(plan, "reason", "NO_SAFE_PLAN"))
                        print(
                            f"[D55-V43-PLAN-REJECT] trial={trial_index+1} candidate={original_index+1} "
                            f"shape={cand.get('d55v40_region_shape')} reason={reason}"
                        )
                        continue
                    geometry = self._v40_plan_geometry_report(plan, cand)
                    if not bool(geometry.get("usable", False)):
                        print(
                            f"[D55-V43-GEOM-REJECT] trial={trial_index+1} candidate={original_index+1} "
                            f"shape={cand.get('d55v40_region_shape')} reason={geometry.get('reason')}"
                        )
                        continue
                    score, leg, shape_report = self._v43_plan_score(obs, cand, plan, geometry, midgeom)
                    support_report = self._v46_support_proximity_report(plan, cand)
                    if bool(support_report.get("available", False)):
                        score += 18.0 * float(support_report.get("quality", 0.0))
                        score -= 12.0 * float(support_report.get("far_penalty", 0.0))
                    metrics = dict(getattr(plan, "metrics", {}) or {})
                    metrics.update({
                        "d55v43_largest_wrinkle_first": True,
                        "d55v43_locked_wrinkle_rank": int(rank),
                        "d55v43_locked_wrinkle": copy.deepcopy(size),
                        "d55v43_midvector_shape_geometry": copy.deepcopy(midgeom),
                        "d55v43_shape_symmetry_direction": copy.deepcopy(shape_report),
                        "d55v43_robot_motion_symmetry_not_objective": True,
                        "d55v43_plan_score": float(score),
                        "d55v46_support_proximity": copy.deepcopy(support_report),
                        "d55v40_bottom_ridge_planner": True,
                        "d55v40_region_shape": str(cand.get("d55v40_region_shape", "UNKNOWN")),
                        "d55v40_original_candidate_index": int(cand.get("_v40_original_index", 0)),
                        "d55v40_local_segment": copy.deepcopy(cand.get("d55v40_local_segment", {})),
                        "d55v40_geometry": copy.deepcopy(geometry),
                        "d55v40_leg_region": copy.deepcopy(leg),
                        "d55v40_hard_normal_limit_deg": 45.0,
                    })
                    plan.metrics = metrics
                    leg_lock = self._v70_press_leg_lock_report(plan, obs)
                    plan.metrics["d55v71_leg_lock_v43"] = copy.deepcopy(leg_lock)
                    if bool(leg_lock.get("enforced", False)) and not bool(leg_lock.get("ok", False)):
                        print(
                            f"[D55-V71-V43-LEG-REJECT] trial={trial_index+1} candidate={original_index+1} "
                            f"leg={leg_lock.get('ridge_leg')} reason={leg_lock.get('reason')}"
                        )
                        continue
                    try:
                        plan.candidate_index = int(original_index)
                    except Exception:
                        pass
                    old_reason = str(getattr(plan, "reason", ""))
                    gain = float(shape_report.get("symmetry_gain", 0.0)) if bool(shape_report.get("available", False)) else 0.0
                    plan.reason = (
                        f"D55-V43 largest wrinkle rank={rank} {cand.get('d55v40_region_shape','RIDGE')} "
                        f"len={size['major_length_mm']:.1f}mm shapeSymGain={gain:+.3f} score={score:.1f}; {old_reason}"
                    )
                    print(
                        f"[D55-V43-PLAN] trial={trial_index+1} candidate={original_index+1} "
                        f"shape={cand.get('d55v40_region_shape')} leg={leg.get('region')} "
                        f"normalErr={geometry.get('max_normal_error_deg')} shapeSymGain={gain:+.3f} "
                        f"shapeDir={float(shape_report.get('direction_score',0.0)):+.2f} "
                        f"supportDist={support_report.get('distance_mm')} score={score:.1f} action={getattr(plan,'action','-')}"
                    )
                    trials.append((float(score), plan, cand, geometry, leg, shape_report, support_report))
                if trials:
                    trials.sort(key=lambda item: item[0], reverse=True)
                    score, plan, cand, geometry, leg, shape_report, support_report = trials[0]
                    gain = float(shape_report.get("symmetry_gain", 0.0)) if bool(shape_report.get("available", False)) else 0.0
                    print(
                        f"[D55-V43-SELECT] lockedRank={rank} candidate={original_index+1} "
                        f"shape={cand.get('d55v40_region_shape')} len={size['major_length_mm']:.1f}mm "
                        f"shapeSymGain={gain:+.3f} supportDist={support_report.get('distance_mm')} "
                        f"score={score:.1f} action={getattr(plan,'action','-')}"
                    )
                    return plan
                print(
                    f"[D55-V43-LOCK-NOPLAN] rank={rank} candidate={original_index+1} "
                    f"len={size['major_length_mm']:.1f}mm reason=NO_SAFE_PLAN_ON_LOCKED_WRINKLE; trying next-largest wrinkle"
                )
            return None

        def _v52_press_mask_trust_report(self, obs):
            mask_obj = getattr(obs, "mask", None)
            mask_u8 = getattr(mask_obj, "mask_u8", None) if mask_obj is not None else None
            if mask_u8 is None:
                return {"ok": False, "reason": "MASK_NONE", "area_px": 0.0, "fallback": False}
            try:
                mask = (np.asarray(mask_u8) > 0).astype(np.uint8)
            except Exception:
                return {"ok": False, "reason": "MASK_ARRAY_INVALID", "area_px": 0.0, "fallback": False}
            if mask.ndim != 2:
                return {"ok": False, "reason": "MASK_NOT_2D", "area_px": 0.0, "fallback": False}
            area = float(np.count_nonzero(mask))
            parts = []
            for obj in (mask_obj, obs):
                for name in ("source", "mask_source", "class_name", "reason", "status", "raw_status"):
                    value = getattr(obj, name, None)
                    if value is not None:
                        parts.append(str(value))
                for name in ("d30_mask_info", "mask_info"):
                    value = getattr(obj, name, None)
                    if isinstance(value, dict):
                        for key in ("source", "fallback", "reason", "raw_status", "status"):
                            if value.get(key) is not None:
                                parts.append(str(value.get(key)))
            source = " | ".join(parts)
            upper = source.upper()
            fallback = any(token in upper for token in (
                "FALLBACK_BGDIFF", "CRUMPLED-FG", "CRUMPLED", "EMPTY-BOARD FOREGROUND", "BOARD_BLOB_FALLBACK"
            ))
            if area < 8000.0:
                return {"ok": False, "reason": "MASK_TOO_SMALL_ABSOLUTE", "area_px": area, "fallback": fallback, "source": source}
            reference = self._v30_last_trusted_mask_area_px
            if fallback and reference is not None and float(reference) >= 30000.0:
                ratio = area / max(float(reference), 1.0)
                if ratio < float(self._v30_fallback_min_ratio):
                    return {
                        "ok": False,
                        "reason": "UNTRUSTED_SMALL_FALLBACK_MASK",
                        "area_px": area,
                        "fallback": True,
                        "source": source,
                        "reference_area_px": float(reference),
                        "ratio": float(ratio),
                    }
            return {"ok": True, "reason": "OK", "area_px": area, "fallback": fallback, "source": source, "mask_u8": mask}

        def _v52_build_mask_shape_press_fallback(self, obs, H, config, cfg, args, shape_state):
            trust = self._v52_press_mask_trust_report(obs)
            if not bool(trust.get("ok", False)):
                print(
                    f"[D55-V52-MASK-FALLBACK-SKIP] reason={trust.get('reason')} "
                    f"area={float(trust.get('area_px',0.0)):.0f}px"
                )
                return None
            if H is None:
                print("[D55-V52-MASK-FALLBACK-SKIP] reason=H_NONE")
                return None
            mask = np.asarray(trust["mask_u8"], np.uint8)
            try:
                dist = self.cv2.distanceTransform(mask, self.cv2.DIST_L2, 5)
                stride = 4
                sy, sx = np.nonzero(mask[::stride, ::stride] > 0)
                if len(sx) < 80:
                    return None
                px = sx.astype(np.float64) * stride + 0.5 * stride
                py = sy.astype(np.float64) * stride + 0.5 * stride
                px = np.clip(px, 0, mask.shape[1] - 1)
                py = np.clip(py, 0, mask.shape[0] - 1)
                xi = np.asarray(np.rint(px), np.int32)
                yi = np.asarray(np.rint(py), np.int32)
                interior = dist[yi, xi].astype(np.float64)
                keep = interior >= 10.0
                if int(np.count_nonzero(keep)) < 40:
                    keep = interior >= 7.0
                px = px[keep]
                py = py[keep]
                interior = interior[keep]
                if len(px) < 30:
                    print("[D55-V52-MASK-FALLBACK-SKIP] reason=NO_DEEP_INTERIOR_POINTS")
                    return None
                pts_px = np.stack([px, py], axis=1).astype(np.float32)
                board = self.cv2.perspectiveTransform(
                    pts_px.reshape(-1, 1, 2), np.asarray(H, np.float32)
                ).reshape(-1, 2).astype(np.float64)
                finite = np.all(np.isfinite(board), axis=1)
                board = board[finite]
                pts_px = pts_px[finite]
                interior = interior[finite]
                if len(board) < 30:
                    return None

                center = np.median(board, axis=0)
                xmin, xmax, ymin, ymax = [float(v) for v in self._v47_board_rect(config, cfg)]
                split_margin = max(32.0, 0.08 * max(1.0, xmax - xmin))
                left_ids = np.nonzero(board[:, 0] <= float(center[0]) - 0.35 * split_margin)[0]
                right_ids = np.nonzero(board[:, 0] >= float(center[0]) + 0.35 * split_margin)[0]
                if len(left_ids) < 4 or len(right_ids) < 4:
                    qx25, qx75 = np.percentile(board[:, 0], [25.0, 75.0])
                    left_ids = np.nonzero(board[:, 0] <= qx25)[0]
                    right_ids = np.nonzero(board[:, 0] >= qx75)[0]
                if len(left_ids) < 3 or len(right_ids) < 3:
                    print("[D55-V52-MASK-FALLBACK-SKIP] reason=NO_DUAL_X_SIDES")
                    return None

                radial = board - center.reshape(1, 2)
                radial_norm = np.linalg.norm(radial, axis=1)
                interior_q = np.clip(interior / 30.0, 0.0, 1.0)
                radial_q = np.clip(1.0 - np.abs(radial_norm - 95.0) / 125.0, 0.0, 1.0)
                board_edge_room = np.minimum.reduce([
                    board[:, 0] - xmin, xmax - board[:, 0], board[:, 1] - ymin, ymax - board[:, 1]
                ])
                edge_q = np.clip(board_edge_room / 55.0, 0.0, 1.0)
                score = 0.62 * interior_q + 0.23 * radial_q + 0.15 * edge_q

                def pick(ids, max_count=8):
                    order = ids[np.argsort(score[ids])[::-1]]
                    chosen = []
                    for raw_idx in order[: min(len(order), 500)]:
                        idx = int(raw_idx)
                        p = board[idx]
                        if any(float(np.linalg.norm(p - old["source_board"])) < 36.0 for old in chosen):
                            continue
                        chosen.append({
                            "source_board": p.astype(np.float64),
                            "source_px": pts_px[idx].astype(np.float64),
                            "interior_depth_px": float(interior[idx]),
                            "score": float(score[idx]),
                        })
                        if len(chosen) >= max_count:
                            break
                    return chosen

                left = pick(left_ids)
                right = pick(right_ids)
                if not left or not right:
                    return None
                compat_fn = getattr(self.d55, "_d55v2_dual_compatibility_report", None)
                pose_fn = getattr(self.d55, "_roarm_pose_for_board", None)
                plan_cls = getattr(self.d55, "DualWrinkleStretchPlan", None)
                if not callable(compat_fn) or not callable(pose_fn) or plan_cls is None:
                    print("[D55-V52-MASK-FALLBACK-SKIP] reason=REQUIRED_D55_API_MISSING")
                    return None

                deficit = str(shape_state.get("deficit_side", "BALANCED")) if isinstance(shape_state, dict) else "BALANCED"
                clear_z = float(getattr(args, "b34_clear_z", -25.0))
                mid_ratio = float(np.clip(getattr(args, "b34_micro_mid_ratio", 0.55), 0.30, 0.75))
                trials = []
                tested = 0
                for lp in left[:8]:
                    for rp in right[:8]:
                        src2 = np.asarray(lp["source_board"], np.float64)
                        src1 = np.asarray(rp["source_board"], np.float64)
                        start_sep = float(np.linalg.norm(src1 - src2))
                        if start_sep < 90.0:
                            continue
                        dirs = {}
                        for arm_key, srcp in (("arm2", src2), ("arm1", src1)):
                            vec = srcp - center
                            n = float(np.linalg.norm(vec))
                            if n < 1e-6:
                                vec = np.asarray([-1.0, 0.0] if arm_key == "arm2" else [1.0, 0.0], np.float64)
                            else:
                                vec = vec / n
                            if arm_key == "arm2" and vec[0] > -0.15:
                                vec = np.asarray([-1.0, 0.0], np.float64)
                            elif arm_key == "arm1" and vec[0] < 0.15:
                                vec = np.asarray([1.0, 0.0], np.float64)
                            dirs[arm_key] = vec / max(float(np.linalg.norm(vec)), 1e-6)
                        requested2 = 75.0 + (12.0 if deficit == "LEFT" else 0.0)
                        requested1 = 75.0 + (12.0 if deficit == "RIGHT" else 0.0)
                        room2 = self._v47_ray_room_to_board(src2, dirs["arm2"], (xmin, xmax, ymin, ymax))
                        room1 = self._v47_ray_room_to_board(src1, dirs["arm1"], (xmin, xmax, ymin, ymax))
                        move2 = min(requested2, 120.0, max(0.0, room2 - 5.0))
                        move1 = min(requested1, 120.0, max(0.0, room1 - 5.0))
                        if move2 < 38.0 or move1 < 38.0:
                            continue
                        dst2 = src2 + dirs["arm2"] * move2
                        dst1 = src1 + dirs["arm1"] * move1
                        compat = dict(compat_fn(src1.astype(np.float32), src2.astype(np.float32), dst1.astype(np.float32), dst2.astype(np.float32), args) or {})
                        tested += 1
                        if not bool(compat.get("ok", False)):
                            continue
                        arm_points = {}
                        pose_failed = False
                        for arm_key, srcp, dstp, meta in (
                            ("arm2", src2, dst2, lp), ("arm1", src1, dst1, rp)
                        ):
                            midp = srcp + mid_ratio * (dstp - srcp)
                            press_z = self._v47_press_z_for_arm(args, arm_key)
                            try:
                                hover = pose_fn(config, cfg, arm_key, srcp, clear_z)
                                down = pose_fn(config, cfg, arm_key, srcp, press_z)
                                mid_pose = pose_fn(config, cfg, arm_key, midp, press_z)
                                sweep = pose_fn(config, cfg, arm_key, dstp, press_z)
                                lift = pose_fn(config, cfg, arm_key, dstp, clear_z)
                                flat = np.asarray(list(hover[:3]) + list(down[:3]) + list(sweep[:3]) + list(lift[:3]), np.float64)
                                if not np.all(np.isfinite(flat)):
                                    pose_failed = True
                                    break
                            except Exception:
                                pose_failed = True
                                break
                            arm_points[arm_key] = {
                                "role": "v52_mask_shape_interior_sweep",
                                "detected_source_board": (float(srcp[0]), float(srcp[1])),
                                "source_board": (float(srcp[0]), float(srcp[1])),
                                "target_board": (float(dstp[0]), float(dstp[1])),
                                "hover": hover,
                                "down": down,
                                "mid": mid_pose,
                                "sweep": sweep,
                                "lift": lift,
                                "v52_interior_depth_px": float(meta.get("interior_depth_px", 0.0)),
                                "v52_mask_source_score": float(meta.get("score", 0.0)),
                                "v52_sweep_distance_mm": float(np.linalg.norm(dstp - srcp)),
                            }
                        if pose_failed or set(arm_points) != {"arm1", "arm2"}:
                            continue
                        center_px = np.mean(np.vstack([lp["source_px"], rp["source_px"]]), axis=0)
                        plan = plan_cls(
                            True,
                            "D55-V52 trustworthy Mask contour/interior fallback",
                            action="D55V14_SAME_WRINKLE_DUAL_COOP",
                            candidate_index=-1,
                            center_px=(float(center_px[0]), float(center_px[1])),
                            center_board=(float(center[0]), float(center[1])),
                            major_axis_px=(0.0, 0.0),
                            normal_axis_board=(1.0, 0.0),
                            arm_points=arm_points,
                            metrics={
                                "policy_family": "D55V52_MASK_SHAPE_INTERIOR_DUAL",
                                "d55_plan_mode": "V52_MASK_SHAPE_INTERIOR_DUAL",
                                "active_arm": "both",
                                "d26_no_grip_open": True,
                                "d55v48_shape_first": True,
                                "d55v48_validator_override": True,
                                "d55v52_mask_shape_fallback": True,
                                "d55v52_only_after_wrinkle_exhausted": True,
                                "d55v52_mask_area_px": float(trust.get("area_px", 0.0)),
                                "d55v52_mask_source": str(trust.get("source", "")),
                                "d55v52_mask_fallback_source": bool(trust.get("fallback", False)),
                                "d55v52_source_separation_mm": float(start_sep),
                                "d55v52_dual_compatibility": copy.deepcopy(compat),
                                "d55v52_board_rect": [xmin, xmax, ymin, ymax],
                                "d55v52_shape_state": copy.deepcopy(shape_state),
                                "arm1_press_distance_mm": float(move1),
                                "arm2_press_distance_mm": float(move2),
                            },
                        )
                        ok, validation = self._v48_validate_press_plan(plan, obs, args)
                        if not bool(ok):
                            continue
                        plan.metrics["d55v52_validation"] = copy.deepcopy(validation)
                        pair_score = (
                            45.0 * float(lp.get("score", 0.0) + rp.get("score", 0.0))
                            + 0.035 * min(start_sep, 420.0)
                            + 0.08 * min(float(lp.get("interior_depth_px", 0.0)), float(rp.get("interior_depth_px", 0.0)), 35.0)
                        )
                        plan.metrics["d55v52_pair_score"] = float(pair_score)
                        plan.reason = (
                            f"D55-V52 Mask fallback: deep interior dual sources sep={start_sep:.1f}mm "
                            f"A2={move2:.1f}mm A1={move1:.1f}mm; wrinkle planners exhausted"
                        )
                        trials.append((float(pair_score), plan))
                if not trials:
                    print(
                        f"[D55-V52-MASK-FALLBACK-NOPLAN] trustworthyMaskArea={float(trust.get('area_px',0.0)):.0f}px "
                        f"testedPairs={tested}; no pair passed board/pose/dual compatibility"
                    )
                    return None
                trials.sort(key=lambda item: item[0], reverse=True)
                chosen = trials[0][1]
                print(
                    f"[D55-V52-MASK-FALLBACK-SELECT] area={float(trust.get('area_px',0.0)):.0f}px "
                    f"testedPairs={tested} sep={float(chosen.metrics.get('d55v52_source_separation_mm',0.0)):.1f}mm "
                    f"A2={float(chosen.metrics.get('arm2_press_distance_mm',0.0)):.1f}mm "
                    f"A1={float(chosen.metrics.get('arm1_press_distance_mm',0.0)):.1f}mm"
                )
                return chosen
            except Exception as exc:
                print(f"[D55-V52-MASK-FALLBACK-ERROR] {type(exc).__name__}: {exc}")
                return None

        def _v95_force_press_pair_from_candidates(self, obs, heat, H, config, cfg, args, ranked_rows, source_label="VALIDATED"):
            plan_cls = getattr(self.d55, "DualWrinkleStretchPlan", None)
            pose_fn = getattr(self.d55, "_roarm_pose_for_board", None)
            if plan_cls is None or not callable(pose_fn) or H is None:
                print("[D55-V95-FORCE-PAIR] required plan/pose API unavailable")
                return None
            mask_obj = getattr(obs, "mask", None)
            mask_u8 = getattr(mask_obj, "mask_u8", None) if mask_obj is not None else None
            board_samples = None
            interior_samples = None
            if mask_u8 is not None:
                try:
                    mask = (np.asarray(mask_u8) > 0).astype(np.uint8)
                    if mask.ndim == 2 and int(np.count_nonzero(mask)) >= 80:
                        dist = self.cv2.distanceTransform(mask, self.cv2.DIST_L2, 5)
                        stride = 4
                        sy, sx = np.nonzero(mask[::stride, ::stride] > 0)
                        if len(sx) >= 20:
                            px = np.stack([
                                sx.astype(np.float64) * stride + 0.5 * stride,
                                sy.astype(np.float64) * stride + 0.5 * stride,
                            ], axis=1)
                            px[:, 0] = np.clip(px[:, 0], 0, mask.shape[1] - 1)
                            px[:, 1] = np.clip(px[:, 1], 0, mask.shape[0] - 1)
                            xi = np.asarray(np.rint(px[:, 0]), np.int32)
                            yi = np.asarray(np.rint(px[:, 1]), np.int32)
                            intr = dist[yi, xi].astype(np.float64)
                            brd = self.cv2.perspectiveTransform(
                                px.astype(np.float32).reshape(-1, 1, 2), np.asarray(H, np.float32)
                            ).reshape(-1, 2).astype(np.float64)
                            finite = np.all(np.isfinite(brd), axis=1)
                            board_samples = brd[finite]
                            interior_samples = intr[finite]
                except Exception as exc:
                    print(f"[D55-V95-FORCE-MASK-WARN] {type(exc).__name__}: {exc}")

            bounds = self._v47_board_rect(config, cfg)
            xmin, xmax, ymin, ymax = [float(v) for v in bounds]
            source_offset = float(getattr(self, "_v95_press_force_source_offset_mm", 38.0))
            sweep_request = float(getattr(self, "_v95_press_force_sweep_mm", 72.0))
            clear_z = float(getattr(args, "b34_clear_z", -25.0))
            mid_ratio = float(np.clip(getattr(args, "b34_micro_mid_ratio", 0.55), 0.30, 0.75))

            for force_rank, row in enumerate(list(ranked_rows or [])[:8], 1):
                try:
                    if len(row) >= 7:
                        _, _, _, original_index, cand, size, ctx = row[:7]
                    else:
                        continue
                    crease_info = dict((ctx or {}).get("v61_crease_evidence", {}) or {})
                    crease_score = float(crease_info.get("score", 0.0) or 0.0) if bool(crease_info.get("available", False)) else 0.0
                    severity = float((size or {}).get("severity", 0.0) or 0.0)
                    if crease_score < float(self._v98_press_strong_crease_min_score) and severity < float(self._v98_press_strong_severity_min):
                        print(
                            f"[D55-V98-WEAK-SKIP] rank={force_rank} candidate={int(original_index)+1} "
                            f"crease={crease_score:.2f} severity={severity:.2f}; stronger wrinkle required"
                        )
                        continue
                    ridge = self._v47_ridge_frame(cand, heat, H)
                    if ridge is None:
                        continue
                    center = np.asarray(ridge["center_board"], np.float64).reshape(2)
                    tangent = np.asarray(ridge["tangent_board"], np.float64).reshape(2)
                    normal = np.asarray(ridge["normal_board"], np.float64).reshape(2)
                    tn = float(np.linalg.norm(tangent)); nn = float(np.linalg.norm(normal))
                    if tn < 1e-6 or nn < 1e-6:
                        continue
                    tangent /= tn; normal /= nn
                    span = float(ridge.get("span_mm", 0.0) or 0.0)
                    along_limit = max(65.0, min(150.0, 0.65 * max(span, 60.0) + 25.0))

                    picked = {}
                    if board_samples is not None and interior_samples is not None and len(board_samples) >= 10:
                        rel = board_samples - center.reshape(1, 2)
                        along = rel @ tangent
                        signed = rel @ normal
                        for sign, name in ((-1.0, "negative"), (1.0, "positive")):
                            valid = (
                                (signed * sign >= 8.0)
                                & (np.abs(along) <= along_limit)
                                & (interior_samples >= float(self._v98_press_mask_inset_px))
                            )
                            ids = np.nonzero(valid)[0]
                            if len(ids):
                                cost = (
                                    1.00 * np.abs(np.abs(signed[ids]) - source_offset)
                                    + 0.22 * np.abs(along[ids])
                                    - 0.28 * np.minimum(interior_samples[ids], 25.0)
                                )
                                j = int(ids[int(np.argmin(cost))])
                                picked[name] = board_samples[j].astype(np.float64)
                    for sign, name in ((-1.0, "negative"), (1.0, "positive")):
                        if name not in picked:
                            picked[name] = center + sign * normal * source_offset

                    min_sep = float(getattr(self, "_v96_press_force_min_pair_sep_mm", 140.0))
                    target_sep = max(min_sep, 160.0)
                    current_sep = float(np.linalg.norm(np.asarray(picked["positive"]) - np.asarray(picked["negative"])))
                    if current_sep < min_sep and board_samples is not None and interior_samples is not None and len(board_samples) >= 20:
                        rel = board_samples - center.reshape(1, 2)
                        along = rel @ tangent
                        signed = rel @ normal
                        valid_neg = np.nonzero((signed <= -8.0) & (np.abs(along) <= along_limit) & (interior_samples >= float(self._v98_press_mask_inset_px)))[0]
                        valid_pos = np.nonzero((signed >= 8.0) & (np.abs(along) <= along_limit) & (interior_samples >= float(self._v98_press_mask_inset_px)))[0]
                        if len(valid_neg) and len(valid_pos):
                            score_neg = np.abs(np.abs(signed[valid_neg]) - source_offset) + 0.10 * np.abs(along[valid_neg]) - 0.20 * np.minimum(interior_samples[valid_neg], 25.0)
                            score_pos = np.abs(np.abs(signed[valid_pos]) - source_offset) + 0.10 * np.abs(along[valid_pos]) - 0.20 * np.minimum(interior_samples[valid_pos], 25.0)
                            neg_ids = valid_neg[np.argsort(score_neg)[:80]]
                            pos_ids = valid_pos[np.argsort(score_pos)[:80]]
                            best_pair = None
                            best_cost = float("inf")
                            for ni in neg_ids:
                                qn = board_samples[int(ni)]
                                for pi in pos_ids:
                                    qp = board_samples[int(pi)]
                                    sep = float(np.linalg.norm(qp - qn))
                                    cost = abs(sep - target_sep) + 0.08 * (abs(float(along[int(ni)])) + abs(float(along[int(pi)])))
                                    if sep < min_sep:
                                        cost += 5.0 * (min_sep - sep)
                                    if cost < best_cost:
                                        best_cost = cost
                                        best_pair = (qn.copy(), qp.copy(), sep)
                            if best_pair is not None:
                                picked["negative"], picked["positive"], chosen_sep = best_pair
                                print(f"[D55-V96-WIDE-SOURCE] requestedSep>={min_sep:.0f}mm selectedSep={chosen_sep:.1f}mm")

                    rows = []
                    for sign, name in ((-1.0, "negative"), (1.0, "positive")):
                        srcp = np.asarray(picked[name], np.float64).reshape(2)
                        direction = sign * normal
                        room = float(self._v47_ray_room_to_board(srcp, direction, (xmin, xmax, ymin, ymax)))
                        move = min(float(sweep_request), max(30.0, room - 5.0))
                        dstp = srcp + direction * move
                        rows.append({"src": srcp, "dst": dstp, "dir": direction, "ridge_side": name})
                    rows.sort(key=lambda x: float(x["src"][0]))
                    assigned = {"arm2": rows[0], "arm1": rows[1]}
                    arm_points = {}
                    pose_failed = False
                    for arm_key in ("arm2", "arm1"):
                        rr = assigned[arm_key]
                        srcp = np.asarray(rr["src"], np.float64)
                        dstp = np.asarray(rr["dst"], np.float64)
                        midp = srcp + mid_ratio * (dstp - srcp)
                        press_z = self._v47_press_z_for_arm(args, arm_key)
                        try:
                            hover = pose_fn(config, cfg, arm_key, srcp, clear_z)
                            down = pose_fn(config, cfg, arm_key, srcp, press_z)
                            mid = pose_fn(config, cfg, arm_key, midp, press_z)
                            sweep = pose_fn(config, cfg, arm_key, dstp, press_z)
                            lift = pose_fn(config, cfg, arm_key, dstp, clear_z)
                            flat = np.asarray(list(hover[:3]) + list(down[:3]) + list(mid[:3]) + list(sweep[:3]) + list(lift[:3]), np.float64)
                            if not np.all(np.isfinite(flat)):
                                raise ValueError("non-finite pose")
                        except Exception:
                            pose_failed = True
                            break
                        arm_points[arm_key] = {
                            "role": "v95_forced_ridge_side_sweep",
                            "detected_source_board": (float(srcp[0]), float(srcp[1])),
                            "source_board": (float(srcp[0]), float(srcp[1])),
                            "target_board": (float(dstp[0]), float(dstp[1])),
                            "wrinkle_tangent_board": (float(tangent[0]), float(tangent[1])),
                            "wrinkle_normal_board": (float(normal[0]), float(normal[1])),
                            "hover": hover,
                            "down": down,
                            "mid": mid,
                            "sweep": sweep,
                            "lift": lift,
                            "v95_forced": True,
                            "v47_ridge_side": str(rr["ridge_side"]),
                            "v47_sweep_distance_mm": float(np.linalg.norm(dstp - srcp)),
                        }
                    if pose_failed or set(arm_points) != {"arm1", "arm2"}:
                        continue
                    center_px = np.asarray(cand.get("center_px", [0.0, 0.0]), np.float64).reshape(-1)
                    major_px = np.asarray(cand.get("major_axis_px", [0.0, 0.0]), np.float64).reshape(-1)
                    plan = plan_cls(
                        True,
                        f"D55-V95 forced deterministic pair rank={force_rank}",
                        action="D55V14_SAME_WRINKLE_DUAL_COOP",
                        candidate_index=int(original_index),
                        center_px=(float(center_px[0]) if center_px.size else 0.0, float(center_px[1]) if center_px.size > 1 else 0.0),
                        center_board=(float(center[0]), float(center[1])),
                        major_axis_px=(float(major_px[0]) if major_px.size else 0.0, float(major_px[1]) if major_px.size > 1 else 0.0),
                        normal_axis_board=(float(normal[0]), float(normal[1])),
                        arm_points=arm_points,
                        metrics={
                            "policy_family": "D55V95_FORCED_DETERMINISTIC_PAIR",
                            "d55_plan_mode": "V95_FORCE_PAIR",
                            "d22_motion_mode": "PRESS_SWEEP",
                            "active_arm": "both",
                            "d26_no_grip_open": True,
                            "d55v95_forced_pair": True,
                            "d55v96_forced_pair": True,
                            "d55v96_final_validator_bypass": True,
                            "d55v95_source_label": str(source_label),
                            "d55v95_safety_veto_bypassed_for_generation": True,
                            "d55v95_original_candidate_index": int(original_index),
                            "d55v95_locked_wrinkle": copy.deepcopy(size),
                            "d55v95_candidate_context": copy.deepcopy(ctx),
                            "d55v47_ridge": {
                                "center_board": [float(v) for v in center],
                                "tangent_board": [float(v) for v in tangent],
                                "normal_board": [float(v) for v in normal],
                                "span_mm": float(span),
                            },
                            "arm1_press_distance_mm": float(np.linalg.norm(np.asarray(arm_points["arm1"]["target_board"]) - np.asarray(arm_points["arm1"]["source_board"]))),
                            "arm2_press_distance_mm": float(np.linalg.norm(np.asarray(arm_points["arm2"]["target_board"]) - np.asarray(arm_points["arm2"]["source_board"]))),
                        },
                    )
                    a2 = np.asarray(arm_points["arm2"]["source_board"], np.float64)
                    a1 = np.asarray(arm_points["arm1"]["source_board"], np.float64)
                    sep = float(np.linalg.norm(a1 - a2))
                    plan.metrics["d55v96_source_separation_mm"] = sep
                    print(
                        f"[D55-V96-FORCE-PAIR] source={source_label} rank={force_rank} candidate={int(original_index)+1} "
                        f"A2=({a2[0]:.1f},{a2[1]:.1f}) A1=({a1[0]:.1f},{a1[1]:.1f}) sep={sep:.1f}mm "
                        f"sweepA2={float(plan.metrics['arm2_press_distance_mm']):.1f}mm "
                        f"sweepA1={float(plan.metrics['arm1_press_distance_mm']):.1f}mm; final safety validator BYPASSED"
                    )
                    return plan
                except Exception as exc:
                    print(f"[D55-V95-FORCE-CANDIDATE-ERROR] rank={force_rank} {type(exc).__name__}: {exc}")
                    continue
            return None

        def _d55_build_perpendicular_plan(self, obs, heat, H, config, cfg, args):
            enhanced_heat, light_report = self._v44_enhance_heat_for_light_garment(obs, heat, H, args)
            candidates = list(getattr(enhanced_heat, "candidates", []) or []) if enhanced_heat is not None else []
            if not candidates:
                print("[D55-V47-RETRY] no wrinkle candidate after adaptive heat")
                return self.d55.DualWrinkleStretchPlan(
                    False,
                    "D55-V47 no wrinkle candidate after adaptive light/contrast pass; REJUDGE",
                    metrics={
                        "d55_failure_category": "V47_NO_WRINKLE_CANDIDATE",
                        "d55v44_light_report": copy.deepcopy(light_report),
                        "d55v47_ridge_side_two_region_dual": True,
                    },
                )

            ranked = []
            for index, cand in enumerate(candidates[:14]):
                if not isinstance(cand, dict):
                    continue
                size = self._v43_wrinkle_size_report(cand, enhanced_heat, H, index)
                ranked.append((float(size["size_score"]), float(size["major_length_mm"]), float(size["area_px"]), index, cand, size))
            if not ranked:
                return self.d55.DualWrinkleStretchPlan(
                    False,
                    "D55-V47 candidate ranking failed; REJUDGE",
                    metrics={"d55_failure_category": "V47_RANKING_FAILED", "d55v47_ridge_side_two_region_dual": True},
                )
            ranked.sort(key=lambda row: (row[0], row[1], row[2]), reverse=True)
            for rank, row in enumerate(ranked[:10], 1):
                _, _, _, index, cand, size = row
                print(
                    f"[D55-V47-WRINKLE-RANK] rank={rank} candidate={index+1} source={cand.get('d55v44_source','BASE')} "
                    f"shape={size['shape']} len={size['major_length_mm']:.1f}mm area={size['area_px']:.0f}px "
                    f"severity={size['severity']:.2f} sizeScore={size['size_score']:.1f}"
                )

            print("[D55-V47-SEARCH] primary: split selected ridge into two cloth regions and move both arms away from it")
            plan = self._v47_build_region_area_plan(obs, enhanced_heat, H, config, cfg, args)
            if plan is not None and bool(getattr(plan, "ok", False)):
                metrics = dict(getattr(plan, "metrics", {}) or {})
                metrics.update({
                    "d55v44_light_report": copy.deepcopy(light_report),
                    "d55v47_ridge_side_two_region_dual": True,
                    "d55v47_stationary_support_is_fallback_only": True,
                    "d55v47_board_rectangle_only_for_targets": True,
                    "d55v47_private_workspace_gate_used": False,
                    "d55v47_dual_collision_guard_preserved": True,
                })
                plan.metrics = metrics
                return plan

            print("[D55-V47-SUPPORT-FALLBACK] no safe two-region dual pair; only now trying v46 stationary-support-capable planner")
            fallback = self._v43_build_largest_wrinkle_plan(obs, enhanced_heat, H, config, cfg, args)
            if fallback is not None and bool(getattr(fallback, "ok", False)):
                metrics = dict(getattr(fallback, "metrics", {}) or {})
                metrics.update({
                    "d55v44_light_report": copy.deepcopy(light_report),
                    "d55v47_ridge_side_two_region_dual": True,
                    "d55v47_stationary_support_fallback": True,
                    "d55v47_board_rectangle_only_for_targets": True,
                })
                fallback.metrics = metrics
                old_reason = str(getattr(fallback, "reason", ""))
                fallback.reason = f"D55-V47 stationary-support fallback after two-region search; {old_reason}"
                print(f"[D55-V47-SUPPORT-FALLBACK-SELECT] action={getattr(fallback,'action','-')}")
                return fallback

            print("[D55-V47-LEGACY-SAFE-SEARCH] region and support searches exhausted; trying inherited D55 planner as final safe fallback")
            try:
                legacy = super()._d55_build_perpendicular_plan(obs, enhanced_heat, H, config, cfg, args)
            except Exception as exc:
                print(f"[D55-V47-LEGACY-SAFE-ERROR] {type(exc).__name__}: {exc}")
                legacy = None
            if legacy is not None and bool(getattr(legacy, "ok", False)):
                metrics = dict(getattr(legacy, "metrics", {}) or {})
                metrics.update({
                    "d55v44_light_report": copy.deepcopy(light_report),
                    "d55v47_legacy_safe_fallback": True,
                    "d55v47_private_workspace_gate_used": False,
                    "d55v47_dual_collision_guard_preserved": True,
                })
                legacy.metrics = metrics
                old_reason = str(getattr(legacy, "reason", ""))
                legacy.reason = f"D55-V47 inherited-safe final fallback; {old_reason}"
                print(f"[D55-V47-LEGACY-SAFE-SELECT] action={getattr(legacy,'action','-')}")
                return legacy

            print("[D55-V47-RETRY] all wrinkle candidates exhausted")
            return self.d55.DualWrinkleStretchPlan(
                False,
                "D55-V47 all detected wrinkles exhausted; REJUDGE",
                metrics={
                    "d55_failure_category": "V47_EXHAUSTIVE_SAFE_SEARCH_FAILED",
                    "d55v44_light_report": copy.deepcopy(light_report),
                    "d55v47_ridge_side_two_region_dual": True,
                    "d55v47_stationary_support_is_fallback_only": True,
                    "d55v47_private_workspace_gate_used": False,
                    "d55v47_dual_collision_guard_preserved": True,
                },
            )


        def _install_v48_d55_validator(self):
            original = getattr(self.d55, "_d55v8_validate_plan", None)
            if not callable(original):
                raise RuntimeError("D55 validator API missing: _d55v8_validate_plan")
            self._v48_original_d55_validator = original

            def _wrapped(plan, obs, args):
                metrics = dict(getattr(plan, "metrics", {}) or {}) if plan is not None else {}
                if bool(metrics.get("d55v96_forced_pair", False)):
                    print("[D55-V96-VALIDATE-BYPASS] forced PRESS pair accepted; legacy inward/radial/leg/dual validator skipped")
                    return True, {"ok": True, "reason": "V96 forced PRESS pair accepted without legacy safety vetoes", "v96_forced": True}
                if bool(metrics.get("d55v48_shape_first", False)):
                    return self._v48_validate_press_plan(plan, obs, args)
                return original(plan, obs, args)

            self.d55._d55v8_validate_plan = _wrapped

        def _v48_validate_press_plan(self, plan, obs, args):
            if plan is None or not bool(getattr(plan, "ok", False)):
                return False, {"ok": False, "reason": "V48 plan unavailable"}
            arm_points = dict(getattr(plan, "arm_points", {}) or {})
            if set(arm_points) != {"arm1", "arm2"}:
                return False, {"ok": False, "reason": "V48 dual arms required", "arm_count": len(arm_points)}
            bounds = self._v47_board_rect(self.config, self.cfg55)
            xmin, xmax, ymin, ymax = [float(v) for v in bounds]
            rows = {}
            moving = 0
            for arm_key in ("arm1", "arm2"):
                points = arm_points.get(arm_key, {})
                src = self._finite_point2(points.get("source_board"))
                dst = self._finite_point2(points.get("target_board"))
                if src is None or dst is None:
                    return False, {"ok": False, "reason": f"V48 {arm_key} source/target missing"}
                src_inside = bool(xmin <= src[0] <= xmax and ymin <= src[1] <= ymax)
                dst_inside = bool(xmin <= dst[0] <= xmax and ymin <= dst[1] <= ymax)
                if not (src_inside and dst_inside):
                    return False, {
                        "ok": False,
                        "reason": f"V48 {arm_key} outside folding-board rectangle",
                        "source": [float(src[0]), float(src[1])],
                        "target": [float(dst[0]), float(dst[1])],
                        "board_rect": [xmin, xmax, ymin, ymax],
                    }
                move = dst - src
                move_mm = float(np.linalg.norm(move))
                role = str(points.get("role", "")).lower()
                is_stationary = bool(move_mm < 3.0 or "support" in role or "anchor" in role)
                if not is_stationary:
                    moving += 1
                    if move_mm < 24.0 or move_mm > 135.0:
                        return False, {"ok": False, "reason": f"V48 {arm_key} sweep length invalid", "move_mm": move_mm}
                    tangent = self._finite_point2(points.get("wrinkle_tangent_board"))
                    if tangent is not None and float(np.linalg.norm(tangent)) > 1e-6:
                        tangent = tangent / float(np.linalg.norm(tangent))
                        move_u = move / max(move_mm, 1e-6)
                        tangent_dot = abs(float(np.dot(tangent, move_u)))
                        normal_error = float(np.degrees(np.arcsin(np.clip(tangent_dot, 0.0, 1.0))))
                        if normal_error > 32.0:
                            return False, {"ok": False, "reason": f"V48 {arm_key} not wrinkle-normal", "normal_error_deg": normal_error}
                rows[arm_key] = {
                    "source": [float(src[0]), float(src[1])],
                    "target": [float(dst[0]), float(dst[1])],
                    "move_mm": move_mm,
                    "stationary": is_stationary,
                }
            if moving < 1:
                return False, {"ok": False, "reason": "V48 no moving PRESS arm", "arms": rows}
            compat_fn = getattr(self.d55, "_d55v2_dual_compatibility_report", None)
            compatibility = {"ok": True, "reason": "compatibility API unavailable"}
            if callable(compat_fn):
                try:
                    a1s = np.asarray(rows["arm1"]["source"], np.float32)
                    a2s = np.asarray(rows["arm2"]["source"], np.float32)
                    a1t = np.asarray(rows["arm1"]["target"], np.float32)
                    a2t = np.asarray(rows["arm2"]["target"], np.float32)
                    compatibility = dict(compat_fn(a1s, a2s, a1t, a2t, args) or {})
                except Exception as exc:
                    return False, {"ok": False, "reason": f"V48 dual compatibility error: {type(exc).__name__}: {exc}"}
                if not bool(compatibility.get("ok", False)):
                    return False, {"ok": False, "reason": "V48 dual collision/separation guard failed", "compatibility": compatibility, "arms": rows}
            report = {
                "ok": True,
                "reason": "V48 board-rectangle + wrinkle-normal + dual-compatibility validation passed",
                "legacy_center_radial_guard_bypassed": True,
                "board_rect": [xmin, xmax, ymin, ymax],
                "arms": rows,
                "compatibility": compatibility,
            }
            print("[D55-V48-VALIDATE] PASS legacy inward/radial-gain guard bypassed; board rectangle + dual compatibility OK")
            return True, report

        def _v48_shape_state(self, obs, H):
            mid = self._v43_midvector_shape_geometry(obs)
            result = {
                "available": False,
                "mode": "UNKNOWN",
                "reason": "MIDVECTOR_OR_MASK_UNAVAILABLE",
                "symmetry_score": None,
                "mask_symmetry": None,
                "hem_symmetry": mid.get("symmetry_before") if isinstance(mid, dict) else None,
                "deficit_side": str(mid.get("deficit_side", "NONE")) if isinstance(mid, dict) else "NONE",
                "left_deficit_ratio": 0.0,
                "right_deficit_ratio": 0.0,
                "missing_left_board": [],
                "missing_right_board": [],
                "midvector": copy.deepcopy(mid),
            }
            if not isinstance(mid, dict) or not bool(mid.get("available", False)) or H is None:
                return result
            mask_obj = getattr(obs, "mask", None)
            mask_u8 = getattr(mask_obj, "mask_u8", None) if mask_obj is not None else None
            if mask_u8 is None:
                return result
            origin = self._finite_point2(mid.get("origin_board"))
            lateral = self._finite_point2(mid.get("lateral_board"))
            if origin is None or lateral is None:
                return result
            ln = float(np.linalg.norm(lateral))
            if ln < 1e-6:
                return result
            lateral = lateral / ln
            try:
                mask = (np.asarray(mask_u8) > 0).astype(np.uint8)
                if mask.ndim != 2 or int(np.count_nonzero(mask)) < 800:
                    return result
                mask_tol = self.cv2.dilate(mask, np.ones((9, 9), np.uint8), iterations=1)
                stride = 5
                yy, xx = np.nonzero(mask[::stride, ::stride] > 0)
                if len(xx) < 120:
                    return result
                px = np.stack([
                    xx.astype(np.float32) * stride + 0.5 * stride,
                    yy.astype(np.float32) * stride + 0.5 * stride,
                ], axis=1)
                px[:, 0] = np.clip(px[:, 0], 0, mask.shape[1] - 1)
                px[:, 1] = np.clip(px[:, 1], 0, mask.shape[0] - 1)
                board = self.cv2.perspectiveTransform(px.reshape(-1, 1, 2), np.asarray(H, np.float32)).reshape(-1, 2).astype(np.float64)
                finite = np.all(np.isfinite(board), axis=1)
                board = board[finite]
                if len(board) < 100:
                    return result
                rel = board - origin.reshape(1, 2)
                signed = rel @ lateral
                reflected = board - 2.0 * signed.reshape(-1, 1) * lateral.reshape(1, 2)
                Hinv = np.linalg.inv(np.asarray(H, np.float64))
                reflected_px = self.cv2.perspectiveTransform(reflected.astype(np.float32).reshape(-1, 1, 2), Hinv.astype(np.float32)).reshape(-1, 2)
                rx = np.rint(reflected_px[:, 0]).astype(np.int32)
                ry = np.rint(reflected_px[:, 1]).astype(np.int32)
                in_frame = (rx >= 0) & (rx < mask.shape[1]) & (ry >= 0) & (ry < mask.shape[0])
                matched = np.zeros(len(reflected), dtype=bool)
                ids = np.nonzero(in_frame)[0]
                if len(ids):
                    matched[ids] = mask_tol[ry[ids], rx[ids]] > 0
                valid_count = int(np.count_nonzero(in_frame))
                if valid_count < 80:
                    return result
                mask_symmetry = float(np.mean(matched[in_frame]))
                reflected_signed = (reflected - origin.reshape(1, 2)) @ lateral
                left_expected = in_frame & (reflected_signed < -5.0)
                right_expected = in_frame & (reflected_signed > 5.0)
                left_missing = left_expected & (~matched)
                right_missing = right_expected & (~matched)
                left_ratio = float(np.count_nonzero(left_missing) / max(1, np.count_nonzero(left_expected)))
                right_ratio = float(np.count_nonzero(right_missing) / max(1, np.count_nonzero(right_expected)))
                miss_left = reflected[left_missing]
                miss_right = reflected[right_missing]
                def _thin(points):
                    arr = np.asarray(points, np.float64).reshape(-1, 2)
                    if len(arr) <= 180:
                        return arr
                    step = max(1, int(math.ceil(len(arr) / 180.0)))
                    return arr[::step]
                miss_left = _thin(miss_left)
                miss_right = _thin(miss_right)
                hem_sym = float(mid.get("symmetry_before", mask_symmetry) or mask_symmetry)
                combined = float(np.clip(0.78 * mask_symmetry + 0.22 * hem_sym, 0.0, 1.0))
                if combined >= 0.90:
                    mode = "SYMMETRIC_WRINKLE_CLEANUP"
                elif combined <= 0.84:
                    mode = "ASYMMETRIC_SHAPE_RECOVERY"
                else:
                    mode = "TRANSITION_SHAPE_RECOVERY"
                if left_ratio > right_ratio + 0.035:
                    deficit_side = "LEFT"
                elif right_ratio > left_ratio + 0.035:
                    deficit_side = "RIGHT"
                else:
                    deficit_side = "BALANCED"
                result.update({
                    "available": True,
                    "mode": mode,
                    "reason": "OK",
                    "symmetry_score": combined,
                    "mask_symmetry": mask_symmetry,
                    "hem_symmetry": hem_sym,
                    "deficit_side": deficit_side,
                    "left_deficit_ratio": left_ratio,
                    "right_deficit_ratio": right_ratio,
                    "missing_left_board": miss_left.tolist(),
                    "missing_right_board": miss_right.tolist(),
                    "origin_board": [float(origin[0]), float(origin[1])],
                    "lateral_board": [float(lateral[0]), float(lateral[1])],
                })
                print(
                    f"[D55-V48-SHAPE] mode={mode} sym={combined:.3f} maskSym={mask_symmetry:.3f} hemSym={hem_sym:.3f} "
                    f"deficit={deficit_side} missL={left_ratio:.3f} missR={right_ratio:.3f}"
                )
                return result
            except Exception as exc:
                result["reason"] = f"SHAPE_ANALYSIS_ERROR:{type(exc).__name__}:{exc}"
                print(f"[D55-V48-SHAPE-ERROR] {type(exc).__name__}: {exc}")
                return result

        def _v48_candidate_context(self, obs, cand, size, shape_state):
            center = self._finite_point2(cand.get("center_board")) if isinstance(cand, dict) else None
            if center is None and isinstance(cand, dict):
                center = self._finite_point2((cand.get("d21_geometry", {}) or {}).get("center_board"))
            context = {
                "side": "UNKNOWN",
                "deficit_ratio": 0.0,
                "missing_distance_mm": None,
                "asymmetry_relevance": 0.0,
                "leg_region": "UNKNOWN",
                "priority_score": float(size.get("size_score", 0.0)),
            }
            if center is None:
                return context
            leg = self._v40_leg_region(obs, center)
            context["leg_region"] = str(leg.get("region", "UNKNOWN"))
            if not isinstance(shape_state, dict) or not bool(shape_state.get("available", False)):
                return context
            origin = self._finite_point2(shape_state.get("origin_board"))
            lateral = self._finite_point2(shape_state.get("lateral_board"))
            if origin is None or lateral is None or float(np.linalg.norm(lateral)) < 1e-6:
                return context
            lateral = lateral / float(np.linalg.norm(lateral))
            signed = float(np.dot(center - origin, lateral))
            side = "LEFT" if signed < 0.0 else "RIGHT"
            context["side"] = side
            deficit_ratio = float(shape_state.get("left_deficit_ratio", 0.0) if side == "LEFT" else shape_state.get("right_deficit_ratio", 0.0))
            context["deficit_ratio"] = deficit_ratio
            missing = np.asarray(shape_state.get("missing_left_board" if side == "LEFT" else "missing_right_board", []), np.float64).reshape(-1, 2)
            relevance = 0.0
            distance = None
            if len(missing):
                d = np.linalg.norm(missing - center.reshape(1, 2), axis=1)
                distance = float(np.percentile(d, 12.0))
                relevance = float(math.exp(-((distance / 115.0) ** 2)))
            context["missing_distance_mm"] = distance
            context["asymmetry_relevance"] = relevance
            mode = str(shape_state.get("mode", "UNKNOWN"))
            priority = float(size.get("size_score", 0.0))
            if mode == "ASYMMETRIC_SHAPE_RECOVERY":
                priority += 42.0 * relevance + 24.0 * min(1.0, deficit_ratio / 0.30)
            elif mode == "TRANSITION_SHAPE_RECOVERY":
                priority += 20.0 * relevance + 10.0 * min(1.0, deficit_ratio / 0.30)
            context["priority_score"] = float(priority)
            return context

        def _v61_wrinkle_crease_evidence(self, obs, cand, heat):
            report = {
                "available": False,
                "score": 0.0,
                "strength_l": 0.0,
                "consistency": 0.0,
                "bilateral_ratio": 0.0,
                "samples": 0,
                "reason": "UNAVAILABLE",
            }
            comp = self._v40_candidate_component(cand, heat)
            frame = getattr(obs, "frame", None)
            if comp is None or frame is None:
                return report
            try:
                img = np.asarray(frame)
                if img.ndim != 3 or img.shape[2] < 3:
                    return report
                h, w = comp["shape"]
                if img.shape[:2] != (h, w):
                    img = self.cv2.resize(img, (w, h), interpolation=self.cv2.INTER_LINEAR)
                lab = self.cv2.cvtColor(img, self.cv2.COLOR_BGR2LAB)
                lum = np.asarray(lab[:, :, 0], np.float32)

                xx = np.asarray(comp["xx"], np.float64)
                yy = np.asarray(comp["yy"], np.float64)
                if len(xx) < 24:
                    report["reason"] = "COMPONENT_TOO_SMALL"
                    return report
                pts = np.column_stack((xx, yy))
                center = np.mean(pts, axis=0)
                q = pts - center[None, :]
                cov = q.T @ q / max(1, len(q)-1)
                vals, vecs = np.linalg.eigh(cov)
                tangent = np.asarray(vecs[:, int(np.argmax(vals))], np.float64)
                tangent /= max(float(np.linalg.norm(tangent)), 1e-9)
                normal = np.asarray([-tangent[1], tangent[0]], np.float64)
                proj = q @ tangent
                lo, hi = np.percentile(proj, [12.0, 88.0])
                if not math.isfinite(float(hi-lo)) or float(hi-lo) < 8.0:
                    report["reason"] = "COMPONENT_SPAN_TOO_SMALL"
                    return report

                mask_obj = getattr(obs, "mask", None)
                mask = getattr(mask_obj, "mask_u8", None) if mask_obj is not None else None
                mask_arr = None
                if mask is not None:
                    mask_arr = np.asarray(mask)
                    if mask_arr.shape[:2] != (h, w):
                        mask_arr = self.cv2.resize(mask_arr.astype(np.uint8), (w, h), interpolation=self.cv2.INTER_NEAREST)

                def patch_mean(pt, radius=2):
                    x = int(round(float(pt[0]))); y = int(round(float(pt[1])))
                    if x-radius < 0 or x+radius >= w or y-radius < 0 or y+radius >= h:
                        return None
                    if mask_arr is not None:
                        roi_m = mask_arr[y-radius:y+radius+1, x-radius:x+radius+1]
                        if roi_m.size == 0 or float(np.mean(roi_m > 0)) < 0.60:
                            return None
                    roi = lum[y-radius:y+radius+1, x-radius:x+radius+1]
                    return float(np.mean(roi)) if roi.size else None

                strengths = []
                balances = []
                sign_ok_count = 0
                valid_count = 0
                for frac in (0.14, 0.26, 0.38, 0.50, 0.62, 0.74, 0.86):
                    along = float(lo + (hi-lo)*frac)
                    band = np.abs(proj-along) <= max(2.0, 0.035*float(hi-lo))
                    if int(np.count_nonzero(band)) >= 2:
                        anchor = np.mean(pts[band], axis=0)
                    else:
                        j = int(np.argmin(np.abs(proj-along)))
                        anchor = pts[j]
                    c0 = patch_mean(anchor, 2)
                    if c0 is None:
                        continue
                    best_strength = 0.0
                    best_balance = 0.0
                    best_same = False
                    for dist in (5.0, 8.0, 11.0, 14.0):
                        lm = patch_mean(anchor-normal*dist, 2)
                        rp = patch_mean(anchor+normal*dist, 2)
                        if lm is None or rp is None:
                            continue
                        dl = c0-lm
                        dr = c0-rp
                        al, ar = abs(dl), abs(dr)
                        bilateral = min(al, ar)
                        balance = min(al, ar)/max(max(al, ar), 1e-6)
                        same = bool(dl*dr > 0.0)
                        value = bilateral * (0.35 + 0.65*balance) * (1.0 if same else 0.20)
                        if value > best_strength:
                            best_strength = float(value)
                            best_balance = float(balance)
                            best_same = same
                    if best_strength <= 0.0:
                        continue
                    valid_count += 1
                    strengths.append(best_strength)
                    balances.append(best_balance)
                    if best_same and best_strength >= 3.5:
                        sign_ok_count += 1

                if valid_count < 3:
                    report.update({"reason": "TOO_FEW_BILATERAL_SAMPLES", "samples": int(valid_count)})
                    return report
                strength = float(np.median(strengths))
                consistency = float(sign_ok_count/max(valid_count, 1))
                balance = float(np.median(balances))
                strength_q = float(np.clip((strength-2.5)/15.0, 0.0, 1.0))
                score = float(np.clip(0.50*strength_q + 0.35*consistency + 0.15*balance, 0.0, 1.0))
                report.update({
                    "available": True,
                    "score": score,
                    "strength_l": strength,
                    "consistency": consistency,
                    "bilateral_ratio": balance,
                    "samples": int(valid_count),
                    "reason": "OK",
                })
                return report
            except Exception as exc:
                report["reason"] = f"ERROR:{type(exc).__name__}"
                return report

        def _v48_ranked_wrinkles(self, obs, heat, H, shape_state):
            if self._v67_prepare_cancel_requested():
                print("[D55-V69-CANCEL] abort before wrinkle ranking")
                return []
            candidates = list(getattr(heat, "candidates", []) or []) if heat is not None else []
            ranked = []
            rejected = 0
            for index, cand in enumerate(candidates[:14]):
                if self._v67_prepare_cancel_requested():
                    print(f"[D55-V69-CANCEL] abort wrinkle ranking at candidate={index+1}")
                    return []
                if not isinstance(cand, dict):
                    continue
                size = self._v43_wrinkle_size_report(cand, heat, H, index)
                crease = self._v61_wrinkle_crease_evidence(obs, cand, heat)
                size["v61_crease_evidence"] = copy.deepcopy(crease)
                source = str(size.get("source", "BASE")).upper()
                shape = str(size.get("shape", "IRREGULAR")).upper()
                major = float(size.get("major_length_mm", 0.0) or 0.0)

                reject_reason = None
                if source == "ADAPTIVE_LIGHT" and shape == "IRREGULAR" and major >= float(self._v61_press_irregular_long_reject_mm):
                    reject_reason = "LONG_ADAPTIVE_IRREGULAR_ARTIFACT"
                elif source == "ADAPTIVE_LIGHT" and bool(crease.get("available", False)) and float(crease.get("score", 0.0)) < float(self._v61_press_crease_min_score):
                    reject_reason = "WEAK_BILATERAL_CREASE_EVIDENCE"
                elif source == "ADAPTIVE_LIGHT" and not bool(crease.get("available", False)) and shape == "IRREGULAR":
                    reject_reason = "IRREGULAR_WITHOUT_CREASE_EVIDENCE"

                if reject_reason is not None:
                    rejected += 1
                    print(
                        f"[D55-V61-CREASE-REJECT] candidate={index+1} source={source} shape={shape} "
                        f"len={major:.1f}mm score={float(crease.get('score',0.0)):.2f} reason={reject_reason}"
                    )
                    continue

                ctx = self._v48_candidate_context(obs, cand, size, shape_state)
                crease_score = float(crease.get("score", 0.0)) if bool(crease.get("available", False)) else 0.35
                ctx["v61_crease_evidence"] = copy.deepcopy(crease)
                ctx["priority_score_pre_crease"] = float(ctx["priority_score"])
                if source == "ADAPTIVE_LIGHT":
                    ctx["priority_score"] = float(ctx["priority_score"] * (0.72 + 0.45*crease_score))
                else:
                    ctx["priority_score"] = float(ctx["priority_score"] * (0.90 + 0.25*crease_score))
                ranked.append((float(ctx["priority_score"]), float(size["major_length_mm"]), float(size["size_score"]), index, cand, size, ctx))

            ranked.sort(key=lambda row: (row[0], row[1], row[2]), reverse=True)
            left_max = 0.0
            right_max = 0.0
            for rank, row in enumerate(ranked, 1):
                priority, _, _, index, _, size, ctx = row
                side = str(ctx.get("side", "UNKNOWN"))
                if side == "LEFT":
                    left_max = max(left_max, float(size.get("size_score", 0.0)))
                elif side == "RIGHT":
                    right_max = max(right_max, float(size.get("size_score", 0.0)))
                crease = dict(ctx.get("v61_crease_evidence", {}) or {})
                print(
                    f"[D55-V48-WRINKLE-RANK] rank={rank} candidate={index+1} side={ctx.get('side')} leg={ctx.get('leg_region')} "
                    f"shape={size['shape']} len={size['major_length_mm']:.1f}mm severity={size['severity']:.2f} "
                    f"base={size['size_score']:.1f} crease={float(crease.get('score',0.0)):.2f} "
                    f"asymRel={float(ctx.get('asymmetry_relevance',0.0)):.2f} "
                    f"deficit={float(ctx.get('deficit_ratio',0.0)):.2f} priority={priority:.1f}"
                )
            print(f"[D55-V61-CREASE] candidates={len(candidates[:14])} accepted={len(ranked)} rejected={rejected} minScore={float(self._v61_press_crease_min_score):.2f}")
            print(f"[D55-V48-LEG-WRINKLE] leftMax={left_max:.1f} rightMax={right_max:.1f}")
            return ranked

        def _v48_shape_plan_score(self, plan, shape_state, base_pair_score):
            score = float(base_pair_score)
            report = self._v43_shape_symmetry_report(plan, shape_state.get("midvector", {}) if isinstance(shape_state, dict) else {})
            gain = float(report.get("symmetry_gain", 0.0)) if bool(report.get("available", False)) else 0.0
            mode = str(shape_state.get("mode", "UNKNOWN")) if isinstance(shape_state, dict) else "UNKNOWN"
            if mode == "ASYMMETRIC_SHAPE_RECOVERY":
                score += 34.0 * float(np.clip(gain / 0.10, -1.0, 1.0))
            elif mode == "TRANSITION_SHAPE_RECOVERY":
                score += 16.0 * float(np.clip(gain / 0.10, -1.0, 1.0))
            return float(score), report

        def _v48_mark_plan(self, plan, shape_state, rank, size, context, mode_reason):
            if plan is None:
                return plan
            metrics = dict(getattr(plan, "metrics", {}) or {})
            metrics.update({
                "d55v48_shape_first": True,
                "d55v48_validator_override": True,
                "d55v48_shape_state": copy.deepcopy(shape_state),
                "d55v48_selected_rank": int(rank),
                "d55v48_selected_wrinkle": copy.deepcopy(size),
                "d55v48_selected_context": copy.deepcopy(context),
                "d55v48_mode_reason": str(mode_reason),
                "d55v48_legacy_center_radial_guard_bypassed": True,
                "d55v48_board_rectangle_only": True,
                "d55v48_dual_collision_guard_preserved": True,
            })
            plan.metrics = metrics
            return plan

        def _v48_build_shape_first_plan(self, obs, heat, H, config, cfg, args, shape_state, ranked):
            mode = str(shape_state.get("mode", "UNKNOWN")) if isinstance(shape_state, dict) else "UNKNOWN"
            for rank, row in enumerate(ranked, 1):
                if self._v67_prepare_cancel_requested():
                    print(f"[D55-V69-CANCEL] abort shape-first planning at rank={rank}")
                    return None
                _, _, _, original_index, original_cand, size, context = row
                segments = self._v40_zone_segments(obs, original_cand, heat, H, original_index)
                candidates_to_try = segments[:8] if segments else [copy.deepcopy(original_cand)]
                trials = []
                for local_index, cand in enumerate(candidates_to_try, 1):
                    if self._v67_prepare_cancel_requested():
                        print(f"[D55-V69-CANCEL] abort local wrinkle planning rank={rank} local={local_index}")
                        return None
                    plan = self._v47_build_two_region_plan(
                        obs, heat, H, config, cfg, args, cand, original_index, rank, size,
                        shape_state.get("midvector", {}) if isinstance(shape_state, dict) else {},
                    )
                    if plan is None or not bool(getattr(plan, "ok", False)):
                        continue
                    base_score = float((getattr(plan, "metrics", {}) or {}).get("d55v47_pair_score", 0.0))
                    final_score, shape_report = self._v48_shape_plan_score(plan, shape_state, base_score)
                    plan.metrics["d55v48_shape_direction_report"] = copy.deepcopy(shape_report)
                    plan.metrics["d55v48_final_plan_score"] = float(final_score)
                    shape_gain = float(shape_report.get("symmetry_gain", 0.0)) if bool(shape_report.get("available", False)) else None
                    if (
                        mode in {"ASYMMETRIC_SHAPE_RECOVERY", "TRANSITION_SHAPE_RECOVERY"}
                        and shape_gain is not None
                        and shape_gain < float(self._v61_press_shape_gain_floor)
                    ):
                        print(
                            f"[D55-V61-SHAPE-REJECT] rank={rank} candidate={original_index+1} local={local_index} "
                            f"shapeGain={shape_gain:+.3f} < {float(self._v61_press_shape_gain_floor):+.3f}"
                        )
                        continue
                    trials.append((float(final_score), plan, local_index))
                if trials:
                    trials.sort(key=lambda item: item[0], reverse=True)
                    final_score, plan, local_index = trials[0]
                    plan = self._v48_mark_plan(plan, shape_state, rank, size, context, mode)
                    old_reason = str(getattr(plan, "reason", ""))
                    plan.reason = (
                        f"D55-V48 {mode} rank={rank} {size['shape']} len={size['major_length_mm']:.1f}mm "
                        f"side={context.get('side')} asymRel={float(context.get('asymmetry_relevance',0.0)):.2f} score={final_score:.1f}; {old_reason}"
                    )
                    print(
                        f"[D55-V48-SELECT] mode={mode} rank={rank} candidate={original_index+1} local={local_index} "
                        f"side={context.get('side')} shape={size['shape']} len={size['major_length_mm']:.1f}mm score={final_score:.1f}"
                    )
                    return plan
                print(
                    f"[D55-V48-RANK-NOPLAN] mode={mode} rank={rank} candidate={original_index+1} "
                    f"shape={size['shape']} len={size['major_length_mm']:.1f}mm; trying next ranked wrinkle"
                )
            return None

        def _d55_build_perpendicular_plan(self, obs, heat, H, config, cfg, args):
            if self._v67_prepare_cancel_requested():
                print("[D55-V69-CANCEL] abort PRESS before enhancement")
                return self.d55.DualWrinkleStretchPlan(False, "V69_CANCELLED_BY_HUMAN_OVERRIDE")
            enhanced_heat, light_report = self._v44_enhance_heat_for_light_garment(obs, heat, H, args)
            if self._v67_prepare_cancel_requested():
                print("[D55-V69-CANCEL] abort PRESS after enhancement")
                return self.d55.DualWrinkleStretchPlan(False, "V69_CANCELLED_BY_HUMAN_OVERRIDE")
            shape_state = self._v48_shape_state(obs, H)
            if self._v67_prepare_cancel_requested():
                print("[D55-V69-CANCEL] abort PRESS after shape analysis")
                return self.d55.DualWrinkleStretchPlan(False, "V69_CANCELLED_BY_HUMAN_OVERRIDE")
            ranked = self._v48_ranked_wrinkles(obs, enhanced_heat, H, shape_state)
            if self._v67_prepare_cancel_requested():
                print("[D55-V67-CANCEL] human selected another action; abort PRESS planning now")
                return self.d55.DualWrinkleStretchPlan(False, "V67_CANCELLED_BY_HUMAN_OVERRIDE")

            filtered_heat = enhanced_heat
            if enhanced_heat is not None:
                try:
                    filtered_heat = copy.copy(enhanced_heat)
                    filtered_heat.candidates = [copy.deepcopy(row[4]) for row in ranked]
                except Exception:
                    filtered_heat = enhanced_heat

            raw_count = len(list(getattr(enhanced_heat, "candidates", []) or [])) if enhanced_heat is not None else 0
            if raw_count > 0 and not ranked:
                pseudo_ranked = []
                for index, cand in enumerate(list(getattr(enhanced_heat, "candidates", []) or [])[:14]):
                    if not isinstance(cand, dict):
                        continue
                    try:
                        size = self._v43_wrinkle_size_report(cand, enhanced_heat, H, index)
                    except Exception:
                        size = {"shape": "WRINKLE", "major_length_mm": 0.0, "size_score": 0.0}
                    pseudo_ranked.append((0.0, float(size.get("major_length_mm", 0.0) or 0.0), float(size.get("size_score", 0.0) or 0.0), index, cand, size, {}))
                forced = self._v95_force_press_pair_from_candidates(
                    obs, enhanced_heat, H, config, cfg, args, pseudo_ranked, source_label="RAW_WRINKLE"
                )
                if forced is not None and bool(getattr(forced, "ok", False)):
                    return forced
                print("[D55-V95-FORCE-BLOCK] raw wrinkle candidates exist but even deterministic pose construction failed")
                return self.d55.DualWrinkleStretchPlan(
                    False,
                    "D55-V95 forced pair construction failed for raw wrinkle candidates; REJUDGE",
                    metrics={
                        "d55_failure_category": "V95_FORCE_PAIR_POSE_BUILD_FAILED",
                        "d55v61_raw_wrinkle_candidates": int(raw_count),
                        "d55v48_shape_state": copy.deepcopy(shape_state),
                        "d55v44_light_report": copy.deepcopy(light_report),
                    },
                )

            if ranked:
                plan = self._v48_build_shape_first_plan(obs, filtered_heat, H, config, cfg, args, shape_state, ranked)
                if plan is not None and bool(getattr(plan, "ok", False)):
                    metrics = dict(getattr(plan, "metrics", {}) or {})
                    metrics["d55v44_light_report"] = copy.deepcopy(light_report)
                    metrics["d55v61_validated_crease_only"] = True
                    plan.metrics = metrics
                    plan = self._v56_finalize_press_plan(plan, obs, H, config, cfg, args)
                    if plan is not None and bool(getattr(plan, "ok", False)):
                        return plan

            if self._v67_prepare_cancel_requested():
                return self.d55.DualWrinkleStretchPlan(False, "V67_CANCELLED_BY_HUMAN_OVERRIDE")
            forced = self._v95_force_press_pair_from_candidates(
                obs, enhanced_heat, H, config, cfg, args, ranked, source_label="VALIDATED_RED_WRINKLE"
            )
            if forced is not None and bool(getattr(forced, "ok", False)):
                return forced
            print(
                "[D55-V95-FORCE-BLOCK] validated wrinkle(s) exist but deterministic PRESS source/target pose construction failed"
            )
            return self.d55.DualWrinkleStretchPlan(
                False,
                "D55-V95 forced pair construction failed; REJUDGE",
                metrics={
                    "d55_failure_category": "V95_FORCE_PAIR_POSE_BUILD_FAILED",
                    "d55v61_validated_candidate_count": int(len(ranked)),
                    "d55v95_force_pair_attempted": True,
                    "d55v48_shape_state": copy.deepcopy(shape_state),
                    "d55v44_light_report": copy.deepcopy(light_report),
                },
            )

        def _v67_prepare_cancel_requested(self):
            try:
                queued = getattr(self, "_queued_human_action", None)
                generation_stale = (
                    bool(getattr(self, "inference_busy", False))
                    and int(getattr(self, "_prepare_generation", 0))
                    != int(getattr(self, "_active_prepare_generation", 0))
                )
                return bool(queued is not None or generation_stale)
            except Exception:
                return bool(getattr(self, "_queued_human_action", None))

        def _v96_install_d60_fast_prelift(self):
            original = getattr(self.d56, "_63_execute_step120_motion_from_d56_plan", None)
            if not callable(original):
                raise RuntimeError("V96 D60 executor missing")
            if bool(getattr(original, "_bottom_vla_v96_fast_prelift", False)):
                return
            source = textwrap.dedent(inspect.getsource(original))
            needle = "plateau_polls_needed = int(np.clip(getattr(args, 'step60_3_pre_pullup_plateau_polls', 3), 2, 6))"
            replacement = "plateau_polls_needed = int(np.clip(getattr(args, 'step60_3_pre_pullup_plateau_polls', 0), 0, 6))"
            if needle not in source:
                raise RuntimeError("V96 pre-pullup plateau gate source signature not found")
            source = source.replace(needle, replacement, 1)
            namespace = self.d56.__dict__
            exec(compile(source, str(getattr(self.d56, "__file__", "60-15.py")), "exec"), namespace, namespace)
            patched = getattr(self.d56, "_63_execute_step120_motion_from_d56_plan", None)
            if not callable(patched):
                raise RuntimeError("V96 D60 executor recompilation failed")
            patched._bottom_vla_v96_fast_prelift = True
            print("[D60-V96-FAST-PRELIFT] post-lift plateau dwell removed; first meaningful feedback continues directly to pull-up")

        def _v96_extend_pullup_motion(self, motion, arms):
            out = copy.deepcopy(motion)
            if not isinstance(out, dict) or not isinstance(out.get("robot"), dict):
                return out
            coord = _V80PullupRecenterCoordinator(self, arms)
            extra = float(getattr(self, "_v96_waist_pullup_extra_mm", 50.0))
            for key in ("arm2", "arm1"):
                robot = out["robot"].get(key, {})
                final_map = out.get("final_board", {}) if isinstance(out.get("final_board"), dict) else {}
                final_board = final_map.get(key)
                if final_board is None:
                    q = np.asarray(robot.get("pullup"), np.float64).reshape(3)
                    board = coord._arm_to_board(key, q[:2])
                else:
                    board = np.asarray(final_board, np.float64).reshape(2)
                board_new = board + np.asarray([0.0, extra], np.float64)
                arm_xy = coord._board_to_arm(key, board_new)
                if isinstance(final_map, dict):
                    final_map[key] = [float(board_new[0]), float(board_new[1])]
                for stage in ("pullup", "place", "release_lift", "clear"):
                    if stage in robot:
                        q = np.asarray(robot[stage], np.float64).reshape(3).copy()
                        q[0] = float(arm_xy[0])
                        q[1] = float(arm_xy[1])
                        robot[stage] = q
            if isinstance(out.get("final_board"), dict):
                vals = [np.asarray(out["final_board"][k], np.float64).reshape(2)[1] for k in ("arm2", "arm1") if k in out["final_board"]]
                if vals:
                    out["final_median_y_mm"] = float(np.median(vals))
            print(f"[D60-V96-PULLUP-EXTEND] pull-up destination moved +{extra:.1f}mm toward board top; release/clear XY shifted with it")
            return out

        def _v80_install_d60_pullup_recenter(self):
            original = getattr(self.d56, "_63_execute_step120_motion_from_d56_plan", None)
            if not callable(original):
                raise RuntimeError("V80 cannot install D60 pull-up recenter: executor missing")
            if bool(getattr(original, "_bottom_vla_v80_pullup_recenter", False)):
                return
            self._v80_d60_executor_original = original

            def wrapped(plan, motion, arms, config, cfg, args, on_verified_start=None):
                if str(getattr(self.args, "mode", "")) == "physical" and isinstance(arms, dict) and all(k in arms for k in ("arm2", "arm1")):
                    motion_v96 = self._v96_extend_pullup_motion(motion, arms)
                    coord = _V80PullupRecenterCoordinator(self, arms)
                    proxied = dict(arms)
                    proxied["arm2"] = _V80PullupArmProxy(arms["arm2"], "arm2", coord)
                    proxied["arm1"] = _V80PullupArmProxy(arms["arm1"], "arm1", coord)
                    return original(plan, motion_v96, proxied, config, cfg, args, on_verified_start=on_verified_start)
                return original(plan, motion, arms, config, cfg, args, on_verified_start=on_verified_start)

            wrapped._bottom_vla_v80_pullup_recenter = True
            self.d56._63_execute_step120_motion_from_d56_plan = wrapped

        def _v80_finalize_basket_motion_only(self, success, report, detail):
            self._v53_align_initial_dual_done = False
            self.align_phase = "DUAL_WAIST_TOP"
            self.align_correction_count = 0
            self._v51_post_basket_position_pending = bool(success)
            with self.state_lock:
                self.motion_busy = False
                self.locked = None
                self.semantic_selected = None
                self.pending_semantic = None
                self.last_executed_semantic = None
                self.selected_action = None
            if success:
                self.auto_recommended = "REJUDGE"
                self._v78_initial_spread_busy = True
                self._v78_initial_spread_done = False
                with self.state_lock:
                    self.status = "BASKET COMPLETE - INITIAL SPREAD PREPARING"
                print("[BASKET-V80-DONE] motion-only BASKET complete; no observation/transition saved; starting INITIAL_SPREAD")
                threading.Thread(target=self._v78_run_initial_spread_after_basket, name="bottom-vla-initial-spread-v80", daemon=True).start()
            else:
                with self.state_lock:
                    self.status = f"BASKET FAILED: {detail}"
                print(f"[BASKET-V80-FAIL] {detail}")

        def _v80_start_basket_motion_only(self):
            with self.state_lock:
                if self.motion_busy or self.inference_busy or self.recorder.review_pending():
                    self.status = "BASKET BLOCKED: MOTION / INFERENCE / REVIEW ACTIVE"
                    print("[BASKET-V80-BLOCK] active motion/inference/review")
                    return
            if not self.empty_baseline_ready:
                with self.state_lock:
                    self.status = "BASKET BLOCKED: EMPTY BOARD E REQUIRED"
                return
            try:
                self._verify_sources_unchanged()
            except Exception as exc:
                with self.state_lock:
                    self.status = f"BASKET BLOCKED: {exc}"
                return
            if not self._ensure_camera_clear("BASKET_V80_MOTION_ONLY_START", allow_move=False):
                with self.state_lock:
                    self.status = "BASKET BLOCKED: CAMERA-CLEAR NOT VERIFIED"
                return
            try:
                start_fb = self._basket_feedback(quiet=True) if self.args.mode != "dry-run" else None
                plan = self._basket_build_plan(start_fb)
            except Exception as exc:
                with self.state_lock:
                    self.status = f"BASKET PLAN ERROR: {exc}"
                print(f"[BASKET-V80-PLAN-ERROR] {exc!r}")
                return
            self._v88_mark_motion_started("BASKET_GRASP")
            with self.state_lock:
                self.motion_busy = True
                self.locked = None
                self.semantic_selected = "BASKET_GRASP"
                self.pending_semantic = "BASKET_GRASP"
                self.last_executed_semantic = "BASKET_GRASP"
                self.status = "BASKET MOTION RUNNING - NO INFERENCE / NO ENTER"
            print("[BASKET-V80-MOTION-ONLY] button accepted -> execute fixed Basket motion immediately; no camera inference or dataset image")

            def worker():
                success = False
                detail = "NOT_STARTED"
                release_verified = False
                report = {"motion_only": True, "grasp_success": False, "standby_reached": False}
                try:
                    if self.args.mode == "dry-run":
                        report.update({"grasp_success": True, "standby_reached": True})
                        success = True
                        detail = "DRY_RUN"
                    else:
                        self._basket_torque_on()
                        self._basket_execute_hover(plan)
                        contact = self._basket_execute_contact_descent(plan)
                        report["contact"] = str((contact or {}).get("reason", ""))
                        wider_fb = self._basket_verify_gripper(
                            float(plan["post_contact_open_angle"]), float(self.front.basket_close_tolerance_rad), 4,
                            float(self.front.basket_post_contact_open_settle_s), "BASKET-POST-CONTACT-OPEN"
                        )
                        report["post_contact_open_angle"] = float(wider_fb["t"])
                        close_fb = self._basket_verify_gripper(
                            float(plan["close_target"]), float(self.front.basket_close_tolerance_rad),
                            int(self.front.basket_close_attempts), float(self.front.basket_gripper_settle_s), "BASKET-CLOSE-VERIFY"
                        )
                        report["close_achieved_angle"] = float(close_fb["t"])
                        final_latch = self._basket_final_grip_latch(plan)
                        report["final_close_feedback_angle"] = final_latch.get("feedback_angle")
                        lift_result = self._basket_lift_adaptive(
                            float(close_fb["x"]), float(close_fb["y"]), float(close_fb["z"]),
                            float(plan["pickup_lift_z"]), float(plan["close_target"])
                        )
                        lift_fb = lift_result["feedback"]
                        if not bool(lift_result.get("reached")) and not bool(lift_result.get("stalled")):
                            raise RuntimeError("basket lift neither reached target nor confirmed saturation")
                        lift_pose = np.asarray([float(lift_fb["x"]), float(lift_fb["y"]), float(lift_fb["z"])], np.float64)
                        retention = self._basket_retention()
                        report["retention"] = str((retention or {}).get("reason", ""))
                        if not bool((retention or {}).get("success", False)):
                            rel = self._basket_release_verified(float(self.front.basket_grip_fully_open), "BASKET-RETENTION-FAIL-RELEASE")
                            release_verified = True
                            report["release_achieved_angle"] = float(rel["t"])
                            self._basket_return_arm2_standby(True)
                            report["standby_reached"] = True
                            raise RuntimeError(f"basket grasp retention failed: {(retention or {}).get('reason')}")
                        report["grasp_success"] = True
                        center_x, center_y = map(float, plan["board_center_xy"])
                        transit_z = float(lift_pose[2])
                        self._basket_move_goal(center_x, center_y, transit_z, float(plan["close_target"]), self.front.basket_board_transit_speed, "BASKET-BOARD-CENTER")
                        placement_x, placement_y = map(float, plan.get("placement_xy", plan["board_center_xy"]))
                        placement_deg = float(plan.get("placement_extra_deg_signed", 0.0))
                        placement_distance = float(np.linalg.norm(np.asarray([placement_x-center_x, placement_y-center_y], np.float64)))
                        if placement_distance > 1.0:
                            center_pose, center_error = self._basket_wait_transit_near(
                                "BASKET-BOARD-CENTER-BLEND", (center_x, center_y, transit_z), float(self.front.basket_board_center_blend_mm)
                            )
                            stage = f"BASKET-PLACEMENT-ROTATE-{abs(placement_deg):.0f}"
                            print(f"[BASKET-CENTER-BLEND] center_error={center_error:.1f}mm -> continuous final target {placement_deg:+.1f}deg")
                            self._basket_move_goal(placement_x, placement_y, transit_z, float(plan["close_target"]), self.front.basket_placement_rotate_speed, stage)
                            placement_pose = self._basket_wait_waypoint(stage, (placement_x, placement_y, transit_z), xy_only=True)
                        else:
                            placement_pose = self._basket_wait_waypoint("BASKET-BOARD-CENTER", (center_x, center_y, transit_z), xy_only=True)
                        lower_x, lower_y, lower_start_z = map(float, placement_pose[:3])
                        lower_target_z = max(float(self.front.basket_placement_min_z), lower_start_z - max(0.0, float(self.front.basket_placement_lower_mm)))
                        if lower_target_z < lower_start_z - 1.0:
                            self._basket_move_goal(lower_x, lower_y, lower_target_z, float(plan["close_target"]), float(self.front.basket_placement_lower_speed), "BASKET-PLACE-SLOW-DESCENT-CLOSED")
                            self._basket_wait_waypoint("BASKET-PLACE-SLOW-DESCENT-CLOSED", (lower_x, lower_y, lower_target_z))
                        release_fb = self._basket_release_verified(float(plan["release_target"]), "BASKET-RELEASE-VERIFY")
                        release_verified = True
                        report["release_achieved_angle"] = float(release_fb["t"])
                        standby_pose = self._basket_return_arm2_standby(True)
                        report["standby_reached"] = True
                        report["standby_actual"] = np.asarray(standby_pose, np.float64).astype(float).tolist()
                        success = True
                        detail = "BASKET_GRASP_ARM2_ONLY_COMPLETE"
                except Exception as exc:
                    detail = repr(exc)
                    print(f"[BASKET-V80-ERROR] {exc!r}")
                    if self.args.mode != "dry-run" and not release_verified:
                        try:
                            fb = self._basket_release_verified(float(self.front.basket_grip_fully_open), "BASKET-FAILSAFE-RELEASE")
                            release_verified = True
                            report["failsafe_release_angle"] = float(fb["t"])
                        except Exception as release_exc:
                            report["failsafe_release_error"] = repr(release_exc)
                    if self.args.mode != "dry-run" and release_verified and not report.get("standby_reached"):
                        try:
                            standby_pose = self._basket_return_arm2_standby(True)
                            report["standby_reached"] = True
                            report["standby_actual"] = np.asarray(standby_pose, np.float64).astype(float).tolist()
                        except Exception as standby_exc:
                            report["standby_error"] = repr(standby_exc)
                finally:
                    self._v80_finalize_basket_motion_only(bool(success), report, detail)

            self.worker = threading.Thread(target=worker, name="bottom-vla-basket-v80-motion-only", daemon=True)
            self.worker.start()

        def _v87_cache_auto_observation(self, bundle, obs, semantic, reason):
            self._v87_finish_cache = {
                "bundle": bundle,
                "obs": obs,
                "semantic": None if semantic is None else str(semantic),
                "reason": str(reason),
                "monotonic": time.monotonic(),
                "motion_epoch": int(getattr(self, "_v88_motion_epoch", 0)),
            }

        def _v87_finish_cache_valid(self, allow_inference_busy=False):
            cache = self._v87_finish_cache
            if not isinstance(cache, dict):
                return False, "NO_CACHE"
            age = time.monotonic() - float(cache.get("monotonic", -1e9))
            if age < 0.0:
                return False, f"CLOCK_ERROR_{age:.2f}s"
            if cache.get("bundle") is None or cache.get("obs") is None:
                return False, "CACHE_PAYLOAD_MISSING"
            cache_epoch = int(cache.get("motion_epoch", -1))
            current_epoch = int(getattr(self, "_v88_motion_epoch", 0))
            if cache_epoch != current_epoch:
                return False, f"MOTION_CHANGED cache={cache_epoch} current={current_epoch}"
            with self.state_lock:
                if bool(self.motion_busy):
                    return False, "MOTION_BUSY"
                if bool(self.inference_busy) and not bool(allow_inference_busy):
                    return False, "INFERENCE_BUSY"
            if self.recorder.review_pending():
                return False, "REVIEW_PENDING"
            return True, f"OK no-motion age={age:.2f}s epoch={current_epoch}"

        def _v87_fast_finish_from_cache(self, origin):
            ok, why = self._v87_finish_cache_valid()
            if not ok:
                print(f"[FINISH-V87-CACHE-MISS] {why}; normal fresh FINISH inference remains available")
                return False
            cache = self._v87_finish_cache
            self._v74_finish_processing = True
            self._v74_finish_complete_banner = False
            self._v76_finish_commit_queued = False
            self.semantic_selected = "FINISH"
            self.pending_semantic = "FINISH"
            self.selected_action = "FINISH"
            self._human_selected = "FINISH"
            self._prepare_origin = str(origin).upper()
            with self.state_lock:
                self.status = "FINISH FAST SAVE... USING LATEST VERIFIED OBSERVATION"
            try:
                self._verify_sources_unchanged()
                self._prepare_finish_from_existing(cache["bundle"], cache["obs"], None)
            except Exception as exc:
                self._v74_finish_processing = False
                self._v87_finish_requested = False
                print(f"[FINISH-V87-CACHE-ERROR] {exc!r}; fall back to fresh FINISH inference")
                return False
            self._v76_finish_commit_queued = True
            self._v87_finish_requested = False
            age = time.monotonic() - float(cache.get("monotonic", time.monotonic()))
            print(
                f"[FINISH-V87-FAST] reused auto-judge observation age={age:.2f}s "
                f"previousAuto={cache.get('semantic')} reason={cache.get('reason')}"
            )
            self._queue("FINISH_COMMIT")
            return True

        def _v98_freeze_human_finish_snapshot(self):
            try:
                with self.live_frame_lock:
                    raw = None if getattr(self, "latest_live_raw", None) is None else self.latest_live_raw.copy()
                    corrected = None if getattr(self, "latest_live_corrected", None) is None else self.latest_live_corrected.copy()
                    mono = float(getattr(self, "latest_live_monotonic", time.monotonic()))
                if raw is None or corrected is None:
                    return False
                with self._v98_human_finish_snapshot_lock:
                    self._v98_human_finish_snapshot = {"raw": raw, "corrected": corrected, "monotonic": mono}
                print(f"[FINISH-V98-FREEZE] HUMAN FINISH exact live frame frozen age={max(0.0,time.monotonic()-mono):.3f}s")
                return True
            except Exception as exc:
                print(f"[FINISH-V98-FREEZE-WARN] {type(exc).__name__}: {exc}")
                return False

        def _v100_finish_obs_with_mask_axis(self, obs):
            mask_info = self._v30_mask_info(SimpleNamespace(observation=obs))
            if (not bool(mask_info.get("valid", False))) or bool(mask_info.get("fallback", False)):
                try:
                    out = copy.copy(obs)
                except Exception:
                    out = obs
                reason = "FALLBACK_MASK_GEOMETRY" if bool(mask_info.get("fallback", False)) else "NO_TRUSTED_MASK"
                try:
                    setattr(out, "bottom_vla_v105_finish_state_invalid", True)
                    setattr(out, "bottom_vla_v105_finish_state_invalid_reason", reason)
                    setattr(out, "bottom_vla_v105_finish_mask_source", str(mask_info.get("source", "NONE")))
                    setattr(out, "bottom_vla_v105_finish_axis_source", "POSE_UNMODIFIED")
                except Exception:
                    pass
                print(
                    f"[FINISH-V105-MASK-AXIS-SKIP] reason={reason} "
                    f"source={str(mask_info.get('source','NONE'))[:160]}; "
                    "Pose axis kept and classifier sample marked ineligible"
                )
                return out
            geom = self._v100_mask_centerline_geometry(obs, None)
            if not isinstance(geom, dict):
                return obs
            waist = self._finite_point2(geom.get("waist_center"))
            lower = self._finite_point2(geom.get("lower_point"))
            if waist is None or lower is None:
                return obs
            pose = getattr(obs, "pose", None)
            if pose is None:
                return obs
            try:
                out = copy.copy(obs)
                pose2 = copy.copy(pose)
                out.pose = pose2
                waist32 = np.asarray(waist, np.float32)
                lower32 = np.asarray(lower, np.float32)
                pose2.waist_center = waist32.copy()
                pose2.crotch = lower32.copy()
                pose2.pose_center = (0.5 * (waist32 + lower32)).astype(np.float32)
                axis = lower32 - waist32
                if hasattr(pose2, "pants_axis_angle_deg"):
                    pose2.pants_axis_angle_deg = float(math.degrees(math.atan2(float(axis[1]), float(axis[0]))))
                if hasattr(pose2, "pants_axis_len_mm"):
                    pose2.pants_axis_len_mm = float(np.linalg.norm(axis))
                board_dict = dict(getattr(pose2, "keypoints_board", {}) or {})
                board_dict["waist_center"] = waist32.copy()
                board_dict["crotch"] = lower32.copy()
                pose2.keypoints_board = board_dict
                px_dict = dict(getattr(pose2, "keypoints_px", {}) or {})
                for name, point in (("waist_center", waist32), ("crotch", lower32)):
                    try:
                        px = self.d56.board_to_pixel(self.H_raw, float(point[0]), float(point[1]))
                        if px is not None:
                            px_dict[name] = np.asarray(px, np.float32)
                    except Exception:
                        pass
                if px_dict:
                    pose2.keypoints_px = px_dict
                setattr(out, "bottom_vla_v100_axis_source", "MASK_WIDTH_CENTERLINE")
                setattr(out, "bottom_vla_v105_finish_state_invalid", False)
                setattr(out, "bottom_vla_v105_finish_mask_source", str(mask_info.get("source", "MASK")))
                setattr(out, "bottom_vla_v105_finish_axis_source", "MASK_WIDTH_CENTERLINE")
                print(
                    f"[FINISH-V100-MASK-AXIS] bins={int(geom.get('mask_centerline_bins',0))} "
                    f"rms={float(geom.get('mask_centerline_fit_rms_mm',0.0)):.1f}mm "
                    f"waist=({waist32[0]:.1f},{waist32[1]:.1f}) lower=({lower32[0]:.1f},{lower32[1]:.1f})"
                )
                return out
            except Exception as exc:
                print(f"[FINISH-V100-MASK-AXIS-WARN] {type(exc).__name__}: {exc}")
                return obs

        def _append_finish_classifier_sample(self, locked, finish_json):
            obs = getattr(locked, "observation", None)
            mask_info = self._v30_mask_info(locked)
            marker_invalid = bool(getattr(obs, "bottom_vla_v105_finish_state_invalid", False)) if obs is not None else False
            mask_invalid = (not bool(mask_info.get("valid", False))) or bool(mask_info.get("fallback", False))
            invalid = bool(marker_invalid or mask_invalid)
            if not invalid:
                return super()._append_finish_classifier_sample(locked, finish_json)
            if marker_invalid:
                reason = str(getattr(obs, "bottom_vla_v105_finish_state_invalid_reason", "FINISH_STATE_QUALITY_INVALID"))
            else:
                reason = "FALLBACK_MASK_GEOMETRY" if bool(mask_info.get("fallback", False)) else "NO_TRUSTED_MASK"
            mask_source = str(getattr(obs, "bottom_vla_v105_finish_mask_source", mask_info.get("source", "NONE"))) if obs is not None else str(mask_info.get("source", "NONE"))
            try:
                path = Path(finish_json)
                with open(path, "r", encoding="utf-8") as f:
                    payload = json.load(f)
                if isinstance(payload, dict):
                    if "training_eligible" in payload:
                        payload["training_eligible"] = False
                    sample = payload.get("classifier_sample")
                    if not isinstance(sample, dict):
                        sample = {}
                        payload["classifier_sample"] = sample
                    sample["action"] = "FINISH"
                    sample["state_schema"] = "bottom_state_v2"
                    sample["training_eligible"] = False
                    sample["planner_diagnostics_are_input"] = False
                    payload["finish_state_quality"] = {
                        "valid_for_classifier": False,
                        "reason": reason,
                        "mask_source": mask_source,
                        "axis_source": str(getattr(obs, "bottom_vla_v105_finish_axis_source", "POSE_UNMODIFIED")),
                    }
                    v23._atomic_json(path, payload)
            except Exception as exc:
                print(f"[FINISH-V105-QUALITY-WARN] failed to patch finish JSON: {type(exc).__name__}: {exc}")
            print(
                f"[BOTTOM-CLASSIFIER-SAMPLE] action=FINISH keep=True eligible=False "
                f"reason={reason}; manifest append skipped"
            )
            return None

        def _v100_queue_next_after_finish(self, semantic, origin="HUMAN"):
            semantic_u = str(semantic).upper()
            if semantic_u in {"", "FINISH", "REJUDGE"}:
                return False
            with self._v100_finish_queue_lock:
                self._v100_finish_queued_action = semantic_u
                self._v100_finish_queued_origin = str(origin).upper()
            with self.state_lock:
                self.status = f"FINISH SAVING... NEXT ACTION QUEUED: {semantic_u}"
            print(f"[FINISH-V100-QUEUE-NEXT] semantic={semantic_u} origin={str(origin).upper()}")
            return True

        def _prepare_finish(self):
            with self._v98_human_finish_snapshot_lock:
                snap = self._v98_human_finish_snapshot
                self._v98_human_finish_snapshot = None
            if isinstance(snap, dict) and snap.get("raw") is not None and snap.get("corrected") is not None:
                self._verify_sources_unchanged()
                bundle = self._capture_i_frame_from_live("D56_WAIST_LIFT_LAYDOWN")
                bundle.raw = np.asarray(snap["raw"]).copy()
                bundle.corrected = np.asarray(snap["corrected"]).copy()
                obs = self._infer_for_action("D56_WAIST_LIFT_LAYDOWN", bundle.corrected)
                obs = self._v100_finish_obs_with_mask_axis(obs)
                print(
                    f"[FINISH-V98-PRESS-FRAME] preparing FINISH from exact human button-press frame "
                    f"frozenAge={max(0.0,time.monotonic()-float(snap.get('monotonic',time.monotonic()))):.3f}s"
                )
                self._prepare_finish_from_existing(bundle, obs, None)
                return
            ok, why = self._v87_finish_cache_valid(allow_inference_busy=True)
            if ok:
                cache = self._v87_finish_cache
                age = time.monotonic() - float(cache.get("monotonic", time.monotonic()))
                print(
                    f"[FINISH-V87-PREP-CACHE] reuse latest auto-judge observation age={age:.2f}s "
                    "inside FINISH prepare worker; duplicate Seg/Pose inference skipped"
                )
                self._verify_sources_unchanged()
                self._prepare_finish_from_existing(cache["bundle"], cache["obs"], None)
                return
            print(f"[FINISH-V87-PREP-FRESH] cache unavailable ({why}); run normal fresh FINISH inference")
            return super()._prepare_finish()

        def _v87_wait_for_autojudge_finish_then_retry(self):
            if self._v87_finish_wait_queued:
                return
            self._v87_finish_wait_queued = True

            def worker():
                try:
                    deadline = time.monotonic() + 6.0
                    while time.monotonic() < deadline:
                        with self._auto_judge_lock:
                            busy = bool(self._auto_judge_busy)
                        if not busy:
                            break
                        time.sleep(0.03)
                finally:
                    self._v87_finish_wait_queued = False
                self._queue("ACTION:FINISH")

            threading.Thread(target=worker, name="bottom-vla-v87-finish-wait-autojudge", daemon=True).start()

        def _select_semantic(self, semantic, origin="HUMAN"):
            semantic_u = str(semantic).upper()
            origin_u = str(origin).upper()
            if (
                self._integration_mode
                and bool(getattr(self, "_v116_finish_terminal_latched", False))
                and semantic_u != "FINISH"
            ):
                print(f"[FINISH-V116-TERMINAL-BLOCK] semantic={semantic_u} origin={origin_u} blocked after model FINISH")
                return None
            if bool(getattr(self, "_v78_initial_spread_busy", False)):
                with self.state_lock:
                    self.status = "INITIAL SPREAD RUNNING... WAIT"
                print(f"[INITIAL-SPREAD-V78-INPUT-BLOCK] semantic={semantic_u}")
                return None
            if semantic_u == "REJUDGE":
                return self._rejudge()
            if semantic_u == "OUTER_PULL":
                with self.state_lock:
                    self.status = "OUTER PULL DISABLED IN V74"
                print("[OUTER-V74-DISABLED] OUTER_PULL is removed from runtime selection")
                return None
            if bool(getattr(self, "_v74_finish_processing", False)):
                if semantic_u != "FINISH" and self._v100_queue_next_after_finish(semantic_u, origin_u):
                    return None
                with self.state_lock:
                    self.status = "FINISH PROCESSING... WAIT FOR SUCCESS MESSAGE"
                print(f"[FINISH-V76-INPUT-BLOCK] blocked semantic={semantic_u} while FINISH is processing")
                return None
            if semantic_u == "BASKET_GRASP":
                with self.state_lock:
                    self.status = "BASKET PREPARING / RECOGNIZING..."
                print("[BASKET-V77-PREPARING] action accepted; preparing snapshot/inference before motion")
            if semantic_u == "FINISH":
                self._v87_finish_requested = True
                if origin_u == "HUMAN":
                    self._v87_finish_cache = None
                    self._v98_freeze_human_finish_snapshot()
                    self._v87_finish_requested = False
                    self._v74_finish_processing = True
                    self._v74_finish_complete_banner = False
                    self._v76_finish_commit_queued = False
                    with self.state_lock:
                        self.status = "FINISH SNAPSHOT CAPTURED - SELECT NEXT ACTION NOW"
                    print("[FINISH-V100-HUMAN] exact button-press frame frozen; FINISH saves in background and next action may be queued immediately")
                else:
                    if self._v87_fast_finish_from_cache(origin_u):
                        return None
                    with self._auto_judge_lock:
                        auto_busy = bool(self._auto_judge_busy)
                    if auto_busy:
                        with self.state_lock:
                            self.status = "FINISH: WAITING ONLY FOR CURRENT NO-MOTION SNAPSHOT..."
                        print("[FINISH-V88-WAIT-AUTO] no same-motion cache yet; wait for the already-running AUTO-JUDGE only, never start duplicate inference")
                        self._v87_wait_for_autojudge_finish_then_retry()
                        return None
                    self._v87_finish_requested = False
                    self._v74_finish_processing = True
                    self._v74_finish_complete_banner = False
                    self._v76_finish_commit_queued = False
                    with self.state_lock:
                        self.status = "FINISH PROCESSING... DO NOT PRESS ENTER / WAIT"
                    print("[FINISH-V87-FALLBACK] no reusable current observation -> one fresh FINISH inference; auto-commit terminal save")
            semantic_map = {
                "D55_PRESS_SWEEP": "PRESS_SWEEP",
                "D58_CIRC_POSITION": "POSITION_ADJUST",
                "D54_OUTER_PULL": "OUTER_PULL",
                "D56_WAIST_LIFT_LAYDOWN": "WAIST_PULL_LAYDOWN",
                "BASKET_GRASP": "BASKET_GRASP",
                "ALIGN": "ALIGN",
                "FINISH": "FINISH",
            }
            with self.state_lock:
                busy = bool(self.inference_busy)
                raw_current = (
                    self.pending_semantic
                    or getattr(self, "semantic_selected", None)
                    or getattr(self, "inference_action", None)
                    or "INFERENCE"
                )
                raw_current_u = str(raw_current).upper()
                current = semantic_map.get(raw_current_u, raw_current_u)
            if origin_u == "HUMAN" and busy and semantic_u != current:
                print(
                    f"[HUMAN-INTERRUPT-V68-CANCEL] current={current} -> {semantic_u}; "
                    "mark current generation stale now; current TensorRT call may finish, but remaining local planning will abort"
                )
            return super()._select_semantic(semantic, origin=origin)

        def _v74_safe_post_finish_standby(self):
            if self._feedback_camera_clear():
                print("[POST-FINISH-V74-STANDBY] already verified; no move needed")
                return True
            if str(getattr(self.args, "mode", "physical")) != "physical":
                return True
            with self.state_lock:
                self.status = "POST-FINISH: RAISING ARMS SAFELY -> STANDBY"
            move_command = int(getattr(self.args, "move_command", 104))
            safe_z = float(self._v74_post_finish_safe_z_mm)
            feedback = {}
            current_t = {}
            for key in ("arm2", "arm1"):
                arm = self.arms.get(key)
                if arm is None:
                    print(f"[POST-FINISH-V74-STANDBY] {key.upper()} missing")
                    return False
                try:
                    fb = self.align._d31_query_feedback(arm, timeout_s=0.9)
                    xyz = self.align._d31_feedback_xyz(fb)
                    if xyz is None:
                        print(f"[POST-FINISH-V74-STANDBY] {key.upper()} feedback unavailable")
                        return False
                    feedback[key] = np.asarray(xyz, np.float64).reshape(3)
                    current_t[key] = float(fb.get("t", getattr(self.align_cfg, "tool_t_rad", 3.14))) if isinstance(fb, dict) else float(getattr(self.align_cfg, "tool_t_rad", 3.14))
                except Exception as exc:
                    print(f"[POST-FINISH-V74-STANDBY] {key.upper()} feedback error={exc!r}")
                    return False
            threads = []
            errors = []
            def raise_one(key):
                try:
                    xyz = feedback[key]
                    target_z = max(float(xyz[2]), safe_z)
                    self.arms[key].move_goal(
                        float(xyz[0]), float(xyz[1]), target_z, float(current_t[key]), float(self._v74_post_finish_raise_speed),
                        move_command=move_command, stage="POST_FINISH_V74_PURE_Z_SAFE_RAISE",
                        caller="_v74_safe_post_finish_standby",
                    )
                except Exception as exc:
                    errors.append((key, repr(exc)))
            for key in ("arm2", "arm1"):
                t = threading.Thread(target=raise_one, args=(key,), daemon=True)
                threads.append(t)
                t.start()
            for t in threads:
                t.join(timeout=3.0)
            if errors:
                print(f"[POST-FINISH-V74-STANDBY] pure-Z raise errors={errors}")
                return False
            time.sleep(1.0)
            moved = self.d56.move_arms_to_standby(
                self.arms, self.cfg56, move_command=move_command,
                reason="BOTTOM_VLA_POST_FINISH_NEXT_ACTION",
            )
            if not moved:
                print("[POST-FINISH-V74-STANDBY] standby command failed")
                return False
            attempts = max(2, int(getattr(self.args, "standby_verify_retries", 2)))
            time.sleep(float(self._v84_standby_first_poll_s))
            for attempt in range(1, attempts + 1):
                if self._feedback_camera_clear():
                    print(f"[POST-FINISH-V84-STANDBY] VERIFIED attempt={attempt}/{attempts}")
                    return True
                if attempt < attempts:
                    time.sleep(float(self._v84_standby_retry_poll_s))
            print("[POST-FINISH-V74-STANDBY] verification failed; next action blocked")
            return False

        def _v102_suspend_finish_stats_refresh(self):
            rec = getattr(self, "recorder", None)
            if rec is None:
                return None
            name = "refresh_dataset_stats"
            fn = getattr(rec, name, None)
            if not callable(fn):
                print("[FINISH-V102-STATS] refresh_dataset_stats not found; no defer hook installed")
                return None
            had_instance = name in getattr(rec, "__dict__", {})
            old_instance = rec.__dict__.get(name) if had_instance else None
            state = {"rec": rec, "name": name, "fn": fn, "had_instance": had_instance, "old_instance": old_instance}
            def fast_refresh(*args, **kwargs):
                print("[FINISH-V102-STATS-DEFER] synchronous full dataset rescan skipped; cached counters kept for immediate next action")
                for attr in ("dataset_stats", "stats", "persistent_stats", "_dataset_stats"):
                    value = getattr(rec, attr, None)
                    if value is not None:
                        return value
                return None
            setattr(rec, name, fast_refresh)
            return state

        def _v102_restore_finish_stats_refresh(self, state):
            if not isinstance(state, dict):
                return
            rec = state.get("rec")
            name = state.get("name")
            if rec is None or not name:
                return
            try:
                if bool(state.get("had_instance", False)):
                    setattr(rec, name, state.get("old_instance"))
                else:
                    rec.__dict__.pop(name, None)
            except Exception as exc:
                print(f"[FINISH-V102-STATS-RESTORE-WARN] {type(exc).__name__}: {exc}")

        def _v102_schedule_finish_stats_refresh(self, state):
            if not isinstance(state, dict) or not callable(state.get("fn")):
                return
            fn = state["fn"]
            def worker():
                deadline = time.monotonic() + 45.0
                while time.monotonic() < deadline:
                    with self.state_lock:
                        busy = bool(self.inference_busy or self.motion_busy)
                    if not busy:
                        break
                    time.sleep(0.25)
                try:
                    fn()
                    print("[FINISH-V102-STATS-BG] deferred dataset stats refresh complete")
                except Exception as exc:
                    print(f"[FINISH-V102-STATS-BG-WARN] {type(exc).__name__}: {exc}")
            threading.Thread(target=worker, name="bottom-vla-v102-stats-refresh", daemon=True).start()

        def _execute_finish(self, locked):
            if bool(getattr(self, "_v110_inference_only", False)):
                self._v74_finish_processing = False
                self._v76_finish_commit_queued = False
                self._v87_finish_requested = False
                self._v87_finish_cache = None
                self._v74_finish_complete_banner = True
                with self.state_lock:
                    self.status = "FINISH SUCCESS - INFERENCE-ONLY / NO DATASET SAVE"
                    self.locked = None
                print("[FINISH-V110-INFERENCE-ONLY] terminal state accepted; dataset/episode save bypassed")
                return {"ok": True, "inference_only": True}
            self._v74_finish_processing = True
            self._v74_finish_complete_banner = False
            success = False
            result = None
            stats_state = self._v102_suspend_finish_stats_refresh()
            with self.state_lock:
                self.status = "FINISH PROCESSING... SAVING / CLOSING EPISODE"
            try:
                result = super()._execute_finish(locked)
                success = True
                return result
            except Exception:
                with self.state_lock:
                    self.status = "FINISH ERROR - RETRY FINISH"
                raise
            finally:
                self._v102_restore_finish_stats_refresh(stats_state)
                self._v74_finish_processing = False
                self._v76_finish_commit_queued = False
                self._v87_finish_requested = False
                self._v87_finish_cache = None
                if success:
                    self._v74_finish_complete_banner = True
                    self._v74_post_finish_standby_pending = False
                    with self._v100_finish_queue_lock:
                        queued = self._v100_finish_queued_action
                        queued_origin = self._v100_finish_queued_origin
                        self._v100_finish_queued_action = None
                        self._v100_finish_queued_origin = "HUMAN"
                    with self.state_lock:
                        self.status = (
                            f"FINISH SUCCESS - STARTING {queued}" if queued
                            else "FINISH SUCCESS / NEXT EPISODE READY - SELECT NEXT ACTION"
                        )
                    if queued:
                        print(f"[FINISH-V102-NEXT-START] terminal files closed; start queued {queued} before deferred dataset rescan")
                        threading.Timer(0.02, lambda q=str(queued): self._queue(f"ACTION:{q}")).start()
                    else:
                        print("[FINISH-V102-COMPLETE] SUCCESS saved; next action enabled immediately")
                    self._v102_schedule_finish_stats_refresh(stats_state)
                else:
                    self._v74_finish_complete_banner = False
                    with self._v100_finish_queue_lock:
                        self._v100_finish_queued_action = None
                        self._v100_finish_queued_origin = "HUMAN"
                    print("[FINISH-V76-UNLOCK] FINISH failed; processing lock released in finally")

        def _v110_schedule_next_after_motion(self, semantic, motion_epoch):
            if self._integration_stop_requested():
                return
            semantic = str(semantic or "ACTION").upper()
            if semantic == "FINISH":
                return
            motion_epoch = int(motion_epoch)
            with self.state_lock:
                if motion_epoch <= int(getattr(self, "_v110_next_scheduled_epoch", -1)):
                    print(f"[INFERENCE-ONLY-V110-NEXT] duplicate schedule ignored epoch={motion_epoch} semantic={semantic}")
                    return
                self._v110_next_scheduled_epoch = motion_epoch

            def worker():
                deadline = time.monotonic() + 180.0
                while time.monotonic() < deadline:
                    with self.state_lock:
                        current_epoch = int(getattr(self, "_v88_motion_epoch", 0))
                        motion_busy = bool(self.motion_busy)
                    if current_epoch != motion_epoch:
                        print(
                            f"[INFERENCE-ONLY-V110-NEXT] stale completion watcher ignored "
                            f"watchEpoch={motion_epoch} currentEpoch={current_epoch} semantic={semantic}"
                        )
                        return
                    if not motion_busy:
                        break
                    time.sleep(0.02)
                else:
                    with self.state_lock:
                        self.status = f"{semantic} MOTION TIMEOUT - MANUAL REJUDGE"
                    print(f"[INFERENCE-ONLY-V110-NEXT-WARN] timeout waiting for physical motion {semantic}")
                    return

                settle_deadline = time.monotonic() + 20.0
                while time.monotonic() < settle_deadline:
                    with self.state_lock:
                        inference_busy = bool(self.inference_busy)
                    if not inference_busy:
                        break
                    time.sleep(0.05)

                with self.state_lock:
                    if int(getattr(self, "_v88_motion_epoch", 0)) != motion_epoch:
                        return
                    if self.motion_busy:
                        return
                    self.locked = None
                    self.display_image = None
                    self.semantic_selected = None
                    self.pending_semantic = None
                    self.selected_action = None
                    self._human_selected = None
                    self.status = f"{semantic} MOTION COMPLETE - FRESH MODEL JUDGE NEXT"
                self._prepare_origin = "AUTO"
                try:
                    self._invalidate_for_new_action("V110_POST_PHYSICAL_MOTION_MODEL_JUDGE")
                except Exception as exc:
                    print(f"[INFERENCE-ONLY-V110-NEXT-WARN] invalidate failed: {exc!r}")
                if self._integration_stop_requested():
                    print(f"[INFERENCE-ONLY-V110-NEXT] stop requested; suppress fresh model judgement semantic={semantic} epoch={motion_epoch}")
                    return
                print(f"[INFERENCE-ONLY-V110-NEXT] physical motion complete semantic={semantic} epoch={motion_epoch} -> fresh model judgement")
                self._schedule_auto_judge(f"after_{semantic.lower()}", 0.10)

            threading.Thread(
                target=worker,
                name=f"bottom-vla-v110-next-{semantic.lower()}-e{motion_epoch}",
                daemon=True,
            ).start()

        def _handle(self, event):
            if bool(getattr(self, "_v110_inference_only", False)) and event in {
                "RESULT:GOOD", "RESULT:BAD", "RESULT:SKIP", "COLLECT:KEEP", "COLLECT:DISCARD"
            }:
                print(f"[INFERENCE-ONLY-V110] collection/review event ignored: {event}")
                return
            if bool(getattr(self, "_v78_initial_spread_busy", False)):
                if event == "QUIT":
                    return super()._handle(event)
                if event.startswith("ACTION:") or event in {"ENTER", "REJUDGE", "EMPTY_BASELINE"}:
                    with self.state_lock:
                        self.status = "INITIAL SPREAD RUNNING... WAIT"
                    print(f"[INITIAL-SPREAD-V78-INPUT-BLOCK] event={event}")
                    return
            if event == "FINISH_COMMIT":
                with self.state_lock:
                    infer_busy = bool(self.inference_busy)
                    locked = self.locked
                if infer_busy:
                    threading.Timer(0.08, lambda: self._queue("FINISH_COMMIT")).start()
                    return
                if locked is None or str(getattr(locked, "action", "")).upper() != "FINISH":
                    self._v74_finish_processing = False
                    self._v76_finish_commit_queued = False
                    self._v74_finish_complete_banner = False
                    with self.state_lock:
                        self.status = "FINISH COMMIT FAILED - RETRY FINISH"
                    print("[FINISH-V76-COMMIT-FAIL] terminal plan missing after inference; lock released")
                    return
                if bool(getattr(self, "_v95_finish_commit_worker_active", False)):
                    print("[FINISH-V95-COMMIT] worker already active")
                    return
                self._v95_finish_commit_worker_active = True
                print("[FINISH-V95-COMMIT] inference complete -> terminal save moved off UI thread; no user ENTER")
                def _finish_worker():
                    try:
                        Base._handle(self, "ENTER")
                    except Exception as exc:
                        print(f"[FINISH-V95-WORKER-ERROR] {type(exc).__name__}: {exc}")
                        traceback.print_exc()
                    finally:
                        self._v95_finish_commit_worker_active = False
                threading.Thread(target=_finish_worker, name="bottom-vla-v95-finish-commit", daemon=True).start()
                return
            if bool(getattr(self, "_v74_finish_processing", False)):
                if event == "QUIT":
                    return super()._handle(event)
                if event.startswith("ACTION:") and event != "ACTION:FINISH":
                    semantic = event.split(":", 1)[1]
                    if self._v100_queue_next_after_finish(semantic, "HUMAN"):
                        return
                if event.startswith("ACTION:") or event in {"ENTER", "REJUDGE", "EMPTY_BASELINE"}:
                    with self.state_lock:
                        self.status = "FINISH PROCESSING... WAIT FOR SUCCESS MESSAGE"
                    print(f"[FINISH-V76-INPUT-BLOCK] event={event}")
                    return
            if event == "ACTION:OUTER_PULL":
                with self.state_lock:
                    self.status = "OUTER PULL DISABLED IN V74"
                print("[OUTER-V74-DISABLED] ignored ACTION:OUTER_PULL")
                return
            if event.startswith("ACTION:") and event != "ACTION:FINISH" and bool(getattr(self, "_v74_finish_complete_banner", False)):
                self._v74_finish_complete_banner = False
                print(f"[FINISH-V95-BANNER-CLEAR] next action={event.split(':',1)[1]}")
            if event.startswith("ACTION:") and event != "ACTION:FINISH" and bool(getattr(self, "_v74_post_finish_standby_pending", False)):
                if not self._v74_safe_post_finish_standby():
                    with self.state_lock:
                        self.status = "POST-FINISH STANDBY FAILED - ACTION BLOCKED"
                    return
                self._v74_post_finish_standby_pending = False
                self._v74_finish_complete_banner = False
                with self.state_lock:
                    self.status = "POST-FINISH STANDBY VERIFIED - PREPARING ACTION"
            if event == "ACTION:BASKET_GRASP":
                return self._v80_start_basket_motion_only()
            if event in {"MASK_ACCURATE", "MASK_INACCURATE"}:
                print(f"[CONTROL] {event} removed in bottom_vla-28")
                return
            if event == "ENTER" and bool(getattr(self, "_v110_inference_only", False)):
                with self._v111_autoexec_lock:
                    auto_enter_token = self._v111_autoexec_token
                with self.state_lock:
                    if self.motion_busy or self.inference_busy:
                        print("[ENTER] blocked while motion/inference is active")
                        if auto_enter_token is not None:
                            print("[AUTO-ENTER-V114-RETRY] transient busy state; requeue ENTER once")
                            threading.Timer(0.05, lambda: self._queue("ENTER")).start()
                        return
                    locked_before = self.locked
                    semantic_before = str(
                        getattr(locked_before, "action", "")
                        or self.pending_semantic or self.semantic_selected or self.selected_action or "ACTION"
                    ).upper()
                    epoch_before = int(getattr(self, "_v88_motion_epoch", 0))
                try:
                    result = super()._handle(event)
                    with self.state_lock:
                        epoch_after = int(getattr(self, "_v88_motion_epoch", 0))
                    if semantic_before != "FINISH" and epoch_after > epoch_before:
                        self._v110_schedule_next_after_motion(semantic_before, epoch_after)
                        print(
                            f"[AUTO-ENTER-V114-ARMED] semantic={semantic_before} "
                            f"epoch={epoch_after} -> post-motion fresh judge armed"
                        )
                    elif semantic_before != "FINISH":
                        print(f"[INFERENCE-ONLY-V110-NEXT] ENTER did not start motion; next judge not scheduled semantic={semantic_before}")
                    return result
                finally:
                    if auto_enter_token is not None:
                        with self._v111_autoexec_lock:
                            if self._v111_autoexec_token == auto_enter_token:
                                self._v111_autoexec_token = None
            return super()._handle(event)

        def _terminal_event(self):
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
                "3": "ACTION:PRESS_SWEEP",
                "press": "ACTION:PRESS_SWEEP",
                "press_sweep": "ACTION:PRESS_SWEEP",
                "4": "ACTION:WAIST_PULL_LAYDOWN",
                "waist": "ACTION:WAIST_PULL_LAYDOWN",
                "waist_pull_laydown": "ACTION:WAIST_PULL_LAYDOWN",
                "5": "ACTION:ALIGN",
                "align": "ACTION:ALIGN",
                "6": "ACTION:FINISH",
                "finish": "ACTION:FINISH",
                "7": "REJUDGE",
                "rejudge": "REJUDGE",
                "8": "ACTION:POSITION_ADJUST",
                "position": "ACTION:POSITION_ADJUST",
                "position_adjust": "ACTION:POSITION_ADJUST",
                "l": "LOCK_H",
                "lock": "LOCK_H",
                "e": "EMPTY_BASELINE",
                "empty": "EMPTY_BASELINE",
                "enter": "ENTER",
                "": "ENTER",
                "q": "QUIT",
                "quit": "QUIT",
                "abort": "ABORT_FAILED",
                "failed": "ABORT_FAILED",
            }
            return mapping.get(line)

        def _draw_panel(self):
            w, h = 1240, 340
            panel = np.full((h, w, 3), 28, np.uint8)
            cv2 = self.cv2
            cv2.putText(panel, "BOTTOM VLA V116 AUTO", (24, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.82, (255, 255, 255), 2)
            active_tag = str(getattr(self, "_v110_active_model", "-"))
            active = "ROB30" if active_tag == "robust30" else active_tag.upper()
            preds = getattr(self, "_v110_last_predictions", {}) or {}
            pred_parts = []
            for tag in ("robust30", "v2", "v6"):
                pred = preds.get(tag)
                if isinstance(pred, dict):
                    label = "ROB30" if tag == "robust30" else tag.upper()
                    pred_parts.append(f"{label}={pred.get('action','-')} {float(pred.get('confidence',0.0)):.2f}")
            cv2.putText(panel, f"ACTIVE={active}  " + " | ".join(pred_parts), (590, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 255, 255), 1)
            rects = []

            def row(items, y, color):
                margin = 20
                gap = 8
                width = int((w - 2 * margin - gap * (len(items) - 1)) / len(items))
                for i, (label, event) in enumerate(items):
                    x0 = margin + i * (width + gap)
                    rect = (x0, y, x0 + width, y + 52)
                    fill = color
                    if event.startswith("ACTION:"):
                        value = event.split(":", 1)[1]
                        if value == self.semantic_selected:
                            fill = tuple(min(255, int(c * 1.35)) for c in color)
                    cv2.rectangle(panel, rect[:2], rect[2:], fill, -1)
                    cv2.rectangle(panel, rect[:2], rect[2:], (230, 230, 230), 1)
                    cv2.putText(panel, label, (x0 + 8, y + 33), cv2.FONT_HERSHEY_SIMPLEX, 0.47, (0, 0, 0), 2)
                    rects.append((event, rect, color))

            row([
                ("1 BASKET", "ACTION:BASKET_GRASP"),
                ("8 POSITION", "ACTION:POSITION_ADJUST"),
                ("3 PRESS SWEEP", "ACTION:PRESS_SWEEP"),
            ], 58, (100, 220, 150))
            row([
                ("4 WAIST", "ACTION:WAIST_PULL_LAYDOWN"),
                ("5 ALIGN", "ACTION:ALIGN"),
                ("6 FINISH", "ACTION:FINISH"),
                ("7 MODEL REJUDGE", "REJUDGE"),
            ], 120, (110, 185, 235))
            row([
                ("E EMPTY BOARD", "EMPTY_BASELINE"),
                ("L LOCK H", "LOCK_H"),
                ("ENTER EXECUTE", "ENTER"),
                ("Q QUIT", "QUIT"),
            ], 182, (220, 190, 95))
            row([
                ("MODEL REJUDGE", "REJUDGE"),
                ("ENTER EXECUTE", "ENTER"),
                ("Q QUIT", "QUIT"),
            ], 244, (190, 150, 220))
            self._panel_rects = rects
            return panel

        def _auto_state_dict(self, state):
            names = list(state.get("names", []))
            values = list(state.get("values", []))
            return {str(k): float(values[i]) for i, k in enumerate(names) if i < len(values)}

        def _v110_init_action_models(self):
            self._v110_inference_only = str(os.environ.get("BOTTOM_INFERENCE_ONLY", "1")).strip().lower() not in {"0", "false", "no"}
            self._v110_active_model = str(os.environ.get("BOTTOM_ACTION_MODEL", "robust30")).strip().lower()
            if self._v110_active_model not in {"robust30", "v2", "v6", "legacy"}:
                raise RuntimeError(
                    f"BOTTOM_ACTION_MODEL must be robust30/v2/v6/legacy, got {self._v110_active_model!r}"
                )

            robust_dir = Path(__file__).resolve().parents[2] / "models/decision/lower/robust30"
            frozen_dir = Path("/workspace/project_train/models/bottom_classifier_frozen_final824")

            def default_model(tag):
                env_name = f"BOTTOM_{tag.upper()}_MODEL"
                explicit = str(os.environ.get(env_name, "")).strip()
                if explicit:
                    return Path(explicit)
                if tag == "robust30":
                    engine = robust_dir / "robust30_aug_fp16.engine"
                    onnx = robust_dir / "robust30_aug_normalized_input.onnx"
                elif tag == "v2":
                    engine = frozen_dir / "v2_30d_fp16.engine"
                    onnx = frozen_dir / "v2_30d_normalized_input.onnx"
                else:
                    engine = frozen_dir / "v6_46d_fp16.engine"
                    onnx = frozen_dir / "v6_46d_normalized_input.onnx"
                return engine if engine.is_file() else onnx

            paths = {
                "robust30_model": default_model("robust30"),
                "robust30_norm": Path(os.environ.get(
                    "BOTTOM_ROBUST30_NORM", str(robust_dir / "robust30_aug_normalization.npz")
                )),
                "v2_model": default_model("v2"),
                "v2_norm": Path(os.environ.get(
                    "BOTTOM_V2_NORM", str(frozen_dir / "v2_30d_normalization.npz")
                )),
                "v6_model": default_model("v6"),
                "v6_norm": Path(os.environ.get(
                    "BOTTOM_V6_NORM", str(frozen_dir / "v6_46d_normalization.npz")
                )),
            }

            expected_actions = (
                "POSITION_ADJUST", "PRESS_SWEEP", "WAIST_PULL_LAYDOWN", "ALIGN", "FINISH"
            )
            if tuple(_V110_ACTIONS) != expected_actions:
                raise RuntimeError(f"ACTION_ORDER_MISMATCH runtime={tuple(_V110_ACTIONS)!r} expected={expected_actions!r}")
            action_order_path = robust_dir / "action_order.json"
            if action_order_path.is_file():
                try:
                    action_meta = json.loads(action_order_path.read_text(encoding="utf-8"))
                    if isinstance(action_meta, list):
                        file_actions = tuple(str(x) for x in action_meta)
                    elif isinstance(action_meta, dict):
                        raw_actions = action_meta.get("action_order", action_meta.get("actions", action_meta.get("classes")))
                        file_actions = tuple(str(x) for x in raw_actions) if isinstance(raw_actions, list) else None
                    else:
                        file_actions = None
                    if file_actions is not None and file_actions != expected_actions:
                        raise RuntimeError(f"ACTION_ORDER_JSON_MISMATCH file={file_actions!r} expected={expected_actions!r}")
                    print(f"[ACTION-MODEL-V116-MAPPING] order={expected_actions} json={action_order_path} verified={file_actions is not None}")
                except Exception as exc:
                    raise RuntimeError(f"ROBUST30 action_order.json validation failed: {exc}") from exc
            else:
                print(
                    f"[ACTION-MODEL-V116-MAPPING-WARN] {action_order_path} not found; "
                    f"using V115/runtime hard-coded order={expected_actions}"
                )

            self._v110_models = {}
            self._v110_last_predictions = {}
            self._v110_last_state = None
            self._v110_model_error = None
            self._v110_finish_rule_gate = False
            self._v116_finish_terminal_latched = False
            required = (
                {self._v110_active_model}
                if self._v110_active_model in {"robust30", "v2", "v6"}
                else set()
            )
            specs = (
                ("robust30", 30, "ROBUST30-AUG"),
                ("v2", 30, "V2"),
                ("v6", 46, "V6"),
            )
            for tag, dim, label in specs:
                mp, npz = paths[f"{tag}_model"], paths[f"{tag}_norm"]
                if not mp.is_file() or not npz.is_file():
                    msg = f"{label} unavailable model={mp} norm={npz}"
                    if tag in required:
                        self._v110_model_error = msg
                        print(f"[ACTION-MODEL-V110-ERROR] {msg}")
                    else:
                        print(f"[ACTION-MODEL-V110-SHADOW-SKIP] {msg}")
                    continue
                try:
                    self._v110_models[tag] = _V110ActionModel(mp, npz, dim, label)
                except Exception as exc:
                    msg = f"{label} load failed: {type(exc).__name__}: {exc}"
                    if tag in required:
                        self._v110_model_error = msg
                    print(f"[ACTION-MODEL-V110-ERROR] {msg}")

            active_label = "ROBUST30-AUG" if self._v110_active_model == "robust30" else self._v110_active_model.upper()
            print(
                f"[ACTION-MODEL-V110] active={active_label} inferenceOnly={self._v110_inference_only} "
                f"loaded={sorted(self._v110_models)} FINISH=ACTIVE_MODEL_TERMINAL legacyOverride=DISABLED"
            )
            if self._v110_inference_only and getattr(self, "recorder", None) is not None:
                try:
                    if not isinstance(self.recorder, _V110InferenceRecorderProxy):
                        self.recorder = _V110InferenceRecorderProxy(self.recorder)
                    print("[INFERENCE-ONLY-V110] recorder replaced with NO-SAVE compatibility proxy; review gate disabled")
                except Exception as exc:
                    raise RuntimeError(f"V110 inference-only recorder isolation failed: {exc!r}") from exc

        def _v110_state_vectors(self, obs, bundle):
            probe = SimpleNamespace(obs=obs, bundle=bundle, H=self.H_raw.copy())
            state = self._build_bottom_state_v2(probe)
            names = list(state.get("names", []))
            values = np.asarray(state.get("values", []), np.float32).reshape(-1)
            if values.size != 30 or len(names) != 30:
                raise RuntimeError(f"BOTTOM_STATE_V2_DIM names={len(names)} values={values.size}; expected=30")
            if not np.all(np.isfinite(values)):
                raise RuntimeError("BOTTOM_STATE_V2_NONFINITE")
            out = {"state_obj": state, "v2": values, "v2_names": names, "v6": None, "v6_report": None}
            mask_obj = getattr(obs, "mask", None)
            mask_u8 = getattr(mask_obj, "mask_u8", None) if mask_obj is not None else None
            corrected = getattr(bundle, "corrected", None)
            if corrected is not None and mask_u8 is not None:
                try:
                    grid16, report = _v110_build_v6_grid16(
                        corrected, mask_u8, self.H_raw.copy(), names, values, self.cv2
                    )
                    out["v6"] = np.concatenate([values, grid16.astype(np.float32)]).astype(np.float32)
                    out["v6_report"] = report
                except Exception as exc:
                    out["v6_report"] = {"valid": False, "reason": f"{type(exc).__name__}:{exc}"}
            else:
                out["v6_report"] = {"valid": False, "reason": "FRAME_OR_MASK_UNAVAILABLE"}
            return out

        def _v110_predict_models(self, vectors):
            preds = {}
            if "robust30" in self._v110_models:
                preds["robust30"] = self._v110_models["robust30"].predict(vectors["v2"])
            if "v2" in self._v110_models:
                preds["v2"] = self._v110_models["v2"].predict(vectors["v2"])
            if "v6" in self._v110_models and vectors.get("v6") is not None:
                preds["v6"] = self._v110_models["v6"].predict(vectors["v6"])
            self._v110_last_predictions = copy.deepcopy(preds)
            self._v110_last_state = {
                "v2": vectors["v2"].astype(float).tolist(),
                "v6": None if vectors.get("v6") is None else vectors["v6"].astype(float).tolist(),
                "v6_report": copy.deepcopy(vectors.get("v6_report")),
            }
            for tag in ("robust30", "v2", "v6"):
                p = preds.get(tag)
                if p is None:
                    continue
                label = "ROBUST30-AUG" if tag == "robust30" else tag.upper()
                prob_text = " ".join(f"{a}={float(p['probs'][i]):.3f}" for i, a in enumerate(_V110_ACTIONS))
                print(
                    f"[ACTION-MODEL-V110-{label}] action={p['action']} conf={p['confidence']:.3f} "
                    f"margin={p['margin']:.3f} | {prob_text}"
                )
            return preds

        def _auto_choose_semantic(self, obs, bundle):
            # Explicit legacy recovery mode is kept for contest emergency fallback.
            if self._v110_active_model == "legacy":
                return self._v110_legacy_auto_choose_semantic(obs, bundle)
            if self._v110_model_error:
                probe = SimpleNamespace(obs=obs, bundle=bundle, H=self.H_raw.copy())
                state = self._build_bottom_state_v2(probe)
                return None, f"ACTION MODEL UNAVAILABLE: {self._v110_model_error}", state
            vectors = self._v110_state_vectors(obs, bundle)
            preds = self._v110_predict_models(vectors)
            active = preds.get(self._v110_active_model)
            active_label = "ROBUST30-AUG" if self._v110_active_model == "robust30" else self._v110_active_model.upper()
            if active is None:
                return None, f"ACTIVE {active_label} PREDICTION UNAVAILABLE: {vectors.get('v6_report')}", vectors["state_obj"]
            semantic = str(active["action"])
            reason = (
                f"{active_label} conf={active['confidence']:.3f} "
                f"margin={active['margin']:.3f}"
            )
            if semantic == "FINISH":
                f = self._auto_state_dict(vectors["state_obj"])
                body_angle = abs(float(f.get("body_axis_angle_norm", 0.0)))
                waist_angle = abs(float(f.get("waist_axis_angle_norm", 0.0)))
                center_err = float(f.get("board_center_error_norm", 0.0))
                wrinkle_edge = float(f.get("wrinkle_edge_ratio", 0.0))
                wrinkle_len = float(f.get("dominant_wrinkle_length_norm", 0.0))
                print(
                    f"[FINISH-V116-MODEL-OWNED] active={active_label} FINISH accepted; "
                    f"legacy override disabled; diagnostic center={center_err:.3f} "
                    f"body={body_angle:.3f} waist={waist_angle:.3f} "
                    f"wrinkleEdge={wrinkle_edge:.3f} wrinkleLen={wrinkle_len:.3f}"
                )
                reason += " + FINISH_OWNER=ACTIVE_MODEL legacyOverride=DISABLED"
            return semantic, reason, vectors["state_obj"]

        def _v114_finish_gate_without_waist_y(self, obs, bundle):
            probe = SimpleNamespace(obs=obs, bundle=bundle, H=self.H_raw.copy())
            state = self._build_bottom_state_v2(probe)
            f = self._auto_state_dict(state)
            mask_valid = f.get("mask_valid", 0.0)
            center_err = f.get("board_center_error_norm", 1.0)
            pose_valid = f.get("pose_valid", 0.0)
            leg_balance = f.get("leg_length_balance", 1.0)
            hem_balance = f.get("hem_width_balance", 1.0)
            symmetry_err = f.get("pose_left_right_symmetry_error_norm", 0.0)
            body_angle = abs(f.get("body_axis_angle_norm", 0.0))
            waist_angle = abs(f.get("waist_axis_angle_norm", 0.0))
            wrinkle_edge = f.get("wrinkle_edge_ratio", 0.0)
            wrinkle_len = f.get("dominant_wrinkle_length_norm", 0.0)
            wrinkle_lin = f.get("dominant_wrinkle_linearity_norm", 0.0)

            if mask_valid < 0.5:
                return None, "mask invalid -> REJUDGE required", state
            if center_err > 0.145:
                return "POSITION_ADJUST", f"board center error={center_err:.3f}", state

            align_geom = self._align_vector_geometry(obs)
            if align_geom is not None:
                mid_error = abs(float(align_geom["mid_error_mm"]))
                if mid_error > float(self._v30_align_line_done_mm):
                    return "ALIGN", f"white-line mid={mid_error:.1f}mm", state

            if body_angle > 0.065 or waist_angle > 0.075:
                return "ALIGN", f"axis body={body_angle:.3f} waist={waist_angle:.3f}", state

            asym_report = self._v104_align_asymmetry_report(obs, align_geom) if align_geom is not None else {"trigger": False}
            if pose_valid >= 0.5 and bool(asym_report.get("trigger", False)):
                return "ALIGN", (
                    f"shape asymmetry mode={asym_report.get('mode')} widthDiff={float(asym_report.get('width_diff_mm',0.0)):.1f}mm "
                    f"legLenDiff={float(asym_report.get('leg_len_diff_mm',0.0)):.1f}mm"
                ), state

            folded_like = bool(
                pose_valid >= 0.5
                and (
                    leg_balance < 0.70
                    or hem_balance < 0.58
                    or symmetry_err > 0.30
                )
            )
            if folded_like:
                return "PRESS_SWEEP", (
                    f"residual fold/asymmetry leg={leg_balance:.2f} hem={hem_balance:.2f} sym={symmetry_err:.2f}"
                ), state

            strong_wrinkle = bool(
                wrinkle_len >= 0.11
                and wrinkle_lin >= 0.10
                and wrinkle_edge >= 0.010
            )
            if strong_wrinkle:
                return "PRESS_SWEEP", (
                    f"wrinkle len={wrinkle_len:.3f} lin={wrinkle_lin:.3f} edge={wrinkle_edge:.3f}"
                ), state

            mid_text = "n/a" if align_geom is None else f"{abs(float(align_geom['mid_error_mm'])):.1f}mm"
            return "FINISH", (
                f"finish geometry pass center={center_err:.3f} mid={mid_text} body={body_angle:.3f} waist={waist_angle:.3f}; waist-Y soft only"
            ), state

        def _v110_legacy_auto_choose_semantic(self, obs, bundle):
            probe = SimpleNamespace(obs=obs, bundle=bundle, H=self.H_raw.copy())
            state = self._build_bottom_state_v2(probe)
            f = self._auto_state_dict(state)

            mask_valid = f.get("mask_valid", 0.0)
            center_err = f.get("board_center_error_norm", 1.0)
            width = f.get("mask_width_norm", 0.0)
            height = f.get("mask_height_norm", 0.0)
            solidity = f.get("mask_solidity", 0.0)
            extent = f.get("mask_extent", 0.0)
            pose_valid = f.get("pose_valid", 0.0)
            leg_balance = f.get("leg_length_balance", 1.0)
            hem_balance = f.get("hem_width_balance", 1.0)
            symmetry_err = f.get("pose_left_right_symmetry_error_norm", 0.0)
            waist_y_err = f.get("waist_target_y_error_norm", 0.0)
            body_angle = abs(f.get("body_axis_angle_norm", 0.0))
            waist_angle = abs(f.get("waist_axis_angle_norm", 0.0))
            wrinkle_edge = f.get("wrinkle_edge_ratio", 0.0)
            wrinkle_len = f.get("dominant_wrinkle_length_norm", 0.0)
            wrinkle_lin = f.get("dominant_wrinkle_linearity_norm", 0.0)

            if mask_valid < 0.5:
                return None, "mask invalid -> REJUDGE required", state

            if center_err > 0.145:
                return "POSITION_ADJUST", f"board center error={center_err:.3f}", state

            if pose_valid >= 0.5 and waist_y_err < -0.075:
                return "WAIST_PULL_LAYDOWN", f"waist target y error={waist_y_err:.3f}", state

            align_geom = self._align_vector_geometry(obs)
            if align_geom is not None:
                mid_error = abs(float(align_geom["mid_error_mm"]))
                max_endpoint_error = float(align_geom["max_endpoint_error_mm"])
                if mid_error > float(self._v30_align_line_done_mm) or max_endpoint_error > float(self._v30_align_large_endpoint_error_mm):
                    return "ALIGN", f"white-line mid={mid_error:.1f}mm maxEnd={max_endpoint_error:.1f}mm", state

            if body_angle > 0.065 or waist_angle > 0.075:
                return "ALIGN", f"axis body={body_angle:.3f} waist={waist_angle:.3f}", state

            asym_report = self._v104_align_asymmetry_report(obs, align_geom) if align_geom is not None else {"trigger": False}
            if pose_valid >= 0.5 and bool(asym_report.get("trigger", False)):
                return "ALIGN", (
                    f"shape asymmetry mode={asym_report.get('mode')} widthDiff={float(asym_report.get('width_diff_mm',0.0)):.1f}mm "
                    f"legLenDiff={float(asym_report.get('leg_len_diff_mm',0.0)):.1f}mm"
                ), state

            folded_like = bool(
                pose_valid >= 0.5
                and (
                    leg_balance < 0.70
                    or hem_balance < 0.58
                    or symmetry_err > 0.30
                )
            )
            if folded_like:
                return "PRESS_SWEEP", (
                    f"residual fold/asymmetry leg={leg_balance:.2f} hem={hem_balance:.2f} sym={symmetry_err:.2f}"
                ), state

            strong_wrinkle = bool(
                wrinkle_len >= 0.11
                and wrinkle_lin >= 0.10
                and wrinkle_edge >= 0.010
            )
            if strong_wrinkle:
                return "PRESS_SWEEP", (
                    f"wrinkle len={wrinkle_len:.3f} lin={wrinkle_lin:.3f} edge={wrinkle_edge:.3f}"
                ), state

            return "FINISH", (
                f"finish center={center_err:.3f} body={body_angle:.3f} waist={waist_angle:.3f}"
            ), state

        def _v101_auto_choose_without_align(self, obs, bundle, state=None):
            if self._v110_active_model in {"robust30", "v2", "v6"}:
                pred = (getattr(self, "_v110_last_predictions", {}) or {}).get(self._v110_active_model)
                if isinstance(pred, dict):
                    order = np.argsort(np.asarray(pred.get("probs", []), np.float32))[::-1]
                    excluded = {"ALIGN"}
                    chosen = None
                    for i in order:
                        cand = _V110_ACTIONS[int(i)]
                        if cand in excluded:
                            continue
                        if cand == "FINISH" and self._v110_finish_rule_gate:
                            legacy_sem, legacy_reason, _ = self._v110_legacy_auto_choose_semantic(obs, bundle)
                            if legacy_sem != "FINISH":
                                excluded.add("FINISH")
                                continue
                        chosen = cand
                        break
                    if chosen is not None:
                        if state is None:
                            probe = SimpleNamespace(obs=obs, bundle=bundle, H=self.H_raw.copy())
                            state = self._build_bottom_state_v2(probe)
                        label = "ROBUST30-AUG" if self._v110_active_model == "robust30" else self._v110_active_model.upper()
                        print(f"[ACTION-MODEL-V110-SKIP-ALIGN-ONCE] active={label} nextNonAlign={chosen}")
                        return chosen, f"model non-ALIGN fallback after completed ALIGN: {chosen}", state
            if state is None:
                probe = SimpleNamespace(obs=obs, bundle=bundle, H=self.H_raw.copy())
                state = self._build_bottom_state_v2(probe)
            f = self._auto_state_dict(state)
            mask_valid = f.get("mask_valid", 0.0)
            center_err = f.get("board_center_error_norm", 1.0)
            pose_valid = f.get("pose_valid", 0.0)
            waist_y_err = f.get("waist_target_y_error_norm", 0.0)
            leg_balance = f.get("leg_length_balance", 1.0)
            hem_balance = f.get("hem_width_balance", 1.0)
            symmetry_err = f.get("pose_left_right_symmetry_error_norm", 0.0)
            wrinkle_edge = f.get("wrinkle_edge_ratio", 0.0)
            wrinkle_len = f.get("dominant_wrinkle_length_norm", 0.0)
            wrinkle_lin = f.get("dominant_wrinkle_linearity_norm", 0.0)
            if mask_valid < 0.5:
                return None, "mask invalid -> REJUDGE required", state
            if center_err > 0.145:
                return "POSITION_ADJUST", f"board center error={center_err:.3f}", state
            if pose_valid >= 0.5 and waist_y_err < -0.075:
                return "WAIST_PULL_LAYDOWN", f"waist target y error={waist_y_err:.3f}", state
            folded_like = bool(pose_valid >= 0.5 and (leg_balance < 0.70 or hem_balance < 0.58 or symmetry_err > 0.30))
            if folded_like:
                return "PRESS_SWEEP", f"post-align fold/asymmetry leg={leg_balance:.2f} hem={hem_balance:.2f} sym={symmetry_err:.2f}", state
            strong_wrinkle = bool(wrinkle_len >= 0.11 and wrinkle_lin >= 0.10 and wrinkle_edge >= 0.010)
            if strong_wrinkle:
                return "PRESS_SWEEP", f"post-align wrinkle len={wrinkle_len:.3f} lin={wrinkle_lin:.3f} edge={wrinkle_edge:.3f}", state
            return "FINISH", f"post-align no remaining POSITION/WAIST/PRESS requirement center={center_err:.3f}", state

        def _auto_judge_next(self, trigger):
            if self._integration_mode and bool(getattr(self, "_v116_finish_terminal_latched", False)):
                print(f"[FINISH-V116-TERMINAL-SKIP] AUTO-JUDGE suppressed after model FINISH trigger={trigger}")
                return
            if self._integration_stop_requested():
                print(f"[BOTTOM-INTEGRATION-AUTO-JUDGE-SKIP] stop requested before trigger={trigger}")
                return
            with self._auto_judge_lock:
                if self._auto_judge_busy:
                    print("[AUTO-JUDGE] already running")
                    return
                self._auto_judge_busy = True
            try:
                with self.state_lock:
                    if self.motion_busy or self.inference_busy or self.recorder.review_pending():
                        print("[AUTO-JUDGE] blocked by active motion/inference/review")
                        return
                    self.status = f"AUTO JUDGE: FRESH SNAPSHOT ({trigger})"

                if not self.empty_baseline_ready:
                    self.status = "AUTO JUDGE BLOCKED: EMPTY BOARD E REQUIRED"
                    return

                if not self._ensure_camera_clear("AUTO_JUDGE_BEFORE_NEXT", allow_move=False):
                    self.status = "AUTO JUDGE BLOCKED: CAMERA-CLEAR NOT VERIFIED"
                    return

                bundle = self._capture_i_frame_from_live("D56_WAIST_LIFT_LAYDOWN")
                obs = self._infer_for_action("D56_WAIST_LIFT_LAYDOWN", bundle.corrected)
                mask_info = self._v30_mask_info(SimpleNamespace(observation=obs))
                if (not bool(mask_info.get("valid", False))) or bool(mask_info.get("fallback", False)):
                    recovered = False
                    for retry_index in range(1, int(self._v111_action_mask_retry_count) + 1):
                        print(
                            f"[ACTION-MODEL-V111-MASK-RETRY] trigger={trigger} retry={retry_index}/{int(self._v111_action_mask_retry_count)} "
                            f"valid={bool(mask_info.get('valid',False))} fallback={bool(mask_info.get('fallback',False))} "
                            f"source={mask_info.get('source','-')}"
                        )
                        time.sleep(float(self._v111_action_mask_retry_wait_s))
                        bundle = self._capture_i_frame_from_live("D56_WAIST_LIFT_LAYDOWN")
                        obs = self._infer_for_action("D56_WAIST_LIFT_LAYDOWN", bundle.corrected)
                        mask_info = self._v30_mask_info(SimpleNamespace(observation=obs))
                        if bool(mask_info.get("valid", False)) and not bool(mask_info.get("fallback", False)):
                            recovered = True
                            print(
                                f"[ACTION-MODEL-V111-MASK-RECOVERED] retry={retry_index} "
                                f"area={float(mask_info.get('area_px',0.0)):.0f}px source={mask_info.get('source','MASK')}"
                            )
                            break
                    if not recovered:
                        semantic = None
                        reason = (
                            f"TRUSTED YOLO MASK UNAVAILABLE AFTER {int(self._v111_action_mask_retry_count)} FRESH RETRIES; "
                            "fallback/BGDIFF is forbidden for next-action classification"
                        )
                        try:
                            probe = SimpleNamespace(obs=obs, bundle=bundle, H=self.H_raw.copy())
                            state = self._build_bottom_state_v2(probe)
                        except Exception:
                            state = None
                        print(f"[ACTION-MODEL-V111-MASK-BLOCK] {reason}")
                    else:
                        semantic, reason, state = self._auto_choose_semantic(obs, bundle)
                else:
                    semantic, reason, state = self._auto_choose_semantic(obs, bundle)
                if bool(getattr(self, "_v101_skip_align_autojudge_once", False)):
                    self._v101_skip_align_autojudge_once = False
                    if str(semantic or "").upper() == "ALIGN":
                        semantic, reason, state = self._v101_auto_choose_without_align(obs, bundle, state)
                        print(f"[AUTO-JUDGE-V101-SKIP-ALIGN-ONCE] fresh ALIGN was already complete; next={semantic} reason={reason}")
                self._v87_cache_auto_observation(bundle, obs, semantic, reason)
                if bool(getattr(self, "_v87_finish_requested", False)):
                    print(
                        f"[FINISH-V87-AUTO-CACHE] current AUTO-JUDGE observation captured for pending FINISH; "
                        f"auto={semantic} reason={reason}; suppress auto action selection"
                    )
                    return

                if semantic is None:
                    self.auto_recommended = "REJUDGE"
                    self.semantic_selected = None
                    self.pending_semantic = None
                    self.selected_action = None
                    self.status = f"AUTO JUDGE: REJUDGE REQUIRED - {reason}"
                    print(f"[AUTO-JUDGE] trigger={trigger} action=REJUDGE reason={reason}")
                    return

                if self._integration_stop_requested():
                    print(f"[BOTTOM-INTEGRATION-AUTO-JUDGE-SKIP] stop requested after inference trigger={trigger}")
                    return
                self.auto_recommended = semantic
                if self._integration_mode and str(semantic).upper() == "FINISH":
                    self._v116_finish_terminal_latched = True
                    print("[FINISH-V116-TERMINAL-LATCH] active model selected FINISH; later AUTO-JUDGE/action replacement blocked")
                print(f"[AUTO-JUDGE] trigger={trigger} action={semantic} reason={reason}")
                self._select_semantic(semantic, origin="AUTO")
            except Exception as exc:
                self.status = f"AUTO JUDGE ERROR: {exc!r}"
                print(f"[AUTO-JUDGE-ERROR] {exc!r}")
            finally:
                with self._auto_judge_lock:
                    self._auto_judge_busy = False

        def _schedule_auto_judge(self, trigger, delay=0.05):
            if self._integration_mode and bool(getattr(self, "_v116_finish_terminal_latched", False)):
                print(f"[FINISH-V116-TERMINAL-SKIP] schedule suppressed after model FINISH trigger={trigger}")
                return
            if not bool(self.front.auto_prepare_next):
                return

            def worker():
                if delay > 0.0 and self._integration_mode:
                    if self._integration_stop_event.wait(delay):
                        return
                elif delay > 0.0:
                    time.sleep(delay)
                if self._integration_stop_requested():
                    return
                deadline = time.time() + 8.0
                while time.time() < deadline:
                    with self.state_lock:
                        blocked = bool(self.motion_busy or self.inference_busy)
                    if self._integration_stop_requested():
                        return
                    if not blocked and not self.recorder.review_pending():
                        break
                    time.sleep(0.05)
                if self._integration_stop_requested():
                    return
                self._auto_judge_next(trigger)

            threading.Thread(target=worker, name="bottom-vla-v30-auto-judge", daemon=True).start()

        def _advance_after_collection(self, keep, semantic):
            semantic = str(semantic or self.auto_recommended or "")
            if bool(getattr(self, "_v110_inference_only", False)):
                print(
                    f"[INFERENCE-ONLY-V110] legacy collection advance ignored for {semantic}; "
                    "post-physical-motion watcher owns the next model judgement"
                )
                return
            self.status = f"{'KEEP' if keep else 'DISCARD'} SAVED: AUTO JUDGE NEXT"
            print(f"[COLLECT] {'KEEP' if keep else 'DISCARD'} fixed auto_next removed; fresh AUTO-JUDGE scheduled")
            self.semantic_selected = None
            self.pending_semantic = None
            self.selected_action = None
            self._human_selected = None
            self._prepare_origin = "AUTO"
            self._invalidate_for_new_action("POST_COLLECTION_AUTO_JUDGE")
            self._schedule_auto_judge(f"after_{semantic.lower()}", 0.08)

        def _v52_basket_gentle_open_to(self, target, stage):
            target = float(target)
            fb = self._basket_feedback(quiet=True)
            current = None
            if fb is not None and "t" in fb:
                try:
                    current = float(fb["t"])
                except Exception:
                    current = None
            if current is None or not math.isfinite(current):
                current = float(self.front.basket_close_target)
            if current <= target + 0.02:
                return fb
            values = []
            value = float(current)
            while value - target > float(self._v52_basket_open_step_rad) + 1e-9:
                value -= float(self._v52_basket_open_step_rad)
                values.append(float(value))
            if not values or values[-1] > target + 0.02:
                values.append(float(target))
            print(
                f"[BASKET-V52-GENTLE-OPEN] stage={stage} current={current:.3f} target={target:.3f} "
                f"steps={len(values)} step<={self._v52_basket_open_step_rad:.2f}rad"
            )
            last_fb = fb
            for index, value in enumerate(values, 1):
                self._basket_set_gripper(float(value), 0.0, f"{stage}-GENTLE_{index}")
                time.sleep(float(self._v52_basket_open_step_wait_s))
                sensed = self._basket_feedback(quiet=True)
                if sensed is not None:
                    last_fb = sensed
            return last_fb

        def _v52_basket_lower_closed_to_contact(self, stage_prefix):
            plan = self._v52_basket_active_plan
            contact = self._v52_basket_contact_feedback
            if not isinstance(plan, dict) or not isinstance(contact, dict):
                raise RuntimeError("BASKET-V52 recovery context unavailable")
            fb = self._basket_feedback(quiet=True)
            if fb is None or not all(k in fb for k in ("x", "y", "z")):
                raise RuntimeError("BASKET-V52 feedback unavailable before vertical recovery descent")
            x = float(fb["x"])
            y = float(fb["y"])
            current_z = float(fb["z"])
            target_z = max(float(plan.get("min_safe_z", contact.get("z", current_z))), float(contact.get("z", current_z)))
            tool_t = float(plan.get("close_target", self.front.basket_close_target))
            approach_z = min(current_z, target_z + 35.0)
            if current_z > approach_z + 8.0:
                stage = f"{stage_prefix}-APPROACH"
                print(f"[BASKET-V52-RELOWER] {stage} sameXY=({x:.1f},{y:.1f}) z={current_z:.1f}->{approach_z:.1f} closed")
                self._basket_move_goal(x, y, approach_z, tool_t, float(self.front.basket_descent_speed), stage)
                self._basket_wait_waypoint(stage, (x, y, approach_z))
            fb2 = self._basket_feedback(quiet=True)
            if fb2 is not None and all(k in fb2 for k in ("x", "y", "z")):
                x = float(fb2["x"])
                y = float(fb2["y"])
                current_z = float(fb2["z"])
            if current_z > target_z + 5.0:
                stage = f"{stage_prefix}-CONTACT-Z"
                print(f"[BASKET-V52-RELOWER] {stage} sameXY=({x:.1f},{y:.1f}) z={current_z:.1f}->{target_z:.1f} closed")
                self._basket_move_goal(x, y, target_z, tool_t, float(self.front.basket_slow_speed), stage)
                self._basket_wait_waypoint(stage, (x, y, target_z))
            return self._basket_feedback(quiet=True)

        def _v52_basket_regrasp_once(self, attempt):
            plan = self._v52_basket_active_plan
            if not isinstance(plan, dict):
                return {"success": False, "reason": "V52_PLAN_CONTEXT_MISSING"}
            print(f"[BASKET-V52-REGRASP] attempt={attempt}/{self._v52_basket_regrasp_max} retention LOW -> same-XY vertical return to grasp height")
            self._v52_basket_lower_closed_to_contact(f"BASKET-V52-REGRASP-{attempt}")
            open_target = float(plan.get("post_contact_open_angle", self.front.basket_grip_fully_open))
            self._v52_basket_gentle_open_to(open_target, f"BASKET-V52-REGRASP-{attempt}-OPEN")
            open_fb = self._basket_verify_gripper(
                open_target,
                float(self.front.basket_close_tolerance_rad),
                3,
                max(0.20, min(0.45, float(self.front.basket_post_contact_open_settle_s))),
                f"BASKET-V52-REGRASP-{attempt}-OPEN-VERIFY",
            )
            close_fb = self._basket_verify_gripper(
                float(plan["close_target"]),
                float(self.front.basket_close_tolerance_rad),
                int(self.front.basket_close_attempts),
                float(self.front.basket_gripper_settle_s),
                "BASKET-CLOSE-VERIFY",
            )
            final_latch = self._basket_final_grip_latch(plan)
            lift_x = float(close_fb["x"])
            lift_y = float(close_fb["y"])
            lift_result = self._basket_lift_adaptive(
                lift_x,
                lift_y,
                float(close_fb["z"]),
                float(plan["pickup_lift_z"]),
                float(plan["close_target"]),
            )
            if not bool(lift_result.get("reached")) and not bool(lift_result.get("stalled")):
                return {"success": False, "reason": "V52_REGRASP_LIFT_UNCONFIRMED", "attempt": int(attempt)}
            retention = dict(super()._basket_retention() or {})
            retention.update({
                "v52_regrasp_attempt": int(attempt),
                "v52_regrasp_open_feedback": None if open_fb is None else float(open_fb.get("t", float("nan"))),
                "v52_regrasp_close_feedback": float(close_fb.get("t", float("nan"))),
                "v52_regrasp_final_latch": copy.deepcopy(final_latch),
            })
            print(
                f"[BASKET-V52-REGRASP-CHECK] attempt={attempt} success={bool(retention.get('success',False))} "
                f"reason={retention.get('reason')}"
            )
            return retention

        def _basket_retention(self):
            initial = dict(super()._basket_retention() or {})
            if bool(initial.get("success", False)):
                return initial
            if not isinstance(self._v52_basket_active_plan, dict) or not isinstance(self._v52_basket_contact_feedback, dict):
                return initial
            last = initial
            for attempt in range(1, int(self._v52_basket_regrasp_max) + 1):
                self._v52_basket_regrasp_count = int(attempt)
                try:
                    last = self._v52_basket_regrasp_once(attempt)
                except Exception as exc:
                    print(f"[BASKET-V52-REGRASP-ERROR] attempt={attempt} {type(exc).__name__}: {exc}; stop automatic retry escalation")
                    last = {"success": False, "reason": f"V52_REGRASP_ERROR:{type(exc).__name__}:{exc}", "attempt": int(attempt)}
                    break
                if bool(last.get("success", False)):
                    out = dict(last)
                    out["v52_recovered_after_retention_low"] = True
                    out["v52_regrasp_attempts_used"] = int(attempt)
                    self._v52_basket_retention_exhausted = False
                    print(f"[BASKET-V52-REGRASP-SUCCESS] attempt={attempt}; continue existing board-center/rotation/place sequence")
                    return out
            print("[BASKET-V52-REGRASP-EXHAUSTED] all in-place attempts failed; lower once more before base verified release")
            try:
                self._v52_basket_lower_closed_to_contact("BASKET-V52-EXHAUSTED")
            except Exception as exc:
                print(f"[BASKET-V52-EXHAUSTED-LOWER-ERROR] {type(exc).__name__}: {exc}")
            self._v52_basket_retention_exhausted = True
            out = dict(last)
            out["success"] = False
            out["reason"] = str(out.get("reason") or initial.get("reason") or "RETENTION_LOW")
            out["v52_regrasp_exhausted"] = True
            out["v52_regrasp_attempts_used"] = int(self._v52_basket_regrasp_count)
            return out

        def _basket_release_verified(self, target, stage):
            if str(stage) == "BASKET-RELEASE-VERIFY":
                self._v52_basket_gentle_open_to(float(target), "BASKET-V52-PLACEMENT-RELEASE")
            return super()._basket_release_verified(target, stage)

        def _basket_execute_contact_descent(self, plan):
            self._v52_basket_active_plan = copy.deepcopy(plan) if isinstance(plan, dict) else plan
            self._v52_basket_contact_feedback = None
            self._v52_basket_regrasp_count = 0
            self._v52_basket_retention_exhausted = False
            self._v57_basket_failure_safe_raised = False
            grasp_x, grasp_y = map(float, plan["grasp_xy"])
            tool_t = float(plan["tool_t"])
            open_angle = float(plan["open_descent_angle"])
            open_fb = self._basket_verify_gripper(
                open_angle, max(0.18, float(self.front.basket_close_tolerance_rad)), 4,
                float(self.front.basket_gripper_settle_s), "BASKET-DESCENT-OPEN"
            )
            current_z = float(open_fb.get("z", plan["hover_z"]))
            baseline = self._basket_collect_baseline(current_z)
            previous = baseline
            fast_targets = self._basket_descending_targets(
                current_z, max(float(plan["rim_mean_z"]), float(plan["min_safe_z"])),
                self.front.basket_fast_step_mm
            )
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
            soft_limit_z = max(
                float(plan["min_safe_z"]),
                float(plan["rim_mean_z"]) - float(self._v51_basket_soft_depth_below_rim_mm),
            )
            slow_targets = self._basket_descending_targets(
                min(previous_z, float(plan["rim_mean_z"])), soft_limit_z, self.front.basket_slow_step_mm
            )
            last_fb = fb0
            for index, target_z in enumerate(slow_targets, 1):
                before_z = previous_z
                self._basket_move_goal(grasp_x, grasp_y, target_z, open_angle, self.front.basket_slow_speed, f"BASKET_SLOW_{index}")
                fb = self._basket_probe_feedback(f"BASKET-SLOW-{index}", target_z)
                last_fb = fb
                shoulder_load = float(baseline[1]) - float(fb.get("torS", 0.0))
                elbow_change = abs(float(fb.get("torE", 0.0)) - float(baseline[2]))
                z_lag = float(fb.get("z", target_z)) - float(target_z)
                candidate = shoulder_load >= float(self.front.basket_contact_shoulder_delta) and (
                    elbow_change >= float(self.front.basket_contact_elbow_delta)
                    or z_lag >= float(self.front.basket_contact_z_lag_mm)
                )
                torque_count = torque_count + 1 if candidate else 0
                commanded_drop = max(0.0, before_z - float(target_z))
                actual_drop = max(0.0, before_z - float(fb.get("z", target_z)))
                stall = commanded_drop >= float(self.front.basket_stall_min_command_mm) and actual_drop <= float(self.front.basket_stall_max_actual_mm)
                stall_count = stall_count + 1 if stall else 0
                hard = max(shoulder_load, elbow_change)
                print(
                    f"[BASKET-CONTACT] step={index} shoulder={shoulder_load:.1f} elbow={elbow_change:.1f} "
                    f"zlag={z_lag:.1f} confirm={torque_count}/{int(self.front.basket_contact_confirm_steps)}"
                )
                previous_z = float(fb.get("z", target_z))
                if hard >= float(self.front.basket_hard_axis_delta):
                    raise RuntimeError("basket hard shoulder/elbow torque stop")
                if torque_count >= int(self.front.basket_contact_confirm_steps):
                    self._v52_basket_contact_feedback = copy.deepcopy(fb)
                    return {"contact": True, "feedback": fb, "reason": "TORQUE_CONTACT"}
                if stall_count >= int(self.front.basket_stall_confirm_steps):
                    raise RuntimeError("basket Z stall before confirmed clothing contact")

            print(
                f"[BASKET-V51-DEPTH-FALLBACK] no confirmed torque contact by z={soft_limit_z:.1f}; "
                "stop descending and attempt grasp here"
            )
            self._v52_basket_contact_feedback = copy.deepcopy(last_fb)
            return {"contact": True, "feedback": last_fb, "reason": "SOFT_DEPTH_FALLBACK"}

        def _v57_basket_vertical_raise_same_xy(self, stage, rise_mm=None, target_z=None):
            fb = self._basket_feedback(quiet=True)
            if fb is None or not all(k in fb for k in ("x", "y", "z", "t")):
                raise RuntimeError(f"{stage}: ARM2 feedback unavailable for same-XY vertical raise")
            x, y, z, tool_t = (float(fb[k]) for k in ("x", "y", "z", "t"))
            plan = self._v52_basket_active_plan if isinstance(self._v52_basket_active_plan, dict) else {}
            if target_z is None:
                target_z = z + float(rise_mm if rise_mm is not None else self._v57_basket_open_relief_mm)
            target_z = float(target_z)
            pickup_z = plan.get("pickup_lift_z")
            if pickup_z is not None:
                target_z = min(target_z, float(pickup_z))
            if target_z <= z + 2.0:
                return fb
            speed = min(0.45, max(0.28, float(getattr(self.front, "basket_slow_speed", 0.35))))
            print(
                f"[BASKET-V57-VERTICAL-RELIEF] stage={stage} sameXY=({x:.1f},{y:.1f}) "
                f"z={z:.1f}->{target_z:.1f} gripperHold={tool_t:.3f} speed={speed:.2f}"
            )
            self._basket_move_goal(x, y, target_z, tool_t, speed, stage)
            try:
                self._basket_wait_waypoint(stage, (x, y, target_z))
            except Exception as exc:
                print(f"[BASKET-V57-VERTICAL-RELIEF-WARN] {stage} arrival verify: {exc}")
            return self._basket_feedback(quiet=True) or fb

        def _v57_basket_return_open_to_contact(self, open_target, stage):
            contact = self._v52_basket_contact_feedback
            fb = self._basket_feedback(quiet=True)
            if not isinstance(contact, dict) or fb is None or not all(k in fb for k in ("x", "y", "z")):
                raise RuntimeError(f"{stage}: contact/feedback unavailable for open return")
            x, y, z = float(fb["x"]), float(fb["y"]), float(fb["z"])
            target_z = float(contact.get("z", z))
            if z <= target_z + 3.0:
                return fb
            speed = min(0.35, max(0.25, float(getattr(self.front, "basket_slow_speed", 0.35))))
            print(
                f"[BASKET-V57-OPEN-RETURN] sameXY=({x:.1f},{y:.1f}) z={z:.1f}->{target_z:.1f} "
                f"OPEN={float(open_target):.3f}; resume original close sequence only after contact-height return"
            )
            self._basket_move_goal(x, y, target_z, float(open_target), speed, stage)
            try:
                self._basket_wait_waypoint(stage, (x, y, target_z))
            except Exception as exc:
                print(f"[BASKET-V57-OPEN-RETURN-WARN] {stage} arrival verify: {exc}")
            return self._basket_feedback(quiet=True) or fb

        def _v57_basket_raise_to_hover_same_xy(self, stage):
            fb = self._basket_feedback(quiet=True)
            if fb is None or not all(k in fb for k in ("z",)):
                raise RuntimeError(f"{stage}: feedback unavailable before emergency vertical raise")
            z = float(fb["z"])
            plan = self._v52_basket_active_plan if isinstance(self._v52_basket_active_plan, dict) else {}
            hover_z = plan.get("hover_z")
            if hover_z is None:
                target_z = z + float(self._v57_basket_emergency_raise_mm)
            else:
                target_z = max(z + float(self._v57_basket_emergency_raise_mm), float(hover_z))
            out = self._v57_basket_vertical_raise_same_xy(stage, target_z=target_z)
            self._v57_basket_failure_safe_raised = True
            return out

        def _basket_verify_gripper(self, target, tolerance, attempts, settle, stage):
            stage_text = str(stage)
            if stage_text == "BASKET-POST-CONTACT-OPEN":
                try:
                    return super()._basket_verify_gripper(target, tolerance, attempts, settle, stage)
                except Exception as first_exc:
                    print(f"[BASKET-V57-OPEN-JAM] initial post-contact OPEN failed: {first_exc}; relieve basket pressure vertically")
                    self._v57_basket_vertical_raise_same_xy(
                        "BASKET-V57-POST-CONTACT-RELIEF", rise_mm=float(self._v57_basket_open_relief_mm)
                    )
                    try:
                        opened_fb = super()._basket_verify_gripper(
                            target, tolerance, max(3, int(attempts)), max(0.30, float(settle)),
                            "BASKET-V57-POST-CONTACT-OPEN-AFTER-RELIEF",
                        )
                        self._v57_basket_return_open_to_contact(
                            float(target), "BASKET-V57-POST-CONTACT-RETURN-OPEN"
                        )
                        return self._basket_feedback(quiet=True) or opened_fb
                    except Exception as second_exc:
                        print(f"[BASKET-V57-OPEN-JAM] +{self._v57_basket_open_relief_mm:.0f}mm relief still blocked: {second_exc}; abort grasp continuation and raise vertically toward hover")
                        self._v57_basket_raise_to_hover_same_xy("BASKET-V57-POST-CONTACT-EMERGENCY-RAISE")
                        try:
                            self._v52_basket_gentle_open_to(1.35, "BASKET-V57-POST-CONTACT-ABORT-RELEASE")
                            super()._basket_verify_gripper(
                                1.35, max(0.18, float(self.front.basket_close_tolerance_rad)), 4,
                                max(0.30, float(settle)), "BASKET-V57-POST-CONTACT-ABORT-RELEASE-VERIFY",
                            )
                        except Exception as release_exc:
                            print(f"[BASKET-V57-ABORT-RELEASE-WARN] safe-height release still unverified: {release_exc}")
                        raise RuntimeError(
                            f"BASKET-V57 post-contact OPEN blocked after vertical relief; ARM2 raised vertically before abort: {second_exc}"
                        ) from second_exc
            if stage_text != "BASKET-CLOSE-VERIFY":
                return super()._basket_verify_gripper(target, tolerance, attempts, settle, stage)
            min_close = 2.80
            soft_min = float(self._v51_basket_close_soft_min_rad)
            stable_span = float(self._v51_basket_close_stable_span_rad)
            last_fb = None
            last_actual = None
            recent = []
            for attempt in range(1, max(1, int(attempts)) + 1):
                self._basket_set_gripper(float(target), 0.0, f"{stage}_{attempt}")
                time.sleep(max(0.0, float(settle)))
                fb = self._basket_feedback(quiet=True)
                if fb is None or "t" not in fb:
                    print(f"[{stage}-V51] attempt={attempt} feedback unavailable")
                    continue
                last_fb = fb
                last_actual = float(fb["t"])
                recent.append(last_actual)
                recent = recent[-2:]
                error = abs(last_actual - float(target))
                accepted = last_actual >= min_close
                stable_cloth = bool(
                    len(recent) >= 2
                    and min(recent) >= soft_min
                    and max(recent) - min(recent) <= stable_span
                )
                print(
                    f"[{stage}-V51] attempt={attempt} target={float(target):.3f} actual={last_actual:.3f} "
                    f"error={error:.3f}rad primary>={min_close:.3f} stableCloth={stable_cloth}"
                )
                if accepted:
                    return fb
                if stable_cloth:
                    print(
                        f"[BASKET-CLOSE-V51-CLOTH-ACCEPT] stable feedback={last_actual:.3f}rad "
                        f">={soft_min:.2f}; treating cloth-blocked close as successful"
                    )
                    return fb
            raise RuntimeError(
                f"{stage} failed primary={min_close:.3f} soft={soft_min:.3f} "
                f"actual={last_actual} feedback={last_fb}"
            )

        def _basket_return_arm2_standby(self, release_verified):
            if not release_verified:
                raise RuntimeError("ARM2 standby blocked until release is verified")
            if bool(self._v52_basket_retention_exhausted):
                fb = self._basket_feedback(quiet=True)
                plan = self._v52_basket_active_plan
                if fb is None or not all(k in fb for k in ("x", "y", "z", "t")) or not isinstance(plan, dict):
                    raise RuntimeError("BASKET-V52 exhausted recovery cannot verify pose for safe vertical raise")
                x, y, z, tool_t = float(fb["x"]), float(fb["y"]), float(fb["z"]), float(fb["t"])
                safe_z = float(plan.get("pickup_lift_z", z + 180.0))
                if safe_z > z + 20.0:
                    stage = "BASKET-V52-FAIL-SAFE-VERTICAL-RAISE"
                    print(f"[BASKET-V52-FAIL-RETURN] released low; raise sameXY=({x:.1f},{y:.1f}) z={z:.1f}->{safe_z:.1f} before standby transit")
                    self._basket_move_goal(x, y, safe_z, tool_t, float(self.front.basket_lift_speed), stage)
                    self._basket_wait_waypoint(stage, (x, y, safe_z))
                self._v52_basket_retention_exhausted = False
            target = np.asarray([
                float(self.front.basket_arm2_standby_x),
                float(self.front.basket_arm2_standby_y),
                float(self.front.basket_arm2_standby_z),
            ], dtype=np.float64)
            tool_t = float(self.front.basket_arm2_standby_t)
            speed = float(self.front.basket_standby_speed)
            tolerance = float(self.front.basket_move_tolerance_mm)
            attempts = 4
            last_xyz = None
            last_error = float("inf")
            for attempt in range(1, attempts + 1):
                print(f"[BASKET-STANDBY-V33] command attempt={attempt}/{attempts} target=({target[0]:.1f},{target[1]:.1f},{target[2]:.1f})")
                self._basket_move_goal(float(target[0]), float(target[1]), float(target[2]), tool_t, speed, f"BASKET_ARM2_STANDBY_RETRY_{attempt}")
                deadline = time.monotonic() + 2.5
                while time.monotonic() < deadline:
                    time.sleep(max(0.15, min(0.35, float(self.front.basket_move_poll_s))))
                    fb = self._basket_feedback(quiet=True)
                    if fb is None or not all(k in fb for k in ("x", "y", "z")):
                        continue
                    last_xyz = np.asarray([float(fb["x"]), float(fb["y"]), float(fb["z"])], dtype=np.float64)
                    last_error = float(np.linalg.norm(last_xyz - target))
                    print(f"[BASKET-STANDBY-V33] attempt={attempt} error={last_error:.1f}mm actual=({last_xyz[0]:.1f},{last_xyz[1]:.1f},{last_xyz[2]:.1f})")
                    if last_error <= tolerance:
                        return last_xyz
            raise RuntimeError(f"BASKET ARM2 standby failed after {attempts} T104 commands error={last_error:.1f}mm actual={None if last_xyz is None else last_xyz.tolist()}")

        def _v78_build_initial_spread_plan(self, obs):
            H = getattr(self, "H_raw", None)
            samples = self._v56_mask_board_samples(obs, H, max_points=12000)
            if samples is None or len(samples) < 120:
                return None, "MASK_SAMPLES_UNAVAILABLE"
            pts = np.asarray(samples, np.float64).reshape(-1, 2)
            pts = pts[np.isfinite(pts).all(axis=1)]
            if len(pts) < 120:
                return None, "MASK_SAMPLES_TOO_SMALL"
            y_hi = float(np.percentile(pts[:, 1], 97.0))
            y_lo = float(np.percentile(pts[:, 1], 5.0))
            band = min(float(self._v78_initial_spread_top_band_mm), max(75.0, 0.30 * max(1.0, y_hi - y_lo)))
            top = pts[pts[:, 1] >= y_hi - band]
            if len(top) < 60:
                return None, "TOP_MASK_BAND_TOO_SMALL"
            x_lo = float(np.percentile(top[:, 0], 4.0))
            x_hi = float(np.percentile(top[:, 0], 96.0))
            if x_hi - x_lo < float(self._v78_initial_spread_min_pair_sep_mm):
                return None, f"TOP_WIDTH_TOO_SMALL:{x_hi-x_lo:.1f}"
            x_mid = 0.5 * (x_lo + x_hi)

            contour_px, contour_board = self._v53_board_contour(obs)
            contour_arr = None if contour_board is None else np.asarray(contour_board, np.float64).reshape(-1, 2)
            if contour_arr is None or len(contour_arr) < 8:
                return None, "MASK_CONTOUR_UNAVAILABLE"

            def local_inward_contact(q, desired_depth):
                p = np.asarray(q, np.float64).reshape(2)
                a = contour_arr
                b = np.roll(a, -1, axis=0)
                ab = b - a
                denom = np.sum(ab * ab, axis=1)
                t = np.zeros(len(a), np.float64)
                valid = denom > 1e-9
                t[valid] = np.clip(np.sum((p - a[valid]) * ab[valid], axis=1) / denom[valid], 0.0, 1.0)
                edge_pts = a + ab * t[:, None]
                dist = np.linalg.norm(edge_pts - p[None, :], axis=1)
                if len(dist) == 0 or not np.isfinite(dist).any():
                    return None
                idx = int(np.nanargmin(dist))
                edge = edge_pts[idx]
                inward = p - edge
                n = float(np.linalg.norm(inward))
                if n < 1.0:
                    return None
                inward /= n
                return edge + inward * float(desired_depth)

            def choose(arm_key):
                if arm_key == "arm2":
                    pool = top[(top[:, 0] <= x_mid) & (top[:, 0] <= x_lo + 0.42 * (x_hi - x_lo))]
                    outer_x = x_lo
                else:
                    pool = top[(top[:, 0] >= x_mid) & (top[:, 0] >= x_hi - 0.42 * (x_hi - x_lo))]
                    outer_x = x_hi
                if len(pool) < 10:
                    return None
                step = max(1, int(math.ceil(len(pool) / 850.0)))
                tiers = tuple(float(v) for v in getattr(self, "_v79_initial_spread_inset_tiers_mm", (55.0, 50.0, 45.0)))
                for target_depth in tiers:
                    rows = []
                    for q0 in pool[::step]:
                        rep0 = self._v53_mask_point_report(obs, q0)
                        if not bool((rep0 or {}).get("ok", False)):
                            continue
                        q = local_inward_contact(q0, target_depth)
                        if q is None or not self._v56_align_board_safe(q, 10.0):
                            continue
                        ok_reach, _, _ = self._v56_align_arm_reach(arm_key, q, final_target=False)
                        if not ok_reach:
                            continue
                        rep = self._v53_mask_point_report(obs, q)
                        depth = rep.get("edge_depth_mm") if isinstance(rep, dict) else None
                        if not bool((rep or {}).get("ok", False)) or depth is None:
                            continue
                        depth = float(depth)
                        if depth < target_depth - 4.0 or depth > target_depth + 18.0:
                            continue
                        support_ok, support = self._v56_mask_support_at_board(
                            obs, H, q,
                            radius_px=int(getattr(self, "_v79_initial_spread_support_radius_px", 7)),
                            min_ratio=float(getattr(self, "_v79_initial_spread_support_min_ratio", 0.82)),
                        )
                        if not support_ok:
                            continue
                        outer_pen = abs(float(q[0]) - outer_x)
                        top_pen = max(0.0, y_hi - float(q[1]))
                        score = 3.0 * abs(depth - target_depth) + 0.14 * outer_pen + 0.08 * top_pen - 15.0 * float(support)
                        rows.append((float(score), np.asarray(q, np.float64), depth, float(support), target_depth))
                    if rows:
                        rows.sort(key=lambda r: r[0])
                        return rows[0]
                return None

            left = choose("arm2")
            right = choose("arm1")
            if left is None or right is None:
                return None, "NO_SAFE_UPPER_OUTER_CONTACT_PAIR"
            src = {"arm2": left[1], "arm1": right[1]}
            sep = float(np.linalg.norm(src["arm1"] - src["arm2"]))
            if sep < float(self._v78_initial_spread_min_pair_sep_mm):
                return None, f"CONTACT_PAIR_TOO_CLOSE:{sep:.1f}"

            requested = float(self._v78_initial_spread_pull_mm)
            desired = {
                "arm2": src["arm2"] + np.asarray([-requested, 0.0], np.float64),
                "arm1": src["arm1"] + np.asarray([requested, 0.0], np.float64),
            }
            safe = {}
            moves = {}
            for k in ("arm2", "arm1"):
                q, frac = self._v56_align_safe_progress(k, src[k], desired[k])
                if q is None:
                    return None, f"{k.upper()}_NO_SAFE_OUTWARD_TARGET"
                safe[k] = np.asarray(q, np.float64)
                moves[k] = float(np.linalg.norm(safe[k] - src[k]))
            common = min(float(moves["arm2"]), float(moves["arm1"]), requested)
            if common < float(self._v78_initial_spread_min_pull_mm):
                return None, f"OUTWARD_TRAVEL_TOO_SHORT:{common:.1f}"
            target = {
                "arm2": src["arm2"] + np.asarray([-common, 0.0], np.float64),
                "arm1": src["arm1"] + np.asarray([common, 0.0], np.float64),
            }
            for k in ("arm2", "arm1"):
                if not self._v56_align_board_safe(target[k], 8.0):
                    return None, f"{k.upper()}_TARGET_OUTSIDE_BOARD"
                ok_reach, why, _ = self._v56_align_arm_reach(k, target[k], final_target=True)
                if not ok_reach:
                    return None, f"{k.upper()}_TARGET_UNREACHABLE:{why}"
            motion = {
                "source_board": {k: src[k].astype(float).tolist() for k in ("arm2", "arm1")},
                "v54_level_board": {k: src[k].astype(float).tolist() for k in ("arm2", "arm1")},
                "target_board": {k: target[k].astype(float).tolist() for k in ("arm2", "arm1")},
                "v63_align_stage_mode": "UP_ONLY",
                "v63_up_remaining_mm": 0.0,
                "v78_initial_spread": True,
            }
            print(
                f"[INITIAL-SPREAD-V79-PLAN] A2={np.round(src['arm2'],1).tolist()} depth={left[2]:.1f}mm targetInset={left[4]:.0f}mm -> {np.round(target['arm2'],1).tolist()} | "
                f"A1={np.round(src['arm1'],1).tolist()} depth={right[2]:.1f}mm targetInset={right[4]:.0f}mm -> {np.round(target['arm1'],1).tolist()} | commonOut={common:.1f}mm"
            )
            return motion, "OK"

        def _v85_execute_initial_spread_parallel_press(self, plan, cfg, args, move_command):
            module = self.d55
            keys = [k for k in ("arm2", "arm1") if k in plan.arm_points]
            if len(keys) != 2 or any(k not in self.arms for k in keys):
                return False
            if bool(getattr(cfg, "dry_run", False)) or not bool(getattr(cfg, "send", True)):
                return True

            close_fn = getattr(module, "_d55v3_force_closed_before_press", None)
            wait_fn = getattr(module, "_d26v3_stage_wait", None)
            angle = float(getattr(args, "d26_press_gripper", 3.05))
            if callable(close_fn):
                ok_close, close_reason = close_fn(keys, self.arms, angle, args)
                if not ok_close:
                    print(f"[INITIAL-SPREAD-V85-CLOSE-BLOCK] {close_reason}")
                    return False
            else:
                errors = []
                def close_one(k):
                    try:
                        self.arms[k].set_gripper(max(angle, 3.14), delay=0.12)
                    except Exception as exc:
                        errors.append((k, repr(exc)))
                threads = [threading.Thread(target=close_one, args=(k,), daemon=True) for k in keys]
                for th in threads:
                    th.start()
                for th in threads:
                    th.join(timeout=1.5)
                if errors:
                    print(f"[INITIAL-SPREAD-V85-CLOSE-ERROR] {errors}")
                    return False

            speeds = {
                "hover": float(self._v85_initial_spread_hover_speed),
                "down": float(self._v85_initial_spread_down_speed),
                "mid": float(self._v85_initial_spread_sweep_speed),
                "sweep": float(self._v85_initial_spread_sweep_speed),
                "lift": float(self._v85_initial_spread_lift_speed),
            }
            labels = {
                "hover": "HOVER", "down": "PRESS-DOWN", "mid": "PRESS-MID",
                "sweep": "PRESS-SWEEP", "lift": "PRESS-LIFT",
            }
            errors = []

            def command_stage(pose_key):
                stage_errors = []
                def move_one(k):
                    try:
                        xyz = plan.arm_points[k][pose_key]
                        print(
                            f"[INITIAL-SPREAD-V85-{labels[pose_key]}] {k.upper()} "
                            f"({float(xyz[0]):.1f},{float(xyz[1]):.1f},{float(xyz[2]):.1f}) speed={speeds[pose_key]:.2f}"
                        )
                        self.arms[k].move_goal(
                            float(xyz[0]), float(xyz[1]), float(xyz[2]),
                            float(getattr(cfg, "tool_t_rad", 3.14)), speeds[pose_key],
                            move_command=int(move_command),
                        )
                    except Exception as exc:
                        stage_errors.append((k, repr(exc)))
                threads = [threading.Thread(target=move_one, args=(k,), daemon=True) for k in keys]
                for th in threads:
                    th.start()
                for th in threads:
                    th.join(timeout=2.0)
                if stage_errors:
                    errors.extend(stage_errors)
                    return False
                return True

            def wait_between(prev_key, next_key, speed, extra, return_move=False):
                if callable(wait_fn):
                    waits = []
                    for k in keys:
                        waits.append(float(wait_fn(
                            plan.arm_points[k][prev_key], plan.arm_points[k][next_key],
                            float(speed), args, float(extra), return_move=bool(return_move),
                        )))
                    time.sleep(max(waits))
                else:
                    time.sleep(0.55 if not return_move else 0.40)

            print(
                "[INITIAL-SPREAD-V85-SYNC] both arms command every stage in parallel; "
                "normal PRESS_SWEEP executor is untouched"
            )
            if not command_stage("hover"):
                return False
            time.sleep(max(0.45, float(getattr(args, "d23_high_z_wait_s", 0.55))))
            if bool(getattr(cfg, "hover_only", False)):
                if not command_stage("lift"):
                    return False
                time.sleep(0.45)
                return True
            if not command_stage("down"):
                return False
            wait_between("hover", "down", speeds["down"], float(getattr(args, "d16_down_extra_wait_s", 0.30)))
            if not command_stage("mid"):
                return False
            wait_between("down", "mid", speeds["mid"], float(getattr(args, "d16_pull_extra_wait_s", 0.35)))
            if not command_stage("sweep"):
                return False
            wait_between("mid", "sweep", speeds["sweep"], float(getattr(args, "d16_pull_extra_wait_s", 0.35)))
            if not command_stage("lift"):
                return False
            wait_between("sweep", "lift", speeds["lift"], float(getattr(args, "d16_lift_extra_wait_s", 0.55)), return_move=True)
            if errors:
                print(f"[INITIAL-SPREAD-V85-ERROR] {errors}")
                return False
            print("[INITIAL-SPREAD-V85-SYNC] completed; both grippers remain closed")
            return True

        def _v82_execute_initial_spread_press_sweep(self, motion):
            if motion is None or bool(motion.get("skip", False)):
                return False, "INITIAL_SPREAD_V85_MOTION_MISSING"
            try:
                source = {k: np.asarray(motion["source_board"][k], np.float64).reshape(2) for k in ("arm2", "arm1")}
                target = {k: np.asarray(motion["target_board"][k], np.float64).reshape(2) for k in ("arm2", "arm1")}
            except Exception as exc:
                return False, f"INITIAL_SPREAD_V85_POINTS_INVALID:{exc!r}"
            if any(k not in self.arms for k in ("arm2", "arm1")):
                return False, "INITIAL_SPREAD_V85_BOTH_ARMS_REQUIRED"
            cfg = self.cfg55
            args = self.d55_args
            module = self.d55
            pose_fn = getattr(module, "_roarm_pose_for_board", None)
            contact_fn = getattr(module, "arm_contact_z", None)
            plan_cls = getattr(module, "DualWrinkleStretchPlan", None)
            if not callable(pose_fn) or not callable(contact_fn) or plan_cls is None:
                return False, "INITIAL_SPREAD_V85_D55_PRESS_API_MISSING"
            clear_z = float(getattr(args, "d47_prespread_clear_z", getattr(args, "d31_hover_z", getattr(cfg, "hover_z", 180.0))))
            press_z_offset = float(getattr(args, "d47_prespread_press_z_offset_mm", 12.0))
            arm_points = {}
            for k in ("arm2", "arm1"):
                mid = source[k] + 0.52 * (target[k] - source[k])
                press_z = float(contact_fn(cfg, k) + press_z_offset)
                arm_points[k] = {
                    "role": "initial_spread_d55_closed_press_outward",
                    "detected_source_board": tuple(float(v) for v in source[k]),
                    "source_board": tuple(float(v) for v in source[k]),
                    "target_board": tuple(float(v) for v in target[k]),
                    "hover": pose_fn(self.config, cfg, k, source[k], clear_z),
                    "down": pose_fn(self.config, cfg, k, source[k], press_z),
                    "mid": pose_fn(self.config, cfg, k, mid, press_z),
                    "sweep": pose_fn(self.config, cfg, k, target[k], press_z),
                    "lift": pose_fn(self.config, cfg, k, target[k], clear_z),
                }
            center = 0.5 * (source["arm2"] + source["arm1"])
            plan = plan_cls(
                True,
                "INITIAL_SPREAD V85 synchronized D55-style dual closed Press-Sweep",
                action="D47_OCCUPANCY_PRESPREAD_DUAL",
                candidate_index=-85,
                center_board=(float(center[0]), float(center[1])),
                arm_points=arm_points,
                metrics={
                    "policy_family": "INITIAL_SPREAD_D55_PRESS_SWEEP",
                    "d22_motion_mode": "PRESS_SWEEP",
                    "d26_no_grip_open": True,
                    "d47_dual_only": True,
                    "active_arm": "both",
                    "v85_initial_spread": True,
                    "v85_parallel_down": True,
                },
            )
            print(
                f"[INITIAL-SPREAD-V85-PRESS] synchronized D55-style Press-Sweep | "
                f"A2 {np.round(source['arm2'],1).tolist()}->{np.round(target['arm2'],1).tolist()} | "
                f"A1 {np.round(source['arm1'],1).tolist()}->{np.round(target['arm1'],1).tolist()}"
            )
            move_command = int(getattr(args, "move_command", 104))
            ok = bool(self._v85_execute_initial_spread_parallel_press(plan, cfg, args, move_command))
            return ok, "ACTION_OK" if ok else "INITIAL_SPREAD_V85_PARALLEL_PRESS_FAILED"

        def _v83_return_initial_spread_standby_verified(self):
            if str(getattr(self.args, "mode", "physical")) != "physical":
                return True
            with self.state_lock:
                self.status = "INITIAL SPREAD: RETURNING BOTH ARMS TO STANDBY"
            move_command = int(getattr(self.args, "move_command", 104))
            attempts = max(2, int(getattr(self.args, "standby_verify_retries", 2)))
            for attempt in range(1, attempts + 1):
                moved = self.d56.move_arms_to_standby(
                    self.arms, self.cfg56, move_command=move_command,
                    reason=f"INITIAL_SPREAD_V85_STANDBY_{attempt}",
                )
                if not moved:
                    print(f"[INITIAL-SPREAD-V85-STANDBY] command failed attempt={attempt}/{attempts}")
                time.sleep(float(self._v84_standby_first_poll_s))
                if self._feedback_camera_clear():
                    print(f"[INITIAL-SPREAD-V85-STANDBY] VERIFIED attempt={attempt}/{attempts}")
                    return True
                print(f"[INITIAL-SPREAD-V85-STANDBY] verify failed attempt={attempt}/{attempts}; retry")
                if attempt < attempts:
                    time.sleep(float(self._v84_standby_retry_poll_s))
            print("[INITIAL-SPREAD-V85-STANDBY-FAILED] both arms not camera-clear; AUTO-JUDGE blocked")
            return False

        def _v78_run_initial_spread_after_basket(self):
            auto_judge_ready = True
            try:
                deadline = time.monotonic() + 6.0
                while time.monotonic() < deadline:
                    with self.state_lock:
                        active = bool(self.motion_busy or self.inference_busy)
                    if not active:
                        break
                    time.sleep(0.08)
                with self.state_lock:
                    self.status = "INITIAL SPREAD: DETECTING UPPER OUTER MASK"
                if not self._ensure_camera_clear("INITIAL_SPREAD_BEFORE_SNAPSHOT", allow_move=False):
                    print("[INITIAL-SPREAD-V78-SKIP] camera-clear not verified")
                    return
                bundle = self._capture_i_frame_from_live("D56_WAIST_LIFT_LAYDOWN")
                obs = self._infer_for_action("D56_WAIST_LIFT_LAYDOWN", bundle.corrected)
                motion, why = self._v78_build_initial_spread_plan(obs)
                if motion is None:
                    print(f"[INITIAL-SPREAD-V78-SKIP] {why}; continue normal pipeline")
                    return
                with self.state_lock:
                    self.status = "INITIAL SPREAD: PRESS TOP CORNERS OUTWARD"
                    self.motion_busy = True
                try:
                    ok, detail = self._v82_execute_initial_spread_press_sweep(motion)
                    if ok:
                        standby_ok = self._v83_return_initial_spread_standby_verified()
                        if not standby_ok:
                            ok = False
                            detail = "INITIAL_SPREAD_STANDBY_VERIFY_FAILED"
                            auto_judge_ready = False
                finally:
                    with self.state_lock:
                        self.motion_busy = False
                if ok:
                    self._v78_initial_spread_done = True
                    print(f"[INITIAL-SPREAD-V78-DONE] {detail}")
                    with self.state_lock:
                        self.status = "INITIAL SPREAD COMPLETE - PREPARING NEXT ACTION"
                else:
                    print(f"[INITIAL-SPREAD-V78-FAIL] {detail}; continue normal pipeline")
                    with self.state_lock:
                        if auto_judge_ready:
                            self.status = "INITIAL SPREAD SKIPPED/FAILED - PREPARING NEXT ACTION"
                        else:
                            self.status = "INITIAL SPREAD STANDBY FAILED - AUTO JUDGE BLOCKED"
            except Exception as exc:
                print(f"[INITIAL-SPREAD-V78-ERROR] {exc!r}; continue normal pipeline")
                with self.state_lock:
                    self.motion_busy = False
                    self.status = "INITIAL SPREAD ERROR - PREPARING NEXT ACTION"
            finally:
                self._v53_align_initial_dual_done = False
                self._v78_initial_spread_busy = False
                self.auto_recommended = "REJUDGE"
                if auto_judge_ready:
                    self._schedule_auto_judge("after_initial_spread", 0.25)
                else:
                    print("[INITIAL-SPREAD-V83-AUTO-JUDGE] BLOCKED because standby verification failed")

        def _finalize_basket_without_review(self, success, report):
            if not bool(success):
                try:
                    fb = self._basket_feedback(quiet=True)
                    plan = self._v52_basket_active_plan if isinstance(self._v52_basket_active_plan, dict) else {}
                    if fb is not None and all(k in fb for k in ("x", "y", "z", "t")):
                        z = float(fb["z"])
                        hover_z = float(plan.get("hover_z", z + float(self._v57_basket_emergency_raise_mm)))
                        if z < hover_z - 12.0:
                            self._v57_basket_raise_to_hover_same_xy("BASKET-V57-FAILURE-SAFE-VERTICAL-RAISE")
                        release_ok = False
                        try:
                            self._v52_basket_gentle_open_to(1.35, "BASKET-V57-FAILURE-OPEN-AFTER-RAISE")
                            rel_fb = super()._basket_verify_gripper(
                                1.35, max(0.18, float(self.front.basket_close_tolerance_rad)), 4,
                                max(0.25, float(self.front.basket_gripper_settle_s)),
                                "BASKET-V57-FAILURE-OPEN-VERIFY",
                            )
                            release_ok = rel_fb is not None
                        except Exception as exc:
                            print(f"[BASKET-V57-FAILURE-OPEN-WARN] release still unverified after safe vertical raise: {exc}")
                        if release_ok:
                            try:
                                self._basket_return_arm2_standby(True)
                                print("[BASKET-V57-FAILURE-RETURN] release verified at safe height; ARM2 returned to standby")
                            except Exception as exc:
                                print(f"[BASKET-V57-FAILURE-RETURN-WARN] standby return failed after safe vertical raise: {exc}")
                        else:
                            print("[BASKET-V57-FAILURE-HOLD] ARM2 left at safe raised same-XY position; no lateral standby transit without verified release")
                except Exception as exc:
                    print(f"[BASKET-V57-FAILURE-RECOVERY-WARN] emergency vertical recovery failed: {exc}")
            super()._finalize_basket_without_review(success, report)
            self._v53_align_initial_dual_done = False
            self.align_phase = "DUAL_WAIST_TOP"
            self.align_correction_count = 0
            self._v51_post_basket_position_pending = True
            print(f"[D58-V51-POST-BASKET] armed for next POSITION; basketSuccess={bool(success)}")
            if success:
                self.auto_recommended = "REJUDGE"
                self._v78_initial_spread_busy = True
                self._v78_initial_spread_done = False
                with self.state_lock:
                    self.status = "BASKET DONE - INITIAL SPREAD PREPARING"
                print("[INITIAL-SPREAD-V78-ARMED] basket success + standby complete -> one automatic top-edge spread")
                threading.Thread(target=self._v78_run_initial_spread_after_basket, name="bottom-vla-initial-spread-v78", daemon=True).start()

        _v62_v61_ranked_wrinkles = _v48_ranked_wrinkles
        _v62_v61_build_two_region_plan = _v47_build_two_region_plan
        _v62_v61_basket_verify_gripper = _basket_verify_gripper

        def _v54_obs_with_refined_waist_uncached(self, obs):
            consistency = self._v53_pose_waist_mask_consistency(obs)
            ref = self._v54_waist_corner_refinement(obs)
            near_count = int(consistency.get("near_count", 0) or 0)
            reports = list(consistency.get("points", []) or [])

            if (not bool(consistency.get("ok", False))) or near_count < int(self._v66_waist_min_edge_points):
                out = dict(ref or {})
                out.update({
                    "ok": False,
                    "reason": "V66_FORMED_WAIST_EDGE_SUPPORT_FAILED",
                    "v66_original_consistency": copy.deepcopy(consistency),
                    "v66_required_edge_points": int(self._v66_waist_min_edge_points),
                })
                print(
                    f"[D60-V67-WAIST-FINAL-CHECK] REJECT edgeSupport={near_count}/3 "
                    f"reason={consistency.get('reason')}"
                )
                return obs, out

            pts = self._pose_board_points(obs)
            wl = self._finite_point2(pts.get("waist_img_left"))
            wc = self._finite_point2(pts.get("waist_center"))
            wr = self._finite_point2(pts.get("waist_img_right"))
            if wl is None or wr is None:
                return obs, {"ok": False, "reason": "V66_FORMED_WAIST_POINTS_MISSING"}
            if wc is None:
                wc = 0.5 * (wl + wr)

            # 2/3 is not an unconditional pass.  The third point must still be inside
            # the real garment and reasonably close to its exterior, and an independent
            # outer-edge estimate must exist and agree with the formed waist.
            conditional_two_of_three = near_count == 2
            if conditional_two_of_three:
                depths = []
                all_inside = len(reports) >= 3
                for rep in reports[:3]:
                    all_inside = all_inside and bool(rep.get("ok", False))
                    d = rep.get("edge_depth_mm")
                    if d is None or not math.isfinite(float(d)):
                        all_inside = False
                    else:
                        depths.append(float(d))
                max_depth = max(depths) if len(depths) == 3 else float("inf")
                if (not all_inside) or max_depth > float(self._v66_waist_conditional_max_edge_mm):
                    out = dict(ref or {})
                    out.update({
                        "ok": False,
                        "reason": "V66_FORMED_WAIST_2OF3_THIRD_POINT_TOO_FAR",
                        "v66_original_consistency": copy.deepcopy(consistency),
                        "v66_max_edge_depth_mm": None if not math.isfinite(max_depth) else float(max_depth),
                    })
                    print(
                        f"[D60-V67-WAIST-FINAL-CHECK] REJECT conditional 2/3: "
                        f"third/all maxEdge={'NA' if not math.isfinite(max_depth) else f'{max_depth:.1f}'}mm "
                        f"> {float(self._v66_waist_conditional_max_edge_mm):.1f}mm or outside real Mask"
                    )
                    return obs, out
                if not (isinstance(ref, dict) and bool(ref.get("ok", False))):
                    out = dict(ref or {})
                    out.update({
                        "ok": False,
                        "reason": "V66_FORMED_WAIST_2OF3_NEEDS_OUTER_SUPPORT",
                        "v66_original_consistency": copy.deepcopy(consistency),
                        "v66_max_edge_depth_mm": float(max_depth),
                    })
                    print(
                        f"[D60-V67-WAIST-FINAL-CHECK] REJECT conditional 2/3: independent outer-edge support unavailable "
                        f"reason={str((ref or {}).get('reason','UNAVAILABLE'))}"
                    )
                    return obs, out

            # When an independent outer-edge estimate is valid, it is a final sanity
            # check only; it does not replace the original Pose/Mask waist geometry.
            if isinstance(ref, dict) and bool(ref.get("ok", False)):
                rl = self._finite_point2(ref.get("waist_left"))
                rc = self._finite_point2(ref.get("waist_center"))
                rr = self._finite_point2(ref.get("waist_right"))
                if rl is not None and rr is not None and rc is not None:
                    a0 = math.degrees(math.atan2(float(wr[1] - wl[1]), float(wr[0] - wl[0])))
                    a1 = math.degrees(math.atan2(float(rr[1] - rl[1]), float(rr[0] - rl[0])))
                    angle_delta = abs((a0 - a1 + 180.0) % 360.0 - 180.0)
                    center_delta = float(np.linalg.norm(wc - rc))
                    if (
                        angle_delta > float(self._v63_waist_outer_max_angle_delta_deg)
                        or center_delta > float(self._v63_waist_outer_max_center_delta_mm)
                    ):
                        out = dict(ref)
                        out.update({
                            "ok": False,
                            "reason": "V66_FORMED_WAIST_NOT_SUPPORTED_BY_OUTER_EDGE",
                            "v66_outer_angle_delta_deg": float(angle_delta),
                            "v66_outer_center_delta_mm": float(center_delta),
                            "v66_original_consistency": copy.deepcopy(consistency),
                        })
                        print(
                            f"[D60-V67-WAIST-FINAL-CHECK] REJECT angleDelta={angle_delta:.1f}>"
                            f"{float(self._v63_waist_outer_max_angle_delta_deg):.1f} or centerDelta={center_delta:.1f}>"
                            f"{float(self._v63_waist_outer_max_center_delta_mm):.1f}"
                        )
                        return obs, out
                    out = dict(ref)
                    out.update({
                        "ok": True,
                        "reason": "V66_ORIGINAL_WAIST_FINAL_CHECK_PASS",
                        "v66_keep_original_waist": True,
                        "v66_edge_support_count": int(near_count),
                        "v66_conditional_2of3": bool(conditional_two_of_three),
                        "v66_outer_angle_delta_deg": float(angle_delta),
                        "v66_outer_center_delta_mm": float(center_delta),
                        "v66_original_consistency": copy.deepcopy(consistency),
                    })
                    print(
                        f"[D60-V67-WAIST-FINAL-CHECK] PASS edgeSupport={near_count}/3 "
                        f"angleDelta={angle_delta:.1f}deg centerDelta={center_delta:.1f}mm; keep formed waist"
                    )
                    return obs, out

            # A strong 3/3 real-edge match remains authoritative even if the auxiliary
            # outer-fit hypothesis itself is unavailable/rough. This avoids sending a
            # clearly formed waist into the very slow legacy mask-recovery cascade.
            if near_count >= 3:
                out = {
                    "ok": True,
                    "reason": "V66_STRONG_3OF3_KEEP_ORIGINAL_WAIST",
                    "v66_keep_original_waist": True,
                    "v66_edge_support_count": int(near_count),
                    "v66_outer_reference_reason": str((ref or {}).get("reason", "UNAVAILABLE")),
                    "v66_original_consistency": copy.deepcopy(consistency),
                }
                print(
                    f"[D60-V67-WAIST-FINAL-CHECK] PASS strong 3/3 real-edge support; "
                    f"aux outer={str((ref or {}).get('reason','UNAVAILABLE'))}; keep formed waist"
                )
                return obs, out

            # Conditional 2/3 cannot reach here because it requires a valid agreeing ref.
            out = dict(ref or {})
            out.update({"ok": False, "reason": "V66_WAIST_FINAL_VALIDATION_UNRESOLVED"})
            return obs, out

        _v80_v79_obs_with_refined_waist_uncached = _v54_obs_with_refined_waist_uncached

        def _v80_obs_from_outer_waist_ref(self, obs, ref, source):
            try:
                obs2 = copy.copy(obs)
                pose = copy.copy(getattr(obs, "pose"))
                kb = dict(getattr(pose, "keypoints_board", {}) or {})
                kp = dict(getattr(pose, "keypoints_px", {}) or {})
                mapping = {
                    "waist_img_left": np.asarray(ref["waist_left"], np.float64),
                    "waist_center": np.asarray(ref["waist_center"], np.float64),
                    "waist_img_right": np.asarray(ref["waist_right"], np.float64),
                }
                for name, point in mapping.items():
                    kb[name] = (float(point[0]), float(point[1]))
                    try:
                        px = self.d56.board_to_pixel(self.H_raw, float(point[0]), float(point[1]))
                        if px is not None:
                            kp[name] = (float(px[0]), float(px[1]))
                    except Exception:
                        pass
                pose.keypoints_board = kb
                if kp:
                    pose.keypoints_px = kp
                pose.waist_left = np.asarray(mapping["waist_img_left"], np.float32)
                pose.waist_center = np.asarray(mapping["waist_center"], np.float32)
                pose.waist_right = np.asarray(mapping["waist_img_right"], np.float32)
                width = float(np.linalg.norm(mapping["waist_img_right"] - mapping["waist_img_left"]))
                if hasattr(pose, "waist_width_mm"):
                    pose.waist_width_mm = width
                if hasattr(pose, "waist_angle_deg"):
                    dv = mapping["waist_img_right"] - mapping["waist_img_left"]
                    pose.waist_angle_deg = float(math.degrees(math.atan2(float(dv[1]), float(dv[0]))))
                obs2.pose = pose
                setattr(obs2, "bottom_vla_v80_waist_outer_source", str(source))
                return obs2
            except Exception as exc:
                print(f"[D60-V80-WAIST-TOP-WARN] refined obs build failed: {exc!r}")
                return None

        def _v80_pose_local_top_outer_ref(self, obs):
            result = {"ok": False, "reason": "V80_LOCAL_TOP_OUTER_UNAVAILABLE"}
            pts = self._pose_board_points(obs)
            wl = self._finite_point2(pts.get("waist_img_left"))
            wc = self._finite_point2(pts.get("waist_center"))
            wr = self._finite_point2(pts.get("waist_img_right"))
            crotch = self._finite_point2(pts.get("crotch"))
            if wl is None or wr is None or crotch is None:
                return result
            if wc is None:
                wc = 0.5 * (wl + wr)
            wv = np.asarray(wr - wl, np.float64)
            width = float(np.linalg.norm(wv))
            bv = np.asarray(crotch - wc, np.float64)
            bl = float(np.linalg.norm(bv))
            if width < 80.0 or bl < 20.0:
                result["reason"] = "V80_POSE_WAIST_GEOMETRY_INVALID"
                return result
            body_u = bv / bl
            waist_u = wv / width
            _, contour = self._v53_board_contour(obs)
            if contour is None or len(contour) < 30:
                result["reason"] = "V80_REAL_MASK_CONTOUR_UNAVAILABLE"
                return result
            c = np.asarray(contour, np.float64).reshape(-1, 2)
            rel = c - wc[None, :]
            axial = rel @ body_u
            lateral = rel @ waist_u
            pose_lateral = {
                "left": float(np.dot(wl - wc, waist_u)),
                "center": 0.0,
                "right": float(np.dot(wr - wc, waist_u)),
            }
            picked = {}
            for name in ("left", "center", "right"):
                target_lat = pose_lateral[name]
                band = 55.0 if name != "center" else 70.0
                valid = (
                    (np.abs(lateral - target_lat) <= band)
                    & (axial >= -90.0)
                    & (axial <= 30.0)
                )
                ids = np.flatnonzero(valid)
                if len(ids) == 0:
                    result["reason"] = f"V80_{name.upper()}_TOP_EDGE_NOT_FOUND"
                    return result
                score = axial[ids] + 0.08 * np.abs(lateral[ids] - target_lat)
                picked[name] = np.asarray(c[int(ids[int(np.argmin(score))])], np.float64)
            left = picked["left"]
            center = picked["center"]
            right = picked["right"]
            if float(np.dot(right-left, waist_u)) < 0.0:
                left, right = right, left
            out_width = float(np.linalg.norm(right-left))
            if out_width < max(100.0, 0.55*width) or out_width > min(560.0, 1.65*width):
                result.update({"reason": "V80_TOP_OUTER_WIDTH_IMPLAUSIBLE", "refined_width_mm": out_width})
                return result
            body_shift = float(np.dot(center-wc, body_u))
            if body_shift > 30.0:
                result.update({"reason": "V80_TOP_OUTER_TOO_FAR_DOWN_BODY", "body_shift_mm": body_shift})
                return result
            result.update({
                "ok": True,
                "reason": "V80_POSE_LOCAL_REAL_MASK_TOP_OUTER",
                "waist_left": left,
                "waist_center": center,
                "waist_right": right,
                "pose_width_mm": width,
                "refined_width_mm": out_width,
                "body_shift_mm": body_shift,
            })
            return result

        def _v54_obs_with_refined_waist_uncached(self, obs):
            strict = self._v54_waist_corner_refinement(obs)
            if isinstance(strict, dict) and bool(strict.get("ok", False)):
                obs2 = self._v80_obs_from_outer_waist_ref(obs, strict, "V80_STRICT_REAL_MASK_TOP_OUTER")
                if obs2 is not None:
                    check = self._v53_pose_waist_mask_consistency(obs2)
                    if bool(check.get("ok", False)):
                        out = dict(strict)
                        out.update({"ok": True, "reason": "V80_STRICT_REAL_MASK_TOP_OUTER", "v80_edge_check": copy.deepcopy(check)})
                        print(
                            f"[D60-V80-WAIST-TOP] strict outer edge USED width={float(strict.get('refined_width_mm',0.0)):.1f}mm "
                            f"bodyShift={float(strict.get('body_shift_mm',0.0)):+.1f}mm; lower MASK_WIDTH_CONTINUITY fallback bypassed"
                        )
                        return obs2, out
            local = self._v80_pose_local_top_outer_ref(obs)
            if bool(local.get("ok", False)):
                obs2 = self._v80_obs_from_outer_waist_ref(obs, local, "V80_POSE_LOCAL_REAL_MASK_TOP_OUTER")
                if obs2 is not None:
                    check = self._v53_pose_waist_mask_consistency(obs2)
                    if bool(check.get("ok", False)):
                        out = dict(local)
                        out["v80_edge_check"] = copy.deepcopy(check)
                        print(
                            f"[D60-V80-WAIST-TOP-RESCUE] local real-Mask top edge USED width={float(local.get('refined_width_mm',0.0)):.1f}mm "
                            f"bodyShift={float(local.get('body_shift_mm',0.0)):+.1f}mm; do not drop waist line into body-width fallback"
                        )
                        return obs2, out
            return self._v80_v79_obs_with_refined_waist_uncached(obs)

        def _v66_press_gap_background_cache(self, obs):
            key = id(obs)
            if key == self._v66_press_gap_bg_cache_id and self._v66_press_gap_bg_cache is not None:
                return self._v66_press_gap_bg_cache
            mask_obj = getattr(obs, "mask", None)
            mask_u8 = getattr(mask_obj, "mask_u8", None) if mask_obj is not None else None
            if mask_u8 is None:
                return None
            try:
                mask = (np.asarray(mask_u8) > 0).astype(np.uint8)
                if mask.ndim != 2 or int(np.count_nonzero(mask)) < 600:
                    return None
                bg = (mask == 0).astype(np.uint8)
                count, labels = self.cv2.connectedComponents(bg, connectivity=8)
                external = set()
                if count > 1:
                    border = np.concatenate((labels[0, :], labels[-1, :], labels[:, 0], labels[:, -1]))
                    external = {int(v) for v in np.unique(border) if int(v) != 0}
                kernel = self.cv2.getStructuringElement(self.cv2.MORPH_ELLIPSE, (3, 3))
                eroded = self.cv2.erode(mask, kernel, iterations=1)
                boundary = ((mask > 0) & (eroded == 0)).astype(np.uint8)
                radius = max(1, int(round(float(self._v66_press_gap_boundary_near_px) / 2.0)))
                boundary_near = self.cv2.dilate(boundary, kernel, iterations=radius)
                out = {
                    "mask": mask,
                    "labels": labels,
                    "external_labels": external,
                    "boundary_near": boundary_near,
                }
                self._v66_press_gap_bg_cache_id = key
                self._v66_press_gap_bg_cache = out
                return out
            except Exception as exc:
                print(f"[D55-V66-GAP-CACHE-WARN] {type(exc).__name__}: {exc}")
                return None

        def _v66_press_interleg_corridor(self, obs, point_board):
            report = {"available": False, "inside": False, "t": None, "lateral_mm": None, "half_width_mm": None}
            p = self._finite_point2(point_board)
            if p is None:
                return report
            pts = self._pose_board_points(obs)
            crotch = self._finite_point2(pts.get("crotch"))

            def pair_center(a_name, b_name):
                a = self._finite_point2(pts.get(a_name))
                b = self._finite_point2(pts.get(b_name))
                if a is not None and b is not None:
                    return 0.5 * (a + b)
                return a if a is not None else b

            left_hem = pair_center("img_left_hem_outer", "img_left_hem_inner")
            right_hem = pair_center("img_right_hem_inner", "img_right_hem_outer")
            if crotch is None or left_hem is None or right_hem is None:
                return report
            mid_hem = 0.5 * (left_hem + right_hem)
            axis = np.asarray(mid_hem - crotch, np.float64)
            axis_len = float(np.linalg.norm(axis))
            hem_sep = float(np.linalg.norm(right_hem - left_hem))
            if not math.isfinite(axis_len) or axis_len < 80.0 or not math.isfinite(hem_sep) or hem_sep < 45.0:
                return report
            axis_u = axis / axis_len
            rel = np.asarray(p - crotch, np.float64)
            t = float(np.dot(rel, axis_u) / axis_len)
            nearest = crotch + axis * float(np.clip(t, 0.0, 1.0))
            lateral = float(np.linalg.norm(p - nearest))
            half_width = float(np.clip(
                float(self._v66_press_gap_corridor_ratio) * hem_sep,
                float(self._v66_press_gap_corridor_min_mm),
                float(self._v66_press_gap_corridor_max_mm),
            ))
            inside = bool(-0.05 <= t <= 1.12 and lateral <= half_width)
            report.update({
                "available": True,
                "inside": inside,
                "t": float(t),
                "lateral_mm": float(lateral),
                "half_width_mm": float(half_width),
                "axis_u": axis_u,
                "hem_separation_mm": float(hem_sep),
            })
            return report

        def _v66_press_interleg_gap_report(self, obs, cand, heat, H):
            report = {
                "available": False,
                "reject": False,
                "reason": "UNAVAILABLE",
                "overlap_exception": False,
            }
            if H is None or not isinstance(cand, dict):
                return report
            cache = self._v66_press_gap_background_cache(obs)
            ridge = self._v47_ridge_frame(cand, heat, H)
            comp = self._v40_candidate_component(cand, heat)
            if cache is None or ridge is None or comp is None:
                return report
            try:
                mask = cache["mask"]
                labels = cache["labels"]
                external_labels = cache["external_labels"]
                boundary_near = cache["boundary_near"]
                h, w = mask.shape[:2]

                xx = np.asarray(comp.get("xx", []), np.float64)
                yy = np.asarray(comp.get("yy", []), np.float64)
                if len(xx) < 12:
                    return report
                step = max(1, int(math.ceil(len(xx) / 240.0)))
                xi = np.clip(np.rint(xx[::step]).astype(np.int32), 0, w - 1)
                yi = np.clip(np.rint(yy[::step]).astype(np.int32), 0, h - 1)
                boundary_fraction = float(np.mean(boundary_near[yi, xi] > 0)) if len(xi) else 0.0

                corridor = self._v66_press_interleg_corridor(obs, ridge["center_board"])
                if not bool(corridor.get("available", False)):
                    report.update({
                        "available": True,
                        "reason": "POSE_INTERLEG_CORRIDOR_UNAVAILABLE_KEEP",
                        "boundary_fraction": boundary_fraction,
                    })
                    return report
                axis_u = np.asarray(corridor.get("axis_u"), np.float64).reshape(2)
                tangent = np.asarray(ridge["tangent_board"], np.float64).reshape(2)
                tangent_body = abs(float(np.dot(tangent, axis_u)))

                center = np.asarray(ridge["center_board"], np.float64).reshape(2)
                normal = np.asarray(ridge["normal_board"], np.float64).reshape(2)
                span = float(ridge.get("span_mm", 40.0))
                anchors = [center + tangent * (float(frac) * 0.70 * span) for frac in (-0.40, -0.20, 0.0, 0.20, 0.40)]
                distances = tuple(float(v) for v in self._v66_press_gap_side_distances_mm)
                side_stats = {}
                for sign, side_name in ((-1.0, "neg"), (1.0, "pos")):
                    valid = 0
                    garment = 0
                    bg_external = 0
                    bg_total = 0
                    for anchor in anchors:
                        for dist in distances:
                            q = anchor + normal * (sign * dist)
                            px = self.d56.board_to_pixel(H, float(q[0]), float(q[1]))
                            if px is None:
                                continue
                            u = int(round(float(px[0])))
                            v = int(round(float(px[1])))
                            if not (0 <= u < w and 0 <= v < h):
                                continue
                            valid += 1
                            if mask[v, u] > 0:
                                garment += 1
                            else:
                                bg_total += 1
                                lab = int(labels[v, u])
                                if lab in external_labels:
                                    bg_external += 1
                    garment_ratio = float(garment / max(valid, 1))
                    external_ratio = float(bg_external / max(bg_total, 1)) if bg_total else 0.0
                    side_stats[side_name] = {
                        "valid": int(valid),
                        "garment_ratio": garment_ratio,
                        "background_samples": int(bg_total),
                        "external_background_ratio": external_ratio,
                    }

                if min(side_stats["neg"]["valid"], side_stats["pos"]["valid"]) < 6:
                    report.update({
                        "available": True,
                        "reason": "TOO_FEW_NORMAL_SIDE_SAMPLES_KEEP",
                        "boundary_fraction": boundary_fraction,
                        "corridor": corridor,
                        "tangent_body_alignment": tangent_body,
                        "sides": side_stats,
                    })
                    return report

                neg_g = float(side_stats["neg"]["garment_ratio"])
                pos_g = float(side_stats["pos"]["garment_ratio"])
                if neg_g <= pos_g:
                    bg_side = "neg"; garment_side = "pos"
                else:
                    bg_side = "pos"; garment_side = "neg"
                bg_g = float(side_stats[bg_side]["garment_ratio"])
                cloth_g = float(side_stats[garment_side]["garment_ratio"])
                external_ratio = float(side_stats[bg_side]["external_background_ratio"])

                one_sided = bool(
                    cloth_g >= float(self._v66_press_gap_garment_side_min)
                    and bg_g <= float(self._v66_press_gap_background_side_max)
                )
                contour_like = bool(boundary_fraction >= float(self._v66_press_gap_boundary_fraction_min))
                exterior_connected = bool(external_ratio >= float(self._v66_press_gap_external_ratio_min))
                in_interleg = bool(corridor.get("inside", False))
                body_aligned = bool(tangent_body >= float(self._v66_press_gap_tangent_body_min))

                # Reject only if all independent gap cues agree.  An internal overlap/fold
                # generally fails contour_like and/or exterior_connected, so it is kept.
                reject = bool(one_sided and contour_like and exterior_connected and in_interleg and body_aligned)
                overlap_exception = bool((not contour_like) or (not exterior_connected) or (not one_sided))
                report.update({
                    "available": True,
                    "reject": reject,
                    "reason": "INTER_LEG_EXTERIOR_GAP_BOUNDARY" if reject else "KEEP_WRINKLE_OR_OVERLAP",
                    "overlap_exception": overlap_exception,
                    "boundary_fraction": float(boundary_fraction),
                    "corridor": corridor,
                    "tangent_body_alignment": float(tangent_body),
                    "background_side": bg_side,
                    "garment_side": garment_side,
                    "background_side_garment_ratio": float(bg_g),
                    "garment_side_garment_ratio": float(cloth_g),
                    "external_background_ratio": float(external_ratio),
                    "sides": side_stats,
                })
                return report
            except Exception as exc:
                report.update({"available": False, "reject": False, "reason": f"ERROR:{type(exc).__name__}"})
                print(f"[D55-V66-GAP-WARN] {type(exc).__name__}: {exc}")
                return report

        def _v48_ranked_wrinkles(self, obs, heat, H, shape_state):
            if self._v67_prepare_cancel_requested():
                print("[D55-V69-CANCEL] abort before validated ranking")
                return []
            ranked=list(self._v62_v61_ranked_wrinkles(obs,heat,H,shape_state) or [])
            if self._v67_prepare_cancel_requested():
                print("[D55-V69-CANCEL] abort after validated ranking")
                return []
            filtered=[]
            for row in ranked:
                if self._v67_prepare_cancel_requested():
                    print("[D55-V69-CANCEL] abort gap filtering")
                    return []
                try:
                    priority,major,size_score,idx,cand,size,ctx=row
                    gap=self._v66_press_interleg_gap_report(obs,cand,heat,H)
                    cand["v66_interleg_gap_report"]=copy.deepcopy(gap)
                    ctx=dict(ctx); ctx["v66_interleg_gap_report"]=copy.deepcopy(gap)
                    row=(priority,major,size_score,idx,cand,size,ctx)
                    if bool(gap.get("reject",False)):
                        corr=gap.get("corridor",{}) if isinstance(gap.get("corridor"),dict) else {}
                        print(
                            f"[D55-V66-GAP-REJECT] candidate={int(idx)+1} reason=INTER_LEG_EXTERIOR_GAP_BOUNDARY "
                            f"boundary={float(gap.get('boundary_fraction',0.0)):.2f} extBg={float(gap.get('external_background_ratio',0.0)):.2f} "
                            f"cloth/bg={float(gap.get('garment_side_garment_ratio',0.0)):.2f}/{float(gap.get('background_side_garment_ratio',0.0)):.2f} "
                            f"corridorT={float(corr.get('t',0.0)):.2f} lat={float(corr.get('lateral_mm',0.0)):.1f}/{float(corr.get('half_width_mm',0.0)):.1f}mm "
                            f"bodyAlign={float(gap.get('tangent_body_alignment',0.0)):.2f}"
                        )
                        continue
                    if bool(gap.get("available",False)) and bool(gap.get("overlap_exception",False)):
                        print(
                            f"[D55-V66-GAP-KEEP] candidate={int(idx)+1} internal/overlap-compatible; "
                            f"boundary={float(gap.get('boundary_fraction',0.0)):.2f} extBg={float(gap.get('external_background_ratio',0.0)):.2f}"
                        )
                    filtered.append(row)
                except Exception as exc:
                    print(f"[D55-V66-GAP-FILTER-WARN] {type(exc).__name__}: {exc}; KEEP candidate")
                    filtered.append(row)
            ranked=filtered
            frames={}
            for row in ranked:
                if self._v67_prepare_cancel_requested():
                    print("[D55-V69-CANCEL] abort cluster frame build")
                    return []
                f=self._v47_ridge_frame(row[4],heat,H)
                if f is not None: frames[int(row[3])]=f
            rescored=[]
            for row in ranked:
                if self._v67_prepare_cancel_requested():
                    print("[D55-V69-CANCEL] abort cluster rescoring")
                    return []
                priority,major,size_score,idx,cand,size,ctx=row
                f=frames.get(int(idx)); members=[]; extent=0.0
                if f is not None:
                    c0=np.asarray(f["center_board"],np.float64); n0=np.asarray(f["normal_board"],np.float64); t0=np.asarray(f["tangent_board"],np.float64)
                    for other in ranked:
                        if self._v67_prepare_cancel_requested():
                            print("[D55-V69-CANCEL] abort cluster neighbor scan")
                            return []
                        leg0=str((ctx or {}).get("leg_region","UNKNOWN"))
                        legj=str(((other[6] if len(other)>6 else {}) or {}).get("leg_region","UNKNOWN"))
                        if leg0 in {"LEFT_LEG","RIGHT_LEG"} and legj in {"LEFT_LEG","RIGHT_LEG"} and leg0 != legj:
                            continue
                        fj=frames.get(int(other[3]))
                        if fj is None: continue
                        cj=np.asarray(fj["center_board"],np.float64); d=cj-c0
                        if float(np.linalg.norm(d))>float(self._v62_press_cluster_radius_mm): continue
                        tj=np.asarray(fj["tangent_board"],np.float64)
                        ang=math.degrees(math.acos(float(np.clip(abs(np.dot(t0,tj)),0.0,1.0))))
                        if ang>50.0: continue
                        members.append(int(other[3])); extent=max(extent,abs(float(np.dot(d,n0))))
                    bonus=10.0*max(0,len(members)-1)+min(30.0,0.35*extent)
                    priority=float(priority)+bonus
                    cand["v62_cluster_members"]=members; cand["v62_cluster_normal_halfspan_mm"]=float(extent); cand["v62_cluster_bonus"]=float(bonus)
                    ctx=dict(ctx); ctx["v62_cluster_count"]=len(members); ctx["v62_cluster_bonus"]=float(bonus)
                rescored.append((float(priority),major,size_score,idx,cand,size,ctx))
            waist_line = None
            try:
                pose_pts = self._pose_board_points(obs)
                wlp = self._finite_point2(pose_pts.get("waist_img_left"))
                wrp = self._finite_point2(pose_pts.get("waist_img_right"))
                if wlp is not None and wrp is not None and float(np.linalg.norm(wrp - wlp)) >= 40.0:
                    waist_line = (np.asarray(wlp, np.float64), np.asarray(wrp, np.float64))
            except Exception:
                waist_line = None

            def _v101_structural_press_score(row):
                ctx = dict((row[6] if len(row) > 6 else {}) or {})
                size = dict((row[5] if len(row) > 5 else {}) or {})
                cand = row[4] if len(row) > 4 else {}
                crease = dict(ctx.get("v61_crease_evidence", {}) or {})
                crease_score = float(crease.get("score", 0.0) or 0.0) if bool(crease.get("available", False)) else 0.0
                severity = float(size.get("severity", 0.0) or 0.0)
                major = float(size.get("major_length_mm", 0.0) or 0.0)
                size_score = float(size.get("size_score", 0.0) or 0.0)
                asym = float(ctx.get("asymmetry_relevance", 0.0) or 0.0)
                cluster_n = len(cand.get("v62_cluster_members", []) or []) if isinstance(cand, dict) else 0
                length_term = float(np.clip(major / 120.0, 0.0, 1.35))
                structural = 72.0 * severity + 42.0 * length_term + 18.0 * asym + 0.12 * size_score + 4.0 * min(cluster_n, 4)
                if major < 45.0:
                    structural -= 34.0 * (1.0 - major / 45.0)
                waist_distance = None
                if waist_line is not None:
                    center = self._candidate_board_center(cand, H)
                    if center is not None:
                        a, b = waist_line
                        ab = b - a
                        denom = float(np.dot(ab, ab))
                        if denom > 1e-6:
                            t = float(np.clip(np.dot(center - a, ab) / denom, 0.0, 1.0))
                            near = a + t * ab
                            waist_distance = float(np.linalg.norm(center - near))
                            if waist_distance < 55.0 and major < 95.0:
                                structural -= 34.0 * (1.0 - waist_distance / 55.0) * (1.0 - min(major, 95.0) / 95.0)
                if isinstance(cand, dict):
                    cand["v101_structural_press_score"] = float(structural)
                    cand["v101_waist_distance_mm"] = waist_distance
                ctx["v101_structural_press_score"] = float(structural)
                ctx["v101_waist_distance_mm"] = waist_distance
                return (float(structural), severity, major, size_score, crease_score, float(row[0]))

            rescored.sort(key=_v101_structural_press_score, reverse=True)
            if rescored:
                top=rescored[0]
                top_ctx=dict((top[6] if len(top)>6 else {}) or {})
                top_crease=dict(top_ctx.get("v61_crease_evidence",{}) or {})
                top_struct=_v101_structural_press_score(top)[0]
                print(
                    f"[D55-V101-STRUCTURAL-FIRST] candidate={int(top[3])+1} structural={top_struct:.1f} "
                    f"len={float((top[5] or {}).get('major_length_mm',0.0)):.1f}mm "
                    f"severity={float((top[5] or {}).get('severity',0.0)):.2f} "
                    f"crease={float(top_crease.get('score',0.0)):.2f}; large shape-changing folds outrank tiny high-contrast lines"
                )
            leg_rows={"LEFT_LEG":[],"RIGHT_LEG":[]}
            for row in rescored:
                leg=str(((row[6] if len(row)>6 else {}) or {}).get("leg_region","UNKNOWN"))
                if leg in leg_rows:
                    leg_rows[leg].append(row)
            if not leg_rows["LEFT_LEG"] and not leg_rows["RIGHT_LEG"]:
                print("[D55-V72-LEG-SELECT] no reliable Pose leg assignment; preserve validated global ranking")
                return rescored
            def leg_score(rows):
                vals=sorted([float(r[0]) for r in rows],reverse=True)
                return (vals[0] if len(vals)>0 else 0.0)+(0.20*vals[1] if len(vals)>1 else 0.0)+(0.10*vals[2] if len(vals)>2 else 0.0)
            left_raw=leg_score(leg_rows["LEFT_LEG"])
            right_raw=leg_score(leg_rows["RIGHT_LEG"])
            right_eff=right_raw
            strongest_leg = str(((rescored[0][6] if rescored else {}) or {}).get("leg_region", "UNKNOWN"))
            if strongest_leg in {"LEFT_LEG", "RIGHT_LEG"} and leg_rows.get(strongest_leg):
                selected = strongest_leg
            elif leg_rows["RIGHT_LEG"] and (not leg_rows["LEFT_LEG"] or right_raw >= left_raw):
                selected = "RIGHT_LEG"
            else:
                selected = "LEFT_LEG"
            selected_rows=[]
            for row in leg_rows[selected]:
                priority,major,size_score,idx,cand,size,ctx=row
                cand=copy.deepcopy(cand)
                ctx=dict(ctx or {})
                cand["v72_selected_leg"]=selected
                ctx["v72_selected_leg"]=selected
                ctx["v72_leg_score_left"]=float(left_raw)
                ctx["v72_leg_score_right_raw"]=float(right_raw)
                ctx["v72_leg_score_right_effective"]=float(right_eff)
                selected_rows.append((priority,major,size_score,idx,cand,size,ctx))
            print(
                f"[D55-V72-LEG-SELECT] left={left_raw:.1f} rightRaw={right_raw:.1f} "
                f"rightEff={right_eff:.1f} strongestLeg={strongest_leg} -> {selected} "
                f"candidates={len(selected_rows)}; opposite leg locked out for this PRESS decision"
            )
            return selected_rows

        def _v47_build_two_region_plan(self, obs, heat, H, config, cfg, args, cand, original_index, rank, size, midgeom):
            if self._v67_prepare_cancel_requested():
                return None
            plan=self._v62_v61_build_two_region_plan(obs,heat,H,config,cfg,args,cand,original_index,rank,size,midgeom)
            if self._v67_prepare_cancel_requested():
                return None
            if plan is None or not bool(getattr(plan,"ok",False)): return plan
            half=float(cand.get("v62_cluster_normal_halfspan_mm",0.0) or 0.0)
            if half<=1.0: return plan
            ridge=self._v47_ridge_frame(cand,heat,H)
            if ridge is None: return plan
            c=np.asarray(ridge["center_board"],np.float64); n=np.asarray(ridge["normal_board"],np.float64); req=half+float(self._v62_press_cluster_normal_margin_mm)
            aps=dict(getattr(plan,"arm_points",{}) or {}); dist=[]
            for k in ("arm1","arm2"):
                p=self._finite_point2((aps.get(k,{}) or {}).get("source_board"))
                if p is not None: dist.append(abs(float(np.dot(p-c,n))))
            if len(dist)==2 and min(dist)<req:
                print(f"[D55-V62-CLUSTER-PAIR-REJECT] candidate={original_index+1} sourceDistance={min(dist):.1f} < requiredOutside={req:.1f}")
                return None
            plan.metrics=dict(getattr(plan,"metrics",{}) or {}); plan.metrics.update({"d55v62_cluster_first":True,"d55v62_cluster_outside_required_mm":req})
            leg_lock = self._v70_press_leg_lock_report(plan, obs, ridge_center=c)
            plan.metrics["d55v70_leg_lock_cluster_wrapper"] = copy.deepcopy(leg_lock)
            if bool(leg_lock.get("enforced", False)) and not bool(leg_lock.get("ok", False)):
                print(f"[D55-V70-LEG-CLUSTER-REJECT] candidate={original_index+1} reason={leg_lock.get('reason')}")
                return None
            return plan

        def _basket_verify_gripper(self, target, tolerance, attempts, settle, stage):
            if str(stage)!="BASKET-CLOSE-VERIFY":
                return self._v62_v61_basket_verify_gripper(target,tolerance,attempts,settle,stage)
            try:
                return self._v62_v61_basket_verify_gripper(target,tolerance,attempts,settle,stage)
            except Exception as first_exc:
                fb=self._basket_feedback(quiet=True); actual=None if fb is None else fb.get("t")
                if actual is None or float(actual)<float(self._v62_basket_close_probe_min_rad): raise
                contact=copy.deepcopy(self._v52_basket_contact_feedback) if isinstance(self._v52_basket_contact_feedback,dict) else None
                for attempt in range(1,int(self._v62_basket_regrasp_attempts)+1):
                    fb0=self._basket_feedback(quiet=True)
                    if fb0 is None or not all(k in fb0 for k in ("x","y","z","t")): break
                    x,y,z,t0=(float(fb0[k]) for k in ("x","y","z","t")); pz=z+float(self._v62_basket_probe_lift_mm)
                    print(f"[BASKET-V62-RETENTION-PROBE] attempt={attempt} close={t0:.3f}; +{self._v62_basket_probe_lift_mm:.0f}mm same-XY lift")
                    self._basket_move_goal(x,y,pz,t0,min(0.45,float(self.front.basket_lift_speed)),f"BASKET-V62-PROBE-LIFT_{attempt}")
                    try:self._basket_wait_waypoint(f"BASKET-V62-PROBE-LIFT_{attempt}",(x,y,pz))
                    except Exception:pass
                    self._basket_set_gripper(float(target),0.0,f"BASKET-V62-PROBE-CLOSE_{attempt}"); time.sleep(max(0.35,float(settle)))
                    fb1=self._basket_feedback(quiet=True)
                    if fb1 is not None and "t" in fb1:
                        t1=float(fb1["t"]); stable=abs(t1-t0)<=float(self._v62_basket_probe_stable_span_rad)
                        if t1>=float(self._v62_basket_close_probe_min_rad) and stable:
                            print(f"[BASKET-V62-RETENTION-ACCEPT] cloth-blocked close stable after lift {t0:.3f}->{t1:.3f}; continue")
                            return fb1
                    if contact is not None:
                        cz=float(contact.get("z",z)); open_t=float(getattr(self.front,"basket_grip_fully_open",1.52))
                        self._basket_move_goal(x,y,cz,open_t,float(self.front.basket_slow_speed),f"BASKET-V62-REGRASP-RETURN_{attempt}")
                        try:self._basket_wait_waypoint(f"BASKET-V62-REGRASP-RETURN_{attempt}",(x,y,cz))
                        except Exception:pass
                        try:self._v52_basket_gentle_open_to(open_t,f"BASKET-V62-REGRASP-OPEN_{attempt}")
                        except Exception:pass
                raise RuntimeError(f"BASKET-V62 cloth-contact close failed retention probe/regrasp: {first_exc}") from first_exc

        def _v111_schedule_auto_execute_locked(self, reason="PLAN_READY"):
            if self._integration_stop_requested():
                return
            if not bool(getattr(self, "_v111_auto_execute", False)):
                return
            with self.state_lock:
                locked = self.locked
                generation = int(getattr(self, "_prepare_generation", 0))
                motion_busy = bool(self.motion_busy)
            if locked is None or motion_busy or not bool(getattr(locked, "plan_ok", False)):
                return
            action = str(getattr(locked, "action", "") or "").upper()
            if action in {"", "NONE", "FINISH", "BASKET_GRASP"}:
                return
            token = (id(locked), generation, action)
            with self._v111_autoexec_lock:
                if self._v111_autoexec_token == token:
                    return
                self._v111_autoexec_token = token
            with self.state_lock:
                self.status = f"{self.pending_semantic or action} PLAN READY - AUTO EXECUTE"
            print(f"[AUTO-EXEC-V114-QUEUED] action={action} generation={generation} reason={reason}")

            def worker():
                try:
                    deadline = time.monotonic() + 12.0
                    while time.monotonic() < deadline:
                        with self.state_lock:
                            same_lock = self.locked is locked
                            busy = bool(self.inference_busy or self.motion_busy)
                            same_generation = int(getattr(self, "_prepare_generation", 0)) == generation
                        if not same_lock or not same_generation:
                            print(f"[AUTO-EXEC-V114-CANCEL] action={action} lock/generation changed before execution")
                            return
                        if not busy:
                            break
                        time.sleep(0.03)
                    else:
                        print(f"[AUTO-EXEC-V114-CANCEL] action={action} inference/motion did not clear before timeout")
                        return
                    with self.state_lock:
                        if self.locked is not locked or not bool(getattr(locked, "plan_ok", False)) or self.motion_busy:
                            return
                    if self._integration_stop_requested():
                        print(f"[AUTO-EXEC-V114-CANCEL] action={action} stop requested before ENTER")
                        return
                    print(f"[AUTO-ENTER-V114-QUEUE] action={action} generation={generation}; queueing normal ENTER path")
                    self._queue("ENTER")
                except Exception as exc:
                    print(f"[AUTO-ENTER-V114-ERROR] action={action} generation={generation} error={type(exc).__name__}: {exc}")
                    traceback.print_exc()
                    with self._v111_autoexec_lock:
                        if self._v111_autoexec_token == token:
                            self._v111_autoexec_token = None

            threading.Thread(
                target=worker, name=f"bottom-vla-v114-autoenter-{action.lower()}-g{generation}", daemon=True
            ).start()

        def _start_execution(self):
            with self.state_lock:
                locked = self.locked
                was_motion_busy = bool(self.motion_busy)
                # Never leave a previous/planner action visible while a new
                # executor acceptance is being decided.
                self._integration_executing_action = None
            action0 = str(getattr(locked, "action", "") or "") if locked is not None else ""
            if action0 != "FINISH":
                self._v87_finish_cache = None
                self._v87_finish_requested = False

            with self.state_lock:
                locked = self.locked
            action = str(getattr(locked, "action", "") or "") if locked is not None else ""
            valid_plan = bool(locked is not None and bool(getattr(locked, "plan_ok", False)))
            consume_post_basket = bool(
                valid_plan
                and action == "D58_CIRC_POSITION"
                and self._v51_post_basket_position_pending
            )
            clear_for_other_motion = bool(
                valid_plan
                and action not in {"D58_CIRC_POSITION", "BASKET_GRASP", "FINISH"}
                and self._v51_post_basket_position_pending
            )

            super()._start_execution()

            with self.state_lock:
                motion_busy_after = bool(self.motion_busy)
                locked_after = self.locked
            lock_consumed = bool(valid_plan and locked_after is None)
            accepted = bool(
                valid_plan
                and action != "FINISH"
                and not was_motion_busy
                and (motion_busy_after or lock_consumed)
            )

            if not accepted:
                if valid_plan and action != "FINISH":
                    print(
                        f"[V110-EXEC-NOT-STARTED] action={action} "
                        f"motionBusy={motion_busy_after} lockConsumed={lock_consumed}; "
                        "motion epoch/state unchanged"
                    )
                return

            self._v88_mark_motion_started(action)
            motion_epoch = int(getattr(self, "_v88_motion_epoch", 0))

            # V20: translate the exact locked.action accepted by the executor
            # into the public macro name.  This replaces the old selected_action
            # telemetry, which could still describe a planner/model candidate.
            _display_map = {
                "D58_CIRC_POSITION": "POSITION_ADJUST",
                "POSITION_ADJUST": "POSITION_ADJUST",
                "D55_PRESS_SWEEP": "PRESS_SWEEP",
                "PRESS_SWEEP": "PRESS_SWEEP",
                "D56_WAIST_LIFT_LAYDOWN": "WAIST_PULL_LAYDOWN",
                "WAIST_PULL_LAYDOWN": "WAIST_PULL_LAYDOWN",
                "ALIGN": "ALIGN",
            }
            _actual_macro = _display_map.get(str(action or "").upper())
            with self.state_lock:
                self._integration_executing_action = _actual_macro
            print(
                f"[V110-EXEC-ACCEPT] action={action} epoch={motion_epoch} "
                f"motionBusy={motion_busy_after}"
            )
            print(
                f"[BOTTOM-ACTION-TELEMETRY-EXEC] lockedAction={action} "
                f"macro={_actual_macro or 'UNMAPPED'} epoch={motion_epoch}",
                flush=True,
            )
            if bool(getattr(self, "_v110_inference_only", False)):
                print(
                    f"[INFERENCE-ONLY-V114-EXEC-ACCEPT] action={action} epoch={motion_epoch}; "
                    "normal ENTER handler will arm the post-motion fresh judge"
                )
            if self._v56_rejected_plans:
                print(
                    f"[REJUDGE-V56-RESET] approved motion started; "
                    f"clearing {len(self._v56_rejected_plans)} rejected-plan signatures"
                )
            self._v56_rejected_plans.clear()
            if consume_post_basket:
                self._v51_post_basket_position_pending = False
                print("[D58-V51-POST-BASKET] first POSITION execution started; one-shot BOTTOM+RIGHT override consumed")
            elif clear_for_other_motion:
                self._v51_post_basket_position_pending = False
                print(f"[D58-V51-POST-BASKET] cleared because {action} actually started before POSITION")


        _v81_prev_force_waist_endpoint_plan = _d60_force_waist_endpoint_plan
        _v81_prev_d60_plan_from_obs = _d60_plan_from_obs
        _v81_prev_waist_endpoint_gate = _d60_waist_endpoint_gate

        def _v87_mask_waist_curve_ref(self, obs):
            result = {"ok": False, "reason": "V87_MASK_WAIST_CURVE_UNAVAILABLE"}
            pts = self._pose_board_points(obs)
            wl = self._finite_point2(pts.get("waist_img_left"))
            wc = self._finite_point2(pts.get("waist_center"))
            wr = self._finite_point2(pts.get("waist_img_right"))
            crotch = self._finite_point2(pts.get("crotch"))
            if wl is None or wr is None or crotch is None:
                result["reason"] = "V87_POSE_WAIST_ANCHOR_UNAVAILABLE"
                return result
            if wc is None:
                wc = 0.5 * (wl + wr)
            waist_vec = np.asarray(wr - wl, np.float64)
            body_vec = np.asarray(crotch - wc, np.float64)
            pose_width = float(np.linalg.norm(waist_vec))
            body_len = float(np.linalg.norm(body_vec))
            if pose_width < 55.0 or body_len < 20.0:
                result["reason"] = "V87_POSE_WAIST_ANCHOR_GEOMETRY_INVALID"
                return result
            waist_u = waist_vec / pose_width
            body_u = body_vec / body_len
            plus = self._v53_mask_point_report(obs, wc + body_u * 22.0)
            minus = self._v53_mask_point_report(obs, wc - body_u * 22.0)
            if not bool((plus or {}).get("ok", False)) and bool((minus or {}).get("ok", False)):
                body_u = -body_u
            _, contour = self._v53_board_contour(obs)
            if contour is None or len(contour) < 30:
                result["reason"] = "V87_REAL_MASK_CONTOUR_UNAVAILABLE"
                return result
            c = np.asarray(contour, np.float64).reshape(-1, 2)
            sl = float(np.dot(wl - wc, waist_u))
            sr = float(np.dot(wr - wc, waist_u))
            if sl > sr:
                waist_u = -waist_u
                sl = float(np.dot(wl - wc, waist_u))
                sr = float(np.dot(wr - wc, waist_u))
            span = max(55.0, float(sr - sl))
            rel = c - wc[None, :]
            sproj = rel @ waist_u
            dproj = rel @ body_u
            pad = float(np.clip(0.16 * span, 24.0, 50.0))
            outward_limit = float(self._v88_waist_curve_outward_limit_mm)
            bodyward_limit = float(self._v88_waist_curve_bodyward_limit_mm)
            valid_global = (
                (sproj >= sl - pad) & (sproj <= sr + pad)
                & (dproj >= -outward_limit) & (dproj <= bodyward_limit)
            )
            if int(np.count_nonzero(valid_global)) < 18:
                result["reason"] = "V88_WAIST_CURVE_NEAR_POSE_OUTER_BAND_EMPTY"
                return result
            targets = np.linspace(sl, sr, 13)
            half = float(np.clip(span / 13.0, 16.0, 30.0))
            curve = []
            curve_s = []
            for ts in targets:
                ids = np.flatnonzero(valid_global & (np.abs(sproj - float(ts)) <= half))
                if len(ids) == 0:
                    continue
                local_d = dproj[ids]
                local_s = sproj[ids]
                score = local_d + 0.035 * np.abs(local_s - float(ts))
                jj = int(ids[int(np.argmin(score))])
                q = np.asarray(c[jj], np.float64)
                if curve and float(np.linalg.norm(q - curve[-1])) > 85.0:
                    order = ids[np.argsort(score)]
                    q2 = None
                    for kk in order[:12]:
                        cand = np.asarray(c[int(kk)], np.float64)
                        if float(np.linalg.norm(cand - curve[-1])) <= 85.0:
                            q2 = cand
                            jj = int(kk)
                            break
                    if q2 is None:
                        continue
                    q = q2
                curve.append(q)
                curve_s.append(float(sproj[jj]))
            if len(curve) < 7:
                result["reason"] = f"V87_WAIST_CURVE_TOO_SHORT_{len(curve)}"
                return result
            order = np.argsort(np.asarray(curve_s, np.float64))
            curve = [np.asarray(curve[int(i)], np.float64) for i in order]
            curve_s = [float(curve_s[int(i)]) for i in order]
            left_idx = min(1, len(curve) - 1)
            right_idx = max(left_idx + 1, len(curve) - 2)
            left = np.asarray(curve[left_idx], np.float64)
            right = np.asarray(curve[right_idx], np.float64)
            center_idx = int(np.argmin(np.abs(np.asarray(curve_s, np.float64))))
            center = np.asarray(curve[center_idx], np.float64)
            refined_width = float(np.linalg.norm(right - left))
            if refined_width < 75.0 or refined_width > min(560.0, max(180.0, 1.85 * pose_width)):
                result.update({"reason": "V87_WAIST_CURVE_WIDTH_IMPLAUSIBLE", "refined_width_mm": refined_width})
                return result
            line_vec = right - left
            line_len = float(np.linalg.norm(line_vec))
            if line_len > 1e-6:
                waist_u_refined = line_vec / line_len
                if float(np.dot(waist_u_refined, waist_u)) < 0.0:
                    waist_u_refined = -waist_u_refined
            else:
                waist_u_refined = waist_u
            shift = center - wc
            body_shift = float(np.dot(shift, body_u))
            if body_shift < -float(self._v88_waist_curve_outward_limit_mm) or body_shift > float(self._v88_waist_curve_bodyward_limit_mm):
                result.update({
                    "reason": "V88_WAIST_CURVE_BODY_SHIFT_TOO_LARGE",
                    "body_shift_mm": float(body_shift),
                    "limit_outward_mm": float(self._v88_waist_curve_outward_limit_mm),
                    "limit_bodyward_mm": float(self._v88_waist_curve_bodyward_limit_mm),
                })
                return result
            result.update({
                "ok": True,
                "reason": "V88_POSE_ANCHORED_REAL_MASK_WAIST_CURVE",
                "waist_left": left,
                "waist_center": center,
                "waist_right": right,
                "waist_curve_board": [q.astype(float).tolist() for q in curve],
                "body_u": body_u,
                "waist_u": waist_u_refined,
                "pose_width_mm": pose_width,
                "refined_width_mm": refined_width,
                "body_shift_mm": body_shift,
                "tangent_shift_mm": float(np.dot(shift, waist_u_refined)),
            })
            return result

        def _v81_pose_constrained_outer_waist(self, obs):
            curve_ref = self._v87_mask_waist_curve_ref(obs)
            if bool((curve_ref or {}).get("ok", False)):
                print(
                    f"[D60-V87-WAIST-CURVE] real Mask outer curve USED poseWidth={float(curve_ref.get('pose_width_mm', 0.0)):.1f}mm "
                    f"curveWidth={float(curve_ref.get('refined_width_mm', 0.0)):.1f}mm points={len(curve_ref.get('waist_curve_board', []))}"
                )
                return curve_ref
            result = {"ok": False, "reason": "V81_OUTER_WAIST_UNAVAILABLE"}
            pts = self._pose_board_points(obs)
            wl = self._finite_point2(pts.get("waist_img_left"))
            wc = self._finite_point2(pts.get("waist_center"))
            wr = self._finite_point2(pts.get("waist_img_right"))
            crotch = self._finite_point2(pts.get("crotch"))
            if wl is None or wr is None or crotch is None:
                result["reason"] = "V81_WAIST_POSE_UNAVAILABLE"
                return result
            if wc is None:
                wc = 0.5 * (wl + wr)
            wv = np.asarray(wr - wl, np.float64)
            width = float(np.linalg.norm(wv))
            bv = np.asarray(crotch - wc, np.float64)
            blen = float(np.linalg.norm(bv))
            if width < 70.0 or blen < 20.0:
                result["reason"] = "V81_WAIST_POSE_GEOMETRY_INVALID"
                return result
            waist_u = wv / width
            body_u = bv / blen
            probe_plus = self._v53_mask_point_report(obs, wc + body_u * 22.0)
            probe_minus = self._v53_mask_point_report(obs, wc - body_u * 22.0)
            if not bool((probe_plus or {}).get("ok", False)) and bool((probe_minus or {}).get("ok", False)):
                body_u = -body_u
            _, contour = self._v53_board_contour(obs)
            if contour is None or len(contour) < 20:
                result["reason"] = "V81_REAL_MASK_CONTOUR_UNAVAILABLE"
                return result
            c = np.asarray(contour, np.float64).reshape(-1, 2)
            picked = {}
            targets = {"left": wl, "center": wc, "right": wr}
            for name, target in targets.items():
                rel = c - target[None, :]
                axial = rel @ body_u
                tangent = rel @ waist_u
                bands = (55.0, 75.0, 100.0) if name != "center" else (65.0, 90.0, 115.0)
                chosen = None
                for band in bands:
                    valid = (np.abs(tangent) <= band) & (axial >= -45.0) & (axial <= 28.0)
                    ids = np.flatnonzero(valid)
                    if len(ids) == 0:
                        continue
                    dist = np.linalg.norm(rel[ids], axis=1)
                    score = axial[ids] + 0.10 * np.abs(tangent[ids]) + 0.015 * dist
                    q = np.asarray(c[int(ids[int(np.argmin(score))])], np.float64)
                    if float(np.linalg.norm(q - target)) <= 125.0:
                        chosen = q
                        break
                if chosen is None:
                    result["reason"] = f"V81_{name.upper()}_OUTER_EDGE_NOT_FOUND"
                    return result
                picked[name] = chosen
            left = picked["left"]
            center = picked["center"]
            right = picked["right"]
            if float(np.dot(right - left, waist_u)) < 0.0:
                left, right = right, left
            refined_width = float(np.linalg.norm(right - left))
            if refined_width < max(80.0, 0.50 * width) or refined_width > min(560.0, 1.55 * width):
                result.update({"reason": "V81_OUTER_WIDTH_IMPLAUSIBLE", "refined_width_mm": refined_width})
                return result
            shift = center - wc
            body_shift = float(np.dot(shift, body_u))
            tangent_shift = float(np.dot(shift, waist_u))
            if body_shift < -45.0 or body_shift > 28.0 or abs(tangent_shift) > 100.0:
                result.update({"reason": "V81_OUTER_SHIFT_TOO_LARGE", "body_shift_mm": body_shift, "tangent_shift_mm": tangent_shift})
                return result
            result.update({
                "ok": True,
                "reason": "V81_POSE_CONSTRAINED_REAL_MASK_OUTER_WAIST",
                "waist_left": left,
                "waist_center": center,
                "waist_right": right,
                "body_u": body_u,
                "waist_u": waist_u,
                "pose_width_mm": width,
                "refined_width_mm": refined_width,
                "body_shift_mm": body_shift,
                "tangent_shift_mm": tangent_shift,
            })
            return result

        def _v54_obs_with_refined_waist_uncached(self, obs):
            prior_ref = getattr(obs, "bottom_vla_v87_waist_curve_ref", None)
            if isinstance(prior_ref, dict) and bool(prior_ref.get("ok", False)):
                return obs, copy.deepcopy(prior_ref)
            ref = self._v81_pose_constrained_outer_waist(obs)
            if bool(ref.get("ok", False)):
                ref_source = str(ref.get("reason", "V81_POSE_CONSTRAINED_REAL_MASK_OUTER_WAIST"))
                obs2 = self._v80_obs_from_outer_waist_ref(obs, ref, ref_source)
                if obs2 is not None:
                    if str(ref_source).startswith(("V87_", "V88_", "V90_")):
                        setattr(obs2, "bottom_vla_v87_waist_curve_ref", copy.deepcopy(ref))
                        setattr(obs2, "bottom_vla_v87_waist_refined_once", True)
                    print(
                        f"[D60-V81-WAIST-OUTER] Pose-constrained real outer waist USED "
                        f"poseWidth={float(ref.get('pose_width_mm',0.0)):.1f}mm outerWidth={float(ref.get('refined_width_mm',0.0)):.1f}mm "
                        f"bodyShift={float(ref.get('body_shift_mm',0.0)):+.1f}mm"
                    )
                    return obs2, ref
            strict = self._v54_waist_corner_refinement(obs)
            if bool((strict or {}).get("ok", False)):
                body_shift = float(strict.get("body_shift_mm", 0.0))
                if -45.0 <= body_shift <= 28.0:
                    obs2 = self._v80_obs_from_outer_waist_ref(obs, strict, "V81_STRICT_REAL_MASK_OUTER_WAIST")
                    if obs2 is not None:
                        out = dict(strict)
                        out["reason"] = "V81_STRICT_REAL_MASK_OUTER_WAIST"
                        print(
                            f"[D60-V81-WAIST-OUTER] strict real outer waist USED "
                            f"outerWidth={float(strict.get('refined_width_mm',0.0)):.1f}mm bodyShift={body_shift:+.1f}mm"
                        )
                        return obs2, out
            fallback = dict(ref or {})
            fallback.update({"ok": False, "reason": str(fallback.get("reason", "V81_OUTER_UNAVAILABLE")), "v81_keep_pose_waist": True})
            print(f"[D60-V81-WAIST-OUTER-FALLBACK] {fallback['reason']}; keep semantic Pose waist, never substitute a lower body-width line")
            return obs, fallback

        def _v81_waist_frame_units(self, obs):
            ref = getattr(obs, "bottom_vla_v87_waist_curve_ref", None)
            if not (isinstance(ref, dict) and bool(ref.get("ok", False))):
                ref = self._v87_mask_waist_curve_ref(obs)
            if bool((ref or {}).get("ok", False)):
                left = self._finite_point2(ref.get("waist_left"))
                center = self._finite_point2(ref.get("waist_center"))
                right = self._finite_point2(ref.get("waist_right"))
                waist_u = self._finite_point2(ref.get("waist_u"))
                body_u = self._finite_point2(ref.get("body_u"))
                if left is not None and center is not None and right is not None and waist_u is not None and body_u is not None:
                    return {"left": left, "center": center, "right": right, "waist_u": waist_u, "body_u": body_u, "curve": copy.deepcopy(ref.get("waist_curve_board", []))}
            pts = self._pose_board_points(obs)
            wl = self._finite_point2(pts.get("waist_img_left"))
            wc = self._finite_point2(pts.get("waist_center"))
            wr = self._finite_point2(pts.get("waist_img_right"))
            crotch = self._finite_point2(pts.get("crotch"))
            if wl is None or wr is None or crotch is None:
                return None
            if wc is None:
                wc = 0.5 * (wl + wr)
            wv = np.asarray(wr - wl, np.float64)
            bv = np.asarray(crotch - wc, np.float64)
            wn = float(np.linalg.norm(wv))
            bn = float(np.linalg.norm(bv))
            if wn < 1e-6 or bn < 1e-6:
                return None
            waist_u = wv / wn
            body_u = bv / bn
            if not bool((self._v53_mask_point_report(obs, wc + body_u * 20.0) or {}).get("ok", False)):
                if bool((self._v53_mask_point_report(obs, wc - body_u * 20.0) or {}).get("ok", False)):
                    body_u = -body_u
            return {"left": wl, "center": wc, "right": wr, "waist_u": waist_u, "body_u": body_u}

        def _v81_direct_inward_candidate(self, obs, arm_key, old, frame, desired_depth):
            old = np.asarray(old, np.float64).reshape(2)
            waist_u = np.asarray(frame["waist_u"], np.float64).reshape(2)
            body_u = np.asarray(frame["body_u"], np.float64).reshape(2)
            wn = float(np.linalg.norm(waist_u))
            bn = float(np.linalg.norm(body_u))
            if wn < 1e-6 or bn < 1e-6:
                return None, {"ok": False, "reason": "V88_WAIST_FRAME_AXIS_INVALID"}
            waist_u /= wn
            body_u /= bn
            center = np.asarray(frame["center"], np.float64).reshape(2)
            endpoints = [np.asarray(frame["left"], np.float64), np.asarray(frame["right"], np.float64)]
            endpoints.sort(key=lambda q: float(np.linalg.norm(q - old)))

            curve_bases = []
            try:
                curve = np.asarray(frame.get("curve", []), np.float64).reshape(-1, 2)
                if len(curve) >= 5:
                    proj = (curve - center.reshape(1, 2)) @ waist_u
                    order = np.argsort(proj)
                    curve = curve[order]
                    proj = proj[order]
                    mid = 0.5 * (float(np.min(proj)) + float(np.max(proj)))
                    if str(arm_key).lower() == "arm2":
                        ids = np.flatnonzero(proj <= mid + 0.12 * max(1.0, float(np.ptp(proj))))
                    else:
                        ids = np.flatnonzero(proj >= mid - 0.12 * max(1.0, float(np.ptp(proj))))
                    pool = curve[ids] if len(ids) else curve
                    if len(pool):
                        take = np.unique(np.linspace(0, len(pool) - 1, min(7, len(pool))).round().astype(int))
                        curve_bases = [np.asarray(pool[int(i)], np.float64) for i in take]
            except Exception:
                curve_bases = []

            if curve_bases:
                bases = curve_bases
                tangent_limit = float(self._v88_waist_grip_tangent_max_mm)
                tangents = tuple(float(v) for v in (0.0, -8.0, 8.0, -16.0, 16.0, -24.0, 24.0, -32.0, 32.0) if abs(float(v)) <= tangent_limit + 1e-6)
                max_depth = float(self._v88_waist_grip_body_max_mm)
                depths = [float(desired_depth)] + [
                    float(v) for v in np.arange(36.0, max_depth + 0.1, 4.0)
                    if abs(float(v) - float(desired_depth)) > 1e-6
                ]
                curve_locked = True
            else:
                bases = [old] + endpoints
                tangents = (0.0, -12.0, 12.0, -24.0, 24.0)
                depths = [float(desired_depth)] + [float(v) for v in np.arange(36.0, 69.0, 4.0) if abs(float(v) - float(desired_depth)) > 1e-6]
                curve_locked = False

            rows = []
            for base_index, base in enumerate(bases):
                base = np.asarray(base, np.float64).reshape(2)
                for depth in depths:
                    for tangent in tangents:
                        q = np.asarray(base + body_u * depth + waist_u * tangent, np.float64)
                        if not self._v56_align_board_safe(q, 8.0):
                            continue
                        hard_selector = getattr(self.d56, "_d56v48_select_arm_for_board_x_hard", None)
                        if callable(hard_selector):
                            try:
                                owner = str(hard_selector(self.cfg56, float(q[0]))).lower()
                            except Exception:
                                owner = ""
                            if owner and owner != str(arm_key).lower():
                                continue
                        rep = self._v53_mask_point_report(obs, q)
                        edge_depth = rep.get("edge_depth_mm") if isinstance(rep, dict) else None
                        if not bool((rep or {}).get("ok", False)) or edge_depth is None or float(edge_depth) < 35.0:
                            continue
                        reach_ok, reach_reason, reach_radius = self._v56_align_arm_reach(arm_key, q, final_target=False)
                        if not reach_ok:
                            continue
                        score = (
                            2.8 * abs(float(edge_depth) - float(desired_depth))
                            + 0.22 * abs(float(tangent))
                            + 0.018 * float(np.linalg.norm(q - old))
                            + 0.8 * float(base_index)
                            + 0.002 * float(reach_radius if math.isfinite(float(reach_radius)) else 9999.0)
                        )
                        rows.append((float(score), q, float(edge_depth), str(reach_reason), base.copy(), float(depth), float(tangent)))
            if not rows:
                return None, {"ok": False, "reason": f"{arm_key.upper()}_NO_REACHABLE_V88_WAIST_CURVE_INTERIOR_POINT"}
            rows.sort(key=lambda row: row[0])
            score, q, edge_depth, reach_reason, base, body_depth, tangent = rows[0]
            return q, {
                "ok": True,
                "reason": "V88_CURVE_LOCAL_GRIP" if curve_locked else "V88_POSE_LOCAL_GRIP",
                "edge_depth_mm": edge_depth,
                "score": score,
                "reach": reach_reason,
                "curve_locked": bool(curve_locked),
                "curve_base_board": base.astype(float).tolist(),
                "body_depth_request_mm": float(body_depth),
                "tangent_shift_mm": float(tangent),
            }

        def _v81_apply_deterministic_waist_pair(self, obs, plan):
            if plan is None:
                return None, {"ok": False, "reason": "PLAN_UNAVAILABLE"}
            frame = self._v81_waist_frame_units(obs)
            if frame is None:
                return None, {"ok": False, "reason": "WAIST_FRAME_UNAVAILABLE"}
            out = copy.deepcopy(plan)
            points = getattr(out, "arm_points", None)
            if not isinstance(points, dict):
                return None, {"ok": False, "reason": "ARM_POINTS_UNAVAILABLE"}
            desired = {"arm1": 40.0, "arm2": 52.0}
            rep = {"ok": True, "reason": "OK", "arms": {}}
            selected = {}
            for arm in ("arm2", "arm1"):
                item = points.get(arm, {})
                if not isinstance(item, dict):
                    return None, {"ok": False, "reason": f"{arm.upper()}_ARM_ITEM_UNAVAILABLE"}
                old = None
                for key in ("grip_board", "source_board", "grasp_board", "target_board"):
                    old = self._finite_point2(item.get(key))
                    if old is not None:
                        break
                if old is None:
                    return None, {"ok": False, "reason": f"{arm.upper()}_GRIP_UNAVAILABLE"}
                q, qrep = self._v81_direct_inward_candidate(obs, arm, old, frame, desired[arm])
                if q is None:
                    return None, qrep
                selected[arm] = np.asarray(q, np.float64)
                for key in ("grip_board", "source_board", "grasp_board", "target_board"):
                    cur = self._finite_point2(item.get(key))
                    if cur is not None and float(np.linalg.norm(cur - old)) <= 3.0:
                        item[key] = (float(q[0]), float(q[1]))
                try:
                    px = self.d56.board_to_pixel(self.H_raw, float(q[0]), float(q[1]))
                    if px is not None and "grip_px" in item:
                        item["grip_px"] = (float(px[0]), float(px[1]))
                except Exception:
                    pass
                points[arm] = item
                rep["arms"][arm] = {
                    "desired_inset_mm": float(desired[arm]),
                    "actual_inset_mm": float(qrep.get("edge_depth_mm", 0.0)),
                    "before_board": np.asarray(old, np.float64).astype(float).tolist(),
                    "after_board": q.astype(float).tolist(),
                }
            if not self._v56_align_board_safe(selected["arm2"], 8.0) or not self._v56_align_board_safe(selected["arm1"], 8.0):
                return None, {"ok": False, "reason": "V81_BOARD_OUTSIDE"}
            if float(np.linalg.norm(selected["arm1"] - selected["arm2"])) < 85.0:
                return None, {"ok": False, "reason": "V81_DUAL_COLLISION_RISK_SOURCE_TOO_CLOSE"}
            out.arm_points = points
            metrics = dict(getattr(out, "metrics", {}) or {})
            metrics.update({
                "bottom_vla_v81_deterministic_waist_pair": True,
                "bottom_vla_v81_board_collision_only_veto": True,
                "bottom_vla_v81_pair_report": copy.deepcopy(rep),
            })
            out.metrics = metrics
            try:
                out.ok = True
            except Exception:
                pass
            print(
                f"[D60-V81-DIRECT-PAIR] A2 inset={rep['arms']['arm2']['actual_inset_mm']:.1f}mm "
                f"A1 inset={rep['arms']['arm1']['actual_inset_mm']:.1f}mm; Pose+Mask valid -> deterministic pair"
            )
            return out, rep

        def _d60_force_waist_endpoint_plan(self, obs, source_plan):
            planning_obs, refinement = self._v54_obs_with_refined_waist(obs)
            plan, motion, report = self._v81_prev_force_waist_endpoint_plan(planning_obs, source_plan)
            if plan is not None and motion is not None:
                direct, drep = self._v81_apply_deterministic_waist_pair(planning_obs, plan)
                if direct is not None:
                    rebuilt, why = self.d56._63_step120_motion_plan_from_d56_plan(direct, self.config, self.cfg56, self.d56_args)
                    if rebuilt is not None:
                        outrep = dict(report or {})
                        outrep["bottom_vla_v81_direct_pair"] = copy.deepcopy(drep)
                        outrep["bottom_vla_v81_motion_reason"] = str(why)
                        return direct, rebuilt, outrep
            broad_tiers = (
                {"body_min_mm": 0.0, "body_max_mm": 130.0, "tangent_max_mm": 170.0, "radius_max_mm": 240.0},
                {"body_min_mm": 0.0, "body_max_mm": 190.0, "tangent_max_mm": 250.0, "radius_max_mm": 340.0},
            )
            failures = []
            old_required = float(self._v53_waist_hard_inset_mm)
            try:
                self._v53_waist_hard_inset_mm = 0.0
                for tier, limits in enumerate(broad_tiers, 1):
                    for overlap in (None, 30.0, 60.0):
                        base_plan, base_motion, base_report = self._v50_d60_force_with_limits(
                            planning_obs, source_plan, limits, "v81_deterministic_outer_recovery", overlap
                        )
                        if base_plan is None:
                            failures.append({"tier": tier, "overlap": overlap, "reason": str((base_report or {}).get("reason", "NO_BASE_PAIR"))})
                            continue
                        direct, drep = self._v81_apply_deterministic_waist_pair(planning_obs, base_plan)
                        if direct is None:
                            failures.append({"tier": tier, "overlap": overlap, "reason": str((drep or {}).get("reason", "DIRECT_PAIR_FAILED"))})
                            continue
                        rebuilt, why = self.d56._63_step120_motion_plan_from_d56_plan(direct, self.config, self.cfg56, self.d56_args)
                        if rebuilt is None:
                            failures.append({"tier": tier, "overlap": overlap, "reason": f"MOTION_BUILD:{why}"})
                            continue
                        outrep = dict(base_report or {})
                        outrep.update({
                            "reason": "OK",
                            "bottom_vla_v81_deterministic_outer_recovery": True,
                            "bottom_vla_v81_direct_pair": copy.deepcopy(drep),
                            "bottom_vla_v81_outer_refinement": copy.deepcopy(refinement),
                            "bottom_vla_v81_motion_reason": str(why),
                        })
                        print(
                            f"[D60-V81-RECOVER] standard candidate search failed; recovered deterministic outer-waist pair "
                            f"tier={tier} overlap={overlap}"
                        )
                        return direct, rebuilt, outrep
            finally:
                self._v53_waist_hard_inset_mm = old_required
            print(f"[D60-V81-BLOCK] deterministic recovery exhausted; board/dual compatibility prevented an executable pair failures={failures[-6:]}")
            return None, None, {
                "reason": "V81_BOARD_OR_DUAL_COLLISION_BLOCK",
                "bottom_vla_v81_failures": failures,
                "bottom_vla_v81_outer_refinement": copy.deepcopy(refinement),
            }

        def _d60_plan_from_obs(self, obs):
            planning_obs, refinement = self._v54_obs_with_refined_waist(obs)
            plan = self._v81_prev_d60_plan_from_obs(planning_obs)
            if plan is not None and bool(getattr(plan, "ok", False)):
                direct, drep = self._v81_apply_deterministic_waist_pair(planning_obs, plan)
                if direct is not None:
                    return direct
                return plan
            if bool(getattr(self, "_v53_align_capture_active", False)):
                return plan
            forced, motion, report = self._d60_force_waist_endpoint_plan(planning_obs, plan)
            if forced is not None and motion is not None and bool(getattr(forced, "ok", False)):
                metrics = dict(getattr(forced, "metrics", {}) or {})
                metrics["bottom_vla_v81_recovered_from_invalid_primary"] = True
                metrics["bottom_vla_v81_recovery_report"] = copy.deepcopy(report or {})
                forced.metrics = metrics
                return forced
            if plan is not None:
                try:
                    plan.reason = "V81_BOARD_OR_DUAL_COLLISION_BLOCK"
                except Exception:
                    pass
            return plan

        def _d60_waist_endpoint_gate(self, obs, plan):
            metrics = dict(getattr(plan, "metrics", {}) or {}) if plan is not None else {}
            if bool(metrics.get("bottom_vla_v81_deterministic_waist_pair", False)):
                points = getattr(plan, "arm_points", {}) or {}
                for arm in ("arm2", "arm1"):
                    item = points.get(arm, {}) if isinstance(points, dict) else {}
                    grip = None
                    if isinstance(item, dict):
                        for key in ("grip_board", "source_board", "grasp_board"):
                            grip = self._finite_point2(item.get(key))
                            if grip is not None:
                                break
                    if grip is None or not self._v56_align_board_safe(grip, 8.0):
                        return False, {"ok": False, "reason": "V81_BOARD_OUTSIDE", "arm": arm}
                g2 = self._finite_point2((points.get("arm2", {}) or {}).get("grip_board"))
                g1 = self._finite_point2((points.get("arm1", {}) or {}).get("grip_board"))
                if g2 is not None and g1 is not None and float(np.linalg.norm(g1 - g2)) < 85.0:
                    return False, {"ok": False, "reason": "V81_DUAL_COLLISION_RISK_SOURCE_TOO_CLOSE"}
                return True, {"ok": True, "reason": "V81_BOARD_COLLISION_ONLY_GATE"}
            return self._v81_prev_waist_endpoint_gate(obs, plan)


        def _v87_mask_waist_curve_ref(self, obs):
            result = {"ok": False, "reason": "V90_MASK_WAIST_CURVE_UNAVAILABLE"}
            pts = self._pose_board_points(obs)
            wl = self._finite_point2(pts.get("waist_img_left"))
            wc = self._finite_point2(pts.get("waist_center"))
            wr = self._finite_point2(pts.get("waist_img_right"))
            crotch = self._finite_point2(pts.get("crotch"))
            if wl is None or wr is None or crotch is None:
                result["reason"] = "V90_POSE_WAIST_ANCHOR_UNAVAILABLE"
                return result
            if wc is None:
                wc = 0.5 * (wl + wr)
            waist_vec = np.asarray(wr - wl, np.float64)
            body_vec = np.asarray(crotch - wc, np.float64)
            pose_width = float(np.linalg.norm(waist_vec))
            body_len = float(np.linalg.norm(body_vec))
            if pose_width < 55.0 or body_len < 20.0:
                result["reason"] = "V90_POSE_WAIST_ANCHOR_GEOMETRY_INVALID"
                return result
            waist_u = waist_vec / pose_width
            body_u = body_vec / body_len
            plus = self._v53_mask_point_report(obs, wc + body_u * 22.0)
            minus = self._v53_mask_point_report(obs, wc - body_u * 22.0)
            if not bool((plus or {}).get("ok", False)) and bool((minus or {}).get("ok", False)):
                body_u = -body_u
            _, contour = self._v53_board_contour(obs)
            if contour is None or len(contour) < 30:
                result["reason"] = "V90_REAL_MASK_CONTOUR_UNAVAILABLE"
                return result
            c = np.asarray(contour, np.float64).reshape(-1, 2)
            sl = float(np.dot(wl - wc, waist_u))
            sr = float(np.dot(wr - wc, waist_u))
            if sl > sr:
                waist_u = -waist_u
                sl = float(np.dot(wl - wc, waist_u))
                sr = float(np.dot(wr - wc, waist_u))
            span = max(55.0, float(sr - sl))
            rel = c - wc[None, :]
            sproj = rel @ waist_u
            dproj = rel @ body_u
            pad = float(np.clip(0.24 * span, 36.0, 78.0))
            outward_limit = float(self._v90_waist_curve_outward_limit_mm)
            bodyward_limit = float(self._v90_waist_curve_bodyward_limit_mm)
            valid_global = (
                (sproj >= sl - pad) & (sproj <= sr + pad)
                & (dproj >= -outward_limit) & (dproj <= bodyward_limit)
            )
            if int(np.count_nonzero(valid_global)) < 18:
                result["reason"] = "V90_WAIST_CURVE_POSE_BAND_EMPTY"
                return result
            targets = np.linspace(sl, sr, 15)
            half = float(np.clip(span / 12.0, 18.0, 36.0))
            curve = []
            curve_s = []
            for ts in targets:
                ids = np.flatnonzero(valid_global & (np.abs(sproj - float(ts)) <= half))
                if len(ids) == 0:
                    continue
                local_d = dproj[ids]
                local_s = sproj[ids]
                score = local_d + 0.02 * np.abs(local_s - float(ts))
                order = ids[np.argsort(score)]
                chosen = None
                chosen_idx = None
                for kk in order[:18]:
                    cand = np.asarray(c[int(kk)], np.float64)
                    if curve and float(np.linalg.norm(cand - curve[-1])) > 100.0:
                        continue
                    chosen = cand
                    chosen_idx = int(kk)
                    break
                if chosen is None:
                    continue
                curve.append(chosen)
                curve_s.append(float(sproj[chosen_idx]))
            if len(curve) < 7:
                result["reason"] = f"V90_WAIST_CURVE_TOO_SHORT_{len(curve)}"
                return result
            order = np.argsort(np.asarray(curve_s, np.float64))
            curve = [np.asarray(curve[int(i)], np.float64) for i in order]
            curve_s = [float(curve_s[int(i)]) for i in order]
            left_idx = min(1, len(curve) - 1)
            right_idx = max(left_idx + 1, len(curve) - 2)
            left = np.asarray(curve[left_idx], np.float64)
            right = np.asarray(curve[right_idx], np.float64)
            center_idx = int(np.argmin(np.abs(np.asarray(curve_s, np.float64))))
            center = np.asarray(curve[center_idx], np.float64)
            refined_width = float(np.linalg.norm(right - left))
            if refined_width < 70.0 or refined_width > min(590.0, max(190.0, 1.95 * pose_width)):
                result.update({"reason": "V90_WAIST_CURVE_WIDTH_IMPLAUSIBLE", "refined_width_mm": refined_width})
                return result
            line_vec = right - left
            line_len = float(np.linalg.norm(line_vec))
            waist_u_refined = line_vec / line_len if line_len > 1e-6 else waist_u
            if float(np.dot(waist_u_refined, waist_u)) < 0.0:
                waist_u_refined = -waist_u_refined
            shift = center - wc
            body_shift = float(np.dot(shift, body_u))
            if body_shift < -outward_limit or body_shift > bodyward_limit:
                result.update({
                    "reason": "V90_WAIST_CURVE_BODY_SHIFT_OUTSIDE_SEMANTIC_BAND",
                    "body_shift_mm": body_shift,
                })
                return result
            result.update({
                "ok": True,
                "reason": "V90_POSE_ANCHORED_REAL_MASK_WAIST_CURVE",
                "waist_left": left,
                "waist_center": center,
                "waist_right": right,
                "waist_curve_board": [q.astype(float).tolist() for q in curve],
                "body_u": body_u,
                "waist_u": waist_u_refined,
                "pose_width_mm": pose_width,
                "refined_width_mm": refined_width,
                "body_shift_mm": body_shift,
                "tangent_shift_mm": float(np.dot(shift, waist_u_refined)),
            })
            return result

        def _v90_waist_candidate_rows(self, obs, arm_key, old, frame, desired_depth):
            old = np.asarray(old, np.float64).reshape(2)
            waist_u = np.asarray(frame["waist_u"], np.float64).reshape(2)
            body_u = np.asarray(frame["body_u"], np.float64).reshape(2)
            wn = float(np.linalg.norm(waist_u))
            bn = float(np.linalg.norm(body_u))
            if wn < 1e-6 or bn < 1e-6:
                return []
            waist_u /= wn
            body_u /= bn
            center = np.asarray(frame["center"], np.float64).reshape(2)
            left = np.asarray(frame["left"], np.float64).reshape(2)
            right = np.asarray(frame["right"], np.float64).reshape(2)
            curve = np.asarray(frame.get("curve", []), np.float64).reshape(-1, 2) if frame.get("curve") else np.empty((0, 2), np.float64)
            bases = []
            if len(curve) >= 5:
                proj = (curve - center.reshape(1, 2)) @ waist_u
                order = np.argsort(proj)
                curve = curve[order]
                proj = proj[order]
                span = max(1.0, float(np.ptp(proj)))
                mid = 0.5 * (float(np.min(proj)) + float(np.max(proj)))
                if str(arm_key).lower() == "arm2":
                    ids = np.flatnonzero(proj <= mid + 0.20 * span)
                else:
                    ids = np.flatnonzero(proj >= mid - 0.20 * span)
                pool = curve[ids] if len(ids) else curve
                if len(pool):
                    take = np.unique(np.linspace(0, len(pool) - 1, min(13, len(pool))).round().astype(int))
                    bases = [np.asarray(pool[int(i)], np.float64) for i in take]
            if not bases:
                endpoint = left if float(np.linalg.norm(left - old)) <= float(np.linalg.norm(right - old)) else right
                shifts = np.arange(-float(self._v90_waist_grip_tangent_max_mm), float(self._v90_waist_grip_tangent_max_mm) + 0.1, 15.0)
                bases = [np.asarray(endpoint + waist_u * float(s), np.float64) for s in shifts]
            depths = [float(desired_depth)] + [
                float(v) for v in np.arange(35.0, float(self._v90_waist_grip_body_max_mm) + 0.1, 3.0)
                if abs(float(v) - float(desired_depth)) > 1e-6
            ]
            micro = (0.0, -9.0, 9.0, -18.0, 18.0)
            hard_min = float(self._v53_waist_hard_inset_mm)
            hard_max = float(self._v90_waist_inset_hard_max_mm)
            rows = []
            for base_index, base in enumerate(bases):
                base = np.asarray(base, np.float64).reshape(2)
                for depth in depths:
                    for tangent in micro:
                        q = np.asarray(base + body_u * float(depth) + waist_u * float(tangent), np.float64)
                        if not self._v56_align_board_safe(q, 8.0):
                            continue
                        rep = self._v53_mask_point_report(obs, q)
                        edge_depth = rep.get("edge_depth_mm") if isinstance(rep, dict) else None
                        if not bool((rep or {}).get("ok", False)) or edge_depth is None:
                            continue
                        edge_depth = float(edge_depth)
                        if edge_depth + float(self._v53_waist_inset_tolerance_mm) < hard_min or edge_depth > hard_max:
                            continue
                        reach_ok, reach_reason, reach_radius = self._v56_align_arm_reach(arm_key, q, final_target=False)
                        if not reach_ok:
                            continue
                        tangent_from_old = float(np.dot(q - old, waist_u))
                        preferred_penalty = 0.0
                        if edge_depth < float(self._v90_waist_inset_preferred_min_mm):
                            preferred_penalty += 3.0 * (float(self._v90_waist_inset_preferred_min_mm) - edge_depth)
                        if edge_depth > float(self._v90_waist_inset_preferred_max_mm):
                            preferred_penalty += 2.5 * (edge_depth - float(self._v90_waist_inset_preferred_max_mm))
                        score = (
                            2.2 * abs(edge_depth - float(desired_depth))
                            + 0.45 * abs(float(depth) - float(desired_depth))
                            + 0.08 * abs(float(tangent))
                            + 0.012 * float(np.linalg.norm(q - old))
                            + 0.35 * float(base_index)
                            + preferred_penalty
                            + 0.0015 * float(reach_radius if math.isfinite(float(reach_radius)) else 9999.0)
                        )
                        rows.append({
                            "score": float(score),
                            "q": q,
                            "edge_depth_mm": edge_depth,
                            "reach": str(reach_reason),
                            "base": base,
                            "body_depth_mm": float(depth),
                            "micro_tangent_mm": float(tangent),
                            "tangent_from_old_mm": tangent_from_old,
                        })
            rows.sort(key=lambda row: row["score"])
            return rows[:48]

        def _v90_plan_with_pair(self, template, old_points, pair_rows):
            out = copy.deepcopy(template)
            points = getattr(out, "arm_points", None)
            if not isinstance(points, dict):
                return None
            for arm in ("arm2", "arm1"):
                item = points.get(arm, {})
                if not isinstance(item, dict):
                    return None
                old = np.asarray(old_points[arm], np.float64)
                q = np.asarray(pair_rows[arm]["q"], np.float64)
                for key in ("grip_board", "source_board", "grasp_board", "target_board"):
                    cur = self._finite_point2(item.get(key))
                    if cur is not None and float(np.linalg.norm(cur - old)) <= 3.0:
                        item[key] = (float(q[0]), float(q[1]))
                try:
                    px = self.d56.board_to_pixel(self.H_raw, float(q[0]), float(q[1]))
                    if px is not None and "grip_px" in item:
                        item["grip_px"] = (float(px[0]), float(px[1]))
                except Exception:
                    pass
                points[arm] = item
            out.arm_points = points
            try:
                out.ok = True
            except Exception:
                pass
            return out

        def _v81_apply_deterministic_waist_pair(self, obs, plan):
            if plan is None:
                return None, {"ok": False, "reason": "PLAN_UNAVAILABLE"}
            frame = self._v81_waist_frame_units(obs)
            if frame is None:
                return None, {"ok": False, "reason": "WAIST_FRAME_UNAVAILABLE"}
            points = getattr(plan, "arm_points", None)
            if not isinstance(points, dict):
                return None, {"ok": False, "reason": "ARM_POINTS_UNAVAILABLE"}
            old_points = {}
            for arm in ("arm2", "arm1"):
                item = points.get(arm, {})
                if not isinstance(item, dict):
                    return None, {"ok": False, "reason": f"{arm.upper()}_ARM_ITEM_UNAVAILABLE"}
                old = None
                for key in ("grip_board", "source_board", "grasp_board", "target_board"):
                    old = self._finite_point2(item.get(key))
                    if old is not None:
                        break
                if old is None:
                    return None, {"ok": False, "reason": f"{arm.upper()}_GRIP_UNAVAILABLE"}
                old_points[arm] = np.asarray(old, np.float64)
            desired = {"arm1": 45.0, "arm2": 52.0}
            rows = {
                arm: self._v90_waist_candidate_rows(obs, arm, old_points[arm], frame, desired[arm])
                for arm in ("arm2", "arm1")
            }
            if not rows["arm2"] or not rows["arm1"]:
                return None, {
                    "ok": False,
                    "reason": "V90_NO_REACHABLE_WAIST_LOCAL_CANDIDATE",
                    "candidate_counts": {k: len(v) for k, v in rows.items()},
                }
            waist_u = np.asarray(frame["waist_u"], np.float64).reshape(2)
            waist_u /= max(float(np.linalg.norm(waist_u)), 1e-9)
            old_sep = float(np.linalg.norm(old_points["arm1"] - old_points["arm2"]))
            pair_candidates = []
            for a2 in rows["arm2"]:
                for a1 in rows["arm1"]:
                    q2 = np.asarray(a2["q"], np.float64)
                    q1 = np.asarray(a1["q"], np.float64)
                    sep = float(np.linalg.norm(q1 - q2))
                    if sep < 85.0:
                        continue
                    tangent_sep = abs(float(np.dot(q1 - q2, waist_u)))
                    if tangent_sep < 55.0:
                        continue
                    shift_delta = abs(float(a1["tangent_from_old_mm"]) - float(a2["tangent_from_old_mm"]))
                    inset_delta = abs(float(a1["edge_depth_mm"]) - float(a2["edge_depth_mm"]))
                    sep_delta = abs(sep - old_sep)
                    score = (
                        float(a1["score"]) + float(a2["score"])
                        + float(self._v90_waist_pair_shift_penalty) * shift_delta
                        + 0.12 * inset_delta
                        + float(self._v90_waist_pair_sep_change_penalty) * sep_delta
                    )
                    pair_candidates.append((float(score), a2, a1, sep, shift_delta, inset_delta, sep_delta))
            if not pair_candidates:
                return None, {"ok": False, "reason": "V90_NO_DUAL_WAIST_PAIR_AFTER_SEPARATION_GATE"}
            pair_candidates.sort(key=lambda row: row[0])
            failures = []
            for rank, (_, a2, a1, sep, shift_delta, inset_delta, sep_delta) in enumerate(pair_candidates[:80], 1):
                pair_rows = {"arm2": a2, "arm1": a1}
                candidate = self._v90_plan_with_pair(plan, old_points, pair_rows)
                if candidate is None:
                    continue
                inset_ok, inset_report = self._v53_validate_waist_safe_pair(obs, candidate, f"V90_PAIR_{rank}")
                if not inset_ok:
                    failures.append({"rank": rank, "reason": "INSET"})
                    continue
                rebuilt, why = self.d56._63_step120_motion_plan_from_d56_plan(candidate, self.config, self.cfg56, self.d56_args)
                if rebuilt is None:
                    failures.append({"rank": rank, "reason": f"MOTION_BUILD:{why}"})
                    continue
                metrics = dict(getattr(candidate, "metrics", {}) or {})
                rep = {
                    "ok": True,
                    "reason": "V90_REACHABLE_WAIST_PAIR",
                    "rank": int(rank),
                    "pair_separation_mm": float(sep),
                    "pair_separation_change_mm": float(sep_delta),
                    "rigid_shift_mismatch_mm": float(shift_delta),
                    "inset_difference_mm": float(inset_delta),
                    "motion_reason": str(why),
                    "arms": {
                        "arm2": {
                            "desired_inset_mm": float(desired["arm2"]),
                            "actual_inset_mm": float(a2["edge_depth_mm"]),
                            "tangent_shift_mm": float(a2["tangent_from_old_mm"]),
                            "before_board": old_points["arm2"].astype(float).tolist(),
                            "after_board": np.asarray(a2["q"], np.float64).astype(float).tolist(),
                        },
                        "arm1": {
                            "desired_inset_mm": float(desired["arm1"]),
                            "actual_inset_mm": float(a1["edge_depth_mm"]),
                            "tangent_shift_mm": float(a1["tangent_from_old_mm"]),
                            "before_board": old_points["arm1"].astype(float).tolist(),
                            "after_board": np.asarray(a1["q"], np.float64).astype(float).tolist(),
                        },
                    },
                    "inset_report": copy.deepcopy(inset_report),
                }
                metrics.update({
                    "bottom_vla_v81_deterministic_waist_pair": True,
                    "bottom_vla_v90_reachable_waist_pair": True,
                    "bottom_vla_v81_board_collision_only_veto": True,
                    "bottom_vla_v81_pair_report": copy.deepcopy(rep),
                })
                candidate.metrics = metrics
                print(
                    f"[D60-V90-REACHABLE-PAIR] rank={rank} A2 inset={float(a2['edge_depth_mm']):.1f}mm shift={float(a2['tangent_from_old_mm']):+.1f}mm | "
                    f"A1 inset={float(a1['edge_depth_mm']):.1f}mm shift={float(a1['tangent_from_old_mm']):+.1f}mm | "
                    f"pairSep={sep:.1f}mm rigidMismatch={shift_delta:.1f}mm"
                )
                return candidate, rep
            return None, {
                "ok": False,
                "reason": "V90_ALL_REACHABLE_PAIRS_FAILED_MOTION_COMPATIBILITY",
                "failures": failures[-12:],
            }

        _v92_prev_apply_deterministic_waist_pair = _v81_apply_deterministic_waist_pair

        def _v92_validate_base_waist_pair(self, obs, plan, label, require_motion_build):
            if plan is None:
                return False, None, {"ok": False, "reason": "V92_BASE_PLAN_UNAVAILABLE"}
            points = getattr(plan, "arm_points", None)
            if not isinstance(points, dict):
                return False, None, {"ok": False, "reason": "V92_BASE_ARM_POINTS_UNAVAILABLE"}
            grips = {}
            reach = {}
            for arm in ("arm2", "arm1"):
                item = points.get(arm, {})
                if not isinstance(item, dict):
                    return False, None, {"ok": False, "reason": f"V92_{arm.upper()}_ITEM_UNAVAILABLE"}
                q = None
                for key in ("grip_board", "source_board", "grasp_board", "target_board"):
                    q = self._finite_point2(item.get(key))
                    if q is not None:
                        break
                if q is None:
                    return False, None, {"ok": False, "reason": f"V92_{arm.upper()}_GRIP_UNAVAILABLE"}
                q = np.asarray(q, np.float64).reshape(2)
                if not self._v56_align_board_safe(q, 8.0):
                    return False, None, {
                        "ok": False,
                        "reason": f"V92_{arm.upper()}_BOARD_UNSAFE",
                        "point": q.astype(float).tolist(),
                    }
                reach_ok, reach_reason, reach_radius = self._v56_align_arm_reach(arm, q, final_target=False)
                if not reach_ok:
                    return False, None, {
                        "ok": False,
                        "reason": f"V92_{arm.upper()}_UNREACHABLE",
                        "reach_reason": str(reach_reason),
                        "reach_radius": float(reach_radius) if math.isfinite(float(reach_radius)) else None,
                    }
                grips[arm] = q
                reach[arm] = {
                    "reason": str(reach_reason),
                    "radius": float(reach_radius) if math.isfinite(float(reach_radius)) else None,
                }
            sep = float(np.linalg.norm(grips["arm1"] - grips["arm2"]))
            if sep < 85.0:
                return False, None, {
                    "ok": False,
                    "reason": "V92_BASE_DUAL_COLLISION_RISK",
                    "pair_separation_mm": sep,
                }
            safe_plan = copy.deepcopy(plan)
            old_required = float(self._v53_waist_hard_inset_mm)
            try:
                self._v53_waist_hard_inset_mm = max(35.0, old_required)
                inset_ok, inset_report = self._v53_validate_waist_safe_pair(obs, safe_plan, label)
            finally:
                self._v53_waist_hard_inset_mm = old_required
            if not inset_ok:
                return False, None, {
                    "ok": False,
                    "reason": "V92_BASE_INSET_FAILED",
                    "pair_separation_mm": sep,
                    "inset_report": copy.deepcopy(inset_report),
                }
            rebuilt = None
            why = "NOT_REQUESTED"
            if require_motion_build:
                rebuilt, why = self.d56._63_step120_motion_plan_from_d56_plan(
                    safe_plan, self.config, self.cfg56, self.d56_args
                )
                if rebuilt is None:
                    return False, None, {
                        "ok": False,
                        "reason": "V92_BASE_MOTION_BUILD_FAILED",
                        "motion_reason": str(why),
                        "pair_separation_mm": sep,
                        "inset_report": copy.deepcopy(inset_report),
                    }
            report = {
                "ok": True,
                "reason": "V92_SAFE_BASE_WAIST_PAIR",
                "pair_separation_mm": sep,
                "reach": reach,
                "inset_report": copy.deepcopy(inset_report),
                "motion_reason": str(why),
                "grips": {arm: grips[arm].astype(float).tolist() for arm in ("arm2", "arm1")},
            }
            return True, rebuilt, report

        def _v92_keep_safe_base_waist_pair(self, obs, plan, trigger_reason):
            ok, rebuilt, report = self._v92_validate_base_waist_pair(
                obs, plan, "V92_BASE_PAIR_GUARANTEE", True
            )
            if not ok or rebuilt is None:
                out = dict(report or {})
                out["v90_failure_reason"] = str(trigger_reason or "-")
                return None, None, out
            safe_plan = copy.deepcopy(plan)
            metrics = dict(getattr(safe_plan, "metrics", {}) or {})
            out = dict(report or {})
            out["v90_failure_reason"] = str(trigger_reason or "-")
            metrics.update({
                "bottom_vla_v92_safe_base_waist_pair": True,
                "bottom_vla_v92_safe_base_waist_report": copy.deepcopy(out),
            })
            safe_plan.metrics = metrics
            try:
                safe_plan.ok = True
            except Exception:
                pass
            g2 = np.asarray(out["grips"]["arm2"], np.float64)
            g1 = np.asarray(out["grips"]["arm1"], np.float64)
            print(
                f"[D60-V92-BASE-PAIR-GUARANTEE] V90 replacement failed={str(trigger_reason or '-')} -> "
                f"keep base pair A2=({g2[0]:.1f},{g2[1]:.1f}) A1=({g1[0]:.1f},{g1[1]:.1f}) "
                f"sep={float(out['pair_separation_mm']):.1f}mm; 35mm inset/board/reach/motion-build PASS"
            )
            return safe_plan, rebuilt, out

        def _v81_apply_deterministic_waist_pair(self, obs, plan):
            direct, drep = self._v92_prev_apply_deterministic_waist_pair(obs, plan)
            if direct is not None:
                return direct, drep
            trigger_reason = str((drep or {}).get("reason", "V90_REPLACEMENT_FAILED"))
            safe_plan, safe_motion, safe_report = self._v92_keep_safe_base_waist_pair(obs, plan, trigger_reason)
            if safe_plan is not None and safe_motion is not None:
                return safe_plan, safe_report
            out = dict(drep or {})
            out["bottom_vla_v92_base_pair_fallback"] = copy.deepcopy(safe_report or {})
            print(
                f"[D60-V92-BASE-PAIR-SKIP] V90 failed={trigger_reason}; "
                f"base pair unsafe/unbuildable reason={str((safe_report or {}).get('reason', '-'))}"
            )
            return None, out

        _v90_prev_force_waist_endpoint_plan = _d60_force_waist_endpoint_plan

        def _v91_safe_source_waist_fallback(self, obs, source_plan):
            if source_plan is None:
                return None, None, {"ok": False, "reason": "V91_SOURCE_PLAN_UNAVAILABLE"}
            points = getattr(source_plan, "arm_points", None)
            if not isinstance(points, dict):
                return None, None, {"ok": False, "reason": "V91_SOURCE_ARM_POINTS_UNAVAILABLE"}
            grips = {}
            reach = {}
            for arm in ("arm2", "arm1"):
                item = points.get(arm, {})
                if not isinstance(item, dict):
                    return None, None, {"ok": False, "reason": f"V91_{arm.upper()}_ITEM_UNAVAILABLE"}
                q = None
                for key in ("grip_board", "source_board", "grasp_board", "target_board"):
                    q = self._finite_point2(item.get(key))
                    if q is not None:
                        break
                if q is None:
                    return None, None, {"ok": False, "reason": f"V91_{arm.upper()}_GRIP_UNAVAILABLE"}
                q = np.asarray(q, np.float64).reshape(2)
                if not self._v56_align_board_safe(q, 8.0):
                    return None, None, {"ok": False, "reason": f"V91_{arm.upper()}_BOARD_UNSAFE", "point": q.astype(float).tolist()}
                reach_ok, reach_reason, reach_radius = self._v56_align_arm_reach(arm, q, final_target=False)
                if not reach_ok:
                    return None, None, {
                        "ok": False,
                        "reason": f"V91_{arm.upper()}_UNREACHABLE",
                        "reach_reason": str(reach_reason),
                        "reach_radius": float(reach_radius) if math.isfinite(float(reach_radius)) else None,
                    }
                grips[arm] = q
                reach[arm] = {
                    "reason": str(reach_reason),
                    "radius": float(reach_radius) if math.isfinite(float(reach_radius)) else None,
                }
            sep = float(np.linalg.norm(grips["arm1"] - grips["arm2"]))
            if sep < 85.0:
                return None, None, {"ok": False, "reason": "V91_SOURCE_DUAL_COLLISION_RISK", "pair_separation_mm": sep}
            safe_plan = copy.deepcopy(source_plan)
            inset_ok, inset_report = self._v53_validate_waist_safe_pair(obs, safe_plan, "V91_SOURCE_SAFE_FALLBACK")
            if not inset_ok:
                return None, None, {
                    "ok": False,
                    "reason": "V91_SOURCE_INSET_FAILED",
                    "pair_separation_mm": sep,
                    "inset_report": copy.deepcopy(inset_report),
                }
            rebuilt, why = self.d56._63_step120_motion_plan_from_d56_plan(
                safe_plan, self.config, self.cfg56, self.d56_args
            )
            if rebuilt is None:
                return None, None, {
                    "ok": False,
                    "reason": "V91_SOURCE_MOTION_BUILD_FAILED",
                    "motion_reason": str(why),
                    "pair_separation_mm": sep,
                }
            metrics = dict(getattr(safe_plan, "metrics", {}) or {})
            report = {
                "ok": True,
                "reason": "V91_SAFE_SOURCE_PLAN_FALLBACK",
                "pair_separation_mm": sep,
                "reach": reach,
                "inset_report": copy.deepcopy(inset_report),
                "motion_reason": str(why),
                "grips": {arm: grips[arm].astype(float).tolist() for arm in ("arm2", "arm1")},
            }
            metrics.update({
                "bottom_vla_v91_safe_source_waist_fallback": True,
                "bottom_vla_v91_safe_source_waist_report": copy.deepcopy(report),
            })
            safe_plan.metrics = metrics
            try:
                safe_plan.ok = True
            except Exception:
                pass
            print(
                f"[D60-V91-SAFE-SOURCE-FALLBACK] keep already-built safe WAIST pair "
                f"A2=({grips['arm2'][0]:.1f},{grips['arm2'][1]:.1f}) "
                f"A1=({grips['arm1'][0]:.1f},{grips['arm1'][1]:.1f}) sep={sep:.1f}mm; "
                f"35mm inset/board/reach/motion-build PASS"
            )
            return safe_plan, rebuilt, report

        def _d60_force_waist_endpoint_plan(self, obs, source_plan):
            planning_obs, refinement = self._v54_obs_with_refined_waist(obs)
            templates = []
            if source_plan is not None:
                templates.append(("SOURCE_PLAN", source_plan))
            for label, template in templates:
                direct, drep = self._v81_apply_deterministic_waist_pair(planning_obs, template)
                if direct is None:
                    print(f"[D60-V91-TEMPLATE-SKIP] {label} reason={str((drep or {}).get('reason', '-'))}")
                    continue
                rebuilt, why = self.d56._63_step120_motion_plan_from_d56_plan(direct, self.config, self.cfg56, self.d56_args)
                if rebuilt is not None:
                    print(f"[D60-V91-TEMPLATE-RECOVER] {label} produced executable waist-local pair")
                    return direct, rebuilt, {
                        "reason": "OK",
                        "bottom_vla_v91_template_recovery": label,
                        "bottom_vla_v90_pair_report": copy.deepcopy(drep),
                        "bottom_vla_v90_outer_refinement": copy.deepcopy(refinement),
                        "bottom_vla_v91_motion_reason": str(why),
                    }
            safe_source, safe_motion, safe_report = self._v91_safe_source_waist_fallback(planning_obs, source_plan)
            if safe_source is not None and safe_motion is not None:
                out = dict(safe_report or {})
                out["bottom_vla_v90_outer_refinement"] = copy.deepcopy(refinement)
                return safe_source, safe_motion, out
            print(f"[D60-V91-SOURCE-FALLBACK-SKIP] reason={str((safe_report or {}).get('reason', '-'))}")
            plan, motion, report = self._v90_prev_force_waist_endpoint_plan(planning_obs, source_plan)
            if plan is not None and motion is not None:
                out = dict(report or {})
                out["bottom_vla_v90_outer_refinement"] = copy.deepcopy(refinement)
                return plan, motion, out
            out = dict(report or {})
            out["bottom_vla_v90_outer_refinement"] = copy.deepcopy(refinement)
            if str(out.get("reason", "")).startswith("V81_"):
                out["reason"] = "V90_BOARD_REACH_OR_DUAL_COMPATIBILITY_BLOCK"
            print(f"[D60-V91-BLOCK] no executable waist-local pair after V91 direction-safe pair search + safe-source fallback reason={out.get('reason', '-')}")
            return None, None, out

        _v90_prev_waist_endpoint_gate = _d60_waist_endpoint_gate

        def _d60_waist_endpoint_gate(self, obs, plan):
            metrics = dict(getattr(plan, "metrics", {}) or {}) if plan is not None else {}
            if bool(metrics.get("bottom_vla_v92_safe_base_waist_pair", False)):
                ok, _, report = self._v92_validate_base_waist_pair(
                    obs, plan, "V92_ENDPOINT_GATE", False
                )
                if not ok:
                    return False, report
                return True, {
                    "ok": True,
                    "reason": "V92_SAFE_BASE_PAIR_ENDPOINT_GATE",
                    "bottom_vla_v92_safe_base_waist_report": copy.deepcopy(report),
                }
            if bool(metrics.get("bottom_vla_v90_reachable_waist_pair", False)):
                ok, report = self._v90_prev_waist_endpoint_gate(obs, plan)
                if not ok:
                    return ok, report
                inset = dict((metrics.get("bottom_vla_v81_pair_report", {}) or {}).get("arms", {}) or {})
                for arm in ("arm2", "arm1"):
                    depth = (inset.get(arm, {}) or {}).get("actual_inset_mm")
                    if depth is None or float(depth) > float(self._v90_waist_inset_hard_max_mm):
                        return False, {"ok": False, "reason": "V90_WAIST_GRIP_TOO_DEEP", "arm": arm, "inset_mm": depth}
                return True, {"ok": True, "reason": "V90_BOARD_REACH_DUAL_COMPATIBILITY_GATE"}
            return self._v90_prev_waist_endpoint_gate(obs, plan)


        def _v93_pose8_mask_ready(self, obs):
            names = (
                "waist_img_left", "waist_center", "waist_img_right", "crotch",
                "img_left_hem_outer", "img_left_hem_inner",
                "img_right_hem_inner", "img_right_hem_outer",
            )
            pts = self._pose_board_points(obs)
            missing = [name for name in names if self._finite_point2(pts.get(name)) is None]
            mask_obj = getattr(obs, "mask", None)
            mask_u8 = getattr(mask_obj, "mask_u8", None) if mask_obj is not None else None
            contour = getattr(mask_obj, "contour", None) if mask_obj is not None else None
            mask_ok = False
            try:
                mask_ok = bool(
                    mask_u8 is not None
                    and np.asarray(mask_u8).ndim == 2
                    and int(np.count_nonzero(np.asarray(mask_u8))) >= 500
                    and contour is not None
                    and len(np.asarray(contour).reshape(-1, 2)) >= 20
                )
            except Exception:
                mask_ok = False
            report = {
                "ok": bool(mask_ok and not missing),
                "mask_ok": bool(mask_ok),
                "pose8_count": int(8 - len(missing)),
                "missing_keypoints": list(missing),
            }
            if not mask_ok:
                report["reason"] = "V93_OUTER_MASK_UNAVAILABLE"
            elif missing:
                report["reason"] = "V93_POSE8_INCOMPLETE"
            else:
                report["reason"] = "V93_MASK_POSE8_READY"
            return bool(report["ok"]), pts, report

        def _v103_waist_pose_semantic_report(self, obs):
            pts = self._pose_board_points(obs)
            required = (
                "waist_img_left", "waist_center", "waist_img_right", "crotch",
                "img_left_hem_outer", "img_left_hem_inner",
                "img_right_hem_inner", "img_right_hem_outer",
            )
            clean = {}
            for name in required:
                p = self._finite_point2(pts.get(name))
                if p is None:
                    return False, {"ok": False, "reason": "V103_POSE_SEMANTIC_POINT_MISSING", "missing": name}
                clean[name] = np.asarray(p, np.float64).reshape(2)
            waist_center = 0.25 * clean["waist_img_left"] + 0.50 * clean["waist_center"] + 0.25 * clean["waist_img_right"]
            hem_center = np.mean(
                np.stack([
                    clean["img_left_hem_outer"], clean["img_left_hem_inner"],
                    clean["img_right_hem_inner"], clean["img_right_hem_outer"],
                ], axis=0),
                axis=0,
            )
            crotch = clean["crotch"]
            axis = hem_center - waist_center
            span = float(np.linalg.norm(axis))
            d_waist_crotch = float(np.linalg.norm(crotch - waist_center))
            d_hem_crotch = float(np.linalg.norm(hem_center - crotch))
            t_crotch = None
            outside_order = False
            if span >= 1e-6:
                t_crotch = float(np.dot(crotch - waist_center, axis) / max(1e-6, span * span))
                outside_order = bool(span >= 180.0 and (t_crotch < -0.15 or t_crotch > 1.15))
            clearly_inverted = bool(
                d_waist_crotch > max(
                    d_hem_crotch * float(self._v103_waist_semantic_inversion_ratio),
                    d_hem_crotch + float(self._v103_waist_semantic_inversion_margin_mm),
                )
            )
            ok = bool(not clearly_inverted and not outside_order)
            reason = "V103_POSE_SEMANTIC_OK" if ok else (
                "V103_POSE_WAIST_HEM_INVERTED" if clearly_inverted else "V103_POSE_BODY_ORDER_INVALID"
            )
            return ok, {
                "ok": bool(ok),
                "reason": reason,
                "waist_center_board": waist_center.astype(float).tolist(),
                "hem_center_board": hem_center.astype(float).tolist(),
                "crotch_board": crotch.astype(float).tolist(),
                "waist_to_crotch_mm": float(d_waist_crotch),
                "hem_to_crotch_mm": float(d_hem_crotch),
                "waist_to_hem_span_mm": float(span),
                "crotch_axis_t": None if t_crotch is None else float(t_crotch),
                "clearly_inverted": bool(clearly_inverted),
                "outside_order": bool(outside_order),
            }

        def _v93_mask_board_samples(self, obs, stride=4):
            mask_obj = getattr(obs, "mask", None)
            mask_u8 = getattr(mask_obj, "mask_u8", None) if mask_obj is not None else None
            if mask_u8 is None:
                return None
            mask = (np.asarray(mask_u8) > 0).astype(np.uint8)
            if mask.ndim != 2 or int(np.count_nonzero(mask)) < 500:
                return None
            stride = max(2, int(stride))
            small = mask[::stride, ::stride]
            yy, xx = np.nonzero(small)
            if len(xx) == 0:
                return None
            px = np.column_stack((xx.astype(np.float64) * stride, yy.astype(np.float64) * stride))
            try:
                dist = self.cv2.distanceTransform(mask, self.cv2.DIST_L2, 3)
                xi = np.clip(np.rint(px[:, 0]).astype(np.int32), 0, mask.shape[1] - 1)
                yi = np.clip(np.rint(px[:, 1]).astype(np.int32), 0, mask.shape[0] - 1)
                depth_px = np.asarray(dist[yi, xi], np.float64)
            except Exception:
                depth_px = np.ones((len(px),), np.float64)
            Hm = np.asarray(self.H_raw, np.float64).reshape(3, 3)
            homo = np.column_stack((px[:, 0], px[:, 1], np.ones(len(px), np.float64)))
            out = homo @ Hm.T
            good = np.isfinite(out).all(axis=1) & (np.abs(out[:, 2]) > 1e-8)
            if int(np.count_nonzero(good)) == 0:
                return None
            board = out[good, :2] / out[good, 2:3]
            return {
                "board": np.asarray(board, np.float64),
                "px": np.asarray(px[good], np.float64),
                "depth_px": np.asarray(depth_px[good], np.float64),
            }

        def _v93_direct_pair_options(self, obs):
            started = time.perf_counter()
            ready, pts, ready_report = self._v93_pose8_mask_ready(obs)
            report = dict(ready_report)
            report["mode"] = "V99_REAL_MASK_WAIST_BALANCED_ARM2_INSET70"
            if not ready:
                report["generation_ms"] = float((time.perf_counter() - started) * 1000.0)
                return [], report
            pose_wl = np.asarray(pts["waist_img_left"], np.float64).reshape(2)
            pose_wc = np.asarray(pts["waist_center"], np.float64).reshape(2)
            pose_wr = np.asarray(pts["waist_img_right"], np.float64).reshape(2)
            crotch = np.asarray(pts["crotch"], np.float64).reshape(2)

            outer_ref = self._v81_pose_constrained_outer_waist(obs)
            use_outer = bool(isinstance(outer_ref, dict) and outer_ref.get("ok", False))
            if use_outer:
                wl = np.asarray(outer_ref["waist_left"], np.float64).reshape(2)
                wc = np.asarray(outer_ref["waist_center"], np.float64).reshape(2)
                wr = np.asarray(outer_ref["waist_right"], np.float64).reshape(2)
                report["waist_reference_source"] = str(outer_ref.get("reason", "REAL_MASK_OUTER_WAIST"))
                report["pose_waist_left_board"] = pose_wl.astype(float).tolist()
                report["pose_waist_center_board"] = pose_wc.astype(float).tolist()
                report["pose_waist_right_board"] = pose_wr.astype(float).tolist()
                print(
                    f"[D60-V99-REAL-WAIST] real Mask outer waist USED "
                    f"poseL=({pose_wl[0]:.1f},{pose_wl[1]:.1f}) poseR=({pose_wr[0]:.1f},{pose_wr[1]:.1f}) "
                    f"maskL=({wl[0]:.1f},{wl[1]:.1f}) maskR=({wr[0]:.1f},{wr[1]:.1f})"
                )
            else:
                wl, wc, wr = pose_wl, pose_wc, pose_wr
                report["waist_reference_source"] = "POSE_FALLBACK_OUTER_REF_UNAVAILABLE"
                report["waist_outer_ref_failure"] = copy.deepcopy(outer_ref if isinstance(outer_ref, dict) else {})
                print(
                    f"[D60-V99-REAL-WAIST-FALLBACK] outer Mask waist unavailable "
                    f"reason={str((outer_ref or {}).get('reason','UNKNOWN'))}; using Pose waist anchors"
                )

            waist_vec = wr - wl
            waist_width = float(np.linalg.norm(waist_vec))
            if not math.isfinite(waist_width) or waist_width < 20.0:
                waist_vec = pose_wr - pose_wl
                waist_width = max(20.0, float(np.linalg.norm(waist_vec)))
            waist_u = waist_vec / max(1e-6, float(np.linalg.norm(waist_vec)))

            ref_body = None
            if use_outer:
                try:
                    ref_body = np.asarray(outer_ref.get("body_u"), np.float64).reshape(2)
                    if not np.isfinite(ref_body).all() or float(np.linalg.norm(ref_body)) < 1e-6:
                        ref_body = None
                except Exception:
                    ref_body = None
            body_vec = ref_body if ref_body is not None else (crotch - wc)
            if float(np.linalg.norm(body_vec)) < 10.0:
                mask_center = self._finite_point2(getattr(getattr(obs, "mask", None), "center_board", None))
                body_vec = (mask_center - wc) if mask_center is not None else np.asarray([0.0, -1.0], np.float64)
            body_u = body_vec / max(1e-6, float(np.linalg.norm(body_vec)))
            plus = self._v53_mask_point_report(obs, wc + body_u * 35.0)
            minus = self._v53_mask_point_report(obs, wc - body_u * 35.0)
            if not bool((plus or {}).get("ok", False)) and bool((minus or {}).get("ok", False)):
                body_u = -body_u
            elif bool((plus or {}).get("ok", False)) == bool((minus or {}).get("ok", False)):
                mask_center = self._finite_point2(getattr(getattr(obs, "mask", None), "center_board", None))
                if mask_center is not None and float(np.dot(mask_center - wc, body_u)) < 0.0:
                    body_u = -body_u
            waist_normal_u = np.asarray([-waist_u[1], waist_u[0]], np.float64)
            if float(np.dot(waist_normal_u, body_u)) < 0.0:
                waist_normal_u = -waist_normal_u
            samples = self._v93_mask_board_samples(obs, stride=4)
            if samples is None:
                report["reason"] = "V94_MASK_INTERIOR_SAMPLES_UNAVAILABLE"
                report["generation_ms"] = float((time.perf_counter() - started) * 1000.0)
                return [], report
            board = np.asarray(samples["board"], np.float64)
            px = np.asarray(samples["px"], np.float64)
            depth_px = np.asarray(samples["depth_px"], np.float64)
            rel = board - wc.reshape(1, 2)
            sproj = rel @ waist_u
            dproj_body = rel @ body_u
            dproj_local = rel @ waist_normal_u
            if float(wl[0]) <= float(wr[0]):
                seeds = {"arm2": wl, "arm1": wr}
            else:
                seeds = {"arm2": wr, "arm1": wl}
            split_x = float(getattr(self.cfg56, "split_board_x", 0.5 * (float(self.cfg56.board_x_min) + float(self.cfg56.board_x_max))))
            dead = max(0.0, float(getattr(self.cfg56, "center_dead_half_width", 0.0)))
            target_depth = 50.0
            tangent_band = max(70.0, min(130.0, 0.42 * max(120.0, waist_width)))
            arm_rows = {}
            arm_meta = {}

            def build_rows(arm, target_local_depth=None):
                seed = np.asarray(seeds[arm], np.float64).reshape(2)
                seed_s = float(np.dot(seed - wc, waist_u))
                target_s = float(seed_s)
                tangent_inset = 0.0
                if arm == "arm2":
                    tangent_inset = min(float(self._v94_arm2_waist_tangent_inset_mm), abs(float(seed_s)))
                    if seed_s > 0.0:
                        target_s = seed_s - tangent_inset
                    elif seed_s < 0.0:
                        target_s = seed_s + tangent_inset
                depth_target = float(target_local_depth if target_local_depth is not None and math.isfinite(float(target_local_depth)) else target_depth)
                tangent_ref = target_s if arm == "arm2" else seed_s
                desired = wc + waist_u * tangent_ref + waist_normal_u * depth_target
                q_depth_axis = dproj_local
                tangent_delta = np.abs(sproj - tangent_ref)
                strict = (tangent_delta <= tangent_band) & (q_depth_axis >= 12.0) & (q_depth_axis <= 105.0)
                relaxed = (tangent_delta <= max(160.0, tangent_band * 1.7)) & (q_depth_axis >= -15.0) & (q_depth_axis <= 145.0)
                candidate_ids = np.flatnonzero(strict)
                band_name = "STRICT_WAIST_BAND"
                if len(candidate_ids) < 8:
                    candidate_ids = np.flatnonzero(relaxed)
                    band_name = "RELAXED_WAIST_BAND"
                if len(candidate_ids) < 4:
                    candidate_ids = np.arange(len(board), dtype=np.int64)
                    band_name = "WHOLE_MASK_LAST_RESORT"
                if arm == "arm2":
                    own_hard = board[candidate_ids, 0] <= split_x - dead
                    own_soft = board[candidate_ids, 0] <= split_x
                else:
                    own_hard = board[candidate_ids, 0] >= split_x + dead
                    own_soft = board[candidate_ids, 0] >= split_x
                if int(np.count_nonzero(own_hard)) >= 4:
                    candidate_ids = candidate_ids[own_hard]
                    ownership = "PRIVATE_SIDE_PREFERRED"
                elif int(np.count_nonzero(own_soft)) >= 4:
                    candidate_ids = candidate_ids[own_soft]
                    ownership = "BOARD_HALF_PREFERRED"
                else:
                    ownership = "NO_OWNERSHIP_FILTER"
                q = board[candidate_ids]
                q_depth = depth_px[candidate_ids]
                q_axis_depth = q_depth_axis[candidate_ids]
                q_local_depth = dproj_local[candidate_ids]
                q_s = sproj[candidate_ids]
                dist_desired = np.linalg.norm(q - desired.reshape(1, 2), axis=1)
                tangent_err = np.abs(q_s - tangent_ref)
                body_err = np.abs(q_axis_depth - depth_target)
                depth_bonus = np.minimum(q_depth, 18.0)
                if arm == "arm2":
                    wrong = np.maximum(0.0, q[:, 0] - split_x)
                else:
                    wrong = np.maximum(0.0, split_x - q[:, 0])
                score = dist_desired + 0.28 * tangent_err + 0.30 * body_err + 1.8 * wrong - 1.25 * depth_bonus
                actual_inset_all = np.zeros(len(candidate_ids), np.float64)
                inset_penalty_all = np.zeros(len(candidate_ids), np.float64)
                inset_mode_all = np.asarray(["NOT_APPLICABLE"] * len(candidate_ids), dtype=object)
                if arm == "arm2":
                    direction_to_center = -1.0 if seed_s > 0.0 else (1.0 if seed_s < 0.0 else 0.0)
                    actual_inset_all = (q_s - seed_s) * direction_to_center
                    pmin = float(self._v101_waist_arm2_preferred_inset_min_mm)
                    pmax = float(self._v101_waist_arm2_preferred_inset_max_mm)
                    rmin = float(self._v101_waist_arm2_relaxed_inset_min_mm)
                    rmax = float(self._v101_waist_arm2_relaxed_inset_max_mm)
                    target_inset = float(self._v101_waist_arm2_target_inset_mm)
                    for jj, actual in enumerate(actual_inset_all.tolist()):
                        if pmin <= actual <= pmax:
                            inset_mode_all[jj] = "PREFERRED_65_75"
                            inset_penalty_all[jj] = abs(actual - target_inset)
                        elif rmin <= actual <= rmax:
                            inset_mode_all[jj] = "RELAXED_55_85"
                            inset_penalty_all[jj] = 45.0 + 1.5 * abs(actual - target_inset)
                        elif actual > 0.0:
                            inset_mode_all[jj] = "FALLBACK_POSITIVE"
                            inset_penalty_all[jj] = 115.0 + 2.0 * abs(actual - target_inset)
                        else:
                            inset_mode_all[jj] = "FALLBACK_EXISTING"
                            inset_penalty_all[jj] = 180.0 + 2.0 * abs(actual - target_inset)
                    score = score + inset_penalty_all
                order = np.argsort(score)[:32]
                rows = []
                for oi in order:
                    idx = int(candidate_ids[int(oi)])
                    q_s_value = float(sproj[idx])
                    actual_inset = float(actual_inset_all[int(oi)]) if arm == "arm2" else 0.0
                    inset_mode = str(inset_mode_all[int(oi)]) if arm == "arm2" else "NOT_APPLICABLE"
                    inset_pref_penalty = float(inset_penalty_all[int(oi)]) if arm == "arm2" else 0.0
                    rows.append({
                        "q": np.asarray(board[idx], np.float64),
                        "px": np.asarray(px[idx], np.float64),
                        "score": float(score[int(oi)]),
                        "depth_px": float(depth_px[idx]),
                        "body_depth_mm": float(dproj_body[idx]),
                        "local_waist_depth_mm": float(dproj_local[idx]),
                        "tangent_from_center_mm": q_s_value,
                        "seed_s_mm": float(seed_s),
                        "target_s_mm": float(target_s),
                        "arm2_tangent_inset_mm": float(tangent_inset),
                        "arm2_actual_inset_mm": float(actual_inset),
                        "arm2_inset_mode": str(inset_mode),
                        "arm2_inset_pref_penalty": float(inset_pref_penalty),
                    })
                arm_rows[arm] = rows
                arm_meta[arm] = {
                    "band": band_name,
                    "ownership": ownership,
                    "candidate_count": int(len(candidate_ids)),
                    "seed_board": seed.astype(float).tolist(),
                    "desired_board": desired.astype(float).tolist(),
                    "seed_s_mm": float(seed_s),
                    "target_s_mm": float(target_s),
                    "tangent_inset_mm": float(tangent_inset),
                    "target_local_depth_mm": float(depth_target),
                }

            build_rows("arm1")
            if not arm_rows.get("arm1"):
                report["reason"] = "V94_DIRECT_ARM1_CANDIDATE_EMPTY"
                report["arm_meta"] = arm_meta
                report["generation_ms"] = float((time.perf_counter() - started) * 1000.0)
                return [], report
            arm1_anchor_depth = float(arm_rows["arm1"][0].get("local_waist_depth_mm", target_depth))
            if not math.isfinite(arm1_anchor_depth) or arm1_anchor_depth < 12.0 or arm1_anchor_depth > 105.0:
                arm1_anchor_depth = target_depth
            build_rows("arm2", target_local_depth=arm1_anchor_depth)
            if not arm_rows.get("arm2"):
                report["reason"] = "V94_DIRECT_ARM2_CANDIDATE_EMPTY"
                report["arm_meta"] = arm_meta
                report["generation_ms"] = float((time.perf_counter() - started) * 1000.0)
                return [], report
            base_arm2_rows = copy.deepcopy(arm_rows.get("arm2", []))
            base_arm2_meta = copy.deepcopy(arm_meta.get("arm2", {}))
            min_actual_inset = float(self._v107_waist_arm2_min_preferred_actual_inset_mm)
            best_initial_actual = max(
                [float(row.get("arm2_actual_inset_mm", -1e9)) for row in base_arm2_rows] or [-1e9]
            )
            bodyward_rescue_used = 0.0
            if best_initial_actual < min_actual_inset:
                rescued_rows = None
                rescued_meta = None
                for bodyward_extra in self._v107_waist_arm2_bodyward_steps_mm:
                    target_depth_rescue = float(np.clip(arm1_anchor_depth + float(bodyward_extra), 12.0, 105.0))
                    build_rows("arm2", target_local_depth=target_depth_rescue)
                    candidate_rows = copy.deepcopy(arm_rows.get("arm2", []))
                    preferred_rows = [
                        row for row in candidate_rows
                        if float(row.get("arm2_actual_inset_mm", -1e9)) >= min_actual_inset
                    ]
                    if preferred_rows:
                        for row in preferred_rows:
                            row["arm2_bodyward_rescue_mm"] = float(bodyward_extra)
                            row["arm2_inset_mode"] = "V107_BODYWARD_INSET_RESCUE"
                        other_rows = [
                            row for row in candidate_rows
                            if float(row.get("arm2_actual_inset_mm", -1e9)) < min_actual_inset
                        ]
                        rescued_rows = preferred_rows + other_rows
                        rescued_meta = copy.deepcopy(arm_meta.get("arm2", {}))
                        bodyward_rescue_used = float(bodyward_extra)
                        print(
                            f"[D60-V107-ARM2-BODYWARD] waist-line inset best={best_initial_actual:.1f}mm < {min_actual_inset:.1f}mm; "
                            f"retry bodyward=+{bodyward_extra:.0f}mm selected candidateInset={float(preferred_rows[0].get('arm2_actual_inset_mm',0.0)):.1f}mm"
                        )
                        break
                if rescued_rows is not None:
                    arm_rows["arm2"] = rescued_rows
                    arm_meta["arm2"] = rescued_meta if rescued_meta is not None else arm_meta.get("arm2", {})
                    arm_meta["arm2"]["v107_bodyward_rescue_mm"] = float(bodyward_rescue_used)
                else:
                    arm_rows["arm2"] = base_arm2_rows
                    arm_meta["arm2"] = base_arm2_meta
                    print(
                        f"[D60-V107-ARM2-BODYWARD-FALLBACK] no >= {min_actual_inset:.1f}mm inset bodyward candidate; "
                        f"keep legacy fallback so WAIST grasp generation is not lost"
                    )
            desired_sep = float(np.clip(max(120.0, 0.72 * waist_width), 120.0, 260.0))
            primary_pairs = []
            secondary_pairs = []
            balance_max = float(self._v94_waist_depth_balance_max_mm)
            balance_gain = float(self._v94_waist_depth_balance_penalty)
            for i1, r1 in enumerate(arm_rows["arm1"]):
                for i2, r2 in enumerate(arm_rows["arm2"]):
                    q2 = np.asarray(r2["q"], np.float64)
                    q1 = np.asarray(r1["q"], np.float64)
                    sep = float(np.linalg.norm(q1 - q2))
                    close_penalty = 0.0 if sep >= 85.0 else 1200.0 + 20.0 * (85.0 - sep)
                    order_penalty = 0.0 if float(q2[0]) < float(q1[0]) else 700.0 + 3.0 * abs(float(q2[0] - q1[0]))
                    d1 = float(r1.get("local_waist_depth_mm", 0.0))
                    d2 = float(r2.get("local_waist_depth_mm", 0.0))
                    depth_diff = abs(d2 - d1)
                    bodyward_rescue_mm = float(r2.get("arm2_bodyward_rescue_mm", 0.0) or 0.0)
                    if bodyward_rescue_mm > 0.0:
                        balance_penalty = 0.35 * balance_gain * depth_diff + 2.0 * max(0.0, depth_diff - 80.0)
                    else:
                        balance_penalty = balance_gain * depth_diff + 12.0 * max(0.0, depth_diff - balance_max)
                    arm1_change_penalty = 35.0 * float(i1)
                    a2_actual_inset = float(r2.get("arm2_actual_inset_mm", 0.0))
                    inset_shortfall_penalty = 20.0 * max(
                        0.0, float(self._v107_waist_arm2_min_preferred_actual_inset_mm) - a2_actual_inset
                    )
                    pair_score = float(
                        r2["score"] + r1["score"] + close_penalty + order_penalty
                        + 0.035 * abs(sep - desired_sep) + balance_penalty + arm1_change_penalty
                        + inset_shortfall_penalty
                    )
                    row = {
                        "score": pair_score,
                        "separation_mm": sep,
                        "local_depth_diff_mm": float(depth_diff),
                        "arm2_tangent_inset_mm": float(r2.get("arm2_tangent_inset_mm", 0.0)),
                        "arm2_actual_inset_mm": float(r2.get("arm2_actual_inset_mm", 0.0)),
                        "arm2_inset_mode": str(r2.get("arm2_inset_mode", "FALLBACK_EXISTING")),
                        "arm2_bodyward_rescue_mm": float(r2.get("arm2_bodyward_rescue_mm", 0.0) or 0.0),
                        "arm2": r2,
                        "arm1": r1,
                    }
                    if i1 == 0:
                        primary_pairs.append(row)
                    else:
                        secondary_pairs.append(row)
            primary_pairs.sort(key=lambda row: float(row["score"]))
            secondary_pairs.sort(key=lambda row: float(row["score"]))
            balanced_primary = [
                row for row in primary_pairs
                if float(row.get("local_depth_diff_mm", 9999.0)) <= (
                    80.0 if float(row.get("arm2_bodyward_rescue_mm", 0.0) or 0.0) > 0.0 else balance_max
                )
            ]
            balanced_secondary = [
                row for row in secondary_pairs
                if float(row.get("local_depth_diff_mm", 9999.0)) <= (
                    80.0 if float(row.get("arm2_bodyward_rescue_mm", 0.0) or 0.0) > 0.0 else balance_max
                )
            ]
            if balanced_primary or balanced_secondary:
                options = balanced_primary[:6] + balanced_secondary[:2]
            else:
                combined = primary_pairs + secondary_pairs
                combined.sort(key=lambda row: (float(row.get("local_depth_diff_mm", 9999.0)), float(row["score"])))
                options = combined[:8]
            if not options:
                report["reason"] = "V94_DIRECT_PAIR_EMPTY"
                report["arm_meta"] = arm_meta
                report["generation_ms"] = float((time.perf_counter() - started) * 1000.0)
                return [], report
            report.update({
                "ok": True,
                "reason": "V94_DIRECT_PAIR_OPTIONS_READY",
                "waist_width_mm": float(waist_width),
                "waist_center_board": wc.astype(float).tolist(),
                "waist_u": waist_u.astype(float).tolist(),
                "waist_normal_u": waist_normal_u.astype(float).tolist(),
                "body_u": body_u.astype(float).tolist(),
                "arm1_anchor_local_depth_mm": float(arm1_anchor_depth),
                "arm2_tangent_inset_mm": float(self._v94_arm2_waist_tangent_inset_mm),
                "depth_balance_max_mm": float(balance_max),
                "arm_meta": arm_meta,
                "sample_count": int(len(board)),
                "option_count": int(len(options)),
                "generation_ms": float((time.perf_counter() - started) * 1000.0),
            })
            return options, report

        def _v93_plan_from_direct_pair(self, obs, pair_row, pair_report):
            arm_points = {}
            selected = {}
            for arm in ("arm2", "arm1"):
                row = pair_row[arm]
                q = np.asarray(row["q"], np.float64).reshape(2)
                px = np.asarray(row["px"], np.float64).reshape(2)
                exact = self._v53_mask_point_report(obs, q)
                arm_points[arm] = {
                    "grip_board": (float(q[0]), float(q[1])),
                    "source_board": (float(q[0]), float(q[1])),
                    "grasp_board": (float(q[0]), float(q[1])),
                    "target_board": (float(q[0]), float(q[1])),
                    "grip_px": (float(px[0]), float(px[1])),
                }
                selected[arm] = {
                    "grip_board": q.astype(float).tolist(),
                    "grip_px": px.astype(float).tolist(),
                    "edge_depth_mm": (None if exact.get("edge_depth_mm") is None else float(exact.get("edge_depth_mm"))),
                    "inside_real_mask": bool(exact.get("ok", False)),
                    "sample_depth_px": float(row.get("depth_px", 0.0)),
                    "body_depth_mm": float(row.get("body_depth_mm", 0.0)),
                    "local_waist_depth_mm": float(row.get("local_waist_depth_mm", 0.0)),
                    "tangent_from_center_mm": float(row.get("tangent_from_center_mm", 0.0)),
                    "arm2_tangent_inset_mm": float(row.get("arm2_tangent_inset_mm", 0.0)),
                }
            metrics = {
                "d42_plan_mode": "D56_15_WAIST_RIBBON",
                "bottom_vla_v93_direct_forced_pair": True,
                "bottom_vla_v93_mask_pose8_ready": True,
                "bottom_vla_v93_pair_report": copy.deepcopy(pair_report),
                "bottom_vla_v93_selected": copy.deepcopy(selected),
                "initial_grip_separation_mm": float(pair_row.get("separation_mm", 0.0)),
                "waist_center_board": copy.deepcopy(pair_report.get("waist_center_board")),
                "waist_width_mm": float(pair_report.get("waist_width_mm", 0.0)),
            }
            plan_cls = getattr(self.d56, "D31DualGraspPlan", None)
            if not callable(plan_cls):
                return None
            plan = plan_cls(
                True,
                "V99 balanced direct waist pair from real Mask waist + Pose8 anchor",
                arm_points=arm_points,
                score=float(-float(pair_row.get("score", 0.0))),
                metrics=metrics,
                safety_warnings=[],
            )
            return plan

        def _v102_relocated_waist_option(self, obs, option, report, shift_mm):
            try:
                waist_u = np.asarray(report.get("waist_u"), np.float64).reshape(2)
                waist_center = np.asarray(report.get("waist_center_board"), np.float64).reshape(2)
                if not np.isfinite(waist_u).all() or float(np.linalg.norm(waist_u)) < 1e-6:
                    return None
                waist_u = waist_u / float(np.linalg.norm(waist_u))
                out = copy.deepcopy(option)
                for arm in ("arm2", "arm1"):
                    q = np.asarray(out[arm]["q"], np.float64).reshape(2) + waist_u * float(shift_mm)
                    rep = self._v53_mask_point_report(obs, q)
                    if not bool((rep or {}).get("ok", False)):
                        return None
                    if not (float(self.cfg56.board_x_min) <= float(q[0]) <= float(self.cfg56.board_x_max)):
                        return None
                    if not (float(self.cfg56.board_y_min) <= float(q[1]) <= float(self.cfg56.board_y_max)):
                        return None
                    out[arm]["q"] = q
                    try:
                        px = self.d56.board_to_pixel(self.H_raw, float(q[0]), float(q[1]))
                        if px is not None:
                            out[arm]["px"] = np.asarray(px, np.float64).reshape(2)
                    except Exception:
                        pass
                    out[arm]["tangent_from_center_mm"] = float(np.dot(q - waist_center, waist_u))
                q2 = np.asarray(out["arm2"]["q"], np.float64).reshape(2)
                q1 = np.asarray(out["arm1"]["q"], np.float64).reshape(2)
                out["separation_mm"] = float(np.linalg.norm(q1 - q2))
                seed_s = float(out["arm2"].get("seed_s_mm", 0.0))
                q_s = float(out["arm2"].get("tangent_from_center_mm", 0.0))
                direction_to_center = -1.0 if seed_s > 0.0 else (1.0 if seed_s < 0.0 else 0.0)
                actual = float((q_s - seed_s) * direction_to_center) if direction_to_center != 0.0 else 0.0
                out["arm2"]["arm2_actual_inset_mm"] = actual
                out["arm2_actual_inset_mm"] = actual
                out["arm2_inset_mode"] = "V102_RIGID_RELOCATED"
                out["v102_rigid_shift_mm"] = float(shift_mm)
                return out
            except Exception:
                return None

        def _v103_relocated_waist_option(self, obs, option, report, tangent_shift_mm, normal_shift_mm):
            try:
                waist_u = np.asarray(report.get("waist_u"), np.float64).reshape(2)
                normal_u = np.asarray(report.get("waist_normal_u"), np.float64).reshape(2)
                waist_center = np.asarray(report.get("waist_center_board"), np.float64).reshape(2)
                if not np.isfinite(waist_u).all() or float(np.linalg.norm(waist_u)) < 1e-6:
                    return None
                if not np.isfinite(normal_u).all() or float(np.linalg.norm(normal_u)) < 1e-6:
                    return None
                waist_u = waist_u / float(np.linalg.norm(waist_u))
                normal_u = normal_u / float(np.linalg.norm(normal_u))
                delta = waist_u * float(tangent_shift_mm) + normal_u * float(normal_shift_mm)
                out = copy.deepcopy(option)
                for arm in ("arm2", "arm1"):
                    q = np.asarray(out[arm]["q"], np.float64).reshape(2) + delta
                    rep = self._v53_mask_point_report(obs, q)
                    if not bool((rep or {}).get("ok", False)):
                        return None
                    if not (float(self.cfg56.board_x_min) <= float(q[0]) <= float(self.cfg56.board_x_max)):
                        return None
                    if not (float(self.cfg56.board_y_min) <= float(q[1]) <= float(self.cfg56.board_y_max)):
                        return None
                    out[arm]["q"] = q
                    try:
                        px = self.d56.board_to_pixel(self.H_raw, float(q[0]), float(q[1]))
                        if px is not None:
                            out[arm]["px"] = np.asarray(px, np.float64).reshape(2)
                    except Exception:
                        pass
                    out[arm]["tangent_from_center_mm"] = float(np.dot(q - waist_center, waist_u))
                    out[arm]["local_waist_depth_mm"] = float(np.dot(q - waist_center, normal_u))
                q2 = np.asarray(out["arm2"]["q"], np.float64).reshape(2)
                q1 = np.asarray(out["arm1"]["q"], np.float64).reshape(2)
                out["separation_mm"] = float(np.linalg.norm(q1 - q2))
                seed_s = float(out["arm2"].get("seed_s_mm", 0.0))
                q_s = float(out["arm2"].get("tangent_from_center_mm", 0.0))
                direction_to_center = -1.0 if seed_s > 0.0 else (1.0 if seed_s < 0.0 else 0.0)
                actual = float((q_s - seed_s) * direction_to_center) if direction_to_center != 0.0 else 0.0
                out["arm2"]["arm2_actual_inset_mm"] = actual
                out["arm2_actual_inset_mm"] = actual
                out["arm2_inset_mode"] = "V103_RIGID_2D_RELOCATED"
                out["v103_tangent_shift_mm"] = float(tangent_shift_mm)
                out["v103_normal_shift_mm"] = float(normal_shift_mm)
                return out
            except Exception:
                return None

        def _v103_waist_relocation_offsets(self):
            rows = []
            seen = set()
            def add(tangent_mm, normal_mm):
                key = (round(float(tangent_mm), 3), round(float(normal_mm), 3))
                if key == (0.0, 0.0) or key in seen:
                    return
                seen.add(key)
                rows.append((float(tangent_mm), float(normal_mm)))
            for normal_mm in self._v103_waist_relocate_normal_steps_mm:
                add(0.0, normal_mm)
            for tangent_mm in self._v102_waist_relocate_steps_mm:
                add(tangent_mm, 0.0)
            for normal_mm in self._v103_waist_relocate_combo_normal_mm:
                for tangent_mm in self._v103_waist_relocate_combo_tangent_mm:
                    if math.hypot(float(tangent_mm), float(normal_mm)) <= 50.0:
                        add(tangent_mm, normal_mm)
            return rows

        def _v102_waist_failure_is_relocatable(self, failures):
            reasons = [str(row.get("reason", "")) for row in failures if isinstance(row, dict)]
            return any(any(token in reason for token in self._v102_waist_relocate_soft_tokens) for reason in reasons)

        def _v105_waist_board_inside(self, q):
            p = self._finite_point2(q)
            if p is None:
                return False
            return bool(
                float(self.cfg56.board_x_min) <= float(p[0]) <= float(self.cfg56.board_x_max)
                and float(self.cfg56.board_y_min) <= float(p[1]) <= float(self.cfg56.board_y_max)
            )

        def _v105_waist_reach_clamp(self, obs, arm_key, q):
            p = np.asarray(q, np.float64).reshape(2)
            try:
                rx, ry = self.d56.board_to_arm_xy(self.config, arm_key, float(p[0]), float(p[1]))
                radius = float(math.hypot(float(rx), float(ry)))
            except Exception as exc:
                return None, {"reason": f"AFFINE:{type(exc).__name__}"}
            limit = float(self._v105_waist_relaxed_reach_max_mm)
            if radius <= limit + 1e-6:
                return p.copy(), {"reason": "UNCHANGED", "radius_before_mm": radius, "radius_after_mm": radius, "shift_mm": 0.0}
            try:
                arm_cfg = dict(self.config.get("dual_roarm", {}).get(arm_key, {}) or {})
                M = np.asarray(arm_cfg.get("board_to_roarm_affine_2x3"), np.float64).reshape(2, 3)
                A = M[:, :2]
                b = M[:, 2]
                off = arm_cfg.get("roarm_xy_offset", [0.0, 0.0])
                offset = np.asarray([
                    float(off[0]) if isinstance(off, (list, tuple)) and len(off) >= 1 else 0.0,
                    float(off[1]) if isinstance(off, (list, tuple)) and len(off) >= 2 else 0.0,
                ], np.float64)
                desired_r = min(float(self._v105_waist_reach_clamp_target_mm), limit - 2.0)
                scale = desired_r / max(radius, 1e-6)
                desired_xy = np.asarray([float(rx), float(ry)], np.float64) * scale
                target_board = np.linalg.solve(A, desired_xy - offset - b)
            except Exception as exc:
                return None, {"reason": f"INVERSE_AFFINE:{type(exc).__name__}", "radius_before_mm": radius}
            shift = float(np.linalg.norm(target_board - p))
            if shift > float(self._v105_waist_reach_clamp_max_shift_mm) + 1e-6:
                return None, {"reason": "CLAMP_SHIFT_TOO_LARGE", "radius_before_mm": radius, "shift_mm": shift}
            if not self._v105_waist_board_inside(target_board):
                return None, {"reason": "CLAMP_OUTSIDE_BOARD", "radius_before_mm": radius, "shift_mm": shift}
            rep = self._v53_mask_point_report(obs, target_board)
            if not bool((rep or {}).get("ok", False)):
                return None, {"reason": "CLAMP_OUTSIDE_MASK", "radius_before_mm": radius, "shift_mm": shift}
            try:
                nrx, nry = self.d56.board_to_arm_xy(self.config, arm_key, float(target_board[0]), float(target_board[1]))
                radius_after = float(math.hypot(float(nrx), float(nry)))
            except Exception:
                return None, {"reason": "CLAMP_VERIFY_AFFINE", "radius_before_mm": radius, "shift_mm": shift}
            if radius_after > limit + 1e-6:
                return None, {"reason": "CLAMP_STILL_UNREACHABLE", "radius_before_mm": radius, "radius_after_mm": radius_after, "shift_mm": shift}
            return target_board.astype(np.float64), {
                "reason": "CLAMPED",
                "radius_before_mm": radius,
                "radius_after_mm": radius_after,
                "shift_mm": shift,
            }

        def _v105_waist_relaxed_option(self, obs, option):
            out = copy.deepcopy(option)
            clamp_diag = {}
            for arm_key in ("arm2", "arm1"):
                q = self._finite_point2(out.get(arm_key, {}).get("q"))
                if q is None or not self._v105_waist_board_inside(q):
                    return None, {"reason": f"{arm_key.upper()}_SOURCE_OUTSIDE_BOARD"}
                rep = self._v53_mask_point_report(obs, q)
                if not bool((rep or {}).get("ok", False)):
                    return None, {"reason": f"{arm_key.upper()}_SOURCE_OUTSIDE_MASK"}
                q2, diag = self._v105_waist_reach_clamp(obs, arm_key, q)
                clamp_diag[arm_key] = diag
                if q2 is None:
                    return None, {"reason": f"{arm_key.upper()}_{diag.get('reason','REACH_CLAMP_FAILED')}", "clamp": clamp_diag}
                out[arm_key]["q"] = np.asarray(q2, np.float64)
                try:
                    px = self.d56.board_to_pixel(self.H_raw, float(q2[0]), float(q2[1]))
                    if px is not None:
                        out[arm_key]["px"] = np.asarray(px, np.float64).reshape(2)
                except Exception:
                    pass
            q2 = np.asarray(out["arm2"]["q"], np.float64).reshape(2)
            q1 = np.asarray(out["arm1"]["q"], np.float64).reshape(2)
            sep = float(np.linalg.norm(q1 - q2))
            if not math.isfinite(sep) or sep < float(self._v105_waist_min_pair_sep_mm):
                return None, {"reason": f"PAIR_TOO_CLOSE:{sep:.1f}", "clamp": clamp_diag}
            if float(q2[0]) >= float(q1[0]) - 1e-6:
                return None, {"reason": "ARM_ORDER_INVERTED", "clamp": clamp_diag}
            out["separation_mm"] = sep
            out["v105_reach_clamp"] = copy.deepcopy(clamp_diag)
            return out, {"reason": "OK", "clamp": clamp_diag, "separation_mm": sep}

        def _v107_arm2_bodyward_waist_option(self, obs, option, report, bodyward_mm):
            try:
                out = copy.deepcopy(option)
                normal_u = np.asarray(report.get("waist_normal_u"), np.float64).reshape(2)
                if not np.isfinite(normal_u).all() or float(np.linalg.norm(normal_u)) < 1e-6:
                    return None
                normal_u = normal_u / float(np.linalg.norm(normal_u))
                q2 = np.asarray(out["arm2"]["q"], np.float64).reshape(2) + normal_u * float(bodyward_mm)
                q1 = np.asarray(out["arm1"]["q"], np.float64).reshape(2)
                rep = self._v53_mask_point_report(obs, q2)
                if not bool((rep or {}).get("ok", False)):
                    return None
                if not self._v105_waist_board_inside(q2):
                    return None
                sep = float(np.linalg.norm(q1 - q2))
                if not math.isfinite(sep) or sep < float(self._v105_waist_min_pair_sep_mm):
                    return None
                if float(q2[0]) >= float(q1[0]) - 1e-6:
                    return None
                out["arm2"]["q"] = q2
                try:
                    px = self.d56.board_to_pixel(self.H_raw, float(q2[0]), float(q2[1]))
                    if px is not None:
                        out["arm2"]["px"] = np.asarray(px, np.float64).reshape(2)
                except Exception:
                    pass
                out["arm2"]["local_waist_depth_mm"] = float(out["arm2"].get("local_waist_depth_mm", 0.0)) + float(bodyward_mm)
                out["arm2"]["body_depth_mm"] = float(out["arm2"].get("body_depth_mm", 0.0)) + float(bodyward_mm)
                out["arm2"]["arm2_bodyward_rescue_mm"] = float(bodyward_mm)
                out["arm2"]["arm2_inset_mode"] = "V107_BODYWARD_REACH_RESCUE"
                out["arm2_bodyward_rescue_mm"] = float(bodyward_mm)
                out["arm2_inset_mode"] = "V107_BODYWARD_REACH_RESCUE"
                out["separation_mm"] = float(sep)
                out["local_depth_diff_mm"] = abs(
                    float(out["arm2"].get("local_waist_depth_mm", 0.0))
                    - float(out["arm1"].get("local_waist_depth_mm", 0.0))
                )
                return out
            except Exception:
                return None

        def _v105_waist_board_relaxed_solution(self, obs, options, report, motion_args, prior_failures):
            planner = getattr(self.d56, "_63_step120_motion_plan_from_d56_plan", None)
            if not callable(planner):
                return None
            for idx, option in enumerate(options[:int(self._v98_waist_top_motion_options)]):
                relaxed, relaxed_diag = self._v105_waist_relaxed_option(obs, option)
                if relaxed is None:
                    continue
                q2 = np.asarray(relaxed["arm2"]["q"], np.float64).reshape(2)
                q1 = np.asarray(relaxed["arm1"]["q"], np.float64).reshape(2)
                relaxed_cfg = copy.copy(self.cfg56)
                relaxed_cfg.center_dead_half_width = 0.0
                relaxed_cfg.private_workspace_margin_mm = 0.0
                relaxed_cfg.split_board_x = float(0.5 * (float(q2[0]) + float(q1[0])))
                relaxed_args = copy.copy(motion_args)
                relaxed_args.step60_reach_radius_max_mm = float(self._v105_waist_relaxed_reach_max_mm)
                plan = self._v93_plan_from_direct_pair(obs, relaxed, report)
                if plan is None:
                    continue
                motion, why = planner(plan, self.config, relaxed_cfg, relaxed_args)
                if motion is None:
                    continue
                metrics = dict(getattr(plan, "metrics", {}) or {})
                metrics.update({
                    "bottom_vla_v105_board_relaxed": True,
                    "bottom_vla_v105_dynamic_split_x_mm": float(relaxed_cfg.split_board_x),
                    "bottom_vla_v105_reach_max_mm": float(self._v105_waist_relaxed_reach_max_mm),
                    "bottom_vla_v105_reach_clamp": copy.deepcopy(relaxed_diag.get("clamp", {})),
                    "bottom_vla_v93_motion_build_ok": True,
                    "bottom_vla_v93_motion_reason": str(why),
                    "bottom_vla_v93_motion_failures_before_v105": copy.deepcopy(prior_failures),
                })
                plan.metrics = metrics
                out = dict(report)
                out.update({
                    "reason": "V105_BOARD_INSIDE_PAIR_EXECUTABLE",
                    "selected_option": int(idx),
                    "motion_build_ok": True,
                    "motion_reason": str(why),
                    "pair_separation_mm": float(relaxed.get("separation_mm", 0.0)),
                    "v105_board_relaxed": True,
                    "v105_dynamic_split_x_mm": float(relaxed_cfg.split_board_x),
                    "v105_reach_clamp": copy.deepcopy(relaxed_diag.get("clamp", {})),
                })
                c2 = relaxed_diag.get("clamp", {}).get("arm2", {})
                c1 = relaxed_diag.get("clamp", {}).get("arm1", {})
                print(
                    f"[D60-V105-BOARD-RELAXED] option={idx} "
                    f"A2=({q2[0]:.1f},{q2[1]:.1f}) A1=({q1[0]:.1f},{q1[1]:.1f}) "
                    f"sep={float(relaxed.get('separation_mm',0.0)):.1f}mm split={float(relaxed_cfg.split_board_x):.1f} "
                    f"reachClampA2={float(c2.get('shift_mm',0.0)):.1f}mm "
                    f"reachClampA1={float(c1.get('shift_mm',0.0)):.1f}mm motion=OK"
                )
                return plan, motion, out
            return None

        def _v93_direct_waist_solution(self, obs):
            options, report = self._v93_direct_pair_options(obs)
            if not options:
                return None, None, report
            first_plan = None
            motion_failures = []
            for idx, option in enumerate(options[:int(self._v98_waist_top_motion_options)]):
                plan = self._v93_plan_from_direct_pair(obs, option, report)
                if plan is None:
                    continue
                if first_plan is None:
                    first_plan = plan
                motion, why = self.d56._63_step120_motion_plan_from_d56_plan(
                    plan, self.config, self.cfg56, self.d56_args
                )
                if motion is not None:
                    metrics = dict(getattr(plan, "metrics", {}) or {})
                    metrics.update({
                        "bottom_vla_v93_motion_build_ok": True,
                        "bottom_vla_v93_option_index": int(idx),
                        "bottom_vla_v93_motion_reason": str(why),
                    })
                    plan.metrics = metrics
                    out = dict(report)
                    out.update({
                        "reason": "V99_DIRECT_PAIR_EXECUTABLE",
                        "selected_option": int(idx),
                        "motion_build_ok": True,
                        "motion_reason": str(why),
                        "pair_separation_mm": float(option.get("separation_mm", 0.0)),
                    })
                    q2 = np.asarray(option["arm2"]["q"], np.float64)
                    q1 = np.asarray(option["arm1"]["q"], np.float64)
                    print(
                        f"[D60-V99-DIRECT-PAIR] A2=({q2[0]:.1f},{q2[1]:.1f}) "
                        f"A1=({q1[0]:.1f},{q1[1]:.1f}) sep={float(option.get('separation_mm',0.0)):.1f}mm "
                        f"depthDiff={float(option.get('local_depth_diff_mm',0.0)):.1f}mm "
                        f"A2TargetInset={float(option.get('arm2_tangent_inset_mm',0.0)):.1f}mm "
                        f"A2ActualInset={float(option.get('arm2_actual_inset_mm',0.0)):.1f}mm "
                        f"A2InsetMode={str(option.get('arm2_inset_mode','-'))} "
                        f"A2Bodyward={float(option.get('arm2_bodyward_rescue_mm',0.0) or 0.0):.1f}mm "
                        f"option={idx} motion=OK generation={float(report.get('generation_ms',0.0)):.1f}ms"
                    )
                    return plan, motion, out
                motion_failures.append({"option": int(idx), "reason": str(why)})
            for bodyward_mm in self._v107_waist_arm2_bodyward_steps_mm:
                for idx, option in enumerate(options[:int(self._v98_waist_top_motion_options)]):
                    shifted = self._v107_arm2_bodyward_waist_option(obs, option, report, bodyward_mm)
                    if shifted is None:
                        continue
                    plan = self._v93_plan_from_direct_pair(obs, shifted, report)
                    if plan is None:
                        continue
                    motion, why = self.d56._63_step120_motion_plan_from_d56_plan(
                        plan, self.config, self.cfg56, self.d56_args
                    )
                    if motion is None:
                        continue
                    metrics = dict(getattr(plan, "metrics", {}) or {})
                    metrics.update({
                        "bottom_vla_v107_arm2_bodyward_reach_rescue": True,
                        "bottom_vla_v107_arm2_bodyward_mm": float(bodyward_mm),
                        "bottom_vla_v93_motion_build_ok": True,
                        "bottom_vla_v93_motion_reason": str(why),
                    })
                    plan.metrics = metrics
                    q2 = np.asarray(shifted["arm2"]["q"], np.float64)
                    q1 = np.asarray(shifted["arm1"]["q"], np.float64)
                    out = dict(report)
                    out.update({
                        "reason": "V107_ARM2_BODYWARD_REACH_RESCUE_EXECUTABLE",
                        "selected_option": int(idx),
                        "motion_build_ok": True,
                        "motion_reason": str(why),
                        "pair_separation_mm": float(shifted.get("separation_mm", 0.0)),
                        "arm2_actual_inset_mm": float(shifted.get("arm2_actual_inset_mm", option.get("arm2_actual_inset_mm", 0.0))),
                        "v107_arm2_bodyward_mm": float(bodyward_mm),
                    })
                    print(
                        f"[D60-V107-ARM2-BODYWARD-REACH] option={idx} bodyward=+{float(bodyward_mm):.0f}mm "
                        f"A2=({q2[0]:.1f},{q2[1]:.1f}) A1=({q1[0]:.1f},{q1[1]:.1f}) "
                        f"A2ActualInset={float(shifted.get('arm2_actual_inset_mm',option.get('arm2_actual_inset_mm',0.0))):.1f}mm motion=OK"
                    )
                    return plan, motion, out
            waist_center = np.asarray(report.get("waist_center_board", [0.0, -9999.0]), np.float64).reshape(-1)
            is_top_waist = bool(waist_center.size >= 2 and float(waist_center[1]) > float(self._v98_waist_top_y_threshold_mm))
            if is_top_waist:
                rise = float(np.clip(float(waist_center[1]) - float(self._v98_waist_top_y_threshold_mm), 0.0, 180.0))
                top_args = copy.copy(self.d56_args)
                base_edge = float(getattr(self.d56_args, "step60_edge_inset_mm", 45.0))
                base_outside = float(getattr(self.d56_args, "step60_target_outside_board_mm", 40.0))
                base_pullup = float(getattr(self.d56_args, "step60_pullup_target_median_y_mm", -190.0))
                top_args.step60_edge_inset_mm = float(np.clip(base_edge - 0.20 * rise, 25.0, 90.0))
                top_args.step60_target_outside_board_mm = float(np.clip(base_outside + 0.20 * rise, 0.0, 60.0))
                top_args.step60_pullup_target_median_y_mm = float(np.clip(base_pullup + 0.25 * rise, -190.0, -140.0))
                top_failures = []
                print(
                    f"[D60-V101-TOP-RETRY] waistY={float(waist_center[1]):.1f}mm rise={rise:.1f}mm "
                    f"edgeInset={top_args.step60_edge_inset_mm:.1f} outside={top_args.step60_target_outside_board_mm:.1f} "
                    f"pullupY={top_args.step60_pullup_target_median_y_mm:.1f}"
                )
                for idx, option in enumerate(options[:int(self._v98_waist_top_motion_options)]):
                    plan = self._v93_plan_from_direct_pair(obs, option, report)
                    if plan is None:
                        continue
                    motion, why = self.d56._63_step120_motion_plan_from_d56_plan(
                        plan, self.config, self.cfg56, top_args
                    )
                    if motion is not None:
                        metrics = dict(getattr(plan, "metrics", {}) or {})
                        metrics.update({
                            "bottom_vla_v100_top_waist_dynamic": True,
                            "bottom_vla_v100_top_edge_inset_mm": float(top_args.step60_edge_inset_mm),
                            "bottom_vla_v100_top_outside_mm": float(top_args.step60_target_outside_board_mm),
                            "bottom_vla_v100_top_pullup_y_mm": float(top_args.step60_pullup_target_median_y_mm),
                            "bottom_vla_v93_motion_build_ok": True,
                            "bottom_vla_v93_option_index": int(idx),
                            "bottom_vla_v93_motion_reason": str(why),
                        })
                        plan.metrics = metrics
                        out = dict(report)
                        out.update({
                            "reason": "V100_TOP_WAIST_DYNAMIC_PULLDOWN_EXECUTABLE",
                            "selected_option": int(idx),
                            "motion_build_ok": True,
                            "motion_reason": str(why),
                            "pair_separation_mm": float(option.get("separation_mm", 0.0)),
                            "v100_top_waist_dynamic": True,
                            "v100_top_edge_inset_mm": float(top_args.step60_edge_inset_mm),
                            "v100_top_outside_mm": float(top_args.step60_target_outside_board_mm),
                            "v100_top_pullup_y_mm": float(top_args.step60_pullup_target_median_y_mm),
                        })
                        q2 = np.asarray(option["arm2"]["q"], np.float64)
                        q1 = np.asarray(option["arm1"]["q"], np.float64)
                        print(
                            f"[D60-V101-TOP-RECOVERED] A2=({q2[0]:.1f},{q2[1]:.1f}) "
                            f"A1=({q1[0]:.1f},{q1[1]:.1f}) option={idx} motion=OK reason={why}"
                        )
                        return plan, motion, out
                    top_failures.append({"option": int(idx), "reason": str(why)})
                motion_failures.extend({"top_dynamic": True, **row} for row in top_failures)
                print(f"[D60-V101-TOP-BLOCK] dynamic deeper pull-down exhausted failures={top_failures[-4:]}")
            if self._v102_waist_failure_is_relocatable(motion_failures):
                relocation_args = top_args if is_top_waist else self.d56_args
                for tangent_shift_mm, normal_shift_mm in self._v103_waist_relocation_offsets():
                    for idx, option in enumerate(options[:int(self._v98_waist_top_motion_options)]):
                        shifted = self._v103_relocated_waist_option(
                            obs, option, report, tangent_shift_mm, normal_shift_mm
                        )
                        if shifted is None:
                            continue
                        plan = self._v93_plan_from_direct_pair(obs, shifted, report)
                        if plan is None:
                            continue
                        motion, why = self.d56._63_step120_motion_plan_from_d56_plan(
                            plan, self.config, self.cfg56, relocation_args
                        )
                        if motion is None:
                            continue
                        metrics = dict(getattr(plan, "metrics", {}) or {})
                        metrics.update({
                            "bottom_vla_v103_rigid_2d_relocation": True,
                            "bottom_vla_v103_tangent_shift_mm": float(tangent_shift_mm),
                            "bottom_vla_v103_normal_shift_mm": float(normal_shift_mm),
                            "bottom_vla_v93_motion_build_ok": True,
                            "bottom_vla_v93_motion_reason": str(why),
                        })
                        plan.metrics = metrics
                        q2 = np.asarray(shifted["arm2"]["q"], np.float64)
                        q1 = np.asarray(shifted["arm1"]["q"], np.float64)
                        out = dict(report)
                        out.update({
                            "reason": "V103_RIGID_2D_RELOCATION_EXECUTABLE",
                            "selected_option": int(idx),
                            "motion_build_ok": True,
                            "motion_reason": str(why),
                            "pair_separation_mm": float(shifted.get("separation_mm", 0.0)),
                            "v103_tangent_shift_mm": float(tangent_shift_mm),
                            "v103_normal_shift_mm": float(normal_shift_mm),
                            "arm2_actual_inset_mm": float(shifted.get("arm2_actual_inset_mm", 0.0)),
                        })
                        print(
                            f"[D60-V103-RIGID-2D-RELOCATE] tangent={float(tangent_shift_mm):+.1f}mm "
                            f"normal={float(normal_shift_mm):+.1f}mm option={idx} "
                            f"A2=({q2[0]:.1f},{q2[1]:.1f}) A1=({q1[0]:.1f},{q1[1]:.1f}) "
                            f"A2ActualInset={float(shifted.get('arm2_actual_inset_mm',0.0)):.1f}mm motion=OK reason={why}"
                        )
                        return plan, motion, out
                print("[D60-V103-RIGID-2D-RELOCATE-BLOCK] soft workspace/reach relocation exhausted in tangent/body-normal search; keep original grasp display")
            v105_motion_args = top_args if is_top_waist else self.d56_args
            v105_relaxed = self._v105_waist_board_relaxed_solution(
                obs, options, report, v105_motion_args, motion_failures
            )
            if v105_relaxed is not None:
                return v105_relaxed
            if first_plan is None:
                out = dict(report)
                out.update({"reason": "V99_DIRECT_PLAN_BUILD_FAILED", "motion_build_ok": False})
                return None, None, out
            metrics = dict(getattr(first_plan, "metrics", {}) or {})
            metrics.update({
                "bottom_vla_v93_motion_build_ok": False,
                "bottom_vla_v93_motion_failures": copy.deepcopy(motion_failures),
            })
            first_plan.metrics = metrics
            option = options[0]
            q2 = np.asarray(option["arm2"]["q"], np.float64)
            q1 = np.asarray(option["arm1"]["q"], np.float64)
            out = dict(report)
            out.update({
                "reason": "V99_GRASP_POINTS_GENERATED_EXECUTION_BLOCKED",
                "selected_option": 0,
                "motion_build_ok": False,
                "motion_failures": motion_failures,
                "pair_separation_mm": float(option.get("separation_mm", 0.0)),
            })
            if is_top_waist:
                print(
                    f"[D60-V98-TOP-WAIST] waistY={float(waist_center[1]):.1f}mm > {float(self._v98_waist_top_y_threshold_mm):.1f}mm; "
                    f"all {min(len(options),int(self._v98_waist_top_motion_options))} direct motion options were tried before block"
                )
            print(
                f"[D60-V99-GRASP-GUARANTEE] A2=({q2[0]:.1f},{q2[1]:.1f}) "
                f"A1=({q1[0]:.1f},{q1[1]:.1f}) sep={float(option.get('separation_mm',0.0)):.1f}mm; "
                "grasp points KEPT, physical execution blocked by STEP63 motion build"
            )
            return first_plan, None, out

        def _prepare_waist_pull_laydown(self):
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
            bundle = None
            obs = None
            d60_plan = None
            motion60 = None
            solution_report = {"ok": False, "reason": "V94_NOT_EVALUATED"}
            perception_reports = []
            for attempt in range(1, int(self._v103_waist_perception_attempts) + 1):
                candidate_bundle = self._capture_i_frame_from_live("D56_WAIST_LIFT_LAYDOWN")
                candidate_obs = self._infer_for_action("D56_WAIST_LIFT_LAYDOWN", candidate_bundle.corrected)
                ready, _, ready_report = self._v93_pose8_mask_ready(candidate_obs)
                if (not ready and bool(ready_report.get("mask_ok", False)) and int(ready_report.get("pose8_count", 0)) < 8):
                    try:
                        rescue_obs = self._infer_for_action("D58_CIRC_POSITION", candidate_bundle.corrected)
                        rescue_pts = self._pose_board_points(rescue_obs)
                        required = (
                            "waist_img_left", "waist_center", "waist_img_right", "crotch",
                            "img_left_hem_outer", "img_left_hem_inner", "img_right_hem_inner", "img_right_hem_outer",
                        )
                        rescue_count = sum(1 for name in required if self._finite_point2(rescue_pts.get(name)) is not None)
                        if rescue_count == 8 and getattr(rescue_obs, "pose", None) is not None:
                            rescued = copy.copy(candidate_obs)
                            setattr(rescued, "pose", getattr(rescue_obs, "pose"))
                            candidate_obs = rescued
                            ready, _, ready_report = self._v93_pose8_mask_ready(candidate_obs)
                            ready_report = dict(ready_report)
                            ready_report["v101_pose8_rescued_from_d58"] = True
                            print(f"[D60-V101-POSE8-RESCUE] D56 Pose8 was incomplete but YOLO Mask was valid; D58 pose-only rescue={rescue_count}/8 ready={ready}")
                        else:
                            print(f"[D60-V101-POSE8-RESCUE] D58 rescue incomplete pose8={rescue_count}/8; keep normal retry")
                    except Exception as exc:
                        print(f"[D60-V101-POSE8-RESCUE-WARN] {type(exc).__name__}: {exc}")
                if ready:
                    semantic_ok, semantic_report = self._v103_waist_pose_semantic_report(candidate_obs)
                    ready_report = dict(ready_report)
                    ready_report["v103_pose_semantic"] = copy.deepcopy(semantic_report)
                    if not semantic_ok:
                        print(
                            f"[D60-V104-POSE-SANITY-SOFT] attempt={attempt}/{int(self._v103_waist_perception_attempts)} "
                            f"diagnostic={semantic_report.get('reason','V103_POSE_SEMANTIC_INVALID')} accepted=True "
                            f"waistCrotch={float(semantic_report.get('waist_to_crotch_mm',0.0)):.1f}mm "
                            f"hemCrotch={float(semantic_report.get('hem_to_crotch_mm',0.0)):.1f}mm t={semantic_report.get('crotch_axis_t')}"
                        )
                perception_reports.append({"attempt": int(attempt), **copy.deepcopy(ready_report)})
                bundle = candidate_bundle
                obs = candidate_obs
                if not ready:
                    solution_report = dict(ready_report)
                    print(
                        f"[D60-V103-PERCEPTION] attempt={attempt}/{int(self._v103_waist_perception_attempts)} ready=False "
                        f"mask={bool(ready_report.get('mask_ok',False))} pose8={int(ready_report.get('pose8_count',0))}/8 "
                        f"reason={ready_report.get('reason','-')}"
                    )
                    continue
                d60_plan, motion60, solution_report = self._v93_direct_waist_solution(candidate_obs)
                break
            if bundle is None or obs is None:
                self.status = "WAIST_PULL_LAYDOWN NO PLAN: SNAPSHOT_UNAVAILABLE"
                return
            pair_generated = bool(d60_plan is not None and bool(getattr(d60_plan, "ok", False)))
            execution_ready = bool(pair_generated and motion60 is not None)
            if isinstance(motion60, dict):
                motion60["bottom_vla_pullup_profile"] = {
                    "enabled": True,
                    "pre_pullup_vertical_lift_mm": 50.0,
                    "high_z_horizontal_first_mm": 0.0,
                }
            if pair_generated:
                display_plan = d60_plan
                if execution_ready:
                    reason = "V99_DIRECT_PAIR_EXECUTABLE"
                else:
                    reason = str(solution_report.get("reason", "V99_GRASP_POINTS_GENERATED_EXECUTION_BLOCKED"))
            else:
                display_plan = None
                last_ready = perception_reports[-1] if perception_reports else {}
                reason = str(solution_report.get("reason") or last_ready.get("reason") or "V95_MASK_OR_POSE8_UNAVAILABLE")
            canvas = self._draw_d60_frozen_overlay(bundle, obs, display_plan, motion60)
            if execution_ready:
                title = "WAIST_PULL_LAYDOWN V99 REAL-WAIST BALANCED GRIPS - AUTO EXECUTE"
                color = (0, 255, 255)
            elif pair_generated:
                title = "WAIST V99 GRIPS GENERATED - EXECUTION BLOCKED"
                color = (0, 165, 255)
            else:
                title = f"WAIST_PULL_LAYDOWN NO PLAN: {reason}"
                color = (0, 0, 255)
            self.cv2.putText(canvas, title[:110], (20, 45), self.cv2.FONT_HERSHEY_SIMPLEX, 0.68, color, 2, self.cv2.LINE_AA)
            arm_points = v23._json_safe(getattr(d60_plan, "arm_points", {})) if pair_generated else {}
            proxy = SimpleNamespace(
                ok=bool(execution_ready),
                reason=str(reason),
                action="WAIST_PULL_LAYDOWN",
                metrics={
                    "source": "60-15",
                    "motion60": v23._json_safe(motion60),
                    "bottom_vla_v93_pair_generated": bool(pair_generated),
                    "bottom_vla_v93_execution_ready": bool(execution_ready),
                    "bottom_vla_v93_solution": v23._json_safe(solution_report),
                },
                arm_points=arm_points,
            )
            token = self._store_d60_runtime({"plan": d60_plan, "motion": motion60}) if execution_ready else ""
            diagnostics = {
                "d60_runtime_token": token,
                "source_motion": "60-15",
                "d60_direct_executor": True,
                "d60_pre_freeze_attempts": int(len(perception_reports)),
                "d60_selected_attempt": int(next((r["attempt"] for r in perception_reports if r.get("ok")), 0)),
                "d60_waist_endpoint_gate_required": False,
                "bottom_vla_v93_direct_pair": True,
                "bottom_vla_v93_pair_generated": bool(pair_generated),
                "bottom_vla_v93_execution_ready": bool(execution_ready),
                "bottom_vla_v93_perception_attempts": v23._json_safe(perception_reports),
                "bottom_vla_v93_solution": v23._json_safe(solution_report),
                **self._semantic_metadata(),
            }
            locked = base.LockedPlan(
                None, "WAIST_PULL_LAYDOWN", bundle, obs, proxy, None, canvas,
                self.H_raw.copy(), time.time(), bool(execution_ready), str(reason),
                None if execution_ready else str(reason), diagnostics
            )
            obs_record = self.recorder.save_observation(locked, "BEFORE_ACTION", self._environment_metadata())
            locked.observation_record = obs_record
            self.recorder.save_decision(locked, obs_record)
            with self.state_lock:
                self.locked = locked
                self.display_image = canvas.copy()
                if execution_ready:
                    self.status = "WAIST_PULL_LAYDOWN V98 BALANCED: GRIPS READY - AUTO EXECUTE"
                elif pair_generated:
                    self.status = "WAIST V98 GRIPS GENERATED: EXECUTION BLOCKED BY MOTION GATE"
                else:
                    self.status = f"WAIST_PULL_LAYDOWN NO PLAN: {reason}"
            self._decorate_current_records()
            print(
                f"[D60-V95-RESULT] pair_generated={pair_generated} execution_ready={execution_ready} "
                f"perception_attempts={len(perception_reports)} reason={reason}"
            )

        def set_integration_empty_board_state(self, state):
            if not self._integration_mode:
                raise RuntimeError("set_integration_empty_board_state is integration-only")
            raw = np.asarray(getattr(state, "raw_bgr", None), dtype=np.uint8)
            corrected = np.asarray(getattr(state, "corrected_bgr", None), dtype=np.uint8)
            if raw.ndim != 3 or corrected.ndim != 3 or raw.size == 0 or corrected.size == 0:
                raise RuntimeError("EmptyBoardState raw/corrected frame is invalid")
            if raw.shape != corrected.shape:
                raise RuntimeError(f"EmptyBoardState raw/corrected shape mismatch: {raw.shape} vs {corrected.shape}")
            expected = (int(self.args.height), int(self.args.width))
            if tuple(raw.shape[:2]) != expected:
                raise RuntimeError(f"EmptyBoardState size {raw.shape[:2]} != lower runtime {expected}")
            state_h = getattr(state, "homography", None)
            if state_h is not None:
                A = np.asarray(state_h, dtype=np.float64)
                B = np.asarray(self.H_corrected, dtype=np.float64)
                if A.shape != (3, 3) or B.shape != (3, 3):
                    raise RuntimeError("EmptyBoardState/runtime Homography shape invalid")

                def _norm_h(H):
                    scale = float(H[2, 2]) if abs(float(H[2, 2])) > 1e-12 else float(np.linalg.norm(H))
                    return H / max(abs(scale), 1e-12)

                # Diagnostic-only instrumentation.  The acceptance rule below remains
                # exactly the historical np.allclose(rtol=1e-5, atol=1e-5) check.
                An = _norm_h(A)
                Bn = _norm_h(B)
                h_match = bool(np.allclose(An, Bn, rtol=1e-5, atol=1e-5))
                baseline_id = str(getattr(state, "baseline_id", "") or "UNKNOWN")
                baseline_source = str(getattr(state, "source", "") or "UNKNOWN")
                state_geom_diag = dict(getattr(state, "camera_geometry", {}) or {})
                lower_geom_diag = dict(getattr(self, "corrected_camera_geometry", {}) or {})
                state_cal_diag = str(state_geom_diag.get("calibration_id", "") or "UNKNOWN")
                lower_cal_diag = str(lower_geom_diag.get("calibration_id", "") or "UNKNOWN")

                max_norm_elem = float(np.max(np.abs(An - Bn)))
                fro_norm = float(np.linalg.norm(An - Bn))
                corner_errors = None
                corner_max_px = float("nan")
                corner_mean_px = float("nan")
                try:
                    marker_map = self.config.get("aruco", {}).get("marker_board_mm", {})
                    marker = np.asarray(list(marker_map.values()), dtype=np.float64).reshape(-1, 2)
                    if marker.shape[0] >= 4 and np.all(np.isfinite(marker)):
                        x0, x1 = float(np.min(marker[:, 0])), float(np.max(marker[:, 0]))
                        y0, y1 = float(np.min(marker[:, 1])), float(np.max(marker[:, 1]))
                        board_corners = np.asarray(
                            [[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=np.float64
                        )

                        def _board_to_px(H, points):
                            Hinv = np.linalg.inv(H)
                            hom = np.column_stack(
                                (points[:, 0], points[:, 1], np.ones(len(points), dtype=np.float64))
                            )
                            out = hom @ Hinv.T
                            if (not np.all(np.isfinite(out))) or np.any(np.abs(out[:, 2]) <= 1e-12):
                                raise ValueError("non-finite board-corner projection")
                            return out[:, :2] / out[:, 2:3]

                        px_a = _board_to_px(A, board_corners)
                        px_b = _board_to_px(B, board_corners)
                        corner_errors = np.linalg.norm(px_a - px_b, axis=1)
                        corner_max_px = float(np.max(corner_errors))
                        corner_mean_px = float(np.mean(corner_errors))
                except Exception as exc:
                    print(f"[BOTTOM-INTEGRATION-H-DIAG-WARN] corner projection unavailable: {exc!r}")

                print(
                    f"[BOTTOM-INTEGRATION-H-DIAG] baseline={baseline_id} source={baseline_source} "
                    f"stateCal={state_cal_diag} lowerCal={lower_cal_diag} "
                    f"strictMatch={h_match} maxNormElemDiff={max_norm_elem:.9g} "
                    f"froNormDiff={fro_norm:.9g} cornerMaxPx={corner_max_px:.3f} "
                    f"cornerMeanPx={corner_mean_px:.3f}"
                )
                if corner_errors is not None:
                    print(
                        "[BOTTOM-INTEGRATION-H-DIAG-CORNERS] "
                        + "errorsPx=["
                        + ", ".join(f"{float(v):.3f}" for v in corner_errors)
                        + "]"
                    )

                # Also compare against the H currently stored on disk.  This is
                # diagnostic-only and never participates in pass/fail decisions; it
                # lets us distinguish a stale main/state H from a changed lower H.
                try:
                    hfile = Path(str(getattr(self.args, "hfile", ""))).expanduser()
                    payload = json.loads(hfile.read_text(encoding="utf-8"))
                    disk_h = np.asarray(payload.get("H"), dtype=np.float64).reshape(3, 3)
                    disk_n = _norm_h(disk_h)
                    state_disk = float(np.max(np.abs(An - disk_n)))
                    lower_disk = float(np.max(np.abs(Bn - disk_n)))
                    disk_sha = hashlib.sha256(hfile.read_bytes()).hexdigest()[:16]
                    disk_mtime_ns = int(hfile.stat().st_mtime_ns)
                    disk_cal = str((payload.get("camera_geometry", {}) or {}).get("calibration_id", "") or "UNKNOWN")
                    print(
                        f"[BOTTOM-INTEGRATION-H-DIAG-DISK] hfile={hfile} sha256_16={disk_sha} "
                        f"mtimeNs={disk_mtime_ns} diskCal={disk_cal} "
                        f"stateVsDiskMaxNormElem={state_disk:.9g} "
                        f"lowerVsDiskMaxNormElem={lower_disk:.9g}"
                    )
                except Exception as exc:
                    print(f"[BOTTOM-INTEGRATION-H-DIAG-DISK-WARN] {type(exc).__name__}: {exc}")

                if not h_match:
                    print(
                        "[BOTTOM-INTEGRATION-H-DIAG-FAIL] "
                        f"stateH={np.array2string(An, precision=9, suppress_small=False, separator=',')} "
                        f"lowerH={np.array2string(Bn, precision=9, suppress_small=False, separator=',')}"
                    )
                    raise RuntimeError("EmptyBoardState Homography differs from lower corrected Homography")
            state_geom = dict(getattr(state, "camera_geometry", {}) or {})
            lower_cal = str((self.corrected_camera_geometry or {}).get("calibration_id", ""))
            state_cal = str(state_geom.get("calibration_id", ""))
            if lower_cal and state_cal and lower_cal != state_cal:
                raise RuntimeError(f"EmptyBoardState calibration mismatch: state={state_cal} lower={lower_cal}")
            jobs = (
                (self.d54, self.d54_args, corrected, self.H_corrected, "D54-CORRECTED"),
                (self.d55, self.d55_args, corrected, self.H_corrected, "D55-CORRECTED"),
                (self.d56, self.d56_args, raw, self.H_raw, "D56-RAW"),
            )
            completed = []
            for module, namespace, frame, action_h, label in jobs:
                setter = getattr(module, "_d34_set_empty_board", None)
                sync = getattr(module, "_d38_sync_e49_runtime", None)
                if setter is None or sync is None:
                    raise RuntimeError(f"{label} empty-board dependency unavailable")
                if not setter(frame.copy(), namespace.d34_empty_board_path, save=False):
                    raise RuntimeError(f"{label} EmptyBoardState install failed")
                sync(namespace, action_h)
                if hasattr(namespace, "_d56v20_fallback_last_trusted_area_px"):
                    namespace._d56v20_fallback_last_trusted_area_px = 0.0
                completed.append(label)
            self.empty_baseline_ready = True
            self.locked = None
            self.display_image = None
            self._integration_empty_board_baseline_id = str(getattr(state, "baseline_id", ""))
            with self.state_lock:
                self.status = "INTEGRATION EMPTY BOARD READY - WAITING FOR BOTTOM CYCLE"
            print(
                f"[BOTTOM-INTEGRATION-EMPTY] baseline={self._integration_empty_board_baseline_id or 'UNKNOWN'} "
                f"installed={' + '.join(completed)}"
            )
            return True

        def _integration_stop_requested(self):
            return bool(self._integration_mode and self._integration_stop_event.is_set())

        def _integration_live_frame_pair(self, raw):
            """Return safe RAW/CORRECTED copies for the integrated live cache.

            The inherited frame router may transiently return None while the
            selected action is being changed by AUTO-JUDGE.  The V22 display
            bridge must never turn that transient state into a fatal .copy()
            exception.  RAW remains the authoritative camera geometry and is a
            safe cache fallback until the next valid corrected frame arrives.
            """
            if not isinstance(raw, np.ndarray) or raw.ndim != 3 or raw.size == 0:
                raise ValueError("integrated RAW frame is invalid")
            try:
                with self.state_lock:
                    action_snapshot = self.selected_action
            except Exception:
                action_snapshot = getattr(self, "selected_action", None)

            corrected = None
            failure = None
            try:
                corrected = self._frame_for_action(raw, action_snapshot)
            except Exception as exc:
                failure = f"{type(exc).__name__}:{exc}"

            valid = bool(
                isinstance(corrected, np.ndarray)
                and corrected.ndim == 3
                and corrected.size > 0
                and corrected.shape[:2] == raw.shape[:2]
            )
            if not valid:
                if failure is None:
                    if corrected is None:
                        failure = "frame_for_action_returned_none"
                    elif not isinstance(corrected, np.ndarray):
                        failure = f"frame_for_action_type={type(corrected).__name__}"
                    else:
                        failure = f"frame_for_action_shape={getattr(corrected, 'shape', None)}"
                corrected = raw
                signature = (str(action_snapshot), str(failure))
                now = time.monotonic()
                if (
                    signature != self._integration_frame_fallback_last_signature
                    or now - float(self._integration_frame_fallback_last_time) >= 1.0
                ):
                    print(
                        f"[BOTTOM-INTEGRATION-FRAME-FALLBACK] action={action_snapshot or 'NONE'} "
                        f"reason={failure}; keeping cycle alive with RAW cache until next valid frame"
                    )
                    self._integration_frame_fallback_last_signature = signature
                    self._integration_frame_fallback_last_time = now

            return raw.copy(), corrected.copy()

        def wait_for_integrated_idle(self, timeout=15.0):
            """Wait until action/planner workers are quiescent before serial close."""
            deadline = time.monotonic() + max(0.1, float(timeout))
            last = None
            while time.monotonic() < deadline:
                with self.state_lock:
                    state = {
                        "motion": bool(getattr(self, "motion_busy", False)),
                        "inference": bool(getattr(self, "inference_busy", False)),
                        "auto_judge": bool(getattr(self, "_auto_judge_busy", False)),
                        "initial_spread": bool(getattr(self, "_v78_initial_spread_busy", False)),
                        "finish": bool(getattr(self, "_v74_finish_processing", False)),
                        "finish_worker": bool(getattr(self, "_v95_finish_commit_worker_active", False)),
                    }
                worker = getattr(self, "worker", None)
                state["worker"] = bool(
                    isinstance(worker, threading.Thread) and worker.is_alive()
                )
                if not any(state.values()):
                    print("[BOTTOM-INTEGRATION-IDLE] action/planner workers quiescent")
                    return True
                last = state
                time.sleep(0.02)
            print(f"[BOTTOM-INTEGRATION-IDLE-TIMEOUT] active={last}")
            return False

        def request_integrated_stop(self):
            self._integration_stop_event.set()

        def run_integrated_cycle(self, timeout=240.0):
            if not self._integration_mode:
                raise RuntimeError("run_integrated_cycle is integration-only")
            if not self.empty_baseline_ready:
                raise RuntimeError("shared EmptyBoardState must be installed before run_integrated_cycle")
            self._v116_finish_terminal_latched = False
            provider = self._integration_external_frame_provider
            if provider is None or not bool(provider.is_alive()):
                raise RuntimeError("V22 shared camera provider is not alive")
            self._integration_stop_event.clear()
            self._v74_finish_complete_banner = False
            self._v74_finish_processing = False
            self._v76_finish_commit_queued = False
            self._v87_finish_requested = False
            self._v87_finish_cache = None
            self._v78_initial_spread_busy = False
            self._v78_initial_spread_done = False
            self._v51_post_basket_position_pending = False
            self.semantic_selected = None
            self.pending_semantic = None
            self.selected_action = None
            self.auto_recommended = "REJUDGE"
            self._human_selected = None
            self._prepare_origin = "AUTO"
            self._invalidate_for_new_action("INTEGRATED_BOARD_READY")
            if hasattr(self.cap, "timeout_s"):
                self.cap.timeout_s = 0.20
            deadline = time.monotonic() + max(5.0, float(timeout))
            first_frame = False
            judge_started = False
            last_status = None
            handled_no_plan_tokens = set()
            repeated_no_plan_key = None
            repeated_no_plan_count = 0
            max_same_no_plan_retries = 4
            auto_rejudge_count = 0
            auto_rejudge_due = None
            auto_rejudge_was_required = False
            auto_rejudge_last_reason = None
            while time.monotonic() < deadline:
                if self._integration_stop_event.is_set():
                    try:
                        provider.clear_override()
                    except Exception:
                        pass
                    return {"status": "STOPPED", "reason": "integration stop requested", "recovery_required": True}
                if not bool(provider.is_alive()):
                    return {"status": "ERROR", "reason": "V22 shared camera provider stopped", "recovery_required": True}
                ok, raw = self._read_raw(flush=0)
                if ok and isinstance(raw, np.ndarray) and raw.ndim == 3 and raw.size > 0:
                    first_frame = True
                    raw_cache, corrected_cache = self._integration_live_frame_pair(raw)
                    with self.live_frame_lock:
                        self.latest_live_raw = raw_cache
                        self.latest_live_corrected = corrected_cache
                        self.latest_live_monotonic = time.monotonic()
                with self.live_frame_lock:
                    live_raw_now = (
                        self.latest_live_raw.copy()
                        if isinstance(getattr(self, "latest_live_raw", None), np.ndarray)
                        and self.latest_live_raw.ndim == 3
                        and self.latest_live_raw.size > 0
                        else None
                    )
                if first_frame and not judge_started:
                    judge_started = True
                    with self.state_lock:
                        self.status = "BOTTOM INTEGRATION: BOARD READY - AUTO JUDGE"
                    print("[BOTTOM-INTEGRATION-START] basket step skipped; garment is already on board -> fresh AUTO-JUDGE")
                    self._schedule_auto_judge("integrated_board_ready", 0.05)
                for event in self._drain_events():
                    if event == "ACTION:BASKET_GRASP":
                        print("[BOTTOM-INTEGRATION-BASKET-BLOCK] BASKET_GRASP ignored; garment_transfer already placed garment")
                        continue
                    if event == "EMPTY_BASELINE":
                        print("[BOTTOM-INTEGRATION-E-BLOCK] live E recapture ignored; shared EmptyBoardState is authoritative")
                        continue
                    self._handle(event)
                with self.state_lock:
                    status = str(self.status)
                    finish_done = bool(self._v74_finish_complete_banner) and not bool(self._v74_finish_processing)
                    locked_now = self.locked
                    motion_busy_now = bool(self.motion_busy)
                    inference_busy_now = bool(self.inference_busy)
                    auto_busy_now = bool(self._auto_judge_busy)
                    display_now = self.display_image.copy() if isinstance(self.display_image, np.ndarray) else None
                    pending_now = str(self.pending_semantic or self.semantic_selected or self.selected_action or "")
                    auto_recommended_now = str(self.auto_recommended or "").upper()
                shown = None
                if first_frame and isinstance(live_raw_now, np.ndarray):
                    try:
                        shown = display_now if display_now is not None else live_raw_now.copy()
                        if not isinstance(shown, np.ndarray) or shown.ndim != 3 or shown.size == 0:
                            shown = live_raw_now.copy()
                        if motion_busy_now:
                            self.cv2.putText(shown, "ROBOT MOTION RUNNING", (20, 40), self.cv2.FONT_HERSHEY_SIMPLEX, 0.78, (0, 0, 255), 3)
                        elif inference_busy_now or auto_busy_now:
                            self.cv2.putText(shown, "AUTO / PLAN PREPARING", (20, 40), self.cv2.FONT_HERSHEY_SIMPLEX, 0.72, (0, 200, 255), 2)
                        self.cv2.putText(shown, status[:130], (20, shown.shape[0] - 24), self.cv2.FONT_HERSHEY_SIMPLEX, 0.53, (0, 255, 255), 2)
                        provider.show_override(shown, status)
                    except Exception as exc:
                        print(f"[BOTTOM-INTEGRATION-DISPLAY-WARN] {type(exc).__name__}: {exc}; UI bridge isolated")
                        try:
                            if isinstance(live_raw_now, np.ndarray) and live_raw_now.size > 0:
                                provider.show_override(live_raw_now, status)
                        except Exception:
                            pass

                rejudge_required_now = bool(
                    auto_recommended_now == "REJUDGE"
                    and status.upper().startswith("AUTO JUDGE: REJUDGE REQUIRED")
                    and locked_now is None
                    and not motion_busy_now
                    and not inference_busy_now
                    and not auto_busy_now
                )
                if rejudge_required_now:
                    now_mono = time.monotonic()
                    reason_now = status.split(" - ", 1)[1] if " - " in status else status
                    if (not auto_rejudge_was_required) or reason_now != auto_rejudge_last_reason:
                        if reason_now != auto_rejudge_last_reason:
                            auto_rejudge_count = 0
                        auto_rejudge_count += 1
                        auto_rejudge_last_reason = reason_now
                        if auto_rejudge_count <= 4:
                            retry_delay = 0.35
                        elif auto_rejudge_count <= 8:
                            retry_delay = 0.75
                        else:
                            retry_delay = 1.25
                        auto_rejudge_due = now_mono + retry_delay
                        print(
                            f"[BOTTOM-INTEGRATION-AUTO-REJUDGE-WAIT] attempt={auto_rejudge_count} "
                            f"delay={retry_delay:.2f}s reason={reason_now}"
                        )
                    auto_rejudge_was_required = True
                    if auto_rejudge_due is not None and now_mono >= float(auto_rejudge_due):
                        print(
                            f"[BOTTOM-INTEGRATION-AUTO-REJUDGE] attempt={auto_rejudge_count} "
                            f"reason={reason_now}"
                        )
                        auto_rejudge_due = None
                        self._rejudge()
                        auto_rejudge_was_required = False
                else:
                    auto_rejudge_was_required = False
                    auto_rejudge_due = None
                    if auto_rejudge_count > 0 and (
                        auto_recommended_now != "REJUDGE"
                        or locked_now is not None
                        or pending_now not in {"", "REJUDGE"}
                    ):
                        print(f"[BOTTOM-INTEGRATION-AUTO-REJUDGE-RECOVERED] attempts={auto_rejudge_count} status={status}")
                        auto_rejudge_count = 0
                        auto_rejudge_last_reason = None

                if locked_now is not None and not bool(getattr(locked_now, "plan_ok", False)) and not motion_busy_now and not inference_busy_now and not auto_busy_now:
                    generation_now = int(getattr(self, "_prepare_generation", 0))
                    token = (id(locked_now), generation_now)
                    if token not in handled_no_plan_tokens:
                        handled_no_plan_tokens.add(token)
                        action_now = str(getattr(locked_now, "action", "") or pending_now or "UNKNOWN").upper()
                        reason_now = str(getattr(locked_now, "reason", "") or getattr(locked_now, "planner_failure", "") or status or "NO_SAFE_PLAN")
                        reason_key = reason_now.split(" failures=", 1)[0][:180]
                        key = (action_now, reason_key)
                        if key == repeated_no_plan_key:
                            repeated_no_plan_count += 1
                        else:
                            repeated_no_plan_key = key
                            repeated_no_plan_count = 1
                        if repeated_no_plan_count > max_same_no_plan_retries:
                            reason = f"repeated NO_SAFE_PLAN exhausted action={action_now} count={repeated_no_plan_count} reason={reason_now}"
                            print(f"[BOTTOM-INTEGRATION-NOPLAN-EXHAUSTED] {reason}")
                            try:
                                if isinstance(shown, np.ndarray) and shown.size > 0:
                                    provider.show_override(shown, reason)
                            except Exception:
                                pass
                            return {"status": "ERROR", "reason": reason, "recovery_required": False}
                        print(
                            f"[BOTTOM-INTEGRATION-NOPLAN-REJUDGE] action={action_now} "
                            f"attempt={repeated_no_plan_count}/{max_same_no_plan_retries} reason={reason_now}"
                        )
                        self._queue("REJUDGE")
                if status != last_status:
                    last_status = status
                    try:
                        provider.set_status(status)
                    except Exception:
                        pass
                if finish_done:
                    print("[BOTTOM-INTEGRATION-SUCCESS] FINISH accepted; returning SUCCESS to V22 integration runtime")
                    try:
                        provider.clear_override()
                    except Exception:
                        pass
                    return {"status": "SUCCESS", "reason": status or "FINISH", "recovery_required": False}
                upper = status.upper()
                if "AUTO JUDGE ERROR" in upper or "FINISH ERROR" in upper or "STARTUP STANDBY FAILED" in upper:
                    return {"status": "ERROR", "reason": status, "recovery_required": True}
                time.sleep(0.01)
            try:
                provider.clear_override()
            except Exception:
                pass
            return {"status": "TIMEOUT", "reason": f"bottom cycle timeout after {float(timeout):.1f}s; last_status={last_status}", "recovery_required": True}

        def run(self):
            window = getattr(self.args, "display", "BOTTOM VLA")
            panel_name = "BOTTOM VLA CONTROLS"
            try:
                self.cv2.namedWindow(window, self.cv2.WINDOW_NORMAL)
                self.cv2.namedWindow(panel_name, self.cv2.WINDOW_NORMAL)
                self.cv2.resizeWindow(window, int(self.args.width), int(self.args.height))
                self.cv2.resizeWindow(panel_name, 1240, 410)
                self.cv2.setMouseCallback(panel_name, self._mouse)
            except Exception as exc:
                print(f"[UI-WARN] window setup failed: {exc!r}")
            print("[KEYS] 1 BASKET | 8 POSITION | 3 PRESS | 4 WAIST | 5 ALIGN | 6 FINISH | 7 MODEL REJUDGE | E empty | L lock | ENTER execute | Q quit")
            active_label = "ROBUST30-AUG" if self._v110_active_model == "robust30" else self._v110_active_model.upper()
            print(f"[MODE] {self.args.mode} V116 inference-only AUTO-ENTER activeModel={active_label}")
            print("[V116-POLICY] FINISH owner=ACTIVE MODEL; legacy geometry is diagnostic only and cannot override FINISH")
            print("[V116-POLICY] next-action classifier requires trusted YOLO Mask; fallback/BGDIFF -> 2 fresh retries -> REJUDGE")
            print("[V116-AUTO-ENTER] valid non-FINISH plan reuses normal ENTER; FINISH uses existing terminal handler and suppresses later integrated AUTO-JUDGE")
            print("[INFERENCE-ONLY-V114] legacy executor DONE: GOOD/BAD/SKIP status is transient only; the normal ENTER handler arms the post-motion watcher, which clears it and immediately schedules fresh ROBUST30/V2/V6 judgement")
            print("[ALIGN-V114-PIVOT-FAST] ARM2-hold/ARM1-move uses live soft-pass transitions + exact-arrival fallback; drag=0.55 clear>=0.65")
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
                    if bool(getattr(self, "_v110_inference_only", False)):
                        status_u = status.upper()
                        if ("GOOD / BAD / SKIP" in status_u or "GOOD/BAD/SKIP" in status_u or "RESULT REQUIRED" in status_u):
                            status = "ACTION COMPLETE - AUTO NEXT MODEL JUDGE"
                    if bool(getattr(self, "_v78_initial_spread_busy", False)):
                        self.cv2.putText(shown, "INITIAL SPREAD - DETECT / PRESS OUTWARD", (20, 40), self.cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255, 120, 0), 2)
                    elif busy:
                        self.cv2.putText(shown, "ROBOT MOTION RUNNING", (20, 40), self.cv2.FONT_HERSHEY_SIMPLEX, 0.78, (0, 0, 255), 3)
                    elif infer_busy or self._auto_judge_busy:
                        prep_semantic = str(self.pending_semantic or getattr(self, "semantic_selected", None) or getattr(self, "inference_action", None) or "").upper()
                        if infer_busy and prep_semantic == "BASKET_GRASP":
                            self.cv2.putText(shown, "BASKET PREPARING / RECOGNIZING...", (20, 40), self.cv2.FONT_HERSHEY_SIMPLEX, 0.72, (0, 200, 255), 2)
                        else:
                            self.cv2.putText(shown, "AUTO / PLAN PREPARING", (20, 40), self.cv2.FONT_HERSHEY_SIMPLEX, 0.72, (0, 200, 255), 2)
                    if bool(getattr(self, "_v74_finish_processing", False)):
                        self.cv2.putText(shown, "FINISH PROCESSING - WAIT / DO NOT PRESS ENTER", (20, 76), self.cv2.FONT_HERSHEY_SIMPLEX, 0.72, (0, 200, 255), 2)
                    elif bool(getattr(self, "_v74_finish_complete_banner", False)):
                        self.cv2.putText(shown, "FINISH SUCCESS - NEXT EPISODE READY", (20, 76), self.cv2.FONT_HERSHEY_SIMPLEX, 0.78, (0, 255, 0), 3)
                    self.cv2.putText(shown, status[:130], (20, shown.shape[0] - 24), self.cv2.FONT_HERSHEY_SIMPLEX, 0.53, (0, 255, 255), 2)
                    self.cv2.imshow(window, shown)
                    self.cv2.imshow(panel_name, self._draw_panel())
                    key = self.cv2.waitKey(1) & 0xFF
                    keymap = {
                        ord("1"): "ACTION:BASKET_GRASP",
                        ord("3"): "ACTION:PRESS_SWEEP",
                        ord("4"): "ACTION:WAIST_PULL_LAYDOWN",
                        ord("5"): "ACTION:ALIGN",
                        ord("6"): "ACTION:FINISH",
                        ord("7"): "REJUDGE",
                        ord("8"): "ACTION:POSITION_ADJUST",
                        ord("e"): "EMPTY_BASELINE",
                        ord("l"): "LOCK_H",
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
                return 0
            finally:
                self.close()

    return BottomVLAApp


def _append_option_if_missing(argv, name, value):
    if any(x == name or str(x).startswith(name + "=") for x in argv):
        return list(argv)
    return [*argv, name, str(value)]


def create_integrated_app(external_frame_provider, external_camera_cap, argv=None):
    if external_frame_provider is None or external_camera_cap is None:
        raise ValueError("external_frame_provider and external_camera_cap are required")
    runtime_dir = Path(__file__).resolve().parent
    args_in = list(argv or [])
    args_in = _append_option_if_missing(args_in, "--mode", "physical")
    args_in = _append_option_if_missing(args_in, "--collection-phase", "official")
    args_in = _append_option_if_missing(args_in, "--base-main", runtime_dir / "main-33.py")
    args_in = _append_option_if_missing(args_in, "--d60-source", runtime_dir / "60-15.py")
    args_in = _append_option_if_missing(args_in, "--position-source", runtime_dir / "58-3.py")
    args_in = _append_option_if_missing(args_in, "--align-source", runtime_dir / "align-11.py")
    args_in = _append_option_if_missing(args_in, "--d54-source", runtime_dir / "54-3.py")
    args_in = _append_option_if_missing(args_in, "--d55-source", runtime_dir / "55-5.py")
    args_in = _append_option_if_missing(args_in, "--d50-source", runtime_dir / "50-1.py")
    front, remaining = v23._parse_front(args_in)
    sources = v23._resolve_sources(front)
    base, align_mod, args = v23._build_runtime(front, remaining, sources)
    args._integration_mode = True
    args._external_frame_provider = external_frame_provider
    args._external_camera_cap = external_camera_cap
    if hasattr(args, "camera_controls_enable"):
        args.camera_controls_enable = False
    if hasattr(args, "camera_controls_stabilization_frames"):
        args.camera_controls_stabilization_frames = 0
    cls = _make_app_class(base, align_mod, front, sources)
    return cls(args)


def main():
    front, remaining = v23._parse_front(sys.argv[1:])
    sources = v23._resolve_sources(front)
    base, align_mod, args = v23._build_runtime(front, remaining, sources)
    cls = _make_app_class(base, align_mod, front, sources)
    app = cls(args)
    return app.run()


if __name__ == "__main__":
    raise SystemExit(_run_with_runtime_log())
