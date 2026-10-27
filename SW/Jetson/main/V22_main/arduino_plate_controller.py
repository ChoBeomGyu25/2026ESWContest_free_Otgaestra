#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Standalone Arduino folding-plate serial controller for the Jerber UI.

This module intentionally avoids importing the large robot runtime. It supports:
- automatic /dev/ttyACM* and /dev/ttyUSB* discovery;
- READY handshake after Arduino reset;
- manual plate commands 1..5 and interrupt commands;
- strict e4/e2 motion V7 firmware + build handshake;
- one-shot F (legacy sequential) and G (synchronized) cycles with DONE wait;
- clean reconnect/close behavior.
"""

from __future__ import annotations

import glob
import threading
import time
from typing import List, Optional

try:
    import serial as pyserial
    from serial.tools import list_ports
except Exception:
    pyserial = None
    list_ports = None


def discover_serial_ports(requested: str = "auto") -> List[str]:
    raw = str(requested or "auto").strip()
    if raw.lower() not in {"", "auto", "detect"}:
        return [raw]

    found: List[str] = []
    if list_ports is not None:
        try:
            found.extend(str(item.device) for item in list_ports.comports())
        except Exception:
            pass
    found.extend(glob.glob("/dev/ttyACM*"))
    found.extend(glob.glob("/dev/ttyUSB*"))

    unique = sorted(
        set(found),
        key=lambda value: (
            0 if "ttyACM" in value else 1 if "ttyUSB" in value else 2,
            value,
        ),
    )
    return unique


class ArduinoPlateController:
    REQUIRED_PROTOCOL_ID = "JERBER_E4_E2_MOTION_V7"
    REQUIRED_BUILD_ID = "JERBER_E4_E2_MOTION_V7_B1_20260817"
    SAFE_COMMANDS = {"1", "2", "3", "4", "5", "S", "X", "D", "H", "?", "F", "G"}
    INTERRUPT_COMMANDS = {"X", "D", "H", "?"}

    def __init__(
        self,
        port: str,
        baudrate: int = 9600,
        ready_timeout_s: float = 20.0,
        done_timeout_s: float = 60.0,
        send_enabled: bool = False,
        allow_during_dry_run: bool = True,
        auto_rearm_on_nonfinish: bool = False,
        rearm_cooldown_s: float = 0.0,
    ) -> None:
        self.requested_port = str(port or "auto")
        self.port = self.requested_port
        self.baudrate = int(baudrate)
        self.ready_timeout_s = max(1.0, float(ready_timeout_s))
        self.done_timeout_s = max(1.0, float(done_timeout_s))
        self.send_enabled = bool(send_enabled)
        self.allow_during_dry_run = bool(allow_during_dry_run)
        self.auto_rearm_on_nonfinish = bool(auto_rearm_on_nonfinish)
        self.rearm_cooldown_s = max(0.0, float(rearm_cooldown_s))

        self.ser = None
        self.connected = False
        self.ready = False
        self.protocol_ok = False
        self.firmware_id = ""
        self.build_id = ""
        self.fold_latched = False
        self.fold_in_progress = False
        self.last_done_at = 0.0
        self.last_error = ""
        self.last_line = ""
        self.runtime_dry_run = False

        self._closing = False
        self._tx_lock = threading.Lock()
        self._state_lock = threading.RLock()
        self._ready_event = threading.Event()
        self._protocol_event = threading.Event()
        self._build_event = threading.Event()
        self._folding_event = threading.Event()
        self._done_event = threading.Event()
        self._abort_event = threading.Event()
        self._manual_done_event = threading.Event()
        self._manual_expected_plate: Optional[str] = None

        # Integration folding sequence needs to distinguish:
        #   plate reached operated/up position
        #   plate returned to HOME
        # without changing the existing manual UI behavior.
        self._plate_operated_events = {
            str(plate): threading.Event() for plate in range(1, 6)
        }
        self._plate_home_events = {
            str(plate): threading.Event() for plate in range(1, 6)
        }
        self._global_home_event = threading.Event()
        self._home_wait_required: set[str] = set()
        self._reader_thread: Optional[threading.Thread] = None

    def _reset_connection_state(self) -> None:
        self.connected = False
        self.ready = False
        self.protocol_ok = False
        self.firmware_id = ""
        self.build_id = ""
        self.last_line = ""
        self._ready_event.clear()
        self._protocol_event.clear()
        self._build_event.clear()
        self._folding_event.clear()
        self._done_event.clear()
        self._abort_event.clear()
        self._manual_done_event.clear()
        self._manual_expected_plate = None
        for event in self._plate_operated_events.values():
            event.clear()
        for event in self._plate_home_events.values():
            event.clear()
        self._global_home_event.clear()
        self._home_wait_required.clear()

    def connect(self) -> bool:
        if pyserial is None:
            self.last_error = "pyserial is not installed"
            return False

        self.close()
        self._closing = False
        candidates = discover_serial_ports(self.requested_port)
        if not candidates:
            self.last_error = (
                f"serial port not found: requested={self.requested_port!r}; "
                "checked /dev/ttyACM* and /dev/ttyUSB*"
            )
            return False

        errors: List[str] = []
        for candidate in candidates:
            self._reset_connection_state()
            try:
                ser_obj = pyserial.Serial(
                    candidate,
                    baudrate=self.baudrate,
                    timeout=0.15,
                    write_timeout=1.0,
                )
                self.ser = ser_obj
                self.port = candidate
                self.connected = True
                self._reader_thread = threading.Thread(
                    target=self._reader_loop,
                    args=(ser_obj,),
                    name="arduino-plate-reader",
                    daemon=True,
                )
                self._reader_thread.start()

                # Opening an Uno/Nano serial port normally resets the MCU. Wait for
                # firmware homing and READY. No probe command is sent before the
                # firmware identifies itself, so a wrong serial candidate is not
                # accidentally commanded during automatic port discovery.
                deadline = time.monotonic() + self.ready_timeout_s
                while time.monotonic() < deadline:
                    if self._ready_event.is_set() and self._protocol_event.is_set() and self._build_event.is_set():
                        self.last_error = ""
                        return True
                    time.sleep(0.03)

                if self._ready_event.is_set() and (not self._protocol_event.is_set() or not self._build_event.is_set()):
                    errors.append(
                        f"{candidate}: firmware protocol mismatch; "
                        f"required_proto={self.REQUIRED_PROTOCOL_ID}, detected_proto={self.firmware_id or 'legacy/unknown'}, "
                        f"required_build={self.REQUIRED_BUILD_ID}, detected_build={self.build_id or 'unknown'}"
                    )
                else:
                    errors.append(f"{candidate}: READY/protocol timeout {self.ready_timeout_s:.1f}s")
            except Exception as exc:
                errors.append(f"{candidate}: {exc!r}")
            finally:
                # Keep the port only when both READY and the exact V6 protocol
                # handshake succeeded. A READY-only legacy firmware must be
                # closed so auto-discovery can continue to the next candidate.
                if not (self.ready and self.protocol_ok and self._build_event.is_set()):
                    self._close_serial_only()

        self.last_error = "; ".join(errors)
        return False

    def _reader_loop(self, ser_obj) -> None:
        while not self._closing and self.ser is ser_obj:
            try:
                raw = ser_obj.readline()
                if not raw:
                    continue
                line = raw.decode("utf-8", errors="replace").strip()
                if not line:
                    continue
                self.last_line = line
                upper = line.upper()
                if self.REQUIRED_PROTOCOL_ID in upper:
                    self.protocol_ok = True
                    self.firmware_id = self.REQUIRED_PROTOCOL_ID
                    self._protocol_event.set()
                elif upper.startswith("PROTO:"):
                    self.firmware_id = line.split(":", 1)[-1].strip()
                if upper.startswith("BUILD:"):
                    self.build_id = line.split(":", 1)[-1].strip()
                    if self.build_id == self.REQUIRED_BUILD_ID:
                        self._build_event.set()
                if upper == "READY" or "CONTROLLER READY" in upper:
                    self.ready = True
                    self._ready_event.set()
                if upper == "FOLDING" or upper.startswith("[FOLD] START"):
                    self._folding_event.set()
                if upper == "DONE" or upper.startswith("[FOLD] DONE"):
                    self._done_event.set()
                # Track physical plate position acknowledgements independently
                # from the legacy one-command UI wait.  The integrated automatic
                # fold uses these events to hold 1/2/4/5 at OPERATED and to wait
                # for plate 3 / HOME completion deterministically.
                for plate in ("1", "2", "3", "4", "5"):
                    prefix = f"[PLATE {plate}]"
                    if not upper.startswith(prefix):
                        continue
                    if "OPERATED POSITION REACHED" in upper:
                        self._plate_operated_events[plate].set()
                    if "HOME POSITION REACHED" in upper:
                        self._plate_home_events[plate].set()
                    break

                if upper.startswith("[HOME]") and (
                    "DONE" in upper
                    or "COMPLETE" in upper
                    or "POSITION REACHED" in upper
                    or "ALL HOME" in upper
                    or "ALL PLATES RETURNED HOME" in upper
                ):
                    # Actual e4/e2 v7 firmware emits:
                    #   [HOME] All plates returned home.
                    # immediately after commanding all servos to their start angles.
                    # Physical settling is handled by integration_runtime after this ACK.
                    self._global_home_event.set()

                expected_plate = self._manual_expected_plate
                if expected_plate is not None:
                    done_prefix = f"[PLATE {expected_plate}]"
                    if upper.startswith(done_prefix) and (
                        "OPERATED POSITION REACHED" in upper
                        or "HOME POSITION REACHED" in upper
                    ):
                        self._manual_done_event.set()
                if upper.startswith("ERROR") or upper.startswith("[ERROR]"):
                    self.last_error = line
                if self.fold_in_progress and (
                    upper.startswith("[STOP]")
                    or upper.startswith("[DETACH]")
                    or upper.startswith("[HOME]")
                    or upper == "ABORTED"
                ):
                    self._abort_event.set()
            except Exception as exc:
                if not self._closing and self.ser is ser_obj:
                    self.last_error = repr(exc)
                break

    def _write_raw(self, command: str, require_ready: bool = True) -> bool:
        command = str(command).strip().upper()
        if command not in self.SAFE_COMMANDS:
            self.last_error = f"unsupported command: {command!r}"
            return False
        if not self.connected or self.ser is None:
            self.last_error = "serial connection is unavailable"
            return False
        if require_ready and not self.ready:
            self.last_error = "READY handshake is not complete"
            return False
        if require_ready and not self.protocol_ok:
            self.last_error = (
                f"firmware protocol mismatch: required={self.REQUIRED_PROTOCOL_ID}, "
                f"detected={self.firmware_id or 'legacy/unknown'}"
            )
            return False
        try:
            with self._tx_lock:
                self.ser.write((command + "\n").encode("ascii"))
                self.ser.flush()
            return True
        except Exception as exc:
            self.last_error = repr(exc)
            return False

    def hardware_allowed(self, dry_run: bool) -> bool:
        if not self.send_enabled:
            return False
        if bool(dry_run) and not self.allow_during_dry_run:
            return False
        return True

    def trigger_fold_and_wait(
        self,
        source: str,
        dry_run: bool = False,
        command: str = "G",
    ) -> str:
        """Run one firmware-owned one-shot sequence and wait for DONE.

        F selects the legacy e2 sequence (1 -> 2 -> 3 -> 4 -> 5).
        G selects the synchronized seven-stage scenario used by the display UI.
        """
        command = str(command).strip().upper()
        if command not in {"F", "G"}:
            self.last_error = f"one-shot command must be F or G, got {command!r}"
            return "UNSUPPORTED_COMMAND"

        with self._state_lock:
            if self.fold_in_progress:
                return "ALREADY_RUNNING"
            if self.fold_latched:
                return "LATCHED"
            if not self.hardware_allowed(dry_run):
                return "DRY_RUN"
            if not self.connected or not self.ready:
                return "NOT_READY"
            self.fold_latched = True
            self.fold_in_progress = True
            self._folding_event.clear()
            self._done_event.clear()
            self._abort_event.clear()

        if not self._write_raw(command):
            with self._state_lock:
                self.fold_in_progress = False
            return "SEND_FAILED"

        deadline = time.monotonic() + self.done_timeout_s
        result = "TIMEOUT"
        while time.monotonic() < deadline:
            if self._done_event.wait(0.10):
                result = "DONE"
                break
            if self._abort_event.is_set():
                result = "ABORTED"
                break
        with self._state_lock:
            self.fold_in_progress = False
            if result == "DONE":
                self.last_done_at = time.time()
        return result

    def send_manual(self, token: str, dry_run: bool = False) -> bool:
        raw = str(token).strip().lower()
        aliases = {
            "fold": "G", "g": "G", "showcase": "G", "sync": "G",
            "sync_v2": "G", "sync_v3": "G",
            "f": "F", "legacy": "F", "legacy_e2": "F",
            "start": "S", "s": "S",
            "stop": "X", "x": "X", "detach": "D", "d": "D",
            "home": "H", "h": "H", "status": "?", "?": "?",
        }
        if raw in {"reset", "rearm", "unlock"}:
            return self.rearm("UI command", force=True)
        command = aliases.get(raw, raw.upper())
        if command not in self.SAFE_COMMANDS:
            self.last_error = f"unsupported command: {command!r}"
            return False
        if command in {"F", "G"}:
            return self.trigger_fold_and_wait(
                "manual", dry_run=dry_run, command=command
            ) == "DONE"
        if not self.hardware_allowed(dry_run):
            self.last_error = "actual transmission is disabled"
            return False
        with self._state_lock:
            if self.fold_in_progress and command not in self.INTERRUPT_COMMANDS:
                self.last_error = f"{command} blocked while fold is running"
                return False

        # Individual plate commands use the original e2 toggle motion.
        # Wait for the firmware's terminal acknowledgement so the UI does not
        # re-enable another plate while the 1 s hold / slow return is still active.
        if command in {"1", "2", "3", "4", "5"}:
            self._manual_expected_plate = command
            self._manual_done_event.clear()
            if not self._write_raw(command):
                self._manual_expected_plate = None
                return False
            if not self._manual_done_event.wait(8.0):
                self.last_error = f"manual plate {command} completion timeout"
                self._manual_expected_plate = None
                return False
            self._manual_expected_plate = None
            self.last_error = ""
            return True

        return self._write_raw(command)


    def operate_plate_and_wait(
        self,
        plate: int,
        *,
        wait_for: str = "operated",
        dry_run: bool = False,
        timeout_s: float = 8.0,
    ) -> str:
        """Send one plate command and wait for a specific physical acknowledgement.

        wait_for="operated"
            Return only after firmware reports `[PLATE N] OPERATED POSITION REACHED`.
            This is used for plates 1/2/4/5 so the next command can run while the
            previous plate is still raised.

        wait_for="home"
            Return only after firmware reports `[PLATE N] HOME POSITION REACHED`.
            This is used for plate 3, which must complete a full up/down cycle.
        """
        plate_s = str(int(plate))
        if plate_s not in {"1", "2", "3", "4", "5"}:
            raise ValueError(f"unsupported plate: {plate!r}")

        mode = str(wait_for or "").strip().lower()
        if mode not in {"operated", "home"}:
            raise ValueError(f"wait_for must be 'operated' or 'home', got {wait_for!r}")

        if not self.hardware_allowed(dry_run):
            self.last_error = "actual transmission is disabled"
            return "DRY_RUN"
        with self._state_lock:
            if self.fold_in_progress:
                self.last_error = f"plate {plate_s} blocked while fold is running"
                return "ALREADY_RUNNING"

        operated_event = self._plate_operated_events[plate_s]
        home_event = self._plate_home_events[plate_s]
        operated_event.clear()
        home_event.clear()

        if not self._write_raw(plate_s):
            return "SEND_FAILED"

        target_event = operated_event if mode == "operated" else home_event
        if target_event.wait(max(0.5, float(timeout_s))):
            self.last_error = ""
            return "OPERATED" if mode == "operated" else "HOME"

        self.last_error = f"plate {plate_s} {mode} acknowledgement timeout"
        return "TIMEOUT"

    def home_and_wait(
        self,
        required_plates,
        *,
        dry_run: bool = False,
        timeout_s: float = 10.0,
    ) -> str:
        """Send global H once and wait until the requested plates reach HOME.

        The Arduino H command is the same HOME command used by the display UI.
        For the final fold stage this allows plates 4 and 5 to start returning
        together rather than releasing one plate and then the other.
        """
        required = {str(int(value)) for value in required_plates}
        if not required or not required.issubset({"1", "2", "3", "4", "5"}):
            raise ValueError(f"invalid required_plates={required_plates!r}")

        if not self.hardware_allowed(dry_run):
            self.last_error = "actual transmission is disabled"
            return "DRY_RUN"

        with self._state_lock:
            if self.fold_in_progress:
                self.last_error = "HOME blocked while firmware-owned fold is running"
                return "ALREADY_RUNNING"

        self._global_home_event.clear()
        self._home_wait_required = set(required)
        for plate in required:
            self._plate_home_events[plate].clear()

        if not self._write_raw("H"):
            self._home_wait_required.clear()
            return "SEND_FAILED"

        deadline = time.monotonic() + max(0.5, float(timeout_s))
        while time.monotonic() < deadline:
            # Some firmware builds emit a single [HOME] completion line; others
            # emit one HOME POSITION REACHED line per plate.  Accept either,
            # while requiring all requested per-plate acknowledgements when the
            # generic completion marker is absent.
            if self._global_home_event.is_set():
                self.last_error = ""
                self._home_wait_required.clear()
                return "HOME"
            if all(self._plate_home_events[plate].is_set() for plate in required):
                self.last_error = ""
                self._home_wait_required.clear()
                return "HOME"
            time.sleep(0.03)

        missing = sorted(
            plate for plate in required
            if not self._plate_home_events[plate].is_set()
        )
        self._home_wait_required.clear()
        self.last_error = f"HOME completion timeout; missing_plates={missing}"
        return "TIMEOUT"

    def rearm(self, reason: str, force: bool = False) -> bool:
        with self._state_lock:
            if self.fold_in_progress:
                return False
            if not self.fold_latched:
                return True
            if not force and self.last_done_at > 0.0:
                if time.time() - self.last_done_at < self.rearm_cooldown_s:
                    return False
            self.fold_latched = False
            self._done_event.clear()
            self._abort_event.clear()
        return True

    def status_text(self) -> str:
        with self._state_lock:
            return (
                f"port={self.port!r} connected={self.connected} ready={self.ready} "
                f"protocol_ok={self.protocol_ok} firmware={self.firmware_id!r} build={self.build_id!r} "
                f"send={self.send_enabled} folding={self.fold_in_progress} "
                f"latched={self.fold_latched} last={self.last_line!r} "
                f"error={self.last_error!r}"
            )

    def _close_serial_only(self) -> None:
        ser_obj = self.ser
        self.ser = None
        self.connected = False
        self.ready = False
        try:
            if ser_obj is not None:
                ser_obj.close()
        except Exception:
            pass

    def close(self) -> None:
        self._closing = True
        self._close_serial_only()

    cleanup = close
