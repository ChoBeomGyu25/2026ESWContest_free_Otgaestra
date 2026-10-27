#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Small V4L2 control helper for the Jerber maintenance UI.

The module deliberately uses ``v4l2-ctl`` instead of assuming that OpenCV's
CAP_PROP values map consistently across UVC cameras.  Controls are discovered
at runtime, so unsupported options can be disabled safely in the UI.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


_CONTROL_RE = re.compile(
    r"^\s*([A-Za-z0-9_]+)\s+0x[0-9a-fA-F]+\s+\(([^)]+)\)\s*:\s*(.*)$"
)
_MENU_RE = re.compile(r"^\s*(-?\d+)\s*:\s*(.+?)\s*$")
_VALUE_RE = re.compile(r"\b(min|max|step|default|value)=(-?\d+)\b")


@dataclass
class V4L2ControlInfo:
    name: str
    kind: str
    minimum: Optional[int] = None
    maximum: Optional[int] = None
    step: Optional[int] = None
    default: Optional[int] = None
    value: Optional[int] = None
    menu: Dict[int, str] = field(default_factory=dict)
    flags: str = ""

    @property
    def writable(self) -> bool:
        lowered = self.flags.lower()
        return "read-only" not in lowered and "inactive" not in lowered


class V4L2ControlError(RuntimeError):
    pass


class V4L2CameraControls:
    """Discover, read and write V4L2 controls for one device."""

    def __init__(self, device: str, timeout_s: float = 2.5) -> None:
        self.device = self._normalise_device(device)
        self.timeout_s = max(0.5, float(timeout_s))
        self.binary = shutil.which("v4l2-ctl")
        self.controls: Dict[str, V4L2ControlInfo] = {}

    @staticmethod
    def _normalise_device(device: str) -> str:
        token = str(device).strip()
        if token.isdigit():
            return f"/dev/video{token}"
        return token

    @property
    def available(self) -> bool:
        return bool(self.binary) and Path(self.device).exists()

    def _run(self, *args: str) -> str:
        if not self.binary:
            raise V4L2ControlError("v4l2-ctl 명령을 찾을 수 없습니다.")
        command = [self.binary, "-d", self.device, *args]
        try:
            completed = subprocess.run(
                command,
                check=False,
                capture_output=True,
                text=True,
                timeout=self.timeout_s,
            )
        except subprocess.TimeoutExpired as exc:
            raise V4L2ControlError(f"카메라 제어 명령 시간 초과: {' '.join(command)}") from exc
        if completed.returncode != 0:
            message = (completed.stderr or completed.stdout or "unknown error").strip()
            raise V4L2ControlError(message)
        return completed.stdout

    def discover(self) -> Dict[str, V4L2ControlInfo]:
        output = self._run("--list-ctrls-menus")
        controls: Dict[str, V4L2ControlInfo] = {}
        current: Optional[V4L2ControlInfo] = None

        for raw_line in output.splitlines():
            match = _CONTROL_RE.match(raw_line)
            if match:
                name, kind, body = match.groups()
                values = {key: int(value) for key, value in _VALUE_RE.findall(body)}
                flags = ""
                if "flags=" in body:
                    flags = body.split("flags=", 1)[1].strip()
                current = V4L2ControlInfo(
                    name=name,
                    kind=kind.strip().lower(),
                    minimum=values.get("min"),
                    maximum=values.get("max"),
                    step=values.get("step"),
                    default=values.get("default"),
                    value=values.get("value"),
                    flags=flags,
                )
                controls[name] = current
                continue

            menu_match = _MENU_RE.match(raw_line)
            if menu_match and current is not None and "menu" in current.kind:
                menu_value = int(menu_match.group(1))
                current.menu[menu_value] = menu_match.group(2).strip()

        self.controls = controls
        return controls

    def refresh(self) -> Dict[str, V4L2ControlInfo]:
        return self.discover()

    def get(self, name: str) -> int:
        output = self._run(f"--get-ctrl={name}")
        match = re.search(rf"\b{re.escape(name)}\s*:\s*(-?\d+)", output)
        if not match:
            raise V4L2ControlError(f"{name} 현재값을 해석할 수 없습니다: {output.strip()}")
        value = int(match.group(1))
        if name in self.controls:
            self.controls[name].value = value
        return value

    def set(self, name: str, value: int) -> None:
        if self.controls and name not in self.controls:
            raise V4L2ControlError(f"지원되지 않는 카메라 제어값: {name}")
        info = self.controls.get(name)
        integer = int(value)
        if info is not None:
            if info.minimum is not None:
                integer = max(info.minimum, integer)
            if info.maximum is not None:
                integer = min(info.maximum, integer)
            if info.step and info.minimum is not None and info.step > 1:
                integer = info.minimum + round((integer - info.minimum) / info.step) * info.step
        self._run(f"--set-ctrl={name}={integer}")
        if info is not None:
            info.value = integer

    def set_many(self, values: Dict[str, int]) -> None:
        for name, value in values.items():
            self.set(name, value)

    def first_existing(self, aliases: Iterable[str]) -> Optional[str]:
        for name in aliases:
            if name in self.controls:
                return name
        return None

    def find_menu_value(self, control_name: str, keywords: Iterable[str]) -> Optional[int]:
        info = self.controls.get(control_name)
        if info is None:
            return None
        lowered_keywords = [token.lower() for token in keywords]
        for value, label in info.menu.items():
            text = label.lower()
            if all(keyword in text for keyword in lowered_keywords):
                return value
        return None

    def find_any_menu_value(
        self,
        control_name: str,
        keyword_groups: Iterable[Iterable[str]],
    ) -> Optional[int]:
        for group in keyword_groups:
            value = self.find_menu_value(control_name, group)
            if value is not None:
                return value
        return None

    def snapshot(self, names: Iterable[str]) -> Dict[str, int]:
        result: Dict[str, int] = {}
        for name in names:
            if name not in self.controls:
                continue
            try:
                result[name] = self.get(name)
            except V4L2ControlError:
                info = self.controls.get(name)
                if info is not None and info.value is not None:
                    result[name] = int(info.value)
        return result

    def defaults(self, names: Iterable[str]) -> Dict[str, int]:
        result: Dict[str, int] = {}
        for name in names:
            info = self.controls.get(name)
            if info is not None and info.default is not None and info.writable:
                result[name] = int(info.default)
        return result


EXPOSURE_MODE_ALIASES: Tuple[str, ...] = (
    "exposure_auto",
    "auto_exposure",
)
EXPOSURE_VALUE_ALIASES: Tuple[str, ...] = (
    "exposure_absolute",
    "exposure_time_absolute",
    "exposure_time",
)
GAIN_ALIASES: Tuple[str, ...] = (
    "gain",
    "analogue_gain",
    "analog_gain",
)
AUTO_GAIN_ALIASES: Tuple[str, ...] = (
    "autogain",
    "gain_automatic",
    "auto_gain",
)
POWER_LINE_ALIASES: Tuple[str, ...] = (
    "power_line_frequency",
)
BACKLIGHT_COMPENSATION_ALIASES: Tuple[str, ...] = (
    "backlight_compensation",
)
EXPOSURE_DYNAMIC_FRAMERATE_ALIASES: Tuple[str, ...] = (
    "exposure_dynamic_framerate",
)
AUTO_WB_ALIASES: Tuple[str, ...] = (
    "white_balance_temperature_auto",
    "white_balance_automatic",
    "auto_white_balance",
)
WB_TEMPERATURE_ALIASES: Tuple[str, ...] = (
    "white_balance_temperature",
)
