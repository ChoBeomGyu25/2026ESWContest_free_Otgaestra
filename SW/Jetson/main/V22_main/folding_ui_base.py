#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PySide6 folding-machine UI with a settings flow and folding-board operation page.

Screen flow:
  Main -> Settings -> Operation check -> Folding-board operation

Folding-board serial control uses the local standalone
arduino_plate_controller.py module. The protocol supports individual plate
commands 1..5, legacy command F, and synchronized command G
(strict JERBER_E4_E2_MOTION_V7 firmware).

Important:
  - Do not run step_e71_bottom_runtime.py at the same time. Both programs would
    compete for /dev/ttyACM0.
  - Real Arduino transmission is enabled by default in the v7 launcher; use --no-arduino-send to disable it.
  - The display UI's `작동` button intentionally calls the controller's G synchronized cycle
    without the garment READY_TO_FOLD perception gate.
"""

from __future__ import annotations

import argparse
import os
import sys
import threading
import time
from pathlib import Path
from typing import Callable, Optional

# Docker + X11 stability. Keep before cv2/PySide6 imports.
os.environ.setdefault("QT_X11_NO_MITSHM", "1")
os.environ.setdefault("GDK_DISABLE_SHM", "1")
os.environ.setdefault("NO_AT_BRIDGE", "1")

import cv2
from PySide6.QtCore import QFile, QObject, QPointF, QRectF, Qt, QTimer, Signal
from PySide6.QtGui import (
    QColor,
    QFont,
    QImage,
    QKeySequence,
    QMouseEvent,
    QPainter,
    QPen,
    QPixmap,
    QShortcut,
)
from PySide6.QtUiTools import QUiLoader
from PySide6.QtWidgets import (
    QApplication,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QSizePolicy,
    QSpacerItem,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

try:
    from arduino_plate_controller import ArduinoPlateController
    RUNTIME_IMPORT_ERROR = ""
except Exception as exc:  # UI/camera can still be previewed without Arduino support.
    ArduinoPlateController = None  # type: ignore[assignment]
    RUNTIME_IMPORT_ERROR = repr(exc)


BASE_DIR = Path(__file__).resolve().parent
UI_PATH = BASE_DIR / "folding_machine.ui"
QSS_PATH = BASE_DIR / "folding_machine.qss"
BOARD_IMAGE_PATH = BASE_DIR / "folding_plate.png"


class UiSignals(QObject):
    log = Signal(str)
    connection_finished = Signal(bool, str)
    action_finished = Signal(str, bool, str)


class PlateBoardWidget(QWidget):
    """Draw the supplied top-view image and emit a plate number on touch.

    Coordinates are measured from the user's 1024x1024 clean image. Plate 5 is
    the lower-center pair that is driven as one command in the current firmware.
    """

    plateClicked = Signal(int)

    # Image-space rectangles: x, y, width, height in the original 1024x1024 PNG.
    PLATE_RECTS = {
        1: QRectF(314.0, 263.0, 127.0, 496.0),
        2: QRectF(599.0, 263.0, 127.0, 496.0),
        3: QRectF(443.0, 263.0, 153.0, 151.0),
        4: QRectF(443.0, 417.0, 153.0, 150.0),
        5: QRectF(443.0, 570.0, 153.0, 189.0),
    }

    def __init__(self, image_path: Path, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setObjectName("plateBoardWidget")
        self.setMinimumSize(520, 520)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMouseTracking(True)

        self._pixmap = QPixmap(str(image_path))
        self._image_size = self._pixmap.size()
        self._draw_rect = QRectF()
        self._hover_plate: Optional[int] = None
        self._selected_plate: Optional[int] = None
        self._active_plates: set[int] = set()
        self._enabled_for_motion = False

    def set_motion_enabled(self, enabled: bool) -> None:
        self._enabled_for_motion = bool(enabled)
        self.setCursor(
            Qt.CursorShape.PointingHandCursor
            if self._enabled_for_motion
            else Qt.CursorShape.ForbiddenCursor
        )
        self.update()

    def set_selected_plate(self, plate: Optional[int]) -> None:
        self._selected_plate = int(plate) if plate in self.PLATE_RECTS else None
        if self._selected_plate is not None:
            self._active_plates = {self._selected_plate}
        else:
            self._active_plates.clear()
        self.update()

    def set_active_plates(self, plates) -> None:
        self._selected_plate = None
        self._active_plates = {int(value) for value in plates if int(value) in self.PLATE_RECTS}
        self.update()

    def _fit_rect(self) -> QRectF:
        if self._pixmap.isNull() or self._image_size.width() <= 0:
            return QRectF(self.rect())
        available = QRectF(self.rect()).adjusted(10.0, 10.0, -10.0, -10.0)
        sx = available.width() / float(self._image_size.width())
        sy = available.height() / float(self._image_size.height())
        scale = min(sx, sy)
        width = self._image_size.width() * scale
        height = self._image_size.height() * scale
        x = available.center().x() - width / 2.0
        y = available.center().y() - height / 2.0
        return QRectF(x, y, width, height)

    def _image_to_widget_rect(self, image_rect: QRectF) -> QRectF:
        if self._image_size.width() <= 0 or self._image_size.height() <= 0:
            return QRectF()
        sx = self._draw_rect.width() / float(self._image_size.width())
        sy = self._draw_rect.height() / float(self._image_size.height())
        return QRectF(
            self._draw_rect.left() + image_rect.left() * sx,
            self._draw_rect.top() + image_rect.top() * sy,
            image_rect.width() * sx,
            image_rect.height() * sy,
        )

    def _plate_at(self, point: QPointF) -> Optional[int]:
        for plate, image_rect in self.PLATE_RECTS.items():
            if self._image_to_widget_rect(image_rect).contains(point):
                return plate
        return None

    def paintEvent(self, event) -> None:  # noqa: N802 (Qt API name)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.fillRect(self.rect(), QColor("#07101C"))

        self._draw_rect = self._fit_rect()
        if self._pixmap.isNull():
            painter.setPen(QColor("#FF8A8A"))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "folding_plate.png 파일이 없습니다")
            return

        painter.drawPixmap(self._draw_rect, self._pixmap, QRectF(self._pixmap.rect()))

        for plate, image_rect in self.PLATE_RECTS.items():
            rect = self._image_to_widget_rect(image_rect)
            active = plate == self._selected_plate or plate in self._active_plates
            hovered = plate == self._hover_plate

            if active:
                fill = QColor(57, 123, 255, 100)
                border = QColor("#6FA5FF")
                width = 5
            elif hovered and self._enabled_for_motion:
                fill = QColor(53, 208, 127, 65)
                border = QColor("#55E0A0")
                width = 4
            else:
                # Draw an opaque common base color over every plate.  The source
                # PNG contains a different tint under plate 1, so a translucent
                # overlay would still make only plate 1 look different.
                fill = QColor("#2F6FD8")
                border = QColor("#77A9FF")
                width = 2

            painter.fillRect(rect, fill)
            painter.setPen(QPen(border, width))
            painter.drawRoundedRect(rect, 8.0, 8.0)

            radius = max(18.0, min(27.0, rect.width() * 0.16))
            center = rect.center()
            badge_rect = QRectF(
                center.x() - radius,
                center.y() - radius,
                radius * 2.0,
                radius * 2.0,
            )
            painter.setPen(Qt.PenStyle.NoPen)
            # Badge and number colors are identical for plates 1~5.
            painter.setBrush(QColor("#1F55B5") if self._enabled_for_motion else QColor("#4C596B"))
            painter.drawEllipse(badge_rect)
            painter.setPen(QColor("#FFFFFF"))
            font = QFont("Noto Sans CJK KR", int(radius * 0.9), QFont.Weight.Bold)
            painter.setFont(font)
            painter.drawText(badge_rect, Qt.AlignmentFlag.AlignCenter, str(plate))

        painter.setPen(QColor("#B7C5D8"))
        painter.setFont(QFont("Noto Sans CJK KR", 11, QFont.Weight.DemiBold))
        note = "판을 터치하면 해당 번호의 단독 테스트 명령을 전송함"
        painter.drawText(
            QRectF(12.0, self.height() - 38.0, self.width() - 24.0, 28.0),
            Qt.AlignmentFlag.AlignCenter,
            note,
        )

    def mouseMoveEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        plate = self._plate_at(event.position()) if self._enabled_for_motion else None
        if plate != self._hover_plate:
            self._hover_plate = plate
            self.update()
        super().mouseMoveEvent(event)

    def leaveEvent(self, event) -> None:  # noqa: N802
        self._hover_plate = None
        self.update()
        super().leaveEvent(event)

    def mousePressEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        if event.button() == Qt.MouseButton.LeftButton and self._enabled_for_motion:
            plate = self._plate_at(event.position())
            if plate is not None:
                self.plateClicked.emit(plate)
                event.accept()
                return
        super().mousePressEvent(event)


class FoldingBoardUiController(QObject):
    def __init__(self, args: argparse.Namespace, app: QApplication) -> None:
        super().__init__()
        self.args = args
        self.app = app
        self.signals = UiSignals()
        self.controller = None
        self.cap: Optional[cv2.VideoCapture] = None
        self._busy = False
        self._closing = False
        self._last_rx_seen = ""

        self.window = self._load_ui()
        self._build_pages()
        self._connect_main_page()
        self._connect_maintenance_page()
        self._setup_camera()
        self._setup_shortcuts()
        self._setup_timers()
        self._connect_signals()

        if self.args.arduino_enable:
            self.connect_arduino_async()
        else:
            self._append_log("Arduino 연결 비활성화 상태")

    @staticmethod
    def _load_ui() -> QMainWindow:
        loader = QUiLoader()
        ui_file = QFile(str(UI_PATH))
        if not ui_file.open(QFile.ReadOnly):
            raise RuntimeError(f"UI 파일을 열 수 없음: {UI_PATH}")
        window = loader.load(ui_file)
        ui_file.close()
        if window is None:
            raise RuntimeError(loader.errorString())
        return window

    def _require(self, cls, name: str):
        widget = self.window.findChild(cls, name)
        if widget is None:
            raise RuntimeError(f"위젯을 찾을 수 없음: {name}")
        return widget

    def _build_pages(self) -> None:
        main_page = self.window.takeCentralWidget()
        if main_page is None:
            raise RuntimeError("기존 centralWidget을 찾을 수 없음")
        main_page.setObjectName("mainPage")

        self.page_stack = QStackedWidget()
        self.page_stack.setObjectName("pageStack")
        self.window.setCentralWidget(self.page_stack)
        self.page_stack.addWidget(main_page)

        self.settings_page = self._create_settings_page()
        self.plate_test_page = self._create_plate_test_page()
        self.page_stack.addWidget(self.settings_page)
        self.page_stack.addWidget(self.plate_test_page)
        self.page_stack.setCurrentIndex(0)

    def _create_page_header(self, title: str, subtitle: str, back_name: str) -> tuple[QFrame, QPushButton]:
        frame = QFrame()
        frame.setObjectName("settingsHeader")
        layout = QHBoxLayout(frame)
        layout.setContentsMargins(18, 14, 18, 14)
        layout.setSpacing(14)

        back = QPushButton("‹  뒤로")
        back.setObjectName(back_name)
        back.setMinimumSize(118, 56)
        layout.addWidget(back)

        title_box = QVBoxLayout()
        title_box.setSpacing(2)
        title_label = QLabel(title)
        title_label.setObjectName("pageTitle")
        subtitle_label = QLabel(subtitle)
        subtitle_label.setObjectName("pageSubtitle")
        title_box.addWidget(title_label)
        title_box.addWidget(subtitle_label)
        layout.addLayout(title_box)
        layout.addStretch(1)
        return frame, back

    def _create_settings_page(self) -> QWidget:
        page = QWidget()
        page.setObjectName("settingsPage")
        root = QVBoxLayout(page)
        root.setContentsMargins(24, 20, 24, 20)
        root.setSpacing(18)

        header, self.settings_back_button = self._create_page_header(
            "설정",
            "기계 상태와 각 구동부를 점검합니다.",
            "settingsBackButton",
        )
        root.addWidget(header)

        card = QFrame()
        card.setObjectName("settingsCard")
        card_layout = QVBoxLayout(card)
        card_layout.setContentsMargins(24, 22, 24, 24)
        card_layout.setSpacing(14)

        section = QLabel("작동 점검")
        section.setObjectName("sectionTitle")
        section_note = QLabel("실제 장치를 개별적으로 움직이는 유지보수용 기능입니다.")
        section_note.setObjectName("sectionNote")
        card_layout.addWidget(section)
        card_layout.addWidget(section_note)

        self.plate_test_entry_button = QPushButton(
            "폴딩판 작동\n판별 단독 동작 및 자동 작동"
        )
        self.plate_test_entry_button.setObjectName("plateTestEntryButton")
        self.plate_test_entry_button.setMinimumHeight(126)
        card_layout.addWidget(self.plate_test_entry_button)

        placeholder_row = QHBoxLayout()
        self.camera_test_entry_button = QPushButton(
            "카메라 점검\n노출·게인·플리커 설정"
        )
        self.camera_test_entry_button.setObjectName("cameraTestEntryButton")
        self.camera_test_entry_button.setMinimumHeight(100)
        placeholder_row.addWidget(self.camera_test_entry_button)

        self.robot_test_entry_button = QPushButton("로봇팔 점검\n자동 경로 및 수동 미세 조작")
        self.robot_test_entry_button.setObjectName("robotTestEntryButton")
        self.robot_test_entry_button.setEnabled(True)
        self.robot_test_entry_button.setMinimumHeight(100)
        placeholder_row.addWidget(self.robot_test_entry_button)
        card_layout.addLayout(placeholder_row)

        root.addWidget(card)
        root.addStretch(1)
        return page

    def _create_plate_test_page(self) -> QWidget:
        page = QWidget()
        page.setObjectName("plateTestPage")
        root = QVBoxLayout(page)
        root.setContentsMargins(24, 20, 24, 20)
        root.setSpacing(16)

        header, self.plate_back_button = self._create_page_header(
            "폴딩판 작동",
            "판을 직접 터치하거나 자동 작동을 실행합니다.",
            "plateBackButton",
        )
        self.arduino_status_badge = QLabel("Arduino 연결 확인 중")
        self.arduino_status_badge.setObjectName("arduinoStatusBadge")
        self.arduino_status_badge.setMinimumSize(250, 52)
        self.arduino_status_badge.setAlignment(Qt.AlignmentFlag.AlignCenter)
        header.layout().addWidget(self.arduino_status_badge)
        root.addWidget(header)

        content = QHBoxLayout()
        content.setSpacing(18)

        board_card = QFrame()
        board_card.setObjectName("plateBoardCard")
        board_layout = QVBoxLayout(board_card)
        board_layout.setContentsMargins(16, 14, 16, 14)
        board_layout.setSpacing(8)
        board_title = QLabel("판 선택")
        board_title.setObjectName("cardTitle")
        board_layout.addWidget(board_title)
        self.plate_board_widget = PlateBoardWidget(BOARD_IMAGE_PATH)
        board_layout.addWidget(self.plate_board_widget, 1)
        content.addWidget(board_card, 3)

        control_card = QFrame()
        control_card.setObjectName("plateControlCard")
        control_layout = QVBoxLayout(control_card)
        control_layout.setContentsMargins(20, 18, 20, 18)
        control_layout.setSpacing(12)

        control_title = QLabel("작동 제어")
        control_title.setObjectName("cardTitle")
        control_layout.addWidget(control_title)

        self.selected_plate_label = QLabel("먼저 로봇팔을 대기 위치로 이동하세요.")
        self.selected_plate_label.setObjectName("selectedPlateLabel")
        self.selected_plate_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.selected_plate_label.setMinimumHeight(48)
        control_layout.addWidget(self.selected_plate_label)

        self.plate_roarm_standby_button = QPushButton(
            "로봇팔 대기 위치로 이동\n완료 후 폴딩판 작동이 활성화됩니다"
        )
        self.plate_roarm_standby_button.setObjectName("plateRoarmStandbyButton")
        self.plate_roarm_standby_button.setMinimumHeight(82)
        control_layout.addWidget(self.plate_roarm_standby_button)

        self.auto_test_button = QPushButton("▶  작동")
        self.auto_test_button.setObjectName("autoPlateTestButton")
        self.auto_test_button.setMinimumHeight(98)
        control_layout.addWidget(self.auto_test_button)

        action_row = QHBoxLayout()
        action_row.setSpacing(10)
        self.stop_button = QPushButton("■  즉시 정지")
        self.stop_button.setObjectName("stopPlateButton")
        self.home_button = QPushButton("⌂  홈 복귀")
        self.home_button.setObjectName("homePlateButton")
        self.stop_button.setMinimumHeight(66)
        self.home_button.setMinimumHeight(66)
        action_row.addWidget(self.stop_button)
        action_row.addWidget(self.home_button)
        control_layout.addLayout(action_row)

        aux_row = QHBoxLayout()
        aux_row.setSpacing(10)
        self.reconnect_button = QPushButton("↻  연결 재시도")
        self.reconnect_button.setObjectName("reconnectButton")
        self.rearm_button = QPushButton("잠금 초기화")
        self.rearm_button.setObjectName("rearmButton")
        self.reconnect_button.setMinimumHeight(56)
        self.rearm_button.setMinimumHeight(56)
        aux_row.addWidget(self.reconnect_button)
        aux_row.addWidget(self.rearm_button)
        control_layout.addLayout(aux_row)

        log_title = QLabel("상태 로그")
        log_title.setObjectName("logTitle")
        control_layout.addWidget(log_title)
        self.log_view = QPlainTextEdit()
        self.log_view.setObjectName("plateLogView")
        self.log_view.setReadOnly(True)
        self.log_view.setMaximumBlockCount(120)
        self.log_view.setMinimumHeight(150)
        control_layout.addWidget(self.log_view, 1)

        warning = QLabel(
            "주의: 테스트 전 보드 위의 옷·도구·손을 모두 치우고, 물리 비상정지 스위치를 즉시 누를 수 있는 상태에서 실행하세요."
        )
        warning.setObjectName("plateWarningLabel")
        warning.setWordWrap(True)
        warning.setMinimumHeight(70)
        control_layout.addWidget(warning)

        content.addWidget(control_card, 2)
        root.addLayout(content, 1)
        return page

    def _connect_main_page(self) -> None:
        self.settings_button = self._require(QPushButton, "settingsButton")
        self.start_button = self._require(QPushButton, "startButton")
        self.pause_button = self._require(QPushButton, "pauseButton")
        self.reset_button = self._require(QPushButton, "resetButton")
        self.state_badge = self._require(QLabel, "stateBadge")
        self.step_label = self._require(QLabel, "stepLabel")
        self.progress_text = self._require(QLabel, "progressText")
        self.progress_bar = self._require(QProgressBar, "progressBar")
        self.camera_label = self._require(QLabel, "cameraLabel")
        self.camera_live_badge = self._require(QLabel, "cameraLiveBadge")
        self.connection_label = self._require(QLabel, "connectionLabel")

        self.settings_button.clicked.connect(lambda: self.page_stack.setCurrentIndex(1))
        self.settings_back_button.clicked.connect(lambda: self.page_stack.setCurrentIndex(0))
        self.plate_test_entry_button.clicked.connect(self._enter_plate_test)
        self.plate_back_button.clicked.connect(self._leave_plate_test)

        self.progress_timer = QTimer(self.window)
        self.progress_timer.setInterval(250)
        self.progress_timer.timeout.connect(self._update_demo_progress)

        self.start_button.clicked.connect(self._start_demo_cycle)
        self.pause_button.clicked.connect(self._pause_demo_cycle)
        self.reset_button.clicked.connect(self._reset_demo_cycle)
        self.pause_button.setEnabled(False)

    def _connect_maintenance_page(self) -> None:
        self.plate_board_widget.plateClicked.connect(self._request_plate_test)
        self.plate_roarm_standby_button.clicked.connect(self._request_plate_roarm_standby)
        self.auto_test_button.clicked.connect(self._request_auto_test)
        self.stop_button.clicked.connect(lambda: self._request_interrupt("X", "즉시 정지"))
        self.home_button.clicked.connect(lambda: self._request_interrupt("H", "홈 복귀"))
        self.reconnect_button.clicked.connect(self.connect_arduino_async)
        self.rearm_button.clicked.connect(self._rearm_controller)

    def _connect_signals(self) -> None:
        self.signals.log.connect(self._append_log)
        self.signals.connection_finished.connect(self._on_connection_finished)
        self.signals.action_finished.connect(self._on_action_finished)

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

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.args.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.args.height)
        cap.set(cv2.CAP_PROP_FPS, self.args.fps)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.cap = cap
        self.connection_label.setText("카메라 연결됨")
        self.camera_live_badge.setText("● LIVE")

        self.camera_timer = QTimer(self.window)
        self.camera_timer.setInterval(max(1, round(1000 / max(self.args.fps, 1))))
        self.camera_timer.timeout.connect(self._update_camera)
        self.camera_timer.start()

    def _setup_shortcuts(self) -> None:
        esc = QShortcut(QKeySequence("Esc"), self.window)
        esc.activated.connect(self._handle_escape)
        quit_shortcut = QShortcut(QKeySequence("Ctrl+Q"), self.window)
        quit_shortcut.activated.connect(self.window.close)
        self._shortcuts = [esc, quit_shortcut]

    def _setup_timers(self) -> None:
        self.status_timer = QTimer(self.window)
        self.status_timer.setInterval(250)
        self.status_timer.timeout.connect(self._refresh_controller_status)
        self.status_timer.start()

    def _handle_escape(self) -> None:
        index = self.page_stack.currentIndex()
        if index == 2:
            self._leave_plate_test()
        elif index == 1:
            self.page_stack.setCurrentIndex(0)
        else:
            self.window.close()

    def _enter_plate_test(self) -> None:
        self.page_stack.setCurrentIndex(2)

    def _leave_plate_test(self) -> None:
        if self.controller is not None and bool(self.controller.fold_in_progress):
            QMessageBox.warning(
                self.window,
                "동작 진행 중",
                "폴딩판 작동이 진행 중입니다. 먼저 '즉시 정지'를 눌러주세요.",
            )
            return
        self.page_stack.setCurrentIndex(1)

    def _update_camera(self) -> None:
        if self.cap is None or not self.cap.isOpened():
            return
        ok, frame = self.cap.read()
        if not ok or frame is None:
            self.camera_label.setText("카메라 프레임 수신 실패")
            self.camera_live_badge.setText("● OFF")
            return

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, channels = rgb.shape
        image = QImage(
            rgb.data,
            w,
            h,
            channels * w,
            QImage.Format.Format_RGB888,
        ).copy()
        pixmap = QPixmap.fromImage(image).scaled(
            self.camera_label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.camera_label.setPixmap(pixmap)

    def connect_arduino_async(self) -> None:
        if self._busy:
            self._append_log("다른 테스트가 진행 중이어서 연결 재시도를 보류함")
            return
        if ArduinoPlateController is None:
            self._append_log(f"runtime import 실패: {RUNTIME_IMPORT_ERROR}")
            self.signals.connection_finished.emit(False, "runtime import 실패")
            return

        if self.controller is not None:
            try:
                self.controller.close()
            except Exception:
                pass

        self.arduino_status_badge.setText("Arduino 연결 중")
        self._append_log(
            f"Arduino 연결 시도: {self.args.arduino_port} @ {self.args.arduino_baud}"
        )

        self.controller = ArduinoPlateController(
            port=self.args.arduino_port,
            baudrate=int(self.args.arduino_baud),
            ready_timeout_s=float(self.args.arduino_ready_timeout_s),
            done_timeout_s=float(self.args.arduino_done_timeout_s),
            send_enabled=bool(self.args.arduino_send),
            allow_during_dry_run=True,
            auto_rearm_on_nonfinish=False,
            rearm_cooldown_s=0.0,
        )

        def worker() -> None:
            assert self.controller is not None
            ok = bool(self.controller.connect())
            detail = self.controller.status_text()
            self.signals.connection_finished.emit(ok, detail)

        threading.Thread(target=worker, name="ui-arduino-connect", daemon=True).start()

    def _on_connection_finished(self, ok: bool, detail: str) -> None:
        self._append_log(("연결 성공: " if ok else "연결 실패: ") + detail)
        self._refresh_controller_status()

    def _controller_ready_for_motion(self) -> tuple[bool, str]:
        if self.controller is None:
            return False, "Arduino controller가 생성되지 않음"
        if not self.controller.connected:
            return False, "Arduino가 연결되지 않음"
        if not self.controller.ready:
            return False, "Arduino READY 수신 전"
        if not bool(getattr(self.controller, "protocol_ok", False)):
            return False, (
                "Arduino 펌웨어가 e4/e2 motion V7와 호환되지 않습니다. "
                "servomotor_control_e4_e2_motion_v7.ino를 업로드하세요. "
                f"감지={getattr(self.controller, 'firmware_id', '') or '구형/알 수 없음'}"
            )
        if not self.controller.send_enabled:
            return False, "실제 전송이 꺼져 있음. --arduino-send로 실행 필요"
        if self._busy:
            return False, "다른 테스트가 진행 중"
        return True, "READY"

    def _refresh_controller_status(self) -> None:
        controller = self.controller
        if controller is None:
            text = "Arduino 없음"
            enabled = False
        elif controller.fold_in_progress:
            text = "폴딩판 작동 진행 중"
            enabled = False
        elif controller.connected and controller.ready and not bool(getattr(controller, "protocol_ok", False)):
            text = "펌웨어 불일치 · e4/e2 V7 필요"
            enabled = False
        elif controller.connected and controller.ready and controller.send_enabled:
            text = "Arduino e4/e2 V7 READY · 실제 전송 ON"
            enabled = not self._busy
        elif controller.connected and controller.ready:
            text = "Arduino e4/e2 V7 READY · 실제 전송 OFF"
            enabled = False
        elif controller.connected:
            text = "Arduino 연결됨 · READY 대기"
            enabled = False
        else:
            text = "Arduino 연결 실패"
            enabled = False

        self.arduino_status_badge.setText(text)
        self._apply_plate_motion_gate(enabled)
        self.home_button.setEnabled(controller is not None and controller.connected and controller.ready and controller.send_enabled)
        self.stop_button.setEnabled(controller is not None and controller.connected and controller.ready and controller.send_enabled)
        self.rearm_button.setEnabled(controller is not None and not controller.fold_in_progress)

        if controller is not None and controller.last_line:
            marker = str(controller.last_line)
            if marker != self._last_rx_seen:
                self._last_rx_seen = marker
                self._append_log(f"RX: {marker}")
                self._handle_plate_runtime_line(marker)

    def _apply_plate_motion_gate(self, controller_enabled: bool) -> None:
        """Subclass hook for additional robot/fixture safety gates."""
        enabled = bool(controller_enabled)
        self.plate_board_widget.set_motion_enabled(enabled)
        self.auto_test_button.setEnabled(enabled)
        self.plate_roarm_standby_button.setEnabled(enabled)

    def _request_plate_roarm_standby(self) -> None:
        QMessageBox.information(
            self.window,
            "지원되지 않음",
            "현재 UI 구성에는 로봇팔 대기 위치 이동 기능이 연결되어 있지 않습니다.",
        )

    def _handle_plate_runtime_line(self, line: str) -> None:
        """Subclass hook for stage-aware board animation."""
        _ = line

    def _confirm_motion(self, title: str, body: str) -> bool:
        if self.args.no_confirm:
            return True
        answer = QMessageBox.question(
            self.window,
            title,
            body + "\n\n보드 위에 사람의 손과 물체가 없는지 다시 확인하세요.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        return answer == QMessageBox.StandardButton.Yes

    def _request_plate_test(self, plate: int) -> None:
        ready, reason = self._controller_ready_for_motion()
        if not ready:
            QMessageBox.warning(self.window, "실행 불가", reason)
            return
        if plate not in (1, 2, 3, 4, 5):
            QMessageBox.warning(self.window, "지원하지 않는 판", f"현재 runtime은 {plate}번 명령을 지원하지 않습니다.")
            return
        self.plate_board_widget.set_selected_plate(plate)
        self.selected_plate_label.setText(f"선택된 판: {plate}번 · 실행 중")

        def task() -> str:
            assert self.controller is not None
            ok = bool(self.controller.send_manual(str(plate), dry_run=False))
            time.sleep(0.45)
            return "명령 전송 완료" if ok else "명령 전송 실패"

        self._run_action_async(f"{plate}번 판", task)

    def _request_auto_test(self) -> None:
        ready, reason = self._controller_ready_for_motion()
        if not ready:
            QMessageBox.warning(self.window, "실행 불가", reason)
            return
        if not self._confirm_motion(
            "폴딩판 작동",
            "G 명령으로 합의된 7단계 동기화 시나리오를 실행하시겠습니까?",
        ):
            return

        self.plate_board_widget.set_selected_plate(None)
        self.selected_plate_label.setText("폴딩판 작동: 동기화 시나리오 실행 중")

        def task() -> str:
            assert self.controller is not None
            self.controller.rearm("UI maintenance auto test", force=True)
            return str(
                self.controller.trigger_fold_and_wait(
                    source="ui-folding-board-operation",
                    dry_run=False,
                    command="G",
                )
            )

        self._run_action_async("폴딩판 작동 G", task)

    def _request_interrupt(self, command: str, label: str) -> None:
        controller = self.controller
        if controller is None or not controller.connected or not controller.ready:
            QMessageBox.warning(self.window, "실행 불가", "Arduino가 READY 상태가 아닙니다.")
            return
        if not controller.send_enabled:
            QMessageBox.warning(self.window, "실행 불가", "--arduino-send 옵션이 필요합니다.")
            return

        def task() -> str:
            ok = bool(controller.send_manual(command, dry_run=False))
            time.sleep(0.25)
            return "명령 전송 완료" if ok else "명령 전송 실패"

        # Interrupt commands remain available even while the auto test is running.
        threading.Thread(
            target=lambda: self.signals.action_finished.emit(
                label,
                "완료" in (result := task()),
                result,
            ),
            name=f"ui-plate-{command.lower()}",
            daemon=True,
        ).start()
        self._append_log(f"{label} 명령 전송 요청")

    def _rearm_controller(self) -> None:
        if self.controller is None:
            self._append_log("잠금 초기화 실패: controller 없음")
            return
        ok = bool(self.controller.rearm("UI maintenance reset", force=True))
        self._append_log("작동 잠금 초기화 완료" if ok else "작동 잠금 초기화 실패")

    def _run_action_async(self, label: str, task: Callable[[], str]) -> None:
        if self._busy:
            self._append_log(f"{label} 차단: 다른 테스트가 진행 중")
            return
        self._busy = True
        self._append_log(f"{label} 시작")
        self._refresh_controller_status()

        def worker() -> None:
            try:
                result = str(task())
                success = result in {"DONE", "명령 전송 완료", "대기 위치 확인 완료"} or result.startswith("OK:")
            except Exception as exc:
                result = repr(exc)
                success = False
            self.signals.action_finished.emit(label, success, result)

        threading.Thread(target=worker, name=f"ui-action-{label}", daemon=True).start()

    def _on_action_finished(self, label: str, success: bool, result: str) -> None:
        if label not in {"즉시 정지", "홈 복귀"}:
            self._busy = False
        self.plate_board_widget.set_selected_plate(None)
        self.selected_plate_label.setText("선택된 판: 없음")
        self._append_log(f"{label} 종료: {'성공' if success else '확인 필요'} · {result}")
        self._refresh_controller_status()

    def _append_log(self, message: str) -> None:
        timestamp = time.strftime("%H:%M:%S")
        self.log_view.appendPlainText(f"[{timestamp}] {message}")
        scrollbar = self.log_view.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def _update_demo_progress(self) -> None:
        value = min(self.progress_bar.value() + 1, 100)
        self.progress_bar.setValue(value)
        self.progress_text.setText(f"{value}%")
        if value < 20:
            self.step_label.setText("1 / 5 · 의류 인식 중")
        elif value < 45:
            self.step_label.setText("2 / 5 · 옷 방향 정렬 중")
        elif value < 70:
            self.step_label.setText("3 / 5 · 주름 펴기 진행 중")
        elif value < 95:
            self.step_label.setText("4 / 5 · 접기 동작 진행 중")
        else:
            self.step_label.setText("5 / 5 · 마무리 중")
        if value >= 100:
            self.progress_timer.stop()
            self.state_badge.setText("완료")
            self.start_button.setEnabled(True)
            self.pause_button.setEnabled(False)

    def _start_demo_cycle(self) -> None:
        if self.progress_bar.value() >= 100:
            self.progress_bar.setValue(0)
        self.state_badge.setText("작동 중")
        self.start_button.setEnabled(False)
        self.pause_button.setEnabled(True)
        self.progress_timer.start()

    def _pause_demo_cycle(self) -> None:
        if self.progress_timer.isActive():
            self.progress_timer.stop()
            self.state_badge.setText("일시정지")
            self.pause_button.setText("▶  계속하기")
        else:
            self.state_badge.setText("작동 중")
            self.pause_button.setText("Ⅱ  일시정지")
            self.progress_timer.start()

    def _reset_demo_cycle(self) -> None:
        self.progress_timer.stop()
        self.progress_bar.setValue(0)
        self.progress_text.setText("0%")
        self.state_badge.setText("대기 중")
        self.step_label.setText("0 / 5 · 시작 대기")
        self.start_button.setEnabled(True)
        self.pause_button.setEnabled(False)
        self.pause_button.setText("Ⅱ  일시정지")

    def show(self) -> None:
        if self.args.windowed:
            self.window.show()
        else:
            self.window.showFullScreen()

    def cleanup(self) -> None:
        if self._closing:
            return
        self._closing = True
        if hasattr(self, "camera_timer"):
            self.camera_timer.stop()
        if self.cap is not None:
            self.cap.release()
        if self.controller is not None:
            try:
                self.controller.close()
            except Exception:
                pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Folding-machine PySide6 UI with folding-board maintenance test"
    )
    parser.add_argument("--camera", default="/dev/video0")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--windowed", action="store_true")

    parser.add_argument("--arduino-enable", dest="arduino_enable", action="store_true", default=True)
    parser.add_argument("--no-arduino-enable", dest="arduino_enable", action="store_false")
    parser.add_argument("--arduino-send", action="store_true", default=False)
    parser.add_argument("--arduino-port", default="auto")
    parser.add_argument("--arduino-baud", type=int, default=9600)
    parser.add_argument("--arduino-ready-timeout-s", type=float, default=20.0)
    parser.add_argument("--arduino-done-timeout-s", type=float, default=60.0)
    parser.add_argument("--no-confirm", action="store_true", default=False)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    app.setFont(QFont("Noto Sans CJK KR", 12))

    if QSS_PATH.exists():
        app.setStyleSheet(QSS_PATH.read_text(encoding="utf-8"))

    controller = FoldingBoardUiController(args, app)
    app.aboutToQuit.connect(controller.cleanup)
    controller.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
