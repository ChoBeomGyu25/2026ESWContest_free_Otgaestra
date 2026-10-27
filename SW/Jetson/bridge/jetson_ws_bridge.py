#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Otgaestra LAN bridge: Flutter -> already-running Jetson robot main.py via shared IPC.

DEMO is strictly manual: no clock can fake a garment or a finished job.
PHYSICAL mode requires ROBOT_COMMAND_JSON pointing at an ALREADY VALIDATED
main/launcher that emits OTGAESTRA_EVENT lines only after real motion.

THIS IS NOT A SAFETY PLC OR AN EMERGENCY-STOP SYSTEM. The installed physical
E-stop, joint/torque safety, and movement guards must remain independent.
Do not expose the plain ws:// server to the public internet.

Run: ROBOT_MODE=demo ROBOT_TOKEN='lab-secret' \
  python3 -m uvicorn jetson_ws_bridge:app --host 0.0.0.0 --port 8000
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timezone
import hmac
import json
import os
from pathlib import Path
import time
import uuid
from typing import Any

from fastapi import FastAPI, Header, HTTPException, Request, WebSocket, WebSocketDisconnect

APP = FastAPI(title="Otgaestra LAN Job Bridge")
app = APP
MODE = os.environ.get("ROBOT_MODE", "demo").lower().strip()
TOKEN = os.environ.get("ROBOT_TOKEN", "")
IPC_DIR = Path(
    os.environ.get(
        "ROBOT_IPC_DIR",
        "/home/deca/project_train/aruco_test/main/v51_latency_test/.otgaestra_ipc",
    )
).resolve()

COMMAND_PATH = IPC_DIR / "command.json"
ACK_PATH = IPC_DIR / "ack.json"
EVENT_PATH = IPC_DIR / "events.jsonl"

DATA_PATH = Path(
    os.environ.get(
        "ROBOT_BRIDGE_STATE",
        str(Path.home() / ".otgaestra_bridge_state.json"),
    )
)

if MODE not in {"demo", "physical"}:
    raise RuntimeError("ROBOT_MODE must be demo or physical")

if not TOKEN:
    raise RuntimeError(
        "Set a nonempty ROBOT_TOKEN; use a long random token for physical mode"
    )

if MODE == "physical" and len(TOKEN) < 12:
    raise RuntimeError(
        "Physical mode requires a >=12 character ROBOT_TOKEN"
    )

IPC_DIR.mkdir(parents=True, exist_ok=True)

class JobState:
    def __init__(self) -> None:
        self.lock = asyncio.Lock()
        self.clients: set[WebSocket] = set()
        self.job_state = "IDLE"
        self.message = ""
        self.active_job_id: str | None = None
        self.target = 1
        self.completed = 0
        self.started: float | None = None
        self.last_completed_job: dict[str, Any] | None = None
        self.proc: asyncio.subprocess.Process | None = None
        self.worker: asyncio.Task | None = None
        self.stop_file: Path | None = None
        self.persisted_active = False
        self._load()

    def _load(self) -> None:
        try:
            saved = json.loads(DATA_PATH.read_text(encoding="utf-8"))
            last = saved.get("lastCompletedJob")
            if isinstance(last, dict):
                self.last_completed_job = last
            if saved.get("activeJobId") is not None:
                # A server reboot does NOT establish what the arms did meanwhile.
                self.job_state = "ERROR"
                self.message = "통신 서버가 작업 도중 재시작됨: 현장에서 로봇 정지와 카메라 상태를 확인하세요."
                self.persisted_active = True
        except FileNotFoundError:
            pass
        except Exception as exc:
            self.job_state = "ERROR"
            self.message = f"저장된 작업 상태를 읽을 수 없음: {exc}"
            self.persisted_active = True

    def persist(self) -> None:
        data = {
            "lastCompletedJob": self.last_completed_job,
            "activeJobId": self.active_job_id,
        }
        DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
        tmp = DATA_PATH.with_name(DATA_PATH.name + ".tmp")
        tmp.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        tmp.replace(DATA_PATH)

    def snapshot(self) -> dict[str, Any]:
        return {
            "type": "STATUS_UPDATE",
            "source": "DEMO_ONLY" if MODE == "demo" else "PHYSICAL",
            "jobState": self.job_state,
            "message": self.message,
            "activeJobId": self.active_job_id,
            "totalTarget": self.target,
            "currentCount": self.completed,
            "lastCompletedJob": self.last_completed_job,
        }


state = JobState()


async def broadcast(payload: dict[str, Any]) -> None:
    stale: list[WebSocket] = []
    for client in tuple(state.clients):
        try:
            await asyncio.wait_for(client.send_json(payload), timeout=1.0)
        except Exception:
            stale.append(client)
    for client in stale:
        state.clients.discard(client)


async def reject(ws: WebSocket, reason: str) -> None:
    await ws.send_json({"type": "COMMAND_REJECTED", "message": reason})


async def begin_job(target: int) -> tuple[bool, str]:
    async with state.lock:
        if state.job_state != "IDLE":
            return False, "이전 작업이 종료되지 않았거나 현장 확인이 필요합니다."

        state.job_state = "RUNNING"
        state.message = "작업 명령 수신"
        state.target = target
        state.completed = 0
        state.active_job_id = str(uuid.uuid4())
        state.started = time.monotonic()
        state.stop_file = None

        state.persist()
        await broadcast(state.snapshot())

        if MODE == "physical":
            state.worker = asyncio.create_task(
                run_physical_job(state.active_job_id)
            )

        return True, ""

async def cloth_done(*, job_id: str | None = None) -> tuple[bool, str]:
    async with state.lock:
        if state.job_state != "RUNNING":
            return False, "진행 중인 작업이 없습니다."
        if job_id is not None and state.active_job_id != job_id:
            return False, "다른 작업 ID의 완료 신호를 거부했습니다."
        if state.completed >= state.target:
            return False, "이미 목표 수량만큼 완료했습니다."
        state.completed += 1
        await broadcast(state.snapshot())
        return True, ""


async def job_done(*, job_id: str | None = None) -> tuple[bool, str]:
    async with state.lock:
        if state.job_state != "RUNNING":
            return False, "완료 신호를 받을 진행 작업이 없습니다."
        if job_id is not None and job_id != state.active_job_id:
            return False, "다른 작업의 완료 신호입니다."
        if state.completed != state.target:
            return False, "모든 옷의 물리적 접기/투하 완료 신호가 필요합니다."
        duration = max(0, round(time.monotonic() - (state.started or time.monotonic())))
        completed = {
            "type": "JOB_COMPLETED",
            "jobId": state.active_job_id,
            "completedCount": state.completed,
            "durationSeconds": duration,
            "completedAt": datetime.now(timezone.utc).isoformat(),
        }
        state.last_completed_job = completed
        state.active_job_id = None
        state.job_state = "IDLE"
        state.message = "작업 완료"
        state.persist()  # Persist before notifying clients; resilient to brief Wi-Fi outages.
        await broadcast(completed)
        await broadcast(state.snapshot())
        return True, ""


async def reported_error(message: str) -> None:
    async with state.lock:
        state.job_state = "ERROR"
        state.message = message
        # Keep activeJobId persisted: crash recovery needs an onsite decision.
        state.persist()
        await broadcast({"type": "ERROR", "message": message})
        await broadcast(state.snapshot())


async def stop_job() -> tuple[bool, str]:
    async with state.lock:
        if state.job_state not in {"RUNNING", "STOP_REQUESTED"}:
            return False, "중단할 작업이 없습니다."

        if MODE == "demo":
            state.active_job_id = None
            state.job_state = "IDLE"
            state.message = "시험 작업 취소"
            state.persist()

        else:
            if not state.active_job_id:
                return False, "활성 작업 ID가 없습니다."

            state.job_state = "STOP_REQUESTED"
            state.message = (
                "중지 요청 전달됨: 실제 정지 확인 전까지 "
                "로봇이 움직일 수 있습니다."
            )

            _write_ipc_command(
                "STOP_JOB",
                job_id=state.active_job_id,
                target=state.target,
            )

        await broadcast(state.snapshot())
        return True, ""

async def stopped_ack(job_id: str) -> None:
    async with state.lock:
        if state.job_state != "STOP_REQUESTED" or state.active_job_id != job_id:
            return
        state.job_state = "IDLE"
        state.active_job_id = None
        state.message = "Jetson이 정상적인 작업 중단을 확인함"
        state.persist()
        await broadcast(state.snapshot())


def _read_ipc_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except Exception:
        return {}


def _write_ipc_command(
    command: str,
    *,
    job_id: str,
    target: int,
) -> str:
    IPC_DIR.mkdir(parents=True, exist_ok=True)

    nonce = str(uuid.uuid4())

    payload = {
        "nonce": nonce,
        "command": str(command),
        "jobId": str(job_id),
        "target": int(target),
        "issuedAt": time.time(),
    }

    tmp = COMMAND_PATH.with_name(
        COMMAND_PATH.name + ".tmp"
    )

    tmp.write_text(
        json.dumps(payload, ensure_ascii=False),
        encoding="utf-8",
    )

    tmp.replace(COMMAND_PATH)

    print(
        f"[BRIDGE-IPC-COMMAND] {payload}",
        flush=True,
    )

    return nonce


def _discard_ipc_command(nonce: str) -> None:
    current = _read_ipc_json(COMMAND_PATH)

    if str(current.get("nonce") or "") != str(nonce):
        return

    try:
        COMMAND_PATH.unlink()
    except FileNotFoundError:
        pass


async def run_physical_job(job_id: str) -> None:
    """
    Communicate with the already-running V22 main.py.

    IMPORTANT:
    This function never launches another main.py process.
    """

    try:
        # Ignore events belonging to previous jobs.
        try:
            event_offset = EVENT_PATH.stat().st_size
        except FileNotFoundError:
            event_offset = 0

        nonce = _write_ipc_command(
            "START_JOB",
            job_id=job_id,
            target=state.target,
        )

        # Wait until the existing V22 process consumes command.json.
        ack_deadline = time.monotonic() + 8.0

        while time.monotonic() < ack_deadline:
            ack = _read_ipc_json(ACK_PATH)

            if (
                str(ack.get("nonce") or "") == nonce
                and str(ack.get("jobId") or "") == job_id
            ):
                print(
                    f"[BRIDGE-IPC-ACK] "
                    f"job={job_id} command=START_JOB",
                    flush=True,
                )
                break

            await asyncio.sleep(0.10)

        else:
            # Do not leave a START command that could execute later.
            _discard_ipc_command(nonce)

            await reported_error(
                "실행 중인 V22 main.py가 START 명령을 "
                "확인하지 않았습니다."
            )
            return

        # Consume only real events generated by the robot runtime.
        while (
            state.active_job_id == job_id
            and state.job_state in {
                "RUNNING",
                "STOP_REQUESTED",
            }
        ):
            await asyncio.sleep(0.15)

            if not EVENT_PATH.exists():
                continue

            try:
                with EVENT_PATH.open(
                    "r",
                    encoding="utf-8",
                ) as fp:
                    fp.seek(event_offset)
                    lines = fp.readlines()
                    event_offset = fp.tell()

            except OSError:
                continue

            for raw in lines:
                raw = raw.strip()

                if not raw:
                    continue

                try:
                    event = json.loads(raw)

                except json.JSONDecodeError as exc:
                    print(
                        f"[BRIDGE-IPC-BAD-EVENT] "
                        f"{exc}: {raw}",
                        flush=True,
                    )
                    continue

                if not isinstance(event, dict):
                    continue

                if str(event.get("jobId") or "") != job_id:
                    continue

                kind = str(
                    event.get("type") or ""
                ).upper()

                print(
                    f"[BRIDGE-IPC-EVENT] "
                    f"job={job_id} "
                    f"type={kind} "
                    f"event={event}",
                    flush=True,
                )

                if kind == "CLOTH_DONE":
                    expected = state.completed + 1

                    if event.get("completed") != expected:
                        print(
                            "[BRIDGE] rejected "
                            "out-of-order CLOTH_DONE",
                            flush=True,
                        )
                        continue

                    await cloth_done(
                        job_id=job_id
                    )

                elif kind == "JOB_DONE":
                    await job_done(
                        job_id=job_id
                    )
                    return

                elif kind == "STOPPED":
                    await stopped_ack(job_id)
                    return

                elif kind == "ERROR":
                    await reported_error(
                        str(
                            event.get("message")
                            or "robot main.py reported error"
                        )
                    )
                    return

                else:
                    print(
                        f"[BRIDGE] ignored IPC event: {kind}",
                        flush=True,
                    )

    except Exception as exc:
        await reported_error(
            f"Jetson IPC error: {exc}"
        )

    finally:
        if state.worker is asyncio.current_task():
            state.worker = None

@APP.get("/health")
async def health() -> dict[str, Any]:
    return {"ok": True, "mode": MODE, "physical_possible": MODE == "physical"}


@APP.websocket("/ws")
async def socket(ws: WebSocket) -> None:
    await ws.accept()
    try:
        auth = await asyncio.wait_for(ws.receive_json(), timeout=7)
        if (not isinstance(auth, dict) or auth.get("status") != "AUTH"
                or not isinstance(auth.get("token"), str)
                or not hmac.compare_digest(auth["token"], TOKEN)):
            await ws.send_json({"type": "AUTH_FAILED"})
            await ws.close(code=4401)
            return
        state.clients.add(ws)
        await ws.send_json({"type": "AUTH_OK"})
        await ws.send_json(state.snapshot())
        while True:
            try:
                data = await ws.receive_json()
            except ValueError:
                await reject(ws, "잘못된 JSON 형식")
                continue
            if not isinstance(data, dict):
                await reject(ws, "객체 JSON만 허용")
                continue
            command = data.get("status")
            print(f"[APP -> JETSON] command={command} data={data}", flush=True)
            if command == "PING":
                await ws.send_json({"type": "PONG"})
            elif command == "GET_STATUS":
                await ws.send_json(state.snapshot())
            elif command == "START_FOLDING":
                target = data.get("total_target")
                if type(target) is not int or not 1 <= target <= 30:
                    await reject(ws, "작업 수량은 1~30이어야 합니다.")
                    continue
                ok, reason = await begin_job(target)
                if not ok:
                    await reject(ws, reason)
            elif command == "STOP_JOB":
                ok, reason = await stop_job()
                if not ok:
                    await reject(ws, reason)
            else:
                await reject(ws, "지원하지 않는 명령입니다.")
    except (WebSocketDisconnect, asyncio.TimeoutError):
        pass
    finally:
        state.clients.discard(ws)


def operator_only(request: Request, token: str | None) -> None:
    if MODE != "demo":
        raise HTTPException(status_code=404, detail="No demo endpoints in physical mode")
    host = request.client.host if request.client is not None else ""
    if host not in {"127.0.0.1", "::1", "testclient"}:
        raise HTTPException(status_code=403, detail="Only Jetson localhost can send demo flags")
    if token is None or not hmac.compare_digest(token, TOKEN):
        raise HTTPException(status_code=401, detail="Invalid token")


@APP.post("/demo/cloth_done")
async def demo_cloth(request: Request, x_robot_token: str | None = Header(default=None)) -> dict[str, Any]:
    operator_only(request, x_robot_token)
    ok, reason = await cloth_done()
    if not ok:
        raise HTTPException(status_code=409, detail=reason)
    return state.snapshot()


@APP.post("/demo/job_done")
async def demo_job(request: Request, x_robot_token: str | None = Header(default=None)) -> dict[str, Any]:
    operator_only(request, x_robot_token)
    ok, reason = await job_done()
    if not ok:
        raise HTTPException(status_code=409, detail=reason)
    return state.last_completed_job or {}


@APP.post("/demo/reset")
async def demo_reset(request: Request, x_robot_token: str | None = Header(default=None)) -> dict[str, Any]:
    operator_only(request, x_robot_token)
    async with state.lock:
        state.job_state = "IDLE"
        state.active_job_id = None
        state.completed = 0
        state.message = "데모 상태 초기화"
        state.persist()
        await broadcast(state.snapshot())
        return state.snapshot()
