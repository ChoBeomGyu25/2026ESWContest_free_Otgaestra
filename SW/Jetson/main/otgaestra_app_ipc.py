#!/usr/bin/env python3

import json
import os
import time
import uuid
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent / ".otgaestra_ipc"
COMMAND_PATH = BASE_DIR / "command.json"
ACK_PATH = BASE_DIR / "ack.json"
EVENT_PATH = BASE_DIR / "events.jsonl"


def ensure_ipc_dir():
    BASE_DIR.mkdir(parents=True, exist_ok=True)


def _read_json(path):
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _atomic_write(path, payload):
    ensure_ipc_dir()
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(
        json.dumps(payload, ensure_ascii=False),
        encoding="utf-8",
    )
    tmp.replace(path)


def write_command(command, job_id, target=1):
    payload = {
        "nonce": str(uuid.uuid4()),
        "command": str(command),
        "jobId": str(job_id),
        "target": int(target),
        "issuedAt": time.time(),
    }
    _atomic_write(COMMAND_PATH, payload)
    return payload


def consume_command():
    payload = _read_json(COMMAND_PATH)

    nonce = str(payload.get("nonce") or "").strip()
    command = str(payload.get("command") or "").strip()

    if not nonce or not command:
        return None

    ack = _read_json(ACK_PATH)

    if str(ack.get("nonce") or "") == nonce:
        return None

    _atomic_write(
        ACK_PATH,
        {
            "nonce": nonce,
            "command": command,
            "jobId": payload.get("jobId"),
            "consumedAt": time.time(),
        },
    )

    return payload


def emit_event(event_type, job_id, **fields):
    ensure_ipc_dir()

    payload = {
        "type": str(event_type),
        "jobId": str(job_id),
        "time": time.time(),
    }
    payload.update(fields)

    with EVENT_PATH.open("a", encoding="utf-8") as fp:
        fp.write(json.dumps(payload, ensure_ascii=False) + "\n")
        fp.flush()
        os.fsync(fp.fileno())

    return payload


ensure_ipc_dir()
