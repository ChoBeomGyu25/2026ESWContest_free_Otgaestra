#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Thin launcher for the V22 + upper-V51 + lower integrated workflow.

Dependency boundary:
  main.py -> V22_main/integration_runtime.py
          -> garment_transfer.py
          -> upper_process.py
          -> upper_vla_process.py
          -> bottom_process.py
               -> bottom/bottom_vla_runtime.py

V22 remains the sole physical camera owner. Lower integration is headless and uses
the shared frame provider plus shared EmptyBoardState.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

DUAL_DIR = Path(__file__).resolve().parent
V22_DIR = DUAL_DIR / "V22_main"
RUNTIME_FILE = V22_DIR / "integration_runtime.py"
GARMENT_TRANSFER_FILE = DUAL_DIR / "garment_transfer.py"
UPPER_PROCESS_FILE = DUAL_DIR / "upper_process.py"
UPPER_VLA_PROCESS_FILE = DUAL_DIR / "upper_vla_process.py"
BOTTOM_PROCESS_FILE = DUAL_DIR / "bottom_process.py"


def _load_runtime():
    if not RUNTIME_FILE.is_file():
        raise FileNotFoundError(f"integration runtime not found: {RUNTIME_FILE}")
    spec = importlib.util.spec_from_file_location(
        "otgaestra_v22_clean_integration_runtime", str(RUNTIME_FILE)
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import integration runtime: {RUNTIME_FILE}")
    module = importlib.util.module_from_spec(spec)

    # Python 3.10 dataclasses resolves annotations through sys.modules while
    # the module body is being executed. Register the dynamically loaded
    # runtime before exec_module(), exactly as the normal import machinery does.
    previous = sys.modules.get(spec.name)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        if previous is None:
            sys.modules.pop(spec.name, None)
        else:
            sys.modules[spec.name] = previous
        raise

    if not hasattr(module, "main"):
        raise RuntimeError(f"integration runtime has no main(): {RUNTIME_FILE}")
    return module


def main() -> int:
    runtime = _load_runtime()
    module_args = [
        "--garment-transfer-module", str(GARMENT_TRANSFER_FILE),
        "--upper-process-module", str(UPPER_PROCESS_FILE),
        "--upper-vla-process-module", str(UPPER_VLA_PROCESS_FILE),
        "--bottom-process-module", str(BOTTOM_PROCESS_FILE),
    ]
    old_argv = sys.argv[:]
    try:
        sys.argv = [str(RUNTIME_FILE)] + module_args + old_argv[1:]
        return int(runtime.main())
    finally:
        sys.argv = old_argv


if __name__ == "__main__":
    raise SystemExit(main())
