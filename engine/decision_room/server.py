"""JSON-RPC 2.0 over stdio, one message per line.

The TypeScript ACP agent spawns this process and calls the methods below.
Long steps stream `progress` notifications so the editor can show what each
specialist is doing. stdout carries protocol messages only; logs go to stderr.
"""

from __future__ import annotations

import json
import sys
import traceback

import numpy as np

from .workspace import Workspace

METHODS = [
    "load", "export", "fit_demand", "backtest", "equilibrium", "recommend", "price_rule", "tradeoff_frontier",
    "whatif", "stress", "red_team", "memo_preview", "write_memo", "context", "brief", "conduct", "save_state", "restore_state", "chart",
]


def _clean(o):
    """Make numpy scalars and arrays JSON-serialisable."""
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, np.ndarray):
        return _clean(o.tolist())
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, float) and not np.isfinite(o):
        return None
    return o


def _send(msg: dict) -> None:
    sys.stdout.write(json.dumps(_clean(msg)) + "\n")
    sys.stdout.flush()


def serve() -> None:
    current = {"id": None}

    def progress(message: str) -> None:
        _send({"jsonrpc": "2.0", "method": "progress", "params": {"id": current["id"], "message": message}})

    ws = Workspace(progress)
    _send({"jsonrpc": "2.0", "method": "ready", "params": {"methods": sorted(METHODS)}})
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except json.JSONDecodeError as exc:
            _send({"jsonrpc": "2.0", "id": None, "error": {"code": -32700, "message": str(exc)}})
            continue
        rid, method, params = req.get("id"), req.get("method"), req.get("params") or {}
        current["id"] = rid
        if method == "shutdown":
            _send({"jsonrpc": "2.0", "id": rid, "result": {}})
            return
        if method not in METHODS:
            _send({"jsonrpc": "2.0", "id": rid, "error": {"code": -32601, "message": f"unknown method {method}"}})
            continue
        try:
            result = getattr(ws, method)(**params)
            _send({"jsonrpc": "2.0", "id": rid, "result": result})
        except Exception as exc:  # report to the agent instead of dying
            traceback.print_exc(file=sys.stderr)
            _send({"jsonrpc": "2.0", "id": rid, "error": {"code": -32000, "message": f"{type(exc).__name__}: {exc}"}})
