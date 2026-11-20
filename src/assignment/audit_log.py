"""
Assignment 11 — Audit Log starter (TODO).

Records every interaction for forensics. Never blocks by itself —
other layers catch attacks; this layer makes them reviewable.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path


def default_audit_log_path() -> str:
    """Always resolve to <repo>/outputs/… (safe when cwd is src/)."""
    repo_root = Path(__file__).resolve().parents[2]
    return str(repo_root / "outputs" / "audit_log.json")


class AuditLogPlugin:
    """Framework-agnostic audit logger (wire into ADK callbacks or your pipeline)."""

    def __init__(self):
        self.name = "audit_log"
        self.logs: list[dict] = []
        self._open: dict[str, dict] = {}

    def record_input(self, *, user_id: str, text: str, request_id: str | None = None):
        """Store input and start timestamp keyed by request ID or user ID."""
        key = request_id or user_id
        self._open[key] = {
            "start": datetime.now(timezone.utc).timestamp(),
            "input": text,
        }

    def record_output(
        self,
        *,
        user_id: str,
        text: str,
        blocked: bool = False,
        layer: str | None = None,
        request_id: str | None = None,
    ):
        """Store output, layer decision, and latency, then append a log entry."""
        key = request_id or user_id
        open_entry = self._open.pop(key, {
            "start": datetime.now(timezone.utc).timestamp(),
            "input": "",
        })
        now = datetime.now(timezone.utc)
        latency_ms = max(0.0, (now.timestamp() - open_entry["start"]) * 1000)
        self.logs.append({
            "timestamp": now.isoformat(),
            "request_id": request_id,
            "user_id": user_id,
            "input": open_entry["input"],
            "output": text,
            "blocked": blocked,
            "layer": layer,
            "latency_ms": round(latency_ms, 3),
        })

    def export_json(self, filepath: str | None = None):
        """Write logs to disk (JSON array) under repo-root ``outputs/`` by default."""
        path = Path(filepath or default_audit_log_path())
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.logs, ensure_ascii=False, indent=2), encoding="utf-8")
        return str(path)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
