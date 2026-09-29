"""
Assignment 11 — Audit Log implementation.

Records every interaction for forensics. Never blocks by itself —
other layers catch attacks; this layer makes them reviewable.
"""
from __future__ import annotations

import json
import time
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
        # Lưu các request đang xử lý: key -> {user_id, input_text, start_time, timestamp}
        self._open: dict[str, dict] = {}

    def _resolve_key(self, user_id: str, request_id: str | None) -> str:
        return request_id or user_id

    def record_input(self, *, user_id: str, text: str, request_id: str | None = None) -> str:
        """Store input + start timestamp keyed by request_id/user_id."""
        key = self._resolve_key(user_id, request_id)
        now_ts = time.time()
        iso_str = utc_now_iso()
        self._open[key] = {
            "request_id": request_id or key,
            "user_id": user_id,
            "input_text": text,
            "start_time": now_ts,
            "timestamp": iso_str,
        }
        return key

    def record_output(
        self,
        *,
        user_id: str,
        text: str,
        blocked: bool = False,
        layer: str | None = None,
        request_id: str | None = None,
    ) -> dict:
        """Store output, layer decision, latency; append to self.logs."""
        key = self._resolve_key(user_id, request_id)
        open_data = self._open.pop(key, None)

        now_ts = time.time()
        if open_data:
            start_ts = open_data.get("start_time", now_ts)
            latency_ms = max(0.0, (now_ts - start_ts) * 1000)
            input_text = open_data.get("input_text", "")
            req_id = open_data.get("request_id", request_id or key)
            created_at = open_data.get("timestamp", utc_now_iso())
        else:
            latency_ms = 0.0
            input_text = ""
            req_id = request_id or key
            created_at = utc_now_iso()

        entry = {
            "request_id": req_id,
            "user_id": user_id,
            "timestamp": created_at,
            "input": input_text,
            "output": text,
            "blocked": blocked,
            "layer": layer,
            "latency_ms": round(latency_ms, 2),
        }
        self.logs.append(entry)
        return entry

    def export_json(self, filepath: str | None = None) -> str:
        """Write logs to disk (JSON array) under repo-root ``outputs/`` by default."""
        target_path = Path(filepath) if filepath else Path(default_audit_log_path())
        target_path.parent.mkdir(parents=True, exist_ok=True)
        target_path.write_text(
            json.dumps(self.logs, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        return str(target_path)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# ============================================================
# Quick tests
# ============================================================

def test_audit_log():
    """Test recording and exporting audit logs."""
    import tempfile

    logger = AuditLogPlugin()
    req_id = "req_101"
    logger.record_input(user_id="user_1", text="What is savings APY?", request_id=req_id)
    time.sleep(0.05)
    entry = logger.record_output(
        user_id="user_1",
        text="Savings APY is 4.25%",
        blocked=False,
        layer=None,
        request_id=req_id,
    )

    assert entry["blocked"] is False
    assert entry["latency_ms"] > 0
    assert len(logger.logs) == 1

    with tempfile.TemporaryDirectory() as tmpdir:
        out_file = Path(tmpdir) / "audit_test.json"
        saved = logger.export_json(str(out_file))
        assert Path(saved).exists()
        loaded = json.loads(Path(saved).read_text(encoding="utf-8"))
        assert len(loaded) == 1
        assert loaded[0]["input"] == "What is savings APY?"
        print("AuditLogPlugin test: PASS")


if __name__ == "__main__":
    test_audit_log()
