"""
Assignment 11 — Monitoring & Alerts implementation.

Tracks block rate, rate-limit hits, judge fail rate.
Fires alerts when thresholds are exceeded.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


def default_metrics_path() -> str:
    """Always resolve to <repo>/outputs/… (safe when cwd is src/)."""
    repo_root = Path(__file__).resolve().parents[2]
    return str(repo_root / "outputs" / "metrics.json")


@dataclass
class Alert:
    metric: str
    value: float
    threshold: float
    message: str


@dataclass
class MonitoringAlert:
    """Aggregate counters from pipeline plugins and emit alerts."""

    block_rate_threshold: float = 0.5
    rate_limit_hit_threshold: int = 5
    judge_fail_rate_threshold: float = 0.3
    alerts: list[Alert] = field(default_factory=list)

    # Counters — update these from your pipeline after each request
    total_requests: int = 0
    blocked_requests: int = 0
    rate_limit_hits: int = 0
    judge_checks: int = 0
    judge_fails: int = 0

    def record_request(
        self,
        *,
        blocked: bool = False,
        is_rate_limit: bool = False,
        judge_checked: bool = False,
        judge_failed: bool = False,
    ):
        """Helper to increment counters conveniently."""
        self.total_requests += 1
        if blocked:
            self.blocked_requests += 1
        if is_rate_limit:
            self.rate_limit_hits += 1
        if judge_checked:
            self.judge_checks += 1
            if judge_failed:
                self.judge_fails += 1

    def check_metrics(self) -> list[Alert]:
        """Compute rates, append Alert objects when thresholds exceeded."""
        current_alerts: list[Alert] = []

        # 1. Kiểm tra ngưỡng Block Rate
        if self.total_requests > 0:
            block_rate = self.blocked_requests / self.total_requests
            if block_rate >= self.block_rate_threshold:
                current_alerts.append(
                    Alert(
                        metric="block_rate",
                        value=round(block_rate, 4),
                        threshold=self.block_rate_threshold,
                        message=f"Block rate {block_rate:.1%} exceeds threshold {self.block_rate_threshold:.1%}",
                    )
                )

        # 2. Kiểm tra ngưỡng Rate Limit Hits
        if self.rate_limit_hits >= self.rate_limit_hit_threshold:
            current_alerts.append(
                Alert(
                    metric="rate_limit_hits",
                    value=float(self.rate_limit_hits),
                    threshold=float(self.rate_limit_hit_threshold),
                    message=f"Rate limit hits ({self.rate_limit_hits}) reached threshold ({self.rate_limit_hit_threshold})",
                )
            )

        # 3. Kiểm tra ngưỡng Judge Fail Rate (nếu có dùng judge)
        if self.judge_checks > 0:
            judge_fail_rate = self.judge_fails / self.judge_checks
            if judge_fail_rate >= self.judge_fail_rate_threshold:
                current_alerts.append(
                    Alert(
                        metric="judge_fail_rate",
                        value=round(judge_fail_rate, 4),
                        threshold=self.judge_fail_rate_threshold,
                        message=f"Judge fail rate {judge_fail_rate:.1%} exceeds threshold {self.judge_fail_rate_threshold:.1%}",
                    )
                )

        self.alerts = current_alerts
        return self.alerts

    def export_json(self, filepath: str | None = None) -> str:
        """Write metrics + alerts to JSON under repo-root ``outputs/`` by default."""
        self.check_metrics()
        target_path = Path(filepath) if filepath else Path(default_metrics_path())
        target_path.parent.mkdir(parents=True, exist_ok=True)
        data = self.snapshot()
        target_path.write_text(
            json.dumps(data, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        return str(target_path)

    def snapshot(self) -> dict:
        block_rate = (
            self.blocked_requests / self.total_requests
            if self.total_requests
            else 0.0
        )
        judge_fail_rate = (
            self.judge_fails / self.judge_checks if self.judge_checks else 0.0
        )
        return {
            "total_requests": self.total_requests,
            "blocked_requests": self.blocked_requests,
            "block_rate": block_rate,
            "rate_limit_hits": self.rate_limit_hits,
            "judge_checks": self.judge_checks,
            "judge_fails": self.judge_fails,
            "judge_fail_rate": judge_fail_rate,
            "alerts": [
                {
                    "metric": a.metric,
                    "value": a.value,
                    "threshold": a.threshold,
                    "message": a.message,
                }
                for a in self.alerts
            ],
        }


# ============================================================
# Quick tests
# ============================================================

def test_monitoring():
    """Test MonitoringAlert calculation and alerting."""
    import tempfile

    mon = MonitoringAlert(block_rate_threshold=0.5, rate_limit_hit_threshold=3)
    
    # 2 requests: 1 safe, 1 blocked -> block_rate = 50% -> trigger alert
    mon.record_request(blocked=False)
    mon.record_request(blocked=True)
    
    # Thêm 3 lần rate limit hits -> trigger rate_limit alert
    mon.rate_limit_hits = 3

    alerts = mon.check_metrics()
    assert len(alerts) == 2
    metrics = {a.metric for a in alerts}
    assert "block_rate" in metrics
    assert "rate_limit_hits" in metrics

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_file = Path(tmpdir) / "metrics_test.json"
        saved = mon.export_json(str(tmp_file))
        assert Path(saved).exists()
        loaded = json.loads(Path(saved).read_text(encoding="utf-8"))
        assert loaded["total_requests"] == 2
        assert loaded["blocked_requests"] == 1
        assert len(loaded["alerts"]) == 2
        print("MonitoringAlert test: PASS")


if __name__ == "__main__":
    test_monitoring()
