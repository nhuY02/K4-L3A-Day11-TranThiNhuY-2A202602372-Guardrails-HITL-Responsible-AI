"""
Assignment 11 — Rate Limiter implementation.

Sliding-window, per-user rate limiting. Blocks abuse that other
guardrail layers do not address (flooding / cost attacks).
"""
from __future__ import annotations

from collections import defaultdict, deque
import time

from google.adk.plugins import base_plugin
from google.genai import types


class RateLimitPlugin(base_plugin.BasePlugin):
    """Block users who exceed max_requests within window_seconds."""

    def __init__(self, max_requests: int = 10, window_seconds: int = 60):
        super().__init__(name="rate_limiter")
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.user_windows: dict[str, deque] = defaultdict(deque)
        self.blocked_count = 0
        self.total_count = 0

    def _block_response(self, message: str) -> types.Content:
        return types.Content(
            role="model",
            parts=[types.Part.from_text(text=message)],
        )

    async def on_user_message_callback(self, *, invocation_context, user_message):
        """Return Content to block, or None to allow."""
        self.total_count += 1
        user_id = getattr(invocation_context, "user_id", None) or "anonymous"
        now = time.time()
        window = self.user_windows[user_id]

        # 1. Loại bỏ các mốc thời gian ngoài cửa sổ trượt (now - window_seconds)
        cutoff = now - self.window_seconds
        while window and window[0] < cutoff:
            window.popleft()

        # 2. Kiểm tra nếu số request trong cửa sổ đạt hoặc vượt max_requests
        if len(window) >= self.max_requests:
            wait = self.window_seconds - (now - window[0])
            if wait < 0:
                wait = 0.0
            self.blocked_count += 1
            return self._block_response(
                f"Rate limit exceeded. Try again in {wait:.0f}s."
            )

        # 3. Nếu chưa vượt ngưỡng, ghi nhận timestamp hiện tại và cho qua (None)
        window.append(now)
        return None


# ============================================================
# Quick tests
# ============================================================

async def test_rate_limiter():
    """Test RateLimitPlugin with burst requests."""
    plugin = RateLimitPlugin(max_requests=3, window_seconds=5)
    
    class MockContext:
        def __init__(self, user_id):
            self.user_id = user_id

    ctx = MockContext(user_id="user_test_1")
    dummy_msg = types.Content(
        role="user", parts=[types.Part.from_text(text="Check balance")]
    )

    print("Testing RateLimitPlugin (limit: 3 req / 5s):")
    for i in range(1, 6):
        res = await plugin.on_user_message_callback(
            invocation_context=ctx, user_message=dummy_msg
        )
        status = "BLOCK" if res else "ALLOW"
        print(f"  Request #{i}: [{status}]")
        if res and res.parts:
            print(f"            -> {res.parts[0].text}")

    print(f"\nStats: {plugin.blocked_count} blocked / {plugin.total_count} total")


if __name__ == "__main__":
    import asyncio
    asyncio.run(test_rate_limiter())
