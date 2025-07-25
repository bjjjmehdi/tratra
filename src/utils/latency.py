"""Latency guard."""
import asyncio
import time
from typing import Final

from utils.config import load_config

cfg = load_config()
_MAX_MS: Final = cfg["risk"]["max_latency_ms"]

class LatencyGuard:
    def __init__(self, max_ms: int) -> None:
        self.max_ms = max_ms

    def too_slow(self) -> bool:
        return time.perf_counter_ns() // 1_000_000 > self.max_ms