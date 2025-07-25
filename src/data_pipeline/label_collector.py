"""Logs (png, action, reward) from live or paper trades for ViT fine-tuning."""
import asyncio
import json
import time
from pathlib import Path
from typing import Dict, Any

import pandas as pd
from ib_insync import Contract, IB   # IB added for type hint

from utils.logger import get_logger
from utils.config import load_config

cfg = load_config()
log = get_logger("LABEL_COLLECTOR")
OUT = Path(cfg["dataset"]["raw_dir"]) / "labels.jsonl"
OUT.parent.mkdir(parents=True, exist_ok=True)


class LabelCollector:
    """
    Async-safe singleton that appends labeled rows:
    {
      "ts": 1712345678.123,
      "png_b64": "<base64>",
      "action": "BUY",
      "reward": 0.0123,
      "contract": "INTC",
      "metadata": {...}
    }
    """
    _lock = asyncio.Lock()

    @staticmethod
    async def log(ib: IB, png: bytes, action: str, contract: Contract, horizon_sec: int = 300) -> None:
        reward = await LabelCollector._compute_reward(ib, contract, horizon_sec)
        row = {
            "ts": time.time(),
            "png_b64": png.hex(),
            "action": action,
            "reward": reward,
            "contract": contract.symbol,
            "metadata": {"horizon_sec": horizon_sec},
        }
        async with LabelCollector._lock:
            with OUT.open("a") as f:
                f.write(json.dumps(row) + "\n")
        log.debug("Label logged: %s reward=%.4f", action, reward)

    @staticmethod
    async def _compute_reward(ib: IB, contract: Contract, horizon_sec: int) -> float:
        """Compute % return horizon_sec after now."""
        px_now = float(ib.reqMktData(contract, "", False, False).last or 0)
        await asyncio.sleep(horizon_sec)
        px_later = float(ib.reqMktData(contract, "", False, False).last or 0)
        return (px_later - px_now) / px_now if px_now else 0.0