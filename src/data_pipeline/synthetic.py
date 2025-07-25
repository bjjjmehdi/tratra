"""
Generate adversarial price charts + LOB tensors for tail-scenario training.
"""
import io
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

from data_ingestion.candle_builder import CandleBuilder
from utils.config import load_config
from utils.logger import get_logger

cfg = load_config()
log = get_logger("SYNTHETIC")
OUT = Path(cfg["dataset"]["raw_dir"]) / "synthetic.jsonl"
OUT.parent.mkdir(parents=True, exist_ok=True)

class SyntheticEngine:
    def __init__(self, n_scenarios: int = 10_000):
        self.n = n_scenarios

    def _make_tail_chart(self) -> bytes:
        # GBM with random σ and fat-tail shocks
        mu, sigma = 0, random.uniform(0.005, 0.05)
        steps = 60
        prices = [100]
        for _ in range(steps - 1):
            shock = random.choice([1, -1]) * random.expovariate(1 / 3) * 0.01
            prices.append(prices[-1] * np.exp(mu + sigma * np.random.normal() + shock))
        df = pd.DataFrame({"open": prices, "high": prices, "low": prices, "close": prices})
        builder = CandleBuilder()
        return builder.render_png(df)

    def generate(self):
        for idx in range(self.n):
            png = self._make_tail_chart()
            label = random.choice(["BUY", "SELL", "HOLD"])
            with OUT.open("a") as f:
                f.write(
                    json.dumps(
                        {
                            "png_b64": png.hex(),
                            "action": label,
                            "reward": random.uniform(-0.02, 0.02),
                            "contract": "SYNTH",
                            "metadata": {"synthetic": True},
                        }
                    )
                    + "\n"
                )
        log.info("Synthetic dataset ready: %s rows", self.n)

if __name__ == "__main__":
    import json
    SyntheticEngine().generate()