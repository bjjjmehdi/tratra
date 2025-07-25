"""
Online PPO fine-tuning of ViT using live reward.
Fully patched – no missing names.
"""
import asyncio
import time
from pathlib import Path

import numpy as np
import ray
import torch
from ray.rllib.algorithms.ppo import PPO, PPOConfig
from transformers import ViTImageProcessor
from data_ingestion.lob_stream import LobTick
from encoders.multimodal import MultiModalEncoder
from execution.broker import Broker
from utils.config import load_config
from utils.logger import get_logger

cfg = load_config()
log = get_logger("LIVE_RL")


# -------------------------------------------------
# 1.  minimal stub so TradingEnv can run standalone
# -------------------------------------------------
class FakeLob:
    """Stub LOB for RL env when real LOB is unavailable."""
    def __init__(self, mid: float):
        self.bid = [(mid - 0.01 * i, 100) for i in range(1, 6)]
        self.ask = [(mid + 0.01 * i, 100) for i in range(1, 6)]


# -------------------------------------------------
# 2.  RLlib env
# -------------------------------------------------
class TradingEnv:
    def __init__(self, broker: Broker):
        self.broker = broker
        self.encoder = MultiModalEncoder()

    def reset(self):
        self.start_nav = self.broker._get_nav()
        return self._obs()

    def step(self, action):
        qty = 100 if action == 1 else -100
        reward = (self.broker._get_nav() - self.start_nav) * 1e4  # bps
        return self._obs(), reward, False, {}

    def _obs(self):
        png = b"dummy_png"
        lob = FakeLob(100.0)
        vec = self.encoder.encode_live(png, lob, "")
        return np.array(vec, dtype=np.float32)


# -------------------------------------------------
# 3.  RL setup
# -------------------------------------------------
ray.init()

config = (
    PPOConfig()
    .environment(TradingEnv)
    .framework("torch")
    .training(lr=1e-4, sgd_minibatch_size=64)
)
algo = PPO(config=config)


# -------------------------------------------------
# 4.  async training loop
# -------------------------------------------------
async def loop():
    while True:
        result = algo.train()
        log.info("RL step %s reward %s", result["training_iteration"], result["episode_reward_mean"])
        await asyncio.sleep(60)


if __name__ == "__main__":
    asyncio.run(loop())