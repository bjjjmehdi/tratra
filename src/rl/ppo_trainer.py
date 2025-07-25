"""Ray RLlib PPO helper."""
from typing import Dict

import ray
from ray.rllib.algorithms.ppo import PPO, PPOConfig

from utils.config import load_config
from utils.logger import get_logger

cfg = load_config()
log = get_logger("PPO_TRAINER")

class PPOTrainer:
    def __init__(self, env_name: str = "TradingEnv") -> None:
        config = (
            PPOConfig()
            .environment(env_name)
            .framework("torch")
            .resources(num_gpus=int(ray.cluster_resources().get("GPU", 0)))
            .training(
                lr=3e-4,
                lambda_=0.95,
                gamma=0.99,
                sgd_minibatch_size=128,
                num_sgd_iter=30,
            )
        )
        self.algo = PPO(config=config)

    def train(self) -> Dict:
        return self.algo.train()

    def save(self, path: str) -> None:
        self.algo.save(path)
        log.info(f"PPO checkpoint saved to {path}")