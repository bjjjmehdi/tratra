"""Efficient deque-based replay buffer."""
import pickle
from collections import deque
from pathlib import Path
from typing import Any, Deque, NamedTuple

import numpy as np

class Transition(NamedTuple):
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool

class ReplayBuffer:
    def __init__(self, max_size: int = 100_000) -> None:
        self.buffer: Deque[Transition] = deque(maxlen=max_size)

    def add(self, transition: Transition) -> None:
        self.buffer.append(transition)

    def sample(self, batch_size: int) -> list[Transition]:
        idx = np.random.choice(len(self.buffer), batch_size)
        return [self.buffer[i] for i in idx]

    def save(self, path: Path) -> None:
        with open(path, "wb") as f:
            pickle.dump(list(self.buffer), f)

    def load(self, path: Path) -> None:
        with open(path, "rb") as f:
            data = pickle.load(f)
            self.buffer.extend(data)