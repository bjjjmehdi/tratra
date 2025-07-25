"""PatchTST time-series encoder (stub)."""
import numpy as np
import torch

class TSEncoder:
    def __init__(self) -> None:
        self.model = None  # TODO: PatchTST integration

    def encode(self, series: np.ndarray) -> np.ndarray:
        """Return latent vector for series."""
        return series[-64:]  # stub