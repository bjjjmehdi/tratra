"""Multi-time-frame, confidence-gated, hybrid technical agent."""
from typing import List, Tuple

import numpy as np
import pandas as pd
import pandas_ta as ta

from encoders.vit_encoder import ViTChartEncoder
from utils.config import load_config
from utils.logger import get_logger

cfg = load_config()
log = get_logger("TECH_AGENT")

# confidence gate
MIN_LOGIT_GAP: float = cfg["model"].get("min_logit_gap", 0.25)

class TechnicalAgent:
    def __init__(self) -> None:
        self.encoder = ViTChartEncoder()
        self._chart_history: List[bytes] = []
        self.max_seq = 3  # last 3 PNGs

    # ------------------------------------------------------------------ #
    # public API
    # ------------------------------------------------------------------ #
    def decide(self, png_bytes: bytes, df: pd.DataFrame) -> Tuple[str, float]:
        """
        Returns (action, confidence).
        confidence = max(logit) - second(logit).  If gap < MIN_LOGIT_GAP -> HOLD.
        """
        # 1. store sequence
        self._chart_history.append(png_bytes)
        if len(self._chart_history) > self.max_seq:
            self._chart_history.pop(0)

        # 2. ViT logits
        logits = self.encoder.encode(png_bytes)
        top2 = sorted(logits, reverse=True)[:2]
        gap = top2[0] - top2[1] if len(top2) == 3 else 0.0
        if gap < MIN_LOGIT_GAP:
            log.debug("Low confidence – HOLD")
            return "HOLD", 0.0

        action_idx = int(np.argmax(logits))
        action = ["HOLD", "BUY", "SELL"][action_idx]
        confidence = gap

        # 3. hybrid check with classic TA
        ta_signal = self._ta_signal(df)
        if ta_signal != action:
            log.debug("TA override – HOLD (%s vs %s)", ta_signal, action)
            return "HOLD", 0.0

        # 4. sequence consistency (majority vote of last N)
        if self._chart_history:
            seq_actions = [np.argmax(self.encoder.encode(c)) for c in self._chart_history]
            majority = "HOLD"
            counts = np.bincount(seq_actions, minlength=3)
            if counts[1] > counts[2] and counts[1] > counts[0]:
                majority = "BUY"
            elif counts[2] > counts[1] and counts[2] > counts[0]:
                majority = "SELL"
            if majority != action:
                log.debug("Sequence mismatch – HOLD")
                return "HOLD", 0.0

        return action, confidence

    # ------------------------------------------------------------------ #
    # helpers
    # ------------------------------------------------------------------ #
    def _ta_signal(self, df: pd.DataFrame) -> str:
        """Classic TA on 1-min bars."""
        if len(df) < 20:
            return "HOLD"
        # fast EMA vs slow EMA
        fast = ta.ema(df["close"], length=5).iloc[-1]
        slow = ta.ema(df["close"], length=20).iloc[-1]
        if fast > slow * 1.001:
            return "BUY"
        if fast < slow * 0.999:
            return "SELL"
        return "HOLD"