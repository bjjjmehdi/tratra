"""Monitors model drift and triggers retraining."""
import datetime as dt
import joblib
import pandas as pd
from pathlib import Path
from sklearn.metrics import log_loss

from performance.pnl_tracker import PnLTracker
from utils.config import load_config
from utils.logger import get_logger

cfg = load_config()
log = get_logger("DRIFT_GUARD")

MODEL_PATH = Path(cfg["model"]["checkpoint_dir"]) / "vit_last.joblib"
DRIFT_WINDOW = cfg["model"].get("drift_window_days", 7)
DRIFT_THRESHOLD = cfg["model"].get("drift_threshold", 0.05)  # increase in log-loss


class DriftGuard:
    def __init__(self, ib) -> None:
        self.tracker = PnLTracker(ib)
        self.last_check = dt.datetime.utcnow()

    def _load_model(self):
        if MODEL_PATH.exists():
            return joblib.load(MODEL_PATH)
        return None

    def _rolling_accuracy(self) -> float:
        # crude: compare predicted vs actual sign of next 1-min return
        df = self.tracker.equity_df()
        if len(df) < DRIFT_WINDOW * 24 * 60:
            return 1.0  # not enough data
        df = df.tail(DRIFT_WINDOW * 24 * 60)
        df["pred"] = df["nav"].shift(1)  # placeholder for model prediction
        df["actual"] = (df["nav"].diff() > 0).astype(int)
        return (df["pred"] == df["actual"]).mean()

    def check_and_retrain(self) -> bool:
        """Return True if retrain should be triggered."""
        acc = self._rolling_accuracy()
        if acc < (1 - DRIFT_THRESHOLD):
            log.warning("Model drift detected (acc={:.3f}) – triggering retrain", acc)
            return True
        return False