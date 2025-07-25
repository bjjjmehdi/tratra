"""Online logistic regression for fill-prob given (qty, queue_ahead, latency)."""
import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression

from utils.config import load_config
from utils.logger import get_logger
from data_ingestion.lob_stream import LobTick

cfg = load_config()
log = get_logger("FILL_MODEL")

class FillModel:
    def __init__(self, symbol: str) -> None:
        self.symbol = symbol
        self.model = LogisticRegression()
        self._X: list[list[float]] = []
        self._y: list[int] = []
        self.path = f"models/fillmodel_{symbol}.joblib"
        try:
            self.model = joblib.load(self.path)
        except FileNotFoundError:
            pass

    def predict(self, qty: int, queue_ahead: int, latency_us: int) -> float:
        """Return probability[0,1] that entire qty fills at passive price."""
        if not hasattr(self.model, "coef_"):
            return 0.5  # cold start
        X = np.array([[qty, queue_ahead, latency_us]])
        return float(self.model.predict_proba(X)[0, 1])

    def update(self, lob: LobTick, qty: int, filled: bool) -> None:
        """Online learning after each order."""
        self._X.append([qty, lob.latency_us, lob.ask[0][1] if lob.ask else 0])
        self._y.append(int(filled))
        if len(self._X) > 1000:  # mini-batch
            self.model.fit(self._X, self._y)
            joblib.dump(self.model, self.path)
            self._X.clear()
            self._y.clear()