"""
SnapTraderAI 1-min HyperScalper - Production Ready
"""

import asyncio
import base64
import json
import os
from typing import Any, Dict, Optional, Tuple

import httpx
from pydantic import BaseModel, Field

from agents.technical_agent import TechnicalAgent
from utils.config import load_config
from utils.logger import get_logger

cfg = load_config()
log = get_logger("BRAIN")


class Decision(BaseModel):
    action: str = Field(pattern=r"^(BUY|SELL|HOLD)$")
    qty: int = Field(ge=0)
    stop_loss: float = Field(gt=0)
    take_profit: float = Field(gt=0)
    pattern_type: str
    confidence: float = Field(ge=0, le=1)  # Decimal between 0-1
    reasoning: str


class KimiDecisionMaker:
    def __init__(self, model_cfg: Optional[Dict[str, Any]] = None) -> None:
        self.model_cfg = model_cfg or cfg["model"]
        self.url = self.model_cfg.get("kimi_url", "https://api.moonshot.cn/v1/chat/completions")
        self.headers = {"Authorization": f"Bearer {os.getenv('KIMI_API_KEY')}"}
        self.hold_counter = 0
        self.max_hold_bars = 20
        self.min_tp_distance = 0.01
        self.min_sl_distance = 0.01

    @staticmethod
    def _risk_budget(nav: float) -> float:
        return nav * 0.015

    @staticmethod
    def _shares_from_budget(budget: float, entry: float, sl: float) -> int:
        risk_per_share = abs(entry - sl)
        return max(1, int(budget / max(risk_per_share, 0.0001)))

    def _validate_levels(self, last: float, sl: float, tp: float) -> Tuple[float, float]:
        sl_valid = min(sl, last - self.min_sl_distance)
        tp_valid = max(tp, last + self.min_tp_distance)
        
        if (tp_valid <= last and sl_valid >= last) or (tp_valid >= last and sl_valid <= last):
            tp_valid = last + abs(last - sl_valid) * 2 if tp_valid > last else last - abs(last - sl_valid) * 2
        
        return sl_valid, tp_valid

    async def decide(
        self,
        png_bytes: bytes,
        agent: TechnicalAgent,
        nav: float,
        last_price: float,
        position: Dict[str, Any],
        lob_imbalance: float = 0.0,
        headline: str = "",
        sent_score: float = 0.0,
        memory: str = "",
        **_
    ) -> Decision:
        img_b64 = base64.b64encode(png_bytes).decode()
        pos_txt = (
            f"Long {position['qty']} @ {position['avg_price']:.4f}"
            if position["qty"] > 0
            else ("Short {abs(position['qty'])} @ {position['avg_price']:.4f}"
                 if position["qty"] < 0 else "FLAT")
        )

        # Dynamic confidence calculation
        base_conf = 0.7
        min_conf = max(0.5, base_conf - 0.01 * self.hold_counter)
        if abs(lob_imbalance) > 0.3:
            min_conf = max(0.9, min_conf + 0.2)

        prompt = (
            f"SNAP-HYPERSCALPER MODE - FORCE DECISION AFTER {self.max_hold_bars} HOLDS\n"
            f"NAV=${nav:,.0f} | Last=${last_price:.4f} | {pos_txt}\n"
            f"LOB={lob_imbalance:+.2f} | HoldStreak={self.hold_counter}/{self.max_hold_bars}\n\n"
            "OUTPUT REQUIREMENTS:\n"
            "- Confidence: 0.0 to 1.0 (NOT percentage)\n"
            "- TP/SL must be chart-derived\n"
            "{\n"
            '  "action": "BUY|SELL|HOLD",\n'
            '  "stop_loss": float (exact price),\n'
            '  "take_profit": float (exact price),\n'
            '  "confidence": float (0.0-1.0),\n'
            '  "pattern_type": "BullFlag|BearTrap|etc",\n'
            '  "reasoning": "Brief rationale"\n'
            "}"
        )

        payload = {
            "model": self.model_cfg["kimi_model"],
            "response_format": {"type": "json_object"},
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}},
                        {"type": "text", "text": prompt},
                    ],
                }
            ],
            "max_tokens": 400,
            "temperature": 0.4,
        }

        # API call with retries
        raw = None
        for attempt in range(3):
            try:
                async with httpx.AsyncClient(timeout=30) as client:
                    r = await client.post(self.url, headers=self.headers, json=payload)
                    if r.status_code == 429:
                        await asyncio.sleep(2 ** attempt)
                        continue
                    r.raise_for_status()
                    raw = r.json()["choices"][0]["message"]["content"]
                    break
            except Exception as e:
                log.warning(f"Attempt {attempt + 1} failed: {str(e)}")
                await asyncio.sleep(1)

        # Parse with fallbacks
        try:
            data = json.loads(raw) if raw else {}
            if not data:
                data = {"action": "HOLD", "confidence": 0}
        except json.JSONDecodeError:
            data = {"action": "HOLD", "confidence": 0}

        # Convert confidence to decimal if needed
        raw_confidence = float(data.get("confidence", 0))
        confidence = raw_confidence / 100 if raw_confidence > 1 else raw_confidence
        confidence = max(0, min(1, confidence))  # Clamp to 0-1

        # Extract and validate levels
        action = str(data.get("action", "HOLD"))
        raw_sl = float(data.get("stop_loss", last_price * 0.98))
        raw_tp = float(data.get("take_profit", last_price * 1.02))
        sl, tp = self._validate_levels(last_price, raw_sl, raw_tp)

        # Force decision logic
        if action == "HOLD":
            self.hold_counter += 1
            if self.hold_counter >= self.max_hold_bars:
                action = "BUY" if lob_imbalance > 0 else "SELL"
                sl = last_price * 0.98
                tp = last_price * 1.02
                confidence = max(min_conf, 0.5)
                self.hold_counter = 0
        else:
            self.hold_counter = 0

        # Calculate position size
        qty = int(data.get("qty", 0))
        if action != "HOLD":
            qty = self._shares_from_budget(self._risk_budget(nav), last_price, sl)

        return Decision(
            action=action,
            qty=qty,
            stop_loss=sl,
            take_profit=tp,
            confidence=confidence,
            pattern_type=str(data.get("pattern_type", "N/A")),
            reasoning=str(data.get("reasoning", "Fallback decision")),
        )