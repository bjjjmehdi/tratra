#!/usr/bin/env python3
"""
Patched Backtest Engine with Proper Mocks
"""

import argparse
import asyncio
from pathlib import Path
import pandas as pd
from tqdm import tqdm
from agents.sentiment_agent import SentimentAgent
from agents.technical_agent import TechnicalAgent
from brain import Decision, KimiDecisionMaker
from data_ingestion.candle_builder import CandleBuilder
from execution.impact_model import ImpactEstimate, ImpactModel
from execution.micro_price import MicroPriceEngine, SizedOrder
from performance.pnl_tracker import PnLTracker
from utils.config import load_config
from utils.logger import get_logger
from dotenv import load_dotenv

load_dotenv("credentials.env")
cfg = load_config()
log = get_logger("LIVE_REPLAY")

class MockPortfolioRisk:
    def snapshot(self) -> dict:
        return {
            "var_95": 0.0,  # Mocked VAR value
            "breach": False
        }

class MockRegTGuard:
    def snapshot(self) -> dict:
        return {
            "breach": False,
            "sma_ratio": 1.0  # Important for brain.py
        }

class LiveStackReplay:
    def __init__(self) -> None:
        self.symbol = "INTC"
        self.technical = TechnicalAgent()
        self.brain = KimiDecisionMaker(cfg["model"])
        self.sentiment = SentimentAgent(cfg["model"]["kimi_key"])
        self.impact = ImpactModel(gamma=cfg["impact"]["gamma"], eta=cfg["impact"]["eta"])
        self.micro = MicroPriceEngine()
        self.port_risk = MockPortfolioRisk()
        self.reg_t = MockRegTGuard()

        self.position = {
            "qty": 0,
            "avg_price": 0.0,
            "real_pnl": 0.0,
            "unreal_pnl": 0.0
        }
        self.trades = []

    def _mock_nav(self) -> float:
        return 100_000.0

    def _mock_margin(self) -> float:
        return 0.0

    def _mock_lob(self, mid: float):
        return type(
            "MockLob",
            (),
            {
                "bid": [(mid - 0.01 * i, 100) for i in range(1, 6)],
                "ask": [(mid + 0.01 * i, 100) for i in range(1, 6)],
                "latency_us": 200,
                "imbalance": 0.0
            },
        )()

    async def run(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.set_index("timestamp").sort_index()
        log.info("Loaded %d bars", len(df))

        builder = CandleBuilder()
        for idx, (ts, row) in enumerate(tqdm(df.iterrows(), total=len(df))):
            if idx % 2:
                continue

            lookback = df.loc[:ts].tail(60)
            if len(lookback) < 20:
                continue

            png = builder.render_png(lookback)
            mid = float(row["close"])
            lob = self._mock_lob(mid)

            decision = await self.brain.decide(
                png_bytes=png,
                agent=self.technical,
                nav=self._mock_nav(),
                last_price=mid,
                position=self.position,
                lob_imbalance=0.0,
                headline="",
                sent_score=0.0,
                memory="",
                var_95=self.port_risk.snapshot()["var_95"],  # Pass mocked VAR
                sma_ratio=self.reg_t.snapshot()["sma_ratio"]  # Pass mocked SMA
            )

            if decision.action == "HOLD":
                continue

            sized = SizedOrder(
                action=decision.action,
                qty=decision.qty,
                limit=mid,
                stop=decision.stop_loss,
                take=decision.take_profit,
            )

            impact = self.impact.estimate(sized.qty, sized.action, lob)
            if impact.slippage_bps > cfg["impact"]["max_slippage_bps"]:
                continue

            micro = self.micro.compute(lob, sized.qty)
            if micro.cost_bps > cfg["micro"]["max_cost_bps"]:
                continue

            fill_px = impact.expected_price
            if decision.action == "BUY":
                new_qty = self.position["qty"] + sized.qty
                self.position["avg_price"] = (
                    self.position["avg_price"] * self.position["qty"] + fill_px * sized.qty
                ) / new_qty
                self.position["qty"] = new_qty
            else:
                closed = min(self.position["qty"], sized.qty)
                self.position["real_pnl"] += (fill_px - self.position["avg_price"]) * closed
                self.position["qty"] -= sized.qty

            self.position["unreal_pnl"] = (mid - self.position["avg_price"]) * self.position["qty"]

            self.trades.append({
                "timestamp": ts.isoformat(),
                "action": decision.action,
                "qty": sized.qty,
                "fill_px": fill_px,
                "stop": decision.stop_loss,
                "take": decision.take_profit,
                "reasoning": decision.reasoning,
                "real_pnl": self.position["real_pnl"],
                "unreal_pnl": self.position["unreal_pnl"],
                "slippage_bps": impact.slippage_bps,
                "confidence": decision.confidence
            })

        return pd.DataFrame(self.trades)

    def summary(self, trades: pd.DataFrame) -> None:
        if trades.empty:
            log.warning("No trades executed")
            return

        stats = [
            ("Total Trades", len(trades)),
            ("Winning %", f"{len(trades[trades['real_pnl'] > 0])/len(trades)*100:.1f}%"),
            ("Avg Confidence", f"{trades['confidence'].mean():.1%}"),
            ("Final PnL", f"${trades['real_pnl'].iloc[-1]:,.2f}"),
            ("Max Drawdown", f"${trades['real_pnl'].cummin().min():,.2f}")
        ]
        
        log.info("=" * 60)
        log.info("BACKTEST RESULTS")
        for metric, value in stats:
            log.info(f"{metric:<15}: {value}")
        log.info("=" * 60)

async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", default="data/replay/INTC_1m.csv")
    args = parser.parse_args()

    csv_path = Path(__file__).parents[2] / args.file
    if not csv_path.exists():
        log.error("CSV not found: %s", csv_path)
        return

    df = pd.read_csv(csv_path, parse_dates=["timestamp"])
    bt = LiveStackReplay()
    trades = await bt.run(df)
    bt.summary(trades)

    out = Path(__file__).parents[2] / "data/backtest/INTC_kimi_trades.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    trades.to_json(out, orient="records", lines=True)
    log.info("Results saved to %s", out)

if __name__ == "__main__":
    asyncio.run(main())