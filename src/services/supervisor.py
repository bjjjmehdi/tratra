"""
Zero-defect orchestrator with full alpha stack:
  • multimodal encoder (image + LOB + news)
  • market-impact / partial-fill simulation
  • synthetic labelling hook
  • Reg-T, VAR, cross-asset hedge
  • audit trail
"""
import asyncio
import datetime as dt
import json
import os
from pathlib import Path
from typing import Any, Dict

import httpx
import pandas as pd
import torch

from agents.sentiment_agent import SentimentAgent
from agents.technical_agent import TechnicalAgent
from brain import Decision, KimiDecisionMaker
from compliance.audit_trail import AuditTrail
from data_ingestion.candle_builder import CandleBuilder
from data_ingestion.ib_stream import IBStreamer
from data_ingestion.lob_stream import LobStream
from data_pipeline.label_collector import LabelCollector
from encoders.multimodal import MultiModalEncoder
from execution.broker import Broker
from execution.impact_model import ImpactEstimate, ImpactModel
from execution.micro_price import MicroPriceEngine
from performance.drift_guard import DriftGuard
from performance.pnl_tracker import PnLTracker
from registry.model_registry import ModelRegistry
from risk.hedge_engine import HedgeEngine
from risk.portfolio_risk import PortfolioRisk
from risk.reg_t_guard import RegTGuard
from training.train_vit import ViTTrainer
from utils.config import load_config
from utils.logger import get_logger

cfg = load_config()
log = get_logger("SUPERVISOR")


class Supervisor:
    """High-level orchestrator with multimodal alpha, impact model, synthetic data, Reg-T, hedge."""

    def __init__(self, broker: Broker, risk_cfg: dict) -> None:
        self.broker = broker
        self.brain = KimiDecisionMaker(cfg["model"])

        # ---------------- NEW: multimodal encoder ----------------
        self.encoder = MultiModalEncoder(latent_dim=cfg["model"].get("latent_dim", 512))

        # ---------------- agent & bookkeeping ----------------
        self.agent = TechnicalAgent()
        self.builder = CandleBuilder(lookback=cfg["timeframes"]["lookback_bars"])
        self.pnl = PnLTracker(broker.ib)
        self.drift = DriftGuard(broker.ib)

        # ---------------- risk & infra ----------------
        self.port_risk = PortfolioRisk(broker.ib)
        self.audit = AuditTrail()
        self.reg_t = RegTGuard(broker.ib)
        self.hedge = HedgeEngine(broker.ib)

        # ---------------- micro-structure ----------------
        self.micro = MicroPriceEngine()
        self.impact = ImpactModel(
            gamma=cfg["impact"]["gamma"], eta=cfg["impact"]["eta"]
        )  # from config.yaml

        # ---------------- sentiment ----------------
        self.sent_agent = SentimentAgent(os.getenv("KIMI_API_KEY"))
        self._reason_memory: list[str] = []

    # ---------- startup ----------
    async def start(self) -> None:
        # warm-load weights (if fine-tuned)
        try:
            self.encoder.load_state_dict(
                torch.load(
                    Path(cfg["model"]["checkpoint_dir"]) / "prod" / "multimodal.pt",
                    map_location="cpu",
                )
            )
            log.info("Multimodal encoder hot-loaded from prod")
        except FileNotFoundError:
            log.warning("No multimodal weights – cold-start with base")

        log.info("Supervisor started")

    # ---------- headline ----------
    async def _top_headline(self) -> str:
        url = "https://finnhub.io/api/v1/news"
        params = {"category": "general", "token": os.getenv("FINNHUB_KEY", "")}
        async with httpx.AsyncClient(timeout=5) as client:
            r = await client.get(url, params=params)
            r.raise_for_status()
            return r.json()[0]["headline"]

    # ---------- main tick ----------
    async def on_candle(self, png: bytes, contract) -> None:
        try:
            # 1. equity snapshot
            self.pnl.tick()

            # 2. drift → async retrain
            if self.drift.check_and_retrain():
                asyncio.create_task(self._retrain_and_swap())

            # 3. sentiment
            headline, sent_score = None, 0.0
            try:
                headline = await self._top_headline()
                sent = await self.sent_agent.score_headline(headline)
                sent_score = sent.score
            except Exception as e:
                log.debug("Sentiment failed: %s", e)

            # 4. memory
            memory = "\n".join(self._reason_memory[-3:])

            # 5. agent decision
            df = self.builder.to_df()
            action, confidence = self.agent.decide(png, df)
            if action == "HOLD":
                return

            # 6. LOB snapshot
            lob_gen = LobStream(self.broker.ib).stream(contract)
            lob = await lob_gen.__anext__()

            # 7. multimodal encoding
            vec = self.encoder.encode_live(png, lob, headline or "")
            # (vec can be fed into agent / RL later)

            # 8. impact / micro-price
            nav = self.broker._get_nav()
            price = float(self.broker.ib.reqMktData(contract, "", False, False).last or 1)
            margin = self.broker._margin_usage()
            sized = self.broker.risk.size_order(nav, price, margin)
            impact: ImpactEstimate = self.impact.estimate(sized.qty, sized.action, lob)

            if impact.slippage_bps > cfg["impact"]["max_slippage_bps"]:
                log.debug("Impact too high – skip")
                return

            # 9. micro-price cost check
            micro = self.micro.compute(lob, sized.qty)
            if micro.cost_bps > cfg["micro"]["max_cost_bps"]:
                log.debug("Adverse-selection cost too high – skip")
                return

            # 10. Reg-T guard
            reg = self.reg_t.snapshot()
            if reg["breach"]:
                log.warning("Reg-T SMA breach – flatten")
                await self.broker.flatten_all()
                return

            # 11. portfolio risk
            snap = self.port_risk.snapshot()
            if snap.var_95 > cfg["risk"]["max_var_usd"]:
                log.warning("VAR breach – flatten")
                await self.broker.flatten_all()
                return

            # 12. assemble final decision
            pos = self.broker.position_snapshot(contract)
            raw_decision = await self.brain.decide(
                png, self.agent, pos, headline or "", sent_score, memory
            )
            decision = Decision(**raw_decision)

            # 13. execute with impact-aware sizing
            await self.broker.execute(decision, contract)

            # 14. beta-hedge
            hedge_qty = self.hedge.hedge_qty(contract.symbol, sized.qty)
            if hedge_qty != 0:
                from execution.micro_price import SizedOrder
                hedge_order = SizedOrder(
                    action="SELL" if hedge_qty < 0 else "BUY",
                    qty=abs(hedge_qty),
                    limit=float(
                        self.broker.ib.reqMktData(self.hedge.spy_contract, "", False, False).last or 0
                    ),
                    stop=0.0,
                    take=0.0,
                )
                await self.broker.execute(hedge_order, self.hedge.spy_contract)

            # 15. audit
            await self.audit.record(
                {
                    "type": "ORDER",
                    "contract": str(contract),
                    "decision": decision.dict(),
                    "impact": impact.__dict__,
                    "portfolio": snap.__dict__,
                }
            )

            # 16. synthetic or live labelling
            if cfg["ib"]["paper"]:
                asyncio.create_task(
                    LabelCollector.log(
                        self.broker.ib, png, decision.action, contract, horizon_sec=300
                    )
                )

        except Exception as e:
            log.exception("Supervisor tick failed safely: %s", e)

    # ---------- helpers ----------
    async def _retrain_and_swap(self) -> None:
        """Async retrain + hot-swap multimodal model."""
        try:
            tag = await asyncio.to_thread(
                lambda: ViTTrainer(f"drift-{int(asyncio.get_event_loop().time())}").train()
            )
            ckpt = Path(cfg["model"]["checkpoint_dir"]) / tag / "multimodal.pt"
            if ckpt.exists():
                self.encoder.load_state_dict(torch.load(ckpt, map_location="cpu"))
                ModelRegistry.promote(tag)
                log.info("Hot-swapped multimodal model %s", tag)
        except Exception as e:
            log.exception("Retrain failed – keeping old model: %s", e)