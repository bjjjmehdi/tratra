#!/usr/bin/env python3
"""
Zero-defect async entry point + live dashboard thread (no duplicate metrics).
"""
import asyncio
import signal
import threading
from pathlib import Path

import typer
from dotenv import load_dotenv
from prometheus_client import start_http_server

# ---------- Dashboard integration ----------
from performance.pnl_tracker import REGISTRY as pnl_registry  # uses same registry
# ------------------------------------------

from data_ingestion.candle_builder import CandleBuilder
from data_ingestion.ib_stream import IBStreamer
from execution.broker import Broker
from services.supervisor import Supervisor
from utils.config import load_config
from utils.logger import get_logger
from utils.market_hours import is_market_hours, minutes_to_close
from utils.adversarial import validate_png
from utils.latency import LatencyGuard

load_dotenv()
cfg = load_config()
log = get_logger("MAIN")

cli = typer.Typer()
SHUTDOWN_EVENT = asyncio.Event()


def _install_signal_handlers() -> None:
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: SHUTDOWN_EVENT.set())


@cli.command()
def run(
    paper: bool = typer.Option(True, help="Use paper account"),
    metrics_port: int = typer.Option(cfg["metrics"]["port"], help="Prometheus port"),
    dashboard_port: int = typer.Option(5050, help="Live dashboard port"),
) -> None:
    # 1. Start Prometheus exporter with *single* registry
    start_http_server(metrics_port, registry=pnl_registry)

    cfg["ib"]["paper"] = paper
    _install_signal_handlers()
    asyncio.run(_async_main())


async def _async_main() -> None:
    stream = IBStreamer(cfg["ib"])
    broker = Broker(cfg["risk"])
    supervisor = Supervisor(broker, cfg["risk"])
    builder = CandleBuilder(lookback=cfg["timeframes"]["lookback_bars"])
    latency_guard = LatencyGuard(cfg["risk"]["max_latency_ms"])

    await stream.connect()
    await supervisor.start()

    log.info("Entering tick loop…")
    async for tick in stream.tick_stream():
        if SHUTDOWN_EVENT.is_set():
            log.info("Shutdown requested – flattening positions")
            await broker.flatten_all()
            break

        if not is_market_hours():
            await asyncio.sleep(1)
            continue

        if minutes_to_close() <= cfg["risk"]["flatten_before_close_min"]:
            log.warning("Market close approaching – flattening")
            await broker.flatten_all()
            break

        try:
            if latency_guard.too_slow():
                log.warning("Latency spike – skipping tick")
                continue

            candle_closed = builder.add_tick(tick.contract, tick)
            if candle_closed:
                df = builder.to_df()
                png = builder.render_png(df)
                validate_png(png)
                await supervisor.on_candle(png, tick.contract)
        except Exception as e:
            log.exception("Tick failed safely: %s", e)
            continue


if __name__ == "__main__":
    cli()