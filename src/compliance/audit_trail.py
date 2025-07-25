"""FINRA CAT / MiFID compliant drop-copy."""
import asyncio
import datetime as dt
import json
import os
from pathlib import Path
from typing import Any, Dict
from risk.portfolio_risk import PortfolioRisk
from utils.config import load_config
import httpx
from prometheus_client import Counter

from utils.logger import get_logger

log = get_logger("AUDIT")
cfg = load_config()

# Prometheus counter
AUDIT_RECORDS = Counter("audit_records_total", "Regulatory records emitted")

class AuditTrail:
    def __init__(self) -> None:
        self.enabled = cfg["compliance"]["audit_enabled"]
        self.endpoint = cfg["compliance"]["audit_endpoint"]  # REST drop-copy
        self.local_path = Path(cfg["compliance"]["local_log"])
        self.local_path.parent.mkdir(parents=True, exist_ok=True)

    async def record(self, event: Dict[str, Any]) -> None:
        if not self.enabled:
            return
        event["ts"] = dt.datetime.utcnow().isoformat()
        line = json.dumps(event) + "\n"

        # 1. local append-only log
        with self.local_path.open("a") as f:
            f.write(line)

        # 2. push to regulatory gateway
        try:
            async with httpx.AsyncClient(timeout=5) as client:
                r = await client.post(self.endpoint, json=event)
                r.raise_for_status()
        except Exception as e:
            log.warning("Audit push failed – only local: %s", e)

        AUDIT_RECORDS.inc()