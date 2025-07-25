"""Structured JSON logging."""
import json
import logging
import logging.handlers
import os
from pathlib import Path

from utils.config import load_config

cfg = load_config()
_LOG_DIR = Path("logs")
_LOG_DIR.mkdir(exist_ok=True)

class JSONFormatter(logging.Formatter):
    def format(self, record):
        return json.dumps(
            {
                "ts": self.formatTime(record),
                "level": record.levelname,
                "logger": record.name,
                "msg": record.getMessage(),
                "module": record.module,
            }
        )

def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(getattr(logging, cfg["logging"]["level"].upper()))

    ch = logging.StreamHandler()
    ch.setFormatter(JSONFormatter())
    logger.addHandler(ch)

    fh = logging.handlers.RotatingFileHandler(
        _LOG_DIR / f"{name}.log",
        maxBytes=cfg["logging"]["max_file_bytes"],
        backupCount=cfg["logging"]["backup_count"],
    )
    fh.setFormatter(JSONFormatter())
    logger.addHandler(fh)
    return logger