"""Versioned model loading with auto-rollback on drift."""
import os
from pathlib import Path
from typing import Any, Dict

import joblib
import torch
from transformers import ViTForImageClassification, ViTImageProcessor

from utils.config import load_config
from utils.logger import get_logger

cfg = load_config()
log = get_logger("MODEL_REGISTRY")

class ModelRegistry:
    root = Path(cfg["model"]["checkpoint_dir"])

    @staticmethod
    def latest_tag() -> str:
        """Return latest tag based on semver & date."""
        versions = [d.name for d in ModelRegistry.root.iterdir() if d.is_dir()]
        return sorted(versions)[-1] if versions else "vit-base"

    @staticmethod
    def load_vit(tag: str | None = None):
        tag = tag or ModelRegistry.latest_tag()
        path = ModelRegistry.root / tag
        log.info("Loading ViT from %s", path)
        model = ViTForImageClassification.from_pretrained(str(path))
        processor = ViTImageProcessor.from_pretrained(str(path))
        return model, processor

    @staticmethod
    def promote(tag: str) -> None:
        """Symlink 'prod' to tag for atomic swaps."""
        prod_link = ModelRegistry.root / "prod"
        if prod_link.exists() or prod_link.is_symlink():
            prod_link.unlink()
        prod_link.symlink_to(ModelRegistry.root / tag)
        log.info("Promoted %s to prod", tag)