"""ViT encoder with graceful fallbacks."""
from io import BytesIO
from pathlib import Path
from typing import List

import torch
from PIL import Image
from transformers import ViTForImageClassification, ViTImageProcessor

from utils.config import load_config
from utils.logger import get_logger

cfg = load_config()
log = get_logger("VIT_ENCODER")

class ViTChartEncoder:
    def __init__(self) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # ── load the *fine-tuned* checkpoint we created with train_vit.py ──
        checkpoint_dir = Path(cfg["model"]["checkpoint_dir"]) / "prod"
        if not checkpoint_dir.exists():
            # fallback for first run – create placeholder dir so load still works
            checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.processor = ViTImageProcessor.from_pretrained(str(checkpoint_dir))
        self.model = (
            ViTForImageClassification.from_pretrained(str(checkpoint_dir))
            .to(self.device)
            .eval()
        )
        log.info("ViT loaded from %s on %s", checkpoint_dir, self.device)

    @torch.inference_mode()
    def encode(self, png_bytes: bytes) -> List[float]:
        try:
            img = Image.open(BytesIO(png_bytes)).convert("RGB")
            inputs = self.processor(images=img, return_tensors="pt").to(self.device)
            logits = self.model(**inputs).logits.squeeze().cpu().tolist()
            return logits
        except Exception as e:
            log.exception("ViT encode failed – returning zeros")
            return [0.0, 0.0, 0.0]