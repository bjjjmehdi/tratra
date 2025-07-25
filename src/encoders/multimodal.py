"""
Multimodal encoder:
  • image (ViT)  
  • LOB tensor (5-depth × 4 fields)  
  • news embedding (MiniLM)  
Concatenated latent → single vector.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import List

import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer
from transformers import ViTModel, ViTImageProcessor

from data_ingestion.lob_stream import LobTick
from utils.config import load_config
from utils.logger import get_logger

cfg = load_config()
log = get_logger("MULTIMODAL")

IMG_SIZE     = 224
LOB_DEPTH    = 5
LOB_FIELDS   = 4    # price, size, imbalance, latency
NEWS_DIM     = 384  # MiniLM

class MultiModalEncoder(nn.Module):
    def __init__(self, latent_dim: int = 512):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # vision
        self.vit = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")
        self.vit_proj = nn.Linear(self.vit.config.hidden_size, latent_dim)

        # LOB tensor
        self.lob_cnn = nn.Sequential(
            nn.Conv2d(1, 32, (3, 2), padding=(1, 0)),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 32)),
            nn.Flatten(),
            nn.Linear(32, latent_dim),
        )

        # text
        self.text_encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        for p in self.text_encoder.parameters():
            p.requires_grad = False
        self.text_proj = nn.Linear(NEWS_DIM, latent_dim)

        # fusion
        self.fusion = nn.Sequential(
            nn.Linear(latent_dim * 3, latent_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(latent_dim, latent_dim),
        )

    def forward(self, img: torch.Tensor, lob: torch.Tensor, text: str):
        img_vec = self.vit(img).pooler_output
        img_vec = self.vit_proj(img_vec)

        lob_vec = self.lob_cnn(lob.unsqueeze(1))  # (B,1,D,H)

        text_vec = self.text_encoder.encode([text], convert_to_tensor=True, device=self.device)
        text_vec = self.text_proj(text_vec)

        fused = torch.cat([img_vec, lob_vec, text_vec], dim=-1)
        return self.fusion(fused)

    @torch.inference_mode()
    def encode_live(self, png_bytes: bytes, lob: LobTick, headline: str) -> List[float]:
        from PIL import Image
        from io import BytesIO
        import numpy as np

        # image
        img = Image.open(BytesIO(png_bytes)).convert("RGB")
        img = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")(
            images=img, return_tensors="pt"
        )["pixel_values"].to(self.device)

        # LOB tensor (depth × fields)
        lob_matrix = np.zeros((LOB_DEPTH, LOB_FIELDS))
        for i, (p, s) in enumerate(lob.bid[:LOB_DEPTH]):
            lob_matrix[i, :2] = [p, s]
        for i, (p, s) in enumerate(lob.ask[:LOB_DEPTH]):
            lob_matrix[i, 2:4] = [p, s]
        lob_tensor = torch.tensor(lob_matrix, dtype=torch.float32).unsqueeze(0).to(self.device)

        vec = self.forward(img, lob_tensor, headline).squeeze().cpu().tolist()
        return vec