"""Unit test."""
import numpy as np
from PIL import Image

from src.encoders.vit_encoder import ViTChartEncoder

def test_vit_encode_graceful() -> None:
    enc = ViTChartEncoder()
    img = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
    buf = img.tobytes()
    logits = enc.encode(buf)
    assert len(logits) == 3
    assert all(isinstance(x, float) for x in logits)