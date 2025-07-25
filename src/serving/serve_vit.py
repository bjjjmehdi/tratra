"""TorchServe handler for ultra-low-latency ViT inference."""
import json
import logging
import os
from io import BytesIO

import torch
from PIL import Image
from ts.torch_handler.base_handler import BaseHandler

from registry.model_registry import ModelRegistry

class ViTHandler(BaseHandler):
    def initialize(self, context):
        self.manifest = context.manifest
        model_dir = context.system_properties.get("model_dir")
        tag = os.getenv("MODEL_TAG", "prod")
        self.model, self.processor = ModelRegistry.load_vit(tag)
        self.model.eval()

    def preprocess(self, data):
        png = list(data[0].values())[0]
        return self.processor(Image.open(BytesIO(png)).convert("RGB"), return_tensors="pt")["pixel_values"]

    def inference(self, inputs):
        with torch.no_grad():
            return self.model(inputs).logits

    def postprocess(self, outputs):
        return [outputs.cpu().tolist()]