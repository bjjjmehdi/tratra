"""Distributed ViT fine-tuning with W&B, early-stop, model registry."""
from io import BytesIO
from PIL import Image
import os
from pathlib import Path
from typing import List, Dict, Any

import wandb
from datasets import Dataset, load_dataset
from transformers import (
    ViTForImageClassification,
    ViTImageProcessor,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
)

from utils.config import load_config
from utils.logger import get_logger

cfg = load_config()
log = get_logger("TRAIN_VIT")
os.environ["WANDB_PROJECT"] = "trader-vit"

class ViTTrainer:
    def __init__(self, run_name: str) -> None:
        self.run_name = run_name
        self.dataset_dir = Path(cfg["dataset"]["raw_dir"])
        self.output_dir = Path(cfg["model"]["checkpoint_dir"]) / run_name

    def load_dataset(self) -> Dataset:
        ds = load_dataset("json", data_files=str(self.dataset_dir / "labels.jsonl"))["train"]
        processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")

        def _process(example):
            png = bytes.fromhex(example["png_b64"])
            img = processor(Image.open(BytesIO(png)).convert("RGB"), return_tensors="pt")
            example["pixel_values"] = img["pixel_values"][0]
            example["labels"] = 0 if example["reward"] < -0.001 else 1 if example["reward"] > 0.001 else 2
            return example

        ds = ds.map(_process, remove_columns=["png_b64", "reward", "metadata"])
        return ds.train_test_split(test_size=0.1)

    def train(self) -> None:
        dataset = self.load_dataset()
        model = ViTForImageClassification.from_pretrained(
            "google/vit-base-patch16-224-in21k",
            num_labels=3,
            ignore_mismatched_sizes=True,
        )
        args = TrainingArguments(
            output_dir=str(self.output_dir),
            run_name=self.run_name,
            num_train_epochs=cfg["training"]["epochs"],
            per_device_train_batch_size=cfg["training"]["batch_size"],
            evaluation_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            logging_steps=50,
            report_to="wandb",
            dataloader_num_workers=4,
        )

        trainer = Trainer(
            model=model,
            args=args,
            train_dataset=dataset["train"],
            eval_dataset=dataset["test"],
            compute_metrics=lambda p: {"accuracy": (p.predictions.argmax(-1) == p.label_ids).mean()},
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
        )
        trainer.train()
        trainer.save_model(self.output_dir)
        log.info("Training done. Saved to %s", self.output_dir)

if __name__ == "__main__":
    import argparse, datetime as dt
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_name", default=f"vit-{dt.date.today().isoformat()}")
    args = parser.parse_args()
    ViTTrainer(args.run_name).train()