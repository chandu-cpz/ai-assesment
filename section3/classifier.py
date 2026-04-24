from __future__ import annotations

import os
from pathlib import Path

import torch
from torch import nn
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from .data_builder import LABELS, MODEL_DIR


class TicketClassifier:
    def __init__(self, model_dir: Path = MODEL_DIR, device: str | torch.device | None = None) -> None:
        if not (model_dir / "config.json").exists():
            raise FileNotFoundError(
                f"Model not found at {model_dir}. Run `python -m section3.train` first."
            )
        self.device = _resolve_device(device)
        if self.device.type == "cpu":
            torch.set_num_threads(min(2, max(1, os.cpu_count() or 1)))
            try:
                torch.set_num_interop_threads(1)
            except RuntimeError:
                pass
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        if self.device.type == "cpu":
            self.model = torch.quantization.quantize_dynamic(self.model, {nn.Linear}, dtype=torch.qint8)
        self.model.to(self.device)
        self.model.eval()
        self._warm_up()

    def predict(self, text: str) -> str:
        with torch.no_grad():
            encoded = self.tokenizer(
                text,
                truncation=True,
                max_length=96,
                return_tensors="pt",
            )
            encoded = {name: value.to(self.device) for name, value in encoded.items()}
            logits = self.model(**encoded).logits
            label_id = int(torch.argmax(logits, dim=1).item())
        label = self.model.config.id2label[label_id]
        if label not in LABELS:
            raise ValueError(f"Unexpected label returned: {label}")
        return label

    def predict_batch(self, texts: list[str]) -> list[str]:
        if not texts:
            return []
        with torch.no_grad():
            encoded = self.tokenizer(
                texts,
                truncation=True,
                padding=True,
                max_length=96,
                return_tensors="pt",
            )
            encoded = {name: value.to(self.device) for name, value in encoded.items()}
            logits = self.model(**encoded).logits
            label_ids = torch.argmax(logits, dim=1).tolist()
        return [self.model.config.id2label[int(label_id)] for label_id in label_ids]

    def _warm_up(self) -> None:
        with torch.no_grad():
            encoded = self.tokenizer(
                "warmup",
                truncation=True,
                max_length=8,
                return_tensors="pt",
            )
            encoded = {name: value.to(self.device) for name, value in encoded.items()}
            self.model(**encoded)


def _resolve_device(device: str | torch.device | None) -> torch.device:
    if device is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    resolved = torch.device(device)
    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA device requested but no CUDA runtime is available.")
    return resolved
