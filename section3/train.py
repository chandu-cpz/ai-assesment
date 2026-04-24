from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from functools import partial

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from .data_builder import ID_TO_LABEL, LABELS, LABEL_TO_ID, MODEL_DIR, MODEL_NAME, load_train_examples


MAX_LENGTH = 96
TRAINING_SEED = 42


class TicketDataset(Dataset):
    def __init__(self, examples: list[dict[str, str]]) -> None:
        self.examples = examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, str]:
        return self.examples[index]


@dataclass(slots=True)
class TrainingSummary:
    train_loss: float
    validation_accuracy: float
    epochs: int
    model_dir: str


def train_model(epochs: int = 3, batch_size: int = 16, learning_rate: float = 5e-5) -> TrainingSummary:
    _set_seed(TRAINING_SEED)
    examples = load_train_examples()
    train_rows, val_rows = train_test_split(
        examples,
        test_size=0.2,
        random_state=42,
        stratify=[row["label"] for row in examples],
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(LABELS),
        label2id=LABEL_TO_ID,
        id2label=ID_TO_LABEL,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    train_loader = DataLoader(
        TicketDataset(train_rows),
        batch_size=batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(TRAINING_SEED),
        collate_fn=partial(_collate_batch, tokenizer),
    )
    val_loader = DataLoader(
        TicketDataset(val_rows),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=partial(_collate_batch, tokenizer),
    )

    optimizer = AdamW(model.parameters(), lr=learning_rate)

    for _epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for batch in train_loader:
            batch = {name: value.to(device) for name, value in batch.items()}
            optimizer.zero_grad()
            output = model(**batch)
            output.loss.backward()
            optimizer.step()
            running_loss += float(output.loss.item())

    validation_accuracy = _evaluate_accuracy(model, val_loader, device)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(MODEL_DIR)
    tokenizer.save_pretrained(MODEL_DIR)
    summary = TrainingSummary(
        train_loss=running_loss / max(1, len(train_loader)),
        validation_accuracy=validation_accuracy,
        epochs=epochs,
        model_dir=str(MODEL_DIR),
    )
    (MODEL_DIR / "training_summary.json").write_text(
        json.dumps(asdict(summary), indent=2), encoding="utf-8"
    )
    return summary


def model_exists() -> bool:
    return (MODEL_DIR / "config.json").exists()


def _collate_batch(tokenizer: AutoTokenizer, batch: list[dict[str, str]]) -> dict[str, torch.Tensor]:
    encoding = tokenizer(
        [item["text"] for item in batch],
        padding=True,
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors="pt",
    )
    encoding["labels"] = torch.tensor([LABEL_TO_ID[item["label"]] for item in batch], dtype=torch.long)
    return encoding


def _evaluate_accuracy(
    model: AutoModelForSequenceClassification,
    loader: DataLoader,
    device: torch.device,
) -> float:
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in loader:
            batch = {name: value.to(device) for name, value in batch.items()}
            logits = model(**batch).logits
            predictions = torch.argmax(logits, dim=1)
            labels = batch["labels"]
            correct += int((predictions == labels).sum().item())
            total += int(labels.size(0))
    return correct / max(1, total)


def _set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main() -> None:
    summary = train_model()
    print(
        f"Saved model to {summary.model_dir} after {summary.epochs} epochs; "
        f"validation accuracy={summary.validation_accuracy:.2%}, loss={summary.train_loss:.4f}"
    )


if __name__ == "__main__":
    main()
