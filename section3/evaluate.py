from __future__ import annotations

from itertools import product
from statistics import median
import time

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import json

from .classifier import TicketClassifier
from .data_builder import DATA_DIR, LABELS, load_eval_examples
from .train import model_exists, train_model


def evaluate_classifier() -> dict[str, object]:
    if not model_exists():
        train_model()
    rows = load_eval_examples()
    startup_started = time.perf_counter()
    classifier = TicketClassifier(device="cpu")
    startup_ms = (time.perf_counter() - startup_started) * 1000
    texts = [row["text"] for row in rows]
    y_true = [row["label"] for row in rows]
    y_pred = classifier.predict_batch(texts)
    accuracy = accuracy_score(y_true, y_pred)
    cls_report = classification_report(y_true, y_pred, labels=LABELS, output_dict=True, zero_division=0)
    matrix = confusion_matrix(y_true, y_pred, labels=LABELS)
    confused_pair, confused_count = _most_confused_pair(matrix.tolist())
    latency = _benchmark_single_inference(classifier, texts)
    report = {
        "accuracy": accuracy,
        "per_class_metrics": {
            label: {
                "precision": cls_report[label]["precision"],
                "recall": cls_report[label]["recall"],
                "f1": cls_report[label]["f1-score"],
            }
            for label in LABELS
        },
        "confusion_matrix": matrix.tolist(),
        "label_order": LABELS,
        "most_confused_pair": confused_pair,
        "most_confused_count": confused_count,
        "most_confused_examples": _most_confused_examples(rows, y_true, y_pred, confused_pair),
        "latency_ms": {
            "startup": round(startup_ms, 2),
            **latency,
        },
    }
    report_path = DATA_DIR / "eval_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    report["report_path"] = str(report_path)
    return report


def _most_confused_pair(matrix: list[list[int]]) -> tuple[tuple[str, str], int]:
    best_pair = (LABELS[0], LABELS[1])
    best_count = -1
    for left, right in product(range(len(LABELS)), repeat=2):
        if left >= right:
            continue
        count = matrix[left][right] + matrix[right][left]
        if count > best_count:
            best_pair = (LABELS[left], LABELS[right])
            best_count = count
    return best_pair, best_count


def _most_confused_examples(
    rows: list[dict[str, str]],
    y_true: list[str],
    y_pred: list[str],
    confused_pair: tuple[str, str],
    limit: int = 3,
) -> list[dict[str, str]]:
    left, right = confused_pair
    examples: list[dict[str, str]] = []
    for row, truth, prediction in zip(rows, y_true, y_pred, strict=True):
        if {truth, prediction} != {left, right}:
            continue
        if truth == prediction:
            continue
        examples.append(
            {
                "text": row["text"],
                "true_label": truth,
                "predicted_label": prediction,
            }
        )
        if len(examples) == limit:
            break
    return examples


def _benchmark_single_inference(
    classifier: TicketClassifier,
    texts: list[str],
) -> dict[str, float]:
    latencies: list[float] = []
    for text in texts:
        start = time.perf_counter()
        classifier.predict(text)
        latencies.append((time.perf_counter() - start) * 1000)
    ordered = sorted(latencies)
    return {
        "p50": round(median(ordered), 2),
        "p95": round(_percentile(ordered, 0.95), 2),
        "max": round(max(ordered), 2),
    }


def _percentile(values: list[float], quantile: float) -> float:
    if not values:
        return 0.0
    index = max(0, min(len(values) - 1, int(round((len(values) - 1) * quantile))))
    return values[index]


def main() -> None:
    report = evaluate_classifier()
    print(f"Accuracy: {report['accuracy']:.2%}")
    print("Per-class metrics:")
    for label, metrics in report["per_class_metrics"].items():
        print(
            f"- {label}: precision={metrics['precision']:.3f}, "
            f"recall={metrics['recall']:.3f}, f1={metrics['f1']:.3f}"
        )
    print("Confusion matrix (rows=true, cols=pred):")
    labels = report["label_order"]
    matrix = report["confusion_matrix"]
    col_w = max(len(lbl) for lbl in labels)
    header = " " * col_w + " | " + " | ".join(f"{lbl:>{col_w}}" for lbl in labels)
    print(header)
    print("-" * len(header))
    for label, row in zip(labels, matrix):
        cells = " | ".join(f"{v:>{col_w}}" for v in row)
        print(f"{label:>{col_w}} | {cells}")
    left, right = report["most_confused_pair"]
    print(f"Most confused pair: {left} vs {right} ({report['most_confused_count']} total confusions)")
    if report["most_confused_examples"]:
        print("Representative confusions:")
        for item in report["most_confused_examples"]:
            print(f"- true={item['true_label']} predicted={item['predicted_label']}: {item['text']}")
    latency = report["latency_ms"]
    print(
        "Warm single-ticket latency (ms): "
        f"p50={latency['p50']:.2f}, p95={latency['p95']:.2f}, max={latency['max']:.2f}"
    )
    print(f"Classifier startup (ms): {latency['startup']:.2f}")
    print(f"Saved evaluation report to {report['report_path']}")


if __name__ == "__main__":
    main()
