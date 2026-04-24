from __future__ import annotations

from .data_builder import ensure_datasets, load_eval_examples, load_train_examples


def main() -> None:
    train_path, eval_path = ensure_datasets()
    print(f"Training examples: {len(load_train_examples())} -> {train_path}")
    print(f"Evaluation examples: {len(load_eval_examples())} -> {eval_path}")


if __name__ == "__main__":
    main()
