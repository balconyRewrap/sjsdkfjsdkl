from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    set_seed,
)

from sentiment_data import (
    DEFAULT_SOURCES,
    ID2LABEL,
    LABEL2ID,
    LABELS,
    build_sentiment_dataframe,
    load_prepared_dataframe,
    make_hf_dataset,
    print_dataset_report,
    save_sentiment_dataframe,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune Russian positive/negative/neutral classifier.")
    parser.add_argument("--data-dir", type=Path, default=Path("."), help="Dataset repository root.")
    parser.add_argument("--dataset-csv", type=Path, help="Use prepared CSV/CSV.GZ with text,label columns.")
    parser.add_argument(
        "--sources",
        nargs="+",
        choices=DEFAULT_SOURCES,
        default=DEFAULT_SOURCES,
        help="Datasets to combine.",
    )
    parser.add_argument(
        "--model-name",
        default="cointegrated/rubert-tiny2",
        help="Hugging Face model checkpoint. Try ai-forever/ruBert-base for higher quality.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("models/rubert-sentiment"))
    parser.add_argument("--max-length", type=int, default=192)
    parser.add_argument("--epochs", type=float, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--validation-size", type=float, default=0.1)
    parser.add_argument("--max-per-label", type=int, default=None, help="Cap examples per class for quick runs.")
    parser.add_argument("--save-dataset", type=Path, help="Save combined text,label,source CSV before training.")
    parser.add_argument("--only-build-dataset", action="store_true", help="Build/save dataset and exit without training.")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    if args.dataset_csv:
        data = load_prepared_dataframe(args.dataset_csv, args.max_per_label, args.seed)
    else:
        data = build_sentiment_dataframe(args.data_dir, args.sources, args.max_per_label, args.seed)
    print_dataset_report(data)

    if args.save_dataset:
        save_sentiment_dataframe(data, args.save_dataset)
        print(f"\nSaved combined dataset to {args.save_dataset}")

    if args.only_build_dataset:
        return

    dataset = make_hf_dataset(data, args.validation_size, args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    def tokenize(batch: dict[str, list[str]]) -> dict[str, list[list[int]]]:
        return tokenizer(batch["text"], truncation=True, max_length=args.max_length)

    tokenized = dataset.map(tokenize, batched=True, remove_columns=["text"])
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=len(LABELS),
        label2id=LABEL2ID,
        id2label=ID2LABEL,
    )

    def compute_metrics(eval_pred: tuple[np.ndarray, np.ndarray]) -> dict[str, float]:
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return {
            "accuracy": accuracy_score(labels, predictions),
            "macro_f1": f1_score(labels, predictions, average="macro"),
        }

    training_args = TrainingArguments(
        output_dir=str(args.output_dir),
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=100,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        report_to="none",
        seed=args.seed,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        processing_class=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_metrics,
    )
    trainer.train()
    metrics = trainer.evaluate()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(metrics)
    print(f"Saved model to {args.output_dir}")


if __name__ == "__main__":
    main()
