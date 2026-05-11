from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
from datasets import Dataset, DatasetDict
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    set_seed,
)


LABELS = ["negative", "neutral", "positive"]
LABEL2ID = {label: idx for idx, label in enumerate(LABELS)}
ID2LABEL = {idx: label for label, idx in LABEL2ID.items()}


def normalize_label(value: object) -> str | None:
    if value is None or pd.isna(value):
        return None
    label = str(value).strip().lower()
    aliases = {
        "neg": "negative",
        "negative": "negative",
        "neautral": "neutral",
        "neutral": "neutral",
        "neu": "neutral",
        "positive": "positive",
        "pos": "positive",
    }
    return aliases.get(label)


def clean_frame(frame: pd.DataFrame, source: str | None) -> pd.DataFrame:
    frame = frame.copy()
    frame["label"] = frame["label"].map(normalize_label)
    frame["text"] = frame["text"].astype(str).str.replace(r"\s+", " ", regex=True).str.strip()
    frame = frame[frame["label"].isin(LABELS) & frame["text"].ne("")]
    if source is not None:
        frame["source"] = source
    elif "source" not in frame.columns:
        frame["source"] = "prepared"
    return frame[["text", "label", "source"]]


def load_rusentiment(root: Path) -> pd.DataFrame:
    files = [
        root / "RuSentiment" / "rusentiment_random_posts.csv",
        root / "RuSentiment" / "rusentiment_preselected_posts.csv",
        root / "RuSentiment" / "rusentiment_test.csv",
    ]
    frames = [pd.read_csv(path) for path in files if path.exists()]
    if not frames:
        return pd.DataFrame(columns=["text", "label", "source"])
    return clean_frame(pd.concat(frames, ignore_index=True), "rusentiment")


def load_rureviews(root: Path) -> pd.DataFrame:
    path = root / "RuReviews" / "women-clothing-accessories.3-class.balanced.csv"
    if not path.exists():
        return pd.DataFrame(columns=["text", "label", "source"])
    frame = pd.read_csv(path, sep="\t").rename(columns={"review": "text", "sentiment": "label"})
    return clean_frame(frame, "rureviews")


def load_kaggle_news(root: Path) -> pd.DataFrame:
    path = root / "Kaggle-Russian-News" / "train.json"
    if not path.exists():
        return pd.DataFrame(columns=["text", "label", "source"])
    with path.open(encoding="utf-8") as file:
        rows = json.load(file)
    frame = pd.DataFrame(rows).rename(columns={"sentiment": "label"})
    return clean_frame(frame, "kaggle-news")


def load_rutweetcorp(root: Path) -> pd.DataFrame:
    frames = []
    for filename, label in [("negative.csv", "negative"), ("positive.csv", "positive")]:
        path = root / "RuTweetCorp" / filename
        if not path.exists():
            continue
        frame = pd.read_csv(
            path,
            sep=";",
            header=None,
            quoting=csv.QUOTE_ALL,
            usecols=[3],
            names=["text"],
            encoding="utf-8",
        )
        frame["label"] = label
        frames.append(frame)
    if not frames:
        return pd.DataFrame(columns=["text", "label", "source"])
    return clean_frame(pd.concat(frames, ignore_index=True), "rutweetcorp")


def numeric_sentiment_to_label(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip().lower()
    if text in {"null", "", "nan"}:
        return None
    try:
        score = int(float(text.replace(",", ".")))
    except ValueError:
        return normalize_label(text)
    if score < 0:
        return "negative"
    if score > 0:
        return "positive"
    return "neutral"


def majority_label(labels: list[str]) -> str | None:
    if not labels:
        return None
    counts = pd.Series(labels).value_counts()
    if len(counts) > 1 and counts.iloc[0] == counts.iloc[1]:
        return None
    return str(counts.index[0])


def load_sentirueval_entity_xml(root: Path) -> pd.DataFrame:
    paths = [
        root / "SentiRuEval-2015-subtask-2" / "SentiRuEval-2015-banks" / "train.xml",
        root / "SentiRuEval-2015-subtask-2" / "SentiRuEval-2015-banks" / "test_etalon.xml",
        root / "SentiRuEval-2015-subtask-2" / "SentiRuEval-2015-telecoms" / "train.xml",
        root / "SentiRuEval-2015-subtask-2" / "SentiRuEval-2015-telecoms" / "test_etalon.xml",
        root / "SentiRuEval-2016" / "bank_train_2016.xml",
        root / "SentiRuEval-2016" / "banks_test_etalon.xml",
        root / "SentiRuEval-2016" / "tkk_train_2016.xml",
        root / "SentiRuEval-2016" / "tkk_test_etalon.xml",
    ]
    rows = []
    ignored_columns = {"id", "twitid", "date", "text"}
    for path in paths:
        if not path.exists():
            continue
        tree = ET.parse(path)
        for table in tree.findall(".//table"):
            values = {
                column.attrib.get("name"): (column.text or "")
                for column in table.findall("column")
            }
            text = values.get("text", "")
            labels = [
                label
                for name, value in values.items()
                if name not in ignored_columns
                for label in [numeric_sentiment_to_label(value)]
                if label
            ]
            label = majority_label(labels)
            if label:
                rows.append({"text": text, "label": label})
    if not rows:
        return pd.DataFrame(columns=["text", "label", "source"])
    return clean_frame(pd.DataFrame(rows), "sentirueval-entity")


def load_sentirueval_review_xml(root: Path) -> pd.DataFrame:
    paths = [
        root / "SentiRuEval-2015-subtask-1" / "SentiRuEval_car_markup_train.xml",
        root / "SentiRuEval-2015-subtask-1" / "SentiRuEval_car_markup_test.xml",
        root / "SentiRuEval-2015-subtask-1" / "SentiRuEval_rest_markup_train.xml",
        root / "SentiRuEval-2015-subtask-1" / "SentiRuEval_rest_markup_test.xml",
    ]
    rows = []
    for path in paths:
        if not path.exists():
            continue
        tree = ET.parse(path)
        for review in tree.findall(".//review"):
            text_node = review.find("text")
            text = "".join(text_node.itertext()) if text_node is not None else ""
            whole = [
                normalize_label(category.attrib.get("sentiment"))
                for category in review.findall("./categories/category")
                if category.attrib.get("name") == "Whole"
            ]
            labels = [label for label in whole if label]
            if not labels:
                labels = [
                    label
                    for category in review.findall("./categories/category")
                    for label in [normalize_label(category.attrib.get("sentiment"))]
                    if label
                ]
            label = majority_label(labels)
            if label:
                rows.append({"text": text, "label": label})
    if not rows:
        return pd.DataFrame(columns=["text", "label", "source"])
    return clean_frame(pd.DataFrame(rows), "sentirueval-review")


def load_linis(root: Path) -> pd.DataFrame:
    paths = [
        root / "Linis-Crowd-2015" / "text_rating_final.xlsx",
        root / "Linis-Crowd-2016" / "doc_comment_summary.xlsx",
    ]
    frames = []
    for path in paths:
        if not path.exists():
            continue
        frame = pd.read_excel(path, header=None, usecols=[0, 1], names=["text", "label"])
        frames.append(frame)
    if not frames:
        return pd.DataFrame(columns=["text", "label", "source"])
    frame = pd.concat(frames, ignore_index=True)
    frame["label"] = frame["label"].map(numeric_sentiment_to_label)
    return clean_frame(frame, "linis")


LOADERS = {
    "rusentiment": load_rusentiment,
    "rureviews": load_rureviews,
    "kaggle-news": load_kaggle_news,
    "rutweetcorp": load_rutweetcorp,
    "sentirueval-entity": load_sentirueval_entity_xml,
    "sentirueval-review": load_sentirueval_review_xml,
    "linis": load_linis,
}


def build_dataframe(root: Path, sources: list[str], max_per_label: int | None) -> pd.DataFrame:
    frames = [LOADERS[source](root) for source in sources]
    data = pd.concat(frames, ignore_index=True)
    data = data.drop_duplicates(subset=["text"]).reset_index(drop=True)
    if data.empty:
        raise ValueError("No training examples found. Check --data-dir and --sources.")

    if max_per_label:
        data = pd.concat(
            [
                group.sample(min(len(group), max_per_label), random_state=42)
                for _, group in data.groupby("label")
            ],
            ignore_index=True,
        )
    return data


def load_prepared_dataset(path: Path, max_per_label: int | None) -> pd.DataFrame:
    frame = pd.read_csv(path)
    missing = {"text", "label"} - set(frame.columns)
    if missing:
        raise ValueError(f"Prepared dataset is missing columns: {', '.join(sorted(missing))}")
    if "source" not in frame.columns:
        frame["source"] = path.stem
    data = clean_frame(frame[["text", "label", "source"]], None)
    if max_per_label:
        data = pd.concat(
            [
                group.sample(min(len(group), max_per_label), random_state=42)
                for _, group in data.groupby("label")
            ],
            ignore_index=True,
        )
    return data


def make_dataset(data: pd.DataFrame, validation_size: float, seed: int) -> DatasetDict:
    min_validation_rows = data["label"].nunique()
    validation_rows = max(math.ceil(len(data) * validation_size), min_validation_rows)
    if validation_rows >= len(data):
        raise ValueError("Validation split is too large for the number of loaded examples.")

    train_df, valid_df = train_test_split(
        data,
        test_size=validation_rows,
        random_state=seed,
        stratify=data["label"],
    )
    train_df = train_df.copy()
    valid_df = valid_df.copy()
    for frame in (train_df, valid_df):
        frame["label"] = frame["label"].map(LABEL2ID).astype(int)
    return DatasetDict(
        {
            "train": Dataset.from_pandas(train_df[["text", "label"]], preserve_index=False),
            "validation": Dataset.from_pandas(valid_df[["text", "label"]], preserve_index=False),
        }
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Russian positive/negative/neutral classifier.")
    parser.add_argument("--data-dir", type=Path, default=Path("."), help="Dataset repository root.")
    parser.add_argument("--dataset-csv", type=Path, help="Use prepared CSV/CSV.GZ with text,label columns.")
    parser.add_argument(
        "--sources",
        nargs="+",
        choices=sorted(LOADERS),
        default=sorted(LOADERS),
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
        data = load_prepared_dataset(args.dataset_csv, args.max_per_label)
    else:
        data = build_dataframe(args.data_dir, args.sources, args.max_per_label)
    print("Loaded examples:")
    print(data["label"].value_counts().reindex(LABELS, fill_value=0).to_string())
    print("\nBy source:")
    print(pd.crosstab(data["source"], data["label"]).reindex(columns=LABELS, fill_value=0).to_string())

    if args.save_dataset:
        args.save_dataset.parent.mkdir(parents=True, exist_ok=True)
        data.to_csv(args.save_dataset, index=False)
        print(f"\nSaved combined dataset to {args.save_dataset}")

    if args.only_build_dataset:
        return

    dataset = make_dataset(data, args.validation_size, args.seed)
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
