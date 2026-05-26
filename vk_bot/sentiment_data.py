from __future__ import annotations

import csv
import json
import math
from pathlib import Path
import xml.etree.ElementTree as ET

import pandas as pd
#from datasets import Dataset, DatasetDict
from sklearn.model_selection import train_test_split


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
    frame = frame[frame["text"].notna()]
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


def load_rus_sen_dat(root: Path) -> pd.DataFrame:
    path = root / "rus_sen_dat" / "datasets.csv"
    part_paths = sorted((root / "rus_sen_dat" / "parts").glob("datasets_part_*.csv"))
    if part_paths:
        frame = pd.concat([pd.read_csv(part_path) for part_path in part_paths], ignore_index=True)
    elif path.exists():
        frame = pd.read_csv(path)
    else:
        return pd.DataFrame(columns=["text", "label", "source"])
    if not {"text", "sentiment"}.issubset(frame.columns):
        raise ValueError("rus_sen_dat must contain text and sentiment columns.")
    label_map = {
        0: "neutral",
        1: "positive",
        2: "negative",
        "0": "neutral",
        "1": "positive",
        "2": "negative",
    }
    frame = frame.rename(columns={"sentiment": "label"})
    frame["label"] = frame["label"].map(label_map)
    return clean_frame(frame, "rus-sen-dat")


def load_kin_sen_dat(root: Path) -> pd.DataFrame:
    dataset_root = root / "kin_set_dat"
    if not dataset_root.exists():
        dataset_root = root / "kin_sen_dat"
    if not dataset_root.exists():
        return pd.DataFrame(columns=["text", "label", "source"])

    folder_labels = {
        "neg": "negative",
        "negative": "negative",
        "neu": "neutral",
        "neutral": "neutral",
        "pos": "positive",
        "positive": "positive",
    }
    rows = []
    for folder_name, label in folder_labels.items():
        folder = dataset_root / folder_name
        if not folder.exists():
            continue
        for path in folder.rglob("*.txt"):
            try:
                text = path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                text = path.read_text(encoding="cp1251")
            rows.append({"text": text, "label": label})
    if not rows:
        return pd.DataFrame(columns=["text", "label", "source"])
    return clean_frame(pd.DataFrame(rows), "kin-sen-dat")


LOADERS = {
    "rusentiment": load_rusentiment,
    "rureviews": load_rureviews,
    "kaggle-news": load_kaggle_news,
    "rutweetcorp": load_rutweetcorp,
    "sentirueval-entity": load_sentirueval_entity_xml,
    "sentirueval-review": load_sentirueval_review_xml,
    "linis": load_linis,
    "rus-sen-dat": load_rus_sen_dat,
    "kin-sen-dat": load_kin_sen_dat,
}
DEFAULT_SOURCES = sorted(LOADERS)


def sample_per_label(data: pd.DataFrame, max_per_label: int | None, seed: int = 42) -> pd.DataFrame:
    if not max_per_label:
        return data
    return pd.concat(
        [
            group.sample(min(len(group), max_per_label), random_state=seed)
            for _, group in data.groupby("label")
        ],
        ignore_index=True,
    )


def build_sentiment_dataframe(
    data_dir: str | Path = ".",
    sources: list[str] | None = None,
    max_per_label: int | None = None,
    seed: int = 42,
) -> pd.DataFrame:
    root = Path(data_dir)
    selected_sources = sources or DEFAULT_SOURCES
    frames = [LOADERS[source](root) for source in selected_sources]
    data = pd.concat(frames, ignore_index=True)
    data = data.drop_duplicates(subset=["text"]).reset_index(drop=True)
    if data.empty:
        raise ValueError("No training examples found. Check data_dir and sources.")
    return sample_per_label(data, max_per_label, seed)


def load_prepared_dataframe(
    path: str | Path,
    max_per_label: int | None = None,
    seed: int = 42,
) -> pd.DataFrame:
    csv_path = Path(path)
    frame = pd.read_csv(csv_path)
    missing = {"text", "label"} - set(frame.columns)
    if missing:
        raise ValueError(f"Prepared dataset is missing columns: {', '.join(sorted(missing))}")
    if "source" not in frame.columns:
        frame["source"] = csv_path.stem
    data = clean_frame(frame[["text", "label", "source"]], None)
    return sample_per_label(data, max_per_label, seed)


def save_sentiment_dataframe(data: pd.DataFrame, path: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(output_path, index=False)


def print_dataset_report(data: pd.DataFrame) -> None:
    print("Loaded examples:")
    print(data["label"].value_counts().reindex(LABELS, fill_value=0).to_string())
    print("\nBy source:")
    print(pd.crosstab(data["source"], data["label"]).reindex(columns=LABELS, fill_value=0).to_string())


def make_hf_dataset(data: pd.DataFrame, validation_size: float, seed: int) -> DatasetDict:
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
