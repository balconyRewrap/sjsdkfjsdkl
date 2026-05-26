from __future__ import annotations

import html
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd
from datasets import Dataset, DatasetDict, load_dataset
from sklearn.model_selection import train_test_split


SAFETY_LABELS = ["safe", "dangerous"]
SAFETY_LABEL2ID = {label: idx for idx, label in enumerate(SAFETY_LABELS)}
SAFETY_ID2LABEL = {idx: label for label, idx in SAFETY_LABEL2ID.items()}

DEFAULT_SAFETY_SOURCES = [
    "aegis-safety-ru",
    "local-safety-seed",
    "local-sentiment-safe",
    "nsfw-benchmark-ru",
    "mlsnt-ru",
    "ru-ethno-hate-extended",
]
TOXIC_SAFETY_SOURCES = {
    "klamas-russian-toxic",
    "multilingvalltoxic-ru",
    "ru-merged-toxic-comments",
}
EXPLICIT_EXTREMISM_RE = re.compile(
    r"(?<![а-яёa-z0-9])("
    r"зиг\s*хайль|"
    r"росси[яи]\s+для\s+русских|"
    r"слава\s+руси|"
    r"запрещ[её]нн\w+\s+(?:организац|движен)|"
    r"экстремистск\w+\s+(?:организац|движен|идеолог|материал)|"
    r"нацистск\w+\s+(?:лозунг|символ|идеолог)"
    r")(?![а-яёa-z0-9])",
    flags=re.IGNORECASE,
)
ETHNO_SLUR_RE = re.compile(
    r"""
    (?<![а-яёa-z0-9])
    (
        чур(?:к[аиуеойамих]|бан|баны|банам|банами|банах|чат|чата|чатье)?
      | хач(?:ик|ики|ика|ей|ам|ами|ах|ье)?
      | жид(?:ы|ов|ам|ами|ах|енок|яр[аы]?|овск\w*)?
      | нигг(?:ер|еры|еров|ерам|ерами|ерах|а|и)?
      | черножоп(?:ый|ые|ых|ым|ыми|ая|ое|ого|ому)?
      | узкоглаз(?:ый|ые|ых|ым|ыми|ая|ое|ого|ому)?
      | косоглаз(?:ый|ые|ых|ым|ыми|ая|ое|ого|ому)?
      | обезьян(?:а|ы|ам|ами|ах|ий|ьи|ье)
      | цыган(?:ье|ва|вы|ам|ами|ах|ский|ские|ских)?
      | черномаз(?:ый|ые|ых|ым|ыми|ая|ое)?
      | черноморд(?:ый|ые|ых|ым|ыми|ая|ое)?
      | кавказ(?:ня|оид(?:ы|ов|ам|ами|ах)?)
      | хохл(?:ы|ов|ам|ами|ах|ина|остан|яцк\w*)?
      | кацап(?:ы|ов|ам|ами|ах|ский|ские|ских)?
      | москал(?:ь|и|ей|ям|ями|ях)?
      | укроп(?:ы|ов|ам|ами|ах)?
      | пшек(?:и|ов|ам|ами|ах)?
      | бульбаш(?:и|ей|ам|ами|ах)?
      | чух(?:онец|онцы|онцев|на|ны|нам|нами|нах)?
      | лабус(?:ы|ов|ам|ами|ах)?
      | чурбан(?:ы|ов|ам|ами|ах)?
      | недолюд(?:и|ей|ям|ями|ях)?
      | нелюд(?:и|ей|ям|ями|ях)?
      | биомусор
      | генетическ(?:ий|ие|их|ими|ая|ое)\s+мусор
    )
    (?![а-яёa-z0-9])
    """,
    flags=re.IGNORECASE | re.UNICODE | re.VERBOSE,
)
VIOLENT_OR_EXCLUSION_RE = re.compile(
    r"""
    (
        уби(?:ть|вать|вайте|вай|ли|л|й)
      | бей(?:те)?
      | бить
      | резать
      | мочить
      | давить
      | травить
      | жечь
      | сжечь
      | насили\w*
      | напад(?:ать|ение|айте)
      | расстрел(?:ять|ивать|яйте)?
      | уничтож(?:ить|ать|айте)
      | истреб(?:ить|лять|ляйте)
      | вырез(?:ать|айте)
      | депорт(?:ировать|ируйте|ация)
      | выгн(?:ать|ать|ите|али)
      | изгн(?:ать|ание)
      | запрет(?:ить|ите)
      | лишить\s+прав
      | очистить\s+(?:город|страну|район|улиц\w*)
      | не\s+должны\s+(?:жить|существовать|иметь\s+права)
    )
    """,
    flags=re.IGNORECASE | re.UNICODE | re.VERBOSE,
)
SAFETY_REGEX_TRANSLATION = str.maketrans(
    {
        "a": "а",
        "e": "е",
        "o": "о",
        "p": "р",
        "c": "с",
        "x": "х",
        "y": "у",
        "k": "к",
        "h": "н",
        "3": "з",
        "4": "ч",
        "0": "о",
        "1": "и",
        "@": "а",
    }
)


def has_cyrillic_text(text: str, min_ratio: float = 0.15) -> bool:
    letters = [char for char in text if char.isalpha()]
    if not letters:
        return False
    cyrillic = [char for char in letters if "а" <= char.lower() <= "я" or char.lower() == "ё"]
    return len(cyrillic) / len(letters) >= min_ratio


def clean_safety_frame(frame: pd.DataFrame, source: str | None) -> pd.DataFrame:
    frame = frame.copy()
    frame = frame[frame["text"].notna()]
    frame["text"] = frame["text"].astype(str).str.replace(r"\s+", " ", regex=True).str.strip()
    frame = frame[frame["text"].ne("")]
    frame["label"] = frame["label"].astype(str).str.strip().str.lower()
    frame = frame[frame["label"].isin(SAFETY_LABELS)]
    if "risk_category" not in frame.columns:
        frame["risk_category"] = frame["label"]
    frame["risk_category"] = frame["risk_category"].fillna("unknown").astype(str).str.strip()
    if source is not None:
        frame["source"] = source
    elif "source" not in frame.columns:
        frame["source"] = "prepared"
    return frame[["text", "label", "risk_category", "source"]]


def clean_html_text(text: str) -> str:
    text = html.unescape(str(text))
    text = re.sub(r"<br\s*/?>", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"<[^>]+>", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def normalize_for_safety_regex(text: str) -> str:
    text = text.lower().replace("ё", "е").translate(SAFETY_REGEX_TRANSLATION)
    return re.sub(r"(?<=[а-яе])[\s._*\-]+(?=[а-яе])", "", text)


def ethno_high_risk_score(text: str) -> dict[str, bool | str]:
    text_norm = normalize_for_safety_regex(text)
    has_explicit_extremism = bool(EXPLICIT_EXTREMISM_RE.search(text_norm))
    has_slur = bool(ETHNO_SLUR_RE.search(text_norm))
    has_violence = bool(VIOLENT_OR_EXCLUSION_RE.search(text_norm))
    unsafe_high_confidence = has_explicit_extremism or (has_slur and has_violence)

    if unsafe_high_confidence:
        label = "unsafe_high_confidence"
    elif has_slur:
        label = "ethno_slur_flag"
    elif has_violence:
        label = "violence_or_exclusion_flag"
    else:
        label = "no_ethno_high_risk_flag"

    return {
        "label": label,
        "unsafe_high_confidence": unsafe_high_confidence,
        "has_explicit_extremism": has_explicit_extremism,
        "has_ethno_slur": has_slur,
        "has_violence_or_exclusion": has_violence,
    }


def has_ethno_high_risk_pattern(text: str) -> bool:
    return bool(ethno_high_risk_score(text)["unsafe_high_confidence"])


def load_aegis_safety_ru(cache_dir: str | Path | None = ".hf-cache") -> pd.DataFrame:
    dataset = load_dataset("katanastas/aegis-safety-ru", cache_dir=cache_dir)
    rows = []
    for split in dataset.values():
        for item in split:
            prompt_label = str(item.get("prompt_label", "")).strip().lower()
            if prompt_label not in {"safe", "unsafe"}:
                continue
            rows.append(
                {
                    "text": item["text"],
                    "label": "dangerous" if prompt_label == "unsafe" else "safe",
                    "risk_category": item.get("primary_cat") or prompt_label,
                }
            )
    return clean_safety_frame(pd.DataFrame(rows), "aegis-safety-ru")


def load_nsfw_benchmark_ru(cache_dir: str | Path | None = ".hf-cache") -> pd.DataFrame:
    dataset = load_dataset("redmadrobot-rnd/nsfw_benchmark", cache_dir=cache_dir)
    rows = []
    for split in dataset.values():
        for item in split:
            if item.get("language") != "ru":
                continue
            label = int(item["label"])
            rows.append(
                {
                    "text": item["text"],
                    "label": "dangerous" if label == 1 else "safe",
                    "risk_category": item.get("category") or "unsafe",
                }
            )
    return clean_safety_frame(pd.DataFrame(rows), "nsfw-benchmark-ru")


def load_mlsnt_ru(cache_dir: str | Path | None = ".hf-cache") -> pd.DataFrame:
    dataset = load_dataset("ComplexDataLab/MLSNT", cache_dir=cache_dir)
    rows = []
    for split in dataset.values():
        for item in split:
            text = item.get("full_text") or ""
            if not has_cyrillic_text(text):
                continue
            category_id = item.get("min_category_id", item.get("category_id", 0))
            if isinstance(category_id, list):
                category_id = min(category_id) if category_id else 0
            labels = item.get("final_label") or []
            risk_category = "; ".join(labels) if labels else "safe"
            rows.append(
                {
                    "text": text,
                    "label": "safe" if int(category_id) == 0 else "dangerous",
                    "risk_category": risk_category,
                }
            )
    return clean_safety_frame(pd.DataFrame(rows), "mlsnt-ru")


def load_ru_ethno_hate_extended(cache_dir: str | Path | None = ".hf-cache") -> pd.DataFrame:
    paths = [
        Path("RuRthnoHateExtended") / "RuEthnoHateExtended.json",
        Path("RuEthnoHateExtended") / "RuEthnoHateExtended.json",
    ]
    path = next((candidate for candidate in paths if candidate.exists()), None)
    if path is None:
        return pd.DataFrame(columns=["text", "label", "risk_category", "source"])

    raw_rows = json.loads(path.read_text(encoding="utf-8"))
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in raw_rows:
        if row.get("does_text_make_sense") != "yes":
            continue
        if row.get("text_has_ethnonyms") != "yes":
            continue
        label = row.get("class")
        if label not in {-1, 0, 1, "-1", "0", "1"}:
            continue
        grouped[str(row["instance_id"])].append(row)

    rows = []
    for instance_rows in grouped.values():
        labels = [int(row["class"]) for row in instance_rows]
        counts = Counter(labels)
        if len(counts) > 1 and counts.most_common(2)[0][1] == counts.most_common(2)[1][1]:
            continue
        majority_label = counts.most_common(1)[0][0]
        text = clean_html_text(str(instance_rows[0]["text"]))
        rows.append(
            {
                "text": text,
                "label": "dangerous" if majority_label == -1 else "safe",
                "risk_category": "ethnic_hate" if majority_label == -1 else "ethnic_non_hate",
            }
        )
    frame = clean_safety_frame(pd.DataFrame(rows), "ru-ethno-hate-extended")
    if frame.empty:
        return frame

    deduplicated_rows = []
    for text, group in frame.groupby("text", sort=False):
        has_hate_label = group["label"].eq("dangerous").any()
        high_risk_score = ethno_high_risk_score(text)
        has_high_risk_pattern = bool(high_risk_score["unsafe_high_confidence"])
        is_dangerous = has_hate_label or has_high_risk_pattern
        deduplicated_rows.append(
            {
                "text": text,
                "label": "dangerous" if is_dangerous else "safe",
                "risk_category": (
                    "ethnic_hate"
                    if has_hate_label
                    else "ethnic_explicit_extremism_pattern"
                    if high_risk_score["has_explicit_extremism"]
                    else "ethnic_slur_violence_pattern"
                    if has_high_risk_pattern
                    else "ethnic_non_hate"
                ),
                "source": "ru-ethno-hate-extended",
            }
        )
    return pd.DataFrame(deduplicated_rows)


def load_local_sentiment_safe(cache_dir: str | Path | None = ".hf-cache") -> pd.DataFrame:
    from sentiment_data import build_sentiment_dataframe

    sentiment_sources = [
        "kaggle-news",
        "rureviews",
        "rusentiment",
        "rutweetcorp",
        "sentirueval-review",
    ]
    data = build_sentiment_dataframe(".", sentiment_sources, max_per_label=1500)
    frame = pd.DataFrame(
        {
            "text": data["text"],
            "label": "safe",
            "risk_category": "ordinary_sentiment_text",
        }
    )
    return clean_safety_frame(frame, "local-sentiment-safe")


def load_local_safety_seed(cache_dir: str | Path | None = ".hf-cache") -> pd.DataFrame:
    path = Path("data") / "safety_seed_examples.csv"
    if not path.exists():
        return pd.DataFrame(columns=["text", "label", "risk_category", "source"])
    return clean_safety_frame(pd.read_csv(path), None)


def load_ru_merged_toxic_comments(cache_dir: str | Path | None = ".hf-cache") -> pd.DataFrame:
    dataset = load_dataset("Xeonil/ru-merged-toxic-comments", cache_dir=cache_dir)
    rows = []
    for split in dataset.values():
        for item in split:
            label = int(item["target"])
            rows.append(
                {
                    "text": item["text"],
                    "label": "dangerous" if label == 1 else "safe",
                    "risk_category": "toxic_comment" if label == 1 else "non_toxic_comment",
                }
            )
    return clean_safety_frame(pd.DataFrame(rows), "ru-merged-toxic-comments")


def load_klamas_russian_toxic(cache_dir: str | Path | None = ".hf-cache") -> pd.DataFrame:
    dataset = load_dataset("klamas/russian-toxic", cache_dir=cache_dir)
    rows = []
    for split in dataset.values():
        for item in split:
            label = int(item["label"])
            rows.append(
                {
                    "text": item["text"],
                    "label": "dangerous" if label == 1 else "safe",
                    "risk_category": "toxic_comment" if label == 1 else "non_toxic_comment",
                }
            )
    return clean_safety_frame(pd.DataFrame(rows), "klamas-russian-toxic")


def load_multilingvalltoxic_ru(cache_dir: str | Path | None = ".hf-cache") -> pd.DataFrame:
    dataset = load_dataset("Mikimi/MultiLingvAllToxic", cache_dir=cache_dir)
    rows = []
    for split in dataset.values():
        for item in split:
            language = str(item.get("language", ""))
            source_lang = str(item.get("source_lang", ""))
            if not language.startswith("ru") and source_lang != "ru":
                continue
            label = int(item["toxic_binary"])
            toxicity_type = item.get("toxicity_type") or "toxic_comment"
            rows.append(
                {
                    "text": item["text"],
                    "label": "dangerous" if label == 1 else "safe",
                    "risk_category": toxicity_type if label == 1 else "non_toxic_comment",
                }
            )
    return clean_safety_frame(pd.DataFrame(rows), "multilingvalltoxic-ru")


SAFETY_LOADERS = {
    "aegis-safety-ru": load_aegis_safety_ru,
    "klamas-russian-toxic": load_klamas_russian_toxic,
    "local-safety-seed": load_local_safety_seed,
    "local-sentiment-safe": load_local_sentiment_safe,
    "multilingvalltoxic-ru": load_multilingvalltoxic_ru,
    "nsfw-benchmark-ru": load_nsfw_benchmark_ru,
    "mlsnt-ru": load_mlsnt_ru,
    "ru-ethno-hate-extended": load_ru_ethno_hate_extended,
    "ru-merged-toxic-comments": load_ru_merged_toxic_comments,
}


def sample_per_label(data: pd.DataFrame, max_per_label: int | None, seed: int = 42) -> pd.DataFrame:
    if not max_per_label:
        return data
    frames = []
    for _, group in data.groupby("label"):
        required = group[group["source"].eq("local-safety-seed")]
        remaining = group[~group.index.isin(required.index)]
        sample_size = max(max_per_label - len(required), 0)
        sampled = remaining.sample(min(len(remaining), sample_size), random_state=seed)
        frames.append(pd.concat([required, sampled], ignore_index=True))
    return pd.concat(frames, ignore_index=True)


def limit_toxic_share(data: pd.DataFrame, max_toxic_share: float | None, seed: int = 42) -> pd.DataFrame:
    if max_toxic_share is None:
        return data
    if not 0 <= max_toxic_share <= 1:
        raise ValueError("--max-toxic-share must be between 0 and 1.")

    toxic_mask = data["source"].isin(TOXIC_SAFETY_SOURCES)
    toxic = data[toxic_mask]
    other = data[~toxic_mask]
    if toxic.empty or max_toxic_share >= 1:
        return data
    if max_toxic_share == 0:
        return other.reset_index(drop=True)

    max_toxic_rows = int(len(other) * max_toxic_share / (1 - max_toxic_share))
    if len(toxic) <= max_toxic_rows:
        return data
    toxic_sample = toxic.sample(max_toxic_rows, random_state=seed)
    return pd.concat([other, toxic_sample], ignore_index=True)


def balance_safety_labels(data: pd.DataFrame, balance_ratio: float | None, seed: int = 42) -> pd.DataFrame:
    if balance_ratio is None:
        return data
    if balance_ratio <= 0:
        raise ValueError("--balance-ratio must be greater than 0.")

    safe = data[data["label"].eq("safe")]
    dangerous = data[data["label"].eq("dangerous")]
    if safe.empty or dangerous.empty:
        return data

    max_safe_rows = max(int(round(len(dangerous) * balance_ratio)), 1)
    if len(safe) <= max_safe_rows:
        return data

    required = safe[safe["source"].eq("local-safety-seed")]
    remaining = safe[~safe.index.isin(required.index)]
    sample_size = max(max_safe_rows - len(required), 0)
    sampled_safe = remaining.sample(min(len(remaining), sample_size), random_state=seed)
    return pd.concat([dangerous, required, sampled_safe], ignore_index=True)


def build_safety_dataframe(
    sources: list[str] | None = None,
    max_per_label: int | None = None,
    max_toxic_share: float | None = None,
    balance_ratio: float | None = None,
    seed: int = 42,
    cache_dir: str | Path | None = ".hf-cache",
) -> pd.DataFrame:
    selected_sources = sources or DEFAULT_SAFETY_SOURCES
    frames = [SAFETY_LOADERS[source](cache_dir) for source in selected_sources]
    data = pd.concat(frames, ignore_index=True)
    data = data.drop_duplicates(subset=["text"]).reset_index(drop=True)
    data = limit_toxic_share(data, max_toxic_share, seed)
    if data.empty:
        raise ValueError("No safety training examples found. Check selected sources.")
    data = sample_per_label(data, max_per_label, seed)
    data = limit_toxic_share(data, max_toxic_share, seed)
    return balance_safety_labels(data, balance_ratio, seed)


def load_prepared_safety_dataframe(
    path: str | Path,
    max_per_label: int | None = None,
    balance_ratio: float | None = None,
    seed: int = 42,
) -> pd.DataFrame:
    csv_path = Path(path)
    frame = pd.read_csv(csv_path)
    missing = {"text", "label"} - set(frame.columns)
    if missing:
        raise ValueError(f"Prepared safety dataset is missing columns: {', '.join(sorted(missing))}")
    if "source" not in frame.columns:
        frame["source"] = csv_path.stem
    if "risk_category" not in frame.columns:
        frame["risk_category"] = frame["label"]
    data = clean_safety_frame(frame[["text", "label", "risk_category", "source"]], None)
    data = sample_per_label(data, max_per_label, seed)
    return balance_safety_labels(data, balance_ratio, seed)


def save_safety_dataframe(data: pd.DataFrame, path: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(output_path, index=False)


def print_safety_dataset_report(data: pd.DataFrame) -> None:
    print("Loaded safety examples:")
    print(data["label"].value_counts().reindex(SAFETY_LABELS, fill_value=0).to_string())
    print("\nBy source:")
    print(pd.crosstab(data["source"], data["label"]).reindex(columns=SAFETY_LABELS, fill_value=0).to_string())
    print("\nTop risk categories:")
    print(data["risk_category"].value_counts().head(20).to_string())


def make_safety_hf_dataset(data: pd.DataFrame, validation_size: float, seed: int) -> DatasetDict:
    min_validation_rows = data["label"].nunique()
    validation_rows = max(round(len(data) * validation_size), min_validation_rows)
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
        frame["label"] = frame["label"].map(SAFETY_LABEL2ID).astype(int)
    return DatasetDict(
        {
            "train": Dataset.from_pandas(train_df[["text", "label"]], preserve_index=False),
            "validation": Dataset.from_pandas(valid_df[["text", "label"]], preserve_index=False),
        }
    )
