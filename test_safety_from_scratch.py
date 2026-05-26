from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from train_from_scratch import ScratchTransformerClassifier, make_collate_fn


DEFAULT_TEXTS = [
    "Как приготовить борщ дома?",
    "Я ненавижу эту доставку, сервис ужасный.",
    "Нужно устроить нападение на группу людей.",
]

DEFAULT_MODEL_DIR = Path("models/scratch-transformer-safety")
BIG_MODEL_DIR = Path("models/scratch-transformer-safety-big")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run inference with a from-scratch safety model.")
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=None,
        help="Path to model artifacts. Overrides --model-size.",
    )
    parser.add_argument(
        "--model-size",
        choices=["auto", "base", "big"],
        default="auto",
        help="Model preset to load when --model-dir is not set. auto prefers big if it exists.",
    )
    parser.add_argument("--text", nargs="+", help="One or more texts to classify.")
    parser.add_argument("--input-file", type=Path, help="UTF-8 text file.")
    parser.add_argument(
        "--input-format",
        choices=["line", "paragraph", "whole"],
        default="line",
        help="How to split --input-file.",
    )
    parser.add_argument("--dangerous-threshold", type=float, default=0.5)
    parser.add_argument("--device", choices=["cpu", "cuda"], default=None)
    return parser.parse_args()


def has_model_artifacts(model_dir: Path) -> bool:
    return all(
        (model_dir / filename).exists()
        for filename in ("config.json", "vocab.json", "model.pt")
    )


def resolve_model_dir(args: argparse.Namespace) -> Path:
    if args.model_dir is not None:
        return args.model_dir

    if args.model_size == "big":
        return BIG_MODEL_DIR

    if args.model_size == "base":
        return DEFAULT_MODEL_DIR

    if has_model_artifacts(BIG_MODEL_DIR):
        return BIG_MODEL_DIR

    return DEFAULT_MODEL_DIR


def load_texts(args: argparse.Namespace) -> list[str]:
    if args.text and args.input_file:
        raise SystemExit("Use either --text or --input-file, not both.")
    if args.input_file:
        content = args.input_file.read_text(encoding="utf-8")
        if args.input_format == "whole":
            return [content.strip()] if content.strip() else []
        if args.input_format == "paragraph":
            return [" ".join(part.split()) for part in content.split("\n\n") if part.strip()]
        return [line.strip() for line in content.splitlines() if line.strip()]
    return args.text or DEFAULT_TEXTS


def main() -> None:
    args = parse_args()
    model_dir = resolve_model_dir(args)
    texts = load_texts(args)
    if not texts:
        raise SystemExit("No texts to classify.")

    config_path = model_dir / "config.json"
    vocab_path = model_dir / "vocab.json"
    weights_path = model_dir / "model.pt"
    if not config_path.exists() or not vocab_path.exists() or not weights_path.exists():
        raise SystemExit(
            f"Missing safety model artifacts in {model_dir}. "
            "Use --model-dir or train with --output-dir."
        )

    config = json.loads(config_path.read_text(encoding="utf-8"))
    vocab = json.loads(vocab_path.read_text(encoding="utf-8"))
    id2label = {int(key): value for key, value in config["id2label"].items()}
    label2id = config["label2id"]
    dangerous_id = int(label2id["dangerous"])
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    model = ScratchTransformerClassifier(
        vocab_size=config["vocab_size"],
        num_labels=len(config["labels"]),
        max_length=config["max_length"],
        d_model=config["d_model"],
        nhead=config["nhead"],
        num_layers=config["num_layers"],
        dim_feedforward=config["dim_feedforward"],
        dropout=config["dropout"],
    ).to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()

    collate = make_collate_fn(vocab, config["max_length"])
    batch = collate([(text, 0) for text in texts])
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)

    with torch.no_grad():
        probabilities = torch.softmax(model(input_ids, attention_mask), dim=-1).cpu()

    print(f"model_dir: {model_dir}")
    print(f"device: {device}")
    for text, probs in zip(texts, probabilities, strict=True):
        dangerous_score = float(probs[dangerous_id])
        if dangerous_score >= args.dangerous_threshold:
            label = "dangerous"
            confidence = dangerous_score
            decision = "manual_review"
        else:
            label = "safe"
            confidence = float(probs[label2id["safe"]])
            decision = "allow"
        scores = ", ".join(f"{id2label.get(idx, str(idx))}={float(score):.3f}" for idx, score in enumerate(probs))
        print()
        print(f"text: {text}")
        print(f"prediction: {label} ({confidence:.3f})")
        print(f"decision: {decision}")
        print(f"scores: {scores}")


if __name__ == "__main__":
    main()
