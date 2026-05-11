from __future__ import annotations

import argparse
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


DEFAULT_TEXTS = [
    "Отличный товар, все понравилось, буду заказывать еще.",
    "Ужасное качество, деньги на ветер.",
    "Посылка пришла вчера, размер соответствует описанию.",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run inference with a trained Russian sentiment model.")
    parser.add_argument("--model-dir", type=Path, default=Path("models/rubert-sentiment"))
    parser.add_argument("--text", nargs="+", help="One or more texts to classify.")
    parser.add_argument("--device", default=None, choices=["cpu", "cuda"], help="Override inference device.")
    parser.add_argument("--max-length", type=int, default=192)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    texts = args.text or DEFAULT_TEXTS

    if not args.model_dir.exists():
        raise SystemExit(f"Model directory does not exist: {args.model_dir}")

    device_name = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_name)

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_dir).to(device)
    model.eval()

    encoded = tokenizer(
        texts,
        truncation=True,
        padding=True,
        max_length=args.max_length,
        return_tensors="pt",
    ).to(device)

    with torch.no_grad():
        logits = model(**encoded).logits
        probabilities = torch.softmax(logits, dim=-1).cpu()

    id2label = model.config.id2label
    print(f"device: {device}")
    for text, probs in zip(texts, probabilities, strict=True):
        best_id = int(probs.argmax())
        label = id2label.get(best_id, str(best_id))
        confidence = float(probs[best_id])
        all_scores = ", ".join(
            f"{id2label.get(idx, str(idx))}={float(score):.3f}" for idx, score in enumerate(probs)
        )
        print()
        print(f"text: {text}")
        print(f"prediction: {label} ({confidence:.3f})")
        print(f"scores: {all_scores}")


if __name__ == "__main__":
    main()
