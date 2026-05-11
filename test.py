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
    parser.add_argument("--input-file", type=Path, help="UTF-8 text file with one text per line.")
    parser.add_argument(
        "--input-format",
        choices=["line", "paragraph", "whole"],
        default="line",
        help="How to split --input-file: line, paragraph separated by blank lines, or whole file.",
    )
    parser.add_argument("--device", default=None, choices=["cpu", "cuda"], help="Override inference device.")
    parser.add_argument("--max-length", type=int, default=192)
    return parser.parse_args()


def load_texts(args: argparse.Namespace) -> list[str]:
    if args.text and args.input_file:
        raise SystemExit("Use either --text or --input-file, not both.")
    if args.input_file:
        if not args.input_file.exists():
            raise SystemExit(f"Input file does not exist: {args.input_file}")
        content = args.input_file.read_text(encoding="utf-8")
        if args.input_format == "whole":
            return [content.strip()] if content.strip() else []
        if args.input_format == "paragraph":
            return [
                " ".join(paragraph.split())
                for paragraph in content.split("\n\n")
                if paragraph.strip()
            ]
        return [line.strip() for line in content.splitlines() if line.strip()]
    return args.text or DEFAULT_TEXTS


def main() -> None:
    args = parse_args()
    texts = load_texts(args)
    if not texts:
        raise SystemExit("No texts to classify.")

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
