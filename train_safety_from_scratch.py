from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from safety_data import (
    DEFAULT_SAFETY_SOURCES,
    SAFETY_ID2LABEL,
    SAFETY_LABEL2ID,
    SAFETY_LABELS,
    SAFETY_LOADERS,
    build_safety_dataframe,
    load_prepared_safety_dataframe,
    print_safety_dataset_report,
    save_safety_dataframe,
)
from train_from_scratch import (
    ScratchTransformerClassifier,
    TextDataset,
    TOKEN_RE,
    build_vocab,
    evaluate,
    make_collate_fn,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a Russian text safety classifier from scratch.")
    parser.add_argument("--dataset-csv", type=Path, help="Use prepared CSV/CSV.GZ with text,label columns.")
    parser.add_argument("--sources", nargs="+", choices=sorted(SAFETY_LOADERS), default=DEFAULT_SAFETY_SOURCES)
    parser.add_argument("--hf-cache-dir", type=Path, default=Path(".hf-cache"))
    parser.add_argument("--output-dir", type=Path, default=Path("models/scratch-transformer-safety"))
    parser.add_argument("--save-dataset", type=Path, help="Save combined text,label,risk_category,source CSV.")
    parser.add_argument("--only-build-dataset", action="store_true")
    parser.add_argument("--max-per-label", type=int, default=None)
    parser.add_argument(
        "--balance-ratio",
        type=float,
        default=None,
        help="Limit safe examples to dangerous_count * ratio. Use 1.0 for roughly 1:1 safe/dangerous.",
    )
    parser.add_argument(
        "--max-toxic-share",
        type=float,
        default=0.2,
        help="Maximum share of examples from broad toxic-comment datasets after deduplication. Use 1 to disable.",
    )
    parser.add_argument("--validation-size", type=float, default=0.1)
    parser.add_argument("--max-length", type=int, default=192)
    parser.add_argument("--max-vocab-size", type=int, default=80_000)
    parser.add_argument("--min-freq", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--nhead", type=int, default=8)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--dim-feedforward", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--no-class-weights", action="store_true", help="Disable inverse-frequency class weights.")
    parser.add_argument(
        "--log-every-steps",
        type=int,
        default=100,
        help="Print train loss every N optimizer steps. Use 1 to log every batch.",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def make_class_weighted_loss(labels: list[int], device: torch.device, enabled: bool) -> nn.Module:
    if not enabled:
        return nn.CrossEntropyLoss()
    counts = np.bincount(labels, minlength=len(SAFETY_LABELS)).astype(np.float32)
    weights = counts.sum() / np.maximum(counts, 1.0)
    weights = weights / weights.mean()
    print("class_weights:", {SAFETY_ID2LABEL[idx]: float(weight) for idx, weight in enumerate(weights)})
    return nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32, device=device))


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.dataset_csv:
        data = load_prepared_safety_dataframe(args.dataset_csv, args.max_per_label, args.balance_ratio, args.seed)
    else:
        data = build_safety_dataframe(
            sources=args.sources,
            max_per_label=args.max_per_label,
            max_toxic_share=args.max_toxic_share,
            balance_ratio=args.balance_ratio,
            seed=args.seed,
            cache_dir=args.hf_cache_dir,
        )
    print_safety_dataset_report(data)

    if args.save_dataset:
        save_safety_dataframe(data, args.save_dataset)
        print(f"\nSaved combined safety dataset to {args.save_dataset}")

    if args.only_build_dataset:
        return

    train_df, valid_df = train_test_split(
        data,
        test_size=max(int(len(data) * args.validation_size), len(SAFETY_LABELS)),
        random_state=args.seed,
        stratify=data["label"],
    )
    train_df = train_df.copy()
    valid_df = valid_df.copy()
    train_texts = train_df["text"].tolist()
    valid_texts = valid_df["text"].tolist()
    train_labels = train_df["label"].map(SAFETY_LABEL2ID).astype(int).tolist()
    valid_labels = valid_df["label"].map(SAFETY_LABEL2ID).astype(int).tolist()

    vocab = build_vocab(train_texts, args.max_vocab_size, args.min_freq)
    print(f"Vocab size: {len(vocab)}")

    collate_fn = make_collate_fn(vocab, args.max_length)
    train_loader = DataLoader(
        TextDataset(train_texts, train_labels),
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    valid_loader = DataLoader(
        TextDataset(valid_texts, valid_labels),
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")
    if device.type == "cuda":
        print(f"gpu: {torch.cuda.get_device_name(0)}")

    model = ScratchTransformerClassifier(
        vocab_size=len(vocab),
        num_labels=len(SAFETY_LABELS),
        max_length=args.max_length,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
    ).to(device)
    criterion = make_class_weighted_loss(train_labels, device, enabled=not args.no_class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    best_f1 = -1.0
    best_path = args.output_dir / "model.pt"

    global_step = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_losses = []
        progress = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")
        for batch_index, batch in enumerate(progress, start=1):
            global_step += 1
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_losses.append(float(loss.item()))
            running_loss = float(np.mean(train_losses))
            progress.set_postfix(train_loss=f"{running_loss:.4f}")
            if args.log_every_steps and global_step % args.log_every_steps == 0:
                print(
                    {
                        "epoch": epoch,
                        "batch": batch_index,
                        "global_step": global_step,
                        "train_loss": running_loss,
                        "last_batch_loss": float(loss.item()),
                    }
                )

        metrics = evaluate(model, valid_loader, criterion, device)
        metrics["train_loss"] = float(np.mean(train_losses))
        metrics["epoch"] = epoch
        print(metrics)
        if metrics["eval_macro_f1"] > best_f1:
            best_f1 = metrics["eval_macro_f1"]
            torch.save(model.state_dict(), best_path)
            print(f"Saved best model to {best_path}")

    config = {
        "labels": SAFETY_LABELS,
        "label2id": SAFETY_LABEL2ID,
        "id2label": SAFETY_ID2LABEL,
        "max_length": args.max_length,
        "vocab_size": len(vocab),
        "d_model": args.d_model,
        "nhead": args.nhead,
        "num_layers": args.num_layers,
        "dim_feedforward": args.dim_feedforward,
        "dropout": args.dropout,
        "token_pattern": TOKEN_RE.pattern,
        "task": "text_safety",
        "sources": args.sources,
    }
    (args.output_dir / "vocab.json").write_text(json.dumps(vocab, ensure_ascii=False), encoding="utf-8")
    (args.output_dir / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Best eval_macro_f1: {best_f1:.4f}")
    print(f"Saved artifacts to {args.output_dir}")


if __name__ == "__main__":
    main()
