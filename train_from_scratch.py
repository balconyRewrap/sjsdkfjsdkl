from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from sentiment_data import (
    DEFAULT_SOURCES,
    ID2LABEL,
    LABEL2ID,
    LABELS,
    build_sentiment_dataframe,
    load_prepared_dataframe,
    print_dataset_report,
    save_sentiment_dataframe,
)


TOKEN_RE = re.compile(r"[\wёЁ]+|[^\w\s]", re.UNICODE)
PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"
CLS_TOKEN = "<cls>"
PAD_ID = 0
UNK_ID = 1
CLS_ID = 2


def tokenize_text(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


def build_vocab(texts: list[str], max_vocab_size: int, min_freq: int) -> dict[str, int]:
    counter: Counter[str] = Counter()
    for text in tqdm(texts, desc="Building vocab"):
        counter.update(tokenize_text(text))

    vocab = {PAD_TOKEN: PAD_ID, UNK_TOKEN: UNK_ID, CLS_TOKEN: CLS_ID}
    for token, count in counter.most_common(max_vocab_size - len(vocab)):
        if count < min_freq:
            break
        vocab[token] = len(vocab)
    return vocab


def encode_text(text: str, vocab: dict[str, int], max_length: int) -> list[int]:
    token_ids = [vocab.get(token, UNK_ID) for token in tokenize_text(text)]
    return [CLS_ID] + token_ids[: max_length - 1]


class TextDataset(Dataset):
    def __init__(self, texts: list[str], labels: list[int]) -> None:
        self.texts = texts
        self.labels = labels

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, index: int) -> tuple[str, int]:
        return self.texts[index], self.labels[index]


def make_collate_fn(vocab: dict[str, int], max_length: int):
    def collate(batch: list[tuple[str, int]]) -> dict[str, torch.Tensor]:
        encoded = [encode_text(text, vocab, max_length) for text, _ in batch]
        labels = torch.tensor([label for _, label in batch], dtype=torch.long)
        batch_max_len = max(len(item) for item in encoded)
        input_ids = torch.full((len(encoded), batch_max_len), PAD_ID, dtype=torch.long)
        attention_mask = torch.zeros((len(encoded), batch_max_len), dtype=torch.bool)
        for row, item in enumerate(encoded):
            input_ids[row, : len(item)] = torch.tensor(item, dtype=torch.long)
            attention_mask[row, : len(item)] = True
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

    return collate


class ScratchTransformerClassifier(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        num_labels: int,
        max_length: int,
        d_model: int,
        nhead: int,
        num_layers: int,
        dim_feedforward: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, d_model, padding_idx=PAD_ID)
        self.position_embedding = nn.Embedding(max_length, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            enable_nested_tensor=False,
        )
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(d_model, num_labels)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len = input_ids.shape
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, seq_len)
        hidden = self.token_embedding(input_ids) * math.sqrt(self.token_embedding.embedding_dim)
        hidden = hidden + self.position_embedding(positions)
        padding_mask = ~attention_mask
        hidden = self.encoder(hidden, src_key_padding_mask=padding_mask)
        cls_hidden = self.norm(hidden[:, 0])
        return self.classifier(self.dropout(cls_hidden))


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    losses = []
    predictions = []
    targets = []
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels)
            losses.append(float(loss.item()))
            predictions.extend(logits.argmax(dim=-1).cpu().numpy().tolist())
            targets.extend(labels.cpu().numpy().tolist())
    return {
        "eval_loss": float(np.mean(losses)),
        "eval_accuracy": accuracy_score(targets, predictions),
        "eval_macro_f1": f1_score(targets, predictions, average="macro"),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a Russian sentiment transformer from scratch.")
    parser.add_argument("--data-dir", type=Path, default=Path("."))
    parser.add_argument("--dataset-csv", type=Path, help="Use prepared CSV/CSV.GZ with text,label columns.")
    parser.add_argument("--sources", nargs="+", choices=DEFAULT_SOURCES, default=DEFAULT_SOURCES)
    parser.add_argument("--output-dir", type=Path, default=Path("models/scratch-transformer-sentiment"))
    parser.add_argument("--save-dataset", type=Path, help="Save combined text,label,source CSV before training.")
    parser.add_argument("--only-build-dataset", action="store_true")
    parser.add_argument("--max-per-label", type=int, default=None)
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
    parser.add_argument(
        "--log-every-steps",
        type=int,
        default=100,
        help="Print train loss every N optimizer steps. Use 1 to log every batch.",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

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

    train_df, valid_df = train_test_split(
        data,
        test_size=max(int(len(data) * args.validation_size), len(LABELS)),
        random_state=args.seed,
        stratify=data["label"],
    )
    train_df = train_df.copy()
    valid_df = valid_df.copy()
    train_texts = train_df["text"].tolist()
    valid_texts = valid_df["text"].tolist()
    train_labels = train_df["label"].map(LABEL2ID).astype(int).tolist()
    valid_labels = valid_df["label"].map(LABEL2ID).astype(int).tolist()

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
        num_labels=len(LABELS),
        max_length=args.max_length,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
    ).to(device)
    criterion = nn.CrossEntropyLoss()
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
        "labels": LABELS,
        "label2id": LABEL2ID,
        "id2label": ID2LABEL,
        "max_length": args.max_length,
        "vocab_size": len(vocab),
        "d_model": args.d_model,
        "nhead": args.nhead,
        "num_layers": args.num_layers,
        "dim_feedforward": args.dim_feedforward,
        "dropout": args.dropout,
        "token_pattern": TOKEN_RE.pattern,
        "pad_token": PAD_TOKEN,
        "unk_token": UNK_TOKEN,
        "cls_token": CLS_TOKEN,
    }
    (args.output_dir / "vocab.json").write_text(json.dumps(vocab, ensure_ascii=False), encoding="utf-8")
    (args.output_dir / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Best eval_macro_f1: {best_f1:.4f}")
    print(f"Saved artifacts to {args.output_dir}")


if __name__ == "__main__":
    main()
