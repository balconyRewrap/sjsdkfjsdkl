import asyncio
import json
from pathlib import Path

import torch
from train_from_scratch import ScratchTransformerClassifier, make_collate_fn


class SafetyAnalyzer:
    def __init__(
        self,
        model_dir: Path = Path("models/scratch-transformer-safety"),
        dangerous_threshold: float = 0.5,
    ):
        self.model_dir = model_dir
        self.dangerous_threshold = dangerous_threshold
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.vocab = None
        self.config = None
        self.id2label = None
        self.label2id = None
        self.collate_fn = None

    async def load_model(self):
        config_path = self.model_dir / "config.json"
        vocab_path = self.model_dir / "vocab.json"
        weights_path = self.model_dir / "model.pt"

        if not all(path.exists() for path in [config_path, vocab_path, weights_path]):
            raise FileNotFoundError(f"Не найдены файлы safety-модели в {self.model_dir}")

        self.config = json.loads(config_path.read_text(encoding="utf-8"))
        self.vocab = json.loads(vocab_path.read_text(encoding="utf-8"))
        self.id2label = {int(key): value for key, value in self.config["id2label"].items()}
        self.label2id = {str(key): int(value) for key, value in self.config["label2id"].items()}

        self.model = ScratchTransformerClassifier(
            vocab_size=self.config["vocab_size"],
            num_labels=len(self.config["labels"]),
            max_length=self.config["max_length"],
            d_model=self.config["d_model"],
            nhead=self.config["nhead"],
            num_layers=self.config["num_layers"],
            dim_feedforward=self.config["dim_feedforward"],
            dropout=self.config["dropout"],
        ).to(self.device)

        self.model.load_state_dict(torch.load(weights_path, map_location=self.device))
        self.model.eval()
        self.collate_fn = make_collate_fn(self.vocab, self.config["max_length"])

    async def predict(self, texts: list[str]) -> list[tuple[str, float]]:
        if self.model is None:
            await self.load_model()

        def _predict_sync():
            batch = self.collate_fn([(text, 0) for text in texts])
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)

            with torch.no_grad():
                probabilities = torch.softmax(self.model(input_ids, attention_mask), dim=-1).cpu()

            dangerous_id = self.label2id["dangerous"]
            safe_id = self.label2id["safe"]
            results = []
            for probs in probabilities:
                dangerous_score = float(probs[dangerous_id])
                if dangerous_score >= self.dangerous_threshold:
                    results.append(("dangerous", dangerous_score))
                else:
                    results.append(("safe", float(probs[safe_id])))
            return results

        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, _predict_sync)
