import asyncio
import json
from pathlib import Path

import torch
from train_from_scratch import ScratchTransformerClassifier  
from test_from_scratch import load_texts 
class SentimentAnalyzer:
    def __init__(self, model_dir: Path = Path("models/scratch-transformer-sentiment")):
        self.model_dir = model_dir
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.vocab = None
        self.config = None
        self.id2label = None
        self.collate_fn = None

    async def load_model(self):
        config_path = self.model_dir / "config.json"
        vocab_path = self.model_dir / "vocab.json"
        weights_path = self.model_dir / "model.pt"

        if not all(p.exists() for p in [config_path, vocab_path, weights_path]):
            raise FileNotFoundError(f"Не найдены файлы модели в {self.model_dir}")

        self.config = json.loads(config_path.read_text(encoding="utf-8"))
        self.vocab = json.loads(vocab_path.read_text(encoding="utf-8"))
        self.id2label = {int(key): value for key, value in self.config["id2label"].items()}

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

        from train_from_scratch import make_collate_fn
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

            results = []
            for probs in probabilities:
                best_id = int(probs.argmax())
                label = self.id2label.get(best_id, str(best_id))
                confidence = float(probs[best_id])
                results.append((label, confidence))
            return results

        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, _predict_sync)