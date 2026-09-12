import json
from pathlib import Path

import torch
from torch.utils.data import Dataset


class DreamDataset(Dataset):
    def __init__(self, path, split=None):
        with Path(path).open(encoding="utf-8") as stream:
            records = [json.loads(line) for line in stream if line.strip()]
        self.records = [record for record in records if split is None or record["split"] == split]
        if not self.records:
            raise ValueError(f"No records found for split={split!r}")
        required = {"dream_text", "psqi_score", "sleep_label", "retrieved_hexagrams"}
        missing = required - self.records[0].keys()
        if missing:
            raise ValueError(f"Missing required fields: {sorted(missing)}")

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        return self.records[index]


class DreamCollator:
    def __init__(self, tokenizer, max_dream_length=128, max_symbol_length=192):
        self.tokenizer = tokenizer
        self.max_dream_length = max_dream_length
        self.max_symbol_length = max_symbol_length

    def __call__(self, records):
        dream_texts = [record["dream_text"] for record in records]
        symbol_texts = [self._symbol_text(record["retrieved_hexagrams"]) for record in records]
        dream_tokens = self.tokenizer(
            dream_texts,
            padding=True,
            truncation=True,
            max_length=self.max_dream_length,
            return_tensors="pt"
        )
        symbol_tokens = self.tokenizer(
            symbol_texts,
            padding=True,
            truncation=True,
            max_length=self.max_symbol_length,
            return_tensors="pt"
        )
        return {
            "dream_input_ids": dream_tokens["input_ids"],
            "dream_attention_mask": dream_tokens["attention_mask"],
            "symbol_input_ids": symbol_tokens["input_ids"],
            "symbol_attention_mask": symbol_tokens["attention_mask"],
            "labels": torch.tensor([record["sleep_label"] for record in records], dtype=torch.float32),
            "psqi": torch.tensor([record["psqi_score"] for record in records], dtype=torch.float32),
            "records": records
        }

    @staticmethod
    def _symbol_text(entries):
        return " [SYMBOL] ".join(entry["document"] for entry in entries)
