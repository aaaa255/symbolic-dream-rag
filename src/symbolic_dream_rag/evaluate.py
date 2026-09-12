import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from .dataset import DreamCollator, DreamDataset
from .metrics import explanation_metrics, prediction_metrics, retrieved_explanation
from .model import SymbolicRAGModel
from .runtime import device_from_arg, move_batch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "dev", "test"), default="test")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    device = device_from_arg(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    config = checkpoint["config"]
    tokenizer = AutoTokenizer.from_pretrained(config["backbone_name"])
    collator = DreamCollator(tokenizer, config["max_dream_length"], config["max_symbol_length"])
    loader = DataLoader(DreamDataset(args.data, args.split), batch_size=config["batch_size"], collate_fn=collator)
    model = SymbolicRAGModel(
        config["backbone_name"], config["mode"], config["attention_heads"], config["dropout"]
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    labels, probabilities, psqi_true, psqi_pred = [], [], [], []
    references, hypotheses = [], []
    with torch.no_grad():
        for batch in loader:
            records = batch.pop("records")
            moved = move_batch(batch, device)
            outputs = model(
                moved["dream_input_ids"], moved["dream_attention_mask"],
                moved["symbol_input_ids"], moved["symbol_attention_mask"]
            )
            batch_probabilities = torch.sigmoid(outputs["class_logits"]).cpu().tolist()
            labels.extend(moved["labels"].cpu().tolist())
            probabilities.extend(batch_probabilities)
            psqi_true.extend(moved["psqi"].cpu().tolist())
            psqi_pred.extend(outputs["psqi_prediction"].cpu().tolist())
            for record, probability in zip(records, batch_probabilities):
                references.append(record["reference_explanation"])
                hypotheses.append(retrieved_explanation(record, probability >= 0.5))

    metrics = prediction_metrics(labels, probabilities, psqi_true, psqi_pred)
    metrics.update(explanation_metrics(references, hypotheses))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
