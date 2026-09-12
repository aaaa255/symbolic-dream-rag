import argparse
import json
from pathlib import Path

import torch
from transformers import AutoTokenizer

from .dataset import DreamCollator
from .metrics import retrieved_explanation
from .model import SymbolicRAGModel
from .retriever import HexagramRetriever
from .runtime import device_from_arg, move_batch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--text", required=True)
    parser.add_argument("--knowledge-base", type=Path, default=Path("data/hexagrams.json"))
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    device = device_from_arg(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    config = checkpoint["config"]
    retriever = HexagramRetriever(args.knowledge_base, config["backbone_name"], str(device)).build()
    retrieved = retriever.search(args.text, args.top_k)
    record = {
        "dream_text": args.text,
        "retrieved_hexagrams": retrieved,
        "sleep_label": 0,
        "psqi_score": 0
    }
    tokenizer = AutoTokenizer.from_pretrained(config["backbone_name"])
    batch = DreamCollator(tokenizer, config["max_dream_length"], config["max_symbol_length"])([record])
    batch.pop("records")
    moved = move_batch(batch, device)
    model = SymbolicRAGModel(
        config["backbone_name"], config["mode"], config["attention_heads"], config["dropout"]
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    with torch.no_grad():
        outputs = model(
            moved["dream_input_ids"], moved["dream_attention_mask"],
            moved["symbol_input_ids"], moved["symbol_attention_mask"]
        )
    probability = torch.sigmoid(outputs["class_logits"])[0].item()
    result = {
        "poor_sleep_probability": probability,
        "predicted_label": "poor" if probability >= 0.5 else "good",
        "predicted_psqi": outputs["psqi_prediction"][0].item(),
        "retrieved_hexagrams": retrieved,
        "explanation": retrieved_explanation(record, probability >= 0.5),
        "warning": "Research demonstration only; not a clinical assessment."
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
