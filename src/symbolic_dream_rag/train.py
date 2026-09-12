import argparse
import json
from pathlib import Path

import torch
import yaml
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup

from .dataset import DreamCollator, DreamDataset
from .metrics import prediction_metrics
from .model import SymbolicRAGModel, multitask_loss
from .runtime import device_from_arg, move_batch, seed_everything


@torch.no_grad()
def validate(model, loader, device):
    model.eval()
    labels, probabilities, psqi_true, psqi_pred = [], [], [], []
    for batch in loader:
        records = batch.pop("records")
        moved = move_batch(batch, device)
        outputs = model(
            dream_input_ids=moved["dream_input_ids"],
            dream_attention_mask=moved["dream_attention_mask"],
            symbol_input_ids=moved["symbol_input_ids"],
            symbol_attention_mask=moved["symbol_attention_mask"]
        )
        labels.extend(moved["labels"].cpu().tolist())
        probabilities.extend(torch.sigmoid(outputs["class_logits"]).cpu().tolist())
        psqi_true.extend(moved["psqi"].cpu().tolist())
        psqi_pred.extend(outputs["psqi_prediction"].cpu().tolist())
    return prediction_metrics(labels, probabilities, psqi_true, psqi_pred)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mode", choices=sorted(SymbolicRAGModel.MODES))
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if args.mode:
        config["mode"] = args.mode
    seed_everything(config["seed"])
    device = device_from_arg(args.device)
    tokenizer = AutoTokenizer.from_pretrained(config["backbone_name"])
    collator = DreamCollator(tokenizer, config["max_dream_length"], config["max_symbol_length"])
    train_loader = DataLoader(
        DreamDataset(args.data, "train"), batch_size=config["batch_size"], shuffle=True,
        num_workers=config["num_workers"], collate_fn=collator
    )
    dev_loader = DataLoader(
        DreamDataset(args.data, "dev"), batch_size=config["batch_size"], shuffle=False,
        num_workers=config["num_workers"], collate_fn=collator
    )

    model = SymbolicRAGModel(
        backbone_name=config["backbone_name"], mode=config["mode"],
        attention_heads=config["attention_heads"], dropout=config["dropout"]
    ).to(device)
    optimizer = AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
    total_steps = config["epochs"] * len(train_loader)
    scheduler = get_cosine_schedule_with_warmup(optimizer, 0, total_steps)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    best_f1 = -1.0
    epochs_without_improvement = 0
    history = []

    for epoch in range(1, config["epochs"] + 1):
        model.train()
        running_loss = 0.0
        progress = tqdm(train_loader, desc=f"epoch {epoch}/{config['epochs']}")
        for batch in progress:
            batch.pop("records")
            moved = move_batch(batch, device)
            optimizer.zero_grad(set_to_none=True)
            outputs = model(
                dream_input_ids=moved["dream_input_ids"],
                dream_attention_mask=moved["dream_attention_mask"],
                symbol_input_ids=moved["symbol_input_ids"],
                symbol_attention_mask=moved["symbol_attention_mask"]
            )
            loss, _ = multitask_loss(outputs, moved["labels"], moved["psqi"], config["alpha"])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config["gradient_clip"])
            optimizer.step()
            scheduler.step()
            running_loss += loss.item()
            progress.set_postfix(loss=f"{loss.item():.4f}")

        dev_metrics = validate(model, dev_loader, device)
        row = {"epoch": epoch, "train_loss": running_loss / len(train_loader), **dev_metrics}
        history.append(row)
        print(json.dumps(row, indent=2))

        if dev_metrics["macro_f1"] > best_f1:
            best_f1 = dev_metrics["macro_f1"]
            epochs_without_improvement = 0
            torch.save({"model_state": model.state_dict(), "config": config, "epoch": epoch}, args.output_dir / "best.pt")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= config["patience"]:
                break

    (args.output_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
