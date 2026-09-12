#!/usr/bin/env python3
import argparse
import json
from collections import Counter
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=Path)
    args = parser.parse_args()

    with args.path.open(encoding="utf-8") as stream:
        records = [json.loads(line) for line in stream if line.strip()]
    required = {
        "dream_id", "split", "dream_text", "imagery_spans", "imagery_labels",
        "hexagram_ids", "psqi_score", "sleep_label", "reference_explanation", "synthetic"
    }
    ids = set()
    counts = Counter()
    labels = Counter()
    word_counts = []
    symbol_counts = []
    for row in records:
        missing = required - row.keys()
        if missing:
            raise ValueError(f"{row.get('dream_id', '<unknown>')} missing {sorted(missing)}")
        if row["dream_id"] in ids:
            raise ValueError(f"Duplicate dream_id: {row['dream_id']}")
        ids.add(row["dream_id"])
        if row["split"] not in {"train", "dev", "test"}:
            raise ValueError(f"Invalid split: {row['split']}")
        if row["sleep_label"] != int(row["psqi_score"] > 5):
            raise ValueError(f"PSQI/label mismatch: {row['dream_id']}")
        if not 0 <= row["psqi_score"] <= 21:
            raise ValueError(f"PSQI outside [0, 21]: {row['dream_id']}")
        counts[row["split"]] += 1
        labels[(row["split"], row["sleep_label"])] += 1
        word_counts.append(len(row["dream_text"].split()))
        symbol_counts.append(len(row["hexagram_ids"]))

    summary = {
        "records": len(records),
        "split_counts": dict(counts),
        "label_counts": {f"{split}:{label}": count for (split, label), count in labels.items()},
        "average_words": sum(word_counts) / len(word_counts),
        "average_annotated_hexagrams": sum(symbol_counts) / len(symbol_counts)
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
