#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

from symbolic_dream_rag.retriever import HexagramRetriever


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--knowledge-base", type=Path, default=Path("data/hexagrams.json"))
    parser.add_argument("--model-name", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--query-field", choices=("dream_text", "imagery_labels"), default="imagery_labels")
    args = parser.parse_args()

    retriever = HexagramRetriever(args.knowledge_base, args.model_name)
    retriever.build()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    with args.input.open(encoding="utf-8") as source, args.output.open("w", encoding="utf-8") as target:
        for line in source:
            record = json.loads(line)
            query = record[args.query_field]
            if isinstance(query, list):
                query = " ".join(query)
            record["retrieved_hexagrams"] = retriever.search(query, args.top_k)
            target.write(json.dumps(record, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
