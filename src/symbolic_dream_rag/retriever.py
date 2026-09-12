import json
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


class HexagramRetriever:
    def __init__(self, knowledge_base, model_name="sentence-transformers/all-MiniLM-L6-v2", device=None):
        self.knowledge_base = Path(knowledge_base)
        self.encoder = SentenceTransformer(model_name, device=device)
        self.entries = []
        self.index = None

    def build(self):
        with self.knowledge_base.open(encoding="utf-8") as stream:
            self.entries = json.load(stream)["hexagrams"]
        documents = [self._document(entry) for entry in self.entries]
        embeddings = self.encoder.encode(
            documents,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False
        ).astype("float32")
        self.index = faiss.IndexFlatIP(embeddings.shape[1])
        self.index.add(embeddings)
        return self

    def search(self, query, top_k=5):
        if self.index is None:
            raise RuntimeError("Call build() before search().")
        if not 1 <= top_k <= len(self.entries):
            raise ValueError(f"top_k must be between 1 and {len(self.entries)}")
        query_vector = self.encoder.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False
        ).astype("float32")
        scores, positions = self.index.search(query_vector, top_k)
        results = []
        for score, position in zip(scores[0], positions[0]):
            entry = dict(self.entries[int(position)])
            entry["score"] = float(score)
            entry["document"] = self._document(entry)
            results.append(entry)
        return results

    @staticmethod
    def _document(entry):
        return (
            f"Hexagram {entry['number']} {entry['chinese']} {entry['pinyin']}, "
            f"{entry['english']}. {entry['retrieval_text']}"
        )
