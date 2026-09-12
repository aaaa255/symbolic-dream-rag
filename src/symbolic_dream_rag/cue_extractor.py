import json
import re
from pathlib import Path


class ControlledCueExtractor:
    """Auditable fallback when a fine-tuned span annotator is unavailable."""

    def __init__(self, mapping_path="data/cue_grounding.json"):
        with Path(mapping_path).open(encoding="utf-8") as stream:
            self.mapping = json.load(stream)["cues"]

    def extract(self, text):
        matches = []
        lowered = text.lower()
        for label, entry in self.mapping.items():
            for alias in entry["aliases"]:
                for match in re.finditer(rf"\b{re.escape(alias.lower())}\b", lowered):
                    matches.append({
                        "label": label,
                        "span": text[match.start():match.end()],
                        "start": match.start(),
                        "end": match.end(),
                        "candidate_hexagram_ids": entry["hexagram_ids"]
                    })
        return sorted(matches, key=lambda item: (item["start"], item["end"]))
