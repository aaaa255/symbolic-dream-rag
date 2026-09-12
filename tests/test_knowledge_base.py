import json
from pathlib import Path


ROOT = Path(__file__).parents[1]


def test_knowledge_base_has_king_wen_sequence():
    data = json.loads((ROOT / "data" / "hexagrams.json").read_text(encoding="utf-8"))
    entries = data["hexagrams"]
    assert len(entries) == 64
    assert [entry["number"] for entry in entries] == list(range(1, 65))
    assert all(entry["retrieval_text"] for entry in entries)


def test_cue_mapping_targets_valid_hexagrams():
    mapping = json.loads((ROOT / "data" / "cue_grounding.json").read_text(encoding="utf-8"))
    for cue in mapping["cues"].values():
        assert cue["aliases"]
        assert 1 <= len(cue["hexagram_ids"]) <= 3
        assert all(1 <= number <= 64 for number in cue["hexagram_ids"])
