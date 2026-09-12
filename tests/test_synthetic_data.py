import importlib.util
from pathlib import Path


MODULE_PATH = Path(__file__).parents[1] / "scripts" / "generate_synthetic_dataset.py"
SPEC = importlib.util.spec_from_file_location("synthetic_generator", MODULE_PATH)
GENERATOR = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(GENERATOR)


def test_paper_split_and_balance():
    records = GENERATOR.generate(2026)
    GENERATOR.validate(records)
    assert len([row for row in records if row["split"] == "train"]) == 280
    assert len([row for row in records if row["split"] == "dev"]) == 40
    assert len([row for row in records if row["split"] == "test"]) == 80


def test_generation_is_deterministic():
    assert GENERATOR.generate(2026) == GENERATOR.generate(2026)


def test_psqi_threshold_matches_label():
    for row in GENERATOR.generate(2026):
        assert row["sleep_label"] == int(row["psqi_score"] > 5)
        assert row["synthetic"] is True
