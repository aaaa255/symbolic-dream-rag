#!/usr/bin/env python3
import argparse
import json
import random
from pathlib import Path


SPLITS = (("train", 280), ("dev", 40), ("test", 80))

CUE_PHRASES = {
    "WATER": ["dark water rose around me", "a flooded road blocked my way", "waves covered the street"],
    "FIRE": ["the house was burning", "bright flames filled the room", "a distant fire grew larger"],
    "ASCENT": ["I climbed a steep mountain", "I kept walking up endless stairs", "I moved upward toward a ridge"],
    "DARKNESS": ["the corridor became completely dark", "night covered every familiar place", "I could not see beyond the shadows"],
    "LIGHT": ["a bright light appeared ahead", "sunlight entered through an open door", "the sky suddenly began to glow"],
    "OBSTACLE": ["a high wall stopped me", "every door was locked", "I was stuck behind a heavy barrier"],
    "LOST": ["I was lost in endless corridors", "I could not find the road home", "the streets became an unfamiliar maze"],
    "MIRROR": ["a broken mirror showed several versions of me", "my reflection moved on its own", "many mirrors surrounded me"],
    "CHASE": ["someone chased me through narrow streets", "I ran from an unseen pursuer", "I tried to escape a fast approaching figure"],
    "HOME": ["I returned to my childhood home", "my family waited in a quiet room", "I found a warm and familiar house"],
    "FALLING": ["I fell from a tall building", "the ground disappeared beneath me", "I plunged downward without control"],
    "FLIGHT": ["I flew calmly above the city", "wings carried me over the hills", "I floated freely across the sky"]
}

HEXAGRAMS = {
    "WATER": [29, 48, 59], "FIRE": [30, 49, 55], "ASCENT": [1, 35, 46],
    "DARKNESS": [29, 36, 47], "LIGHT": [30, 35, 55], "OBSTACLE": [39, 47, 52],
    "LOST": [4, 29, 56], "MIRROR": [2, 20, 38], "CHASE": [33, 40, 51],
    "HOME": [37, 48, 50], "FALLING": [23, 28, 29], "FLIGHT": [1, 14, 56]
}

POOR_CUES = ("WATER", "FIRE", "DARKNESS", "OBSTACLE", "LOST", "MIRROR", "CHASE", "FALLING")
GOOD_CUES = ("ASCENT", "LIGHT", "HOME", "FLIGHT")

OPENERS = [
    "I dreamed that", "In the dream", "I remember that", "During my dream",
    "Just before waking", "In one dream", "I found myself and"
]

CLOSERS_GOOD = [
    "I woke feeling calm.", "The scene ended peacefully.",
    "I woke rested and hopeful.", "Nothing felt threatening."
]

CLOSERS_POOR = [
    "I woke with my heart racing.", "The tension remained after waking.",
    "I woke exhausted and uneasy.", "The scene repeated until I woke."
]


def make_record(index, split, label, rng):
    primary_pool = POOR_CUES if label else GOOD_CUES
    primary = rng.choice(primary_pool)
    secondary_pool = [cue for cue in CUE_PHRASES if cue != primary]
    secondary = rng.choice(secondary_pool)
    cues = [primary, secondary]
    if index % 10 == 0:
        third_pool = [cue for cue in secondary_pool if cue != secondary]
        cues.append(rng.choice(third_pool))

    spans = [rng.choice(CUE_PHRASES[cue]) for cue in cues]
    connector = rng.choice(["and then", "while", "before", "as"])
    text = f"{rng.choice(OPENERS)} {spans[0]} {connector} {spans[1]}."
    if len(spans) == 3:
        text += f" Later, {spans[2]}."
    text += " " + rng.choice(CLOSERS_POOR if label else CLOSERS_GOOD)

    if label:
        psqi = rng.randint(6, 17)
    else:
        psqi = rng.randint(1, 5)

    hexagram_ids = []
    for cue in cues:
        candidates = HEXAGRAMS[cue]
        rotated = candidates[index % len(candidates):] + candidates[:index % len(candidates)]
        candidate = next((number for number in rotated if number not in hexagram_ids), rotated[0])
        hexagram_ids.append(candidate)
    hexagram_ids = hexagram_ids[:3]

    explanation = (
        f"The explicit cues {', '.join(cues)} are grounded in hexagrams "
        f"{', '.join(map(str, hexagram_ids))}; together they support the model's "
        f"{'poor' if label else 'good'} sleep-quality prediction."
    )
    return {
        "dream_id": f"SYN-{index:04d}",
        "split": split,
        "dream_text": text,
        "imagery_spans": spans,
        "imagery_labels": cues,
        "hexagram_ids": hexagram_ids,
        "psqi_score": psqi,
        "sleep_label": label,
        "reference_explanation": explanation,
        "synthetic": True
    }


def generate(seed=2026):
    rng = random.Random(seed)
    records = []
    index = 1
    for split, count in SPLITS:
        labels = [0] * (count // 2) + [1] * (count // 2)
        rng.shuffle(labels)
        for label in labels:
            records.append(make_record(index, split, label, rng))
            index += 1
    return records


def validate(records):
    assert len(records) == 400
    for split, expected in SPLITS:
        subset = [row for row in records if row["split"] == split]
        assert len(subset) == expected
        assert sum(row["sleep_label"] == 0 for row in subset) == expected // 2
        assert sum(row["sleep_label"] == 1 for row in subset) == expected // 2
    for row in records:
        assert row["sleep_label"] == int(row["psqi_score"] > 5)
        assert 1 <= len(row["hexagram_ids"]) <= 3
        assert row["synthetic"] is True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("data/synthetic_dreams.jsonl"))
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    records = generate(args.seed)
    validate(records)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"Wrote {len(records)} synthetic records to {args.output}")


if __name__ == "__main__":
    main()
