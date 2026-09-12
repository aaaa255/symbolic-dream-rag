import math

import numpy as np
from rouge_score import rouge_scorer
from sacrebleu.metrics import BLEU
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, mean_squared_error


def prediction_metrics(labels, probabilities, psqi_true, psqi_pred):
    labels = np.asarray(labels, dtype=int)
    predictions = (np.asarray(probabilities) >= 0.5).astype(int)
    psqi_true = np.asarray(psqi_true, dtype=float)
    psqi_pred = np.asarray(psqi_pred, dtype=float)
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "macro_f1": float(f1_score(labels, predictions, average="macro", zero_division=0)),
        "rmse": float(math.sqrt(mean_squared_error(psqi_true, psqi_pred))),
        "mae": float(mean_absolute_error(psqi_true, psqi_pred))
    }


def explanation_metrics(references, hypotheses):
    bleu = BLEU(effective_order=True).corpus_score(hypotheses, [references]).score
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    rouge = [scorer.score(reference, hypothesis)["rougeL"].fmeasure for reference, hypothesis in zip(references, hypotheses)]
    return {"bleu_4": float(bleu), "rouge_l": float(np.mean(rouge) * 100.0)}


def retrieved_explanation(record, predicted_label):
    entries = record["retrieved_hexagrams"]
    evidence = "; ".join(
        f"{entry['number']} {entry['pinyin']} ({entry['english']}): {entry['retrieval_text']}"
        for entry in entries
    )
    outcome = "poor" if predicted_label else "good"
    return f"Retrieved symbolic evidence: {evidence}. This evidence supports a {outcome} sleep-quality prediction."
