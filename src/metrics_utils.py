from typing import Any, Dict, List

import sacrebleu

from .xcomet_utils import comet_predict, load_comet_model, scores_to_list


def compute_bleu(predictions: List[str], references: List[str], tokenize: str = "zh") -> float:
    if not predictions:
        return 0.0
    return float(sacrebleu.corpus_bleu(predictions, [references], tokenize=tokenize).score)


def compute_comet_average(
    rows: List[Dict[str, Any]],
    model_path_or_name: str,
    batch_size: int = 8,
    gpus: int = 1,
) -> float:
    if not rows:
        return 0.0
    model = load_comet_model(model_path_or_name)
    data = [{"src": row["query_src"], "mt": row["prediction"], "ref": row["ref"]} for row in rows]
    output = comet_predict(model, data, batch_size=batch_size, gpus=gpus)
    scores = scores_to_list(output.get("scores"))
    if len(scores) == 0:
        return 0.0
    return float(sum(float(score) for score in scores) / len(scores))
