import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.io_utils import ensure_parent_dir, read_jsonl
from src.metrics_utils import compute_bleu
from src.xcomet_utils import comet_predict, load_comet_model, scores_to_list


def nonempty(value: Any) -> bool:
    return bool(str(value or "").strip())


def group_key(row: Dict[str, Any], fields: List[str]) -> Tuple[Any, ...]:
    return tuple(row.get(field) for field in fields)


def compute_comet(model: Any, rows: List[Dict[str, Any]], batch_size: int, gpus: int) -> float:
    if not rows:
        return 0.0
    data = [{"src": row.get("query_src", ""), "mt": row.get("prediction", ""), "ref": row.get("ref", "")} for row in rows]
    output = comet_predict(model, data, batch_size=batch_size, gpus=gpus)
    scores = scores_to_list(output.get("scores"))
    if len(scores) == 0:
        return 0.0
    return float(sum(float(score) for score in scores) / len(scores))


def summarize_group(
    name_values: Dict[str, Any],
    rows: List[Dict[str, Any]],
    comet20_model: Any,
    comet22_model: Any,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    empty_count = sum(1 for row in rows if not nonempty(row.get("prediction")))
    valid_rows = [row for row in rows if nonempty(row.get("prediction")) and nonempty(row.get("ref"))]
    predictions = [str(row.get("prediction", "")) for row in valid_rows]
    references = [str(row.get("ref", "")) for row in valid_rows]

    result = dict(name_values)
    result.update(
        {
            "num_samples": len(rows),
            "empty_prediction_count": empty_count,
            "bleu": compute_bleu(predictions, references, tokenize=args.bleu_tokenize),
            "comet20": compute_comet(comet20_model, valid_rows, args.batch_size, args.gpus),
            "comet22": compute_comet(comet22_model, valid_rows, args.batch_size, args.gpus),
        }
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", required=True)
    parser.add_argument("--output_file", required=True)
    parser.add_argument("--comet20_model", required=True)
    parser.add_argument("--comet22_model", required=True)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument("--bleu_tokenize", default="zh")
    parser.add_argument("--group_by", default="selector_method,k")
    args = parser.parse_args()

    rows = read_jsonl(args.input_file)
    group_fields = [field.strip() for field in args.group_by.split(",") if field.strip()]

    comet20_model = load_comet_model(args.comet20_model)
    comet22_model = load_comet_model(args.comet22_model)

    summaries: List[Dict[str, Any]] = []
    summaries.append(
        summarize_group(
            {"selector_method": "ALL", "k": "ALL"},
            rows,
            comet20_model,
            comet22_model,
            args,
        )
    )

    grouped: Dict[Tuple[Any, ...], List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[group_key(row, group_fields)].append(row)

    for key, group_rows in sorted(grouped.items(), key=lambda item: str(item[0])):
        name_values = {field: value for field, value in zip(group_fields, key)}
        summaries.append(summarize_group(name_values, group_rows, comet20_model, comet22_model, args))

    csv_path = f"{args.output_file}.csv"
    json_path = f"{args.output_file}.json"
    ensure_parent_dir(csv_path)
    pd.DataFrame(summaries).to_csv(csv_path, index=False, encoding="utf-8")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summaries, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
