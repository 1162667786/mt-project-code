import argparse
import csv
import math
import re
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.io_utils import ensure_parent_dir, read_jsonl, write_jsonl
from src.xcomet_utils import comet_predict, scores_to_list


DEFAULT_QE_MODEL = "/path/to/model/wmt22-cometkiwi-da"


def nonempty(value: Any) -> bool:
    return bool(str(value or "").strip())


def coalesce(row: Dict[str, Any], *keys: str, default: Any = "") -> Any:
    for key in keys:
        value = row.get(key)
        if value is not None:
            return value
    return default


def parse_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def infer_method_from_path(path: str) -> str:
    stem = Path(path).stem
    if stem == "query_baseline":
        return "zero_shot"
    if stem.startswith("debug_"):
        return stem[len("debug_") :]
    return stem


def canonical_method(method: Any) -> str:
    value = str(method)
    return "retriever_topk" if value == "bm25_topk" else value


def safe_filename(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    return cleaned or "unknown"


def normalize_debug_row(path: str, line_no: int, row: Dict[str, Any]) -> Dict[str, Any]:
    idx = coalesce(row, "idx", "sample_id", "query_id", "id", default=line_no)
    method = canonical_method(coalesce(row, "selector_method", "method", default=infer_method_from_path(path)))
    k = parse_int(row.get("k"), 0 if method == "zero_shot" else -1)
    return {
        "idx": parse_int(idx, line_no),
        "src": str(coalesce(row, "src", "source", "query_src", default="")),
        "ref": str(coalesce(row, "ref", "reference", default="")),
        "prediction": str(coalesce(row, "prediction", "baseline_pred", "raw_generation", default="")),
        "method": str(method),
        "k": k,
    }


def load_debug_rows(paths: List[str]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for path in paths:
        for line_no, row in enumerate(read_jsonl(path)):
            rows.append(normalize_debug_row(path, line_no, row))
    return rows


def build_reference_samples(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    by_idx: Dict[int, Dict[str, Any]] = {}
    for row in rows:
        idx = int(row["idx"])
        sample = {"idx": idx, "src": row["src"], "ref": row["ref"]}
        existing = by_idx.get(idx)
        if existing is None:
            by_idx[idx] = sample
            continue
        if existing["src"] != sample["src"] or existing["ref"] != sample["ref"]:
            print(
                f"Warning: inconsistent src/ref for idx={idx}; keeping the first occurrence.",
                file=sys.stderr,
            )
    return [by_idx[idx] for idx in sorted(by_idx)]


def find_preferred_ckpt(directory: Path) -> Optional[Path]:
    ckpts = sorted(directory.rglob("*.ckpt"))
    if not ckpts:
        return None

    def priority(path: Path) -> Tuple[int, int, str]:
        text = str(path).lower()
        name = path.name.lower()
        if "model" in name:
            rank = 0
        elif "checkpoints" in text:
            rank = 1
        else:
            rank = 2
        return rank, len(path.parts), str(path)

    return sorted(ckpts, key=priority)[0]


def load_qe_model(path_or_name: str) -> Any:
    from comet import load_from_checkpoint

    path = Path(path_or_name).expanduser()
    load_errors: List[str] = []
    candidates: List[Path] = []

    if path.is_file() and path.suffix == ".ckpt":
        candidates.append(path)
    elif path.is_dir():
        preferred = find_preferred_ckpt(path)
        if preferred is not None:
            candidates.append(preferred)
        candidates.append(path)
    else:
        candidates.append(path)

    seen = set()
    for candidate in candidates:
        key = str(candidate)
        if key in seen:
            continue
        seen.add(key)
        try:
            model = load_from_checkpoint(str(candidate))
            print(f"Loaded COMET-QE checkpoint: {candidate}")
            return model
        except Exception as exc:  # noqa: BLE001 - keep trying fallback candidates.
            load_errors.append(f"{candidate}: {exc}")

    detail = "\n".join(load_errors)
    raise RuntimeError(f"Failed to load COMET-QE model from {path_or_name}.\n{detail}")


def cache_line_count_matches(path: Path, expected: int) -> bool:
    if not path.is_file():
        return False
    with path.open("r", encoding="utf-8") as f:
        return sum(1 for line in f if line.strip()) == expected


def score_candidates(
    model: Any,
    rows: List[Dict[str, Any]],
    mt_field: str,
    score_field: str,
    batch_size: int,
    gpus: int,
) -> List[Dict[str, Any]]:
    scored = [dict(row) for row in rows]
    pending_positions = [i for i, row in enumerate(scored) if nonempty(row.get(mt_field))]
    if not pending_positions:
        for row in scored:
            row[score_field] = math.nan
        return scored

    data = [{"src": scored[i].get("src", ""), "mt": scored[i].get(mt_field, "")} for i in pending_positions]
    output = comet_predict(model, data, batch_size=batch_size, gpus=gpus)
    scores = scores_to_list(output.get("scores"))
    if len(scores) != len(pending_positions):
        raise RuntimeError(f"COMET-QE returned {len(scores)} scores for {len(pending_positions)} inputs.")

    for row in scored:
        row[score_field] = math.nan
    for position, score in zip(pending_positions, scores):
        scored[position][score_field] = float(score)
    return scored


def load_or_score_ref_qe(
    model: Any,
    samples: List[Dict[str, Any]],
    output_dir: Path,
    batch_size: int,
    gpus: int,
    force: bool,
) -> List[Dict[str, Any]]:
    path = output_dir / "test_ref_qe.jsonl"
    if not force and cache_line_count_matches(path, len(samples)):
        print(f"Reusing cached reference QE: {path}")
        return read_jsonl(str(path))

    rows = score_candidates(model, samples, "ref", "ref_qe_score", batch_size, gpus)
    output_rows = [
        {"idx": row["idx"], "src": row["src"], "ref": row["ref"], "ref_qe_score": row["ref_qe_score"]}
        for row in rows
    ]
    write_jsonl(str(path), output_rows)
    return output_rows


def load_or_score_pred_qe(
    model: Any,
    rows: List[Dict[str, Any]],
    lang: str,
    method: str,
    k: int,
    output_dir: Path,
    batch_size: int,
    gpus: int,
    force: bool,
) -> List[Dict[str, Any]]:
    path = output_dir / f"pred_qe_{safe_filename(method)}_k{k}.jsonl"
    if not force and cache_line_count_matches(path, len(rows)):
        print(f"Reusing cached prediction QE: {path}")
        return read_jsonl(str(path))

    scored = score_candidates(model, rows, "prediction", "pred_qe_score", batch_size, gpus)
    output_rows = [
        {
            "idx": row["idx"],
            "lang": lang,
            "method": method,
            "k": k,
            "src": row["src"],
            "ref": row["ref"],
            "prediction": row["prediction"],
            "pred_qe_score": row["pred_qe_score"],
        }
        for row in scored
    ]
    write_jsonl(str(path), output_rows)
    return output_rows


def finite_values(values: Iterable[Any]) -> List[float]:
    out = []
    for value in values:
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(number):
            out.append(number)
    return out


def mean_or_nan(values: Iterable[Any]) -> float:
    nums = finite_values(values)
    return float(mean(nums)) if nums else math.nan


def median_or_nan(values: Iterable[Any]) -> float:
    nums = finite_values(values)
    return float(median(nums)) if nums else math.nan


def build_ref_score_map(ref_rows: List[Dict[str, Any]]) -> Dict[int, float]:
    return {int(row["idx"]): float(row["ref_qe_score"]) for row in ref_rows}


def write_compare_csv(
    pred_rows: List[Dict[str, Any]],
    ref_scores: Dict[int, float],
    output_dir: Path,
    method: str,
    k: int,
) -> List[Dict[str, Any]]:
    compare_rows: List[Dict[str, Any]] = []
    for row in pred_rows:
        ref_score = ref_scores[int(row["idx"])]
        pred_score = row.get("pred_qe_score", math.nan)
        pred_is_finite = math.isfinite(float(pred_score)) if pred_score is not None else False
        delta = float(pred_score) - ref_score if pred_is_finite else math.nan
        compare_rows.append(
            {
                "idx": row["idx"],
                "lang": row["lang"],
                "method": row["method"],
                "k": row["k"],
                "src": row["src"],
                "ref": row["ref"],
                "prediction": row["prediction"],
                "ref_qe_score": ref_score,
                "pred_qe_score": pred_score,
                "delta_qe": delta,
                "pred_better_than_ref": bool(pred_is_finite and float(pred_score) > ref_score),
            }
        )

    path = output_dir / f"ref_pred_qe_compare_{safe_filename(method)}_k{k}.csv"
    ensure_parent_dir(str(path))
    pd.DataFrame(compare_rows).to_csv(path, index=False, encoding="utf-8", quoting=csv.QUOTE_MINIMAL)
    return compare_rows


def summarize_compare_rows(compare_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "lang": compare_rows[0]["lang"] if compare_rows else "",
        "method": compare_rows[0]["method"] if compare_rows else "",
        "k": compare_rows[0]["k"] if compare_rows else -1,
        "n": len(compare_rows),
        "ref_qe_mean": mean_or_nan(row.get("ref_qe_score") for row in compare_rows),
        "ref_qe_median": median_or_nan(row.get("ref_qe_score") for row in compare_rows),
        "pred_qe_mean": mean_or_nan(row.get("pred_qe_score") for row in compare_rows),
        "pred_qe_median": median_or_nan(row.get("pred_qe_score") for row in compare_rows),
        "delta_qe_mean": mean_or_nan(row.get("delta_qe") for row in compare_rows),
        "delta_qe_median": median_or_nan(row.get("delta_qe") for row in compare_rows),
        "pred_better_rate": (
            mean(1.0 if row["pred_better_than_ref"] else 0.0 for row in compare_rows)
            if compare_rows
            else math.nan
        ),
    }


def assign_ref_quality_buckets(ref_rows: List[Dict[str, Any]]) -> Dict[int, str]:
    ordered = sorted(ref_rows, key=lambda row: (float(row["ref_qe_score"]), int(row["idx"])))
    n = len(ordered)
    low_count = int(n * 0.30)
    high_count = int(n * 0.30)
    buckets: Dict[int, str] = {}
    for rank, row in enumerate(ordered):
        if rank < low_count:
            bucket = "low_ref_quality"
        elif rank >= n - high_count:
            bucket = "high_ref_quality"
        else:
            bucket = "mid_ref_quality"
        buckets[int(row["idx"])] = bucket
    return buckets


def summarize_buckets(compare_rows: List[Dict[str, Any]], buckets: Dict[int, str]) -> List[Dict[str, Any]]:
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in compare_rows:
        grouped[buckets[int(row["idx"])]].append(row)

    summaries = []
    for bucket in ("low_ref_quality", "mid_ref_quality", "high_ref_quality"):
        rows = grouped.get(bucket, [])
        if not rows:
            continue
        summaries.append(
            {
                "lang": rows[0]["lang"],
                "method": rows[0]["method"],
                "k": rows[0]["k"],
                "bucket": bucket,
                "n": len(rows),
                "ref_qe_mean": mean_or_nan(row.get("ref_qe_score") for row in rows),
                "pred_qe_mean": mean_or_nan(row.get("pred_qe_score") for row in rows),
                "delta_qe_mean": mean_or_nan(row.get("delta_qe") for row in rows),
                "pred_better_rate": (
                    mean(1.0 if row["pred_better_than_ref"] else 0.0 for row in rows)
                    if rows
                    else math.nan
                ),
            }
        )
    return summaries


def print_summary(rows: List[Dict[str, Any]]) -> None:
    print("method,k,n,ref_qe_mean,pred_qe_mean,delta_qe_mean,pred_better_rate")
    for row in rows:
        print(
            f"{row['method']},{row['k']},{row['n']},"
            f"{row['ref_qe_mean']:.6f},{row['pred_qe_mean']:.6f},"
            f"{row['delta_qe_mean']:.6f},{row['pred_better_rate']:.6f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug_files", nargs="+", required=True)
    parser.add_argument("--qe_model", default=DEFAULT_QE_MODEL)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument("--lang", default="")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    debug_rows = load_debug_rows(args.debug_files)
    if not debug_rows:
        raise ValueError("No rows found in --debug_files.")

    grouped: Dict[Tuple[str, int], List[Dict[str, Any]]] = defaultdict(list)
    for row in debug_rows:
        grouped[(row["method"], int(row["k"]))].append(row)

    ref_samples = build_reference_samples(debug_rows)
    model = load_qe_model(args.qe_model)

    ref_rows = load_or_score_ref_qe(model, ref_samples, output_dir, args.batch_size, args.gpus, args.force)
    ref_scores = build_ref_score_map(ref_rows)
    buckets = assign_ref_quality_buckets(ref_rows)

    summary_rows: List[Dict[str, Any]] = []
    bucket_summary_rows: List[Dict[str, Any]] = []

    for (method, k), rows in sorted(grouped.items(), key=lambda item: (item[0][0], item[0][1])):
        pred_rows = load_or_score_pred_qe(
            model,
            rows,
            args.lang,
            method,
            k,
            output_dir,
            args.batch_size,
            args.gpus,
            args.force,
        )
        compare_rows = write_compare_csv(pred_rows, ref_scores, output_dir, method, k)
        summary_rows.append(summarize_compare_rows(compare_rows))
        bucket_summary_rows.extend(summarize_buckets(compare_rows, buckets))

    summary_path = output_dir / "ref_pred_qe_summary.csv"
    bucket_path = output_dir / "ref_quality_bucket_summary.csv"
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False, encoding="utf-8")
    pd.DataFrame(bucket_summary_rows).to_csv(bucket_path, index=False, encoding="utf-8")

    print_summary(summary_rows)


if __name__ == "__main__":
    main()
