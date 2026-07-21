import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Set

from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.io_utils import append_jsonl, ensure_parent_dir, get_id, read_jsonl
from src.xcomet_utils import (
    comet_predict,
    extract_sample_error_spans,
    extract_xcomet_features,
    load_comet_model,
    safe_get,
    scores_to_list,
)


def existing_ids(path: str) -> Set[str]:
    output_path = Path(path)
    if not output_path.exists():
        return set()
    return {get_id(row) for row in read_jsonl(path)}


def chunks(rows: List[Dict[str, Any]], batch_size: int):
    for start in range(0, len(rows), batch_size):
        yield rows[start : start + batch_size]


def save_debug_raw_output(path: str, output: Dict[str, Any]) -> None:
    raw_output = output.get("raw_output")
    metadata = safe_get(raw_output, "metadata", output.get("metadata"))
    scores = output.get("scores")
    ensure_parent_dir(path)
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"raw_output_type: {type(raw_output)!r}\n")
        f.write(f"metadata_type: {type(metadata)!r}\n")
        f.write(f"scores_type: {type(scores)!r}\n")
        f.write("\nraw_output_repr:\n")
        f.write(repr(raw_output))
        f.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--demo_pool", required=True)
    parser.add_argument("--output_file", required=True)
    parser.add_argument("--xcomet_model", required=True)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument("--max_samples", type=int, default=-1)
    parser.add_argument("--debug_raw_output", default=None)
    args = parser.parse_args()

    demo_pool = read_jsonl(args.demo_pool, max_samples=args.max_samples)
    done_ids = existing_ids(args.output_file)
    pending = [row for row in demo_pool if get_id(row) not in done_ids]

    model = load_comet_model(args.xcomet_model)
    progress = tqdm(total=len(pending), desc="precompute demo xCOMET")
    debug_saved = False
    for batch in chunks(pending, max(1, args.batch_size)):
        data = [{"src": row.get("src", ""), "mt": row.get("tgt", "")} for row in batch]
        output = comet_predict(model, data, batch_size=args.batch_size, gpus=args.gpus)
        if args.debug_raw_output and not debug_saved:
            save_debug_raw_output(args.debug_raw_output, output)
            debug_saved = True
        scores = scores_to_list(output.get("scores"))
        raw_output = output.get("raw_output")

        for idx, row in enumerate(batch):
            sample_score = scores[idx] if idx < len(scores) else 0.0
            sample_spans = extract_sample_error_spans(raw_output, idx)
            features = extract_xcomet_features(
                score=sample_score,
                error_spans_for_one_sample=sample_spans,
                mt_text=row.get("tgt", ""),
            )
            out_row = dict(row)
            out_row.update(features)
            append_jsonl(args.output_file, out_row)
            progress.update(1)
    progress.close()


if __name__ == "__main__":
    main()
