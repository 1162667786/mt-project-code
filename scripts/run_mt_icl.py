import argparse
import hashlib
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.io_utils import append_jsonl, get_id, read_jsonl
from src.llm_utils import (
    PROMPT_VERSION,
    build_translation_prompt,
    clean_translation_output,
    generate_text,
    load_model_and_tokenizer,
)
from src.retriever_backends import build_retriever
from src.selection_methods import select_demos


SELECTOR_METHODS = [
    "retriever_topk",
    "bm25_topk",
    "random_in_topn",
    "quality_only",
    "low_quality",
    "error_filter",
    "quality_rerank",
    "error_profile_rerank",
    "xcomet_span_coverage",
]

DEFAULT_GTE_MODEL = "/path/to/model/gte-multilingual-base"
DEFAULT_SONAR_MODEL = "/path/to/model/SONAR_200_text_encoder"


def parse_shots(value: str) -> List[int]:
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def parse_src_code(lang_pair: str) -> str:
    return (lang_pair or "").split("-", 1)[0].strip()


def stable_int_hash(text: str) -> int:
    digest = hashlib.md5(str(text).encode("utf-8")).hexdigest()
    return int(digest, 16) % (2**31)


def load_query_profiles(path: Optional[str]) -> Dict[str, Dict[str, Any]]:
    if not path:
        return {}
    profile_path = Path(path)
    if not profile_path.exists():
        return {}
    return {get_id(row): row for row in read_jsonl(path)}


def is_bad_prediction(prediction: str) -> bool:
    return prediction.strip() in ("", "<think>", "</think>")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--demo_pool", required=True)
    parser.add_argument("--test_file", required=True)
    parser.add_argument("--query_profile_file", default=None)
    parser.add_argument("--output_file", required=True)
    parser.add_argument("--src_lang", default="English")
    parser.add_argument("--tgt_lang", default="Chinese")
    parser.add_argument("--retriever", default="bm25", choices=["bm25", "gte", "sonar"])
    parser.add_argument("--retriever_embedding_model", default=DEFAULT_GTE_MODEL)
    parser.add_argument("--sonar_model", default=DEFAULT_SONAR_MODEL)
    parser.add_argument("--sonar_backend", default="auto", choices=["auto", "official", "transformers"])
    parser.add_argument("--retriever_cache_dir", default=None)
    parser.add_argument("--retriever_batch_size", type=int, default=64)
    parser.add_argument("--force_recompute_retriever_cache", action="store_true")
    parser.add_argument("--selector_method", default="retriever_topk", choices=SELECTOR_METHODS)
    parser.add_argument("--candidate_size", type=int, default=50)
    parser.add_argument("--shots", default="3,5,8")
    parser.add_argument("--translator_model", default="Qwen/Qwen3-8B")
    parser.add_argument("--model_family", default="qwen3", choices=["qwen3", "llama3", "llama31", "bloomz"])
    parser.add_argument("--model_tag", default="")
    parser.add_argument("--lang_pair", default="")
    parser.add_argument("--src_code", default=None)
    parser.add_argument("--prompt_version", default=PROMPT_VERSION)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--max_test_samples", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_prompts", action="store_true")
    parser.add_argument("--save_raw_outputs", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--alpha", type=float, default=None)
    parser.add_argument("--beta", type=float, default=None)
    parser.add_argument("--gamma", type=float, default=None)
    parser.add_argument("--eta", type=float, default=0.15)
    parser.add_argument("--lambda_quality", type=float, default=None)
    parser.add_argument("--keep_top_r", type=int, default=None)
    parser.add_argument("--coverage_lambda", type=float, default=0.30)
    parser.add_argument("--span_embedding_model", default="Alibaba-NLP/gte-multilingual-base")
    parser.add_argument("--span_demo_embedding_file", default=None)
    parser.add_argument("--span_query_embedding_file", default=None)
    parser.add_argument("--critical_threshold", type=int, default=0)
    parser.add_argument("--major_threshold", type=int, default=2)
    args = parser.parse_args()

    output_path = Path(args.output_file)
    if args.overwrite and output_path.exists():
        output_path.unlink()

    demo_pool = read_jsonl(args.demo_pool)
    test_rows = read_jsonl(args.test_file, max_samples=args.max_test_samples)
    query_profiles = load_query_profiles(args.query_profile_file)
    source_lang_code = args.src_code or parse_src_code(args.lang_pair)
    retriever = build_retriever(
        name=args.retriever,
        demo_pool=demo_pool,
        cache_dir=args.retriever_cache_dir or str(output_path.parent / "retriever_cache"),
        embedding_model=args.retriever_embedding_model,
        source_lang_code=source_lang_code,
        device=args.device,
        batch_size=args.retriever_batch_size,
        sonar_model=args.sonar_model,
        sonar_backend=args.sonar_backend,
        force_recompute=args.force_recompute_retriever_cache,
    )
    model, tokenizer = load_model_and_tokenizer(args.translator_model, args.device, args.model_family)
    shots = parse_shots(args.shots)
    if args.selector_method == "error_profile_rerank":
        default_alpha, default_beta, default_gamma = 0.75, 0.20, 0.05
    else:
        default_alpha, default_beta, default_gamma = 0.6, 0.3, 0.1
    weights = {
        "alpha": args.alpha if args.alpha is not None else default_alpha,
        "beta": args.beta if args.beta is not None else default_beta,
        "gamma": args.gamma if args.gamma is not None else default_gamma,
        "eta": args.eta,
        "lambda_quality": args.lambda_quality if args.lambda_quality is not None else 0.15,
        "keep_top_r": args.keep_top_r if args.keep_top_r is not None else 2,
        "coverage_lambda": args.coverage_lambda,
        "span_embedding_model": args.span_embedding_model,
        "span_demo_embedding_file": args.span_demo_embedding_file,
        "span_query_embedding_file": args.span_query_embedding_file,
        "critical_threshold": args.critical_threshold,
        "major_threshold": args.major_threshold,
    }

    for row in tqdm(test_rows, desc=f"run {args.selector_method}"):
        query_id = get_id(row)
        candidates = retriever.retrieve(row.get("src", ""), args.candidate_size)
        candidate_ids = [candidate.get("id") for candidate in candidates]
        query_profile = query_profiles.get(query_id)

        for k in shots:
            selected_demos, selected_indices = select_demos(
                query=row,
                candidates=candidates,
                k=k,
                method=args.selector_method,
                weights=weights,
                query_profile=query_profile,
                random_seed=args.seed + stable_int_hash(query_id) + k * 1009,
            )
            translation_prompt = build_translation_prompt(
                args.src_lang,
                args.tgt_lang,
                row.get("src", ""),
                selected_demos,
                args.model_family,
                tokenizer=tokenizer,
            )
            raw_output = generate_text(
                model,
                tokenizer,
                translation_prompt,
                args.model_family,
                max_new_tokens=args.max_new_tokens,
            )
            prediction = clean_translation_output(raw_output, args.tgt_lang, args.model_family)
            out_row = {
                "query_id": query_id,
                "query_src": row.get("src", ""),
                "ref": row.get("tgt"),
                "k": k,
                "retriever": args.retriever,
                "retriever_embedding_model": args.retriever_embedding_model if args.retriever != "bm25" else "",
                "sonar_model": args.sonar_model if args.retriever == "sonar" else "",
                "sonar_backend": args.sonar_backend if args.retriever == "sonar" else "",
                "sonar_backend_requested": args.sonar_backend if args.retriever == "sonar" else "",
                "sonar_backend_actual": getattr(retriever, "actual_backend", "") if args.retriever == "sonar" else "",
                "selector_method": args.selector_method,
                "model_family": args.model_family,
                "translator_model": args.translator_model,
                "model_tag": args.model_tag,
                "lang_pair": args.lang_pair,
                "prompt_version": args.prompt_version,
                "candidate_size": args.candidate_size,
                "candidate_ids": candidate_ids,
                "selected_candidate_indices": selected_indices,
                "selected_demo_ids": [demo.get("id") for demo in selected_demos],
                "selected_demos": selected_demos,
                "prompt": translation_prompt,
                "raw_output": raw_output,
                "prediction": prediction,
                "weights": weights,
            }
            for debug_key in (
                "num_high_risk_in_candidate_pool",
                "num_high_risk_in_original_topk",
                "num_fixed_safe_anchors",
                "num_unsafe_anchors_released",
                "num_filtered_candidates_before_selected_full",
                "num_refill_fallback",
            ):
                if selected_demos and debug_key in selected_demos[0]:
                    out_row[debug_key] = selected_demos[0][debug_key]
            if is_bad_prediction(prediction):
                print(
                    f"Warning: suspicious prediction for query_id={query_id}, k={k}: {prediction!r}",
                    file=sys.stderr,
                )
                out_row["raw_generation"] = raw_output
            if args.save_raw_outputs:
                out_row["raw_generation"] = raw_output
            if args.save_prompts:
                out_row["translation_prompt"] = translation_prompt
            append_jsonl(args.output_file, out_row)


if __name__ == "__main__":
    main()
