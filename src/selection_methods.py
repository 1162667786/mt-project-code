import math
import random
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .normalization import minmax_normalize, safe_float


DEFAULT_WEIGHTS = {
    "alpha": 0.6,
    "beta": 0.3,
    "gamma": 0.1,
    "eta": 0.15,
    "lambda_quality": 0.15,
    "keep_top_r": 2,
    "coverage_lambda": 0.30,
    "chunk_size": 12,
    "chunk_stride": 6,
    "min_span_chars": 2,
    "max_query_spans": 5,
    "critical_threshold": 0,
    "major_threshold": 2,
}

_SPAN_COVERAGE_CACHE: Dict[Tuple[str, str], Dict[str, Any]] = {}


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def _with_norms(candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    copied = [deepcopy(candidate) for candidate in candidates]
    bm25_norms = minmax_normalize([candidate.get("bm25_score", candidate.get("bm25_norm", 0.0)) for candidate in copied])
    quality_norms = minmax_normalize([candidate.get("xcomet_score", 0.0) for candidate in copied])
    penalty_norms = minmax_normalize([candidate.get("error_penalty", 0.0) for candidate in copied])

    for idx, candidate in enumerate(copied):
        candidate["_candidate_index"] = idx + 1
        candidate["original_rank"] = int(_rank_value(candidate, idx + 1))
        candidate["bm25_norm"] = bm25_norms[idx]
        candidate["quality_norm"] = quality_norms[idx]
        candidate["error_penalty_norm"] = penalty_norms[idx]
    return copied


def _rank_value(candidate: Dict[str, Any], default: float = 10**9) -> float:
    return safe_float(candidate.get("bm25_rank", candidate.get("candidate_rank")), default)


def _sort_by_bm25(candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return sorted(candidates, key=lambda c: (_rank_value(c), c.get("_candidate_index", 10**9)))


def _top_by_score(candidates: List[Dict[str, Any]], k: int) -> List[Dict[str, Any]]:
    return sorted(
        candidates,
        key=lambda c: (-safe_float(c.get("selection_score"), 0.0), _rank_value(c)),
    )[:k]


def _quantile(values: List[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    q = _clamp(q)
    pos = (len(ordered) - 1) * q
    lower = int(math.floor(pos))
    upper = int(math.ceil(pos))
    if lower == upper:
        return ordered[lower]
    ratio = pos - lower
    return ordered[lower] * (1.0 - ratio) + ordered[upper] * ratio


def _finish(selected: List[Dict[str, Any]], k: int) -> Tuple[List[Dict[str, Any]], List[int]]:
    selected = selected[: max(0, k)]
    indices = [int(candidate.get("_candidate_index", 0)) for candidate in selected]
    cleaned = []
    for candidate in selected:
        item = deepcopy(candidate)
        item.pop("_candidate_index", None)
        cleaned.append(item)
    return cleaned, indices


def _weights(weights: Optional[Dict[str, Any]]) -> Dict[str, float]:
    merged = DEFAULT_WEIGHTS.copy()
    if weights:
        for key, value in weights.items():
            merged[key] = safe_float(value, merged.get(key, 0.0))
    return merged


def _cosine_similarity(a: List[float], b: List[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.5
    return _clamp((dot / (norm_a * norm_b) + 1.0) / 2.0)


def _normalize_against_candidates(value: Any, candidates: List[Dict[str, Any]], field: str) -> float:
    numeric = [safe_float(candidate.get(field, 0.0)) for candidate in candidates]
    if not numeric:
        return 0.5
    v_min = min(numeric)
    v_max = max(numeric)
    if v_max == v_min:
        return 0.5
    return _clamp((safe_float(value, 0.0) - v_min) / (v_max - v_min))


def _is_truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return False


def _coverage_fallback_bm25(
    normalized: List[Dict[str, Any]],
    k: int,
    reason: str,
    num_query_error_spans: int = 0,
) -> Tuple[List[Dict[str, Any]], List[int]]:
    selected = _sort_by_bm25(normalized)[:k]
    for item in selected:
        item["selection_score"] = item["bm25_norm"]
        item["marginal_coverage_gain"] = 0.0
        item["set_coverage_after_selection"] = 0.0
        item["matched_error_span"] = None
        item["matched_error_span_severity"] = None
        item["matched_demo_chunk"] = None
        item["matched_chunk_sim"] = 0.0
        item["num_query_error_spans"] = num_query_error_spans
        item["coverage_triggered"] = False
        item["fallback_used"] = True
        item["fallback_reason"] = reason
        item["selection_reason"] = "fallback_retriever_topk"
    return _finish(selected, k)


def _record_id(row: Optional[Dict[str, Any]]) -> Optional[str]:
    if not row:
        return None
    for key in ("id", "query_id"):
        if key in row and row[key] is not None:
            return str(row[key])
    return None


def _torch_load(path: str) -> Dict[str, Any]:
    import torch

    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _span_coverage_cache(demo_embedding_file: str, query_embedding_file: str) -> Dict[str, Any]:
    demo_path = Path(demo_embedding_file).expanduser()
    query_path = Path(query_embedding_file).expanduser()
    missing = [str(path) for path in (demo_path, query_path) if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Missing xcomet_span_coverage embedding cache file(s): "
            + ", ".join(missing)
            + ". Run scripts/precompute_span_coverage_embeddings.py before using selector_method=xcomet_span_coverage."
        )

    cache_key = (str(demo_path.resolve()), str(query_path.resolve()))
    if cache_key in _SPAN_COVERAGE_CACHE:
        return _SPAN_COVERAGE_CACHE[cache_key]

    import torch

    demo_cache = _torch_load(str(demo_path))
    query_cache = _torch_load(str(query_path))
    demo_embeddings = torch.as_tensor(demo_cache.get("embeddings"), dtype=torch.float32)
    query_embeddings = torch.as_tensor(query_cache.get("embeddings"), dtype=torch.float32)
    if demo_embeddings.numel() > 0:
        demo_embeddings = torch.nn.functional.normalize(demo_embeddings, p=2, dim=1)
    if query_embeddings.numel() > 0:
        query_embeddings = torch.nn.functional.normalize(query_embeddings, p=2, dim=1)

    demo_index: Dict[str, List[int]] = {}
    for idx, demo_id in enumerate(demo_cache.get("chunk_demo_ids", [])):
        demo_index.setdefault(str(demo_id), []).append(idx)

    query_index: Dict[str, List[int]] = {}
    for idx, query_id in enumerate(query_cache.get("query_ids", [])):
        query_index.setdefault(str(query_id), []).append(idx)

    query_span_counts = {str(key): int(value) for key, value in (query_cache.get("query_span_counts", {}) or {}).items()}
    loaded = {
        "demo_embeddings": demo_embeddings,
        "demo_chunk_texts": demo_cache.get("chunk_texts", []),
        "demo_index": demo_index,
        "query_embeddings": query_embeddings,
        "query_span_texts": query_cache.get("span_texts", []),
        "query_span_contexts": query_cache.get("span_contexts", []),
        "query_severities": query_cache.get("severities", []),
        "query_severity_weights": query_cache.get("severity_weights", []),
        "query_index": query_index,
        "query_span_counts": query_span_counts,
    }
    _SPAN_COVERAGE_CACHE[cache_key] = loaded
    return loaded


def select_demos(
    query: Dict[str, Any],
    candidates: List[Dict[str, Any]],
    k: int,
    method: str,
    weights: Optional[Dict[str, Any]],
    query_profile: Optional[Dict[str, Any]] = None,
    random_seed: int = 42,
) -> Tuple[List[Dict[str, Any]], List[int]]:
    normalized = _with_norms(candidates)
    w = _weights(weights)
    method = method.strip()
    if method == "bm25_topk":
        method = "retriever_topk"

    if k <= 0 or not normalized:
        return [], []

    if method == "retriever_topk":
        selected = _sort_by_bm25(normalized)[:k]
        for item in selected:
            item["selection_score"] = item["bm25_norm"]
            item["selection_reason"] = "retriever_topk"
        return _finish(selected, k)

    if method == "random_in_topn":
        rng = random.Random(random_seed)
        selected = deepcopy(normalized)
        rng.shuffle(selected)
        selected = selected[:k]
        for item in selected:
            item["selection_score"] = item["bm25_norm"]
            item["selection_reason"] = "random_in_topn"
        return _finish(selected, k)

    if method == "quality_only":
        for item in normalized:
            item["selection_score"] = safe_float(item.get("xcomet_score"), 0.0)
            item["selection_reason"] = "quality_only"
        return _finish(_top_by_score(normalized, k), k)

    if method == "low_quality":
        selected = sorted(
            normalized,
            key=lambda c: (safe_float(c.get("xcomet_score"), 0.0), _rank_value(c)),
        )[:k]
        for item in selected:
            item["selection_score"] = safe_float(item.get("xcomet_score"), 0.0)
            item["selection_reason"] = "low_quality"
        return _finish(selected, k)

    if method == "error_filter":
        xcomet_scores = [safe_float(item.get("xcomet_score"), 0.0) for item in normalized]
        quality_q20 = _quantile(xcomet_scores, 0.20)

        def risk_reason(item: Dict[str, Any]) -> Tuple[bool, str]:
            is_low_quality_tail = safe_float(item.get("xcomet_score"), 0.0) <= quality_q20
            has_critical_error = safe_float(item.get("num_critical"), 0.0) > 0.0
            if is_low_quality_tail and has_critical_error:
                return True, "filtered_low_quality_or_critical"
            if is_low_quality_tail:
                return True, "filtered_low_quality_tail"
            if has_critical_error:
                return True, "filtered_critical_error"
            return False, "kept_by_relevance_order"

        bm25_ordered = _sort_by_bm25(normalized)
        for item in bm25_ordered:
            is_high_risk, reason = risk_reason(item)
            item["quality_q20"] = quality_q20
            item["is_high_risk"] = is_high_risk
            item["filtered_by_error_filter"] = is_high_risk
            item["selection_reason"] = reason
            item["selection_score"] = item["bm25_norm"]

        original_topk = bm25_ordered[:k]
        selected: List[Dict[str, Any]] = []
        filtered: List[Dict[str, Any]] = []
        num_filtered_before_selected_full = 0

        for item in bm25_ordered:
            if len(selected) >= k:
                break
            if item["is_high_risk"]:
                filtered.append(item)
                num_filtered_before_selected_full += 1
                continue
            selected.append(item)

        num_refill_fallback = 0
        if len(selected) < k:
            selected_indices = {item["_candidate_index"] for item in selected}
            for item in filtered:
                if len(selected) >= k:
                    break
                if item["_candidate_index"] in selected_indices:
                    continue
                item["selection_reason"] = "risk_refill_fallback"
                selected.append(item)
                selected_indices.add(item["_candidate_index"])
                num_refill_fallback += 1

        if len(selected) < k:
            selected_indices = {item["_candidate_index"] for item in selected}
            for item in bm25_ordered:
                if len(selected) >= k:
                    break
                if item["_candidate_index"] in selected_indices:
                    continue
                item["selection_reason"] = "risk_refill_fallback"
                selected.append(item)
                selected_indices.add(item["_candidate_index"])
                num_refill_fallback += 1

        summary = {
            "num_high_risk_in_original_topk": sum(1 for item in original_topk if item["is_high_risk"]),
            "num_filtered_candidates_before_selected_full": num_filtered_before_selected_full,
            "num_refill_fallback": num_refill_fallback,
        }
        for item in selected:
            item.update(summary)

        selected = _sort_by_bm25(selected)
        return _finish(selected, k)

    if method == "quality_rerank":
        lambda_quality = safe_float(w.get("lambda_quality"), 0.15)
        xcomet_scores = [safe_float(item.get("xcomet_score"), 0.0) for item in normalized]
        quality_q20 = _quantile(xcomet_scores, 0.20)
        min_xcomet_score = min(xcomet_scores) if xcomet_scores else 0.0
        penalty_denominator = max(quality_q20 - min_xcomet_score, 1e-8)
        for item in normalized:
            xcomet_score = safe_float(item.get("xcomet_score"), 0.0)
            low_quality_penalty = 0.0
            if xcomet_score < quality_q20:
                low_quality_penalty = (quality_q20 - xcomet_score) / penalty_denominator
            item["quality_q20"] = quality_q20
            item["low_quality_penalty"] = low_quality_penalty
            item["selection_score"] = item["bm25_norm"] - lambda_quality * low_quality_penalty
            item["selection_reason"] = "quality_penalty_rerank"
        return _finish(_top_by_score(normalized, k), k)

    if method == "error_profile_rerank":
        alpha = safe_float(weights.get("alpha"), 0.75) if weights else 0.75
        beta = safe_float(weights.get("beta"), 0.20) if weights else 0.20
        gamma = safe_float(weights.get("gamma"), 0.05) if weights else 0.05
        keep_top_r = max(0, int(safe_float(w.get("keep_top_r"), 2.0)))
        xcomet_scores = [safe_float(item.get("xcomet_score"), 0.0) for item in normalized]
        quality_q20 = _quantile(xcomet_scores, 0.20)
        bm25_ordered = _sort_by_bm25(normalized)
        anchor_indices = {item["_candidate_index"] for item in bm25_ordered[:keep_top_r]}

        for item in bm25_ordered:
            item["quality_q20"] = quality_q20
            item["original_anchor_candidate"] = item["_candidate_index"] in anchor_indices
            item["is_high_risk"] = (
                safe_float(item.get("xcomet_score"), 0.0) <= quality_q20
                or safe_float(item.get("num_critical"), 0.0) > 0.0
            )
            item["fixed_by_safe_anchor"] = item["original_anchor_candidate"] and not item["is_high_risk"]
            item["fixed_by_bm25_anchor"] = item["fixed_by_safe_anchor"]
            item["unsafe_anchor_released"] = item["original_anchor_candidate"] and item["is_high_risk"]
            item["keep_top_r"] = keep_top_r
            if item["fixed_by_safe_anchor"]:
                item["selection_score"] = item["bm25_norm"]
                item["selection_reason"] = "fixed_safe_anchor"
            else:
                item["selection_score"] = (
                    alpha * item["bm25_norm"] + beta * item["quality_norm"] - gamma * item["error_penalty_norm"]
                )
                item["selection_reason"] = (
                    "unsafe_anchor_reranked"
                    if item["unsafe_anchor_released"]
                    else "safe_anchor_error_profile_rerank"
                )

        safe_anchors = [item for item in bm25_ordered if item["fixed_by_safe_anchor"]]
        selected = safe_anchors[:k]
        selected_indices = {item["_candidate_index"] for item in selected}

        if len(selected) < k:
            rerank_pool = [
                item
                for item in bm25_ordered
                if item["_candidate_index"] not in selected_indices
            ]
            selected.extend(_top_by_score(rerank_pool, k - len(selected)))

        summary = {
            "num_high_risk_in_candidate_pool": sum(1 for item in bm25_ordered if item["is_high_risk"]),
            "num_high_risk_in_original_topk": sum(1 for item in bm25_ordered[:k] if item["is_high_risk"]),
            "num_fixed_safe_anchors": sum(1 for item in selected if item.get("fixed_by_safe_anchor")),
            "num_unsafe_anchors_released": sum(1 for item in bm25_ordered[:keep_top_r] if item["is_high_risk"]),
        }
        for item in selected:
            item.update(summary)

        return _finish(selected, k)

    if method == "xcomet_span_coverage":
        coverage_lambda = safe_float(w.get("coverage_lambda"), 0.30)
        if _is_truthy((query_profile or {}).get("baseline_has_repetition", False)):
            return _coverage_fallback_bm25(normalized, k, "baseline_repetition", 0)

        demo_embedding_file = str(weights.get("span_demo_embedding_file", "") if weights else "")
        query_embedding_file = str(weights.get("span_query_embedding_file", "") if weights else "")
        if not demo_embedding_file or not query_embedding_file:
            raise FileNotFoundError(
                "xcomet_span_coverage requires --span_demo_embedding_file and --span_query_embedding_file. "
                "Run scripts/precompute_span_coverage_embeddings.py before using this selector."
            )

        cache = _span_coverage_cache(demo_embedding_file, query_embedding_file)
        query_id = _record_id(query) or _record_id(query_profile)
        if query_id is None:
            return _coverage_fallback_bm25(normalized, k, "query_id_missing", 0)
        query_indices = cache["query_index"].get(str(query_id), [])
        if not query_indices:
            span_count = cache["query_span_counts"].get(str(query_id))
            reason = "no_major_critical_spans" if span_count == 0 else "query_id_not_found_in_embedding_cache"
            return _coverage_fallback_bm25(normalized, k, reason, int(span_count or 0))

        import torch

        query_embeddings = cache["query_embeddings"][query_indices]
        span_weights = torch.tensor(
            [safe_float(cache["query_severity_weights"][idx], 0.0) for idx in query_indices],
            dtype=torch.float32,
        )
        total_weight = float(torch.clamp(span_weights.sum(), min=1e-8).item())
        query_spans = [
            {
                "text": cache["query_span_texts"][idx],
                "severity": cache["query_severities"][idx],
                "weight": float(span_weights[pos].item()),
            }
            for pos, idx in enumerate(query_indices)
        ]

        missing_demo_ids = [
            str(item.get("id"))
            for item in normalized
            if str(item.get("id")) not in cache["demo_index"]
        ]
        if missing_demo_ids:
            return _coverage_fallback_bm25(
                normalized,
                k,
                "demo_id_not_found_in_embedding_cache",
                len(query_spans),
            )

        for item in normalized:
            demo_indices = cache["demo_index"][str(item.get("id"))]
            demo_embeddings = cache["demo_embeddings"][demo_indices]
            sims = torch.clamp(query_embeddings @ demo_embeddings.T, min=0.0)
            best_sims, best_chunk_positions = torch.max(sims, dim=1)
            item["_coverage_span_sims"] = [float(value) for value in best_sims.tolist()]
            item["_coverage_span_chunks"] = [
                cache["demo_chunk_texts"][demo_indices[int(pos)]]
                for pos in best_chunk_positions.tolist()
            ]

        selected: List[Dict[str, Any]] = []
        selected_indices = set()
        best_sims = [0.0 for _ in query_spans]
        span_weight_values = [safe_float(span.get("weight"), 0.0) for span in query_spans]

        while len(selected) < k and len(selected_indices) < len(normalized):
            best_candidate = None
            best_score = None
            best_gain = 0.0
            best_match_idx = None
            for item in normalized:
                if item["_candidate_index"] in selected_indices:
                    continue
                span_sims = item.get("_coverage_span_sims", [])
                gain = (
                    sum(
                        weight * max(0.0, safe_float(sim, 0.0) - best_sims[idx])
                        for idx, (weight, sim) in enumerate(zip(span_weight_values, span_sims))
                    )
                    / total_weight
                )
                score = item["bm25_norm"] + coverage_lambda * gain
                match_idx = None
                match_delta = 0.0
                for idx, sim in enumerate(span_sims):
                    delta = max(0.0, safe_float(sim, 0.0) - best_sims[idx])
                    weighted_delta = span_weight_values[idx] * delta
                    if weighted_delta > match_delta:
                        match_delta = weighted_delta
                        match_idx = idx
                rank_key = (
                    -score,
                    _rank_value(item),
                    item.get("_candidate_index", 10**9),
                )
                if best_score is None or rank_key < best_score:
                    best_candidate = item
                    best_score = rank_key
                    best_gain = gain
                    best_match_idx = match_idx

            if best_candidate is None:
                break

            span_sims = best_candidate.get("_coverage_span_sims", [])
            best_sims = [
                max(best_sims[idx], safe_float(span_sims[idx], 0.0))
                for idx in range(min(len(best_sims), len(span_sims)))
            ]
            set_coverage = sum(weight * sim for weight, sim in zip(span_weight_values, best_sims)) / total_weight
            matched_span = query_spans[best_match_idx] if best_match_idx is not None else None
            matched_chunks = best_candidate.get("_coverage_span_chunks", [])
            best_candidate["selection_score"] = -safe_float(best_score[0], 0.0) if best_score else best_candidate["bm25_norm"]
            best_candidate["marginal_coverage_gain"] = best_gain
            best_candidate["set_coverage_after_selection"] = set_coverage
            best_candidate["matched_error_span"] = matched_span.get("text") if matched_span else None
            best_candidate["matched_error_span_severity"] = matched_span.get("severity") if matched_span else None
            best_candidate["matched_demo_chunk"] = (
                matched_chunks[best_match_idx]
                if best_match_idx is not None and best_match_idx < len(matched_chunks)
                else None
            )
            best_candidate["matched_chunk_sim"] = (
                safe_float(span_sims[best_match_idx], 0.0)
                if best_match_idx is not None and best_match_idx < len(span_sims)
                else 0.0
            )
            best_candidate["num_query_error_spans"] = len(query_spans)
            best_candidate["coverage_triggered"] = best_gain > 0.0
            best_candidate["fallback_used"] = False
            best_candidate["fallback_reason"] = None
            best_candidate["selection_reason"] = "xcomet_span_coverage"
            best_candidate.pop("_coverage_span_sims", None)
            best_candidate.pop("_coverage_span_chunks", None)
            selected.append(best_candidate)
            selected_indices.add(best_candidate["_candidate_index"])

        return _finish(selected, k)

    raise ValueError(f"Unknown selector method: {method}")
