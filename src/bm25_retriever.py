from copy import deepcopy
from typing import Any, Dict, List

import numpy as np
from rank_bm25 import BM25Okapi


def simple_tokenize(text: str) -> List[str]:
    text = "" if text is None else str(text).strip()
    if not text:
        return []
    if any(ch.isspace() for ch in text):
        return text.split()
    return list(text)


def build_bm25(demo_pool: List[Dict[str, Any]]) -> BM25Okapi:
    tokenized_corpus = [simple_tokenize(demo.get("src", "")) for demo in demo_pool]
    return BM25Okapi(tokenized_corpus)


def retrieve_candidates(
    query_src: str,
    demo_pool: List[Dict[str, Any]],
    bm25: BM25Okapi,
    candidate_size: int,
) -> List[Dict[str, Any]]:
    if candidate_size <= 0:
        candidate_size = len(demo_pool)
    query_tokens = simple_tokenize(query_src)
    scores = bm25.get_scores(query_tokens)
    ranked_indices = np.argsort(scores)[::-1][:candidate_size]

    candidates: List[Dict[str, Any]] = []
    for rank, idx in enumerate(ranked_indices, start=1):
        candidate = deepcopy(demo_pool[int(idx)])
        candidate["bm25_rank"] = rank
        candidate["bm25_score"] = float(scores[int(idx)])
        candidates.append(candidate)
    return candidates
