from copy import deepcopy
from typing import Any, Dict, Iterable, List


def safe_float(x: Any, default: float = 0.0) -> float:
    try:
        if x is None:
            return default
        value = float(x)
    except (TypeError, ValueError):
        return default
    if value != value:
        return default
    if value in (float("inf"), float("-inf")):
        return default
    return value


def minmax_normalize(values: Iterable[Any]) -> List[float]:
    numeric = [safe_float(v, 0.0) for v in values]
    if not numeric:
        return []
    v_min = min(numeric)
    v_max = max(numeric)
    if v_max == v_min:
        return [0.5 for _ in numeric]
    return [(v - v_min) / (v_max - v_min) for v in numeric]


def normalize_candidate_scores(
    candidates: List[Dict[str, Any]],
    fields: Iterable[str],
) -> List[Dict[str, Any]]:
    normalized = [deepcopy(candidate) for candidate in candidates]
    for field in fields:
        values = [candidate.get(field, 0.0) for candidate in normalized]
        norm_values = minmax_normalize(values)
        norm_field = f"{field}_norm"
        for candidate, value in zip(normalized, norm_values):
            candidate[norm_field] = value
    return normalized
