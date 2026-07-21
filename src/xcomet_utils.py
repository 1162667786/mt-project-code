from pathlib import Path
import os
from typing import Any, Dict, Iterable, List, Optional

from .normalization import safe_float


def resolve_comet_checkpoint(path_or_name: str) -> str:
    path = Path(path_or_name).expanduser()
    if path.is_file() and path.suffix == ".ckpt":
        return str(path)
    if path.is_dir():
        preferred = [path / "checkpoints" / "model.ckpt", path / "model.ckpt"]
        for candidate in preferred:
            if candidate.is_file():
                return str(candidate)
        ckpts = sorted(path.rglob("*.ckpt"))
        if ckpts:
            return str(ckpts[0])

    from comet import download_model

    return download_model(path_or_name)


def load_comet_model(path_or_name: str) -> Any:
    from comet import load_from_checkpoint

    checkpoint = resolve_comet_checkpoint(path_or_name)
    return load_from_checkpoint(checkpoint)


def configure_torch_load_for_comet_predictions() -> None:
    """Allow COMET prediction writer outputs under PyTorch 2.6+.

    COMET writes temporary Prediction objects with torch.save during distributed
    prediction, then gathers them with torch.load. PyTorch 2.6 changed
    torch.load's default to weights_only=True, which rejects COMET's trusted
    local Prediction class unless it is allowlisted.
    """
    os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
    try:
        import torch
        from comet.models.utils import Prediction

        torch.serialization.add_safe_globals([Prediction])
    except Exception:
        pass


def safe_get(obj: Any, key: str, default: Any = None) -> Any:
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(key, default)
    if isinstance(obj, (list, tuple)):
        for item in obj:
            if isinstance(item, (list, tuple)) and len(item) == 2 and item[0] == key:
                return item[1]
    try:
        return obj[key]
    except (KeyError, IndexError, TypeError, AttributeError):
        pass
    value = getattr(obj, key, default)
    return value


def _result_get(result: Any, key: str, default: Any = None) -> Any:
    return safe_get(result, key, default)


def _is_pair_sequence(obj: Any) -> bool:
    if not isinstance(obj, (list, tuple)):
        return False
    return all(isinstance(item, (list, tuple)) and len(item) == 2 for item in obj)


def _as_pair_dict(obj: Any) -> Dict[str, Any]:
    if _is_pair_sequence(obj):
        return {str(key): value for key, value in obj}
    return {}


def _flatten_numeric_values(value: Any) -> List[float]:
    if isinstance(value, (list, tuple)):
        flattened: List[float] = []
        for item in value:
            flattened.extend(_flatten_numeric_values(item))
        return flattened
    return [safe_float(value, 0.0)]


def scores_to_list(scores: Any) -> List[float]:
    if scores is None:
        return []
    if isinstance(scores, (int, float)):
        return [safe_float(scores, 0.0)]
    if isinstance(scores, (list, tuple)):
        return _flatten_numeric_values(scores)

    value = scores
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "tolist"):
        return _flatten_numeric_values(value.tolist())
    if hasattr(value, "item"):
        try:
            return [safe_float(value.item(), 0.0)]
        except (TypeError, ValueError):
            pass
    return [safe_float(value, 0.0)]


def comet_predict(model: Any, data: List[Dict[str, str]], batch_size: int = 8, gpus: int = 1) -> Dict[str, Any]:
    configure_torch_load_for_comet_predictions()
    try:
        raw_output = model.predict(data, batch_size=batch_size, gpus=gpus)
    except TypeError:
        try:
            raw_output = model.predict(data, batch_size=batch_size, devices=gpus)
        except TypeError:
            raw_output = model.predict(data, batch_size=batch_size)

    scores = _result_get(raw_output, "scores", None)
    if scores is None:
        scores = _result_get(raw_output, "system_score", None)
    metadata = _result_get(raw_output, "metadata", None)
    return {"raw_output": raw_output, "scores": scores, "metadata": metadata}


def _as_dict(obj: Any) -> Dict[str, Any]:
    if obj is None:
        return {}
    if isinstance(obj, dict):
        return obj
    pair_dict = _as_pair_dict(obj)
    if pair_dict:
        return pair_dict
    if hasattr(obj, "__dict__"):
        return vars(obj)
    return {}


def _iter_nested_values(obj: Any) -> Iterable[Any]:
    yield obj
    if isinstance(obj, dict):
        for value in obj.values():
            yield from _iter_nested_values(value)
    elif isinstance(obj, (list, tuple)):
        for value in obj:
            yield from _iter_nested_values(value)
    elif hasattr(obj, "__dict__"):
        yield from _iter_nested_values(vars(obj))


def _find_score(obj: Any) -> float:
    if isinstance(obj, (int, float)):
        return safe_float(obj, 0.0)
    for key in ("xcomet_score", "score", "scores", "prediction", "quality_score"):
        value = safe_get(obj, key, None)
        if key == "scores":
            score_values = scores_to_list(value)
            if score_values:
                return score_values[0]
        if value is not None and not isinstance(value, (dict, list, tuple)):
            return safe_float(value, 0.0)
    return 0.0


def _looks_like_span(obj: Any) -> bool:
    if not isinstance(obj, (dict, list, tuple)) and not hasattr(obj, "__dict__"):
        return False
    return any(safe_get(obj, key, None) is not None for key in ("severity", "confidence", "start", "end", "text", "span"))


def _find_error_spans(obj: Any) -> List[Any]:
    for value in _iter_nested_values(obj):
        for key in ("error_spans", "error_spans_info", "spans", "errors"):
            candidate = safe_get(value, key, None)
            if isinstance(candidate, list) and (not candidate or _looks_like_span(candidate[0])):
                return candidate
        if isinstance(value, list) and value and _looks_like_span(value[0]):
            return value
    return []


def extract_sample_error_spans(raw_output: Any, index: int) -> List[Any]:
    metadata = safe_get(raw_output, "metadata", None)
    all_spans = safe_get(metadata, "error_spans", None)
    if all_spans is None:
        all_spans = safe_get(raw_output, "error_spans", None)
    if isinstance(all_spans, (list, tuple)) and index < len(all_spans):
        sample_spans = all_spans[index]
        if isinstance(sample_spans, list):
            return sample_spans
        if sample_spans is None:
            return []
        return [sample_spans]

    metadata_item = None
    if isinstance(metadata, (list, tuple)) and index < len(metadata) and not _is_pair_sequence(metadata):
        metadata_item = metadata[index]
    if metadata_item is not None:
        spans = _find_error_spans(metadata_item)
        if spans:
            return spans
    return []


def _severity_name(span: Any) -> str:
    for key in ("severity", "severity_name", "label", "type"):
        value = safe_get(span, key, None)
        if value is not None:
            return str(value).lower()
    return ""


def _span_text(span: Any) -> str:
    text = safe_get(span, "text", None)
    if text is None:
        text = safe_get(span, "span", None)
    if text is None:
        text = safe_get(span, "mt_span", "")
    return "" if text is None else str(text)


def _span_start_end(span: Any) -> tuple:
    start = safe_get(span, "start", None)
    if start is None:
        start = safe_get(span, "start_idx", None)
    if start is None:
        start = safe_get(span, "start_offset", None)
    end = safe_get(span, "end", None)
    if end is None:
        end = safe_get(span, "end_idx", None)
    if end is None:
        end = safe_get(span, "end_offset", None)
    return start, end


def _span_length(span: Any) -> int:
    start, end = _span_start_end(span)
    if start is not None and end is not None:
        return max(0, int(safe_float(end, 0.0) - safe_float(start, 0.0)))
    return len(_span_text(span))


def _jsonable_error_span(span: Any) -> Dict[str, Any]:
    start, end = _span_start_end(span)
    return {
        "text": _span_text(span),
        "severity": str(safe_get(span, "severity", "")),
        "confidence": safe_float(safe_get(span, "confidence", 0.0), 0.0),
        "start": None if start is None else int(safe_float(start, 0.0)),
        "end": None if end is None else int(safe_float(end, 0.0)),
    }


def _score_at(scores: Any, index: int) -> Optional[float]:
    score_values = scores_to_list(scores)
    if index < len(score_values):
        return score_values[index]
    return None


def extract_xcomet_features(
    result_item: Any = None,
    raw_output: Any = None,
    index: Optional[int] = None,
    score: Any = None,
    error_spans_for_one_sample: Any = None,
    mt_text: Optional[str] = None,
) -> Dict[str, Any]:
    if score is None:
        score = _find_score(result_item)
        if score == 0.0 and raw_output is not None and index is not None:
            raw_scores = _result_get(raw_output, "scores", None)
            score_at_index = _score_at(raw_scores, index)
            score = score_at_index if score_at_index is not None else score

    if error_spans_for_one_sample is None:
        spans = _find_error_spans(result_item)
        if not spans and raw_output is not None and index is not None:
            spans = extract_sample_error_spans(raw_output, index)
    elif isinstance(error_spans_for_one_sample, list):
        spans = error_spans_for_one_sample
    else:
        spans = [error_spans_for_one_sample]

    jsonable_spans = [_jsonable_error_span(span) for span in spans if _looks_like_span(span)]

    severity_to_level = {"minor": 1, "major": 2, "critical": 3}
    num_minor = 0
    num_major = 0
    num_critical = 0
    max_error_severity = 0
    total_span_chars = 0

    for span in jsonable_spans:
        severity = _severity_name(span)
        if "critical" in severity:
            num_critical += 1
            max_error_severity = max(max_error_severity, severity_to_level["critical"])
        elif "major" in severity:
            num_major += 1
            max_error_severity = max(max_error_severity, severity_to_level["major"])
        elif "minor" in severity:
            num_minor += 1
            max_error_severity = max(max_error_severity, severity_to_level["minor"])
        total_span_chars += _span_length(span)

    if mt_text is None:
        mt_text = ""
        for key in ("mt", "prediction", "translation"):
            value = safe_get(result_item, key, None)
            if isinstance(value, str):
                mt_text = value
                break
    denom = max(1, len(mt_text or ""))
    error_span_ratio = min(1.0, total_span_chars / denom) if jsonable_spans else 0.0
    error_penalty = 0.1 * num_minor + 0.5 * num_major + 1.0 * num_critical + 0.3 * error_span_ratio

    return {
        "xcomet_score": safe_float(score, 0.0),
        "num_minor": int(num_minor),
        "num_major": int(num_major),
        "num_critical": int(num_critical),
        "error_span_ratio": float(error_span_ratio),
        "error_penalty": float(error_penalty),
        "max_error_severity": int(max_error_severity),
        "error_spans": jsonable_spans,
    }
