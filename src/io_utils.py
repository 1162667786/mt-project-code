import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


def ensure_parent_dir(path: str) -> None:
    parent = Path(path).expanduser().parent
    if str(parent) and str(parent) != ".":
        parent.mkdir(parents=True, exist_ok=True)


def read_jsonl(path: str, max_samples: Optional[int] = None) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    limit = None if max_samples is None or max_samples <= 0 else max_samples
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
            if limit is not None and len(rows) >= limit:
                break
    return rows


def write_jsonl(path: str, rows: Iterable[Dict[str, Any]]) -> None:
    ensure_parent_dir(path)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def append_jsonl(path: str, row: Dict[str, Any]) -> None:
    ensure_parent_dir(path)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
        f.flush()


def get_id(row: Dict[str, Any]) -> str:
    if "id" in row and row["id"] is not None:
        return str(row["id"])
    if "query_id" in row and row["query_id"] is not None:
        return str(row["query_id"])
    raise KeyError("Row does not contain an 'id' or 'query_id' field.")
