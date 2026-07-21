import hashlib
import re
import sys
from abc import ABC, abstractmethod
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List

from .bm25_retriever import build_bm25, retrieve_candidates


SONAR_OFFICIAL_INSTALL_ERROR = (
    "Official SONAR requires sonar-space and fairseq2 matching your torch/CUDA. "
    "Use SONAR_BACKEND=transformers or SONAR_BACKEND=auto to fallback."
)

SONAR_TRANSFORMERS_LOAD_ERROR = (
    "Transformers SONAR backend requires a local Transformers-format SONAR_200_text_encoder. "
    "Check --sonar_model or use SONAR_BACKEND=official."
)


class BaseRetriever(ABC):
    @abstractmethod
    def retrieve(self, query_src: str, candidate_size: int) -> List[Dict[str, Any]]:
        raise NotImplementedError


class BM25Retriever(BaseRetriever):
    def __init__(self, demo_pool: List[Dict[str, Any]]) -> None:
        self.demo_pool = demo_pool
        self.bm25 = build_bm25(demo_pool)

    def retrieve(self, query_src: str, candidate_size: int) -> List[Dict[str, Any]]:
        candidates = retrieve_candidates(query_src, self.demo_pool, self.bm25, candidate_size)
        for candidate in candidates:
            candidate["retriever_rank"] = candidate.get("bm25_rank")
            candidate["retriever_score"] = candidate.get("bm25_score")
            candidate["retriever"] = "bm25"
        return candidates


class GTERetriever(BaseRetriever):
    def __init__(
        self,
        demo_pool: List[Dict[str, Any]],
        cache_dir: str,
        embedding_model: str,
        device: str,
        batch_size: int,
        force_recompute: bool = False,
    ) -> None:
        self.demo_pool = demo_pool
        self.embedding_model = embedding_model
        self.device = device
        self.batch_size = batch_size
        self.cache_path = Path(cache_dir) / (
            f"gte_{_model_cache_tag(embedding_model)}_n{len(demo_pool)}_demo_embeddings.pt"
        )
        self.model = self._load_model(embedding_model, device)
        self.demo_embeddings = self._load_or_compute_embeddings(force_recompute)

    def retrieve(self, query_src: str, candidate_size: int) -> List[Dict[str, Any]]:
        import torch

        if candidate_size <= 0:
            candidate_size = len(self.demo_pool)
        candidate_size = min(candidate_size, len(self.demo_pool))
        if candidate_size <= 0:
            return []

        query_emb = self.model.encode(
            [query_src or ""],
            batch_size=self.batch_size,
            normalize_embeddings=True,
            convert_to_tensor=True,
        )
        query_emb = query_emb.to(self.demo_embeddings.device)
        scores = (query_emb @ self.demo_embeddings.T).squeeze(0)
        ranked_indices = torch.argsort(scores, descending=True)[:candidate_size]
        return _ranked_candidates(self.demo_pool, ranked_indices.tolist(), scores, "gte")

    @staticmethod
    def _load_model(model_name: str, device: str) -> Any:
        from sentence_transformers import SentenceTransformer

        return SentenceTransformer(
            model_name,
            device=device,
            trust_remote_code=True,
        )

    def _load_or_compute_embeddings(self, force_recompute: bool) -> Any:
        import torch

        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        if not force_recompute and self.cache_path.is_file():
            cache = _torch_load(str(self.cache_path))
            cached_ids = cache.get("demo_ids", [])
            if (
                cache.get("retriever") == "gte"
                and cache.get("model_path", cache.get("embedding_model")) == self.embedding_model
                and len(cached_ids) == len(self.demo_pool)
                and cache.get("demo_src_hash") == _demo_src_hash(self.demo_pool)
            ):
                return torch.as_tensor(cache["embeddings"], dtype=torch.float32, device=self.device)

        src_texts = [str(demo.get("src", "") or "") for demo in self.demo_pool]
        embeddings = self.model.encode(
            src_texts,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            convert_to_tensor=True,
        )
        embeddings = torch.as_tensor(embeddings, dtype=torch.float32)
        cache = {
            "retriever": "gte",
            "model_path": self.embedding_model,
            "embedding_model": self.embedding_model,
            "demo_ids": _demo_ids(self.demo_pool),
            "demo_src_hash": _demo_src_hash(self.demo_pool),
            "embeddings": embeddings.detach().cpu(),
        }
        torch.save(cache, self.cache_path)
        return embeddings.to(self.device)


class SONARRetriever(BaseRetriever):
    def __init__(
        self,
        demo_pool: List[Dict[str, Any]],
        cache_dir: str,
        model_path: str,
        backend: str,
        source_lang_code: str,
        device: str,
        batch_size: int,
        force_recompute: bool = False,
    ) -> None:
        self.demo_pool = demo_pool
        self.model_path = model_path
        self.requested_backend = backend.strip().lower()
        self.actual_backend = ""
        self.source_lang_code = source_lang_code
        self.device = device
        self.batch_size = batch_size
        self.cache_dir = Path(cache_dir)
        self.cache_path = self.cache_dir / "sonar_uninitialized_demo_embeddings.pt"
        self.tokenizer = None
        self.model = None
        self.embedder = None
        self.demo_embeddings = self._initialize_backend(force_recompute)
        print(
            f"SONAR backend requested={self.requested_backend} actual={self.actual_backend}",
            file=sys.stderr,
        )

    def retrieve(self, query_src: str, candidate_size: int) -> List[Dict[str, Any]]:
        import torch

        if candidate_size <= 0:
            candidate_size = len(self.demo_pool)
        candidate_size = min(candidate_size, len(self.demo_pool))
        if candidate_size <= 0:
            return []

        try:
            query_emb = self._encode([query_src or ""])
        except Exception as exc:
            if self.requested_backend != "auto" or self.actual_backend != "official":
                raise
            self._warn_auto_fallback(exc)
            self.demo_embeddings = self._switch_to_transformers(force_recompute=False)
            print(
                f"SONAR backend requested={self.requested_backend} actual={self.actual_backend}",
                file=sys.stderr,
            )
            query_emb = self._encode([query_src or ""])
        scores = (query_emb @ self.demo_embeddings.T).squeeze(0)
        ranked_indices = torch.argsort(scores, descending=True)[:candidate_size]
        candidates = _ranked_candidates(self.demo_pool, ranked_indices.tolist(), scores, "sonar")
        for candidate in candidates:
            candidate["sonar_backend_requested"] = self.requested_backend
            candidate["sonar_backend_actual"] = self.actual_backend
        return candidates

    def _initialize_backend(self, force_recompute: bool) -> Any:
        if self.requested_backend == "official":
            self._set_official_backend()
            return self._load_or_compute_embeddings(force_recompute)
        if self.requested_backend == "transformers":
            self._set_transformers_backend()
            return self._load_or_compute_embeddings(force_recompute)
        if self.requested_backend == "auto":
            try:
                self._set_official_backend()
                return self._load_or_compute_embeddings(force_recompute)
            except Exception as exc:
                self._warn_auto_fallback(exc)
                return self._switch_to_transformers(force_recompute)
        raise ValueError(f"Unknown SONAR backend: {self.requested_backend}")

    def _set_official_backend(self) -> None:
        self.actual_backend = "official"
        self.tokenizer = None
        self.model = None
        self.embedder = self._load_official_embedder(self.device)
        self.cache_path = self.cache_dir / (
            f"sonar_official_{_safe_name(self.source_lang_code)}_"
            f"n{len(self.demo_pool)}_demo_embeddings.pt"
        )

    def _set_transformers_backend(self) -> None:
        self.actual_backend = "transformers"
        self.tokenizer, self.model = self._load_transformers_model(self.model_path, self.device)
        self.embedder = None
        self.cache_path = self.cache_dir / (
            f"sonar_transformers_{_model_cache_tag(self.model_path)}_"
            f"{_safe_name(self.source_lang_code)}_n{len(self.demo_pool)}_demo_embeddings.pt"
        )

    def _switch_to_transformers(self, force_recompute: bool) -> Any:
        self._set_transformers_backend()
        return self._load_or_compute_embeddings(force_recompute)

    @staticmethod
    def _warn_auto_fallback(exc: Exception) -> None:
        print(
            "[WARN] Official SONAR backend unavailable. Falling back to Transformers SONAR.",
            file=sys.stderr,
        )
        print(f"[WARN] Original SONAR exception: {exc}", file=sys.stderr)

    @staticmethod
    def _load_transformers_model(model_path: str, device: str) -> Any:
        from transformers import AutoModel, AutoTokenizer

        try:
            tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
            model = AutoModel.from_pretrained(model_path, trust_remote_code=True)
            return tokenizer, model.to(device).eval()
        except Exception as trust_remote_exc:
            try:
                tokenizer = AutoTokenizer.from_pretrained(model_path)
                model = AutoModel.from_pretrained(model_path)
                return tokenizer, model.to(device).eval()
            except Exception as fallback_exc:
                raise RuntimeError(f"{SONAR_TRANSFORMERS_LOAD_ERROR} {fallback_exc}") from trust_remote_exc

    @staticmethod
    def _load_official_embedder(device: str) -> Any:
        try:
            import torch
            from sonar.inference_pipelines.text import TextToEmbeddingModelPipeline
        except (ImportError, ModuleNotFoundError, RuntimeError) as exc:
            raise ImportError(SONAR_OFFICIAL_INSTALL_ERROR) from exc

        try:
            return TextToEmbeddingModelPipeline(
                encoder="text_sonar_basic_encoder",
                tokenizer="text_sonar_basic_encoder",
                device=torch.device(device),
                dtype=torch.float16 if "cuda" in device else torch.float32,
            )
        except (ImportError, ModuleNotFoundError, RuntimeError) as exc:
            raise RuntimeError(SONAR_OFFICIAL_INSTALL_ERROR) from exc

    def _encode(self, sentences: List[str]) -> Any:
        try:
            if self.actual_backend == "official":
                return self._encode_official(sentences)
            return self._encode_transformers(sentences)
        except Exception as exc:
            if self.actual_backend == "official":
                raise RuntimeError(f"{SONAR_OFFICIAL_INSTALL_ERROR} {exc}") from exc
            raise

    def _encode_transformers(self, sentences: List[str]) -> Any:
        import torch

        if self.tokenizer is None or self.model is None:
            raise RuntimeError("SONAR transformers backend is not initialized.")
        encoded = self.tokenizer(
            sentences,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        encoded = {
            key: encoded[key].to(self.device)
            for key in ("input_ids", "attention_mask")
            if key in encoded
        }
        if "attention_mask" not in encoded:
            encoded["attention_mask"] = torch.ones_like(encoded["input_ids"])

        with torch.no_grad():
            if hasattr(self.model, "get_encoder"):
                encoder = self.model.get_encoder()
                output = encoder(
                    input_ids=encoded["input_ids"],
                    attention_mask=encoded.get("attention_mask"),
                    return_dict=True,
                )
            else:
                output = self.model(
                    input_ids=encoded["input_ids"],
                    attention_mask=encoded.get("attention_mask"),
                    return_dict=True,
                )
        hidden = output.last_hidden_state
        attention_mask = encoded["attention_mask"].unsqueeze(-1).to(hidden.dtype)
        embeddings = (hidden * attention_mask).sum(dim=1) / attention_mask.sum(dim=1).clamp(min=1e-8)
        return torch.nn.functional.normalize(embeddings.float(), p=2, dim=1)

    def _encode_official(self, sentences: List[str]) -> Any:
        import torch

        if self.embedder is None:
            raise RuntimeError("SONAR official backend is not initialized.")
        embeddings = self.embedder.predict(sentences, source_lang=self.source_lang_code)
        embeddings = torch.as_tensor(embeddings, dtype=torch.float32, device=self.device)
        return torch.nn.functional.normalize(embeddings, p=2, dim=1)

    def _load_or_compute_embeddings(self, force_recompute: bool) -> Any:
        import torch

        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        if not force_recompute and self.cache_path.is_file():
            cache = _torch_load(str(self.cache_path))
            cached_ids = cache.get("demo_ids", [])
            if (
                cache.get("retriever") == "sonar"
                and cache.get("model_path") == self._cache_model_path()
                and cache.get("sonar_backend_actual", cache.get("sonar_backend", cache.get("backend")))
                == self.actual_backend
                and cache.get("src_code", cache.get("source_lang_code")) == self.source_lang_code
                and len(cached_ids) == len(self.demo_pool)
                and cache.get("demo_src_hash") == _demo_src_hash(self.demo_pool)
            ):
                return torch.as_tensor(cache["embeddings"], dtype=torch.float32, device=self.device)

        src_texts = [str(demo.get("src", "") or "") for demo in self.demo_pool]
        chunks = [
            src_texts[start : start + self.batch_size]
            for start in range(0, len(src_texts), self.batch_size)
        ]
        embeddings = torch.cat([self._encode(chunk) for chunk in chunks], dim=0) if chunks else torch.empty((0, 0))
        cache = {
            "retriever": "sonar",
            "model_path": self._cache_model_path(),
            "sonar_model": self.model_path,
            "sonar_backend": self.actual_backend,
            "sonar_backend_requested": self.requested_backend,
            "sonar_backend_actual": self.actual_backend,
            "src_code": self.source_lang_code,
            "source_lang_code": self.source_lang_code,
            "demo_ids": _demo_ids(self.demo_pool),
            "demo_src_hash": _demo_src_hash(self.demo_pool),
            "embeddings": embeddings.detach().cpu(),
        }
        torch.save(cache, self.cache_path)
        return embeddings.to(self.device)

    def _cache_model_path(self) -> str:
        if self.actual_backend == "official":
            return "text_sonar_basic_encoder"
        return self.model_path


def build_retriever(
    name: str,
    demo_pool: List[Dict[str, Any]],
    cache_dir: str,
    embedding_model: str,
    source_lang_code: str,
    device: str,
    batch_size: int,
    sonar_model: str = "",
    sonar_backend: str = "auto",
    force_recompute: bool = False,
) -> BaseRetriever:
    normalized_name = name.strip().lower()
    if normalized_name == "bm25":
        return BM25Retriever(demo_pool)
    if normalized_name == "gte":
        return GTERetriever(
            demo_pool=demo_pool,
            cache_dir=cache_dir,
            embedding_model=embedding_model,
            device=device,
            batch_size=batch_size,
            force_recompute=force_recompute,
        )
    if normalized_name == "sonar":
        return SONARRetriever(
            demo_pool=demo_pool,
            cache_dir=cache_dir,
            model_path=sonar_model,
            backend=sonar_backend,
            source_lang_code=source_lang_code,
            device=device,
            batch_size=batch_size,
            force_recompute=force_recompute,
        )
    raise ValueError(f"Unknown retriever backend: {name}")


def _ranked_candidates(
    demo_pool: List[Dict[str, Any]],
    ranked_indices: List[int],
    scores: Any,
    retriever_name: str,
) -> List[Dict[str, Any]]:
    candidates: List[Dict[str, Any]] = []
    selected_scores = [float(scores[int(idx)].item()) for idx in ranked_indices]
    score_norms = _minmax_normalize(selected_scores)
    for rank, idx in enumerate(ranked_indices, start=1):
        score = selected_scores[rank - 1]
        candidate = deepcopy(demo_pool[int(idx)])
        candidate["bm25_rank"] = rank
        candidate["bm25_score"] = score
        candidate["bm25_norm"] = score_norms[rank - 1]
        candidate["retriever_rank"] = rank
        candidate["retriever_score"] = score
        candidate["retriever"] = retriever_name
        candidates.append(candidate)
    return candidates


def _demo_ids(demo_pool: List[Dict[str, Any]]) -> List[str]:
    return [str(demo.get("id", idx)) for idx, demo in enumerate(demo_pool)]


def _demo_src_hash(demo_pool: List[Dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    for demo in demo_pool:
        digest.update(str(demo.get("src", "") or "").encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "default"


def _model_cache_tag(model_path: str) -> str:
    path = str(model_path or "default")
    name = _safe_name(Path(path).name or path)[:80]
    digest = hashlib.sha1(path.encode("utf-8")).hexdigest()[:12]
    return f"{name}_{digest}"


def _minmax_normalize(values: List[float]) -> List[float]:
    if not values:
        return []
    low = min(values)
    high = max(values)
    if high == low:
        return [0.5 for _ in values]
    return [(value - low) / (high - low) for value in values]


def _torch_load(path: str) -> Dict[str, Any]:
    import torch

    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")
