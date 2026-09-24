"""Shared MLOps and LLMOps helpers."""

import math
import os
import platform
import re
import threading
import time
import hashlib
from pathlib import Path
from collections import defaultdict
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence


_METRICS_LOCK = threading.Lock()
_METRICS = defaultdict(float)
_SENSITIVE_PATTERNS = (
    (re.compile(r"Bearer\s+[A-Za-z0-9._~+/=-]+", re.IGNORECASE), "Bearer [REDACTED]"),
    (re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"), "[EMAIL REDACTED]"),
    (re.compile(r"\b(?:sk|pk|api)[-_][A-Za-z0-9_-]{12,}\b", re.IGNORECASE), "[KEY REDACTED]"),
    (re.compile(r"\b\+?\d[\d ()-]{7,}\d\b"), "[PHONE REDACTED]"),
)


def get_runtime_metadata() -> Dict[str, str]:
    """Return versioned metadata attached to model and LLM observations."""
    return {
        "service_version": os.getenv("SERVICE_VERSION", "dev"),
        "embedding_model": os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2"),
        "llm_provider": os.getenv("LLM_PROVIDER", "mistral"),
        "llm_model": os.getenv("MISTRAL_MODEL", "mistral-tiny"),
        "prompt_version": os.getenv("PROMPT_VERSION", "v1"),
        "environment": os.getenv("APP_ENV", "development"),
        "python_version": platform.python_version(),
    }


def file_fingerprint(path: Optional[str]) -> str:
    """Return a stable fingerprint for a local model directory or file."""
    if not path:
        return ""
    candidate = Path(path)
    if not candidate.exists():
        return "missing"
    digest = hashlib.sha256()
    files = [candidate] if candidate.is_file() else sorted(item for item in candidate.rglob("*") if item.is_file())
    for item in files:
        digest.update(str(item.relative_to(candidate if candidate.is_dir() else candidate.parent)).encode("utf-8"))
        digest.update(str(item.stat().st_size).encode("ascii"))
        digest.update(str(item.stat().st_mtime_ns).encode("ascii"))
    return digest.hexdigest()[:16]


def embedding_model_manifest(model_name: str, model_path: Optional[str], backend: str, dimension: int) -> Dict[str, Any]:
    """Describe the exact embedding model/backend used by the service."""
    return {
        "model_name": model_name,
        "model_path": model_path or "",
        "model_fingerprint": file_fingerprint(model_path),
        "backend": backend,
        "dimension": dimension,
        "service_version": os.getenv("SERVICE_VERSION", "dev"),
    }


def embedding_drift(reference: Sequence[Sequence[float]], current: Sequence[Sequence[float]]) -> Dict[str, float]:
    """Compare embedding norm and centroid statistics between two samples."""
    import numpy as np

    reference_array = np.asarray(reference, dtype=np.float32)
    current_array = np.asarray(current, dtype=np.float32)
    if reference_array.ndim != 2 or current_array.ndim != 2 or not len(reference_array) or not len(current_array):
        return {"centroid_cosine": 0.0, "mean_norm_delta": 0.0, "dimension_match": 0.0}
    reference_centroid = reference_array.mean(axis=0)
    current_centroid = current_array.mean(axis=0)
    dimensions_match = float(reference_array.shape[1] == current_array.shape[1])
    if dimensions_match:
        denominator = float(np.linalg.norm(reference_centroid) * np.linalg.norm(current_centroid))
        centroid_cosine = float(np.dot(reference_centroid, current_centroid) / denominator) if denominator else 1.0
    else:
        centroid_cosine = 0.0
    return {
        "centroid_cosine": centroid_cosine,
        "mean_norm_delta": float(abs(np.linalg.norm(reference_array, axis=1).mean() - np.linalg.norm(current_array, axis=1).mean())),
        "dimension_match": dimensions_match,
    }


def record_embedding_observation(latency: float, dimension: int, error: bool = False) -> None:
    record_metric("embedding_requests_total")
    record_metric("embedding_latency_seconds_total", latency)
    record_metric("embedding_dimension", dimension)
    if error:
        record_metric("embedding_errors_total")


def redact_text(value: Optional[str]) -> str:
    """Remove common secrets and personal identifiers before external tracing."""
    redacted = value or ""
    for pattern, replacement in _SENSITIVE_PATTERNS:
        redacted = pattern.sub(replacement, redacted)
    return redacted


def redact_payload(value: Any) -> Any:
    """Recursively redact strings in a JSON-like payload."""
    if isinstance(value, str):
        return redact_text(value)
    if isinstance(value, Mapping):
        return {str(key): redact_payload(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [redact_payload(item) for item in value]
    return value


def get_registry_config() -> Dict[str, str]:
    """Return model/prompt registry settings without exposing credentials."""
    return {
        "model_name": os.getenv("MODEL_REGISTRY_NAME", "task-assistant"),
        "model_version": os.getenv("MODEL_VERSION", os.getenv("MISTRAL_MODEL", "mistral-tiny")),
        "prompt_version": os.getenv("PROMPT_VERSION", "v1"),
        "stage": os.getenv("MODEL_STAGE", "development"),
        "registry_uri": os.getenv("MLFLOW_REGISTRY_URI", ""),
    }


def validate_operational_config() -> list[str]:
    """Return production configuration warnings, without failing local startup."""
    if os.getenv("APP_ENV", "development").lower() not in {"prod", "production"}:
        return []
    warnings = []
    if not os.getenv("MLFLOW_TRACKING_URI"):
        warnings.append("MLFLOW_TRACKING_URI is not configured")
    if not os.getenv("MODEL_VERSION"):
        warnings.append("MODEL_VERSION is not configured")
    if os.getenv("LANGSMITH_API_KEY") and not os.getenv("LANGSMITH_PROJECT"):
        warnings.append("LANGSMITH_PROJECT is not configured")
    return warnings


def record_metric(name: str, value: float = 1.0) -> None:
    """Record a process-local metric for health checks and lightweight dashboards."""
    with _METRICS_LOCK:
        _METRICS[name] += float(value)


def get_metrics_snapshot() -> Dict[str, float]:
    with _METRICS_LOCK:
        return dict(_METRICS)


def reset_metrics() -> None:
    with _METRICS_LOCK:
        _METRICS.clear()


def record_llm_observation(latency: float, prompt_tokens: int, response_tokens: int, error: bool = False) -> None:
    record_metric("llm_requests_total")
    record_metric("llm_latency_seconds_total", latency)
    record_metric("llm_prompt_tokens_total", prompt_tokens)
    record_metric("llm_response_tokens_total", response_tokens)
    if error:
        record_metric("llm_errors_total")


def calculate_llm_cost(prompt_tokens: int, response_tokens: int) -> float:
    """Calculate cost using configurable USD rates per million tokens."""
    prompt_rate = float(os.getenv("LLM_INPUT_USD_PER_MILLION", "0"))
    response_rate = float(os.getenv("LLM_OUTPUT_USD_PER_MILLION", "0"))
    return (prompt_tokens * prompt_rate + response_tokens * response_rate) / 1_000_000


def evaluate_answer_quality(response: str, context: Optional[str] = None, tool_results: Optional[str] = None) -> Dict[str, float]:
    """Return lightweight, explainable answer-quality signals without another LLM call."""
    answer = (response or "").strip()
    evidence = f"{context or ''}\n{tool_results or ''}".lower()
    answer_terms = set(re.findall(r"[a-z0-9]+", answer.lower()))
    evidence_terms = set(re.findall(r"[a-z0-9]+", evidence))
    groundedness = len(answer_terms & evidence_terms) / max(len(answer_terms), 1) if evidence else 0.0
    has_answer = float(bool(answer))
    contains_uncertainty = float(bool(re.search(r"\b(i don't know|cannot verify|not sure|could not find)\b", answer, re.I)))
    return {
        "answer_present": has_answer,
        "groundedness_overlap": min(1.0, groundedness),
        "uncertainty_disclosed": contains_uncertainty,
        "quality_score": (has_answer * 0.4) + (min(1.0, groundedness) * 0.5) + (contains_uncertainty * 0.1),
    }


def get_provider_config() -> Dict[str, Any]:
    """Return configured provider order without exposing provider credentials."""
    provider = os.getenv("LLM_PROVIDER", "mistral").strip().lower()
    fallback = os.getenv("LLM_FALLBACK_PROVIDER", "local").strip().lower()
    return {
        "primary": provider,
        "fallback": fallback,
        "configured_providers": [
            name for name, configured in (
                ("mistral", bool(os.getenv("MISTRAL_API_KEY"))),
                ("ollama", bool(os.getenv("OLLAMA_BASE_URL"))),
                ("local", True),
            ) if configured
        ],
    }


def estimate_tokens(text: Optional[str]) -> int:
    """Estimate tokens without requiring a provider-specific tokenizer."""
    return max(0, math.ceil(len(text or "") / 4))


def reciprocal_rank(retrieved_ids: Sequence[Any], relevant_ids: Iterable[Any]) -> float:
    relevant = set(relevant_ids)
    for position, item_id in enumerate(retrieved_ids, start=1):
        if item_id in relevant:
            return 1.0 / position
    return 0.0


def ndcg_at_k(retrieved_ids: Sequence[Any], relevant_ids: Iterable[Any], k: int = 10) -> float:
    """Calculate binary-relevance nDCG at k."""
    relevant = set(relevant_ids)
    ranked = list(retrieved_ids[:k])
    dcg = sum(
        1.0 / math.log2(position + 1)
        for position, item_id in enumerate(ranked, start=1)
        if item_id in relevant
    )
    ideal_hits = min(len(relevant), k)
    ideal_dcg = sum(1.0 / math.log2(position + 1) for position in range(1, ideal_hits + 1))
    return dcg / ideal_dcg if ideal_dcg else 0.0


def evaluate_retrieval(
    retrieved_ids: Sequence[Any],
    relevant_ids: Iterable[Any],
    k: int = 10,
) -> Dict[str, float]:
    """Return stable retrieval metrics suitable for CI and MLflow."""
    relevant = set(relevant_ids)
    retrieved = list(retrieved_ids[:k])
    hits = sum(item_id in relevant for item_id in retrieved)
    return {
        "precision_at_k": hits / len(retrieved) if retrieved else 0.0,
        "recall_at_k": hits / len(relevant) if relevant else 0.0,
        "mrr_at_k": reciprocal_rank(retrieved, relevant),
        "ndcg_at_k": ndcg_at_k(retrieved, relevant, k=k),
    }


def create_observation(operation: str, started_at: float, **values: Any) -> Dict[str, Any]:
    """Build a serializable request observation for either telemetry backend."""
    observation = {
        "operation": operation,
        "latency_seconds": max(0.0, time.time() - started_at),
        **get_runtime_metadata(),
    }
    observation.update(values)
    return observation