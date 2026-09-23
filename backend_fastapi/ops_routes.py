"""Operational health, metrics, and model configuration endpoints."""

import hmac
import os
from typing import Optional

from fastapi import APIRouter, Header, HTTPException, status

from backend_fastapi.langsmith_tracing import is_langsmith_enabled
from backend_fastapi.embeddings import embedding_model_manifest_status
from backend_fastapi.ops import (
    get_metrics_snapshot,
    get_provider_config,
    get_registry_config,
    get_runtime_metadata,
    validate_operational_config,
)

router = APIRouter(prefix="/api/ops", tags=["operations"])


def _authorize_metrics(token: Optional[str]) -> None:
    expected = os.getenv("OPS_METRICS_TOKEN", "").strip()
    if expected and not token or expected and not hmac.compare_digest(token or "", expected):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid metrics token")


@router.get("/health")
def operational_health():
    warnings = validate_operational_config()
    return {
        "status": "degraded" if warnings else "ready",
        "warnings": warnings,
        "mlflow_configured": bool(os.getenv("MLFLOW_TRACKING_URI")),
        "langsmith_enabled": is_langsmith_enabled(),
        "runtime": get_runtime_metadata(),
        "registry": get_registry_config(),
        "embedding_model": embedding_model_manifest_status(),
        "providers": get_provider_config(),
    }


@router.get("/metrics")
def operational_metrics(x_ops_token: Optional[str] = Header(default=None)):
    _authorize_metrics(x_ops_token)
    return {
        "metrics": get_metrics_snapshot(),
        "runtime": get_runtime_metadata(),
    }
