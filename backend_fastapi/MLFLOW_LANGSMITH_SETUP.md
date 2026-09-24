# MLflow and LangSmith Integration Guide

This document explains how to set up and use MLflow and LangSmith for experiment tracking and workflow tracing in the Task Assistant AI project.

## Overview

- **MLflow**: Tracks AI experiments, metrics, and artifacts (prompts, configurations, results)
- **LangSmith**: Traces LangGraph workflows and LLM calls for debugging and visibility

## Quick Start

### 1. Install Dependencies

Dependencies are already added to `requirements.txt`:
```bash
mlflow>=2.11.0
langsmith>=0.1.0
```

Install them:
```bash
pip install -r backend_fastapi/requirements.txt
```

### 2. Configure Environment Variables

Create or update your `.env` file:

```bash
# MLflow Configuration
MLFLOW_TRACKING_URI=file:./mlruns  # Local file storage (default)
# Or use remote server:
# MLFLOW_TRACKING_URI=http://localhost:5000

MLFLOW_EXPERIMENT_NAME=task-assistant-ai

# LangSmith Configuration (optional)
LANGSMITH_TRACING=true
LANGSMITH_API_KEY=your_api_key_here  # Get from https://smith.langchain.com
LANGSMITH_PROJECT=task-assistant-ai
LANGSMITH_ENDPOINT=https://api.smith.langchain.com

# MLOps and LLMOps version metadata
SERVICE_VERSION=dev
APP_ENV=development
EMBEDDING_MODEL=all-MiniLM-L6-v2
LLM_PROVIDER=mistral
MISTRAL_MODEL=mistral-tiny
PROMPT_VERSION=v1

# Production operations
OPS_METRICS_TOKEN=change-this-in-production
MODEL_REGISTRY_NAME=task-assistant
MODEL_VERSION=2026-09-10
MODEL_STAGE=production
MLFLOW_REGISTRY_URI=http://localhost:5000
LLM_INPUT_USD_PER_MILLION=0
LLM_OUTPUT_USD_PER_MILLION=0
```

### Example: Production AWS Configuration

```bash
# LangSmith with AWS endpoint (us-west-2)
export LANGSMITH_TRACING=true
export LANGSMITH_ENDPOINT=https://aws.api.smith.langchain.com
export LANGSMITH_API_KEY=lsv2_pt_YOUR_API_KEY_HERE
export LANGSMITH_PROJECT=task
```

## Using MLflow

### Local Setup (File-Based)

By default, MLflow stores runs locally in `./mlruns`:

```bash
cd backend_fastapi
python -m mlflow.server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns --host 0.0.0.0 --port 5000
```

Then access the UI at `http://localhost:5000`

### Remote Setup (PostgreSQL + S3/MinIO)

For production, use a database backend:

```bash
# Create PostgreSQL database
createdb mlflow

# Start MLflow server
mlflow server \
  --backend-store-uri postgresql://user:password@localhost/mlflow \
  --default-artifact-root s3://mybucket/mlflow \
  --host 0.0.0.0 --port 5000
```

## What Gets Tracked

### Chat Requests (`/api/ai/chat`)

MLflow tracks:
- Input message length
- Context usage flag
- Tool usage flag
- Response length
- Number of tool calls
- Number of context documents

Example:
```
Run: ai_chat
├── Params:
│   ├── message_length: 45
│   ├── use_context: True
│   ├── use_tools: True
│   ├── response_length: 230
│   ├── tool_calls_count: 2
│   └── context_docs: 3
└── Status: success
```

### Vector Search (`/api/ai/search`)

MLflow tracks:
- Query length
- Number of results
- Similarity threshold
- Execution time (seconds)

LangSmith traces:
- Query text
- Result count
- Top similarity score
- Execution latency

### Workflow Execution (`/api/workflow/execute`)

MLflow tracks:
- Input text length
- Execution time
- Number of agents used
- Workflow stages

LangSmith traces:
- User input
- Agent messages
- All workflow stages
- Execution timeline

### LLM Calls (Mistral)

MLflow tracks:
- Model name
- Temperature
- Prompt length
- Response length
- Estimated token usage
- Latency

LangSmith traces:
- Full prompt
- Full response
- Model settings
- Timestamps

### Retrieval Evaluation

Use `backend_fastapi.ops.evaluate_retrieval()` with retrieved IDs and labeled
relevant IDs to calculate `precision_at_k`, `recall_at_k`, `mrr_at_k`, and
`ndcg_at_k`. The same metrics are emitted by
`log_retrieval_quality()` into MLflow, making semantic, lexical, and RRF
experiments comparable.

Every MLflow run also receives the service version, embedding model, LLM
model, prompt version, environment, and Python version as tags. This keeps
production observations tied to the code and model configuration that created
them.

## Production Operations

The FastAPI service exposes:

- `GET /api/ops/health` for readiness checks and safe runtime/model metadata.
- `GET /api/ops/metrics` for process-local counters. Set `OPS_METRICS_TOKEN`
  to require the `X-Ops-Token` header.

LangSmith inputs, outputs, and errors are redacted for common bearer tokens,
API keys, email addresses, and phone numbers before they leave the service.
Applications should still avoid sending secrets in prompts.

LLM cost is calculated from token estimates and the configured per-million-token
rates. Set `LLM_INPUT_USD_PER_MILLION` and `LLM_OUTPUT_USD_PER_MILLION` for
meaningful cost estimates.

`MODEL_VERSION`, `PROMPT_VERSION`, and `MODEL_STAGE` provide the deployment
contract used by the health endpoint and telemetry tags. External approval and
artifact storage remain the responsibility of the configured MLflow registry.

### LLMOps Features

- Send an optional `conversation_id` to `/api/ai/chat/`; user and assistant
  messages are persisted in `conversation_messages` and the latest turns are
  supplied as bounded context.
- Configure `LLM_PROVIDER=mistral` or `LLM_PROVIDER=ollama`. If Mistral fails
  and `OLLAMA_BASE_URL` is configured, Ollama is attempted before the local
  deterministic fallback.
- Chat responses include lightweight `quality` signals: answer presence,
  evidence overlap, uncertainty disclosure, and a combined quality score.
  These are logged to MLflow as `answer_*` metrics.

The quality score is an operational signal based on available context and tool
results. It is not a replacement for human review or a model-based factuality
evaluation, so production teams should add labeled answer evaluations before
using it as an automatic release blocker.

### CI Quality Gate

The workflow in `.github/workflows/quality-gate.yml` runs operations tests and
the JSON-driven retrieval evaluation. Replace
`backend_fastapi/evaluation/retrieval_cases.json` with labeled task queries as
real evaluation data becomes available. Gate thresholds are controlled by
`MIN_RECALL_AT_K`, `MIN_MRR_AT_K`, and `MIN_NDCG_AT_K`.

### Embedding MLOps

Embedding training is run with
`backend_fastapi/sentence_transformer_task_finetune.py`. It creates query/task
positive pairs, trains with a contrastive retrieval loss, saves the model, and
writes `embedding_manifest.json` with the model fingerprint, dimension, and
training settings.

The deployed service reports the active embedding backend and model identity at
`GET /api/ops/health`. Docker backend selection is configurable with
`USE_SENTENCE_TRANSFORMERS` and `USE_REMOTE_EMBEDDING_API`; local models are
mounted through `LOCAL_EMBEDDING_MODEL_PATH`.

Embedding latency, request count, dimension, and errors are available through
`GET /api/ops/metrics`. Run
`evaluate_embedding_drift.py` against a reference sample and a current sample
to detect dimension changes, centroid movement, or norm changes before
promotion.

## Accessing Tracked Data

### MLflow UI

1. Start the server: `mlflow ui` (or use the server command above)
2. Open `http://localhost:5000`
3. Browse experiments by name
4. Compare runs side-by-side
5. Download artifacts

### MLflow API (Python)

```python
import mlflow

# Get current experiment
experiment = mlflow.get_experiment_by_name("task-assistant-ai")

# Get recent runs
runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id], max_results=10)

# Get specific run details
run = mlflow.get_run(run_id)
print(f"Metrics: {run.data.metrics}")
print(f"Params: {run.data.params}")
print(f"Tags: {run.data.tags}")
```

### LangSmith Dashboard

1. Visit `https://smith.langchain.com`
2. Select your project: "task-assistant-ai"
3. View recent runs
4. Click on runs to see full traces
5. Inspect tool calls and LLM prompts

## Experiment Tracking

### Comparing Prompt Versions

Use MLflow to test different prompts:

```python
from backend_fastapi.mlflow_tracking import log_prompt_experiment

# Run experiment 1: Original prompt
metrics_v1 = {
    "accuracy": 0.85,
    "latency_seconds": 2.3,
    "relevance_score": 0.92,
}
log_prompt_experiment(
    prompt_name="task_search_v1",
    prompt_template="Find tasks matching: {query}",
    model="mistral-tiny",
    temperature=0.0,
    metrics=metrics_v1
)

# Run experiment 2: Improved prompt
metrics_v2 = {
    "accuracy": 0.91,
    "latency_seconds": 2.1,
    "relevance_score": 0.95,
}
log_prompt_experiment(
    prompt_name="task_search_v2",
    prompt_template="Search for tasks containing: {query}. Return only relevant matches.",
    model="mistral-tiny",
    temperature=0.0,
    metrics=metrics_v2
)
```

Then compare in MLflow UI by searching "prompt_experiment" experiments.

### Evaluating Retrieval Quality

```python
from backend_fastapi.mlflow_tracking import log_retrieval_quality

# Test vector search quality
query = "important tasks due this week"
results = [...]  # Retrieval results
ground_truth_ids = [1, 5, 12]  # Known relevant task IDs

log_retrieval_quality(
    query=query,
    results=results,
    ground_truth_ids=ground_truth_ids
)
```

MLflow will log precision, recall, and average similarity score.

## Docker Integration

When running in Docker, MLflow needs artifact storage:

### Option 1: Local Volume

```yaml
services:
  fastapi:
    environment:
      MLFLOW_TRACKING_URI: file:./mlruns
    volumes:
      - mlflow_data:/app/mlruns
  
  mlflow:
    image: ghcr.io/mlflow/mlflow:latest
    ports:
      - "5000:5000"
    volumes:
      - mlflow_data:/mlflow
    command: mlflow server --host 0.0.0.0 --port 5000 --backend-store-uri sqlite:////mlflow/mlflow.db

volumes:
  mlflow_data:
```

### Option 2: Remote Storage

```yaml
services:
  fastapi:
    environment:
      MLFLOW_TRACKING_URI: postgresql://user:pass@postgres:5432/mlflow
      MLFLOW_ARTIFACT_S3_ENDPOINT_URL: http://minio:9000
      AWS_ACCESS_KEY_ID: minioadmin
      AWS_SECRET_ACCESS_KEY: minioadmin
```

## Best Practices

### 1. Log Systematically

Always log:
- Input parameters (query, message, etc.)
- Output metrics (latency, quality scores, token usage)
- Configuration (model name, temperature, thresholds)

### 2. Use Meaningful Tags

```python
mlflow.set_tags({
    "environment": "production",
    "mistral_model": "mistral-tiny",
    "retrieval_method": "vector_search",
    "user_id": user_id
})
```

### 3. Compare Runs Regularly

Compare different configurations to find optimal settings:
- Different embedding models
- Different prompt templates
- Different retrieval thresholds
- Different agent combinations

### 4. Archive Results

Download artifacts and evaluation results:
```bash
mlflow artifacts download -r <run_id> -d ./archive
```

## Troubleshooting

### MLflow Connection Issues

```python
import mlflow

# Check connection
try:
    mlflow.get_experiment_by_name("task-assistant-ai")
    print("✓ MLflow connected")
except Exception as e:
    print(f"✗ MLflow error: {e}")
```

### LangSmith Not Tracing

Check environment:
```bash
echo $LANGSMITH_API_KEY  # Should not be empty
echo $LANGSMITH_PROJECT_NAME  # Should be set
```

If not tracing, LangSmith is disabled gracefully and logging continues without traces.

### Artifact Upload Failures

Ensure artifact storage is writable:
```bash
ls -la ./mlruns  # Should be readable/writable
```

## File Structure

```
backend_fastapi/
├── mlflow_tracking.py        # MLflow utilities
├── langsmith_tracing.py       # LangSmith utilities
├── ai.py                      # Tracking decorators on endpoints
├── agents/
│   └── langraph_workflow.py   # Workflow tracing
└── requirements.txt           # Dependencies
```

## API Reference

### MLflow Functions

```python
from backend_fastapi.mlflow_tracking import (
    MLflowTracker,  # Context manager for runs
    track_chat_request,  # Decorator for chat endpoints
    track_vector_search,  # Log search metrics
    track_workflow_execution,  # Log workflow metrics
    log_prompt_experiment,  # Compare prompts
    log_retrieval_quality,  # Evaluate retrieval
    log_mistral_call,  # Log LLM calls
)
```

### LangSmith Functions

```python
from backend_fastapi.langsmith_tracing import (
    LangSmithTracer,  # Context manager for traces
    trace_workflow_execution,  # Trace workflows
    trace_agent_execution,  # Trace agents
    trace_tool_call,  # Trace tool calls
    trace_llm_call,  # Trace LLM calls
    trace_vector_search,  # Trace searches
    get_project_runs,  # Fetch recent runs
    is_langsmith_enabled,  # Check if configured
)
```

## Next Steps

1. **Start MLflow server**: `mlflow ui`
2. **Run a chat request**: Test `/api/ai/chat` endpoint
3. **View in MLflow UI**: http://localhost:5000
4. **Set LangSmith API key**: Get from https://smith.langchain.com
5. **Compare experiments**: Use MLflow UI to find optimal settings

---

For more information:
- MLflow Docs: https://mlflow.org/docs
- LangSmith Docs: https://docs.smith.langchain.com
- LangGraph Docs: https://langchain-ai.github.io/langgraph/
