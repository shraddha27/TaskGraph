#!/usr/bin/env python3
"""
Upload a JSONL file to Mistral, start a fine-tuning job, and print the final tuned model id.

Usage:
    export MISTRAL_API_KEY="..."
    export MISTRAL_BASE_URL="https://api.mistral.ai/v1"
    python upload_and_finetune_mistral.py \
      --file backend_fastapi/mistral_finetune_training_data_combined.jsonl \
      --base-model mistral-small-latest \
      --suffix task-assistant-v1
"""

import argparse
import json
import os
import time
from pathlib import Path

import requests


def _verify_ssl() -> bool:
    value = os.getenv("MISTRAL_VERIFY_SSL", "true").strip().lower()
    return value not in {"0", "false", "no", "off"}


def upload_training_file(api_key: str, file_path: Path, base_url: str = "https://api.mistral.ai/v1") -> dict:
    url = f"{base_url.rstrip('/')}/files"
    verify_ssl = _verify_ssl()
    with file_path.open("rb") as fh:
        files = {"file": (file_path.name, fh, "application/jsonl")}
        data = {"purpose": "fine-tune"}
        response = requests.post(
            url,
            headers={"Authorization": f"Bearer {api_key}"},
            files=files,
            data=data,
            timeout=120,
            verify=verify_ssl,
        )

    try:
        response.raise_for_status()
    except requests.HTTPError as exc:
        raise RuntimeError(f"Mistral upload failed: {response.text}") from exc

    return response.json()


def create_fine_tuning_job(
    api_key: str,
    training_file_id: str,
    base_model: str,
    suffix: str,
    base_url: str = "https://api.mistral.ai/v1",
) -> dict:
    url = f"{base_url.rstrip('/')}/fine_tuning/jobs"
    payload = {
        "model": base_model,
        "training_files": [training_file_id],
        "suffix": suffix,
    }
    verify_ssl = _verify_ssl()

    response = requests.post(
        url,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json=payload,
        timeout=120,
        verify=verify_ssl,
    )

    try:
        response.raise_for_status()
    except requests.HTTPError as exc:
        raise RuntimeError(f"Fine-tuning job creation failed: {response.text}") from exc

    return response.json()


def get_job_status(api_key: str, job_id: str, base_url: str = "https://api.mistral.ai/v1") -> dict:
    url = f"{base_url.rstrip('/')}/fine_tuning/jobs/{job_id}"
    verify_ssl = _verify_ssl()
    response = requests.get(
        url,
        headers={"Authorization": f"Bearer {api_key}"},
        timeout=120,
        verify=verify_ssl,
    )

    try:
        response.raise_for_status()
    except requests.HTTPError as exc:
        raise RuntimeError(f"Job status request failed: {response.text}") from exc

    return response.json()


def wait_for_tuned_model_id(api_key: str, job_id: str, base_url: str = "https://api.mistral.ai/v1", timeout_seconds: int = 1800) -> str:
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        status = get_job_status(api_key, job_id, base_url)
        state = str(status.get("status", "")).lower()

        tuned_model_id = (
            status.get("model")
            or status.get("result_model")
            or status.get("fine_tuned_model")
            or status.get("trained_model")
        )
        if tuned_model_id:
            return tuned_model_id

        if state in {"failed", "cancelled", "canceled"}:
            raise RuntimeError(f"Fine-tuning job failed: {json.dumps(status, indent=2)}")

        if state in {"succeeded", "completed", "success"}:
            tuned_model_id = (
                status.get("model")
                or status.get("result_model")
                or status.get("fine_tuned_model")
                or status.get("trained_model")
            )
            if tuned_model_id:
                return tuned_model_id
            raise RuntimeError(f"Training completed without model id: {json.dumps(status, indent=2)}")

        time.sleep(15)

    raise TimeoutError(f"Timed out after {timeout_seconds}s waiting for job {job_id}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Upload a JSONL file to Mistral and fine-tune a model.")
    parser.add_argument("--file", required=True, help="Path to the JSONL file to upload")
    parser.add_argument("--base-model", default="mistral-small-latest", help="Base model to fine-tune")
    parser.add_argument("--suffix", default="task-assistant-v1", help="Suffix for tuned model name")
    parser.add_argument("--base-url", default=os.getenv("MISTRAL_BASE_URL", "https://api.mistral.ai/v1"), help="Mistral base URL")
    args = parser.parse_args()

    api_key = os.getenv("MISTRAL_API_KEY", "").strip()
    if not api_key:
        raise SystemExit("Set MISTRAL_API_KEY first.")

    file_path = Path(args.file).expanduser().resolve()
    if not file_path.exists():
        raise SystemExit(f"File does not exist: {file_path}")

    print(f"Uploading {file_path.name} to Mistral...")
    upload_result = upload_training_file(api_key, file_path, args.base_url)
    training_file_id = upload_result.get("id") or upload_result.get("file_id")
    if not training_file_id:
        print(json.dumps(upload_result, indent=2))
        raise SystemExit("Upload succeeded but no file id was returned.")

    print(f"training_file_id={training_file_id}")

    print("Starting fine-tuning job...")
    job_result = create_fine_tuning_job(
        api_key=api_key,
        training_file_id=training_file_id,
        base_model=args.base_model,
        suffix=args.suffix,
        base_url=args.base_url,
    )
    job_id = job_result.get("id") or job_result.get("job_id")
    if not job_id:
        print(json.dumps(job_result, indent=2))
        raise SystemExit("Fine-tune job created but no job id returned.")

    print(f"job_id={job_id}")
    tuned_model_id = wait_for_tuned_model_id(api_key, job_id, args.base_url)
    print(f"tuned_model_id={tuned_model_id}")
    print("Set MISTRAL_MODEL=" + tuned_model_id + " in your environment to use the tuned model.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
