#!/usr/bin/env python3
"""
Upload a JSONL training file to Mistral and print the returned training_file_id.

Usage:
    export MISTRAL_API_KEY="..."
    export MISTRAL_BASE_URL="https://api.mistral.ai/v1"
    python upload_mistral_training_file.py --file backend_fastapi/mistral_finetune_training_data_combined.jsonl
"""

import argparse
import json
import os
from pathlib import Path

import requests


def upload_training_file(api_key: str, file_path: Path, base_url: str = "https://api.mistral.ai/v1") -> dict:
    url = f"{base_url.rstrip('/')}/files"

    with file_path.open("rb") as fh:
        files = {"file": (file_path.name, fh, "application/jsonl")}
        data = {"purpose": "fine-tune"}
        response = requests.post(
            url,
            headers={"Authorization": f"Bearer {api_key}"},
            files=files,
            data=data,
            timeout=120,
        )

    try:
        response.raise_for_status()
    except requests.HTTPError as exc:
        detail = response.text
        raise RuntimeError(f"Mistral upload failed: {detail}") from exc

    payload = response.json()
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Upload a JSONL file to Mistral and print the training_file_id.")
    parser.add_argument("--file", required=True, help="Path to the JSONL training file.")
    parser.add_argument("--base-url", default=os.getenv("MISTRAL_BASE_URL", "https://api.mistral.ai/v1"), help="Mistral API base URL")
    args = parser.parse_args()

    api_key = os.getenv("MISTRAL_API_KEY", "").strip()
    if not api_key:
        raise SystemExit("Please set MISTRAL_API_KEY before running this script.")

    file_path = Path(args.file).expanduser().resolve()
    if not file_path.exists():
        raise SystemExit(f"File not found: {file_path}")

    result = upload_training_file(api_key=api_key, file_path=file_path, base_url=args.base_url)
    training_file_id = result.get("id") or result.get("file_id")

    if not training_file_id:
        print(json.dumps(result, indent=2))
        raise SystemExit("Upload succeeded but no training_file_id was returned.")

    print(f"training_file_id={training_file_id}")
    print(json.dumps({"id": training_file_id, "filename": result.get("filename"), "purpose": result.get("purpose"), "status": result.get("status")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
