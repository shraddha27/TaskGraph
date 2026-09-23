"""
Fine-tune the local sentence-transformer embedding model for task intent matching.

This is the sentence-transformer equivalent of SFT for embeddings:
- user task queries are positive matches for the canonical task titles/descriptions
- we train with a contrastive objective so similar task intents map close together
- the local model remains all-MiniLM-L6-v2, but is adapted to your task domain

This script reads the project task dataset from it_projects_training_data.jsonl and
creates query -> task-text pairs for training.

Important note:
For sentence-transformer embedding models, the standard fine-tuning method is not
Causal-LM SFT with `trl.SFTTrainer`. Instead, we use contrastive training
(MultipleNegativesRankingLoss or CosineSimilarityLoss), which is the correct
supervised fine-tuning pattern for semantic search / retrieval.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

from sentence_transformers import InputExample, SentenceTransformer, losses
from torch.utils.data import DataLoader

from backend_fastapi.ops import embedding_model_manifest


DEFAULT_LOCAL_MODEL = "./hg_model"
DEFAULT_DATASETS = [
    "../it_projects_training_data.jsonl",
    "../it_projects_training_data_intent_style.jsonl",
    "../it_projects_training_data_combined_intent.jsonl",
]


def _clean_text(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    return " ".join(text.split())


def _extract_assistant_payload(content: str) -> Dict[str, Any]:
    """Parse the assistant JSON payload if it is present in the message."""
    text = content.strip()
    if not text:
        return {}

    # The assistant output is usually JSON text, sometimes wrapped in backticks
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        if cleaned.lower().startswith("json"):
            cleaned = cleaned[4:].lstrip()

    try:
        return json.loads(cleaned)
    except Exception:
        return {}


def build_training_pairs(jsonl_paths: str | Path | List[str | Path]) -> List[Tuple[str, str]]:
    """
    Convert the task dataset(s) into query -> relevant text pairs.

    This loads the main project dataset plus the intent-style and combined-intent
    variants so the embedding model learns both single-action and multi-action task prompts.
    """
    if isinstance(jsonl_paths, (str, Path)):
        jsonl_paths = [jsonl_paths]

    pairs: List[Tuple[str, str]] = []

    for file_path in jsonl_paths:
        jsonl_path = Path(file_path)
        if not jsonl_path.exists():
            raise FileNotFoundError(f"Training data not found: {jsonl_path}")

        with jsonl_path.open("r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue

                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    print(f"Skipping malformed JSON on line {line_no}: {exc}")
                    continue

                messages = record.get("messages") or []
                if len(messages) < 2:
                    continue

                user_text = _clean_text(messages[0].get("content"))
                assistant_text = _clean_text(messages[-1].get("content"))

                if not user_text or not assistant_text:
                    continue

                payload = _extract_assistant_payload(assistant_text)
                if not isinstance(payload, dict):
                    continue

                args = payload.get("args")
                if not isinstance(args, dict):
                    continue

                title = _clean_text(args.get("title"))
                description = _clean_text(args.get("description"))

                positives = []
                if title:
                    positives.append(title)
                if description:
                    positives.append(description)

                if title and description:
                    positives.append(f"{title} - {description}")

                for positive_text in positives:
                    if user_text and positive_text:
                        pairs.append((user_text, positive_text))

    unique_pairs: List[Tuple[str, str]] = []
    seen = set()
    for pair in pairs:
        key = (pair[0].lower(), pair[1].lower())
        if key not in seen:
            seen.add(key)
            unique_pairs.append(pair)

    return unique_pairs


def make_training_examples(pairs: Iterable[Tuple[str, str]]) -> List[InputExample]:
    """Package query-positive pairs into sentence-transformers InputExample objects."""
    examples: List[InputExample] = []
    for query, positive in pairs:
        examples.append(InputExample(texts=[query, positive]))
    return examples


def train_sentence_transformer(
    model_name_or_path: str,
    training_data_paths: str | Path | List[str | Path],
    output_dir: str,
    epochs: int = 3,
    batch_size: int = 16,
    warmup_steps: int = 50,
    learning_rate: float = 2e-5,
    use_cosine_loss: bool = False,
) -> SentenceTransformer:
    """
    Train the local sentence-transformer model using a supervised semantic-search objective.

    This is the equivalent of SFT for an embedding model:
      - query -> positive task text pairs
      - model learns to map semantically similar task intents close in embedding space
    """
    pairs = build_training_pairs(training_data_paths)
    if not pairs:
        raise ValueError(f"No training pairs were created from the provided training files: {training_data_paths}")

    print(f"Built {len(pairs)} training pairs from {training_data_paths}")
    examples = make_training_examples(pairs)

    model = SentenceTransformer(model_name_or_path)

    # Use the standard sentence-transformers objective for retrieval / semantic matching.
    # MultipleNegativesRankingLoss is the best default when each query has a positive answer.
    # It works very naturally for task search and request matching.
    train_examples = examples
    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=batch_size)

    if use_cosine_loss:
        loss = losses.CosineSimilarityLoss(model=model)
    else:
        loss = losses.MultipleNegativesRankingLoss(model=model)

    model.fit(
        train_objectives=[(train_dataloader, loss)],
        epochs=epochs,
        warmup_steps=warmup_steps,
        optimizer_params={"lr": learning_rate},
        output_path=output_dir,
        save_best_model=True,
    )

    manifest = embedding_model_manifest(
        model_name=model_name_or_path,
        model_path=output_dir,
        backend="local_sentence_transformer",
        dimension=model.get_sentence_embedding_dimension(),
    )
    manifest.update({
        "training_pairs": len(pairs),
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
    })
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    (Path(output_dir) / "embedding_manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )

    return model


def encode_similarity(model_path: str, query: str, candidates: List[str]) -> List[Tuple[str, float]]:
    """Example: compute similarity from query to candidate task texts."""
    model = SentenceTransformer(model_path)
    query_embedding = model.encode(query)
    candidate_embeddings = model.encode(candidates)

    from sklearn.metrics.pairwise import cosine_similarity

    sims = cosine_similarity([query_embedding], candidate_embeddings)[0]
    ranked = sorted(zip(candidates, sims), key=lambda item: item[1], reverse=True)
    return ranked


if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parent
    local_model = str(base_dir / DEFAULT_LOCAL_MODEL.lstrip("./"))
    training_data_paths = [
        str(base_dir.parent / "it_projects_training_data.jsonl"),
        str(base_dir.parent / "it_projects_training_data_intent_style.jsonl"),
        str(base_dir.parent / "it_projects_training_data_combined_intent.jsonl"),
    ]

    output_dir = str(base_dir / "fine_tuned_task_embedder")

    print("=" * 72)
    print("Training local sentence-transformer for task intent and semantic search")
    print("=" * 72)
    print(f"Local model: {local_model}")
    print(f"Training data: {training_data_paths}")
    print(f"Output dir: {output_dir}")

    model = train_sentence_transformer(
        model_name_or_path=local_model,
        training_data_paths=training_data_paths,
        output_dir=output_dir,
        epochs=3,
        batch_size=16,
        warmup_steps=50,
        learning_rate=2e-5,
    )

    sample_query = "Create a task for implementing OAuth 2.0 login"
    sample_candidates = [
        "Implement user authentication",
        "Set up CI/CD pipeline with GitHub Actions",
        "Add Redis caching layer for task list APIs",
        "Implement rate limiting and DDoS protection",
    ]

    print("\nSample ranking:")
    for text, score in encode_similarity(output_dir, sample_query, sample_candidates):
        print(f"  score={score:.4f} | {text}")
