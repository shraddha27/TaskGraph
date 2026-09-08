#!/usr/bin/env python3
"""Compare the original and fine-tuned sentence-transformer models on several task prompts."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Tuple

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

ROOT = Path(__file__).resolve().parent.parent
ORIGINAL_MODEL = str(ROOT / "backend_fastapi" / "hg_model")
FINE_TUNED_MODEL = str(ROOT / "backend_fastapi" / "fine_tuned_task_embedder")

PROMPTS = [
    "create a task to Demonstrate the company project",
    "How many tasks are pending?",
    "Search for all demo tasks",
    "Update the launch task to mark it completed",
    "Create a task for Testing & QA and update the launch task",
    "List all completed tasks",
    "Get task stats for high priority tasks",
    "Delete the completed personal task",
]

CANDIDATE_TEXTS = {
    "demo": [
        "Please add a task to demonstrate my company project",
        "Create a task for Internal Project Demo",
        "Demonstrate the company project",
        "Company Project Demonstration",
        "Create a task for Test Project Thoroughly and Prepare Demo",
        "Search for all demo tasks",
        "Create a task for Launch",
        "Review Q3 Budget",
        "Create a task for Front-End Development",
    ],
    "stats": [
        "How many tasks are pending?",
        "Count the pending tasks.",
        "Show me the number of open tasks.",
        "Give me pending task statistics.",
        "Get task stats for high priority tasks",
        "List all completed tasks",
        "How many tasks are in reopen status?",
    ],
    "search": [
        "Search for all demo tasks",
        "Find all QA tasks",
        "Find all frontend tasks",
        "List all completed tasks",
        "Search for design tasks and update the wireframe creation task",
        "Create a task for Launch",
    ],
    "update": [
        "Update the launch task to mark it completed",
        "Update the wireframe creation task to reopen it",
        "Update the review Q3 budget task to reopen it",
        "Create a task for Launch",
        "Delete the completed personal task",
        "Create a task for Testing & QA and update the launch task",
    ],
    "delete": [
        "Delete the old logout and attend function task",
        "Delete the completed personal task",
        "Delete the old personal task and search for all completed tasks",
        "Create a task for Launch",
        "Update the launch task to mark it completed",
    ],
    "list": [
        "List all completed tasks",
        "List all reopened tasks",
        "Show all open tasks",
        "Get task stats for high priority tasks",
        "Search for all demo tasks",
        "Create a task for UI Design",
    ],
}


def rank_candidates(model_name: str, query: str, candidates: Iterable[str]) -> List[Tuple[str, float]]:
    model = SentenceTransformer(model_name)
    candidate_list = list(candidates)
    q_emb = model.encode(query, normalize_embeddings=True)
    c_emb = model.encode(candidate_list, normalize_embeddings=True)
    sims = cosine_similarity([q_emb], c_emb)[0]
    ranked = sorted(zip(candidate_list, sims), key=lambda item: item[1], reverse=True)
    return ranked


def choose_candidates(prompt: str) -> List[str]:
    lowered = prompt.lower()
    if "company project" in lowered or "demonstrate" in lowered:
        return CANDIDATE_TEXTS["demo"]
    if "pending" in lowered or "stats" in lowered or "how many" in lowered or "count" in lowered:
        return CANDIDATE_TEXTS["stats"]
    if "search" in lowered or "find" in lowered:
        return CANDIDATE_TEXTS["search"]
    if "update" in lowered:
        return CANDIDATE_TEXTS["update"]
    if "delete" in lowered:
        return CANDIDATE_TEXTS["delete"]
    if "list" in lowered or "show" in lowered:
        return CANDIDATE_TEXTS["list"]
    return CANDIDATE_TEXTS["demo"]


def main() -> None:
    models = {
        "original": ORIGINAL_MODEL,
        "fine_tuned": FINE_TUNED_MODEL,
    }

    for prompt in PROMPTS:
        candidates = choose_candidates(prompt)
        print("\n" + "=" * 90)
        print(f"PROMPT: {prompt}")
        print("=" * 90)
        print(f"CANDIDATE SET: {len(candidates)} items")

        for label, model_path in models.items():
            print(f"\nMODEL: {label} -> {model_path}")
            for text, score in rank_candidates(model_path, prompt, candidates)[:5]:
                print(f"  {score:.4f} | {text}")


if __name__ == "__main__":
    main()
