#!/usr/bin/env python3
"""
Compare the original base model against a trained LoRA adapter side by side.

Example:
    python backend_fastapi/compare_lora_vs_base.py \
        --base-model distilgpt2 \
        --adapter-dir ./local_lora_adapter \
        --prompt "User: Give me a short project summary.\n\nAssistant:"
"""

import argparse
import json
import os
import re
import sys

os.environ.setdefault("TRANSFORMERS_NO_TORCHVISION", "1")

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def resolve_model_path(model_id: str):
    return os.path.abspath(model_id) if os.path.exists(model_id) else model_id


def load_tokenizer(model_id: str, local_files_only: bool = False):
    model_id = resolve_model_path(model_id)
    tokenizer = AutoTokenizer.from_pretrained(model_id, local_files_only=local_files_only)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_base_model(model_id: str, local_files_only: bool = False):
    model_id = resolve_model_path(model_id)
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    kwargs = {"torch_dtype": dtype}
    if torch.cuda.is_available():
        kwargs["device_map"] = "auto"

    model = AutoModelForCausalLM.from_pretrained(model_id, local_files_only=local_files_only, **kwargs)
    model.eval()
    return model


def load_lora_model(base_model_id: str, adapter_dir: str, local_files_only: bool = False):
    adapter_config_path = os.path.join(adapter_dir, "adapter_config.json")
    if os.path.exists(adapter_config_path):
        with open(adapter_config_path, "r", encoding="utf-8") as f:
            adapter_config = json.load(f)
        expected_base_model = adapter_config.get("base_model_name_or_path")
        if expected_base_model and resolve_model_path(base_model_id) != resolve_model_path(expected_base_model):
            raise ValueError(
                f"Adapter {adapter_dir!r} was trained for base model {expected_base_model!r}, "
                f"but --base-model is {base_model_id!r}. Use --base-model {expected_base_model!r}."
            )
    base_model = load_base_model(base_model_id, local_files_only=local_files_only)
    model = PeftModel.from_pretrained(base_model, adapter_dir, local_files_only=local_files_only)
    model.eval()
    return model


def default_db_tasks():
    return [
        {"task": "Fix login bug", "description": "Status: open. Priority: high. Due: Friday. Fix the login bug by Friday."},
        {"task": "Prepare release notes", "description": "Status: open. Priority: medium. Due: Friday. Prepare release notes for the mobile app."},
        {"task": "Review deployment checklist", "description": "Status: open. Priority: medium. Review deployment checklist before release."},
        {"task": "Update user profile page", "description": "Status: in progress. Update user profile page design and user settings."},
        {"task": "Launch", "description": "Status: completed. Deploy the redesigned website to the live server and monitor for issues."},
        {"task": "Testing & QA", "description": "Status: completed. Conduct cross-browser and device testing; fix bugs and optimize performance."},
        {"task": "Wireframe Creation", "description": "Status: reopened. Develop low-fidelity wireframes for homepage and key landing pages."},
        {"task": "Review Q3 Budget", "description": "Status: reopened. Analyze quarterly spending report and identify cost reductions."},
        {"task": "Internal Project Demo", "description": "Status: completed. Show the demo of the project by the 21st of July."},
        {"task": "Complete AI Project Feature Update", "description": "Status: completed. Finalize the project for task management by today."},
    ]


def normalize_task_record(item):
    if not isinstance(item, dict):
        return {"task": "Untitled task", "description": ""}

    title = item.get("task") or item.get("title") or item.get("name") or "Untitled task"
    description = item.get("description") or item.get("details") or ""

    # Support older or mixed records where status/priority/due metadata may still exist as fields.
    if not description:
        extra_bits = []
        for key, value in item.items():
            if key in {"task", "title", "name", "id"}:
                continue
            if value not in (None, ""):
                extra_bits.append(f"{key.replace('_', ' ').title()}: {value}")
        description = "; ".join(extra_bits)

    return {"task": str(title), "description": str(description).strip()}


def load_tasks_from_file(path: str):
    if not path or not os.path.exists(path):
        return default_db_tasks()

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict):
        if "tasks" in data and isinstance(data["tasks"], list):
            data = data["tasks"]
        elif "records" in data and isinstance(data["records"], list):
            data = data["records"]
        else:
            data = [data]

    if not isinstance(data, list):
        raise ValueError(f"Expected a list of task objects in {path}")

    normalized = [normalize_task_record(item) for item in data if isinstance(item, dict)]
    return normalized or default_db_tasks()


def format_task_record(task):
    title = task.get("task") or task.get("title") or "Untitled task"
    description = task.get("description") or ""
    if description:
        return f"Task: {title}. Description: {description}"
    return f"Task: {title}."


TASK_MANAGER_SYSTEM_PROMPT = (
    "You are TaskFlow AI. For every task action, output exactly one JSON object and nothing else. "
    "Use the schema {\"tool\": string, \"args\": object, \"intent\": string, \"confirm\": boolean}."
)


def build_rag_prompt(question: str, tasks=None, tokenizer=None):
    tasks = tasks or default_db_tasks()
    retrieved_chunks = [format_task_record(task) for task in tasks[:8]]
    context_text = "\n".join(f"- {chunk}" for chunk in retrieved_chunks)
    messages = [
        {"role": "system", "content": TASK_MANAGER_SYSTEM_PROMPT},
        {"role": "user", "content": f"Context:\n{context_text}\n\nTask request: {question}"},
    ]
    if tokenizer is not None:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return f"System:\n{TASK_MANAGER_SYSTEM_PROMPT}\n\nContext:\n{context_text}\n\nUser Question: Task request: {question}\n\nAssistant:"


def build_strict_prompt(question: str, tokenizer):
    messages = [
        {"role": "system", "content": TASK_MANAGER_SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def detect_response_format(response: str):
    text = (response or "").strip()
    if not text:
        return "empty"
    if text.startswith("{") or text.startswith("["):
        try:
            json.loads(text)
            return "json"
        except json.JSONDecodeError:
            return "json-like"
    return "text"


def expected_tool_for_intent(intent: str):
    return {
        "create": "create_task",
        "complete": "complete_task",
        "update": "update_task",
        "delete": "delete_task",
        "deadline": "search_tasks",
        "stats": "get_task_stats",
        "search": "search_tasks",
        "list": "list_tasks",
    }.get(intent)


def response_matches_expected_tool(response: str, expected_tool: str):
    if not expected_tool:
        return "n/a"
    try:
        payload = json.loads((response or "").strip())
    except json.JSONDecodeError:
        return "no"
    return "yes" if payload.get("tool") == expected_tool else "no"


def generate_response(model, tokenizer, prompt: str, max_new_tokens: int = 120, temperature: float = 0.8):
    inputs = tokenizer(prompt, return_tensors="pt")
    device = next(model.parameters()).device
    inputs = {key: value.to(device) for key, value in inputs.items()}
    do_sample = temperature > 0

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            **({"temperature": temperature, "top_p": 0.95} if do_sample else {}),
            repetition_penalty=1.1,
            pad_token_id=tokenizer.eos_token_id,
        )

    prompt_length = inputs["input_ids"].shape[1]
    return tokenizer.decode(outputs[0][prompt_length:], skip_special_tokens=True).strip()


def format_json_output(response: str):
    try:
        payload = json.loads(response.strip())
    except json.JSONDecodeError:
        return "Not valid JSON"
    return json.dumps(payload, indent=2)


def parse_args():
    parser = argparse.ArgumentParser(description="Compare a base model with a trained LoRA adapter.")
    parser.add_argument(
        "--base-model",
        default=os.getenv("LOCAL_LORA_BASE_MODEL_ID", "distilgpt2"),
        help="Base model used for the LoRA adapter (default: distilgpt2)",
    )
    parser.add_argument(
        "--adapter-dir",
        default=os.getenv("LOCAL_LORA_ADAPTER_DIR", "./local_lora_adapter"),
        help="Path to the trained adapter directory",
    )
    parser.add_argument(
        "--tasks-file",
        default=os.getenv("LOCAL_LORA_TASKS_FILE", ""),
        help="Optional JSON file containing actual task records to use as retrieved context.",
    )
    parser.add_argument(
        "--prompt",
        action="append",
        default=[],
        help="Prompt to compare. Repeat this flag for multiple prompts.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=120)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Load the base model only from the local Hugging Face cache.",
    )
    return parser.parse_args()


INTENT_LABELS = [
    "create",
    "complete",
    "update",
    "delete",
    "deadline",
    "stats",
    "search",
    "list",
    "general",
]

INTENT_KEYWORDS = {
    "create": ["create task", "new task", "add task", "make task", "create a task"],
    "complete": ["complete", "finish", "mark done", "done", "close task"],
    "update": ["update", "edit", "rename", "change task", "modify"],
    "delete": ["delete", "remove", "erase", "trash"],
    "deadline": ["deadline", "due", "when", "overdue", "by when"],
    "stats": ["stats", "count", "how many", "completed", "pending", "total tasks"],
    "search": ["search", "find", "look for", "where is", "query"],
    "list": ["list", "show", "what tasks", "tasks", "task list"],
}


def _contains_any(text: str, keywords):
    return any(keyword in text for keyword in keywords)


def classify_intent_with_keywords(text: str):
    lowered = text.lower()
    for intent, keywords in INTENT_KEYWORDS.items():
        if _contains_any(lowered, keywords):
            return intent
    return "general"


def default_prompts():
    return [
        "Create a task with title \"Build login page\" and description \"Implement the frontend login form.\"",
        "List all open tasks.",
        "Which tasks are due this week?",
        "Update task #7 description to \"Review deployment checklist before release.\"",
        "Complete task #1.",
        "Delete task #12.",
        "Search for tasks related to login.",
        "How many tasks are pending?",
    ]


def main():
    args = parse_args()
    prompts = args.prompt or default_prompts()

    if not os.path.isdir(args.adapter_dir):
        raise FileNotFoundError(f"Adapter directory not found: {args.adapter_dir}")

    task_records = load_tasks_from_file(args.tasks_file)
    print(f"Loaded {len(task_records)} task records for comparison context.")

    print(f"Loading base model: {args.base_model}")
    base_tokenizer = load_tokenizer(args.base_model, local_files_only=args.local_files_only)
    base_model = load_base_model(args.base_model, local_files_only=args.local_files_only)
    print(f"Loading LoRA adapter: {args.adapter_dir}")
    lora_model = load_lora_model(args.base_model, args.adapter_dir, local_files_only=args.local_files_only)

    for i, prompt in enumerate(prompts, start=1):
        if "strict" in os.path.basename(os.path.normpath(args.adapter_dir)).lower():
            assembled_prompt = build_strict_prompt(prompt, base_tokenizer)
        else:
            assembled_prompt = build_rag_prompt(prompt, task_records, base_tokenizer)

        base_response = generate_response(base_model, base_tokenizer, assembled_prompt, args.max_new_tokens, args.temperature)
        lora_response = generate_response(lora_model, base_tokenizer, assembled_prompt, args.max_new_tokens, args.temperature)

        print("\n" + "=" * 80)
        print(f"TEST PROMPT {i}")
        print("=" * 80)
        intent = classify_intent_with_keywords(prompt)
        expected_tool = expected_tool_for_intent(intent)
        print(f"User question: {prompt}")
        print(f"Intent label: {intent}")
        print(f"Expected tool: {expected_tool or 'clarification/general response'}")
        print("\n--- Assembled Prompt ---")
        print(assembled_prompt)
        print(f"\n--- Original Base Model ({args.base_model}) ---")
        print("Text output:")
        print(base_response)
        print("JSON output:")
        print(format_json_output(base_response))
        print(f"[format: {detect_response_format(base_response)}]")
        print(f"[matches expected tool: {response_matches_expected_tool(base_response, expected_tool)}]")
        print("\n--- LoRA Adapter ---")
        print("Text output:")
        print(lora_response)
        print("JSON output:")
        print(format_json_output(lora_response))
        print(f"[format: {detect_response_format(lora_response)}]")
        print(f"[matches expected tool: {response_matches_expected_tool(lora_response, expected_tool)}]")
        print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
