#!/usr/bin/env python3
"""Generate a LoRA training dataset that matches the actual task schema:
only task + description fields, with extra metadata embedded in description text.
"""

import argparse
import json
from pathlib import Path

SYSTEM_PROMPT = (
    "You are TaskFlow AI, a concise task-management assistant. "
    "Use the supplied vector-search context and tool results as the source of truth. "
    "Answer the user's question in concise, natural language. Do not invent task details."
)


def normalize_task_record(item):
    if not isinstance(item, dict):
        return {"task": "Untitled task", "description": ""}
    task = item.get("task") or item.get("title") or item.get("name") or "Untitled task"
    description = item.get("description") or item.get("details") or ""
    if not description:
        fields = []
        for key, value in item.items():
            if key in {"task", "title", "name", "id"}:
                continue
            if value not in (None, ""):
                fields.append(f"{key.replace('_', ' ').title()}: {value}")
        description = "; ".join(fields)
    return {"task": str(task), "description": str(description).strip()}


def load_tasks(path: str):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict):
        data = data.get("tasks") or data.get("records") or [data]
    if not isinstance(data, list):
        raise ValueError(f"Expected a list of task objects in {path}")

    return [normalize_task_record(item) for item in data if isinstance(item, dict)]


def build_context(tasks):
    if isinstance(tasks, dict):
        tasks = [tasks]
    return "\n".join(
        f"- Task: {task['task']}. Description: {task['description']}"
        for task in tasks
    )


def create_example(prompt, tasks, tool_result, answer):
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"Context from vector search:\n{build_context(tasks)}\n\nQuestion: {prompt}",
            },
            {"role": "tool", "content": f"Tool Results:\n{json.dumps(tool_result, ensure_ascii=False)}"},
            {"role": "assistant", "content": answer},
        ]
    }


def generate_examples(tasks):
    examples = []

    for task in tasks[:10]:
        title = task["task"]
        desc = task["description"]

        examples.append(
            create_example(
                "What is the task and what does it involve?",
                task,
                {"tool": "get_task_details", "results": [{"title": title, "description": desc}], "count": 1},
                json.dumps({"task": title, "description": desc}, ensure_ascii=False),
            )
        )

        examples.append(
            create_example(
                "Summarize this task in one sentence.",
                task,
                {"tool": "search_tasks", "results": [{"title": title, "description": desc}], "count": 1},
                f"{title}: {desc}",
            )
        )

    login_task = {
        "task": "Fix login bug",
        "description": "Status: open. Priority: high. Due: Friday. Fix the login bug by Friday.",
    }
    examples.extend([
        create_example(
            'Create a task called "Build login page" with description "Implement the frontend login form."',
            [login_task],
            {
                "tool": "create_task",
                "id": 11,
                "title": "Build login page",
                "description": "Implement the frontend login form.",
                "status": "pending",
            },
            'Task "Build login page" was created successfully with a pending status.',
        ),
        create_example(
            'Update task #7 description to "Review deployment checklist before release."',
            [tasks[2] if len(tasks) > 2 else login_task],
            {
                "tool": "update_task",
                "id": 7,
                "title": "Review deployment checklist",
                "description": "Review deployment checklist before release.",
                "status": "pending",
            },
            'Task #7 "Review deployment checklist" was updated successfully.',
        ),
        create_example(
            "Delete task #12.",
            [tasks[0] if tasks else login_task],
            {"tool": "delete_task", "id": 12, "title": "Fix login bug", "status": "deleted"},
            'Task #12 "Fix login bug" was deleted successfully.',
        ),
        create_example(
            "Search for tasks related to login.",
            [login_task],
            {
                "tool": "search_tasks",
                "results": [{"id": 1, "title": "Fix login bug", "status": "pending"}],
                "count": 1,
            },
            'I found 1 task related to login: "Fix login bug" (pending).',
        ),
    ])
    return examples


def parse_args():
    parser = argparse.ArgumentParser(description="Build task examples matching the real DB schema.")
    parser.add_argument("--tasks-file", required=True, help="JSON file of task records")
    parser.add_argument("--output-file", required=True, help="Output JSONL path")
    return parser.parse_args()


def main():
    args = parse_args()
    tasks = load_tasks(args.tasks_file)
    examples = generate_examples(tasks)

    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        for example in examples:
            f.write(json.dumps(example, ensure_ascii=False) + "\n")

    print(f"Generated {len(examples)} task examples in {output_path}")


if __name__ == "__main__":
    main()
