import json
import math
import os
import re
import time
from difflib import SequenceMatcher
from typing import Any, Dict, List, Optional

from sqlalchemy import text
from sqlalchemy.orm import Session

from backend_fastapi.embeddings import HF_EMBEDDING_API_TOKEN, generate_embedding, generate_embeddings_batch
from backend_fastapi.models import DocumentModel, TaskModel
from backend_fastapi.rag_tools import (
    execute_tool,
    extract_create_task_fields,
    extract_update_task_fields,
    looks_like_explicit_create_task_request,
    looks_like_task_status_update_request,
    normalize_task_search_query,
    filter_tasks_by_query,
    _detect_explicit_date_constraint,
    search_tasks as rag_search_tasks,
)
from backend_fastapi.utils import format_pgvector_literal, cosine_similarity

INTENT_EXAMPLE_EMBEDDINGS_CACHE: Dict[str, List[List[float]]] = {}
_EMBEDDING_CACHE: Dict[str, tuple[float, List[float]]] = {}
_HYDE_CACHE: Dict[str, tuple[float, str]] = {}
_CACHE_TTL_SECONDS = 900


def clear_retrieval_cache() -> None:
    """Invalidate derived retrieval data after documents are changed."""
    _EMBEDDING_CACHE.clear()
    _HYDE_CACHE.clear()

DEFAULT_CHUNK_SIZE = 1200
DEFAULT_CHUNK_OVERLAP = 200


def cleanup_duplicate_documents(db: Session):
    db.execute(
        text(
            """
            WITH ranked AS (
                SELECT id,
                       ROW_NUMBER() OVER (
                           PARTITION BY task_id, chunk_index
                           ORDER BY created_at DESC, id DESC
                       ) AS rn
                FROM documents
                WHERE task_id IS NOT NULL
            )
            DELETE FROM documents
            WHERE id IN (
                SELECT id
                FROM ranked
                WHERE rn > 1
            )
            """
        )
    )
    db.commit()


def sync_task_document(db: Session, task: TaskModel):
    clear_retrieval_cache()
    db.query(DocumentModel).filter(DocumentModel.task_id == task.id).delete()
    content = f"{task.title}\n{task.description}"
    chunks = chunk_text(content)
    embeddings = generate_embeddings_batch(chunks)
    for chunk_index, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
        db.add(DocumentModel(
            task_id=task.id,
            title=task.title,
            content=chunk,
            embedding=embedding,
            chunk_index=chunk_index,
            chunk_count=len(chunks),
        ))


def chunk_text(text: str, chunk_size: int = DEFAULT_CHUNK_SIZE, overlap: int = DEFAULT_CHUNK_OVERLAP) -> List[str]:
    """Split text at document boundaries, preserving a word overlap between chunks."""
    source = (text or "").strip()
    if not source:
        return [""]
    if chunk_size <= 0 or overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be non-negative and smaller than chunk_size")

    units = []
    for unit in _chunk_units(source):
        words = unit.split()
        if len(words) <= chunk_size:
            units.append(unit)
            continue
        step = chunk_size - overlap
        units.extend(" ".join(words[index:index + chunk_size]) for index in range(0, len(words), step))

    chunks = []
    start = 0
    while start < len(units):
        end = start
        word_count = 0
        while end < len(units):
            unit_words = len(units[end].split())
            if end > start and word_count + unit_words > chunk_size:
                break
            word_count += unit_words
            end += 1
            if word_count >= chunk_size:
                break

        if end == start:
            end += 1
        chunk = "\n\n".join(units[start:end]).strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(units):
            break

        # Move back to a natural unit boundary while retaining approximately
        # the requested overlap for context continuity.
        overlap_words = 0
        next_start = end
        while end - start > 1 and next_start > start and overlap_words < overlap:
            next_start -= 1
            overlap_words += len(units[next_start].split())
        start = next_start if next_start > start else end
    return chunks


def _chunk_units(text: str) -> List[str]:
    """Build paragraph/sentence units while keeping fenced code blocks intact."""
    lines = text.splitlines()
    units = []
    current = []
    in_code_block = False

    def flush() -> None:
        value = "\n".join(current).strip()
        if value:
            units.append(value)
        current.clear()

    for line in lines:
        stripped = line.strip()
        if stripped.startswith("```"):
            if in_code_block:
                current.append(line)
                flush()
                in_code_block = False
            else:
                flush()
                current.append(line)
                in_code_block = True
            continue
        if in_code_block:
            current.append(line)
            continue
        if not stripped:
            flush()
            continue
        if re.match(r"^#{1,6}\s+", stripped) and current:
            flush()
        current.append(line)
    flush()

    sentence_units = []
    for unit in units:
        if unit.lstrip().startswith("```"):
            sentence_units.append(unit)
            continue
        heading = ""
        heading_match = re.match(r"^(#{1,6}\s+[^\n]+)\n([\s\S]*)$", unit)
        body = unit
        if heading_match:
            heading = heading_match.group(1).strip()
            body = heading_match.group(2).strip()
        sentences = [part.strip() for part in re.split(r"(?<=[.!?])\s+", body) if part.strip()]
        if heading and sentences:
            sentence_units.append(f"{heading}\n{sentences[0]}")
            sentence_units.extend(sentences[1:])
        elif heading:
            sentence_units.append(heading)
        else:
            sentence_units.extend(sentences or [body])
    return sentence_units


def _precompute_intent_embeddings() -> None:
    intent_examples = {
        "list_tasks": [
            "show all tasks",
            "list all tasks",
            "display all tasks",
            "show tasks",
            "what tasks do I have",
        ],
        "sort_tasks_by_time": [
            "prioritize tasks by time",
            "sort tasks by urgency",
            "show tasks due soon",
            "what task comes next",
            "rank tasks by current time",
        ],
        "search_tasks": [
            "find tasks related to shopping",
            "search tasks about training",
            "look for my work task",
            "what task is related to lunch",
            "find the task about market",
        ],
        "get_task_stats": [
            "how many tasks are completed",
            "how many tasks are pending",
            "task counts",
            "show completed and pending tasks",
        ],
        "list_completed_tasks": [
            "which tasks are completed",
            "show completed tasks",
            "list done tasks",
            "what tasks have been completed",
            "completed task list",
        ],
        "list_pending_tasks": [
            "which tasks are pending",
            "show pending tasks",
            "open tasks",
            "unfinished tasks",
            "what tasks are not done",
        ],
        "get_task_details": [
            "tell me about my training task",
            "show task details for this task",
            "what is task 7",
            "task info",
        ],
        "complete_task": [
            "complete my eating lunch task",
            "mark this task done",
            "finish the work task",
            "complete the shopping task",
        ],
        "reopen_task": [
            "reopen my work related task",
            "mark this task pending",
            "undo complete on the task",
            "reopen the completed task",
        ],
        "delete_task": [
            "delete shopping task",
            "remove the market task",
            "erase this task",
            "trash the training task",
        ],
        "update_task": [
            "update task title",
            "rename task",
            "edit the task description",
            "change task title to sprint retro planning",
        ],
        "create_task": [
            "create a new task",
            "add a task",
            "make a new task",
            "create task title and description",
        ],
    }

    for intent_name, examples in intent_examples.items():
        INTENT_EXAMPLE_EMBEDDINGS_CACHE[intent_name] = generate_embeddings_batch(examples)


def _semantic_intent(message: str) -> tuple[str, float]:
    if message is None:
        return "", 0.0

    msg_lower = message.lower()

    if looks_like_explicit_create_task_request(message):
        return "create_task", 0.95

    status_action = looks_like_task_status_update_request(message)
    if status_action:
        return status_action, 0.95

    is_task_list_request = bool(re.search(r"\b(?:list|show|display|find|get)\b.*\b(?:task|tasks)\b", msg_lower))
    is_pending_request = bool(re.search(r"\b(?:pending|open|unfinished|not done|incomplete)\b", msg_lower))
    if is_task_list_request and is_pending_request:
        if re.search(r"\b(?:prioritize|prioritise|priority|rank|sort|order|urgency|deadline|due)\b", msg_lower):
            return "sort_tasks_by_time", 0.95
        return "list_pending_tasks", 0.95

    keyword_intents = {
        "list_tasks": ["list", "show all", "display", "all tasks", "what tasks"],
        "sort_tasks_by_time": ["urgent", "priority", "due soon", "due date", "deadline", "next task", "rank"],
        "get_task_stats": ["how many", "completed", "pending", "count", "statistics", "summary"],
        "list_completed_tasks": ["completed", "done", "finished", "complete"],
        "list_pending_tasks": ["pending", "open", "unfinished", "not done", "incomplete"],
        "delete_task": ["delete", "remove", "erase", "trash"],
        "complete_task": ["complete", "done", "finish", "mark done"],
        "reopen_task": ["reopen", "undo complete", "mark pending", "reactivate"],
        "update_task": ["update", "edit", "rename", "change", "modify"],
    }
    for intent, keywords in keyword_intents.items():
        if any(keyword in msg_lower for keyword in keywords):
            return intent, 0.9

    message_embedding = generate_embedding(message)
    best_intent = ""
    best_score = 0.0
    for intent_name, cached_embeddings in INTENT_EXAMPLE_EMBEDDINGS_CACHE.items():
        for example_embedding in cached_embeddings:
            example_score = cosine_similarity(message_embedding, example_embedding)
            if example_score > best_score:
                best_score = example_score
                best_intent = intent_name

    return best_intent, best_score


def extract_task_id(text: str) -> Optional[int]:
    if not text:
        return None
    match = re.search(r"\btask\s*(?:with\s+)?(?:id\s*)?#?(\d+)\b", text, flags=re.IGNORECASE)
    if match:
        return int(match.group(1))
    match = re.search(r"\bid\s*#?(\d+)\b", text, flags=re.IGNORECASE)
    if match:
        return int(match.group(1))
    match = re.search(r"\b(\d+)\b", text)
    return int(match.group(1)) if match else None


def _normalize_task_target_query(text: str) -> str:
    normalized = re.sub(r"\s+", " ", (text or "").strip())
    if not normalized:
        return ""
    normalized = re.sub(
        r"\b(?:please\s+)?(?:complete|reopen|re-open|open again|uncomplete|undo complete|mark pending|mark not done|delete|remove|erase|trash|discard)\b",
        "",
        normalized,
        flags=re.IGNORECASE,
    )
    normalized = re.sub(
        r"\b(?:task|tasks|task\s+id|task\s+with\s+id|id)\b",
        "",
        normalized,
        flags=re.IGNORECASE,
    )
    normalized = re.sub(
        r"\b(?:when|due|deadline|by when|due by|complete by)\b",
        "",
        normalized,
        flags=re.IGNORECASE,
    )
    normalized = re.sub(
        r"\b(?:related to|about|regarding|associated with|matching|called|named|for)\b",
        "",
        normalized,
        flags=re.IGNORECASE,
    )
    normalized = re.sub(r"\b(?:the|a|an|this|that|my|with|as|of)\b", "", normalized, flags=re.IGNORECASE)
    normalized = re.sub(r"\s+", " ", normalized).strip(" ,.;:-\"'")
    return normalized


def _is_deadline_question(text: str) -> bool:
    if not text:
        return False
    lower_text = text.lower()
    return (
        any(word in lower_text for word in ["when", "due", "deadline", "by when", "complete by"])
        and any(word in lower_text for word in ["task", "review", "edit", "work", "project", "job"])
    )


def _format_deadline_response(task_details: dict) -> str:
    deadline_display = task_details.get("deadline_display")
    if deadline_display:
        return f"The task is due on {deadline_display}."
    deadline_at = task_details.get("deadline_at")
    if deadline_at:
        return f"The task is due on {deadline_at}."
    return "I found the task, but I could not find a due date in its description."


def _execute_tool(db: Session, tool_name: str, tool_args: dict) -> tuple[str, Any]:
    try:
        tool_result_text = execute_tool(tool_name, tool_args, db)
        try:
            tool_result = json.loads(tool_result_text)
        except json.JSONDecodeError:
            tool_result = tool_result_text
        return tool_result_text, tool_result
    except Exception as exc:
        error_text = str(exc)
        return error_text, {"error": error_text}


def _tool_call(db: Session, tool_name: str, tool_args: dict, tool_calls: List[dict]) -> str:
    tool_result_text, tool_result = _execute_tool(db, tool_name, tool_args)
    tool_calls.append({"name": tool_name, "args": tool_args, "result": tool_result})
    return f"Tool: {tool_name}\nResult: {tool_result_text}"


def _resolve_task_id_for_action(
    message: str,
    db: Session,
    candidate_docs: Optional[List[dict]] = None,
    desired_status: Optional[str] = None,
) -> Optional[int]:
    task_id = extract_task_id(message)
    if task_id is not None:
        return task_id
    return resolve_task_id_from_query(message, db=db, desired_status=desired_status, candidate_docs=candidate_docs)


def _tool_from_semantic_intent(
    intent: str,
    message: str,
    db: Session,
    candidate_docs: Optional[List[dict]] = None,
) -> tuple[Optional[str], Optional[dict]]:
    explicit_date_constraint = _detect_explicit_date_constraint(message)
    if explicit_date_constraint and intent in {"list_tasks", "list_completed_tasks", "list_pending_tasks", "sort_tasks_by_time"}:
        return "search_tasks", {"query": message}
    if intent == "list_tasks":
        return "list_tasks", {"completed": None, "limit": 100, "offset": 0}
    if intent == "sort_tasks_by_time":
        return "sort_tasks_by_time", {"completed": False, "limit": 100, "offset": 0}
    if intent == "search_tasks":
        return "search_tasks", {"query": message}
    if intent == "get_task_stats":
        return "get_task_stats", {}
    if intent == "list_completed_tasks":
        return "list_tasks", {"completed": True, "limit": 100, "offset": 0}
    if intent == "list_pending_tasks":
        return "list_tasks", {"completed": False, "limit": 100, "offset": 0}
    if intent == "get_task_details":
        task_id = _resolve_task_id_for_action(message, db, candidate_docs)
        return ("get_task_details", {"task_id": task_id}) if task_id is not None else (None, None)
    if intent == "complete_task":
        task_id = _resolve_task_id_for_action(message, db, candidate_docs, desired_status="pending")
        return ("complete_task", {"task_id": task_id}) if task_id is not None else (None, None)
    if intent == "reopen_task":
        task_id = _resolve_task_id_for_action(message, db, candidate_docs, desired_status="completed")
        return ("reopen_task", {"task_id": task_id}) if task_id is not None else (None, None)
    if intent == "delete_task":
        task_id = _resolve_task_id_for_action(message, db, candidate_docs)
        return ("delete_task", {"task_id": task_id}) if task_id is not None else (None, None)
    if intent == "create_task":
        title, description = extract_create_task_fields(message)
        return "create_task", {"title": title, "description": description}
    if intent == "update_task":
        task_id, title, description = extract_update_task_fields(message)
        if task_id is not None and (title is not None or description is not None):
            return "update_task", {"task_id": task_id, "title": title, "description": description}
        return None, None
    return None, None


def _infer_tool_from_keywords(
    message_lower: str,
    message: str,
    db: Session,
    candidate_docs: Optional[List[dict]] = None,
) -> tuple[Optional[str], Optional[dict]]:
    if any(word in message_lower for word in ["show all tasks", "list all tasks", "display all tasks", "show tasks", "list tasks", "display tasks", "all tasks"]):
        return "list_tasks", {"completed": None, "limit": 100, "offset": 0}

    if any(word in message_lower for word in ["sort", "order", "rank", "prioritize", "prioritise", "priority", "importance", "urgency", "soon", "asap", "earliest", "later", "by time", "time-based", "deadline", "schedule"]):
        return "sort_tasks_by_time", {"completed": False, "limit": 100, "offset": 0}

    if any(word in message_lower for word in ["related to", "about", "regarding", "associated with", "my training", "training"]):
        return "search_tasks", {"query": message}

    if any(word in message_lower for word in ["how many completed", "how many pending", "completed and pending", "completed tasks", "pending tasks", "task counts", "task count", "count of tasks", "how many tasks", "stats", "summary", "count"]):
        return "get_task_stats", {}

    if _detect_explicit_date_constraint(message):
        return "search_tasks", {"query": message}

    status_action = looks_like_task_status_update_request(message)
    if status_action == "complete_task":
        task_id = _resolve_task_id_for_action(message, db, candidate_docs, desired_status="pending")
        return ("complete_task", {"task_id": task_id}) if task_id is not None else (None, None)
    if status_action == "reopen_task":
        task_id = _resolve_task_id_for_action(message, db, candidate_docs, desired_status="completed")
        return ("reopen_task", {"task_id": task_id}) if task_id is not None else (None, None)

    if any(word in message_lower for word in ["update", "edit", "rename", "change", "modify"]) and "task" in message_lower:
        task_id, title, description = extract_update_task_fields(message)
        if task_id is not None and (title is not None or description is not None):
            return "update_task", {"task_id": task_id, "title": title, "description": description}

    if any(word in message_lower for word in ["details", "detail", "tell me about", "what is", "task info", "task information"]):
        task_id = _resolve_task_id_for_action(message, db, candidate_docs)
        if task_id is not None:
            return "get_task_details", {"task_id": task_id}
        return "search_tasks", {"query": message}

    status_action = looks_like_task_status_update_request(message)
    if status_action == "complete_task":
        task_id = _resolve_task_id_for_action(message, db, candidate_docs, desired_status="pending")
        return ("complete_task", {"task_id": task_id}) if task_id is not None else (None, None)
    if status_action == "reopen_task":
        task_id = _resolve_task_id_for_action(message, db, candidate_docs, desired_status="completed")
        return ("reopen_task", {"task_id": task_id}) if task_id is not None else (None, None)

    if looks_like_explicit_create_task_request(message):
        title, description = extract_create_task_fields(message)
        return "create_task", {"title": title, "description": description}

    return None, None


def resolve_task_id_from_query(
    text: str,
    db: Session,
    desired_status: Optional[str] = None,
    candidate_docs: Optional[List[dict]] = None,
) -> Optional[int]:
    query_text = _normalize_task_target_query(text)
    if not query_text:
        return None

    candidates: List[dict] = []
    if candidate_docs:
        for doc in candidate_docs:
            task_id = doc.get("task_id") or doc.get("id")
            if task_id is None:
                continue
            candidates.append(
                {
                    "id": task_id,
                    "title": doc.get("title", ""),
                    "similarity_score": doc.get("similarity_score", 0.0),
                }
            )

    if not candidates:
        tool_result = execute_tool("search_tasks", {"query": query_text}, db)
        try:
            candidates = json.loads(tool_result)
        except json.JSONDecodeError:
            return None

    if not isinstance(candidates, list) or not candidates:
        return None

    if desired_status:
        filtered = []
        for item in candidates:
            task_id = item.get("id")
            if task_id is None:
                continue
            task = db.query(TaskModel).filter(TaskModel.id == task_id).first()
            if not task:
                continue
            task_status = "completed" if task.completed else "pending"
            if task_status == desired_status:
                filtered.append(item)
        if filtered:
            candidates = filtered

    lowered_query = query_text.lower().strip()
    for candidate in candidates:
        candidate_title = str(candidate.get("title", "")).lower().strip()
        if candidate_title == lowered_query or lowered_query in candidate_title:
            return candidate.get("id")

    candidates.sort(key=lambda item: float(item.get("similarity_score", 0.0)), reverse=True)
    return candidates[0].get("id")


def vector_search(db: Session, query_embedding: List[float], limit: int = 5) -> List[dict]:
    try:
        embedding_literal = format_pgvector_literal(query_embedding)
        results = db.execute(
            text(
                """
                  SELECT d.id, d.task_id, d.title, d.content, d.chunk_index, d.chunk_count,
                      d.project, d.document_type, d.owner_user_id,
                       1 - (embedding <=> CAST(:embedding AS vector(384))) as similarity_score
                FROM documents d
                INNER JOIN tasks t ON t.id = d.task_id
                ORDER BY d.embedding <=> CAST(:embedding AS vector(384))
                LIMIT :limit
                """
            ),
            {"embedding": embedding_literal, "limit": limit},
        ).fetchall()

        return [
            {
                "id": r[0],
                "task_id": r[1],
                "title": r[2],
                "content": r[3],
                "chunk_index": r[4],
                "chunk_count": r[5],
                "project": r[6],
                "document_type": r[7],
                "owner_user_id": r[8],
                "similarity_score": float(r[9]),
            }
            for r in results
        ]
    except Exception:
        return []


def _bm25_tokens(value: str) -> List[str]:
    """Tokenize searchable text for the local BM25 candidate stage."""
    return re.findall(r"[a-z0-9_]+", (value or "").lower())


def _matches_filters(document: Any, filters: Optional[Dict[str, Any]]) -> bool:
    if not filters:
        return True
    for field in ("task_id", "project", "document_type", "owner_user_id"):
        expected = filters.get(field)
        if expected is not None and getattr(document, field, None) != expected:
            return False
    return True


def bm25_search(db: Session, query: str, limit: int = 20, filters: Optional[Dict[str, Any]] = None) -> List[dict]:
    """Rank stored documents by BM25 lexical relevance."""
    query_tokens = _bm25_tokens(query)
    if not query_tokens:
        return []

    documents = [document for document in db.query(DocumentModel).all() if _matches_filters(document, filters)]
    tokenized_documents = []
    document_frequency: Dict[str, int] = {}
    for document in documents:
        # Repeat the title so exact title matches receive a modest boost.
        tokens = _bm25_tokens(f"{document.title} {document.title} {document.content}")
        tokenized_documents.append((document, tokens))
        for token in set(tokens):
            document_frequency[token] = document_frequency.get(token, 0) + 1

    if not tokenized_documents:
        return []

    average_length = sum(len(tokens) for _, tokens in tokenized_documents) / len(tokenized_documents)
    total_documents = len(tokenized_documents)
    scores = []
    for document, tokens in tokenized_documents:
        term_frequency = {}
        for token in tokens:
            term_frequency[token] = term_frequency.get(token, 0) + 1
        document_length = len(tokens) or 1
        score = 0.0
        for token in query_tokens:
            frequency = term_frequency.get(token, 0)
            if not frequency:
                continue
            document_count = document_frequency.get(token, 0)
            inverse_document_frequency = math.log(
                1 + (total_documents - document_count + 0.5) / (document_count + 0.5)
            )
            denominator = frequency + 1.5 * (
                1 - 0.75 + 0.75 * document_length / max(average_length, 1.0)
            )
            score += inverse_document_frequency * (frequency * 2.5 / denominator)
        if score > 0:
            scores.append((score, document))

    scores.sort(key=lambda item: item[0], reverse=True)
    return [
        {
            "id": document.id,
            "task_id": document.task_id,
            "title": document.title,
            "content": document.content,
            "chunk_index": getattr(document, "chunk_index", 0),
            "chunk_count": getattr(document, "chunk_count", 1),
            "citation_id": f"doc-{document.id}-chunk-{getattr(document, 'chunk_index', 0)}",
            "bm25_score": float(score),
        }
        for score, document in scores[:limit]
    ]


def _hyde_query(query: str) -> str:
    """Create a hypothetical answer for semantic retrieval, with a safe fallback."""
    if os.getenv("ENABLE_HYDE_RETRIEVAL", "true").lower() not in {"1", "true", "yes", "on"}:
        return query

    cached = _HYDE_CACHE.get(query)
    if cached and time.time() - cached[0] < _CACHE_TTL_SECONDS:
        return cached[1]
    try:
        from backend_fastapi.mistral_client import generate_response

        hypothetical = generate_response(
            "Write a short hypothetical technical answer that could be found in the knowledge base. "
            "Do not mention that it is hypothetical. Question: " + query,
            temperature=0.2,
        )
        hypothetical = re.sub(r"\s+", " ", str(hypothetical or "")).strip()
        result = hypothetical[:4000] or query
        _HYDE_CACHE[query] = (time.time(), result)
        return result
    except Exception:
        return query


def reciprocal_rank_fusion(result_lists: List[List[dict]], limit: int = 5, k: int = 60) -> List[dict]:
    """Fuse ranked result lists using standard reciprocal rank fusion."""
    fused: Dict[Any, dict] = {}
    for result_list in result_lists:
        for rank, result in enumerate(result_list, start=1):
            result_id = result.get("id")
            if result_id is None:
                continue
            if result_id not in fused:
                fused[result_id] = dict(result)
                fused[result_id]["rrf_score"] = 0.0
            fused[result_id]["rrf_score"] += 1.0 / (k + rank)
            for key in ("bm25_score", "similarity_score", "dense_score"):
                if key in result and key not in fused[result_id]:
                    fused[result_id][key] = result[key]

    ranked = sorted(fused.values(), key=lambda item: item["rrf_score"], reverse=True)[:limit]
    maximum = max((item["rrf_score"] for item in ranked), default=1.0)
    for item in ranked:
        item["rrf_score"] = float(item["rrf_score"] / maximum) if maximum else 0.0
        item["retrieval_score"] = item["rrf_score"]
        item.setdefault("similarity_score", float(item.get("dense_score", 0.0) or 0.0))
    return ranked


def cross_encoder_rerank(query: str, results: List[dict], limit: int) -> List[dict]:
    """Optionally rerank fused results through Hugging Face or a local CrossEncoder."""
    if os.getenv("ENABLE_CROSS_ENCODER_RERANKING", "false").lower() not in {"1", "true", "yes", "on"}:
        return results[:limit]

    pairs = [(query, f"{item.get('title', '')}\n{item.get('content', '')}") for item in results]
    api_url = os.getenv("CROSS_ENCODER_API_URL") or os.getenv("HF_CROSS_ENCODER_API_URL")
    api_token = (
        os.getenv("CROSS_ENCODER_API_TOKEN")
        or os.getenv("HF_CROSS_ENCODER_API_TOKEN")
        or os.getenv("HF_API_TOKEN")
        or HF_EMBEDDING_API_TOKEN
        or os.getenv("HF_EMBEDDING_API_TOKEN")
        or os.getenv("HUGGINGFACEHUB_API_TOKEN")
    )
    model_name = os.getenv(
        "CROSS_ENCODER_MODEL",
        "cross-encoder/ms-marco-MiniLM-L-6-v2",
    )
    if not api_url and api_token:
        api_base = os.getenv(
            "HF_INFERENCE_API_URL",
            "https://router.huggingface.co/hf-inference/models",
        ).rstrip("/")
        api_url = f"{api_base}/{model_name}"
    if api_url and api_token:
        try:
            import requests

            response = requests.post(
                api_url,
                headers={
                    "Authorization": f"Bearer {api_token}",
                    "Content-Type": "application/json",
                },
                json={"inputs": pairs, "parameters": {"top_k": 1}},
                timeout=float(os.getenv("CROSS_ENCODER_API_TIMEOUT", "20")),
            )
            response.raise_for_status()
            payload = response.json()
            scores = _parse_huggingface_reranker_scores(payload, len(results))
            if isinstance(scores, list) and len(scores) == len(results):
                reranked = []
                for item, score in zip(results, scores):
                    enriched = dict(item)
                    enriched["cross_encoder_score"] = float(score)
                    enriched["reranker"] = "api"
                    reranked.append(enriched)
                return sorted(reranked, key=lambda item: item["cross_encoder_score"], reverse=True)[:limit]
        except Exception:
            logger = __import__("logging").getLogger(__name__)
            logger.warning("Cross-encoder API failed; falling back to local reranking", exc_info=True)
    try:
        from sentence_transformers import CrossEncoder

        model = CrossEncoder(model_name)
        scores = model.predict(pairs)
        reranked = []
        for item, score in zip(results, scores):
            enriched = dict(item)
            enriched["cross_encoder_score"] = float(score)
            enriched["reranker"] = "local"
            reranked.append(enriched)
        return sorted(reranked, key=lambda item: item["cross_encoder_score"], reverse=True)[:limit]
    except Exception:
        return results[:limit]


def _parse_huggingface_reranker_scores(payload: Any, expected_count: int) -> Optional[List[float]]:
    """Normalize common Hugging Face text-classification response shapes."""
    if isinstance(payload, dict) and isinstance(payload.get("scores"), list):
        return [float(score) for score in payload["scores"]]
    if not isinstance(payload, list) or len(payload) != expected_count:
        return None

    scores = []
    for item in payload:
        candidates = item if isinstance(item, list) else [item]
        if not candidates or not all(isinstance(candidate, dict) for candidate in candidates):
            return None
        positive = [
            candidate for candidate in candidates
            if str(candidate.get("label", "")).lower() in {"1", "positive", "relevant", "entailment"}
        ]
        selected = positive[0] if positive else max(candidates, key=lambda candidate: float(candidate.get("score", 0.0)))
        scores.append(float(selected.get("score", 0.0)))
    return scores


def _hybrid_search_once(db: Session, query: str, limit: int = 5, use_hyde: bool = True, filters: Optional[Dict[str, Any]] = None) -> List[dict]:
    """Run one BM25 and pgvector retrieval, then combine them with RRF."""
    lexical_results = bm25_search(db, query, limit=max(limit * 4, 20), filters=filters)
    semantic_query = _hyde_query(query) if use_hyde else query
    embedding = _EMBEDDING_CACHE.get(semantic_query)
    if embedding and time.time() - embedding[0] < _CACHE_TTL_SECONDS:
        query_embedding = embedding[1]
    else:
        query_embedding = generate_embedding(semantic_query)
        _EMBEDDING_CACHE[semantic_query] = (time.time(), query_embedding)
    dense_results = vector_search(db, query_embedding, limit=max(limit * 4, 20))
    if filters:
        dense_results = [item for item in dense_results if all(
            expected is None or item.get(field) == expected
            for field, expected in filters.items()
            if field in {"task_id", "project", "document_type", "owner_user_id"}
        )]
    for result in dense_results:
        result["dense_score"] = result.get("similarity_score", 0.0)
    fused = reciprocal_rank_fusion([lexical_results, dense_results], limit=max(limit * 2, 10))
    return cross_encoder_rerank(query, fused, limit)


def decompose_query(query: str, max_queries: int = 3) -> List[str]:
    """Split a compound request into focused retrieval queries."""
    normalized = re.sub(r"\s+", " ", (query or "").strip())
    if not normalized:
        return []

    parts = re.split(
        r"\s+(?:and|also|plus|then|as well as)\s+|[;?]\s*",
        normalized,
        flags=re.IGNORECASE,
    )
    queries = []
    for part in parts:
        part = part.strip(" .,:")
        if len(_bm25_tokens(part)) >= 2 and part.lower() not in {item.lower() for item in queries}:
            queries.append(part)

    if len(queries) < 2:
        return [normalized]
    return queries[:max_queries]


def plan_query(query: str, max_queries: int = 3) -> Dict[str, Any]:
    """Create a validated retrieval plan, falling back to rule-based splitting."""
    fallback_queries = decompose_query(query, max_queries=max_queries)
    plan = {
        "subqueries": fallback_queries,
        "use_hyde": True,
        "needs_analysis": len(fallback_queries) > 1,
        "planner": "rules",
    }
    if len(fallback_queries) < 2:
        return plan
    if os.getenv("ENABLE_QUERY_PLANNER", "true").lower() not in {"1", "true", "yes", "on"}:
        return plan

    try:
        from backend_fastapi.mistral_client import generate_response

        response = generate_response(
            "Return JSON only with keys subqueries, use_hyde, and needs_analysis. "
            f"Create up to {max_queries} focused retrieval queries for: {query}",
            temperature=0.1,
        )
        match = re.search(r"\{[\s\S]*\}", str(response or ""))
        payload = json.loads(match.group(0)) if match else {}
        subqueries = payload.get("subqueries")
        if isinstance(subqueries, list):
            cleaned = []
            for item in subqueries:
                item = re.sub(r"\s+", " ", str(item or "")).strip()
                if len(_bm25_tokens(item)) >= 2 and item.lower() not in {q.lower() for q in cleaned}:
                    cleaned.append(item)
            if cleaned:
                plan["subqueries"] = cleaned[:max_queries]
                plan["use_hyde"] = bool(payload.get("use_hyde", True))
                plan["needs_analysis"] = bool(payload.get("needs_analysis", len(cleaned) > 1))
                plan["planner"] = "llm"
    except Exception:
        pass
    return plan


def multi_query_hybrid_search(
    db: Session,
    query: str,
    limit: int = 5,
    use_hyde: bool = True,
    subqueries: Optional[List[str]] = None,
    filters: Optional[Dict[str, Any]] = None,
) -> List[dict]:
    """Retrieve each decomposed query and fuse all candidates with RRF."""
    if subqueries is None:
        query_plan = plan_query(query)
        subqueries = query_plan["subqueries"]
        use_hyde = query_plan["use_hyde"]
    result_lists = [
        _hybrid_search_once(db, subquery, limit=max(limit * 2, 10), use_hyde=use_hyde, filters=filters)
        for subquery in subqueries
    ]
    return reciprocal_rank_fusion(result_lists, limit=limit)


def hybrid_search(
    db: Session,
    query: str,
    limit: int = 5,
    use_hyde: bool = True,
    decompose: bool = True,
    filters: Optional[Dict[str, Any]] = None,
) -> List[dict]:
    """Run hybrid retrieval, decomposing compound questions when useful."""
    if decompose and len(decompose_query(query)) > 1:
        query_plan = plan_query(query)
        return multi_query_hybrid_search(
            db,
            query,
            limit=limit,
            use_hyde=use_hyde and query_plan["use_hyde"],
            subqueries=query_plan["subqueries"],
            filters=filters,
        )
    return _hybrid_search_once(db, query, limit=limit, use_hyde=use_hyde, filters=filters)


def _search_relevance_boost(query: str, title: str, content: str) -> float:
    query_text = (query or "").strip().lower()
    if not query_text:
        return 0.0

    doc_text = f"{title or ''}\n{content or ''}".lower()
    query_terms = set(re.findall(r"[a-z0-9]+", query_text))
    doc_terms = set(re.findall(r"[a-z0-9]+", doc_text))
    if not query_terms or not doc_terms:
        return 0.0

    lexical_overlap = len(query_terms & doc_terms) / max(len(query_terms), 1)
    temporal_terms = {
        "today",
        "tomorrow",
        "tonight",
        "deadline",
        "due",
        "soon",
        "asap",
        "later",
        "schedule",
        "time",
        "urgent",
        "urgency",
    }
    temporal_query = bool(query_terms & temporal_terms)
    temporal_doc = bool(doc_terms & temporal_terms)

    boost = 0.65 * lexical_overlap
    if temporal_query and temporal_doc:
        boost += 0.15
    return min(1.0, boost)


def _token_similarity(query_terms: set[str], doc_terms: set[str]) -> float:
    if not query_terms or not doc_terms:
        return 0.0

    exact_overlap = len(query_terms & doc_terms) / max(len(query_terms), 1)
    fuzzy_total = 0.0
    for query_term in query_terms:
        best_ratio = 0.0
        for doc_term in doc_terms:
            ratio = SequenceMatcher(None, query_term, doc_term).ratio()
            if ratio > best_ratio:
                best_ratio = ratio
        fuzzy_total += best_ratio

    fuzzy_overlap = fuzzy_total / max(len(query_terms), 1)
    prefix_bonus = 0.0
    if any(
        query_term.startswith(doc_term) or doc_term.startswith(query_term)
        for query_term in query_terms
        for doc_term in doc_terms
    ):
        prefix_bonus = 0.15

    return min(1.0, (0.30 * exact_overlap) + (0.55 * fuzzy_overlap) + prefix_bonus)


def _rank_document_search_result(query: str, query_embedding: List[float], document) -> float:
    doc_embedding_values = document.embedding if document.embedding is not None else []
    doc_embedding = list(doc_embedding_values)
    semantic_score = cosine_similarity(query_embedding, doc_embedding) if doc_embedding else 0.0

    query_text = (query or "").strip().lower()
    doc_text = f"{document.title or ''}\n{document.content or ''}".lower()
    query_terms = set(re.findall(r"[a-z0-9]+", query_text))
    doc_terms = set(re.findall(r"[a-z0-9]+", doc_text))
    text_score = _token_similarity(query_terms, doc_terms)

    temporal_terms = {
        "today",
        "tomorrow",
        "tonight",
        "deadline",
        "due",
        "soon",
        "asap",
        "later",
        "schedule",
        "time",
        "urgent",
        "urgency",
    }
    if query_terms & temporal_terms and doc_terms & temporal_terms:
        text_score = min(1.0, text_score + 0.12)

    return min(1.0, (0.58 * semantic_score) + (0.42 * text_score))
