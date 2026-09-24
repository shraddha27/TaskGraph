import glob
import json
import math
import os
import re
from collections import Counter
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional

import joblib
import numpy as np
from scipy import sparse
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import HashingVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.pipeline import Pipeline

DEFAULT_TASK_MODEL_PATH = os.path.join(
    os.path.dirname(__file__), "task_completion_model.joblib"
)

_FALLBACK_TRAINING_TASKS = [
    {
        "title": "Submit weekly status update",
        "description": "Share progress and confirm blockers for the team.",
        "completed": True,
        "created_at": datetime.utcnow(),
        "priority": "low",
    },
    {
        "title": "Resolve deployment blocker",
        "description": "Investigate the env issue and unblock release activities.",
        "completed": False,
        "created_at": datetime.utcnow(),
        "priority": "high",
    },
    {
        "title": "Review sprint tasks",
        "description": "Prioritize work and check capacity for the next iteration.",
        "completed": True,
        "created_at": datetime.utcnow(),
        "priority": "medium",
    },
    {
        "title": "Follow up on support ticket",
        "description": "Confirm customer response and close the incident.",
        "completed": False,
        "created_at": datetime.utcnow(),
        "priority": "medium",
    },
]


class TaskTextAgeFeatures(BaseEstimator, TransformerMixin):
    def __init__(self, text_vectorizer: Optional[TfidfVectorizer] = None):
        self.text_vectorizer = text_vectorizer or TfidfVectorizer(
            lowercase=True,
            token_pattern=r"[a-z0-9]+",
            ngram_range=(1, 2),
        )

    def fit(self, X, y=None):
        texts = [row["text"] if isinstance(row, dict) else row[0] for row in X]
        self.text_vectorizer.fit(texts)
        return self

    def transform(self, X):
        texts = [row["text"] if isinstance(row, dict) else row[0] for row in X]
        ages = [float(row["age"] if isinstance(row, dict) else row[1]) for row in X]
        text_matrix = self.text_vectorizer.transform(texts)
        age_matrix = np.asarray(ages, dtype=np.float32).reshape(-1, 1)
        return np.hstack([text_matrix.toarray(), age_matrix])


class TaskScikitPredictionModel:
    def __init__(self, vectorizer: HashingVectorizer, classifier: SGDClassifier):
        self.vectorizer = vectorizer
        self.classifier = classifier

    def _build_features(self, rows: Iterable[Dict[str, Any]]):
        rows = list(rows)
        if not rows:
            return sparse.csr_matrix((0, self.vectorizer.n_features + 1))

        texts = [row["text"] if isinstance(row, dict) else str(row[0]) for row in rows]
        ages = np.asarray([float(row["age"] if isinstance(row, dict) else float(row[1])) for row in rows], dtype=np.float32).reshape(-1, 1)
        text_matrix = self.vectorizer.transform(texts)
        age_matrix = sparse.csr_matrix(ages)
        return sparse.hstack([text_matrix, age_matrix], format="csr")

    def predict_proba(self, rows: Iterable[Dict[str, Any]]):
        features = self._build_features(rows)
        return self.classifier.predict_proba(features)

    def predict(self, rows: Iterable[Dict[str, Any]]):
        probabilities = self.predict_proba(rows)
        return np.argmax(probabilities, axis=1)


def _tokenize_text(value: str) -> List[str]:
    if not value:
        return []
    return re.findall(r"[a-z0-9]+", value.lower())


def _task_text(task: Dict[str, Any]) -> str:
    return " ".join(
        filter(
            None,
            [
                task.get("title") or "",
                task.get("description") or "",
                task.get("priority") or "",
            ],
        )
    )


def _task_time_feature(task: Dict[str, Any]) -> float:
    created_at = task.get("created_at")
    if not created_at:
        return 0.0
    if isinstance(created_at, str):
        created_at = datetime.fromisoformat(created_at)
    now = datetime.utcnow()
    delta_days = (now - created_at).total_seconds() / 86400.0
    return max(0.0, min(delta_days, 365.0)) / 365.0


def _task_feature_row(task: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "text": _task_text(task),
        "age": _task_time_feature(task),
        "completed": bool(task.get("completed")),
    }


def _build_vocabulary(tasks: Iterable[Dict[str, Any]]) -> Dict[str, int]:
    vocab: Dict[str, int] = {}
    for task in tasks:
        for token in _tokenize_text(_task_text(task)):
            if token not in vocab:
                vocab[token] = len(vocab)
    return vocab


def _compute_idf(tasks: Iterable[Dict[str, Any]], vocab: Dict[str, int]) -> Dict[str, float]:
    task_list = list(tasks)
    if not task_list:
        return {}

    doc_frequency = {token: 0 for token in vocab}
    for task in task_list:
        seen = set()
        for token in _tokenize_text(_task_text(task)):
            if token in vocab and token not in seen:
                seen.add(token)
                doc_frequency[token] += 1

    num_docs = len(task_list)
    return {
        token: math.log((1 + num_docs) / (1 + doc_frequency.get(token, 0))) + 1.0
        for token in vocab
    }


def _tfidf_vector(task: Dict[str, Any], vocab: Dict[str, int], idf: Dict[str, float]) -> List[float]:
    tokens = _tokenize_text(_task_text(task))
    counts = Counter(tokens)
    total = sum(counts.values()) or 1
    vector = [0.0] * len(vocab)
    for token, count in counts.items():
        idx = vocab.get(token)
        if idx is None:
            continue
        tf = count / total
        vector[idx] = tf * idf.get(token, 1.0)
    vector.append(_task_time_feature(task))
    return vector


def _prepare_training_data(tasks: Iterable[Dict[str, Any]]) -> tuple[List[Dict[str, Any]], List[float], Dict[str, Any]]:
    task_list = list(tasks)
    if not task_list:
        raise ValueError("At least one task record is required to train the model.")

    rows = [_task_feature_row(task) for task in task_list]
    labels = [float(1 if task.get("completed") else 0) for task in task_list]
    vocab = _build_vocabulary(task_list)
    idf = _compute_idf(task_list, vocab)
    return rows, labels, {"vocab": vocab, "idf": idf}


def _build_sklearn_pipeline() -> TaskScikitPredictionModel:
    vectorizer = HashingVectorizer(
        lowercase=True,
        token_pattern=r"[a-z0-9]+",
        ngram_range=(1, 2),
        alternate_sign=False,
        n_features=2**18,
    )
    classifier = SGDClassifier(
        loss="log_loss",
        penalty="l2",
        alpha=1e-4,
        max_iter=2000,
        tol=1e-3,
        random_state=42,
    )
    return TaskScikitPredictionModel(vectorizer=vectorizer, classifier=classifier)


def _iter_task_batches(tasks: Iterable[Dict[str, Any]], batch_size: int = 500) -> Iterable[List[Dict[str, Any]]]:
    chunk: List[Dict[str, Any]] = []
    for task in tasks:
        chunk.append(task)
        if len(chunk) >= batch_size:
            yield chunk
            chunk = []
    if chunk:
        yield chunk


def train_task_completion_model(
    tasks: Optional[Iterable[Dict[str, Any]]] = None,
    *,
    epochs: int = 40,
    learning_rate: float = 0.05,
    save_path: Optional[str] = DEFAULT_TASK_MODEL_PATH,
    batch_size: int = 500,
) -> Dict[str, Any]:
    if tasks is None:
        tasks = []

    task_list = list(tasks)
    if not task_list:
        raise ValueError("No task training data supplied.")

    pipeline = _build_sklearn_pipeline()
    first_batch = True
    total_rows = 0
    for batch in _iter_task_batches(task_list, batch_size=batch_size):
        batch_rows = [{"text": _task_text(task), "age": _task_time_feature(task)} for task in batch]
        labels = np.asarray([1 if task.get("completed") else 0 for task in batch], dtype=np.int32)
        features = pipeline._build_features(batch_rows)
        if first_batch:
            pipeline.classifier.partial_fit(features, labels, classes=np.array([0, 1]))
            first_batch = False
        else:
            pipeline.classifier.partial_fit(features, labels)
        total_rows += len(batch)

    metadata = {
        "vocabulary": {},
        "idf": {},
        "model_type": "sklearn_logistic_regression",
        "vectorizer": "tfidf",
        "trained": True,
        "epochs": epochs,
        "learning_rate": learning_rate,
        "vocabulary_size": pipeline.vectorizer.n_features,
        "created_at": datetime.utcnow().isoformat(),
        "rows_seen": total_rows,
    }
    payload = {
        "pipeline": pipeline,
        "model_state": {
            "linear.weight": np.asarray(pipeline.classifier.coef_, dtype=np.float32).reshape(1, -1),
            "linear.bias": np.asarray(pipeline.classifier.intercept_, dtype=np.float32).reshape(1),
        },
        "metadata": metadata,
    }

    if save_path:
        artifact_dir = os.path.dirname(save_path) or "."
        os.makedirs(artifact_dir, exist_ok=True)
        joblib.dump(payload, save_path)
        save_task_completion_model_artifact(payload, model_dir=artifact_dir)

    return payload


def iter_task_training_rows_from_db(db, batch_size: int = 1000) -> Iterable[List[Dict[str, Any]]]:
    try:
        from backend_fastapi.models import TaskModel
    except Exception:  # pragma: no cover
        TaskModel = None

    if TaskModel is None:
        raise RuntimeError("Task model table is not available for DB-backed training.")

    offset = 0
    while True:
        batch = db.query(TaskModel).order_by(TaskModel.id.asc()).offset(offset).limit(batch_size).all()
        if not batch:
            break
        yield [
            build_task_training_row(
                {
                    "title": task.title,
                    "description": task.description,
                    "completed": bool(task.completed),
                    "created_at": task.created_at,
                    "priority": task.priority,
                }
            )
            for task in batch
        ]
        offset += len(batch)


def train_task_completion_model_from_db(db, batch_size: int = 1000, save_path: Optional[str] = DEFAULT_TASK_MODEL_PATH):
    rows_stream = iter_task_training_rows_from_db(db, batch_size=batch_size)
    first_batch = True
    pipeline = _build_sklearn_pipeline()
    total_rows = 0

    for batch in rows_stream:
        if not batch:
            continue
        labels = np.asarray([1 if item.get("completed") else 0 for item in batch], dtype=np.int32)
        batch_rows = [{"text": _task_text(item), "age": _task_time_feature(item)} for item in batch]
        features = pipeline._build_features(batch_rows)
        if first_batch:
            pipeline.classifier.partial_fit(features, labels, classes=np.array([0, 1]))
            first_batch = False
        else:
            pipeline.classifier.partial_fit(features, labels)
        total_rows += len(batch)

    if total_rows == 0:
        raise ValueError("No task records found in the database for training.")

    payload = {
        "pipeline": pipeline,
        "model_state": {
            "linear.weight": np.asarray(pipeline.classifier.coef_, dtype=np.float32).reshape(1, -1),
            "linear.bias": np.asarray(pipeline.classifier.intercept_, dtype=np.float32).reshape(1),
        },
        "metadata": {
            "vocabulary": {},
            "idf": {},
            "model_type": "sklearn_logistic_regression",
            "vectorizer": "tfidf",
            "trained": True,
            "epochs": 40,
            "learning_rate": 0.05,
            "vocabulary_size": pipeline.vectorizer.n_features,
            "created_at": datetime.utcnow().isoformat(),
            "rows_seen": total_rows,
        },
    }

    if save_path:
        artifact_dir = os.path.dirname(save_path) or "."
        os.makedirs(artifact_dir, exist_ok=True)
        joblib.dump(payload, save_path)
        save_task_completion_model_artifact(payload, model_dir=artifact_dir)

    return payload


def ensure_task_completion_model(model_path: str = DEFAULT_TASK_MODEL_PATH) -> Dict[str, Any]:
    if os.path.exists(model_path):
        return load_task_completion_model(model_path)

    os.makedirs(os.path.dirname(model_path) or ".", exist_ok=True)
    fallback_model = train_task_completion_model(
        tasks=_FALLBACK_TRAINING_TASKS,
        epochs=25,
        save_path=model_path,
    )
    return {"pipeline": fallback_model["pipeline"], "metadata": fallback_model["metadata"]}


def save_task_completion_model_artifact(model: Dict[str, Any], model_dir: Optional[str] = None) -> str:
    artifact_dir = model_dir or os.path.dirname(DEFAULT_TASK_MODEL_PATH)
    os.makedirs(artifact_dir, exist_ok=True)

    created_at = model.get("metadata", {}).get("created_at") or datetime.utcnow().isoformat()
    timestamp = created_at.replace(":", "-").replace(".", "-")
    artifact_path = os.path.join(artifact_dir, f"task_completion_model_{timestamp}.joblib")
    joblib.dump(model, artifact_path)
    return artifact_path


def get_latest_task_completion_model_path(model_dir: Optional[str] = None) -> str:
    artifact_dir = model_dir or os.path.dirname(DEFAULT_TASK_MODEL_PATH)
    if not os.path.exists(artifact_dir):
        return DEFAULT_TASK_MODEL_PATH

    matches = sorted(
        glob.glob(os.path.join(artifact_dir, "task_completion_model_*.joblib")),
        key=os.path.getmtime,
        reverse=True,
    )
    if matches:
        return matches[0]
    return DEFAULT_TASK_MODEL_PATH


def load_task_completion_model(model_path: str = DEFAULT_TASK_MODEL_PATH) -> Dict[str, Any]:
    if not os.path.exists(model_path):
        if model_path == DEFAULT_TASK_MODEL_PATH and os.path.exists(os.path.dirname(model_path) or "."):
            latest_path = get_latest_task_completion_model_path(os.path.dirname(model_path) or ".")
            if latest_path != DEFAULT_TASK_MODEL_PATH and os.path.exists(latest_path):
                model_path = latest_path
            else:
                return ensure_task_completion_model(model_path)
        else:
            return ensure_task_completion_model(model_path)

    try:
        payload = joblib.load(model_path)
    except Exception:
        try:
            payload = joblib.load(model_path.replace(".joblib", ".pkl"))
        except Exception:
            payload = {"pipeline": None, "metadata": {}}

    if isinstance(payload, dict) and "metadata" in payload:
        return payload

    if payload.get("pipeline") is not None:
        return payload

    raise ValueError(f"Unsupported model artifact at {model_path}. Expected a sklearn pipeline artifact.")


def build_task_training_row(task: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "title": task.get("title") or "",
        "description": task.get("description") or "",
        "completed": bool(task.get("completed")),
        "created_at": task.get("created_at") or datetime.utcnow(),
        "priority": task.get("priority") or "medium",
    }


def retrain_task_completion_model(tasks: Iterable[Dict[str, Any]], save_path: str = DEFAULT_TASK_MODEL_PATH) -> Dict[str, Any]:
    return train_task_completion_model(tasks=list(tasks), epochs=40, save_path=save_path)


def predict_task_completion(task: Dict[str, Any], model: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    if model is None:
        model = ensure_task_completion_model(DEFAULT_TASK_MODEL_PATH)

    pipeline = model.get("pipeline")
    if pipeline is None:
        raise ValueError("No trained task completion pipeline is available.")

    feature_row = [{"text": _task_text(task), "age": _task_time_feature(task)}]
    probability = float(pipeline.predict_proba(feature_row)[0, 1])
    probability = max(0.0, min(1.0, probability))
    label = "completed" if probability >= 0.5 else "incomplete"

    return {
        "probability": probability,
        "label": label,
        "status": "completed" if probability >= 0.5 else "incomplete",
    }


def predict_task_intervention(task: Dict[str, Any], model: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    prediction = predict_task_completion(task, model=model)
    probability = prediction["probability"]
    completed = bool(task.get("completed"))
    age_days = max(0.0, _task_time_feature(task) * 365.0)

    if completed:
        return {
            "probability": probability,
            "label": "completed",
            "status": "completed",
            "risk_level": "low",
            "recommended_action": "monitor",
            "suggested_priority": (task.get("priority") or "medium").lower(),
            "intervention_required": False,
            "reason": "Task is already marked as completed.",
        }

    if probability < 0.30 and age_days > 7:
        risk_level = "high"
        recommended_action = "escalate"
        suggested_priority = "high"
        reason = "Task is both old and has a significantly low completion probability."
    elif probability < 0.50 or age_days > 3:
        risk_level = "medium"
        recommended_action = "follow_up"
        suggested_priority = "medium"
        reason = "Task is at moderate risk and may need follow-up."
    else:
        risk_level = "low"
        recommended_action = "monitor"
        suggested_priority = "low"
        reason = "Task is healthy and likely to complete."

    return {
        "probability": probability,
        "label": prediction["label"],
        "status": prediction["status"],
        "risk_level": risk_level,
        "recommended_action": recommended_action,
        "suggested_priority": suggested_priority,
        "intervention_required": risk_level != "low",
        "reason": reason if isinstance(reason, str) else reason[0],
    }
