from datetime import datetime
from types import SimpleNamespace

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

import backend_fastapi.task_prediction as task_prediction
import backend_fastapi.tasks as tasks_module
from backend_fastapi.models import Base, TaskModel
from backend_fastapi.schemas import BulkCreateRequest, TaskCreate
from backend_fastapi.task_prediction import (
    predict_task_completion,
    train_task_completion_model,
)


def test_train_task_completion_model_uses_existing_task_rows():
    tasks = [
        {
            "title": "Plan sprint backlog",
            "description": "Review stories and assign tasks for the sprint.",
            "completed": True,
            "created_at": datetime(2026, 1, 10, 9, 30),
        },
        {
            "title": "Prepare release checklist",
            "description": "Check deployment steps and verify environment readiness.",
            "completed": False,
            "created_at": datetime(2026, 1, 11, 11, 0),
        },
        {
            "title": "Fix production bug",
            "description": "Investigate incident logs and deploy hotfix.",
            "completed": True,
            "created_at": datetime(2026, 2, 3, 16, 45),
        },
        {
            "title": "Update support documentation",
            "description": "Document the new onboarding steps for the team.",
            "completed": False,
            "created_at": datetime(2026, 2, 4, 10, 15),
        },
    ]

    model = train_task_completion_model(tasks=tasks, epochs=5, save_path=None)

    assert model["metadata"]["trained"] is True
    assert model["metadata"]["vocabulary_size"] > 0
    assert model["model_state"]["linear.weight"].shape[0] == 1


def test_predict_task_completion_returns_probability_and_label():
    model = train_task_completion_model(
        tasks=[
            {
                "title": "Write release notes",
                "description": "Document the launch criteria and verify customer updates.",
                "completed": True,
                "created_at": datetime(2026, 3, 1, 8, 0),
            },
            {
                "title": "Clean up backlog",
                "description": "Archive old tickets and prioritize urgent fixes.",
                "completed": False,
                "created_at": datetime(2026, 3, 2, 9, 0),
            },
        ],
        epochs=3,
        save_path=None,
    )

    prediction = predict_task_completion(
        {
            "title": "Release notes",
            "description": "Verify launch checklist and prepare customer communication.",
            "created_at": datetime(2026, 3, 5, 12, 0),
        },
        model=model,
    )

    assert 0.0 <= prediction["probability"] <= 1.0
    assert prediction["label"] in {"completed", "incomplete"}


def test_predict_task_completion_handles_missing_model_file(tmp_path):
    missing_path = tmp_path / "missing_model.pt"
    original_path = task_prediction.DEFAULT_TASK_MODEL_PATH
    task_prediction.DEFAULT_TASK_MODEL_PATH = str(missing_path)

    try:
        prediction = predict_task_completion(
            {
                "title": "Prepare sprint handoff",
                "description": "Document blockers and share dependencies before launch.",
                "completed": False,
                "created_at": datetime(2026, 9, 10, 8, 0),
            }
        )

        assert 0.0 <= prediction["probability"] <= 1.0
        assert prediction["label"] in {"completed", "incomplete"}
    finally:
        task_prediction.DEFAULT_TASK_MODEL_PATH = original_path


def test_task_prediction_uses_sklearn_pipeline_artifact():
    tasks = [
        {
            "title": "Ship deployment",
            "description": "Validate environment and release the patch.",
            "completed": True,
            "created_at": datetime(2026, 3, 1, 8, 0),
            "priority": "high",
        },
        {
            "title": "Review support queue",
            "description": "Answer tickets and clear blockers.",
            "completed": False,
            "created_at": datetime(2026, 3, 2, 9, 0),
            "priority": "medium",
        },
    ]

    model = train_task_completion_model(tasks=tasks, epochs=10, save_path=None)

    assert model["metadata"]["model_type"] == "sklearn_logistic_regression"
    assert model["metadata"]["vectorizer"] == "tfidf"
    assert "pipeline" in model


def test_bulk_create_tasks_preserves_requested_priorities():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(bind=engine)
    Session = sessionmaker(bind=engine, autoflush=False, autocommit=False)
    db = Session()

    original_get_current_user_dep = tasks_module.get_current_user_dep
    original_get_user_roles = tasks_module.get_user_roles
    original_sync_task_document = tasks_module.sync_task_document
    original_retrain = tasks_module.retrain_task_completion_model

    async def fake_get_current_user_dep(request):
        return object()

    tasks_module.get_current_user_dep = fake_get_current_user_dep
    tasks_module.get_user_roles = lambda user: ["admin"]
    tasks_module.sync_task_document = lambda db, task: None
    tasks_module.retrain_task_completion_model = lambda rows: None

    try:
        payload = BulkCreateRequest(
            tasks=[
                TaskCreate(title="Ship migration", description="Move data", priority="high"),
                TaskCreate(title="Review backlog", description="Clean old tickets", priority="low"),
            ]
        )

        response = tasks_module.bulk_create_tasks(payload, request=SimpleNamespace(headers={}), db=db)

        import asyncio
        result = asyncio.run(response)
        assert result.success is True

        saved = db.query(TaskModel).order_by(TaskModel.id).all()
        assert [task.priority for task in saved] == ["high", "low"]
    finally:
        tasks_module.get_current_user_dep = original_get_current_user_dep
        tasks_module.get_user_roles = original_get_user_roles
        tasks_module.sync_task_document = original_sync_task_document
        tasks_module.retrain_task_completion_model = original_retrain
        db.close()


def test_versioned_artifact_loader_tracks_latest_model(tmp_path):
    model_dir = tmp_path / "model_artifacts"
    model_dir.mkdir()

    first_model = {
        "pipeline": None,
        "metadata": {"model_type": "sklearn_logistic_regression", "created_at": "2026-01-01T00:00:00"},
    }
    second_model = {
        "pipeline": None,
        "metadata": {"model_type": "sklearn_logistic_regression", "created_at": "2026-01-02T00:00:00"},
    }

    first_path = task_prediction.save_task_completion_model_artifact(first_model, model_dir=str(model_dir))
    second_path = task_prediction.save_task_completion_model_artifact(second_model, model_dir=str(model_dir))

    latest_path = task_prediction.get_latest_task_completion_model_path(model_dir=str(model_dir))

    assert latest_path == second_path
    assert task_prediction.load_task_completion_model(latest_path)["metadata"]["created_at"] == "2026-01-02T00:00:00"
