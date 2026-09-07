import mlflow
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

query = 'Search for UI tasks'
candidates = [
    'Search for UI tasks',
    'Search for all demo tasks',
    'Find all frontend tasks',
    'Search for all QA tasks',
    'Create a task for Front-End Development',
    'Create a task for UI Design',
    'Search for design tasks and update the wireframe creation task',
    'List all completed tasks',
    'Update the wireframe creation task to reopen it',
]

models = [
    'C:/Users/shradd163152/Documents/python/backend_fastapi/hg_model',
    'C:/Users/shradd163152/Documents/python/backend_fastapi/fine_tuned_task_embedder',
]

mlflow.set_tracking_uri('file:./mlruns')
mlflow.set_experiment('sentence-transformer-comparison')

with mlflow.start_run(run_name='ui-task-search-comparison'):
    mlflow.log_param('query', query)
    mlflow.log_param('candidate_count', str(len(candidates)))

    for model_name in models:
        model = SentenceTransformer(model_name)
        q = model.encode(query, normalize_embeddings=True)
        cs = model.encode(candidates, normalize_embeddings=True)
        sims = cosine_similarity([q], cs)[0]
        ranked = sorted(zip(candidates, sims), key=lambda x: x[1], reverse=True)

        top_text, top_score = ranked[0]
        model_key = model_name.split('/')[-1]
        mlflow.log_metric(f'{model_key}_top1_score', float(top_score))
        mlflow.log_metric(f'{model_key}_top1_text_rank', 1.0)

        print(f'\nMODEL: {model_name}')
        for text, score in ranked[:5]:
            print(f'{score:.4f} | {text}')

        mlflow.log_text('\n'.join(f'{score:.4f} | {text}' for text, score in ranked[:5]), f'{model_key}_top5.txt')
