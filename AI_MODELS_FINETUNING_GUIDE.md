# AI Models Fine-Tuning Guide

This project uses two different model types for two different jobs:

| Model | Job | Fine-tuning method | Output |
|---|---|---|---|
| `TinyLlama/TinyLlama-1.1B-Chat-v1.0` | Generate task-oriented chat responses and tool proposals | Supervised fine-tuning with LoRA | A small adapter directory |
| `nreimers/MiniLM-L6-H384-uncased` | Convert task queries and task text into embeddings for semantic search | Sentence-transformers contrastive training | A complete sentence-transformer model |

These models should not be fine-tuned in the same way. TinyLlama is a causal language model, so it learns assistant responses from chat examples. MiniLM is an embedding model, so it learns which queries and task texts should be close in vector space.

## 1. TinyLlama LoRA Fine-Tuning

### Purpose

Use TinyLlama when the assistant needs to learn the project's task-management response format, intent labels, and tool arguments. The training examples should contain chat messages ending with an assistant response, for example:

```json
{"messages":[{"role":"user","content":"Create a task to add API rate limiting"},{"role":"assistant","content":"{\"tool\":\"create_task\",\"args\":{\"title\":\"Add API rate limiting\"},\"confirm\":false}"}]}
```

### Requirements

Install the training dependencies in the active virtual environment:

```bash
pip install torch transformers datasets peft trl accelerate certifi
```

A CUDA GPU is strongly recommended. CPU training is supported by the script but is intended for small experiments and will be slow.

### Train the adapter

From the repository root:

```bash
python backend_fastapi/local_lora_finetune.py \
  --model-id TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --train-file backend_fastapi/strict_task_tool_training_data.jsonl \
  --output-dir backend_fastapi/local_lora_adapter_tinyllama \
  --epochs 2
```

The script downloads the base model, applies LoRA to the attention projection layers, formats the JSONL chat examples with the tokenizer's chat template, and saves only the adapter plus tokenizer files in the output directory.

Useful options include:

```bash
--per-device-train-batch-size 2
--gradient-accumulation-steps 4
--learning-rate 1e-4
--max-seq-length 512
```

### Test the adapter

The current training script saves the adapter. A small inference example can load it with PEFT:

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
adapter_dir = "backend_fastapi/local_lora_adapter_tinyllama"

tokenizer = AutoTokenizer.from_pretrained(base_id)
model = AutoModelForCausalLM.from_pretrained(base_id)
model = PeftModel.from_pretrained(model, adapter_dir)

prompt = "Create a task to implement OAuth login"
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=160)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

Fine-tuning does not automatically make the FastAPI application use the adapter. The application must be extended to load the base model and adapter, or the adapter must be merged and served through the configured model service. The current `mistral_client.py` integration uses the model selected by `MISTRAL_MODEL`; it does not load this adapter automatically.

## 2. MiniLM Embedding Fine-Tuning

### Purpose

Use MiniLM when semantic search should understand project-specific language. The model learns pairs such as:

```text
Query:   "Add throttling to the API endpoints"
Positive: "Implement API rate limiting"
```

After training, these texts should produce nearby embeddings and rank together in pgvector search.

The base model is `nreimers/MiniLM-L6-H384-uncased`. The repository's `backend_fastapi/hg_model/` directory is a local model copy, and `backend_fastapi/fine_tuned_task_embedder/` is the expected fine-tuning output directory.

### Requirements

Install the embedding-training dependencies:

```bash
pip install sentence-transformers torch scikit-learn
```

### Train the embedding model

From the repository root:

```bash
python backend_fastapi/sentence_transformer_task_finetune.py
```

The script reads these datasets:

- `it_projects_training_data.jsonl`
- `it_projects_training_data_intent_style.jsonl`
- `it_projects_training_data_combined_intent.jsonl`

It extracts the user query and task title/description, removes duplicate pairs, and trains with `MultipleNegativesRankingLoss`. The resulting model is written to:

```text
backend_fastapi/fine_tuned_task_embedder/
```

For a custom run, call `train_sentence_transformer` from Python with a different model path, dataset list, output directory, epoch count, batch size, or learning rate.

### Use the trained embedding model

Point the embedding configuration at the output directory before starting the FastAPI service:

```bash
export LOCAL_EMBEDDING_MODEL_PATH=backend_fastapi/fine_tuned_task_embedder
export USE_SENTENCE_TRANSFORMERS=true
```

On Windows PowerShell:

```powershell
$env:LOCAL_EMBEDDING_MODEL_PATH = "backend_fastapi/fine_tuned_task_embedder"
$env:USE_SENTENCE_TRANSFORMERS = "true"
```

Re-index existing tasks after changing the embedding model. Vectors generated by different embedding models should not be mixed in the same search index, even when they have the same dimension.

## Which Model Should Be Fine-Tuned?

| Requirement | Model |
|---|---|
| Better semantic task search | MiniLM |
| More consistent assistant/tool JSON output | TinyLlama |
| Lowest operational complexity | Existing configured Mistral API or current local fallback |
| Training on a modest GPU | TinyLlama with LoRA |
| Retrieval quality over project terminology | MiniLM with contrastive pairs |

Start with MiniLM if the main problem is finding the right tasks. Fine-tune TinyLlama only when the assistant's response style or tool-call format remains unreliable after improving prompts and validation.

## Evaluation Checklist

Before replacing the current model, compare the fine-tuned model with the base model using a held-out dataset:

- Tool name and argument accuracy for TinyLlama
- Valid JSON rate for tool proposals
- Retrieval recall and top-k ranking for MiniLM
- False matches between unrelated task types
- Response latency and memory usage
- Behavior on ambiguous requests and prompt-injection attempts

Keep the base model and training data versioned so the fine-tuned output can be reproduced or rolled back.