#!/usr/bin/env python3
"""
Local fine-tuning with PEFT/LoRA for a small open model.

This is the replacement for the deprecated hosted Mistral fine-tuning endpoint.
It keeps the Docker footprint small by training a tiny local model with LoRA,
not by downloading a full 7B model unless you explicitly choose a larger one.

Recommended default:
    TinyLlama/TinyLlama-1.1B-Chat-v1.0

Usage:
d_fastapi/local_lora_finetune.py \
        --model-id TinyLlama/TinyLlama-1.1B-Chat-v1.0 \    python backen
        --train-file backend_fastapi/strict_task_tool_training_data.jsonl \
        --output-dir /tmp/tinyllama-task-lora
"""

import argparse
import inspect
import json
import os
from pathlib import Path

import certifi
import requests
import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
from trl import SFTConfig, SFTTrainer


def configure_environment_for_local_cert_issue():
    """Use Netskope CA certificates with the standard Python trust store.

    This is safer than disabling TLS verification entirely and matches the
    corporate certificate setup used in this environment.
    """
    base_dir = Path(__file__).resolve().parent
    candidate_files = [
        base_dir / "netskope-ca.pem",
        base_dir / "netskope-ca.crt",
    ]

    bundle_path = None
    for candidate in candidate_files:
        if candidate.exists():
            bundle_path = candidate
            break

    if bundle_path is not None:
        combined_bundle = base_dir / "netskope-ca-combined.pem"
        certifi_bundle = certifi.where()
        certifi_text = Path(certifi_bundle).read_text(encoding="utf-8")
        netskope_text = bundle_path.read_text(encoding="utf-8")
        if not combined_bundle.exists() or combined_bundle.read_text(encoding="utf-8") != certifi_text + "\n" + netskope_text:
            combined_bundle.write_text(certifi_text + "\n" + netskope_text, encoding="utf-8")
        os.environ["REQUESTS_CA_BUNDLE"] = str(combined_bundle)
        os.environ["SSL_CERT_FILE"] = str(combined_bundle)
        os.environ["CURL_CA_BUNDLE"] = str(combined_bundle)
        os.environ["HF_HUB_DISABLE_SSL_VERIFY"] = "0"
        os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
        os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
        os.environ["HF_XET_DISABLED"] = "1"


configure_environment_for_local_cert_issue()


def format_example(messages, tokenizer):
    if not isinstance(messages, list) or len(messages) < 2:
        raise ValueError(f"Expected at least 2 messages, got: {messages!r}")

    if messages[-1].get("role") != "assistant":
        raise ValueError("Each training example must end with an assistant response")
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)


def load_jsonl_dataset(file_path: str, tokenizer):
    rows = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            rows.append({"text": format_example(obj.get("messages", []), tokenizer)})
    return Dataset.from_list(rows)


def resolve_model_path(model_id: str) -> str:
    if os.path.exists(model_id):
        return os.path.abspath(model_id)
    return model_id


def infer_target_modules(model):
    model_type = getattr(model.config, "model_type", "")
    if model_type in {"gpt2", "distilgpt2"}:
        return ["c_attn"]
    return ["q_proj", "v_proj"]


def _load_causal_lm_with_compatible_dtype(model_id: str):
    kwargs = {"device_map": "auto"}
    model_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    try:
        signature = inspect.signature(AutoModelForCausalLM.from_pretrained)
        if "dtype" in signature.parameters:
            kwargs["dtype"] = model_dtype
        else:
            kwargs["torch_dtype"] = model_dtype
    except (TypeError, ValueError):
        kwargs["torch_dtype"] = model_dtype
    return AutoModelForCausalLM.from_pretrained(model_id, **kwargs)


def build_model_and_tokenizer(model_id: str):
    resolved_model_id = resolve_model_path(model_id)
    tokenizer = AutoTokenizer.from_pretrained(resolved_model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = _load_causal_lm_with_compatible_dtype(resolved_model_id)

    config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=infer_target_modules(model),
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, config)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {trainable:,} / {total:,} ({100.0 * trainable / total:.2f}%)")
    return model, tokenizer


def train(args):
    model, tokenizer = build_model_and_tokenizer(args.model_id)
    dataset = load_jsonl_dataset(args.train_file, tokenizer)

    training_args = SFTConfig(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        num_train_epochs=args.epochs,
        logging_steps=10,
        save_strategy="epoch",
        report_to=[],
        fp16=torch.cuda.is_available(),
        use_cpu=not torch.cuda.is_available(),
        max_seq_length=args.max_seq_length,
        dataset_text_field="text",
        packing=False,
    )

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=dataset,
        args=training_args,
    )

    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Saved local LoRA adapter to: {args.output_dir}")


def generate_answer(model_id: str, checkpoint_dir: str, prompt: str, max_new_tokens: int = 200):
    resolved_model_id = resolve_model_path(model_id)
    tokenizer = AutoTokenizer.from_pretrained(resolved_model_id)
    model = _load_causal_lm_with_compatible_dtype(resolved_model_id)

    from peft import PeftModel
    model = PeftModel.from_pretrained(model, checkpoint_dir)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=True, temperature=0.7)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


def parse_args():
    parser = argparse.ArgumentParser(description="Local LoRA fine-tuning for small open models.")
    parser.add_argument("--model-id", default=os.getenv("LOCAL_LORA_MODEL_ID", "TinyLlama/TinyLlama-1.1B-Chat-v1.0"))
    parser.add_argument("--train-file", default=os.getenv("LOCAL_LORA_TRAIN_FILE", "backend_fastapi/strict_task_tool_training_data.jsonl"))
    parser.add_argument("--output-dir", default=os.getenv("LOCAL_LORA_OUTPUT_DIR", "./local_lora_adapter"))
    parser.add_argument("--epochs", type=int, default=int(os.getenv("LOCAL_LORA_EPOCHS", "2")))
    parser.add_argument("--learning-rate", type=float, default=float(os.getenv("LOCAL_LORA_LR", "1e-4")))
    parser.add_argument("--per-device-train-batch-size", type=int, default=int(os.getenv("LOCAL_LORA_BATCH_SIZE", "2")))
    parser.add_argument("--gradient-accumulation-steps", type=int, default=int(os.getenv("LOCAL_LORA_GRAD_ACCUM", "4")))
    parser.add_argument("--max-seq-length", type=int, default=int(os.getenv("LOCAL_LORA_MAX_SEQ", "512")))
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if not Path(args.train_file).exists():
        raise FileNotFoundError(f"Training file not found: {args.train_file}")
    train(args)
