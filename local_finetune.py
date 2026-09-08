#!/usr/bin/env python3
"""
Quick Start: Local SFT + LoRA Fine-Tuning for Mistral 7B

This script provides a simplified entry point to fine-tune Mistral locally
without needing the Mistral API. Perfect for experimenting and domain adaptation.

Requirements:
- GPU with 30GB+ VRAM (A100, RTX 4090, or equivalent)
- Python 3.9+
"""

import argparse
from pathlib import Path
from backend_fastapi.sft_lora_training import SFTWithLoRA, TrainingConfig


def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune Mistral 7B locally using SFT + LoRA",
        epilog="""
Examples:
  # 1. Train on IT projects data
  python local_finetune.py train \\
    --data-file it_projects_training_data.jsonl \\
    --output-dir mistral-task-mgmt \\
    --epochs 3 \\
    --batch-size 8
  
  # 2. Use trained model for inference
  python local_finetune.py inference \\
    --checkpoint mistral-task-mgmt \\
    --prompt "Create a task for database optimization"
  
  # 3. Compare base vs fine-tuned model
  python local_finetune.py compare \\
    --checkpoint mistral-task-mgmt \\
    --prompt "Implement GraphQL API"
  
  # 4. Use custom LoRA rank (higher = more capacity)
  python local_finetune.py train \\
    --data-file it_projects_training_data.jsonl \\
    --lora-rank 16 \\
    --lora-alpha 32
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Train command
    train_cmd = subparsers.add_parser('train', help='Train model with SFT + LoRA')
    train_cmd.add_argument('--data-file', required=True, help='Path to JSONL training data')
    train_cmd.add_argument('--output-dir', default='mistral-lora-task-mgmt', help='Output directory')
    train_cmd.add_argument('--model-id', default='mistralai/Mistral-7B', help='Base model ID')
    train_cmd.add_argument('--epochs', type=int, default=3, help='Number of epochs')
    train_cmd.add_argument('--batch-size', type=int, default=8, help='Batch size per GPU')
    train_cmd.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    train_cmd.add_argument('--lora-rank', type=int, default=8, help='LoRA rank')
    train_cmd.add_argument('--lora-alpha', type=int, default=16, help='LoRA alpha')
    train_cmd.add_argument('--warmup-steps', type=int, default=100, help='Warmup steps')
    train_cmd.add_argument('--max-seq-length', type=int, default=512, help='Max sequence length')
    
    # Inference command
    infer_cmd = subparsers.add_parser('inference', help='Use trained model')
    infer_cmd.add_argument('--checkpoint', required=True, help='Path to checkpoint')
    infer_cmd.add_argument('--prompt', required=True, help='Input prompt')
    infer_cmd.add_argument('--model-id', default='mistralai/Mistral-7B', help='Base model ID')
    
    # Compare command
    compare_cmd = subparsers.add_parser('compare', help='Compare base vs fine-tuned')
    compare_cmd.add_argument('--checkpoint', required=True, help='Path to checkpoint')
    compare_cmd.add_argument('--prompt', required=True, help='Input prompt')
    compare_cmd.add_argument('--model-id', default='mistralai/Mistral-7B', help='Base model ID')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # Create config
    config = TrainingConfig(
        model_id=getattr(args, 'model_id', 'mistralai/Mistral-7B'),
        output_dir=getattr(args, 'output_dir', 'mistral-lora-task-mgmt'),
        num_epochs=getattr(args, 'epochs', 3),
        batch_size=getattr(args, 'batch_size', 8),
        learning_rate=getattr(args, 'lr', 1e-4),
        lora_rank=getattr(args, 'lora_rank', 8),
        lora_alpha=getattr(args, 'lora_alpha', 16),
        warmup_steps=getattr(args, 'warmup_steps', 100),
        max_seq_length=getattr(args, 'max_seq_length', 512),
    )
    
    # Initialize trainer
    trainer = SFTWithLoRA(config)
    
    if args.command == 'train':
        print("\n" + "="*60)
        print("🚀 LOCAL SFT + LoRA FINE-TUNING")
        print("="*60)
        print("\nWhat this does:")
        print("  1. Loads Mistral 7B base model (frozen)")
        print("  2. Adds LoRA adapters (~0.3M trainable parameters)")
        print("  3. Trains adapters on your task management data")
        print("  4. Saves trained adapters (~1-2 MB)")
        print("\nWhy it's efficient:")
        print("  ✓ Only trains 0.1% of model weights")
        print("  ✓ Memory: 30 GB instead of 84 GB")
        print("  ✓ Speed: 4-8 hours instead of 7-14 days")
        print("  ✓ Quality: 98-99% of full fine-tuning")
        print("\n" + "="*60)
        
        trainer.train(args.data_file)
        
        print("\n✅ Training complete!")
        print(f"📁 Checkpoint saved to: {config.output_dir}")
        print(f"\nNext step: Run inference")
        print(f"  python local_finetune.py inference \\")
        print(f"    --checkpoint {config.output_dir} \\")
        print(f"    --prompt 'Create a task for API rate limiting'")
    
    elif args.command == 'inference':
        print("\n" + "="*60)
        print("🤖 INFERENCE WITH TRAINED LoRA")
        print("="*60)
        trainer.inference(args.checkpoint, args.prompt)
    
    elif args.command == 'compare':
        print("\n" + "="*60)
        print("🔄 COMPARISON: Base vs Fine-Tuned Model")
        print("="*60)
        trainer.compare_with_base(args.prompt)


if __name__ == "__main__":
    main()
