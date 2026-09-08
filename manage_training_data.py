#!/usr/bin/env python3
"""
Merge and combine training data from multiple sources.

Combine IT projects data with generic task data for comprehensive fine-tuning.
"""

import json
from pathlib import Path
from typing import List, Dict, Any


def merge_jsonl_files(input_files: List[str], output_file: str) -> int:
    """
    Merge multiple JSONL files into one.
    
    Args:
        input_files: List of input JSONL file paths
        output_file: Output JSONL file path
    
    Returns:
        Total number of examples merged
    """
    count = 0
    
    with open(output_file, 'w', encoding='utf-8') as out_f:
        for input_file in input_files:
            try:
                with open(input_file, 'r', encoding='utf-8') as in_f:
                    for line in in_f:
                        if line.strip():
                            out_f.write(line)
                            count += 1
                print(f"✓ Merged {input_file}: {count} total examples so far")
            except FileNotFoundError:
                print(f"✗ File not found: {input_file}")
            except Exception as e:
                print(f"✗ Error reading {input_file}: {e}")
    
    print(f"\n✓ Merged {count} total examples to {output_file}")
    return count


def deduplicate_jsonl(input_file: str, output_file: str) -> int:
    """
    Remove duplicate examples from JSONL file.
    
    Args:
        input_file: Input JSONL file
        output_file: Output deduplicated JSONL file
    
    Returns:
        Number of deduplicated examples
    """
    seen = set()
    count = 0
    duplicates = 0
    
    with open(input_file, 'r', encoding='utf-8') as in_f:
        with open(output_file, 'w', encoding='utf-8') as out_f:
            for line in in_f:
                if not line.strip():
                    continue
                
                # Use the line itself as the key (exact match deduplication)
                line_hash = hash(line.strip())
                
                if line_hash not in seen:
                    seen.add(line_hash)
                    out_f.write(line)
                    count += 1
                else:
                    duplicates += 1
    
    print(f"✓ Deduplicated: {count} unique, {duplicates} removed")
    return count


def balance_task_types(input_file: str, output_file: str) -> Dict[str, int]:
    """
    Balance training data by task type (feature, bug, refactor, etc).
    
    Args:
        input_file: Input JSONL file
        output_file: Balanced output JSONL file
    
    Returns:
        Dictionary with counts by type
    """
    tasks_by_type = {}
    
    # First pass: count by type
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            
            try:
                obj = json.loads(line)
                messages = obj.get('messages', [])
                if messages:
                    # Look for task type in assistant message
                    assistant_msg = messages[-1]['content']
                    if '"type":' in assistant_msg:
                        task_type = 'feature'
                        if '"type": "bug"' in assistant_msg:
                            task_type = 'bug'
                        elif '"type": "refactor"' in assistant_msg:
                            task_type = 'refactor'
                        elif '"type": "technical-debt"' in assistant_msg:
                            task_type = 'technical-debt'
                    else:
                        task_type = 'feature'
                    
                    if task_type not in tasks_by_type:
                        tasks_by_type[task_type] = []
                    tasks_by_type[task_type].append(line)
            except json.JSONDecodeError:
                pass
    
    # Calculate target count (use the minority type as baseline)
    if tasks_by_type:
        min_count = min(len(v) for v in tasks_by_type.values())
        target_count = min_count
        
        print(f"Balancing training data to {target_count} examples per type:")
        print(f"Types found: {list(tasks_by_type.keys())}")
    
    # Second pass: write balanced data
    with open(output_file, 'w', encoding='utf-8') as out_f:
        for task_type, lines in tasks_by_type.items():
            # Take first target_count examples from each type
            for line in lines[:target_count]:
                out_f.write(line)
            print(f"  {task_type}: {min(len(lines), target_count)} examples")
    
    return {k: min(len(v), target_count) for k, v in tasks_by_type.items()}


def filter_by_component(input_file: str, component: str, output_file: str) -> int:
    """
    Filter training data by component.
    
    Args:
        input_file: Input JSONL file
        component: Component to filter (backend, frontend, devops, etc)
        output_file: Output JSONL file
    
    Returns:
        Number of examples for this component
    """
    count = 0
    
    with open(input_file, 'r', encoding='utf-8') as in_f:
        with open(output_file, 'w', encoding='utf-8') as out_f:
            for line in in_f:
                if not line.strip():
                    continue
                
                if f'"component": "{component}"' in line or f'"component":"{component}"' in line:
                    out_f.write(line)
                    count += 1
    
    print(f"✓ Filtered {count} examples for component: {component}")
    return count


def split_train_test(input_file: str, train_file: str, test_file: str, test_ratio: float = 0.2) -> tuple:
    """
    Split training data into train/test sets.
    
    Args:
        input_file: Input JSONL file
        train_file: Output training file
        test_file: Output test file
        test_ratio: Ratio of test examples (0.2 = 20%)
    
    Returns:
        Tuple of (train_count, test_count)
    """
    lines = []
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = [l for l in f if l.strip()]
    
    split_point = int(len(lines) * (1 - test_ratio))
    train_lines = lines[:split_point]
    test_lines = lines[split_point:]
    
    with open(train_file, 'w', encoding='utf-8') as f:
        f.writelines(train_lines)
    
    with open(test_file, 'w', encoding='utf-8') as f:
        f.writelines(test_lines)
    
    print(f"✓ Split data: {len(train_lines)} train, {len(test_lines)} test (ratio: {test_ratio})")
    return len(train_lines), len(test_lines)


def validate_and_report(input_file: str) -> Dict[str, Any]:
    """
    Validate JSONL file and generate report.
    
    Args:
        input_file: Input JSONL file
    
    Returns:
        Dictionary with validation results
    """
    stats = {
        'total': 0,
        'valid': 0,
        'errors': 0,
        'components': {},
        'types': {},
        'priorities': {},
    }
    
    with open(input_file, 'r', encoding='utf-8') as f:
        for idx, line in enumerate(f, 1):
            if not line.strip():
                continue
            
            stats['total'] += 1
            
            try:
                obj = json.loads(line)
                
                # Validate structure
                if 'messages' not in obj:
                    stats['errors'] += 1
                    continue
                
                messages = obj['messages']
                if not isinstance(messages, list) or len(messages) < 2:
                    stats['errors'] += 1
                    continue
                
                stats['valid'] += 1
                
                # Extract stats
                assistant_msg = messages[-1]['content']
                
                # Count components
                for component in ['frontend', 'backend', 'database', 'devops', 'mobile', 'api', 'security']:
                    if f'"{component}"' in assistant_msg:
                        stats['components'][component] = stats['components'].get(component, 0) + 1
                
                # Count types
                for task_type in ['bug', 'feature', 'refactor', 'technical-debt']:
                    if f'"{task_type}"' in assistant_msg or f"'{task_type}'" in assistant_msg:
                        stats['types'][task_type] = stats['types'].get(task_type, 0) + 1
                
                # Count priorities
                for priority in ['critical', 'high', 'medium', 'low']:
                    if f'"{priority}"' in assistant_msg:
                        stats['priorities'][priority] = stats['priorities'].get(priority, 0) + 1
            
            except json.JSONDecodeError as e:
                print(f"✗ JSON error at line {idx}: {e}")
                stats['errors'] += 1
    
    return stats


def print_report(stats: Dict[str, Any]):
    """Print validation report."""
    print("\n" + "="*60)
    print("VALIDATION REPORT")
    print("="*60)
    print(f"Total examples: {stats['total']}")
    print(f"Valid examples: {stats['valid']}")
    print(f"Errors: {stats['errors']}")
    
    if stats['components']:
        print("\nComponents:")
        for comp, count in sorted(stats['components'].items(), key=lambda x: x[1], reverse=True):
            print(f"  {comp}: {count}")
    
    if stats['types']:
        print("\nTask Types:")
        for type_, count in sorted(stats['types'].items(), key=lambda x: x[1], reverse=True):
            print(f"  {type_}: {count}")
    
    if stats['priorities']:
        print("\nPriorities:")
        for priority, count in sorted(stats['priorities'].items(), key=lambda x: x[1], reverse=True):
            print(f"  {priority}: {count}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Training data management utilities",
        epilog="""
Examples:
  # Merge multiple files
  python manage_training_data.py merge \\
    --input sample_training_data.jsonl it_projects_training_data.jsonl \\
    --output combined_training_data.jsonl
  
  # Deduplicate data
  python manage_training_data.py deduplicate \\
    --input combined_training_data.jsonl \\
    --output deduplicated.jsonl
  
  # Balance by task type
  python manage_training_data.py balance \\
    --input combined_training_data.jsonl \\
    --output balanced.jsonl
  
  # Filter by component
  python manage_training_data.py filter \\
    --input combined_training_data.jsonl \\
    --component backend \\
    --output backend_only.jsonl
  
  # Split train/test
  python manage_training_data.py split \\
    --input combined_training_data.jsonl \\
    --train train.jsonl \\
    --test test.jsonl \\
    --ratio 0.2
  
  # Validate and report
  python manage_training_data.py validate \\
    --input combined_training_data.jsonl
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command')
    
    # Merge
    merge_cmd = subparsers.add_parser('merge', help='Merge multiple files')
    merge_cmd.add_argument('--input', nargs='+', required=True, help='Input files')
    merge_cmd.add_argument('--output', required=True, help='Output file')
    
    # Deduplicate
    dedup_cmd = subparsers.add_parser('deduplicate', help='Remove duplicates')
    dedup_cmd.add_argument('--input', required=True, help='Input file')
    dedup_cmd.add_argument('--output', required=True, help='Output file')
    
    # Balance
    balance_cmd = subparsers.add_parser('balance', help='Balance by task type')
    balance_cmd.add_argument('--input', required=True, help='Input file')
    balance_cmd.add_argument('--output', required=True, help='Output file')
    
    # Filter
    filter_cmd = subparsers.add_parser('filter', help='Filter by component')
    filter_cmd.add_argument('--input', required=True, help='Input file')
    filter_cmd.add_argument('--component', required=True, help='Component')
    filter_cmd.add_argument('--output', required=True, help='Output file')
    
    # Split
    split_cmd = subparsers.add_parser('split', help='Split train/test')
    split_cmd.add_argument('--input', required=True, help='Input file')
    split_cmd.add_argument('--train', required=True, help='Train output file')
    split_cmd.add_argument('--test', required=True, help='Test output file')
    split_cmd.add_argument('--ratio', type=float, default=0.2, help='Test ratio')
    
    # Validate
    validate_cmd = subparsers.add_parser('validate', help='Validate data')
    validate_cmd.add_argument('--input', required=True, help='Input file')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
    elif args.command == 'merge':
        merge_jsonl_files(args.input, args.output)
    elif args.command == 'deduplicate':
        deduplicate_jsonl(args.input, args.output)
    elif args.command == 'balance':
        balance_task_types(args.input, args.output)
    elif args.command == 'filter':
        filter_by_component(args.input, args.component, args.output)
    elif args.command == 'split':
        split_train_test(args.input, args.train, args.test, args.ratio)
    elif args.command == 'validate':
        stats = validate_and_report(args.input)
        print_report(stats)
