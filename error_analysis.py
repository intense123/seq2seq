"""
Error analysis script for Seq2Seq models.
Analyzes failure modes, syntax errors, and performance patterns.
"""

import os
import json
import argparse
import pickle
import ast
from typing import List, Dict, Tuple
from collections import defaultdict

import torch
import numpy as np
from tqdm import tqdm

from data import create_dataloaders, PAD_IDX, SOS_IDX, EOS_IDX, Vocabulary
from models import (
    create_rnn_seq2seq,
    create_lstm_seq2seq,
    create_lstm_attention_seq2seq
)


def check_syntax_error(code_str: str) -> Tuple[bool, str]:
    """
    Check if generated code has syntax errors.
    
    Args:
        code_str: Generated code as string
    
    Returns:
        (has_error, error_message)
    """
    try:
        ast.parse(code_str)
        return False, ""
    except SyntaxError as e:
        return True, str(e)
    except Exception as e:
        return True, f"Parse error: {str(e)}"


def decode_sequence(indices: List[int], vocab: Vocabulary) -> str:
    """
    Decode a sequence to a string.
    
    Args:
        indices: Token indices
        vocab: Vocabulary
    
    Returns:
        Decoded string
    """
    tokens = []
    for idx in indices:
        if idx == EOS_IDX or idx == PAD_IDX:
            break
        if idx == SOS_IDX:
            continue
        tokens.append(vocab.idx2token.get(idx, '<unk>'))
    return ' '.join(tokens)


def categorize_errors(
    predictions: List[str],
    references: List[str],
    sources: List[str]
) -> Dict:
    """
    Categorize different types of errors.
    
    Args:
        predictions: Predicted code strings
        references: Reference code strings
        sources: Source docstrings
    
    Returns:
        Dictionary with error categories and examples
    """
    error_categories = {
        'syntax_errors': [],
        'missing_keywords': [],
        'wrong_operators': [],
        'incomplete_code': [],
        'extra_tokens': [],
        'variable_name_errors': [],
        'indentation_issues': []
    }
    
    # Keywords to check
    python_keywords = {'def', 'return', 'if', 'else', 'for', 'while', 'class', 'import', 'from'}
    
    for i, (pred, ref, src) in enumerate(zip(predictions, references, sources)):
        pred_tokens = set(pred.split())
        ref_tokens = set(ref.split())
        
        # Check syntax errors
        has_syntax_error, error_msg = check_syntax_error(pred)
        if has_syntax_error:
            error_categories['syntax_errors'].append({
                'index': i,
                'source': src,
                'prediction': pred,
                'reference': ref,
                'error': error_msg
            })
        
        # Check missing keywords
        missing_keywords = (ref_tokens & python_keywords) - (pred_tokens & python_keywords)
        if missing_keywords:
            error_categories['missing_keywords'].append({
                'index': i,
                'source': src,
                'prediction': pred,
                'reference': ref,
                'missing': list(missing_keywords)
            })
        
        # Check for incomplete code (prediction much shorter than reference)
        pred_len = len(pred.split())
        ref_len = len(ref.split())
        if pred_len < ref_len * 0.5:
            error_categories['incomplete_code'].append({
                'index': i,
                'source': src,
                'prediction': pred,
                'reference': ref,
                'pred_len': pred_len,
                'ref_len': ref_len
            })
        
        # Check for extra tokens
        if pred_len > ref_len * 1.5:
            error_categories['extra_tokens'].append({
                'index': i,
                'source': src,
                'prediction': pred,
                'reference': ref,
                'pred_len': pred_len,
                'ref_len': ref_len
            })
    
    return error_categories


def analyze_length_correlation(
    predictions: List[str],
    references: List[str],
    sources: List[str]
) -> Dict:
    """
    Analyze correlation between source length and performance.
    
    Args:
        predictions: Predicted code
        references: Reference code
        sources: Source docstrings
    
    Returns:
        Analysis results
    """
    length_buckets = defaultdict(lambda: {'correct': 0, 'total': 0, 'examples': []})
    
    for pred, ref, src in zip(predictions, references, sources):
        src_len = len(src.split())
        
        # Determine bucket
        if src_len <= 10:
            bucket = '0-10'
        elif src_len <= 20:
            bucket = '11-20'
        elif src_len <= 30:
            bucket = '21-30'
        elif src_len <= 40:
            bucket = '31-40'
        else:
            bucket = '41+'
        
        length_buckets[bucket]['total'] += 1
        
        if pred == ref:
            length_buckets[bucket]['correct'] += 1
        elif len(length_buckets[bucket]['examples']) < 3:
            length_buckets[bucket]['examples'].append({
                'source': src,
                'prediction': pred,
                'reference': ref
            })
    
    # Calculate accuracy per bucket
    results = {}
    for bucket, data in length_buckets.items():
        results[bucket] = {
            'accuracy': data['correct'] / data['total'] if data['total'] > 0 else 0,
            'total_samples': data['total'],
            'correct_samples': data['correct'],
            'examples': data['examples']
        }
    
    return results


def analyze_common_patterns(
    predictions: List[str],
    references: List[str],
    sources: List[str],
    top_k: int = 10
) -> Dict:
    """
    Analyze common patterns in successful and failed predictions.
    
    Args:
        predictions: Predicted code
        references: Reference code
        sources: Source docstrings
        top_k: Number of top patterns to return
    
    Returns:
        Analysis results
    """
    success_patterns = defaultdict(int)
    failure_patterns = defaultdict(int)
    
    for pred, ref, src in zip(predictions, references, sources):
        # Extract key tokens from source
        src_tokens = src.lower().split()
        
        # Check if prediction is correct
        is_correct = (pred.strip() == ref.strip())
        
        # Count patterns
        for token in src_tokens:
            if len(token) > 3:  # Filter out short words
                if is_correct:
                    success_patterns[token] += 1
                else:
                    failure_patterns[token] += 1
    
    # Get top patterns
    top_success = sorted(success_patterns.items(), key=lambda x: x[1], reverse=True)[:top_k]
    top_failure = sorted(failure_patterns.items(), key=lambda x: x[1], reverse=True)[:top_k]
    
    return {
        'success_patterns': dict(top_success),
        'failure_patterns': dict(top_failure)
    }


def generate_error_report(
    model_name: str,
    predictions: List[str],
    references: List[str],
    sources: List[str],
    output_path: str
):
    """
    Generate comprehensive error analysis report.
    
    Args:
        model_name: Name of the model
        predictions: Predicted code
        references: Reference code
        sources: Source docstrings
        output_path: Path to save report
    """
    with open(output_path, 'w') as f:
        f.write(f"ERROR ANALYSIS REPORT: {model_name}\n")
        f.write("=" * 80 + "\n\n")
        
        # Overall statistics
        total = len(predictions)
        exact_matches = sum(p == r for p, r in zip(predictions, references))
        f.write(f"Overall Statistics:\n")
        f.write(f"  Total samples: {total}\n")
        f.write(f"  Exact matches: {exact_matches} ({exact_matches/total*100:.2f}%)\n")
        f.write(f"  Errors: {total - exact_matches} ({(total-exact_matches)/total*100:.2f}%)\n\n")
        
        # Error categorization
        f.write("=" * 80 + "\n")
        f.write("ERROR CATEGORIZATION\n")
        f.write("=" * 80 + "\n\n")
        
        error_categories = categorize_errors(predictions, references, sources)
        
        for category, errors in error_categories.items():
            f.write(f"\n{category.upper().replace('_', ' ')}:\n")
            f.write(f"  Count: {len(errors)}\n")
            
            if errors:
                f.write(f"\n  Examples (showing up to 3):\n")
                for i, error in enumerate(errors[:3], 1):
                    f.write(f"\n  Example {i}:\n")
                    f.write(f"    Source: {error['source']}\n")
                    f.write(f"    Reference: {error['reference']}\n")
                    f.write(f"    Prediction: {error['prediction']}\n")
                    if 'error' in error:
                        f.write(f"    Error: {error['error']}\n")
                    if 'missing' in error:
                        f.write(f"    Missing keywords: {error['missing']}\n")
        
        # Length correlation analysis
        f.write("\n" + "=" * 80 + "\n")
        f.write("PERFORMANCE BY SOURCE LENGTH\n")
        f.write("=" * 80 + "\n\n")
        
        length_analysis = analyze_length_correlation(predictions, references, sources)
        
        for bucket in sorted(length_analysis.keys()):
            data = length_analysis[bucket]
            f.write(f"\nLength bucket {bucket} tokens:\n")
            f.write(f"  Total samples: {data['total_samples']}\n")
            f.write(f"  Correct: {data['correct_samples']}\n")
            f.write(f"  Accuracy: {data['accuracy']:.4f}\n")
            
            if data['examples']:
                f.write(f"\n  Failure examples:\n")
                for i, example in enumerate(data['examples'], 1):
                    f.write(f"\n    Example {i}:\n")
                    f.write(f"      Source: {example['source']}\n")
                    f.write(f"      Reference: {example['reference']}\n")
                    f.write(f"      Prediction: {example['prediction']}\n")
        
        # Common patterns
        f.write("\n" + "=" * 80 + "\n")
        f.write("COMMON PATTERNS\n")
        f.write("=" * 80 + "\n\n")
        
        patterns = analyze_common_patterns(predictions, references, sources)
        
        f.write("Success-associated tokens (appear in correct predictions):\n")
        for token, count in patterns['success_patterns'].items():
            f.write(f"  {token}: {count}\n")
        
        f.write("\nFailure-associated tokens (appear in incorrect predictions):\n")
        for token, count in patterns['failure_patterns'].items():
            f.write(f"  {token}: {count}\n")


def main():
    parser = argparse.ArgumentParser(description='Error analysis for Seq2Seq models')
    parser.add_argument('--model', type=str, default='all', choices=['rnn', 'lstm', 'attention', 'all'])
    parser.add_argument('--data_dir', type=str, default='./data')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints')
    parser.add_argument('--results_dir', type=str, default='./results')
    parser.add_argument('--embed_dim', type=int, default=256)
    parser.add_argument('--hidden_dim', type=int, default=256)
    parser.add_argument('--num_layers', type=int, default=1)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--batch_size', type=int, default=32)
    
    args = parser.parse_args()
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Load data
    print('Loading data...')
    _, _, test_loader, src_vocab, tgt_vocab = create_dataloaders(
        args.data_dir,
        batch_size=args.batch_size
    )
    
    # Load test data
    with open(f'{args.data_dir}/test_data.pkl', 'rb') as f:
        test_data = pickle.load(f)
    
    src_texts = [doc for doc, _ in test_data]
    
    # Models to analyze
    models_to_analyze = ['rnn', 'lstm', 'attention'] if args.model == 'all' else [args.model]
    
    for model_type in models_to_analyze:
        print(f'\n{"="*60}')
        print(f'Analyzing {model_type.upper()} model')
        print(f'{"="*60}')
        
        # Create model
        if model_type == 'rnn':
            model = create_rnn_seq2seq(
                len(src_vocab), len(tgt_vocab), args.embed_dim,
                args.hidden_dim, args.num_layers, args.dropout, device
            )
            model_name = 'rnn_seq2seq'
        elif model_type == 'lstm':
            model = create_lstm_seq2seq(
                len(src_vocab), len(tgt_vocab), args.embed_dim,
                args.hidden_dim, args.num_layers, args.dropout, device
            )
            model_name = 'lstm_seq2seq'
        else:
            model = create_lstm_attention_seq2seq(
                len(src_vocab), len(tgt_vocab), args.embed_dim,
                args.hidden_dim, args.num_layers, args.dropout, device
            )
            model_name = 'lstm_attention_seq2seq'
        
        # Load checkpoint
        checkpoint_path = os.path.join(args.checkpoint_dir, f'{model_name}_best.pt')
        if not os.path.exists(checkpoint_path):
            print(f'Checkpoint not found: {checkpoint_path}')
            continue
        
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        # Generate predictions
        print('Generating predictions...')
        predictions = []
        references = []
        
        with torch.no_grad():
            for src, tgt, src_lengths, _ in tqdm(test_loader):
                src = src.to(device)
                src_lengths = src_lengths.to(device)
                
                # Generate
                if model_type == 'attention':
                    generated, _ = model.generate(src, src_lengths, 100, SOS_IDX, EOS_IDX)
                else:
                    generated = model.generate(src, src_lengths, 100, SOS_IDX, EOS_IDX)
                
                # Decode
                for gen, ref in zip(generated.cpu().numpy(), tgt.cpu().numpy()):
                    pred_str = decode_sequence(gen.tolist(), tgt_vocab)
                    ref_str = decode_sequence(ref.tolist(), tgt_vocab)
                    predictions.append(pred_str)
                    references.append(ref_str)
        
        # Generate error report
        report_path = os.path.join(args.results_dir, f'{model_name}_error_analysis.txt')
        generate_error_report(model_name, predictions, references, src_texts, report_path)
        print(f'Error analysis saved to {report_path}')
    
    print('\nError analysis complete!')


if __name__ == '__main__':
    main()
