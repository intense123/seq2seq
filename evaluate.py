"""
Evaluation script for Seq2Seq models.
Computes BLEU score, token accuracy, and exact match accuracy.
"""

import os
import json
import argparse
import pickle
from typing import List, Dict, Tuple
from collections import defaultdict

import torch
import numpy as np
from tqdm import tqdm
from nltk.translate.bleu_score import sentence_bleu, corpus_bleu, SmoothingFunction

from data import create_dataloaders, PAD_IDX, SOS_IDX, EOS_IDX, Vocabulary
from models import (
    create_rnn_seq2seq,
    create_lstm_seq2seq,
    create_lstm_attention_seq2seq
)


def calculate_token_accuracy(
    predictions: List[List[int]],
    references: List[List[int]]
) -> float:
    """
    Calculate token-level accuracy.
    
    Args:
        predictions: List of predicted token sequences
        references: List of reference token sequences
    
    Returns:
        Token-level accuracy (0-1)
    """
    total_tokens = 0
    correct_tokens = 0
    
    for pred, ref in zip(predictions, references):
        # Truncate to minimum length for fair comparison
        min_len = min(len(pred), len(ref))
        
        total_tokens += min_len
        correct_tokens += sum(p == r for p, r in zip(pred[:min_len], ref[:min_len]))
    
    return correct_tokens / total_tokens if total_tokens > 0 else 0.0


def calculate_exact_match(
    predictions: List[List[int]],
    references: List[List[int]]
) -> float:
    """
    Calculate exact match accuracy (proportion of perfectly matched sequences).
    
    Args:
        predictions: List of predicted token sequences
        references: List of reference token sequences
    
    Returns:
        Exact match accuracy (0-1)
    """
    exact_matches = sum(pred == ref for pred, ref in zip(predictions, references))
    return exact_matches / len(predictions) if predictions else 0.0


def calculate_bleu(
    predictions: List[List[str]],
    references: List[List[str]]
) -> Dict[str, float]:
    """
    Calculate BLEU scores (BLEU-1, BLEU-2, BLEU-3, BLEU-4, and corpus BLEU).
    
    Args:
        predictions: List of predicted token sequences (as strings)
        references: List of reference token sequences (as strings)
    
    Returns:
        Dictionary with BLEU scores
    """
    smoothing = SmoothingFunction().method1
    
    # Calculate sentence-level BLEU scores
    bleu_1_scores = []
    bleu_2_scores = []
    bleu_3_scores = []
    bleu_4_scores = []
    
    for pred, ref in zip(predictions, references):
        # Each reference needs to be a list of token lists
        ref_wrapped = [ref]
        
        # BLEU-1
        bleu_1 = sentence_bleu(ref_wrapped, pred, weights=(1, 0, 0, 0), smoothing_function=smoothing)
        bleu_1_scores.append(bleu_1)
        
        # BLEU-2
        bleu_2 = sentence_bleu(ref_wrapped, pred, weights=(0.5, 0.5, 0, 0), smoothing_function=smoothing)
        bleu_2_scores.append(bleu_2)
        
        # BLEU-3
        bleu_3 = sentence_bleu(ref_wrapped, pred, weights=(0.33, 0.33, 0.33, 0), smoothing_function=smoothing)
        bleu_3_scores.append(bleu_3)
        
        # BLEU-4
        bleu_4 = sentence_bleu(ref_wrapped, pred, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smoothing)
        bleu_4_scores.append(bleu_4)
    
    # Calculate corpus-level BLEU
    references_wrapped = [[ref] for ref in references]
    corpus_bleu_score = corpus_bleu(references_wrapped, predictions, smoothing_function=smoothing)
    
    return {
        'bleu_1': np.mean(bleu_1_scores),
        'bleu_2': np.mean(bleu_2_scores),
        'bleu_3': np.mean(bleu_3_scores),
        'bleu_4': np.mean(bleu_4_scores),
        'corpus_bleu': corpus_bleu_score
    }


def decode_sequence(indices: List[int], vocab: Vocabulary) -> List[str]:
    """
    Decode a sequence of indices to tokens, removing special tokens.
    
    Args:
        indices: List of token indices
        vocab: Vocabulary object
    
    Returns:
        List of token strings
    """
    tokens = []
    for idx in indices:
        # Stop at EOS or PAD token
        if idx == EOS_IDX or idx == PAD_IDX:
            break
        # Skip SOS token
        if idx == SOS_IDX:
            continue
        tokens.append(vocab.idx2token.get(idx, '<unk>'))
    return tokens


def generate_predictions(
    model,
    dataloader,
    device,
    tgt_vocab: Vocabulary,
    max_len: int = 100,
    model_type: str = 'rnn'
) -> Tuple[List[List[int]], List[List[int]], List[List[str]], List[List[str]]]:
    """
    Generate predictions for entire dataset.
    
    Args:
        model: Seq2Seq model
        dataloader: Dataloader for dataset
        device: Device to run on
        tgt_vocab: Target vocabulary
        max_len: Maximum generation length
        model_type: Type of model ('rnn', 'lstm', 'attention')
    
    Returns:
        pred_indices: List of predicted token index sequences
        ref_indices: List of reference token index sequences
        pred_tokens: List of predicted token string sequences
        ref_tokens: List of reference token string sequences
    """
    model.eval()
    
    pred_indices = []
    ref_indices = []
    pred_tokens = []
    ref_tokens = []
    
    with torch.no_grad():
        for src, tgt, src_lengths, tgt_lengths in tqdm(dataloader, desc="Generating predictions"):
            src = src.to(device)
            src_lengths = src_lengths.to(device)
            
            # Generate predictions
            if model_type == 'attention':
                generated, _ = model.generate(src, src_lengths, max_len, SOS_IDX, EOS_IDX)
            else:
                generated = model.generate(src, src_lengths, max_len, SOS_IDX, EOS_IDX)
            
            # Convert to lists
            generated = generated.cpu().numpy()
            tgt = tgt.cpu().numpy()
            
            for gen, ref in zip(generated, tgt):
                # Store indices (without special tokens for reference)
                gen_list = gen.tolist()
                ref_list = ref.tolist()
                
                pred_indices.append(gen_list)
                ref_indices.append(ref_list)
                
                # Decode to tokens
                gen_tokens = decode_sequence(gen_list, tgt_vocab)
                ref_tokens_seq = decode_sequence(ref_list, tgt_vocab)
                
                pred_tokens.append(gen_tokens)
                ref_tokens.append(ref_tokens_seq)
    
    return pred_indices, ref_indices, pred_tokens, ref_tokens


def analyze_by_length(
    pred_indices: List[List[int]],
    ref_indices: List[List[int]],
    pred_tokens: List[List[str]],
    ref_tokens: List[List[str]]
) -> Dict:
    """
    Analyze performance by reference sequence length.
    
    Args:
        pred_indices: Predicted token indices
        ref_indices: Reference token indices
        pred_tokens: Predicted tokens
        ref_tokens: Reference tokens
    
    Returns:
        Dictionary with performance metrics by length range
    """
    # Group by length ranges
    length_ranges = [
        (0, 10),
        (10, 20),
        (20, 30),
        (30, 50),
        (50, 100)
    ]
    
    grouped_data = defaultdict(lambda: {'pred_idx': [], 'ref_idx': [], 'pred_tok': [], 'ref_tok': []})
    
    for pred_idx, ref_idx, pred_tok, ref_tok in zip(pred_indices, ref_indices, pred_tokens, ref_tokens):
        ref_len = len([idx for idx in ref_idx if idx not in [PAD_IDX, SOS_IDX, EOS_IDX]])
        
        for min_len, max_len in length_ranges:
            if min_len <= ref_len < max_len:
                grouped_data[(min_len, max_len)]['pred_idx'].append(pred_idx)
                grouped_data[(min_len, max_len)]['ref_idx'].append(ref_idx)
                grouped_data[(min_len, max_len)]['pred_tok'].append(pred_tok)
                grouped_data[(min_len, max_len)]['ref_tok'].append(ref_tok)
                break
    
    # Calculate metrics for each length range
    results = {}
    for (min_len, max_len), data in grouped_data.items():
        if not data['pred_idx']:
            continue
        
        range_name = f'{min_len}-{max_len}'
        
        token_acc = calculate_token_accuracy(data['pred_idx'], data['ref_idx'])
        exact_match = calculate_exact_match(data['pred_idx'], data['ref_idx'])
        bleu_scores = calculate_bleu(data['pred_tok'], data['ref_tok'])
        
        results[range_name] = {
            'num_samples': len(data['pred_idx']),
            'token_accuracy': token_acc,
            'exact_match': exact_match,
            'bleu_4': bleu_scores['bleu_4']
        }
    
    return results


def evaluate_model(
    model,
    test_loader,
    device,
    tgt_vocab: Vocabulary,
    model_name: str,
    model_type: str = 'rnn'
) -> Dict:
    """
    Evaluate model and return all metrics.
    
    Args:
        model: Seq2Seq model
        test_loader: Test dataloader
        device: Device to run on
        tgt_vocab: Target vocabulary
        model_name: Name of the model
        model_type: Type of model
    
    Returns:
        Dictionary with all evaluation metrics
    """
    print(f'\nEvaluating {model_name}...')
    
    # Generate predictions
    pred_indices, ref_indices, pred_tokens, ref_tokens = generate_predictions(
        model, test_loader, device, tgt_vocab, max_len=100, model_type=model_type
    )
    
    # Calculate metrics
    print('Calculating metrics...')
    
    token_acc = calculate_token_accuracy(pred_indices, ref_indices)
    exact_match = calculate_exact_match(pred_indices, ref_indices)
    bleu_scores = calculate_bleu(pred_tokens, ref_tokens)
    length_analysis = analyze_by_length(pred_indices, ref_indices, pred_tokens, ref_tokens)
    
    results = {
        'token_accuracy': token_acc,
        'exact_match': exact_match,
        'bleu_scores': bleu_scores,
        'length_analysis': length_analysis
    }
    
    # Print results
    print(f'\n{model_name} Results:')
    print(f'  Token Accuracy: {token_acc:.4f}')
    print(f'  Exact Match: {exact_match:.4f}')
    print(f'  BLEU-1: {bleu_scores["bleu_1"]:.4f}')
    print(f'  BLEU-2: {bleu_scores["bleu_2"]:.4f}')
    print(f'  BLEU-3: {bleu_scores["bleu_3"]:.4f}')
    print(f'  BLEU-4: {bleu_scores["bleu_4"]:.4f}')
    print(f'  Corpus BLEU: {bleu_scores["corpus_bleu"]:.4f}')
    
    print('\nPerformance by sequence length:')
    for range_name, metrics in sorted(length_analysis.items()):
        print(f'  {range_name}: {metrics["num_samples"]} samples, '
              f'Token Acc: {metrics["token_accuracy"]:.4f}, '
              f'Exact Match: {metrics["exact_match"]:.4f}, '
              f'BLEU-4: {metrics["bleu_4"]:.4f}')
    
    return results, pred_tokens, ref_tokens


def save_examples(
    pred_tokens: List[List[str]],
    ref_tokens: List[List[str]],
    src_data: List[str],
    output_path: str,
    num_examples: int = 20
):
    """
    Save example predictions to file.
    
    Args:
        pred_tokens: Predicted token sequences
        ref_tokens: Reference token sequences
        src_data: Source text data
        output_path: Path to save examples
        num_examples: Number of examples to save
    """
    with open(output_path, 'w') as f:
        for i in range(min(num_examples, len(pred_tokens))):
            f.write(f'Example {i+1}:\n')
            f.write(f'Source: {src_data[i]}\n')
            f.write(f'Reference: {" ".join(ref_tokens[i])}\n')
            f.write(f'Prediction: {" ".join(pred_tokens[i])}\n')
            f.write(f'Match: {pred_tokens[i] == ref_tokens[i]}\n')
            f.write('\n' + '-'*80 + '\n\n')


def main():
    parser = argparse.ArgumentParser(description='Evaluate Seq2Seq models')
    parser.add_argument('--model', type=str, default='all', choices=['rnn', 'lstm', 'attention', 'all'],
                        help='Model to evaluate')
    parser.add_argument('--data_dir', type=str, default='./data',
                        help='Directory containing preprocessed data')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints',
                        help='Directory containing model checkpoints')
    parser.add_argument('--results_dir', type=str, default='./results',
                        help='Directory to save results')
    parser.add_argument('--embed_dim', type=int, default=256,
                        help='Embedding dimension')
    parser.add_argument('--hidden_dim', type=int, default=256,
                        help='Hidden dimension')
    parser.add_argument('--num_layers', type=int, default=1,
                        help='Number of layers')
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='Dropout')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size')
    
    args = parser.parse_args()
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Load data
    print('Loading data...')
    train_loader, val_loader, test_loader, src_vocab, tgt_vocab = create_dataloaders(
        args.data_dir,
        batch_size=args.batch_size
    )
    
    # Load test data for examples
    with open(f'{args.data_dir}/test_data.pkl', 'rb') as f:
        test_data = pickle.load(f)
    src_data = [doc for doc, _ in test_data]
    
    # Models to evaluate
    models_to_eval = ['rnn', 'lstm', 'attention'] if args.model == 'all' else [args.model]
    
    all_results = {}
    
    for model_type in models_to_eval:
        print(f'\n{"="*60}')
        print(f'Evaluating {model_type.upper()} model')
        print(f'{"="*60}')
        
        # Create model
        if model_type == 'rnn':
            model = create_rnn_seq2seq(
                src_vocab_size=len(src_vocab),
                tgt_vocab_size=len(tgt_vocab),
                embed_dim=args.embed_dim,
                hidden_dim=args.hidden_dim,
                num_layers=args.num_layers,
                dropout=args.dropout,
                device=device
            )
            model_name = 'rnn_seq2seq'
        elif model_type == 'lstm':
            model = create_lstm_seq2seq(
                src_vocab_size=len(src_vocab),
                tgt_vocab_size=len(tgt_vocab),
                embed_dim=args.embed_dim,
                hidden_dim=args.hidden_dim,
                num_layers=args.num_layers,
                dropout=args.dropout,
                device=device
            )
            model_name = 'lstm_seq2seq'
        else:  # attention
            model = create_lstm_attention_seq2seq(
                src_vocab_size=len(src_vocab),
                tgt_vocab_size=len(tgt_vocab),
                embed_dim=args.embed_dim,
                hidden_dim=args.hidden_dim,
                num_layers=args.num_layers,
                dropout=args.dropout,
                device=device
            )
            model_name = 'lstm_attention_seq2seq'
        
        # Load best checkpoint
        checkpoint_path = os.path.join(args.checkpoint_dir, f'{model_name}_best.pt')
        if not os.path.exists(checkpoint_path):
            print(f'Checkpoint not found: {checkpoint_path}')
            continue
        
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f'Loaded checkpoint from {checkpoint_path}')
        
        # Evaluate
        results, pred_tokens, ref_tokens = evaluate_model(
            model, test_loader, device, tgt_vocab, model_name, model_type
        )
        
        all_results[model_name] = results
        
        # Save examples
        examples_path = os.path.join(args.results_dir, f'{model_name}_examples.txt')
        save_examples(pred_tokens, ref_tokens, src_data, examples_path, num_examples=20)
        print(f'Saved examples to {examples_path}')
    
    # Save all results
    results_path = os.path.join(args.results_dir, 'evaluation_results.json')
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f'\nEvaluation complete! Results saved to {results_path}')
    
    # Print comparison
    print('\n' + '='*60)
    print('EVALUATION SUMMARY')
    print('='*60)
    print(f'{"Model":<30} {"Token Acc":<12} {"Exact Match":<12} {"BLEU-4":<12}')
    print('-'*60)
    for model_name, results in all_results.items():
        print(f'{model_name:<30} {results["token_accuracy"]:<12.4f} '
              f'{results["exact_match"]:<12.4f} {results["bleu_scores"]["bleu_4"]:<12.4f}')


if __name__ == '__main__':
    main()
