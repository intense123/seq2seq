"""
Attention visualization script for LSTM with Attention model.
Creates heatmaps showing attention weights between source and target tokens.
"""

import os
import argparse
import pickle
from typing import List, Tuple

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from data import create_dataloaders, SOS_IDX, EOS_IDX, Vocabulary
from models import create_lstm_attention_seq2seq


def visualize_attention(
    src_tokens: List[str],
    tgt_tokens: List[str],
    attention_weights: np.ndarray,
    save_path: str,
    title: str = 'Attention Heatmap'
):
    """
    Create and save attention heatmap.
    
    Args:
        src_tokens: Source tokens (docstring)
        tgt_tokens: Target tokens (generated code)
        attention_weights: Attention weights matrix [tgt_len, src_len]
        save_path: Path to save the figure
        title: Title for the plot
    """
    # Create figure
    fig, ax = plt.subplots(figsize=(max(12, len(src_tokens) * 0.5), max(8, len(tgt_tokens) * 0.4)))
    
    # Create heatmap
    sns.heatmap(
        attention_weights,
        xticklabels=src_tokens,
        yticklabels=tgt_tokens,
        cmap='YlOrRd',
        cbar=True,
        square=False,
        linewidths=0.5,
        linecolor='gray',
        ax=ax,
        vmin=0,
        vmax=1
    )
    
    # Set labels
    ax.set_xlabel('Source (Docstring)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Target (Generated Code)', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.yticks(rotation=0, fontsize=10)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f'Saved attention heatmap to {save_path}')


def decode_sequence(indices: List[int], vocab: Vocabulary, include_special: bool = False) -> List[str]:
    """
    Decode a sequence of indices to tokens.
    
    Args:
        indices: List of token indices
        vocab: Vocabulary object
        include_special: Whether to include special tokens
    
    Returns:
        List of token strings
    """
    tokens = []
    for idx in indices:
        if idx == EOS_IDX:
            if include_special:
                tokens.append('<eos>')
            break
        if idx == SOS_IDX:
            if include_special:
                tokens.append('<sos>')
            continue
        token = vocab.idx2token.get(idx, '<unk>')
        tokens.append(token)
    return tokens


def generate_with_attention(
    model,
    src: torch.Tensor,
    src_lengths: torch.Tensor,
    src_vocab: Vocabulary,
    tgt_vocab: Vocabulary,
    device: torch.device,
    max_len: int = 100
) -> Tuple[List[str], List[str], np.ndarray]:
    """
    Generate code with attention weights for a single example.
    
    Args:
        model: LSTM Attention Seq2Seq model
        src: Source sequence [1, src_len]
        src_lengths: Source sequence length [1]
        src_vocab: Source vocabulary
        tgt_vocab: Target vocabulary
        device: Device to run on
        max_len: Maximum generation length
    
    Returns:
        src_tokens: Source tokens
        tgt_tokens: Generated target tokens
        attention_weights: Attention weights [tgt_len, src_len]
    """
    model.eval()
    
    with torch.no_grad():
        src = src.to(device)
        src_lengths = src_lengths.to(device)
        
        # Generate with attention
        generated, attention_weights = model.generate(
            src, src_lengths, max_len, SOS_IDX, EOS_IDX
        )
        
        # Convert to numpy
        generated = generated.cpu().numpy()[0]
        attention_weights = attention_weights.cpu().numpy()[0]
        src = src.cpu().numpy()[0]
        
        # Decode sequences
        src_tokens = decode_sequence(src.tolist(), src_vocab)
        tgt_tokens = decode_sequence(generated.tolist(), tgt_vocab)
        
        # Trim attention weights to match actual lengths
        attention_weights = attention_weights[:len(tgt_tokens), :len(src_tokens)]
    
    return src_tokens, tgt_tokens, attention_weights


def analyze_attention_patterns(
    src_tokens: List[str],
    tgt_tokens: List[str],
    attention_weights: np.ndarray
) -> str:
    """
    Analyze attention patterns and provide interpretation.
    
    Args:
        src_tokens: Source tokens
        tgt_tokens: Target tokens
        attention_weights: Attention weights
    
    Returns:
        Analysis text
    """
    analysis = []
    analysis.append("Attention Pattern Analysis:")
    analysis.append("-" * 60)
    
    # Find most attended source tokens for each target token
    for i, tgt_token in enumerate(tgt_tokens[:10]):  # Limit to first 10 for readability
        if i >= len(attention_weights):
            break
        
        top_k = min(3, len(src_tokens))
        top_indices = np.argsort(attention_weights[i])[-top_k:][::-1]
        top_weights = attention_weights[i][top_indices]
        top_src_tokens = [src_tokens[idx] for idx in top_indices]
        
        analysis.append(f"\nTarget token '{tgt_token}' attends most to:")
        for src_token, weight in zip(top_src_tokens, top_weights):
            analysis.append(f"  - '{src_token}': {weight:.3f}")
    
    # Find most important source tokens overall
    avg_attention = np.mean(attention_weights, axis=0)
    top_k = min(5, len(src_tokens))
    top_src_indices = np.argsort(avg_attention)[-top_k:][::-1]
    
    analysis.append(f"\n{'-'*60}")
    analysis.append("Most important source tokens overall:")
    for idx in top_src_indices:
        analysis.append(f"  - '{src_tokens[idx]}': {avg_attention[idx]:.3f}")
    
    return '\n'.join(analysis)


def main():
    parser = argparse.ArgumentParser(description='Visualize attention for LSTM Attention model')
    parser.add_argument('--data_dir', type=str, default='./data',
                        help='Directory containing preprocessed data')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints',
                        help='Directory containing model checkpoints')
    parser.add_argument('--results_dir', type=str, default='./results',
                        help='Directory to save visualizations')
    parser.add_argument('--embed_dim', type=int, default=256,
                        help='Embedding dimension')
    parser.add_argument('--hidden_dim', type=int, default=256,
                        help='Hidden dimension')
    parser.add_argument('--num_layers', type=int, default=1,
                        help='Number of layers')
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='Dropout')
    parser.add_argument('--num_examples', type=int, default=5,
                        help='Number of examples to visualize')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(os.path.join(args.results_dir, 'plots'), exist_ok=True)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Load data
    print('Loading data...')
    _, _, test_loader, src_vocab, tgt_vocab = create_dataloaders(
        args.data_dir,
        batch_size=1  # Process one at a time for visualization
    )
    
    # Load test data
    with open(f'{args.data_dir}/test_data.pkl', 'rb') as f:
        test_data = pickle.load(f)
    
    # Create model
    print('Creating model...')
    model = create_lstm_attention_seq2seq(
        src_vocab_size=len(src_vocab),
        tgt_vocab_size=len(tgt_vocab),
        embed_dim=args.embed_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        device=device
    )
    
    # Load checkpoint
    checkpoint_path = os.path.join(args.checkpoint_dir, 'lstm_attention_seq2seq_best.pt')
    if not os.path.exists(checkpoint_path):
        print(f'Checkpoint not found: {checkpoint_path}')
        return
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f'Loaded checkpoint from {checkpoint_path}')
    
    # Select diverse examples for visualization
    # We'll try to get examples of different lengths
    selected_indices = []
    
    # Sort by docstring length
    sorted_test_data = sorted(enumerate(test_data), key=lambda x: len(x[1][0].split()))
    
    # Select examples from different length ranges
    step = len(sorted_test_data) // (args.num_examples + 1)
    for i in range(1, args.num_examples + 1):
        idx = min(i * step, len(sorted_test_data) - 1)
        selected_indices.append(sorted_test_data[idx][0])
    
    print(f'\nGenerating attention visualizations for {len(selected_indices)} examples...')
    
    # Create analysis file
    analysis_path = os.path.join(args.results_dir, 'attention_analysis.txt')
    with open(analysis_path, 'w') as analysis_file:
        analysis_file.write("ATTENTION ANALYSIS\n")
        analysis_file.write("=" * 80 + "\n\n")
        
        # Visualize attention for selected examples
        for example_num, test_idx in enumerate(selected_indices, 1):
            print(f'\nProcessing example {example_num}/{len(selected_indices)}...')
            
            # Get source and target from test data
            docstring, reference_code = test_data[test_idx]
            
            # Get batch from dataloader
            # We need to manually create the batch for the specific index
            src_batch = []
            src_lengths_list = []
            
            for i, (src, tgt, src_len, tgt_len) in enumerate(test_loader):
                if i == test_idx:
                    src_batch = src
                    src_lengths_list = src_len
                    break
            
            if not isinstance(src_batch, torch.Tensor):
                print(f'Could not find example {test_idx}')
                continue
            
            # Generate with attention
            src_tokens, tgt_tokens, attention_weights = generate_with_attention(
                model, src_batch, src_lengths_list, src_vocab, tgt_vocab, device
            )
            
            # Create visualization
            title = f'Example {example_num}: Attention Heatmap'
            save_path = os.path.join(
                args.results_dir, 'plots', f'attention_example_{example_num}.png'
            )
            visualize_attention(src_tokens, tgt_tokens, attention_weights, save_path, title)
            
            # Write analysis to file
            analysis_file.write(f"EXAMPLE {example_num}\n")
            analysis_file.write("=" * 80 + "\n\n")
            analysis_file.write(f"Docstring: {docstring}\n\n")
            analysis_file.write(f"Reference Code: {reference_code}\n\n")
            analysis_file.write(f"Generated Code: {' '.join(tgt_tokens)}\n\n")
            analysis_file.write(f"Source tokens: {src_tokens}\n\n")
            analysis_file.write(f"Target tokens: {tgt_tokens}\n\n")
            
            # Analyze attention patterns
            pattern_analysis = analyze_attention_patterns(src_tokens, tgt_tokens, attention_weights)
            analysis_file.write(pattern_analysis)
            analysis_file.write("\n\n" + "=" * 80 + "\n\n")
    
    print(f'\nAttention visualization complete!')
    print(f'Heatmaps saved to {os.path.join(args.results_dir, "plots")}')
    print(f'Analysis saved to {analysis_path}')
    
    # Print summary
    print('\n' + '='*60)
    print('ATTENTION VISUALIZATION SUMMARY')
    print('='*60)
    print(f'Number of examples visualized: {len(selected_indices)}')
    print(f'Output directory: {os.path.join(args.results_dir, "plots")}')
    print('\nKey insights:')
    print('- Check the heatmaps to see which source tokens the model attends to')
    print('- Bright colors indicate strong attention')
    print('- Look for semantic alignment between source and target tokens')


if __name__ == '__main__':
    main()
