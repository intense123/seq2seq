"""
Training script for all three Seq2Seq models.
Supports training with consistent hyperparameters across models.
"""

import os
import json
import argparse
import time
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt

from data import create_dataloaders, PAD_IDX, SOS_IDX, EOS_IDX
from models import (
    create_rnn_seq2seq,
    create_lstm_seq2seq,
    create_lstm_attention_seq2seq
)


def train_epoch(
    model,
    dataloader,
    optimizer,
    criterion,
    device,
    teacher_forcing_ratio: float = 0.5,
    clip_grad: float = 1.0,
    model_type: str = 'rnn'
) -> float:
    """
    Train model for one epoch.
    
    Args:
        model: Seq2Seq model
        dataloader: Training dataloader
        optimizer: Optimizer
        criterion: Loss function
        device: Device to train on
        teacher_forcing_ratio: Probability of teacher forcing
        clip_grad: Gradient clipping threshold
        model_type: Type of model ('rnn', 'lstm', 'attention')
    
    Returns:
        Average training loss for the epoch
    """
    model.train()
    epoch_loss = 0
    
    for src, tgt, src_lengths, tgt_lengths in tqdm(dataloader, desc="Training"):
        src = src.to(device)
        tgt = tgt.to(device)
        src_lengths = src_lengths.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        if model_type == 'attention':
            output, _ = model(src, src_lengths, tgt, teacher_forcing_ratio)
        else:
            output = model(src, src_lengths, tgt, teacher_forcing_ratio)
        
        # output: [batch_size, tgt_len, vocab_size]
        # tgt: [batch_size, tgt_len]
        
        # Reshape for loss computation
        output_dim = output.shape[-1]
        output = output[:, 1:].contiguous().view(-1, output_dim)  # [batch_size * (tgt_len-1), vocab_size]
        tgt = tgt[:, 1:].contiguous().view(-1)  # [batch_size * (tgt_len-1)]
        
        # Compute loss
        loss = criterion(output, tgt)
        
        # Backward pass
        loss.backward()
        
        # Clip gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
        
        # Update weights
        optimizer.step()
        
        epoch_loss += loss.item()
    
    return epoch_loss / len(dataloader)


def evaluate(
    model,
    dataloader,
    criterion,
    device,
    model_type: str = 'rnn'
) -> float:
    """
    Evaluate model on validation/test set.
    
    Args:
        model: Seq2Seq model
        dataloader: Evaluation dataloader
        criterion: Loss function
        device: Device to evaluate on
        model_type: Type of model ('rnn', 'lstm', 'attention')
    
    Returns:
        Average evaluation loss
    """
    model.eval()
    epoch_loss = 0
    
    with torch.no_grad():
        for src, tgt, src_lengths, tgt_lengths in tqdm(dataloader, desc="Evaluating"):
            src = src.to(device)
            tgt = tgt.to(device)
            src_lengths = src_lengths.to(device)
            
            # Forward pass (no teacher forcing during evaluation)
            if model_type == 'attention':
                output, _ = model(src, src_lengths, tgt, teacher_forcing_ratio=0.0)
            else:
                output = model(src, src_lengths, tgt, teacher_forcing_ratio=0.0)
            
            # Reshape for loss computation
            output_dim = output.shape[-1]
            output = output[:, 1:].contiguous().view(-1, output_dim)
            tgt = tgt[:, 1:].contiguous().view(-1)
            
            # Compute loss
            loss = criterion(output, tgt)
            
            epoch_loss += loss.item()
    
    return epoch_loss / len(dataloader)


def train_model(
    model,
    train_loader,
    val_loader,
    optimizer,
    criterion,
    device,
    num_epochs: int,
    checkpoint_dir: str,
    model_name: str,
    teacher_forcing_ratio: float = 0.5,
    model_type: str = 'rnn'
) -> Tuple[List[float], List[float]]:
    """
    Train model for multiple epochs and save checkpoints.
    
    Args:
        model: Seq2Seq model
        train_loader: Training dataloader
        val_loader: Validation dataloader
        optimizer: Optimizer
        criterion: Loss function
        device: Device to train on
        num_epochs: Number of epochs
        checkpoint_dir: Directory to save checkpoints
        model_name: Name of the model (for saving)
        teacher_forcing_ratio: Probability of teacher forcing
        model_type: Type of model ('rnn', 'lstm', 'attention')
    
    Returns:
        train_losses: List of training losses per epoch
        val_losses: List of validation losses per epoch
    """
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    
    print(f"\nTraining {model_name}...")
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    
    for epoch in range(num_epochs):
        start_time = time.time()
        
        # Train
        train_loss = train_epoch(
            model, train_loader, optimizer, criterion, device,
            teacher_forcing_ratio, model_type=model_type
        )
        
        # Validate
        val_loss = evaluate(model, val_loader, criterion, device, model_type=model_type)
        
        end_time = time.time()
        epoch_time = end_time - start_time
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        print(f'Epoch: {epoch+1:02} | Time: {epoch_time:.2f}s')
        print(f'\tTrain Loss: {train_loss:.4f} | Train PPL: {torch.exp(torch.tensor(train_loss)):.4f}')
        print(f'\t Val. Loss: {val_loss:.4f} |  Val. PPL: {torch.exp(torch.tensor(val_loss)):.4f}')
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint_path = os.path.join(checkpoint_dir, f'{model_name}_best.pt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
            }, checkpoint_path)
            print(f'\tSaved best model to {checkpoint_path}')
        
        # Save checkpoint every 5 epochs
        if (epoch + 1) % 5 == 0:
            checkpoint_path = os.path.join(checkpoint_dir, f'{model_name}_epoch_{epoch+1}.pt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
            }, checkpoint_path)
    
    # Save final model
    final_checkpoint_path = os.path.join(checkpoint_dir, f'{model_name}_final.pt')
    torch.save({
        'epoch': num_epochs - 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_losses': train_losses,
        'val_losses': val_losses,
    }, final_checkpoint_path)
    
    return train_losses, val_losses


def plot_losses(
    train_losses: List[float],
    val_losses: List[float],
    model_name: str,
    output_dir: str
):
    """
    Plot and save training and validation losses.
    
    Args:
        train_losses: Training losses per epoch
        val_losses: Validation losses per epoch
        model_name: Name of the model
        output_dir: Directory to save plot
    """
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Train Loss', linewidth=2)
    plt.plot(val_losses, label='Validation Loss', linewidth=2)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title(f'{model_name} - Training and Validation Loss', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plot_path = os.path.join(output_dir, f'{model_name}_loss_curve.png')
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f'Saved loss plot to {plot_path}')


def main():
    parser = argparse.ArgumentParser(description='Train Seq2Seq models for code generation')
    parser.add_argument('--model', type=str, default='all', choices=['rnn', 'lstm', 'attention', 'all'],
                        help='Model to train')
    parser.add_argument('--data_dir', type=str, default='./data',
                        help='Directory containing preprocessed data')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints',
                        help='Directory to save checkpoints')
    parser.add_argument('--results_dir', type=str, default='./results',
                        help='Directory to save results')
    parser.add_argument('--embed_dim', type=int, default=256,
                        help='Embedding dimension')
    parser.add_argument('--hidden_dim', type=int, default=256,
                        help='Hidden dimension')
    parser.add_argument('--num_layers', type=int, default=1,
                        help='Number of RNN/LSTM layers')
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='Dropout probability')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=20,
                        help='Number of epochs')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--teacher_forcing_ratio', type=float, default=0.5,
                        help='Teacher forcing ratio')
    parser.add_argument('--clip_grad', type=float, default=1.0,
                        help='Gradient clipping threshold')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    
    args = parser.parse_args()
    
    # Set random seed
    torch.manual_seed(args.seed)
    
    # Create directories
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.results_dir, exist_ok=True)
    os.makedirs(os.path.join(args.results_dir, 'plots'), exist_ok=True)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Load data
    print('Loading data...')
    train_loader, val_loader, test_loader, src_vocab, tgt_vocab = create_dataloaders(
        args.data_dir,
        batch_size=args.batch_size
    )
    
    print(f'Source vocabulary size: {len(src_vocab)}')
    print(f'Target vocabulary size: {len(tgt_vocab)}')
    print(f'Train batches: {len(train_loader)}')
    print(f'Val batches: {len(val_loader)}')
    print(f'Test batches: {len(test_loader)}')
    
    # Define loss function (ignore padding tokens)
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)
    
    # Store results for all models
    all_results = {}
    
    # Models to train
    models_to_train = ['rnn', 'lstm', 'attention'] if args.model == 'all' else [args.model]
    
    for model_type in models_to_train:
        print(f'\n{"="*60}')
        print(f'Training {model_type.upper()} model')
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
        
        # Initialize optimizer
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
        
        # Train model
        train_losses, val_losses = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            num_epochs=args.num_epochs,
            checkpoint_dir=args.checkpoint_dir,
            model_name=model_name,
            teacher_forcing_ratio=args.teacher_forcing_ratio,
            model_type=model_type
        )
        
        # Plot losses
        plot_losses(
            train_losses,
            val_losses,
            model_name,
            os.path.join(args.results_dir, 'plots')
        )
        
        # Store results
        all_results[model_name] = {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'final_train_loss': train_losses[-1],
            'final_val_loss': val_losses[-1],
            'best_val_loss': min(val_losses),
            'num_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad)
        }
    
    # Save training results
    results_path = os.path.join(args.results_dir, 'training_results.json')
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f'\nTraining complete! Results saved to {results_path}')
    
    # Print summary
    print('\n' + '='*60)
    print('TRAINING SUMMARY')
    print('='*60)
    for model_name, results in all_results.items():
        print(f'\n{model_name}:')
        print(f'  Parameters: {results["num_parameters"]:,}')
        print(f'  Best Val Loss: {results["best_val_loss"]:.4f}')
        print(f'  Final Train Loss: {results["final_train_loss"]:.4f}')
        print(f'  Final Val Loss: {results["final_val_loss"]:.4f}')


if __name__ == '__main__':
    main()
