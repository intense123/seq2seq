"""
PyTorch Dataset and DataLoader utilities for Seq2Seq training.
"""

import pickle
from typing import List, Tuple
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

from .preprocess import Tokenizer, Vocabulary, PAD_IDX, SOS_IDX, EOS_IDX


class CodeGenerationDataset(Dataset):
    """Dataset for text-to-code generation."""
    
    def __init__(
        self,
        data: List[Tuple[str, str]],
        src_vocab: Vocabulary,
        tgt_vocab: Vocabulary
    ):
        """
        Args:
            data: List of (docstring, code) tuples
            src_vocab: Source vocabulary
            tgt_vocab: Target vocabulary
        """
        self.data = data
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        
        # Initialize tokenizers
        self.src_tokenizer = Tokenizer(lowercase=True)
        self.tgt_tokenizer = Tokenizer(lowercase=False)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        docstring, code = self.data[idx]
        
        # Tokenize
        src_tokens = self.src_tokenizer.tokenize(docstring)
        tgt_tokens = self.tgt_tokenizer.tokenize(code)
        
        # Convert to indices
        src_indices = self.src_vocab.encode(src_tokens)
        tgt_indices = self.tgt_vocab.encode(tgt_tokens)
        
        # Add SOS and EOS tokens to target
        tgt_indices = [SOS_IDX] + tgt_indices + [EOS_IDX]
        
        return torch.tensor(src_indices), torch.tensor(tgt_indices)


def collate_fn(batch):
    """
    Collate function to pad sequences in a batch.
    
    Args:
        batch: List of (src_tensor, tgt_tensor) tuples
    
    Returns:
        src_padded: Padded source sequences [batch_size, max_src_len]
        tgt_padded: Padded target sequences [batch_size, max_tgt_len]
        src_lengths: Original source sequence lengths
        tgt_lengths: Original target sequence lengths
    """
    src_batch, tgt_batch = zip(*batch)
    
    # Get original lengths
    src_lengths = torch.tensor([len(s) for s in src_batch])
    tgt_lengths = torch.tensor([len(t) for t in tgt_batch])
    
    # Pad sequences (pad_sequence expects [seq_len, batch_size])
    src_padded = pad_sequence(src_batch, batch_first=True, padding_value=PAD_IDX)
    tgt_padded = pad_sequence(tgt_batch, batch_first=True, padding_value=PAD_IDX)
    
    return src_padded, tgt_padded, src_lengths, tgt_lengths


def create_dataloaders(
    data_dir: str,
    batch_size: int = 32,
    num_workers: int = 0
) -> Tuple[DataLoader, DataLoader, DataLoader, Vocabulary, Vocabulary]:
    """
    Create train, validation, and test dataloaders.
    
    Args:
        data_dir: Directory containing preprocessed data
        batch_size: Batch size for dataloaders
        num_workers: Number of workers for dataloaders
    
    Returns:
        train_loader, val_loader, test_loader, src_vocab, tgt_vocab
    """
    # Load vocabularies
    src_vocab = Vocabulary.load(f'{data_dir}/src_vocab.pkl')
    tgt_vocab = Vocabulary.load(f'{data_dir}/tgt_vocab.pkl')
    
    # Load data splits
    with open(f'{data_dir}/train_data.pkl', 'rb') as f:
        train_data = pickle.load(f)
    
    with open(f'{data_dir}/val_data.pkl', 'rb') as f:
        val_data = pickle.load(f)
    
    with open(f'{data_dir}/test_data.pkl', 'rb') as f:
        test_data = pickle.load(f)
    
    # Create datasets
    train_dataset = CodeGenerationDataset(train_data, src_vocab, tgt_vocab)
    val_dataset = CodeGenerationDataset(val_data, src_vocab, tgt_vocab)
    test_dataset = CodeGenerationDataset(test_data, src_vocab, tgt_vocab)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=num_workers
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=num_workers
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=num_workers
    )
    
    return train_loader, val_loader, test_loader, src_vocab, tgt_vocab
