"""
Data preprocessing module for CodeSearchNet Python dataset.
Handles loading, filtering, tokenization, and vocabulary building.
"""

import os
import json
import pickle
from typing import List, Tuple, Dict
from collections import Counter
from datasets import load_dataset
from tqdm import tqdm


# Special tokens
PAD_TOKEN = '<pad>'
SOS_TOKEN = '<sos>'
EOS_TOKEN = '<eos>'
UNK_TOKEN = '<unk>'

# Special token indices
PAD_IDX = 0
SOS_IDX = 1
EOS_IDX = 2
UNK_IDX = 3


class Tokenizer:
    """Simple whitespace tokenizer."""
    
    def __init__(self, lowercase: bool = False):
        self.lowercase = lowercase
    
    def tokenize(self, text: str) -> List[str]:
        """Tokenize text by whitespace."""
        if self.lowercase:
            text = text.lower()
        return text.split()


class Vocabulary:
    """Vocabulary for source or target sequences."""
    
    def __init__(self, min_freq: int = 2):
        self.min_freq = min_freq
        self.token2idx = {
            PAD_TOKEN: PAD_IDX,
            SOS_TOKEN: SOS_IDX,
            EOS_TOKEN: EOS_IDX,
            UNK_TOKEN: UNK_IDX
        }
        self.idx2token = {
            PAD_IDX: PAD_TOKEN,
            SOS_IDX: SOS_TOKEN,
            EOS_IDX: EOS_TOKEN,
            UNK_IDX: UNK_TOKEN
        }
        self.token_counts = Counter()
    
    def build(self, tokenized_texts: List[List[str]]):
        """Build vocabulary from tokenized texts."""
        # Count all tokens
        for tokens in tokenized_texts:
            self.token_counts.update(tokens)
        
        # Add tokens that meet minimum frequency
        idx = len(self.token2idx)
        for token, count in self.token_counts.items():
            if count >= self.min_freq and token not in self.token2idx:
                self.token2idx[token] = idx
                self.idx2token[idx] = token
                idx += 1
        
        print(f"Vocabulary size: {len(self.token2idx)} (min_freq={self.min_freq})")
    
    def encode(self, tokens: List[str]) -> List[int]:
        """Convert tokens to indices."""
        return [self.token2idx.get(token, UNK_IDX) for token in tokens]
    
    def decode(self, indices: List[int]) -> List[str]:
        """Convert indices to tokens."""
        return [self.idx2token.get(idx, UNK_TOKEN) for idx in indices]
    
    def __len__(self):
        return len(self.token2idx)
    
    def save(self, path: str):
        """Save vocabulary to file."""
        with open(path, 'wb') as f:
            pickle.dump({
                'token2idx': self.token2idx,
                'idx2token': self.idx2token,
                'token_counts': self.token_counts,
                'min_freq': self.min_freq
            }, f)
    
    @classmethod
    def load(cls, path: str):
        """Load vocabulary from file."""
        with open(path, 'rb') as f:
            data = pickle.load(f)
        
        vocab = cls(min_freq=data['min_freq'])
        vocab.token2idx = data['token2idx']
        vocab.idx2token = data['idx2token']
        vocab.token_counts = data['token_counts']
        return vocab


def load_and_filter_dataset(
    max_docstring_len: int = 50,
    max_code_len: int = 80,
    train_size: int = 8000,
    val_size: int = 1000,
    test_size: int = 1000,
    cache_dir: str = './data/cache'
) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]], List[Tuple[str, str]]]:
    """
    Load CodeSearchNet Python dataset and filter by length constraints.
    
    Args:
        max_docstring_len: Maximum number of tokens in docstring
        max_code_len: Maximum number of tokens in code
        train_size: Number of training samples
        val_size: Number of validation samples
        test_size: Number of test samples
        cache_dir: Directory to cache the dataset
    
    Returns:
        Tuple of (train_data, val_data, test_data)
        Each is a list of (docstring, code) tuples
    """
    print("Loading CodeSearchNet Python dataset...")
    
    # Load dataset
    dataset = load_dataset(
        "Nan-Do/code-search-net-python",
        split="train",
        cache_dir=cache_dir
    )
    
    print(f"Total samples in dataset: {len(dataset)}")
    
    # Initialize tokenizers
    docstring_tokenizer = Tokenizer(lowercase=True)
    code_tokenizer = Tokenizer(lowercase=False)
    
    # Filter valid samples
    valid_samples = []
    print("Filtering samples by length constraints...")
    
    for sample in tqdm(dataset):
        docstring = sample.get('docstring', '').strip()
        code = sample.get('code', '').strip()
        
        # Skip empty samples
        if not docstring or not code:
            continue
        
        # Tokenize
        doc_tokens = docstring_tokenizer.tokenize(docstring)
        code_tokens = code_tokenizer.tokenize(code)
        
        # Check length constraints
        if len(doc_tokens) <= max_docstring_len and len(code_tokens) <= max_code_len:
            valid_samples.append((docstring, code))
        
        # Stop if we have enough samples
        if len(valid_samples) >= train_size + val_size + test_size:
            break
    
    print(f"Valid samples after filtering: {len(valid_samples)}")
    
    # Split into train/val/test
    train_data = valid_samples[:train_size]
    val_data = valid_samples[train_size:train_size + val_size]
    test_data = valid_samples[train_size + val_size:train_size + val_size + test_size]
    
    print(f"Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")
    
    return train_data, val_data, test_data


def build_vocabularies(
    train_data: List[Tuple[str, str]],
    min_freq: int = 2
) -> Tuple[Vocabulary, Vocabulary]:
    """
    Build source (docstring) and target (code) vocabularies from training data.
    
    Args:
        train_data: List of (docstring, code) tuples
        min_freq: Minimum frequency for a token to be included
    
    Returns:
        Tuple of (src_vocab, tgt_vocab)
    """
    print("Building vocabularies...")
    
    # Initialize tokenizers
    docstring_tokenizer = Tokenizer(lowercase=True)
    code_tokenizer = Tokenizer(lowercase=False)
    
    # Tokenize all training data
    src_tokenized = []
    tgt_tokenized = []
    
    for docstring, code in tqdm(train_data):
        src_tokenized.append(docstring_tokenizer.tokenize(docstring))
        tgt_tokenized.append(code_tokenizer.tokenize(code))
    
    # Build vocabularies
    src_vocab = Vocabulary(min_freq=min_freq)
    tgt_vocab = Vocabulary(min_freq=min_freq)
    
    src_vocab.build(src_tokenized)
    tgt_vocab.build(tgt_tokenized)
    
    return src_vocab, tgt_vocab


def prepare_data(
    output_dir: str = './data',
    max_docstring_len: int = 50,
    max_code_len: int = 80,
    train_size: int = 8000,
    val_size: int = 1000,
    test_size: int = 1000,
    min_freq: int = 2
):
    """
    Main function to prepare and save all data and vocabularies.
    
    Args:
        output_dir: Directory to save processed data
        max_docstring_len: Maximum docstring length
        max_code_len: Maximum code length
        train_size: Number of training samples
        val_size: Number of validation samples
        test_size: Number of test samples
        min_freq: Minimum token frequency for vocabulary
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Load and filter dataset
    train_data, val_data, test_data = load_and_filter_dataset(
        max_docstring_len=max_docstring_len,
        max_code_len=max_code_len,
        train_size=train_size,
        val_size=val_size,
        test_size=test_size,
        cache_dir=os.path.join(output_dir, 'cache')
    )
    
    # Build vocabularies from training data only
    src_vocab, tgt_vocab = build_vocabularies(train_data, min_freq=min_freq)
    
    # Save data splits
    print("Saving data splits...")
    with open(os.path.join(output_dir, 'train_data.pkl'), 'wb') as f:
        pickle.dump(train_data, f)
    
    with open(os.path.join(output_dir, 'val_data.pkl'), 'wb') as f:
        pickle.dump(val_data, f)
    
    with open(os.path.join(output_dir, 'test_data.pkl'), 'wb') as f:
        pickle.dump(test_data, f)
    
    # Save vocabularies
    print("Saving vocabularies...")
    src_vocab.save(os.path.join(output_dir, 'src_vocab.pkl'))
    tgt_vocab.save(os.path.join(output_dir, 'tgt_vocab.pkl'))
    
    # Save metadata
    metadata = {
        'max_docstring_len': max_docstring_len,
        'max_code_len': max_code_len,
        'train_size': len(train_data),
        'val_size': len(val_data),
        'test_size': len(test_data),
        'src_vocab_size': len(src_vocab),
        'tgt_vocab_size': len(tgt_vocab),
        'min_freq': min_freq
    }
    
    with open(os.path.join(output_dir, 'metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print("Data preparation complete!")
    print(f"Metadata: {metadata}")


if __name__ == '__main__':
    prepare_data()
