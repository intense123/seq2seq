"""
Model 1: Vanilla RNN Seq2Seq (Baseline)

Architecture:
- Encoder: Simple RNN
- Decoder: Simple RNN
- Fixed-length context vector (encoder's final hidden state)
- No attention mechanism
- Teacher forcing during training

Purpose: Establish baseline performance and demonstrate limitations
of vanilla RNNs in capturing long-term dependencies.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class RNNEncoder(nn.Module):
    """Vanilla RNN Encoder."""
    
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        hidden_dim: int,
        num_layers: int = 1,
        dropout: float = 0.1
    ):
        """
        Args:
            vocab_size: Size of source vocabulary
            embed_dim: Dimension of embeddings
            hidden_dim: Dimension of hidden state
            num_layers: Number of RNN layers
            dropout: Dropout probability
        """
        super(RNNEncoder, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        
        # RNN layer
        self.rnn = nn.RNN(
            embed_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, src: torch.Tensor, src_lengths: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            src: Source sequences [batch_size, src_len]
            src_lengths: Original lengths of source sequences [batch_size]
        
        Returns:
            outputs: All hidden states [batch_size, src_len, hidden_dim]
            hidden: Final hidden state [num_layers, batch_size, hidden_dim]
        """
        # Embed source sequences
        embedded = self.dropout(self.embedding(src))  # [batch_size, src_len, embed_dim]
        
        # Pack padded sequences for efficient RNN processing
        packed_embedded = nn.utils.rnn.pack_padded_sequence(
            embedded,
            src_lengths.cpu(),
            batch_first=True,
            enforce_sorted=False
        )
        
        # Pass through RNN
        packed_outputs, hidden = self.rnn(packed_embedded)
        
        # Unpack sequences
        outputs, _ = nn.utils.rnn.pad_packed_sequence(packed_outputs, batch_first=True)
        
        # outputs: [batch_size, src_len, hidden_dim]
        # hidden: [num_layers, batch_size, hidden_dim]
        
        return outputs, hidden


class RNNDecoder(nn.Module):
    """Vanilla RNN Decoder."""
    
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        hidden_dim: int,
        num_layers: int = 1,
        dropout: float = 0.1
    ):
        """
        Args:
            vocab_size: Size of target vocabulary
            embed_dim: Dimension of embeddings
            hidden_dim: Dimension of hidden state
            num_layers: Number of RNN layers
            dropout: Dropout probability
        """
        super(RNNDecoder, self).__init__()
        
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        
        # RNN layer
        self.rnn = nn.RNN(
            embed_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Output projection layer
        self.fc_out = nn.Linear(hidden_dim, vocab_size)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        input: torch.Tensor,
        hidden: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for one decoding step.
        
        Args:
            input: Input token [batch_size, 1]
            hidden: Previous hidden state [num_layers, batch_size, hidden_dim]
        
        Returns:
            output: Output logits [batch_size, vocab_size]
            hidden: New hidden state [num_layers, batch_size, hidden_dim]
        """
        # Embed input token
        embedded = self.dropout(self.embedding(input))  # [batch_size, 1, embed_dim]
        
        # Pass through RNN
        rnn_output, hidden = self.rnn(embedded, hidden)
        # rnn_output: [batch_size, 1, hidden_dim]
        
        # Project to vocabulary size
        output = self.fc_out(rnn_output.squeeze(1))  # [batch_size, vocab_size]
        
        return output, hidden


class RNNSeq2Seq(nn.Module):
    """Vanilla RNN Seq2Seq model."""
    
    def __init__(
        self,
        encoder: RNNEncoder,
        decoder: RNNDecoder,
        device: torch.device
    ):
        """
        Args:
            encoder: RNN encoder
            decoder: RNN decoder
            device: Device to run model on
        """
        super(RNNSeq2Seq, self).__init__()
        
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
    
    def forward(
        self,
        src: torch.Tensor,
        src_lengths: torch.Tensor,
        tgt: torch.Tensor,
        teacher_forcing_ratio: float = 0.5
    ) -> torch.Tensor:
        """
        Forward pass with teacher forcing.
        
        Args:
            src: Source sequences [batch_size, src_len]
            src_lengths: Source sequence lengths [batch_size]
            tgt: Target sequences [batch_size, tgt_len]
            teacher_forcing_ratio: Probability of using teacher forcing
        
        Returns:
            outputs: Output logits [batch_size, tgt_len, vocab_size]
        """
        batch_size = src.shape[0]
        tgt_len = tgt.shape[1]
        tgt_vocab_size = self.decoder.vocab_size
        
        # Tensor to store decoder outputs
        outputs = torch.zeros(batch_size, tgt_len, tgt_vocab_size).to(self.device)
        
        # Encode source sequence
        _, hidden = self.encoder(src, src_lengths)
        # hidden: [num_layers, batch_size, hidden_dim] - this is the context vector
        
        # First decoder input is <sos> token
        decoder_input = tgt[:, 0].unsqueeze(1)  # [batch_size, 1]
        
        # Decode one token at a time
        for t in range(1, tgt_len):
            # Decode step
            output, hidden = self.decoder(decoder_input, hidden)
            
            # Store output
            outputs[:, t, :] = output
            
            # Decide whether to use teacher forcing
            teacher_force = torch.rand(1).item() < teacher_forcing_ratio
            
            # Get next input
            top1 = output.argmax(1).unsqueeze(1)  # [batch_size, 1]
            decoder_input = tgt[:, t].unsqueeze(1) if teacher_force else top1
        
        return outputs
    
    def generate(
        self,
        src: torch.Tensor,
        src_lengths: torch.Tensor,
        max_len: int,
        sos_idx: int,
        eos_idx: int
    ) -> torch.Tensor:
        """
        Generate target sequence (inference mode).
        
        Args:
            src: Source sequence [batch_size, src_len]
            src_lengths: Source sequence lengths [batch_size]
            max_len: Maximum generation length
            sos_idx: Start-of-sequence token index
            eos_idx: End-of-sequence token index
        
        Returns:
            Generated sequences [batch_size, generated_len]
        """
        batch_size = src.shape[0]
        
        # Encode source
        _, hidden = self.encoder(src, src_lengths)
        
        # Initialize with <sos> token
        decoder_input = torch.full((batch_size, 1), sos_idx, dtype=torch.long).to(self.device)
        
        # Track which sequences have finished
        finished = torch.zeros(batch_size, dtype=torch.bool).to(self.device)
        
        # Store generated tokens
        generated = [decoder_input]
        
        for _ in range(max_len - 1):
            # Decode step
            output, hidden = self.decoder(decoder_input, hidden)
            
            # Get predicted token
            decoder_input = output.argmax(1).unsqueeze(1)  # [batch_size, 1]
            
            # Store token
            generated.append(decoder_input)
            
            # Check for <eos> token
            finished |= (decoder_input.squeeze(1) == eos_idx)
            
            # Stop if all sequences have finished
            if finished.all():
                break
        
        # Concatenate all generated tokens
        generated = torch.cat(generated, dim=1)  # [batch_size, generated_len]
        
        return generated


def create_rnn_seq2seq(
    src_vocab_size: int,
    tgt_vocab_size: int,
    embed_dim: int = 256,
    hidden_dim: int = 256,
    num_layers: int = 1,
    dropout: float = 0.1,
    device: torch.device = torch.device('cpu')
) -> RNNSeq2Seq:
    """
    Factory function to create a Vanilla RNN Seq2Seq model.
    
    Args:
        src_vocab_size: Source vocabulary size
        tgt_vocab_size: Target vocabulary size
        embed_dim: Embedding dimension
        hidden_dim: Hidden dimension
        num_layers: Number of RNN layers
        dropout: Dropout probability
        device: Device to run model on
    
    Returns:
        RNNSeq2Seq model
    """
    encoder = RNNEncoder(src_vocab_size, embed_dim, hidden_dim, num_layers, dropout)
    decoder = RNNDecoder(tgt_vocab_size, embed_dim, hidden_dim, num_layers, dropout)
    
    model = RNNSeq2Seq(encoder, decoder, device).to(device)
    
    return model
