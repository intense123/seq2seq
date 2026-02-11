"""
Model 3: LSTM with Bahdanau Attention

Architecture:
- Encoder: Bidirectional LSTM
- Decoder: LSTM with Bahdanau (additive) attention
- Dynamic context vector computed at each decoding step
- Attention weights stored for visualization
- Teacher forcing during training

Purpose: Remove the fixed-context bottleneck, improve generation quality,
and enable interpretability through attention visualization.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class BidirectionalLSTMEncoder(nn.Module):
    """Bidirectional LSTM Encoder."""
    
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
            hidden_dim: Dimension of hidden state (per direction)
            num_layers: Number of LSTM layers
            dropout: Dropout probability
        """
        super(BidirectionalLSTMEncoder, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        
        # Bidirectional LSTM layer
        self.lstm = nn.LSTM(
            embed_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        
        # Linear layers to combine bidirectional hidden and cell states
        # Bidirectional LSTM outputs 2*hidden_dim, we project back to hidden_dim
        self.fc_hidden = nn.Linear(hidden_dim * 2, hidden_dim)
        self.fc_cell = nn.Linear(hidden_dim * 2, hidden_dim)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        src: torch.Tensor,
        src_lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Args:
            src: Source sequences [batch_size, src_len]
            src_lengths: Original lengths of source sequences [batch_size]
        
        Returns:
            outputs: Encoder outputs (all hidden states) [batch_size, src_len, hidden_dim*2]
            hidden: Tuple of (hidden_state, cell_state)
                hidden_state: [num_layers, batch_size, hidden_dim]
                cell_state: [num_layers, batch_size, hidden_dim]
        """
        # Embed source sequences
        embedded = self.dropout(self.embedding(src))  # [batch_size, src_len, embed_dim]
        
        # Pack padded sequences
        packed_embedded = nn.utils.rnn.pack_padded_sequence(
            embedded,
            src_lengths.cpu(),
            batch_first=True,
            enforce_sorted=False
        )
        
        # Pass through bidirectional LSTM
        packed_outputs, (hidden, cell) = self.lstm(packed_embedded)
        
        # Unpack sequences
        outputs, _ = nn.utils.rnn.pad_packed_sequence(packed_outputs, batch_first=True)
        # outputs: [batch_size, src_len, hidden_dim*2]
        
        # hidden and cell: [num_layers*2, batch_size, hidden_dim]
        # Combine forward and backward directions
        # hidden[-2, :, :] is the last forward hidden state
        # hidden[-1, :, :] is the last backward hidden state
        
        # For multi-layer, we need to handle each layer
        hidden_forward = hidden[0:hidden.size(0):2]  # [num_layers, batch_size, hidden_dim]
        hidden_backward = hidden[1:hidden.size(0):2]  # [num_layers, batch_size, hidden_dim]
        
        cell_forward = cell[0:cell.size(0):2]  # [num_layers, batch_size, hidden_dim]
        cell_backward = cell[1:cell.size(0):2]  # [num_layers, batch_size, hidden_dim]
        
        # Concatenate and project to hidden_dim
        hidden_combined = torch.cat([hidden_forward, hidden_backward], dim=2)  # [num_layers, batch_size, hidden_dim*2]
        cell_combined = torch.cat([cell_forward, cell_backward], dim=2)  # [num_layers, batch_size, hidden_dim*2]
        
        # Project back to hidden_dim
        hidden = torch.tanh(self.fc_hidden(hidden_combined))  # [num_layers, batch_size, hidden_dim]
        cell = torch.tanh(self.fc_cell(cell_combined))  # [num_layers, batch_size, hidden_dim]
        
        return outputs, (hidden, cell)


class BahdanauAttention(nn.Module):
    """Bahdanau (additive) attention mechanism."""
    
    def __init__(self, hidden_dim: int, encoder_dim: int):
        """
        Args:
            hidden_dim: Dimension of decoder hidden state
            encoder_dim: Dimension of encoder outputs (hidden_dim*2 for bidirectional)
        """
        super(BahdanauAttention, self).__init__()
        
        # Linear transformations for attention
        self.attn_hidden = nn.Linear(hidden_dim, hidden_dim)
        self.attn_encoder = nn.Linear(encoder_dim, hidden_dim)
        self.attn_combine = nn.Linear(hidden_dim, 1)
    
    def forward(
        self,
        decoder_hidden: torch.Tensor,
        encoder_outputs: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute attention weights and context vector.
        
        Args:
            decoder_hidden: Current decoder hidden state [batch_size, hidden_dim]
            encoder_outputs: All encoder outputs [batch_size, src_len, encoder_dim]
            mask: Mask for padded positions [batch_size, src_len]
        
        Returns:
            context: Context vector [batch_size, encoder_dim]
            attention_weights: Attention weights [batch_size, src_len]
        """
        batch_size = encoder_outputs.shape[0]
        src_len = encoder_outputs.shape[1]
        
        # Transform decoder hidden state: [batch_size, hidden_dim]
        decoder_hidden = self.attn_hidden(decoder_hidden)  # [batch_size, hidden_dim]
        
        # Transform encoder outputs: [batch_size, src_len, hidden_dim]
        encoder_outputs_transformed = self.attn_encoder(encoder_outputs)
        
        # Expand decoder hidden to match encoder outputs
        # [batch_size, 1, hidden_dim] -> [batch_size, src_len, hidden_dim]
        decoder_hidden = decoder_hidden.unsqueeze(1).expand(-1, src_len, -1)
        
        # Compute attention scores using tanh(Wa*hidden + Ua*encoder_output)
        energy = torch.tanh(decoder_hidden + encoder_outputs_transformed)  # [batch_size, src_len, hidden_dim]
        
        # Project to scalar scores
        attention_scores = self.attn_combine(energy).squeeze(2)  # [batch_size, src_len]
        
        # Apply mask if provided (set padded positions to large negative value)
        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == 0, -1e10)
        
        # Compute attention weights
        attention_weights = F.softmax(attention_scores, dim=1)  # [batch_size, src_len]
        
        # Compute context vector as weighted sum of encoder outputs
        # [batch_size, 1, src_len] @ [batch_size, src_len, encoder_dim] -> [batch_size, 1, encoder_dim]
        context = torch.bmm(attention_weights.unsqueeze(1), encoder_outputs)
        context = context.squeeze(1)  # [batch_size, encoder_dim]
        
        return context, attention_weights


class AttentionLSTMDecoder(nn.Module):
    """LSTM Decoder with Bahdanau Attention."""
    
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        hidden_dim: int,
        encoder_dim: int,
        num_layers: int = 1,
        dropout: float = 0.1
    ):
        """
        Args:
            vocab_size: Size of target vocabulary
            embed_dim: Dimension of embeddings
            hidden_dim: Dimension of hidden state
            encoder_dim: Dimension of encoder outputs (hidden_dim*2 for bidirectional)
            num_layers: Number of LSTM layers
            dropout: Dropout probability
        """
        super(AttentionLSTMDecoder, self).__init__()
        
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        
        # Attention mechanism
        self.attention = BahdanauAttention(hidden_dim, encoder_dim)
        
        # LSTM layer (input is embedding + context vector)
        self.lstm = nn.LSTM(
            embed_dim + encoder_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Output projection layer
        self.fc_out = nn.Linear(hidden_dim + encoder_dim + embed_dim, vocab_size)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        input: torch.Tensor,
        hidden: Tuple[torch.Tensor, torch.Tensor],
        encoder_outputs: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        """
        Forward pass for one decoding step with attention.
        
        Args:
            input: Input token [batch_size, 1]
            hidden: Tuple of (hidden_state, cell_state)
                hidden_state: [num_layers, batch_size, hidden_dim]
                cell_state: [num_layers, batch_size, hidden_dim]
            encoder_outputs: All encoder outputs [batch_size, src_len, encoder_dim]
            mask: Mask for padded positions [batch_size, src_len]
        
        Returns:
            output: Output logits [batch_size, vocab_size]
            hidden: New (hidden_state, cell_state) tuple
            attention_weights: Attention weights [batch_size, src_len]
        """
        # Embed input token
        embedded = self.dropout(self.embedding(input))  # [batch_size, 1, embed_dim]
        
        # Compute attention using the top layer's hidden state
        hidden_state, cell_state = hidden
        top_hidden = hidden_state[-1]  # [batch_size, hidden_dim]
        
        context, attention_weights = self.attention(top_hidden, encoder_outputs, mask)
        # context: [batch_size, encoder_dim]
        # attention_weights: [batch_size, src_len]
        
        # Concatenate embedded input and context vector
        lstm_input = torch.cat([embedded, context.unsqueeze(1)], dim=2)
        # lstm_input: [batch_size, 1, embed_dim + encoder_dim]
        
        # Pass through LSTM
        lstm_output, (hidden_state, cell_state) = self.lstm(lstm_input, (hidden_state, cell_state))
        # lstm_output: [batch_size, 1, hidden_dim]
        
        # Concatenate LSTM output, context, and embedded input for final prediction
        lstm_output = lstm_output.squeeze(1)  # [batch_size, hidden_dim]
        embedded = embedded.squeeze(1)  # [batch_size, embed_dim]
        
        output_input = torch.cat([lstm_output, context, embedded], dim=1)
        # output_input: [batch_size, hidden_dim + encoder_dim + embed_dim]
        
        # Project to vocabulary size
        output = self.fc_out(output_input)  # [batch_size, vocab_size]
        
        return output, (hidden_state, cell_state), attention_weights


class LSTMAttentionSeq2Seq(nn.Module):
    """LSTM Seq2Seq model with Bahdanau Attention."""
    
    def __init__(
        self,
        encoder: BidirectionalLSTMEncoder,
        decoder: AttentionLSTMDecoder,
        device: torch.device
    ):
        """
        Args:
            encoder: Bidirectional LSTM encoder
            decoder: LSTM decoder with attention
            device: Device to run model on
        """
        super(LSTMAttentionSeq2Seq, self).__init__()
        
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
    
    def create_mask(self, src: torch.Tensor, src_lengths: torch.Tensor) -> torch.Tensor:
        """
        Create mask for padded positions.
        
        Args:
            src: Source sequences [batch_size, src_len]
            src_lengths: Source sequence lengths [batch_size]
        
        Returns:
            mask: [batch_size, src_len] (1 for valid positions, 0 for padded)
        """
        batch_size = src.shape[0]
        src_len = src.shape[1]
        
        mask = torch.zeros(batch_size, src_len).to(self.device)
        for i, length in enumerate(src_lengths):
            mask[i, :length] = 1
        
        return mask
    
    def forward(
        self,
        src: torch.Tensor,
        src_lengths: torch.Tensor,
        tgt: torch.Tensor,
        teacher_forcing_ratio: float = 0.5
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with teacher forcing.
        
        Args:
            src: Source sequences [batch_size, src_len]
            src_lengths: Source sequence lengths [batch_size]
            tgt: Target sequences [batch_size, tgt_len]
            teacher_forcing_ratio: Probability of using teacher forcing
        
        Returns:
            outputs: Output logits [batch_size, tgt_len, vocab_size]
            attention_weights: Attention weights [batch_size, tgt_len, src_len]
        """
        batch_size = src.shape[0]
        tgt_len = tgt.shape[1]
        tgt_vocab_size = self.decoder.vocab_size
        src_len = src.shape[1]
        
        # Tensors to store outputs and attention weights
        outputs = torch.zeros(batch_size, tgt_len, tgt_vocab_size).to(self.device)
        attention_weights_all = torch.zeros(batch_size, tgt_len, src_len).to(self.device)
        
        # Encode source sequence
        encoder_outputs, (hidden, cell) = self.encoder(src, src_lengths)
        # encoder_outputs: [batch_size, src_len, hidden_dim*2]
        
        # Create mask for padded positions
        mask = self.create_mask(src, src_lengths)
        
        # First decoder input is <sos> token
        decoder_input = tgt[:, 0].unsqueeze(1)  # [batch_size, 1]
        
        # Decode one token at a time
        for t in range(1, tgt_len):
            # Decode step with attention
            output, (hidden, cell), attention_weights = self.decoder(
                decoder_input, (hidden, cell), encoder_outputs, mask
            )
            
            # Store output and attention weights
            outputs[:, t, :] = output
            attention_weights_all[:, t, :] = attention_weights
            
            # Decide whether to use teacher forcing
            teacher_force = torch.rand(1).item() < teacher_forcing_ratio
            
            # Get next input
            top1 = output.argmax(1).unsqueeze(1)  # [batch_size, 1]
            decoder_input = tgt[:, t].unsqueeze(1) if teacher_force else top1
        
        return outputs, attention_weights_all
    
    def generate(
        self,
        src: torch.Tensor,
        src_lengths: torch.Tensor,
        max_len: int,
        sos_idx: int,
        eos_idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate target sequence (inference mode) with attention weights.
        
        Args:
            src: Source sequence [batch_size, src_len]
            src_lengths: Source sequence lengths [batch_size]
            max_len: Maximum generation length
            sos_idx: Start-of-sequence token index
            eos_idx: End-of-sequence token index
        
        Returns:
            generated: Generated sequences [batch_size, generated_len]
            attention_weights_all: Attention weights [batch_size, generated_len, src_len]
        """
        batch_size = src.shape[0]
        src_len = src.shape[1]
        
        # Encode source
        encoder_outputs, (hidden, cell) = self.encoder(src, src_lengths)
        
        # Create mask
        mask = self.create_mask(src, src_lengths)
        
        # Initialize with <sos> token
        decoder_input = torch.full((batch_size, 1), sos_idx, dtype=torch.long).to(self.device)
        
        # Track which sequences have finished
        finished = torch.zeros(batch_size, dtype=torch.bool).to(self.device)
        
        # Store generated tokens and attention weights
        generated = [decoder_input]
        attention_weights_list = []
        
        for _ in range(max_len - 1):
            # Decode step
            output, (hidden, cell), attention_weights = self.decoder(
                decoder_input, (hidden, cell), encoder_outputs, mask
            )
            
            # Store attention weights
            attention_weights_list.append(attention_weights)
            
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
        
        # Stack attention weights
        if attention_weights_list:
            attention_weights_all = torch.stack(attention_weights_list, dim=1)  # [batch_size, generated_len-1, src_len]
        else:
            attention_weights_all = torch.zeros(batch_size, 1, src_len).to(self.device)
        
        return generated, attention_weights_all


def create_lstm_attention_seq2seq(
    src_vocab_size: int,
    tgt_vocab_size: int,
    embed_dim: int = 256,
    hidden_dim: int = 256,
    num_layers: int = 1,
    dropout: float = 0.1,
    device: torch.device = torch.device('cpu')
) -> LSTMAttentionSeq2Seq:
    """
    Factory function to create an LSTM Seq2Seq model with attention.
    
    Args:
        src_vocab_size: Source vocabulary size
        tgt_vocab_size: Target vocabulary size
        embed_dim: Embedding dimension
        hidden_dim: Hidden dimension
        num_layers: Number of LSTM layers
        dropout: Dropout probability
        device: Device to run model on
    
    Returns:
        LSTMAttentionSeq2Seq model
    """
    encoder = BidirectionalLSTMEncoder(src_vocab_size, embed_dim, hidden_dim, num_layers, dropout)
    decoder = AttentionLSTMDecoder(
        tgt_vocab_size, embed_dim, hidden_dim, hidden_dim * 2, num_layers, dropout
    )
    
    model = LSTMAttentionSeq2Seq(encoder, decoder, device).to(device)
    
    return model
