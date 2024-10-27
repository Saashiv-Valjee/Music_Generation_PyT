# src/model.py

import torch
import torch.nn as nn
import math

class MusicTransformer(nn.Module):
    def __init__(self, vocab_size, embedding_dim=128, num_heads=4, num_layers=2, dim_feedforward=512, dropout=0.1, max_len=5000):
        """
        Simplified Transformer model for music generation.

        Args:
            vocab_size (int): Size of the vocabulary (number of unique tokens).
            embedding_dim (int): Dimension of the embeddings.
            num_heads (int): Number of attention heads.
            num_layers (int): Number of Transformer Encoder layers.
            dim_feedforward (int): Dimension of the feedforward network.
            dropout (float): Dropout probability.
            max_len (int): Maximum length of the input sequences.
        """
        super(MusicTransformer, self).__init__()

        self.embedding_dim = embedding_dim
        self.vocab_size = vocab_size
        self.max_len = max_len

        # Token embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        # Positional encoding
        self.positional_encoding = PositionalEncoding(embedding_dim, dropout, max_len)

        # Transformer Encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Output linear layer to map to vocabulary size
        self.fc_out = nn.Linear(embedding_dim, vocab_size)

        # Initialize weights
        self._reset_parameters()

    def _reset_parameters(self):
        """
        Initialize model parameters.
        """
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, src_mask=None):
        """
        Forward pass of the Simple Transformer.

        Args:
            src (Tensor): Input tensor of shape (batch_size, seq_length).
            src_mask (Tensor, optional): Source mask tensor.

        Returns:
            Tensor: Output logits of shape (batch_size, seq_length, vocab_size).
        """
        # Embedding layer: Convert token indices to embeddings
        src = self.embedding(src) * math.sqrt(self.embedding_dim)  # Scaling factor
        # src shape: (batch_size, seq_length, embedding_dim)

        # Add positional encoding
        src = self.positional_encoding(src)

        # Transpose for transformer encoder (required shape: seq_length, batch_size, embedding_dim)
        src = src.transpose(0, 1)

        # Pass through Transformer Encoder
        output = self.transformer_encoder(src, src_key_padding_mask=src_mask)
        # Output shape: (seq_length, batch_size, embedding_dim)

        # Transpose back
        output = output.transpose(0, 1)
        # Output shape: (batch_size, seq_length, embedding_dim)

        # Output linear layer to get logits over vocabulary
        output = self.fc_out(output)  # (batch_size, seq_length, vocab_size)

        return output

class PositionalEncoding(nn.Module):
    def __init__(self, embedding_dim, dropout=0.1, max_len=5000):
        """
        Positional Encoding module.

        Args:
            embedding_dim (int): Dimension of the embeddings.
            dropout (float): Dropout probability.
            max_len (int): Maximum length of the input sequences.
        """
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Create constant positional encoding matrix with values dependent on
        # pos and i
        pe = torch.zeros(max_len, embedding_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embedding_dim, 2).float() * (-math.log(10000.0) / embedding_dim))
        pe[:, 0::2] = torch.sin(position * div_term)  # Apply sine to even indices
        pe[:, 1::2] = torch.cos(position * div_term)  # Apply cosine to odd indices
        pe = pe.unsqueeze(0)  # Shape: (1, max_len, embedding_dim)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Forward pass for positional encoding.

        Args:
            x (Tensor): Input tensor of shape (batch_size, seq_length, embedding_dim).

        Returns:
            Tensor: Output tensor with positional encoding added.
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)
