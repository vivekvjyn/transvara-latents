import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.nn import TransformerEncoder, TransformerEncoderLayer

class PositionalEncoding(nn.Module):
    def __init__(self, emb_size: int, dropout: float = 0.0, maxlen: int = 5000):
        super(PositionalEncoding, self).__init__()
        den = torch.exp(-torch.arange(0, emb_size, 2) * math.log(10000) / emb_size)
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)
        pos_embedding = torch.zeros((maxlen, emb_size))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        pos_embedding = pos_embedding.unsqueeze(-2)

        self.dropout = nn.Dropout(dropout)
        self.register_buffer("pos_embedding", pos_embedding)

    def forward(self, token_embedding: torch.Tensor):
        return self.dropout(
            token_embedding + self.pos_embedding[: token_embedding.size(0), :]
        )

class Transformer(nn.Module):
    def __init__(
        self,
        d_in: int,
        d_model: int,
        nhead: int,
        d_hid: int,
        nlayers: int,
        num_classes: int = 7,
        seq_in_len: int = 5000,
        dropout: float = 0,
        batch_first: bool = False,
        norm_first: bool = False,
    ):
        super(Transformer, self).__init__()

        self.has_linear_in = d_in != d_model
        if self.has_linear_in:
            self.linear_in = nn.Linear(d_in, d_model)

        self.pos_encoder = PositionalEncoding(d_model, dropout, seq_in_len)

        encoder_layers = TransformerEncoderLayer(d_model, nhead, dim_feedforward=d_hid, dropout=dropout, batch_first=batch_first, norm_first=norm_first)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)

        self.linear_1 = nn.Linear(d_model * seq_in_len, 128)
        self.linear_2 = nn.Linear(128, 128)
        self.linear_3 = nn.Linear(128, num_classes)

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x: torch.Tensor, x_mask: torch.Tensor = None) -> torch.Tensor:
        if self.has_linear_in:
            x = self.linear_in(x)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x,src_key_padding_mask=x_mask)

        x = torch.permute(x, (1, 0, 2))
        x = x.flatten(start_dim=1)
        x = self.linear_1(x)
        x = nn.ReLU()(x)
        x = self.linear_2(x)
        x = nn.ReLU()(x)
        x = self.linear_3(x)

        return x
