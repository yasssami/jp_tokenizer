from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F




class ManualLSTMCell(nn.Module):
    """
    manually implemented LSTM cell:
      i,f,g,o = sigmoid/sigmoid/tanh/sigmoid(Wx + Uh + b)
      c' = f*c + i*g
      h' = o*tanh(c')
    """
    def __init__(self, input_size: int, hidden_size: int) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        # combine gates for efficiency: 4*hidden
        self.W = nn.Parameter(torch.empty(input_size, 4 * hidden_size))
        self.U = nn.Parameter(torch.empty(hidden_size, 4 * hidden_size))
        self.b = nn.Parameter(torch.zeros(4 * hidden_size))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.W)
        nn.init.orthogonal_(self.U)
        nn.init.zeros_(self.b)

    def forward(
        self,
        x_t: torch.Tensor,          # (B, input_size)
        state: Tuple[torch.Tensor, torch.Tensor],  # (h, c) each (B, H)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        h, c = state
        gates = x_t @ self.W + h @ self.U + self.b
        i, f, g, o = gates.chunk(4, dim=-1)
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        g = torch.tanh(g)
        o = torch.sigmoid(o)
        c_new = f * c + i * g
        h_new = o * torch.tanh(c_new)
        return h_new, c_new


class ManualLSTM(nn.Module):
    def __init__(self, input_size: int, hidden_size: int) -> None:
        super().__init__()
        self.cell = ManualLSTMCell(input_size, hidden_size)
        self.hidden_size = hidden_size

    def forward(
        self,
        x: torch.Tensor,            # (B, T, D)
        lengths: torch.Tensor,      # (B,)
    ) -> torch.Tensor:
        B, T, D = x.shape
        device = x.device
        h = torch.zeros(B, self.hidden_size, device=device, dtype=x.dtype)
        c = torch.zeros(B, self.hidden_size, device=device, dtype=x.dtype)
        outs = []
        for t in range(T):
            h, c = self.cell(x[:, t, :], (h, c))
            outs.append(h.unsqueeze(1))
        out = torch.cat(outs, dim=1)  # (B, T, H)

        # mask out padding positions to avoid garbage gradients TODO I LOWKEY DONT KNOW IF THIS IS DONE RIGHT
        mask = torch.arange(T, device=device).unsqueeze(0) < lengths.unsqueeze(1)
        out = out * mask.unsqueeze(-1)
        return out


@dataclass(frozen=True)
class BoundaryBatch:
    x: torch.Tensor         # (B, T) int64
    y: Optional[torch.Tensor]  # (B, T) int64 or None
    lengths: torch.Tensor   # (B,) int64


class BoundaryLSTM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        emb_dim: int = 128,
        hidden_size: int = 256,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.lstm = ManualLSTM(emb_dim, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.proj = nn.Linear(hidden_size, 4)  # B,I,E,S

    def forward(self, batch: BoundaryBatch) -> torch.Tensor:
        x = batch.x
        lengths = batch.lengths
        emb = self.emb(x)
        h = self.lstm(emb, lengths)
        h = self.dropout(h)
        logits = self.proj(h)  # (B, T, 4)
        return logits

    @torch.inference_mode()
    def predict_logprobs(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        logits = self.forward(BoundaryBatch(x=x, y=None, lengths=lengths))
        return F.log_softmax(logits, dim=-1)
