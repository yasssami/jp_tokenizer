from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict
import torch
import warnings
from ..util import read_json, write_json
from .model import BoundaryLSTM

@dataclass(frozen=True)
class NeuralCheckpoint:
    dir_path: Path
    @property
    def weights_path(self) -> Path:
        return self.dir_path / "weights.pt"
    @property
    def vocab_path(self) -> Path:
        return self.dir_path / "vocab.json"
    def exists(self) -> bool:
        return self.weights_path.exists() and self.vocab_path.exists()

    def save(self, model: BoundaryLSTM, vocab: Dict[str, int]) -> None:
        self.dir_path.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), self.weights_path)
        meta = {
            "format_version": 1,
            "emb_dim": int(model.emb.embedding_dim),
            "hidden_size": int(model.lstm.hidden_size),
            "dropout": float(model.dropout.p),
            "pad_id": int(vocab.get("<PAD>", 0)),
            "unk_id": int(vocab.get("<UNK>", 1)),
        }
        write_json(self.vocab_path, {"vocab": vocab, "meta": meta})

    def load(self, device: str = "cpu") -> tuple[BoundaryLSTM, Dict[str, int], Dict[str, int | float]]:
        data = read_json(self.vocab_path)
        vocab = data["vocab"]
        meta = data.get("meta") or {}
        sd = torch.load(self.weights_path, map_location=device)

        if meta:
            emb_dim = int(meta["emb_dim"])
            hidden_size = int(meta["hidden_size"])
            dropout = float(meta.get("dropout", 0.1))
        else:
            emb_weight = sd.get("emb.weight")
            u_weight = sd.get("lstm.cell.U")
            w_weight = sd.get("lstm.cell.W")
            if emb_weight is None or (u_weight is None and w_weight is None):
                raise RuntimeError("Could not infer model dimensions from checkpoint state_dict.")
            emb_dim = int(emb_weight.shape[1])
            if u_weight is not None:
                hidden_size = int(u_weight.shape[0])
            else:
                hidden_size = int(w_weight.shape[1] // 4)
            dropout = 0.1
            meta = {
                "format_version": 0,
                "emb_dim": emb_dim,
                "hidden_size": hidden_size,
                "dropout": dropout,
                "pad_id": int(vocab.get("<PAD>", 0)),
                "unk_id": int(vocab.get("<UNK>", 1)),
            }
            warnings.warn(
                "Checkpoint metadata missing; inferred model dimensions from weights and defaulted dropout to 0.1.",
                UserWarning,
            )

        model = BoundaryLSTM(
            vocab_size=max(vocab.values()) + 1,
            emb_dim=emb_dim,
            hidden_size=hidden_size,
            dropout=dropout,
        )
        model.load_state_dict(sd)
        model.eval()
        return model, vocab, meta
