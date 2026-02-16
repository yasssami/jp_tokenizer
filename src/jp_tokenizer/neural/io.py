from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List
import torch
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
        write_json(self.vocab_path, {"vocab": vocab})

    def load(self, device: str = "cpu") -> tuple[BoundaryLSTM, Dict[str, int]]:
        meta = read_json(self.vocab_path)
        vocab = meta["vocab"]
        model = BoundaryLSTM(vocab_size=max(vocab.values()) + 1)
        sd = torch.load(self.weights_path, map_location=device)
        model.load_state_dict(sd)
        model.eval()
        return model, vocab
