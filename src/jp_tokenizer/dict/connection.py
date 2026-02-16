from __future__ import annotations
from pathlib import Path
from typing import Tuple
import numpy as np


class ConnectionCostMatrix:
    """
    load mecab matrix.def

    common format:
      <right_size> <left_size>
      <right_id> <left_id> <cost>
      ...
    (triplets: right-context-id, left-context-id, cost)
    """
    def __init__(self, matrix_def_path: Path) -> None:
        self.matrix_def_path = matrix_def_path
        self.right_size = 0
        self.left_size = 0
        self.mat: np.ndarray | None = None

    def load(self) -> None:
        if self.mat is not None:
            return
        with self.matrix_def_path.open("r", encoding="utf-8") as f:
            header = ""
            while header.strip() == "":
                header = f.readline()
                if header == "":
                    raise RuntimeError("matrix.def is empty")
            a, b = header.strip().split()
            self.right_size = int(a)
            self.left_size = int(b)
            mat = np.zeros((self.right_size, self.left_size), dtype=np.int32)
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                r, l, c = line.split()
                mat[int(r), int(l)] = int(c)
        self.mat = mat

    def cost(self, prev_right_id: int, next_left_id: int) -> int:
        self.load()
        assert self.mat is not None
        if 0 <= prev_right_id < self.right_size and 0 <= next_left_id < self.left_size:
            return int(self.mat[prev_right_id, next_left_id])
        # graceful degradation for oor IDs
        return 0
