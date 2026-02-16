from __future__ import annotations
from pathlib import Path
from typing import Tuple, Optional
import numpy as np

def _ct_id_def(path: Path) -> int:
    n = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        n += 1
    return n

class ConnectionCostMatrix:
    """
    load mecab matrix.def + handle header orientation
    try to more consistently expose cost(prev_right_id, next_left_id)
    use left-id.def & right-id.def line cts to decide whether matrix.def header is L R or R L
    maybe fixes weird POS/segmentation errs
    """
    def __init__(
        self,
        matrix_def_path: Path,
        left_id_def_path: Optional[Path] = None,
        right_id_def_path: Optional[Path] = None,
    ) -> None:
        self.matrix_def_path = matrix_def_path
        self.left_id_def_path = left_id_def_path
        self.right_id_def_path = right_id_def_path
        self.right_size = 0
        self.left_size = 0
        self.mat: np.ndarray | None = None

    def load(self) -> None:
        if self.mat is not None:
            return

        left_ct = _ct_id_def(self.left_id_def_path) if self.left_id_def_path else None
        right_ct = _ct_id_def(self.right_id_def_path) if self.right_id_def_path else None

        with self.matrix_def_path.open("r", encoding="utf-8", errors="ignore") as f:
            # header: 2 ints
            header = ""
            while header.strip() == "":
                header = f.readline()
                if header == "":
                    raise RuntimeError("matrix.def is empty")
            a_str, b_str = header.strip().split()
            a = int(a_str)
            b = int(b_str)

            # decide orientation
            # mode RL: header = (right_size left_size), lines = (right left cost)
            # mode LR: header = (left_size right_size), lines = (left right cost) -> transpose at fill
            mode = "RL"

            if left_ct is not None and right_ct is not None:
                if a == right_ct and b == left_ct:
                    mode = "RL"
                    self.right_size, self.left_size = a, b
                elif a == left_ct and b == right_ct:
                    mode = "LR"
                    self.right_size, self.left_size = b, a
                else:
                    # Fallback: trust header as RL, but keep counts for validation
                    mode = "RL"
                    self.right_size, self.left_size = a, b
            else:
                # No id.def sizes available: assume RL (common MeCab convention)
                mode = "RL"
                self.right_size, self.left_size = a, b

            # Use int32 to avoid overflow surprises.
            mat = np.zeros((self.right_size, self.left_size), dtype=np.int32)

            # Fill
            line_no = 1
            for line in f:
                line_no += 1
                s = line.strip()
                if not s or s.startswith("#"):
                    continue
                p = s.split()
                if len(p) != 3:
                    continue
                i0 = int(p[0]); i1 = int(p[1]); c = int(p[2])

                if mode == "RL":
                    r, l = i0, i1
                else:
                    # header/lines are left,right: transpose into [right,left]
                    l, r = i0, i1

                if 0 <= r < self.right_size and 0 <= l < self.left_size:
                    mat[r, l] = c
                else:
                    # should be rare as long as dict is formed ok
                    # try not to hang on failure, just warn & skip
                    raise RuntimeError(
                        f"matrix.def index out of bounds at line {line_no}: "
                        f"(r={r}, l={l}) not in (right={self.right_size}, left={self.left_size}). "
                        f"Header was ({a},{b}), mode={mode}, left-id lines={left_ct}, right-id lines={right_ct}."
                    )

        self.mat = mat

    def cost(self, prev_right_id: int, next_left_id: int) -> int:
        self.load()
        assert self.mat is not None
        if 0 <= prev_right_id < self.right_size and 0 <= next_left_id < self.left_size:
            return int(self.mat[prev_right_id, next_left_id])
        return 0
    # def __init__(self, matrix_def_path: Path) -> None:
    #     self.matrix_def_path = matrix_def_path
    #     self.right_size = 0
    #     self.left_size = 0
    #     self.mat: np.ndarray | None = None

    # def load(self) -> None:
    #     if self.mat is not None:
    #         return
    #     with self.matrix_def_path.open("r", encoding="utf-8") as f:
    #         header = ""
    #         while header.strip() == "":
    #             header = f.readline()
    #             if header == "":
    #                 raise RuntimeError("matrix.def is empty")
    #         a, b = header.strip().split()
    #         self.right_size = int(a)
    #         self.left_size = int(b)
    #         mat = np.zeros((self.right_size, self.left_size), dtype=np.int32)
    #         for line in f:
    #             line = line.strip()
    #             if not line or line.startswith("#"):
    #                 continue
    #             r, l, c = line.split()
    #             mat[int(r), int(l)] = int(c)
    #     self.mat = mat

    # def cost(self, prev_right_id: int, next_left_id: int) -> int:
    #     self.load()
    #     assert self.mat is not None
    #     if 0 <= prev_right_id < self.right_size and 0 <= next_left_id < self.left_size:
    #         return int(self.mat[prev_right_id, next_left_id])
    #     # graceful degradation for oor IDs
    #     return 0
