from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple
import numpy as np

@dataclass(frozen=True)
class IdDefInfo:
    count: int
    label_to_id: Dict[str, int]


def _parse_id_def(path: Optional[Path]) -> Optional[IdDefInfo]:
    if path is None:
        return None
    text = path.read_text(encoding="utf-8", errors="ignore")
    label_to_id: Dict[str, int] = {}
    line_count = 0
    explicit = False
    max_id = -1
    for line in text.splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        parts = s.split()
        if not parts:
            continue
        line_count += 1
        label = ""
        if parts[0].isdigit():
            explicit = True
            id_val = int(parts[0])
            max_id = max(max_id, id_val)
            if len(parts) > 1:
                label = parts[1]
        else:
            id_val = line_count - 1
            label = parts[0]
        if label:
            label_to_id[label] = id_val
    if explicit and max_id >= 0:
        count = max_id + 1
    else:
        count = line_count
    return IdDefInfo(count=count, label_to_id=label_to_id)

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
        self._left_info: Optional[IdDefInfo] = None
        self._right_info: Optional[IdDefInfo] = None

    def _ensure_id_defs(self) -> None:
        if self._left_info is None:
            self._left_info = _parse_id_def(self.left_id_def_path)
        if self._right_info is None:
            self._right_info = _parse_id_def(self.right_id_def_path)

    def bos_right_id(self) -> int:
        self._ensure_id_defs()
        if self._right_info is not None:
            for key in ("BOS/EOS", "BOS"):
                if key in self._right_info.label_to_id:
                    return self._right_info.label_to_id[key]
        return 0

    def eos_left_id(self) -> int:
        self._ensure_id_defs()
        if self._left_info is not None:
            for key in ("BOS/EOS", "EOS"):
                if key in self._left_info.label_to_id:
                    return self._left_info.label_to_id[key]
        return 0

    def load(self) -> None:
        if self.mat is not None:
            return

        self._ensure_id_defs()
        left_ct = self._left_info.count if self._left_info is not None else None
        right_ct = self._right_info.count if self._right_info is not None else None

        def _load_with_mode(mode: str) -> np.ndarray:
            with self.matrix_def_path.open("r", encoding="utf-8", errors="ignore") as f:
                header = ""
                while header.strip() == "":
                    header = f.readline()
                    if header == "":
                        raise RuntimeError("matrix.def is empty") #TODO check if rte avoidable; maybe allow no header and infer from lines
                a_str, b_str = header.strip().split()
                a = int(a_str)
                b = int(b_str)

                if mode == "RL":
                    right_size, left_size = a, b
                else:
                    right_size, left_size = b, a

                mat = np.zeros((right_size, left_size), dtype=np.int32)

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
                        l, r = i0, i1
                    if 0 <= r < right_size and 0 <= l < left_size:
                        mat[r, l] = c
                    else:
                        raise RuntimeError(
                            f"matrix.def index out of bounds at line {line_no}: "
                            f"(r={r}, l={l}) not in (right={right_size}, left={left_size}). "
                            f"Header was ({a},{b}), mode={mode}, left-id lines={left_ct}, right-id lines={right_ct}."
                        )
            # set sizes only after successful load
            self.right_size = right_size
            self.left_size = left_size
            return mat

        # decide candidate mode
        # mode RL: header = (right_size left_size), lines = (right left cost)
        # mode LR: header = (left_size right_size), lines = (left right cost) -> transpose at fill
        modes: list[str] = []
        if left_ct is not None and right_ct is not None:
            with self.matrix_def_path.open("r", encoding="utf-8", errors="ignore") as f:
                header = ""
                while header.strip() == "":
                    header = f.readline()
                    if header == "":
                        raise RuntimeError("matrix.def is empty")
                a_str, b_str = header.strip().split()
                a = int(a_str); b = int(b_str)
            if a == right_ct and b == left_ct:
                modes = ["RL", "LR"]
            elif a == left_ct and b == right_ct:
                modes = ["LR", "RL"]
            else:
                modes = ["RL", "LR"]
        else:
            modes = ["RL", "LR"]

        last_err: Optional[Exception] = None
        for mode in modes:
            try:
                self.mat = _load_with_mode(mode)
                return
            except Exception as e:
                last_err = e
                continue
        if last_err is not None:
            raise last_err

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
