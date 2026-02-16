from __future__ import annotations
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from ..types import Morpheme

@dataclass(frozen=True)
class CharCategory:
    name: str
    invoke: int
    group: int
    max_len: int
    # list of (start_cp, end_cp_inclusive)
    ranges: Tuple[Tuple[int, int], ...]


@dataclass(frozen=True)
class UnkEntry:
    category: str
    left_id: int
    right_id: int
    cost: int
    feature: str

    @property
    def pos(self) -> str:
        parts = self.feature.split(",")
        pos_parts = parts[:4] if len(parts) >= 4 else parts
        pos_parts = [p for p in pos_parts if p and p != "*"]
        return "-".join(pos_parts) if pos_parts else "UNK"


class CharClassifier:
    """
    parse mecab char.def to map char to category + grouping
    """
    def __init__(self, char_def_path: Path) -> None:
        self.char_def_path = char_def_path
        self.categories: Dict[str, CharCategory] = {}
        self._loaded = False
        self._compiled: List[CharCategory] = []

    def load(self) -> None:
        if self._loaded:
            return
        raw = self.char_def_path.read_text(encoding="utf-8", errors="ignore")
        lines = raw.replace("\r\n", "\n").replace("\r", "\n").split("\n")
        # TODO check char.def variant flattening. maybe try to re-split heuristically.
        if len(lines) <= 2 and "0x" in raw:
            # split around instance of " 0x" ranges to recover structure
            # still safe: parse token-by-token
            lines = [t.strip() for t in raw.split("#") if t.strip()]

        # two kinds of lines:
        # 1) CATEGORY invoke group max_len
        # 2) 0xXXXX[..0xYYYY] CATEGORY
        cat_defs: Dict[str, Tuple[int, int, int]] = {}
        ranges: Dict[str, List[Tuple[int, int]]] = {}

        for line in lines:
            line = line.split("#", 1)[0].strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) >= 4 and not parts[0].startswith("0x"):
                # cat definition
                name = parts[0]
                try:
                    invoke, group, max_len = int(parts[1]), int(parts[2]), int(parts[3])
                except ValueError:
                    continue
                cat_defs[name] = (invoke, group, max_len)
                continue

            if parts[0].startswith("0x") and len(parts) >= 2:
                r = parts[0]
                cat = parts[1]
                if ".." in r:
                    a, b = r.split("..")
                    start = int(a, 16)
                    end = int(b, 16)
                else:
                    start = int(r, 16)
                    end = start
                ranges.setdefault(cat, []).append((start, end))

        for name, (invoke, group, max_len) in cat_defs.items():
            rr = tuple(ranges.get(name, []))
            self.categories[name] = CharCategory(name, invoke, group, max_len, rr)

        # Ensure DEFAULT exists
        if "DEFAULT" not in self.categories:
            self.categories["DEFAULT"] = CharCategory("DEFAULT", 0, 1, 1, tuple())

        self._compiled = list(self.categories.values())
        self._loaded = True

    def category_of(self, ch: str) -> CharCategory:
        self.load()
        cp = ord(ch)
        # check explicit ranges first
        for cat in self._compiled:
            for a, b in cat.ranges:
                if a <= cp <= b:
                    return cat
        return self.categories["DEFAULT"]

    def max_group_len(self, ch: str) -> int:
        cat = self.category_of(ch)
        if cat.group == 0:
            return 1
        return max(1, cat.max_len)


class UnkLexicon:
    """
    parse mecab unk.def:
      CATEGORY,left_id,right_id,cost,feature...
    and use CharClassifier to produce UNK candidates
    """
    def __init__(self, unk_def_path: Path, classifier: CharClassifier, default_penalty: int) -> None:
        self.unk_def_path = unk_def_path
        self.classifier = classifier
        self.default_penalty = default_penalty
        self.by_cat: Dict[str, List[UnkEntry]] = {}
        self._loaded = False

    def load(self) -> None:
        if self._loaded:
            return
        with self.unk_def_path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.reader(f)
            for row in reader:
                if not row:
                    continue
                if len(row) < 5:
                    continue
                cat = row[0]
                try:
                    left_id = int(row[1]); right_id = int(row[2]); cost = int(row[3])
                except ValueError:
                    continue
                feature = ",".join(row[4:])
                self.by_cat.setdefault(cat, []).append(UnkEntry(cat, left_id, right_id, cost, feature))
        self._loaded = True

    def unknown_candidates(self, text: str, start: int, max_span_len: int) -> List[Tuple[int, Morpheme]]:
        self.load()
        if start >= len(text):
            return []
        cat = self.classifier.category_of(text[start])
        max_len = min(self.classifier.max_group_len(text[start]), max_span_len, len(text) - start)
        entries = self.by_cat.get(cat.name) or self.by_cat.get("DEFAULT") or []
        if not entries:
            # hard fallback if unk.def is weird/missing/whatever
            return [(
                start + 1,
                Morpheme(text[start], "UNK", 0, 0, self.default_penalty, feature="", source="UNK")
            )]

        out: List[Tuple[int, Morpheme]] = []
        for L in range(1, max_len + 1):
            surf = text[start:start + L]
            for entry in entries:
                # slightly increase cost with length to avoid swallowing too much.. TODO calibrate
                cost = int(entry.cost + self.default_penalty + 30 * (L - 1))
                out.append((
                    start + L,
                    Morpheme(surf, entry.pos or "UNK", entry.left_id, entry.right_id, cost, entry.feature, "UNK")
                ))
        return out
