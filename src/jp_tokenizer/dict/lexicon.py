from __future__ import annotations
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple
from ..types import Morpheme


@dataclass(frozen=True)
class LexEntry:
    surface: str
    left_id: int
    right_id: int
    cost: int
    feature: str

    @property
    def pos(self) -> str:
        # mecab schema: feature starts with pos1,pos2,pos3,pos4,...
        # keep compact POS string
        parts = self.feature.split(",")
        pos_parts = parts[:4] if len(parts) >= 4 else parts
        pos_parts = [p for p in pos_parts if p and p != "*"]
        return "-".join(pos_parts) if pos_parts else "UNK"


class TrieNode:
    __slots__ = ("children", "entries")

    def __init__(self) -> None:
        self.children: Dict[str, TrieNode] = {}
        self.entries: List[LexEntry] = []


class Trie:
    def __init__(self) -> None:
        self.root = TrieNode()

    def insert(self, key: str, entry: LexEntry) -> None:
        node = self.root
        for ch in key:
            nxt = node.children.get(ch)
            if nxt is None:
                nxt = TrieNode()
                node.children[ch] = nxt
            node = nxt
        node.entries.append(entry)

    def common_prefix_search(self, text: str, start: int, max_len: int) -> Iterable[Tuple[int, LexEntry]]:
        node = self.root
        end = start
        for _ in range(max_len):
            if end >= len(text):
                break
            ch = text[end]
            node = node.children.get(ch)
            if node is None:
                break
            end += 1
            if node.entries:
                for e in node.entries:
                    yield end, e


class UniDicLexicon:
    """
    load csv into a trie
    """
    def __init__(self, lex_csv_path: Path) -> None:
        self.lex_csv_path = lex_csv_path
        self.trie = Trie()
        self._loaded = False

    def load(self) -> None:
        if self._loaded:
            return
        with self.lex_csv_path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.reader(f)
            for row in reader:
                if not row:
                    continue
                # mecab csv: surface,left_id,right_id,cost,feature...
                if len(row) < 5:
                    continue
                surface = row[0]
                try:
                    left_id = int(row[1])
                    right_id = int(row[2])
                    cost = int(row[3])
                except ValueError:
                    continue
                feature = ",".join(row[4:])
                self.trie.insert(surface, LexEntry(surface, left_id, right_id, cost, feature))
        self._loaded = True

    def lookup(self, text: str, start: int, max_len: int) -> List[Tuple[int, Morpheme]]:
        """
        return list of (end_index, Morpheme) candidates starting at `start`
        """
        self.load()
        out: List[Tuple[int, Morpheme]] = []
        for end, e in self.trie.common_prefix_search(text, start, max_len):
            out.append((
                end,
                Morpheme(
                    surface=e.surface,
                    pos=e.pos,
                    left_id=e.left_id,
                    right_id=e.right_id,
                    word_cost=e.cost,
                    feature=e.feature,
                    source="DICT",
                )
            ))
        return out
