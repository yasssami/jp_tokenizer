from __future__ import annotations
from dataclasses import dataclass
from typing import List
from .types import LatticeNode, Morpheme
from .dict.lexicon import UniDicLexicon
from .dict.charclasses import UnkLexicon

@dataclass
class Lattice:
    text: str
    # nodes_start[i] = list of nodes that begin at i
    nodes_start: List[List[LatticeNode]]

    @classmethod
    def build(
        cls,
        text: str,
        lexicon: UniDicLexicon,
        unk_lex: UnkLexicon,
        max_word_len: int,
    ) -> "Lattice":
        n = len(text)
        nodes_start: List[List[LatticeNode]] = [[] for _ in range(n + 1)]
        for i in range(n):
            cands = lexicon.lookup(text, i, max_word_len)
            for end, morph in cands:
                nodes_start[i].append(LatticeNode(i, end, morph))
            # UNK candidates are always added (but dict wins via cost)
            unk_cands = unk_lex.unknown_candidates(text, i, max_word_len)
            for end, morph in unk_cands:
                nodes_start[i].append(LatticeNode(i, end, morph))
        return cls(text=text, nodes_start=nodes_start)

    def candidates_from(self, start: int) -> List[LatticeNode]:
        if start < 0 or start >= len(self.nodes_start):
            return []
        return self.nodes_start[start]
