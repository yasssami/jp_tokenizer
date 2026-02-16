from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Sequence


@dataclass(frozen=True)
class Morpheme:
    surface: str
    pos: str
    left_id: int
    right_id: int
    word_cost: int
    feature: str = ""
    source: str = "DICT"


@dataclass(frozen=True)
class LatticeNode:
    start: int
    end: int
    morph: Morpheme


@dataclass(frozen=True)
class Token:
    surface: str
    pos: str
    feature: str
    start: int
    end: int
    total_cost: int
    source: str
