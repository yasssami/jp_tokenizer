from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Tuple

# consume existing KNP files that contain surfaces per morpheme line
# TODO see if can attempt to reconstruct original sentences (ktc repo only has annotations)

@dataclass(frozen=True)
class CharExample:
    chars: List[str]
    labels: List[int]  # 0=B,1=I,2=E,3=S


def _bies_for_tokens(tokens: List[str]) -> CharExample:
    chars: List[str] = []
    labels: List[int] = []
    for tok in tokens:
        if not tok:
            continue
        cs = list(tok)
        if len(cs) == 1:
            chars.append(cs[0]); labels.append(3)
        else:
            for i, ch in enumerate(cs):
                chars.append(ch)
                if i == 0:
                    labels.append(0)
                elif i == len(cs) - 1:
                    labels.append(2)
                else:
                    labels.append(1)
    return CharExample(chars, labels)


def iter_kyoto_knp(knp_root: Path) -> Iterable[CharExample]:
    """
    in: KNP files with morpheme lines:
      surface \t reading \t lemma \t pos ...
    interpret each morpheme surface as "token" boundary
    """
    for path in knp_root.rglob("*.knp"):
        tokens: List[str] = []
        for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
            if not line or line.startswith("#") or line.startswith("*") or line.startswith("+"):
                continue
            if line == "EOS":
                if tokens:
                    yield _bies_for_tokens(tokens)
                tokens = []
                continue
            cols = line.split("\t")
            if cols:
                tokens.append(cols[0])
        if tokens:
            yield _bies_for_tokens(tokens)
