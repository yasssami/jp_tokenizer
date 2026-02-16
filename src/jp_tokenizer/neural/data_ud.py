from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple
from datasets import load_dataset

# create char-lv B/I/E/S labels from tokenized sentences
# UD japanese uses tokenized long-unit-words; treat those word boundaries as segmentation targets


@dataclass(frozen=True)
class CharExample:
    chars: List[str]
    labels: List[int]  # map 0-3 to b, i, e, s respectively


def tokens_to_bies(tokens: List[str]) -> CharExample:
    chars: List[str] = []
    labels: List[int] = []
    for tok in tokens:
        if tok == "":
            continue
        cs = list(tok)
        if len(cs) == 1:
            chars.append(cs[0]); labels.append(3)  # S
        else:
            for i, ch in enumerate(cs):
                chars.append(ch)
                if i == 0:
                    labels.append(0)  # B
                elif i == len(cs) - 1:
                    labels.append(2)  # E
                else:
                    labels.append(1)  # I
    return CharExample(chars, labels)


def iter_ud_japanese(split: str = "train") -> Iterable[CharExample]:
    """
    load UD japanese (GSD / GSDLUW depending on datasets availability in HF)
    try GSDLUW first (LUW tokenization), then fallback to GSD
    """
    for name in ["universal_dependencies", "universal_dependencies"]:
        # builder uses config like "ja_gsdluw" or "ja_gsd"
        for cfg in ["ja_gsdluw", "ja_gsd"]:
            try:
                ds = load_dataset(name, cfg, split=split)
                for ex in ds:
                    toks = ex.get("tokens") or ex.get("token") or ex.get("words")
                    if toks is None:
                        # fallback: reconstruct from text if present
                        continue
                    yield tokens_to_bies(list(toks))
                return
            except Exception:
                continue
    raise RuntimeError("Could not load UD Japanese via datasets. Ensure internet access and that 'datasets' is installed.")
