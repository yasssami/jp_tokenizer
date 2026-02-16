from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple
from .config import DictConfig, TokenizerConfig
from .dict import ensure_unidic_mecab, UniDicLexicon, ConnectionCostMatrix, CharClassifier, UnkLexicon
from .lattice import Lattice
from .viterbi import viterbi_decode
from .types import Token
from .neural.io import NeuralCheckpoint
from .neural.constraints import constrained_best_path, B, I, E, S
from .neural.model import BoundaryBatch


def _span_cost_per_char(tokens: List[Token], i0: int, i1: int) -> float:
    # cost diff across token ends / span length
    if i0 >= i1:
        return 0.0
    start = tokens[i0].start
    end = tokens[i1 - 1].end
    if end <= start:
        return 0.0
    cost0 = tokens[i0 - 1].total_cost if i0 > 0 else 0
    cost1 = tokens[i1 - 1].total_cost
    return float(cost1 - cost0) / float(end - start)


def _find_fallback_spans(tokens: List[Token], cfg: TokenizerConfig) -> List[Tuple[int, int]]:
    spans: List[Tuple[int, int]] = []
    n = len(tokens)
    i = 0
    while i < n:
        bad = (tokens[i].source == "UNK")
        if not bad:
            # also consider high-cost regions
            # local span length 1 token
            cpc = _span_cost_per_char(tokens, i, i + 1)
            bad = cpc >= cfg.fallback_cost_per_char_threshold
        if not bad:
            i += 1
            continue
        j = i + 1
        while j < n:
            bad_j = (tokens[j].source == "UNK") or (_span_cost_per_char(tokens, j, j + 1) >= cfg.fallback_cost_per_char_threshold)
            if not bad_j:
                break
            j += 1
        spans.append((i, j))
        i = j
    return spans


def _merge_and_expand_spans(tokens: List[Token], spans: List[Tuple[int, int]], cfg: TokenizerConfig) -> List[Tuple[int, int]]:
    if not spans:
        return []
    merged: List[Tuple[int, int]] = []
    cur0, cur1 = spans[0]
    for a, b in spans[1:]:
        if a <= cur1:
            cur1 = max(cur1, b)
        else:
            merged.append((cur0, cur1))
            cur0, cur1 = a, b
    merged.append((cur0, cur1))

    expanded: List[Tuple[int, int]] = []
    for a, b in merged:
        # expand by chars, not tokens
        start_char = tokens[a].start
        end_char = tokens[b - 1].end
        start_char = max(0, start_char - cfg.fallback_expand_chars)
        end_char = min(tokens[-1].end, end_char + cfg.fallback_expand_chars)
        if end_char - start_char > cfg.fallback_max_span_chars:
            # clamp length
            end_char = start_char + cfg.fallback_max_span_chars

        # map back to token indices covering [start_char, end_char[
        ia = a
        while ia > 0 and tokens[ia - 1].end > start_char:
            ia -= 1
        ib = b
        while ib < len(tokens) and tokens[ib].start < end_char:
            ib += 1
        expanded.append((ia, ib))
    return expanded


@dataclass
class HybridTokenizer:
    dict_cfg: DictConfig = DictConfig()
    cfg: TokenizerConfig = TokenizerConfig()
    neural_ckpt_dir: Optional[Path] = None
    device: str = "cpu"

    def __post_init__(self) -> None:
        ensure_unidic_mecab(self.dict_cfg)
        self.lexicon = UniDicLexicon(self.dict_cfg.lex_csv)
        self.conn = ConnectionCostMatrix(self.dict_cfg.matrix_def)
        self.classifier = CharClassifier(self.dict_cfg.char_def)
        self.unk_lex = UnkLexicon(self.dict_cfg.unk_def, self.classifier, self.cfg.unk_penalty)

        self._neural = None
        self._vocab = None
        if self.neural_ckpt_dir is not None and NeuralCheckpoint(self.neural_ckpt_dir).exists():
            self._load_neural()

    def _load_neural(self) -> None:
        ckpt = NeuralCheckpoint(self.neural_ckpt_dir)  # type: ignore[arg-type]
        model, vocab = ckpt.load(device=self.device)
        self._neural = model
        self._neural = self._neural.to(self.device)
        self._vocab = vocab

    def tokenize_dict(self, text: str) -> List[Token]:
        lat = Lattice.build(text, self.lexicon, self.unk_lex, self.cfg.max_word_len)
        res = viterbi_decode(lat, self.conn)
        return res.tokens

    def _neural_segment(self, text: str, start: int, end: int) -> List[Token]:
        import torch # try importing torch locally instead
        assert self._neural is not None and self._vocab is not None
        sub = text[start:end]
        chars = list(sub)
        ids = [self._vocab.get(ch, 1) for ch in chars]  # 1=<UNK>
        x = torch.tensor([ids], dtype=torch.long, device=self.device)
        lengths = torch.tensor([len(ids)], dtype=torch.long, device=self.device)
        logp = self._neural.predict_logprobs(x, lengths)[0]  # (T,4)
        log_probs = [[float(logp[t, k].item()) for k in range(4)] for t in range(len(ids))]
        path = constrained_best_path(log_probs)

        # convert BIES to tokens
        toks: List[Token] = []
        i = 0
        cum_cost = 0
        while i < len(chars):
            lab = path[i]
            if lab == S:
                tok = chars[i]
                toks.append(Token(tok, "NEURAL", "", start + i, start + i + 1, cum_cost, "NEURAL"))
                i += 1
                continue
            # lab == B: consume until E
            if lab != B:
                # repair: treat as S
                tok = chars[i]
                toks.append(Token(tok, "NEURAL", "", start + i, start + i + 1, cum_cost, "NEURAL"))
                i += 1
                continue
            j = i + 1
            while j < len(chars) and path[j] != E:
                j += 1
            if j >= len(chars):
                # no E found; fallback to remaining as one token
                tok = "".join(chars[i:])
                toks.append(Token(tok, "NEURAL", "", start + i, end, cum_cost, "NEURAL"))
                break
            tok = "".join(chars[i:j + 1])
            toks.append(Token(tok, "NEURAL", "", start + i, start + j + 1, cum_cost, "NEURAL"))
            i = j + 1
        return toks

    def tokenize(self, text: str) -> List[Token]:
        toks = self.tokenize_dict(text)
        if not self.cfg.enable_neural_fallback:
            return toks
        if self._neural is None:
            return toks  # dict only until neural checkpoint provided

        spans = _find_fallback_spans(toks, self.cfg)
        spans = _merge_and_expand_spans(toks, spans, self.cfg)
        if not spans:
            return toks

        # replacements: l->r
        out: List[Token] = []
        cur_tok_idx = 0
        for a, b in spans:
            # append untouched
            out.extend(toks[cur_tok_idx:a])
            # replace [a,b[ w/ neural segmentation over its char span
            start = toks[a].start
            end = toks[b - 1].end
            out.extend(self._neural_segment(text, start, end))
            cur_tok_idx = b
        out.extend(toks[cur_tok_idx:])

        # recompute cumulative costs deterministically (dict costs not applicable for neural)
        # keep original dictionary cumulative cost where possible; +0 for neural segments

        # think this is finding first matching elem, results are weird for dupes
        # cum = 0
        # for t in out:
        #     if t.source != "NEURAL":
        #         cum = t.total_cost
        #     out[out.index(t)] = Token(t.surface, t.pos, t.feature, t.start, t.end, cum, t.source)
        # return out
        fix: List[Token] = []
        last_dict_cost = 0
        for t in out:
            if t.source != "NEURAL":
                last_dict_cost = t.total_cost
                fix.append(t)
            else:
                fix.append(Token(
                    surface=t.surface,
                    pos=t.pos,
                    feature=t.feature,
                    start=t.start,
                    end=t.end,
                    total_cost=last_dict_cost,
                    source=t.source,
                ))
        return fix