from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple
from .config import DictConfig, TokenizerConfig
from .dict import ensure_unidic_mecab, dict_files_present, UniDicLexicon, ConnectionCostMatrix, CharClassifier, UnkLexicon
from .lattice import Lattice
from .viterbi import viterbi_decode
from .types import Morpheme, Token
from .neural.io import NeuralCheckpoint
from .neural.constraints import constrained_best_path, B, I, E, S


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
    if not expanded:
        return []
    expanded.sort()
    remerged: List[Tuple[int, int]] = []
    cur0, cur1 = expanded[0]
    for a, b in expanded[1:]:
        if a <= cur1:
            cur1 = max(cur1, b)
        else:
            remerged.append((cur0, cur1))
            cur0, cur1 = a, b
    remerged.append((cur0, cur1))
    return remerged


@dataclass
class HybridTokenizer:
    dict_cfg: DictConfig = DictConfig()
    cfg: TokenizerConfig = TokenizerConfig()
    neural_ckpt_dir: Optional[Path] = None
    device: str = "cpu"

    def __post_init__(self) -> None:
        self.lexicon = None
        self.conn = None
        self.classifier = None
        self.unk_lex = None
        self._dict_ready = False

        self._neural = None
        self._vocab = None
        self._vocab_meta = {}
        if self.neural_ckpt_dir is not None and NeuralCheckpoint(self.neural_ckpt_dir).exists():
            self._load_neural()

    def _load_dict(self) -> None:
        self.lexicon = UniDicLexicon(self.dict_cfg.lex_csv)
        self.conn = ConnectionCostMatrix(
            self.dict_cfg.matrix_def,
            left_id_def_path=self.dict_cfg.left_id_def,
            right_id_def_path=self.dict_cfg.right_id_def,
        )
        self.classifier = CharClassifier(self.dict_cfg.char_def)
        self.unk_lex = UnkLexicon(self.dict_cfg.unk_def, self.classifier, self.cfg.unk_penalty)
        self._dict_ready = True

    def _ensure_dict(self, required: bool) -> bool:
        if self._dict_ready:
            return True
        if dict_files_present(self.dict_cfg):
            self._load_dict()
            return True
        if not required:
            return False
        ensure_unidic_mecab(self.dict_cfg)
        self._load_dict()
        return True

    def _load_neural(self) -> None:
        ckpt = NeuralCheckpoint(self.neural_ckpt_dir)  # type: ignore[arg-type]
        model, vocab, meta = ckpt.load(device=self.device)
        self._neural = model
        self._neural = self._neural.to(self.device)
        self._vocab = vocab
        self._vocab_meta = meta

    def tokenize_dict(self, text: str) -> List[Token]:
        self._ensure_dict(required=True)
        assert self.lexicon is not None and self.unk_lex is not None
        lat = Lattice.build(text, self.lexicon, self.unk_lex, self.cfg.max_word_len)
        assert self.conn is not None
        res = viterbi_decode(lat, self.conn)
        return res.tokens

    def _span_candidates(self, text: str, start: int, end: int) -> List[Morpheme]:
        assert self.lexicon is not None and self.unk_lex is not None
        max_len = end - start
        cands: List[Morpheme] = []
        for e, m in self.lexicon.lookup(text, start, max_len):
            if e == end:
                cands.append(m)
        for e, m in self.unk_lex.unknown_candidates(text, start, max_len):
            if e == end:
                cands.append(m)
        if not cands:
            cands.append(Morpheme(
                surface=text[start:end],
                pos="UNK",
                left_id=0,
                right_id=0,
                word_cost=self.cfg.unk_penalty,
                feature="",
                source="UNK",
            ))
        return cands

    def _tag_fixed_spans(self, text: str, spans: List[Tuple[int, int]]) -> List[Morpheme]:
        if not spans:
            return []
        assert self.conn is not None
        cand_lists: List[List[Morpheme]] = [self._span_candidates(text, s, e) for s, e in spans]
        INF = 10**18
        dp: List[List[int]] = [[INF] * len(cands) for cands in cand_lists]
        bp: List[List[int]] = [[-1] * len(cands) for cands in cand_lists]

        bos_right = self.conn.bos_right_id()
        for j, cand in enumerate(cand_lists[0]):
            dp[0][j] = self.conn.cost(bos_right, cand.left_id) + int(cand.word_cost)

        for i in range(1, len(cand_lists)):
            for j, cand in enumerate(cand_lists[i]):
                best = INF
                best_prev = -1
                for k, prev in enumerate(cand_lists[i - 1]):
                    prev_cost = dp[i - 1][k]
                    if prev_cost >= INF:
                        continue
                    score = prev_cost + self.conn.cost(prev.right_id, cand.left_id) + int(cand.word_cost)
                    if score < best:
                        best = score
                        best_prev = k
                dp[i][j] = best
                bp[i][j] = best_prev

        eos_left = self.conn.eos_left_id()
        best_final = INF
        best_j = -1
        last = len(cand_lists) - 1
        for j, cand in enumerate(cand_lists[last]):
            total = dp[last][j] + self.conn.cost(cand.right_id, eos_left)
            if total < best_final:
                best_final = total
                best_j = j

        if best_j < 0:
            return [cands[0] for cands in cand_lists]

        out: List[Morpheme] = [cand_lists[last][best_j]]
        for i in range(last, 0, -1):
            best_j = bp[i][best_j]
            if best_j < 0:
                best_j = 0
            out.append(cand_lists[i - 1][best_j])
        out.reverse()
        return out

    def _neural_segment(self, text: str, start: int, end: int) -> List[Token]:
        import torch # try importing torch locally instead
        assert self._neural is not None and self._vocab is not None
        sub = text[start:end]
        if not sub:
            return []
        chars = list(sub)
        unk_id = int(self._vocab_meta.get("unk_id", 1))
        ids = [self._vocab.get(ch, unk_id) for ch in chars]
        x = torch.tensor([ids], dtype=torch.long, device=self.device)
        lengths = torch.tensor([len(ids)], dtype=torch.long, device=self.device)
        logp = self._neural.predict_logprobs(x, lengths)[0]  # (T,4)
        log_probs = [[float(logp[t, k].item()) for k in range(4)] for t in range(len(ids))]
        path = constrained_best_path(log_probs)
        neg_logp = [float(-logp[t, path[t]].item()) for t in range(len(ids))]

        # convert BIES to spans
        spans: List[Tuple[int, int]] = []
        i = 0
        while i < len(chars):
            lab = path[i]
            if lab == S:
                spans.append((start + i, start + i + 1))
                i += 1
                continue
            # lab == B: consume until E
            if lab != B:
                # repair: treat as S
                spans.append((start + i, start + i + 1))
                i += 1
                continue
            j = i + 1
            while j < len(chars) and path[j] != E:
                j += 1
            if j >= len(chars):
                # no E found; fallback to remaining as one token
                spans.append((start + i, end))
                break
            spans.append((start + i, start + j + 1))
            i = j + 1

        # assign POS/feature using dictionary candidates for fixed spans if available
        if self._dict_ready:
            morphs = self._tag_fixed_spans(text, spans)
        else:
            morphs = [
                Morpheme(
                    surface=text[s:e],
                    pos="UNK",
                    left_id=0,
                    right_id=0,
                    word_cost=self.cfg.unk_penalty,
                    feature="",
                    source="UNK",
                )
                for s, e in spans
            ]
        toks: List[Token] = []
        cum_cost = 0.0
        for (s, e), m in zip(spans, morphs):
            surface = text[s:e]
            local_start = s - start
            local_end = e - start
            span_cost = sum(neg_logp[local_start:local_end])
            cum_cost += span_cost
            toks.append(Token(surface, m.pos, m.feature, s, e, cum_cost, "NEURAL"))
        return toks

    def tokenize(self, text: str) -> List[Token]:
        if self.cfg.force_neural:
            if self._neural is None:
                raise RuntimeError("force_neural requires a valid neural checkpoint")
            self._ensure_dict(required=False)
            return self._neural_segment(text, 0, len(text))
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

        # total_cost semantics:
        # - dict tokens: Viterbi cumulative cost
        # - neural tokens: segment-cumulative negative log-prob
        # These are not directly comparable across sources.
        return out
