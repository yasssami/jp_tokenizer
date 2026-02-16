from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Tuple
from .types import LatticeNode, Morpheme, Token
from .dict.connection import ConnectionCostMatrix
from .lattice import Lattice

@dataclass
class ViterbiResult:
    tokens: List[Token]
    total_cost: int


def viterbi_decode(
    lattice: Lattice,
    conn: ConnectionCostMatrix,
) -> ViterbiResult:
    text = lattice.text
    n = len(text)
    INF = 10**18
    # build node idx for per-node dp (bigram trans cost)
    nodes: List[LatticeNode] = []
    nodes_by_start: List[List[int]] = [[] for _ in range(n + 1)]
    nodes_by_end: List[List[int]] = [[] for _ in range(n + 1)]
    for i in range(n):
        for node in lattice.candidates_from(i):
            idx = len(nodes)
            nodes.append(node)
            nodes_by_start[i].append(idx)
            nodes_by_end[node.end].append(idx)

    if not nodes:
        # graceful degradation: fallback is per-char tokens
        toks: List[Token] = []
        running = 0
        for i, ch in enumerate(text):
            running += 1
            toks.append(Token(ch, "UNK", "", i, i + 1, running, "UNK"))
        return ViterbiResult(toks, running)

    best_cost: List[int] = [INF] * len(nodes)
    best_prev: List[Optional[int]] = [None] * len(nodes)

    bos_right_id = conn.bos_right_id()
    # init nodes starting at 0 from BOS
    for idx in nodes_by_start[0]:
        node = nodes[idx]
        cost = conn.cost(bos_right_id, node.morph.left_id) + int(node.morph.word_cost)
        best_cost[idx] = cost
        best_prev[idx] = None

    # dp fwd by start pos
    for i in range(1, n + 1):
        if not nodes_by_start[i]:
            continue
        prev_nodes = nodes_by_end[i]
        if not prev_nodes:
            continue
        for idx in nodes_by_start[i]:
            node = nodes[idx]
            best = INF
            best_prev_idx: Optional[int] = None
            for pidx in prev_nodes:
                prev_cost = best_cost[pidx]
                if prev_cost >= INF:
                    continue
                prev_node = nodes[pidx]
                tcost = conn.cost(prev_node.morph.right_id, node.morph.left_id)
                cand = prev_cost + tcost + int(node.morph.word_cost)
                if cand < best:
                    best = cand
                    best_prev_idx = pidx
            best_cost[idx] = best
            best_prev[idx] = best_prev_idx

    # EOS transition from n
    eos_left_id = conn.eos_left_id()
    total = INF
    last_idx: Optional[int] = None
    for idx in nodes_by_end[n]:
        cost = best_cost[idx]
        if cost >= INF:
            continue
        node = nodes[idx]
        total_cost = cost + conn.cost(node.morph.right_id, eos_left_id)
        if total_cost < total:
            total = total_cost
            last_idx = idx

    if total >= INF or last_idx is None:
        # graceful degradation: fallback is per-char tokens
        toks = []
        running = 0
        for i, ch in enumerate(text):
            running += 1
            toks.append(Token(ch, "UNK", "", i, i + 1, running, "UNK"))
        return ViterbiResult(toks, running)

    # reconstruct best path
    nodes_rev: List[int] = []
    cur: Optional[int] = last_idx
    while cur is not None:
        nodes_rev.append(cur)
        cur = best_prev[cur]
    nodes_rev.reverse()

    # token conversion; include cumulative total cost at each token end
    toks: List[Token] = []
    for idx in nodes_rev:
        node = nodes[idx]
        toks.append(Token(
            surface=node.morph.surface,
            pos=node.morph.pos,
            feature=node.morph.feature,
            start=node.start,
            end=node.end,
            total_cost=int(best_cost[idx]),
            source=node.morph.source,
        ))

    return ViterbiResult(toks, int(total))
