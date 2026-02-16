from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Tuple
from .types import LatticeNode, Morpheme, Token
from .dict.connection import ConnectionCostMatrix
from .lattice import Lattice

@dataclass(frozen=True)
class _BackPtr:
    prev_pos: int
    prev_idx: int


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
    # dp[pos] -> (best_cost_to_reach_pos, via which node at prev start)
    # store per-node ending at pos; but traverse by start pos
    INF = 10**18
    # at each start pos i, consider edges to end on
    # store best cost arriving at pos i, plus backpointer ref the node used to reach i
    best_cost: List[int] = [INF] * (n + 1)
    best_prev: List[Optional[Tuple[int, LatticeNode]]] = [None] * (n + 1)
    # BOS state: treat as node ending at 0 with IDs = 0
    best_cost[0] = 0
    bos_right_id = 0

    # also keep right_id at pos i for best path ending there
    best_right_id: List[int] = [0] * (n + 1)
    best_right_id[0] = bos_right_id

    for i in range(n):
        if best_cost[i] >= INF:
            continue
        prev_right_id = best_right_id[i]
        for node in lattice.candidates_from(i):
            # transition cost + word cost
            tcost = conn.cost(prev_right_id, node.morph.left_id)
            new_cost = best_cost[i] + tcost + int(node.morph.word_cost)
            if new_cost < best_cost[node.end]:
                best_cost[node.end] = new_cost
                best_prev[node.end] = (i, node)
                best_right_id[node.end] = node.morph.right_id

    # EOS transition from n
    total = best_cost[n]
    if total >= INF or best_prev[n] is None:
        # graceful degradation: fallback is per-char tokens
        toks: List[Token] = []
        running = 0
        for i, ch in enumerate(text):
            running += 1
            toks.append(Token(ch, "UNK", "", i, i + 1, running, "UNK"))
        return ViterbiResult(toks, running)

    # reconstruct
    nodes_rev: List[LatticeNode] = []
    cur = n
    while cur > 0:
        prev = best_prev[cur]
        if prev is None:
            break
        i, node = prev
        nodes_rev.append(node)
        cur = i
    nodes_rev.reverse()

    # token conversion; include cumulative total cost at each token end for debugging
    toks: List[Token] = []
    cum = 0
    prev_right = bos_right_id
    for node in nodes_rev:
        cum += conn.cost(prev_right, node.morph.left_id) + int(node.morph.word_cost)
        toks.append(Token(
            surface=node.morph.surface,
            pos=node.morph.pos,
            feature=node.morph.feature,
            start=node.start,
            end=node.end,
            total_cost=cum,
            source=node.morph.source,
        ))
        prev_right = node.morph.right_id

    return ViterbiResult(toks, int(total))
