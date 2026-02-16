from __future__ import annotations
from typing import List

B, I, E, S = 0, 1, 2, 3

ALLOWED_NEXT = {
    B: [I, E],
    I: [I, E],
    E: [B, S],
    S: [B, S],
}

ALLOWED_START = [B, S]
ALLOWED_END = [E, S]


def constrained_best_path(log_probs: List[List[float]]) -> List[int]:
    """
    find best valid B/I/E/S sequence under transition constraints,
    using DP over log probabilities
    log_probs[t][k] = log p(label=k at position t)
    """
    T = len(log_probs)
    if T == 0:
        return []

    NEG = -1e30
    dp = [[NEG] * 4 for _ in range(T)]
    bp = [[-1] * 4 for _ in range(T)]

    for k in range(4):
        dp[0][k] = log_probs[0][k] if k in ALLOWED_START else NEG

    for t in range(1, T):
        for k in range(4):
            best = NEG
            best_prev = -1
            for pk in range(4):
                if k in ALLOWED_NEXT.get(pk, []):
                    cand = dp[t - 1][pk] + log_probs[t][k]
                    if cand > best:
                        best = cand
                        best_prev = pk
            dp[t][k] = best
            bp[t][k] = best_prev

    # end constraint
    best_end = max(ALLOWED_END, key=lambda k: dp[T - 1][k])
    if dp[T - 1][best_end] <= NEG / 2:
        # fallback to greedy S everywhere
        return [S] * T

    out = [0] * T
    out[T - 1] = best_end
    for t in range(T - 1, 0, -1):
        out[t - 1] = bp[t][out[t]]
        if out[t - 1] < 0:
            out[t - 1] = S
    # ensure first is valid
    if out[0] not in ALLOWED_START:
        out[0] = S
    return out
