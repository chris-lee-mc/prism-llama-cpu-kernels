"""Speculative decoding bookkeeping shared by e5_cost, e5_spec and r1 (pure functions, stdlib only).

Two acceptance numbers are in use and they differ a lot:
  r      llama.cpp acceptance: draft_n_accepted / draft_n, over all drafted tokens of a run.
  alpha  per-token acceptance of the independence model: each drafted token is accepted with probability alpha,
         and a step accepts the longest accepted prefix. A step that drafts K tokens accepts on average
             A(alpha, K) = alpha + alpha^2 + ... + alpha^K = alpha (1 - alpha^K) / (1 - alpha)
         and emits 1 + A tokens (the target adds one token every step).
Conversion: r = A(alpha, K) / K, so alpha_from_r(r, K) solves A(alpha, K) = r K. Example: K = 4, r = 0.216 gives alpha 0.48.
r falls with K at fixed alpha (later drafts need the whole prefix right), alpha does not, so compare alpha across K.
Measured runs: steps = tokens - accepted - n_first (every step emits accepted + 1 tokens; the first token of each request
comes from prefill, not from a decode step, so n_first = number of requests), K_eff = drafted / steps.
alpha only means something for draft-simple, where every step drafts about K tokens; n-gram types draft variable lengths
(often 0), so callers report r and tokens per step for them and leave alpha out.
"""
from __future__ import annotations

import random


def accept_sum(alpha: float, k: float) -> float:
    """A(alpha, K): expected accepted draft tokens per step. k may be fractional (mean drafted tokens per step)."""
    if k <= 0 or alpha <= 0.0:
        return 0.0
    if alpha >= 1.0:
        return float(k)
    return alpha * (1.0 - alpha ** k) / (1.0 - alpha)


def tokens_per_step(alpha: float, k: float) -> float:
    return 1.0 + accept_sum(alpha, k)


def r_from_alpha(alpha: float, k: float) -> float:
    return accept_sum(alpha, k) / k if k > 0 else 0.0


def alpha_from_r(r: float | None, k: float) -> float | None:
    """Invert r_from_alpha by bisection (A is increasing in alpha). None for missing input; r >= 1 gives 1.0."""
    if r is None or k is None or k <= 0:
        return None
    if r <= 0.0:
        return 0.0
    if r >= 1.0:
        return 1.0
    target = r * k
    lo, hi = 0.0, 1.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if accept_sum(mid, k) < target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def steps_from_counts(tokens: int, accepted: int, n_first: int = 0) -> int:
    """Decode steps of a run: each emits accepted + 1 tokens, so steps = tokens - accepted - n_first (at least 0).
    n_first: requests in the sum, each has one token that prefill sampled."""
    return max(int(tokens) - int(accepted) - int(n_first), 0)


def run_stats(tokens: int, drafted: int, accepted: int, n_first: int = 0, with_alpha: bool = True) -> dict:
    """r, alpha, steps, K_eff and tokens/step for summed counts of one config. Fields are None when undefined.
    with_alpha False (n-gram types) leaves alpha and K_eff out: their drafts vary in length."""
    steps = steps_from_counts(tokens, accepted, n_first)
    k_eff = drafted / steps if steps else None
    r = accepted / drafted if drafted else None
    alpha = alpha_from_r(r, k_eff) if (with_alpha and r is not None and k_eff) else None
    return {"r": r, "alpha": alpha, "steps": steps, "k_eff": k_eff if with_alpha else None,
            "tokens_per_step": (tokens - n_first) / steps if steps else None}


def break_even_alpha(k: int, t_draft_ms: float, t_verify_ms: float, t_plain_ms: float, t_draft_batch_ms: float = 0.0) -> float | None:
    """Smallest alpha (step 0.001) where K drafts plus one verify of K + 1 tokens match plain decode; None if never.

    Speed of a step is (1 + A) / (k t_draft + t_draft_batch + t_verify); plain decode is 1 / t_plain. t_draft_batch is the
    draft re-eval of the K + 1 verify batch that llama.cpp draft-simple runs on the draft context every step (process()).
    """
    t_step = k * t_draft_ms + t_draft_batch_ms + t_verify_ms
    for i in range(1001):
        a = i / 1000.0
        if tokens_per_step(a, k) / t_step >= 1.0 / t_plain_ms:
            return a
    return None


def observed_step_ms(predicted_ms: float, tokens: int, accepted: int, n_first: int = 0) -> float | None:
    """Mean wall time of one decode step from the server timings of a run."""
    steps = steps_from_counts(tokens, accepted, n_first)
    return predicted_ms / steps if steps else None


def bootstrap_speedup_ci(pairs: list[tuple[float, float, float, float]], n_boot: int = 1000, seed: int = 20261013,
                         level: float = 0.95) -> tuple[float, float] | None:
    """Percentile bootstrap CI of a throughput ratio. pairs: per prompt (tokens_cfg, ms_cfg, tokens_base, ms_base).
    Resamples prompts with replacement; ratio = (sum tokens_cfg / sum ms_cfg) / (sum tokens_base / sum ms_base). None if < 2 pairs."""
    n = len(pairs)
    if n < 2:
        return None
    rng = random.Random(seed)
    vals = []
    for _ in range(n_boot):
        tc = mc = tb = mb = 0.0
        for i in (rng.randrange(n) for _ in range(n)):
            a, b, c, d = pairs[i]
            tc, mc, tb, mb = tc + a, mc + b, tb + c, mb + d
        if mc > 0 and mb > 0 and tb > 0:
            vals.append((tc / mc) / (tb / mb))
    if not vals:
        return None
    vals.sort()
    lo, hi = (1.0 - level) / 2.0, 1.0 - (1.0 - level) / 2.0
    return vals[int(lo * (len(vals) - 1))], vals[int(round(hi * (len(vals) - 1)))]


def common_prefix_len(a: str, b: str) -> int:
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i
