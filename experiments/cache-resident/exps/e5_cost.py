"""E5 step cost: target verify cost per batch width and draft decode cost, from llama-bench.

  python -m exps.e5_cost [--widths 1,2,3,4,5,6,7,8,9] [--depth 512] [--reps 5]

Target (Bonsai 2 27B PQ2_0): llama-bench -p W -n 0 -d DEPTH for each width W; one batch of W tokens at depth DEPTH
is what a K = W - 1 verify step costs. Drafts (Qwen3.5-0.8B BF16 and Q8_0): llama-bench -p 0 -n 128 -d DEPTH, the cost
of one draft token. Same flags as E4 (-fa on, f16 KV, -ub 2048). Row: {model, kind, width, depth, tok_s, ms_per_batch}.
The summary gives the speculative step model: t_step(K) = K * t_draft + t_verify(K + 1), and the acceptance a at which
E[tokens/step] / t_step(K) equals the plain decode rate, for each K, with E = (1 - a^(K+1)) / (1 - a).
"""
from __future__ import annotations

import argparse
import sys

from exps import common, e4_bonsai, e4_pdl

EXP = "e5_cost"
DRAFTS = ("draft_bf16", "draft_q8_0")


def target_cmd(bin_dir, model: str, widths: list[int], depth: int, reps: int) -> list[str]:
    cmd = e4_bonsai.bench_cmd(bin_dir, model, depth, "f16", reps, 0)
    cmd[cmd.index("-p") + 1] = ",".join(str(w) for w in widths)
    return cmd


def draft_cmd(bin_dir, model: str, depth: int, reps: int) -> list[str]:
    return e4_bonsai.bench_cmd(bin_dir, model, depth, "f16", reps, e4_bonsai.N_GEN)


def rows_from(entries: list[dict], model: str, depth: int) -> list[dict]:
    out = []
    for e in entries:
        n_p, n_g, ts = int(e.get("n_prompt", 0) or 0), int(e.get("n_gen", 0) or 0), float(e.get("avg_ts", 0) or 0)
        if not ts:
            continue
        kind, width = ("verify", n_p) if n_p > 0 else ("decode", 1)
        out.append({"model": model, "kind": kind, "width": width, "depth": depth, "tok_s": ts,
                    "stddev": float(e.get("stddev_ts", 0) or 0), "ms_per_batch": 1e3 * width / ts, "n_gen": n_g})
    return out


def break_even(t_draft_ms: float, t_verify_ms: dict, t_plain_ms: float, k: int) -> float | None:
    """Smallest acceptance a (0..1, step 0.001) at which spec decoding with K = k matches plain decode; None if never."""
    tv = t_verify_ms.get(k + 1)
    if tv is None:
        return None
    t_step = k * t_draft_ms + tv
    for i in range(1001):
        a = i / 1000.0
        e = k + 1.0 if a >= 1.0 else (1.0 - a ** (k + 1)) / (1.0 - a)
        if e / t_step >= 1.0 / t_plain_ms:
            return a
    return None


def summarize(rows: list[dict], ks=(4, 6, 8)) -> dict:
    tv = {r["width"]: r["ms_per_batch"] for r in rows if r["model"] == "pq2" and r["kind"] == "verify"}
    out = {"target_verify_ms": tv, "drafts": {}}
    plain = tv.get(1)
    for d in DRAFTS:
        td = next((r["ms_per_batch"] for r in rows if r["model"] == d and r["kind"] == "decode"), None)
        if td is None or plain is None:
            continue
        out["drafts"][d] = {"t_draft_ms": td, "break_even_acceptance": {k: break_even(td, tv, plain, k) for k in ks},
                            "t_step_ms": {k: k * td + tv[k + 1] for k in ks if k + 1 in tv}}
    # a free draft (t_draft = 0) bounds what any draft can reach: verify cost alone
    if plain:
        out["free_draft_break_even"] = {k: break_even(0.0, tv, plain, k) for k in ks}
    return out


def run(args) -> int:
    notes: list[str] = []
    args.gpu = common.gpu_tag(args)
    bin_dir = e4_bonsai.llama_bin()
    if not (bin_dir / "llama-bench").exists():
        print(f"error: {bin_dir / 'llama-bench'} not found", file=sys.stderr)
        return 2
    budget = common.Budget(args.budget_min or 10)
    config = {"widths": args.widths, "depth": args.depth, "reps": args.reps, "models_info": {},
              "target_cmd": target_cmd(bin_dir, "<pq2.gguf>", args.widths, args.depth, args.reps),
              "draft_cmd": draft_cmd(bin_dir, "<draft.gguf>", args.depth, args.reps)}
    rows: list[dict] = []
    for m in ("pq2",) + DRAFTS:
        if budget.expired(60):
            notes.append(f"time budget used up; {m} skipped")
            continue
        try:
            path, config["models_info"][m] = e4_pdl.model_path(m, notes)
        except Exception as e:
            notes.append(f"{m}: model download failed: {type(e).__name__}: {e}")
            continue
        cmd = target_cmd(bin_dir, path, args.widths, args.depth, args.reps) if m == "pq2" else draft_cmd(bin_dir, path, args.depth, args.reps)
        with e4_pdl.SmiWatch() as watch:
            got, err = e4_bonsai.run_bench_entries(cmd, budget.timeout(900, reserve_s=30))
        if err:
            notes.append(f"{m} bench failed: {err}")
        new = rows_from(got, m, args.depth)
        for r in new:
            r["smi"] = watch.summary()
        rows += new
        common.write_partial(EXP, args, config, rows, notes, False)
        print("[e5_cost] %s %s" % (m, {r["width"]: round(r["ms_per_batch"], 3) for r in new}), flush=True)
    summ = summarize(rows)
    common.write_result(EXP, args, config, rows, notes, False, extra={"summary": summ})
    return 0 if rows else 1


def _ints(text):
    return [int(t) for t in text.split(",") if t.strip()]


def parse_args(argv=None):
    ap = argparse.ArgumentParser(prog="exps.e5_cost", description=__doc__.split("\n")[0])
    common.add_common_args(ap)
    ap.add_argument("--widths", type=_ints, default=list(range(1, 10)))
    ap.add_argument("--depth", type=int, default=512)
    ap.add_argument("--reps", type=int, default=5)
    return ap.parse_args(argv)


def main(argv=None) -> int:
    a = parse_args(argv)
    if a.dry_run:
        bin_dir = e4_bonsai.llama_bin()
        common.print_plan({"exp": EXP, "target_cmd": target_cmd(bin_dir, "<pq2.gguf>", a.widths, a.depth, a.reps),
                           "draft_cmd": draft_cmd(bin_dir, "<draft.gguf>", a.depth, a.reps)})
        return 0
    with common.term_exits():
        return run(a)


if __name__ == "__main__":
    sys.exit(main())
