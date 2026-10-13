"""E5 step cost: target verify cost per batch width, plain target decode and draft decode cost, from llama-bench.

  python -m exps.e5_cost [--widths 1,2,3,4,5,6,7,8,9] [--depth 512] [--reps 5]
  python -m exps.e5_cost --resummarize RESULT.json [--plain-from E5_BASE_RESULT.json | --plain-ms MS]

Target (Bonsai 2 27B PQ2_0): llama-bench -p W -n 0 -d DEPTH for each width W; one batch of W tokens at depth DEPTH
is what a K = W - 1 verify step costs. The plain decode baseline is the target's own tg run (-p 0 -n 128 -d DEPTH);
the width-1 prompt path is a different code path and is NOT the baseline (it was 8.64 ms against 7.2 ms for tg).
Drafts (Qwen3.5-0.8B BF16 and Q8_0): llama-bench -p 0 -n 128 -d DEPTH, the cost of one draft token. Same flags as E4
(-fa on, f16 KV, -ub 2048). Row: {model, kind, width, depth, tok_s, ms_per_batch, samples_ts}; tok_s is the median of
the per-rep samples (samples_ts, from llama-bench), tok_s_mean the mean.
The summary gives the step model t_step(K) = K * t_draft + t_verify(K + 1) and, per K, the break-even acceptance at
which (1 + A) / t_step equals 1 / t_plain. Acceptance is reported both as alpha (per-token, independence model) and r
(llama.cpp draft_n_accepted / draft_n); see exps/spec_math.py for the conversion r = A(alpha, K) / K.
--resummarize rebuilds the summary from a saved result without a GPU. Old results have no tg row for the target, so the
baseline comes from --plain-from (tok_s_weighted of an E5 target-alone result) or --plain-ms, and the output says so.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import pathlib
import sys

from exps import common, e4_bonsai, e4_pdl, spec_math

EXP = "e5_cost"
DRAFTS = ("draft_bf16", "draft_q8_0")
KS = (2, 3, 4, 6, 8)


def target_cmd(bin_dir, model: str, widths: list[int], depth: int, reps: int) -> list[str]:
    cmd = e4_bonsai.bench_cmd(bin_dir, model, depth, "f16", reps, 0)
    cmd[cmd.index("-p") + 1] = ",".join(str(w) for w in widths)
    return cmd


def draft_cmd(bin_dir, model: str, depth: int, reps: int) -> list[str]:
    return e4_bonsai.bench_cmd(bin_dir, model, depth, "f16", reps, e4_bonsai.N_GEN)


plain_cmd = draft_cmd  # same tg run for the target: the plain decode baseline


def rows_from(entries: list[dict], model: str, depth: int) -> list[dict]:
    """llama-bench entries -> rows. tok_s is the median of the per-rep samples (mean if there are none)."""
    out = []
    for e in entries:
        n_p, n_g = int(e.get("n_prompt", 0) or 0), int(e.get("n_gen", 0) or 0)
        samples = [float(x) for x in (e.get("samples_ts") or []) if x]
        mean = float(e.get("avg_ts", 0) or 0)
        ts = common.median(samples) if samples else mean
        if not ts:
            continue
        kind, width = ("verify", n_p) if n_p > 0 else ("decode", 1)
        out.append({"model": model, "kind": kind, "width": width, "depth": depth, "tok_s": ts, "tok_s_mean": mean,
                    "samples_ts": samples, "stddev": float(e.get("stddev_ts", 0) or 0), "ms_per_batch": 1e3 * width / ts,
                    "n_gen": n_g})
    return out


def break_even(t_draft_ms: float, t_verify_ms: dict, t_plain_ms: float, k: int) -> float | None:
    """Break-even per-token acceptance alpha for K = k (0..1, step 0.001); None if never or no verify row for k + 1."""
    tv = t_verify_ms.get(k + 1)
    return None if tv is None else spec_math.break_even_alpha(k, t_draft_ms, tv, t_plain_ms)


def _be(alpha: float | None, k: int) -> dict:
    return {"alpha": alpha, "r": None if alpha is None else spec_math.r_from_alpha(alpha, k)}


def summarize(rows: list[dict], ks=KS, plain_ms: float | None = None, plain_source: str | None = None) -> dict:
    """Step model and break-evens. Baseline: the target tg row; else plain_ms (with plain_source); else none (no break-evens)."""
    tv = {r["width"]: r["ms_per_batch"] for r in rows if r["model"] == "pq2" and r["kind"] == "verify"}
    tg = next((r for r in rows if r["model"] == "pq2" and r["kind"] == "decode"), None)
    plain, source = None, None
    if tg:
        plain, source = tg["ms_per_batch"], "llama-bench tg (-p 0 -n %d) on the target" % (tg.get("n_gen") or e4_bonsai.N_GEN)
    elif plain_ms:
        plain, source = plain_ms, plain_source or "supplied"
    out = {"baseline": {"plain_ms": plain, "tok_s": 1e3 / plain if plain else None, "source": source},
           "target_verify_ms": tv, "target_verify_w1_prompt_path_ms": tv.get(1), "drafts": {},
           "conversion": "r = A(alpha, K) / K with A = alpha (1 - alpha^K) / (1 - alpha); steps emit 1 + A tokens"}
    if plain is None:
        out["error"] = "no plain decode baseline (no target tg row, no --plain-from/--plain-ms): no break-even computed"
        return out
    for d in DRAFTS:
        td = next((r["ms_per_batch"] for r in rows if r["model"] == d and r["kind"] == "decode"), None)
        if td is None:
            continue
        out["drafts"][d] = {"t_draft_ms": td, "break_even": {k: _be(break_even(td, tv, plain, k), k) for k in ks},
                            "t_step_ms": {k: k * td + tv[k + 1] for k in ks if k + 1 in tv}}
    # a free draft (t_draft = 0) bounds what any draft can reach: verify cost alone
    out["free_draft_break_even"] = {k: _be(break_even(0.0, tv, plain, k), k) for k in ks}
    return out


def plain_ms_from_e5(path: str | pathlib.Path) -> tuple[float, str]:
    """(ms per token, source text) from an E5 target-alone result: 1000 / summary.tok_s_weighted."""
    doc = json.loads(pathlib.Path(path).read_text())
    s = doc["data"]["summary"]
    ts = s.get("tok_s_weighted")
    if not ts:
        raise ValueError("no tok_s_weighted in %s" % path)
    return 1e3 / ts, "E5 target-alone server run %s (%d prompts, tok_s_weighted %.2f)" % (
        pathlib.Path(path).name, s.get("n_ok", 0), ts)


def resummarize(src: str | pathlib.Path, plain_ms: float | None = None, plain_source: str | None = None) -> dict:
    """Rebuild the summary of a saved e5_cost result. Returns the document to write next to it."""
    src = pathlib.Path(src)
    doc = json.loads(src.read_text())
    data = doc["data"]
    rows = data["rows"]
    notes = ["resummary of %s: per-rep samples were not saved by that run, so tok_s there is llama-bench avg_ts "
             "(mean of %s reps), not a median" % (src.name, (data.get("config") or {}).get("reps"))]
    summ = summarize(rows, plain_ms=plain_ms, plain_source=plain_source)
    if summ["baseline"]["source"] and "tg" not in summ["baseline"]["source"]:
        notes.append("baseline is not from this host's llama-bench tg run: depth differs (E5 prompts are short, rows here are "
                     "at depth %s) and it is a server run" % (rows[0].get("depth") if rows else "?"))
    return {"schema": doc.get("schema", 1), "exp": doc.get("exp", EXP), "gpu": doc.get("gpu"),
            "utc": dt.datetime.now(dt.timezone.utc).isoformat(), "derived_from": src.name, "kind": "resummary",
            "data": {"rows": rows, "notes": notes, "summary": summ}}


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
              "plain_cmd": plain_cmd(bin_dir, "<pq2.gguf>", args.depth, args.reps),
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
        cmds = [draft_cmd(bin_dir, path, args.depth, args.reps)]
        if m == "pq2":  # plain tg first: it is the baseline of every break-even
            cmds = [plain_cmd(bin_dir, path, args.depth, args.reps), target_cmd(bin_dir, path, args.widths, args.depth, args.reps)]
        for cmd in cmds:
            with e4_pdl.SmiWatch() as watch:
                got, err = e4_bonsai.run_bench_entries(cmd, budget.timeout(900, reserve_s=30))
            if err:
                notes.append(f"{m} bench failed: {err}")
            new = rows_from(got, m, args.depth)
            for r in new:
                r["smi"] = watch.summary()
            rows += new
            common.write_partial(EXP, args, config, rows, notes, False)
            print("[e5_cost] %s %s" % (m, {(r["kind"], r["width"]): round(r["ms_per_batch"], 3) for r in new}), flush=True)
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
    ap.add_argument("--resummarize", metavar="RESULT.json", default=None,
                    help="rebuild the summary of a saved result offline (writes RESULT.resummary.json) and exit")
    ap.add_argument("--plain-from", metavar="E5_BASE.json", default=None, help="plain decode baseline from an E5 target-alone result")
    ap.add_argument("--plain-ms", type=float, default=None, help="plain decode baseline in ms per token")
    return ap.parse_args(argv)


def resummarize_main(a) -> int:
    pm, src = a.plain_ms, "supplied via --plain-ms"
    if a.plain_from:
        pm, src = plain_ms_from_e5(a.plain_from)
    doc = resummarize(a.resummarize, pm, src if pm else None)
    out = pathlib.Path(a.resummarize).with_suffix(".resummary.json")
    out.write_text(json.dumps(doc, indent=1, default=str, allow_nan=False) + "\n", encoding="ascii")
    print(f"[e5_cost] wrote {out}")
    return 0 if "error" not in doc["data"]["summary"] else 1


def main(argv=None) -> int:
    a = parse_args(argv)
    if a.resummarize:
        return resummarize_main(a)
    if a.dry_run:
        bin_dir = e4_bonsai.llama_bin()
        common.print_plan({"exp": EXP, "target_cmd": target_cmd(bin_dir, "<pq2.gguf>", a.widths, a.depth, a.reps),
                           "plain_cmd": plain_cmd(bin_dir, "<pq2.gguf>", a.depth, a.reps),
                           "draft_cmd": draft_cmd(bin_dir, "<draft.gguf>", a.depth, a.reps)})
        return 0
    with common.term_exits():
        return run(a)


if __name__ == "__main__":
    sys.exit(main())
