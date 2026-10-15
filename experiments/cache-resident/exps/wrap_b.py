"""Wrap B: test build for two upstream investigations (measure first, issue before code), one 600 W PRO 6000 pod session.

  python -m exps.fetch_models --models pq2,draft_q8_0 --timeout-min 6 && python -m exps.wrap_b [--parts r,m] [--budget-min 24]

Experiment-only switches in this branch (env, default off, read once):
- GGML_EXP_SKIP_DRAFT_REEVAL=1 (B1, upstream TODO TAG_SPEC_AVOID_DRAFT_REEVAL): the server keeps the drafted tokens in the draft context
  instead of removing them after drafting; draft-simple process() skips a verify batch whose first token the draft context already
  holds; the partial-acceptance rollback is the usual seq_rm (n_rs_seq window); draft() decodes the tokens the draft context misses
  (the last draft token after a full acceptance) in one batch with id_last.
- GGML_EXP_DRAFT_CKPT=1 (control for B1): the server treats the draft context as FULL, so it saves the draft state before drafting and
  restores it after (the existing checkpoint path). This is the draft state without any rollback after single-token decodes.
- GGML_EXP_NGRAM_RS=N (B2): need_n_rs_seq() returns at least N when an n-gram type is in --spec-type, so the target takes the in-batch
  (RS) rollback instead of checkpoint + replay; n-gram drafts longer than N are cut to N (counted, printed with the statistics).
Why the control: the recurrent snapshot of a GDN layer is written only for the tokens of the last ubatch (gated_delta_net: "older slots
are caller-owned"), so seq_rm after one-token decodes (draft-simple drafting) restores a stale snapshot. Part r measures this directly.
Gate: the GPU tag must be pro6000-ws or pro6000-server and the default power limit >= 550 W (as wrap_a); --allow-any-gpu for development.
r  rollback probe (exps/csrc/cr_rs_rollback_probe.cpp, built against libllama of the pod build): prompt of 64 tokens, then 4 tokens
   decoded one at a time or as one batch, seq_rm back to 0-3 kept tokens, one probe token; logits against a context that decoded only
   the kept tokens the same way. On the Qwen3.5-0.8B Q8_0 draft and on the Bonsai 2 27B PQ2_0 target, n_rs_seq 4. Untimed.
m  matrix, the R2a balanced subset (24 prompts, 8 per category, same seed, ctx 16384, -lv 4), interleaved by prompt blocks (--blocks,
   default 4, a fresh server per block and config, config order changes per block):
   base        target alone
   k4q8        draft-simple K=4, Qwen3.5-0.8B Q8_0 draft (the CONFIG-GUIDANCE setting)
   k4q8_skip   k4q8 with GGML_EXP_SKIP_DRAFT_REEVAL=1 (B1)
   k4q8_ckpt   k4q8 with GGML_EXP_DRAFT_CKPT=1 (control: draft state restored from a checkpoint)
   ngram       --spec-type ngram-mod, fork defaults (checkpoint + replay on the hybrid target)
   ngram_rs16  ngram with GGML_EXP_NGRAM_RS=16 (drafts cut to 16)
   ngram_rs64  ngram with GGML_EXP_NGRAM_RS=64 (no cut at the fork default n_max 64; 65 recurrent snapshots of the target)
   Reported: per config and category tok/s, speedup over base with bootstrap 95% CIs (prompts paired), r, alpha, greedy text equality
   vs base; the pairs B1 vs k4q8, control vs k4q8, B1 vs control, rs16 / rs64 vs ngram per category with paired bootstrap CIs; ms per
   step and its split from the server statistics (dur g of the drafting implementation vs the rest); path verification from the -lv 4
   logs (n_rs_seq of the target and the draft, rollback path, the GGML_EXP_* switch lines, the cut counters).
   ngram-mod keeps its n-gram pool per server: with a fresh server per block it sees fewer earlier prompts than in R2a (same for every
   ngram config, so the pairs stay fair).
Parts run in the order r, m with weights 2 : 22 of --budget-min. An unhealthy host skips m (r still runs).
Result: results/wrap_b/<gpu>/<stamp>.json with data.gate, data.probe, data.matrix (runs, summary, pairs, paths), data.host.
Time: failsafe 55 min = bootstrap with the cold build ~14, downloads <= 6, wrap_b 24, reserve 5 (the matrix needs about 16).
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import subprocess
import sys

from exps import common, e4_bonsai, e4_pdl, hf_ids, make_prompts, r1, r2a_longctx as r2a, r2b_multicol as mc, r2b_spec as sp

EXP = "wrap_b"
SKIP, CKPT, NGRAM_RS = "GGML_EXP_SKIP_DRAFT_REEVAL", "GGML_EXP_DRAFT_CKPT", "GGML_EXP_NGRAM_RS"
CONFIG_NAMES = ["base", "k4q8", "k4q8_skip", "k4q8_ckpt", "ngram", "ngram_rs16", "ngram_rs64"]
PAIRS = [("k4q8_skip", "k4q8"), ("k4q8_ckpt", "k4q8"), ("k4q8_skip", "k4q8_ckpt"), ("ngram_rs16", "ngram"), ("ngram_rs64", "ngram"),
         ("ngram_rs64", "ngram_rs16")]
BLOCKS = 4
PARTS = "rm"
WEIGHTS = {"r": 2.0, "m": 22.0}
DEFAULT_TOTAL_MIN = 24.0
END_RESERVE_S = 25.0
PROBE_SRC = common.EXP_DIR / "exps" / "csrc" / "cr_rs_rollback_probe.cpp"
PROBE_ARGS = ["--n-rs", "4", "--prompt", "64", "--extra", "4"]
REPO_ROOT = common.EXP_DIR.parent.parent


def matrix_configs(names=CONFIG_NAMES) -> list[dict]:
    by = {c["name"]: c for c in sp.matrix_configs(["base", "k4_q8"])}
    k, ng = by["k4_q8"], r2a.matrix_configs(["ngram_mod"])[0]
    allc = {
        "base": dict(by["base"], env={}),
        "k4q8": dict(k, name="k4q8", env={}, desc="draft-simple K=4, Qwen3.5-0.8B Q8_0 draft"),
        "k4q8_skip": dict(k, name="k4q8_skip", env={SKIP: "1"}, desc="k4q8, the draft context keeps its drafted tokens (no re-eval)"),
        "k4q8_ckpt": dict(k, name="k4q8_ckpt", env={CKPT: "1"}, desc="k4q8, draft state restored from a checkpoint after drafting"),
        "ngram": dict(ng, name="ngram", env={}, desc="ngram-mod, fork defaults (checkpoint path on the hybrid target)"),
        "ngram_rs16": dict(ng, name="ngram_rs16", env={NGRAM_RS: "16"}, desc="ngram-mod, in-batch rollback window 16, drafts cut to 16"),
        "ngram_rs64": dict(ng, name="ngram_rs64", env={NGRAM_RS: "64"}, desc="ngram-mod, in-batch rollback window 64"),
    }
    return [allc[n] for n in names]


def expected_path(cfg: dict) -> dict:
    """What the -lv 4 log must show for a config: rollback path, n_rs_seq of the target, the switch line of the server."""
    env = cfg.get("env") or {}
    if cfg["spec_type"] == "none":
        return {"rollback_path": "none", "n_rs_seq_target": 0, "switches": []}
    if sp.has_draft_model(cfg):
        return {"rollback_path": "in_batch", "n_rs_seq_target": int(cfg["k"]), "switches": [k for k in (SKIP, CKPT) if k in env]}
    n = int(env.get(NGRAM_RS, 0))
    return {"rollback_path": "in_batch" if n else "checkpoint", "n_rs_seq_target": n, "switches": []}


def verify_paths(cfgs: list[dict], runs: dict) -> dict:
    out = {}
    for c in cfgs:
        run = runs.get(c["name"]) or {}
        want = expected_path(c)
        seen = []
        for bl in run.get("blocks", []):
            ck = bl.get("checks") or {}
            st = bl.get("stats") or {}
            seen.append({"block": bl["block"], "rollback_path": ck.get("rollback_path"), "n_rs_seq": ck.get("n_rs_seq"),
                         "n_rs_seq_target": ck.get("n_rs_seq_target"), "switches": bl.get("exp_switches", []),
                         "rs_window": next((d["rs_window"] for d in st.values() if "rs_window" in d), None)})
        ok = bool(seen) and all(s["rollback_path"] == want["rollback_path"] and (s["n_rs_seq_target"] or 0) == want["n_rs_seq_target"]
                                and sorted(s["switches"]) == sorted(want["switches"]) for s in seen)
        n = int((c.get("env") or {}).get(NGRAM_RS, 0))
        if n:
            ok = ok and all(s["rs_window"] == n for s in seen)
        out[c["name"]] = {"expected": want, "seen": seen, "ok": ok}
    return out


def matrix_summary(cfgs: list[dict], runs: dict) -> dict:
    st = {n: r.get("stats") or {} for n, r in runs.items()}
    cut = {n: {impl: {k: d.get(k) for k in ("cut_drafts", "cut_tokens", "gen_drafts", "gen_tokens", "rs_window") if k in d}
               for impl, d in s.items() if "cut_drafts" in d} for n, s in st.items()}
    skip = {n: {impl: {k: d.get(k) for k in ("skipped_batches", "catchup_tokens", "calls_g") if k in d}
                for impl, d in s.items() if "skipped_batches" in d} for n, s in st.items()}
    return {"vs_base": r2a.summarize(runs, ci=True), "pairs": {f"{a}_vs_{b}": sp.pair_by_category(runs, a, b) for a, b in PAIRS
                                                                if a in runs and b in runs},
            "step_split": {n: r.get("step_split") for n, r in runs.items()}, "paths": verify_paths(cfgs, runs),
            "ngram_cut": {n: v for n, v in cut.items() if v}, "skip_reeval": {n: v for n, v in skip.items() if v}}


def verdict_lines(summ: dict) -> list[str]:
    out = []
    for key, p in (summ.get("pairs") or {}).items():
        a = (p or {}).get("all") or {}
        if a.get("speedup"):
            out.append("%s: %.3fx (95%% CI %s), alpha %s vs %s, r %s, text equal %s/%s" % (
                key, a["speedup"], _ci(a.get("speedup_ci95")), _f(a.get("alpha")), _f(a.get("base_alpha")), _f(a.get("r")), a.get("text_equal"),
                a.get("text_compared")))
    bad = [n for n, v in (summ.get("paths") or {}).items() if not v["ok"]]
    if bad:
        out.append(f"path verification failed for {bad}: read data.matrix.summary.paths before any number")
    return out


def _f(x):
    return "n/a" if x is None else "%.3f" % x


def _ci(ci):
    return "n/a" if not ci else "[%.3f, %.3f]" % tuple(ci)


# ---------------------------------------------------------------- part r

def build_probe(bin_dir: pathlib.Path, notes: list) -> pathlib.Path | None:
    """Compile cr_rs_rollback_probe against libllama in bin_dir (as e4_bonsai.build_prof_tool). Returns the binary or None."""
    out = pathlib.Path(bin_dir) / "cr_rs_rollback_probe"
    if out.exists() and out.stat().st_mtime >= PROBE_SRC.stat().st_mtime:
        return out
    cmd = ["g++", "-O2", "-std=c++17", "-I", str(REPO_ROOT / "include"), "-I", str(REPO_ROOT / "ggml" / "include"), str(PROBE_SRC),
           "-o", str(out), f"-L{bin_dir}", "-lllama", "-lggml", "-lggml-base", f"-Wl,-rpath,{bin_dir}"]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    except Exception as e:
        notes.append(f"r: probe build failed to run: {e}")
        return None
    if r.returncode != 0:
        notes.append("r: probe build failed: " + (r.stderr or "")[-400:])
        return None
    return out


def run_probe(exe, model: str, ngl: int, timeout: float) -> dict:
    cmd = [str(exe), "-m", model, "-ngl", str(ngl)] + PROBE_ARGS
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=max(timeout, 5.0))
    except subprocess.TimeoutExpired:
        return {"cmd": cmd, "error": "timeout"}
    try:
        return dict(json.loads(p.stdout.strip().splitlines()[-1]), cmd=cmd, returncode=p.returncode)
    except (ValueError, IndexError):
        return {"cmd": cmd, "error": f"exit {p.returncode}: {(p.stderr or p.stdout)[-300:].strip()}"}


def probe_verdict(res: dict) -> dict:
    """Per decode mode: largest logit difference and top-1 agreement over the rollbacks. single: one-token decodes (draft-simple drafting),
    rollback 1..n_extra; batch: one batch, rollback 1..n_extra - 1 (a verify batch keeps its first token); batch_whole: the whole batch."""
    out = {}
    n_extra = res.get("n_extra")
    for mode in ("single", "batch", "batch_whole"):
        dec = "single" if mode == "single" else "batch"
        cs = [c for c in res.get("cases", []) if c.get("decode") == dec and "max_abs_diff" in c
              and (mode == "single" or (c.get("rollback") == n_extra) == (mode == "batch_whole"))]
        if cs:
            out[mode] = {"max_abs_diff": max(c["max_abs_diff"] for c in cs), "max_abs_ref": max(c["max_abs_ref"] for c in cs),
                         "top1_equal": sum(1 for c in cs if c["top1_equal"]), "n": len(cs),
                         "by_rollback": {str(c["rollback"]): c["max_abs_diff"] for c in cs}}
    return out


def probe_part(bin_dir, models: dict, budget: common.Budget, notes: list) -> dict:
    out: dict = {"src": os.path.relpath(PROBE_SRC, common.EXP_DIR), "args": PROBE_ARGS, "models": {}}
    exe = build_probe(bin_dir, notes)
    if not exe:
        out["error"] = "probe build failed"
        return out
    for name, path in models.items():
        if budget.expired(20):
            notes.append(f"r: time budget used up, {name} skipped")
            continue
        res = run_probe(exe, path, 99, budget.timeout(240, reserve_s=10))
        res["verdict"] = probe_verdict(res)
        out["models"][name] = res
        v = res["verdict"]
        if v:
            notes.append("r %s: rollback after one-token decodes: max logit diff %s, top-1 equal %s/%s; after one batch: max diff %s" % (
                name, _f((v.get("single") or {}).get("max_abs_diff")), (v.get("single") or {}).get("top1_equal"), (v.get("single") or {}).get("n"),
                _f((v.get("batch") or {}).get("max_abs_diff"))))
    out["note"] = ("batch rows with rollback == n_extra roll back the whole batch: the snapshot before its first token was never written; "
                   "the server never does that with a verify batch (K + 1 tokens, rollback <= K)")
    return out


# ---------------------------------------------------------------- run

def part_budget_s(left_s: float, part: str, remaining: list[str]) -> float:
    tot = sum(WEIGHTS[p] for p in remaining) or 1.0
    return max(left_s - END_RESERVE_S, 0.0) * WEIGHTS[part] / tot


def run(args) -> int:
    notes: list[str] = []
    args.gpu = common.gpu_tag(args)
    for k in [k for k in os.environ if k == mc.ENV_VAR or k.startswith("GGML_EXP_")]:       # the configs set them, nothing else may
        notes.append(f"{k} was set in the environment; removed")
        os.environ.pop(k)
    bin_dir = e4_bonsai.llama_bin()
    total = common.Budget(args.budget_min)
    secrets = r1.secret_values()
    order = sp.balanced_subset(make_prompts.load_longctx(args.prompts), args.seed, args.per_category)
    if args.quick:
        order = [dict(p, n_predict=min(p["n_predict"], 32)) for p in order]
    blocks = sp.prompt_blocks(order, args.blocks)
    cfgs = matrix_configs(args.configs)
    config = {"parts": args.parts, "budget_min": args.budget_min, "blocks": args.blocks, "configs": args.configs, "seed": args.seed,
              "per_category": args.per_category, "ctx": args.ctx, "prompt_ids": [p["id"] for p in order], "models": {}, "quick": args.quick,
              "envs": {c["name"]: c["env"] for c in cfgs}}
    doc: dict = {"gate": None, "probe": None, "matrix": None, "host": None}

    def save(final: bool = False) -> None:
        (common.write_result if final else common.write_partial)(EXP, args, r1.scrub(config, secrets), [], r1.scrub(notes, secrets), False,
                                                                 extra=r1.scrub(doc, secrets))

    if not (bin_dir / "llama-server").exists():
        notes.append(f"llama-server not found in {bin_dir}: build llama.cpp (CR_BUILD_LLAMA=1)")
        save(final=True)
        return 2
    smi = e4_pdl.smi_sample()
    bad = mc.gpu600_problem(args.gpu, smi)
    doc["gate"] = {"gpu": args.gpu, "smi": smi, "problem": bad, "allow_any_gpu": args.allow_any_gpu}
    if bad and not args.allow_any_gpu:
        notes.append(f"gate: {bad}; no run on this host (B needs a 600 W PRO 6000)")
        save(final=True)
        return 1
    if bad:
        notes.append(f"gate (--allow-any-gpu): {bad}; the timings are not comparable to the PRO 6000 runs")
    paths: dict = {}
    try:
        for name in ("pq2", "draft_q8_0"):
            paths[name], config["models"][name] = e4_pdl.model_path(name, notes)
    except Exception as e:
        notes.append(f"model download failed: {type(e).__name__}: {e}")
        save(final=True)
        return 3
    parts = [p for p in PARTS if p in args.parts]
    rc = 0
    cut = {"part": None}
    try:
        for i, p in enumerate(parts):
            sub_s = part_budget_s(total.left(), p, parts[i:])
            if p == "r":
                sub_s = max(sub_s, min(120.0, total.left()))     # two short probe runs, not worth a skip
            if sub_s < 60:
                notes.append(f"part {p} skipped: only {sub_s:.0f} s of budget left")
                continue
            cut["part"] = p
            print(f"[{EXP}] part {p}: {sub_s / 60:.1f} min of {total.left() / 60:.1f} min left", flush=True)
            try:
                if p == "r":
                    doc["probe"] = probe_part(bin_dir, {"draft_q8_0": paths["draft_q8_0"], "pq2": paths["pq2"]}, common.Budget(sub_s / 60.0), notes)
                else:
                    sub = common.Budget(sub_s / 60.0)
                    probe = r1.host_probe(bin_dir, paths["pq2"])
                    hbad, cap = r1.host_health(e4_pdl.smi_sample(), (probe or {}).get("smi"), check_power=not args.allow_power_cap)
                    doc["host"] = {"problem": hbad, "power_cap_note": cap, "probe": probe}
                    if hbad:
                        notes.append(f"unhealthy host: {hbad}; part m skipped")
                        rc = 1
                    else:
                        drafts = {"q8_0": paths["draft_q8_0"]}
                        doc["matrix"] = sp.run_blocks(
                            cfgs=cfgs, blocks=blocks, orders=sp.block_orders([c["name"] for c in cfgs], len(blocks)), bin_dir=bin_dir,
                            target=paths["pq2"], cmd_for=lambda c: sp.config_cmd(bin_dir, paths["pq2"], drafts, c, ctx=args.ctx),
                            budget_s=sub.left(), notes=notes, gpu=args.gpu, exp=EXP, ctx=args.ctx,
                            on_update=lambda m: (doc.__setitem__("matrix", m), save()), tag=EXP)
            except Exception as e:  # a failed part must not lose the others
                notes.append(f"part {p} crashed: {type(e).__name__}: {e}")
            cut["part"] = None
            save()
    finally:
        if cut["part"]:
            notes.append(f"run cut during part {cut['part']}: the results of that part are partial")
        if (doc.get("matrix") or {}).get("runs"):
            try:
                doc["matrix"]["summary"] = matrix_summary(cfgs, doc["matrix"]["runs"])
                notes.extend(verdict_lines(doc["matrix"]["summary"]))
                if any(not v["ok"] for v in doc["matrix"]["summary"]["paths"].values()):
                    rc = max(rc, 1)
            except Exception as e:
                notes.append(f"final matrix summary failed: {type(e).__name__}: {e}")
        notes.append(r2a.NGRAM_STATE_NOTE + "; here one server per block and config")
        save(final=True)
    return rc


# ---------------------------------------------------------------- CLI

def _names(text):
    return [t.strip() for t in text.split(",") if t.strip()]


def parse_args(argv=None):
    ap = argparse.ArgumentParser(prog="exps.wrap_b", description=__doc__.split("\n")[0])
    common.add_common_args(ap)
    ap.add_argument("--parts", type=lambda t: t.replace(",", ""), default=PARTS, help="r (rollback probe), m (matrix)")
    ap.add_argument("--configs", type=_names, default=list(CONFIG_NAMES), help=f"default {','.join(CONFIG_NAMES)}")
    ap.add_argument("--blocks", type=int, default=BLOCKS)
    ap.add_argument("--per-category", type=int, default=sp.PER_CATEGORY)
    ap.add_argument("--seed", type=int, default=sp.SEED)
    ap.add_argument("--ctx", type=int, default=sp.CTX)
    ap.add_argument("--prompts", default=str(make_prompts.R2A_OUT))
    ap.add_argument("--allow-power-cap", action="store_true", help="a power cap below default is a note, not a skip of part m")
    ap.add_argument("--allow-any-gpu", action="store_true", help="run on a host that is not a 600 W PRO 6000 (development only)")
    a = ap.parse_args(argv)
    bad = [p for p in a.parts if p not in PARTS]
    badc = [c for c in a.configs if c not in CONFIG_NAMES]
    if bad or badc:
        ap.error(f"unknown part(s) {bad} or config(s) {badc}")
    if "base" not in a.configs and "m" in a.parts:
        ap.error("--configs must include base: every ratio is against target alone")
    a.budget_min = a.budget_min or (8.0 if a.quick else DEFAULT_TOTAL_MIN)
    if a.quick:
        a.per_category, a.blocks = min(a.per_category, 2), min(a.blocks, 2)
    return a


def plan_json(a) -> dict:
    bin_dir = e4_bonsai.llama_bin()
    order = sp.balanced_subset(make_prompts.load_longctx(a.prompts), a.seed, a.per_category)
    blocks = sp.prompt_blocks(order, a.blocks)
    cfgs = matrix_configs(a.configs)
    drafts = {"q8_0": "<draft_q8_0.gguf>"}
    return {"exp": EXP, "parts": a.parts, "budget_min": a.budget_min, "weights": WEIGHTS, "gate": list(mc.GPU600_TAGS),
            "model": hf_ids.resolve("bonsai2", "PQ2_0", offline=True), "probe": {"src": str(PROBE_SRC.name), "args": PROBE_ARGS},
            "blocks": [[p["id"] for p in b] for b in blocks], "orders": sp.block_orders([c["name"] for c in cfgs], len(blocks)),
            "configs": [{"name": c["name"], "env": c["env"], "expected": expected_path(c),
                         "cmd": sp.config_cmd(bin_dir, "<pq2.gguf>", drafts, c, ctx=a.ctx)} for c in cfgs],
            "pairs": PAIRS, "quick": a.quick}


def main(argv=None) -> int:
    a = parse_args(argv)
    if a.dry_run:
        common.print_plan(plan_json(a))
        return 0
    with common.term_exits():
        return run(a)


if __name__ == "__main__":
    sys.exit(main())
