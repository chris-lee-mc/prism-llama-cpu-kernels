"""Wrap B: test build for two upstream investigations (measure first, issue before code), one pod session on a sm_120 card.

  python -m exps.fetch_models --models pq2,draft_q8_0 --timeout-min 6 && python -m exps.wrap_b [--parts r,m] [--budget-min 22]

Experiment-only switches in this branch (env, default off, read once):
- GGML_EXP_DRAFT_CKPT=1 (the correct draft state): the server treats the draft context as FULL, so it saves the draft state before drafting
  and restores it after (the existing checkpoint path). Without it the draft context of a hybrid draft is rolled back with seq_rm after
  one-token decodes. The recurrent snapshot of a GDN layer is written only for the tokens of the last ubatch (gated_delta_net: "older
  slots are caller-owned"), so that seq_rm restores a stale state; part r measures this directly and k4q8 vs k4q8_ckpt shows it in alpha.
- GGML_EXP_DRAFT_TIME=1 (B1, upstream TODO TAG_SPEC_AVOID_DRAFT_REEVAL): draft-simple process() first waits for the target (llama_synchronize
  of ctx_tgt: llama_decode is async and the server calls process() right after the target verify), then times the decode of the verify
  batch on the draft context (llama_synchronize of ctx_dft after llama_decode) and prints it on the exp stats line (dur_p). With the correct
  state (k4q8_ckpt_t) dur_p is the isolated draft-context cost of the re-eval. Without the instrumentation the re-eval overlaps the target
  verify, so the saving of skipping it is at most the isolated cost and at least the end-to-end delta of k4q8_ckpt_t minus k4q8_ckpt
  (the synchronizes remove the overlap). It does not skip anything.
- GGML_EXP_SKIP_DRAFT_REEVAL=1 (B1 prototype, not in this matrix): the draft context keeps its drafted tokens and process() skips the verify
  batch. It is allowed only for one slot, draft-simple alone and an attention-only draft (seq_rm exact). A hybrid draft (the Qwen3.5-0.8B
  of this matrix) inherits the stale rollback above, so the switch is ignored there (warning in the server log). Not tested on a GPU.
- GGML_EXP_NGRAM_RS=N (B2): need_n_rs_seq() returns at least N when an n-gram type is in --spec-type, so the target takes the in-batch
  (RS) rollback instead of checkpoint + replay; n-gram drafts longer than N are cut to N (counted, printed with the statistics).
Gate: the GPU tag must be a sm_120 card with enough memory (pro6000-*, rtx5090); --allow-any-gpu for development. The 600 W gate of wrap_a does
not apply: every comparison here is paired inside one pod and is not PR evidence.
r  rollback probe (exps/csrc/cr_rs_rollback_probe.cpp, built against libllama of the pod build): prompt of 64 tokens, then extra tokens
   decoded one at a time or as one batch, seq_rm back to the kept tokens, one probe token; logits against a context that decoded only
   the kept tokens the same way. Runs: n4 (n_rs_seq 4, 5 extra, both decode modes) on the Qwen3.5-0.8B Q8_0 draft and the Bonsai 2 27B PQ2_0
   target; n16 and n64 (n_rs_seq 16 / 64, 17 / 65 extra, batch mode only) on the target: the window sizes B2 uses. The batch has n_rs_seq + 1
   tokens like a verify batch (K + 1 with K = n_rs_seq): rollbacks of 1 to n_rs_seq tokens, the last one (kept 1, a full rejection, snapshot
   plane n_rs_seq) is reported as the boundary row. kept 0 (whole batch) is out of the window and not a server case. Untimed.
m  matrix, the R2a balanced subset (24 prompts, 8 per category, same seed, ctx 16384, -lv 4), interleaved by prompt blocks (--blocks,
   default 2, a fresh server per block and config, config order changes per block):
   base         target alone
   k4q8         draft-simple K=4, Qwen3.5-0.8B Q8_0 draft (the CONFIG-GUIDANCE setting; stale draft state after partial acceptance)
   k4q8_ckpt    k4q8 with GGML_EXP_DRAFT_CKPT=1 (correct draft state, the reference)
   k4q8_ckpt_t  k4q8_ckpt with GGML_EXP_DRAFT_TIME=1 (target and draft synchronized, draft-context decode of the verify batch timed; this
                removes the overlap with the target, so it is a separate config and k4q8_ckpt_t vs k4q8_ckpt is the end-to-end delta that
                the overlap hides in k4q8_ckpt)
   ngram        --spec-type ngram-mod, fork defaults (checkpoint + replay on the hybrid target)
   ngram_rs16   ngram with GGML_EXP_NGRAM_RS=16 (drafts cut to 16: a draft-length setting as well as the rollback path)
   ngram_rs64   ngram with GGML_EXP_NGRAM_RS=64 (no cut at the fork default n_max 64: the clean rollback-path comparison with ngram)
   Reported: per config and category tok/s, speedup over base with bootstrap 95% CIs (prompts paired), r, alpha, greedy text equality
   vs base; the pairs k4q8_ckpt vs k4q8, k4q8_ckpt_t vs k4q8_ckpt, rs16 / rs64 vs ngram per category with paired bootstrap CIs; ms per
   step and its split from the server statistics (dur g of the drafting implementation vs the rest); the re-eval time per step of
   k4q8_ckpt_t (reeval_ms_per_step, the isolated cost, and its share of the step) next to the end-to-end ms per step delta of k4q8_ckpt_t
   over k4q8_ckpt (reeval_bounds); the low-acceptance resets of ngram-mod per config (ngram_cut.*.resets: the
   cut drafts of rs16 change the acceptance fraction and so the reset rule); path verification from the -lv 4 logs (n_rs_seq of the
   target and the draft, rollback path, the GGML_EXP_* switch lines, the cut counters).
   The draft state is checked by alpha, not by text: the target verifies every token, so a wrong draft state lowers alpha and leaves the
   output alone. Greedy text equality checks only the target side and has a numerics floor (88-100% in R2a / R2b: batch shapes change
   the accumulation order). alpha of k4q8 below alpha of k4q8_ckpt means k4q8 inherits the stale rollback.
   ngram-mod keeps its n-gram pool per server: with a fresh server per block it sees fewer earlier prompts than in R2a (same for every
   ngram config, so the pairs stay fair).
Parts run in the order r, m with weights 3 : 20 of --budget-min. An unhealthy host skips m (r still runs).
Result: results/wrap_b/<gpu>/<stamp>.json with data.gate, data.probe, data.matrix (runs, summary, pairs, paths), data.host.
Time: failsafe 55 min = bootstrap with the cold build ~14, downloads <= 6, wrap_b 22, reserve 5 (the cost model gives about 16 min for the matrix with 2 blocks, 19 with 4).
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
SKIP, CKPT, TIME, NGRAM_RS = "GGML_EXP_SKIP_DRAFT_REEVAL", "GGML_EXP_DRAFT_CKPT", "GGML_EXP_DRAFT_TIME", "GGML_EXP_NGRAM_RS"
CONFIG_NAMES = ["base", "k4q8", "k4q8_ckpt", "k4q8_ckpt_t", "ngram", "ngram_rs16", "ngram_rs64"]
PAIRS = [("k4q8_ckpt", "k4q8"), ("k4q8_ckpt_t", "k4q8_ckpt"), ("ngram_rs64", "ngram"), ("ngram_rs16", "ngram"), ("ngram_rs64", "ngram_rs16")]
DRAFT_STATE_REF = "k4q8_ckpt"           # alpha of this config is the draft-state reference
DRAFT_STATE_CHECK = ("k4q8", "k4q8_ckpt_t")
ALPHA_TOL = 0.02
BLOCKS = 2
PARTS = "rm"
WEIGHTS = {"r": 3.0, "m": 20.0}
DEFAULT_TOTAL_MIN = 22.0
WB_TAGS = ("pro6000-ws", "pro6000-server", "pro6000-maxq", "rtx5090")        # sm_120, >= 32 GB (n_rs_seq 64 keeps 65 snapshots of the 27B state)
END_RESERVE_S = 25.0
PROBE_SRC = common.EXP_DIR / "exps" / "csrc" / "cr_rs_rollback_probe.cpp"
# (label, models, argv): n4 is the draft-simple window (both decode modes); n16 and n64 are the windows B2 gives the target (verify batches only)
# extra = n_rs + 1: a verify batch of K + 1 tokens with n_rs = K; kept 1 is the full rejection (rollback K = n_rs, the snapshot plane n_rs)
PROBE_RUNS = [("n4", ("draft_q8_0", "pq2"), ["--n-rs", "4", "--prompt", "64", "--extra", "5"]),
              ("n16", ("pq2",), ["--n-rs", "16", "--prompt", "64", "--extra", "17", "--modes", "batch"]),
              ("n64", ("pq2",), ["--n-rs", "64", "--prompt", "64", "--extra", "65", "--modes", "batch"])]
PROBE_ARGS = PROBE_RUNS[0][2]
EXACT_REL = 1e-3          # a rollback is exact when the logits differ by less than this fraction of the largest logit
REPO_ROOT = common.EXP_DIR.parent.parent


def gate_problem(tag: str) -> str | None:
    """Why this card cannot run B (an sm_120 card with 32 GB or more), else None. Not a 600 W gate: B is not PR evidence."""
    return None if tag in WB_TAGS else f"GPU tag {tag} is not one of {', '.join(WB_TAGS)}"


def matrix_configs(names=CONFIG_NAMES) -> list[dict]:
    by = {c["name"]: c for c in sp.matrix_configs(["base", "k4_q8"])}
    k, ng = by["k4_q8"], r2a.matrix_configs(["ngram_mod"])[0]
    allc = {
        "base": dict(by["base"], env={}),
        "k4q8": dict(k, name="k4q8", env={}, desc="draft-simple K=4, Qwen3.5-0.8B Q8_0 draft (draft state rolled back with seq_rm: stale)"),
        "k4q8_ckpt": dict(k, name="k4q8_ckpt", env={CKPT: "1"}, desc="k4q8, draft state restored from a checkpoint after drafting (reference)"),
        "k4q8_ckpt_t": dict(k, name="k4q8_ckpt_t", env={CKPT: "1", TIME: "1"},
                            desc="k4q8_ckpt, the draft-context decode of the verify batch timed (synchronizes)"),
        "ngram": dict(ng, name="ngram", env={}, desc="ngram-mod, fork defaults (checkpoint path on the hybrid target)"),
        "ngram_rs16": dict(ng, name="ngram_rs16", env={NGRAM_RS: "16"}, desc="ngram-mod, in-batch rollback window 16, drafts cut to 16"),
        "ngram_rs64": dict(ng, name="ngram_rs64", env={NGRAM_RS: "64"}, desc="ngram-mod, in-batch rollback window 64 (no cut)"),
    }
    return [allc[n] for n in names]


def expected_path(cfg: dict) -> dict:
    """What the -lv 4 log must show for a config: rollback path, n_rs_seq of the target, the switch line of the server."""
    env = cfg.get("env") or {}
    if cfg["spec_type"] == "none":
        return {"rollback_path": "none", "n_rs_seq_target": 0, "switches": []}
    if sp.has_draft_model(cfg):
        return {"rollback_path": "in_batch", "n_rs_seq_target": int(cfg["k"]), "switches": [k for k in (SKIP, CKPT, TIME) if k in env]}
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
    cut = {n: {impl: {k: d.get(k) for k in ("cut_drafts", "cut_tokens", "gen_drafts", "gen_tokens", "rs_window", "resets") if k in d}
               for impl, d in s.items() if "cut_drafts" in d or "resets" in d} for n, s in st.items()}
    vs_base = r2a.summarize(runs, ci=True)
    split = {n: r.get("step_split") for n, r in runs.items()}
    return {"vs_base": vs_base, "pairs": {f"{a}_vs_{b}": sp.pair_by_category(runs, a, b) for a, b in PAIRS if a in runs and b in runs},
            "step_split": split, "paths": verify_paths(cfgs, runs), "ngram_cut": {n: v for n, v in cut.items() if v},
            "draft_state": draft_state(vs_base), "reeval_bounds": reeval_bounds(split), "reeval": {n: {k: v for k, v in (d or {}).items() if k.startswith(("reeval", "prefill"))}
                                                            for n, d in split.items() if d and "reeval_ms" in d}}


def reeval_bounds(split: dict, timed: str = "k4q8_ckpt_t", ref: str = DRAFT_STATE_REF) -> dict | None:
    """Bounds of the saving per step from skipping the draft re-eval. isolated: dur_p of the timed config (target synchronized first, so only
    the draft-context cost). end_to_end_delta: ms per step of the timed config minus the reference (the synchronizes remove the overlap of
    the re-eval with the target verify). Achievable saving: at most isolated, at least end_to_end_delta."""
    a, b = (split.get(timed) or {}), (split.get(ref) or {})
    if a.get("reeval_ms_per_step") is None or not a.get("ms_per_step") or not b.get("ms_per_step"):
        return None
    return {"timed": timed, "reference": ref, "isolated_ms_per_step": a["reeval_ms_per_step"], "end_to_end_delta_ms_per_step": a["ms_per_step"] - b["ms_per_step"],
            "ms_per_step_timed": a["ms_per_step"], "ms_per_step_reference": b["ms_per_step"]}


def draft_state(vs_base: dict) -> dict | None:
    """alpha of the draft-simple configs against the reference (correct draft state, k4q8_ckpt). Text equality cannot show a wrong draft
    state (the target verifies every token); alpha can. A config more than ALPHA_TOL below the reference has a stale draft state."""
    alpha = {n: ((vs_base.get(n) or {}).get("all") or {}).get("alpha") for n in (DRAFT_STATE_REF,) + DRAFT_STATE_CHECK}
    ref = alpha.get(DRAFT_STATE_REF)
    if ref is None:
        return None
    return {"reference": DRAFT_STATE_REF, "alpha": alpha, "tol": ALPHA_TOL,
            "matches_reference": {n: abs(alpha[n] - ref) <= ALPHA_TOL if alpha.get(n) is not None else None for n in DRAFT_STATE_CHECK}}


NOTE_EQ = ("text equality checks only the target side (it verifies every token) and has a numerics floor of 88-100% from batch shapes; "
           "it says nothing about the draft state, alpha does")


def verdict_lines(summ: dict) -> list[str]:
    out = []
    for key, p in (summ.get("pairs") or {}).items():
        a = (p or {}).get("all") or {}
        if a.get("speedup"):
            out.append("%s: %.3fx (95%% CI %s), alpha %s vs %s, r %s, text equal %s/%s (informational)" % (
                key, a["speedup"], _ci(a.get("speedup_ci95")), _f(a.get("alpha")), _f(a.get("base_alpha")), _f(a.get("r")), a.get("text_equal"),
                a.get("text_compared")))
    ds = summ.get("draft_state")
    if ds:
        out.append("draft state: alpha " + ", ".join("%s %s" % (n, _f(v)) for n, v in ds["alpha"].items()) + "; within %.2f of %s: %s" % (
            ds["tol"], ds["reference"], ds["matches_reference"]))
    rb = summ.get("reeval_bounds")
    for n, d in (summ.get("reeval") or {}).items():
        line = "%s: draft-context decode of the verify batch, isolated cost (target synchronized first): %s ms per call, %s ms per step (%s of the step)" % (
            n, _f(d.get("reeval_ms_per_call")), _f(d.get("reeval_ms_per_step")), _f(d.get("reeval_share_of_step")))
        if rb and rb["timed"] == n:
            line += ("; end-to-end %s minus %s: %s ms per step (the overlap with the target hides the rest in the untimed path); achievable "
                     "saving of skipping the re-eval per step: at most %s ms (isolated), at least %s ms (end-to-end delta)" % (
                         rb["timed"], rb["reference"], _f(rb["end_to_end_delta_ms_per_step"]), _f(rb["isolated_ms_per_step"]),
                         _f(rb["end_to_end_delta_ms_per_step"])))
        out.append(line)
    for n, v in (summ.get("ngram_cut") or {}).items():
        for impl, d in v.items():
            if "resets" in d:
                out.append("%s: ngram-mod low-acceptance resets %s, cut drafts %s" % (n, d["resets"], d.get("cut_drafts", 0)))
    bad = [n for n, v in (summ.get("paths") or {}).items() if not v["ok"]]
    if bad:
        out.append(f"path verification failed for {bad}: read data.matrix.summary.paths before any number")
    out.append(NOTE_EQ)
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


def run_probe(exe, model: str, ngl: int, timeout: float, args: list | None = None) -> dict:
    cmd = [str(exe), "-m", model, "-ngl", str(ngl)] + list(PROBE_ARGS if args is None else args)
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
    rollbacks 1..n_rs; batch: one batch, kept >= 1 (rollback <= n_rs when n_extra <= n_rs + 1; with n_extra = n_rs + 1 this includes the
    server's full rejection of a K + 1 token verify batch, rollback == n_rs, reported as boundary); batch_whole: kept 0, the whole batch
    (never done by the server; with n_extra = n_rs + 1 it is out of the window and may fail to decode, then it has no entry).
    exact: every case of the mode is within EXACT_REL of the largest logit, has the same top-1 and a seq_rm that returned true."""
    out = {}
    n_rs = res.get("n_rs_seq")
    for mode in ("single", "batch", "batch_whole"):
        dec = "single" if mode == "single" else "batch"
        cs = [c for c in res.get("cases", []) if c.get("decode") == dec and "max_abs_diff" in c
              and (mode == "single" and (n_rs is None or c.get("rollback", 0) <= n_rs) or mode == "batch" and c.get("kept", 0) >= 1
                   or mode == "batch_whole" and c.get("kept") == 0)]
        if cs:
            out[mode] = {"max_abs_diff": max(c["max_abs_diff"] for c in cs), "max_abs_ref": max(c["max_abs_ref"] for c in cs),
                         "top1_equal": sum(1 for c in cs if c["top1_equal"]), "n": len(cs),
                         "exact": all(_case_exact(c) for c in cs), "max_rollback": max(c["rollback"] for c in cs),
                         "by_rollback": {str(c["rollback"]): c["max_abs_diff"] for c in cs}}
    cb = [c for c in res.get("cases", []) if c.get("decode") == "batch" and "max_abs_diff" in c and n_rs and c.get("rollback") == n_rs
          and c.get("kept", 0) >= 1]
    if cb:
        c = cb[0]
        out["boundary"] = {"rollback": n_rs, "kept": c["kept"], "max_abs_diff": c["max_abs_diff"], "max_abs_ref": c["max_abs_ref"],
                           "top1_equal": c["top1_equal"], "exact": _case_exact(c)}
    return out


def _case_exact(c: dict) -> bool:
    return c["max_abs_diff"] <= EXACT_REL * max(c["max_abs_ref"], 1.0) and c["top1_equal"] and c.get("seq_rm_ok", True)


def probe_notes(label: str, name: str, res: dict) -> str:
    """One console / notes line of a probe run: the modes and, when present, the rollback == n_rs row."""
    v = res.get("verdict") or {}
    parts = ["%s: max logit diff %.3g, exact %s" % (m, v[m]["max_abs_diff"], v[m]["exact"]) for m in ("single", "batch") if m in v]
    b = v.get("boundary")
    if b:
        parts.append("rollback %d == n_rs_seq (kept %d): diff %.3g, top-1 equal %s, exact %s" % (b["rollback"], b["kept"], b["max_abs_diff"],
                                                                                               b["top1_equal"], b["exact"]))
    elif "batch" in v:
        parts.append("rollback == n_rs_seq not probed (largest batch rollback %d)" % v["batch"]["max_rollback"])
    return "r %s %s (n_rs_seq %s): " % (label, name, res.get("n_rs_seq")) + "; ".join(parts)


def probe_part(bin_dir, models: dict, budget: common.Budget, notes: list) -> dict:
    out: dict = {"src": os.path.relpath(PROBE_SRC, common.EXP_DIR), "runs": {}}
    exe = build_probe(bin_dir, notes)
    if not exe:
        out["error"] = "probe build failed"
        return out
    for label, names, argv in PROBE_RUNS:
        ent = out["runs"][label] = {"args": argv, "models": {}}
        for name in names:
            if name not in models:
                continue
            if budget.expired(20):
                notes.append(f"r: time budget used up, {label} {name} skipped")
                continue
            res = run_probe(exe, models[name], 99, budget.timeout(240, reserve_s=10), argv)
            res["verdict"] = probe_verdict(res)
            ent["models"][name] = res
            if res["verdict"]:
                notes.append(probe_notes(label, name, res))
            elif res.get("error"):
                notes.append(f"r {label} {name}: {res['error']}")
    out["note"] = ("batch rows with kept 0 roll back the whole batch: the snapshot before its first token was never written; the server never "
                   "does that with a verify batch (K + 1 tokens, rollback <= K = n_rs_seq). The boundary row (kept 1, rollback == n_rs_seq) is "
                   "the full rejection of a verify batch and reads snapshot plane n_rs_seq; single rows are the draft-simple drafting case")
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
    bad = gate_problem(args.gpu)
    doc["gate"] = {"gpu": args.gpu, "smi": smi, "problem": bad, "allow_any_gpu": args.allow_any_gpu}
    if bad and not args.allow_any_gpu:
        notes.append(f"gate: {bad}; no run on this host")
        save(final=True)
        return 1
    if bad:
        notes.append(f"gate (--allow-any-gpu): {bad}; the n_rs_seq 64 probe and the matrix may not fit in memory")
    notes.append("every comparison is paired inside this pod; absolute timings are not comparable to the 600 W PRO 6000 runs of R2a / R2b")
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
                sub_s = max(sub_s, min(180.0, total.left()))     # four short probe runs (model loads), not worth a skip
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
    ap.add_argument("--allow-any-gpu", action="store_true", help="run on a card that is not an sm_120 card with 32 GB or more (development only)")
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
    return {"exp": EXP, "parts": a.parts, "budget_min": a.budget_min, "weights": WEIGHTS, "gate": list(WB_TAGS),
            "model": hf_ids.resolve("bonsai2", "PQ2_0", offline=True), "probe": {"src": str(PROBE_SRC.name), "runs": [{"label": l, "models": list(m), "args": a_} for l, m, a_ in PROBE_RUNS]},
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
