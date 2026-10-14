"""R2b spec: draft precision and n-gram + draft combined, on a balanced subset of the R2a prompts, one PRO 6000 pod session.

  python -m exps.fetch_models --models pq2,draft_bf16 --timeout-min 6 && python -m exps.r2b_spec [--parts m,p] [--budget-min 22]

R2a (exps.r2a_longctx) ran 53 long-context agentic prompts. This run keeps its prompt file, seed, ctx, n_predict, -lv 4, metrics and
bootstrap intervals, and takes the first 8 prompts of each category from the same seeded stratified order (24 prompts, a prefix of
what R2a ran, so every prompt has an R2a row to anchor to). Every config runs the same prompts in the same order.
m  matrix. One llama-server per config, base first and the rest in a seeded shuffled order, base again at the end (drift bound):
   base     target alone
   k4       draft-simple K=4, Qwen3.5-0.8B BF16 (the R2a config: anchor)
   k4_q8    draft-simple K=4, Qwen3.5-0.8B Q8_0 (ggml-org/Qwen3.5-0.8B-GGUF, file checked with the HF API and downloaded by
            this script, not by fetch_models; llama-quantize of the BF16 file on the pod when the repo or the file is missing)
   combo    --spec-type ngram-mod,draft-simple, ngram-mod fork defaults (n_match 24, n_min 48, n_max 64), draft K=4 BF16
   Per config, per category and overall: tok/s, speedup, r, alpha (draft-simple only) with bootstrap 95% intervals (prompts
   resampled) and greedy output equality against base. 8 prompts per category is below R2a's MIN_N_CATEGORY, so n_ge_min is false
   and a category is exploratory: the intervals are the evidence, not the point values. data.anchor compares base and k4 with the
   R2a rows of the same prompt ids (tok/s ratio, output equality), when results/r2a has a result of the same GPU tag.
p  rollback probe, -lv 5 with LLAMA_TRACE=1, one prompt per category (R2a's PROBE_N_PREDICT), untimed: k4, ngram_mod (fork defaults,
   new here), ngram_mod_short, ngram_map_k_short and combo. For combo it counts, from the debug log, which implementation made each
   draft and how long it was, how many drafts were cut ("truncating draft"), how many checkpoints were created and restored, and
   whether created checkpoints equal the drafts longer than n_rs_seq. The debug logs live in a private dir and are deleted.
What the fork does with ngram-mod,draft-simple (read from the source, checked by p):
 - --spec-type takes a comma list (common/arg.cpp). need_n_rs_seq() (common/common.h) is the draft n_max when draft-simple is in the
   list, so the target gets n_rs_seq = K = 4 and takes the in-batch (RS) path; without a draft type it is 0 (checkpoint path).
 - common_speculative_draft() (common/speculative.cpp) asks the implementations in priority order (n-gram types first, draft-simple
   last) and stops at the first that returns a non-empty draft. The draft-simple context then only catches up with the accepted tokens.
 - A draft is cut to dp.n_max, which the server sets to the room left in the slot (get_n_draft_max()), not to --spec-draft-n-max. An
   ngram-mod draft can be up to 64 tokens although K = 4.
 - A draft longer than n_rs_seq makes the server save a checkpoint (draft.size() > llama_n_rs_seq), and a partial acceptance with
   more than n_rs_seq rolled-back tokens restores it and replays the target pass (server-context.cpp). So the n-gram part of the
   combo falls back to checkpoints for its long drafts, the draft-simple part does not.
Time: failsafe 55 min = bootstrap with the cold build ~14, downloads <= 6, r2b_spec 22, end reserve 5 (the margin covers the Q8_0
quantize fallback). Result: results/r2b_spec/<gpu>/<stamp>.json with data.matrix, data.probe, data.anchor, data.host, data.prompt_set.
An unhealthy host skips m (p still runs), as in R2a.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import pathlib
import re
import subprocess
import sys

from exps import common, e4_bonsai, e4_pdl, e5_spec, hf_ids, make_prompts, r1, r2a_longctx as r2a

EXP = "r2b_spec"
SEED = r2a.SEED
CTX = r2a.CTX
PER_CATEGORY = 8
CONFIG_NAMES = ["base", "k4", "k4_q8", "combo"]
PROBE_NAMES = ["k4", "ngram_mod", "ngram_mod_short", "ngram_map_k_short", "combo"]
DROP_ORDER = ["k4_q8", "combo"]               # whole configs dropped (first to last) before any prompt; base and k4 stay
WEIGHTS = {"m": 15.0, "p": 6.0}
PART_ORDER = "mp"
DEFAULT_TOTAL_MIN = 22.0
COMBO_TYPES = "ngram-mod,draft-simple"
K = 4
Q8_FILE = hf_ids.DRAFT_GGUF[1].replace("BF16", "Q8_0")
ANCHOR_CONFIGS = ("base", "k4")
CALLED_RE = re.compile(r"called impl (\S+), hist size = (\d+), call_count = (\d+), gen = (\d+)")
TRUNC_RE = re.compile(r"truncating draft to (\d+) tokens")
N_RS_SEQ_NOTE = ("need_n_rs_seq() is the draft n_max with draft-simple in the list: the target keeps n_rs_seq = K, so the "
                 "combo takes the in-batch path for drafts up to K and a checkpoint for longer n-gram drafts")


# ---------------------------------------------------------------- configs and commands

def matrix_configs(names=CONFIG_NAMES) -> list[dict]:
    """R2a configs (same flags, -lv 4) plus k4_q8 and combo. Draft: 'bf16' or 'q8_0' (a key of the drafts dict of config_cmd)."""
    by = {c["name"]: c for c in r2a.matrix_configs(["base", "k4", "ngram_mod", "ngram_mod_short", "ngram_map_k_short"])}
    by["k4_q8"] = dict(by["k4"], name="k4_q8", draft="q8_0", desc="draft-simple K=4, Qwen3.5-0.8B Q8_0 draft")
    by["combo"] = dict(by["k4"], name="combo", spec_type=COMBO_TYPES,
                       desc="--spec-type ngram-mod,draft-simple: ngram-mod fork defaults first, draft-simple K=4 (BF16) when it has no draft")
    return [by[n] for n in names]


def config_cmd(bin_dir, target: str, drafts: dict, cfg: dict, port: int | None = None, ctx: int = CTX) -> list[str]:
    """llama-server argv. drafts: {'bf16': path, 'q8_0': path}. A config with a draft model gets -md, -ngld 99 and --spec-draft-n-max K
    (for 'k4' this is the argv of R2a, flag for flag)."""
    cmd = e5_spec.server_cmd(bin_dir, target, 0, None, port or e5_spec.PORT, ctx)
    t = cfg["spec_type"]
    has_draft = has_draft_model(cfg)
    if has_draft:
        cmd += ["-md", drafts[cfg["draft"]], "-ngld", "99"]
    if t != "none":
        cmd += ["--spec-type", t]
    if has_draft:
        cmd += ["--spec-draft-n-max", str(cfg["k"])]
    return cmd + list(cfg["extra"])


def has_draft_model(cfg: dict) -> bool:
    """The R2a config table gives every config a draft key; only draft-simple in the spec types loads the model."""
    return "draft-simple" in cfg["spec_type"].split(",")


def cmd_for(bin_dir, target: str, drafts: dict, ctx: int = CTX):
    return lambda cfg: config_cmd(bin_dir, target, drafts, cfg, ctx=ctx)


def probe_configs() -> list[dict]:
    by = {c["name"]: c for c in matrix_configs(CONFIG_NAMES)}
    by.update({c["name"]: c for c in r2a.matrix_configs(["ngram_mod", "ngram_mod_short", "ngram_map_k_short"])})
    return [r2a.with_lv(by[n], r2a.PROBE_LV) for n in PROBE_NAMES]


def balanced_subset(rows: list[dict], seed: int = SEED, per_category: int = PER_CATEGORY) -> list[dict]:
    """First per_category prompts of every category in the seeded stratified order of R2a, in that order (a prefix of the R2a order
    when per_category is the same for all categories, so each prompt has an R2a row)."""
    out, n = [], {}
    for p in make_prompts.shuffled(rows, seed, stratify=True):
        if n.get(p["category"], 0) < per_category:
            n[p["category"]] = n.get(p["category"], 0) + 1
            out.append(p)
    return out


# ---------------------------------------------------------------- draft models

def resolve_q8(fetch=None, offline: bool = False) -> dict:
    """The Q8_0 draft in the repo of the BF16 draft, checked with the HF API: {repo, file, exists (None = API unreachable)}."""
    repo = hf_ids.DRAFT_GGUF[0]
    out = {"repo": repo, "file": Q8_FILE, "exists": None, "source": "pinned", "note": None}
    if offline:
        out["note"] = "offline: pinned value, not verified"
        return out
    exists, files, err = hf_ids._check_repo(repo, fetch or hf_ids._http_json)
    if exists is None:
        out["note"] = f"HF API unreachable ({err}); using pinned value"
        return out
    out.update(source="api", exists=exists and Q8_FILE in files, siblings=[f for f in files if f.endswith(".gguf")])
    if not out["exists"]:
        out["note"] = f"{repo} has no {Q8_FILE}: quantize the BF16 file"
    return out


def quantize_q8(bin_dir, bf16: str, out_path: str, timeout: float = 240.0) -> tuple[str | None, str | None]:
    """llama-quantize BF16 -> Q8_0 on the pod. Returns (path, error)."""
    exe = pathlib.Path(bin_dir) / "llama-quantize"
    if not exe.exists():
        return None, f"{exe} not found"
    try:
        p = subprocess.run([str(exe), bf16, out_path, "Q8_0"], capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return None, "llama-quantize timeout"
    if p.returncode != 0 or not os.path.exists(out_path):
        return None, f"llama-quantize exit {p.returncode}: {(p.stderr or '')[-300:].strip()}"
    return out_path, None


def q8_path(bin_dir, bf16: str, notes: list) -> tuple[str | None, dict]:
    """Path of the Q8_0 draft: the ggml-org file when the API lists it, else llama-quantize of the BF16 file. (None, info) when both fail."""
    info = resolve_q8()
    if info["exists"] is not False:
        try:
            from huggingface_hub import hf_hub_download
            path = hf_hub_download(info["repo"], info["file"])
            return path, dict(info, origin="hf", path_size=os.path.getsize(path))
        except Exception as e:
            notes.append(f"Q8_0 draft download failed: {type(e).__name__}: {e}; quantizing the BF16 file")
    out = os.path.join(os.path.dirname(os.path.abspath(bf16)), "Qwen3.5-0.8B-Q8_0.quantized.gguf")
    path, err = quantize_q8(bin_dir, bf16, out)
    if path:
        return path, dict(info, origin="llama-quantize", path_size=os.path.getsize(path))
    notes.append(f"Q8_0 draft unavailable: {err}")
    return None, dict(info, origin=None, error=err)


# ---------------------------------------------------------------- log counts and the R2a anchor

def count_mixed_events(text: str, cfg: dict | None = None) -> dict:
    """Per-step events of a -lv 5 log (LLAMA_TRACE=1) when the target can have both a rollback window and checkpoints (combo).
    From the debug 'called impl X ... gen = G' lines: per implementation the number of drafts and their length. 'truncating draft to
    N': drafts cut to the room left in the slot. 'created speculative checkpoint': one per fresh draft longer than n_rs_seq (K for a
    config with a draft model, else 0). 'restoring speculative checkpoint' and the INFO 'accepted a/b draft tokens (restore
    checkpoint)' (LLAMA_TRACE): partial acceptance beyond the window; the step after it replays and is not counted as a draft.
    Accepted / drafted per implementation come from the INFO lines. Without LLAMA_TRACE only the draft-side numbers exist."""
    k_rs = int(cfg["k"]) if cfg and has_draft_model(cfg) else 0
    impls: dict = {}
    created = restored = truncated = 0
    trunc_to: set = set()
    cur = None
    replay_next = False
    for ln in text.splitlines():
        if (m := CALLED_RE.search(ln)):
            cur = m.group(1)
            d = impls.setdefault(cur, {"drafts": 0, "tokens": 0, "max_len": 0, "longer_than_k": 0, "drafted": 0, "accepted": 0, "restored": 0})
            g = int(m.group(4))
            d["drafts"] += 1
            d["tokens"] += g
            d["max_len"] = max(d["max_len"], g)
            d["longer_than_k"] += g > k_rs
        elif (m := TRUNC_RE.search(ln)):
            truncated += 1
            trunc_to.add(int(m.group(1)))
        elif r2a.CREATED_RE.search(ln):
            created += 1
        elif r2a.RESTORED_RE.search(ln):
            restored += 1
        elif (m := r2a.ACCEPT_TRACE_RE.search(ln)):
            if replay_next:                      # the replay step after a restore is not a new draft
                replay_next = False
                continue
            d = impls.get(cur)
            if d is not None:
                d["accepted"] += int(m.group(1))
                d["drafted"] += int(m.group(2))
                if m.group(3):
                    d["restored"] += 1
            replay_next = bool(m.group(3))
    longer = sum(d["longer_than_k"] for d in impls.values())
    out: dict = {"n_rs_seq": k_rs, "impls": {n: dict(d, mean_len=d["tokens"] / d["drafts"] if d["drafts"] else None,
                                                    r=d["accepted"] / d["drafted"] if d["drafted"] else None) for n, d in impls.items()},
                 "created": created, "restored": restored, "truncated": truncated, "truncated_to": sorted(trunc_to)[:8],
                 "drafts": sum(d["drafts"] for d in impls.values()), "drafts_longer_than_n_rs_seq": longer,
                 "created_matches_long_drafts": created == longer if impls else None}
    out["restore_per_checkpoint"] = restored / created if created else None
    return out


def probe_events(text: str, cfg: dict | None = None) -> dict:
    """r2a's counts for the pure types (k4: RS window; ngram types: checkpoint path); the mixed counts for any config with both."""
    if cfg and "," in cfg["spec_type"]:
        return count_mixed_events(text, cfg)
    return r2a.count_spec_events(text, cfg)


def load_r2a_runs(gpu: str, root=None) -> tuple[dict, str | None]:
    """{config: {prompt id: row}} of the newest R2a result of this GPU tag, and its path; ({}, None) when there is none."""
    root = pathlib.Path(root) if root else common.EXP_DIR / "results" / "r2a"
    files = sorted(glob.glob(str(root / gpu / "2*.json")))
    if not files:
        return {}, None
    doc = json.loads(pathlib.Path(files[-1]).read_text())
    runs = ((doc.get("data") or {}).get("matrix") or {}).get("runs") or {}
    return {n: {r["id"]: r for r in v.get("rows", []) if e5_spec._ok(r)} for n, v in runs.items()}, files[-1]


def anchor(runs: dict, old: dict, names=ANCHOR_CONFIGS) -> dict:
    """This run against R2a on the prompt ids both finished: aggregate tok/s of each, their ratio (now / R2a) and the share of equal
    output texts. A ratio far from 1.0 means a different host, not a different config."""
    out: dict = {}
    for name in names:
        now = {r["id"]: r for r in (runs.get(name) or {}).get("rows", []) if e5_spec._ok(r)}
        was = old.get(name) or {}
        ids = [i for i in now if i in was]
        if not ids:
            continue
        st = (runs[name].get("spec_type") or "none")
        a, b = e5_spec._agg([now[i] for i in ids], st), e5_spec._agg([was[i] for i in ids], st)
        eq = [now[i].get("text") == was[i].get("text") for i in ids if now[i].get("text") is not None and was[i].get("text") is not None]
        out[name] = {"n": len(ids), "tok_s_now": a["tok_s"], "tok_s_r2a": b["tok_s"], "ratio": a["tok_s"] / b["tok_s"] if a["tok_s"] and b["tok_s"] else None,
                     "r_now": a["r"], "r_r2a": b["r"], "text_equal_frac": sum(eq) / len(eq) if eq else None}
    return out


# ---------------------------------------------------------------- run

def part_budget_s(left_s: float, part: str, remaining: list[str]) -> float:
    tot = sum(WEIGHTS[p] for p in remaining) or 1.0
    return max(left_s - r2a.END_RESERVE_S, 0.0) * WEIGHTS[part] / tot


def run(args) -> int:
    from infra.results import run_dir
    notes: list[str] = []
    args.gpu = common.gpu_tag(args)
    bin_dir = e4_bonsai.llama_bin()
    rows_all = make_prompts.load_longctx(args.prompts)
    order = balanced_subset(rows_all, args.seed, args.per_category)
    if args.quick:
        order = [dict(p, n_predict=min(p["n_predict"], 32)) for p in order]
    parts = [p for p in PART_ORDER if p in args.parts]
    runnable = list(parts)
    if not (bin_dir / "llama-server").exists():
        notes.append(f"parts {parts} skipped: llama-server not found in {bin_dir} (build llama.cpp with CR_BUILD_LLAMA=1)")
        runnable = []
    total = common.Budget(args.budget_min)
    secrets = r1.secret_values()
    config = {"parts": parts, "budget_min": args.budget_min, "weights": WEIGHTS, "seed": args.seed, "per_category": args.per_category,
              "n_prompts": len(order), "ctx": args.ctx, "configs": args.configs, "lv": r2a.LV, "n_predict": make_prompts.R2A_N_PREDICT,
              "quick": args.quick, "prompts_file": str(args.prompts), "models": {}, "prompt_ids": [p["id"] for p in order]}
    doc: dict = {"prompt_set": make_prompts.longctx_stats(order), "matrix": None, "probe": None, "anchor": None, "host": None,
                 "source_facts": N_RS_SEQ_NOTE}

    def save(final: bool = False) -> None:
        extra = r1.scrub(doc, secrets)
        (common.write_result if final else common.write_partial)(EXP, args, r1.scrub(config, secrets), r2a.summary_rows(doc),
                                                                 r1.scrub(notes, secrets), False, extra=extra)

    paths: dict = {}
    try:
        for name in ("pq2", "draft_bf16"):
            paths[name], config["models"][name] = e4_pdl.model_path(name, notes)
    except Exception as e:
        notes.append(f"model download failed: {type(e).__name__}: {e}")
        save(final=True)
        print(f"error: {notes[-1]}", file=sys.stderr)
        return 3
    cfgs = matrix_configs(args.configs)
    if "k4_q8" in args.configs:
        paths["draft_q8_0"], config["models"]["draft_q8_0"] = q8_path(bin_dir, paths["draft_bf16"], notes)
        if not paths["draft_q8_0"]:
            cfgs = [c for c in cfgs if c["name"] != "k4_q8"]
            notes.append("k4_q8 dropped: no Q8_0 draft")
    drafts = {"bf16": paths["draft_bf16"], "q8_0": paths.get("draft_q8_0")}
    mk = cmd_for(bin_dir, paths["pq2"], drafts, args.ctx)
    (run_dir(EXP, args.gpu) / "raw").mkdir(parents=True, exist_ok=True)
    smi = e4_pdl.smi_sample()
    probe = r1.host_probe(bin_dir, paths["pq2"]) if "m" in runnable else None
    bad, cap = r1.host_health(smi, (probe or {}).get("smi"), check_power=not args.allow_power_cap)
    config["host_smi"] = smi
    doc["host"] = {"problem": bad, "allow_power_cap": args.allow_power_cap, "parts_skipped": [], "probe": probe}
    rc = 0
    if bad:
        doc["host"]["parts_skipped"] = [p for p in runnable if p == "m"]
        notes.append(f"unhealthy host: {bad}; part m (timings) skipped, part p (no timing) still runs")
        runnable = [p for p in runnable if p == "p"]
        rc = 1
    elif cap:
        doc["host"]["power_cap_note"] = cap
        notes.append(f"power cap accepted (--allow-power-cap): {cap}; timings are for the capped host")
    cut = {"part": None}
    try:
        for i, p in enumerate(runnable):
            budget_s = part_budget_s(total.left(), p, runnable[i:])
            print(f"[r2b_spec] part {p}: {budget_s / 60:.1f} min of {total.left() / 60:.1f} min left", flush=True)
            if budget_s < 60:
                notes.append(f"part {p} skipped: only {budget_s:.0f} s of budget left")
                continue
            cut["part"] = p
            try:
                if p == "m":
                    doc["matrix"] = r2a.run_matrix(args=args, bin_dir=bin_dir, target=paths["pq2"], draft=paths["draft_bf16"], order=order,
                                                   budget_s=budget_s, notes=notes, gpu=args.gpu, all_cfgs=cfgs, cmd_for=mk,
                                                   drop_order=DROP_ORDER, exp=EXP,
                                                   on_update=lambda m: (doc.__setitem__("matrix", m), save()))
                else:
                    pcfgs = [c for c in probe_configs() if c["name"] in args.probe]
                    doc["probe"] = r2a.run_probe(bin_dir=bin_dir, target=paths["pq2"], draft=paths["draft_bf16"], order=order,
                                                 budget=common.Budget(budget_s / 60.0), notes=notes, gpu=args.gpu, ctx=args.ctx,
                                                 names=[c["name"] for c in pcfgs], all_cfgs=pcfgs, cmd_for=mk, count_fn=probe_events)
            except Exception as e:  # a failed part must not lose the others
                notes.append(f"part {p} crashed: {type(e).__name__}: {e}")
            cut["part"] = None
            save()
    finally:
        if cut["part"]:
            notes.append(f"run cut during part {cut['part']}: the results of that part are partial")
        if (doc.get("matrix") or {}).get("runs"):
            try:
                doc["matrix"]["summary"] = r2a.summarize(doc["matrix"]["runs"], ci=True)
                old, old_path = load_r2a_runs(args.gpu)
                doc["anchor"] = {"r2a_result": os.path.relpath(old_path, common.EXP_DIR) if old_path else None, "gpu": args.gpu,
                                 "configs": anchor(doc["matrix"]["runs"], old) if old else {}}
                if not old:
                    notes.append(f"no R2a result for GPU tag {args.gpu} in results/r2a: no anchor")
            except Exception as e:
                notes.append(f"final matrix summary failed: {type(e).__name__}: {e}")
        save(final=True)
    return rc


# ---------------------------------------------------------------- CLI

def _names(text):
    return [t.strip() for t in text.split(",") if t.strip()]


def parse_args(argv=None):
    ap = argparse.ArgumentParser(prog="exps.r2b_spec", description=__doc__.split("\n")[0])
    common.add_common_args(ap)
    ap.add_argument("--parts", type=_names, default=list(PART_ORDER), help="m (matrix), p (rollback probe)")
    ap.add_argument("--configs", type=_names, default=list(CONFIG_NAMES), help=f"default {','.join(CONFIG_NAMES)}")
    ap.add_argument("--probe", type=_names, default=list(PROBE_NAMES), help=f"default {','.join(PROBE_NAMES)}")
    ap.add_argument("--per-category", type=int, default=PER_CATEGORY)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--ctx", type=int, default=CTX)
    ap.add_argument("--prompts", default=str(make_prompts.R2A_OUT))
    ap.add_argument("--allow-power-cap", action="store_true", help="a power cap below default is a note, not a skip of part m")
    a = ap.parse_args(argv)
    bad = [p for p in a.parts if p not in WEIGHTS]
    badc = [c for c in a.configs if c not in CONFIG_NAMES]
    badp = [c for c in a.probe if c not in PROBE_NAMES]
    if bad or badc or badp:
        ap.error(f"unknown part(s) {bad}, config(s) {badc} or probe config(s) {badp}")
    if "base" not in a.configs and "m" in a.parts:
        ap.error("--configs must include base: every ratio is against target alone")
    if a.ctx < 16384:
        ap.error("--ctx must be at least 16384: prompts reach 8K tokens and edits answer with 512")
    a.budget_min = a.budget_min or (8.0 if a.quick else DEFAULT_TOTAL_MIN)
    if a.quick:
        a.per_category = min(a.per_category, 2)
    return a


def plan_json(a) -> dict:
    bin_dir = e4_bonsai.llama_bin()
    rows = make_prompts.load_longctx(a.prompts)
    order = balanced_subset(rows, a.seed, a.per_category)
    parts = [p for p in PART_ORDER if p in a.parts]
    drafts = {"bf16": "<draft_bf16.gguf>", "q8_0": "<draft_q8_0.gguf>"}
    m_s = part_budget_s(a.budget_min * 60.0, "m", parts) if "m" in parts else 0.0
    cfgs = r2a.config_order(matrix_configs(a.configs), a.seed)
    m_plan = None
    if m_s:
        cfgs, m_plan = r2a.fit_configs(cfgs, order, max(m_s - r2a.DRIFT_RESERVE_S, 0.0), a.per_category, drop_order=DROP_ORDER)
    return {"exp": EXP, "parts": parts, "budget_min": a.budget_min, "weights": WEIGHTS, "seed": a.seed, "ctx": a.ctx,
            "prompt_set": make_prompts.longctx_stats(order), "prompt_ids": [p["id"] for p in order], "per_category": a.per_category,
            "n_predict": make_prompts.R2A_N_PREDICT,
            "matrix": {"config_order": [c["name"] for c in cfgs], "plan": m_plan,
                       "configs": [{"name": c["name"], "desc": c["desc"], "cmd": config_cmd(bin_dir, "<pq2.gguf>", drafts, c, ctx=a.ctx)} for c in cfgs]},
            "probe": {"configs": [c["name"] for c in probe_configs() if c["name"] in a.probe], "env": r2a.PROBE_ENV,
                      "prompt_categories": r2a.PROBE_CATEGORIES, "n_predict": r2a.PROBE_N_PREDICT,
                      "cmds": {c["name"]: config_cmd(bin_dir, "<pq2.gguf>", drafts, c, ctx=a.ctx) for c in probe_configs() if c["name"] in a.probe}},
            "q8_0_draft": resolve_q8(offline=True), "quick": a.quick}


def main(argv=None) -> int:
    a = parse_args(argv)
    if a.dry_run:
        common.print_plan(plan_json(a))
        return 0
    with common.term_exits():
        return run(a)


if __name__ == "__main__":
    sys.exit(main())
