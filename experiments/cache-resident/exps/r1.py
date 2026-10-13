"""R1: the pre-E6 measurements in one pod session: (a) draft agreement, (b) full E5 matrix, (c) per-step overhead.

  python -m exps.fetch_models --models pq2,draft_bf16,draft_q8_0 && python -m exps.r1 [--parts a,b,c] [--budget-min 37]

a  exps.r1_agree: llama-perplexity KL against Bonsai 2 27B PQ2_0 for the Qwen3.5-0.8B draft in BF16, Q8_0 and a post-hoc
   PTQ1_0 (llama-quantize on the pod): top-1 agreement and mean KL on a fixed 8192-token text.
b  exps.e5_spec matrix: 200 prompts in a fixed-seed, category-balanced order; one llama-server per config: target alone,
   draft-simple K=4/3/2 with draft backend sampling, K=4 without (control), ngram-mod, ngram-map-k (fork defaults, and a
   short n-gram variant on the code categories). Rows keep the output text. Per category: acceptance r and alpha, tok/s,
   speedup against target alone and greedy output equality, all on prompts both runs finished.
c  exps.r1_step: one nsys capture of ~20 K=4 steps, split into verify, draft, copies and idle time, parsed to JSON.
Time: every part gets a share of --budget-min (weights 7 : 24 : 6, unused time rolls forward) and the pod deadline caps it
all. Part b divides its share over the configs by expected cost, so a short budget gives fewer paired prompts per config,
never a missing config. Partial results are written to raw/partial.json after every config and every ten prompts.
Result: results/r1/<gpu>/<stamp>.json with data.agree, data.e5, data.step and data.verdict (inputs of the E6 go/no-go).
Needs llama-server, llama-perplexity, llama-quantize and, for c, nsys (CR_NSYS or PATH); a missing tool skips its part.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import pathlib
import sys
import tempfile
import time

from exps import common, e4_bonsai, e4_pdl, e5_spec, make_prompts, r1_agree, r1_step

EXP = "r1"
WEIGHTS = {"a": 7.0, "b": 24.0, "c": 6.0}
DEFAULT_TOTAL_MIN = 37.0
END_RESERVE_S = 25.0
NEEDS = {"a": ("llama-perplexity", "llama-quantize"), "b": ("llama-server",), "c": ("llama-server",)}
SECRET_ENV_RE = e4_bonsai.SECRET_ENV_RE
E6_THRESHOLDS = {"overhead_ms": 2.0, "agreement": 0.5, "ptq1_alpha_drop": 0.05, "ngram_speedup": 1.15}


# ---------------------------------------------------------------- pure helpers

def part_budget_s(left_s: float, part: str, remaining: list[str]) -> float:
    """Seconds for `part` from left_s: its weight share of the remaining parts, less the end reserve."""
    tot = sum(WEIGHTS[p] for p in remaining) or 1.0
    return max(left_s - END_RESERVE_S, 0.0) * WEIGHTS[part] / tot


def secret_values(environ=None) -> list[str]:
    env = os.environ if environ is None else environ
    return [v for k, v in env.items() if SECRET_ENV_RE.search(k) and len(v) >= 8]


def scrub(obj, secrets: list[str]):
    """Replace secret values in every string of a JSON-like object."""
    if not secrets:
        return obj
    if isinstance(obj, str):
        for s in secrets:
            obj = obj.replace(s, "<redacted>")
        return obj
    if isinstance(obj, dict):
        return {k: scrub(v, secrets) for k, v in obj.items()}
    if isinstance(obj, list):
        return [scrub(v, secrets) for v in obj]
    return obj


def verdict(doc: dict) -> dict:
    """Inputs of the three E6 go/no-go criteria, with booleans where the data allows. None means not measured.

    c1 per-step overhead can be cut to <= 2 ms: step.analysis c1_floor (what stays if backend sampling and no re-eval take
       the removable part), plus the K=4 gain of draft backend sampling from part b.
    c2 0.8B top-1 agreement >= 0.5 and post-hoc PTQ1_0 within 0.05 of BF16.
    c3 n-gram speculation gives < 1.15x on code_fix and refactor (go) or >= 1.15x (no-go).
    """
    out: dict = {"thresholds": E6_THRESHOLDS}
    an = ((doc.get("step") or {}).get("analysis")) or {}
    med = an.get("median_ms") or {}
    c1: dict = {"measured": bool(an.get("ok"))}
    if an.get("ok"):
        c1.update(step_wall_ms=med.get("wall"), non_compute_ms=med.get("non_compute"), removable_ms=med.get("c1_removable"),
                  floor_ms=med.get("c1_floor"), floor_le_2ms=med.get("c1_floor", 1e9) <= E6_THRESHOLDS["overhead_ms"])
    e5s = ((doc.get("e5") or {}).get("summary")) or {}
    bs, nobs = (e5s.get("k4_bs") or {}).get("all"), (e5s.get("k4_nobs") or {}).get("all")
    if bs and nobs and bs.get("tok_s") and nobs.get("tok_s"):
        c1["k4_backend_sampling_gain"] = bs["tok_s"] / nobs["tok_s"]
    out["c1_overhead"] = c1
    ag = ((doc.get("agree") or {}).get("summary")) or {}
    out["c2_agreement"] = {"bf16": ag.get("bf16_agreement"), "ptq1_0": ag.get("ptq1_0_agreement"),
                           "bf16_ge_0p5": ag.get("bf16_ge_0p5"), "ptq1_delta": ag.get("ptq1_delta_vs_bf16"),
                           "ptq1_within_0p05": ag.get("ptq1_within_0p05")}
    best: dict = {}
    for name in ("ngram_mod", "ngram_map_k", "ngram_mod_short", "ngram_map_k_short"):
        cats = ((e5s.get(name) or {}).get("by_category")) or {}
        for cat in e5_spec.CODE_CATS:
            sp = (cats.get(cat) or {}).get("speedup")
            if sp is not None:
                best[f"{name}/{cat}"] = {"speedup": sp, "n": cats[cat]["n"]}
    top = max((v["speedup"] for v in best.values()), default=None)
    out["c3_ngram"] = {"by_config_category": best, "best_speedup": top,
                       "ngram_ge_1p15": None if top is None else top >= E6_THRESHOLDS["ngram_speedup"],
                       "go": None if top is None else top < E6_THRESHOLDS["ngram_speedup"]}
    return out


def latest_cost_summary(root=None) -> dict | None:
    """summary of the newest committed e5_cost resummary (or result) with a plain baseline, for the step-model residual."""
    root = pathlib.Path(root) if root else common.EXP_DIR / "results" / "e5_cost"
    for pat in ("*/*.resummary.json", "*/2*Z.json"):
        for f in sorted(glob.glob(str(root / pat)), reverse=True):
            try:
                s = json.loads(pathlib.Path(f).read_text())["data"]["summary"]
            except (OSError, ValueError, KeyError):
                continue
            if (s.get("baseline") or {}).get("plain_ms"):
                s["_file"] = os.path.basename(f)
                return s
    return None


def e5_residuals(summary: dict, cost: dict | None) -> dict:
    out = {}
    for name, ent in summary.items():
        if ent.get("spec_type") == "draft-simple" and ent.get("all"):
            r = e5_spec.step_residual(ent["all"], int(ent["k"]), cost)
            if r:
                out[name] = r
    return out


# ---------------------------------------------------------------- parts

def run_e5(*, args, bin_dir, target: str, draft: str, budget_s: float, notes: list, gpu: str, on_update) -> dict:
    from infra.results import artifact_path
    prompts = make_prompts.shuffled(make_prompts.load(args.prompts), args.seed, stratify=True)[:args.limit]
    cfgs = [c for c in e5_spec.matrix_configs() if c["name"] in args.configs]
    runs: dict = {}
    out: dict = {"seed": args.seed, "order": [p["id"] for p in prompts], "n_predict": args.n_predict, "runs": runs, "summary": {}}
    t_end = time.monotonic() + budget_s
    last_save = [0.0]

    def refresh() -> None:
        out["summary"] = e5_spec.summarize_matrix(runs)
        on_update(out)

    for i, cfg in enumerate(cfgs):
        left = t_end - time.monotonic()
        if left < 60:
            notes.append(f"e5: time budget used up; {[c['name'] for c in cfgs[i:]]} skipped")
            break
        share = e5_spec.time_shares(cfgs[i:], left)[0]
        base_ids = {r["id"] for r in (runs.get("base") or {}).get("rows", []) if "error" not in r}
        ps = prompts if cfg["name"] == "base" or not base_ids else [p for p in prompts if p["id"] in base_ids]
        if cfg["categories"]:
            ps = [p for p in ps if p["category"] in cfg["categories"]]
        if not ps:
            notes.append(f"e5: {cfg['name']} has no prompts")
            continue
        runs[cfg["name"]] = {"name": cfg["name"], "spec_type": cfg["spec_type"], "k": cfg["k"], "rows": [], "planned": len(ps), "partial": True}

        def on_row(rows, name=cfg["name"]) -> None:
            runs[name]["rows"] = list(rows)
            if time.monotonic() - last_save[0] > 15:
                last_save[0] = time.monotonic()
                refresh()
        res = e5_spec.run_config(cfg, bin_dir=bin_dir, target=target, draft=draft, prompts=ps, n_predict=args.n_predict,
                                 budget_s=share, notes=notes, log_path=str(artifact_path(EXP, f"server-{cfg['name']}.log", gpu)),
                                 on_row=on_row)
        res["desc"], res["extra_args"] = cfg["desc"], cfg["extra"]
        runs[cfg["name"]] = res
        if res.get("error"):
            notes.append(f"e5 {cfg['name']}: {res['error']}")
        elif (res.get("chat") or {}).get("draft_not_used"):
            notes.append(f"e5 {cfg['name']}: draft-simple drafted 0 tokens in the first prompts (spec log: {res.get('spec_log')})")
        refresh()
    out["cost_model"] = None
    cost = latest_cost_summary()
    if cost:
        out["cost_model"] = {"file": cost.get("_file"), "plain_ms": cost["baseline"]["plain_ms"]}
        out["step_residual"] = e5_residuals(out["summary"], cost)
    return out


def run_step(*, bin_dir, target: str, draft: str, budget: common.Budget, notes: list, gpu: str, prompts_file) -> dict:
    from infra.results import artifact_path
    nsys = e4_bonsai.find_nsys()
    if not nsys:
        return {"ok": False, "error": "nsys not found (CR_NSYS, PATH, /usr/local/cuda*/bin, /opt/nvidia/nsight-systems)"}
    cfg = next(c for c in e5_spec.matrix_configs() if c["name"] == "k4_bs")
    rows = make_prompts.load(prompts_file)
    prompt = next((r["prompt"] for r in rows if r["category"] == "code_fix"), rows[0]["prompt"])
    wd = r1_step.private_workdir()
    res = r1_step.capture(nsys=nsys, server_argv=e5_spec.config_cmd(bin_dir, target, draft, cfg), port=e5_spec.PORT, prompt=prompt,
                          budget=budget, workdir=wd, log_path=str(artifact_path(EXP, "server-nsys-k4.log", gpu)), notes=notes)
    res["leftover_files"] = sorted(glob.glob(os.path.join(wd, "*"))) if os.path.isdir(wd) else []
    return res


# ---------------------------------------------------------------- run

def summary_rows(doc: dict) -> list[dict]:
    rows = []
    for name, v in ((doc.get("agree") or {}).get("variants") or {}).items():
        rows.append({"part": "a", "name": name, "same_top": v.get("same_top"), "mean_kld": v.get("mean_kld"), "error": v.get("error")})
    for name, v in ((doc.get("e5") or {}).get("summary") or {}).items():
        a = v.get("all") or {}
        rows.append({"part": "b", "name": name, "n_paired": v.get("n_paired"), "tok_s": a.get("tok_s"), "speedup": a.get("speedup"),
                     "r": a.get("r"), "alpha": a.get("alpha"), "text_equal_frac": a.get("text_equal_frac"), "error": v.get("error")})
    st = doc.get("step") or {}
    if st:
        rows.append({"part": "c", "name": "nsys_k4", "ok": st.get("ok"), "error": st.get("error"),
                     "median_ms": ((st.get("analysis") or {}).get("median_ms"))})
    return rows


def run(args) -> int:
    notes: list[str] = []
    args.gpu = common.gpu_tag(args)
    bin_dir = e4_bonsai.llama_bin()
    parts = [p for p in "abc" if p in args.parts]
    runnable = []
    for p in parts:
        miss = [b for b in NEEDS[p] if not (bin_dir / b).exists()]
        if miss:
            notes.append(f"part {p} skipped: {miss} not found in {bin_dir} (build llama.cpp with CR_BUILD_LLAMA=1)")
        else:
            runnable.append(p)
    total = common.Budget(args.budget_min)
    secrets = secret_values()
    config = {"parts": parts, "budget_min": args.budget_min, "weights": WEIGHTS, "seed": args.seed, "limit": args.limit,
              "n_predict": args.n_predict, "configs": args.configs, "variants": args.variants, "quick": args.quick,
              "thresholds": E6_THRESHOLDS, "prompts_file": str(args.prompts), "models": {}}
    doc: dict = {"agree": None, "e5": None, "step": None}

    def save(final: bool = False) -> None:
        doc["verdict"] = verdict(doc)
        extra = scrub(doc, secrets)
        rows = summary_rows(doc)
        if final:
            common.write_result(EXP, args, config, rows, notes, False, extra=extra)
        else:
            common.write_partial(EXP, args, config, rows, notes, False, extra=extra)

    paths: dict = {}
    try:
        for name in ("pq2", "draft_bf16", "draft_q8_0"):
            if name == "draft_q8_0" and "a" not in runnable:
                continue
            paths[name], config["models"][name] = e4_pdl.model_path(name, notes)
    except Exception as e:
        notes.append(f"model download failed: {type(e).__name__}: {e}")
        save(final=True)
        print(f"error: {notes[-1]}", file=sys.stderr)
        return 3
    from infra.results import run_dir
    (run_dir(EXP, args.gpu) / "raw").mkdir(parents=True, exist_ok=True)
    smi = e4_pdl.smi_sample()
    bad = e4_pdl.health_problem(e4_pdl.smi_summary(smi, [], smi), check_power=not args.allow_power_cap) if smi else None
    config["host_smi"] = smi
    if bad:
        notes.append(f"ABORT: unhealthy host: {bad}")
        save(final=True)
        return 1
    for i, p in enumerate(runnable):
        budget_s = part_budget_s(total.left(), p, runnable[i:])
        print(f"[r1] part {p}: {budget_s / 60:.1f} min of {total.left() / 60:.1f} min left", flush=True)
        if budget_s < 60:
            notes.append(f"part {p} skipped: only {budget_s:.0f} s of budget left")
            continue
        sub = common.Budget(budget_s / 60.0)
        try:
            if p == "a":
                with tempfile.TemporaryDirectory(prefix="cr-r1a-") as wd:
                    doc["agree"] = r1_agree.run_agree(
                        bin_dir=bin_dir, target=paths["pq2"], drafts={"bf16": paths["draft_bf16"], "q8_0": paths["draft_q8_0"]},
                        workdir=wd, budget=sub, notes=notes, variants=args.variants,
                        log_dir=str(run_dir(EXP, args.gpu) / "raw"),
                        on_update=lambda a: (doc.__setitem__("agree", a), save()))
            elif p == "b":
                doc["e5"] = run_e5(args=args, bin_dir=bin_dir, target=paths["pq2"], draft=paths["draft_bf16"], budget_s=budget_s,
                                   notes=notes, gpu=args.gpu, on_update=lambda e: (doc.__setitem__("e5", e), save()))
            else:
                doc["step"] = run_step(bin_dir=bin_dir, target=paths["pq2"], draft=paths["draft_bf16"], budget=sub, notes=notes,
                                       gpu=args.gpu, prompts_file=args.prompts)
        except Exception as e:  # a failed part must not lose the others
            notes.append(f"part {p} crashed: {type(e).__name__}: {e}")
        save()
    notes.append("post-hoc PTQ1_0/PQ2_0 in this fork scale each 128 group by its absolute maximum and ignore the imatrix "
                 "(ggml-quants.c quantize_ptq1_0), so a collapse in agreement says the recipe is naive, not that ternary 0.8B is hopeless")
    notes.append("draft backend sampling is ON by default in this fork (common.h draft.backend_sampling = true): k4_nobs is the "
                 "--no-spec-draft-backend-sampling control, and the earlier E5 runs already had it on")
    save(final=True)
    return 0


# ---------------------------------------------------------------- CLI

def _names(text):
    return [t.strip() for t in text.split(",") if t.strip()]


def parse_args(argv=None):
    ap = argparse.ArgumentParser(prog="exps.r1", description=__doc__.split("\n")[0])
    common.add_common_args(ap)
    ap.add_argument("--parts", type=_names, default=list("abc"), help="a (agreement), b (E5 matrix), c (nsys step split)")
    ap.add_argument("--limit", type=int, default=200, help="prompts, taken from the fixed-seed order")
    ap.add_argument("--seed", type=int, default=e5_spec.SEED)
    ap.add_argument("--n-predict", type=int, default=192)
    ap.add_argument("--prompts", default=str(make_prompts.DEFAULT_OUT))
    ap.add_argument("--configs", type=_names, default=[c["name"] for c in e5_spec.matrix_configs()])
    ap.add_argument("--variants", type=_names, default=list(r1_agree.VARIANTS))
    ap.add_argument("--allow-power-cap", action="store_true", help="a power cap below default is a note, not an abort")
    a = ap.parse_args(argv)
    bad = [p for p in a.parts if p not in WEIGHTS]
    names = {c["name"] for c in e5_spec.matrix_configs()}
    badc = [c for c in a.configs if c not in names]
    if bad or badc:
        ap.error(f"unknown part(s) {bad} or config(s) {badc}; configs: {sorted(names)}")
    if a.configs and "base" not in a.configs and "b" in a.parts:
        ap.error("--configs must include base: every ratio is against target alone")
    a.budget_min = a.budget_min or (8.0 if a.quick else DEFAULT_TOTAL_MIN)
    if a.quick:
        a.limit, a.n_predict = min(a.limit, 8), min(a.n_predict, 32)
    return a


def plan_json(a) -> dict:
    bin_dir = e4_bonsai.llama_bin()
    cfgs = [c for c in e5_spec.matrix_configs() if c["name"] in a.configs]
    k4 = next(c for c in e5_spec.matrix_configs() if c["name"] == "k4_bs")
    text, meta = r1_agree.build_text()
    return {"exp": EXP, "parts": a.parts, "budget_min": a.budget_min, "weights": WEIGHTS,
            "agree": {"variants": a.variants, "text": meta, "base_cmd": r1_agree.perplexity_cmd(bin_dir, "<pq2.gguf>", "<text>", "<logprobs>", False),
                      "kl_cmd": r1_agree.perplexity_cmd(bin_dir, "<draft.gguf>", "<text>", "<logprobs>", True),
                      "quantize_cmd": r1_agree.quantize_cmd(bin_dir, "<draft-bf16.gguf>", "<out.gguf>", "PTQ1_0")},
            "e5": {"seed": a.seed, "n_prompts": len(make_prompts.load(a.prompts)[:a.limit]), "n_predict": a.n_predict,
                   "configs": [{"name": c["name"], "desc": c["desc"], "weight": c["weight"], "categories": c["categories"],
                                "cmd": e5_spec.config_cmd(bin_dir, "<pq2.gguf>", "<draft.gguf>", c)} for c in cfgs]},
            "step": {"server_cmd": e5_spec.config_cmd(bin_dir, "<pq2.gguf>", "<draft.gguf>", k4),
                     "nsys_profile": r1_step.profile_cmd("nsys", "<tmp>/r1step", ["<server_cmd>"]),
                     "nsys_start": r1_step.control_cmd("nsys", "start"), "nsys_stop": r1_step.control_cmd("nsys", "stop"),
                     "env": "secret-like variables removed", "files": "private temp dir, deleted after parsing"},
            "quick": a.quick}


def main(argv=None) -> int:
    a = parse_args(argv)
    if a.dry_run:
        common.print_plan(plan_json(a))
        return 0
    with common.term_exits():
        return run(a)


if __name__ == "__main__":
    sys.exit(main())
