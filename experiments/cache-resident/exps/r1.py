"""R1: the pre-E6 measurements in one pod session: (a) draft agreement, (b) full E5 matrix, (c) per-step overhead.

  python -m exps.fetch_models --models pq2,draft_bf16,draft_q8_0 --timeout-min 6 && python -m exps.r1 [--parts a,b,c] [--budget-min 29]

Parts run in the order b, a, c: a builds its text from the target-alone outputs of b.
b  exps.e5_spec matrix: category-balanced fixed-seed prompts, one llama-server per config: target alone, draft-simple K=4
   (control, as in E5), K=4 with target backend sampling (-bs), K=4 with the post-hoc PTQ1_0 draft, and ngram-mod /
   ngram-map-k (fork defaults and short variants) on the code categories. K=2/3 exist (--configs) but are not default: they
   cost about 20% of the part and wait for a follow-up pod. The number of prompts N is fixed up front from a per-prompt cost
   model so that all configs fit the part budget (e5_spec.plan_n); base runs N prompts and every other config a subset. Base and
   the n-gram configs also run extra code_fix / refactor prompts past the N-prefix (--code-floor per category, default 20), so
   criterion 3 reaches n >= 15 whatever N is. Rows keep the output text. Per category: acceptance r (and alpha for draft-simple), tok/s, speedup with a
   bootstrap 95% interval against target alone, and greedy output equality, all on prompts both runs finished.
   ngram-mod keeps its token pool across requests on one server (cleared at 25% occupancy), so a prompt sees what earlier
   prompts of the same server left; the code-only configs build a different pool than a full run would. Recorded, not removed.
   Rollback path: need_n_rs_seq() (common.h) gives the target context n_rs_seq = 0 for ngram-mod / ngram-map-k, so a hybrid
   (GDN) target takes the KV-checkpoint path (save at every drafting step, restore and replay on partial acceptance), which
   draft-simple (n_rs_seq = K) does not pay. Each server log's `n_rs_seq` INFO line is parsed into checks.rollback_path and
   checks.checkpoint_replay; criterion 3 is then a lower bound on n-gram speedup (a go is provisional, see ngram_decision).
a  exps.r1_agree: llama-perplexity KL against Bonsai 2 27B PQ2_0 for the Qwen3.5-0.8B draft in BF16, Q8_0 and a post-hoc
   PTQ1_0 (llama-quantize on the pod): top-1 agreement and mean KL on a fixed 8192-token text. Primary text: the chat-templated
   E5 prompts with the target's own greedy outputs from part b (the distribution alpha is measured on); second text: raw repo code.
c  exps.r1_step: one nsys capture of ~20 K=4 steps (plain k4 config), split by stream role into verify, draft re-eval, draft
   steps, copies and idle, parsed to JSON, then compared with the unprofiled step time of part b (step_wall_check).
Time: the pod failsafe is 55 min: bootstrap with the cold build ~14, downloads <= 6 (fetch_models --timeout-min 6), r1 29, end
reserve 5. Every part gets a share of --budget-min (weights 6 : 26 : 5, unused time rolls forward) and the pod deadline caps it
all. Partial results are written to raw/partial.json after every config and every ten prompts, and a run cut by SIGTERM
still writes its result (the notes say which part was cut).
Result: results/r1/<gpu>/<stamp>.json with data.agree, data.e5, data.step and data.verdict (inputs of the E6 go/no-go).
Needs llama-server, llama-perplexity, llama-quantize and, for c, nsys (CR_NSYS or PATH); a missing tool skips its part.
An unhealthy host (throttled clocks, or a power cap without --allow-power-cap) still runs part a, which has no timing.
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

from exps import common, e4_bonsai, e4_pdl, e5_spec, make_prompts, r1_agree, r1_step, spec_math

EXP = "r1"
WEIGHTS = {"a": 6.0, "b": 26.0, "c": 5.0}
PART_ORDER = "bac"
DEFAULT_TOTAL_MIN = 29.0   # 55 min failsafe - 5 reserve - ~14 bootstrap and build - <= 6 downloads
END_RESERVE_S = 25.0
NEEDS = {"a": ("llama-perplexity", "llama-quantize"), "b": ("llama-server",), "c": ("llama-server",)}
SECRET_ENV_RE = e4_bonsai.SECRET_ENV_RE
E6_THRESHOLDS = {"overhead_ms": 2.0, "agreement": 0.5, "ptq1_alpha_drop": 0.05, "ngram_speedup": 1.15,
                 "min_n_category": e5_spec.MIN_N_CATEGORY, "step_wall_tol": 0.10}
NGRAM_CONFIGS = ("ngram_mod", "ngram_map_k", "ngram_mod_short", "ngram_map_k_short")
NGRAM_ROLLBACK_NOTE = ("the n-gram types get n_rs_seq = 0 in need_n_rs_seq() (common.h), so a hybrid target runs the KV-checkpoint "
                       "path with a replay of the target pass on every partial acceptance, which draft-simple does not pay; the "
                       "n-gram speedup here is a lower bound, and a c3 go is provisional until n-gram is measured with rollback "
                       "(need_n_rs_seq extended to the n-gram types in a test build)")
NGRAM_STATE_NOTE = ("ngram-mod keeps its token pool across requests on one server (cleared at 25% occupancy, seeded by the "
                    "warm-up request), so results depend on prompt order and the code-only configs see a different pool than a "
                    "full-category run would")


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
    if isinstance(obj, (list, tuple)):
        return [scrub(v, secrets) for v in obj]
    return obj


def step_wall_check(analysis: dict | None, request: dict | None, partb_ms: float | None, tol: float = E6_THRESHOLDS["step_wall_tol"]) -> dict:
    """Median step wall of the nsys capture against the request's own timing (does the step split match what the server saw)
    and against the unprofiled part b step time (node-level graph tracing adds launch overhead). A gap above tol is flagged."""
    med = (((analysis or {}).get("median_ms")) or {}).get("wall")
    req_ms = None
    if request and request.get("predicted_ms"):
        req_ms = spec_math.observed_step_ms(request["predicted_ms"], request.get("tokens_predicted", 0),
                                            request.get("draft_n_accepted", 0), n_first=1)

    def gap(a, b):
        return None if not a or not b else a / b - 1.0
    vs_req, vs_b = gap(med, req_ms), gap(med, partb_ms)
    return {"tol": tol, "nsys_step_wall_ms": med, "request_ms_per_step": req_ms, "partb_ms_per_step": partb_ms,
            "vs_request": vs_req, "vs_partb": vs_b,
            "split_matches_request": None if vs_req is None else abs(vs_req) <= tol,
            "profiler_overhead_flag": None if vs_b is None else vs_b > tol,
            "profiler_overhead_ms": None if med is None or partb_ms is None else med - partb_ms}


def _ngram_cells(e5: dict) -> tuple[dict, list[str]]:
    """{config/category: {speedup, ci95, n}} for the n-gram configs on the code categories, and the configs that did not run."""
    summ = e5.get("summary") or {}
    cells: dict = {}
    for name in NGRAM_CONFIGS:
        cats = (summ.get(name) or {}).get("by_category") or {}
        for cat in e5_spec.CODE_CATS:
            ent = cats.get(cat)
            if ent and ent.get("speedup") is not None:
                cells[f"{name}/{cat}"] = {"speedup": ent["speedup"], "ci95": ent.get("speedup_ci95"), "n": ent["n"]}
    asked = [c for c in (e5.get("configs") or []) if c in NGRAM_CONFIGS]
    return cells, [c for c in asked if c not in summ]


def _ngram_rollback(e5: dict) -> dict:
    """{config: rollback info} of the n-gram configs from the server log checks, and the overall checkpoint_replay flag."""
    summ = e5.get("summary") or {}
    per = {name: {k: (summ[name]["checks"] or {}).get(k) for k in ("n_rs_seq_target", "rollback_path", "checkpoint_replay")}
           for name in NGRAM_CONFIGS if (summ.get(name) or {}).get("checks")}
    flags = [v["checkpoint_replay"] for v in per.values()]
    replay = True if any(f is True for f in flags) else (False if flags and all(f is False for f in flags) else None)
    return {"by_config": per, "checkpoint_replay": replay}


def ngram_decision(e5: dict) -> dict:
    """Criterion 3 from the n-gram cells. A cell decides only with n >= MIN_N_CATEGORY and a bootstrap interval; the decision
    uses the interval lower bound, never the largest of the noisy point estimates.
      no_go          some cell has CI lower bound >= 1.15: n-gram speculation already gives the speedup.
      go             every asked cell is decided, none has CI lower >= 1.15 and no point estimate reaches 1.15.
      straddles      a point estimate reaches 1.15 but its interval includes less.
      insufficient_n a cell has too few prompts (or its config did not run): no decision.
    On the KV-checkpoint path (checkpoint_replay true, or unknown) every speedup is a lower bound: a no_go stands, a go is
    provisional (out["provisional"]) until n-gram is measured with rollback.
    """
    thr, min_n = E6_THRESHOLDS["ngram_speedup"], E6_THRESHOLDS["min_n_category"]
    cells, missing = _ngram_cells(e5)
    out: dict = {"by_config_category": cells, "configs_missing": missing,
                 "best_speedup": max((c["speedup"] for c in cells.values()), default=None)}
    weak = [k for k, c in cells.items() if c["n"] < min_n or not c["ci95"]]
    strong = {k: c for k, c in cells.items() if k not in weak}
    rb = _ngram_rollback(e5)
    out["rollback"] = rb["by_config"]
    out["checkpoint_replay"] = rb["checkpoint_replay"]
    out["speedup_is"] = "lower bound on n-gram speedup" if rb["checkpoint_replay"] is not False else "n-gram speedup"
    out["weak_cells"] = weak
    out["best_ci_lower"] = max((c["ci95"][0] for c in strong.values()), default=None)
    if not cells:
        out.update(status="not_measured", ngram_ge_1p15=None, go=None)
    elif any(c["ci95"][0] >= thr for c in strong.values()):
        out.update(status="no_go", ngram_ge_1p15=True, go=False)
    elif weak or missing:
        out.update(status="insufficient_n", ngram_ge_1p15=None, go=None)
    elif any(c["speedup"] >= thr for c in cells.values()):
        out.update(status="straddles", ngram_ge_1p15=None, go=None)
    else:
        out.update(status="go", ngram_ge_1p15=False, go=True)
    out["provisional"] = out["status"] == "go" and rb["checkpoint_replay"] is not False
    if out["provisional"]:
        out["provisional_note"] = NGRAM_ROLLBACK_NOTE
    return out


def verdict(doc: dict) -> dict:
    """Inputs of the three E6 go/no-go criteria, with booleans where the data allows. None means not measured.

    c1 per-step overhead can be cut to <= 2 ms: step.analysis c1_floor (what stays if backend sampling and no re-eval take
       the removable part; also with the nsys launch overhead taken off, see wall_check), plus the K=4 gain of target
       backend sampling (k4_tbs against k4, same prompts) from part b.
    c2 0.8B top-1 agreement >= 0.5 and post-hoc PTQ1_0 within 0.05 of BF16 (measured alpha of the k4_ptq1 run when there is one,
       else the KL proxy). The teacher-forced value sits next to the K=4 alpha of part b as a calibration of the proxy.
    c3 n-gram speculation gives < 1.15x on code_fix and refactor (go) or >= 1.15x (no-go), see ngram_decision. Measured on
       the KV-checkpoint path, so a go is provisional (a lower bound on n-gram speedup).
    """
    out: dict = {"thresholds": E6_THRESHOLDS, "host": doc.get("host")}
    st = doc.get("step") or {}
    an = st.get("analysis") or {}
    med = an.get("median_ms") or {}
    c1: dict = {"measured": bool(an.get("ok"))}
    if an.get("ok"):
        floor = med.get("c1_floor")
        c1.update(step_wall_ms=med.get("wall"), non_compute_ms=med.get("non_compute"), removable_ms=med.get("c1_removable"),
                  floor_ms=floor, floor_le_2ms=floor is not None and floor <= E6_THRESHOLDS["overhead_ms"],
                  split_mode=an.get("split_mode"), checks=an.get("checks"))
        wc = st.get("wall_check")
        if wc:
            c1["wall_check"] = wc
            ov = wc.get("profiler_overhead_ms")
            if ov is not None and floor is not None:
                c1["floor_ms_minus_profiler_overhead"] = max(floor - max(ov, 0.0), 0.0)
                c1["floor_adj_le_2ms"] = c1["floor_ms_minus_profiler_overhead"] <= E6_THRESHOLDS["overhead_ms"]
    e5 = doc.get("e5") or {}
    pair = (e5.get("pairs") or {}).get("k4_tbs_vs_k4")
    if pair and pair.get("speedup") is not None:
        c1["k4_tbs_gain_vs_k4"] = pair["speedup"]
        c1["k4_tbs_gain_ci95"] = pair.get("speedup_ci95")
        c1["k4_tbs_n_paired"] = pair.get("n")
        chk = ((e5.get("summary") or {}).get("k4_tbs") or {}).get("checks") or {}
        c1["target_backend_sampling_reported"] = chk.get("backend_sampling_reported")
    out["c1_overhead"] = c1

    ag = doc.get("agree") or {}
    sm = ag.get("summary") or {}
    c2: dict = {"text_kind": (ag.get("text") or {}).get("kind", "repo" if ag else None), "bf16": sm.get("bf16_agreement"),
                "q8_0": sm.get("q8_0_agreement"), "ptq1_0": sm.get("ptq1_0_agreement"), "bf16_ge_0p5": sm.get("bf16_ge_0p5"),
                "ptq1_delta": sm.get("ptq1_delta_vs_bf16"), "ptq1_within_0p05": sm.get("ptq1_within_0p05"), "ptq1_basis": "kl_proxy"}
    k4 = ((e5.get("summary") or {}).get("k4") or {}).get("all") or {}
    if k4.get("alpha") is not None:
        c2["calibration"] = {"e5_k4_alpha": k4["alpha"], "e5_k4_r": k4.get("r"),
                             "teacher_forced_bf16_minus_alpha": None if c2["bf16"] is None else c2["bf16"] - k4["alpha"]}
    c2["secondary_text"] = {kind: {"bf16": (r.get("summary") or {}).get("bf16_agreement"), "ptq1_0": (r.get("summary") or {}).get("ptq1_0_agreement")}
                            for kind, r in (ag.get("secondary") or {}).items()}
    pq = (e5.get("pairs") or {}).get("k4_ptq1_vs_k4")
    if pq and pq.get("alpha") is not None and pq.get("base_alpha") is not None:
        drop = pq["base_alpha"] - pq["alpha"]
        c2.update(e5_alpha_k4=pq["base_alpha"], e5_alpha_k4_ptq1=pq["alpha"], e5_ptq1_alpha_drop=drop, e5_ptq1_n_paired=pq.get("n"),
                  e5_ptq1_within_0p05=drop <= E6_THRESHOLDS["ptq1_alpha_drop"])
        if (pq.get("n") or 0) >= E6_THRESHOLDS["min_n_category"]:
            c2["ptq1_within_0p05"], c2["ptq1_basis"] = c2["e5_ptq1_within_0p05"], "e5_k4"
    out["c2_agreement"] = c2
    out["c3_ngram"] = ngram_decision(e5)
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
    """Step-model residual per BF16 draft-simple config, with and without the draft re-eval term."""
    out = {}
    for name, ent in summary.items():
        if ent.get("spec_type") == "draft-simple" and ent.get("all") and (ent.get("draft") or "bf16") == "bf16":
            r = e5_spec.step_residual(ent["all"], int(ent["k"]), cost)
            if r:
                out[name] = r
    return out


# ---------------------------------------------------------------- parts

def run_e5(*, args, bin_dir, target: str, drafts: dict, budget_s: float, notes: list, gpu: str, on_update) -> dict:
    from infra.results import artifact_path
    prompts = make_prompts.shuffled(make_prompts.load(args.prompts), args.seed, stratify=True)[:args.limit]
    cfgs = []
    for c in e5_spec.matrix_configs():
        if c["name"] not in args.configs:
            continue
        if c["spec_type"] == "draft-simple" and not drafts.get(c["draft"]):
            notes.append(f"e5: {c['name']} skipped, no {c['draft']} draft GGUF")
            continue
        cfgs.append(c)
    plan = e5_spec.plan_n(cfgs, prompts, budget_s, code_floor=args.code_floor)
    main = prompts[:plan["n"]]
    extra = e5_spec.code_extra(prompts, plan["n"], args.code_floor)
    if not plan["fits"]:
        notes.append(f"e5: the matrix does not fit {budget_s:.0f} s even at {plan['n']} prompts (expected {plan['est_s']:.0f} s)")
    if plan["min_code_category"] < e5_spec.MIN_N_CATEGORY:
        notes.append(f"e5: {plan['min_code_category']} prompts per code category (< {e5_spec.MIN_N_CATEGORY}): criterion 3 cannot be decided")
    runs: dict = {}
    out: dict = {"seed": args.seed, "order": [p["id"] for p in prompts], "n_main": plan["n"], "n_extra": len(extra), "plan": plan, "n_predict": args.n_predict,
                 "configs": [c["name"] for c in cfgs], "runs": runs, "summary": {}, "pairs": {}, "state_note": NGRAM_STATE_NOTE,
                 "rollback_note": NGRAM_ROLLBACK_NOTE}
    t_end = time.monotonic() + budget_s
    last_save = [0.0]

    def refresh(final: bool = False) -> None:
        out["summary"] = e5_spec.summarize_matrix(runs, ci=final)
        out["pairs"] = {key: e5_spec.compare_pair(runs, a, b, ci=final) for key, (a, b) in
                        {"k4_tbs_vs_k4": ("k4_tbs", "k4"), "k4_ptq1_vs_k4": ("k4_ptq1", "k4")}.items() if a in runs and b in runs}
        on_update(out)

    for i, cfg in enumerate(cfgs):
        left = t_end - time.monotonic()
        if left < 60:
            notes.append(f"e5: time budget used up; {[c['name'] for c in cfgs[i:]]} skipped")
            break
        base_ids = {r["id"] for r in (runs.get("base") or {}).get("rows", []) if "error" not in r}
        ps = select_for(cfg, main, base_ids, extra)
        if not ps:
            notes.append(f"e5: {cfg['name']} has no prompts")
            continue
        costs = [e5_spec.cost_s(cfg, len(ps))] + [e5_spec.cost_s(c, len(e5_spec.select_prompts(c, main, extra))) for c in cfgs[i + 1:]]
        share = e5_spec.time_shares(costs, left)[0]
        runs[cfg["name"]] = {"name": cfg["name"], "spec_type": cfg["spec_type"], "k": cfg["k"], "draft": cfg["draft"], "rows": [],
                             "planned": len(ps), "partial": True}

        def on_row(rows, name=cfg["name"]) -> None:
            runs[name]["rows"] = list(rows)
            if time.monotonic() - last_save[0] > 15:
                last_save[0] = time.monotonic()
                refresh()
        res = e5_spec.run_config(cfg, bin_dir=bin_dir, target=target, draft=drafts.get(cfg["draft"]), prompts=ps, n_predict=args.n_predict,
                                 budget_s=share, notes=notes, log_path=str(artifact_path(EXP, f"server-{cfg['name']}.log", gpu)),
                                 on_row=on_row, keep_prompt=cfg["name"] == "base")
        res["desc"], res["extra_args"], res["draft"] = cfg["desc"], cfg["extra"], cfg["draft"]
        if cfg["spec_type"].startswith("ngram"):
            res["state_note"] = NGRAM_STATE_NOTE
        runs[cfg["name"]] = res
        notes.extend(config_notes(cfg["name"], res))
        refresh()
    refresh(final=True)
    out["cost_model"] = None
    cost = latest_cost_summary()
    if cost:
        out["cost_model"] = {"file": cost.get("_file"), "plain_ms": cost["baseline"]["plain_ms"]}
        out["step_residual"] = e5_residuals(out["summary"], cost)
    return out


def select_for(cfg: dict, main: list[dict], base_ids: set, extra: list[dict] = ()) -> list[dict]:
    """Prompts of one config: base takes all main and extra prompts, the others only those that base finished."""
    if cfg["name"] == "base":
        return e5_spec.select_prompts(cfg, main, extra)
    ps = e5_spec.select_prompts(cfg, main, extra)
    return [p for p in ps if p["id"] in base_ids] if base_ids else ps


def config_notes(name: str, res: dict) -> list[str]:
    out = []
    ck = res.get("checks") or {}
    n_err = sum(1 for r in res.get("rows") or [] if "error" in r)
    if n_err:
        first = next(r["error"] for r in res["rows"] if "error" in r)
        out.append(f"e5 {name}: {n_err} prompts failed, first: {first}")
    if res.get("error"):
        out.append(f"e5 {name}: {res['error']}")
    elif (res.get("chat") or {}).get("draft_not_used"):
        out.append(f"e5 {name}: draft-simple drafted 0 tokens in the first prompts (spec log: {res.get('spec_log')})")
    if ck.get("backend_sampling_expected") and ck.get("backend_sampling_reported") is not True:
        out.append(f"e5 {name}: -bs was asked but the server reports backend_sampling={ck.get('backend_sampling_reported')}")
    if ck.get("backend_sampling_problems"):
        out.append(f"e5 {name}: backend sampling lines in the server log: {ck['backend_sampling_problems']}")
    if ck.get("kv_checkpoint_fallback"):
        out.append(f"e5 {name}: server fell back to KV checkpoints (partial acceptance replays the target pass)")
    if name in NGRAM_CONFIGS and ck.get("checkpoint_replay") is not False:
        out.append(f"e5 {name}: rollback_path={ck.get('rollback_path')} (n_rs_seq={ck.get('n_rs_seq')}); {NGRAM_ROLLBACK_NOTE}")
    elif ck.get("rollback_path") == "checkpoint":
        out.append(f"e5 {name}: target n_rs_seq = 0, so partial acceptance replays the target pass")
    return out


def run_agree_part(*, bin_dir, paths: dict, doc: dict, budget: common.Budget, notes: list, args, gpu: str, root, quant_info, save) -> None:
    """Part a: the E5 chat text first (when part b left enough target-alone outputs), then the raw repo text if time allows."""
    from infra.results import run_dir
    base_rows = (((doc.get("e5") or {}).get("runs") or {}).get("base") or {}).get("rows") or []
    texts = []
    chat, cmeta = r1_agree.build_chat_text(base_rows)
    if chat:
        texts.append(("e5_chat", chat, cmeta))
    else:
        notes.append(f"part a: no E5 chat text ({cmeta['n_chars']} chars from {cmeta['n_samples']} target-alone outputs); repo text is the primary")
    rtext, rmeta = r1_agree.build_text(root)
    texts.append(("repo", rtext, dict(rmeta, kind="repo")))
    drafts = {"bf16": paths["draft_bf16"], "q8_0": paths.get("draft_q8_0")}
    if paths.get("draft_ptq1_0"):
        drafts["ptq1_0"] = paths["draft_ptq1_0"]
    results: dict = {}
    for j, (kind, text, meta) in enumerate(texts):
        if j > 0 and budget.expired(150):
            notes.append(f"part a: {kind} text skipped, {budget.left():.0f} s left")
            break

        def upd(a, j=j, kind=kind) -> None:
            if j == 0:
                a["text"] = meta
                doc["agree"] = a
            elif doc.get("agree") is not None:
                doc["agree"].setdefault("secondary", {})[kind] = a
            save()
        with tempfile.TemporaryDirectory(prefix="cr-r1a-") as wd:
            results[kind] = r1_agree.run_agree(
                bin_dir=bin_dir, target=paths["pq2"], drafts=drafts, workdir=wd, budget=budget, notes=notes, variants=args.variants,
                log_dir=str(run_dir(EXP, gpu) / "raw"), text=text, text_meta=meta, quant_info=quant_info, on_update=upd)
        results[kind]["text"] = meta
        if j == 0:
            doc["agree"] = results[kind]
        elif doc.get("agree") is not None:
            doc["agree"].setdefault("secondary", {})[kind] = results[kind]
        save()


def run_step(*, bin_dir, target: str, draft: str, budget: common.Budget, notes: list, gpu: str, prompts_file,
             cfg_name: str = "k4", on_stage=None) -> dict:
    from infra.results import artifact_path
    nsys = e4_bonsai.find_nsys()
    if not nsys:
        return {"ok": False, "error": "nsys not found (CR_NSYS, PATH, /usr/local/cuda*/bin, /opt/nvidia/nsight-systems)"}
    cfg = next(c for c in e5_spec.matrix_configs() if c["name"] == cfg_name)
    rows = make_prompts.load(prompts_file)
    prompt = next((r["prompt"] for r in rows if r["category"] == "code_fix"), rows[0]["prompt"])
    wd = r1_step.private_workdir()
    res = r1_step.capture(nsys=nsys, server_argv=e5_spec.config_cmd(bin_dir, target, draft, cfg), port=e5_spec.PORT, prompt=prompt,
                          budget=budget, workdir=wd, log_path=str(artifact_path(EXP, "server-nsys-k4.log", gpu)), notes=notes,
                          k=cfg["k"], on_stage=on_stage)
    res["config"] = cfg_name
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
                     "speedup_ci95": a.get("speedup_ci95"), "r": a.get("r"), "alpha": a.get("alpha"),
                     "text_equal_frac": a.get("text_equal_frac"), "error": v.get("error")})
    st = doc.get("step") or {}
    if st:
        rows.append({"part": "c", "name": "nsys_k4", "ok": st.get("ok"), "error": st.get("error"),
                     "median_ms": ((st.get("analysis") or {}).get("median_ms"))})
    return rows


def run(args) -> int:
    notes: list[str] = []
    args.gpu = common.gpu_tag(args)
    bin_dir = e4_bonsai.llama_bin()
    parts = [p for p in PART_ORDER if p in args.parts]
    runnable = []
    for p in parts:
        miss = [b for b in NEEDS[p] if not (bin_dir / b).exists()]
        if miss:
            notes.append(f"part {p} skipped: {miss} not found in {bin_dir} (build llama.cpp with CR_BUILD_LLAMA=1)")
        else:
            runnable.append(p)
    total = common.Budget(args.budget_min)
    secrets = secret_values()
    config = {"parts": parts, "part_order": PART_ORDER, "budget_min": args.budget_min, "weights": WEIGHTS, "seed": args.seed,
              "limit": args.limit, "code_floor": args.code_floor, "n_predict": args.n_predict, "configs": args.configs, "variants": args.variants, "quick": args.quick,
              "thresholds": E6_THRESHOLDS, "prompts_file": str(args.prompts), "step_config": args.step_config, "models": {}}
    doc: dict = {"agree": None, "e5": None, "step": None, "host": None}

    def save(final: bool = False) -> None:
        doc["verdict"] = verdict(doc)
        extra = scrub(doc, secrets)
        rows = summary_rows(doc)
        if final:
            common.write_result(EXP, args, scrub(config, secrets), rows, scrub(notes, secrets), False, extra=extra)
        else:
            common.write_partial(EXP, args, scrub(config, secrets), rows, scrub(notes, secrets), False, extra=extra)

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
    doc["host"] = {"problem": bad, "allow_power_cap": args.allow_power_cap, "parts_skipped": []}
    rc = 0
    if bad:
        skipped = [p for p in runnable if p != "a"]
        doc["host"]["parts_skipped"] = skipped
        notes.append(f"unhealthy host: {bad}; parts {skipped} (timings) skipped, part a (no timing) still runs")
        runnable = [p for p in runnable if p == "a"]
        rc = 1
    elif args.allow_power_cap and smi:
        cap = e4_pdl.health_problem(e4_pdl.smi_summary(smi, [], smi), check_power=True)
        if cap:
            doc["host"]["power_cap_note"] = cap
            notes.append(f"power cap accepted (--allow-power-cap): {cap}; timings are for the capped host")
    cut = {"part": None}
    try:
        with tempfile.TemporaryDirectory(prefix="cr-r1q-") as qdir:
            quant_info: dict = {}
            need_ptq1 = ("b" in runnable and "k4_ptq1" in args.configs) or ("a" in runnable and "ptq1_0" in args.variants)
            if need_ptq1:  # shared by part b (k4_ptq1 config) and part a (KL pass)
                dst = os.path.join(qdir, "draft-ptq1_0.gguf")
                rec = r1_agree.quantize_variant(bin_dir, paths["draft_bf16"], dst, "ptq1_0", common.Budget(min(5.0, total.left() / 60.0)),
                                                str(run_dir(EXP, args.gpu) / "raw" / "r1-quantize-ptq1_0.log"))
                quant_info["ptq1_0"] = rec["quantize"]
                config["ptq1_0_quantize"] = rec
                if rec.get("error"):
                    notes.append(f"ptq1_0: {rec['error']}")
                else:
                    paths["draft_ptq1_0"] = dst
            for i, p in enumerate(runnable):
                budget_s = part_budget_s(total.left(), p, runnable[i:])
                print(f"[r1] part {p}: {budget_s / 60:.1f} min of {total.left() / 60:.1f} min left", flush=True)
                if budget_s < 60:
                    notes.append(f"part {p} skipped: only {budget_s:.0f} s of budget left")
                    continue
                sub = common.Budget(budget_s / 60.0)
                cut["part"] = p
                try:
                    if p == "a":
                        run_agree_part(bin_dir=bin_dir, paths=paths, doc=doc, budget=sub, notes=notes, args=args, gpu=args.gpu,
                                       root=r1_agree.REPO_ROOT, quant_info=quant_info, save=save)
                    elif p == "b":
                        drafts = {"bf16": paths["draft_bf16"], "ptq1_0": paths.get("draft_ptq1_0")}
                        doc["e5"] = run_e5(args=args, bin_dir=bin_dir, target=paths["pq2"], drafts=drafts, budget_s=budget_s,
                                           notes=notes, gpu=args.gpu, on_update=lambda e: (doc.__setitem__("e5", e), save()))
                    else:
                        doc["step"] = run_step(bin_dir=bin_dir, target=paths["pq2"], draft=paths["draft_bf16"], budget=sub, notes=notes,
                                               gpu=args.gpu, prompts_file=args.prompts, cfg_name=args.step_config,
                                               on_stage=lambda r: (doc.__setitem__("step", dict(r, partial=True)), save()))
                        ref = (((doc.get("e5") or {}).get("summary") or {}).get(args.step_config) or {}).get("all") or {}
                        doc["step"]["wall_check"] = step_wall_check(doc["step"].get("analysis"), doc["step"].get("request"),
                                                                    ref.get("ms_per_step"))
                except Exception as e:  # a failed part must not lose the others
                    notes.append(f"part {p} crashed: {type(e).__name__}: {e}")
                cut["part"] = None
                save()
    finally:
        if cut["part"]:
            notes.append(f"run cut during part {cut['part']}: the results of that part are partial")
        notes.append("post-hoc PTQ1_0/PQ2_0 in this fork scale each 128 group by its absolute maximum and ignore the imatrix "
                     "(ggml-quants.c quantize_ptq1_0), so a collapse in agreement says the recipe is naive, not that ternary 0.8B is hopeless")
        notes.append("draft-simple samples its draft tokens on the CPU in this fork (common/speculative.cpp); --spec-draft-backend-sampling "
                     "only reaches the eagle3, dflash and mtp types. Backend sampling in R1 is the target's own -bs (k4_tbs against k4), "
                     "and no earlier E5 run had it (the flag defaults to off)")
        save(final=True)
    return rc


# ---------------------------------------------------------------- CLI

def _names(text):
    return [t.strip() for t in text.split(",") if t.strip()]


def parse_args(argv=None):
    ap = argparse.ArgumentParser(prog="exps.r1", description=__doc__.split("\n")[0])
    common.add_common_args(ap)
    ap.add_argument("--parts", type=_names, default=list("abc"), help="a (agreement), b (E5 matrix), c (nsys step split)")
    ap.add_argument("--limit", type=int, default=200, help="most prompts for part b, taken from the fixed-seed order")
    ap.add_argument("--seed", type=int, default=e5_spec.SEED)
    ap.add_argument("--n-predict", type=int, default=192)
    ap.add_argument("--prompts", default=str(make_prompts.DEFAULT_OUT))
    ap.add_argument("--code-floor", type=int, default=e5_spec.CODE_FLOOR,
                    help="code_fix / refactor prompts per category that base and the n-gram configs run whatever N is (0 = none)")
    ap.add_argument("--configs", type=_names, default=[c["name"] for c in e5_spec.matrix_configs() if c["default"]],
                    help="default leaves out k2 and k3 (follow-up pod)")
    ap.add_argument("--variants", type=_names, default=list(r1_agree.VARIANTS))
    ap.add_argument("--step-config", default="k4", help="draft-simple config of the nsys capture (k4 or k4_tbs)")
    ap.add_argument("--allow-power-cap", action="store_true", help="a power cap below default is a note, not a skip of parts b and c")
    a = ap.parse_args(argv)
    bad = [p for p in a.parts if p not in WEIGHTS]
    names = {c["name"] for c in e5_spec.matrix_configs()}
    badc = [c for c in a.configs if c not in names]
    if bad or badc:
        ap.error(f"unknown part(s) {bad} or config(s) {badc}; configs: {sorted(names)}")
    if a.configs and "base" not in a.configs and "b" in a.parts:
        ap.error("--configs must include base: every ratio is against target alone")
    steps = {c["name"] for c in e5_spec.matrix_configs() if c["spec_type"] == "draft-simple" and c["draft"] == "bf16"}
    if a.step_config not in steps:
        ap.error(f"--step-config must be one of {sorted(steps)}")
    a.budget_min = a.budget_min or (8.0 if a.quick else DEFAULT_TOTAL_MIN)
    if a.quick:
        a.limit, a.n_predict = min(a.limit, 8), min(a.n_predict, 32)
    a.code_floor = max(a.code_floor, 0)
    return a


def plan_json(a) -> dict:
    bin_dir = e4_bonsai.llama_bin()
    cfgs = [c for c in e5_spec.matrix_configs() if c["name"] in a.configs]
    step = next(c for c in e5_spec.matrix_configs() if c["name"] == a.step_config)
    text, meta = r1_agree.build_text()
    prompts = make_prompts.shuffled(make_prompts.load(a.prompts), a.seed, stratify=True)[:a.limit]
    parts = [p for p in PART_ORDER if p in a.parts]
    b_s = part_budget_s(a.budget_min * 60.0, "b", parts) if "b" in parts else 0.0
    return {"exp": EXP, "parts": parts, "budget_min": a.budget_min, "weights": WEIGHTS,
            "agree": {"variants": a.variants, "texts": {"e5_chat": "built after part b from the base config outputs", "repo": meta},
                      "base_cmd": r1_agree.perplexity_cmd(bin_dir, "<pq2.gguf>", "<text>", "<logprobs>", False),
                      "kl_cmd": r1_agree.perplexity_cmd(bin_dir, "<draft.gguf>", "<text>", "<logprobs>", True),
                      "quantize_cmd": r1_agree.quantize_cmd(bin_dir, "<draft-bf16.gguf>", "<out.gguf>", "PTQ1_0")},
            "e5": {"seed": a.seed, "n_prompts": len(prompts), "n_predict": a.n_predict, "code_floor": a.code_floor,
                   "plan": e5_spec.plan_n(cfgs, prompts, b_s, code_floor=a.code_floor) if b_s else None,
                   "configs": [{"name": c["name"], "desc": c["desc"], "prompt_s": c["prompt_s"], "frac": c["frac"], "categories": c["categories"],
                                "draft": c["draft"], "cmd": e5_spec.config_cmd(bin_dir, "<pq2.gguf>", f"<{c['draft']}.gguf>", c)} for c in cfgs]},
            "step": {"config": a.step_config, "server_cmd": e5_spec.config_cmd(bin_dir, "<pq2.gguf>", "<draft.gguf>", step),
                     "nsys_profile": r1_step.profile_cmd("nsys", "<tmp>/r1step", ["<server_cmd>"]),
                     "nsys_start": r1_step.control_cmd("nsys", "start"), "nsys_stop": r1_step.control_cmd("nsys", "stop"),
                     "env": "allowlisted (PATH, LD_LIBRARY_PATH, HOME, TMPDIR=<private dir>, LANG, CUDA_*, ...), secret-like names dropped",
                     "files": "private temp dir, deleted after parsing"},
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
