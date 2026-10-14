"""R2a: speculative decoding on long-context agentic prompts, one PRO 6000 pod session.

  python -m exps.fetch_models --models pq2,draft_bf16 --timeout-min 6 && python -m exps.r2a_longctx [--parts m,c,p] [--budget-min 29]

R1 used short prompts (about 100 tokens). R2a asks the same questions on prompts that make the model copy: 40 code-edit prompts that
embed one real source file of 2-8K tokens (20 ask for the modified file from a few lines before the edit to the end, 20 for a unified
diff) and 20 multi-turn tool-call traces in the model's native tool format (OpenAI tools and tool_calls, rendered by /apply-template;
3-5 earlier calls with real results, then the next call). The set is exps/data/prompts_r2a.jsonl, built offline by
exps.make_prompts --longctx. Every record keeps its category and approximate tokens (chars / 3.5); the server's own tokenizer count
(prompt_tokens, before the request) and prompt_n are stored per row, so the estimate can be checked, and a prompt that does not fit
the context is skipped with an error row.
The plan gives N about 43 (14-15 per category; MIN_N_CATEGORY is 15) at the 29 min budget, with the decode cost of K=4 taken from R1 (0.49-0.68x
on code). The cut after base can lower it; a category below 15 paired prompts is exploratory only (n_ge_min is false) and gives no
go / no-go. K=2 and K=3 are not in the default configs (a K=4 run costs about the same, and R1 left the n-gram question open);
run them on a follow-up pod with --configs.
m  matrix. One llama-server per config, all on this pod and with -lv 4: target alone (base), draft-simple K=4 (Qwen3.5-0.8B
   BF16), ngram-mod and ngram-map-k with the fork defaults and the R1 'short' parameters. Base runs first (every ratio is against
   it); the other configs follow in an order shuffled with the seed, and base runs again at the end on its first prompts
   (base_end: a drift bound for clocks and temperature, reported as matrix.drift). The prompts are shuffled with the seed too,
   category balanced, so a prefix is balanced. n_predict 320 for full-file edits, 512 for diffs and 128 for tool calls, ctx 16384.
   Per config, per category and overall: tok/s, speedup, r, alpha with bootstrap 95% intervals (prompts resampled), and greedy
   output equality against base. The output text is kept in the rows.
   Time: the prompt count N is fixed before any server starts from a cost model (plan_n). After base the model is rescaled with the
   measured time per prompt and N is cut again if the rest does not fit. A cut drops the tail of the balanced order (so every
   category loses about the same) and is logged in notes and in matrix.plan / matrix.replan with the dropped ids.
   Spec path: -lv 4 makes the INFO line `n_rs_seq` and the lines about partial sequence removal visible (llama-internal INFO is a
   trace-level message in this fork). They give the rollback path per config: RS (in-batch rollback, draft-simple) or FULL
   (KV checkpoint plus replay on partial acceptance, what the n-gram types get because need_n_rs_seq() is 0 for them). The per-step
   events (checkpoint created, restored, accepted a/b) are debug-level and appear only in part p.
c  overhead control. K=4 on a fixed subset of the stratified order (default 10 prompts, short n_predict), first unprofiled and then,
   from a fresh server, under nsys with the same requests. nsys runs with an allowlisted environment and a private 0700 work dir
   (r1_step.capture); the report and sqlite are deleted after parsing and never reach the results dir. Prefill bursts are cut
   away per request, so only decode steps are split. Result: unprofiled ms/step, nsys step wall, the profiler overhead on this
   host, and the step split with the overhead taken off.
p  rollback probe. A short -lv 5 run (one prompt per category, 128 tokens for edits, 64 for the tool call) of k4 and two n-gram
   configs, with LLAMA_TRACE=1; counts the checkpoint created / restored lines and the replay steps. On the checkpoint (FULL) path
   the restore fraction is restored / created, and the original a/b of a restored step comes from the trace line. The debug logs
   live in a private 0700 dir outside the results dir and are deleted. Untimed.
Parts run in the order m, c, p; each has a share of --budget-min (weights 21.5 : 5.5 : 2, unused time rolls forward) and the pod
deadline caps all. The pod failsafe is 55 min: bootstrap with the cold build ~14, downloads <= 6, r2a 29, end reserve 5.
An unhealthy host (clocks throttled under load, or a power cap without --allow-power-cap) skips m and c; p still runs.
Partial results go to raw/partial.json, at most every 15 s while prompts run and after every part.
Result: results/r2a/<gpu>/<stamp>.json with data.matrix, data.control, data.probe, data.host, data.prompt_set.
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import random
import re
import shutil
import statistics
import sys
import time
import urllib.error

from exps import common, e4_bonsai, e4_pdl, e5_spec, make_prompts, r1, r1_step, spec_math

EXP = "r2a"
SEED = make_prompts.R2A_SEED
CTX = 16384
LV = ["-lv", "4"]
PROBE_LV = ["-lv", "5"]
CONFIG_NAMES = ["base", "k2", "k3", "k4", "ngram_mod", "ngram_map_k", "ngram_mod_short", "ngram_map_k_short"]
DEFAULT_CONFIGS = ["base", "k4", "ngram_mod", "ngram_map_k", "ngram_mod_short", "ngram_map_k_short"]     # k2 and k3: a follow-up pod
DRIFT_NAME, DRIFT_N, DRIFT_RESERVE_S = "base_end", 5, 45.0   # base again on its first prompts, last; its time is kept out of the plan
DRIFT_FLAG = 0.03           # a config whose median SM clock differs from base by more than this is flagged
PROBE_CONFIGS = ["k4", "ngram_mod_short", "ngram_map_k_short"]
WEIGHTS = {"m": 21.5, "c": 5.5, "p": 2.0}
PART_ORDER = "mcp"
DEFAULT_TOTAL_MIN = 29.0
END_RESERVE_S = 25.0
NEEDS = {"m": ("llama-server",), "c": ("llama-server",), "p": ("llama-server",)}
CONTROL_N = 10
NSYS_N_PREDICT = 48
REQUEST_GAP_S = 0.4         # idle before each request of part c, so nsys sees one gap per request
PROBE_N_PREDICT = {"edit_full_file": 128, "edit_diff": 128, "toolcall_trace": 64}
PROBE_CATEGORIES = ["edit_full_file", "edit_diff", "toolcall_trace"]
PROBE_ENV = {"LLAMA_TRACE": "1"}      # server-context.cpp prints an INFO line with the original accepted a/b of every step
CTX_MARGIN = 64             # tokens kept free above prompt + n_predict
LONG_VERIFY_FACTOR = 2.5    # a decode step whose target burst is this much longer than the median verify is not a decode step
# cost model of one prompt, replaced by the measured base time after the first config
STARTUP_S = 15.0            # model load and warm-up of a server (page cache warm)
FIRST_STARTUP_EXTRA_S = 20.0
PREFILL_TOK_S = 2500.0
DECODE_TOK_S = 140.0        # R1: target alone 141.7 tok/s
REQUEST_S = 0.35            # two HTTP calls, template, sampler setup
OUT_TOKENS = {"edit_full_file": 320, "edit_diff": 220, "toolcall_trace": 60}   # diffs and tool calls stop early
DECODE_FACTOR = {"none": 1.0, "draft-simple": 1.7, "ngram-mod": 1.1, "ngram-map-k": 1.1}   # R1: K=4 ran at 0.49-0.68x on code
PLAN_MARGIN = 0.9
N_FLOOR = 6
SCALE_RANGE = (0.4, 3.0)
BOOT = 1000
SEG_GAP_MS = 150.0          # idle between two requests in the nsys capture
PREFILL_MS = 40.0           # a target burst this long is prompt processing (a decode verify pass is ~10 ms)
MIN_N_CATEGORY = e5_spec.MIN_N_CATEGORY
NGRAM_STATE_NOTE = ("ngram-mod keeps its token pool across requests on one server (cleared at 25% occupancy), so a prompt sees the "
                    "prompts and outputs of earlier prompts of the same server; the order is the seeded shuffle")
CHECKPOINT_NOTE = ("the n-gram types get n_rs_seq = 0 from need_n_rs_seq() (common.h), so the hybrid target takes the KV-checkpoint "
                   "path: a checkpoint per drafting step and a restore plus replay of the target pass on partial acceptance, which "
                   "draft-simple (n_rs_seq = K) does not pay")


# ---------------------------------------------------------------- configs and commands

def matrix_configs(names=CONFIG_NAMES) -> list[dict]:
    """The R1 configs by name, all on every prompt and with -lv 4 added."""
    by = {c["name"]: c for c in e5_spec.matrix_configs()}
    out = []
    for n in names:
        c = by[n]
        out.append(dict(c, extra=list(c["extra"]) + LV, categories=None, frac=1.0, extra_code=False))
    return out


def config_order(cfgs: list[dict], seed: int = SEED) -> list[dict]:
    """Base first, the rest shuffled with the seed (base must exist for every ratio, so it is not shuffled)."""
    base = [c for c in cfgs if c["name"] == "base"]
    rest = [c for c in cfgs if c["name"] != "base"]
    random.Random(seed).shuffle(rest)
    return base + rest


def with_lv(cfg: dict, lv: list[str]) -> dict:
    extra = list(cfg["extra"])
    if "-lv" in extra:
        i = extra.index("-lv")
        extra[i:i + 2] = lv
    else:
        extra += lv
    return dict(cfg, extra=extra)


def config_cmd(bin_dir, target: str, draft: str | None, cfg: dict, port: int | None = None, ctx: int = CTX) -> list[str]:
    port = port or e5_spec.PORT
    t = cfg["spec_type"]
    if t == "draft-simple":
        return e5_spec.server_cmd(bin_dir, target, cfg["k"], draft, port, ctx, extra=cfg["extra"])
    cmd = e5_spec.server_cmd(bin_dir, target, 0, None, port, ctx)
    return cmd + (["--spec-type", t] if t != "none" else []) + cfg["extra"]


# ---------------------------------------------------------------- plan

def prompt_cost_s(p: dict, cfg: dict, scale: float = 1.0) -> float:
    pre = p["approx_tokens"] / PREFILL_TOK_S * (1.1 if cfg["spec_type"] == "draft-simple" else 1.0)
    dec = min(OUT_TOKENS.get(p["category"], p["n_predict"]), p["n_predict"]) / DECODE_TOK_S * DECODE_FACTOR.get(cfg["spec_type"], 1.0)
    return scale * (pre + dec) + REQUEST_S


def config_cost_s(cfg: dict, prompts: list[dict], scale: float = 1.0, first: bool = False) -> float:
    if not prompts:
        return 0.0
    return STARTUP_S + (FIRST_STARTUP_EXTRA_S if first else 0.0) + sum(prompt_cost_s(p, cfg, scale) for p in prompts)


def _count(prompts: list[dict]) -> dict:
    out: dict = {}
    for p in prompts:
        out[p["category"]] = out.get(p["category"], 0) + 1
    return out


def plan_n(cfgs: list[dict], prompts: list[dict], budget_s: float, scale: float = 1.0, margin: float = PLAN_MARGIN,
           floor: int = N_FLOOR, first: bool = True) -> dict:
    """Largest prefix of the (category balanced) prompt order whose summed expected cost over all configs fits margin * budget_s.
    first: the first config is a cold start. The tail that does not fit is listed per category, so a cut is visible."""
    best = None
    for n in range(len(prompts), 0, -1):
        if sum(config_cost_s(c, prompts[:n], scale, first and i == 0) for i, c in enumerate(cfgs)) <= margin * budget_s:
            best = n
            break
    n = best if best is not None else min(floor, len(prompts))
    main, dropped = prompts[:n], prompts[n:]
    return {"n": n, "fits": best is not None, "scale": scale, "budget_s": budget_s, "margin": margin,
            "est_s": sum(config_cost_s(c, main, scale, first and i == 0) for i, c in enumerate(cfgs)),
            "per_config_est_s": {c["name"]: config_cost_s(c, main, scale, first and i == 0) for i, c in enumerate(cfgs)},
            "per_category": _count(main), "n_total": len(prompts), "dropped": [p["id"] for p in dropped],
            "dropped_per_category": _count(dropped)}


def measured_scale(cfg: dict, res: dict, prompts: list[dict]) -> float | None:
    """Real seconds per prompt of a finished config over the model's seconds, for the prompts that ran."""
    ran = {r["id"] for r in res.get("rows", [])}
    ps = [p for p in prompts if p["id"] in ran]
    exp = sum(prompt_cost_s(p, cfg) - REQUEST_S for p in ps)
    wall = res.get("wall_s")
    if not ps or not wall or exp <= 0:
        return None
    work = wall - (res.get("startup_s") or 0.0) - REQUEST_S * len(ps)
    return max(SCALE_RANGE[0], min(SCALE_RANGE[1], work / exp))


def part_budget_s(left_s: float, part: str, remaining: list[str]) -> float:
    tot = sum(WEIGHTS[p] for p in remaining) or 1.0
    return max(left_s - END_RESERVE_S, 0.0) * WEIGHTS[part] / tot


# ---------------------------------------------------------------- requests

def template_body(messages: list[dict], thinking: bool | None, tools: list[dict] | None = None) -> dict:
    body: dict = {"messages": messages}
    if tools:
        body["tools"] = tools
    if thinking is False:
        body["chat_template_kwargs"] = {"enable_thinking": False}
    return body


def apply_template(base: str, messages: list[dict], state: dict, tools: list[dict] | None = None) -> str:
    """Chat-templated prompt text. The first call finds out whether enable_thinking=false is accepted; later calls use that."""
    if state.get("thinking_disabled") is not False:
        try:
            text = e5_spec.post_json(base + "/apply-template", template_body(messages, False, tools))["prompt"]
            if state.get("thinking_disabled") is None:
                plain = e5_spec.post_json(base + "/apply-template", template_body(messages, None, tools))["prompt"]
                state["template_supports_enable_thinking"] = text != plain
            state["thinking_disabled"] = True
            return text
        except urllib.error.HTTPError:
            state["thinking_disabled"] = False
    return e5_spec.post_json(base + "/apply-template", template_body(messages, None, tools))["prompt"]


def count_tokens(base: str, text: str) -> int | None:
    """Prompt length in tokens by the server's own tokenizer (the request tokenizes the same way); None when the call fails."""
    try:
        return len(e5_spec.post_json(base + "/tokenize", {"content": text, "add_special": True, "parse_special": True}, 60.0)["tokens"])
    except Exception:
        return None


def run_requests(prompts: list[dict], budget: common.Budget, notes: list, cfg: dict, on_row=None, keep_text: bool = True,
                 gap_s: float = 0.0, reserve_s: float = 10.0, ctx: int = CTX) -> tuple[list[dict], dict]:
    """Run the prompts against the live server in order, each with its own n_predict. Stops when the budget is used up. A prompt
    whose token count plus n_predict plus CTX_MARGIN exceeds ctx is not sent (error row)."""
    base = f"http://127.0.0.1:{e5_spec.PORT}"
    state: dict = {"thinking_disabled": None, "template_supports_enable_thinking": None, "draft_not_used": False, "cut_after": None}
    rows: list[dict] = []
    for i, p in enumerate(prompts):
        if budget.expired(reserve_s):
            state["cut_after"] = i
            notes.append(f"{cfg['name']}: time budget used up after {i} of {len(prompts)} prompts")
            break
        row = {"id": p["id"], "category": p["category"], "approx_tokens": p["approx_tokens"], "n_predict": p["n_predict"]}
        try:
            if gap_s:
                time.sleep(gap_s)
            text = apply_template(base, p["messages"], state, p.get("tools"))
            row["prompt_tokens"] = count_tokens(base, text)
            if row["prompt_tokens"] and row["prompt_tokens"] + p["n_predict"] + CTX_MARGIN > ctx:
                raise ValueError(f"prompt {row['prompt_tokens']} + n_predict {p['n_predict']} + {CTX_MARGIN} does not fit ctx {ctx}")
            resp = e5_spec.post_json(base + "/completion", e5_spec.completion_body(text, p["n_predict"]), budget.timeout(300.0, reserve_s=10))
            row.update(e5_spec.parse_completion(resp, keep_text))
        except Exception as e:
            row["error"] = f"{type(e).__name__}: {e}"
        rows.append(row)
        if on_row:
            try:
                on_row(rows)
            except Exception as e:
                if not state.get("on_row_error"):
                    notes.append(f"partial save failed: {type(e).__name__}: {e}")
                state["on_row_error"] = True
        ok = [r for r in rows if "error" not in r]
        if cfg["spec_type"] == "draft-simple" and len(ok) >= 3 and sum(r.get("draft_n", 0) for r in ok[:3]) == 0:
            state["draft_not_used"] = True
            break
    return rows, state


# ---------------------------------------------------------------- log checks

def log_checks(text: str, spec_type: str) -> dict:
    """Spec path of a server log. n_rs_seq (INFO line of llama-context.cpp, trace level here) and the common_context_can_seq_rm
    lines give the rollback path: RS = in-batch rollback bounded by n_rs_seq, FULL = KV checkpoints and replay. Both are printed at
    -lv 4 and in context order (target first, draft second). seq_rm_types and rollback_path must agree (path_consistent)."""
    lines = text.splitlines()
    out = dict(e5_spec.log_flags(text), **e5_spec.rollback_flags(text, spec_type))
    rm = []
    for ln in lines:
        if "supports bounded partial sequence removal" in ln:
            rm.append("RS")
        elif "does not support partial sequence removal" in ln:
            rm.append("FULL")
    out["seq_rm_types"] = rm[:2]
    out["spec_uses_checkpoints"] = any("speculative decoding will use checkpoints" in ln for ln in lines)
    out["spec_unsupported"] = any("speculative decoding not supported by this context" in ln for ln in lines)
    out["prompt_checkpoints_logged"] = sum(1 for ln in lines if "created context checkpoint" in ln)
    out["impl_lines"] = [ln.strip()[:200] for ln in lines if "adding speculative implementation" in ln][:3]
    path = out["rollback_path"]
    if spec_type == "none":
        out["path_consistent"] = None
    else:
        want = {"in_batch": "RS", "checkpoint": "FULL"}.get(path)
        out["path_consistent"] = None if want is None or not rm else (rm[0] == want and (out["spec_uses_checkpoints"] == (want == "FULL")))
    return out


CREATED_RE = re.compile(r"created speculative checkpoint")
RESTORED_RE = re.compile(r"restoring speculative checkpoint")
ACCEPT_DBG_RE = re.compile(r"accepted\s+(\d+)/\s*(\d+) draft tokens, new n_tokens")           # SLT_DBG, printed after a step that did not restore
ACCEPT_TRACE_RE = re.compile(r"accepted\s+(\d+)/\s*(\d+) draft tokens( \(restore checkpoint\))?\s*$")   # SLT_INF, only with LLAMA_TRACE > 0 (%2zu padded)


def count_spec_events(text: str) -> dict:
    """Per-step events of a debug-level (-lv 5) log of the server. What the server prints:
    - a step with a draft makes a checkpoint when the target needs one (FULL path: every such step; RS path: only a draft longer
      than n_rs_seq). 'created speculative checkpoint' counts these steps.
    - partial acceptance on the FULL path logs 'restoring speculative checkpoint' and returns: there is no 'accepted a/b' DEBUG line
      for that step. The next step replays the accepted tokens as a draft of size a + 1 and logs 'accepted a/(a + 1)'; that line is
      a replay step, not a draft step, so it is counted apart (replay_steps) and never classified.
    - with LLAMA_TRACE > 0 an INFO line has the original a/b of every step; a restored one ends in '(restore checkpoint)'.
    FULL path (created > 0): draft steps = created, full accepts = created - restored, restore_per_step = restored / created.
    Otherwise (in-batch rollback) the steps come from the 'accepted a/b' DEBUG lines of the non replay steps. The last step of a request
    has no such line (the request ends first), so that count is short by up to one step per request."""
    created = restored = replays = 0
    plain: list[tuple[int, int]] = []
    restored_ab: list[tuple[int, int]] = []
    pending = False
    for ln in text.splitlines():
        if CREATED_RE.search(ln):
            created += 1
        elif RESTORED_RE.search(ln):
            restored += 1
            pending = True
        elif (m := ACCEPT_DBG_RE.search(ln)):
            if pending:
                replays += 1
                pending = False
            else:
                plain.append((int(m.group(1)), int(m.group(2))))
        elif (m := ACCEPT_TRACE_RE.search(ln)) and m.group(3):
            restored_ab.append((int(m.group(1)), int(m.group(2))))
    out: dict = {"created": created, "restored": restored, "replay_steps": replays, "restored_ab_logged": len(restored_ab)}
    if created:
        out.update(path="checkpoint", steps_with_draft=created, full_accept=max(created - restored, 0), restored_steps=restored,
                   partial_accept=sum(1 for a, _ in restored_ab if a > 0) if restored_ab else None,
                   zero_accept=sum(1 for a, _ in restored_ab if a == 0) if restored_ab else None)
        if restored_ab:
            out["restored_mean_accepted"] = sum(a for a, _ in restored_ab) / len(restored_ab)
    else:
        drafted = [(a, b) for a, b in plain if b > 0]
        out.update(path="in_batch", steps_with_draft=len(drafted), full_accept=sum(1 for a, b in drafted if a == b),
                   partial_accept=sum(1 for a, b in drafted if 0 < a < b), zero_accept=sum(1 for a, b in drafted if a == 0))
    d = out["steps_with_draft"]
    out["restore_per_step"] = restored / d if d else None
    out["checkpoint_per_step"] = created / d if d else None
    return out


# ---------------------------------------------------------------- one server

def run_config(cfg: dict, *, bin_dir, target: str, draft: str | None, prompts: list[dict], budget_s: float, notes: list, log_path: str,
               ctx: int = CTX, on_row=None, keep_text: bool = True, gap_s: float = 0.0, env: dict | None = None) -> dict:
    """Start one server, run the prompts in order within budget_s (startup included), stop it. Never raises. env: extra variables."""
    cmd = config_cmd(bin_dir, target, draft, cfg, ctx=ctx)
    res = {"name": cfg["name"], "spec_type": cfg["spec_type"], "k": cfg["k"], "cmd": cmd, "log": log_path, "rows": [], "error": None,
           "budget_s": budget_s, "planned": len(prompts), "desc": cfg.get("desc"), "draft": cfg.get("draft")}
    budget = common.Budget(budget_s / 60.0)
    srv = e5_spec.Server(cmd, log_path, env)
    t0 = time.monotonic()
    try:
        with e4_pdl.SmiWatch() as watch:
            ready, err = srv.wait_ready(budget.timeout(600.0, reserve_s=10))
            res["startup_s"] = time.monotonic() - t0
            if not ready:
                res["error"] = err
                res["log_tail"] = srv.tail()
            else:
                e5_spec.warmup()
                run_budget = common.Budget(max(budget_s - (time.monotonic() - t0), 0.3 * budget_s) / 60.0)
                rows, info = run_requests(prompts, run_budget, notes, cfg, on_row, keep_text, gap_s, ctx=ctx)
                res["rows"], res["chat"] = rows, info
        res["smi"] = watch.summary()
    except Exception as e:  # keep the other configs going
        res["error"] = f"{type(e).__name__}: {e}"
    finally:
        log = srv.head_tail()
        res["spec_log"] = e5_spec.spec_log_lines(log, 8)
        res["checks"] = log_checks(log, cfg["spec_type"])
        res["log_lines"] = log.count("\n")
        srv.stop()
    res["wall_s"] = time.monotonic() - t0
    if cfg["spec_type"].startswith("ngram"):
        res["state_note"] = NGRAM_STATE_NOTE
    return res


def config_notes(name: str, res: dict) -> list[str]:
    out = []
    ck = res.get("checks") or {}
    n_err = sum(1 for r in res.get("rows") or [] if "error" in r)
    if n_err:
        out.append(f"r2a {name}: {n_err} prompts failed, first: {next(r['error'] for r in res['rows'] if 'error' in r)}")
    if res.get("error"):
        out.append(f"r2a {name}: {res['error']}")
    elif (res.get("chat") or {}).get("draft_not_used"):
        out.append(f"r2a {name}: draft-simple drafted 0 tokens in the first prompts (spec log: {res.get('spec_log')})")
    if ck.get("rollback_path") == "unknown" and res.get("spec_type") != "none":
        out.append(f"r2a {name}: no n_rs_seq line in the server log (-lv 4 expected to show it)")
    if ck.get("path_consistent") is False:
        out.append(f"r2a {name}: n_rs_seq says {ck.get('rollback_path')} but the seq_rm lines say {ck.get('seq_rm_types')}")
    if any(r.get("truncated") for r in res.get("rows") or []):
        out.append(f"r2a {name}: a prompt was truncated by the context size")
    if name.startswith("ngram") and ck.get("checkpoint_replay") is not False:
        out.append(f"r2a {name}: rollback_path={ck.get('rollback_path')} (n_rs_seq={ck.get('n_rs_seq')}); {CHECKPOINT_NOTE}")
    return out


# ---------------------------------------------------------------- summary

def _pct(vals: list[float], level: float = 0.95):
    if len(vals) < 2:
        return None
    vals = sorted(vals)
    lo, hi = (1.0 - level) / 2.0, 1.0 - (1.0 - level) / 2.0
    return [vals[int(lo * (len(vals) - 1))], vals[int(round(hi * (len(vals) - 1)))]]


def boot_intervals(c_rows: list[dict], b_rows: list[dict], spec_type: str, base_type: str = "none", n_boot: int = BOOT,
                   seed: int = SEED) -> dict:
    """Bootstrap 95% intervals (paired prompts resampled) of tok/s of the config and of base, r and alpha. None for undefined ones."""
    n = len(c_rows)
    if n < 2:
        return {}
    rng = random.Random(seed)
    vals: dict[str, list[float]] = {"tok_s": [], "base_tok_s": [], "r": [], "alpha": []}
    for _ in range(n_boot):
        idx = [rng.randrange(n) for _ in range(n)]
        c, b = e5_spec._agg([c_rows[i] for i in idx], spec_type), e5_spec._agg([b_rows[i] for i in idx], base_type)
        for key, v in (("tok_s", c["tok_s"]), ("base_tok_s", b["tok_s"]), ("r", c["r"]), ("alpha", c["alpha"])):
            if v is not None:
                vals[key].append(v)
    return {f"{k}_ci95": _pct(v) for k, v in vals.items()}


def summarize(runs: dict, base: str = "base", ci: bool = False) -> dict:
    """Per config, per category and overall, on the prompts both the config and base finished. ci adds the bootstrap intervals
    (speedup, tok/s, r, alpha); the point values and the equality counts come from e5_spec._pair_stats."""
    out = e5_spec.summarize_matrix(runs, base, ci)
    base_rows = {r["id"]: r for r in (runs.get(base) or {}).get("rows", []) if e5_spec._ok(r)}
    for name, run in runs.items():
        ent = out[name]
        rows = {r["id"]: r for r in run.get("rows", []) if e5_spec._ok(r)}
        ids = [i for i in rows if i in base_rows]
        if not ci or not ids:
            continue
        st = run.get("spec_type") or "none"
        ent["all"].update(boot_intervals([rows[i] for i in ids], [base_rows[i] for i in ids], st))
        for cat, sub in ent["by_category"].items():
            sel = [i for i in ids if rows[i]["category"] == cat]
            sub.update(boot_intervals([rows[i] for i in sel], [base_rows[i] for i in sel], st))
        ent["n_by_category"] = _count([rows[i] for i in ids])
        ent["min_n_category"] = min(ent["n_by_category"].values())
        ent["n_ge_min"] = ent["min_n_category"] >= MIN_N_CATEGORY
    return out


def observed_prompt_set(prompts: list[dict], rows: list[dict]) -> dict:
    """Server prompt_n against the chars / 3.5 estimate, by category, from the rows of one config."""
    by_id = {p["id"]: p for p in prompts}
    out: dict = {}
    for r in rows:
        p = by_id.get(r["id"])
        if p and r.get("prompt_n"):
            d = out.setdefault(r["category"], {"prompt_n": [], "ratio": []})
            d["prompt_n"].append(r["prompt_n"])
            d["ratio"].append(r["prompt_n"] / p["approx_tokens"])
    return {c: {"n": len(d["prompt_n"]), "prompt_n_min": min(d["prompt_n"]), "prompt_n_median": int(statistics.median(d["prompt_n"])),
                "prompt_n_max": max(d["prompt_n"]), "prompt_n_over_estimate_median": statistics.median(d["ratio"])} for c, d in out.items()}


# ---------------------------------------------------------------- part m

def drift_summary(runs: dict) -> dict | None:
    """base_end against base on the prompts both ran: the ratio of the same config at the start and at the end of the matrix."""
    pair = e5_spec.compare_pair(runs, DRIFT_NAME, "base", ci=False)
    if not pair:
        return None
    return {"n": pair["n"], "ratio": pair.get("speedup"), "tok_s_end": pair.get("tok_s"), "tok_s_start": pair.get("base_tok_s"),
            "text_equal_frac": pair.get("text_equal_frac"),
            "note": "base again after all configs, on its first prompts; a ratio far from 1.0 is time or thermal drift in every speedup"}


def clock_flags(runs: dict, tol: float = DRIFT_FLAG) -> dict:
    """Median SM clock of every config against base; flagged when it differs by more than tol (the speedups then mix in a clock change)."""
    ref = ((runs.get("base") or {}).get("smi") or {}).get("sm_mhz_med")
    out: dict = {}
    for name, run in runs.items():
        sm = (run.get("smi") or {}).get("sm_mhz_med")
        if ref and sm:
            out[name] = {"sm_mhz_med": sm, "vs_base": sm / ref, "flag": abs(sm / ref - 1.0) > tol}
    return out


def run_matrix(*, args, bin_dir, target: str, draft: str, order: list[dict], budget_s: float, notes: list, gpu: str, on_update) -> dict:
    from infra.results import artifact_path
    cfgs = config_order(matrix_configs(args.configs), args.seed)
    plan = plan_n(cfgs, order, max(budget_s - DRIFT_RESERVE_S, 0.0))
    main = order[:plan["n"]]
    runs: dict = {}
    out: dict = {"seed": args.seed, "ctx": args.ctx, "n_predict": make_prompts.R2A_N_PREDICT, "order": [p["id"] for p in order],
                 "config_order": [c["name"] for c in cfgs], "drift_config": DRIFT_NAME, "drift": None, "clocks": {}, "plan": plan,
                 "replan": None, "runs": runs, "summary": {},
                 "state_note": NGRAM_STATE_NOTE, "checkpoint_note": CHECKPOINT_NOTE}
    if plan["dropped"]:
        notes.append(f"m: budget {budget_s:.0f} s fits {plan['n']} of {plan['n_total']} prompts at the start; dropped (evenly, tail of the "
                     f"balanced order) per category {plan['dropped_per_category']}: {plan['dropped']}")
    if not plan["fits"]:
        notes.append(f"m: the matrix does not fit even at {plan['n']} prompts (expected {plan['est_s']:.0f} s of {budget_s:.0f} s)")
    t_end = time.monotonic() + budget_s
    last_save = [0.0]

    def refresh(final: bool = False) -> None:
        out["summary"] = summarize(runs, ci=final)
        out["prompt_tokens_observed"] = observed_prompt_set(order, (runs.get("base") or {}).get("rows", []))
        on_update(out)

    for i, cfg in enumerate(cfgs):
        left = t_end - time.monotonic() - DRIFT_RESERVE_S
        if left < 45:
            notes.append(f"m: time budget used up; {[c['name'] for c in cfgs[i:]]} skipped")
            break
        base_ids = {r["id"] for r in (runs.get("base") or {}).get("rows", []) if "error" not in r}
        ps = main if cfg["name"] == "base" else ([p for p in main if p["id"] in base_ids] if base_ids else main)
        if not ps:
            notes.append(f"m: {cfg['name']} has no prompts")
            continue
        costs = [config_cost_s(cfg, ps)] + [config_cost_s(c, main) for c in cfgs[i + 1:]]
        share = e5_spec.time_shares(costs, left)[0]
        runs[cfg["name"]] = {"name": cfg["name"], "spec_type": cfg["spec_type"], "k": cfg["k"], "rows": [], "planned": len(ps), "partial": True}

        def on_row(rows, name=cfg["name"]) -> None:
            runs[name]["rows"] = list(rows)
            if time.monotonic() - last_save[0] > 15:
                last_save[0] = time.monotonic()
                refresh()
        res = run_config(cfg, bin_dir=bin_dir, target=target, draft=draft, prompts=ps, budget_s=share, notes=notes, ctx=args.ctx,
                         log_path=str(artifact_path(EXP, f"server-{cfg['name']}.log", gpu)), on_row=on_row)
        runs[cfg["name"]] = res
        notes.extend(config_notes(cfg["name"], res))
        if cfg["name"] == "base":
            sc = measured_scale(cfg, res, ps)
            out["measured_scale"] = sc
            rest = cfgs[i + 1:]
            if sc is not None and rest:
                base_ok = [p for p in main if p["id"] in {r["id"] for r in res["rows"] if "error" not in r}]
                new = plan_n(rest, base_ok, t_end - time.monotonic() - DRIFT_RESERVE_S, scale=sc, first=False, floor=min(N_FLOOR, len(base_ok)))
                out["replan"] = new
                if new["n"] < len(base_ok):
                    main = base_ok[:new["n"]]
                    notes.append(f"m: after base (time per prompt x{sc:.2f} of the model) the other configs are cut from {len(base_ok)} to "
                                 f"{new['n']} prompts; dropped per category {new['dropped_per_category']}: {new['dropped']}")
        refresh()
    drift_ids = [r["id"] for r in (runs.get("base") or {}).get("rows", []) if "error" not in r][:DRIFT_N]
    if drift_ids and len(runs) > 1 and t_end - time.monotonic() > 30:
        ps = [p for p in order if p["id"] in set(drift_ids)]
        res = run_config(dict(matrix_configs(["base"])[0], name=DRIFT_NAME), bin_dir=bin_dir, target=target, draft=draft, prompts=ps, ctx=args.ctx,
                         budget_s=max(t_end - time.monotonic(), 30.0), notes=notes, log_path=str(artifact_path(EXP, f"server-{DRIFT_NAME}.log", gpu)))
        runs[DRIFT_NAME] = res
        notes.extend(config_notes(DRIFT_NAME, res))
        out["drift"] = drift_summary(runs)
        if out["drift"] and out["drift"]["ratio"] and abs(out["drift"]["ratio"] - 1.0) > DRIFT_FLAG:
            notes.append(f"m: base at the end ran at {out['drift']['ratio']:.3f}x of base at the start: drift of more than {DRIFT_FLAG:.0%} in the speedups")
    elif len(runs) > 1:
        notes.append("m: no time left for the drift control (base again at the end)")
    out["clocks"] = clock_flags(runs)
    for name, c in out["clocks"].items():
        if c["flag"]:
            notes.append(f"m: {name} median SM clock {c['sm_mhz_med']:.0f} MHz is {c['vs_base']:.3f}x of base: its speedup includes a clock change")
    refresh(final=True)
    return out


# ---------------------------------------------------------------- part c

def split_segments(acts: list[dict], gap_ms: float = SEG_GAP_MS) -> list[list[dict]]:
    """Activity cut at idle gaps longer than gap_ms: one segment per request of the capture."""
    segs: list[list[dict]] = []
    cur: list[dict] = []
    end = None
    for a in sorted(acts, key=lambda x: x["t0"]):
        if cur and a["t0"] - end > gap_ms * 1e6:
            segs.append(cur)
            cur, end = [], None
        cur.append(a)
        end = a["t1"] if end is None else max(end, a["t1"])
    return segs + ([cur] if cur else [])


def decode_window(seg: list[dict], roles: dict, prefill_ms: float = PREFILL_MS) -> tuple[list[dict], dict]:
    """The part of one request after its prompt processing: starts where the last target burst of prefill_ms or more ends."""
    tgt = [x for x in seg if roles.get(x["stream"]) == "target"] if roles else seg
    long = [b for b in r1_step.make_bursts(tgt, r1_step.GAP_US * 1e3) if b["t1"] - b["t0"] >= prefill_ms * 1e6]
    start = long[-1]["t1"] if long else seg[0]["t0"]
    return [x for x in seg if x["t0"] >= start], {"n_prefill_bursts": len(long), "prefill_ms": (start - seg[0]["t0"]) / 1e6}


def analyze_segments(acts: list[dict], k: int, gap_ms: float = SEG_GAP_MS, prefill_ms: float = PREFILL_MS, skip: int = 2,
                     long_factor: float = LONG_VERIFY_FACTOR) -> dict:
    """r1_step.analyze over the decode part of every request of one capture; the step lists are pooled. The stream roles come from the
    whole capture (a window alone can mistake a long draft prompt pass for the target). A request whose window has too few steps is
    listed in segments with its error and left out. skip steps per request are dropped (the first ones can hold the tail of the prompt
    batch or the first draft prompt pass); of the pooled steps those with a target burst longer than long_factor x the median verify
    are dropped too (a prefill batch that came after the last long burst of the cut), and counted in n_steps_dropped_long."""
    out: dict = {"ok": False, "error": None, "k": k, "n_acts": len(acts), "mode": "segments", "gap_ms": gap_ms, "prefill_ms": prefill_ms}
    if not acts:
        out["error"] = "no GPU activity in the capture"
        return out
    acts = sorted(acts, key=lambda x: x["t0"])
    roles = r1_step.assign_roles(acts)
    segs = split_segments(acts, gap_ms)
    steps: list[dict] = []
    per_seg: list[dict] = []
    for seg in segs:
        win, info = decode_window(seg, roles, prefill_ms)
        a = r1_step.analyze(win, k, skip=skip, max_steps=10 ** 6, roles=roles) if win else {"ok": False, "error": "empty window"}
        per_seg.append(dict(info, n_acts=len(seg), n_steps=a.get("n_steps", 0) if a.get("ok") else 0, error=a.get("error")))
        if a.get("ok"):
            steps += a["steps_ms"]
    out.update(n_segments=len(segs), n_segments_used=sum(1 for s in per_seg if s["n_steps"]), segments=per_seg[:40],
               split_mode="stream" if roles else "duration", streams=r1_step.stream_summary(acts))
    if not steps:
        out["error"] = f"no decode steps in {len(segs)} segments"
        return out
    keys = r1_step.MS_KEYS
    med_verify = statistics.median(s["target_verify"] for s in steps)
    kept = [s for s in steps if s["target_verify"] <= long_factor * med_verify]
    out["n_steps_dropped_long"] = len(steps) - len(kept)
    steps = kept
    out.update(ok=True, n_steps=len(steps), median_ms={key: statistics.median(s[key] for s in steps) for key in keys},
               mean_ms={key: statistics.fmean(s[key] for s in steps) for key in keys}, steps_ms=steps[:400])
    out["checks"] = {"split_mode": out["split_mode"], "c1_floor_le_2ms": out["median_ms"]["c1_floor"] <= 2.0,
                     "prefill_cut_in_every_segment": all(s["n_prefill_bursts"] > 0 for s in per_seg if s["n_steps"]),
                     "note": f"steps are decode only: prompt processing is cut from every request; skip={skip} steps per request"}
    return out


def step_ms(rows: list[dict]) -> float | None:
    """Mean ms of one decode step from the server timings of rows (requests of one run)."""
    ok = [r for r in rows if e5_spec._ok(r)]
    if not ok:
        return None
    return spec_math.observed_step_ms(sum(r["predicted_ms"] for r in ok), sum(r["tokens_predicted"] for r in ok),
                                      sum(r.get("draft_n_accepted", 0) for r in ok), n_first=len(ok))


def overhead(unprof_rows: list[dict], nsys_rows: list[dict], analysis: dict | None) -> dict:
    """Profiler overhead on this host. unprof: mean ms per step of the unprofiled run (server timings); nsys: MEAN step wall of the
    capture (a mean against a mean: the per-request first steps and early stops of the draft shift a median away from it). The
    medians are kept as side values. The step split with the overhead taken off: kernel-busy parts of the capture are kept (the
    profiler does not change kernel time much), the rest of the unprofiled step is non-compute time."""
    un, pr = step_ms(unprof_rows), step_ms(nsys_rows)
    out: dict = {"unprofiled_ms_per_step": un, "profiled_request_ms_per_step": pr}
    mean = (analysis or {}).get("mean_ms") or {}
    med = (analysis or {}).get("median_ms") or {}
    wall = mean.get("wall")
    out["nsys_step_wall_ms"] = wall
    out["nsys_step_wall_median_ms"] = med.get("wall")
    if un and wall:
        out["overhead_ms"] = wall - un
        out["overhead_frac"] = wall / un - 1.0
        out["nsys_wall_over_unprofiled"] = wall / un
    if un and pr:
        out["profiled_request_over_unprofiled"] = pr / un
    if un and mean:
        busy = mean["target_verify"] + mean["draft_gen"] + mean["draft_reeval"]
        out["split_ms"] = {"target_verify": mean["target_verify"], "draft_gen": mean["draft_gen"], "draft_reeval": mean["draft_reeval"],
                           "kernel_busy_sum": busy}
        if med:
            out["split_median_ms"] = {k: med[k] for k in ("target_verify", "draft_gen", "draft_reeval")}
        out["unprofiled_non_compute_ms"] = un - busy
        out["unprofiled_non_compute_frac"] = (un - busy) / un
        floor = mean.get("c1_floor")
        if floor is not None and "overhead_ms" in out:
            out["c1_floor_nsys_ms"] = floor
            out["c1_floor_median_ms"] = med.get("c1_floor")
            out["c1_floor_minus_overhead_ms"] = max(floor - max(out["overhead_ms"], 0.0), 0.0)
            out["c1_floor_adj_le_2ms"] = out["c1_floor_minus_overhead_ms"] <= 2.0
        out["note"] = ("same host, same server flags, same prompts and n_predict; means of the pooled decode steps; the unprofiled "
                       "non-compute time assumes the profiler leaves kernel durations unchanged")
    return out


def control_subset(order: list[dict], n: int = CONTROL_N) -> list[dict]:
    """The first n prompts of the balanced order: fixed by the seed, spread over the categories, inside the matrix prompts."""
    return order[:n]


def run_control(*, bin_dir, target: str, draft: str, subset: list[dict], budget: common.Budget, notes: list, gpu: str, ctx: int,
                nsys_n_predict: int, on_stage=None) -> dict:
    """Unprofiled K=4 on the subset, then the same requests from a fresh server under nsys. Never raises."""
    from infra.results import artifact_path
    cfg = next(c for c in matrix_configs(["k4"]))
    short = [dict(p, n_predict=min(p["n_predict"], nsys_n_predict)) for p in subset]
    out: dict = {"config": "k4", "subset_ids": [p["id"] for p in subset], "subset_categories": _count(subset), "n_predict": nsys_n_predict,
                 "request_gap_s": REQUEST_GAP_S, "unprofiled": None, "nsys": None, "overhead": None}
    un = run_config(cfg, bin_dir=bin_dir, target=target, draft=draft, prompts=short, ctx=ctx, notes=notes, gap_s=REQUEST_GAP_S,
                    budget_s=max(budget.left() * 0.4, 60.0), log_path=str(artifact_path(EXP, "server-control-unprofiled.log", gpu)))
    out["unprofiled"] = {k: un.get(k) for k in ("rows", "error", "startup_s", "wall_s", "smi", "checks", "cmd", "spec_log")}
    out["unprofiled"]["ms_per_step"] = step_ms(un.get("rows", []))
    if on_stage:
        on_stage(out)
    nsys = e4_bonsai.find_nsys()
    if not nsys:
        out["nsys"] = {"ok": False, "error": "nsys not found (CR_NSYS, PATH, /usr/local/cuda*/bin, /opt/nvidia/nsight-systems)"}
        return out
    wd = r1_step.private_workdir()
    reqs: list = []

    def driver(base: str) -> list[dict]:
        rows, _ = run_requests(short, common.Budget(max(budget.left() / 60.0 - 0.5, 0.5)), notes, cfg, keep_text=False, gap_s=REQUEST_GAP_S)
        reqs.extend(rows)
        return rows

    res = r1_step.capture(nsys=nsys, server_argv=config_cmd(bin_dir, target, draft, cfg, ctx=ctx), port=e5_spec.PORT,
                          prompt="Write a Python function that adds two numbers.", budget=budget, workdir=wd,
                          log_path=str(artifact_path(EXP, "server-control-nsys.log", gpu)), notes=notes, k=cfg["k"], driver=driver,
                          analyzer=analyze_segments,
                          on_stage=lambda r: on_stage(dict(out, nsys=dict(r, partial=True))) if on_stage else None)
    res["leftover_files"] = sorted(os.listdir(wd)) if os.path.isdir(wd) else []
    out["nsys"] = res
    res["n_requests"] = len(reqs)
    an = res.get("analysis") or {}
    if an.get("ok"):
        an["segments_match_requests"] = an.get("n_segments") == len(reqs)
    out["overhead"] = overhead(out["unprofiled"]["rows"], res.get("requests") or reqs, an)
    return out


# ---------------------------------------------------------------- part p

def run_probe(*, bin_dir, target: str, draft: str, order: list[dict], budget: common.Budget, notes: list, gpu: str, ctx: int,
              names: list[str]) -> dict:
    """One prompt per category, each config at -lv 5 with LLAMA_TRACE=1 (the counts include the 16-token warm-up request). The debug logs are written to a private dir (not the results
    dir, which the results server serves) and deleted after parsing, also when the run is cut."""
    ps = []
    for cat in PROBE_CATEGORIES:
        first = next((p for p in order if p["category"] == cat), None)
        if first:
            ps.append(dict(first, n_predict=PROBE_N_PREDICT[cat]))
    out: dict = {"prompt_ids": [p["id"] for p in ps], "n_predict": {p["id"]: p["n_predict"] for p in ps}, "configs": {}, "lv": PROBE_LV[1],
                 "llama_trace": PROBE_ENV["LLAMA_TRACE"], "note": "untimed: debug-level logging; counts only"}
    cfgs = {c["name"]: c for c in matrix_configs(CONFIG_NAMES)}
    wd = r1_step.private_workdir()
    try:
        for i, name in enumerate(names):
            if name not in cfgs:
                continue
            left = budget.left()
            if left < 50:
                notes.append(f"p: time budget used up; {names[i:]} skipped")
                break
            cfg = with_lv(cfgs[name], PROBE_LV)
            path = os.path.join(wd, f"server-probe-{name}.log")
            res = run_config(cfg, bin_dir=bin_dir, target=target, draft=draft, prompts=ps, budget_s=min(left / max(len(names) - i, 1), 120.0),
                             notes=notes, ctx=ctx, log_path=path, keep_text=False, env=PROBE_ENV)
            try:
                ev = count_spec_events(pathlib.Path(path).read_text(errors="replace"))
            except OSError:
                ev = {}
            out["configs"][name] = {"events": ev, "error": res.get("error"), "checks": res.get("checks"), "n_rows": len(res.get("rows") or []),
                                    "draft_tokens": sum(r.get("draft_n", 0) for r in res.get("rows") or []),
                                    "accepted_tokens": sum(r.get("draft_n_accepted", 0) for r in res.get("rows") or [])}
            try:
                os.unlink(path)
            except OSError:
                pass
    finally:
        shutil.rmtree(wd, ignore_errors=True)
    return out


# ---------------------------------------------------------------- run

def summary_rows(doc: dict) -> list[dict]:
    rows = []
    for name, v in ((doc.get("matrix") or {}).get("summary") or {}).items():
        a = v.get("all") or {}
        rows.append({"part": "m", "name": name, "n_paired": v.get("n_paired"), "tok_s": a.get("tok_s"), "speedup": a.get("speedup"),
                     "speedup_ci95": a.get("speedup_ci95"), "r": a.get("r"), "alpha": a.get("alpha"),
                     "text_equal_frac": a.get("text_equal_frac"), "rollback_path": (v.get("checks") or {}).get("rollback_path"),
                     "error": v.get("error")})
    drift = (doc.get("matrix") or {}).get("drift")
    if drift:
        rows.append({"part": "m", "name": "drift", "n_paired": drift.get("n"), "speedup": drift.get("ratio")})
    c = doc.get("control") or {}
    if c:
        rows.append({"part": "c", "name": "overhead", "nsys_ok": (c.get("nsys") or {}).get("ok"), "overhead": c.get("overhead")})
    for name, v in ((doc.get("probe") or {}).get("configs") or {}).items():
        rows.append({"part": "p", "name": name, "events": v.get("events"), "error": v.get("error")})
    return rows


def run(args) -> int:
    notes: list[str] = []
    args.gpu = common.gpu_tag(args)
    bin_dir = e4_bonsai.llama_bin()
    rows_all = make_prompts.load_longctx(args.prompts)
    order = make_prompts.shuffled(rows_all, args.seed, stratify=True)[:args.limit]
    if args.quick:
        order = [dict(p, n_predict=min(p["n_predict"], 32)) for p in order]
    parts = [p for p in PART_ORDER if p in args.parts]
    runnable = []
    for p in parts:
        miss = [b for b in NEEDS[p] if not (bin_dir / b).exists()]
        if miss:
            notes.append(f"part {p} skipped: {miss} not found in {bin_dir} (build llama.cpp with CR_BUILD_LLAMA=1)")
        else:
            runnable.append(p)
    total = common.Budget(args.budget_min)
    secrets = r1.secret_values()
    config = {"parts": parts, "budget_min": args.budget_min, "weights": WEIGHTS, "seed": args.seed, "limit": args.limit, "ctx": args.ctx,
              "configs": args.configs, "lv": LV, "n_predict": make_prompts.R2A_N_PREDICT, "quick": args.quick, "control_n": args.control_n,
              "nsys_n_predict": args.nsys_n_predict, "prompts_file": str(args.prompts), "models": {}}
    doc: dict = {"prompt_set": make_prompts.longctx_stats(rows_all), "matrix": None, "control": None, "probe": None, "host": None}

    def save(final: bool = False) -> None:
        extra = r1.scrub(doc, secrets)
        if final:
            common.write_result(EXP, args, r1.scrub(config, secrets), summary_rows(doc), r1.scrub(notes, secrets), False, extra=extra)
        else:
            common.write_partial(EXP, args, r1.scrub(config, secrets), summary_rows(doc), r1.scrub(notes, secrets), False, extra=extra)

    paths: dict = {}
    try:
        for name in ("pq2", "draft_bf16"):
            paths[name], config["models"][name] = e4_pdl.model_path(name, notes)
    except Exception as e:
        notes.append(f"model download failed: {type(e).__name__}: {e}")
        save(final=True)
        print(f"error: {notes[-1]}", file=sys.stderr)
        return 3
    from infra.results import run_dir
    (run_dir(EXP, args.gpu) / "raw").mkdir(parents=True, exist_ok=True)
    smi = e4_pdl.smi_sample()
    probe = r1.host_probe(bin_dir, paths["pq2"]) if {"m", "c"} & set(runnable) else None
    bad, cap = r1.host_health(smi, (probe or {}).get("smi"), check_power=not args.allow_power_cap)
    config["host_smi"] = smi
    doc["host"] = {"problem": bad, "allow_power_cap": args.allow_power_cap, "parts_skipped": [], "probe": probe}
    rc = 0
    if bad:
        skipped = [p for p in runnable if p in "mc"]
        doc["host"]["parts_skipped"] = skipped
        notes.append(f"unhealthy host: {bad}; parts {skipped} (timings) skipped, part p (no timing) still runs")
        runnable = [p for p in runnable if p == "p"]
        rc = 1
    elif cap:
        doc["host"]["power_cap_note"] = cap
        notes.append(f"power cap accepted (--allow-power-cap): {cap}; timings are for the capped host")
    cut = {"part": None}
    try:
        for i, p in enumerate(runnable):
            budget_s = part_budget_s(total.left(), p, runnable[i:])
            print(f"[r2a] part {p}: {budget_s / 60:.1f} min of {total.left() / 60:.1f} min left", flush=True)
            if budget_s < 60:
                notes.append(f"part {p} skipped: only {budget_s:.0f} s of budget left")
                continue
            sub = common.Budget(budget_s / 60.0)
            cut["part"] = p
            try:
                if p == "m":
                    doc["matrix"] = run_matrix(args=args, bin_dir=bin_dir, target=paths["pq2"], draft=paths["draft_bf16"], order=order,
                                               budget_s=budget_s, notes=notes, gpu=args.gpu,
                                               on_update=lambda m: (doc.__setitem__("matrix", m), save()))
                elif p == "c":
                    doc["control"] = run_control(bin_dir=bin_dir, target=paths["pq2"], draft=paths["draft_bf16"],
                                                 subset=control_subset(order, args.control_n), budget=sub, notes=notes, gpu=args.gpu,
                                                 ctx=args.ctx, nsys_n_predict=args.nsys_n_predict,
                                                 on_stage=lambda c: (doc.__setitem__("control", c), save()))
                else:
                    doc["probe"] = run_probe(bin_dir=bin_dir, target=paths["pq2"], draft=paths["draft_bf16"], order=order, budget=sub,
                                             notes=notes, gpu=args.gpu, ctx=args.ctx, names=[n for n in PROBE_CONFIGS if n in args.configs])
            except Exception as e:  # a failed part must not lose the others
                notes.append(f"part {p} crashed: {type(e).__name__}: {e}")
            cut["part"] = None
            save()
    finally:
        if cut["part"]:
            notes.append(f"run cut during part {cut['part']}: the results of that part are partial")
        if (doc.get("matrix") or {}).get("runs"):          # a cut leaves the last 15 s refresh: summarize again, with intervals
            try:
                doc["matrix"]["summary"] = summarize(doc["matrix"]["runs"], ci=True)
            except Exception as e:
                notes.append(f"final matrix summary failed: {type(e).__name__}: {e}")
        save(final=True)
    return rc


# ---------------------------------------------------------------- CLI

def _names(text):
    return [t.strip() for t in text.split(",") if t.strip()]


def parse_args(argv=None):
    ap = argparse.ArgumentParser(prog="exps.r2a_longctx", description=__doc__.split("\n")[0])
    common.add_common_args(ap)
    ap.add_argument("--parts", type=_names, default=list(PART_ORDER), help="m (matrix), c (overhead control), p (rollback probe)")
    ap.add_argument("--configs", type=_names, default=list(DEFAULT_CONFIGS), help=f"default {','.join(DEFAULT_CONFIGS)}; also k2, k3 (follow-up pod)")
    ap.add_argument("--limit", type=int, default=60, help="most prompts, taken from the seeded balanced order")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--ctx", type=int, default=CTX)
    ap.add_argument("--prompts", default=str(make_prompts.R2A_OUT))
    ap.add_argument("--control-n", type=int, default=CONTROL_N)
    ap.add_argument("--nsys-n-predict", type=int, default=NSYS_N_PREDICT)
    ap.add_argument("--allow-power-cap", action="store_true", help="a power cap below default is a note, not a skip of parts m and c")
    a = ap.parse_args(argv)
    bad = [p for p in a.parts if p not in WEIGHTS]
    names = set(CONFIG_NAMES)
    badc = [c for c in a.configs if c not in names]
    if bad or badc:
        ap.error(f"unknown part(s) {bad} or config(s) {badc}; configs: {sorted(names)}")
    if "base" not in a.configs and ("m" in a.parts):
        ap.error("--configs must include base: every ratio is against target alone")
    if a.ctx < 16384:
        ap.error("--ctx must be at least 16384: prompts reach 8K tokens and edits answer with 512")
    a.budget_min = a.budget_min or (8.0 if a.quick else DEFAULT_TOTAL_MIN)
    if a.quick:
        a.limit, a.control_n, a.nsys_n_predict = min(a.limit, 6), min(a.control_n, 3), min(a.nsys_n_predict, 16)
    return a


def plan_json(a) -> dict:
    bin_dir = e4_bonsai.llama_bin()
    rows = make_prompts.load_longctx(a.prompts)
    order = make_prompts.shuffled(rows, a.seed, stratify=True)[:a.limit]
    cfgs = config_order(matrix_configs(a.configs), a.seed)
    parts = [p for p in PART_ORDER if p in a.parts]
    m_s = part_budget_s(a.budget_min * 60.0, "m", parts) if "m" in parts else 0.0
    probe_cfg = {c["name"]: with_lv(c, PROBE_LV) for c in matrix_configs(a.configs)}
    return {"exp": EXP, "parts": parts, "budget_min": a.budget_min, "weights": WEIGHTS, "seed": a.seed, "ctx": a.ctx,
            "prompt_set": make_prompts.longctx_stats(rows), "n_prompts": len(order), "n_predict": make_prompts.R2A_N_PREDICT,
            "matrix": {"config_order": [c["name"] for c in cfgs], "plan": plan_n(cfgs, order, m_s) if m_s else None,
                       "configs": [{"name": c["name"], "desc": c["desc"], "cmd": config_cmd(bin_dir, "<pq2.gguf>", "<draft_bf16.gguf>", c, ctx=a.ctx)}
                                   for c in cfgs]},
            "control": {"subset_ids": [p["id"] for p in control_subset(order, a.control_n)], "n_predict": a.nsys_n_predict,
                        "server_cmd": config_cmd(bin_dir, "<pq2.gguf>", "<draft_bf16.gguf>", matrix_configs(["k4"])[0], ctx=a.ctx),
                        "nsys_profile": r1_step.profile_cmd("nsys", "<tmp>/r1step", ["<server_cmd>"]),
                        "env": "allowlisted (PATH, LD_LIBRARY_PATH, HOME, TMPDIR=<private dir>, LANG, CUDA_*, ...), secret-like names dropped",
                        "files": "private temp dir, deleted after parsing"},
            "probe": {"configs": [n for n in PROBE_CONFIGS if n in a.configs], "env": PROBE_ENV,
                      "prompt_categories": PROBE_CATEGORIES, "n_predict": PROBE_N_PREDICT,
                      "server_cmd": config_cmd(bin_dir, "<pq2.gguf>", "<draft_bf16.gguf>", probe_cfg["k4"], ctx=a.ctx) if "k4" in probe_cfg else None},
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
