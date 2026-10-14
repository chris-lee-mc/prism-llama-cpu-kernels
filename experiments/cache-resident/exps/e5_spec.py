"""E5: speculative decoding baseline. Bonsai 2 27B PQ2_0 target + Qwen3.5-0.8B BF16 draft via llama-server.

  python -m exps.e5_spec --k 0|4|6|8 [--limit 200] [--n-predict 192] [--prompts exps/data/prompts_e5.jsonl]

k=0 is the plain target. k>0 adds `-md DRAFT -ngld 99 --spec-type draft-simple --spec-draft-n-max K`
(flags from common/arg.cpp of this fork). Prompts go through /apply-template (thinking disabled when the
template supports it), then /completion with greedy sampling. If the server rejects the draft (vocab
mismatch; llama.cpp allows a small vocab size difference) the run is recorded as draft_incompatible and
repeated with `--spec-type ngram-simple`, clearly labelled.
R1 (exps.r1) reuses this module: matrix_configs() lists one server config per variant (K, target backend sampling -bs, PTQ1_0
draft, ngram-mod, ngram-map-k), plan_n() sizes the prompt count up front from a per-prompt cost model, run_config() runs
the shuffled prompts on one server, summarize_matrix() pairs each config with target alone per category. Target alone and
the n-gram configs also run a few extra code prompts past the N-prefix (code_extra), so criterion 3 does not depend on N.
Acceptance is reported as r (accepted / drafted) and, for draft-simple only, alpha (exps/spec_math.py).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import pathlib
import re
import shlex
import signal
import statistics
import subprocess
import sys
import time
import urllib.error
import urllib.request

from exps import common, hf_ids, make_prompts, spec_math

PORT = 8081
CTX = 16384
EXP_NAMES = {0: "e5_base", 4: "e5_k4", 6: "e5_k6", 8: "e5_k8"}  # result dir must equal the manifest key


# ---------------------------------------------------------------- pure helpers

def llama_bin() -> pathlib.Path:
    return pathlib.Path(os.environ.get("CR_LLAMA_BIN") or (common.EXP_DIR.parent.parent / "build-cuda" / "bin"))


def server_cmd(bin_dir, model: str, k: int, draft: str | None = None, port: int = PORT, ctx: int = CTX,
               spec_type: str = "draft-simple", extra: list[str] | None = None) -> list[str]:
    cmd = [str(pathlib.Path(bin_dir) / "llama-server"), "-m", model, "-ngl", "99", "-fa", "on", "-c", str(ctx), "-np", "1",
           "--host", "127.0.0.1", "--port", str(port)]
    if k > 0:
        if spec_type == "draft-simple":
            if not draft:
                raise ValueError("draft model path required for draft-simple")
            cmd += ["-md", draft, "-ngld", "99"]
        cmd += ["--spec-type", spec_type, "--spec-draft-n-max", str(k)]
    return cmd + list(extra or [])


def completion_body(prompt: str, n_predict: int) -> dict:
    return {"prompt": prompt, "n_predict": n_predict, "temperature": 0, "top_k": 1, "seed": 42, "cache_prompt": False}


def template_body(prompt: str, thinking: bool | None) -> dict:
    body: dict = {"messages": [{"role": "user", "content": prompt}]}
    if thinking is False:
        body["chat_template_kwargs"] = {"enable_thinking": False}
    return body


def parse_completion(resp: dict, keep_text: bool = False) -> dict:
    """Pull the per-prompt numbers out of a /completion response (and the output text when keep_text)."""
    t = resp.get("timings") or {}
    out = {"tokens_predicted": int(resp.get("tokens_predicted", t.get("predicted_n", 0)) or 0),
            "predicted_per_second": t.get("predicted_per_second"), "predicted_ms": t.get("predicted_ms"),
            "prompt_n": t.get("prompt_n"), "prompt_ms": t.get("prompt_ms"),
            "draft_n": int(t.get("draft_n", 0) or 0), "draft_n_accepted": int(t.get("draft_n_accepted", 0) or 0),
            "stop_type": resp.get("stop_type"), "truncated": resp.get("truncated"),
            "bs_flag": (resp.get("generation_settings") or {}).get("backend_sampling")}
    if keep_text:
        out["text"] = resp.get("content")
    return out


def summarize(rows: list[dict], k: int, wall_s: float) -> dict:
    ok = [r for r in rows if r.get("predicted_per_second")]
    drafted = sum(r.get("draft_n", 0) for r in rows)
    accepted = sum(r.get("draft_n_accepted", 0) for r in rows)
    ms = sum(r.get("predicted_ms") or 0 for r in ok)
    toks = sum(r.get("tokens_predicted", 0) for r in ok)
    speeds = [r["predicted_per_second"] for r in ok]
    st = spec_math.run_stats(toks, drafted, accepted, n_first=len(ok))
    return {"k": k, "n_prompts": len(rows), "n_ok": len(ok), "acceptance": accepted / drafted if drafted else None,
            "alpha": st["alpha"], "tokens_per_step": st["tokens_per_step"], "steps": st["steps"], "k_eff": st["k_eff"],
            "draft_n": drafted, "draft_n_accepted": accepted,
            "tok_s_mean": statistics.fmean(speeds) if speeds else None,
            "tok_s_median": statistics.median(speeds) if speeds else None,
            "tok_s_weighted": toks / (ms / 1000.0) if ms else None, "tokens": toks, "wall_s": wall_s}


def log_says_draft_rejected(text: str) -> bool:
    low = text.lower()
    return "vocab" in low and ("draft" in low or "speculat" in low)


# ---------------------------------------------------------------- http (stdlib)

def post_json(url: str, body: dict, timeout: float = 300.0) -> dict:
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def get_ok(url: str, timeout: float = 5.0) -> bool:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return r.status == 200
    except Exception:
        return False


def _die_with_parent() -> None:
    """Child side (Linux): SIGTERM the server when this process dies, even by SIGKILL."""
    try:
        import ctypes
        ctypes.CDLL("libc.so.6").prctl(1, signal.SIGTERM)  # PR_SET_PDEATHSIG
    except Exception:
        pass


class Server:
    """llama-server child process with its log in log_path.

    It stays in our process group, so the runner's group kill takes it down too; the parent-death signal covers
    a kill that reaches only this process.
    """

    def __init__(self, cmd: list[str], log_path: str, env: dict | None = None):
        """env: extra environment variables on top of ours (for example LLAMA_TRACE)."""
        self.cmd, self.log_path = cmd, log_path
        self.log = open(log_path, "w")
        self.proc = subprocess.Popen(cmd, stdout=self.log, stderr=subprocess.STDOUT, env=dict(os.environ, **env) if env else None,
                                     preexec_fn=_die_with_parent if sys.platform.startswith("linux") else None)

    def tail(self, n: int = 4000) -> str:
        self.log.flush()
        try:
            return pathlib.Path(self.log_path).read_text(errors="replace")[-n:]
        except OSError:
            return ""

    def head_tail(self, head: int = 200000, tail: int = 400000) -> str:
        """Start and end of the log: the startup lines (context parameters) are far from the end after a long run."""
        self.log.flush()
        try:
            data = pathlib.Path(self.log_path).read_bytes()
        except OSError:
            return ""
        if len(data) > head + tail:
            data = data[:head] + b"\n" + data[-tail:]
        return data.decode("utf-8", errors="replace")

    def wait_ready(self, timeout_s: float) -> tuple[bool, str | None]:
        t0 = time.monotonic()
        while time.monotonic() - t0 < timeout_s:
            if self.proc.poll() is not None:
                return False, f"server exited with code {self.proc.returncode}"
            if get_ok(f"http://127.0.0.1:{PORT}/health"):
                return True, None
            time.sleep(1.0)
        return False, "timeout waiting for /health"

    def stop(self) -> None:
        if self.proc.poll() is None:
            try:
                self.proc.terminate()
                self.proc.wait(timeout=20)
            except Exception:
                try:
                    self.proc.kill()
                    self.proc.wait(timeout=10)
                except Exception:
                    pass
        self.log.close()


def run_prompts(prompts: list[dict], n_predict: int, budget: common.Budget, notes: list, k: int, spec_type: str,
                on_row=None, keep_text: bool = False, keep_prompt: bool = False, reserve_s: float = 10.0):
    """Run all prompts against the live server. on_row(rows) is called after each prompt. Returns (rows, info).
    keep_prompt: also keep the chat-templated prompt text of each row (R1a builds its text from it). reserve_s: stop
    starting prompts this long before the budget ends."""
    base = f"http://127.0.0.1:{PORT}"
    info = {"thinking_disabled": None, "template_supports_enable_thinking": None, "draft_not_used": False,
            "on_row_error": False}
    rows = []
    for i, p in enumerate(prompts):
        if budget.expired(reserve_s):
            notes.append(f"time budget used up after {i} prompts")
            break
        row = {"id": p["id"], "category": p["category"]}
        try:
            plain = post_json(base + "/apply-template", template_body(p["prompt"], None))["prompt"]
            text = plain
            if info["thinking_disabled"] is not False:
                try:
                    nothink = post_json(base + "/apply-template", template_body(p["prompt"], False))["prompt"]
                    if info["template_supports_enable_thinking"] is None:
                        info["template_supports_enable_thinking"] = nothink != plain
                    text, info["thinking_disabled"] = nothink, True
                except urllib.error.HTTPError:
                    info["thinking_disabled"] = False
            resp = post_json(base + "/completion", completion_body(text, n_predict), budget.timeout(300.0, reserve_s=10))
            row.update(parse_completion(resp, keep_text))
            if keep_prompt:
                row["prompt_text"] = text
        except Exception as e:
            row["error"] = f"{type(e).__name__}: {e}"
        rows.append(row)
        if on_row:
            try:
                on_row(rows)
            except Exception as e:
                if not info.get("on_row_error"):
                    notes.append(f"partial save failed: {type(e).__name__}: {e}")
                info["on_row_error"] = True
        ok = [r for r in rows if "error" not in r]  # error rows say nothing about the draft
        if k > 0 and spec_type == "draft-simple" and len(ok) >= 3 and sum(r.get("draft_n", 0) for r in ok[:3]) == 0:
            info["draft_not_used"] = True
            break
    return rows, info


def load_gguf(kind: str, key: str | None, notes: list) -> tuple[str, dict]:
    from huggingface_hub import hf_hub_download
    info = hf_ids.resolve(kind, key)
    if info["exists"] is False:
        notes.append(f"{kind} {info['repo']}/{info['file']} not found; alternatives {info['alternatives']}")
    return hf_hub_download(info["repo"], info["file"]), info


def run(args) -> int:
    from infra.results import artifact_path
    notes: list[str] = []
    args.gpu = common.gpu_tag(args)
    bin_dir = llama_bin()
    if not (bin_dir / "llama-server").exists():
        print(f"error: {bin_dir / 'llama-server'} not found; build llama.cpp (CR_BUILD_LLAMA=1) or set CR_LLAMA_BIN",
              file=sys.stderr)
        return 2
    exp = EXP_NAMES[args.k]
    prompts = make_prompts.load(args.prompts, args.limit)
    budget = common.Budget(args.budget_min or (6 if args.quick else 27))
    try:
        target, tinfo = load_gguf("bonsai2", "PQ2_0", notes)
        draft, dinfo = (load_gguf("draft", None, notes) if args.k > 0 else (None, None))
    except Exception as e:
        notes.append(f"model download failed: {type(e).__name__}: {e}")
        common.write_result(exp, args, {"k": args.k, "target": hf_ids.resolve("bonsai2", "PQ2_0"),
                                        "draft": hf_ids.resolve("draft") if args.k > 0 else None}, [], notes, False)
        print(f"error: {notes[-1]}", file=sys.stderr)
        return 3
    attempts: list[dict] = []
    spec_type = "draft-simple" if args.k > 0 else "none"
    rows: list[dict] = []
    info: dict = {}
    wall = 0.0
    run_t0 = time.monotonic()

    def save_rows(done: list) -> None:
        common.write_partial(exp, args, {"k": args.k, "spec_type": spec_type, "target": tinfo, "draft": dinfo,
                                         "attempts": attempts}, done, notes, False,
                             extra={"summary": summarize(done, args.k, time.monotonic() - run_t0)})

    for attempt in range(2):
        cmd = server_cmd(bin_dir, target, args.k, draft, spec_type=spec_type) + shlex.split(args.server_args)
        log_path = str(artifact_path(exp, f"server-{spec_type}.log", args.gpu))
        srv = Server(cmd, log_path)
        t0 = time.monotonic()
        att = {"spec_type": spec_type, "cmd": cmd, "log": log_path}
        try:
            ready, err = srv.wait_ready(budget.timeout(900.0, reserve_s=30))
            if not ready:
                att["error"] = err
                att["log_tail"] = srv.tail()
                rows, info = [], {}
            else:
                rows, info = run_prompts(prompts, args.n_predict, budget, notes, args.k, spec_type, save_rows)
            wall = time.monotonic() - t0
        finally:
            srv.stop()
        att["draft_rejected"] = bool(args.k > 0 and spec_type == "draft-simple" and (
            info.get("draft_not_used") or (att.get("error") and log_says_draft_rejected(att.get("log_tail", "")))))
        attempts.append(att)
        if att["draft_rejected"]:
            notes.append("draft_incompatible: " + (att.get("error") or "server ran but drafted 0 tokens")
                         + "; rerunning with ngram-simple (NOT the Qwen3.5-0.8B draft)")
            spec_type = "ngram-simple"
            continue
        break
    summary = summarize(rows, args.k, wall)
    summary["spec_type"] = spec_type
    summary["draft_incompatible"] = any(a.get("draft_rejected") for a in attempts)
    if summary["draft_incompatible"]:
        summary["label"] = "ngram-simple fallback, not the draft model"
    config = {"k": args.k, "limit": args.limit, "server_args": args.server_args, "n_predict": args.n_predict, "ctx": CTX, "prompts_file": str(args.prompts),
              "target": tinfo, "draft": dinfo, "attempts": attempts, "chat": info, "sampling": completion_body("", args.n_predict),
              "quick": args.quick}
    if not rows:
        notes.append("no prompt completed; see attempts[].log_tail")
    common.write_result(exp, args, config, rows, notes, False, extra={"summary": summary})
    return 0 if rows else 1


# ---------------------------------------------------------------- R1 matrix: one server per config, same prompts

SEED = 20261013
CODE_CATS = ("code_fix", "refactor")
# Target-side GPU sampling (llama-server --backend-sampling, common.h sampling.backend_sampling, default off). The draft
# of draft-simple always samples on the CPU (common_sampler_sample in common/speculative.cpp); --spec-draft-backend-sampling
# only reaches the eagle3, dflash and mtp types, so it is not a knob of this matrix.
TARGET_BS = "-bs"
NGRAM_MOD_SHORT = ["--spec-ngram-mod-n-match", "4", "--spec-ngram-mod-n-min", "2", "--spec-ngram-mod-n-max", "16"]
NGRAM_MAP_K_SHORT = ["--spec-ngram-map-k-size-n", "4", "--spec-ngram-map-k-size-m", "16", "--spec-ngram-map-k-min-hits", "1"]
STARTUP_S = 25.0       # model load and warm-up per server
PLAN_MARGIN = 0.9      # plan for this share of the part budget; the rest absorbs slow prompts
N_FLOOR = 8            # prompts when even this many do not fit the budget
MIN_N_CATEGORY = 15    # paired prompts per category below which a speedup is too noisy to decide a go / no-go
CODE_FLOOR = MIN_N_CATEGORY + 5  # code_fix / refactor prompts that base and the n-gram configs run whatever N is (spare for failed prompts)


def _cfg(name, spec_type, k=0, extra=(), prompt_s=1.2, categories=None, frac=1.0, draft="bf16", desc="", default=True,
         extra_code=False):
    """prompt_s: expected seconds per prompt (E5 runs: target alone 1.2, K=4 1.9). frac: share of the main prompts (a
    prefix of the category-balanced order). draft: which draft GGUF (bf16 or ptq1_0). default: in the R1 default --configs.
    extra_code: also runs the extra code prompts (code_extra); base and the n-gram configs, the pairs of criterion 3."""
    return {"name": name, "spec_type": spec_type, "k": k, "extra": list(extra), "prompt_s": prompt_s, "frac": frac,
            "categories": list(categories) if categories else None, "draft": draft, "desc": desc, "default": default,
            "extra_code": extra_code}


def matrix_configs() -> list[dict]:
    """R1b configs in run order; base is first and every other config runs a subset of its prompts. The n-gram types only
    run on the code categories (the only ones criterion 3 needs), and they and base also run the extra code prompts: this
    keeps n per code category above MIN_N_CATEGORY whatever N is. K=2/3 are not in the R1 default --configs (default False):
    they cost about 20% of part b, which a 30 min pod cannot spare; run them on a follow-up pod.
    The defaults (n_match 24 / size_n 12) need long repeats, which 150-token prompts rarely have, hence the short variants."""
    return [
        _cfg("base", "none", desc="target alone", extra_code=True),
        _cfg("k4", "draft-simple", 4, prompt_s=1.9, desc="draft-simple K=4, as in E5 (control)"),
        _cfg("k4_tbs", "draft-simple", 4, [TARGET_BS], 1.9, desc="draft-simple K=4 with target backend sampling (-bs)"),
        _cfg("k2", "draft-simple", 2, prompt_s=1.6, frac=0.5, desc="draft-simple K=2", default=False),
        _cfg("k3", "draft-simple", 3, prompt_s=1.75, frac=0.5, desc="draft-simple K=3", default=False),
        _cfg("k4_ptq1", "draft-simple", 4, prompt_s=2.0, frac=0.5, draft="ptq1_0",
             desc="draft-simple K=4 with the post-hoc PTQ1_0 draft (real alpha drop for criterion 2)"),
        _cfg("ngram_mod", "ngram-mod", prompt_s=1.3, categories=CODE_CATS, extra_code=True,
             desc="ngram-mod, fork defaults (n_match 24, n_min 48, n_max 64); code_fix and refactor only"),
        _cfg("ngram_map_k", "ngram-map-k", prompt_s=1.3, categories=CODE_CATS, extra_code=True,
             desc="ngram-map-k, fork defaults (size_n 12, size_m 48, min_hits 1); code_fix and refactor only"),
        _cfg("ngram_mod_short", "ngram-mod", extra=NGRAM_MOD_SHORT, prompt_s=1.3, categories=CODE_CATS, extra_code=True,
             desc="ngram-mod n_match 4, n_min 2, n_max 16; code_fix and refactor only"),
        _cfg("ngram_map_k_short", "ngram-map-k", extra=NGRAM_MAP_K_SHORT, prompt_s=1.3, categories=CODE_CATS, extra_code=True,
             desc="ngram-map-k size_n 4, size_m 16; code_fix and refactor only"),
    ]


def config_cmd(bin_dir, target: str, draft: str | None, cfg: dict, port: int | None = None) -> list[str]:
    port = port or PORT
    t = cfg["spec_type"]
    if t == "draft-simple":
        return server_cmd(bin_dir, target, cfg["k"], draft, port, extra=cfg["extra"])
    cmd = server_cmd(bin_dir, target, 0, None, port)
    return cmd + (["--spec-type", t] if t != "none" else []) + cfg["extra"]


def code_extra(prompts: list[dict], n: int, per_cat: int) -> list[dict]:
    """Code prompts after the first n (in order) that bring each code category up to per_cat prompts in prompts[:n] + extra."""
    have = {c: sum(1 for p in prompts[:n] if p["category"] == c) for c in CODE_CATS}
    out = []
    for p in prompts[n:]:
        c = p["category"]
        if c in have and have[c] < per_cat:
            have[c] += 1
            out.append(p)
    return out


def select_prompts(cfg: dict, main: list[dict], extra: list[dict] = ()) -> list[dict]:
    """Prompts of one config: its share of the main prompts (a prefix, so still category balanced), plus the extra code prompts
    for extra_code configs, then its categories."""
    ps = main[:max(1, math.ceil(cfg["frac"] * len(main)))] if main else []
    if cfg.get("extra_code"):
        ps = ps + list(extra)
    if cfg["categories"]:
        ps = [p for p in ps if p["category"] in cfg["categories"]]
    return ps


def cost_s(cfg: dict, n_prompts: int) -> float:
    return STARTUP_S + n_prompts * cfg["prompt_s"] if n_prompts else 0.0


def plan_n(cfgs: list[dict], prompts: list[dict], budget_s: float, margin: float = PLAN_MARGIN, floor: int = N_FLOOR,
           code_floor: int = 0) -> dict:
    """Largest number N of main prompts (a prefix of the category-balanced order) for which the summed expected cost of all
    configs fits margin * budget_s. Base runs these N prompts, every other config a subset of them. code_floor > 0 adds
    code_extra prompts (counted in the cost) so base and the n-gram configs see that many per code category at any N."""
    best = None
    for n in range(len(prompts), 0, -1):
        extra = code_extra(prompts, n, code_floor)
        est = sum(cost_s(c, len(select_prompts(c, prompts[:n], extra))) for c in cfgs)
        if est <= margin * budget_s:
            best = n
            break
    fits = best is not None
    n = best if fits else min(floor, len(prompts))
    main = prompts[:n]
    extra = code_extra(prompts, n, code_floor)
    per_cat: dict = {}
    for p in main:
        per_cat[p["category"]] = per_cat.get(p["category"], 0) + 1
    code_cat = {c: per_cat.get(c, 0) + sum(1 for p in extra if p["category"] == c) for c in CODE_CATS}
    return {"n": n, "fits": fits, "est_s": sum(cost_s(c, len(select_prompts(c, main, extra))) for c in cfgs), "budget_s": budget_s,
            "per_config": {c["name"]: len(select_prompts(c, main, extra)) for c in cfgs}, "per_category": per_cat,
            "code_floor": code_floor, "n_extra": len(extra), "per_code_category": code_cat,
            "min_code_category": min(code_cat.values(), default=0)}


def time_shares(costs: list[float], left_s: float) -> list[float]:
    """Seconds for each config from left_s, proportional to its expected cost."""
    tot = sum(costs) or 1.0
    return [left_s * c / tot for c in costs]


def spec_log_lines(text: str, n: int = 6) -> list[str]:
    """Server log lines about speculative decoding (enabled / disabled / checkpoints), to show the type really ran."""
    return [ln.strip()[:240] for ln in text.splitlines() if "specul" in ln.lower()][:n]


BS_PROBLEM_RE = re.compile(r"backend sampl\w*[^\n]*(not supported|not compatible|at most|disabl|fail)|cannot be offloaded|invalid backend sampled", re.I)


def log_flags(text: str) -> dict:
    """Server log signs that backend sampling was refused or that speculation fell back to KV checkpoints (replay path)."""
    lines = text.splitlines()
    return {"backend_sampling_problems": [ln.strip()[:240] for ln in lines if BS_PROBLEM_RE.search(ln)][:3],
            "kv_checkpoint_fallback": any("KV checkpoints instead" in ln for ln in lines)}


N_RS_SEQ_RE = re.compile(r"(?:^|[\s:])n_rs_seq\s+=\s+(\d+)\s*$")


def rollback_flags(text: str, spec_type: str) -> dict:
    """Rollback path of the target context from the INFO line `n_rs_seq = N` of llama-context.cpp (printed once per context;
    the target comes first, the draft context second). need_n_rs_seq() (common.h) gives N = draft n_max for draft-simple and
    0 for ngram-mod / ngram-map-k: with N = 0 a hybrid (GDN) target takes the KV-checkpoint path, every drafting step saves a
    checkpoint and every partial acceptance restores it and replays the target pass. The server logs that path at TRC level
    only, so the INFO line is the only sign in a default log. rollback_path: none (no speculation), in_batch, checkpoint or
    unknown (no line found). checkpoint_replay: True on the checkpoint path, None when unknown."""
    vals = [int(m.group(1)) for ln in text.splitlines() for m in [N_RS_SEQ_RE.search(ln)] if m]
    tgt = vals[0] if vals else None
    if spec_type == "none":
        path, replay = "none", False
    elif tgt is None:
        path, replay = "unknown", None
    elif tgt == 0:
        path, replay = "checkpoint", True
    else:
        path, replay = "in_batch", False
    return {"n_rs_seq": vals[:2], "n_rs_seq_target": tgt, "rollback_path": path, "checkpoint_replay": replay}


def warmup(n_predict: int = 16) -> None:
    """Untimed request: CUDA graph capture and allocator growth must not land in the first timed prompt."""
    try:
        post_json(f"http://127.0.0.1:{PORT}/completion", completion_body("Write a Python function that adds two numbers.", n_predict), 120.0)
    except Exception:
        pass


def run_config(cfg: dict, *, bin_dir, target: str, draft: str | None, prompts: list[dict], n_predict: int, budget_s: float,
               notes: list, log_path: str, on_row=None, keep_prompt: bool = False) -> dict:
    """Start one server, run the prompts in order within budget_s (startup included), stop it. Never raises."""
    from exps import e4_pdl
    cmd = config_cmd(bin_dir, target, draft, cfg)
    res = {"name": cfg["name"], "spec_type": cfg["spec_type"], "k": cfg["k"], "cmd": cmd, "log": log_path, "rows": [],
           "error": None, "budget_s": budget_s, "planned": len(prompts)}
    budget = common.Budget(budget_s / 60.0)
    srv = Server(cmd, log_path)
    t0 = time.monotonic()
    try:
        with e4_pdl.SmiWatch() as watch:
            ready, err = srv.wait_ready(budget.timeout(600.0, reserve_s=10))
            res["startup_s"] = time.monotonic() - t0
            if not ready:
                res["error"] = err
                res["log_tail"] = srv.tail()
            else:
                warmup()
                # model load is part of the share, but prompts keep at least 30% of it
                run_budget = common.Budget(max(budget_s - (time.monotonic() - t0), 0.3 * budget_s) / 60.0)
                rows, info = run_prompts(prompts, n_predict, run_budget, notes, cfg["k"], cfg["spec_type"], on_row, keep_text=True,
                                         keep_prompt=keep_prompt)
                res["rows"], res["chat"] = rows, info
        res["smi"] = watch.summary()
    except Exception as e:  # keep the other configs going
        res["error"] = f"{type(e).__name__}: {e}"
    finally:
        log = srv.head_tail()
        res["spec_log"] = spec_log_lines(log)
        res["checks"] = {"backend_sampling_expected": TARGET_BS in cfg["extra"],
                         "backend_sampling_reported": next((r["bs_flag"] for r in res["rows"] if r.get("bs_flag") is not None), None),
                         **log_flags(log), **rollback_flags(log, cfg["spec_type"])}
        srv.stop()
    res["wall_s"] = time.monotonic() - t0
    return res


def _ok(r: dict) -> bool:
    return "error" not in r and bool(r.get("predicted_per_second")) and bool(r.get("predicted_ms"))


def _agg(rows: list[dict], spec_type: str = "draft-simple") -> dict:
    """Sums over rows. alpha and K_eff only for draft-simple; the first token of each request is not a decode step."""
    toks = sum(r["tokens_predicted"] for r in rows)
    ms = sum(r["predicted_ms"] for r in rows)
    acc = sum(r.get("draft_n_accepted", 0) for r in rows)
    n_first = sum(1 for r in rows if r["tokens_predicted"] > 0)
    st = spec_math.run_stats(toks, sum(r.get("draft_n", 0) for r in rows), acc, n_first, with_alpha=spec_type == "draft-simple")
    return {"n": len(rows), "tokens": toks, "ms": ms, "tok_s": toks / (ms / 1e3) if ms else None, "r": st["r"],
            "alpha": st["alpha"], "k_eff": st["k_eff"], "steps": st["steps"], "tokens_per_step": st["tokens_per_step"],
            "ms_per_step": spec_math.observed_step_ms(ms, toks, acc, n_first)}


def _pair_stats(c_rows: list[dict], b_rows: list[dict], spec_type: str = "draft-simple", ci: bool = False,
                base_type: str = "none") -> dict:
    c, b = _agg(c_rows, spec_type), _agg(b_rows, base_type)
    ratios = [x["predicted_per_second"] / y["predicted_per_second"] for x, y in zip(c_rows, b_rows)]
    texts = [(x.get("text"), y.get("text")) for x, y in zip(c_rows, b_rows) if x.get("text") is not None and y.get("text") is not None]
    eq = [a == b_ for a, b_ in texts]
    fracs = [spec_math.common_prefix_len(a, b_) / max(len(b_), 1) for a, b_ in texts]
    out = {"n": c["n"], "tok_s": c["tok_s"], "base_tok_s": b["tok_s"],
           "speedup": c["tok_s"] / b["tok_s"] if c["tok_s"] and b["tok_s"] else None,
           "speedup_median_prompt": common.median(ratios), "r": c["r"], "alpha": c["alpha"], "base_alpha": b["alpha"], "k_eff": c["k_eff"],
           "tokens_per_step": c["tokens_per_step"], "ms_per_step": c["ms_per_step"], "base_ms_per_token": 1e3 / b["tok_s"] if b["tok_s"] else None,
           "draft_n": sum(x.get("draft_n", 0) for x in c_rows), "draft_n_accepted": sum(x.get("draft_n_accepted", 0) for x in c_rows),
           "text_equal": sum(eq), "text_compared": len(eq), "text_equal_frac": sum(eq) / len(eq) if eq else None,
           "median_prefix_frac": common.median(fracs)}
    if ci:
        pairs = [(x["tokens_predicted"], x["predicted_ms"], y["tokens_predicted"], y["predicted_ms"]) for x, y in zip(c_rows, b_rows)]
        out["speedup_ci95"] = spec_math.bootstrap_speedup_ci(pairs)
    return out


def summarize_matrix(runs: dict, base: str = "base", ci: bool = False) -> dict:
    """Per config, per category and overall: tok/s, speedup, r, alpha, output equality, all on the prompts that the
    config and the target-alone run both finished (paired), so a time-budget cut cannot bias the ratio.
    ci: add a bootstrap 95% interval of the speedup (prompts resampled) to each entry; the final summary uses it."""
    out: dict = {}
    base_rows = {r["id"]: r for r in (runs.get(base) or {}).get("rows", []) if _ok(r)}
    for name, run in runs.items():
        rows = {r["id"]: r for r in run.get("rows", []) if _ok(r)}
        ids = [i for i in rows if i in base_rows]
        st = run.get("spec_type") or "none"
        entry = {"planned": run.get("planned"), "n_ok": len(rows), "n_paired": len(ids), "error": run.get("error"), "k": run.get("k"),
                 "spec_type": run.get("spec_type"), "draft": run.get("draft"), "checks": run.get("checks"), "by_category": {},
                 # on the checkpoint path a replay pass is one more step in tokens_per_step and ms_per_step
                 "steps_count_replays": bool((run.get("checks") or {}).get("checkpoint_replay"))}
        if ids:
            entry["all"] = _pair_stats([rows[i] for i in ids], [base_rows[i] for i in ids], st, ci)
            for cat in sorted({rows[i]["category"] for i in ids}):
                sel = [i for i in ids if rows[i]["category"] == cat]
                entry["by_category"][cat] = _pair_stats([rows[i] for i in sel], [base_rows[i] for i in sel], st, ci)
        out[name] = entry
    return out


def compare_pair(runs: dict, a: str, b: str, ci: bool = True) -> dict | None:
    """Config a against config b (not target alone) on the prompts both finished: speedup, alpha of both, text equality."""
    ra = {r["id"]: r for r in (runs.get(a) or {}).get("rows", []) if _ok(r)}
    rb = {r["id"]: r for r in (runs.get(b) or {}).get("rows", []) if _ok(r)}
    ids = [i for i in ra if i in rb]
    if not ids:
        return None
    return _pair_stats([ra[i] for i in ids], [rb[i] for i in ids], runs[a].get("spec_type") or "none", ci, runs[b].get("spec_type") or "none")


def _width(d: dict, w: int):
    return d.get(str(w), d.get(w))


def step_residual(stats: dict, k: int, cost: dict | None, draft_key: str = "draft_bf16") -> dict | None:
    """Observed ms per step minus the e5_cost step model k t_draft + t_draft_batch(k + 1) + t_verify(k + 1); the part no
    kernel time explains. The model without the draft re-eval term is kept next to it (cost files from before the draft
    width rows have no re-eval term: reeval_ms None, and both models agree)."""
    try:
        d = cost["drafts"][draft_key]
        td = d["t_draft_ms"]
        tv = _width(cost["target_verify_ms"], k + 1)
    except (KeyError, TypeError):
        return None
    obs = stats.get("ms_per_step")
    if obs is None or tv is None:
        return None
    reeval = _width(d.get("t_draft_batch_ms") or {}, k + 1)
    base_model = k * td + tv
    model = base_model + (reeval or 0.0)
    return {"observed_ms": obs, "model_ms": model, "residual_ms": obs - model, "reeval_ms": reeval,
            "model_ms_no_reeval": base_model, "residual_ms_no_reeval": obs - base_model}


# ---------------------------------------------------------------- CLI

def parse_args(argv=None):
    ap = argparse.ArgumentParser(prog="exps.e5_spec", description=__doc__.split("\n")[0])
    common.add_common_args(ap)
    ap.add_argument("--k", type=int, choices=sorted(EXP_NAMES), default=0, help="draft tokens per step, 0 = no speculation")
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--n-predict", type=int, default=192)
    ap.add_argument("--prompts", default=str(make_prompts.DEFAULT_OUT))
    ap.add_argument("--server-args", default="", help="extra llama-server flags, e.g. --server-args=-bs")
    a = ap.parse_args(argv)
    if a.quick:
        a.limit, a.n_predict = min(a.limit, 6), min(a.n_predict, 48)
    return a


def plan_json(a) -> dict:
    bin_dir = llama_bin()
    spec = "draft-simple" if a.k > 0 else "none"
    return {"exp": EXP_NAMES[a.k], "k": a.k, "limit": a.limit, "n_predict": a.n_predict,
            "target": hf_ids.resolve("bonsai2", "PQ2_0", offline=True),
            "draft": hf_ids.resolve("draft", None, offline=True) if a.k > 0 else None,
            "server_cmd": server_cmd(bin_dir, "<target.gguf>", a.k, "<draft.gguf>", spec_type=spec),
            "fallback_cmd": server_cmd(bin_dir, "<target.gguf>", a.k, None, spec_type="ngram-simple") if a.k > 0 else None,
            "endpoints": ["/health", "/apply-template", "/completion"], "completion_body": completion_body("<prompt>", a.n_predict),
            "n_prompts": len(make_prompts.load(a.prompts, a.limit)), "quick": a.quick}


def main(argv=None) -> int:
    a = parse_args(argv)
    if a.dry_run:
        common.print_plan(plan_json(a))
        return 0
    with common.term_exits():
        return run(a)


if __name__ == "__main__":
    sys.exit(main())
