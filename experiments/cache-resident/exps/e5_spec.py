"""E5: speculative decoding baseline. Bonsai 2 27B PQ2_0 target + Qwen3.5-0.8B BF16 draft via llama-server.

  python -m exps.e5_spec --k 0|4|6|8 [--limit 200] [--n-predict 192] [--prompts exps/data/prompts_e5.jsonl]

k=0 is the plain target. k>0 adds `-md DRAFT -ngld 99 --spec-type draft-simple --spec-draft-n-max K`
(flags from common/arg.cpp of this fork). Prompts go through /apply-template (thinking disabled when the
template supports it), then /completion with greedy sampling. If the server rejects the draft (vocab
mismatch; llama.cpp allows a small vocab size difference) the run is recorded as draft_incompatible and
repeated with `--spec-type ngram-simple`, clearly labelled.
R1 (exps.r1) reuses this module: matrix_configs() lists one server config per variant (K, draft backend sampling flag,
ngram-mod, ngram-map-k), run_config() runs the same shuffled prompts on one server, summarize_matrix() pairs each config
with target alone per category. Acceptance is reported as r (accepted / drafted) and alpha (exps/spec_math.py).
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
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
            "stop_type": resp.get("stop_type"), "truncated": resp.get("truncated")}
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
    st = spec_math.run_stats(toks, drafted, accepted)
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

    def __init__(self, cmd: list[str], log_path: str):
        self.cmd, self.log_path = cmd, log_path
        self.log = open(log_path, "w")
        self.proc = subprocess.Popen(cmd, stdout=self.log, stderr=subprocess.STDOUT,
                                     preexec_fn=_die_with_parent if sys.platform.startswith("linux") else None)

    def tail(self, n: int = 4000) -> str:
        self.log.flush()
        try:
            return pathlib.Path(self.log_path).read_text(errors="replace")[-n:]
        except OSError:
            return ""

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
                on_row=None, keep_text: bool = False):
    """Run all prompts against the live server. on_row(rows) is called after each prompt. Returns (rows, info)."""
    base = f"http://127.0.0.1:{PORT}"
    info = {"thinking_disabled": None, "template_supports_enable_thinking": None, "draft_not_used": False,
            "on_row_error": False}
    rows = []
    for i, p in enumerate(prompts):
        if budget.expired(30):
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
BS_ON, BS_OFF = "--spec-draft-backend-sampling", "--no-spec-draft-backend-sampling"  # common/arg.cpp; the fork default is ON
NGRAM_MOD_SHORT = ["--spec-ngram-mod-n-match", "4", "--spec-ngram-mod-n-min", "2", "--spec-ngram-mod-n-max", "16"]
NGRAM_MAP_K_SHORT = ["--spec-ngram-map-k-size-n", "4", "--spec-ngram-map-k-size-m", "16", "--spec-ngram-map-k-min-hits", "1"]


def _cfg(name, spec_type, k=0, extra=(), weight=1.0, categories=None, desc=""):
    return {"name": name, "spec_type": spec_type, "k": k, "extra": list(extra), "weight": weight,
            "categories": list(categories) if categories else None, "desc": desc}


def matrix_configs() -> list[dict]:
    """R1b configs in run order. weight = expected cost relative to target alone (time share). Short n-gram variants only
    run on the code categories: the defaults (n_match 24 / size_n 12) need long repeats, which 150-token prompts rarely have."""
    return [
        _cfg("base", "none", desc="target alone"),
        _cfg("k4_bs", "draft-simple", 4, [BS_ON], 2.0, desc="draft-simple K=4, draft backend sampling on"),
        _cfg("k4_nobs", "draft-simple", 4, [BS_OFF], 2.0, desc="control: draft-simple K=4, draft sampling on the CPU"),
        _cfg("k2_bs", "draft-simple", 2, [BS_ON], 1.6, desc="draft-simple K=2, backend sampling"),
        _cfg("k3_bs", "draft-simple", 3, [BS_ON], 1.8, desc="draft-simple K=3, backend sampling"),
        _cfg("ngram_mod", "ngram-mod", desc="ngram-mod, fork defaults (n_match 24, n_min 48, n_max 64)"),
        _cfg("ngram_map_k", "ngram-map-k", desc="ngram-map-k, fork defaults (size_n 12, size_m 48, min_hits 1)"),
        _cfg("ngram_mod_short", "ngram-mod", extra=NGRAM_MOD_SHORT, categories=CODE_CATS,
             desc="ngram-mod n_match 4, n_min 2, n_max 16; code_fix and refactor only"),
        _cfg("ngram_map_k_short", "ngram-map-k", extra=NGRAM_MAP_K_SHORT, categories=CODE_CATS,
             desc="ngram-map-k size_n 4, size_m 16; code_fix and refactor only"),
    ]


def config_cmd(bin_dir, target: str, draft: str | None, cfg: dict, port: int | None = None) -> list[str]:
    port = port or PORT
    t = cfg["spec_type"]
    if t == "draft-simple":
        return server_cmd(bin_dir, target, cfg["k"], draft, port, extra=cfg["extra"])
    cmd = server_cmd(bin_dir, target, 0, None, port)
    return cmd + (["--spec-type", t] if t != "none" else []) + cfg["extra"]


def time_shares(cfgs: list[dict], left_s: float) -> list[float]:
    """Seconds for each config from left_s, proportional to weight."""
    tot = sum(c["weight"] for c in cfgs) or 1.0
    return [left_s * c["weight"] / tot for c in cfgs]


def spec_log_lines(text: str, n: int = 6) -> list[str]:
    """Server log lines about speculative decoding (enabled / disabled / checkpoints), to show the type really ran."""
    return [ln.strip()[:240] for ln in text.splitlines() if "specul" in ln.lower()][:n]


def warmup(n_predict: int = 16) -> None:
    """Untimed request: CUDA graph capture and allocator growth must not land in the first timed prompt."""
    try:
        post_json(f"http://127.0.0.1:{PORT}/completion", completion_body("Write a Python function that adds two numbers.", n_predict), 120.0)
    except Exception:
        pass


def run_config(cfg: dict, *, bin_dir, target: str, draft: str | None, prompts: list[dict], n_predict: int, budget_s: float,
               notes: list, log_path: str, on_row=None) -> dict:
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
                # model load is part of the share, but prompts keep at least 60% of it
                run_budget = common.Budget(max(budget_s - (time.monotonic() - t0), 0.6 * budget_s) / 60.0)
                rows, info = run_prompts(prompts, n_predict, run_budget, notes, cfg["k"], cfg["spec_type"], on_row, keep_text=True)
                res["rows"], res["chat"] = rows, info
        res["smi"] = watch.summary()
    except Exception as e:  # keep the other configs going
        res["error"] = f"{type(e).__name__}: {e}"
    finally:
        res["spec_log"] = spec_log_lines(srv.tail(200000))
        srv.stop()
    res["wall_s"] = time.monotonic() - t0
    return res


def _ok(r: dict) -> bool:
    return "error" not in r and bool(r.get("predicted_per_second")) and bool(r.get("predicted_ms"))


def _agg(rows: list[dict]) -> dict:
    toks = sum(r["tokens_predicted"] for r in rows)
    ms = sum(r["predicted_ms"] for r in rows)
    st = spec_math.run_stats(toks, sum(r.get("draft_n", 0) for r in rows), sum(r.get("draft_n_accepted", 0) for r in rows))
    return {"n": len(rows), "tokens": toks, "ms": ms, "tok_s": toks / (ms / 1e3) if ms else None, "r": st["r"],
            "alpha": st["alpha"], "k_eff": st["k_eff"], "steps": st["steps"], "tokens_per_step": st["tokens_per_step"],
            "ms_per_step": spec_math.observed_step_ms(ms, toks, sum(r.get("draft_n_accepted", 0) for r in rows))}


def _pair_stats(c_rows: list[dict], b_rows: list[dict]) -> dict:
    c, b = _agg(c_rows), _agg(b_rows)
    ratios = [x["predicted_per_second"] / y["predicted_per_second"] for x, y in zip(c_rows, b_rows)]
    texts = [(x.get("text"), y.get("text")) for x, y in zip(c_rows, b_rows) if x.get("text") is not None and y.get("text") is not None]
    eq = [a == b_ for a, b_ in texts]
    fracs = [spec_math.common_prefix_len(a, b_) / max(len(b_), 1) for a, b_ in texts]
    return {"n": c["n"], "tok_s": c["tok_s"], "base_tok_s": b["tok_s"],
            "speedup": c["tok_s"] / b["tok_s"] if c["tok_s"] and b["tok_s"] else None,
            "speedup_median_prompt": common.median(ratios), "r": c["r"], "alpha": c["alpha"], "k_eff": c["k_eff"],
            "tokens_per_step": c["tokens_per_step"], "ms_per_step": c["ms_per_step"], "base_ms_per_token": 1e3 / b["tok_s"] if b["tok_s"] else None,
            "draft_n": sum(x.get("draft_n", 0) for x in c_rows), "draft_n_accepted": sum(x.get("draft_n_accepted", 0) for x in c_rows),
            "text_equal": sum(eq), "text_compared": len(eq), "text_equal_frac": sum(eq) / len(eq) if eq else None,
            "median_prefix_frac": common.median(fracs)}


def summarize_matrix(runs: dict, base: str = "base") -> dict:
    """Per config, per category and overall: tok/s, speedup, r, alpha, output equality, all on the prompts that the
    config and the target-alone run both finished (paired), so a time-budget cut cannot bias the ratio."""
    out: dict = {}
    base_rows = {r["id"]: r for r in (runs.get(base) or {}).get("rows", []) if _ok(r)}
    for name, run in runs.items():
        rows = {r["id"]: r for r in run.get("rows", []) if _ok(r)}
        ids = [i for i in rows if i in base_rows]
        entry = {"planned": run.get("planned"), "n_ok": len(rows), "n_paired": len(ids), "error": run.get("error"), "k": run.get("k"),
                 "spec_type": run.get("spec_type"), "by_category": {}}
        if ids:
            entry["all"] = _pair_stats([rows[i] for i in ids], [base_rows[i] for i in ids])
            for cat in sorted({rows[i]["category"] for i in ids}):
                sel = [i for i in ids if rows[i]["category"] == cat]
                entry["by_category"][cat] = _pair_stats([rows[i] for i in sel], [base_rows[i] for i in sel])
        out[name] = entry
    return out


def step_residual(stats: dict, k: int, cost: dict | None, draft_key: str = "draft_bf16") -> dict | None:
    """Observed ms per step minus the e5_cost step model (k t_draft + t_verify(k + 1)); the part no kernel time explains."""
    try:
        td = cost["drafts"][draft_key]["t_draft_ms"]
        tv = cost["target_verify_ms"][str(k + 1)] if str(k + 1) in cost["target_verify_ms"] else cost["target_verify_ms"][k + 1]
    except (KeyError, TypeError):
        return None
    obs = stats.get("ms_per_step")
    if obs is None:
        return None
    model = k * td + tv
    return {"observed_ms": obs, "model_ms": model, "residual_ms": obs - model}


# ---------------------------------------------------------------- CLI

def parse_args(argv=None):
    ap = argparse.ArgumentParser(prog="exps.e5_spec", description=__doc__.split("\n")[0])
    common.add_common_args(ap)
    ap.add_argument("--k", type=int, choices=sorted(EXP_NAMES), default=0, help="draft tokens per step, 0 = no speculation")
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--n-predict", type=int, default=192)
    ap.add_argument("--prompts", default=str(make_prompts.DEFAULT_OUT))
    ap.add_argument("--server-args", default="", help="extra llama-server flags, e.g. --server-args=--spec-draft-backend-sampling")
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
