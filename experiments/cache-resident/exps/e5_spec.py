"""E5: speculative decoding baseline. Bonsai 2 27B PQ2_0 target + Qwen3.5-0.8B BF16 draft via llama-server.

  python -m exps.e5_spec --k 0|4|6|8 [--limit 200] [--n-predict 192] [--prompts exps/data/prompts_e5.jsonl]

k=0 is the plain target. k>0 adds `-md DRAFT -ngld 99 --spec-type draft-simple --spec-draft-n-max K`
(flags from common/arg.cpp of this fork). Prompts go through /apply-template (thinking disabled when the
template supports it), then /completion with greedy sampling. If the server rejects the draft (vocab
mismatch; llama.cpp allows a small vocab size difference) the run is recorded as draft_incompatible and
repeated with `--spec-type ngram-simple`, clearly labelled.
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

from exps import common, hf_ids, make_prompts

PORT = 8081
CTX = 16384
EXP_NAMES = {0: "e5_base", 4: "e5_k4", 6: "e5_k6", 8: "e5_k8"}  # result dir must equal the manifest key


# ---------------------------------------------------------------- pure helpers

def llama_bin() -> pathlib.Path:
    return pathlib.Path(os.environ.get("CR_LLAMA_BIN") or (common.EXP_DIR.parent.parent / "build-cuda" / "bin"))


def server_cmd(bin_dir, model: str, k: int, draft: str | None = None, port: int = PORT, ctx: int = CTX,
               spec_type: str = "draft-simple") -> list[str]:
    cmd = [str(pathlib.Path(bin_dir) / "llama-server"), "-m", model, "-ngl", "99", "-fa", "on", "-c", str(ctx), "-np", "1",
           "--host", "127.0.0.1", "--port", str(port)]
    if k > 0:
        if spec_type == "draft-simple":
            if not draft:
                raise ValueError("draft model path required for draft-simple")
            cmd += ["-md", draft, "-ngld", "99"]
        cmd += ["--spec-type", spec_type, "--spec-draft-n-max", str(k)]
    return cmd


def completion_body(prompt: str, n_predict: int) -> dict:
    return {"prompt": prompt, "n_predict": n_predict, "temperature": 0, "top_k": 1, "seed": 42, "cache_prompt": False}


def template_body(prompt: str, thinking: bool | None) -> dict:
    body: dict = {"messages": [{"role": "user", "content": prompt}]}
    if thinking is False:
        body["chat_template_kwargs"] = {"enable_thinking": False}
    return body


def parse_completion(resp: dict) -> dict:
    """Pull the per-prompt numbers out of a /completion response."""
    t = resp.get("timings") or {}
    return {"tokens_predicted": int(resp.get("tokens_predicted", t.get("predicted_n", 0)) or 0),
            "predicted_per_second": t.get("predicted_per_second"), "predicted_ms": t.get("predicted_ms"),
            "prompt_n": t.get("prompt_n"), "prompt_ms": t.get("prompt_ms"),
            "draft_n": int(t.get("draft_n", 0) or 0), "draft_n_accepted": int(t.get("draft_n_accepted", 0) or 0),
            "stop_type": resp.get("stop_type"), "truncated": resp.get("truncated")}


def summarize(rows: list[dict], k: int, wall_s: float) -> dict:
    ok = [r for r in rows if r.get("predicted_per_second")]
    drafted = sum(r.get("draft_n", 0) for r in rows)
    accepted = sum(r.get("draft_n_accepted", 0) for r in rows)
    ms = sum(r.get("predicted_ms") or 0 for r in ok)
    toks = sum(r.get("tokens_predicted", 0) for r in ok)
    speeds = [r["predicted_per_second"] for r in ok]
    return {"k": k, "n_prompts": len(rows), "n_ok": len(ok), "acceptance": accepted / drafted if drafted else None,
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
                on_row=None):
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
            row.update(parse_completion(resp))
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
