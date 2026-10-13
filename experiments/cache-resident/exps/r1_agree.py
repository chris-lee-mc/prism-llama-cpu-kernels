"""R1a: teacher-forced agreement of Qwen3.5-0.8B draft variants with the Bonsai 2 27B PQ2_0 target.

Not a stand-alone experiment: exps.r1 calls run_agree(). For a quick run on a pod host:
  python -m exps.r1_agree [--variants bf16,q8_0,ptq1_0] [--budget-min 7]   (writes nothing but prints the JSON)

Steps: build a fixed text (build_text) -> llama-perplexity --kl-divergence-base on the target (saves its log-probs for
4 x 2048 tokens) -> llama-perplexity --kl-divergence for each draft variant against that file. "Same top p" is the share of
scored positions where the draft's top-1 token equals the target's top-1 token with both fed the same text, which is
the per-token acceptance alpha of greedy speculation when the draft prefix is right. Only the second half of every
2048 window is scored (4 x 1023 positions). The text is the model's input, not the target's own output, so the numbers
are a proxy for E5 alpha, not a replacement. Variants: bf16 and q8_0 are the ggml-org GGUFs; ptq1_0 (or any other
type) is made from the BF16 file with llama-quantize on the host. In this fork quantize_ptq1_0 ignores the imatrix and
scales each 128 group by its absolute maximum (ggml-quants.c), so post-hoc output is coarse; the log is recorded.
Vocabulary: llama-perplexity only logs "inconsistent vocabulary" and carries on, so the driver reads n_vocab from each log
and marks a variant invalid on a mismatch.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import random
import re
import subprocess
import sys
import time

from exps import common

N_CTX = 2048
CHUNKS = 4          # 4 x 2048 = 8192 tokens
TEXT_SEED = 20261013
TEXT_CHARS = 48000
WINDOW_LINES = 60
VARIANTS = ("bf16", "q8_0", "ptq1_0")
QUANT_TYPES = {"ptq1_0": "PTQ1_0", "pq2_0": "PQ2_0"}  # made on the host with llama-quantize from the BF16 file
REPO_ROOT = common.EXP_DIR.parent.parent
TEXT_FILES = ["common/speculative.cpp", "tools/server/server-context.cpp", "common/sampling.cpp", "src/llama-quant.cpp",
              "gguf-py/gguf/gguf_writer.py", "tools/server/tests/utils.py", "scripts/compare-commits.sh", "AGENTS.md", "CONTRIBUTING.md"]
PROMPTS_FILE = "experiments/cache-resident/exps/data/prompts_e5.jsonl"
NARRATION = ["I will read the relevant code first.", "Let me look at the surrounding code before I change anything.",
             "The next step is to check how this function is called.", "I need to see the rest of this file.",
             "That explains the behavior. Now I will look at the tests.", "Before editing I want to confirm the data flow.",
             "This part handles the error path, so I will read it closely.", "Now I will search for other users of this helper."]


# ---------------------------------------------------------------- text

def build_text(root=REPO_ROOT, seed: int = TEXT_SEED, chars: int = TEXT_CHARS, files=TEXT_FILES) -> tuple[str, dict]:
    """Deterministic agent-style transcript: a task, narration, and shell reads of real code windows. Same commit, same text. Files are upstream
    code, not experiments/, so that edits to the drivers do not change the text."""
    root = pathlib.Path(root)
    rng = random.Random(seed)
    srcs, meta_files = {}, []
    for rel in files:
        p = root / rel
        if p.is_file():
            raw = p.read_text(errors="replace")
            srcs[rel] = raw.splitlines()
            meta_files.append({"path": rel, "lines": len(srcs[rel]), "sha256_16": hashlib.sha256(raw.encode()).hexdigest()[:16]})
    pf = root / PROMPTS_FILE
    prompts = [json.loads(ln)["prompt"] for ln in pf.read_text().splitlines() if ln.strip()] if pf.is_file() else []
    names = sorted(srcs)
    parts, n = [], 0
    while sum(len(x) for x in parts) < chars and (names or prompts):
        n += 1
        task = rng.choice(prompts) if prompts else "Review the code and explain what it does."
        parts.append(f"### Task {n}\nUser: {task}\n")
        for _ in range(rng.randrange(1, 3)):
            parts.append("Assistant: " + rng.choice(NARRATION) + "\n")
            if not names:
                break
            rel = rng.choice(names)
            lines = srcs[rel]
            a = rng.randrange(0, max(1, len(lines) - WINDOW_LINES))
            win = [ln[:200] for ln in lines[a:a + WINDOW_LINES]]
            parts.append(f"$ sed -n '{a + 1},{a + len(win)}p' {rel}\n" + "\n".join(win) + "\n")
        parts.append("Assistant: " + rng.choice(NARRATION) + "\n\n")
    text = "".join(parts)
    meta = {"seed": seed, "n_chars": len(text), "sha256_16": hashlib.sha256(text.encode()).hexdigest()[:16], "files": meta_files,
            "prompts_file": PROMPTS_FILE if prompts else None, "n_tasks": n,
            "kind": "synthetic agent transcript (task + narration + sed reads) built from this repo at the run commit",
            "git_sha": _git_sha(root)}
    return text, meta


def _git_sha(root) -> str | None:
    try:
        r = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True, timeout=10)
        return r.stdout.strip() or None if r.returncode == 0 else None
    except (OSError, subprocess.SubprocessError):
        return None


# ---------------------------------------------------------------- commands and parsing

def perplexity_cmd(bin_dir, model: str, text_file: str, logits_file: str, kl: bool, n_ctx: int = N_CTX, chunks: int = CHUNKS) -> list[str]:
    cmd = [str(pathlib.Path(bin_dir) / "llama-perplexity"), "-m", model, "-f", text_file, "-c", str(n_ctx), "-b", str(n_ctx),
           "--chunks", str(chunks), "-ngl", "99", "-fa", "on", "--kl-divergence-base", logits_file]
    return cmd + (["--kl-divergence"] if kl else [])


def quantize_cmd(bin_dir, src: str, dst: str, qtype: str) -> list[str]:
    return [str(pathlib.Path(bin_dir) / "llama-quantize"), src, dst, qtype]


def _pair(text: str, label: str) -> tuple[float | None, float | None]:
    """'label: V +- U' (the plus-minus sign may come through as junk) -> (V, U); U None if absent."""
    m = re.search(re.escape(label) + r"[ \t]*:[ \t]*(-?[\d.]+(?:[eE][-+]?\d+)?)(?:[ \t]*\S{1,3}[ \t]*(-?[\d.]+(?:[eE][-+]?\d+)?))?", text)
    if not m:
        return None, None
    return float(m.group(1)), float(m.group(2)) if m.group(2) else None


def parse_kld(text: str) -> dict:
    """Numbers of a llama-perplexity --kl-divergence report. Top-1 agreement is a fraction (0..1)."""
    top, top_u = _pair(text, "Same top p")
    kld, kld_u = _pair(text, "Mean    KLD")
    if kld is None:
        kld, kld_u = _pair(text, "Mean KLD")
    med, _ = _pair(text, "Median  KLD")
    p99, _ = _pair(text, "99.0%   KLD")
    ppl_q, _ = _pair(text, "Mean PPL(Q)")
    ppl_b, _ = _pair(text, "Mean PPL(base)")
    nv = [int(x) for x in re.findall(r"n_vocab\s*=\s*(\d+)", text)]
    mm = re.search(r"inconsistent vocabulary \((\d+) vs (\d+)\)", text)
    return {"same_top": None if top is None else top / 100.0, "same_top_unc": None if top_u is None else top_u / 100.0,
            "mean_kld": kld, "mean_kld_unc": kld_u, "median_kld": med, "kld_p99": p99, "ppl_q": ppl_q, "ppl_base": ppl_b,
            "n_vocab": nv[0] if nv else None, "vocab_mismatch": (int(mm.group(1)), int(mm.group(2))) if mm else None}


def parse_quantize(text: str) -> dict:
    """Tensor types chosen by llama-quantize (from the 'converting to X' lines), fallbacks and sizes."""
    types: dict[str, int] = {}
    for t in re.findall(r"converting to (\w+)", text):
        types[t] = types.get(t, 0) + 1
    falls = re.findall(r"falling back to\s+(\w+)", text)
    src = re.search(r"model size\s*=\s*([\d.]+)\s*MiB", text)
    dst = re.search(r"quant size\s*=\s*([\d.]+)\s*MiB", text)
    return {"tensor_types": types, "n_fallback": len(falls), "src_mib": float(src.group(1)) if src else None,
            "dst_mib": float(dst.group(1)) if dst else None, "needs_imatrix": bool(re.search(r"requires? (an )?imatrix|imatrix.*required", text, re.I))}


def run_logged(cmd: list[str], timeout: float, log_path: str | None = None) -> tuple[int | None, str, float]:
    """Run cmd, return (exit code or None on timeout, output text, seconds). Output is also written to log_path."""
    t0 = time.monotonic()
    try:
        p = subprocess.run(cmd, capture_output=True, timeout=max(1.0, timeout))
        out, code = (p.stdout or b"") + (p.stderr or b""), p.returncode
    except subprocess.TimeoutExpired as e:
        out, code = (e.stdout or b"") + (e.stderr or b""), None
    except OSError as e:
        out, code = str(e).encode(), 127
    text = out.decode("utf-8", errors="replace")
    if log_path:
        try:
            pathlib.Path(log_path).write_text(text.encode("ascii", "replace").decode("ascii")[-400000:], encoding="ascii")
        except OSError:
            pass
    return code, text, time.monotonic() - t0


# ---------------------------------------------------------------- run

def summarize(variants: dict) -> dict:
    """Go/no-go inputs for criterion 2: BF16 agreement >= 0.5 and PTQ1_0 within 0.05 of it. None when a number is missing."""
    bf = (variants.get("bf16") or {}).get("same_top")
    pt = (variants.get("ptq1_0") or {}).get("same_top")
    q8 = (variants.get("q8_0") or {}).get("same_top")
    delta = pt - bf if (pt is not None and bf is not None) else None
    return {"bf16_agreement": bf, "q8_0_agreement": q8, "ptq1_0_agreement": pt,
            "bf16_ge_0p5": None if bf is None else bf >= 0.5,
            "ptq1_delta_vs_bf16": delta, "ptq1_within_0p05": None if delta is None else delta >= -0.05,
            "note": "agreement is teacher-forced top-1 on the fixed text; it stands in for per-token alpha"}


def run_agree(*, bin_dir, target: str, drafts: dict, workdir, budget: common.Budget, notes: list, variants=VARIANTS,
              root=REPO_ROOT, log_dir=None, on_update=None) -> dict:
    """drafts: {"bf16": path, "q8_0": path}. workdir holds the text, the ~2 GB log-prob file and quantized GGUFs (deleted by the
    caller). Returns {"text", "base", "variants", "summary", "errors"}; never raises, partial results stay in the dict."""
    wd = pathlib.Path(workdir)
    out: dict = {"text": None, "base": None, "variants": {}, "summary": {}, "errors": [], "chunks": CHUNKS, "n_ctx": N_CTX}

    def update() -> None:
        out["summary"] = summarize(out["variants"])
        if on_update:
            on_update(out)

    def logp(name: str) -> str | None:
        return str(pathlib.Path(log_dir) / f"r1a-{name}.log") if log_dir else None

    text, out["text"] = build_text(root)
    text_file = wd / "agree_text.txt"
    text_file.write_text(text)
    logits = str(wd / "target_logprobs.bin")
    if budget.expired(60):
        out["errors"].append("time budget used up before the target pass")
        return out
    code, log, secs = run_logged(perplexity_cmd(bin_dir, target, str(text_file), logits, kl=False), budget.timeout(600, reserve_s=30),
                                 logp("base"))
    nv = parse_kld(log)["n_vocab"]
    out["base"] = {"exit": code, "seconds": secs, "n_vocab": nv, "logits_bytes": os.path.getsize(logits) if os.path.exists(logits) else None}
    if code != 0 or not os.path.exists(logits):
        out["errors"].append(f"target pass failed (exit {code}): {log[-400:].strip()}")
        update()
        return out
    update()
    for v in variants:
        if budget.expired(45):
            out["errors"].append(f"time budget used up; {v} and later skipped")
            break
        rec: dict = {"name": v}
        out["variants"][v] = rec
        path = drafts.get(v)
        if v in QUANT_TYPES:
            src = drafts.get("bf16")
            if not src:
                rec["error"] = "needs the bf16 GGUF as the quantize source"
                continue
            path = str(wd / f"draft-{v}.gguf")
            code, qlog, qsecs = run_logged(quantize_cmd(bin_dir, src, path, QUANT_TYPES[v]), budget.timeout(300, reserve_s=45), logp(f"quantize-{v}"))
            rec["quantize"] = dict(parse_quantize(qlog), exit=code, seconds=qsecs, tail=qlog[-300:].strip() if code != 0 else None)
            if code != 0 or not os.path.exists(path):
                rec["error"] = f"llama-quantize {QUANT_TYPES[v]} failed (exit {code})"
                notes.append(f"{v}: {rec['error']}: {qlog[-200:].strip()}")
                update()
                continue
            rec["size_bytes"] = os.path.getsize(path)
        if not path:
            rec["error"] = "no GGUF path for this variant"
            continue
        code, log, secs = run_logged(perplexity_cmd(bin_dir, path, str(text_file), logits, kl=True), budget.timeout(300, reserve_s=15), logp(v))
        rec.update(parse_kld(log), exit=code, seconds=secs)
        if code != 0 or rec["same_top"] is None:
            rec["error"] = f"kl pass failed (exit {code}): {log[-300:].strip()}"
        elif rec["vocab_mismatch"] or (nv and rec["n_vocab"] and rec["n_vocab"] != nv):
            rec["error"] = f"vocabulary mismatch (target {nv}, draft {rec['n_vocab']}, tool {rec['vocab_mismatch']}): numbers invalid"
            for k in ("same_top", "same_top_unc", "mean_kld", "mean_kld_unc"):
                rec[k] = None
        if rec.get("error"):
            notes.append(f"{v}: {rec['error']}")
        if v in QUANT_TYPES:
            try:
                os.remove(path)
            except OSError:
                pass
        update()
    try:
        os.remove(logits)
    except OSError:
        pass
    return out


def main(argv=None) -> int:
    import tempfile
    from exps import e4_pdl
    ap = argparse.ArgumentParser(prog="exps.r1_agree", description=__doc__.split("\n")[0])
    ap.add_argument("--variants", default=",".join(VARIANTS))
    ap.add_argument("--budget-min", type=float, default=7.0)
    ap.add_argument("--print-text", action="store_true", help="print the fixed text and its metadata, run nothing")
    a = ap.parse_args(argv)
    if a.print_text:
        text, meta = build_text()
        print(json.dumps(meta, indent=1))
        print(text)
        return 0
    from exps import e4_bonsai
    notes: list = []
    bin_dir = e4_bonsai.llama_bin()
    target, _ = e4_pdl.model_path("pq2", notes)
    drafts = {"bf16": e4_pdl.model_path("draft_bf16", notes)[0], "q8_0": e4_pdl.model_path("draft_q8_0", notes)[0]}
    with tempfile.TemporaryDirectory(prefix="cr-r1a-") as wd:
        res = run_agree(bin_dir=bin_dir, target=target, drafts=drafts, workdir=wd, budget=common.Budget(a.budget_min), notes=notes,
                        variants=[v for v in a.variants.split(",") if v])
    res["text"].pop("files", None)
    print(json.dumps({"agree": res, "notes": notes}, indent=1, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
