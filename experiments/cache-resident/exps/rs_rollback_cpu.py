"""Offline (CPU) check of the draft-context rollback and of the wrap_b switches, on tiny random models.

  CR_LLAMA_BIN=<cpu build>/bin python -m exps.rs_rollback_cpu

Needs a llama.cpp build of this branch (any backend; libllama and llama-server in CR_LLAMA_BIN), numpy and gguf-py of the repo. No GPU.
Models: a 4-layer qwen35 (3 GDN layers, 1 attention layer, the tokenizer of models/ggml-vocab-qwen35.gguf) with seeded random weights
of std 0.25; only token rows whose text is printable ASCII get an embedding (the others are zero), so outputs stay valid UTF-8. The
near draft has the same draws at std 0.27 (the same weights x 1.08): close to the target, not equal. The flat model (std 0.02) repeats
itself and the mid model (std 0.05) repeats with breaks: n-gram drafts that are accepted fully (flat) and partly (mid). The same
shape as an attention-only llama model (l, lnear: std 0.25 / 0.27) gives a draft context where seq_rm is exact (type PART).
p  cr_rs_rollback_probe on the probe model (the target weights without the ASCII-only rows): logits after a seq_rm back over one-token
   decodes or one batch, against a context that decoded only the kept tokens: n_rs_seq 4 (both modes), 16 and 64 (batches), the runs of
   wrap_b.PROBE_RUNS (n_rs_seq + 1 extra tokens: the rollback == n_rs_seq row of a full rejection is the boundary).
s  llama-server, greedy, 3 short prompts of 96 tokens, -c 512 (LLAMA_TRACE=1: the per-step accepted a/b lines give the rollback counts):
   hybrid target, draft-simple K=4 (k4), with GGML_EXP_DRAFT_CKPT=1 (k4_ckpt, the correct draft state), with GGML_EXP_SKIP_DRAFT_REEVAL=1
   (k4_skip: must be ignored, a recurrent draft inherits the stale rollback), with GGML_EXP_DRAFT_TIME=1 on top of the checkpoint
   (k4_ckpt_t); the target itself as draft (same: a correct draft state accepts every token, so no rollback happens) and the near
   draft (partial acceptance: this is the case that rolls back). Attention-only target and draft (llama arch): k4_l, and k4_l_skip with
   the skip switch, which must give the same drafted / accepted counts as k4_l (seq_rm is exact) and skipped batches. ngram-mod
   (n-match 4, n-min 2, n-max 16) with and without GGML_EXP_NGRAM_RS=8 on the flat model (full acceptance) and on the mid model
   (partial acceptance: the in-batch rollback of the server runs, partial steps are counted). Reported: drafted and accepted
   tokens, partial and restore steps, greedy text equality with the model alone (informational: batch numerics), the experiment
   counters of the statistics lines, the rollback path of the log.
Result: results/rs_rollback_cpu/<gpu>/<stamp>.json (gpu tag cpu off a GPU host). The build directory is stored by its name only.
"""
from __future__ import annotations

import argparse
import os
import pathlib
import shutil
import sys
import tempfile
import time

from exps import common, e4_bonsai, e5_spec, r2a_longctx as r2a, r2b_spec as sp, wrap_b

EXP = "rs_rollback_cpu"
REPO_ROOT = common.EXP_DIR.parent.parent
VOCAB = REPO_ROOT / "models" / "ggml-vocab-qwen35.gguf"
STD_TARGET, STD_NEAR, STD_FLAT, STD_MID = 0.25, 0.27, 0.02, 0.05
PROMPTS = ["def add(a, b):\n    return a + b\n" * 3, "The quick brown fox jumps over the lazy dog. " * 4, "hello world"]
N_PREDICT = 96
DS = ["--spec-type", "draft-simple", "--spec-draft-n-max", "4"]
NG = ["--spec-type", "ngram-mod", "--spec-ngram-mod-n-match", "4", "--spec-ngram-mod-n-min", "2", "--spec-ngram-mod-n-max", "16"]
SKIP, CKPT, TIME, NGRAM_RS = wrap_b.SKIP, wrap_b.CKPT, wrap_b.TIME, wrap_b.NGRAM_RS
# tiny qwen35 shape (as tests/gen-tiny-qwen35.py of ba62b7023)
N_LAYER, N_EMBD, N_FF, N_HEAD, N_HEAD_KV, N_CTX = 4, 64, 128, 2, 1, 512
D_CONV, D_STATE, N_V_HEADS, N_K_HEADS = 4, 16, 4, 2


def write_tiny(out_path: str, std: float, seed: int = 1234, ascii_only: bool = True, arch: str = "qwen35") -> int:
    """Random-weight GGUF (all f32), arch qwen35 (hybrid GDN, recurrent state) or llama (attention only: seq_rm is exact). Returns n_vocab.
    ascii_only: rows of tokens that are not printable ASCII (or space, newline) are zero (the server needs valid UTF-8; the probe needs
    every row, its token ids are random)."""
    import numpy as np
    sys.path.insert(0, str(REPO_ROOT / "gguf-py"))
    from gguf import GGUFReader, GGUFValueType, GGUFWriter
    value_dim, key_dim = D_STATE * N_V_HEADS, D_STATE * N_K_HEADS
    conv_dim, head_k = 2 * key_dim + value_dim, N_EMBD // N_HEAD
    rng = np.random.default_rng(seed)

    def rand(*shape):
        return rng.normal(0.0, std, size=shape).astype(np.float32)

    w = GGUFWriter(out_path, arch)
    w.add_name(f"tiny-random-{arch}")
    w.add_file_type(0)
    w.add_block_count(N_LAYER)
    w.add_context_length(N_CTX)
    w.add_embedding_length(N_EMBD)
    w.add_feed_forward_length(N_FF)
    w.add_head_count(N_HEAD)
    w.add_head_count_kv(N_HEAD_KV)
    w.add_layer_norm_rms_eps(1e-6)
    w.add_rope_freq_base(10000.0)
    if arch == "qwen35":
        w.add_rope_dimension_sections([head_k // 4, head_k // 8, head_k // 8, 0])
        w.add_ssm_conv_kernel(D_CONV)
        w.add_ssm_inner_size(value_dim)
        w.add_ssm_state_size(D_STATE)
        w.add_ssm_time_step_rank(N_V_HEADS)
        w.add_ssm_group_count(N_K_HEADS)
        w.add_key_value("qwen35.full_attention_interval", N_LAYER, GGUFValueType.UINT32)
    r = GGUFReader(VOCAB)
    toks: list = []
    for name, field in r.fields.items():
        if not name.startswith("tokenizer."):
            continue
        vtype = field.types[0]
        if vtype == GGUFValueType.ARRAY:
            w.add_key_value(name, field.contents(), vtype, sub_type=field.types[-1])
        else:
            w.add_key_value(name, field.contents(), vtype)
        if name == "tokenizer.ggml.tokens":
            toks = list(field.contents())
    emb = rand(len(toks), N_EMBD)
    ok = {chr(c) for c in range(0x21, 0x7F)} | {chr(0x120), chr(0x10A)}        # byte-level BPE: 0x120 is a space, 0x10A a newline
    if ascii_only:
        emb[[i for i, t in enumerate(toks) if not t or any(ch not in ok for ch in t)]] = 0.0
    w.add_tensor("token_embd.weight", emb)
    w.add_tensor("output_norm.weight", np.ones(N_EMBD, dtype=np.float32))
    for il in range(N_LAYER):
        w.add_tensor(f"blk.{il}.attn_norm.weight", np.ones(N_EMBD, dtype=np.float32))
        w.add_tensor(f"blk.{il}.{'ffn_norm' if arch == 'llama' else 'post_attention_norm'}.weight", np.ones(N_EMBD, dtype=np.float32))
        if arch == "llama":
            w.add_tensor(f"blk.{il}.attn_q.weight", rand(head_k * N_HEAD, N_EMBD))
            w.add_tensor(f"blk.{il}.attn_k.weight", rand(head_k * N_HEAD_KV, N_EMBD))
            w.add_tensor(f"blk.{il}.attn_v.weight", rand(head_k * N_HEAD_KV, N_EMBD))
            w.add_tensor(f"blk.{il}.attn_output.weight", rand(N_EMBD, head_k * N_HEAD))
        elif (il + 1) % N_LAYER != 0:
            w.add_tensor(f"blk.{il}.attn_qkv.weight", rand(conv_dim, N_EMBD))
            w.add_tensor(f"blk.{il}.attn_gate.weight", rand(value_dim, N_EMBD))
            w.add_tensor(f"blk.{il}.ssm_conv1d.weight", rand(conv_dim, D_CONV))
            w.add_tensor(f"blk.{il}.ssm_dt.bias", rand(N_V_HEADS))
            w.add_tensor(f"blk.{il}.ssm_a", rng.uniform(-0.6, -0.2, size=N_V_HEADS).astype(np.float32))
            w.add_tensor(f"blk.{il}.ssm_beta.weight", rand(N_V_HEADS, N_EMBD))
            w.add_tensor(f"blk.{il}.ssm_alpha.weight", rand(N_V_HEADS, N_EMBD))
            w.add_tensor(f"blk.{il}.ssm_norm.weight", np.ones(D_STATE, dtype=np.float32))
            w.add_tensor(f"blk.{il}.ssm_out.weight", rand(N_EMBD, value_dim))
        else:
            w.add_tensor(f"blk.{il}.attn_q.weight", rand(head_k * N_HEAD * 2, N_EMBD))
            w.add_tensor(f"blk.{il}.attn_k.weight", rand(head_k * N_HEAD_KV, N_EMBD))
            w.add_tensor(f"blk.{il}.attn_v.weight", rand(head_k * N_HEAD_KV, N_EMBD))
            w.add_tensor(f"blk.{il}.attn_output.weight", rand(N_EMBD, head_k * N_HEAD))
            w.add_tensor(f"blk.{il}.attn_q_norm.weight", np.ones(head_k, dtype=np.float32))
            w.add_tensor(f"blk.{il}.attn_k_norm.weight", np.ones(head_k, dtype=np.float32))
        w.add_tensor(f"blk.{il}.ffn_gate.weight", rand(N_FF, N_EMBD))
        w.add_tensor(f"blk.{il}.ffn_down.weight", rand(N_EMBD, N_FF))
        w.add_tensor(f"blk.{il}.ffn_up.weight", rand(N_FF, N_EMBD))
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    return len(toks)


def server_configs() -> list[dict]:
    """(name, model, draft model key, extra argv, env, base) of part s. Models: target, flat, mid (hybrid qwen35), l (attention only);
    drafts: same, near, lnear (keys of the files dict)."""
    out = []

    def add(name, model, draft, argv, env, base):
        out.append({"name": name, "model": model, "draft": draft, "argv": argv, "env": env, "base": base})

    add("base", "target", None, [], {}, "base")
    for label in ("same", "near"):
        d = ["-md", label] + DS
        add(f"k4_{label}", "target", label, d, {}, "base")
        add(f"k4_ckpt_{label}", "target", label, d, {CKPT: "1"}, "base")
    add("k4_skip_near", "target", "near", ["-md", "near"] + DS, {SKIP: "1"}, "base")
    add("k4_ckpt_t_near", "target", "near", ["-md", "near"] + DS, {CKPT: "1", TIME: "1"}, "base")
    add("base_l", "l", None, [], {}, "base_l")
    for label, key in (("same", "l"), ("near", "lnear")):
        add(f"k4_l_{label}", "l", key, ["-md", key] + DS, {}, "base_l")
        add(f"k4_l_skip_{label}", "l", key, ["-md", key] + DS, {SKIP: "1"}, "base_l")
    add("k4_l_skip_t_near", "l", "lnear", ["-md", "lnear"] + DS, {SKIP: "1", TIME: "1"}, "base_l")
    for m in ("flat", "mid"):
        add(f"base_{m}", m, None, [], {}, f"base_{m}")
        add(f"ngram_{m}", m, None, NG, {}, f"base_{m}")
        add(f"ngram_rs8_{m}", m, None, NG, {NGRAM_RS: "8"}, f"base_{m}")
    return out


def run_server(bin_dir, files: dict, cfg: dict, log_path: str, threads: int) -> dict:
    cmd = [str(pathlib.Path(bin_dir) / "llama-server"), "-m", files[cfg["model"]], "-c", "512", "-np", "1", "--host", "127.0.0.1", "--port",
           str(e5_spec.PORT), "-lv", "4", "-t", str(threads), "--reasoning-format", "none"] + [files.get(a, a) for a in cfg["argv"]]
    env = dict(cfg["env"], LLAMA_TRACE="1")
    res: dict = {"name": cfg["name"], "model": cfg["model"], "draft": cfg["draft"], "env": cfg["env"], "argv": cfg["argv"], "rows": [], "error": None}
    srv = e5_spec.Server(cmd, log_path, env)
    t0 = time.monotonic()
    try:
        ready, err = srv.wait_ready(300.0)
        if not ready:
            res["error"] = err
        else:
            for pr in PROMPTS:
                r = e5_spec.post_json(f"http://127.0.0.1:{e5_spec.PORT}/completion", e5_spec.completion_body(pr, N_PREDICT), 300.0)
                res["rows"].append(e5_spec.parse_completion(r, keep_text=True))
    except Exception as e:
        res["error"] = f"{type(e).__name__}: {e}"
    finally:
        srv.stop()
    res["wall_s"] = time.monotonic() - t0
    try:
        text = pathlib.Path(log_path).read_text(errors="replace")
    except OSError:
        text = ""
    res["stats"] = {k: v[-1] for k, v in sp.parse_stats(text).items() if v}       # cumulative over all requests of the server
    res["exp_switches"] = sorted({m.group(1) for ln in text.splitlines() if (m := sp.EXP_MARK_RE.search(ln))})
    res["ignored"] = any("ignored" in ln and "GGML_EXP_" in ln for ln in text.splitlines())
    res["drafted"] = sum(r.get("draft_n", 0) for r in res["rows"])
    res["accepted"] = sum(r.get("draft_n_accepted", 0) for r in res["rows"])
    res["acceptance"] = res["accepted"] / res["drafted"] if res["drafted"] else None
    steps = [(int(m.group(1)), int(m.group(2)), bool(m.group(3))) for ln in text.splitlines() for m in [r2a.ACCEPT_TRACE_RE.search(ln)] if m]
    res["steps"] = {"verified": len(steps), "partial": sum(1 for a, b, _ in steps if a < b), "restored": sum(1 for _, _, r in steps if r),
                    "rolled_back_tokens": sum(b - a for a, b, _ in steps)}
    ck = r2a.log_checks(text, "none" if not cfg["argv"] else "draft-simple" if "-md" in cfg["argv"] else "ngram-mod")
    res["path"] = {k: ck.get(k) for k in ("rollback_path", "n_rs_seq", "seq_rm_types", "spec_uses_checkpoints")}
    if cfg["model"] == "l":         # attention only: seq_rm is exact (type PART), the log has neither the RS nor the FULL line
        res["path"]["rollback_path"] = "part" if not res["path"]["spec_uses_checkpoints"] else res["path"]["rollback_path"]
    return res


def run(args) -> int:
    notes: list[str] = []
    args.gpu = common.gpu_tag(args)
    bin_dir = e4_bonsai.llama_bin()
    config = {"build": pathlib.Path(bin_dir).parent.name, "std_target": STD_TARGET, "std_near": STD_NEAR, "std_mid": STD_MID,
              "prompts": PROMPTS, "n_predict": N_PREDICT, "threads": args.threads, "parts": args.parts}
    doc: dict = {"models": {}, "probe": None, "server": {}}
    work = pathlib.Path(tempfile.mkdtemp(prefix="rs-cpu-"))
    try:
        specs = (("target", STD_TARGET, True, "qwen35"), ("near", STD_NEAR, True, "qwen35"), ("flat", STD_FLAT, True, "qwen35"),
                 ("mid", STD_MID, True, "qwen35"), ("probe", STD_TARGET, False, "qwen35"), ("l", STD_TARGET, True, "llama"),
                 ("lnear", STD_NEAR, True, "llama"))
        files = {n: str(work / f"tiny-{n}.gguf") for n, _, _, _ in specs}
        files["same"] = files["target"]
        doc["models"] = {n: {"std": std, "ascii_only": asc, "arch": arch, "n_vocab": write_tiny(files[n], std, ascii_only=asc, arch=arch)}
                         for n, std, asc, arch in specs}
        if "p" in args.parts:
            exe = wrap_b.build_probe(bin_dir, notes)
            if exe:
                doc["probe"] = {"runs": {}}
                for label, _, argv in wrap_b.PROBE_RUNS:
                    res = wrap_b.run_probe(exe, files["probe"], 0, 900.0, argv)
                    res["verdict"] = v = wrap_b.probe_verdict(res)
                    res["cmd"] = [pathlib.Path(c).name if os.sep in str(c) else c for c in res.get("cmd", [])]
                    doc["probe"]["runs"][label] = res
                    notes.append(wrap_b.probe_notes(label, "probe", res))
        if "s" in args.parts and (bin_dir / "llama-server").exists():
            from infra.results import artifact_path
            cfgs = server_configs()
            for cfg in cfgs:
                log = str(artifact_path(EXP, f"server-{cfg['name']}.log", args.gpu))
                res = run_server(bin_dir, files, cfg, log, args.threads)
                res["argv"] = [pathlib.Path(a).name if a.endswith(".gguf") else a for a in res["argv"]]
                doc["server"][cfg["name"]] = res
                print(f"[{EXP}] {cfg['name']}: drafted {res['drafted']} accepted {res['accepted']} partial steps {res['steps']['partial']}", flush=True)
            bases = {c["name"]: c["base"] for c in cfgs}
            texts = {n: [r.get("text") for r in res["rows"]] for n, res in doc["server"].items()}
            for name, res in doc["server"].items():
                pairs = list(zip(texts[name], texts.get(bases[name], [])))
                res["base"] = bases[name]
                res["text_equal_base"] = sum(1 for a, b in pairs if a is not None and a == b)
                res["text_compared"] = len(pairs)
                for r in res["rows"]:
                    r.pop("text", None)
        elif "s" in args.parts:
            notes.append(f"llama-server not found in {bin_dir}; part s skipped")
    finally:
        shutil.rmtree(work, ignore_errors=True)
    notes.extend(summary_notes(doc["server"]))
    common.write_result(EXP, args, config, [], notes, False, extra=doc)
    return 0


def _acc(s: dict, name: str) -> str:
    v = (s.get(name) or {}).get("acceptance")
    return "n/a" if v is None else "%.3f" % v


def summary_notes(s: dict) -> list[str]:
    """One line per question the run answers. 'same' drafts never roll back (every draft is accepted), so they say nothing about the
    rollback; the near and mid rows do."""
    out = []
    for n in ("same", "near"):
        names = [f"k4_{n}", f"k4_ckpt_{n}"] + ([f"k4_skip_{n}"] if n == "near" else []) + (["k4_ckpt_t_near"] if n == "near" else [])
        if any(x in s for x in names):
            out.append(f"hybrid draft {n}: acceptance " + ", ".join(f"{x} {_acc(s, x)}" for x in names if x in s) +
                       ("; stale rollback = k4 below k4_ckpt" if n == "near" else "; no rollback happens (full acceptance)"))
    if "k4_skip_near" in s:
        out.append("hybrid draft, skip switch ignored: %s (acceptance equals k4_near: %s)" % (
            s["k4_skip_near"].get("ignored"), _acc(s, "k4_skip_near") == _acc(s, "k4_near")))
    for n in ("same", "near"):
        if f"k4_l_{n}" in s and f"k4_l_skip_{n}" in s:
            sk = (s[f"k4_l_skip_{n}"].get("stats") or {}).get("draft-simple", {})
            out.append("attention-only draft %s: acceptance k4_l %s, k4_l_skip %s (exact seq_rm: equal), skipped batches %s, catch-up tokens %s, "
                       "rolled-back tokens %s" % (n, _acc(s, f"k4_l_{n}"), _acc(s, f"k4_l_skip_{n}"), sk.get("skipped_batches"), sk.get("catchup_tokens"),
                                                  s[f"k4_l_skip_{n}"]["steps"]["rolled_back_tokens"]))
    t = ((s.get("k4_ckpt_t_near") or {}).get("stats") or {}).get("draft-simple", {})
    if "dur_p_ms" in t:
        out.append("draft-time (hybrid ckpt): dur_p %.3f ms in %s calls (%.3f ms per call), prefill %.3f ms in %s calls" % (
            t["dur_p_ms"], t["calls_p"], t["dur_p_ms"] / max(t["calls_p"], 1), t["dur_pf_ms"], t["calls_pf"]))
    for m in ("flat", "mid"):
        if f"ngram_{m}" in s:
            for n in (f"ngram_{m}", f"ngram_rs8_{m}"):
                if n in s:
                    st = s[n]["steps"]
                    out.append(f"{n}: drafted {s[n]['drafted']}, accepted {s[n]['accepted']}, partial steps {st['partial']} of {st['verified']}, "
                               f"rolled-back tokens {st['rolled_back_tokens']}, path {s[n]['path'].get('rollback_path')}, "
                               f"text equal {s[n].get('text_equal_base')}/{s[n].get('text_compared')} (informational)")
    return out


def parse_args(argv=None):
    ap = argparse.ArgumentParser(prog="exps.rs_rollback_cpu", description=__doc__.split("\n")[0])
    common.add_common_args(ap)
    ap.add_argument("--parts", type=lambda t: t.replace(",", ""), default="ps", help="p (rollback probe), s (server)")
    ap.add_argument("--threads", type=int, default=4)
    return ap.parse_args(argv)


def main(argv=None) -> int:
    a = parse_args(argv)
    if a.dry_run:
        common.print_plan({"exp": EXP, "parts": a.parts, "configs": [{k: c[k] for k in ("name", "model", "draft", "env", "base")} for c in server_configs()],
                           "probe_runs": [{"label": l, "models": list(m), "args": a_} for l, m, a_ in wrap_b.PROBE_RUNS], "prompts": len(PROMPTS),
                           "n_predict": N_PREDICT, "quick": a.quick})
        return 0
    return run(a)


if __name__ == "__main__":
    sys.exit(main())
