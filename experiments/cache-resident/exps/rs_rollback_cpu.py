"""Offline (CPU) check of the draft-context rollback and of the wrap_b switches, on tiny random qwen35 (hybrid GDN) models.

  CR_LLAMA_BIN=<cpu build>/bin python -m exps.rs_rollback_cpu

Needs a llama.cpp build of this branch (any backend; libllama and llama-server in CR_LLAMA_BIN), numpy and gguf-py of the repo. No GPU.
Models: a 4-layer qwen35 (3 GDN layers, 1 attention layer, the tokenizer of models/ggml-vocab-qwen35.gguf) with seeded random weights
of std 0.25; only token rows whose text is printable ASCII get an embedding (the others are zero), so outputs stay valid UTF-8. The
near draft has the same draws at std 0.27 (the same weights x 1.08): close to the target, not equal. The flat model (std 0.02) repeats
itself, which gives the n-gram type drafts.
p  cr_rs_rollback_probe (n_rs_seq 4) on the probe model (the target weights without the ASCII-only rows): logits after a seq_rm back
   over one-token decodes or one batch, against a context that decoded only the kept tokens.
s  llama-server, greedy, 3 short prompts of 96 tokens, -c 512: target alone, draft-simple K=4 (k4), k4 with GGML_EXP_SKIP_DRAFT_REEVAL=1
   (k4_skip) and with GGML_EXP_DRAFT_CKPT=1 (k4_ckpt), with the target itself as draft (same: a correct draft state accepts every
   token) and with the near draft; on the flat model ngram-mod (n-match 4, n-min 2, n-max 16) with and without GGML_EXP_NGRAM_RS=8
   (and the flat model alone as their base). Reported: drafted
   and accepted tokens, greedy text equality with target alone, the experiment counters of the statistics lines.
Result: results/rs_rollback_cpu/<gpu>/<stamp>.json (gpu tag cpu off a GPU host).
"""
from __future__ import annotations

import argparse
import os
import pathlib
import shutil
import sys
import tempfile
import time

from exps import common, e4_bonsai, e5_spec, r2b_spec as sp, wrap_b

EXP = "rs_rollback_cpu"
REPO_ROOT = common.EXP_DIR.parent.parent
VOCAB = REPO_ROOT / "models" / "ggml-vocab-qwen35.gguf"
STD_TARGET, STD_NEAR, STD_FLAT = 0.25, 0.27, 0.02
PROMPTS = ["def add(a, b):\n    return a + b\n" * 3, "The quick brown fox jumps over the lazy dog. " * 4, "hello world"]
N_PREDICT = 96
DS = ["--spec-type", "draft-simple", "--spec-draft-n-max", "4"]
NG = ["--spec-type", "ngram-mod", "--spec-ngram-mod-n-match", "4", "--spec-ngram-mod-n-min", "2", "--spec-ngram-mod-n-max", "16"]
SKIP, CKPT, NGRAM_RS = wrap_b.SKIP, wrap_b.CKPT, wrap_b.NGRAM_RS
# tiny qwen35 shape (as tests/gen-tiny-qwen35.py of ba62b7023)
N_LAYER, N_EMBD, N_FF, N_HEAD, N_HEAD_KV, N_CTX = 4, 64, 128, 2, 1, 512
D_CONV, D_STATE, N_V_HEADS, N_K_HEADS = 4, 16, 4, 2


def write_tiny(out_path: str, std: float, seed: int = 1234, ascii_only: bool = True) -> int:
    """Random-weight qwen35 GGUF (all f32). Returns n_vocab. ascii_only: rows of tokens that are not printable ASCII (or space, newline)
    are zero (the server needs valid UTF-8; the probe needs every row, its token ids are random)."""
    import numpy as np
    sys.path.insert(0, str(REPO_ROOT / "gguf-py"))
    from gguf import GGUFReader, GGUFValueType, GGUFWriter
    value_dim, key_dim = D_STATE * N_V_HEADS, D_STATE * N_K_HEADS
    conv_dim, head_k = 2 * key_dim + value_dim, N_EMBD // N_HEAD
    rng = np.random.default_rng(seed)

    def rand(*shape):
        return rng.normal(0.0, std, size=shape).astype(np.float32)

    w = GGUFWriter(out_path, "qwen35")
    w.add_name("tiny-random-qwen35")
    w.add_file_type(0)
    w.add_block_count(N_LAYER)
    w.add_context_length(N_CTX)
    w.add_embedding_length(N_EMBD)
    w.add_feed_forward_length(N_FF)
    w.add_head_count(N_HEAD)
    w.add_head_count_kv(N_HEAD_KV)
    w.add_layer_norm_rms_eps(1e-6)
    w.add_rope_freq_base(10000.0)
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
        w.add_tensor(f"blk.{il}.post_attention_norm.weight", np.ones(N_EMBD, dtype=np.float32))
        if (il + 1) % N_LAYER != 0:
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


def server_configs(target: str, near: str) -> list[dict]:
    """(name, model, draft label, extra argv, env, base) of part s."""
    out = [{"name": "base", "model": "target", "draft": None, "argv": [], "env": {}, "base": "base"}]
    for label, dft in (("same", target), ("near", near)):
        for n, env in (("k4", {}), ("k4_skip", {SKIP: "1"}), ("k4_ckpt", {CKPT: "1"})):
            out.append({"name": f"{n}_{label}", "model": "target", "draft": label, "argv": ["-md", dft] + DS, "env": env, "base": "base"})
    out += [{"name": "base_flat", "model": "flat", "draft": None, "argv": [], "env": {}, "base": "base_flat"},
            {"name": "ngram_flat", "model": "flat", "draft": None, "argv": NG, "env": {}, "base": "base_flat"},
            {"name": "ngram_rs8_flat", "model": "flat", "draft": None, "argv": NG, "env": {NGRAM_RS: "8"}, "base": "base_flat"}]
    return out


def run_server(bin_dir, target: str, cfg: dict, log_path: str, threads: int) -> dict:
    cmd = [str(pathlib.Path(bin_dir) / "llama-server"), "-m", target, "-c", "512", "-np", "1", "--host", "127.0.0.1", "--port", str(e5_spec.PORT),
           "-lv", "4", "-t", str(threads), "--reasoning-format", "none"] + cfg["argv"]
    res: dict = {"name": cfg["name"], "model": cfg["model"], "draft": cfg["draft"], "env": cfg["env"], "argv": cfg["argv"], "rows": [], "error": None}
    srv = e5_spec.Server(cmd, log_path, cfg["env"] or None)
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
    res["drafted"] = sum(r.get("draft_n", 0) for r in res["rows"])
    res["accepted"] = sum(r.get("draft_n_accepted", 0) for r in res["rows"])
    res["acceptance"] = res["accepted"] / res["drafted"] if res["drafted"] else None
    return res


def run(args) -> int:
    notes: list[str] = []
    args.gpu = common.gpu_tag(args)
    bin_dir = e4_bonsai.llama_bin()
    config = {"bin_dir": str(bin_dir), "std_target": STD_TARGET, "std_near": STD_NEAR, "prompts": PROMPTS, "n_predict": N_PREDICT,
              "threads": args.threads, "parts": args.parts}
    doc: dict = {"models": {}, "probe": None, "server": {}}
    work = pathlib.Path(tempfile.mkdtemp(prefix="rs-cpu-"))
    try:
        specs = (("target", STD_TARGET, True), ("near", STD_NEAR, True), ("flat", STD_FLAT, True), ("probe", STD_TARGET, False))
        files = {n: str(work / f"tiny-{n}.gguf") for n, _, _ in specs}
        doc["models"] = {n: {"std": std, "ascii_only": asc, "n_vocab": write_tiny(files[n], std, ascii_only=asc)} for n, std, asc in specs}
        target, near = files["target"], files["near"]
        if "p" in args.parts:
            exe = wrap_b.build_probe(bin_dir, notes)
            if exe:
                res = wrap_b.run_probe(exe, files["probe"], 0, 300.0)
                res["verdict"] = v = wrap_b.probe_verdict(res)
                if v.get("single") and v.get("batch"):
                    notes.append("probe: seq_rm over one-token decodes: max logit diff %.3g (logits up to %.3g), top-1 equal %d/%d; inside one "
                                 "batch: max diff %.3g" % (v["single"]["max_abs_diff"], v["single"]["max_abs_ref"], v["single"]["top1_equal"],
                                                           v["single"]["n"], v["batch"]["max_abs_diff"]))
                res["cmd"] = [pathlib.Path(c).name if os.sep in str(c) else c for c in res.get("cmd", [])]
                doc["probe"] = res
        if "s" in args.parts and (bin_dir / "llama-server").exists():
            from infra.results import artifact_path
            for cfg in server_configs(target, near):
                log = str(artifact_path(EXP, f"server-{cfg['name']}.log", args.gpu))
                res = run_server(bin_dir, files[cfg["model"]], cfg, log, args.threads)
                res["argv"] = [pathlib.Path(a).name if a.endswith(".gguf") else a for a in res["argv"]]
                doc["server"][cfg["name"]] = res
                print(f"[{EXP}] {cfg['name']}: drafted {res['drafted']} accepted {res['accepted']}", flush=True)
            bases = {c["name"]: c["base"] for c in server_configs(target, near)}
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
    s = doc["server"]
    for n in ("same", "near"):
        acc = {k: (s.get(f"{k}_{n}") or {}).get("acceptance") for k in ("k4", "k4_skip", "k4_ckpt")}
        if any(v is not None for v in acc.values()):
            notes.append(f"draft {n}: acceptance " + ", ".join(f"{k} {'n/a' if v is None else '%.3f' % v}" for k, v in acc.items()))
    common.write_result(EXP, args, config, [], notes, False, extra=doc)
    return 0


def parse_args(argv=None):
    ap = argparse.ArgumentParser(prog="exps.rs_rollback_cpu", description=__doc__.split("\n")[0])
    common.add_common_args(ap)
    ap.add_argument("--parts", type=lambda t: t.replace(",", ""), default="ps", help="p (rollback probe), s (server)")
    ap.add_argument("--threads", type=int, default=4)
    return ap.parse_args(argv)


def main(argv=None) -> int:
    a = parse_args(argv)
    if a.dry_run:
        common.print_plan({"exp": EXP, "parts": a.parts, "configs": [{k: c[k] for k in ("name", "model", "draft", "env", "base")} for c in server_configs("<t>", "<n>")],
                           "probe_args": wrap_b.PROBE_ARGS, "prompts": len(PROMPTS), "n_predict": N_PREDICT, "quick": a.quick})
        return 0
    return run(a)


if __name__ == "__main__":
    sys.exit(main())
