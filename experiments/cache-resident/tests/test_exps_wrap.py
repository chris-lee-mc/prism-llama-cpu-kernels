"""Wrap A (A1 + A2), A3 and B drivers, their helpers in r2b_multicol / r2b_spec, the experiment switches in common/ and the server."""
import json
import os
import pathlib
import shutil
import subprocess
import sys

import pytest

import test_exps_r2b as r2bt
from exps import make_prompts, r2a_longctx as r2a, r2b_multicol as mc, r2b_spec as sp, rs_rollback_cpu, wrap_a, wrap_a3, wrap_b

EXP = pathlib.Path(__file__).resolve().parent.parent
REPO = EXP.parent.parent
SMI = {"clocks.sm": 2800.0, "clocks.max.sm": 3090.0, "clocks.mem": 13365.0, "clocks.max.mem": 14001.0, "power.draw": 400.0,
       "power.limit": 600.0, "power.default_limit": 600.0, "temperature.gpu": 60.0}


# ---------------------------------------------------------------- the switches in the C++ sources (compile check: scratch cmake object builds)

def test_switches_are_env_gated_and_default_off():
    com = (REPO / "common" / "common.h").read_text()
    f = com[com.index("static inline uint32_t common_exp_ngram_rs()"):][:400]
    assert 'getenv("GGML_EXP_NGRAM_RS")' in f and "static const uint32_t n" in f and ": 0u" in f
    spec = (REPO / "common" / "speculative.cpp").read_text()
    f = spec[spec.index("static bool common_exp_skip_draft_reeval()"):][:400]
    assert 'getenv("GGML_EXP_SKIP_DRAFT_REEVAL")' in f and "static const bool enabled" in f
    proc = spec[spec.index("struct common_speculative_impl_draft_simple"):]
    proc = proc[proc.index("bool process(const llama_batch & batch) override {"):proc.index("void draft(common_speculative_draft_params_vec")]
    assert "if (common_exp_skip_draft_reeval() &&" in proc and "llama_memory_seq_pos_max(mem, s)" in proc
    assert proc.index("const int ret = llama_decode(ctx_dft, batch_dft);") > proc.index("common_exp_skip_draft_reeval()")
    assert "common_sampler_sample(smpl, ctx_dft, i == 0 ? i_last[seq_id] : i_batch, true);" in spec       # id_last may not be batch row 0
    srv = (REPO / "tools" / "server" / "server-context.cpp").read_text()
    assert 'getenv("GGML_EXP_DRAFT_CKPT")' in srv and 'getenv("GGML_EXP_SKIP_DRAFT_REEVAL")' in srv
    assert "exp_skip_reeval = env_skip && atoi(env_skip) != 0 && ctx_dft_seq_rm_type == COMMON_CONTEXT_SEQ_RM_TYPE_RS;" in srv
    assert "if (!keep_dft && !llama_memory_seq_rm(llama_get_memory(ctx_dft), slot.id, ckpt.pos_max + 1, -1))" in srv
    for p in (REPO / "common" / "common.h", REPO / "common" / "speculative.cpp", REPO / "tools" / "server" / "server-context.cpp"):
        txt = p.read_text()
        for key in ("GGML_EXP_", "exp skip-reeval", "exp ngram-rs"):
            for ln in [ln for ln in txt.splitlines() if key in ln]:
                assert all(ord(c) < 0x80 for c in ln), ln


def test_the_printed_counter_lines_match_the_parser():
    spec = (REPO / "common" / "speculative.cpp").read_text()
    assert 'SPC_TRC("exp ngram-rs     %16s: window = %u, cut drafts = %zu, cut tokens = %zu\\n"' in spec
    assert 'SPC_TRC("exp skip-reeval  %16s: skipped batches = %zu, catch-up tokens = %zu\\n"' in spec
    assert sp.EXP_CUT_RE.search("spec common_specu: exp ngram-rs            ngram-mod: window = 16, cut drafts = 3, cut tokens = 90")
    assert sp.EXP_SKIP_RE.search("spec common_specu: exp skip-reeval      draft-simple: skipped batches = 57, catch-up tokens = 54")
    srv = (REPO / "tools" / "server" / "server-context.cpp").read_text()
    for name in ("GGML_EXP_DRAFT_CKPT", "GGML_EXP_SKIP_DRAFT_REEVAL"):
        line = next(ln for ln in srv.splitlines() if 'SRV_INF("%s", "' + name + ":" in ln)
        assert sp.EXP_MARK_RE.search(line.split('"%s", "')[1]).group(1) == name


# ---------------------------------------------------------------- helpers

def test_gpu600_gate():
    assert mc.gpu600_problem("pro6000-ws", SMI) is None and mc.gpu600_problem("pro6000-server", None) is None
    assert "not a 600 W" in mc.gpu600_problem("pro6000-maxq", SMI) and "not a 600 W" in mc.gpu600_problem("rtx5090", SMI)
    assert "default power limit 300 W" in mc.gpu600_problem("pro6000-ws", dict(SMI, **{"power.default_limit": 300.0}))


def test_rounds_alternate_and_paired_gain_per_round():
    cfgs = [("off", {}), ("on", {"X": "1"})]
    assert [[n for n, _ in mc.round_configs(cfgs, r)] for r in range(4)] == [["off", "on"], ["on", "off"]] * 2
    rows = []
    for rnd, g in enumerate((1.10, 1.05, 1.20)):
        for cfg, mult in (("off", 1.0), ("on", g)):
            rows += [{"config": cfg, "round": rnd, "kind": "pp", "width": 4, "tok_s": 400.0 * mult},
                     {"config": cfg, "round": rnd, "kind": "tg", "width": 1, "tok_s": 100.0 * (1.0 if cfg == "off" else 0.99)}]
    rows.append({"config": "off", "round": 3, "kind": "pp", "width": 4, "tok_s": 1.0})        # on never ran round 3: not paired
    p = mc.paired_rounds(rows, "off", "on")
    e = p["pp"]["4"]
    assert e["rounds"] == [0, 1, 2] and e["gain_pct"] == pytest.approx([10.0, 5.0, 20.0]) and e["median"] == pytest.approx(10.0)
    assert e["min"] == pytest.approx(5.0) and e["max"] == pytest.approx(20.0) and e["same_sign"] is True
    assert p["tg"]["1"]["median"] == pytest.approx(-1.0) and p["tg"]["1"]["same_sign"] is True
    s = mc.summarize_bench([dict(r, config={"off": "base", "on": "head"}[r["config"]]) for r in rows], ["base", "head"], ref="base")
    assert s["pp"]["4"]["gain_pct_head"] == pytest.approx(10.0) and "gain_pct_base" not in s["pp"]["4"]


def test_prompt_blocks_keep_the_categories_balanced_and_the_order():
    order = sp.balanced_subset(make_prompts.load_longctx(make_prompts.R2A_OUT))
    blocks = sp.prompt_blocks(order, 4)
    assert [len(b) for b in blocks] == [6] * 4 and all(r2a._count(b) == {c: 2 for c in make_prompts.R2A_CATEGORIES} for b in blocks)
    assert sorted(p["id"] for b in blocks for p in b) == sorted(p["id"] for p in order)
    pos = {p["id"]: i for i, p in enumerate(order)}
    assert all([pos[p["id"]] for p in b] == sorted(pos[p["id"]] for p in b) for b in blocks)
    assert [len(b) for b in sp.prompt_blocks(order[:7], 3)] and sum(len(b) for b in sp.prompt_blocks(order[:7], 3)) == 7


def test_block_orders_run_every_pair_both_ways():
    for names in (["base", "off", "on"], wrap_b.CONFIG_NAMES):
        orders = sp.block_orders(names, 4)
        assert all(sorted(o) == sorted(names) for o in orders)
        for i, a in enumerate(names):
            for b in names[i + 1:]:
                first = [o.index(a) < o.index(b) for o in orders]
                assert any(first) and not all(first), (a, b)


STATS_LOG = """0.01 I llama_context: n_rs_seq              = 4
spec common_specu: statistics     draft-simple: #calls(b,g,a) =    1      6      6, #gen drafts =      6, #acc drafts =     3, #gen tokens =     21, #acc tokens =     9, #mean acc len = 2.50, #acc rate/pos = (0.500, 0.500, 0.333, 0.167), dur(b,g,a) = 0.000, 50.000, 0.004 ms
spec common_specu: exp skip-reeval      draft-simple: skipped batches = 6, catch-up tokens = 2
spec common_specu: statistics     draft-simple: #calls(b,g,a) =    2     56     56, #gen drafts =     56, #acc drafts =    40, #gen tokens =    221, #acc tokens =   150, #mean acc len = 3.70, #acc rate/pos = (0.9), dur(b,g,a) = 0.000, 350.000, 0.050 ms
spec common_specu: exp skip-reeval      draft-simple: skipped batches = 56, catch-up tokens = 22
"""


def test_parse_and_difference_the_statistics_lines():
    st = sp.parse_stats(STATS_LOG)
    assert len(st["draft-simple"]) == 2 and st["draft-simple"][1]["dur_g_ms"] == 350.0 and st["draft-simple"][1]["skipped_batches"] == 56
    t = sp.timed_stats(st)["draft-simple"]
    assert (t["calls_g"], t["gen_tokens"], t["acc_tokens"], t["dur_g_ms"], t["n_requests"]) == (50, 200, 141, pytest.approx(300.0), 1)
    assert (t["skipped_batches"], t["catchup_tokens"]) == (50, 20)
    both = sp.add_stats({"draft-simple": t}, {"draft-simple": t, "ngram-mod": {"calls_g": 3, "rs_window": 16}})
    assert both["draft-simple"]["calls_g"] == 100 and both["ngram-mod"] == {"calls_g": 3, "rs_window": 16}
    cut = sp.parse_stats("spec common_specu: statistics        ngram-mod: #calls(b,g,a) =    1     37      8, #gen drafts =      8, #acc drafts =     8, "
                         "#gen tokens =     58, #acc tokens =    58, dur(b,g,a) = 0.031, 0.166, 0.011 ms\n"
                         "spec common_specu: exp ngram-rs            ngram-mod: window = 8, cut drafts = 8, cut tokens = 64\n")
    assert cut["ngram-mod"][0]["rs_window"] == 8 and sp.timed_stats(cut)["ngram-mod"]["rs_window"] == 8
    assert sp.timed_stats({}) == {} and sp.parse_stats("nothing") == {}


def _row(i, cat, tps, text="t", dn=40, da=20, n=100):
    return {"id": f"r{i:03d}", "category": cat, "tokens_predicted": n, "predicted_per_second": tps, "predicted_ms": 1e3 * n / tps,
            "draft_n": dn, "draft_n_accepted": da, "text": text}


def test_step_split_and_pairs_by_category():
    cats = ["edit_diff", "edit_full_file"]
    runs = {"a": {"spec_type": "draft-simple", "rows": [_row(i, cats[i % 2], 120.0) for i in range(4)],
                  "stats": {"draft-simple": {"dur_g_ms": 200.0, "calls_g": 40}}},
            "b": {"spec_type": "draft-simple", "rows": [_row(i, cats[i % 2], 100.0, "x" if i == 0 else "t") for i in range(4)]}}
    ss = sp.step_split(runs["a"])
    assert ss["draft_ms_per_call"] == pytest.approx(5.0) and ss["rest_ms_per_step"] == pytest.approx(ss["ms_per_step"] - 200.0 / ss["steps"])
    assert sp.step_split(runs["b"])["ms_per_step"] and "draft_ms_per_call" not in sp.step_split(runs["b"]) and sp.step_split({}) == {}
    p = sp.pair_by_category(runs, "a", "b")
    assert p["all"]["speedup"] == pytest.approx(1.2) and p["all"]["text_equal"] == 3 and p["all"]["speedup_ci95"]
    assert set(p["by_category"]) == set(cats) and p["by_category"]["edit_diff"]["text_equal"] == 1
    assert sp.pair_by_category(runs, "a", "missing") is None


def test_wrap_b_expected_paths_and_verification():
    cfgs = {c["name"]: c for c in wrap_b.matrix_configs()}
    exp = {n: wrap_b.expected_path(c) for n, c in cfgs.items()}
    assert exp["base"]["rollback_path"] == "none" and exp["ngram"] == {"rollback_path": "checkpoint", "n_rs_seq_target": 0, "switches": []}
    assert exp["ngram_rs64"]["n_rs_seq_target"] == 64 and exp["k4q8_skip"]["switches"] == [wrap_b.SKIP] and exp["k4q8"]["n_rs_seq_target"] == 4
    runs = {"ngram_rs16": {"blocks": [{"block": 0, "checks": {"rollback_path": "in_batch", "n_rs_seq_target": 16}, "exp_switches": [],
                                       "stats": {"ngram-mod": {"rs_window": 16}}}]}}
    v = wrap_b.verify_paths([cfgs["ngram_rs16"], cfgs["ngram"]], runs)
    assert v["ngram_rs16"]["ok"] is True and v["ngram"]["ok"] is False                 # a config that never ran does not pass
    runs["ngram_rs16"]["blocks"][0]["stats"] = {"ngram-mod": {"rs_window": 8}}
    assert wrap_b.verify_paths([cfgs["ngram_rs16"]], runs)["ngram_rs16"]["ok"] is False


PROBE_OUT = {"model": "m.gguf", "n_vocab": 10, "n_rs_seq": 4, "n_prompt": 64, "n_extra": 4, "ngl": 99, "cases": [
    {"kept": 0, "rollback": 4, "decode": "single", "seq_rm_ok": True, "max_abs_diff": 13.0, "max_abs_ref": 9.0, "top1_equal": False},
    {"kept": 0, "rollback": 4, "decode": "batch", "seq_rm_ok": True, "max_abs_diff": 11.0, "max_abs_ref": 9.0, "top1_equal": False},
    {"kept": 1, "rollback": 3, "decode": "single", "seq_rm_ok": True, "max_abs_diff": 12.0, "max_abs_ref": 9.0, "top1_equal": False},
    {"kept": 1, "rollback": 3, "decode": "batch", "seq_rm_ok": True, "max_abs_diff": 1e-6, "max_abs_ref": 9.0, "top1_equal": True}]}


def test_probe_verdict_keeps_the_whole_batch_case_apart():
    v = wrap_b.probe_verdict(PROBE_OUT)
    assert v["single"]["max_abs_diff"] == 13.0 and v["single"]["n"] == 2 and v["single"]["top1_equal"] == 0
    assert v["batch"]["max_abs_diff"] == pytest.approx(1e-6) and v["batch"]["n"] == 1 and v["batch_whole"]["by_rollback"] == {"4": 11.0}


def test_probe_source_builds_against_llama_h_only():
    src = wrap_b.PROBE_SRC.read_text()
    assert '#include "llama.h"' in src and "common.h" not in src and "llama_memory_seq_rm(mem, 0, n_prompt + m, -1)" in src
    assert "cp.n_rs_seq = (uint32_t) n_rs" in src


def test_a3_patch_is_the_pr_change_and_reverses_on_this_tree():
    txt = wrap_a3.PATCH.read_text()
    files = sorted({ln.split(" b/")[1] for ln in txt.splitlines() if ln.startswith("diff --git ")})
    assert files == sorted(wrap_a3.PATCH_FILES) and "ggml_cuda_pdl_sync();" in txt and 'getenv("GGML_CUDA_PQ2_MULTICOL")' in txt
    if not shutil.which("git") or not (REPO / ".git").exists():
        pytest.skip("no git checkout")
    r = subprocess.run(["git", "apply", "-R", "--check", str(wrap_a3.PATCH)], cwd=REPO, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr


def test_a3_copy_keeps_cmake_build_info_and_drops_build_dirs():
    top = wrap_a3.copy_ignore(str(REPO), ["build-cuda", "build-xcframework.sh", ".git", "experiments", "ggml", "cmake", "x.gguf", "__pycache__"])
    assert top == {"build-cuda", "build-xcframework.sh", ".git", "experiments", "x.gguf", "__pycache__"}
    assert wrap_a3.copy_ignore(str(REPO / "cmake"), ["build-info.cmake", "x.gguf"]) == {"x.gguf"}


def test_a3_cmake_flags_are_the_bootstrap_flags():
    boot = (EXP / "infra" / "bootstrap.sh").read_text()
    for flag in wrap_a3.cmake_cmd("/s", "/s/b", "89", True)[6:]:
        assert flag.replace("89", '"$CC"') in boot or flag in boot, flag
    assert "CCACHE_BASEDIR" in boot and "CCACHE_NOHASHDIR=1" in boot


# ---------------------------------------------------------------- fakes

FAKE_SERVER = """#!{py}
import json, os, sys
from http.server import BaseHTTPRequestHandler, HTTPServer
a = sys.argv
port = int(a[a.index("--port") + 1])
spec = a[a.index("--spec-type") + 1] if "--spec-type" in a else "none"
draft = "-md" in a
mc = os.environ.get("GGML_CUDA_PQ2_MULTICOL") == "1"
ign = os.environ.get("FAKE_IGNORE_EXP") == "1"
skip = os.environ.get("GGML_EXP_SKIP_DRAFT_REEVAL") == "1" and not ign
ckpt = os.environ.get("GGML_EXP_DRAFT_CKPT") == "1" and not ign
nrs = 0 if ign else int(os.environ.get("GGML_EXP_NGRAM_RS", "0"))
impl = "draft-simple" if draft else ("ngram-mod" if spec != "none" else None)
st = dict(g=0, gt=0, at=0, dg=0.0, sk=0, cu=0, cd=0, ct=0)
def stats():
    print("spec common_specu: statistics %16s: #calls(b,g,a) =    1 %6d %6d, #gen drafts = %6d, #acc drafts = %5d, #gen tokens = %6d, "
          "#acc tokens = %5d, dur(b,g,a) = 0.000, %.3f, 0.004 ms" % (impl, st["g"], st["g"], st["g"], st["g"], st["gt"], st["at"], st["dg"]), flush=True)
    if skip and impl == "draft-simple":
        print("spec common_specu: exp skip-reeval  %16s: skipped batches = %d, catch-up tokens = %d" % (impl, st["sk"], st["cu"]), flush=True)
    if nrs and impl == "ngram-mod":
        print("spec common_specu: exp ngram-rs     %16s: window = %d, cut drafts = %d, cut tokens = %d" % (impl, nrs, st["cd"], st["ct"]), flush=True)
class H(BaseHTTPRequestHandler):
    def log_message(self, *x): pass
    def _send(self, obj):
        d = json.dumps(obj).encode()
        self.send_response(200); self.send_header("Content-Length", str(len(d))); self.end_headers(); self.wfile.write(d)
    def do_GET(self):
        self._send({{"status": "ok"}})
    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        if self.path == "/apply-template":
            return self._send({{"prompt": "T:" + body["messages"][-1]["content"][-60:]}})
        if self.path == "/tokenize":
            return self._send({{"tokens": list(range(len(body["content"]) // 4))}})
        n = body["n_predict"]
        speed = 100.0
        if impl == "draft-simple":
            speed = 120.0 * (1.1 if mc else 1.0) * (1.05 if skip else 1.0) * (1.2 if ckpt else 1.0)
        elif impl == "ngram-mod":
            speed = 130.0 * {{0: 1.0, 16: 1.1, 64: 1.2}}.get(nrs, 1.0)
        t = {{"predicted_per_second": speed, "predicted_ms": 1e3 * n / speed, "predicted_n": n, "prompt_n": 3000, "prompt_ms": 1000.0}}
        if impl:
            t.update(draft_n=n, draft_n_accepted=n // 2)
            st["g"] += max(n // 4, 1); st["gt"] += n; st["at"] += n // 2; st["dg"] += 2.0 * max(n // 4, 1)
            st["sk"] += max(n // 4, 1); st["cu"] += 1; st["cd"] += 1; st["ct"] += 10
        self._send({{"content": "out:" + body["prompt"], "tokens_predicted": n, "timings": t, "stop_type": "limit"}})
        if impl:
            stats()
if spec == "none":
    print("llama_context: n_rs_seq              = 0", flush=True)
else:
    rs = 4 if draft else nrs
    for _ in range(2 if draft else 1):
        print("llama_context: n_rs_seq              = %d" % rs, flush=True)
        print("cmn  common_conte: the context " + ("supports bounded" if rs else "does not support") + " partial sequence removal", flush=True)
    if not rs:
        print("srv    load_model: speculative decoding will use checkpoints", flush=True)
    if skip:
        print("srv    load_model: GGML_EXP_SKIP_DRAFT_REEVAL: the draft context keeps its drafted tokens", flush=True)
    if ckpt:
        print("srv    load_model: GGML_EXP_DRAFT_CKPT: the draft context uses checkpoints", flush=True)
HTTPServer(("127.0.0.1", port), H).serve_forever()
"""

FAKE_PROBE = "#!{py}\nprint(" + repr(json.dumps(PROBE_OUT)).replace("{", "{{").replace("}", "}}") + ")\n"


@pytest.fixture
def wpod(tmp_path, monkeypatch):
    b = tmp_path / "bin"
    b.mkdir()
    r2bt._script(b / "llama-bench", r2bt.FAKE_BENCH)
    r2bt._script(b / "test-backend-ops", r2bt.FAKE_TBO)
    r2bt._script(b / "llama-server", FAKE_SERVER)
    r2bt._script(tmp_path / "probe", FAKE_PROBE)
    files = {"pq2": "target.gguf", "draft_q8_0": "draft-q8.gguf"}
    for n in files.values():
        (tmp_path / n).write_bytes(b"gguf")
    monkeypatch.setenv("CR_RESULTS_DIR", str(tmp_path / "results"))
    monkeypatch.setenv("CR_GPU_TAG", "pro6000-ws")
    monkeypatch.setenv("CR_LLAMA_BIN", str(b))
    monkeypatch.setenv("HF_TOKEN", "hf_secretsecret123")
    monkeypatch.setenv("RUNPOD_API_KEY", "rpa_secretsecret456")
    for k in [k for k in os.environ if k.startswith("GGML_EXP_") or k == mc.ENV_VAR]:
        monkeypatch.delenv(k)
    monkeypatch.setattr(mc.e4_pdl, "model_path", lambda name, notes: (str(tmp_path / files[name]), {"file": files[name]}))
    monkeypatch.setattr(mc.e4_pdl, "smi_sample", lambda: dict(SMI))
    monkeypatch.setattr(wrap_b, "build_probe", lambda bin_dir, notes: tmp_path / "probe")
    return tmp_path


def _result(tmp, exp, gpu):
    files = sorted((tmp / "results" / exp / gpu).glob("2*.json"))
    assert len(files) == 1
    return json.loads(files[0].read_text())


def test_wrap_a_end_to_end_with_fakes(wpod):
    assert wrap_a.main(["--budget-min", "10", "--per-category", "2", "--blocks", "2"]) == 0
    d = _result(wpod, "wrap_a", "pro6000-ws")["data"]
    assert d["gate"]["problem"] is None and [t["config"] for t in d["tbo"]] == ["on", "off"] and all(mc.tbo_passed(t) for t in d["tbo"])
    assert len(d["bench"]) == 4 * 2 * 10 and [r["config"] for r in d["bench"][::10]] == ["off", "on", "on", "off", "off", "on", "on", "off"]
    bp = d["bench_paired"]
    assert bp["pp"]["4"]["n"] == 4 and bp["pp"]["4"]["median"] == pytest.approx(100.0) and bp["pp"]["2"]["median"] == pytest.approx(0.0)
    s = d["spec"]
    assert s["orders"] == sp.block_orders(wrap_a.SPEC_NAMES, 2) and len(s["blocks"]) == 2
    runs = s["runs"]
    assert all(len(runs[n]["rows"]) == 6 and len(runs[n]["blocks"]) == 2 for n in wrap_a.SPEC_NAMES)
    assert runs["k4q8_on"]["env"] == {mc.ENV_VAR: "1"} and runs["k4q8_off"]["env"] == {}
    cmd = runs["k4q8_on"]["cmd"]
    assert cmd[cmd.index("-md") + 1].endswith("draft-q8.gguf") and cmd[cmd.index("--spec-draft-n-max") + 1] == "4" and cmd[cmd.index("-lv") + 1] == "4"
    summ = s["summary"]
    assert summ["on_vs_off"]["all"]["speedup"] == pytest.approx(1.1) and summ["on_vs_off"]["all"]["text_equal_frac"] == 1.0
    assert set(summ["on_vs_off"]["by_category"]) == set(make_prompts.R2A_CATEGORIES)
    assert summ["vs_base"]["k4q8_off"]["all"]["speedup"] == pytest.approx(1.2) and summ["vs_base"]["k4q8_on"]["all"]["speedup_ci95"]
    sp_on = summ["step_split"]["k4q8_on"]
    assert sp_on["draft_ms_per_call"] == pytest.approx(2.0) and sp_on["impls"]["draft-simple"]["n_requests"] == 6
    assert any("k4q8_on over k4q8_off 1.100x" in n for n in d["notes"]) and any("width 4: on over off +100.0%" in n for n in d["notes"])
    assert "hf_secretsecret123" not in json.dumps(d) and not (wpod / "results" / "wrap_a" / "pro6000-ws" / "raw" / "partial.json").exists()
    raw = {f.name.split("-", 1)[1] for f in (wpod / "results" / "wrap_a" / "pro6000-ws" / "raw").glob("*-server-*.log")}
    assert {"server-b0-base.log", "server-b1-k4q8_on.log"} <= raw


def test_wrap_a_refuses_a_max_q_and_stops_on_a_failing_kernel(wpod, monkeypatch):
    monkeypatch.setenv("CR_GPU_TAG", "pro6000-maxq")
    assert wrap_a.main(["--budget-min", "6"]) == 1
    d = _result(wpod, "wrap_a", "pro6000-maxq")["data"]
    assert "not a 600 W" in d["gate"]["problem"] and d["bench"] == [] and d["tbo"] == [] and d["spec"] is None
    monkeypatch.setenv("CR_GPU_TAG", "pro6000-server")
    monkeypatch.setenv("FAKE_TBO_FAIL", "1")
    assert wrap_a.main(["--budget-min", "6"]) == 1
    d = _result(wpod, "wrap_a", "pro6000-server")["data"]
    assert d["bench"] == [] and d["spec"] is None and any(n.startswith("stop: test-backend-ops did not pass") for n in d["notes"])


def test_wrap_b_end_to_end_with_fakes(wpod):
    assert wrap_b.main(["--budget-min", "10", "--per-category", "2", "--blocks", "2"]) == 0
    d = _result(wpod, "wrap_b", "pro6000-ws")["data"]
    pr = d["probe"]["models"]
    assert set(pr) == {"draft_q8_0", "pq2"} and pr["pq2"]["verdict"]["single"]["max_abs_diff"] == 13.0
    m = d["matrix"]
    assert set(m["runs"]) == set(wrap_b.CONFIG_NAMES) and all(len(r["rows"]) == 6 for r in m["runs"].values())
    summ = m["summary"]
    assert all(v["ok"] for v in summ["paths"].values()), summ["paths"]
    p = summ["pairs"]
    assert p["k4q8_skip_vs_k4q8"]["all"]["speedup"] == pytest.approx(1.05) and p["k4q8_ckpt_vs_k4q8"]["all"]["speedup"] == pytest.approx(1.2)
    assert p["ngram_rs16_vs_ngram"]["all"]["speedup"] == pytest.approx(1.1) and p["ngram_rs64_vs_ngram"]["all"]["speedup_ci95"]
    assert set(p["ngram_rs64_vs_ngram"]["by_category"]) == set(make_prompts.R2A_CATEGORIES)
    assert summ["ngram_cut"]["ngram_rs16"]["ngram-mod"]["rs_window"] == 16 and summ["ngram_cut"]["ngram_rs16"]["ngram-mod"]["cut_tokens"] == 60
    assert summ["skip_reeval"]["k4q8_skip"]["draft-simple"]["skipped_batches"] > 0 and "k4q8" not in summ["skip_reeval"]
    assert summ["vs_base"]["ngram"]["checks"]["rollback_path"] == "checkpoint" and summ["vs_base"]["ngram_rs64"]["checks"]["n_rs_seq_target"] == 64
    assert m["runs"]["k4q8_ckpt"]["blocks"][0]["exp_switches"] == [wrap_b.CKPT] and m["runs"]["k4q8"]["blocks"][0]["exp_switches"] == []
    assert summ["vs_base"]["k4q8_skip"]["all"]["text_equal_frac"] == 1.0 and d["config"]["envs"]["ngram_rs64"] == {wrap_b.NGRAM_RS: "64"}
    assert "rpa_secretsecret456" not in json.dumps(d)


def test_wrap_b_flags_a_switch_that_did_not_take_effect(wpod, monkeypatch):
    monkeypatch.setenv("FAKE_IGNORE_EXP", "1")
    assert wrap_b.main(["--parts", "m", "--configs", "base,ngram_rs16,k4q8_skip", "--budget-min", "6", "--per-category", "2", "--blocks", "2"]) == 1
    d = _result(wpod, "wrap_b", "pro6000-ws")["data"]
    paths = d["matrix"]["summary"]["paths"]
    assert paths["base"]["ok"] is True and paths["ngram_rs16"]["ok"] is False and paths["k4q8_skip"]["ok"] is False
    assert any("path verification failed" in n for n in d["notes"])


def test_wrap_a3_end_to_end_with_a_fake_base_build(wpod, monkeypatch):
    monkeypatch.setenv("CR_GPU_TAG", "rtx4090")
    base = wpod / "base" / "bin"
    base.mkdir(parents=True)
    for t in ("llama-bench", "test-backend-ops"):
        shutil.copy(wpod / "bin" / t, base / t)

    def fake_build(args, budget, notes, doc, config, head_bin):
        doc["build"] = {"applied": True, "differs": True}
        return base, str(wpod / "target.gguf")
    monkeypatch.setattr(wrap_a3, "build_part", fake_build)
    assert wrap_a3.main(["--budget-min", "10"]) == 0
    d = _result(wpod, "wrap_a3", "rtx4090")["data"]
    assert [t["config"] for t in d["tbo"]] == ["head", "base"] and all(mc.tbo_passed(t) for t in d["tbo"])
    assert len(d["bench"]) == 4 * 2 * 10 and {r["config"] for r in d["bench"]} == {"base", "head"}
    assert d["bench_paired"]["a"] == "base" and d["bench_paired"]["pp"]["4"]["median"] == pytest.approx(0.0)
    assert d["perf"]["gain"] and all(g["gain_pct"] == pytest.approx(0.0) for g in d["perf"]["gain"])
    assert d["bench_summary"]["pp"]["4"]["gain_pct_head"] == pytest.approx(0.0)
    monkeypatch.setenv("CR_GPU_TAG", "rtx5090")
    assert wrap_a3.main(["--budget-min", "6"]) == 1
    assert "not an RTX 4090" in _result(wpod, "wrap_a3", "rtx5090")["data"]["gate"]["problem"]


@pytest.mark.parametrize("mod", [wrap_a, wrap_a3, wrap_b, rs_rollback_cpu])
def test_dry_runs(mod, capsys):
    assert mod.main(["--dry-run"]) == 0
    plan = json.loads(capsys.readouterr().out)
    assert plan["exp"] == mod.EXP


def test_args():
    with pytest.raises(SystemExit):
        wrap_a.parse_args(["--rounds", "2"])
    with pytest.raises(SystemExit):
        wrap_b.parse_args(["--configs", "k4q8"])
    with pytest.raises(SystemExit):
        wrap_a3.parse_args(["--parts", "build,x"])
    assert wrap_a.parse_args(["--parts", "t,b"]).parts == "tb" and wrap_b.parse_args(["--quick"]).blocks == 2
