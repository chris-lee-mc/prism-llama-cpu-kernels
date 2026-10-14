"""R2b: the PQ2_0 multi-column switch (CUDA source checks, exps.r2b_multicol) and the draft / combined-type matrix (exps.r2b_spec)."""
import json
import os
import pathlib
import re
import stat
import sys

import pytest

from exps import common, e5_spec, make_prompts, r2a_longctx as r2a, r2b_multicol as mc, r2b_spec as sp

EXP = pathlib.Path(__file__).resolve().parent.parent
REPO = EXP.parent.parent
CUDA = REPO / "ggml" / "src" / "ggml-cuda"


def _script(path, text):
    path.write_text(text.format(py=sys.executable))
    path.chmod(path.stat().st_mode | stat.S_IXUSR)


# ---------------------------------------------------------------- the CUDA change (source checks; the compile check is a cmake object build)

def test_switch_is_read_once_and_default_stays_ada_only():
    src = (CUDA / "common.cuh").read_text()
    body = src[src.index("static inline bool ggml_cuda_pq2_multicol"):]
    body = body[:body.index("\n}\n")]
    assert 'getenv("GGML_CUDA_PQ2_MULTICOL")' in body and "static const bool" in body
    assert "env_newer ? cc >= GGML_CUDA_CC_ADA_LOVELACE : cc == GGML_CUDA_CC_ADA_LOVELACE" in body      # unset: Ada only, as before
    assert "GGML_CUDA_CC_IS_NVIDIA(cc)" in body and "!ggml_cuda_batch_invariant()" in body and "ncols_dst >= 3 && ncols_dst <= 8" in body
    assert "GGML_USE_HIP" in body and "return false" in body


def test_mc_kernel_waits_for_the_quantizer_before_it_reads_vy():
    src = (CUDA / "mmvq-pq2_0.cuh").read_text()
    k = src[src.index("mul_mat_vec_pq2_0_mc(const void"):src.index("mul_mat_vec_pq2_0_mc_launch")]
    assert "__restrict__" not in k and k.count("* GGML_CUDA_RESTRICT") == 3                    # restrict off the parameters, as mul_mat_vec_q
    sync = k.index("ggml_cuda_pdl_sync();")
    assert k.count("ggml_cuda_pdl_sync()") == 1 and sync < k.index("for (int c = lane") and sync < k.index("yc + c*QK8_1")
    assert "ggml_cuda_kernel_launch(mul_mat_vec_pq2_0_mc<ncols>" in src                      # still launched with PDL
    quant = (CUDA / "quantize.cu").read_text()
    q = quant[quant.index("static __global__ void quantize_q8_1("):]
    assert "ggml_cuda_pdl_lc();" in q[:q.index("ggml_cuda_pdl_sync();")]                      # the producer triggers early: the wait is needed


def test_source_comments_are_ascii():
    for f in ("common.cuh", "mmvq-pq2_0.cuh"):
        assert all(b < 0x80 for b in (CUDA / f).read_bytes()), f


# ---------------------------------------------------------------- r2b_multicol

PERF = """  MUL_MAT(type_a=pq2_0,type_b=f32,m=17408,n=4,k=5120,bs=[1,1],nr=[1,1],per=[0,1,2,3],k_v=0,o=1):      9000 runs -   120.00 us/run -  1.00 MFLOP/run - \x1b[1;34m 8.00 GFLOPS\x1b[0m
  MUL_MAT(type_a=pq2_0,type_b=f32,m=17408,n=1,k=5120,bs=[1,1],nr=[1,1],per=[0,1,2,3],k_v=0,o=1):      9000 runs -    60.00 us/run -  1.00 MFLOP/run - \x1b[1;34m 8.00 GFLOPS\x1b[0m
  MUL_MAT(type_a=q8_0,type_b=f32,m=17408,n=4,k=5120,bs=[1,1],nr=[1,1],per=[0,1,2,3],k_v=0,o=1):      9000 runs -   120.00 us/run
  MUL_MAT(type_a=pq2_0,type_b=f32,m=5120,n=4,k=17408,bs=[1,1],nr=[1,1],per=[0,1,2,3],k_v=0,o=1): not supported
"""


def test_parse_perf_and_gain():
    rows = mc.parse_perf(PERF)
    assert [(r["m"], r["n"], r["k"], r["us"]) for r in rows] == [(17408, 4, 5120, 120.0), (17408, 1, 5120, 60.0)]   # only pq2_0 lines that ran
    on = [dict(r, us=r["us"] / 2 if r["n"] == 4 else r["us"]) for r in rows]
    g = mc.perf_gain(rows, on)
    assert [(x["n"], x["gain_pct"]) for x in g] == [(1, pytest.approx(0.0)), (4, pytest.approx(100.0))]
    assert mc.perf_gain(rows, []) == [] and mc.parse_perf("nothing") == []


def test_commands():
    b = pathlib.Path("/b")
    assert mc.tbo_cmd(b) == ["/b/test-backend-ops", "test", "-o", "MUL_MAT", "-b", "CUDA0", "-p", "type_a=pq2_0,"]
    assert mc.tbo_cmd(b, "perf", None) == ["/b/test-backend-ops", "perf", "-o", "MUL_MAT", "-b", "CUDA0"]
    pp = mc.bench_pp_cmd(b, "/m.gguf")
    assert pp[pp.index("-p") + 1] == "1,2,3,4,5,6,7,8,9" and pp[pp.index("-n") + 1] == "0" and pp[pp.index("-r") + 1] == "5" and pp[pp.index("-d") + 1] == "0"
    tg = mc.bench_tg_cmd(b, "/m.gguf")
    assert tg[tg.index("-p") + 1] == "0" and tg[tg.index("-n") + 1] == "128" and tg[tg.index("-r") + 1] == "5"
    for c in (pp, tg):
        assert c[c.index("-fa") + 1] == "on" and c[c.index("-ub") + 1] == "2048" and c[c.index("-ctk") + 1] == "f16" and c[-2:] == ["-o", "json"]
    # the env of a config is the whole difference: off has neither knob, even when the pod env sets them
    assert mc.ENV_VAR not in mc.run_env({}) and mc.run_env({mc.ENV_VAR: "1"})[mc.ENV_VAR] == "1"
    assert dict(mc.CONFIGS) == {"off": {}, "on": {mc.ENV_VAR: "1"}, "on_pdl0": {mc.ENV_VAR: "1", "GGML_CUDA_PDL": "0"}}


def test_summarize_bench_median_over_complete_rounds_and_gain():
    rows = []
    for rnd in (0, 1):
        for cfg, mult in (("off", 1.0), ("on", 2.0), ("on_pdl0", 1.5)):
            for w in (1, 4):
                rows.append({"config": cfg, "round": rnd, "kind": "pp", "width": w, "tok_s": 100.0 * w * (mult if w == 4 else 1.0)})
            rows.append({"config": cfg, "round": rnd, "kind": "tg", "width": 1, "tok_s": 100.0 * (1.1 if cfg == "on" else 1.0)})
    rows.append({"config": "off", "round": 2, "kind": "pp", "width": 4, "tok_s": 1.0})        # round 2 has only one config: ignored
    s = mc.summarize_bench(rows, ["off", "on", "on_pdl0"])
    assert s["pp"]["4"]["ms"]["off"] == pytest.approx(10.0) and s["pp"]["4"]["ms"]["on"] == pytest.approx(5.0)
    assert s["pp"]["4"]["gain_pct_on"] == pytest.approx(100.0) and s["pp"]["4"]["gain_pct_on_pdl0"] == pytest.approx(50.0)
    assert s["pp"]["4"]["uses_kernel"] is True and s["pp"]["1"]["uses_kernel"] is False and s["pp"]["1"]["gain_pct_on"] == pytest.approx(0.0)
    assert s["tg_gain_pct"]["on"] == pytest.approx(10.0) and s["pp"]["4"]["rounds"] == [0, 1]
    assert any("width 4" in ln and "+100.0%" in ln for ln in mc.verdict_lines(s))
    assert mc.summarize_bench([], ["off", "on"]) == {"pp": {}, "tg": {}}


def test_mc_kernels_from_nsys_rows():
    k = mc.mc_kernels([{"name": "void mul_mat_vec_pq2_0_mc<(int)4>(const void *, const void *, float *, int, int, int, int, int)", "time_ns": 5.0, "count": 12},
                       {"name": "void mul_mat_vec_q<(ggml_type)142, (int)4, (bool)0, (bool)0>(const void *, const void *, const int *, float *, int)", "time_ns": 9.0, "count": 3},
                       {"name": "void mul_mat_vec_q<(ggml_type)8, (int)4, (bool)0, (bool)0>(...)", "time_ns": 1.0, "count": 7}])
    assert k["mc_kernel_instances"] == 12 and k["mc_widths"] == [4] and k["top"][0]["count"] == 3
    assert k["generic_pq2_instances"] == 3 and k["generic_pq2_widths"] == [4]            # other types (here 8) are not counted
    assert mc.mc_kernels([])["mc_kernel_instances"] == 0


FAKE_BENCH = """#!{py}
import json, os, sys
a = sys.argv
on = os.environ.get("GGML_CUDA_PQ2_MULTICOL") == "1"
pdl0 = os.environ.get("GGML_CUDA_PDL") == "0"
ps = [int(x) for x in a[a.index("-p") + 1].split(",")]
n = int(a[a.index("-n") + 1])
ents = []
for p in ps:
    if p > 0:
        ts = 100.0 * p * (2.0 if on and 3 <= p <= 8 else 1.0) * (0.9 if pdl0 else 1.0)
        ents.append({{"n_prompt": p, "n_gen": 0, "n_depth": 0, "avg_ts": ts, "stddev_ts": 1.0, "samples_ts": [ts, ts, ts]}})
if n > 0:
    ents.append({{"n_prompt": 0, "n_gen": n, "n_depth": 0, "avg_ts": 140.0, "stddev_ts": 1.0, "samples_ts": [140.0]}})
print(json.dumps(ents))
"""

FAKE_TBO = """#!{py}
import os, sys
a = sys.argv
on = os.environ.get("GGML_CUDA_PQ2_MULTICOL") == "1"
if "-p" in a and a[a.index("-p") + 1] == "type_a=pq2_0," and os.environ.get("FAKE_NO_FILTER"):
    print("  0/0 tests passed"); sys.exit(0)
if a[1] == "perf":
    for n in (1, 3, 4, 8):
        us = 100.0 / (2.0 if on and n >= 3 else 1.0)
        print("  MUL_MAT(type_a=pq2_0,type_b=f32,m=17408,n=%d,k=5120,bs=[1,1],nr=[1,1],per=[0,1,2,3],k_v=0,o=1):  1000 runs -  %.2f us/run - 1 MFLOP/run" % (n, us))
else:
    print("  MUL_MAT(type_a=pq2_0,type_b=f32,m=97,n=4,k=1024): OK")
    print("  1/1 tests passed" if not os.environ.get("FAKE_TBO_FAIL") else "  \\033[1;31mFAIL\\033[0m MUL_MAT(type_a=pq2_0,n=4)\\n  0/1 tests passed")
    sys.exit(1 if os.environ.get("FAKE_TBO_FAIL") else 0)
"""

FAKE_NSYS = """#!{py}
import os, sys
a = sys.argv
if a[1] == "profile":
    prefix = a[a.index("-o") + 1]
    open(prefix + ".nsys-rep", "w").write("report")
    open(os.path.join(os.path.dirname(prefix), "mc_env.txt"), "w").write(os.environ.get("GGML_CUDA_PQ2_MULTICOL", ""))
    assert "HF_TOKEN" not in os.environ and "RUNPOD_API_KEY" not in os.environ
elif a[1] == "stats":
    rep = a[-1]
    on = open(os.path.join(os.path.dirname(rep), "mc_env.txt")).read() == "1"
    print("Time (%),Total Time (ns),Instances,Avg (ns),Name")
    print('50,500,12,40,"void mul_mat_vec_pq2_0_mc<(int)4>(const void *, const void *, float *, int, int, int, int, int)"' if on else
          '50,500,12,40,"void mul_mat_vec_q<(ggml_type)142, (int)4, (bool)0, (bool)0>(const void *, const void *, const int *, float *, int)"')
"""


@pytest.fixture
def pod(tmp_path, monkeypatch):
    b = tmp_path / "bin"
    b.mkdir()
    _script(b / "llama-bench", FAKE_BENCH)
    _script(b / "test-backend-ops", FAKE_TBO)
    _script(tmp_path / "nsys", FAKE_NSYS)
    (tmp_path / "pq2.gguf").write_bytes(b"gguf")
    monkeypatch.setenv("CR_RESULTS_DIR", str(tmp_path / "results"))
    monkeypatch.setenv("CR_GPU_TAG", "rtx5090")
    monkeypatch.setenv("CR_LLAMA_BIN", str(b))
    monkeypatch.setenv("CR_NSYS", str(tmp_path / "nsys"))
    monkeypatch.setenv("HF_TOKEN", "hf_secretsecret123")
    monkeypatch.setenv("RUNPOD_API_KEY", "rpa_secretsecret456")
    monkeypatch.delenv("GGML_CUDA_PQ2_MULTICOL", raising=False)
    monkeypatch.setattr(mc.e4_pdl, "model_path", lambda name, notes: (str(tmp_path / "pq2.gguf"), {"file": "pq2.gguf"}))
    monkeypatch.setattr(mc.e4_pdl, "smi_sample", lambda: None)
    return tmp_path


def _result(tmp, exp, gpu):
    files = sorted((tmp / "results" / exp / gpu).glob("2*.json"))
    assert len(files) == 1
    return json.loads(files[0].read_text())


def test_multicol_end_to_end_with_fakes(pod):
    assert mc.main(["--budget-min", "10"]) == 0
    d = _result(pod, "r2b_mc", "rtx5090")["data"]
    assert [t["config"] for t in d["tbo"]] == ["on", "on_pdl0", "off"] and all(t["ok"] and t["passed"] == 1 == t["total"] for t in d["tbo"])
    assert d["tbo"][0]["env"] == {mc.ENV_VAR: "1"} and d["tbo"][0]["cmd"][-2:] == ["-p", "type_a=pq2_0,"]
    p = d["path"]
    assert p["on_uses_mc_kernel"] is True and p["off_uses_mc_kernel"] is False and p["configs"]["on"]["mc_widths"] == [4]
    assert p["off_uses_generic_w4"] is True and p["on_uses_generic_w4"] is False and not any("not on exactly" in n for n in d["notes"])
    assert p["configs"]["on"]["files_deleted"] == ["mc.nsys-rep", "mc_env.txt"] and not list(pod.rglob("*.nsys-rep"))      # deleted with the private dir
    g = {(x["n"]): x["gain_pct"] for x in d["perf"]["gain"]}
    assert g == {1: pytest.approx(0.0), 3: pytest.approx(100.0), 4: pytest.approx(100.0), 8: pytest.approx(100.0)}
    s = d["summary"]
    assert [x["config"] for x in d["bench"][:2]] == ["off", "off"] and len(d["bench"]) == 2 * 3 * (9 + 1)      # 2 rounds x 3 configs x (9 widths + tg)
    assert d["bench"][-1]["config"] == "off"                                                                    # round 1 runs the configs in reverse
    assert s["pp"]["4"]["gain_pct_on"] == pytest.approx(100.0) and s["pp"]["2"]["gain_pct_on"] == pytest.approx(0.0)
    assert s["pp"]["4"]["ms"]["off"] == pytest.approx(10.0) and s["pp"]["4"]["gain_pct_on_pdl0"] == pytest.approx(80.0)
    assert s["pp"]["4"]["rounds"] == [0, 1] and s["tg"]["1"]["tok_s"]["off"] == 140.0
    assert any("width 4" in n for n in d["notes"]) and any("ggml_cuda_pdl_lc" in n for n in d["notes"])
    assert "hf_secretsecret123" not in json.dumps(d) and not (pod / "results" / "r2b_mc" / "rtx5090" / "raw" / "partial.json").exists()


def test_multicol_failures_are_reported_and_a_blind_filter_runs_the_whole_op(pod, monkeypatch):
    monkeypatch.setenv("FAKE_TBO_FAIL", "1")
    assert mc.main(["--parts", "t", "--budget-min", "6"]) == 1
    d = _result(pod, "r2b_mc", "rtx5090")["data"]
    assert not any(t["ok"] for t in d["tbo"]) and d["tbo"][0]["n_fail"] == 1 and d["tbo"][0]["fail_lines"]
    assert any("reported failures" in n for n in d["notes"])


def test_multicol_failure_with_the_env_set_stops_before_any_benchmark(pod, monkeypatch):
    monkeypatch.setenv("FAKE_TBO_FAIL", "1")
    assert mc.main(["--budget-min", "10"]) == 1
    d = _result(pod, "r2b_mc", "rtx5090")["data"]
    assert d["path"] is None and d["perf"] is None and d["bench"] == []
    assert any(n.startswith("stop: test-backend-ops did not pass") and "on, on_pdl0" in n for n in d["notes"])
    assert any("FAIL MUL_MAT" in ln for ln in d["tbo"][0]["problem_lines"]) and "0/1 tests passed" in d["tbo"][0]["out_tail"]
    assert "\x1b" not in d["tbo"][0]["out_tail"] and "hf_secretsecret123" not in json.dumps(d)


def test_multicol_filter_that_matches_nothing_falls_back_to_the_whole_op(pod, monkeypatch):
    monkeypatch.setenv("FAKE_NO_FILTER", "1")
    assert mc.main(["--parts", "t", "--budget-min", "6"]) == 0
    d = _result(pod, "r2b_mc", "rtx5090")["data"]
    assert all(t["cmd"][-1] == "CUDA0" and t["filter"] is None and t["total"] == 1 for t in d["tbo"])
    assert any("MUL_MAT run whole" in n for n in d["notes"])


def test_multicol_without_nsys_skips_only_the_path_probe(pod, monkeypatch):
    monkeypatch.setenv("CR_NSYS", str(pod / "missing"))
    assert mc.main(["--parts", "kb", "--budget-min", "6"]) == 0
    d = _result(pod, "r2b_mc", "rtx5090")["data"]
    assert "nsys not found" in d["path"]["error"] and d["summary"]["pp"]


def test_multicol_dry_run_and_args(capsys):
    assert mc.main(["--dry-run"]) == 0
    plan = json.loads(capsys.readouterr().out)
    assert plan["exp"] == "r2b_mc" and plan["order"] == [["off", "on", "on_pdl0"], ["on_pdl0", "on", "off"]]
    assert plan["bench_pp_cmd"][plan["bench_pp_cmd"].index("-p") + 1] == "1,2,3,4,5,6,7,8,9"
    with pytest.raises(SystemExit):
        mc.parse_args(["--parts", "x"])


# ---------------------------------------------------------------- r2b_spec: configs and prompts

def test_k4_is_the_r2a_command_flag_for_flag_and_the_others_differ_where_intended():
    cfgs = {c["name"]: c for c in sp.matrix_configs()}
    drafts = {"bf16": "/d16.gguf", "q8_0": "/d8.gguf"}
    r2 = {c["name"]: c for c in r2a.matrix_configs()}
    assert list(cfgs) == sp.CONFIG_NAMES == ["base", "k4", "k4_q8", "combo"]
    for n in ("base", "k4"):
        assert sp.config_cmd("/b", "/t.gguf", drafts, cfgs[n]) == r2a.config_cmd("/b", "/t.gguf", "/d16.gguf", r2[n])
    q8 = sp.config_cmd("/b", "/t.gguf", drafts, cfgs["k4_q8"])
    assert q8 == [("/d8.gguf" if a == "/d16.gguf" else a) for a in sp.config_cmd("/b", "/t.gguf", drafts, cfgs["k4"])]
    combo = sp.config_cmd("/b", "/t.gguf", drafts, cfgs["combo"])
    assert combo[combo.index("--spec-type") + 1] == "ngram-mod,draft-simple" and combo[combo.index("--spec-draft-n-max") + 1] == "4"
    assert combo[combo.index("-md") + 1] == "/d16.gguf" and "--spec-ngram-mod-n-match" not in combo       # ngram-mod fork defaults
    for c in cfgs.values():
        cmd = sp.config_cmd("/b", "/t.gguf", drafts, c)
        assert cmd[cmd.index("-lv") + 1] == "4" and cmd[cmd.index("-c") + 1] == "16384" and cmd[cmd.index("-np") + 1] == "1"
    assert "-md" not in sp.config_cmd("/b", "/t.gguf", drafts, cfgs["base"])
    # the probe adds ngram-mod with the fork defaults and runs at -lv 5
    probe = {c["name"]: c for c in sp.probe_configs()}
    assert list(probe) == sp.PROBE_NAMES and "ngram_mod" in probe
    pc = sp.config_cmd("/b", "/t.gguf", drafts, probe["ngram_mod"])
    assert pc[-4:] == ["--spec-type", "ngram-mod", "-lv", "5"]
    assert all(sp.config_cmd("/b", "/t.gguf", drafts, c)[-2:] == ["-lv", "5"] for c in probe.values())


def test_fork_accepts_a_type_list_and_the_source_facts_hold():
    arg = (REPO / "common" / "arg.cpp").read_text()
    assert "comma-separated list of types of speculative decoding" in arg and "string_split<std::string>(value, ',')" in arg
    com = (REPO / "common" / "common.h").read_text()
    need = com[com.index("uint32_t need_n_rs_seq()"):][:700]
    assert "COMMON_SPECULATIVE_TYPE_DRAFT_SIMPLE" in need and "return needs_rs_seq ? draft.n_max : 0u;" in need and "NGRAM" not in need
    spec = (REPO / "common" / "speculative.cpp").read_text()
    init = spec[spec.index("add_config_if_enabled(COMMON_SPECULATIVE_TYPE_NGRAM_SIMPLE)"):]
    assert init.index("COMMON_SPECULATIVE_TYPE_NGRAM_MOD") < init.index("COMMON_SPECULATIVE_TYPE_DRAFT_SIMPLE")         # n-gram types come first
    assert 'SPC_DBG("truncating draft to %d tokens\\n", dp.n_max);' in spec and 'SPC_DBG("called impl %s, hist size = %zu, call_count = %zu, gen = %zu\\n"' in spec
    srv = (REPO / "tools" / "server" / "server-context.cpp").read_text()
    assert "/* .n_max    = */ n_draft_max," in srv                                    # dp.n_max is the room in the slot, not --spec-draft-n-max
    assert "draft.size() > llama_n_rs_seq(ctx_tgt)" in srv and "n_rollback > llama_n_rs_seq(ctx_tgt)" in srv


def test_subset_is_the_balanced_prefix_of_the_r2a_order():
    rows = make_prompts.load_longctx(make_prompts.R2A_OUT)
    order = make_prompts.shuffled(rows, r2a.SEED, stratify=True)
    sub = sp.balanced_subset(rows)
    assert [p["id"] for p in sub] == [p["id"] for p in order[:24]] and r2a._count(sub) == {c: 8 for c in make_prompts.R2A_CATEGORIES}
    assert sp.balanced_subset(rows) == sub and [p["id"] for p in sp.balanced_subset(rows, per_category=2)] == [p["id"] for p in order[:6]]
    assert {p["id"] for p in sub} <= {p["id"] for p in order[:53]}                    # R2a ran the first 53: each prompt has an R2a row
    assert sp.balanced_subset(rows, seed=1) != sub


def test_plan_fits_the_budget_with_all_four_configs_and_no_drop():
    rows = make_prompts.load_longctx(make_prompts.R2A_OUT)
    sub = sp.balanced_subset(rows)
    m_s = sp.part_budget_s(sp.DEFAULT_TOTAL_MIN * 60 - 30, "m", list("mp")) - r2a.DRIFT_RESERVE_S
    kept, plan = r2a.fit_configs(r2a.config_order(sp.matrix_configs()), sub, m_s, 8, drop_order=sp.DROP_ORDER)
    assert plan["n"] == 24 and plan["configs_dropped"] == [] and len(kept) == 4 and plan["est_s"] <= 0.9 * m_s
    # the combined type is costed like a draft-simple config, not like target alone
    c = {x["name"]: x for x in sp.matrix_configs()}
    assert r2a.config_cost_s(c["combo"], sub) == pytest.approx(r2a.config_cost_s(c["k4"], sub))
    assert r2a.config_cost_s(c["combo"], sub) > r2a.config_cost_s(c["base"], sub)
    _, tight = r2a.fit_configs(r2a.config_order(sp.matrix_configs()), sub, 300.0, 8, drop_order=sp.DROP_ORDER)
    assert tight["configs_dropped"][:2] == ["k4_q8", "combo"] and "base" not in tight["configs_dropped"] and "k4" not in tight["configs_dropped"]


def test_default_r2a_plan_is_unchanged_by_the_parameters():
    rows = make_prompts.load_longctx(make_prompts.R2A_OUT)
    order = make_prompts.shuffled(rows, r2a.SEED, stratify=True)
    full = r2a.config_order(r2a.matrix_configs())
    a, pa = r2a.fit_configs(full, order, 900.0, 17)
    b, pb = r2a.fit_configs(full, order, 900.0, 17, drop_order=r2a.DROP_ORDER)
    assert [c["name"] for c in a] == [c["name"] for c in b] and pa["n"] == pb["n"] and pa["configs_dropped"] == pb["configs_dropped"]


# ---------------------------------------------------------------- r2b_spec: draft models, log counts, anchor

def test_resolve_q8_checks_the_repo_listing():
    files = {"siblings": [{"rfilename": "Qwen3.5-0.8B-BF16.gguf"}, {"rfilename": "Qwen3.5-0.8B-Q8_0.gguf"}]}
    r = sp.resolve_q8(fetch=lambda url: files)
    assert r["exists"] is True and r["repo"] == "ggml-org/Qwen3.5-0.8B-GGUF" and r["file"] == "Qwen3.5-0.8B-Q8_0.gguf" and r["source"] == "api"
    r = sp.resolve_q8(fetch=lambda url: {"siblings": [{"rfilename": "Qwen3.5-0.8B-BF16.gguf"}]})
    assert r["exists"] is False and "quantize" in r["note"]

    def boom(url):
        raise OSError("no network")
    assert sp.resolve_q8(fetch=boom)["exists"] is None and sp.resolve_q8(offline=True)["exists"] is None


def test_q8_falls_back_to_llama_quantize(tmp_path, monkeypatch):
    b = tmp_path / "bin"
    b.mkdir()
    _script(b / "llama-quantize", "#!{py}\nimport shutil, sys\nshutil.copy(sys.argv[1], sys.argv[2])\nassert sys.argv[3] == 'Q8_0'\n")
    bf16 = tmp_path / "draft-bf16.gguf"
    bf16.write_bytes(b"gguf")
    monkeypatch.setattr(sp, "resolve_q8", lambda *a, **k: {"repo": "r", "file": "f", "exists": False})
    notes = []
    path, info = sp.q8_path(b, str(bf16), notes)
    assert path and os.path.exists(path) and info["origin"] == "llama-quantize" and path != str(bf16)
    (b / "llama-quantize").unlink()
    path, info = sp.q8_path(b, str(bf16), notes)
    assert path is None and info["origin"] is None and "not found" in info["error"] and any("unavailable" in n for n in notes)


MIXED = """spec common_specu: called impl ngram-mod, hist size = 5000, call_count = 1, gen = 64
slot x: created speculative checkpoint (pos_min = 1, pos_max = 17, n_tokens = 18, size = 1.0 MiB, draft = 0.0 MiB)
slot x: accepted 10/64 draft tokens (restore checkpoint)
slot x: restoring speculative checkpoint (pos_min = 1, pos_max = 17, size = 1.0 MiB)
slot x: accepted 10/11 draft tokens
spec common_specu: called impl draft-simple, hist size = 5011, call_count = 2, gen = 4
slot x: accepted  3/ 4 draft tokens
spec common_specu: truncating draft to 20 tokens
spec common_specu: called impl ngram-mod, hist size = 5014, call_count = 3, gen = 20
slot x: created speculative checkpoint (pos_min = 1, pos_max = 20, n_tokens = 21, size = 1.0 MiB, draft = 0.0 MiB)
slot x: accepted 20/20 draft tokens
spec common_specu: called impl ngram-mod, hist size = 5035, call_count = 4, gen = 4
slot x: accepted  2/ 4 draft tokens
"""


def test_count_mixed_events_separates_the_implementations_and_checks_the_checkpoint_rule():
    cfg = {c["name"]: c for c in sp.matrix_configs()}["combo"]
    ev = sp.count_mixed_events(MIXED, cfg)
    assert ev["n_rs_seq"] == 4 and ev["drafts"] == 4 and ev["created"] == 2
    ng, ds = ev["impls"]["ngram-mod"], ev["impls"]["draft-simple"]
    assert (ng["drafts"], ng["max_len"], ng["mean_len"], ng["longer_than_k"]) == (3, 64, (64 + 20 + 4) / 3, 2)
    assert (ds["drafts"], ds["longer_than_k"], ds["drafted"], ds["accepted"]) == (1, 0, 4, 3) and ds["r"] == pytest.approx(0.75)
    # the replay line after the restore (10/11) is not a draft: ngram-mod drafted 64 + 20 + 4 and accepted 10 + 20 + 2
    assert (ng["drafted"], ng["accepted"], ng["restored"]) == (88, 32, 1)
    assert ev["created"] == 2 == ev["drafts_longer_than_n_rs_seq"] and ev["created_matches_long_drafts"] is True
    assert ev["restored"] == 1 and ev["restore_per_checkpoint"] == pytest.approx(0.5) and ev["truncated"] == 1 and ev["truncated_to"] == [20]
    # with n_rs_seq 0 (no draft model) every draft is longer than the window: the same log would need 4 checkpoints, it has 2
    ng_cfg = {c["name"]: c for c in r2a.matrix_configs()}["ngram_mod"]
    ev0 = sp.count_mixed_events(MIXED.replace("draft-simple", "ngram-mod"), ng_cfg)
    assert ev0["n_rs_seq"] == 0 and ev0["drafts_longer_than_n_rs_seq"] == 4 and ev0["created_matches_long_drafts"] is False
    assert sp.count_mixed_events("", cfg)["created_matches_long_drafts"] is None
    # probe_events picks the mixed counter only for a list of types
    assert sp.probe_events(MIXED, cfg)["impls"] and "impls" not in sp.probe_events(MIXED, {c["name"]: c for c in r2a.matrix_configs()}["k4"])


def test_anchor_compares_with_the_same_ids_of_the_r2a_result(tmp_path):
    def row(i, tps, text, drafted=0, acc=0):
        return {"id": f"r{i:03d}", "category": "edit_diff", "tokens_predicted": 100, "predicted_per_second": tps, "predicted_ms": 1e5 / tps,
                "draft_n": drafted, "draft_n_accepted": acc, "text": text}
    old = {"base": {f"r{i:03d}": row(i, 100.0, f"t{i}") for i in range(4)}, "k4": {f"r{i:03d}": row(i, 120.0, f"t{i}", 40, 20) for i in range(4)}}
    runs = {"base": {"spec_type": "none", "rows": [row(i, 90.0, f"t{i}") for i in range(3)] + [row(9, 90.0, "t9")]},
            "k4": {"spec_type": "draft-simple", "rows": [row(i, 108.0, "t0" if i == 0 else "x", 40, 22) for i in range(3)]}}
    a = sp.anchor(runs, old)
    assert a["base"]["n"] == 3 and a["base"]["ratio"] == pytest.approx(0.9) and a["base"]["text_equal_frac"] == 1.0         # r009 has no R2a row
    assert a["k4"]["ratio"] == pytest.approx(0.9) and a["k4"]["text_equal_frac"] == pytest.approx(1 / 3) and a["k4"]["r_now"] == pytest.approx(0.55)
    assert sp.anchor({}, old) == {}
    d = tmp_path / "r2a" / "pro6000-ws"
    d.mkdir(parents=True)
    (d / "20261014T000000Z.json").write_text(json.dumps({"data": {"matrix": {"runs": {"base": {"rows": [row(0, 100.0, "t0")]}}}}}))
    (d / "20261014T010000Z.json").write_text(json.dumps({"data": {"matrix": {"runs": {"base": {"rows": [row(1, 100.0, "t1")]}}}}}))
    runs_old, path = sp.load_r2a_runs("pro6000-ws", tmp_path / "r2a")
    assert list(runs_old["base"]) == ["r001"] and path.endswith("010000Z.json") and sp.load_r2a_runs("rtx5090", tmp_path / "r2a") == ({}, None)


def test_committed_r2a_result_has_the_rows_to_anchor_to():
    old, path = sp.load_r2a_runs("pro6000-ws")
    assert path and {"base", "k4"} <= set(old)
    ids = {p["id"] for p in sp.balanced_subset(make_prompts.load_longctx(make_prompts.R2A_OUT))}
    assert ids <= set(old["base"]) and ids <= set(old["k4"])


# ---------------------------------------------------------------- r2b_spec end to end with a fake server

FAKE_SERVER = """#!{py}
import json, os, sys
from http.server import BaseHTTPRequestHandler, HTTPServer
a = sys.argv
port = int(a[a.index("--port") + 1])
spec = a[a.index("--spec-type") + 1] if "--spec-type" in a else "none"
lv = int(a[a.index("-lv") + 1]) if "-lv" in a else 3
draft = a[a.index("-md") + 1] if "-md" in a else ""
combo = "," in spec
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
            return self._send({{"prompt": "T:" + body["messages"][-1]["content"][-60:] + ("|tools" if body.get("tools") else "")}})
        if self.path == "/tokenize":
            return self._send({{"tokens": list(range(len(body["content"])))}})
        n = body["n_predict"]
        speed = 100.0 if spec == "none" else (140.0 if combo else (125.0 if "q8" in draft else 120.0))
        t = {{"predicted_per_second": speed, "predicted_ms": 1e3 * n / speed, "predicted_n": n, "prompt_n": 3000, "prompt_ms": 1000.0}}
        if spec != "none":
            t.update(draft_n=2 * n, draft_n_accepted=n // 2)
        if lv >= 5 and combo:
            print("spec common_specu: called impl ngram-mod, hist size = 5000, call_count = 1, gen = 30", flush=True)
            print("slot x: created speculative checkpoint (pos_min = 1)", flush=True)
            print("slot x: accepted 30/30 draft tokens", flush=True)
            print("spec common_specu: called impl draft-simple, hist size = 5030, call_count = 2, gen = 4", flush=True)
            print("slot x: accepted  2/ 4 draft tokens", flush=True)
        self._send({{"content": "out:" + body["prompt"], "tokens_predicted": n, "timings": t, "stop_type": "limit"}})
rs = 4 if "draft-simple" in spec else 0
if spec != "none":
    for _ in range(2 if rs else 1):
        print("llama_context: n_rs_seq              = %d" % rs, flush=True)
        print("cmn  common_conte: the context " + ("supports bounded" if rs else "does not support") + " partial sequence removal", flush=True)
    if not rs:
        print("srv    load_model: speculative decoding will use checkpoints", flush=True)
else:
    print("llama_context: n_rs_seq              = 0", flush=True)
HTTPServer(("127.0.0.1", port), H).serve_forever()
"""


@pytest.fixture
def spod(tmp_path, monkeypatch):
    b = tmp_path / "bin"
    b.mkdir()
    _script(b / "llama-server", FAKE_SERVER)
    files = {"pq2": "target.gguf", "draft_bf16": "draft-bf16.gguf"}
    for n in list(files.values()) + ["draft-q8.gguf"]:
        (tmp_path / n).write_bytes(b"gguf")
    monkeypatch.setenv("CR_RESULTS_DIR", str(tmp_path / "results"))
    monkeypatch.setenv("CR_GPU_TAG", "pro6000-ws")
    monkeypatch.setenv("CR_LLAMA_BIN", str(b))
    monkeypatch.setenv("HF_TOKEN", "hf_secretsecret123")
    monkeypatch.setenv("RUNPOD_API_KEY", "rpa_secretsecret456")
    monkeypatch.setattr(sp.e4_pdl, "model_path", lambda name, notes: (str(tmp_path / files[name]), {"file": files[name]}))
    monkeypatch.setattr(sp.e4_pdl, "smi_sample", lambda: None)
    monkeypatch.setattr(sp, "q8_path", lambda bin_dir, bf16, notes: (str(tmp_path / "draft-q8.gguf"), {"origin": "hf", "file": "Qwen3.5-0.8B-Q8_0.gguf"}))
    monkeypatch.setattr(sp, "load_r2a_runs", lambda gpu, root=None: ({}, None))
    return tmp_path


def test_r2b_spec_end_to_end_with_fakes(spod):
    assert sp.main(["--per-category", "2", "--budget-min", "10"]) == 0
    d = _result(spod, "r2b_spec", "pro6000-ws")["data"]
    m = d["matrix"]
    assert m["config_order"][0] == "base" and sorted(m["config_order"]) == sorted(sp.CONFIG_NAMES) and m["configs_dropped"] == []
    assert list(m["runs"]) == m["config_order"] + ["base_end"] and m["plan"]["n"] == 6
    ids = d["config"]["prompt_ids"]
    assert len(ids) == 6 and all([r["id"] for r in m["runs"][c]["rows"]] == ids for c in sp.CONFIG_NAMES)           # same prompts, same order, every config
    cmds = {n: r["cmd"] for n, r in m["runs"].items()}
    assert cmds["k4_q8"][cmds["k4_q8"].index("-md") + 1].endswith("draft-q8.gguf") and cmds["k4"][cmds["k4"].index("-md") + 1].endswith("draft-bf16.gguf")
    assert cmds["combo"][cmds["combo"].index("--spec-type") + 1] == "ngram-mod,draft-simple" and "-md" not in cmds["base"]
    s = m["summary"]
    assert s["k4"]["all"]["speedup"] == pytest.approx(1.2) and s["k4_q8"]["all"]["speedup"] == pytest.approx(1.25) and s["combo"]["all"]["speedup"] == pytest.approx(1.4)
    assert s["combo"]["all"]["speedup_ci95"] and s["combo"]["all"]["alpha"] is None and s["k4"]["all"]["alpha"] is not None
    assert s["combo"]["n_ge_min"] is False and s["combo"]["n_by_category"] == {c: 2 for c in make_prompts.R2A_CATEGORIES}       # exploratory at this n
    assert s["combo"]["checks"]["rollback_path"] == "in_batch" and s["combo"]["checks"]["n_rs_seq"] == [4, 4] and s["combo"]["checks"]["path_consistent"] is True
    assert m["drift"]["n"] == r2a.DRIFT_N and m["drift"]["ratio"] == pytest.approx(1.0)
    p = d["probe"]
    assert set(p["configs"]) == set(sp.PROBE_NAMES) and p["lv"] == "5" and p["llama_trace"] == "1"
    ev = p["configs"]["combo"]["events"]
    assert ev["n_rs_seq"] == 4 and set(ev["impls"]) == {"ngram-mod", "draft-simple"} and ev["impls"]["ngram-mod"]["longer_than_k"] == ev["created"] == 4
    assert ev["created_matches_long_drafts"] is True and p["configs"]["ngram_mod"]["events"]["path"] in ("checkpoint", "in_batch")
    assert d["anchor"]["configs"] == {} and any("no R2a result" in n for n in d["notes"])
    assert d["source_facts"].startswith("need_n_rs_seq()")
    assert "hf_secretsecret123" not in json.dumps(d) and "rpa_secretsecret456" not in json.dumps(d)
    raw = spod / "results" / "r2b_spec" / "pro6000-ws" / "raw"
    assert not (raw / "partial.json").exists() and not list(raw.glob("*probe*")) and {"server-base.log", "server-combo.log"} <= {f.name.split("-", 1)[1] for f in raw.glob("*-server-*.log")}


def test_r2b_spec_without_a_q8_draft_drops_only_that_config(spod, monkeypatch):
    monkeypatch.setattr(sp, "q8_path", lambda bin_dir, bf16, notes: (None, {"origin": None, "error": "gone"}))
    assert sp.main(["--parts", "m", "--per-category", "2", "--budget-min", "8"]) == 0
    d = _result(spod, "r2b_spec", "pro6000-ws")["data"]
    assert sorted(d["matrix"]["config_order"]) == ["base", "combo", "k4"] and any("k4_q8 dropped" in n for n in d["notes"])


def test_r2b_spec_unhealthy_host_skips_the_matrix_but_runs_the_probe(spod, monkeypatch):
    monkeypatch.setattr(sp.r1, "host_probe", lambda *a: {"smi": {"n": 3}})
    monkeypatch.setattr(sp.r1, "host_health", lambda *a, **k: ("SM clock 100 MHz < 40% of max 3000", None))
    assert sp.main(["--per-category", "2", "--budget-min", "8"]) == 1
    d = _result(spod, "r2b_spec", "pro6000-ws")["data"]
    assert d["matrix"] is None and d["host"]["parts_skipped"] == ["m"] and d["probe"]["configs"]


def test_r2b_spec_dry_run_and_args(capsys):
    assert sp.main(["--dry-run"]) == 0
    plan = json.loads(capsys.readouterr().out)
    assert plan["exp"] == "r2b_spec" and plan["per_category"] == 8 and len(plan["prompt_ids"]) == 24 and plan["matrix"]["plan"]["configs_dropped"] == []
    assert plan["probe"]["configs"] == sp.PROBE_NAMES and plan["q8_0_draft"]["file"] == "Qwen3.5-0.8B-Q8_0.gguf"
    with pytest.raises(SystemExit):
        sp.parse_args(["--configs", "k4"])                        # base is required
    with pytest.raises(SystemExit):
        sp.parse_args(["--configs", "base,k9"])
    assert sp.parse_args(["--quick"]).per_category == 2
