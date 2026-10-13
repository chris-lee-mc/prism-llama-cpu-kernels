import json
import stat
import sys

import pytest

from exps import e4_pdl as ep

TBO_OUT = """\
Testing 2 devices

Backend 1/2: CUDA0
  Device description: NVIDIA RTX PRO 6000 Blackwell
  MUL_MAT(type_a=pq2_0,type_b=f32,m=16,n=1,k=256,bs=[1,1],nr=[1,1],per=[0,1,2,3],k_v=0,o=1): \033[1;32mOK\033[0m
  MUL_MAT(type_a=ptq1_0,type_b=f32,m=16,n=3,k=256,bs=[1,1],nr=[1,1],per=[0,1,2,3],k_v=0,o=1): [MUL_MAT] NMSE = 0.1 > 0.0005 \033[1;31mFAIL\033[0m
  31/32 tests passed
  Backend CUDA0: \033[1;31mFAIL\033[0m
Backend 2/2: CPU
  Skipping CPU backend
1/2 backends passed
\033[1;31mFAIL\033[0m
"""

FAKE_BENCH = """#!{py}
import json, os, sys
a = sys.argv
depths = [int(x) for x in a[a.index("-d") + 1].split(",")]
n = int(a[a.index("-n") + 1])
with open(os.environ["FAKE_LOG"], "a") as f:
    f.write(json.dumps({{"pdl": os.environ.get("GGML_CUDA_PDL"), "pf": os.environ.get("GGML_CUDA_PDL_PREFETCH"),
                        "m": a[a.index("-m") + 1]}}) + "\\n")
base = 100.0 + (5.0 if os.environ.get("GGML_CUDA_PDL") == "1" else 0.0) + float(os.environ.get("GGML_CUDA_PDL_PREFETCH", "0"))
print(json.dumps([{{"n_prompt": 0, "n_gen": n, "n_depth": d, "avg_ts": base - d / 10000.0, "stddev_ts": 0.5}} for d in depths]))
"""

FAKE_TBO = """#!{py}
import os
print("  MUL_MAT(type_a=q8_0,...): OK")
print("  12/12 tests passed")
"""


def test_configs_and_order():
    cfgs = ep.configs([1, 4])
    assert [c for c, _ in cfgs] == ["pdl0", "pdl1", "pf1", "pf4", "pdl0_pf4"]
    assert cfgs[0][1] == {"GGML_CUDA_PDL": "0"} and cfgs[1][1] == {"GGML_CUDA_PDL": "1"}
    assert cfgs[2][1] == {"GGML_CUDA_PDL": "1", "GGML_CUDA_PDL_PREFETCH": "1"}
    assert cfgs[4][1] == {"GGML_CUDA_PDL": "0", "GGML_CUDA_PDL_PREFETCH": "4"}  # control: same kernels, nothing to overlap
    assert [c for c, _ in ep.round_order(cfgs, 1)] == ["pdl0_pf4", "pf4", "pf1", "pdl1", "pdl0"]


def test_bench_cmd_flags():
    cmd = ep.bench_cmd("/bin", "/m.gguf", [0, 32768], 5)
    assert cmd == ["/bin/llama-bench", "-m", "/m.gguf", "-ngl", "99", "-fa", "on", "-p", "0", "-n", "128", "-d", "0,32768",
                   "-ctk", "f16", "-ctv", "f16", "-ub", "2048", "-r", "5", "-o", "json"]


def test_tbo_cmd_and_filter_regex():
    import pathlib
    import re
    cmd = ep.tbo_cmd(pathlib.Path("/bin"))
    assert cmd[:2] == ["/bin/test-backend-ops", "test"] and cmd[cmd.index("-o") + 1] == "MUL_MAT,MUL_MAT_ID,MUL_MAT_VEC_FUSION"
    rx = re.compile(cmd[cmd.index("-p") + 1])
    assert rx.search("type_a=pq2_0,type_b=f32,m=16") and rx.search("type_a=ptq1_0,type_b=f32")
    assert rx.search("type_a=q8_0,type_b=f32") and not rx.search("type_a=tq2_0,type_b=f32")
    # the fused up/gate mat-vec test names its type "type=" (VARS_TO_STR13(type, glu_op, ...))
    assert rx.search("type=ptq1_0,glu_op=SWIGLU,m=1,n=64,k=196480") and rx.search("type=q8_0,glu_op=GEGLU,m=1")
    assert rx.search("type=pq2_0,glu_op=SWIGLU") and not rx.search("type=q4_0,glu_op=SWIGLU")


def test_parse_tbo():
    r = ep.parse_tbo(TBO_OUT)
    assert r["passed"] == 31 and r["total"] == 32
    assert len(r["fail_lines"]) == 1 and "ptq1_0" in r["fail_lines"][0]


def test_tg_rows_by_depth():
    entries = [{"n_gen": 128, "n_depth": 0, "avg_ts": 150.0, "stddev_ts": 1.0},
               {"n_gen": 128, "n_depth": 32768, "avg_ts": 120.0}, {"n_gen": 0, "n_prompt": 512, "avg_ts": 1.0}]
    assert ep.tg_rows(entries) == {0: {"tok_s": 150.0, "stddev": 1.0}, 32768: {"tok_s": 120.0, "stddev": 0.0}}


def test_summarize_gains():
    rows = [{"model": "pq2", "depth": 0, "config": c, "tok_s": v} for c, v in
            [("pdl0", 100.0), ("pdl0", 102.0), ("pdl1", 110.0), ("pf1", 121.0), ("pf1", None)]]
    s = ep.summarize(rows, ["pdl0", "pdl1", "pf1"])["pq2"]["0"]
    assert s["tok_s"]["pdl0"] == pytest.approx(101.0) and s["n"] == {"pdl0": 2, "pdl1": 1, "pf1": 1}
    assert s["pdl1_vs_pdl0_pct"] == pytest.approx(100 * (110 / 101 - 1))
    assert s["pf1_vs_pdl1_pct"] == pytest.approx(10.0)
    assert "pf1_vs_pdl1_pct=+10.0%" in ep.verdict_lines({"pq2": {"0": s}})[0]


def test_summarize_control():
    rows = [{"model": "pq2", "depth": 0, "config": c, "tok_s": v} for c, v in
            [("pdl0", 100.0), ("pdl1", 110.0), ("pf4", 121.0), ("pdl0_pf4", 98.0)]]
    s = ep.summarize(rows, ["pdl0", "pdl1", "pf4", "pdl0_pf4"])["pq2"]["0"]
    assert s["pdl0_pf4_vs_pdl0_pct"] == pytest.approx(-2.0) and s["pf4_vs_pdl1_pct"] == pytest.approx(10.0)
    assert "pdl0_pf4_vs_pdl0_pct" not in ep.summarize(rows[:3], ["pdl0", "pdl1", "pf4", "pdl0_pf4"])["pq2"]["0"]
    assert "pdl0_pf4_vs_pdl0_pct=-2.0%" in ep.verdict_lines({"pq2": {"0": s}})[0]


def test_summarize_uses_complete_rounds_only():
    # round 1 lost pf1 (budget cut): its pdl0/pdl1 results must not skew the pair
    rows = [{"model": "m", "depth": 0, "config": c, "round": r, "tok_s": v} for c, r, v in
            [("pdl0", 0, 100.0), ("pdl1", 0, 110.0), ("pf1", 0, 121.0), ("pf1", 1, 130.0), ("pdl1", 1, 150.0)]]
    s = ep.summarize(rows, ["pdl0", "pdl1", "pf1"])["m"]["0"]
    assert s["rounds"] == [0] and s["n"] == {"pdl0": 1, "pdl1": 1, "pf1": 1}
    assert s["tok_s"]["pdl1"] == pytest.approx(110.0) and s["pf1_vs_pdl1_pct"] == pytest.approx(10.0)
    s = ep.summarize(rows[:2] + rows[3:], ["pdl0", "pdl1", "pf1"])["m"]["0"]  # no round has all configs
    assert s["rounds"] == [] and "pdl1_vs_pdl0_pct" not in s


SMI_LINE = "2400, 2430, 14001, 14001, 420.5, 600.00, 600.00, 71\n"


def test_parse_smi_and_summary():
    d = ep.parse_smi(SMI_LINE)
    assert d["clocks.sm"] == 2400 and d["power.limit"] == 600 and d["temperature.gpu"] == 71
    assert ep.parse_smi("2400, [N/A], 1, 1, 1, 1, 1, 1\n")["clocks.max.sm"] is None
    assert ep.parse_smi("") is None and ep.parse_smi("1, 2\n") is None
    during = [dict(d, **{"clocks.sm": 2000}), dict(d, **{"clocks.sm": 2200}), dict(d, **{"clocks.sm": 2400})]
    s = ep.smi_summary(d, during, d)
    assert s["sm_mhz_med"] == 2200 and s["sm_mhz_min"] == 2000 and s["sm_mhz_max"] == 2430 and s["n"] == 3
    assert ep.health_problem(s) is None and ep.smi_summary(None, [], None) == {}


def test_health_problem():
    ok = {"sm_mhz_med": 2000, "sm_mhz_max": 2430, "mem_mhz_med": 14001, "mem_mhz_max": 14001, "power_cap_w": 600, "power_default_w": 600}
    assert ep.health_problem(ok) is None and ep.health_problem({}) is None
    assert "SM clock" in ep.health_problem(dict(ok, sm_mhz_med=620, sm_mhz_max=2430))  # the degraded E3d host
    assert "mem clock" in ep.health_problem(dict(ok, mem_mhz_med=3000))
    assert "power cap" in ep.health_problem(dict(ok, power_cap_w=350))


def test_args():
    a = ep.parse_args([])
    assert a.models == list(ep.MODELS) == ["pq2", "ptq1", "draft_q8_0", "draft_bf16"] and a.depths == [0, 32768] and a.prefetch == [1, 4] and a.reps == 5
    with pytest.raises(SystemExit):
        ep.parse_args(["--reps", "3"])
    with pytest.raises(SystemExit):
        ep.parse_args(["--prefetch", "0"])
    with pytest.raises(SystemExit):
        ep.parse_args(["--models", "pq2,nope"])
    q = ep.parse_args(["--quick"])
    assert q.models == ["draft_q8_0"] and q.rounds == 1 and q.prefetch == [1]


def test_dry_run_plan(capsys):
    assert ep.main(["--dry-run"]) == 0
    plan = json.loads(capsys.readouterr().out)
    assert plan["exp"] == "e4_pdl" and plan["n_bench"] == 2 * 4 * 5
    assert plan["order"][1] == list(reversed(plan["order"][0])) and plan["tbo"]["prefetch"] == [0, 4]


def _script(path, text):
    path.write_text(text.format(py=sys.executable))
    path.chmod(path.stat().st_mode | stat.S_IXUSR)


def test_run_with_fake_binaries(tmp_path, monkeypatch):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    _script(bin_dir / "llama-bench", FAKE_BENCH)
    _script(bin_dir / "test-backend-ops", FAKE_TBO)
    log = tmp_path / "log.jsonl"
    monkeypatch.setenv("FAKE_LOG", str(log))
    monkeypatch.setenv("CR_RESULTS_DIR", str(tmp_path / "results"))
    monkeypatch.setenv("CR_GPU_TAG", "pro6000-ws")
    monkeypatch.setenv("CR_LLAMA_BIN", str(bin_dir))
    monkeypatch.setenv("GGML_CUDA_PDL_PREFETCH", "9")  # must not leak into the pdl0/pdl1 configs
    monkeypatch.setattr(ep, "model_path", lambda m, notes: (f"/{m}.gguf", {"repo": "r"}))
    monkeypatch.setattr(ep, "smi_sample", lambda: ep.parse_smi(SMI_LINE))
    monkeypatch.setattr(ep, "SMI_PERIOD_S", 0.01)
    assert ep.main(["--models", "pq2,draft_q8_0", "--prefetch", "1", "--rounds", "2"]) == 0
    calls = [json.loads(x) for x in log.read_text().splitlines()]
    assert [(c["pdl"], c["pf"]) for c in calls[:4]] == [("0", None), ("1", None), ("1", "1"), ("0", "1")]
    # both rounds of a model back to back, round 2 reversed; then the next model
    assert [(c["m"], c["pdl"], c["pf"]) for c in calls[4:8]] == [("/pq2.gguf", "0", "1"), ("/pq2.gguf", "1", "1"),
                                                                  ("/pq2.gguf", "1", None), ("/pq2.gguf", "0", None)]
    assert [c["m"] for c in calls] == ["/pq2.gguf"] * 8 + ["/draft_q8_0.gguf"] * 8
    files = sorted((tmp_path / "results" / "e4_pdl" / "pro6000-ws").glob("*.json"))
    assert len(files) == 1
    data = json.loads(files[0].read_text())["data"]
    assert len(data["rows"]) == 2 * 2 * 4 * 2
    assert [t["prefetch"] for t in data["tbo"]] == [0, 4] and all(t["ok"] for t in data["tbo"])
    s = data["summary"]["pq2"]["32768"]
    assert s["pdl1_vs_pdl0_pct"] == pytest.approx(100 * (105 - 3.2768) / (100 - 3.2768) - 100)
    assert s["pf1_vs_pdl1_pct"] > 0 and s["rounds"] == [0, 1]
    assert s["pdl0_pf1_vs_pdl0_pct"] == pytest.approx(100 * (101 - 3.2768) / (100 - 3.2768) - 100)
    assert data["rows"][0]["smi"]["sm_mhz_max"] == 2430 and data["rows"][0]["smi"]["power_cap_w"] == 600
    assert any("addresses are in range" in n for n in data["notes"])


def test_run_aborts_on_degraded_host(tmp_path, monkeypatch):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    _script(bin_dir / "llama-bench", FAKE_BENCH)
    log = tmp_path / "log.jsonl"
    monkeypatch.setenv("FAKE_LOG", str(log))
    monkeypatch.setenv("CR_RESULTS_DIR", str(tmp_path / "results"))
    monkeypatch.setenv("CR_GPU_TAG", "pro6000-ws")
    monkeypatch.setenv("CR_LLAMA_BIN", str(bin_dir))
    monkeypatch.setattr(ep, "model_path", lambda m, notes: (f"/{m}.gguf", {"repo": "r"}))
    monkeypatch.setattr(ep, "smi_sample", lambda: ep.parse_smi("620, 2430, 14001, 14001, 300, 350, 600, 80\n"))
    monkeypatch.setattr(ep, "SMI_PERIOD_S", 0.01)
    assert ep.main(["--models", "pq2", "--no-tbo"]) == 1
    assert len(log.read_text().splitlines()) == 1
    data = json.loads(next((tmp_path / "results" / "e4_pdl" / "pro6000-ws").glob("*.json")).read_text())["data"]
    assert any(n.startswith("ABORT: unhealthy host") for n in data["notes"])


def test_run_without_nvidia_smi(tmp_path, monkeypatch):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    _script(bin_dir / "llama-bench", FAKE_BENCH)
    monkeypatch.setenv("FAKE_LOG", str(tmp_path / "log.jsonl"))
    monkeypatch.setenv("CR_RESULTS_DIR", str(tmp_path / "results"))
    monkeypatch.setenv("CR_GPU_TAG", "pro6000-ws")
    monkeypatch.setenv("CR_LLAMA_BIN", str(bin_dir))
    monkeypatch.setenv("PATH", str(tmp_path))  # no nvidia-smi
    monkeypatch.setattr(ep, "model_path", lambda m, notes: (f"/{m}.gguf", {"repo": "r"}))
    assert ep.main(["--models", "draft_q8_0", "--prefetch", "1", "--rounds", "1", "--no-tbo"]) == 0
