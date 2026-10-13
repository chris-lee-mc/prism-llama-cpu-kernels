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
    assert [c for c, _ in cfgs] == ["pdl0", "pdl1", "pf1", "pf4"]
    assert cfgs[0][1] == {"GGML_CUDA_PDL": "0"} and cfgs[1][1] == {"GGML_CUDA_PDL": "1"}
    assert cfgs[2][1] == {"GGML_CUDA_PDL": "1", "GGML_CUDA_PDL_PREFETCH": "1"}
    assert [c for c, _ in ep.round_order(cfgs, 1)] == ["pf4", "pf1", "pdl1", "pdl0"]


def test_bench_cmd_flags():
    cmd = ep.bench_cmd("/bin", "/m.gguf", [0, 32768], 5)
    assert cmd == ["/bin/llama-bench", "-m", "/m.gguf", "-ngl", "99", "-fa", "on", "-p", "0", "-n", "128", "-d", "0,32768",
                   "-ctk", "f16", "-ctv", "f16", "-ub", "2048", "-r", "5", "-o", "json"]


def test_tbo_cmd_and_filter_regex():
    import pathlib
    import re
    cmd = ep.tbo_cmd(pathlib.Path("/bin"))
    assert cmd[:2] == ["/bin/test-backend-ops", "test"] and cmd[cmd.index("-o") + 1] == "MUL_MAT,MUL_MAT_ID"
    rx = re.compile(cmd[cmd.index("-p") + 1])
    assert rx.search("type_a=pq2_0,type_b=f32,m=16") and rx.search("type_a=ptq1_0,type_b=f32")
    assert rx.search("type_a=q8_0,type_b=f32") and not rx.search("type_a=tq2_0,type_b=f32")


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


def test_args():
    a = ep.parse_args([])
    assert a.models == list(ep.MODELS) and a.depths == [0, 32768] and a.prefetch == [1, 4] and a.reps == 5
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
    assert plan["exp"] == "e4_pdl" and plan["n_bench"] == 2 * 4 * 4
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
    assert ep.main(["--models", "pq2,draft_q8_0", "--prefetch", "1", "--rounds", "2"]) == 0
    calls = [json.loads(x) for x in log.read_text().splitlines()]
    assert [(c["pdl"], c["pf"]) for c in calls[:3]] == [("0", None), ("1", None), ("1", "1")]
    assert [(c["pdl"], c["pf"]) for c in calls[6:9]] == [("1", "1"), ("1", None), ("0", None)]
    files = sorted((tmp_path / "results" / "e4_pdl" / "pro6000-ws").glob("*.json"))
    assert len(files) == 1
    data = json.loads(files[0].read_text())["data"]
    assert len(data["rows"]) == 2 * 2 * 3 * 2
    assert [t["prefetch"] for t in data["tbo"]] == [0, 4] and all(t["ok"] for t in data["tbo"])
    s = data["summary"]["pq2"]["32768"]
    assert s["pdl1_vs_pdl0_pct"] == pytest.approx(100 * (105 - 3.2768) / (100 - 3.2768) - 100)
    assert s["pf1_vs_pdl1_pct"] > 0
