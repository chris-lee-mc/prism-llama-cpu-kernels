import json
import pathlib
import shutil
import subprocess

import pytest

from exps import e4_bonsai as e4
from harness import counters

BENCH_OUT = """\
load_backend: loaded CUDA backend
[
  {
    "build_commit": "abc1234",
    "build_number": 1,
    "model_filename": "/m/Ternary-Bonsai-2-27B-PQ2_0.gguf",
    "model_type": "qwen35 27B PQ2_0",
    "n_gpu_layers": 99,
    "n_prompt": 0,
    "n_gen": 128,
    "n_depth": 32768,
    "type_k": "q8_0",
    "type_v": "q8_0",
    "test_time": "2026-10-13T00:00:00Z",
    "avg_ns": 1000000000,
    "stddev_ns": 5000000,
    "avg_ts": 121.5,
    "stddev_ts": 0.6,
    "samples_ns": [ 1000000000 ],
    "samples_ts": [ 121.5 ]
  }
]
"""


def test_bench_cmd_exact_flags():
    cmd = e4.bench_cmd("/bin", "/m.gguf", 32768, "q8_0")
    assert cmd == ["/bin/llama-bench", "-m", "/m.gguf", "-ngl", "99", "-fa", "on", "-p", "0", "-n", "128", "-d", "32768",
                   "-ctk", "q8_0", "-ctv", "q8_0", "-ub", "2048", "-r", "3", "-o", "json"]
    assert e4.bench_cmd("/bin", "/m.gguf", 131072, "f16")[cmd.index("-r") + 1] == "1"
    assert e4.reps_for(0) == 3 and e4.reps_for(32768) == 3 and e4.reps_for(32769) == 1 and e4.reps_for(262144) == 1


def test_ncu_commands():
    assert e4.bench_ncu_cmd("/bin", "/m.gguf", "f16") == [
        "/bin/llama-bench", "-m", "/m.gguf", "-ngl", "99", "-fa", "on", "-p", "0", "-n", "4", "-d", "0", "-ctk", "f16",
        "-ctv", "f16", "-ub", "2048", "-r", "1", "--no-warmup", "-o", "json"]
    p = e4.prof_cmd("/bin/cr_decode_prof", "/m.gguf", 32768, "q8_0")
    assert p[0] == "/bin/cr_decode_prof" and p[p.index("-d") + 1] == "32768" and p[p.index("-n") + 1] == "4"
    full = counters.build_ncu_cmd(p, extra=("--profile-from-start", "off"))
    assert "--cache-control" in full and full[full.index("--cache-control") + 1] == "none"
    assert "--profile-from-start" in full


def test_parse_bench_json_fixture():
    entries = e4.parse_bench_json(BENCH_OUT)
    assert len(entries) == 1
    tg = e4.tg_result(entries)
    assert tg["tok_s"] == 121.5 and tg["n_depth"] == 32768 and tg["type_k"] == "q8_0" and tg["n_gen"] == 128


def test_parse_bench_json_bad_input():
    assert e4.parse_bench_json("") == [] and e4.parse_bench_json("[ not json ]") == []
    assert e4.tg_result([{"n_gen": 0, "n_prompt": 512, "avg_ts": 1.0}]) is None


def test_bucket_config_on_kernel_names():
    names = {
        "hadamard": ["fwht_cuda", "fwht_cuda_smem", "fwht_quantize_q8_1"],
        "act_quant": ["quantize_q8_1", "quantize_mmq_q8_1"],
        "mul_mat_vec": ["mul_mat_vec_q", "mul_mat_vec_pq2_0_mc", "mul_mat_vec_ptq1_0_pt", "mul_mat_vec_f"],
        "flash_attn": ["flash_attn_ext_vec", "flash_attn_tile", "flash_attn_ext_f16"],
        "gated_delta_net": ["gated_delta_net_cuda", "gdn_precompute_exp"],
        "other": ["rms_norm_f32", "k_get_rows"],
    }
    rows = []
    for bucket, ks in names.items():
        for k in ks:
            rows.append({"id": len(rows), "name": k, "hit_sectors": 10.0, "miss_sectors": 10.0, "hit_rate": 0.5,
                         "dram_bytes": 1000.0, "time_ns": 2000.0})
    out = e4.bucket_summary(rows, tokens=2)
    for bucket, ks in names.items():
        assert out[bucket]["count"] == len(ks), bucket
    assert out["mul_mat_vec"]["us_per_token"] == pytest.approx(4 * 2000.0 / 1000.0 / 2)
    assert sum(v["time_share"] for v in out.values()) == pytest.approx(1.0)


def test_bw_bound_ratio_and_row():
    # 1.0e9 bytes in 1000 us = 1000 GB/s -> 0.5 of a 2000 GB/s peak
    assert e4.bw_bound_ratio(1.0e9, 1000.0, 2000.0) == pytest.approx(0.5)
    assert e4.bw_bound_ratio(None, 1000.0, 2000.0) is None
    ncu = {"summary": {"dram_bytes_per_token": 7.0e9, "us_per_token": 5000.0, "hit_rate": 0.1}, "buckets": {"x": 1}}
    row = e4.make_row("PQ2_0", 0, "f16", {"tok_s": 125.0, "stddev": 1.0}, ncu, 1792.0)
    assert row["us_per_token"] == pytest.approx(8000.0)
    assert row["bw_bound_ratio"] == pytest.approx(7.0e9 / (8000.0 * 1e3) / 1792.0)
    assert set(row) >= {"quant", "depth", "kv", "tg_tok_s", "us_per_token", "buckets", "dram_bytes_per_token", "bw_bound_ratio"}
    assert e4.make_row("PQ2_0", 1, "f16", None, None, None)["us_per_token"] is None


def test_verdict_line():
    rows = [{"depth": 0, "kv": "f16", "bw_bound_ratio": 0.62, "buckets": {
        "mul_mat_vec": {"time_share": 0.8}, "hadamard": {"time_share": 0.05}, "act_quant": {"time_share": 0.03}}}]
    assert "mul_mat_vec 80%" in e4.verdict(rows)
    assert e4.verdict([]) is None


def test_args_and_exp_names():
    a = e4.parse_args(["--quant", "PTQ1_0"])
    assert a.depths == [0, 32768, 131072, 262144] and a.kv == ["f16", "q8_0"] and a.ncu_depths == [0, 32768]
    assert e4.EXP_NAMES == {"PQ2_0": "e4_pq2", "PTQ1_0": "e4_ptq1"}
    with pytest.raises(SystemExit):
        e4.parse_args(["--kv", "q4_0"])


@pytest.mark.skipif(not shutil.which("g++"), reason="no g++")
def test_prof_tool_syntax():
    root = pathlib.Path(e4.REPO_ROOT)
    r = subprocess.run(["g++", "-std=c++17", "-fsyntax-only", f"-I{root / 'include'}", f"-I{root / 'ggml' / 'include'}",
                        str(e4.PROF_SRC)], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr


def test_dry_run_plan_has_all_depths(capsys):
    assert e4.main(["--dry-run", "--quant", "PQ2_0"]) == 0
    plan = json.loads(capsys.readouterr().out)
    assert len(plan["runs"]) == 8 and plan["buckets"] == e4.BUCKETS


def test_run_does_all_bench_rows_before_ncu(tmp_path, monkeypatch):
    import types
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (bin_dir / "llama-bench").write_text("")
    calls = []
    monkeypatch.setenv("CR_LLAMA_BIN", str(bin_dir))
    monkeypatch.setattr(e4.common, "gpu_tag", lambda a: "gpu")
    monkeypatch.setattr(e4.common, "counters_available", lambda a: (True, None))
    monkeypatch.setattr(e4.common, "write_partial", lambda *a, **k: None)
    monkeypatch.setattr(e4.common, "write_result", lambda exp, args, config, rows, notes, ok, **k: calls.append(("result", rows)))
    monkeypatch.setattr(e4, "download", lambda q, n: ("m.gguf", {}))
    monkeypatch.setattr(e4, "build_prof_tool", lambda b, n: tmp_path / "tool")
    monkeypatch.setattr(e4, "raw_path", lambda *a: "x.txt")
    monkeypatch.setattr(e4, "run_bench", lambda cmd, t: (calls.append(("bench", int(cmd[cmd.index("-d") + 1]))) or
                                                       {"tok_s": 50.0, "stddev": 0.1}, None))
    summ = {"dram_bytes_per_token": 1e9, "us_per_token": 1000.0, "hit_rate": 0.1}
    monkeypatch.setattr(e4, "ncu_split", lambda argv, n, t, csv, fso=True: (calls.append(("ncu", int(argv[argv.index("-d") + 1]))) or
                                                                         {"ok": True, "error": None, "summary": summ,
                                                                          "replay_mode": "kernel", "passes": 1, "l2_state_ok": True}))
    a = e4.parse_args(["--depths", "0,32768,131072,262144", "--kv", "f16", "--ncu-kv", "f16"])
    assert e4.run(a) == 0
    kinds = [c[0] for c in calls if c[0] != "result"]
    assert kinds == ["bench"] * 4 + ["ncu"] * 2
    assert [c[1] for c in calls if c[0] == "ncu"] == [0, 32768]
    rows = calls[-1][1]
    assert [r["depth"] for r in rows] == [0, 32768, 131072, 262144]
    assert rows[0]["dram_bytes_per_token"] == 1e9 and rows[2]["dram_bytes_per_token"] is None and rows[2]["tg_tok_s"] == 50.0


KERN_SUM = """Processing [x.sqlite] with [cuda_gpu_kern_sum.py]...
"Time (%)","Total Time (ns)","Instances","Avg (ns)","Med (ns)","Min (ns)","Max (ns)","StdDev (ns)","Name"
60.0,6000000,1600,3750.0,3700.0,3000,5000,100.0,"void mul_mat_vec_q<(ggml_type)41, 1, false, false>(...)"
20.0,2000000,320,6250.0,6000.0,5000,8000,100.0,"void gated_delta_net_cuda<128>(...)"
10.0,1000000,160,6250.0,6000.0,5000,8000,100.0,"void fwht_quantize_q8_1<256, 64, 1, float, true>(...)"
10.0,1000000,500,2000.0,2000.0,1000,3000,100.0,"void rms_norm_f32<1024>(...)"
"""


def test_nsys_kern_sum_buckets():
    k = e4.parse_kern_sum(KERN_SUM)
    assert [x["count"] for x in k] == [1600, 320, 160, 500] and k[0]["time_ns"] == 6e6
    b = e4.nsys_buckets(k, 16)
    assert b["mul_mat_vec"]["us_per_token"] == pytest.approx(375.0) and b["mul_mat_vec"]["time_share"] == pytest.approx(0.6)
    assert b["hadamard"]["count"] == 160 and b["gated_delta_net"]["count"] == 320 and b["other"]["count"] == 500
    assert e4.parse_kern_sum("no header here") == []


def test_nsys_cmds():
    p = e4.nsys_profile_cmd("nsys", ["tool", "-d", "0"], "/o/x")
    assert p[:2] == ["nsys", "profile"] and "--capture-range=cudaProfilerApi" in p and "--cuda-graph-trace=node" in p
    assert p[-3:] == ["tool", "-d", "0"]
    assert e4.nsys_stats_cmd("nsys", "/o/x.nsys-rep")[-1] == "/o/x.nsys-rep"


def test_run_uses_nsys_split_without_counters(tmp_path, monkeypatch):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (bin_dir / "llama-bench").write_text("")
    calls, res = [], []
    monkeypatch.setenv("CR_LLAMA_BIN", str(bin_dir))
    monkeypatch.setattr(e4.common, "gpu_tag", lambda a: "gpu")
    monkeypatch.setattr(e4.common, "counters_available", lambda a: (False, "no counters"))
    monkeypatch.setattr(e4.common, "write_partial", lambda *a, **k: None)
    monkeypatch.setattr(e4.common, "write_result", lambda exp, args, config, rows, notes, ok, **k: res.append((rows, notes)))
    monkeypatch.setattr(e4, "download", lambda q, n: ("m.gguf", {}))
    monkeypatch.setattr(e4, "find_nsys", lambda: "/x/nsys")
    monkeypatch.setattr(e4, "build_prof_tool", lambda b, n: tmp_path / "tool")
    monkeypatch.setattr(e4, "raw_path", lambda *a: "x")
    monkeypatch.setattr(e4, "run_bench", lambda cmd, t: ({"tok_s": 50.0, "stddev": 0.1}, None))
    monkeypatch.setattr(e4, "nsys_split", lambda nsys, argv, n, t, out: calls.append(argv) or
                        {"ok": True, "error": None, "buckets": {"mul_mat_vec": {"time_share": 0.7}}, "kernel_us_per_token": 9000.0,
                         "kernels_top": []})
    a = e4.parse_args(["--depths", "0,32768,131072", "--kv", "f16", "--ncu-kv", "f16"])
    assert e4.run(a) == 0
    assert [int(c[c.index("-d") + 1]) for c in calls] == [0, 32768]
    assert calls[0][calls[0].index("-n") + 1] == str(e4.NSYS_TOKENS)
    rows, notes = res[-1]
    assert rows[0]["nsys_kernel_us_per_token"] == 9000.0 and rows[2].get("nsys_buckets") is None
    assert any("mul_mat_vec 70%" in n for n in notes)
