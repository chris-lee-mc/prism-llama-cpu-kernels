import json
import os
import shutil
import subprocess
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from harness import counters, p0_l2_check as p0  # noqa: E402

MIB = 1024 * 1024
NVCC = "/usr/local/cuda-12.8/bin/nvcc"
INFO_5090 = {"name": "RTX 5090", "cc": "12.0", "l2_bytes": 96 * MIB, "persisting_max_bytes": 72 * MIB,
             "window_max_bytes": 128 * MIB, "persisting_limit_bytes": 0, "backend": "fake"}
INFO_3090 = dict(INFO_5090, name="RTX 3090", cc="8.6", l2_bytes=6 * MIB, persisting_max_bytes=4 * MIB, window_max_bytes=128 * MIB)


def cond(us, hit=None, dram=None, ok=None):
    return {"us_per_rep": us, "hit_rate": hit, "dram_bytes_per_rep": dram,
            "counters_ok": hit is not None if ok is None else ok}


def test_make_plan_5090_and_3090():
    p = p0.make_plan(64, 1.0, INFO_5090)
    assert p["buffer_bytes"] == 64 * MIB and p["window_bytes"] == 64 * MIB and not p["window_clamped"]
    assert p["pollute_bytes"] == 192 * MIB and p["control"] is False
    c = p0.make_plan(64, 1.0, INFO_3090)
    assert c["control"] is True and c["window_bytes"] == 4 * MIB and c["window_clamped"]
    assert c["pollute_bytes"] == 64 * MIB
    assert p0.make_plan(64, 1.0, INFO_5090, pollute_mb=0)["pollute_bytes"] == 0
    big = p0.make_plan(96, 0.5, INFO_5090)
    assert big["window_bytes"] == 72 * MIB and big["window_clamped"] and big["control"] is False
    with pytest.raises(ValueError):
        p0.make_plan(0, 1.0, INFO_5090)


def test_decide_counters_pass_and_fail():
    plan = p0.make_plan(64, 1.0, INFO_5090)
    v = p0.decide(cond(100, 0.10, 64e6), cond(40, 0.90, 2e6), plan)
    assert v["pass"] and v["basis"] == "counters" and v["hit_delta"] == pytest.approx(0.8)
    assert v["dram_ratio"] == pytest.approx(2 / 64) and v["speedup"] == pytest.approx(2.5)
    assert not p0.decide(cond(100, 0.10, 64e6), cond(90, 0.25, 60e6), plan)["pass"]
    assert not p0.decide(cond(100, 0.10, 64e6), cond(40, 0.90, 40e6), plan)["pass"]


def test_decide_control_counters():
    plan = p0.make_plan(64, 1.0, INFO_3090)
    v = p0.decide(cond(100, 0.01, 64e6), cond(100, 0.05, 62e6), plan)
    assert v["pass"] and v["tolerance"] == pytest.approx(0.05 + 4 / 64)
    assert not p0.decide(cond(100, 0.01, 64e6), cond(50, 0.60, 20e6), plan)["pass"]


def test_decide_timing_fallback():
    plan = p0.make_plan(64, 1.0, INFO_5090)
    v = p0.decide(cond(100), cond(60), plan)
    assert v["basis"] == "timing" and v["counters_ok"] is False and v["pass"]
    assert not p0.decide(cond(100), cond(90), plan)["pass"]
    ctl = p0.make_plan(64, 1.0, INFO_3090)
    assert p0.decide(cond(100), cond(98), ctl)["pass"]
    assert not p0.decide(cond(100), cond(50), ctl)["pass"]
    v = p0.decide(cond(100, 0.1, 5e6, ok=False), cond(50, 0.9, 1e6, ok=True), plan)
    assert v["basis"] == "timing"


def test_trimmed_stats_drops_cold_reps():
    st = p0.trimmed_stats([1000.0] * 5 + [10.0, 12.0, 11.0])
    assert st["us_per_rep"] == 11.0 and st["n_timed"] == 3
    assert p0.trimmed_stats([5.0, 6.0])["n_timed"] == 2


def test_parse_args_defaults():
    a = p0.parse_args([])
    assert (a.mb, a.reps, a.hit_ratio, a.no_ncu, a.inner, a.method, a.pollute_mb) == (64.0, 100, 1.0, False, None, "eager", None)
    a = p0.parse_args(["--mb", "32", "--reps", "20", "--hit-ratio", "0.5", "--no-ncu", "--inner", "on", "--method", "graph-nodes"])
    assert (a.mb, a.reps, a.hit_ratio, a.inner, a.method) == (32.0, 20, 0.5, "on", "graph-nodes")


def test_inner_argv_round_trip():
    a = p0.parse_args(["--mb", "48", "--reps", "30", "--hit-ratio", "0.75", "--method", "graph-stream", "--pollute-mb", "0"])
    argv = p0._inner_argv("on", a)
    assert argv[:3] == [sys.executable, "-m", "harness.p0_l2_check"]
    b = p0.parse_args(argv[3:])
    assert (b.inner, b.mb, b.reps, b.hit_ratio, b.method, b.pollute_mb, b.no_ncu) == ("on", 48.0, 30, 0.75, "graph-stream", 0.0, True)


def test_dry_run_prints_ncu_command(capsys):
    assert p0.main(["--dry-run"]) == 0
    out = json.loads(capsys.readouterr().out)
    cmd = " ".join(out["ncu_cmd"])
    for frag in ["--cache-control none", "regex:cr_reread", "--launch-skip 5", "--launch-count 95", "--inner on", "--replay-mode application"]:
        assert frag in cmd, frag


def test_reps_too_small(capsys):
    assert p0.main(["--reps", "5"]) == 2


def test_cuda_source_names():
    src = p0.cuda_source()
    assert 'extern "C" __global__ void cr_reread(' in src and 'extern "C" __global__ void cr_pollute(' in src
    assert "cr_f4" in src and src.count("struct alignas(16) cr_f4") == 1 and "%" not in src
    full = p0.cuda_source(with_launchers=True)
    assert "cr_launch_reread" in full and "cr_launch_pollute" in full


@pytest.mark.nvcc
@pytest.mark.parametrize("sm", [86, 89, 120])
def test_kernel_compiles(tmp_path, sm):
    nvcc = NVCC if os.path.exists(NVCC) else shutil.which("nvcc")
    if not nvcc:
        pytest.skip("nvcc missing")
    f = tmp_path / "rr.cu"
    f.write_text(p0.cuda_source(with_launchers=True))
    r = subprocess.run([nvcc, "-std=c++17", "-O3", "-c", "-gencode", "arch=compute_%d,code=sm_%d" % (sm, sm), str(f),
                        "-o", str(tmp_path / "rr.o")], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-2000:]


# ---- main() wiring with the GPU and ncu parts replaced

@pytest.fixture
def wired(monkeypatch):
    from harness import l2persist

    state = {"payload": None, "ncu_calls": []}
    monkeypatch.setattr(l2persist, "log_device", lambda dev=0: dict(INFO_5090))

    def fake_run(c, args, plan, timed):
        us = 100.0 if c == "off" else 40.0
        return {"kernels": "fake", "method": args.method, "times_us": [us] * args.reps}

    def fake_ncu(c, args):
        state["ncu_calls"].append(c)
        if state.get("perm"):
            return {"counters_ok": False, "ncu_error": "ERR_NVGPUCTRPERM", "hit_rate": None, "dram_bytes_per_rep": None,
                    "ncu_us_per_rep": None, "ncu_kernels": 0}
        hit, dram = (0.1, 64e6) if c == "off" else (0.9, 1e6)
        return {"counters_ok": True, "ncu_error": None, "hit_rate": hit, "dram_bytes_per_rep": dram, "ncu_us_per_rep": 50.0,
                "ncu_kernels": 95}

    monkeypatch.setattr(p0, "run_condition", fake_run)
    monkeypatch.setattr(p0, "ncu_condition", fake_ncu)
    monkeypatch.setattr(p0, "write_result", lambda payload: state.__setitem__("payload", payload) or "/tmp/fake.json")
    monkeypatch.setattr(counters, "ncu_path", lambda: "/usr/bin/ncu")
    return state


def test_main_pass_with_counters(wired, capsys):
    assert p0.main(["--reps", "20"]) == 0
    pl = wired["payload"]
    assert wired["ncu_calls"] == ["off", "on"]
    assert pl["pass"] is True and pl["counters_ok"] is True and pl["control"] is False
    assert pl["verdict"]["basis"] == "counters" and len(pl["rows"]) == 2
    assert pl["conditions"]["on"]["us_per_rep"] == 40.0 and pl["conditions"]["on"]["hit_rate"] == 0.9
    assert pl["device"]["persisting_max_bytes"] == 72 * MIB and pl["config"]["buffer_bytes"] == 64 * MIB
    json.dumps(pl, default=str)


def test_main_counters_denied_falls_back_to_timing(wired):
    wired["perm"] = True
    assert p0.main(["--reps", "20"]) == 0
    pl = wired["payload"]
    assert pl["counters_ok"] is False and pl["verdict"]["basis"] == "timing"
    assert pl["conditions"]["off"]["ncu_error"] == "ERR_NVGPUCTRPERM"
    assert any("ERR_NVGPUCTRPERM" in n for n in pl["notes"])


def test_main_no_ncu_and_inner(wired, capsys):
    assert p0.main(["--reps", "20", "--no-ncu"]) == 0
    assert wired["ncu_calls"] == [] and wired["payload"]["counters_ok"] is False
    assert any("--no-ncu" in n for n in wired["payload"]["notes"])
    wired["payload"] = None
    assert p0.main(["--reps", "20", "--inner", "on"]) == 0
    assert wired["payload"] is None
    assert capsys.readouterr().out.count(p0.INNER_TAG) == 1


def test_main_fail_returns_1(wired, monkeypatch):
    monkeypatch.setattr(p0, "run_condition", lambda c, a, pl, timed: {"times_us": [100.0] * a.reps})
    assert p0.main(["--reps", "20", "--no-ncu"]) == 1
    assert wired["payload"]["pass"] is False


def test_main_no_cuda_exit_2(monkeypatch, capsys):
    from harness import l2persist

    def boom(dev=0):
        raise RuntimeError("no CUDA device")

    monkeypatch.setattr(l2persist, "log_device", boom)
    assert p0.main(["--reps", "20"]) == 2


def test_ncu_condition_retries_without_app_replay(monkeypatch):
    calls = []
    csv_ok = {"ok": True, "error": None, "rows": [{"hit_sectors": 90.0, "miss_sectors": 10.0, "dram_bytes": 1000.0, "time_ns": 2000.0}]}

    def fake(argv, tokens, **kw):
        calls.append(kw.get("extra"))
        if kw.get("extra"):
            return {"ok": False, "error": "ncu exit code 1", "rows": []}
        return csv_ok

    monkeypatch.setattr(counters, "run_ncu", fake)
    monkeypatch.setattr(counters, "ncu_path", lambda: "/usr/bin/ncu")
    r = p0.ncu_condition("on", p0.parse_args(["--reps", "20"]))
    assert calls == [("--replay-mode", "application"), None]
    assert r["counters_ok"] and r["hit_rate"] == pytest.approx(0.9) and r["dram_bytes_per_rep"] == 1000.0
    assert r["ncu_us_per_rep"] == pytest.approx(2.0)
    calls.clear()
    monkeypatch.setattr(counters, "run_ncu", lambda a, t, **kw: calls.append(1) or {"ok": False, "error": "ERR_NVGPUCTRPERM", "rows": []})
    r = p0.ncu_condition("off", p0.parse_args([]))
    assert len(calls) == 1 and not r["counters_ok"] and r["ncu_error"] == "ERR_NVGPUCTRPERM"


def test_write_result_fallback_layout(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "infra.results", None)
    monkeypatch.setenv("CR_RESULTS_DIR", str(tmp_path))
    monkeypatch.setenv("CR_GPU_TAG", "rtx5090")
    path = p0.write_result({"pass": True})
    assert path.startswith(str(tmp_path / "p0_l2" / "rtx5090")) and path.endswith("Z.json")
    d = json.load(open(path))
    assert d["exp"] == "p0_l2" and d["gpu"] == "rtx5090" and d["data"] == {"pass": True} and d["schema"] == 1


# ---- real run on a GPU pod

@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
@pytest.mark.parametrize("method", ["eager", "graph-nodes", "graph-stream"])
def test_gpu_p0_l2_check_runs(method, tmp_path, monkeypatch):
    monkeypatch.setenv("CR_RESULTS_DIR", str(tmp_path))
    rc = p0.main(["--mb", "64", "--reps", "30", "--no-ncu", "--method", method])
    files = list(tmp_path.rglob("*.json"))
    assert len(files) == 1
    data = json.load(open(files[0]))["data"]
    assert data["conditions"]["off"]["us_per_rep"] > 0 and data["conditions"]["on"]["us_per_rep"] > 0
    assert rc == (0 if data["pass"] else 1)
    if data["control"]:
        assert data["pass"]
