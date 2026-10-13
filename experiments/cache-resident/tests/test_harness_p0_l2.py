import json
import os
import shutil
import subprocess
import sys
import time

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
    assert not p0.decide(cond(100, 0.01, 64e6), cond(100, 0.01 + v["tolerance"] + 0.01, 62e6), plan)["pass"]


def test_decide_flags_kernel_replay_counters():
    plan = p0.make_plan(64, 1.0, INFO_5090)
    off, on = cond(100, 0.10, 64e6), cond(40, 0.90, 2e6)
    assert p0.decide(off, on, plan)["counters_suspect"] is False
    assert p0.decide(dict(off, replay_mode="application"), dict(on, replay_mode="kernel"), plan)["counters_suspect"] is True


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
    assert (a.mb, a.reps, a.hit_ratio, a.no_ncu, a.inner, a.method, a.pollute_mb) == (64.0, 100, 1.0, False, None, "all", None)
    assert (a.ncu_timeout, a.ncu_budget) == (p0.NCU_CALL_TIMEOUT, p0.NCU_BUDGET)
    a = p0.parse_args(["--mb", "32", "--reps", "20", "--hit-ratio", "0.5", "--no-ncu", "--inner", "on", "--method", "graph-nodes"])
    assert (a.mb, a.reps, a.hit_ratio, a.inner, a.method) == (32.0, 20, 0.5, "on", "graph-nodes")


def test_inner_argv_round_trip():
    a = p0.parse_args(["--mb", "48", "--reps", "30", "--hit-ratio", "0.75", "--method", "graph-stream", "--pollute-mb", "0"])
    argv = p0._inner_argv("on", a)
    assert argv[:3] == [sys.executable, "-m", "harness.p0_l2_check"]
    b = p0.parse_args(argv[3:])
    assert (b.inner, b.mb, b.reps, b.hit_ratio, b.method, b.pollute_mb, b.no_ncu) == ("on", 48.0, 30, 0.75, "graph-stream", 0.0, True)


def test_inner_argv_rejects_all():
    with pytest.raises(ValueError):
        p0._inner_argv("on", p0.parse_args([]))


def test_dry_run_prints_ncu_command(capsys):
    assert p0.main(["--dry-run"]) == 0
    out = json.loads(capsys.readouterr().out)
    cmd = " ".join(out["ncu_cmd"])
    for frag in ["--cache-control none", "regex:cr_reread", "--launch-skip 5", "--launch-count 95", "--inner on", "--replay-mode application"]:
        assert frag in cmd, frag
    assert out["methods"] == list(p0.METHODS)
    for m in p0.METHODS:
        assert "--method " + m + " " in " ".join(out["ncu_cmds"][m]) + " "
    assert p0.main(["--dry-run", "--method", "graph-nodes"]) == 0
    assert json.loads(capsys.readouterr().out)["methods"] == ["graph-nodes"]


def test_reps_too_small(capsys):
    assert p0.main(["--reps", "5"]) == 2


def test_inner_needs_one_method(capsys):
    assert p0.main(["--inner", "on"]) == 2


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
def wired(monkeypatch, tmp_path):
    from harness import l2persist

    state = {"payload": None, "payloads": [], "ncu_calls": [], "run_calls": [], "removed": [], "fail_methods": (), "ncu_timeouts": []}
    monkeypatch.setattr(l2persist, "log_device", lambda dev=0: dict(INFO_5090))

    def fake_run(c, args, plan, timed):
        state["run_calls"].append((c, args.method))
        if args.method in state["fail_methods"]:
            raise RuntimeError("boom " + args.method)
        # graph-stream: speed-up only when the node window was copied
        us = 100.0 if c == "off" else state.get("on_us", {}).get(args.method, 40.0)
        out = {"kernels": "fake", "method": args.method, "times_us": [us] * args.reps}
        if c == "on" and args.method != "eager":
            out["node_window_readback"] = {"num_bytes": plan["window_bytes"], "hit_prop": "persisting", "hit_ratio": 1.0}
        return out

    def fake_ncu(c, args, timeout=p0.NCU_CALL_TIMEOUT):
        state["ncu_calls"].append((args.method, c))
        state["ncu_timeouts"].append(timeout)
        if state.get("perm"):
            return {"counters_ok": False, "ncu_error": "ERR_NVGPUCTRPERM", "hit_rate": None, "dram_bytes_per_rep": None,
                    "ncu_us_per_rep": None, "ncu_kernels": 0, "replay_mode": "application", "counters_basis": "application_replay",
                    "ncu_cmd": ["ncu"]}
        hit, dram = (0.1, 64e6) if c == "off" else (state.get("hit_on", 0.9), 1e6)
        return {"counters_ok": True, "ncu_error": None, "hit_rate": hit, "dram_bytes_per_rep": dram, "ncu_us_per_rep": 50.0,
                "ncu_kernels": 95, "replay_mode": state.get("replay", "application"),
                "counters_basis": state.get("replay", "application") + "_replay", "ncu_cmd": ["ncu", "x"]}

    def fake_write(payload):
        state["payload"] = json.loads(json.dumps(payload, default=str))
        state["payloads"].append(state["payload"])
        return str(tmp_path / ("r%d.json" % len(state["payloads"])))

    monkeypatch.setattr(p0, "run_condition", fake_run)
    monkeypatch.setattr(p0, "ncu_condition", fake_ncu)
    monkeypatch.setattr(p0, "write_result", fake_write)
    monkeypatch.setattr(p0.os, "remove", lambda path: state["removed"].append(path))
    monkeypatch.setattr(counters, "ncu_path", lambda: "/usr/bin/ncu")
    return state


def test_main_default_runs_all_methods(wired, capsys):
    assert p0.main(["--reps", "20"]) == 0
    pl = wired["payload"]
    assert [c for c in wired["run_calls"] if c[0] == "on"] == [("on", m) for m in p0.METHODS]
    assert wired["ncu_calls"] == [(m, c) for m in p0.METHODS for c in ("off", "on")]
    assert list(pl["methods"]) == list(p0.METHODS) and pl["primary_method"] == "eager"
    assert pl["methods_honored"] == list(p0.METHODS) and pl["methods_passed"] == list(p0.METHODS)
    for m in p0.METHODS:
        assert pl["methods"][m]["pass"] is True and pl["methods"][m]["verdict"]["basis"] == "counters"
    assert pl["methods"]["eager"]["node_window_ok"] is None
    assert pl["methods"]["graph-nodes"]["node_window_ok"] is True and pl["methods"]["graph-stream"]["node_window_ok"] is True
    assert pl["pass"] is True and pl["counters_ok"] is True and pl["control"] is False and pl["partial"] is False
    assert pl["verdict"]["basis"] == "counters" and len(pl["rows"]) == 6
    assert pl["conditions"]["on"]["us_per_rep"] == 40.0 and pl["conditions"]["on"]["hit_rate"] == 0.9
    assert pl["conditions"]["on"]["replay_mode"] == "application" and pl["conditions"]["on"]["ncu_cmd"] == ["ncu", "x"]
    assert pl["device"]["persisting_max_bytes"] == 72 * MIB and pl["config"]["buffer_bytes"] == 64 * MIB


def test_main_single_method(wired):
    assert p0.main(["--reps", "20", "--method", "graph-nodes"]) == 0
    pl = wired["payload"]
    assert list(pl["methods"]) == ["graph-nodes"] and pl["methods_honored"] == ["graph-nodes"]
    assert wired["ncu_calls"] == [("graph-nodes", "off"), ("graph-nodes", "on")]


def test_main_one_method_fails_verdict_per_method(wired, monkeypatch):
    orig = p0.ncu_condition

    def ncu(c, args, timeout=0):
        r = orig(c, args, timeout)
        if args.method == "graph-stream" and c == "on":
            r["hit_rate"] = 0.11
        return r

    monkeypatch.setattr(p0, "ncu_condition", ncu)
    assert p0.main(["--reps", "20"]) == 1
    pl = wired["payload"]
    assert pl["pass"] is False
    assert pl["methods_honored"] == ["eager", "graph-nodes"] and pl["methods_passed"] == ["eager", "graph-nodes"]
    assert pl["methods"]["graph-stream"]["pass"] is False


def test_main_method_exception_is_recorded_not_fatal(wired):
    wired["fail_methods"] = ("graph-nodes",)
    assert p0.main(["--reps", "20"]) == 1
    pl = wired["payload"]
    assert pl["methods"]["graph-nodes"]["error"].startswith("RuntimeError") and pl["methods"]["graph-nodes"]["pass"] is False
    assert pl["methods_honored"] == ["eager", "graph-stream"]
    assert all(m != "graph-nodes" for m, _ in wired["ncu_calls"])
    assert any("graph-nodes: failed" in n for n in pl["notes"])


def test_main_node_window_readback_mismatch_is_noted(wired, monkeypatch):
    orig = p0.run_condition

    def bad(c, args, plan, timed):
        r = orig(c, args, plan, timed)
        if "node_window_readback" in r:
            r["node_window_readback"] = {"num_bytes": 0, "hit_prop": "normal"}
        return r

    monkeypatch.setattr(p0, "run_condition", bad)
    p0.main(["--reps", "20", "--no-ncu", "--method", "graph-stream"])
    pl = wired["payload"]
    assert pl["methods"]["graph-stream"]["node_window_ok"] is False
    assert any("kernel node window" in n for n in pl["notes"])


def test_main_writes_timing_result_before_ncu_then_replaces_it(wired, tmp_path, monkeypatch):
    seen = {}
    orig = p0.ncu_condition

    def ncu(c, args, timeout=0):
        seen.setdefault("payloads_at_first_ncu", len(wired["payloads"]))
        return orig(c, args, timeout)

    monkeypatch.setattr(p0, "ncu_condition", ncu)
    assert p0.main(["--reps", "20"]) == 0
    assert seen["payloads_at_first_ncu"] == 1
    first, last = wired["payloads"]
    assert first["partial"] is True and first["counters_ok"] is False and first["verdict"]["basis"] == "timing"
    assert any("partial" in n for n in first["notes"])
    assert last["partial"] is False and last["counters_ok"] is True
    assert wired["removed"] == [str(tmp_path / "r1.json")]


def test_main_ncu_interrupted_keeps_partial_result(wired, monkeypatch):
    def hang(c, args, timeout=0):
        raise KeyboardInterrupt

    monkeypatch.setattr(p0, "ncu_condition", hang)
    with pytest.raises(KeyboardInterrupt):
        p0.main(["--reps", "20"])
    assert len(wired["payloads"]) == 1 and wired["payloads"][0]["partial"] is True
    assert wired["removed"] == []


def test_main_ncu_budget_caps_calls(wired, monkeypatch):
    clock = {"t": 1000.0}
    monkeypatch.setattr(p0.time, "monotonic", lambda: clock["t"])
    orig = p0.ncu_condition

    def slow(c, args, timeout=0):
        r = orig(c, args, timeout)
        clock["t"] += timeout
        return r

    monkeypatch.setattr(p0, "ncu_condition", slow)
    assert p0.main(["--reps", "20", "--ncu-timeout", "300", "--ncu-budget", "700"]) == 0
    pl = wired["payload"]
    assert wired["ncu_timeouts"] == [300.0, 300.0, 100.0]
    c = pl["methods"]["graph-nodes"]["conditions"]["on"]
    assert c["counters_ok"] is False and "budget exhausted" in c["ncu_error"]
    assert pl["methods"]["eager"]["verdict"]["basis"] == "counters"
    assert pl["methods"]["graph-nodes"]["verdict"]["basis"] == "timing"
    assert pl["methods"]["graph-stream"]["verdict"]["basis"] == "timing"


def test_main_kernel_replay_counters_are_flagged(wired):
    wired["replay"] = "kernel"
    assert p0.main(["--reps", "20", "--method", "eager"]) == 0
    pl = wired["payload"]
    assert pl["verdict"]["counters_suspect"] is True and pl["conditions"]["off"]["counters_basis"] == "kernel_replay"
    assert any("kernel replay" in n for n in pl["notes"])


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
    assert len(wired["payloads"]) == 1 and wired["payload"]["partial"] is False
    assert any("--no-ncu" in n for n in wired["payload"]["notes"])
    wired["payload"] = None
    assert p0.main(["--reps", "20", "--inner", "on", "--method", "eager"]) == 0
    assert wired["payload"] is None
    assert capsys.readouterr().out.count(p0.INNER_TAG) == 1


def test_main_fail_returns_1(wired, monkeypatch):
    monkeypatch.setattr(p0, "run_condition", lambda c, a, pl, timed: {"times_us": [100.0] * a.reps})
    assert p0.main(["--reps", "20", "--no-ncu"]) == 1
    assert wired["payload"]["pass"] is False and wired["payload"]["methods_honored"] == []


def test_main_control_has_no_honored_methods(wired, monkeypatch):
    from harness import l2persist

    monkeypatch.setattr(l2persist, "log_device", lambda dev=0: dict(INFO_3090))
    wired["on_us"] = {m: 100.0 for m in p0.METHODS}
    wired["hit_on"] = 0.12
    assert p0.main(["--reps", "20"]) == 0
    pl = wired["payload"]
    assert pl["control"] is True and pl["methods_honored"] == [] and pl["methods_passed"] == list(p0.METHODS)


def test_main_no_cuda_exit_2(monkeypatch, capsys):
    from harness import l2persist

    def boom(dev=0):
        raise RuntimeError("no CUDA device")

    monkeypatch.setattr(l2persist, "log_device", boom)
    assert p0.main(["--reps", "20"]) == 2


def _ncu_args(*extra):
    return p0.parse_args(["--reps", "20", "--method", "eager"] + list(extra))


def test_ncu_condition_retries_without_app_replay(monkeypatch):
    calls = []
    csv_ok = {"ok": True, "error": None, "rows": [{"hit_sectors": 90.0, "miss_sectors": 10.0, "dram_bytes": 1000.0, "time_ns": 2000.0}],
              "cmd": ["ncu", "k"]}

    def fake(argv, tokens, **kw):
        calls.append((kw.get("extra"), kw["timeout"]))
        if kw.get("extra"):
            return {"ok": False, "error": "ncu exit code 1", "rows": [], "cmd": ["ncu", "app"]}
        return csv_ok

    monkeypatch.setattr(counters, "run_ncu", fake)
    monkeypatch.setattr(counters, "ncu_path", lambda: "/usr/bin/ncu")
    r = p0.ncu_condition("on", _ncu_args(), timeout=200)
    assert [c[0] for c in calls] == [("--replay-mode", "application"), None]
    assert calls[0][1] == 200 and 190 < calls[1][1] <= 200
    assert r["counters_ok"] and r["hit_rate"] == pytest.approx(0.9) and r["dram_bytes_per_rep"] == 1000.0
    assert r["ncu_us_per_rep"] == pytest.approx(2.0)
    assert r["replay_mode"] == "kernel" and r["counters_basis"] == "kernel_replay" and r["ncu_cmd"] == ["ncu", "k"]
    calls.clear()
    monkeypatch.setattr(counters, "run_ncu", lambda a, t, **kw: calls.append(1) or {"ok": False, "error": "ERR_NVGPUCTRPERM", "rows": []})
    r = p0.ncu_condition("off", _ncu_args())
    assert len(calls) == 1 and not r["counters_ok"] and r["ncu_error"] == "ERR_NVGPUCTRPERM" and r["replay_mode"] == "application"


def test_ncu_condition_application_replay_ok_and_timeout_no_retry(monkeypatch):
    calls = []
    monkeypatch.setattr(counters, "ncu_path", lambda: "/usr/bin/ncu")
    rows = [{"hit_sectors": 1.0, "miss_sectors": 1.0, "dram_bytes": 1.0, "time_ns": 1.0}]
    monkeypatch.setattr(counters, "run_ncu", lambda a, t, **kw: calls.append(kw) or {"ok": True, "error": None, "rows": rows, "cmd": ["ncu"]})
    r = p0.ncu_condition("on", _ncu_args())
    assert len(calls) == 1 and r["replay_mode"] == "application" and r["counters_basis"] == "application_replay"
    calls.clear()
    monkeypatch.setattr(counters, "run_ncu", lambda a, t, **kw: calls.append(kw) or {
        "ok": False, "error": "timeout after 300s", "timed_out": True, "rows": [], "cmd": ["ncu"]})
    r = p0.ncu_condition("on", _ncu_args())
    assert len(calls) == 1 and r["ncu_timed_out"] is True and not r["counters_ok"]


def test_write_result_fallback_layout(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "infra.results", None)
    monkeypatch.setenv("CR_RESULTS_DIR", str(tmp_path))
    monkeypatch.setenv("CR_GPU_TAG", "rtx5090")
    path = p0.write_result({"pass": True})
    assert path.startswith(str(tmp_path / "p0_l2" / "rtx5090")) and path.endswith("Z.json")
    d = json.load(open(path))
    assert d["exp"] == "p0_l2" and d["gpu"] == "rtx5090" and d["data"] == {"pass": True} and d["schema"] == 1


# ---- run_condition call order with CUDA replaced by recorders

@pytest.fixture
def fake_cuda(monkeypatch):
    import contextlib
    import types

    from harness import l2persist, timing

    log = []

    class Stream:
        cuda_stream = 77

        def synchronize(self):
            log.append("sync")

    class Event:
        def __init__(self, enable_timing=False):
            pass

        def record(self, s):
            pass

        def elapsed_time(self, other):
            return 0.01

    class Graph:
        def instantiate(self):
            log.append("instantiate")

        def replay(self):
            log.append("replay")

    class Kernels:
        how = "cupy"

        def __init__(self, buf, pol, grid):
            pass

        def reread(self, h):
            log.append("reread")

        def pollute(self, h):
            log.append("pollute")

    def capture(fn, warmup=3, stream=None, keep_graph=False):
        log.append("capture keep=%s" % keep_graph)
        return Graph()

    cu = torch.cuda
    monkeypatch.setattr(cu, "current_device", lambda: 0)
    monkeypatch.setattr(cu, "get_device_properties", lambda d: types.SimpleNamespace(multi_processor_count=4))
    monkeypatch.setattr(cu, "Stream", Stream)
    monkeypatch.setattr(cu, "Event", Event)
    monkeypatch.setattr(cu, "stream", lambda s: contextlib.nullcontext())
    monkeypatch.setattr(cu, "synchronize", lambda: None)
    monkeypatch.setattr(cu, "empty_cache", lambda: None)
    monkeypatch.setattr(torch, "rand", lambda n, device=None: torch.zeros(n))
    real_zeros = torch.zeros
    monkeypatch.setattr(torch, "zeros", lambda *a, device=None, **k: real_zeros(*a, **k))
    monkeypatch.setattr(p0, "_Kernels", Kernels)
    monkeypatch.setattr(timing, "capture_graph", capture)
    monkeypatch.setattr(l2persist, "set_limit", lambda n=None: log.append("set_limit %s" % n))
    monkeypatch.setattr(l2persist, "reset", lambda: log.append("reset"))
    monkeypatch.setattr(l2persist, "set_stream_window", lambda *a, **k: log.append("set_stream_window") or {"num_bytes": 1})
    monkeypatch.setattr(l2persist, "get_stream_window", lambda s=None: log.append("get_stream_window") or {"num_bytes": 0})
    monkeypatch.setattr(l2persist, "clear_stream_window", lambda s=None: log.append("clear_stream_window"))
    monkeypatch.setattr(l2persist, "apply_window_to_graph", lambda *a, **k: log.append("apply_graph") or 1)
    monkeypatch.setattr(l2persist, "get_graph_window", lambda g: log.append("get_graph_window") or {"num_bytes": 5})
    return log


def _run(cond, method, reps=3):
    args = p0.parse_args(["--method", method, "--reps", str(reps)])
    return p0.run_condition(cond, args, p0.make_plan(64, 1.0, INFO_5090), timed=True)


def test_run_condition_graph_stream_clears_window_after_capture(fake_cuda):
    info = _run("on", "graph-stream")
    seq = [e for e in fake_cuda if e in ("set_stream_window", "capture keep=True", "get_graph_window", "clear_stream_window",
                                          "get_stream_window", "instantiate", "replay")]
    assert seq[:6] == ["set_stream_window", "get_stream_window", "capture keep=True", "get_graph_window", "clear_stream_window",
                       "get_stream_window"]
    assert seq[6] == "instantiate" and seq[7:10] == ["replay"] * 3
    assert info["node_window_readback"] == {"num_bytes": 5} and info["window_readback_after_clear"] == {"num_bytes": 0}
    assert "apply_graph" not in fake_cuda


def test_run_condition_graph_nodes_sets_node_window_not_stream(fake_cuda):
    info = _run("on", "graph-nodes")
    assert "set_stream_window" not in fake_cuda
    seq = [e for e in fake_cuda if e in ("capture keep=True", "apply_graph", "get_graph_window", "instantiate", "replay")]
    assert seq[:4] == ["capture keep=True", "apply_graph", "get_graph_window", "instantiate"]
    assert info["node_window_readback"] == {"num_bytes": 5} and info["graph_kernel_nodes"] == 1


def test_run_condition_off_and_eager_do_not_touch_nodes(fake_cuda):
    _run("off", "graph-stream")
    _run("on", "eager")
    assert "get_graph_window" not in fake_cuda and "apply_graph" not in fake_cuda and "capture keep=True" not in fake_cuda
    assert fake_cuda.count("capture keep=False") == 1 and fake_cuda.count("set_stream_window") == 1
    with pytest.raises(ValueError):
        _run("on", "all")


# ---- real run on a GPU pod

@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
@pytest.mark.parametrize("method", ["all", "eager", "graph-nodes", "graph-stream"])
def test_gpu_p0_l2_check_runs(method, tmp_path, monkeypatch):
    monkeypatch.setenv("CR_RESULTS_DIR", str(tmp_path))
    rc = p0.main(["--mb", "64", "--reps", "100", "--no-ncu", "--method", method])
    files = list(tmp_path.rglob("*.json"))
    assert len(files) == 1
    data = json.load(open(files[0]))["data"]
    wanted = list(p0.METHODS) if method == "all" else [method]
    assert list(data["methods"]) == wanted
    for m in wanted:
        conds = data["methods"][m]["conditions"]
        assert conds["off"]["us_per_rep"] > 0 and conds["on"]["us_per_rep"] > 0
        rb = conds["on"].get("node_window_readback")
        if m == "graph-nodes":
            assert rb["num_bytes"] == data["config"]["window_bytes"] and rb["hit_prop"] == "persisting", rb
        if m == "graph-stream":
            assert conds["on"]["window_readback_after_clear"]["num_bytes"] == 0
    assert rc == (0 if data["pass"] else 1)
    if data["control"]:
        assert data["pass"]
    else:
        assert data["pass"], {m: x["verdict"] for m, x in data["methods"].items()}
        assert data["methods_honored"] == wanted
