"""Drive the GPU code paths of E1, E2 and E3 end to end with CUDA and the heavy parts mocked (plumbing check)."""
import json
import types

import pytest
import torch

from exps import common
from exps import e1_sweep as e1
from exps import e2_hybrid_bf16 as e2
from exps import e3_residency as e3
from harness import l2persist, timing

MIB = 1 << 20


class _Ev:
    t = 0.0

    def __init__(self, enable_timing=False):
        pass

    def record(self, *a):
        _Ev.t += 1000.0   # 1 s between the two events of a pair

    def elapsed_time(self, other):
        return 1000.0 * 64   # ms


@pytest.fixture
def fake_cuda(tmp_path, monkeypatch):
    monkeypatch.setenv("CR_RESULTS_DIR", str(tmp_path / "results"))
    monkeypatch.setenv("CR_GPU_TAG", "rtx5090")
    monkeypatch.delenv("CR_START_EPOCH", raising=False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *a, **k: None)
    monkeypatch.setattr(torch.cuda, "Event", _Ev)
    monkeypatch.setattr(torch._dynamo, "reset", lambda: None)
    monkeypatch.setattr(l2persist, "log_device", lambda dev=0: {"name": "fake", "l2_bytes": 96 * MIB,
                                                                "persisting_max_bytes": 72 * MIB})
    for name in ("reset", "clear_stream_window"):
        monkeypatch.setattr(l2persist, name, lambda *a, **k: None)
    monkeypatch.setattr(l2persist, "set_limit", lambda n=None: 72 * MIB if n is None else n)
    monkeypatch.setattr(l2persist, "max_window_bytes", lambda dev=0: 128 * MIB)
    monkeypatch.setattr(l2persist, "max_persisting_bytes", lambda dev=0: 72 * MIB)
    monkeypatch.setattr(common, "counters_available", lambda a: (True, None))
    def fake_ncu_run(argv, tokens, **kw):
        calls.append((argv, tokens, kw))
        hit = 0.2 if "off" in argv else 0.8   # E1 inner argv carries the mode; other experiments do not care
        return {"ok": True, "error": None, "rows": [1], "replay_mode": "application", "passes": 2,
                "summary": {"hit_rate": hit, "dram_bytes_per_token": 1.0e9, "us_per_token": 50.0, "n_kernels": 3}}

    calls = []
    monkeypatch.setattr(common, "ncu_run", fake_ncu_run)
    return tmp_path, calls


def _result(tmp, exp):
    files = sorted((tmp / "results" / exp / "rtx5090").glob("*.json"))
    assert len(files) == 1
    return json.loads(files[0].read_text())["data"]


def _patch_e1(monkeypatch, us=100.0):
    class FakeChain:
        def __init__(self, L, variant, **kw):
            self.L, self.variant, self.pollute_bytes = L, variant, kw.get("pollute_bytes", 0)
            self.bytes_per_token = L * 4456448
            self.arena = torch.zeros(4)
            self.arena_bytes = L * 4456448

        def finite(self):
            return True

    monkeypatch.setattr(e1, "Chain", FakeChain)
    monkeypatch.setattr(e1, "choose_variant", lambda v: ("lut", {"requested": v}))
    monkeypatch.setattr(e1, "build_graph", lambda chain, method, ratio, notes: (object(), {
        "window_bytes": 0 if ratio is None else chain.arena_bytes, "persist_limit_bytes": 0 if ratio is None else 72 * MIB,
        "window_method_used": None if ratio is None else "graph", "clamped": False}))
    monkeypatch.setattr(e1, "measure", lambda chain, g, tokens, repeats: {"us_per_token": us, "finite": True})
    monkeypatch.setattr(e1, "measure_pollution", lambda chain, repeats: 10.0)


def test_e1_run_plumbing(fake_cuda, monkeypatch):
    tmp, calls = fake_cuda
    _patch_e1(monkeypatch)   # equal times on and off -> stop rule fires on rtx5090
    assert e1.main(["--quick"]) == 0
    d = _result(tmp, "e1")
    assert [(r["W_mib"], r["mode"]) for r in d["rows"]] == [(16, "off"), (16, "1.0"), (64, "off"), (64, "1.0")]
    r = d["rows"][1]
    assert r["l2_hit_rate"] == 0.8 and r["dram_bytes_per_token"] == 1.0e9 and r["counters_ok"] and r["window_bytes"] > 0
    assert r["us_per_token"] == 90.0 and r["us_per_token_gross"] == 100.0 and r["pollute_us_per_token"] == 10.0
    assert r["pollute_bytes"] == 192 * MIB and d["config"]["pollute_bytes"] == 192 * MIB
    assert r["gbps_eff"] == pytest.approx(r["bytes_per_token"] / 90.0 / 1e3)
    assert r["ncu_replay_mode"] == "application" and r["ncu_passes"] == 2
    assert r["ncu_n_kernels"] == 3 and r["ncu_n_kernels_ok"] is False   # fake reports 3, expected 5*L
    assert any("5*L" in n for n in d["notes"])
    assert d["counters_ok"] is True and any("STOP" in n for n in d["notes"])
    assert not list((tmp / "results" / "e1" / "rtx5090" / "raw").glob("partial.json"))   # dropped after the final write
    argv, tokens, kw = next(c for c in calls if c[0][c[0].index("--W") + 1] == "16")
    assert tokens == 5 and kw["kernel_regex"] == "cr_gemv" and kw["launch_skip"] == 12 and kw["launch_count"] == 20
    assert kw["profile_from_start_off"] is True and kw["kernel_fallback"] is False and "--inner" in argv
    assert argv[argv.index("--pollute-bytes") + 1] == str(192 * MIB)


def test_e1_timing_first_then_ncu_near_knee(fake_cuda, monkeypatch):
    tmp, calls = fake_cuda
    _patch_e1(monkeypatch)
    assert e1.main(["--W", "16,64,128,512", "--modes", "off", "--no-pollute"]) == 0
    d = _result(tmp, "e1")
    assert [r["W_mib"] for r in d["rows"]] == [16, 64, 128, 512] and all(r["pollute_bytes"] == 0 for r in d["rows"])
    # nearest to persisting max (72) and L2 (96) first
    assert [int(c[0][c[0].index("--W") + 1]) for c in calls] == [64, 128, 16, 512]
    assert all("--pollute-bytes" not in c[0] for c in calls)
    assert any("no pollution" in n for n in d["notes"])


def test_e1_partial_written_per_row_and_failures_kept(fake_cuda, monkeypatch):
    tmp, calls = fake_cuda
    _patch_e1(monkeypatch)
    seen = []
    orig = common.write_partial
    monkeypatch.setattr(common, "write_partial", lambda *a, **k: seen.append(len(a[3])) or orig(*a, **k))
    monkeypatch.setattr(e1, "measure", lambda chain, g, tokens, repeats: (_ for _ in ()).throw(RuntimeError("boom")))
    assert e1.main(["--quick", "--no-ncu"]) == 0
    d = _result(tmp, "e1")
    assert all(r["us_per_token"] is None and "boom" in r["error"] for r in d["rows"])
    assert seen == [1, 2, 3, 4]


def test_e1_inner_argv_path(monkeypatch):
    """--inner must work with the argv the driver builds (modes arrive as tuples from parse_args)."""
    built, events = [], []

    class FakeChain:
        def __init__(self, L, variant, **kw):
            built.append((L, variant, kw))

    class FakeGraph:
        def replay(self):
            events.append("replay")

    graphs = []

    def fake_build(chain, method, ratio, notes):
        graphs.append((method, ratio))
        return FakeGraph(), {}

    monkeypatch.setattr(e1, "Chain", FakeChain)
    monkeypatch.setattr(e1, "build_graph", fake_build)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *a, **k: None)
    monkeypatch.setattr(torch.cuda.profiler, "start", lambda: events.append("start"))
    monkeypatch.setattr(torch.cuda.profiler, "stop", lambda: events.append("stop"))
    monkeypatch.setattr(l2persist, "reset", lambda *a, **k: None)
    assert e1.main(e1.inner_argv(64, "0.75", "lut", "graph", 1 << 20)[3:]) == 0
    assert built == [(15, "lut", {"pollute_bytes": 1 << 20})] and graphs == [("graph", 0.75)]
    assert events == ["replay", "start", "replay", "stop"]
    assert e1.main(e1.inner_argv(16, "off", "fp4", "stream")[3:]) == 0
    assert graphs[-1] == ("stream", None) and built[-1] == (4, "fp4", {"pollute_bytes": 0})


class _FakeDecoder:
    def __init__(self):
        self.pos, self.tok = torch.zeros(1, dtype=torch.long), torch.zeros(1, 1, dtype=torch.long)

    def once(self):
        pass

    def reset(self):
        pass


def test_e2_run_plumbing(fake_cuda, monkeypatch):
    tmp, calls = fake_cuda
    model = torch.nn.Linear(8, 8, bias=False).to(torch.bfloat16)
    monkeypatch.setattr(e2, "setup", lambda args, info: (model, "logits_to_keep"))
    monkeypatch.setattr(e2, "make_cache", lambda m, n, info: object())
    monkeypatch.setattr(e2, "prefill", lambda m, c, ctx, kk, seed=1234: c)
    monkeypatch.setattr(e2, "pick_decoder", lambda *a, **k: (_FakeDecoder(), "reduce-overhead"))
    dram = {0: 1.60e9, 1024: 1.60e9 + 12288 * 1600}
    monkeypatch.setattr(common, "ncu_run", lambda argv, tokens, **kw: {
        "ok": True, "error": None, "rows": [1], "summary": {
            "hit_rate": 0.1, "dram_bytes_per_token": dram[int(argv[argv.index("--ctx") + 1])], "us_per_token": 900.0, "n_kernels": 9}})
    assert e2.main(["--model", "0.8B", "--quick"]) == 0
    d = _result(tmp, "e2_08b")
    assert [r["ctx"] for r in d["rows"]] == [0, 1024]
    r0, r1 = d["rows"]
    assert r0["tok_s"] == pytest.approx(32 / 64.0) and r0["compile_mode_used"] == "reduce-overhead"
    assert r1["split"]["kv"] == pytest.approx(12288 * 1600) and r0["split"]["gdn_state"] == 2 * 18874368
    assert r0["analytic"]["weight_bytes"] == 64 * 2      # parameter bytes of the fake model
    assert d["fit"] is not None and d["fit"]["bytes_source"] == "ncu_dram_bytes"
    assert d["counters_ok"] is True


def _e2_stubs(monkeypatch, tmp_dram=True):
    model = torch.nn.Linear(8, 8, bias=False).to(torch.bfloat16)
    monkeypatch.setattr(e2, "setup", lambda args, info: (model, "logits_to_keep"))
    monkeypatch.setattr(e2, "pick_decoder", lambda *a, **k: (_FakeDecoder(), "eager"))


def test_e2_ctx_failure_keeps_other_rows(fake_cuda, monkeypatch):
    tmp, calls = fake_cuda
    _e2_stubs(monkeypatch)
    monkeypatch.setattr(e2, "make_cache", lambda m, n, info: object())

    def prefill(m, c, ctx, kk, seed=1234):
        if ctx >= 1024:
            raise RuntimeError("CUDA out of memory")
        return c
    monkeypatch.setattr(e2, "prefill", prefill)
    assert e2.main(["--model", "0.8B", "--quick", "--no-ncu"]) == 0
    d = _result(tmp, "e2_08b")
    r0, r1 = d["rows"]
    assert r0["ctx"] == 0 and r0["us_per_token"] is not None and "error" not in r0
    assert r1["ctx"] == 1024 and r1["us_per_token"] is None and "out of memory" in r1["error"]
    assert "static" in r1["error"] and "dynamic" in r1["error"]   # both cache kinds were tried
    assert any("ctx=1024 failed" in n for n in d["notes"])


def test_e2_static_cache_failure_retries_dynamic(fake_cuda, monkeypatch):
    tmp, calls = fake_cuda
    _e2_stubs(monkeypatch)
    made = []
    monkeypatch.setattr(e2, "make_cache", lambda m, n, info: made.append(n) or object())
    seen = []

    def prefill(m, c, ctx, kk, seed=1234):
        seen.append((c is None, ctx))
        if c is not None:
            raise RuntimeError("static cache not accepted by GDN layers")
        return object()
    monkeypatch.setattr(e2, "prefill", prefill)
    assert e2.main(["--model", "0.8B", "--quick", "--no-ncu", "--ctx", "1024"]) == 0
    row = _result(tmp, "e2_08b")["rows"][0]
    assert row["cache_kind"] == "dynamic" and "static cache not accepted" in row["static_cache_error"]
    assert row["us_per_token"] is not None and len(made) == 1 and seen == [(False, 1024), (True, 1024)]


def test_e2_dynamic_cache_runs_eager_only(fake_cuda, monkeypatch):
    tmp, calls = fake_cuda
    model = torch.nn.Linear(8, 8, bias=False).to(torch.bfloat16)
    monkeypatch.setattr(e2, "setup", lambda args, info: (model, "logits_to_keep"))
    monkeypatch.setattr(e2, "make_cache", lambda m, n, info: None)   # unsupported: dynamic from the start
    monkeypatch.setattr(e2, "prefill", lambda m, c, ctx, kk, seed=1234: object())
    seen = []

    def pick(model, cache, ctx, kk, start, info, static=True, exclude=()):
        seen.append(static)
        return _FakeDecoder(), "eager" if not static else start
    monkeypatch.setattr(e2, "pick_decoder", pick)
    assert e2.main(["--model", "0.8B", "--quick", "--no-ncu", "--ctx", "1024"]) == 0
    d = _result(tmp, "e2_08b")
    assert seen == [False] and d["rows"][0]["cache_kind"] == "dynamic" and d["rows"][0]["compile_mode_used"] == "eager"
    assert any("dynamic" in n and "eager" in n for n in d["notes"])


def test_e2_timed_loop_recompile_fails_over(fake_cuda, monkeypatch):
    tmp, calls = fake_cuda
    model = torch.nn.Linear(8, 8, bias=False).to(torch.bfloat16)
    monkeypatch.setattr(e2, "setup", lambda args, info: (model, "logits_to_keep"))
    monkeypatch.setattr(e2, "make_cache", lambda m, n, info: object())
    monkeypatch.setattr(e2, "prefill", lambda m, c, ctx, kk, seed=1234: c)
    graphs = [0]

    class Dec(_FakeDecoder):
        def __init__(self, mode):
            super().__init__()
            self.mode = mode

        def once(self):
            if self.mode == "reduce-overhead":
                graphs[0] += 1   # recompiles on every timed step

    def pick(model, cache, ctx, kk, start, info, static=True, exclude=()):
        mode = next(m for m in e2.MODES if m not in exclude)
        return Dec(mode), mode
    monkeypatch.setattr(e2, "pick_decoder", pick)
    monkeypatch.setattr(e2, "dynamo_graphs", lambda: graphs[0])
    assert e2.main(["--model", "0.8B", "--quick", "--no-ncu", "--ctx", "0"]) == 0
    row = _result(tmp, "e2_08b")["rows"][0]
    assert row["compile_mode_used"] == "default" and row["modes_rejected_recompile"] == ["reduce-overhead"]
    assert row["dynamo_recompiles"] == 0


def test_e2_ncu_inner_gets_cache_kind(fake_cuda, monkeypatch):
    tmp, calls = fake_cuda
    model = torch.nn.Linear(8, 8, bias=False).to(torch.bfloat16)
    monkeypatch.setattr(e2, "setup", lambda args, info: (model, "logits_to_keep"))
    monkeypatch.setattr(e2, "make_cache", lambda m, n, info: None)
    monkeypatch.setattr(e2, "prefill", lambda m, c, ctx, kk, seed=1234: object())
    monkeypatch.setattr(e2, "pick_decoder", lambda *a, **k: (_FakeDecoder(), "eager"))
    assert e2.main(["--model", "0.8B", "--quick", "--ctx", "1024"]) == 0
    argv = calls[-1][0]
    assert argv[argv.index("--cache-kind") + 1] == "dynamic" and argv[argv.index("--compile-mode") + 1] == "eager"


def test_e2_inner_falls_back_to_dynamic(monkeypatch):
    model = torch.nn.Linear(8, 8, bias=False).to(torch.bfloat16)
    monkeypatch.setattr(e2, "setup", lambda args, info: (model, "logits_to_keep"))
    monkeypatch.setattr(e2, "make_cache", lambda m, n, info: object())
    monkeypatch.setattr(e2, "fill_cache_random", lambda c: 1)
    monkeypatch.setattr(e2, "Decoder", lambda *a, **k: types.SimpleNamespace(once=lambda: None))
    monkeypatch.setattr(e2, "prefill", lambda m, c, ctx, kk, seed=1234: object())
    monkeypatch.setattr(e2, "free_gpu", lambda: None)
    kinds = []

    def pick(model, cache, ctx, kk, start, info, static=True, exclude=()):
        kinds.append(static)
        if static:
            raise RuntimeError("static cache fails at use time")
        return _FakeDecoder(), "eager"
    monkeypatch.setattr(e2, "pick_decoder", pick)
    for fn in ("synchronize",):
        monkeypatch.setattr(torch.cuda, fn, lambda *a, **k: None)
    for obj, names in ((torch.cuda.profiler, ("start", "stop")), (torch.cuda.nvtx, ("range_push", "range_pop"))):
        for n in names:
            monkeypatch.setattr(obj, n, lambda *a, **k: None)
    assert e2.main(["--model", "0.8B", "--quick", "--inner", "--ctx", "1024", "--cache-kind", "static"]) == 0
    assert kinds == [True, False]


def test_e2_result_written_when_interrupted(fake_cuda, monkeypatch):
    tmp, calls = fake_cuda
    _e2_stubs(monkeypatch)
    monkeypatch.setattr(e2, "make_cache", lambda m, n, info: object())

    def prefill(m, c, ctx, kk, seed=1234):
        if ctx > 0:
            raise SystemExit(143)   # what the SIGTERM handler raises
        return c
    monkeypatch.setattr(e2, "prefill", prefill)
    with pytest.raises(SystemExit):
        e2.main(["--model", "0.8B", "--quick", "--no-ncu"])
    d = _result(tmp, "e2_08b")
    assert [r["ctx"] for r in d["rows"]] == [0] and any("interrupted" in n for n in d["notes"])


def test_e3_run_plumbing(fake_cuda, monkeypatch):
    tmp, calls = fake_cuda

    class _Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.lm_head = torch.nn.Linear(4, 4, bias=False)

    model = _Model()
    monkeypatch.setattr(e2, "setup", lambda args, info: (model, "logits_to_keep"))
    lay = e3.arena_layout(18874368, 50331648, 196608)
    arena = torch.zeros(4, dtype=torch.uint8)
    monkeypatch.setattr(e3, "setup_config", lambda m, kk, ctx, kind, dec, info: (
        object(), arena, lay, [1, 2], "views_pinned_copy", m.lm_head))
    monkeypatch.setattr(e3, "capture_step", lambda m, c, ctx, kk, region, ar, meth, notes: (object(), {
        "window_bytes": 0 if region is None else region[1], "hit_ratio": None if region is None else 1.0,
        "persisted_bytes": 0 if region is None else min(region[1], 72 * MIB), "window_method_used": "graph"}))
    us = {"off": 1000.0, "gdn": 990.0, "head": 980.0, "both": 940.0}
    seen = []
    monkeypatch.setattr(timing, "replay_time_us", lambda g, iters=100, warmup=5: us[seen[-1]])
    orig_window = e3.window_region

    def spy(cond, lay_):
        seen.append(cond)
        return orig_window(cond, lay_)
    monkeypatch.setattr(e3, "window_region", spy)
    dram = {"off": 1.6e9, "gdn": 1.58e9, "head": 1.57e9, "both": 1.53e9}
    monkeypatch.setattr(common, "ncu_run", lambda argv, tokens, **kw: {
        "ok": True, "error": None, "rows": [1], "summary": {
            "hit_rate": 0.5, "dram_bytes_per_token": dram[argv[argv.index("--conditions") + 1]], "us_per_token": 10.0, "n_kernels": 4}})
    assert e3.main(["--quick"]) == 0
    d = _result(tmp, "e3")
    rows = d["rows"]
    assert len(rows) == 2 * 2 * 4 and {r["head"] for r in rows} == {"bf16_full", "int8_48k"}
    ncu_rows = [r for r in rows if r["dram_bytes_per_token"] is not None]
    assert {(r["ctx"], r["head"]) for r in ncu_rows} == {(0, "int8_48k")} and len(ncu_rows) == 4
    assert d["analysis"]["go_no_go"]["go"] is True
    assert any("GO/NO-GO" in n for n in d["notes"])
    assert model.lm_head is not None


def test_e3_run_skips_gdn_when_states_unavailable(fake_cuda, monkeypatch):
    tmp, calls = fake_cuda
    model = types.SimpleNamespace(lm_head=object())
    monkeypatch.setattr(e2, "setup", lambda args, info: (model, None))
    lay = e3.arena_layout(0, 1024, 0)
    monkeypatch.setattr(e3, "setup_config", lambda *a: (object(), torch.zeros(4, dtype=torch.uint8), lay, [], "unavailable", None))
    monkeypatch.setattr(e3, "capture_step", lambda *a: (object(), {"window_bytes": 0, "hit_ratio": None, "persisted_bytes": 0}))
    monkeypatch.setattr(timing, "replay_time_us", lambda g, iters=100, warmup=5: 5.0)
    assert e3.main(["--quick", "--no-ncu", "--allow-no-gdn"]) == 0
    d = _result(tmp, "e3")
    assert {r["condition"] for r in d["rows"]} == {"off", "head"}
    assert any("skipped" in n for n in d["notes"])


def test_e3_run_fails_fast_when_states_unavailable(fake_cuda, monkeypatch):
    tmp, calls = fake_cuda
    plain = object()
    model = types.SimpleNamespace(lm_head=plain)
    monkeypatch.setattr(e2, "setup", lambda args, info: (model, None))
    lay = e3.arena_layout(0, 1024, 0)
    monkeypatch.setattr(e3, "setup_config", lambda *a: (object(), torch.zeros(4, dtype=torch.uint8), lay, [], "unavailable", None))

    def no_capture(*a):
        raise AssertionError("timing loop must not start")
    monkeypatch.setattr(e3, "capture_step", no_capture)
    assert e3.main(["--quick", "--no-ncu"]) == 4
    d = _result(tmp, "e3")
    assert d["rows"] == [] and any("Aborting before timing" in n for n in d["notes"])
    monkeypatch.setattr(e3, "capture_step", lambda *a: (object(), {"window_bytes": 0, "hit_ratio": None, "persisted_bytes": 0}))
    monkeypatch.setattr(timing, "replay_time_us", lambda g, iters=100, warmup=5: 5.0)
    assert e3.main(["--quick", "--no-ncu", "--conditions", "off,head"]) == 0   # no gdn condition asked: no abort


def test_e3_run_restores_head_after_failed_config(fake_cuda, monkeypatch):
    tmp, calls = fake_cuda

    class _Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.lm_head = torch.nn.Linear(4, 4, bias=False)

    model = _Model()
    plain = model.lm_head
    monkeypatch.setattr(e2, "setup", lambda args, info: (model, "logits_to_keep"))
    lay = e3.arena_layout(0, 1024, 0)
    seen = []

    def setup_config(m, kk, ctx, kind, dec, info):
        seen.append(m.lm_head)
        m.lm_head = torch.nn.Identity()   # what a real setup_config leaves behind
        if kind == "bf16_full":
            raise RuntimeError("boom after the swap")
        return object(), torch.zeros(4, dtype=torch.uint8), lay, [], "unavailable", plain
    monkeypatch.setattr(e3, "setup_config", setup_config)
    monkeypatch.setattr(e3, "capture_step", lambda *a: (object(), {"window_bytes": 0, "hit_ratio": None, "persisted_bytes": 0}))
    monkeypatch.setattr(timing, "replay_time_us", lambda g, iters=100, warmup=5: 5.0)
    assert e3.main(["--quick", "--no-ncu", "--allow-no-gdn"]) == 0
    assert all(h is plain for h in seen) and model.lm_head is plain
    d = _result(tmp, "e3")
    assert any("setup failed" in n for n in d["notes"]) and {r["head"] for r in d["rows"]} == {"int8_48k"}


class _FakeStream:
    n = 0

    def __init__(self):
        _FakeStream.n += 1
        self.cuda_stream = 1000 + _FakeStream.n


def _window_log(monkeypatch):
    """Record set/clear of stream windows in order; state maps stream handle -> live window bytes."""
    log, state = [], {}
    monkeypatch.setattr(torch.cuda, "Stream", _FakeStream)
    monkeypatch.setattr(l2persist, "set_stream_window", lambda ptr, nb, hr, stream=None: (
        log.append(("set", stream.cuda_stream)), state.__setitem__(stream.cuda_stream, nb)))
    monkeypatch.setattr(l2persist, "clear_stream_window", lambda stream=None: (
        log.append(("clear", stream.cuda_stream)), state.pop(stream.cuda_stream, None)))
    monkeypatch.setattr(l2persist, "get_stream_window", lambda stream=None: {"num_bytes": state.get(stream.cuda_stream, 0)})
    monkeypatch.setattr(l2persist, "apply_window_to_graph", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("no graph window")))
    monkeypatch.setattr(l2persist, "get_limit", lambda: 0)
    monkeypatch.setattr(l2persist, "get_graph_window", lambda g: {"num_bytes": 0})
    return log, state


def test_e1_build_graph_clears_stream_window(fake_cuda, monkeypatch):
    log, state = _window_log(monkeypatch)
    captured = []
    monkeypatch.setattr(timing, "capture_graph", lambda fn, warmup=3, stream=None, keep_graph=False: (
        captured.append(state.get(stream.cuda_stream, 0) if stream else 0), types.SimpleNamespace(instantiate=lambda: None))[1])
    chain = types.SimpleNamespace(arena=torch.zeros(4), arena_bytes=32 * MIB)
    g, info = e1.build_graph(chain, "stream", 0.5, [])
    assert info["window_method_used"] == "stream" and info["window_readback"]["num_bytes"] == 32 * MIB
    assert captured == [32 * MIB]                      # the window was live during capture
    st = next(s for k, s in log if k == "set")
    assert log[-1] == ("clear", st) and not state       # and is gone afterwards
    # a later off capture on a recycled pool stream must not see a window
    state[1777] = 5 * MIB
    monkeypatch.setattr(_FakeStream, "n", 776)
    captured.clear()
    e1.build_graph(chain, "stream", None, [])
    assert captured == [0] and 1777 not in state


def test_e1_build_graph_clears_window_when_capture_fails(fake_cuda, monkeypatch):
    log, state = _window_log(monkeypatch)
    monkeypatch.setattr(timing, "capture_graph", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("capture failed")))
    chain = types.SimpleNamespace(arena=torch.zeros(4), arena_bytes=32 * MIB)
    with pytest.raises(RuntimeError):
        e1.build_graph(chain, "stream", 1.0, [])
    assert log[-1][0] == "clear" and not state


def test_e3_capture_step_clears_stream_window(fake_cuda, monkeypatch):
    log, state = _window_log(monkeypatch)
    monkeypatch.setattr(e3, "step_fn", lambda *a: (lambda: None))
    monkeypatch.setattr(timing, "capture_graph", lambda fn, warmup=3, stream=None, keep_graph=False: object())
    arena = torch.zeros(8, dtype=torch.uint8)
    g, info = e3.capture_step(None, None, 0, None, (0, 4 * MIB), arena, "stream", [])
    assert info["window_method_used"] == "stream" and info["window_readback"]["num_bytes"] == 4 * MIB
    assert log[-1][0] == "clear" and not state


def test_e2_times_every_ctx_before_ncu(fake_cuda, monkeypatch):
    tmp, calls = fake_cuda
    _e2_stubs(monkeypatch)
    monkeypatch.setattr(e2, "make_cache", lambda m, n, info: object())
    monkeypatch.setattr(e2, "prefill", lambda m, c, ctx, kk, seed=1234: c)
    log = []
    orig = e2.measure_ctx
    monkeypatch.setattr(e2, "measure_ctx", lambda m, kk, a, ctx, info, wb: (log.append(("time", ctx)), orig(m, kk, a, ctx, info, wb))[1])
    monkeypatch.setattr(common, "ncu_run", lambda argv, tokens, **kw: (log.append(("ncu", int(argv[argv.index("--ctx") + 1]))), {
        "ok": True, "error": None, "rows": [1], "summary": {"hit_rate": 0.1, "dram_bytes_per_token": 1.0e9, "us_per_token": 9.0,
                                                           "n_kernels": 9}})[1])
    assert e2.main(["--model", "0.8B", "--quick", "--ctx", "0,4096,1024,65536"]) == 0
    assert log[:4] == [("time", 0), ("time", 65536), ("time", 1024), ("time", 4096)]
    assert [c for k, c in log[4:] if k == "ncu"] == [0, 65536, 1024, 4096]
    assert all(r["dram_bytes_per_token"] == 1.0e9 for r in _result(tmp, "e2_08b")["rows"])


def test_e2_ncu_budget_cut_keeps_timed_rows(fake_cuda, monkeypatch):
    tmp, calls = fake_cuda
    _e2_stubs(monkeypatch)
    monkeypatch.setattr(e2, "make_cache", lambda m, n, info: object())
    monkeypatch.setattr(e2, "prefill", lambda m, c, ctx, kk, seed=1234: c)
    seen = []

    def ncu(argv, tokens, **kw):
        seen.append(kw["timeout"])
        monkeypatch.setattr(common.Budget, "expired", lambda self, reserve_s=0.0: reserve_s >= 180)   # budget gone after job 1
        return {"ok": True, "error": None, "rows": [1], "summary": {"hit_rate": 0.1, "dram_bytes_per_token": 1.0e9,
                                                                   "us_per_token": 9.0, "n_kernels": 9}}
    monkeypatch.setattr(common, "ncu_run", ncu)
    assert e2.main(["--model", "0.8B", "--quick", "--ctx", "0,1024,4096", "--budget-min", "20"]) == 0
    d = _result(tmp, "e2_08b")
    assert len(d["rows"]) == 3 and all(r["us_per_token"] is not None for r in d["rows"])
    assert [r.get("ncu_error") for r in d["rows"]] == [None, "skipped: time budget", "skipped: time budget"]
    assert len(seen) == 1 and seen[0] <= 20 * 60 / 3 + 1   # first job gets a third of the time left, not the full cap
    assert d["counters_ok"] is False
