"""Offline tests for exps.e3d_resident_draft and exps.plot_e3d (sizing, deltas, projection, plumbing with mocked CUDA)."""
import json
import types

import pytest
import torch

from exps import common
from exps import e3d_resident_draft as e3d
from exps import plot_e3d
from harness import l2persist

MIB = 1 << 20


# ---------------------------------------------------------------- sizing

@pytest.mark.parametrize("head", ["ternary", "int8"])
def test_layout_regions_aligned_and_disjoint(head):
    cfg = e3d.make_cfg(768, 8, 2560)
    lay = e3d.draft_layout(cfg, 4096, head)
    qb, sb = e3d.head_bytes(head, 4096, 768)
    spans = [(lay["head_q"], qb), (lay["head_s"], sb), (lay["state"], 6 * 12 * 128 * 128 * 4)]
    spans += [(o, cfg.ctx * cfg.Hkv * cfg.D * 2) for kv in lay["kv"].values() for o in kv]
    spans += [(lay[n], 2 * cfg.d) for n in ("x0", "x", "xn", "mix")]
    for off, _ in spans:
        assert off % e3d.ALIGN == 0
    spans.sort()
    assert all(a + n <= b for (a, n), (b, _) in zip(spans, spans[1:])) and spans[-1][0] + spans[-1][1] <= lay["total"]
    for e in lay["ent"].values():
        assert e.get("q_off", 0) % 256 == 0 and e.get("w_off", 0) % 256 == 0 and e.get("q_off", 0) < lay["total"]
    by = lay["bytes"]
    assert by["total"] == lay["total"] and by["pad"] >= 0 and sum(v for k, v in by.items() if k != "total") == by["total"]
    assert (("logits" in lay) == (head == "ternary"))
    assert len(lay["kv"]) == 2 and by["state"] == 6 * 12 * 128 * 128 * 4
    # read-write and per-token data first, matrices last (a clamped window then drops matrices)
    front = max(o + n for o, n in spans if o < lay["mat_start"])
    mats = [e for e in lay["ent"].values() if "q_off" in e]
    assert front <= lay["mat_start"] and min(e["q_off"] for e in mats) == lay["mat_start"]
    assert all(e["w_off"] < lay["mat_start"] for e in lay["ent"].values() if "w_off" in e) and lay["final_norm"] < lay["head_q"]
    assert lay["state"] == 0 and all(lay[n] < lay["mat_start"] for n in ("head_q", "head_s", "act", "x0"))
    assert max(e["s_off"] for e in mats) < lay["total"]


@pytest.mark.parametrize("head", ["ternary", "int8"])
def test_every_plan_pointer_in_arena(head):
    """All raw pointers of the draft step lie in the arena; the known exceptions are listed in OUTSIDE_ARENA."""
    cfg = e3d.make_cfg(768, 8, 2560)
    m = e3d.DraftModel.__new__(e3d.DraftModel)
    m.torch, m.cfg, m.vocab, m.head, m.vi = torch, cfg, 4096, head, 0
    m.lay = e3d.draft_layout(cfg, 4096, head)
    m.total = m.lay["total"]
    m.arena = torch.zeros(m.total, dtype=torch.uint8)
    m.base = m.arena.data_ptr()
    m._views()
    calls = []

    class Ext:
        def __getattr__(self, name):
            return lambda *a: calls.append((name, a))

    m.ext = Ext()
    m.plan = m._build_plan()
    for fn in m.plan:
        fn(0)
    ptrs = [a for _, args in calls for a in args if isinstance(a, int) and a >= 1 << 32]
    assert len(ptrs) > 50 and all(m.base <= p < m.base + m.total for p in ptrs)
    if head == "ternary":
        assert m.base <= m.logits.data_ptr() < m.base + m.total
    else:
        assert m.logits is not None and not m.base <= m.logits.data_ptr() < m.base + m.total   # int8 logits: outside
    assert len(e3d.OUTSIDE_ARENA) == 3 and any("g_attn" in x for x in e3d.OUTSIDE_ARENA)


@pytest.mark.parametrize("d", e3d.DEFAULT_D)
def test_plan_draft_hits_target(d):
    p = e3d.plan_draft(d)
    assert p["within_tol"] and abs(p["layout"]["total"] / (d * MIB) - 1) <= e3d.PLAN_TOL
    assert p["layout"]["total"] <= d * MIB   # one-sided: D=60 must not exceed the 5090 persisting max of 60 MiB
    cfg = p["cfg"]
    assert cfg.layers % 4 == 0 and cfg.F % 128 == 0
    from kernels import toy_hybrid
    toy_hybrid.validate(cfg)
    b = p["layout"]["bytes"]
    assert b["weights"] > 0 and b["head"] > 0 and b["state"] > 0 and b["kv"] > 0


def test_plan_draft_grows_and_int8_head_costs_more():
    ps = [e3d.plan_draft(d) for d in (16, 72, 160)]
    assert [p["cfg"].d for p in ps] == sorted(p["cfg"].d for p in ps)
    assert ps[0]["layout"]["bytes"]["head"] < e3d.plan_draft(16, "int8")["layout"]["bytes"]["head"]


def test_plan_verify_shapes_and_bytes():
    shapes = e3d.plan_verify(1 << 30)
    assert all(k % 128 == 0 for _, k in shapes)
    assert abs(e1_bytes(shapes) / (1 << 30) - 1) < 0.02
    assert shapes[0] == (10240, 5120) and (17408, 5120) in shapes and (12288, 5120) in shapes
    big = e3d.plan_verify(8 << 30)
    assert abs(e1_bytes(big) / (8 << 30) - 1) < 0.02 and len(big) > 430   # more than the 64 layers: cycles


def e1_bytes(shapes):
    from exps import e1_sweep
    return e1_sweep.set_bytes(shapes)


def test_choose_verify_and_window():
    assert e3d.choose_verify_mib(6144, None, 160) == (6144, False)
    assert e3d.choose_verify_mib(6144, 32 * (1 << 30), 160) == (6144, False)
    v, scaled = e3d.choose_verify_mib(6144, 4 * (1 << 30), 160)
    assert scaled and v == pytest.approx(2048 - 160)
    w = e3d.window_params(48 * MIB, 128 * MIB, 60 * MIB, "auto")
    assert w == {"window_bytes": 48 * MIB, "clamped": False, "limit": 48 * MIB, "hit_ratio": 1.0}
    w = e3d.window_params(160 * MIB, 128 * MIB, 60 * MIB, "auto")
    assert w["clamped"] and w["limit"] == 60 * MIB and w["hit_ratio"] == pytest.approx(60 / 128)
    assert e3d.window_params(16 * MIB, 128 * MIB, 60 * MIB, "0.5")["hit_ratio"] == 0.5


def test_window_problems_and_floor():
    rb = {"base_ptr": 4096, "num_bytes": 100, "hit_ratio": 0.5, "hit_prop": "persisting", "miss_prop": "streaming"}
    assert e3d.window_problems(rb, 4096, 100, 0.5) == []
    assert len(e3d.window_problems(dict(rb, base_ptr=0, num_bytes=0, hit_prop="normal"), 4096, 100, 0.5)) == 3
    assert e3d.window_problems(dict(rb, hit_ratio=0.75), 4096, 100, 0.5) and e3d.window_problems(None, 4096, 100, 0.5)
    assert e3d.window_problems(None, 4096, 0, None) == [] and e3d.window_problems(dict(rb, num_bytes=0), 4096, 0, None) == []
    assert e3d.window_problems(rb, 4096, 0, None)
    r = {"kernel_nodes": 200, "step_us": 400.0}
    e3d.floor_fields(r, 1.0)
    assert r["floor_us"] == 200.0 and r["floor_frac"] == 0.5
    r = {"kernel_nodes": None, "step_us": 400.0}
    e3d.floor_fields(r, 1.0)
    e3d.floor_fields(dict(r, kernel_nodes=5), None)
    assert "floor_us" not in r


# ---------------------------------------------------------------- summaries, deltas, projection

def test_summarize_cycles():
    raw = [(4000.0, [500.0, 300.0, 310.0], 5200.0), (4100.0, [520.0, 320.0, 330.0], 5300.0), (4200.0, [480.0, 280.0, 290.0], 5100.0)]
    s = e3d.summarize_cycles(raw)
    assert s["verify_us"] == 4100.0 and s["by_index"] == [500.0, 300.0, 310.0] and s["first_us"] == 500.0
    assert s["later_us"] == pytest.approx(305.0) and s["refill_us"] == pytest.approx(195.0)
    assert s["step_us"] == pytest.approx(raw[0][1] and sum(raw[0][1]) / 3) and s["cycle_us"] == 5200.0
    assert e3d.summarize_cycles([(1.0, [5.0], 6.0)])["later_us"] == 5.0


def test_cells():
    c = e3d.make_cells(["S1", "S2", "S3"], [4, 8])
    assert len(c) == 12 and c[0] == ("S1", 4, "off") and c[-1] == ("S3", 8, "on")
    assert all(s != "S3" for s, _, _ in e3d.make_cells(["S1", "S2", "S3"], [4], s3_ok=False))


def _row(scn, k, mode, **m):
    r = {"D_mib": 48, "pdl": 1, "scenario": scn, "K": k, "mode": mode}
    for key, vals in m.items():
        r[key + "_all"] = vals
        r[key] = sorted(vals)[len(vals) // 2]
    return r


def _rows():
    return [
        _row("S1", 4, "off", step_us=[300.0, 302.0, 301.0]), _row("S1", 4, "on", step_us=[250.0, 251.0, 249.0]),
        _row("S2", 4, "off", step_us=[400.0, 402.0, 401.0], first_us=[900.0, 905.0, 902.0], later_us=[300.0, 301.0, 300.0],
             refill_us=[600.0, 604.0, 602.0], cycle_us=[5200.0, 5210.0, 5205.0], verify_us=[4000.0, 4002.0, 4001.0]),
        _row("S2", 4, "on", step_us=[300.0, 301.0, 299.0], first_us=[400.0, 402.0, 398.0], later_us=[250.0, 251.0, 249.0],
             refill_us=[150.0, 151.0, 149.0], cycle_us=[5000.0, 5001.0, 4999.0], verify_us=[4000.0, 4003.0, 4001.0]),
        _row("S3", 4, "off", step_us=[390.0, 392.0, 391.0], first_us=[800.0, 802.0, 801.0], later_us=[300.0, 301.0, 300.0],
             refill_us=[500.0, 501.0, 501.0], cycle_us=[5100.0, 5101.0, 5100.0], verify_us=[4000.0, 4001.0, 4002.0]),
        _row("S3", 4, "on", step_us=[290.0, 291.0, 289.0], first_us=[300.0, 301.0, 299.0], later_us=[250.0, 251.0, 249.0],
             refill_us=[50.0, 50.0, 50.0], cycle_us=[4900.0, 4901.0, 4899.0], verify_us=[4000.0, 4001.0, 4002.0]),
    ]


def test_add_deltas_and_cross():
    rows = _rows()
    e3d.add_deltas(rows)
    e3d.add_cross(rows)
    s1on, s2on, s3on = rows[1], rows[3], rows[5]
    assert s1on["delta_step_us"] == pytest.approx(51.0) and s1on["gain_step_us"] == pytest.approx(51.0 / 301.0)
    assert s1on["above_step_us"] is True and "delta_first_us" not in s1on
    assert s2on["delta_first_us"] == pytest.approx(503.0) and s2on["delta_refill_us"] > 400 and "delta_step_us" not in rows[2]
    assert s2on["delta_all_first_us"] == [500.0, 503.0, 504.0]
    assert s3on["s3_delta_first_us"] == pytest.approx(100.0) and "first_minus_steady_us" not in rows[2]
    assert rows[4]["s3_delta_first_us"] == pytest.approx(101.0)


def test_spec_math():
    assert e3d.tokens_per_cycle(0.8, 4) == pytest.approx(3.3616)
    assert e3d.tokens_per_cycle(1.0, 4) == 5.0
    assert e3d.spec_speedup(0.8, 4, 4000.0, 300.0, 4000.0) == pytest.approx(3.3616 * 4000 / 5200)


def test_projection():
    rows = _rows()
    e3d.add_deltas(rows)
    p = e3d.projection(rows)
    srcs = {x["source"] for x in p}
    assert srcs == {"S1", "S2", "S3"} and len(p) == 9
    x = next(x for x in p if x["source"] == "S2" and x["acc"] == 0.8)
    assert x["speedup_off"] == pytest.approx(3.3616 * 4001 / (4001 + 4 * 401))
    assert x["speedup_on"] == pytest.approx(3.3616 * 4001 / (4001 + 4 * 300)) and x["speedup_gain"] > 0
    assert x["cycle_off_us"] == pytest.approx(4001 + 4 * 401)
    s1 = next(x for x in p if x["source"] == "S1" and x["acc"] == 0.6)   # S1 step with the verify of the S2 row
    assert s1["cycle_on_us"] == pytest.approx(4001 + 4 * 250)
    assert e3d.projection([r for r in rows if r["scenario"] != "S2"]) == [] and x["baseline_us"] == pytest.approx(4001)


def test_projection_kp1_uses_one_column_baseline():
    rows = _rows()
    for r in rows:
        if r["scenario"] != "S1":
            r["verify_cols"] = 5
    e3d.add_deltas(rows)
    assert e3d.projection(rows) == []   # multi-column verify without a 1-column baseline: skipped
    p = e3d.projection(rows, base_us={1: 800.0})
    x = next(x for x in p if x["source"] == "S2" and x["acc"] == 0.8)
    assert x["baseline_us"] == 800.0 and x["speedup_off"] == pytest.approx(3.3616 * 800 / (4001 + 4 * 401))
    assert e3d.projection(rows, base_us={0: 800.0}) == []   # other PDL setting


# ---------------------------------------------------------------- CLI and plumbing

def test_quick_and_full_args(monkeypatch, capsys):
    a = e3d.parse_args(["--quick"])
    assert a.D == [16, 72] and a.K == [4] and a.rounds == 3 and a.V == 1024.0 and a.pdl == "both"
    a = e3d.parse_args([])
    assert a.D == [16, 32, 48, 60, 72, 96, 160] and a.K == [4, 8] and a.scenarios == ["S1", "S2", "S3"] and a.hit_ratio == "auto"
    a = e3d.parse_args(["--quick", "--D", "32", "--rounds", "5"])
    assert a.D == [32] and a.rounds == 5
    with pytest.raises(SystemExit):
        e3d.parse_args(["--scenarios", "S9"])
    with pytest.raises(SystemExit):
        e3d.parse_args(["--hit-ratio", "1.5"])
    assert e3d.main(["--dry-run", "--D", "16,160"]) == 0
    plan = json.loads(capsys.readouterr().out)
    assert [x["D_mib"] for x in plan["D"]] == [16, 160] and plan["pdl"] == [0, 1] and plan["cells_per_D"] == 12
    assert plan["verify"]["matrices"] > 400 and all(x["within_tol"] for x in plan["D"])


def test_child_argv_roundtrip():
    a = e3d.parse_args(["--quick", "--gpu", "rtx5090"])
    argv = e3d.child_argv(a, 1, "/tmp/x.json", 3.5, calib=False)
    b = e3d.parse_args(argv[3:])
    assert (b.pdl, b.emit, b.D, b.K, b.V, b.rounds, b.cycles, b.bursts, b.calib, b.gpu) == (
        "1", "/tmp/x.json", [16, 72], [4], 1024.0, 3, 12, 15, False, "rtx5090")
    assert b.budget_min == 3.5 and b.quick
    assert "--no-calib" not in e3d.child_argv(a, 0, "/tmp/x.json", 3.5, calib=True)


@pytest.fixture
def fake_cuda(tmp_path, monkeypatch):
    monkeypatch.setenv("CR_RESULTS_DIR", str(tmp_path / "results"))
    monkeypatch.setenv("CR_GPU_TAG", "rtx5090")
    monkeypatch.delenv("CR_START_EPOCH", raising=False)
    monkeypatch.delenv("CR_GEMV_PDL", raising=False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setattr(l2persist, "log_device", lambda dev=0: {"name": "fake", "l2_bytes": 96 * MIB, "persisting_max_bytes": 60 * MIB})
    monkeypatch.setattr(l2persist, "max_window_bytes", lambda dev=0: 128 * MIB)
    monkeypatch.setattr(e3d, "device_cc_major", lambda: 12)
    monkeypatch.setattr(e3d, "free_vram", lambda: 30 * (1 << 30))
    return tmp_path


def _patch_gpu(monkeypatch, probe_ok=True, fail_d=None, bad_window=False, clip_limit=False):
    log = {"limits": [], "graphs": [], "vgraphs": [], "models": []}

    class FakeModel:
        def __init__(self, cfg, vocab, head, variant, seed=0):
            if seed == fail_d:
                raise RuntimeError("boom")
            lay = e3d.draft_layout(cfg, vocab, head)
            self.total, self.bytes, self.step_bytes, self.base = lay["total"], lay["bytes"], lay["step_bytes"], 4096
            log["models"].append(seed)

        def check(self):
            return {"finite": True, "ok": True, "max_abs_err": 0.0}

    class FakeVerify:
        def __init__(self, shapes, variant, seed=0):
            self.shapes = list(shapes)
            self.arena = types.SimpleNamespace(nbytes=e1_bytes(shapes), base=8192)

    def bdg(model, ratio, nbytes):
        log["graphs"].append((ratio, nbytes))
        rb = None if ratio is None else {"base_ptr": model.base + (1 if bad_window else 0), "num_bytes": nbytes, "hit_ratio": ratio,
                                         "hit_prop": "persisting", "miss_prop": "streaming"}
        return ("g", ratio), {"kernel_nodes": None if ratio is None else 7, "window_readback": rb}

    def bvg(ver, cols, stream):
        log["vgraphs"].append((cols, stream))
        return ("v", stream), {"window_readback": None}

    def steady(dg, k, bursts):
        return 300.0 if dg[1] is None else 250.0

    def cycles(dg, vg, k, n):
        first = 900.0 if dg[1] is None else 400.0
        if vg[1]:
            first -= 50.0
        steps = [first] + [300.0] * (k - 1)
        return [(4000.0, steps, 4000.0 + sum(steps))] * 3

    monkeypatch.setattr(e3d, "DraftModel", FakeModel)
    monkeypatch.setattr(e3d, "Verify", FakeVerify)
    monkeypatch.setattr(e3d, "build_draft_graph", bdg)
    monkeypatch.setattr(e3d, "build_verify_graph", bvg)
    monkeypatch.setattr(e3d, "probe_midcapture", lambda ver: {"supported": probe_ok})
    monkeypatch.setattr(e3d, "time_steady", steady)
    monkeypatch.setattr(e3d, "time_cycles", cycles)
    monkeypatch.setattr(e3d, "set_persist", lambda lim: log["limits"].append(lim) or (lim // 2 if clip_limit and lim else lim))
    monkeypatch.setattr(e3d, "time_verify", lambda vg, n=30: 800.0)
    monkeypatch.setattr(e3d, "nvml_state", lambda: {"sm_mhz": 2400})
    monkeypatch.setattr(e3d, "calibrate", lambda *a: {"copy_gbps": 1.0, "graph_node_gap_us": 1.5})
    return log


def _result(tmp):
    files = sorted((tmp / "results" / "e3d" / "rtx5090").glob("*.json"))
    assert len(files) == 1
    return json.loads(files[0].read_text())


def test_run_single_pdl_plumbing(fake_cuda, monkeypatch):
    log = _patch_gpu(monkeypatch)
    assert e3d.main(["--quick", "--pdl", "1", "--D", "16,160", "--rounds", "3"]) == 0
    doc = _result(fake_cuda)
    d, rows = doc["data"], doc["data"]["rows"]
    assert len(rows) == 2 * 6 and d["config"]["pdl"] == 1 and d["config"]["calibration"] == {"copy_gbps": 1.0, "graph_node_gap_us": 1.5}
    assert d["config"]["s3_probe"] == {"supported": True} and d["config"]["verify_cols"] == {"4": 1}
    assert d["config"]["verify_bytes"] > 0.9 * 1024 * MIB
    on = next(r for r in rows if (r["D_mib"], r["scenario"], r["mode"]) == (16, "S2", "on"))
    off = next(r for r in rows if (r["D_mib"], r["scenario"], r["mode"]) == (16, "S2", "off"))
    assert on["first_us"] == 400.0 and off["first_us"] == 900.0 and on["delta_first_us"] == 500.0 and on["by_index"][0] == 400.0
    assert on["persist_limit_bytes"] == on["D_bytes"] and on["hit_ratio"] == 1.0 and on["verify"] == "plain" and off["hit_ratio"] is None
    assert on["persist_limit_req_bytes"] == on["D_bytes"] and on["window_ok"] and on["limit_ok"] and off["persist_limit_bytes"] == 0
    assert on["d_le_l2"] and off["d_le_l2"] and on["kernel_nodes"] == off["kernel_nodes"] == 7   # off graph: same step() as on
    assert not any("WARNING" in n for n in d["notes"]) and "verify1_us" not in d["config"]
    assert any("naturally L2-resident" in n for n in d["notes"]) and d["config"]["l2_bytes"] == 96 * MIB
    assert any("g_attn" in n for n in d["notes"]) and on["floor_us"] == 10.5 and on["floor_frac"] == pytest.approx(10.5 / on["step_us"])
    assert len(on["first_us_all"]) == 3 and on["noise_first_us"] == 0.0 and on["verify_gbps"] > 0
    big = next(r for r in rows if (r["D_mib"], r["scenario"], r["mode"]) == (160, "S1", "on"))
    assert big["persist_limit_bytes"] == 60 * MIB and big["window_bytes"] == 128 * MIB and big["window_clamped"]
    assert big["hit_ratio"] == pytest.approx(60 / 128) and big["kernel_nodes"] == 7 and big["gain_step_us"] == pytest.approx(50 / 300)
    s3 = next(r for r in rows if (r["D_mib"], r["scenario"], r["mode"]) == (16, "S3", "on"))
    assert s3["verify"] == "stream" and s3["s3_delta_first_us"] == pytest.approx(50.0)
    assert {p["source"] for p in d["projection"]} == {"S1", "S2", "S3"} and d["projection"][0]["pdl"] == 1
    assert {0, 16 * MIB * 0 + on["D_bytes"], 60 * MIB} <= set(log["limits"]) and log["limits"][-1] == 0
    assert (1, False) in log["vgraphs"] and (1, True) in log["vgraphs"]
    assert any("window" in n for n in d["notes"][-1:]) or d["notes"]
    import os
    assert os.environ["CR_GEMV_PDL"] == "1"
    assert not list((fake_cuda / "results" / "e3d" / "rtx5090" / "raw").glob("partial.json"))


def test_s3_skipped_when_probe_fails_and_kp1(fake_cuda, monkeypatch):
    log = _patch_gpu(monkeypatch, probe_ok=False)
    assert e3d.main(["--quick", "--pdl", "0", "--D", "16", "--K", "4,8", "--verify-cols", "kp1", "--rounds", "3"]) == 0
    d = _result(fake_cuda)["data"]
    assert {r["scenario"] for r in d["rows"]} == {"S1", "S2"} and any("S3 skipped" in n for n in d["notes"])
    assert d["config"]["verify_cols"] == {"4": 5, "8": 9} and d["config"]["verify1_us"] == 800.0
    assert next(r for r in d["rows"] if r["scenario"] == "S2" and r["K"] == 8)["verify_cols"] == 9
    assert d["projection"] and all(p["baseline_us"] == 800.0 for p in d["projection"])
    assert (1, False) in log["vgraphs"]


def test_failed_D_kept_and_next_D_runs(fake_cuda, monkeypatch):
    _patch_gpu(monkeypatch, fail_d=16)
    assert e3d.main(["--quick", "--pdl", "0", "--D", "16,72", "--no-calib", "--rounds", "3"]) == 0
    d = _result(fake_cuda)["data"]
    bad = [r for r in d["rows"] if r["D_mib"] == 16]
    assert len(bad) == 6 and all("boom" in r["error"] and r["step_us"] is None for r in bad)
    assert all(r.get("step_us") for r in d["rows"] if r["D_mib"] == 72 and r["scenario"] == "S1")
    assert "calibration" not in d["config"] and any("D=16 failed" in n for n in d["notes"])


@pytest.mark.parametrize("kw", [{"bad_window": True}, {"clip_limit": True}])
def test_window_or_limit_mismatch_is_flagged(fake_cuda, monkeypatch, kw):
    _patch_gpu(monkeypatch, **kw)
    assert e3d.main(["--quick", "--pdl", "0", "--D", "16", "--no-calib", "--rounds", "3"]) == 0
    d = _result(fake_cuda)["data"]
    on = [r for r in d["rows"] if r["mode"] == "on"]
    off = [r for r in d["rows"] if r["mode"] == "off"]
    assert all(r["window_ok"] is False for r in on) if "bad_window" in kw else all(r["limit_ok"] is False for r in on)
    assert all(r["window_ok"] and r["limit_ok"] for r in off)
    assert any("WARNING: persisting window or limit" in n and "[16]" in n for n in d["notes"])
    assert "floor_us" not in on[0]   # no calibration gap


def test_both_pdl_runs_children_and_merges(fake_cuda, monkeypatch):
    calls = []

    def fake_run(argv, cwd=None, timeout=None):
        calls.append(argv)
        pdl = int(argv[argv.index("--pdl") + 1])
        rows = [dict(r, pdl=pdl) for r in _rows()]
        for r in rows:
            r["D_mib"] = 16
        e3d.dump_json(argv[argv.index("--emit") + 1], {"config": {"pdl": pdl, "device": {"x": 1}}, "rows": rows, "notes": [f"n{pdl}"]})
        return types.SimpleNamespace(returncode=0 if pdl == 0 else 3)

    monkeypatch.setattr(e3d.subprocess, "run", fake_run)
    assert e3d.main(["--quick", "--D", "16"]) == 0
    assert len(calls) == 2 and "--no-calib" not in calls[0] and "--no-calib" in calls[1]
    d = _result(fake_cuda)["data"]
    assert d["config"]["pdl_modes"] == [0, 1] and d["config"]["child_exit"] == {"0": 0, "1": 3}
    assert {r["pdl"] for r in d["rows"]} == {0, 1} and len(d["rows"]) == 12
    assert any("[pdl=1] n1" in n for n in d["notes"]) and any("pdl=1 child exit 3" in n for n in d["notes"])
    assert {p["pdl"] for p in d["projection"]} == {0, 1}
    assert all("delta_step_us" in r for r in d["rows"] if r["scenario"] == "S2" and r["mode"] == "on")


def test_no_pdl_run_on_old_gpu(fake_cuda, monkeypatch):
    _patch_gpu(monkeypatch)
    monkeypatch.setattr(e3d, "device_cc_major", lambda: 8)
    assert e3d.main(["--quick", "--D", "16", "--rounds", "3"]) == 0
    d = _result(fake_cuda)["data"]
    assert {r["pdl"] for r in d["rows"]} == {0} and any("PDL needs sm_90" in n for n in d["notes"])


# ---------------------------------------------------------------- plot

def _doc(gpu="rtx5090"):
    rows = []
    for d in (16, 48, 96):
        for pdl in (0, 1):
            for r in _rows():
                r = dict(r, D_mib=d, pdl=pdl)
                rows.append(r)
    e3d.add_deltas(rows)
    e3d.add_cross(rows)
    return {"schema": 1, "exp": "e3d", "gpu": gpu, "env": {"persisting_l2_max_bytes": 60 * MIB},
            "data": {"rows": rows, "projection": e3d.projection(rows), "config": {"l2_bytes": 96 * MIB}}}


def test_plot_synthetic(tmp_path):
    root = tmp_path / "e3d"
    (root / "rtx5090").mkdir(parents=True)
    (root / "rtx5090" / "20261013T000000Z.json").write_text(json.dumps(_doc()))
    (root / "rtx5090" / "20261013T010000Z.json").write_text(json.dumps({"data": {"rows": [{"D_mib": 1}]}}))  # other schema
    (root / "_status").mkdir()
    docs = plot_e3d.load_latest(root)
    assert list(docs) == ["rtx5090"] and plot_e3d.pdl_settings(docs["rtx5090"]) == [0, 1]
    out = plot_e3d.plot(docs["rtx5090"], tmp_path / "a.png", 1, 4)
    assert out.stat().st_size > 5000
    docs["rtx5090"]["data"].pop("config")   # older result without the L2 size
    assert plot_e3d.plot(docs["rtx5090"], tmp_path / "b.png", 0, 4).stat().st_size > 5000
    assert plot_e3d.main(["--results", str(root), "--out-dir", str(tmp_path / "o"), "--K", "4"]) == 0
    assert len(list((tmp_path / "o").glob("e3d_rtx5090_pdl*_K4.png"))) == 2
    g = plot_e3d.gain_series(docs["rtx5090"], 1, "S2", 4, "first_us")
    assert [x[0] for x in g] == [16, 48, 96] and g[0][1] == pytest.approx(100 * 503 / 902)
    assert plot_e3d.proj_series(docs["rtx5090"], 1, 4, 0.8, "speedup_on")[0][0] == 16


def test_plot_empty_dir(tmp_path):
    assert plot_e3d.main(["--results", str(tmp_path / "none")]) == 1
