import json

import pytest

from exps import e1_sweep as e1
from exps import plot_e1


def test_layer_math():
    q, s = e1.layer_bytes()
    assert (q, s) == (4 * 1024 * 1024, 256 * 1024)
    assert e1.layer_stride() % 256 == 0 and e1.layer_stride() == q + s
    assert e1.plan_layers(16) == 4            # 16 / 4.25 = 3.76
    assert e1.plan_layers(1024) == 241        # 1024 / 4.25 = 240.9
    assert e1.plan_layers(1) == 1
    assert e1.plan_layers(64) == 15


def test_schedule_chains_layers_across_tokens():
    seq = list(e1.schedule(3, 2))
    assert [x[0] for x in seq] == [0, 1, 2, 0, 1, 2]
    assert [(s, d) for _, s, d in seq] == [(0, 1), (1, 0), (0, 1), (1, 0), (0, 1), (1, 0)]
    assert len(list(e1.schedule(241, 10))) == 2410


def test_modes_and_tokens():
    assert e1.parse_modes("off,0.5,0.75,1.0") == [("off", None), ("0.5", 0.5), ("0.75", 0.75), ("1.0", 1.0)]
    with pytest.raises(ValueError):
        e1.parse_modes("1.5")
    assert e1.graph_tokens(1000) == 1000 and e1.graph_tokens(1001) == 1010 and e1.graph_tokens(1) == 10


def test_window_plan_clamp():
    p = e1.window_plan(100 << 20, 64 << 20, 72 << 20, 0.5)
    assert p["window_bytes"] == 64 << 20 and p["clamped"]
    assert p["persisted_bytes_est"] == 32 << 20
    p = e1.window_plan(16 << 20, 64 << 20, 72 << 20, 1.0)
    assert not p["clamped"] and p["persisted_bytes_est"] == 16 << 20


def _row(w, mode, us):
    return {"W_mib": w, "mode": mode, "us_per_token": us}


def test_stop_flag():
    weak = [_row(16, "off", 100), _row(16, "1.0", 95), _row(64, "off", 400), _row(64, "1.0", 380)]
    assert "STOP" in e1.stop_flag(weak, "rtx5090")
    assert e1.stop_flag(weak, "rtx3090") is None
    good = weak + [_row(32, "off", 200), _row(32, "0.75", 120)]
    assert e1.stop_flag(good, "rtx5090") is None
    assert e1.stop_flag([_row(256, "off", 1), _row(256, "1.0", 1)], "rtx5090") is None


def test_stop_flag_needs_pollution_and_cold_off_rows():
    weak = [_row(16, "off", 100), _row(16, "1.0", 95), _row(64, "off", 400), _row(64, "1.0", 380)]
    assert e1.stop_flag(weak, "rtx5090", pollute=False) is None
    hot = [dict(r, l2_hit_rate=0.9) if r["mode"] == "off" else r for r in weak]   # off already L2 resident
    assert e1.stop_flag(hot, "rtx5090") is None
    cold = [dict(r, l2_hit_rate=0.1) if r["mode"] == "off" else r for r in weak]
    assert "STOP" in e1.stop_flag(cold, "rtx5090")


def test_pollute_bytes_and_ncu_order():
    assert e1.pollute_bytes_for(96 << 20) == 192 << 20 and e1.pollute_bytes_for(None) == 192 << 20
    assert e1.pollute_bytes_for(96 << 20, 0) == 0 and e1.pollute_bytes_for(1000, 2.0) % 256 == 0
    rows = [{"W_mib": w, "mode": m} for w in (16, 64, 128, 1024) for m in ("off", "1.0")]
    order = [(r["W_mib"], r["mode"]) for r in e1.ncu_order(rows, 72.0, 96.0)]
    assert order[:4] == [(64, "off"), (64, "1.0"), (128, "off"), (128, "1.0")] and order[-1] == (1024, "1.0")
    assert [r["W_mib"] for r in e1.ncu_order(rows, None, None)] == [16, 16, 64, 64, 128, 128, 1024, 1024]


def test_ncu_args_and_inner_argv():
    a = e1.ncu_args(10)
    assert a == {"kernel_regex": "cr_gemv", "launch_skip": 30, "launch_count": 50}
    argv = e1.inner_argv(64, "0.75", "lut", "graph")
    assert argv[1:3] == ["-m", "exps.e1_sweep"] and "--inner" in argv and argv[argv.index("--W") + 1] == "64"
    assert "--pollute-bytes" not in argv
    assert e1.inner_argv(64, "off", "lut", "graph", 1 << 20)[-2:] == ["--pollute-bytes", str(1 << 20)]
    parsed = e1.parse_args(e1.inner_argv(64, "0.75", "lut", "graph", 4096)[3:])
    assert parsed.inner and parsed.modes == [("0.75", 0.75)] and parsed.W == [64] and parsed.pollute_bytes == 4096


def test_quick_args():
    a = e1.parse_args(["--quick"])
    assert a.W == [16, 64] and a.tokens == 100 and [m for m, _ in a.modes] == ["off", "1.0"]


def _doc(gpu, pers_mib, scale):
    rows = []
    for w in (16, 32, 64, 128, 256):
        for mode, f in (("off", 1.0), ("0.5", 0.9), ("1.0", 0.7)):
            rows.append({"W_mib": w, "L": w // 4, "bytes_per_token": w << 20, "mode": mode, "us_per_token": scale * w * f,
                         "dram_bytes_per_token": (w << 20) * f if gpu != "rtx3090" or mode == "off" else None})
    return {"schema": 1, "exp": "e1", "gpu": gpu, "env": {"persisting_l2_max_bytes": pers_mib << 20}, "data": {"rows": rows}}


def test_plot_synthetic(tmp_path):
    root = tmp_path / "e1"
    for gpu, pm, sc in (("rtx5090", 72, 0.6), ("pro6000-ws", 96, 0.55), ("rtx3090", 4, 1.1)):
        d = root / gpu
        d.mkdir(parents=True)
        (d / "20261013T000000Z.json").write_text(json.dumps(_doc(gpu, pm, sc)))
        (d / "20261013T010000Z.json").write_text(json.dumps(_doc(gpu, pm, sc * 2)))
    (root / "_status").mkdir()
    docs = plot_e1.load_latest(root)
    assert set(docs) == {"rtx5090", "pro6000-ws", "rtx3090"}
    assert docs["rtx5090"]["data"]["rows"][0]["us_per_token"] == pytest.approx(0.6 * 2 * 16)  # newest file wins
    out = tmp_path / "knee.png"
    assert plot_e1.plot(docs, out) == out
    assert out.stat().st_size > 5000
    assert plot_e1.main(["--results", str(root), "--out", str(tmp_path / "k2.png")]) == 0


def test_plot_control_only(tmp_path):
    root = tmp_path / "e1" / "rtx3090"
    root.mkdir(parents=True)
    (root / "20261013T000000Z.json").write_text(json.dumps(_doc("rtx3090", 4, 1.0)))
    out = plot_e1.plot(plot_e1.load_latest(tmp_path / "e1"), tmp_path / "c.png")
    assert out.exists()


def test_plot_empty_dir(tmp_path):
    assert plot_e1.main(["--results", str(tmp_path / "none")]) == 1


def test_chain_arena_layout_and_gain_on_cpu(monkeypatch):
    import torch
    from kernels import ternary
    calls = []
    monkeypatch.setattr(ternary, "gemv_ptr", lambda *a, **k: calls.append(a))
    L = 6
    ch = e1.Chain(L, "lut", device="cpu")
    assert ch.arena.numel() == L * e1.layer_stride() and ch.bytes_per_token == L * (4 * 1024 * 1024 + 256 * 1024)
    ch.tokens(2, 0)
    assert len(calls) == 2 * L
    variant, q_ptr, s_ptr, x_ptr, y_ptr, n, k, stream = calls[1]
    assert (variant, n, k) == ("lut", 4096, 4096) and q_ptr == ch.base + ch.stride and s_ptr == q_ptr + ch.qb
    assert x_ptr == ch.bufs[1].data_ptr() and y_ptr == ch.bufs[0].data_ptr()
    # layers are 256 B aligned and hold valid ternary data with rescaled scales
    for i in (0, 5):
        o = i * ch.stride
        assert o % 256 == 0  # the CUDA caching allocator aligns the base to 512 B
        q = ch.arena[o:o + ch.qb].view(torch.int32).view(4096, 256)
        t = ternary.unpack(q, 4096)
        assert set(t.unique().tolist()) <= {-1, 0, 1} and 0.6 < (t != 0).float().mean() < 0.73
    # activation scale stays near 1 through the chain (GAIN)
    x = ch.bufs[0].float()
    rms0 = x.pow(2).mean().sqrt().item()
    for i in range(L):
        o = i * ch.stride
        q = ch.arena[o:o + ch.qb].view(torch.int32).view(4096, 256)
        s = ch.arena[o + ch.qb:o + ch.qb + ch.sb].view(torch.float16).view(4096, 32)
        x = ternary.ref_gemv(q, s, x.half())
    assert 0.3 < x.pow(2).mean().sqrt().item() / rms0 < 3.0


def test_chain_pollutes_after_each_token(monkeypatch):
    from kernels import ternary
    events = []
    monkeypatch.setattr(ternary, "gemv_ptr", lambda *a, **k: events.append("g"))
    ch = e1.Chain(3, "lut", device="cpu", pollute_bytes=1024)
    assert ch.pbuf.numel() == 256 and ch.pollute_bytes == 1024
    ch.tokens(2, 0)
    assert events == ["g", "g", "g"] * 2 and int(ch.pbuf[0]) == 2   # one add_ per token
    plain = e1.Chain(2, "lut", device="cpu")
    assert plain.pbuf is None


def test_repeats_for_small_w():
    assert e1.repeats_for(16, 3) == 7 and e1.repeats_for(72, 3) == 7 and e1.repeats_for(80, 3) == 3
    assert e1.repeats_for(16, 10) == 10 and e1.repeats_for(16, 1, small=1) == 1


def test_gross_deltas_and_noise():
    def row(w, mode, gross, all_):
        return {"W_mib": w, "mode": mode, "us_per_token_gross": gross, "us_per_token": gross - 5.0, "us_per_token_all": all_}
    rows = [row(16, "off", 100.0, [99.0, 100.0, 101.0]), row(16, "1.0", 90.0, [89.5, 90.0, 90.5]),
            row(32, "off", 100.0, [98.0, 100.0, 102.0]), row(32, "1.0", 99.0, [97.0, 99.0, 101.0]),
            row(48, "off", 100.0, [100.0]), row(48, "1.0", 90.0, [90.0]), {"W_mib": 64, "mode": "1.0", "us_per_token_gross": None}]
    e1.gross_deltas(rows)
    assert rows[1]["gross_delta_vs_off_us"] == 10.0 and rows[1]["delta_noise_us"] == 2.0 and rows[1]["delta_above_noise"] is True
    assert rows[3]["gross_delta_vs_off_us"] == 1.0 and rows[3]["delta_above_noise"] is False
    assert rows[5]["gross_delta_vs_off_us"] == 10.0 and rows[5]["delta_noise_us"] is None and rows[5]["delta_above_noise"] is None
    assert "gross_delta_vs_off_us" not in rows[0] and "gross_delta_vs_off_us" not in rows[6]


def test_small_w_repeats_arg():
    assert e1.parse_args(["--dry-run"]).small_w_repeats == 7
    a = e1.parse_args(["--quick", "--dry-run"])
    assert a.repeats == 1 and a.small_w_repeats == 1


def test_stop_flag_prefers_ncu_gemv_time():
    def rows(off_net, on_net, off_ncu, on_ncu):
        return [dict(_row(16, "off", off_net), ncu_us_per_token=off_ncu, counters_ok=True),
                dict(_row(16, "1.0", on_net), ncu_us_per_token=on_ncu, counters_ok=True)]
    # net times show no gain (pollution noise), ncu GEMV time shows 50%: no stop
    assert e1.stop_flag(rows(100, 99, 40, 20), "rtx5090") is None
    # net times show 50%, ncu shows none: stop, basis ncu
    assert "basis ncu" in e1.stop_flag(rows(100, 50, 40, 39), "rtx5090")
    # counters failed: fall back to the net time
    bad = [dict(r, counters_ok=False) for r in rows(100, 99, 40, 20)]
    assert "basis net" in e1.stop_flag(bad, "rtx5090")


def test_ncu_deltas():
    def row(mode, ncu, ok=True):
        return {"W_mib": 16, "mode": mode, "ncu_us_per_token": ncu, "counters_ok": ok}
    rows = [row("off", 40.0), row("0.5", 30.0), row("1.0", 20.0, ok=False)]
    e1.ncu_deltas(rows)
    assert rows[1]["ncu_delta_vs_off_us"] == 10.0 and rows[1]["ncu_gain_vs_off"] == 0.25
    assert "ncu_delta_vs_off_us" not in rows[0] and "ncu_delta_vs_off_us" not in rows[2]
