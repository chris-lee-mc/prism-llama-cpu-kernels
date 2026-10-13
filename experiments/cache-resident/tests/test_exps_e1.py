import json

import pytest

from exps import e1_sweep as e1
from exps import plot_e1

MIB = 1 << 20


def test_layer_math():
    q, s = e1.layer_bytes(4096, 4096)
    assert (q, s) == (4 * MIB, 256 * 1024)
    assert e1.layer_stride(4096, 4096) == q + s and e1.layer_stride(17408, 5120) % 256 == 0
    assert e1.plan_hot(16) == 4 and e1.plan_hot(1024) == 241 and e1.plan_hot(1) == 1 and e1.plan_hot(64) == 15


def test_plan_cold():
    assert e1.plan_cold(0) == []
    c = e1.plan_cold(512)
    assert len(c) == 23 and c[:4] == [(17408, 5120), (17408, 5120), (5120, 17408), (17408, 5120)]
    assert abs(e1.set_bytes(c) - 512 * MIB) < 0.05 * 512 * MIB


def test_modes_and_limits():
    assert e1.parse_modes("off,0.5,1.0L") == [("off", None, None), ("0.5", 0.5, "max"), ("1.0L", 1.0, "W")]
    with pytest.raises(ValueError):
        e1.parse_modes("1.5")
    assert e1.limit_for(None, 10, 60) == 0 and e1.limit_for("max", 10, 60) == 60
    assert e1.limit_for("W", 10, 60) == 10 and e1.limit_for("W", 100, 60) == 60


def test_window_plan_and_graph_sizes():
    p = e1.window_plan(200 * MIB, 128 * MIB, 60 * MIB, 0.5)
    assert p["window_bytes"] == 128 * MIB and p["clamped"] and p["persisted_bytes_est"] == 60 * MIB
    assert not e1.window_plan(16 * MIB, 128 * MIB, 60 * MIB, 1.0)["clamped"]
    assert e1.tokens_per_graph(4) == 20 and e1.tokens_per_graph(241) == 8 and e1.tokens_per_graph(5000) == 1
    assert e1.replays_for(200, 8) == 25 and e1.replays_for(1, 20) == 1


def test_paired_and_deltas():
    p = e1.paired([10, 11, 12], [5, 6, 9])
    assert p["delta"] == 5 and p["all"] == [5, 5, 3] and p["noise"] == 1.0 and p["above"] is True
    assert e1.paired([10], [9])["noise"] is None and e1.paired([], []) is None

    def row(c, mode, us, hot):
        return {"W_mib": 16, "C_mib": c, "mode": mode, "us_per_token": sorted(us)[1], "us_per_token_all": us,
                "hot_us": sorted(hot)[1], "hot_us_all": hot}
    rows = [row(512, "off", [500, 510, 520], [20, 21, 22]), row(512, "1.0", [490, 500, 510], [10, 11, 12]),
            row(0, "off", [20, 20, 20], [20, 20, 20]), row(0, "1.0", [19, 20, 21], [19, 20, 21])]
    e1.add_deltas(rows)
    r = rows[1]
    assert r["delta_vs_off_us"] == 10 and r["gain_vs_off"] == pytest.approx(10 / 510)
    assert r["hot_delta_vs_off_us"] == 10 and r["hot_gain_vs_off"] == pytest.approx(10 / 21) and r["hot_delta_above_noise"]
    assert rows[3]["hot_delta_vs_off_us"] == 0 and rows[3]["hot_delta_above_noise"] is False
    assert "delta_vs_off_us" not in rows[0]


def _r(w, c, mode, gain):
    return {"W_mib": w, "C_mib": c, "mode": mode, "hot_gain_vs_off": gain}


def test_stop_check():
    good = [_r(16, 512, "off", None), _r(16, 512, "1.0", 0.05), _r(48, 512, "0.5", 0.35), _r(128, 512, "1.0", 0.9)]
    met, note = e1.stop_check(good, "rtx5090", 60.0, 512)
    assert met is True and "W=48" in note
    weak = [_r(16, 512, "1.0", 0.05), _r(48, 512, "0.5", 0.1), _r(128, 512, "1.0", 0.9)]
    met, note = e1.stop_check(weak, "rtx5090", 60.0, 512)
    assert met is False and note.startswith("STOP")
    assert e1.stop_check(weak, "rtx3090", 60.0, 512)[0] is None
    assert e1.stop_check(weak, "rtx5090", None, 512)[0] is False   # default 72 MiB
    assert e1.stop_check(weak, "rtx5090", 60.0, 0)[0] is None       # no C=0 rows


def test_quick_args_and_plan():
    a = e1.parse_args(["--quick"])
    assert a.W == [16, 64] and a.tokens == 50 and a.rounds == 3 and [m for m, _, _ in a.modes] == ["off", "1.0"]
    a = e1.parse_args(["--dry-run"])
    assert a.C == 512 and a.rounds == 5 and a.calib
    plan = e1.plan_json(a)
    assert plan["cold_layers"] == 23 and plan["layers"][-1]["L"] == 241 and plan["modes"][-1] == "1.0L"


def test_layers_on_cpu(monkeypatch):
    import torch
    from kernels import ternary
    calls = []
    monkeypatch.setattr(ternary, "gemv_ptr", lambda *a, **k: calls.append(a))
    lay = e1.Layers([(4096, 4096)] * 3 + [(256, 512)], "lut", device="cpu")
    a = lay.arena
    assert a.offs[1] == e1.layer_stride(4096, 4096) and all(o % 256 == 0 for o in a.offs)
    lay.launch(0)
    assert len(calls) == 4
    v, q, s, x, y, n, k, st = calls[3]
    assert (v, n, k) == ("lut", 256, 512) and (q, s) == a.ptrs(3) and x == lay.x.data_ptr() and y == lay.y.data_ptr()
    q3, s3 = a.qs(3)
    t = ternary.unpack(q3, 512)
    assert set(t.unique().tolist()) <= {-1, 0, 1} and 0.6 < (t != 0).float().mean() < 0.73
    y = ternary.ref_gemv(q3, s3, lay.x[:512])
    assert torch.isfinite(y).all() and 0.3 < y.pow(2).mean().sqrt().item() < 3.0   # fixed x keeps outputs O(1)


def _doc(gpu, pm, scale):
    rows = []
    for c in (512, 0):
        for w in (16, 32, 64, 128, 256):
            for mode, f in (("off", 1.0), ("0.5", 0.8), ("1.0", 0.6), ("1.0L", 0.55)):
                hot = scale * w * f
                r = {"W_mib": w, "C_mib": c, "mode": mode, "hot_us": hot, "hot_us_all": [hot * 0.98, hot, hot * 1.03],
                     "us_per_token": hot + (300 if c else 0)}
                if mode != "off":
                    r.update(hot_gain_vs_off=1 - f, hot_delta_vs_off_us=scale * w * (1 - f), hot_delta_noise_us=0.5)
                rows.append(r)
    return {"schema": 1, "exp": "e1", "gpu": gpu, "env": {"persisting_l2_max_bytes": pm * MIB}, "data": {"rows": rows}}


def test_plot_synthetic(tmp_path):
    root = tmp_path / "e1"
    for gpu, pm, sc in (("rtx5090", 60, 0.6), ("pro6000-ws", 96, 0.55), ("rtx3090", 4, 1.1)):
        d = root / gpu
        d.mkdir(parents=True)
        (d / "20261013T000000Z.json").write_text(json.dumps(_doc(gpu, pm, sc)))
        (d / "20261013T010000Z.json").write_text(json.dumps(_doc(gpu, pm, sc * 2)))
        (d / "20261013T020000Z.json").write_text(json.dumps({"data": {"rows": [{"W_mib": 16, "mode": "off"}]}}))  # old schema
    (root / "_status").mkdir()
    docs = plot_e1.load_latest(root)
    assert set(docs) == {"rtx5090", "pro6000-ws", "rtx3090"}
    assert docs["rtx5090"]["data"]["rows"][0]["hot_us"] == pytest.approx(0.6 * 2 * 16)  # newest hot/cold file wins
    out = tmp_path / "knee.png"
    assert plot_e1.plot(docs, out) == out and out.stat().st_size > 5000
    assert plot_e1.main(["--results", str(root), "--out", str(tmp_path / "k2.png")]) == 0
    assert plot_e1.mode_order(["1.0L", "off", "1.0", "0.5"]) == ["off", "0.5", "1.0", "1.0L"]


def test_plot_control_only(tmp_path):
    root = tmp_path / "e1" / "rtx3090"
    root.mkdir(parents=True)
    (root / "20261013T000000Z.json").write_text(json.dumps(_doc("rtx3090", 4, 1.0)))
    assert plot_e1.plot(plot_e1.load_latest(tmp_path / "e1"), tmp_path / "c.png").exists()


def test_plot_empty_dir(tmp_path):
    assert plot_e1.main(["--results", str(tmp_path / "none")]) == 1
