import json

import pytest

from exps import proj_hybrid as ph


def _m():
    return ph.ModelSpec("t", 10, 1_000_000, 500_000, 100_000, 10)


def _hw(**kw):
    d = dict(name="h", bw_dram_gbps=1.0, bw_l2_gbps=4.0, floor_us_per_layer=5.0, persist_max_bytes=300_000)
    d.update(kw)
    return ph.HwSpec(**d)


def test_bytes_over_bandwidth_identity():
    # 1 GB/s = 1000 bytes/us. total = 1.0M + 0.5M + 2*0.1M + 10*1000 = 1.71M bytes
    r = ph.decode_us(_m(), _hw(), ctx=1000)
    assert r["dram_bytes"] == 1_710_000 and r["l2_bytes"] == 0
    assert r["floor_us"] == 50.0
    assert r["us"] == pytest.approx(50.0 + 1710.0)


def test_persisted_bytes_move_to_l2():
    r = ph.decode_us(_m(), _hw(), ctx=1000, persisted_bytes=400_000, hit=0.5)   # capped at 300k, x0.5 = 150k
    assert r["l2_bytes"] == 150_000 and r["dram_bytes"] == 1_560_000
    assert r["us"] == pytest.approx(50.0 + 1560.0 + 150_000 / 4000.0)
    assert ph.decode_us(_m(), _hw(), persisted_bytes=10**12)["dram_bytes"] >= 0


def test_ctx_adds_kv_only():
    a, b = ph.decode_us(_m(), _hw(), 0), ph.decode_us(_m(), _hw(), 2000)
    assert b["dram_bytes"] - a["dram_bytes"] == 20_000


def test_spec_formula():
    k, td, tv = 6, 100.0, 5000.0
    assert ph.spec_tok_s(0.0, k, td, tv) == pytest.approx(1e6 / (k * td + tv))
    assert ph.spec_tokens_per_verify(1.0, k) == k + 1
    assert ph.spec_tokens_per_verify(0.5, 3) == pytest.approx((1 - 0.5 ** 4) / 0.5)
    assert ph.spec_tok_s(0.8, 4, td, tv) > ph.spec_tok_s(0.5, 4, td, tv)


def test_models_registry():
    assert set(ph.MODELS) == {"qwen3.5-0.8b", "qwen3.5-2b", "ternary-0.8b-g128", "bonsai2-27b-pq2_0"}
    q = ph.MODELS["qwen3.5-0.8b"]
    assert q.n_layers == 24 and q.state_bytes == 18874368 and q.kv_bytes_per_ctx_token == 12288
    t = ph.MODELS["ternary-0.8b-g128"]
    assert t.weight_bytes == pytest.approx(q.weight_bytes * 2.125 / 16, rel=1e-3)
    assert ph.MODELS["bonsai2-27b-pq2_0"].n_layers == 64


def test_hw_from_e2_result(tmp_path):
    p = tmp_path / "r.json"
    p.write_text(json.dumps({"gpu": "rtx5090", "env": {"gpu_tag": "rtx5090", "persisting_l2_max_bytes": 75497472},
                             "data": {"fit": {"floor_us_per_layer": 11.5, "eff_bw_gbps": 1350.0}}}))
    hw = ph.hw_from_results(p)
    assert (hw.floor_us_per_layer, hw.bw_dram_gbps, hw.persist_max_bytes, hw.name) == (11.5, 1350.0, 75497472, "rtx5090")
    assert hw.bw_l2_gbps == pytest.approx(1350.0 * ph.L2_BW_RATIO)
    assert ph.hw_from_results(tmp_path).floor_us_per_layer == 11.5   # directory: newest file


def test_hw_from_e2_result_flags_torch_fallback(tmp_path):
    p = tmp_path / "r.json"
    p.write_text(json.dumps({"data": {"fit": {"floor_us_per_layer": 11.5, "eff_bw_gbps": 1350.0, "gdn_impl": "torch_fallback"}}}))
    notes = []
    with pytest.warns(UserWarning, match="torch_fallback"):
        hw = ph.hw_from_results(p, notes=notes)
    assert hw.floor_us_per_layer == 11.5 and len(notes) == 1
    p.write_text(json.dumps({"data": {"fit": {"floor_us_per_layer": 11.5, "eff_bw_gbps": 1350.0, "gdn_impl": "fla"}}}))
    notes = []
    ph.hw_from_results(p, notes=notes)
    assert notes == []


def test_hw_from_toy_result(tmp_path):
    p = tmp_path / "toy.json"
    p.write_text(json.dumps({"data": {"us_per_layer_graph": 9.0, "eff_gbps": 1200.0}}))
    hw = ph.hw_from_results(p)
    assert hw.floor_us_per_layer == 9.0 and hw.bw_dram_gbps == 1200.0
    p.write_text(json.dumps({"data": {}}))
    with pytest.raises(ValueError):
        ph.hw_from_results(p)


def test_hw_from_toy_prefers_floor_and_net_bandwidth(tmp_path):
    p = tmp_path / "toy.json"
    row = {"us_per_layer_graph": 9.0, "eff_gbps": 1200.0, "floor_us_per_layer": 4.0, "eff_gbps_net": 1500.0, "l2_assisted": False}
    p.write_text(json.dumps({"data": {"rows": [row]}}))
    notes = []
    hw = ph.hw_from_results(p, notes=notes)
    assert hw.floor_us_per_layer == 4.0 and hw.bw_dram_gbps == 1500.0 and notes == []
    row["l2_assisted"] = True
    p.write_text(json.dumps({"data": {"rows": [row]}}))
    with pytest.warns(UserWarning, match="L2-assisted"):
        hw = ph.hw_from_results(p, notes=notes)
    assert hw.floor_us_per_layer == 4.0 and len(notes) == 1


def test_hw_from_toy_legacy_warns_double_count(tmp_path):
    p = tmp_path / "toy.json"
    p.write_text(json.dumps({"data": {"us_per_layer_graph": 9.0, "eff_gbps": 1200.0}}))
    with pytest.warns(UserWarning, match="twice"):
        assert ph.hw_from_results(p).floor_us_per_layer == 9.0


def test_project_gain_is_nonnegative():
    rows = ph.project(list(ph.MODELS), _hw(bw_dram_gbps=1400.0, bw_l2_gbps=4200.0, floor_us_per_layer=12.0,
                                           persist_max_bytes=72 << 20), [0, 32768], 1.0)
    assert len(rows) == 8 and all(r["gain_pct"] >= 0 for r in rows)


def test_cli_write_result(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("CR_RESULTS_DIR", str(tmp_path))
    monkeypatch.setenv("CR_GPU_TAG", "cpu")
    assert ph.main(["--floor-us", "12", "--bw-gbps", "1400", "--persist-mb", "72", "--l2-gbps", "5000", "--write-result"]) == 0
    files = list((tmp_path / "proj_hybrid" / "cpu").glob("*.json"))
    assert len(files) == 1
    doc = json.loads(files[0].read_text())
    assert doc["data"]["config"]["reimplementation"] is True and doc["data"]["rows"]


def test_hw_from_results_dir_skips_partial_and_raw(tmp_path):
    d = tmp_path / "e2_08b" / "gpu"
    (d / "raw").mkdir(parents=True)
    (d / "logs").mkdir()
    final = {"data": {"fit": {"floor_us_per_layer": 11.5, "eff_bw_gbps": 1350.0}}}
    other = {"data": {"fit": {"floor_us_per_layer": 99.0, "eff_bw_gbps": 1.0}}}
    (d / "20261013T000000Z.json").write_text(json.dumps(final))
    (d / "raw" / "partial.json").write_text(json.dumps(other))     # sorts after the final file by path
    (d / "logs" / "z.json").write_text(json.dumps(other))
    (d / "partial.json").write_text(json.dumps(other))
    assert ph.hw_from_results(tmp_path).floor_us_per_layer == 11.5
    (d / "20261013T000000Z.json").unlink()
    with pytest.raises(FileNotFoundError):
        ph.hw_from_results(tmp_path)
