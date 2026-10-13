import urllib.error

import pytest

from exps import e2_hybrid_bf16 as e2
from exps import hf_ids, modelspec


def test_config_facts_08b():
    f = modelspec.facts(modelspec.TEXT_CFG["0.8B"])
    assert (f["n_layers"], f["n_gdn"], f["n_full"]) == (24, 18, 6)
    assert f["state_bytes"] == 18 * 16 * 128 * 128 * 4 == 18874368   # 18.9 MB
    assert f["kv_bytes_per_token"] == 6 * 2 * 2 * 256 * 2 == 12288
    assert f["head_bytes"] == 248320 * 1024 * 2                      # 508.6 MB
    assert 1.4e9 < f["weight_bytes"] < 1.6e9


def test_config_facts_27b():
    f = modelspec.facts(modelspec.TEXT_CFG["27B"])
    assert (f["n_gdn"], f["n_full"]) == (48, 16)
    assert f["kv_bytes_per_token"] == 65536
    assert abs(f["body_weight_bytes"] / 2 - 24.35e9) < 0.1e9          # 24.35B language backbone


def test_ctx_order_and_cache_len():
    assert e2.ctx_order([0, 4096, 16384, 32768, 65536]) == [0, 65536, 4096, 16384, 32768]
    assert e2.ctx_order([5]) == [5]
    assert e2.cache_len_for(0, 512) % 64 == 0 and e2.cache_len_for(0, 512) >= 512 + e2.MARGIN
    assert e2.cache_len_for(4096, 512) > e2.cache_len_for(0, 512)


def test_analytic_static_vs_dynamic():
    a = e2.analytic("0.8B", 4096, 512, static=True)
    assert a["kv_bytes"] == 12288 * a["cache_len"]
    b = e2.analytic("0.8B", 4096, 512, static=False)
    assert b["kv_bytes"] == 12288 * (4096 + 256)
    assert a["gdn_state_bytes"] == 18874368


def _rows():
    st = 18874368
    rows = []
    for ctx, dram in ((0, 1.60e9), (4096, 1.60e9 + 50e6), (16384, 1.60e9 + 200e6)):
        rows.append({"ctx": ctx, "dram_bytes_per_token": dram, "us_per_token": 100.0 + ctx / 100.0,
                     "analytic": {"gdn_state_bytes": st, "weight_bytes": 1.5e9, "kv_bytes": 12288 * ctx}})
    return rows


def test_split_by_diffing_contexts():
    rows = _rows()
    e2.split_bytes(rows)
    st = 18874368
    assert rows[0]["split"] == {"weights": 1.60e9 - 2 * st, "kv": 0.0, "gdn_state": 2 * st}
    assert rows[2]["split"]["kv"] == pytest.approx(200e6)
    assert rows[2]["split"]["weights"] == rows[0]["split"]["weights"]
    assert rows[1]["split_read_only"]["gdn_state"] == st


def test_split_needs_ctx0_and_bytes():
    rows = [{"ctx": 4096, "dram_bytes_per_token": 1e9, "analytic": {"gdn_state_bytes": 1}}]
    e2.split_bytes(rows)
    assert rows[0]["split"] is None


def test_fit_recovers_floor_and_bandwidth():
    floor_us, bw = 240.0, 1400.0   # 10 us/layer x 24, GB/s
    pts = [(b, floor_us + b / (bw * 1e3)) for b in (1.5e9, 1.55e9, 1.7e9, 2.4e9)]
    f = e2.fit_floor_bw(pts, 24)
    assert f["floor_us_per_layer"] == pytest.approx(10.0, rel=1e-6)
    assert f["eff_bw_gbps"] == pytest.approx(bw, rel=1e-6)
    assert f["r2"] == pytest.approx(1.0)


def test_fit_degenerate():
    assert e2.fit_floor_bw([(1e9, 100.0)], 24) is None
    assert e2.fit_floor_bw([(1e9, 100.0), (1e9, 110.0)], 24) is None
    f = e2.fit_floor_bw([(1e9, 100.0), (2e9, 90.0)], 24)  # negative slope: no bandwidth
    assert f["eff_bw_gbps"] is None


def test_fit_rows_falls_back_to_analytic():
    rows = [{"ctx": c, "us_per_token": 300.0 + c / 100.0, "dram_bytes_per_token": None,
             "analytic": {"weight_bytes": 1.5e9, "gdn_state_bytes": 1.9e7, "kv_bytes": 12288 * (c + 576)}} for c in (0, 4096, 16384)]
    f = e2.fit_rows(rows, 24)
    assert f["bytes_source"] == "analytic_bytes" and f["n_points"] == 3


def test_gdn_impl_detection():
    import torch

    def torch_rule(*a):
        return None

    fla_rule = type("F", (), {"__module__": "fla.ops.gated_delta_rule"})()

    class Layer(torch.nn.Module):
        def __init__(self, fn):
            super().__init__()
            self.recurrent_gated_delta_rule = fn

    assert e2.gdn_impl(torch.nn.Sequential(torch.nn.Linear(1, 1), Layer(torch_rule))) == "torch_fallback"
    assert e2.gdn_impl(torch.nn.Sequential(Layer(fla_rule))) == "fla"
    assert e2.gdn_impl(torch.nn.Linear(1, 1)) == "unknown"
    info = {}
    e2.record_gdn_impl(Layer(torch_rule), info)
    assert info["gdn_impl"] == "torch_fallback" and set(info) >= {"fla_available", "causal_conv1d_available"}


def test_parse_ctx():
    assert e2.parse_ctx("0,4096") == [0, 4096]


# ---- hf_ids

def _fake(table):
    def fetch(url):
        for k, v in table.items():
            if url.startswith(k):
                if isinstance(v, Exception):
                    raise v
                return v
        raise urllib.error.HTTPError(url, 404, "nf", None, None)
    return fetch


def test_resolve_offline_is_pinned():
    r = hf_ids.resolve("qwen", "2B", offline=True)
    assert r["repo"] == "Qwen/Qwen3.5-2B" and r["source"] == "pinned" and r["exists"] is None


def test_resolve_qwen_exists_and_missing():
    ok = hf_ids.resolve("qwen", "0.8B", fetch=_fake({hf_ids.API + "/models/Qwen/Qwen3.5-0.8B": {"siblings": []}}))
    assert ok["exists"] is True and ok["source"] == "api"
    table = {hf_ids.API + "/models/Qwen/Qwen3.5-0.8B": urllib.error.HTTPError("u", 404, "nf", None, None),
             hf_ids.API + "/models?search=Qwen3.5": [{"id": "Qwen/Qwen3.5-0.8B-Instruct"}, {"id": "Qwen/Qwen3.5-0.8B-Base"},
                                                      {"id": "Qwen/Qwen3.5-2B"}]}
    gone = hf_ids.resolve("qwen", "0.8B", fetch=_fake(table))
    assert gone["exists"] is False and gone["alternatives"] == ["Qwen/Qwen3.5-0.8B-Instruct"]


def test_resolve_network_error_falls_back():
    r = hf_ids.resolve("bonsai2", "PQ2_0", fetch=_fake({hf_ids.API: OSError("no route")}))
    assert r["exists"] is None and r["source"] == "pinned" and "unreachable" in r["note"]
    assert r["file"] == "Ternary-Bonsai-2-27B-PQ2_0.gguf"


def test_resolve_bonsai_file_check():
    repo = hf_ids.BONSAI2_GGUF[0]
    good = {hf_ids.API + "/models/" + repo: {"siblings": [{"rfilename": "Ternary-Bonsai-2-27B-PTQ1_0.gguf"}]}}
    assert hf_ids.resolve("bonsai2", "PTQ1_0", fetch=_fake(good))["exists"] is True
    assert hf_ids.resolve("bonsai2", "PQ2_0", fetch=_fake(good))["exists"] is False


def test_resolve_draft_fallback():
    fb = hf_ids.DRAFT_GGUF_FALLBACK
    table = {hf_ids.API + "/models/" + hf_ids.DRAFT_GGUF[0]: urllib.error.HTTPError("u", 404, "nf", None, None),
             hf_ids.API + "/models/" + fb[0]: {"siblings": [{"rfilename": fb[1]}]}}
    r = hf_ids.resolve("draft", fetch=_fake(table))
    assert r["repo"] == fb[0] and r["exists"] is True


def test_gdn_impl_reads_hub_wrapper_closure(monkeypatch):
    """transformers 5.x: module-level GDN functions wrapped with functools.wraps keep the torch function's __module__."""
    import functools
    import sys
    import types
    import torch

    def torch_fn():
        return None
    torch_fn.__module__ = "transformers.models.qwen3_5.modeling_qwen3_5"

    def wrap(implementation):
        @functools.wraps(torch_fn)
        def wrapped(*a, **k):
            return implementation(*a, **k)
        return wrapped

    fla_impl = type("F", (), {"__module__": "fla.ops.gated_delta_rule", "__call__": lambda self: None})()
    mod = types.ModuleType("fake_modeling")
    monkeypatch.setitem(sys.modules, "fake_modeling", mod)

    class GatedDeltaNet(torch.nn.Module):
        pass
    GatedDeltaNet.__module__ = "fake_modeling"
    model = torch.nn.Sequential(GatedDeltaNet())
    for names_impl, want in ((torch_fn, "torch_fallback"), (fla_impl, "fla")):
        for n in e2.GDN_FNS:
            setattr(mod, n, wrap(names_impl))
        assert e2.gdn_impl(model) == want
    mod.torch_chunk_gated_delta_rule = wrap(torch_fn)   # one of the two on the torch path: not fla
    assert e2.gdn_impl(model) == "torch_fallback"
    monkeypatch.delattr(mod, "torch_chunk_gated_delta_rule")
    monkeypatch.delattr(mod, "torch_recurrent_gated_delta_rule")
    assert e2.gdn_impl(model) == "unknown"
