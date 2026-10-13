import pytest
import torch

from exps import e3_residency as e3
from exps import int8_head


def test_head_bytes_match_spec():
    v, h = 248320, 1024
    assert e3.head_bytes("bf16_full", v, h)["weight"] == 508559360
    assert e3.head_bytes("int8_full", v, h)["weight"] == 254279680
    assert e3.head_bytes("int8_48k", v, h)["weight"] == 49152 * 1024 == 50331648
    assert e3.head_bytes("bf16_48k", v, h)["weight"] == 100663296
    assert e3.head_bytes("int8_48k", v, h)["scale"] == 49152 * 4
    assert e3.head_bytes("bf16_full", v, h)["scale"] == 0


def test_arena_layout_and_windows():
    lay = e3.arena_layout(18874368, 50331648, 196608)
    assert lay["head_off"] % 256 == 0 and lay["scale_off"] % 256 == 0 and lay["total"] % 256 == 0
    assert lay["head_off"] >= 18874368 and lay["scale_off"] >= lay["head_off"] + 50331648
    assert e3.window_region("off", lay) is None
    assert e3.window_region("gdn", lay) == (0, 18874368)
    assert e3.window_region("both", lay) == (0, lay["total"])
    off, n = e3.window_region("head", lay)
    assert off == lay["head_off"] and off + n == lay["total"]
    with pytest.raises(ValueError):
        e3.window_region("nope", lay)


def _rows():
    r = lambda c, cond, us, dram, pb: {"ctx": c, "head": "int8_48k", "condition": cond, "us_per_token": us,
                                       "dram_bytes_per_token": dram, "persisted_bytes": pb}
    return [r(0, "off", 1000.0, 1.6e9, 0), r(0, "gdn", 990.0, 1.58e9, 18874368), r(0, "both", 950.0, 1.53e9, 69e6)]


def test_analyze_go_no_go():
    ana = e3.analyze(_rows())
    both = next(e for e in ana["per_config"] if e["condition"] == "both")
    assert both["time_gain_pct"] == pytest.approx(5.0)
    assert both["dram_saved_bytes_per_token"] == pytest.approx(70e6)
    assert both["saved_per_persisted_byte"] == pytest.approx(70e6 / 69e6)
    assert ana["go_no_go"]["go"] is True
    bad = _rows()
    bad[2]["dram_bytes_per_token"] = 1.59e9
    assert e3.analyze(bad)["go_no_go"]["go"] is False


def test_analyze_without_counters():
    rows = _rows()
    for r in rows:
        r["dram_bytes_per_token"] = None
    ana = e3.analyze(rows)
    assert ana["go_no_go"] is None
    assert all("time_gain_pct" in e for e in ana["per_config"])


# ---- state re-pointing with fake HF-like caches

class _Layer:
    def __init__(self, t):
        self.recurrent_states = t


class _LayerCache:
    def __init__(self):
        self.layers = [_Layer(torch.randn(2, 3, 4)), _Layer(None), _Layer(torch.randn(2, 3, 4))]


class _ListCache:
    def __init__(self):
        self.recurrent_states = [torch.randn(5, 4), None, torch.randn(5, 4)]


def test_locate_layer_cache():
    c = _LayerCache()
    slots = e3.locate_states(c)
    assert len(slots) == 2 and all(s.kind == "attr" for s in slots)


class _DictLayer:
    """transformers 5.x style: per-layer dict, updated in place with copy_."""
    def __init__(self, t):
        self.recurrent_states = {0: t}


class _DictLayerCache:
    def __init__(self):
        self.layers = [_DictLayer(torch.randn(2, 3, 4)), type("Full", (), {})(), _DictLayer(torch.randn(2, 3, 4))]


def test_locate_per_layer_dict_and_list():
    slots = e3.locate_states(_DictLayerCache())
    assert len(slots) == 2 and all(s.kind == "item" and s.key == 0 and s.name == "recurrent_states" for s in slots)

    class LL:
        recurrent_states = [torch.zeros(3), None]
    c = type("C", (), {"layers": [LL()]})()
    assert [(s.kind, s.key) for s in e3.locate_states(c)] == [("item", 0)]


def test_dict_layer_inplace_update_stays_in_arena():
    c = _DictLayerCache()
    slots = e3.locate_states(c)
    arena = torch.zeros(4096, dtype=torch.uint8)
    views = e3.repoint(slots, arena)
    assert all(e3.in_arena(ly.recurrent_states[0], arena) for ly in c.layers if hasattr(ly, "recurrent_states"))

    def step():
        for ly in c.layers:
            if hasattr(ly, "recurrent_states"):
                ly.recurrent_states[0].copy_(ly.recurrent_states[0] * 2.0 + 1.0)
    assert e3.settle_states(slots, views, arena, step) == "views_inplace"


def test_dict_layer_reassigning_update_gets_pinned():
    c = _DictLayerCache()
    slots = e3.locate_states(c)
    arena = torch.zeros(4096, dtype=torch.uint8)
    views = e3.repoint(slots, arena)

    def step():
        for ly in c.layers:
            if hasattr(ly, "recurrent_states"):
                ly.recurrent_states[0] = ly.recurrent_states[0] + 1.0
    assert e3.settle_states(slots, views, arena, step) == "views_pinned_copy"
    step()
    assert all(e3.in_arena(ly.recurrent_states[0], arena) for ly in c.layers if hasattr(ly, "recurrent_states"))


def test_arena_view_copy_is_a_kernel_not_memcpy(monkeypatch):
    arena = torch.zeros(1024, dtype=torch.uint8)
    t = torch.randn(4, 5)
    (v,) = e3.repoint([e3.Slot("attr", type("L", (), {"recurrent_states": t})(), "recurrent_states")], arena)
    assert isinstance(v, e3.ArenaView) and e3.in_arena(v, arena)
    calls = []
    real = torch.add
    monkeypatch.setattr(torch, "add", lambda *a, **k: (calls.append(1), real(*a, **k))[1])
    src = torch.randn(4, 5)
    v.copy_(src)
    assert calls == [1] and torch.equal(v, src)
    v.copy_(torch.randn(4, 5, dtype=torch.float64))   # other dtype: stock copy_ (already a cast kernel)
    v.copy_(torch.tensor(3.0))                          # broadcast: stock copy_
    assert calls == [1] and type(v * 2) is torch.Tensor


def test_graph_node_counts_stub(monkeypatch):
    import sys
    import types
    NT = types.SimpleNamespace(cudaGraphNodeTypeKernel=1, cudaGraphNodeTypeMemcpy=2, cudaGraphNodeTypeMemset=3)
    types_ = [1, 1, 2, 2, 2, 4]
    rt = types.SimpleNamespace(cudaError_t=types.SimpleNamespace(cudaSuccess=0), cudaGraphNodeType=NT,
                               cudaGraphGetNodes=lambda g, n=None: (0, list(range(len(types_))), len(types_)),
                               cudaGraphNodeGetType=lambda node: (0, types_[node]))
    bindings = types.ModuleType("cuda.bindings")
    bindings.runtime = rt
    monkeypatch.setitem(sys.modules, "cuda", types.ModuleType("cuda"))
    monkeypatch.setitem(sys.modules, "cuda.bindings", bindings)
    monkeypatch.setitem(sys.modules, "cuda.bindings.runtime", rt)
    g = types.SimpleNamespace(raw_cuda_graph=lambda: 1)
    assert e3.graph_node_counts(g) == {"kernel": 2, "memcpy": 3, "memset": 0, "other": 1}
    assert e3.graph_node_counts(object()) == {}


def test_locate_list_cache_and_none():
    assert [s.key for s in e3.locate_states(_ListCache())] == [0, 2]

    class Empty:
        pass
    assert e3.locate_states(Empty()) == []


def test_repoint_copies_values_into_arena():
    c = _LayerCache()
    before = [ly.recurrent_states.clone() for ly in c.layers if ly.recurrent_states is not None]
    slots = e3.locate_states(c)
    arena = torch.zeros(4096, dtype=torch.uint8)
    views = e3.repoint(slots, arena)
    for s, v, b in zip(slots, views, before):
        assert s.get() is v and torch.equal(v, b) and e3.in_arena(v, arena)
    assert views[1].data_ptr() - views[0].data_ptr() == 256   # 24 floats = 96 B, aligned to 256


def test_settle_inplace_update():
    c = _LayerCache()
    slots = e3.locate_states(c)
    arena = torch.zeros(4096, dtype=torch.uint8)
    views = e3.repoint(slots, arena)

    def step():
        for s in slots:
            s.get().mul_(2.0)
    assert e3.settle_states(slots, views, arena, step) == "views_inplace"


def test_settle_reassigning_update_gets_pinned():
    c = _LayerCache()
    slots = e3.locate_states(c)
    arena = torch.zeros(4096, dtype=torch.uint8)
    views = e3.repoint(slots, arena)
    start = views[0].clone()

    def step():
        for ly in c.layers:
            if ly.recurrent_states is not None:
                ly.recurrent_states = ly.recurrent_states * 2.0   # new tensor each step

    assert e3.settle_states(slots, views, arena, step) == "views_pinned_copy"
    assert all(e3.in_arena(ly.recurrent_states, arena) for ly in c.layers if ly.recurrent_states is not None)
    step()
    assert torch.allclose(c.layers[0].recurrent_states, start * 4.0)   # the unpinned first step is discarded
    assert c.layers[0].recurrent_states.data_ptr() == views[0].data_ptr()


def test_settle_list_container_pinned():
    c = _ListCache()
    slots = e3.locate_states(c)
    arena = torch.zeros(4096, dtype=torch.uint8)
    views = e3.repoint(slots, arena)

    def step():
        for i, t in enumerate(c.recurrent_states):
            if t is not None:
                c.recurrent_states[i] = t + 1.0

    assert e3.settle_states(slots, views, arena, step) == "views_pinned_copy"
    step()
    assert all(e3.in_arena(t, arena) for t in c.recurrent_states if t is not None)


def test_settle_unavailable_when_pin_cannot_hold():
    class Weird:
        pass
    c = Weird()
    c.recurrent_states = (torch.randn(4),)        # tuple container: not supported by pin
    slots = e3.locate_states(c)
    assert slots == []                              # tuples are ignored, no crash


# ---- int8 head (CPU torch path)

def test_quantize_rows_roundtrip():
    torch.manual_seed(0)
    w = torch.randn(300, 64) * torch.rand(300, 1)
    q, s = int8_head.quantize_rows_int8(w, chunk=128)
    assert q.dtype == torch.int8 and s.dtype == torch.float32 and q.abs().max() <= 127
    err = (q.float() * s[:, None] - w).abs().max()
    assert err <= s.max() * 0.51


def test_int8_gemv_matches_dense():
    torch.manual_seed(1)
    w, x = torch.randn(200, 96), torch.randn(96)
    q, s = int8_head.quantize_rows_int8(w)
    y = int8_head.int8_gemv(q, s, x, out_dtype=torch.float32)
    ref = w @ x
    assert torch.allclose(y, ref, atol=0.05 * ref.abs().max())
    assert torch.allclose(y, int8_head.int8_gemv_ref(q, s, x))


def test_head_modules_shapes():
    torch.manual_seed(2)
    w = torch.randn(50, 32)
    h = torch.randn(1, 3, 32)
    q, s = int8_head.quantize_rows_int8(w)
    out = int8_head.Int8Head(q, s, out_dtype=torch.float32)(h)
    assert out.shape == (1, 1, 50)
    out2 = int8_head.Bf16Head(w)(h)
    assert out2.shape == (1, 1, 50)
    assert torch.allclose(out2[0, 0], w @ h[0, -1], atol=1e-4)


def test_e3_args():
    a = e3.parse_args(["--quick", "--dry-run"])
    assert a.heads == ["bf16_full", "int8_48k"] and a.ncu_heads == ["int8_48k"]
    with pytest.raises(SystemExit):
        e3.parse_args(["--heads", "nope"])


# ---- setup_config / Decoder with a tiny fake model on CPU

class _FakeModel(torch.nn.Module):
    def __init__(self, vocab=100000, hidden=16):
        super().__init__()
        self.emb = torch.nn.Embedding(vocab, hidden)
        self.lm_head = torch.nn.Linear(hidden, vocab, bias=False)
        self.to(torch.bfloat16)

    def forward(self, input_ids=None, past_key_values=None, cache_position=None, use_cache=True, logits_to_keep=1):
        from types import SimpleNamespace
        h = self.emb(input_ids)
        for ly in past_key_values.layers:
            if getattr(ly, "recurrent_states", None) is not None:
                ly.recurrent_states = ly.recurrent_states * 0.9 + 0.1   # reassigns: forces pinning
        return SimpleNamespace(logits=self.lm_head(h[:, -logits_to_keep:]))


@pytest.fixture
def cpu_env(monkeypatch):
    from exps import e2_hybrid_bf16 as e2
    monkeypatch.setattr(e2, "DEVICE", "cpu")
    monkeypatch.setattr(e2, "make_cache", lambda model, n, info: _LayerCache())
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *a, **k: None)
    return e2


@pytest.mark.parametrize("kind", ["bf16_full", "int8_full", "int8_48k", "bf16_48k"])
def test_setup_config_places_states_and_head_in_arena(cpu_env, kind):
    torch.manual_seed(0)
    model = _FakeModel()
    orig_w = model.lm_head.weight.data.clone()
    kk = cpu_env.keep_kw(model)
    assert kk == "logits_to_keep"
    cache, arena, lay, slots, method, orig = e3.setup_config(model, kk, 5, kind, 8, {})
    assert method == "views_pinned_copy" and len(slots) == 2
    assert all(e3.in_arena(s.get(), arena) for s in slots)
    head = model.lm_head
    assert orig.weight.data.equal(orig_w) and head is not orig
    w = head.weight if hasattr(head, "weight") else head.q
    assert e3.in_arena(w, arena) and w.data_ptr() >= arena.data_ptr() + lay["head_off"]
    out = head(torch.randn(1, 1, 16).to(torch.bfloat16))
    assert out.shape[:2] == (1, 1) and out.shape[2] == (49152 if kind.endswith('48k') else 100000)
    dec = cpu_env.Decoder(model, cache, 5, kk, "eager")
    dec.once()
    assert int(dec.pos.item()) == 6 and 0 <= int(dec.tok.item()) < 100000
    assert all(e3.in_arena(s.get(), arena) for s in slots)   # still pinned after more steps


def test_setup_config_restores_head_when_a_step_fails(cpu_env, monkeypatch):
    model = _FakeModel()
    orig = model.lm_head

    def boom(slots, views, arena, run_step):
        raise RuntimeError("step failed")
    monkeypatch.setattr(e3, "settle_states", boom)
    with pytest.raises(RuntimeError, match="step failed"):
        e3.setup_config(model, "logits_to_keep", 5, "int8_48k", 8, {})
    assert model.lm_head is orig and hasattr(model.lm_head, "weight")   # the next config still finds a plain head


def test_setup_config_without_state_slots(cpu_env, monkeypatch):
    class NoState:
        layers = [type("L", (), {})()]
    monkeypatch.setattr(cpu_env, "make_cache", lambda model, n, info: NoState())
    model = _FakeModel()
    monkeypatch.setattr(model, "forward", lambda **kw: __import__("types").SimpleNamespace(logits=torch.zeros(1, 1, 100000)))
    _, arena, lay, slots, method, _ = e3.setup_config(model, "logits_to_keep", 0, "bf16_48k", 8, {})
    assert slots == [] and method == "unavailable" and lay["gdn_bytes"] == 0


def test_decoder_prefill_and_modes_on_cpu(cpu_env):
    model = _FakeModel()
    cache = _LayerCache()
    cache = cpu_env.prefill(model, cache, 5000, "logits_to_keep")
    assert cache is not None
    d, mode = cpu_env.pick_decoder(model, cache, 5, "logits_to_keep", "eager", {})
    assert mode == "eager"
    start = int(d.pos.item())
    d.once()
    assert int(d.pos.item()) == start + 1
    d.reset()
    assert int(d.pos.item()) == 5 and int(d.tok.item()) == 1234


def test_pick_decoder_falls_back_and_reports(cpu_env, monkeypatch):
    model = _FakeModel()
    cache = _LayerCache()
    orig = cpu_env.Decoder

    class Flaky(orig):
        def __init__(self, model, cache, ctx, kk, mode):
            if mode in ("reduce-overhead", "default", "cuda_graph_manual"):
                raise RuntimeError("compile boom " + mode)
            super().__init__(model, cache, ctx, kk, mode)

    monkeypatch.setattr(cpu_env, "Decoder", Flaky)
    info = {}
    d, mode = cpu_env.pick_decoder(model, cache, 3, "logits_to_keep", "reduce-overhead", info)
    assert mode == "eager" and len(info["mode_errors"]) == 3 and "compile boom" in info["mode_errors"][0]


def test_pick_decoder_dynamic_cache_is_eager_only(cpu_env, monkeypatch):
    model = _FakeModel()
    modes = []
    orig = cpu_env.Decoder

    class Spy(orig):
        def __init__(self, model, cache, ctx, kk, mode):
            modes.append(mode)
            super().__init__(model, cache, ctx, kk, mode)

    monkeypatch.setattr(cpu_env, "Decoder", Spy)
    d, mode = cpu_env.pick_decoder(model, _LayerCache(), 3, "logits_to_keep", "reduce-overhead", {}, static=False)
    assert mode == "eager" and set(modes) == {"eager"}


def test_pick_decoder_rejects_mode_that_recompiles(cpu_env, monkeypatch):
    model = _FakeModel()
    n = [0]

    def graphs():
        n[0] += 1   # every read sees a new graph: recompiling on every step
        return n[0]

    monkeypatch.setattr(cpu_env, "dynamo_graphs", graphs)
    orig = cpu_env.Decoder

    class Fake(orig):
        def __init__(self, model, cache, ctx, kk, mode):
            if mode == "cuda_graph_manual":
                raise RuntimeError("no graph")
            super().__init__(model, cache, ctx, kk, "eager")   # compiled modes run eager here
            self.mode = mode

    monkeypatch.setattr(cpu_env, "Decoder", Fake)
    info = {}
    d, mode = cpu_env.pick_decoder(model, _LayerCache(), 3, "logits_to_keep", "reduce-overhead", info)
    assert mode == "eager" and sum("recompiles" in e for e in info["mode_errors"]) == 2   # reduce-overhead, default
    d, mode = cpu_env.pick_decoder(model, _LayerCache(), 3, "logits_to_keep", "reduce-overhead", {},
                                   exclude=("reduce-overhead", "default"))
    assert mode == "eager"


def test_fill_cache_random_walks_nested_objects():
    from exps import e2_hybrid_bf16 as e2
    c = _LayerCache()
    c.extra = {"k": [torch.zeros(3), torch.zeros(3, dtype=torch.int64)]}
    n = e2.fill_cache_random(c)
    assert n == 3 and c.extra["k"][0].abs().sum() > 0 and c.extra["k"][1].abs().sum() == 0


# ---- transformers 5.x static layers advance a device-side length counter on every forward

class _CountLayer:
    def __init__(self):
        self.cumulative_length = torch.tensor(0, dtype=torch.long)


class _CountCache:
    def __init__(self, n=3):
        self.layers = [_CountLayer() for _ in range(n)] + [type("Lin", (), {"cumulative_length": 7})()]


def test_fix_cache_len_sets_tensor_counters_only():
    from exps import e2_hybrid_bf16 as e2
    c = _CountCache()
    assert e2.fix_cache_len(c, 9) == 3
    assert [int(ly.cumulative_length) for ly in c.layers[:3]] == [9, 9, 9] and c.layers[3].cumulative_length == 7
    assert e2.fix_cache_len(object(), 9) == 0 and e2.fix_cache_len(None, 9) == 0


def test_step_fn_keeps_counter_fixed_over_many_calls(cpu_env):
    class Adv(_FakeModel):
        def forward(self, **kw):
            for ly in kw["past_key_values"].layers:
                if isinstance(getattr(ly, "cumulative_length", None), torch.Tensor):
                    ly.cumulative_length.add_(1)   # what StaticLayer.update does
            return super().forward(**kw)

    model, cache = Adv(), _CountCache()
    fn = e3.step_fn(model, cache, 5, "logits_to_keep")
    for _ in range(300):
        fn()
    assert [int(ly.cumulative_length) for ly in cache.layers[:3]] == [6, 6, 6]   # reset to ctx, then +1
    dec = cpu_env.Decoder(model, cache, 5, "logits_to_keep", "eager")
    dec.reset()
    assert int(cache.layers[0].cumulative_length) == 5


def test_real_static_cache_stays_in_bounds_for_2x_cache_len(monkeypatch):
    pytest.importorskip("transformers")
    try:
        from transformers import Qwen3_5ForCausalLM, Qwen3_5TextConfig
    except ImportError:
        pytest.skip("transformers without Qwen3_5")
    from exps import e2_hybrid_bf16 as e2
    monkeypatch.setattr(e2, "DEVICE", "cpu")
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *a, **k: None)
    cfg = Qwen3_5TextConfig(vocab_size=2048, hidden_size=64, intermediate_size=128, num_hidden_layers=4, num_attention_heads=2,
                            num_key_value_heads=1, head_dim=32, linear_conv_kernel_dim=4, linear_key_head_dim=16,
                            linear_value_head_dim=16, linear_num_key_heads=2, linear_num_value_heads=2,
                            layer_types=["linear_attention"] * 3 + ["full_attention"])
    model = Qwen3_5ForCausalLM(cfg).to(torch.bfloat16).eval()
    kk, ctx, decode = e2.keep_kw(model), 4, 8
    info: dict = {}
    with torch.no_grad():
        cache, arena, lay, slots, method, _ = e3.setup_config(model, kk, ctx, "bf16_full", decode, info)
        counters = [ly.cumulative_length for ly in cache.layers if isinstance(getattr(ly, "cumulative_length", None), torch.Tensor)]
        if not counters:
            pytest.skip("this transformers has no device-side cache length counter")
        assert all(int(c) == ctx for c in counters)
        fn = e3.step_fn(model, cache, ctx, kk)
        for _ in range(2 * e2.cache_len_for(ctx, decode)):   # more steps than the cache holds
            fn()
    assert all(int(c) == ctx + 1 for c in counters)
    assert len(slots) == 3 and method in ("views_inplace", "views_pinned_copy")


def test_analyze_time_proxy():
    rows = [{"ctx": 0, "head": "int8_48k", "condition": "off", "us_per_token": 1000.0, "persisted_bytes": 0},
            {"ctx": 0, "head": "int8_48k", "condition": "both", "us_per_token": 990.0, "persisted_bytes": 10_000_000}]
    out = e3.analyze(rows, peak_gbps=1000.0)
    assert out["go_no_go"] is None
    px = out["go_no_go_time_proxy"]
    assert abs(px["value"] - 1.0) < 1e-9 and px["go"] is True
    assert e3.analyze(rows)["go_no_go_time_proxy"] is None
