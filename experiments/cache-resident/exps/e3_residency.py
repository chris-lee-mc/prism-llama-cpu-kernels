"""E3: persist the GDN recurrent state and/or the LM head in L2 (Qwen3.5-0.8B decode).

One contiguous arena = [GDN states | head weights | head scales]. HF cache state tensors are re-pointed to
views of the arena. A decode step (static cache, eager model) is captured in a CUDA graph, the access
policy window is applied to every kernel node, and the graph is replayed. Conditions:
off, gdn (window over states), head (window over head), both (one window over the whole arena).

  python -m exps.e3_residency [--ctx 0,16384] [--heads bf16_full,int8_full,int8_48k,bf16_48k]
                              [--conditions off,gdn,head,both] [--ncu-heads int8_48k,bf16_full] [--ncu-ctx 0]

Pruned heads keep vocab ids 0..49151 (measurement only; output ids are not valid tokens).
PLAN's "48K-pruned INT8 98 MB" matches bf16_48k (or int8 at hidden 2048); both are run.
Cache contents are random (the cache is filled, not prefilled): bytes and time do not depend on values.
Go/no-go metric: DRAM bytes saved per persisted byte for condition both with int8_48k.
State write-back runs as an elementwise add kernel (ArenaView), not as a cudaMemcpy graph node: a memcpy node
gets no access policy window and ncu does not profile it. Rows record graph_nodes (kernel / memcpy counts).
Exit 4 before any timing if gdn/both are requested and the state cannot be placed in the arena (--allow-no-gdn skips them).
"""
from __future__ import annotations

import argparse
import sys

import torch

from exps import common, e2_hybrid_bf16 as e2, hf_ids, modelspec

EXP = "e3"
ALIGN = 256
HEADS = {"bf16_full": ("bf16", None), "int8_full": ("int8", None), "int8_48k": ("int8", 49152), "bf16_48k": ("bf16", 49152)}
CONDITIONS = ["off", "gdn", "head", "both"]
GO_RATIO = 0.5  # heuristic: each persisted byte should save >= 0.5 DRAM byte per token
INNER_REPLAYS = 3


# ---------------------------------------------------------------- pure helpers

def round_up(x: int, a: int = ALIGN) -> int:
    return (x + a - 1) // a * a


def head_bytes(kind: str, vocab: int, hidden: int) -> dict:
    dt, rows = HEADS[kind]
    n = rows or vocab
    w = n * hidden * (2 if dt == "bf16" else 1)
    s = n * 4 if dt == "int8" else 0
    return {"rows": n, "weight": w, "scale": s, "total": w + s}


def arena_layout(gdn_bytes: int, head_w: int, head_s: int) -> dict:
    """Offsets of [gdn | head weights | head scales], each 256 B aligned."""
    go = 0
    ho = round_up(gdn_bytes)
    so = round_up(ho + head_w)
    return {"gdn_off": go, "gdn_bytes": gdn_bytes, "head_off": ho, "head_bytes": head_w + head_s,
            "scale_off": so, "total": round_up(so + head_s)}


def window_region(cond: str, lay: dict) -> tuple[int, int] | None:
    """(offset, nbytes) of the window for a condition; None for off."""
    if cond == "off":
        return None
    if cond == "gdn":
        return lay["gdn_off"], lay["gdn_bytes"]
    if cond == "head":
        return lay["head_off"], lay["total"] - lay["head_off"]
    if cond == "both":
        return 0, lay["total"]
    raise ValueError(cond)


def analyze(rows: list[dict]) -> dict:
    """Per (ctx, head): gain vs off, DRAM bytes saved per persisted byte. Go/no-go on both + int8_48k."""
    out: dict = {"per_config": [], "go_no_go": None}
    keys = sorted({(r["ctx"], r["head"]) for r in rows})
    for ctx, head in keys:
        grp = {r["condition"]: r for r in rows if r["ctx"] == ctx and r["head"] == head}
        off = grp.get("off")
        if not off:
            continue
        for cond, r in grp.items():
            if cond == "off":
                continue
            e: dict = {"ctx": ctx, "head": head, "condition": cond, "persisted_bytes": r.get("persisted_bytes")}
            if off.get("us_per_token") and r.get("us_per_token"):
                e["time_gain_pct"] = 100.0 * (off["us_per_token"] - r["us_per_token"]) / off["us_per_token"]
            if off.get("dram_bytes_per_token") is not None and r.get("dram_bytes_per_token") is not None:
                saved = off["dram_bytes_per_token"] - r["dram_bytes_per_token"]
                e["dram_saved_bytes_per_token"] = saved
                pb = r.get("persisted_bytes")
                e["saved_per_persisted_byte"] = saved / pb if pb else None
            out["per_config"].append(e)
    both = [e for e in out["per_config"] if e["condition"] == "both" and e["head"] == "int8_48k"
            and e.get("saved_per_persisted_byte") is not None]
    if both:
        worst = min(e["saved_per_persisted_byte"] for e in both)
        out["go_no_go"] = {"metric": "dram bytes saved / persisted byte (both, int8_48k)", "value": worst,
                           "threshold": GO_RATIO, "go": worst >= GO_RATIO}
    return out


# ---------------------------------------------------------------- cache state re-pointing

class Slot:
    """One GDN state tensor inside a cache: ('attr', obj, name) or ('item', owner, container_name, key)."""

    def __init__(self, kind, owner, name, key=None):
        self.kind, self.owner, self.name, self.key = kind, owner, name, key

    def get(self):
        if self.kind == "attr":
            return getattr(self.owner, self.name)
        return getattr(self.owner, self.name)[self.key]

    def set(self, t):
        if self.kind == "attr":
            setattr(self.owner, self.name, t)
        else:
            getattr(self.owner, self.name)[self.key] = t


STATE_ATTRS = ("recurrent_states", "recurrent_state", "ssm_states", "ssm_state")


def _tensor_keys(cont) -> list:
    if isinstance(cont, list):
        return [i for i, t in enumerate(cont) if isinstance(t, torch.Tensor)]
    if isinstance(cont, dict):
        return [k for k, t in cont.items() if isinstance(t, torch.Tensor)]
    return []


def locate_states(cache) -> list[Slot]:
    """Find the recurrent-state tensors: per-layer tensor, list or dict attributes (cache.layers[i].X), or lists/dicts on the cache."""
    slots: list[Slot] = []
    layers = getattr(cache, "layers", None)
    if isinstance(layers, (list, tuple)):
        for ly in layers:
            for name in STATE_ATTRS:
                v = getattr(ly, name, None)
                if isinstance(v, torch.Tensor):
                    slots.append(Slot("attr", ly, name))
                    break
                keys = _tensor_keys(v)
                if keys:
                    slots += [Slot("item", ly, name, k) for k in keys]
                    break
    if slots:
        return slots
    for name in STATE_ATTRS:
        keys = _tensor_keys(getattr(cache, name, None))
        if keys:
            return [Slot("item", cache, name, k) for k in keys]
    return slots


class ArenaView(torch.Tensor):
    """Arena view whose same-dtype copy_ runs as an add kernel. A plain copy_ is a cudaMemcpy graph node."""

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        with torch._C.DisableTorchFunctionSubclass():
            if func is torch.Tensor.copy_ and len(args) >= 2 and isinstance(args[1], torch.Tensor):
                dst, src = args[0], args[1]
                if src.dtype == dst.dtype and src.shape == dst.shape and src.device == dst.device:
                    torch.add(src, 0, out=dst)
                    return dst
            return func(*args, **kwargs)


def repoint(slots: list[Slot], arena: torch.Tensor, base: int = 0) -> list[torch.Tensor]:
    """Copy each state into a view of arena[base:...] and make the cache use the view. Returns the views (ArenaView)."""
    views, off = [], base
    for s in slots:
        t = s.get()
        nb = t.numel() * t.element_size()
        v = arena[off:off + nb].view(t.dtype).view(t.shape).as_subclass(ArenaView)
        v.copy_(t)
        s.set(v)
        views.append(v)
        off += round_up(nb)
    return views


class _PinnedList(list):
    pins: dict = {}

    def __setitem__(self, i, v):
        if i in self.pins and v is not self.pins[i]:
            self.pins[i].copy_(v)
        else:
            super().__setitem__(i, v)


class _PinnedDict(dict):
    pins: dict = {}

    def __setitem__(self, k, v):
        if k in self.pins and v is not self.pins[k]:
            self.pins[k].copy_(v)
        else:
            super().__setitem__(k, v)


def pin(slot: Slot, view: torch.Tensor) -> None:
    """Make later assignments to the slot copy into view (keeps the state inside the arena; one extra copy)."""
    if slot.kind == "attr":
        obj, name = slot.owner, slot.name

        def setter(self, v, _view=view):
            if v is not _view:
                _view.copy_(v)

        sub = type(type(obj).__name__ + "Pinned", (type(obj),), {name: property(lambda self, _v=view: _v, setter)})
        obj.__class__ = sub
        return
    cont = getattr(slot.owner, slot.name)
    if not isinstance(cont, (_PinnedList, _PinnedDict)):
        new = _PinnedList(cont) if isinstance(cont, list) else _PinnedDict(cont)
        new.pins = {}
        setattr(slot.owner, slot.name, new)
        cont = new
    cont.pins[slot.key] = view


def in_arena(t: torch.Tensor, arena: torch.Tensor) -> bool:
    lo = arena.data_ptr()
    return lo <= t.data_ptr() and t.data_ptr() + t.numel() * t.element_size() <= lo + arena.numel()


def settle_states(slots, views, arena, run_step) -> str:
    """Run a step; if any state left the arena, pin and run again. Returns the method name."""
    run_step()
    if all(in_arena(s.get(), arena) for s in slots):
        return "views_inplace"
    for s, v in zip(slots, views):
        s.set(v)
        pin(s, v)
    run_step()
    if all(in_arena(s.get(), arena) for s in slots):
        return "views_pinned_copy"
    return "unavailable"


# ---------------------------------------------------------------- GPU side

def build_head(kind: str, full_w: torch.Tensor, arena: torch.Tensor, lay: dict):
    """Place head weights (and scales) in the arena and return the head module."""
    from exps import int8_head
    dt, rows = HEADS[kind]
    w = full_w[:rows] if rows else full_w
    n, h = w.shape
    if dt == "bf16":
        v = arena[lay["head_off"]:lay["head_off"] + n * h * 2].view(torch.bfloat16).view(n, h)
        v.copy_(w)
        return int8_head.Bf16Head(v)
    q, sc = int8_head.quantize_rows_int8(w)
    qv = arena[lay["head_off"]:lay["head_off"] + n * h].view(torch.int8).view(n, h)
    sv = arena[lay["scale_off"]:lay["scale_off"] + n * 4].view(torch.float32)
    qv.copy_(q)
    sv.copy_(sc)
    return int8_head.Int8Head(qv, sv)


def setup_config(model, kk, ctx: int, kind: str, decode_len: int, info: dict):
    """Fresh cache + arena + head for (ctx, head). Returns (cache, arena, lay, slots, method, original_head)."""
    cache = e2.make_cache(model, e2.cache_len_for(ctx, decode_len), info)
    if cache is None:
        raise RuntimeError("static cache unavailable: " + "; ".join(info.get("cache_errors", [])[-2:]))
    e2.Decoder(model, cache, ctx, kk, "eager").once()  # builds lazy cache layers
    e2.fill_cache_random(cache)
    e2.fix_cache_len(cache, ctx)
    slots = locate_states(cache)
    orig = model.lm_head
    src = getattr(orig, "weight", None)
    if src is None:
        raise RuntimeError("lm_head has no weight")
    vocab, hidden = src.shape
    hb = head_bytes(kind, vocab, hidden)
    gdn_bytes = sum(round_up(s.get().numel() * s.get().element_size()) for s in slots)
    lay = arena_layout(gdn_bytes, hb["weight"], hb["scale"])
    arena = torch.zeros(lay["total"], dtype=torch.uint8, device=e2.DEVICE)
    views = repoint(slots, arena) if slots else []
    try:
        model.lm_head = build_head(kind, src.data, arena, lay)
        dec = e2.Decoder(model, cache, ctx, kk, "eager")

        def step():
            e2.fix_cache_len(cache, ctx)
            dec.once()

        method = settle_states(slots, views, arena, step) if slots else "unavailable"
        e2.fix_cache_len(cache, ctx)
    except BaseException:
        model.lm_head = orig
        raise
    return cache, arena, lay, slots, method, orig


def graph_node_counts(g) -> dict:
    """Node counts by kind of a keep_graph template: {kernel, memcpy, memset, other}. Empty dict if the driver API is unavailable."""
    try:
        try:
            from cuda.bindings import runtime as rt
        except ImportError:
            from cuda import cudart as rt
        graph = int(g.raw_cuda_graph())
        res = rt.cudaGraphGetNodes(graph)
        n = int(res[2] if len(res) > 2 else res[1])
        out = {"kernel": 0, "memcpy": 0, "memset": 0, "other": 0}
        if n:
            res = rt.cudaGraphGetNodes(graph, n)
            if res[0] != rt.cudaError_t.cudaSuccess:
                return {}
            kinds = {rt.cudaGraphNodeType.cudaGraphNodeTypeKernel: "kernel", rt.cudaGraphNodeType.cudaGraphNodeTypeMemcpy: "memcpy",
                     rt.cudaGraphNodeType.cudaGraphNodeTypeMemset: "memset"}
            for node in list(res[1])[:n]:
                err, t = rt.cudaGraphNodeGetType(node)
                out[kinds.get(t, "other") if err == rt.cudaError_t.cudaSuccess else "other"] += 1
        return out
    except Exception:
        return {}


def step_fn(model, cache, ctx, kk):
    """Decode step for capture/timing: resets the cache length first so every call (replay) writes the same KV slot.

    transformers 5.x StaticLayer ignores cache_position and advances its own counter, so without the reset a graph
    replayed more than cache_len - ctx times writes out of bounds.
    """
    dec = e2.Decoder(model, cache, ctx, kk, "eager")

    def fn():
        e2.fix_cache_len(cache, ctx)
        dec.fn()

    return fn


def capture_step(model, cache, ctx, kk, region, arena, method_pref, notes):
    """Capture one eager decode step. region None = no window. Returns (graph, info)."""
    from harness import l2persist, timing
    fn = step_fn(model, cache, ctx, kk)
    l2persist.reset()
    info = {"window_bytes": 0, "hit_ratio": None, "persisted_bytes": 0, "window_method_used": None, "graph_nodes": {},
            "window_readback": None}
    common.clear_capture_windows()

    if region is None:
        l2persist.set_limit(0)
        try:
            g = timing.capture_graph(fn, keep_graph=True)
            info["graph_nodes"] = graph_node_counts(g)
            info["window_readback"] = common.window_readback(g, graph=True)
            g.instantiate()
        except Exception:
            g = timing.capture_graph(fn)
        return g, info
    off, nb = region
    limit = l2persist.set_limit(None)
    nb = min(nb, l2persist.max_window_bytes()) or nb
    hr = l2persist.suggest_hit_ratio(nb, limit)
    info.update(window_bytes=nb, hit_ratio=hr, persisted_bytes=int(min(nb * hr, limit)))
    ptr = arena.data_ptr() + off
    if method_pref == "graph":
        try:
            g = timing.capture_graph(fn, keep_graph=True)
            info["graph_nodes"] = graph_node_counts(g)
            n = l2persist.apply_window_to_graph(g, ptr, nb, hr)
            if n <= 0:
                raise RuntimeError("no kernel nodes in graph")
            info["window_readback"] = common.window_readback(g, graph=True)
            g.instantiate()
            info.update(window_method_used="graph", kernel_nodes=n)
            return g, info
        except Exception as e:
            notes.append(f"graph window failed ({type(e).__name__}: {e}); using stream window")
    st = torch.cuda.Stream()
    try:
        l2persist.set_stream_window(ptr, nb, hr, stream=st)
        info["window_readback"] = common.window_readback(st)
        g = timing.capture_graph(fn, stream=st)
    finally:
        common.clear_capture_windows(st)
    info["window_method_used"] = "stream"
    return g, info


def eager_time(model, cache, ctx, kk, region, arena, iters, repeats):
    """Fallback when graph capture fails: eager steps with a stream window. Launch bound; use with care."""
    from harness import l2persist, timing
    fn = step_fn(model, cache, ctx, kk)
    l2persist.reset()
    info = {"window_bytes": 0, "hit_ratio": None, "persisted_bytes": 0, "window_method_used": "stream_eager"}
    if region is None:
        l2persist.set_limit(0)
    else:
        limit = l2persist.set_limit(None)
        nb = min(region[1], l2persist.max_window_bytes()) or region[1]
        hr = l2persist.suggest_hit_ratio(nb, limit)
        info.update(window_bytes=nb, hit_ratio=hr, persisted_bytes=int(min(nb * hr, limit)))
        l2persist.set_stream_window(arena.data_ptr() + region[0], nb, hr)
    try:
        per = [timing.cuda_time_us(fn, iters=max(20, iters // 4), warmup=10) for _ in range(repeats)]
    finally:
        if region is not None:
            l2persist.clear_stream_window()
    return per, info


def run_inner(args) -> int:
    """ncu target for one (ctx, head, condition)."""
    from harness import l2persist
    info: dict = {}
    model, kk = e2.setup(args, info)
    ctx, kind, cond = args.ctx[0], args.heads[0], args.conditions[0]
    cache, arena, lay, slots, method, _ = setup_config(model, kk, ctx, kind, args.decode, info)
    if cond in ("gdn", "both") and method == "unavailable":
        print("gdn state not in arena; condition invalid", file=sys.stderr)
        return 3
    g, _ = capture_step(model, cache, ctx, kk, window_region(cond, lay), arena, args.window_method, [])
    g.replay()
    torch.cuda.synchronize()
    torch.cuda.profiler.start()
    for _ in range(INNER_REPLAYS):
        g.replay()
    torch.cuda.synchronize()
    torch.cuda.profiler.stop()
    l2persist.reset()
    return 0


def inner_argv(args, ctx, kind, cond) -> list[str]:
    return common.py_module_argv("exps.e3_residency", "--inner", "--ctx", str(ctx), "--heads", kind, "--conditions", cond,
                                 "--decode", str(args.decode), "--window-method", args.window_method)


def run(args) -> int:
    from harness import l2persist, timing
    if not torch.cuda.is_available():
        print("error: no CUDA device (use --dry-run for the plan)", file=sys.stderr)
        return 2
    args.gpu = common.gpu_tag(args)
    notes: list[str] = []
    info: dict = {}
    ncu_ok, ncu_note = common.counters_available(args)
    if ncu_note:
        notes.append(ncu_note)
    budget = common.Budget(args.budget_min or (8 if args.quick else 27))
    dev = l2persist.log_device(0)
    try:
        model, kk = e2.setup(args, info)
    except Exception as e:
        notes.append(f"model setup failed: {type(e).__name__}: {e}")
        common.write_result(EXP, args, {"load": info, "device": dev}, [], notes, False)
        print(f"error: {notes[-1]}", file=sys.stderr)
        return 3
    rows: list[dict] = []
    methods: dict = {}
    counters_all = ncu_ok
    ncu_jobs: list[tuple] = []
    orig_head = model.lm_head  # setup_config swaps the head; always put this one back
    config = {"ctx": args.ctx, "heads": args.heads, "conditions": args.conditions, "window_method": args.window_method,
              "iters": args.iters, "repeats": args.repeats, "ncu_ctx": args.ncu_ctx, "ncu_heads": args.ncu_heads,
              "gdn_persist_method": methods, "cache_fill": "random", "head_bytes": {
                  k: head_bytes(k, 248320, 1024) for k in HEADS}, "quick": args.quick, "device": dev, "load": info}

    def save():
        common.write_partial(EXP, args, config, rows, notes, False)

    need_gdn = any(c in ("gdn", "both") for c in args.conditions)
    out_of_time = False
    for ctx in args.ctx:
        for kind in args.heads:
            if budget.expired(60):
                out_of_time = True
                notes.append(f"time budget used up before ctx={ctx} head={kind}; later configs skipped")
                break
            try:
                cache, arena, lay, slots, method, _ = setup_config(model, kk, ctx, kind, args.decode, info)
            except Exception as e:
                model.lm_head = orig_head
                notes.append(f"ctx={ctx} head={kind} setup failed: {type(e).__name__}: {e}")
                continue
            methods[f"{ctx}/{kind}"] = method
            if method == "unavailable" and need_gdn and not args.allow_no_gdn:
                model.lm_head = orig_head
                notes.append(f"ctx={ctx} head={kind}: GDN state cannot be placed in the arena (no state slots found or "
                             "write-back escapes the arena); the go/no-go metric needs it. Aborting before timing; "
                             "--allow-no-gdn runs the off and head rows only")
                common.write_result(EXP, args, config, rows, notes, False)
                print(f"error: {notes[-1]}", file=sys.stderr)
                return 4
            for cond in args.conditions:
                if cond in ("gdn", "both") and method == "unavailable":
                    notes.append(f"ctx={ctx} head={kind} cond={cond} skipped: GDN state cannot be placed in the arena")
                    continue
                region = window_region(cond, lay)
                row = {"ctx": ctx, "head": kind, "condition": cond, "window_bytes": 0, "hit_ratio": None,
                       "persisted_bytes": 0, "us_per_token": None, "dram_bytes_per_token": None, "l2_hit_rate": None,
                       "gdn_persist_method": method, "arena_bytes": lay["total"], "counters_ok": False,
                       "gdn_impl": info.get("gdn_impl"), "graph_nodes": {}, "window_readback": None}
                try:
                    g, ginfo = capture_step(model, cache, ctx, kk, region, arena, args.window_method, notes)
                    row.update(ginfo)
                    per = [timing.replay_time_us(g, iters=args.iters, warmup=10) for _ in range(args.repeats)]
                    row["us_per_token"] = sorted(per)[len(per) // 2]
                    row["us_per_token_all"] = per
                    del g
                except Exception as e:
                    notes.append(f"ctx={ctx} head={kind} cond={cond} graph path failed ({type(e).__name__}: {str(e)[:200]}); "
                                 "using eager timing, launch bound")
                    try:
                        per, ginfo = eager_time(model, cache, ctx, kk, region, arena, args.iters, args.repeats)
                        row.update(ginfo)
                        row["us_per_token"] = sorted(per)[len(per) // 2]
                        row["us_per_token_all"] = per
                    except Exception as e2_:
                        row["error"] = f"{type(e2_).__name__}: {e2_}"
                        notes.append(f"ctx={ctx} head={kind} cond={cond} failed: {row['error']}")
                l2persist.reset()
                l2persist.set_limit(0)
                rows.append(row)
                save()
                if ncu_ok and ctx in args.ncu_ctx and kind in args.ncu_heads and row["us_per_token"] is not None:
                    ncu_jobs.append((row, ctx, kind, cond))
                print("[e3] ctx=%d head=%s cond=%s us/token=%s" % (ctx, kind, cond, row["us_per_token"]), flush=True)
            model.lm_head = orig_head
            del cache, arena
            torch.cuda.empty_cache()
        if out_of_time:
            break
    order = {c: i for i, c in enumerate(["off", "both", "gdn", "head"])}
    ncu_jobs.sort(key=lambda j: (args.ncu_heads.index(j[2]), j[1], order.get(j[3], 9)))
    del model
    torch.cuda.empty_cache()
    for row, ctx, kind, cond in ncu_jobs:
        if budget.expired(150):
            row["ncu_error"] = "skipped: time budget"
            counters_all = False
            continue
        res = common.ncu_run(inner_argv(args, ctx, kind, cond), INNER_REPLAYS, profile_from_start_off=True, app_replay=True,
                             kernel_fallback=False, timeout=budget.timeout(1500, reserve_s=30), env=common.inner_env())
        st = common.ncu_stats(res)
        row.update(dram_bytes_per_token=st["dram_bytes_per_token"], l2_hit_rate=st["l2_hit_rate"],
                   counters_ok=st["counters_ok"], ncu_error=st["ncu_error"], ncu_us_per_token=st["ncu_us_per_token"],
                   ncu_replay_mode=st["ncu_replay_mode"], ncu_passes=st["ncu_passes"])
        counters_all = counters_all and st["counters_ok"]
        save()
        print("[e3] ncu ctx=%d head=%s cond=%s dram/tok=%s hit=%s" % (ctx, kind, cond, row["dram_bytes_per_token"],
                                                                      row["l2_hit_rate"]), flush=True)
    if ncu_jobs:
        notes.append("ncu pass limited to ncu_ctx x ncu_heads (see config); other rows have timing only")
    ana = analyze(rows)
    notes.append("pruned heads use vocab ids 0..49151 (measurement only); bf16_48k matches PLAN's 98 MB figure")
    notes.append("GDN state write-back is an add kernel (ArenaView), so it is a kernel node with the window and counted by ncu; "
                 "ncu DRAM bytes still exclude any remaining memcpy graph nodes (see rows[].graph_nodes.memcpy)")
    notes.append("each captured step starts with a cache-length reset (one tiny fill kernel per full-attention layer) so every "
                 "replay writes the same KV slot; the cost is the same in all conditions")
    if info.get("gdn_impl") != "fla":
        notes.append(f"gdn_impl={info.get('gdn_impl')}: GatedDeltaNet runs HF torch ops (many state-sized fp32 temporaries per layer), "
                     "not a fused fla kernel; the state is a small part of GDN traffic and gdn/both effects are not representative")
    if ana["go_no_go"]:
        notes.append("GO/NO-GO %s: %.3f vs threshold %.2f -> %s" % (ana["go_no_go"]["metric"], ana["go_no_go"]["value"],
                                                                     GO_RATIO, "GO" if ana["go_no_go"]["go"] else "NO-GO"))
    common.write_result(EXP, args, config, rows, notes, counters_all, extra={"analysis": ana})
    return 0


# ---------------------------------------------------------------- CLI

def _csv(text):
    return [t.strip() for t in text.split(",") if t.strip()]


def parse_args(argv=None):
    ap = argparse.ArgumentParser(prog="exps.e3_residency", description=__doc__.split("\n")[0])
    common.add_common_args(ap)
    ap.add_argument("--model", default="0.8B", choices=["0.8B"])
    ap.add_argument("--ctx", type=e2.parse_ctx, default=None)
    ap.add_argument("--heads", type=_csv, default=list(HEADS))
    ap.add_argument("--conditions", type=_csv, default=list(CONDITIONS))
    ap.add_argument("--ncu-ctx", type=e2.parse_ctx, default=[0])
    ap.add_argument("--ncu-heads", type=_csv, default=["int8_48k", "bf16_full"])
    ap.add_argument("--window-method", choices=["graph", "stream"], default="graph")
    ap.add_argument("--decode", type=int, default=64, help="cache headroom in tokens (positions are fixed)")
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--allow-no-gdn", action="store_true",
                    help="continue (off and head rows only) if the GDN state cannot be placed in the arena; default is to exit 4")
    ap.add_argument("--inner", action="store_true", help="ncu target (internal)")
    a = ap.parse_args(argv)
    if a.ctx is None:
        a.ctx = [0, 1024] if a.quick else [0, 16384]
    if a.quick:
        a.heads = [h for h in a.heads if h in ("bf16_full", "int8_48k")]
        a.iters, a.repeats = 50, 1
        a.ncu_ctx = [a.ctx[0]]
        a.ncu_heads = [a.heads[-1]]
    for h in a.heads + a.ncu_heads:
        if h not in HEADS:
            ap.error(f"unknown head {h}; choose from {list(HEADS)}")
    for c in a.conditions:
        if c not in CONDITIONS:
            ap.error(f"unknown condition {c}")
    return a


def plan_json(a) -> dict:
    f = modelspec.facts(modelspec.TEXT_CFG[a.model])
    heads = {k: head_bytes(k, f["vocab"], f["hidden"]) for k in a.heads}
    plans = {}
    for k, hb in heads.items():
        lay = arena_layout(f["state_bytes"], hb["weight"], hb["scale"])
        plans[k] = {"layout": lay, "windows": {c: window_region(c, lay) for c in a.conditions}}
    return {"exp": EXP, "model": a.model, "hf": hf_ids.resolve("qwen", a.model, offline=True), "ctx": a.ctx,
            "heads": heads, "gdn_state_bytes": f["state_bytes"], "arena": plans, "conditions": a.conditions,
            "ncu": {"ctx": a.ncu_ctx, "heads": a.ncu_heads, "no_ncu": a.no_ncu}, "window_method": a.window_method,
            "quick": a.quick}


def main(argv=None) -> int:
    a = parse_args(argv)
    if a.dry_run:
        common.print_plan(plan_json(a))
        return 0
    if a.inner:
        return run_inner(a)
    with common.term_exits():
        return run(a)


if __name__ == "__main__":
    sys.exit(main())
