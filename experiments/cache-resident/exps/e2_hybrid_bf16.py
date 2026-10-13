"""E2: stock hybrid models (Qwen3.5-0.8B / 2B, BF16) decode with torch.compile reduce-overhead.

For each context length: prefill random token ids, then greedy-decode with a static cache. DRAM bytes per
token come from an ncu pass (separate process, --inner); the split into weights / KV / GDN state is made by
diffing contexts. The fit us/token = a + bytes/bw feeds exps.proj_hybrid.

  python -m exps.e2_hybrid_bf16 --model 0.8B|2B [--ctx 0,4096,16384,32768,65536] [--decode 512]
                                [--compile-mode auto|reduce-overhead|default|cuda_graph_manual|eager]

The static cache is sized ctx + decode + margin and attention reads all of it every step, so KV bytes
scale with the cache length. Hybrid caches may not compile cleanly: the driver falls back
reduce-overhead -> default -> cuda_graph_manual -> eager and records compile_mode_used.
A dynamic (growing) cache changes the KV length every step, so compile would recompile each step and
graph replay is invalid: such rows run eager only. A compiled mode that recompiles in the timed loop is
rejected and the next mode is tried; dynamo_recompiles is recorded per row.
Phase 1 times every ctx; phase 2 runs the ncu passes (NCU_STEPS decode steps each) on the leftover budget, smallest and
largest ctx first, each capped to its share of the time left. Without ncu bytes on every row the fit uses analytic bytes.
"""
from __future__ import annotations

import argparse
import gc
import inspect
import sys
import time

import torch
import torch._dynamo

from exps import common, hf_ids, modelspec

EXP_NAMES = {"0.8B": "e2_08b", "2B": "e2_2b"}
DEFAULT_CTX = [0, 4096, 16384, 32768, 65536]
DEVICE = "cuda"  # tests switch this to cpu
MODES = ["reduce-overhead", "default", "cuda_graph_manual", "eager"]
COMPILED_MODES = ("reduce-overhead", "default")
CHUNK = 2048
MARGIN = 64
WARMUP_STEPS = 8
NCU_STEPS = 3
NCU_TIMEOUT_S = 1500


# ---------------------------------------------------------------- pure helpers

def parse_ctx(text: str) -> list[int]:
    return [int(t) for t in text.split(",") if t.strip() != ""]


def ctx_order(ctxs: list[int]) -> list[int]:
    """Run order: smallest and largest first so the fit has two points even if time runs out."""
    s = sorted(set(ctxs))
    if len(s) <= 2:
        return s
    mid = s[1:-1]
    return [s[0], s[-1]] + mid


def cache_len_for(ctx: int, decode: int) -> int:
    n = ctx + decode + MARGIN
    return (n + 63) // 64 * 64


def analytic(model: str, ctx: int, decode: int, static: bool = True, weight_bytes: int | None = None) -> dict:
    f = modelspec.facts(modelspec.TEXT_CFG[model])
    clen = cache_len_for(ctx, decode)
    kv_tokens = clen if static else ctx + decode // 2
    return {"weight_bytes": weight_bytes or f["weight_bytes"], "kv_bytes": f["kv_bytes_per_token"] * kv_tokens,
            "gdn_state_bytes": f["state_bytes"], "conv_state_bytes": f["conv_state_bytes"], "cache_len": clen,
            "kv_bytes_per_token": f["kv_bytes_per_token"], "n_layers": f["n_layers"]}


def split_bytes(rows: list[dict]) -> None:
    """Add row['split'] by diffing contexts against ctx 0 (needs dram_bytes_per_token on the ctx 0 row)."""
    base = next((r for r in rows if r["ctx"] == 0 and r.get("dram_bytes_per_token") is not None), None)
    for r in rows:
        r["split"] = r["split_read_only"] = None
        if base is None or r.get("dram_bytes_per_token") is None:
            continue
        state = r["analytic"]["gdn_state_bytes"]
        kv = r["dram_bytes_per_token"] - base["dram_bytes_per_token"]
        r["split"] = {"weights": base["dram_bytes_per_token"] - 2 * state, "kv": kv, "gdn_state": 2 * state}
        r["split_read_only"] = {"weights": base["dram_bytes_per_token"] - state, "kv": kv, "gdn_state": state}


def fit_floor_bw(points: list[tuple[float, float]], n_layers: int) -> dict | None:
    """Least squares us = a + bytes/bw over (bytes, us) points. Returns None with < 2 distinct byte values."""
    pts = [(float(b), float(u)) for b, u in points if b is not None and u is not None]
    if len({b for b, _ in pts}) < 2:
        return None
    n = len(pts)
    mx = sum(b for b, _ in pts) / n
    my = sum(u for _, u in pts) / n
    sxx = sum((b - mx) ** 2 for b, _ in pts)
    slope = sum((b - mx) * (u - my) for b, u in pts) / sxx  # us per byte
    a = my - slope * mx
    ss_tot = sum((u - my) ** 2 for _, u in pts)
    ss_res = sum((u - (a + slope * b)) ** 2 for b, u in pts)
    out = {"intercept_us": a, "n_points": n, "r2": 1.0 - ss_res / ss_tot if ss_tot > 0 else None,
           "floor_us_per_layer": a / n_layers if a > 0 else None,
           "eff_bw_gbps": 1e-3 / slope if slope > 0 else None}
    return out


def fit_rows(rows: list[dict], n_layers: int) -> dict | None:
    meas = [r for r in rows if r.get("us_per_token") is not None]
    if all(r.get("dram_bytes_per_token") is not None for r in meas) and meas:
        pts, src = [(r["dram_bytes_per_token"], r["us_per_token"]) for r in meas], "ncu_dram_bytes"
    else:
        pts = [(r["analytic"]["weight_bytes"] + 2 * r["analytic"]["gdn_state_bytes"] + r["analytic"]["kv_bytes"],
                r["us_per_token"]) for r in meas]
        src = "analytic_bytes"
    f = fit_floor_bw(pts, n_layers)
    if f:
        f["bytes_source"] = src
    return f


# ---------------------------------------------------------------- model loading (GPU pod)

class LMWrap(torch.nn.Module):
    """Text-only wrapper for a conditional-generation checkpoint: language_model + lm_head."""

    def __init__(self, full):
        super().__init__()
        self.body = full.model.language_model
        self.lm_head = full.lm_head
        self.config = full.config

    def forward(self, input_ids=None, past_key_values=None, cache_position=None, use_cache=True, logits_to_keep=1, **kw):
        from types import SimpleNamespace
        o = self.body(input_ids=input_ids, past_key_values=past_key_values, cache_position=cache_position,
                      use_cache=use_cache)
        h = o.last_hidden_state[:, -logits_to_keep:, :]
        return SimpleNamespace(logits=self.lm_head(h), past_key_values=getattr(o, "past_key_values", past_key_values))


def load_text_model(repo: str, info: dict):
    """Load text-only BF16 model on cuda. Records the class used and every failed attempt in info."""
    import transformers
    info["transformers"] = transformers.__version__
    errs = info.setdefault("load_errors", [])
    model = None
    for kw in ({"dtype": torch.bfloat16}, {"torch_dtype": torch.bfloat16}):
        try:
            model = transformers.AutoModelForCausalLM.from_pretrained(repo, attn_implementation="sdpa", **kw)
            info["load_path"] = "AutoModelForCausalLM"
            break
        except TypeError as e:
            errs.append(f"AutoModelForCausalLM {list(kw)}: {e}")
        except Exception as e:
            errs.append(f"AutoModelForCausalLM: {type(e).__name__}: {e}")
            break
    if model is None:
        full = None
        for name in ("AutoModelForImageTextToText", "Qwen3_5ForConditionalGeneration"):
            cls = getattr(transformers, name, None)
            if cls is None:
                continue
            try:
                full = cls.from_pretrained(repo, dtype=torch.bfloat16, attn_implementation="sdpa")
                info["load_path"] = f"{name}.model.language_model"
                break
            except Exception as e:
                errs.append(f"{name}: {type(e).__name__}: {e}")
        if full is None:
            raise RuntimeError("cannot load model: " + " | ".join(errs))
        model = LMWrap(full)
    info["class"] = type(model).__name__
    model = model.to(DEVICE).eval()
    return model


def text_config(model):
    cfg = model.config
    getter = getattr(cfg, "get_text_config", None)
    return getter() if getter else cfg


def keep_kw(model) -> str | None:
    try:
        params = inspect.signature(model.forward).parameters
    except (TypeError, ValueError):
        return "logits_to_keep"
    for k in ("logits_to_keep", "num_logits_to_keep"):
        if k in params:
            return k
    return "logits_to_keep" if any(p.kind == p.VAR_KEYWORD for p in params.values()) else None


def make_cache(model, max_len: int, info: dict):
    """StaticCache sized max_len; None (model builds its own dynamic cache) if unsupported."""
    from transformers import StaticCache
    cfg = text_config(model)
    tries = [dict(config=cfg, max_batch_size=1, max_cache_len=max_len, device=DEVICE, dtype=torch.bfloat16),
             dict(config=cfg, max_cache_len=max_len, device=DEVICE, dtype=torch.bfloat16),
             dict(config=cfg, max_cache_len=max_len)]
    for kw in tries:
        try:
            c = StaticCache(**kw)
            info["cache_kind"] = "static"
            return c
        except Exception as e:
            info.setdefault("cache_errors", []).append(f"{list(kw)}: {type(e).__name__}: {e}")
    info["cache_kind"] = "dynamic"
    return None


def fwd(model, ids, cache, pos, kk):
    kw = dict(input_ids=ids, past_key_values=cache, cache_position=pos, use_cache=True)
    if kk:
        kw[kk] = 1
    return model(**kw)


def iter_cache_tensors(cache):
    """Yield float tensors found in a cache object (layers, key/value lists, state lists)."""
    seen = set()

    def walk(o, depth):
        if depth > 4 or o is None or id(o) in seen:
            return
        seen.add(id(o))
        if isinstance(o, torch.Tensor):
            if o.is_floating_point():
                yield o
        elif isinstance(o, (list, tuple)):
            for x in o:
                yield from walk(x, depth + 1)
        elif isinstance(o, dict):
            for x in o.values():
                yield from walk(x, depth + 1)
        elif hasattr(o, "__dict__"):
            for x in vars(o).values():
                yield from walk(x, depth + 1)

    yield from walk(cache, 0)


def fill_cache_random(cache) -> int:
    """Fill every float cache tensor with small random values (bytes, not content, matter). Returns count."""
    n = 0
    for t in iter_cache_tensors(cache):
        t.normal_(0.0, 0.02)
        n += 1
    return n


def fix_cache_len(cache, n: int) -> int:
    """Set every static layer's device-side length counter to n; returns how many were set.

    transformers 5.x StaticLayer writes the KV at cumulative_length and adds to it on every forward (cache_position
    is ignored), so a replayed CUDA graph would walk off the end of the cache. Dynamic layers keep an int: untouched.
    """
    k = 0
    for ly in getattr(cache, "layers", None) or ():
        t = getattr(ly, "cumulative_length", None)
        if isinstance(t, torch.Tensor):
            t.fill_(n)
            k += 1
    return k


def prefill(model, cache, ctx: int, kk, seed: int = 1234):
    """Chunked eager prefill of random ids; returns the cache (a dynamic cache is created on the first call)."""
    g = torch.Generator().manual_seed(seed)
    ids = torch.randint(1000, 100000, (1, ctx), generator=g).to(DEVICE)
    for s in range(0, ctx, CHUNK):
        e = min(ctx, s + CHUNK)
        pos = torch.arange(s, e, device=DEVICE)
        out = fwd(model, ids[:, s:e], cache, pos, kk)
        if cache is None:
            cache = getattr(out, "past_key_values", None)
    return cache


class Decoder:
    """One decode step in a chosen mode; tok/pos are static tensors updated in place."""

    def __init__(self, model, cache, ctx: int, kk, mode: str):
        self.mode = mode
        self.tok = torch.full((1, 1), 1234, dtype=torch.long, device=DEVICE)
        self.pos = torch.full((1,), ctx, dtype=torch.long, device=DEVICE)
        self.ctx = ctx
        self.cache = cache

        def step():
            out = fwd(model, self.tok, cache, self.pos, kk)
            return out.logits[:, -1, :].argmax(-1)

        if mode in ("reduce-overhead", "default"):
            torch._dynamo.config.cache_size_limit = max(64, torch._dynamo.config.cache_size_limit)
            self.fn = torch.compile(step, mode=mode, fullgraph=False, dynamic=False)
        elif mode == "cuda_graph_manual":
            side = torch.cuda.Stream()
            side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side):
                for _ in range(3):
                    step()
            torch.cuda.current_stream().wait_stream(side)
            torch.cuda.synchronize()
            self.graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph):
                self.out = step()
            self.fn = lambda: (self.graph.replay(), self.out)[1]
        elif mode == "eager":
            self.fn = step
        else:
            raise ValueError(mode)

    def once(self):
        nxt = self.fn()
        self.tok.copy_(nxt.reshape(1, 1))
        self.pos.add_(1)

    def reset(self):
        self.tok.fill_(1234)
        self.pos.fill_(self.ctx)
        fix_cache_len(self.cache, self.ctx)


def dynamo_graphs() -> int:
    """Count of graphs dynamo has compiled so far; a rise during a timed loop means recompiles."""
    try:
        return int(torch._dynamo.utils.counters["stats"]["unique_graphs"])
    except Exception:
        return 0


def pick_decoder(model, cache, ctx, kk, start_mode: str, info: dict, static: bool = True, exclude=()):
    """Try modes from start_mode down the fallback chain; return (Decoder, mode).

    A dynamic cache (static False) only works in eager: compile recompiles on every KV length and graph replay is invalid.
    """
    if static:
        chain = MODES[MODES.index(start_mode):] if start_mode in MODES else MODES
    else:
        chain = ["eager"]
    chain = [m for m in chain if m not in exclude]
    init = Decoder(model, cache, ctx, kk, "eager")  # one eager step: lazy cache layers are built outside any graph
    init.once()
    torch.cuda.synchronize()
    del init
    vocab = modelspec.TEXT_CFG["0.8B"]["vocab_size"]
    for mode in chain:
        try:
            torch._dynamo.reset()
            if static:
                fix_cache_len(cache, ctx)  # a failed earlier mode may have advanced the counter
            d = Decoder(model, cache, ctx, kk, mode)
            for _ in range(WARMUP_STEPS):
                d.once()
            torch.cuda.synchronize()
            if not 0 <= int(d.tok.item()) < vocab:
                raise RuntimeError("token id out of range")
            if mode in COMPILED_MODES:  # steady state must not compile again
                g0 = dynamo_graphs()
                for _ in range(2):
                    d.once()
                torch.cuda.synchronize()
                if dynamo_graphs() != g0:
                    raise RuntimeError("dynamo recompiles on every step")
            return d, mode
        except Exception as e:
            info.setdefault("mode_errors", []).append(f"ctx={ctx} {mode}: {type(e).__name__}: {str(e)[:300]}")
            try:
                torch.cuda.synchronize()
            except Exception:
                pass
    raise RuntimeError("no decode mode worked: " + " | ".join(info.get("mode_errors", [])[-4:]))


GDN_FNS = ("torch_recurrent_gated_delta_rule", "torch_chunk_gated_delta_rule")


def _origin_module(fn) -> str:
    """Module of the function fn runs. HF hub wrappers copy the torch fallback's name and module (functools.wraps), so
    read the dispatch target from the wrapper's closure when there is one."""
    try:
        impl = inspect.getclosurevars(fn).nonlocals.get("implementation")
    except (TypeError, ValueError):
        impl = None
    return str(getattr(impl if callable(impl) else fn, "__module__", "") or "")


def gdn_impl(model) -> str:
    """'fla' if the GatedDeltaNet layers use fla kernels, 'torch_fallback' if they use HF torch ops, 'unknown' if not found.

    transformers 5.x: module-level GDN functions wrapped by use_kernel_func_from_hub_with_fallback. 4.x: layer attributes.
    """
    fns = []
    for m in model.modules():
        fn = getattr(m, "recurrent_gated_delta_rule", None) or getattr(m, "chunk_gated_delta_rule", None)
        if fn is not None:
            fns.append(fn)
        elif "GatedDeltaNet" in type(m).__name__:
            mod = sys.modules.get(type(m).__module__)
            fns += [f for f in (getattr(mod, n, None) for n in GDN_FNS) if f is not None]
        if fns:
            break
    if not fns:
        return "unknown"
    return "fla" if all(_origin_module(f).split(".")[0] == "fla" for f in fns) else "torch_fallback"


def record_gdn_impl(model, info: dict) -> None:
    for mod in ("fla", "causal_conv1d"):
        try:
            __import__(mod)
            info[f"{mod}_available"] = True
        except Exception:
            info[f"{mod}_available"] = False
    try:
        info["gdn_impl"] = gdn_impl(model)
    except Exception:
        info["gdn_impl"] = "unknown"


def setup(args, info):
    """Resolve + load the model; returns (model, kk). Uses an alternative repo if the pinned one is gone."""
    rid = hf_ids.resolve("qwen", args.model)
    info["hf"] = rid
    repo = rid["repo"]
    if rid["exists"] is False and rid["alternatives"]:
        repo = rid["alternatives"][0]
        info["hf_note"] = f"{rid['repo']} missing; using {repo} (alternatives {rid['alternatives']})"
    info["repo_used"] = repo
    model = load_text_model(repo, info)
    record_gdn_impl(model, info)
    return model, keep_kw(model)


def inner_decoder(model, kk, args, ctx: int, kind: str, info: dict):
    """Cache + decoder for one cache kind, as measure_ctx builds them (static cache is random filled, not prefilled)."""
    cache = make_cache(model, cache_len_for(ctx, args.decode), info) if kind == "static" else None
    static = cache is not None
    eff = ctx if static else max(ctx, 1)
    if not static:
        info["cache_kind"] = "dynamic"
        cache = prefill(model, None, eff, kk)
    elif eff > 0:
        init = Decoder(model, cache, eff, kk, "eager")  # build lazy layers, then fill with random values
        init.once()
        if not fill_cache_random(cache):
            cache = prefill(model, cache, eff, kk)
        fix_cache_len(cache, eff)
    d, _ = pick_decoder(model, cache, eff, kk, args.compile_mode, info, static=static)
    return d


def run_inner(args) -> int:
    """ncu target: set up, warm up, then run NCU_STEPS decode steps between profiler start/stop."""
    info: dict = {}
    model, kk = setup(args, info)
    ctx = args.ctx[0]
    kinds = ["dynamic"] if args.cache_kind == "dynamic" else ["static", "dynamic"]
    d = None
    for i, kind in enumerate(kinds):
        try:
            d = inner_decoder(model, kk, args, ctx, kind, info)
            break
        except Exception as e:
            if i == len(kinds) - 1:
                raise
            print(f"[e2 inner] {kind} cache failed ({type(e).__name__}: {str(e)[:200]}); retrying dynamic", file=sys.stderr,
                  flush=True)
            free_gpu()
    d.reset()
    torch.cuda.synchronize()
    torch.cuda.profiler.start()
    torch.cuda.nvtx.range_push("cr_decode")
    for _ in range(NCU_STEPS):
        d.once()
    torch.cuda.synchronize()
    torch.cuda.nvtx.range_pop()
    torch.cuda.profiler.stop()
    return 0


def free_gpu() -> None:
    gc.collect()
    torch._dynamo.reset()
    torch.cuda.empty_cache()


def measure_ctx(model, kk, args, ctx: int, info: dict, weight_bytes: int) -> dict:
    """Prefill and time the decode at one ctx. A failure with the static cache is retried with the dynamic one."""
    t0 = time.monotonic()
    static_error = None
    for try_static in (True, False):
        cache, d, static = None, None, False
        try:
            if try_static:
                cache = make_cache(model, cache_len_for(ctx, args.decode), info)
                static = cache is not None
            else:
                info["cache_kind"] = "dynamic"
            eff = ctx if static else max(ctx, 1)  # a dynamic cache needs at least one prefilled token
            if eff > 0:
                cache = prefill(model, cache, eff, kk)
            skipped: list[str] = []
            while True:
                d, mode = pick_decoder(model, cache, eff, kk, args.compile_mode, info, static=static, exclude=skipped)
                d.reset()
                g0 = dynamo_graphs()
                ev = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
                ev[0].record()
                for _ in range(args.decode):
                    d.once()
                ev[1].record()
                torch.cuda.synchronize()
                sec = ev[0].elapsed_time(ev[1]) / 1000.0
                recompiles = dynamo_graphs() - g0
                if recompiles <= 0 or mode not in COMPILED_MODES:
                    break
                info.setdefault("mode_errors", []).append(f"ctx={ctx} {mode}: {recompiles} dynamo recompiles in timed loop")
                skipped.append(mode)  # compile time is inside the timing: reject, try the next mode
                d = None
                torch.cuda.empty_cache()
            row = {"model": args.model, "ctx": ctx, "tok_s": args.decode / sec, "us_per_token": sec * 1e6 / args.decode,
                   "compile_mode_used": mode, "cache_kind": "static" if static else "dynamic", "dynamo_recompiles": max(recompiles, 0),
                   "modes_rejected_recompile": skipped, "dram_bytes_per_token": None,
                   "l2_hit_rate": None, "ncu_us_per_token": None, "counters_ok": False, "gdn_impl": info.get("gdn_impl"),
                   "analytic": analytic(args.model, ctx, args.decode, static, weight_bytes),
                   "prefill_and_compile_s": time.monotonic() - t0}
            if static_error:
                row["static_cache_error"] = static_error
            return row
        except Exception as e:
            err = f"{type(e).__name__}: {str(e)[:300]}"
            if not (try_static and static):
                raise RuntimeError(err if static_error is None else f"static: {static_error} | dynamic: {err}") from e
            static_error = err
        finally:
            del cache, d
            free_gpu()
    raise AssertionError("unreachable")


def finish(args, rows, notes, info, dev, counters_all, nl, budget_note) -> None:
    """Fit, split and write the final result; never raises past the write."""
    rows.sort(key=lambda r: r["ctx"])
    fit = None
    try:
        split_bytes(rows)
        fit = fit_rows(rows, nl)
    except Exception as e:
        notes.append(f"fit/split failed: {type(e).__name__}: {e}")
    if budget_note:
        notes.append(budget_note)
    if not counters_all:
        notes.append("fit and split use analytic bytes where ncu bytes are missing (see fit.bytes_source)")
    notes.append("ncu dram_bytes_per_token counts reads only; split has the spec 2x state, split_read_only has 1x")
    notes.append("ncu is kernel replay; rows with ncu_l2_state_ok false ran more than one pass on a warmed L2 "
                 "(--cache-control none): dram bytes may be under-reported")
    notes.append("static cache is read in full every step, so KV bytes scale with cache_len not ctx")
    if any(r.get("cache_kind") == "dynamic" for r in rows):
        notes.append("rows with cache_kind dynamic ran eager only (growing KV length would recompile every step)")
    notes.append("reduce-overhead can skip CUDA graphs silently (mutated inputs); compare us/token with the other modes")
    if info.get("gdn_impl") != "fla":
        notes.append(f"gdn_impl={info.get('gdn_impl')}: GatedDeltaNet runs HF torch ops (several state-sized fp32 temporaries per "
                     "layer per token), not a fused fla kernel; fit.floor_us_per_layer and fit.eff_bw_gbps describe that path, "
                     "so proj_hybrid built on them is not representative of a fused implementation")
    if fit:
        fit["gdn_impl"] = info.get("gdn_impl")
    notes.append("ncu counts kernel nodes only; any cudaMemcpy node (e.g. a same-dtype copy_ into the cache) is excluded from dram_bytes_per_token")
    config = {"model": args.model, "ctx": args.ctx, "decode": args.decode, "compile_mode_requested": args.compile_mode,
              "torch": torch.__version__, "quick": args.quick, "device": dev, "load": info}
    common.write_result(EXP_NAMES[args.model], args, config, rows, notes, counters_all, extra={"fit": fit})


def run(args) -> int:
    from harness import l2persist
    if not torch.cuda.is_available():
        print("error: no CUDA device (use --dry-run for the plan)", file=sys.stderr)
        return 2
    args.gpu = common.gpu_tag(args)
    exp = EXP_NAMES[args.model]
    info: dict = {}
    notes: list[str] = []
    ncu_ok, ncu_note = common.counters_available(args)
    if ncu_note:
        notes.append(ncu_note)
    budget = common.Budget(args.budget_min or (6 if args.quick else 27))
    dev = l2persist.log_device(0)
    try:
        model, kk = setup(args, info)
    except Exception as e:
        notes.append(f"model setup failed: {type(e).__name__}: {e}")
        common.write_result(exp, args, {"model": args.model, "load": info, "device": dev}, [], notes, False)
        print(f"error: {notes[-1]}", file=sys.stderr)
        return 3
    weight_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    nl = modelspec.facts(modelspec.TEXT_CFG[args.model])["n_layers"]
    rows: list[dict] = []
    counters_all = ncu_ok
    budget_note = None
    partial_cfg = {"model": args.model, "ctx": args.ctx, "decode": args.decode, "load": info, "device": dev}
    try:
        for ctx in ctx_order(args.ctx):
            if budget.expired(120):
                budget_note = f"time budget used up; ctx {ctx} and later skipped"
                break
            free_gpu()
            try:
                row = measure_ctx(model, kk, args, ctx, info, weight_bytes)
            except Exception as e:
                notes.append(f"ctx={ctx} failed: {e}")
                counters_all = False
                rows.append({"model": args.model, "ctx": ctx, "tok_s": None, "us_per_token": None, "error": str(e),
                             "compile_mode_used": None, "cache_kind": None, "dram_bytes_per_token": None,
                             "l2_hit_rate": None, "ncu_us_per_token": None, "counters_ok": False, "gdn_impl": info.get("gdn_impl"),
                             "analytic": analytic(args.model, ctx, args.decode, True, weight_bytes)})
                common.write_partial(exp, args, partial_cfg, rows, notes, False)
                print("[e2] %s ctx=%d failed: %s" % (args.model, ctx, e), flush=True)
                continue
            rows.append(row)
            common.write_partial(exp, args, partial_cfg, rows, notes, False)
            print("[e2] %s ctx=%d tok/s=%.1f mode=%s" % (args.model, ctx, row["tok_s"], row["compile_mode_used"]), flush=True)
        # phase 2: ncu on the leftover budget, in ctx_order (smallest and largest ctx first)
        jobs = [r for r in rows if r.get("us_per_token") is not None] if ncu_ok else []
        for i, row in enumerate(jobs):
            if budget.expired(180):
                for r in jobs[i:]:
                    r["ncu_error"] = "skipped: time budget"
                counters_all = False
                break
            share = budget.left() / (len(jobs) - i) if budget.limit is not None else NCU_TIMEOUT_S
            res = common.ncu_run(common.py_module_argv("exps.e2_hybrid_bf16", "--inner", "--model", args.model, "--ctx", str(row["ctx"]),
                                                       "--decode", str(args.decode), "--compile-mode", row["compile_mode_used"],
                                                       "--cache-kind", row["cache_kind"]),
                                 NCU_STEPS, profile_from_start_off=True, app_replay=False,
                                 timeout=budget.timeout(min(NCU_TIMEOUT_S, share), reserve_s=30), env=common.inner_env())
            st = common.ncu_stats(res)
            row.update(dram_bytes_per_token=st["dram_bytes_per_token"], l2_hit_rate=st["l2_hit_rate"],
                       ncu_us_per_token=st["ncu_us_per_token"], counters_ok=st["counters_ok"], ncu_error=st["ncu_error"],
                       ncu_replay_mode=st["ncu_replay_mode"], ncu_passes=st["ncu_passes"],
                       ncu_l2_state_ok=st["ncu_l2_state_ok"])
            counters_all = counters_all and st["counters_ok"]
            common.write_partial(exp, args, partial_cfg, rows, notes, False)
            print("[e2] ncu %s ctx=%d dram/tok=%s" % (args.model, row["ctx"], row["dram_bytes_per_token"]), flush=True)
    except BaseException as e:
        notes.append(f"run interrupted: {type(e).__name__}: {e}")
        raise
    finally:
        finish(args, rows, notes, info, dev, counters_all, nl, budget_note)
    return 0 if any(r["us_per_token"] is not None for r in rows) else 1


# ---------------------------------------------------------------- CLI

def parse_args(argv=None):
    ap = argparse.ArgumentParser(prog="exps.e2_hybrid_bf16", description=__doc__.split("\n")[0])
    common.add_common_args(ap)
    ap.add_argument("--model", choices=sorted(EXP_NAMES), default="0.8B")
    ap.add_argument("--ctx", type=parse_ctx, default=None)
    ap.add_argument("--decode", type=int, default=512)
    ap.add_argument("--compile-mode", default="reduce-overhead", choices=["auto"] + MODES)
    ap.add_argument("--cache-kind", choices=["static", "dynamic"], default="static", help="ncu target cache kind (internal)")
    ap.add_argument("--inner", action="store_true", help="ncu target (internal)")
    a = ap.parse_args(argv)
    if a.compile_mode == "auto":
        a.compile_mode = MODES[0]
    if a.ctx is None:
        a.ctx = [0, 1024] if a.quick else list(DEFAULT_CTX)
    if a.quick:
        a.decode = min(a.decode, 32)
    return a


def plan_json(a) -> dict:
    return {"exp": EXP_NAMES[a.model], "model": a.model, "hf": hf_ids.resolve("qwen", a.model, offline=True),
            "ctx_order": ctx_order(a.ctx), "decode": a.decode, "compile_mode": a.compile_mode, "fallback_chain": MODES,
            "analytic": {str(c): analytic(a.model, c, a.decode) for c in a.ctx}, "no_ncu": a.no_ncu, "quick": a.quick}


def main(argv=None) -> int:
    a = parse_args(argv)
    if a.dry_run:
        common.print_plan(plan_json(a))
        return 0
    if a.inner:
        a.ctx = [a.ctx[0]]
        return run_inner(a)
    with common.term_exits():
        return run(a)


if __name__ == "__main__":
    sys.exit(main())
