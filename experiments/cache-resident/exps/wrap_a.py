"""Wrap A (A1 + A2): evidence for the fork PR that enables the PQ2_0 multi-column kernel on sm_120, one 600 W PRO 6000 pod session.

  python -m exps.fetch_models --models pq2,draft_q8_0 --timeout-min 6 && python -m exps.wrap_a [--parts t,b,s] [--budget-min 12]

The kernel is the #306 mat-vec (mmvq-pq2_0.cuh) for 3-8 columns; the prototype switch GGML_CUDA_PQ2_MULTICOL=1 enables it on cc >= Ada
(default: Ada only) and the kernel calls ggml_cuda_pdl_sync() before it reads the activations. off = variable unset, on = set to 1.
Gate: the GPU tag must be pro6000-ws or pro6000-server (600 W) and the default power limit >= 550 W; a Max-Q or any other GPU gets no
timing (exit 1). --allow-any-gpu turns the gate into a note (development only).
Health: the 600 W card must also run at its default limit during the timing. r1.host_probe / host_health run before part b and before part s
(an unhealthy host skips the part, exit 1). During part b every run is checked (clocks, power cap); the first unhealthy round and all later
rounds are left out of the paired gains (data.bench_health), the verdict lines then start with UNHEALTHY HOST and the exit code is 1, so a
capped host cannot pass as PR evidence.
t  correctness. test-backend-ops test -o MUL_MAT -b CUDA0 -p type_a=pq2_0, with on three times (a PDL race is timing dependent), then off
   (r2b_multicol part t). A failure with on stops the run before any timing.
b  A1, paired kernel timing. llama-bench on Bonsai 2 27B PQ2_0 (-fa on, -ub 2048, f16 KV, depth 0): -p 1,...,9 -n 0 -r 5 and
   -p 0 -n 128 -r 5, off and on alternating, --rounds rounds (default 4: off on / on off / off on / on off). Reported: ms per width
   per config (median over rounds), the gain of on over off per round and its median / min / max (widths 1, 2 and 9 do not use
   the kernel: their gain is the noise floor), decode tok/s.
s  A2, end to end. llama-server, the R2a balanced subset (--per-category prompts per category, default 4, same seed, ctx 16384, -lv 4):
   target alone (base), draft-simple K=4 with the Qwen3.5-0.8B Q8_0 draft with the kernel off (k4q8_off) and on (k4q8_on).
   Interleaved by prompt blocks: --blocks blocks (default 2); in every block each config gets a fresh
   server; the config order changes per block (r2b_spec.block_orders) so on runs before off as often as after. Reported: per config
   and category tok/s, speedup over base with bootstrap 95% CIs, r, alpha, ms per step and its split from the server statistics
   (draft dur g vs the rest, which holds the target verify the kernel changes), greedy text equality of on vs off and of each vs base
   (informational: the kernel changes the accumulation order, so mismatches are expected numerics), and on over off per category
   (paired bootstrap CI). The draft state of k4q8 is the stale-rollback one (see wrap_b); it is the same in off and on.
Parts run in the order t, b, s with weights 2 : 6 : 4.5 of --budget-min (unused time rolls forward).
Result: results/wrap_a/<gpu>/<stamp>.json with data.gate, data.tbo, data.bench, data.bench_summary, data.bench_paired, data.bench_health,
data.spec, data.host, data.host_b.
Time: failsafe 40 min = bootstrap with the cold build ~15, downloads <= 6, wrap_a 12, reserve 5 (cost model: s about 4 min at the defaults).
Cost at the community price of about $1.69 per hour: pod time about 32 min, about $0.9; the secure price of $2.09 gives about $1.1.
"""
from __future__ import annotations

import argparse
import os
import sys

from exps import common, e4_bonsai, e4_pdl, hf_ids, make_prompts, r1, r2a_longctx as r2a, r2b_multicol as mc, r2b_spec as sp

EXP = "wrap_a"
ENV_VAR = mc.ENV_VAR
KCONFIGS = [("off", {}), ("on", {ENV_VAR: "1"})]
TBO_ORDER = [KCONFIGS[1]] * 3 + [KCONFIGS[0]]     # on three times: a PDL race is timing dependent; off last (a short budget drops it first)
ROUNDS = 4
BLOCKS = 2
PER_CATEGORY = 4
SPEC_NAMES = ["base", "k4q8_off", "k4q8_on"]
PARTS = "tbs"
WEIGHTS = {"t": 2.0, "b": 6.0, "s": 4.5}
DEFAULT_TOTAL_MIN = 12.0
END_RESERVE_S = 25.0


def spec_configs() -> list[dict]:
    """base and the R2b k4_q8 config (draft-simple K=4, Q8_0 draft, -lv 4), the latter with the kernel off and on."""
    by = {c["name"]: c for c in sp.matrix_configs(["base", "k4_q8"])}
    return [dict(by["base"], env={}),
            dict(by["k4_q8"], name="k4q8_off", env={}, desc="draft-simple K=4, Qwen3.5-0.8B Q8_0 draft, PQ2_0 multi-column kernel off"),
            dict(by["k4_q8"], name="k4q8_on", env={ENV_VAR: "1"}, desc="draft-simple K=4, Qwen3.5-0.8B Q8_0 draft, PQ2_0 multi-column kernel on")]


def bench_verdict(paired: dict, health: dict | None = None) -> list[str]:
    """Paired gains per width. After an unhealthy host every line starts with UNHEALTHY HOST and says which rounds were left out."""
    pre = ""
    if health and health.get("first_bad_round") is not None:
        pre = "UNHEALTHY HOST (%s during round %d %s; that round and later ones left out): " % (
            health["problem"], health["first_bad_round"], health.get("where", ""))
    out = []
    for w, e in paired.get("pp", {}).items():
        out.append("%swidth %s: on over off %+.1f%% (rounds %s, min %+.1f%%, max %+.1f%%)%s" % (
            pre, w, e["median"], e["n"], e["min"], e["max"], "" if int(w) in mc.MC_WIDTHS else " (not a multi-column width)"))
    tg = paired.get("tg", {}).get("1")
    if tg:
        out.append("%sdecode tg128: on over off %+.1f%% (min %+.1f%%, max %+.1f%%)" % (pre, tg["median"], tg["min"], tg["max"]))
    if pre and not out:
        out.append(pre + "no healthy round: no paired gain")
    return out


def spec_summary(runs: dict) -> dict:
    out = {"vs_base": r2a.summarize(runs, ci=True), "on_vs_off": sp.pair_by_category(runs, "k4q8_on", "k4q8_off"),
           "step_split": {n: r.get("step_split") for n, r in runs.items()}}
    a = (out["on_vs_off"] or {}).get("all") or {}
    ms = {n: (out["step_split"].get(n) or {}).get("ms_per_step") for n in ("k4q8_on", "k4q8_off")}
    out["verdict"] = None if not a.get("speedup") else ("k4q8_on over k4q8_off %.3fx (95%% CI %s), greedy text equal %s/%s (informational: "
                                                         "the kernel changes the accumulation order), ms/step on %s vs off %s" % (
        a["speedup"], "n/a" if not a.get("speedup_ci95") else "[%.3f, %.3f]" % tuple(a["speedup_ci95"]), a.get("text_equal"), a.get("text_compared"),
        *("%.2f" % ms[n] if ms[n] else "n/a" for n in ("k4q8_on", "k4q8_off"))))
    return out


def part_budget_s(left_s: float, part: str, remaining: list[str]) -> float:
    tot = sum(WEIGHTS[p] for p in remaining) or 1.0
    return max(left_s - END_RESERVE_S, 0.0) * WEIGHTS[part] / tot


def run(args) -> int:
    notes: list[str] = []
    args.gpu = common.gpu_tag(args)
    for k in [k for k in os.environ if k == ENV_VAR or k.startswith("GGML_EXP_")]:       # the configs set them, nothing else may
        notes.append(f"{k} was set in the environment; removed")
        os.environ.pop(k)
    bin_dir = e4_bonsai.llama_bin()
    missing = [b for b in ("llama-bench", "test-backend-ops", "llama-server") if not (bin_dir / b).exists()]
    total = common.Budget(args.budget_min)
    secrets = r1.secret_values()
    order = sp.balanced_subset(make_prompts.load_longctx(args.prompts), args.seed, args.per_category)
    if args.quick:
        order = [dict(p, n_predict=min(p["n_predict"], 32)) for p in order]
    blocks = sp.prompt_blocks(order, args.blocks)
    config = {"parts": args.parts, "budget_min": args.budget_min, "rounds": args.rounds, "blocks": args.blocks, "kernel_configs": dict(KCONFIGS),
              "spec_configs": [c["name"] for c in spec_configs()], "seed": args.seed, "per_category": args.per_category, "ctx": args.ctx,
              "prompt_ids": [p["id"] for p in order], "models": {}, "quick": args.quick}
    doc: dict = {"gate": None, "tbo": [], "bench": [], "bench_summary": {}, "bench_paired": {}, "bench_health": {}, "spec": None, "host": None,
                 "host_b": None}
    rows: list[dict] = doc["bench"]
    health: dict = doc["bench_health"]          # first_bad_round / problem of part b; rounds from there on are left out of the gains

    def save(final: bool = False) -> None:
        doc["bench_summary"] = mc.summarize_bench(rows, [c for c, _ in KCONFIGS], before_round=health.get("first_bad_round"))
        doc["bench_paired"] = mc.paired_rounds(rows, "off", "on", before_round=health.get("first_bad_round"))
        (common.write_result if final else common.write_partial)(EXP, args, r1.scrub(config, secrets), [], r1.scrub(notes, secrets), False,
                                                                 extra=r1.scrub(doc, secrets))

    if missing:
        notes.append(f"{missing} not found in {bin_dir}: build llama.cpp (CR_BUILD_LLAMA=1)")
        save(final=True)
        return 2
    smi = e4_pdl.smi_sample()
    bad = mc.gpu600_problem(args.gpu, smi)
    doc["gate"] = {"gpu": args.gpu, "smi": smi, "problem": bad, "allow_any_gpu": args.allow_any_gpu}
    if bad and not args.allow_any_gpu:
        notes.append(f"gate: {bad}; no timing on this host (A1 and A2 need a 600 W PRO 6000)")
        save(final=True)
        return 1
    if bad:
        notes.append(f"gate (--allow-any-gpu): {bad}; the timings are not the PR evidence")
    try:
        target, config["models"]["pq2"] = e4_pdl.model_path("pq2", notes)
    except Exception as e:
        notes.append(f"model download failed: {type(e).__name__}: {e}")
        save(final=True)
        return 3
    parts = [p for p in PARTS if p in args.parts]
    rc = 0
    cut = {"part": None}
    try:
        for i, p in enumerate(parts):
            sub_s = part_budget_s(total.left(), p, parts[i:])
            if p == "t":
                sub_s = max(sub_s, min(90.0, total.left()))      # correctness gates the timings: never dropped for budget
            if sub_s < 60:
                notes.append(f"part {p} skipped: only {sub_s:.0f} s of budget left")
                continue
            sub = common.Budget(sub_s / 60.0)
            cut["part"] = p
            print(f"[{EXP}] part {p}: {sub_s / 60:.1f} min of {total.left() / 60:.1f} min left", flush=True)
            try:
                if p == "t":
                    config["gpu_mem_before_t"] = mc.gpu_mem()
                    doc["tbo"] = mc.tbo_part(bin_dir, sub, notes, order=TBO_ORDER)
                    on = [t for t in doc["tbo"] if t.get("config") == "on"]
                    if not on or not all(mc.tbo_passed(t) for t in on):
                        notes.append(f"stop: test-backend-ops did not pass with {ENV_VAR}=1; parts b and s not run")
                        rc = 1
                        cut["part"] = None
                        break
                    if any(not mc.tbo_passed(t) for t in doc["tbo"]):
                        notes.append("test-backend-ops failed with the kernel off: see data.tbo")
                        rc = 1
                elif p == "b":
                    probe = r1.host_probe(bin_dir, target)
                    hbad, cap = r1.host_health(e4_pdl.smi_sample(), (probe or {}).get("smi"), check_power=not args.allow_power_cap)
                    doc["host_b"] = {"problem": hbad, "power_cap_note": cap, "probe": probe}
                    if hbad:
                        notes.append(f"UNHEALTHY HOST: {hbad}; part b skipped (A1 needs a healthy 600 W card)")
                        rc = 1
                    else:
                        mc.bench_part(bin_dir, target, sub, notes, rows, args.allow_power_cap, save, configs=KCONFIGS, rounds=args.rounds, tag=EXP,
                                      health=health)
                        if health.get("first_bad_round") is not None:
                            rc = 1
                else:
                    rc = max(rc, spec_part(args, bin_dir, target, blocks, sub_s, notes, doc, config, save))
            except Exception as e:  # a failed part must not lose the others
                notes.append(f"part {p} crashed: {type(e).__name__}: {e}")
            cut["part"] = None
            save()
    finally:
        if cut["part"]:
            notes.append(f"run cut during part {cut['part']}: the results of that part are partial")
        if (doc.get("spec") or {}).get("runs"):
            try:
                doc["spec"]["summary"] = spec_summary(doc["spec"]["runs"])
            except Exception as e:
                notes.append(f"final spec summary failed: {type(e).__name__}: {e}")
        paired = mc.paired_rounds(rows, "off", "on", before_round=health.get("first_bad_round"))
        notes.extend(bench_verdict(paired, health))
        if doc.get("host_b") and doc["host_b"]["problem"] or any(n.startswith("UNHEALTHY HOST") for n in notes):
            rc = max(rc, 1)
        if (doc.get("spec") or {}).get("summary", {}).get("verdict"):
            notes.append(doc["spec"]["summary"]["verdict"])
        notes.append(mc.PDL_LC_NOTE)
        save(final=True)
    return rc


def spec_part(args, bin_dir, target: str, blocks: list, budget_s: float, notes: list, doc: dict, config: dict, save) -> int:
    try:
        draft, config["models"]["draft_q8_0"] = e4_pdl.model_path("draft_q8_0", notes)
    except Exception as e:
        notes.append(f"s: Q8_0 draft unavailable ({type(e).__name__}: {e}); part s not run")
        return 1
    t0 = common.Budget(budget_s / 60.0)
    probe = r1.host_probe(bin_dir, target)
    bad, cap = r1.host_health(e4_pdl.smi_sample(), (probe or {}).get("smi"), check_power=not args.allow_power_cap)
    doc["host"] = {"problem": bad, "power_cap_note": cap, "probe": probe}
    if bad:
        notes.append(f"UNHEALTHY HOST: {bad}; part s skipped")
        return 1
    cfgs = spec_configs()
    drafts = {"q8_0": draft}
    orders = sp.block_orders([c["name"] for c in cfgs], len(blocks))
    doc["spec"] = sp.run_blocks(cfgs=cfgs, blocks=blocks, orders=orders, bin_dir=bin_dir, target=target,
                                cmd_for=lambda c: sp.config_cmd(bin_dir, target, drafts, c, ctx=args.ctx), budget_s=t0.left(), notes=notes,
                                gpu=args.gpu, exp=EXP, ctx=args.ctx, on_update=lambda s: (doc.__setitem__("spec", s), save()), tag=EXP)
    return 0


# ---------------------------------------------------------------- CLI

def parse_args(argv=None):
    ap = argparse.ArgumentParser(prog="exps.wrap_a", description=__doc__.split("\n")[0])
    common.add_common_args(ap)
    ap.add_argument("--parts", type=lambda t: t.replace(",", ""), default=PARTS, help="t (test-backend-ops), b (llama-bench), s (spec run)")
    ap.add_argument("--rounds", type=int, default=ROUNDS)
    ap.add_argument("--blocks", type=int, default=BLOCKS)
    ap.add_argument("--per-category", type=int, default=PER_CATEGORY)
    ap.add_argument("--seed", type=int, default=sp.SEED)
    ap.add_argument("--ctx", type=int, default=sp.CTX)
    ap.add_argument("--prompts", default=str(make_prompts.R2A_OUT))
    ap.add_argument("--allow-power-cap", action="store_true", help="a power cap below default is a note, not a health problem")
    ap.add_argument("--allow-any-gpu", action="store_true", help="run on a host that is not a 600 W PRO 6000 (development only)")
    a = ap.parse_args(argv)
    bad = [p for p in a.parts if p not in PARTS]
    if bad:
        ap.error(f"unknown part(s) {bad}; parts: {PARTS}")
    if a.rounds < 3:
        ap.error("--rounds must be at least 3")
    a.budget_min = a.budget_min or (8.0 if a.quick else DEFAULT_TOTAL_MIN)
    if a.quick:
        a.per_category, a.blocks = min(a.per_category, 2), min(a.blocks, 2)
    return a


def plan_json(a) -> dict:
    bin_dir = e4_bonsai.llama_bin()
    order = sp.balanced_subset(make_prompts.load_longctx(a.prompts), a.seed, a.per_category)
    blocks = sp.prompt_blocks(order, a.blocks)
    cfgs = spec_configs()
    drafts = {"q8_0": "<draft_q8_0.gguf>"}
    return {"exp": EXP, "parts": a.parts, "budget_min": a.budget_min, "weights": WEIGHTS, "gate": list(mc.GPU600_TAGS),
            "model": hf_ids.resolve("bonsai2", "PQ2_0", offline=True), "tbo_cmd": mc.tbo_cmd(bin_dir), "tbo_order": [n for n, _ in TBO_ORDER],
            "bench_pp_cmd": mc.bench_pp_cmd(bin_dir, "<pq2.gguf>"), "bench_tg_cmd": mc.bench_tg_cmd(bin_dir, "<pq2.gguf>"),
            "bench_order": [[n for n, _ in mc.round_configs(KCONFIGS, r)] for r in range(a.rounds)],
            "spec": {"blocks": [[p["id"] for p in b] for b in blocks], "orders": sp.block_orders([c["name"] for c in cfgs], len(blocks)),
                     "configs": [{"name": c["name"], "env": c["env"], "cmd": sp.config_cmd(bin_dir, "<pq2.gguf>", drafts, c, ctx=a.ctx)} for c in cfgs]},
            "quick": a.quick}


def main(argv=None) -> int:
    a = parse_args(argv)
    if a.dry_run:
        common.print_plan(plan_json(a))
        return 0
    with common.term_exits():
        return run(a)


if __name__ == "__main__":
    sys.exit(main())
