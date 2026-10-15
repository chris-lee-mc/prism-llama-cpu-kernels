"""Wrap A (A1 + A2): evidence for the fork PR that enables the PQ2_0 multi-column kernel on sm_120, one 600 W PRO 6000 pod session.

  python -m exps.fetch_models --models pq2,draft_q8_0 --timeout-min 6 && python -m exps.wrap_a [--parts t,k,b,s] [--budget-min 14]

The kernel is the #306 mat-vec (mmvq-pq2_0.cuh) for 3-8 columns; the prototype switch GGML_CUDA_PQ2_MULTICOL=1 enables it on cc >= Ada
(default: Ada only) and the kernel calls ggml_cuda_pdl_sync() before it reads the activations. off = variable unset, on = set to 1.
Gate: the GPU tag must be pro6000-ws or pro6000-server (600 W) and the default power limit >= 550 W; a Max-Q or any other GPU gets no
timing (exit 1). --allow-any-gpu turns the gate into a note (development only).
Health: the 600 W card must also run at its default limit during the timing: power.limit equal to power.default_limit within 1%
(the bootstrap checks it first, CR_POWER_TOL, and stops a capped pod; data.power.bootstrap holds the pair it saw). PowerGuard reads both
values at the start, before and after each timed part (b, s) and before each server run of s, and records them in data.power.checks.
A drop before a timed part skips it (exit 1); a drop seen after it leaves the last round out (b) or marks the run (s: data.spec.power_problem).
Correctness (t) is never skipped for power. r1.host_probe / host_health run before part b and before part s (an unhealthy host skips the
part, exit 1). During part b every run is checked (clocks, power cap against the default within 1%); the first unhealthy round and all later
rounds are left out of the paired gains (data.bench_health), the verdict lines then start with UNHEALTHY HOST and the exit code is 1, so a
capped host cannot pass as PR evidence. --allow-power-cap turns the power checks into recorded values (development only).
t  correctness. test-backend-ops test -o MUL_MAT -b CUDA0 -p type_a=pq2_0, in the order on, off, on, on (a PDL race is timing dependent;
   off is second so a short budget cannot drop it; it gets at least 150 s; a missing off run is an INCOMPLETE note and exit 1). A failure
   with on stops the run before any timing.
k  kernel path. One nsys kernel-name list (r2b_multicol.path_probe: llama-bench -p 4 -n 0 -r 1 under nsys, private dir, report deleted)
   with the kernel on, then off: on must show mul_mat_vec_pq2_0_mc<4> and off must not. The same binary and variable as parts b and s. If on
   shows no mc launch, or off does, parts b and s are not run (KERNEL NOT ACTIVE, exit 1); without nsys the path is noted as unverified.
   Result: data.path with status (ok, bad, unverified).
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
Parts run in the order t, k, b, s with weights 2 : 1.5 : 6 : 4.5 of --budget-min (unused time rolls forward; t and k have floors of 150 and 120 s).
Result: results/wrap_a/<gpu>/<stamp>.json with data.gate, data.power, data.tbo, data.path, data.bench, data.bench_summary, data.bench_paired,
data.bench_health, data.spec, data.host, data.host_b.
Time: failsafe 40 min = bootstrap with the cold build ~15, downloads <= 6, wrap_a 14, reserve 5 (cost model: s about 4 min at the defaults).
Cost at the community price of about $1.69 per hour: pod time about 35 min, about $1.0; the secure price of $2.09 gives about $1.2.
"""
from __future__ import annotations

import argparse
import os
import sys

from exps import common, e4_bonsai, e4_pdl, hf_ids, make_prompts, r1, r2a_longctx as r2a, r2b_multicol as mc, r2b_spec as sp

EXP = "wrap_a"
ENV_VAR = mc.ENV_VAR
KCONFIGS = [("off", {}), ("on", {ENV_VAR: "1"})]
TBO_ORDER = [KCONFIGS[1], KCONFIGS[0], KCONFIGS[1], KCONFIGS[1]]     # on three times (a PDL race is timing dependent), off second: a short budget cannot drop it
PATH_ORDER = [KCONFIGS[1], KCONFIGS[0]]       # part k: the on probe first
ROUNDS = 4
BLOCKS = 2
PER_CATEGORY = 4
SPEC_NAMES = ["base", "k4q8_off", "k4q8_on"]
PARTS = "tkbs"
WEIGHTS = {"t": 2.0, "k": 1.5, "b": 6.0, "s": 4.5}
DEFAULT_TOTAL_MIN = 14.0
TBO_MIN_S = 150.0
PATH_MIN_S = 120.0
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


def kernel_status(path: dict | None) -> tuple[str, str]:
    """(ok | bad | unverified, why) from data.path: the multi-column kernel must launch with the variable set and never without it."""
    cfg = (path or {}).get("configs") or {}
    on, off = cfg.get("on") or {}, cfg.get("off") or {}
    if "mc_kernel_instances" not in on:
        return "unverified", on.get("error") or (path or {}).get("error") or "no kernel list of the on config"
    if on["mc_kernel_instances"] <= 0:
        return "bad", "no mul_mat_vec_pq2_0_mc launch with %s=1" % ENV_VAR
    if off.get("mc_kernel_instances", 0) > 0:
        return "bad", "mul_mat_vec_pq2_0_mc also launched with %s unset" % ENV_VAR
    return "ok", "mul_mat_vec_pq2_0_mc widths %s with the variable set%s" % (
        on.get("mc_widths"), "" if "mc_kernel_instances" in off else " (off not probed)")


def spec_summary(runs: dict, power_problem: dict | None = None) -> dict:
    out = {"vs_base": r2a.summarize(runs, ci=True), "on_vs_off": sp.pair_by_category(runs, "k4q8_on", "k4q8_off"),
           "step_split": {n: r.get("step_split") for n, r in runs.items()}}
    a = (out["on_vs_off"] or {}).get("all") or {}
    ms = {n: (out["step_split"].get(n) or {}).get("ms_per_step") for n in ("k4q8_on", "k4q8_off")}
    out["verdict"] = None if not a.get("speedup") else ("k4q8_on over k4q8_off %.3fx (95%% CI %s), greedy text equal %s/%s (informational: "
                                                         "the kernel changes the accumulation order), ms/step on %s vs off %s" % (
        a["speedup"], "n/a" if not a.get("speedup_ci95") else "[%.3f, %.3f]" % tuple(a["speedup_ci95"]), a.get("text_equal"), a.get("text_compared"),
        *("%.2f" % ms[n] if ms[n] else "n/a" for n in ("k4q8_on", "k4q8_off"))))
    if power_problem and out["verdict"]:
        out["verdict"] = "UNHEALTHY HOST (%s at %s; the run before that check and later ones are not valid): %s" % (
            power_problem["problem"], power_problem["at"], out["verdict"])
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
    guard = mc.PowerGuard(enforce=not args.allow_power_cap, bootstrap=mc.bootstrap_power())
    config = {"parts": args.parts, "budget_min": args.budget_min, "rounds": args.rounds, "blocks": args.blocks, "kernel_configs": dict(KCONFIGS),
              "spec_configs": [c["name"] for c in spec_configs()], "seed": args.seed, "per_category": args.per_category, "ctx": args.ctx,
              "prompt_ids": [p["id"] for p in order], "models": {}, "quick": args.quick}
    doc: dict = {"gate": None, "power": guard.doc, "tbo": [], "path": None, "bench": [], "bench_summary": {}, "bench_paired": {}, "bench_health": {}, "spec": None, "host": None,
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
    kernel_bad = None            # why the multi-column kernel is not shown active: no timing then
    bad = guard.check("start")
    if bad:
        notes.append(f"UNHEALTHY HOST: {bad} at the start; correctness still runs, no timing")
    try:
        for i, p in enumerate(parts):
            sub_s = part_budget_s(total.left(), p, parts[i:])
            if p == "t":
                sub_s = max(sub_s, min(TBO_MIN_S, total.left()))      # correctness gates the timings: never dropped for budget
            elif p == "k":
                sub_s = max(sub_s, min(PATH_MIN_S, total.left()))     # the kernel path is read before any timing
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
                    if not any(t.get("config") == "off" for t in doc["tbo"]):
                        notes.append("INCOMPLETE: test-backend-ops did not run with the kernel off (time budget): see data.tbo")
                        rc = 1
                elif p == "k":
                    doc["path"] = mc.path_probe(e4_bonsai.find_nsys(), bin_dir, target, sub, notes, configs=PATH_ORDER)
                    status, why = kernel_status(doc["path"])
                    doc["path"].update(status=status, why=why)
                    notes.append(f"k: kernel path {status}: {why}")
                    if status == "bad":
                        kernel_bad = why
                        notes.append(f"KERNEL NOT ACTIVE: {why}; parts b and s not run")
                        rc = 1
                elif kernel_bad:
                    notes.append(f"part {p} skipped: kernel not active ({kernel_bad})")
                elif p == "b":
                    pbad = guard.check("before b")
                    probe = r1.host_probe(bin_dir, target)
                    hbad, cap = r1.host_health(e4_pdl.smi_sample(), (probe or {}).get("smi"), check_power=not args.allow_power_cap)
                    doc["host_b"] = {"problem": hbad, "power_cap_note": cap, "probe": probe}
                    if pbad or hbad:
                        notes.append(f"UNHEALTHY HOST: {pbad or hbad}; part b skipped (A1 needs a healthy 600 W card at its default power limit)")
                        rc = 1
                    else:
                        mc.bench_part(bin_dir, target, sub, notes, rows, args.allow_power_cap, save, configs=KCONFIGS, rounds=args.rounds, tag=EXP,
                                      health=health, power_tol=None if args.allow_power_cap else mc.POWER_TOL)
                        pbad = guard.check("after b")
                        if pbad and health.get("first_bad_round") is None:
                            last = max([r["round"] for r in rows], default=0)
                            health.update(first_bad_round=last, problem=pbad, where="after part b")
                            notes.append(f"UNHEALTHY HOST: {pbad} after part b; round {last} and later left out")
                        if health.get("first_bad_round") is not None:
                            rc = 1
                else:
                    rc = max(rc, spec_part(args, bin_dir, target, blocks, sub_s, notes, doc, config, save, guard))
            except Exception as e:  # a failed part must not lose the others
                notes.append(f"part {p} crashed: {type(e).__name__}: {e}")
            cut["part"] = None
            save()
    finally:
        if cut["part"]:
            notes.append(f"run cut during part {cut['part']}: the results of that part are partial")
        if (doc.get("spec") or {}).get("runs"):
            try:
                doc["spec"]["summary"] = spec_summary(doc["spec"]["runs"], doc["spec"].get("power_problem"))
            except Exception as e:
                notes.append(f"final spec summary failed: {type(e).__name__}: {e}")
        paired = mc.paired_rounds(rows, "off", "on", before_round=health.get("first_bad_round"))
        notes.extend(bench_verdict(paired, health))
        if doc.get("host_b") and doc["host_b"]["problem"] or any(n.startswith("UNHEALTHY HOST") for n in notes):
            rc = max(rc, 1)
        if (doc.get("spec") or {}).get("summary", {}).get("verdict"):
            notes.append(doc["spec"]["summary"]["verdict"])
        notes.append(mc.power_line(doc["power"]))
        if doc["path"] is None and not kernel_bad:
            notes.append("kernel path unverified: part k did not run")
        notes.append(mc.PDL_LC_NOTE)
        save(final=True)
    return rc


def spec_part(args, bin_dir, target: str, blocks: list, budget_s: float, notes: list, doc: dict, config: dict, save, guard) -> int:
    try:
        draft, config["models"]["draft_q8_0"] = e4_pdl.model_path("draft_q8_0", notes)
    except Exception as e:
        notes.append(f"s: Q8_0 draft unavailable ({type(e).__name__}: {e}); part s not run")
        return 1
    t0 = common.Budget(budget_s / 60.0)
    pbad = guard.check("before s")
    probe = r1.host_probe(bin_dir, target)
    bad, cap = r1.host_health(e4_pdl.smi_sample(), (probe or {}).get("smi"), check_power=not args.allow_power_cap)
    doc["host"] = {"problem": bad, "power_cap_note": cap, "probe": probe}
    if pbad or bad:
        notes.append(f"UNHEALTHY HOST: {pbad or bad}; part s skipped")
        return 1
    cfgs = spec_configs()
    drafts = {"q8_0": draft}
    orders = sp.block_orders([c["name"] for c in cfgs], len(blocks))
    seen = {"n": 0, "checks": len(guard.doc["checks"])}

    def should_stop():
        seen["n"] += 1
        return guard.check("s run %d" % seen["n"])       # before each server run: a drop ends the timing

    doc["spec"] = sp.run_blocks(cfgs=cfgs, blocks=blocks, orders=orders, bin_dir=bin_dir, target=target,
                                cmd_for=lambda c: sp.config_cmd(bin_dir, target, drafts, c, ctx=args.ctx), budget_s=t0.left(), notes=notes,
                                gpu=args.gpu, exp=EXP, ctx=args.ctx, on_update=lambda s: (doc.__setitem__("spec", s), save()), tag=EXP,
                                should_stop=should_stop)
    guard.check("after s")
    bads = [c for c in guard.doc["checks"][seen["checks"]:] if c.get("problem")] if guard.enforce else []
    if bads:
        doc["spec"]["power_problem"] = {"at": bads[0]["at"], "problem": bads[0]["problem"]}
        notes.append("UNHEALTHY HOST: %s at %s; the run before that check and later ones are not valid" % (bads[0]["problem"], bads[0]["at"]))
        return 1
    return 0


# ---------------------------------------------------------------- CLI

def parse_args(argv=None):
    ap = argparse.ArgumentParser(prog="exps.wrap_a", description=__doc__.split("\n")[0])
    common.add_common_args(ap)
    ap.add_argument("--parts", type=lambda t: t.replace(",", ""), default=PARTS,
                    help="t (test-backend-ops), k (kernel path, nsys), b (llama-bench), s (spec run)")
    ap.add_argument("--rounds", type=int, default=ROUNDS)
    ap.add_argument("--blocks", type=int, default=BLOCKS)
    ap.add_argument("--per-category", type=int, default=PER_CATEGORY)
    ap.add_argument("--seed", type=int, default=sp.SEED)
    ap.add_argument("--ctx", type=int, default=sp.CTX)
    ap.add_argument("--prompts", default=str(make_prompts.R2A_OUT))
    ap.add_argument("--allow-power-cap", action="store_true", help="a power limit that is not the default is a recorded value, not a health problem (development only)")
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
            "power_tol": mc.POWER_TOL, "path_probe": {"cmd": mc.bench_pp_cmd(bin_dir, "<pq2.gguf>", [4], 1), "order": [n for n, _ in PATH_ORDER]},
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
