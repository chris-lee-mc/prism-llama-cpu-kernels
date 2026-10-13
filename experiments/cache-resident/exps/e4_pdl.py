"""E4 PDL A/B: decode throughput with PDL off, PDL on, and PDL on plus the pre-wait weight prefetch.

  python -m exps.e4_pdl [--models pq2,ptq1,draft_bf16,draft_q8_0] [--depths 0,32768] [--prefetch 1,4] [--reps 5]
                        [--rounds 2] [--no-tbo]

Configs (environment of each llama-bench process): pdl0 = GGML_CUDA_PDL=0, pdl1 = GGML_CUDA_PDL=1,
pf<N> = GGML_CUDA_PDL=1 GGML_CUDA_PDL_PREFETCH=N (mul_mat_vec_q in mmvq.cu issues prefetch.global.L2 of the
first N K iterations of its weight rows before cudaGridDependencySynchronize),
pdl0_pf<N> = GGML_CUDA_PDL=0 GGML_CUDA_PDL_PREFETCH=N for the largest N (control). The prefetch kernels are separate
instantiations, so their code generation (registers, occupancy) differs from the knob-off kernels even where nothing
overlaps. pdl0_pf<N> vs pdl0 is that codegen plus prefetch cost with no previous kernel to overlap; read the overlap
gain as pf<N> vs pdl1 next to pdl0_pf<N> vs pdl0. One llama-bench per
(model, config) covers all depths: -p 0 -n 128 -d 0,32768 -r 5, f16 KV, -fa on, -ub 2048.
Models: Bonsai 2 27B PQ2_0 and PTQ1_0 (targets) and the E5 draft Qwen3.5-0.8B alone, as Q8_0 (mmvq, prefetch active)
and BF16 (mul_mat_vec_f: the prefetch is not there, so pf rows are a control).
Per model, round 1 runs configs in order and round 2 in reverse (drift cancels in the median); a model that is cut
by the time budget is lost whole, and the summary uses only rounds in which every config finished.
Correctness first: test-backend-ops -o MUL_MAT,MUL_MAT_ID,MUL_MAT_VEC_FUSION -b CUDA0 on PQ2_0/PTQ1_0/Q8_0 cases vs
CPU, with the prefetch off and on. This shows results are unchanged (and that moving the wait broke nothing). It
cannot show that prefetch addresses are in range: prefetch.global.L2 returns no data and a bad address is ignored.
Prefetch kernels are separate template instantiations (pdl_pf), picked on the host when the knob is > 0; the knob-off
kernels have the same registers and shared memory as before the prefetch (nvcc -Xptxas -v, sm_86 and sm_120).
Not covered by the prefetch: more than one column (so PQ2_0 verify widths 3-8, mul_mat_vec_pq2_0_mc), all other quant
types, ids (MoE) mat-vecs, mul_mat_vec_ptq1_0_pt (GGML_CUDA_BATCH_INVARIANT=1 or no PDL; PTQ1_0 decode on sm_90+ runs
the y_soa kernel of mul_mat_vec_q), and the tail bytes of blocks that cross a 128 B line (first byte only).
Host health: nvidia-smi is sampled during every llama-bench call (SM/mem clocks, power draw and cap, temperature);
a degraded host (SM clock < 40% of max, mem clock < 50%, power cap < 75% of default) on the first call stops the run.
Needs CR_LLAMA_BIN with llama-bench and test-backend-ops built (CR_BUILD_LLAMA=1).
"""
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import threading

from exps import common, e4_bonsai, hf_ids

EXP = "e4_pdl"
MODELS = ("pq2", "ptq1", "draft_q8_0", "draft_bf16")
DEFAULT_DEPTHS = [0, 32768]
TBO_OPS = "MUL_MAT,MUL_MAT_ID,MUL_MAT_VEC_FUSION"  # the fused up/gate mat-vec (gate prefetch) is its own op
TBO_TYPES = "type(_a)?=(pq2_0|ptq1_0|q8_0),"  # MUL_MAT(_ID) vars start with type_a=, the fusion test with type=
TBO_PREFETCH = (0, 4)
SMI_FIELDS = ("clocks.sm", "clocks.max.sm", "clocks.mem", "clocks.max.mem", "power.draw", "power.limit",
              "power.default_limit", "temperature.gpu")
SMI_PERIOD_S = 2.0
HEALTH_MIN_SM, HEALTH_MIN_MEM, HEALTH_MIN_POWER_CAP = 0.4, 0.5, 0.75  # E3d host ran at SM 620 MHz, 350 W of 600 W


# ---------------------------------------------------------------- pure helpers

def configs(prefetch: list[int]) -> list[tuple[str, dict]]:
    out = [("pdl0", {"GGML_CUDA_PDL": "0"}), ("pdl1", {"GGML_CUDA_PDL": "1"})]
    for n in prefetch:
        out.append((f"pf{n}", {"GGML_CUDA_PDL": "1", "GGML_CUDA_PDL_PREFETCH": str(n)}))
    if prefetch:
        n = max(prefetch)
        out.append((f"pdl0_pf{n}", {"GGML_CUDA_PDL": "0", "GGML_CUDA_PDL_PREFETCH": str(n)}))
    return out


def bench_cmd(bin_dir, model: str, depths: list[int], reps: int, n_gen: int = e4_bonsai.N_GEN) -> list[str]:
    """llama-bench tg over all depths in one process. Flags as e4_bonsai.bench_cmd, f16 KV."""
    cmd = e4_bonsai.bench_cmd(bin_dir, model, 0, "f16", reps, n_gen)
    cmd[cmd.index("-d") + 1] = ",".join(str(d) for d in depths)
    return cmd


def tbo_cmd(bin_dir) -> list[str]:
    return [str(bin_dir / "test-backend-ops"), "test", "-o", TBO_OPS, "-b", "CUDA0", "-p", TBO_TYPES]


def parse_tbo(text: str) -> dict:
    """Sum the per-backend 'N/M tests passed' lines and collect FAIL case lines."""
    passed = total = 0
    for a, b in re.findall(r"(\d+)/(\d+) tests passed", text):
        passed, total = passed + int(a), total + int(b)
    fails = [ln.strip() for ln in text.splitlines() if "FAIL" in ln and "(" in ln][:20]  # FAIL is ANSI-colored
    return {"passed": passed, "total": total, "fail_lines": fails}


def tg_rows(entries: list[dict]) -> dict[int, dict]:
    """Generation entries keyed by depth: {depth: {tok_s, stddev}}."""
    out = {}
    for e in entries:
        if int(e.get("n_gen", 0) or 0) > 0:
            out[int(e.get("n_depth", 0) or 0)] = {"tok_s": float(e["avg_ts"]), "stddev": float(e.get("stddev_ts", 0.0) or 0.0)}
    return out


def round_order(cfgs: list, rnd: int) -> list:
    return list(cfgs) if rnd % 2 == 0 else list(reversed(cfgs))


def _median(xs):
    return common.median([x for x in xs if x])


def summarize(rows: list[dict], cfg_names: list[str]) -> dict:
    """Per model and depth: median tok/s per config and the gains pdl1 vs pdl0, pf<N> vs pdl1 and the control pdl0_pf<N> vs pdl0 (percent).

    Only rounds in which every config has a result count, so a budget cut cannot leave half a reverse-order pair.
    """
    cells: dict = {}  # (model, depth) -> {config: {round: [tok_s]}}
    for r in rows:
        if r.get("tok_s"):
            cells.setdefault((r["model"], str(r["depth"])), {}).setdefault(r["config"], {}).setdefault(
                r.get("round", 0), []).append(r["tok_s"])
    summ: dict = {}
    for (model, d), by_c in cells.items():
        done = [set(by_c.get(c, {})) for c in cfg_names]
        rounds = sorted(set.intersection(*done)) if done else []
        vals = {c: [x for rnd in rounds for x in by_c.get(c, {}).get(rnd, [])] for c in cfg_names}
        med = {c: _median(vals[c]) for c in cfg_names}
        gains = {}
        if rounds:
            if med.get("pdl0") and med.get("pdl1"):
                gains["pdl1_vs_pdl0_pct"] = 100.0 * (med["pdl1"] / med["pdl0"] - 1.0)
            for c in cfg_names:
                if c.startswith("pf") and med.get(c) and med.get("pdl1"):
                    gains[f"{c}_vs_pdl1_pct"] = 100.0 * (med[c] / med["pdl1"] - 1.0)
                if c.startswith("pdl0_pf") and med.get(c) and med.get("pdl0"):
                    gains[f"{c}_vs_pdl0_pct"] = 100.0 * (med[c] / med["pdl0"] - 1.0)
        summ.setdefault(model, {})[d] = {"tok_s": med, "n": {c: len(vals[c]) for c in cfg_names}, "rounds": rounds, **gains}
    return summ


def verdict_lines(summ: dict) -> list[str]:
    lines = []
    for model, by_d in summ.items():
        for d, s in by_d.items():
            parts = [f"{k}={v:+.1f}%" for k, v in s.items() if k.endswith("_pct")]
            lines.append(f"{model} depth {d}: " + (", ".join(parts) if parts else "no complete pair"))
    return lines


def parse_smi(text: str) -> dict | None:
    """First CSV line of nvidia-smi --query-gpu=SMI_FIELDS; '[N/A]' and junk become None."""
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if not lines:
        return None
    vals = [v.strip() for v in lines[0].split(",")]
    if len(vals) != len(SMI_FIELDS):
        return None
    out = {}
    for k, v in zip(SMI_FIELDS, vals):
        try:
            out[k] = float(v)
        except ValueError:
            out[k] = None
    return out


def smi_sample() -> dict | None:
    cmd = ["nvidia-smi", "--query-gpu=" + ",".join(SMI_FIELDS), "--format=csv,noheader,nounits", "-i", "0"]
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return None
    return parse_smi(p.stdout) if p.returncode == 0 else None


def smi_summary(before: dict | None, during: list[dict], after: dict | None) -> dict:
    """Clocks, power and temperature of one llama-bench call: samples taken during it plus one before and after."""
    pts = list(during) or [x for x in (before, after) if x]
    if not pts:
        return {}

    def col(k, f=_median):
        xs = [p[k] for p in pts if p.get(k) is not None]
        return f(xs) if xs else None
    last = pts[-1]
    return {"n": len(during), "sm_mhz_med": col("clocks.sm"), "sm_mhz_min": col("clocks.sm", min), "sm_mhz_max": last.get("clocks.max.sm"),
            "mem_mhz_med": col("clocks.mem"), "mem_mhz_max": last.get("clocks.max.mem"), "power_w_med": col("power.draw"),
            "power_w_max": col("power.draw", max), "power_cap_w": last.get("power.limit"),
            "power_default_w": last.get("power.default_limit"), "temp_c_max": col("temperature.gpu", max),
            "before": before, "after": after}


def health_problem(s: dict, check_power: bool = True) -> str | None:
    """Reason the host looks throttled or power capped, else None. Missing fields skip their check."""
    sm, sm_max, mem, mem_max = s.get("sm_mhz_med"), s.get("sm_mhz_max"), s.get("mem_mhz_med"), s.get("mem_mhz_max")
    cap, cap_def = s.get("power_cap_w"), s.get("power_default_w")
    if sm and sm_max and sm < HEALTH_MIN_SM * sm_max:
        return "SM clock %.0f MHz < %.0f%% of max %.0f" % (sm, 100 * HEALTH_MIN_SM, sm_max)
    if mem and mem_max and mem < HEALTH_MIN_MEM * mem_max:
        return "mem clock %.0f MHz < %.0f%% of max %.0f" % (mem, 100 * HEALTH_MIN_MEM, mem_max)
    if check_power and cap and cap_def and cap < HEALTH_MIN_POWER_CAP * cap_def:
        return "power cap %.0f W < %.0f%% of default %.0f W" % (cap, 100 * HEALTH_MIN_POWER_CAP, cap_def)
    return None


class SmiWatch:
    """Context manager: samples nvidia-smi every SMI_PERIOD_S while the body runs. No nvidia-smi gives an empty summary."""

    def __enter__(self):
        self.before, self.after, self.during = smi_sample(), None, []
        self._stop = threading.Event()
        self._t = threading.Thread(target=self._loop, daemon=True)
        self._t.start()
        return self

    def _loop(self):
        while not self._stop.wait(SMI_PERIOD_S):
            s = smi_sample()
            if s:
                self.during.append(s)

    def __exit__(self, *exc):
        self._stop.set()
        self._t.join(timeout=15)
        self.after = smi_sample()
        return False

    def summary(self) -> dict:
        return smi_summary(self.before, self.during, self.after)


# ---------------------------------------------------------------- running

def model_path(name: str, notes: list) -> tuple[str, dict]:
    from huggingface_hub import hf_hub_download
    if name in ("pq2", "ptq1"):
        return e4_bonsai.download("PQ2_0" if name == "pq2" else "PTQ1_0", notes)
    info = hf_ids.resolve("draft")
    if name == "draft_q8_0":
        info = dict(info, file=info["file"].replace("BF16", "Q8_0"))
    return hf_hub_download(info["repo"], info["file"]), info


def run_env(extra: dict) -> dict:
    env = {k: v for k, v in os.environ.items() if k not in ("GGML_CUDA_PDL", "GGML_CUDA_PDL_PREFETCH")}
    env.update(extra)
    return env


def run_tbo(bin_dir, budget: common.Budget, notes: list) -> list[dict]:
    out = []
    exe = bin_dir / "test-backend-ops"
    if not exe.exists():
        notes.append(f"{exe} not found; correctness check skipped")
        return out
    for n in TBO_PREFETCH:
        if budget.expired(600):
            notes.append(f"time budget: test-backend-ops prefetch={n} skipped")
            continue
        env = run_env({"GGML_CUDA_PDL": "1", "GGML_CUDA_PDL_PREFETCH": str(n)})
        row = {"prefetch": n, "cmd": tbo_cmd(bin_dir)}
        try:
            p = subprocess.run(row["cmd"], env=env, capture_output=True, text=True, timeout=budget.timeout(900, reserve_s=300))
            row.update(parse_tbo(p.stdout), returncode=p.returncode, ok=p.returncode == 0)
        except subprocess.TimeoutExpired:
            row.update(ok=False, error="timeout")
        out.append(row)
        print("[e4_pdl] test-backend-ops prefetch=%d ok=%s %s/%s" % (n, row.get("ok"), row.get("passed"), row.get("total")),
              flush=True)
    return out


def bench_once(cmd: list[str], env: dict, timeout) -> tuple[dict, str | None]:
    try:
        p = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return {}, "timeout"
    if p.returncode != 0:
        return {}, f"exit {p.returncode}: {(p.stderr or '')[-300:].strip()}"
    tg = tg_rows(e4_bonsai.parse_bench_json(p.stdout))
    return tg, None if tg else "no generation entry in llama-bench output"


def run(args) -> int:
    notes: list[str] = []
    args.gpu = common.gpu_tag(args)
    bin_dir = e4_bonsai.llama_bin()
    if not (bin_dir / "llama-bench").exists():
        print(f"error: {bin_dir / 'llama-bench'} not found; build llama.cpp (CR_BUILD_LLAMA=1) or set CR_LLAMA_BIN",
              file=sys.stderr)
        return 2
    budget = common.Budget(args.budget_min or (6 if args.quick else 55))
    cfgs = configs(args.prefetch)
    names = [c for c, _ in cfgs]
    config = {"models": args.models, "depths": args.depths, "reps": args.reps, "rounds": args.rounds, "configs": dict(cfgs),
              "n_gen": 32 if args.quick else e4_bonsai.N_GEN, "bench_cmd_example": bench_cmd(bin_dir, "<model.gguf>", args.depths, args.reps),
              "tbo_cmd": tbo_cmd(bin_dir), "quick": args.quick, "models_info": {}, "health_gate": {"sm": HEALTH_MIN_SM, "mem": HEALTH_MIN_MEM, "power_cap": HEALTH_MIN_POWER_CAP}}
    rows: list[dict] = []
    extra: dict = {"tbo": [], "summary": {}}

    def save():
        extra["summary"] = summarize(rows, names)
        common.write_partial(EXP, args, config, rows, notes, False, extra=extra)

    if not args.no_tbo:
        extra["tbo"] = run_tbo(bin_dir, budget, notes)
        save()
    paths = {}
    for m in args.models:
        try:
            paths[m], config["models_info"][m] = model_path(m, notes)
        except Exception as e:
            notes.append(f"{m}: model download failed: {type(e).__name__}: {e}")
    n_bench, unhealthy, aborted = 0, False, False
    for m in args.models:  # both rounds of a model back to back: a budget cut drops whole models, not half pairs
        if m not in paths or aborted:
            continue
        for rnd in range(args.rounds):
            for cname, cenv in round_order(cfgs, rnd):
                if aborted:
                    break
                if budget.expired(120):
                    notes.append(f"time budget used up; round {rnd} {m} {cname} and later skipped")
                    continue
                cmd = bench_cmd(bin_dir, paths[m], args.depths, args.reps, config["n_gen"])
                with SmiWatch() as watch:
                    tg, err = bench_once(cmd, run_env(cenv), budget.timeout(1800, reserve_s=60))
                smi = watch.summary()
                bad = health_problem(smi, check_power=not args.allow_power_cap)
                capped = args.allow_power_cap and health_problem(smi) and not bad
                if capped and not any(n.startswith("power-capped host") for n in notes):
                    notes.append(f"power-capped host accepted (--allow-power-cap): {health_problem(smi)}; configs compare within this host only")
                n_bench += 1
                if bad and n_bench == 1:
                    notes.append(f"ABORT: unhealthy host: {bad}")
                    aborted = True
                elif bad and not unhealthy:
                    unhealthy = True
                    notes.append(f"host health: {bad} during round {rnd} {m} {cname}; timings after this may be invalid")
                if err:
                    notes.append(f"round {rnd} {m} {cname} bench failed: {err}")
                for d in args.depths:
                    r = tg.get(d)
                    rows.append({"model": m, "depth": d, "config": cname, "env": cenv, "round": rnd,
                                 "tok_s": r["tok_s"] if r else None, "stddev": r["stddev"] if r else None,
                                 "us_per_token": 1e6 / r["tok_s"] if r and r["tok_s"] else None, "error": None if r else err,
                                 "smi": smi})
                save()
                print("[e4_pdl] r%d %s %s %s smi sm %s MHz" % (rnd, m, cname, {d: (tg.get(d) or {}).get("tok_s") for d in args.depths},
                                                               smi.get("sm_mhz_med")), flush=True)
    extra["summary"] = summarize(rows, names)
    notes.extend(verdict_lines(extra["summary"]))
    notes.append("draft_bf16 runs mul_mat_vec_f, which has no pre-wait prefetch: its pf rows are a control")
    notes.append("pdl0_pf<N> vs pdl0 = codegen plus prefetch cost without overlap; compare it with pf<N> vs pdl1 for the overlap gain")
    notes.append("test-backend-ops checks that results are unchanged, not that prefetch addresses are in range")
    notes.append("not covered by the prefetch: more than one column (PQ2_0 verify widths 3-8), ids mat-vecs, other quant types")
    if extra["tbo"] and not all(t.get("ok") for t in extra["tbo"]):
        notes.append("test-backend-ops reported failures; compare prefetch=0 and prefetch=4 rows before reading timings")
    common.write_result(EXP, args, config, rows, notes, False, extra=extra)
    return 0 if any(r.get("tok_s") for r in rows) and not aborted else 1


# ---------------------------------------------------------------- CLI

def _ints(text):
    return [int(t) for t in text.split(",") if t.strip() != ""]


def _models(text):
    out = [t.strip() for t in text.split(",") if t.strip()]
    bad = [m for m in out if m not in MODELS]
    if bad:
        raise argparse.ArgumentTypeError(f"unknown model(s) {bad}; choose from {MODELS}")
    return out


def parse_args(argv=None):
    ap = argparse.ArgumentParser(prog="exps.e4_pdl", description=__doc__.split("\n")[0])
    common.add_common_args(ap)
    ap.add_argument("--models", type=_models, default=list(MODELS))
    ap.add_argument("--depths", type=_ints, default=list(DEFAULT_DEPTHS))
    ap.add_argument("--prefetch", type=_ints, default=[1, 4], help="GGML_CUDA_PDL_PREFETCH values, one config each")
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--no-tbo", action="store_true", help="skip the test-backend-ops correctness check")
    ap.add_argument("--allow-power-cap", action="store_true", help="a power cap below default is a note, not an abort (clock checks stay)")
    a = ap.parse_args(argv)
    if any(n <= 0 for n in a.prefetch):
        ap.error("--prefetch values must be > 0 (pdl1 is the prefetch-off config)")
    if a.reps < 5 and not a.quick:
        ap.error("--reps must be >= 5")
    if a.quick:
        a.models, a.depths, a.reps, a.rounds, a.prefetch = ["draft_q8_0"], [0], 2, 1, a.prefetch[:1]
    return a


def plan_json(a) -> dict:
    bin_dir = e4_bonsai.llama_bin()
    cfgs = configs(a.prefetch)
    return {"exp": EXP, "models": a.models, "depths": a.depths, "reps": a.reps, "rounds": a.rounds,
            "configs": dict(cfgs), "order": [[c for c, _ in round_order(cfgs, r)] for r in range(a.rounds)],
            "bench_cmd": bench_cmd(bin_dir, "<model.gguf>", a.depths, a.reps),
            "tbo": None if a.no_tbo else {"cmd": tbo_cmd(bin_dir), "prefetch": list(TBO_PREFETCH)},
            "n_bench": a.rounds * len(a.models) * len(cfgs), "quick": a.quick}


def main(argv=None) -> int:
    a = parse_args(argv)
    if a.dry_run:
        common.print_plan(plan_json(a))
        return 0
    with common.term_exits():
        return run(a)


if __name__ == "__main__":
    sys.exit(main())
