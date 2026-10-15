"""Wrap A3: Ada regression check for the PQ2_0 multi-column PR change (pdl_sync, restrict off the parameters), one RTX 4090 pod session.

  python -m exps.wrap_a3 [--parts build,sass,t,b,f] [--budget-min 28]

On Ada the #306 kernel is on by default with or without the change, and PDL is compiled out below sm_90, so no change is expected. Two
binaries on the same pod, same cmake flags (-DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=<cc> -DCMAKE_BUILD_TYPE=Release -DLLAMA_CURL=OFF,
ccache):
  head  the bootstrap build of this branch (CR_LLAMA_BIN, build-cuda/bin)
  base  a copy of this tree with exps/data/a3_pq2_multicol.patch reversed: common.cuh and mmvq-pq2_0.cuh as in 4a4866a60, the commit
        before the multicol / pdl_sync change. Only those two files differ between the two builds. Only llama-bench and
        test-backend-ops are built (ccache with CCACHE_BASEDIR reuses the CPU objects of the head build). The model download runs while
        the base build runs.
Gate: the GPU tag must be rtx4090 (Ada, cc 8.9); --allow-any-gpu for development.
Health: PR evidence, so the card must run at its default power limit: power.limit equal to power.default_limit within 1% (the bootstrap
checks it first, CR_POWER_TOL; data.power.bootstrap). PowerGuard records both values at the start and before and after each timed part
(b, f) in data.power.checks; a drop before a part skips it (exit 1), during part b the first unhealthy round and later ones are left out
of the paired gains (data.bench_health), a drop seen after part f marks data.perf.power_problem. Parts build, sass and t are never
skipped for power. --allow-power-cap turns the power checks into recorded values (development only).
The pod query asks for 16 vCPUs when the llama build is on (infra/runpod.py), the base build gets 14 of 28 min (weight 14 of 28); a build
that does not finish ends the run with exit 1 and no timing.
sass  cuobjdump -sass of mul_mat_vec_pq2_0_mc from both builds (sm_89), instruction text without offsets and encodings. On sm_89 the
   change is a no-op by construction (ggml_cuda_pdl_sync is compiled out below sm_90, GGML_CUDA_RESTRICT stays __restrict__), so the
   kernels must be identical; any difference is reported before the timings are read. No cuobjdump: a note, the timings still run.
t  test-backend-ops test -o MUL_MAT -b CUDA0 -p type_a=pq2_0, with head, then base (environment without GGML_CUDA_PQ2_MULTICOL). A head
   failure stops the run before any timing.
b  llama-bench on Bonsai 2 27B PQ2_0 (flags of r2b_multicol): -p 1,...,9 -n 0 -r 5 and -p 0 -n 128 -r 5, base and head alternating,
   --rounds rounds (default 4). Reported: ms per width per build, the gain of head over base per round, and its median / min / max.
f  test-backend-ops perf -o MUL_MAT -b CUDA0 -p type_a=pq2_0, (us per op, widths 1-32 at the 27B shapes), base and head.
Result: results/wrap_a3/<gpu>/<stamp>.json with data.gate, data.power, data.build, data.tbo, data.bench, data.bench_summary, data.bench_paired,
data.bench_health, data.perf.
Time: failsafe 55 min = bootstrap with the head build ~15, wrap_a3 28 (base build up to 14, download overlapped, runs ~10), reserve 5.
Cost at the 4090 community price of about $0.34 per hour: about 50 min of pod time, about $0.3.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import pathlib
import re
import shutil
import subprocess
import sys
import tempfile
import time

from exps import common, e4_bonsai, e4_pdl, hf_ids, r1, r2b_multicol as mc

EXP = "wrap_a3"
REPO_ROOT = common.EXP_DIR.parent.parent
PATCH = common.EXP_DIR / "exps" / "data" / "a3_pq2_multicol.patch"
PATCH_FILES = ("ggml/src/ggml-cuda/common.cuh", "ggml/src/ggml-cuda/mmvq-pq2_0.cuh")
BASE_COMMIT = "4a4866a60778437aec748e5ad0c45439b14fd1db"
TARGETS = ("llama-bench", "test-backend-ops")
TOP_IGNORE = (".git", "experiments", "models")        # top level only, with every name that starts with build (build dirs)
ANY_IGNORE = ("__pycache__", "node_modules", ".venv")
BUILDS = [("base", {}), ("head", {})]
ROUNDS = 4
PARTS = ("build", "sass", "t", "b", "f")
WEIGHTS = {"build": 14.0, "sass": 1.0, "t": 2.0, "b": 8.0, "f": 3.0}
DEFAULT_TOTAL_MIN = 28.0
SASS_KERNEL = "mul_mat_vec_pq2_0_mc"
END_RESERVE_S = 25.0
ADA_TAGS = ("rtx4090",)


def cmake_cmd(src, build, cc: str, ccache: bool) -> list[str]:
    """The cmake flags of infra/bootstrap.sh (head build), so the two builds differ only in the patched files."""
    cmd = ["cmake", "-S", str(src), "-B", str(build), "-G", "Ninja", "-DGGML_CUDA=ON", f"-DCMAKE_CUDA_ARCHITECTURES={cc}",
           "-DCMAKE_BUILD_TYPE=Release", "-DLLAMA_CURL=OFF"]
    if ccache:
        cmd += ["-DCMAKE_CUDA_COMPILER_LAUNCHER=ccache", "-DCMAKE_C_COMPILER_LAUNCHER=ccache", "-DCMAKE_CXX_COMPILER_LAUNCHER=ccache"]
    return cmd


def compute_cap() -> str | None:
    try:
        p = subprocess.run(["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"], capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return None
    v = (p.stdout or "").strip().splitlines()
    return v[0].replace(".", "").strip() if p.returncode == 0 and v else None


def head_cache_value(bin_dir: pathlib.Path, key: str) -> str | None:
    """A value of CMakeCache.txt next to the head binaries (build-cuda/CMakeCache.txt)."""
    p = pathlib.Path(bin_dir).parent / "CMakeCache.txt"
    try:
        for ln in p.read_text(errors="replace").splitlines():
            if ln.startswith(key + ":"):
                return ln.split("=", 1)[1]
    except OSError:
        pass
    return None


def sha(p: pathlib.Path) -> str | None:
    try:
        return hashlib.sha256(p.read_bytes()).hexdigest()
    except OSError:
        return None


def copy_ignore(d: str, names: list[str]) -> set[str]:
    """shutil.copytree ignore: build dirs and the big dirs at the top only (cmake/build-info.cmake must stay), caches everywhere."""
    out = {n for n in names if n in ANY_IGNORE or n.endswith(".gguf")}
    if pathlib.Path(d).resolve() == REPO_ROOT.resolve():
        out |= {n for n in names if n in TOP_IGNORE or n.startswith("build")}
    return out


def make_base_tree(dst: pathlib.Path, notes: list) -> dict:
    """Copy of the source tree (no .git, builds, models, experiments) with the PR patch reversed. Returns facts for the result."""
    out: dict = {"dir": str(dst), "patch": os.path.relpath(PATCH, common.EXP_DIR), "base_commit": BASE_COMMIT}
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(REPO_ROOT, dst, ignore=copy_ignore, symlinks=True)
    # the ceiling keeps git from finding an enclosing repo, so the paths are relative to dst
    p = subprocess.run(["git", "apply", "-R", "--whitespace=nowarn", str(PATCH)], cwd=dst, capture_output=True, text=True, timeout=60,
                       env=dict(os.environ, GIT_CEILING_DIRECTORIES=str(dst.parent)))
    out["applied"] = p.returncode == 0
    if not out["applied"]:
        out["error"] = (p.stderr or p.stdout)[-400:]
        notes.append(f"build: the reverse patch does not apply: {out['error']}")
    out["files"] = {f: {"head": sha(REPO_ROOT / f), "base": sha(dst / f)} for f in PATCH_FILES}
    out["differs"] = all(v["head"] != v["base"] for v in out["files"].values())
    return out


def start_build(src: pathlib.Path, cc: str, log_path: pathlib.Path) -> tuple[subprocess.Popen | None, list[str], str | None]:
    """Configure (blocking) and start the build of TARGETS (background). Returns (proc, build cmd, error)."""
    build = src / "build-cuda"
    env = dict(os.environ)
    ccache = shutil.which("ccache") is not None
    if pathlib.Path("/workspace").is_dir():
        env.setdefault("CCACHE_DIR", "/workspace/.ccache")      # the bootstrap cache of the head build
    env.update(CCACHE_BASEDIR=os.path.commonpath([str(REPO_ROOT.resolve()), str(src.resolve())]), CCACHE_NOHASHDIR="1")    # as bootstrap.sh
    log = open(log_path, "w")
    p = subprocess.run(cmake_cmd(src, build, cc, ccache), env=env, stdout=log, stderr=subprocess.STDOUT, timeout=600)
    if p.returncode != 0:
        log.close()
        return None, [], f"cmake configure exit {p.returncode}"
    cmd = ["cmake", "--build", str(build), "-j", str(os.cpu_count() or 8), "--target", *TARGETS]
    return subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT), cmd, None


def build_part(args, budget: common.Budget, notes: list, doc: dict, config: dict, head_bin: pathlib.Path) -> tuple[pathlib.Path | None, str | None]:
    """Base tree, base build, and the model download while it builds. Returns (base bin dir, model path)."""
    from infra.results import artifact_path
    cc = compute_cap() or args.cc
    work = pathlib.Path(args.work or (tempfile.mkdtemp(prefix="a3-") if not pathlib.Path("/workspace").is_dir() else "/workspace/a3"))
    info = make_base_tree(work / "src", notes)
    info.update(cc=cc, head_cuda_arch=head_cache_value(head_bin, "CMAKE_CUDA_ARCHITECTURES"),
                head_build_type=head_cache_value(head_bin, "CMAKE_BUILD_TYPE"), cmake=cmake_cmd("<src>", "<src>/build-cuda", cc, True))
    doc["build"] = info
    if not info["applied"] or not info["differs"]:
        notes.append("build: the base tree is not the pre-change state; no base build")
        return None, None
    if info["head_cuda_arch"] and info["head_cuda_arch"] != cc:
        notes.append(f"build: head was built for {info['head_cuda_arch']}, base for {cc}")
    log = artifact_path(EXP, "build-base.log", args.gpu)
    t0 = time.monotonic()
    proc, cmd, err = start_build(work / "src", cc, log)
    info["build_cmd"] = cmd
    model = None
    try:
        model, config["models"]["pq2"] = e4_pdl.model_path("pq2", notes)
    except Exception as e:
        notes.append(f"model download failed: {type(e).__name__}: {e}")
    if err or proc is None:
        info["error"] = err
        notes.append(f"build: {err}")
        return None, model
    try:
        code = proc.wait(timeout=max(budget.left() - 10, 1))
    except subprocess.TimeoutExpired:
        proc.kill()
        info["error"] = "build timeout"
        notes.append("build: base build did not finish in its budget")
        return None, model
    info["build_s"] = time.monotonic() - t0
    info["returncode"] = code
    base_bin = work / "src" / "build-cuda" / "bin"
    if code != 0 or not all((base_bin / t).exists() for t in TARGETS):
        info["error"] = f"build exit {code}"
        try:
            info["log_tail"] = r1.scrub(log.read_text(errors="replace")[-2000:], r1.secret_values())
        except OSError:
            pass
        notes.append(f"build: base build failed (exit {code}); see data.build.log_tail")
        return None, model
    return base_bin, model


FUNC_RE = re.compile(r"^\s*Function : (\S+)")
SASS_OFFSET_RE = re.compile(r"^\s*/\*[0-9a-f]{4}\*/\s*")
SASS_ENC_RE = re.compile(r"\s*/\*\s*0x[0-9a-f]+\s*\*/\s*$")


def sass_kernels(lines, pat: str = SASS_KERNEL) -> dict:
    """{function name: [instruction text]} from cuobjdump -sass lines, for the functions whose name contains pat. The instruction text
    has neither the offset nor the encoding comment, so two builds compare by their instructions."""
    out: dict = {}
    cur = None
    for ln in lines:
        m = FUNC_RE.match(ln)
        if m:
            cur = out.setdefault(m.group(1), []) if pat in m.group(1) else None
            continue
        if cur is None or not SASS_OFFSET_RE.match(ln):
            continue
        txt = SASS_ENC_RE.sub("", SASS_OFFSET_RE.sub("", ln.rstrip("\n"))).strip()
        if txt:
            cur.append(txt)
    return out


def sass_compare(head: dict, base: dict) -> dict:
    """Same function names and the same instructions in both builds."""
    diff = sorted(n for n in set(head) | set(base) if head.get(n) != base.get(n))
    return {"n_kernels": {"head": len(head), "base": len(base)}, "n_instructions": {"head": sum(map(len, head.values())),
                                                                                  "base": sum(map(len, base.values()))},
            "different": diff, "identical": bool(head) and head.keys() == base.keys() and not diff}


def find_cuobjdump() -> str | None:
    return shutil.which("cuobjdump") or next((c for c in ("/usr/local/cuda/bin/cuobjdump", "/usr/local/cuda-12.8/bin/cuobjdump")
                                              if os.path.exists(c)), None)


def sass_of(bin_dir, tool: str, arch: str, timeout: float) -> dict:
    """Kernels of SASS_KERNEL in libggml-cuda (or the executable when the build is static) of one build. Streams the dump."""
    libs = sorted(pathlib.Path(bin_dir).glob("libggml-cuda.so*")) or [pathlib.Path(bin_dir) / "llama-bench"]
    p = subprocess.Popen([tool, "-sass", "-arch", f"sm_{arch}", str(libs[0])], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True,
                         errors="replace")
    try:
        out = sass_kernels(p.stdout)
        p.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        p.kill()
        raise
    return out


def sass_part(bins: dict, cc: str, budget: common.Budget, notes: list) -> dict:
    out: dict = {"kernel": SASS_KERNEL, "arch": f"sm_{cc}"}
    tool = find_cuobjdump()
    if not tool:
        out["error"] = "cuobjdump not found"
        notes.append("sass: cuobjdump not found; the kernel comparison is skipped (the timings still run)")
        return out
    out["tool"] = os.path.basename(tool)
    try:
        k = {n: sass_of(bins[n], tool, cc, budget.timeout(240, reserve_s=10)) for n in ("head", "base")}
    except Exception as e:
        out["error"] = f"{type(e).__name__}: {e}"
        notes.append(f"sass: {out['error']}")
        return out
    out.update(sass_compare(k["head"], k["base"]))
    notes.append("sass %s %s: %s (%s instructions in %s kernels)" % (
        SASS_KERNEL, out["arch"], "identical in both builds" if out["identical"] else "NOT identical or not found: %s" % out["different"][:3],
        out["n_instructions"], out["n_kernels"]))
    return out


def bench_verdict(paired: dict, health: dict | None = None) -> list[str]:
    """Paired gains per width. After an unhealthy host every line starts with UNHEALTHY HOST and says which rounds were left out."""
    pre = ""
    if health and health.get("first_bad_round") is not None:
        pre = "UNHEALTHY HOST (%s during round %d %s; that round and later ones left out): " % (
            health["problem"], health["first_bad_round"], health.get("where", ""))
    out = []
    for w, e in paired.get("pp", {}).items():
        out.append("%swidth %s: head over base %+.1f%% (rounds %s, min %+.1f%%, max %+.1f%%)" % (pre, w, e["median"], e["n"], e["min"], e["max"]))
    tg = paired.get("tg", {}).get("1")
    if tg:
        out.append("%sdecode tg128: head over base %+.1f%% (min %+.1f%%, max %+.1f%%)" % (pre, tg["median"], tg["min"], tg["max"]))
    if pre and not out:
        out.append(pre + "no healthy round: no paired gain")
    return out


def part_budget_s(left_s: float, part: str, remaining: list[str]) -> float:
    tot = sum(WEIGHTS[p] for p in remaining) or 1.0
    return max(left_s - END_RESERVE_S, 0.0) * WEIGHTS[part] / tot


def run(args) -> int:
    notes: list[str] = []
    args.gpu = common.gpu_tag(args)
    os.environ.pop(mc.ENV_VAR, None)
    head_bin = e4_bonsai.llama_bin()
    total = common.Budget(args.budget_min)
    secrets = r1.secret_values()
    config = {"parts": args.parts, "budget_min": args.budget_min, "rounds": args.rounds, "builds": [b for b, _ in BUILDS], "models": {},
              "targets": TARGETS, "head_bin": str(head_bin), "quick": args.quick}
    guard = mc.PowerGuard(enforce=not args.allow_power_cap, bootstrap=mc.bootstrap_power())
    doc: dict = {"gate": None, "power": guard.doc, "build": None, "sass": None, "tbo": [], "bench": [], "bench_summary": {}, "bench_paired": {},
                 "bench_health": {}, "perf": None}
    rows: list[dict] = doc["bench"]
    health: dict = doc["bench_health"]          # first_bad_round / problem of part b; rounds from there on are left out of the gains

    def save(final: bool = False) -> None:
        doc["bench_summary"] = mc.summarize_bench(rows, [b for b, _ in BUILDS], ref="base", before_round=health.get("first_bad_round"))
        doc["bench_paired"] = mc.paired_rounds(rows, "base", "head", before_round=health.get("first_bad_round"))
        (common.write_result if final else common.write_partial)(EXP, args, r1.scrub(config, secrets), [], r1.scrub(notes, secrets), False,
                                                                 extra=r1.scrub(doc, secrets))

    missing = [t for t in TARGETS if not (head_bin / t).exists()]
    if missing:
        notes.append(f"{missing} not found in {head_bin}: build llama.cpp (CR_BUILD_LLAMA=1)")
        save(final=True)
        return 2
    bad = None if args.gpu.split("-")[0] in ADA_TAGS else f"GPU tag {args.gpu} is not an RTX 4090 (Ada)"
    doc["gate"] = {"gpu": args.gpu, "problem": bad, "cc": compute_cap(), "allow_any_gpu": args.allow_any_gpu}
    if bad and not args.allow_any_gpu:
        notes.append(f"gate: {bad}; no run")
        save(final=True)
        return 1
    parts = [p for p in PARTS if p in args.parts]
    bins = {"head": head_bin}
    model = None
    rc = 0
    cut = {"part": None}
    bad = guard.check("start")
    if bad:
        notes.append(f"UNHEALTHY HOST: {bad} at the start; build, sass and correctness still run, no timing")
    try:
        for i, p in enumerate(parts):
            sub_s = part_budget_s(total.left(), p, parts[i:])
            if p == "t":
                sub_s = max(sub_s, min(90.0, total.left()))      # correctness gates the timings: never dropped for budget
            elif p == "sass":
                sub_s = max(sub_s, min(120.0, total.left()))     # one cuobjdump pass per build, read before any timing
            if sub_s < 60:
                notes.append(f"part {p} skipped: only {sub_s:.0f} s of budget left")
                continue
            sub = common.Budget(sub_s / 60.0)
            cut["part"] = p
            print(f"[{EXP}] part {p}: {sub_s / 60:.1f} min of {total.left() / 60:.1f} min left", flush=True)
            try:
                if p == "build":
                    base_bin, model = build_part(args, sub, notes, doc, config, head_bin)
                    if not base_bin:
                        rc = 1
                        cut["part"] = None
                        break
                    bins["base"] = base_bin
                elif "base" not in bins:
                    notes.append(f"part {p} skipped: no base build (run part build)")
                    rc = 1
                elif p == "sass":
                    doc["sass"] = sass_part(bins, compute_cap() or args.cc, sub, notes)
                elif p == "t":
                    for name in ("head", "base"):
                        row = mc.run_tbo(mc.tbo_cmd(bins[name]), mc.run_env({}), sub.timeout(600, reserve_s=30))
                        row.update(config=name)
                        doc["tbo"].append(row)
                        print("[%s] test-backend-ops %s ok=%s %s/%s" % (EXP, name, row.get("ok"), row.get("passed"), row.get("total")), flush=True)
                    if not mc.tbo_passed(doc["tbo"][0]):
                        notes.append("stop: test-backend-ops did not pass with the head build; parts b and f not run")
                        rc = 1
                        cut["part"] = None
                        break
                    if not all(mc.tbo_passed(t) for t in doc["tbo"]):
                        notes.append("test-backend-ops failed with the base build: see data.tbo")
                elif p == "b":
                    if model is None:
                        model, config["models"]["pq2"] = e4_pdl.model_path("pq2", notes)
                    pbad = guard.check("before b")
                    if pbad:
                        notes.append(f"UNHEALTHY HOST: {pbad}; part b skipped (the timings need the default power limit)")
                        rc = 1
                    else:
                        mc.bench_part(head_bin, model, sub, notes, rows, args.allow_power_cap, save, configs=BUILDS, rounds=args.rounds, bins=bins,
                                      tag=EXP, health=health, power_tol=None if args.allow_power_cap else mc.POWER_TOL)
                        pbad = guard.check("after b")
                        if pbad and health.get("first_bad_round") is None:
                            last = max([r["round"] for r in rows], default=0)
                            health.update(first_bad_round=last, problem=pbad, where="after part b")
                            notes.append(f"UNHEALTHY HOST: {pbad} after part b; round {last} and later left out")
                        if health.get("first_bad_round") is not None:
                            rc = 1
                else:
                    pbad = guard.check("before f")
                    if pbad:
                        notes.append(f"UNHEALTHY HOST: {pbad}; part f skipped (the timings need the default power limit)")
                        rc = 1
                    else:
                        doc["perf"] = perf_part(bins, sub, notes)
                        pbad = guard.check("after f")
                        if pbad:
                            doc["perf"]["power_problem"] = {"at": "after f", "problem": pbad}
                            notes.append(f"UNHEALTHY HOST: {pbad} after part f; the perf timings are not valid")
                            rc = 1
            except Exception as e:  # a failed part must not lose the others
                notes.append(f"part {p} crashed: {type(e).__name__}: {e}")
            cut["part"] = None
            save()
    finally:
        if cut["part"]:
            notes.append(f"run cut during part {cut['part']}: the results of that part are partial")
        notes.extend(bench_verdict(mc.paired_rounds(rows, "base", "head", before_round=health.get("first_bad_round")), health))
        notes.append(mc.power_line(doc["power"]))
        notes.append("Ada: the multi-column kernel runs in both builds (default on Ada); ggml_cuda_pdl_sync() is compiled out below sm_90")
        save(final=True)
    return rc


def perf_part(bins: dict, budget: common.Budget, notes: list) -> dict:
    out: dict = {"configs": {}, "gain": []}
    for name in ("base", "head"):
        if budget.expired(60):
            notes.append(f"f: time budget used up, {name} skipped")
            continue
        cmd = mc.tbo_cmd(bins[name], "perf")
        try:
            p = subprocess.run(cmd, env=mc.run_env({}), capture_output=True, text=True, timeout=budget.timeout(600, reserve_s=30))
        except subprocess.TimeoutExpired:
            out["configs"][name] = {"cmd": cmd, "error": "timeout", "rows": []}
            continue
        out["configs"][name] = {"cmd": cmd, "returncode": p.returncode, "rows": mc.parse_perf(p.stdout)}
    if out["configs"].get("base", {}).get("rows") and out["configs"].get("head", {}).get("rows"):
        out["gain"] = mc.perf_gain(out["configs"]["base"]["rows"], out["configs"]["head"]["rows"])    # gain_pct: head over base
    return out


# ---------------------------------------------------------------- CLI

def _parts(text):
    ps = [t.strip() for t in text.split(",") if t.strip()]
    bad = [p for p in ps if p not in PARTS]
    if bad:
        raise argparse.ArgumentTypeError(f"unknown part(s) {bad}; parts: {','.join(PARTS)}")
    return ps


def parse_args(argv=None):
    ap = argparse.ArgumentParser(prog="exps.wrap_a3", description=__doc__.split("\n")[0])
    common.add_common_args(ap)
    ap.add_argument("--parts", type=_parts, default=list(PARTS), help="build, sass (kernel SASS of both builds), t (test-backend-ops), b (llama-bench), f (op timing)")
    ap.add_argument("--rounds", type=int, default=ROUNDS)
    ap.add_argument("--work", default=None, help="dir for the base tree and build (default /workspace/a3 on a pod, else a temp dir)")
    ap.add_argument("--cc", default="89", help="CUDA arch when nvidia-smi does not report one")
    ap.add_argument("--allow-power-cap", action="store_true", help="a power cap below default is a note, not a health problem")
    ap.add_argument("--allow-any-gpu", action="store_true", help="run on a host that is not an RTX 4090 (development only)")
    a = ap.parse_args(argv)
    if a.rounds < 3:
        ap.error("--rounds must be at least 3")
    a.budget_min = a.budget_min or (12.0 if a.quick else DEFAULT_TOTAL_MIN)
    return a


def plan_json(a) -> dict:
    bin_dir = e4_bonsai.llama_bin()
    return {"exp": EXP, "parts": a.parts, "budget_min": a.budget_min, "weights": WEIGHTS, "gate": list(ADA_TAGS), "base_commit": BASE_COMMIT,
            "patch": os.path.relpath(PATCH, common.EXP_DIR), "patch_files": PATCH_FILES, "targets": TARGETS,
            "cmake": cmake_cmd("<src>", "<src>/build-cuda", a.cc, True), "model": hf_ids.resolve("bonsai2", "PQ2_0", offline=True),
            "tbo_cmd": mc.tbo_cmd(bin_dir), "bench_pp_cmd": mc.bench_pp_cmd(bin_dir, "<pq2.gguf>"), "bench_tg_cmd": mc.bench_tg_cmd(bin_dir, "<pq2.gguf>"),
            "bench_order": [[n for n, _ in mc.round_configs(BUILDS, r)] for r in range(a.rounds)], "power_tol": mc.POWER_TOL, "quick": a.quick}


def main(argv=None) -> int:
    a = parse_args(argv)
    if a.dry_run:
        common.print_plan(plan_json(a))
        return 0
    with common.term_exits():
        return run(a)


if __name__ == "__main__":
    sys.exit(main())
