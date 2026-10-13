"""Nsight Compute wrapper (PLAN P0.3): one call returns per-kernel L2 hit rate, DRAM bytes/token, us/token.

Fixed flags: --csv --page raw --target-processes all --cache-control none --clock-control none
--print-units base. `--cache-control none` is mandatory: the default flushes L2 before each kernel and
removes the effect we measure.

Counter permission. ncu needs GPU performance counters. On hosts where the driver has
NVreg_RestrictProfilingToAdminUsers=1 (the default), ncu fails with ERR_NVGPUCTRPERM even as root.
The host needs NVreg_RestrictProfilingToAdminUsers=0 (kernel module option, set before the driver
loads); RunPod containers cannot change it. Fallback: timing-only runs plus analytic bytes (from
bytes_moved or tensor sizes); results then carry counters_ok: false. To get real counters, retry on
another pod or run on the GCP PRO 6000 (infra/pod_runner.py --no-serve). CR_REQUIRE_COUNTERS=1 makes
the runner abort early with status no_counters when probe() fails.

CLI:
  python -m harness.counters --probe
  python -m harness.counters --tokens N [--kernel-regex R] [--nvtx] [--launch-skip K] [--launch-count C] -- <cmd...>
"""
from __future__ import annotations

import argparse
import contextlib
import csv
import glob
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import threading

METRICS = [
    "lts__t_sectors_lookup_hit.sum",
    "lts__t_sectors_lookup_miss.sum",
    "dram__bytes_read.sum",
    "gpu__time_duration.sum",
]
NVTX_DECODE = "cr_decode/"
ERR_PERM = "ERR_NVGPUCTRPERM"
NO_COUNTERS_NOTE = ("ncu counters unavailable (%s needs host NVreg_RestrictProfilingToAdminUsers=0): "
                    "timing-only run, analytic bytes, counters_ok=false" % ERR_PERM)

_M_HIT, _M_MISS, _M_DRAM, _M_TIME = METRICS
# unit name -> factor to base unit. ncu uses decimal prefixes (32 MiB prints as 33.55 Mbyte).
_BYTE_UNITS = {"byte": 1.0, "bytes": 1.0, "b": 1.0, "kbyte": 1e3, "mbyte": 1e6, "gbyte": 1e9, "tbyte": 1e12}
_TIME_UNITS = {"nsecond": 1.0, "ns": 1.0, "nsec": 1.0, "usecond": 1e3, "us": 1e3, "usec": 1e3,
               "msecond": 1e6, "ms": 1e6, "msec": 1e6, "second": 1e9, "s": 1e9, "sec": 1e9}
_COUNT_UNITS = {"": 1.0, "sector": 1.0, "sectors": 1.0}


def ncu_path() -> str | None:
    """CR_NCU, then PATH, then the usual install dirs (newest first)."""
    env = os.environ.get("CR_NCU")
    if env and os.path.isfile(env) and os.access(env, os.X_OK):
        return env
    found = shutil.which("ncu")
    if found:
        return found
    pats = ["/opt/nvidia/nsight-compute/*/ncu", "/usr/local/cuda*/nsight-compute*/ncu", "/usr/local/cuda*/bin/ncu"]
    for pat in pats:
        hits = sorted(glob.glob(pat), reverse=True)
        for h in hits:
            if os.path.isfile(h) and os.access(h, os.X_OK):
                return h
    return None


def build_ncu_cmd(argv, *, kernel_regex=None, nvtx_include=None, launch_skip=0, launch_count=None,
                  metrics=METRICS, extra=()) -> list[str]:
    cmd = [ncu_path() or "ncu", "--csv", "--page", "raw", "--target-processes", "all",
           "--cache-control", "none", "--clock-control", "none", "--print-units", "base",
           "--metrics", ",".join(metrics)]
    if kernel_regex:
        cmd += ["-k", "regex:" + kernel_regex]
    cmd += ["--kernel-name-base", "function"]
    if nvtx_include:
        cmd += ["--nvtx", "--nvtx-include", nvtx_include]
    if launch_skip:
        cmd += ["--launch-skip", str(int(launch_skip))]
    if launch_count:
        cmd += ["--launch-count", str(int(launch_count))]
    cmd += list(extra)
    cmd += list(argv)
    return cmd


# ---------------------------------------------------------------- CSV parsing

def _num(text):
    if text is None:
        return None
    t = text.strip().replace(",", "")
    if not t or t.upper() in ("N/A", "NAN"):
        return None
    try:
        return float(t)
    except ValueError:
        return None


def _scale(value, unit, table):
    if value is None:
        return None
    return value * table.get((unit or "").strip().lower(), 1.0)


def _csv_blocks(text):
    """Yield lists of csv rows, one per header line starting with "ID". Skips ncu banners and app output."""
    block = None
    for line in text.splitlines():
        s = line.strip()
        if not s or s.startswith("=="):
            continue
        if s.startswith('"ID"'):
            if block:
                yield block
            block = [next(csv.reader([s]))]
            continue
        if block is not None and s.startswith('"'):
            block.append(next(csv.reader([s])))
    if block:
        yield block


def _wide_rows(block):
    """--page raw layout: header row, units row (empty ID), then one row per kernel."""
    header = block[0]
    idx = {name: i for i, name in enumerate(header)}
    if "Kernel Name" not in idx or _M_TIME not in idx and _M_DRAM not in idx and _M_HIT not in idx:
        return []
    units = {}
    out = []
    for r in block[1:]:
        if len(r) != len(header):
            continue
        if r[0].strip() == "":
            units = {name: r[i] for name, i in idx.items()}
            continue
        vals = {}
        for m, table in ((_M_HIT, _COUNT_UNITS), (_M_MISS, _COUNT_UNITS), (_M_DRAM, _BYTE_UNITS), (_M_TIME, _TIME_UNITS)):
            vals[m] = _scale(_num(r[idx[m]]), units.get(m, ""), table) if m in idx else None
        out.append(_make_row(r[idx["ID"]], r[idx["Kernel Name"]], vals))
    return out


def _long_rows(block):
    """--page details layout: one row per (kernel, metric)."""
    header = block[0]
    idx = {name: i for i, name in enumerate(header)}
    need = ("Kernel Name", "Metric Name", "Metric Unit", "Metric Value")
    if any(n not in idx for n in need):
        return []
    order, per, names = [], {}, {}
    for r in block[1:]:
        if len(r) != len(header):
            continue
        kid = r[idx["ID"]]
        if kid not in per:
            per[kid] = {}
            names[kid] = r[idx["Kernel Name"]]
            order.append(kid)
        m = r[idx["Metric Name"]]
        table = _TIME_UNITS if m == _M_TIME else _BYTE_UNITS if m == _M_DRAM else _COUNT_UNITS
        per[kid][m] = _scale(_num(r[idx["Metric Value"]]), r[idx["Metric Unit"]], table)
    return [_make_row(k, names[k], per[k]) for k in order]


def _make_row(kid, name, vals):
    hit, miss = vals.get(_M_HIT), vals.get(_M_MISS)
    tot = (hit or 0.0) + (miss or 0.0)
    try:
        kid = int(kid)
    except (TypeError, ValueError):
        pass
    return {"id": kid, "name": name, "hit_sectors": hit, "miss_sectors": miss,
            "hit_rate": (hit or 0.0) / tot if tot > 0 else None,
            "dram_bytes": vals.get(_M_DRAM), "time_ns": vals.get(_M_TIME)}


def parse_ncu_csv(text: str) -> list[dict]:
    """Parse ncu --csv output (raw or details page) into rows: id, name, hit/miss sectors, hit_rate, dram_bytes, time_ns.

    Skips ==PROF== / ==WARNING== lines and target app output, the units row, and thousands separators.
    Units (byte/Kbyte/Mbyte/Gbyte, ns/usecond/msecond) are converted to bytes and ns.
    """
    rows = []
    for block in _csv_blocks(text):
        rows += _wide_rows(block) or _long_rows(block)
    return rows


# ---------------------------------------------------------------- aggregation

def summarize(rows: list[dict], tokens: int) -> dict:
    if tokens <= 0:
        raise ValueError("tokens must be > 0")
    hit = sum(r.get("hit_sectors") or 0.0 for r in rows)
    miss = sum(r.get("miss_sectors") or 0.0 for r in rows)
    dram = sum(r.get("dram_bytes") or 0.0 for r in rows)
    tns = sum(r.get("time_ns") or 0.0 for r in rows)
    return {"hit_rate": hit / (hit + miss) if hit + miss > 0 else None,
            "dram_bytes_per_token": dram / tokens, "us_per_token": tns / 1000.0 / tokens,
            "n_kernels": len(rows), "dram_bytes_total": dram, "time_ns_total": tns}


def bucket(rows: list[dict], buckets: dict[str, str]) -> dict:
    """Group rows by first matching regex (case-insensitive, searched in the kernel name); the rest go to "other"."""
    pats = [(name, re.compile(rx, re.IGNORECASE)) for name, rx in buckets.items()]
    acc = {name: [] for name, _ in pats}
    acc.setdefault("other", [])
    for r in rows:
        for name, p in pats:
            if p.search(r.get("name") or ""):
                acc[name].append(r)
                break
        else:
            acc["other"].append(r)
    out = {}
    for name, rs in acc.items():
        s = summarize(rs, 1)
        out[name] = {"time_ns": s["time_ns_total"], "dram_bytes": s["dram_bytes_total"],
                     "hit_rate": s["hit_rate"], "count": len(rs)}
    return out


# ---------------------------------------------------------------- running ncu

def _error_from(text: str, rc: int) -> str:
    if ERR_PERM in text:
        return ERR_PERM
    errs = [ln.strip() for ln in text.splitlines() if ln.startswith("==ERROR==")]
    if errs:
        return errs[-1]
    return "ncu exit code %d" % rc


def _kill_group(proc) -> None:
    """SIGKILL the whole process group of proc (ncu and the profiled target)."""
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except OSError:
        pass
    try:
        proc.kill()
    except OSError:
        pass


def _drain(proc, wait: float = 30.0) -> str:
    """Collect the output left in the pipes of a killed process and reap it."""
    try:
        out, _ = proc.communicate(timeout=wait)
        return out or ""
    except (subprocess.TimeoutExpired, OSError, ValueError):
        return ""


@contextlib.contextmanager
def _term_raises():
    """Main thread only: SIGTERM raises SystemExit so the ncu group (own session) is killed with us."""
    old = None
    if threading.current_thread() is threading.main_thread():
        def _handler(signum, frame):
            raise SystemExit(128 + signum)
        old = signal.signal(signal.SIGTERM, _handler)
    try:
        yield
    finally:
        if old is not None:
            signal.signal(signal.SIGTERM, old)


def run_ncu(argv, tokens: int, *, kernel_regex=None, nvtx_include=None, launch_skip=0, launch_count=None,
            env=None, timeout=1800, csv_out: str | None = None, extra=(), cwd=None) -> dict:
    """Run argv under ncu. Returns {ok, error, timed_out, cmd, rows, summary, csv_path}.

    csv_out: where to save the raw ncu output (stdout+stderr). ok is False with error "ERR_NVGPUCTRPERM"
    when the host denies counters (see module doc for the fallback).
    ncu runs in its own session; on timeout (or SIGTERM / Ctrl-C) the whole group is killed, so the
    profiled target cannot outlive ncu and keep using the GPU.
    """
    cmd = build_ncu_cmd(argv, kernel_regex=kernel_regex, nvtx_include=nvtx_include, launch_skip=launch_skip,
                        launch_count=launch_count, extra=extra)
    res = {"ok": False, "error": None, "timed_out": False, "cmd": cmd, "rows": [], "summary": None, "csv_path": None}
    if ncu_path() is None:
        res["error"] = "ncu not found (set CR_NCU or install nsight-compute)"
        return res
    full_env = dict(os.environ)
    if env:
        full_env.update(env)
    try:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, errors="replace",
                                env=full_env, cwd=cwd, start_new_session=True)
    except OSError as e:
        res["error"] = "cannot run ncu: %s" % e
        return res
    with _term_raises():
        try:
            text, _ = proc.communicate(timeout=timeout)
            text, rc = text or "", proc.returncode
        except subprocess.TimeoutExpired:
            _kill_group(proc)
            text, rc = _drain(proc), -1
            res["error"] = "timeout after %ss" % timeout
            res["timed_out"] = True
        except BaseException:
            _kill_group(proc)
            _drain(proc, 5.0)
            raise
    if csv_out:
        os.makedirs(os.path.dirname(os.path.abspath(csv_out)), exist_ok=True)
        with open(csv_out, "w") as f:
            f.write(text)
        res["csv_path"] = csv_out
    rows = parse_ncu_csv(text)
    res["rows"] = rows
    if rows:
        res["summary"] = summarize(rows, tokens)
    if ERR_PERM in text:
        res["error"] = ERR_PERM
    elif res["error"] is None and rc != 0:
        res["error"] = _error_from(text, rc)
    elif res["error"] is None and not rows:
        res["error"] = "no kernels profiled (check kernel regex / launch skip)"
    res["ok"] = res["error"] is None and bool(rows)
    return res


_PROBE_CODE = (
    "import torch\n"
    "assert torch.cuda.is_available(), 'no CUDA device'\n"
    "x = torch.ones(1 << 16, device='cuda')\n"
    "x.add_(1.0)\n"
    "torch.cuda.synchronize()\n"
)


def probe() -> dict:
    """Profile one tiny torch kernel. Returns {"ok": bool, "error": str | None}."""
    if ncu_path() is None:
        return {"ok": False, "error": "ncu not found (set CR_NCU or install nsight-compute)"}
    with tempfile.TemporaryDirectory(prefix="cr_probe_") as d:
        r = run_ncu([sys.executable, "-c", _PROBE_CODE], 1, launch_count=1, timeout=300,
                    csv_out=os.path.join(d, "probe.txt"))
        if not r["ok"]:
            err = r["error"]
            if err not in (ERR_PERM,) and r["csv_path"]:
                try:
                    with open(r["csv_path"]) as f:
                        tail = [ln.strip() for ln in f.read().splitlines() if ln.strip()][-3:]
                    err = "%s | %s" % (err, " / ".join(tail))
                except OSError:
                    pass
            return {"ok": False, "error": err}
    return {"ok": True, "error": None}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="harness.counters", description=__doc__.split("\n")[0])
    ap.add_argument("--probe", action="store_true", help="check that ncu can read counters")
    ap.add_argument("--tokens", type=int, default=1)
    ap.add_argument("--kernel-regex")
    ap.add_argument("--nvtx", action="store_true", help="only kernels inside NVTX range cr_decode")
    ap.add_argument("--launch-skip", type=int, default=0)
    ap.add_argument("--launch-count", type=int)
    ap.add_argument("--csv-out")
    ap.add_argument("cmd", nargs=argparse.REMAINDER)
    a = ap.parse_args(argv)
    if a.probe:
        r = probe()
        print(json.dumps(r))
        return 0 if r["ok"] else 1
    cmd = a.cmd[1:] if a.cmd and a.cmd[0] == "--" else a.cmd
    if not cmd:
        ap.error("need a command after --")
    r = run_ncu(cmd, a.tokens, kernel_regex=a.kernel_regex, nvtx_include=NVTX_DECODE if a.nvtx else None,
                launch_skip=a.launch_skip, launch_count=a.launch_count, csv_out=a.csv_out)
    print(json.dumps({"ok": r["ok"], "error": r["error"], "summary": r["summary"], "n_rows": len(r["rows"]),
                      "csv_path": r["csv_path"], "cmd": r["cmd"]}, indent=1))
    return 0 if r["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
