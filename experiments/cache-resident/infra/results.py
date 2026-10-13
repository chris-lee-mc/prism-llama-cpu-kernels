"""Result files with environment metadata: results/<exp>/<gpu>/<UTCSTAMP>.json."""
from __future__ import annotations

import datetime as _dt
import json
import math
import os
import pathlib
import re
import socket
import subprocess
import sys

EXP_DIR = pathlib.Path(__file__).resolve().parent.parent
SCHEMA = 1
_NAME_RE = re.compile(r"^[a-z0-9_.-]+$")


def results_root() -> pathlib.Path:
    return pathlib.Path(os.environ.get("CR_RESULTS_DIR") or (EXP_DIR / "results"))


def _check_name(kind: str, value: str) -> str:
    if not isinstance(value, str) or not _NAME_RE.match(value) or set(value) <= {"."}:
        raise ValueError(f"bad {kind} name {value!r}: must match [a-z0-9_.-]+")
    return value


def _run(cmd: list[str], cwd=None, timeout: float = 10.0) -> str | None:
    try:
        r = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=timeout)
    except Exception:
        return None
    return r.stdout.strip() if r.returncode == 0 else None


def _git(*args: str) -> str | None:
    return _run(["git", *args], cwd=str(EXP_DIR))


def _smi_driver() -> tuple[str | None, str | None]:
    """Return (driver version, max CUDA version of the driver) from nvidia-smi."""
    drv = _run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"])
    drv = drv.splitlines()[0].strip() if drv else None
    cuda = None
    out = _run(["nvidia-smi"])
    if out:
        m = re.search(r"CUDA Version:\s*([0-9.]+)", out)
        cuda = m.group(1) if m else None
    return drv, cuda


def _smi_name() -> str | None:
    out = _run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"])
    return out.splitlines()[0].strip() if out else None


def _torch_name() -> str | None:
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.get_device_name(0)
    except Exception:
        pass
    return None


def gpu_tag() -> str:
    """CR_GPU_TAG, else tag from the device name, else cpu."""
    env = os.environ.get("CR_GPU_TAG")
    if env:
        return env
    from .gpus import tag_from_name
    return tag_from_name(_torch_name() or _smi_name())


def env_metadata() -> dict:
    """Describe the machine. Never raises; missing fields are None."""
    from .gpus import tag_from_name
    md: dict = {k: None for k in (
        "gpu_name", "gpu_tag", "cc", "sm", "driver", "cuda_runtime", "cuda_driver", "torch",
        "vram_bytes", "sm_count", "l2_bytes", "persisting_l2_max_bytes",
        "access_policy_window_max_bytes", "persisting_l2_limit_bytes", "git_sha", "git_branch",
        "hostname", "pod_id", "python", "counters_ok")}
    try:
        md["python"] = sys.version.split()[0]
        md["hostname"] = socket.gethostname()
        md["pod_id"] = os.environ.get("RUNPOD_POD_ID")
        md["git_sha"] = os.environ.get("CR_GIT_SHA") or _git("rev-parse", "HEAD")
        md["git_branch"] = _git("rev-parse", "--abbrev-ref", "HEAD")
        md["driver"], md["cuda_driver"] = _smi_driver()
    except Exception:
        pass
    try:
        import torch
        md["torch"] = torch.__version__
        md["cuda_runtime"] = torch.version.cuda
        if torch.cuda.is_available():
            p = torch.cuda.get_device_properties(0)
            md["gpu_name"] = p.name
            md["cc"] = f"{p.major}.{p.minor}"
            md["sm"] = f"sm_{p.major}{p.minor}"
            md["vram_bytes"] = int(p.total_memory)
            md["sm_count"] = int(p.multi_processor_count)
            l2 = getattr(p, "L2_cache_size", None)
            md["l2_bytes"] = int(l2) if l2 else None
    except Exception:
        pass
    if md["gpu_name"] is None:
        try:
            md["gpu_name"] = _smi_name()
        except Exception:
            pass
    try:
        md["gpu_tag"] = os.environ.get("CR_GPU_TAG") or tag_from_name(md["gpu_name"])
    except Exception:
        pass
    try:
        from harness import l2persist
        info = l2persist.device_info(0)
        md["persisting_l2_max_bytes"] = info.get("persisting_max_bytes")
        md["access_policy_window_max_bytes"] = info.get("window_max_bytes")
        md["persisting_l2_limit_bytes"] = info.get("persisting_limit_bytes")
        if md["l2_bytes"] is None:
            md["l2_bytes"] = info.get("l2_bytes")
    except Exception:
        pass
    return md


def run_dir(exp: str, gpu: str | None = None) -> pathlib.Path:
    _check_name("exp", exp)
    gpu = _check_name("gpu", gpu if gpu is not None else gpu_tag())
    d = results_root() / exp / gpu
    d.mkdir(parents=True, exist_ok=True)
    return d


def _stamp(now: _dt.datetime | None = None) -> str:
    return (now or _dt.datetime.now(_dt.timezone.utc)).strftime("%Y%m%dT%H%M%SZ")


def artifact_path(exp: str, name: str, gpu: str | None = None) -> pathlib.Path:
    """Path for a raw artifact: results/<exp>/<gpu>/raw/<UTCSTAMP>-<name>."""
    if not name or "/" in name or "\\" in name or name in (".", ".."):
        raise ValueError(f"bad artifact name {name!r}")
    d = run_dir(exp, gpu) / "raw"
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{_stamp()}-{name}"


def _clean(x):
    """Make x JSON-safe: NaN/inf -> None, numpy/torch scalars -> python, tuples -> lists."""
    if isinstance(x, dict):
        return {str(k): _clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple, set)):
        return [_clean(v) for v in x]
    if isinstance(x, bool) or x is None or isinstance(x, (int, str)):
        return x
    if isinstance(x, float):
        return x if math.isfinite(x) else None
    if hasattr(x, "tolist") and callable(x.tolist):
        try:
            return _clean(x.tolist())
        except Exception:
            return str(x)
    if hasattr(x, "item") and callable(x.item):
        try:
            return _clean(x.item())
        except Exception:
            return str(x)
    return str(x)


def write_result(exp: str, gpu: str | None, payload: dict, root: str | os.PathLike | None = None) -> pathlib.Path:
    """Write payload as results/<exp>/<gpu>/<UTCSTAMP>.json (suffix -1, -2 on collision)."""
    _check_name("exp", exp)
    gpu = _check_name("gpu", gpu if gpu is not None else gpu_tag())
    base = pathlib.Path(root) if root is not None else results_root()
    d = base / exp / gpu
    d.mkdir(parents=True, exist_ok=True)
    now = _dt.datetime.now(_dt.timezone.utc)
    doc = _clean({"schema": SCHEMA, "exp": exp, "gpu": gpu, "utc": now.isoformat(),
                  "env": env_metadata(), "data": payload})
    text = json.dumps(doc, indent=1, default=str, allow_nan=False)
    stamp = _stamp(now)
    for i in range(10000):
        path = d / (f"{stamp}.json" if i == 0 else f"{stamp}-{i}.json")
        try:
            with open(path, "x", encoding="ascii") as f:  # "x" never overwrites
                f.write(text + "\n")
            return path
        except FileExistsError:
            continue
    raise RuntimeError("could not find a free result file name")
