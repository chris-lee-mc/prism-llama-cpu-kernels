"""Pod-side runner: run one named experiment, write MANIFEST/DONE, serve results, self-terminate.

Runs on RunPod (RUNPOD_POD_ID set) or any CUDA host. Stdlib only: it starts before bootstrap installs anything.
  python3 infra/pod_runner.py --exp NAME [--cmd STR] [--failsafe-min 40] [--linger-min 10] [--port 8000]
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import http.server
import json
import os
import pathlib
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
import urllib.request
from functools import partial

INFRA_DIR = pathlib.Path(__file__).resolve().parent
EXP_DIR = INFRA_DIR.parent
if str(EXP_DIR) not in sys.path:
    sys.path.insert(0, str(EXP_DIR))

from infra.gpus import tag_from_name  # noqa: E402

GRAPHQL_URL = "https://api.runpod.io/graphql"
_NAME_RE = re.compile(r"^[a-z0-9_.-]+$")


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def stamp() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def log(msg: str) -> None:
    print(f"[pod_runner {utc_now()}] {msg}", flush=True)


def write_json_atomic(path: pathlib.Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(obj, indent=1) + "\n", encoding="ascii")
    os.replace(tmp, path)


def detect_gpu_tag() -> str:
    """CR_GPU_TAG, else tag from nvidia-smi (no CUDA context), else cpu."""
    env = os.environ.get("CR_GPU_TAG")
    if env:
        return env
    try:
        r = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                           capture_output=True, text=True, timeout=20)
        name = r.stdout.strip().splitlines()[0] if r.returncode == 0 and r.stdout.strip() else ""
    except Exception:
        name = ""
    return tag_from_name(name)


def self_terminate() -> None:
    """Terminate this pod via the RunPod API, then SIGTERM all processes but pid 1. No-op off RunPod."""
    pod = os.environ.get("RUNPOD_POD_ID")
    if not pod:
        log("self_terminate: RUNPOD_POD_ID not set, nothing to do")
        return
    key = os.environ.get("RUNPOD_API_KEY")
    if key:
        body = json.dumps({"query": 'mutation { podTerminate(input: {podId: "%s"}) }' % pod}).encode()
        for attempt in range(3):
            try:
                req = urllib.request.Request(GRAPHQL_URL, data=body, method="POST", headers={
                    "Authorization": f"Bearer {key}", "Content-Type": "application/json",
                    "User-Agent": "cache-resident-runner/1"})
                with urllib.request.urlopen(req, timeout=20) as r:
                    status = r.status
                    try:
                        errs = json.loads(r.read().decode("utf-8", "replace")).get("errors")
                    except ValueError:
                        errs = None
                if errs:
                    # GraphQL errors come back in a 200 body
                    msg = "; ".join(str(e.get("message")) for e in errs if isinstance(e, dict))
                    log(f"podTerminate attempt {attempt + 1} API error: {msg[:200]}")
                    time.sleep(2)
                    continue
                log(f"podTerminate http {status}")
                break
            except Exception as e:
                log(f"podTerminate attempt {attempt + 1} failed: {type(e).__name__}")
                time.sleep(2)
    else:
        log("self_terminate: RUNPOD_API_KEY not set")
    # pid 1 ignores default-disposition signals; -1 reaches the rest so the container exits
    try:
        os.kill(-1, signal.SIGTERM)
    except Exception:
        pass


class Status:
    """Writes results/_status/STATUS.json on each phase change."""

    def __init__(self, root: pathlib.Path, exp: str, gpu: str, port):
        self.path = root / "_status" / "STATUS.json"
        self.d = {"exp": exp, "gpu": gpu, "phase": "bootstrap", "started_utc": utc_now(),
                  "updated_utc": utc_now(), "pid": os.getpid(), "port": port}
        self.write()

    def write(self) -> None:
        self.d["updated_utc"] = utc_now()
        write_json_atomic(self.path, self.d)

    def phase(self, name: str) -> None:
        self.d["phase"] = name
        self.write()
        log(f"phase {name}")


class _Handler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def log_message(self, *a):
        pass


def start_server(root: pathlib.Path, port: int):
    handler = partial(_Handler, directory=str(root))
    srv = http.server.ThreadingHTTPServer(("0.0.0.0", port), handler)
    srv.daemon_threads = True
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    log(f"serving {root} on :{srv.server_address[1]}")
    return srv


def sha256_file(p: pathlib.Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_manifest(root: pathlib.Path, exp: str, since: float | None = None) -> list[dict]:
    """All files under root/exp, plus files of other exp dirs modified at or after since (chained commands)."""
    files = []
    bases = [root / exp]
    if since is not None and root.is_dir():
        bases += [d for d in sorted(root.iterdir()) if d.is_dir() and d.name not in (exp, "_status") and _NAME_RE.match(d.name)]
    for base in bases:
        if not base.is_dir():
            continue
        for p in sorted(base.rglob("*")):
            if not p.is_file() or p.name.startswith("."):
                continue
            if base.name != exp and p.stat().st_mtime < since:
                continue
            files.append({"path": p.relative_to(root).as_posix(), "size": p.stat().st_size, "sha256": sha256_file(p)})
    write_json_atomic(root / "_status" / "MANIFEST.json", {"files": files})
    return files


def load_cmd(exp: str, manifest: str) -> str | None:
    p = pathlib.Path(manifest)
    if not p.is_absolute():
        p = EXP_DIR / p
    try:
        return json.loads(p.read_text())[exp]["cmd"]
    except Exception:
        return None


def kill_group(proc: subprocess.Popen) -> None:
    for sig, wait in ((signal.SIGTERM, 5), (signal.SIGKILL, 5)):
        try:
            os.killpg(proc.pid, sig)
        except ProcessLookupError:
            return
        try:
            proc.wait(timeout=wait)
            return
        except subprocess.TimeoutExpired:
            continue


def run_logged(argv: list[str], env: dict, logf, timeout: float, holder: list) -> tuple[int, bool]:
    """Run argv in its own process group, tee output to logf. Returns (exit_code, timed_out)."""
    proc = subprocess.Popen(argv, cwd=str(EXP_DIR), env=env, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, start_new_session=True)
    holder.append(proc)

    def tee():
        for line in iter(proc.stdout.readline, b""):
            text = line.decode("utf-8", "replace")
            logf.write(text)
            logf.flush()
            sys.stdout.write(text)
            sys.stdout.flush()

    t = threading.Thread(target=tee, daemon=True)
    t.start()
    timed_out = False
    try:
        proc.wait(timeout=max(timeout, 0.1))
    except subprocess.TimeoutExpired:
        timed_out = True
        log(f"child timeout after {timeout:.0f}s, killing group")
        kill_group(proc)
    t.join(timeout=10)
    return (proc.returncode if proc.returncode is not None else -9), timed_out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--exp", default=os.environ.get("CR_EXP"))
    ap.add_argument("--cmd", default=None)
    ap.add_argument("--failsafe-min", type=float, default=float(os.environ.get("CR_FAILSAFE_MIN", 40)))
    ap.add_argument("--linger-min", type=float, default=float(os.environ.get("CR_LINGER_MIN", 10)))
    ap.add_argument("--port", type=int, default=int(os.environ.get("CR_PORT", 8000)))
    ap.add_argument("--serve", dest="serve", action="store_true", default=None)
    ap.add_argument("--no-serve", dest="serve", action="store_false")
    ap.add_argument("--manifest", default="exps/manifest.json")
    ap.add_argument("--bootstrap", default=None, help="bootstrap script to run first (pod_entry.sh sets it)")
    a = ap.parse_args(argv)

    if not a.exp or not _NAME_RE.match(a.exp):
        print("error: --exp (or CR_EXP) must match [a-z0-9_.-]+", file=sys.stderr)
        return 2
    cmd = a.cmd or os.environ.get("CR_CMD") or load_cmd(a.exp, a.manifest)
    if not cmd:
        print(f"error: no command for exp {a.exp!r} (no --cmd, CR_CMD, or manifest entry)", file=sys.stderr)
        return 2

    t_start = time.time()
    # The bash failsafe counts from container start; CR_START_EPOCH carries that moment.
    try:
        t_origin = float(os.environ.get("CR_START_EPOCH", t_start))
    except ValueError:
        t_origin = t_start
    failsafe_s = a.failsafe_min * 60.0
    deadline = t_origin + failsafe_s
    margin = min(180.0, 0.25 * failsafe_s)
    on_pod = bool(os.environ.get("RUNPOD_POD_ID"))
    serve = on_pod if a.serve is None else a.serve

    root = pathlib.Path(os.environ.get("CR_RESULTS_DIR") or (EXP_DIR / "results"))
    root.mkdir(parents=True, exist_ok=True)
    gpu = detect_gpu_tag()
    started = utc_now()
    srv = start_server(root, a.port) if serve else None
    port = srv.server_address[1] if srv else None
    st = Status(root, a.exp, gpu, port)

    holder: list = []
    finished = threading.Event()

    def failsafe():
        if finished.wait(max(deadline - time.time(), 0)):
            return
        log("failsafe expired")
        for p in holder:
            try:
                kill_group(p)
            except Exception:
                pass
        self_terminate()
        os._exit(124)

    threading.Thread(target=failsafe, daemon=True).start()

    run_d = root / a.exp / gpu
    (run_d / "logs").mkdir(parents=True, exist_ok=True)
    logf = open(run_d / "logs" / f"{stamp()}-runner.log", "w", encoding="utf-8", errors="replace")
    env = dict(os.environ)
    env.pop("RUNPOD_API_KEY", None)
    env.update({"CR_EXP": a.exp, "CR_GPU_TAG": gpu, "CR_RESULTS_DIR": str(root), "CR_EXP_DIR": str(EXP_DIR),
                "CR_LLAMA_BIN": os.environ.get("CR_LLAMA_BIN") or str(EXP_DIR.parent.parent / "build-cuda" / "bin"),
                "PYTHONUNBUFFERED": "1"})

    status, code = "ok", 0
    remaining = lambda: deadline - time.time() - margin  # noqa: E731

    if a.bootstrap and pathlib.Path(a.bootstrap).is_file():
        log("bootstrap")
        code, to = run_logged(["bash", a.bootstrap], env, logf, remaining(), holder)
        bl = root / "_status" / "bootstrap.log"
        if bl.is_file():
            shutil.copyfile(bl, run_d / "logs" / f"{stamp()}-bootstrap.log")
        if to:
            status, code = "timeout", 124
        elif code != 0:
            status = "failed"

    if status == "ok" and os.environ.get("CR_REQUIRE_COUNTERS") == "1":
        log("probing profiler counters")
        c, to = run_logged([sys.executable, "-m", "harness.counters", "--probe"], env, logf,
                           min(300.0, max(remaining(), 1.0)), holder)
        if c != 0 or to:
            status, code = "no_counters", 3

    if status == "ok":
        st.phase("running")
        log(f"cmd: {cmd}")
        code, to = run_logged(["bash", "-lc", cmd], env, logf, remaining(), holder)
        if to:
            status, code = "timeout", 124
        elif code != 0:
            status = "failed"
    logf.close()

    st.phase("uploading")
    files = write_manifest(root, a.exp, since=t_start)
    write_json_atomic(root / "_status" / "DONE.json", {
        "exp": a.exp, "gpu": gpu, "status": status, "exit_code": code, "cmd": cmd,
        "started_utc": started, "finished_utc": utc_now()})
    st.phase("done")
    log(f"DONE status={status} exit_code={code} files={len(files)}")

    linger = max(min(a.linger_min * 60.0, deadline - time.time() - 20.0), 0.0)
    if serve and linger > 0:
        log(f"linger {linger:.0f}s")
        time.sleep(linger)
    finished.set()
    if srv:
        srv.shutdown()
    if on_pod:
        self_terminate()
        time.sleep(30)
    return 0 if status == "ok" else (code if code not in (0, None) else 1)


if __name__ == "__main__":
    sys.exit(main())
