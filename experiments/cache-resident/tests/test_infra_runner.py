from __future__ import annotations

import hashlib
import json
import os
import pathlib
import subprocess
import sys
import time
import urllib.request

import pytest

from infra import pod_runner

EXP_DIR = pathlib.Path(__file__).resolve().parent.parent
RUNNER = EXP_DIR / "infra" / "pod_runner.py"


def run_runner(results_dir, *args, env_extra=None, timeout=60):
    env = dict(os.environ)
    for k in list(env):
        if k.startswith("RUNPOD_") or k.startswith("CR_"):
            env.pop(k)
    env.update({"CR_RESULTS_DIR": str(results_dir), "CR_GPU_TAG": "testgpu"})
    env.update(env_extra or {})
    return subprocess.run([sys.executable, str(RUNNER), *args], cwd=str(EXP_DIR), env=env,
                          capture_output=True, text=True, timeout=timeout)


def read(results_dir, rel):
    return json.loads((results_dir / rel).read_text())


WRITE_CMD = ("python -c \"from infra.results import write_result; "
             "write_result('t1', None, {'rows': [1]})\"")


def test_ok_run_writes_result_manifest_done(results_dir):
    r = run_runner(results_dir, "--exp", "t1", "--cmd", WRITE_CMD, "--no-serve", "--linger-min", "0")
    assert r.returncode == 0, r.stdout + r.stderr
    done = read(results_dir, "_status/DONE.json")
    assert done["status"] == "ok" and done["exit_code"] == 0 and done["exp"] == "t1" and done["gpu"] == "testgpu"
    assert done["cmd"] == WRITE_CMD and done["started_utc"] <= done["finished_utc"]
    man = read(results_dir, "_status/MANIFEST.json")["files"]
    paths = {f["path"] for f in man}
    assert any(p.startswith("t1/testgpu/") and p.endswith(".json") for p in paths)
    assert any(p.endswith("-runner.log") for p in paths)
    for f in man:
        data = (results_dir / f["path"]).read_bytes()
        assert f["size"] == len(data) and f["sha256"] == hashlib.sha256(data).hexdigest()
    res = [p for p in paths if p.endswith(".json")][0]
    doc = read(results_dir, res)
    assert doc["data"] == {"rows": [1]} and doc["gpu"] == "testgpu"
    # DONE is written last
    assert (results_dir / "_status" / "DONE.json").stat().st_mtime_ns >= (results_dir / "_status" / "MANIFEST.json").stat().st_mtime_ns
    st = read(results_dir, "_status/STATUS.json")
    assert st["phase"] == "done" and st["exp"] == "t1" and st["port"] is None and st["pid"] > 0


def test_failed_cmd(results_dir):
    r = run_runner(results_dir, "--exp", "t2", "--cmd", "echo boom; exit 3", "--no-serve", "--linger-min", "0")
    assert r.returncode == 3
    done = read(results_dir, "_status/DONE.json")
    assert done["status"] == "failed" and done["exit_code"] == 3
    log = next((results_dir / "t2" / "testgpu" / "logs").glob("*-runner.log"))
    assert "boom" in log.read_text()


def test_failsafe_kills_child(results_dir):
    t0 = time.time()
    r = run_runner(results_dir, "--exp", "t3", "--cmd", "sleep 60", "--no-serve", "--linger-min", "0",
                   "--failsafe-min", "0.02")
    assert time.time() - t0 < 30
    done = read(results_dir, "_status/DONE.json")
    assert done["status"] == "timeout", r.stdout + r.stderr
    assert r.returncode != 0


def test_no_network_without_pod_id(monkeypatch):
    calls = []
    monkeypatch.setattr(pod_runner.urllib.request, "urlopen", lambda *a, **k: calls.append(a))
    monkeypatch.delenv("RUNPOD_POD_ID", raising=False)
    monkeypatch.setenv("RUNPOD_API_KEY", "k")
    killed = []
    monkeypatch.setattr(pod_runner.os, "kill", lambda *a: killed.append(a))
    pod_runner.self_terminate()
    assert calls == [] and killed == []


def test_self_terminate_calls_api(monkeypatch):
    seen = {}

    class Resp:
        status = 200
        def read(self): return b'{"data": {"podTerminate": null}}'
        def __enter__(self): return self
        def __exit__(self, *a): return False

    def fake_urlopen(req, timeout=0):
        seen["url"] = req.full_url
        seen["auth"] = req.get_header("Authorization")
        seen["body"] = json.loads(req.data)
        return Resp()

    monkeypatch.setattr(pod_runner.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setenv("RUNPOD_POD_ID", "pod9")
    monkeypatch.setenv("RUNPOD_API_KEY", "k9")
    killed = []
    monkeypatch.setattr(pod_runner.os, "kill", lambda *a: killed.append(a))
    pod_runner.self_terminate()
    assert seen["url"] == "https://api.runpod.io/graphql" and seen["auth"] == "Bearer k9"
    assert 'podTerminate(input: {podId: "pod9"})' in seen["body"]["query"]
    assert killed and killed[0][0] == -1  # pid 1 drops default signals; -1 reaches the others


def test_self_terminate_retries_graphql_errors(monkeypatch):
    calls = []

    class Resp:
        status = 200
        def read(self): return b'{"errors": [{"message": "bad scope"}]}'
        def __enter__(self): return self
        def __exit__(self, *a): return False

    monkeypatch.setattr(pod_runner.urllib.request, "urlopen", lambda req, timeout=0: calls.append(1) or Resp())
    monkeypatch.setattr(pod_runner.time, "sleep", lambda s: None)
    monkeypatch.setenv("RUNPOD_POD_ID", "pod9")
    monkeypatch.setenv("RUNPOD_API_KEY", "k9")
    monkeypatch.setattr(pod_runner.os, "kill", lambda *a: None)
    pod_runner.self_terminate()
    assert len(calls) == 3


def test_require_counters_blocks_cmd(results_dir):
    marker = results_dir / "ran.txt"
    r = run_runner(results_dir, "--exp", "t4", "--cmd", f"touch {marker}", "--no-serve", "--linger-min", "0",
                   env_extra={"CR_REQUIRE_COUNTERS": "1"})
    done = read(results_dir, "_status/DONE.json")
    # No profiler on this host (or no harness yet): probe fails, cmd must not run.
    assert done["status"] == "no_counters", r.stdout + r.stderr
    assert not marker.exists() and r.returncode != 0


def test_unknown_exp_without_cmd(results_dir):
    r = run_runner(results_dir, "--exp", "no_such_exp_zz", "--no-serve")
    assert r.returncode == 2
    r = run_runner(results_dir, "--exp", "Bad Name", "--cmd", "true")
    assert r.returncode == 2


def test_api_key_not_in_child_env(results_dir):
    out = results_dir / "envdump.txt"
    run_runner(results_dir, "--exp", "t5", "--cmd", f"env > {out}", "--no-serve", "--linger-min", "0",
               env_extra={"RUNPOD_API_KEY": "sekrit-key"})
    text = out.read_text()
    assert "sekrit-key" not in text
    for k in ("CR_EXP=t5", "CR_GPU_TAG=testgpu", f"CR_RESULTS_DIR={results_dir}", f"CR_EXP_DIR={EXP_DIR}", "CR_LLAMA_BIN="):
        assert k in text


def test_serve_done_json(results_dir):
    env = dict(os.environ)
    for k in list(env):
        if k.startswith("RUNPOD_") or k.startswith("CR_"):
            env.pop(k)
    env.update({"CR_RESULTS_DIR": str(results_dir), "CR_GPU_TAG": "testgpu"})
    p = subprocess.Popen([sys.executable, str(RUNNER), "--exp", "t6", "--cmd", "echo hi", "--serve",
                          "--port", "0", "--linger-min", "0.1"], cwd=str(EXP_DIR), env=env,
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    try:
        body = None
        deadline = time.time() + 40
        port = None
        while time.time() < deadline and body is None:
            try:
                if port is None:
                    port = read(results_dir, "_status/STATUS.json")["port"]
                with urllib.request.urlopen(f"http://127.0.0.1:{port}/_status/DONE.json", timeout=2) as r:
                    assert r.status == 200
                    body = json.loads(r.read())
            except Exception:
                time.sleep(0.2)
        assert body is not None, "DONE.json not served"
        assert body["status"] == "ok" and body["exp"] == "t6"
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/", timeout=2) as r:
            assert r.status == 200
    finally:
        p.wait(timeout=40)
    assert p.returncode == 0


def test_manifest_cmd_lookup(tmp_path, monkeypatch):
    m = tmp_path / "m.json"
    m.write_text(json.dumps({"x": {"cmd": "echo from-manifest"}}))
    assert pod_runner.load_cmd("x", str(m)) == "echo from-manifest"
    assert pod_runner.load_cmd("y", str(m)) is None
    assert pod_runner.load_cmd("x", str(tmp_path / "missing.json")) is None
