from __future__ import annotations

import base64
import hashlib
import json
import os
import pathlib
import re
import shutil
import subprocess

import pytest

from infra import runpod
from infra.runpod import Ledger, RunPod, budget_gate, est_cost, proxy_url, startup_script

TYPES = [
    {"id": "NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition",
     "displayName": "RTX PRO 6000 Max-Q", "memoryInGb": 96, "secureCloud": True, "communityCloud": True,
     "securePrice": 1.5, "communityPrice": 0.5, "lowestPrice": {"uninterruptablePrice": 0.5, "stockStatus": "Low"}},
    {"id": "NVIDIA RTX PRO 6000 Blackwell Workstation Edition",
     "displayName": "RTX PRO 6000", "memoryInGb": 96, "secureCloud": True, "communityCloud": True,
     "securePrice": 2.09, "communityPrice": 1.69, "lowestPrice": {"uninterruptablePrice": 1.69, "stockStatus": "High"}},
    {"id": "NVIDIA GeForce RTX 5090", "displayName": "RTX 5090", "memoryInGb": 32, "secureCloud": True,
     "communityCloud": True, "securePrice": 1.19, "communityPrice": 0.69,
     "lowestPrice": {"uninterruptablePrice": 0.69, "stockStatus": "High"}},
    {"id": "NVIDIA GeForce RTX 3090", "displayName": "RTX 3090", "memoryInGb": 24, "secureCloud": True,
     "communityCloud": False, "securePrice": 0.43, "communityPrice": None,
     "lowestPrice": {"uninterruptablePrice": 0.43, "stockStatus": "High"}},
]
MAN = {"p0_x": {"cmd": "echo hi", "gpus": ["rtx5090", "rtx3090"], "est_min": 5, "failsafe_min": 40,
                "build_llama": False, "disk_gb": 40, "desc": "t"}}
FILE = b'{"x": 1}\n'
FILES = {"p0_x/rtx5090/r.json": FILE}


class FakeAPI(RunPod):
    def __init__(self, balance=50.0, create_errors=(), types=TYPES, pod_errors=(), term_errors=(), stuck=False):
        super().__init__(api_key="sekrit-key")
        self.pod_errors = list(pod_errors)    # raised by successive pod() queries
        self.term_errors = list(term_errors)  # raised by successive terminate calls
        self.stuck = stuck                    # terminate never takes effect
        self.pod_queries = 0
        self.balance_v = balance
        self.create_errors = list(create_errors)
        self.types = types
        self.calls = []
        self.created = []
        self.terminated = []
        self.alive = False

    def _post(self, query, variables=None):
        variables = variables or {}
        if "clientBalance" in query:
            return {"myself": {"clientBalance": self.balance_v, "currentSpendPerHr": 0, "spendLimit": 100}}
        if "gpuTypes" in query:
            return {"gpuTypes": self.types}
        if "podFindAndDeployOnDemand" in query:
            inp = variables["input"]
            self.calls.append(inp)
            if self.create_errors:
                raise RuntimeError(self.create_errors.pop(0))
            self.created.append(inp)
            self.alive = True
            self.name = inp["name"]
            return {"podFindAndDeployOnDemand": {"id": "pod1", "desiredStatus": "RUNNING", "costPerHr": 0.69}}
        if "podTerminate" in query:
            self.terminated.append(variables["id"])
            if self.term_errors:
                raise RuntimeError(self.term_errors.pop(0))
            self.alive = self.stuck
            return {"podTerminate": None}
        if "myself { pods" in query:
            if not self.alive:
                return {"myself": {"pods": []}}
            return {"myself": {"pods": [{"id": "pod1", "name": self.name, "desiredStatus": "RUNNING"}]}}
        if "pod(input" in query:
            self.pod_queries += 1
            if self.pod_errors:
                raise RuntimeError(self.pod_errors.pop(0))
            if not self.alive:
                return {"pod": None}
            return {"pod": {"id": "pod1", "desiredStatus": "RUNNING", "costPerHr": 0.69,
                            "runtime": {"uptimeInSeconds": 5}}}
        raise AssertionError(query)


@pytest.fixture
def env(tmp_path, monkeypatch):
    for k in list(os.environ):
        if k.startswith("RUNPOD_") or k.startswith("CR_"):
            monkeypatch.delenv(k)
    res = tmp_path / "res"
    monkeypatch.setenv("CR_RESULTS_DIR", str(res))
    monkeypatch.setenv("CR_SESSION_ID", "s1")
    monkeypatch.setattr(runpod, "_sleep", lambda s: None)
    monkeypatch.setattr(runpod, "_new_token", lambda: "TOK")
    return res, Ledger(tmp_path / "ledger.jsonl")


def serve_http(monkeypatch, files=FILES, done_status="ok", fail=None):
    man = {"files": [{"path": p, "size": len(b), "sha256": hashlib.sha256(b).hexdigest()} for p, b in files.items()]}
    base = proxy_url("pod1", token="TOK")

    def fake_get(url, timeout=20.0):
        if fail:
            raise fail
        if url == base:
            return 200, b"listing"
        if url == base + "_status/DONE.json":
            return 200, json.dumps({"exp": "p0_x", "gpu": "rtx5090", "status": done_status, "exit_code": 0}).encode()
        if url == base + "_status/STATUS.json":
            return 200, b'{"phase": "done"}'
        if url == base + "_status/MANIFEST.json":
            return 200, json.dumps(man).encode()
        rel = url[len(base):]
        if rel in files:
            return 200, files[rel]
        return 404, b""

    monkeypatch.setattr(runpod, "http_get", fake_get)


# ---- gate and ledger

def test_est_cost():
    assert est_cost(1.2, 40) == pytest.approx(1.2 * 45 / 60)


@pytest.mark.parametrize("bal,price,spent,cap,ok", [
    (50, 0.69, 0, 25, True),
    (2.5, 0.69, 0, 25, True),       # low balance only warns (auto top-up)
    (0.3, 0.69, 0, 25, False),      # below this run's est cost
    (50, 0.69, 24.9, 25, False),    # cap
    (50, 0.69, 24.4, 25, True),
])
def test_budget_gate(bal, price, spent, cap, ok):
    r, why = budget_gate(bal, price, 40, spent, cap)
    assert r is ok and why


def test_budget_gate_low_balance_warning():
    r, why = budget_gate(1.0, 0.69, 40, 0, 25)
    assert r and "WARNING" in why
    assert "WARNING" not in budget_gate(50, 0.69, 40, 0, 25)[1]
    assert budget_gate(None, 0.69, 40, 0, 25)[0]


def test_ledger_math(tmp_path, monkeypatch):
    monkeypatch.setenv("CR_SESSION_ID", "A")
    led = Ledger(tmp_path / "l.jsonl")
    assert led.session_spent() == 0.0
    led.record_launch("p1", "e", "rtx5090", "t", "COMMUNITY", 1.2)
    assert led.session_spent() == pytest.approx(est_cost(1.2, 40))
    led.record_end("p1", 0.30)
    assert led.session_spent() == pytest.approx(0.30)
    led.record_launch("p2", "e", "rtx5090", "t", "SECURE", 0.6, 30)
    assert led.session_spent() == pytest.approx(0.30 + est_cost(0.6, 30))
    monkeypatch.setenv("CR_SESSION_ID", "B")
    led.record_launch("p3", "e", "g", "t", "SECURE", 3.0)
    assert led.session_spent() == pytest.approx(est_cost(3.0, 40))
    assert led.session_spent("A") == pytest.approx(0.30 + est_cost(0.6, 30))
    (tmp_path / "l.jsonl").write_text((tmp_path / "l.jsonl").read_text() + "garbage\n")
    assert led.session_spent("A") == pytest.approx(0.30 + est_cost(0.6, 30))


# ---- startup script

def test_startup_script_content():
    s = startup_script({"RUNPOD_API_KEY": "sekrit-key", "CR_BRANCH": "exp/cache-resident", "CR_FAILSAFE_MIN": 40})
    assert "sekrit-key" not in s
    assert ".cr_started" in s
    assert "CR_FAILSAFE_MIN * 60" in s and "sleep" in s
    assert "git clone --depth 1 -b \"$CR_BRANCH\"" in s and "exp/cache-resident" in s
    assert "podTerminate" in s and "kill -TERM -1" in s and "kill -TERM 1" not in s
    assert "/workspace/repo" in s and "infra/pod_entry.sh" in s
    assert s.index(".cr_started") < s.index("sleep $((") < s.index("git clone")
    assert s.rstrip().splitlines()[-1].startswith("exec bash")
    s.encode("ascii")
    r = subprocess.run(["bash", "-n"], input=s, text=True, capture_output=True)
    assert r.returncode == 0, r.stderr


def test_startup_script_installs_python_before_exec():
    s = startup_script({})
    apt = s[s.index("apt-get install"):s.index("mkdir -p /workspace")]
    for pkg in ("git", "curl", "python3", "python3-pip", "python3-dev", "build-essential"):
        assert re.search(rf"(?<![\w-]){pkg}(?![\w-])", apt), pkg
    assert "command -v python3" in s
    assert s.index("apt-get install") < s.index("exec bash ")


def test_startup_script_uses_custom_branch():
    s = startup_script({"CR_BRANCH": "feat/x", "CR_REPO_URL": "https://example.com/r", "CR_FAILSAFE_MIN": 25})
    assert ":=feat/x}" in s and ":=https://example.com/r}" in s and ":=25}" in s


def test_docker_args_roundtrip():
    s = startup_script({})
    d = runpod.docker_args(s)
    m = re.fullmatch(r'bash -c "echo ([A-Za-z0-9+/=]+) \| base64 -d \| bash"', d)
    assert m and base64.b64decode(m.group(1)).decode() == s


def test_cr_term_function_sends_terminate(tmp_path):
    s = startup_script({})
    fn = s[s.index("cr_term() {"):s.index("\n}\n", s.index("cr_term() {")) + 3]
    bindir = tmp_path / "bin"
    bindir.mkdir()
    out = tmp_path / "curl.out"
    (bindir / "curl").write_text(f'#!/bin/bash\nprintf "%s\\n" "$@" > {out}\n')
    (bindir / "curl").chmod(0o755)
    script = fn + '\nkill() { echo "KILL $*" > ' + str(tmp_path / "kill.out") + '; }\ncr_term\n'
    env = {"PATH": f"{bindir}:/usr/bin:/bin", "RUNPOD_POD_ID": "abc123", "RUNPOD_API_KEY": "k"}
    r = subprocess.run(["bash", "-c", script], env=env, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    args = out.read_text().splitlines()
    assert "https://api.runpod.io/graphql" in args and "Authorization: Bearer k" in args
    body = json.loads(args[args.index("-d") + 1])
    assert body["query"] == 'mutation { podTerminate(input: {podId: "abc123"}) }'
    assert (tmp_path / "kill.out").read_text().strip() == "KILL -TERM -1"


def test_proxy_url():
    assert proxy_url("abc") == "https://abc-8000.proxy.runpod.net/"
    assert proxy_url("abc", 9) == "https://abc-9.proxy.runpod.net/"
    assert proxy_url("abc", token="t") == "https://abc-8000.proxy.runpod.net/t/"


def test_token_passed_to_pod_and_not_printed(env, monkeypatch, capsys):
    res, led = env
    serve_http(monkeypatch)
    urls = []
    real = runpod.http_get
    monkeypatch.setattr(runpod, "http_get", lambda url, timeout=20.0: urls.append(url) or real(url, timeout))
    api = FakeAPI()
    assert runpod.launch("p0_x", "rtx5090", manifest=MAN, client=api, ledger=led) == 0
    assert urls and all("/TOK/" in u for u in urls)
    assert "TOK" not in capsys.readouterr().out
    assert "CR_RESULTS_TOKEN" in json.dumps(api.created[-1])
    assert not any("TOK" in p.read_text() for p in res.rglob("*.json"))


def test_new_token_is_random():
    assert runpod.secrets.token_urlsafe(24) != runpod.secrets.token_urlsafe(24)
    assert len(runpod._new_token()) >= 32


# ---- API client

def test_create_pod_input(env):
    api = FakeAPI()
    pid = api.create_pod(name="n", gpu_type_id="g", cloud="COMMUNITY", image="img", env={"A": 1, "B": "x"},
                         docker_args="da", disk_gb=40)
    assert pid == "pod1"
    inp = api.created[0]
    assert inp["cloudType"] == "COMMUNITY" and inp["gpuTypeId"] == "g" and inp["imageName"] == "img"
    assert inp["containerDiskInGb"] == 40 and inp["ports"] == "8000/http" and inp["dockerArgs"] == "da"
    assert {"key": "A", "value": "1"} in inp["env"] and {"key": "B", "value": "x"} in inp["env"]


def test_match_gpu_type_and_price():
    t = runpod.match_gpu_type(TYPES, "NVIDIA RTX PRO 6000 Blackwell Workstation Edition")
    assert t["id"].endswith("Workstation Edition") and "Max-Q" not in t["id"]
    t2 = runpod.match_gpu_type(TYPES, "nvidia geforce rtx 5090")
    assert t2["displayName"] == "RTX 5090"
    assert runpod.match_gpu_type(TYPES, "NVIDIA H100") is None
    assert runpod.price_for(TYPES[3], "COMMUNITY") is None
    assert runpod.price_for(TYPES[3], "SECURE") == 0.43
    oos = dict(TYPES[2], lowestPrice={"stockStatus": "None"})
    assert runpod.price_for(oos, "COMMUNITY") is None


# ---- launch

def test_launch_success_downloads_and_terminates(env, monkeypatch, capsys):
    res, led = env
    serve_http(monkeypatch)
    api = FakeAPI()
    rc = runpod.launch("p0_x", None, manifest=MAN, client=api, ledger=led)
    assert rc == 0
    assert api.terminated == ["pod1"] and not api.alive
    assert (res / "p0_x" / "rtx5090" / "r.json").read_bytes() == FILE
    ev = [json.loads(x) for x in (led.path).read_text().splitlines()]
    assert [e["event"] for e in ev] == ["launch", "end"]
    assert ev[0]["price_hr"] == 0.69 and ev[0]["cloud"] == "COMMUNITY" and ev[1]["pod_id"] == "pod1"
    inp = api.created[0]
    keys = {e["key"]: e["value"] for e in inp["env"]}
    assert keys["CR_EXP"] == "p0_x" and keys["CR_FAILSAFE_MIN"] == "40" and keys["RUNPOD_API_KEY"] == "sekrit-key"
    assert keys["CR_BRANCH"] == "exp/cache-resident" and keys["HF_HUB_ENABLE_HF_TRANSFER"] == "1"
    assert "CR_CMD" not in keys and inp["containerDiskInGb"] == 40 and inp["ports"] == "8000/http"
    assert re.fullmatch(r"cr-p0_x-rtx5090-\d{4}", inp["name"])
    assert "sekrit-key" not in capsys.readouterr().out


def test_launch_cmd_override_and_build_llama(env, monkeypatch):
    res, led = env
    serve_http(monkeypatch)
    api = FakeAPI()
    assert runpod.launch("p0_x", "rtx5090", cmd="echo zz", build_llama=True, manifest=MAN, client=api, ledger=led) == 0
    keys = {e["key"]: e["value"] for e in api.created[0]["env"]}
    assert keys["CR_CMD"] == "echo zz" and keys["CR_BUILD_LLAMA"] == "1"


def test_cascade_falls_back_on_no_capacity(env, monkeypatch):
    res, led = env
    serve_http(monkeypatch)
    api = FakeAPI(create_errors=["There are no longer any instances available with the requested specifications."])
    rc = runpod.launch("p0_x", "rtx5090", manifest=MAN, client=api, ledger=led)
    assert rc == 0
    assert [c["cloudType"] for c in api.calls] == ["COMMUNITY", "SECURE"]
    assert api.terminated == ["pod1"]
    launches = [json.loads(x) for x in led.path.read_text().splitlines() if '"launch"' in x]
    assert len(launches) == 1 and launches[0]["cloud"] == "SECURE" and launches[0]["price_hr"] == 1.19


def test_pro6000_cascade_prefers_workstation_then_maxq(env, monkeypatch):
    res, led = env
    serve_http(monkeypatch, files={"p0_x/pro6000/r.json": FILE})
    api = FakeAPI(create_errors=["no longer any instances", "no longer any instances", "no longer any instances"])
    man = {"p0_x": dict(MAN["p0_x"], gpus=["pro6000"])}
    assert runpod.launch("p0_x", None, manifest=man, client=api, ledger=led) == 0
    ids = [c["gpuTypeId"] for c in api.calls]
    assert ids[0].endswith("Blackwell Workstation Edition") and "Max-Q" not in ids[0] and ids[1] == ids[0]
    assert ids[2].endswith("Max-Q Workstation Edition")  # Server is not in TYPES
    assert ids[3] == ids[2] and len(ids) == 4


def test_non_capacity_error_aborts(env, monkeypatch):
    res, led = env
    api = FakeAPI(create_errors=["Unauthorized"])
    assert runpod.launch("p0_x", "rtx5090", manifest=MAN, client=api, ledger=led) == 1
    assert len(api.calls) == 1 and not led.path.exists()


def test_all_capacity_errors(env, monkeypatch):
    res, led = env
    api = FakeAPI(create_errors=["no longer any instances"] * 5)
    assert runpod.launch("p0_x", "rtx5090", manifest=MAN, client=api, ledger=led) == 1
    assert api.terminated == [] and not led.path.exists()


def test_terminate_on_exception(env, monkeypatch):
    res, led = env
    serve_http(monkeypatch, fail=RuntimeError("network down"))
    api = FakeAPI()
    with pytest.raises(RuntimeError):
        runpod.launch("p0_x", "rtx5090", manifest=MAN, client=api, ledger=led)
    assert api.terminated == ["pod1"]
    assert [json.loads(x)["event"] for x in led.path.read_text().splitlines()] == ["launch", "end"]


def test_terminate_on_keyboard_interrupt(env, monkeypatch):
    res, led = env
    serve_http(monkeypatch, fail=KeyboardInterrupt())
    api = FakeAPI()
    assert runpod.launch("p0_x", "rtx5090", manifest=MAN, client=api, ledger=led) == 130
    assert api.terminated == ["pod1"]


def test_failed_experiment_still_downloads_and_terminates(env, monkeypatch):
    res, led = env
    serve_http(monkeypatch, done_status="failed")
    api = FakeAPI()
    assert runpod.launch("p0_x", "rtx5090", manifest=MAN, client=api, ledger=led) == 1
    assert (res / "p0_x" / "rtx5090" / "r.json").exists() and api.terminated == ["pod1"]


def test_manifest_fetch_retried(env, monkeypatch):
    res, led = env
    serve_http(monkeypatch)
    real = runpod.http_get
    n = [0]

    def flaky(url, timeout=20.0):
        if url.endswith("MANIFEST.json"):
            n[0] += 1
            if n[0] < 3:
                return (502, b"")
        return real(url, timeout)

    monkeypatch.setattr(runpod, "http_get", flaky)
    api = FakeAPI()
    assert runpod.launch("p0_x", "rtx5090", manifest=MAN, client=api, ledger=led) == 0
    assert n[0] == 3 and (res / "p0_x" / "rtx5090" / "r.json").exists() and api.terminated == ["pod1"]


@pytest.mark.parametrize("bad", [(502, b""), (200, b"not json"), (200, b"[1]"), (200, b'{"files": 3}')])
def test_manifest_unavailable_fails(env, monkeypatch, bad):
    res, led = env
    serve_http(monkeypatch)
    real = runpod.http_get
    monkeypatch.setattr(runpod, "http_get", lambda url, timeout=20.0: bad if url.endswith("MANIFEST.json") else real(url, timeout))
    api = FakeAPI()
    assert runpod.launch("p0_x", "rtx5090", manifest=MAN, client=api, ledger=led) == 1
    assert api.terminated == ["pod1"]


def test_ok_without_result_json_fails(env, monkeypatch):
    res, led = env
    serve_http(monkeypatch, files={"p0_x/rtx5090/log.txt": b"x"})
    api = FakeAPI()
    assert runpod.launch("p0_x", "rtx5090", manifest=MAN, client=api, ledger=led) == 1
    assert api.terminated == ["pod1"]


def test_is_result_json_accepts_tag_variants_and_nesting():
    f = runpod._is_result_json
    assert f("e2_08b/pro6000-server/20261013T122904Z.json", "e2_08b", "pro6000")
    assert f("p0_smoke/p0_l2/rtx5090/20261013T041003Z.json", "p0_smoke", "rtx5090")
    assert f("e1/rtx5090/x.json", "e1", "rtx5090")
    assert not f("e2_08b/pro6000-server/raw/partial.json", "e2_08b", "pro6000")
    assert not f("e2_08b/pro6000-server/logs/x.log", "e2_08b", "pro6000")
    assert not f("e2_08b/rtx5090/x.json", "e2_08b", "pro6000")
    assert not f("e3/rtx5090/x.json", "e2_08b", "rtx5090")
    assert f("wrap_a/pro6000-ws/20261015T000000Z.json", "wrap_a", "pro6000_600w")          # the 600 W key writes pro6000-* tags
    assert not f("wrap_a/rtx5090/20261015T000000Z.json", "wrap_a", "pro6000_600w")


def test_create_pod_cuda_filter_is_opt_in(env, monkeypatch):
    api = FakeAPI()
    monkeypatch.delenv("CR_ALLOWED_CUDA", raising=False)
    api.create_pod(name="n", gpu_type_id="g", cloud="SECURE", image="i", env={}, docker_args="d", disk_gb=40)
    assert "allowedCudaVersions" not in api.created[-1]
    monkeypatch.setenv("CR_ALLOWED_CUDA", "12.8, 12.9")
    api.create_pod(name="n", gpu_type_id="g", cloud="SECURE", image="i", env={}, docker_args="d", disk_gb=40)
    assert api.created[-1]["allowedCudaVersions"] == ["12.8", "12.9"]

def test_bad_done_json_is_polled_again(env, monkeypatch):
    res, led = env
    serve_http(monkeypatch)
    real = runpod.http_get
    n = [0]

    def torn(url, timeout=20.0):
        if url.endswith("DONE.json"):
            n[0] += 1
            if n[0] == 1:
                return 200, b'{"exp": '
        return real(url, timeout)

    monkeypatch.setattr(runpod, "http_get", torn)
    api = FakeAPI()
    assert runpod.launch("p0_x", "rtx5090", manifest=MAN, client=api, ledger=led) == 0
    assert n[0] == 2 and api.terminated == ["pod1"]


def test_bad_sha_not_saved(env, monkeypatch):
    res, led = env
    serve_http(monkeypatch)
    real = runpod.http_get

    def tamper(url, timeout=20.0):
        c, b = real(url, timeout)
        return (c, b + b"x") if url.endswith("r.json") else (c, b)

    monkeypatch.setattr(runpod, "http_get", tamper)
    api = FakeAPI()
    assert runpod.launch("p0_x", "rtx5090", manifest=MAN, client=api, ledger=led) == 1
    assert not (res / "p0_x" / "rtx5090" / "r.json").exists() and api.terminated == ["pod1"]


def test_path_traversal_skipped(env, monkeypatch):
    res, led = env
    serve_http(monkeypatch, files={"../evil.txt": b"x"})
    api = FakeAPI()
    assert runpod.launch("p0_x", "rtx5090", manifest=MAN, client=api, ledger=led) == 1
    assert not (res.parent / "evil.txt").exists()


def test_low_balance_only_warns(env, monkeypatch, capsys):
    res, led = env
    serve_http(monkeypatch)
    api = FakeAPI(balance=1.0)
    assert runpod.launch("p0_x", "rtx5090", manifest=MAN, client=api, ledger=led) == 0
    assert len(api.calls) == 1 and api.terminated == ["pod1"]
    assert "low balance" in capsys.readouterr().out


def test_gate_refuses_session_cap(env, monkeypatch):
    res, led = env
    monkeypatch.setenv("CR_SESSION_CAP_USD", "25")
    led.record_launch("old", "e", "g", "t", "SECURE", 1.0)
    led.record_end("old", 24.9)
    api = FakeAPI()
    assert runpod.launch("p0_x", "rtx5090", manifest=MAN, client=api, ledger=led) == 1
    assert api.calls == []


def test_max_price_skips_candidates(env, monkeypatch):
    res, led = env
    api = FakeAPI()
    assert runpod.launch("p0_x", "rtx5090", max_price=0.10, manifest=MAN, client=api, ledger=led) == 1
    assert api.calls == []
    serve_http(monkeypatch)
    # default 5090 cap is 1.30: SECURE at 1.19 is allowed, so community-out-of-stock still launches
    t = [dict(TYPES[2], communityCloud=False)]
    api = FakeAPI(types=t)
    assert runpod.launch("p0_x", "rtx5090", manifest=MAN, client=api, ledger=led) == 0
    assert api.calls[0]["cloudType"] == "SECURE"


def test_failsafe_limits(env):
    res, led = env
    api = FakeAPI()
    assert runpod.launch("p0_x", "rtx5090", failsafe_min=60, manifest=MAN, client=api, ledger=led) == 2
    assert api.calls == []


def test_unknown_inputs(env):
    res, led = env
    api = FakeAPI()
    assert runpod.launch("nope", None, manifest=MAN, client=api, ledger=led) == 2
    assert runpod.launch("p0_x", "h100", manifest=MAN, client=api, ledger=led) == 2


def test_dry_run_creates_nothing(env, capsys):
    res, led = env
    api = FakeAPI()
    assert runpod.launch("p0_x", "rtx5090", dry_run=True, manifest=MAN, client=api, ledger=led) == 0
    out = capsys.readouterr().out
    assert "PASS" in out and "startup script" in out and "CR_START_EPOCH" in out
    assert api.calls == [] and api.terminated == [] and not led.path.exists()
    assert "sekrit-key" not in out


def test_dry_run_reports_gate_refusal(env, capsys, monkeypatch):
    res, led = env
    monkeypatch.setenv("CR_SESSION_CAP_USD", "0.1")
    api = FakeAPI()
    assert runpod.launch("p0_x", "rtx5090", dry_run=True, manifest=MAN, client=api, ledger=led) == 1
    assert "REFUSE" in capsys.readouterr().out


def test_cli_dry_run_without_key(env, monkeypatch, capsys):
    rc = runpod.main(["launch", "--exp", "zz", "--gpu", "rtx5090", "--cmd", "true", "--dry-run"])
    assert rc == 0
    assert "NVIDIA GeForce RTX 5090" in capsys.readouterr().out


def test_cli_failsafe_requires_allow_long(env, capsys):
    assert runpod.main(["launch", "--exp", "zz", "--gpu", "rtx5090", "--cmd", "true", "--failsafe-min", "60",
                        "--dry-run"]) == 2


def test_cli_prices(env, monkeypatch, capsys):
    monkeypatch.setattr(runpod.RunPod, "_post", lambda self, q, v=None: {"gpuTypes": TYPES})
    monkeypatch.setenv("RUNPOD_API_KEY", "k")
    assert runpod.main(["prices", "--gpu", "rtx5090"]) == 0
    assert "0.69" in capsys.readouterr().out


# ---- real _post and idempotent terminate

class _Resp:
    def __init__(self, body, code=200):
        self._b, self.status_code = body, code

    def json(self):
        if isinstance(self._b, Exception):
            raise self._b
        return self._b


def test_post_http_and_errors(monkeypatch):
    import requests
    seen = {}

    def fake_post(url, json=None, timeout=0, headers=None):
        seen.update(url=url, json=json, headers=headers)
        return _Resp(seen.get("resp"))

    monkeypatch.setattr(requests, "post", fake_post)
    api = RunPod(api_key="k1")
    seen["resp"] = {"data": {"myself": {"clientBalance": 3}}}
    assert api.balance()["clientBalance"] == 3
    assert seen["url"] == "https://api.runpod.io/graphql" and seen["headers"]["Authorization"] == "Bearer k1"
    seen["resp"] = {"errors": [{"message": "boom"}]}
    with pytest.raises(RuntimeError, match="boom"):
        api.balance()
    monkeypatch.delenv("RUNPOD_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="RUNPOD_API_KEY"):
        RunPod(api_key=None)._post("query { x }")


def test_terminate_idempotent_and_pod_none():
    class A(RunPod):
        def _post(self, q, v=None):
            raise RuntimeError("RunPod API error: pod not found")

    a = A(api_key="k")
    a.terminate("gone")
    assert a.pod("gone") is None

    class B(RunPod):
        def _post(self, q, v=None):
            raise RuntimeError("RunPod API error: unauthorized")

    with pytest.raises(RuntimeError):
        B(api_key="k").terminate("x")


# ---- poll resilience, terminate retry, orphans, fit check

def test_transient_pod_errors_do_not_kill_run(env, monkeypatch):
    res, led = env
    serve_http(monkeypatch)
    api = FakeAPI(pod_errors=["RunPod request failed: ConnectionError", "RunPod HTTP 502", "RunPod HTTP 502"])
    assert runpod.launch("p0_x", "rtx5090", manifest=MAN, client=api, ledger=led) == 0
    assert (res / "p0_x" / "rtx5090" / "r.json").read_bytes() == FILE
    assert api.terminated == ["pod1"]


def test_pod_errors_give_up_after_n_in_a_row(env, monkeypatch):
    res, led = env
    serve_http(monkeypatch, fail=None)
    monkeypatch.setattr(runpod, "http_get", lambda url, timeout=20.0: (200, b"listing") if url.endswith("/TOK/") else (404, b""))
    api = FakeAPI(pod_errors=["boom"] * runpod.MAX_POD_FAILS)
    assert runpod.launch("p0_x", "rtx5090", manifest=MAN, client=api, ledger=led) == 1
    assert api.terminated == ["pod1"]
    assert [json.loads(x)["event"] for x in led.path.read_text().splitlines()] == ["launch", "end"]


def test_post_wraps_request_exceptions(monkeypatch):
    import requests

    def boom(*a, **k):
        raise requests.ConnectionError("down")

    monkeypatch.setattr(requests, "post", boom)
    with pytest.raises(RuntimeError, match="ConnectionError"):
        RunPod(api_key="k")._post("query { x }")


def test_terminate_is_retried(env, monkeypatch):
    res, led = env
    serve_http(monkeypatch)
    api = FakeAPI(term_errors=["api down", "api down"])
    assert runpod.launch("p0_x", "rtx5090", manifest=MAN, client=api, ledger=led) == 0
    assert api.terminated == ["pod1"] * 3 and not api.alive


def test_unconfirmed_terminate_records_worst_case(env, monkeypatch, capsys):
    res, led = env
    serve_http(monkeypatch)
    api = FakeAPI(stuck=True)
    runpod.launch("p0_x", "rtx5090", manifest=MAN, client=api, ledger=led)
    assert len(api.terminated) > 1
    assert "could not confirm" in capsys.readouterr().out
    end = [json.loads(x) for x in led.path.read_text().splitlines()][-1]
    assert end["event"] == "end" and end["cost_usd"] == pytest.approx(est_cost(0.69, 40), abs=1e-3)


def test_orphan_pod_reaped_after_create_error(env, monkeypatch):
    res, led = env

    class A(FakeAPI):
        def create_pod(self, **kw):
            super().create_pod(**kw)  # pod exists, but the response is lost
            raise RuntimeError("RunPod request failed: ReadTimeout")

    api = A()
    assert runpod.launch("p0_x", "rtx5090", manifest=MAN, client=api, ledger=led) == 1
    assert api.terminated == ["pod1"] and not api.alive
    ev = [json.loads(x) for x in led.path.read_text().splitlines()]
    assert [e["event"] for e in ev] == ["launch", "end"] and ev[0]["pod_id"] == "pod1"


def test_interrupt_during_create_reaps_orphan(env, monkeypatch):
    res, led = env

    class A(FakeAPI):
        def create_pod(self, **kw):
            super().create_pod(**kw)
            raise KeyboardInterrupt()

    api = A()
    assert runpod.launch("p0_x", "rtx5090", manifest=MAN, client=api, ledger=led) == 130
    assert api.terminated == ["pod1"]


def test_ready_loop_stops_when_pod_is_gone(env, monkeypatch, capsys):
    res, led = env
    monkeypatch.setattr(runpod, "http_get", lambda url, timeout=20.0: (0, b""))

    class A(FakeAPI):
        def _post(self, query, variables=None):
            r = super()._post(query, variables)
            if "pod(input" in query and self.pod_queries >= 2:
                self.alive = False
            return r

    api = A()
    assert runpod.launch("p0_x", "rtx5090", manifest=MAN, client=api, ledger=led) == 1
    assert "gone before it became ready" in capsys.readouterr().out
    assert api.pod_queries < 20


def test_launch_refuses_when_est_exceeds_failsafe(env, capsys):
    res, led = env
    man = {"e4": dict(MAN["p0_x"], est_min=35, build_llama=True, gpus=["pro6000"])}
    api = FakeAPI()
    assert runpod.launch("e4", None, manifest=man, client=api, ledger=led) == 2
    out = capsys.readouterr().out
    sug = -(-(35 + runpod.BOOTSTRAP_MIN + runpod.LLAMA_BUILD_MIN + 5) // 5) * 5
    assert "WARNING" in out and "--allow-long" in out and f"--failsafe-min {sug}" in out
    assert api.calls == []
    # the warning is required in a dry run too
    assert runpod.launch("e4", None, dry_run=True, manifest=man, client=api, ledger=led) == 2
    assert "WARNING" in capsys.readouterr().out


def test_long_failsafe_with_allow_long_launches(env, monkeypatch):
    res, led = env
    serve_http(monkeypatch, files={"e4/rtx5090/r.json": FILE})
    man = {"e4": dict(MAN["p0_x"], est_min=35, build_llama=True, gpus=["rtx5090"])}
    api = FakeAPI()
    assert runpod.launch("e4", None, failsafe_min=60, allow_long=True, manifest=man, client=api, ledger=led) == 0
    assert api.created[0]["minVcpuCount"] == 16


def test_min_vcpu_default_without_llama_build(env, monkeypatch):
    res, led = env
    serve_http(monkeypatch)
    api = FakeAPI()
    assert runpod.launch("p0_x", "rtx5090", manifest=MAN, client=api, ledger=led) == 0
    assert api.created[0]["minVcpuCount"] == 4


def test_600w_cascade_is_capped_at_the_community_price_and_maxq_has_its_own_key(env, monkeypatch, capsys):
    res, led = env
    assert runpod.DEFAULT_MAX_PRICE["pro6000_600w"] == 1.75 and runpod.DEFAULT_MAX_PRICE["pro6000_maxq"] < 1.75
    serve_http(monkeypatch, files={"p0_x/pro6000-ws/r.json": FILE})
    # no community stock: the secure price (2.09) is above the cap, so the cascade refuses it instead of paying it
    types = [dict(TYPES[1], communityCloud=False)]
    man = {"p0_x": dict(MAN["p0_x"], gpus=["pro6000_600w"])}
    api = FakeAPI(types=types)
    assert runpod.launch("p0_x", None, manifest=man, client=api, ledger=led) == 1
    assert api.created == [] and "$2.09/hr > max $1.75" in capsys.readouterr().out
    # --max-price lifts the cap
    api = FakeAPI(types=types)
    assert runpod.launch("p0_x", None, max_price=2.2, manifest=man, client=api, ledger=led) == 0
    # the Max-Q key never asks for a 600 W edition
    api = FakeAPI()
    man = {"p0_x": dict(MAN["p0_x"], gpus=["pro6000_maxq"])}
    assert runpod.launch("p0_x", None, manifest=man, client=api, ledger=led) == 0
    assert api.created[0]["gpuTypeId"].endswith("Max-Q Workstation Edition")


def test_chained_entry_checks_the_result_dir_of_its_first_command(env, monkeypatch, capsys):
    res, led = env
    serve_http(monkeypatch, files={"p0_x/rtx5090/r.json": FILE, "p0_y/rtx5090/r.json": FILE})
    man = {"chain": dict(MAN["p0_x"], gpus=["rtx5090"], result_exp="p0_x")}
    assert runpod.launch("chain", None, manifest=man, client=FakeAPI(), ledger=led) == 0
    assert (res / "p0_y" / "rtx5090" / "r.json").exists()                              # the second command's results come along
    man = {"chain": dict(MAN["p0_x"], gpus=["rtx5090"])}
    assert runpod.launch("chain", None, manifest=man, client=FakeAPI(), ledger=led) == 1
    assert "MANIFEST lists no chain/.../rtx5090" in capsys.readouterr().out
