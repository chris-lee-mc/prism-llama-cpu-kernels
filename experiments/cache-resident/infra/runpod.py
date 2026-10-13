"""RunPod orchestrator (local side): budget gate, GPU cascade, launch, poll, download, always terminate.

  python -m infra.runpod balance | prices [--gpu KEY] | pods | terminate POD_ID
  python -m infra.runpod launch --exp E [--gpu KEY] [--cmd S] [--cloud any|community|secure]
                                [--failsafe-min N] [--max-price X] [--build-llama] [--dry-run]
The pod clones CR_BRANCH from CR_REPO_URL, so push the branch before launching.
"""
from __future__ import annotations

import argparse
import base64
import datetime as dt
import hashlib
import json
import os
import pathlib
import signal
import sys
import time

from .gpus import RUNPOD_CANDIDATES, cascade

EXP_DIR = pathlib.Path(__file__).resolve().parent.parent
GRAPHQL_URL = "https://api.runpod.io/graphql"
IMAGE = "nvidia/cuda:12.8.1-devel-ubuntu22.04"
DEFAULT_REPO = "https://github.com/chris-lee-mc/prism-llama-cpu-kernels"
DEFAULT_BRANCH = "exp/cache-resident"
DEFAULT_MAX_PRICE = {"rtx3090": 0.50, "rtx4090": 0.80, "rtx5090": 1.30, "pro6000": 2.60, "l40s": 1.20}
MAX_FAILSAFE_MIN = 90
CAPACITY_HINTS = ("no longer any instances", "no instances", "not available", "unavailable", "out of stock",
                  "capacity", "no gpus", "required specifications", "does not have the resources")
OUT_OF_STOCK = ("none", "out of stock", "outofstock", "out_of_stock", "unavailable")

# Indirection so tests can run without sleeping.
_sleep = time.sleep
_now = time.time


def utc_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def http_get(url: str, timeout: float = 20.0) -> tuple[int, bytes]:
    """GET url. Returns (status, body); (0, b"") on connection errors."""
    import requests
    try:
        r = requests.get(url, timeout=timeout)
        return r.status_code, r.content
    except requests.RequestException:
        return 0, b""


# ---------------------------------------------------------------- budget

def est_cost(price_hr: float, failsafe_min: int) -> float:
    """Worst case: failsafe lifetime plus 5 min for startup and teardown."""
    return price_hr * (failsafe_min + 5) / 60.0


def budget_gate(balance: float, price_hr: float, failsafe_min: int, session_spent: float, cap: float,
                margin: float = 2.0) -> tuple[bool, str]:
    est = est_cost(price_hr, failsafe_min)
    if balance < est + margin:
        return False, f"balance ${balance:.2f} < est ${est:.2f} + margin ${margin:.2f}"
    if session_spent + est > cap:
        return False, f"session spend ${session_spent:.2f} + est ${est:.2f} > cap ${cap:.2f}"
    return True, f"ok: est ${est:.2f}, balance ${balance:.2f}, session ${session_spent:.2f}/{cap:.2f}"


def session_id_default() -> str:
    return os.environ.get("CR_SESSION_ID") or dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d")


class Ledger:
    """JSONL spend ledger. Events: launch (with price) and end (with cost)."""

    def __init__(self, path: str | os.PathLike | None = None):
        self.path = pathlib.Path(path or os.environ.get("CR_LEDGER") or (EXP_DIR / ".runpod_ledger.jsonl"))

    def _append(self, rec: dict) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "a", encoding="ascii") as f:
            f.write(json.dumps(rec) + "\n")

    def _read(self) -> list[dict]:
        if not self.path.is_file():
            return []
        out = []
        for line in self.path.read_text(encoding="ascii", errors="replace").splitlines():
            try:
                out.append(json.loads(line))
            except ValueError:
                continue
        return out

    def record_launch(self, pod_id, exp, gpu, gpu_type_id, cloud, price_hr, failsafe_min: int = 40) -> None:
        self._append({"event": "launch", "utc": utc_iso(), "session": session_id_default(), "pod_id": pod_id,
                      "exp": exp, "gpu": gpu, "gpu_type_id": gpu_type_id, "cloud": cloud,
                      "price_hr": price_hr, "failsafe_min": failsafe_min})

    def record_end(self, pod_id, cost_usd) -> None:
        self._append({"event": "end", "utc": utc_iso(), "pod_id": pod_id, "cost_usd": cost_usd})

    def session_spent(self, session_id: str | None = None) -> float:
        """Ended pods count their cost, open pods count est_cost."""
        sid = session_id or session_id_default()
        launches: dict[str, dict] = {}
        ended: dict[str, float] = {}
        for r in self._read():
            if r.get("event") == "launch" and r.get("session") == sid:
                launches[r["pod_id"]] = r
            elif r.get("event") == "end":
                ended[r["pod_id"]] = float(r.get("cost_usd") or 0.0)
        total = 0.0
        for pid, r in launches.items():
            if pid in ended:
                total += ended[pid]
            else:
                total += est_cost(float(r.get("price_hr") or 0.0), int(r.get("failsafe_min") or 40))
        return total


# ---------------------------------------------------------------- pod startup

def proxy_url(pod_id: str, port: int = 8000) -> str:
    return f"https://{pod_id}-{port}.proxy.runpod.net/"


def startup_script(env: dict) -> str:
    """Bash run as the container command. Holds no secret values; reads them from the pod env at run time."""
    branch = env.get("CR_BRANCH") or DEFAULT_BRANCH
    repo = env.get("CR_REPO_URL") or DEFAULT_REPO
    failsafe = env.get("CR_FAILSAFE_MIN") or 40
    return f"""#!/bin/bash
: "${{CR_BRANCH:={branch}}}"
: "${{CR_REPO_URL:={repo}}}"
: "${{CR_FAILSAFE_MIN:={failsafe}}}"
export CR_BRANCH CR_REPO_URL CR_FAILSAFE_MIN
export CR_START_EPOCH=$(date +%s)
cr_term() {{
  if [ -n "${{RUNPOD_POD_ID:-}}" ]; then
    if [ -n "${{RUNPOD_API_KEY:-}}" ] && command -v curl >/dev/null 2>&1; then
      curl -s -m 20 -X POST https://api.runpod.io/graphql -H "Authorization: Bearer $RUNPOD_API_KEY" \\
        -H "Content-Type: application/json" \\
        -d "{{\\"query\\":\\"mutation {{ podTerminate(input: {{podId: \\\\\\"$RUNPOD_POD_ID\\\\\\"}}) }}\\"}}" >/dev/null 2>&1
    elif command -v runpodctl >/dev/null 2>&1; then
      runpodctl remove pod "$RUNPOD_POD_ID" >/dev/null 2>&1
    fi
  fi
  kill -TERM 1
}}
if [ -e /root/.cr_started ]; then
  echo "container restart detected, terminating"
  cr_term
  sleep 60
  exit 0
fi
touch /root/.cr_started
( sleep $((CR_FAILSAFE_MIN * 60)); cr_term ) &
if ! command -v git >/dev/null 2>&1 || ! command -v curl >/dev/null 2>&1; then
  export DEBIAN_FRONTEND=noninteractive
  apt-get update -qq && apt-get install -y -qq --no-install-recommends git curl ca-certificates
fi
mkdir -p /workspace
ok=0
for i in 1 2 3; do
  rm -rf /workspace/repo
  if git clone --depth 1 -b "$CR_BRANCH" "$CR_REPO_URL" /workspace/repo; then ok=1; break; fi
  sleep 5
done
if [ "$ok" != "1" ]; then echo "git clone failed"; cr_term; sleep 60; exit 1; fi
exec bash /workspace/repo/experiments/cache-resident/infra/pod_entry.sh
"""


def docker_args(script: str) -> str:
    """Wrap the script into one docker CMD string that survives arg parsing (base64, no quoting issues)."""
    b64 = base64.b64encode(script.encode("ascii")).decode("ascii")
    return f'bash -c "echo {b64} | base64 -d | bash"'


# ---------------------------------------------------------------- API client

class RunPod:
    """Thin GraphQL client. All HTTP goes through _post so tests can mock it."""

    def __init__(self, api_key: str | None = None):
        self.api_key = api_key or os.environ.get("RUNPOD_API_KEY")

    def _post(self, query: str, variables: dict | None = None) -> dict:
        import requests
        if not self.api_key:
            raise RuntimeError("RUNPOD_API_KEY is not set")
        r = requests.post(GRAPHQL_URL, json={"query": query, "variables": variables or {}}, timeout=60,
                          headers={"Authorization": f"Bearer {self.api_key}", "User-Agent": "cache-resident/1"})
        try:
            body = r.json()
        except ValueError:
            raise RuntimeError(f"RunPod HTTP {r.status_code}: non-JSON response")
        if body.get("errors"):
            msgs = "; ".join(str(e.get("message")) for e in body["errors"])
            raise RuntimeError(f"RunPod API error: {msgs}")
        if r.status_code >= 400:
            raise RuntimeError(f"RunPod HTTP {r.status_code}")
        return body.get("data") or {}

    def balance(self) -> dict:
        d = self._post("query { myself { clientBalance currentSpendPerHr spendLimit } }")
        return d["myself"]

    def gpu_types(self) -> list[dict]:
        q = ("query { gpuTypes { id displayName memoryInGb secureCloud communityCloud securePrice communityPrice "
             "lowestPrice(input: {gpuCount: 1}) { uninterruptablePrice stockStatus } } }")
        return self._post(q)["gpuTypes"]

    def pods(self) -> list[dict]:
        q = "query { myself { pods { id name desiredStatus costPerHr runtime { uptimeInSeconds } } } }"
        return self._post(q)["myself"]["pods"] or []

    def create_pod(self, *, name, gpu_type_id, cloud, image, env: dict, docker_args: str, disk_gb: int,
                   ports="8000/http") -> str:
        q = ("mutation($input: PodFindAndDeployOnDemandInput) { podFindAndDeployOnDemand(input: $input) "
             "{ id desiredStatus costPerHr machineId } }")
        inp = {"cloudType": cloud, "gpuCount": 1, "gpuTypeId": gpu_type_id, "name": name, "imageName": image,
               "dockerArgs": docker_args, "containerDiskInGb": int(disk_gb), "volumeInGb": 0,
               "minVcpuCount": 4, "minMemoryInGb": 16, "ports": ports,
               "env": [{"key": k, "value": str(v)} for k, v in env.items()]}
        d = self._post(q, {"input": inp})
        pod = d.get("podFindAndDeployOnDemand")
        if not pod or not pod.get("id"):
            raise RuntimeError("RunPod API error: no pod returned (no longer any instances available?)")
        return pod["id"]

    def pod(self, pod_id) -> dict | None:
        q = ("query($id: String!) { pod(input: {podId: $id}) { id desiredStatus costPerHr "
             "runtime { uptimeInSeconds } } }")
        try:
            return self._post(q, {"id": pod_id}).get("pod")
        except RuntimeError as e:
            if "not found" in str(e).lower() or "does not exist" in str(e).lower():
                return None
            raise

    def terminate(self, pod_id) -> None:
        try:
            self._post("mutation($id: String!) { podTerminate(input: {podId: $id}) }", {"id": pod_id})
        except RuntimeError as e:
            if "not found" in str(e).lower() or "does not exist" in str(e).lower():
                return
            raise


# ---------------------------------------------------------------- launch

def load_manifest(path: str | os.PathLike | None = None) -> dict:
    p = pathlib.Path(path) if path else EXP_DIR / "exps" / "manifest.json"
    try:
        return json.loads(p.read_text())
    except (OSError, ValueError):
        return {}


def match_gpu_type(types: list[dict], name: str) -> dict | None:
    """Match a candidate name to the live list: exact first, then case-insensitive substring of id or displayName."""
    n = name.lower()
    for t in types:
        if n in ((t.get("id") or "").lower(), (t.get("displayName") or "").lower()):
            return t
    for t in types:
        if n in (t.get("id") or "").lower() or n in (t.get("displayName") or "").lower():
            return t
    return None


def price_for(t: dict, cloud: str) -> float | None:
    """Per-hour price for one cloud, or None when the cloud is not offered or has no price / stock."""
    if cloud == "COMMUNITY":
        ok, price = t.get("communityCloud"), t.get("communityPrice")
    else:
        ok, price = t.get("secureCloud"), t.get("securePrice")
    if not ok or not price:
        return None
    lp = t.get("lowestPrice") or {}
    if str(lp.get("stockStatus") or "x").strip().lower() in OUT_OF_STOCK:
        return None
    return float(price)


def _is_capacity_error(e: Exception) -> bool:
    m = str(e).lower()
    return any(h in m for h in CAPACITY_HINTS)


class _Interrupt(BaseException):
    pass


def _fetch_results(base: str, files: list[dict], dest: pathlib.Path) -> bool:
    """Download every listed file, verify sha256. Returns True when all files are good."""
    all_ok = True
    for ent in files:
        rel = pathlib.PurePosixPath(ent["path"])
        if rel.is_absolute() or ".." in rel.parts:
            print(f"  skip unsafe path {ent['path']}")
            all_ok = False
            continue
        out = dest / pathlib.Path(*rel.parts)
        good = False
        for _ in range(3):
            code, body = http_get(base + str(rel), timeout=120)
            if code == 200 and hashlib.sha256(body).hexdigest() == ent.get("sha256"):
                out.parent.mkdir(parents=True, exist_ok=True)
                out.write_bytes(body)
                good = True
                break
            _sleep(2)
        print(f"  {'got ' if good else 'FAIL'} {rel} ({ent.get('size')} B)")
        all_ok = all_ok and good
    return all_ok


def launch(exp: str, gpu: str | None = None, *, cmd=None, cloud="any", failsafe_min=40, build_llama=None,
           max_price=None, dry_run=False, allow_long=False, manifest: dict | None = None,
           client: RunPod | None = None, ledger: Ledger | None = None) -> int:
    man = manifest if manifest is not None else load_manifest()
    ent = man.get(exp)
    if ent is None and not (cmd and gpu):
        print(f"error: exp {exp!r} not in exps/manifest.json (pass --gpu and --cmd to run it anyway)")
        return 2
    ent = ent or {}
    if failsafe_min > 40 and not allow_long:
        print("error: failsafe_min > 40 requires allow_long")
        return 2
    failsafe_min = int(min(failsafe_min, MAX_FAILSAFE_MIN))
    keys = [gpu] if gpu else list(ent.get("gpus") or [])
    for k in keys:
        if k not in RUNPOD_CANDIDATES:
            print(f"error: unknown gpu key {k!r}; known: {', '.join(RUNPOD_CANDIDATES)}")
            return 2
    if not keys:
        print("error: no gpu given and none in manifest")
        return 2
    if build_llama is None:
        build_llama = bool(ent.get("build_llama"))
    disk_gb = int(ent.get("disk_gb") or 40)
    cap = float(os.environ.get("CR_SESSION_CAP_USD", 25))
    ledger = ledger or Ledger()
    env = {"RUNPOD_API_KEY": "<redacted>", "CR_EXP": exp, "CR_FAILSAFE_MIN": failsafe_min,
           "CR_LINGER_MIN": os.environ.get("CR_LINGER_MIN", 10), "CR_BUILD_LLAMA": int(build_llama),
           "CR_BRANCH": os.environ.get("CR_BRANCH", DEFAULT_BRANCH),
           "CR_REPO_URL": os.environ.get("CR_REPO_URL", DEFAULT_REPO), "HF_HUB_ENABLE_HF_TRANSFER": 1}
    if cmd:
        env["CR_CMD"] = cmd
    script = startup_script(env)

    client = client or RunPod()
    types, balance = [], None
    if client.api_key or not dry_run:
        try:
            balance = float(client.balance()["clientBalance"])
            types = client.gpu_types()
        except Exception as e:
            if not dry_run:
                print(f"error: cannot query RunPod: {e}")
                return 1
            print(f"note: RunPod query failed ({e}); showing candidates without prices")
    spent = ledger.session_spent()
    print(f"exp={exp} keys={keys} cloud={cloud} failsafe={failsafe_min}min build_llama={build_llama} "
          f"balance={balance} session_spent=${spent:.2f} cap=${cap:.2f}")

    plan = []  # (gpu_key, type_name, type_id, cloud_name, price)
    for key in keys:
        limit = max_price if max_price is not None else DEFAULT_MAX_PRICE.get(key, 1e9)
        for name, cl in cascade(key, cloud):
            t = match_gpu_type(types, name)
            price = price_for(t, cl) if t else None
            if price is None:
                print(f"  skip {name} [{cl}]: {'not in live list' if not t else 'no price/stock'}")
                continue
            if price > limit:
                print(f"  skip {name} [{cl}]: ${price:.2f}/hr > max ${limit:.2f}")
                continue
            ok, why = budget_gate(balance, price, failsafe_min, spent, cap)
            print(f"  cand {name} [{cl}] ${price:.2f}/hr -> {'PASS' if ok else 'REFUSE'} ({why})")
            if ok:
                plan.append((key, name, t["id"], cl, price))
    if dry_run:
        if not types:
            for key in keys:
                print(f"  {key}: {[f'{n} [{c}]' for n, c in cascade(key, cloud)]}")
        print("--- startup script ---")
        print(script)
        print(f"--- docker args length {len(docker_args(script))} ---")
        return 0 if (plan or not types) else 1
    if not plan:
        print("error: no candidate passed the gate")
        return 1

    run_env = dict(env)
    run_env["RUNPOD_API_KEY"] = client.api_key
    pod_id, chosen = None, None
    for key, name, tid, cl, price in plan:
        label = f"cr-{exp}-{key}-{dt.datetime.now(dt.timezone.utc).strftime('%H%M')}"
        try:
            pod_id = client.create_pod(name=label, gpu_type_id=tid, cloud=cl, image=IMAGE, env=run_env,
                                       docker_args=docker_args(script), disk_gb=disk_gb)
        except Exception as e:
            print(f"  create failed for {name} [{cl}]: {e}")
            if _is_capacity_error(e):
                continue
            return 1
        chosen = (key, name, tid, cl, price)
        break
    if not pod_id:
        print("error: no capacity on any candidate")
        return 1
    key, name, tid, cl, price = chosen
    print(f"created pod {pod_id} ({name} [{cl}] ${price:.2f}/hr)")
    ledger.record_launch(pod_id, exp, key, tid, cl, price, failsafe_min)

    def on_signal(signum, frame):
        raise _Interrupt(f"signal {signum}")

    old = {}
    for s in (signal.SIGINT, signal.SIGTERM):
        try:
            old[s] = signal.signal(s, on_signal)
        except ValueError:
            pass  # not in main thread

    t0 = _now()
    cost_hr = price
    rc = 1
    try:
        base = proxy_url(pod_id)
        ready_deadline = t0 + 15 * 60
        while True:
            p = client.pod(pod_id)
            if p and p.get("costPerHr"):
                cost_hr = float(p["costPerHr"])
            up = bool(p and p.get("desiredStatus") == "RUNNING" and p.get("runtime"))
            if up and http_get(base)[0] == 200:
                break
            if _now() > ready_deadline:
                print("error: pod not ready within 15 min")
                return 1
            _sleep(10)
        print(f"pod ready after {_now() - t0:.0f}s: {base}")
        deadline = _now() + (failsafe_min + 2) * 60
        phase, done = None, None
        while _now() < deadline:
            code, body = http_get(base + "_status/DONE.json")
            if code == 200:
                done = json.loads(body.decode("utf-8"))
                break
            code, body = http_get(base + "_status/STATUS.json")
            if code == 200:
                try:
                    ph = json.loads(body.decode("utf-8")).get("phase")
                except ValueError:
                    ph = phase
                if ph != phase:
                    print(f"  phase: {ph}")
                    phase = ph
            p = client.pod(pod_id)
            if p is None or p.get("desiredStatus") in ("TERMINATED", "EXITED"):
                print("pod is gone before DONE")
                break
            if p.get("costPerHr"):
                cost_hr = float(p["costPerHr"])
            _sleep(20)
        if done is None:
            print("error: no DONE marker")
            return 1
        print(f"DONE status={done.get('status')} exit_code={done.get('exit_code')}")
        code, body = http_get(base + "_status/MANIFEST.json")
        files = json.loads(body.decode("utf-8")).get("files", []) if code == 200 else []
        dest = pathlib.Path(os.environ.get("CR_RESULTS_DIR") or (EXP_DIR / "results"))
        got = _fetch_results(base, files, dest)
        print(f"downloaded {len(files)} files to {dest}" if got else "error: some downloads failed")
        rc = 0 if (got and done.get("status") == "ok") else 1
        return rc
    except (_Interrupt, KeyboardInterrupt) as e:
        print(f"interrupted: {e}")
        return 130
    finally:
        for s in (signal.SIGINT, signal.SIGTERM):  # finish cleanup even on a second Ctrl-C
            try:
                signal.signal(s, signal.SIG_IGN)
            except ValueError:
                pass
        wall_h = (_now() - t0) / 3600.0
        term_ok = False
        try:
            client.terminate(pod_id)
            for _ in range(12):
                p = client.pod(pod_id)
                if p is None or p.get("desiredStatus") in ("TERMINATED", "EXITED"):
                    term_ok = True
                    break
                _sleep(5)
        except Exception as e:
            print(f"WARNING: terminate raised: {e}")
        if not term_ok:
            print(f"WARNING: could not confirm pod {pod_id} is terminated; check the RunPod console now")
        cost = cost_hr * wall_h
        ledger.record_end(pod_id, round(cost, 4))
        print(f"pod {pod_id} terminated={term_ok} wall={wall_h * 60:.1f}min cost~${cost:.3f}")
        for s, h in old.items():
            signal.signal(s, h)


# ---------------------------------------------------------------- CLI

def _cmd_prices(client: RunPod, key: str | None) -> int:
    types = client.gpu_types()
    keys = [key] if key else list(RUNPOD_CANDIDATES)
    print(f"{'key':8} {'id':36} {'GB':>4} {'community':>10} {'secure':>8}  stock")
    for k in keys:
        for name in RUNPOD_CANDIDATES[k]:
            t = match_gpu_type(types, name)
            if not t:
                print(f"{k:8} {name:36} not in live list")
                continue
            lp = t.get("lowestPrice") or {}
            print(f"{k:8} {t['id'][:36]:36} {t.get('memoryInGb') or 0:>4} "
                  f"{t.get('communityPrice') or '-':>10} {t.get('securePrice') or '-':>8}  {lp.get('stockStatus')}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m infra.runpod", description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="op", required=True)
    sub.add_parser("balance")
    p = sub.add_parser("prices")
    p.add_argument("--gpu", default=None)
    sub.add_parser("pods")
    p = sub.add_parser("terminate")
    p.add_argument("pod_id")
    p = sub.add_parser("launch")
    p.add_argument("--exp", required=True)
    p.add_argument("--gpu", default=None)
    p.add_argument("--cmd", default=None)
    p.add_argument("--cloud", choices=["any", "community", "secure"], default="any")
    p.add_argument("--failsafe-min", type=int, default=None)
    p.add_argument("--allow-long", action="store_true")
    p.add_argument("--max-price", type=float, default=None)
    p.add_argument("--build-llama", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)

    client = RunPod()
    try:
        if a.op == "balance":
            print(json.dumps(client.balance(), indent=1))
        elif a.op == "prices":
            return _cmd_prices(client, a.gpu)
        elif a.op == "pods":
            for x in client.pods():
                print(json.dumps(x))
        elif a.op == "terminate":
            client.terminate(a.pod_id)
            print(f"terminate sent for {a.pod_id}; pod now: {client.pod(a.pod_id)}")
        else:
            man = load_manifest()
            fs = a.failsafe_min
            if fs is None:
                fs = int((man.get(a.exp) or {}).get("failsafe_min") or 40)
            if fs > 40 and not a.allow_long:
                print("error: --failsafe-min > 40 requires --allow-long (capped at 90)")
                return 2
            return launch(a.exp, a.gpu, cmd=a.cmd, cloud=a.cloud, failsafe_min=fs,
                          build_llama=True if a.build_llama else None, max_price=a.max_price,
                          dry_run=a.dry_run, allow_long=a.allow_long, manifest=man, client=client)
    except RuntimeError as e:
        print(f"error: {e}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
