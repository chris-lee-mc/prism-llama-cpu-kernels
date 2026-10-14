# Cache-Resident LM - Implementation Spec (Phase 0 + Phase 1)

Scope: code for PLAN.md P0.1-P0.6 and E1-E5. Four units built in parallel by different agents, disjoint dirs. Everything below is relative to `experiments/cache-resident/` (call it EXP_DIR). This file is the contract: if a unit needs something from another unit that is not pinned here, it must not invent it; it stubs and documents.

## 0. Global rules

- Units and ownership: `infra/` (P0.1, P0.6, runner, orchestrator), `harness/` (P0.2, P0.3, power, timing), `kernels/` (P0.4, P0.5), `exps/` (E1-E5, projections, manifest). Tests: `tests/test_<unit>_*.py`, owned by that unit. `tests/conftest.py`, `pytest.ini`, `.gitignore` are owned by infra.
- Every unit dir is a Python package (`__init__.py`). All code runs with cwd = EXP_DIR as `python -m <unit>.<module>`. Python >= 3.10 (pods use Ubuntu 22.04 python3.10; local venv is 3.13). Use `from __future__ import annotations`.
- Note: a PyPI package named `kernels` exists; cwd-first sys.path makes ours win. Do not pip install `kernels`.
- ASCII only in all code, comments and docs (no em dash, unicode arrows, x-sign). Concise comments (1-2 lines). Each unit has `tests/test_<unit>_ascii.py` that scans its own dir for bytes > 0x7f.
- Imports at module top may only be stdlib, numpy, torch. GPU-only deps (cuda-python, cupy, pynvml, triton, transformers, huggingface_hub, matplotlib) are imported lazily inside functions so every module imports offline on CPU torch.
- No pod launch, no spend, no git commit by implementing agents. Offline checks only.
- CUDA kernel symbol names start with `cr_` (ncu filters depend on it): `cr_gemv_lut*`, `cr_gemv_fp4*`, `cr_gdn_*`, `cr_attn_*`, `cr_rmsnorm*`, `cr_silu_mul*`, `cr_add*`, `cr_reread*`, `cr_int8_gemv*`.
- NVTX range for measured decode steps: `cr_decode`.

## 1. Env vars (pinned)

| Var | Default | Used by |
|---|---|---|
| RUNPOD_API_KEY | (required for runpod.py) | orchestrator; passed into pod env for self-terminate only |
| RUNPOD_POD_ID | injected by RunPod | pod_runner failsafe; absence = not on RunPod |
| CR_REPO_URL | https://github.com/chris-lee-mc/prism-llama-cpu-kernels | pod startup |
| CR_BRANCH | exp/cache-resident | pod startup |
| CR_EXP | - | experiment name (manifest key) |
| CR_CMD | from manifest | override command |
| CR_FAILSAFE_MIN | 40 | pod lifetime cap, counted from container start |
| CR_LINGER_MIN | 10 | serve results after DONE, then self-terminate (still bounded by failsafe) |
| CR_PORT | 8000 | results HTTP server |
| CR_RESULTS_TOKEN | unset | secret path prefix for the results server; required to serve (set by runpod.py launch) |
| CR_BUILD_LLAMA | 0 | bootstrap builds llama.cpp when 1 |
| CR_LLAMA_BIN | `<repo>/build-cuda/bin` | E4, E5 |
| CR_RESULTS_DIR | `EXP_DIR/results` | results helper, runner |
| CR_GPU_TAG | auto from device name | results helper |
| CR_GIT_SHA | from `git rev-parse HEAD` | results helper fallback |
| CR_NCU | auto search | counters |
| CR_L2_BACKEND | auto (`cuda-python`, then `inline`); `auto` = unset | l2persist |
| CR_REQUIRE_COUNTERS | 0 | runner aborts early with status `no_counters` if ncu probe fails |
| CR_SESSION_CAP_USD | 300 | budget gate |
| CR_ALLOWED_CUDA | unset (opt-in) | runpod.create_pod: passes allowedCudaVersions (e.g. 12.8,12.9,13.0). Leave unset: when set, every candidate showed no capacity, while an unfiltered PRO 6000 Server host worked |
| CR_SESSION_ID | UTC date YYYY-MM-DD | ledger grouping |
| CR_LEDGER | `EXP_DIR/.runpod_ledger.jsonl` | budget gate (gitignored) |
| HF_HOME | `/workspace/hf` if /workspace exists else default | model downloads |

## 2. infra/ (P0.1, P0.6, runner, orchestrator)

Files: `infra/__init__.py, gpus.py, results.py, bootstrap.sh, pod_entry.sh, pod_runner.py, runpod.py, requirements-pod.txt`, plus `tests/conftest.py`, `pytest.ini`, `.gitignore` (ignores `.runpod_ledger.jsonl`, `results/_status/`, `__pycache__/`).

### 2.1 infra/gpus.py
```python
TAGS: dict[str, str]                 # substring of device name -> tag, matched in order
def tag_from_name(name: str) -> str  # see table; else lowercase slug [a-z0-9-]; "" or None -> "cpu"
def peak_dram_gbps(tag: str) -> float | None
RUNPOD_CANDIDATES: dict[str, list[str]]   # cascade key -> ordered RunPod gpuTypeId display names
def cascade(key: str, cloud: str = "any") -> list[tuple[str, str]]  # [(gpu_type_name, "COMMUNITY"|"SECURE")]
```
Tag table (name substring -> tag, peak GB/s): "RTX PRO 6000 Blackwell Max-Q" -> pro6000-maxq 1792; "RTX PRO 6000 Blackwell Server" -> pro6000-server 1792; "RTX PRO 6000 Blackwell" -> pro6000-ws 1792; "RTX 5090" -> rtx5090 1792; "RTX 4090" -> rtx4090 1008; "RTX 3090" -> rtx3090 936; "L40S" -> l40s 864; "RTX 6000 Ada" -> rtx6000ada 960. Order matters (Max-Q before generic PRO 6000).
Cascade keys: `pro6000` = [Workstation, Server, Max-Q Workstation] names "NVIDIA RTX PRO 6000 Blackwell Workstation Edition", "NVIDIA RTX PRO 6000 Blackwell Server Edition", "NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition" (600 W editions first: the E4/E5 baselines are from a 600 W Workstation host; Max-Q is 300 W); `rtx5090` = ["NVIDIA GeForce RTX 5090"]; `rtx4090`, `rtx3090` likewise; `l40s` = ["NVIDIA L40S"]. For each name: COMMUNITY then SECURE (`cloud="any"`), so order is (name1,C),(name1,S),(name2,C)... Names are matched at runtime against the live gpuTypes list (`id` or `displayName`, case-insensitive substring) so renames do not break the cascade.

### 2.2 infra/results.py (P0.6)
```python
def results_root() -> pathlib.Path                    # CR_RESULTS_DIR or EXP_DIR/results
def gpu_tag() -> str                                  # CR_GPU_TAG or tag_from_name(torch/nvidia-smi name) or "cpu"
def env_metadata() -> dict                            # never raises; missing fields -> None
def run_dir(exp: str, gpu: str | None = None) -> pathlib.Path   # results/<exp>/<gpu>/, created
def artifact_path(exp: str, name: str, gpu: str | None = None) -> pathlib.Path  # results/<exp>/<gpu>/raw/<UTCSTAMP>-<name>
def write_result(exp: str, gpu: str | None, payload: dict, root: str | os.PathLike | None = None) -> pathlib.Path
```
- File: `results/<exp>/<gpu>/<UTCSTAMP>.json`, UTCSTAMP = `%Y%m%dT%H%M%SZ`; on collision append `-1`, `-2`. gpu None -> `gpu_tag()`. exp and gpu must match `[a-z0-9_.-]+`.
- JSON: `{"schema": 1, "exp", "gpu", "utc" (ISO 8601), "env": env_metadata(), "data": payload}`; `json.dump(..., indent=1, default=str)`, NaN/inf -> null.
- env_metadata keys: gpu_name, gpu_tag, cc ("12.0"), sm ("sm_120"), driver, cuda_runtime, cuda_driver, torch, vram_bytes, sm_count, l2_bytes, persisting_l2_max_bytes, access_policy_window_max_bytes, persisting_l2_limit_bytes, git_sha, git_branch, hostname, pod_id, python, counters_ok (None unless caller sets). Persisting fields come from `harness.l2persist.device_info()` (lazy import, catch all).

### 2.3 Pod side
Startup (dockerArgs) is generated by `runpod.startup_script(env) -> str` (pure, unit-tested). It must, in order: (1) if `/root/.cr_started` exists (container restart) -> self-terminate now; else touch it; (2) start a background failsafe: `sleep CR_FAILSAFE_MIN*60` then self-terminate; (3) `apt-get install -y git curl` if missing, `git clone --depth 1 -b $CR_BRANCH $CR_REPO_URL /workspace/repo`; (4) `exec bash /workspace/repo/experiments/cache-resident/infra/pod_entry.sh`.
Self-terminate = POST `https://api.runpod.io/graphql` with `Authorization: Bearer $RUNPOD_API_KEY`, body `mutation { podTerminate(input: {podId: "$RUNPOD_POD_ID"}) }`; then `kill -TERM 1`.

`infra/bootstrap.sh` (P0.1; idempotent, logs to `results/_status/bootstrap.log`, stamp `/root/.cr_bootstrap_ok`): apt: git curl cmake ninja-build ccache python3-pip python3-venv; pip (`-r infra/requirements-pod.txt`): torch from `--index-url https://download.pytorch.org/whl/cu128`, then cuda-python>=12.8,<13, cupy-cuda12x, nvidia-ml-py, numpy, matplotlib, requests, pytest, transformers, accelerate, huggingface_hub[hf_transfer], triton (if not pulled by torch). ncu: use `command -v ncu`, else `/opt/nvidia/nsight-compute/*/ncu`, else `apt-get install -y cuda-nsight-compute-12-8`; symlink to `/usr/local/bin/ncu`. nsys likewise (`command -v nsys`, `/usr/local/cuda*/bin/nsys`, else `apt-get install -y cuda-nsight-systems-12-8`; optional, r1 part c needs it: the first R1 pod had none). If CR_BUILD_LLAMA=1: `CC=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d .)`; `cmake -S <repo> -B <repo>/build-cuda -G Ninja -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=$CC -DCMAKE_BUILD_TYPE=Release -DLLAMA_CURL=OFF -DCMAKE_CUDA_COMPILER_LAUNCHER=ccache`; `cmake --build build-cuda -j --target llama-bench llama-cli llama-server llama-speculative-simple` (ggml maps 120 -> 120a itself). Must also run on a non-RunPod host (GCP): skip apt when not root, print what is missing. Target: < 4 min without llama build.

`infra/pod_entry.sh`: run bootstrap, then `exec python3 infra/pod_runner.py --exp "$CR_EXP"` (cmd from CR_CMD if set).

`infra/pod_runner.py` CLI (also the way to run on GCP/any CUDA host, no RunPod dependency):
```
python3 infra/pod_runner.py --exp NAME [--cmd STR] [--failsafe-min 40] [--linger-min 10] [--port 8000] [--serve|--no-serve] [--manifest exps/manifest.json]
```
- Serve default: on iff RUNPOD_POD_ID set. Server: `http.server.ThreadingHTTPServer` serving CR_RESULTS_DIR on 0.0.0.0:port, started first.
- Auth: the proxy is public, so serving requires env CR_RESULTS_TOKEN (launch sets `secrets.token_urlsafe(24)` per launch). Unset and serving -> error exit 2 (use `--no-serve`; local no-serve mode is unchanged). Only paths under `/<token>/` answer (constant-time compare); everything else, including `/`, is 404, so no listing outside it. 404 also for `*.nsys-rep`, `*.sqlite`, `*.qdrep` and for text files < 10 MB containing the value of RUNPOD_API_KEY or HF_TOKEN (binaries are not scanned). The token is removed from the child env, logged only as `ab***`, and never written to result JSON.
- Writes `results/_status/STATUS.json` `{exp, gpu, phase: "bootstrap"|"running"|"uploading"|"done", started_utc, updated_utc, pid}` on each phase change.
- Runs cmd via `bash -lc` with cwd EXP_DIR, env + {CR_EXP, CR_GPU_TAG, CR_RESULTS_DIR, CR_LLAMA_BIN, CR_EXP_DIR}; new process group; tee to `results/<exp>/<gpu>/logs/<UTCSTAMP>-runner.log`. Child timeout = failsafe - 3 min; on timeout kill the group, status `timeout`.
- If CR_REQUIRE_COUNTERS=1: run `python -m harness.counters --probe` first; failure -> status `no_counters`, skip cmd.
- After cmd: write `results/_status/MANIFEST.json` `{"files": [{"path", "size", "sha256"}]}` for every file under `results/<exp>/` (paths relative to results root), then `results/_status/DONE.json` `{exp, gpu, status: "ok"|"failed"|"timeout"|"no_counters", exit_code, cmd, started_utc, finished_utc}`. DONE is written last.
- Then linger `--linger-min` serving, then self-terminate if RUNPOD_POD_ID else exit with child exit code. In-process failsafe thread mirrors the bash one (belt and braces).

### 2.4 infra/runpod.py (orchestrator, local, uses `requests`)
```python
class RunPod:  # all HTTP goes through self._post(query: str, variables: dict) -> dict (mockable)
    def balance(self) -> dict                       # myself { clientBalance currentSpendPerHr spendLimit }
    def gpu_types(self) -> list[dict]               # gpuTypes { id displayName memoryInGb secureCloud communityCloud securePrice communityPrice lowestPrice(input:{gpuCount:1}){ uninterruptablePrice stockStatus } }
    def create_pod(self, *, name, gpu_type_id, cloud, image, env: dict, docker_args: str, disk_gb: int, ports="8000/http") -> str  # podFindAndDeployOnDemand; returns pod id
    def pod(self, pod_id) -> dict | None            # pod(input:{podId}) { id desiredStatus costPerHr runtime { uptimeInSeconds } }
    def terminate(self, pod_id) -> None             # podTerminate; idempotent
def budget_gate(balance: float, price_hr: float, failsafe_min: int, session_spent: float, cap: float, margin: float = 2.0) -> tuple[bool, str]
def est_cost(price_hr: float, failsafe_min: int) -> float   # price_hr * (failsafe_min + 5) / 60
class Ledger:  # JSONL at CR_LEDGER
    def record_launch(self, pod_id, exp, gpu, gpu_type_id, cloud, price_hr) -> None
    def record_end(self, pod_id, cost_usd) -> None
    def session_spent(self, session_id: str | None = None) -> float   # ended costs + est_cost of open pods
def startup_script(env: dict) -> str
def proxy_url(pod_id: str, port: int = 8000, token: str | None = None) -> str   # https://<id>-<port>.proxy.runpod.net/[<token>/]
def launch(exp: str, gpu: str | None = None, *, cmd=None, cloud="any", failsafe_min=40, build_llama=None, max_price=None, dry_run=False) -> int
```
- Gate rule: refuse if `balance < est` (balance < est + 2.0 only warns; the account auto-tops-up) or `session_spent + est > CR_SESSION_CAP_USD` or `price_hr > max_price` (defaults: rtx3090 0.50, rtx4090 0.80, rtx5090 1.30, pro6000 2.60, l40s 1.20). Price = communityPrice or securePrice for the chosen cloud; skip candidates with no price/stock.
- Launch flow: manifest entry -> gate per candidate in cascade order -> create (on "no longer any instances available"-type error, try next) -> ledger launch -> `try:` wait `desiredStatus RUNNING` and proxy reachable (<= 15 min) -> poll `proxy_url(token)/_status/DONE.json` every 20 s until present or deadline `failsafe_min + 2` min (print STATUS.json phase changes) -> fetch MANIFEST.json, download each file to local CR_RESULTS_DIR preserving path, verify sha256 -> `finally:` terminate, confirm `pod()` is None or TERMINATED, ledger end with cost = costPerHr * wall time. SIGINT goes through finally.
- Pod params: image `nvidia/cuda:12.8.1-devel-ubuntu22.04`, disk from manifest (`disk_gb`), ports `8000/http`, env = {RUNPOD_API_KEY, CR_RESULTS_TOKEN (random per launch; shown as <redacted> in dry-run), CR_EXP, CR_CMD?, CR_FAILSAFE_MIN, CR_LINGER_MIN, CR_BUILD_LLAMA, CR_BRANCH, CR_REPO_URL, HF_HUB_ENABLE_HF_TRANSFER=1}, name `cr-<exp>-<gpu>-<HHMM>`.
- CLI: `python -m infra.runpod balance | prices [--gpu KEY] | pods | terminate POD_ID | launch --exp E [--gpu KEY] [--cmd S] [--cloud any|community|secure] [--failsafe-min N] [--max-price X] [--build-llama] [--dry-run]`. `--failsafe-min` > 40 requires `--allow-long` and is capped at 90. `--dry-run` prints candidates, prices, gate decision, startup script; creates nothing.

### 2.5 infra tests (offline)
`test_infra_results.py` (tmp root, monkeypatched metadata, filename/collision/schema, no GPU), `test_infra_gpus.py` (tag table incl. Max-Q ordering, cascade order), `test_infra_runpod.py` (mock `_post`: gate pass/fail cases, cascade fallback on no-capacity, terminate always called on exception and on success, ledger math, startup_script contains failsafe, restart marker, clone of CR_BRANCH, no secrets printed), `test_infra_runner.py` (run pod_runner with `--cmd "python -c ..."` writing a result, `--no-serve`; MANIFEST+DONE content; serve on ephemeral port and GET <token>/_status/DONE.json; mock-server test: 404 without/with wrong token, 200 with token, 404 for denied names and secret-bearing files; refuses to serve without token; failsafe of 0.02 min kills `sleep 60` child, status timeout, no network call without RUNPOD_POD_ID), `bash -n` on the .sh files, ascii.
On-GPU acceptance (later): one `launch --exp p0_smoke --gpu rtx5090` end to end: results downloaded, pod terminated, ledger cost recorded; cold start to running cmd < 5 min.

## 3. harness/ (P0.2, P0.3, power, timing)

Files: `harness/__init__.py, l2persist.py, csrc/l2persist_ext.cpp, p0_l2_check.py, counters.py, power.py, timing.py`.

### 3.1 harness/l2persist.py
```python
def backend() -> str                                   # "cuda-python" | "inline"; CR_L2_BACKEND forces
def device_info(dev: int = 0) -> dict                  # {name, cc, l2_bytes, persisting_max_bytes, window_max_bytes, persisting_limit_bytes, backend}
def max_persisting_bytes(dev: int = 0) -> int          # cudaDevAttrMaxPersistingL2CacheSize
def max_window_bytes(dev: int = 0) -> int              # cudaDevAttrMaxAccessPolicyWindowSize
def get_limit() -> int                                 # cudaDeviceGetLimit(PersistingL2CacheSize)
def set_limit(nbytes: int | None = None) -> int        # clamp to max; returns get_limit() after set
def set_stream_window(ptr, nbytes: int | None = None, hit_ratio: float = 1.0, stream=None) -> dict
def clear_stream_window(stream=None) -> None           # num_bytes=0, hitProp Normal
def reset() -> None                                    # cudaCtxResetPersistingL2Cache
@contextlib.contextmanager
def persist(ptr, nbytes=None, hit_ratio=1.0, stream=None, limit=None)  # set_limit, set window; on exit clear, reset, set_limit(0)
def apply_window_to_graph(graph, ptr, nbytes: int, hit_ratio: float = 1.0) -> int
def log_device(dev: int = 0) -> dict                   # prints one line, returns device_info
```
- `ptr` is a torch tensor (data_ptr, nbytes from numel*element_size) or an int device pointer (then nbytes required). nbytes clamped to `max_window_bytes`. Returned dict: `{base_ptr, num_bytes, hit_ratio, hit_prop: "persisting", miss_prop: "streaming", clamped: bool}`. `stream`: torch.cuda.Stream, int handle, or None (current stream). hitProp Persisting, missProp Streaming, as PLAN section 8.
- `apply_window_to_graph`: `graph` is a `torch.cuda.CUDAGraph` built with `keep_graph=True` (use `graph.raw_cuda_graph()`), or an int `cudaGraph_t`. Enumerate `cudaGraphGetNodes`, for every `cudaGraphNodeTypeKernel` call `cudaGraphKernelNodeSetAttribute(node, cudaKernelNodeAttributeAccessPolicyWindow, value)`. Returns count. Caller then calls `graph.instantiate()` before `replay()`. Alternative path (also supported, decided on GPU in P0.2): call `set_stream_window` on the capture stream before capture; stream attributes are copied into captured kernel nodes.
- Without CUDA: functions raise `RuntimeError("no CUDA device")`; import never fails.
- Inline fallback: `csrc/l2persist_ext.cpp` (torch/extension.h + cuda_runtime.h, pybind11 functions with int64 ptr/stream args mirroring the API), built via `torch.utils.cpp_extension.load(name="cr_l2persist_ext", ...)`.

### 3.2 harness/p0_l2_check.py (Phase 0 exit check)
`python -m harness.p0_l2_check [--mb 64] [--reps 100] [--hit-ratio 1.0] [--no-ncu] [--inner off|on] [--method all|eager|graph-nodes|graph-stream] [--ncu-timeout 300] [--ncu-budget 900]`
- Buffer of `--mb` MiB fp32; reread kernel `cr_reread` (cupy RawKernel; float4 loads, block partial sums) launched `--reps` times on one stream. Window covers the buffer, clamped to persisting max (record the clamp).
- `--method all` (default) repeats the check for eager (stream window), graph-nodes (window on graph kernel nodes) and graph-stream (stream window set before capture, then cleared, so only the window copied into the nodes counts). One verdict per method under `methods`, plus `methods_honored` (passed, non-control). Graph methods record the window read back from a kernel node. Top-level `conditions` / `verdict` are those of the first method; `pass` is true only when every method passes.
- Outer mode: for window off/on, time with CUDA events (always) and run itself under ncu (`--inner off|on`) with `kernel_regex="cr_reread"`, `launch_skip=5`. Writes `write_result("p0_l2", None, {...})` with `device_info`, both conditions (us per rep, hit_rate, dram_bytes per rep, replay_mode, ncu_cmd), `pass`. A timing-only result (`partial: true`) is written before ncu starts and replaced by the final file; each ncu call is limited by `--ncu-timeout`, all together by `--ncu-budget`.
- Pass rule: `hit_on - hit_off >= 0.20` and `dram_on <= 0.5 * dram_off` when `persisting_max >= 0.75 * buffer`; else `"control": true` and pass means `abs(hit_on - hit_off) < 0.05 + persisting_max / buffer` (a working window holds up to persisting_max bytes, so that share of re-reads hits legitimately; a plain 0.05 bound fails a correct 3090 run). If counters unavailable, decide on timing (on >= 1.3x faster) and set `counters_ok: false`. Counters taken with kernel replay are flagged (`counters_basis`, `counters_suspect`).

### 3.3 harness/counters.py (P0.3)
```python
METRICS = ["lts__t_sectors_lookup_hit.sum", "lts__t_sectors_lookup_miss.sum", "dram__bytes_read.sum", "gpu__time_duration.sum"]
def ncu_path() -> str | None
def build_ncu_cmd(argv: list[str], *, kernel_regex=None, nvtx_include=None, launch_skip=0, launch_count=None, metrics=METRICS, extra=()) -> list[str]
def parse_ncu_csv(text: str) -> list[dict]   # rows: {id, name, hit_sectors, miss_sectors, hit_rate, dram_bytes, time_ns}
def summarize(rows: list[dict], tokens: int) -> dict   # {hit_rate, dram_bytes_per_token, us_per_token, n_kernels, dram_bytes_total, time_ns_total}
def bucket(rows: list[dict], buckets: dict[str, str]) -> dict  # name -> {time_ns, dram_bytes, hit_rate, count}; unmatched -> "other"; first regex wins
def run_ncu(argv, tokens: int, *, kernel_regex=None, nvtx_include=None, launch_skip=0, launch_count=None, env=None, timeout=1800, csv_out: str | None = None) -> dict   # ncu runs in its own session; timeout or SIGTERM kills the whole group; result also has timed_out
def probe() -> dict   # profiles a tiny kernel; {"ok": bool, "error": str | None}
```
- Fixed ncu flags: `--csv --page raw --target-processes all --cache-control none --clock-control none --metrics <list>`. `--cache-control none` is mandatory (default flushes L2 and kills the effect we measure). Kernel filter: `-k regex:<re> --kernel-name-base function`. NVTX: `--nvtx --nvtx-include "cr_decode/"`.
- CSV: skip ncu banner lines (`==PROF==`, `==WARNING==`), skip the units row, strip thousands separators, convert units (bytes, Kbyte, Mbyte, Gbyte; ns, usecond, msecond) to bytes and ns. hit_rate = hit/(hit+miss), None if 0.
- run_ncu result: `{"ok", "error", "cmd", "rows", "summary", "csv_path"}`. On `ERR_NVGPUCTRPERM` in output: ok False, error "ERR_NVGPUCTRPERM". Doc string must state: counters need host `NVreg_RestrictProfilingToAdminUsers=0`; RunPod containers cannot change it; fallback is timing-only plus analytic bytes, results carry `counters_ok: false`; retry on another host or use the GCP PRO 6000.
- CLI: `python -m harness.counters --probe` (exit 0/1, prints JSON), `python -m harness.counters --tokens N [--kernel-regex R] [--nvtx] -- <cmd...>` prints summary JSON.

### 3.4 harness/power.py
```python
class PowerSampler:  # context manager; thread samples nvmlDeviceGetPowerUsage every interval_s
    def __init__(self, dev: int = 0, interval_s: float = 0.010, nvml=None)  # nvml injectable for tests
    def start(self) -> None
    def stop(self) -> dict   # {joules, avg_w, max_w, samples, duration_s, method: "energy_counter"|"trapezoid"}
def tok_per_joule(tokens: int, joules: float) -> float | None
```
Prefer `nvmlDeviceGetTotalEnergyConsumption` delta (mJ) when supported; else trapezoid over samples.

### 3.5 harness/timing.py
```python
def cuda_time_us(fn, iters: int = 100, warmup: int = 10, stream=None) -> float   # mean us per call via CUDA events
def capture_graph(fn, warmup: int = 3, stream=None, keep_graph: bool = False) -> torch.cuda.CUDAGraph
def replay_time_us(graph, iters: int = 100, warmup: int = 5) -> float
```
`capture_graph` with keep_graph=True returns an un-instantiated graph so `apply_window_to_graph` can edit it; caller calls `instantiate()`.

### 3.6 harness tests
Offline: CSV fixtures (with banner, units row, commas, Kbyte/usecond units, two kernels), build_ncu_cmd flags, ERR_NVGPUCTRPERM detection via fake subprocess, bucket/summarize math, PowerSampler with fake nvml (both methods), l2persist imports and raises cleanly without CUDA, window clamp logic tested via a pure helper `_clamp_window(nbytes, max_window)`, `g++ -fsyntax-only` of `csrc/l2persist_ext.cpp` with torch include paths and `/usr/local/cuda-12.8/include`. On-GPU (pod, `-m gpu`): device_info nonzero on sm_89/120; window set/clear round trip; graph node count > 0 and replay works; `p0_l2_check` passes on rtx5090 and reports control on rtx3090.

## 4. kernels/ (P0.4 ternary GEMV, P0.5 toy hybrid)

Files: `kernels/__init__.py, csrc/ternary_gemv.h, csrc/ternary_gemv.cu, csrc/toy_ops.h, csrc/toy_ops.cu, csrc/binding.cpp, build.py, ternary.py, bench_gemv.py, toy_hybrid.py, FP4.md`.

### 4.1 Packed weight layout (byte exact)
- Logical W is [N, K] (N outputs, K inputs), y = W x. Requires `K % 128 == 0`, N >= 1.
- Ternary code per weight, 2 bits: `0b00 -> 0`, `0b01 -> +1`, `0b10 -> -1`, `0b11 -> 0` (never produced). Decode: `t = (c & 1) - (c >> 1)`. Bit planes of a word w: plus = `w & 0x55555555`, minus = `(w >> 1) & 0x55555555`.
- `q`: torch.int32 [N, K/16], C-contiguous, row stride K/4 bytes. Word `q[n, j]` (read as uint32, little endian) holds k = 16j + i at bits `[2i+1 : 2i]`, i = 0..15. Equivalently byte b of row n holds k = 4b..4b+3, weight 4b+i at bits 2i.
- `s`: torch.float16 [N, K/128], C-contiguous; group g covers k in [128g, 128g+128).
- `x`: float16 [K]; `y`: float16 [N]. `y[n] = fp16( sum_g float(s[n,g]) * sum_{k in g} t[n,k] * float(x[k]) )`, FP32 accumulate.
- Alignment: q and x 16 B aligned, s and y 2 B aligned. Kernels must handle s/y unaligned to 16 B.
- `bytes_moved(N, K) = N*K/4 + N*(K/128)*2 + 2*K + 2*N`.
- Golden test (pinned): row t = [+1, -1, 0, +1, then 0 x 124] -> first byte `0b01_00_10_01 = 0x49`, q[0,0] = 0x00000049.

### 4.2 C ABI (csrc/ternary_gemv.h, csrc/toy_ops.h)
All launchers: no host sync, no allocation, no cudaMemset; launch on `st`; return `cudaGetLastError()`. Graph-capturable.
```c
cudaError_t cr_gemv_lut(const uint32_t* q, const __half* s, const __half* x, __half* y, int N, int K, cudaStream_t st);
cudaError_t cr_gemv_fp4(const uint32_t* q, const __half* s, const __half* x, __half* y, int N, int K, cudaStream_t st);
// GDN step, per head h: alpha=sigmoid(ab[h]), beta=sigmoid(ab[H+h]); q,k L2-normalized;
// S_h = alpha*S_h - beta*(alpha*S_h k) k^T + beta v k^T  (== alpha S (I - beta k k^T) + beta v k^T); o_h = S_h q
cudaError_t cr_gdn_step(float* S /*[H,dv,dk]*/, const __half* qkv /*[H*(dk+dk+dv)] as q|k|v blocks*/, const __half* ab /*[2H]*/, __half* o /*[H*dv]*/, int H, int dk, int dv, cudaStream_t st);
// single-query GQA attention over a fixed-length fp16 cache, Kc/Vc layout [T, Hkv, D]
cudaError_t cr_attn_decode(const __half* q /*[Hq*D]*/, const __half* Kc, const __half* Vc, __half* o /*[Hq*D]*/, int Hq, int Hkv, int D, int T, float scale, cudaStream_t st);
cudaError_t cr_rmsnorm(const __half* x, const __half* w, __half* y, int n, float eps, cudaStream_t st);
cudaError_t cr_silu_mul(const __half* g, const __half* u, __half* y, int n, cudaStream_t st);
cudaError_t cr_add(__half* a, const __half* b, int n, cudaStream_t st);   // a += b
```
Declared `extern "C"`. Kernel (`__global__`) names begin with the launcher name (e.g. `cr_gemv_lut_kernel<...>`).
- Variant (a) LUT: per block, build in shared memory a LUT of partial sums over small groups of x (e.g. 16-entry table per 2-weight nibble, or 81/256 per 4-weight byte), tiled over K; rows in the block index it with the 2-bit fields as stored. A bit-plane/popcount formulation on the same layout is acceptable; the layout may not change.
- Variant (b) FP4: stage tiles of q to shared memory as E2M1 nibbles (0 -> 0x0, +1 -> 0x2, -1 -> 0xA), then decode nibble -> half via a 16-entry table and do the dot in FP32. On sm_120 an `mma ... kind::f8f6f4` / block-scaled FP4 path may be added behind `#if defined(CR_FP4_MMA)` (built for `sm_120a` only, x quantized to e4m3, tolerance documented separately); default build never uses it. `FP4.md` states honestly what the variant is and whether MMA helped at batch 1.

### 4.3 Python API (kernels/ternary.py, kernels/build.py)
```python
QK = 128
def pack(t: torch.Tensor) -> torch.Tensor                     # int8 {-1,0,1} [N,K] -> int32 [N,K/16]; pure torch, CPU or CUDA
def unpack(q: torch.Tensor, K: int) -> torch.Tensor           # -> int8 [N,K]
def quantize_ternary(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]  # absmean per g128: s=mean|w| (min 1e-8), t=round(clamp(w/s,-1,1)); -> (q, s fp16)
def random_ternary(N: int, K: int, *, density: float = 2/3, seed: int = 0, device="cpu") -> tuple[q, s, t]  # s ~ U(0.5,1.5)/sqrt(K)
def dequant(q, s, K: int, dtype=torch.float32) -> torch.Tensor  # [N,K]
def ref_gemv(q, s, x) -> torch.Tensor                          # fp32 [N]; CPU or CUDA
def bytes_moved(N: int, K: int) -> int
def gemv_lut(q, s, x, out=None) -> torch.Tensor                # fp16 [N] on current stream
def gemv_fp4(q, s, x, out=None) -> torch.Tensor
def gemv_ptr(variant: str, q_ptr: int, s_ptr: int, x_ptr: int, y_ptr: int, N: int, K: int, stream: int = 0) -> None  # variant "lut"|"fp4"
# kernels/build.py
def load_ext(verbose: bool = False)   # torch.utils.cpp_extension.load(name="cr_ternary_ext", sources=[binding.cpp, ternary_gemv.cu, toy_ops.cu]), -O3 -std=c++17, -gencode for current device cc; CR_FP4_MMA=1 adds sm_120a + define
```
Extension (pybind) functions, all ints are raw pointers / stream handles: `gemv(variant:int 0=lut 1=fp4, q, s, x, y, N, K, stream)`, `gdn_step(S, qkv, ab, o, H, dk, dv, stream)`, `attn_decode(q, Kc, Vc, o, Hq, Hkv, D, T, scale, stream)`, `rmsnorm(x, w, y, n, eps, stream)`, `silu_mul(g, u, y, n, stream)`, `add_(a, b, n, stream)`. Each raises RuntimeError on non-success cudaError.
Correctness tolerance (pinned): `allclose(y.float(), ref, rtol=2e-3, atol=2e-3 * ref.abs().max())`.

### 4.4 kernels/bench_gemv.py
`python -m kernels.bench_gemv [--shapes 4096x4096,8192x8192,4096x16384,16384x4096,28672x8192] [--variants lut,fp4] [--iters 200] [--result-exp p0_gemv]` (shape = NxK). Cold timing rotates over a pool of distinct weight copies totalling >= 4x L2 size (L2 from torch device props); hot timing reuses one copy. Rows: `{N, K, variant, bytes, us_cold, us_hot, gbps_cold, gbps_hot, pct_peak_cold (vs infra.gpus.peak_dram_gbps, None if unknown), max_abs_err, ok}`. Pass target on rtx5090: `pct_peak_cold >= 0.70` at 4096x4096 for the best variant. With `--result-exp`, calls `infra.results.write_result(exp, None, {...})` (lazy import).

### 4.5 kernels/toy_hybrid.py (P0.5)
`python -m kernels.toy_hybrid [--layers 24] [--d 1024] [--attn-every 4] [--H 16] [--dk 128] [--dv 128] [--Hq 8] [--Hkv 2] [--D 256] [--F 3584] [--ctx 4096] [--tokens 200] [--variant lut|fp4] [--result-exp p0_toy]`. Defaults mirror Qwen3.5-0.8B text config (24 layers, every 4th full attention).
- Layer: x -> rmsnorm -> mixer -> add -> rmsnorm -> MLP (gate, up ternary [F,d]; silu_mul; down ternary [d,F]) -> add. GDN mixer: qkv proj ternary [H*(2dk+dv), d], ab proj ternary [2H, d] (pad rows ok), cr_gdn_step, out proj [d, H*dv]. Attention mixer: q proj [Hq*D, d], kv proj [2*Hkv*D, d] (computed, not appended; cache is fixed length ctx, random fp16), cr_attn_decode, o proj [d, Hq*D]. All weights in one contiguous arena (256 B aligned offsets) so one access-policy window can cover them.
- `class ToyHybrid: __init__(cfg, device, variant); step() -> None` (one token, raw-pointer launches on current stream); `weight_arena() -> torch.Tensor`; `state_arena() -> torch.Tensor` (GDN states contiguous).
- CPU references `ref_gdn_step`, `ref_attn_decode` (fp32 torch) used by tests.
- Measures: `{us_per_token_eager, us_per_token_graph, us_per_layer_graph, us_per_layer_gdn, us_per_layer_attn, launches_per_token, weight_bytes_per_token, eff_gbps}`; per-type us/layer from homogeneous all-GDN and all-attn models with the same L.

### 4.6 kernels tests
Offline: nvcc compile of `ternary_gemv.cu` and `toy_ops.cu` with `/usr/local/cuda-12.8/bin/nvcc -std=c++17 -O3 -c -gencode arch=compute_XX,code=sm_XX` for XX in 86, 89, 120 (marker `nvcc`, skip if missing); `g++ -fsyntax-only` of binding.cpp against torch + CUDA includes; pack/unpack round trip incl. non-square, K in {128, 384, 4096}; golden byte test; ref_gemv vs dense fp32 matmul on CPU; bytes_moved; quantize_ternary idempotent on ternary input; ref_gdn_step / ref_attn_decode shape and a hand-checked tiny case. On-GPU: gemv_lut/gemv_fp4 vs ref_gemv for shapes {(1,128),(7,384),(4096,4096),(11008,4096),(4096,11008)}; gemv_ptr inside a captured CUDA graph equals eager; toy step graph == eager output; bench target on rtx5090.

## 5. exps/ (E1-E5, projections, manifest)

Files: `exps/__init__.py, manifest.json, common.py, e1_sweep.py, plot_e1.py, e2_hybrid_bf16.py, e3_residency.py, int8_head.py, e4_bonsai.py, e5_spec.py, make_prompts.py, data/prompts_e5.jsonl, proj_hybrid.py, hf_ids.py, run_p0.sh`.

### 5.1 Runner contract (exps <-> infra)
`exps/manifest.json`: `{ "<exp>": {"cmd": str, "gpus": [cascade keys], "est_min": int, "failsafe_min": 40, "build_llama": bool, "disk_gb": int, "desc": str} }`. infra reads it (`launch --exp`, `pod_runner --exp`); `cmd` runs via `bash -lc` in EXP_DIR with env from section 1. Commands must exit nonzero on failure, write all outputs under `results/<exp>/<gpu>/` (via `infra.results`), and finish within `failsafe_min - 5`.
Common driver flags (`exps/common.py` helpers): `--dry-run` (no CUDA; print plan JSON; exit 0), `--quick` (tiny sizes for smoke), `--no-ncu`, `--gpu TAG` (override tag). Each invocation writes one result JSON: `data = {"config": {...}, "rows": [...], "notes": [...], "counters_ok": bool}`.

| exp | cmd | gpus | est_min | build_llama | disk_gb |
|---|---|---|---|---|---|
| p0_smoke | `bash exps/run_p0.sh` (p0_l2_check, bench_gemv, toy_hybrid, pytest -m gpu) | rtx5090, rtx4090, rtx3090 | 25 | no | 40 |
| p0_l2 | `python -m harness.p0_l2_check` | rtx5090, rtx3090, pro6000 | 8 | no | 40 |
| e1 | `python -m exps.e1_sweep` | rtx5090, pro6000, rtx3090 | 30 | no | 40 |
| e2_08b / e2_2b | `python -m exps.e2_hybrid_bf16 --model 0.8B` / `--model 2B` | rtx5090 | 30 | no | 60 |
| e3 | `python -m exps.e3_residency` | rtx5090 | 30 | no | 60 |
| e4_pq2 / e4_ptq1 | `python -m exps.e4_bonsai --quant PQ2_0` / `PTQ1_0` | pro6000, rtx5090 | 35 | yes | 80 |
| e4_pdl | `python -m exps.e4_pdl` (PDL 0/1, pre-wait L2 prefetch A/B and a PDL-off prefetch control, test-backend-ops check; failsafe 70, launch with `--allow-long`) | pro6000, rtx5090 | 43 | yes | 80 |
| e5_base, e5_k4, e5_k6, e5_k8 | `python -m exps.e5_spec --k 0|4|6|8` | pro6000 | 30 | yes | 80 |
| e5_cost | `python -m exps.e5_cost` (target verify cost for widths 1-9 at depth 512, plain target tg, draft decode cost and draft batch cost for widths 1-9 (the per-step re-eval), break-even alpha and r per K) | pro6000 | 10 | yes | 80 |
| r1 | `python -m exps.fetch_models --models pq2,draft_bf16,draft_q8_0 --timeout-min 6 && python -m exps.r1 --budget-min 29` (E5 matrix, draft agreement, nsys step split; failsafe 55 is required and launch needs `--allow-long`; est_min 35 = downloads 6 + r1 29, on top come ~14 for apt, torch and the cold sm_120 build (850 s measured) and the 5 min end reserve) | pro6000 | 35 | yes | 80 |
| r2a | `python -m exps.fetch_models --models pq2,draft_bf16 --timeout-min 6 && python -m exps.r2a_longctx --budget-min 29` (long-context agentic prompts: matrix, same-pod nsys overhead control, -lv 5 rollback probe; failsafe 55, launch needs `--allow-long`; est_min 35 = downloads 6 + r2a 29) | pro6000 | 35 | yes | 80 |

### 5.2 E1 working-set sweep (exps/e1_sweep.py)
Revised after the first run (pollution-subtraction design was invalid). `--W 8,16,24,32,48,56,64,80,96,128,192,256,512,1024` (MiB) `--C 512` `--modes off,0.5,0.75,1.0,1.0L` `--tokens 200` `--rounds 5` `--variant lut|fp4`. One token = hot GEMVs over W ([4096 x 4096] ternary layers, one 256 B aligned arena) then cold GEMVs over C (27B MLP shapes 17408x5120, 17408x5120, 5120x17408 cycled, own arena), all P0.4 `gemv_ptr`, one CUDA graph of up to 2000 kernel nodes. Every GEMV reads a fixed random x (outputs stay finite; last hot and cold outputs checked against `ref_gemv`). "<r>": `set_limit(max)`, window over the hot arena (clamped to window max) with hitRatio r on the graph kernel nodes via `apply_window_to_graph`; "<r>L": limit min(W, max). Modes timed in interleaved rounds; row per (W, C in {C, 0}, mode): `{W_mib, C_mib, L, L_cold, hot_bytes, cold_bytes, mode, hit_ratio, limit_policy, window_bytes, persist_limit_bytes, us_per_token(_all), hot_us(_all), cold_us(_all), delta_vs_off_us, gain_vs_off, hot_delta_vs_off_us, hot_gain_vs_off, *_delta_noise_us, finite, check_ok}`. hot_us/cold_us come from 1-token hot and cold graphs replayed in turn with events between. Config holds a calibration (copy GB/s, bench_gemv on the hot and cold shapes) and a W=1024 chain diag (fixed x vs ping-pong). Stop rule: rtx5090 / pro6000 best hot gain >= 20% for some W <= persisting max (C > 0 rows). No ncu (no counters on RunPod).
`exps/plot_e1.py [--results results/e1] [--out results/e1/e1_knee.png]`: newest hot/cold JSON per gpu; rows: hot us/token vs W (log2 x) with C streamed, hot us/token with C=0, hot gain % vs W; one line per mode; rtx3090 dashed grey overlay; vertical line at each GPU's persisting max.

### 5.3 E2 stock hybrid BF16 (exps/e2_hybrid_bf16.py, exps/hf_ids.py)
`hf_ids.py` pins ids verified on 2026-10-13 via `https://huggingface.co/api/models?search=Qwen3.5`: `QWEN = {"0.8B": "Qwen/Qwen3.5-0.8B", "2B": "Qwen/Qwen3.5-2B"}` (also exist: -Base variants, 4B, 9B, 27B); `BONSAI2_GGUF = ("prism-ml/Ternary-Bonsai-2-27B-gguf", {"PQ2_0": "Ternary-Bonsai-2-27B-PQ2_0.gguf" (7.21 GB), "PTQ1_0": "Ternary-Bonsai-2-27B-PTQ1_0.gguf" (5.95 GB)})`; `DRAFT_GGUF = ("ggml-org/Qwen3.5-0.8B-GGUF", "Qwen3.5-0.8B-BF16.gguf")` fallback `unsloth/Qwen3.5-0.8B-GGUF`. `def resolve(kind, key) -> dict` re-checks via HF API at run time (falls back to pinned value offline) and records what exists.
Config facts (0.8B): hidden 1024, 24 layers (18 linear_attention GDN + 6 full_attention), GDN 16 heads x 128 x 128 fp32 state = 18.9 MB total, attn 8 q / 2 kv heads, head_dim 256, vocab 248320, tied embeddings. HF arch `Qwen3_5ForConditionalGeneration`; load text-only (AutoModelForCausalLM, fallback language_model submodule), record the class used.
Run: `--model 0.8B|2B --ctx 0,4096,16384,32768,65536 --decode 512`. Prefill random token ids to ctx, then greedy decode 512 with `torch.compile(mode="reduce-overhead")` (static cache if supported; else fall back to `mode="default"` then eager; record `compile_mode_used`). ncu pass: 8 decode steps inside NVTX `cr_decode`. Row: `{model, ctx, tok_s, us_per_token, compile_mode_used, dram_bytes_per_token, l2_hit_rate, analytic: {weight_bytes, kv_bytes, gdn_state_bytes}, split: {weights, kv, gdn_state}}` where split.kv = bytes(ctx) - bytes(0), split.gdn_state = analytic 2x state (read+write), split.weights = bytes(0) - gdn_state. Fit across ctx: `us_per_token = a + bytes/bw` -> `fit: {floor_us_per_layer: a/n_layers, eff_bw_gbps}` (feeds proj_hybrid).

### 5.4 E3 GDN state + LM head residency (exps/e3_residency.py, exps/int8_head.py)
Model 0.8B, `--ctx 0,16384`, `--heads bf16_full,int8_full,int8_48k,bf16_48k`, conditions `off, gdn, head, both`. One contiguous arena = [GDN states | head weights]; HF cache state tensors are re-pointed to views of the arena (record `gdn_persist_method`; if impossible, use a manual decode loop). "both" = one window over both regions (single window per kernel). Head bytes: bf16_full 508.6 MB, int8_full 254.3 MB, int8_48k 50.3 MB (49152 x 1024), bf16_48k 100.7 MB. Note in results: PLAN's "48K-pruned INT8 98 MB" matches bf16_48k (or int8 at hidden 2048), so both are run. Pruned ids = 0..49151 (measurement-only, documented). `int8_head.py`: per-row symmetric INT8, Triton GEMV kernel named `cr_int8_gemv` (torch fallback), replaces lm_head only. Row: `{ctx, head, condition, window_bytes, hit_ratio, us_per_token, dram_bytes_per_token, l2_hit_rate, persisted_bytes}`. Go/no-go metric in notes: DRAM bytes saved / persisted bytes for `both` with int8_48k.

### 5.5 E4 Bonsai 2 27B (exps/e4_bonsai.py)
`--quant PQ2_0|PTQ1_0 --depths 0,32768,131072,262144 --kv f16,q8_0 --ncu-depths 0,32768`. Downloads GGUF with `huggingface_hub.hf_hub_download`. Bench cmd (pinned flags from tools/llama-bench): `$CR_LLAMA_BIN/llama-bench -m M -ngl 99 -fa on -p 0 -n 128 -d D -ctk T -ctv T -ub 2048 -r R -o json` with R=3 for D <= 32768 else 1. ncu: depth 0 -> `-p 0 -n 4 -r 1 --no-warmup`, all kernels, tokens=4; depth > 0 -> count matching launches in a `-n 0` run, then `--launch-skip` that count with the regex below, tokens=4. Buckets: `{"hadamard": "fwht|hadamard", "act_quant": "quantize_q8_1|quantize_mmq", "mul_mat_vec": "mul_mat_vec", "flash_attn": "flash_attn", "gated_delta_net": "gated_delta|gdn"}`; note that PQ2_0/PTQ1_0 unpack is fused inside mul_mat_vec. If a depth fails (OOM, time), record and continue. Row: `{quant, depth, kv, tg_tok_s, us_per_token, buckets, dram_bytes_per_token, bw_bound_ratio: dram_bytes_per_token / us_per_token / peak}`.

`exps/fetch_models.py` downloads the E4/E5 GGUFs in parallel so chained drivers on one pod do not spend their budget on downloads. Without ncu counters, E4 takes the kernel split from nsys (cr_decode_prof, profiler-API capture range, graph nodes traced; under PDL a kernel's duration includes its wait at the grid dependency sync, so bucket sums exceed wall time).

### 5.6 E5 speculative baseline (exps/e5_spec.py, exps/make_prompts.py)
`make_prompts.py` writes `data/prompts_e5.jsonl` deterministically (no RNG beyond `random.Random(1234)`): 200 lines `{"id": "p000".., "category": one of code_write|code_fix|code_explain|shell|tool_call|json_extract|plan|refactor, "prompt": str}`, 25 per category; file is committed; test checks sha256 is stable across two generations.
`e5_spec.py --k 0|4|6|8 [--limit 200] [--n-predict 192]`: start `$CR_LLAMA_BIN/llama-server -m <PQ2_0 gguf> -ngl 99 -fa on -c 16384 -np 1 --port 8081` plus, for k > 0, `-md <draft gguf> -ngld 99 --spec-type draft-simple --spec-draft-n-max K`; wait `/health`; per prompt: `/apply-template` (disable thinking if the template supports it, record) then POST `/completion` `{prompt, n_predict, temperature: 0, top_k: 1, seed: 42, cache_prompt: false}`; read `timings.predicted_per_second`, `timings.draft_n`, `timings.draft_n_accepted`, `tokens_predicted`. Row per prompt plus summary `{k, acceptance = sum accepted / sum drafted, tok_s_mean, tok_s_median, wall_s}`. If the server rejects the draft (vocab mismatch; llama.cpp allows <= 128 vocab-size difference), record `draft_incompatible` and rerun with `--spec-type ngram-simple` clearly labelled.

### 5.7 exps/proj_hybrid.py (re-implementation)
Header: "Re-implementation of the proposal's projection; original proj.py/proj_hybrid.py not in repo." Roofline: `us_token = n_layers*floor_us + dram_bytes/bw_dram + l2_bytes/bw_l2`, with `l2_bytes = min(persisted, persist_max) * hit`, `dram_bytes = weight + head + state_rw + kv(ctx) - l2_bytes`. Speculation: `E[tokens/verify] = (1 - a^(k+1)) / (1 - a)`, `tok_s = E / (k*t_draft + t_verify(k+1))`.
```python
@dataclass class ModelSpec: name, n_layers, weight_bytes, head_bytes, state_bytes, kv_bytes_per_ctx_token
@dataclass class HwSpec: name, bw_dram_gbps, bw_l2_gbps, floor_us_per_layer, persist_max_bytes
MODELS: dict[str, ModelSpec]   # qwen3.5-0.8b, qwen3.5-2b, ternary-0.8b-g128, bonsai2-27b-pq2_0
def decode_us(m, hw, ctx=0, persisted_bytes=0, hit=1.0) -> dict  # {us, dram_bytes, l2_bytes, floor_us}
def spec_tok_s(a, k, t_draft_us, t_verify_us) -> float
def hw_from_results(path) -> HwSpec   # e2 fit or p0_toy json
```
CLI: `python -m exps.proj_hybrid --floor-us F --bw-gbps B [--from-results DIR] [--models ...] [--ctx 0,32768] [--write-result]`.

### 5.8 exps tests
Offline: manifest schema (keys, gpus are cascade keys, failsafe <= 40 except e4_pdl at 70, r1 and r2a at 55, modules for exps commands exist); every driver `--dry-run` exits 0 without CUDA; prompts determinism, 200 lines, unique ids; llama-bench cmd builder exact flags; llama-bench JSON fixture parsing; llama-server `/completion` JSON fixture parsing (timings fields); counters-bucket config; proj_hybrid numbers on a hand-computed case (bytes/bw identity, spec formula at a=0 gives 1/(k*td+tv)); plot_e1 on synthetic results to a tmp png (Agg backend). On-GPU: each manifest command with `--quick` on its target GPU.

### 5.9 R0 / R1: acceptance bookkeeping and the pre-E6 session (exps/spec_math.py, e5_cost.py, e5_spec.py, r1*.py)
Acceptance has two numbers. `r` = llama.cpp `draft_n_accepted / draft_n`. `alpha` = per-token acceptance of the independence model: a step that drafts K tokens accepts A = alpha + ... + alpha^K and emits 1 + A tokens, so `r = A(alpha, K) / K` (`spec_math.alpha_from_r` inverts it by bisection; K may be fractional = mean drafted per step). Steps of a run = tokens - accepted - n_requests (the first token of a request comes from prefill, not from a decode step). r falls with K at fixed alpha, so compare alpha across K. E5 K=4 (r 0.216) is alpha 0.477. alpha is reported for draft-simple only: the n-gram types draft variable lengths (often 0), so their rows carry r and tokens per step, alpha null.
R0 (`e5_cost`): the plain decode baseline is the target's own `llama-bench -p 0 -n 128 -d 512` row (kind `decode`), not the width-1 prompt path (8.64 ms against 7.19 ms). Rows carry `samples_ts` (per-rep) and use their median. Step model: draft-simple (common/speculative.cpp) decodes the whole K + 1 verify batch on the draft context in `process()` and then runs K single-token draft decodes, so `t_step(K) = K t_draft + t_draft_batch(K + 1) + t_verify(K + 1)`; the draft rows now include `llama-bench -p W -n 0 -d 512` for W = 1..9 (kind `verify`, model `draft_*`). Summary: `baseline`, `target_verify_ms`, per draft `t_draft_ms`, `t_draft_batch_ms`, `reeval_included`, `break_even[K] = {alpha, r}` (with the re-eval term), `break_even_no_reeval`, `t_step_ms`, `t_step_ms_no_reeval`, `free_draft_break_even`. `--resummarize RESULT.json --plain-from E5_BASE.json` rebuilds it offline into `RESULT.resummary.json` (old runs have only avg_ts, no target tg row and no draft width rows, so the baseline is the E5 target-alone server rate and `reeval_included` is false; stated in the file). With that baseline and no re-eval term the K=4 BF16 draft needs alpha 0.63 (r 0.36), not 0.53; the re-eval term raises it further once measured.
R1 (`exps.r1`, one result `results/r1/<gpu>/<stamp>.json` with `data.agree`, `data.e5`, `data.step`, `data.verdict`; budget 29 min split 6 : 26 : 5 over parts a, b, c, run in the order b, a, c, unused time rolls forward; the pod must fit one failsafe of 55 min: bootstrap with the cold build ~14 min (850 s measured, `results/e5_cost/pro6000-ws/logs/*-bootstrap.log`), downloads at most 6 (`fetch_models --timeout-min 6` ends a stalled download), r1 29, `common.deadline_left_s` end reserve 5; if the downloads eat their whole timeout the deadline caps r1 at about 28.8 min and the plan below still holds; est_min 35 = downloads 6 + r1 29, and `test_manifest_r1_fits_the_failsafe` / `test_k2_k3_are_opt_in_and_the_pod_budget_decides_criterion_3` check that timeline):
- b `e5_spec.matrix_configs()`: prompts = `make_prompts.shuffled(seed 20261013, stratify=True)` (every prefix category balanced). The prompt count N is fixed before any server starts by `e5_spec.plan_n`: the largest prefix whose summed expected cost (25 s startup + prompt_s per prompt, prompt_s from E5: target alone 1.2 s, K=4 1.9 s) fits 90% of the part budget. Criterion 3 does not depend on N: base and the four n-gram configs also run `code_extra` prompts, the code_fix / refactor prompts after the prefix up to `--code-floor` (default 20, i.e. 15 plus spare for failed prompts) per code category, and the plan counts their cost. For the 29 min budget N is about 110 (14 per category) plus 12 extra code prompts, 20 per code category; without the floor 29 min gives N 121 with exactly 15 per code category when k2/k3 are dropped, and 92 with 11 when they are kept (the 37 min plan of the first draft did not fit the pod at all). Base runs the N prompts plus the extras; every other config runs a subset of what base finished. One llama-server per config, untimed warm-up request, greedy /completion, text kept; a requested `-bs` is verified (the response's `generation_settings.backend_sampling` and the server log are checked and noted): `base`; `k4` draft-simple K=4 exactly as in E5 (control); `k4_tbs` = k4 plus `-bs` (target backend sampling; `--backend-sampling`, off by default); `k2`, `k3` on half the prompts (opt-in through `--configs`, not in the default: they cost about 20% of part b, so they wait for a follow-up pod); `k4_ptq1` K=4 with the post-hoc PTQ1_0 draft on half the prompts (real alpha drop for criterion 2); `ngram_mod`, `ngram_map_k` (fork defaults) and `ngram_mod_short`, `ngram_map_k_short` on code_fix and refactor only (plus the extra code prompts), so that those cells get n >= 15. Rollback path: `need_n_rs_seq()` (common.h) returns draft.n_max only for draft-simple, mtp, eagle3, dflash and dspark and 0 for ngram-mod and ngram-map-k, so the target context of an n-gram server has `n_rs_seq = 0`, `common_context_can_seq_rm` returns FULL for the hybrid (GDN) Bonsai target, every drafting step saves a target checkpoint and every partial acceptance restores it and replays the target pass (server-context.cpp); draft-simple (n_rs_seq = K) rolls back in the batch and pays none of this. The server logs the path at TRC level only, so each config's log (head and tail, the startup lines are far from the end) is searched for the INFO line `n_rs_seq = N` of llama-context.cpp (target context first, draft context second) and `checks` gets `n_rs_seq`, `n_rs_seq_target`, `rollback_path` (none, in_batch, checkpoint, unknown) and `checkpoint_replay`; `summary[cfg].steps_count_replays` says that tokens/step and ms/step count each replay pass as a step; a note names every n-gram config on that path. Fork facts (common.h, arg.cpp, speculative.cpp): `--spec-draft-backend-sampling` is read only by the eagle3, dflash and mtp types; draft-simple always samples its draft with `common_sampler_sample` on the CPU, so a K=4 pair that differs only in that flag would be the same server (tests guard this). ngram-mod defaults n_match 24, n_min 48, n_max 64; ngram-map-k size_n 12, size_m 48, min_hits 1; ngram-mod keeps its token pool across requests (cleared at 25% occupancy, seeded by the warm-up), so results depend on prompt order and the code-only configs build a different pool than a full run; recorded in `data.e5.state_note`, not removed. Summary per config and category, on prompts both runs finished: tok/s, speedup with a 95% bootstrap interval (prompts resampled, `speedup_ci95`), r, alpha (draft-simple), tokens/step, ms/step, text equality and common-prefix fraction; `data.e5.pairs` holds k4_tbs against k4 and k4_ptq1 against k4 on their common prompts; `step_residual` per config with and without the draft re-eval term.
- a `r1_agree`: before the 27B pass, n_vocab of the target and draft GGUFs is read with gguf-py; a mismatch skips the variant (and the pass when all mismatch), because `llama-perplexity --kl-divergence` needs equal n_vocab. Primary text `e5_chat`: the chat-templated E5 prompts (`prompt_text` of the base rows) followed by the target-alone greedy outputs, about 48000 chars, i.e. the distribution alpha is measured on (llama-perplexity tokenizes special tokens as plain text and scores prompt tokens too; `out_char_frac` is recorded). Second text `repo`: raw upstream code with canned narration (no chat template), run when 150 s remain. `llama-perplexity -c 2048 -b 2048 --chunks 4 --kl-divergence-base` on Bonsai 2 27B PQ2_0, then `--kl-divergence` for the 0.8B BF16, Q8_0 and a PTQ1_0 made once on the pod with `llama-quantize` from the BF16 file (shared with `k4_ptq1`; log, tensor types, errors recorded; quantize_ptq1_0 ignores the imatrix and scales by absmax). Same top p = teacher-forced top-1 agreement on the second half of each 2048 window. The ~2 GB log-prob file and quantized GGUF live in temp dirs and are deleted.
- c `r1_step`: server (plain `k4`, or `--step-config k4_tbs`) under `nsys profile --start-later=true --session-new=cr_r1 --cuda-graph-trace=node`; warm-up request, `nsys start`, one request of 56 tokens, `nsys stop`, SIGTERM to the server, sqlite export, parse, delete. The partial result (request timings) is saved before the export. nsys and the server get an allowlisted environment (PATH, LD_LIBRARY_PATH, HOME, TMPDIR = the private 0700 work dir, LANG, LC_*, CUDA_*, NVIDIA_*, GGML_*, LLAMA_*; secret-looking names dropped even then); `nsys cancel` and a group kill run in the finally block; `*.nsys-rep`, `*.sqlite*`, `*.qdrep`, `*.qdstrm` are in `.gitignore` and on the results-server deny list. Per step (verify start to verify start): activity is split by stream role (target main stream = longest burst, streams mostly inside its active spans are target branch streams, the rest the draft): target verify, draft re-eval (first draft burst, the server runs `common_speculative_process` right after verify), K draft token steps, snapshot / restore / readback / small copies, idle (intra-burst, after verify, between drafts, before verify), c1_removable and c1_floor. With one stream only the duration split is possible (`split_mode: duration`), which merges an abutting re-eval into the verify burst. `data.step.wall_check` compares the median step wall with the request's own timing and with part b's unprofiled k4 ms/step and flags a gap above 10% (node-level graph tracing adds launch overhead); `verdict.c1_overhead` carries the floor also with that overhead removed.
- Host health: throttled clocks, or a power cap without `--allow-power-cap`, skip parts b and c (their timings would be wrong) but part a still runs; the exit code is 1. With `--allow-power-cap` the cap is recorded in `verdict.host`. A SIGTERM or an exception in a part still writes the result (`try/finally`), with a note naming the cut part; notes, config and doc are scrubbed of secret values.
`verdict` holds the inputs of the E6 criteria. c1: `c1_floor` <= 2 ms (and with profiler overhead removed) plus `k4_tbs_gain_vs_k4` with its interval and whether the server reported backend sampling on. c2: BF16 agreement >= 0.5 on the E5 chat text, next to the part b K=4 alpha as a calibration of the proxy (`calibration.teacher_forced_bf16_minus_alpha`), the repo-text values, and PTQ1_0 within 0.05 measured as the alpha of `k4_ptq1` against `k4` on common prompts (basis `e5_k4`; the KL proxy only when that run has fewer than 15 prompts). c3 (`ngram_decision`): a cell (config x code category) decides only with n >= 15 and an interval; `no_go` when some interval lower bound is >= 1.15, `go` when every cell is decided, no lower bound reaches 1.15 and no point estimate does, `straddles` when a point estimate reaches 1.15 but its interval includes less, `insufficient_n` otherwise; the raw maximum is reported but never decides. `ngram_decision` also carries `rollback` per n-gram config, `checkpoint_replay`, `speedup_is` ("lower bound on n-gram speedup" on the checkpoint path) and `provisional`: the n-gram speedups are measured WITH the replay penalty, so they are lower bounds. A `no_go` stands (the true speedup is at least that), but a `go` is provisional (`provisional: true`, `provisional_note`) until n-gram is measured with rollback, for example with `need_n_rs_seq` extended to the n-gram types in a test build; the same holds when no `n_rs_seq` line was found. Nothing in R1 has run on a GPU yet; the pod flow is tested against fake binaries only.

### 5.10 R2a: long-context agentic prompts (exps/r2a_longctx.py, make_prompts --longctx)
Prompt set `exps/data/prompts_r2a.jsonl` (60 rows, `python -m exps.make_prompts --longctx`, stats with `--stats`, `--check` reports drift): 20 `edit_full_file` + 20 `edit_diff` (one tracked ASCII source file of 2-8K tokens each, estimated as chars / 3.5, five edit kinds: rename with call sites, add a parameter, described bug, add logging, extract a block) and 20 `toolcall_trace` (tool schema, 3-5 earlier calls with real list_dir / search_code / read_file results, ask for the next call). The file lists are pinned in make_prompts.py, no file is used twice, every row records the sha of its sources, so a later edit of a source file is drift (`drift_rows`), not a silent change. Row: `{id, category, n_predict (512 / 512 / 128), approx_tokens, chars, messages, meta}`.
`exps.r2a_longctx` (result `results/r2a/<gpu>/<stamp>.json`: `data.matrix`, `data.control`, `data.probe`, `data.host`, `data.prompt_set`; budget 29 min split 21.5 : 5.5 : 2 over parts m, c, p; same pod rules as R1). Every server gets `-lv 4` (`-lv N` is in common/arg.cpp; llama-internal INFO lines are trace level in common/log.cpp, so the `n_rs_seq` line and the `common_context_can_seq_rm` lines show at 4). Servers run with `-c 16384 -np 1`.
- m: configs base, k2, k3, k4 (BF16 draft), ngram_mod, ngram_map_k, ngram_mod_short, ngram_map_k_short (R1 parameters), base first and the rest in `random.Random(seed).shuffle` order; prompts `make_prompts.shuffled(seed 20261014, stratify=True)`. N is fixed by `plan_n` (cost model: prefill 2500 tok/s, decode 140 tok/s, 15 s per server start) and cut again after base with the measured time per prompt; a cut drops the tail of the balanced order (so each category loses about the same) and `matrix.plan` / `matrix.replan` / notes list the dropped ids and counts. At the 29 min budget the model gives N about 25 of 60 (8-9 per category); all 60 need about 65 min. Per config, category and overall on prompts paired with base: tok/s, speedup, r, alpha (draft-simple only) with bootstrap 95% intervals (1000 resamples of prompts), greedy equality against base (`text_equal_frac`, `median_prefix_frac`), `n_by_category` and `n_ge_min` (>= 15 per category decides). `checks`: `n_rs_seq` per context, `seq_rm_types` (RS = in-batch rollback, FULL = KV checkpoint plus replay), `spec_uses_checkpoints`, `rollback_path`, `path_consistent`.
- c: K=4 on the first 10 prompts of the balanced order (n_predict 48), unprofiled and then from a fresh server under `nsys profile --start-later` (r1_step.capture with a driver; allowlisted env, private 0700 dir, files deleted). A request is a segment (idle > 150 ms); its decode window starts after the last target burst >= 40 ms (prefill), roles come from the whole capture; `r1_step.analyze` runs per window with skip 1 and the steps are pooled. `overhead`: unprofiled ms/step, nsys step wall, profiler overhead, split with kernel-busy parts, unprofiled non-compute time, `c1_floor` minus the overhead.
- p: `-lv 5` runs of k4, ngram_mod_short, ngram_map_k_short on 3 prompts x 64 tokens (untimed): counts of `created speculative checkpoint`, `restoring speculative checkpoint` and `accepted a/b draft tokens` (partial / full / zero), debug logs deleted.
- Unhealthy host (clocks under load, or power cap without `--allow-power-cap`): m and c skipped, p runs, exit code 1.

## 6. Experiment-to-GPU map and cost

Prices (PLAN, Oct 2026; re-queried by `runpod.py prices` before each launch): rtx5090 0.69 C / 1.19 S, rtx4090 0.34 / 0.74, rtx3090 0.22 / 0.43, pro6000 1.69 C / 2.09 S (Max-Q sometimes ~0.50). Worst-case per run = price x (40 + 5) / 60 (failsafe-capped).

| Run | GPU | Runs | Worst-case $/run (C) | Subtotal (C) |
|---|---|---|---|---|
| p0_smoke | rtx5090 | 3 (iterate) | 0.52 | 1.56 |
| p0_smoke | rtx3090 | 1 | 0.17 | 0.17 |
| p0_l2 | pro6000 | 1 | 1.27 | 1.27 |
| e1 | rtx5090 / rtx3090 / pro6000 | 2 / 1 / 1 | 0.52 / 0.17 / 1.27 | 2.48 |
| e2_08b, e2_2b | rtx5090 | 2 x 2 | 0.52 | 2.08 |
| e3 | rtx5090 | 2 | 0.52 | 1.04 |
| e4_pq2, e4_ptq1 | pro6000 | 2 | 1.27 | 2.54 |
| e5_base, k4, k6, k8 | pro6000 | 4 | 1.27 | 5.08 |
| Total |  | 21 |  | ~16.2 C (~21 S) |

Within PLAN Phase 0+1 budget (~$50). PRO 6000 rows can run on the GCP PRO 6000 via `python3 infra/pod_runner.py --exp E --no-serve` at $0. Default session cap 25 USD.

## 7. Acceptance summary

Offline (all units, now): `cd EXP_DIR && /home/user/venv/bin/python -m pytest -q tests/test_<unit>_*.py` passes; nvcc builds for sm_86/89/120; every module imports on CPU torch; ASCII scan clean.
On GPU (later, pods): p0_smoke on rtx5090 (l2 check pass, GEMV >= 70% peak cold at 4096x4096, toy graph us/token < eager), rtx3090 control, then E1-E5 as in section 6.
