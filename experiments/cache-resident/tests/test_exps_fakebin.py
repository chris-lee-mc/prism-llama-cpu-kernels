"""End-to-end runs of the E4 and E5 drivers against fake llama.cpp binaries (no GPU, no model)."""
import json
import os
import stat
import sys

import pytest

from exps import e4_bonsai as e4
from exps import e5_spec as e5

FAKE_BENCH = """#!{py}
import json, sys
a = sys.argv
depth = int(a[a.index("-d") + 1]); kv = a[a.index("-ctk") + 1]; n = int(a[a.index("-n") + 1])
print("load_backend: fake", file=sys.stderr)
print(json.dumps([{{"n_prompt": 0, "n_gen": n, "n_depth": depth, "type_k": kv, "type_v": kv,
                    "avg_ts": 100.0 - depth / 10000.0, "stddev_ts": 0.5, "model_type": "fake"}}], indent=1))
"""

FAKE_SERVER = """#!{py}
import json, os, sys
from http.server import BaseHTTPRequestHandler, HTTPServer
a = sys.argv
port = int(a[a.index("--port") + 1])
has_draft = "-md" in a and "draft-simple" in a
if has_draft and os.environ.get("FAKE_REJECT"):
    print("error: draft model vocab must match target model to use speculation", flush=True)
    sys.exit(1)
spec = a[a.index("--spec-type") + 1] if "--spec-type" in a else "none"
class H(BaseHTTPRequestHandler):
    def log_message(self, *x): pass
    def _send(self, obj):
        d = json.dumps(obj).encode()
        self.send_response(200); self.send_header("Content-Length", str(len(d))); self.end_headers(); self.wfile.write(d)
    def do_GET(self):
        self._send({{"status": "ok"}})
    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        if self.path == "/apply-template":
            return self._send({{"prompt": "T:" + body["messages"][0]["content"]}})
        t = {{"predicted_per_second": 90.0, "predicted_ms": 2000.0, "predicted_n": 180}}
        if spec != "none":
            t.update(draft_n=100, draft_n_accepted=70 if spec == "draft-simple" else 10)
        self._send({{"tokens_predicted": 180, "timings": t, "stop_type": "limit"}})
HTTPServer(("127.0.0.1", port), H).serve_forever()
"""


def _script(path, text):
    path.write_text(text.format(py=sys.executable))
    path.chmod(path.stat().st_mode | stat.S_IXUSR)


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setenv("CR_RESULTS_DIR", str(tmp_path / "results"))
    monkeypatch.setenv("CR_GPU_TAG", "pro6000-ws")
    monkeypatch.setenv("CR_LLAMA_BIN", str(tmp_path / "bin"))
    (tmp_path / "bin").mkdir()
    return tmp_path


def _results(tmp, exp):
    files = sorted((tmp / "results" / exp / "pro6000-ws").glob("*.json"))
    assert len(files) == 1
    return json.loads(files[0].read_text())


def test_e4_run_with_fake_bench(env, monkeypatch):
    _script(env / "bin" / "llama-bench", FAKE_BENCH)
    monkeypatch.setattr(e4, "download", lambda q, notes: ("/fake.gguf", {"repo": "r", "file": "f"}))
    monkeypatch.setattr(e4.common, "counters_available", lambda a: (False, "ncu skipped for test"))
    assert e4.main(["--quant", "PQ2_0", "--depths", "0,32768", "--kv", "f16,q8_0"]) == 0
    doc = _results(env, "e4_pq2")
    rows = doc["data"]["rows"]
    assert [(r["depth"], r["kv"]) for r in rows] == [(0, "f16"), (0, "q8_0"), (32768, "f16"), (32768, "q8_0")]
    assert rows[0]["tg_tok_s"] == 100.0 and rows[0]["us_per_token"] == pytest.approx(1e4)
    assert rows[0]["dram_bytes_per_token"] is None and doc["data"]["counters_ok"] is False
    assert doc["data"]["config"]["bench_cmd_example"][-1] == "json"


class _RecordNcu:
    def __init__(self):
        self.calls = []

    def __call__(self, argv, tokens, **kw):
        self.calls.append((argv, kw))
        return {"ok": True, "error": None, "timed_out": False, "rows": [], "summary": None, "cmd": [], "csv_path": None}


def _e4_ncu_env(env, monkeypatch, tool):
    from harness import counters
    _script(env / "bin" / "llama-bench", FAKE_BENCH)
    monkeypatch.setattr(e4, "download", lambda q, notes: ("/fake.gguf", {"repo": "r", "file": "f"}))
    monkeypatch.setattr(e4.common, "counters_available", lambda a: (True, None))
    monkeypatch.setattr(e4, "build_prof_tool", lambda bin_dir, notes: tool)
    rec = _RecordNcu()
    monkeypatch.setattr(counters, "run_ncu", rec)
    return rec


def test_e4_bench_fallback_ncu_does_not_use_profile_from_start_off(env, monkeypatch):
    """llama-bench never calls cudaProfilerStart: with --profile-from-start off ncu would profile nothing."""
    from harness import counters
    rec = _e4_ncu_env(env, monkeypatch, None)
    assert e4.main(["--quant", "PQ2_0", "--depths", "0,32768", "--kv", "f16", "--ncu-depths", "0,32768", "--ncu-kv", "f16"]) == 0
    assert len(rec.calls) == 1   # depth 0 only; depth 32768 has no kernel split without the tool
    argv, kw = rec.calls[0]
    assert argv[0].endswith("llama-bench") and argv[argv.index("-n") + 1] == "4" and "--no-warmup" in argv
    assert "--profile-from-start" not in kw["extra"] and kw["extra"] == ()
    assert "--profile-from-start" not in counters.build_ncu_cmd(argv, extra=kw["extra"])
    doc = _results(env, "e4_pq2")
    assert any("no kernel split" in n for n in doc["data"]["notes"])
    assert doc["data"]["rows"][0]["ncu_replay_mode"] == "kernel"


def test_e4_prof_tool_ncu_profiles_from_start_off(env, monkeypatch):
    rec = _e4_ncu_env(env, monkeypatch, env / "bin" / "cr_decode_prof")
    assert e4.main(["--quant", "PQ2_0", "--depths", "0,32768", "--kv", "f16", "--ncu-depths", "0,32768", "--ncu-kv", "f16"]) == 0
    assert len(rec.calls) == 2
    for argv, kw in rec.calls:
        assert argv[0].endswith("cr_decode_prof") and kw["extra"] == ("--profile-from-start", "off")


def test_e4_ncu_target_choice():
    assert e4.ncu_target(None, "/b", "/m", 32768, "f16") is None
    argv, off = e4.ncu_target(None, "/b", "/m", 0, "q8_0")
    assert off is False and argv == e4.bench_ncu_cmd("/b", "/m", "q8_0")
    argv, off = e4.ncu_target("/b/cr_decode_prof", "/b", "/m", 32768, "f16")
    assert off is True and argv == e4.prof_cmd("/b/cr_decode_prof", "/m", 32768, "f16")


def test_e4_timeouts_follow_the_pod_deadline_and_partial_is_written(env, monkeypatch):
    import time
    # 40 min failsafe, 33 min used: 7 min - 5 min reserve = 120 s of budget
    monkeypatch.setenv("CR_START_EPOCH", str(time.time() - 33 * 60))
    monkeypatch.delenv("CR_FAILSAFE_MIN", raising=False)
    rec = _e4_ncu_env(env, monkeypatch, None)
    seen = []
    orig = e4.run_bench
    monkeypatch.setattr(e4, "run_bench", lambda cmd, timeout: seen.append(timeout) or orig(cmd, timeout))
    parts = []
    monkeypatch.setattr(e4.common, "write_partial", lambda exp, args, cfg, rows, notes, ok, extra=None: parts.append(len(rows)))
    assert e4.main(["--quant", "PQ2_0", "--depths", "0,131072", "--kv", "f16"]) == 0
    assert seen and all(t <= 121 for t in seen)           # was budget.left() with a 30 s floor, from a 31 min budget
    assert rec.calls == []                                # < 240 s left: no ncu
    assert parts == [1, 2] and len(seen) == 2


def test_e4_missing_binary(env, capsys):
    (env / "bin").rmdir()
    assert e4.main(["--quick"]) == 2


def test_e5_baseline_and_speculative(env, monkeypatch):
    _script(env / "bin" / "llama-server", FAKE_SERVER)
    monkeypatch.setattr(e5, "load_gguf", lambda kind, key, notes: ("/fake.gguf", {"repo": "r", "kind": kind}))
    assert e5.main(["--k", "0", "--limit", "4"]) == 0
    base = _results(env, "e5_base")["data"]
    assert base["summary"]["acceptance"] is None and base["summary"]["n_ok"] == 4
    assert base["summary"]["tok_s_mean"] == 90.0
    assert e5.main(["--k", "4", "--limit", "4"]) == 0
    k4 = _results(env, "e5_k4")["data"]
    assert k4["summary"]["acceptance"] == pytest.approx(0.7) and k4["summary"]["spec_type"] == "draft-simple"
    assert k4["summary"]["draft_incompatible"] is False
    assert len(k4["rows"]) == 4 and k4["config"]["attempts"][0]["cmd"][-2:] == ["--spec-draft-n-max", "4"]


def test_e5_draft_rejected_falls_back_to_ngram(env, monkeypatch):
    _script(env / "bin" / "llama-server", FAKE_SERVER)
    monkeypatch.setenv("FAKE_REJECT", "1")
    monkeypatch.setattr(e5, "load_gguf", lambda kind, key, notes: ("/fake.gguf", {"repo": "r", "kind": kind}))
    assert e5.main(["--k", "6", "--limit", "3"]) == 0
    data = _results(env, "e5_k6")["data"]
    s = data["summary"]
    assert s["draft_incompatible"] is True and s["spec_type"] == "ngram-simple" and "fallback" in s["label"]
    assert any("draft_incompatible" in n for n in data["notes"])
    assert [a["spec_type"] for a in data["config"]["attempts"]] == ["draft-simple", "ngram-simple"]
    assert s["acceptance"] == pytest.approx(0.1)
    assert os.path.exists(data["config"]["attempts"][0]["log"])


def test_e4_download_failure_still_writes_a_result(env, monkeypatch):
    _script(env / "bin" / "llama-bench", FAKE_BENCH)

    def boom(q, notes):
        raise OSError("no network")
    monkeypatch.setattr(e4, "download", boom)
    monkeypatch.setattr(e4.common, "counters_available", lambda a: (False, "x"))
    assert e4.main(["--quick"]) == 3
    d = _results(env, "e4_pq2")["data"]
    assert d["rows"] == [] and any("download failed" in n for n in d["notes"]) and d["config"]["hf"]["repo"]


def test_e5_download_failure_still_writes_a_result(env, monkeypatch):
    _script(env / "bin" / "llama-server", FAKE_SERVER)

    def boom(kind, key, notes):
        raise OSError("no network")
    monkeypatch.setattr(e5, "load_gguf", boom)
    assert e5.main(["--k", "4", "--limit", "2"]) == 3
    d = _results(env, "e5_k4")["data"]
    assert d["rows"] == [] and d["config"]["draft"]["repo"]
