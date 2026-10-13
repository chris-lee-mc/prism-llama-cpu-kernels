import json
import os
import pathlib
import signal
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from exps import common
from exps import e5_spec as e5

TIMINGS = {"cache_n": 0, "prompt_n": 30, "prompt_ms": 20.0, "predicted_n": 192, "predicted_ms": 1500.0,
           "predicted_per_token_ms": 7.8, "predicted_per_second": 128.0, "draft_n": 160, "draft_n_accepted": 120}
RESP = {"index": 0, "content": "x", "tokens_predicted": 192, "tokens_evaluated": 30, "stop_type": "limit",
        "truncated": False, "timings": TIMINGS}


def test_server_cmd_baseline_and_draft():
    base = e5.server_cmd("/b", "/t.gguf", 0)
    assert base == ["/b/llama-server", "-m", "/t.gguf", "-ngl", "99", "-fa", "on", "-c", "16384", "-np", "1",
                    "--host", "127.0.0.1", "--port", "8081"]
    d = e5.server_cmd("/b", "/t.gguf", 6, "/d.gguf")
    assert d[:len(base)] == base
    assert d[len(base):] == ["-md", "/d.gguf", "-ngld", "99", "--spec-type", "draft-simple", "--spec-draft-n-max", "6"]
    ng = e5.server_cmd("/b", "/t.gguf", 4, None, spec_type="ngram-simple")
    assert "-md" not in ng and ng[-4:] == ["--spec-type", "ngram-simple", "--spec-draft-n-max", "4"]
    with pytest.raises(ValueError):
        e5.server_cmd("/b", "/t.gguf", 4, None)


def test_completion_body_and_template_body():
    b = e5.completion_body("hi", 192)
    assert b == {"prompt": "hi", "n_predict": 192, "temperature": 0, "top_k": 1, "seed": 42, "cache_prompt": False}
    assert "chat_template_kwargs" not in e5.template_body("p", None)
    assert e5.template_body("p", False)["chat_template_kwargs"] == {"enable_thinking": False}


def test_parse_completion_fixture():
    r = e5.parse_completion(RESP)
    assert r["tokens_predicted"] == 192 and r["predicted_per_second"] == 128.0
    assert r["draft_n"] == 160 and r["draft_n_accepted"] == 120
    plain = e5.parse_completion({"tokens_predicted": 10, "timings": {"predicted_per_second": 50.0, "predicted_ms": 200.0}})
    assert plain["draft_n"] == 0 and plain["draft_n_accepted"] == 0


def test_summarize():
    rows = [e5.parse_completion(RESP), e5.parse_completion(dict(RESP, timings=dict(TIMINGS, predicted_per_second=64.0,
                                                                                     draft_n=40, draft_n_accepted=0))), {"error": "x", "draft_n": 0, "draft_n_accepted": 0, "tokens_predicted": 0}]
    s = e5.summarize(rows, 4, 12.5)
    assert s["acceptance"] == pytest.approx(120 / 200)
    assert s["tok_s_mean"] == pytest.approx(96.0) and s["tok_s_median"] == pytest.approx(96.0)
    assert s["n_prompts"] == 3 and s["n_ok"] == 2 and s["wall_s"] == 12.5 and s["k"] == 4
    assert e5.summarize([e5.parse_completion({"tokens_predicted": 1, "timings": {"predicted_per_second": 5.0}})], 0, 1)["acceptance"] is None


def test_summarize_error_row_without_draft_fields():
    ok = e5.parse_completion(RESP)
    s = e5.summarize([ok, {"id": "p001", "category": "plan", "error": "URLError: timed out"}], 4, 1.0)
    assert s["n_prompts"] == 2 and s["n_ok"] == 1 and s["acceptance"] == pytest.approx(120 / 160)


def test_draft_rejection_log_detection():
    assert e5.log_says_draft_rejected("error: draft model vocab must match target model to use speculation")
    assert not e5.log_says_draft_rejected("CUDA out of memory")


class _H(BaseHTTPRequestHandler):
    log = []

    def log_message(self, *a):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        _H.log.append((self.path, body))
        if self.path == "/apply-template":
            kw = body.get("chat_template_kwargs")
            out = {"prompt": "<u>" + body["messages"][0]["content"] + ("<think></think>" if kw else "")}
        else:
            out = RESP
        data = json.dumps(out).encode()
        self.send_response(200)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def test_run_prompts_against_fake_server(monkeypatch):
    srv = HTTPServer(("127.0.0.1", 0), _H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    monkeypatch.setattr(e5, "PORT", srv.server_address[1])
    _H.log.clear()
    prompts = [{"id": "p000", "category": "shell", "prompt": "a"}, {"id": "p001", "category": "plan", "prompt": "b"}]
    rows, info = e5.run_prompts(prompts, 192, common.Budget(None), [], 4, "draft-simple")
    srv.shutdown()
    assert [r["id"] for r in rows] == ["p000", "p001"] and all(r["draft_n"] == 160 for r in rows)
    assert info["thinking_disabled"] is True and info["template_supports_enable_thinking"] is True
    comp = [b for p, b in _H.log if p == "/completion"]
    assert comp[0]["prompt"] == "<u>a<think></think>" and comp[0]["temperature"] == 0 and comp[0]["cache_prompt"] is False


def test_run_prompts_flags_unused_draft(monkeypatch):
    srv = HTTPServer(("127.0.0.1", 0), _H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    monkeypatch.setattr(e5, "PORT", srv.server_address[1])
    monkeypatch.setitem(RESP, "timings", dict(TIMINGS, draft_n=0, draft_n_accepted=0))
    prompts = [{"id": "p%03d" % i, "category": "shell", "prompt": "a"} for i in range(5)]
    rows, info = e5.run_prompts(prompts, 8, common.Budget(None), [], 4, "draft-simple")
    srv.shutdown()
    assert info["draft_not_used"] is True and len(rows) == 3


def test_run_prompts_survives_error_rows_and_failing_on_row(monkeypatch):
    srv = HTTPServer(("127.0.0.1", 0), _H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    monkeypatch.setattr(e5, "PORT", srv.server_address[1])
    monkeypatch.setitem(RESP, "timings", dict(TIMINGS))
    real = e5.post_json

    def flaky(url, body, timeout=300.0):
        if url.endswith("/completion") and body["prompt"].startswith("<u>b"):
            raise TimeoutError("completion timed out")
        return real(url, body, timeout)
    monkeypatch.setattr(e5, "post_json", flaky)
    seen, notes = [], []

    def on_row(rows):
        seen.append(len(rows))
        e5.summarize(rows, 4, 1.0)
        raise OSError("disk full")
    prompts = [{"id": "p%03d" % i, "category": "shell", "prompt": c} for i, c in enumerate("aba")]
    rows, info = e5.run_prompts(prompts, 8, common.Budget(None), notes, 4, "draft-simple", on_row)
    srv.shutdown()
    assert seen == [1, 2, 3] and "error" in rows[1] and "draft_n" not in rows[1]
    assert len([n for n in notes if "partial save failed" in n]) == 1
    assert e5.summarize(rows, 4, 1.0)["n_ok"] == 2


def test_args_and_exp_names():
    assert e5.EXP_NAMES == {0: "e5_base", 4: "e5_k4", 6: "e5_k6", 8: "e5_k8"}
    a = e5.parse_args(["--k", "8"])
    assert (a.limit, a.n_predict) == (200, 192)
    with pytest.raises(SystemExit):
        e5.parse_args(["--k", "5"])


def test_dry_run_plan(capsys):
    assert e5.main(["--k", "4", "--dry-run"]) == 0
    plan = json.loads(capsys.readouterr().out)
    assert plan["n_prompts"] == 200 and "--spec-type" in plan["server_cmd"] and plan["fallback_cmd"] is not None


def _alive(pid):
    try:
        return pathlib.Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[0] != "Z"
    except OSError:
        return False


def test_server_stays_in_our_process_group_and_stops(tmp_path):
    srv = e5.Server([sys.executable, "-c", "import time; time.sleep(60)"], str(tmp_path / "s.log"))
    try:
        assert os.getpgid(srv.proc.pid) == os.getpgid(0)   # a group kill of the runner child reaches it
        assert srv.proc.poll() is None
    finally:
        srv.stop()
    assert srv.proc.poll() is not None


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="PR_SET_PDEATHSIG is Linux only")
def test_server_dies_with_a_killed_parent(tmp_path):
    exp_dir = pathlib.Path(e5.__file__).resolve().parent.parent
    code = ("import sys, time\nsys.path.insert(0, %r)\nfrom exps import e5_spec\n"
            "s = e5_spec.Server([sys.executable, '-c', 'import time; time.sleep(60)'], %r)\n"
            "print(s.proc.pid, flush=True)\ntime.sleep(60)\n") % (str(exp_dir), str(tmp_path / "p.log"))
    parent = subprocess.Popen([sys.executable, "-c", code], stdout=subprocess.PIPE, text=True)
    child = int(parent.stdout.readline())
    try:
        assert _alive(child)
        parent.kill()
        parent.wait()
        for _ in range(50):
            if not _alive(child):
                break
            time.sleep(0.1)
        assert not _alive(child)
    finally:
        try:
            os.kill(child, signal.SIGKILL)
        except OSError:
            pass


def _flaky_server(monkeypatch, fail_prompts, timings):
    srv = HTTPServer(("127.0.0.1", 0), _H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    monkeypatch.setattr(e5, "PORT", srv.server_address[1])
    monkeypatch.setitem(RESP, "timings", timings)
    real = e5.post_json

    def flaky(url, body, timeout=300.0):
        if url.endswith("/completion") and any(body["prompt"].startswith("<u>" + f) for f in fail_prompts):
            raise TimeoutError("transient")
        return real(url, body, timeout)
    monkeypatch.setattr(e5, "post_json", flaky)
    return srv


def test_draft_not_used_needs_three_successful_rows(monkeypatch):
    srv = _flaky_server(monkeypatch, ["a", "b", "c"], dict(TIMINGS, draft_n=0, draft_n_accepted=0))
    prompts = [{"id": "p%03d" % i, "category": "shell", "prompt": c} for i, c in enumerate("abcdefg")]
    rows, info = e5.run_prompts(prompts, 8, common.Budget(None), [], 4, "draft-simple")
    srv.shutdown()
    assert [("error" in r) for r in rows] == [True, True, True, False, False, False]   # stops after 3 ok rows, not 3 rows
    assert info["draft_not_used"] is True


def test_all_error_rows_do_not_flag_draft(monkeypatch):
    srv = _flaky_server(monkeypatch, ["a", "b", "c", "d"], dict(TIMINGS, draft_n=0, draft_n_accepted=0))
    prompts = [{"id": "p%03d" % i, "category": "shell", "prompt": c} for i, c in enumerate("abcd")]
    rows, info = e5.run_prompts(prompts, 8, common.Budget(None), [], 4, "draft-simple")
    srv.shutdown()
    assert len(rows) == 4 and all("error" in r for r in rows) and info["draft_not_used"] is False


def test_e5_cost_cmds_rows_and_break_even():
    from exps import e5_cost as ec
    c = ec.target_cmd("/b", "/m.gguf", [1, 5, 9], 512, 5)
    assert c[c.index("-p") + 1] == "1,5,9" and c[c.index("-n") + 1] == "0" and c[c.index("-d") + 1] == "512"
    d = ec.draft_cmd("/b", "/d.gguf", 512, 5)
    assert d[d.index("-p") + 1] == "0" and d[d.index("-n") + 1] == "128"
    rows = ec.rows_from([{"n_prompt": 5, "n_gen": 0, "avg_ts": 500.0}, {"n_prompt": 0, "n_gen": 128, "avg_ts": 400.0}], "x", 512)
    assert rows[0]["kind"] == "verify" and rows[0]["ms_per_batch"] == pytest.approx(10.0)
    assert rows[1]["kind"] == "decode" and rows[1]["ms_per_batch"] == pytest.approx(2.5)
    # verify of 5 costs 2x one decode, free draft: need E = 2 -> (1 - a^5) / (1 - a) = 2 -> a ~ 0.519
    assert ec.break_even(0.0, {1: 8.0, 5: 16.0}, 8.0, 4) == pytest.approx(0.519, abs=0.003)
    assert ec.break_even(100.0, {1: 8.0, 5: 16.0}, 8.0, 4) is None
    s = ec.summarize([{"model": "pq2", "kind": "verify", "width": w, "ms_per_batch": 8.0 + w} for w in range(1, 10)]
                     + [{"model": "pq2", "kind": "decode", "width": 1, "ms_per_batch": 8.0},
                        {"model": "draft_bf16", "kind": "decode", "width": 1, "ms_per_batch": 2.0}])
    assert s["drafts"]["draft_bf16"]["t_step_ms"][4] == pytest.approx(4 * 2.0 + 13.0) and "draft_q8_0" not in s["drafts"]
    assert s["baseline"]["plain_ms"] == 8.0   # the tg row, not the width-1 prompt path (9.0)
