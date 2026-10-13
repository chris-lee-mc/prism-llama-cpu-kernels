"""exps.common: pod deadline budget, ncu replay mode bookkeeping, partial result snapshots."""
import json
import os
import time

import pytest

from exps import common
from harness import counters


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for k in ("CR_START_EPOCH", "CR_FAILSAFE_MIN"):
        monkeypatch.delenv(k, raising=False)


def test_budget_without_pod_clock():
    b = common.Budget(10)
    assert 595 < b.left() <= 600 and not b.expired()
    assert common.Budget(None).left() == float("inf")
    assert common.Budget(None).timeout(None) is None and common.Budget(None).timeout(50) == 50


def test_budget_capped_by_pod_deadline(monkeypatch):
    # 40 min failsafe, 35 min already used by clone and bootstrap: 5 min minus the 300 s reserve = nothing left
    monkeypatch.setenv("CR_START_EPOCH", str(time.time() - 35 * 60))
    assert common.Budget(31).expired()
    # 10 min used: 30 min - 5 min reserve = 25 min < the 31 min asked for
    monkeypatch.setenv("CR_START_EPOCH", str(time.time() - 10 * 60))
    b = common.Budget(31)
    assert 24.9 * 60 < b.left() <= 25 * 60
    assert common.Budget(5).limit == 300.0   # a smaller ask wins
    # no explicit minutes: the deadline alone bounds the run
    assert 24.9 * 60 < common.Budget(None).left() <= 25 * 60
    monkeypatch.setenv("CR_FAILSAFE_MIN", "12")
    assert common.Budget(31).expired()


def test_budget_timeout_is_bounded_by_what_is_left():
    b = common.Budget(10)
    assert b.timeout(1500) == pytest.approx(600, abs=2)
    assert b.timeout(100) == 100
    assert b.timeout(reserve_s=60) == pytest.approx(540, abs=2)
    assert common.Budget(0.001).timeout(1500, reserve_s=60) == 1.0   # never below 1 s


def test_parse_passes():
    text = ('==PROF== Connected to process 1\n'
            '==PROF== Profiling "cr_gemv" - 0: 0%....50%....100% - 3 passes\n'
            '==PROF== Profiling "cr_gemv" - 1: 0%....50%....100% - 1 pass\n"cr_gemv",5 passes\n')
    assert common.parse_passes(text) == 3
    assert common.parse_passes("no profiler lines") is None


class _Ncu:
    """Stand-in for counters.run_ncu: records kwargs, writes the passes line to csv_out."""

    def __init__(self, results):
        self.results, self.calls = list(results), []

    def __call__(self, argv, tokens, **kw):
        self.calls.append(kw)
        res = self.results.pop(0)
        if kw.get("csv_out"):
            os.makedirs(os.path.dirname(kw["csv_out"]), exist_ok=True)
            with open(kw["csv_out"], "w") as f:
                f.write('==PROF== Profiling "k" - 0: 0%%....100%% - %d passes\n' % res.pop("passes_in_log", 2))
        return dict({"ok": True, "error": None, "rows": [1], "summary": None, "timed_out": False}, **res)


def test_ncu_run_application_replay_records_mode_and_passes(monkeypatch):
    fake = _Ncu([{"passes_in_log": 3}])
    monkeypatch.setattr(counters, "run_ncu", fake)
    res = common.ncu_run(["x"], 1, profile_from_start_off=True)
    assert res["replay_mode"] == "application" and res["passes"] == 3
    assert fake.calls[0]["extra"] == ("--profile-from-start", "off", "--replay-mode", "application")
    st = common.ncu_stats(res)
    assert st["ncu_replay_mode"] == "application" and st["ncu_passes"] == 3 and st["ncu_l2_state_ok"] is True


def test_ncu_run_kernel_fallback_can_be_disabled(monkeypatch):
    fail = {"ok": False, "error": "==ERROR== replay failed", "rows": []}
    monkeypatch.setattr(counters, "ncu_path", lambda: "/usr/bin/ncu")
    fake = _Ncu([dict(fail), {"passes_in_log": 4}])
    monkeypatch.setattr(counters, "run_ncu", fake)
    res = common.ncu_run(["x"], 1)
    assert res["replay_mode"] == "kernel" and res["passes"] == 4 and res["app_replay_error"] == fail["error"]
    assert "--replay-mode" not in fake.calls[1]["extra"]
    st = common.ncu_stats(res)
    assert st["ncu_l2_state_ok"] is False and st["ncu_app_replay_error"]    # kernel replay, 4 passes on a warm L2
    # E1/E3: no fallback, the failed application replay is returned as is
    fake = _Ncu([dict(fail)])
    monkeypatch.setattr(counters, "run_ncu", fake)
    res = common.ncu_run(["x"], 1, kernel_fallback=False)
    assert res["ok"] is False and res["replay_mode"] == "application" and len(fake.calls) == 1


def test_ncu_run_kernel_replay_single_pass_is_fine(monkeypatch):
    monkeypatch.setattr(counters, "run_ncu", _Ncu([{"passes_in_log": 1}]))
    res = common.ncu_run(["x"], 1, app_replay=False)
    assert res["replay_mode"] == "kernel" and common.replay_info(res)["ncu_l2_state_ok"] is True
    assert common.replay_info(None) == {"ncu_replay_mode": None, "ncu_passes": None, "ncu_l2_state_ok": None}


def test_ncu_run_keeps_requested_csv_and_removes_temp(monkeypatch, tmp_path):
    fake = _Ncu([{}, {}])
    monkeypatch.setattr(counters, "run_ncu", fake)
    common.ncu_run(["x"], 1, app_replay=False)
    assert not os.path.exists(fake.calls[0]["csv_out"])
    out = tmp_path / "raw" / "ncu.txt"
    common.ncu_run(["x"], 1, app_replay=False, csv_out=str(out))
    assert out.exists()


def test_partial_snapshot_then_final_result(monkeypatch, tmp_path):
    monkeypatch.setenv("CR_RESULTS_DIR", str(tmp_path))
    args = type("A", (), {"gpu": "rtx5090"})()
    common.write_partial("e9", args, {"c": 1}, [{"x": float("nan"), "y": 2}], ["n"], False, extra={"fit": None})
    p = tmp_path / "e9" / "rtx5090" / "raw" / "partial.json"
    doc = json.loads(p.read_text())
    assert doc["partial"] is True and doc["data"]["rows"] == [{"x": None, "y": 2}] and doc["data"]["fit"] is None
    common.write_partial("e9", args, {"c": 1}, [1, 2], [], False)
    assert len(json.loads(p.read_text())["data"]["rows"]) == 2
    assert not [f for f in p.parent.iterdir() if f.name.startswith(".")]   # temp file renamed away
    final = common.write_result("e9", args, {"c": 1}, [1, 2], [], True)
    assert final.parent == tmp_path / "e9" / "rtx5090" and final.exists() and not p.exists()


def test_term_exits_turns_sigterm_into_systemexit():
    import signal
    old = signal.getsignal(signal.SIGTERM)
    with common.term_exits():
        with pytest.raises(SystemExit) as ei:
            os.kill(os.getpid(), signal.SIGTERM)
            time.sleep(1)
        assert ei.value.code == 128 + signal.SIGTERM
    assert signal.getsignal(signal.SIGTERM) == old
