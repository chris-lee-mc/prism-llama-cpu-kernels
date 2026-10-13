import json
import os
import subprocess
import sys
import types

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from harness import counters  # noqa: E402

HEADER = ('"ID","Process ID","Process Name","Host Name","Kernel Name","Context","Stream","Block Size","Grid Size",'
          '"Device","CC","lts__t_sectors_lookup_hit.sum","lts__t_sectors_lookup_miss.sum","dram__bytes_read.sum",'
          '"gpu__time_duration.sum"')
UNITS_SCALED = '"","","","","","","","","","","","sector","sector","Mbyte","usecond"'
UNITS_BASE = '"","","","","","","","","","","","sector","sector","byte","nsecond"'


def raw_csv(units, rows):
    lines = ['==PROF== Connected to process 4242 (/usr/bin/python3.10)',
             '==PROF== Profiling "cr_reread" - 0: 0%....50%....100% - 3 passes',
             'app says hello', HEADER, units]
    for i, (name, hit, miss, dram, t) in enumerate(rows):
        lines.append('"%d","4242","python3","pod","%s","1","7","(256, 1, 1)","(1024, 1, 1)","0","12.0","%s","%s","%s","%s"'
                     % (i, name, hit, miss, dram, t))
        lines.append("app output between kernels")
    lines.append('==PROF== Disconnected from process 4242')
    lines.append('==WARNING== No kernels were profiled.' if not rows else '==PROF== done')
    return "\n".join(lines) + "\n"


SCALED = raw_csv(UNITS_SCALED, [("cr_reread", "1,000,000", "250,000", "16.78", "85.12"),
                                ("cr_gemv_lut_kernel", "300", "700", "1,500", "2,000.5")])


def test_parse_scaled_units_and_noise():
    rows = counters.parse_ncu_csv(SCALED)
    assert len(rows) == 2
    r0, r1 = rows
    assert r0["id"] == 0 and r0["name"] == "cr_reread"
    assert r0["hit_sectors"] == 1_000_000 and r0["miss_sectors"] == 250_000
    assert r0["hit_rate"] == pytest.approx(0.8)
    assert r0["dram_bytes"] == pytest.approx(16.78e6)
    assert r0["time_ns"] == pytest.approx(85120.0)
    assert r1["hit_rate"] == pytest.approx(0.3)
    assert r1["dram_bytes"] == pytest.approx(1.5e9)
    assert r1["time_ns"] == pytest.approx(2_000_500.0)


def test_parse_kbyte_msecond_and_base_units():
    text = raw_csv('"","","","","","","","","","","","sector","sector","Kbyte","msecond"', [("k", "10", "10", "2.5", "1.5")])
    r = counters.parse_ncu_csv(text)[0]
    assert r["dram_bytes"] == pytest.approx(2500.0) and r["time_ns"] == pytest.approx(1.5e6)
    text = raw_csv(UNITS_BASE, [("k", "10", "30", "4096", "777")])
    r = counters.parse_ncu_csv(text)[0]
    assert r["dram_bytes"] == 4096 and r["time_ns"] == 777 and r["hit_rate"] == pytest.approx(0.25)


def test_parse_zero_sectors_and_empty():
    r = counters.parse_ncu_csv(raw_csv(UNITS_BASE, [("k", "0", "0", "0", "5")]))[0]
    assert r["hit_rate"] is None
    assert counters.parse_ncu_csv(raw_csv(UNITS_BASE, [])) == []
    assert counters.parse_ncu_csv("") == []
    assert counters.parse_ncu_csv("==ERROR== boom\n") == []


def test_parse_details_page_long_format():
    hdr = '"ID","Process ID","Process Name","Host Name","Kernel Name","Context","Stream","Section Name","Metric Name","Metric Unit","Metric Value"'
    lines = [hdr]
    for name, m, u, v in [("a", "lts__t_sectors_lookup_hit.sum", "sector", "90"),
                          ("a", "lts__t_sectors_lookup_miss.sum", "sector", "10"),
                          ("a", "dram__bytes_read.sum", "Mbyte", "2"),
                          ("a", "gpu__time_duration.sum", "usecond", "3")]:
        lines.append('"0","1","p","h","%s","1","7","Command line profiler metrics","%s","%s","%s"' % (name, m, u, v))
    r = counters.parse_ncu_csv("\n".join(lines))
    assert len(r) == 1 and r[0]["hit_rate"] == pytest.approx(0.9)
    assert r[0]["dram_bytes"] == 2e6 and r[0]["time_ns"] == 3000


def test_summarize_math():
    rows = counters.parse_ncu_csv(raw_csv(UNITS_BASE, [("a", "900", "100", "1000", "2000"), ("b", "0", "400", "3000", "4000")]))
    s = counters.summarize(rows, tokens=2)
    assert s["hit_rate"] == pytest.approx(900 / 1400)
    assert s["dram_bytes_per_token"] == 2000.0
    assert s["us_per_token"] == pytest.approx(3.0)
    assert s["n_kernels"] == 2 and s["dram_bytes_total"] == 4000 and s["time_ns_total"] == 6000
    with pytest.raises(ValueError):
        counters.summarize(rows, 0)
    assert counters.summarize([], 1)["hit_rate"] is None


def test_bucket_first_regex_wins_and_other():
    rows = [{"name": "k_fwht_unpack", "time_ns": 10.0, "dram_bytes": 5.0, "hit_sectors": 1.0, "miss_sectors": 1.0},
            {"name": "mul_mat_vec_q_hadamard", "time_ns": 20.0, "dram_bytes": 7.0, "hit_sectors": 3.0, "miss_sectors": 1.0},
            {"name": "mul_mat_vec_q", "time_ns": 30.0, "dram_bytes": 9.0, "hit_sectors": 0.0, "miss_sectors": 2.0},
            {"name": "something_else", "time_ns": 40.0, "dram_bytes": 1.0, "hit_sectors": None, "miss_sectors": None}]
    b = counters.bucket(rows, {"hadamard": "fwht|hadamard", "mul_mat_vec": "mul_mat_vec", "flash": "flash_attn"})
    assert b["hadamard"]["count"] == 2 and b["hadamard"]["time_ns"] == 30.0 and b["hadamard"]["dram_bytes"] == 12.0
    assert b["hadamard"]["hit_rate"] == pytest.approx(4 / 6)
    assert b["mul_mat_vec"]["count"] == 1 and b["mul_mat_vec"]["hit_rate"] == 0.0
    assert b["flash"]["count"] == 0 and b["flash"]["hit_rate"] is None
    assert b["other"]["count"] == 1 and b["other"]["time_ns"] == 40.0


def _fake_ncu(tmp_path, monkeypatch):
    f = tmp_path / "ncu"
    f.write_text("#!/bin/sh\n")
    f.chmod(0o755)
    monkeypatch.setenv("CR_NCU", str(f))
    return str(f)


def test_ncu_path_env_and_missing(tmp_path, monkeypatch):
    path = _fake_ncu(tmp_path, monkeypatch)
    assert counters.ncu_path() == path
    monkeypatch.delenv("CR_NCU")
    monkeypatch.setenv("PATH", str(tmp_path / "empty"))
    monkeypatch.setattr(counters.glob, "glob", lambda p: [])
    assert counters.ncu_path() is None


def test_build_ncu_cmd_flags(tmp_path, monkeypatch):
    path = _fake_ncu(tmp_path, monkeypatch)
    cmd = counters.build_ncu_cmd(["python", "x.py", "--a"], kernel_regex="cr_gemv", nvtx_include="cr_decode/",
                                 launch_skip=15, launch_count=25, extra=("--replay-mode", "application"))
    assert cmd[0] == path
    j = " ".join(cmd)
    for frag in ["--csv", "--page raw", "--target-processes all", "--cache-control none", "--clock-control none",
                 "-k regex:cr_gemv", "--kernel-name-base function", "--nvtx --nvtx-include cr_decode/",
                 "--launch-skip 15", "--launch-count 25", "--replay-mode application"]:
        assert frag in j, frag
    m = cmd[cmd.index("--metrics") + 1]
    assert m == ("lts__t_sectors_lookup_hit.sum,lts__t_sectors_lookup_miss.sum,dram__bytes_read.sum,gpu__time_duration.sum")
    assert cmd[-3:] == ["python", "x.py", "--a"]
    plain = counters.build_ncu_cmd(["a"])
    assert "-k" not in plain and "--nvtx" not in plain and "--launch-skip" not in plain and "--launch-count" not in plain
    assert plain[-1] == "a"


def _fake_run(monkeypatch, text, rc):
    calls = []

    def fake(cmd, **kw):
        calls.append((cmd, kw))
        return types.SimpleNamespace(stdout=text, returncode=rc)

    monkeypatch.setattr(counters.subprocess, "run", fake)
    return calls


def test_run_ncu_success(tmp_path, monkeypatch):
    _fake_ncu(tmp_path, monkeypatch)
    calls = _fake_run(monkeypatch, SCALED, 0)
    out = tmp_path / "raw" / "ncu.txt"
    r = counters.run_ncu(["python", "-c", "pass"], tokens=2, kernel_regex="cr_", csv_out=str(out), env={"X": "1"}, cwd=str(tmp_path))
    assert r["ok"] and r["error"] is None and len(r["rows"]) == 2
    assert r["summary"]["n_kernels"] == 2
    assert r["summary"]["dram_bytes_per_token"] == pytest.approx((16.78e6 + 1.5e9) / 2)
    assert r["csv_path"] == str(out) and out.read_text() == SCALED
    assert calls[0][1]["env"]["X"] == "1" and calls[0][1]["cwd"] == str(tmp_path)


def test_run_ncu_perm_error(tmp_path, monkeypatch):
    _fake_ncu(tmp_path, monkeypatch)
    text = ('==PROF== Connected to process 1\n==ERROR== ERR_NVGPUCTRPERM - The user does not have permission to '
            'access NVIDIA GPU Performance Counters on the target device 0.\n')
    _fake_run(monkeypatch, text, 1)
    r = counters.run_ncu(["true"], 1)
    assert r["ok"] is False and r["error"] == "ERR_NVGPUCTRPERM" and r["rows"] == [] and r["summary"] is None


def test_run_ncu_other_failures(tmp_path, monkeypatch):
    _fake_ncu(tmp_path, monkeypatch)
    _fake_run(monkeypatch, "==ERROR== Failed to find metric foo\n", 1)
    r = counters.run_ncu(["true"], 1)
    assert not r["ok"] and "Failed to find metric" in r["error"]
    _fake_run(monkeypatch, "==WARNING== No kernels were profiled.\n", 0)
    r = counters.run_ncu(["true"], 1)
    assert not r["ok"] and "no kernels" in r["error"]

    def boom(cmd, **kw):
        raise subprocess.TimeoutExpired(cmd, 1, output=b"partial")

    monkeypatch.setattr(counters.subprocess, "run", boom)
    r = counters.run_ncu(["true"], 1, timeout=1)
    assert not r["ok"] and "timeout" in r["error"]


def test_run_ncu_no_binary(monkeypatch):
    monkeypatch.setattr(counters, "ncu_path", lambda: None)
    r = counters.run_ncu(["true"], 1)
    assert not r["ok"] and "ncu not found" in r["error"]
    assert counters.probe() == {"ok": False, "error": r["error"]}


def test_probe_uses_run_ncu(tmp_path, monkeypatch):
    _fake_ncu(tmp_path, monkeypatch)
    seen = {}

    def fake(argv, tokens, **kw):
        seen.update(kw)
        return {"ok": True, "error": None, "rows": [{}], "csv_path": None}

    monkeypatch.setattr(counters, "run_ncu", fake)
    assert counters.probe() == {"ok": True, "error": None}
    assert seen["launch_count"] == 1
    monkeypatch.setattr(counters, "run_ncu", lambda *a, **k: {"ok": False, "error": "ERR_NVGPUCTRPERM", "rows": [], "csv_path": None})
    assert counters.probe() == {"ok": False, "error": "ERR_NVGPUCTRPERM"}


def test_cli_probe_exit_codes(monkeypatch, capsys):
    monkeypatch.setattr(counters, "probe", lambda: {"ok": False, "error": "ERR_NVGPUCTRPERM"})
    assert counters.main(["--probe"]) == 1
    assert json.loads(capsys.readouterr().out)["error"] == "ERR_NVGPUCTRPERM"
    monkeypatch.setattr(counters, "probe", lambda: {"ok": True, "error": None})
    assert counters.main(["--probe"]) == 0


def test_cli_run_prints_summary(tmp_path, monkeypatch, capsys):
    seen = {}

    def fake(cmd, tokens, **kw):
        seen.update(kw, cmd=cmd, tokens=tokens)
        rows = counters.parse_ncu_csv(SCALED)
        return {"ok": True, "error": None, "cmd": ["ncu"], "rows": rows, "summary": counters.summarize(rows, tokens), "csv_path": None}

    monkeypatch.setattr(counters, "run_ncu", fake)
    rc = counters.main(["--tokens", "4", "--kernel-regex", "cr_", "--nvtx", "--launch-skip", "3", "--", "python", "x.py"])
    assert rc == 0
    assert seen["cmd"] == ["python", "x.py"] and seen["tokens"] == 4 and seen["nvtx_include"] == "cr_decode/"
    assert seen["kernel_regex"] == "cr_" and seen["launch_skip"] == 3
    out = json.loads(capsys.readouterr().out)
    assert out["ok"] and out["n_rows"] == 2 and out["summary"]["n_kernels"] == 2


def test_doc_states_fallback():
    assert "NVreg_RestrictProfilingToAdminUsers=0" in counters.__doc__
    assert "counters_ok" in counters.__doc__
    assert "ERR_NVGPUCTRPERM" in counters.NO_COUNTERS_NOTE
