"""R2a: long-context prompt set, plan, log checks, summary, segmented nsys analysis and a run against fake binaries."""
import json
import os
import pathlib
import stat
import sys

import pytest

from exps import common, e5_spec, make_prompts, r1, r1_step, r2a_longctx as r2a

EXP = pathlib.Path(__file__).resolve().parent.parent
FILE = make_prompts.R2A_OUT


@pytest.fixture(scope="module")
def rows():
    return make_prompts.load_longctx(FILE)


# ---------------------------------------------------------------- prompt set

def test_committed_set_shape(rows):
    assert len(rows) == 60 and [r["id"] for r in rows] == ["r%03d" % i for i in range(60)]
    cats = {c: [r for r in rows if r["category"] == c] for c in make_prompts.R2A_CATEGORIES}
    assert {c: len(v) for c, v in cats.items()} == {"edit_full_file": 20, "edit_diff": 20, "toolcall_trace": 20}
    for r in rows:
        assert set(r) == {"id", "category", "n_predict", "approx_tokens", "chars", "messages", "meta"}
        assert r["n_predict"] == make_prompts.R2A_N_PREDICT[r["category"]]
        assert r["chars"] == sum(len(m["content"]) for m in r["messages"])
        assert r["approx_tokens"] == round(r["chars"] / make_prompts.CHARS_PER_TOKEN)
    for r in cats["edit_full_file"] + cats["edit_diff"]:
        assert 2000 <= r["approx_tokens"] <= 8000 and len(r["messages"]) == 1 and r["messages"][0]["role"] == "user"
    assert all(1500 <= r["approx_tokens"] <= 6000 for r in cats["toolcall_trace"])
    assert 128 <= max(r["n_predict"] for r in rows) - 384 and {r["n_predict"] for r in rows} == {512, 128}
    assert len({r["meta"]["file"] for r in rows}) == 60                    # no file twice: the n-gram pool must not see a file early
    assert len({m["content"] for r in rows for m in r["messages"][-1:]}) == 60


def test_committed_file_is_ascii_and_matches_the_generator(rows):
    assert all(b < 0x80 for b in FILE.read_bytes())
    fresh = make_prompts.generate_longctx()
    d = make_prompts.drift_rows(rows, fresh)
    assert d["mismatch"] == []                 # a row may drift only when a source file changed (its recorded sha differs)
    if not d["drift"]:
        assert FILE.read_text() == make_prompts.longctx_to_text(fresh)
    a = make_prompts.longctx_to_text(make_prompts.generate_longctx())
    assert a == make_prompts.longctx_to_text(make_prompts.generate_longctx())          # deterministic


def test_edit_prompts_embed_the_real_file_and_a_concrete_instruction(rows):
    kinds = {"full_file": {}, "diff": {}}
    for r in rows:
        if not r["category"].startswith("edit_"):
            continue
        m = r["meta"]
        body = r["messages"][0]["content"]
        src = make_prompts.read_source(make_prompts.R2A_ROOT, m["file"])
        if src is not None and make_prompts.sha16(src) == m["sources"][m["file"]]:
            assert src in body
        assert m["function"] in body.split("Task:")[1]
        assert body.startswith(f"File: {m['file']}\n")
        if m["variant"] == "full_file":
            assert r["category"] == "edit_full_file" and "complete modified file" in body
        else:
            assert r["category"] == "edit_diff" and "unified diff" in body and f"--- a/{m['file']}" in body
        kinds[m["variant"]][m["kind"]] = kinds[m["variant"]].get(m["kind"], 0) + 1
    for v in kinds.values():
        assert set(v) == set(make_prompts.EDIT_KINDS) and min(v.values()) >= 2
    sizes = lambda c: sorted(r["approx_tokens"] for r in rows if r["category"] == c)
    assert abs(sum(sizes("edit_full_file")) - sum(sizes("edit_diff"))) < 0.1 * sum(sizes("edit_diff"))      # same size mix


def test_trace_prompts_are_multi_turn_tool_calls(rows):
    names = {t["name"] for t in make_prompts.TOOL_SCHEMA}
    turns = []
    for r in (x for x in rows if x["category"] == "toolcall_trace"):
        msgs = r["messages"]
        assert msgs[0]["role"] == "system" and msgs[1]["role"] == "user"
        assert [m["role"] for m in msgs[2:]] == ["assistant", "user"] * r["meta"]["n_turns"]
        for m in msgs[2::2]:
            call = json.loads(m["content"])
            assert call["tool"] in names and isinstance(call["args"], dict)
        assert all(m["content"].startswith("Tool result (") for m in msgs[3::2])
        assert msgs[-1]["content"].endswith("Reply with the next tool call as one JSON object.")
        assert all(t in msgs[0]["content"] for t in names)
        turns.append(r["meta"]["n_turns"])
    assert set(turns) == {3, 4, 5}


def test_stats_and_cli(rows, capsys):
    st = make_prompts.longctx_stats(rows)
    assert st["n"] == 60 and st["files"] == 60 and sum(st["edit_kinds"].values()) == 40
    assert st["by_category"]["edit_diff"]["tokens_min"] >= 2000 and st["by_category"]["toolcall_trace"]["n_predict"] == 128
    assert make_prompts.main(["--longctx", "--check"]) == 0
    assert make_prompts.main(["--check"]) == 0
    capsys.readouterr()
    assert make_prompts.main(["--longctx", "--stats"]) == 0
    assert json.loads(capsys.readouterr().out)["n"] == 60


def test_find_functions_c_and_python():
    c = ("#include <x.h>\n\nstatic int add_one(int * p, int n) {\n    for (int i = 0; i < n; ++i) {\n        p[i] += 1;\n    }\n"
         "    return n;\n}\n\nint main(void) {\n    int a[2] = {0, 0};\n    add_one(a, 2);\n    return 0;\n}\n\n"
         "struct foo {\n    int x;\n};\nvoid decl(int a);\n")
    fs = {f["name"]: f for f in make_prompts.find_functions(c, "cpp")}
    assert set(fs) == {"add_one", "main"} and fs["add_one"]["calls"] == 1 and fs["add_one"]["first_param"] == "p" and fs["add_one"]["first_ptr"]
    assert fs["add_one"]["loop"] and fs["add_one"]["loop_lt"] and fs["main"]["calls"] == 0
    py = "import os\n\ndef f(a, b=1):\n    for x in range(a):\n        print(x)\n    return b\n\nclass K:\n    def g(self, y):\n        return f(y)\n"
    ps = {f["name"]: f for f in make_prompts.find_functions(py, "python")}
    assert set(ps) == {"f", "g"} and ps["f"]["calls"] == 1 and ps["f"]["loop_lt"] and ps["g"]["first_param"] == "y"
    assert make_prompts._first_param("const char * name, int n") == ("name", True)


def test_generate_rejects_a_missing_or_non_ascii_file(tmp_path):
    (tmp_path / "a.py").write_text("def f():\n    return 1\n")
    (tmp_path / "b.py").write_text("x = '\u00e9'\n")
    with pytest.raises(RuntimeError, match="missing"):
        make_prompts.generate_longctx(tmp_path, ["nope.py"], [])
    with pytest.raises(RuntimeError, match="not plain ASCII"):
        make_prompts.generate_longctx(tmp_path, ["b.py"], [])


# ---------------------------------------------------------------- configs

def test_lv_flag_exists_in_this_forks_arg_cpp():
    src = (EXP.parent.parent / "common" / "arg.cpp").read_text()
    assert '{"-lv", "--verbosity", "--log-verbosity"}' in src
    log = (EXP.parent.parent / "common" / "log.cpp").read_text()
    assert "case GGML_LOG_LEVEL_INFO:  return LOG_LEVEL_TRACE;" in log      # llama-internal INFO (n_rs_seq line) shows at -lv 4
    ctx = (EXP.parent.parent / "src" / "llama-context.cpp").read_text()
    assert 'LLAMA_LOG_INFO("%s: n_rs_seq' in ctx
    com = (EXP.parent.parent / "common" / "common.cpp").read_text()
    assert "the context does not support partial sequence removal" in com and "supports bounded partial sequence removal" in com
    srv = (EXP.parent.parent / "tools" / "server" / "server-context.cpp").read_text()
    assert "speculative decoding will use checkpoints" in srv and 'SLT_DBG(slot, "created speculative checkpoint' in srv


def test_configs_match_r1_and_all_carry_lv4():
    cfgs = {c["name"]: c for c in r2a.matrix_configs()}
    r1c = {c["name"]: c for c in e5_spec.matrix_configs()}
    assert list(cfgs) == r2a.CONFIG_NAMES == ["base", "k2", "k3", "k4", "ngram_mod", "ngram_map_k", "ngram_mod_short", "ngram_map_k_short"]
    for n, c in cfgs.items():
        cmd = r2a.config_cmd("/b", "/t.gguf", "/d.gguf", c)
        assert cmd.count("-lv") == 1 and cmd[cmd.index("-lv") + 1] == "4" and cmd[cmd.index("-c") + 1] == "16384"
        assert "-bs" not in cmd and c["categories"] is None and c["frac"] == 1.0
        assert [a for a in cmd if a not in ("-lv", "4")] == [a for a in e5_spec.config_cmd("/b", "/t.gguf", "/d.gguf", r1c[n]) if a not in ("-lv", "4")]
    k = lambda n: r2a.config_cmd("/b", "/t.gguf", "/d.gguf", cfgs[n])
    assert [k(n)[k(n).index("--spec-draft-n-max") + 1] for n in ("k2", "k3", "k4")] == ["2", "3", "4"]
    assert "-md" in k("k4") and "-md" not in k("ngram_mod") and "--spec-type" not in k("base")
    assert e5_spec.NGRAM_MOD_SHORT == k("ngram_mod_short")[-8:-2] and e5_spec.NGRAM_MAP_K_SHORT == k("ngram_map_k_short")[-8:-2]
    assert r2a.with_lv(cfgs["k4"], r2a.PROBE_LV)["extra"] == ["-lv", "5"] and cfgs["k4"]["extra"] == ["-lv", "4"]


def test_config_order_puts_base_first_and_is_seeded():
    cfgs = r2a.matrix_configs()
    a, b = r2a.config_order(cfgs, 7), r2a.config_order(cfgs, 7)
    assert [c["name"] for c in a] == [c["name"] for c in b] and a[0]["name"] == "base" and len(a) == 8
    assert [c["name"] for c in r2a.config_order(cfgs, 8)] != [c["name"] for c in a]
    assert sorted(c["name"] for c in a) == sorted(c["name"] for c in cfgs)


# ---------------------------------------------------------------- plan

def test_plan_fits_cuts_evenly_and_lists_what_it_dropped(rows):
    order = make_prompts.shuffled(rows, r2a.SEED, stratify=True)
    pre = [r["category"] for r in order[:30]]
    assert all(pre.count(c) == 10 for c in make_prompts.R2A_CATEGORIES)
    cfgs = r2a.config_order(r2a.matrix_configs())
    big = r2a.plan_n(cfgs, order, 3600.0)
    assert big["n"] == 60 and big["fits"] and big["dropped"] == [] and big["est_s"] <= 0.9 * 3600
    p = r2a.plan_n(cfgs, order, 21.5 * 60)
    assert 12 <= p["n"] < 60 and p["fits"] and p["est_s"] <= 0.9 * 21.5 * 60
    assert p["dropped"] == [r["id"] for r in order[p["n"]:]] and sum(p["dropped_per_category"].values()) == 60 - p["n"]
    assert max(p["per_category"].values()) - min(p["per_category"].values()) <= 1
    assert max(p["dropped_per_category"].values()) - min(p["dropped_per_category"].values()) <= 1
    tiny = r2a.plan_n(cfgs, order, 60.0)
    assert tiny["n"] == r2a.N_FLOOR and not tiny["fits"]
    slow = r2a.plan_n(cfgs, order, 21.5 * 60, scale=2.0)
    assert slow["n"] < p["n"]


def test_measured_scale_and_part_budget(rows):
    cfg = r2a.matrix_configs(["base"])[0]
    ps = [dict(p) for p in rows[:4]]
    exp = sum(r2a.prompt_cost_s(p, cfg) for p in ps)
    res = {"rows": [{"id": p["id"]} for p in ps], "wall_s": 10.0 + exp, "startup_s": 10.0}
    assert r2a.measured_scale(cfg, res, ps) == pytest.approx(1.0)
    res["wall_s"] = 10.0 + 1.5 * (exp - 4 * r2a.REQUEST_S) + 4 * r2a.REQUEST_S
    assert r2a.measured_scale(cfg, res, ps) == pytest.approx(1.5)
    assert r2a.measured_scale(cfg, {"rows": [], "wall_s": 5}, ps) is None
    m = r2a.part_budget_s(29 * 60, "m", list("mcp"))
    c = r2a.part_budget_s(29 * 60 - m, "c", list("cp"))
    assert m + c + r2a.part_budget_s(29 * 60 - m - c, "p", list("p")) < 29 * 60 and c == pytest.approx((29 * 60 - m - 25) * 5.5 / 7.5)
    assert r2a.part_budget_s(29 * 60, "m", list("mcp")) == pytest.approx((29 * 60 - 25) * 21.5 / 29.0)


# ---------------------------------------------------------------- logs

LOG_RS = """0.01.0 I llama_context: n_rs_seq              = 4
0.01.1 I cmn  common_conte: the context supports bounded partial sequence removal
0.01.2 I llama_context: n_rs_seq              = 4
0.01.3 I cmn  common_conte: the context supports bounded partial sequence removal
0.01.4 I srv    load_model: speculative decoding enabled: draft-simple
0.02.0 I slot update_slots: id  0 | task 0 | created context checkpoint 1 of 32 (pos_min = 1, pos_max = 2)
"""
LOG_FULL = """0.01.0 I llama_context: n_rs_seq              = 0
0.01.1 I cmn  common_conte: the context does not support partial sequence removal
0.01.2 I srv    load_model: speculative decoding will use checkpoints
0.01.3 I srv    load_model: speculative decoding enabled: ngram-mod
0.01.4 I spec  common_specu: adding speculative implementation 'ngram-mod'
"""


def test_log_checks_read_the_path_from_the_trace_lines():
    a = r2a.log_checks(LOG_RS, "draft-simple")
    assert a["rollback_path"] == "in_batch" and a["seq_rm_types"] == ["RS", "RS"] and a["path_consistent"] is True
    assert a["n_rs_seq"] == [4, 4] and not a["spec_uses_checkpoints"] and a["prompt_checkpoints_logged"] == 1 and a["checkpoint_replay"] is False
    b = r2a.log_checks(LOG_FULL, "ngram-mod")
    assert b["rollback_path"] == "checkpoint" and b["seq_rm_types"] == ["FULL"] and b["spec_uses_checkpoints"] and b["path_consistent"] is True
    assert b["checkpoint_replay"] is True and b["impl_lines"] and "ngram-mod" in b["impl_lines"][0]
    c = r2a.log_checks(LOG_FULL.replace("does not support partial", "supports bounded partial"), "ngram-mod")
    assert c["path_consistent"] is False and "seq_rm lines say" in r2a.config_notes("ngram_mod", {"checks": c, "spec_type": "ngram-mod"})[0]
    d = r2a.log_checks("nothing useful\n", "draft-simple")
    assert d["rollback_path"] == "unknown" and d["path_consistent"] is None
    assert "no n_rs_seq line" in r2a.config_notes("k4", {"checks": d, "spec_type": "draft-simple"})[0]
    assert r2a.log_checks("", "none")["rollback_path"] == "none"
    notes = r2a.config_notes("ngram_mod", {"checks": b, "spec_type": "ngram-mod"})
    assert any("KV-checkpoint" in n and "replay" in n for n in notes)


def test_count_spec_events_from_a_debug_log():
    log = "\n".join(["slot x: created speculative checkpoint (pos_min = 1)", "slot x: accepted 4/4 draft tokens, new n_tokens = 9",
                     "slot x: created speculative checkpoint (pos_min = 9)", "slot x: restoring speculative checkpoint (pos_min = 9)",
                     "slot x: accepted 1/4 draft tokens, new n_tokens = 11", "slot x: accepted 0/0 draft tokens", "slot x: accepted 0/3 draft tokens"])
    ev = r2a.count_spec_events(log)
    assert ev["created"] == 2 and ev["restored"] == 1 and ev["steps_with_draft"] == 3 and ev["full_accept"] == 1
    assert ev["partial_accept"] == 1 and ev["zero_accept"] == 1 and ev["restore_per_step"] == pytest.approx(1 / 3)


# ---------------------------------------------------------------- summary

def _row(i, cat, text, tps, toks=100, drafted=0, acc=0):
    return {"id": f"r{i:03d}", "category": cat, "tokens_predicted": toks, "predicted_per_second": tps, "predicted_ms": 1e3 * toks / tps,
            "draft_n": drafted, "draft_n_accepted": acc, "text": text, "prompt_n": 3000}


def test_summarize_has_intervals_per_category_and_overall():
    cats = ["edit_full_file", "edit_diff", "toolcall_trace"] * 8
    base = [_row(i, c, f"t{i}", 100.0 + (i % 3)) for i, c in enumerate(cats)]
    k4 = [_row(i, c, f"t{i}" if i % 4 else "x", 150.0 + (i % 5), drafted=100, acc=40 + (i % 7)) for i, c in enumerate(cats)]
    ng = [_row(i, c, f"t{i}", 90.0, drafted=40, acc=20) for i, c in enumerate(cats)]
    runs = {"base": {"spec_type": "none", "rows": base, "checks": {}}, "k4": {"spec_type": "draft-simple", "k": 4, "rows": k4, "checks": {}},
            "ngram_mod": {"spec_type": "ngram-mod", "rows": ng, "checks": {"checkpoint_replay": True}}}
    s = r2a.summarize(runs, ci=True)
    a = s["k4"]["all"]
    assert a["n"] == 24 and 1.4 < a["speedup"] < 1.6 and a["speedup_ci95"][0] < a["speedup"] < a["speedup_ci95"][1]
    assert a["tok_s_ci95"][0] < a["tok_s"] < a["tok_s_ci95"][1] and a["base_tok_s_ci95"] and a["r_ci95"] and a["alpha_ci95"]
    assert a["text_equal"] == 18 and a["text_compared"] == 24 and a["text_equal_frac"] == pytest.approx(0.75)
    assert set(s["k4"]["by_category"]) == set(cats) and s["k4"]["by_category"]["edit_diff"]["speedup_ci95"]
    assert s["k4"]["n_by_category"] == {"edit_full_file": 8, "edit_diff": 8, "toolcall_trace": 8} and s["k4"]["n_ge_min"] is False
    n = s["ngram_mod"]["all"]
    assert n["alpha"] is None and n["r"] == pytest.approx(0.5) and n["r_ci95"] and "alpha_ci95" in n and n["alpha_ci95"] is None
    assert s["ngram_mod"]["steps_count_replays"] is True and s["base"]["all"]["speedup"] == pytest.approx(1.0)
    assert "speedup_ci95" not in r2a.summarize(runs)["k4"]["all"] or r2a.summarize(runs)["k4"]["all"].get("speedup_ci95") is None
    one = r2a.summarize({"base": runs["base"], "k4": {"spec_type": "draft-simple", "k": 4, "rows": k4[:1], "checks": {}}}, ci=True)
    assert one["k4"]["all"]["n"] == 1 and "tok_s_ci95" not in one["k4"]["all"]


def test_observed_prompt_set():
    ps = [{"id": "r000", "approx_tokens": 1000}, {"id": "r001", "approx_tokens": 2000}]
    out = r2a.observed_prompt_set(ps, [{"id": "r000", "category": "edit_diff", "prompt_n": 1100}, {"id": "r001", "category": "edit_diff", "prompt_n": 2400}])
    assert out["edit_diff"]["n"] == 2 and out["edit_diff"]["prompt_n_over_estimate_median"] == pytest.approx(1.15)


# ---------------------------------------------------------------- segmented nsys analysis

def _request_acts(t, steps=8, prefill_ms=200):
    """One request: prefill (one long target burst, then a draft prompt pass), then `steps` decode steps as in r1's two-stream fixture."""
    ms = 1_000_000
    acts = []

    def burst(t0, dur, stream, n=20):
        d = dur // n
        for i in range(n):
            acts.append({"t0": t0 + i * d, "t1": t0 + i * d + d - 1000, "kind": "kernel", "stream": stream, "name": f"k{stream}"})
        return t0 + dur
    if prefill_ms:
        t = burst(t, prefill_ms * ms, 7, n=200)
        t += 200_000
        t = burst(t, 15 * ms, 9)
        t += 300_000
    for _ in range(steps):
        t = burst(t, 10 * ms, 7)
        t += 5_000
        t = burst(t, 2 * ms, 9)
        for _ in range(3):
            t += 400_000
            t = burst(t, int(1.6 * ms), 9)
        t += 300_000
    return acts, t


def _capture_acts(n_req=3, **kw):
    acts, t = [], 0
    for _ in range(n_req):
        a, t = _request_acts(t, **kw)
        acts += a
        t += 400_000_000                                   # 400 ms idle between requests
    return acts


def test_analyze_segments_cuts_prefill_and_pools_decode_steps():
    acts = _capture_acts(3)
    res = r2a.analyze_segments(acts, 4)
    assert res["ok"] and res["n_segments"] == 3 and res["n_segments_used"] == 3 and res["split_mode"] == "stream"
    assert res["n_steps"] == 3 * 6                         # 8 verifies, 1 skipped, 7 left -> 6 steps per request
    m = res["median_ms"]
    assert m["target_verify"] == pytest.approx(9.98, abs=0.05) and m["draft_reeval"] == pytest.approx(2.0, abs=0.05)
    assert m["draft_gen"] == pytest.approx(4.8, abs=0.1) and m["wall"] == pytest.approx(18.3, abs=0.2)
    assert all(s["n_prefill_bursts"] >= 1 and s["prefill_ms"] > 150 for s in res["segments"]) and res["checks"]["prefill_cut_in_every_segment"]
    plain = r1_step.analyze(acts, 4, max_steps=100)         # without the cut the prefill and the idle gaps become steps
    assert plain["ok"] and plain["median_ms"]["wall"] >= m["wall"] and max(s["wall"] for s in plain["steps_ms"]) > 150
    no_pre = r2a.analyze_segments(_capture_acts(3, prefill_ms=0), 4)
    assert no_pre["ok"] and not no_pre["checks"]["prefill_cut_in_every_segment"] and no_pre["median_ms"]["wall"] == pytest.approx(18.3, abs=0.2)


def test_analyze_segments_failure_modes():
    assert r2a.analyze_segments([], 4)["error"]
    res = r2a.analyze_segments(_capture_acts(2, steps=2), 4)
    assert not res["ok"] and "no decode steps" in res["error"] and len(res["segments"]) == 2
    a, _ = _request_acts(0)
    segs = r2a.split_segments(a + [dict(x, t0=x["t0"] + 10 ** 10, t1=x["t1"] + 10 ** 10) for x in a])
    assert len(segs) == 2


def test_overhead_compares_the_nsys_step_with_the_unprofiled_step():
    un = [_row(i, "x", "t", 100.0, toks=48, drafted=44, acc=20) for i in range(4)]          # steps = 48 - 20 - 1 = 27 per row
    pr = [_row(i, "x", "t", 80.0, toks=48, drafted=44, acc=20) for i in range(4)]
    an = {"median_ms": {"wall": 20.0, "target_verify": 10.0, "draft_gen": 5.0, "draft_reeval": 1.5, "c1_floor": 3.0}}
    o = r2a.overhead(un, pr, an)
    assert o["unprofiled_ms_per_step"] == pytest.approx(480 / 27 / 1.0) and o["profiled_request_over_unprofiled"] == pytest.approx(1.25)
    assert o["overhead_ms"] == pytest.approx(20.0 - 480 / 27) and o["nsys_wall_over_unprofiled"] == pytest.approx(20.0 / (480 / 27))
    assert o["split_ms"]["kernel_busy_sum"] == pytest.approx(16.5) and o["unprofiled_non_compute_ms"] == pytest.approx(480 / 27 - 16.5)
    assert o["c1_floor_minus_overhead_ms"] == pytest.approx(max(3.0 - (20.0 - 480 / 27), 0.0))
    assert r2a.overhead([], [], None) == {"unprofiled_ms_per_step": None, "profiled_request_ms_per_step": None, "nsys_step_wall_ms": None}


def test_control_subset_is_a_balanced_prefix(rows):
    order = make_prompts.shuffled(rows, r2a.SEED, stratify=True)
    sub = r2a.control_subset(order, 10)
    assert [p["id"] for p in sub] == [p["id"] for p in order[:10]] and sorted(r2a._count(sub).values()) == [3, 3, 4]


# ---------------------------------------------------------------- end to end with fake binaries

FAKE_SERVER = """#!{py}
import json, os, sys
from http.server import BaseHTTPRequestHandler, HTTPServer
a = sys.argv
port = int(a[a.index("--port") + 1])
spec = a[a.index("--spec-type") + 1] if "--spec-type" in a else "none"
lv = int(a[a.index("-lv") + 1]) if "-lv" in a else 3
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
            return self._send({{"prompt": "T:" + body["messages"][-1]["content"][-60:]}})
        text = "out:" + body["prompt"]
        n = body["n_predict"]
        speed = {{"none": 100.0, "draft-simple": 120.0}}.get(spec, 130.0)
        t = {{"predicted_per_second": speed, "predicted_ms": 1e3 * n / speed, "predicted_n": n, "prompt_n": 3000, "prompt_ms": 1000.0}}
        if spec != "none":
            t.update(draft_n=2 * n, draft_n_accepted=n // 2)
        if lv >= 5:
            print("slot x: created speculative checkpoint (pos_min = 1)", flush=True)
            print("slot x: restoring speculative checkpoint (pos_min = 1)", flush=True)
            print("slot x: accepted 1/4 draft tokens, new n_tokens = 5", flush=True)
        self._send({{"content": text, "tokens_predicted": n, "timings": t, "stop_type": "limit"}})
print("srv: speculative decoding enabled: " + spec, flush=True)
if spec == "draft-simple":
    print("llama_context: n_rs_seq              = 4", flush=True)
    print("cmn  common_conte: the context supports bounded partial sequence removal", flush=True)
    print("llama_context: n_rs_seq              = 4", flush=True)
    print("cmn  common_conte: the context supports bounded partial sequence removal", flush=True)
elif spec != "none":
    print("llama_context: n_rs_seq              = 0", flush=True)
    print("cmn  common_conte: the context does not support partial sequence removal", flush=True)
    print("srv    load_model: speculative decoding will use checkpoints", flush=True)
else:
    print("llama_context: n_rs_seq              = 0", flush=True)
HTTPServer(("127.0.0.1", port), H).serve_forever()
"""

FAKE_NSYS = """#!{py}
import json, os, sqlite3, subprocess, sys
a = sys.argv
here = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(here, "nsys.log"), "a") as f:
    f.write(json.dumps({{"verb": a[1] if len(a) > 1 else "", "env_keys": sorted(os.environ)}}) + "\\n")
if a[1:2] == ["--version"]:
    print("NVIDIA Nsight Systems version 0.0.0-fake")
elif a[1:2] == ["profile"]:
    prefix = a[a.index("-o") + 1]
    child = subprocess.Popen(a[a.index("-o") + 2:])
    child.wait()
    open(prefix + ".nsys-rep", "w").write("report")
    con = sqlite3.connect(prefix + ".sqlite")
    con.execute("create table StringIds (id integer primary key, value text)")
    con.execute("create table CUPTI_ACTIVITY_KIND_KERNEL (start integer, end integer, streamId integer, demangledName integer)")
    con.execute("create table CUPTI_ACTIVITY_KIND_MEMCPY (start integer, end integer, streamId integer, copyKind integer, bytes integer)")
    for t0, t1, st in json.load(open(os.path.join(here, "acts.json"))):
        con.execute("insert into CUPTI_ACTIVITY_KIND_KERNEL values (?, ?, ?, 0)", (t0, t1, st))
    con.commit(); con.close()
"""


def _script(path, text):
    path.write_text(text.format(py=sys.executable))
    path.chmod(path.stat().st_mode | stat.S_IXUSR)


@pytest.fixture
def pod(tmp_path, monkeypatch):
    b = tmp_path / "bin"
    b.mkdir()
    _script(b / "llama-server", FAKE_SERVER)
    _script(tmp_path / "nsys", FAKE_NSYS)
    acts = [[a["t0"], a["t1"], a["stream"]] for a in _capture_acts(3) if a["kind"] == "kernel"]
    (tmp_path / "acts.json").write_text(json.dumps(acts))
    for n in ("target.gguf", "draft-bf16.gguf"):
        (tmp_path / n).write_bytes(b"gguf")
    monkeypatch.setenv("CR_RESULTS_DIR", str(tmp_path / "results"))
    monkeypatch.setenv("CR_GPU_TAG", "pro6000-ws")
    monkeypatch.setenv("CR_LLAMA_BIN", str(b))
    monkeypatch.setenv("CR_NSYS", str(tmp_path / "nsys"))
    monkeypatch.setenv("HF_TOKEN", "hf_secretsecret123")
    monkeypatch.setenv("RUNPOD_API_KEY", "rpa_secretsecret456")
    files = {"pq2": "target.gguf", "draft_bf16": "draft-bf16.gguf"}
    monkeypatch.setattr(r2a.e4_pdl, "model_path", lambda name, notes: (str(tmp_path / files[name]), {"file": files[name]}))
    monkeypatch.setattr(r2a.e4_pdl, "smi_sample", lambda: None)
    return tmp_path


def _result(tmp):
    files = sorted((tmp / "results" / "r2a" / "pro6000-ws").glob("2*.json"))
    assert len(files) == 1
    return json.loads(files[0].read_text())


def test_r2a_end_to_end_with_fakes(pod):
    args = ["--configs", "base,k2,k4,ngram_mod,ngram_map_k_short", "--limit", "9", "--control-n", "3", "--nsys-n-predict", "16", "--budget-min", "12"]
    assert r2a.main(args) == 0
    doc = _result(pod)
    d = doc["data"]
    assert d["prompt_set"]["n"] == 60 and d["host"]["problem"] is None
    m = d["matrix"]
    assert m["config_order"][0] == "base" and sorted(m["config_order"]) == sorted(["base", "k2", "k4", "ngram_mod", "ngram_map_k_short"])
    assert m["config_order"] == [c["name"] for c in r2a.config_order(r2a.matrix_configs(["base", "k2", "k4", "ngram_mod", "ngram_map_k_short"]))]
    assert m["plan"]["n"] == 9 and m["plan"]["dropped"] == [] and set(m["plan"]["per_category"].values()) == {3}
    assert list(m["runs"]) == m["config_order"]
    base = m["runs"]["base"]
    assert len(base["rows"]) == 9 and base["rows"][0]["text"].startswith("out:") and {r["n_predict"] for r in base["rows"]} == {512, 128}
    for name, run in m["runs"].items():
        cmd = run["cmd"]
        assert cmd[cmd.index("-lv") + 1] == "4" and cmd[cmd.index("-c") + 1] == "16384" and run["checks"]["rollback_path"] != "unknown"
    s = m["summary"]
    assert s["k4"]["all"]["speedup"] == pytest.approx(1.2) and s["k4"]["all"]["speedup_ci95"] and s["k4"]["all"]["tok_s_ci95"]
    assert s["k4"]["all"]["text_equal_frac"] == 1.0 and s["k4"]["all"]["alpha"] is not None and s["ngram_mod"]["all"]["alpha"] is None
    assert set(s["k4"]["by_category"]) == set(make_prompts.R2A_CATEGORIES) and s["k4"]["n_by_category"] == {c: 3 for c in make_prompts.R2A_CATEGORIES}
    assert s["k4"]["checks"]["rollback_path"] == "in_batch" and s["k4"]["checks"]["seq_rm_types"] == ["RS", "RS"]
    assert s["ngram_mod"]["checks"]["rollback_path"] == "checkpoint" and s["ngram_mod"]["checks"]["spec_uses_checkpoints"]
    assert s["ngram_mod"]["steps_count_replays"] is True and s["base"]["checks"]["rollback_path"] == "none"
    assert m["prompt_tokens_observed"]["edit_diff"]["prompt_n_median"] == 3000
    assert any("ngram_mod: rollback_path=checkpoint" in n for n in d["notes"])
    c = d["control"]
    assert c["subset_ids"] == m["order"][:3] and c["n_predict"] == 16
    assert c["unprofiled"]["ms_per_step"] and len(c["unprofiled"]["rows"]) == 3
    ns = c["nsys"]
    assert ns["ok"] and ns["leftover_files"] == [] and ns["analysis"]["mode"] == "segments" and ns["analysis"]["n_segments"] == 3
    assert ns["analysis"]["segments_match_requests"] is True and len(ns["requests"]) == 3 and "text" not in ns["requests"][0]
    ov = c["overhead"]
    assert ov["unprofiled_ms_per_step"] and ov["nsys_step_wall_ms"] == pytest.approx(18.3, abs=0.3) and "overhead_ms" in ov
    assert ov["split_ms"]["target_verify"] == pytest.approx(9.98, abs=0.05)
    p = d["probe"]
    assert set(p["configs"]) == {"k4", "ngram_map_k_short"} and p["lv"] == "5"
    ev = p["configs"]["k4"]["events"]
    assert ev["created"] >= 3 and ev["restored"] >= 3 and ev["partial_accept"] >= 3 and ev["restore_per_step"] == pytest.approx(1.0)
    text = json.dumps(doc)
    assert "hf_secretsecret123" not in text and "rpa_secretsecret456" not in text
    verbs = [json.loads(ln) for ln in (pod / "nsys.log").read_text().splitlines()]
    assert {"profile", "start", "stop"} <= {v["verb"] for v in verbs}
    for v in verbs:
        assert "HF_TOKEN" not in v["env_keys"] and "RUNPOD_API_KEY" not in v["env_keys"] and "TMPDIR" in v["env_keys"]
    assert not list(pod.rglob("*.nsys-rep")) and not list(pod.rglob("*.sqlite"))
    raw = pod / "results" / "r2a" / "pro6000-ws" / "raw"
    assert not (raw / "partial.json").exists() and not list(raw.glob("*probe*"))          # debug logs are not kept
    assert sorted(f.name.split("-", 1)[1] for f in raw.glob("*-server-*.log")) >= ["server-base.log", "server-control-nsys.log"]


def test_r2a_cuts_the_prompt_count_evenly_and_logs_it(pod):
    assert r2a.main(["--parts", "m", "--configs", "base,k2,k3,k4,ngram_mod,ngram_map_k,ngram_mod_short,ngram_map_k_short", "--budget-min", "9"]) == 0
    d = _result(pod)["data"]
    m = d["matrix"]
    pl = m["plan"]
    assert pl["n"] < 60 and pl["dropped"] and sum(pl["dropped_per_category"].values()) == 60 - pl["n"]
    assert max(pl["per_category"].values()) - min(pl["per_category"].values()) <= 1
    assert any("dropped (evenly" in n for n in d["notes"]) or any("cut from" in n for n in d["notes"])
    n_run = {n: len(r["rows"]) for n, r in m["runs"].items()}
    assert n_run["base"] <= pl["n"] and all(v <= n_run["base"] for v in n_run.values())
    assert d["control"] is None and d["probe"] is None


def test_r2a_unhealthy_host_skips_timed_parts(pod, monkeypatch):
    monkeypatch.setattr(r1, "host_health", lambda smi, load, check_power: ("SM clock 900 MHz < 85% of max 2430", None))
    assert r2a.main(["--quick", "--configs", "base,k4", "--limit", "6", "--budget-min", "8"]) == 1
    d = _result(pod)["data"]
    assert d["matrix"] is None and d["control"] is None and d["host"]["parts_skipped"] == ["m", "c"] and d["probe"] is not None
    assert any("unhealthy host" in n for n in d["notes"])


def test_r2a_without_nsys_records_the_error_and_keeps_the_rest(pod, monkeypatch):
    monkeypatch.delenv("CR_NSYS")
    monkeypatch.setattr(r2a.e4_bonsai, "find_nsys", lambda: None)
    assert r2a.main(["--quick", "--parts", "m,c", "--configs", "base,k4", "--limit", "6", "--budget-min", "8"]) == 0
    d = _result(pod)["data"]
    assert d["control"]["nsys"]["ok"] is False and "nsys not found" in d["control"]["nsys"]["error"] and d["control"]["unprofiled"]["ms_per_step"]
    assert d["control"]["overhead"] is None and d["matrix"]["summary"]["k4"]["all"]["n"] == 6


def test_r2a_dry_run_and_args(capsys):
    assert r2a.main(["--dry-run"]) == 0
    plan = json.loads(capsys.readouterr().out)
    assert plan["exp"] == "r2a" and plan["n_prompts"] == 60 and plan["prompt_set"]["n"] == 60 and plan["parts"] == ["m", "c", "p"]
    assert plan["matrix"]["plan"]["n"] < 60 and plan["matrix"]["config_order"][0] == "base" and len(plan["control"]["subset_ids"]) == 10
    assert all("-lv" in c["cmd"] for c in plan["matrix"]["configs"]) and plan["probe"]["server_cmd"][-1] == "5"
    assert "--start-later=true" in plan["control"]["nsys_profile"]
    with pytest.raises(SystemExit):
        r2a.parse_args(["--configs", "k4"])
    with pytest.raises(SystemExit):
        r2a.parse_args(["--configs", "base,k9"])
    with pytest.raises(SystemExit):
        r2a.parse_args(["--ctx", "8192"])
    a = r2a.parse_args(["--quick"])
    assert a.limit == 6 and a.control_n == 3 and a.budget_min == 8.0
    assert r2a.parse_args([]).budget_min == r2a.DEFAULT_TOTAL_MIN == 29.0
