"""Wrap A (A1 + A2), A3 and B drivers, their helpers in r2b_multicol / r2b_spec, the experiment switches in common/ and the server."""
import json
import os
import stat
import pathlib
import shutil
import subprocess
import sys

import pytest

import test_exps_r2b as r2bt
from exps import make_prompts, r2a_longctx as r2a, r2b_multicol as mc, r2b_spec as sp, rs_rollback_cpu, wrap_a, wrap_a3, wrap_b

EXP = pathlib.Path(__file__).resolve().parent.parent
REPO = EXP.parent.parent
SMI = {"clocks.sm": 2800.0, "clocks.max.sm": 3090.0, "clocks.mem": 13365.0, "clocks.max.mem": 14001.0, "power.draw": 400.0,
       "power.limit": 600.0, "power.default_limit": 600.0, "temperature.gpu": 60.0}


# ---------------------------------------------------------------- the switches in the C++ sources (compile check: scratch cmake object builds)

def test_switches_are_env_gated_and_default_off():
    com = (REPO / "common" / "common.h").read_text()
    f = com[com.index("static inline uint32_t common_exp_ngram_rs()"):][:400]
    assert 'getenv("GGML_EXP_NGRAM_RS")' in f and "static const uint32_t n" in f and ": 0u" in f
    f = com[com.index("static inline bool common_exp_skip_draft_reeval("):][:700]
    assert 'getenv("GGML_EXP_SKIP_DRAFT_REEVAL")' in f and "llama_n_rs_seq(ctx_dft) > 0) {\n        return false;" in f      # a recurrent draft is out
    assert "COMMON_SPECULATIVE_TYPE_DRAFT_SIMPLE" in f and "std::all_of" in f                                          # draft-simple alone
    spec = (REPO / "common" / "speculative.cpp").read_text()
    f = spec[spec.index("static bool common_exp_draft_time()"):][:300]
    assert 'getenv("GGML_EXP_DRAFT_TIME")' in f and "static const bool enabled" in f
    assert "exp_skip_reeval = n_seq == 1 && common_exp_skip_draft_reeval(params, ctx_dft);" in spec
    proc = spec[spec.index("struct common_speculative_impl_draft_simple"):]
    proc = proc[proc.index("bool process(const llama_batch & batch) override {"):proc.index("void draft(common_speculative_draft_params_vec")]
    assert "if (exp_skip_reeval && batch.token" in proc and "batch.n_tokens <= (int32_t) n_batch_dft" in proc and "llama_memory_seq_pos_max(mem, s)" in proc
    assert proc.index("const int ret = llama_decode(ctx_dft, batch_dft);") > proc.index("exp_skip_reeval")
    assert "if (exp_draft_time) {\n            llama_synchronize(ctx_dft);" in proc                                   # the timing needs the synchronize
    # llama_decode is async: the target verify must be done before the timed window opens, else dur_p includes target time
    assert proc.index("llama_synchronize(params.ctx_tgt);") < proc.index("const int64_t t_us = exp_draft_time") < proc.index("llama_decode(ctx_dft, batch_dft)")
    assert "batch.n_tokens + (dp.n_past - p0) + (llama_pos) n_seq <= (llama_pos) n_batch_dft" in spec                  # catch-up fits the batch
    assert "common_sampler_sample(smpl, ctx_dft, i == 0 ? i_last[seq_id] : i_batch, true);" in spec       # id_last may not be batch row 0
    srv = (REPO / "tools" / "server" / "server-context.cpp").read_text()
    assert 'getenv("GGML_EXP_DRAFT_CKPT")' in srv
    assert ("exp_skip_reeval = ctx_dft_seq_rm_type == COMMON_CONTEXT_SEQ_RM_TYPE_PART && params_base.n_parallel == 1 &&\n"
            "                              common_exp_skip_draft_reeval(params_base.speculative, ctx_dft);") in srv
    assert "if (!keep_dft && !llama_memory_seq_rm(llama_get_memory(ctx_dft), slot.id, ckpt.pos_max + 1, -1))" in srv
    for p in (REPO / "common" / "common.h", REPO / "common" / "speculative.cpp", REPO / "tools" / "server" / "server-context.cpp"):
        txt = p.read_text()
        for key in ("GGML_EXP_", "exp skip-reeval", "exp ngram-rs", "exp draft-time", "exp ngram-resets"):
            for ln in [ln for ln in txt.splitlines() if key in ln]:
                assert all(ord(c) < 0x80 for c in ln), ln


def test_the_printed_counter_lines_match_the_parser():
    spec = (REPO / "common" / "speculative.cpp").read_text()
    assert 'SPC_TRC("exp ngram-rs     %16s: window = %u, cut drafts = %zu, cut tokens = %zu\\n"' in spec
    assert 'SPC_TRC("exp skip-reeval  %16s: skipped batches = %zu, catch-up tokens = %zu\\n"' in spec
    assert 'SPC_TRC("exp draft-time   %16s: dur(p,pf) = %.3f, %.3f ms, #calls(p,pf) = %zu %zu\\n"' in spec
    assert 'SPC_TRC("exp ngram-resets %16s: low-acceptance resets = %zu\\n"' in spec
    assert sp.EXP_CUT_RE.search("spec common_specu: exp ngram-rs            ngram-mod: window = 16, cut drafts = 3, cut tokens = 90")
    assert sp.EXP_SKIP_RE.search("spec common_specu: exp skip-reeval      draft-simple: skipped batches = 57, catch-up tokens = 54")
    assert sp.EXP_TIME_RE.search("spec common_specu: exp draft-time        draft-simple: dur(p,pf) = 488.500, 6.474 ms, #calls(p,pf) = 204 2")
    assert sp.EXP_RESET_RE.search("spec common_specu: exp ngram-resets            ngram-mod: low-acceptance resets = 2")
    srv = (REPO / "tools" / "server" / "server-context.cpp").read_text()
    for name in ("GGML_EXP_DRAFT_CKPT", "GGML_EXP_SKIP_DRAFT_REEVAL"):
        line = next(ln for ln in srv.splitlines() if 'SRV_INF("%s", "' + name + ":" in ln)
        assert sp.EXP_MARK_RE.search(line.split('"%s", "')[1]).group(1) == name
    line = next(ln for ln in spec.splitlines() if 'SPC_TRC("%s", "GGML_EXP_DRAFT_TIME:' in ln)
    assert sp.EXP_MARK_RE.search(line.split('"%s", "')[1]).group(1) == "GGML_EXP_DRAFT_TIME"


# ---------------------------------------------------------------- helpers

def test_gpu600_gate():
    assert mc.gpu600_problem("pro6000-ws", SMI) is None and mc.gpu600_problem("pro6000-server", None) is None
    assert "not a 600 W" in mc.gpu600_problem("pro6000-maxq", SMI) and "not a 600 W" in mc.gpu600_problem("rtx5090", SMI)
    assert "default power limit 300 W" in mc.gpu600_problem("pro6000-ws", dict(SMI, **{"power.default_limit": 300.0}))


def test_rounds_alternate_and_paired_gain_per_round():
    cfgs = [("off", {}), ("on", {"X": "1"})]
    assert [[n for n, _ in mc.round_configs(cfgs, r)] for r in range(4)] == [["off", "on"], ["on", "off"]] * 2
    rows = []
    for rnd, g in enumerate((1.10, 1.05, 1.20)):
        for cfg, mult in (("off", 1.0), ("on", g)):
            rows += [{"config": cfg, "round": rnd, "kind": "pp", "width": 4, "tok_s": 400.0 * mult},
                     {"config": cfg, "round": rnd, "kind": "tg", "width": 1, "tok_s": 100.0 * (1.0 if cfg == "off" else 0.99)}]
    rows.append({"config": "off", "round": 3, "kind": "pp", "width": 4, "tok_s": 1.0})        # on never ran round 3: not paired
    p = mc.paired_rounds(rows, "off", "on")
    e = p["pp"]["4"]
    assert e["rounds"] == [0, 1, 2] and e["gain_pct"] == pytest.approx([10.0, 5.0, 20.0]) and e["median"] == pytest.approx(10.0)
    assert e["min"] == pytest.approx(5.0) and e["max"] == pytest.approx(20.0) and e["same_sign"] is True
    assert p["tg"]["1"]["median"] == pytest.approx(-1.0) and p["tg"]["1"]["same_sign"] is True
    s = mc.summarize_bench([dict(r, config={"off": "base", "on": "head"}[r["config"]]) for r in rows], ["base", "head"], ref="base")
    assert s["pp"]["4"]["gain_pct_head"] == pytest.approx(10.0) and "gain_pct_base" not in s["pp"]["4"]


def test_paired_gains_leave_out_the_unhealthy_rounds():
    rows = []
    for rnd, g in enumerate((1.10, 1.05, 1.20, 0.50)):
        for cfg, mult in (("off", 1.0), ("on", g)):
            rows.append({"config": cfg, "round": rnd, "kind": "pp", "width": 4, "tok_s": 400.0 * mult})
    assert mc.paired_rounds(rows)["pp"]["4"]["n"] == 4
    p = mc.paired_rounds(rows, "off", "on", before_round=3)
    assert p["pp"]["4"]["rounds"] == [0, 1, 2] and p["pp"]["4"]["min"] == pytest.approx(5.0)
    assert mc.paired_rounds(rows, "off", "on", before_round=0)["pp"] == {}
    assert mc.summarize_bench(rows, ["off", "on"], before_round=2)["pp"]["4"]["rounds"] == [0, 1]
    v = wrap_a.bench_verdict(p, {"first_bad_round": 3, "problem": "power cap", "where": "on pp"})
    assert v and all(x.startswith("UNHEALTHY HOST (power cap during round 3 on pp;") for x in v)
    assert not any("UNHEALTHY" in x for x in wrap_a.bench_verdict(p, {}))
    assert wrap_a.bench_verdict({"pp": {}, "tg": {}}, {"first_bad_round": 0, "problem": "p"})[0].endswith("no healthy round: no paired gain")


def test_prompt_blocks_keep_the_categories_balanced_and_the_order():
    order = sp.balanced_subset(make_prompts.load_longctx(make_prompts.R2A_OUT))
    blocks = sp.prompt_blocks(order, 4)
    assert [len(b) for b in blocks] == [6] * 4 and all(r2a._count(b) == {c: 2 for c in make_prompts.R2A_CATEGORIES} for b in blocks)
    assert sorted(p["id"] for b in blocks for p in b) == sorted(p["id"] for p in order)
    pos = {p["id"]: i for i, p in enumerate(order)}
    assert all([pos[p["id"]] for p in b] == sorted(pos[p["id"]] for p in b) for b in blocks)
    assert [len(b) for b in sp.prompt_blocks(order[:7], 3)] and sum(len(b) for b in sp.prompt_blocks(order[:7], 3)) == 7


def test_block_orders_run_every_pair_both_ways():
    for names in (["base", "off", "on"], wrap_b.CONFIG_NAMES):
        orders = sp.block_orders(names, 4)
        assert all(sorted(o) == sorted(names) for o in orders)
        for i, a in enumerate(names):
            for b in names[i + 1:]:
                first = [o.index(a) < o.index(b) for o in orders]
                assert any(first) and not all(first), (a, b)


STATS_LOG = """0.01 I llama_context: n_rs_seq              = 4
spec common_specu: statistics     draft-simple: #calls(b,g,a) =    1      6      6, #gen drafts =      6, #acc drafts =     3, #gen tokens =     21, #acc tokens =     9, #mean acc len = 2.50, #acc rate/pos = (0.500, 0.500, 0.333, 0.167), dur(b,g,a) = 0.000, 50.000, 0.004 ms
spec common_specu: exp skip-reeval      draft-simple: skipped batches = 6, catch-up tokens = 2
spec common_specu: statistics     draft-simple: #calls(b,g,a) =    2     56     56, #gen drafts =     56, #acc drafts =    40, #gen tokens =    221, #acc tokens =   150, #mean acc len = 3.70, #acc rate/pos = (0.9), dur(b,g,a) = 0.000, 350.000, 0.050 ms
spec common_specu: exp skip-reeval      draft-simple: skipped batches = 56, catch-up tokens = 22
spec common_specu: exp draft-time        draft-simple: dur(p,pf) = 120.000, 30.000 ms, #calls(p,pf) = 60 3
"""


def test_parse_and_difference_the_statistics_lines():
    st = sp.parse_stats(STATS_LOG)
    assert len(st["draft-simple"]) == 2 and st["draft-simple"][1]["dur_g_ms"] == 350.0 and st["draft-simple"][1]["skipped_batches"] == 56
    t = sp.timed_stats(st)["draft-simple"]
    assert (t["calls_g"], t["gen_tokens"], t["acc_tokens"], t["dur_g_ms"], t["n_requests"]) == (50, 200, 141, pytest.approx(300.0), 1)
    assert (t["skipped_batches"], t["catchup_tokens"]) == (50, 20)
    assert st["draft-simple"][1]["dur_p_ms"] == 120.0 and st["draft-simple"][1]["calls_pf"] == 3 and "dur_p_ms" not in st["draft-simple"][0]
    res = sp.parse_stats("spec common_specu: statistics        ngram-mod: #calls(b,g,a) =    1     37      8, #gen drafts =      8, #acc drafts =     8, "
                         "#gen tokens =     58, #acc tokens =    58, dur(b,g,a) = 0.031, 0.166, 0.011 ms\n"
                         "spec common_specu: exp ngram-resets        ngram-mod: low-acceptance resets = 3\n")
    assert res["ngram-mod"][0]["resets"] == 3 and sp.timed_stats({"ngram-mod": [res["ngram-mod"][0], res["ngram-mod"][0]]})["ngram-mod"]["resets"] == 0
    both = sp.add_stats({"draft-simple": t}, {"draft-simple": t, "ngram-mod": {"calls_g": 3, "rs_window": 16}})
    assert both["draft-simple"]["calls_g"] == 100 and both["ngram-mod"] == {"calls_g": 3, "rs_window": 16}
    cut = sp.parse_stats("spec common_specu: statistics        ngram-mod: #calls(b,g,a) =    1     37      8, #gen drafts =      8, #acc drafts =     8, "
                         "#gen tokens =     58, #acc tokens =    58, dur(b,g,a) = 0.031, 0.166, 0.011 ms\n"
                         "spec common_specu: exp ngram-rs            ngram-mod: window = 8, cut drafts = 8, cut tokens = 64\n")
    assert cut["ngram-mod"][0]["rs_window"] == 8 and sp.timed_stats(cut)["ngram-mod"]["rs_window"] == 8
    assert sp.timed_stats({}) == {} and sp.parse_stats("nothing") == {}


def _row(i, cat, tps, text="t", dn=40, da=20, n=100):
    return {"id": f"r{i:03d}", "category": cat, "tokens_predicted": n, "predicted_per_second": tps, "predicted_ms": 1e3 * n / tps,
            "draft_n": dn, "draft_n_accepted": da, "text": text}


def test_step_split_and_pairs_by_category():
    cats = ["edit_diff", "edit_full_file"]
    runs = {"a": {"spec_type": "draft-simple", "rows": [_row(i, cats[i % 2], 120.0) for i in range(4)],
                  "stats": {"draft-simple": {"dur_g_ms": 200.0, "calls_g": 40}}},
            "b": {"spec_type": "draft-simple", "rows": [_row(i, cats[i % 2], 100.0, "x" if i == 0 else "t") for i in range(4)]}}
    ss = sp.step_split(runs["a"])
    assert ss["draft_ms_per_call"] == pytest.approx(5.0) and ss["rest_ms_per_step"] == pytest.approx(ss["ms_per_step"] - 200.0 / ss["steps"])
    assert sp.step_split(runs["b"])["ms_per_step"] and "draft_ms_per_call" not in sp.step_split(runs["b"]) and sp.step_split({}) == {}
    assert "reeval_ms" not in ss
    runs["a"]["stats"]["draft-simple"].update(dur_p_ms=100.0, calls_p=50, dur_pf_ms=7.0)
    t = sp.step_split(runs["a"])
    assert t["reeval_ms_per_call"] == pytest.approx(2.0) and t["reeval_ms_per_step"] == pytest.approx(100.0 / t["steps"]) and t["prefill_ms"] == 7.0
    assert t["reeval_share_of_step"] == pytest.approx(t["reeval_ms_per_step"] / t["ms_per_step"])
    p = sp.pair_by_category(runs, "a", "b")
    assert p["all"]["speedup"] == pytest.approx(1.2) and p["all"]["text_equal"] == 3 and p["all"]["speedup_ci95"]
    assert set(p["by_category"]) == set(cats) and p["by_category"]["edit_diff"]["text_equal"] == 1
    assert sp.pair_by_category(runs, "a", "missing") is None


def test_wrap_b_expected_paths_and_verification():
    cfgs = {c["name"]: c for c in wrap_b.matrix_configs()}
    exp = {n: wrap_b.expected_path(c) for n, c in cfgs.items()}
    assert exp["base"]["rollback_path"] == "none" and exp["ngram"] == {"rollback_path": "checkpoint", "n_rs_seq_target": 0, "switches": []}
    assert exp["ngram_rs64"]["n_rs_seq_target"] == 64 and exp["k4q8_ckpt_t"]["switches"] == [wrap_b.CKPT, wrap_b.TIME] and exp["k4q8"]["n_rs_seq_target"] == 4
    assert "k4q8_skip" not in cfgs and wrap_b.SKIP not in {k for c in cfgs.values() for k in c["env"]}      # the skip prototype is not in the matrix
    runs = {"ngram_rs16": {"blocks": [{"block": 0, "checks": {"rollback_path": "in_batch", "n_rs_seq_target": 16}, "exp_switches": [],
                                       "stats": {"ngram-mod": {"rs_window": 16}}}]}}
    v = wrap_b.verify_paths([cfgs["ngram_rs16"], cfgs["ngram"]], runs)
    assert v["ngram_rs16"]["ok"] is True and v["ngram"]["ok"] is False                 # a config that never ran does not pass
    runs["ngram_rs16"]["blocks"][0]["stats"] = {"ngram-mod": {"rs_window": 8}}
    assert wrap_b.verify_paths([cfgs["ngram_rs16"]], runs)["ngram_rs16"]["ok"] is False


PROBE_OUT = {"model": "m.gguf", "n_vocab": 10, "n_rs_seq": 4, "n_prompt": 64, "n_extra": 4, "ngl": 99, "cases": [
    {"kept": 0, "rollback": 4, "decode": "single", "seq_rm_ok": True, "max_abs_diff": 13.0, "max_abs_ref": 9.0, "top1_equal": False},
    {"kept": 0, "rollback": 4, "decode": "batch", "seq_rm_ok": True, "max_abs_diff": 11.0, "max_abs_ref": 9.0, "top1_equal": False},
    {"kept": 1, "rollback": 3, "decode": "single", "seq_rm_ok": True, "max_abs_diff": 12.0, "max_abs_ref": 9.0, "top1_equal": False},
    {"kept": 1, "rollback": 3, "decode": "batch", "seq_rm_ok": True, "max_abs_diff": 1e-6, "max_abs_ref": 9.0, "top1_equal": True}]}


def test_probe_verdict_keeps_the_whole_batch_case_apart():
    v = wrap_b.probe_verdict(PROBE_OUT)
    assert v["single"]["max_abs_diff"] == 13.0 and v["single"]["n"] == 2 and v["single"]["top1_equal"] == 0
    assert v["batch"]["max_abs_diff"] == pytest.approx(1e-6) and v["batch"]["n"] == 1 and v["batch_whole"]["by_rollback"] == {"4": 11.0}
    assert v["batch"]["exact"] is True and v["single"]["exact"] is False and v["batch_whole"]["exact"] is False


BOUNDARY_OUT = {"n_vocab": 10, "n_rs_seq": 4, "n_prompt": 64, "n_extra": 5, "ngl": 99, "cases": [
    {"kept": 0, "rollback": 5, "decode": "batch", "error": "decode failed"},
    {"kept": 1, "rollback": 4, "decode": "batch", "seq_rm_ok": True, "max_abs_diff": 2e-6, "max_abs_ref": 9.0, "top1_equal": True},
    {"kept": 2, "rollback": 3, "decode": "batch", "seq_rm_ok": True, "max_abs_diff": 1e-6, "max_abs_ref": 9.0, "top1_equal": True},
    {"kept": 4, "rollback": 1, "decode": "batch", "seq_rm_ok": True, "max_abs_diff": 1e-6, "max_abs_ref": 9.0, "top1_equal": True},
    {"kept": 1, "rollback": 4, "decode": "single", "seq_rm_ok": True, "max_abs_diff": 12.0, "max_abs_ref": 9.0, "top1_equal": False}]}


def test_probe_verdict_reports_the_rollback_equal_n_rs_row():
    v = wrap_b.probe_verdict(BOUNDARY_OUT)
    assert "batch_whole" not in v                                                  # kept 0 failed to decode: out of the window, no entry
    assert v["batch"]["n"] == 3 and v["batch"]["max_rollback"] == 4 and v["batch"]["exact"] is True and set(v["batch"]["by_rollback"]) == {"4", "3", "1"}
    assert v["boundary"] == {"rollback": 4, "kept": 1, "max_abs_diff": 2e-6, "max_abs_ref": 9.0, "top1_equal": True, "exact": True}
    assert v["single"]["max_rollback"] == 4 and v["single"]["exact"] is False
    line = wrap_b.probe_notes("n4", "pq2", dict(BOUNDARY_OUT, verdict=v))
    assert "rollback 4 == n_rs_seq (kept 1)" in line and "exact True" in line
    # a boundary row with a failed seq_rm is not exact even when the logits agree
    bad = dict(BOUNDARY_OUT, cases=[dict(c, seq_rm_ok=False) if c["kept"] == 1 and c["decode"] == "batch" and "error" not in c else c for c in BOUNDARY_OUT["cases"]])
    assert wrap_b.probe_verdict(bad)["boundary"]["exact"] is False and wrap_b.probe_verdict(bad)["batch"]["exact"] is False
    # the old shape (n_extra == n_rs): no boundary row beyond the largest kept >= 1 case, kept 0 stays the whole-batch case
    old = wrap_b.probe_verdict(PROBE_OUT)
    assert old["batch_whole"]["by_rollback"] == {"4": 11.0} and old["batch"]["max_rollback"] == 3 and "boundary" not in old
    assert "not probed" in wrap_b.probe_notes("n4", "pq2", dict(PROBE_OUT, verdict=old))


def test_probe_runs_cover_the_b2_windows():
    runs = {l: (m, a) for l, m, a in wrap_b.PROBE_RUNS}
    assert set(runs) == {"n4", "n16", "n64"} and runs["n4"][0] == ("draft_q8_0", "pq2") and runs["n16"][0] == runs["n64"][0] == ("pq2",)
    for label, n in (("n16", 16), ("n64", 64)):
        a = runs[label][1]
        assert a[a.index("--n-rs") + 1] == str(n) and a[a.index("--extra") + 1] == str(n + 1) and a[a.index("--modes") + 1] == "batch"
    a = runs["n4"][1]
    assert a[a.index("--n-rs") + 1] == "4" and a[a.index("--extra") + 1] == "5"       # K + 1 tokens, rollback up to K == n_rs_seq
    src = wrap_b.PROBE_SRC.read_text()
    assert '--modes' in src and 'modes == "batch"' in src and "n_extra > n_rs + 1" in src and "n_extra - m > n_rs) continue" in src


def test_wrap_b_gate_and_draft_state_check():
    assert wrap_b.gate_problem("pro6000-maxq") is None and wrap_b.gate_problem("rtx5090") is None and wrap_b.gate_problem("pro6000-ws") is None
    assert "not one of" in wrap_b.gate_problem("rtx4090")
    vs = {n: {"all": {"alpha": a}} for n, a in (("k4q8_ckpt", 0.60), ("k4q8", 0.01), ("k4q8_ckpt_t", 0.61))}
    ds = wrap_b.draft_state(vs)
    assert ds["matches_reference"] == {"k4q8": False, "k4q8_ckpt_t": True} and wrap_b.draft_state({}) is None
    rb = {"timed": "k4q8_ckpt_t", "reference": "k4q8_ckpt", "isolated_ms_per_step": 1.5, "end_to_end_delta_ms_per_step": 0.4,
          "ms_per_step_timed": 20.4, "ms_per_step_reference": 20.0}
    lines = wrap_b.verdict_lines({"pairs": {}, "draft_state": ds, "reeval_bounds": rb,
                                  "reeval": {"k4q8_ckpt_t": {"reeval_ms_per_call": 1.5, "reeval_ms_per_step": 1.5, "reeval_share_of_step": 0.07}},
                                  "ngram_cut": {"ngram": {"ngram-mod": {"resets": 2}}}, "paths": {}})
    assert any(l.startswith("draft state: alpha k4q8_ckpt 0.600") and "'k4q8': False" in l for l in lines)
    assert any("1.500 ms per step" in l and "isolated cost" in l and "at most 1.500 ms (isolated), at least 0.400 ms (end-to-end delta)" in l
               for l in lines) and any("low-acceptance resets 2" in l for l in lines)
    assert wrap_b.reeval_bounds({"k4q8_ckpt_t": {"reeval_ms_per_step": 1.5, "ms_per_step": 20.4}, "k4q8_ckpt": {"ms_per_step": 20.0}}) == dict(
        rb, end_to_end_delta_ms_per_step=pytest.approx(0.4))
    assert wrap_b.reeval_bounds({"k4q8_ckpt_t": {"ms_per_step": 20.4}, "k4q8_ckpt": {"ms_per_step": 20.0}}) is None
    assert lines[-1] == wrap_b.NOTE_EQ


def test_probe_source_builds_against_llama_h_only():
    src = wrap_b.PROBE_SRC.read_text()
    assert '#include "llama.h"' in src and "common.h" not in src and "llama_memory_seq_rm(mem, 0, n_prompt + m, -1)" in src
    assert "cp.n_rs_seq = (uint32_t) n_rs" in src


def test_a3_patch_is_the_pr_change_and_reverses_on_this_tree():
    txt = wrap_a3.PATCH.read_text()
    files = sorted({ln.split(" b/")[1] for ln in txt.splitlines() if ln.startswith("diff --git ")})
    assert files == sorted(wrap_a3.PATCH_FILES) and "ggml_cuda_pdl_sync();" in txt and 'getenv("GGML_CUDA_PQ2_MULTICOL")' in txt
    if not shutil.which("git") or not (REPO / ".git").exists():
        pytest.skip("no git checkout")
    r = subprocess.run(["git", "apply", "-R", "--check", str(wrap_a3.PATCH)], cwd=REPO, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr


def test_a3_copy_keeps_cmake_build_info_and_drops_build_dirs():
    top = wrap_a3.copy_ignore(str(REPO), ["build-cuda", "build-xcframework.sh", ".git", "experiments", "ggml", "cmake", "x.gguf", "__pycache__"])
    assert top == {"build-cuda", "build-xcframework.sh", ".git", "experiments", "x.gguf", "__pycache__"}
    assert wrap_a3.copy_ignore(str(REPO / "cmake"), ["build-info.cmake", "x.gguf"]) == {"x.gguf"}


def test_a3_cmake_flags_are_the_bootstrap_flags():
    boot = (EXP / "infra" / "bootstrap.sh").read_text()
    for flag in wrap_a3.cmake_cmd("/s", "/s/b", "89", True)[6:]:
        assert flag.replace("89", '"$CC"') in boot or flag in boot, flag
    assert "CCACHE_BASEDIR" in boot and "CCACHE_NOHASHDIR=1" in boot


# ---------------------------------------------------------------- fakes

FAKE_SERVER = """#!{py}
import json, os, sys
from http.server import BaseHTTPRequestHandler, HTTPServer
a = sys.argv
port = int(a[a.index("--port") + 1])
spec = a[a.index("--spec-type") + 1] if "--spec-type" in a else "none"
draft = "-md" in a
mc = os.environ.get("GGML_CUDA_PQ2_MULTICOL") == "1"
ign = os.environ.get("FAKE_IGNORE_EXP") == "1"
skip = os.environ.get("GGML_EXP_SKIP_DRAFT_REEVAL") == "1" and not ign
ckpt = os.environ.get("GGML_EXP_DRAFT_CKPT") == "1" and not ign
timeit = os.environ.get("GGML_EXP_DRAFT_TIME") == "1" and not ign
nrs = 0 if ign else int(os.environ.get("GGML_EXP_NGRAM_RS", "0"))
impl = "draft-simple" if draft else ("ngram-mod" if spec != "none" else None)
st = dict(g=0, gt=0, at=0, dg=0.0, sk=0, cu=0, cd=0, ct=0)
def stats():
    print("spec common_specu: statistics %16s: #calls(b,g,a) =    1 %6d %6d, #gen drafts = %6d, #acc drafts = %5d, #gen tokens = %6d, "
          "#acc tokens = %5d, dur(b,g,a) = 0.000, %.3f, 0.004 ms" % (impl, st["g"], st["g"], st["g"], st["g"], st["gt"], st["at"], st["dg"]), flush=True)
    if timeit and impl == "draft-simple":
        print("spec common_specu: exp draft-time   %16s: dur(p,pf) = %.3f, %.3f ms, #calls(p,pf) = %d %d" % (impl, 1.5 * st["g"], 20.0, st["g"], 2), flush=True)
    if impl == "ngram-mod":
        print("spec common_specu: exp ngram-resets %16s: low-acceptance resets = %d" % (impl, st["g"] // 10), flush=True)
    if skip and impl == "draft-simple":
        print("spec common_specu: exp skip-reeval  %16s: skipped batches = %d, catch-up tokens = %d" % (impl, st["sk"], st["cu"]), flush=True)
    if nrs and impl == "ngram-mod":
        print("spec common_specu: exp ngram-rs     %16s: window = %d, cut drafts = %d, cut tokens = %d" % (impl, nrs, st["cd"], st["ct"]), flush=True)
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
        if self.path == "/tokenize":
            return self._send({{"tokens": list(range(len(body["content"]) // 4))}})
        n = body["n_predict"]
        speed = 100.0
        if impl == "draft-simple":
            speed = 120.0 * (1.1 if mc else 1.0) * (1.05 if skip else 1.0) * (1.2 if ckpt else 1.0)
        elif impl == "ngram-mod":
            speed = 130.0 * {{0: 1.0, 16: 1.1, 64: 1.2}}.get(nrs, 1.0)
        t = {{"predicted_per_second": speed, "predicted_ms": 1e3 * n / speed, "predicted_n": n, "prompt_n": 3000, "prompt_ms": 1000.0}}
        if impl:
            t.update(draft_n=n, draft_n_accepted=n // 2)
            st["g"] += max(n // 4, 1); st["gt"] += n; st["at"] += n // 2; st["dg"] += 2.0 * max(n // 4, 1)
            st["sk"] += max(n // 4, 1); st["cu"] += 1; st["cd"] += 1; st["ct"] += 10
        self._send({{"content": "out:" + body["prompt"], "tokens_predicted": n, "timings": t, "stop_type": "limit"}})
        if impl:
            stats()
if spec == "none":
    print("llama_context: n_rs_seq              = 0", flush=True)
else:
    rs = 4 if draft else nrs
    for _ in range(2 if draft else 1):
        print("llama_context: n_rs_seq              = %d" % rs, flush=True)
        print("cmn  common_conte: the context " + ("supports bounded" if rs else "does not support") + " partial sequence removal", flush=True)
    if not rs:
        print("srv    load_model: speculative decoding will use checkpoints", flush=True)
    if skip:
        print("srv    load_model: GGML_EXP_SKIP_DRAFT_REEVAL: the draft context keeps its drafted tokens", flush=True)
    if ckpt:
        print("srv    load_model: GGML_EXP_DRAFT_CKPT: the draft context uses checkpoints", flush=True)
    if timeit:
        print("spec common_specu: GGML_EXP_DRAFT_TIME: the decode of the verify batch on the draft context is timed", flush=True)
HTTPServer(("127.0.0.1", port), H).serve_forever()
"""

FAKE_PROBE = "#!{py}\nprint(" + repr(json.dumps(PROBE_OUT)).replace("{", "{{").replace("}", "}}") + ")\n"


@pytest.fixture
def wpod(tmp_path, monkeypatch):
    b = tmp_path / "bin"
    b.mkdir()
    r2bt._script(b / "llama-bench", r2bt.FAKE_BENCH)
    r2bt._script(b / "test-backend-ops", r2bt.FAKE_TBO)
    r2bt._script(b / "llama-server", FAKE_SERVER)
    r2bt._script(tmp_path / "probe", FAKE_PROBE)
    files = {"pq2": "target.gguf", "draft_q8_0": "draft-q8.gguf"}
    for n in files.values():
        (tmp_path / n).write_bytes(b"gguf")
    monkeypatch.setenv("CR_RESULTS_DIR", str(tmp_path / "results"))
    monkeypatch.setenv("CR_GPU_TAG", "pro6000-ws")
    monkeypatch.setenv("CR_LLAMA_BIN", str(b))
    monkeypatch.setenv("HF_TOKEN", "hf_secretsecret123")
    monkeypatch.setenv("RUNPOD_API_KEY", "rpa_secretsecret456")
    for k in [k for k in os.environ if k.startswith("GGML_EXP_") or k == mc.ENV_VAR]:
        monkeypatch.delenv(k)
    monkeypatch.setattr(mc.e4_pdl, "model_path", lambda name, notes: (str(tmp_path / files[name]), {"file": files[name]}))
    monkeypatch.setattr(mc.e4_pdl, "smi_sample", lambda: dict(SMI))
    monkeypatch.setattr(wrap_b, "build_probe", lambda bin_dir, notes: tmp_path / "probe")
    return tmp_path


def _result(tmp, exp, gpu):
    files = sorted((tmp / "results" / exp / gpu).glob("2*.json"))
    assert len(files) == 1
    return json.loads(files[0].read_text())


def test_wrap_a_end_to_end_with_fakes(wpod):
    assert wrap_a.main(["--budget-min", "10", "--per-category", "2", "--blocks", "2"]) == 0
    d = _result(wpod, "wrap_a", "pro6000-ws")["data"]
    assert d["gate"]["problem"] is None and [t["config"] for t in d["tbo"]] == ["on", "on", "on", "off"] and all(mc.tbo_passed(t) for t in d["tbo"])
    assert d["bench_health"] == {} and d["host_b"]["problem"] is None
    assert len(d["bench"]) == 4 * 2 * 10 and [r["config"] for r in d["bench"][::10]] == ["off", "on", "on", "off", "off", "on", "on", "off"]
    bp = d["bench_paired"]
    assert bp["pp"]["4"]["n"] == 4 and bp["pp"]["4"]["median"] == pytest.approx(100.0) and bp["pp"]["2"]["median"] == pytest.approx(0.0)
    s = d["spec"]
    assert s["orders"] == sp.block_orders(wrap_a.SPEC_NAMES, 2) and len(s["blocks"]) == 2
    runs = s["runs"]
    assert all(len(runs[n]["rows"]) == 6 and len(runs[n]["blocks"]) == 2 for n in wrap_a.SPEC_NAMES)
    assert runs["k4q8_on"]["env"] == {mc.ENV_VAR: "1"} and runs["k4q8_off"]["env"] == {}
    cmd = runs["k4q8_on"]["cmd"]
    assert cmd[cmd.index("-md") + 1].endswith("draft-q8.gguf") and cmd[cmd.index("--spec-draft-n-max") + 1] == "4" and cmd[cmd.index("-lv") + 1] == "4"
    summ = s["summary"]
    assert summ["on_vs_off"]["all"]["speedup"] == pytest.approx(1.1) and summ["on_vs_off"]["all"]["text_equal_frac"] == 1.0
    assert set(summ["on_vs_off"]["by_category"]) == set(make_prompts.R2A_CATEGORIES)
    assert summ["vs_base"]["k4q8_off"]["all"]["speedup"] == pytest.approx(1.2) and summ["vs_base"]["k4q8_on"]["all"]["speedup_ci95"]
    sp_on = summ["step_split"]["k4q8_on"]
    assert sp_on["draft_ms_per_call"] == pytest.approx(2.0) and sp_on["impls"]["draft-simple"]["n_requests"] == 6
    assert any("k4q8_on over k4q8_off 1.100x" in n for n in d["notes"]) and any("width 4: on over off +100.0%" in n for n in d["notes"])
    assert "hf_secretsecret123" not in json.dumps(d) and not (wpod / "results" / "wrap_a" / "pro6000-ws" / "raw" / "partial.json").exists()
    raw = {f.name.split("-", 1)[1] for f in (wpod / "results" / "wrap_a" / "pro6000-ws" / "raw").glob("*-server-*.log")}
    assert {"server-b0-base.log", "server-b1-k4q8_on.log"} <= raw


def test_wrap_a_refuses_a_max_q_and_stops_on_a_failing_kernel(wpod, monkeypatch):
    monkeypatch.setenv("CR_GPU_TAG", "pro6000-maxq")
    assert wrap_a.main(["--budget-min", "6"]) == 1
    d = _result(wpod, "wrap_a", "pro6000-maxq")["data"]
    assert "not a 600 W" in d["gate"]["problem"] and d["bench"] == [] and d["tbo"] == [] and d["spec"] is None
    monkeypatch.setenv("CR_GPU_TAG", "pro6000-server")
    monkeypatch.setenv("FAKE_TBO_FAIL", "1")
    assert wrap_a.main(["--budget-min", "6"]) == 1
    d = _result(wpod, "wrap_a", "pro6000-server")["data"]
    assert d["bench"] == [] and d["spec"] is None and any(n.startswith("stop: test-backend-ops did not pass") for n in d["notes"])


def test_wrap_a_unhealthy_host_is_not_pr_evidence(wpod, monkeypatch):
    # a throttled round: the first unhealthy round and the later ones are left out, the verdict says so, the exit code is 1
    calls = {"n": 0}

    def health(smi, check_power=True):
        calls["n"] += 1
        return "power cap 400 W < 75% of default 600 W" if calls["n"] >= 5 else None       # 4 runs per round: the 5th is round 1
    monkeypatch.setattr(mc.e4_pdl, "health_problem", health)
    monkeypatch.setattr(wrap_a.r1, "host_health", lambda smi, probe_smi, check_power=True: (None, None))      # healthy before the timing
    assert wrap_a.main(["--parts", "b", "--budget-min", "10"]) == 1
    d = _result(wpod, "wrap_a", "pro6000-ws")["data"]
    assert d["bench_health"]["first_bad_round"] == 1 and "power cap" in d["bench_health"]["problem"]
    assert d["bench_paired"]["pp"]["4"]["rounds"] == [0] and d["bench_summary"]["pp"]["4"]["rounds"] == [0]
    assert any(r["unhealthy"] for r in d["bench"]) and not any(r["unhealthy"] for r in d["bench"] if r["round"] == 0)
    verdict = [n for n in d["notes"] if "width 4: on over off" in n]
    assert verdict and verdict[0].startswith("UNHEALTHY HOST (power cap")


def test_wrap_a_skips_the_timing_on_an_unhealthy_host(wpod, monkeypatch):
    monkeypatch.setattr(wrap_a.r1, "host_health", lambda smi, probe_smi, check_power=True: ("SM clock 1200 MHz under load", None))
    assert wrap_a.main(["--budget-min", "10", "--per-category", "2", "--blocks", "2"]) == 1
    d = _result(wpod, "wrap_a", "pro6000-ws")["data"]
    assert d["bench"] == [] and d["spec"] is None and d["host_b"]["problem"] and d["host"]["problem"]
    assert sum(1 for n in d["notes"] if n.startswith("UNHEALTHY HOST")) == 2          # part b and part s


def test_wrap_b_end_to_end_with_fakes(wpod):
    assert wrap_b.main(["--budget-min", "10", "--per-category", "2", "--blocks", "2"]) == 0
    d = _result(wpod, "wrap_b", "pro6000-ws")["data"]
    runs_p = d["probe"]["runs"]
    assert set(runs_p) == {"n4", "n16", "n64"} and set(runs_p["n4"]["models"]) == {"draft_q8_0", "pq2"} and set(runs_p["n64"]["models"]) == {"pq2"}
    assert runs_p["n4"]["models"]["pq2"]["verdict"]["single"]["max_abs_diff"] == 13.0 and "--modes" in runs_p["n64"]["models"]["pq2"]["cmd"]
    m = d["matrix"]
    assert set(m["runs"]) == set(wrap_b.CONFIG_NAMES) and all(len(r["rows"]) == 6 for r in m["runs"].values())
    summ = m["summary"]
    assert all(v["ok"] for v in summ["paths"].values()), summ["paths"]
    p = summ["pairs"]
    assert p["k4q8_ckpt_vs_k4q8"]["all"]["speedup"] == pytest.approx(1.2) and p["k4q8_ckpt_t_vs_k4q8_ckpt"]["all"]["speedup"] == pytest.approx(1.0)
    assert p["ngram_rs16_vs_ngram"]["all"]["speedup"] == pytest.approx(1.1) and p["ngram_rs64_vs_ngram"]["all"]["speedup_ci95"]
    assert set(p["ngram_rs64_vs_ngram"]["by_category"]) == set(make_prompts.R2A_CATEGORIES)
    assert summ["ngram_cut"]["ngram_rs16"]["ngram-mod"]["rs_window"] == 16 and summ["ngram_cut"]["ngram_rs16"]["ngram-mod"]["cut_tokens"] == 60
    assert "resets" in summ["ngram_cut"]["ngram"]["ngram-mod"] and "cut_drafts" not in summ["ngram_cut"]["ngram"]["ngram-mod"]
    assert set(summ["reeval"]) == {"k4q8_ckpt_t"} and summ["reeval"]["k4q8_ckpt_t"]["reeval_ms_per_call"] > 0
    assert summ["draft_state"]["reference"] == "k4q8_ckpt" and all(summ["draft_state"]["matches_reference"].values())
    assert summ["vs_base"]["ngram"]["checks"]["rollback_path"] == "checkpoint" and summ["vs_base"]["ngram_rs64"]["checks"]["n_rs_seq_target"] == 64
    assert m["runs"]["k4q8_ckpt"]["blocks"][0]["exp_switches"] == [wrap_b.CKPT] and m["runs"]["k4q8"]["blocks"][0]["exp_switches"] == []
    assert m["runs"]["k4q8_ckpt_t"]["blocks"][0]["exp_switches"] == [wrap_b.CKPT, wrap_b.TIME]
    assert d["config"]["envs"]["ngram_rs64"] == {wrap_b.NGRAM_RS: "64"} and wrap_b.NOTE_EQ in d["notes"]
    assert any(n.startswith("draft state: alpha") for n in d["notes"]) and any("ms per step" in n and "at most" in n and "at least" in n for n in d["notes"])
    assert "rpa_secretsecret456" not in json.dumps(d)


def test_wrap_b_runs_on_a_maxq_and_refuses_an_ada_card(wpod, monkeypatch):
    monkeypatch.setenv("CR_GPU_TAG", "pro6000-maxq")                 # not PR evidence: no 600 W gate
    assert wrap_b.main(["--parts", "r", "--budget-min", "6"]) == 0
    assert _result(wpod, "wrap_b", "pro6000-maxq")["data"]["gate"]["problem"] is None
    monkeypatch.setenv("CR_GPU_TAG", "rtx4090")
    assert wrap_b.main(["--parts", "r", "--budget-min", "6"]) == 1
    assert "not one of" in _result(wpod, "wrap_b", "rtx4090")["data"]["gate"]["problem"]


def test_wrap_b_flags_a_switch_that_did_not_take_effect(wpod, monkeypatch):
    monkeypatch.setenv("FAKE_IGNORE_EXP", "1")
    assert wrap_b.main(["--parts", "m", "--configs", "base,ngram_rs16,k4q8_ckpt_t", "--budget-min", "6", "--per-category", "2", "--blocks", "2"]) == 1
    d = _result(wpod, "wrap_b", "pro6000-ws")["data"]
    paths = d["matrix"]["summary"]["paths"]
    assert paths["base"]["ok"] is True and paths["ngram_rs16"]["ok"] is False and paths["k4q8_ckpt_t"]["ok"] is False
    assert any("path verification failed" in n for n in d["notes"])


def test_wrap_a3_end_to_end_with_a_fake_base_build(wpod, monkeypatch):
    monkeypatch.setenv("CR_GPU_TAG", "rtx4090")
    base = wpod / "base" / "bin"
    base.mkdir(parents=True)
    for t in ("llama-bench", "test-backend-ops"):
        shutil.copy(wpod / "bin" / t, base / t)

    def fake_build(args, budget, notes, doc, config, head_bin):
        doc["build"] = {"applied": True, "differs": True}
        return base, str(wpod / "target.gguf")
    monkeypatch.setattr(wrap_a3, "build_part", fake_build)
    monkeypatch.setattr(wrap_a3, "find_cuobjdump", lambda: None)
    assert wrap_a3.main(["--budget-min", "10"]) == 0
    d = _result(wpod, "wrap_a3", "rtx4090")["data"]
    assert d["sass"]["error"] == "cuobjdump not found" and any("sass: cuobjdump not found" in n for n in d["notes"])
    assert [t["config"] for t in d["tbo"]] == ["head", "base"] and all(mc.tbo_passed(t) for t in d["tbo"])
    assert len(d["bench"]) == 4 * 2 * 10 and {r["config"] for r in d["bench"]} == {"base", "head"}
    assert d["bench_paired"]["a"] == "base" and d["bench_paired"]["pp"]["4"]["median"] == pytest.approx(0.0)
    assert d["perf"]["gain"] and all(g["gain_pct"] == pytest.approx(0.0) for g in d["perf"]["gain"])
    assert d["bench_summary"]["pp"]["4"]["gain_pct_head"] == pytest.approx(0.0)
    monkeypatch.setenv("CR_GPU_TAG", "rtx5090")
    assert wrap_a3.main(["--budget-min", "6"]) == 1
    assert "not an RTX 4090" in _result(wpod, "wrap_a3", "rtx5090")["data"]["gate"]["problem"]


SASS = """
        Function : _Z22mul_mat_vec_pq2_0_mc4Pv
        .headerflags    @"EF_CUDA_SM89"
        /*0000*/                   MOV R1, c[0x0][0x28] ;                         /* 0x00000a0000017a02 */
                                                                                   /* 0x000fc40000000f00 */
        /*0010*/                   S2R R0, SR_TID.X ;                             /* 0x0000000000007919 */
                                                                                   /* 0x000e220000002100 */
        Function : _Z6othercPv
        /*0000*/                   NOP ;                                          /* 0x0000000000007918 */
"""


def test_sass_kernels_compare_by_instruction_text():
    k = wrap_a3.sass_kernels(SASS.splitlines(True))
    assert list(k) == ["_Z22mul_mat_vec_pq2_0_mc4Pv"] and k["_Z22mul_mat_vec_pq2_0_mc4Pv"] == ["MOV R1, c[0x0][0x28] ;", "S2R R0, SR_TID.X ;"]
    moved = SASS.replace("/*0010*/", "/*0020*/").replace("0x000e220000002100", "0x1111111111111111")
    assert wrap_a3.sass_compare(k, wrap_a3.sass_kernels(moved.splitlines(True)))["identical"] is True
    other = wrap_a3.sass_kernels(SASS.replace("SR_TID.X", "SR_TID.Y").splitlines(True))
    c = wrap_a3.sass_compare(k, other)
    assert c["identical"] is False and c["different"] == ["_Z22mul_mat_vec_pq2_0_mc4Pv"] and wrap_a3.sass_compare({}, {})["identical"] is False


def test_sass_part_with_a_fake_cuobjdump(tmp_path, monkeypatch):
    tool = tmp_path / "cuobjdump"
    tool.write_text("#!%s\nimport sys\nprint(open(sys.argv[-1]).read())\n" % sys.executable)
    tool.chmod(tool.stat().st_mode | stat.S_IXUSR)
    for n, txt in (("head", SASS), ("base", SASS), ("diff", SASS.replace("SR_TID.X", "SR_TID.Y"))):
        (tmp_path / n).mkdir()
        (tmp_path / n / "libggml-cuda.so").write_text(txt)
    monkeypatch.setattr(wrap_a3, "find_cuobjdump", lambda: str(tool))
    notes = []
    out = wrap_a3.sass_part({"head": tmp_path / "head", "base": tmp_path / "base"}, "89", common_budget(), notes)
    assert out["identical"] is True and out["arch"] == "sm_89" and notes[0].startswith("sass mul_mat_vec_pq2_0_mc sm_89: identical in both builds")
    out = wrap_a3.sass_part({"head": tmp_path / "head", "base": tmp_path / "diff"}, "89", common_budget(), notes)
    assert out["identical"] is False and "NOT identical" in notes[1]


def common_budget():
    from exps import common
    return common.Budget(2.0)


def test_wrap_costs_fit_the_trimmed_estimates():
    # the cost model of the server part at the defaults: wrap_a about 4 min, wrap_b about 16 min; both inside their budgets
    rows = make_prompts.load_longctx(make_prompts.R2A_OUT)
    for mod, cfgs, per, nb, part_w in ((wrap_a, wrap_a.spec_configs(), wrap_a.PER_CATEGORY, wrap_a.BLOCKS, "s"),
                                       (wrap_b, wrap_b.matrix_configs(), sp.PER_CATEGORY, wrap_b.BLOCKS, "m")):
        blocks = sp.prompt_blocks(sp.balanced_subset(rows, sp.SEED, per), nb)
        est = sum(r2a.config_cost_s(c, b) for b in blocks for c in cfgs) / 60.0
        share = (mod.DEFAULT_TOTAL_MIN - 25 / 60.0) * mod.WEIGHTS[part_w] / sum(mod.WEIGHTS.values())
        assert est <= share * 1.05, (mod.EXP, est, share)


def test_rs_rollback_cpu_configs_and_summary():
    cfgs = {c["name"]: c for c in rs_rollback_cpu.server_configs()}
    assert len(cfgs) == len(rs_rollback_cpu.server_configs()) and all(c["base"] in cfgs for c in cfgs.values())
    assert cfgs["k4_skip_near"]["model"] == "target" and cfgs["k4_skip_near"]["env"] == {wrap_b.SKIP: "1"}          # hybrid draft: ignored
    assert cfgs["k4_l_skip_near"]["model"] == "l" and cfgs["k4_l_skip_near"]["draft"] == "lnear" and cfgs["k4_l_near"]["env"] == {}
    assert cfgs["k4_ckpt_t_near"]["env"] == {wrap_b.CKPT: "1", wrap_b.TIME: "1"} and cfgs["ngram_rs8_mid"]["env"] == {wrap_b.NGRAM_RS: "8"}
    assert cfgs["ngram_mid"]["base"] == "base_mid" and cfgs["k4_l_same"]["base"] == "base_l"

    def srv(acc, drafted, partial=0, **kw):
        return dict({"acceptance": acc, "drafted": drafted, "accepted": int((acc or 0) * drafted), "ignored": False, "text_equal_base": 3, "text_compared": 3,
                     "steps": {"verified": 10, "partial": partial, "restored": 0, "rolled_back_tokens": 4 * partial}, "path": {"rollback_path": "in_batch"},
                     "stats": {}}, **kw)
    s = {"k4_near": srv(0.001, 1000), "k4_ckpt_near": srv(0.106, 800, 190), "k4_skip_near": srv(0.001, 1000, ignored=True),
         "k4_same": srv(0.011, 1000), "k4_ckpt_same": srv(1.0, 200),
         "k4_l_near": srv(0.329, 487, 107), "k4_l_skip_near": srv(0.329, 487, 107, stats={"draft-simple": {"skipped_batches": 125, "catchup_tokens": 15}}),
         "ngram_mid": srv(None, 158, 2), "ngram_rs8_mid": srv(None, 137, 2)}
    notes = rs_rollback_cpu.summary_notes(s)
    assert any("hybrid draft near: acceptance k4_near 0.001, k4_ckpt_near 0.106" in n and "stale rollback" in n for n in notes)
    assert any("no rollback happens" in n and "k4_same 0.011" in n for n in notes)                       # the 'same' row is not rollback evidence
    assert any("skip switch ignored: True (acceptance equals k4_near: True)" in n for n in notes)
    assert any("attention-only draft near: acceptance k4_l 0.329, k4_l_skip 0.329" in n and "skipped batches 125" in n for n in notes)
    assert any(n.startswith("ngram_rs8_mid: drafted 137") and "partial steps 2 of 10" in n for n in notes)
    import inspect
    src = inspect.getsource(rs_rollback_cpu.run)
    assert '"bin_dir"' not in src and '"build": pathlib.Path(bin_dir).parent.name' in src                # the result keeps the build name, not its path


@pytest.mark.parametrize("mod", [wrap_a, wrap_a3, wrap_b, rs_rollback_cpu])
def test_dry_runs(mod, capsys):
    assert mod.main(["--dry-run"]) == 0
    plan = json.loads(capsys.readouterr().out)
    assert plan["exp"] == mod.EXP


def test_args():
    with pytest.raises(SystemExit):
        wrap_a.parse_args(["--rounds", "2"])
    with pytest.raises(SystemExit):
        wrap_b.parse_args(["--configs", "k4q8"])
    with pytest.raises(SystemExit):
        wrap_a3.parse_args(["--parts", "build,x"])
    assert wrap_a.parse_args(["--parts", "t,b"]).parts == "tb" and wrap_b.parse_args(["--quick"]).blocks == 2
