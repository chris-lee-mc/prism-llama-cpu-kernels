"""R0 + R1 drivers, offline: pure helpers, committed-result resummary, fake binaries for the end-to-end run (no GPU, no model)."""
import json
import os
import pathlib
import re
import sqlite3
import stat
import sys

import pytest

from exps import common, e5_cost, e5_spec, fetch_models, make_prompts, r1, r1_agree, r1_step, spec_math

EXP = pathlib.Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------- spec_math

def test_alpha_r_roundtrip_and_known_value():
    for k in (1, 2, 4, 8):
        for a in (0.1, 0.48, 0.9):
            assert spec_math.alpha_from_r(spec_math.r_from_alpha(a, k), k) == pytest.approx(a, abs=1e-6)
    # E5 K=4 run: r 0.2158 is alpha 0.477
    assert spec_math.alpha_from_r(0.2157567, 4) == pytest.approx(0.477, abs=0.002)
    assert spec_math.accept_sum(1.0, 4) == 4 and spec_math.r_from_alpha(1.0, 4) == 1.0
    assert spec_math.alpha_from_r(None, 4) is None and spec_math.alpha_from_r(0.0, 4) == 0.0 and spec_math.alpha_from_r(1.0, 4) == 1.0


def test_run_stats_counts():
    # E5 K=4 50-prompt run: 7828 tokens, 16704 drafted, 3604 accepted
    st = spec_math.run_stats(7828, 16704, 3604)
    assert st["steps"] == 4224 and st["k_eff"] == pytest.approx(3.955, abs=0.001)
    assert st["r"] == pytest.approx(0.2158, abs=1e-4) and 0.45 < st["alpha"] < 0.5
    assert st["tokens_per_step"] == pytest.approx(7828 / 4224)
    plain = spec_math.run_stats(100, 0, 0)
    assert plain["r"] is None and plain["alpha"] is None and plain["steps"] == 100


def test_run_stats_first_token_and_ngram():
    # 3 requests: the first token of each is sampled from prefill, not by a decode step
    st = spec_math.run_stats(300, 800, 100, n_first=3)
    assert st["steps"] == 197 and st["tokens_per_step"] == pytest.approx(297 / 197) and st["alpha"] is not None
    ng = spec_math.run_stats(300, 800, 100, n_first=3, with_alpha=False)
    assert ng["alpha"] is None and ng["k_eff"] is None and ng["r"] == pytest.approx(0.125) and ng["steps"] == 197
    assert spec_math.observed_step_ms(1970.0, 300, 100, n_first=3) == pytest.approx(10.0)


def test_bootstrap_speedup_ci_brackets_the_ratio_and_is_deterministic():
    pairs = [(100.0, 1000.0 / (1.5 + 0.1 * (i % 3)), 100.0, 1000.0) for i in range(30)]
    ci = spec_math.bootstrap_speedup_ci(pairs)
    point = (sum(p[0] for p in pairs) / sum(p[1] for p in pairs)) / (sum(p[2] for p in pairs) / sum(p[3] for p in pairs))
    assert ci[0] <= point <= ci[1] and ci == spec_math.bootstrap_speedup_ci(pairs)
    assert spec_math.bootstrap_speedup_ci(pairs[:1]) is None
    noisy = [(100.0, 100.0 * (1 + (-1) ** i * 0.9), 100.0, 100.0) for i in range(8)]
    lo, hi = spec_math.bootstrap_speedup_ci(noisy)
    assert hi - lo > 0.5


def test_break_even_alpha_matches_closed_form():
    # free draft, verify costs 2x a plain step, K=4: 1 + A = 2 -> alpha 0.519
    assert spec_math.break_even_alpha(4, 0.0, 16.0, 8.0) == pytest.approx(0.519, abs=0.002)
    assert spec_math.break_even_alpha(4, 100.0, 16.0, 8.0) is None
    # the draft re-eval of the verify batch adds to every step and raises the bar
    assert spec_math.break_even_alpha(4, 1.6, 13.6, 7.0, 3.0) > spec_math.break_even_alpha(4, 1.6, 13.6, 7.0)


# ---------------------------------------------------------------- R0: e5_cost

def test_rows_from_uses_median_of_samples_and_keeps_them():
    e = {"n_prompt": 0, "n_gen": 128, "avg_ts": 100.0, "stddev_ts": 40.0, "samples_ts": [140.0, 139.0, 141.0, 30.0, 138.0]}
    r = e5_cost.rows_from([e], "pq2", 512)[0]
    assert r["tok_s"] == 139.0 and r["tok_s_mean"] == 100.0 and r["samples_ts"] == e["samples_ts"]
    assert r["kind"] == "decode" and r["ms_per_batch"] == pytest.approx(1e3 / 139.0)
    old = e5_cost.rows_from([{"n_prompt": 5, "n_gen": 0, "avg_ts": 500.0}], "x", 512)[0]
    assert old["tok_s"] == 500.0 and old["samples_ts"] == []


def _rows(plain=True):
    rows = [{"model": "pq2", "kind": "verify", "width": w, "ms_per_batch": 8.6 + w} for w in range(1, 10)]
    if plain:
        rows.append({"model": "pq2", "kind": "decode", "width": 1, "ms_per_batch": 7.0, "n_gen": 128})
    return rows + [{"model": "draft_bf16", "kind": "decode", "width": 1, "ms_per_batch": 1.6}]


def test_summarize_baseline_is_tg_not_width_one_prompt_path():
    s = e5_cost.summarize(_rows())
    assert s["baseline"]["plain_ms"] == 7.0 and "tg" in s["baseline"]["source"]
    assert s["target_verify_w1_prompt_path_ms"] == pytest.approx(9.6)
    be = s["drafts"]["draft_bf16"]["break_even"][4]
    assert be["alpha"] == pytest.approx(spec_math.break_even_alpha(4, 1.6, 8.6 + 5, 7.0), abs=1e-9)
    assert be["r"] == pytest.approx(spec_math.r_from_alpha(be["alpha"], 4))
    assert set(s["drafts"]["draft_bf16"]["break_even"]) == set(e5_cost.KS)


def test_summarize_step_model_includes_the_draft_reeval():
    rows = _rows() + [{"model": "draft_bf16", "kind": "verify", "width": w, "ms_per_batch": 1.7 + 0.1 * w} for w in range(1, 10)]
    d = e5_cost.summarize(rows)["drafts"]["draft_bf16"]
    assert d["reeval_included"] and d["t_draft_batch_ms"][5] == pytest.approx(2.2)
    assert d["t_step_ms"][4] == pytest.approx(4 * 1.6 + 2.2 + 13.6) and d["t_step_ms_no_reeval"][4] == pytest.approx(4 * 1.6 + 13.6)
    assert d["break_even"][4]["alpha"] == pytest.approx(spec_math.break_even_alpha(4, 1.6, 13.6, 7.0, 2.2), abs=1e-9)
    assert d["break_even"][4]["alpha"] > d["break_even_no_reeval"][4]["alpha"]
    old = e5_cost.summarize(_rows())["drafts"]["draft_bf16"]      # results from before the draft width rows
    assert not old["reeval_included"] and old["break_even"] == old["break_even_no_reeval"]


def test_e5_cost_plan_has_draft_batch_cmd(capsys):
    assert e5_cost.main(["--dry-run"]) == 0
    c = json.loads(capsys.readouterr().out)["draft_batch_cmd"]
    assert c[c.index("-p") + 1] == "1,2,3,4,5,6,7,8,9" and c[c.index("-n") + 1] == "0" and c[c.index("-d") + 1] == "512"


def test_summarize_without_tg_row_needs_a_supplied_baseline():
    s = e5_cost.summarize(_rows(plain=False))
    assert "error" in s and s["baseline"]["plain_ms"] is None and s["drafts"] == {}
    s2 = e5_cost.summarize(_rows(plain=False), plain_ms=7.2, plain_source="supplied")
    assert s2["baseline"]["source"] == "supplied" and "draft_bf16" in s2["drafts"]


def test_resummarize_committed_result(tmp_path):
    src = EXP / "results" / "e5_cost" / "pro6000-ws" / "20261013T183456Z.json"
    base = EXP / "results" / "e5_base" / "pro6000-ws" / "20261013T183917Z.json"
    pm, why = e5_cost.plain_ms_from_e5(base)
    assert pm == pytest.approx(7.194, abs=0.001) and "tok_s_weighted" in why
    doc = e5_cost.resummarize(src, pm, why)
    s = doc["data"]["summary"]
    assert s["baseline"]["plain_ms"] == pytest.approx(7.194, abs=0.001)
    assert s["target_verify_w1_prompt_path_ms"] == pytest.approx(8.642, abs=0.001)
    # the old file gave 0.528 against the 8.64 ms width-1 prompt path; against 7.19 ms the bar is higher
    be = s["drafts"]["draft_bf16"]["break_even"][4]
    assert be["alpha"] == pytest.approx(0.631, abs=0.002) and be["r"] == pytest.approx(0.36, abs=0.005)
    assert any("per-rep samples were not saved" in n for n in doc["data"]["notes"])
    # CLI path writes next to the source
    cp = tmp_path / "r.json"
    cp.write_text(src.read_text())
    assert e5_cost.main(["--resummarize", str(cp), "--plain-ms", "7.19"]) == 0
    out = json.loads((tmp_path / "r.resummary.json").read_text())
    assert out["kind"] == "resummary" and out["derived_from"] == "r.json"
    assert e5_cost.main(["--resummarize", str(cp)]) == 1   # no baseline: refuses to invent one


def test_committed_resummary_matches_the_code():
    f = EXP / "results" / "e5_cost" / "pro6000-ws" / "20261013T183456Z.resummary.json"
    d = json.loads(f.read_text())
    assert d["derived_from"] == "20261013T183456Z.json"
    s = d["data"]["summary"]
    assert s["baseline"]["plain_ms"] == pytest.approx(7.194, abs=0.001) and "E5 target-alone" in s["baseline"]["source"]
    assert s["drafts"]["draft_bf16"]["break_even"]["4"]["alpha"] == pytest.approx(0.631, abs=0.002)


def test_e5_cost_plan_has_plain_cmd(capsys):
    assert e5_cost.main(["--dry-run"]) == 0
    plan = json.loads(capsys.readouterr().out)
    c = plan["plain_cmd"]
    assert c[c.index("-p") + 1] == "0" and c[c.index("-n") + 1] == "128" and c[c.index("-d") + 1] == "512"


# ---------------------------------------------------------------- prompts, matrix configs

def test_shuffled_is_a_fixed_permutation_and_stratified_prefixes_balance():
    rows = make_prompts.load()
    a, b = make_prompts.shuffled(rows, 7), make_prompts.shuffled(rows, 7)
    assert [r["id"] for r in a] == [r["id"] for r in b] and sorted(r["id"] for r in a) == sorted(r["id"] for r in rows)
    assert [r["id"] for r in a] != [r["id"] for r in rows]
    assert [r["id"] for r in make_prompts.shuffled(rows, 8)] != [r["id"] for r in a]
    st = make_prompts.shuffled(rows, 7, stratify=True)
    assert sorted(r["id"] for r in st) == sorted(r["id"] for r in rows)
    pre = [r["category"] for r in st[:40]]
    assert all(pre.count(c) == 5 for c in set(pre)) and len(set(pre)) == 8
    assert [r["id"] for r in st] == [r["id"] for r in make_prompts.shuffled(rows, 7, stratify=True)]


def test_matrix_configs_and_commands():
    cfgs = {c["name"]: c for c in e5_spec.matrix_configs()}
    assert list(cfgs)[0] == "base"
    assert {"base", "k2", "k3", "k4", "k4_tbs", "k4_ptq1", "ngram_mod", "ngram_map_k"} <= set(cfgs)
    cmd = lambda n: e5_spec.config_cmd("/b", "/t.gguf", "/d.gguf", cfgs[n])
    assert cmd("base") == e5_spec.server_cmd("/b", "/t.gguf", 0)
    k4 = cmd("k4")
    assert k4 == e5_spec.server_cmd("/b", "/t.gguf", 4, "/d.gguf")          # the control is the E5 server, no extra flag
    assert k4[k4.index("--spec-type") + 1] == "draft-simple" and k4[k4.index("--spec-draft-n-max") + 1] == "4" and k4[k4.index("-md") + 1] == "/d.gguf"
    assert cmd("k4_tbs") == k4 + ["-bs"]                                  # target backend sampling is the only difference
    assert not any("draft-backend-sampling" in a for c in cfgs for a in cmd(c))
    for n, k in (("k2", "2"), ("k3", "3")):
        assert cmd(n)[cmd(n).index("--spec-draft-n-max") + 1] == k
    assert cfgs["k4_ptq1"]["draft"] == "ptq1_0" and cfgs["k4"]["draft"] == "bf16" and cfgs["k4_ptq1"]["frac"] == 0.5
    nm = cmd("ngram_mod")
    assert nm[-2:] == ["--spec-type", "ngram-mod"] and "-md" not in nm and "--spec-draft-n-max" not in nm
    assert cmd("ngram_map_k")[-2:] == ["--spec-type", "ngram-map-k"]
    assert cmd("ngram_mod_short")[-8:-6] == ["--spec-type", "ngram-mod"] and "--spec-ngram-mod-n-match" in cmd("ngram_mod_short")
    assert "--spec-ngram-map-k-size-n" in cmd("ngram_map_k_short")
    for n in ("ngram_mod", "ngram_map_k", "ngram_mod_short", "ngram_map_k_short"):
        assert cfgs[n]["categories"] == ["code_fix", "refactor"]


def test_flags_exist_in_this_forks_arg_cpp():
    src = (EXP.parent.parent / "common" / "arg.cpp").read_text()
    for flag in (e5_spec.TARGET_BS, "--backend-sampling", "--spec-type", "--spec-draft-n-max", "--spec-ngram-mod-n-match",
                 "--spec-ngram-mod-n-min", "--spec-ngram-mod-n-max", "--spec-ngram-map-k-size-n", "--spec-ngram-map-k-size-m",
                 "--spec-ngram-map-k-min-hits", "--kl-divergence-base", "--kl-divergence", "--chunks"):
        assert f'"{flag}"' in src, flag
    spec = (EXP.parent.parent / "common" / "speculative.cpp").read_text()
    assert '"ngram-mod"' in spec and '"ngram-map-k"' in spec and '"draft-simple"' in spec


def test_draft_simple_never_reads_the_draft_backend_sampling_flag():
    """The reason k4_nobs was dropped: in this fork only eagle3, dflash and mtp read draft.backend_sampling."""
    spec = (EXP.parent.parent / "common" / "speculative.cpp").read_text()
    a = spec.index("struct common_speculative_impl_draft_simple")
    b = spec.index("\nstruct ", a + 10)
    assert "backend_sampling" not in spec[a:b] and "common_sampler_sample(" in spec[a:b]


def test_plan_n_fits_the_budget_and_is_category_balanced():
    prompts = make_prompts.shuffled(make_prompts.load(), e5_spec.SEED, stratify=True)
    cfgs = e5_spec.matrix_configs()
    big = e5_spec.plan_n(cfgs, prompts, 1540.0)
    assert big["fits"] and big["est_s"] <= 0.9 * 1540.0 and big["n"] >= 100
    assert big["min_code_category"] >= e5_spec.MIN_N_CATEGORY - 1 and max(big["per_category"].values()) - min(big["per_category"].values()) <= 1
    assert big["per_config"]["base"] == big["n"] and big["per_config"]["k2"] == (big["n"] + 1) // 2
    assert big["per_config"]["ngram_mod"] == big["per_category"]["code_fix"] + big["per_category"]["refactor"]
    assert e5_spec.plan_n(cfgs, prompts, 100000.0)["n"] == 200
    small = e5_spec.plan_n(cfgs, prompts, 400.0)
    assert small["n"] < big["n"] and small["est_s"] <= 0.9 * 400.0
    tiny = e5_spec.plan_n(cfgs, prompts, 30.0)
    assert not tiny["fits"] and tiny["n"] == e5_spec.N_FLOOR
    only_base = e5_spec.plan_n(cfgs[:1], prompts, 1540.0)
    assert only_base["n"] == 200 or only_base["n"] > big["n"]
    sel = e5_spec.select_prompts(next(c for c in cfgs if c["name"] == "k2"), prompts[:40])
    assert sel == prompts[:20]


def test_time_shares_follow_cost_and_part_weights():
    cfgs = e5_spec.matrix_configs()[:3]
    sh = e5_spec.time_shares([e5_spec.cost_s(c, 100) for c in cfgs], 500.0)
    assert sum(sh) == pytest.approx(500.0) and sh[1] > sh[0] and sh[1] == pytest.approx(sh[2])
    assert e5_spec.cost_s(cfgs[0], 0) == 0.0
    assert r1.part_budget_s(100.0, "b", ["b", "a", "c"]) == pytest.approx(75.0 * 26 / 37)
    assert r1.part_budget_s(100.0, "c", ["c"]) == pytest.approx(75.0)
    assert r1.PART_ORDER == "bac"                      # a needs the target-alone outputs of b


def _row(i, cat, text, tps=100.0, toks=100, drafted=0, acc=0):
    return {"id": f"p{i:03d}", "category": cat, "tokens_predicted": toks, "predicted_ms": toks / tps * 1e3, "predicted_per_second": tps,
            "draft_n": drafted, "draft_n_accepted": acc, "text": text}


def test_summarize_matrix_pairs_prompts_and_compares_text():
    base = [_row(0, "code_fix", "aaaa"), _row(1, "code_fix", "bbbb"), _row(2, "refactor", "cccc"), _row(3, "refactor", "dddd")]
    spec = [_row(0, "code_fix", "aaaa", 150.0, 100, 200, 40), _row(1, "code_fix", "bbxx", 150.0, 100, 200, 40),
            _row(2, "refactor", "cccc", 50.0, 100, 200, 10), {"id": "p009", "category": "plan", "error": "x"}]
    runs = {"base": {"rows": base, "planned": 4}, "k4": {"rows": spec, "planned": 4, "k": 4, "spec_type": "draft-simple"}}
    s = e5_spec.summarize_matrix(runs)
    k4 = s["k4"]
    assert k4["n_ok"] == 3 and k4["n_paired"] == 3 and k4["error"] is None
    cf = k4["by_category"]["code_fix"]
    assert cf["speedup"] == pytest.approx(1.5) and cf["text_equal"] == 1 and cf["text_compared"] == 2
    assert cf["median_prefix_frac"] == pytest.approx((1.0 + 0.5) / 2) and cf["r"] == pytest.approx(0.2)
    # 2 prompts: the first token of each is not a decode step, so steps = 200 - 80 - 2
    assert 0.0 < cf["alpha"] < 0.5 and cf["tokens_per_step"] == pytest.approx(198 / 118) and cf["k_eff"] == pytest.approx(400 / 118)
    assert k4["by_category"]["refactor"]["speedup"] == pytest.approx(0.5)
    assert k4["all"]["n"] == 3 and k4["all"]["speedup"] == pytest.approx(0.9)
    assert s["base"]["all"]["speedup"] == pytest.approx(1.0) and s["base"]["all"]["text_equal_frac"] == 1.0


def test_ngram_configs_have_no_alpha_and_ci_is_optional():
    base = [_row(i, "code_fix", "t", 100.0, 100) for i in range(6)]
    ng = [_row(i, "code_fix", "t", 130.0, 100, 40, 20) for i in range(6)]
    runs = {"base": {"rows": base}, "ngram_mod": {"rows": ng, "spec_type": "ngram-mod"}}
    a = e5_spec.summarize_matrix(runs)["ngram_mod"]["all"]
    assert a["alpha"] is None and a["k_eff"] is None and a["r"] == pytest.approx(0.5) and a["tokens_per_step"] is not None
    assert "speedup_ci95" not in a
    ci = e5_spec.summarize_matrix(runs, ci=True)["ngram_mod"]["by_category"]["code_fix"]["speedup_ci95"]
    assert ci[0] == pytest.approx(1.3) and ci[1] == pytest.approx(1.3)           # identical prompts: zero spread
    runs["k4"] = {"rows": ng, "spec_type": "draft-simple", "k": 4}
    assert e5_spec.summarize_matrix(runs)["k4"]["all"]["alpha"] is not None


def test_compare_pair_uses_the_prompts_both_configs_finished():
    a = [_row(i, "plan", "t", 120.0, 100, 200, 40) for i in range(5)]
    b = [_row(i, "plan", "t", 100.0, 100, 200, 40) for i in range(3)]
    runs = {"k4_tbs": {"rows": a, "spec_type": "draft-simple"}, "k4": {"rows": b, "spec_type": "draft-simple"}}
    c = e5_spec.compare_pair(runs, "k4_tbs", "k4")
    assert c["n"] == 3 and c["speedup"] == pytest.approx(1.2) and c["alpha"] == pytest.approx(c["base_alpha"]) and "speedup_ci95" in c
    assert e5_spec.compare_pair(runs, "k4_tbs", "missing") is None


def test_log_flags_and_backend_sampling_report():
    f = e5_spec.log_flags("srv: ok\nllama_context: backend sampling not supported with SPLIT_MODE_TENSOR; using CPU\n"
                          "common_context_params: using KV checkpoints instead (raise -ub)\n")
    assert f["kv_checkpoint_fallback"] and len(f["backend_sampling_problems"]) == 1
    assert e5_spec.log_flags("all fine") == {"backend_sampling_problems": [], "kv_checkpoint_fallback": False}
    r = e5_spec.parse_completion({"tokens_predicted": 3, "generation_settings": {"backend_sampling": True}, "timings": {}})
    assert r["bs_flag"] is True and e5_spec.parse_completion({"tokens_predicted": 3, "timings": {}})["bs_flag"] is None


def test_step_residual_against_the_cost_model():
    cost = {"drafts": {"draft_bf16": {"t_draft_ms": 1.6}}, "target_verify_ms": {"5": 11.14}}
    r = e5_spec.step_residual({"ms_per_step": 22.0}, 4, cost)
    assert r["model_ms"] == pytest.approx(4 * 1.6 + 11.14) and r["residual_ms"] == pytest.approx(22.0 - 4 * 1.6 - 11.14)
    assert e5_spec.step_residual({"ms_per_step": 22.0}, 4, None) is None
    cost2 = json.loads(json.dumps(cost))
    cost2["target_verify_ms"] = {5: 11.14}
    assert e5_spec.step_residual({"ms_per_step": 22.0}, 4, cost2)["model_ms"] == pytest.approx(17.54)
    assert r["reeval_ms"] is None and r["residual_ms"] == r["residual_ms_no_reeval"]
    cost3 = json.loads(json.dumps(cost))
    cost3["drafts"]["draft_bf16"]["t_draft_batch_ms"] = {"5": 2.2}
    r3 = e5_spec.step_residual({"ms_per_step": 22.0}, 4, cost3)
    assert r3["reeval_ms"] == 2.2 and r3["model_ms"] == pytest.approx(17.54 + 2.2) and r3["residual_ms"] == pytest.approx(22.0 - 19.74)
    assert r3["residual_ms_no_reeval"] == pytest.approx(22.0 - 17.54)


def test_parse_completion_keeps_text_only_on_request():
    resp = {"content": "hi", "tokens_predicted": 2, "timings": {"predicted_per_second": 5.0, "predicted_ms": 400.0}}
    assert "text" not in e5_spec.parse_completion(resp) and e5_spec.parse_completion(resp, True)["text"] == "hi"
    s = e5_spec.summarize([e5_spec.parse_completion(dict(resp, timings=dict(resp["timings"], draft_n=8, draft_n_accepted=2)))], 4, 1.0)
    assert s["steps"] == 0 and s["alpha"] is None and s["acceptance"] == pytest.approx(0.25)


# ---------------------------------------------------------------- R1a

def test_build_text_is_deterministic_and_long_enough():
    t1, m1 = r1_agree.build_text()
    t2, m2 = r1_agree.build_text()
    assert t1 == t2 and m1 == m2 and len(t1) >= r1_agree.TEXT_CHARS
    assert m1["files"] and m1["n_tasks"] > 3 and "### Task 1" in t1 and "$ sed -n" in t1
    t3, _ = r1_agree.build_text(seed=1)
    assert t3 != t1
    assert len(t1) / 4 > r1_agree.CHUNKS * r1_agree.N_CTX / 2     # >= 8192 tokens at any plausible chars per token


KL_REPORT = """chunk             PPL
====== KL divergence statistics ======
Mean    KLD:   0.412345 \u00b1  0.003210
Maximum KLD:   14.5
99.0%   KLD:   3.25
Median  KLD:   0.0912
====== Token probability statistics ======
Same top p: 71.234 \u00b1 0.700 %
print_info: n_vocab          = 248320
"""


def test_parse_kld_report_and_quantize_log():
    p = r1_agree.parse_kld(KL_REPORT)
    assert p["same_top"] == pytest.approx(0.71234) and p["same_top_unc"] == pytest.approx(0.007)
    assert p["mean_kld"] == pytest.approx(0.412345) and p["median_kld"] == pytest.approx(0.0912) and p["kld_p99"] == 3.25
    assert p["n_vocab"] == 248320 and p["vocab_mismatch"] is None
    bad = r1_agree.parse_kld(KL_REPORT + "kl_divergence: inconsistent vocabulary (248320 vs 151936)\n")
    assert bad["vocab_mismatch"] == (248320, 151936)
    assert r1_agree.parse_kld("nothing")["same_top"] is None
    q = r1_agree.parse_quantize("converting to ptq1_0 .. size\nconverting to ptq1_0\nconverting to q8_0\nfalling back to q4_0\n"
                                "llama_model_quantize_impl: model size  =  1434.00 MiB\nllama_model_quantize_impl: quant size  =   400.00 MiB\n")
    assert q["tensor_types"] == {"ptq1_0": 2, "q8_0": 1} and q["n_fallback"] == 1 and q["dst_mib"] == 400.0 and q["src_mib"] == 1434.0


def test_agree_commands():
    b = r1_agree.perplexity_cmd("/b", "/m.gguf", "/t.txt", "/l.bin", kl=False)
    assert b[:3] == ["/b/llama-perplexity", "-m", "/m.gguf"] and "--kl-divergence" not in b
    assert b[b.index("--kl-divergence-base") + 1] == "/l.bin" and b[b.index("--chunks") + 1] == "4" and b[b.index("-c") + 1] == "2048"
    assert r1_agree.perplexity_cmd("/b", "/m.gguf", "/t.txt", "/l.bin", kl=True)[-1] == "--kl-divergence"
    assert r1_agree.quantize_cmd("/b", "a.gguf", "b.gguf", "PTQ1_0") == ["/b/llama-quantize", "a.gguf", "b.gguf", "PTQ1_0"]


def test_agree_summary_flags():
    v = {"bf16": {"same_top": 0.62}, "q8_0": {"same_top": 0.61}, "ptq1_0": {"same_top": 0.30}}
    s = r1_agree.summarize(v)
    assert s["bf16_ge_0p5"] is True and s["ptq1_within_0p05"] is False and s["ptq1_delta_vs_bf16"] == pytest.approx(-0.32)
    s2 = r1_agree.summarize({"bf16": {"same_top": 0.45}, "ptq1_0": {"same_top": 0.44}})
    assert s2["bf16_ge_0p5"] is False and s2["ptq1_within_0p05"] is True
    assert r1_agree.summarize({})["ptq1_within_0p05"] is None


# ---------------------------------------------------------------- R1c

def _synthetic_acts(steps=9):
    ms = 1_000_000
    acts, t = [], 0

    def burst(t0, dur, stream, n=20):
        d = dur // n
        for i in range(n):
            acts.append({"t0": t0 + i * d, "t1": t0 + i * d + d - 1000, "kind": "kernel", "stream": stream, "name": f"k{stream}"})
        return t0 + dur
    for _ in range(steps):
        t = burst(t, 10 * ms, 7)                                                    # verify 10 ms
        acts.append({"t0": t + 100_000, "t1": t + 300_000, "kind": "memcpy", "stream": 7, "bytes": 5_000_000, "copy_kind": "DtoH"})
        acts.append({"t0": t + 310_000, "t1": t + 330_000, "kind": "memcpy", "stream": 7, "bytes": 1 << 20, "copy_kind": "DtoD"})
        t += 1_200_000                                                              # cpu sampling gap
        t = burst(t, 2 * ms, 9)                                                     # draft re-eval 2 ms
        for _ in range(3):                                                          # 3 draft token steps of 1.6 ms
            t += 400_000
            t = burst(t, int(1.6 * ms), 9)
        t += 300_000
    return acts


def test_analyze_splits_a_synthetic_step():
    res = r1_step.analyze(_synthetic_acts(), 4)
    assert res["ok"] and res["n_long"] == 9 and res["n_steps"] == 6   # 2 skipped, 8 pairs left, capped by max_steps? 9 verifies: 7 used -> 6 steps
    m = res["median_ms"]
    assert m["target_verify"] == pytest.approx(9.98, abs=0.05) and m["draft_reeval"] == pytest.approx(2.0, abs=0.05)
    assert m["draft_gen"] == pytest.approx(3 * 1.6, abs=0.1)
    assert m["readback"] == pytest.approx(0.2, abs=0.01) and m["snapshot_copy"] == pytest.approx(0.02, abs=0.005)
    assert m["idle_after_verify"] == pytest.approx(1.0, abs=0.1) and m["idle_between_draft"] == pytest.approx(1.2, abs=0.1)
    assert m["wall"] == pytest.approx(m["target_verify"] + m["draft_gen"] + m["draft_reeval"] + m["readback"] + m["snapshot_copy"]
                                      + m["idle_total"], abs=0.1)
    parts = m["idle_intra_burst"] + m["idle_after_verify"] + m["idle_between_draft"] + m["idle_before_verify"]
    assert parts == pytest.approx(m["idle_total"], abs=0.05)
    assert res["shorts_per_step_median"] == 4 and res["checks"]["shorts_per_step_close_to_k"]
    assert m["c1_removable"] == pytest.approx(m["draft_reeval"] + m["readback"] + m["idle_after_verify"] + m["idle_between_draft"])
    assert not res["checks"]["c1_floor_le_2ms"] or m["c1_floor"] <= 2.0
    assert len(res["steps_ms"]) == res["n_steps"] and res["streams"]["7"]["busy_ms"] > res["streams"]["9"]["busy_ms"] * 0.5


def test_analyze_failure_modes():
    assert r1_step.analyze([], 4)["error"]
    flat = [{"t0": i * 1_000_000, "t1": i * 1_000_000 + 600_000, "kind": "kernel", "stream": 1, "name": "k"} for i in range(30)]
    r = r1_step.analyze(flat, 4)
    assert not r["ok"] and "split" in r["error"]
    few = r1_step.analyze(_synthetic_acts(steps=2), 4)
    assert not few["ok"] and "too few" in few["error"]


def _two_stream_acts(steps=9, gap_ns=5_000, branch=False):
    """Verify on stream 7, then the draft re-eval on stream 9 starting `gap_ns` after verify ends (abutting when small), then
    3 single-token draft passes. Optional branch stream 8 runs beside the verify pass (a ggml-cuda fork inside the target)."""
    ms = 1_000_000
    acts, t = [], 0

    def burst(t0, dur, stream, n=20):
        d = dur // n
        for i in range(n):
            acts.append({"t0": t0 + i * d, "t1": t0 + i * d + d - 1000, "kind": "kernel", "stream": stream, "name": f"k{stream}"})
        return t0 + dur
    for _ in range(steps):
        t0 = t
        t = burst(t, 10 * ms, 7)
        if branch:
            burst(t0 + 2 * ms, 4 * ms, 8, n=8)
        t += gap_ns
        t = burst(t, 2 * ms, 9)
        for _ in range(3):
            t += 400_000
            t = burst(t, int(1.6 * ms), 9)
        t += 300_000
    return acts


def test_stream_roles_split_an_abutting_draft_reeval_that_duration_misses():
    acts = _two_stream_acts(gap_ns=5_000)      # re-eval 5 us after verify: below GAP_US, so one burst by duration alone
    flat = [dict(a, stream=1) for a in acts]
    by_dur = r1_step.analyze(flat, 4)
    # by duration the abutting re-eval merges into the verify burst: verify reads 12 ms and one draft step is taken for the re-eval
    assert by_dur["split_mode"] == "duration" and by_dur["ok"] and by_dur["median_ms"]["target_verify"] > 11.5
    assert by_dur["median_ms"]["draft_reeval"] == pytest.approx(1.6, abs=0.1)
    res = r1_step.analyze(acts, 4)
    assert res["ok"] and res["split_mode"] == "stream" and res["roles"] == {"7": "target", "9": "draft"}
    m = res["median_ms"]
    assert m["target_verify"] == pytest.approx(9.98, abs=0.05) and m["draft_reeval"] == pytest.approx(2.0, abs=0.05)
    assert m["draft_gen"] == pytest.approx(4.8, abs=0.1) and res["shorts_per_step_median"] == 4
    assert res["checks"]["split_mode"] == "stream" and not res["checks"]["merged_draft_bursts_suspected"]


def test_stream_roles_keep_target_branch_streams_with_the_target():
    res = r1_step.analyze(_two_stream_acts(branch=True, gap_ns=500_000), 4)
    assert res["roles"] == {"7": "target", "8": "target", "9": "draft"} and res["ok"]
    assert res["median_ms"]["draft_reeval"] == pytest.approx(2.0, abs=0.05)
    assert res["median_ms"]["other_kernel"] == pytest.approx(0.0, abs=0.01)
    assert r1_step.assign_roles([dict(a, stream=1) for a in _two_stream_acts()]) == {}      # one stream: no roles


def test_wall_check_flags_profiler_overhead_and_parse_mismatch():
    an = {"median_ms": {"wall": 24.0}}
    req = {"predicted_ms": 1000.0, "tokens_predicted": 56, "draft_n_accepted": 6}      # 49 steps -> 20.4 ms
    wc = r1.step_wall_check(an, req, 20.0)
    assert wc["request_ms_per_step"] == pytest.approx(1000.0 / 49) and wc["split_matches_request"] is False
    assert wc["profiler_overhead_flag"] is True and wc["profiler_overhead_ms"] == pytest.approx(4.0)
    ok = r1.step_wall_check({"median_ms": {"wall": 20.5}}, req, 20.0)
    assert ok["profiler_overhead_flag"] is False and ok["split_matches_request"] is True
    assert r1.step_wall_check(None, None, None)["profiler_overhead_flag"] is None


def _write_sqlite(path, acts):
    con = sqlite3.connect(path)
    con.execute("create table StringIds (id integer primary key, value text)")
    con.execute("create table CUPTI_ACTIVITY_KIND_KERNEL (start integer, end integer, streamId integer, demangledName integer)")
    con.execute("create table CUPTI_ACTIVITY_KIND_MEMCPY (start integer, end integer, streamId integer, copyKind integer, bytes integer)")
    con.execute("insert into StringIds values (1, 'mul_mat_vec_q')")
    for a in acts:
        if a["kind"] == "kernel":
            con.execute("insert into CUPTI_ACTIVITY_KIND_KERNEL values (?, ?, ?, 1)", (a["t0"], a["t1"], a["stream"]))
        else:
            con.execute("insert into CUPTI_ACTIVITY_KIND_MEMCPY values (?, ?, ?, ?, ?)",
                        (a["t0"], a["t1"], a["stream"], {"DtoH": 2, "DtoD": 8, "HtoD": 1}[a["copy_kind"]], a["bytes"]))
    con.commit()
    con.close()


def test_read_sqlite_roundtrip(tmp_path):
    acts = _synthetic_acts(4)
    db = tmp_path / "x.sqlite"
    _write_sqlite(str(db), acts)
    got = r1_step.read_sqlite(db)
    assert len(got) == len(acts) and got[0]["name"] == "mul_mat_vec_q"
    assert {a["copy_kind"] for a in got if a["kind"] == "memcpy"} == {"DtoH", "DtoD"}
    assert r1_step.analyze(got, 4)["ok"]
    empty = tmp_path / "e.sqlite"
    sqlite3.connect(empty).close()
    with pytest.raises(ValueError):
        r1_step.read_sqlite(empty)


def test_classify_copy_and_cmds():
    assert r1_step.classify_copy("DtoD", 1 << 20) == "snapshot_copy" and r1_step.classify_copy("DtoD", 100) == "small_copy"
    assert r1_step.classify_copy("HtoD", 1 << 20) == "restore_copy" and r1_step.classify_copy("DtoH", 4) == "readback"
    c = r1_step.profile_cmd("nsys", "/t/x", ["srv", "--port", "1"])
    assert c[:2] == ["nsys", "profile"] and "--start-later=true" in c and "--cuda-graph-trace=node" in c and c[-3:] == ["srv", "--port", "1"]
    assert r1_step.control_cmd("nsys", "stop") == ["nsys", "stop", "--session=cr_r1"]


# ---------------------------------------------------------------- secrets and ignore rules

def test_scrub_and_secret_values(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "hf_abcdefgh12345")
    monkeypatch.setenv("SOME_KEY", "short")
    monkeypatch.setenv("PLAIN", "hf_abcdefgh12345x")
    sec = r1.secret_values()
    assert "hf_abcdefgh12345" in sec and "short" not in sec and "hf_abcdefgh12345x" not in sec
    out = r1.scrub({"a": ["x hf_abcdefgh12345 y", 3], "b": {"c": "ok"}}, sec)
    assert out == {"a": ["x <redacted> y", 3], "b": {"c": "ok"}}
    for name in ("DEPLOY_PAT", "GH_PAT_X", "MY_PASSWD", "SVC_CREDS", "SESSION_COOKIE", "AUTH_HEADER", "RUNPOD_POD_ID"):
        assert r1.SECRET_ENV_RE.search(name), name
    for name in ("PATH", "CUDA_VISIBLE_DEVICES", "PLAIN", "LD_LIBRARY_PATH"):
        assert not r1.SECRET_ENV_RE.search(name), name


def test_nsys_env_is_an_allowlist_with_a_private_tmpdir(monkeypatch):
    for k, v in {"HF_TOKEN": "x", "LLAMA_API_KEY": "y", "DEPLOY_PAT": "z", "CR_RESULTS_TOKEN": "t", "PLAIN": "p", "GITHUB_COOKIE": "c"}.items():
        monkeypatch.setenv(k, v)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    monkeypatch.setenv("GGML_CUDA_PDL", "1")
    env = r1_step.nsys_env("/private/dir")
    assert env["TMPDIR"] == "/private/dir" and env["CUDA_VISIBLE_DEVICES"] == "0" and env["GGML_CUDA_PDL"] == "1" and "PATH" in env
    for k in ("HF_TOKEN", "LLAMA_API_KEY", "DEPLOY_PAT", "CR_RESULTS_TOKEN", "PLAIN", "GITHUB_COOKIE"):
        assert k not in env, k
    assert r1_step.nsys_env("/d", {"HOME": "/h", "FOO": "1", "LC_ALL": "C"}) == {"HOME": "/h", "LC_ALL": "C", "TMPDIR": "/d"}


def test_nsys_files_are_ignored_and_never_served():
    ig = (EXP / ".gitignore").read_text()
    for pat in ("*.nsys-rep", "*.sqlite", "*.sqlite-wal", "*.sqlite-shm", "*.qdrep", "*.qdstrm"):
        assert pat in ig
    from infra import pod_runner
    for name in ("x.nsys-rep", "x.sqlite", "x.sqlite-wal", "x.sqlite-shm", "x.qdrep", "x.qdstrm"):
        assert pod_runner.is_blocked_file(pathlib.Path(name), [])
    wd = r1_step.private_workdir()
    try:
        assert not pathlib.Path(wd).resolve().is_relative_to(EXP) and oct(os.stat(wd).st_mode & 0o777) == "0o700"
    finally:
        r1_step.cleanup(wd)
    assert not os.path.exists(wd)


# ---------------------------------------------------------------- verdict

def _cell(sp, lo, hi, n):
    return {"speedup": sp, "speedup_ci95": [lo, hi], "n": n}


def _ng(code_fix, refactor):
    return {"by_category": {"code_fix": code_fix, "refactor": refactor}}


def test_ngram_decision_uses_the_ci_lower_bound_and_needs_n():
    ok = _cell(1.0, 0.9, 1.1, 16)
    go = r1.ngram_decision({"configs": ["ngram_mod"], "summary": {"ngram_mod": _ng(ok, ok)}})
    assert go["status"] == "go" and go["go"] is True and go["ngram_ge_1p15"] is False
    # a large raw maximum from 8 noisy cells does not decide: the interval includes 1.0
    lucky = _cell(1.3, 0.9, 1.7, 16)
    st = r1.ngram_decision({"configs": ["ngram_mod"], "summary": {"ngram_mod": _ng(lucky, ok)}})
    assert st["status"] == "straddles" and st["go"] is None and st["best_speedup"] == 1.3 and st["best_ci_lower"] == pytest.approx(0.9)
    sure = _cell(1.4, 1.2, 1.6, 16)
    ng = r1.ngram_decision({"configs": ["ngram_mod"], "summary": {"ngram_mod": _ng(sure, ok)}})
    assert ng["status"] == "no_go" and ng["go"] is False and ng["ngram_ge_1p15"] is True
    few = _cell(1.0, 0.8, 1.2, 7)
    weak = r1.ngram_decision({"configs": ["ngram_mod"], "summary": {"ngram_mod": _ng(few, ok)}})
    assert weak["status"] == "insufficient_n" and weak["go"] is None and weak["weak_cells"] == ["ngram_mod/code_fix"]
    miss = r1.ngram_decision({"configs": ["ngram_mod", "ngram_map_k"], "summary": {"ngram_mod": _ng(ok, ok)}})
    assert miss["status"] == "insufficient_n" and miss["configs_missing"] == ["ngram_map_k"]
    assert r1.ngram_decision({})["status"] == "not_measured"
    # a decided no-go stands even when other cells are weak
    both = r1.ngram_decision({"configs": ["ngram_mod"], "summary": {"ngram_mod": _ng(sure, few)}})
    assert both["status"] == "no_go"


def test_verdict_inputs():
    ok = _cell(1.0, 0.9, 1.1, 16)
    doc = {"step": {"analysis": {"ok": True, "split_mode": "stream", "median_ms": {"wall": 21.0, "non_compute": 4.0, "c1_removable": 3.0, "c1_floor": 2.5}},
                    "wall_check": {"profiler_overhead_ms": 1.0, "profiler_overhead_flag": True}},
           "agree": {"text": {"kind": "e5_chat"},
                     "summary": {"bf16_agreement": 0.6, "ptq1_0_agreement": 0.2, "bf16_ge_0p5": True, "ptq1_delta_vs_bf16": -0.4,
                                 "ptq1_within_0p05": False},
                     "secondary": {"repo": {"summary": {"bf16_agreement": 0.8, "ptq1_0_agreement": 0.3}}}},
           "e5": {"configs": ["ngram_mod"],
                  "pairs": {"k4_tbs_vs_k4": {"speedup": 1.25, "speedup_ci95": [1.1, 1.4], "n": 100},
                            "k4_ptq1_vs_k4": {"alpha": 0.40, "base_alpha": 0.47, "n": 60}},
                  "summary": {"k4": {"all": {"alpha": 0.47, "r": 0.21}}, "k4_tbs": {"checks": {"backend_sampling_reported": True}},
                              "ngram_mod": _ng(_cell(1.3, 1.2, 1.4, 16), ok)}}}
    v = r1.verdict(doc)
    c1 = v["c1_overhead"]
    assert c1["floor_le_2ms"] is False and c1["floor_ms_minus_profiler_overhead"] == pytest.approx(1.5) and c1["floor_adj_le_2ms"] is True
    assert c1["k4_tbs_gain_vs_k4"] == 1.25 and c1["target_backend_sampling_reported"] is True and c1["split_mode"] == "stream"
    c2 = v["c2_agreement"]
    assert c2["text_kind"] == "e5_chat" and c2["calibration"]["e5_k4_alpha"] == 0.47
    assert c2["calibration"]["teacher_forced_bf16_minus_alpha"] == pytest.approx(0.13)
    assert c2["secondary_text"]["repo"]["bf16"] == 0.8
    assert c2["e5_ptq1_alpha_drop"] == pytest.approx(0.07) and c2["e5_ptq1_within_0p05"] is False
    assert c2["ptq1_basis"] == "e5_k4" and c2["ptq1_within_0p05"] is False
    assert v["c3_ngram"]["status"] == "no_go" and v["c3_ngram"]["go"] is False
    empty = r1.verdict({})
    assert empty["c1_overhead"]["measured"] is False and empty["c3_ngram"]["go"] is None and empty["c2_agreement"]["bf16"] is None


# ---------------------------------------------------------------- end to end with fake binaries

FAKE_SERVER = """#!{py}
import json, os, sys
from http.server import BaseHTTPRequestHandler, HTTPServer
a = sys.argv
port = int(a[a.index("--port") + 1])
spec = a[a.index("--spec-type") + 1] if "--spec-type" in a else "none"
tbs = "-bs" in a
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
        text = "out:" + body["prompt"]
        if spec == "ngram-mod" and os.environ.get("FAKE_DIVERGE"):
            text = text[:8] + "!"
        speed = {{"none": 100.0, "draft-simple": 150.0 if tbs else 120.0}}.get(spec, 130.0)
        t = {{"predicted_per_second": speed, "predicted_ms": 1e3 * 100 / speed, "predicted_n": 100}}
        if spec != "none":
            t.update(draft_n=200 if spec == "draft-simple" else 40, draft_n_accepted=40 if spec == "draft-simple" else 20)
        self._send({{"content": text, "tokens_predicted": 100, "timings": t, "stop_type": "limit",
                    "generation_settings": {{"backend_sampling": tbs}}}})
print("srv: speculative decoding enabled: " + spec, flush=True)
HTTPServer(("127.0.0.1", port), H).serve_forever()
"""

FAKE_PERPLEXITY = """#!{py}
import sys
a = sys.argv
m = a[a.index("-m") + 1]
if "--kl-divergence" not in a:
    open(a[a.index("--kl-divergence-base") + 1], "wb").write(b"_logits_" + b"\\0" * 64)
    print("print_info: n_vocab          = 248320")
    sys.exit(0)
vocab = 151936 if "badvocab" in m else 248320
top = 20.0 if "ptq1" in m else (70.5 if "q8" in m else 71.0)
print("print_info: n_vocab          = %d" % vocab)
if "badvocab" in m:
    print("kl_divergence: inconsistent vocabulary (248320 vs 151936)")
print("====== KL divergence statistics ======")
print("Mean    KLD:   %.6f \\u00b1  0.003000" % (0.4 if top > 50 else 3.5))
print("Median  KLD:   0.05")
print("====== Token probability statistics ======")
print("Same top p: %.3f \\u00b1 0.700 %%" % top)
"""

FAKE_QUANTIZE = """#!{py}
import os, shutil, sys
if os.environ.get("FAKE_QUANT_FAIL"):
    print("llama_model_quantize: failed to quantize: imatrix required", file=sys.stderr)
    sys.exit(1)
shutil.copy(sys.argv[1], sys.argv[2])
print("converting to ptq1_0 .. size")
"""

FAKE_NSYS = """#!{py}
import json, os, signal, sqlite3, subprocess, sys
a = sys.argv
here = os.path.dirname(os.path.abspath(__file__))        # the capture env is an allowlist: files, not variables
log = os.path.join(here, "nsys.log")
with open(log, "a") as f:
    f.write(json.dumps({{"verb": a[1] if len(a) > 1 else "", "env_keys": sorted(os.environ)}}) + "\\n")
if a[1:2] == ["--version"]:
    print("NVIDIA Nsight Systems version 0.0.0-fake")
elif a[1:2] == ["profile"]:
    prefix = a[a.index("-o") + 1]
    cmd = a[a.index("-o") + 2:]
    child = subprocess.Popen(cmd)
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
    path.write_text(text.format(py=sys.executable).replace("\\u00b1", "+-"))
    path.chmod(path.stat().st_mode | stat.S_IXUSR)


@pytest.fixture
def pod(tmp_path, monkeypatch):
    b = tmp_path / "bin"
    b.mkdir()
    _script(b / "llama-server", FAKE_SERVER)
    _script(b / "llama-perplexity", FAKE_PERPLEXITY)
    _script(b / "llama-quantize", FAKE_QUANTIZE)
    _script(tmp_path / "nsys", FAKE_NSYS)
    acts = [[a["t0"], a["t1"], a["stream"]] for a in _synthetic_acts(8) if a["kind"] == "kernel"]
    (tmp_path / "acts.json").write_text(json.dumps(acts))
    for n in ("target.gguf", "draft-bf16.gguf", "draft-q8_0.gguf"):
        (tmp_path / n).write_bytes(b"gguf")
    monkeypatch.setenv("CR_RESULTS_DIR", str(tmp_path / "results"))
    monkeypatch.setenv("CR_GPU_TAG", "pro6000-ws")
    monkeypatch.setenv("CR_LLAMA_BIN", str(b))
    monkeypatch.setenv("CR_NSYS", str(tmp_path / "nsys"))
    monkeypatch.setenv("HF_TOKEN", "hf_secretsecret123")
    monkeypatch.setenv("RUNPOD_API_KEY", "rpa_secretsecret456")
    files = {"pq2": "target.gguf", "draft_bf16": "draft-bf16.gguf", "draft_q8_0": "draft-q8_0.gguf"}
    monkeypatch.setattr(r1.e4_pdl, "model_path", lambda name, notes: (str(tmp_path / files[name]), {"file": files[name]}))
    monkeypatch.setattr(r1.e4_pdl, "smi_sample", lambda: None)
    return tmp_path


def _result(tmp):
    files = sorted((tmp / "results" / "r1" / "pro6000-ws").glob("2*.json"))
    assert len(files) == 1
    return json.loads(files[0].read_text())


def test_agree_with_fake_binaries(pod):
    notes: list = []
    with __import__("tempfile").TemporaryDirectory() as wd:
        res = r1_agree.run_agree(bin_dir=pod / "bin", target=str(pod / "target.gguf"),
                                 drafts={"bf16": str(pod / "draft-bf16.gguf"), "q8_0": str(pod / "draft-q8_0.gguf")},
                                 workdir=wd, budget=common.Budget(2), notes=notes)
        assert not [f for f in os.listdir(wd) if f.endswith(".bin") or f.startswith("draft-")]   # log-probs and quantized GGUF removed
    assert res["errors"] == [] and res["base"]["n_vocab"] == 248320
    assert res["variants"]["bf16"]["same_top"] == pytest.approx(0.71) and res["variants"]["q8_0"]["same_top"] == pytest.approx(0.705)
    assert res["variants"]["ptq1_0"]["same_top"] == pytest.approx(0.20) and res["variants"]["ptq1_0"]["quantize"]["exit"] == 0
    assert res["summary"]["ptq1_within_0p05"] is False and res["summary"]["bf16_ge_0p5"] is True
    assert res["text"]["sha256_16"] and "files" in res["text"]


def test_agree_records_unsupported_quantize_and_vocab_mismatch(pod, monkeypatch):
    monkeypatch.setenv("FAKE_QUANT_FAIL", "1")
    notes: list = []
    with __import__("tempfile").TemporaryDirectory() as wd:
        res = r1_agree.run_agree(bin_dir=pod / "bin", target=str(pod / "target.gguf"), drafts={"bf16": str(pod / "draft-bf16.gguf")},
                                 workdir=wd, budget=common.Budget(2), notes=notes, variants=["bf16", "ptq1_0"])
    q = res["variants"]["ptq1_0"]
    assert "same_top" not in q and "failed" in q["error"] and "imatrix" in q["quantize"]["tail"]
    assert any("ptq1_0" in n for n in notes) and res["variants"]["bf16"]["same_top"] == pytest.approx(0.71)
    (pod / "draft-badvocab.gguf").write_bytes(b"gguf")
    with __import__("tempfile").TemporaryDirectory() as wd:
        res = r1_agree.run_agree(bin_dir=pod / "bin", target=str(pod / "target.gguf"), drafts={"bf16": str(pod / "draft-badvocab.gguf")},
                                 workdir=wd, budget=common.Budget(2), notes=[], variants=["bf16"])
    v = res["variants"]["bf16"]
    assert "vocabulary mismatch" in v["error"] and v["same_top"] is None


def test_capture_with_fake_nsys_cleans_up_and_scrubs_env(pod):
    from infra.results import artifact_path
    cfg = next(c for c in e5_spec.matrix_configs() if c["name"] == "k4")
    wd = r1_step.private_workdir()
    stages: list = []
    res = r1_step.capture(nsys=str(pod / "nsys"), server_argv=e5_spec.config_cmd(pod / "bin", "t", "d", cfg), port=e5_spec.PORT,
                          prompt="fix the bug", budget=common.Budget(3), workdir=wd, log_path=str(artifact_path("r1", "s.log", "pro6000-ws")),
                          on_stage=lambda r: stages.append(dict(r)))
    assert res["ok"], res["error"]
    assert len(stages) == 1 and stages[0]["request"]["tokens_predicted"] == 100 and stages[0]["analysis"] is None   # saved before export / parse
    assert res["analysis"]["n_steps"] >= 3 and res["request"]["tokens_predicted"] == 100
    assert not os.path.exists(wd)                                                  # nsys-rep, sqlite and the dir are gone
    verbs = [json.loads(ln) for ln in (pod / "nsys.log").read_text().splitlines()]
    assert {"profile", "start", "stop"} <= {v["verb"] for v in verbs}
    for v in verbs:
        assert "HF_TOKEN" not in v["env_keys"] and "RUNPOD_API_KEY" not in v["env_keys"] and "PYTEST_CURRENT_TEST" not in v["env_keys"]
        assert "TMPDIR" in v["env_keys"] and "PATH" in v["env_keys"]            # allowlist, not a denylist
    assert not list(pod.rglob("*.nsys-rep")) and not list(pod.rglob("*.sqlite"))


def test_capture_without_nsys_report_fails_cleanly(pod, monkeypatch):
    from infra.results import artifact_path
    (pod / "acts.json").write_text("[]")
    cfg = next(c for c in e5_spec.matrix_configs() if c["name"] == "k4")
    wd = r1_step.private_workdir()
    res = r1_step.capture(nsys=str(pod / "nsys"), server_argv=e5_spec.config_cmd(pod / "bin", "t", "d", cfg), port=e5_spec.PORT,
                          prompt="p", budget=common.Budget(3), workdir=wd, log_path=str(artifact_path("r1", "s.log", "pro6000-ws")))
    assert not res["ok"] and "no GPU activity" in res["error"] and not os.path.exists(wd)


def test_r1_end_to_end_with_fakes(pod, monkeypatch):
    monkeypatch.setattr(r1_agree, "TEXT_CHARS", 100)          # let the 8 tiny fake outputs make the chat text
    args = ["--quick", "--parts", "a,b,c", "--configs", "base,k4,k4_tbs,k4_ptq1,ngram_mod,ngram_map_k", "--limit", "8", "--budget-min", "10"]
    assert r1.main(args) == 0
    doc = _result(pod)
    d = doc["data"]
    ag = d["agree"]
    assert ag["text"]["kind"] == "e5_chat" and ag["text"]["n_samples"] >= 1 and set(ag["secondary"]) == {"repo"}
    assert ag["summary"]["bf16_agreement"] == pytest.approx(0.71) and ag["variants"]["ptq1_0"]["same_top"] == pytest.approx(0.20)
    assert ag["variants"]["ptq1_0"]["quantize"]["exit"] == 0 and ag["secondary"]["repo"]["summary"]["bf16_agreement"] == pytest.approx(0.71)
    e5 = d["e5"]
    assert list(e5["runs"]) == ["base", "k4", "k4_tbs", "k4_ptq1", "ngram_mod", "ngram_map_k"]
    assert len(e5["order"]) == 8 and e5["n_main"] == 8 and e5["runs"]["base"]["rows"][0]["text"].startswith("out:")
    assert e5["runs"]["base"]["rows"][0]["prompt_text"].startswith("T:") and "prompt_text" not in e5["runs"]["k4"]["rows"][0]
    s = e5["summary"]
    assert s["base"]["n_ok"] == 8 and s["k4"]["n_paired"] == 8
    assert s["k4"]["all"]["speedup"] == pytest.approx(1.2) and s["k4_tbs"]["all"]["speedup"] == pytest.approx(1.5)
    assert s["k4"]["all"]["r"] == pytest.approx(0.2) and 0.3 < s["k4"]["all"]["alpha"] < 0.5 and s["k4"]["all"]["speedup_ci95"]
    assert s["ngram_mod"]["all"]["alpha"] is None and s["ngram_mod"]["n_paired"] == 2          # code categories only
    assert s["k4"]["all"]["text_equal_frac"] == 1.0
    cmds = {n: e5["runs"][n]["cmd"] for n in e5["runs"]}
    assert "-bs" in cmds["k4_tbs"] and "-bs" not in cmds["k4"] and not any("draft-backend-sampling" in a for c in cmds.values() for a in c)
    assert cmds["k4_ptq1"][cmds["k4_ptq1"].index("-md") + 1].endswith("draft-ptq1_0.gguf")
    assert any("enabled: draft-simple" in ln for ln in e5["runs"]["k4"]["spec_log"])
    assert s["k4_tbs"]["checks"]["backend_sampling_reported"] is True and s["k4"]["checks"]["backend_sampling_reported"] is False
    assert e5["runs"]["ngram_mod"]["state_note"] and e5["state_note"]
    assert not any("-bs was asked" in n for n in d["notes"])
    assert set(s["k4"]["by_category"]) == {"code_fix", "code_explain", "code_write", "json_extract", "plan", "refactor", "shell", "tool_call"}
    assert e5["pairs"]["k4_tbs_vs_k4"]["speedup"] == pytest.approx(1.25) and e5["pairs"]["k4_ptq1_vs_k4"]["n"] == 4
    st = d["step"]
    assert st["ok"] and st["config"] == "k4" and st["leftover_files"] == [] and st["analysis"]["n_steps"] >= 3
    assert st["analysis"]["split_mode"] == "stream"
    assert st["wall_check"]["partb_ms_per_step"] == pytest.approx(s["k4"]["all"]["ms_per_step"]) and st["wall_check"]["nsys_step_wall_ms"]
    v = d["verdict"]
    assert v["c1_overhead"]["measured"] is True and v["c1_overhead"]["k4_tbs_gain_vs_k4"] == pytest.approx(1.25)
    assert v["c1_overhead"]["target_backend_sampling_reported"] is True
    assert v["c2_agreement"]["ptq1_within_0p05"] is False and v["c2_agreement"]["calibration"]["e5_k4_alpha"] == s["k4"]["all"]["alpha"]
    assert v["c3_ngram"]["status"] in ("insufficient_n", "go", "straddles", "no_go")
    text = json.dumps(doc)
    assert "hf_secretsecret123" not in text and "rpa_secretsecret456" not in text
    assert not list(pod.rglob("*.nsys-rep")) and not list(pod.rglob("*.sqlite")) and not list((pod / "results").rglob("*.bin"))
    assert not list((pod / "results").rglob("*.gguf")) and not (pod / "results" / "r1" / "pro6000-ws" / "raw" / "partial.json").exists()


def test_r1_falls_back_to_the_repo_text_without_enough_e5_output(pod):
    assert r1.main(["--quick", "--parts", "a,b", "--configs", "base,k4", "--limit", "4", "--budget-min", "6"]) == 0
    d = _result(pod)["data"]
    assert d["agree"]["text"]["kind"] == "repo" and "secondary" not in d["agree"] and any("no E5 chat text" in n for n in d["notes"])


def test_r1_text_divergence_is_reported(pod, monkeypatch):
    monkeypatch.setenv("FAKE_DIVERGE", "1")
    assert r1.main(["--parts", "b", "--configs", "base,ngram_mod", "--limit", "16", "--budget-min", "6"]) == 0
    s = _result(pod)["data"]["e5"]["summary"]["ngram_mod"]["all"]
    assert s["text_equal"] == 0 and s["text_compared"] == 4 and s["median_prefix_frac"] < 1.0


def test_r1_skips_parts_whose_binary_is_missing(pod):
    (pod / "bin" / "llama-perplexity").unlink()
    assert r1.main(["--quick", "--parts", "a,b", "--configs", "base", "--limit", "4", "--budget-min", "4"]) == 0
    d = _result(pod)
    assert any("part a skipped" in n for n in d["data"]["notes"]) and d["data"]["agree"] is None and d["data"]["e5"]["summary"]["base"]["n_ok"] == 4


CAPPED = {"clocks.sm": 2000.0, "clocks.max.sm": 2400.0, "clocks.mem": 14000.0, "clocks.max.mem": 14000.0, "power.draw": 100.0,
          "power.limit": 350.0, "power.default_limit": 600.0, "temperature.gpu": 40.0}


def test_r1_power_capped_host_still_runs_part_a(pod, monkeypatch):
    monkeypatch.setattr(r1.e4_pdl, "smi_sample", lambda: dict(CAPPED))
    assert r1.main(["--quick", "--parts", "a,b,c", "--configs", "base", "--limit", "4", "--budget-min", "6"]) == 1
    d = _result(pod)["data"]
    assert d["agree"]["summary"]["bf16_agreement"] == pytest.approx(0.71) and d["e5"] is None and d["step"] is None
    assert d["verdict"]["host"]["parts_skipped"] == ["b", "c"] and "power cap" in d["verdict"]["host"]["problem"]
    assert any("unhealthy host" in n and "part a" in n for n in d["notes"])


def test_r1_allow_power_cap_runs_everything_and_records_the_cap(pod, monkeypatch):
    monkeypatch.setattr(r1.e4_pdl, "smi_sample", lambda: dict(CAPPED))
    assert r1.main(["--quick", "--parts", "b", "--configs", "base", "--limit", "4", "--budget-min", "4", "--allow-power-cap"]) == 0
    d = _result(pod)["data"]
    assert d["e5"]["summary"]["base"]["n_ok"] == 4 and "power cap" in d["verdict"]["host"]["power_cap_note"]
    assert d["verdict"]["host"]["parts_skipped"] == [] and any("power cap accepted" in n for n in d["notes"])


def test_r1_sigterm_still_writes_the_result_and_names_the_cut_part(pod, monkeypatch):
    real = r1.run_e5

    def cut(**kw):
        kw["notes"].append("config note with hf_secretsecret123 inside")
        real(**kw)
        raise SystemExit(143)                  # what common.term_exits turns SIGTERM into
    monkeypatch.setattr(r1, "run_e5", cut)
    with pytest.raises(SystemExit):
        r1.main(["--quick", "--parts", "b,a", "--configs", "base", "--limit", "4", "--budget-min", "4"])
    doc = _result(pod)
    d = doc["data"]
    assert any("run cut during part b" in n for n in d["notes"]) and d["e5"]["summary"]["base"]["n_ok"] == 4
    assert "hf_secretsecret123" not in json.dumps(doc) and any("<redacted>" in n for n in d["notes"])


def test_r1_dry_run_and_args(capsys):
    assert r1.main(["--dry-run"]) == 0
    plan = json.loads(capsys.readouterr().out)
    assert plan["exp"] == "r1" and len(plan["e5"]["configs"]) == len(e5_spec.matrix_configs()) and plan["e5"]["n_prompts"] == 200
    assert plan["parts"] == ["b", "a", "c"] and plan["e5"]["plan"]["fits"] and plan["e5"]["plan"]["min_code_category"] >= 15
    assert plan["agree"]["quantize_cmd"][-1] == "PTQ1_0" and plan["step"]["nsys_profile"][:2] == ["nsys", "profile"] and plan["step"]["config"] == "k4"
    assert "-bs" in next(c["cmd"] for c in plan["e5"]["configs"] if c["name"] == "k4_tbs")
    with pytest.raises(SystemExit):
        r1.parse_args(["--configs", "k4"])              # no base
    with pytest.raises(SystemExit):
        r1.parse_args(["--parts", "d"])
    with pytest.raises(SystemExit):
        r1.parse_args(["--configs", "k4_bs"])           # the draft-sampling variants are gone
    with pytest.raises(SystemExit):
        r1.parse_args(["--step-config", "ngram_mod"])
    assert r1.parse_args([]).budget_min == r1.DEFAULT_TOTAL_MIN and r1.parse_args(["--step-config", "k4_tbs"]).step_config == "k4_tbs"


def test_manifest_r1_fits_the_failsafe():
    m = json.loads((EXP / "exps" / "manifest.json").read_text())["r1"]
    # est_min covers downloads and the r1 budget; clone + cold build + bootstrap (~14 min on the last pod) come on top
    assert m["est_min"] >= r1.DEFAULT_TOTAL_MIN + 5 and m["est_min"] + 15 <= m["failsafe_min"] and "--timeout-min" in m["cmd"]


# ---------------------------------------------------------------- fetch_models timeout, vocab pre-check

def test_fetch_all_times_out_a_stalled_download():
    import threading
    gate = threading.Event()

    def fetch(m, notes):
        if m == "slow":
            gate.wait(30)
        if m == "bad":
            raise OSError("no route")
        return f"/models/{m}.gguf"
    try:
        res = fetch_models.fetch_all(["fast", "slow", "bad"], 0.5, fetch)
    finally:
        gate.set()
    assert res["fast"][0] == "/models/fast.gguf" and res["slow"][0] is None and "timeout" in res["slow"][1][0]
    assert res["bad"][0] is None and "no route" in res["bad"][1][0]


def test_gguf_n_vocab_reads_the_token_list(tmp_path):
    sys.path.append(str(EXP.parent.parent / "gguf-py"))
    import gguf
    f = tmp_path / "t.gguf"
    w = gguf.GGUFWriter(str(f), "qwen35")
    w.add_tokenizer_model("gpt2")
    w.add_token_list(["a", "b", "c", "d", "e"])
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_ti_data_to_file()
    w.close()
    assert r1_agree.gguf_n_vocab(f) == 5 and r1_agree.gguf_n_vocab(tmp_path / "missing.gguf") is None


def test_vocab_mismatch_is_caught_before_the_target_pass(pod):
    counts = {str(pod / "target.gguf"): 248320, str(pod / "draft-bf16.gguf"): 151936, str(pod / "draft-q8_0.gguf"): 151936}
    notes: list = []
    with __import__("tempfile").TemporaryDirectory() as wd:
        res = r1_agree.run_agree(bin_dir=pod / "bin", target=str(pod / "target.gguf"),
                                 drafts={"bf16": str(pod / "draft-bf16.gguf"), "q8_0": str(pod / "draft-q8_0.gguf")}, workdir=wd,
                                 budget=common.Budget(2), notes=notes, vocab_fn=lambda p: counts.get(str(p)))
    assert res["base"] is None and "target pass skipped" in res["errors"][0]            # the 27B pass never ran
    assert all("vocabulary mismatch" in v["error"] for v in res["variants"].values()) and len(notes) == 3
    counts[str(pod / "draft-q8_0.gguf")] = 248320
    counts[str(pod / "draft-bf16.gguf")] = 248320
    with __import__("tempfile").TemporaryDirectory() as wd:
        ok = r1_agree.run_agree(bin_dir=pod / "bin", target=str(pod / "target.gguf"),
                                drafts={"bf16": str(pod / "draft-bf16.gguf"), "q8_0": str(pod / "draft-q8_0.gguf")}, workdir=wd,
                                budget=common.Budget(2), notes=[], vocab_fn=lambda p: counts.get(str(p)))
    assert ok["base"]["exit"] == 0 and ok["vocab_gguf"]["target"] == 248320 and ok["variants"]["bf16"]["same_top"] == pytest.approx(0.71)


def test_build_chat_text_uses_template_and_greedy_outputs():
    rows = [{"id": f"p{i}", "prompt_text": f"<|im_start|>user\nq{i}<|im_end|>\n<|im_start|>assistant\n", "text": "answer " * 20} for i in range(10)]
    rows.append({"id": "bad", "error": "x"})
    text, meta = r1_agree.build_chat_text(rows, chars=600)
    assert text.startswith("<|im_start|>user\nq0") and "answer answer" in text and text.count("<|im_end|>\n<|im_start|>assistant") >= 3
    assert meta["kind"] == "e5_chat" and meta["n_samples"] == meta["n_samples"] <= 10 and 0.5 < meta["out_char_frac"] < 1.0
    assert r1_agree.build_chat_text(rows, chars=100000)[0] is None             # too little text for the 8192-token pass
    assert r1_agree.build_chat_text([], 10)[0] is None
