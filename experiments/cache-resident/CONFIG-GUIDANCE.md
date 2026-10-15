# Speculative decoding configuration guidance

Scope: Bonsai 2 27B PQ2_0 target on llama-server in this fork, one slot, greedy, `-fa on -ngl 99`. Numbers come from R1, R2a and R2b (REPORT.html, `results/r1`, `results/r2a`, `results/r2b_spec`). Speedup = tok/s over target alone, paired per prompt on the same pod. Intervals are 95% bootstrap over prompts and do not include host-to-host variation. No code change is needed for any item here.

## Summary

| Workload | Setting | Speedup |
|---|---|---|
| Full-file edits (answer copies the prompt) | ngram-mod, or ngram-mod,draft-simple | 2.3-2.6x |
| Long-context agentic mix (2-8K prompts) | ngram-mod, or ngram-mod,draft-simple | 1.3-1.4x |
| Diff edits | no speculation | all modes 0.85-0.96x |
| Short chat, planning, code-fix prompts | no speculation | all modes below 1.0x |
| draft-simple, any draft | use the Q8_0 draft, not BF16 | +9% vs BF16 |

## Data and caveats

Hosts: R1 on a RTX PRO 6000 Workstation (600 W) and Server; R2a on a PRO 6000 Workstation (600 W); R2b spec on a PRO 6000 Max-Q (300 W, target alone 0.90x of the R2a host, so compare only within a pod).

R1 (short agentic and code prompts, 154 prompts, n per category 19-40):
- draft-simple K=4 BF16 draft: 0.61x [0.59, 0.63] overall, alpha 0.43. Wins on json_extract (1.33x) and tool_call (1.16x), loses on plan (0.49x).
- ngram-mod 0.99x [0.98, 0.99] and ngram-map-k 0.99x on code_fix/refactor (n = 40): no gain, no loss.

R2a (53 long-context prompts, ctx 16384, drift check 0.998):

| Mode | Full-file edit (n=18) | Diff edit (n=18) | Tool trace (n=17) | All (n=53) |
|---|---|---|---|---|
| draft-simple K=4 BF16 | 1.26x [1.20, 1.32] | 0.88x [0.84, 0.92] | 1.12x [1.03, 1.24] | 1.05x [0.98, 1.12] |
| ngram-mod (fork defaults) | 2.63x [2.28, 3.12] | 0.96x [0.94, 0.97] | 1.09x [1.01, 1.18] | 1.41x [1.22, 1.66] |
| ngram-mod short (n-match 4, n-min 2, n-max 16) | 1.01x | 0.82x | 0.70x | 0.89x |

R2b spec (PRO 6000 Max-Q, 24 prompts, 8 per category; exploratory, rely on the intervals):
- Q8_0 vs BF16 draft: same alpha (0.757), step 23.5 vs 25.8 ms. Paired Q8_0/BF16 = 1.09x [1.07, 1.13] overall, 1.08-1.13x per category. Clock and run order explain part of it; the gain is at least about 4%.
- ngram-mod,draft-simple (BF16 draft): 1.31x [1.06, 1.74] overall, 2.54x [1.97, 3.47] full-file, 0.89x diff, 1.05x tool trace. ngram-mod alone on R2a prompts: 1.33x, 2.32x (different host, not paired). Combined with a Q8_0 draft is untested. Not tried on short prompts.

Other caveats:
- draft-simple numbers may be lower bounds. On CPU with a tiny hybrid model, the fork's draft context restores a stale recurrent state after every drafting step (see PLAN.md, Status); `wrap_b` measures the effect on the real draft (k4q8 vs k4q8_ckpt) before any guidance changes.
- ngram-mod alone checkpoints on every drafting step (n_rs_seq = 0 on the hybrid target), so its numbers are lower bounds. With draft-simple in the list, drafts up to K roll back in the batch.
- Speculation changes outputs slightly through batch numerics: 88-100% of outputs matched target alone token for token.
- Results are for this model, hardware and prompt mix. Re-measure for other workloads.

## Flags

Names checked against `common/arg.cpp`. Common base:

```
llama-server -m <bonsai2-27b-pq2_0.gguf> -ngl 99 -fa on -c 16384 -np 1
```

Draft-simple with the Q8_0 draft (Qwen3.5-0.8B-Q8_0.gguf from the same repo as the BF16 file, ggml-org/Qwen3.5-0.8B-GGUF):

```
  --spec-type draft-simple -md <Qwen3.5-0.8B-Q8_0.gguf> -ngld 99 --spec-draft-n-max 4
```

ngram-mod (defaults: n-match 24, n-min 48, n-max 64; no draft model):

```
  --spec-type ngram-mod
```

Combined (ngram-mod first, draft-simple when it has no draft):

```
  --spec-type ngram-mod,draft-simple -md <Qwen3.5-0.8B-Q8_0.gguf> -ngld 99 --spec-draft-n-max 4
```

Off (diff edits, short chat and planning): omit `--spec-type` and `-md`.

Aliases: `--spec-draft-model` = `-md`, `--spec-draft-ngl` = `-ngld`. Tuning flags, not swept here: `--spec-ngram-mod-n-match`, `--spec-ngram-mod-n-min`, `--spec-ngram-mod-n-max`. The old `--draft-max` and `--draft-min` are removed.
