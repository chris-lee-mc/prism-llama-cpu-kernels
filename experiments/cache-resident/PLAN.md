# Cache-Resident LM - RunPod Execution Plan

*Companion to proposal v3.1. Sized to run on rented RunPod pods, with the RTX PRO 6000 Blackwell as the primary target and cheaper consumer cards used wherever the result transfers. Prices are RunPod on-demand as of Oct 2026; re-query before launching.*

## Status (closed tracks)

- L2-residency PR track (section 10, original goal): closed as a negative result. E1 best gain 17% vs the 20% gate (3090 control 0-1%), E3 no gain on a real 0.8B model, E3d at most about 1% end to end.
- E6 (ternary draft training) dropped: the draft is latency bound, ternary saves about 0.1 ms per token, and Phase 2 never started.
- Section 10 now points to the candidates that survived: the PQ2_0 multi-column kernel on sm_120 (fork PR, conditional on a 600 W PRO 6000 paired rerun, a llama-server spec run and a 4090 check) and two upstream investigations (in-batch rollback for n-gram types, skipping the draft re-eval; measure first, issue before code). Config guidance: CONFIG-GUIDANCE.md. Evidence: REPORT.html.
- Prepared, not run (manifest entries): `wrap_a` (A1 paired kernel timing + A2 llama-server spec run, kernel off/on, on a 600 W PRO 6000: cascade key `pro6000_600w`, the driver refuses a Max-Q and an unhealthy host), `wrap_a3` (Ada check on an RTX 4090: base build with `exps/data/a3_pq2_multicol.patch` reversed vs head, SASS of the multi-column kernel compared first), `wrap_b` (test build for the two upstream investigations: `GGML_EXP_DRAFT_CKPT` reference draft state, `GGML_EXP_DRAFT_TIME` timing of the draft-context decode of the verify batch, `GGML_EXP_NGRAM_RS=N`, plus a rollback probe at n_rs_seq 4, 16 and 64). `GGML_EXP_SKIP_DRAFT_REEVAL` (the skip prototype) is allowed only for one slot, draft-simple alone and an attention-only draft; it is not in the `wrap_b` matrix. The switches are experiment-only, env-gated and default off; the fork PR and any upstream issue are written by the human contributor.
- Pre-flight changes after the failed attempt (ed133e886, 0811abc1c: a power-capped 600 W PRO 6000, a 4090 without CUDA, no community capacity). Power: PR-evidence entries (`wrap_a`, `wrap_a3`, and `wrap_ab` that contains `wrap_a`) carry `power_tol` 0.01 in the manifest; the bootstrap stops the pod before the cold build unless `power.limit` equals `power.default_limit` within 1% (`CR_POWER_TOL`), and `PowerGuard` re-reads both values at the start and before and after each timed part (`wrap_a` b and s, with a check before every server run of s; `wrap_a3` b and f), recording them in `data.power` (with `enforced.power.limit`, recorded only); a drop skips the part or leaves the affected rounds out, exit 1. During part b the lowest `power.limit` of all samples of a run is judged, not the last one. A pod without `nvidia-smi` stops at bootstrap when `CR_POWER_TOL` is set. `wrap_b` keeps the 75% gate. Kernel path: `wrap_a` has part k (one nsys kernel-name list, on then off): without `mul_mat_vec_pq2_0_mc` in the on run (or with it in the off run) parts b and s do not run. Part t runs the kernel off second (on, off, on, on) and flags a missing off run. Rollback probe: each batch-mode run now has a batch-shape control (the same tokens in the reference and test batch shapes with no `seq_rm`), so every B2 diff is exact, a batch-shape effect or a rollback. B1: `wrap_b` adds `k4q8_t` (`GGML_EXP_DRAFT_TIME=1` on the stale-state path) next to `k4q8` and `k4q8_ckpt`, so alpha of the stale path against the checkpoint-correct one and the `dur_p` ms per step bounds exist for the real hybrid Q8_0 draft. Budgets: `wrap_a` 14 min (est 20), `wrap_b` 24 min (est 30), `wrap_ab` est 44, failsafe 64.
- Cost of the evidence for the fork PR (item 2), at the RunPod prices of Oct 2026 (600 W PRO 6000 $1.69 community, $2.49-2.59 secure; 4090 $0.34 community, $0.89 secure). The first drivers asked for more than the $0.60-0.80 that was approved: `wrap_a` 26 min of run plus about 15 min of bootstrap and cold build was about $1.15-1.45, `wrap_a3` $0.23-0.51, and the budget gate reserved up to $2.60 per `wrap_a` pod. Now: `wrap_a` has `--budget-min 12` (4 bench rounds, 2 blocks of 4 prompts per category in the server run; the cost model gives about 4 min for the server part) and failsafe 40 min: about 32 min of pod time, about $0.9 at the community price and about $1.1 at the secure price; the cascade is capped at $1.75 (`--max-price` lifts it), so the gate reserves at most $1.31. `wrap_a3` about $0.3. Item 2 as a whole is about $1.2 to $1.4, still above $0.80, because the cold build alone is about 11 min ($0.3). Neither `wrap_a` nor `wrap_a3` is launched until the human approves that figure or picks a cheaper option. The power cap of `wrap_a` is checked in the driver after the bootstrap, cold build and downloads (about $0.5-0.6 before the first check), and both earlier R2b multicol runs hit a 400-420 W cap on the host, so a capped pod would spend that and exit 1 without evidence. Options, none applied: (a) check `nvidia-smi` `power.limit` against `power.default_limit` and the GPU name in `pod_entry.sh` / `bootstrap.sh` before the cold build and terminate the pod on a cap; (b) for `wrap_a`, skip the torch install and build only `llama-bench`, `llama-server` and `test-backend-ops` instead of the full target list; (c) skip or shrink `wrap_a3` (on Ada the change is a no-op by construction, the sass part checks it). Options to get under it: skip `wrap_a3` (on Ada the change is a no-op by construction: `ggml_cuda_pdl_sync` is compiled out below sm_90; the sass part of `wrap_a3` checks it), or accept the figure above. `wrap_b` is not PR evidence and has no 600 W gate: `pro6000_maxq` (about $0.5 per hour when in stock) or `rtx5090`, about 43 min, about $0.4-0.5; `wrap_ab` runs `wrap_a` and `wrap_b` in one 600 W session (shares the bootstrap, build and download; about 55 min, about $1.6) and is dearer than the two single sessions when a Max-Q or 5090 is in stock.
- Found while building B1 (CPU, tiny random models, `results/rs_rollback_cpu`): a GDN layer writes rollback snapshots only for the tokens of the last ubatch, so `seq_rm` after one-token decodes restores a stale recurrent state. draft-simple drafts one token at a time and then rolls the draft context back with `seq_rm`, so the hybrid draft's state is stale after every partial acceptance. The tiny near draft (partial acceptance, so it does roll back) shows it: acceptance 0.001 with the default path against 0.106 with the correct state (`GGML_EXP_DRAFT_CKPT=1`, the existing draft checkpoint path). The earlier "same" rows (target as its own draft: 0.011 default, 1.000 with the checkpoint) are not rollback evidence for the skip switch: with full acceptance no step rolls back, and the 1.000 of the old `k4_skip_same` only said that nothing was rolled back.
- B1 (skip the draft re-eval) as first built inherited that stale rollback: it kept the drafted tokens in the draft context and relied on `seq_rm(pos_next)` for partial acceptance, and the old gate enabled it exactly for the recurrent (RS) draft, the one case where it cannot be correct (near draft 0.001, the same as the default path). The `wrap_b` k4q8_skip speedup mixed the saved re-eval with a wrong draft state and is not read as the gain of `TAG_SPEC_AVOID_DRAFT_REEVAL`; the config is gone from the matrix. Now the gate allows the skip only for one slot, draft-simple alone and an attention-only draft (type PART, `seq_rm` exact). On CPU with a tiny llama-arch pair (`k4_l_*`) the skip gives the same drafted and accepted counts as the default path (near: 487 drafted, 160 accepted in both, 125 skipped batches) and the same text. The saving is measured directly in the correct state: `GGML_EXP_DRAFT_TIME=1` on `k4q8_ckpt_t` first waits for the target (`llama_decode` is async and the server calls `process()` right after the target verify, so an unsynchronized window would include target time), then times the draft-context decode of the verify batch and prints it as `dur_p` on the exp stats line. `dur_p` is the isolated cost. In the untimed path the re-eval overlaps the target verify, so the saving of skipping it is at most the isolated ms per step and at least the end-to-end ms per step delta of `k4q8_ckpt_t` over `k4q8_ckpt` (the synchronizes remove the overlap); `wrap_b` reports both (`reeval_bounds`). The CPU backend is synchronous and cannot show the overlap. Text equality with the target cannot show a wrong draft state (the target verifies every token), so `wrap_b` reads alpha: `k4q8` and `k4q8_ckpt_t` against `k4q8_ckpt`.
- The stale-state bug is fork-specific, checked against upstream ggml-org/llama.cpp master a0e6dc738 (2026-10-14): `common/speculative.cpp:2626` and `common/common.cpp:1260` set the draft context's `n_rs_seq = 0`, so upstream's hybrid draft is FULL and takes the draft checkpoint path; the fork's #173 (36761482b) keeps the target window for the draft context. Upstream has the `TAG_SPEC_AVOID_DRAFT_REEVAL` TODO at `tools/server/server-context.cpp:3428` and `:4244`. The real draft may be hit less than the random tiny model (R2b alpha was 0.757 with the fork default path); `wrap_b` k4q8 vs k4q8_ckpt decides.
- B2 (in-batch rollback for the n-gram types): the GPU probe runs n_rs_seq 4, 16 and 64 with n_rs_seq + 1 extra tokens, the shape of a verify batch (K + 1 tokens, rollback up to K = n_rs_seq). The boundary row (kept 1, rollback == n_rs_seq: a full rejection, snapshot plane n_rs_seq) is reported on its own (`boundary`), exact within 1e-3 of the largest logit; the whole-batch row (kept 0) is out of the window and not a server case. Rollbacks up to n-1 were all the first version of the probe covered, which missed the case the server hits. On CPU (tiny hybrid, `results/rs_rollback_cpu`, rerun with n_rs_seq + 1 extra) the boundary row is exact at 4, 16 and 64 (max logit diff 1.0e-5, 5.6e-6 and 4.5e-6, top-1 equal); the CUDA GDN snapshot path at that plane is still untested until `wrap_b` runs. The CPU `dur_p` (3.3 ms per call, tiny model) is unchanged by the target synchronize (the CPU backend is synchronous). The earlier server check never rolled back (172/172 and 184/184 drafted tokens accepted); the mid model (partial acceptance) now does under `GGML_EXP_NGRAM_RS=8`: 2 partial steps, 13 tokens rolled back in the batch (checkpoint path without the switch: 2 restores). Greedy text equality is informational (88-100% from batch numerics). The cut drafts of `ngram_rs16` also change the low-acceptance reset rule of ngram-mod (`accept()` uses the cut length), so `ngram_rs64` vs `ngram` is the clean rollback-path comparison and rs16 a separate draft-length setting; the reset counts are reported per config.

## 1. GPU matrix

| GPU | Arch / SM | L2 | VRAM | BW | RunPod $/hr (community / secure) | Role |
|---|---|---|---|---|---|---|
| RTX PRO 6000 Blackwell (Workstation or Max-Q) | GB202 / sm_120 | 128 MB | 96 GB | 1.79 TB/s | $1.69 / $2.49-2.59 (Max-Q sometimes ~$0.50) | Final numbers; two-tier (27B + draft) runs; anything needing > 32 GB |
| RTX 5090 | GB202 / sm_120 | 96 MB | 32 GB | 1.79 TB/s | $0.69 / $1.19 | Primary dev card. Same die, same kernels, same persisting-L2 API. Everything except 27B two-tier runs. |
| RTX 4090 | AD102 / sm_89 | 72 MB | 24 GB | 1.01 TB/s | $0.34 / $0.89 | Cheap kernel-dev and cross-arch check. |
| RTX 3090 | GA102 / sm_86 | 6 MB | 24 GB | 0.94 TB/s | ~$0.22 / ~$0.43 | Negative control. Persisting L2 ~4 MB; any "cache win" here is launch-overhead or a bug. |
| L40S / RTX 6000 Ada | AD102 / sm_89 | 96 MB | 48 GB | 0.86 TB/s | ~$0.99 secure | Optional two-tier fallback. |

Develop on 5090, validate on PRO 6000, sanity-check on 3090.

## 2. Budget

Phase 0 ~$15 (20 h 5090/4090); Phase 1 ~$35 (25 h 5090, 6 h PRO 6000, 4 h 3090); Phase 2 $45-130; Phase 3 ~$40; contingency 30%. Total ~$200-300. Patterns: budget gate, 40-min failsafe, result push.

## 3. Phase 0 - Harness and kernels (day 1) (5090 community, fallback 4090)

- P0.1 Pod template. Image `nvidia/cuda:12.8.1-devel-ubuntu22.04` (CUDA >= 12.8 for sm_120). Startup installs: PyTorch cu128, `cuda-python`, CuPy, Nsight Compute, PrismML llama.cpp fork built with `-DGGML_CUDA=ON`. Cold start < 3 min.
- P0.2 Persisting-L2 control. Exposes `cudaDeviceGetAttribute(MaxPersistingL2CacheSize)`, `cudaDeviceSetLimit(PersistingL2CacheSize)`, `cudaStreamSetAttribute(AccessPolicyWindow)` with hitRatio, hitProp=Persisting, missProp=Streaming, `cudaCtxResetPersistingL2Cache()`. Expose to PyTorch via `torch.utils.cpp_extension.load_inline`. Deliverable `l2persist.py`. Log device-reported max on every GPU.
- P0.3 Counter harness. `ncu --metrics lts__t_sectors_lookup_hit.sum,lts__t_sectors_lookup_miss.sum,dram__bytes_read.sum,gpu__time_duration.sum` wrapped so one call returns per-kernel hit rate, DRAM bytes/token, us/token. NVML power sampling at 10 ms for tok/J.
- P0.4 Ternary GEMV kernel. Batch-1 decode, 2-bit-packed ternary, FP16 group scales (g128). Variants: (a) LUT/popcount, (b) unpack-to-FP4-E2M1 in shared memory. Correctness vs FP16 reference; GB/s vs 1.79 TB/s. Target >= 70% of peak at 4096x4096.
- P0.5 CUDA Graph decode step. Toy hybrid (GDN + attention + MLP, ternary) in a CUDA Graph; us/token graph vs eager. Replaces 12 us/layer assumption in `proj.py`.
- P0.6 Result pipeline. `results/<exp>/<gpu>/<timestamp>.json`, retrieved before the failsafe kills the pod.

Exit: l2persist raises `lts__t_sectors_lookup_hit` on a 64 MB buffer re-read 100x; ternary GEMV >= 70% BW on 5090.

## 4. Phase 1 - Measure the ceiling

- E1 Synthetic working-set sweep. 5090 main, PRO 6000 confirm, 3090 control. W in {16,32,48,64,80,96,128,192,256,512,1024} MB ternary weights; decode loop reads all of W per token via P0.4 GEMV, 1000 tokens, CUDA Graph. Persisting off / on at hitRatio {0.5,0.75,1.0}. Record us/token, L2 hit %, DRAM bytes/token. Expect flat below persisting max then knee; 3090 on/off coincide. If 5090 "on" < 20% gain anywhere below 72 MB, stop and debug P0.2.
- E2 Stock hybrid models BF16. 5090. Qwen3.5-0.8B and Qwen3.5-2B, HF Transformers + torch.compile(mode="reduce-overhead"). Decode 512 tokens at context {0,4K,16K,32K,64K}. Record tok/s, DRAM bytes/token split weights / KV / GDN state. Replaces 1.4 TB/s and 12 us/layer constants in `proj_hybrid.py`.
- E3 GDN state + LM head residency. 5090. Persist (a) GDN recurrent state (19 MB), (b) LM head, (c) both. INT8 head (254 MB) and 48K-pruned INT8 head (98 MB). Expect <= 5% from (a); measurable gain from (c) with pruned head.
- E4 Bonsai 2 27B in the PrismML fork. PRO 6000. `llama-bench` / `llama-cli` with `PTQ1_0` vs `PQ2_0`, context {0,32K,128K,256K}, q8 KV on/off. ncu one decode step. Record tok/s, kernel time split (Hadamard unpack vs GEMV vs attention vs GDN), DRAM bytes/token. Answers: bandwidth-bound or unpack-bound.
- E5 Baseline speculative decoding. PRO 6000. Bonsai 2 27B target + Qwen3.5-0.8B BF16 draft, K in {4,6,8}, 200 agentic/code prompts. llama.cpp draft path or a 150-line PyTorch loop. Record acceptance rate, end-to-end tok/s.

Phase 1 exit: measured persisting max on 3 GPUs; E1 knee plot with 3090 control; measured floor + bandwidth -> `proj_hybrid.py` re-run; E4 kernel split; E5 acceptance + tok/s. Go/no-go: does persisting a <= 90 MB set cut DRAM bytes/token in proportion on a real model (E3c)? If no, write up and stop.

## 5. Phase 2 - Reuse architectures (5090 or PRO 6000)

- E6 Ternarize Qwen3.5-0.8B: BitNet b1.58 QAT + blockwise Hadamard (g128), 2-4B tokens, KL to Bonsai 2 27B teacher; full-vocab and 48K-vocab heads. Exit <= 10% ppl loss.
- E7 Looped core: blocks 1-2 (8 layers), loop-index embeddings, R in {2,3,4}; compare at equal storage to unrolled 0.8B and unrolled 8-layer. Thesis test: R=3 looped + persisted beats unrolled-0.8B-streamed on quality-per-us.
- E8 Hot core + cold skills (only if E7 margin < 15%).
- E9 Dense control: unrolled ternary 0.8B, 48K head, streamed.
- Exit: quality-vs-us/token figure, pick Phase 3 draft.

## 6. Phase 3 - Two-tier deployment (PRO 6000)

- E10 Draft + Bonsai 2 27B PQ2_0, K in {4,6,8,12}, q8 KV, context {0,32K,128K}, 500 agentic traces; baselines; L2 hit % before/after each verification.
- E11 Quality-at-throughput chart.

## 7. Launch cascade

PRO 6000: try `NVIDIA RTX PRO 6000 Blackwell Workstation Edition`, then Server Edition (both 600 W), then Max-Q Workstation (300 W; timings not comparable to the 600 W E4/E5 runs). 5090: community cloud.

## 8. l2persist sketch (untested)

```python
from cuda.bindings import runtime as rt
import torch

def _chk(r):
    err, *vals = r
    assert err == rt.cudaError_t.cudaSuccess, err
    return vals[0] if len(vals) == 1 else vals

def max_persisting_bytes(dev=0):
    return _chk(rt.cudaDeviceGetAttribute(
        rt.cudaDeviceAttr.cudaDevAttrMaxPersistingL2CacheSize, dev))

def enable(dev=0, nbytes=None):
    nbytes = nbytes or max_persisting_bytes(dev)
    _chk(rt.cudaDeviceSetLimit(rt.cudaLimit.cudaLimitPersistingL2CacheSize, nbytes))
    return nbytes

def persist(tensor, hit_ratio=1.0, stream=None):
    stream = stream or torch.cuda.current_stream()
    attr = rt.cudaStreamAttrValue()
    w = attr.accessPolicyWindow
    w.base_ptr = tensor.data_ptr()
    w.num_bytes = tensor.numel() * tensor.element_size()
    w.hitRatio = hit_ratio
    w.hitProp = rt.cudaAccessProperty.cudaAccessPropertyPersisting
    w.missProp = rt.cudaAccessProperty.cudaAccessPropertyStreaming
    _chk(rt.cudaStreamSetAttribute(stream.cuda_stream,
         rt.cudaStreamAttrID.cudaStreamAttributeAccessPolicyWindow, attr))

def reset(dev=0):
    _chk(rt.cudaCtxResetPersistingL2Cache())
```

`num_bytes` is capped by `cudaDevAttrMaxAccessPolicyWindowSize`. For CUDA-Graph-captured decode, set the attribute on kernel nodes (`cudaGraphKernelNodeSetAttribute`) or on the capture stream before capture - verify which the driver honors on sm_120 in P0.2.

## 9. Schedule (days)

- Day 1: P0.1-P0.6 on 5090. Exit: hit-rate counter moves; GEMV >= 70% BW.
- Day 2: E1 on 5090 + 3090; E2, E3 on 5090. Regenerate projections.
- Day 3: E1 confirm + E4 + E5 on PRO 6000. Go/no-go.
- Day 4-5: E6 ternary QAT (full + 48K vocab).
- Day 6-7: E7 looped core, R sweep.
- Day 8: E9 dense control; E8 only if E7 margin < 15%.
- Day 9: Phase 2 figure; pick draft.
- Day 10: E10 on PRO 6000.
- Day 11: E11 chart, write-up.

No purchase decision. RunPod is the primary platform for all runs, PRO 6000 included (cheaper than GCP). A PRO 6000 on GCP at work is the fallback only when RunPod has no PRO 6000 capacity.

## 10. PR track (end goal)

Status: the L2-residency scope below is closed (see Status). Current candidates: (1) fork PR enabling the PQ2_0 multi-column kernel on sm_120 plus `ggml_cuda_pdl_sync()` before its first activation read (prototype: `GGML_CUDA_PQ2_MULTICOL` on this branch, reference only); (2) upstream investigations: in-batch rollback for n-gram types, and skipping the draft re-eval (upstream TODO `TAG_SPEC_AVOID_DRAFT_REEVAL`). The authorship rules below still apply. The remainder of this section is the original, closed L2 plan.

Target: a PR to PrismML-Eng/llama.cpp `prism` with an L2-residency optimization for speculative decoding on large-L2 Blackwell cards, plus a matching draft model released as GGUF on Hugging Face (referenced from the PR, not included in it).

- Scope of the PR: a small, opt-in ggml-cuda change that pins the draft model's weights and LM head in persisting L2 (access policy window on CUDA Graph kernel nodes) during speculative decoding. Enabled only when the device reports a large persisting-L2 max; zero behavior change otherwise.
- Not in the PR: everything under `experiments/cache-resident/` (PyTorch harness, prototype kernels, RunPod scripts). These produce the evidence only.
- Draft model constraint: must run in llama.cpp with an existing architecture and the fork's existing quant types (`PQ2_0` / `PTQ1_0`). E9 dense ternary + 48K head is the default PR candidate; E7 looped core goes forward only if it wins by a wide margin, since it needs new arch support.
- Evidence the PR needs: E10 measured inside llama.cpp (not a PyTorch loop); L2 hit % of the draft before/after each target verification (eviction risk); no regression on 3090 (small L2), Ada, and with the feature off; `test-backend-ops` passes.
- Authorship: PrismML inherits llama.cpp AGENTS.md / CONTRIBUTING. The upstream patch must be authored or closely owned by the human contributor; no AI-written PR description or commit messages. Prototypes and measurements here are supporting material.
- Rebase onto the latest upstream `prism` immediately before submitting, not earlier.
- Every judge stage from Phase 2 on applies an upstreamability check: is the change small, opt-in, consistent with ggml-cuda patterns, free of new subsystems, and reviewable by a human who must defend it?
