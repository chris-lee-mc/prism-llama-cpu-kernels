# Cache-Resident LM - RunPod Execution Plan

*Companion to proposal v3.1. Sized to run on rented RunPod pods, with the RTX PRO 6000 Blackwell as the primary target and cheaper consumer cards used wherever the result transfers. Prices are RunPod on-demand as of Oct 2026; re-query before launching.*

## 1. GPU matrix

| GPU | Arch / SM | L2 | VRAM | BW | RunPod $/hr (community / secure) | Role |
|---|---|---|---|---|---|---|
| RTX PRO 6000 Blackwell (Workstation or Max-Q) | GB202 / sm_120 | 128 MB | 96 GB | 1.79 TB/s | $1.69 / $2.09 (Max-Q sometimes ~$0.50) | Final numbers; two-tier (27B + draft) runs; anything needing > 32 GB |
| RTX 5090 | GB202 / sm_120 | 96 MB | 32 GB | 1.79 TB/s | $0.69 / $1.19 | Primary dev card. Same die, same kernels, same persisting-L2 API. Everything except 27B two-tier runs. |
| RTX 4090 | AD102 / sm_89 | 72 MB | 24 GB | 1.01 TB/s | $0.34 / $0.74 | Cheap kernel-dev and cross-arch check. |
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

PRO 6000: try `NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition`, then Workstation, then Server Edition. 5090: community cloud.

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
