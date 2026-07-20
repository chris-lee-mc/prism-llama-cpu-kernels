# Phase 7: hybrid native storage and decode-only Q2_0 dispatch

The repack buffer now holds the original Prism Q2_0 bytes first, followed by the opt-in packed-code cache. `tensor->data` therefore remains a valid native Q2_0 tensor for all generic ggml operations.

The tbkern trait claims only F32 `MUL_MAT` operations with exactly one activation row. Operations with more than one row return `false` and fall through to Prism's original Q2_0 GEMM/GEMV implementation. For the real Qwen35/Gated-DeltaNet PPL command, the runtime presented one-row matmuls despite `-b 32`; it therefore used the opt-in decode path. Native storage still permits an explicit native PPL run by leaving `GGML_TBKERN_Q2_0_VNNI` unset.

For the decode-only path, the work buffer contains native `block_q8_0` activations plus one `int32_t` sum of `qs` per Q8 block. The sum is computed once after quantization and subtracted from each raw `dot(code, q8)` result. This removes the repeated VNNI all-ones correction dot from every output row while retaining exact integer arithmetic.

## C3 real-weight validation

On the eight-thread C3 host and real 27B Q2_0 GGUF, `llama-debug` logits/tokens were byte-identical to native (NMSE 0). Matched `llama-bench -p 0 -n 16 -t 8 -ngl 0 -b 32 -ub 32 -r 1` improved from 1.091734 native tok/s to 1.372374 opt-in tok/s (+25.7%).

One-chunk PPL remained exactly `1.0645 +/- 0.03006`, but the opt-in run took about 72.6 seconds versus 30.7 seconds native because that recurrent PPL workload exposes one-row matmuls. It must be reported separately from decode and is not a prefill/GEMM fallback demonstration.

Hybrid storage intentionally trades memory for generic fallback: native Q2_0 bytes plus the 2-bit auxiliary code cache are roughly 1.94x the native weight payload for affected tensors.
