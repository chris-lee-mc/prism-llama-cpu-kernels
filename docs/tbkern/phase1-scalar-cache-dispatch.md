# Phase 1: scalar Q2_0 cache dispatch

## Scope

This phase provides an opt-in, CPU-threadpool-owned functional path for Prism Q2_0 (`QK2_0=128`) matrices. It is a correctness and ownership baseline, not a SIMD performance path.

## Opt-in and ownership

`GGML_TBKERN_Q2_0=1` selects CPU_REPACK only for eligible 2D Q2_0 matrices with positive dimensions and a K dimension divisible by `QK2_0` (128). This ensures Kp equals K. Set this variable before model load and keep it unchanged for the process lifetime: disabling it after a tensor is cached would not reconstruct Prism's native layout. Without the explicit environment variable, the tensor retains Prism's native layout and its native Q2_0 vec-dot is unchanged. Token embeddings remain native.

The repacked tensor owns its packed-code cache and the original Q2_0 binary16 block-scale bits. It does not pre-widen those scales: the scalar loop converts each scale immediately before multiplying its four Q8 partial sums, matching Prism's native Q2_0 accumulation representation and order. Each `MUL_MAT` obtains activation scratch exclusively from ggml's `wdata`; no per-invocation heap allocation or tbkern OpenMP/SIMD entry point is used.

## Layout and numerics

Prism's branch uses Q2_0 groups of 128 weights: each source block is 34 bytes (two-byte binary16 scale plus 32 packed code bytes).

Cache layout is tbkern `CODES`: every 64 weights occupy 16 bytes, with each byte encoding positions `b, b+16, b+32, b+48`. Codes decode as `code - 1`. The activation scratch is native `block_q8_0`: one fp16 scale and 32 signed int8 values per block. It is produced with ggml's `quantize_row_q8_0`, so its quantization and fp16 rounding match Prism's Q2_0 x Q8_0 input path.

After a threadpool barrier, decode (`n_tokens=1`) statically partitions output rows `[M*ith/nth, M*(ith+1)/nth)`. Every row is written exactly once by a ggml worker. Arithmetic remains one Q2_0-128 block as four 32-wide Q8 dot products, each multiplied by its widened Q8 fp16 scale; those four partials are accumulated, then multiplied once by the Q2 scale widened directly from its preserved binary16 bits.

## Acceptance checks

1. Without `GGML_TBKERN_Q2_0`, native Prism loading and decode remain unchanged.
2. With `GGML_TBKERN_Q2_0=1`, deterministic generated tokens and top-1 logits match native on a real Q2_0 model. Compare full logits with an explicit tolerance: Phase 1 accepts NMSE <= 1e-4 and records max absolute error; bit-exact logits require matching Prism's optimized ISA and reduction order.
3. Multi-thread decode covers all rows exactly once; sanitizers show no scratch races.
4. Perplexity remains within the predefined tolerance before later SIMD work.
