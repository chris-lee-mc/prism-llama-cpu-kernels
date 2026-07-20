# Phase 6: vectorized AVX-512/VNNI Q2_0 decode

Phase 6 removes the scalar per-code expansion and output-lane reduction from
the opt-in `GGML_TBKERN_Q2_0_VNNI=1` path. Each Q8_0 dot covers 32 weights;
the Prism packed layout stores those codes as two 16-byte 2-bit planes inside
one 64-weight group. The implementation now loads the 16 packed bytes once,
extracts the low/high planes with byte masks and shifts, and combines them into
32 code bytes in registers. `_mm256_dpbusd_epi32` computes both the code dot
and the all-ones bias dot; horizontal integer reduction then preserves the
exact `(code - 1) * q8` mapping used by Prism's reference path.

The route remains opt-in and capability-gated (`__AVX512VNNI__`,
`__AVX512VL__`, and `ggml_cpu_has_avx512_vnni()`). Native Prism Q2_0 remains
the default fallback, and the scalar/AVX2 paths are unchanged. The helper
does not allocate or launch nested OpenMP work; row partitioning stays in the
ggml CPU threadpool.

Validation gates for this phase are:

* build the CPU backend with AVX-512/VNNI enabled;
* deterministic logits and generated-token parity against native Prism;
* one-chunk `llama-perplexity` equality within the existing tolerance;
* matched `llama-bench` measurements on the same host and thread count.
Build validation: a temporary C3 c3-highmem-4 compiled llama-debug, llama-bench, and llama-perplexity with -march=native and AVX-512VNNI present; the VM was deleted afterward. Real-model parity and timing for this revision remain Phase 7 because the validation VM did not contain the 27B GGUF.
