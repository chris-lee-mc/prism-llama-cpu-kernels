# Phase 3: opt-in AVX2 Q2_0 dispatch

`GGML_TBKERN_Q2_0_AVX2=1` enables the AVX2 implementation for eligible Prism Q2_0 matrices. It is deliberately separate from `GGML_TBKERN_Q2_0=1`: the latter remains the scalar correctness route, while the AVX2 selector is only active in binaries compiled with `__AVX2__`. On unsupported builds or hosts, the selector is ignored and Prism's native Q2_0 vec-dot path remains the fallback.

The AVX2 route uses the same Phase 1 cache ownership and threadpool row partition. Weight codes are decoded to unsigned values (0..3) for a 32-element block; `_mm256_maddubs_epi16` computes `code * q8`, and the scalar sum of q8 is subtracted to apply Prism's signed `code - 1` mapping. Pairwise products cannot saturate the int16 intermediate. Q8 activation scales and raw binary16 Q2 scales are widened at the same accumulation boundaries as the scalar route.

No nested OpenMP or per-operation heap allocation is introduced. The native Prism route remains the default when the environment variable is absent, and token embeddings are never repacked. AVX2 parity must be checked against the Phase 2 harness before collecting performance numbers.
