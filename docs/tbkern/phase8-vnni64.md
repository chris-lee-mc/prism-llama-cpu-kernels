# Phase 8: 64-wide VNNI cache decode

Phase 8 adds an experimental, separately opt-in Q2_0 decode path:

```text
GGML_TBKERN_Q2_0_VNNI64=1
```

It retains native Prism Q2_0 storage and the Phase 7 cache allocation. For each 64-weight cache subgroup it loads the 16 packed code bytes once, expands the four 2-bit planes into a 64-byte AVX-512 vector, and performs two adjacent 32-element Q8_0 dot products in one VNNI operation. Q8_0 scales and precomputed sums remain per 32 values; accumulation order is unchanged.

The selector is intentionally opt-in. When it is disabled or unavailable, the Phase 7 32-wide VNNI selector (`GGML_TBKERN_Q2_0_VNNI=1`) and native Prism route retain their prior behavior. The route requires an AVX-512F + AVX-512VNNI build and host capability.

Validation requirements before performance claims:

- native-versus-VNNI64 deterministic logits and generated-token parity;
- `llama-perplexity` parity on the same 27B GGUF and command line;
- matched 8-thread CPU decode benchmarks, reported independently from PPL.