# Phase 9: Prism 27B real-weight evidence

This document records the promotion evidence for the opt-in Phase 9 CPU
decode path. It is deliberately not a claim about prompt processing or
perplexity throughput.

## Configuration

- Model: `Ternary-Bonsai-27B-Q2_0.gguf`, 7,165,121,600 bytes,
  SHA-256 `868c11714cf8fe47f5ec9eeb2be0ab1a337112886f92ee0ede6b855c4fa31757`.
- Host: Google Cloud C3 highmem-8, Intel Xeon Platinum 8481C, 4 physical
  cores plus SMT siblings, one NUMA node, AVX-512F/VNNI/VBMI.
- Decode command: `llama-bench -m MODEL -p 0 -n 16 -t 8 -ngl 0 -b 32 -ub 32`.
- Selector: `GGML_TBKERN_Q2_0_VNNI64_4R=1`.

## Correctness

`llama-debug` with prompt `Hello`, deterministic generation, and the same
runtime settings produced bit-identical files relative to the unmodified Prism
path:

- logits SHA-256: `65d581397ae42288fe1115e7fa434589700cacd98edc67a750c0fba1765062f2`
- generated-token SHA-256: `7a7748eacf971049271242b9d921628019d6c44698574e9301da9b8c88026381`
- NMSE: 0; top-1 mismatches: 0.

One-chunk `llama-perplexity` was also identical: `1.0645 +/- 0.03006`.

## Decode measurement

Twelve alternating one-repetition `llama-bench` rounds on the same VM and
model produced approximately 1.07--1.09 tok/s for native Prism and 1.45--1.53
tok/s for Phase 9. The later stable rounds were 1.53 tok/s, corresponding to
about a 40% gain over the native baseline.

The Phase 9 path can make the one-row recurrent perplexity workload slower
despite matching PPL, so it remains opt-in and decode-focused. Full raw logs
are intentionally not committed because they include a multi-gigabyte model
workflow; the command line, model hash, parity hashes, and metric values are
recorded above for reproduction.
