# Phase 5: real-weight evidence snapshot

Validated on a temporary Google Compute Engine C3 `c3-highmem-8` VM (`Intel(R) Xeon(R) Platinum 8481C CPU @ 2.70GHz`) with eight threads, CPU-only execution, and the real `Ternary-Bonsai-27B-Q2_0.gguf` (7,154,128,896 bytes; 26,895,998,464 parameters). The VM was deleted after artifact transfer.

## Deterministic parity

The native Prism path and the AVX2 and VNNI opt-in routes were run with `llama-debug -p Hello -t 8 --save-logits`. The Phase 3 AVX2 and Phase 4 AVX-512/VNNI captures both pass `scripts/tbkern_compare.py` against native: 248,320 finite logits, max absolute error 0, NMSE 0, equal top-1 token, and byte-identical generated-token files. JSON reports and binary captures are in the local, untracked `prism-bench-results/phase3` and `prism-bench-results/phase4` directories.

## Perplexity

Matched one-chunk runs (`-f ppl.txt -t 8 -c 32 -b 32 --chunks 1`) report `PPL = 1.0645 +/- 0.03006` for both native and VNNI. The native run completed in about 31 seconds; VNNI took about 233 seconds. This is a correctness match, not a performance claim.

## Matched decode benchmark

`llama-bench -p 0 -n 16 -t 8 -ngl 0 -b 32 -ub 32 -r 1` on the same model measured:

| route | tok/s | avg ns |
|---|---:|---:|
| Prism native fallback | 1.090108 | 14,677,447,513 |
| opt-in AVX-512/VNNI | 0.086687 | 184,571,796,547 |

The VNNI route is currently approximately 12.6x slower than Prism's native vec-dot on this host. It is therefore an arithmetic/parity prototype, not yet an optimization. The overhead is expected to be investigated before proposing an upstream default; the native fallback remains the safe default.

Phase 3's matched AVX2 run was 0.131113 tok/s versus 1.091091 tok/s native under the same short decode shape. Earlier scalar/AVX2/VNNI long-shape CSVs are retained in `prism-bench-results/` for comparison, but their batch/context settings differ and should not be mixed with the table above.

## Memory and build notes

The GGUF remains mmap-loadable without a persistent repack requirement. C3 logs record the host-memory projection for model load; a separate resident-set experiment is still required to publish a final repack-memory number. The VNNI selector is compile-time gated by AVX-512VNNI/VL and opt-in via `GGML_TBKERN_Q2_0_VNNI=1`; otherwise Prism's native Q2_0 path owns the tensor.
