# TBKERN Phase 16: opt-in Q2_0 activation preparation

Phase 16 adds an experimental path that prepares each activation row as a
pair of contiguous 32-byte Q8 vectors before the existing cache-backed
Q2_0 VNNI64 matvec. It is intended for measurements on real models; it is
not the default dispatch path.

## Enabling the path

Set both selectors:

```sh
GGML_TBKERN_Q2_0_VNNI64_4R=1 \
GGML_TBKERN_Q2_0_VNNI64_Q8PREP=1
```

The first selector enables the Phase 9 four-row cache-backed path. Q8PREP is
effective only when that selector is enabled, the binary was compiled with
AVX-512F/VNNI, and the runtime has AVX-512 VNNI support. If Q8PREP is not
effective, existing selector dispatch is unchanged: Phase 9 remains active
when selected, while views, embeddings, and non-contiguous tensors retain the
Prism-native path.

## Scratch ownership and size

The prepared vectors are ephemeral graph scratch owned by the ggml operation;
they are not persisted in the model or tensor cache. For one activation row of
width `K`, the implementation prepares `K/64` pair records, each 64 bytes, so
the prepared-vector region is exactly `K` bytes. The exact layout is:
`q8_bytes=(K/32)*sizeof(block_q8_0)` (34 bytes per block);
`sum_offset=PAD(q8_bytes,alignof(int32_t))`;
`base=PAD(sum_offset+(K/32)*sizeof(int32_t),GGML_MEM_ALIGN)`;
`prep_offset=PAD(base,64)`; `prep_bytes=(K/64)*64=K`; and total scratch
`PAD(prep_offset+K,GGML_MEM_ALIGN)`. This is one activation scratch region
per MUL_MAT invocation, not per output row. A short allocation or an
ineligible shape uses the non-prep route without changing results.

The operation synchronizes preparation and consumption with the ggml thread
barrier. Loads are deliberately unaligned because `params->wdata` only
guarantees ggml's allocation alignment, not a 64-byte AVX-512 boundary.

## Correctness and promotion status

Phase 16 passed deterministic logits/token parity and perplexity checks on the
real Bonsai 27B Q2_0 model, with no measured peak-RSS increase. The matched
8-thread decode result was approximately 0.7% above the Phase 9 control, below
the 3% promotion gate. It therefore remains an opt-in experiment and Phase 9
cache-backed VNNI64 remains the performance default.

When reporting a Q8PREP run, record the selector environment, source/binary
hashes, model hash, tensor `K` inventory, peak RSS, and the parity/PPL logs. Do not
infer an 8B speedup from the 27B result: smaller rows can make preparation
overhead dominant.