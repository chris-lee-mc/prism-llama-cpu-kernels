# Phase 14: tensor-selective hybrid Q2_0 residency

With native-direct enabled, `GGML_TBKERN_Q2_0_HYBRID_CACHE` selects only named Q2_0 tensors for the existing cache-backed route:

```text
GGML_TBKERN_Q2_0_VNNI64_4R=1
GGML_TBKERN_Q2_0_VNNI64_NATIVE=1
GGML_TBKERN_Q2_0_HYBRID_CACHE=blk.*.attn_q.weight,blk.*.attn_k.weight
```

Entries are comma-separated complete tensor names or one `*` wildcard. Matching is anchored to the complete name using the prefix and suffix around the wildcard, so patterns such as `blk.*.attn_q.weight` are valid. Empty, malformed, or multiple-wildcard entries fail closed. Token embeddings remain excluded.

Matched tensors keep the native Q2_0 payload plus the existing packed-code cache and use the cache VNNI64/4-row path. Unmatched tensors keep native-only storage and use the Phase 13 direct path. Allocation, repack, and compute dispatch use the same per-tensor mode; Prism remains the fallback when selectors are unset or unsupported.

Set selectors before model load. Validate exact logits/token parity, unchanged perplexity, matched throughput, and resident memory before selecting a tensor family.
