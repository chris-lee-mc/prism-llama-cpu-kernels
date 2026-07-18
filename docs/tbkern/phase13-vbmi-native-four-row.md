# Phase 13: VBMI native-direct four-row tile

When both selectors are enabled,

```text
GGML_TBKERN_Q2_0_VNNI64_NATIVE_4R=1
GGML_TBKERN_Q2_0_VNNI64_NATIVE_VBMI=1
```

the Phase 11 four-row native tile uses the Phase 12 AVX-512 VBMI expander for every full tile as well as its native-direct tails. This removes the prior mixed-expander behavior while retaining native-only storage and scalar per-row accumulation chains required for bit-exact parity.

Phase 9 cache-tiled VNNI64 remains the default performance route. This combined path is an opt-in low-memory alternative and must earn selection by matched decode results.