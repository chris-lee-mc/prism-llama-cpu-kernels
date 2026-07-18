# Phase 12: VBMI native Q2_0 expansion

Phase 12 adds a separate opt-in native-direct selector:

```text
GGML_TBKERN_Q2_0_VNNI64_NATIVE_VBMI=1
```

For a native 64-value Q2_0 subgroup, it constructs four 16-byte logical two-bit code planes from the packed native bytes and uses AVX-512 VBMI byte permutation to interleave them into sequential code order before the existing 64-wide VNNI dot. This replaces Phase 10's two 32-value shuffle/widen/multiply expansions.

It requires AVX2, AVX-512F, AVX-512VNNI, and AVX-512VBMI at compile and runtime. It remains native-only storage: no auxiliary cache or repack expansion. Phase 9 cache-tiled VNNI64 remains the performance default unless measurements show otherwise.