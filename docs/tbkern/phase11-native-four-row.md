# Phase 11: native-direct four-row VNNI64 tile

Phase 11 adds a native-storage, four-output-row opt-in route:

```text
GGML_TBKERN_Q2_0_VNNI64_NATIVE_4R=1
```

It keeps Phase 10's native-only allocation and native Q2_0 byte expansion, but forms each 64-value Q8 activation vector once per subgroup and shares it across four independent output rows. Each row retains its original block order, Q8 scales and sum correction, and Q2 scale accumulation. Remaining one to three rows use the Phase 10 native-direct path.

Phase 9 cache-tiled VNNI64 remains the performance-default candidate. This route is an opt-in memory-constrained comparator; no cache selector or default Prism behavior changes.