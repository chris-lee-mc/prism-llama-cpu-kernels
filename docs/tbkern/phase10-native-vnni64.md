# Phase 10: native-storage VNNI64 decode

Phase 10 adds an opt-in direct route:

```text
GGML_TBKERN_Q2_0_VNNI64_NATIVE=1
```

It leaves Prism `block_q2_0` storage in place and decodes its native sequential two-bit bytes directly into two 32-code vectors, then combines them for a 64-wide VNNI dot product. It retains the Phase 7 precomputed Q8-sum correction and the existing Q8 and Q2 scale/accumulation order.

Unlike cache-backed routes, this selector allocates only the native Q2_0 payload and stops after copying it during repack. It therefore removes the auxiliary code-cache allocation and pack/copy work. The selector takes precedence when another cache selector is also set. Native Prism is still the default; cache-backed Phase 7--9 routes remain available as independent controls.

The phase must demonstrate exact logits/token and PPL parity, then separately report matched decode throughput and real native-vs-cache load/repack memory cost. Direct decoding may trade some throughput for substantially lower memory and loading overhead.