# TBKERN Phase 19: threadpool-owned shared Q8 activation cache

Phase 19 adds an opt-in activation-side cache for Q2_0 CPU matvecs. Set
GGML_TBKERN_Q2_0_SHARED_Q8=1 together with the existing Q2_0 dispatch selector
to enable it. The default remains unchanged.

The cache is owned by the ggml threadpool and is keyed by graph epoch, source
tensor/data identity, K, and Q8PREP mode. Worker 0 reserves and quantizes the
Q8 blocks and integer sums before the existing graph barrier; all workers then
consume the immutable published buffer. A reserve failure falls back to the
normal params->wdata scratch. The cache is freed with its owning threadpool.

OpenMP graph epochs are advanced at every graph dispatch so a reused source
pointer cannot reuse stale activation data across graph executions. Native
Prism/direct-native paths are not changed.

The real group-128 Bonsai 8B test on the C3 VM passed exact logits/token parity
and identical perplexity, but showed no decode gain (4.91 tok/s for both native
and shared paths) and identical peak RSS. Consequently this remains
experimental/off by default; Phase 9 VNNI64 four-row dispatch is still the
performance default.
