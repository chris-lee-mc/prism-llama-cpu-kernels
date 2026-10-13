// Ternary GEMV for batch-1 decode: y = W x, W 2-bit ternary [N,K] with fp16 group scales g128, x/y fp16, fp32 accumulate.
// Layout: see SPEC.md section 4.1. K % 128 == 0, q and x 16 B aligned.
// Launchers do no sync, no allocation, no memset: safe inside CUDA graph capture.
#pragma once
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Variant (a): shared-memory LUT of x pair sums, indexed by the packed 4-bit pair of codes.
cudaError_t cr_gemv_lut(const uint32_t* q, const __half* s, const __half* x, __half* y, int N, int K, cudaStream_t st);

// Variant (b): unpack codes to E2M1 nibbles in shared memory, decode through a 16-entry table, fp32 dot.
cudaError_t cr_gemv_fp4(const uint32_t* q, const __half* s, const __half* x, __half* y, int N, int K, cudaStream_t st);

#ifdef __cplusplus
}
#endif
