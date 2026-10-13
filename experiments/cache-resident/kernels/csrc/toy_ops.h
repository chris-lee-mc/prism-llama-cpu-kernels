// Small decode-step ops for the toy hybrid model (P0.5). All launchers are graph-capturable:
// no sync, no allocation, no memset; they launch on `st` and return cudaGetLastError().
#pragma once
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// GDN step, per head h: alpha=sigmoid(ab[h]), beta=sigmoid(ab[H+h]); q,k L2-normalized (x * rsqrt(sum x^2 + 1e-6));
// S_h = alpha*S_h - beta*(alpha*S_h k) k^T + beta v k^T ; o_h = S_h q.
// qkv = [q (H*dk) | k (H*dk) | v (H*dv)], head h uses slice h of each block. S is [H,dv,dk] fp32, updated in place.
// dk in {32,64,128,256}.
cudaError_t cr_gdn_step(float* S, const __half* qkv, const __half* ab, __half* o, int H, int dk, int dv, cudaStream_t st);

// Single-query GQA attention over a fixed-length fp16 cache. Kc/Vc are [T,Hkv,D]; q-head j of kv-head hk is hk*(Hq/Hkv)+j.
// D in {64,128,256}, Hq/Hkv <= 8, Hq <= 32. Uses a static device scratch: do not run two calls concurrently.
cudaError_t cr_attn_decode(const __half* q, const __half* Kc, const __half* Vc, __half* o, int Hq, int Hkv, int D, int T, float scale, cudaStream_t st);

// y = x * rsqrt(mean(x^2) + eps) * w. y may alias x.
cudaError_t cr_rmsnorm(const __half* x, const __half* w, __half* y, int n, float eps, cudaStream_t st);

// y = silu(g) * u
cudaError_t cr_silu_mul(const __half* g, const __half* u, __half* y, int n, cudaStream_t st);

// a += b
cudaError_t cr_add(__half* a, const __half* b, int n, cudaStream_t st);

#ifdef __cplusplus
}
#endif
