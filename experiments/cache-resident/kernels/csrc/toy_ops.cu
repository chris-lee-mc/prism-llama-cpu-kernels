// Decode-step ops for the toy hybrid model, see toy_ops.h.
#include "toy_ops.h"

#include <math.h>

namespace {

constexpr unsigned kFull = 0xffffffffu;

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(kFull, v, o);
  return v;
}

__device__ __forceinline__ float warp_max(float v) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(kFull, v, o));
  return v;
}

__device__ __forceinline__ float sigmoid_f(float x) { return 1.f / (1.f + expf(-x)); }

// ---- GDN step: grid (H, ceil(dv / 8)), block 256 = 8 warps, one warp per row v of S_h ----
template <int NC>
__global__ void __launch_bounds__(256) cr_gdn_step_kernel(float* __restrict__ S, const __half* __restrict__ qkv,
                                                          const __half* __restrict__ ab, __half* __restrict__ o, int H, int dv) {
  constexpr int dk = NC * 32;
  extern __shared__ float sm[];  // [q (dk) | k (dk)], normalized
  float* qs = sm;
  float* ks = sm + dk;
  const int h = blockIdx.x;
  const int tid = threadIdx.x;
  const int warp = tid >> 5;
  const int lane = tid & 31;

  const __half* qh = qkv + (size_t)h * dk;
  const __half* kh = qkv + (size_t)H * dk + (size_t)h * dk;
  const __half* vh = qkv + (size_t)2 * H * dk + (size_t)h * dv;
  for (int i = tid; i < dk; i += blockDim.x) {
    qs[i] = __half2float(qh[i]);
    ks[i] = __half2float(kh[i]);
  }
  __syncthreads();
  if (warp < 2) {
    float* p = warp == 0 ? qs : ks;
    float ss = 0.f;
#pragma unroll
    for (int c = 0; c < NC; ++c) {
      const float v = p[lane + 32 * c];
      ss += v * v;
    }
    ss = warp_sum(ss);
    const float inv = rsqrtf(ss + 1e-6f);
#pragma unroll
    for (int c = 0; c < NC; ++c) p[lane + 32 * c] *= inv;
  }
  __syncthreads();

  const int v = blockIdx.y * 8 + warp;
  if (v >= dv) return;
  const float alpha = sigmoid_f(__half2float(ab[h]));
  const float beta = sigmoid_f(__half2float(ab[H + h]));
  float qr[NC], kr[NC], sv[NC];
  float* Sr = S + ((size_t)h * dv + v) * dk;
  float dot = 0.f;
#pragma unroll
  for (int c = 0; c < NC; ++c) {
    qr[c] = qs[lane + 32 * c];
    kr[c] = ks[lane + 32 * c];
    sv[c] = Sr[lane + 32 * c];
    dot += sv[c] * kr[c];
  }
  dot = warp_sum(dot);
  const float delta = beta * (__half2float(vh[v]) - alpha * dot);
  float od = 0.f;
#pragma unroll
  for (int c = 0; c < NC; ++c) {
    const float n = alpha * sv[c] + delta * kr[c];
    Sr[lane + 32 * c] = n;
    od += n * qr[c];
  }
  od = warp_sum(od);
  if (lane == 0) o[(size_t)h * dv + v] = __float2half(od);
}

// ---- Attention decode: split over T (grid.x) and kv heads (grid.y), then combine ----
constexpr int kAttMaxGrp = 8;
constexpr int kAttMaxSplit = 128;
constexpr int kAttMaxHq = 32;
constexpr int kAttMaxD = 256;
constexpr int kAttCH = 64;  // tokens per sub-chunk, 2 per lane
constexpr int kAttThreads = 128;

__device__ float g_attn_acc[kAttMaxHq * kAttMaxSplit * kAttMaxD];
__device__ float g_attn_m[kAttMaxHq * kAttMaxSplit];
__device__ float g_attn_l[kAttMaxHq * kAttMaxSplit];

template <int E>
struct HVec;
template <>
struct HVec<8> {
  using T = uint4;
};
template <>
struct HVec<4> {
  using T = uint2;
};
template <>
struct HVec<2> {
  using T = unsigned int;
};

template <int E>
__device__ __forceinline__ void load_halves(const __half* p, float* f) {
  const typename HVec<E>::T v = *reinterpret_cast<const typename HVec<E>::T*>(p);
  const __half2* h2 = reinterpret_cast<const __half2*>(&v);
#pragma unroll
  for (int i = 0; i < E / 2; ++i) {
    const float2 t = __half22float2(h2[i]);
    f[2 * i] = t.x;
    f[2 * i + 1] = t.y;
  }
}

// E = D / 32 halves per lane. Block: 4 warps. Heads of one kv group share the K/V reads.
template <int E>
__global__ void __launch_bounds__(kAttThreads) cr_attn_decode_part_kernel(const __half* __restrict__ q, const __half* __restrict__ Kc,
                                                                         const __half* __restrict__ Vc, int Hq, int Hkv, int T,
                                                                         int chunk, float scale) {
  constexpr int D = E * 32;
  __shared__ float sc[kAttMaxGrp][kAttCH];
  __shared__ float m_s[kAttMaxGrp], l_s[kAttMaxGrp], f_s[kAttMaxGrp];

  const int grp = Hq / Hkv;
  const int hk = blockIdx.y;
  const int split = blockIdx.x;
  const int t0 = split * chunk;
  const int t1 = min(T, t0 + chunk);
  const int tid = threadIdx.x;
  const int warp = tid >> 5;
  const int lane = tid & 31;

  if (tid < kAttMaxGrp) {
    m_s[tid] = -INFINITY;
    l_s[tid] = 0.f;
    f_s[tid] = 1.f;
  }
  float qr[kAttMaxGrp][E];
#pragma unroll
  for (int j = 0; j < kAttMaxGrp; ++j) {
    if (j < grp) {
      load_halves<E>(q + (size_t)(hk * grp + j) * D + lane * E, qr[j]);
    } else {
#pragma unroll
      for (int e = 0; e < E; ++e) qr[j][e] = 0.f;
    }
  }
  float2 acc[kAttMaxGrp];
#pragma unroll
  for (int j = 0; j < kAttMaxGrp; ++j) acc[j] = make_float2(0.f, 0.f);
  __syncthreads();

  for (int cb = t0; cb < t1; cb += kAttCH) {
    const int n = min(kAttCH, t1 - cb);
    // phase A: scores, one warp per token
    for (int tl = warp; tl < n; tl += 4) {
      float kf[E];
      load_halves<E>(Kc + ((size_t)(cb + tl) * Hkv + hk) * D + lane * E, kf);
#pragma unroll
      for (int j = 0; j < kAttMaxGrp; ++j) {
        if (j < grp) {
          float d = 0.f;
#pragma unroll
          for (int e = 0; e < E; ++e) d += qr[j][e] * kf[e];
          d = warp_sum(d);
          if (lane == 0) sc[j][tl] = d * scale;
        }
      }
    }
    __syncthreads();
    // phase B: online softmax statistics, warp w handles heads w, w + 4
    for (int j = warp; j < grp; j += 4) {
      const float s0 = lane < n ? sc[j][lane] : -INFINITY;
      const float s1 = lane + 32 < n ? sc[j][lane + 32] : -INFINITY;
      const float mo = m_s[j];
      const float mn = fmaxf(mo, warp_max(fmaxf(s0, s1)));
      const float f = expf(mo - mn);
      const float p0 = lane < n ? expf(s0 - mn) : 0.f;
      const float p1 = lane + 32 < n ? expf(s1 - mn) : 0.f;
      sc[j][lane] = p0;
      sc[j][lane + 32] = p1;
      const float ps = warp_sum(p0 + p1);
      if (lane == 0) {
        l_s[j] = l_s[j] * f + ps;
        m_s[j] = mn;
        f_s[j] = f;
      }
    }
    __syncthreads();
    // phase C: weighted V sum, thread = one half2 of D
    if (tid < D / 2) {
#pragma unroll
      for (int j = 0; j < kAttMaxGrp; ++j) {
        acc[j].x *= f_s[j];
        acc[j].y *= f_s[j];
      }
#pragma unroll 4
      for (int tl = 0; tl < n; ++tl) {
        const float2 vf = __half22float2(reinterpret_cast<const __half2*>(Vc + ((size_t)(cb + tl) * Hkv + hk) * D)[tid]);
#pragma unroll
        for (int j = 0; j < kAttMaxGrp; ++j) {
          if (j < grp) {
            const float p = sc[j][tl];
            acc[j].x += p * vf.x;
            acc[j].y += p * vf.y;
          }
        }
      }
    }
    __syncthreads();
  }

  const int hq0 = hk * grp;
  if (tid < grp) {
    g_attn_m[(hq0 + tid) * kAttMaxSplit + split] = m_s[tid];
    g_attn_l[(hq0 + tid) * kAttMaxSplit + split] = l_s[tid];
  }
  if (tid < D / 2) {
#pragma unroll
    for (int j = 0; j < kAttMaxGrp; ++j) {
      if (j < grp) *reinterpret_cast<float2*>(&g_attn_acc[((size_t)(hq0 + j) * kAttMaxSplit + split) * kAttMaxD + 2 * tid]) = acc[j];
    }
  }
}

__global__ void __launch_bounds__(kAttThreads) cr_attn_decode_combine_kernel(__half* __restrict__ o, int D, int nsplit) {
  __shared__ float wgt[kAttMaxSplit];
  __shared__ float inv_l;
  const int h = blockIdx.x;
  const int tid = threadIdx.x;
  if (tid < 32) {
    float M = -INFINITY;
    for (int s = tid; s < nsplit; s += 32) M = fmaxf(M, g_attn_m[h * kAttMaxSplit + s]);
    M = warp_max(M);
    float L = 0.f;
    for (int s = tid; s < nsplit; s += 32) {
      const float w = expf(g_attn_m[h * kAttMaxSplit + s] - M);
      wgt[s] = w;
      L += w * g_attn_l[h * kAttMaxSplit + s];
    }
    L = warp_sum(L);
    if (tid == 0) inv_l = 1.f / L;
  }
  __syncthreads();
  if (tid < D / 2) {
    float ax = 0.f, ay = 0.f;
    for (int s = 0; s < nsplit; ++s) {
      const float w = wgt[s];
      const float2 v = *reinterpret_cast<const float2*>(&g_attn_acc[((size_t)h * kAttMaxSplit + s) * kAttMaxD + 2 * tid]);
      ax += w * v.x;
      ay += w * v.y;
    }
    *reinterpret_cast<__half2*>(o + (size_t)h * D + 2 * tid) = __floats2half2_rn(ax * inv_l, ay * inv_l);
  }
}

// ---- elementwise ----
__global__ void __launch_bounds__(256) cr_rmsnorm_kernel(const __half* x, const __half* __restrict__ w, __half* y, int n, float eps) {
  __shared__ float red[32];
  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;
  float ss = 0.f;
  for (int i = tid; i < n; i += blockDim.x) {
    const float v = __half2float(x[i]);
    ss += v * v;
  }
  ss = warp_sum(ss);
  if (lane == 0) red[warp] = ss;
  __syncthreads();
  float t = lane < (blockDim.x >> 5) ? red[lane] : 0.f;
  t = warp_sum(t);
  const float inv = rsqrtf(t / (float)n + eps);
  for (int i = tid; i < n; i += blockDim.x) y[i] = __float2half(__half2float(x[i]) * inv * __half2float(w[i]));
}

__global__ void cr_silu_mul_kernel(const __half* __restrict__ g, const __half* __restrict__ u, __half* __restrict__ y, int n) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
    const float a = __half2float(g[i]);
    y[i] = __float2half(a / (1.f + expf(-a)) * __half2float(u[i]));
  }
}

__global__ void cr_add_kernel(__half* __restrict__ a, const __half* __restrict__ b, int n) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x)
    a[i] = __float2half(__half2float(a[i]) + __half2float(b[i]));
}

inline int ew_blocks(int n) {
  const int b = (n + 255) / 256;
  return b < 1024 ? b : 1024;
}

}  // namespace

extern "C" cudaError_t cr_gdn_step(float* S, const __half* qkv, const __half* ab, __half* o, int H, int dk, int dv, cudaStream_t st) {
  if (!S || !qkv || !ab || !o || H < 1 || dv < 1) return cudaErrorInvalidValue;
  const dim3 grid(H, (dv + 7) / 8);
  const size_t smem = 2 * (size_t)dk * sizeof(float);
  switch (dk) {
    case 32: cr_gdn_step_kernel<1><<<grid, 256, smem, st>>>(S, qkv, ab, o, H, dv); break;
    case 64: cr_gdn_step_kernel<2><<<grid, 256, smem, st>>>(S, qkv, ab, o, H, dv); break;
    case 128: cr_gdn_step_kernel<4><<<grid, 256, smem, st>>>(S, qkv, ab, o, H, dv); break;
    case 256: cr_gdn_step_kernel<8><<<grid, 256, smem, st>>>(S, qkv, ab, o, H, dv); break;
    default: return cudaErrorInvalidValue;
  }
  return cudaGetLastError();
}

extern "C" cudaError_t cr_attn_decode(const __half* q, const __half* Kc, const __half* Vc, __half* o, int Hq, int Hkv, int D, int T,
                                      float scale, cudaStream_t st) {
  if (!q || !Kc || !Vc || !o || Hq < 1 || Hkv < 1 || T < 1 || Hq % Hkv != 0 || Hq > kAttMaxHq || Hq / Hkv > kAttMaxGrp) return cudaErrorInvalidValue;
  int ns = (T + 31) / 32;
  if (ns > kAttMaxSplit) ns = kAttMaxSplit;
  const int chunk = (T + ns - 1) / ns;
  const int nsplit = (T + chunk - 1) / chunk;
  const dim3 grid(nsplit, Hkv);
  switch (D) {
    case 64: cr_attn_decode_part_kernel<2><<<grid, kAttThreads, 0, st>>>(q, Kc, Vc, Hq, Hkv, T, chunk, scale); break;
    case 128: cr_attn_decode_part_kernel<4><<<grid, kAttThreads, 0, st>>>(q, Kc, Vc, Hq, Hkv, T, chunk, scale); break;
    case 256: cr_attn_decode_part_kernel<8><<<grid, kAttThreads, 0, st>>>(q, Kc, Vc, Hq, Hkv, T, chunk, scale); break;
    default: return cudaErrorInvalidValue;
  }
  cudaError_t e = cudaGetLastError();
  if (e != cudaSuccess) return e;
  cr_attn_decode_combine_kernel<<<Hq, kAttThreads, 0, st>>>(o, D, nsplit);
  return cudaGetLastError();
}

extern "C" cudaError_t cr_rmsnorm(const __half* x, const __half* w, __half* y, int n, float eps, cudaStream_t st) {
  if (!x || !w || !y || n < 1) return cudaErrorInvalidValue;
  cr_rmsnorm_kernel<<<1, 256, 0, st>>>(x, w, y, n, eps);
  return cudaGetLastError();
}

extern "C" cudaError_t cr_silu_mul(const __half* g, const __half* u, __half* y, int n, cudaStream_t st) {
  if (!g || !u || !y || n < 1) return cudaErrorInvalidValue;
  cr_silu_mul_kernel<<<ew_blocks(n), 256, 0, st>>>(g, u, y, n);
  return cudaGetLastError();
}

extern "C" cudaError_t cr_add(__half* a, const __half* b, int n, cudaStream_t st) {
  if (!a || !b || n < 1) return cudaErrorInvalidValue;
  cr_add_kernel<<<ew_blocks(n), 256, 0, st>>>(a, b, n);
  return cudaGetLastError();
}
