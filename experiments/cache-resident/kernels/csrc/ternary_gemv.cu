// Ternary GEMV kernels, see ternary_gemv.h and SPEC.md section 4.
//
// Mapping (both variants): block = CR_GEMV_WARPS warps handling 16 rows. Lane l = row (l & 15) + 16 * half (l >> 4).
// Warp w owns groups g = w, w + W, w + 2W, ... of K; the half picks 64 of the 128 weights of a group (one uint4).
// Each warp builds its own per-group tables, so only __syncwarp is needed. Partial sums meet in shared memory at the end.
// CR_FP4_MMA is reserved: no MMA path exists, see FP4.md. Tuning macros: CR_GEMV_WARPS, CR_GEMV_GB (pass via CR_NVCC_EXTRA).
#include "ternary_gemv.h"

#ifndef CR_GEMV_WARPS
#define CR_GEMV_WARPS 8
#endif
#ifndef CR_GEMV_GB
#define CR_GEMV_GB 4
#endif

namespace {

constexpr int kWarps = CR_GEMV_WARPS;
constexpr int kRows = 16;
constexpr int kGB = CR_GEMV_GB;  // groups prefetched per batch
constexpr unsigned kFull = 0xffffffffu;

// LUT layout (floats): pair p (weights 2p, 2p+1 of the group) owns 16 entries at p*16 + 16*(p>>5).
// The extra 16 floats per 32 pairs put the two halves in different banks.
// Inside a pair row the 4-entry chunk c1 (high 2 bits of the nibble) is stored at chunk c1 ^ ((p >> 1) & 3).
// That keeps the 128-bit stores of 8 neighbouring lanes in different banks.
constexpr int kLutRow = 16;
constexpr int kLutHalf = 33 * kLutRow;  // float offset of pair 32 relative to pair 0
constexpr int kLutFloats = 2 * kLutHalf;

struct Batch {
  uint4 q[kGB];
  uint32_t x0[kGB];  // half2 of pair `lane`
  uint32_t x1[kGB];  // half2 of pair `lane + 32`
  unsigned short s[kGB];
};

__device__ __forceinline__ void load_batch(Batch& b, int g0, int G, bool row_ok, const uint4* qrow, const unsigned short* srow,
                                           const uint32_t* x2, int lane, int hid) {
#pragma unroll
  for (int u = 0; u < kGB; ++u) {
    b.q[u] = make_uint4(0u, 0u, 0u, 0u);
    b.x0[u] = 0u;
    b.x1[u] = 0u;
    b.s[u] = 0;
    const int g = g0 + kWarps * u;
    if (g < G) {
      b.x0[u] = x2[g * 64 + lane];
      b.x1[u] = x2[g * 64 + 32 + lane];
      if (row_ok) {
        b.q[u] = __ldg(qrow + g * 2 + hid);
        b.s[u] = srow[g];
      }
    }
  }
}

__device__ __forceinline__ float2 h2f2(uint32_t v) {
  return __half22float2(*reinterpret_cast<const __half2*>(&v));
}

__device__ __forceinline__ float h2f(unsigned short v) {
  return __half2float(__ushort_as_half(v));
}

__device__ __forceinline__ void block_reduce_store(float acc, float (*part)[kRows], __half* y, int N) {
  const int tid = threadIdx.x;
  const int warp = tid >> 5;
  const int lane = tid & 31;
  acc += __shfl_xor_sync(kFull, acc, 16);
  if (lane < kRows) part[warp][lane] = acc;
  __syncthreads();
  if (tid < kRows) {
    float t = 0.f;
#pragma unroll
    for (int w = 0; w < kWarps; ++w) t += part[w][tid];
    const int row = blockIdx.x * kRows + tid;
    if (row < N) y[row] = __float2half(t);
  }
}

// Fill the 16 entries of one pair: entry (c0 | c1 << 2) = t(c0) * x0 + t(c1) * x1, t = {0, +1, -1, 0}.
__device__ __forceinline__ void lut_build_pair(float* dst, float x0, float x1, int r) {
  const float4 v0 = make_float4(0.f, x0, -x0, 0.f);
  const float4 v1 = make_float4(x1, x1 + x0, x1 - x0, x1);
  const float4 v2 = make_float4(-x1, x0 - x1, -x0 - x1, -x1);
  *reinterpret_cast<float4*>(dst + 4 * (0 ^ r)) = v0;
  *reinterpret_cast<float4*>(dst + 4 * (1 ^ r)) = v1;
  *reinterpret_cast<float4*>(dst + 4 * (2 ^ r)) = v2;
  *reinterpret_cast<float4*>(dst + 4 * (3 ^ r)) = v0;
}

__global__ void __launch_bounds__(kWarps * 32, 2) cr_gemv_lut_kernel(const uint32_t* __restrict__ q, const __half* __restrict__ s,
                                                                 const __half* __restrict__ x, __half* __restrict__ y, int N, int K) {
  __shared__ __align__(16) float lut[kWarps][kLutFloats];
  __shared__ float part[kWarps][kRows];

  const int tid = threadIdx.x;
  const int warp = tid >> 5;
  const int lane = tid & 31;
  const int hid = lane >> 4;
  const int row = blockIdx.x * kRows + (lane & 15);
  const bool row_ok = row < N;
  const int G = K >> 7;

  const uint4* qrow = reinterpret_cast<const uint4*>(q) + (size_t)(row_ok ? row : 0) * (size_t)(K >> 6);
  const unsigned short* srow = reinterpret_cast<const unsigned short*>(s) + (size_t)(row_ok ? row : 0) * (size_t)G;
  const uint32_t* x2 = reinterpret_cast<const uint32_t*>(x);

  float* lw = lut[warp];
  const int rb = (lane >> 1) & 3;                       // swizzle of pairs `lane` and `lane + 32`
  float* dst0 = lw + lane * kLutRow;                    // pair `lane`
  float* dst1 = lw + kLutHalf + lane * kLutRow;         // pair `lane + 32`
  const float* lb = lw + hid * kLutHalf;                // lookup base of this lane's half

  float acc = 0.f;
  Batch cur;
  load_batch(cur, warp, G, row_ok, qrow, srow, x2, lane, hid);

  for (int g0 = warp; g0 < G; g0 += kWarps * kGB) {
    const int gn = g0 + kWarps * kGB;
    Batch nxt;
    const bool more = gn < G;
    if (more) load_batch(nxt, gn, G, row_ok, qrow, srow, x2, lane, hid);

#pragma unroll
    for (int u = 0; u < kGB; ++u) {
      const int g = g0 + kWarps * u;
      if (g >= G) break;
      __syncwarp();
      const float2 xa = h2f2(cur.x0[u]);
      const float2 xb = h2f2(cur.x1[u]);
      lut_build_pair(dst0, xa.x, xa.y, rb);
      lut_build_pair(dst1, xb.x, xb.y, rb);
      __syncwarp();

      const uint32_t w4[4] = {cur.q[u].x, cur.q[u].y, cur.q[u].z, cur.q[u].w};
      float sum = 0.f;
#pragma unroll
      for (int wd = 0; wd < 4; ++wd) {
        float a = 0.f, b = 0.f;
#pragma unroll
        for (int j = 0; j < 8; j += 2) {
          const uint32_t n0 = ((w4[wd] >> (4 * j)) & 15u) ^ (uint32_t)((j >> 1) << 2);
          const uint32_t n1 = ((w4[wd] >> (4 * (j + 1))) & 15u) ^ (uint32_t)(((j + 1) >> 1) << 2);
          a += lb[(8 * wd + j) * kLutRow + n0];
          b += lb[(8 * wd + j + 1) * kLutRow + n1];
        }
        sum += a + b;
      }
      acc = fmaf(h2f(cur.s[u]), sum, acc);
    }
    if (!more) break;
    cur = nxt;
  }
  block_reduce_store(acc, part, y, N);
}

// E2M1 value of a 4-bit code: sign bit 3, exponent bits 2:1, mantissa bit 0, bias 1.
__device__ __forceinline__ float e2m1_value(int n) {
  const int m = n & 7;
  const float v = (m < 2) ? 0.5f * (float)m : ldexpf(1.f + 0.5f * (float)(m & 1), (m >> 1) - 1);
  return (n & 8) ? -v : v;
}

// 16 ternary codes -> two words of 8 E2M1 nibbles: `e` holds weights 0,2,4,..,14 and `o` weights 1,3,..,15.
// Codes: 00 -> 0, 01 -> +1 (0x2), 10 -> -1 (0xA), 11 -> 0.
__device__ __forceinline__ void to_e2m1(uint32_t w, uint32_t& e, uint32_t& o) {
  const uint32_t b0 = w & 0x55555555u;
  const uint32_t b1 = (w >> 1) & 0x55555555u;
  const uint32_t nz = b0 ^ b1;
  const uint32_t ng = b1 & ~b0;
  e = ((nz & 0x11111111u) << 1) | ((ng & 0x11111111u) << 3);
  o = (((nz >> 2) & 0x11111111u) << 1) | (((ng >> 2) & 0x11111111u) << 3);
}

__global__ void __launch_bounds__(kWarps * 32) cr_gemv_fp4_kernel(const uint32_t* __restrict__ q, const __half* __restrict__ s,
                                                                 const __half* __restrict__ x, __half* __restrict__ y, int N, int K) {
  __shared__ __align__(16) uint32_t stage[kWarps][256];  // per lane: 8 words of nibbles at [lane*4 ..] and [128 + lane*4 ..]
  __shared__ __align__(16) float2 xs[kWarps][64];        // x pairs of the current group
  __shared__ float tab[16];
  __shared__ float part[kWarps][kRows];

  const int tid = threadIdx.x;
  const int warp = tid >> 5;
  const int lane = tid & 31;
  const int hid = lane >> 4;
  const int row = blockIdx.x * kRows + (lane & 15);
  const bool row_ok = row < N;
  const int G = K >> 7;

  if (tid < 16) tab[tid] = e2m1_value(tid);
  __syncthreads();

  const uint4* qrow = reinterpret_cast<const uint4*>(q) + (size_t)(row_ok ? row : 0) * (size_t)(K >> 6);
  const unsigned short* srow = reinterpret_cast<const unsigned short*>(s) + (size_t)(row_ok ? row : 0) * (size_t)G;
  const uint32_t* x2 = reinterpret_cast<const uint32_t*>(x);

  uint32_t* st = stage[warp];
  float2* xw = xs[warp];

  float acc = 0.f;
  Batch cur;
  load_batch(cur, warp, G, row_ok, qrow, srow, x2, lane, hid);

  for (int g0 = warp; g0 < G; g0 += kWarps * kGB) {
    const int gn = g0 + kWarps * kGB;
    Batch nxt;
    const bool more = gn < G;
    if (more) load_batch(nxt, gn, G, row_ok, qrow, srow, x2, lane, hid);

#pragma unroll
    for (int u = 0; u < kGB; ++u) {
      const int g = g0 + kWarps * u;
      if (g >= G) break;
      __syncwarp();
      {
        uint32_t e[8];
        to_e2m1(cur.q[u].x, e[0], e[1]);
        to_e2m1(cur.q[u].y, e[2], e[3]);
        to_e2m1(cur.q[u].z, e[4], e[5]);
        to_e2m1(cur.q[u].w, e[6], e[7]);
        *reinterpret_cast<uint4*>(st + lane * 4) = make_uint4(e[0], e[1], e[2], e[3]);
        *reinterpret_cast<uint4*>(st + 128 + lane * 4) = make_uint4(e[4], e[5], e[6], e[7]);
        xw[lane] = h2f2(cur.x0[u]);
        xw[lane + 32] = h2f2(cur.x1[u]);
      }
      __syncwarp();

      const uint4 la = *reinterpret_cast<const uint4*>(st + lane * 4);
      const uint4 lb = *reinterpret_cast<const uint4*>(st + 128 + lane * 4);
      const uint32_t sw[8] = {la.x, la.y, la.z, la.w, lb.x, lb.y, lb.z, lb.w};
      float sum = 0.f;
#pragma unroll
      for (int wd = 0; wd < 4; ++wd) {
        const uint32_t ev = sw[2 * wd];
        const uint32_t od = sw[2 * wd + 1];
        float a = 0.f, b = 0.f;
#pragma unroll
        for (int j = 0; j < 8; j += 2) {
          const float2 xa = xw[hid * 32 + 8 * wd + j];
          const float2 xb = xw[hid * 32 + 8 * wd + j + 1];
          a += tab[(ev >> (4 * j)) & 15u] * xa.x + tab[(od >> (4 * j)) & 15u] * xa.y;
          b += tab[(ev >> (4 * (j + 1))) & 15u] * xb.x + tab[(od >> (4 * (j + 1))) & 15u] * xb.y;
        }
        sum += a + b;
      }
      acc = fmaf(h2f(cur.s[u]), sum, acc);
    }
    if (!more) break;
    cur = nxt;
  }
  block_reduce_store(acc, part, y, N);
}

inline bool args_ok(const void* q, const void* s, const void* x, const void* y, int N, int K) {
  if (!q || !s || !x || !y || N < 1 || K < 128 || (K & 127)) return false;
  if ((reinterpret_cast<uintptr_t>(q) | reinterpret_cast<uintptr_t>(x)) & 15u) return false;
  return true;
}

}  // namespace

extern "C" cudaError_t cr_gemv_lut(const uint32_t* q, const __half* s, const __half* x, __half* y, int N, int K, cudaStream_t st) {
  if (!args_ok(q, s, x, y, N, K)) return cudaErrorInvalidValue;
  cr_gemv_lut_kernel<<<(N + kRows - 1) / kRows, kWarps * 32, 0, st>>>(q, s, x, y, N, K);
  return cudaGetLastError();
}

extern "C" cudaError_t cr_gemv_fp4(const uint32_t* q, const __half* s, const __half* x, __half* y, int N, int K, cudaStream_t st) {
  if (!args_ok(q, s, x, y, N, K)) return cudaErrorInvalidValue;
  cr_gemv_fp4_kernel<<<(N + kRows - 1) / kRows, kWarps * 32, 0, st>>>(q, s, x, y, N, K);
  return cudaGetLastError();
}
