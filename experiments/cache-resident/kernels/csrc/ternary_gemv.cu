// Ternary GEMV kernels, see ternary_gemv.h and SPEC.md section 4.
//
// Mapping (both variants): block = W warps = 16 * W rows. Lane l = row (l >> 1) of its warp, half h = l & 1.
// Per group (128 weights, 32 B of a row) a lane holds one uint4: weights 64h .. 64h + 63. Lane pairs read full 32 B sectors.
// All lanes of a block walk the same K range in tiles of kTile groups, so one shared table per tile serves 16 * W rows.
// The q loads of the next tile are in flight while the current tile is decoded.
// Split-K (sm_90+): S blocks of one cluster share a row block and split its groups; partial sums meet over DSMEM.
// PDL (sm_90+, opt-in with CR_GEMV_PDL=1): weights are loaded before griddepcontrol.wait, x and y only after it.
// CR_FP4_MMA is reserved: no MMA path exists, see FP4.md.
#include "ternary_gemv.h"

#include <cooperative_groups.h>
#include <stdlib.h>

namespace cg = cooperative_groups;

namespace {

constexpr int kTile = 4;         // groups per tile
constexpr int kMaxSplit = 8;     // portable cluster size
constexpr int kSlot = 32;        // LUT floats per slot: pair j (h = 0) then pair 32 + j (h = 1)
constexpr int kLutGroup = 32 * kSlot;
constexpr int kXHalf = 68;       // fp4 x tile: 64 floats + 4 pad so the two halves use different banks
constexpr int kXGroup = 2 * kXHalf;
constexpr unsigned kFull = 0xffffffffu;

struct Tile {
  uint4 q[kTile];
  unsigned short s[kTile];
};

// Weights are read once: skip L1, keep the default L2 policy (L2 persistence experiments rely on it).
__device__ __forceinline__ uint4 ld_stream(const uint4* p) {
  uint4 v;
  asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0, %1, %2, %3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p));
  return v;
}

__device__ __forceinline__ void load_tile(Tile& t, const uint4* qrow, const unsigned short* srow, int g0, int ge, int dbg) {
#pragma unroll
  for (int u = 0; u < kTile; ++u) {
    const int g = g0 + u;
    t.q[u] = make_uint4(0u, 0u, 0u, 0u);
    t.s[u] = 0;
    if (g < ge) {
      t.q[u] = (dbg & 2) ? __ldg(qrow + 2 * g) : ld_stream(qrow + 2 * g);
      t.s[u] = __ldg(srow + g);
    }
  }
}

__device__ __forceinline__ float2 h2f2(uint32_t v) { return __half22float2(*reinterpret_cast<const __half2*>(&v)); }

__device__ __forceinline__ float h2f(unsigned short v) { return __half2float(__ushort_as_half(v)); }

__device__ __forceinline__ void grid_dep_wait() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  asm volatile("griddepcontrol.wait;" ::: "memory");
#endif
}

__device__ __forceinline__ void grid_dep_launch() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  asm volatile("griddepcontrol.launch_dependents;");
#endif
}

// byte b of v, zero extended
__device__ __forceinline__ uint32_t byte_of(uint32_t v, int b) { return __byte_perm(v, 0u, 0x4440u | (uint32_t)b); }

__device__ __forceinline__ float lds_at(const char* base, uint32_t off) { return *reinterpret_cast<const float*>(base + off); }

// x of one tile in registers, loaded one tile ahead. LUT: one half2 pair per table chunk. FP4: 4 halves per uint2.
template <int NT, bool FP4>
struct XTile {
  static constexpr int kN = FP4 ? kTile * 64 / NT : kTile * 256 / NT;
  uint32_t v[kN];
};

template <int NT, bool FP4>
__device__ __forceinline__ void x_load(XTile<NT, FP4>& t, const __half* x, int g0, int ng) {
  if (FP4) {
#pragma unroll
    for (int it = 0; it < XTile<NT, FP4>::kN / 2; ++it) {
      const int idx = it * NT + threadIdx.x;
      const int u = idx >> 5;
      uint2 v = make_uint2(0u, 0u);
      if (u < ng) v = reinterpret_cast<const uint2*>(x)[(g0 + u) * 32 + (idx & 31)];
      t.v[2 * it] = v.x;
      t.v[2 * it + 1] = v.y;
    }
  } else {
#pragma unroll
    for (int it = 0; it < XTile<NT, FP4>::kN; ++it) {
      const int e = it * NT + threadIdx.x;
      const int u = e >> 8, j = (e >> 3) & 31, c = e & 7;
      t.v[it] = u < ng ? reinterpret_cast<const uint32_t*>(x)[(g0 + u) * 64 + j + ((c >> 2) << 5)] : 0u;
    }
  }
}

// LUT tile: group u, slot j, entry n = c0 | c1 << 2 of pair p = j + 32h sits at float u * kLutGroup + j * kSlot + 16h + n.
// Entry value t(c0) * x[2p] + t(c1) * x[2p + 1], t = {0, +1, -1, 0}. 8 lanes write one 128 B slot: no bank conflict.
// fp4 x tile: group u, half h, weight i at float u * kXGroup + h * kXHalf + i.
template <int NT, bool FP4>
__device__ __forceinline__ void x_store(float* sm, const XTile<NT, FP4>& t, int ng) {
  if (FP4) {
#pragma unroll
    for (int it = 0; it < XTile<NT, FP4>::kN / 2; ++it) {
      const int idx = it * NT + threadIdx.x;
      const int u = idx >> 5, k = (idx & 31) * 4;
      if (u < ng) {
        const float2 a = h2f2(t.v[2 * it]), b = h2f2(t.v[2 * it + 1]);
        *reinterpret_cast<float4*>(sm + u * kXGroup + (k >> 6) * kXHalf + (k & 63)) = make_float4(a.x, a.y, b.x, b.y);
      }
    }
  } else {
#pragma unroll
    for (int it = 0; it < XTile<NT, FP4>::kN; ++it) {
      const int e = it * NT + threadIdx.x;
      const int u = e >> 8, j = (e >> 3) & 31, c = e & 7;
      if (u < ng) {
        const float2 xv = h2f2(t.v[it]);
        const int c1 = c & 3;
        const float t1 = c1 == 1 ? xv.y : (c1 == 2 ? -xv.y : 0.f);
        *reinterpret_cast<float4*>(sm + u * kLutGroup + j * kSlot + 4 * c) = make_float4(t1, t1 + xv.x, t1 - xv.x, t1);
      }
    }
  }
}

// Dot of one lane's 64 weights with the group LUT. hb = 0x40 in each byte for h = 1 (second half of the slot).
// Even nibbles of word w give byte offsets (w << 2) & 0x3c, odd nibbles (w >> 2) & 0x3c: one PRMT per lookup.
__device__ __forceinline__ float lut_dot(const float* lutg, uint4 w, uint32_t hb) {
  const char* base = reinterpret_cast<const char*>(lutg);
  const uint32_t ws[4] = {w.x, w.y, w.z, w.w};
  float a[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
  for (int wd = 0; wd < 4; ++wd) {
    const uint32_t ev = ((ws[wd] << 2) & 0x3c3c3c3cu) | hb;
    const uint32_t od = ((ws[wd] >> 2) & 0x3c3c3c3cu) | hb;
#pragma unroll
    for (int b = 0; b < 4; ++b) {
      const int slot = 8 * wd + 2 * b;
      a[b & 1] += lds_at(base + slot * kSlot * 4, byte_of(ev, b));
      a[2 + (b & 1)] += lds_at(base + (slot + 1) * kSlot * 4, byte_of(od, b));
    }
  }
  return (a[0] + a[1]) + (a[2] + a[3]);
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

// Stage the lane's 64 weights as E2M1 nibbles in shared memory, read back, decode via the 16-entry table, fp32 dot.
__device__ __forceinline__ float fp4_dot(const float* xh, const char* tabb, uint32_t* st, int lane, uint4 w) {
  uint32_t e[8];
  to_e2m1(w.x, e[0], e[1]);
  to_e2m1(w.y, e[2], e[3]);
  to_e2m1(w.z, e[4], e[5]);
  to_e2m1(w.w, e[6], e[7]);
  *reinterpret_cast<uint4*>(st + lane * 4) = make_uint4(e[0], e[1], e[2], e[3]);
  *reinterpret_cast<uint4*>(st + 128 + lane * 4) = make_uint4(e[4], e[5], e[6], e[7]);
  const uint4 la = *reinterpret_cast<const uint4*>(st + lane * 4);
  const uint4 lb = *reinterpret_cast<const uint4*>(st + 128 + lane * 4);
  const uint32_t sw[8] = {la.x, la.y, la.z, la.w, lb.x, lb.y, lb.z, lb.w};
  const float4* x4 = reinterpret_cast<const float4*>(xh);
  float a[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
  for (int wd = 0; wd < 4; ++wd) {
    const uint32_t ev = sw[2 * wd], od = sw[2 * wd + 1];
    const uint32_t el = (ev << 2) & 0x3c3c3c3cu, eh = (ev >> 2) & 0x3c3c3c3cu;  // weights 4b, 4b + 2
    const uint32_t ol = (od << 2) & 0x3c3c3c3cu, oh = (od >> 2) & 0x3c3c3c3cu;  // weights 4b + 1, 4b + 3
#pragma unroll
    for (int b = 0; b < 4; ++b) {
      const float4 xv = x4[4 * wd + b];
      a[b & 1] = fmaf(lds_at(tabb, byte_of(el, b)), xv.x, a[b & 1]);
      a[b & 1] = fmaf(lds_at(tabb, byte_of(ol, b)), xv.y, a[b & 1]);
      a[2 + (b & 1)] = fmaf(lds_at(tabb, byte_of(eh, b)), xv.z, a[2 + (b & 1)]);
      a[2 + (b & 1)] = fmaf(lds_at(tabb, byte_of(oh, b)), xv.w, a[2 + (b & 1)]);
    }
  }
  return (a[0] + a[1]) + (a[2] + a[3]);
}

template <int W, bool FP4>
__device__ __forceinline__ void gemv_body(const uint32_t* __restrict__ q, const __half* __restrict__ s, const __half* __restrict__ x,
                                          __half* __restrict__ y, int N, int K, int S, int dbg) {
  constexpr int NT = W * 32;
  constexpr int RB = 16 * W;
  constexpr int kSmem = FP4 ? kTile * kXGroup : kTile * kLutGroup;
  __shared__ __align__(16) float tile_smem[kSmem];
  __shared__ __align__(16) uint32_t stage[FP4 ? W * 256 : 1];
  __shared__ float tab[16];

  const int tid = threadIdx.x;
  const int warp = tid >> 5;
  const int lane = tid & 31;
  const int h = lane & 1;
  const int rblk = blockIdx.x / S;
  const int split = blockIdx.x - rblk * S;
  const int row = rblk * RB + warp * 16 + (lane >> 1);
  const int rowc = row < N ? row : N - 1;
  const int G = K >> 7;
  const int gb = (int)((long long)G * split / S);
  const int ge = (int)((long long)G * (split + 1) / S);

  const uint4* qrow = reinterpret_cast<const uint4*>(q) + (size_t)rowc * (size_t)(K >> 6) + h;
  const unsigned short* srow = reinterpret_cast<const unsigned short*>(s) + (size_t)rowc * (size_t)G;

  if (dbg & 16) {  // debug: coalesced streaming read of all q, the bandwidth floor of this launch shape
    const size_t n4 = (size_t)N * (size_t)(K >> 6);
    const size_t st = (size_t)gridDim.x * NT;
    uint32_t v = 0;
    size_t i = (size_t)blockIdx.x * NT + tid;
    for (; i + 3 * st < n4; i += 4 * st) {
      const uint4 a = ld_stream(reinterpret_cast<const uint4*>(q) + i), b = ld_stream(reinterpret_cast<const uint4*>(q) + i + st);
      const uint4 c = ld_stream(reinterpret_cast<const uint4*>(q) + i + 2 * st), d = ld_stream(reinterpret_cast<const uint4*>(q) + i + 3 * st);
      v ^= a.x ^ a.y ^ a.z ^ a.w ^ b.x ^ b.y ^ b.z ^ b.w ^ c.x ^ c.y ^ c.z ^ c.w ^ d.x ^ d.y ^ d.z ^ d.w;
    }
    for (; i < n4; i += st) {
      const uint4 a = ld_stream(reinterpret_cast<const uint4*>(q) + i);
      v ^= a.x ^ a.y ^ a.z ^ a.w;
    }
    if (v == 0x12345678u) y[0] = __float2half(1.f);
    return;
  }
  grid_dep_launch();
  Tile cur;
  load_tile(cur, qrow, srow, gb, ge, dbg);
  grid_dep_wait();
  if (FP4 && tid < 16) tab[tid] = e2m1_value(tid);

  XTile<NT, FP4> xc;
  x_load(xc, x, gb, min(kTile, ge - gb));

  const uint32_t hb = h ? 0x40404040u : 0u;
  float acc = 0.f;
  for (int g0 = gb; g0 < ge; g0 += kTile) {
    const int ng = min(kTile, ge - g0);
    const bool more = g0 + kTile < ge;
    Tile nxt;
    XTile<NT, FP4> xn;
    if (more) {
      load_tile(nxt, qrow, srow, g0 + kTile, ge, dbg);
      x_load(xn, x, g0 + kTile, min(kTile, ge - g0 - kTile));
    }
    if (!(dbg & 8)) {
      if (g0 != gb) __syncthreads();
      x_store(tile_smem, xc, ng);
      __syncthreads();
    }
#pragma unroll
    for (int u = 0; u < kTile; ++u) {
      if (u < ng) {
        float d;
        if (dbg & 1) {
          d = __uint_as_float((cur.q[u].x ^ cur.q[u].y ^ cur.q[u].z ^ cur.q[u].w) & 0x3fffffffu) * 1e-30f;
        } else if (FP4) {
          d = fp4_dot(tile_smem + u * kXGroup + h * kXHalf, reinterpret_cast<const char*>(tab), stage + warp * 256, lane, cur.q[u]);
        } else {
          d = lut_dot(tile_smem + u * kLutGroup, cur.q[u], hb);
        }
        acc = fmaf(h2f(cur.s[u]), d, acc);
      }
    }
    if (!more) break;
    cur = nxt;
    xc = xn;
  }
  acc += __shfl_xor_sync(kFull, acc, 1);

  if (S == 1) {
    if (h == 0 && row < N) y[row] = __float2half(acc);
    return;
  }
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  __shared__ float red[RB];
  if (h == 0) red[warp * 16 + (lane >> 1)] = acc;
  cg::cluster_group cl = cg::this_cluster();
  cl.sync();
  const int per = RB / S;
  if (tid < per) {
    const int lr = split * per + tid;
    float t = 0.f;
    for (int k = 0; k < S; ++k) t += cl.map_shared_rank(red, k)[lr];
    const int gr = rblk * RB + lr;
    if (gr < N) y[gr] = __float2half(t);
  }
  cl.sync();
#endif
}

// Kernel symbols start with the launcher name (ncu filters use it).
template <int W>
__global__ void __launch_bounds__(W * 32, 20 / W) cr_gemv_lut_kernel(const uint32_t* __restrict__ q, const __half* __restrict__ s, const __half* __restrict__ x,
                                                             __half* __restrict__ y, int N, int K, int S, int dbg) {
  gemv_body<W, false>(q, s, x, y, N, K, S, dbg);
}

template <int W>
__global__ void __launch_bounds__(W * 32, 20 / W) cr_gemv_fp4_kernel(const uint32_t* __restrict__ q, const __half* __restrict__ s, const __half* __restrict__ x,
                                                             __half* __restrict__ y, int N, int K, int S, int dbg) {
  gemv_body<W, true>(q, s, x, y, N, K, S, dbg);
}

inline bool args_ok(const void* q, const void* s, const void* x, const void* y, int N, int K) {
  if (!q || !s || !x || !y || N < 1 || K < 128 || (K & 127)) return false;
  if ((reinterpret_cast<uintptr_t>(q) | reinterpret_cast<uintptr_t>(x)) & 15u) return false;
  return true;
}

struct DevInfo {
  int major = 0;
  int sms = 0;
};

DevInfo dev_info() {
  static DevInfo cache[32];
  int d = 0;
  if (cudaGetDevice(&d) != cudaSuccess || d < 0 || d >= 32) return DevInfo{};
  DevInfo& c = cache[d];
  if (c.sms == 0) {
    int major = 0, sms = 0;
    cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, d);
    cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, d);
    c.major = major;
    c.sms = sms > 0 ? sms : 1;
  }
  return c;
}

int env_int(const char* name, int def) {
  const char* e = getenv(name);
  return e && *e ? atoi(e) : def;
}

// Tuning knobs, read once: CR_GEMV_PDL=1 enables PDL, CR_GEMV_BPS = target blocks per SM for split-K (default 2).
bool pdl_on() {
  static const int v = env_int("CR_GEMV_PDL", 0);
  return v != 0;
}

// debug only: 1 skip dot, 2 plain __ldg for q, 8 skip table build, 16 coalesced read floor
int dbg_bits() {
  static const int v = env_int("CR_GEMV_DBG", 0);
  return v;
}

int blocks_per_sm() {
  static const int v = env_int("CR_GEMV_BPS", 2);
  return v;
}

template <int W, bool FP4>
cudaError_t launch_w(const uint32_t* q, const __half* s, const __half* x, __half* y, int N, int K, cudaStream_t st, const DevInfo& di) {
  constexpr int RB = 16 * W;
  const int rblocks = (N + RB - 1) / RB;
  const int G = K >> 7;
  int S = 1;
  // split K over a cluster until there are about 2 blocks per SM
  if (di.major >= 9) {
    while (S < kMaxSplit && 2 * S <= G && rblocks * S < blocks_per_sm() * di.sms) S *= 2;
  }
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3((unsigned)(rblocks * S));
  cfg.blockDim = dim3(W * 32);
  cfg.dynamicSmemBytes = 0;
  cfg.stream = st;
  cudaLaunchAttribute at[2];
  int na = 0;
  if (S > 1) {
    at[na].id = cudaLaunchAttributeClusterDimension;
    at[na].val.clusterDim.x = (unsigned)S;
    at[na].val.clusterDim.y = 1;
    at[na].val.clusterDim.z = 1;
    ++na;
  }
  if (di.major >= 9 && pdl_on()) {
    at[na].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    at[na].val.programmaticStreamSerializationAllowed = 1;
    ++na;
  }
  cfg.attrs = at;
  cfg.numAttrs = (unsigned)na;
  if (FP4) {
    cudaLaunchKernelEx(&cfg, cr_gemv_fp4_kernel<W>, q, s, x, y, N, K, S, dbg_bits());
  } else {
    cudaLaunchKernelEx(&cfg, cr_gemv_lut_kernel<W>, q, s, x, y, N, K, S, dbg_bits());
  }
  return cudaGetLastError();
}

template <bool FP4>
cudaError_t launch(const uint32_t* q, const __half* s, const __half* x, __half* y, int N, int K, cudaStream_t st) {
  if (!args_ok(q, s, x, y, N, K)) return cudaErrorInvalidValue;
  const DevInfo di = dev_info();
  // without clusters, use smaller row blocks when 64-row blocks cannot fill the GPU
  if (di.major < 9 && (N + 63) / 64 < di.sms) return launch_w<2, FP4>(q, s, x, y, N, K, st, di);
  return launch_w<4, FP4>(q, s, x, y, N, K, st, di);
}

}  // namespace

extern "C" cudaError_t cr_gemv_lut(const uint32_t* q, const __half* s, const __half* x, __half* y, int N, int K, cudaStream_t st) {
  return launch<false>(q, s, x, y, N, K, st);
}

extern "C" cudaError_t cr_gemv_fp4(const uint32_t* q, const __half* s, const __half* x, __half* y, int N, int K, cudaStream_t st) {
  return launch<true>(q, s, x, y, N, K, st);
}
