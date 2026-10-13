// Ternary GEMV kernels, see ternary_gemv.h and SPEC.md section 4.
//
// Block = W warps = RB = 16 * W rows. Lane l works on row (l >> 1) of its warp, half h = l & 1 of each group (64 weights).
// K is cut in tiles of kTile groups (128 B of a row). Tiles stream through a kStages ring in shared memory with cp.async:
// each warp copy instruction reads 4 full 128 B lines, so DRAM sees long contiguous runs. x tiles ride in the same ring.
// Per tile the block builds one table (LUT: pair sums of x, FP4: x as float) that all RB rows share.
// Split-K (sm_90+): S blocks of one cluster share a row block and take tiles split, split + S, ...; at any time a
// cluster reads S neighbouring tiles of each row. Partial sums meet over DSMEM.
// PDL (sm_90+, opt-in with CR_GEMV_PDL=1): weight copies start before griddepcontrol.wait, x and y only after it.
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
constexpr int kStages = 4;       // cp.async ring depth
constexpr int kTileBytes = kTile * 32;
constexpr unsigned kFull = 0xffffffffu;

// Weights are read once: skip L1, keep the default L2 policy (L2 persistence experiments rely on it).
__device__ __forceinline__ uint4 ld_stream(const uint4* p) {
  uint4 v;
  asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0, %1, %2, %3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p));
  return v;
}

__device__ __forceinline__ void cp_async16(void* smem, const void* g) {
  const unsigned sa = (unsigned)__cvta_generic_to_shared(smem);
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(sa), "l"(g) : "memory");
}

__device__ __forceinline__ void cp_async_commit() { asm volatile("cp.async.commit_group;" ::: "memory"); }

template <int N>
__device__ __forceinline__ void cp_async_wait() {
  asm volatile("cp.async.wait_group %0;" ::"n"(N) : "memory");
}

// 16 B chunk c (0..7) of tile row r sits at chunk c ^ ((r & 3) << 1): the 8 lanes of a 128-bit read phase hit 8 bank groups.
__device__ __forceinline__ int q_chunk(int r, int c) { return r * 8 + (c ^ ((r & 3) << 1)); }

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

// x of one tile in registers, loaded one tile ahead. LUT: one half2 pair per table chunk. FP4: one half2 per item.
template <int NT, bool FP4>
struct XTile {
  static constexpr int kN = FP4 ? kTile * 64 / NT : kTile * 256 / NT;
  uint32_t v[kN];
};

template <int NT, bool FP4>
__device__ __forceinline__ void x_load(XTile<NT, FP4>& t, const __half* x, int g0, int ng) {
  if (FP4) {
#pragma unroll
    for (int it = 0; it < XTile<NT, FP4>::kN; ++it) {
      const int idx = it * NT + threadIdx.x;
      const int u = idx >> 6;
      t.v[it] = u < ng ? reinterpret_cast<const uint32_t*>(x)[(g0 + u) * 64 + (idx & 63)] : 0u;
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
    for (int it = 0; it < XTile<NT, FP4>::kN; ++it) {
      const int idx = it * NT + threadIdx.x;
      const int u = idx >> 6, k = (idx & 63) * 2;
      if (u < ng) *reinterpret_cast<float2*>(sm + u * kXGroup + (k >> 6) * kXHalf + (k & 63)) = h2f2(t.v[it]);
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

// Ring slots actually used: a block never holds more than its tile count.
__host__ __device__ __forceinline__ int ring_slots(int K, int S) {
  const int T = ((K >> 7) + kTile - 1) / kTile;
  return min(kStages, (T + S - 1) / S);
}

template <int W, bool FP4>
__host__ __device__ constexpr int smem_bytes(int slots) {
  return slots * 16 * W * kTileBytes + (FP4 ? kTile * kXGroup : kTile * kLutGroup) * 4;
}

template <int W, bool FP4>
__device__ __forceinline__ void gemv_body(const uint32_t* __restrict__ q, const __half* __restrict__ s, const __half* __restrict__ x,
                                          __half* __restrict__ y, int N, int K, int S, int dbg) {
  constexpr int NT = W * 32;
  constexpr int RB = 16 * W;
  constexpr int kQStage = RB * kTileBytes;  // bytes
  extern __shared__ __align__(16) unsigned char dsm[];
  unsigned char* qs = dsm;  // [kStages][RB * 8] uint4, swizzled
  float* table = reinterpret_cast<float*>(dsm + ring_slots(K, S) * kQStage);
  __shared__ __align__(16) uint32_t stage[FP4 ? W * 256 : 1];
  __shared__ float tab[16];

  const int tid = threadIdx.x;
  const int warp = tid >> 5;
  const int lane = tid & 31;
  const int h = lane & 1;
  const int rblk = blockIdx.x / S;
  const int split = blockIdx.x - rblk * S;
  const int rl = warp * 16 + (lane >> 1);  // row inside the block
  const int row = rblk * RB + rl;
  const int G = K >> 7;
  const int T = (G + kTile - 1) / kTile;
  const int nt = (T - split + S - 1) / S;  // tiles of this block: split, split + S, ...
  const size_t qpitch = (size_t)(K >> 2);  // bytes per row

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

  const unsigned char* qb = reinterpret_cast<const unsigned char*>(q);
  // copy tile number tt of the row block into ring slot sl
  auto issue_q = [&](int tt, int sl) {
    const int nc = min(kTile, G - tt * kTile) * 2;  // valid 16 B chunks per row
#pragma unroll
    for (int it = 0; it < RB * 8 / NT; ++it) {
      const int i = it * NT + tid;
      const int r = i >> 3, c = i & 7;
      const int gr = min(rblk * RB + r, N - 1);
      if (c < nc) {
        unsigned char* dst = qs + sl * kQStage + q_chunk(r, c) * 16;
        const unsigned char* src = qb + (size_t)gr * qpitch + (size_t)tt * kTileBytes + c * 16;
        if (dbg & 64) {
          *reinterpret_cast<uint4*>(dst) = ld_stream(reinterpret_cast<const uint4*>(src));
        } else {
          cp_async16(dst, src);
        }
      }
    }
  };
  const unsigned short* srow = reinterpret_cast<const unsigned short*>(s) + (size_t)(row < N ? row : N - 1) * (size_t)G;
  auto load_s = [&](int tt, unsigned short* v) {
#pragma unroll
    for (int u = 0; u < kTile; ++u) {
      const int g = tt * kTile + u;
      v[u] = (g < G && !(dbg & 128)) ? __ldg(srow + g) : (unsigned short)0;
    }
  };

  grid_dep_launch();
#pragma unroll
  for (int p = 0; p < kStages - 1; ++p) {
    if (p < nt) issue_q(split + p * S, p);
    cp_async_commit();
  }
  unsigned short sc[kTile], sn[kTile];
  load_s(split, sc);
  grid_dep_wait();
  XTile<NT, FP4> xc, xn;
  x_load(xc, x, split * kTile, min(kTile, G - split * kTile));
  if (FP4 && tid < 16) tab[tid] = e2m1_value(tid);

  const uint32_t hb = h ? 0x40404040u : 0u;
  float acc = 0.f;
  for (int i = 0; i < nt; ++i) {
    const int tt = split + i * S;
    const int sl = i % kStages;
    cp_async_wait<kStages - 2>();  // one group per tile: tile i has landed
    __syncthreads();               // tile i visible; everyone is done with tile i - 1 and the table
    if (i + kStages - 1 < nt) issue_q(split + (i + kStages - 1) * S, (i + kStages - 1) % kStages);
    cp_async_commit();
    if (i + 1 < nt) {
      load_s(tt + S, sn);
      x_load(xn, x, (tt + S) * kTile, min(kTile, G - (tt + S) * kTile));
    }
    const int ng = min(kTile, G - tt * kTile);
    if (!(dbg & 8)) x_store(table, xc, ng);
    __syncthreads();
    const uint4* qt = reinterpret_cast<const uint4*>(qs + sl * kQStage);
#pragma unroll
    for (int u = 0; u < kTile; ++u) {
      if (u < ng) {
        const uint4 w = qt[q_chunk(rl, 2 * u + h)];
        float d;
        if (dbg & 1) {
          d = __uint_as_float((w.x ^ w.y ^ w.z ^ w.w) & 0x3fffffffu) * 1e-30f;
        } else if (FP4) {
          d = fp4_dot(table + u * kXGroup + h * kXHalf, reinterpret_cast<const char*>(tab), stage + warp * 256, lane, w);
        } else {
          d = lut_dot(table + u * kLutGroup, w, hb);
        }
        acc = fmaf(h2f(sc[u]), d, acc);
      }
    }
#pragma unroll
    for (int u = 0; u < kTile; ++u) sc[u] = sn[u];
    xc = xn;
  }
  acc += __shfl_xor_sync(kFull, acc, 1);

  if (S == 1 || (dbg & 32)) {
    if (h == 0 && row < N && split == 0) y[row] = __float2half(acc);
    return;
  }
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  __shared__ float red[RB];
  if (h == 0) red[rl] = acc;
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

// Tuning knobs, read once: CR_GEMV_PDL=1 enables PDL, CR_GEMV_BPS = target blocks per SM for split-K (default 3), CR_GEMV_W = warps per block.
bool pdl_on() {
  static const int v = env_int("CR_GEMV_PDL", 0);
  return v != 0;
}

// debug only: 1 skip dot, 8 skip table build, 16 coalesced read floor, 32 skip split-K reduce, 64 q via registers, 128 no s
int dbg_bits() {
  static const int v = env_int("CR_GEMV_DBG", 0);
  return v;
}

int blocks_per_sm() {
  static const int v = env_int("CR_GEMV_BPS", 3);
  return v;
}

template <int W, bool FP4>
cudaError_t launch_w(const uint32_t* q, const __half* s, const __half* x, __half* y, int N, int K, cudaStream_t st, const DevInfo& di) {
  constexpr int RB = 16 * W;
  const int rblocks = (N + RB - 1) / RB;
  const int T = ((K >> 7) + kTile - 1) / kTile;
  int S = 1;
  // split K over a cluster until there are about blocks_per_sm() blocks per SM
  if (di.major >= 9) {
    while (S < kMaxSplit && 2 * S <= T && rblocks * S < blocks_per_sm() * di.sms) S *= 2;
  }
  const int smem = smem_bytes<W, FP4>(ring_slots(K, S));
  static bool attr_done = false;
  if (!attr_done) {
    constexpr int smax = smem_bytes<W, FP4>(kStages);
    if (FP4) {
      cudaFuncSetAttribute(cr_gemv_fp4_kernel<W>, cudaFuncAttributeMaxDynamicSharedMemorySize, smax);
    } else {
      cudaFuncSetAttribute(cr_gemv_lut_kernel<W>, cudaFuncAttributeMaxDynamicSharedMemorySize, smax);
    }
    attr_done = true;
  }
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3((unsigned)(rblocks * S));
  cfg.blockDim = dim3(W * 32);
  cfg.dynamicSmemBytes = smem;
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
  static const int wenv = env_int("CR_GEMV_W", 0);
  int w = wenv;
  if (w == 0) w = (di.major < 9 && (N + 63) / 64 < di.sms) ? 2 : 4;
  if (w == 2) return launch_w<2, FP4>(q, s, x, y, N, K, st, di);
  if (w == 8) return launch_w<8, FP4>(q, s, x, y, N, K, st, di);
  return launch_w<4, FP4>(q, s, x, y, N, K, st, di);
}

}  // namespace

extern "C" cudaError_t cr_gemv_lut(const uint32_t* q, const __half* s, const __half* x, __half* y, int N, int K, cudaStream_t st) {
  return launch<false>(q, s, x, y, N, K, st);
}

extern "C" cudaError_t cr_gemv_fp4(const uint32_t* q, const __half* s, const __half* x, __half* y, int N, int K, cudaStream_t st) {
  return launch<true>(q, s, x, y, N, K, st);
}
