// Ternary GEMV kernels, see ternary_gemv.h and SPEC.md section 4.
//
// Block = W warps, RB = 32 * W rows, lane = row. K is cut in tiles of kTile groups (128 B of a row).
// Tiles stream through a ring of D slots in shared memory with cp.async (q and scales); each warp copy
// instruction reads 4 full 128 B lines. Per tile the block builds one table from x (LUT: pair sums,
// FP4: x as float) that all RB rows share; a lane then decodes its row from shared memory.
// Split-K (sm_90+): S blocks of one cluster share a row block and take tiles split, split + S, ...,
// so a cluster reads S neighbouring tiles of each row at a time. Partials meet in block 0 over DSMEM.
// The launcher picks (W, S, D) from the occupancy API: one wave if possible, most resident warps.
// PDL (sm_90+, opt-in with CR_GEMV_PDL=1): weight copies start before griddepcontrol.wait, x and y only after it.
// CR_FP4_MMA is reserved: no MMA path exists, see FP4.md.
#include "ternary_gemv.h"

#include <stdio.h>
#include <stdlib.h>

#include <algorithm>

namespace {

constexpr int kTile = 4;         // groups per tile
constexpr int kTileBytes = kTile * 32;
constexpr int kMaxSplit = 8;     // portable cluster size
constexpr int kMaxDepth = 3;     // ring slots
constexpr int kPair = 12;        // LUT floats per pair: entries c0 | c1 << 2 with c1 <= 2
constexpr int kLutGroup = 64 * kPair;
__host__ __device__ constexpr int table_floats(bool fp4) { return fp4 ? kTile * 128 : kTile * kLutGroup; }

__device__ __forceinline__ void cp_async16(void* smem, const void* g) {
  const unsigned sa = (unsigned)__cvta_generic_to_shared(smem);
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(sa), "l"(g) : "memory");
}

__device__ __forceinline__ void cp_async8(void* smem, const void* g) {
  const unsigned sa = (unsigned)__cvta_generic_to_shared(smem);
  asm volatile("cp.async.ca.shared.global [%0], [%1], 8;" ::"r"(sa), "l"(g) : "memory");
}

// Cluster barrier halves (sm_90+). Arrive early, wait late: the wait then costs nothing.
__device__ __forceinline__ void cluster_arrive_relaxed() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  asm volatile("barrier.cluster.arrive.relaxed.aligned;" ::: "memory");
#endif
}

__device__ __forceinline__ __attribute__((unused)) void cluster_arrive_release() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  asm volatile("barrier.cluster.arrive.release.aligned;" ::: "memory");
#endif
}

__device__ __forceinline__ __attribute__((unused)) void cluster_wait() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");
#endif
}

// Store v to the same shared variable in cluster block `rank`.
__device__ __forceinline__ __attribute__((unused)) void st_cluster(float* p, int rank, float v) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  const unsigned la = (unsigned)__cvta_generic_to_shared(p);
  unsigned ra;
  asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(ra) : "r"(la), "r"(rank));
  asm volatile("st.shared::cluster.f32 [%0], %1;" ::"r"(ra), "f"(v) : "memory");
#endif
}

__device__ __forceinline__ void cp_async_commit() { asm volatile("cp.async.commit_group;" ::: "memory"); }

template <int N>
__device__ __forceinline__ void cp_async_wait() {
  asm volatile("cp.async.wait_group %0;" ::"n"(N) : "memory");
}

// 16 B chunk c (0..7) of tile row r sits at chunk c ^ (r & 7): the 8 lanes (rows) of a 128-bit read phase hit 8 bank groups.
__device__ __forceinline__ int q_chunk(int r, int c) { return r * 8 + (c ^ (r & 7)); }

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
  static constexpr int kN = FP4 ? kTile * 64 / NT : kTile * 192 / NT;
  uint32_t v[kN];
};

template <int NT, bool FP4>
__device__ __forceinline__ void x_load(XTile<NT, FP4>& t, const __half* x, int g0, int ng) {
  const uint32_t* x2 = reinterpret_cast<const uint32_t*>(x);
#pragma unroll
  for (int it = 0; it < XTile<NT, FP4>::kN; ++it) {
    const int e = it * NT + threadIdx.x;
    const int u = FP4 ? e >> 6 : e / 192;
    const int p = FP4 ? e & 63 : (e - u * 192) / 3;
    t.v[it] = u < ng ? x2[(g0 + u) * 64 + p] : 0u;
  }
}

// LUT tile: group u, pair p, entry n = c0 | c1 << 2 (c1 <= 2) at float u * kLutGroup + p * kPair + n.
// Entry value t(c0) * x[2p] + t(c1) * x[2p + 1], t = {0, +1, -1, 0}. Chunk e = 3 * (64u + p) + c1 is stored at float 4e.
// FP4 tile: x of group u as float at u * 128 + k.
template <int NT, bool FP4>
__device__ __forceinline__ void x_store(float* sm, const XTile<NT, FP4>& t, int ng) {
#pragma unroll
  for (int it = 0; it < XTile<NT, FP4>::kN; ++it) {
    const int e = it * NT + threadIdx.x;
    const float2 xv = h2f2(t.v[it]);
    if (FP4) {
      if ((e >> 6) < ng) *reinterpret_cast<float2*>(sm + 2 * e) = xv;
    } else if (e / 192 < ng) {
      const int c1 = e % 3;
      const float t1 = c1 == 1 ? xv.y : (c1 == 2 ? -xv.y : 0.f);
      *reinterpret_cast<float4*>(sm + 4 * e) = make_float4(t1, t1 + xv.x, t1 - xv.x, t1);
    }
  }
}

// Dot of 64 weights (one uint4, pairs p0 .. p0 + 31 of a group) with the group LUT.
// Even nibbles of word w give byte offsets (w << 2) & 0x3c, odd nibbles (w >> 2) & 0x3c: one PRMT per lookup.
// All lanes look up the same pair, so the 12 entries sit in 12 banks: no conflict.
__device__ __forceinline__ float lut_dot(const float* lutp, uint4 w) {
  const char* base = reinterpret_cast<const char*>(lutp);
  const uint32_t ws[4] = {w.x, w.y, w.z, w.w};
  float a[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
  for (int wd = 0; wd < 4; ++wd) {
    const uint32_t ev = (ws[wd] << 2) & 0x3c3c3c3cu;
    const uint32_t od = (ws[wd] >> 2) & 0x3c3c3c3cu;
#pragma unroll
    for (int b = 0; b < 4; ++b) {
      const int pr = 8 * wd + 2 * b;
      a[b & 1] += lds_at(base + pr * kPair * 4, byte_of(ev, b));
      a[2 + (b & 1)] += lds_at(base + (pr + 1) * kPair * 4, byte_of(od, b));
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
// xh: the 64 x values of these weights (same for all lanes: broadcast reads).
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

__host__ __device__ __forceinline__ int n_tiles(int K) { return ((K >> 7) + kTile - 1) / kTile; }

// Ring slots actually allocated: a block never holds more than its tile count.
__host__ __device__ __forceinline__ int ring_slots(int K, int S, int D) { return min(D, (n_tiles(K) + S - 1) / S); }

// Dynamic shared memory: [q ring][scale ring][table][split-K partials].
__host__ __device__ __forceinline__ int smem_bytes(int W, bool fp4, int K, int S, int D) {
  const int RB = 32 * W;
  return ring_slots(K, S, D) * RB * (kTileBytes + 2 * kTile) + table_floats(fp4) * 4 + (S > 1 ? S * RB * 4 : 0);
}

template <int W, bool FP4>
__device__ __forceinline__ void gemv_body(const uint32_t* __restrict__ q, const __half* __restrict__ s, const __half* __restrict__ x,
                                          __half* __restrict__ y, int N, int K, int S, int D) {
  constexpr int NT = W * 32;
  constexpr int RB = 32 * W;
  constexpr int kQStage = RB * kTileBytes;  // bytes
  extern __shared__ __align__(16) unsigned char dsm[];
  const int slots = ring_slots(K, S, D);
  unsigned char* qs = dsm;                                                          // [slots][RB * 8] uint4, swizzled
  unsigned short* ss = reinterpret_cast<unsigned short*>(dsm + slots * kQStage);  // [slots][RB][kTile]
  float* table = reinterpret_cast<float*>(dsm + slots * (kQStage + RB * 2 * kTile));
  __shared__ __align__(16) uint32_t stage[FP4 ? W * 256 : 1];
  __shared__ float tab[16];

  const int tid = threadIdx.x;
  const int warp = tid >> 5;
  const int lane = tid & 31;
  const int rblk = blockIdx.x / S;
  const int split = blockIdx.x - rblk * S;
  const int rl = tid;  // row inside the block
  const int row = rblk * RB + rl;
  const int G = K >> 7;
  const int nt = (n_tiles(K) - split + S - 1) / S;  // tiles of this block: split, split + S, ...
  const size_t qpitch = (size_t)(K >> 2);           // bytes per row

  const unsigned char* qb = reinterpret_cast<const unsigned char*>(q);
  // scales ride in the ring when 8 B copies are legal, else they are loaded one tile ahead into registers
  const bool s8 = ((reinterpret_cast<uintptr_t>(s) & 7u) == 0) && ((G & 3) == 0);
  // copy tile tt of the row block into ring slot sl
  auto issue = [&](int tt, int sl) {
    const int nc = min(kTile, G - tt * kTile) * 2;  // valid 16 B chunks per row
#pragma unroll
    for (int it = 0; it < 8; ++it) {
      const int i = it * NT + tid;
      const int r = i >> 3, c = i & 7;
      const int gr = min(rblk * RB + r, N - 1);
      if (c < nc) cp_async16(qs + sl * kQStage + q_chunk(r, c) * 16, qb + (size_t)gr * qpitch + (size_t)tt * kTileBytes + c * 16);
    }
    if (s8) {
      const int gr = min(row, N - 1);
      cp_async8(ss + (sl * RB + rl) * kTile, reinterpret_cast<const unsigned short*>(s) + (size_t)gr * G + tt * kTile);
    }
  };
  const unsigned short* srow = reinterpret_cast<const unsigned short*>(s) + (size_t)(row < N ? row : N - 1) * (size_t)G;
  auto load_s = [&](int tt, unsigned short* v) {
#pragma unroll
    for (int u = 0; u < kTile; ++u) {
      const int g = tt * kTile + u;
      v[u] = g < G ? __ldg(srow + g) : (unsigned short)0;
    }
  };

  grid_dep_launch();
  if (S > 1) cluster_arrive_relaxed();  // phase 1: this block has started (DSMEM target is live)
  for (int p = 0; p < D - 1; ++p) {
    if (p < nt) issue(split + p * S, p);
    cp_async_commit();
  }
  unsigned short sc[kTile], sn[kTile];
  if (!s8) load_s(split, sc);
  grid_dep_wait();
  XTile<NT, FP4> xc, xn;
  x_load(xc, x, split * kTile, min(kTile, G - split * kTile));
  if (FP4 && tid < 16) tab[tid] = e2m1_value(tid);

  float acc = 0.f;
  for (int i = 0; i < nt; ++i) {
    const int tt = split + i * S;
    const int sl = i % D;
    if (D == 3) {
      cp_async_wait<1>();  // one group per tile: tile i has landed
    } else {
      cp_async_wait<0>();
    }
    __syncthreads();  // tile i visible; everyone is done with tile i - 1 and the table
    if (i + D - 1 < nt) issue(split + (i + D - 1) * S, (i + D - 1) % D);
    cp_async_commit();
    if (s8) {
      const uint2 v = *reinterpret_cast<const uint2*>(ss + (sl * RB + rl) * kTile);
      sc[0] = v.x & 0xffffu;
      sc[1] = v.x >> 16;
      sc[2] = v.y & 0xffffu;
      sc[3] = v.y >> 16;
    }
    if (i + 1 < nt) {
      if (!s8) load_s(tt + S, sn);
      x_load(xn, x, (tt + S) * kTile, min(kTile, G - (tt + S) * kTile));
    }
    const int ng = min(kTile, G - tt * kTile);
    x_store(table, xc, ng);
    __syncthreads();
    const uint4* qt = reinterpret_cast<const uint4*>(qs + sl * kQStage);
#pragma unroll
    for (int u = 0; u < kTile; ++u) {
      if (u < ng) {
        float d = 0.f;
#pragma unroll
        for (int hh = 0; hh < 2; ++hh) {
          const uint4 w = qt[q_chunk(rl, 2 * u + hh)];
          if (FP4) {
            d += fp4_dot(table + u * 128 + 64 * hh, reinterpret_cast<const char*>(tab), stage + warp * 256, lane, w);
          } else {
            d += lut_dot(table + u * kLutGroup + 32 * hh * kPair, w);
          }
        }
        acc = fmaf(h2f(sc[u]), d, acc);
      }
    }
    if (!s8) {
#pragma unroll
      for (int u = 0; u < kTile; ++u) sc[u] = sn[u];
    }
    xc = xn;
  }

  if (S == 1) {
    if (row < N) y[row] = __float2half(acc);
    return;
  }
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  // split-K: every block stores its partials into block 0 of the cluster, block 0 sums in a fixed order
  float* red = table + table_floats(FP4);  // [S][RB]
  cluster_wait();                          // phase 1 done: all blocks of the cluster run
  st_cluster(red + split * RB + rl, 0, acc);
  cluster_arrive_release();
  if (split != 0) return;
  cluster_wait();
  float t = 0.f;
  for (int k = 0; k < S; ++k) t += red[k * RB + rl];
  if (row < N) y[row] = __float2half(t);
#endif
}

// Kernel symbols start with the launcher name (ncu filters use it).
template <int W>
__global__ void __launch_bounds__(W * 32, W == 2 ? 6 : 16 / W) cr_gemv_lut_kernel(const uint32_t* __restrict__ q, const __half* __restrict__ s, const __half* __restrict__ x,
                                                             __half* __restrict__ y, int N, int K, int S, int D) {
  gemv_body<W, false>(q, s, x, y, N, K, S, D);
}

template <int W>
__global__ void __launch_bounds__(W * 32, W == 2 ? 6 : 16 / W) cr_gemv_fp4_kernel(const uint32_t* __restrict__ q, const __half* __restrict__ s, const __half* __restrict__ x,
                                                             __half* __restrict__ y, int N, int K, int S, int D) {
  gemv_body<W, true>(q, s, x, y, N, K, S, D);
}

inline bool args_ok(const void* q, const void* s, const void* x, const void* y, int N, int K) {
  if (!q || !s || !x || !y || N < 1 || K < 128 || (K & 127)) return false;
  if ((reinterpret_cast<uintptr_t>(q) | reinterpret_cast<uintptr_t>(x)) & 15u) return false;
  return true;
}

int env_int(const char* name, int def) {
  const char* e = getenv(name);
  return e && *e ? atoi(e) : def;
}

// Knobs, read once. CR_GEMV_PDL=1 enables PDL. CR_GEMV_W / _S / _D force the config, CR_GEMV_VERBOSE=1 prints it (tuning).
struct Knobs {
  int pdl = env_int("CR_GEMV_PDL", 0);
  int w = env_int("CR_GEMV_W", 0);
  int s = env_int("CR_GEMV_S", 0);
  int d = env_int("CR_GEMV_D", 0);
  int verbose = env_int("CR_GEMV_VERBOSE", 0);
};

const Knobs& knobs() {
  static const Knobs k;
  return k;
}

template <bool FP4>
const void* kernel_of(int W) {
  if (W == 2) return FP4 ? (const void*)cr_gemv_fp4_kernel<2> : (const void*)cr_gemv_lut_kernel<2>;
  if (W == 8) return FP4 ? (const void*)cr_gemv_fp4_kernel<8> : (const void*)cr_gemv_lut_kernel<8>;
  return FP4 ? (const void*)cr_gemv_fp4_kernel<4> : (const void*)cr_gemv_lut_kernel<4>;
}

struct Dev {
  int major = 0;
  int sms = 0;
  bool attr[2][3] = {};
  int occ[2][3][64] = {};  // blocks per SM by (fp4, W index, smem in KiB), 0 = unknown, -1 = none fit
};

Dev* dev() {
  static Dev cache[16];
  int d = 0;
  if (cudaGetDevice(&d) != cudaSuccess || d < 0 || d >= 16) return nullptr;
  Dev& c = cache[d];
  if (c.sms == 0) {
    int major = 0, sms = 0;
    cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, d);
    cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, d);
    c.major = major;
    c.sms = sms > 0 ? sms : 1;
  }
  return &c;
}

int widx(int W) { return W == 2 ? 0 : (W == 4 ? 1 : 2); }

template <bool FP4>
int occupancy(Dev& dv, int W, int smem) {
  const int kb = (smem + 1023) >> 10;
  if (kb >= 64) return 0;
  int& o = dv.occ[FP4][widx(W)][kb];
  if (o == 0) {
    const void* f = kernel_of<FP4>(W);
    if (!dv.attr[FP4][widx(W)]) {
      cudaFuncSetAttribute(f, cudaFuncAttributeMaxDynamicSharedMemorySize, 63 << 10);
      dv.attr[FP4][widx(W)] = true;
    }
    int n = 0;
    if (cudaOccupancyMaxActiveBlocksPerMultiprocessor(&n, f, W * 32, kb << 10) != cudaSuccess) n = 0;
    o = n > 0 ? n : -1;
  }
  return o > 0 ? o : 0;
}

struct Cfg {
  int W = 4, S = 1, D = 2, smem = 0;
};

// Pick (W, S, D): every SM busy, fewest waves, most resident warps, deepest ring, widest block, least split.
template <bool FP4>
Cfg choose(Dev& dv, int N, int K) {
  const Knobs& kn = knobs();
  const int T = n_tiles(K);
  Cfg best;
  long long bkey[5] = {0, 0, 0, 0, 0};
  bool have = false;
  for (int W : {2, 4, 8}) {
    if (kn.w && kn.w != W) continue;
    const int rblocks = (N + 32 * W - 1) / (32 * W);
    for (int S = 1; S <= kMaxSplit; S *= 2) {
      if (S > T || (S > 1 && dv.major < 9) || (kn.s && kn.s != S)) continue;
      for (int D = 2; D <= kMaxDepth; ++D) {
        if (kn.d && kn.d != D) continue;
        const int smem = smem_bytes(W, FP4, K, S, D);
        const int occ = occupancy<FP4>(dv, W, smem);
        if (occ == 0) continue;
        const long long blocks = (long long)rblocks * S, cap = (long long)occ * dv.sms;
        const long long waves = (blocks + cap - 1) / cap;
        const long long key[5] = {blocks >= dv.sms ? 1 : 0, -waves, std::min(blocks, cap) * W, ring_slots(K, S, D) * 16 + W, -S};
        bool better = !have;
        for (int i = 0; i < 5 && !better; ++i) {
          if (key[i] != bkey[i]) {
            better = key[i] > bkey[i];
            break;
          }
        }
        if (better) {
          have = true;
          for (int i = 0; i < 5; ++i) bkey[i] = key[i];
          best.W = W;
          best.S = S;
          best.D = D;
          best.smem = smem;
        }
      }
    }
  }
  return best;
}

template <bool FP4>
cudaError_t launch(const uint32_t* q, const __half* s, const __half* x, __half* y, int N, int K, cudaStream_t st) {
  if (!args_ok(q, s, x, y, N, K)) return cudaErrorInvalidValue;
  Dev* dv = dev();
  if (!dv) return cudaErrorInvalidDevice;
  const Cfg c = choose<FP4>(*dv, N, K);
  if (c.smem == 0) return cudaErrorInvalidConfiguration;
  if (knobs().verbose) {
    static int last_n = -1, last_k = -1;
    if (N != last_n || K != last_k) fprintf(stderr, "cr_gemv %s N=%d K=%d W=%d S=%d D=%d smem=%d\n", FP4 ? "fp4" : "lut", N, K, c.W, c.S, c.D, c.smem);
    last_n = N;
    last_k = K;
  }
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3((unsigned)(((N + 32 * c.W - 1) / (32 * c.W)) * c.S));
  cfg.blockDim = dim3(c.W * 32);
  cfg.dynamicSmemBytes = (size_t)c.smem;
  cfg.stream = st;
  cudaLaunchAttribute at[2];
  int na = 0;
  if (c.S > 1) {
    at[na].id = cudaLaunchAttributeClusterDimension;
    at[na].val.clusterDim.x = (unsigned)c.S;
    at[na].val.clusterDim.y = 1;
    at[na].val.clusterDim.z = 1;
    ++na;
  }
  if (dv->major >= 9 && knobs().pdl) {
    at[na].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    at[na].val.programmaticStreamSerializationAllowed = 1;
    ++na;
  }
  cfg.attrs = at;
  cfg.numAttrs = (unsigned)na;
  if (c.W == 2) {
    cudaLaunchKernelEx(&cfg, FP4 ? cr_gemv_fp4_kernel<2> : cr_gemv_lut_kernel<2>, q, s, x, y, N, K, c.S, c.D);
  } else if (c.W == 8) {
    cudaLaunchKernelEx(&cfg, FP4 ? cr_gemv_fp4_kernel<8> : cr_gemv_lut_kernel<8>, q, s, x, y, N, K, c.S, c.D);
  } else {
    cudaLaunchKernelEx(&cfg, FP4 ? cr_gemv_fp4_kernel<4> : cr_gemv_lut_kernel<4>, q, s, x, y, N, K, c.S, c.D);
  }
  return cudaGetLastError();
}

}  // namespace

extern "C" cudaError_t cr_gemv_lut(const uint32_t* q, const __half* s, const __half* x, __half* y, int N, int K, cudaStream_t st) {
  return launch<false>(q, s, x, y, N, K, st);
}

extern "C" cudaError_t cr_gemv_fp4(const uint32_t* q, const __half* s, const __half* x, __half* y, int N, int K, cudaStream_t st) {
  return launch<true>(q, s, x, y, N, K, st);
}
