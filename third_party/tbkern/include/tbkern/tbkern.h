/* tbkern/tbkern.h — FROZEN Phase-0 public contract.
 *
 * This header is the inter-agent contract for the whole library: the packed
 * matrix handle, error codes, the kernel-dispatch API, packing/quant/gemv/gemm
 * entry points, and the memory-arena API. Eight downstream agents code against
 * it. It is designed to be complete enough that none of them must ask a
 * question about a signature or a field.
 *
 * RULE (docs/CONTEXT.md §Rules): after Phase 0 this header may only be changed
 * by the orchestrator. If you think it is wrong, STOP and report — do not edit.
 *
 * See docs/FORMAT.md for the on-disk (GGUF Q2_0) and in-memory (bitplane /
 * interleaved-codes) formats, and docs/CONTEXT.md for the project brief.
 */
#ifndef TBKERN_TBKERN_H
#define TBKERN_TBKERN_H

#include <stdint.h>
#include <stddef.h>
#include "tbkern/format.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ================================================================== *
 *  Error codes  (0 == success, negative == error)
 * ================================================================== */
#define TBK_OK            0   /* success                                        */
#define TBK_EINVAL       -1   /* invalid argument (null ptr, bad G, bad shape)  */
#define TBK_ENOMEM       -2   /* allocation failure                             */
#define TBK_ENOSYS       -3   /* kernel/feature not implemented (e.g. AMX slot) */
#define TBK_ENOTTERNARY  -4   /* code 3 (+2) found where a ternary-only layout  */
                              /* (bitplane / MF path) was requested             */
#define TBK_EIO          -5   /* I/O or file-format error (GGUF reader)         */
#define TBK_ERANGE       -6   /* value out of representable range               */

/* Human-readable string for an error code (never NULL). */
const char *tbk_strerror(int err);

/* ================================================================== *
 *  Packed-matrix layouts
 * ================================================================== */

/* Kernel-native in-memory weight layout kind. Both are repacked once from the
 * GGUF Q2_0 blocks at model load. See docs/FORMAT.md for exact bit layouts. */
typedef enum {
    /* Layout A: per row, per 16-weight chunk a pair of 16-bit mask words
     * [P, N]; bit i of P => w=+1, bit i of N => w=-1, neither => 0. Ternary
     * ONLY (no +2 representable); packing a code-3 weight returns
     * TBK_ENOTTERNARY. Consumed by gemv_avx512_mf.c (multiplication-free). */
    TBK_LAYOUT_BITPLANE = 0,

    /* Layout B: per row, 64-weight subgroups stored as 16 interleaved bytes
     * (byte b holds codes for k = base+b, +16, +32, +48). Codes are the
     * unsigned u = w+1 in {0..3}, fed directly to vpdpbusd. Represents all 4
     * codes incl. +2. Consumed by the VNNI gemv/gemm and AVX2 paths. */
    TBK_LAYOUT_CODES    = 1
} tbk_layout_kind;

/* Packed matrix handle. Rows = output features (M), columns = input features
 * (K, padded to Kp). Allocated/filled by tbk_pack_from_q2, released by
 * tbk_mat_free. Treat fields as read-only after packing. */
typedef struct {
    int32_t M;                 /* rows (output features)                        */
    int32_t K;                 /* logical input width (unpadded)                */
    int32_t Kp;                /* padded input width, multiple of G (>= K)      */
    int32_t G;                 /* group size, 64 or 128                         */
    tbk_layout_kind layout;    /* which in-memory layout `data` holds           */

    /* Packed weight codes/masks, M rows of `row_stride_bytes` each, row r at
     * (uint8_t*)data + (size_t)r * row_stride_bytes. Row start is 64-byte
     * aligned. Interpretation depends on `layout` (see docs/FORMAT.md). */
    void   *data;
    size_t  row_stride_bytes;  /* bytes between consecutive rows in `data`      */

    /* fp32 group scales, widened from the GGUF fp16 "d" at pack time. Length
     * M * (Kp/G), row-major: scales[(size_t)r * (Kp/G) + g]. Pad groups (from
     * the K..Kp region) carry the scale of the GGUF block they came from; their
     * weights are all 0 so the scale value is immaterial. */
    float  *scales;

    /* Optional per-row bias, length M, or NULL. Qwen3 linears have no bias;
     * the API carries it for generality. Kernels add it once per output row. */
    float  *bias;

    /* 1 if the matrix is guaranteed free of +2 (code 3) weights — always true
     * for TBK_LAYOUT_BITPLANE, and set by the packer for CODES layouts whose
     * source contained no code 3. Lets a dispatcher pick the MF path safely. */
    int32_t ternary_only;

    /* Opaque owner pointer for the single backing allocation; do not touch.
     * tbk_mat_free uses it. NULL for a zero-initialized/unpacked handle. */
    void   *_owner;
} tbk_mat;

/* ================================================================== *
 *  Kernel identity + dispatch
 * ================================================================== */

/* Stable kernel identifiers. Names (see tbk_kernel_name) are what
 * TBKERN_FORCE_KERNEL accepts. The AMX slot is reserved and returns
 * TBK_ENOSYS on this hardware. */
typedef enum {
    TBK_KERNEL_AUTO        = 0,  /* runtime __builtin_cpu_supports dispatch     */
    TBK_KERNEL_SCALAR      = 1,  /* portable C fp32 reference (gemv_ref.c)      */
    TBK_KERNEL_SCALAR_I8   = 2,  /* portable C int8 twin of the VNNI algorithm  */
    TBK_KERNEL_MF_AVX512   = 3,  /* bitplane multiplication-free (gemv_avx512_mf)*/
    TBK_KERNEL_VNNI_AVX512 = 4,  /* interleaved-codes vpdpbusd (gemv_avx512_vnni)*/
    TBK_KERNEL_AVX2        = 5,  /* AVX2+FMA fallback (gemv_avx2.c)             */
    TBK_KERNEL_SKIP_AVX512 = 6,  /* zero-skipping AVX-512 experiment (gemv_skip)*/
    TBK_KERNEL_F16_STRAW   = 7,  /* unpack-to-f16 straw-man (gemv_f16_straw)    */
    TBK_KERNEL_AMX         = 8,  /* reserved; returns TBK_ENOSYS here           */
    TBK_KERNEL_COUNT
} tbk_kernel_id;

/* Canonical lowercase name for a kernel id ("scalar", "vnni_avx512", ...),
 * or "unknown". Never NULL. Accepted by TBKERN_FORCE_KERNEL. */
const char   *tbk_kernel_name(tbk_kernel_id id);

/* Parse a kernel name (case-insensitive) to an id, or TBK_KERNEL_COUNT if
 * unrecognized. */
tbk_kernel_id tbk_kernel_from_name(const char *name);

/* Choose the best available kernel for `mat` on this CPU (honors
 * ternary_only / layout / __builtin_cpu_supports). Never returns AUTO. */
tbk_kernel_id tbk_dispatch_select(const tbk_mat *mat);

/* 1 if a kernel is implemented and usable on this CPU for `mat`, else 0. */
int           tbk_kernel_available(tbk_kernel_id id, const tbk_mat *mat);

/* ================================================================== *
 *  GEMV — y[M] = W[M,K] * x[K]  (+ bias)
 *
 *  x is a dense fp32 activation vector of length K (callers pass K, not Kp;
 *  kernels treat columns [K,Kp) as zero-padded). y is fp32 length M.
 *  Returns TBK_OK or a negative error code.
 * ================================================================== */

/* Auto-dispatch entry point. Consults TBKERN_FORCE_KERNEL (env, by kernel
 * name) first; otherwise tbk_dispatch_select. This is the normal entry point. */
int tbk_gemv(const tbk_mat *mat, const float *x, float *y);

/* Forced-kernel entry point. Returns TBK_ENOSYS if `id` is unimplemented or
 * unavailable on this CPU, TBK_EINVAL on a layout/kernel mismatch. */
int tbk_gemv_kernel(tbk_kernel_id id, const tbk_mat *mat,
                    const float *x, float *y);

/* Individual kernel entry points (the tbk_gemv_<variant> contract). Each has
 * the SAME signature. Only tbk_gemv_scalar / tbk_gemv_scalar_i8 exist in
 * Phase 0; the rest are provided by later agents. A kernel may require a
 * specific layout (bitplane for MF, codes for VNNI/AVX2) and returns
 * TBK_EINVAL otherwise. */
int tbk_gemv_scalar     (const tbk_mat *mat, const float *x, float *y); /* fp32 ref, both layouts */
int tbk_gemv_scalar_i8  (const tbk_mat *mat, const float *x, float *y); /* int8 twin, codes layout */
int tbk_gemv_mf_avx512  (const tbk_mat *mat, const float *x, float *y); /* bitplane */
int tbk_gemv_vnni_avx512(const tbk_mat *mat, const float *x, float *y); /* codes    */
int tbk_gemv_avx2       (const tbk_mat *mat, const float *x, float *y); /* codes    */
int tbk_gemv_skip_avx512(const tbk_mat *mat, const float *x, float *y); /* bitplane */
int tbk_gemv_f16_straw  (const tbk_mat *mat, const float *x, float *y); /* codes    */

/* ------------------------------------------------------------------ *
 *  Hoisted-OpenMP-region GEMV variants (ADDITIVE, Phase 6).
 *
 *  A "_region" variant performs ONLY the OpenMP worksharing (the shared
 *  activation quant via `omp single`, the rows via `omp for schedule(static)`)
 *  and REQUIRES the caller to already be executing inside a `#pragma omp
 *  parallel` region. This lets a caller launch ONE thread team and run many
 *  GEMVs inside it — replacing N per-GEMV team launch/join round-trips with 1
 *  launch + N cheap `omp for` barriers. The self-contained tbk_gemv_<variant>
 *  entry points above are unchanged: each still launches its own team (the
 *  self-contained one is a thin `#pragma omp parallel { <variant>_region(...) }`
 *  wrapper). A region variant is BIT-FOR-BIT identical to its self-contained
 *  twin (same static row assignment, same fixed per-row reduction order), and
 *  is also correct (serial) if called with no enclosing team.
 *
 *  The dispatched router returns TBK_ENOSYS for ids without a region variant,
 *  so callers can fall back to the per-call self-contained path. Same
 *  (mat, x, y) contract as the GEMVs.
 * ------------------------------------------------------------------ */
int tbk_gemv_vnni_avx512_region(const tbk_mat *mat, const float *x, float *y); /* codes    */
int tbk_gemv_mf_avx512_region  (const tbk_mat *mat, const float *x, float *y); /* bitplane */
int tbk_gemv_skip_avx512_region(const tbk_mat *mat, const float *x, float *y); /* bitplane */
int tbk_gemv_avx2_region       (const tbk_mat *mat, const float *x, float *y); /* codes    */

/* Dispatched region GEMV (region analogue of tbk_gemv_kernel). MUST be called
 * from inside a `#pragma omp parallel` region. Resolves TBK_KERNEL_AUTO via
 * tbk_dispatch_select. Returns TBK_ENOSYS if `id` has no region variant or is
 * unavailable on this CPU, TBK_EINVAL on null args / layout mismatch. */
int tbk_gemv_kernel_region(tbk_kernel_id id, const tbk_mat *mat,
                           const float *x, float *y);

/* 1 if `id` (or the AUTO-selected kernel for `mat`) has a usable region variant
 * on this CPU for `mat`, else 0. Cheap to call from a serial context to decide
 * whether to enter a hoisted parallel region. */
int tbk_gemv_kernel_region_available(tbk_kernel_id id, const tbk_mat *mat);

/* ================================================================== *
 *  GEMM — Y[n_tokens, M] = X[n_tokens, K] * W^T   (+ bias per output col)
 *
 *  X is row-major n_tokens x K with leading dimension ldx (>= K):
 *      x_token_t = X + (size_t)t * ldx.
 *  Y is row-major n_tokens x M with leading dimension ldy (>= M):
 *      y_token_t = Y + (size_t)t * ldy.
 *  Semantically equivalent to calling tbk_gemv for each token; a real GEMM
 *  kernel reuses the streamed weights across tokens. Returns TBK_OK or error.
 * ================================================================== */
int tbk_gemm(const tbk_mat *mat, const float *X, int n_tokens, int ldx,
             float *Y, int ldy);

int tbk_gemm_vnni_avx512(const tbk_mat *mat, const float *X, int n_tokens,
                         int ldx, float *Y, int ldy);

/* ================================================================== *
 *  Packing (GGUF Q2_0 -> kernel-native)
 * ================================================================== */

/* Repack a row-major array of GGUF Q2_0 blocks into a kernel-native tbk_mat.
 *
 *   gguf_q2_blocks : M rows, each ceil(K/G) contiguous Q2_0 blocks
 *                    (block = fp16 scale + G/4 code bytes). See docs/FORMAT.md.
 *   M, K           : logical output/input dims.
 *   G              : group size, 64 or 128.
 *   layout         : TBK_LAYOUT_BITPLANE (ternary only) or TBK_LAYOUT_CODES.
 *   out            : zero-initialized handle to fill; owns one allocation.
 *
 * Columns [K, Kp) are padded with code 01 (w=0). fp16 scales are widened to
 * fp32. Returns TBK_OK; TBK_EINVAL on bad args; TBK_ENOMEM on OOM;
 * TBK_ENOTTERNARY if layout is BITPLANE and a code-3 (+2) weight is present. */
int tbk_pack_from_q2(const uint8_t *gguf_q2_blocks, int M, int K, int G,
                     tbk_layout_kind layout, tbk_mat *out);

/* Release everything owned by a handle packed by tbk_pack_from_q2 and zero it.
 * Safe on a zeroed handle and on NULL. */
void tbk_mat_free(tbk_mat *mat);

/* Decode one packed row to signed int8 weight values (the plain-C oracle used
 * by tests). Writes Kp values into w_out (caller-allocated, length >= Kp):
 * {-1,0,+1} for bitplane, {-1,0,+1,+2} for codes. Padding columns decode to 0.
 * Returns TBK_OK, or TBK_EINVAL on bad args. */
int tbk_unpack_row(const tbk_mat *mat, int row, int8_t *w_out);

/* ================================================================== *
 *  Activation quantization (VNNI path)
 * ================================================================== */

/* Quantize one fp32 activation vector to per-token absmax int8, and precompute
 * the per-group correction sums used by the VNNI epilogue.
 *
 *   x         : input activations, length K.
 *   K, G      : logical width and group size; Kp = pad_up(K, G).
 *   xq        : output int8, length Kp. Columns [K,Kp) are written as 0.
 *   a_scale   : output scalar = absmax(x[0:K]) / 127  (0 if x is all-zero).
 *   sum_xq_g  : output int32, length Kp/G; sum of xq over each group (the
 *               padded tail group's sum includes the zeroed columns, i.e. they
 *               contribute 0 to both the dot product and this correction).
 *
 * The VNNI dot for a group is  vpdpbusd(u, xq) = dot(w+1, xq) = dot(w,xq)+sum(xq);
 * the epilogue subtracts sum_xq_g to recover dot(w,xq). Returns TBK_OK/error. */
int tbk_quant_act(const float *x, int K, int G,
                  int8_t *xq, float *a_scale, int32_t *sum_xq_g);

/* ================================================================== *
 *  Memory arena  (declarations only in Phase 0; arena.c lands later)
 *
 *  A bump allocator for the big weight repack, with backing-memory modes for
 *  huge pages and mlock. Until arena.c exists, weak fallback definitions in
 *  pack.c satisfy the linker with a plain-malloc implementation, so the whole
 *  library links and runs from Phase 0.
 * ================================================================== */

/* Backing-memory mode (low bits) combinable with the TBK_ARENA_MLOCK flag. */
typedef enum {
    TBK_ARENA_BASE    = 0,   /* plain anonymous pages (malloc/mmap)             */
    TBK_ARENA_THP     = 1,   /* madvise(MADV_HUGEPAGE) transparent huge pages   */
    TBK_ARENA_HUGETLB = 2     /* explicit hugetlb pages (MAP_HUGETLB)           */
} tbk_arena_mode;
#define TBK_ARENA_MODE_MASK  0x00FF
#define TBK_ARENA_MLOCK      0x0100   /* OR into the mode flags to mlock pages   */

typedef struct tbk_arena tbk_arena;  /* opaque                                  */

/* Create an arena able to hand out at least `size` bytes. `mode_flags` is a
 * tbk_arena_mode optionally OR'd with TBK_ARENA_MLOCK. Returns NULL on failure.
 * The Phase-0 weak fallback ignores huge-page/mlock hints and uses malloc. */
tbk_arena *tbk_arena_create(size_t size, int mode_flags);

/* Bump-allocate `size` bytes aligned to `align` (power of two, >=1) from the
 * arena. Returns NULL if exhausted. Memory is owned by the arena. */
void      *tbk_arena_alloc(tbk_arena *a, size_t size, size_t align);

/* Destroy an arena and free all memory it handed out. Safe on NULL. */
void       tbk_arena_free(tbk_arena *a);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* TBKERN_TBKERN_H */
