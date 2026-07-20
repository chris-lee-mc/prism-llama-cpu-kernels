/* gemv_avx2.c — AVX2+FMA fp32 fallback GEMV (Agent C).
 *
 * Entry point: tbk_gemv_avx2 (declared in tbkern.h; the runtime dispatcher in
 * dispatch.c maps TBK_KERNEL_AVX2 -> this symbol by name). This is the SIMD
 * fallback for machines without AVX-512 on the interleaved-codes path.
 *
 * LAYOUT / NUMERIC CLASS (frozen contract):
 *   docs/FORMAT.md §5 and tbkern.h both bind the AVX2 path to TBK_LAYOUT_CODES
 *   ("Layout B — interleaved codes (gemv_avx512_vnni, gemm, avx2)"), and
 *   docs/CONTEXT.md "Correctness policy" classes it with the fp paths
 *   ("fp paths (scalar/MF/AVX2) vs dequant_ref64"). So this kernel consumes the
 *   CODES layout but accumulates in fp32 against the raw activations (NOT the
 *   int8/VNNI numeric path): it is a vectorized twin of tbk_gemv_scalar over the
 *   CODES layout, validated against dequant_ref64 within the calibrated envelope
 *   (tests/test_dispatch.py, tolerance.py). It handles code 3 (+2) exactly
 *   because the decoded weight value is (code-1) in {-1,0,+1,+2}.
 *
 * Algorithm (per docs/FORMAT.md §5):
 *   For each 64-weight subgroup (16 interleaved code bytes), decode the four
 *   2-bit lanes q=0..3 with a 16-bit variable-free shift + per-byte mask:
 *       codes_q = (raw >> 2q) & 0x03      // byte b -> code(base + 16q + b)
 *   the 16-bit word shift bleeds bits across a byte boundary, but the &0x03 per
 *   byte discards them, leaving byte b's true 2-bit code (verified against
 *   pack.c/pack_row_codes and tbk_unpack_row). Each half (8 bytes) maps to 8
 *   CONSECUTIVE activation columns k = base+16q+{0..7} / {8..15}, so:
 *       w = cvt(codes) - 1.0f  (fp32 {-1,0,1,2}) ; acc = fmadd(w, x, acc)
 *   Per scale-group: horizontal-sum the fp32 accumulator and apply the group
 *   scale with one fmaf into the row accumulator (group rescale).
 *
 * fp32 accumulation. OpenMP schedule(static) over 64-row blocks (matches the
 * other kernels); each y[r] is produced by one thread in a fixed group order and
 * a fixed reduction tree, so the result is deterministic across thread counts.
 *
 * ------------------------------------------------------------------------- *
 * Phase 6 (hoisted OpenMP region + prefetch): the row work-share body now lives
 * in tbk_gemv_avx2_region(), which contains ONLY one orphaned OpenMP
 * worksharing construct (`omp for schedule(static)` over the rows) and therefore
 * REQUIRES the caller to already be inside a `#pragma omp parallel` region. The
 * self-contained tbk_gemv_avx2() (unchanged signature/behaviour for tests) is
 * now a thin wrapper: `#pragma omp parallel { region(...) }`. This mirrors
 * gemv_avx512_vnni.c so a caller (the e2e forward loop) can launch ONE thread
 * team and run all per-token GEMVs inside it as cheap `omp for` barriers instead
 * of paying a team launch/join round-trip per GEMV. Determinism is identical:
 * schedule(static) assigns the same 64-row blocks regardless of the team-launch
 * structure, and each y[r] is still produced by a single thread over a fixed
 * group order / reduction tree, so the fp32 result is bit-for-bit the same as
 * the pre-Phase-6 self-contained kernel.
 *
 * The hot loop also issues a _mm_prefetch (T0) into the packed code stream a
 * fixed distance ahead: each row's 16-byte-per-subgroup codes are read strictly
 * sequentially and the kernel is DRAM-bound at per-layer sizes, so hiding the
 * next lines' load latency is the point of Phase 6. Prefetch is a hint only and
 * does not alter the computed bits.
 */
#include "tbkern/tbkern.h"

#include <immintrin.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* Software-prefetch distance into the packed weight-code stream, in bytes.
 * 512 B = 8 cache lines ahead; matches the VNNI kernel's tuned distance. Each
 * row's codes are read strictly sequentially (16 bytes per 64-weight subgroup),
 * so at DRAM-bound per-row streaming the next lines' load latency must be
 * hidden. 8 lines is far enough to cover DRAM latency at this per-row stride yet
 * close enough not to thrash L1/L2 on the row working set. T0 (all levels)
 * because the code stream is consumed immediately, once. */
#define TBK_AVX2_PREFETCH_BYTES 512

/* Deterministic horizontal sum of a 256-bit fp32 vector (fixed reduction tree). */
static inline float hsum256(__m256 v) {
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 hi = _mm256_extractf128_ps(v, 1);
    __m128 s  = _mm_add_ps(lo, hi);
    s = _mm_hadd_ps(s, s);
    s = _mm_hadd_ps(s, s);
    return _mm_cvtss_f32(s);
}

/* Decode 8 natural-order codes held in the low 64 bits of `codes8`, convert to
 * fp32 weight values (code-1), and FMA with 8 activations at `xp`.
 *
 * Uses an UNALIGNED load: the region variant pads the activations into a
 * thread-private stack buffer whose 32-byte alignment is not guaranteed, and an
 * unaligned load of aligned or unaligned data yields bit-identical results, so
 * the self-contained kernel's output is unchanged from the pre-Phase-6 aligned
 * version. */
static inline __m256 fma_half(__m256 acc, __m128i codes8, const float *xp) {
    __m256i ci = _mm256_cvtepu8_epi32(codes8);          /* 8 codes -> int32     */
    __m256  w  = _mm256_cvtepi32_ps(ci);
    w = _mm256_sub_ps(w, _mm256_set1_ps(1.0f));         /* value = code - 1     */
    __m256  x  = _mm256_loadu_ps(xp);                   /* 8 x f32 (unaligned)  */
    return _mm256_fmadd_ps(w, x, acc);
}

/* Accumulate one packed CODES row into a scalar dot-with-scales result. */
static float avx2_row(const uint8_t *restrict rowp, const float *restrict xp,
                      const float *restrict rscales, int NG, int sg_per_grp,
                      float y0) {
    const __m128i m3 = _mm_set1_epi8(3);
    float yr = y0;
    int s = 0;                                  /* running 64-weight subgroup idx */

    for (int g = 0; g < NG; g++) {
        __m256 acc = _mm256_setzero_ps();
        for (int sg = 0; sg < sg_per_grp; sg++, s++) {
            const uint8_t *bp = rowp + (size_t)s * 16;
            const float   *xb = xp   + (size_t)s * 64;
            /* Prefetch the code stream a fixed distance ahead of this row's
             * strictly-sequential read; hint only, does not change the result. */
            _mm_prefetch((const char *)(bp + TBK_AVX2_PREFETCH_BYTES), _MM_HINT_T0);
            __m128i raw = _mm_loadu_si128((const __m128i *)bp);
            /* Unrolled over q so the 16-bit shift amount is an immediate. Each
             * q covers columns base+16q .. base+16q+15 (two 8-lane halves). */
            #define DO_Q(SH, QOFF) do {                                        \
                __m128i codes = _mm_and_si128(_mm_srli_epi16(raw, (SH)), m3);  \
                acc = fma_half(acc, codes, xb + (QOFF));                       \
                acc = fma_half(acc, _mm_srli_si128(codes, 8), xb + (QOFF) + 8);\
            } while (0)
            DO_Q(0,  0);
            DO_Q(2, 16);
            DO_Q(4, 32);
            DO_Q(6, 48);
            #undef DO_Q
        }
        float gsum = hsum256(acc);
        yr = fmaf(rscales[g], gsum, yr);        /* group rescale (one FMA)       */
    }
    return yr;
}

/* ------------------------------------------------------------------ *
 *  Region variant — MUST be called from inside a `#pragma omp parallel`.
 *
 *  Contains only ONE orphaned worksharing construct (`omp for`), so many of
 *  these share a single thread team (one launch/join per token instead of per
 *  GEMV) with exactly one barrier each. Also correct if called with no
 *  enclosing team (runs serially as a team of one). Bit-for-bit identical
 *  output to the self-contained entry point.
 *
 *  Activation padding is done THREAD-PRIVATELY into a stack buffer: [0,K)=x,
 *  [K,Kp)=0. A stack VLA (not malloc) is used so there is NO allocation-failure
 *  path that could return early before the `omp for` implicit barrier and
 *  deadlock the team; the guards below all test shared `mat` state and so are
 *  uniform across the team. Padding is redundant across threads (O(K) each) but
 *  bandwidth-trivial next to the O(M*K) GEMV — the same trade the VNNI region
 *  makes to avoid a shared-buffer barrier. Size: Kp <= 17408 (27B ffn) fp32 =>
 *  ~68 KiB of worker stack, within the default OpenMP worker stack. Reads use
 *  unaligned loads (fma_half), so the VLA needs no special alignment.
 * ------------------------------------------------------------------ */
int tbk_gemv_avx2_region(const tbk_mat *mat, const float *x, float *y) {
    if (!mat || !mat->data || !mat->scales || !x || !y) return TBK_EINVAL;
    if (mat->layout != TBK_LAYOUT_CODES) return TBK_EINVAL;   /* AVX2: codes only */
    if (mat->G != 64 && mat->G != 128) return TBK_EINVAL;
    /* All guards test shared `mat`, so every team member takes the same branch:
     * a uniform early return needs no barrier and cannot deadlock the team. */

    const int M  = mat->M, K = mat->K, Kp = mat->Kp, G = mat->G;
    const int    NG         = Kp / G;
    const int    sg_per_grp = G / 64;               /* 1 (G=64) or 2 (G=128)     */
    const size_t stride     = mat->row_stride_bytes;
    const float *scales     = mat->scales;
    const float *bias       = mat->bias;

    /* Thread-private padded activation buffer: [0,K)=x, [K,Kp)=0. */
    float xp[Kp];                                   /* thread-private scratch     */
    memcpy(xp, x, (size_t)K * sizeof(float));
    if (Kp > K) memset(xp + K, 0, (size_t)(Kp - K) * sizeof(float));

    #pragma omp for schedule(static)
    for (int rb = 0; rb < M; rb += 64) {
        const int r1 = (rb + 64 < M) ? (rb + 64) : M;
        for (int r = rb; r < r1; r++) {
            const uint8_t *rowp = (const uint8_t *)mat->data + (size_t)r * stride;
            const float    y0   = bias ? bias[r] : 0.0f;
            y[r] = avx2_row(rowp, xp, scales + (size_t)r * NG, NG, sg_per_grp, y0);
        }
    }
    /* implicit barrier at end of `omp for`: y is fully written team-wide.
     * (Thread-private scratch is on the stack — nothing to free.) */
    return TBK_OK;
}

/* ------------------------------------------------------------------ *
 *  Self-contained entry point (unchanged contract): launches its own thread
 *  team wrapping the region work-share. This is what the test suite and the
 *  per-call bench path exercise; its output is bit-for-bit the pre-Phase-6
 *  result (same schedule, same fixed reduction order; loadu==load bitwise).
 * ------------------------------------------------------------------ */
int tbk_gemv_avx2(const tbk_mat *mat, const float *x, float *y) {
    if (!mat || !mat->data || !mat->scales || !x || !y) return TBK_EINVAL;
    if (mat->layout != TBK_LAYOUT_CODES) return TBK_EINVAL;   /* AVX2: codes only */
    if (mat->G != 64 && mat->G != 128) return TBK_EINVAL;

    int rc = TBK_OK;
    #pragma omp parallel
    {
        int r = tbk_gemv_avx2_region(mat, x, y);
        /* r is uniform across the team (guards are on shared state). Publish once. */
        #pragma omp single nowait
        rc = r;
    }
    return rc;
}
