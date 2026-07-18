/* gemv_avx512_vnni.c — interleaved-codes int8 GEMV via VPDPBUSD (Agent B).
 *
 * Implements tbk_gemv_vnni_avx512 (header-declared entry point; the runtime
 * dispatcher wires it into the kernel table). Consumes TBK_LAYOUT_CODES weights
 * (docs/FORMAT.md §5) and per-token int8 activations from tbk_quant_act (§7).
 *
 * Algorithm (per docs/FORMAT.md §5, bit-exact mirror of tbk_gemv_scalar_i8):
 *   For each 64-weight subgroup, expand the 16 interleaved code bytes to 64
 *   natural-order u8 codes (u = w+1 in {0..3}) with NO shuffles:
 *       vbroadcasti32x4  (16 bytes -> all four 128-bit lanes)
 *       vpsrlvw          (per-word shifts {0,0,..,2,..,4,..,6})
 *       vpandd 0x03      (per byte)  => byte j == code for k = base + j
 *   Then VPDPBUSD(acc, u_bytes, xq_bytes) = Σ u·xq = dot(w,xq) + Σxq per group.
 *   Epilogue per group: corrected = reduce_i32(acc) - sum_xq_g[g]  (exact int),
 *   then  y_row += (fp32(corrected) * scale_g) * a_scale — accumulating the
 *   RESCALED fp32 group results into the row sum (never raw int32 across a wide
 *   K). Handles code 3 (+2) for free: it is just u = 3 fed to vpdpbusd.
 *
 * Blocking: 4 rows share each xq subgroup load. OpenMP schedule(static) over
 * 64-row blocks. Each y[r] is produced entirely by one thread in fixed group
 * order, so the result is thread-count-independent (deterministic, bitwise).
 *
 * FP-contraction: gcc ignores `#pragma STDC FP_CONTRACT` and would fuse the
 * epilogue's `t*a_scale + acc` into an FMA (single rounding), diverging from the
 * scalar twin which is compiled -ffp-contract=off (two roundings). This file
 * cannot set that flag (CMakeLists is not owned here), so an inline-asm barrier
 * forces `t` to a rounded fp32 register before the add — reproducing the twin's
 * mul,mul,add rounding exactly. Verified bitwise by tests/test_gemv_vnni.py.
 *
 * ------------------------------------------------------------------------- *
 * Phase 6 (hoisted OpenMP region): the row work-share body now lives in
 * tbk_gemv_vnni_avx512_region(), which contains ONLY one orphaned OpenMP
 * worksharing construct (`omp for schedule(static)` over the rows; the
 * activation quant is done thread-privately, no barrier) and therefore REQUIRES
 * the caller to already be inside a `#pragma omp parallel` region. The
 * self-contained tbk_gemv_vnni_avx512() (unchanged signature/behaviour for
 * tests) is now a thin wrapper: `#pragma omp parallel { region(...) }`.
 *
 * ------------------------------------------------------------------------- *
 * Phase 8 (bandwidth / overhead micro-opts). THREE independent, individually
 * toggleable changes are added here, each behind a flag/dispatch so the gate can
 * A/B and attribute each one and revert cleanly. All three preserve the bitwise
 * result (VNNI stays bit-for-bit vs ref_i8 / the scalar_i8 twin):
 *
 *   (1) TUNABLE CODE-STREAM PREFETCH HINT. The packed code stream is read once
 *       per token and pollutes cache for nothing. This is purely a software-
 *       PREFETCH-HINT flip: when enabled, the code-stream prefetch uses
 *       _MM_HINT_NTA (non-temporal, an early-eviction hint) instead of _MM_HINT_T0,
 *       and the prefetch distance is tunable. NOTE: there is NO genuine
 *       non-temporal load here — the actual code reads still go through an
 *       ordinary temporal _mm_loadu_si128 in expand_codes()/expand_codes_vbmi();
 *       PREFETCHNTA is frequently implemented as a normal-hierarchy fill with an
 *       early-eviction hint, not a cache bypass, so the bandwidth/pollution
 *       benefit is a hint, not a guarantee (weaker than a literal "streaming"
 *       load would imply). Prefetch hints NEVER change loaded values, so this is
 *       bitwise-identical to the T0 path it replaces. Toggle: tbk_vnni_set_nt() /
 *       env TBK_VNNI_NT; distance via tbk_vnni_set_prefetch_bytes() /
 *       env TBK_VNNI_PF_BYTES.
 *
 *   (2) EPILOGUE-FUSED ACTIVATION QUANT. tbk_gemv_vnni_avx512_fq() computes y
 *       exactly as the plain GEMV, then folds the NEXT layer's activation quant
 *       into the same call over the just-produced (cache-hot) y, emitting int8
 *       xq + a_scale + sum_xq_g directly, so the producing layer's output is not
 *       re-streamed from DRAM by a separate quant pass. tbk_gemv_vnni_avx512_pre()
 *       is the consumer twin: a GEMV that takes a pre-quantized activation and
 *       SKIPS its internal quant. Both are bit-exact: the fold is literally
 *       tbk_quant_act over the finished y (a two-pass absmax-then-quant cannot be
 *       fused perfectly single-pass and stay bit-exact — see the note on _fq —
 *       so the quant runs as a second sweep over the still-hot output; this IS
 *       bitwise-identical to standalone tbk_quant_act).
 *
 *   (3) GATED VBMI EXPAND. On AVX512-VBMI CPUs the 3-instr srlv/pand expansion
 *       is replaced by a single VBMI bit-field extract (vpmultishiftqb) that
 *       produces the identical low-2-bit u codes. Dispatched by
 *       __builtin_cpu_supports("avx512vbmi") (this box lacks VBMI, so the path
 *       is compiled — via a per-function target attribute so the TU's global
 *       flags are untouched — but never executed here; correctness is proven by
 *       tests/test_stream_epilogue.py against the 3-instr expansion). NOTE:
 *       vpermb is a pure byte permute and cannot extract sub-byte bit fields, so
 *       the single-op VBMI expander uses vpmultishiftqb (the VBMI variable
 *       bit-field extract), which yields the same u = w+1 codes after the
 *       & 0x03 mask.
 */
#include "tbkern/tbkern.h"

#include <immintrin.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

/* Additive, ADDITIVE-ONLY local prototypes for the Phase-8 entry points. The
 * frozen contract header include/tbkern/tbkern.h is orchestrator-owned
 * (docs/CONTEXT.md Rule 1), so — exactly as src/fused.c does for the Phase-7
 * fused API — the new entry points are declared here (owned by this TU) and any
 * other C consumer (bench/bench_e2e.c, the ctypes test) carries its own extern
 * prototype. The frozen header is left byte-for-byte unchanged. */
int  tbk_gemv_vnni_avx512_region(const tbk_mat *mat, const float *x, float *y);
int  tbk_gemv_vnni_avx512(const tbk_mat *mat, const float *x, float *y);
int  tbk_gemv_vnni_avx512_pre(const tbk_mat *mat, const int8_t *xq,
                              float a_scale, const int32_t *sum_xq_g, float *y);
int  tbk_gemv_vnni_avx512_pre_region(const tbk_mat *mat, const int8_t *xq,
                                     float a_scale, const int32_t *sum_xq_g,
                                     float *y);
int  tbk_gemv_vnni_avx512_fq(const tbk_mat *mat, const float *x, float *y,
                             int Gq, int8_t *xq_out, float *a_scale_out,
                             int32_t *sum_xq_g_out);
void tbk_vnni_set_nt(int on);
void tbk_vnni_set_prefetch_bytes(int bytes);
void tbk_vnni_init_flags(void);
int  tbk_vnni_get_nt(void);
int  tbk_vnni_get_prefetch_bytes(void);
int  tbk_vnni_using_vbmi(void);
/* test-only introspection (used by tests/test_stream_epilogue.py) */
void tbk_vnni_expand_default(const uint8_t *in16, uint8_t *out64);
int  tbk_vnni_vbmi_control_byte(int p);

/* Software-prefetch distance into the packed weight-code stream, in bytes.
 * 512 B = 8 cache lines ahead (see the Phase-6 note). Runtime-tunable via
 * tbk_vnni_set_prefetch_bytes() / env TBK_VNNI_PF_BYTES. */
#define TBK_VNNI_PREFETCH_BYTES 512

/* ------------------------------------------------------------------ *
 *  Phase-8 runtime flags (change 1 + change 3 dispatch)
 * ------------------------------------------------------------------ */
static int g_nt       = -1;   /* 1 => NTA prefetch; 0 => T0 (default) */
static int g_pfd      = -1;   /* prefetch distance in bytes           */
static int g_use_vbmi = -1;   /* 1 => VBMI vpmultishiftqb expand       */

static void vnni_init_flags(void) {
    if (g_nt < 0) {
        const char *e = getenv("TBK_VNNI_NT");
        g_nt = (e && (e[0] == '1' || e[0] == 'y' || e[0] == 'Y')) ? 1 : 0;
    }
    if (g_pfd < 0) {
        const char *e = getenv("TBK_VNNI_PF_BYTES");
        int v = e ? atoi(e) : TBK_VNNI_PREFETCH_BYTES;
        g_pfd = (v < 0) ? 0 : v;
    }
    if (g_use_vbmi < 0) {
        int s = __builtin_cpu_supports("avx512vbmi");
        const char *e = getenv("TBK_VNNI_VBMI");
        if (e && e[0] == '0') s = 0;   /* env can force OFF (never ON: no HW) */
        g_use_vbmi = s ? 1 : 0;
    }
}

/* Resolve ALL dispatch flags once, single-threaded, at library load — BEFORE
 * any OpenMP team can exist. This closes a data race on the lazy init above:
 * the _region() entry points call vnni_init_flags() as their first statement,
 * and in hoisted mode (bench_e2e --region=hoisted -> forward_hoisted) that first
 * call happens on ALL team threads concurrently, with no prior single-threaded
 * VNNI call to prime the globals. g_nt/g_pfd are additionally primed by
 * bench_e2e via the setters, but g_use_vbmi has no setter, so absent this
 * constructor every thread would race on its read-check-then-write. A ctor runs
 * before main() spawns any thread, so by the time any region body executes, all
 * three globals are >= 0 and vnni_init_flags() degenerates to pure (race-free)
 * reads. Setters called later (single-threaded in main) can still override g_nt/
 * g_pfd. Also declared as a resolver entry point so a hoisted/region caller can
 * explicitly prime the flags before spawning its team (see bench_e2e.c), which
 * makes the requirement visible at the call site rather than relying on the ctor
 * alone. */
void tbk_vnni_init_flags(void) { vnni_init_flags(); }
__attribute__((constructor)) static void vnni_flags_ctor(void) { vnni_init_flags(); }

void tbk_vnni_set_nt(int on)              { g_nt  = on ? 1 : 0; }
void tbk_vnni_set_prefetch_bytes(int b)   { g_pfd = (b < 0) ? 0 : b; }
int  tbk_vnni_get_nt(void)                { vnni_init_flags(); return g_nt; }
int  tbk_vnni_get_prefetch_bytes(void)    { vnni_init_flags(); return g_pfd; }
int  tbk_vnni_using_vbmi(void)            { vnni_init_flags(); return g_use_vbmi; }

/* Branch the (compile-time-immediate) prefetch hint on the NT toggle. The
 * branch is loop-invariant (nt is hoisted to a register) and perfectly
 * predicted; prefetch changes no architectural state that affects the result. */
static inline void pf_code(const char *p, int nt) {
    if (nt) _mm_prefetch(p, _MM_HINT_NTA);
    else    _mm_prefetch(p, _MM_HINT_T0);
}

/* ------------------------------------------------------------------ *
 *  Code expansion (change 3: default vs VBMI, bit-identical after &3)
 * ------------------------------------------------------------------ */

/* Default (no VBMI): expand 16 interleaved code bytes to 64 natural-order u8
 * codes. byte p == (raw[p%16] >> (2*(p/16))) & 3 (docs/FORMAT.md §5). */
static inline __m512i expand_codes(const uint8_t *ptr,
                                   const __m512i shifts, const __m512i mask3) {
    __m128i raw = _mm_loadu_si128((const __m128i *)ptr);
    __m512i bc  = _mm512_broadcast_i32x4(raw);
    __m512i sh  = _mm512_srlv_epi16(bc, shifts);   /* per-word variable shift  */
    return _mm512_and_si512(sh, mask3);            /* keep low 2 bits per byte */
}

/* VBMI control byte for output position p: vpmultishiftqb selects 8 contiguous
 * bits of the source qword starting at (control & 63). With the 16 code bytes
 * broadcast across the 4 lanes, output byte p reads the qword that holds
 * raw[p%16] (byte (p%8) of that qword, bit base 8*(p%8)); adding 2*(p/16) lands
 * the window low-2-bits on the wanted 2-bit code. After & 3 this equals the
 * default expansion byte-for-byte. Exposed for the correctness proof. */
int tbk_vnni_vbmi_control_byte(int p) { return 8 * (p % 8) + 2 * (p / 16); }

/* VBMI (single bit-extract) expand. Compiled with a per-function target
 * attribute so the TU's global -mavx512vnni flags are untouched; only reached
 * when __builtin_cpu_supports("avx512vbmi") (never on this box). */
__attribute__((target("avx512f,avx512bw,avx512vl,avx512dq,avx512vbmi")))
static inline __m512i expand_codes_vbmi(const uint8_t *ptr,
                                        const __m512i control,
                                        const __m512i mask3) {
    __m128i raw = _mm_loadu_si128((const __m128i *)ptr);
    __m512i bc  = _mm512_broadcast_i32x4(raw);
    __m512i ms  = _mm512_multishift_epi64_epi8(control, bc);  /* vpmultishiftqb */
    return _mm512_and_si512(ms, mask3);
}

/* Rescale one group's corrected int32 dot into the row accumulator, matching
 * the scalar twin's two-rounding sequence (mul, mul, add) exactly. */
static inline float epilogue_add(float acc, int32_t corrected,
                                 float scale_g, float a_scale) {
    float t = (float)corrected * scale_g;   /* round 1 */
    t = t * a_scale;                         /* round 2 */
    __asm__("" : "+x"(t));                    /* barrier: block FMA fusion */
    return acc + t;
}

/* ------------------------------------------------------------------ *
 *  Row work-share body (shared by default + VBMI variants via macro).
 *
 *  Parameterized ONLY by the per-subgroup code expansion EXP(PTR); the loop
 *  structure — schedule(static) 64-row blocks, 4-row inner blocking, fixed
 *  per-row group order, epilogue — is IDENTICAL across ISA variants, so the two
 *  instantiations are provably bit-for-bit each other and their scalar twin.
 *  Uses caller-provided int8 activations (xq/sum_xq_g/a_scale) — the internal
 *  quant is done by the entry points, not here.
 * ------------------------------------------------------------------ */
#define TBK_VNNI_ROWS_BODY(EXP)                                                  \
    _Pragma("omp for schedule(static)")                                         \
    for (int rb = 0; rb < M; rb += 64) {                                        \
        int rend = rb + 64; if (rend > M) rend = M;                            \
        int r = rb;                                                             \
        for (; r + 4 <= rend; r += 4) {                                         \
            const uint8_t *rp0 = data + (size_t)(r + 0) * stride;              \
            const uint8_t *rp1 = data + (size_t)(r + 1) * stride;              \
            const uint8_t *rp2 = data + (size_t)(r + 2) * stride;              \
            const uint8_t *rp3 = data + (size_t)(r + 3) * stride;              \
            float acc0 = bias ? bias[r + 0] : 0.0f;                            \
            float acc1 = bias ? bias[r + 1] : 0.0f;                            \
            float acc2 = bias ? bias[r + 2] : 0.0f;                            \
            float acc3 = bias ? bias[r + 3] : 0.0f;                            \
            for (int g = 0; g < NG; g++) {                                     \
                __m512i a0 = _mm512_setzero_si512();                           \
                __m512i a1 = _mm512_setzero_si512();                           \
                __m512i a2 = _mm512_setzero_si512();                           \
                __m512i a3 = _mm512_setzero_si512();                           \
                const int s0 = g * sg_per_grp;                                 \
                for (int sg = 0; sg < sg_per_grp; sg++) {                      \
                    const int s = s0 + sg;                                     \
                    const size_t off = (size_t)s * 16;                         \
                    pf_code((const char *)(rp0 + off + pfd), nt);              \
                    pf_code((const char *)(rp1 + off + pfd), nt);              \
                    pf_code((const char *)(rp2 + off + pfd), nt);              \
                    pf_code((const char *)(rp3 + off + pfd), nt);              \
                    const __m512i xqv = _mm512_loadu_si512(xq + (size_t)s*64); \
                    a0 = _mm512_dpbusd_epi32(a0, EXP(rp0 + off), xqv);         \
                    a1 = _mm512_dpbusd_epi32(a1, EXP(rp1 + off), xqv);         \
                    a2 = _mm512_dpbusd_epi32(a2, EXP(rp2 + off), xqv);         \
                    a3 = _mm512_dpbusd_epi32(a3, EXP(rp3 + off), xqv);         \
                }                                                              \
                const int32_t s_g = sum_xq_g[g];                              \
                int32_t c0 = _mm512_reduce_add_epi32(a0) - s_g;              \
                int32_t c1 = _mm512_reduce_add_epi32(a1) - s_g;              \
                int32_t c2 = _mm512_reduce_add_epi32(a2) - s_g;              \
                int32_t c3 = _mm512_reduce_add_epi32(a3) - s_g;              \
                acc0 = epilogue_add(acc0, c0, scales[(size_t)(r+0)*NG+g], a_scale); \
                acc1 = epilogue_add(acc1, c1, scales[(size_t)(r+1)*NG+g], a_scale); \
                acc2 = epilogue_add(acc2, c2, scales[(size_t)(r+2)*NG+g], a_scale); \
                acc3 = epilogue_add(acc3, c3, scales[(size_t)(r+3)*NG+g], a_scale); \
            }                                                                  \
            y[r + 0] = acc0; y[r + 1] = acc1; y[r + 2] = acc2; y[r + 3] = acc3; \
        }                                                                      \
        for (; r < rend; r++) {                                                \
            const uint8_t *rp = data + (size_t)r * stride;                     \
            float acc = bias ? bias[r] : 0.0f;                                 \
            for (int g = 0; g < NG; g++) {                                     \
                __m512i a = _mm512_setzero_si512();                            \
                const int s0 = g * sg_per_grp;                                 \
                for (int sg = 0; sg < sg_per_grp; sg++) {                      \
                    const int s = s0 + sg;                                     \
                    const size_t off = (size_t)s * 16;                         \
                    pf_code((const char *)(rp + off + pfd), nt);              \
                    const __m512i xqv = _mm512_loadu_si512(xq + (size_t)s*64); \
                    a = _mm512_dpbusd_epi32(a, EXP(rp + off), xqv);           \
                }                                                              \
                int32_t c = _mm512_reduce_add_epi32(a) - sum_xq_g[g];         \
                acc = epilogue_add(acc, c, scales[(size_t)r*NG+g], a_scale);   \
            }                                                                  \
            y[r] = acc;                                                        \
        }                                                                      \
    }

/* Default (srlv/pand) row work-share. */
static void vnni_rows_default(int M, const uint8_t *data, size_t stride,
                              const float *scales, const float *bias, int NG,
                              int sg_per_grp, const int8_t *xq, float a_scale,
                              const int32_t *sum_xq_g, float *y, int nt, int pfd) {
    uint16_t shbuf[32];
    for (int w = 0; w < 32; w++) shbuf[w] = (uint16_t)(2 * (w / 8));
    const __m512i shifts = _mm512_loadu_si512(shbuf);
    const __m512i mask3  = _mm512_set1_epi8(3);
    #define EXP_DEFAULT(PTR) expand_codes((PTR), shifts, mask3)
    TBK_VNNI_ROWS_BODY(EXP_DEFAULT)
    #undef EXP_DEFAULT
}

/* VBMI (vpmultishiftqb) row work-share. Per-function target attribute keeps the
 * TU's global flags at -mavx512vnni; only entered under VBMI dispatch. */
__attribute__((target("avx512f,avx512bw,avx512vl,avx512dq,avx512vnni,avx512vbmi")))
static void vnni_rows_vbmi(int M, const uint8_t *data, size_t stride,
                           const float *scales, const float *bias, int NG,
                           int sg_per_grp, const int8_t *xq, float a_scale,
                           const int32_t *sum_xq_g, float *y, int nt, int pfd) {
    uint8_t ctrlbuf[64];
    for (int p = 0; p < 64; p++) ctrlbuf[p] = (uint8_t)(8 * (p % 8) + 2 * (p / 16));
    const __m512i control = _mm512_loadu_si512(ctrlbuf);
    const __m512i mask3   = _mm512_set1_epi8(3);
    #define EXP_VBMI(PTR) expand_codes_vbmi((PTR), control, mask3)
    TBK_VNNI_ROWS_BODY(EXP_VBMI)
    #undef EXP_VBMI
}

/* Dispatch the row work-share by the cached VBMI flag. All team members read
 * the same g_use_vbmi, so they call the SAME variant and encounter the SAME
 * single `omp for` — no barrier-count divergence. */
static inline void vnni_rows(int M, const uint8_t *data, size_t stride,
                             const float *scales, const float *bias, int NG,
                             int sg_per_grp, const int8_t *xq, float a_scale,
                             const int32_t *sum_xq_g, float *y) {
    const int nt = g_nt, pfd = g_pfd;
    if (g_use_vbmi)
        vnni_rows_vbmi(M, data, stride, scales, bias, NG, sg_per_grp,
                       xq, a_scale, sum_xq_g, y, nt, pfd);
    else
        vnni_rows_default(M, data, stride, scales, bias, NG, sg_per_grp,
                          xq, a_scale, sum_xq_g, y, nt, pfd);
}

/* ------------------------------------------------------------------ *
 *  Region variant — MUST be called from inside a `#pragma omp parallel`.
 * ------------------------------------------------------------------ */
int tbk_gemv_vnni_avx512_region(const tbk_mat *mat, const float *x, float *y) {
    if (!mat || !mat->data || !x || !y) return TBK_EINVAL;
    if (mat->layout != TBK_LAYOUT_CODES) return TBK_EINVAL;

    vnni_init_flags();
    const int M  = mat->M, K = mat->K, Kp = mat->Kp, G = mat->G;
    const int NG = Kp / G;
    const int sg_per_grp = G / 64;

    const uint8_t *data   = (const uint8_t *)mat->data;
    const size_t   stride = mat->row_stride_bytes;
    const float   *scales = mat->scales;
    const float   *bias   = mat->bias;

    /* Per-token activation quant, thread-privately (see the Phase-6 note): a
     * pure function of (x,K,G), so all threads compute identical xq/sum/a_scale.
     * Stack VLAs => no allocation-failure path before the `omp for` barrier. */
    int8_t  xq[Kp];
    int32_t sum_xq_g[NG];
    float   a_scale = 0.0f;
    int     rc = tbk_quant_act(x, K, G, xq, &a_scale, sum_xq_g);
    if (rc != TBK_OK) return rc;

    vnni_rows(M, data, stride, scales, bias, NG, sg_per_grp,
              xq, a_scale, sum_xq_g, y);
    return TBK_OK;
}

/* ------------------------------------------------------------------ *
 *  Self-contained entry point (unchanged contract).
 * ------------------------------------------------------------------ */
int tbk_gemv_vnni_avx512(const tbk_mat *mat, const float *x, float *y) {
    if (!mat || !mat->data || !x || !y) return TBK_EINVAL;
    if (mat->layout != TBK_LAYOUT_CODES) return TBK_EINVAL;

    vnni_init_flags();   /* resolve dispatch flags before spawning the team */
    int rc = TBK_OK;
    #pragma omp parallel
    {
        int r = tbk_gemv_vnni_avx512_region(mat, x, y);
        #pragma omp single nowait
        rc = r;
    }
    return rc;
}

/* ================================================================== *
 *  Phase-8 change 2: epilogue-fused activation quant
 * ================================================================== */

/* Consumer twin — GEMV that takes a PRE-QUANTIZED activation and SKIPS its
 * internal quant. Bit-exact to tbk_gemv_vnni_avx512(mat, x, y) whenever
 * (xq, a_scale, sum_xq_g) == tbk_quant_act(x, mat->K, mat->G, ...). Region
 * form: MUST be called from inside a `#pragma omp parallel`. */
int tbk_gemv_vnni_avx512_pre_region(const tbk_mat *mat, const int8_t *xq,
                                    float a_scale, const int32_t *sum_xq_g,
                                    float *y) {
    if (!mat || !mat->data || !xq || !sum_xq_g || !y) return TBK_EINVAL;
    if (mat->layout != TBK_LAYOUT_CODES) return TBK_EINVAL;
    vnni_init_flags();
    const int M  = mat->M, Kp = mat->Kp, G = mat->G;
    const int NG = Kp / G;
    const int sg_per_grp = G / 64;
    const uint8_t *data   = (const uint8_t *)mat->data;
    const size_t   stride = mat->row_stride_bytes;
    const float   *scales = mat->scales;
    const float   *bias   = mat->bias;
    vnni_rows(M, data, stride, scales, bias, NG, sg_per_grp,
              xq, a_scale, sum_xq_g, y);
    return TBK_OK;
}

int tbk_gemv_vnni_avx512_pre(const tbk_mat *mat, const int8_t *xq,
                             float a_scale, const int32_t *sum_xq_g, float *y) {
    if (!mat || !mat->data || !xq || !sum_xq_g || !y) return TBK_EINVAL;
    if (mat->layout != TBK_LAYOUT_CODES) return TBK_EINVAL;
    vnni_init_flags();
    int rc = TBK_OK;
    #pragma omp parallel
    {
        int r = tbk_gemv_vnni_avx512_pre_region(mat, xq, a_scale, sum_xq_g, y);
        #pragma omp single nowait
        rc = r;
    }
    return rc;
}

/* Producer with fused epilogue quant. Computes y exactly as the plain GEMV,
 * then folds the NEXT layer's activation quant into the SAME call over the
 * just-produced (cache-hot) y — emitting int8 xq_out + a_scale_out + sum_xq_g_out
 * for a downstream layer whose input width is this layer's M, group Gq.
 *
 * BIT-EXACTNESS / THE TWO-PASS TRADEOFF. A per-token absmax quant is inherently
 * two-pass (absmax over ALL of y must precede any quant), so it cannot be fused
 * perfectly single-pass into the row writeback and stay bit-exact. We therefore
 * fold it as a SECOND SWEEP over the finished y while it is still hot (never
 * evicted / re-streamed from DRAM), which is byte-for-byte tbk_quant_act(y, M,
 * Gq, ...). The win is locality (no separate DRAM pass over the fp32 output),
 * not fewer arithmetic ops — consistent with docs/RESULTS.md §6 (per-layer fixed
 * overhead shrinks as M*K grows). Gated behind the bench --fusedquant flag. */
int tbk_gemv_vnni_avx512_fq(const tbk_mat *mat, const float *x, float *y,
                            int Gq, int8_t *xq_out, float *a_scale_out,
                            int32_t *sum_xq_g_out) {
    if (!mat || !mat->data || !x || !y) return TBK_EINVAL;
    if (!xq_out || !a_scale_out || !sum_xq_g_out) return TBK_EINVAL;
    if (mat->layout != TBK_LAYOUT_CODES) return TBK_EINVAL;
    if (Gq != 64 && Gq != 128) return TBK_EINVAL;

    int rc = tbk_gemv_vnni_avx512(mat, x, y);
    if (rc != TBK_OK) return rc;
    /* Fold: quantize the hot output y for the next layer (K == this layer's M). */
    return tbk_quant_act(y, mat->M, Gq, xq_out, a_scale_out, sum_xq_g_out);
}

/* ------------------------------------------------------------------ *
 *  Test-only introspection: run the REAL default AVX-512 expansion so the
 *  correctness test can compare it against a scalar model of the VBMI
 *  (vpmultishiftqb) expansion. This box lacks VBMI, so the VBMI intrinsic is
 *  never executed here — the proof is a scalar model keyed by the actual C
 *  control constant (tbk_vnni_vbmi_control_byte). See tests/test_stream_epilogue.py.
 * ------------------------------------------------------------------ */
void tbk_vnni_expand_default(const uint8_t *in16, uint8_t *out64) {
    uint16_t shbuf[32];
    for (int w = 0; w < 32; w++) shbuf[w] = (uint16_t)(2 * (w / 8));
    const __m512i shifts = _mm512_loadu_si512(shbuf);
    const __m512i mask3  = _mm512_set1_epi8(3);
    __m512i v = expand_codes(in16, shifts, mask3);
    _mm512_storeu_si512(out64, v);
}
