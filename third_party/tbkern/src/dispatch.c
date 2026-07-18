/* dispatch.c — real runtime kernel dispatch (Agent C).
 *
 * Replaces the Phase-0 dispatch_stub.c (deleted). Provides STRONG definitions of
 * the header-declared dispatch surface:
 *     tbk_kernel_name / tbk_kernel_from_name / tbk_kernel_available
 *     tbk_dispatch_select / tbk_gemv_kernel / tbk_gemv
 * and a WEAK generic tbk_gemm (a dedicated GEMM translation unit may override).
 *
 * Kernel table: a per-id function-pointer lookup. The optional SIMD kernels are
 * referenced WEAKLY (#pragma weak), so a build that has not compiled/linked one
 * of them resolves its address to NULL rather than failing to link; the table
 * reports such a kernel unavailable. scalar / scalar_i8 live in gemv_ref.c and
 * are always present.
 *
 * CPU gating: __builtin_cpu_supports feature flags are probed ONCE (constructor
 * + pthread_once lazy guard, thread-safe). A kernel is "available" only if its
 * symbol is linked AND this CPU supports its ISA AND (when a mat is supplied)
 * its required layout matches. The reserved AMX slot has no entry point and is
 * always TBK_ENOSYS.
 *
 * Auto-dispatch order (docs/CONTEXT.md: avx512vnni -> avx512f -> avx2 -> scalar),
 * resolved per layout since a packed matrix commits to one layout:
 *   BITPLANE : mf_avx512 (AVX-512F)            else scalar (fp)
 *   CODES    : vnni_avx512 (AVX-512VNNI) -> avx2 (AVX2+FMA) else scalar (fp)
 * The CODES fallback is the fp scalar path (not scalar_i8): the fp paths agree
 * with the fp64 oracle within the calibrated envelope, keeping AUTO numerically
 * consistent with the fp SIMD kernels.
 *
 * TBKERN_FORCE_KERNEL=<name> (env) overrides selection in tbk_gemv: a recognized
 * name routes straight to tbk_gemv_kernel (which returns TBK_ENOSYS if that
 * kernel is unimplemented/unlinked/CPU-unsupported, or lets the kernel return
 * TBK_EINVAL on a layout mismatch); an unrecognized name falls back to AUTO.
 */
#include "tbkern/tbkern.h"

#include <stdlib.h>
#include <string.h>
#include <strings.h>   /* strcasecmp */
#include <pthread.h>

/* Optional SIMD kernels: weak references so absent ones resolve to NULL. */
#pragma weak tbk_gemv_mf_avx512
#pragma weak tbk_gemv_vnni_avx512
#pragma weak tbk_gemv_avx2
#pragma weak tbk_gemv_skip_avx512
#pragma weak tbk_gemv_f16_straw

typedef int (*tbk_gemv_fn)(const tbk_mat *, const float *, float *);

/* ------------------------------------------------------------------ *
 *  CPU feature probe (once, thread-safe)
 * ------------------------------------------------------------------ */
static int feat_avx2, feat_fma, feat_avx512f, feat_avx512vnni, feat_f16c;
static pthread_once_t g_once = PTHREAD_ONCE_INIT;

static void detect_features(void) {
    __builtin_cpu_init();
    feat_avx2       = __builtin_cpu_supports("avx2")       ? 1 : 0;
    feat_fma        = __builtin_cpu_supports("fma")        ? 1 : 0;
    feat_avx512f    = __builtin_cpu_supports("avx512f")    ? 1 : 0;
    feat_avx512vnni = __builtin_cpu_supports("avx512vnni") ? 1 : 0;
    feat_f16c       = __builtin_cpu_supports("f16c")       ? 1 : 0;
}

static inline void ensure_init(void) { pthread_once(&g_once, detect_features); }

/* Run the probe eagerly at load; ensure_init() below still guards every use. */
__attribute__((constructor))
static void dispatch_ctor(void) { ensure_init(); }

/* ------------------------------------------------------------------ *
 *  Kernel table / capability helpers
 * ------------------------------------------------------------------ */
static tbk_gemv_fn kernel_fn(tbk_kernel_id id) {
    switch (id) {
        case TBK_KERNEL_SCALAR:      return tbk_gemv_scalar;
        case TBK_KERNEL_SCALAR_I8:   return tbk_gemv_scalar_i8;
        case TBK_KERNEL_MF_AVX512:   return tbk_gemv_mf_avx512;    /* weak */
        case TBK_KERNEL_VNNI_AVX512: return tbk_gemv_vnni_avx512;  /* weak */
        case TBK_KERNEL_AVX2:        return tbk_gemv_avx2;         /* weak */
        case TBK_KERNEL_SKIP_AVX512: return tbk_gemv_skip_avx512;  /* weak */
        case TBK_KERNEL_F16_STRAW:   return tbk_gemv_f16_straw;    /* weak */
        default:                     return NULL;  /* AUTO / AMX / COUNT / oob */
    }
}

/* Does this CPU support the kernel's ISA? (independent of linkage/layout) */
static int cpu_ok(tbk_kernel_id id) {
    ensure_init();
    switch (id) {
        case TBK_KERNEL_SCALAR:
        case TBK_KERNEL_SCALAR_I8:   return 1;
        case TBK_KERNEL_MF_AVX512:
        case TBK_KERNEL_SKIP_AVX512: return feat_avx512f;
        case TBK_KERNEL_VNNI_AVX512: return feat_avx512f && feat_avx512vnni;
        case TBK_KERNEL_AVX2:        return feat_avx2 && feat_fma;
        case TBK_KERNEL_F16_STRAW:   return feat_avx2 && feat_f16c;
        default:                     return 0;   /* AMX (reserved) etc. */
    }
}

/* Required packed layout, or -1 if the kernel accepts either. */
static int kernel_layout_req(tbk_kernel_id id) {
    switch (id) {
        case TBK_KERNEL_MF_AVX512:
        case TBK_KERNEL_SKIP_AVX512: return TBK_LAYOUT_BITPLANE;
        case TBK_KERNEL_SCALAR_I8:
        case TBK_KERNEL_VNNI_AVX512:
        case TBK_KERNEL_AVX2:
        case TBK_KERNEL_F16_STRAW:   return TBK_LAYOUT_CODES;
        default:                     return -1;  /* SCALAR: both layouts */
    }
}

int tbk_kernel_available(tbk_kernel_id id, const tbk_mat *mat) {
    if (!kernel_fn(id)) return 0;      /* unimplemented or not linked           */
    if (!cpu_ok(id))    return 0;      /* ISA unsupported on this CPU            */
    if (mat) {
        const int req = kernel_layout_req(id);
        if (req >= 0 && (int)mat->layout != req) return 0;
    }
    return 1;
}

/* ------------------------------------------------------------------ *
 *  Names
 * ------------------------------------------------------------------ */
const char *tbk_kernel_name(tbk_kernel_id id) {
    switch (id) {
        case TBK_KERNEL_AUTO:        return "auto";
        case TBK_KERNEL_SCALAR:      return "scalar";
        case TBK_KERNEL_SCALAR_I8:   return "scalar_i8";
        case TBK_KERNEL_MF_AVX512:   return "mf_avx512";
        case TBK_KERNEL_VNNI_AVX512: return "vnni_avx512";
        case TBK_KERNEL_AVX2:        return "avx2";
        case TBK_KERNEL_SKIP_AVX512: return "skip_avx512";
        case TBK_KERNEL_F16_STRAW:   return "f16_straw";
        case TBK_KERNEL_AMX:         return "amx";
        default:                     return "unknown";
    }
}

tbk_kernel_id tbk_kernel_from_name(const char *name) {
    if (!name) return TBK_KERNEL_COUNT;
    for (int id = 0; id < TBK_KERNEL_COUNT; id++) {
        if (strcasecmp(name, tbk_kernel_name((tbk_kernel_id)id)) == 0)
            return (tbk_kernel_id)id;
    }
    return TBK_KERNEL_COUNT;
}

/* ------------------------------------------------------------------ *
 *  Selection
 * ------------------------------------------------------------------ */
tbk_kernel_id tbk_dispatch_select(const tbk_mat *mat) {
    if (!mat) return TBK_KERNEL_SCALAR;

    if (mat->layout == TBK_LAYOUT_BITPLANE) {
        if (tbk_kernel_available(TBK_KERNEL_MF_AVX512, mat))
            return TBK_KERNEL_MF_AVX512;
        return TBK_KERNEL_SCALAR;
    }

    /* TBK_LAYOUT_CODES: vnni -> avx2 -> scalar (fp fallback). */
    if (tbk_kernel_available(TBK_KERNEL_VNNI_AVX512, mat))
        return TBK_KERNEL_VNNI_AVX512;
    if (tbk_kernel_available(TBK_KERNEL_AVX2, mat))
        return TBK_KERNEL_AVX2;
    return TBK_KERNEL_SCALAR;
}

/* ------------------------------------------------------------------ *
 *  GEMV entry points
 * ------------------------------------------------------------------ */
int tbk_gemv_kernel(tbk_kernel_id id, const tbk_mat *mat,
                    const float *x, float *y) {
    if (!mat || !x || !y) return TBK_EINVAL;
    if (id == TBK_KERNEL_AUTO) id = tbk_dispatch_select(mat);
    tbk_gemv_fn fn = kernel_fn(id);
    if (!fn || !cpu_ok(id)) return TBK_ENOSYS;  /* unlinked/unsupported/AMX slot */
    return fn(mat, x, y);                        /* kernel self-checks the layout */
}

int tbk_gemv(const tbk_mat *mat, const float *x, float *y) {
    if (!mat || !x || !y) return TBK_EINVAL;
    const char *forced = getenv("TBKERN_FORCE_KERNEL");
    if (forced && *forced) {
        tbk_kernel_id id = tbk_kernel_from_name(forced);
        if (id != TBK_KERNEL_COUNT)             /* recognized: honor it (may err) */
            return tbk_gemv_kernel(id, mat, x, y);
        /* unrecognized name: silently fall back to auto dispatch. */
    }
    return tbk_gemv_kernel(tbk_dispatch_select(mat), mat, x, y);
}

/* ------------------------------------------------------------------ *
 *  GEMM — generic token loop (WEAK: a real GEMM kernel TU may override)
 * ------------------------------------------------------------------ */
__attribute__((weak))
int tbk_gemm(const tbk_mat *mat, const float *X, int n_tokens, int ldx,
             float *Y, int ldy) {
    if (!mat || !X || !Y || n_tokens < 0) return TBK_EINVAL;
    if (ldx < mat->K || ldy < mat->M) return TBK_EINVAL;
    for (int t = 0; t < n_tokens; t++) {
        int rc = tbk_gemv(mat, X + (size_t)t * ldx, Y + (size_t)t * ldy);
        if (rc != TBK_OK) return rc;
    }
    return TBK_OK;
}
