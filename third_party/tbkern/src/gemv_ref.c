/* gemv_ref.c — portable scalar reference kernels.
 *
 * Owner: Phase-0 contracts agent.
 *   tbk_gemv_scalar    : fp32 reference over BOTH layouts (group-rescaled,
 *                        optional bias). Ground-truth-adjacent fp path; tested
 *                        against the Python dequant_ref64 oracle within the
 *                        calibrated envelope.
 *   tbk_gemv_scalar_i8 : integer twin of the VNNI algorithm — the C-side mirror
 *                        of the Python ref_i8 oracle (must agree bit for bit).
 *   tbk_quant_act      : WEAK per-token absmax int8 activation quant + per-group
 *                        correction sums (see docs/FORMAT.md §7). A later
 *                        quant_act.c provides a strong (SIMD) override.
 *
 * All fp32 arithmetic here is ordered to mirror the Python oracles exactly; the
 * file is compiled with -ffp-contract=off (see CMakeLists) so no source-level
 * multiply-add is fused, keeping the int8 twin bit-identical to numpy.
 */
#include "tbkern/tbkern.h"

#include <stdlib.h>
#include <string.h>
#include <math.h>

/* ------------------------------------------------------------------ *
 *  Weak activation quant (Phase-0 fallback; quant_act.c overrides later)
 * ------------------------------------------------------------------ */
__attribute__((weak))
int tbk_quant_act(const float *x, int K, int G,
                  int8_t *xq, float *a_scale, int32_t *sum_xq_g) {
    if (!x || !xq || !a_scale || !sum_xq_g || K <= 0 || (G != 64 && G != 128))
        return TBK_EINVAL;
    const int Kp = tbk_pad_k(K, G);
    const int NG = Kp / G;

    /* NaN must propagate to match the numpy oracle (np.abs(x).max()); see the
     * strong override in quant_act.c. `isnan(a) || a > absmax` makes NaN sticky
     * rather than silently dropped by a bare `a > absmax`. */
    float absmax = 0.0f;
    for (int k = 0; k < K; k++) {
        float a = fabsf(x[k]);
        if (isnan(a) || a > absmax) absmax = a;
    }
    const float as = absmax / 127.0f;
    const float id = (as > 0.0f) ? (1.0f / as) : 0.0f;
    *a_scale = as;

    for (int k = 0; k < K; k++) {
        float q = roundf(x[k] * id);          /* round half away from zero */
        int   qi = (int)q;
        if (qi >  127) qi =  127;
        if (qi < -127) qi = -127;
        xq[k] = (int8_t)qi;
    }
    for (int k = K; k < Kp; k++) xq[k] = 0;    /* zero-pad */

    for (int g = 0; g < NG; g++) {
        int32_t s = 0;
        for (int j = 0; j < G; j++) s += xq[g * G + j];
        sum_xq_g[g] = s;
    }
    return TBK_OK;
}

/* ------------------------------------------------------------------ *
 *  Scalar fp32 reference gemv (both layouts)
 * ------------------------------------------------------------------ */
int tbk_gemv_scalar(const tbk_mat *mat, const float *x, float *y) {
    if (!mat || !mat->data || !x || !y) return TBK_EINVAL;
    const int M  = mat->M, K = mat->K, Kp = mat->Kp, G = mat->G;
    const int NG = Kp / G;

    int8_t *w = (int8_t *)malloc((size_t)Kp);
    if (!w) return TBK_ENOMEM;

    for (int r = 0; r < M; r++) {
        int rc = tbk_unpack_row(mat, r, w);
        if (rc != TBK_OK) { free(w); return rc; }
        float yr = mat->bias ? mat->bias[r] : 0.0f;
        for (int g = 0; g < NG; g++) {
            float gacc = 0.0f;
            const int base = g * G;
            for (int j = 0; j < G; j++) {
                int k = base + j;
                float xk = (k < K) ? x[k] : 0.0f;
                gacc += (float)w[k] * xk;
            }
            yr += mat->scales[(size_t)r * NG + g] * gacc;
        }
        y[r] = yr;
    }
    free(w);
    return TBK_OK;
}

/* ------------------------------------------------------------------ *
 *  Scalar int8 twin (bit-exact mirror of Python ref_i8 / the VNNI path)
 * ------------------------------------------------------------------ */
int tbk_gemv_scalar_i8(const tbk_mat *mat, const float *x, float *y) {
    if (!mat || !mat->data || !x || !y) return TBK_EINVAL;
    const int M  = mat->M, K = mat->K, Kp = mat->Kp, G = mat->G;
    const int NG = Kp / G;

    int8_t  *xq       = (int8_t *)malloc((size_t)Kp);
    int32_t *sum_xq_g = (int32_t *)malloc((size_t)NG * sizeof(int32_t));
    int8_t  *w        = (int8_t *)malloc((size_t)Kp);
    if (!xq || !sum_xq_g || !w) { free(xq); free(sum_xq_g); free(w); return TBK_ENOMEM; }

    float a_scale = 0.0f;
    int rc = tbk_quant_act(x, K, G, xq, &a_scale, sum_xq_g);
    if (rc != TBK_OK) { free(xq); free(sum_xq_g); free(w); return rc; }

    for (int r = 0; r < M; r++) {
        rc = tbk_unpack_row(mat, r, w);
        if (rc != TBK_OK) break;
        float yr = mat->bias ? mat->bias[r] : 0.0f;
        for (int g = 0; g < NG; g++) {
            int32_t acc = 0;
            const int base = g * G;
            for (int j = 0; j < G; j++) {
                int u = (int)w[base + j] + 1;          /* u = w + 1 (code)      */
                acc += u * (int)xq[base + j];           /* vpdpbusd equivalent  */
            }
            int32_t corrected = acc - sum_xq_g[g];      /* subtract sum(xq)     */
            float t = (float)corrected * mat->scales[(size_t)r * NG + g];
            t = t * a_scale;
            yr += t;
        }
        y[r] = yr;
    }
    free(xq); free(sum_xq_g); free(w);
    return rc;
}
