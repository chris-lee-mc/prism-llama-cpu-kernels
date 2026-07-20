/* pack.c — GGUF Q2_0 -> kernel-native repacking, unpack oracle, error strings,
 * and the Phase-0 weak-fallback memory arena.
 *
 * Owner: Phase-0 contracts agent. Implements tbk_pack_from_q2 (both layouts),
 * tbk_unpack_row, tbk_mat_free, tbk_strerror, and weak tbk_arena_* stubs that
 * let the library link before arena.c exists. See docs/FORMAT.md for layouts.
 */
#include "tbkern/tbkern.h"

#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#if defined(_MSC_VER)
#include <malloc.h>
#endif

#if defined(_MSC_VER)
#define TBK_WEAK
static void * tbk_aligned_alloc(size_t alignment, size_t size) { return _aligned_malloc(size, alignment); }
static void tbk_aligned_free(void * p) { _aligned_free(p); }
#else
#define TBK_WEAK __attribute__((weak))
static void * tbk_aligned_alloc(size_t alignment, size_t size) { return aligned_alloc(alignment, size); }
static void tbk_aligned_free(void * p) { free(p); }
#endif

/* ------------------------------------------------------------------ *
 *  Error strings
 * ------------------------------------------------------------------ */
const char *tbk_strerror(int err) {
    switch (err) {
        case TBK_OK:           return "ok";
        case TBK_EINVAL:       return "invalid argument";
        case TBK_ENOMEM:       return "out of memory";
        case TBK_ENOSYS:       return "not implemented";
        case TBK_ENOTTERNARY:  return "non-ternary code (+2) in ternary-only layout";
        case TBK_EIO:          return "I/O or format error";
        case TBK_ERANGE:       return "value out of range";
        default:               return "unknown error";
    }
}

/* ------------------------------------------------------------------ *
 *  GGUF block reading helpers
 * ------------------------------------------------------------------ */

/* Read 2-bit code of local weight j (0..G-1) from a block's qs bytes. */
static inline uint8_t block_code(const uint8_t *qs, int j) {
    return (uint8_t)((qs[j >> 2] >> ((j & 3) * 2)) & 0x3u);
}

/* Expand one GGUF row (nb = Kp/G blocks) into Kp codes; pad columns [K,Kp) are
 * forced to TBK_CODE_PAD (01 = w0). Also fills the row's fp32 group scales. */
static void read_row(const uint8_t *row_blocks, int K, int G, int Kp,
                     uint8_t *codes /* len Kp */, float *scales /* len Kp/G */) {
    const int    nb = Kp / G;                       /* == ceil(K/G) */
    const size_t bb = TBK_GGUF_BLOCK_BYTES(G);
    for (int i = 0; i < nb; i++) {
        const uint8_t *blk = row_blocks + (size_t)i * bb;
        uint16_t hbits = (uint16_t)(blk[0] | ((uint16_t)blk[1] << 8)); /* LE fp16 */
        scales[i] = tbk_fp16_to_fp32(hbits);
        const uint8_t *qs = blk + TBK_GGUF_SCALE_BYTES;
        for (int j = 0; j < G; j++) {
            int k = i * G + j;
            codes[k] = (k < K) ? block_code(qs, j) : (uint8_t)TBK_CODE_PAD;
        }
    }
}

/* ------------------------------------------------------------------ *
 *  Layout packers (from a flat Kp-code array)
 * ------------------------------------------------------------------ */

/* Bitplane (Layout A): per 16-weight chunk two LE mask words [P, N].
 * Returns TBK_ENOTTERNARY on a code-3 (+2) weight. */
static int pack_row_bitplane(const uint8_t *codes, int Kp, uint8_t *row) {
    const int nchunks = Kp / 16;
    for (int c = 0; c < nchunks; c++) {
        uint16_t P = 0, N = 0;
        for (int i = 0; i < 16; i++) {
            uint8_t code = codes[16 * c + i];
            if      (code == TBK_CODE_POS1) P |= (uint16_t)(1u << i);
            else if (code == TBK_CODE_NEG1) N |= (uint16_t)(1u << i);
            else if (code == TBK_CODE_POS2) return TBK_ENOTTERNARY;
            /* TBK_CODE_ZERO: leave both clear */
        }
        uint8_t *w = row + (size_t)c * 4;
        w[0] = (uint8_t)(P & 0xFF); w[1] = (uint8_t)(P >> 8);
        w[2] = (uint8_t)(N & 0xFF); w[3] = (uint8_t)(N >> 8);
    }
    return TBK_OK;
}

/* Interleaved codes (Layout B): per 64-weight subgroup 16 bytes, byte b packs
 * codes for k = base+b, +16, +32, +48. Sets *any_plus2 if a code-3 is present. */
static void pack_row_codes(const uint8_t *codes, int Kp, uint8_t *row,
                           int *any_plus2) {
    const int ns = Kp / 64;
    for (int s = 0; s < ns; s++) {
        const int base = 64 * s;
        for (int b = 0; b < 16; b++) {
            uint8_t c0 = codes[base +  0 + b];
            uint8_t c1 = codes[base + 16 + b];
            uint8_t c2 = codes[base + 32 + b];
            uint8_t c3 = codes[base + 48 + b];
            if ((c0 | c1 | c2 | c3) & TBK_CODE_POS2) {
                if (c0 == TBK_CODE_POS2 || c1 == TBK_CODE_POS2 ||
                    c2 == TBK_CODE_POS2 || c3 == TBK_CODE_POS2) *any_plus2 = 1;
            }
            row[(size_t)s * 16 + b] = (uint8_t)(c0 | (c1 << 2) | (c2 << 4) | (c3 << 6));
        }
    }
}

/* ------------------------------------------------------------------ *
 *  Public packing API
 * ------------------------------------------------------------------ */
int tbk_pack_from_q2(const uint8_t *gguf_q2_blocks, int M, int K, int G,
                     tbk_layout_kind layout, tbk_mat *out) {
    if (!gguf_q2_blocks || !out || M <= 0 || K <= 0) return TBK_EINVAL;
    if (G != 64 && G != 128) return TBK_EINVAL;
    if (layout != TBK_LAYOUT_BITPLANE && layout != TBK_LAYOUT_CODES) return TBK_EINVAL;

    const int    Kp     = tbk_pad_k(K, G);
    const int    nb     = Kp / G;                       /* blocks == groups/row */
    const size_t bb     = TBK_GGUF_BLOCK_BYTES(G);
    const size_t stride = tbk_row_stride_bytes(Kp);
    const size_t data_sz   = (size_t)M * stride;
    const size_t scales_sz = (size_t)M * nb * sizeof(float);

    memset(out, 0, sizeof(*out));

    uint8_t *data   = (uint8_t *)tbk_aligned_alloc(TBK_ROW_ALIGN,
                                               tbk_align_up(data_sz, TBK_ROW_ALIGN));
    float   *scales = (float *)malloc(scales_sz);
    uint8_t *codes  = (uint8_t *)malloc((size_t)Kp);      /* scratch, one row     */
    if (!data || !scales || !codes) {
        tbk_aligned_free(data); free(scales); free(codes);
        return TBK_ENOMEM;
    }
    memset(data, 0, data_sz);

    int any_plus2 = 0;
    int rc = TBK_OK;
    for (int r = 0; r < M; r++) {
        const uint8_t *row_blocks = gguf_q2_blocks + (size_t)r * nb * bb;
        uint8_t       *row        = data + (size_t)r * stride;
        read_row(row_blocks, K, G, Kp, codes, scales + (size_t)r * nb);
        if (layout == TBK_LAYOUT_BITPLANE) {
            rc = pack_row_bitplane(codes, Kp, row);
            if (rc != TBK_OK) break;
        } else {
            pack_row_codes(codes, Kp, row, &any_plus2);
        }
    }
    free(codes);
    if (rc != TBK_OK) { tbk_aligned_free(data); free(scales); return rc; }

    out->M               = M;
    out->K               = K;
    out->Kp              = Kp;
    out->G               = G;
    out->layout          = layout;
    out->data            = data;
    out->row_stride_bytes = stride;
    out->scales          = scales;
    out->bias            = NULL;
    out->ternary_only    = (layout == TBK_LAYOUT_BITPLANE) ? 1 : (any_plus2 ? 0 : 1);
    out->_owner          = data;
    return TBK_OK;
}

void tbk_mat_free(tbk_mat *mat) {
    if (!mat || !mat->_owner) { if (mat) memset(mat, 0, sizeof(*mat)); return; }
    tbk_aligned_free(mat->data);
    free(mat->scales);
    /* bias is not owned by the packer (always NULL here) */
    memset(mat, 0, sizeof(*mat));
}

/* ------------------------------------------------------------------ *
 *  Unpack oracle: one packed row -> signed int8 weights (length Kp)
 * ------------------------------------------------------------------ */
int tbk_unpack_row(const tbk_mat *mat, int row, int8_t *w_out) {
    if (!mat || !mat->data || !w_out || row < 0 || row >= mat->M) return TBK_EINVAL;
    const uint8_t *rp = (const uint8_t *)mat->data + (size_t)row * mat->row_stride_bytes;
    const int Kp = mat->Kp;

    if (mat->layout == TBK_LAYOUT_BITPLANE) {
        const int nchunks = Kp / 16;
        for (int c = 0; c < nchunks; c++) {
            const uint8_t *w = rp + (size_t)c * 4;
            uint16_t P = (uint16_t)(w[0] | ((uint16_t)w[1] << 8));
            uint16_t N = (uint16_t)(w[2] | ((uint16_t)w[3] << 8));
            for (int i = 0; i < 16; i++) {
                int8_t v = 0;
                if (P & (1u << i)) v = 1;
                else if (N & (1u << i)) v = -1;
                w_out[16 * c + i] = v;
            }
        }
    } else { /* TBK_LAYOUT_CODES */
        const int ns = Kp / 64;
        for (int s = 0; s < ns; s++) {
            const int base = 64 * s;
            for (int b = 0; b < 16; b++) {
                uint8_t byte = rp[(size_t)s * 16 + b];
                for (int q = 0; q < 4; q++) {
                    uint8_t code = (uint8_t)((byte >> (2 * q)) & 0x3u);
                    w_out[base + 16 * q + b] = (int8_t)((int)code - 1);
                }
            }
        }
    }
    return TBK_OK;
}

/* ================================================================== *
 *  Weak-fallback memory arena (Phase 0).
 *
 *  A minimal malloc-backed bump allocator so the library links and runs
 *  before arena.c lands. arena.c will provide STRONG definitions of these
 *  three symbols (huge pages / mlock) that override these weak ones.
 * ================================================================== */
struct tbk_arena {
    uint8_t *base;
    size_t   size;
    size_t   used;
};

TBK_WEAK
tbk_arena *tbk_arena_create(size_t size, int mode_flags) {
    (void)mode_flags;                       /* fallback ignores THP/HUGETLB/MLOCK */
    if (size == 0) size = 1;
    struct tbk_arena *a = (struct tbk_arena *)malloc(sizeof(*a));
    if (!a) return NULL;
    a->base = (uint8_t *)tbk_aligned_alloc(TBK_ROW_ALIGN, tbk_align_up(size, TBK_ROW_ALIGN));
    if (!a->base) { free(a); return NULL; }
    a->size = size;
    a->used = 0;
    return a;
}

TBK_WEAK
void *tbk_arena_alloc(tbk_arena *a, size_t size, size_t align) {
    if (!a || align == 0) return NULL;
    size_t off = tbk_align_up(a->used, align);
    if (off + size > a->size) return NULL;
    a->used = off + size;
    return a->base + off;
}

TBK_WEAK
void tbk_arena_free(tbk_arena *a) {
    if (!a) return;
    tbk_aligned_free(a->base);
    free(a);
}
