/* ===========================================================================
 * ZENITH — Zero-rebuild Eigenmetric Nearest-neighbor Index
 *          with Tunable Hyperplanes
 *
 * Core observation (Spectral Sign Invariance):
 *   sign(sqrt(w_k) * (Phi x)_k) == sign((Phi x)_k)  for all w_k > 0.
 * A single static sign-sketch index can therefore be re-scored under any
 * positive diagonal spectral metric chosen at query time. No rebuild.
 *
 * Important limitations (read these):
 *   - MIH/Gray/Hybrid candidate generation is weight-agnostic. Hamming
 *     ranks packed sketches by true Hamming distance. Posterior uses the
 *     query and metric to score quantized codes. Changing weights can change
 *     true nearest neighbors even when codes stay the same. The index is a
 *     candidate generator + leading-subspace ranker, not a guarantee of exact
 *     kNN under arbitrary W. MIH backfills from the Gray ordering if sparse
 *     buckets leave fewer than ef candidates.
 *   - Reported distances are lower bounds (leading weighted distance +
 *     residual lower bound using min tail weight). They are not exact d_W
 *     unless Mcoef == D.
 *   - MIH expands radius progressively up to a hard limit (default 2).
 *     When ef reaches N, the implementation falls back to an exhaustive
 *     sorted-key scan rather than pretending that radius-limited MIH saw all
 *     candidates. Use Gray scan + ef = N (or Mcoef == D) for exact subspace
 *     ranking with predictable traversal order.
 *
 * Designed for the narrow but real niche:
 *   - Static corpus
 *   - Fixed orthonormal basis (DCT-II, optionally after a deterministic
 *     input-side ±1 diagonal)
 *   - Many positive diagonal spectral metrics at query time
 *   - Tolerance for approximate candidates + optional exact re-rank
 *
 * SIMD: compile-time AVX-512 / AVX2+FMA / NEON / scalar kernels live under
 *       include/zenith/arch/, with a reserved CUDA slot in the same folder.
 *       The DCT-II projector (dense / FFTW) is include/zenith/dct.h; OpenMP
 *       helpers are include/zenith/threads.h. Optional FFTW3 for large DCT-II,
 *       optional OpenMP for NUMA-friendly parallel build/query.
 *
 * Copyright (c) 2026 Damus <damus@straylightrun.org>
 * =========================================================================== */

#ifndef ZENITH_H
#define ZENITH_H

#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include <float.h>

/* ---------- Precision & alignment ---------- */
#if defined(__GNUC__) || defined(__clang__)
#  define ZENITH_ALIGN64          __attribute__((aligned(64)))
#  define ZENITH_ALWAYS_INLINE    static inline __attribute__((always_inline))
#  define ZENITH_HOT              __attribute__((hot))
#  define ZENITH_LIKELY(x)        __builtin_expect(!!(x), 1)
#  define ZENITH_UNLIKELY(x)      __builtin_expect(!!(x), 0)
#  define ZENITH_PREFETCH(p)      __builtin_prefetch((const void *)(p), 0, 3)
#elif defined(_MSC_VER)
#  define ZENITH_ALIGN64          __declspec(align(64))
#  define ZENITH_ALWAYS_INLINE    static __forceinline
#  define ZENITH_HOT
#  define ZENITH_LIKELY(x)        (x)
#  define ZENITH_UNLIKELY(x)      (x)
#  define ZENITH_PREFETCH(p)      ((void)0)
#else
#  define ZENITH_ALIGN64
#  define ZENITH_ALWAYS_INLINE    static inline
#  define ZENITH_HOT
#  define ZENITH_LIKELY(x)        (x)
#  define ZENITH_UNLIKELY(x)      (x)
#  define ZENITH_PREFETCH(p)      ((void)0)
#endif

#if defined(__cplusplus)
#  if defined(__GNUC__) || defined(__clang__)
#    define ZENITH_RESTRICT __restrict__
#  elif defined(_MSC_VER)
#    define ZENITH_RESTRICT __restrict
#  else
#    define ZENITH_RESTRICT
#  endif
#else
#  define ZENITH_RESTRICT restrict
#endif

#if defined(__cplusplus)
#  define ZENITH_THREAD_LOCAL thread_local
#elif defined(_MSC_VER)
#  define ZENITH_THREAD_LOCAL __declspec(thread)
#elif defined(__STDC_VERSION__) && __STDC_VERSION__ >= 201112L
#  define ZENITH_THREAD_LOCAL _Thread_local
#else
#  define ZENITH_THREAD_LOCAL /* C99 fallback: last-error is process-local */
#endif

#ifndef ZENITH_PI
#define ZENITH_PI 3.14159265358979323846
#endif

#define ZENITH_EPS_DIV 1e-30f
#define ZENITH_EPS_DC  1e-12f

#ifndef ZENITH_SKETCH_BITS
#define ZENITH_SKETCH_BITS 64u
#endif

#define ZENITH_VERSION_MAJOR 1
#define ZENITH_VERSION_MINOR 0
#define ZENITH_VERSION_PATCH 0
#define ZENITH_VERSION_STRING "1.0.0"

/* ---------- Error codes ---------- */
typedef enum {
    ZENITH_OK            = 0,
    ZENITH_ERR_ALLOC     = 1,
    ZENITH_ERR_ARG       = 2,
    ZENITH_ERR_IO        = 3,
    ZENITH_ERR_WEIGHT    = 4,   /* non-finite or negative leading weight */
    ZENITH_ERR_PERM      = 5,   /* bad frequency permutation */
    ZENITH_ERR_FORMAT    = 6    /* malformed or incompatible persisted index */
} zenith_err_t;

/* ---------- Projection generators ---------- */
typedef enum {
    ZENITH_GEN_DCT = 0,        /* pure orthonormal DCT-II */
    ZENITH_GEN_SIGNED_DCT = 1  /* deterministic input-side ±1 diagonal + DCT */
} zenith_generator_t;

/* ---------- Candidate generation ---------- */
typedef enum {
    ZENITH_SEARCH_MIH       = 0, /* multi-index hashing (fast default) */
    ZENITH_SEARCH_GRAY      = 1, /* bidirectional Gray-code scan */
    ZENITH_SEARCH_HYBRID    = 2, /* split ef between MIH and Gray candidates */
    ZENITH_SEARCH_HAMMING   = 3, /* exact packed-key Hamming candidate selection */
    ZENITH_SEARCH_POSTERIOR = 4  /* query/metric-aware quantized-code posterior */
} zenith_search_t;

/* ---------- Scalar quantization ---------- */
typedef enum {
    ZENITH_QUANT_SIGN     = 0, /* one sign bit per coded coefficient (default) */
    ZENITH_QUANT_LM2      = 1, /* packed 2-bit Lloyd-Max Gaussian codes */
    ZENITH_QUANT_SIGN_MAG = 2  /* one sign key plus one Lloyd-Max magnitude key */
} zenith_quantizer_t;

/* ---------- Build options ---------- */
typedef struct {
    uint32_t Mcoef;               /* leading coefficients kept (default min(D,128)) */
    uint32_t nbits;               /* sketch bits (default 64) */
    uint64_t seed;                /* deterministic seed for signed-DCT diagonal */
    zenith_generator_t gen;       /* projection family */
    zenith_search_t search;       /* candidate generator (default MIH) */
    int      use_polar;           /* 2-bit polar quadrant encoding */
    int      use_perm;            /* spectral bit allocation permutation */
    int      auto_perm;           /* derive permutation from corpus spectral energy */
    uint8_t  freq_perm[64];       /* if use_perm and !auto_perm: frequency order */
    const float *whiten_baseline; /* optional positive baseline weights */
    uint32_t mih_max_radius;      /* progressive MIH radius, capped at 2 (0 = default 2) */
    zenith_quantizer_t quantizer; /* scalar code family (default SIGN) */
    int      nthreads;            /* 0 = zenith_thread_count(); stored for query */
    int      fftw;                /* 0 auto, >0 force FFTW, <0 dense DCT only */
} zenith_opts;

/* ---------- Index (flattened, mmap-ready) ---------- */
typedef struct {
    uint32_t N, D, Mcoef, nbits;
    uint64_t *keys;               /* Gray-coded sign keys, or raw quantized codes */
    uint64_t *mag_keys;           /* SIGN_MAG only: Lloyd-Max magnitude bits */
    float    *quant_scale;        /* LM2/SIGN_MAG: per-coded-coordinate std dev */
    float    *quant_bias;         /* LM2/SIGN_MAG: per-coded-coordinate mean */
    uint32_t  ncodes;             /* coded coefficients (nbits/2 for LM2) */
    int       quantizer;          /* zenith_quantizer_t */
    uint32_t *ids;                /* original vector ids, permuted with keys */
    float    *coeffs;             /* N * Mcoef leading spectral coeffs, 64B aligned */
    float    *residual;           /* N tail L2 norms */
    float    *whiten_sq;          /* if used: sqrt(baseline) per coef */
    int       use_whiten;
    int       use_polar;
    int       use_perm;
    int       gen;                /* zenith_generator_t used at build time */
    uint64_t  gen_seed;           /* deterministic sign seed */
    uint8_t   freq_perm[64];
    int       search;             /* zenith_search_t */
    uint32_t  mih_m;
    uint32_t  mih_bits;
    uint32_t  mih_max_radius;     /* progressive expansion limit */
    uint32_t *mih_off;            /* m * ((1<<mih_bits)+1) CSR offsets */
    uint32_t *mih_idx;            /* m * N  index positions, table-major */
    /* internal DCT plan (not serialized) */
    int       plan_D, plan_M;
    float    *plan_t;             /* dense M×D rows; NULL when use_fftw */
    int       use_fftw;
    void     *fftw_plan;          /* fftwf_plan, opaque */
    float    *fftw_in, *fftw_out; /* serial FFTW scratch */
    float    *fftw_scale;         /* length D orthonormal REDFT10 scales */
    int       nthreads;           /* build/query OpenMP team size */
    /* optional deterministic input signs (recomputed from seed on load) */
    int8_t   *input_signs;        /* D entries, ±1; NULL for pure DCT */
    /* mmap state */
    int       mmapped;
    void     *map_base;
    size_t    map_len;
} zenith_index_t;

/* ---------- Public API ---------- */

#ifdef __cplusplus
extern "C" {
#endif

/* Build a static index. vectors is N*D row-major float. Returns NULL on error. */
zenith_index_t *zenith_build(const float *ZENITH_RESTRICT vectors,
                             uint32_t N, uint32_t D, zenith_opts opts);

/* Free an index built by zenith_build or loaded by zenith_load. */
void zenith_free(zenith_index_t *idx);

/* Query under a non-negative diagonal spectral weight of length D.
 * Leading coefficients use weight[0..Mcoef). If Mcoef < D, the tail term
 * uses min(weight[Mcoef..D)). Returns hits written (<=k). out_dist receives
 * sqrt of the scored lower bound. Negative / non-finite weights are rejected. */
uint32_t zenith_query_ef(const zenith_index_t *ZENITH_RESTRICT idx,
                         const float *ZENITH_RESTRICT q,
                         const float *ZENITH_RESTRICT weight,
                         uint32_t k, uint32_t ef,
                         uint32_t *ZENITH_RESTRICT out_ids,
                         float *ZENITH_RESTRICT out_dist);

/* Return the raw candidate pool before retained-distance top-k selection.
 * out_ids has capacity ef and receives up to min(ef,N) distinct candidates
 * in candidate-generator order. weight has length D. */
uint32_t zenith_candidates_ef(const zenith_index_t *ZENITH_RESTRICT idx,
                              const float *ZENITH_RESTRICT q,
                              const float *ZENITH_RESTRICT weight,
                              uint32_t ef,
                              uint32_t *ZENITH_RESTRICT out_ids);

/* Convenience recall-oriented default: ef = max(512, 24*k) clamped to N. */
uint32_t zenith_query(const zenith_index_t *ZENITH_RESTRICT idx,
                      const float *ZENITH_RESTRICT q,
                      const float *ZENITH_RESTRICT weight,
                      uint32_t k,
                      uint32_t *ZENITH_RESTRICT out_ids,
                      float *ZENITH_RESTRICT out_dist);

/* Coefficient-space query: caller already has q in the index's signed basis
 * (see zenith_project). Avoids the dense DCT. q_coeff length Mcoef, weight
 * length D. Valid only when ncodes <= Mcoef (for LM2, ncodes is nbits/2).
 * Otherwise the function returns ZENITH_ERR_ARG. Use zenith_query_ef for
 * those indexes. */
uint32_t zenith_query_coeffs(const zenith_index_t *ZENITH_RESTRICT idx,
                             const float *ZENITH_RESTRICT q_coeff,
                             float q_residual,
                             const float *ZENITH_RESTRICT weight,
                             uint32_t k, uint32_t ef,
                             uint32_t *ZENITH_RESTRICT out_ids,
                             float *ZENITH_RESTRICT out_dist);

/* Project one vector into the index's signed spectral basis. out_coeff has
 * length Mcoef; out_residual receives the exact tail L2 norm (0 if Mcoef=D). */
int zenith_project(const zenith_index_t *ZENITH_RESTRICT idx,
                   const float *ZENITH_RESTRICT x,
                   float *ZENITH_RESTRICT out_coeff,
                   float *ZENITH_RESTRICT out_residual);

/* Exact re-rank helper: recompute the true signed-spectral d_W against the
 * original vectors. idx supplies the basis/sign convention; weight has length D.
 * This is intended for a modest candidate set, not as the primary ANN path. */
uint32_t zenith_rerank_exact(const zenith_index_t *ZENITH_RESTRICT idx,
                             const float *ZENITH_RESTRICT vectors,
                             const float *ZENITH_RESTRICT q,
                             const float *ZENITH_RESTRICT weight,
                             const uint32_t *cand_ids,
                             uint32_t ncand,
                             uint32_t k,
                             uint32_t *out_ids,
                             float *ZENITH_RESTRICT out_dist);

/* Minimum of weight[Mcoef..D). Used internally by query when Mcoef < D;
 * exposed for callers who score a truncated subspace themselves. */
float zenith_tail_min(const float *ZENITH_RESTRICT weight, int D, int Mcoef);

/* Zero an options struct. Equivalent to memset(opts, 0, sizeof(*opts)). */
ZENITH_ALWAYS_INLINE void zenith_opts_init(zenith_opts *opts) {
    if (opts) memset(opts, 0, sizeof(*opts));
}

/* Last error recorded by the calling thread (per translation unit). */
zenith_err_t zenith_last_error(void);
void zenith_clear_error(void);

/* Persistence. The file format is version 1, endian-checked, and 64-byte
 * aligned for SIMD-friendly mmap. POSIX builds memory-map the arrays;
 * zenith_load_copy always owns a private copy. */
int zenith_save(const zenith_index_t *idx, const char *path);
zenith_index_t *zenith_load(const char *path);
zenith_index_t *zenith_load_copy(const char *path);
void zenith_unload(zenith_index_t *idx);

/* Weight constructors (O(D), strictly positive after flooring).
 * mu is typically zenith_laplacian_eigs(): mu_k = 2 (1 - cos(pi k / D)).
 *   identity:   w_k = 1
 *   sobolev:    w_k = (1 + mu_k)^s
 *   matern:     w_k = (mu_k + kappa^2)^nu     (precision exponent nu)
 *   fractional: w_k = mu_k^s                  ((-Delta)^s energy)
 *   roughvol:   w_k = mu_k^(2H+1)             (Hurst-H roughness knob) */
void zenith_w_identity(int D, float *ZENITH_RESTRICT w);
void zenith_w_sobolev(const double *ZENITH_RESTRICT mu, int D, double s,
                      float *ZENITH_RESTRICT w);
void zenith_w_matern(const double *ZENITH_RESTRICT mu, int D,
                     double kappa, double nu, float *ZENITH_RESTRICT w);
void zenith_w_fractional(const double *ZENITH_RESTRICT mu, int D, double s,
                         float *ZENITH_RESTRICT w);
void zenith_w_roughvol(const double *ZENITH_RESTRICT mu, int D, double H,
                       float *ZENITH_RESTRICT w);

/* Laplacian eigenvalues for the weight constructors (1D Neumann / DCT-II).
 * mu_k = 2 (1 - cos(pi k / D)), k = 0..D-1. */
void zenith_laplacian_eigs(int D, double *ZENITH_RESTRICT mu);

/* Spectral bit-allocation helpers. */
void zenith_sba_matern_envelope(const double *mu, int D,
                                const double *kappas, const double *nus, int nk,
                                float *density);
int  zenith_sba_perm(const float *density, int D, int nbits, uint8_t *perm);

/* Runtime SIMD level: 0=scalar, 1=NEON, 2=AVX2, 3=AVX-512 (compile-time). */
int zenith_simd_level(void);
int zenith_fftw_enabled(void);
int zenith_openmp_enabled(void);
int zenith_thread_count(void);
void zenith_set_threads(int n);

/* Parallel queries: Q is nq×D row-major. out_ids/out_dist are nq×k. */
uint32_t zenith_query_many(const zenith_index_t *ZENITH_RESTRICT idx,
                           const float *ZENITH_RESTRICT Q, uint32_t nq,
                           const float *ZENITH_RESTRICT weight,
                           uint32_t k, uint32_t ef,
                           uint32_t *ZENITH_RESTRICT out_ids,
                           float *ZENITH_RESTRICT out_dist);

#ifdef __cplusplus
}
#endif

/* ===========================================================================
 * IMPLEMENTATION
 * =========================================================================== */
#ifdef ZENITH_IMPLEMENTATION

#if defined(_WIN32)
#  include <malloc.h>
#endif
#if !defined(ZENITH_NO_MMAP) && (defined(__unix__) || defined(__APPLE__))
#  define ZENITH_HAVE_MMAP 1
#  include <sys/mman.h>
#  include <sys/stat.h>
#  include <fcntl.h>
#  include <unistd.h>
#endif

#include "zenith/arch/simd.h"
#include "zenith/arch/cuda.h"

/* Last error is thread-local in the implementation translation unit. */
static ZENITH_THREAD_LOCAL zenith_err_t zenith__last_error = ZENITH_OK;
zenith_err_t zenith_last_error(void) { return zenith__last_error; }
void zenith_clear_error(void) { zenith__last_error = ZENITH_OK; }
static void zenith__set_error(zenith_err_t e) { zenith__last_error = e; }

/* ---------- Pluggable allocator ---------- */
#ifndef ZENITH_MALLOC
#  define ZENITH_MALLOC(sz)        malloc(sz)
#endif
#ifndef ZENITH_FREE
#  define ZENITH_FREE(p)           free(p)
#endif
#ifndef ZENITH_REALLOC
#  define ZENITH_REALLOC(p, sz)    realloc((p), (sz))
#endif

#if !defined(ZENITH_ALIGNED_ALLOC)
static void *zenith__default_aligned_alloc(size_t alignment, size_t size) {
    if (alignment < sizeof(void *)) alignment = sizeof(void *);
    if (size == 0) size = alignment;
#if defined(_WIN32)
    return _aligned_malloc(size, alignment);
#elif defined(__STDC_VERSION__) && __STDC_VERSION__ >= 201112L && \
      !defined(__MINGW32__)
    {
        size_t aligned_size = (size + alignment - 1u) & ~(alignment - 1u);
        if (aligned_size == 0) return NULL;
        return aligned_alloc(alignment, aligned_size);
    }
#else
    {
        size_t pad = alignment - 1u + sizeof(void *);
        char *raw = (char *)ZENITH_MALLOC(size + pad);
        void *p;
        if (!raw) return NULL;
        p = (void *)(((uintptr_t)raw + sizeof(void *) + (alignment - 1u))
                     & ~(uintptr_t)(alignment - 1u));
        memcpy((char *)p - sizeof(void *), &raw, sizeof(void *));
        return p;
    }
#endif
}
#  define ZENITH_ALIGNED_ALLOC(align, sz) zenith__default_aligned_alloc((align), (sz))
#endif

#if !defined(ZENITH_ALIGNED_FREE)
static void zenith__default_aligned_free(void *p) {
    if (!p) return;
#if defined(_WIN32)
    _aligned_free(p);
#elif defined(__STDC_VERSION__) && __STDC_VERSION__ >= 201112L && \
      !defined(__MINGW32__)
    free(p);
#else
    {
        void *raw;
        memcpy(&raw, (char *)p - sizeof(void *), sizeof(void *));
        ZENITH_FREE(raw);
    }
#endif
}
#  define ZENITH_ALIGNED_FREE(p) zenith__default_aligned_free(p)
#endif

/* ---------- Gray code ---------- */
ZENITH_ALWAYS_INLINE uint64_t zenith_gray(uint64_t x) {
    return x ^ (x >> 1);
}
ZENITH_ALWAYS_INLINE uint64_t zenith_ungray(uint64_t x) {
    x ^= x >> 1;
    x ^= x >> 2;
    x ^= x >> 4;
    x ^= x >> 8;
    x ^= x >> 16;
    x ^= x >> 32;
    return x;
}

ZENITH_ALWAYS_INLINE uint32_t zenith__popcount64(uint64_t x) {
#if defined(__GNUC__) || defined(__clang__)
    return (uint32_t)__builtin_popcountll(x);
#else
    x -= (x >> 1) & UINT64_C(0x5555555555555555);
    x = (x & UINT64_C(0x3333333333333333)) + ((x >> 2) & UINT64_C(0x3333333333333333));
    x = (x + (x >> 4)) & UINT64_C(0x0f0f0f0f0f0f0f0f);
    return (uint32_t)((x * UINT64_C(0x0101010101010101)) >> 56);
#endif
}

ZENITH_ALWAYS_INLINE uint32_t zenith__hamming(uint64_t a, uint64_t b, uint32_t nbits) {
    uint64_t mask = (nbits >= 64u) ? ~UINT64_C(0) : ((UINT64_C(1) << nbits) - 1u);
    return zenith__popcount64((a ^ b) & mask);
}

/* ---------- Simple checksum (FNV-1a 32-bit style) ---------- */
static uint32_t zenith__checksum(const void *data, size_t len) {
    const uint8_t *p = (const uint8_t *)data;
    uint32_t h = 2166136261u;
    size_t i;
    for (i = 0; i < len; ++i) {
        h ^= p[i];
        h *= 16777619u;
    }
    return h;
}

/* ---------- Multi-index hashing layout ----------
 * Sketches are packed LSB-first: bit i corresponds to spectral bit i.  MIH
 * chunks therefore also run LSB-first, so chunk 0 contains the lowest-frequency
 * coefficients.  The final chunk may be narrower; every sketch bit is covered. */
static void zenith__mih_layout(uint32_t nbits, uint32_t *m_out, uint32_t *bits_out) {
    uint32_t m, bits;
    if (nbits == 0) { *m_out = 0; *bits_out = 0; return; }
    if (nbits <= 16u)      m = 1u;
    else if (nbits <= 32u) m = 2u;
    else                   m = 4u;
    bits = (nbits + m - 1u) / m;
    if (bits > 16u) { /* defensive; nbits is capped at 64 */
        bits = 16u;
        m = (nbits + bits - 1u) / bits;
    }
    *m_out = m;
    *bits_out = bits;
}

ZENITH_ALWAYS_INLINE uint32_t zenith__mih_chunk_bits(uint32_t nbits,
                                                     uint32_t base_bits,
                                                     uint32_t t) {
    uint32_t begin = t * base_bits;
    uint32_t rem = nbits - begin;
    return rem < base_bits ? rem : base_bits;
}

ZENITH_ALWAYS_INLINE uint32_t zenith__mih_chunk(uint64_t raw, uint32_t nbits,
                                                uint32_t t, uint32_t bits) {
    uint32_t cbits = zenith__mih_chunk_bits(nbits, bits, t);
    uint32_t shift = t * bits;
    uint32_t mask = (cbits >= 32u) ? 0xFFFFFFFFu : ((1u << cbits) - 1u);
    return (uint32_t)(raw >> shift) & mask;
}

static int zenith__mih_build(zenith_index_t *idx) {
    uint32_t N, m, bits, nb, t, i, b;
    uint32_t *count, *cursor;
    size_t off_n, idx_n;

    zenith__mih_layout(idx->nbits, &m, &bits);
    if (m == 0 || bits == 0 || bits > 16u) return ZENITH_ERR_ARG;
    N = idx->N;
    nb = 1u << bits;
    off_n = (size_t)m * (size_t)(nb + 1u);
    idx_n = (size_t)m * (size_t)N;
    if (N != 0 && idx_n / (size_t)N != (size_t)m) return ZENITH_ERR_ALLOC;

    count  = (uint32_t *)ZENITH_MALLOC(sizeof(uint32_t) * (size_t)m * nb);
    cursor = (uint32_t *)ZENITH_MALLOC(sizeof(uint32_t) * (size_t)m * nb);
    idx->mih_off = (uint32_t *)ZENITH_ALIGNED_ALLOC(64, sizeof(uint32_t) * off_n);
    idx->mih_idx = (uint32_t *)ZENITH_ALIGNED_ALLOC(64, sizeof(uint32_t) * idx_n);
    if (!count || !cursor || !idx->mih_off || !idx->mih_idx) {
        ZENITH_FREE(count); ZENITH_FREE(cursor);
        ZENITH_ALIGNED_FREE(idx->mih_off); ZENITH_ALIGNED_FREE(idx->mih_idx);
        idx->mih_off = NULL; idx->mih_idx = NULL;
        return ZENITH_ERR_ALLOC;
    }
    memset(count, 0, sizeof(uint32_t) * (size_t)m * nb);

    for (i = 0; i < N; ++i) {
        uint64_t raw = zenith_ungray(idx->keys[i]);
        for (t = 0; t < m; ++t) {
            uint32_t c = zenith__mih_chunk(raw, idx->nbits, t, bits);
            count[(size_t)t * nb + c]++;
        }
    }

    for (t = 0; t < m; ++t) {
        uint32_t sum = 0;
        uint32_t *off = idx->mih_off + (size_t)t * (nb + 1u);
        for (b = 0; b < nb; ++b) {
            off[b] = sum;
            sum += count[(size_t)t * nb + b];
        }
        off[nb] = sum;
    }

    for (t = 0; t < m; ++t)
        memcpy(cursor + (size_t)t * nb,
               idx->mih_off + (size_t)t * (nb + 1u),
               sizeof(uint32_t) * nb);

    for (i = 0; i < N; ++i) {
        uint64_t raw = zenith_ungray(idx->keys[i]);
        for (t = 0; t < m; ++t) {
            uint32_t c = zenith__mih_chunk(raw, idx->nbits, t, bits);
            uint32_t at = cursor[(size_t)t * nb + c]++;
            idx->mih_idx[(size_t)t * N + at] = i;
        }
    }

    ZENITH_FREE(count); ZENITH_FREE(cursor);
    idx->mih_m = m;
    idx->mih_bits = bits;
    return ZENITH_OK;
}

#include "zenith/dct.h"

/* ---------- Deterministic signed DCT ----------
 * A random-looking ±1 diagonal before an orthonormal DCT preserves Euclidean
 * geometry while decorrelating coefficient signs.  Use a SplitMix64 hash of
 * (seed, dimension); unlike an LCG low bit, every coordinate is well mixed. */
static uint64_t zenith__mix64(uint64_t z) {
    z = (z ^ (z >> 30)) * UINT64_C(0xbf58476d1ce4e5b9);
    z = (z ^ (z >> 27)) * UINT64_C(0x94d049bb133111eb);
    return z ^ (z >> 31);
}

static void zenith__gen_input_signs(int8_t *signs, uint32_t D, uint64_t seed) {
    uint32_t i;
    for (i = 0; i < D; ++i) {
        uint64_t z = zenith__mix64(seed ^
                                   (UINT64_C(0x9e3779b97f4a7c15) * (uint64_t)(i + 1u)));
        signs[i] = (z >> 63) ? (int8_t)1 : (int8_t)-1;
    }
}

/* ---------- Sketch helpers (sign + optional polar) ---------- */
static uint64_t zenith_sketch(const float *X, int nbits, const uint8_t *perm) {
    uint64_t h = 0;
    int b;
    for (b = 0; b < nbits; ++b) {
        int idx = perm ? (int)perm[b] : b;
        if (X[idx] >= 0.0f)
            h |= (1ULL << b);
    }
    return h;
}

/* Optimal four-level Lloyd-Max quantizer for a unit Gaussian.  The symbols
 * use 2-bit Gray order so adjacent bins differ in one packed-code bit. */
#define ZENITH_LM2_THRESHOLD 0.9815995201384864
#define ZENITH_LM2_INNER     0.4527804905201459
#define ZENITH_LM2_OUTER     1.5104185497568270
#define ZENITH_LM2_E2_INNER  0.2819141710577494
#define ZENITH_LM2_E2_OUTER  2.4826268484413014

static uint32_t zenith__lm2_symbol(double z, double scale) {
    double t = ZENITH_LM2_THRESHOLD * scale;
    if (z < -t) return 0u;
    if (z < 0.0) return 1u;
    if (z <  t) return 3u;
    return 2u;
}

static uint64_t zenith_sketch_lm2(const float *X, int ncodes,
                                  const uint8_t *perm, const float *scale,
                                  const float *bias) {
    uint64_t h = 0;
    int b;
    for (b = 0; b < ncodes; ++b) {
        int idx = perm ? (int)perm[b] : b;
        double z = (double)X[idx] - (bias ? bias[b] : 0.0);
        uint64_t s = zenith__lm2_symbol(z, scale ? scale[b] : 1.0f);
        h |= s << (2 * b);
    }
    return h;
}

static uint64_t zenith_sketch_sign_calibrated(const float *X, int ncodes,
                                              const uint8_t *perm,
                                              const float *bias) {
    uint64_t h = 0;
    int b;
    for (b = 0; b < ncodes; ++b) {
        int idx = perm ? (int)perm[b] : b;
        double z = (double)X[idx] - (bias ? bias[b] : 0.0);
        if (z >= 0.0)
            h |= (1ULL << b);
    }
    return h;
}

static uint64_t zenith_sketch_magnitude(const float *X, int nbits,
                                        const uint8_t *perm, const float *scale,
                                        const float *bias) {
    uint64_t h = 0;
    int b;
    for (b = 0; b < nbits; ++b) {
        int idx = perm ? (int)perm[b] : b;
        double z = (double)X[idx] - (bias ? bias[b] : 0.0);
        double t = ZENITH_LM2_THRESHOLD * (scale ? scale[b] : 1.0f);
        if (fabs(z) >= t)
            h |= (1ULL << b);
    }
    return h;
}

static uint64_t zenith_sketch_polar(const float *X, int npairs,
                                    const uint8_t *perm) {
    uint64_t h = 0;
    int p;
    for (p = 0; p < npairs; ++p) {
        /* With a permutation, each polar pair consumes the next two selected
         * frequencies.  Without one, preserve the original consecutive-pair
         * layout.  The quadrant map is Gray-coded so adjacent quadrants differ
         * in one bit. */
        int ia = perm ? (int)perm[2*p]     : 2*p;
        int ib = perm ? (int)perm[2*p + 1] : 2*p + 1;
        float a = X[ia], b = X[ib];
        uint32_t code = 0;
        static const uint32_t gray2[4] = {0, 1, 3, 2};
        if (a >= 0.0f) code |= 1u;
        if (b >= 0.0f) code |= 2u;
        h |= ((uint64_t)gray2[code] << (2*p));
    }
    return h;
}

/* Greedy maximum-energy spectral bit allocation. */
static int zenith__auto_perm(const double *ZENITH_RESTRICT energy,
                             uint32_t Mcoef, uint32_t nbits,
                             uint8_t *ZENITH_RESTRICT perm) {
    uint8_t *used;
    uint32_t b, i;
    if (!energy || !perm || nbits == 0 || nbits > Mcoef) return ZENITH_ERR_ARG;
    used = (uint8_t *)ZENITH_MALLOC(Mcoef);
    if (!used) return ZENITH_ERR_ALLOC;
    memset(used, 0, Mcoef);
    for (b = 0; b < nbits; ++b) {
        uint32_t best = 0;
        double best_e = -1.0;
        for (i = 0; i < Mcoef; ++i) {
            if (!used[i] && energy[i] > best_e) {
                best_e = energy[i];
                best = i;
            }
        }
        used[best] = 1;
        perm[b] = (uint8_t)best;
    }
    ZENITH_FREE(used);
    return ZENITH_OK;
}

/* ---------- Sort (re-entrant: no global) ---------- */
typedef struct {
    uint64_t key;
    uint64_t mag_key;
    uint32_t id;
} zenith__kv;

static int zenith__kv_cmp(const void *a, const void *b) {
    const zenith__kv *ka = (const zenith__kv *)a;
    const zenith__kv *kb = (const zenith__kv *)b;
    return (ka->key > kb->key) - (ka->key < kb->key);
}

/* ---------- Heap for top-k ---------- */
typedef struct { float dist; uint32_t id; } zenith_hit;

ZENITH_ALWAYS_INLINE int zenith_hit_worse(zenith_hit a, zenith_hit b) {
    if (a.dist > b.dist) return 1;
    if (a.dist < b.dist) return 0;
    return a.id > b.id;
}
ZENITH_ALWAYS_INLINE int zenith_hit_better(zenith_hit a, zenith_hit b) {
    if (a.dist < b.dist) return 1;
    if (a.dist > b.dist) return 0;
    return a.id < b.id;
}

ZENITH_ALWAYS_INLINE void zenith_heap_sift(zenith_hit *h, int n, int i) {
    for (;;) {
        int l = 2*i+1, r = 2*i+2, m = i;
        if (l < n && zenith_hit_worse(h[l], h[m])) m = l;
        if (r < n && zenith_hit_worse(h[r], h[m])) m = r;
        if (m == i) break;
        { zenith_hit t = h[i]; h[i] = h[m]; h[m] = t; }
        i = m;
    }
}

#include "zenith/threads.h"

/* ---------- Query context ---------- */
typedef struct {
    zenith_hit *heap;
    int hn;
    uint32_t k, Mc;
    int use_tail;
    float w_tail_min, Rq;
    const float *Q_use, *w_use;
    uint32_t *collect_ids;
    uint32_t collect_cap, collect_n;
} zenith__qctx;

static int zenith__consider(const zenith_index_t *ZENITH_RESTRICT idx,
                            uint32_t pick, zenith__qctx *ZENITH_RESTRICT c) {
    float tail_lb = 0.0f, d;
    const float *ZENITH_RESTRICT X;
    if (c->collect_ids) {
        if (c->collect_n < c->collect_cap)
            c->collect_ids[c->collect_n++] = idx->ids[pick];
        return 1;
    }
    if (c->use_tail) {
        float dr = idx->residual[pick] - c->Rq;
        tail_lb = c->w_tail_min * dr * dr;
    }
    if (ZENITH_UNLIKELY(tail_lb > 0.0f && c->hn >= (int)c->k && tail_lb > c->heap[0].dist))
        return 0;
    X = idx->coeffs + (size_t)pick * c->Mc;
    d = zenith__wdist(c->Q_use, X, c->w_use, c->Mc) + tail_lb;
    if (c->hn < (int)c->k) {
        c->heap[c->hn].dist = d;
        c->heap[c->hn].id   = idx->ids[pick];
        c->hn++;
        if (c->hn == (int)c->k) {
            int ii;
            for (ii = (int)c->k / 2 - 1; ii >= 0; --ii)
                zenith_heap_sift(c->heap, c->hn, ii);
        }
    } else {
        zenith_hit cand;
        cand.dist = d;
        cand.id = idx->ids[pick];
        if (zenith_hit_better(cand, c->heap[0])) {
            c->heap[0] = cand;
            zenith_heap_sift(c->heap, c->hn, 0);
        }
    }
    return 1;
}

static uint32_t zenith__query_all(const zenith_index_t *ZENITH_RESTRICT idx,
                                  zenith__qctx *c) {
    uint32_t pos, scored = 0;
    int nt;
    if (c->collect_ids) {
        for (pos = 0; pos < idx->N && c->collect_n < c->collect_cap; ++pos)
            c->collect_ids[c->collect_n++] = idx->ids[pos];
        return c->collect_n;
    }
    nt = zenith__index_threads(idx);
    if (!zenith__want_parallel(idx, idx->N) || nt <= 1) {
        for (pos = 0; pos < idx->N; ++pos)
            if (zenith__consider(idx, pos, c)) ++scored;
        return scored;
    }
#if defined(_OPENMP)
#pragma omp parallel num_threads(nt)
    {
        zenith_hit *local_heap = (zenith_hit *)ZENITH_MALLOC(sizeof(zenith_hit) * c->k);
        zenith__qctx local = *c;
        int lhn = 0;
        if (local_heap) {
            local.heap = local_heap;
            local.hn = 0;
#pragma omp for schedule(static)
            for (pos = 0; pos < idx->N; ++pos)
                zenith__consider(idx, pos, &local);
            lhn = local.hn;
        }
#pragma omp critical
        {
            if (local_heap)
                zenith__heap_merge(c->heap, &c->hn, c->k, local_heap, lhn);
        }
        ZENITH_FREE(local_heap);
    }
    scored = (uint32_t)c->hn;
#else
    (void)nt;
    for (pos = 0; pos < idx->N; ++pos)
        if (zenith__consider(idx, pos, c)) ++scored;
#endif
    return scored;
}

typedef struct { uint32_t pos; uint32_t ham; } zenith__cand;

static int zenith__cand_cmp(const void *a, const void *b) {
    const zenith__cand *ca = (const zenith__cand *)a;
    const zenith__cand *cb = (const zenith__cand *)b;
    if (ca->ham != cb->ham) return (ca->ham > cb->ham) - (ca->ham < cb->ham);
    return (ca->pos > cb->pos) - (ca->pos < cb->pos);
}

static int zenith__mih_push(const zenith_index_t *ZENITH_RESTRICT idx,
                            uint32_t pos, uint64_t raw,
                            uint64_t *ZENITH_RESTRICT seen,
                            zenith__cand *ZENITH_RESTRICT cands,
                            uint32_t *ZENITH_RESTRICT n, uint32_t cap) {
    uint32_t w = pos >> 6, bit = pos & 63u;
    uint64_t mask = 1ULL << bit;
    if (seen[w] & mask) return 0;
    seen[w] |= mask;
    if (*n >= cap) return 1;
    cands[*n].pos = pos;
    cands[*n].ham = zenith__hamming(raw, zenith_ungray(idx->keys[pos]), idx->nbits);
    (*n)++;
    return *n >= cap;
}

/* Drain one MIH bucket, starting at *at (absolute table index). */
static int zenith__mih_collect_bucket(const zenith_index_t *ZENITH_RESTRICT idx,
                                      uint32_t t, uint32_t bucket, uint32_t take,
                                      uint64_t raw, uint64_t *ZENITH_RESTRICT seen,
                                      zenith__cand *cands, uint32_t *n, uint32_t cap,
                                      uint32_t *ZENITH_RESTRICT at) {
    uint32_t nb = 1u << idx->mih_bits;
    uint32_t end = idx->mih_off[(size_t)t * (nb + 1u) + bucket + 1u];
    uint32_t got = 0;
    while (*at < end && got < take) {
        uint32_t pos = idx->mih_idx[(size_t)t * idx->N + *at];
        ++(*at);
        ++got;
        if (zenith__mih_push(idx, pos, raw, seen, cands, n, cap)) return 1;
    }
    return 0;
}

/* Progressive radius expansion. Collect unique candidates round-robin across
 * chunks, rank them by full-sketch Hamming distance, then score the closest.
 * Collecting more than ef lets the Hamming ranker spend the budget on the
 * nearest sketches instead of the first overflowing bucket. */
static uint32_t zenith__query_mih(const zenith_index_t *ZENITH_RESTRICT idx,
                                  uint64_t raw, uint32_t ef,
                                  uint64_t *ZENITH_RESTRICT seen,
                                  zenith__qctx *c) {
    enum { STK = 256, MAXPR = 1024 };
    zenith__cand stk[STK];
    zenith__cand *cands = stk;
    int cands_heap = 0;
    uint32_t m = idx->mih_m, bits = idx->mih_bits, t, ncand = 0, scored = 0, i;
    uint32_t chunk[8];
    uint32_t max_r = idx->mih_max_radius ? idx->mih_max_radius : 2u;
    uint32_t cap, radius;

    if (max_r > 2u) max_r = 2u;
    if (m == 0 || m > 8u || !seen || ef == 0) return 0;

    cap = (ef > (UINT32_MAX / 8u)) ? UINT32_MAX : ef * 8u;
    if (cap < 256u) cap = 256u;
    if (cap > idx->N) cap = idx->N;
    if (cap > STK) {
        cands = (zenith__cand *)ZENITH_MALLOC(sizeof(zenith__cand) * cap);
        if (!cands) return 0;
        cands_heap = 1;
    }

    for (t = 0; t < m; ++t)
        chunk[t] = zenith__mih_chunk(raw, idx->nbits, t, bits);

    for (radius = 0; radius <= max_r && ncand < cap; ++radius) {
        uint32_t ts[MAXPR], bks[MAXPR], cur[MAXPR], npr = 0, p;
        uint32_t nb = 1u << bits;
        int progress;
        if (radius == 0) {
            for (t = 0; t < m && npr < MAXPR; ++t) {
                ts[npr] = t; bks[npr] = chunk[t];
                cur[npr] = idx->mih_off[(size_t)t * (nb + 1u) + chunk[t]];
                npr++;
            }
        } else if (radius == 1) {
            for (t = 0; t < m; ++t) {
                uint32_t cbits = zenith__mih_chunk_bits(idx->nbits, bits, t);
                uint32_t b;
                for (b = 0; b < cbits && npr < MAXPR; ++b) {
                    uint32_t nbr = chunk[t] ^ (1u << b);
                    ts[npr] = t; bks[npr] = nbr;
                    cur[npr] = idx->mih_off[(size_t)t * (nb + 1u) + nbr];
                    npr++;
                }
            }
        } else {
            for (t = 0; t < m; ++t) {
                uint32_t cbits = zenith__mih_chunk_bits(idx->nbits, bits, t);
                uint32_t b1, b2;
                for (b1 = 0; b1 < cbits; ++b1) {
                    for (b2 = b1 + 1; b2 < cbits && npr < MAXPR; ++b2) {
                        uint32_t nbr = chunk[t] ^ (1u << b1) ^ (1u << b2);
                        ts[npr] = t; bks[npr] = nbr;
                        cur[npr] = idx->mih_off[(size_t)t * (nb + 1u) + nbr];
                        npr++;
                    }
                }
            }
        }

        do {
            progress = 0;
            for (p = 0; p < npr && ncand < cap; ++p) {
                uint32_t before = ncand;
                uint32_t end = idx->mih_off[(size_t)ts[p] * (nb + 1u) + bks[p] + 1u];
                if (cur[p] >= end) continue;
                progress = 1;
                if (zenith__mih_collect_bucket(idx, ts[p], bks[p], 1u, raw, seen,
                                               cands, &ncand, cap, &cur[p]))
                    break;
                (void)before;
            }
        } while (progress && ncand < cap);
    }

    if (ncand > 1u)
        qsort(cands, ncand, sizeof(zenith__cand), zenith__cand_cmp);
    for (i = 0; i < ncand && scored < ef; ++i) {
        if (i + 1u < ncand)
            ZENITH_PREFETCH(idx->coeffs + (size_t)cands[i + 1u].pos * c->Mc);
        if (zenith__consider(idx, cands[i].pos, c))
            ++scored;
    }
    /* Leave only visited candidates marked so Gray backfill can still
     * consider Hamming-far MIH hits that we chose not to score. */
    {
        uint32_t u;
        for (u = i; u < ncand; ++u) {
            uint32_t pos = cands[u].pos;
            seen[pos >> 6] &= ~(1ULL << (pos & 63u));
        }
    }
    if (cands_heap) ZENITH_FREE(cands);
    return scored;
}

/* Select the ef best packed-code candidates by true Hamming distance. Hamming
 * and posterior modes store raw codes rather than Gray-coded keys. */
static uint32_t zenith__query_hamming(const zenith_index_t *ZENITH_RESTRICT idx,
                                      uint64_t query_raw,
                                      uint64_t query_mag,
                                      uint32_t ef,
                                      zenith__qctx *ZENITH_RESTRICT qc) {
    uint32_t hist[129];
    uint32_t pos, scored = 0, d, threshold = 0, tie_budget = 0;
    uint32_t max_dist = idx->quantizer == ZENITH_QUANT_SIGN_MAG ?
                        2u * idx->nbits : idx->nbits;
    uint64_t acc = 0;
    if (ef == 0) return 0;
    if (max_dist > 128u) max_dist = 128u;
    memset(hist, 0, sizeof(hist));
#if defined(_OPENMP)
    if (zenith__want_parallel(idx, idx->N)) {
#pragma omp parallel for num_threads(zenith__index_threads(idx)) schedule(static) reduction(+:hist[:129])
        for (pos = 0; pos < idx->N; ++pos) {
            uint32_t dd = zenith__popcount64(idx->keys[pos] ^ query_raw);
            if (idx->quantizer == ZENITH_QUANT_SIGN_MAG)
                dd += zenith__popcount64(idx->mag_keys[pos] ^ query_mag);
            if (dd > max_dist) dd = max_dist;
            hist[dd]++;
        }
    } else
#endif
    for (pos = 0; pos < idx->N; ++pos) {
        d = zenith__popcount64(idx->keys[pos] ^ query_raw);
        if (idx->quantizer == ZENITH_QUANT_SIGN_MAG)
            d += zenith__popcount64(idx->mag_keys[pos] ^ query_mag);
        if (d > max_dist) d = max_dist;
        ++hist[d];
    }
    while (threshold < max_dist && acc + hist[threshold] < ef) {
        acc += hist[threshold];
        ++threshold;
    }
    tie_budget = (uint32_t)(ef - acc);

    for (pos = 0; pos < idx->N && scored < ef; ++pos) {
        d = zenith__popcount64(idx->keys[pos] ^ query_raw);
        if (idx->quantizer == ZENITH_QUANT_SIGN_MAG)
            d += zenith__popcount64(idx->mag_keys[pos] ^ query_mag);
        if (d < threshold && zenith__consider(idx, pos, qc))
            ++scored;
    }
    for (pos = 0; pos < idx->N && scored < ef && tie_budget > 0; ++pos) {
        d = zenith__popcount64(idx->keys[pos] ^ query_raw);
        if (idx->quantizer == ZENITH_QUANT_SIGN_MAG)
            d += zenith__popcount64(idx->mag_keys[pos] ^ query_mag);
        if (d == threshold && zenith__consider(idx, pos, qc)) {
            ++scored;
            --tie_budget;
        }
    }
    return scored;
}

static void zenith__lm2_moments(uint32_t symbol, double *level, double *e2) {
    switch (symbol & 3u) {
    case 0u: *level = -ZENITH_LM2_OUTER; *e2 = ZENITH_LM2_E2_OUTER; break;
    case 1u: *level = -ZENITH_LM2_INNER; *e2 = ZENITH_LM2_E2_INNER; break;
    case 3u: *level =  ZENITH_LM2_INNER; *e2 = ZENITH_LM2_E2_INNER; break;
    default: *level =  ZENITH_LM2_OUTER; *e2 = ZENITH_LM2_E2_OUTER; break;
    }
}

static double zenith__lm_coordinate_score(const zenith_index_t *ZENITH_RESTRICT idx,
                                          const float *ZENITH_RESTRICT Q,
                                          const float *ZENITH_RESTRICT weight,
                                          uint32_t b, double level,
                                          double e2_unit) {
    uint32_t coef = idx->use_perm ? (uint32_t)idx->freq_perm[b] : b;
    double w = coef < idx->Mcoef ? (double)weight[coef] : 1.0;
    double q = (double)Q[coef];
    double bias = (double)idx->quant_bias[b];
    double scale = (double)idx->quant_scale[b];
    double mean = bias + scale * level;
    double e2 = bias * bias + 2.0 * bias * scale * level +
                scale * scale * e2_unit;
    return w * (2.0 * q * mean - e2);
}

static double zenith__lm_symbol_score(const zenith_index_t *ZENITH_RESTRICT idx,
                                      const float *ZENITH_RESTRICT Q,
                                      const float *ZENITH_RESTRICT weight,
                                      uint32_t b, uint32_t symbol) {
    double level, e2;
    zenith__lm2_moments(symbol, &level, &e2);
    return zenith__lm_coordinate_score(idx, Q, weight, b, level, e2);
}

static double zenith__lm_signmag_score(const zenith_index_t *ZENITH_RESTRICT idx,
                                       const float *ZENITH_RESTRICT Q,
                                       const float *ZENITH_RESTRICT weight,
                                       uint32_t b, uint32_t sign_bit,
                                       uint32_t mag_bit) {
    double level = mag_bit ? ZENITH_LM2_OUTER : ZENITH_LM2_INNER;
    double e2 = mag_bit ? ZENITH_LM2_E2_OUTER : ZENITH_LM2_E2_INNER;
    if (!sign_bit) level = -level;
    return zenith__lm_coordinate_score(idx, Q, weight, b, level, e2);
}

static uint32_t zenith__query_lm_posterior(const zenith_index_t *ZENITH_RESTRICT idx,
                                           const float *ZENITH_RESTRICT Q,
                                           const float *ZENITH_RESTRICT weight,
                                           uint32_t ef,
                                           zenith__qctx *ZENITH_RESTRICT qc) {
    enum { POSTERIOR_BINS = 2048 };
    double lut[16][256];
    double total_abs = 0.0;
    uint16_t *score_bin;
    uint32_t hist[POSTERIOR_BINS];
    uint32_t nchunks, chunk, value, pos, scored = 0, b;
    uint32_t above = 0, tie_budget = 0;
    int threshold = 0, bi;

    if (ef == 0) return 0;
    if (!idx->quant_scale || !idx->quant_bias ||
        (idx->quantizer == ZENITH_QUANT_SIGN_MAG && !idx->mag_keys)) {
        zenith__set_error(ZENITH_ERR_FORMAT);
        return 0;
    }

    nchunks = (idx->ncodes + 3u) / 4u;
    if (nchunks > 16u) nchunks = 16u;

    for (b = 0; b < idx->ncodes; ++b) {
        double max_abs = 0.0;
        uint32_t symbol;
        for (symbol = 0; symbol < 4u; ++symbol) {
            double s = idx->quantizer == ZENITH_QUANT_LM2 ?
                zenith__lm_symbol_score(idx, Q, weight, b, symbol) :
                zenith__lm_signmag_score(idx, Q, weight, b,
                                         symbol & 1u, symbol >> 1u);
            double a = fabs(s);
            if (a > max_abs) max_abs = a;
        }
        total_abs += max_abs;
    }

    for (chunk = 0; chunk < nchunks; ++chunk) {
        for (value = 0; value < 256u; ++value) {
            double s = 0.0;
            uint32_t j;
            if (idx->quantizer == ZENITH_QUANT_LM2) {
                for (j = 0; j < 4u; ++j) {
                    uint32_t cb = 4u * chunk + j;
                    if (cb >= idx->ncodes) break;
                    s += zenith__lm_symbol_score(idx, Q, weight, cb,
                                                 (value >> (2u * j)) & 3u);
                }
            } else {
                for (j = 0; j < 4u; ++j) {
                    uint32_t cb = 4u * chunk + j;
                    if (cb >= idx->ncodes) break;
                    s += zenith__lm_signmag_score(idx, Q, weight, cb,
                                                  (value >> j) & 1u,
                                                  (value >> (4u + j)) & 1u);
                }
            }
            lut[chunk][value] = s;
        }
    }

    score_bin = (uint16_t *)ZENITH_MALLOC(sizeof(uint16_t) * (size_t)idx->N);
    if (!score_bin) {
        zenith__set_error(ZENITH_ERR_ALLOC);
        return 0;
    }
    memset(hist, 0, sizeof(hist));
    for (pos = 0; pos < idx->N; ++pos) {
        double score = 0.0;
        uint32_t bin;
        for (chunk = 0; chunk < nchunks; ++chunk) {
            if (idx->quantizer == ZENITH_QUANT_LM2) {
                uint32_t piece = (uint32_t)((idx->keys[pos] >> (8u * chunk)) & 255u);
                score += lut[chunk][piece];
            } else {
                uint32_t sp = (uint32_t)((idx->keys[pos] >> (4u * chunk)) & 15u);
                uint32_t mp = (uint32_t)((idx->mag_keys[pos] >> (4u * chunk)) & 15u);
                score += lut[chunk][sp | (mp << 4u)];
            }
        }
        if (total_abs > 0.0) {
            double u = (score + total_abs) *
                       ((double)(POSTERIOR_BINS - 1) / (2.0 * total_abs));
            if (u < 0.0) u = 0.0;
            if (u > (double)(POSTERIOR_BINS - 1)) u = POSTERIOR_BINS - 1;
            bin = (uint32_t)u;
        } else {
            bin = POSTERIOR_BINS / 2u;
        }
        score_bin[pos] = (uint16_t)bin;
        ++hist[bin];
    }

    for (bi = POSTERIOR_BINS - 1; bi >= 0; --bi) {
        if (above + hist[bi] >= ef) {
            threshold = bi;
            tie_budget = ef - above;
            break;
        }
        above += hist[bi];
    }

    for (pos = 0; pos < idx->N && scored < ef; ++pos) {
        if ((int)score_bin[pos] > threshold &&
            zenith__consider(idx, pos, qc))
            ++scored;
    }
    for (pos = 0; pos < idx->N && scored < ef && tie_budget > 0; ++pos) {
        if ((int)score_bin[pos] == threshold &&
            zenith__consider(idx, pos, qc)) {
            ++scored;
            --tie_budget;
        }
    }
    ZENITH_FREE(score_bin);
    return scored;
}

static uint32_t zenith__query_posterior(const zenith_index_t *ZENITH_RESTRICT idx,
                                        const float *ZENITH_RESTRICT Q,
                                        const float *ZENITH_RESTRICT Q_use,
                                        const float *ZENITH_RESTRICT w_use,
                                        uint32_t ef,
                                        zenith__qctx *ZENITH_RESTRICT qc) {
    enum { POSTERIOR_BINS = 2048 };
    double lut[8][256];
    double a_bit[64];
    double total_abs = 0.0;
    uint16_t *score_bin;
    uint32_t hist[POSTERIOR_BINS];
    uint32_t b, chunk, value, pos, scored = 0;
    uint32_t above = 0, tie_budget = 0;
    int threshold = 0, bi;

    if (ef == 0) return 0;
    memset(a_bit, 0, sizeof(a_bit));
    for (b = 0; b < idx->nbits; ++b) {
        uint32_t coef = idx->use_perm ? (uint32_t)idx->freq_perm[b] : b;
        double a;
        if (coef < idx->Mcoef)
            a = (double)w_use[coef] * (double)Q_use[coef];
        else
            a = (double)Q[coef];
        a_bit[b] = a;
        total_abs += fabs(a);
    }
    for (chunk = 0; chunk < 8u; ++chunk) {
        for (value = 0; value < 256u; ++value) {
            double s = 0.0;
            uint32_t bit;
            for (bit = 0; bit < 8u; ++bit) {
                b = chunk * 8u + bit;
                if (b >= idx->nbits) break;
                s += (value & (1u << bit)) ? a_bit[b] : -a_bit[b];
            }
            lut[chunk][value] = s;
        }
    }

    score_bin = (uint16_t *)ZENITH_MALLOC(sizeof(uint16_t) * (size_t)idx->N);
    if (!score_bin) {
        zenith__set_error(ZENITH_ERR_ALLOC);
        return 0;
    }
    memset(hist, 0, sizeof(hist));
#if defined(_OPENMP)
#pragma omp parallel for num_threads(zenith__index_threads(idx)) schedule(static) reduction(+:hist[:2048]) if(zenith__want_parallel(idx, idx->N))
#endif
    for (pos = 0; pos < idx->N; ++pos) {
        uint64_t raw = idx->keys[pos];
        double score = lut[0][raw & 255u]
                     + lut[1][(raw >> 8) & 255u]
                     + lut[2][(raw >> 16) & 255u]
                     + lut[3][(raw >> 24) & 255u]
                     + lut[4][(raw >> 32) & 255u]
                     + lut[5][(raw >> 40) & 255u]
                     + lut[6][(raw >> 48) & 255u]
                     + lut[7][(raw >> 56) & 255u];
        uint32_t bin;
        if (total_abs > 0.0) {
            double u = (score + total_abs) *
                       ((double)(POSTERIOR_BINS - 1) / (2.0 * total_abs));
            if (u < 0.0) u = 0.0;
            if (u > (double)(POSTERIOR_BINS - 1)) u = POSTERIOR_BINS - 1;
            bin = (uint32_t)u;
        } else {
            bin = POSTERIOR_BINS / 2u;
        }
        score_bin[pos] = (uint16_t)bin;
        ++hist[bin];
    }

    for (bi = POSTERIOR_BINS - 1; bi >= 0; --bi) {
        if (above + hist[bi] >= ef) {
            threshold = bi;
            tie_budget = ef - above;
            break;
        }
        above += hist[bi];
    }

    for (pos = 0; pos < idx->N && scored < ef; ++pos) {
        if ((int)score_bin[pos] > threshold &&
            zenith__consider(idx, pos, qc))
            ++scored;
    }
    for (pos = 0; pos < idx->N && scored < ef && tie_budget > 0; ++pos) {
        if ((int)score_bin[pos] == threshold &&
            zenith__consider(idx, pos, qc)) {
            ++scored;
            --tie_budget;
        }
    }
    ZENITH_FREE(score_bin);
    return scored;
}

/* Lower bound binary search on sorted Gray keys */
static int64_t zenith_lower_pos(const uint64_t *keys, uint32_t N, uint64_t g) {
    int64_t lo = 0, hi = (int64_t)N;
    while (lo < hi) {
        int64_t mid = lo + (hi - lo) / 2;
        if (keys[mid] < g) lo = mid + 1;
        else hi = mid;
    }
    return lo;
}

/* Bidirectional Gray scan.  With ef=N it is exhaustive over sorted positions. */
static uint32_t zenith__query_gray(const zenith_index_t *ZENITH_RESTRICT idx,
                                   uint64_t g, uint32_t ef,
                                   uint64_t *ZENITH_RESTRICT seen,
                                   zenith__qctx *c) {
    int64_t lo, hi, center;
    uint32_t evaluated = 0, visited = 0;
    center = zenith_lower_pos(idx->keys, idx->N, g);
    lo = center - 1; /* nearest predecessor */
    hi = center;     /* first >= g; exact when present */
    while (evaluated < ef && visited < idx->N && (lo >= 0 || hi < (int64_t)idx->N)) {
        int64_t pick;
        uint32_t pos;
        if (lo < 0)                     pick = hi++;
        else if (hi >= (int64_t)idx->N) pick = lo--;
        else {
            uint64_t dl = g - idx->keys[lo];
            uint64_t dh = idx->keys[hi] - g;
            if (dl <= dh) pick = lo--; else pick = hi++;
        }
        pos = (uint32_t)pick;
        ++visited;
        if (seen) {
            uint32_t w = pos >> 6, bit = pos & 63u;
            uint64_t mask = 1ULL << bit;
            if (seen[w] & mask) continue;
            seen[w] |= mask;
        }
        if (hi < (int64_t)idx->N)
            ZENITH_PREFETCH(idx->coeffs + (size_t)hi * c->Mc);
        if (lo >= 0)
            ZENITH_PREFETCH(idx->coeffs + (size_t)lo * c->Mc);
        if (zenith__consider(idx, pos, c))
            ++evaluated;
    }
    return evaluated;
}

/* ---------- Build ---------- */
zenith_index_t *zenith_build(const float *ZENITH_RESTRICT vectors,
                             uint32_t N, uint32_t D, zenith_opts opts) {
    zenith_index_t *idx;
    uint32_t Mcoef, nbits, ncodes, i, Mtab;
    zenith_quantizer_t quantizer;
    float *X, *coeff_by_id, *res_by_id, *lm_values;
    double *spec_energy;
    zenith__kv *kv;
    int rc;

    if (!vectors || N == 0 || D == 0) {
        zenith__set_error(ZENITH_ERR_ARG);
        return NULL;
    }
    if (opts.gen != ZENITH_GEN_DCT && opts.gen != ZENITH_GEN_SIGNED_DCT) {
        zenith__set_error(ZENITH_ERR_ARG);
        return NULL;
    }
    if (opts.search != ZENITH_SEARCH_MIH && opts.search != ZENITH_SEARCH_GRAY &&
        opts.search != ZENITH_SEARCH_HYBRID &&
        opts.search != ZENITH_SEARCH_HAMMING &&
        opts.search != ZENITH_SEARCH_POSTERIOR) {
        zenith__set_error(ZENITH_ERR_ARG);
        return NULL;
    }
    quantizer = opts.quantizer;
    if (quantizer != ZENITH_QUANT_SIGN && quantizer != ZENITH_QUANT_LM2 &&
        quantizer != ZENITH_QUANT_SIGN_MAG) {
        zenith__set_error(ZENITH_ERR_ARG);
        return NULL;
    }
    if (opts.search == ZENITH_SEARCH_POSTERIOR && opts.use_polar) {
        zenith__set_error(ZENITH_ERR_ARG);
        return NULL;
    }
    if (quantizer != ZENITH_QUANT_SIGN &&
        (opts.use_polar ||
         (opts.search != ZENITH_SEARCH_HAMMING &&
          opts.search != ZENITH_SEARCH_POSTERIOR))) {
        zenith__set_error(ZENITH_ERR_ARG);
        return NULL;
    }

    Mcoef = opts.Mcoef ? opts.Mcoef : (D < 128u ? D : 128u);
    if (Mcoef > D) Mcoef = D;
    if ((size_t)N > ((size_t)-1) / (size_t)Mcoef / sizeof(float)) {
        zenith__set_error(ZENITH_ERR_ALLOC);
        return NULL;
    }
    nbits = opts.nbits ? opts.nbits : ZENITH_SKETCH_BITS;
    if (quantizer == ZENITH_QUANT_LM2) {
        uint32_t max_bits = D > 32u ? 64u : 2u * D;
        if (nbits > max_bits) nbits = max_bits;
        nbits &= ~1u;
        if (nbits < 2u) {
            zenith__set_error(ZENITH_ERR_ARG);
            return NULL;
        }
        ncodes = nbits / 2u;
    } else {
        if (nbits > 64u) nbits = 64u;
        if (nbits > D) nbits = D;
        ncodes = nbits;
    }
    if (opts.auto_perm) opts.use_perm = 1;
    if (opts.use_perm) {
        if (Mcoef > 256u) {
            zenith__set_error(ZENITH_ERR_ARG);
            return NULL;
        }
        if (ncodes > Mcoef) {
            ncodes = Mcoef;
            nbits = quantizer == ZENITH_QUANT_LM2 ? 2u * ncodes : ncodes;
        }
    }
    if (opts.use_polar) {
        if (nbits < 2u) opts.use_polar = 0;
        else nbits &= ~1u;
        ncodes = nbits;
    }
    if (nbits == 0u) {
        nbits = 1u;
        ncodes = 1u;
    }

    /* user permutation validation: bounded and duplicate-free */
    if (opts.use_perm && !opts.auto_perm) {
        uint32_t b, j;
        for (b = 0; b < ncodes && b < 64u; ++b) {
            if (opts.freq_perm[b] >= Mcoef) {
                zenith__set_error(ZENITH_ERR_PERM);
                return NULL;
            }
            for (j = 0; j < b; ++j) {
                if (opts.freq_perm[j] == opts.freq_perm[b]) {
                    zenith__set_error(ZENITH_ERR_PERM);
                    return NULL;
                }
            }
        }
    }

    idx = (zenith_index_t *)ZENITH_MALLOC(sizeof(*idx));
    if (!idx) {
        zenith__set_error(ZENITH_ERR_ALLOC);
        return NULL;
    }
    memset(idx, 0, sizeof(*idx));
    idx->N = N; idx->D = D; idx->Mcoef = Mcoef; idx->nbits = nbits;
    idx->ncodes = ncodes;
    idx->quantizer = (int)quantizer;
    idx->use_polar = opts.use_polar;
    idx->use_perm  = opts.use_perm;
    idx->mih_max_radius = opts.mih_max_radius ? opts.mih_max_radius : 2u;
    if (idx->mih_max_radius > 2u) idx->mih_max_radius = 2u;
    if (opts.use_perm && !opts.auto_perm)
        memcpy(idx->freq_perm, opts.freq_perm, ncodes);

    idx->gen = (int)opts.gen;
    idx->gen_seed = opts.seed;
    if (opts.gen == ZENITH_GEN_SIGNED_DCT) {
        idx->input_signs = (int8_t *)ZENITH_MALLOC(sizeof(int8_t) * D);
        if (!idx->input_signs) {
            ZENITH_FREE(idx);
            zenith__set_error(ZENITH_ERR_ALLOC);
            return NULL;
        }
        zenith__gen_input_signs(idx->input_signs, D, idx->gen_seed);
    }

    Mtab = Mcoef > ncodes ? Mcoef : ncodes;
    idx->nthreads = opts.nthreads > 0 ? opts.nthreads : zenith_thread_count();
    rc = zenith_dct_plan_init(idx, (int)D, (int)Mtab, opts.fftw);
    if (rc != ZENITH_OK) {
        ZENITH_FREE(idx->input_signs);
        ZENITH_FREE(idx);
        zenith__set_error((zenith_err_t)rc);
        return NULL;
    }
    zenith_dct_plan_apply_signs(idx);

    if (opts.whiten_baseline) {
        uint32_t k;
        for (k = 0; k < Mcoef; ++k)
            if (!isfinite(opts.whiten_baseline[k]) || opts.whiten_baseline[k] <= 0.0f) {
                ZENITH_FREE(idx->input_signs);
                zenith_dct_plan_free(idx); ZENITH_FREE(idx);
                zenith__set_error(ZENITH_ERR_WEIGHT);
                return NULL;
            }
        idx->use_whiten = 1;
        idx->whiten_sq = (float *)ZENITH_MALLOC(sizeof(float) * Mcoef);
        if (!idx->whiten_sq) {
            ZENITH_FREE(idx->input_signs);
            zenith_dct_plan_free(idx); ZENITH_FREE(idx);
            zenith__set_error(ZENITH_ERR_ALLOC);
            return NULL;
        }
        for (k = 0; k < Mcoef; ++k)
            idx->whiten_sq[k] = sqrtf(opts.whiten_baseline[k]);
    }

    X = (float *)ZENITH_MALLOC(sizeof(float) * idx->plan_M);
    coeff_by_id = (float *)ZENITH_MALLOC(sizeof(float) * (size_t)N * Mcoef);
    res_by_id   = (float *)ZENITH_MALLOC(sizeof(float) * N);
    kv          = (zenith__kv *)ZENITH_MALLOC(sizeof(zenith__kv) * N);
    spec_energy = opts.auto_perm ? (double *)ZENITH_MALLOC(sizeof(double) * Mcoef) : NULL;
    lm_values   = NULL;
    if (spec_energy) memset(spec_energy, 0, sizeof(double) * Mcoef);
    if (quantizer != ZENITH_QUANT_SIGN) {
        if ((size_t)N > ((size_t)-1) / (size_t)ncodes / sizeof(float)) {
            ZENITH_FREE(X); ZENITH_FREE(coeff_by_id); ZENITH_FREE(res_by_id);
            ZENITH_FREE(kv); ZENITH_FREE(spec_energy); zenith_free(idx);
            zenith__set_error(ZENITH_ERR_ALLOC);
            return NULL;
        }
        idx->quant_scale = (float *)ZENITH_MALLOC(sizeof(float) * ncodes);
        idx->quant_bias  = (float *)ZENITH_MALLOC(sizeof(float) * ncodes);
        lm_values = (float *)ZENITH_MALLOC(sizeof(float) * (size_t)N * ncodes);
        if (!idx->quant_scale || !idx->quant_bias || !lm_values) {
            ZENITH_FREE(X); ZENITH_FREE(coeff_by_id); ZENITH_FREE(res_by_id);
            ZENITH_FREE(kv); ZENITH_FREE(spec_energy); ZENITH_FREE(lm_values);
            zenith_free(idx);
            zenith__set_error(ZENITH_ERR_ALLOC);
            return NULL;
        }
    }
    if (!X || !coeff_by_id || !res_by_id || !kv || (opts.auto_perm && !spec_energy)) {
        ZENITH_FREE(X); ZENITH_FREE(coeff_by_id); ZENITH_FREE(res_by_id); ZENITH_FREE(kv); ZENITH_FREE(spec_energy);
        ZENITH_FREE(lm_values);
        zenith_free(idx);
        zenith__set_error(ZENITH_ERR_ALLOC);
        return NULL;
    }

    {
        int bad = 0;
#if defined(_OPENMP)
        int nt = idx->nthreads > 0 ? idx->nthreads : 1;
#pragma omp parallel num_threads(nt)
#endif
        {
            float *tlX = X;
            double *local_e = NULL;
#if defined(_OPENMP)
            tlX = (float *)ZENITH_MALLOC(sizeof(float) * idx->plan_M);
            if (spec_energy) {
                local_e = (double *)ZENITH_MALLOC(sizeof(double) * Mcoef);
                if (local_e) memset(local_e, 0, sizeof(double) * Mcoef);
            }
#endif
            if (!tlX
#if defined(_OPENMP)
                || (spec_energy && !local_e)
#endif
                ) {
#if defined(_OPENMP)
#pragma omp atomic write
#endif
                bad = 1;
            } else {
                uint32_t i;
#if defined(_OPENMP)
#pragma omp for schedule(static)
#endif
                for (i = 0; i < N; ++i) {
                    const float *ZENITH_RESTRICT vi = vectors + (size_t)i * D;
                    double norm_sq, lead_sq, energy;
                    uint32_t k;
                    uint64_t h, g;
                    if (bad) continue;
                    zenith_dct_lead(idx, vi, tlX);
                    norm_sq = 0.0;
                    for (k = 0; k < D; ++k) norm_sq += (double)vi[k] * (double)vi[k];
                    lead_sq = 0.0;
                    for (k = 0; k < Mcoef; ++k) lead_sq += (double)tlX[k] * (double)tlX[k];
                    if (!isfinite(norm_sq) || !isfinite(lead_sq)) {
#if defined(_OPENMP)
#pragma omp atomic write
#endif
                        bad = 1;
                        continue;
                    }
                    energy = norm_sq - lead_sq;
                    res_by_id[i] = (energy > 0.0) ? (float)sqrt(energy) : 0.0f;
                    for (k = 0; k < Mcoef; ++k) {
                        float c = tlX[k];
                        if (local_e) local_e[k] += (double)c * (double)c;
                        else if (spec_energy) spec_energy[k] += (double)c * (double)c;
                        if (idx->use_whiten) c *= idx->whiten_sq[k];
                        coeff_by_id[(size_t)i * Mcoef + k] = c;
                    }
                    kv[i].id = i;
                    kv[i].key = 0;
                    kv[i].mag_key = 0;
                    if (!spec_energy && quantizer == ZENITH_QUANT_SIGN) {
                        if (idx->use_polar)
                            h = zenith_sketch_polar(tlX, (int)(nbits / 2),
                                                    idx->use_perm ? idx->freq_perm : NULL);
                        else
                            h = zenith_sketch(tlX, (int)nbits,
                                              idx->use_perm ? idx->freq_perm : NULL);
                        g = (opts.search == ZENITH_SEARCH_HAMMING ||
                             opts.search == ZENITH_SEARCH_POSTERIOR) ? h : zenith_gray(h);
                        kv[i].key = g;
                    }
                }
#if defined(_OPENMP)
                if (local_e && spec_energy) {
                    uint32_t k;
#pragma omp critical
                    for (k = 0; k < Mcoef; ++k) spec_energy[k] += local_e[k];
                }
                if (tlX != X) ZENITH_FREE(tlX);
                ZENITH_FREE(local_e);
#endif
            }
        }
        if (bad) {
            ZENITH_FREE(X); ZENITH_FREE(coeff_by_id); ZENITH_FREE(res_by_id);
            ZENITH_FREE(kv); ZENITH_FREE(spec_energy); ZENITH_FREE(lm_values);
            zenith_free(idx);
            zenith__set_error(ZENITH_ERR_ARG);
            return NULL;
        }
    }

    if (spec_energy) {
        rc = zenith__auto_perm(spec_energy, Mcoef, ncodes, idx->freq_perm);
        if (rc != ZENITH_OK) {
            ZENITH_FREE(X); ZENITH_FREE(coeff_by_id); ZENITH_FREE(res_by_id);
            ZENITH_FREE(kv); ZENITH_FREE(spec_energy); ZENITH_FREE(lm_values);
            zenith_free(idx);
            zenith__set_error((zenith_err_t)rc);
            return NULL;
        }
        if (quantizer == ZENITH_QUANT_SIGN) {
            for (i = 0; i < N; ++i) {
                const float *row = coeff_by_id + (size_t)i * Mcoef;
                uint64_t hs = idx->use_polar ? zenith_sketch_polar(row, (int)(nbits / 2), idx->freq_perm)
                                             : zenith_sketch(row, (int)nbits, idx->freq_perm);
                kv[i].key = (opts.search == ZENITH_SEARCH_HAMMING ||
                             opts.search == ZENITH_SEARCH_POSTERIOR) ? hs : zenith_gray(hs);
            }
        }
        ZENITH_FREE(spec_energy);
        spec_energy = NULL;
    }

    if (quantizer != ZENITH_QUANT_SIGN) {
        double *sum = (double *)ZENITH_MALLOC(sizeof(double) * ncodes);
        double *ssq = (double *)ZENITH_MALLOC(sizeof(double) * ncodes);
        if (!sum || !ssq) {
            ZENITH_FREE(sum); ZENITH_FREE(ssq);
            ZENITH_FREE(X); ZENITH_FREE(coeff_by_id); ZENITH_FREE(res_by_id);
            ZENITH_FREE(kv); ZENITH_FREE(lm_values); zenith_free(idx);
            zenith__set_error(ZENITH_ERR_ALLOC);
            return NULL;
        }
        memset(sum, 0, sizeof(double) * ncodes);
        memset(ssq, 0, sizeof(double) * ncodes);
#if defined(_OPENMP)
#pragma omp parallel num_threads(idx->nthreads > 0 ? idx->nthreads : 1)
        {
            float *tlX = (float *)ZENITH_MALLOC(sizeof(float) * idx->plan_M);
            double *lsum = (double *)ZENITH_MALLOC(sizeof(double) * ncodes);
            double *lssq = (double *)ZENITH_MALLOC(sizeof(double) * ncodes);
            if (tlX && lsum && lssq) {
                uint32_t i, b;
                memset(lsum, 0, sizeof(double) * ncodes);
                memset(lssq, 0, sizeof(double) * ncodes);
#pragma omp for schedule(static)
                for (i = 0; i < N; ++i) {
                    zenith_dct_lead(idx, vectors + (size_t)i * D, tlX);
                    for (b = 0; b < ncodes; ++b) {
                        uint32_t coef = idx->use_perm ? (uint32_t)idx->freq_perm[b] : b;
                        double v = tlX[coef];
                        lm_values[(size_t)i * ncodes + b] = (float)v;
                        lsum[b] += v;
                        lssq[b] += v * v;
                    }
                }
#pragma omp critical
                for (b = 0; b < ncodes; ++b) {
                    sum[b] += lsum[b];
                    ssq[b] += lssq[b];
                }
            }
            ZENITH_FREE(tlX); ZENITH_FREE(lsum); ZENITH_FREE(lssq);
        }
#else
        for (i = 0; i < N; ++i) {
            uint32_t b;
            zenith_dct_lead(idx, vectors + (size_t)i * D, X);
            for (b = 0; b < ncodes; ++b) {
                uint32_t coef = idx->use_perm ? (uint32_t)idx->freq_perm[b] : b;
                double v = X[coef];
                lm_values[(size_t)i * ncodes + b] = (float)v;
                sum[b] += v;
                ssq[b] += v * v;
            }
        }
#endif
        for (i = 0; i < ncodes; ++i) {
            double mean = sum[i] / (double)N;
            double var = ssq[i] / (double)N - mean * mean;
            if (!isfinite(mean)) mean = 0.0;
            if (!isfinite(var) || var < 1e-24) var = 1e-24;
            idx->quant_bias[i] = (float)mean;
            idx->quant_scale[i] = (float)sqrt(var);
        }
        for (i = 0; i < N; ++i) {
            const float *row = lm_values + (size_t)i * ncodes;
            if (quantizer == ZENITH_QUANT_LM2) {
                kv[i].key = zenith_sketch_lm2(row, (int)ncodes, NULL,
                                              idx->quant_scale, idx->quant_bias);
            } else {
                kv[i].key = zenith_sketch_sign_calibrated(row, (int)ncodes, NULL,
                                                          idx->quant_bias);
                kv[i].mag_key = zenith_sketch_magnitude(row, (int)nbits, NULL,
                                                        idx->quant_scale, idx->quant_bias);
            }
        }
        ZENITH_FREE(sum); ZENITH_FREE(ssq);
        ZENITH_FREE(lm_values); lm_values = NULL;
    }

    qsort(kv, N, sizeof(zenith__kv), zenith__kv_cmp);

    idx->keys     = (uint64_t *)ZENITH_ALIGNED_ALLOC(64, sizeof(uint64_t) * N);
    idx->ids      = (uint32_t *)ZENITH_ALIGNED_ALLOC(64, sizeof(uint32_t) * N);
    idx->coeffs   = (float *)ZENITH_ALIGNED_ALLOC(64, sizeof(float) * (size_t)N * Mcoef);
    idx->residual = (float *)ZENITH_ALIGNED_ALLOC(64, sizeof(float) * N);
    if (quantizer == ZENITH_QUANT_SIGN_MAG)
        idx->mag_keys = (uint64_t *)ZENITH_ALIGNED_ALLOC(64, sizeof(uint64_t) * N);
    if (!idx->keys || !idx->ids || !idx->coeffs || !idx->residual ||
        (quantizer == ZENITH_QUANT_SIGN_MAG && !idx->mag_keys)) {
        ZENITH_FREE(X); ZENITH_FREE(coeff_by_id); ZENITH_FREE(res_by_id); ZENITH_FREE(kv);
        zenith_free(idx);
        zenith__set_error(ZENITH_ERR_ALLOC);
        return NULL;
    }
#if defined(_OPENMP)
#pragma omp parallel for num_threads(idx->nthreads > 0 ? idx->nthreads : 1) schedule(static)
#endif
    for (i = 0; i < N; ++i) {
        uint32_t src = kv[i].id;
        idx->keys[i]     = kv[i].key;
        if (idx->mag_keys)
            idx->mag_keys[i] = kv[i].mag_key;
        idx->ids[i]      = src;
        idx->residual[i] = res_by_id[src];
        memcpy(idx->coeffs + (size_t)i * Mcoef,
               coeff_by_id + (size_t)src * Mcoef,
               sizeof(float) * Mcoef);
    }

    ZENITH_FREE(X); ZENITH_FREE(coeff_by_id); ZENITH_FREE(res_by_id); ZENITH_FREE(kv);

    if (opts.search == ZENITH_SEARCH_GRAY ||
        opts.search == ZENITH_SEARCH_HAMMING ||
        opts.search == ZENITH_SEARCH_POSTERIOR) {
        idx->search = (int)opts.search;
        idx->mih_m = 0;
        idx->mih_bits = 0;
    } else {
        rc = zenith__mih_build(idx);
        if (rc != ZENITH_OK) {
            zenith_free(idx);
            zenith__set_error((zenith_err_t)rc);
            return NULL;
        }
        idx->search = (int)opts.search;
    }
    zenith__set_error(ZENITH_OK);
    return idx;
}

void zenith_free(zenith_index_t *idx) {
    if (!idx) return;
    zenith_dct_plan_free(idx);
    ZENITH_FREE(idx->input_signs);
    if (idx->mmapped) {
#if defined(ZENITH_HAVE_MMAP)
        if (idx->map_base)
            munmap(idx->map_base, idx->map_len);
#endif
    } else {
        ZENITH_FREE(idx->whiten_sq);
        ZENITH_FREE(idx->quant_scale);
        ZENITH_FREE(idx->quant_bias);
        ZENITH_ALIGNED_FREE(idx->residual);
        ZENITH_ALIGNED_FREE(idx->mag_keys);
        ZENITH_ALIGNED_FREE(idx->keys);
        ZENITH_ALIGNED_FREE(idx->ids);
        ZENITH_ALIGNED_FREE(idx->coeffs);
        ZENITH_ALIGNED_FREE(idx->mih_off);
        ZENITH_ALIGNED_FREE(idx->mih_idx);
    }
    ZENITH_FREE(idx);
}

/* ---------- Core query (shared by query_ef and query_coeffs) ---------- */
static uint32_t zenith__query_core(const zenith_index_t *ZENITH_RESTRICT idx,
                                   const float *ZENITH_RESTRICT Q,
                                   float Rq,
                                   const float *ZENITH_RESTRICT weight,
                                   uint32_t k, uint32_t ef,
                                   uint32_t *ZENITH_RESTRICT out_ids,
                                   float *ZENITH_RESTRICT out_dist,
                                   int collect_candidates) {
    ZENITH_ALIGN64 float Qw_stk[1024];
    ZENITH_ALIGN64 float we_stk[1024];
    zenith_hit heap_stk[256];
    uint64_t seen_stk[1024];
    uint64_t *seen = NULL;
    float *Q_white = NULL, *w_eff = NULL;
    int heap_on_heap = 0, white_on_heap = 0, seen_heap = 0;
    const float *Q_use, *w_use;
    uint64_t h, g, query_mag;
    uint32_t Mc, count, r;
    zenith_hit *heap;
    float w_tail_min = 0.0f;
    int use_tail;
    zenith__qctx qc;
    uint32_t wi;

    if (!idx || !Q || !weight || !out_ids || k == 0) {
        zenith__set_error(ZENITH_ERR_ARG);
        return 0;
    }
    zenith__set_error(ZENITH_OK);
    if (k > idx->N) k = idx->N;
    if (ef < k) ef = k;
    if (ef > idx->N) ef = idx->N;
    Mc = idx->Mcoef;
    use_tail = (idx->residual && Mc < idx->D);

    /* Weight validation: full length-D diagonal. Tail floor is the minimum
     * omitted weight, not a caller-supplied extra slot. */
    for (wi = 0; wi < idx->D; ++wi) {
        if (!isfinite(weight[wi]) || weight[wi] < 0.0f) {
            zenith__set_error(ZENITH_ERR_WEIGHT);
            return 0;
        }
    }
    if (use_tail) {
        w_tail_min = weight[Mc];
        for (wi = Mc + 1; wi < idx->D; ++wi)
            if (weight[wi] < w_tail_min) w_tail_min = weight[wi];
    }

    if (collect_candidates) {
        heap = NULL;
    } else if (k <= 256) {
        heap = heap_stk;
    } else {
        heap = (zenith_hit *)ZENITH_MALLOC(sizeof(zenith_hit) * k);
        if (!heap) {
            zenith__set_error(ZENITH_ERR_ALLOC);
            return 0;
        }
        heap_on_heap = 1;
    }

    query_mag = 0;
    if (idx->quantizer == ZENITH_QUANT_LM2) {
        h = zenith_sketch_lm2(Q, (int)idx->ncodes,
                              idx->use_perm ? idx->freq_perm : NULL,
                              idx->quant_scale, idx->quant_bias);
    } else if (idx->quantizer == ZENITH_QUANT_SIGN_MAG) {
        h = zenith_sketch_sign_calibrated(Q, (int)idx->ncodes,
                                          idx->use_perm ? idx->freq_perm : NULL,
                                          idx->quant_bias);
        query_mag = zenith_sketch_magnitude(Q, (int)idx->ncodes,
                                            idx->use_perm ? idx->freq_perm : NULL,
                                            idx->quant_scale, idx->quant_bias);
    } else if (idx->use_polar) {
        h = zenith_sketch_polar(Q, (int)(idx->nbits / 2),
                                idx->use_perm ? idx->freq_perm : NULL);
    } else {
        h = zenith_sketch(Q, (int)idx->nbits, idx->use_perm ? idx->freq_perm : NULL);
    }
    g = idx->quantizer == ZENITH_QUANT_SIGN ? zenith_gray(h) : h;

    if (idx->use_whiten) {
        if (Mc <= 1024) {
            Q_white = Qw_stk;
            w_eff = we_stk;
        } else {
            Q_white = (float *)ZENITH_ALIGNED_ALLOC(64, sizeof(float) * Mc);
            w_eff   = (float *)ZENITH_ALIGNED_ALLOC(64, sizeof(float) * Mc);
            if (!Q_white || !w_eff) {
                if (Q_white) ZENITH_ALIGNED_FREE(Q_white);
                if (w_eff) ZENITH_ALIGNED_FREE(w_eff);
                if (heap_on_heap) ZENITH_FREE(heap);
                zenith__set_error(ZENITH_ERR_ALLOC);
                return 0;
            }
            white_on_heap = 1;
        }
        for (wi = 0; wi < Mc; ++wi) {
            float ws = idx->whiten_sq[wi];
            Q_white[wi] = Q[wi] * ws;
            w_eff[wi]   = weight[wi] / (ws * ws + ZENITH_EPS_DIV);
        }
        Q_use = Q_white; w_use = w_eff;
    } else {
        Q_use = Q;
        w_use = weight;
    }

    qc.heap = heap; qc.hn = 0; qc.k = k; qc.Mc = Mc;
    qc.use_tail = use_tail; qc.w_tail_min = w_tail_min; qc.Rq = Rq;
    qc.Q_use = Q_use; qc.w_use = w_use;
    qc.collect_ids = collect_candidates ? out_ids : NULL;
    qc.collect_cap = collect_candidates ? ef : 0u;
    qc.collect_n = 0u;

    if (ef >= idx->N) {
        zenith__query_all(idx, &qc);
    } else if (idx->search == ZENITH_SEARCH_HAMMING) {
        zenith__query_hamming(idx, h, query_mag, ef, &qc);
    } else if (idx->search == ZENITH_SEARCH_POSTERIOR) {
        if (idx->quantizer == ZENITH_QUANT_SIGN)
            zenith__query_posterior(idx, Q, Q_use, w_use, ef, &qc);
        else
            zenith__query_lm_posterior(idx, Q, weight, ef, &qc);
    } else if ((idx->search == ZENITH_SEARCH_MIH || idx->search == ZENITH_SEARCH_HYBRID) &&
               idx->mih_m > 0 && idx->mih_off && idx->mih_idx) {
        uint32_t nwords = (idx->N + 63u) / 64u;
        uint32_t spent;
        if (nwords <= 1024u) {
            seen = seen_stk;
        } else {
            seen = (uint64_t *)ZENITH_MALLOC(sizeof(uint64_t) * nwords);
            if (seen) seen_heap = 1;
        }
        if (seen) {
            memset(seen, 0, sizeof(uint64_t) * nwords);
            if (idx->search == ZENITH_SEARCH_HYBRID) {
                uint32_t mih_cap = (uint32_t)(((uint64_t)ef * 3u) / 4u);
                if (mih_cap < k) mih_cap = k;
                if (mih_cap > ef) mih_cap = ef;
                spent = zenith__query_mih(idx, h, mih_cap, seen, &qc);
                if (spent < ef)
                    zenith__query_gray(idx, g, ef - spent, seen, &qc);
            } else {
                spent = zenith__query_mih(idx, h, ef, seen, &qc);
                /* Sparse buckets can leave pure MIH short of k even after
                 * radius expansion.  Backfill from the sorted Gray keys so a
                 * small ef still returns useful, distinct candidates. */
                if (spent < ef)
                    zenith__query_gray(idx, g, ef - spent, seen, &qc);
            }
        } else {
            zenith__query_gray(idx, g, ef, NULL, &qc);
        }
        if (seen_heap) ZENITH_FREE(seen);
    } else {
        zenith__query_gray(idx, g, ef, NULL, &qc);
    }

    if (collect_candidates) {
        if (white_on_heap) {
            ZENITH_ALIGNED_FREE(Q_white);
            ZENITH_ALIGNED_FREE(w_eff);
        }
        return qc.collect_n;
    }

    {
        int ii;
        for (ii = qc.hn / 2 - 1; ii >= 0; --ii)
            zenith_heap_sift(qc.heap, qc.hn, ii);
    }
    count = (uint32_t)qc.hn;
    for (r = 0; r < count; ++r) {
        zenith_hit top = qc.heap[0];
        qc.heap[0] = qc.heap[qc.hn - 1];
        qc.hn--;
        if (qc.hn > 0) zenith_heap_sift(qc.heap, qc.hn, 0);
        out_ids[count - 1 - r] = top.id;
        if (out_dist)
            out_dist[count - 1 - r] = sqrtf(top.dist < 0.0f ? 0.0f : top.dist);
    }

    if (white_on_heap) {
        ZENITH_ALIGNED_FREE(Q_white);
        ZENITH_ALIGNED_FREE(w_eff);
    }
    if (heap_on_heap) ZENITH_FREE(heap);
    return count;
}

uint32_t zenith_query_ef(const zenith_index_t *ZENITH_RESTRICT idx,
                         const float *ZENITH_RESTRICT q,
                         const float *ZENITH_RESTRICT weight,
                         uint32_t k, uint32_t ef,
                         uint32_t *ZENITH_RESTRICT out_ids,
                         float *ZENITH_RESTRICT out_dist) {
    ZENITH_ALIGN64 float Q_stk[1024];
    float *Q;
    int Q_on_heap = 0;
    float Rq = 0.0f;
    uint32_t Mc, ans;

    if (!idx || !q) {
        zenith__set_error(ZENITH_ERR_ARG);
        return 0;
    }
    {
        uint32_t di;
        for (di = 0; di < idx->D; ++di) {
            if (!isfinite(q[di])) {
                zenith__set_error(ZENITH_ERR_ARG);
                return 0;
            }
        }
    }
    Mc = idx->Mcoef;

    if ((uint32_t)idx->plan_M <= 1024) {
        Q = Q_stk;
    } else {
        Q = (float *)ZENITH_ALIGNED_ALLOC(64, sizeof(float) * (size_t)idx->plan_M);
        if (!Q) {
            zenith__set_error(ZENITH_ERR_ALLOC);
            return 0;
        }
        Q_on_heap = 1;
    }

    zenith_dct_lead(idx, q, Q);

    if (idx->residual && Mc < idx->D) {
        double qn = 0.0, ql = 0.0;
        uint32_t i;
        for (i = 0; i < idx->D; ++i) qn += (double)q[i] * (double)q[i];
        for (i = 0; i < Mc; ++i) ql += (double)Q[i] * (double)Q[i];
        {
            double energy = qn - ql;
            Rq = (energy > 0.0) ? (float)sqrt(energy) : 0.0f;
        }
    }

    ans = zenith__query_core(idx, Q, Rq, weight, k, ef, out_ids, out_dist, 0);
    if (Q_on_heap) ZENITH_ALIGNED_FREE(Q);
    return ans;
}

uint32_t zenith_query_many(const zenith_index_t *ZENITH_RESTRICT idx,
                           const float *ZENITH_RESTRICT Q, uint32_t nq,
                           const float *ZENITH_RESTRICT weight,
                           uint32_t k, uint32_t ef,
                           uint32_t *ZENITH_RESTRICT out_ids,
                           float *ZENITH_RESTRICT out_dist) {
    uint32_t qi, total = 0;
    if (!idx || !Q || nq == 0) {
        zenith__set_error(ZENITH_ERR_ARG);
        return 0;
    }
    if (nq == 1)
        return zenith_query_ef(idx, Q, weight, k, ef, out_ids, out_dist);
#if defined(_OPENMP)
#pragma omp parallel for num_threads(zenith__index_threads(idx)) schedule(static) reduction(+:total)
#endif
    for (qi = 0; qi < nq; ++qi) {
        uint32_t *ids = out_ids ? out_ids + (size_t)qi * k : NULL;
        float *dist = out_dist ? out_dist + (size_t)qi * k : NULL;
        total += zenith_query_ef(idx, Q + (size_t)qi * idx->D, weight, k, ef, ids, dist);
    }
    return total;
}

uint32_t zenith_candidates_ef(const zenith_index_t *ZENITH_RESTRICT idx,
                              const float *ZENITH_RESTRICT q,
                              const float *ZENITH_RESTRICT weight,
                              uint32_t ef,
                              uint32_t *ZENITH_RESTRICT out_ids) {
    ZENITH_ALIGN64 float Q_stk[1024];
    float *Q;
    int Q_on_heap = 0;
    float Rq = 0.0f;
    uint32_t Mc, ans, di;

    if (!idx || !q || !weight || !out_ids || ef == 0) {
        zenith__set_error(ZENITH_ERR_ARG);
        return 0;
    }
    for (di = 0; di < idx->D; ++di) {
        if (!isfinite(q[di])) {
            zenith__set_error(ZENITH_ERR_ARG);
            return 0;
        }
    }
    Mc = idx->Mcoef;
    if (ef > idx->N) ef = idx->N;
    if ((uint32_t)idx->plan_M <= 1024) {
        Q = Q_stk;
    } else {
        Q = (float *)ZENITH_ALIGNED_ALLOC(64, sizeof(float) * (size_t)idx->plan_M);
        if (!Q) {
            zenith__set_error(ZENITH_ERR_ALLOC);
            return 0;
        }
        Q_on_heap = 1;
    }
    zenith_dct_lead(idx, q, Q);
    if (idx->residual && Mc < idx->D) {
        double qn = 0.0, ql = 0.0;
        uint32_t i;
        for (i = 0; i < idx->D; ++i) qn += (double)q[i] * (double)q[i];
        for (i = 0; i < Mc; ++i) ql += (double)Q[i] * (double)Q[i];
        {
            double energy = qn - ql;
            Rq = (energy > 0.0) ? (float)sqrt(energy) : 0.0f;
        }
    }
    ans = zenith__query_core(idx, Q, Rq, weight, 1u, ef, out_ids, NULL, 1);
    if (Q_on_heap) ZENITH_ALIGNED_FREE(Q);
    return ans;
}

uint32_t zenith_query_coeffs(const zenith_index_t *ZENITH_RESTRICT idx,
                             const float *ZENITH_RESTRICT q_coeff,
                             float q_residual,
                             const float *ZENITH_RESTRICT weight,
                             uint32_t k, uint32_t ef,
                             uint32_t *ZENITH_RESTRICT out_ids,
                             float *ZENITH_RESTRICT out_dist) {
    if (!idx || !q_coeff) {
        zenith__set_error(ZENITH_ERR_ARG);
        return 0;
    }
    if (!isfinite(q_residual) || q_residual < 0.0f) {
        zenith__set_error(ZENITH_ERR_ARG);
        return 0;
    }
    if (idx->ncodes > idx->Mcoef) {
        zenith__set_error(ZENITH_ERR_ARG);
        return 0;
    }
    {
        uint32_t ci;
        for (ci = 0; ci < idx->Mcoef; ++ci) {
            if (!isfinite(q_coeff[ci])) {
                zenith__set_error(ZENITH_ERR_ARG);
                return 0;
            }
        }
    }
    return zenith__query_core(idx, q_coeff, q_residual, weight, k, ef, out_ids, out_dist, 0);
}

int zenith_project(const zenith_index_t *ZENITH_RESTRICT idx,
                   const float *ZENITH_RESTRICT x,
                   float *ZENITH_RESTRICT out_coeff,
                   float *ZENITH_RESTRICT out_residual) {
    uint32_t i;
    if (!idx || !x || !out_coeff) {
        zenith__set_error(ZENITH_ERR_ARG);
        return ZENITH_ERR_ARG;
    }
    /* Public contract is Mcoef outputs, even when the index keeps a wider
     * internal projection for sketch generation. */
    zenith_dct_lead_signed(idx, x, out_coeff, (int)idx->Mcoef);
    if (out_residual) {
        if (idx->Mcoef < idx->D) {
            double norm_sq = 0.0, lead_sq = 0.0, energy;
            for (i = 0; i < idx->D; ++i)
                norm_sq += (double)x[i] * (double)x[i];
            for (i = 0; i < idx->Mcoef; ++i)
                lead_sq += (double)out_coeff[i] * (double)out_coeff[i];
            energy = norm_sq - lead_sq;
            *out_residual = energy > 0.0 ? (float)sqrt(energy) : 0.0f;
        } else {
            *out_residual = 0.0f;
        }
    }
    zenith__set_error(ZENITH_OK);
    return ZENITH_OK;
}

uint32_t zenith_query(const zenith_index_t *ZENITH_RESTRICT idx,
                      const float *ZENITH_RESTRICT q,
                      const float *ZENITH_RESTRICT weight,
                      uint32_t k,
                      uint32_t *ZENITH_RESTRICT out_ids,
                      float *ZENITH_RESTRICT out_dist) {
    uint64_t want;
    uint32_t ef;
    if (!idx) {
        zenith__set_error(ZENITH_ERR_ARG);
        return 0;
    }
    want = 24u * (uint64_t)k;
    ef = want > UINT32_MAX ? UINT32_MAX : (uint32_t)want;
    if (ef < 512u) ef = 512u;
    if (ef > idx->N) ef = idx->N;
    return zenith_query_ef(idx, q, weight, k, ef, out_ids, out_dist);
}

/* Full signed DCT-II in double precision.  A rotation recurrence avoids a
 * libm cosine call for every element and is refreshed periodically to keep
 * phase drift negligible. */
static void zenith__full_dct_signed(const zenith_index_t *ZENITH_RESTRICT idx,
                                    const float *ZENITH_RESTRICT x,
                                    double *ZENITH_RESTRICT out) {
    uint32_t D = idx->D, k, n;
    for (k = 0; k < D; ++k) {
        double scale = (k == 0) ? sqrt(1.0 / (double)D)
                                : sqrt(2.0 / (double)D);
        double theta = ZENITH_PI * (double)k / (double)D;
        double step_c = cos(theta), step_s = sin(theta);
        double c = cos(0.5 * theta), s = sin(0.5 * theta);
        double acc = 0.0;
        for (n = 0; n < D; ++n) {
            double y = (double)x[n];
            double nc, ns;
            if (idx->input_signs) y *= (double)idx->input_signs[n];
            acc += scale * c * y;
            nc = c * step_c - s * step_s;
            ns = s * step_c + c * step_s;
            c = nc; s = ns;
            if ((n & 31u) == 31u && n + 1u < D) {
                double angle = theta * ((double)n + 1.5);
                c = cos(angle);
                s = sin(angle);
            }
        }
        out[k] = acc;
    }
}

/* Exact re-rank against original vectors.  weight has length D. */
uint32_t zenith_rerank_exact(const zenith_index_t *ZENITH_RESTRICT idx,
                             const float *ZENITH_RESTRICT vectors,
                             const float *ZENITH_RESTRICT q,
                             const float *ZENITH_RESTRICT weight,
                             const uint32_t *cand_ids,
                             uint32_t ncand,
                             uint32_t k,
                             uint32_t *out_ids,
                             float *ZENITH_RESTRICT out_dist) {
    zenith_hit *heap;
    double *qcoef = NULL, *xcoef = NULL;
    uint32_t i, count = 0, n;
    int constant_weight = 1;
    float first_w;

    if (!idx || !vectors || !q || !weight || !cand_ids || !out_ids ||
        k == 0 || ncand == 0) {
        zenith__set_error(ZENITH_ERR_ARG);
        return 0;
    }
    for (n = 0; n < idx->D; ++n) {
        if (!isfinite(weight[n]) || weight[n] < 0.0f) {
            zenith__set_error(ZENITH_ERR_WEIGHT);
            return 0;
        }
        if (!isfinite(q[n])) {
            zenith__set_error(ZENITH_ERR_ARG);
            return 0;
        }
    }
    for (i = 0; i < ncand; ++i) {
        if (cand_ids[i] >= idx->N) {
            zenith__set_error(ZENITH_ERR_ARG);
            return 0;
        }
    }
    zenith__set_error(ZENITH_OK);
    if (k > ncand) k = ncand;

    heap = (zenith_hit *)ZENITH_MALLOC(sizeof(zenith_hit) * k);
    if (!heap) {
        zenith__set_error(ZENITH_ERR_ALLOC);
        return 0;
    }

    first_w = weight[0];
    for (n = 1; n < idx->D; ++n) {
        if (weight[n] != first_w) {
            constant_weight = 0;
            break;
        }
    }

    if (!constant_weight) {
        qcoef = (double *)ZENITH_MALLOC(sizeof(double) * (size_t)idx->D);
        xcoef = (double *)ZENITH_MALLOC(sizeof(double) * (size_t)idx->D);
        if (!qcoef || !xcoef) {
            ZENITH_FREE(qcoef); ZENITH_FREE(xcoef); ZENITH_FREE(heap);
            zenith__set_error(ZENITH_ERR_ALLOC);
            return 0;
        }
        zenith__full_dct_signed(idx, q, qcoef);
    }

    for (i = 0; i < ncand; ++i) {
        uint32_t id = cand_ids[i];
        const float *x = vectors + (size_t)id * (size_t)idx->D;
        double dd = 0.0;
        float d;
        if (constant_weight) {
            for (n = 0; n < idx->D; ++n) {
                double e = (double)q[n] - (double)x[n];
                dd += e * e;
            }
            dd *= (double)first_w;
        } else {
            zenith__full_dct_signed(idx, x, xcoef);
            for (n = 0; n < idx->D; ++n) {
                double e = qcoef[n] - xcoef[n];
                dd += (double)weight[n] * e * e;
            }
        }
        d = (dd > (double)FLT_MAX) ? FLT_MAX : (float)dd;
        if (count < k) {
            heap[count].dist = d;
            heap[count].id = id;
            count++;
            if (count == k) {
                int ii;
                for (ii = (int)k / 2 - 1; ii >= 0; --ii)
                    zenith_heap_sift(heap, (int)count, ii);
            }
        } else {
            zenith_hit cand;
            cand.dist = d;
            cand.id = id;
            if (zenith_hit_better(cand, heap[0])) {
                heap[0] = cand;
                zenith_heap_sift(heap, (int)k, 0);
            }
        }
    }

    {
        uint32_t nout = count, r;
        for (r = 0; r < nout; ++r) {
            zenith_hit top = heap[0];
            heap[0] = heap[count - 1];
            count--;
            if (count > 0) zenith_heap_sift(heap, (int)count, 0);
            out_ids[nout - 1u - r] = top.id;
            if (out_dist)
                out_dist[nout - 1u - r] = sqrtf(top.dist < 0.0f ? 0.0f : top.dist);
        }
        count = nout;
    }

    ZENITH_FREE(qcoef);
    ZENITH_FREE(xcoef);
    ZENITH_FREE(heap);
    return count;
}

float zenith_tail_min(const float *ZENITH_RESTRICT weight, int D, int Mcoef) {
    int i;
    float m;
    if (!weight || D <= 0 || Mcoef < 0 || Mcoef >= D) {
        if (!weight || D <= 0 || Mcoef < 0 || Mcoef > D)
            zenith__set_error(ZENITH_ERR_ARG);
        else
            zenith__set_error(ZENITH_OK);
        return 0.0f;
    }
    m = weight[Mcoef];
    for (i = Mcoef; i < D; ++i) {
        if (!isfinite(weight[i]) || weight[i] < 0.0f) {
            zenith__set_error(ZENITH_ERR_WEIGHT);
            return 0.0f;
        }
        if (weight[i] < m) m = weight[i];
    }
    zenith__set_error(ZENITH_OK);
    return m;
}

/* ---------- Persistence (format version 1) ---------- */
#define ZENITH_MAGIC        0x5A4E5448UL /* "ZNTH" */
#define ZENITH_FILE_VERSION 1u
#define ZENITH_ENDIAN_MARK  0x01020304u

typedef struct {
    uint32_t magic;
    uint32_t version;
    uint32_t N, D, Mcoef, nbits;
    uint64_t seed;
    uint32_t checksum;          /* XOR of padded block checksums after the header */
    uint32_t endian;            /* ZENITH_ENDIAN_MARK in writer byte order */
    uint8_t  freq_perm[64];
    uint8_t  use_perm, use_whiten, use_polar, gen;
    uint8_t  search, mih_m, mih_bits, mih_max_radius;
    uint8_t  quantizer, reserved[15]; /* fixed 128-byte header; payload is 64B aligned */
} zenith_file_hdr;
typedef char zenith__file_hdr_size[(sizeof(zenith_file_hdr) == 128) ? 1 : -1];

typedef struct {
    size_t whiten_off, whiten_bytes;
    size_t residual_off, residual_bytes;
    size_t keys_off, keys_bytes;
    size_t ids_off, ids_bytes;
    size_t coeffs_off, coeffs_bytes;
    size_t quant_scale_off, quant_scale_bytes;
    size_t quant_bias_off, quant_bias_bytes;
    size_t mag_keys_off, mag_keys_bytes;
    size_t mih_table_off, mih_table_bytes;
    size_t mih_index_off, mih_index_bytes;
    size_t total_bytes;
} zenith_file_layout;

static size_t zenith__pad64(size_t n) {
    if (n > ((size_t)-1) - 63u) return (size_t)-1;
    return (n + 63u) & ~(size_t)63u;
}
static int zenith__padded(size_t n, size_t *out) {
    size_t p = zenith__pad64(n);
    if (n != 0 && p < n) return 0;
    *out = p;
    return 1;
}

static int zenith__size_mul(size_t a, size_t b, size_t *out) {
    if (a != 0 && b > ((size_t)-1) / a) return 0;
    *out = a * b;
    return 1;
}
static int zenith__size_add(size_t a, size_t b, size_t *out) {
    if (b > ((size_t)-1) - a) return 0;
    *out = a + b;
    return 1;
}

static int zenith__validate_file_hdr(const zenith_file_hdr *h) {
    uint32_t expect_m, expect_bits, i, lim;
    if (!h || h->magic != ZENITH_MAGIC || h->version != ZENITH_FILE_VERSION ||
        h->endian != ZENITH_ENDIAN_MARK)
        return ZENITH_ERR_FORMAT;
    if (h->N == 0 || h->D == 0 || h->Mcoef == 0 || h->Mcoef > h->D ||
        h->nbits == 0 || h->nbits > 64u)
        return ZENITH_ERR_FORMAT;
    if (h->quantizer == ZENITH_QUANT_LM2) {
        if ((h->nbits / 2u) == 0 || (h->nbits / 2u) > h->D)
            return ZENITH_ERR_FORMAT;
    } else if (h->nbits > h->D) {
        return ZENITH_ERR_FORMAT;
    }
    if (h->use_perm > 1u || h->use_whiten > 1u || h->use_polar > 1u)
        return ZENITH_ERR_FORMAT;
    if (h->gen != ZENITH_GEN_DCT && h->gen != ZENITH_GEN_SIGNED_DCT)
        return ZENITH_ERR_FORMAT;
    if (h->search != ZENITH_SEARCH_MIH && h->search != ZENITH_SEARCH_GRAY &&
        h->search != ZENITH_SEARCH_HYBRID &&
        h->search != ZENITH_SEARCH_HAMMING &&
        h->search != ZENITH_SEARCH_POSTERIOR)
        return ZENITH_ERR_FORMAT;
    if (h->quantizer != ZENITH_QUANT_SIGN && h->quantizer != ZENITH_QUANT_LM2 &&
        h->quantizer != ZENITH_QUANT_SIGN_MAG)
        return ZENITH_ERR_FORMAT;
    if (h->quantizer == ZENITH_QUANT_LM2 && (h->nbits & 1u))
        return ZENITH_ERR_FORMAT;
    if (h->quantizer != ZENITH_QUANT_SIGN &&
        (h->use_polar ||
         (h->search != ZENITH_SEARCH_HAMMING && h->search != ZENITH_SEARCH_POSTERIOR)))
        return ZENITH_ERR_FORMAT;
    if (h->search == ZENITH_SEARCH_POSTERIOR && h->use_polar)
        return ZENITH_ERR_FORMAT;
    if (h->use_polar && (h->nbits < 2u || (h->nbits & 1u)))
        return ZENITH_ERR_FORMAT;
    if (h->use_perm) {
        uint32_t j;
        if (h->nbits > h->Mcoef) return ZENITH_ERR_FORMAT;
        lim = h->nbits < 64u ? h->nbits : 64u;
        for (i = 0; i < lim; ++i) {
            if (h->freq_perm[i] >= h->Mcoef) return ZENITH_ERR_FORMAT;
            for (j = 0; j < i; ++j)
                if (h->freq_perm[j] == h->freq_perm[i]) return ZENITH_ERR_FORMAT;
        }
    }
    if (h->search == ZENITH_SEARCH_MIH || h->search == ZENITH_SEARCH_HYBRID) {
        zenith__mih_layout(h->nbits, &expect_m, &expect_bits);
        if (h->mih_m != expect_m || h->mih_bits != expect_bits ||
            h->mih_m == 0 || h->mih_m > 8u || h->mih_bits > 16u)
            return ZENITH_ERR_FORMAT;
    } else if (h->mih_m != 0 || h->mih_bits != 0) {
        return ZENITH_ERR_FORMAT;
    }
    return ZENITH_OK;
}

static int zenith__file_layout(const zenith_file_hdr *h, zenith_file_layout *L) {
    size_t v, nb, raw;
    memset(L, 0, sizeof(*L));
    L->whiten_off = sizeof(*h);
    if (!zenith__size_mul((size_t)h->Mcoef, sizeof(float), &raw) ||
        !zenith__padded(raw, &L->whiten_bytes)) return 0;
    if (!zenith__size_add(L->whiten_off, L->whiten_bytes, &L->residual_off)) return 0;
    if (!zenith__size_mul((size_t)h->N, sizeof(float), &raw) ||
        !zenith__padded(raw, &L->residual_bytes)) return 0;
    if (!zenith__size_add(L->residual_off, L->residual_bytes, &L->keys_off)) return 0;
    if (!zenith__size_mul((size_t)h->N, sizeof(uint64_t), &raw) ||
        !zenith__padded(raw, &L->keys_bytes)) return 0;
    if (!zenith__size_add(L->keys_off, L->keys_bytes, &L->ids_off)) return 0;
    if (!zenith__size_mul((size_t)h->N, sizeof(uint32_t), &raw) ||
        !zenith__padded(raw, &L->ids_bytes)) return 0;
    if (!zenith__size_add(L->ids_off, L->ids_bytes, &L->coeffs_off)) return 0;
    if (!zenith__size_mul((size_t)h->N, (size_t)h->Mcoef, &v)) return 0;
    if (!zenith__size_mul(v, sizeof(float), &raw) ||
        !zenith__padded(raw, &L->coeffs_bytes)) return 0;
    if (!zenith__size_add(L->coeffs_off, L->coeffs_bytes, &L->total_bytes)) return 0;
    if (h->quantizer != ZENITH_QUANT_SIGN) {
        uint32_t ncodes = h->quantizer == ZENITH_QUANT_LM2 ? h->nbits / 2u : h->nbits;
        L->quant_scale_off = L->total_bytes;
        if (!zenith__size_mul((size_t)ncodes, sizeof(float), &raw) ||
            !zenith__padded(raw, &L->quant_scale_bytes)) return 0;
        if (!zenith__size_add(L->quant_scale_off, L->quant_scale_bytes, &L->quant_bias_off)) return 0;
        if (!zenith__padded(raw, &L->quant_bias_bytes)) return 0;
        if (!zenith__size_add(L->quant_bias_off, L->quant_bias_bytes, &L->total_bytes)) return 0;
        if (h->quantizer == ZENITH_QUANT_SIGN_MAG) {
            L->mag_keys_off = L->total_bytes;
            if (!zenith__size_mul((size_t)h->N, sizeof(uint64_t), &raw) ||
                !zenith__padded(raw, &L->mag_keys_bytes)) return 0;
            if (!zenith__size_add(L->mag_keys_off, L->mag_keys_bytes, &L->total_bytes)) return 0;
        }
    }
    if (h->mih_m > 0) {
        nb = (size_t)1u << h->mih_bits;
        if (!zenith__size_mul((size_t)h->mih_m, nb + 1u, &v)) return 0;
        if (!zenith__size_mul(v, sizeof(uint32_t), &raw) ||
            !zenith__padded(raw, &L->mih_table_bytes)) return 0;
        L->mih_table_off = L->total_bytes;
        if (!zenith__size_add(L->mih_table_off, L->mih_table_bytes, &L->mih_index_off)) return 0;
        if (!zenith__size_mul((size_t)h->mih_m, (size_t)h->N, &v)) return 0;
        if (!zenith__size_mul(v, sizeof(uint32_t), &raw) ||
            !zenith__padded(raw, &L->mih_index_bytes)) return 0;
        if (!zenith__size_add(L->mih_index_off, L->mih_index_bytes, &L->total_bytes)) return 0;
    }
    return 1;
}

static zenith_index_t *zenith__index_from_hdr(const zenith_file_hdr *h) {
    zenith_index_t *idx = (zenith_index_t *)ZENITH_MALLOC(sizeof(*idx));
    if (!idx) return NULL;
    memset(idx, 0, sizeof(*idx));
    idx->N = h->N; idx->D = h->D; idx->Mcoef = h->Mcoef; idx->nbits = h->nbits;
    memcpy(idx->freq_perm, h->freq_perm, 64);
    idx->use_perm = h->use_perm;
    idx->use_whiten = h->use_whiten;
    idx->use_polar = h->use_polar;
    idx->gen = h->gen;
    idx->gen_seed = h->seed;
    idx->search = h->search;
    idx->mih_m = h->mih_m;
    idx->mih_bits = h->mih_bits;
    idx->mih_max_radius = h->mih_max_radius ? h->mih_max_radius : 2u;
    idx->quantizer = (int)h->quantizer;
    idx->ncodes = h->quantizer == ZENITH_QUANT_LM2 ? h->nbits / 2u : h->nbits;
    return idx;
}

static int zenith__prepare_loaded_plan(zenith_index_t *idx) {
    uint32_t Mtab = idx->Mcoef > idx->ncodes ? idx->Mcoef : idx->ncodes;
    int rc;
    if (idx->gen == ZENITH_GEN_SIGNED_DCT) {
        idx->input_signs = (int8_t *)ZENITH_MALLOC(sizeof(int8_t) * idx->D);
        if (!idx->input_signs) return ZENITH_ERR_ALLOC;
        zenith__gen_input_signs(idx->input_signs, idx->D, idx->gen_seed);
    }
    if (idx->nthreads <= 0) idx->nthreads = zenith_thread_count();
    rc = zenith_dct_plan_init(idx, (int)idx->D, (int)Mtab, 0);
    if (rc != ZENITH_OK) return rc;
    zenith_dct_plan_apply_signs(idx);
    return ZENITH_OK;
}

static int zenith__validate_whiten(const zenith_index_t *idx) {
    uint32_t i;
    if (!idx->use_whiten) return ZENITH_OK;
    if (!idx->whiten_sq) return ZENITH_ERR_FORMAT;
    for (i = 0; i < idx->Mcoef; ++i)
        if (!isfinite(idx->whiten_sq[i]) || idx->whiten_sq[i] <= 0.0f)
            return ZENITH_ERR_FORMAT;
    return ZENITH_OK;
}

static int zenith__validate_quant(const zenith_index_t *idx) {
    uint32_t i;
    if (idx->quantizer == ZENITH_QUANT_SIGN) return ZENITH_OK;
    if (!idx->quant_scale || !idx->quant_bias) return ZENITH_ERR_FORMAT;
    if (idx->quantizer == ZENITH_QUANT_SIGN_MAG && !idx->mag_keys)
        return ZENITH_ERR_FORMAT;
    for (i = 0; i < idx->ncodes; ++i) {
        if (!isfinite(idx->quant_scale[i]) || idx->quant_scale[i] <= 0.0f ||
            !isfinite(idx->quant_bias[i]))
            return ZENITH_ERR_FORMAT;
    }
    return ZENITH_OK;
}

static int zenith__write_padded(FILE *f, const void *src, size_t raw, size_t padded,
                                uint32_t *csum) {
    uint8_t *buf;
    int rc;
    if (padded < raw) return 0;
    buf = (uint8_t *)ZENITH_MALLOC(padded);
    if (!buf) return -1;
    memset(buf, 0, padded);
    if (src && raw) memcpy(buf, src, raw);
    *csum ^= zenith__checksum(buf, padded);
    rc = (fwrite(buf, 1, padded, f) == padded);
    ZENITH_FREE(buf);
    return rc ? 1 : 0;
}

int zenith_save(const zenith_index_t *idx, const char *path) {
    FILE *f;
    zenith_file_hdr h;
    zenith_file_layout L;
    uint32_t csum = 0;
    int wr;

    if (!idx || !path || !idx->keys || !idx->ids || !idx->coeffs || !idx->residual ||
        (idx->use_whiten && !idx->whiten_sq) ||
        (idx->quantizer != ZENITH_QUANT_SIGN &&
         (!idx->quant_scale || !idx->quant_bias)) ||
        (idx->quantizer == ZENITH_QUANT_SIGN_MAG && !idx->mag_keys)) {
        zenith__set_error(ZENITH_ERR_ARG);
        return ZENITH_ERR_ARG;
    }
    if (sizeof(h) != 128u) {
        zenith__set_error(ZENITH_ERR_FORMAT);
        return ZENITH_ERR_FORMAT;
    }
    memset(&h, 0, sizeof(h));
    h.magic = ZENITH_MAGIC;
    h.version = ZENITH_FILE_VERSION;
    h.N = idx->N; h.D = idx->D; h.Mcoef = idx->Mcoef; h.nbits = idx->nbits;
    h.seed = idx->gen_seed;
    memcpy(h.freq_perm, idx->freq_perm, 64);
    h.use_perm = (uint8_t)!!idx->use_perm;
    h.use_whiten = (uint8_t)!!idx->use_whiten;
    h.use_polar = (uint8_t)!!idx->use_polar;
    h.gen = (uint8_t)idx->gen;
    h.search = (uint8_t)idx->search;
    h.mih_m = (uint8_t)idx->mih_m;
    h.mih_bits = (uint8_t)idx->mih_bits;
    h.mih_max_radius = (uint8_t)idx->mih_max_radius;
    h.quantizer = (uint8_t)idx->quantizer;
    h.endian = ZENITH_ENDIAN_MARK;
    if (zenith__validate_file_hdr(&h) != ZENITH_OK || !zenith__file_layout(&h, &L)) {
        zenith__set_error(ZENITH_ERR_FORMAT);
        return ZENITH_ERR_FORMAT;
    }

    f = fopen(path, "wb");
    if (!f) {
        zenith__set_error(ZENITH_ERR_IO);
        return ZENITH_ERR_IO;
    }
    if (fwrite(&h, sizeof(h), 1, f) != 1) goto io_fail;

    wr = zenith__write_padded(f, idx->use_whiten ? idx->whiten_sq : NULL,
                              idx->use_whiten ? (size_t)idx->Mcoef * sizeof(float) : 0,
                              L.whiten_bytes, &csum);
    if (wr < 0) goto alloc_fail;
    if (!wr) goto io_fail;
    wr = zenith__write_padded(f, idx->residual, (size_t)idx->N * sizeof(float),
                              L.residual_bytes, &csum);
    if (wr < 0) goto alloc_fail;
    if (!wr) goto io_fail;
    wr = zenith__write_padded(f, idx->keys, (size_t)idx->N * sizeof(uint64_t),
                              L.keys_bytes, &csum);
    if (wr < 0) goto alloc_fail;
    if (!wr) goto io_fail;
    wr = zenith__write_padded(f, idx->ids, (size_t)idx->N * sizeof(uint32_t),
                              L.ids_bytes, &csum);
    if (wr < 0) goto alloc_fail;
    if (!wr) goto io_fail;
    wr = zenith__write_padded(f, idx->coeffs,
                              (size_t)idx->N * idx->Mcoef * sizeof(float),
                              L.coeffs_bytes, &csum);
    if (wr < 0) goto alloc_fail;
    if (!wr) goto io_fail;

    if (idx->quantizer != ZENITH_QUANT_SIGN) {
        wr = zenith__write_padded(f, idx->quant_scale,
                                  (size_t)idx->ncodes * sizeof(float),
                                  L.quant_scale_bytes, &csum);
        if (wr < 0) goto alloc_fail;
        if (!wr) goto io_fail;
        wr = zenith__write_padded(f, idx->quant_bias,
                                  (size_t)idx->ncodes * sizeof(float),
                                  L.quant_bias_bytes, &csum);
        if (wr < 0) goto alloc_fail;
        if (!wr) goto io_fail;
        if (idx->quantizer == ZENITH_QUANT_SIGN_MAG) {
            wr = zenith__write_padded(f, idx->mag_keys,
                                      (size_t)idx->N * sizeof(uint64_t),
                                      L.mag_keys_bytes, &csum);
            if (wr < 0) goto alloc_fail;
            if (!wr) goto io_fail;
        }
    }

    if (h.mih_m > 0) {
        size_t nb, v;
        if (!idx->mih_off || !idx->mih_idx) goto format_fail;
        nb = (size_t)1u << h.mih_bits;
        v = (size_t)h.mih_m * (nb + 1u);
        wr = zenith__write_padded(f, idx->mih_off, v * sizeof(uint32_t),
                                  L.mih_table_bytes, &csum);
        if (wr < 0) goto alloc_fail;
        if (!wr) goto io_fail;
        wr = zenith__write_padded(f, idx->mih_idx,
                                  (size_t)h.mih_m * (size_t)h.N * sizeof(uint32_t),
                                  L.mih_index_bytes, &csum);
        if (wr < 0) goto alloc_fail;
        if (!wr) goto io_fail;
    }

    h.checksum = csum;
    if (fseek(f, 0, SEEK_SET) != 0 || fwrite(&h, sizeof(h), 1, f) != 1)
        goto io_fail;
    if (fclose(f) != 0) {
        zenith__set_error(ZENITH_ERR_IO);
        return ZENITH_ERR_IO;
    }
    zenith__set_error(ZENITH_OK);
    return ZENITH_OK;

alloc_fail:
    fclose(f);
    zenith__set_error(ZENITH_ERR_ALLOC);
    return ZENITH_ERR_ALLOC;
format_fail:
    fclose(f);
    zenith__set_error(ZENITH_ERR_FORMAT);
    return ZENITH_ERR_FORMAT;
io_fail:
    fclose(f);
    zenith__set_error(ZENITH_ERR_IO);
    return ZENITH_ERR_IO;
}

zenith_index_t *zenith_load_copy(const char *path) {
    FILE *f;
    zenith_file_hdr h;
    zenith_file_layout L;
    zenith_index_t *idx = NULL;
    uint32_t csum = 0;
    int rc;

    if (!path) {
        zenith__set_error(ZENITH_ERR_ARG);
        return NULL;
    }
    f = fopen(path, "rb");
    if (!f) {
        zenith__set_error(ZENITH_ERR_IO);
        return NULL;
    }
    if (fread(&h, sizeof(h), 1, f) != 1 ||
        zenith__validate_file_hdr(&h) != ZENITH_OK ||
        !zenith__file_layout(&h, &L)) {
        fclose(f);
        zenith__set_error(ZENITH_ERR_FORMAT);
        return NULL;
    }
    idx = zenith__index_from_hdr(&h);
    if (!idx) {
        fclose(f);
        zenith__set_error(ZENITH_ERR_ALLOC);
        return NULL;
    }

    {
        uint8_t *buf = (uint8_t *)ZENITH_MALLOC(L.whiten_bytes);
        if (!buf || fread(buf, 1, L.whiten_bytes, f) != L.whiten_bytes) {
            ZENITH_FREE(buf); goto format_fail;
        }
        csum ^= zenith__checksum(buf, L.whiten_bytes);
        if (h.use_whiten) {
            idx->whiten_sq = (float *)ZENITH_MALLOC((size_t)h.Mcoef * sizeof(float));
            if (!idx->whiten_sq) { ZENITH_FREE(buf); goto alloc_fail; }
            memcpy(idx->whiten_sq, buf, (size_t)h.Mcoef * sizeof(float));
        }
        ZENITH_FREE(buf);
    }
    idx->residual = (float *)ZENITH_ALIGNED_ALLOC(64, L.residual_bytes);
    idx->keys     = (uint64_t *)ZENITH_ALIGNED_ALLOC(64, L.keys_bytes);
    idx->ids      = (uint32_t *)ZENITH_ALIGNED_ALLOC(64, L.ids_bytes);
    idx->coeffs   = (float *)ZENITH_ALIGNED_ALLOC(64, L.coeffs_bytes);
    if (!idx->residual || !idx->keys || !idx->ids || !idx->coeffs) goto alloc_fail;
    if (fread(idx->residual, 1, L.residual_bytes, f) != L.residual_bytes ||
        fread(idx->keys, 1, L.keys_bytes, f) != L.keys_bytes ||
        fread(idx->ids, 1, L.ids_bytes, f) != L.ids_bytes ||
        fread(idx->coeffs, 1, L.coeffs_bytes, f) != L.coeffs_bytes)
        goto format_fail;
    csum ^= zenith__checksum(idx->residual, L.residual_bytes);
    csum ^= zenith__checksum(idx->keys, L.keys_bytes);
    csum ^= zenith__checksum(idx->ids, L.ids_bytes);
    csum ^= zenith__checksum(idx->coeffs, L.coeffs_bytes);

    if (h.quantizer != ZENITH_QUANT_SIGN) {
        idx->quant_scale = (float *)ZENITH_MALLOC((size_t)idx->ncodes * sizeof(float));
        idx->quant_bias  = (float *)ZENITH_MALLOC((size_t)idx->ncodes * sizeof(float));
        if (!idx->quant_scale || !idx->quant_bias) goto alloc_fail;
        {
            uint8_t *buf = (uint8_t *)ZENITH_MALLOC(L.quant_scale_bytes > L.quant_bias_bytes ?
                                                    L.quant_scale_bytes : L.quant_bias_bytes);
            if (!buf) goto alloc_fail;
            if (fread(buf, 1, L.quant_scale_bytes, f) != L.quant_scale_bytes) {
                ZENITH_FREE(buf); goto format_fail;
            }
            csum ^= zenith__checksum(buf, L.quant_scale_bytes);
            memcpy(idx->quant_scale, buf, (size_t)idx->ncodes * sizeof(float));
            if (fread(buf, 1, L.quant_bias_bytes, f) != L.quant_bias_bytes) {
                ZENITH_FREE(buf); goto format_fail;
            }
            csum ^= zenith__checksum(buf, L.quant_bias_bytes);
            memcpy(idx->quant_bias, buf, (size_t)idx->ncodes * sizeof(float));
            ZENITH_FREE(buf);
        }
        if (h.quantizer == ZENITH_QUANT_SIGN_MAG) {
            idx->mag_keys = (uint64_t *)ZENITH_ALIGNED_ALLOC(64, L.mag_keys_bytes);
            if (!idx->mag_keys) goto alloc_fail;
            if (fread(idx->mag_keys, 1, L.mag_keys_bytes, f) != L.mag_keys_bytes)
                goto format_fail;
            csum ^= zenith__checksum(idx->mag_keys, L.mag_keys_bytes);
        }
    }

    if (h.mih_m > 0) {
        idx->mih_off = (uint32_t *)ZENITH_ALIGNED_ALLOC(64, L.mih_table_bytes);
        idx->mih_idx = (uint32_t *)ZENITH_ALIGNED_ALLOC(64, L.mih_index_bytes);
        if (!idx->mih_off || !idx->mih_idx) goto alloc_fail;
        if (fread(idx->mih_off, 1, L.mih_table_bytes, f) != L.mih_table_bytes ||
            fread(idx->mih_idx, 1, L.mih_index_bytes, f) != L.mih_index_bytes)
            goto format_fail;
        csum ^= zenith__checksum(idx->mih_off, L.mih_table_bytes);
        csum ^= zenith__checksum(idx->mih_idx, L.mih_index_bytes);
    }

    if (csum != h.checksum) goto format_fail;
    fclose(f);
    f = NULL;
    if (zenith__validate_whiten(idx) != ZENITH_OK) goto format_fail;
    if (zenith__validate_quant(idx) != ZENITH_OK) goto format_fail;
    rc = zenith__prepare_loaded_plan(idx);
    if (rc != ZENITH_OK) {
        zenith_free(idx);
        zenith__set_error((zenith_err_t)rc);
        return NULL;
    }
    zenith__set_error(ZENITH_OK);
    return idx;

alloc_fail:
    if (f) fclose(f);
    zenith_free(idx);
    zenith__set_error(ZENITH_ERR_ALLOC);
    return NULL;
format_fail:
    if (f) fclose(f);
    zenith_free(idx);
    zenith__set_error(ZENITH_ERR_FORMAT);
    return NULL;
}

#if defined(ZENITH_HAVE_MMAP)
zenith_index_t *zenith_load(const char *path) {
    int fd;
    struct stat st;
    uint8_t *base;
    zenith_file_hdr h;
    zenith_file_layout L;
    zenith_index_t *idx;
    uint32_t csum = 0;
    size_t file_len;
    int rc;

    if (!path) {
        zenith__set_error(ZENITH_ERR_ARG);
        return NULL;
    }
    fd = open(path, O_RDONLY);
    if (fd < 0) return zenith_load_copy(path);
    if (fstat(fd, &st) != 0 || st.st_size < 0) {
        close(fd);
        return zenith_load_copy(path);
    }
    file_len = (size_t)st.st_size;
    if ((off_t)file_len != st.st_size || file_len < sizeof(zenith_file_hdr)) {
        close(fd);
        zenith__set_error(ZENITH_ERR_FORMAT);
        return NULL;
    }
    base = (uint8_t *)mmap(NULL, file_len, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);
    if (base == MAP_FAILED) return zenith_load_copy(path);

    memcpy(&h, base, sizeof(h));
    rc = zenith__validate_file_hdr(&h);
    if (rc != ZENITH_OK || !zenith__file_layout(&h, &L) || L.total_bytes > file_len) {
        munmap(base, file_len);
        zenith__set_error(ZENITH_ERR_FORMAT);
        return NULL;
    }

    csum ^= zenith__checksum(base + L.whiten_off, L.whiten_bytes);
    csum ^= zenith__checksum(base + L.residual_off, L.residual_bytes);
    csum ^= zenith__checksum(base + L.keys_off, L.keys_bytes);
    csum ^= zenith__checksum(base + L.ids_off, L.ids_bytes);
    csum ^= zenith__checksum(base + L.coeffs_off, L.coeffs_bytes);
    if (h.quantizer != ZENITH_QUANT_SIGN) {
        csum ^= zenith__checksum(base + L.quant_scale_off, L.quant_scale_bytes);
        csum ^= zenith__checksum(base + L.quant_bias_off, L.quant_bias_bytes);
        if (h.quantizer == ZENITH_QUANT_SIGN_MAG)
            csum ^= zenith__checksum(base + L.mag_keys_off, L.mag_keys_bytes);
    }
    if (h.mih_m > 0) {
        csum ^= zenith__checksum(base + L.mih_table_off, L.mih_table_bytes);
        csum ^= zenith__checksum(base + L.mih_index_off, L.mih_index_bytes);
    }
    if (csum != h.checksum) {
        munmap(base, file_len);
        zenith__set_error(ZENITH_ERR_FORMAT);
        return NULL;
    }

    idx = zenith__index_from_hdr(&h);
    if (!idx) {
        munmap(base, file_len);
        zenith__set_error(ZENITH_ERR_ALLOC);
        return NULL;
    }
    idx->mmapped = 1;
    idx->map_base = base;
    idx->map_len = file_len;
    if (h.use_whiten)
        idx->whiten_sq = (float *)(base + L.whiten_off);
    idx->residual = (float *)(base + L.residual_off);
    idx->keys = (uint64_t *)(base + L.keys_off);
    idx->ids = (uint32_t *)(base + L.ids_off);
    idx->coeffs = (float *)(base + L.coeffs_off);
    if (h.quantizer != ZENITH_QUANT_SIGN) {
        idx->quant_scale = (float *)(base + L.quant_scale_off);
        idx->quant_bias = (float *)(base + L.quant_bias_off);
        if (h.quantizer == ZENITH_QUANT_SIGN_MAG)
            idx->mag_keys = (uint64_t *)(base + L.mag_keys_off);
    }
    if (h.mih_m > 0) {
        idx->mih_off = (uint32_t *)(base + L.mih_table_off);
        idx->mih_idx = (uint32_t *)(base + L.mih_index_off);
    }

    rc = zenith__validate_whiten(idx);
    if (rc == ZENITH_OK)
        rc = zenith__validate_quant(idx);
    if (rc == ZENITH_OK)
        rc = zenith__prepare_loaded_plan(idx);
    if (rc != ZENITH_OK) {
        zenith_free(idx);
        zenith__set_error((zenith_err_t)rc);
        return NULL;
    }
    zenith__set_error(ZENITH_OK);
    return idx;
}
#else
zenith_index_t *zenith_load(const char *path) {
    return zenith_load_copy(path);
}
#endif

void zenith_unload(zenith_index_t *idx) { zenith_free(idx); }

/* ---------- Weight constructors ----------
 * Strictly positive, nondecreasing roughness penalties on the Neumann
 * Laplacian spectrum. Query takes the full length-D vector and uses
 * min(weight[Mcoef..D)) as the tail floor when the index is truncated. */
void zenith_w_identity(int D, float *ZENITH_RESTRICT w) {
    int i;
    if (!w || D <= 0) {
        zenith__set_error(ZENITH_ERR_ARG);
        return;
    }
    for (i = 0; i < D; ++i) w[i] = 1.0f;
    zenith__set_error(ZENITH_OK);
}

void zenith_laplacian_eigs(int D, double *ZENITH_RESTRICT mu) {
    int k;
    if (!mu || D <= 0) {
        zenith__set_error(ZENITH_ERR_ARG);
        return;
    }
    for (k = 0; k < D; ++k)
        mu[k] = 2.0 * (1.0 - cos(ZENITH_PI * (double)k / (double)D));
    zenith__set_error(ZENITH_OK);
}

static float zenith__weight_clamp(double x) {
    if (isnan(x) || x <= 0.0) return 1e-12f;
    if (!isfinite(x) || x > (double)FLT_MAX) return FLT_MAX;
    return (float)x;
}

void zenith_w_sobolev(const double *ZENITH_RESTRICT mu, int D, double s,
                      float *ZENITH_RESTRICT w) {
    int k;
    if (!mu || !w || D <= 0) {
        zenith__set_error(ZENITH_ERR_ARG);
        return;
    }
    if (s < 0.0) s = 0.0;
    for (k = 0; k < D; ++k) {
        double m = mu[k] < 0.0 ? 0.0 : mu[k];
        w[k] = zenith__weight_clamp(pow(1.0 + m, s));
    }
    zenith__set_error(ZENITH_OK);
}

void zenith_w_matern(const double *ZENITH_RESTRICT mu, int D,
                     double kappa, double nu, float *ZENITH_RESTRICT w) {
    int k;
    if (!mu || !w || D <= 0) {
        zenith__set_error(ZENITH_ERR_ARG);
        return;
    }
    if (nu < 0.0) nu = 0.0;
    for (k = 0; k < D; ++k) {
        double m = (mu[k] < 0.0 ? 0.0 : mu[k]) + kappa * kappa;
        if (m < ZENITH_EPS_DC) m = ZENITH_EPS_DC;
        w[k] = zenith__weight_clamp(pow(m, nu));
    }
    zenith__set_error(ZENITH_OK);
}

void zenith_w_fractional(const double *ZENITH_RESTRICT mu, int D, double s,
                         float *ZENITH_RESTRICT w) {
    int k;
    if (!mu || !w || D <= 0) {
        zenith__set_error(ZENITH_ERR_ARG);
        return;
    }
    if (s < 0.0) s = 0.0;
    for (k = 0; k < D; ++k) {
        double m = mu[k] < ZENITH_EPS_DC ? ZENITH_EPS_DC : mu[k];
        w[k] = zenith__weight_clamp(pow(m, s));
    }
    zenith__set_error(ZENITH_OK);
}

void zenith_w_roughvol(const double *ZENITH_RESTRICT mu, int D, double H,
                       float *ZENITH_RESTRICT w) {
    int k;
    double expn;
    if (!mu || !w || D <= 0) {
        zenith__set_error(ZENITH_ERR_ARG);
        return;
    }
    expn = 2.0 * H + 1.0;
    if (expn < 0.0) expn = 0.0;
    for (k = 0; k < D; ++k) {
        double m = mu[k] < ZENITH_EPS_DC ? ZENITH_EPS_DC : mu[k];
        w[k] = zenith__weight_clamp(pow(m, expn));
    }
    zenith__set_error(ZENITH_OK);
}

void zenith_sba_matern_envelope(const double *mu, int D,
                                const double *kappas, const double *nus, int nk,
                                float *density) {
    int i, j;
    if (!mu || !kappas || !nus || !density || D <= 0 || nk <= 0) {
        zenith__set_error(ZENITH_ERR_ARG);
        return;
    }
    for (i = 0; i < D; ++i) {
        double acc = 0.0;
        for (j = 0; j < nk; ++j) {
            double m = mu[i] + kappas[j] * kappas[j];
            if (m < ZENITH_EPS_DC) m = ZENITH_EPS_DC;
            acc += pow(m, -nus[j]);
        }
        density[i] = isfinite(acc) ? (float)acc : FLT_MAX;
    }
    zenith__set_error(ZENITH_OK);
}

int zenith_sba_perm(const float *density, int D, int nbits, uint8_t *perm) {
    /* Simple greedy: take the highest-density frequencies, no pure repeats.
     * If you want multi-bit on the same coordinate you need different
     * random projections, not the same sign. */
    int i, b = 0;
    uint8_t used[256];
    if (D <= 0 || D > 256 || nbits <= 0 || nbits > 64 || !density || !perm) {
        zenith__set_error(ZENITH_ERR_ARG);
        return ZENITH_ERR_ARG;
    }
    memset(used, 0, sizeof(used));
    while (b < nbits) {
        int best = -1;
        float best_d = -FLT_MAX;
        for (i = 0; i < D; ++i) {
            float d = density[i];
            if (!isfinite(d)) continue;
            if (!used[i] && d > best_d) {
                best_d = d;
                best = i;
            }
        }
        if (best < 0) {
            zenith__set_error(ZENITH_ERR_ARG);
            return ZENITH_ERR_ARG;
        }
        perm[b++] = (uint8_t)best;
        used[best] = 1;
    }
    zenith__set_error(ZENITH_OK);
    return ZENITH_OK;
}

#endif /* ZENITH_IMPLEMENTATION */
#endif /* ZENITH_H */
