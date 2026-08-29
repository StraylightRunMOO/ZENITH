#ifndef ZENITH_DCT_H
#define ZENITH_DCT_H

/* Dense orthonormal DCT-II projector, with an optional FFTW3 backend for
 * large (D, M) where O(D log D) beats the M×D matrix-vector product.
 * FFTW uses REDFT10; outputs are rescaled to the same orthonormal convention
 * as the dense plan. This is a shared backend, not an ISA-specific kernel. */

#ifndef ZENITH_FFTW_MIN_D
#define ZENITH_FFTW_MIN_D 64u
#endif

#if defined(ZENITH_USE_FFTW)
#  include <fftw3.h>
#endif
#if defined(_OPENMP)
#  include <omp.h>
#endif

int zenith_fftw_enabled(void) {
#if defined(ZENITH_USE_FFTW)
    return 1;
#else
    return 0;
#endif
}

static int zenith__should_fftw(uint32_t D, uint32_t M, int policy) {
#if defined(ZENITH_USE_FFTW)
    double logd;
    uint32_t t;
    if (policy < 0) return 0;
    if (policy > 0) return D >= 2u && M >= 1u;
    if (D < ZENITH_FFTW_MIN_D) return 0;
    logd = 0.0;
    t = D;
    while (t > 1u) { logd += 1.0; t >>= 1; }
    return (double)M >= 2.0 * logd + 8.0;
#else
    (void)D; (void)M; (void)policy;
    return 0;
#endif
}

/* Planning happens only on the build/load thread. OpenMP workers only execute
 * an existing plan via fftwf_execute_r2r, which is allowed on distinct arrays. */

static void zenith_dct_plan_free(zenith_index_t *idx) {
    if (!idx) return;
    if (idx->plan_t) {
        ZENITH_ALIGNED_FREE(idx->plan_t);
        idx->plan_t = NULL;
    }
#if defined(ZENITH_USE_FFTW)
    if (idx->fftw_plan) {
        fftwf_destroy_plan((fftwf_plan)idx->fftw_plan);
        idx->fftw_plan = NULL;
    }
    if (idx->fftw_in) { fftwf_free(idx->fftw_in); idx->fftw_in = NULL; }
    if (idx->fftw_out) { fftwf_free(idx->fftw_out); idx->fftw_out = NULL; }
#endif
    if (idx->fftw_scale) {
        ZENITH_FREE(idx->fftw_scale);
        idx->fftw_scale = NULL;
    }
    idx->use_fftw = 0;
}

static int zenith__fftw_plan_init(zenith_index_t *idx, int D, int M) {
#if defined(ZENITH_USE_FFTW)
    int k;
    fftwf_plan plan;
    idx->fftw_in = (float *)fftwf_malloc(sizeof(float) * (size_t)D);
    idx->fftw_out = (float *)fftwf_malloc(sizeof(float) * (size_t)D);
    idx->fftw_scale = (float *)ZENITH_MALLOC(sizeof(float) * (size_t)D);
    if (!idx->fftw_in || !idx->fftw_out || !idx->fftw_scale) return ZENITH_ERR_ALLOC;
    plan = fftwf_plan_r2r_1d(D, idx->fftw_in, idx->fftw_out, FFTW_REDFT10,
                             FFTW_ESTIMATE);
    if (!plan) return ZENITH_ERR_ALLOC;
    idx->fftw_plan = (void *)plan;
    idx->fftw_scale[0] = (float)(0.5 * sqrt(1.0 / (double)D));
    for (k = 1; k < D; ++k)
        idx->fftw_scale[k] = (float)(0.5 * sqrt(2.0 / (double)D));
    idx->plan_D = D;
    idx->plan_M = M;
    idx->use_fftw = 1;
    return ZENITH_OK;
#else
    (void)idx; (void)D; (void)M;
    return ZENITH_ERR_ARG;
#endif
}

static int zenith_dct_plan_init(zenith_index_t *idx, int D, int M, int fftw_policy) {
    int k, n;
    if (D <= 0 || M <= 0 || M > D) return ZENITH_ERR_ARG;
    idx->plan_D = D;
    idx->plan_M = M;
    idx->use_fftw = 0;
    if (zenith__should_fftw((uint32_t)D, (uint32_t)M, fftw_policy)) {
        int rc = zenith__fftw_plan_init(idx, D, M);
        if (rc == ZENITH_OK) return ZENITH_OK;
        zenith_dct_plan_free(idx);
        if (fftw_policy > 0) return rc;
    }
    if ((size_t)M > ((size_t)-1) / (size_t)D / sizeof(float))
        return ZENITH_ERR_ALLOC;
    idx->plan_t = (float *)ZENITH_ALIGNED_ALLOC(64, (size_t)M * (size_t)D * sizeof(float));
    if (!idx->plan_t) return ZENITH_ERR_ALLOC;
    for (k = 0; k < M; ++k) {
        double a = (k == 0) ? sqrt(1.0 / D) : sqrt(2.0 / D);
        float *row = idx->plan_t + (size_t)k * D;
        for (n = 0; n < D; ++n)
            row[n] = (float)(a * cos(ZENITH_PI / D * ((double)n + 0.5) * (double)k));
    }
    return ZENITH_OK;
}

static void zenith_dct_plan_apply_signs(zenith_index_t *idx) {
    int k, n;
    if (!idx || !idx->input_signs || !idx->plan_t) return;
    for (k = 0; k < idx->plan_M; ++k) {
        float *row = idx->plan_t + (size_t)k * (size_t)idx->plan_D;
        for (n = 0; n < idx->plan_D; ++n)
            row[n] *= (float)idx->input_signs[n];
    }
}

static void zenith__dct_fftw(const zenith_index_t *idx,
                             const float *ZENITH_RESTRICT x,
                             float *ZENITH_RESTRICT out,
                             int M) {
#if defined(ZENITH_USE_FFTW)
    float *in = idx->fftw_in, *yo = idx->fftw_out;
    int D = idx->plan_D, k, n;
    int scratch = 0;
    if (M > idx->plan_M) M = idx->plan_M;
#if defined(_OPENMP)
    if (omp_in_parallel()) {
        in = (float *)fftwf_malloc(sizeof(float) * (size_t)D);
        yo = (float *)fftwf_malloc(sizeof(float) * (size_t)D);
        scratch = 1;
        if (!in || !yo) {
            if (in) fftwf_free(in);
            if (yo) fftwf_free(yo);
            for (k = 0; k < M; ++k) out[k] = 0.0f;
            return;
        }
    }
#endif
    if (idx->input_signs) {
        for (n = 0; n < D; ++n)
            in[n] = x[n] * (float)idx->input_signs[n];
    } else {
        memcpy(in, x, sizeof(float) * (size_t)D);
    }
    fftwf_execute_r2r((fftwf_plan)idx->fftw_plan, in, yo);
    for (k = 0; k < M; ++k)
        out[k] = yo[k] * idx->fftw_scale[k];
    if (scratch) {
        fftwf_free(in);
        fftwf_free(yo);
    }
#else
    (void)idx; (void)x; (void)out; (void)M;
#endif
}

static void zenith_dct_lead_signed(const zenith_index_t *idx,
                                   const float *ZENITH_RESTRICT x,
                                   float *ZENITH_RESTRICT out,
                                   int M) {
    int D, k;
    if (idx->use_fftw) {
        zenith__dct_fftw(idx, x, out, M);
        return;
    }
    D = idx->plan_D;
    if (M > idx->plan_M) M = idx->plan_M;
    for (k = 0; k < M; ++k)
        out[k] = zenith__dot(idx->plan_t + (size_t)k * (size_t)D, x, (uint32_t)D);
}

static void zenith_dct_lead(const zenith_index_t *idx,
                            const float *ZENITH_RESTRICT x,
                            float *ZENITH_RESTRICT out) {
    zenith_dct_lead_signed(idx, x, out, idx->plan_M);
}

#endif /* ZENITH_DCT_H */
