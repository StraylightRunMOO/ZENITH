#ifndef ZENITH_TEST_COMMON_H
#define ZENITH_TEST_COMMON_H

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#define ZTEST_CHECK(cond) do { \
    if (!(cond)) { \
        fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
        return 1; \
    } \
} while (0)

#define ZTEST_CHECK_MSG(cond, msg) do { \
    if (!(cond)) { \
        fprintf(stderr, "FAIL %s:%d: %s (%s)\n", __FILE__, __LINE__, #cond, msg); \
        return 1; \
    } \
} while (0)

#if defined(__GNUC__) || defined(__clang__)
#  define ZTEST_UNUSED __attribute__((unused))
#else
#  define ZTEST_UNUSED
#endif

static uint64_t ztest_rng_state = UINT64_C(0x9e3779b97f4a7c15);

static uint64_t ztest_splitmix64(void) {
    uint64_t z = (ztest_rng_state += UINT64_C(0x9e3779b97f4a7c15));
    z = (z ^ (z >> 30)) * UINT64_C(0xbf58476d1ce4e5b9);
    z = (z ^ (z >> 27)) * UINT64_C(0x94d049bb133111eb);
    return z ^ (z >> 31);
}

static float ztest_uniform(void) {
    return (float)((ztest_splitmix64() >> 11) * (1.0 / 9007199254740992.0));
}

static float ztest_gauss(void) {
    float u1 = ztest_uniform();
    float u2 = ztest_uniform();
    if (u1 < 1e-12f) u1 = 1e-12f;
    return sqrtf(-2.0f * logf(u1)) * cosf(6.28318530717958647692f * u2);
}

static void ztest_seed(uint64_t seed) {
    ztest_rng_state = seed ? seed : UINT64_C(0x9e3779b97f4a7c15);
}

static void ztest_make_gaussian(float *v, uint32_t N, uint32_t D) {
    uint32_t i, j;
    for (i = 0; i < N; ++i)
        for (j = 0; j < D; ++j)
            v[(size_t)i * D + j] = ztest_gauss();
}

static ZTEST_UNUSED void ztest_make_clustered(float *v, uint32_t N, uint32_t D, uint32_t nclusters) {
    float *centers;
    uint32_t i, j, c;
    if (nclusters == 0) nclusters = 1;
    centers = (float *)malloc(sizeof(float) * (size_t)nclusters * D);
    if (!centers) return;
    for (c = 0; c < nclusters; ++c)
        for (j = 0; j < D; ++j)
            centers[(size_t)c * D + j] = 3.0f * ztest_gauss();
    for (i = 0; i < N; ++i) {
        c = i % nclusters;
        for (j = 0; j < D; ++j)
            v[(size_t)i * D + j] = centers[(size_t)c * D + j] + 0.15f * ztest_gauss();
    }
    free(centers);
}

static inline int ztest_float_close(float a, float b, float atol, float rtol) {
    float diff = fabsf(a - b);
    float scale = fmaxf(fabsf(a), fabsf(b));
    return diff <= atol + rtol * scale;
}

static ZTEST_UNUSED uint32_t ztest_brute_weighted(const zenith_index_t *idx,
                                     const float *vectors,
                                     const float *q,
                                     const float *full_w,
                                     uint32_t k,
                                     uint32_t *out_ids,
                                     float *out_dist) {
    float *qc = (float *)malloc(sizeof(float) * idx->D);
    float *xc = (float *)malloc(sizeof(float) * idx->D);
    double *ds = (double *)malloc(sizeof(double) * idx->N);
    uint32_t *ids = (uint32_t *)malloc(sizeof(uint32_t) * idx->N);
    uint32_t i, j, n;
    float qr;
    if (!qc || !xc || !ds || !ids) {
        free(qc); free(xc); free(ds); free(ids);
        return 0;
    }
    for (i = 0; i < idx->N; ++i) ids[i] = i;
    if (zenith_project(idx, q, qc, &qr) != ZENITH_OK) {
        free(qc); free(xc); free(ds); free(ids);
        return 0;
    }
    for (i = 0; i < idx->N; ++i) {
        double d = 0.0;
        float xr;
        zenith_project(idx, vectors + (size_t)i * idx->D, xc, &xr);
        for (j = 0; j < idx->Mcoef; ++j) {
            double e = (double)qc[j] - (double)xc[j];
            d += (double)full_w[j] * e * e;
        }
        if (idx->Mcoef < idx->D) {
            double dr = (double)xr - (double)qr;
            d += (double)zenith_tail_min(full_w, (int)idx->D, (int)idx->Mcoef) * dr * dr;
        }
        ds[i] = d;
    }
    for (i = 0; i < k && i < idx->N; ++i) {
        uint32_t best = i;
        for (n = i + 1; n < idx->N; ++n)
            if (ds[n] < ds[best]) best = n;
        {
            double td;
            uint32_t ti;
            td = ds[i]; ds[i] = ds[best]; ds[best] = td;
            ti = ids[i]; ids[i] = ids[best]; ids[best] = ti;
            out_ids[i] = ids[i];
            if (out_dist) out_dist[i] = (float)sqrt(ds[i]);
        }
    }
    free(qc); free(xc); free(ds); free(ids);
    return k < idx->N ? k : idx->N;
}

#endif
