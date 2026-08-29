#define ZENITH_IMPLEMENTATION
#include "zenith.h"
#include "test_common.h"

static int test_gray_roundtrip(void) {
    static const uint64_t vals[] = {
        0u, 1u, 2u, 3u, 0x7fffffffu, 0x80000000u,
        UINT64_C(0xffffffffffffffff), UINT64_C(0x9e3779b97f4a7c15)
    };
    uint32_t i;
    for (i = 0; i < sizeof(vals) / sizeof(vals[0]); ++i)
        ZTEST_CHECK(zenith_ungray(zenith_gray(vals[i])) == vals[i]);
    for (i = 0; i < 10000; ++i) {
        uint64_t x = ztest_splitmix64();
        ZTEST_CHECK(zenith_ungray(zenith_gray(x)) == x);
    }
    return 0;
}

static int test_projection_and_residual(void) {
    enum { N = 12, D = 32, M = 16 };
    float vectors[N * D];
    float qc[M];
    float residual = -1.0f;
    zenith_opts opts;
    zenith_index_t *idx;
    uint32_t i, j;

    ztest_seed(101);
    ztest_make_gaussian(vectors, N, D);
    memset(&opts, 0, sizeof(opts));
    opts.Mcoef = M;
    opts.nbits = M;
    opts.gen = ZENITH_GEN_SIGNED_DCT;
    opts.seed = 1234;
    opts.search = ZENITH_SEARCH_GRAY;
    idx = zenith_build(vectors, N, D, opts);
    ZTEST_CHECK(idx != NULL);

    for (i = 0; i < (uint32_t)idx->plan_M; ++i) {
        const float *row = idx->plan_t + (size_t)i * D;
        double nrm = 0.0;
        for (j = 0; j < D; ++j) nrm += (double)row[j] * row[j];
        ZTEST_CHECK(fabs(nrm - 1.0) < 2e-5);
    }

    ZTEST_CHECK(zenith_project(idx, vectors, qc, &residual) == ZENITH_OK);
    {
        double norm = 0.0, lead = 0.0, expect;
        for (j = 0; j < D; ++j) norm += (double)vectors[j] * vectors[j];
        for (j = 0; j < M; ++j) lead += (double)qc[j] * qc[j];
        expect = norm - lead;
        if (expect < 0.0) expect = 0.0;
        ZTEST_CHECK(fabs((double)residual - sqrt(expect)) < 2e-5);
    }
    zenith_free(idx);
    return 0;
}

static int test_exact_full_space_queries(void) {
    enum { N = 192, D = 24, K = 7 };
    float vectors[N * D];
    float queries[16 * D];
    double mu[D];
    float weight[D];
    zenith_opts opts;
    zenith_index_t *idx;
    uint32_t qi, r;

    ztest_seed(202);
    ztest_make_gaussian(vectors, N, D);
    ztest_make_gaussian(queries, 16, D);
    zenith_laplacian_eigs(D, mu);
    zenith_w_sobolev(mu, D, 0.7, weight);

    memset(&opts, 0, sizeof(opts));
    opts.Mcoef = D;
    opts.nbits = D;
    opts.gen = ZENITH_GEN_SIGNED_DCT;
    opts.seed = 77;
    opts.search = ZENITH_SEARCH_MIH;
    idx = zenith_build(vectors, N, D, opts);
    ZTEST_CHECK(idx != NULL);

    for (qi = 0; qi < 16; ++qi) {
        uint32_t got[K], truth[K];
        float got_d[K], truth_d[K];
        ZTEST_CHECK(zenith_query_ef(idx, queries + (size_t)qi * D, weight,
                                    K, N, got, got_d) == K);
        ZTEST_CHECK(ztest_brute_weighted(idx, vectors, queries + (size_t)qi * D,
                                         weight, K, truth, truth_d) == K);
        for (r = 0; r < K; ++r) {
            ZTEST_CHECK(got[r] == truth[r]);
            if (!ztest_float_close(got_d[r], truth_d[r], 2e-5f, 2e-5f)) {
                fprintf(stderr, "q=%u r=%u id=%u got=%.9g truth=%.9g\n",
                        qi, r, got[r], got_d[r], truth_d[r]);
                return 1;
            }
        }
    }
    zenith_free(idx);
    return 0;
}

static int test_reported_distance_is_lower_bound(void) {
    enum { N = 160, D = 32, M = 12, K = 8 };
    float vectors[N * D];
    float q[D];
    float weight[D];
    uint32_t ids[K];
    float lb[K], exact[K], exact_all[K];
    zenith_opts opts;
    zenith_index_t *idx;
    uint32_t n, r;

    ztest_seed(303);
    ztest_make_clustered(vectors, N, D, 6);
    ztest_make_gaussian(q, 1, D);
    zenith_w_identity(D, weight);
    memset(&opts, 0, sizeof(opts));
    opts.Mcoef = M;
    opts.nbits = 16;
    opts.gen = ZENITH_GEN_SIGNED_DCT;
    opts.seed = 9;
    opts.search = ZENITH_SEARCH_HYBRID;
    idx = zenith_build(vectors, N, D, opts);
    ZTEST_CHECK(idx != NULL);
    n = zenith_query_ef(idx, q, weight, K, 48, ids, lb);
    ZTEST_CHECK(n == K);
    ZTEST_CHECK(zenith_rerank_exact(idx, vectors, q, weight, ids, n, K,
                                    ids, exact) == K);
    for (r = 0; r < K; ++r)
        ZTEST_CHECK(lb[r] <= exact[r] + 2e-4f);

    /* With ef=N the candidate generator must become exhaustive over the
     * retained subspace.  Distances remain lower bounds when M<D. */
    n = zenith_query_ef(idx, q, weight, K, N, ids, lb);
    ZTEST_CHECK(n == K);
    ZTEST_CHECK(zenith_rerank_exact(idx, vectors, q, weight, ids, n, K,
                                    ids, exact_all) == K);
    for (r = 0; r < K; ++r)
        ZTEST_CHECK(lb[r] <= exact_all[r] + 2e-4f);
    zenith_free(idx);
    return 0;
}

static int test_coefficient_query_contract(void) {
    enum { N = 64, D = 32 };
    float vectors[N * D];
    float q[D], qc[16];
    float weight[D];
    uint32_t ids[4];
    float dist[4];
    zenith_opts opts;
    zenith_index_t *idx;

    ztest_seed(404);
    ztest_make_gaussian(vectors, N, D);
    ztest_make_gaussian(q, 1, D);
    zenith_w_identity(D, weight);

    memset(&opts, 0, sizeof(opts));
    opts.Mcoef = 16;
    opts.nbits = 32; /* sketch uses coefficients the caller did not supply */
    opts.search = ZENITH_SEARCH_GRAY;
    idx = zenith_build(vectors, N, D, opts);
    ZTEST_CHECK(idx != NULL);
    ZTEST_CHECK(zenith_project(idx, q, qc, NULL) == ZENITH_OK);
    ZTEST_CHECK(zenith_query_coeffs(idx, qc, 0.0f, weight, 4, 32, ids, dist) == 0);
    ZTEST_CHECK(zenith_last_error() == ZENITH_ERR_ARG);
    zenith_free(idx);

    memset(&opts, 0, sizeof(opts));
    opts.Mcoef = D;
    opts.nbits = D;
    opts.search = ZENITH_SEARCH_GRAY;
    idx = zenith_build(vectors, N, D, opts);
    ZTEST_CHECK(idx != NULL);
    {
        float full_qc[D];
        uint32_t a[4], b[4];
        float da[4], db[4];
        ZTEST_CHECK(zenith_project(idx, q, full_qc, NULL) == ZENITH_OK);
        ZTEST_CHECK(zenith_query_ef(idx, q, weight, 4, N, a, da) == 4);
        ZTEST_CHECK(zenith_query_coeffs(idx, full_qc, 0.0f, weight, 4, N, b, db) == 4);
        ZTEST_CHECK(memcmp(a, b, sizeof(a)) == 0);
        ZTEST_CHECK(memcmp(da, db, sizeof(da)) == 0);
    }
    zenith_free(idx);
    return 0;
}

static int test_invalid_inputs_fail_closed(void) {
    enum { N = 16, D = 8 };
    float vectors[N * D];
    float q[D];
    float weight[D];
    uint32_t ids[2];
    zenith_opts opts;
    zenith_index_t *idx;

    ztest_seed(505);
    ztest_make_gaussian(vectors, N, D);
    ztest_make_gaussian(q, 1, D);
    zenith_w_identity(D, weight);
    memset(&opts, 0, sizeof(opts));
    idx = zenith_build(vectors, N, D, opts);
    ZTEST_CHECK(idx != NULL);

    weight[0] = -1.0f;
    ZTEST_CHECK(zenith_query_ef(idx, q, weight, 2, 8, ids, NULL) == 0);
    ZTEST_CHECK(zenith_last_error() == ZENITH_ERR_WEIGHT);
    weight[0] = 1.0f;
    q[0] = NAN;
    ZTEST_CHECK(zenith_query_ef(idx, q, weight, 2, 8, ids, NULL) == 0);
    ZTEST_CHECK(zenith_last_error() == ZENITH_ERR_ARG);
    q[0] = 0.0f;

    vectors[3] = INFINITY;
    ZTEST_CHECK(zenith_build(vectors, N, D, opts) == NULL);
    ZTEST_CHECK(zenith_last_error() == ZENITH_ERR_ARG);
    zenith_free(idx);
    return 0;
}

int main(void) {
    ztest_seed(1);
    if (test_gray_roundtrip()) return 1;
    if (test_projection_and_residual()) return 1;
    if (test_exact_full_space_queries()) return 1;
    if (test_reported_distance_is_lower_bound()) return 1;
    if (test_coefficient_query_contract()) return 1;
    if (test_invalid_inputs_fail_closed()) return 1;
    puts("test_core: OK");
    return 0;
}
