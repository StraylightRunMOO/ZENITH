#define ZENITH_IMPLEMENTATION
#include "zenith.h"
#include "test_common.h"

static int test_backend_flags(void) {
    ZTEST_CHECK(zenith_simd_level() >= 0);
    ZTEST_CHECK(zenith_thread_count() >= 1);
    ZTEST_CHECK(zenith_openmp_enabled() == 0 || zenith_openmp_enabled() == 1);
    ZTEST_CHECK(zenith_fftw_enabled() == 0 || zenith_fftw_enabled() == 1);
    zenith_set_threads(1);
    ZTEST_CHECK(zenith_thread_count() == 1);
    zenith_set_threads(0);
    ZTEST_CHECK(zenith_thread_count() >= 1);
    return 0;
}

static int test_query_many_matches_serial(void) {
    enum { N = 128, D = 32, Q = 8, K = 5 };
    float vectors[N * D], queries[Q * D], w[D];
    uint32_t ids_s[Q * K], ids_b[Q * K];
    float dist_s[Q * K], dist_b[Q * K];
    zenith_opts opts;
    zenith_index_t *idx;
    uint32_t qi, r;

    ztest_seed(4242);
    ztest_make_clustered(vectors, N, D, 4);
    ztest_make_gaussian(queries, Q, D);
    zenith_w_identity(D, w);
    zenith_opts_init(&opts);
    opts.Mcoef = D;
    opts.nbits = 32;
    opts.search = ZENITH_SEARCH_GRAY;
    opts.nthreads = 1;
    idx = zenith_build(vectors, N, D, opts);
    ZTEST_CHECK(idx != NULL);
    for (qi = 0; qi < Q; ++qi)
        ZTEST_CHECK(zenith_query_ef(idx, queries + (size_t)qi * D, w, K, N,
                                    ids_s + (size_t)qi * K,
                                    dist_s + (size_t)qi * K) == K);
    ZTEST_CHECK(zenith_query_many(idx, queries, Q, w, K, N, ids_b, dist_b) == Q * K);
    for (qi = 0; qi < Q; ++qi)
        for (r = 0; r < K; ++r) {
            ZTEST_CHECK(ids_s[(size_t)qi * K + r] == ids_b[(size_t)qi * K + r]);
            ZTEST_CHECK(ztest_float_close(dist_s[(size_t)qi * K + r],
                                          dist_b[(size_t)qi * K + r],
                                          2e-5f, 2e-5f));
        }
    zenith_free(idx);
    return 0;
}

static int test_threaded_build_matches_serial(void) {
    enum { N = 96, D = 24, K = 6 };
    float vectors[N * D], q[D], w[D];
    uint32_t a[K], b[K];
    float da[K], db[K];
    zenith_opts opts;
    zenith_index_t *serial, *parallel;
    uint32_t r;

    ztest_seed(77);
    ztest_make_gaussian(vectors, N, D);
    ztest_make_gaussian(q, 1, D);
    zenith_w_identity(D, w);
    zenith_opts_init(&opts);
    opts.Mcoef = D;
    opts.nbits = D;
    opts.search = ZENITH_SEARCH_HAMMING;
    opts.nthreads = 1;
    serial = zenith_build(vectors, N, D, opts);
    opts.nthreads = 2;
    parallel = zenith_build(vectors, N, D, opts);
    ZTEST_CHECK(serial && parallel);
    ZTEST_CHECK(memcmp(serial->keys, parallel->keys, sizeof(uint64_t) * N) == 0);
    ZTEST_CHECK(zenith_query_ef(serial, q, w, K, N, a, da) == K);
    ZTEST_CHECK(zenith_query_ef(parallel, q, w, K, N, b, db) == K);
    for (r = 0; r < K; ++r) {
        ZTEST_CHECK(a[r] == b[r]);
        ZTEST_CHECK(ztest_float_close(da[r], db[r], 2e-5f, 2e-5f));
    }
    zenith_free(serial);
    zenith_free(parallel);
    return 0;
}

static int test_fftw_matches_dense(void) {
    enum { N = 32, D = 128, M = 64 };
    float vectors[N * D], q[D];
    float ca[M], cb[M];
    float ra, rb;
    zenith_opts opts;
    zenith_index_t *dense, *fast;
    uint32_t j;

    if (!zenith_fftw_enabled()) return 0;
    ztest_seed(909);
    ztest_make_gaussian(vectors, N, D);
    ztest_make_gaussian(q, 1, D);
    zenith_opts_init(&opts);
    opts.Mcoef = M;
    opts.nbits = 32;
    opts.search = ZENITH_SEARCH_GRAY;
    opts.fftw = -1;
    dense = zenith_build(vectors, N, D, opts);
    opts.fftw = 1;
    fast = zenith_build(vectors, N, D, opts);
    ZTEST_CHECK(dense && fast);
    ZTEST_CHECK(dense->use_fftw == 0);
    ZTEST_CHECK(fast->use_fftw == 1);
    ZTEST_CHECK(zenith_project(dense, q, ca, &ra) == ZENITH_OK);
    ZTEST_CHECK(zenith_project(fast, q, cb, &rb) == ZENITH_OK);
    for (j = 0; j < M; ++j)
        ZTEST_CHECK(ztest_float_close(ca[j], cb[j], 2e-4f, 2e-4f));
    ZTEST_CHECK(ztest_float_close(ra, rb, 2e-4f, 2e-4f));
    zenith_free(dense);
    zenith_free(fast);
    return 0;
}

int main(void) {
    if (test_backend_flags()) return 1;
    if (test_query_many_matches_serial()) return 1;
    if (test_threaded_build_matches_serial()) return 1;
    if (test_fftw_matches_dense()) return 1;
    puts("test_backend: OK");
    return 0;
}
