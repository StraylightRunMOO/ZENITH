#define ZENITH_IMPLEMENTATION
#include "zenith.h"
#include "test_common.h"

static int test_whitening_equivalence(void) {
    enum { N = 128, D = 32, M = 16, K = 6 };
    float vectors[N * D], q[D], baseline[D];
    double mu[D];
    zenith_opts opts;
    zenith_index_t *plain = NULL, *white = NULL;
    uint32_t a[K], b[K];
    float da[K], db[K];
    uint32_t i, na, nb;

    ztest_seed(606);
    ztest_make_clustered(vectors, N, D, 5);
    ztest_make_gaussian(q, 1, D);
    zenith_laplacian_eigs(D, mu);
    zenith_w_sobolev(mu, D, 0.8, baseline);

    memset(&opts, 0, sizeof(opts));
    opts.Mcoef = M;
    opts.nbits = 16;
    opts.seed = 42;
    opts.gen = ZENITH_GEN_SIGNED_DCT;
    opts.search = ZENITH_SEARCH_GRAY;
    plain = zenith_build(vectors, N, D, opts);
    ZTEST_CHECK(plain != NULL);

    opts.whiten_baseline = baseline;
    white = zenith_build(vectors, N, D, opts);
    ZTEST_CHECK(white != NULL);

    na = zenith_query_ef(plain, q, baseline, K, N, a, da);
    nb = zenith_query_ef(white, q, baseline, K, N, b, db);
    ZTEST_CHECK(na == K && nb == K);
    for (i = 0; i < K; ++i) {
        ZTEST_CHECK(a[i] == b[i]);
        ZTEST_CHECK(ztest_float_close(da[i], db[i], 3e-5f, 3e-5f));
    }
    zenith_free(plain);
    zenith_free(white);
    return 0;
}

static int test_permutation_validation_and_determinism(void) {
    enum { N = 96, D = 32, M = 20 };
    float vectors[N * D];
    zenith_opts opts;
    zenith_index_t *a = NULL, *b = NULL;
    uint32_t i;

    ztest_seed(707);
    ztest_make_gaussian(vectors, N, D);
    memset(&opts, 0, sizeof(opts));
    opts.Mcoef = M;
    opts.nbits = 16;
    opts.use_perm = 1;
    for (i = 0; i < 16; ++i) opts.freq_perm[i] = (uint8_t)(15 - i);
    a = zenith_build(vectors, N, D, opts);
    ZTEST_CHECK(a != NULL);
    b = zenith_build(vectors, N, D, opts);
    ZTEST_CHECK(b != NULL);
    ZTEST_CHECK(memcmp(a->freq_perm, b->freq_perm, 64) == 0);
    ZTEST_CHECK(memcmp(a->keys, b->keys, sizeof(uint64_t) * N) == 0);
    zenith_free(a);
    zenith_free(b);

    opts.freq_perm[3] = opts.freq_perm[0];
    ZTEST_CHECK(zenith_build(vectors, N, D, opts) == NULL);
    ZTEST_CHECK(zenith_last_error() == ZENITH_ERR_PERM);
    opts.freq_perm[3] = 12;
    opts.freq_perm[4] = M;
    ZTEST_CHECK(zenith_build(vectors, N, D, opts) == NULL);
    ZTEST_CHECK(zenith_last_error() == ZENITH_ERR_PERM);

    memset(&opts, 0, sizeof(opts));
    opts.Mcoef = M;
    opts.nbits = 16;
    opts.auto_perm = 1;
    opts.search = ZENITH_SEARCH_GRAY;
    a = zenith_build(vectors, N, D, opts);
    b = zenith_build(vectors, N, D, opts);
    ZTEST_CHECK(a != NULL && b != NULL);
    ZTEST_CHECK(memcmp(a->freq_perm, b->freq_perm, 64) == 0);
    ZTEST_CHECK(memcmp(a->keys, b->keys, sizeof(uint64_t) * N) == 0);
    zenith_free(a);
    zenith_free(b);
    return 0;
}

static int test_polar_permutation_semantics(void) {
    float x[8] = {1.0f, -1.0f, 2.0f, -2.0f, 3.0f, -3.0f, 4.0f, -4.0f};
    uint8_t perm[2] = {1, 0}; /* pair is (-1,+1), quadrant code 2 -> Gray 3 */
    uint64_t h = zenith_sketch_polar(x, 1, perm);
    ZTEST_CHECK(h == 3u);
    h = zenith_sketch_polar(x, 1, NULL);
    ZTEST_CHECK(h == 1u); /* pair (+1,-1), quadrant code 1 -> Gray 1 */
    return 0;
}

static int test_weight_constructors(void) {
    enum { D = 32 };
    double mu[D];
    float w[D];
    int i;

    zenith_laplacian_eigs(D, mu);
    for (i = 1; i < D; ++i)
        ZTEST_CHECK(mu[i] >= mu[i - 1]);

    zenith_w_identity(D, w);
    for (i = 0; i < D; ++i) ZTEST_CHECK(w[i] == 1.0f);

    zenith_w_sobolev(mu, D, 1.25, w);
    for (i = 0; i < D; ++i) {
        ZTEST_CHECK(isfinite(w[i]) && w[i] > 0.0f);
        if (i) ZTEST_CHECK(w[i] >= w[i - 1]);
    }

    zenith_w_matern(mu, D, 0.2, 1.5, w);
    for (i = 0; i < D; ++i) ZTEST_CHECK(isfinite(w[i]) && w[i] > 0.0f);

    zenith_w_fractional(mu, D, 0.7, w);
    for (i = 0; i < D; ++i) ZTEST_CHECK(isfinite(w[i]) && w[i] > 0.0f);

    zenith_w_roughvol(mu, D, 0.25, w);
    for (i = 0; i < D; ++i) ZTEST_CHECK(isfinite(w[i]) && w[i] > 0.0f);

    w[0] = 5.0f; w[1] = 2.0f; w[2] = 4.0f;
    ZTEST_CHECK(zenith_tail_min(w, 3, 0) == 2.0f);
    ZTEST_CHECK(zenith_tail_min(w, 3, 1) == 2.0f);
    ZTEST_CHECK(zenith_tail_min(w, 3, 3) == 0.0f);
    ZTEST_CHECK(zenith_last_error() == ZENITH_OK);
    return 0;
}

static int test_spectral_bit_allocation(void) {
    enum { D = 24 };
    double mu[D], kappas[2] = {0.1, 1.0}, nus[2] = {0.5, 1.5};
    float density[D];
    uint8_t perm[16];
    int i;

    zenith_laplacian_eigs(D, mu);
    zenith_sba_matern_envelope(mu, D, kappas, nus, 2, density);
    ZTEST_CHECK(zenith_last_error() == ZENITH_OK);
    for (i = 0; i < D; ++i) ZTEST_CHECK(isfinite(density[i]) && density[i] > 0.0f);
    ZTEST_CHECK(zenith_sba_perm(density, D, 16, perm) == ZENITH_OK);
    for (i = 1; i < 16; ++i) {
        int j;
        for (j = 0; j < i; ++j) ZTEST_CHECK(perm[i] != perm[j]);
    }

    zenith_sba_matern_envelope(NULL, D, kappas, nus, 2, density);
    ZTEST_CHECK(zenith_last_error() == ZENITH_ERR_ARG);
    ZTEST_CHECK(zenith_sba_perm(density, D, 0, perm) == ZENITH_ERR_ARG);
    ZTEST_CHECK(zenith_sba_perm(density, 300, 16, perm) == ZENITH_ERR_ARG);
    return 0;
}

static int test_nonmonotone_query_weights(void) {
    enum { N = 80, D = 16, M = 8, K = 4 };
    float vectors[N * D], q[D], weight[D];
    uint32_t ids[K];
    float dist[K];
    zenith_opts opts;
    zenith_index_t *idx;
    int i;

    ztest_seed(1212);
    ztest_make_gaussian(vectors, N, D);
    ztest_make_gaussian(q, 1, D);
    for (i = 0; i < D; ++i) weight[i] = 1.0f;
    weight[M] = 4.0f;
    weight[M + 2] = 0.25f;
    zenith_opts_init(&opts);
    opts.Mcoef = M;
    opts.nbits = M;
    opts.search = ZENITH_SEARCH_GRAY;
    idx = zenith_build(vectors, N, D, opts);
    ZTEST_CHECK(idx != NULL);
    ZTEST_CHECK(zenith_tail_min(weight, D, M) == 0.25f);
    ZTEST_CHECK(zenith_query_ef(idx, q, weight, K, N, ids, dist) == K);
    ZTEST_CHECK(zenith_last_error() == ZENITH_OK);
    zenith_free(idx);
    return 0;
}

int main(void) {
    if (test_whitening_equivalence()) return 1;
    if (test_permutation_validation_and_determinism()) return 1;
    if (test_polar_permutation_semantics()) return 1;
    if (test_weight_constructors()) return 1;
    if (test_spectral_bit_allocation()) return 1;
    if (test_nonmonotone_query_weights()) return 1;
    puts("test_features: OK");
    return 0;
}
