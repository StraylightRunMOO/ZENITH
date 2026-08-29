#define ZENITH_IMPLEMENTATION
#include "zenith.h"
#include "test_common.h"

static void idct4(const float *coef, float *x) {
    uint32_t j, k;
    for (j = 0; j < 4; ++j) {
        double s = 0.0;
        for (k = 0; k < 4; ++k) {
            double a = k == 0 ? sqrt(0.25) : sqrt(0.5);
            s += (double)coef[k] * a *
                 cos(ZENITH_PI / 4.0 * ((double)j + 0.5) * (double)k);
        }
        x[j] = (float)s;
    }
}

static int test_posterior_beats_equal_weight_hamming(void) {
    float acoef[4] = {-1.0f,  1.0f,  1.0f,  1.0f};
    float bcoef[4] = { 1.0f, -1.0f, -1.0f, -1.0f};
    float qcoef[4] = {10.0f,  1.0f,  1.0f,  1.0f};
    float vectors[8], q[4], w[4], dist[1];
    zenith_opts opts;
    zenith_index_t *ham, *post;
    uint32_t idh[1], idp[1], pool[2];

    idct4(acoef, vectors);
    idct4(bcoef, vectors + 4);
    idct4(qcoef, q);
    zenith_w_identity(4, w);

    zenith_opts_init(&opts);
    opts.Mcoef = 4;
    opts.nbits = 4;
    opts.gen = ZENITH_GEN_DCT;
    opts.search = ZENITH_SEARCH_HAMMING;
    ham = zenith_build(vectors, 2, 4, opts);
    opts.search = ZENITH_SEARCH_POSTERIOR;
    post = zenith_build(vectors, 2, 4, opts);
    ZTEST_CHECK(ham && post);
    ZTEST_CHECK(zenith_query_ef(ham, q, w, 1, 1, idh, dist) == 1);
    ZTEST_CHECK(zenith_query_ef(post, q, w, 1, 1, idp, dist) == 1);
    ZTEST_CHECK(idh[0] == 0);
    ZTEST_CHECK(idp[0] == 1);
    ZTEST_CHECK(zenith_candidates_ef(ham, q, w, 1, pool) == 1);
    ZTEST_CHECK(pool[0] == 0);
    ZTEST_CHECK(zenith_candidates_ef(post, q, w, 1, pool) == 1);
    ZTEST_CHECK(pool[0] == 1);
    ZTEST_CHECK(zenith_candidates_ef(post, q, w, 2, pool) == 2);
    ZTEST_CHECK((pool[0] == 0 && pool[1] == 1) ||
                (pool[0] == 1 && pool[1] == 0));
    zenith_free(ham);
    zenith_free(post);
    return 0;
}

static int test_candidates_and_persistence(void) {
    enum { N = 257, D = 32, EF = 37, K = 10 };
    float *vectors = (float *)malloc(sizeof(float) * N * D);
    float q[D], w[D], dist[K], dist2[K];
    uint32_t pool[EF], ids[K], ids2[K];
    zenith_opts opts;
    zenith_index_t *idx, *loaded;
    const char *path = "test_gaussian_modes.zni";
    uint32_t i, j, nc, n1, n2;

    ZTEST_CHECK(vectors != NULL);
    ztest_seed(99173);
    ztest_make_gaussian(vectors, N, D);
    for (j = 0; j < D; ++j) q[j] = ztest_gauss();
    zenith_w_identity(D, w);

    zenith_opts_init(&opts);
    opts.Mcoef = D;
    opts.nbits = 32;
    opts.seed = 1717;
    opts.gen = ZENITH_GEN_SIGNED_DCT;
    opts.search = ZENITH_SEARCH_POSTERIOR;
    idx = zenith_build(vectors, N, D, opts);
    ZTEST_CHECK(idx != NULL);
    nc = zenith_candidates_ef(idx, q, w, EF, pool);
    ZTEST_CHECK(nc == EF);
    for (i = 0; i < nc; ++i)
        for (j = i + 1; j < nc; ++j)
            ZTEST_CHECK(pool[i] != pool[j]);
    n1 = zenith_query_ef(idx, q, w, K, EF, ids, dist);
    ZTEST_CHECK(n1 == K);
    ZTEST_CHECK(zenith_save(idx, path) == ZENITH_OK);
    loaded = zenith_load(path);
    ZTEST_CHECK(loaded != NULL);
    ZTEST_CHECK(loaded->search == ZENITH_SEARCH_POSTERIOR);
    n2 = zenith_query_ef(loaded, q, w, K, EF, ids2, dist2);
    ZTEST_CHECK(n2 == n1);
    for (i = 0; i < n1; ++i) {
        ZTEST_CHECK(ids2[i] == ids[i]);
        ZTEST_CHECK(ztest_float_close(dist2[i], dist[i], 1e-5f, 1e-5f));
    }
    zenith_unload(loaded);
    zenith_free(idx);
    remove(path);
    free(vectors);
    return 0;
}

static int test_posterior_rejects_polar(void) {
    zenith_opts opts;
    zenith_index_t *idx;
    float v[16];
    uint32_t i;

    for (i = 0; i < 16; ++i) v[i] = (float)i;
    zenith_opts_init(&opts);
    opts.Mcoef = 4;
    opts.nbits = 4;
    opts.use_polar = 1;
    opts.search = ZENITH_SEARCH_POSTERIOR;
    idx = zenith_build(v, 4, 4, opts);
    ZTEST_CHECK(idx == NULL);
    ZTEST_CHECK(zenith_last_error() == ZENITH_ERR_ARG);
    return 0;
}

int main(void) {
    if (test_posterior_beats_equal_weight_hamming()) return 1;
    if (test_candidates_and_persistence()) return 1;
    if (test_posterior_rejects_polar()) return 1;
    puts("test_gaussian_modes: OK");
    return 0;
}
