#define ZENITH_IMPLEMENTATION
#include "zenith.h"
#include "test_common.h"

static void idct8(const float *coef, float *x) {
    uint32_t j, k;
    for (j = 0; j < 8; ++j) {
        double s = 0.0;
        for (k = 0; k < 8; ++k) {
            double a = k == 0 ? sqrt(0.125) : sqrt(0.25);
            s += (double)coef[k] * a *
                 cos(ZENITH_PI / 8.0 * ((double)j + 0.5) * (double)k);
        }
        x[j] = (float)s;
    }
}

static int test_symbols_and_calibration(void) {
    static const float coef[4][8] = {
        {-2.0f, -1.0f, -0.5f, 0.0f, 0.25f, 0.5f, 1.0f, 1.5f},
        {-0.4f, -0.2f, -0.1f, 0.1f, 0.20f, 0.4f, 0.6f, 0.8f},
        { 0.4f,  0.2f,  0.1f, 0.2f, 0.30f, 0.5f, 0.7f, 0.9f},
        { 2.0f,  1.0f,  0.5f, 0.3f, 0.35f, 0.6f, 0.8f, 1.0f}
    };
    float vectors[4 * 8];
    zenith_opts opts;
    zenith_index_t *idx;
    uint32_t i;
    double mean[8], var[8];

    ZTEST_CHECK(zenith__lm2_symbol(-1.0, 1.0) == 0u);
    ZTEST_CHECK(zenith__lm2_symbol(-0.5, 1.0) == 1u);
    ZTEST_CHECK(zenith__lm2_symbol( 0.5, 1.0) == 3u);
    ZTEST_CHECK(zenith__lm2_symbol( 1.0, 1.0) == 2u);
    ZTEST_CHECK(zenith__lm2_symbol(-2.0, 2.0) == 0u);
    ZTEST_CHECK(zenith__lm2_symbol( 1.5, 2.0) == 3u);

    for (i = 0; i < 8; ++i) {
        uint32_t r;
        mean[i] = 0.0;
        var[i] = 0.0;
        for (r = 0; r < 4; ++r) mean[i] += coef[r][i];
        mean[i] /= 4.0;
        for (r = 0; r < 4; ++r) {
            double e = coef[r][i] - mean[i];
            var[i] += e * e;
        }
        var[i] /= 4.0;
    }
    for (i = 0; i < 4; ++i) idct8(coef[i], vectors + (size_t)i * 8);

    zenith_opts_init(&opts);
    opts.Mcoef = 8;
    opts.nbits = 16;
    opts.gen = ZENITH_GEN_DCT;
    opts.search = ZENITH_SEARCH_POSTERIOR;
    opts.quantizer = ZENITH_QUANT_LM2;
    idx = zenith_build(vectors, 4, 8, opts);
    ZTEST_CHECK(idx != NULL);
    ZTEST_CHECK(idx->quantizer == ZENITH_QUANT_LM2);
    ZTEST_CHECK(idx->ncodes == 8);
    for (i = 0; i < 8; ++i) {
        ZTEST_CHECK(fabs((double)idx->quant_bias[i] - mean[i]) < 2e-5);
        ZTEST_CHECK(fabs((double)idx->quant_scale[i] - sqrt(var[i])) < 2e-5);
    }
    zenith_free(idx);
    return 0;
}

static int brute_lm2_best(const zenith_index_t *idx, const float *Q,
                          const float *w) {
    uint32_t pos, b;
    uint32_t best = 0;
    double best_score = -HUGE_VAL;
    for (pos = 0; pos < idx->N; ++pos) {
        double score = 0.0;
        for (b = 0; b < idx->ncodes; ++b) {
            uint32_t symbol = (uint32_t)((idx->keys[pos] >> (2u * b)) & 3u);
            score += zenith__lm_symbol_score(idx, Q, w, b, symbol);
        }
        if (score > best_score) {
            best_score = score;
            best = pos;
        }
    }
    return (int)best;
}

static int test_lm2_posterior_matches_brute_score(void) {
    static const float coef[4][8] = {
        {-2.0f, -1.0f, -0.5f, 0.0f, 0.25f, 0.5f, 1.0f, 1.5f},
        {-0.4f, -0.2f, -0.1f, 0.1f, 0.20f, 0.4f, 0.6f, 0.8f},
        { 0.4f,  0.2f,  0.1f, 0.2f, 0.30f, 0.5f, 0.7f, 0.9f},
        { 2.0f,  1.0f,  0.5f, 0.3f, 0.35f, 0.6f, 0.8f, 1.0f}
    };
    float vectors[4 * 8], q[8], Q[8], w[8], residual;
    zenith_opts opts;
    zenith_index_t *idx;
    uint32_t got[1];
    int best;
    uint32_t i;

    for (i = 0; i < 4; ++i) idct8(coef[i], vectors + (size_t)i * 8);
    idct8(coef[3], q);
    for (i = 0; i < 8; ++i) w[i] = 1.0f + 0.125f * (float)i;

    zenith_opts_init(&opts);
    opts.Mcoef = 8;
    opts.nbits = 16;
    opts.gen = ZENITH_GEN_DCT;
    opts.search = ZENITH_SEARCH_POSTERIOR;
    opts.quantizer = ZENITH_QUANT_LM2;
    idx = zenith_build(vectors, 4, 8, opts);
    ZTEST_CHECK(idx != NULL);
    ZTEST_CHECK(zenith_project(idx, q, Q, &residual) == ZENITH_OK);
    best = brute_lm2_best(idx, Q, w);
    ZTEST_CHECK(best >= 0);
    ZTEST_CHECK(zenith_candidates_ef(idx, q, w, 1, got) == 1);
    ZTEST_CHECK(got[0] == idx->ids[best]);
    zenith_free(idx);
    return 0;
}

static int test_sign_mag_pool_and_exact_fallback(void) {
    enum { N = 192, D = 32, K = 7, EF = 48 };
    float *vectors = (float *)malloc(sizeof(float) * (size_t)N * D);
    float q[D], w[D], exact[K];
    uint32_t pool[EF], ids[K], truth[K];
    zenith_opts opts;
    zenith_index_t *idx;
    uint32_t i, j, n;

    ZTEST_CHECK(vectors != NULL);
    ztest_seed(4409);
    ztest_make_gaussian(vectors, N, D);
    ztest_make_gaussian(q, 1, D);
    zenith_w_identity(D, w);

    zenith_opts_init(&opts);
    opts.Mcoef = D;
    opts.nbits = 64;
    opts.gen = ZENITH_GEN_SIGNED_DCT;
    opts.seed = 8181;
    opts.search = ZENITH_SEARCH_HAMMING;
    opts.quantizer = ZENITH_QUANT_SIGN_MAG;
    idx = zenith_build(vectors, N, D, opts);
    ZTEST_CHECK(idx != NULL);
    ZTEST_CHECK(idx->mag_keys != NULL);
    n = zenith_candidates_ef(idx, q, w, EF, pool);
    ZTEST_CHECK(n == EF);
    for (i = 0; i < n; ++i)
        for (j = i + 1; j < n; ++j)
            ZTEST_CHECK(pool[i] != pool[j]);

    ZTEST_CHECK(zenith_query_ef(idx, q, w, K, N, ids, exact) == K);
    ZTEST_CHECK(ztest_brute_weighted(idx, vectors, q, w, K, truth, NULL) == K);
    ZTEST_CHECK(memcmp(ids, truth, sizeof(ids)) == 0);
    zenith_free(idx);
    free(vectors);
    return 0;
}

static int test_contracts(void) {
    float vectors[64], w[32];
    zenith_opts opts;
    zenith_index_t *idx;
    uint32_t i;

    for (i = 0; i < 64; ++i) vectors[i] = (float)i * 0.01f;
    zenith_w_identity(32, w);
    zenith_opts_init(&opts);
    opts.Mcoef = 32;
    opts.nbits = 64;
    opts.quantizer = ZENITH_QUANT_LM2;
    opts.search = ZENITH_SEARCH_MIH;
    idx = zenith_build(vectors, 2, 32, opts);
    ZTEST_CHECK(idx == NULL);
    ZTEST_CHECK(zenith_last_error() == ZENITH_ERR_ARG);
    opts.search = ZENITH_SEARCH_GRAY;
    idx = zenith_build(vectors, 2, 32, opts);
    ZTEST_CHECK(idx == NULL);
    opts.search = ZENITH_SEARCH_HYBRID;
    idx = zenith_build(vectors, 2, 32, opts);
    ZTEST_CHECK(idx == NULL);
    opts.search = ZENITH_SEARCH_POSTERIOR;
    opts.use_polar = 1;
    idx = zenith_build(vectors, 2, 32, opts);
    ZTEST_CHECK(idx == NULL);

    opts.use_polar = 0;
    idx = zenith_build(vectors, 2, 32, opts);
    ZTEST_CHECK(idx != NULL);
    ZTEST_CHECK(idx->ncodes == 32);
    {
        float qc[32], dist[1];
        uint32_t id[1];
        ZTEST_CHECK(zenith_project(idx, vectors, qc, NULL) == ZENITH_OK);
        ZTEST_CHECK(zenith_query_coeffs(idx, qc, 0.0f, w, 1, 1, id, dist) == 1);
    }
    zenith_free(idx);
    return 0;
}

static int test_lm_persistence(void) {
    enum { N = 129, D = 40, M = 32, K = 6, EF = 32 };
    float *vectors = (float *)malloc(sizeof(float) * (size_t)N * D);
    float q[D], w[D];
    uint32_t ids_a[K], ids_b[K];
    float dist_a[K], dist_b[K];
    zenith_opts opts;
    zenith_index_t *idx, *loaded;
    const char *path = "test_lloyd_max.zni";
    uint32_t i;

    ZTEST_CHECK(vectors != NULL);
    ztest_seed(7717);
    ztest_make_gaussian(vectors, N, D);
    ztest_make_gaussian(q, 1, D);
    zenith_w_identity(D, w);

    zenith_opts_init(&opts);
    opts.Mcoef = M;
    opts.nbits = 64;
    opts.gen = ZENITH_GEN_SIGNED_DCT;
    opts.seed = 6161;
    opts.search = ZENITH_SEARCH_POSTERIOR;
    opts.quantizer = ZENITH_QUANT_SIGN_MAG;
    idx = zenith_build(vectors, N, D, opts);
    ZTEST_CHECK(idx != NULL);
    ZTEST_CHECK(zenith_query_ef(idx, q, w, K, EF, ids_a, dist_a) == K);
    ZTEST_CHECK(zenith_save(idx, path) == ZENITH_OK);
    loaded = zenith_load(path);
    ZTEST_CHECK(loaded != NULL);
    ZTEST_CHECK(loaded->quantizer == ZENITH_QUANT_SIGN_MAG);
    ZTEST_CHECK(loaded->ncodes == D);
    ZTEST_CHECK(loaded->mag_keys != NULL);
    ZTEST_CHECK(loaded->quant_scale != NULL && loaded->quant_bias != NULL);
    ZTEST_CHECK(zenith_query_ef(loaded, q, w, K, EF, ids_b, dist_b) == K);
    ZTEST_CHECK(memcmp(ids_a, ids_b, sizeof(ids_a)) == 0);
    for (i = 0; i < K; ++i)
        ZTEST_CHECK(ztest_float_close(dist_a[i], dist_b[i], 1e-6f, 1e-6f));
    zenith_unload(loaded);
    zenith_free(idx);
    remove(path);

    zenith_opts_init(&opts);
    opts.Mcoef = M;
    opts.nbits = 64;
    opts.gen = ZENITH_GEN_SIGNED_DCT;
    opts.seed = 6161;
    opts.search = ZENITH_SEARCH_POSTERIOR;
    opts.quantizer = ZENITH_QUANT_LM2;
    idx = zenith_build(vectors, N, D, opts);
    ZTEST_CHECK(idx != NULL);
    ZTEST_CHECK(idx->ncodes == 32);
    ZTEST_CHECK(zenith_query_ef(idx, q, w, K, EF, ids_a, dist_a) == K);
    ZTEST_CHECK(zenith_save(idx, path) == ZENITH_OK);
    loaded = zenith_load(path);
    ZTEST_CHECK(loaded != NULL);
    ZTEST_CHECK(loaded->quantizer == ZENITH_QUANT_LM2);
    ZTEST_CHECK(loaded->ncodes == 32);
    ZTEST_CHECK(loaded->mag_keys == NULL);
    ZTEST_CHECK(loaded->quant_scale != NULL && loaded->quant_bias != NULL);
    ZTEST_CHECK(zenith_query_ef(loaded, q, w, K, EF, ids_b, dist_b) == K);
    ZTEST_CHECK(memcmp(ids_a, ids_b, sizeof(ids_a)) == 0);
    for (i = 0; i < K; ++i)
        ZTEST_CHECK(ztest_float_close(dist_a[i], dist_b[i], 1e-6f, 1e-6f));
    zenith_unload(loaded);
    zenith_free(idx);
    remove(path);
    free(vectors);
    return 0;
}

int main(void) {
    if (test_symbols_and_calibration()) return 1;
    if (test_lm2_posterior_matches_brute_score()) return 1;
    if (test_sign_mag_pool_and_exact_fallback()) return 1;
    if (test_contracts()) return 1;
    if (test_lm_persistence()) return 1;
    puts("test_lloyd_max: OK");
    return 0;
}
