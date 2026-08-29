#define ZENITH_IMPLEMENTATION
#include "zenith.h"
#include "test_common.h"

static uint32_t brute_l2(const float *vectors, uint32_t N, uint32_t D,
                         const float *q, uint32_t k,
                         uint32_t *out_ids, float *out_dist) {
    double *ds = (double *)malloc(sizeof(double) * N);
    uint32_t *ids = (uint32_t *)malloc(sizeof(uint32_t) * N);
    uint32_t i, j, n;
    if (!ds || !ids) {
        free(ds); free(ids);
        return 0;
    }
    for (i = 0; i < N; ++i) {
        double d = 0.0;
        for (j = 0; j < D; ++j) {
            double e = (double)q[j] - vectors[(size_t)i * D + j];
            d += e * e;
        }
        ds[i] = d;
        ids[i] = i;
    }
    if (k > N) k = N;
    for (i = 0; i < k; ++i) {
        uint32_t best = i;
        for (n = i + 1; n < N; ++n)
            if (ds[n] < ds[best] || (ds[n] == ds[best] && ids[n] < ids[best]))
                best = n;
        {
            double td = ds[i]; ds[i] = ds[best]; ds[best] = td;
            n = ids[i]; ids[i] = ids[best]; ids[best] = n;
            out_ids[i] = ids[i];
            if (out_dist) out_dist[i] = (float)sqrt(ds[i]);
        }
    }
    free(ds);
    free(ids);
    return k;
}

static float recall_at_k(const uint32_t *truth, const uint32_t *got, uint32_t k) {
    uint32_t i, j, hit = 0;
    for (i = 0; i < k; ++i)
        for (j = 0; j < k; ++j)
            if (truth[i] == got[j]) { ++hit; break; }
    return (float)hit / (float)k;
}

static int test_exhaustive_recall(void) {
    enum { N = 256, D = 24, K = 5, Q = 16 };
    float vectors[N * D], queries[Q * D], weight[D];
    zenith_search_t modes[3] = {ZENITH_SEARCH_MIH, ZENITH_SEARCH_GRAY, ZENITH_SEARCH_HYBRID};
    uint32_t mode;

    ztest_seed(909);
    ztest_make_clustered(vectors, N, D, 8);
    ztest_make_gaussian(queries, Q, D);
    zenith_w_identity(D, weight);

    for (mode = 0; mode < 3; ++mode) {
        zenith_opts opts;
        zenith_index_t *idx;
        uint32_t qi, r;
        memset(&opts, 0, sizeof(opts));
        opts.Mcoef = D;
        opts.nbits = D;
        opts.seed = 123;
        opts.gen = ZENITH_GEN_SIGNED_DCT;
        opts.search = modes[mode];
        idx = zenith_build(vectors, N, D, opts);
        ZTEST_CHECK(idx != NULL);
        for (qi = 0; qi < Q; ++qi) {
            uint32_t got[K], truth[K];
            float gd[K], td[K];
            ZTEST_CHECK(zenith_query_ef(idx, queries + (size_t)qi * D, weight,
                                        K, N, got, gd) == K);
            ZTEST_CHECK(brute_l2(vectors, N, D, queries + (size_t)qi * D,
                                 K, truth, td) == K);
            ZTEST_CHECK(recall_at_k(truth, got, K) == 1.0f);
            for (r = 1; r < K; ++r) ZTEST_CHECK(gd[r] >= gd[r - 1]);
        }
        zenith_free(idx);
    }
    return 0;
}

static int test_truncated_subspace_exhaustive_ranking(void) {
    enum { N = 192, D = 32, M = 16, K = 6, Q = 8 };
    float vectors[N * D], queries[Q * D], weight[D];
    zenith_opts opts;
    zenith_index_t *idx;
    uint32_t qi, r;

    ztest_seed(1010);
    ztest_make_clustered(vectors, N, D, 6);
    ztest_make_gaussian(queries, Q, D);
    zenith_w_identity(D, weight);
    zenith_opts_init(&opts);
    opts.Mcoef = M;
    opts.nbits = M;
    opts.seed = 17;
    opts.gen = ZENITH_GEN_SIGNED_DCT;
    opts.search = ZENITH_SEARCH_MIH;
    idx = zenith_build(vectors, N, D, opts);
    ZTEST_CHECK(idx != NULL);

    for (qi = 0; qi < Q; ++qi) {
        uint32_t got[K], truth[K];
        float gd[K], td[K];
        ZTEST_CHECK(zenith_query_ef(idx, queries + (size_t)qi * D, weight,
                                    K, N, got, gd) == K);
        ZTEST_CHECK(ztest_brute_weighted(idx, vectors, queries + (size_t)qi * D,
                                         weight, K, truth, td) == K);
        ZTEST_CHECK(recall_at_k(truth, got, K) == 1.0f);
        for (r = 1; r < K; ++r) ZTEST_CHECK(gd[r] >= gd[r - 1]);
    }
    zenith_free(idx);
    return 0;
}

static int test_approximate_recall_sanity(void) {
    enum { N = 512, D = 32, M = 16, K = 5, Q = 16 };
    float vectors[N * D], queries[Q * D], weight[D];
    zenith_opts opts;
    zenith_index_t *idx;
    float total_recall = 0.0f;
    uint32_t qi;

    ztest_seed(1111);
    ztest_make_clustered(vectors, N, D, 8);
    for (uint32_t qq = 0; qq < Q; ++qq) {
        const float *base = vectors + (size_t)((qq * 37u) % N) * D;
        for (uint32_t dd = 0; dd < D; ++dd)
            queries[(size_t)qq * D + dd] = base[dd] + 0.05f * ztest_gauss();
    }
    zenith_w_identity(D, weight);
    zenith_opts_init(&opts);
    opts.Mcoef = M;
    opts.nbits = 32;
    opts.seed = 88;
    opts.gen = ZENITH_GEN_SIGNED_DCT;
    opts.search = ZENITH_SEARCH_HYBRID;
    idx = zenith_build(vectors, N, D, opts);
    ZTEST_CHECK(idx != NULL);

    for (qi = 0; qi < Q; ++qi) {
        uint32_t got[K], truth[K];
        ZTEST_CHECK(zenith_query_ef(idx, queries + (size_t)qi * D, weight,
                                    K, 64, got, NULL) == K);
        ZTEST_CHECK(brute_l2(vectors, N, D, queries + (size_t)qi * D,
                             K, truth, NULL) == K);
        total_recall += recall_at_k(truth, got, K);
    }
    ZTEST_CHECK(total_recall / Q >= 0.20f);
    zenith_free(idx);
    return 0;
}

int main(void) {
    if (test_exhaustive_recall()) return 1;
    if (test_truncated_subspace_exhaustive_ranking()) return 1;
    if (test_approximate_recall_sanity()) return 1;
    puts("test_recall: OK");
    return 0;
}
