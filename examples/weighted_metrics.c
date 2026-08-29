#define ZENITH_IMPLEMENTATION
#include "zenith.h"

#include <stdio.h>
#include <stdlib.h>

static float frand(uint64_t *s) {
    uint64_t z = (*s += UINT64_C(0x9e3779b97f4a7c15));
    z = (z ^ (z >> 30)) * UINT64_C(0xbf58476d1ce4e5b9);
    z = (z ^ (z >> 27)) * UINT64_C(0x94d049bb133111eb);
    z ^= z >> 31;
    return (float)((z >> 11) * (1.0 / 9007199254740992.0));
}

static void run_metric(const zenith_index_t *idx, const float *q,
                       const char *name, const float *full_w) {
    enum { K = 5 };
    uint32_t ids[K];
    float dist[K];
    uint32_t r;
    if (zenith_query(idx, q, full_w, K, ids, dist) != K) return;
    printf("%-18s", name);
    for (r = 0; r < K; ++r) printf(" %4u", ids[r]);
    printf("   | distances:");
    for (r = 0; r < K; ++r) printf(" %.3f", dist[r]);
    putchar('\n');
}

int main(void) {
    enum { N = 1024, D = 48, M = 24 };
    float *vectors = (float *)malloc(sizeof(float) * (size_t)N * D);
    float q[D], identity[D], sobolev[D], matern[D], rough[D];
    double mu[D];
    zenith_opts opts;
    zenith_index_t *idx;
    uint64_t rng = 19;
    uint32_t i;

    if (!vectors) return 1;
    for (i = 0; i < (uint32_t)(N * D); ++i) vectors[i] = 2.0f * frand(&rng) - 1.0f;
    for (i = 0; i < D; ++i) q[i] = 2.0f * frand(&rng) - 1.0f;

    zenith_opts_init(&opts);
    opts.Mcoef = M;
    opts.nbits = 32;
    opts.gen = ZENITH_GEN_SIGNED_DCT;
    opts.seed = 99;
    opts.search = ZENITH_SEARCH_HYBRID;
    idx = zenith_build(vectors, N, D, opts);
    if (!idx) { free(vectors); return 1; }

    zenith_laplacian_eigs(D, mu);
    zenith_w_identity(D, identity);
    zenith_w_sobolev(mu, D, 0.8, sobolev);
    zenith_w_matern(mu, D, 0.15, 1.3, matern);
    zenith_w_roughvol(mu, D, 0.20, rough);

    puts("Same corpus, same query, different spectral metrics:");
    run_metric(idx, q, "identity", identity);
    run_metric(idx, q, "Sobolev s=0.8", sobolev);
    run_metric(idx, q, "Matern", matern);
    run_metric(idx, q, "rough H=0.20", rough);

    zenith_free(idx);
    free(vectors);
    return 0;
}
