#define ZENITH_IMPLEMENTATION
#include "zenith.h"

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

static float frand(uint64_t *s) {
    uint64_t z = (*s += UINT64_C(0x9e3779b97f4a7c15));
    z = (z ^ (z >> 30)) * UINT64_C(0xbf58476d1ce4e5b9);
    z = (z ^ (z >> 27)) * UINT64_C(0x94d049bb133111eb);
    z ^= z >> 31;
    return (float)((z >> 11) * (1.0 / 9007199254740992.0)) * 2.0f - 1.0f;
}

int main(void) {
    enum { N = 2048, D = 48, M = 16, K = 5, EF = 64 };
    float *vectors = (float *)malloc(sizeof(float) * (size_t)N * D);
    float q[D], full_w[D];
    uint32_t approx[EF], exact[K];
    float approx_d[EF], exact_d[K];
    double mu[D];
    zenith_opts opts;
    zenith_index_t *idx;
    uint64_t rng = 31;
    uint32_t i, nc, r;

    if (!vectors) return 1;
    for (i = 0; i < (uint32_t)(N * D); ++i) vectors[i] = frand(&rng);
    for (i = 0; i < D; ++i) q[i] = frand(&rng);

    zenith_laplacian_eigs(D, mu);
    zenith_w_matern(mu, D, 0.1, 1.0, full_w);

    zenith_opts_init(&opts);
    opts.Mcoef = M;
    opts.nbits = 32;
    opts.seed = 777;
    opts.gen = ZENITH_GEN_SIGNED_DCT;
    opts.search = ZENITH_SEARCH_MIH;
    idx = zenith_build(vectors, N, D, opts);
    if (!idx) { free(vectors); return 1; }

    nc = zenith_query_ef(idx, q, full_w, EF, EF, approx, approx_d);
    if (nc == 0) { zenith_free(idx); free(vectors); return 1; }
    if (zenith_rerank_exact(idx, vectors, q, full_w, approx, nc, K,
                            exact, exact_d) != K) {
        zenith_free(idx); free(vectors); return 1;
    }

    puts("Approximate candidates, exactly re-ranked:");
    for (r = 0; r < K; ++r) {
        uint32_t c;
        float lb = -1.0f;
        for (c = 0; c < nc; ++c) {
            if (approx[c] == exact[r]) { lb = approx_d[c]; break; }
        }
        if (lb >= 0.0f)
            printf("  rank %u: id=%u exact=%.6f (candidate lower bound %.6f)\n",
                   r + 1, exact[r], exact_d[r], lb);
        else
            printf("  rank %u: id=%u exact=%.6f\n", r + 1, exact[r], exact_d[r]);
    }

    zenith_free(idx);
    free(vectors);
    return 0;
}
