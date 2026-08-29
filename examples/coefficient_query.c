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

int main(void) {
    enum { N = 512, D = 32, M = 32, K = 5 };
    float *vectors = (float *)malloc(sizeof(float) * (size_t)N * D);
    float q[D], q_coeff[M], w[D];
    float residual = 0.0f;
    uint32_t ids[K];
    float dist[K];
    zenith_opts opts;
    zenith_index_t *idx;
    uint64_t rng = 41;
    uint32_t i, r;

    if (!vectors) return 1;
    for (i = 0; i < (uint32_t)(N * D); ++i) vectors[i] = frand(&rng);
    for (i = 0; i < D; ++i) q[i] = frand(&rng);
    zenith_w_identity(D, w);

    zenith_opts_init(&opts);
    opts.Mcoef = M;
    opts.nbits = M; /* coefficient queries require nbits <= Mcoef */
    opts.seed = 12;
    opts.gen = ZENITH_GEN_SIGNED_DCT;
    opts.search = ZENITH_SEARCH_MIH;
    idx = zenith_build(vectors, N, D, opts);
    if (!idx) { free(vectors); return 1; }

    if (zenith_project(idx, q, q_coeff, &residual) != ZENITH_OK ||
        zenith_query_coeffs(idx, q_coeff, residual, w, K, 64, ids, dist) != K) {
        fprintf(stderr, "coefficient query failed: %d\n", zenith_last_error());
        zenith_free(idx); free(vectors); return 1;
    }

    puts("Coefficient-space query result:");
    for (r = 0; r < K; ++r)
        printf("  id=%u distance=%.6f\n", ids[r], dist[r]);

    zenith_free(idx);
    free(vectors);
    return 0;
}
