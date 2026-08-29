#define ZENITH_IMPLEMENTATION
#include "zenith.h"

#include <stdio.h>
#include <stdlib.h>

int main(int argc, char **argv) {
    enum { N = 256, D = 32, M = 16, K = 5 };
    const char *path = argc > 1 ? argv[1] : "example.zni";
    float *vectors = (float *)malloc(sizeof(float) * (size_t)N * D);
    float q[D], w[D];
    uint32_t ids[K];
    float dist[K];
    zenith_opts opts;
    zenith_index_t *idx, *loaded;
    uint32_t i, r;

    if (!vectors) return 1;
    for (i = 0; i < (uint32_t)(N * D); ++i)
        vectors[i] = (float)((int)(i * 2654435761u % 1000u) - 500) / 500.0f;
    for (i = 0; i < D; ++i) q[i] = (float)i / D - 0.5f;
    zenith_w_identity(D, w);

    zenith_opts_init(&opts);
    opts.Mcoef = M;
    opts.nbits = 32;
    opts.seed = 5;
    opts.gen = ZENITH_GEN_SIGNED_DCT;
    opts.search = ZENITH_SEARCH_HYBRID;
    idx = zenith_build(vectors, N, D, opts);
    if (!idx) { free(vectors); return 1; }

    if (zenith_save(idx, path) != ZENITH_OK) {
        fprintf(stderr, "save failed: %d\n", zenith_last_error());
        zenith_free(idx); free(vectors); return 1;
    }
    loaded = zenith_load(path);
    if (!loaded) {
        fprintf(stderr, "load failed: %d\n", zenith_last_error());
        zenith_free(idx); free(vectors); return 1;
    }

    if (zenith_query_ef(loaded, q, w, K, 64, ids, dist) == K) {
        printf("Loaded %s. Top %u:\n", path, K);
        for (r = 0; r < K; ++r) printf("  id=%u bound=%.6f\n", ids[r], dist[r]);
    }

    zenith_unload(loaded);
    zenith_free(idx);
    free(vectors);
    return 0;
}
