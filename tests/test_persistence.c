#define ZENITH_IMPLEMENTATION
#include "zenith.h"
#include "test_common.h"

static const char *test_path = "zenith_persistence_test.zni";
static const char *bad_path = "zenith_persistence_bad.zni";

static int write_corrupt_copy(void) {
    FILE *in = fopen(test_path, "rb");
    FILE *out;
    long size;
    uint8_t *buf;
    if (!in) return 1;
    if (fseek(in, 0, SEEK_END) != 0 || (size = ftell(in)) <= 0 ||
        fseek(in, 0, SEEK_SET) != 0) {
        fclose(in); return 1;
    }
    buf = (uint8_t *)malloc((size_t)size);
    if (!buf) { fclose(in); return 1; }
    if (fread(buf, 1, (size_t)size, in) != (size_t)size) {
        free(buf); fclose(in); return 1;
    }
    fclose(in);
    buf[(size_t)size - 1u] ^= 0x5Au;
    out = fopen(bad_path, "wb");
    if (!out) { free(buf); return 1; }
    if (fwrite(buf, 1, (size_t)size, out) != (size_t)size) {
        free(buf); fclose(out); return 1;
    }
    if (fclose(out) != 0) {
        free(buf); return 1;
    }
    free(buf);
    return 0;
}

int main(void) {
    enum { N = 256, D = 40, M = 24, K = 8 };
    float *vectors = (float *)malloc(sizeof(float) * (size_t)N * D);
    float q[D], weight[D];
    uint32_t ids_a[K], ids_b[K], ids_c[K];
    float dist_a[K], dist_b[K], dist_c[K];
    zenith_opts opts;
    zenith_index_t *idx, *loaded, *loaded_copy;
    uint32_t i;

    remove(test_path);
    remove(bad_path);
    ZTEST_CHECK(vectors != NULL);
    ztest_seed(808);
    ztest_make_clustered(vectors, N, D, 8);
    ztest_make_gaussian(q, 1, D);
    zenith_w_identity(D, weight);

    zenith_opts_init(&opts);
    opts.Mcoef = M;
    opts.nbits = 32;
    opts.seed = UINT64_C(0x1234567900000277);
    opts.gen = ZENITH_GEN_SIGNED_DCT;
    opts.search = ZENITH_SEARCH_HYBRID;
    opts.auto_perm = 1;
    idx = zenith_build(vectors, N, D, opts);
    ZTEST_CHECK(idx != NULL);
    ZTEST_CHECK(idx->gen_seed == opts.seed);

    ZTEST_CHECK(zenith_query_ef(idx, q, weight, K, 64, ids_a, dist_a) == K);
    ZTEST_CHECK(zenith_save(idx, test_path) == ZENITH_OK);

    loaded = zenith_load(test_path);
    ZTEST_CHECK(loaded != NULL);
#if defined(__unix__) || defined(__APPLE__)
    ZTEST_CHECK(loaded->mmapped == 1);
#endif
    ZTEST_CHECK(loaded->N == N && loaded->D == D && loaded->Mcoef == M);
    ZTEST_CHECK(loaded->search == ZENITH_SEARCH_HYBRID);
    ZTEST_CHECK(loaded->gen_seed == opts.seed);
    ZTEST_CHECK(zenith_query_ef(loaded, q, weight, K, 64, ids_b, dist_b) == K);
    ZTEST_CHECK(memcmp(ids_a, ids_b, sizeof(ids_a)) == 0);
    for (i = 0; i < K; ++i)
        ZTEST_CHECK(ztest_float_close(dist_a[i], dist_b[i], 1e-6f, 1e-6f));

    loaded_copy = zenith_load_copy(test_path);
    ZTEST_CHECK(loaded_copy != NULL);
    ZTEST_CHECK(loaded_copy->mmapped == 0);
    ZTEST_CHECK(loaded_copy->gen_seed == opts.seed);
    ZTEST_CHECK(zenith_query_ef(loaded_copy, q, weight, K, 64, ids_c, dist_c) == K);
    ZTEST_CHECK(memcmp(ids_a, ids_c, sizeof(ids_a)) == 0);
    for (i = 0; i < K; ++i)
        ZTEST_CHECK(ztest_float_close(dist_a[i], dist_c[i], 1e-6f, 1e-6f));

    ZTEST_CHECK(write_corrupt_copy() == 0);
    ZTEST_CHECK(zenith_load(bad_path) == NULL);
    ZTEST_CHECK(zenith_last_error() == ZENITH_ERR_FORMAT);

    zenith_unload(loaded);
    zenith_unload(loaded_copy);
    zenith_free(idx);
    free(vectors);
    remove(test_path);
    remove(bad_path);
    puts("test_persistence: OK");
    return 0;
}
