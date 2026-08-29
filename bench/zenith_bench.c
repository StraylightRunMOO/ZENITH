#define _POSIX_C_SOURCE 199309L
#define ZENITH_IMPLEMENTATION
#include "zenith.h"

#include <errno.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#if defined(_WIN32)
#include <windows.h>
#endif

#define ARRAY_LEN(a) (sizeof(a) / sizeof((a)[0]))

typedef struct {
    const char *name;
    uint32_t Mcoef;
    uint32_t nbits;
    zenith_search_t search;
    zenith_quantizer_t quantizer;
} bench_config;

typedef struct {
    const char *name;
    int family;
} weight_case;

typedef struct {
    char dataset[32];
    char weight[32];
    char config[64];
    char search[16];
    uint32_t N, D, Mcoef, nbits, k, ef;
    double build_ms;
    double memory_mb;
    double avg_us, p50_us, p95_us, qps, batch_qps;
    float recall;
    double exact_avg_us;
    double speedup;
    int use_fftw;
} bench_row;

typedef struct {
    uint32_t N, D, Q, K;
    uint32_t seed;
    int quick;
    const char *dataset;
    const char *out_path;
} bench_args;

static uint64_t rng_state = UINT64_C(0x243f6a8885a308d3);

static uint64_t splitmix64(void) {
    uint64_t z = (rng_state += UINT64_C(0x9e3779b97f4a7c15));
    z = (z ^ (z >> 30)) * UINT64_C(0xbf58476d1ce4e5b9);
    z = (z ^ (z >> 27)) * UINT64_C(0x94d049bb133111eb);
    return z ^ (z >> 31);
}

static double urand(void) {
    return (double)(splitmix64() >> 11) * (1.0 / 9007199254740992.0);
}

static float grand(void) {
    double u1 = urand(), u2 = urand();
    if (u1 < 1e-12) u1 = 1e-12;
    return (float)(sqrt(-2.0 * log(u1)) * cos(2.0 * ZENITH_PI * u2));
}

static uint64_t now_ns(void) {
#if defined(_WIN32)
    LARGE_INTEGER f, c;
    QueryPerformanceFrequency(&f);
    QueryPerformanceCounter(&c);
    return (uint64_t)((c.QuadPart * 1000000000ull) / (uint64_t)f.QuadPart);
#else
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ull + (uint64_t)ts.tv_nsec;
#endif
}

static int cmp_double(const void *a, const void *b) {
    double x = *(const double *)a, y = *(const double *)b;
    return (x > y) - (x < y);
}

static void make_dataset(float *v, uint32_t N, uint32_t D, const char *kind) {
    uint32_t i, j;
    if (strcmp(kind, "clustered") == 0) {
        uint32_t C = 16, c;
        float *centers = (float *)malloc(sizeof(float) * C * D);
        if (!centers) { fprintf(stderr, "allocation failed\n"); exit(2); }
        for (c = 0; c < C; ++c)
            for (j = 0; j < D; ++j)
                centers[(size_t)c * D + j] = 4.0f * grand();
        for (i = 0; i < N; ++i) {
            c = (uint32_t)(splitmix64() % C);
            for (j = 0; j < D; ++j)
                v[(size_t)i * D + j] = centers[(size_t)c * D + j] + 0.20f * grand();
        }
        free(centers);
    } else if (strcmp(kind, "rough") == 0) {
        for (i = 0; i < N; ++i) {
            double phase[8];
            double amp[8];
            uint32_t f;
            for (f = 0; f < 8; ++f) {
                phase[f] = 2.0 * ZENITH_PI * urand();
                amp[f] = pow((double)f + 1.0, -1.25);
            }
            for (j = 0; j < D; ++j) {
                double x = 0.0;
                for (f = 0; f < 8; ++f)
                    x += amp[f] * cos(2.0 * ZENITH_PI * (double)(f + 1) * j / D + phase[f]);
                v[(size_t)i * D + j] = (float)(x + 0.05 * grand());
            }
        }
    } else {
        for (i = 0; i < N; ++i)
            for (j = 0; j < D; ++j)
                v[(size_t)i * D + j] = grand();
    }
}

static void make_queries(float *queries, const float *vectors,
                         uint32_t N, uint32_t D, uint32_t Q,
                         const char *kind) {
    uint32_t i, j;
    if (strcmp(kind, "gaussian") == 0) {
        for (i = 0; i < Q; ++i)
            for (j = 0; j < D; ++j)
                queries[(size_t)i * D + j] = grand();
        return;
    }
    /* ANN recall is most informative when queries follow the corpus
     * distribution.  Use held-out-style jittered corpus points. */
    for (i = 0; i < Q; ++i) {
        const float *base = vectors + (size_t)((i * 131u + 17u) % N) * D;
        float noise = strcmp(kind, "clustered") == 0 ? 0.05f : 0.02f;
        for (j = 0; j < D; ++j)
            queries[(size_t)i * D + j] = base[j] + noise * grand();
    }
}

static void make_weight(const char *name, int D, float *w) {
    double *mu = (double *)malloc(sizeof(double) * (size_t)D);
    if (!mu) { fprintf(stderr, "allocation failed\n"); exit(2); }
    zenith_laplacian_eigs(D, mu);
    if (strcmp(name, "matern") == 0)
        zenith_w_matern(mu, D, 0.12, 1.25, w);
    else if (strcmp(name, "sobolev") == 0)
        zenith_w_sobolev(mu, D, 0.85, w);
    else if (strcmp(name, "rough") == 0)
        zenith_w_roughvol(mu, D, 0.22, w);
    else
        zenith_w_identity(D, w);
    free(mu);
}

static void exact_topk(const double *data_coeffs, const double *qc,
                       const float *w, uint32_t N, uint32_t D, uint32_t k,
                       uint32_t *ids, double *best_ds) {
    uint32_t i, r;
    for (r = 0; r < k; ++r) { ids[r] = UINT32_MAX; best_ds[r] = HUGE_VAL; }
    for (i = 0; i < N; ++i) {
        const double *x = data_coeffs + (size_t)i * D;
        double d = 0.0;
        uint32_t j;
        for (j = 0; j < D; ++j) {
            double e = qc[j] - x[j];
            d += (double)w[j] * e * e;
        }
        if (d < best_ds[k - 1]) {
            uint32_t at = k - 1;
            while (at > 0 && best_ds[at - 1] > d) {
                best_ds[at] = best_ds[at - 1];
                ids[at] = ids[at - 1];
                --at;
            }
            best_ds[at] = d;
            ids[at] = i;
        }
    }
}

static float recall_at_k(const uint32_t *truth, const uint32_t *got,
                         uint32_t got_n, uint32_t k) {
    uint32_t i, j, hit = 0;
    if (got_n > k) got_n = k;
    for (i = 0; i < k; ++i)
        for (j = 0; j < got_n; ++j)
            if (truth[i] == got[j]) { ++hit; break; }
    return (float)hit / (float)k;
}

static double estimate_memory_mb(const zenith_index_t *idx) {
    size_t bytes = 0;
    bytes += (size_t)idx->N * sizeof(uint64_t);
    bytes += (size_t)idx->N * sizeof(uint32_t);
    bytes += (size_t)idx->N * idx->Mcoef * sizeof(float);
    bytes += (size_t)idx->N * sizeof(float);
    bytes += (size_t)idx->Mcoef * sizeof(float);
    bytes += (size_t)idx->plan_M * idx->plan_D * sizeof(float);
    if (idx->input_signs) bytes += idx->D;
    if (idx->mih_off && idx->mih_idx) {
        size_t nb = (size_t)1u << idx->mih_bits;
        bytes += (size_t)idx->mih_m * (nb + 1u) * sizeof(uint32_t);
        bytes += (size_t)idx->mih_m * idx->N * sizeof(uint32_t);
    }
    return (double)bytes / (1024.0 * 1024.0);
}

static const char *search_name(int s) {
    switch (s) {
    case ZENITH_SEARCH_MIH: return "mih";
    case ZENITH_SEARCH_GRAY: return "gray";
    case ZENITH_SEARCH_HYBRID: return "hybrid";
    case ZENITH_SEARCH_HAMMING: return "hamming";
    case ZENITH_SEARCH_POSTERIOR: return "posterior";
    default: return "unknown";
    }
}

static void usage(const char *prog) {
    fprintf(stderr,
        "Usage: %s [options]\n"
        "  --quick              Quick CI-sized benchmark (default: N=4096 D=64)\n"
        "  --full               Larger corpus (N=16384 D=96)\n"
        "  --perf               High-power suite (N=8192 D=256), FFTW-friendly\n"
        "  --n N                Corpus vectors\n"
        "  --dim D              Input dimension\n"
        "  --queries Q          Query count\n"
        "  --k K                Neighbors returned\n"
        "  --dataset NAME       gaussian | clustered | rough\n"
        "  --seed S             Deterministic seed\n"
        "  --out PATH           CSV output path\n",
        prog);
}

static bench_args parse_args(int argc, char **argv) {
    bench_args a;
    int i;
    memset(&a, 0, sizeof(a));
    a.N = 4096; a.D = 64; a.Q = 32; a.K = 10;
    a.seed = 12345; a.quick = 1;
    a.dataset = "clustered";
    a.out_path = "zenith_benchmarks.csv";
    for (i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--quick") == 0) {
            a.quick = 1;
        } else if (strcmp(argv[i], "--full") == 0) {
            a.quick = 0;
            a.N = 16384; a.D = 96; a.Q = 64; a.K = 10;
        } else if (strcmp(argv[i], "--perf") == 0) {
            a.quick = 0;
            a.N = 8192; a.D = 256; a.Q = 48; a.K = 10;
        } else if (strcmp(argv[i], "--n") == 0 && i + 1 < argc) {
            a.N = (uint32_t)strtoul(argv[++i], NULL, 10);
        } else if (strcmp(argv[i], "--dim") == 0 && i + 1 < argc) {
            a.D = (uint32_t)strtoul(argv[++i], NULL, 10);
        } else if (strcmp(argv[i], "--queries") == 0 && i + 1 < argc) {
            a.Q = (uint32_t)strtoul(argv[++i], NULL, 10);
        } else if (strcmp(argv[i], "--k") == 0 && i + 1 < argc) {
            a.K = (uint32_t)strtoul(argv[++i], NULL, 10);
        } else if (strcmp(argv[i], "--dataset") == 0 && i + 1 < argc) {
            a.dataset = argv[++i];
        } else if (strcmp(argv[i], "--seed") == 0 && i + 1 < argc) {
            a.seed = (uint32_t)strtoul(argv[++i], NULL, 10);
        } else if (strcmp(argv[i], "--out") == 0 && i + 1 < argc) {
            a.out_path = argv[++i];
        } else if (strcmp(argv[i], "--help") == 0) {
            usage(argv[0]); exit(0);
        } else {
            usage(argv[0]); exit(2);
        }
    }
    if (!a.N || !a.D || !a.Q || !a.K || a.K > a.N || a.D > 256) {
        fprintf(stderr, "invalid benchmark dimensions\n");
        exit(2);
    }
    return a;
}

int main(int argc, char **argv) {
    static const bench_config configs[] = {
        {"mih_m16_b32", 16, 32, ZENITH_SEARCH_MIH, ZENITH_QUANT_SIGN},
        {"mih_m32_b32", 32, 32, ZENITH_SEARCH_MIH, ZENITH_QUANT_SIGN},
        {"mih_m32_b64", 32, 64, ZENITH_SEARCH_MIH, ZENITH_QUANT_SIGN},
        {"hybrid_m32_b64", 32, 64, ZENITH_SEARCH_HYBRID, ZENITH_QUANT_SIGN},
        {"gray_m32_b32", 32, 32, ZENITH_SEARCH_GRAY, ZENITH_QUANT_SIGN},
        {"gray_full", 0, 0, ZENITH_SEARCH_GRAY, ZENITH_QUANT_SIGN},
        {"hamming_m32_b64", 32, 64, ZENITH_SEARCH_HAMMING, ZENITH_QUANT_SIGN},
        {"posterior_m32_b64", 32, 64, ZENITH_SEARCH_POSTERIOR, ZENITH_QUANT_SIGN},
        {"hamming_full", 0, 0, ZENITH_SEARCH_HAMMING, ZENITH_QUANT_SIGN},
        {"posterior_full", 0, 0, ZENITH_SEARCH_POSTERIOR, ZENITH_QUANT_SIGN},
        {"lm2_ham_full", 0, 64, ZENITH_SEARCH_HAMMING, ZENITH_QUANT_LM2},
        {"lm2_post_full", 0, 64, ZENITH_SEARCH_POSTERIOR, ZENITH_QUANT_LM2},
        {"signmag_ham_full", 0, 0, ZENITH_SEARCH_HAMMING, ZENITH_QUANT_SIGN_MAG},
        {"signmag_post_full", 0, 0, ZENITH_SEARCH_POSTERIOR, ZENITH_QUANT_SIGN_MAG}
    };
    static const char *weight_names[] = {"identity", "matern", "sobolev"};
    bench_args args = parse_args(argc, argv);
    uint32_t efs[8], n_efs = 0;
    float *vectors, *queries;
    double *data_coeffs;
    uint32_t **truth;
    double **truth_ds;
    float (*weights)[256];
    zenith_opts ref_opts;
    zenith_index_t *ref_idx;
    bench_row *rows;
    size_t max_rows, row_count = 0;
    uint32_t wi, qi, ci, ei, i;
    FILE *out;
    double exact_avg_us[ARRAY_LEN(weight_names)];

    fprintf(stderr, "ZENITH %s  SIMD=%d (%s)  FFTW=%s  OpenMP=%s  threads=%d\n",
            ZENITH_VERSION_STRING, zenith_simd_level(),
            zenith_simd_level() == 3 ? "AVX-512" :
            zenith_simd_level() == 2 ? "AVX2" :
            zenith_simd_level() == 1 ? "NEON" : "scalar",
            zenith_fftw_enabled() ? "on" : "off",
            zenith_openmp_enabled() ? "on" : "off",
            zenith_thread_count());
    rng_state = args.seed ? args.seed : 1;
    vectors = (float *)malloc(sizeof(float) * (size_t)args.N * args.D);
    queries = (float *)malloc(sizeof(float) * (size_t)args.Q * args.D);
    if (!vectors || !queries) { fprintf(stderr, "allocation failed\n"); return 2; }
    make_dataset(vectors, args.N, args.D, args.dataset);
    make_queries(queries, vectors, args.N, args.D, args.Q, args.dataset);

    weights = (float (*)[256])malloc(sizeof(float) * 256 * ARRAY_LEN(weight_names));
    if (!weights) { fprintf(stderr, "allocation failed\n"); return 2; }
    for (wi = 0; wi < ARRAY_LEN(weight_names); ++wi)
        make_weight(weight_names[wi], (int)args.D, weights[wi]);

    zenith_opts_init(&ref_opts);
    ref_opts.Mcoef = args.D;
    ref_opts.nbits = args.D < 64 ? args.D : 64;
    ref_opts.seed = 424242;
    ref_opts.gen = ZENITH_GEN_SIGNED_DCT;
    ref_opts.search = ZENITH_SEARCH_GRAY;
    ref_idx = zenith_build(vectors, args.N, args.D, ref_opts);
    if (!ref_idx) { fprintf(stderr, "reference build failed\n"); return 2; }

    data_coeffs = (double *)malloc(sizeof(double) * (size_t)args.N * args.D);
    truth = (uint32_t **)malloc(sizeof(uint32_t *) * ARRAY_LEN(weight_names));
    truth_ds = (double **)malloc(sizeof(double *) * ARRAY_LEN(weight_names));
    if (!data_coeffs || !truth || !truth_ds) {
        fprintf(stderr, "allocation failed\n"); return 2;
    }
    for (wi = 0; wi < ARRAY_LEN(weight_names); ++wi) {
        truth[wi] = (uint32_t *)malloc(sizeof(uint32_t) * (size_t)args.Q * args.K);
        truth_ds[wi] = (double *)malloc(sizeof(double) * (size_t)args.Q * args.K);
        if (!truth[wi] || !truth_ds[wi]) { fprintf(stderr, "allocation failed\n"); return 2; }
    }

    {
        uint64_t t0 = now_ns();
        for (i = 0; i < args.N; ++i)
            zenith__full_dct_signed(ref_idx, vectors + (size_t)i * args.D,
                                    data_coeffs + (size_t)i * args.D);
        fprintf(stderr, "Ground-truth spectral projection: %.1f ms\n",
                (double)(now_ns() - t0) / 1e6);
    }

    for (wi = 0; wi < ARRAY_LEN(weight_names); ++wi) {
        double *lat = (double *)malloc(sizeof(double) * args.Q);
        double *qtmp = (double *)malloc(sizeof(double) * args.D);
        double sum = 0.0;
        if (!lat || !qtmp) return 2;
        for (qi = 0; qi < args.Q; ++qi) {
            uint64_t t0 = now_ns();
            /* End-to-end exact baseline: project the raw query, then scan all
             * precomputed corpus coefficients. */
            zenith__full_dct_signed(ref_idx, queries + (size_t)qi * args.D, qtmp);
            exact_topk(data_coeffs, qtmp,
                       weights[wi], args.N, args.D, args.K,
                       truth[wi] + (size_t)qi * args.K,
                       truth_ds[wi] + (size_t)qi * args.K);
            lat[qi] = (double)(now_ns() - t0) / 1000.0;
            sum += lat[qi];
        }
        qsort(lat, args.Q, sizeof(double), cmp_double);
        exact_avg_us[wi] = sum / args.Q;
        fprintf(stderr, "Exact %-8s: %.1f us/query, %.0f QPS\n",
                weight_names[wi], exact_avg_us[wi], 1e6 / exact_avg_us[wi]);
        free(lat); free(qtmp);
    }

    efs[n_efs++] = args.K;
    efs[n_efs++] = args.K * 2 <= args.N ? args.K * 2 : args.N;
    efs[n_efs++] = args.K * 4 <= args.N ? args.K * 4 : args.N;
    efs[n_efs++] = args.K * 8 <= args.N ? args.K * 8 : args.N;
    efs[n_efs++] = args.K * 16 <= args.N ? args.K * 16 : args.N;
    if (efs[n_efs - 1] != args.N) efs[n_efs++] = args.N;

    max_rows = ARRAY_LEN(configs) * ARRAY_LEN(weight_names) * n_efs;
    rows = (bench_row *)calloc(max_rows, sizeof(bench_row));
    if (!rows) return 2;

    for (ci = 0; ci < ARRAY_LEN(configs); ++ci) {
        bench_config cfg = configs[ci];
        zenith_opts opts;
        zenith_index_t *idx;
        uint64_t build_t0, build_t1;
        uint32_t M = cfg.Mcoef ? cfg.Mcoef : args.D;
        uint32_t nb = cfg.nbits ? cfg.nbits : (args.D < 64 ? args.D : 64);
        if (M > args.D) M = args.D;
        if (cfg.quantizer != ZENITH_QUANT_LM2 && nb > args.D) nb = args.D;
        if (nb > 64) nb = 64;

        zenith_opts_init(&opts);
        opts.Mcoef = M;
        opts.nbits = nb;
        opts.seed = 424242;
        opts.gen = ZENITH_GEN_SIGNED_DCT;
        opts.search = cfg.search;
        opts.quantizer = cfg.quantizer;
        opts.auto_perm = 0;
        opts.fftw = 0; /* auto: dense for small D, FFTW when it pays */

        build_t0 = now_ns();
        idx = zenith_build(vectors, args.N, args.D, opts);
        build_t1 = now_ns();
        if (!idx) {
            fprintf(stderr, "build failed for %s\n", cfg.name);
            return 2;
        }

        for (wi = 0; wi < ARRAY_LEN(weight_names); ++wi) {
            float *full_w = weights[wi];

            for (ei = 0; ei < n_efs; ++ei) {
                uint32_t ef = efs[ei];
                double *lat = (double *)malloc(sizeof(double) * args.Q);
                uint32_t *ids = (uint32_t *)malloc(sizeof(uint32_t) * args.K);
                float *dist = (float *)malloc(sizeof(float) * args.K);
                double sum = 0.0, recall_sum = 0.0;
                bench_row *row;
                if (!lat || !ids || !dist) return 2;

                zenith_query_ef(idx, queries, full_w, args.K, ef, ids, dist);
                for (qi = 0; qi < args.Q; ++qi) {
                    uint64_t t0 = now_ns();
                    uint32_t got = zenith_query_ef(idx, queries + (size_t)qi * args.D,
                                                   full_w, args.K, ef, ids, dist);
                    uint64_t t1 = now_ns();
                    lat[qi] = (double)(t1 - t0) / 1000.0;
                    sum += lat[qi];
                    recall_sum += recall_at_k(truth[wi] + (size_t)qi * args.K,
                                              ids, got, args.K);
                }
                qsort(lat, args.Q, sizeof(double), cmp_double);

                {
                    uint32_t *batch_ids = (uint32_t *)malloc(sizeof(uint32_t) * (size_t)args.Q * args.K);
                    float *batch_dist = (float *)malloc(sizeof(float) * (size_t)args.Q * args.K);
                    uint64_t bt0, bt1;
                    if (!batch_ids || !batch_dist) return 2;
                    zenith_query_many(idx, queries, args.Q, full_w, args.K, ef,
                                      batch_ids, batch_dist);
                    bt0 = now_ns();
                    zenith_query_many(idx, queries, args.Q, full_w, args.K, ef,
                                      batch_ids, batch_dist);
                    bt1 = now_ns();
                    row = &rows[row_count++];
                    memset(row, 0, sizeof(*row));
                    snprintf(row->dataset, sizeof(row->dataset), "%s", args.dataset);
                    snprintf(row->weight, sizeof(row->weight), "%s", weight_names[wi]);
                    snprintf(row->config, sizeof(row->config), "%s", cfg.name);
                    snprintf(row->search, sizeof(row->search), "%s", search_name(idx->search));
                    row->N = args.N; row->D = args.D; row->Mcoef = M; row->nbits = nb;
                    row->k = args.K; row->ef = ef;
                    row->build_ms = (double)(build_t1 - build_t0) / 1e6;
                    row->memory_mb = estimate_memory_mb(idx);
                    row->avg_us = sum / args.Q;
                    row->p50_us = lat[args.Q / 2];
                    row->p95_us = lat[(size_t)((args.Q - 1) * 0.95)];
                    row->qps = 1e6 / row->avg_us;
                    row->batch_qps = (double)args.Q * 1e9 / (double)(bt1 - bt0);
                    row->recall = (float)(recall_sum / args.Q);
                    row->exact_avg_us = exact_avg_us[wi];
                    row->speedup = exact_avg_us[wi] / row->avg_us;
                    row->use_fftw = idx->use_fftw;
                    free(batch_ids); free(batch_dist);
                }

                free(lat); free(ids); free(dist);
            }
        }
        fprintf(stderr, "built/measured %-16s M=%-3u bits=%-2u build=%.1f ms mem=%.2f MB fftw=%d\n",
                cfg.name, M, nb, (double)(build_t1 - build_t0) / 1e6,
                estimate_memory_mb(idx), idx->use_fftw);
        zenith_free(idx);
    }

    out = fopen(args.out_path, "w");
    if (!out) {
        fprintf(stderr, "cannot open %s: %s\n", args.out_path, strerror(errno));
        return 2;
    }
    fprintf(out, "dataset,weight,config,search,N,D,Mcoef,nbits,k,ef,build_ms,memory_mb,avg_us,p50_us,p95_us,qps,batch_qps,recall_at_k,exact_avg_us,speedup_vs_exact,fftw\n");
    for (i = 0; i < row_count; ++i) {
        bench_row *r = &rows[i];
        fprintf(out, "%s,%s,%s,%s,%u,%u,%u,%u,%u,%u,%.3f,%.3f,%.3f,%.3f,%.3f,%.1f,%.1f,%.6f,%.3f,%.2f,%d\n",
                r->dataset, r->weight, r->config, r->search, r->N, r->D,
                r->Mcoef, r->nbits, r->k, r->ef, r->build_ms, r->memory_mb,
                r->avg_us, r->p50_us, r->p95_us, r->qps, r->batch_qps, r->recall,
                r->exact_avg_us, r->speedup, r->use_fftw);
    }
    fclose(out);

    printf("\nBenchmark summary (%s, N=%u, D=%u, Q=%u, k=%u, SIMD=%d, FFTW=%s, OpenMP=%s, threads=%d)\n",
           args.dataset, args.N, args.D, args.Q, args.K, zenith_simd_level(),
           zenith_fftw_enabled() ? "on" : "off",
           zenith_openmp_enabled() ? "on" : "off",
           zenith_thread_count());
    printf("%-9s %-16s %-5s %8s %8s %9s %10s %8s %9s\n",
           "weight", "config", "ef", "avg us", "p95 us", "QPS", "batch QPS", "recall", "speedup");
    for (i = 0; i < row_count; ++i) {
        bench_row *r = &rows[i];
        if (r->ef == args.K || r->ef == args.K * 4 || r->ef == args.N)
            printf("%-9s %-16s %-5u %8.1f %8.1f %9.0f %10.0f %8.3f %8.1fx\n",
                   r->weight, r->config, r->ef, r->avg_us, r->p95_us,
                   r->qps, r->batch_qps, r->recall, r->speedup);
    }

    printf("\nPareto frontier by weight (higher recall, lower p95 latency):\n");
    for (wi = 0; wi < ARRAY_LEN(weight_names); ++wi) {
        printf("  %s:\n", weight_names[wi]);
        for (i = 0; i < row_count; ++i) {
            bench_row *a = &rows[i];
            int dominated = 0;
            size_t bidx;
            if (strcmp(a->weight, weight_names[wi]) != 0) continue;
            for (bidx = 0; bidx < row_count; ++bidx) {
                bench_row *b = &rows[bidx];
                if (b == a || strcmp(b->weight, weight_names[wi]) != 0) continue;
                if (b->recall >= a->recall && b->p95_us <= a->p95_us &&
                    (b->recall > a->recall || b->p95_us < a->p95_us)) {
                    dominated = 1; break;
                }
            }
            if (!dominated)
                printf("    %-16s ef=%-5u recall=%.3f p95=%.1f us QPS=%.0f\n",
                       a->config, a->ef, a->recall, a->p95_us, a->qps);
        }
    }
    printf("\nCSV written to %s\n", args.out_path);

    for (wi = 0; wi < ARRAY_LEN(weight_names); ++wi) {
        free(truth[wi]); free(truth_ds[wi]);
    }
    free(truth); free(truth_ds);
    free(data_coeffs);
    zenith_free(ref_idx);
    free(weights); free(vectors); free(queries); free(rows);
    return 0;
}
