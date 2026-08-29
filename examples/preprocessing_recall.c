#define _POSIX_C_SOURCE 199309L
#define ZENITH_IMPLEMENTATION
#include "zenith.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static uint64_t zenith__test_now_ns(void);

#define N 4096u
#define D 64u
#define Q 64u
#define K 10u
#define POOL 160u
#define MCOEF 32u

typedef enum { BASE_GAUSSIAN, BASE_LOGNORMAL, BASE_OUTLIER } base_kind;
typedef enum { TR_NONE, TR_L2, TR_FRACTIONAL, TR_LOG, TR_RANK } transform_kind;

typedef struct {
    const char *base;
    const char *prep;
    base_kind base_kind;
    transform_kind transform;
} case_def;

static uint64_t rng_state;

static uint64_t splitmix64(void) {
    uint64_t z = (rng_state += UINT64_C(0x9e3779b97f4a7c15));
    z = (z ^ (z >> 30)) * UINT64_C(0xbf58476d1ce4e5b9);
    z = (z ^ (z >> 27)) * UINT64_C(0x94d049bb133111eb);
    return z ^ (z >> 31);
}

static double urand(void) {
    return (double)(splitmix64() >> 11) * (1.0 / 9007199254740992.0);
}

static double grand(void) {
    double u = urand();
    if (u < 1e-12) u = 1e-12;
    return sqrt(-2.0 * log(u)) * cos(2.0 * ZENITH_PI * urand());
}

static int cmp_float(const void *pa, const void *pb) {
    float a = *(const float *)pa, b = *(const float *)pb;
    return (a > b) - (a < b);
}

static int cmp_double(const void *pa, const void *pb) {
    double a = *(const double *)pa, b = *(const double *)pb;
    return (a > b) - (a < b);
}

static double normal_quantile(double p) {
    static const double a[] = {
        -3.969683028665376e+01,  2.209460984245205e+02,
        -2.759285104469687e+02,  1.383577518672690e+02,
        -3.066479806614716e+01,  2.506628277459239e+00
    };
    static const double b[] = {
        -5.447609879822406e+01,  1.615858368580409e+02,
        -1.556989798598866e+02,  6.680131188771972e+01,
        -1.328068155288572e+01
    };
    static const double c[] = {
        -7.784894002430293e-03, -3.223964580411365e-01,
        -2.400758277161838e+00, -2.549732539343734e+00,
         4.374664141464968e+00,  2.938163982698783e+00
    };
    static const double d[] = {
         7.784695709041462e-03,  3.224671290700398e-01,
         2.445134137142996e+00,  3.754408661907416e+00
    };
    double q, r;
    if (p < 0.02425) {
        q = sqrt(-2.0 * log(p));
        return (((((c[0]*q + c[1])*q + c[2])*q + c[3])*q + c[4])*q + c[5]) /
               ((((d[0]*q + d[1])*q + d[2])*q + d[3])*q + 1.0);
    }
    if (p > 0.97575) {
        q = sqrt(-2.0 * log(1.0 - p));
        return -(((((c[0]*q + c[1])*q + c[2])*q + c[3])*q + c[4])*q + c[5]) /
                ((((d[0]*q + d[1])*q + d[2])*q + d[3])*q + 1.0);
    }
    q = p - 0.5;
    r = q * q;
    return (((((a[0]*r + a[1])*r + a[2])*r + a[3])*r + a[4])*r + a[5])*q /
           (((((b[0]*r + b[1])*r + b[2])*r + b[3])*r + b[4])*r + 1.0);
}

static void make_base(float *x, float *qs, base_kind kind) {
    uint32_t i, j;
    for (i = 0; i < N; ++i) {
        for (j = 0; j < D; ++j) {
            double g = grand();
            double v = g;
            if (kind == BASE_LOGNORMAL) v = exp(0.85 * g);
            else if (kind == BASE_OUTLIER && urand() < 0.025)
                v += (urand() < 0.5 ? -1.0 : 1.0) * (6.0 + 4.0 * fabs(grand()));
            x[(size_t)i * D + j] = (float)v;
        }
    }
    for (i = 0; i < Q; ++i) {
        for (j = 0; j < D; ++j) {
            double g = grand();
            double v = g;
            if (kind == BASE_LOGNORMAL) v = exp(0.85 * g);
            else if (kind == BASE_OUTLIER && urand() < 0.025)
                v += (urand() < 0.5 ? -1.0 : 1.0) * (6.0 + 4.0 * fabs(grand()));
            qs[(size_t)i * D + j] = (float)v;
        }
    }
}

static void l2_rows(float *x, uint32_t n) {
    uint32_t i, j;
    for (i = 0; i < n; ++i) {
        double ss = 0.0;
        for (j = 0; j < D; ++j) {
            double v = x[(size_t)i * D + j];
            ss += v * v;
        }
        ss = sqrt(ss);
        if (ss < 1e-12) ss = 1e-12;
        for (j = 0; j < D; ++j)
            x[(size_t)i * D + j] = (float)(x[(size_t)i * D + j] / ss);
    }
}

static void fractional_rows(float *x, uint32_t n) {
    /* Grünwald-Letnikov fractional integral of order 0.75.  The kernel is
     * the binomial expansion of (1-z)^(-0.75); order 1 is ordinary
     * cumulative summation.  Per-row centering removes the DC spike and RMS
     * normalization keeps scales comparable across corpora. */
    enum { FRAC_NUM = 3, FRAC_DEN = 4 };
    double coef[D];
    uint32_t i, j, r;
    coef[0] = 1.0;
    for (r = 1; r < D; ++r)
        coef[r] = coef[r - 1] *
                  ((double)r - 1.0 + (double)FRAC_NUM / FRAC_DEN) / (double)r;
    for (i = 0; i < n; ++i) {
        double mean = 0.0, rms = 0.0;
        float old[D];
        memcpy(old, x + (size_t)i * D, sizeof(old));
        for (j = 0; j < D; ++j) {
            double acc = 0.0;
            for (r = 0; r <= j; ++r)
                acc += coef[r] * old[j - r];
            x[(size_t)i * D + j] = (float)acc;
            mean += acc;
        }
        mean /= (double)D;
        for (j = 0; j < D; ++j) {
            double v = x[(size_t)i * D + j] - mean;
            x[(size_t)i * D + j] = (float)v;
            rms += v * v;
        }
        rms = sqrt(rms / (double)D);
        if (rms < 1e-12) rms = 1e-12;
        for (j = 0; j < D; ++j)
            x[(size_t)i * D + j] = (float)(x[(size_t)i * D + j] / rms);
    }
}

static void log_standardize(float *x, float *qs) {
    double mean[D], var[D];
    uint32_t i, j;
    for (j = 0; j < D; ++j) mean[j] = var[j] = 0.0;
    for (i = 0; i < N; ++i) {
        for (j = 0; j < D; ++j) {
            double v = log((double)x[(size_t)i * D + j]);
            x[(size_t)i * D + j] = (float)v;
            mean[j] += v;
        }
    }
    for (j = 0; j < D; ++j) mean[j] /= (double)N;
    for (i = 0; i < N; ++i) {
        for (j = 0; j < D; ++j) {
            double e = x[(size_t)i * D + j] - mean[j];
            var[j] += e * e;
        }
    }
    for (j = 0; j < D; ++j) {
        double sd = sqrt(var[j] / (double)N);
        if (sd < 1e-12) sd = 1e-12;
        for (i = 0; i < N; ++i)
            x[(size_t)i * D + j] = (float)((x[(size_t)i * D + j] - mean[j]) / sd);
        for (i = 0; i < Q; ++i) {
            double v = log((double)qs[(size_t)i * D + j]);
            qs[(size_t)i * D + j] = (float)((v - mean[j]) / sd);
        }
    }
}

static uint32_t lower_pos(const float *sorted, float x) {
    uint32_t lo = 0, hi = N;
    while (lo < hi) {
        uint32_t mid = lo + (hi - lo) / 2u;
        if (sorted[mid] < x) lo = mid + 1u;
        else hi = mid;
    }
    return lo;
}

static void rank_gaussianize(float *x, float *qs) {
    float *sorted = (float *)malloc(sizeof(float) * N);
    uint32_t i, j;
    if (!sorted) exit(2);
    for (j = 0; j < D; ++j) {
        for (i = 0; i < N; ++i) sorted[i] = x[(size_t)i * D + j];
        qsort(sorted, N, sizeof(float), cmp_float);
        for (i = 0; i < N; ++i) {
            uint32_t r = lower_pos(sorted, x[(size_t)i * D + j]);
            double u = ((double)r + 0.5) / (double)N;
            x[(size_t)i * D + j] = (float)normal_quantile(u);
        }
        for (i = 0; i < Q; ++i) {
            uint32_t r = lower_pos(sorted, qs[(size_t)i * D + j]);
            double u;
            if (r >= N) r = N - 1u;
            u = ((double)r + 0.5) / (double)N;
            qs[(size_t)i * D + j] = (float)normal_quantile(u);
        }
    }
    free(sorted);
}

static void apply_transform(float *x, float *qs, transform_kind tr) {
    if (tr == TR_L2) {
        l2_rows(x, N);
        l2_rows(qs, Q);
    } else if (tr == TR_FRACTIONAL) {
        fractional_rows(x, N);
        fractional_rows(qs, Q);
    } else if (tr == TR_LOG) {
        log_standardize(x, qs);
    } else if (tr == TR_RANK) {
        rank_gaussianize(x, qs);
    }
}

static void exact_topk(const float *x, const float *q, uint32_t *ids) {
    double best[K];
    uint32_t i, r;
    for (r = 0; r < K; ++r) { best[r] = HUGE_VAL; ids[r] = UINT32_MAX; }
    for (i = 0; i < N; ++i) {
        const float *v = x + (size_t)i * D;
        double d = 0.0;
        uint32_t j, at;
        for (j = 0; j < D; ++j) {
            double e = (double)q[j] - (double)v[j];
            d += e * e;
        }
        if (d < best[K - 1]) {
            at = K - 1;
            while (at > 0 && best[at - 1] > d) {
                best[at] = best[at - 1]; ids[at] = ids[at - 1]; --at;
            }
            best[at] = d; ids[at] = i;
        }
    }
}

static float overlap(const uint32_t *a, const uint32_t *b, uint32_t n) {
    uint32_t i, j, hit = 0;
    for (i = 0; i < n; ++i)
        for (j = 0; j < n; ++j)
            if (a[i] == b[j]) { ++hit; break; }
    return (float)hit / (float)n;
}

static const char *search_name(zenith_search_t s) {
    if (s == ZENITH_SEARCH_GRAY) return "gray";
    if (s == ZENITH_SEARCH_HAMMING) return "hamming";
    return "posterior";
}

static void evaluate_case(const case_def *def, const float *x, const float *qs,
                          const uint32_t *raw_truth) {
    static const zenith_search_t searches[] = {
        ZENITH_SEARCH_GRAY, ZENITH_SEARCH_HAMMING, ZENITH_SEARCH_POSTERIOR
    };
    uint32_t truth[Q * K];
    float full_w[D];
    double truth_shift = 0.0;
    uint32_t qi, si, j;

    for (qi = 0; qi < Q; ++qi)
        exact_topk(x, qs + (size_t)qi * D, truth + (size_t)qi * K);
    for (qi = 0; qi < Q; ++qi)
        truth_shift += overlap(truth + (size_t)qi * K,
                               raw_truth + (size_t)qi * K, K);
    truth_shift /= (double)Q;

    zenith_w_identity(D, full_w);

    for (si = 0; si < sizeof(searches) / sizeof(searches[0]); ++si) {
        zenith_opts opts;
        zenith_index_t *idx;
        double recall = 0.0, pool_recall = 0.0, rerank_recall = 0.0;
        double raw_task = 0.0, tail_fraction = 0.0, lat[Q];
        uint32_t pool[POOL], rids[K];
        float dist[K], rdist[K];

        memset(&opts, 0, sizeof(opts));
        opts.Mcoef = MCOEF;
        opts.nbits = 64;
        opts.seed = 20260830u;
        opts.gen = ZENITH_GEN_SIGNED_DCT;
        opts.search = searches[si];
        idx = zenith_build(x, N, D, opts);
        if (!idx) exit(2);

        for (qi = 0; qi < N; ++qi) {
            double ss = 0.0;
            for (j = 0; j < D; ++j) {
                double v = x[(size_t)qi * D + j];
                ss += v * v;
            }
            if (ss > 0.0)
                tail_fraction += (double)idx->residual[qi] * idx->residual[qi] / ss;
        }
        tail_fraction /= (double)N;

        zenith_query_ef(idx, qs, full_w, K, POOL, rids, dist);
        for (qi = 0; qi < Q; ++qi) {
            const float *q = qs + (size_t)qi * D;
            uint64_t t0, t1;
            uint32_t got, nc, ng;
            t0 = zenith__test_now_ns();
            got = zenith_query_ef(idx, q, full_w, K, POOL, rids, dist);
            t1 = zenith__test_now_ns();
            lat[qi] = (double)(t1 - t0) / 1000.0;
            recall += overlap(truth + (size_t)qi * K, rids, got < K ? got : K);
            raw_task += overlap(raw_truth + (size_t)qi * K, rids,
                                got < K ? got : K);
            nc = zenith_candidates_ef(idx, q, full_w, POOL, pool);
            {
                uint32_t a, b, hit = 0;
                for (a = 0; a < K; ++a)
                    for (b = 0; b < nc; ++b)
                        if (truth[qi * K + a] == pool[b]) { ++hit; break; }
                pool_recall += (double)hit / (double)K;
            }
            ng = zenith_rerank_exact(idx, x, q, full_w, pool, nc, K,
                                     rids, rdist);
            rerank_recall += overlap(truth + (size_t)qi * K, rids,
                                     ng < K ? ng : K);
        }
        qsort(lat, Q, sizeof(double), cmp_double);
        printf("%-9s %-20s %-9s %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.1f %9.0f\n",
               def->base, def->prep, search_name(searches[si]),
               recall / Q, pool_recall / Q, rerank_recall / Q,
               truth_shift, raw_task / Q, tail_fraction,
               lat[(Q * 95u) / 100u], 1e6 / (lat[Q / 2] > 0.0 ? lat[Q / 2] : 1e-9));
        zenith_free(idx);
    }
}

/* Local clock helper kept separate so the example remains readable. */
static uint64_t zenith__test_now_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * UINT64_C(1000000000) + (uint64_t)ts.tv_nsec;
}

int main(void) {
    static const case_def cases[] = {
        {"gaussian",  "raw",                  BASE_GAUSSIAN, TR_NONE},
        {"gaussian",  "l2 normalize",         BASE_GAUSSIAN, TR_L2},
        {"gaussian",  "fractional integrate", BASE_GAUSSIAN, TR_FRACTIONAL},
        {"lognormal", "raw",                  BASE_LOGNORMAL, TR_NONE},
        {"lognormal", "log + standardize",    BASE_LOGNORMAL, TR_LOG},
        {"outlier",   "raw",                  BASE_OUTLIER,  TR_NONE},
        {"outlier",   "rank-to-Gaussian",     BASE_OUTLIER,  TR_RANK}
    };
    float *base[3], *queries[3];
    uint32_t *raw_truth[3];
    float *x = (float *)malloc(sizeof(float) * N * D);
    float *qs = (float *)malloc(sizeof(float) * Q * D);
    size_t c;
    int b;

    if (!x || !qs) return 2;
    for (b = 0; b < 3; ++b) {
        base[b] = (float *)malloc(sizeof(float) * N * D);
        queries[b] = (float *)malloc(sizeof(float) * Q * D);
        raw_truth[b] = (uint32_t *)malloc(sizeof(uint32_t) * Q * K);
        if (!base[b] || !queries[b] || !raw_truth[b]) return 2;
        rng_state = UINT64_C(0x5eed0000) + (uint64_t)b * UINT64_C(0x9e3779b9);
        make_base(base[b], queries[b], (base_kind)b);
        for (uint32_t qi = 0; qi < Q; ++qi)
            exact_topk(base[b], queries[b] + (size_t)qi * D,
                       raw_truth[b] + (size_t)qi * K);
    }

    puts("Preprocessing and Gaussian candidate modes (identity metric, M=32, pool=160)");
    puts("recall/pool/rerank are against exact top-10 in the transformed space;");
    puts("truth-shift is exact top-10 overlap with the raw-space task.");
    printf("%-9s %-20s %-9s %8s %8s %8s %8s %8s %8s %8s %9s\n",
           "base", "preprocess", "search", "recall", "pool", "rerank",
           "shift", "raw-task", "tail-E", "p95 us", "QPS");
    for (c = 0; c < sizeof(cases) / sizeof(cases[0]); ++c) {
        memcpy(x, base[cases[c].base_kind], sizeof(float) * N * D);
        memcpy(qs, queries[cases[c].base_kind], sizeof(float) * Q * D);
        apply_transform(x, qs, cases[c].transform);
        evaluate_case(&cases[c], x, qs, raw_truth[cases[c].base_kind]);
    }

    for (b = 0; b < 3; ++b) {
        free(base[b]); free(queries[b]); free(raw_truth[b]);
    }
    free(x); free(qs);
    return 0;
}
