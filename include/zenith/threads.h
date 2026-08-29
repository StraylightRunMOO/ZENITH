#ifndef ZENITH_THREADS_H
#define ZENITH_THREADS_H

/* OpenMP helpers. Parallel regions use static schedules so each thread
 * repeatedly touches the same index slice (first-touch NUMA). Nested
 * parallel regions are refused so a batch of queries does not oversubscribe.
 * Shared host-side threading, not an ISA-specific kernel. */

#ifndef ZENITH_OMP_MIN_WORK
#define ZENITH_OMP_MIN_WORK 512u
#endif

#if defined(_OPENMP)
#  include <omp.h>
#endif

static int zenith__threads_override = 0;

void zenith_set_threads(int n) {
    zenith__threads_override = n < 0 ? 0 : n;
}

int zenith_openmp_enabled(void) {
#if defined(_OPENMP)
    return 1;
#else
    return 0;
#endif
}

int zenith_thread_count(void) {
#if defined(_OPENMP)
    {
        int n = zenith__threads_override > 0 ? zenith__threads_override
                                             : omp_get_max_threads();
        return n < 1 ? 1 : n;
    }
#else
    (void)zenith__threads_override;
    return 1;
#endif
}

static int zenith__index_threads(const zenith_index_t *idx) {
    if (idx && idx->nthreads > 0) return (int)idx->nthreads;
    return zenith_thread_count();
}

static int zenith__want_parallel(const zenith_index_t *idx, uint32_t work) {
#if defined(_OPENMP)
    if (omp_in_parallel()) return 0;
    if (work < ZENITH_OMP_MIN_WORK) return 0;
    return zenith__index_threads(idx) > 1;
#else
    (void)idx; (void)work;
    return 0;
#endif
}

#if defined(_OPENMP)
static void zenith__hit_insert(zenith_hit *heap, int *hn, uint32_t k,
                               float dist, uint32_t id) {
    zenith_hit cand;
    cand.dist = dist;
    cand.id = id;
    if (*hn < (int)k) {
        heap[*hn] = cand;
        (*hn)++;
        if (*hn == (int)k) {
            int ii;
            for (ii = (int)k / 2 - 1; ii >= 0; --ii)
                zenith_heap_sift(heap, *hn, ii);
        }
    } else if (zenith_hit_better(cand, heap[0])) {
        heap[0] = cand;
        zenith_heap_sift(heap, *hn, 0);
    }
}

static void zenith__heap_merge(zenith_hit *dst, int *dn, uint32_t k,
                               const zenith_hit *src, int sn) {
    int i;
    for (i = 0; i < sn; ++i)
        zenith__hit_insert(dst, dn, k, src[i].dist, src[i].id);
}
#endif /* _OPENMP */

#endif /* ZENITH_THREADS_H */
