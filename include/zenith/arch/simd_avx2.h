#ifndef ZENITH_ARCH_SIMD_AVX2_H
#define ZENITH_ARCH_SIMD_AVX2_H

static float zenith__hsum256(__m256 v) {
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 hi = _mm256_extractf128_ps(v, 1);
    lo = _mm_add_ps(lo, hi);
    {
        __m128 sh = _mm_movehdup_ps(lo);
        lo = _mm_add_ps(lo, sh);
        sh = _mm_movehl_ps(sh, lo);
        lo = _mm_add_ss(lo, sh);
    }
    return _mm_cvtss_f32(lo);
}

static float zenith__dot(const float *ZENITH_RESTRICT a,
                         const float *ZENITH_RESTRICT b,
                         uint32_t n) {
    __m256 s0 = _mm256_setzero_ps(), s1 = _mm256_setzero_ps();
    __m256 s2 = _mm256_setzero_ps(), s3 = _mm256_setzero_ps();
    uint32_t i = 0;
    float d;
    for (; i + 32 <= n; i += 32) {
        s0 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i),      _mm256_loadu_ps(b + i),      s0);
        s1 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i + 8),  _mm256_loadu_ps(b + i + 8),  s1);
        s2 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i + 16), _mm256_loadu_ps(b + i + 16), s2);
        s3 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i + 24), _mm256_loadu_ps(b + i + 24), s3);
    }
    s0 = _mm256_add_ps(_mm256_add_ps(s0, s1), _mm256_add_ps(s2, s3));
    for (; i + 8 <= n; i += 8)
        s0 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i), s0);
    d = zenith__hsum256(s0);
    for (; i < n; ++i) d += a[i] * b[i];
    return d;
}

static float zenith__wdist(const float *ZENITH_RESTRICT Q,
                           const float *ZENITH_RESTRICT X,
                           const float *ZENITH_RESTRICT w,
                           uint32_t Mc) {
    __m256 sm0 = _mm256_setzero_ps(), sm1 = _mm256_setzero_ps();
    __m256 sm2 = _mm256_setzero_ps(), sm3 = _mm256_setzero_ps();
    uint32_t j = 0;
    float d;
    for (; j + 32 <= Mc; j += 32) {
        __m256 e0 = _mm256_sub_ps(_mm256_loadu_ps(Q + j),      _mm256_loadu_ps(X + j));
        __m256 e1 = _mm256_sub_ps(_mm256_loadu_ps(Q + j + 8),  _mm256_loadu_ps(X + j + 8));
        __m256 e2 = _mm256_sub_ps(_mm256_loadu_ps(Q + j + 16), _mm256_loadu_ps(X + j + 16));
        __m256 e3 = _mm256_sub_ps(_mm256_loadu_ps(Q + j + 24), _mm256_loadu_ps(X + j + 24));
        sm0 = _mm256_fmadd_ps(_mm256_loadu_ps(w + j),      _mm256_mul_ps(e0, e0), sm0);
        sm1 = _mm256_fmadd_ps(_mm256_loadu_ps(w + j + 8),  _mm256_mul_ps(e1, e1), sm1);
        sm2 = _mm256_fmadd_ps(_mm256_loadu_ps(w + j + 16), _mm256_mul_ps(e2, e2), sm2);
        sm3 = _mm256_fmadd_ps(_mm256_loadu_ps(w + j + 24), _mm256_mul_ps(e3, e3), sm3);
    }
    sm0 = _mm256_add_ps(_mm256_add_ps(sm0, sm1), _mm256_add_ps(sm2, sm3));
    for (; j + 8 <= Mc; j += 8) {
        __m256 e = _mm256_sub_ps(_mm256_loadu_ps(Q + j), _mm256_loadu_ps(X + j));
        sm0 = _mm256_fmadd_ps(_mm256_loadu_ps(w + j), _mm256_mul_ps(e, e), sm0);
    }
    d = zenith__hsum256(sm0);
    for (; j < Mc; ++j) {
        float e = Q[j] - X[j];
        d += w[j] * e * e;
    }
    return d;
}

#endif
