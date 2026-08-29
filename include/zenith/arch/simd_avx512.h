#ifndef ZENITH_ARCH_SIMD_AVX512_H
#define ZENITH_ARCH_SIMD_AVX512_H

static float zenith__dot(const float *ZENITH_RESTRICT a,
                         const float *ZENITH_RESTRICT b,
                         uint32_t n) {
    __m512 s0 = _mm512_setzero_ps(), s1 = _mm512_setzero_ps();
    __m512 s2 = _mm512_setzero_ps(), s3 = _mm512_setzero_ps();
    uint32_t i = 0;
    float d;
    for (; i + 64 <= n; i += 64) {
        s0 = _mm512_fmadd_ps(_mm512_loadu_ps(a + i),      _mm512_loadu_ps(b + i),      s0);
        s1 = _mm512_fmadd_ps(_mm512_loadu_ps(a + i + 16), _mm512_loadu_ps(b + i + 16), s1);
        s2 = _mm512_fmadd_ps(_mm512_loadu_ps(a + i + 32), _mm512_loadu_ps(b + i + 32), s2);
        s3 = _mm512_fmadd_ps(_mm512_loadu_ps(a + i + 48), _mm512_loadu_ps(b + i + 48), s3);
    }
    s0 = _mm512_add_ps(_mm512_add_ps(s0, s1), _mm512_add_ps(s2, s3));
    for (; i + 16 <= n; i += 16)
        s0 = _mm512_fmadd_ps(_mm512_loadu_ps(a + i), _mm512_loadu_ps(b + i), s0);
    d = _mm512_reduce_add_ps(s0);
    for (; i < n; ++i) d += a[i] * b[i];
    return d;
}

static float zenith__wdist(const float *ZENITH_RESTRICT Q,
                           const float *ZENITH_RESTRICT X,
                           const float *ZENITH_RESTRICT w,
                           uint32_t Mc) {
    __m512 sm0 = _mm512_setzero_ps(), sm1 = _mm512_setzero_ps();
    __m512 sm2 = _mm512_setzero_ps(), sm3 = _mm512_setzero_ps();
    uint32_t j = 0;
    float d;
    for (; j + 64 <= Mc; j += 64) {
        __m512 e0 = _mm512_sub_ps(_mm512_loadu_ps(Q + j),      _mm512_loadu_ps(X + j));
        __m512 e1 = _mm512_sub_ps(_mm512_loadu_ps(Q + j + 16), _mm512_loadu_ps(X + j + 16));
        __m512 e2 = _mm512_sub_ps(_mm512_loadu_ps(Q + j + 32), _mm512_loadu_ps(X + j + 32));
        __m512 e3 = _mm512_sub_ps(_mm512_loadu_ps(Q + j + 48), _mm512_loadu_ps(X + j + 48));
        sm0 = _mm512_fmadd_ps(_mm512_loadu_ps(w + j),      _mm512_mul_ps(e0, e0), sm0);
        sm1 = _mm512_fmadd_ps(_mm512_loadu_ps(w + j + 16), _mm512_mul_ps(e1, e1), sm1);
        sm2 = _mm512_fmadd_ps(_mm512_loadu_ps(w + j + 32), _mm512_mul_ps(e2, e2), sm2);
        sm3 = _mm512_fmadd_ps(_mm512_loadu_ps(w + j + 48), _mm512_mul_ps(e3, e3), sm3);
    }
    sm0 = _mm512_add_ps(_mm512_add_ps(sm0, sm1), _mm512_add_ps(sm2, sm3));
    for (; j + 16 <= Mc; j += 16) {
        __m512 e = _mm512_sub_ps(_mm512_loadu_ps(Q + j), _mm512_loadu_ps(X + j));
        sm0 = _mm512_fmadd_ps(_mm512_loadu_ps(w + j), _mm512_mul_ps(e, e), sm0);
    }
    d = _mm512_reduce_add_ps(sm0);
    for (; j < Mc; ++j) {
        float e = Q[j] - X[j];
        d += w[j] * e * e;
    }
    return d;
}

#endif
