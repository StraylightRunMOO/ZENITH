#ifndef ZENITH_ARCH_SIMD_SCALAR_H
#define ZENITH_ARCH_SIMD_SCALAR_H

static float zenith__dot(const float *ZENITH_RESTRICT a,
                         const float *ZENITH_RESTRICT b,
                         uint32_t n) {
    uint32_t i = 0;
    float s0 = 0.0f, s1 = 0.0f, s2 = 0.0f, s3 = 0.0f;
    for (; i + 4 <= n; i += 4) {
        s0 += a[i]     * b[i];
        s1 += a[i + 1] * b[i + 1];
        s2 += a[i + 2] * b[i + 2];
        s3 += a[i + 3] * b[i + 3];
    }
    s0 = (s0 + s1) + (s2 + s3);
    for (; i < n; ++i) s0 += a[i] * b[i];
    return s0;
}

static float zenith__wdist(const float *ZENITH_RESTRICT Q,
                           const float *ZENITH_RESTRICT X,
                           const float *ZENITH_RESTRICT w,
                           uint32_t Mc) {
    uint32_t j = 0;
    float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
    for (; j + 4 <= Mc; j += 4) {
        float e0 = Q[j]     - X[j];
        float e1 = Q[j + 1] - X[j + 1];
        float e2 = Q[j + 2] - X[j + 2];
        float e3 = Q[j + 3] - X[j + 3];
        a0 += w[j]     * e0 * e0;
        a1 += w[j + 1] * e1 * e1;
        a2 += w[j + 2] * e2 * e2;
        a3 += w[j + 3] * e3 * e3;
    }
    a0 = (a0 + a1) + (a2 + a3);
    for (; j < Mc; ++j) {
        float e = Q[j] - X[j];
        a0 += w[j] * e * e;
    }
    return a0;
}

#endif
