#ifndef ZENITH_ARCH_SIMD_NEON_H
#define ZENITH_ARCH_SIMD_NEON_H

static float zenith__vsumq_f32(float32x4_t v) {
    float32x2_t t = vadd_f32(vget_low_f32(v), vget_high_f32(v));
    return vget_lane_f32(vpadd_f32(t, t), 0);
}

static float zenith__dot(const float *ZENITH_RESTRICT a,
                         const float *ZENITH_RESTRICT b,
                         uint32_t n) {
    float32x4_t s0 = vdupq_n_f32(0.0f), s1 = vdupq_n_f32(0.0f);
    float32x4_t s2 = vdupq_n_f32(0.0f), s3 = vdupq_n_f32(0.0f);
    uint32_t i = 0;
    float d;
    for (; i + 16 <= n; i += 16) {
        s0 = vmlaq_f32(s0, vld1q_f32(a + i),      vld1q_f32(b + i));
        s1 = vmlaq_f32(s1, vld1q_f32(a + i + 4),  vld1q_f32(b + i + 4));
        s2 = vmlaq_f32(s2, vld1q_f32(a + i + 8),  vld1q_f32(b + i + 8));
        s3 = vmlaq_f32(s3, vld1q_f32(a + i + 12), vld1q_f32(b + i + 12));
    }
    s0 = vaddq_f32(vaddq_f32(s0, s1), vaddq_f32(s2, s3));
    for (; i + 4 <= n; i += 4)
        s0 = vmlaq_f32(s0, vld1q_f32(a + i), vld1q_f32(b + i));
    d = zenith__vsumq_f32(s0);
    for (; i < n; ++i) d += a[i] * b[i];
    return d;
}

static float zenith__wdist(const float *ZENITH_RESTRICT Q,
                           const float *ZENITH_RESTRICT X,
                           const float *ZENITH_RESTRICT w,
                           uint32_t Mc) {
    float32x4_t s0 = vdupq_n_f32(0.0f), s1 = vdupq_n_f32(0.0f);
    float32x4_t s2 = vdupq_n_f32(0.0f), s3 = vdupq_n_f32(0.0f);
    uint32_t j = 0;
    float d;
    for (; j + 16 <= Mc; j += 16) {
        float32x4_t e0 = vsubq_f32(vld1q_f32(Q + j),      vld1q_f32(X + j));
        float32x4_t e1 = vsubq_f32(vld1q_f32(Q + j + 4),  vld1q_f32(X + j + 4));
        float32x4_t e2 = vsubq_f32(vld1q_f32(Q + j + 8),  vld1q_f32(X + j + 8));
        float32x4_t e3 = vsubq_f32(vld1q_f32(Q + j + 12), vld1q_f32(X + j + 12));
        s0 = vmlaq_f32(s0, vld1q_f32(w + j),      vmulq_f32(e0, e0));
        s1 = vmlaq_f32(s1, vld1q_f32(w + j + 4),  vmulq_f32(e1, e1));
        s2 = vmlaq_f32(s2, vld1q_f32(w + j + 8),  vmulq_f32(e2, e2));
        s3 = vmlaq_f32(s3, vld1q_f32(w + j + 12), vmulq_f32(e3, e3));
    }
    s0 = vaddq_f32(vaddq_f32(s0, s1), vaddq_f32(s2, s3));
    for (; j + 4 <= Mc; j += 4) {
        float32x4_t e = vsubq_f32(vld1q_f32(Q + j), vld1q_f32(X + j));
        s0 = vmlaq_f32(s0, vld1q_f32(w + j), vmulq_f32(e, e));
    }
    d = zenith__vsumq_f32(s0);
    for (; j < Mc; ++j) {
        float e = Q[j] - X[j];
        d += w[j] * e * e;
    }
    return d;
}

#endif
