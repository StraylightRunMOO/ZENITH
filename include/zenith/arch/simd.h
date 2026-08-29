#ifndef ZENITH_ARCH_SIMD_H
#define ZENITH_ARCH_SIMD_H

/* Compile-time SIMD kernels for projection dots and weighted scoring.
 * These files, plus cuda.h, are the interchangeable architecture-specific
 * interfaces. There is no runtime dispatch. */

#if !defined(ZENITH_SIMD_AVX512) && !defined(ZENITH_SIMD_AVX2) && \
    !defined(ZENITH_SIMD_NEON)   && !defined(ZENITH_SIMD_SCALAR)
#  if defined(__AVX512F__)
#    define ZENITH_SIMD_AVX512
#  elif defined(__AVX2__) && defined(__FMA__)
#    define ZENITH_SIMD_AVX2
#  elif defined(__ARM_NEON) || defined(__aarch64__)
#    define ZENITH_SIMD_NEON
#  else
#    define ZENITH_SIMD_SCALAR
#  endif
#endif

#if defined(ZENITH_SIMD_AVX512) || defined(ZENITH_SIMD_AVX2)
#  include <immintrin.h>
#endif
#if defined(ZENITH_SIMD_NEON)
#  include <arm_neon.h>
#endif

int zenith_simd_level(void) {
#if defined(ZENITH_SIMD_AVX512)
    return 3;
#elif defined(ZENITH_SIMD_AVX2)
    return 2;
#elif defined(ZENITH_SIMD_NEON)
    return 1;
#else
    return 0;
#endif
}

#if defined(ZENITH_SIMD_AVX512)
#  include "simd_avx512.h"
#elif defined(ZENITH_SIMD_AVX2)
#  include "simd_avx2.h"
#elif defined(ZENITH_SIMD_NEON)
#  include "simd_neon.h"
#else
#  include "simd_scalar.h"
#endif

#endif
