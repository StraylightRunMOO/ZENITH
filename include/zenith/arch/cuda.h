#ifndef ZENITH_ARCH_CUDA_H
#define ZENITH_ARCH_CUDA_H

/* Reserved GPU architecture slot. A future CUDA path should expose the same
 * kernels as simd.h (zenith__dot, zenith__wdist) plus a device DCT-II
 * projector and a batch query entry that owns device copies of coeffs/keys.
 * Host OpenMP (zenith/threads.h) and FFTW (zenith/dct.h) stay on the CPU
 * side; do not mix planner state with the device stream. */

#if defined(ZENITH_USE_CUDA)
#  error "ZENITH_USE_CUDA is reserved; the CUDA backend is not in this release."
#endif

#endif
