/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    layernorm.cpp

Abstract:

    This module implements the dispatch for platform-optimized
    LayerNorm/RMSNorm kernels.

--*/

#include "mlasi.h"

#if defined(MLAS_USE_APPLE_ACCELERATE) && defined(__APPLE__) && defined(MLAS_TARGET_ARM64)

#include <cassert>
#include <cmath>
#include <memory>

//
// This kernel only needs a handful of vDSP vector/reduction routines and
// never touches BLAS/LAPACK. Do not include <Accelerate/Accelerate.h>: on
// recent macOS SDKs (observed on Xcode 26.3 / MacOSX26.2, see PR #32036's
// discovery for an earlier, unrelated Apple Accelerate kernel -- that PR was
// closed without merging, but the SDK incompatibility it found still
// applies here) vecLib's cblas.h/cblas_new.h forward-declare BLAS enums
// without an inline definition, which is valid C but rejected by ISO C++
// ("ISO C++ forbids
// forward references to 'enum' types"), breaking the build for any C++
// translation unit that includes the umbrella header on that SDK. Forward-
// declaring exactly the vDSP entry points used here avoids the umbrella
// header (and therefore the broken BLAS headers) entirely; the Accelerate
// framework is already linked unconditionally under this option via
// cmake/onnxruntime_mlas.cmake, so these symbols resolve at link time exactly
// as they would via the header.
//
// Signatures verified against Apple's public, long-stable vDSP API
// (vDSP_Length = unsigned long, vDSP_Stride = long on this ABI; spelled out
// directly below instead of re-declaring those typedefs to avoid any risk of
// colliding with a real Accelerate header elsewhere in the translation unit).
//
extern "C" {
void vDSP_meanv(const float* __A, long __IA, float* __C, unsigned long __N);
void vDSP_measqv(const float* __A, long __IA, float* __C, unsigned long __N);
void vDSP_svesq(const float* __A, long __IA, float* __C, unsigned long __N);
void vDSP_vsadd(const float* __A, long __IA, const float* __B, float* __C, long __IC, unsigned long __N);
void vDSP_vsmul(const float* __A, long __IA, const float* __B, float* __C, long __IC, unsigned long __N);
void vDSP_vmul(const float* __A, long __IA, const float* __B, long __IB, float* __C, long __IC, unsigned long __N);
void vDSP_vma(
    const float* __A, long __IA, const float* __B, long __IB, const float* __C, long __IC, float* __D, long __ID,
    unsigned long __N
);
}

// Fixed-size on-stack scratch buffer covering the hidden sizes of essentially
// every current transformer model (BERT 768, GPT-2 768/1024/1280/1600,
// Llama-2/3 4096-8192, Phi-3 3072, Gemma 3072) without a heap allocation in
// the hot per-row path. NormSize beyond this falls back to a one-off heap
// allocation; still correct, just not zero-allocation for that rare case.
constexpr size_t kApplePerRowStackScratch = 8192;

// The vDSP call overhead outweighs its vectorization benefit for very short
// rows. Apple Silicon benchmarks show the crossover at 64 elements, so let
// the caller use its scalar fallback below that size.
constexpr size_t kAppleAccelerateLayerNormMinimumElements = 64;

void
MLASCALL
MlasLayerNormKernelAppleAccelerate(
    const float* Input,
    const float* Scale,
    const float* Bias,
    float* Output,
    float* MeanOut,
    float* InvStdDevOut,
    size_t NormSize,
    float Epsilon,
    bool Simplified
    )
/*++

Routine Description:

    This routine computes LayerNorm/RMSNorm for one normalization row using
    Apple's Accelerate vDSP library, available on macOS arm64 when
    onnxruntime_USE_APPLE_ACCELERATE is enabled.

    ARM64 has no other SIMD-vectorized MlasLayerNormF32 kernel. Without this
    kernel, layer_norm_impl.cc uses scalar Welford or sum-of-squares loops.

    Algorithm: centered two-pass, rather than a single-pass "E[x^2] -
    mean^2" formula. The uncentered formula suffers catastrophic
    cancellation in fp32 for large-base/small-spread inputs; this kernel
    deliberately avoids that failure mode at the cost of a second pass over
    each row:

      RMSNorm (Simplified):   meanSq = vDSP_measqv(x)              -- E[x^2]
                              denom  = sqrt(meanSq + eps)
      LayerNorm (full):       mean   = vDSP_meanv(x)
                              c[i]   = x[i] - mean                 -- vDSP_vsadd
                              sumSq  = vDSP_svesq(c)                -- sum(c^2)
                              denom  = sqrt(sumSq / N + eps)

    then in both cases:        y[i]   = c[i] * (1 / denom)          -- vDSP_vsmul
                              out[i] = y[i] * Scale[i] + Bias[i]   -- vDSP_vma
                                        (vDSP_vmul if Bias == nullptr)

    Aliasing: Input and Output may be the same buffer (several real ORT
    LayerNorm/SkipLayerNorm call sites reuse the input tensor as the output
    tensor). This is safe here because Input is only ever *read* (during the
    reduction passes and the initial vsadd/vsmul into a separate scratch
    buffer) before Output is written; Output is written only in the final
    vDSP_vma/vDSP_vmul call, whose operands are the scratch buffer, Scale, and
    Bias -- never Input directly. No vDSP call in this routine has Input and
    Output as the same argument.

    vDSP is a synchronous vectorized call with no internal GCD/thread-pool
    dispatch of its own, so it does not oversubscribe the ORT threadpool
    (same property already established for the sibling vForce Tanh kernel).

Arguments:

    Input - Supplies the input buffer for one normalization row (NormSize
        elements).

    Scale - Supplies the per-element scale buffer (NormSize elements).

    Bias - Supplies the optional per-element bias buffer (NormSize elements),
        or nullptr. Simplified (RMSNorm) callers must pass nullptr here --
        RMSNorm has no bias term in the ONNX SimplifiedLayerNormalization
        contract.

    Output - Supplies the output buffer (NormSize elements). May alias Input.

    MeanOut - Supplies an optional pointer to receive the computed mean, or
        nullptr. Written for both modes for consistency with the RVV kernel,
        though the RMSNorm/Simplified ONNX contract has no Mean output and
        essentially never passes a non-null pointer here.

    InvStdDevOut - Supplies an optional pointer to receive 1/denom, or
        nullptr.

    NormSize - Supplies the number of elements in the row.

    Epsilon - Supplies the epsilon added to the variance (or mean-square, for
        RMSNorm) before the square root, matching the ONNX
        (Simplified)LayerNormalization contract exactly (epsilon is inside
        the sqrt, not added after).

    Simplified - Supplies true for RMSNorm (no mean subtraction, no bias),
        false for full LayerNorm.

Return Value:

    None.

--*/
{
    assert(!Simplified || Bias == nullptr);

    if (NormSize == 0) {
        if (MeanOut != nullptr) {
            *MeanOut = 0.0f;
        }
        if (InvStdDevOut != nullptr) {
            *InvStdDevOut = 0.0f;
        }
        return;
    }

    float stack_scratch[kApplePerRowStackScratch];
    // Every element of `scratch` (stack or heap) is fully overwritten by the
    // vDSP reduction/elementwise calls below before it is ever read, so an
    // uninitialized heap allocation is safe and avoids the zero-fill cost a
    // std::vector<float>::resize() would otherwise pay on every large-row
    // call (NormSize > kApplePerRowStackScratch).
    std::unique_ptr<float[]> heap_scratch;
    float* scratch = stack_scratch;
    if (NormSize > kApplePerRowStackScratch) {
        heap_scratch.reset(new float[NormSize]);
        scratch = heap_scratch.get();
    }

    const auto n = static_cast<unsigned long>(NormSize);
    float mean = 0.0f;
    float denom;

    if (Simplified) {
        float mean_sq;
        vDSP_measqv(Input, 1, &mean_sq, n);
        denom = std::sqrt(mean_sq + Epsilon);

        const float inv_denom = 1.0f / denom;
        vDSP_vsmul(Input, 1, &inv_denom, scratch, 1, n);

        if (MeanOut != nullptr) {
            vDSP_meanv(Input, 1, &mean, n);
        }
    } else {
        vDSP_meanv(Input, 1, &mean, n);
        const float neg_mean = -mean;
        // scratch[i] = Input[i] - mean
        vDSP_vsadd(Input, 1, &neg_mean, scratch, 1, n);

        float sum_sq;
        vDSP_svesq(scratch, 1, &sum_sq, n);
        denom = std::sqrt(sum_sq / static_cast<float>(NormSize) + Epsilon);

        const float inv_denom = 1.0f / denom;
        // scratch[i] = (Input[i] - mean) * inv_denom, in place.
        vDSP_vsmul(scratch, 1, &inv_denom, scratch, 1, n);
    }

    if (Bias != nullptr) {
        vDSP_vma(scratch, 1, Scale, 1, Bias, 1, Output, 1, n);
    } else {
        vDSP_vmul(scratch, 1, Scale, 1, Output, 1, n);
    }

    if (MeanOut != nullptr) {
        *MeanOut = mean;
    }
    if (InvStdDevOut != nullptr) {
        *InvStdDevOut = 1.0f / denom;
    }
}

#endif  // MLAS_USE_APPLE_ACCELERATE && __APPLE__ && MLAS_TARGET_ARM64

bool
    MLASCALL
    MlasLayerNormF32(
        const float* Input,
        const float* Scale,
        const float* Bias,
        float* Output,
        float* MeanOut,
        float* InvStdDevOut,
        size_t NormSize,
        float Epsilon,
        bool Simplified
    )
{
#if defined(MLAS_USE_APPLE_ACCELERATE) && defined(__APPLE__) && defined(MLAS_TARGET_ARM64)
    if (NormSize > 0 && NormSize < kAppleAccelerateLayerNormMinimumElements) {
        return false;
    }
#endif

    auto kernel = GetMlasPlatform().LayerNormF32Kernel;
    if (kernel == nullptr) {
        return false;
    }

    kernel(Input, Scale, Bias, Output, MeanOut, InvStdDevOut, NormSize, Epsilon, Simplified);
    return true;
}
