/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    tanh.cpp

Abstract:

    This module implements routines to compute the hyperbolic tangent function.

    This implementation uses the same polynomial coefficients and algorithm as
    found in Eigen. Our usage requires building platform specific versions of
    the algorithm to target different instruction sets. The implementation below
    targets the base instruction set (typically SSE2) while assembly
    implementations target newer instruction sets (such as FMA3).

--*/

#include "mlasi.h"
#include "elementwise_constants.h"
#include "softmax.h"

#if defined(MLAS_USE_APPLE_ACCELERATE) && defined(__APPLE__) && defined(MLAS_TARGET_ARM64)
//
// This kernel only needs vForce's vvtanhf() and does not use any BLAS/LAPACK
// functionality from the Accelerate umbrella header. On recent macOS SDKs
// (observed on the Xcode 26.3 / MacOSX26.2 SDK), Accelerate's new LAPACK-ABI
// cblas.h forward-declares enums such as CBLAS_TRANSPOSE without an inline
// definition, which is valid C but is rejected by ISO C++ ("ISO C++ forbids
// forward references to 'enum' types"), breaking the build for any C++
// translation unit that includes <Accelerate/Accelerate.h> as-is. Forcing the
// legacy (non-ILP64) CBLAS/LAPACK headers via these macros avoids the
// offending forward declarations; it has no effect on vForce, which is what
// this file actually uses. See: https://forums.swift.org/t/71695 and
// Apple's "Updating Code That Uses the New LAPACK and BLAS APIs" guidance.
//
#define ACCELERATE_NEW_LAPACK 0
#define ACCELERATE_LAPACK_ILP64 0
#include <Accelerate/Accelerate.h>
#endif

//
// Bundles the floating point constants for use by kernels written in assembly.
//

MLAS_INTERNAL_DATA const MLAS_TANH_CONSTANTS MlasTanhConstants = {
    -9.0f,
    9.0f,
    -2.76076847742355e-16f,
    2.00018790482477e-13f,
    -8.60467152213735e-11f,
    5.12229709037114e-08f,
    1.48572235717979e-05f,
    6.37261928875436e-04f,
    4.89352455891786e-03f,
    1.19825839466702e-06f,
    1.18534705686654e-04f,
    2.26843463243900e-03f,
    4.89352518554385e-03f,
};

void
MLASCALL
MlasTanhKernel(
    const float* Input,
    float* Output,
    size_t N
    )
/*++

Routine Description:

    This routine implements the generic kernel for the hyperbolic tangent function.

Arguments:

    Input - Supplies the input buffer.

    Output - Supplies the output buffer.

    N - Supplies the number of elements to process.

Return Value:

    None.

--*/
{
    while (N >= 4) {

        MLAS_FLOAT32X4 Value = MlasLoadFloat32x4(Input);

        Value = MlasMaximumFloat32x4(MlasBroadcastFloat32x4(MlasTanhConstants.LowerRange), Value);
        Value = MlasMinimumFloat32x4(MlasBroadcastFloat32x4(MlasTanhConstants.UpperRange), Value);

        MLAS_FLOAT32X4 ValueSquared = MlasMultiplyFloat32x4(Value, Value);

        MLAS_FLOAT32X4 p;
        p = MlasMultiplyAddFloat32x4(ValueSquared, MlasBroadcastFloat32x4(MlasTanhConstants.alpha_13),
            MlasBroadcastFloat32x4(MlasTanhConstants.alpha_11));
        p = MlasMultiplyAddFloat32x4(p, ValueSquared, MlasBroadcastFloat32x4(MlasTanhConstants.alpha_9));
        p = MlasMultiplyAddFloat32x4(p, ValueSquared, MlasBroadcastFloat32x4(MlasTanhConstants.alpha_7));
        p = MlasMultiplyAddFloat32x4(p, ValueSquared, MlasBroadcastFloat32x4(MlasTanhConstants.alpha_5));
        p = MlasMultiplyAddFloat32x4(p, ValueSquared, MlasBroadcastFloat32x4(MlasTanhConstants.alpha_3));
        p = MlasMultiplyAddFloat32x4(p, ValueSquared, MlasBroadcastFloat32x4(MlasTanhConstants.alpha_1));
        p = MlasMultiplyFloat32x4(p, Value);

        MLAS_FLOAT32X4 q;
        q = MlasMultiplyAddFloat32x4(ValueSquared, MlasBroadcastFloat32x4(MlasTanhConstants.beta_6),
            MlasBroadcastFloat32x4(MlasTanhConstants.beta_4));
        q = MlasMultiplyAddFloat32x4(q, ValueSquared, MlasBroadcastFloat32x4(MlasTanhConstants.beta_2));
        q = MlasMultiplyAddFloat32x4(q, ValueSquared, MlasBroadcastFloat32x4(MlasTanhConstants.beta_0));

        MlasStoreFloat32x4(Output, MlasDivideFloat32x4(p, q));

        Input += 4;
        Output += 4;
        N -= 4;
    }

    while (N > 0) {

        float Value = *Input++;

        // This odd two-step process exists to ensure an input value of NaN carries through
        // without modification because "std::min" and "std::max" return unreliable results
        // when NaNs are involved, and it's clear from the test's reference outputs that
        // they want a NaN on output whenever the input is a NaN.
        float v_tmp;
        v_tmp = (Value < MlasTanhConstants.LowerRange) ? MlasTanhConstants.LowerRange : Value;
        Value = (v_tmp > MlasTanhConstants.UpperRange) ? MlasTanhConstants.UpperRange : v_tmp;

        float ValueSquared = Value * Value;

        float p;
        p = ValueSquared * MlasTanhConstants.alpha_13 + MlasTanhConstants.alpha_11;
        p = p * ValueSquared + MlasTanhConstants.alpha_9;
        p = p * ValueSquared + MlasTanhConstants.alpha_7;
        p = p * ValueSquared + MlasTanhConstants.alpha_5;
        p = p * ValueSquared + MlasTanhConstants.alpha_3;
        p = p * ValueSquared + MlasTanhConstants.alpha_1;
        p = p * Value;

        float q;
        q = ValueSquared * MlasTanhConstants.beta_6 + MlasTanhConstants.beta_4;
        q = q * ValueSquared + MlasTanhConstants.beta_2;
        q = q * ValueSquared + MlasTanhConstants.beta_0;

        *Output++ = (p / q);

        N -= 1;
    }
}

#if defined(MLAS_USE_APPLE_ACCELERATE) && defined(__APPLE__) && defined(MLAS_TARGET_ARM64)

void
MLASCALL
MlasTanhKernelAppleAccelerate(
    const float* Input,
    float* Output,
    size_t N
    )
/*++

Routine Description:

    This routine computes the hyperbolic tangent function using the vForce
    library (part of Apple's Accelerate framework), available on macOS arm64
    when onnxruntime_USE_APPLE_ACCELERATE is enabled.

    vvtanhf computes each output element from only the corresponding input
    element, so this call is safe for in-place use (Input == Output), matching
    the aliasing contract callers already rely on for MlasTanhKernel above
    (see e.g. providers/cpu/tensor/gelu.cc, providers/cpu/ml/svmclassifier.h,
    and the RNN cell helpers in providers/cpu/rnn/rnn_helpers.cc, all of which
    call MlasComputeTanh with the same buffer as both Input and Output). It
    performs a synchronous vectorized computation on the calling thread with
    no internal GCD/thread-pool dispatch of its own, so it does not
    oversubscribe the ORT threadpool.

Arguments:

    Input - Supplies the input buffer.

    Output - Supplies the output buffer.

    N - Supplies the number of elements to process.

Return Value:

    None.

--*/
{
    // vvtanhf takes the element count as a pointer to a signed 32-bit int
    // (a vForce API convention). Chunk defensively so this remains correct
    // even if N exceeds INT32_MAX, though no current MLAS caller passes
    // buffers that large.
    constexpr size_t kMaxChunk = static_cast<size_t>(std::numeric_limits<int>::max());

    while (N > 0) {
        const int Chunk = static_cast<int>(std::min(N, kMaxChunk));
        vvtanhf(Output, Input, &Chunk);
        Input += Chunk;
        Output += Chunk;
        N -= static_cast<size_t>(Chunk);
    }
}

#endif  // MLAS_USE_APPLE_ACCELERATE && __APPLE__ && MLAS_TARGET_ARM64

template <>
void
MLASCALL
MlasComputeTanh<float>(
    const float* Input,
    float* Output,
    size_t N
    )
/*++

Routine Description:

    This routine computes the hyperbolic tangent function.

Arguments:

    Input - Supplies the input buffer.

    Output - Supplies the output buffer.

    N - Supplies the number of elements to process.

Return Value:

    None.

--*/
{
#if defined(MLAS_USE_APPLE_ACCELERATE) && defined(__APPLE__) && defined(MLAS_TARGET_ARM64)
    MlasTanhKernelAppleAccelerate(Input, Output, N);
#elif defined(MLAS_TARGET_AMD64) || defined(MLAS_TARGET_RISCV64) || defined(MLAS_USE_SVE)
    GetMlasPlatform().TanhKernelRoutine(Input, Output, N);
#else
    MlasTanhKernel(Input, Output, N);
#endif
}

template <>
void
MLASCALL
MlasComputeTanh<MLAS_FP16>(
    const MLAS_FP16* Input,
    MLAS_FP16* Output,
    size_t N
) {
    if(GetMlasPlatform().TanhFP16KernelRoutine){
        GetMlasPlatform().TanhFP16KernelRoutine(Input, Output, N);
        return;
    }
    const auto* dispatch = GetMlasPlatform().SoftmaxDispatch;
    if (dispatch == nullptr || dispatch->Tanh_Fp16 == nullptr) {
        MLAS_THROW_EX(std::runtime_error, "Tanh_Fp16 is not supported.");
    }
    dispatch->Tanh_Fp16(Input, Output, N);
}

template <>
void
MLASCALL
MlasComputeSoftcap<float>(
    const float* Input,
    float* Output,
    size_t N,
    float cap
) {
    for (size_t i = 0; i < N; i++) {
        Output[i] = Input[i] / cap;
        Output[i] = std::tanh(Output[i]);
        Output[i] = Output[i] * cap;
    }
}

template <>
void
MLASCALL
MlasComputeSoftcap<MLAS_FP16>(
    const MLAS_FP16* Input,
    MLAS_FP16* Output,
    size_t N,
    MLAS_FP16 cap
) {
    const auto* dispatch = GetMlasPlatform().SoftmaxDispatch;
    if (dispatch == nullptr || dispatch->Softcap_Fp16 == nullptr) {
        MLAS_THROW_EX(std::runtime_error, "Softcap_Fp16 is not supported.");
    }
    dispatch->Softcap_Fp16(Input, Output, N, cap);
}
