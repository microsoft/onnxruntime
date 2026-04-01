/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    gelu.cpp

Abstract:

    This module implements routines to compute the exact Gelu function.

--*/

#include "mlasi.h"

namespace {

constexpr float kInvSqrt2 = 0.70710678118654752440f;

}  // namespace

void
MLASCALL
MlasGeluErfKernel(
    const float* Input,
    float* Output,
    size_t N
    )
{
    // This kernel is not buffer alias safe because it is implemented in
    // multiple passes: first scale Input into Output, then apply erf in place,
    // and finally combine that intermediate with the original Input values.
    // Callers must guarantee that Input and Output do not overlap (see mlas.h for aliasing requirements).
    for (size_t i = 0; i < N; ++i) {
        Output[i] = Input[i] * kInvSqrt2;
    }

    MlasComputeErf(Output, Output, N);

    for (size_t i = 0; i < N; ++i) {
        Output[i] = 0.5f * Input[i] * (Output[i] + 1.0f);
    }
}

void
MLASCALL
MlasComputeGeluErf(
    const float* Input,
    float* Output,
    size_t N
    )
{
#if defined(MLAS_TARGET_AMD64)
    // TODO: Add an intermediate fused AVX2/FMA3 GELU(erf) path on AMD64.
    // Today the dispatch jumps from the generic multi-pass implementation to
    // AVX512F, so non-AVX512 x64 machines fall back to the generic kernel.
    GetMlasPlatform().GeluErfKernelRoutine(Input, Output, N);
#else
    MlasGeluErfKernel(Input, Output, N);
#endif
}

void
MLASCALL
MlasComputeFP16Gelu(const MLAS_FP16* input,
                    MLAS_FP16* output,
                    MLAS_FP16* temp,
                    size_t count,
                    MLAS_GELU_ALGORITHM algo)
{
    if(GetMlasPlatform().GeluFP16KernelRoutine){
        GetMlasPlatform().GeluFP16KernelRoutine(input, output, temp, count, algo);
        return;
    }
    MLAS_UNREFERENCED_PARAMETER(temp); // 'temp' is only used by vectorized kernel implementations and it is unused in the scalar fallback path.
    for (size_t i = 0; i < count; ++i) {
        float x = static_cast<float>(input[i]);
        float gelu_val;

        if (algo == MlasGeluTanh) {
            // GELU approximation (tanh)
            const float B = 0.7978845608f;
            const float C = 0.044715f * B;
            float tanh_arg = x * (B + C * x * x);
            float tanh_res = std::tanh(tanh_arg);
            gelu_val = 0.5f * x * (1.0f + tanh_res);
        } else {
            // GELU exact (erf)
            gelu_val = 0.5f * x *
                (1.0f + std::erf(x * static_cast<float>(M_SQRT1_2)));
        }

        output[i] = MLAS_FP16(gelu_val);
    }
}