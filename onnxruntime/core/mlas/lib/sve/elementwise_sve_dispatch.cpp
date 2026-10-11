/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    elementwise_sve_dispatch.cpp

Abstract:

    This module contains the public entry points of the SVE elementwise
    kernels. It is plain C++ with no SVE intrinsics, so it compiles with any
    toolchain (including MSVC, which has no SVE support); the vector loops
    are provided by extern "C" *Impl functions that come either from the SVE
    intrinsics translation units (GCC/Clang builds) or from the portable
    machine-code assembly variant (elementwise_sve_asm.S) — both export the
    same symbols, and the build links exactly one of them.

    This layer owns everything that does not belong in assembly: scalar
    fast paths, constant-table plumbing, and the multi-pass orchestration of
    Gelu.

--*/

#include "mlasi_sve.h"

#include <cmath>

//
// Vector-loop implementations (intrinsics TUs or elementwise_sve_asm.S).
//

extern "C" {

void MLASCALL
MlasSveErfKernelImpl(const float* Input, float* Output, size_t N, const MLAS_ERF_CONSTANTS* Constants);

void MLASCALL
MlasSveLogisticKernelImpl(const float* Input, float* Output, size_t N, const MLAS_LOGISTIC_CONSTANTS* Constants);

void
MLASCALL
MlasSveTanhKernelImpl(const float* Input, float* Output, size_t N, const MLAS_TANH_CONSTANTS* Constants);

void MLASCALL
MlasSveComputeExpF32KernelImpl(const float* Input, float* Output, size_t N, const MLAS_SVE_EXP_CONSTANTS* Constants);

float MLASCALL
MlasSveComputeSumExpF32KernelImpl(const float* Input, float* Output, size_t N, const float* NegativeMaximum, const MLAS_EXP_CONSTANTS* Constants, const MLAS_SVE_EXP_CONSTANTS* AclConstants);

float MLASCALL
MlasSveReduceMaximumF32KernelImpl(const float* Input, size_t N, float MinimumValue);

void MLASCALL
MlasSveReduceMinimumMaximumF32KernelImpl(const float* Input, float* Min, float* Max, size_t N);

void MLASCALL
MlasSveComputeSoftmaxOutputF32KernelImpl(float* Output, size_t N, const float* Parameters);

void MLASCALL
MlasSveComputeLogSoftmaxOutputF32KernelImpl(const float* Input, float* Output, size_t N, const float* Parameters);

#if defined(MLAS_F16VEC_INTRINSICS_SUPPORTED)

void MLASCALL
MlasSveTanhFP16KernelImpl(const MLAS_FP16* Input, MLAS_FP16* Output, size_t N);

void MLASCALL
MlasSveErfFP16KernelImpl(const MLAS_FP16* Input, MLAS_FP16* Output, size_t N);

void MLASCALL
MlasSveGeluTanhArgFP16KernelImpl(const MLAS_FP16* Input, MLAS_FP16* Temp, size_t N);

void MLASCALL
MlasSveGeluScaleFP16KernelImpl(const MLAS_FP16* Input, MLAS_FP16* Temp, size_t N);

void MLASCALL
MlasSveGeluCombineFP16KernelImpl(const MLAS_FP16* Input, const MLAS_FP16* Inner, MLAS_FP16* Output, size_t N);

#endif

}  // extern "C"

//
// Constants of the ARM Compute Library exp() approximation (see
// elementwise_sve.cpp for the algorithm), as float bit patterns.
//

extern "C" const MLAS_SVE_EXP_CONSTANTS MlasSveExpConstants = {
    0x3f7ffff6,  // x^1: 0x1.ffffecp-1f
    0x3efffedb,  // x^2: 0x1.fffdb6p-2f
    0x3e2aaf33,  // x^3: 0x1.555e66p-3f
    0x3d2b9f17,  // x^4: 0x1.573e2ep-5f
    0x3c072010,  // x^5: 0x1.0e4020p-7f
    0x4b00007f,  // 2^23 + 127
    0x3fb8aa3b,  // 1 / ln(2)
    0xbf317200,  // -ln(2), bits -1..-19
    0xb5bfbe8e,  // -ln(2), bits -20..-42
    0x7f800000,  // +infinity
    0x42b0bd71,  // 88.37f, approximately ln(2^127.5)
    0xc2ad47ae,  // -86.64f, approximately ln(2^-125)
    0x48481fc0,  // 0x1.903f8p17, FEXPA rounding shift with embedded exponent bias
    0x3f000000,  // 0.5
    0xc2b00000,  // -88.0, FEXPA-safe SumExp lower clamp (> -127*ln2)
};

//
// Float32 entry points.
//

void
MLASCALL
MlasSveErfKernel(
    const float* Input,
    float* Output,
    size_t N
    )
{
    MlasSveErfKernelImpl(Input, Output, N, &MlasErfConstants);
}

void
MLASCALL
MlasSveLogisticKernel(
    const float* Input,
    float* Output,
    size_t N
    )
{
    MlasSveLogisticKernelImpl(Input, Output, N, &MlasLogisticConstants);
}

void
MLASCALL
MlasSveTanhKernel(
    const float* Input,
    float* Output,
    size_t N
    )
{
    MlasSveTanhKernelImpl(Input, Output, N, &MlasTanhConstants);
}

void
MLASCALL
MlasSveComputeExpF32Kernel(
    const float* Input,
    float* Output,
    size_t N
    )
{
    if (N == 1) {
        Output[0] = expf(Input[0]);
        return;
    }

    MlasSveComputeExpF32KernelImpl(Input, Output, N, &MlasSveExpConstants);
}

float
MLASCALL
MlasSveComputeSumExpF32Kernel(
    const float* Input,
    float* Output,
    size_t N,
    const float* NegativeMaximum
    )
{
    if (N == 1) {
        float result = expf(Input[0] + *NegativeMaximum);
        if (Output != nullptr) {
            Output[0] = result;
        }
        return result;
    }

    return MlasSveComputeSumExpF32KernelImpl(Input, Output, N, NegativeMaximum, &MlasExpConstants, &MlasSveExpConstants);
}

float
MLASCALL
MlasSveReduceMaximumF32Kernel(
    const float* Input,
    size_t N
    )
{
    return MlasSveReduceMaximumF32KernelImpl(Input, N, MlasMinimumF32Value);
}

void
MLASCALL
MlasSveReduceMinimumMaximumF32Kernel(
    const float* Input,
    float* Min,
    float* Max,
    size_t N
    )
{
    MlasSveReduceMinimumMaximumF32KernelImpl(Input, Min, Max, N);
}

void
MLASCALL
MlasSveComputeSoftmaxOutputF32Kernel(
    float* Output,
    size_t N,
    const float* Parameters
    )
{
    MlasSveComputeSoftmaxOutputF32KernelImpl(Output, N, Parameters);
}

void
MLASCALL
MlasSveComputeLogSoftmaxOutputF32Kernel(
    const float* Input,
    float* Output,
    size_t N,
    const float* Parameters
    )
{
    MlasSveComputeLogSoftmaxOutputF32KernelImpl(Input, Output, N, Parameters);
}

//
// Float16 entry points.
//

#if defined(MLAS_F16VEC_INTRINSICS_SUPPORTED)

void
MLASCALL
MlasSveTanhFP16Kernel(
    const MLAS_FP16* Input,
    MLAS_FP16* Output,
    size_t N
    )
{
    MlasSveTanhFP16KernelImpl(Input, Output, N);
}

void
MLASCALL
MlasSveErfFP16Kernel(
    const MLAS_FP16* Input,
    MLAS_FP16* Output,
    size_t N
    )
{
    MlasSveErfFP16KernelImpl(Input, Output, N);
}

void
MLASCALL
MlasSveGeluFP16Kernel(
    const MLAS_FP16* Input,
    MLAS_FP16* Output,
    MLAS_FP16* Temp,
    size_t N,
    MLAS_GELU_ALGORITHM Algo
    )
{
    if (Algo == MlasGeluTanh) {
        MlasSveGeluTanhArgFP16KernelImpl(Input, Temp, N);
        MlasSveTanhFP16KernelImpl(Temp, Temp, N);
    } else {
        MlasSveGeluScaleFP16KernelImpl(Input, Temp, N);
        MlasSveErfFP16KernelImpl(Temp, Temp, N);
    }

    MlasSveGeluCombineFP16KernelImpl(Input, Temp, Output, N);
}

#endif  // MLAS_F16VEC_INTRINSICS_SUPPORTED
