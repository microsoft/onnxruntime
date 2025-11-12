/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    sbconv_kernel_neon.cpp

Abstract:

    This module implements bfloat16 precision convolution kernels for ARM NEON.

--*/

#if defined(__aarch64__) && defined(__linux__)

#include <vector>

#include "arm_neon.h"
#include "mlasi.h"
#include "sconv.h"

constexpr size_t BlockSize = MLAS_PLATFORM::MLAS_NEON_NCHWC_BLOCK_SIZE;

//
// BF16 Pointwise Convolution Kernel
//
void MLASCALL
MlasConvPointwiseBf16KernelNeon(
    const float* Input,
    const float* Filter,
    float* Output,
    size_t StrideWidth,
    size_t InputChannels, /* numChannels/BlockSize = 16/16 = 1 */
    size_t FilterCount,
    size_t /*InputStride*/,
    size_t FilterStride,
    size_t OutputStride,
    size_t OutputCount,
    const float* Bias,
    unsigned KernelFlags
)
{
    const bool BiasAddition = (KernelFlags & MLAS_CONV_KERNEL_FLAG_BIAS_ADDITION) != 0;
    const bool ReluActivation = (KernelFlags & MLAS_CONV_KERNEL_FLAG_RELU_ACTIVATION) != 0;

    const size_t StrideWidthElements = StrideWidth / sizeof(float);
    const size_t FilterStrideElements = FilterStride / sizeof(float);
    const size_t OutputStrideElements = OutputStride / sizeof(float);

    std::vector<MLAS_SBGEMM_DATA_PARAMS> gemm_params(FilterCount);

    for (size_t f = 0; f < FilterCount; f++) {
        const float* filter = Filter + f * FilterStrideElements;
        float* output = Output + f * OutputStrideElements;

        gemm_params[f].A = Input;
        gemm_params[f].B = filter;
        gemm_params[f].C = output;
        gemm_params[f].lda = StrideWidthElements;
        gemm_params[f].ldb = BlockSize;
        gemm_params[f].ldc = BlockSize;
        gemm_params[f].Bias = BiasAddition ? (Bias + f * BlockSize) : nullptr;
        gemm_params[f].AIsfp32 = true;
        gemm_params[f].BIsfp32 = true;
        gemm_params[f].OutputProcessor = nullptr;
    }

    MlasSBGemmBatch(OutputCount, BlockSize, InputChannels * BlockSize, FilterCount, gemm_params.data(), nullptr);

    if (ReluActivation) {
        const float32x4_t ZeroVector = MlasBroadcastFloat32x4(0.0f);

        for (size_t f = 0; f < FilterCount; f++) {
            float* output = Output + f * OutputStrideElements;

            for (size_t output_idx = 0; output_idx < OutputCount; output_idx++) {
                float32x4_t Accumulator0 = MlasLoadFloat32x4(&output[output_idx * BlockSize]);
                float32x4_t Accumulator1 = MlasLoadFloat32x4(&output[output_idx * BlockSize + 4]);
                float32x4_t Accumulator2 = MlasLoadFloat32x4(&output[output_idx * BlockSize + 8]);
                float32x4_t Accumulator3 = MlasLoadFloat32x4(&output[output_idx * BlockSize + 12]);

                Accumulator0 = MlasMaximumFloat32x4(Accumulator0, ZeroVector);
                Accumulator1 = MlasMaximumFloat32x4(Accumulator1, ZeroVector);
                Accumulator2 = MlasMaximumFloat32x4(Accumulator2, ZeroVector);
                Accumulator3 = MlasMaximumFloat32x4(Accumulator3, ZeroVector);

                MlasStoreFloat32x4(&output[output_idx * BlockSize], Accumulator0);
                MlasStoreFloat32x4(&output[output_idx * BlockSize + 4], Accumulator1);
                MlasStoreFloat32x4(&output[output_idx * BlockSize + 8], Accumulator2);
                MlasStoreFloat32x4(&output[output_idx * BlockSize + 12], Accumulator3);
            }
        }
    }
}

#endif  // defined(__aarch64__) && defined(__linux__)
