/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    sbconv_kernel_neon.cpp

Abstract:

    This module implements bfloat16 precision convolution kernels for ARM NEON.

--*/

#if defined(MLAS_USE_ARM_NEON_NCHWC) && defined(__linux__)

#include "mlasi.h"
#include "sconv_nchwc_kernel_neon.h"

constexpr size_t BlockSize = MLAS_PLATFORM::MLAS_NEON_NCHWC_BLOCK_SIZE;

//
// BF16 Pointwise (1x1) Convolution Kernel using SBGEMM.
//
void MLASCALL
MlasConvPointwiseBf16KernelNeon(
    const float* Input,
    const float* Filter,
    float* Output,
    size_t StrideWidth,
    size_t InputChannels,
    size_t FilterCount,
    size_t InputStride,
    size_t FilterStride,
    size_t OutputStride,
    size_t OutputCount,
    const float* Bias,
    unsigned KernelFlags
)
{
    const bool AccumulateOutput = (KernelFlags & MLAS_CONV_KERNEL_FLAG_ACCUMULATE_OUTPUT) != 0;
    const bool BiasAddition = (KernelFlags & MLAS_CONV_KERNEL_FLAG_BIAS_ADDITION) != 0;
    const bool ReluActivation = (KernelFlags & MLAS_CONV_KERNEL_FLAG_RELU_ACTIVATION) != 0;

    const size_t StrideWidthElements = StrideWidth / sizeof(float);
    const size_t InputStrideElements = InputStride / sizeof(float);
    const size_t FilterStrideElements = FilterStride / sizeof(float);
    const size_t OutputStrideElements = OutputStride / sizeof(float);

    // SBGEMM only adds bias when ZeroMode=true. When accumulating (ZeroMode=false),
    // pre-add bias to existing output before the GEMM operations.
    if (BiasAddition && AccumulateOutput) {
        for (size_t f = 0; f < FilterCount; f++) {
            float* output = Output + f * OutputStrideElements;
            const float32x4_t b0 = MlasLoadFloat32x4(&Bias[f * BlockSize]);
            const float32x4_t b1 = MlasLoadFloat32x4(&Bias[f * BlockSize + 4]);
            const float32x4_t b2 = MlasLoadFloat32x4(&Bias[f * BlockSize + 8]);
            const float32x4_t b3 = MlasLoadFloat32x4(&Bias[f * BlockSize + 12]);
            for (size_t i = 0; i < OutputCount; i++) {
                MlasStoreFloat32x4(&output[i * BlockSize], MlasAddFloat32x4(b0, MlasLoadFloat32x4(&output[i * BlockSize])));
                MlasStoreFloat32x4(&output[i * BlockSize + 4], MlasAddFloat32x4(b1, MlasLoadFloat32x4(&output[i * BlockSize + 4])));
                MlasStoreFloat32x4(&output[i * BlockSize + 8], MlasAddFloat32x4(b2, MlasLoadFloat32x4(&output[i * BlockSize + 8])));
                MlasStoreFloat32x4(&output[i * BlockSize + 12], MlasAddFloat32x4(b3, MlasLoadFloat32x4(&output[i * BlockSize + 12])));
            }
        }
    }

    // Build SBGEMM params for all (filter, input_channel) combinations.
    // FilterCount <= 4, InputChannels <= 8, so max 32 elements.
    // Bias is set on all elements but SBGEMM only uses it when ZeroMode=true.
    MLAS_SBGEMM_DATA_PARAMS gemm_params[32];

    size_t idx = 0;
    for (size_t f = 0; f < FilterCount; f++) {
        const float* filter = Filter + f * FilterStrideElements;
        float* output = Output + f * OutputStrideElements;
        for (size_t ic = 0; ic < InputChannels; ic++, idx++) {
            gemm_params[idx].A = Input + ic * InputStrideElements;
            gemm_params[idx].B = filter + ic * BlockSize * BlockSize;
            gemm_params[idx].C = output;
            gemm_params[idx].lda = StrideWidthElements;
            gemm_params[idx].ldb = BlockSize;
            gemm_params[idx].ldc = BlockSize;
            gemm_params[idx].Bias = BiasAddition ? (Bias + f * BlockSize) : nullptr;
            gemm_params[idx].AIsfp32 = true;
            gemm_params[idx].BIsfp32 = true;
            gemm_params[idx].ZeroMode = (ic == 0) && !AccumulateOutput;
            gemm_params[idx].OutputProcessor = nullptr;
        }
    }

    MlasSBGemmBatch(OutputCount, BlockSize, BlockSize, idx, gemm_params, nullptr);

    if (ReluActivation) {
        const float32x4_t ZeroVector = MlasBroadcastFloat32x4(0.0f);
        for (size_t f = 0; f < FilterCount; f++) {
            float* output = Output + f * OutputStrideElements;
            for (size_t i = 0; i < OutputCount; i++) {
                MlasStoreFloat32x4(&output[i * BlockSize], MlasMaximumFloat32x4(MlasLoadFloat32x4(&output[i * BlockSize]), ZeroVector));
                MlasStoreFloat32x4(&output[i * BlockSize + 4], MlasMaximumFloat32x4(MlasLoadFloat32x4(&output[i * BlockSize + 4]), ZeroVector));
                MlasStoreFloat32x4(&output[i * BlockSize + 8], MlasMaximumFloat32x4(MlasLoadFloat32x4(&output[i * BlockSize + 8]), ZeroVector));
                MlasStoreFloat32x4(&output[i * BlockSize + 12], MlasMaximumFloat32x4(MlasLoadFloat32x4(&output[i * BlockSize + 12]), ZeroVector));
            }
        }
    }
}

#endif
