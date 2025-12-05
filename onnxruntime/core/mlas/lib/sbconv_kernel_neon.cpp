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

inline void MLASCALL
MlasRowDot(const float* Arow, const float* Brow, float* out, int index, size_t len)
{
    float32x4_t acc4 = MlasBroadcastFloat32x4(0.f);

    size_t i = 0;
    for (; i + 8 <= len; i += 8) {
        float32x4_t a0 = MlasLoadFloat32x4(Arow + i);
        float32x4_t a1 = MlasLoadFloat32x4(Arow + i + 4);
        float32x4_t b0 = MlasLoadFloat32x4(Brow + i);
        float32x4_t b1 = MlasLoadFloat32x4(Brow + i + 4);

        bfloat16x8_t a_bf16 = vcvtq_low_bf16_f32(a0);
        a_bf16 = vcvtq_high_bf16_f32(a_bf16, a1);

        bfloat16x8_t b_bf16 = vcvtq_low_bf16_f32(b0);
        b_bf16 = vcvtq_high_bf16_f32(b_bf16, b1);

        acc4 = vbfdotq_f32(acc4, a_bf16, b_bf16);
    }

    float sum = vaddvq_f32(acc4);

    for (; i < len; i++)
        sum += Arow[i] * Brow[i];

    out[index] = sum;
}

inline void MLASCALL
MlasSBDotRowWise(const float* A, const float* B, size_t len, float* out)
{
    float tmpA[len];
    float tmpB[len];

    for (size_t r = 0; r < 16; r++) {
        for (size_t j = 0; j < len; j++)
            tmpA[j] = A[j * 16 + r];

        for (size_t j = 0; j < len; j++)
            tmpB[j] = B[j * 16 + r];

        MlasRowDot(tmpA, tmpB, out, r, len);
    }
}

void
    MLASCALL
    MlasConvDepthwiseBf16KernelNeon(
        const float* Input,
        const float* Filter,
        float* Output,
        size_t StrideWidth,
        size_t DilationWidth,
        size_t InputStride,
        size_t KernelHeight,
        size_t KernelWidth,
        const float* InputBase,
        size_t InputWidth,
        size_t DilatedInputWidth,
        size_t OutputCountLeftPad,
        size_t OutputCount,
        size_t OutputCountRightPad,
        const float* Bias,
        unsigned KernelFlags
    )
{
    const float32x4_t ZeroVector = MlasBroadcastFloat32x4(0.0f);
    const float32x4_t AccumulateMask = vreinterpretq_f32_s32(MlasBroadcastInt32x4(-(KernelFlags & MLAS_CONV_KERNEL_FLAG_ACCUMULATE_OUTPUT)));
    const float32x4_t BiasMask = vreinterpretq_f32_s32(MlasBroadcastInt32x4(-(KernelFlags & MLAS_CONV_KERNEL_FLAG_BIAS_ADDITION)));
    const float32x4_t ReluMask = vreinterpretq_f32_s32(MlasBroadcastInt32x4(-(KernelFlags & MLAS_CONV_KERNEL_FLAG_RELU_ACTIVATION)));

    const size_t StrideWidthElements = StrideWidth / sizeof(float);
    const size_t DilationWidthElements = DilationWidth / sizeof(float);
    const size_t DilatedInputWidthElements = DilatedInputWidth / sizeof(float);

    MLAS_UNREFERENCED_PARAMETER(InputStride);

    const size_t InputWidthElements = InputWidth / sizeof(float);

    const size_t TotalOutputCount = OutputCountLeftPad + OutputCount + OutputCountRightPad;
    
    const size_t MaxKernelPositions = KernelHeight * KernelWidth;
    float tmpInput[MaxKernelPositions * BlockSize];

    for (size_t output_idx = 0; output_idx < TotalOutputCount; output_idx++) {
        float* output_ptr = &Output[output_idx * BlockSize];

        float32x4_t OldOutput0 = MlasLoadFloat32x4(output_ptr);
        float32x4_t OldOutput1 = MlasLoadFloat32x4(output_ptr + 4);
        float32x4_t OldOutput2 = MlasLoadFloat32x4(output_ptr + 8);
        float32x4_t OldOutput3 = MlasLoadFloat32x4(output_ptr + 12);
        
        for (size_t kernel_pos = 0; kernel_pos < MaxKernelPositions; kernel_pos++) {
            size_t kh = kernel_pos / KernelWidth;
            size_t kw = kernel_pos % KernelWidth;
            const float* input_base = Input + output_idx * StrideWidthElements +
                                      kh * DilatedInputWidthElements + kw * DilationWidthElements;
            const float* row_start = InputBase + kh * DilatedInputWidthElements;
            const float* row_end = row_start + InputWidthElements;

            bool valid = (input_base >= row_start) && (input_base + 15 < row_end);
            const float* safe_ptr = input_base;
            safe_ptr = (input_base < row_start) ? row_start : safe_ptr;
            safe_ptr = (input_base + 15 >= row_end) ? row_start : safe_ptr;
            
            float32x4_t validMask = vreinterpretq_f32_s32(MlasBroadcastInt32x4(valid ? -1 : 0));
            
            float32x4_t loaded0 = MlasLoadFloat32x4(safe_ptr);
            float32x4_t loaded1 = MlasLoadFloat32x4(safe_ptr + 4);
            float32x4_t loaded2 = MlasLoadFloat32x4(safe_ptr + 8);
            float32x4_t loaded3 = MlasLoadFloat32x4(safe_ptr + 12);
            
            float32x4_t data0 = MlasBlendFloat32x4(ZeroVector, loaded0, validMask);
            float32x4_t data1 = MlasBlendFloat32x4(ZeroVector, loaded1, validMask);
            float32x4_t data2 = MlasBlendFloat32x4(ZeroVector, loaded2, validMask);
            float32x4_t data3 = MlasBlendFloat32x4(ZeroVector, loaded3, validMask);
            
            MlasStoreFloat32x4(&tmpInput[kernel_pos * BlockSize], data0);
            MlasStoreFloat32x4(&tmpInput[kernel_pos * BlockSize + 4], data1);
            MlasStoreFloat32x4(&tmpInput[kernel_pos * BlockSize + 8], data2);
            MlasStoreFloat32x4(&tmpInput[kernel_pos * BlockSize + 12], data3);
        }

        MlasSBDotRowWise(tmpInput, Filter, MaxKernelPositions, output_ptr);

        float32x4_t Accumulator0 = MlasLoadFloat32x4(output_ptr);
        float32x4_t Accumulator1 = MlasLoadFloat32x4(output_ptr + 4);
        float32x4_t Accumulator2 = MlasLoadFloat32x4(output_ptr + 8);
        float32x4_t Accumulator3 = MlasLoadFloat32x4(output_ptr + 12);

        Accumulator0 = MlasAddFloat32x4(Accumulator0, MlasAndFloat32x4(OldOutput0, AccumulateMask));
        Accumulator1 = MlasAddFloat32x4(Accumulator1, MlasAndFloat32x4(OldOutput1, AccumulateMask));
        Accumulator2 = MlasAddFloat32x4(Accumulator2, MlasAndFloat32x4(OldOutput2, AccumulateMask));
        Accumulator3 = MlasAddFloat32x4(Accumulator3, MlasAndFloat32x4(OldOutput3, AccumulateMask));

        Accumulator0 = MlasAddFloat32x4(Accumulator0, MlasAndFloat32x4(MlasLoadFloat32x4(Bias), BiasMask));
        Accumulator1 = MlasAddFloat32x4(Accumulator1, MlasAndFloat32x4(MlasLoadFloat32x4(Bias + 4), BiasMask));
        Accumulator2 = MlasAddFloat32x4(Accumulator2, MlasAndFloat32x4(MlasLoadFloat32x4(Bias + 8), BiasMask));
        Accumulator3 = MlasAddFloat32x4(Accumulator3, MlasAndFloat32x4(MlasLoadFloat32x4(Bias + 12), BiasMask));

        float32x4_t Relu0 = MlasMaximumFloat32x4(Accumulator0, ZeroVector);
        float32x4_t Relu1 = MlasMaximumFloat32x4(Accumulator1, ZeroVector);
        float32x4_t Relu2 = MlasMaximumFloat32x4(Accumulator2, ZeroVector);
        float32x4_t Relu3 = MlasMaximumFloat32x4(Accumulator3, ZeroVector);

        Accumulator0 = MlasBlendFloat32x4(Accumulator0, Relu0, ReluMask);
        Accumulator1 = MlasBlendFloat32x4(Accumulator1, Relu1, ReluMask);
        Accumulator2 = MlasBlendFloat32x4(Accumulator2, Relu2, ReluMask);
        Accumulator3 = MlasBlendFloat32x4(Accumulator3, Relu3, ReluMask);

        MlasStoreFloat32x4(output_ptr, Accumulator0);
        MlasStoreFloat32x4(output_ptr + 4, Accumulator1);
        MlasStoreFloat32x4(output_ptr + 8, Accumulator2);
        MlasStoreFloat32x4(output_ptr + 12, Accumulator3);
    }
}

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

    const size_t StrideWidthElements = StrideWidth / sizeof(float);
    const size_t FilterStrideElements = FilterStride / sizeof(float);
    const size_t OutputStrideElements = OutputStride / sizeof(float);

    const float32x4_t ZeroVector = MlasBroadcastFloat32x4(0.0f);
    const float32x4_t ReluMask = vreinterpretq_f32_s32(MlasBroadcastInt32x4(-(KernelFlags & MLAS_CONV_KERNEL_FLAG_RELU_ACTIVATION)));

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

    for (size_t f = 0; f < FilterCount; f++) {
        float* output = Output + f * OutputStrideElements;

        for (size_t output_idx = 0; output_idx < OutputCount; output_idx++) {
            float32x4_t Accumulator0 = MlasLoadFloat32x4(&output[output_idx * BlockSize]);
            float32x4_t Accumulator1 = MlasLoadFloat32x4(&output[output_idx * BlockSize + 4]);
            float32x4_t Accumulator2 = MlasLoadFloat32x4(&output[output_idx * BlockSize + 8]);
            float32x4_t Accumulator3 = MlasLoadFloat32x4(&output[output_idx * BlockSize + 12]);

            float32x4_t Relu0 = MlasMaximumFloat32x4(Accumulator0, ZeroVector);
            float32x4_t Relu1 = MlasMaximumFloat32x4(Accumulator1, ZeroVector);
            float32x4_t Relu2 = MlasMaximumFloat32x4(Accumulator2, ZeroVector);
            float32x4_t Relu3 = MlasMaximumFloat32x4(Accumulator3, ZeroVector);

            Accumulator0 = MlasBlendFloat32x4(Accumulator0, Relu0, ReluMask);
            Accumulator1 = MlasBlendFloat32x4(Accumulator1, Relu1, ReluMask);
            Accumulator2 = MlasBlendFloat32x4(Accumulator2, Relu2, ReluMask);
            Accumulator3 = MlasBlendFloat32x4(Accumulator3, Relu3, ReluMask);

            MlasStoreFloat32x4(&output[output_idx * BlockSize], Accumulator0);
            MlasStoreFloat32x4(&output[output_idx * BlockSize + 4], Accumulator1);
            MlasStoreFloat32x4(&output[output_idx * BlockSize + 8], Accumulator2);
            MlasStoreFloat32x4(&output[output_idx * BlockSize + 12], Accumulator3);
        }
    }
}

#endif  // defined(__aarch64__) && defined(__linux__)
