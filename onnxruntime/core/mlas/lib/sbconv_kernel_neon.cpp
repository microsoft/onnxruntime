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
    const bool AccumulateOutput = KernelFlags & MLAS_CONV_KERNEL_FLAG_ACCUMULATE_OUTPUT;
    const bool BiasAddition = KernelFlags & MLAS_CONV_KERNEL_FLAG_BIAS_ADDITION;
    const bool ReluActivation = KernelFlags & MLAS_CONV_KERNEL_FLAG_RELU_ACTIVATION;

    const size_t StrideWidthElements = StrideWidth / sizeof(float);
    const size_t DilationWidthElements = DilationWidth / sizeof(float);
    const size_t DilatedInputWidthElements = DilatedInputWidth / sizeof(float);
    const size_t InputWidthElements = InputWidth / sizeof(float);
    const size_t TotalOutputCount = OutputCountLeftPad + OutputCount + OutputCountRightPad;
    const size_t KernelSize = KernelHeight * KernelWidth;

    MLAS_UNREFERENCED_PARAMETER(InputStride);

    // Depthwise: 16 independent channels, each doing [TotalOutputCount][KernelSize] x [KernelSize][1]
    // Batch all 16 channels into one MlasSBGemmBatch call
    
    std::vector<float> im2col_buffer(BlockSize * TotalOutputCount * KernelSize);
    std::vector<float> filter_cols(BlockSize * KernelSize);
    std::vector<float> output_buffer(BlockSize * TotalOutputCount);
    
    // Prepare filter columns: transpose [KernelSize][16] -> 16 separate [KernelSize] vectors
    for (size_t c = 0; c < BlockSize; c++) {
        for (size_t k = 0; k < KernelSize; k++) {
            filter_cols[c * KernelSize + k] = Filter[k * BlockSize + c];
        }
    }
    
    // im2col for all channels: [c][out_idx][kpos]
    for (size_t c = 0; c < BlockSize; c++) {
        for (size_t out_idx = 0; out_idx < TotalOutputCount; out_idx++) {
            for (size_t kpos = 0; kpos < KernelSize; kpos++) {
                size_t kh = kpos / KernelWidth;
                size_t kw = kpos % KernelWidth;
                const float* input_ptr = Input + out_idx * StrideWidthElements +
                                          kh * DilatedInputWidthElements + kw * DilationWidthElements + c;
                const float* row_start = InputBase + kh * DilatedInputWidthElements;
                const float* row_end = row_start + InputWidthElements;
                im2col_buffer[c * TotalOutputCount * KernelSize + out_idx * KernelSize + kpos] = 
                    (input_ptr >= row_start && input_ptr < row_end) ? *input_ptr : 0.0f;
            }
        }
    }
    
    // Batched SBGEMM: 16 independent GEMMs, each M=TotalOutputCount, N=1, K=KernelSize
    MLAS_SBGEMM_DATA_PARAMS params[16];
    for (size_t c = 0; c < BlockSize; c++) {
        params[c].A = &im2col_buffer[c * TotalOutputCount * KernelSize];
        params[c].B = &filter_cols[c * KernelSize];
        params[c].C = &output_buffer[c * TotalOutputCount];
        params[c].lda = KernelSize;
        params[c].ldb = 1;
        params[c].ldc = 1;
        params[c].Bias = nullptr;
        params[c].AIsfp32 = true;
        params[c].BIsfp32 = true;
        params[c].ZeroMode = true;
        params[c].OutputProcessor = nullptr;
    }
    MlasSBGemmBatch(TotalOutputCount, 1, KernelSize, BlockSize, params, nullptr);
    
    // Scatter results back to output and apply post-processing
    for (size_t out_idx = 0; out_idx < TotalOutputCount; out_idx++) {
        float* output_ptr = &Output[out_idx * BlockSize];
        for (size_t c = 0; c < BlockSize; c++) {
            float val = output_buffer[c * TotalOutputCount + out_idx];
            if (AccumulateOutput) val += output_ptr[c];
            if (BiasAddition) val += Bias[c];
            if (ReluActivation && val < 0) val = 0;
            output_ptr[c] = val;
        }
    }
}

//
// BF16 NCHW/NCHWc Convolution Kernel using im2col + SBGEMM.
//   NCHW: 1 input channel per kernel position, single GEMM with K=KernelSize
//   NCHWc: BlockSize input channels per kernel position, loop over kpos with K=BlockSize
//
// BF16 NCHW/NCHWc Convolution Kernel using im2col + SBGEMM.
//   NCHW: 1 input channel per kernel position, single GEMM with K=KernelSize
//   NCHWc: BlockSize input channels per kernel position, loop over kpos with K=BlockSize
//
template <bool IsNchwcFormat>
void MLASCALL
MlasConvBf16KernelNeonImpl(
    const float* Input,
    const float* Filter,
    float* Output,
    size_t StrideWidth,
    size_t DilationWidth,
    size_t FilterCount,
    size_t InputStride,
    size_t FilterStride,
    size_t OutputStride,
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
    const bool AccumulateOutput = (KernelFlags & MLAS_CONV_KERNEL_FLAG_ACCUMULATE_OUTPUT) != 0;
    const bool BiasAddition = (KernelFlags & MLAS_CONV_KERNEL_FLAG_BIAS_ADDITION) != 0;
    const bool ReluActivation = (KernelFlags & MLAS_CONV_KERNEL_FLAG_RELU_ACTIVATION) != 0;

    const size_t StrideWidthElements = StrideWidth / sizeof(float);
    const size_t DilationWidthElements = DilationWidth / sizeof(float);
    const size_t FilterStrideElements = FilterStride / sizeof(float);
    const size_t OutputStrideElements = OutputStride / sizeof(float);
    const size_t InputWidthElements = InputWidth / sizeof(float);
    const size_t DilatedInputWidthElements = DilatedInputWidth / sizeof(float);

    MLAS_UNREFERENCED_PARAMETER(InputStride);

    const size_t TotalOutputCount = OutputCountLeftPad + OutputCount + OutputCountRightPad;
    const size_t KernelSize = KernelHeight * KernelWidth;

    std::vector<float> im2col_buffer(TotalOutputCount * (IsNchwcFormat ? BlockSize : KernelSize));

    if (BiasAddition && AccumulateOutput) {
        for (size_t f = 0; f < FilterCount; f++) {
            float* output = Output + f * OutputStrideElements;
            const float32x4_t b0 = MlasLoadFloat32x4(&Bias[f * BlockSize]);
            const float32x4_t b1 = MlasLoadFloat32x4(&Bias[f * BlockSize + 4]);
            const float32x4_t b2 = MlasLoadFloat32x4(&Bias[f * BlockSize + 8]);
            const float32x4_t b3 = MlasLoadFloat32x4(&Bias[f * BlockSize + 12]);
            for (size_t i = 0; i < TotalOutputCount; i++) {
                MlasStoreFloat32x4(&output[i * BlockSize], MlasAddFloat32x4(b0, MlasLoadFloat32x4(&output[i * BlockSize])));
                MlasStoreFloat32x4(&output[i * BlockSize + 4], MlasAddFloat32x4(b1, MlasLoadFloat32x4(&output[i * BlockSize + 4])));
                MlasStoreFloat32x4(&output[i * BlockSize + 8], MlasAddFloat32x4(b2, MlasLoadFloat32x4(&output[i * BlockSize + 8])));
                MlasStoreFloat32x4(&output[i * BlockSize + 12], MlasAddFloat32x4(b3, MlasLoadFloat32x4(&output[i * BlockSize + 12])));
            }
        }
    }

    MLAS_SBGEMM_DATA_PARAMS gemm_params[16];
    const size_t K = IsNchwcFormat ? BlockSize : KernelSize;

    // Helper lambda for im2col extraction at a kernel position
    auto extractIm2Col = [&](size_t kpos, float* col_base, size_t col_stride) {
        size_t kh = kpos / KernelWidth;
        size_t kw = kpos % KernelWidth;
        const float* row_start = InputBase + kh * DilatedInputWidthElements;
        const float* row_end = row_start + InputWidthElements;

        for (size_t out_idx = 0; out_idx < TotalOutputCount; out_idx++) {
            const float* input_base = Input + out_idx * StrideWidthElements +
                                      kh * DilatedInputWidthElements + kw * DilationWidthElements;
            float* col_ptr = col_base + out_idx * col_stride;

            if constexpr (IsNchwcFormat) {
                for (size_t ic = 0; ic < BlockSize; ic++) {
                    const float* ie = input_base + ic;
                    col_ptr[ic] = (ie >= row_start && ie < row_end) ? *ie : 0.0f;
                }
            } else {
                col_ptr[kpos] = (input_base >= row_start && input_base < row_end) ? *input_base : 0.0f;
            }
        }
    };

    // Helper lambda to setup GEMM params
    auto setupGemmParams = [&](size_t filter_offset, bool zeroMode) {
        size_t idx = 0;
        for (size_t f = 0; f < FilterCount; f++) {
            gemm_params[idx].A = im2col_buffer.data();
            gemm_params[idx].B = Filter + f * FilterStrideElements + filter_offset;
            gemm_params[idx].C = Output + f * OutputStrideElements;
            gemm_params[idx].lda = K;
            gemm_params[idx].ldb = BlockSize;
            gemm_params[idx].ldc = BlockSize;
            gemm_params[idx].Bias = BiasAddition ? (Bias + f * BlockSize) : nullptr;
            gemm_params[idx].AIsfp32 = true;
            gemm_params[idx].BIsfp32 = true;
            gemm_params[idx].ZeroMode = zeroMode;
            gemm_params[idx].OutputProcessor = nullptr;
            idx++;
        }
        return idx;
    };

    const size_t numGemmCalls = IsNchwcFormat ? KernelSize : 1;
    for (size_t g = 0; g < numGemmCalls; g++) {
        if constexpr (IsNchwcFormat) {
            extractIm2Col(g, im2col_buffer.data(), BlockSize);
        } else {
            for (size_t kpos = 0; kpos < KernelSize; kpos++) {
                extractIm2Col(kpos, im2col_buffer.data(), KernelSize);
            }
        }
        size_t kh = g / KernelWidth, kw = g % KernelWidth;
        size_t filter_offset = IsNchwcFormat ? kh * (KernelWidth * BlockSize * BlockSize) + kw * (BlockSize * BlockSize) : 0;
        size_t idx = setupGemmParams(filter_offset, (g == 0) && !AccumulateOutput);
        MlasSBGemmBatch(TotalOutputCount, BlockSize, K, idx, gemm_params, nullptr);
    }

    if (ReluActivation) {
        const float32x4_t ZeroVector = MlasBroadcastFloat32x4(0.0f);
        for (size_t f = 0; f < FilterCount; f++) {
            float* output = Output + f * OutputStrideElements;
            for (size_t i = 0; i < TotalOutputCount; i++) {
                MlasStoreFloat32x4(&output[i * BlockSize], MlasMaximumFloat32x4(MlasLoadFloat32x4(&output[i * BlockSize]), ZeroVector));
                MlasStoreFloat32x4(&output[i * BlockSize + 4], MlasMaximumFloat32x4(MlasLoadFloat32x4(&output[i * BlockSize + 4]), ZeroVector));
                MlasStoreFloat32x4(&output[i * BlockSize + 8], MlasMaximumFloat32x4(MlasLoadFloat32x4(&output[i * BlockSize + 8]), ZeroVector));
                MlasStoreFloat32x4(&output[i * BlockSize + 12], MlasMaximumFloat32x4(MlasLoadFloat32x4(&output[i * BlockSize + 12]), ZeroVector));
            }
        }
    }
}

void MLASCALL MlasConvNchwcBf16KernelNeon(
    const float* Input, const float* Filter, float* Output,
    size_t StrideWidth, size_t DilationWidth, size_t FilterCount,
    size_t InputStride, size_t FilterStride, size_t OutputStride,
    size_t KernelHeight, size_t KernelWidth, const float* InputBase,
    size_t InputWidth, size_t DilatedInputWidth,
    size_t OutputCountLeftPad, size_t OutputCount, size_t OutputCountRightPad,
    const float* Bias, unsigned KernelFlags)
{
    MlasConvBf16KernelNeonImpl<true>(Input, Filter, Output, StrideWidth, DilationWidth,
        FilterCount, InputStride, FilterStride, OutputStride, KernelHeight, KernelWidth,
        InputBase, InputWidth, DilatedInputWidth, OutputCountLeftPad, OutputCount,
        OutputCountRightPad, Bias, KernelFlags);
}

void MLASCALL MlasConvNchwBf16KernelNeon(
    const float* Input, const float* Filter, float* Output,
    size_t StrideWidth, size_t DilationWidth, size_t FilterCount,
    size_t InputStride, size_t FilterStride, size_t OutputStride,
    size_t KernelHeight, size_t KernelWidth, const float* InputBase,
    size_t InputWidth, size_t DilatedInputWidth,
    size_t OutputCountLeftPad, size_t OutputCount, size_t OutputCountRightPad,
    const float* Bias, unsigned KernelFlags)
{
    MlasConvBf16KernelNeonImpl<false>(Input, Filter, Output, StrideWidth, DilationWidth,
        FilterCount, InputStride, FilterStride, OutputStride, KernelHeight, KernelWidth,
        InputBase, InputWidth, DilatedInputWidth, OutputCountLeftPad, OutputCount,
        OutputCountRightPad, Bias, KernelFlags);
}

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

#endif  // defined(__aarch64__) && defined(__linux__)
