/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    dwconv.cpp

Abstract:

    This module implements the half precision floating point depthwise convolution routines.

--*/


#include "fp16_common.h"

#ifdef MLAS_F16VEC_INTRINSICS_SUPPORTED

MLAS_FORCEINLINE
void
MlasConvDepthwiseKernel(
    const _mlas_fp16_* const* Input,
    const _mlas_fp16_* Filter,
    _mlas_fp16_* Output,
    size_t Channels,
    size_t OutputCount,
    size_t KernelSize,
    MLAS_HALF_GEMM_POSTPROCESSOR* PostProc
    )
{
    while (OutputCount > 0) {
        size_t ChannelOffset = 0;
        size_t c = Channels;

        while (c >= 8) {
            MLAS_FLOAT16X8 Accumulator = MlasZeroFloat16x8();
            size_t ChannelKernelOffset = ChannelOffset;

            for (size_t k = 0; k < KernelSize; k++) {
                MLAS_FLOAT16X8 InputVector = MlasLoadFloat16x8(&Input[k][ChannelOffset]);
                MLAS_FLOAT16X8 FilterVector = MlasLoadFloat16x8(&Filter[ChannelKernelOffset]);

                Accumulator = MlasMultiplyAddFloat16x8(InputVector, FilterVector, Accumulator);
                ChannelKernelOffset += Channels;
            }
            MlasStoreFloat16x8(Output, Accumulator);
            Output += 8;

            ChannelOffset += 8;
            c -= 8;
        }

        if (c >= 4) {
            MLAS_FLOAT16X4 Accumulator = MlasZeroFloat16x4();
            size_t ChannelKernelOffset = ChannelOffset;

            for (size_t k = 0; k < KernelSize; k++) {
                MLAS_FLOAT16X4 InputVector = MlasLoadFloat16x4(&Input[k][ChannelOffset]);
                MLAS_FLOAT16X4 FilterVector = MlasLoadFloat16x4(&Filter[ChannelKernelOffset]);

                Accumulator = MlasMultiplyAddFloat16x4(InputVector, FilterVector, Accumulator);
                ChannelKernelOffset += Channels;
            }
            MlasStoreFloat16x4(Output, Accumulator);
            Output += 4;

            ChannelOffset += 4;
            c -= 4;
        }

        if (c > 0) {
            MLAS_FLOAT16X4 Accumulator = MlasZeroFloat16x4();
            size_t ChannelKernelOffset = ChannelOffset;

            for (size_t k = 0; k < KernelSize; k++) {
                MLAS_FLOAT16X4 InputValue = MlasLoadFloat16x4(&Input[k][ChannelOffset]);
                MLAS_FLOAT16X4 FilterValue = MlasLoadFloat16x4(&Filter[ChannelKernelOffset]);

                Accumulator = MlasMultiplyAddFloat16x4(InputValue, FilterValue, Accumulator);
                ChannelKernelOffset += Channels;
            }
            MlasStorePartialFloat16x4(Output, Accumulator, c);
            Output += c;
        }
        if (PostProc) {
            PostProc->Process(reinterpret_cast<MLAS_FP16*>(Output - Channels), 0, 0, 1, Channels,
                              Channels);
        }
        Input += KernelSize;
        OutputCount -= 1;
    }
}

#else

MLAS_FORCEINLINE
void
MlasConvDepthwiseKernel(
    const _mlas_fp16_* const* Input,
    const _mlas_fp16_* Filter,
    _mlas_fp16_* Output,
    size_t Channels,
    size_t OutputCount,
    size_t KernelSize,
    MLAS_HALF_GEMM_POSTPROCESSOR* PostProc
    )
{
    while (OutputCount > 0) {
        for (size_t ChannelOffset = 0; ChannelOffset < Channels; ChannelOffset++) {
            float Accumulator = 0.0f;
            size_t ChannelKernelOffset = ChannelOffset;

            for (size_t k = 0; k < KernelSize; k++) {
                Accumulator += MLAS_Half2Float(Input[k][ChannelOffset]) * MLAS_Half2Float(Filter[ChannelKernelOffset]);
                ChannelKernelOffset += Channels;
            }
            *Output++ = MLAS_Float2Half(Accumulator);
        }
        if (PostProc) {
            PostProc->Process(reinterpret_cast<MLAS_FP16*>(Output - Channels), 0, 0, 1, Channels,
                              Channels);
        }
        Input += KernelSize;
        OutputCount -= 1;
    }
}

#endif // MLAS_F16VEC_INTRINSICS_SUPPORTED


void
MLASCALL
MlasConvDepthwise(
    const MLAS_FP16* const* Input,
    const MLAS_FP16* Filter,
    MLAS_FP16* Output,
    size_t Channels,
    size_t OutputCount,
    size_t KernelSize,
    MLAS_HALF_GEMM_POSTPROCESSOR* PostProc
    )
{
    MlasConvDepthwiseKernel(
        reinterpret_cast<const _mlas_fp16_* const*>(Input),
        reinterpret_cast<const _mlas_fp16_*>(Filter),
        reinterpret_cast<_mlas_fp16_*>(Output),
        Channels,
        OutputCount,
        KernelSize,
        PostProc);
}
