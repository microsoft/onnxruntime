/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    sconv_depthwise_avx512f.cpp

Abstract:

    This module implements a sliding window AVX-512 NCHWc depthwise
    convolution kernel (block size 16) for stride 1, dilation 1 and kernel
    widths 3, 5 and 7.

    MlasConvDepthwiseFloatKernelAvx512F keeps six output accumulators and
    reloads the input for every kernel tap, and evaluates outputs touching
    the left/right padding one at a time. This kernel instead loads each
    input column of a kernel row once into a register and reuses it for
    every kernel column, keeps more accumulators in flight than the FMA
    latency x throughput product, and handles padding columns with masks.

    The per output arithmetic is identical to the assembly kernel:

        acc = +0.0
        for each effective kernel row, then each kernel column, ascending:
            if the input column is inside the row:
                acc = vfmadd231ps(acc, Filter, Input)
        if ACCUMULATE: acc = vaddps(acc, Output)
        if BIAS:       acc = vaddps(acc, Bias)
        if RELU:       acc = vmaxps(0, acc)

    Taps that fall into the padding are skipped (merge masked FMA), not
    multiplied by zero, so the result is bitwise identical, including for
    non-finite filter values.

--*/

#include <immintrin.h>

#include "mlasi.h"
#include "snchwc.h"

namespace {

constexpr size_t BlockSize = 16;

//
// Outputs per tile: enough independent accumulators to cover the FMA latency,
// while the accumulators plus the OutputCount + KernelWidth - 1 input columns
// of a kernel row still fit in the 32 vector registers.
//

template <size_t KernelWidth>
constexpr size_t MaximumOutputCount = (KernelWidth <= 3) ? 12 : (KernelWidth <= 5) ? 10 : 8;

#if defined(__clang__)
#define MLAS_DW_UNROLL _Pragma("clang loop unroll(full)")
#elif defined(__GNUC__)
#define MLAS_DW_UNROLL _Pragma("GCC unroll 16")
#else
#define MLAS_DW_UNROLL
#endif

struct MLAS_DW_SLIDING_ARGS {
    const float* Filter;
    size_t KernelHeight;
    size_t InputRowStride;      // floats between kernel rows
    size_t InputColumns;        // valid input columns per row
    const float* Bias;
    unsigned KernelFlags;
};

//
// Computes OutputCount consecutive outputs. Input addresses the input column
// read by the first kernel column of the first output; Column is its column
// index within the row (negative inside the left padding).
//

template <size_t KernelWidth, size_t OutputCount, bool Masked>
MLAS_FORCEINLINE
void
DepthwiseTile(
    const MLAS_DW_SLIDING_ARGS& Args,
    const float* Input,
    ptrdiff_t Column,
    float* Output
    )
{
    constexpr size_t InputCount = OutputCount + KernelWidth - 1;

    __m512 Accumulator[OutputCount];

    MLAS_DW_UNROLL
    for (size_t o = 0; o < OutputCount; o++) {
        Accumulator[o] = _mm512_setzero_ps();
    }

    __mmask16 Valid[InputCount];

    if constexpr (Masked) {
        MLAS_DW_UNROLL
        for (size_t j = 0; j < InputCount; j++) {
            Valid[j] = (size_t(Column + ptrdiff_t(j)) < Args.InputColumns) ? __mmask16(0xFFFF) : __mmask16(0);
        }
    }

    const float* filter = Args.Filter;

    for (size_t kh = 0; kh < Args.KernelHeight; kh++) {

        __m512 InputVector[InputCount];

        MLAS_DW_UNROLL
        for (size_t j = 0; j < InputCount; j++) {
            if constexpr (Masked) {
                InputVector[j] = _mm512_maskz_loadu_ps(Valid[j], Input + j * BlockSize);
            } else {
                InputVector[j] = _mm512_loadu_ps(Input + j * BlockSize);
            }
        }

        MLAS_DW_UNROLL
        for (size_t kw = 0; kw < KernelWidth; kw++) {

            const __m512 FilterVector = _mm512_loadu_ps(filter + kw * BlockSize);

            MLAS_DW_UNROLL
            for (size_t o = 0; o < OutputCount; o++) {
                if constexpr (Masked) {
                    Accumulator[o] = _mm512_mask3_fmadd_ps(FilterVector, InputVector[o + kw],
                        Accumulator[o], Valid[o + kw]);
                } else {
                    Accumulator[o] = _mm512_fmadd_ps(FilterVector, InputVector[o + kw], Accumulator[o]);
                }
            }
        }

        Input += Args.InputRowStride;
        filter += KernelWidth * BlockSize;
    }

    const unsigned KernelFlags = Args.KernelFlags;

    if ((KernelFlags & MLAS_CONV_KERNEL_FLAG_ACCUMULATE_OUTPUT) != 0) {
        MLAS_DW_UNROLL
        for (size_t o = 0; o < OutputCount; o++) {
            Accumulator[o] = _mm512_add_ps(Accumulator[o], _mm512_loadu_ps(Output + o * BlockSize));
        }
    }

    if ((KernelFlags & MLAS_CONV_KERNEL_FLAG_BIAS_ADDITION) != 0) {
        const __m512 BiasVector = _mm512_loadu_ps(Args.Bias);
        MLAS_DW_UNROLL
        for (size_t o = 0; o < OutputCount; o++) {
            Accumulator[o] = _mm512_add_ps(Accumulator[o], BiasVector);
        }
    }

    if ((KernelFlags & MLAS_CONV_KERNEL_FLAG_RELU_ACTIVATION) != 0) {
        const __m512 ZeroVector = _mm512_setzero_ps();
        MLAS_DW_UNROLL
        for (size_t o = 0; o < OutputCount; o++) {
            Accumulator[o] = _mm512_max_ps(ZeroVector, Accumulator[o]);
        }
    }

    MLAS_DW_UNROLL
    for (size_t o = 0; o < OutputCount; o++) {
        _mm512_storeu_ps(Output + o * BlockSize, Accumulator[o]);
    }
}

template <size_t KernelWidth, bool Masked, size_t Count = MaximumOutputCount<KernelWidth> - 1>
MLAS_FORCEINLINE
void
DepthwiseTileRemainder(
    const MLAS_DW_SLIDING_ARGS& Args,
    const float* Input,
    ptrdiff_t Column,
    float* Output,
    size_t OutputCount
    )
{
    if constexpr (Count > 0) {
        if (OutputCount == Count) {
            DepthwiseTile<KernelWidth, Count, Masked>(Args, Input, Column, Output);
        } else {
            DepthwiseTileRemainder<KernelWidth, Masked, Count - 1>(Args, Input, Column, Output, OutputCount);
        }
    }
}

template <size_t KernelWidth>
void
DepthwiseRow(
    const MLAS_DW_SLIDING_ARGS& Args,
    const float* Input,
    ptrdiff_t Column,
    float* Output,
    size_t OutputCount
    )
{
    const ptrdiff_t InputColumns = ptrdiff_t(Args.InputColumns);

    while (OutputCount > 0) {

        constexpr size_t MaximumCount = MaximumOutputCount<KernelWidth>;

        const size_t Count = std::min(OutputCount, MaximumCount);
        const bool Masked = Column < 0 || Column + ptrdiff_t(Count + KernelWidth - 1) > InputColumns;

        if (Count == MaximumCount) {
            if (Masked) {
                DepthwiseTile<KernelWidth, MaximumCount, true>(Args, Input, Column, Output);
            } else {
                DepthwiseTile<KernelWidth, MaximumCount, false>(Args, Input, Column, Output);
            }
        } else {
            if (Masked) {
                DepthwiseTileRemainder<KernelWidth, true>(Args, Input, Column, Output, Count);
            } else {
                DepthwiseTileRemainder<KernelWidth, false>(Args, Input, Column, Output, Count);
            }
        }

        Input += Count * BlockSize;
        Column += ptrdiff_t(Count);
        Output += Count * BlockSize;
        OutputCount -= Count;
    }
}

}  // namespace

void
MLASCALL
MlasConvDepthwiseFloatKernelAvx512FSliding(
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
/*++

Routine Description:

    This routine is a drop in replacement for
    MlasConvDepthwiseFloatKernelAvx512F (same arguments, bitwise identical
    results). Unsupported geometries are forwarded to it.

--*/
{
    constexpr size_t BlockBytes = BlockSize * sizeof(float);

    if (StrideWidth != BlockBytes || DilationWidth != BlockBytes ||
        (KernelWidth != 3 && KernelWidth != 5 && KernelWidth != 7) ||
        (DilatedInputWidth % BlockBytes) != 0 || (InputWidth % BlockBytes) != 0) {
        MlasConvDepthwiseFloatKernelAvx512F(Input, Filter, Output, StrideWidth, DilationWidth,
            InputStride, KernelHeight, KernelWidth, InputBase, InputWidth, DilatedInputWidth,
            OutputCountLeftPad, OutputCount, OutputCountRightPad, Bias, KernelFlags);
        return;
    }

    MLAS_DW_SLIDING_ARGS Args;
    Args.Filter = Filter;
    Args.KernelHeight = KernelHeight;
    Args.InputRowStride = DilatedInputWidth / sizeof(float);
    Args.InputColumns = InputWidth / BlockBytes;
    Args.Bias = Bias;
    Args.KernelFlags = KernelFlags;

    const ptrdiff_t Column = (Input - InputBase) / ptrdiff_t(BlockSize);
    const size_t TotalOutputCount = OutputCountLeftPad + OutputCount + OutputCountRightPad;

    switch (KernelWidth) {
        case 3: DepthwiseRow<3>(Args, Input, Column, Output, TotalOutputCount); break;
        case 5: DepthwiseRow<5>(Args, Input, Column, Output, TotalOutputCount); break;
        case 7: DepthwiseRow<7>(Args, Input, Column, Output, TotalOutputCount); break;
        default: break;
    }
}
