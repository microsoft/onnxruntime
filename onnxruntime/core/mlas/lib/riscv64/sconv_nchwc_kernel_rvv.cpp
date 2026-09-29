/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    sconv_nchwc_kernel_rvv.cpp

Abstract:

    This module implements RVV kernels for the single precision NCHWc
    convolution operations on riscv64: direct NCHW, direct NCHWc,
    depthwise, and pointwise convolution.

    BlockSize is fixed at 16, matching the ARM64 NEON NCHWc layout, so a channel
    block is held in one register group and the multiplier that group needs
    follows the vector length: four registers at 128 bits, two at 256, one at 512.

--*/

#include "mlasi.h"

#if defined(MLAS_USE_RVV)

#include <riscv_vector.h>
#include <cassert>
#include <limits>

#define MLAS_CONV_KERNEL_FLAG_ACCUMULATE_OUTPUT     0x00000001
#define MLAS_CONV_KERNEL_FLAG_BIAS_ADDITION         0x00000002
#define MLAS_CONV_KERNEL_FLAG_RELU_ACTIVATION       0x00000004
#define MLAS_CONV_KERNEL_FLAG_OTHER_ACTIVATION      0x00000008

namespace {

constexpr size_t BlockSize = 16;

//
// A channel block has to fit in one register group and the group should be no
// wider: the cost of a vector instruction follows the multiplier it was issued
// with, not the number of elements asked for. Which multiplier that is depends on
// the vector length, so the kernels are written once against these accessors.
//

template <size_t Lmul>
struct MLAS_RVV_BLOCK;

#define MLAS_RVV_BLOCK_FOR_LMUL(Lmul, Suffix)                                                  \
    template <>                                                                                \
    struct MLAS_RVV_BLOCK<Lmul> {                                                              \
        using Vector = vfloat32##Suffix##_t;                                                   \
        static MLAS_FORCEINLINE size_t VectorLength()                                          \
            { return __riscv_vsetvl_e32##Suffix(BlockSize); }                                  \
        static MLAS_FORCEINLINE Vector Broadcast(float Value, size_t vl)                       \
            { return __riscv_vfmv_v_f_f32##Suffix(Value, vl); }                                \
        static MLAS_FORCEINLINE Vector Load(const float* Address, size_t vl)                   \
            { return __riscv_vle32_v_f32##Suffix(Address, vl); }                               \
        static MLAS_FORCEINLINE void Store(float* Address, Vector Value, size_t vl)            \
            { __riscv_vse32_v_f32##Suffix(Address, Value, vl); }                               \
        static MLAS_FORCEINLINE Vector MultiplyAdd(Vector Accumulator, float Multiplicand,     \
                                                   Vector Multiplier, size_t vl)               \
            { return __riscv_vfmacc_vf_f32##Suffix(Accumulator, Multiplicand, Multiplier, vl); }\
        static MLAS_FORCEINLINE Vector MultiplyAdd(Vector Accumulator, Vector Multiplicand,    \
                                                   Vector Multiplier, size_t vl)               \
            { return __riscv_vfmacc_vv_f32##Suffix(Accumulator, Multiplicand, Multiplier, vl); }\
        static MLAS_FORCEINLINE Vector Add(Vector First, Vector Second, size_t vl)             \
            { return __riscv_vfadd_vv_f32##Suffix(First, Second, vl); }                        \
        static MLAS_FORCEINLINE Vector Maximum(Vector First, Vector Second, size_t vl)         \
            { return __riscv_vfmax_vv_f32##Suffix(First, Second, vl); }                        \
        static MLAS_FORCEINLINE Vector Divide(Vector Dividend, Vector Divisor, size_t vl)      \
            { return __riscv_vfdiv_vv_f32##Suffix(Dividend, Divisor, vl); }                    \
    }

MLAS_RVV_BLOCK_FOR_LMUL(1, m1);
MLAS_RVV_BLOCK_FOR_LMUL(2, m2);
MLAS_RVV_BLOCK_FOR_LMUL(4, m4);

#undef MLAS_RVV_BLOCK_FOR_LMUL

//
// Smallest multiplier whose register group still holds a block.
//

MLAS_FORCEINLINE
size_t
MlasConvNchwcLmulRvv(
    void
    )
{
    const size_t BlockBytes = BlockSize * sizeof(float);
    const size_t RegisterBytes = __riscv_vlenb();

    if (RegisterBytes >= BlockBytes) {
        return 1;
    }

    if (2 * RegisterBytes >= BlockBytes) {
        return 2;
    }

    return 4;
}

//
// Enters Body at the multiplier this part wants. A kernel is entered once per
// output row, so asking each time costs a control status register read.
//

#define MLAS_RVV_DISPATCH_ON_LMUL(Body, ...)     \
    switch (MlasConvNchwcLmulRvv()) {            \
        case 1:                                  \
            Body<1>(__VA_ARGS__);                \
            break;                               \
        case 2:                                  \
            Body<2>(__VA_ARGS__);                \
            break;                               \
        default:                                 \
            Body<4>(__VA_ARGS__);                \
            break;                               \
    }

//
// Number of output positions a kernel accumulates at once.
//
// A block of channels fills a register group, so no parallelism is left inside
// one: the multiply accumulates for an output position form a chain. It comes
// from the output positions instead, and they share one filter load. Four of
// them plus that filter occupy twenty of the thirty two registers at LMUL 4.
//
// A register group cannot be held in an array, so the positions are spelled out,
// and a group holding fewer than four repeats the first position in the slots it
// does not use. Those slots are not stored, which lets one body serve both a
// full group and the remainder.
//
constexpr size_t OutputStep = 4;

template <size_t Lmul>
MLAS_FORCEINLINE
void
ApplyPostProcessing(
    typename MLAS_RVV_BLOCK<Lmul>::Vector& acc,
    const float* Output,
    const float* Bias,
    unsigned KernelFlags,
    size_t vl
    )
{
    using Block = MLAS_RVV_BLOCK<Lmul>;
    using Vector = typename Block::Vector;

    if (KernelFlags & MLAS_CONV_KERNEL_FLAG_ACCUMULATE_OUTPUT) {
        Vector old_output = Block::Load(Output, vl);
        acc = Block::Add(acc, old_output, vl);
    }

    if (KernelFlags & MLAS_CONV_KERNEL_FLAG_BIAS_ADDITION) {
        assert(Bias != nullptr);
        Vector bias_vec = Block::Load(Bias, vl);
        acc = Block::Add(acc, bias_vec, vl);
    }

    if (KernelFlags & MLAS_CONV_KERNEL_FLAG_RELU_ACTIVATION) {
        Vector zero = Block::Broadcast(0.0f, vl);
        acc = Block::Maximum(acc, zero, vl);
    }
}

MLAS_FORCEINLINE
float
LoadInRange(
    const float* Address,
    const float* RowStart,
    const float* RowEnd
    )
{
    return (Address >= RowStart && Address < RowEnd) ? *Address : 0.0f;
}

//
// Accumulates a group of output positions of the direct NCHW convolution. Input
// and Output address the first of them; the filter is already at its block.
//

template <size_t Lmul>
MLAS_FORCEINLINE
void
MlasConvNchwFloatGroupRvv(
    const float* Input,
    const float* Filter,
    float* Output,
    size_t StrideWidthElements,
    size_t DilationWidthElements,
    size_t InputWidthElements,
    size_t DilatedInputWidthElements,
    size_t KernelHeight,
    size_t KernelWidth,
    const float* InputBase,
    const float* Bias,
    unsigned KernelFlags,
    size_t vl,
    size_t OutputCountThisIteration
    )
{
    using Block = MLAS_RVV_BLOCK<Lmul>;
    using Vector = typename Block::Vector;

    const size_t offset1 = (OutputCountThisIteration > 1 ? 1 : 0) * StrideWidthElements;
    const size_t offset2 = (OutputCountThisIteration > 2 ? 2 : 0) * StrideWidthElements;
    const size_t offset3 = (OutputCountThisIteration > 3 ? 3 : 0) * StrideWidthElements;

    const size_t offset_last = (OutputCountThisIteration - 1) * StrideWidthElements;

    Vector acc0 = Block::Broadcast(0.0f, vl);
    Vector acc1 = acc0;
    Vector acc2 = acc0;
    Vector acc3 = acc0;

    for (size_t kh = 0; kh < KernelHeight; kh++) {

        const float* input_row_start = InputBase + kh * DilatedInputWidthElements;
        const float* input_row_end = input_row_start + InputWidthElements;

        for (size_t kw = 0; kw < KernelWidth; kw++) {

            const float* input_pos =
                Input + kh * DilatedInputWidthElements + kw * DilationWidthElements;

            Vector filt =
                Block::Load(&Filter[(kh * KernelWidth + kw) * BlockSize], vl);

            if (input_pos >= input_row_start && (input_pos + offset_last) < input_row_end) {
                acc0 = Block::MultiplyAdd(acc0, input_pos[0], filt, vl);
                acc1 = Block::MultiplyAdd(acc1, input_pos[offset1], filt, vl);
                acc2 = Block::MultiplyAdd(acc2, input_pos[offset2], filt, vl);
                acc3 = Block::MultiplyAdd(acc3, input_pos[offset3], filt, vl);
            } else {
                acc0 = Block::MultiplyAdd(
                    acc0, LoadInRange(input_pos, input_row_start, input_row_end), filt, vl);
                acc1 = Block::MultiplyAdd(
                    acc1, LoadInRange(input_pos + offset1, input_row_start, input_row_end),
                    filt, vl);
                acc2 = Block::MultiplyAdd(
                    acc2, LoadInRange(input_pos + offset2, input_row_start, input_row_end),
                    filt, vl);
                acc3 = Block::MultiplyAdd(
                    acc3, LoadInRange(input_pos + offset3, input_row_start, input_row_end),
                    filt, vl);
            }
        }
    }

    ApplyPostProcessing<Lmul>(acc0, &Output[0], Bias, KernelFlags, vl);
    Block::Store(&Output[0], acc0, vl);

    if (OutputCountThisIteration > 1) {
        ApplyPostProcessing<Lmul>(acc1, &Output[BlockSize], Bias, KernelFlags, vl);
        Block::Store(&Output[BlockSize], acc1, vl);
    }

    if (OutputCountThisIteration > 2) {
        ApplyPostProcessing<Lmul>(acc2, &Output[2 * BlockSize], Bias, KernelFlags, vl);
        Block::Store(&Output[2 * BlockSize], acc2, vl);
    }

    if (OutputCountThisIteration > 3) {
        ApplyPostProcessing<Lmul>(acc3, &Output[3 * BlockSize], Bias, KernelFlags, vl);
        Block::Store(&Output[3 * BlockSize], acc3, vl);
    }
}

//
// Direct NCHW convolution kernel.
//
// Input is in NCHW format (single channel per kernel position).
// Filter is laid out as [KH][KW][BlockSize] — one scalar input is
// broadcast and multiplied with BlockSize filter values.
// Output is in NCHWc (BlockSize channels interleaved).
//

template <size_t Lmul>
void
MlasConvNchwFloatKernelRvvImpl(
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
    MLAS_UNREFERENCED_PARAMETER(InputStride);

    using Block = MLAS_RVV_BLOCK<Lmul>;

    const size_t vl = Block::VectorLength();
    const size_t StrideWidthElements = StrideWidth / sizeof(float);
    const size_t DilationWidthElements = DilationWidth / sizeof(float);
    const size_t FilterStrideElements = FilterStride / sizeof(float);
    const size_t OutputStrideElements = OutputStride / sizeof(float);
    const size_t InputWidthElements = InputWidth / sizeof(float);
    const size_t DilatedInputWidthElements = DilatedInputWidth / sizeof(float);

    const size_t TotalOutputCount = OutputCountLeftPad + OutputCount + OutputCountRightPad;

    for (size_t filterSetBlock = 0; filterSetBlock < FilterCount; filterSetBlock++) {

        const float* filter = Filter + filterSetBlock * FilterStrideElements;
        float* output = Output + filterSetBlock * OutputStrideElements;
        const float* bias = (Bias != nullptr) ? &Bias[filterSetBlock * BlockSize] : nullptr;

        size_t output_idx = 0;

        for (; output_idx + OutputStep <= TotalOutputCount; output_idx += OutputStep) {
            MlasConvNchwFloatGroupRvv<Lmul>(
                Input + output_idx * StrideWidthElements, filter, &output[output_idx * BlockSize],
                StrideWidthElements, DilationWidthElements, InputWidthElements,
                DilatedInputWidthElements, KernelHeight, KernelWidth, InputBase, bias,
                KernelFlags, vl, OutputStep);
        }

        if (output_idx < TotalOutputCount) {
            MlasConvNchwFloatGroupRvv<Lmul>(
                Input + output_idx * StrideWidthElements, filter, &output[output_idx * BlockSize],
                StrideWidthElements, DilationWidthElements, InputWidthElements,
                DilatedInputWidthElements, KernelHeight, KernelWidth, InputBase, bias,
                KernelFlags, vl, TotalOutputCount - output_idx);
        }
    }
}

}  // namespace

void
MLASCALL
MlasConvNchwFloatKernelRvv(
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
    MLAS_RVV_DISPATCH_ON_LMUL(MlasConvNchwFloatKernelRvvImpl, Input, Filter, Output, StrideWidth,
                              DilationWidth, FilterCount, InputStride, FilterStride, OutputStride,
                              KernelHeight, KernelWidth, InputBase, InputWidth, DilatedInputWidth,
                              OutputCountLeftPad, OutputCount, OutputCountRightPad, Bias,
                              KernelFlags)
}

namespace {

//
// Accumulates a group of output positions of the direct NCHWc convolution. Input
// and Output address the first of them; the filter is already at its block.
//

template <size_t Lmul>
MLAS_FORCEINLINE
void
MlasConvNchwcFloatGroupRvv(
    const float* Input,
    const float* Filter,
    float* Output,
    size_t StrideWidthElements,
    size_t DilationWidthElements,
    size_t InputWidthElements,
    size_t DilatedInputWidthElements,
    size_t KernelHeight,
    size_t KernelWidth,
    const float* InputBase,
    const float* Bias,
    unsigned KernelFlags,
    size_t vl,
    size_t OutputCountThisIteration
    )
{
    using Block = MLAS_RVV_BLOCK<Lmul>;
    using Vector = typename Block::Vector;

    const size_t offset1 = (OutputCountThisIteration > 1 ? 1 : 0) * StrideWidthElements;
    const size_t offset2 = (OutputCountThisIteration > 2 ? 2 : 0) * StrideWidthElements;
    const size_t offset3 = (OutputCountThisIteration > 3 ? 3 : 0) * StrideWidthElements;

    // Furthest position the group stores; the slots it does not use sit at zero.
    const size_t offset_last = (OutputCountThisIteration - 1) * StrideWidthElements;

    Vector acc0 = Block::Broadcast(0.0f, vl);
    Vector acc1 = acc0;
    Vector acc2 = acc0;
    Vector acc3 = acc0;

    for (size_t kh = 0; kh < KernelHeight; kh++) {

        const float* input_row_start = InputBase + kh * DilatedInputWidthElements;
        const float* input_row_end = input_row_start + InputWidthElements;

        for (size_t kw = 0; kw < KernelWidth; kw++) {

            const float* filter_position =
                Filter + (kh * KernelWidth + kw) * BlockSize * BlockSize;

            const float* input_base =
                Input + kh * DilatedInputWidthElements + kw * DilationWidthElements;
            const float* input_base0 = input_base;
            const float* input_base1 = input_base + offset1;
            const float* input_base2 = input_base + offset2;
            const float* input_base3 = input_base + offset3;

            const bool all_in_bounds = (input_base >= input_row_start) &&
                                     ((input_base + offset_last + BlockSize) <= input_row_end);

            if (all_in_bounds) {
                for (size_t ic = 0; ic < BlockSize; ic++) {
                    Vector filt =
                        Block::Load(&filter_position[ic * BlockSize], vl);
                    acc0 = Block::MultiplyAdd(acc0, input_base0[ic], filt, vl);
                    acc1 = Block::MultiplyAdd(acc1, input_base1[ic], filt, vl);
                    acc2 = Block::MultiplyAdd(acc2, input_base2[ic], filt, vl);
                    acc3 = Block::MultiplyAdd(acc3, input_base3[ic], filt, vl);
                }
            } else {
                for (size_t ic = 0; ic < BlockSize; ic++) {
                    Vector filt =
                        Block::Load(&filter_position[ic * BlockSize], vl);
                    acc0 = Block::MultiplyAdd(
                        acc0, LoadInRange(input_base0 + ic, input_row_start, input_row_end),
                        filt, vl);
                    acc1 = Block::MultiplyAdd(
                        acc1, LoadInRange(input_base1 + ic, input_row_start, input_row_end),
                        filt, vl);
                    acc2 = Block::MultiplyAdd(
                        acc2, LoadInRange(input_base2 + ic, input_row_start, input_row_end),
                        filt, vl);
                    acc3 = Block::MultiplyAdd(
                        acc3, LoadInRange(input_base3 + ic, input_row_start, input_row_end),
                        filt, vl);
                }
            }
        }
    }

    ApplyPostProcessing<Lmul>(acc0, &Output[0], Bias, KernelFlags, vl);
    Block::Store(&Output[0], acc0, vl);

    if (OutputCountThisIteration > 1) {
        ApplyPostProcessing<Lmul>(acc1, &Output[BlockSize], Bias, KernelFlags, vl);
        Block::Store(&Output[BlockSize], acc1, vl);
    }

    if (OutputCountThisIteration > 2) {
        ApplyPostProcessing<Lmul>(acc2, &Output[2 * BlockSize], Bias, KernelFlags, vl);
        Block::Store(&Output[2 * BlockSize], acc2, vl);
    }

    if (OutputCountThisIteration > 3) {
        ApplyPostProcessing<Lmul>(acc3, &Output[3 * BlockSize], Bias, KernelFlags, vl);
        Block::Store(&Output[3 * BlockSize], acc3, vl);
    }
}

//
// Direct NCHWc convolution kernel.
//
// Input is in NCHWc format (BlockSize channels interleaved per spatial position).
// Filter layout: [KH][KW][BlockSize_in][BlockSize_out].
// For each kernel position and input channel, one input scalar is broadcast
// and multiplied with BlockSize output filter values.
//

template <size_t Lmul>
void
MlasConvNchwcFloatKernelRvvImpl(
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
    MLAS_UNREFERENCED_PARAMETER(InputStride);

    using Block = MLAS_RVV_BLOCK<Lmul>;

    const size_t vl = Block::VectorLength();
    const size_t StrideWidthElements = StrideWidth / sizeof(float);
    const size_t DilationWidthElements = DilationWidth / sizeof(float);
    const size_t FilterStrideElements = FilterStride / sizeof(float);
    const size_t OutputStrideElements = OutputStride / sizeof(float);
    const size_t InputWidthElements = InputWidth / sizeof(float);
    const size_t DilatedInputWidthElements = DilatedInputWidth / sizeof(float);

    const size_t TotalOutputCount = OutputCountLeftPad + OutputCount + OutputCountRightPad;

    for (size_t filterSetBlock = 0; filterSetBlock < FilterCount; filterSetBlock++) {

        const float* filter = Filter + filterSetBlock * FilterStrideElements;
        float* output = Output + filterSetBlock * OutputStrideElements;
        const float* bias = (Bias != nullptr) ? &Bias[filterSetBlock * BlockSize] : nullptr;

        size_t output_idx = 0;

        for (; output_idx + OutputStep <= TotalOutputCount; output_idx += OutputStep) {
            MlasConvNchwcFloatGroupRvv<Lmul>(
                Input + output_idx * StrideWidthElements, filter, &output[output_idx * BlockSize],
                StrideWidthElements, DilationWidthElements, InputWidthElements,
                DilatedInputWidthElements, KernelHeight, KernelWidth, InputBase, bias,
                KernelFlags, vl, OutputStep);
        }

        if (output_idx < TotalOutputCount) {
            MlasConvNchwcFloatGroupRvv<Lmul>(
                Input + output_idx * StrideWidthElements, filter, &output[output_idx * BlockSize],
                StrideWidthElements, DilationWidthElements, InputWidthElements,
                DilatedInputWidthElements, KernelHeight, KernelWidth, InputBase, bias,
                KernelFlags, vl, TotalOutputCount - output_idx);
        }
    }
}

}  // namespace

void
MLASCALL
MlasConvNchwcFloatKernelRvv(
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
    MLAS_RVV_DISPATCH_ON_LMUL(MlasConvNchwcFloatKernelRvvImpl, Input, Filter, Output, StrideWidth,
                              DilationWidth, FilterCount, InputStride, FilterStride, OutputStride,
                              KernelHeight, KernelWidth, InputBase, InputWidth, DilatedInputWidth,
                              OutputCountLeftPad, OutputCount, OutputCountRightPad, Bias,
                              KernelFlags)
}

namespace {

template <size_t Lmul>
MLAS_FORCEINLINE
typename MLAS_RVV_BLOCK<Lmul>::Vector
LoadBlockInRange(
    const float* Address,
    const float* RowStart,
    const float* RowEnd,
    size_t vl
    )
{
    using Block = MLAS_RVV_BLOCK<Lmul>;

    if (Address >= RowStart && (Address + BlockSize - 1) < RowEnd) {
        return Block::Load(Address, vl);
    }

    return Block::Broadcast(0.0f, vl);
}

//
// Accumulates a group of output positions of the depthwise convolution.
//

template <size_t Lmul>
MLAS_FORCEINLINE
void
MlasConvDepthwiseFloatGroupRvv(
    const float* Input,
    const float* Filter,
    float* Output,
    size_t StrideWidthElements,
    size_t DilationWidthElements,
    size_t InputWidthElements,
    size_t DilatedInputWidthElements,
    size_t KernelHeight,
    size_t KernelWidth,
    const float* InputBase,
    const float* Bias,
    unsigned KernelFlags,
    size_t vl,
    size_t OutputCountThisIteration
    )
{
    using Block = MLAS_RVV_BLOCK<Lmul>;
    using Vector = typename Block::Vector;

    const size_t offset1 = (OutputCountThisIteration > 1 ? 1 : 0) * StrideWidthElements;
    const size_t offset2 = (OutputCountThisIteration > 2 ? 2 : 0) * StrideWidthElements;
    const size_t offset3 = (OutputCountThisIteration > 3 ? 3 : 0) * StrideWidthElements;

    const size_t offset_last = (OutputCountThisIteration - 1) * StrideWidthElements;

    Vector acc0 = Block::Broadcast(0.0f, vl);
    Vector acc1 = acc0;
    Vector acc2 = acc0;
    Vector acc3 = acc0;

    for (size_t kh = 0; kh < KernelHeight; kh++) {

        const float* input_row_start = InputBase + kh * DilatedInputWidthElements;
        const float* input_row_end = input_row_start + InputWidthElements;

        for (size_t kw = 0; kw < KernelWidth; kw++) {

            const float* input_base =
                Input + kh * DilatedInputWidthElements + kw * DilationWidthElements;

            const float* input_base0 = input_base;
            const float* input_base1 = input_base + offset1;
            const float* input_base2 = input_base + offset2;
            const float* input_base3 = input_base + offset3;

            Vector filt = Block::Load(&Filter[(kh * KernelWidth + kw) * BlockSize], vl);

            if (input_base0 >= input_row_start &&
                (input_base + offset_last + BlockSize - 1) < input_row_end) {
                acc0 = Block::MultiplyAdd(acc0, Block::Load(input_base0, vl), filt, vl);
                acc1 = Block::MultiplyAdd(acc1, Block::Load(input_base1, vl), filt, vl);
                acc2 = Block::MultiplyAdd(acc2, Block::Load(input_base2, vl), filt, vl);
                acc3 = Block::MultiplyAdd(acc3, Block::Load(input_base3, vl), filt, vl);
            } else {
                acc0 = Block::MultiplyAdd(
                    acc0, LoadBlockInRange<Lmul>(input_base0, input_row_start, input_row_end, vl), filt, vl);
                acc1 = Block::MultiplyAdd(
                    acc1, LoadBlockInRange<Lmul>(input_base1, input_row_start, input_row_end, vl), filt, vl);
                acc2 = Block::MultiplyAdd(
                    acc2, LoadBlockInRange<Lmul>(input_base2, input_row_start, input_row_end, vl), filt, vl);
                acc3 = Block::MultiplyAdd(
                    acc3, LoadBlockInRange<Lmul>(input_base3, input_row_start, input_row_end, vl), filt, vl);
            }
        }
    }

    ApplyPostProcessing<Lmul>(acc0, &Output[0], Bias, KernelFlags, vl);
    Block::Store(&Output[0], acc0, vl);

    if (OutputCountThisIteration > 1) {
        ApplyPostProcessing<Lmul>(acc1, &Output[BlockSize], Bias, KernelFlags, vl);
        Block::Store(&Output[BlockSize], acc1, vl);
    }

    if (OutputCountThisIteration > 2) {
        ApplyPostProcessing<Lmul>(acc2, &Output[2 * BlockSize], Bias, KernelFlags, vl);
        Block::Store(&Output[2 * BlockSize], acc2, vl);
    }

    if (OutputCountThisIteration > 3) {
        ApplyPostProcessing<Lmul>(acc3, &Output[3 * BlockSize], Bias, KernelFlags, vl);
        Block::Store(&Output[3 * BlockSize], acc3, vl);
    }
}

//
// Depthwise NCHWc convolution kernel.
//
// Each channel is convolved with its own filter (element-wise).
// Input is NCHWc, filter is [KH][KW][BlockSize].
//

template <size_t Lmul>
void
MlasConvDepthwiseFloatKernelRvvImpl(
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
    MLAS_UNREFERENCED_PARAMETER(InputStride);

    using Block = MLAS_RVV_BLOCK<Lmul>;

    const size_t vl = Block::VectorLength();
    const size_t StrideWidthElements = StrideWidth / sizeof(float);
    const size_t DilationWidthElements = DilationWidth / sizeof(float);
    const size_t InputWidthElements = InputWidth / sizeof(float);
    const size_t DilatedInputWidthElements = DilatedInputWidth / sizeof(float);

    const size_t TotalOutputCount = OutputCountLeftPad + OutputCount + OutputCountRightPad;

    size_t output_idx = 0;

    for (; output_idx + OutputStep <= TotalOutputCount; output_idx += OutputStep) {
        MlasConvDepthwiseFloatGroupRvv<Lmul>(
            Input + output_idx * StrideWidthElements, Filter, &Output[output_idx * BlockSize],
            StrideWidthElements, DilationWidthElements, InputWidthElements,
            DilatedInputWidthElements, KernelHeight, KernelWidth, InputBase, Bias, KernelFlags,
            vl, OutputStep);
    }

    if (output_idx < TotalOutputCount) {
        MlasConvDepthwiseFloatGroupRvv<Lmul>(
            Input + output_idx * StrideWidthElements, Filter, &Output[output_idx * BlockSize],
            StrideWidthElements, DilationWidthElements, InputWidthElements,
            DilatedInputWidthElements, KernelHeight, KernelWidth, InputBase, Bias, KernelFlags,
            vl, TotalOutputCount - output_idx);
    }
}

}  // namespace

void
MLASCALL
MlasConvDepthwiseFloatKernelRvv(
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
    MLAS_RVV_DISPATCH_ON_LMUL(MlasConvDepthwiseFloatKernelRvvImpl, Input, Filter, Output,
                              StrideWidth, DilationWidth, InputStride, KernelHeight, KernelWidth,
                              InputBase, InputWidth, DilatedInputWidth, OutputCountLeftPad,
                              OutputCount, OutputCountRightPad, Bias, KernelFlags)
}

namespace {

//
// Accumulates a group of output positions of the pointwise convolution. Nothing
// is padded here, so no position needs a range test.
//

template <size_t Lmul>
MLAS_FORCEINLINE
void
MlasConvPointwiseFloatGroupRvv(
    const float* Input,
    const float* Filter,
    float* Output,
    size_t StrideWidthElements,
    size_t InputStrideElements,
    size_t InputChannels,
    const float* Bias,
    unsigned KernelFlags,
    size_t vl,
    size_t OutputCountThisIteration
    )
{
    using Block = MLAS_RVV_BLOCK<Lmul>;
    using Vector = typename Block::Vector;

    const size_t offset1 = (OutputCountThisIteration > 1 ? 1 : 0) * StrideWidthElements;
    const size_t offset2 = (OutputCountThisIteration > 2 ? 2 : 0) * StrideWidthElements;
    const size_t offset3 = (OutputCountThisIteration > 3 ? 3 : 0) * StrideWidthElements;

    Vector acc0 = Block::Broadcast(0.0f, vl);
    Vector acc1 = acc0;
    Vector acc2 = acc0;
    Vector acc3 = acc0;

    for (size_t ic = 0; ic < InputChannels; ic++) {

        const float* input = Input + ic * InputStrideElements;
        const float* filter = Filter + ic * BlockSize * BlockSize;

        for (size_t j = 0; j < BlockSize; j++) {
            Vector filt = Block::Load(&filter[j * BlockSize], vl);
            acc0 = Block::MultiplyAdd(acc0, input[j], filt, vl);
            acc1 = Block::MultiplyAdd(acc1, input[offset1 + j], filt, vl);
            acc2 = Block::MultiplyAdd(acc2, input[offset2 + j], filt, vl);
            acc3 = Block::MultiplyAdd(acc3, input[offset3 + j], filt, vl);
        }
    }

    ApplyPostProcessing<Lmul>(acc0, &Output[0], Bias, KernelFlags, vl);
    Block::Store(&Output[0], acc0, vl);

    if (OutputCountThisIteration > 1) {
        ApplyPostProcessing<Lmul>(acc1, &Output[BlockSize], Bias, KernelFlags, vl);
        Block::Store(&Output[BlockSize], acc1, vl);
    }

    if (OutputCountThisIteration > 2) {
        ApplyPostProcessing<Lmul>(acc2, &Output[2 * BlockSize], Bias, KernelFlags, vl);
        Block::Store(&Output[2 * BlockSize], acc2, vl);
    }

    if (OutputCountThisIteration > 3) {
        ApplyPostProcessing<Lmul>(acc3, &Output[3 * BlockSize], Bias, KernelFlags, vl);
        Block::Store(&Output[3 * BlockSize], acc3, vl);
    }
}

//
// Pointwise (1x1) NCHWc convolution kernel.
//
// No padding, kernel size = 1.
// Processes OutputCount output positions, accumulating over InputChannels
// (counted in blocks of BlockSize).
//

template <size_t Lmul>
void
MlasConvPointwiseFloatKernelRvvImpl(
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
    using Block = MLAS_RVV_BLOCK<Lmul>;

    const size_t vl = Block::VectorLength();
    const size_t StrideWidthElements = StrideWidth / sizeof(float);
    const size_t InputStrideElements = InputStride / sizeof(float);
    const size_t FilterStrideElements = FilterStride / sizeof(float);
    const size_t OutputStrideElements = OutputStride / sizeof(float);

    for (size_t f = 0; f < FilterCount; f++) {

        const float* filter = Filter + f * FilterStrideElements;
        float* output = Output + f * OutputStrideElements;
        const float* bias = (Bias != nullptr) ? &Bias[f * BlockSize] : nullptr;

        size_t out = 0;

        for (; out + OutputStep <= OutputCount; out += OutputStep) {
            MlasConvPointwiseFloatGroupRvv<Lmul>(
                Input + out * StrideWidthElements, filter, &output[out * BlockSize],
                StrideWidthElements, InputStrideElements, InputChannels, bias, KernelFlags, vl,
                OutputStep);
        }

        if (out < OutputCount) {
            MlasConvPointwiseFloatGroupRvv<Lmul>(
                Input + out * StrideWidthElements, filter, &output[out * BlockSize],
                StrideWidthElements, InputStrideElements, InputChannels, bias, KernelFlags, vl,
                OutputCount - out);
        }
    }
}

}  // namespace

void
MLASCALL
MlasConvPointwiseFloatKernelRvv(
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
    MLAS_RVV_DISPATCH_ON_LMUL(MlasConvPointwiseFloatKernelRvvImpl, Input, Filter, Output,
                              StrideWidth, InputChannels, FilterCount, InputStride, FilterStride,
                              OutputStride, OutputCount, Bias, KernelFlags)
}

//
// Max pooling kernel for NCHWc format.
//

namespace {

template <size_t Lmul>
void
MlasPoolMaximumFloatKernelRvvImpl(
    const float* Input,
    float* Output,
    size_t StrideWidth,
    size_t DilationWidth,
    size_t InputStride,
    size_t ActualKernelSize,
    size_t KernelHeight,
    size_t KernelWidth,
    const float* InputBase,
    size_t InputWidth,
    size_t DilatedInputWidth,
    size_t OutputCountLeftPad,
    size_t OutputCount,
    size_t OutputCountRightPad
    )
{
    MLAS_UNREFERENCED_PARAMETER(ActualKernelSize);
    MLAS_UNREFERENCED_PARAMETER(InputStride);

    using Block = MLAS_RVV_BLOCK<Lmul>;
    using Vector = typename Block::Vector;

    const size_t vl = Block::VectorLength();
    const size_t StrideWidthElements = StrideWidth / sizeof(float);
    const size_t DilationWidthElements = DilationWidth / sizeof(float);
    const size_t InputWidthElements = InputWidth / sizeof(float);
    const size_t DilatedInputWidthElements = DilatedInputWidth / sizeof(float);
    const size_t TotalOutputCount = OutputCountLeftPad + OutputCount + OutputCountRightPad;

    const float PadValue = std::numeric_limits<float>::lowest();

    for (size_t output_idx = 0; output_idx < TotalOutputCount; output_idx++) {

        Vector max_vec = Block::Broadcast(PadValue, vl);

        for (size_t kh = 0; kh < KernelHeight; kh++) {
            const float* row_start = InputBase + kh * DilatedInputWidthElements;
            const float* row_end = row_start + InputWidthElements;

            for (size_t kw = 0; kw < KernelWidth; kw++) {
                const float* input_ptr = Input + output_idx * StrideWidthElements +
                                         kh * DilatedInputWidthElements + kw * DilationWidthElements;

                if (input_ptr >= row_start && (input_ptr + BlockSize) <= row_end) {
                    Vector inp = Block::Load(input_ptr, vl);
                    max_vec = Block::Maximum(max_vec, inp, vl);
                } else {
                    float values[BlockSize];
                    for (size_t i = 0; i < BlockSize; i++) {
                        const float* ep = input_ptr + i;
                        values[i] = (ep >= row_start && ep < row_end) ? *ep : PadValue;
                    }
                    Vector inp = Block::Load(values, vl);
                    max_vec = Block::Maximum(max_vec, inp, vl);
                }
            }
        }

        Block::Store(&Output[output_idx * BlockSize], max_vec, vl);
    }
}

}  // namespace

void
MLASCALL
MlasPoolMaximumFloatKernelRvv(
    const float* Input,
    float* Output,
    size_t StrideWidth,
    size_t DilationWidth,
    size_t InputStride,
    size_t ActualKernelSize,
    size_t KernelHeight,
    size_t KernelWidth,
    const float* InputBase,
    size_t InputWidth,
    size_t DilatedInputWidth,
    size_t OutputCountLeftPad,
    size_t OutputCount,
    size_t OutputCountRightPad
    )
{
    MLAS_RVV_DISPATCH_ON_LMUL(MlasPoolMaximumFloatKernelRvvImpl, Input, Output, StrideWidth,
                              DilationWidth, InputStride, ActualKernelSize, KernelHeight,
                              KernelWidth, InputBase, InputWidth, DilatedInputWidth,
                              OutputCountLeftPad, OutputCount, OutputCountRightPad)
}

//
// Average pooling kernel (shared implementation).
//

namespace {

template <size_t Lmul>
MLAS_FORCEINLINE
void
MlasPoolAverageFloatKernelRvvImpl(
    const float* Input,
    float* Output,
    size_t StrideWidth,
    size_t DilationWidth,
    size_t ActualKernelSize,
    size_t KernelHeight,
    size_t KernelWidth,
    const float* InputBase,
    size_t InputWidth,
    size_t DilatedInputWidth,
    size_t OutputCountLeftPad,
    size_t OutputCount,
    size_t OutputCountRightPad,
    bool ExcludePad
    )
{
    using Block = MLAS_RVV_BLOCK<Lmul>;
    using Vector = typename Block::Vector;

    const size_t vl = Block::VectorLength();
    const size_t StrideWidthElements = StrideWidth / sizeof(float);
    const size_t DilationWidthElements = DilationWidth / sizeof(float);
    const size_t InputWidthElements = InputWidth / sizeof(float);
    const size_t DilatedInputWidthElements = DilatedInputWidth / sizeof(float);
    const size_t TotalOutputCount = OutputCountLeftPad + OutputCount + OutputCountRightPad;

    for (size_t output_idx = 0; output_idx < TotalOutputCount; output_idx++) {

        Vector sum_vec = Block::Broadcast(0.0f, vl);
        uint32_t valid_count[BlockSize];

        if (ExcludePad) {
            for (size_t i = 0; i < BlockSize; i++) {
                valid_count[i] = 0;
            }
        }

        for (size_t kh = 0; kh < KernelHeight; kh++) {
            const float* row_start = InputBase + kh * DilatedInputWidthElements;
            const float* row_end = row_start + InputWidthElements;

            for (size_t kw = 0; kw < KernelWidth; kw++) {
                const float* input_ptr = Input + output_idx * StrideWidthElements +
                                         kh * DilatedInputWidthElements + kw * DilationWidthElements;

                if (input_ptr >= row_start && (input_ptr + BlockSize) <= row_end) {
                    Vector inp = Block::Load(input_ptr, vl);
                    sum_vec = Block::Add(sum_vec, inp, vl);

                    if (ExcludePad) {
                        for (size_t i = 0; i < BlockSize; i++) {
                            valid_count[i]++;
                        }
                    }
                } else {
                    float values[BlockSize];
                    for (size_t i = 0; i < BlockSize; i++) {
                        const float* ep = input_ptr + i;
                        if (ep >= row_start && ep < row_end) {
                            values[i] = *ep;
                            if (ExcludePad) {
                                valid_count[i]++;
                            }
                        } else {
                            values[i] = 0.0f;
                        }
                    }
                    Vector inp = Block::Load(values, vl);
                    sum_vec = Block::Add(sum_vec, inp, vl);
                }
            }
        }

        if (ExcludePad) {
            float results[BlockSize];
            Block::Store(results, sum_vec, vl);
            for (size_t i = 0; i < BlockSize; i++) {
                results[i] = (valid_count[i] > 0)
                    ? results[i] / static_cast<float>(valid_count[i])
                    : 0.0f;
            }
            Vector result_vec = Block::Load(results, vl);
            Block::Store(&Output[output_idx * BlockSize], result_vec, vl);
        } else {
            Vector divisor = Block::Broadcast(
                static_cast<float>(ActualKernelSize), vl);
            Vector result_vec = Block::Divide(sum_vec, divisor, vl);
            Block::Store(&Output[output_idx * BlockSize], result_vec, vl);
        }
    }
}

}  // namespace

void
MLASCALL
MlasPoolAverageExcludePadFloatKernelRvv(
    const float* Input,
    float* Output,
    size_t StrideWidth,
    size_t DilationWidth,
    size_t InputStride,
    size_t ActualKernelSize,
    size_t KernelHeight,
    size_t KernelWidth,
    const float* InputBase,
    size_t InputWidth,
    size_t DilatedInputWidth,
    size_t OutputCountLeftPad,
    size_t OutputCount,
    size_t OutputCountRightPad
    )
{
    MLAS_UNREFERENCED_PARAMETER(InputStride);

    MLAS_RVV_DISPATCH_ON_LMUL(MlasPoolAverageFloatKernelRvvImpl, Input, Output, StrideWidth,
                              DilationWidth, ActualKernelSize, KernelHeight, KernelWidth,
                              InputBase, InputWidth, DilatedInputWidth, OutputCountLeftPad,
                              OutputCount, OutputCountRightPad, true)
}

void
MLASCALL
MlasPoolAverageIncludePadFloatKernelRvv(
    const float* Input,
    float* Output,
    size_t StrideWidth,
    size_t DilationWidth,
    size_t InputStride,
    size_t ActualKernelSize,
    size_t KernelHeight,
    size_t KernelWidth,
    const float* InputBase,
    size_t InputWidth,
    size_t DilatedInputWidth,
    size_t OutputCountLeftPad,
    size_t OutputCount,
    size_t OutputCountRightPad
    )
{
    MLAS_UNREFERENCED_PARAMETER(InputStride);

    MLAS_RVV_DISPATCH_ON_LMUL(MlasPoolAverageFloatKernelRvvImpl, Input, Output, StrideWidth,
                              DilationWidth, ActualKernelSize, KernelHeight, KernelWidth,
                              InputBase, InputWidth, DilatedInputWidth, OutputCountLeftPad,
                              OutputCount, OutputCountRightPad, false)
}

#endif  // MLAS_USE_RVV
