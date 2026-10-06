/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    sconv_depthwise_kernel_rvv.cpp

Abstract:

    This module implements RVV kernels for the single precision NCHW depthwise
    convolution that MlasConvPrepare() routes to MlasConvAlgorithmDepthwise: one
    for the 3x3 kernel, dilation 1, padding at most 1 shape, and one for any other
    kernel shape at dilation 1.

    The second exists because the architecture independent kernel covers 3x3
    alone, so every other kernel shape was left to im2col and a GEMM, once per
    channel. A depthwise convolution gives that GEMM a single row, so almost all
    of its time goes into building the column buffer and packing a panel for one
    row of arithmetic.

    The 3x3 body follows the shape of the kernel it replaces. The row pointers
    that kernel keeps -- row0, row1 and row2, each biased by -pad_left and each
    replaced by the caller's zero-filled buffer when it would fall outside the
    image -- already make the three taps of a row contiguous, so the inner loop
    vectorizes directly: one strided load per tap, nine multiply-accumulates and
    one store produce vl output elements at a time. Only the interior of each
    output row is vectorized; the pad_left and pad_right columns stay scalar and
    keep the generic kernel's arithmetic, because they drop the taps that fall
    outside the image rather than reading a zero for them.

    The nine taps are accumulated left to right into a single accumulator and the
    beta term is added last, which is the order the generic kernel's expression is
    written in. The results are still not bit-identical to it and cannot be: the
    generic kernel leaves that expression for the compiler to contract, and the
    summation topology that produces differs. This is the same class of difference
    the direct convolution path already introduces relative to the im2col path it
    replaces.

    Two loop bodies are emitted, one using unit-stride loads for stride_w == 1 and
    one using strided loads otherwise; both strides appear in a depthwise separable
    network.

    The general body cannot borrow that shape, because a wider kernel loses too
    much to the edges: on a seven by seven image a 5x5 kernel pads by two, so
    leaving the padded columns scalar would leave four of seven columns scalar. It
    vectorizes the whole row instead and asks, per tap, which output columns that
    tap can reach. A tap reads its input columns at the convolution's stride, so
    the answer is an interval, and the two ends are not alike:

      starts before the image   only the leading output columns, whose lanes sit
                                at the front of a vector where neither vl nor a
                                pointer offset reaches them, so those columns are
                                scalar - and there are at most pad_left of them
      reaches past the image     the lanes lost are at the end of the vector, so
                                shortening vl drops exactly those
      row outside the image      the tap contributes nothing and is skipped, so
                                the general body needs no zero-filled row

    Two details of that cost more than the arithmetic on a small image, and both
    are hoisted. The bound on how far a tap reaches is computed once per filter
    column, not per tap, because deriving it needs a division. And the taps that
    reach a full vector are a prefix of the filter columns, so the tap loop splits
    there: the full ones use a plain multiply-add and the partial ones a tail
    preserving one, which keeps the vector type from changing on every tap.

    LMUL is chosen at run time, not fixed. A convolution's row is only as wide as
    the image, and a vector operation costs what its LMUL says regardless of how
    much of it the current vl actually uses, so a tile wider than the row wastes
    most of the machine, and how wide a fixed LMUL is depends on VLEN. The body
    below is written once and instantiated at LMUL 1, 2 and 4, and the dispatch
    picks the smallest one whose vector still covers the row.

--*/

#include "mlasi.h"

#if defined(MLAS_USE_RVV)

#include <riscv_vector.h>

#include <algorithm>

namespace {

//
// The LMUL of an RVV intrinsic is part of its name, so the three instantiations
// need three sets of wrappers. They are generated rather than written out to
// keep the arithmetic in one place.
//

#define MLAS_DEPTHWISE_RVV_TRAITS(TraitsName, Suffix)                                        \
    struct TraitsName {                                                                      \
        using Vector = vfloat32##Suffix##_t;                                                 \
                                                                                             \
        static MLAS_FORCEINLINE size_t MaxVectorLength()                                     \
        {                                                                                    \
            return __riscv_vsetvlmax_e32##Suffix();                                          \
        }                                                                                    \
                                                                                             \
        static MLAS_FORCEINLINE size_t VectorLength(size_t Count)                            \
        {                                                                                    \
            return __riscv_vsetvl_e32##Suffix(Count);                                        \
        }                                                                                    \
                                                                                             \
        static MLAS_FORCEINLINE Vector Load(const float* Address, size_t vl)                 \
        {                                                                                    \
            return __riscv_vle32_v_f32##Suffix(Address, vl);                                 \
        }                                                                                    \
                                                                                             \
        static MLAS_FORCEINLINE Vector                                                       \
        LoadStrided(const float* Address, ptrdiff_t ByteStride, size_t vl)                   \
        {                                                                                    \
            return __riscv_vlse32_v_f32##Suffix(Address, ByteStride, vl);                    \
        }                                                                                    \
                                                                                             \
        static MLAS_FORCEINLINE void Store(float* Address, Vector Value, size_t vl)          \
        {                                                                                    \
            __riscv_vse32_v_f32##Suffix(Address, Value, vl);                                 \
        }                                                                                    \
                                                                                             \
        static MLAS_FORCEINLINE Vector Multiply(Vector Value, float Scalar, size_t vl)       \
        {                                                                                    \
            return __riscv_vfmul_vf_f32##Suffix(Value, Scalar, vl);                          \
        }                                                                                    \
                                                                                             \
        static MLAS_FORCEINLINE Vector                                                       \
        MultiplyAdd(Vector Accumulator, float Scalar, Vector Value, size_t vl)               \
        {                                                                                    \
            return __riscv_vfmacc_vf_f32##Suffix(Accumulator, Scalar, Value, vl);            \
        }                                                                                    \
                                                                                             \
        static MLAS_FORCEINLINE Vector Zero(size_t vl)                                       \
        {                                                                                    \
            return __riscv_vfmv_v_f_f32##Suffix(0.0f, vl);                                   \
        }                                                                                    \
                                                                                             \
        /* Leaves the elements past vl in the accumulator alone, which is what */            \
        /* lets a tap that runs out of input columns simply shorten vl. */                   \
        static MLAS_FORCEINLINE Vector                                                       \
        MultiplyAddPartial(Vector Accumulator, float Scalar, Vector Value, size_t vl)         \
        {                                                                                    \
            return __riscv_vfmacc_vf_f32##Suffix##_tu(Accumulator, Scalar, Value, vl);        \
        }                                                                                    \
    }

MLAS_DEPTHWISE_RVV_TRAITS(MlasDepthwiseRvvM1, m1);
MLAS_DEPTHWISE_RVV_TRAITS(MlasDepthwiseRvvM2, m2);
MLAS_DEPTHWISE_RVV_TRAITS(MlasDepthwiseRvvM4, m4);

#undef MLAS_DEPTHWISE_RVV_TRAITS

//
// Accumulates the nine taps for one chunk of an output row.
//
// Row0, Row1 and Row2 point at the first tap of the chunk for each of the
// three filter rows, already biased by -pad_left by the caller. ByteStride is
// the distance in bytes between the elements that consecutive output columns
// read, that is stride_w * sizeof(float).
//

template <typename Traits>
MLAS_FORCEINLINE
typename Traits::Vector
MlasConvDepthwise3x3AccumulateRvv(
    const float* Row0,
    const float* Row1,
    const float* Row2,
    ptrdiff_t ByteStride,
    float w00,
    float w01,
    float w02,
    float w10,
    float w11,
    float w12,
    float w20,
    float w21,
    float w22,
    size_t vl
    )
{
    typename Traits::Vector Accumulator =
        Traits::Multiply(Traits::LoadStrided(Row0, ByteStride, vl), w00, vl);

    Accumulator =
        Traits::MultiplyAdd(Accumulator, w01, Traits::LoadStrided(Row0 + 1, ByteStride, vl), vl);
    Accumulator =
        Traits::MultiplyAdd(Accumulator, w02, Traits::LoadStrided(Row0 + 2, ByteStride, vl), vl);

    Accumulator =
        Traits::MultiplyAdd(Accumulator, w10, Traits::LoadStrided(Row1, ByteStride, vl), vl);
    Accumulator =
        Traits::MultiplyAdd(Accumulator, w11, Traits::LoadStrided(Row1 + 1, ByteStride, vl), vl);
    Accumulator =
        Traits::MultiplyAdd(Accumulator, w12, Traits::LoadStrided(Row1 + 2, ByteStride, vl), vl);

    Accumulator =
        Traits::MultiplyAdd(Accumulator, w20, Traits::LoadStrided(Row2, ByteStride, vl), vl);
    Accumulator =
        Traits::MultiplyAdd(Accumulator, w21, Traits::LoadStrided(Row2 + 1, ByteStride, vl), vl);
    Accumulator =
        Traits::MultiplyAdd(Accumulator, w22, Traits::LoadStrided(Row2 + 2, ByteStride, vl), vl);

    return Accumulator;
}

//
// The unit-stride form of the above. Kept separate so that the common
// stride_w == 1 case uses vle32 rather than vlse32.
//

template <typename Traits>
MLAS_FORCEINLINE
typename Traits::Vector
MlasConvDepthwise3x3AccumulateUnitStrideRvv(
    const float* Row0,
    const float* Row1,
    const float* Row2,
    float w00,
    float w01,
    float w02,
    float w10,
    float w11,
    float w12,
    float w20,
    float w21,
    float w22,
    size_t vl
    )
{
    typename Traits::Vector Accumulator = Traits::Multiply(Traits::Load(Row0, vl), w00, vl);

    Accumulator = Traits::MultiplyAdd(Accumulator, w01, Traits::Load(Row0 + 1, vl), vl);
    Accumulator = Traits::MultiplyAdd(Accumulator, w02, Traits::Load(Row0 + 2, vl), vl);

    Accumulator = Traits::MultiplyAdd(Accumulator, w10, Traits::Load(Row1, vl), vl);
    Accumulator = Traits::MultiplyAdd(Accumulator, w11, Traits::Load(Row1 + 1, vl), vl);
    Accumulator = Traits::MultiplyAdd(Accumulator, w12, Traits::Load(Row1 + 2, vl), vl);

    Accumulator = Traits::MultiplyAdd(Accumulator, w20, Traits::Load(Row2, vl), vl);
    Accumulator = Traits::MultiplyAdd(Accumulator, w21, Traits::Load(Row2 + 1, vl), vl);
    Accumulator = Traits::MultiplyAdd(Accumulator, w22, Traits::Load(Row2 + 2, vl), vl);

    return Accumulator;
}

//
// The trailing padding is derived rather than read from Parameters, because
// pad_right and pad_bottom are allowed not to match the other parameters
// exactly. This mirrors the generic kernel.
//

MLAS_FORCEINLINE
size_t
MlasConvDepthwisePadRight(
    size_t OutputWidth,
    size_t StrideWidth,
    size_t PadLeft,
    size_t InputWidth
    )
{
    if (OutputWidth == 0) {
        return 0;
    }

    return (((OutputWidth - 1) * StrideWidth + 3) > (PadLeft + InputWidth)) ? 1 : 0;
}

//
// Number of columns in an output row that have all nine taps inside the image
// or inside the caller's zero-filled buffer. Every output row has the same
// count, so the dispatch can use it to choose LMUL once.
//

MLAS_FORCEINLINE
size_t
MlasConvDepthwiseInteriorColumns(
    size_t OutputWidth,
    size_t PadLeft,
    size_t PadRight
    )
{
    const size_t AfterPadLeft = (PadLeft == 1 && OutputWidth > 0) ? OutputWidth - 1 : OutputWidth;
    return (AfterPadLeft > PadRight) ? (AfterPadLeft - PadRight) : 0;
}

template <typename Traits>
void
MlasConvDepthwiseFloat_CHW_RvvImpl(
    const MLAS_CONV_PARAMETERS* Parameters,
    const float* Input,
    const float* Filter,
    float* Output,
    const float* Zeros
    )
{
    const float w00 = Filter[0];
    const float w01 = Filter[1];
    const float w02 = Filter[2];
    const float w10 = Filter[3];
    const float w11 = Filter[4];
    const float w12 = Filter[5];
    const float w20 = Filter[6];
    const float w21 = Filter[7];
    const float w22 = Filter[8];

    const size_t H = Parameters->InputShape[0];
    const size_t W = Parameters->InputShape[1];
    const size_t out_rows = Parameters->OutputShape[0];
    const size_t out_cols = Parameters->OutputShape[1];
    const size_t pad_top = Parameters->Padding[0];
    const size_t pad_left = Parameters->Padding[1];
    const size_t stride_h = Parameters->StrideShape[0];
    const size_t stride_w = Parameters->StrideShape[1];

    const float beta = Parameters->Beta;
    const bool accumulate = beta != 0.0f;

    const size_t pad_right = MlasConvDepthwisePadRight(out_cols, stride_w, pad_left, W);
    const size_t interior_columns = MlasConvDepthwiseInteriorColumns(out_cols, pad_left, pad_right);

    //
    // Row pointer bookkeeping, taken from the generic kernel unchanged.
    //

    const float* row0 = (pad_top > 0) ? Zeros : (Input - pad_left);
    const float* row1 = (H + pad_top <= 1) ? Zeros : (Input + (1 - pad_top) * W) - pad_left;
    const float* row2 = (H + pad_top <= 2) ? Zeros : (row1 + W);

    const ptrdiff_t byte_stride = static_cast<ptrdiff_t>(stride_w * sizeof(float));

    for (size_t h = 0, out_row = out_rows; out_row > 0; --out_row) {
        size_t out_col = out_cols;

        if (pad_left == 1) {
            float dotsum = w01 * row0[1] + w02 * row0[2] + w11 * row1[1] + w12 * row1[2] +
                           w21 * row2[1] + w22 * row2[2] + (accumulate ? *Output * beta : 0.f);
            *Output++ = dotsum;
            out_col--;
            row0 += stride_w;
            row1 += stride_w;
            row2 += stride_w;
        }

        size_t interior = interior_columns;
        out_col -= interior;

        if (stride_w == 1) {
            while (interior > 0) {
                const size_t vl = Traits::VectorLength(interior);

                typename Traits::Vector Accumulator =
                    MlasConvDepthwise3x3AccumulateUnitStrideRvv<Traits>(
                        row0, row1, row2, w00, w01, w02, w10, w11, w12, w20, w21, w22, vl);

                if (accumulate) {
                    Accumulator =
                        Traits::MultiplyAdd(Accumulator, beta, Traits::Load(Output, vl), vl);
                }

                Traits::Store(Output, Accumulator, vl);

                Output += vl;
                row0 += vl;
                row1 += vl;
                row2 += vl;
                interior -= vl;
            }
        } else {
            while (interior > 0) {
                const size_t vl = Traits::VectorLength(interior);

                typename Traits::Vector Accumulator = MlasConvDepthwise3x3AccumulateRvv<Traits>(
                    row0, row1, row2, byte_stride, w00, w01, w02, w10, w11, w12, w20, w21, w22,
                    vl);

                if (accumulate) {
                    Accumulator =
                        Traits::MultiplyAdd(Accumulator, beta, Traits::Load(Output, vl), vl);
                }

                Traits::Store(Output, Accumulator, vl);

                Output += vl;
                row0 += vl * stride_w;
                row1 += vl * stride_w;
                row2 += vl * stride_w;
                interior -= vl;
            }
        }

        if (out_col == 1) {  // pad_right == 1
            float dotsum = w00 * row0[0] + w01 * row0[1] + w10 * row1[0] + w11 * row1[1] +
                           w20 * row2[0] + w21 * row2[1] + (accumulate ? *Output * beta : 0.f);
            *Output++ = dotsum;
        }

        h += stride_h;
        row0 = (Input + (h - pad_top) * W) - pad_left;
        row1 = row0 + W;
        row2 = (h + 2 >= H + pad_top) ? Zeros : (row1 + W);
    }
}

//
// One output column of any kernel shape, computed with scalars.
//
// Only the leading columns need this: those are the ones whose leftmost taps
// fall outside the image, and the lanes they would occupy sit at the front of a
// vector, where neither vl nor a pointer offset can reach them. The trailing
// columns do not, because the taps they lose are at the end of the vector and
// shortening vl drops exactly those.
//
// Which taps a column keeps is decided once, as a range of filter columns,
// rather than tested per tap: consecutive filter columns read consecutive input
// columns, so the taps a column keeps are exactly the ones whose filter column
// lands inside the image, and that is an interval.
//

MLAS_FORCEINLINE
float
MlasConvDepthwiseLeadingColumnRvv(
    const float* Input,
    const float* Filter,
    size_t H,
    size_t W,
    size_t KernelHeight,
    size_t KernelWidth,
    ptrdiff_t InputRow,
    ptrdiff_t InputColumn,
    float Initial
    )
{
    const size_t FilterBegin =
        (InputColumn < 0) ? static_cast<size_t>(-InputColumn) : 0;
    const size_t Reach = static_cast<size_t>(static_cast<ptrdiff_t>(W) - InputColumn);
    const size_t FilterEnd = std::min(KernelWidth, Reach);

    float Accumulator = Initial;

    for (size_t kh = 0; kh < KernelHeight; kh++) {
        const ptrdiff_t ih = InputRow + static_cast<ptrdiff_t>(kh);

        if (ih < 0 || static_cast<size_t>(ih) >= H) {
            continue;
        }

        const float* InputRowBase =
            Input + static_cast<size_t>(ih) * W + static_cast<size_t>(InputColumn + ptrdiff_t(FilterBegin));
        const float* FilterRow = Filter + kh * KernelWidth + FilterBegin;

        for (size_t kw = FilterBegin; kw < FilterEnd; kw++) {
            Accumulator += *FilterRow++ * *InputRowBase++;
        }
    }

    return Accumulator;
}

//
// Any kernel shape, dilation 1.
//
// An output row is taken in one pass per vector's worth of columns, and every
// tap accumulates into that vector before it is stored, so the output row is
// written once no matter how many taps the kernel has. That is what separates
// this from the path it replaces: routing a depthwise convolution through im2col
// and a GEMM gives that GEMM a single row, so it spends its time building the
// column buffer and packing a panel for one row of arithmetic.
//
// A tap reads a run of input columns whose stride is the convolution's, so the
// only shape questions are where that run starts and how far it reaches:
//
//   starts before the image   the leading output columns, handled as scalars
//                             above, because those lanes are at the front
//   reaches past the image    handled here by shortening vl, because those
//                             lanes are at the end
//   row outside the image     the whole tap contributes nothing and is skipped,
//                             so no zero-filled row is needed
//

//
// The general body keeps one bound per filter column on the stack, sized by the
// shared kDepthwiseGeneralMaxKernelWidth, which is what removes the division
// from the tap loop.
//

template <typename Traits>
void
MlasConvDepthwiseGeneralRvvImpl(
    const MLAS_CONV_PARAMETERS* Parameters,
    const float* Input,
    const float* Filter,
    float* Output
    )
{
    const size_t H = Parameters->InputShape[0];
    const size_t W = Parameters->InputShape[1];
    const size_t KernelHeight = Parameters->KernelShape[0];
    const size_t KernelWidth = Parameters->KernelShape[1];
    const size_t out_rows = Parameters->OutputShape[0];
    const size_t out_cols = Parameters->OutputShape[1];
    const size_t pad_top = Parameters->Padding[0];
    const size_t pad_left = Parameters->Padding[1];
    const size_t stride_h = Parameters->StrideShape[0];
    const size_t stride_w = Parameters->StrideShape[1];

    const float beta = Parameters->Beta;
    const bool accumulate = beta != 0.0f;

    //
    // The columns whose leftmost tap starts before the image. Every later
    // column has all of its taps at or after column zero, which is what lets
    // the vectorized body below drop the left hand test entirely.
    //

    const size_t leading_columns =
        std::min(out_cols, (pad_left + stride_w - 1) / stride_w);

    //
    // The last output column each filter column can still read, so that the tap
    // loop shortens vl with a comparison rather than a division. A division per
    // tap is what a first revision of this did, and on a 7x7 image with a 5x5
    // kernel it cost more than the arithmetic.
    //
    // A negative bound means the filter column never lands inside the image,
    // which happens once the padding exceeds the image width.
    //

    ptrdiff_t last_column[kDepthwiseGeneralMaxKernelWidth];

    for (size_t kw = 0; kw < KernelWidth; kw++) {
        const ptrdiff_t reach =
            static_cast<ptrdiff_t>(W - 1 + pad_left) - static_cast<ptrdiff_t>(kw);

        last_column[kw] = (reach < 0) ? -1 : (reach / static_cast<ptrdiff_t>(stride_w));
    }

    const ptrdiff_t byte_stride = static_cast<ptrdiff_t>(stride_w * sizeof(float));

    for (size_t oh = 0; oh < out_rows; oh++) {
        const ptrdiff_t input_row =
            static_cast<ptrdiff_t>(oh * stride_h) - static_cast<ptrdiff_t>(pad_top);
        float* out_row = Output + oh * out_cols;

        for (size_t oc = 0; oc < leading_columns; oc++) {
            const ptrdiff_t input_column =
                static_cast<ptrdiff_t>(oc * stride_w) - static_cast<ptrdiff_t>(pad_left);

            out_row[oc] = MlasConvDepthwiseLeadingColumnRvv(
                Input, Filter, H, W, KernelHeight, KernelWidth, input_row, input_column,
                accumulate ? out_row[oc] * beta : 0.0f);
        }

        size_t oc = leading_columns;

        while (oc < out_cols) {
            const size_t vl = Traits::VectorLength(out_cols - oc);

            typename Traits::Vector Accumulator =
                accumulate ? Traits::Multiply(Traits::Load(out_row + oc, vl), beta, vl)
                           : Traits::Zero(vl);

            //
            // How many of this chunk's taps read a full vector. Reach shrinks as
            // the filter column grows, so those are a prefix, and splitting the
            // tap loop there is what keeps the common tap free of a bound, a
            // branch, and a vector type change: a full width load next to a tail
            // preserving multiply-add toggles the type on every tap, which cost
            // more than the arithmetic on a small image.
            //

            size_t taps_full = 0;

            while (taps_full < KernelWidth &&
                   last_column[taps_full] >= static_cast<ptrdiff_t>(oc + vl - 1)) {
                taps_full++;
            }

            for (size_t kh = 0; kh < KernelHeight; kh++) {
                const ptrdiff_t ih = input_row + static_cast<ptrdiff_t>(kh);

                if (ih < 0 || static_cast<size_t>(ih) >= H) {
                    continue;
                }

                //
                // The first input column of this filter row's leftmost tap. It
                // cannot be negative here: oc is at least leading_columns.
                //

                const float* TapBase =
                    Input + static_cast<size_t>(ih) * W + (oc * stride_w - pad_left);
                const float* FilterRow = Filter + kh * KernelWidth;

                if (stride_w == 1) {
                    for (size_t kw = 0; kw < taps_full; kw++) {
                        Accumulator = Traits::MultiplyAdd(Accumulator, FilterRow[kw],
                                                         Traits::Load(TapBase + kw, vl), vl);
                    }
                } else {
                    for (size_t kw = 0; kw < taps_full; kw++) {
                        Accumulator =
                            Traits::MultiplyAdd(Accumulator, FilterRow[kw],
                                                Traits::LoadStrided(TapBase + kw, byte_stride, vl),
                                                vl);
                    }
                }

                for (size_t kw = taps_full; kw < KernelWidth; kw++) {
                    const ptrdiff_t remaining = last_column[kw] - static_cast<ptrdiff_t>(oc) + 1;

                    if (remaining <= 0) {
                        continue;
                    }

                    const size_t count = std::min(vl, static_cast<size_t>(remaining));

                    const typename Traits::Vector Values =
                        (stride_w == 1) ? Traits::Load(TapBase + kw, count)
                                        : Traits::LoadStrided(TapBase + kw, byte_stride, count);

                    Accumulator =
                        Traits::MultiplyAddPartial(Accumulator, FilterRow[kw], Values, count);
                }
            }

            Traits::Store(out_row + oc, Accumulator, vl);
            oc += vl;
        }
    }
}

}  // namespace

void
MLASCALL
MlasConvDepthwiseFloat_CHW(
    const MLAS_CONV_PARAMETERS* Parameters,
    const float* Input,
    const float* Filter,
    float* Output,
    const float* Zeros
    )
/*++

Routine Description:

    Computes one channel of a depthwise convolution using RVV, at the smallest
    LMUL whose vector still covers an output row.

    The 3x3 / padding <= 1 shape has its own body; every other kernel shape goes
    to the general one.

Arguments:

    Parameters - Supplies the prepared convolution parameters.

    Input - Supplies one input channel slice in H x W layout.

    Filter - Supplies that channel's filter, in KH x KW layout.

    Output - Supplies one output channel slice in OH x OW layout.

    Zeros - Supplies the caller's zero-filled buffer of InputShape[1] + 2
        elements, used in place of a row that falls outside the image. The
        general body skips such a row instead, so it does not read this.

--*/
{
    const size_t W = Parameters->InputShape[1];
    const size_t out_cols = Parameters->OutputShape[1];
    const size_t pad_left = Parameters->Padding[1];
    const size_t stride_w = Parameters->StrideShape[1];

    //
    // Any shape other than the 3x3 one below goes to the general body, which
    // covers every kernel shape MlasConvPrepare admits: it routes only shapes
    // whose kernel width is at most kDepthwiseGeneralMaxKernelWidth.
    //

    if (Parameters->KernelShape[0] != 3 || Parameters->KernelShape[1] != 3 ||
        Parameters->Padding[0] > 1 || Parameters->Padding[1] > 1 ||
        Parameters->Padding[2] > 1 || Parameters->Padding[3] > 1 ||
        W <= 1) {

        const size_t leading_columns =
            std::min(out_cols, (pad_left + stride_w - 1) / stride_w);
        const size_t vector_columns = out_cols - leading_columns;
        const size_t vlmax_m1 = MlasDepthwiseRvvM1::MaxVectorLength();

        if (vector_columns <= vlmax_m1) {
            MlasConvDepthwiseGeneralRvvImpl<MlasDepthwiseRvvM1>(Parameters, Input, Filter, Output);
        } else if (vector_columns <= 2 * vlmax_m1) {
            MlasConvDepthwiseGeneralRvvImpl<MlasDepthwiseRvvM2>(Parameters, Input, Filter, Output);
        } else {
            MlasConvDepthwiseGeneralRvvImpl<MlasDepthwiseRvvM4>(Parameters, Input, Filter, Output);
        }

        return;
    }

    const size_t pad_right = MlasConvDepthwisePadRight(out_cols, stride_w, pad_left, W);
    const size_t interior_columns = MlasConvDepthwiseInteriorColumns(out_cols, pad_left, pad_right);

    //
    // Pick the smallest LMUL whose vector holds the whole interior, so that a
    // narrow row does not pay for a wide tile. A row too wide for LMUL=4 is
    // taken in several LMUL=4 passes, where only the last one is partial and
    // the waste is amortized.
    //
    // vlmax is queried rather than derived from a VLEN constant, which keeps
    // this correct on any VLEN.
    //

    const size_t vlmax_m1 = MlasDepthwiseRvvM1::MaxVectorLength();

    if (interior_columns <= vlmax_m1) {
        MlasConvDepthwiseFloat_CHW_RvvImpl<MlasDepthwiseRvvM1>(Parameters, Input, Filter, Output,
                                                               Zeros);
    } else if (interior_columns <= 2 * vlmax_m1) {
        MlasConvDepthwiseFloat_CHW_RvvImpl<MlasDepthwiseRvvM2>(Parameters, Input, Filter, Output,
                                                               Zeros);
    } else {
        MlasConvDepthwiseFloat_CHW_RvvImpl<MlasDepthwiseRvvM4>(Parameters, Input, Filter, Output,
                                                               Zeros);
    }
}

#endif  // defined(MLAS_USE_RVV)
