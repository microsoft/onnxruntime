/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    pooling.cpp

Abstract:

    This module implements the pooling operation.

--*/

#include "mlasi.h"

//
// Define the parameters to execute segments of a pooling operation on worker
// threads.
//

struct MLAS_POOL_WORK_BLOCK
{
    MLAS_POOLING_KIND PoolingKind;
    size_t InputShape[3];
    size_t InputSize;
    size_t OutputShape[3];
    int64_t KernelShape[3];
    int64_t Padding[6];
    int64_t StrideShape[3];
};

//
// Define the prototype of the pooling kernel routine.
//

typedef
void
(MLAS_POOL_KERNEL_ROUTINE)(
    const MLAS_POOL_WORK_BLOCK* WorkBlock,
    size_t ChannelCount,
    const float* Input,
    float* Output
    );

//
// Define the number of elements to allocate on the stack for the reduction
// buffer in the vectorized kernels.
//

#define MLAS_POOL_REDUCTION_BUFFER_STACK    2048

//
// Define the number of reduction buffer elements reserved for over-reading
// an entire vector to avoid special handling at the right edge of the
// buffer.
//

#define MLAS_POOL_REDUCTION_BUFFER_PADDING  ((sizeof(MLAS_FLOAT32X4) / sizeof(float)) - 1)

//
// Abstraction for maximum pooling.
//

struct MLAS_MAXIMUM_POOLING
{
    static float InitialValue()
    {
        return std::numeric_limits<float>::lowest();
    }

    static MLAS_FLOAT32X4 InitialVector()
    {
        return MlasBroadcastFloat32x4(InitialValue());
    }

    static float Reduce(float Reduction, float Value)
    {
        return std::max(Reduction, Value);
    }

    static MLAS_FLOAT32X4 Reduce(MLAS_FLOAT32X4 Reduction, MLAS_FLOAT32X4 Value)
    {
        return MlasMaximumFloat32x4(Reduction, Value);
    }

    static float Reduce(MLAS_FLOAT32X4 Reduction)
    {
        return MlasReduceMaximumFloat32x4(Reduction);
    }

    static float AveragePool(float Reduction, float Size)
    {
        MLAS_UNREFERENCED_PARAMETER(Size);

        return Reduction;
    }

    struct DividerVectorContext
    {
        void PrepareExcludePad(size_t PaddingLeftWidth, size_t InputWidth, size_t KernelWidth)
        {
            MLAS_UNREFERENCED_PARAMETER(PaddingLeftWidth);
            MLAS_UNREFERENCED_PARAMETER(InputWidth);
            MLAS_UNREFERENCED_PARAMETER(KernelWidth);
        }

        void PrepareIncludePad(size_t KernelSize)
        {
            MLAS_UNREFERENCED_PARAMETER(KernelSize);
        }

        void StartNextOutputRow(size_t InputRowsCount)
        {
            MLAS_UNREFERENCED_PARAMETER(InputRowsCount);
        }

        MLAS_FLOAT32X4 DivideExcludePad(MLAS_FLOAT32X4 Reduction)
        {
            return Reduction;
        }

        MLAS_FLOAT32X4 DivideIncludePad(MLAS_FLOAT32X4 Reduction)
        {
            return Reduction;
        }
    };
};

//
// Abstraction for average pooling.
//

MLAS_DECLSPEC_ALIGN(static const float MlasInitialReductionInputIndex[], sizeof(MLAS_FLOAT32X4)) = { 0.0f, 1.0f, 2.0f, 3.0f };

struct MLAS_AVERAGE_POOLING
{
    static float InitialValue()
    {
        return 0.0f;
    }

    static MLAS_FLOAT32X4 InitialVector()
    {
        return MlasZeroFloat32x4();
    }

    static float Reduce(float Reduction, float Value)
    {
        return Reduction + Value;
    }

    static MLAS_FLOAT32X4 Reduce(MLAS_FLOAT32X4 Reduction, MLAS_FLOAT32X4 Value)
    {
        return MlasAddFloat32x4(Reduction, Value);
    }

    static float Reduce(MLAS_FLOAT32X4 Reduction)
    {
        return MlasReduceAddFloat32x4(Reduction);
    }

    static float AveragePool(float Reduction, float Size)
    {
        return Reduction / Size;
    }

    struct DividerVectorContext
    {
        MLAS_FLOAT32X4 KernelSizeBroadcast;
        MLAS_FLOAT32X4 KernelWidthBroadcast;
        MLAS_FLOAT32X4 PaddingLowerBound;
        MLAS_FLOAT32X4 PaddingUpperBound;
        MLAS_FLOAT32X4 ReductionInputIndex;
        MLAS_FLOAT32X4 InputRowsBroadcast;

        void PrepareExcludePad(size_t PaddingLeftWidth, size_t InputWidth, size_t KernelWidth)
        {
            KernelWidthBroadcast = MlasBroadcastFloat32x4(float(unsigned(KernelWidth)));
            PaddingLowerBound = MlasBroadcastFloat32x4(float(unsigned(PaddingLeftWidth)));
            PaddingUpperBound = MlasBroadcastFloat32x4(float(unsigned(PaddingLeftWidth + InputWidth)));
        }

        void PrepareIncludePad(size_t KernelSize)
        {
            KernelSizeBroadcast = MlasBroadcastFloat32x4(float(unsigned(KernelSize)));
        }

        void StartNextOutputRow(size_t InputRowsCount)
        {
            ReductionInputIndex = MlasLoadFloat32x4(MlasInitialReductionInputIndex);
            InputRowsBroadcast = MlasBroadcastFloat32x4(float(unsigned(InputRowsCount)));
        }

        MLAS_FLOAT32X4 DivideExcludePad(MLAS_FLOAT32X4 Reduction)
        {
            MLAS_FLOAT32X4 Divisor;

            //
            // Compute the ending input index for each column and bound the index
            // range by the padding indices, then compute the number of input
            // column contributions from the delta.
            //

            MLAS_FLOAT32X4 ReductionInputEndingIndex =
                MlasAddFloat32x4(ReductionInputIndex, KernelWidthBroadcast);

            MLAS_FLOAT32X4 LowerInputIndex =
                MlasMaximumFloat32x4(ReductionInputIndex, PaddingLowerBound);
            MLAS_FLOAT32X4 UpperInputIndex =
                MlasMinimumFloat32x4(ReductionInputEndingIndex, PaddingUpperBound);

            MLAS_FLOAT32X4 InputIndexDelta =
                MlasSubtractFloat32x4(UpperInputIndex, LowerInputIndex);

            //
            // Advance the input index vector for the next iteration.
            //

            ReductionInputIndex =
                MlasAddFloat32x4(ReductionInputIndex, MlasBroadcastFloat32x4(4.0f));

            //
            // Compute the per-column number of input elements used for the sum.
            //
            // At the end of the input row, the index range computed above may be
            // zero for unused trailing vector elements, so avoid any divide by zero
            // penalty by enforcing a minimum of 1.0f.
            //

            Divisor = MlasMultiplyFloat32x4(InputIndexDelta, InputRowsBroadcast);
            Divisor = MlasMaximumFloat32x4(Divisor, MlasBroadcastFloat32x4(1.0f));

            return MlasDivideFloat32x4(Reduction, Divisor);
        }

        MLAS_FLOAT32X4 DivideIncludePad(MLAS_FLOAT32X4 Reduction)
        {
            return MlasDivideFloat32x4(Reduction, KernelSizeBroadcast);
        }
    };
};

template<typename PoolingType>
void
MlasPool1DKernel(
    const MLAS_POOL_WORK_BLOCK* WorkBlock,
    size_t ChannelCount,
    const float* Input,
    float* Output
    )
/*++

Routine Description:

    This routine implements the 1D pooling operation using generic constructs.

Arguments:

    WorkBlock - Supplies the structure that contains the pooling parameters.

    ChannelCount - Supplies the number of channels to process.

    Input - Supplies the input tensor.

    Output - Supplies the output tensor.

Return Value:

    None.

--*/
{
    constexpr size_t WidthShapeIndex = 0;

    const MLAS_POOLING_KIND PoolingKind = WorkBlock->PoolingKind;

    const size_t InputWidth = WorkBlock->InputShape[WidthShapeIndex];
    const size_t OutputWidth = WorkBlock->OutputShape[WidthShapeIndex];

    const int64_t KernelWidth = WorkBlock->KernelShape[WidthShapeIndex];
    const int64_t PaddingLeftWidth = WorkBlock->Padding[WidthShapeIndex];
    const int64_t StrideWidth = WorkBlock->StrideShape[WidthShapeIndex];

    for (size_t c = 0; c < ChannelCount; c++) {

        for (size_t pw = 0; pw < OutputWidth; pw++) {

            const int64_t iwStart64 = pw * StrideWidth - PaddingLeftWidth;
            const int64_t iwEnd64 = iwStart64 + KernelWidth;

            const size_t iwStart = size_t(std::max(iwStart64, int64_t(0)));
            const size_t iwEnd = size_t(std::min(iwEnd64, int64_t(InputWidth)));

            float m = PoolingType::InitialValue();

            for (size_t iw = size_t(iwStart); iw < size_t(iwEnd); iw++) {
                m = PoolingType::Reduce(m, Input[iw]);
            }

            if (PoolingKind == MlasAveragePoolingExcludePad) {
                m = PoolingType::AveragePool(m, float(iwEnd - iwStart));
            } else {
                m = PoolingType::AveragePool(m, float(KernelWidth));
            }

            *Output++ = m;
        }

        Input += InputWidth;
    }
}

template<typename PoolingType>
void
MlasPool2DKernel(
    const MLAS_POOL_WORK_BLOCK* WorkBlock,
    size_t ChannelCount,
    const float* Input,
    float* Output
    )
/*++

Routine Description:

    This routine implements the 2D pooling operation using generic constructs.

Arguments:

    WorkBlock - Supplies the structure that contains the pooling parameters.

    ChannelCount - Supplies the number of channels to process.

    Input - Supplies the input tensor.

    Output - Supplies the output tensor.

Return Value:

    None.

--*/
{
    constexpr size_t HeightShapeIndex = 0;
    constexpr size_t WidthShapeIndex = 1;

    const MLAS_POOLING_KIND PoolingKind = WorkBlock->PoolingKind;

    const size_t InputHeight = WorkBlock->InputShape[HeightShapeIndex];
    const size_t InputWidth = WorkBlock->InputShape[WidthShapeIndex];
    const size_t InputSize = WorkBlock->InputSize;
    const size_t OutputHeight = WorkBlock->OutputShape[HeightShapeIndex];
    const size_t OutputWidth = WorkBlock->OutputShape[WidthShapeIndex];

    const int64_t KernelHeight = WorkBlock->KernelShape[HeightShapeIndex];
    const int64_t KernelWidth = WorkBlock->KernelShape[WidthShapeIndex];
    const int64_t PaddingLeftHeight = WorkBlock->Padding[HeightShapeIndex];
    const int64_t PaddingLeftWidth = WorkBlock->Padding[WidthShapeIndex];
    const int64_t StrideHeight = WorkBlock->StrideShape[HeightShapeIndex];
    const int64_t StrideWidth = WorkBlock->StrideShape[WidthShapeIndex];

    for (size_t c = 0; c < ChannelCount; c++) {

        for (size_t ph = 0; ph < OutputHeight; ph++) {

            const int64_t ihStart64 = ph * StrideHeight - PaddingLeftHeight;
            const int64_t ihEnd64 = ihStart64 + KernelHeight;

            const size_t ihStart = size_t(std::max(ihStart64, int64_t(0)));
            const size_t ihEnd = size_t(std::min(ihEnd64, int64_t(InputHeight)));

            for (size_t pw = 0; pw < OutputWidth; pw++) {

                const int64_t iwStart64 = pw * StrideWidth - PaddingLeftWidth;
                const int64_t iwEnd64 = iwStart64 + KernelWidth;

                const size_t iwStart = size_t(std::max(iwStart64, int64_t(0)));
                const size_t iwEnd = size_t(std::min(iwEnd64, int64_t(InputWidth)));

                float m = PoolingType::InitialValue();

                for (size_t ih = ihStart; ih < ihEnd; ih++) {
                    for (size_t iw = iwStart; iw < iwEnd; iw++) {
                        m = PoolingType::Reduce(m, Input[ih * InputWidth + iw]);
                    }
                }

                if (PoolingKind == MlasAveragePoolingExcludePad) {
                    m = PoolingType::AveragePool(m, float((ihEnd - ihStart) * (iwEnd - iwStart)));
                } else {
                    m = PoolingType::AveragePool(m, float(KernelHeight * KernelWidth));
                }

                *Output++ = m;
            }
        }

        Input += InputSize;
    }
}

template<typename PoolingType>
void
MlasPool2DVectorKernel(
    const MLAS_POOL_WORK_BLOCK* WorkBlock,
    size_t ChannelCount,
    const float* Input,
    float* Output
    )
/*++

Routine Description:

    This routine implements an optimized 2D pooling operation using vector
    instructions.

Arguments:

    WorkBlock - Supplies the structure that contains the pooling parameters.

    ChannelCount - Supplies the number of channels to process.

    Input - Supplies the input tensor.

    Output - Supplies the output tensor.

Return Value:

    None.

--*/
{
    constexpr size_t Dimensions = 2;

    constexpr size_t HeightShapeIndex = 0;
    constexpr size_t WidthShapeIndex = 1;

    const MLAS_POOLING_KIND PoolingKind = WorkBlock->PoolingKind;

    const size_t InputHeight = WorkBlock->InputShape[HeightShapeIndex];
    const size_t InputWidth = WorkBlock->InputShape[WidthShapeIndex];
    const size_t InputSize = WorkBlock->InputSize;
    const size_t OutputHeight = WorkBlock->OutputShape[HeightShapeIndex];
    const size_t OutputWidth = WorkBlock->OutputShape[WidthShapeIndex];

    const size_t KernelHeight = size_t(WorkBlock->KernelShape[HeightShapeIndex]);
    const size_t KernelWidth = size_t(WorkBlock->KernelShape[WidthShapeIndex]);
    const size_t PaddingLeftHeight = size_t(WorkBlock->Padding[HeightShapeIndex]);
    const size_t PaddingLeftWidth = size_t(WorkBlock->Padding[WidthShapeIndex]);
    const size_t PaddingRightWidth = size_t(WorkBlock->Padding[Dimensions + WidthShapeIndex]);
    const size_t StrideHeight = size_t(WorkBlock->StrideShape[HeightShapeIndex]);
    const size_t StrideWidth = size_t(WorkBlock->StrideShape[WidthShapeIndex]);

    float ReductionBuffer[MLAS_POOL_REDUCTION_BUFFER_STACK];

    //
    // Fill the edges of the reduction buffer with the padding value.
    //

    float* FillReductionBuffer = ReductionBuffer;
    float* FillReductionBufferEnd = FillReductionBuffer + PaddingLeftWidth;

    while (FillReductionBuffer < FillReductionBufferEnd) {
        *FillReductionBuffer++ = PoolingType::InitialValue();
    }

    FillReductionBuffer = FillReductionBuffer + InputWidth;
    FillReductionBufferEnd = FillReductionBuffer + PaddingRightWidth + MLAS_POOL_REDUCTION_BUFFER_PADDING;

    while (FillReductionBuffer < FillReductionBufferEnd) {
        *FillReductionBuffer++ = PoolingType::InitialValue();
    }

    //
    // Apply the pooling operation to each channel.
    //

    typename PoolingType::DividerVectorContext divider;
    divider.PrepareExcludePad(PaddingLeftWidth, InputWidth, KernelWidth);
    divider.PrepareIncludePad(KernelHeight * KernelWidth);

    for (size_t c = 0; c < ChannelCount; c++) {

        for (size_t ph = 0; ph < OutputHeight; ph++) {

            size_t ihStart = ph * StrideHeight - PaddingLeftHeight;
            size_t ihEnd = ihStart + KernelHeight;

            if (ihStart >= InputHeight) {
                ihStart = 0;
            }

            if (ihEnd > InputHeight) {
                ihEnd = InputHeight;
            }

            divider.StartNextOutputRow(ihEnd - ihStart);

            //
            // Reduce the input across the kernel height and store in a local
            // reduction buffer.
            //

            const float* InputRowStart = &Input[ihStart * InputWidth];
            const size_t InputRowsCount = ihEnd - ihStart - 1;
            size_t InputWidthRemaining = InputWidth;
            float* ReductionOutput = &ReductionBuffer[PaddingLeftWidth];

            while (InputWidthRemaining >= 4) {

                const float* InputRow = InputRowStart;
                size_t InputRowsRemaining = InputRowsCount;
                MLAS_FLOAT32X4 Reduction = MlasLoadFloat32x4(InputRow);

                while (InputRowsRemaining > 0) {
                    InputRow += InputWidth;
                    Reduction = PoolingType::Reduce(Reduction, MlasLoadFloat32x4(InputRow));
                    InputRowsRemaining--;
                }

                MlasStoreFloat32x4(ReductionOutput, Reduction);
                ReductionOutput += 4;

                InputRowStart += 4;
                InputWidthRemaining -= 4;
            }

            while (InputWidthRemaining > 0) {

                const float* InputRow = InputRowStart;
                size_t InputRowsRemaining = InputRowsCount;
                float Reduction = *InputRow;

                while (InputRowsRemaining > 0) {
                    InputRow += InputWidth;
                    Reduction = PoolingType::Reduce(Reduction, *InputRow);
                    InputRowsRemaining--;
                }

                *ReductionOutput++ = Reduction;

                InputRowStart += 1;
                InputWidthRemaining -= 1;
            }

            //
            // Reduce the input across the kernel width and store to the output
            // tensor.
            //

            size_t OutputWidthRemaining = OutputWidth;
            const float* ReductionInputStart = ReductionBuffer;

            do {

                const float* ReductionInput = ReductionInputStart;
                const float* ReductionInputEnd = ReductionInput + KernelWidth;
                MLAS_FLOAT32X4 Reduction = MlasLoadFloat32x4(ReductionInput++);

                while (ReductionInput < ReductionInputEnd) {
                    Reduction = PoolingType::Reduce(Reduction, MlasLoadFloat32x4(ReductionInput++));
                }

                if (PoolingKind == MlasAveragePoolingExcludePad) {
                    Reduction = divider.DivideExcludePad(Reduction);
                } else {
                    Reduction = divider.DivideIncludePad(Reduction);
                }

                if (StrideWidth == 1) {

                    if (OutputWidthRemaining < 4) {

                        if (OutputWidthRemaining >= 2) {

                            MlasStoreLowHalfFloat32x4(Output, Reduction);

                            if (OutputWidthRemaining > 2) {
                                MlasStoreLaneFloat32x4<2>(Output + 2, Reduction);
                            }

                        } else {
                            MlasStoreLaneFloat32x4<0>(Output, Reduction);
                        }

                        Output += OutputWidthRemaining;

                        break;
                    }

                    MlasStoreFloat32x4(Output, Reduction);

                    Output += 4;
                    OutputWidthRemaining -= 4;

                } else {

                    if (OutputWidthRemaining == 1) {
                        MlasStoreLaneFloat32x4<0>(Output++, Reduction);
                        break;
                    }

#if defined(MLAS_SSE2_INTRINSICS)
                    Reduction = _mm_shuffle_ps(Reduction, Reduction, _MM_SHUFFLE(2, 0, 2, 0));
                    MlasStoreLowHalfFloat32x4(Output, Reduction);
#else
                    MlasStoreLaneFloat32x4<0>(Output, Reduction);
                    MlasStoreLaneFloat32x4<2>(Output + 1, Reduction);
#endif

                    Output += 2;
                    OutputWidthRemaining -= 2;
                }

                ReductionInputStart += 4;

            } while (OutputWidthRemaining > 0);
        }

        Input += InputSize;
    }
}

template<typename PoolingType>
void
MlasPool3DKernel(
    const MLAS_POOL_WORK_BLOCK* WorkBlock,
    size_t ChannelCount,
    const float* Input,
    float* Output
    )
/*++

Routine Description:

    This routine implements the 3D pooling operation using generic constructs.

Arguments:

    WorkBlock - Supplies the structure that contains the pooling parameters.

    ChannelCount - Supplies the number of channels to process.

    Input - Supplies the input tensor.

    Output - Supplies the output tensor.

Return Value:

    None.

--*/
{
    constexpr size_t DepthShapeIndex = 0;
    constexpr size_t HeightShapeIndex = 1;
    constexpr size_t WidthShapeIndex = 2;

    const MLAS_POOLING_KIND PoolingKind = WorkBlock->PoolingKind;

    const size_t InputDepth = WorkBlock->InputShape[DepthShapeIndex];
    const size_t InputHeight = WorkBlock->InputShape[HeightShapeIndex];
    const size_t InputWidth = WorkBlock->InputShape[WidthShapeIndex];
    const size_t InputSize = WorkBlock->InputSize;
    const size_t OutputDepth = WorkBlock->OutputShape[DepthShapeIndex];
    const size_t OutputHeight = WorkBlock->OutputShape[HeightShapeIndex];
    const size_t OutputWidth = WorkBlock->OutputShape[WidthShapeIndex];

    const int64_t KernelDepth = WorkBlock->KernelShape[DepthShapeIndex];
    const int64_t KernelHeight = WorkBlock->KernelShape[HeightShapeIndex];
    const int64_t KernelWidth = WorkBlock->KernelShape[WidthShapeIndex];
    const int64_t PaddingLeftDepth = WorkBlock->Padding[DepthShapeIndex];
    const int64_t PaddingLeftHeight = WorkBlock->Padding[HeightShapeIndex];
    const int64_t PaddingLeftWidth = WorkBlock->Padding[WidthShapeIndex];
    const int64_t StrideDepth = WorkBlock->StrideShape[DepthShapeIndex];
    const int64_t StrideHeight = WorkBlock->StrideShape[HeightShapeIndex];
    const int64_t StrideWidth = WorkBlock->StrideShape[WidthShapeIndex];

    for (size_t c = 0; c < ChannelCount; c++) {

        for (size_t pd = 0; pd < OutputDepth; pd++) {

            const int64_t idStart64 = pd * StrideDepth - PaddingLeftDepth;
            const int64_t idEnd64 = idStart64 + KernelDepth;

            const size_t idStart = size_t(std::max(idStart64, int64_t(0)));
            const size_t idEnd = size_t(std::min(idEnd64, int64_t(InputDepth)));

            for (size_t ph = 0; ph < OutputHeight; ph++) {

                const int64_t ihStart64 = ph * StrideHeight - PaddingLeftHeight;
                const int64_t ihEnd64 = ihStart64 + KernelHeight;

                const size_t ihStart = size_t(std::max(ihStart64, int64_t(0)));
                const size_t ihEnd = size_t(std::min(ihEnd64, int64_t(InputHeight)));

                for (size_t pw = 0; pw < OutputWidth; pw++) {

                    const int64_t iwStart64 = pw * StrideWidth - PaddingLeftWidth;
                    const int64_t iwEnd64 = iwStart64 + KernelWidth;

                    const size_t iwStart = size_t(std::max(iwStart64, int64_t(0)));
                    const size_t iwEnd = size_t(std::min(iwEnd64, int64_t(InputWidth)));

                    float m = PoolingType::InitialValue();

                    for (size_t id = idStart; id < idEnd; id++) {
                        for (size_t ih = ihStart; ih < ihEnd; ih++) {
                            for (size_t iw = iwStart; iw < iwEnd; iw++) {
                                m = PoolingType::Reduce(m, Input[id * InputHeight * InputWidth + ih * InputWidth + iw]);
                            }
                        }
                    }

                    if (PoolingKind == MlasAveragePoolingExcludePad) {
                        m = PoolingType::AveragePool(m, float((idEnd - idStart) * (ihEnd - ihStart) * (iwEnd - iwStart)));
                    } else {
                        m = PoolingType::AveragePool(m, float(KernelDepth * KernelHeight * KernelWidth));
                    }

                    *Output++ = m;
                }
            }
        }

        Input += InputSize;
    }
}

template<typename PoolingType>
void
MlasPool3DVectorKernel(
    const MLAS_POOL_WORK_BLOCK* WorkBlock,
    size_t ChannelCount,
    const float* Input,
    float* Output
    )
/*++

Routine Description:

    This routine implements an optimized 2D pooling operation using vector
    instructions.

Arguments:

    WorkBlock - Supplies the structure that contains the pooling parameters.

    ChannelCount - Supplies the number of channels to process.

    Input - Supplies the input tensor.

    Output - Supplies the output tensor.

Return Value:

    None.

--*/
{
    constexpr size_t Dimensions = 3;

    constexpr size_t DepthShapeIndex = 0;
    constexpr size_t HeightShapeIndex = 1;
    constexpr size_t WidthShapeIndex = 2;

    const MLAS_POOLING_KIND PoolingKind = WorkBlock->PoolingKind;

    const size_t InputDepth = WorkBlock->InputShape[DepthShapeIndex];
    const size_t InputHeight = WorkBlock->InputShape[HeightShapeIndex];
    const size_t InputWidth = WorkBlock->InputShape[WidthShapeIndex];
    const size_t InputSize = WorkBlock->InputSize;
    const size_t OutputDepth = WorkBlock->OutputShape[DepthShapeIndex];
    const size_t OutputHeight = WorkBlock->OutputShape[HeightShapeIndex];
    const size_t OutputWidth = WorkBlock->OutputShape[WidthShapeIndex];

    const size_t KernelDepth = size_t(WorkBlock->KernelShape[DepthShapeIndex]);
    const size_t KernelHeight = size_t(WorkBlock->KernelShape[HeightShapeIndex]);
    const size_t KernelWidth = size_t(WorkBlock->KernelShape[WidthShapeIndex]);
    const size_t PaddingLeftDepth = size_t(WorkBlock->Padding[DepthShapeIndex]);
    const size_t PaddingLeftHeight = size_t(WorkBlock->Padding[HeightShapeIndex]);
    const size_t PaddingLeftWidth = size_t(WorkBlock->Padding[WidthShapeIndex]);
    const size_t PaddingRightWidth = size_t(WorkBlock->Padding[Dimensions + WidthShapeIndex]);
    const size_t StrideDepth = size_t(WorkBlock->StrideShape[DepthShapeIndex]);
    const size_t StrideHeight = size_t(WorkBlock->StrideShape[HeightShapeIndex]);
    const size_t StrideWidth = size_t(WorkBlock->StrideShape[WidthShapeIndex]);

    float ReductionBuffer[MLAS_POOL_REDUCTION_BUFFER_STACK];

    //
    // Fill the edges of the reduction buffer with the padding value.
    //

    float* FillReductionBuffer = ReductionBuffer;
    float* FillReductionBufferEnd = FillReductionBuffer + PaddingLeftWidth;

    while (FillReductionBuffer < FillReductionBufferEnd) {
        *FillReductionBuffer++ = PoolingType::InitialValue();
    }

    FillReductionBuffer = FillReductionBuffer + InputWidth;
    FillReductionBufferEnd = FillReductionBuffer + PaddingRightWidth + MLAS_POOL_REDUCTION_BUFFER_PADDING;

    while (FillReductionBuffer < FillReductionBufferEnd) {
        *FillReductionBuffer++ = PoolingType::InitialValue();
    }

    //
    // Apply the pooling operation to each channel.
    //

    typename PoolingType::DividerVectorContext divider;
    divider.PrepareExcludePad(PaddingLeftWidth, InputWidth, KernelWidth);
    divider.PrepareIncludePad(KernelDepth * KernelHeight * KernelWidth);

    for (size_t c = 0; c < ChannelCount; c++) {

        for (size_t pd = 0; pd < OutputDepth; pd++) {

            size_t idStart = pd * StrideDepth - PaddingLeftDepth;
            size_t idEnd = idStart + KernelDepth;

            if (idStart >= InputDepth) {
                idStart = 0;
            }

            if (idEnd > InputDepth) {
                idEnd = InputDepth;
            }

            for (size_t ph = 0; ph < OutputHeight; ph++) {

                size_t ihStart = ph * StrideHeight - PaddingLeftHeight;
                size_t ihEnd = ihStart + KernelHeight;

                if (ihStart >= InputHeight) {
                    ihStart = 0;
                }

                if (ihEnd > InputHeight) {
                    ihEnd = InputHeight;
                }

                divider.StartNextOutputRow((idEnd - idStart) * (ihEnd - ihStart));

                //
                // Reduce the input across the kernel height and store in a local
                // reduction buffer.
                //

                const float* InputRowStart = &Input[idStart * InputHeight * InputWidth + ihStart * InputWidth];
                const size_t InputPlanesCount = idEnd - idStart;
                const size_t InputRowsCount = ihEnd - ihStart;
                size_t InputWidthRemaining = InputWidth;
                float* ReductionOutput = &ReductionBuffer[PaddingLeftWidth];
                const size_t InputAdvancePlane = (InputHeight - InputRowsCount) * InputWidth;

                while (InputWidthRemaining >= 4) {

                    const float* InputRow = InputRowStart;
                    size_t InputPlanesRemaining = InputPlanesCount;
                    MLAS_FLOAT32X4 Reduction = PoolingType::InitialVector();

                    do {

                        size_t InputRowsRemaining = InputRowsCount;

                        do {

                            Reduction = PoolingType::Reduce(Reduction, MlasLoadFloat32x4(InputRow));
                            InputRow += InputWidth;
                            InputRowsRemaining--;

                        } while (InputRowsRemaining > 0);

                        InputRow += InputAdvancePlane;
                        InputPlanesRemaining--;

                    } while (InputPlanesRemaining > 0);

                    MlasStoreFloat32x4(ReductionOutput, Reduction);
                    ReductionOutput += 4;

                    InputRowStart += 4;
                    InputWidthRemaining -= 4;
                }

                while (InputWidthRemaining > 0) {

                    const float* InputRow = InputRowStart;
                    size_t InputPlanesRemaining = InputPlanesCount;
                    float Reduction = PoolingType::InitialValue();

                    do {

                        size_t InputRowsRemaining = InputRowsCount;

                        do {

                            Reduction = PoolingType::Reduce(Reduction, *InputRow);
                            InputRow += InputWidth;
                            InputRowsRemaining--;

                        } while (InputRowsRemaining > 0);

                        InputRow += InputAdvancePlane;
                        InputPlanesRemaining--;

                    } while (InputPlanesRemaining > 0);

                    *ReductionOutput++ = Reduction;

                    InputRowStart += 1;
                    InputWidthRemaining -= 1;
                }

                //
                // Reduce the input across the kernel width and store to the output
                // tensor.
                //

                size_t OutputWidthRemaining = OutputWidth;
                const float* ReductionInputStart = ReductionBuffer;

                do {

                    const float* ReductionInput = ReductionInputStart;
                    const float* ReductionInputEnd = ReductionInput + KernelWidth;
                    MLAS_FLOAT32X4 Reduction = MlasLoadFloat32x4(ReductionInput++);

                    while (ReductionInput < ReductionInputEnd) {
                        Reduction = PoolingType::Reduce(Reduction, MlasLoadFloat32x4(ReductionInput++));
                    }

                    if (PoolingKind == MlasAveragePoolingExcludePad) {
                        Reduction = divider.DivideExcludePad(Reduction);
                    } else {
                        Reduction = divider.DivideIncludePad(Reduction);
                    }

                    if (StrideWidth == 1) {

                        if (OutputWidthRemaining < 4) {

                            if (OutputWidthRemaining >= 2) {

                                MlasStoreLowHalfFloat32x4(Output, Reduction);

                                if (OutputWidthRemaining > 2) {
                                    MlasStoreLaneFloat32x4<2>(Output + 2, Reduction);
                                }

                            } else {
                                MlasStoreLaneFloat32x4<0>(Output, Reduction);
                            }

                            Output += OutputWidthRemaining;

                            break;
                        }

                        MlasStoreFloat32x4(Output, Reduction);

                        Output += 4;
                        OutputWidthRemaining -= 4;

                    } else {

                        if (OutputWidthRemaining == 1) {
                            MlasStoreLaneFloat32x4<0>(Output++, Reduction);
                            break;
                        }

#if defined(MLAS_SSE2_INTRINSICS)
                        Reduction = _mm_shuffle_ps(Reduction, Reduction, _MM_SHUFFLE(2, 0, 2, 0));
                        MlasStoreLowHalfFloat32x4(Output, Reduction);
#else
                        MlasStoreLaneFloat32x4<0>(Output, Reduction);
                        MlasStoreLaneFloat32x4<2>(Output + 1, Reduction);
#endif

                        Output += 2;
                        OutputWidthRemaining -= 2;
                    }

                    ReductionInputStart += 4;

                } while (OutputWidthRemaining > 0);
            }
        }

        Input += InputSize;
    }
}

template<typename PoolingType>
void
MlasPoolGlobalKernel(
    const MLAS_POOL_WORK_BLOCK* WorkBlock,
    size_t ChannelCount,
    const float* Input,
    float* Output
    )
/*++

Routine Description:

    This routine implements a global pooling operation.

Arguments:

    WorkBlock - Supplies the structure that contains the pooling parameters.

    ChannelCount - Supplies the number of channels to process.

    Input - Supplies the input tensor.

    Output - Supplies the output tensor.

Return Value:

    None.

--*/
{
    const size_t InputSize = WorkBlock->InputSize;
    const float InputSizeFloat = float(InputSize);

    //
    // Apply the pooling operation to each channel.
    //

    for (size_t c = 0; c < ChannelCount; c++) {

        size_t InputSizeRemaining = InputSize;

        //
        // Iterate over the input buffer a vector at a time.
        //

        MLAS_FLOAT32X4 Reduction = PoolingType::InitialVector();

        while (InputSizeRemaining >= 4) {
            Reduction = PoolingType::Reduce(Reduction, MlasLoadFloat32x4(Input));
            Input += 4;
            InputSizeRemaining -= 4;
        }

        //
        // Reduce the vector to a single float value.
        //

        float ReductionValue = PoolingType::Reduce(Reduction);

        //
        // Iterate over the remaining input buffer an element at a time.
        //

        while (InputSizeRemaining > 0) {
            ReductionValue = PoolingType::Reduce(ReductionValue, *Input++);
            InputSizeRemaining -= 1;
        }

        //
        // Apply average pooling if necessary.
        //

        ReductionValue = PoolingType::AveragePool(ReductionValue, InputSizeFloat);

        *Output++ = ReductionValue;
    }
}

//
// Stores pointers to the pooling kernel routines.
//

static MLAS_POOL_KERNEL_ROUTINE* const MlasPoolGenericKernels[][3] =
{
    {
        MlasPool1DKernel<MLAS_MAXIMUM_POOLING>,
        MlasPool2DKernel<MLAS_MAXIMUM_POOLING>,
        MlasPool3DKernel<MLAS_MAXIMUM_POOLING>,
    },
    {
        MlasPool1DKernel<MLAS_AVERAGE_POOLING>,
        MlasPool2DKernel<MLAS_AVERAGE_POOLING>,
        MlasPool3DKernel<MLAS_AVERAGE_POOLING>,
    },
    {
        MlasPool1DKernel<MLAS_AVERAGE_POOLING>,
        MlasPool2DKernel<MLAS_AVERAGE_POOLING>,
        MlasPool3DKernel<MLAS_AVERAGE_POOLING>,
    },
};

static MLAS_POOL_KERNEL_ROUTINE* const MlasPoolGlobalKernels[] =
{
    MlasPoolGlobalKernel<MLAS_MAXIMUM_POOLING>,
    MlasPoolGlobalKernel<MLAS_AVERAGE_POOLING>,
    MlasPoolGlobalKernel<MLAS_AVERAGE_POOLING>,
};

static MLAS_POOL_KERNEL_ROUTINE* const MlasPoolVectorKernels[][2] =
{
    {
        MlasPool2DVectorKernel<MLAS_MAXIMUM_POOLING>,
        MlasPool3DVectorKernel<MLAS_MAXIMUM_POOLING>,
    },
    {
        MlasPool2DVectorKernel<MLAS_AVERAGE_POOLING>,
        MlasPool3DVectorKernel<MLAS_AVERAGE_POOLING>,
    },
    {
        MlasPool2DVectorKernel<MLAS_AVERAGE_POOLING>,
        MlasPool3DVectorKernel<MLAS_AVERAGE_POOLING>,
    },
};

void
MLASCALL
MlasPool(
    MLAS_POOLING_KIND PoolingKind,
    size_t Dimensions,
    const int64_t* InputShape,
    const int64_t* KernelShape,
    const int64_t* Padding,
    const int64_t* StrideShape,
    const int64_t* OutputShape,
    const float* Input,
    float* Output,
    MLAS_THREADPOOL* ThreadPool
    )
/*++

Routine Description:

    This routine implements the pooling operation.

Arguments:

    PoolingKind - Supplies the kind of pooling operation to perform.

    Dimensions - Supplies the number of dimensions.

    InputShape - Supplies the shape of the input tensor.

    KernelShape - Supplies the shape of the kernel transform.

    Padding - Supplies the number of padding elements at the edge of the input
        tensor.

    StrideShape - Supplies the shape of the stride.

    OutputShape - Supplies the shape of the output tensor.

    Input - Supplies the input tensor.

    Output - Supplies the output tensor.

    ThreadPool - Supplies the thread pool object to use, else nullptr if the
        base library threading support should be used.

Return Value:

    None.

--*/
{
    MLAS_POOL_WORK_BLOCK WorkBlock;

    WorkBlock.PoolingKind = PoolingKind;

    //
    // Compute the total number of channels to process and advance the input
    // and output shapes over the batch and channel counts.
    //

    size_t TotalChannelCount = size_t(InputShape[0]) * size_t(InputShape[1]);

    InputShape += 2;
    OutputShape += 2;

    //
    // Save the pooling parameters.
    //

    size_t InputSize = 1;
    size_t OutputSize = 1;

    bool InputAndKernelShapeMatch = true;
    bool AllStridesAreOne = true;
    bool AllPaddingIsZero = true;
    bool AllKernelsAreSmall = true;

    if (Dimensions > 3) {
#ifdef MLAS_NO_EXCEPTION
        abort();
#else
        throw std::runtime_error("bad dimensions");
#endif
    }

    for (size_t dim = 0; dim < Dimensions; dim++) {

        WorkBlock.InputShape[dim] = size_t(InputShape[dim]);
        WorkBlock.OutputShape[dim] = size_t(OutputShape[dim]);

        if (KernelShape != nullptr) {
            WorkBlock.KernelShape[dim] = KernelShape[dim];
        } else {
            WorkBlock.KernelShape[dim] = InputShape[dim];
        }

        if (Padding != nullptr) {
            WorkBlock.Padding[dim] = Padding[dim];
            WorkBlock.Padding[dim + Dimensions] = Padding[dim + Dimensions];
        } else {
            WorkBlock.Padding[dim] = 0;
            WorkBlock.Padding[dim + Dimensions] = 0;
        }

        if (StrideShape != nullptr) {
            WorkBlock.StrideShape[dim] = StrideShape[dim];
        } else {
            WorkBlock.StrideShape[dim] = 1;
        }

        InputSize *= WorkBlock.InputShape[dim];
        OutputSize *= WorkBlock.OutputShape[dim];

        InputAndKernelShapeMatch &= (WorkBlock.KernelShape[dim] == int64_t(WorkBlock.InputShape[dim]));
        AllStridesAreOne &= (WorkBlock.StrideShape[dim] == 1);
        AllPaddingIsZero &= (WorkBlock.Padding[dim] == 0 && WorkBlock.Padding[dim + Dimensions] == 0);
        AllKernelsAreSmall &= (WorkBlock.KernelShape[dim] <= 32);
    }

    WorkBlock.InputSize = InputSize;

    //
    // Determine which pooling kernel routine to use.
    //
    // The vectorized kernels only support strides of 1 or 2. The kernel size
    // should be kept low in order to keep the divisors for average pooling to
    // be exactly representable as float. The input width plus padding must fit
    // in the reduction buffer.
    //

    MLAS_POOL_KERNEL_ROUTINE* PoolKernelRoutine = MlasPoolGenericKernels[PoolingKind][Dimensions - 1];

    if (InputAndKernelShapeMatch && AllStridesAreOne && AllPaddingIsZero) {

        PoolKernelRoutine = MlasPoolGlobalKernels[PoolingKind];

    } else if (Dimensions >= 2 && WorkBlock.StrideShape[Dimensions - 1] <= 2 && AllKernelsAreSmall) {

        int64_t ReductionBufferRemaining = MLAS_POOL_REDUCTION_BUFFER_STACK - MLAS_POOL_REDUCTION_BUFFER_PADDING;

        if (ReductionBufferRemaining >= WorkBlock.Padding[Dimensions - 1]) {
            ReductionBufferRemaining -= WorkBlock.Padding[Dimensions - 1];
        } else {
            ReductionBufferRemaining = 0;
        }

        if (ReductionBufferRemaining >= WorkBlock.Padding[Dimensions * 2 - 1]) {
            ReductionBufferRemaining -= WorkBlock.Padding[Dimensions * 2 - 1];
        } else {
            ReductionBufferRemaining = 0;
        }

        if (ReductionBufferRemaining >= int64_t(WorkBlock.InputShape[Dimensions - 1])) {
            PoolKernelRoutine = MlasPoolVectorKernels[PoolingKind][Dimensions - 2];
        }
    }

#ifdef MLAS_NO_ONNXRUNTIME_THREADPOOL
    MLAS_UNREFERENCED_PARAMETER(ThreadPool);
    //
    // Execute the pooling kernel routine.
    //

#if defined(_OPENMP)

#pragma omp parallel for
    for (int64_t c = 0; c < int64_t(TotalChannelCount); c++) {
      PoolKernelRoutine(&WorkBlock, 1, Input + c * InputSize, Output + c * OutputSize);
    }

#else

    PoolKernelRoutine(&WorkBlock, TotalChannelCount, Input, Output);

#endif
#else
    //
    // Use an external thread pool if one is provided.
    // TODO: change to use MlasExecuteThreaded
    onnxruntime::concurrency::ThreadPool::TryBatchParallelFor(ThreadPool, static_cast<ptrdiff_t>(TotalChannelCount), [&](ptrdiff_t c) {
      PoolKernelRoutine(&WorkBlock, 1, Input + c * InputSize, Output + c * OutputSize);
    }, 0);
    return;
#endif
}

void
MLASCALL
MlasMaximumPool(
    const uint8_t* const* Input,
    uint8_t* Output,
    size_t Channels,
    size_t OutputCount,
    size_t KernelSize
    )
/*++

Routine Description:

    This routine implements the maximum pooling operation.

    The input is supplied as an indirection buffer. Every pointer in the
    indirection buffer points at a Channels length vector (either from the
    input tensor or a vector of padding values). These are grouped in batches
    of length KernelSize that are processed by the kernel to produce a single
    output of length Channels. These batches are then repeated OutputCount
    times.

Arguments:

    Input - Supplies an indirection buffer to the elements of the input tensor.

    Output - Supplies the output tensor in channels last format.

    Channels - Supplies the number of channels.

    OutputCount - Supplies the number of channel sized output elements to
        produce.

    KernelSize - Supplies the total number of channel sized kernel elements to
        consume.

Return Value:

    None.

--*/
{
    while (OutputCount > 0) {

        size_t ChannelOffset = 0;
        size_t c = Channels;

#if defined(MLAS_SSE2_INTRINSICS)

        while (c >= 32) {

            __m128i MaximumVector0 = _mm_setzero_si128();
            __m128i MaximumVector1 = _mm_setzero_si128();

            for (size_t k = 0; k < KernelSize; k++) {

                __m128i InputVector0 = _mm_loadu_si128((const __m128i*)&Input[k][ChannelOffset]);
                __m128i InputVector1 = _mm_loadu_si128((const __m128i*)&Input[k][ChannelOffset + 16]);

                MaximumVector0 = _mm_max_epu8(MaximumVector0, InputVector0);
                MaximumVector1 = _mm_max_epu8(MaximumVector1, InputVector1);
            }

            _mm_storeu_si128((__m128i*)&Output[0], MaximumVector0);
            _mm_storeu_si128((__m128i*)&Output[16], MaximumVector1);
            Output += 32;

            ChannelOffset += 32;
            c -= 32;
        }

        while (c >= 16) {

            __m128i MaximumVector0 = _mm_setzero_si128();

            for (size_t k = 0; k < KernelSize; k++) {

                __m128i InputVector0 = _mm_loadu_si128((const __m128i*)&Input[k][ChannelOffset]);

                MaximumVector0 = _mm_max_epu8(MaximumVector0, InputVector0);
            }

            _mm_storeu_si128((__m128i*)&Output[0], MaximumVector0);
            Output += 16;

            ChannelOffset += 16;
            c -= 16;
        }

        if (c >= 8) {

            __m128i MaximumVector0 = _mm_setzero_si128();

            for (size_t k = 0; k < KernelSize; k++) {

                __m128i InputVector0 = _mm_loadl_epi64((const __m128i*)&Input[k][ChannelOffset]);

                MaximumVector0 = _mm_max_epu8(MaximumVector0, InputVector0);
            }

            _mm_storel_epi64((__m128i*)&Output[0], MaximumVector0);
            Output += 8;

            ChannelOffset += 8;
            c -= 8;
        }

#elif defined(MLAS_NEON_INTRINSICS)

        while (c >= 32) {

            uint8x16_t MaximumVector0 = vdupq_n_u8(0);
            uint8x16_t MaximumVector1 = vdupq_n_u8(0);

            for (size_t k = 0; k < KernelSize; k++) {

                uint8x16_t InputVector0 = vld1q_u8(&Input[k][ChannelOffset]);
                uint8x16_t InputVector1 = vld1q_u8(&Input[k][ChannelOffset + 16]);

                MaximumVector0 = vmaxq_u8(MaximumVector0, InputVector0);
                MaximumVector1 = vmaxq_u8(MaximumVector1, InputVector1);
            }

            vst1q_u8(&Output[0], MaximumVector0);
            vst1q_u8(&Output[16], MaximumVector1);
            Output += 32;

            ChannelOffset += 32;
            c -= 32;
        }

        while (c >= 16) {

            uint8x16_t MaximumVector0 = vdupq_n_u8(0);

            for (size_t k = 0; k < KernelSize; k++) {

                uint8x16_t InputVector0 = vld1q_u8(&Input[k][ChannelOffset]);

                MaximumVector0 = vmaxq_u8(MaximumVector0, InputVector0);
            }

            vst1q_u8(&Output[0], MaximumVector0);
            Output += 16;

            ChannelOffset += 16;
            c -= 16;
        }

        if (c >= 8) {

            uint8x8_t MaximumVector0 = vdup_n_u8(0);

            for (size_t k = 0; k < KernelSize; k++) {

                uint8x8_t InputVector0 = vld1_u8(&Input[k][ChannelOffset]);

                MaximumVector0 = vmax_u8(MaximumVector0, InputVector0);
            }

            vst1_u8(&Output[0], MaximumVector0);
            Output += 8;

            ChannelOffset += 8;
            c -= 8;
        }

#endif

        while (c > 0) {

            int32_t MaximumValue = 0;

            for (size_t k = 0; k < KernelSize; k++) {
                MaximumValue = std::max(MaximumValue, int32_t(Input[k][ChannelOffset]));
            }

            *Output++ = uint8_t(MaximumValue);

            ChannelOffset += 1;
            c -= 1;
        }

        Input += KernelSize;
        OutputCount -= 1;
    }
}
