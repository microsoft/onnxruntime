/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    convolve.cpp

Abstract:

    This module implements the convolution operation.

--*/

#include "mlasi.h"

//
// Define the number of working buffer elements required per thread.
//

#define MLAS_CONV_WORKING_BUFFER_SIZE_PER_THREAD \
    (MLAS_SGEMM_STRIDEN * MLAS_SGEMM_STRIDEK)

//
// Define the parameters to execute segments of a convolution operation on
// worker threads.
//

struct MLAS_CONV_WORK_BLOCK {
    const MLAS_CONV_PARAMETERS* Parameters;
    const float* Input;
    const float* Filter;
    const float* Bias;
    float* WorkingBuffer;
    float* Output;
    struct SEGMENT {
        size_t StartN;
        size_t CountN;
    } Segments[MLAS_MAXIMUM_THREAD_COUNT];
    int32_t TargetThreadCount;
};

void
MlasConvIm2Col(
    const MLAS_CONV_PARAMETERS* Parameters,
    const float* Input,
    float* ColumnBuffer,
    size_t k,
    size_t CountK,
    size_t n,
    size_t CountN
    )
/*++

Routine Description:

    This routine converts the input image to a set of convolution patches
    appropriate for use with a GEMM operation.

    This implementation supports sampling a portion of the convolution
    patches. This avoids the need to allocate very large buffers to store
    all of the convolution patches at once, when the underyling GEMM
    implementation will already break up the operation into panels. Multiple
    threads can also be used to process different portions of the image.

Arguments:

    Parameters - Supplies the structure that contains the convolution
        parameters.

    Input - Supplies the input tensor.

    ColumnBuffer - Supplies the buffer to receive the convolution patches.

    k - Supplies the K to begin sampling the convolution patches.

    CountK - Supplies the count of K to sample for the convolution patches.

    n - Supplies the N to begin sampling the convolution patches.

    CountN - Supplies the count of N to sample for the convolution patches.

Return Value:

    None.

--*/
{
    constexpr size_t HeightShapeIndex = 0;
    constexpr size_t WidthShapeIndex = 1;

    const size_t OutputWidth = Parameters->OutputShape[WidthShapeIndex];

    const size_t StrideHeight = Parameters->StrideShape[HeightShapeIndex];
    const size_t StrideWidth = Parameters->StrideShape[WidthShapeIndex];

    const size_t nx = (n % OutputWidth);
    const size_t ny = (n / OutputWidth);

    const size_t OriginInputX = nx * StrideWidth;
    const size_t OriginInputY = ny * StrideHeight;

    size_t OutputCountX = OutputWidth - nx;

    const size_t InputHeight = Parameters->InputShape[HeightShapeIndex];
    const size_t InputWidth = Parameters->InputShape[WidthShapeIndex];
    const size_t InputSize = Parameters->InputSize;

    const size_t KernelHeight = Parameters->KernelShape[HeightShapeIndex];
    const size_t KernelWidth = Parameters->KernelShape[WidthShapeIndex];

    size_t kx = (k % KernelWidth);
    size_t ky = (k / KernelWidth) % KernelHeight;

    Input = Input + (k / (KernelHeight * KernelWidth)) * InputSize;

    const size_t DilationHeight = Parameters->DilationShape[HeightShapeIndex];
    const size_t DilationWidth = Parameters->DilationShape[WidthShapeIndex];

    const size_t PaddingLeftY = Parameters->Padding[HeightShapeIndex];
    const size_t PaddingLeftX = Parameters->Padding[WidthShapeIndex];

    for (size_t EndingK = k + CountK; k < EndingK; k++) {

        size_t CountX = OutputCountX;
        size_t InputY = (ky * DilationHeight) + OriginInputY - PaddingLeftY;
        const size_t RowInitialInputX = (kx * DilationWidth) - PaddingLeftX;
        size_t InitialInputX = RowInitialInputX + OriginInputX;
        size_t RemainingN = CountN;

        do {

            if (CountX > RemainingN) {
                CountX = RemainingN;
            }

            RemainingN -= CountX;

            //
            // Check if the input is in the top/bottom padding region.
            //

            if (InputY < InputHeight) {

                size_t InputX = InitialInputX;
                const float* InputRow = &Input[InputY * InputWidth];

                do {

                    //
                    // Check if the input is in the left/right padding region.
                    //

                    if (InputX >= InputWidth) {

                        *ColumnBuffer++ = 0;
                        InputX += StrideWidth;
                        CountX--;

                    } else if (StrideWidth == 1) {

                        //
                        // Copy input elements to the column buffer.
                        //

                        size_t CountCopyX = InputWidth - InputX;

                        if (CountCopyX > CountX) {
                            CountCopyX = CountX;
                        }

                        CountX -= CountCopyX;

                        while (CountCopyX >= 4) {
                            MlasStoreFloat32x4(ColumnBuffer, MlasLoadFloat32x4(&InputRow[InputX]));
                            ColumnBuffer += 4;
                            InputX += 4;
                            CountCopyX -= 4;
                        }

                        while (CountCopyX > 0) {
                            *ColumnBuffer++ = InputRow[InputX++];
                            CountCopyX--;
                        }

                    } else if (InputX + CountX * StrideWidth <= InputWidth) {

                        do {
                            *ColumnBuffer++ = InputRow[InputX];
                            InputX += StrideWidth;
                        } while (--CountX > 0);

                    } else {

                        do {
                            *ColumnBuffer++ = (InputX < InputWidth) ? InputRow[InputX] : 0;
                            InputX += StrideWidth;
                        } while (--CountX > 0);
                    }

                } while (CountX > 0);

            } else {

                //
                // The entire input row is in the padding region.
                //

                MLAS_FLOAT32X4 ZeroFloat32x4 = MlasZeroFloat32x4();

                while (CountX >= 4) {
                    MlasStoreFloat32x4(ColumnBuffer, ZeroFloat32x4);
                    ColumnBuffer += 4;
                    CountX -= 4;
                }

                while (CountX > 0) {
                    MlasStoreLaneFloat32x4<0>(ColumnBuffer, ZeroFloat32x4);
                    ColumnBuffer++;
                    CountX--;
                }
            }

            CountX = OutputWidth;
            InputY += StrideHeight;
            InitialInputX = RowInitialInputX;

        } while (RemainingN > 0);

        //
        // Advance the kernel indices and advance to the next channel if the
        // entire kernel is complete.
        //

        if (++kx == KernelWidth) {

            if (++ky == KernelHeight) {

                Input += InputSize;

                ky = 0;
            }

            kx = 0;
        }
    }
}

void
MlasConvVol2Col(
    const MLAS_CONV_PARAMETERS* Parameters,
    const float* Input,
    float* ColumnBuffer,
    size_t k,
    size_t CountK,
    size_t n,
    size_t CountN
    )
/*++

Routine Description:

    This routine converts the input volume to a set of convolution patches
    appropriate for use with a GEMM operation.

    This implementation supports sampling a portion of the convolution
    patches. This avoids the need to allocate very large buffers to store
    all of the convolution patches at once, when the underyling GEMM
    implementation will already break up the operation into panels. Multiple
    threads can also be used to process different portions of the image.

Arguments:

    Parameters - Supplies the structure that contains the convolution
        parameters.

    Input - Supplies the input tensor.

    ColumnBuffer - Supplies the buffer to receive the convolution patches.

    k - Supplies the K to begin sampling the convolution patches.

    CountK - Supplies the count of K to sample for the convolution patches.

    n - Supplies the N to begin sampling the convolution patches.

    CountN - Supplies the count of N to sample for the convolution patches.

Return Value:

    None.

--*/
{
    constexpr size_t DepthShapeIndex = 0;
    constexpr size_t HeightShapeIndex = 1;
    constexpr size_t WidthShapeIndex = 2;

    const size_t OutputHeight = Parameters->OutputShape[HeightShapeIndex];
    const size_t OutputWidth = Parameters->OutputShape[WidthShapeIndex];

    const size_t StrideDepth = Parameters->StrideShape[DepthShapeIndex];
    const size_t StrideHeight = Parameters->StrideShape[HeightShapeIndex];
    const size_t StrideWidth = Parameters->StrideShape[WidthShapeIndex];

    const size_t nx = (n % OutputWidth);
    const size_t ny = ((n / OutputWidth) % OutputHeight);
    const size_t nz = ((n / OutputWidth) / OutputHeight);

    size_t OutputCountX = OutputWidth - nx;
    size_t OutputCountY = OutputHeight - ny;

    const size_t OriginInputX = nx * StrideWidth;
    const size_t OriginInputY = ny * StrideHeight;
    const size_t OriginInputZ = nz * StrideDepth;

    const size_t InputDepth = Parameters->InputShape[DepthShapeIndex];
    const size_t InputHeight = Parameters->InputShape[HeightShapeIndex];
    const size_t InputWidth = Parameters->InputShape[WidthShapeIndex];
    const size_t InputSize = Parameters->InputSize;

    const size_t KernelDepth = Parameters->KernelShape[DepthShapeIndex];
    const size_t KernelHeight = Parameters->KernelShape[HeightShapeIndex];
    const size_t KernelWidth = Parameters->KernelShape[WidthShapeIndex];

    size_t kx = (k % KernelWidth);
    size_t ky = (k / KernelWidth) % KernelHeight;
    size_t kz = ((k / KernelWidth) / KernelHeight) % KernelDepth;

    Input = Input + (k / (KernelDepth * KernelHeight * KernelWidth)) * InputSize;

    const size_t DilationDepth = Parameters->DilationShape[DepthShapeIndex];
    const size_t DilationHeight = Parameters->DilationShape[HeightShapeIndex];
    const size_t DilationWidth = Parameters->DilationShape[WidthShapeIndex];

    const size_t PaddingLeftZ = Parameters->Padding[DepthShapeIndex];
    const size_t PaddingLeftY = Parameters->Padding[HeightShapeIndex];
    const size_t PaddingLeftX = Parameters->Padding[WidthShapeIndex];

    for (size_t EndingK = k + CountK; k < EndingK; k++) {

        size_t CountY = OutputCountY;
        size_t CountX = OutputCountX;
        size_t InputZ = (kz * DilationDepth) + OriginInputZ - PaddingLeftZ;
        const size_t RowInitialInputY = (ky * DilationHeight) - PaddingLeftY;
        size_t InputY = RowInitialInputY + OriginInputY;
        const size_t RowInitialInputX = (kx * DilationWidth) - PaddingLeftX;
        size_t InitialInputX = RowInitialInputX + OriginInputX;
        size_t RemainingN = CountN;

        do {

            if (CountX > RemainingN) {
                CountX = RemainingN;
            }

            RemainingN -= CountX;

            //
            // Check if the input is in the top/bottom or front/back padding region.
            //

            if (InputY < InputHeight && InputZ < InputDepth) {

                size_t InputX = InitialInputX;
                const float* InputRow = &Input[InputZ * (InputHeight * InputWidth) + InputY * InputWidth];

                do {

                    //
                    // Check if the input is in the left/right padding region.
                    //

                    if (InputX >= InputWidth) {

                        *ColumnBuffer++ = 0;
                        InputX += StrideWidth;
                        CountX--;

                    } else if (StrideWidth == 1) {

                        //
                        // Copy input elements to the column buffer.
                        //

                        size_t CountCopyX = InputWidth - InputX;

                        if (CountCopyX > CountX) {
                            CountCopyX = CountX;
                        }

                        CountX -= CountCopyX;

                        while (CountCopyX >= 4) {
                            MlasStoreFloat32x4(ColumnBuffer, MlasLoadFloat32x4(&InputRow[InputX]));
                            ColumnBuffer += 4;
                            InputX += 4;
                            CountCopyX -= 4;
                        }

                        while (CountCopyX > 0) {
                            *ColumnBuffer++ = InputRow[InputX++];
                            CountCopyX--;
                        }

                    } else if (InputX + CountX * StrideWidth <= InputWidth) {

                        do {
                            *ColumnBuffer++ = InputRow[InputX];
                            InputX += StrideWidth;
                        } while (--CountX > 0);

                    } else {

                        do {
                            *ColumnBuffer++ = (InputX < InputWidth) ? InputRow[InputX] : 0;
                            InputX += StrideWidth;
                        } while (--CountX > 0);
                    }

                } while (CountX > 0);

            } else {

                //
                // The entire input row is in the padding region.
                //

                MLAS_FLOAT32X4 ZeroFloat32x4 = MlasZeroFloat32x4();

                while (CountX >= 4) {
                    MlasStoreFloat32x4(ColumnBuffer, ZeroFloat32x4);
                    ColumnBuffer += 4;
                    CountX -= 4;
                }

                while (CountX > 0) {
                    MlasStoreLaneFloat32x4<0>(ColumnBuffer, ZeroFloat32x4);
                    ColumnBuffer++;
                    CountX--;
                }
            }

            CountX = OutputWidth;
            InputY += StrideHeight;
            InitialInputX = RowInitialInputX;

            if (--CountY == 0) {

                InputY = RowInitialInputY;
                InputZ += StrideDepth;

                CountY = OutputHeight;
            }

        } while (RemainingN > 0);

        //
        // Advance the kernel indices and advance to the next channel if the
        // entire kernel is complete.
        //

        if (++kx == KernelWidth) {

            if (++ky == KernelHeight) {

                if (++kz == KernelDepth) {

                    Input += InputSize;

                    kz = 0;
                }

                ky = 0;
            }

            kx = 0;
        }
    }
}

void
MlasConvOperation(
    const MLAS_CONV_PARAMETERS* Parameters,
    const float* Input,
    const float* Filter,
    const float* Bias,
    float* ColumnBuffer,
    float* Output,
    size_t SegmentStartN,
    size_t SegmentCountN
    )
/*++

Routine Description:

    This routine implements the convolution operation.

Arguments:

    Parameters - Supplies the structure that contains the convolution
        parameters.

    Input - Supplies the input tensor.

    Filter - Supplies the filter tensor.

    Bias - Optionally supplies the bias vector.

    ColumnBuffer - Supplies the thread local slice of the working buffer.

    Output - Supplies the output tensor.

    SegmentStartN - Supplies the N to begin sampling the convolution patches.

    SegmentCountN - Supplies the count of N to sample for the convolution
        patches.

Return Value:

    None.

--*/
{
    const size_t FilterCount = Parameters->FilterCount;
    const size_t OutputSize = Parameters->OutputSize;
    const size_t K = Parameters->K;

    //
    // Compute the strides to step through slices of the local segment.
    //
    // See MlasSgemmOperation.
    //

    uint32_t StrideN = MLAS_SGEMM_STRIDEN;
    uint32_t StrideK = MLAS_SGEMM_STRIDEK;

    if (SegmentCountN >= K) {

        while (StrideK / 2 >= K) {
            StrideN *= 2;
            StrideK /= 2;
        }

    } else {

        while (StrideN > 16 && StrideN / 2 >= SegmentCountN) {
            StrideK *= 2;
            StrideN /= 2;
        }
    }

    //
    // Step through each slice of the input tensor along the N dimension.
    //

    size_t CountN;

    for (size_t n = 0; n < SegmentCountN; n += CountN) {

        CountN = SegmentCountN - n;

        if (CountN > StrideN) {
            CountN = StrideN;
        }

        //
        // Step through each slice of the input tensor along the K dimension.
        //

        size_t CountK;
        float beta = 0.0f;
        float* SegmentOutput = Output + SegmentStartN + n;

        for (size_t k = 0; k < K; k += CountK) {

            CountK = K - k;

            if (CountK > StrideK) {
                CountK = StrideK;
            }

            if (Parameters->Dimensions == 2) {
                MlasConvIm2Col(Parameters, Input, ColumnBuffer, k, CountK,
                    SegmentStartN + n, CountN);
            } else {
                MlasConvVol2Col(Parameters, Input, ColumnBuffer, k, CountK,
                    SegmentStartN + n, CountN);
            }

            MlasSgemmOperation(CblasNoTrans, CblasNoTrans, FilterCount, CountN,
                CountK, 1.0f, Filter + k, K, ColumnBuffer, CountN, beta,
                SegmentOutput, OutputSize);

            beta = 1.0f;
        }

        //
        // Apply the activation with optional bias.
        //

        MlasActivation(Parameters->Activation, SegmentOutput, Bias, FilterCount,
            CountN, OutputSize);
    }
}

void
MlasConvOperationThreaded(
    void* Context,
    int32_t Index
    )
/*++

Routine Description:

    This routine is invoked from a worker thread to execute a segment of a
    convolution operation.

Arguments:

    Context - Supplies the pointer to the context for the threaded operation.

    Index - Supplies the current index of the threaded operation.

Return Value:

    None.

--*/
{
    MLAS_CONV_WORK_BLOCK* WorkBlock = (MLAS_CONV_WORK_BLOCK*)Context;

    MLAS_CONV_WORK_BLOCK::SEGMENT* Segment = &WorkBlock->Segments[Index];

    float* ColumnBuffer =
        WorkBlock->WorkingBuffer + Index * MLAS_CONV_WORKING_BUFFER_SIZE_PER_THREAD;

    MlasConvOperation(WorkBlock->Parameters, WorkBlock->Input, WorkBlock->Filter,
        WorkBlock->Bias, ColumnBuffer, WorkBlock->Output, Segment->StartN,
        Segment->CountN);
}

void
MlasConvGemmDirectThreaded(
    void* Context,
    int32_t Index
    )
/*++

Routine Description:

    This routine is invoked from a worker thread to execute a segment of a
    convolution operation.

Arguments:

    Context - Supplies the pointer to the context for the threaded operation.

    Index - Supplies the current index of the threaded operation.

Return Value:

    None.

--*/
{
    MLAS_CONV_WORK_BLOCK* WorkBlock = (MLAS_CONV_WORK_BLOCK*)Context;

    const MLAS_CONV_PARAMETERS* Parameters = WorkBlock->Parameters;

    //
    // Compute the range of indices to use for this thread.
    //

    const size_t GroupCount = Parameters->GroupCount;
    const size_t BatchGroupCount = Parameters->BatchCount * GroupCount;

    const size_t TargetThreadCount = WorkBlock->TargetThreadCount;

    const size_t BatchGroupCountPerThread = BatchGroupCount / TargetThreadCount;
    const size_t BatchGroupCountExtra = BatchGroupCount % TargetThreadCount;

    size_t BatchGroupStart;
    size_t BatchGroupEnd;

    if (uint32_t(Index) < BatchGroupCountExtra) {
        BatchGroupStart = (BatchGroupCountPerThread + 1) * Index;
        BatchGroupEnd = BatchGroupStart + BatchGroupCountPerThread + 1;
    } else {
        BatchGroupStart = BatchGroupCountPerThread * Index + BatchGroupCountExtra;
        BatchGroupEnd = BatchGroupStart + BatchGroupCountPerThread;
    }

    //
    // Iterate over the batch and groups allocated to this thread.
    //

    const size_t FilterCount = Parameters->FilterCount;
    const size_t OutputSize = Parameters->OutputSize;
    const size_t K = Parameters->K;

    const size_t InputGroupSize = Parameters->InputChannels * Parameters->InputSize;
    const size_t OutputGroupSize = FilterCount * OutputSize;
    const size_t FilterGroupSize = FilterCount * K;

    for (size_t bg = BatchGroupStart; bg < BatchGroupEnd; bg++) {

        size_t group = bg % GroupCount;

        const float* input = WorkBlock->Input + bg * InputGroupSize;
        const float* filter = WorkBlock->Filter + group * FilterGroupSize;
        float* output = WorkBlock->Output + bg * OutputGroupSize;

        //
        // Invoke the non-threaded GEMM directly with the input tensor.
        //

        MlasSgemmOperation(CblasNoTrans, Parameters->u.GemmDirect.TransB, FilterCount,
            OutputSize, K, 1.0f, filter, K, input, Parameters->u.GemmDirect.ldb, 0.0f,
            output, OutputSize);

        //
        // Apply the activation with optional bias.
        //

        const float* bias = WorkBlock->Bias;

        if (bias != nullptr) {
            bias += group * FilterCount;
        }

        MlasActivation(Parameters->Activation, output, bias, FilterCount,
            OutputSize, OutputSize);
    }
}

inline
bool
MlasConvTryMultithread(
    const MLAS_CONV_PARAMETERS* Parameters,
    const float* Input,
    const float* Filter,
    const float* Bias,
    float* WorkingBuffer,
    float* Output,
    MLAS_THREADPOOL* ThreadPool
    )
/*++

Routine Description:

    This routine attempts to launch a convolution operation across multiple
    threads.

Arguments:

    Parameters - Supplies the structure that contains the convolution
        parameters.

    Input - Supplies the input tensor.

    Filter - Supplies the filter tensor.

    Bias - Optionally supplies the bias vector.

    WorkingBuffer - Supplies a working buffer sized to the number of elements
        returned by MlasConvPrepare.

    Output - Supplies the output tensor.

    ThreadPool - Supplies the thread pool object to use, else nullptr if the
        base library threading support should be used.

Return Value:

    Returns true if the operation was completed across multiple threads, else
    false if the operation should fall back to a single thread.

--*/
{
    MLAS_CONV_WORK_BLOCK WorkBlock;

    const size_t OutputSize = Parameters->OutputSize;
    const size_t ThreadStrideN = Parameters->u.ExpandThenGemmSegmented.ThreadStrideN;

    if (ThreadStrideN >= OutputSize) {
        return false;
    }

    //
    // Initialize the common fields of the work block.
    //

    WorkBlock.Parameters = Parameters;
    WorkBlock.Input = Input;
    WorkBlock.Filter = Filter;
    WorkBlock.Bias = Bias;
    WorkBlock.WorkingBuffer = WorkingBuffer;
    WorkBlock.Output = Output;

    //
    // Segment the operation across multiple threads.
    //

    int32_t Index = 0;
    size_t SegmentCountN;

    for (size_t SegmentStartN = 0; SegmentStartN < OutputSize; SegmentStartN += SegmentCountN) {

        SegmentCountN = OutputSize - SegmentStartN;

        if (SegmentCountN > ThreadStrideN) {
            SegmentCountN = ThreadStrideN;
        }

        WorkBlock.Segments[Index].StartN = SegmentStartN;
        WorkBlock.Segments[Index].CountN = SegmentCountN;

        Index++;
    }

    MlasExecuteThreaded(MlasConvOperationThreaded, &WorkBlock, Index, ThreadPool);

    return true;
}

void
MLASCALL
MlasConv(
    const MLAS_CONV_PARAMETERS* Parameters,
    const float* Input,
    const float* Filter,
    const float* Bias,
    float* WorkingBuffer,
    float* Output,
    MLAS_THREADPOOL* ThreadPool
    )
/*++

Routine Description:

    This routine implements the convolution operation.

Arguments:

    Parameters - Supplies the structure that contains the convolution
        parameters.

    Input - Supplies the input tensor.

    Filter - Supplies the filter tensor.

    Bias - Optionally supplies the bias vector.

    WorkingBuffer - Supplies a working buffer sized to the number of elements
        returned by MlasConvPrepare.

    Output - Supplies the output tensor.

    ThreadPool - Supplies the thread pool object to use, else nullptr if the
        base library threading support should be used.

Return Value:

    None.

--*/
{
    const size_t FilterCount = Parameters->FilterCount;
    const size_t OutputSize = Parameters->OutputSize;
    const size_t K = Parameters->K;

    const size_t InputGroupSize = Parameters->InputChannels * Parameters->InputSize;
    const size_t OutputGroupSize = FilterCount * OutputSize;
    const size_t FilterGroupSize = FilterCount * K;

    const size_t BatchCount = Parameters->BatchCount;
    const size_t GroupCount = Parameters->GroupCount;

    const MLAS_CONV_ALGORITHM Algorithm = Parameters->Algorithm;

    //
    // Schedule batches of GEMMs across multiple threads.
    //

    if (Algorithm == MlasConvAlgorithmGemmDirect && ((BatchCount > 1) || (GroupCount > 1))) {

        const size_t BatchGroupCount = BatchCount * GroupCount;

        int32_t TargetThreadCount = MlasGetMaximumThreadCount(ThreadPool);

        if (size_t(TargetThreadCount) >= BatchGroupCount) {
            TargetThreadCount = int32_t(BatchGroupCount);
        }

        MLAS_CONV_WORK_BLOCK WorkBlock;

        WorkBlock.Parameters = Parameters;
        WorkBlock.Input = Input;
        WorkBlock.Filter = Filter;
        WorkBlock.Bias = Bias;
        WorkBlock.WorkingBuffer = nullptr;
        WorkBlock.Output = Output;
        WorkBlock.TargetThreadCount = TargetThreadCount;

        MlasExecuteThreaded(MlasConvGemmDirectThreaded, &WorkBlock, TargetThreadCount, ThreadPool);

        return;
    }

#if defined(MLAS_TARGET_WASM)

    if (Algorithm == MlasConvAlgorithmDepthwise || Algorithm == MlasConvAlgorithmDirectConv) {
        std::fill_n(WorkingBuffer, Parameters->InputShape[1] + 2, 0.0f);
    }

#endif

    //
    // Iterate over each batch and group.
    //
    for (size_t batch = 0; batch < BatchCount; batch++) {

        const float* filter = Filter;
        const float* bias = Bias;

        for (size_t group = 0; group < GroupCount; group++) {

            //
            // Dispatch the convolution.
            //

            switch (Algorithm) {

                case MlasConvAlgorithmGemmDirect:
                {
                    //
                    // Invoke the threaded GEMM directly with the input tensor.
                    //

                    MlasGemm(CblasNoTrans, Parameters->u.GemmDirect.TransB, FilterCount,
                        OutputSize, K, 1.0f, filter, K, Input, Parameters->u.GemmDirect.ldb, 0.0f,
                        Output, OutputSize, ThreadPool);

                    //
                    // Apply the activation with optional bias.
                    //

                    MlasActivation(Parameters->Activation, Output, bias, FilterCount,
                        OutputSize, OutputSize);

                    break;
                }

                case MlasConvAlgorithmExpandThenGemm:
                {
                    //
                    // Expand the input tensor to the working buffer and then invoke the
                    // threaded GEMM.
                    //

                    if (Parameters->Dimensions == 2) {
                        MlasConvIm2Col(Parameters, Input, WorkingBuffer, 0, K, 0, OutputSize);
                    } else {
                        MlasConvVol2Col(Parameters, Input, WorkingBuffer, 0, K, 0, OutputSize);
                    }

                    MlasGemm(CblasNoTrans, CblasNoTrans, FilterCount, OutputSize, K, 1.0f, filter,
                        K, WorkingBuffer, OutputSize, 0.0f, Output, OutputSize, ThreadPool);

                    //
                    // Apply the activation with optional bias.
                    //

                    MlasActivation(Parameters->Activation, Output, bias, FilterCount,
                        OutputSize, OutputSize);

                    break;
                }

#if defined(MLAS_TARGET_WASM)

                case MlasConvAlgorithmDepthwise:
                {
                    MlasConvDepthwiseFloat_CHW(Parameters, Input, filter, Output, WorkingBuffer);
                    MlasActivation(Parameters->Activation, Output, bias, FilterCount, OutputSize, OutputSize);
                    break;
                }

                case MlasConvAlgorithmDirectConv:
                {
                    MlasConvDirectFloat_CHW(Parameters, Input, filter, Output, WorkingBuffer);
                    MlasActivation(Parameters->Activation, Output, bias, FilterCount, OutputSize, OutputSize);
                    break;
                }

#endif

                case MlasConvAlgorithmExpandThenGemmSegmented:
                {
                    //
                    // Attempt to launch the convolution across multiple threads or fall
                    // back to a single thread.
                    //

                    if (!MlasConvTryMultithread(Parameters, Input, filter, bias, WorkingBuffer,
                        Output, ThreadPool)) {
                        MlasConvOperation(Parameters, Input, filter, bias, WorkingBuffer,
                            Output, 0, OutputSize);
                    }

                    break;
                }
            }

            //
            // Advance the buffer pointers.
            //

            if (bias != nullptr) {
                bias += FilterCount;
            }

            filter += FilterGroupSize;
            Input += InputGroupSize;
            Output += OutputGroupSize;
        }
    }
}

void
MLASCALL
MlasConvPrepare(
    MLAS_CONV_PARAMETERS* Parameters,
    size_t Dimensions,
    size_t BatchCount,
    size_t GroupCount,
    size_t InputChannels,
    const int64_t* InputShape,
    const int64_t* KernelShape,
    const int64_t* DilationShape,
    const int64_t* Padding,
    const int64_t* StrideShape,
    const int64_t* OutputShape,
    size_t FilterCount,
    const MLAS_ACTIVATION* Activation,
    size_t* WorkingBufferSize,
    MLAS_THREADPOOL* ThreadPool
    )
/*++

Routine Description:

    This routine prepares for a convolution operation by computing required
    parameters including the required working buffer size for intermediate
    results.

Arguments:

    Parameters - Supplies the structure that stores the provided and computed
        parameters for the convolution operation.

    Dimensions - Supplies the number of dimensions (must be between 1 and 3).

    BatchCount - Supplies the number of batches to the processed.

    GroupCount - Supplies the number of channel groups.

    InputChannels - Supplies the number of input channels per group.

    InputShape - Supplies the shape of the input tensor.

    KernelShape - Supplies the shape of the kernel transform.

    DilationShape - Supplies the shape of the dilation.

    Padding - Supplies the number of zero padding elements at the edge of the
        input tensor.

    StrideShape - Supplies the shape of the stride.

    OutputShape - Supplies the shape of the output tensor.

    FilterCount - Supplies the number of rows of the filter matrix per group.

    Activation - Supplies the parameters for the activation to apply to the
        convolution output.

    WorkingBufferSize - Receives the number of elements to allocate for the
        working buffer for intermediate results.

    ThreadPool - Supplies the thread pool object to use, else nullptr if the
        base library threading support should be used.

Return Value:

    None.

--*/
{
    //
    // Save the convolution parameters.
    //

    Parameters->Activation = Activation;
    Parameters->BatchCount = BatchCount;
    Parameters->GroupCount = GroupCount;
    Parameters->InputChannels = InputChannels;
    Parameters->FilterCount = FilterCount;

    size_t InputSize = 1;
    size_t OutputSize = 1;
    size_t K = InputChannels;

    bool AllStridesAreOne = true;
    bool AllDilationsAreOne = true;
    bool AllPaddingIsZero = true;

    for (size_t dim = 0; dim < Dimensions; dim++) {

        Parameters->InputShape[dim] = size_t(InputShape[dim]);
        Parameters->OutputShape[dim] = size_t(OutputShape[dim]);
        Parameters->KernelShape[dim] = size_t(KernelShape[dim]);
        Parameters->DilationShape[dim] = size_t(DilationShape[dim]);
        Parameters->Padding[dim] = size_t(Padding[dim]);
        Parameters->Padding[dim + Dimensions] = size_t(Padding[dim + Dimensions]);
        Parameters->StrideShape[dim] = size_t(StrideShape[dim]);

        InputSize *= Parameters->InputShape[dim];
        OutputSize *= Parameters->OutputShape[dim];
        K *= Parameters->KernelShape[dim];

        AllStridesAreOne &= (Parameters->StrideShape[dim] == 1);
        AllDilationsAreOne &= (Parameters->DilationShape[dim] == 1);
        AllPaddingIsZero &= (Parameters->Padding[dim] == 0 && Parameters->Padding[dim + Dimensions] == 0);
    }

    Parameters->InputSize = InputSize;
    Parameters->OutputSize = OutputSize;
    Parameters->K = K;

    //
    // Promote 1D convolutions to 2D convolutions.
    //

    if (Dimensions == 1) {

        Parameters->InputShape[1] = Parameters->InputShape[0];
        Parameters->InputShape[0] = 1;
        Parameters->OutputShape[1] = Parameters->OutputShape[0];
        Parameters->OutputShape[0] = 1;
        Parameters->KernelShape[1] = Parameters->KernelShape[0];
        Parameters->KernelShape[0] = 1;
        Parameters->DilationShape[1] = Parameters->DilationShape[0];
        Parameters->DilationShape[0] = 1;
        Parameters->Padding[3] = Parameters->Padding[1];
        Parameters->Padding[2] = 0;
        Parameters->Padding[1] = Parameters->Padding[0];
        Parameters->Padding[0] = 0;
        Parameters->StrideShape[1] = Parameters->StrideShape[0];
        Parameters->StrideShape[0] = 1;

        Dimensions = 2;
    }

    Parameters->Dimensions = Dimensions;

    //
    // Evaluate how the convolution will be performed.
    //

    *WorkingBufferSize = 0;

    if (AllStridesAreOne && AllPaddingIsZero) {

        //
        // Detect a pointwise convolution.
        //

        if (K == InputChannels) {

            Parameters->Algorithm = MlasConvAlgorithmGemmDirect;
            Parameters->u.GemmDirect.TransB = CblasNoTrans;
            Parameters->u.GemmDirect.ldb = OutputSize;

            return;
        }

        if (Dimensions == 2 && AllDilationsAreOne && InputChannels == 1) {

            //
            // Detect convolutions where the kernel is using the entire input
            // width or height.
            //

            if (Parameters->KernelShape[1] == Parameters->InputShape[1]) {

                Parameters->Algorithm = MlasConvAlgorithmGemmDirect;
                Parameters->u.GemmDirect.TransB = CblasTrans;
                Parameters->u.GemmDirect.ldb = Parameters->InputShape[1];

                return;
            }

            if (Parameters->KernelShape[0] == Parameters->InputShape[0] &&
                Parameters->KernelShape[1] == 1) {

                Parameters->Algorithm = MlasConvAlgorithmGemmDirect;
                Parameters->u.GemmDirect.TransB = CblasNoTrans;
                Parameters->u.GemmDirect.ldb = Parameters->InputShape[1];

                return;
            }
        }
    }

    if (FilterCount > OutputSize) {

        //
        // The filter count is larger than the output dimensions, so perform the
        // full matrix expansion and then invoke the threaded GEMM.
        //

        Parameters->Algorithm = MlasConvAlgorithmExpandThenGemm;

        *WorkingBufferSize = OutputSize * K;

    } else {

#if defined(MLAS_TARGET_WASM) && !defined(MLAS_TARGET_WASMSIMD)

        // Fast scalar direct conv for wasm without SIMD.

        if (Dimensions == 2
                && Parameters->KernelShape[0] == 3 && Parameters->KernelShape[1] == 3
                && Parameters->Padding[0] <= 1 && Parameters->Padding[1] <= 1
                && Parameters->Padding[2] <= 1 && Parameters->Padding[3] <= 1
                && Parameters->DilationShape[0] == 1 && Parameters->DilationShape[1] == 1) {

            *WorkingBufferSize = Parameters->InputShape[1] + 2;
            if (Parameters->FilterCount == 1 && Parameters->InputChannels == 1) {
                Parameters->Algorithm = MlasConvAlgorithmDepthwise;
            } else {
                Parameters->Algorithm = MlasConvAlgorithmDirectConv;
            }
            return;

        }

#endif

        //
        // Segment the operation across multiple threads by slicing the N
        // dimension (see MlasSgemmTryMultithread).
        //
        // Compute the number of target threads given the complexity of the
        // convolution operation. Small requests should run using the single
        // threaded path.
        //

        int32_t TargetThreadCount;
        double Complexity = double(FilterCount) * double(OutputSize) * double(K);

        if (Complexity < double(MLAS_SGEMM_THREAD_COMPLEXITY * MLAS_MAXIMUM_THREAD_COUNT)) {
            TargetThreadCount = int32_t(Complexity / double(MLAS_SGEMM_THREAD_COMPLEXITY)) + 1;
        } else {
            TargetThreadCount = MLAS_MAXIMUM_THREAD_COUNT;
        }

        int32_t MaximumThreadCount = MlasGetMaximumThreadCount(ThreadPool);

        if (TargetThreadCount >= MaximumThreadCount) {
            TargetThreadCount = MaximumThreadCount;
        }

        //
        // Compute the thread stride for slicing the N dimension.
        //

        size_t StrideN = OutputSize / TargetThreadCount;

        if ((StrideN * TargetThreadCount) != OutputSize) {
            StrideN++;
        }

        if (TargetThreadCount > 1) {

            StrideN = (StrideN + MLAS_SGEMM_STRIDEN_THREAD_ALIGN - 1) & ~(MLAS_SGEMM_STRIDEN_THREAD_ALIGN - 1);

            if (StrideN >= OutputSize) {
                TargetThreadCount = 1;
            } else if (StrideN * (TargetThreadCount - 1) >= OutputSize) {
                TargetThreadCount--;
            }
        }

        Parameters->ThreadCount = TargetThreadCount;

        Parameters->Algorithm = MlasConvAlgorithmExpandThenGemmSegmented;
        Parameters->u.ExpandThenGemmSegmented.ThreadStrideN = StrideN;

        *WorkingBufferSize = TargetThreadCount * MLAS_CONV_WORKING_BUFFER_SIZE_PER_THREAD;
    }
}
