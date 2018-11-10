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

struct MLAS_POOL_WORK_BLOCK {
    volatile LONG Counter;
    const MLAS_POOL_PARAMETERS* Parameters;
    const float* Input;
    float* Output;
    struct SEGMENT {
        size_t StartC;
        size_t CountC;
    } Segments[MLAS_MAXIMUM_THREAD_COUNT];
};

VOID
MlasPoolOperation(
    const MLAS_POOL_PARAMETERS* Parameters,
    const float* Input,
    float* Output,
    size_t SegmentStartC,
    size_t SegmentCountC
    )
/*++

Routine Description:

    This routine implements the pooling operation.

Arguments:

    Parameters - Supplies the structure that contains the pooling parameters.

    Input - Supplies the input tensor.

    Output - Supplies the output tensor.

Return Value:

    None.

--*/
{
    size_t N = Parameters->InputShape[0];
//    size_t C = Parameters->InputShape[1];

    switch (Parameters->Dimensions) {

        case 2:
        {
            size_t InputHeight = Parameters->InputShape[2];
            size_t InputWidth = Parameters->InputShape[3];

            size_t OutputHeight = Parameters->OutputShape[0];
            size_t OutputWidth = Parameters->OutputShape[1];

            Input += SegmentStartC * InputHeight * InputWidth;
            Output += SegmentStartC * OutputHeight * OutputWidth;

            for (size_t n = 0; n < N; n++) {

                for (size_t c = 0; c < SegmentCountC; c++) {

                    for (size_t ph = 0; ph < OutputHeight; ph++) {

                        size_t ihStart = ph * Parameters->StrideShape[0] - Parameters->Padding[0];
                        size_t ihEnd = ihStart + Parameters->KernelShape[0];

                        if (ihStart >= InputHeight) {
                            ihStart = 0;
                        }

                        if (ihEnd > InputHeight) {
                            ihEnd = InputHeight;
                        }

                        for (size_t pw = 0; pw < OutputWidth; pw++) {

                            size_t iwStart = pw * Parameters->StrideShape[1] - Parameters->Padding[1];
                            size_t iwEnd = iwStart + Parameters->KernelShape[1];

                            if (iwStart >= InputWidth) {
                                iwStart = 0;
                            }

                            if (iwEnd > InputWidth) {
                                iwEnd = InputWidth;
                            }

                            float m = 0;

                            for (size_t ih = ihStart; ih < ihEnd; ih++) {

                                for (size_t iw = iwStart; iw < iwEnd; iw++) {

                                    if (Input[ih * InputWidth + iw] > m) {
                                        m = Input[ih * InputWidth + iw];
                                    }
                                }
                            }

                            Output[ph * OutputHeight + pw] = m;
                        }
                    }

                    Input += InputHeight * InputWidth;
                    Output += OutputHeight * OutputWidth;
                }
            }

            break;
        }
    }
}

VOID
CALLBACK
MlasPoolWorkCallback(
    PTP_CALLBACK_INSTANCE Instance,
    PVOID Context,
    PTP_WORK WorkObject
    )
/*++

Routine Description:

    This routine is invoked from a worker thread to execute a segment of a
    convolution operation.

Arguments:

    Instance - Supplies the callback instance object.

    Context - Supplies the pointer to the parameters for the SGEMM operation.

    WorkObject - Supplies the threadpool work object.

Return Value:

    None.

--*/
{
    UNREFERENCED_PARAMETER(Instance);
    UNREFERENCED_PARAMETER(WorkObject);

    MLAS_POOL_WORK_BLOCK* WorkBlock = (MLAS_POOL_WORK_BLOCK*)Context;

    LONG Index = InterlockedIncrement(&WorkBlock->Counter) - 1;

    MLAS_POOL_WORK_BLOCK::SEGMENT* Segment = &WorkBlock->Segments[Index];

    MlasPoolOperation(WorkBlock->Parameters, WorkBlock->Input, WorkBlock->Output,
        Segment->StartC, Segment->CountC);
}

VOID
MLASCALL
MlasPool(
    const MLAS_POOL_PARAMETERS* Parameters,
    const float* Input,
    float* Output
    )
/*++

Routine Description:

    This routine implements the pooling operation.

Arguments:

    Parameters - Supplies the structure that contains the pooling parameters.

    Input - Supplies the input tensor.

    Output - Supplies the output tensor.

Return Value:

    None.

--*/
{
    MLAS_POOL_WORK_BLOCK WorkBlock;

    PTP_WORK WorkObject = CreateThreadpoolWork(MlasPoolWorkCallback, &WorkBlock, nullptr);

//    if (WorkObject == nullptr) {
//        return false;
//    }

    WorkBlock.Counter = 0;
    WorkBlock.Parameters = Parameters;
    WorkBlock.Input = Input;
    WorkBlock.Output = Output;

    //
    // Segment the operation across multiple threads.
    //

    uint32_t Index = 0;
    size_t SegmentCountC;

    size_t C = Parameters->InputShape[1];

    size_t ThreadStrideC = C / 4;
    if (C & 1) ThreadStrideC++;

    for (size_t SegmentStartC = 0; SegmentStartC < C; SegmentStartC += SegmentCountC) {

        SegmentCountC = C - SegmentStartC;

        if (SegmentCountC > ThreadStrideC) {
            SegmentCountC = ThreadStrideC;
        }

        WorkBlock.Segments[Index].StartC = SegmentStartC;
        WorkBlock.Segments[Index].CountC = SegmentCountC;

        //
        // Execute one of the segments on a worker thread.
        //

        if (Index > 0) {
            SubmitThreadpoolWork(WorkObject);
        }

        Index++;
    }

    //
    // Execute the remaining segment on this thread.
    //

    MlasPoolWorkCallback(nullptr, &WorkBlock, WorkObject);

    //
    // Wait for the worker threads to complete.
    //

    WaitForThreadpoolWorkCallbacks(WorkObject, FALSE);
    CloseThreadpoolWork(WorkObject);

//    return true;
}

bool
MLASCALL
MlasPoolPrepare(
    MLAS_POOL_PARAMETERS* Parameters,
    int64_t Dimensions,
    const int64_t* InputShape,
    const int64_t* KernelShape,
    const int64_t* Padding,
    const int64_t* StrideShape
    )
/*++

Routine Description:

    This routine prepares for a pooling operation by computing required
    parameters.

Arguments:

    Parameters - Supplies the structure that stores the provided and computed
        parameters for the pooling operation.

    Dimensions - Supplies the number of dimensions (must be between 1 and 3).

    InputShape - Supplies the shape of the input tensor.

    KernelShape - Supplies the shape of the kernel transform.

    PaddingShape - Supplies the number of zero padding elements at the edge of
        the input tensor.

    StrideShape - Supplies the shape of the stride.

Return Value:

    Returns true if implementation can support this operation.

--*/
{
    //
    // Support only 1D, 2D, or 3D operations.
    //

    if (!(Dimensions >= 1 && Dimensions <= 3)) {
        return false;
    }

    //
    // Save the pooling parameters.
    //

    Parameters->Dimensions = size_t(Dimensions);

    for (size_t dim = 0; dim < size_t(Dimensions) + 2; dim++) {
        Parameters->InputShape[dim] = size_t(InputShape[dim]);
    }

    for (size_t dim = 0; dim < size_t(Dimensions); dim++) {
        Parameters->KernelShape[dim] = size_t(KernelShape[dim]);
        Parameters->Padding[dim * 2] = size_t(Padding[dim * 2]);
        Parameters->Padding[dim * 2 + 1] = size_t(Padding[dim * 2 + 1]);
        Parameters->StrideShape[dim] = (StrideShape != nullptr) ? size_t(StrideShape[dim]) : 1;
    }

    //
    // Compute the output shape.
    //

    for (size_t dim = 0; dim < size_t(Dimensions); dim++) {

if (StrideShape != nullptr) {
        int64_t OutputShape = (InputShape[dim + 2] + Padding[dim] +
            Padding[dim + Dimensions] - KernelShape[dim]) / StrideShape[dim] + 1;

        if (OutputShape <= 0) {
            return false;
        }

        Parameters->OutputShape[dim] = size_t(OutputShape);
} else {
        int64_t OutputShape = (InputShape[dim + 2] + Padding[dim] * 2
             - KernelShape[dim]) + 1;

        if (OutputShape <= 0) {
            return false;
        }

        Parameters->OutputShape[dim] = size_t(OutputShape);

}
    }

    return true;
}
