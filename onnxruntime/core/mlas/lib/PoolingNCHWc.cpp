/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    PoolingNCHWc.cpp

Abstract:

    This module implements the pooling operation with an output format of NCHWc.

--*/

#include "mlasi.h"

struct MLAS_POOL_WORK_BLOCK {
    const float* Input;
    float* Output;
    MLAS_POOLING_KIND PoolingKind;
    size_t TotalChannelCount;
    size_t InputShape[3];
    size_t InputSize;
    size_t OutputShape[3];
    size_t OutputSize;
    int64_t KernelShape[3];
    int64_t DilationShape[3];
    int64_t Padding[6];
    int64_t StrideShape[3];
    int32_t tids;
};

//#define TIDS 12
#define TIDS (MlasPlatform.GetMaximumThreadCount())

void
Pooler(
    void* Context,
    int32_t Index
    )
{
    const size_t NCHWC = MlasPlatform.NchwcBlockSize;

    constexpr size_t HeightShapeIndex = 0;
    constexpr size_t WidthShapeIndex = 1;

    MLAS_POOL_WORK_BLOCK* WorkBlock = (MLAS_POOL_WORK_BLOCK*)Context;

    const size_t TotalChannelCount = WorkBlock->TotalChannelCount;

    const size_t InputHeight = WorkBlock->InputShape[HeightShapeIndex];
    const size_t InputWidth = WorkBlock->InputShape[WidthShapeIndex];
    const size_t InputSize = WorkBlock->InputSize;

    const size_t OutputHeight = WorkBlock->OutputShape[HeightShapeIndex];
    const size_t OutputWidth = WorkBlock->OutputShape[WidthShapeIndex];
    const size_t OutputSize = WorkBlock->OutputSize;

    const size_t PaddingLeftY = WorkBlock->Padding[HeightShapeIndex];
    const size_t PaddingLeftX = WorkBlock->Padding[WidthShapeIndex];

    const size_t StrideHeight = WorkBlock->StrideShape[HeightShapeIndex];
    const size_t StrideWidth = WorkBlock->StrideShape[WidthShapeIndex];

    const size_t DilationHeight = WorkBlock->DilationShape[HeightShapeIndex];
    const size_t DilationWidth = WorkBlock->DilationShape[WidthShapeIndex];

    const size_t KernelHeight = WorkBlock->KernelShape[HeightShapeIndex];
    const size_t KernelWidth = WorkBlock->KernelShape[WidthShapeIndex];
    const size_t KernelSize = KernelHeight * KernelWidth;

    const size_t SpanHeight = DilationHeight * (KernelHeight - 1) + 1;
    const size_t SpanWidth = DilationWidth * (KernelWidth - 1) + 1;

    //
    //
    //

    size_t OutputCountPadNoneH;

    if (InputHeight >= SpanHeight) {
        OutputCountPadNoneH = (InputHeight - SpanHeight) / StrideHeight + 1;
    } else {
        OutputCountPadNoneH = 0;
    }

    size_t OutputCountWithLeftPaddingH;

    if (InputHeight + PaddingLeftX >= SpanHeight) {
        OutputCountWithLeftPaddingH = (InputHeight + PaddingLeftY - SpanHeight) / StrideHeight + 1;
    } else {
        OutputCountWithLeftPaddingH = OutputHeight;
    }

    size_t OutputCountPadLeftH = OutputCountWithLeftPaddingH - OutputCountPadNoneH;

    if (OutputCountPadLeftH == 0 && PaddingLeftY > 0) {
        OutputCountPadLeftH = 1;
        OutputCountPadNoneH--;
    }

    //
    //
    //

    size_t OutputCountPadNoneW;

    if (InputWidth >= SpanWidth) {
        OutputCountPadNoneW = (InputWidth - SpanWidth) / StrideWidth + 1;
    } else {
        OutputCountPadNoneW = 0;
    }

    size_t OutputCountWithLeftPaddingW;

    if (InputWidth + PaddingLeftX >= SpanWidth) {
        OutputCountWithLeftPaddingW = (InputWidth + PaddingLeftX - SpanWidth) / StrideWidth + 1;
    } else {
        OutputCountWithLeftPaddingW = OutputWidth;
    }

    size_t OutputCountPadLeftW = OutputCountWithLeftPaddingW - OutputCountPadNoneW;

    if (OutputCountPadLeftW == 0 && PaddingLeftX > 0) {
        OutputCountPadLeftW = 1;
        OutputCountPadNoneW--;
    }

    size_t OutputCountPadRightW = OutputWidth - OutputCountWithLeftPaddingW;

    const size_t TotalWork = ((TotalChannelCount + NCHWC - 1) / NCHWC) * OutputHeight;
    const size_t WorkPerThread = TotalWork / WorkBlock->tids;
    const size_t WorkPerThreadExtra = TotalWork % WorkBlock->tids;

    size_t WorkIndex;
    size_t WorkIndexEnd;

    if (uint32_t(Index) < WorkPerThreadExtra) {
        WorkIndex = (WorkPerThread + 1) * Index;
        WorkIndexEnd = WorkIndex + WorkPerThread + 1;
    } else {
        WorkIndex = WorkPerThread * Index + WorkPerThreadExtra;
        WorkIndexEnd = WorkIndex + WorkPerThread;
    }

    const float* Input = WorkBlock->Input;
    float* Output = WorkBlock->Output;

    Input += (WorkIndex / OutputHeight) * InputSize * NCHWC;
    Output += WorkIndex * NCHWC * OutputWidth;

    PMLAS_POOL_FLOAT_KERNEL PoolFloatKernel = nullptr;
    switch (WorkBlock->PoolingKind) {
        case MlasMaximumPooling:
            PoolFloatKernel = MlasPlatform.PoolMaximumFloatKernel;
            break;
        case MlasAveragePoolingExcludePad:
            PoolFloatKernel = MlasPlatform.PoolAverageExcludePadFloatKernel;
            break;
        case MlasAveragePoolingIncludePad:
            PoolFloatKernel = MlasPlatform.PoolAverageIncludePadFloatKernel;
            break;
    }

    while (WorkIndex < WorkIndexEnd) {

        const size_t ph = WorkIndex % OutputHeight;

        size_t ihStart = ph * StrideHeight - PaddingLeftY;

        size_t mykh = KernelHeight;

        if (ph < OutputCountPadLeftH || ph >= OutputCountWithLeftPaddingH) {

            size_t ih = ihStart;

            for (size_t kh = 0; kh < KernelHeight; kh++) {

                if (ih >= InputHeight) {

                    if (ih == ihStart) {
                        ihStart += DilationHeight;
                    }

                    mykh -= 1;
                }

                ih += DilationHeight;
            }
        }

        PoolFloatKernel(Input + NCHWC * (ihStart * InputWidth - PaddingLeftX),
                          Output,
                          NCHWC * StrideWidth * sizeof(float),
                          NCHWC * DilationWidth * sizeof(float),
                          NCHWC * DilationHeight * InputWidth * sizeof(float) - NCHWC * KernelWidth * DilationWidth * sizeof(float),
                          mykh,
                          KernelWidth,
                          KernelSize,
                          Input + NCHWC * (ihStart * InputWidth),
                          NCHWC * InputWidth * sizeof(float),
                          NCHWC * DilationHeight * InputWidth * sizeof(float),
                          OutputCountPadLeftW,
                          OutputCountPadNoneW,
                          OutputCountPadRightW);

        WorkIndex++;

        Output += NCHWC * OutputWidth;

        if ((WorkIndex % OutputHeight) == 0) {
            Input += NCHWC * InputSize;
        }
    }
}

void
MLASCALL
MlasPoolNchwc(
    MLAS_POOLING_KIND PoolingKind,
    size_t Dimensions,
    const int64_t* InputShape,
    const int64_t* KernelShape,
    const int64_t* DilationShape,
    const int64_t* Padding,
    const int64_t* StrideShape,
    const int64_t* OutputShape,
    const float* Input,
    float* Output
    )
{
    MLAS_POOL_WORK_BLOCK WorkBlock;

    WorkBlock.PoolingKind = PoolingKind;

    //
    // Compute the total number of channels to process and advance the input
    // and output shapes over the batch and channel counts.
    //

    WorkBlock.TotalChannelCount = size_t(InputShape[0]) * size_t(InputShape[1]);

    InputShape += 2;
    OutputShape += 2;

    //
    // Save the pooling parameters.
    //

    size_t InputSize = 1;
    size_t OutputSize = 1;

    for (size_t dim = 0; dim < Dimensions; dim++) {

        WorkBlock.InputShape[dim] = size_t(InputShape[dim]);
        WorkBlock.OutputShape[dim] = size_t(OutputShape[dim]);

        if (KernelShape != nullptr) {
            WorkBlock.KernelShape[dim] = KernelShape[dim];
        } else {
            WorkBlock.KernelShape[dim] = InputShape[dim];
        }

        if (DilationShape != nullptr) {
            WorkBlock.DilationShape[dim] = DilationShape[dim];
        } else {
            WorkBlock.DilationShape[dim] = 1;
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
    }

    WorkBlock.InputSize = InputSize;
    WorkBlock.OutputSize = OutputSize;

    WorkBlock.Input = Input;
    WorkBlock.Output = Output;

    WorkBlock.tids = TIDS;

    MlasExecuteThreaded(Pooler, &WorkBlock, TIDS);
}
