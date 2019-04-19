/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    snchwc.cpp

Abstract:

    This module implements the single precision operations using the NCHWc
    blocking format.

--*/

#include "mlasi.h"

//
// Define the base thead context for NCWHc convolution or pooling operations.
//

struct MLAS_NCHWC_WORK_BLOCK {
    size_t BatchCount;
    size_t InputChannels;
    size_t InputShape[3];
    size_t InputSize;
    size_t OutputChannels;
    size_t OutputShape[3];
    size_t OutputSize;
    size_t KernelShape[3];
    size_t DilationShape[3];
    size_t Padding[6];
    size_t StrideShape[3];
    size_t OutputCountLeftPad[3];
    size_t OutputCount[3];
    size_t OutputCountRightPad[3];
};

//
// Define the worker thread context for a NCHWc convolution operation.
//

struct MLAS_CONV_NCHWC_WORK_BLOCK : MLAS_NCHWC_WORK_BLOCK {
    const float* Input;
    const float* Filter;
    const float* Bias;
    const MLAS_ACTIVATION* Activation;
    float* Output;
    size_t GroupCount;
    int32_t tids;
    bool ZeroMode;
};

//
// Define the worker thread context for a NCHWc pooling operation.
//

struct MLAS_POOL_NCHWC_WORK_BLOCK : MLAS_NCHWC_WORK_BLOCK {
    const float* Input;
    float* Output;
    MLAS_POOLING_KIND PoolingKind;
    int32_t tids;
};

void
MlasReorderInput(
    const int64_t* InputShape,
    const float* S,
    float* D
    )
{
    const size_t NCHWC = MlasPlatform.GetNchwcBlockSize();

    const size_t InputChannels = size_t(InputShape[1]);
    const size_t InputSize = size_t(InputShape[2]) * size_t(InputShape[3]);

    for (size_t c = 0; c < InputChannels; c += NCHWC) {

        const float* s = S;

        for (size_t i = 0; i < InputSize; i++) {

            __m128 v1 = _mm_load_ss(&s[0 * InputSize]);
            v1 = _mm_insert_ps(v1, _mm_load_ss(&s[1 * InputSize]), 0x10);
            v1 = _mm_insert_ps(v1, _mm_load_ss(&s[2 * InputSize]), 0x20);
            v1 = _mm_insert_ps(v1, _mm_load_ss(&s[3 * InputSize]), 0x30);

            __m128 v2 = _mm_load_ss(&s[4 * InputSize]);
            v2 = _mm_insert_ps(v2, _mm_load_ss(&s[5 * InputSize]), 0x10);
            v2 = _mm_insert_ps(v2, _mm_load_ss(&s[6 * InputSize]), 0x20);
            v2 = _mm_insert_ps(v2, _mm_load_ss(&s[7 * InputSize]), 0x30);

            _mm_storeu_ps(&D[0], v1);
            _mm_storeu_ps(&D[4], v2);

            if (NCHWC == 16) {

                __m128 v3 = _mm_load_ss(&s[8 * InputSize]);
                v3 = _mm_insert_ps(v3, _mm_load_ss(&s[9 * InputSize]), 0x10);
                v3 = _mm_insert_ps(v3, _mm_load_ss(&s[10 * InputSize]), 0x20);
                v3 = _mm_insert_ps(v3, _mm_load_ss(&s[11 * InputSize]), 0x30);

                __m128 v4 = _mm_load_ss(&s[12 * InputSize]);
                v4 = _mm_insert_ps(v4, _mm_load_ss(&s[13 * InputSize]), 0x10);
                v4 = _mm_insert_ps(v4, _mm_load_ss(&s[14 * InputSize]), 0x20);
                v4 = _mm_insert_ps(v4, _mm_load_ss(&s[15 * InputSize]), 0x30);

                _mm_storeu_ps(&D[8], v3);
                _mm_storeu_ps(&D[12], v4);
            }

            D += NCHWC;
            s += 1;
        }

        S += NCHWC * InputSize;
    }
}

void
MlasReorderOutput(
    const int64_t* OutputShape,
    const float* S,
    float* D
    )
{
    const size_t NCHWC = MlasPlatform.GetNchwcBlockSize();

    const size_t OutputChannels = size_t(OutputShape[1]);
    const size_t OutputSize = size_t(OutputShape[2]) * size_t(OutputShape[3]);

    for (size_t c = 0; c < OutputChannels; c += NCHWC) {

        const float* s = S;

        for (size_t z = 0; z < NCHWC; z++) {

            const float* ss = s;
            size_t i = OutputSize;

            while (i >= 8) {

                __m128 v1 = _mm_load_ss(&ss[0]);
                v1 = _mm_insert_ps(v1, _mm_load_ss(&ss[NCHWC * 1]), 0x10);
                v1 = _mm_insert_ps(v1, _mm_load_ss(&ss[NCHWC * 2]), 0x20);
                v1 = _mm_insert_ps(v1, _mm_load_ss(&ss[NCHWC * 3]), 0x30);

                __m128 v2 = _mm_load_ss(&ss[NCHWC * 4]);
                v2 = _mm_insert_ps(v2, _mm_load_ss(&ss[NCHWC * 5]), 0x10);
                v2 = _mm_insert_ps(v2, _mm_load_ss(&ss[NCHWC * 6]), 0x20);
                v2 = _mm_insert_ps(v2, _mm_load_ss(&ss[NCHWC * 7]), 0x30);

                _mm_storeu_ps(&D[0], v1);
                _mm_storeu_ps(&D[4], v2);

                D += 8;
                ss += NCHWC * 8;
                i -= 8;
            }

            while (i > 0) {
                *D = *ss;
                D += 1;
                ss += NCHWC;
                i -= 1;
            }

            s += 1;
        }

        S += NCHWC * OutputSize;
    }
}

void
MlasConvReorderFilter(
    const int64_t* FilterShape,
    const float* S,
    float* D
    )
{
    const size_t NCHWC = MlasPlatform.GetNchwcBlockSize();

    constexpr size_t HeightShapeIndex = 0;
    constexpr size_t WidthShapeIndex = 1;

    const size_t OutputChannels = size_t(FilterShape[0]);
    const size_t InputChannels = size_t(FilterShape[1]);
    const size_t KernelHeight = size_t(FilterShape[2]);
    const size_t KernelWidth = size_t(FilterShape[3]);

    const size_t KernelSize = KernelHeight * KernelWidth;
    const size_t InputStride = InputChannels * KernelSize;

    for (size_t o = 0; o < OutputChannels; o += NCHWC) {

        const float* inputI = S;

        for (size_t i = 0; i < InputChannels; i += NCHWC) {

            const float* inputK = inputI;

            for (size_t k = 0; k < KernelSize; k++) {

                const float* input = inputK;

                for (unsigned ik = 0; ik < NCHWC; ik++) {

                    __m128 v1 = _mm_load_ss(&input[0 * InputStride]);
                    v1 = _mm_insert_ps(v1, _mm_load_ss(&input[1 * InputStride]), 0x10);
                    v1 = _mm_insert_ps(v1, _mm_load_ss(&input[2 * InputStride]), 0x20);
                    v1 = _mm_insert_ps(v1, _mm_load_ss(&input[3 * InputStride]), 0x30);

                    __m128 v2 = _mm_load_ss(&input[4 * InputStride]);
                    v2 = _mm_insert_ps(v2, _mm_load_ss(&input[5 * InputStride]), 0x10);
                    v2 = _mm_insert_ps(v2, _mm_load_ss(&input[6 * InputStride]), 0x20);
                    v2 = _mm_insert_ps(v2, _mm_load_ss(&input[7 * InputStride]), 0x30);

                    _mm_storeu_ps(&D[0], v1);
                    _mm_storeu_ps(&D[4], v2);

                    if (NCHWC == 16) {

                        __m128 v3 = _mm_load_ss(&input[8 * InputStride]);
                        v3 = _mm_insert_ps(v3, _mm_load_ss(&input[9 * InputStride]), 0x10);
                        v3 = _mm_insert_ps(v3, _mm_load_ss(&input[10 * InputStride]), 0x20);
                        v3 = _mm_insert_ps(v3, _mm_load_ss(&input[11 * InputStride]), 0x30);

                        __m128 v4 = _mm_load_ss(&input[12 * InputStride]);
                        v4 = _mm_insert_ps(v4, _mm_load_ss(&input[13 * InputStride]), 0x10);
                        v4 = _mm_insert_ps(v4, _mm_load_ss(&input[14 * InputStride]), 0x20);
                        v4 = _mm_insert_ps(v4, _mm_load_ss(&input[15 * InputStride]), 0x30);

                        _mm_storeu_ps(&D[8], v3);
                        _mm_storeu_ps(&D[12], v4);
                    }

                    D += NCHWC;
                    input += KernelSize;
                }

                inputK++;
            }

            inputI += NCHWC * KernelSize;
        }

        S += NCHWC * InputStride;
    }
}

void
MlasConvReorderFilter2(
    const int64_t* FilterShape,
    const float* S,
    float* D
    )
{
    const size_t NCHWC = MlasPlatform.GetNchwcBlockSize();

    constexpr size_t HeightShapeIndex = 0;
    constexpr size_t WidthShapeIndex = 1;

    const size_t OutputChannels = size_t(FilterShape[0]);
    const size_t InputChannels = size_t(FilterShape[1]);
    const size_t KernelHeight = size_t(FilterShape[2]);
    const size_t KernelWidth = size_t(FilterShape[3]);

    const size_t KernelSize = KernelHeight * KernelWidth;
    const size_t InputStride = InputChannels * KernelSize;

    for (size_t o = 0; o < OutputChannels; o += NCHWC) {

        const float* inputI = S;

        for (size_t i = 0; i < InputChannels; i += 1) {

            const float* inputK = inputI;

            for (size_t k = 0; k < KernelSize; k++) {

                const float* input = inputK;

                __m128 v1 = _mm_load_ss(&input[0 * InputStride]);
                v1 = _mm_insert_ps(v1, _mm_load_ss(&input[1 * InputStride]), 0x10);
                v1 = _mm_insert_ps(v1, _mm_load_ss(&input[2 * InputStride]), 0x20);
                v1 = _mm_insert_ps(v1, _mm_load_ss(&input[3 * InputStride]), 0x30);

                __m128 v2 = _mm_load_ss(&input[4 * InputStride]);
                v2 = _mm_insert_ps(v2, _mm_load_ss(&input[5 * InputStride]), 0x10);
                v2 = _mm_insert_ps(v2, _mm_load_ss(&input[6 * InputStride]), 0x20);
                v2 = _mm_insert_ps(v2, _mm_load_ss(&input[7 * InputStride]), 0x30);

                _mm_storeu_ps(&D[0], v1);
                _mm_storeu_ps(&D[4], v2);

                if (NCHWC == 16) {

                    __m128 v3 = _mm_load_ss(&input[8 * InputStride]);
                    v3 = _mm_insert_ps(v3, _mm_load_ss(&input[9 * InputStride]), 0x10);
                    v3 = _mm_insert_ps(v3, _mm_load_ss(&input[10 * InputStride]), 0x20);
                    v3 = _mm_insert_ps(v3, _mm_load_ss(&input[11 * InputStride]), 0x30);

                    __m128 v4 = _mm_load_ss(&input[12 * InputStride]);
                    v4 = _mm_insert_ps(v4, _mm_load_ss(&input[13 * InputStride]), 0x10);
                    v4 = _mm_insert_ps(v4, _mm_load_ss(&input[14 * InputStride]), 0x20);
                    v4 = _mm_insert_ps(v4, _mm_load_ss(&input[15 * InputStride]), 0x30);

                    _mm_storeu_ps(&D[8], v3);
                    _mm_storeu_ps(&D[12], v4);
                }

                D += NCHWC;

                inputK++;
            }

            inputI += KernelSize;
        }

        S += NCHWC * InputStride;
    }
}

void
MlasPrepareNchwcWorkBlock(
    MLAS_NCHWC_WORK_BLOCK* WorkBlock,
    size_t Dimensions,
    const int64_t* InputShape,
    const int64_t* KernelShape,
    const int64_t* DilationShape,
    const int64_t* Padding,
    const int64_t* StrideShape,
    const int64_t* OutputShape
    )
{
    //
    // Extract and skip over the the batch and channel counts.
    //

    WorkBlock->BatchCount = size_t(InputShape[0]);
    WorkBlock->InputChannels = size_t(InputShape[1]);
    WorkBlock->OutputChannels = size_t(OutputShape[1]);

    InputShape += 2;
    OutputShape += 2;

    //
    // Extract the shape information along each dimension.
    //

    size_t InputSize = 1;
    size_t OutputSize = 1;

    for (size_t dim = 0; dim < Dimensions; dim++) {

        const size_t InputValue = size_t(InputShape[dim]);
        const size_t OutputValue = size_t(OutputShape[dim]);

        WorkBlock->InputShape[dim] = InputValue;
        WorkBlock->OutputShape[dim] = OutputValue;

        InputSize *= InputValue;
        OutputSize *= OutputValue;

        if (KernelShape != nullptr) {
            WorkBlock->KernelShape[dim] = size_t(KernelShape[dim]);
        } else {
            WorkBlock->KernelShape[dim] = InputValue;
        }

        if (DilationShape != nullptr) {
            WorkBlock->DilationShape[dim] = size_t(DilationShape[dim]);
        } else {
            WorkBlock->DilationShape[dim] = 1;
        }

        if (Padding != nullptr) {
            WorkBlock->Padding[dim] = size_t(Padding[dim]);
            WorkBlock->Padding[dim + Dimensions] = size_t(Padding[dim + Dimensions]);
        } else {
            WorkBlock->Padding[dim] = 0;
            WorkBlock->Padding[dim + Dimensions] = 0;
        }

        if (StrideShape != nullptr) {
            WorkBlock->StrideShape[dim] = size_t(StrideShape[dim]);
        } else {
            WorkBlock->StrideShape[dim] = 1;
        }

        //
        //
        //

        const size_t SpanValue =
            WorkBlock->DilationShape[dim] * (WorkBlock->KernelShape[dim] - 1) + 1;
        const size_t StrideValue = WorkBlock->StrideShape[dim];
        const size_t PaddingLeftValue = WorkBlock->Padding[dim];

        size_t OutputCount;

        if (InputValue >= SpanValue) {
            OutputCount = (InputValue - SpanValue) / StrideValue + 1;
        } else {
            OutputCount = 0;
        }

        size_t OutputCountWithLeftPad;

        if (InputValue + PaddingLeftValue >= SpanValue) {
            OutputCountWithLeftPad = (InputValue + PaddingLeftValue - SpanValue) / StrideValue + 1;
        } else {
            OutputCountWithLeftPad = OutputValue;
        }

        size_t OutputCountLeftPad = OutputCountWithLeftPad - OutputCount;

        if (OutputCountLeftPad == 0 && PaddingLeftValue > 0) {
            OutputCountLeftPad = 1;
            OutputCount--;
        }

        size_t OutputCountRightPad = OutputValue - OutputCountWithLeftPad;

        WorkBlock->OutputCountLeftPad[dim] = OutputCountLeftPad;
        WorkBlock->OutputCount[dim] = OutputCount;
        WorkBlock->OutputCountRightPad[dim] = OutputCountRightPad;
    }

    WorkBlock->InputSize = InputSize;
    WorkBlock->OutputSize = OutputSize;
}

#define KERNEL_POINTWISE    MlasPlatform.ConvPointwiseFloatKernel
#define KERNEL              MlasPlatform.ConvNchwcFloatKernel
#define KERNEL_NCHW         MlasPlatform.ConvNchwFloatKernel
#define KERNEL_DEPTHWISE    MlasPlatform.ConvDepthwiseFloatKernel

void
MlasConvPointwiseThreaded(
    void* Context,
    int32_t Index
    )
{
    const size_t NCHWC = MlasPlatform.GetNchwcBlockSize();

    MLAS_CONV_NCHWC_WORK_BLOCK* WorkBlock = (MLAS_CONV_NCHWC_WORK_BLOCK*)Context;
    const MLAS_ACTIVATION* Activation = WorkBlock->Activation;
    const bool ZeroMode = WorkBlock->ZeroMode;

    constexpr size_t HeightShapeIndex = 0;
    constexpr size_t WidthShapeIndex = 1;

    const size_t InputChannels = WorkBlock->InputChannels;
    const size_t OutputChannels = WorkBlock->OutputChannels;

    const size_t InputHeight = WorkBlock->InputShape[HeightShapeIndex];
    const size_t InputWidth = WorkBlock->InputShape[WidthShapeIndex];
    const size_t InputSize = WorkBlock->InputSize;

    const size_t OutputHeight = WorkBlock->OutputShape[HeightShapeIndex];
    const size_t OutputWidth = WorkBlock->OutputShape[WidthShapeIndex];
    const size_t OutputSize = WorkBlock->OutputSize;

    const size_t StrideHeight = WorkBlock->StrideShape[HeightShapeIndex];
    const size_t StrideWidth = WorkBlock->StrideShape[WidthShapeIndex];

    //
    //
    //

    const size_t kbatch = 4;

    const size_t TotalWork = ((OutputChannels + (NCHWC * kbatch) - 1) / (NCHWC * kbatch)) * OutputHeight;
    const size_t WorkPerThread = TotalWork / WorkBlock->tids;
    const size_t WorkPerThreadExtra = TotalWork % WorkBlock->tids;

    size_t WorkIndex;
    size_t WorkRemaining;

    if (uint32_t(Index) < WorkPerThreadExtra) {
        WorkIndex = (WorkPerThread + 1) * Index;
        WorkRemaining = WorkPerThread + 1;
    } else {
        WorkIndex = WorkPerThread * Index + WorkPerThreadExtra;
        WorkRemaining = WorkPerThread;
    }

    const float* Input = WorkBlock->Input;
    const float* Filter = WorkBlock->Filter;
    const float* Bias = WorkBlock->Bias;
    float* Output = WorkBlock->Output;

    size_t batch = WorkIndex / OutputHeight;
    size_t ph = WorkIndex % OutputHeight;

    Filter += batch * NCHWC * kbatch * InputChannels;
    Output += batch * NCHWC * kbatch * OutputSize;

    if (Bias != nullptr) {
        Bias += batch * NCHWC * kbatch;
    }

    float* output = Output + NCHWC * ph * OutputWidth;

    //
    //
    //

    while (WorkRemaining > 0) {

        size_t ih = ph * StrideHeight;

        const float* input = Input;
        const float* filter = Filter;

        size_t FilterCount = (std::min)(kbatch, (OutputChannels / NCHWC) - batch * kbatch);

        size_t WorkThisPass;
        size_t OutputThisPass;

        if (StrideHeight == 1 && StrideWidth == 1) {

            WorkThisPass = WorkRemaining;
            size_t lines = OutputHeight - ph;
            if (WorkThisPass > lines) {
                WorkThisPass = lines;
            }

            OutputThisPass = OutputWidth * WorkThisPass;

        } else {
            WorkThisPass = 1;
            OutputThisPass = OutputWidth;
        }

        const size_t ICSTRIDE = 128;

        size_t cic;

        for (size_t icc = 0; icc < InputChannels; icc += cic) {

            cic = ICSTRIDE;

            if (cic > (InputChannels - icc)) {
                cic = InputChannels - icc;
            }

            unsigned Flags = 0;

            if (icc != 0 || !ZeroMode) {
                Flags |= 1;
            }

            if (icc + cic == InputChannels) {

                if (Bias != nullptr) {
                    Flags |= 2;
                }

                if (Activation->ActivationKind == MlasReluActivation) {
                    Flags |= 4;
                }
            }

            KERNEL_POINTWISE(input + NCHWC * (ih * InputWidth),
                       filter,
                       output,
                       NCHWC * StrideWidth * sizeof(float),
                       cic / NCHWC,
                       FilterCount,
                       NCHWC * InputSize * sizeof(float),
                       NCHWC * InputChannels * sizeof(float),
                       NCHWC * OutputSize * sizeof(float),
                       OutputThisPass,
                       Bias,
                       Flags);

            input += ICSTRIDE * InputSize;
            filter += NCHWC * ICSTRIDE;
        }

        output += NCHWC * OutputWidth * WorkThisPass;

        ph += WorkThisPass;
        WorkRemaining -= WorkThisPass;

        if (ph == OutputHeight) {

            Filter += NCHWC * kbatch * InputChannels;
            Output += NCHWC * kbatch * OutputSize;

            if (Bias != nullptr) {
                Bias += NCHWC * kbatch;
            }

            output = Output;

            batch++;
            ph = 0;
        }
    }
}

void
MlasConvNchwcThreaded(
    void* Context,
    int32_t Index
    )
{
    const size_t NCHWC = MlasPlatform.GetNchwcBlockSize();

    MLAS_CONV_NCHWC_WORK_BLOCK* WorkBlock = (MLAS_CONV_NCHWC_WORK_BLOCK*)Context;

    constexpr size_t HeightShapeIndex = 0;
    constexpr size_t WidthShapeIndex = 1;

    const size_t InputChannels = WorkBlock->InputChannels;
    const size_t OutputChannels = WorkBlock->OutputChannels;

    const size_t InputHeight = WorkBlock->InputShape[HeightShapeIndex];
    const size_t InputWidth = WorkBlock->InputShape[WidthShapeIndex];
    const size_t InputSize = WorkBlock->InputSize;

    const size_t OutputHeight = WorkBlock->OutputShape[HeightShapeIndex];
    const size_t OutputWidth = WorkBlock->OutputShape[WidthShapeIndex];
    const size_t OutputSize = WorkBlock->OutputSize;

    const size_t KernelHeight = WorkBlock->KernelShape[HeightShapeIndex];
    const size_t KernelWidth = WorkBlock->KernelShape[WidthShapeIndex];
    const size_t KernelSize = KernelHeight * KernelWidth;

    const size_t DilationHeight = WorkBlock->DilationShape[HeightShapeIndex];
    const size_t DilationWidth = WorkBlock->DilationShape[WidthShapeIndex];

    const size_t PaddingLeftY = WorkBlock->Padding[HeightShapeIndex];
    const size_t PaddingLeftX = WorkBlock->Padding[WidthShapeIndex];

    const size_t StrideHeight = WorkBlock->StrideShape[HeightShapeIndex];
    const size_t StrideWidth = WorkBlock->StrideShape[WidthShapeIndex];

    const size_t OutputCountLeftPadY = WorkBlock->OutputCountLeftPad[HeightShapeIndex];
    const size_t OutputCountY = WorkBlock->OutputCount[HeightShapeIndex];

    const size_t OutputCountLeftPadX = WorkBlock->OutputCountLeftPad[WidthShapeIndex];
    const size_t OutputCountX = WorkBlock->OutputCount[WidthShapeIndex];
    const size_t OutputCountRightPadX = WorkBlock->OutputCountRightPad[WidthShapeIndex];

    //
    //
    //

    const size_t kbatch = 4;
//    const size_t kbatch = 1;

    const size_t TotalWork = ((OutputChannels + (NCHWC * kbatch) - 1) / (NCHWC * kbatch)) * OutputHeight;
    const size_t WorkPerThread = TotalWork / WorkBlock->tids;
    const size_t WorkPerThreadExtra = TotalWork % WorkBlock->tids;

    size_t WorkIndex;
    size_t WorkRemaining;

    if (uint32_t(Index) < WorkPerThreadExtra) {
        WorkIndex = (WorkPerThread + 1) * Index;
        WorkRemaining = WorkPerThread + 1;
    } else {
        WorkIndex = WorkPerThread * Index + WorkPerThreadExtra;
        WorkRemaining = WorkPerThread;
    }

    size_t batch = WorkIndex / OutputHeight;
    size_t phStart = WorkIndex % OutputHeight;

    const float* Input = WorkBlock->Input;
    const float* Filter = WorkBlock->Filter;
    const float* Bias = WorkBlock->Bias;
    float* Output = WorkBlock->Output;
    const MLAS_ACTIVATION* Activation = WorkBlock->Activation;
    const bool ZeroMode = WorkBlock->ZeroMode;

    Filter += batch * NCHWC * kbatch * InputChannels * KernelSize;
    Output += batch * NCHWC * kbatch * OutputSize;

    if (Bias != nullptr) {
        Bias += batch * NCHWC * kbatch;
    }

    const size_t FilterStride = NCHWC * InputChannels * KernelSize * sizeof(float);
    const size_t OutputStride = NCHWC * OutputSize * sizeof(float);

    //
    //
    //

    while (WorkRemaining > 0) {

        size_t FilterCount = (std::min)(kbatch, (OutputChannels / NCHWC) - batch * kbatch);

        size_t WorkThisPass = WorkRemaining;
        size_t lines = OutputHeight - phStart;

        if (WorkThisPass > lines) {
            WorkThisPass = lines;
        }

        for (size_t icc = 0; icc < InputChannels; icc += NCHWC) {

            const float* input = Input + icc * InputSize;
            float* output = Output + NCHWC * phStart * OutputWidth;

            //
            //
            //

            unsigned Flags = 0;

            if (icc != 0 || !ZeroMode) {
                Flags |= 1;
            }

            if (icc + NCHWC == InputChannels) {

                if (Bias != nullptr) {
                    Flags |= 2;
                }

                if (Activation->ActivationKind == MlasReluActivation) {
                    Flags |= 4;
                }
            }

            for (size_t rrr = 0; rrr < WorkThisPass; rrr++) {

                const size_t ph = phStart + rrr;

                const float* filter = Filter + icc * NCHWC * KernelSize;

                //
                // Compute the first input row and kernel height. If this output
                // row uses padding from one or more input padding rows, then
                // adjust the kernel parameters to keep within the input bounds.
                //

                size_t ih = ph * StrideHeight - PaddingLeftY;
                size_t EffectiveKernelHeight = KernelHeight;

                if ((ph - OutputCountLeftPadY) >= OutputCountY) {

                    size_t ihStep = ih;

                    for (size_t kh = 0; kh < KernelHeight; kh++) {

                        if (ihStep >= InputHeight) {

                            if (ihStep == ih) {
                                ih += DilationHeight;
                                filter += NCHWC * NCHWC * KernelWidth;
                            }

                            EffectiveKernelHeight -= 1;
                        }

                        ihStep += DilationHeight;
                    }
                }

                KERNEL(input + NCHWC * (ih * InputWidth - PaddingLeftX),
                       filter,
                       output,
                       NCHWC * StrideWidth * sizeof(float),
                       NCHWC * DilationWidth * sizeof(float),
                       FilterCount,
                       NCHWC * DilationHeight * InputWidth * sizeof(float) - NCHWC * KernelWidth * DilationWidth * sizeof(float),
                       FilterStride,
                       OutputStride,
                       EffectiveKernelHeight,
                       KernelWidth,
                       input + NCHWC * (ih * InputWidth),
                       NCHWC * InputWidth * sizeof(float),
                       NCHWC * DilationHeight * InputWidth * sizeof(float),
                       OutputCountLeftPadX,
                       OutputCountX,
                       OutputCountRightPadX,
                       Bias,
                       Flags);

                output += NCHWC * OutputWidth;
            }
        }

        phStart += WorkThisPass;
        WorkRemaining -= WorkThisPass;

        if (phStart == OutputHeight) {

            Filter += NCHWC * kbatch * InputChannels * KernelSize;
            Output += NCHWC * kbatch * OutputSize;

            if (Bias != nullptr) {
                Bias += NCHWC * kbatch;
            }

            batch++;
            phStart = 0;
        }
    }
}

void
MlasConvNchwcNchwThreaded(
    void* Context,
    int32_t Index
    )
{
    const size_t NCHWC = MlasPlatform.GetNchwcBlockSize();

    MLAS_CONV_NCHWC_WORK_BLOCK* WorkBlock = (MLAS_CONV_NCHWC_WORK_BLOCK*)Context;

    constexpr size_t HeightShapeIndex = 0;
    constexpr size_t WidthShapeIndex = 1;

    const size_t InputChannels = WorkBlock->InputChannels;
    const size_t OutputChannels = WorkBlock->OutputChannels;

    const size_t InputHeight = WorkBlock->InputShape[HeightShapeIndex];
    const size_t InputWidth = WorkBlock->InputShape[WidthShapeIndex];
    const size_t InputSize = WorkBlock->InputSize;

    const size_t OutputHeight = WorkBlock->OutputShape[HeightShapeIndex];
    const size_t OutputWidth = WorkBlock->OutputShape[WidthShapeIndex];
    const size_t OutputSize = WorkBlock->OutputSize;

    const size_t KernelHeight = WorkBlock->KernelShape[HeightShapeIndex];
    const size_t KernelWidth = WorkBlock->KernelShape[WidthShapeIndex];
    const size_t KernelSize = KernelHeight * KernelWidth;

    const size_t DilationHeight = WorkBlock->DilationShape[HeightShapeIndex];
    const size_t DilationWidth = WorkBlock->DilationShape[WidthShapeIndex];

    const size_t PaddingLeftY = WorkBlock->Padding[HeightShapeIndex];
    const size_t PaddingLeftX = WorkBlock->Padding[WidthShapeIndex];

    const size_t StrideHeight = WorkBlock->StrideShape[HeightShapeIndex];
    const size_t StrideWidth = WorkBlock->StrideShape[WidthShapeIndex];

    const size_t OutputCountLeftPadY = WorkBlock->OutputCountLeftPad[HeightShapeIndex];
    const size_t OutputCountY = WorkBlock->OutputCount[HeightShapeIndex];

    const size_t OutputCountLeftPadX = WorkBlock->OutputCountLeftPad[WidthShapeIndex];
    const size_t OutputCountX = WorkBlock->OutputCount[WidthShapeIndex];
    const size_t OutputCountRightPadX = WorkBlock->OutputCountRightPad[WidthShapeIndex];

    //
    //
    //

    const size_t kbatch = 4;

//    const size_t TotalWork = (OutputChannels / (NCHWC * kbatch)) * OutputHeight;
    const size_t TotalWork = ((OutputChannels + (NCHWC * kbatch) - 1) / (NCHWC * kbatch)) * OutputHeight;
    const size_t WorkPerThread = TotalWork / WorkBlock->tids;
    const size_t WorkPerThreadExtra = TotalWork % WorkBlock->tids;

    size_t WorkIndex;
    size_t WorkRemaining;

    if (uint32_t(Index) < WorkPerThreadExtra) {
        WorkIndex = (WorkPerThread + 1) * Index;
        WorkRemaining = WorkPerThread + 1;
    } else {
        WorkIndex = WorkPerThread * Index + WorkPerThreadExtra;
        WorkRemaining = WorkPerThread;
    }

    size_t batch = WorkIndex / OutputHeight;
    size_t ph = WorkIndex % OutputHeight;

    const float* Input = WorkBlock->Input;
    const float* Filter = WorkBlock->Filter;
    const float* Bias = WorkBlock->Bias;
    float* Output = WorkBlock->Output;
    const MLAS_ACTIVATION* Activation = WorkBlock->Activation;
    const bool ZeroMode = WorkBlock->ZeroMode;

    Filter += batch * NCHWC * kbatch * InputChannels * KernelSize;
    Output += batch * NCHWC * kbatch * OutputSize;

    if (Bias != nullptr) {
        Bias += batch * NCHWC * kbatch;
    }

    float* output = Output + NCHWC * ph * OutputWidth;

    const size_t FilterStride = NCHWC * InputChannels * KernelSize * sizeof(float);
    const size_t OutputStride = NCHWC * OutputSize * sizeof(float);

    //
    //
    //

    while (WorkRemaining > 0) {

        const float* input = Input;
        const float* filter = Filter;

        //
        // Compute the first input row and kernel height. If this output row
        // uses padding from one or more input padding rows, then adjust the
        // kernel parameters to keep within the input bounds.
        //

        size_t ih = ph * StrideHeight - PaddingLeftY;
        size_t EffectiveKernelHeight = KernelHeight;

        if ((ph - OutputCountLeftPadY) >= OutputCountY) {

            size_t ihStep = ih;

            for (size_t kh = 0; kh < KernelHeight; kh++) {

                if (ihStep >= InputHeight) {

                    if (ihStep == ih) {
                        ih += DilationHeight;
                        filter += NCHWC * KernelWidth;
                    }

                    EffectiveKernelHeight -= 1;
                }

                ihStep += DilationHeight;
            }
        }

        size_t FilterCount = (std::min)(kbatch, (OutputChannels / NCHWC) - batch * kbatch);

        for (size_t icc = 0; icc < InputChannels; icc += 1) {

            unsigned Flags = 0;

            if (icc != 0 || !ZeroMode) {
                Flags |= 1;
            }

            if (icc + 1 == InputChannels) {

                if (Bias != nullptr) {
                    Flags |= 2;
                }

                if (Activation->ActivationKind == MlasReluActivation) {
                    Flags |= 4;
                }
            }

            KERNEL_NCHW(input + (ih * InputWidth - PaddingLeftX),
                   filter,
                   output,
                   StrideWidth * sizeof(float),
                   DilationWidth * sizeof(float),
                   FilterCount,
                   DilationHeight * InputWidth * sizeof(float) - KernelWidth * DilationWidth * sizeof(float),
                   FilterStride,
                   OutputStride,
                   EffectiveKernelHeight,
                   KernelWidth,
                   input + (ih * InputWidth),
                   InputWidth * sizeof(float),
                   DilationHeight * InputWidth * sizeof(float),
                   OutputCountLeftPadX,
                   OutputCountX,
                   OutputCountRightPadX,
                   Bias,
                   Flags);

            input += InputSize;
            filter += NCHWC * KernelSize;
        }

        output += NCHWC * OutputWidth;
        ph += 1;
        WorkRemaining -= 1;

        //
        //
        //

        if (ph == OutputHeight) {

            Filter += NCHWC * kbatch * InputChannels * KernelSize;
            Output += NCHWC * kbatch * OutputSize;

            if (Bias != nullptr) {
                Bias += NCHWC * kbatch;
            }

            output = Output;

            batch++;
            ph = 0;
        }
    }
}

void
MlasConvDepthwiseFloatThreaded(
    void* Context,
    int32_t Index
    )
{
    const size_t NCHWC = MlasPlatform.GetNchwcBlockSize();

    MLAS_CONV_NCHWC_WORK_BLOCK* WorkBlock = (MLAS_CONV_NCHWC_WORK_BLOCK*)Context;

    constexpr size_t HeightShapeIndex = 0;
    constexpr size_t WidthShapeIndex = 1;

    const size_t InputChannels = WorkBlock->InputChannels;
    const size_t OutputChannels = WorkBlock->OutputChannels;

    const size_t InputHeight = WorkBlock->InputShape[HeightShapeIndex];
    const size_t InputWidth = WorkBlock->InputShape[WidthShapeIndex];
    const size_t InputSize = WorkBlock->InputSize;

    const size_t OutputHeight = WorkBlock->OutputShape[HeightShapeIndex];
    const size_t OutputWidth = WorkBlock->OutputShape[WidthShapeIndex];
    const size_t OutputSize = WorkBlock->OutputSize;

    const size_t KernelHeight = WorkBlock->KernelShape[HeightShapeIndex];
    const size_t KernelWidth = WorkBlock->KernelShape[WidthShapeIndex];
    const size_t KernelSize = KernelHeight * KernelWidth;

    const size_t DilationHeight = WorkBlock->DilationShape[HeightShapeIndex];
    const size_t DilationWidth = WorkBlock->DilationShape[WidthShapeIndex];

    const size_t PaddingLeftY = WorkBlock->Padding[HeightShapeIndex];
    const size_t PaddingLeftX = WorkBlock->Padding[WidthShapeIndex];

    const size_t StrideHeight = WorkBlock->StrideShape[HeightShapeIndex];
    const size_t StrideWidth = WorkBlock->StrideShape[WidthShapeIndex];

    const size_t OutputCountLeftPadY = WorkBlock->OutputCountLeftPad[HeightShapeIndex];
    const size_t OutputCountY = WorkBlock->OutputCount[HeightShapeIndex];

    const size_t OutputCountLeftPadX = WorkBlock->OutputCountLeftPad[WidthShapeIndex];
    const size_t OutputCountX = WorkBlock->OutputCount[WidthShapeIndex];
    const size_t OutputCountRightPadX = WorkBlock->OutputCountRightPad[WidthShapeIndex];

    //
    //
    //

    const size_t kbatch = 1;

//    const size_t TotalWork = (OutputChannels / (NCHWC * kbatch)) * OutputHeight;
    const size_t TotalWork = ((OutputChannels + (NCHWC * kbatch) - 1) / (NCHWC * kbatch)) * OutputHeight;
    const size_t WorkPerThread = TotalWork / WorkBlock->tids;
    const size_t WorkPerThreadExtra = TotalWork % WorkBlock->tids;

    size_t WorkIndex;
    size_t WorkRemaining;

    if (uint32_t(Index) < WorkPerThreadExtra) {
        WorkIndex = (WorkPerThread + 1) * Index;
        WorkRemaining = WorkPerThread + 1;
    } else {
        WorkIndex = WorkPerThread * Index + WorkPerThreadExtra;
        WorkRemaining = WorkPerThread;
    }

    size_t batch = WorkIndex / OutputHeight;
    size_t ph = WorkIndex % OutputHeight;

    const float* Input = WorkBlock->Input;
    const float* Filter = WorkBlock->Filter;
    const float* Bias = WorkBlock->Bias;
    float* Output = WorkBlock->Output;
    const MLAS_ACTIVATION* Activation = WorkBlock->Activation;
    const bool ZeroMode = WorkBlock->ZeroMode;

    Input += batch * NCHWC * kbatch * InputSize;
    Filter += batch * NCHWC * kbatch * KernelSize;
    Output += batch * NCHWC * kbatch * OutputSize;

    if (Bias != nullptr) {
        Bias += batch * NCHWC * kbatch;
    }

    float* output = Output + NCHWC * ph * OutputWidth;

    //
    //
    //

    while (WorkRemaining > 0) {

        const float* input = Input;
        const float* filter = Filter;

        //
        // Compute the first input row and kernel height. If this output row
        // uses padding from one or more input padding rows, then adjust the
        // kernel parameters to keep within the input bounds.
        //

        size_t ih = ph * StrideHeight - PaddingLeftY;
        size_t EffectiveKernelHeight = KernelHeight;

        if ((ph - OutputCountLeftPadY) >= OutputCountY) {

            size_t ihStep = ih;

            for (size_t kh = 0; kh < KernelHeight; kh++) {

                if (ihStep >= InputHeight) {

                    if (ihStep == ih) {
                        ih += DilationHeight;
                        filter += NCHWC * KernelWidth;
                    }

                    EffectiveKernelHeight -= 1;
                }

                ihStep += DilationHeight;
            }
        }

        unsigned Flags = 0;

        if (!ZeroMode) {
            Flags |= 1;
        }

        if (Bias != nullptr) {
            Flags |= 2;
        }

        if (Activation->ActivationKind == MlasReluActivation) {
            Flags |= 4;
        }

        KERNEL_DEPTHWISE(input + NCHWC * (ih * InputWidth - PaddingLeftX),
               filter,
               output,
               NCHWC * StrideWidth * sizeof(float),
               NCHWC * DilationWidth * sizeof(float),
               NCHWC * DilationHeight * InputWidth * sizeof(float) - NCHWC * KernelWidth * DilationWidth * sizeof(float),
               EffectiveKernelHeight,
               KernelWidth,
               input + NCHWC * (ih * InputWidth),
               NCHWC * InputWidth * sizeof(float),
               NCHWC * DilationHeight * InputWidth * sizeof(float),
               OutputCountLeftPadX,
               OutputCountX,
               OutputCountRightPadX,
               Bias,
               Flags);

        output += NCHWC * OutputWidth;
        ph += 1;
        WorkRemaining -= 1;

        //
        //
        //

        if (ph == OutputHeight) {

            Input += NCHWC * kbatch * InputSize;
            Filter += NCHWC * kbatch * KernelSize;
            Output += NCHWC * kbatch * OutputSize;

            if (Bias != nullptr) {
                Bias += NCHWC * kbatch;
            }

            output = Output;

            batch++;
            ph = 0;
        }
    }
}

void
MLASCALL
MlasConvNchwc(
    size_t Dimensions,
    const int64_t* InputShape,
    const int64_t* KernelShape,
    const int64_t* DilationShape,
    const int64_t* Padding,
    const int64_t* StrideShape,
    const int64_t* OutputShape,
    size_t GroupCount,
    const float* Input,
    const float* Filter,
    const float* Bias,
    float* Output,
    const MLAS_ACTIVATION* Activation,
    bool ZeroMode
    )
{
    MLAS_CONV_NCHWC_WORK_BLOCK WorkBlock;

    //
    //
    //

    WorkBlock.Input = Input;
    WorkBlock.Output = Output;
    WorkBlock.GroupCount = GroupCount;
    WorkBlock.Filter = Filter;
    WorkBlock.Bias = Bias;
    WorkBlock.Activation = Activation;
    WorkBlock.ZeroMode = ZeroMode;

    //
    //
    //

    MlasPrepareNchwcWorkBlock(&WorkBlock, Dimensions, InputShape, KernelShape,
        DilationShape, Padding, StrideShape, OutputShape);

    //
    //
    //

    WorkBlock.tids = MlasPlatform.GetMaximumThreadCount();

    PMLAS_THREADED_ROUTINE ConvolverRoutine;

    if (WorkBlock.BatchCount > 1) __debugbreak();
    if (WorkBlock.InputChannels >= MlasPlatform.GetNchwcBlockSize()) {
        if (WorkBlock.InputChannels == WorkBlock.GroupCount &&
            WorkBlock.InputChannels == WorkBlock.OutputChannels) {
            ConvolverRoutine = MlasConvDepthwiseFloatThreaded;
        } else {
            if (WorkBlock.GroupCount > 1) __debugbreak();
            if (WorkBlock.KernelShape[0] == 1 &&
                WorkBlock.KernelShape[1] == 1 &&
                WorkBlock.Padding[0] == 0 &&
                WorkBlock.Padding[1] == 0 &&
                WorkBlock.Padding[2] == 0 &&
                WorkBlock.Padding[3] == 0 &&
                MlasPlatform.ConvPointwiseFloatKernel != nullptr) {
                ConvolverRoutine = MlasConvPointwiseThreaded;
            } else {
                ConvolverRoutine = MlasConvNchwcThreaded;
            }
        }
    } else {
        ConvolverRoutine = MlasConvNchwcNchwThreaded;
    }

    MlasExecuteThreaded(ConvolverRoutine, &WorkBlock, WorkBlock.tids);
}

void
MlasPoolNchwcThreaded(
    void* Context,
    int32_t Index
    )
/*++

Routine Description:

    This routine is invoked from a worker thread to execute a segment of a
    NCHWc pooling operation.

Arguments:

    Context - Supplies the pointer to the context for the threaded operation.

    Index - Supplies the current index of the threaded operation.

Return Value:

    None.

--*/
{
    const size_t NCHWC = MlasPlatform.GetNchwcBlockSize();

    constexpr size_t HeightShapeIndex = 0;
    constexpr size_t WidthShapeIndex = 1;

    MLAS_POOL_NCHWC_WORK_BLOCK* WorkBlock = (MLAS_POOL_NCHWC_WORK_BLOCK*)Context;

    const size_t TotalChannelCount = WorkBlock->BatchCount * WorkBlock->InputChannels;

    const size_t InputHeight = WorkBlock->InputShape[HeightShapeIndex];
    const size_t InputWidth = WorkBlock->InputShape[WidthShapeIndex];
    const size_t InputSize = WorkBlock->InputSize;

    const size_t OutputHeight = WorkBlock->OutputShape[HeightShapeIndex];
    const size_t OutputWidth = WorkBlock->OutputShape[WidthShapeIndex];
    const size_t OutputSize = WorkBlock->OutputSize;

    const size_t KernelHeight = WorkBlock->KernelShape[HeightShapeIndex];
    const size_t KernelWidth = WorkBlock->KernelShape[WidthShapeIndex];
    const size_t KernelSize = KernelHeight * KernelWidth;

    const size_t DilationHeight = WorkBlock->DilationShape[HeightShapeIndex];
    const size_t DilationWidth = WorkBlock->DilationShape[WidthShapeIndex];

    const size_t PaddingLeftY = WorkBlock->Padding[HeightShapeIndex];
    const size_t PaddingLeftX = WorkBlock->Padding[WidthShapeIndex];

    const size_t StrideHeight = WorkBlock->StrideShape[HeightShapeIndex];
    const size_t StrideWidth = WorkBlock->StrideShape[WidthShapeIndex];

    const size_t OutputCountLeftPadY = WorkBlock->OutputCountLeftPad[HeightShapeIndex];
    const size_t OutputCountY = WorkBlock->OutputCount[HeightShapeIndex];

    const size_t OutputCountLeftPadX = WorkBlock->OutputCountLeftPad[WidthShapeIndex];
    const size_t OutputCountX = WorkBlock->OutputCount[WidthShapeIndex];
    const size_t OutputCountRightPadX = WorkBlock->OutputCountRightPad[WidthShapeIndex];

    //
    //
    //

    const size_t TotalWork = ((TotalChannelCount + NCHWC - 1) / NCHWC) * OutputHeight;
    const size_t WorkPerThread = TotalWork / WorkBlock->tids;
    const size_t WorkPerThreadExtra = TotalWork % WorkBlock->tids;

    size_t WorkIndex;
    size_t WorkRemaining;

    if (uint32_t(Index) < WorkPerThreadExtra) {
        WorkIndex = (WorkPerThread + 1) * Index;
        WorkRemaining = WorkPerThread + 1;
    } else {
        WorkIndex = WorkPerThread * Index + WorkPerThreadExtra;
        WorkRemaining = WorkPerThread;
    }

    size_t ph = WorkIndex % OutputHeight;

    const float* Input = WorkBlock->Input;
    float* Output = WorkBlock->Output;

    Input += (WorkIndex / OutputHeight) * InputSize * NCHWC;
    Output += WorkIndex * NCHWC * OutputWidth;

    MLAS_POOL_FLOAT_KERNEL* PoolFloatKernel = nullptr;
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

    while (WorkRemaining > 0) {

        //
        // Compute the first input row and kernel height. If this output row
        // uses padding from one or more input padding rows, then adjust the
        // kernel parameters to keep within the input bounds.
        //

        size_t ih = ph * StrideHeight - PaddingLeftY;
        size_t EffectiveKernelHeight = KernelHeight;

        if ((ph - OutputCountLeftPadY) >= OutputCountY) {

            size_t ihStep = ih;

            for (size_t kh = 0; kh < KernelHeight; kh++) {

                if (ihStep >= InputHeight) {

                    if (ihStep == ih) {
                        ih += DilationHeight;
                    }

                    EffectiveKernelHeight -= 1;
                }

                ihStep += DilationHeight;
            }
        }

        //
        //
        //

        PoolFloatKernel(Input + NCHWC * (ih * InputWidth - PaddingLeftX),
                          Output,
                          NCHWC * StrideWidth * sizeof(float),
                          NCHWC * DilationWidth * sizeof(float),
                          NCHWC * DilationHeight * InputWidth * sizeof(float) - NCHWC * KernelWidth * DilationWidth * sizeof(float),
                          EffectiveKernelHeight,
                          KernelWidth,
                          KernelSize,
                          Input + NCHWC * (ih * InputWidth),
                          NCHWC * InputWidth * sizeof(float),
                          NCHWC * DilationHeight * InputWidth * sizeof(float),
                          OutputCountLeftPadX,
                          OutputCountX,
                          OutputCountRightPadX);

        Output += NCHWC * OutputWidth;

        ph += 1;
        WorkRemaining -= 1;

        //
        //
        //

        if (ph == OutputHeight) {
            Input += NCHWC * InputSize;
            ph = 0;
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
    MLAS_POOL_NCHWC_WORK_BLOCK WorkBlock;

    //
    //
    //

    WorkBlock.Input = Input;
    WorkBlock.Output = Output;
    WorkBlock.PoolingKind = PoolingKind;

    //
    //
    //

    MlasPrepareNchwcWorkBlock(&WorkBlock, Dimensions, InputShape, KernelShape,
        DilationShape, Padding, StrideShape, OutputShape);

    //
    //
    //

    WorkBlock.tids = MlasPlatform.GetMaximumThreadCount();

    MlasExecuteThreaded(MlasPoolNchwcThreaded, &WorkBlock, WorkBlock.tids);
}
