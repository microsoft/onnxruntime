/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    ConvolveNCHWc.cpp

Abstract:

    This module implements the convolution operation with an output format of
    NCHWc.

--*/

#include "mlasi.h"

struct MLAS_CONV2_WORK_BLOCK {
    const MLAS_CONV_PARAMETERS* Parameters;
    const float* Input;
    const float* Filter;
    const float* Bias;
    float* Output;
    int32_t tids;
    bool ZeroMode;
};

//#define TIDS 12
#define TIDS (MlasPlatform.GetMaximumThreadCount())

void
MlasConvReorderInput(
    const MLAS_CONV_PARAMETERS* Parameters,
    const float* S,
    float* D,
    size_t InputChannels
    )
{
    const size_t NCHWC = MlasPlatform.NchwcBlockSize;

    const size_t InputSize = Parameters->InputSize;

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
MlasConvReorderFilter(
    const MLAS_CONV_PARAMETERS* Parameters,
    const float* S,
    float* D
    )
{
    const size_t NCHWC = MlasPlatform.NchwcBlockSize;

    constexpr size_t HeightShapeIndex = 0;
    constexpr size_t WidthShapeIndex = 1;

    const size_t InputChannels = Parameters->InputChannels;
    const size_t OutputChannels = Parameters->FilterCount;

    const size_t KernelHeight = Parameters->KernelShape[HeightShapeIndex];
    const size_t KernelWidth = Parameters->KernelShape[WidthShapeIndex];
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
    const MLAS_CONV_PARAMETERS* Parameters,
    const float* S,
    float* D
    )
{
    const size_t NCHWC = MlasPlatform.NchwcBlockSize;

    constexpr size_t HeightShapeIndex = 0;
    constexpr size_t WidthShapeIndex = 1;

    const size_t InputChannels = Parameters->InputChannels;
    const size_t OutputChannels = Parameters->FilterCount;

    const size_t KernelHeight = Parameters->KernelShape[HeightShapeIndex];
    const size_t KernelWidth = Parameters->KernelShape[WidthShapeIndex];
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
MlasConvReorderOutput(
    const MLAS_CONV_PARAMETERS* Parameters,
    const float* S,
    float* D,
    size_t OutputChannels
    )
{
    const size_t NCHWC = MlasPlatform.NchwcBlockSize;

    const size_t OutputSize = Parameters->OutputSize;

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

#define KERNEL_1x1      MlasPlatform.SconvKernel1x1Routine
#define KERNEL          MlasPlatform.SconvKernelNchwcRoutine
#define KERNEL_NCHW     MlasPlatform.SconvKernelNchwRoutine

void
Convolver_1x1(
    void* Context,
    int32_t Index
    )
{
    const size_t NCHWC = MlasPlatform.NchwcBlockSize;

    MLAS_CONV2_WORK_BLOCK* WorkBlock = (MLAS_CONV2_WORK_BLOCK*)Context;
    const MLAS_CONV_PARAMETERS* Parameters = WorkBlock->Parameters;
    const MLAS_ACTIVATION* Activation = Parameters->Activation;
    const bool ZeroMode = WorkBlock->ZeroMode;

    constexpr size_t HeightShapeIndex = 0;
    constexpr size_t WidthShapeIndex = 1;

    const size_t InputChannels = Parameters->InputChannels;
    const size_t OutputChannels = Parameters->FilterCount;

    const size_t InputHeight = Parameters->InputShape[HeightShapeIndex];
    const size_t InputWidth = Parameters->InputShape[WidthShapeIndex];
    const size_t InputSize = Parameters->InputSize;

    const size_t OutputHeight = Parameters->OutputShape[HeightShapeIndex];
    const size_t OutputWidth = Parameters->OutputShape[WidthShapeIndex];
    const size_t OutputSize = Parameters->OutputSize;

    const size_t StrideHeight = Parameters->StrideShape[HeightShapeIndex];
    const size_t StrideWidth = Parameters->StrideShape[WidthShapeIndex];

    //
    //
    //

    const size_t kbatch = 4;

    const size_t TotalWork = ((OutputChannels + (NCHWC * kbatch) - 1) / (NCHWC * kbatch)) * OutputHeight;
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

    size_t batch = (WorkIndex / OutputHeight);

    const float* Input = WorkBlock->Input;
    const float* Filter = WorkBlock->Filter;
    const float* Bias = WorkBlock->Bias;
    float* Output = WorkBlock->Output;

    Filter += batch * NCHWC * kbatch * InputChannels;
    Output += batch * NCHWC * kbatch * OutputSize;

    if (Bias != nullptr) {
        Bias += batch * NCHWC * kbatch;
    }

    float* output = Output + NCHWC * (WorkIndex % OutputHeight) * OutputWidth;

    //
    //
    //

    while (WorkIndex < WorkIndexEnd) {

        const size_t ph = WorkIndex % OutputHeight;
        size_t ihStart = ph * StrideHeight;

        const float* input = Input;
        const float* filter = Filter;

        size_t FilterCount = (std::min)(kbatch, (OutputChannels / NCHWC) - batch * kbatch);

        size_t WorkThisPass;
        size_t OutputThisPass;
        if (StrideHeight == 1 && StrideWidth == 1) {

            WorkThisPass = WorkIndexEnd - WorkIndex;
            size_t lines = OutputHeight - (WorkIndex % OutputHeight);
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

            KERNEL_1x1(input + NCHWC * (ihStart * InputWidth),
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

        WorkIndex += WorkThisPass;

        if ((WorkIndex % OutputHeight) == 0) {
            Filter += NCHWC * kbatch * InputChannels;
            Output += NCHWC * kbatch * OutputSize;

            if (Bias != nullptr) {
                Bias += NCHWC * kbatch;
            }

            output = Output;

            batch++;
        }
    }
}

void
Convolver(
    void* Context,
    int32_t Index
    )
{
    const size_t NCHWC = MlasPlatform.NchwcBlockSize;

    MLAS_CONV2_WORK_BLOCK* WorkBlock = (MLAS_CONV2_WORK_BLOCK*)Context;
    const MLAS_CONV_PARAMETERS* Parameters = WorkBlock->Parameters;
    const MLAS_ACTIVATION* Activation = Parameters->Activation;
    const bool ZeroMode = WorkBlock->ZeroMode;

    constexpr size_t HeightShapeIndex = 0;
    constexpr size_t WidthShapeIndex = 1;

    const size_t InputChannels = Parameters->InputChannels;
    const size_t OutputChannels = Parameters->FilterCount;

    const size_t InputHeight = Parameters->InputShape[HeightShapeIndex];
    const size_t InputWidth = Parameters->InputShape[WidthShapeIndex];
    const size_t InputSize = Parameters->InputSize;

    const size_t OutputHeight = Parameters->OutputShape[HeightShapeIndex];
    const size_t OutputWidth = Parameters->OutputShape[WidthShapeIndex];
    const size_t OutputSize = Parameters->OutputSize;

    const size_t PaddingLeftY = Parameters->Padding[HeightShapeIndex];
    const size_t PaddingLeftX = Parameters->Padding[WidthShapeIndex];

    const size_t StrideHeight = Parameters->StrideShape[HeightShapeIndex];
    const size_t StrideWidth = Parameters->StrideShape[WidthShapeIndex];

    const size_t DilationHeight = Parameters->DilationShape[HeightShapeIndex];
    const size_t DilationWidth = Parameters->DilationShape[WidthShapeIndex];

    const size_t KernelHeight = Parameters->KernelShape[HeightShapeIndex];
    const size_t KernelWidth = Parameters->KernelShape[WidthShapeIndex];
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

    //
    //
    //

    const size_t kbatch = 4;
//    const size_t kbatch = 1;

//    const size_t TotalWork = (OutputChannels / (NCHWC * kbatch)) * OutputHeight;
    const size_t TotalWork = ((OutputChannels + (NCHWC * kbatch) - 1) / (NCHWC * kbatch)) * OutputHeight;
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

    size_t batch = (WorkIndex / OutputHeight);

    const float* Input = WorkBlock->Input;
    const float* Filter = WorkBlock->Filter;
    const float* Bias = WorkBlock->Bias;
    float* Output = WorkBlock->Output;

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

    while (WorkIndex < WorkIndexEnd) {

        size_t FilterCount = (std::min)(kbatch, (OutputChannels / NCHWC) - batch * kbatch);

        const size_t phStart = WorkIndex % OutputHeight;

        size_t WorkThisPass;
        WorkThisPass = WorkIndexEnd - WorkIndex;
        size_t lines = OutputHeight - phStart;
        if (WorkThisPass > lines) {
            WorkThisPass = lines;
        }

        for (size_t icc = 0; icc < InputChannels; icc += NCHWC) {

            float* output = Output + NCHWC * phStart * OutputWidth;

            for (size_t rrr = 0; rrr < WorkThisPass; rrr++) {

                const size_t ph = phStart + rrr;
        //printf("doing %zd, %zd %zd  %zd\n", ph, WorkIndex, OutputHeight, OutputCountPadLeftH);

                size_t ihStart = ph * StrideHeight - PaddingLeftY;

                const float* input = Input + icc * InputSize;
                const float* filter = Filter + icc * NCHWC * KernelSize;

                size_t mykh = KernelHeight;

                if (ph < OutputCountPadLeftH || ph >= OutputCountWithLeftPaddingH) {

                    size_t ih = ihStart;

                    for (size_t kh = 0; kh < KernelHeight; kh++) {

                        if (ih >= InputHeight) {

                            if (ih == ihStart) {
                                ihStart += DilationHeight;
                                filter += NCHWC * NCHWC * KernelWidth;
                            }

                            mykh -= 1;
                        }

                        ih += DilationHeight;
                    }
                }

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

                KERNEL(input + NCHWC * (ihStart * InputWidth - PaddingLeftX),
                       filter,
                       output,
                       NCHWC * StrideWidth * sizeof(float),
                       NCHWC * DilationWidth * sizeof(float),
                       FilterCount,
                       NCHWC * DilationHeight * InputWidth * sizeof(float) - NCHWC * KernelWidth * DilationWidth * sizeof(float),
                       FilterStride,
                       OutputStride,
                       mykh, // KernelHeight,
                       KernelWidth,
                       input + NCHWC * (ihStart * InputWidth),
                       NCHWC * InputWidth * sizeof(float),
                       NCHWC * DilationHeight * InputWidth * sizeof(float),
                       OutputCountPadLeftW,
                       OutputCountPadNoneW,
                       OutputCountPadRightW,
                       Bias,
                       Flags);

                input += NCHWC * InputSize;
                filter += NCHWC * NCHWC * KernelSize;

                output += NCHWC * OutputWidth;
            }
        }

        WorkIndex += WorkThisPass;

        if ((WorkIndex % OutputHeight) == 0) {
            Filter += NCHWC * kbatch * InputChannels * KernelSize;
            Output += NCHWC * kbatch * OutputSize;

            if (Bias != nullptr) {
                Bias += NCHWC * kbatch;
            }

            batch++;
        }
    }
}

void
Convolver_NCHW(
    void* Context,
    int32_t Index
    )
{
    const size_t NCHWC = MlasPlatform.NchwcBlockSize;

    MLAS_CONV2_WORK_BLOCK* WorkBlock = (MLAS_CONV2_WORK_BLOCK*)Context;
    const MLAS_CONV_PARAMETERS* Parameters = WorkBlock->Parameters;
    const MLAS_ACTIVATION* Activation = Parameters->Activation;
    const bool ZeroMode = WorkBlock->ZeroMode;

    constexpr size_t HeightShapeIndex = 0;
    constexpr size_t WidthShapeIndex = 1;

    const size_t InputChannels = Parameters->InputChannels;
    const size_t OutputChannels = Parameters->FilterCount;

    const size_t InputHeight = Parameters->InputShape[HeightShapeIndex];
    const size_t InputWidth = Parameters->InputShape[WidthShapeIndex];
    const size_t InputSize = Parameters->InputSize;

    const size_t OutputHeight = Parameters->OutputShape[HeightShapeIndex];
    const size_t OutputWidth = Parameters->OutputShape[WidthShapeIndex];
    const size_t OutputSize = Parameters->OutputSize;

    const size_t PaddingLeftY = Parameters->Padding[HeightShapeIndex];
    const size_t PaddingLeftX = Parameters->Padding[WidthShapeIndex];

    const size_t StrideHeight = Parameters->StrideShape[HeightShapeIndex];
    const size_t StrideWidth = Parameters->StrideShape[WidthShapeIndex];

    const size_t DilationHeight = Parameters->DilationShape[HeightShapeIndex];
    const size_t DilationWidth = Parameters->DilationShape[WidthShapeIndex];

    const size_t KernelHeight = Parameters->KernelShape[HeightShapeIndex];
    const size_t KernelWidth = Parameters->KernelShape[WidthShapeIndex];
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

    //
    //
    //

    const size_t kbatch = 4;

//    const size_t TotalWork = (OutputChannels / (NCHWC * kbatch)) * OutputHeight;
    const size_t TotalWork = ((OutputChannels + (NCHWC * kbatch) - 1) / (NCHWC * kbatch)) * OutputHeight;
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

    size_t batch = (WorkIndex / OutputHeight);

    const float* Input = WorkBlock->Input;
    const float* Filter = WorkBlock->Filter;
    const float* Bias = WorkBlock->Bias;
    float* Output = WorkBlock->Output;

    Filter += batch * NCHWC * kbatch * InputChannels * KernelSize;
    Output += batch * NCHWC * kbatch * OutputSize;

    if (Bias != nullptr) {
        Bias += batch * NCHWC * kbatch;
    }

    float* output = Output + NCHWC * (WorkIndex % OutputHeight) * OutputWidth;

    const size_t FilterStride = NCHWC * InputChannels * KernelSize * sizeof(float);
    const size_t OutputStride = NCHWC * OutputSize * sizeof(float);

    //
    //
    //

    while (WorkIndex < WorkIndexEnd) {

        const size_t ph = WorkIndex % OutputHeight;
//printf("doing %zd, %zd %zd\n", ph, WorkIndex, OutputHeight);

        size_t ihStart = ph * StrideHeight - PaddingLeftY;

        const float* input = Input;
        const float* filter = Filter;

        size_t mykh = KernelHeight;

        if (ph < OutputCountPadLeftH || ph >= OutputCountWithLeftPaddingH) {

            size_t ih = ihStart;

            for (size_t kh = 0; kh < KernelHeight; kh++) {

                if (ih >= InputHeight) {

                    if (ih == ihStart) {
                        ihStart += DilationHeight;
                        filter += NCHWC * KernelWidth;
                    }

                    mykh -= 1;
                }

                ih += DilationHeight;
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

            KERNEL_NCHW(input + (ihStart * InputWidth - PaddingLeftX),
                   filter,
                   output,
                   StrideWidth * sizeof(float),
                   DilationWidth * sizeof(float),
                   FilterCount,
                   DilationHeight * InputWidth * sizeof(float) - KernelWidth * DilationWidth * sizeof(float),
                   FilterStride,
                   OutputStride,
                   mykh, // KernelHeight,
                   KernelWidth,
                   input + (ihStart * InputWidth),
                   InputWidth * sizeof(float),
                   DilationHeight * InputWidth * sizeof(float),
                   OutputCountPadLeftW,
                   OutputCountPadNoneW,
                   OutputCountPadRightW,
                   Bias,
                   Flags);

            input += InputSize;
            filter += NCHWC * KernelSize;
        }

        output += NCHWC * OutputWidth;

        WorkIndex++;

        if ((WorkIndex % OutputHeight) == 0) {
            Filter += NCHWC * kbatch * InputChannels * KernelSize;
            Output += NCHWC * kbatch * OutputSize;

            if (Bias != nullptr) {
                Bias += NCHWC * kbatch;
            }

            output = Output;

            batch++;
        }
    }
}

void
MLASCALL
MlasConvNchwc(
    const MLAS_CONV_PARAMETERS* Parameters,
    const float* Input,
    const float* Filter,
    const float* Bias,
    float* Output,
    bool ZeroMode
    )
{
    const size_t NCHWC = MlasPlatform.NchwcBlockSize;

    MLAS_CONV2_WORK_BLOCK wb;

    if (Parameters->BatchCount > 1 || Parameters->GroupCount > 1) __debugbreak();

    wb.Parameters = Parameters;
    wb.Input = Input;
    wb.Filter = Filter;
    wb.Bias = Bias;
    wb.Output = Output;
    wb.tids = TIDS;
    wb.ZeroMode = ZeroMode;

    PMLAS_THREADED_ROUTINE ConvolverRoutine;

    if (Parameters->InputChannels >= NCHWC) {
        if (Parameters->KernelShape[0] == 1 &&
            Parameters->KernelShape[1] == 1 &&
            Parameters->Padding[0] == 0 &&
            Parameters->Padding[1] == 0 &&
            Parameters->Padding[2] == 0 &&
            Parameters->Padding[3] == 0 &&
            MlasPlatform.SconvKernel1x1Routine != nullptr) {
            ConvolverRoutine = Convolver_1x1;
        } else {
            ConvolverRoutine = Convolver;
        }
    } else {
        ConvolverRoutine = Convolver_NCHW;
    }

    MlasExecuteThreaded(ConvolverRoutine, &wb, TIDS);
}
