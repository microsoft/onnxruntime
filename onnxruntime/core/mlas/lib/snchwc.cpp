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

#if defined(__GNUC__)
#pragma GCC target "sse4.1"
#endif

//
// Define the base thread context for NCWHc convolution or pooling operations.
//

struct MLAS_NCHWC_WORK_BLOCK
{
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

struct MLAS_NCHWC_CONV_WORK_BLOCK : MLAS_NCHWC_WORK_BLOCK
{
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

struct MLAS_NCHWC_POOL_WORK_BLOCK : MLAS_NCHWC_WORK_BLOCK
{
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

    const size_t InputChannels = size_t(InputShape[0] * InputShape[1]);
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

    const size_t OutputChannels = size_t(OutputShape[0] * OutputShape[1]);
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

//
// Base implementation for neural network algorithms (convolution and pooling).
//

struct MLAS_NCHWC_NN_ALGORITHM
{
    static constexpr size_t HeightShapeIndex = 0;
    static constexpr size_t WidthShapeIndex = 1;

    const size_t NCHWC = MlasPlatform.GetNchwcBlockSize();

    //
    // Capture these values from the work block for use as local constants.
    //

    const size_t BatchCount;
    const size_t InputChannels;
    const size_t OutputChannels;
    const size_t InputHeight;
    const size_t InputWidth;
    const size_t InputSize;
    const size_t OutputHeight;
    const size_t OutputWidth;
    const size_t OutputSize;
    const size_t KernelHeight;
    const size_t KernelWidth;
    const size_t KernelSize;
    const size_t DilationHeight;
    const size_t DilationWidth;
    const size_t PaddingLeftY;
    const size_t PaddingLeftX;
    const size_t StrideHeight;
    const size_t StrideWidth;
    const size_t OutputCountLeftPadY;
    const size_t OutputCountY;
    const size_t OutputCountLeftPadX;
    const size_t OutputCountX;
    const size_t OutputCountRightPadX;

    MLAS_NCHWC_NN_ALGORITHM(const MLAS_NCHWC_WORK_BLOCK* WorkBlock) :
        BatchCount(WorkBlock->BatchCount),
        InputChannels(WorkBlock->InputChannels),
        OutputChannels(WorkBlock->OutputChannels),
        InputHeight(WorkBlock->InputShape[HeightShapeIndex]),
        InputWidth(WorkBlock->InputShape[WidthShapeIndex]),
        InputSize(WorkBlock->InputSize),
        OutputHeight(WorkBlock->OutputShape[HeightShapeIndex]),
        OutputWidth(WorkBlock->OutputShape[WidthShapeIndex]),
        OutputSize(WorkBlock->OutputSize),
        KernelHeight(WorkBlock->KernelShape[HeightShapeIndex]),
        KernelWidth(WorkBlock->KernelShape[WidthShapeIndex]),
        KernelSize(KernelHeight * KernelWidth),
        DilationHeight(WorkBlock->DilationShape[HeightShapeIndex]),
        DilationWidth(WorkBlock->DilationShape[WidthShapeIndex]),
        PaddingLeftY(WorkBlock->Padding[HeightShapeIndex]),
        PaddingLeftX(WorkBlock->Padding[WidthShapeIndex]),
        StrideHeight(WorkBlock->StrideShape[HeightShapeIndex]),
        StrideWidth(WorkBlock->StrideShape[WidthShapeIndex]),
        OutputCountLeftPadY(WorkBlock->OutputCountLeftPad[HeightShapeIndex]),
        OutputCountY(WorkBlock->OutputCount[HeightShapeIndex]),
        OutputCountLeftPadX(WorkBlock->OutputCountLeftPad[WidthShapeIndex]),
        OutputCountX(WorkBlock->OutputCount[WidthShapeIndex]),
        OutputCountRightPadX(WorkBlock->OutputCountRightPad[WidthShapeIndex])
    {
    }
};

template<typename AlgorithmType>
void
MlasNchwcThreaded(
    void* Context,
    int32_t Index
    )
{
    AlgorithmType((decltype(AlgorithmType::WorkBlock))Context).Execute(Index);
}

//
// Base implementation for convolution algorithms.
//

struct MLAS_NCHWC_CONV_ALGORITHM : MLAS_NCHWC_NN_ALGORITHM
{
    //
    // Capture these values from the work block for use as local constants.
    //

    const MLAS_NCHWC_CONV_WORK_BLOCK* WorkBlock;
    const size_t GroupCount;
    const MLAS_ACTIVATION* Activation;
    const bool ZeroMode;

    //
    // Capture the buffer pointers from the work block.
    //
    // These fields are updated as the threads step through the convolution
    // operation.
    //

    const float* Input;
    const float* Filter;
    const float* Bias;
    float* Output;

    MLAS_NCHWC_CONV_ALGORITHM(const MLAS_NCHWC_CONV_WORK_BLOCK* WorkBlock) :
        MLAS_NCHWC_NN_ALGORITHM(WorkBlock),
        WorkBlock(WorkBlock),
        GroupCount(WorkBlock->GroupCount),
        Activation(WorkBlock->Activation),
        ZeroMode(WorkBlock->ZeroMode)
    {
        Input = WorkBlock->Input;
        Filter = WorkBlock->Filter;
        Bias = WorkBlock->Bias;
        Output = WorkBlock->Output;
    }

    void
    ComputeEffectiveKernel(
        size_t ph,
        size_t FilterStride,
        const float** filter,
        size_t* ih,
        size_t* EffectiveKernelHeight
        )
    {
        //
        // Compute the first input row and kernel height. If this output row
        // uses padding from one or more input padding rows, then adjust the
        // kernel parameters to keep within the input bounds.
        //

        *ih = ph * StrideHeight - PaddingLeftY;
        *EffectiveKernelHeight = KernelHeight;

        if ((ph - OutputCountLeftPadY) >= OutputCountY) {

            size_t ihStep = *ih;

            for (size_t kh = 0; kh < KernelHeight; kh++) {

                if (ihStep >= InputHeight) {

                    if (ihStep == *ih) {
                        *ih += DilationHeight;
                        *filter += FilterStride;
                    }

                    *EffectiveKernelHeight -= 1;
                }

                ihStep += DilationHeight;
            }
        }
    }
};

//
// Base implementation for grouped convolution algorithms.
//

struct MLAS_NCHWC_GROUPED_CONV_ALGORITHM : MLAS_NCHWC_CONV_ALGORITHM
{
    //
    // Slice the convolution operation such that multiple filter blocks are
    // reused for a given set of input inside the kernel.
    //

    static constexpr size_t FilterSetSize = 4;

    const size_t FilterSetCount;

    //
    // Stores the current output line, filter cluster, and group that this thread
    // is operating on.
    //

    size_t ph;
    size_t FilterSet;
    size_t Group;
    size_t WorkRemaining;
    size_t FilterCount;

    MLAS_NCHWC_GROUPED_CONV_ALGORITHM(const MLAS_NCHWC_CONV_WORK_BLOCK* WorkBlock) :
        MLAS_NCHWC_CONV_ALGORITHM(WorkBlock),
        FilterSetCount((OutputChannels + (NCHWC * FilterSetSize) - 1) / (NCHWC * FilterSetSize))
    {
    }

    void ComputeFilterCount(void)
    {
        FilterCount = (std::min)(FilterSetSize, (OutputChannels / NCHWC) - FilterSet * FilterSetSize);
    }

    void PrepareWork(int32_t Index)
    {
        const size_t TotalWork = BatchCount * GroupCount * FilterSetCount * OutputHeight;

        const size_t WorkPerThread = TotalWork / WorkBlock->tids;
        const size_t WorkPerThreadExtra = TotalWork % WorkBlock->tids;

        size_t WorkIndex;

        if (uint32_t(Index) < WorkPerThreadExtra) {
            WorkIndex = (WorkPerThread + 1) * Index;
            WorkRemaining = WorkPerThread + 1;
        } else {
            WorkIndex = WorkPerThread * Index + WorkPerThreadExtra;
            WorkRemaining = WorkPerThread;
        }

        //
        // Extract the current batch, group, filter cluster, and output line
        // from the starting work index.
        //

        ph = WorkIndex % OutputHeight;
        const size_t BatchGroupFilterSet = WorkIndex / OutputHeight;

        FilterSet = BatchGroupFilterSet % FilterSetCount;
        const size_t BatchGroup = BatchGroupFilterSet / FilterSetCount;

        Group = BatchGroup % GroupCount;

        //
        // Advance the convolution buffer pointers to the current position
        // computed above.
        //

        Input += BatchGroup * InputChannels * InputSize;

        Output += BatchGroup * OutputChannels * OutputSize;
        Output += NCHWC * FilterSet * FilterSetSize * OutputSize;

        Filter += Group * OutputChannels * InputChannels * KernelSize;
        Filter += NCHWC * FilterSet * FilterSetSize * InputChannels * KernelSize;

        if (Bias != nullptr) {
            Bias += Group * OutputChannels;
            Bias += NCHWC * FilterSet * FilterSetSize;
        }

        //
        // Compute the number of filter set to use for the next iteration.
        //

        ComputeFilterCount();
    }

    void CompleteWork(size_t WorkThisIteration)
    {
        //
        // Adjust the amount of work remaining and check if the end of an output
        // image has been reached.
        //

        WorkRemaining -= WorkThisIteration;

        if ((ph += WorkThisIteration) == OutputHeight) {

            size_t BlockedFilterCount = NCHWC * FilterCount;

            Output += BlockedFilterCount * OutputSize;
            Filter += BlockedFilterCount * InputChannels * KernelSize;

            if (Bias != nullptr) {
                Bias += BlockedFilterCount;
            }

            //
            // Advance the input if the all filter sets have been processed.
            //

            if (++FilterSet == FilterSetCount) {

                Input += InputChannels * InputSize;

                //
                // Reset filter and bias if all groups have been processed.
                //

                if (++Group == GroupCount) {

                    Filter = WorkBlock->Filter;
                    Bias = WorkBlock->Bias;

                    Group = 0;
                }

                FilterSet = 0;
            }

            ComputeFilterCount();

            ph = 0;
        }
    }
};

//
// Implementation of the direct convolution algorithm where the input buffer is
// in NCHWc format.
//

struct MLAS_NCHWC_CONV_NCHWC_ALGORITHM : MLAS_NCHWC_GROUPED_CONV_ALGORITHM
{
    MLAS_NCHWC_CONV_NCHWC_ALGORITHM(const MLAS_NCHWC_CONV_WORK_BLOCK* WorkBlock) :
        MLAS_NCHWC_GROUPED_CONV_ALGORITHM(WorkBlock)
    {
    }

    void Execute(int32_t Index)
    {
        //
        // Setup the convolution state based on the thread index.
        //

        PrepareWork(Index);

        //
        // Loop until all of the work has been completed.
        //

        const size_t StrideWidthBytes = NCHWC * StrideWidth * sizeof(float);
        const size_t DilationWidthBytes = NCHWC * DilationWidth * sizeof(float);
        const size_t FilterStrideBytes = NCHWC * InputChannels * KernelSize * sizeof(float);
        const size_t OutputStrideBytes = NCHWC * OutputSize * sizeof(float);
        const size_t InputWidthBytes = NCHWC * InputWidth * sizeof(float);
        const size_t DilatedInputWidthBytes = NCHWC * DilationHeight * InputWidth * sizeof(float);
        const size_t InputStrideBytes = DilatedInputWidthBytes - KernelWidth * DilationWidthBytes;

        const size_t BlockedOutputWidth = NCHWC * OutputWidth;

        MLAS_CONV_FLOAT_KERNEL* Kernel = MlasPlatform.GetConvNchwcFloatKernel();

        while (WorkRemaining > 0) {

            //
            // Compute the number of output lines to process in this iteration.
            //

            size_t WorkThisIteration = (std::min)(WorkRemaining, OutputHeight - ph);

            //
            // Walk over each input image organized as a set of NCHWc blocks.
            //

            for (size_t ic = 0; ic < InputChannels; ic += NCHWC) {

                //
                //
                //

                unsigned Flags = 0;

                if (ic != 0 || !ZeroMode) {
                    Flags |= 1;
                }

                if (ic + NCHWC == InputChannels) {

                    if (Bias != nullptr) {
                        Flags |= 2;
                    }

                    if (Activation->ActivationKind == MlasReluActivation) {
                        Flags |= 4;
                    }
                }

                //
                // Apply the convolution kernel to each row of the output batch.
                //

                const float* input = Input + ic * InputSize;
                float* output = Output + ph * BlockedOutputWidth;

                for (size_t work = 0; work < WorkThisIteration; work++) {

                    //
                    // Constrain the effective kernel parameters if the output row
                    // uses one or more input padding rows.
                    //

                    const float* filter = Filter + NCHWC * ic * KernelSize;
                    size_t ih;
                    size_t EffectiveKernelHeight;

                    ComputeEffectiveKernel(ph + work, NCHWC * NCHWC * KernelWidth,
                        &filter, &ih, &EffectiveKernelHeight);

                    Kernel(input + NCHWC * (ih * InputWidth - PaddingLeftX),
                        filter, output, StrideWidthBytes, DilationWidthBytes,
                        FilterCount, InputStrideBytes, FilterStrideBytes,
                        OutputStrideBytes, EffectiveKernelHeight, KernelWidth,
                        input + NCHWC * (ih * InputWidth), InputWidthBytes,
                        DilatedInputWidthBytes, OutputCountLeftPadX, OutputCountX,
                        OutputCountRightPadX, Bias, Flags);

                    output += BlockedOutputWidth;
                }
            }

            //
            // Advance the convolution state based on the completed work.
            //

            CompleteWork(WorkThisIteration);
        }
    }
};

//
// Implementation of the direct convolution algorithm where the input buffer is
// in NCHW format.
//

struct MLAS_NCHWC_CONV_NCHW_ALGORITHM : MLAS_NCHWC_GROUPED_CONV_ALGORITHM
{
    MLAS_NCHWC_CONV_NCHW_ALGORITHM(const MLAS_NCHWC_CONV_WORK_BLOCK* WorkBlock) :
        MLAS_NCHWC_GROUPED_CONV_ALGORITHM(WorkBlock)
    {
    }

    void Execute(int32_t Index)
    {
        //
        // Setup the convolution state based on the thread index.
        //

        PrepareWork(Index);

        //
        // Loop until all of the work has been completed.
        //

        const size_t StrideWidthBytes = StrideWidth * sizeof(float);
        const size_t DilationWidthBytes = DilationWidth * sizeof(float);
        const size_t FilterStrideBytes = NCHWC * InputChannels * KernelSize * sizeof(float);
        const size_t OutputStrideBytes = NCHWC * OutputSize * sizeof(float);
        const size_t InputWidthBytes = InputWidth * sizeof(float);
        const size_t DilatedInputWidthBytes = DilationHeight * InputWidth * sizeof(float);
        const size_t InputStrideBytes = DilatedInputWidthBytes - KernelWidth * DilationWidthBytes;

        MLAS_CONV_FLOAT_KERNEL* Kernel = MlasPlatform.GetConvNchwFloatKernel();

        while (WorkRemaining > 0) {

            //
            // Constrain the effective kernel parameters if the output row uses
            // one or more input padding rows.
            //

            const float* filter = Filter;
            size_t ih;
            size_t EffectiveKernelHeight;

            ComputeEffectiveKernel(ph, NCHWC * KernelWidth, &filter, &ih,
                &EffectiveKernelHeight);

            //
            // Apply the convolution kernel to each channel of the input tensor.
            //

            const float* input = Input;
            float* output = Output + NCHWC * ph * OutputWidth;

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

                Kernel(input + (ih * InputWidth - PaddingLeftX), filter, output,
                    StrideWidthBytes, DilationWidthBytes, FilterCount, InputStrideBytes,
                    FilterStrideBytes, OutputStrideBytes, EffectiveKernelHeight,
                    KernelWidth, input + (ih * InputWidth), InputWidthBytes,
                    DilatedInputWidthBytes, OutputCountLeftPadX, OutputCountX,
                    OutputCountRightPadX, Bias, Flags);

                input += InputSize;
                filter += NCHWC * KernelSize;
            }

            //
            // Advance the convolution state based on the completed work.
            //

            CompleteWork(1);
        }
    }
};

//
// Implementation of the pointwise convolution algorithm.
//
// Pointwise convolutions have a kernel size of one. To simplify this
// implementation, no input padding is allowed, which matches typical
// usage in models.
//

struct MLAS_NCHWC_CONV_POINTWISE_ALGORITHM : MLAS_NCHWC_GROUPED_CONV_ALGORITHM
{
    MLAS_NCHWC_CONV_POINTWISE_ALGORITHM(const MLAS_NCHWC_CONV_WORK_BLOCK* WorkBlock) :
        MLAS_NCHWC_GROUPED_CONV_ALGORITHM(WorkBlock)
    {
    }

    void Execute(int32_t Index)
    {
        //
        // Setup the convolution state based on the thread index.
        //

        PrepareWork(Index);

        //
        // Loop until all of the work has been completed.
        //

        const size_t StrideWidthBytes = NCHWC * StrideWidth * sizeof(float);
        const size_t InputStrideBytes = NCHWC * InputSize * sizeof(float);
        const size_t FilterStrideBytes = NCHWC * InputChannels * sizeof(float);
        const size_t OutputStrideBytes = NCHWC * OutputSize * sizeof(float);

        MLAS_CONV_POINTWISE_FLOAT_KERNEL* Kernel = MlasPlatform.GetConvPointwiseFloatKernel();

        while (WorkRemaining > 0) {

            //
            // Compute the number of output blocks that can be computed in this
            // iteration. Unstrided convolutions can treat the input and output
            // as a single line which in turn allows the kernel to use wider
            // multiply/accumulate loops. Otherwise, a strided convolution can
            // output a single line at a time.
            //

            size_t WorkThisIteration;

            if (StrideHeight == 1 && StrideWidth == 1) {
                WorkThisIteration = (std::min)(WorkRemaining, OutputHeight - ph);
            } else {
                WorkThisIteration = 1;
            }

            const size_t OutputThisIteration = WorkThisIteration * OutputWidth;

            //
            // Apply the convolution kernel to batches of the input tensor.
            //
            // Shrinking the batch size causes a slowdown from additional
            // flushing of intermediate results to the output tensor. Extending
            // the batc sizes causes a slowdown from processor cache thrashing.
            //

            const float* input = Input + NCHWC * (ph * StrideHeight * InputWidth);
            const float* filter = Filter;
            float* output = Output + NCHWC * ph * OutputWidth;

            size_t InputChannelBatch;

            for (size_t ic = 0; ic < InputChannels; ic += InputChannelBatch) {

                constexpr size_t MaximumInputChannelBatch = 128;

                InputChannelBatch = (std::min)(InputChannels - ic, MaximumInputChannelBatch);

                unsigned Flags = 0;

                if (ic != 0 || !ZeroMode) {
                    Flags |= 1;
                }

                if (ic + InputChannelBatch == InputChannels) {

                    if (Bias != nullptr) {
                        Flags |= 2;
                    }

                    if (Activation->ActivationKind == MlasReluActivation) {
                        Flags |= 4;
                    }
                }

                Kernel(input, filter, output, StrideWidthBytes, InputChannelBatch /
                    NCHWC, FilterCount, InputStrideBytes, FilterStrideBytes,
                    OutputStrideBytes, OutputThisIteration, Bias, Flags);

                input += MaximumInputChannelBatch * InputSize;
                filter += NCHWC * MaximumInputChannelBatch;
            }

            //
            // Advance the convolution state based on the completed work.
            //

            CompleteWork(WorkThisIteration);
        }
    }
};

//
// Implementation of the depthwise separable convolution algorithm.
//
// Depthwise separable convolutions are a form of grouped convolution where
// the number of input and output channels per group are one.
//

struct MLAS_NCHWC_CONV_DEPTHWISE_ALGORITHM : MLAS_NCHWC_CONV_ALGORITHM
{
    MLAS_NCHWC_CONV_DEPTHWISE_ALGORITHM(const MLAS_NCHWC_CONV_WORK_BLOCK* WorkBlock) :
        MLAS_NCHWC_CONV_ALGORITHM(WorkBlock)
    {
    }

    void Execute(int32_t Index)
    {
        const size_t GroupBlockCount = ((GroupCount + NCHWC - 1) / NCHWC);

        const size_t TotalWork = BatchCount * GroupBlockCount * OutputHeight;

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

        //
        // Extract the current batch, group block, and output line from the
        // starting work index.
        //

        size_t ph = WorkIndex % OutputHeight;
        const size_t BatchGroup = WorkIndex / OutputHeight;

        size_t Group = BatchGroup % GroupBlockCount;

        //
        // Advance the convolution buffer pointers to the current position
        // computed above.
        //

        Input += BatchGroup * NCHWC * InputSize;
        Output += WorkIndex * NCHWC * OutputWidth;
        Filter += Group * NCHWC * KernelSize;

        if (Bias != nullptr) {
            Bias += NCHWC * Group;
        }

        //
        // Loop until all of the work has been completed.
        //

        const size_t StrideWidthBytes = NCHWC * StrideWidth * sizeof(float);
        const size_t DilationWidthBytes = NCHWC * DilationWidth * sizeof(float);
        const size_t InputWidthBytes = NCHWC * InputWidth * sizeof(float);
        const size_t DilatedInputWidthBytes = NCHWC * DilationHeight * InputWidth * sizeof(float);
        const size_t InputStrideBytes = DilatedInputWidthBytes - KernelWidth * DilationWidthBytes;

        MLAS_CONV_DEPTHWISE_FLOAT_KERNEL* Kernel = MlasPlatform.GetConvDepthwiseFloatKernel();

        while (WorkRemaining > 0) {

            //
            // Constrain the effective kernel parameters if the output row uses
            // one or more input padding rows.
            //

            const float* filter = Filter;
            size_t ih;
            size_t EffectiveKernelHeight;

            ComputeEffectiveKernel(ph, NCHWC * KernelWidth, &filter, &ih, &EffectiveKernelHeight);

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

            Kernel(Input + NCHWC * (ih * InputWidth - PaddingLeftX), filter,
                Output, StrideWidthBytes, DilationWidthBytes, InputStrideBytes,
                EffectiveKernelHeight, KernelWidth, Input + NCHWC * (ih * InputWidth),
                InputWidthBytes, DilatedInputWidthBytes, OutputCountLeftPadX,
                OutputCountX, OutputCountRightPadX, Bias, Flags);

            Output += NCHWC * OutputWidth;

            //
            // Adjust the amount of work remaining and check if the end of an output
            // image has been reached.
            //

            WorkRemaining -= 1;

            if (++ph == OutputHeight) {

                Input += NCHWC * InputSize;
                Filter += NCHWC * KernelSize;

                if (Bias != nullptr) {
                    Bias += NCHWC;
                }

                if (++Group == GroupBlockCount) {

                    Filter = WorkBlock->Filter;
                    Bias = WorkBlock->Bias;

                    Group = 0;
                }

                ph = 0;
            }
        }
    }
};

//
// Implementation of the pooling algorithm.
//

struct MLAS_NCHWC_POOL_ALGORITHM : MLAS_NCHWC_NN_ALGORITHM
{
    const MLAS_NCHWC_POOL_WORK_BLOCK* WorkBlock;

    MLAS_NCHWC_POOL_ALGORITHM(const MLAS_NCHWC_POOL_WORK_BLOCK* WorkBlock) :
        MLAS_NCHWC_NN_ALGORITHM(WorkBlock),
        WorkBlock(WorkBlock)
    {
    }

    void Execute(int32_t Index)
    {
        const size_t TotalWork = ((BatchCount * InputChannels + NCHWC - 1) / NCHWC) * OutputHeight;
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
        const size_t BatchChannel = WorkIndex / OutputHeight;

        const float* Input = WorkBlock->Input + BatchChannel * NCHWC * InputSize;
        float* Output = WorkBlock->Output + WorkIndex * NCHWC * OutputWidth;

        //
        // Loop until all of the work has been completed.
        //

        const size_t StrideWidthBytes = NCHWC * StrideWidth * sizeof(float);
        const size_t DilationWidthBytes = NCHWC * DilationWidth * sizeof(float);
        const size_t InputWidthBytes = NCHWC * InputWidth * sizeof(float);
        const size_t DilatedInputWidthBytes = NCHWC * DilationHeight * InputWidth * sizeof(float);
        const size_t InputStrideBytes = DilatedInputWidthBytes - KernelWidth * DilationWidthBytes;

        MLAS_POOL_FLOAT_KERNEL* Kernel = MlasPlatform.PoolFloatKernel[WorkBlock->PoolingKind];

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
            // Invoke the pooling kernel.
            //

            Kernel(Input + NCHWC * (ih * InputWidth - PaddingLeftX), Output,
                StrideWidthBytes, DilationWidthBytes, InputStrideBytes,
                EffectiveKernelHeight, KernelWidth, KernelSize,
                Input + NCHWC * (ih * InputWidth), InputWidthBytes,
                DilatedInputWidthBytes, OutputCountLeftPadX, OutputCountX,
                OutputCountRightPadX);

            Output += NCHWC * OutputWidth;

            //
            // Adjust the amount of work remaining and check if the end of an output
            // image has been reached.
            //

            WorkRemaining -= 1;

            if (++ph == OutputHeight) {

                Input += NCHWC * InputSize;

                ph = 0;
            }
        }
    }
};

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
    bool ZeroMode,
    MLAS_THREADPOOL* ThreadPool
    )
{
    MLAS_NCHWC_CONV_WORK_BLOCK WorkBlock;

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

    WorkBlock.InputChannels /= GroupCount;
    WorkBlock.OutputChannels /= GroupCount;

    //
    //
    //

    WorkBlock.tids = MlasGetMaximumThreadCount(ThreadPool);

    PMLAS_THREADED_ROUTINE ConvolverRoutine;

    if (WorkBlock.InputChannels >= MlasPlatform.GetNchwcBlockSize()) {
        if (WorkBlock.KernelShape[0] == 1 &&
            WorkBlock.KernelShape[1] == 1 &&
            WorkBlock.Padding[0] == 0 &&
            WorkBlock.Padding[1] == 0 &&
            WorkBlock.Padding[2] == 0 &&
            WorkBlock.Padding[3] == 0 &&
            MlasPlatform.ConvPointwiseFloatKernel != nullptr) {
            ConvolverRoutine = MlasNchwcThreaded<MLAS_NCHWC_CONV_POINTWISE_ALGORITHM>;
        } else {
            ConvolverRoutine = MlasNchwcThreaded<MLAS_NCHWC_CONV_NCHWC_ALGORITHM>;
        }
    } else if (WorkBlock.InputChannels == 1 && WorkBlock.OutputChannels == 1) {
        ConvolverRoutine = MlasNchwcThreaded<MLAS_NCHWC_CONV_DEPTHWISE_ALGORITHM>;
    } else {
        ConvolverRoutine = MlasNchwcThreaded<MLAS_NCHWC_CONV_NCHW_ALGORITHM>;
    }

    MlasExecuteThreaded(ConvolverRoutine, &WorkBlock, WorkBlock.tids, ThreadPool);
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
    float* Output,
    MLAS_THREADPOOL* ThreadPool
    )
{
    MLAS_NCHWC_POOL_WORK_BLOCK WorkBlock;

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

    WorkBlock.tids = MlasGetMaximumThreadCount(ThreadPool);

    MlasExecuteThreaded(MlasNchwcThreaded<MLAS_NCHWC_POOL_ALGORITHM>, &WorkBlock, WorkBlock.tids, ThreadPool);
}
