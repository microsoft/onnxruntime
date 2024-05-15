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
// Define the base thread context for NCWHc convolution or pooling operations.
//

struct MLAS_NCHWC_WORK_BLOCK
{
    ptrdiff_t tids;
    size_t BatchCount;
    size_t InputChannels;
    size_t InputShape[2];
    size_t InputSize;
    size_t OutputChannels;
    size_t OutputShape[2];
    size_t OutputSize;
    size_t KernelShape[2];
    size_t DilationShape[2];
    size_t Padding[4];
    size_t StrideShape[2];
    size_t OutputCountLeftPad[2];
    size_t OutputCount[2];
    size_t OutputCountRightPad[2];
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
};

//
// Define the convolution kernel flags.
//

#define MLAS_CONV_KERNEL_FLAG_ACCUMULATE_OUTPUT     0x00000001
#define MLAS_CONV_KERNEL_FLAG_BIAS_ADDITION         0x00000002
#define MLAS_CONV_KERNEL_FLAG_RELU_ACTIVATION       0x00000004
#define MLAS_CONV_KERNEL_FLAG_OTHER_ACTIVATION      0x00000008

size_t
MLASCALL
MlasNchwcGetBlockSize(
    void
    )
/*++

Routine Description:

    This routine returns the NCHWc block size for the platform.

Arguments:

    None.

Return Value:

    Returns the NCHWc block size for the platform. If NCHWc support is not
    available for the platform, then returns one.

    N.B. Using the value one as the flag to indicate no support avoids compiler
    warnings in optimized builds when using this value in division or modulus
    math.

--*/
{
#if defined(MLAS_TARGET_AMD64) || defined(MLAS_TARGET_LARCH64)
    return GetMlasPlatform().NchwcBlockSize;
#else
    return 1;
#endif
}

void
MlasNchwcPrepareWorkBlock(
    MLAS_NCHWC_WORK_BLOCK* WorkBlock,
    const int64_t* InputShape,
    const int64_t* KernelShape,
    const int64_t* DilationShape,
    const int64_t* Padding,
    const int64_t* StrideShape,
    const int64_t* OutputShape
    )
/*++

Routine Description:

    This routine prepares for a convolution or pooling operation by computing
    required parameters given the shape attributes.

Arguments:

    WorkBlock - Supplies the structure that contains the common convolution
        and pooling parameters.

    InputShape - Supplies the shape of the input tensor.

    KernelShape - Supplies the shape of the kernel transform.

    DilationShape - Supplies the shape of the dilation.

    Padding - Supplies the number of padding elements at the edge of the input
        tensor.

    StrideShape - Supplies the shape of the stride.

    OutputShape - Supplies the shape of the output tensor.

Return Value:

    None.

--*/
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
    bool CanFlattenShape = true;

    for (size_t dim = 0; dim < 2; dim++) {

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

        CanFlattenShape &= (WorkBlock->DilationShape[dim] == 1);

        if (Padding != nullptr) {
            WorkBlock->Padding[dim] = size_t(Padding[dim]);
            WorkBlock->Padding[dim + 2] = size_t(Padding[dim + 2]);
        } else {
            WorkBlock->Padding[dim] = 0;
            WorkBlock->Padding[dim + 2] = 0;
        }

        CanFlattenShape &= (WorkBlock->Padding[dim] == 0 && WorkBlock->Padding[dim + 2] == 0);

        if (StrideShape != nullptr) {
            WorkBlock->StrideShape[dim] = size_t(StrideShape[dim]);
        } else {
            WorkBlock->StrideShape[dim] = 1;
        }

        CanFlattenShape &= (WorkBlock->StrideShape[dim] == 1);
    }

    WorkBlock->InputSize = InputSize;
    WorkBlock->OutputSize = OutputSize;

    //
    // Detect operations where the kernel is using the entire input width,
    // has strides and dilations set to one, and no padding. These operations
    // are transformed from outputting [N][1] to [1][N] by flattening the
    // operation to a single line using striding equal to the original width.
    //
    // With the originally shape, the NCHWc kernels would process a single
    // output per output line. After reshaping, the NCHWc kernels are able to
    // process multiple outputs per output line which typically performs better,
    // despite potentially using fewer threads due to the decreased output
    // height.
    //

    if (CanFlattenShape && (WorkBlock->InputShape[1] == WorkBlock->KernelShape[1])) {

        WorkBlock->StrideShape[1] = WorkBlock->InputShape[1];

        WorkBlock->InputShape[1] *= WorkBlock->InputShape[0];
        WorkBlock->InputShape[0] = 1;

        WorkBlock->OutputShape[1] *= WorkBlock->OutputShape[0];
        WorkBlock->OutputShape[0] = 1;

        WorkBlock->KernelShape[1] *= WorkBlock->KernelShape[0];
        WorkBlock->KernelShape[0] = 1;
    }

    //
    // Compute the number of output elements affected by left and right padding.
    //

    for (size_t dim = 0; dim < 2; dim++) {

        const size_t SpanValue =
            WorkBlock->DilationShape[dim] * (WorkBlock->KernelShape[dim] - 1) + 1;
        const size_t StrideValue = WorkBlock->StrideShape[dim];
        const size_t PaddingLeftValue = WorkBlock->Padding[dim];
        const size_t InputValue = WorkBlock->InputShape[dim];

        size_t OutputCountWithLeftPad;

        if (InputValue + PaddingLeftValue >= SpanValue) {
            OutputCountWithLeftPad = (InputValue + PaddingLeftValue - SpanValue) / StrideValue + 1;
        } else {
            OutputCountWithLeftPad = 0;
        }

        size_t OutputCountLeftPad = (PaddingLeftValue + StrideValue - 1) / StrideValue;

        if (OutputCountLeftPad > OutputCountWithLeftPad) {
            OutputCountLeftPad = OutputCountWithLeftPad;
        }

        const size_t OutputValue = WorkBlock->OutputShape[dim];

        WorkBlock->OutputCountLeftPad[dim] = OutputCountLeftPad;
        WorkBlock->OutputCount[dim] = OutputCountWithLeftPad - OutputCountLeftPad;
        WorkBlock->OutputCountRightPad[dim] = OutputValue - OutputCountWithLeftPad;
    }
}

//
// Base implementation for neural network algorithms (convolution and pooling).
//

struct MLAS_NCHWC_NN_ALGORITHM
{
    static constexpr size_t HeightShapeIndex = 0;
    static constexpr size_t WidthShapeIndex = 1;

    const size_t BlockSize = MlasNchwcGetBlockSize();

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

constexpr size_t MLAS_NCHWC_NN_ALGORITHM::HeightShapeIndex;
constexpr size_t MLAS_NCHWC_NN_ALGORITHM::WidthShapeIndex;

template<typename AlgorithmType>
void
MlasNchwcThreaded(
    void* Context,
    ptrdiff_t Index
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
    const MLAS_ACTIVATION_KIND ActivationKind;
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
        ActivationKind(Activation->ActivationKind),
        ZeroMode(WorkBlock->ZeroMode)
    {
        Input = WorkBlock->Input;
        Filter = WorkBlock->Filter;
        Bias = WorkBlock->Bias;
        Output = WorkBlock->Output;
    }

    unsigned
    ComputeKernelFlags(
        size_t ic,
        size_t ChannelCount
        )
    {
        unsigned KernelFlags = 0;

        //
        // Accumulate into the output buffer if this isn't the first input
        // channel contributing to the output element or if the caller has
        // requested that the output buffer not be zero initialized (Conv/Sum
        // fusion).
        //

        if (ic != 0 || !ZeroMode) {
            KernelFlags |= MLAS_CONV_KERNEL_FLAG_ACCUMULATE_OUTPUT;
        }

        if (ic + ChannelCount == InputChannels) {

            //
            // Add the bias buffer into the output buffer if necessary.
            //

            if (Bias != nullptr) {
                KernelFlags |= MLAS_CONV_KERNEL_FLAG_BIAS_ADDITION;
            }

            //
            // Test for fused ReLU activation or other types of activation run
            // outside of the convolution kernel.
            //

            if (ActivationKind == MlasReluActivation) {
                KernelFlags |= MLAS_CONV_KERNEL_FLAG_RELU_ACTIVATION;
            } else if (ActivationKind != MlasIdentityActivation) {
                KernelFlags |= MLAS_CONV_KERNEL_FLAG_OTHER_ACTIVATION;
            }
        }

        return KernelFlags;
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

    void
    DoActivation(
        float* output,
        size_t FilterCount,
        size_t BlockedOutputWidth
        )
    {
        //
        // Invoke activation doing an inplace update.
        //
        // The width of the output matrix is the number of written output
        // elements. Pointwise convolution may write multiple logical rows
        // at once, so this output count may be greater than OutputWidth.
        //
        // The convolution kernels write to one or more output positions
        // across NCHWc output planes, so the stride is set to the blocked
        // output size instead of the output width as done in NCHW convolution.
        //

        MlasActivation(Activation, output, nullptr, FilterCount,
            BlockedOutputWidth, BlockSize * OutputSize);
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
        FilterSetCount((OutputChannels + (BlockSize * FilterSetSize) - 1) / (BlockSize * FilterSetSize))
    {
    }

    void ComputeFilterCount(void)
    {
        FilterCount = std::min(FilterSetSize, (OutputChannels / BlockSize) - FilterSet * FilterSetSize);
    }

    void PrepareWork(ptrdiff_t Index)
    {
        const size_t TotalWork = BatchCount * GroupCount * FilterSetCount * OutputHeight;

        size_t WorkIndex;

        MlasPartitionWork(Index, WorkBlock->tids, TotalWork, &WorkIndex, &WorkRemaining);

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
        Output += BlockSize * FilterSet * FilterSetSize * OutputSize;

        Filter += Group * OutputChannels * InputChannels * KernelSize;
        Filter += BlockSize * FilterSet * FilterSetSize * InputChannels * KernelSize;

        if (Bias != nullptr) {
            Bias += Group * OutputChannels;
            Bias += BlockSize * FilterSet * FilterSetSize;
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

            size_t BlockedFilterCount = BlockSize * FilterCount;

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

constexpr size_t MLAS_NCHWC_GROUPED_CONV_ALGORITHM::FilterSetSize;

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

    void Execute(ptrdiff_t Index)
    {
        //
        // Setup the convolution state based on the thread index.
        //

        PrepareWork(Index);

        //
        // Loop until all of the work has been completed.
        //

        const size_t StrideWidthBytes = BlockSize * StrideWidth * sizeof(float);
        const size_t DilationWidthBytes = BlockSize * DilationWidth * sizeof(float);
        const size_t FilterStrideBytes = BlockSize * InputChannels * KernelSize * sizeof(float);
        const size_t OutputStrideBytes = BlockSize * OutputSize * sizeof(float);
        const size_t InputWidthBytes = BlockSize * InputWidth * sizeof(float);
        const size_t DilatedInputWidthBytes = BlockSize * DilationHeight * InputWidth * sizeof(float);
        const size_t InputStrideBytes = DilatedInputWidthBytes - KernelWidth * DilationWidthBytes;

        const size_t BlockedOutputWidth = BlockSize * OutputWidth;

#if defined(MLAS_TARGET_AMD64) || defined(MLAS_TARGET_LARCH64)
        MLAS_CONV_FLOAT_KERNEL* Kernel = GetMlasPlatform().ConvNchwcFloatKernel;
#else
        MLAS_CONV_FLOAT_KERNEL* Kernel = MlasConvNchwcFloatKernel;
#endif

        while (WorkRemaining > 0) {

            //
            // Compute the number of output lines to process in this iteration.
            //

            size_t WorkThisIteration = std::min(WorkRemaining, OutputHeight - ph);

            //
            // Walk over each input image organized as a set of NCHWc blocks.
            //

            for (size_t ic = 0; ic < InputChannels; ic += BlockSize) {

                unsigned KernelFlags = ComputeKernelFlags(ic, BlockSize);

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

                    const float* filter = Filter + BlockSize * ic * KernelSize;
                    size_t ih;
                    size_t EffectiveKernelHeight;

                    ComputeEffectiveKernel(ph + work, BlockSize * BlockSize * KernelWidth,
                        &filter, &ih, &EffectiveKernelHeight);

                    //
                    // Invoke the convolution kernel.
                    //

                    Kernel(input + BlockSize * (ih * InputWidth - PaddingLeftX),
                        filter, output, StrideWidthBytes, DilationWidthBytes,
                        FilterCount, InputStrideBytes, FilterStrideBytes,
                        OutputStrideBytes, EffectiveKernelHeight, KernelWidth,
                        input + BlockSize * (ih * InputWidth), InputWidthBytes,
                        DilatedInputWidthBytes, OutputCountLeftPadX, OutputCountX,
                        OutputCountRightPadX, Bias, KernelFlags);

                    //
                    // Test for fused non-ReLU activation.
                    //

                    if ((KernelFlags & MLAS_CONV_KERNEL_FLAG_OTHER_ACTIVATION) != 0) {
                        DoActivation(output, FilterCount, BlockedOutputWidth);
                    }

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

    void Execute(ptrdiff_t Index)
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
        const size_t FilterStrideBytes = BlockSize * InputChannels * KernelSize * sizeof(float);
        const size_t OutputStrideBytes = BlockSize * OutputSize * sizeof(float);
        const size_t InputWidthBytes = InputWidth * sizeof(float);
        const size_t DilatedInputWidthBytes = DilationHeight * InputWidth * sizeof(float);
        const size_t InputStrideBytes = DilatedInputWidthBytes - KernelWidth * DilationWidthBytes;

        const size_t BlockedOutputWidth = BlockSize * OutputWidth;

#if defined(MLAS_TARGET_AMD64) || defined(MLAS_TARGET_LARCH64)
        MLAS_CONV_FLOAT_KERNEL* Kernel = GetMlasPlatform().ConvNchwFloatKernel;
#else
        MLAS_CONV_FLOAT_KERNEL* Kernel = MlasConvNchwFloatKernel;
#endif

        while (WorkRemaining > 0) {

            //
            // Constrain the effective kernel parameters if the output row uses
            // one or more input padding rows.
            //

            const float* filter = Filter;
            size_t ih;
            size_t EffectiveKernelHeight;

            ComputeEffectiveKernel(ph, BlockSize * KernelWidth, &filter, &ih,
                &EffectiveKernelHeight);

            //
            // Apply the convolution kernel to each channel of the input tensor.
            //

            const float* input = Input;
            float* output = Output + BlockSize * ph * OutputWidth;

            for (size_t ic = 0; ic < InputChannels; ic += 1) {

                unsigned KernelFlags = ComputeKernelFlags(ic, 1);

                //
                // Invoke the convolution kernel.
                //

                Kernel(input + (ih * InputWidth - PaddingLeftX), filter, output,
                    StrideWidthBytes, DilationWidthBytes, FilterCount, InputStrideBytes,
                    FilterStrideBytes, OutputStrideBytes, EffectiveKernelHeight,
                    KernelWidth, input + (ih * InputWidth), InputWidthBytes,
                    DilatedInputWidthBytes, OutputCountLeftPadX, OutputCountX,
                    OutputCountRightPadX, Bias, KernelFlags);

                //
                // Test for fused non-ReLU activation.
                //

                if ((KernelFlags & MLAS_CONV_KERNEL_FLAG_OTHER_ACTIVATION) != 0) {
                    DoActivation(output, FilterCount, BlockedOutputWidth);
                }

                input += InputSize;
                filter += BlockSize * KernelSize;
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

    void Execute(ptrdiff_t Index)
    {
        //
        // Setup the convolution state based on the thread index.
        //

        PrepareWork(Index);

        //
        // Loop until all of the work has been completed.
        //

        const size_t StrideWidthBytes = BlockSize * StrideWidth * sizeof(float);
        const size_t InputStrideBytes = BlockSize * InputSize * sizeof(float);
        const size_t FilterStrideBytes = BlockSize * InputChannels * sizeof(float);
        const size_t OutputStrideBytes = BlockSize * OutputSize * sizeof(float);

#if defined(MLAS_TARGET_AMD64) || defined(MLAS_TARGET_LARCH64)
        MLAS_CONV_POINTWISE_FLOAT_KERNEL* Kernel = GetMlasPlatform().ConvPointwiseFloatKernel;
#else
        MLAS_CONV_POINTWISE_FLOAT_KERNEL* Kernel = MlasConvPointwiseFloatKernel;
#endif

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
                WorkThisIteration = std::min(WorkRemaining, OutputHeight - ph);
            } else {
                WorkThisIteration = 1;
            }

            const size_t OutputThisIteration = WorkThisIteration * OutputWidth;

            //
            // Apply the convolution kernel to batches of the input tensor.
            //
            // Shrinking the batch size causes a slowdown from additional
            // flushing of intermediate results to the output tensor. Extending
            // the batch sizes causes a slowdown from processor cache thrashing.
            //

            const float* input = Input + BlockSize * (ph * StrideHeight * InputWidth);
            const float* filter = Filter;
            float* output = Output + BlockSize * ph * OutputWidth;

            size_t InputChannelBatch;

            for (size_t ic = 0; ic < InputChannels; ic += InputChannelBatch) {

                constexpr size_t MaximumInputChannelBatch = 128;

                InputChannelBatch = std::min(InputChannels - ic, MaximumInputChannelBatch);

                unsigned KernelFlags = ComputeKernelFlags(ic, InputChannelBatch);

                //
                // Invoke the convolution kernel.
                //

                Kernel(input, filter, output, StrideWidthBytes, InputChannelBatch /
                    BlockSize, FilterCount, InputStrideBytes, FilterStrideBytes,
                    OutputStrideBytes, OutputThisIteration, Bias, KernelFlags);

                //
                // Test for fused non-ReLU activation.
                //

                if ((KernelFlags & MLAS_CONV_KERNEL_FLAG_OTHER_ACTIVATION) != 0) {
                    DoActivation(output, FilterCount, BlockSize * OutputThisIteration);
                }

                input += MaximumInputChannelBatch * InputSize;
                filter += BlockSize * MaximumInputChannelBatch;
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

    void Execute(ptrdiff_t Index)
    {
        const size_t GroupBlockCount = ((GroupCount + BlockSize - 1) / BlockSize);

        const size_t TotalWork = BatchCount * GroupBlockCount * OutputHeight;

        size_t WorkIndex;
        size_t WorkRemaining;

        MlasPartitionWork(Index, WorkBlock->tids, TotalWork, &WorkIndex, &WorkRemaining);

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

        Input += BatchGroup * BlockSize * InputSize;
        Output += WorkIndex * BlockSize * OutputWidth;
        Filter += Group * BlockSize * KernelSize;

        if (Bias != nullptr) {
            Bias += BlockSize * Group;
        }

        //
        // Loop until all of the work has been completed.
        //

        const size_t StrideWidthBytes = BlockSize * StrideWidth * sizeof(float);
        const size_t DilationWidthBytes = BlockSize * DilationWidth * sizeof(float);
        const size_t InputWidthBytes = BlockSize * InputWidth * sizeof(float);
        const size_t DilatedInputWidthBytes = BlockSize * DilationHeight * InputWidth * sizeof(float);
        const size_t InputStrideBytes = DilatedInputWidthBytes - KernelWidth * DilationWidthBytes;

        const size_t BlockedOutputWidth = BlockSize * OutputWidth;

#if defined(MLAS_TARGET_AMD64) || defined(MLAS_TARGET_LARCH64)
        MLAS_CONV_DEPTHWISE_FLOAT_KERNEL* Kernel = GetMlasPlatform().ConvDepthwiseFloatKernel;
#else
        MLAS_CONV_DEPTHWISE_FLOAT_KERNEL* Kernel = MlasConvDepthwiseFloatKernel;
#endif

        unsigned KernelFlags = ComputeKernelFlags(0, InputChannels);

        while (WorkRemaining > 0) {

            //
            // Constrain the effective kernel parameters if the output row uses
            // one or more input padding rows.
            //

            const float* filter = Filter;
            size_t ih;
            size_t EffectiveKernelHeight;

            ComputeEffectiveKernel(ph, BlockSize * KernelWidth, &filter, &ih, &EffectiveKernelHeight);

            //
            // Invoke the convolution kernel.
            //

            Kernel(Input + BlockSize * (ih * InputWidth - PaddingLeftX), filter,
                Output, StrideWidthBytes, DilationWidthBytes, InputStrideBytes,
                EffectiveKernelHeight, KernelWidth, Input + BlockSize * (ih * InputWidth),
                InputWidthBytes, DilatedInputWidthBytes, OutputCountLeftPadX,
                OutputCountX, OutputCountRightPadX, Bias, KernelFlags);

            //
            // Test for fused non-ReLU activation.
            //

            if ((KernelFlags & MLAS_CONV_KERNEL_FLAG_OTHER_ACTIVATION) != 0) {
                DoActivation(Output, 1, BlockedOutputWidth);
            }

            Output += BlockedOutputWidth;

            //
            // Adjust the amount of work remaining and check if the end of an
            // output image has been reached.
            //

            WorkRemaining -= 1;

            if (++ph == OutputHeight) {

                Input += BlockSize * InputSize;
                Filter += BlockSize * KernelSize;

                if (Bias != nullptr) {
                    Bias += BlockSize;
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
#if !defined(MLAS_TARGET_AMD64) && !defined(MLAS_TARGET_LARCH64)
    static MLAS_POOL_FLOAT_KERNEL* const PoolKernels[];
#endif

    const MLAS_NCHWC_POOL_WORK_BLOCK* WorkBlock;

    MLAS_NCHWC_POOL_ALGORITHM(const MLAS_NCHWC_POOL_WORK_BLOCK* WorkBlock) :
        MLAS_NCHWC_NN_ALGORITHM(WorkBlock),
        WorkBlock(WorkBlock)
    {
    }

    void Execute(ptrdiff_t Index)
    {
        const size_t TotalWork =
            ((BatchCount * InputChannels + BlockSize - 1) / BlockSize) * OutputHeight;

        size_t WorkIndex;
        size_t WorkRemaining;

        MlasPartitionWork(Index, WorkBlock->tids, TotalWork, &WorkIndex, &WorkRemaining);

        size_t ph = WorkIndex % OutputHeight;
        const size_t BatchChannel = WorkIndex / OutputHeight;

        const float* Input = WorkBlock->Input + BatchChannel * BlockSize * InputSize;
        float* Output = WorkBlock->Output + WorkIndex * BlockSize * OutputWidth;

        //
        // Loop until all of the work has been completed.
        //

        const size_t StrideWidthBytes = BlockSize * StrideWidth * sizeof(float);
        const size_t DilationWidthBytes = BlockSize * DilationWidth * sizeof(float);
        const size_t InputWidthBytes = BlockSize * InputWidth * sizeof(float);
        const size_t DilatedInputWidthBytes = BlockSize * DilationHeight * InputWidth * sizeof(float);
        const size_t InputStrideBytes = DilatedInputWidthBytes - KernelWidth * DilationWidthBytes;

#if defined(MLAS_TARGET_AMD64) || defined(MLAS_TARGET_LARCH64)
        MLAS_POOL_FLOAT_KERNEL* Kernel = GetMlasPlatform().PoolFloatKernel[WorkBlock->PoolingKind];
#else
        MLAS_POOL_FLOAT_KERNEL* Kernel = PoolKernels[WorkBlock->PoolingKind];
#endif

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

            Kernel(Input + BlockSize * (ih * InputWidth - PaddingLeftX), Output,
                StrideWidthBytes, DilationWidthBytes, InputStrideBytes,
                KernelSize, EffectiveKernelHeight, KernelWidth,
                Input + BlockSize * (ih * InputWidth), InputWidthBytes,
                DilatedInputWidthBytes, OutputCountLeftPadX, OutputCountX,
                OutputCountRightPadX);

            Output += BlockSize * OutputWidth;

            //
            // Adjust the amount of work remaining and check if the end of an output
            // image has been reached.
            //

            WorkRemaining -= 1;

            if (++ph == OutputHeight) {

                Input += BlockSize * InputSize;

                ph = 0;
            }
        }
    }
};

#if !defined(MLAS_TARGET_AMD64) && !defined(MLAS_TARGET_LARCH64)

MLAS_POOL_FLOAT_KERNEL* const MLAS_NCHWC_POOL_ALGORITHM::PoolKernels[] =
{
    MlasPoolMaximumFloatKernel,
    MlasPoolAverageExcludePadFloatKernel,
    MlasPoolAverageIncludePadFloatKernel,
};

#endif

void
MLASCALL
MlasNchwcConv(
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
/*++

Routine Description:

    This routine implements the NCHWc convolution operation.

Arguments:

    Dimensions - Supplies the number of dimensions.

    InputShape - Supplies the shape of the input tensor.

    KernelShape - Supplies the shape of the kernel transform.

    DilationShape - Supplies the shape of the dilation.

    Padding - Supplies the number of padding elements at the edge of the input
        tensor.

    StrideShape - Supplies the shape of the stride.

    OutputShape - Supplies the shape of the output tensor.

    GroupCount - Supplies the number of channel groups.

    Input - Supplies the input tensor.

    Filter - Supplies the filter tensor.

    Bias - Optionally supplies the bias vector.

    Output - Supplies the output tensor.

    Activation - Supplies the parameters for the activation to apply to the
        convolution output.

    ZeroMode - Supplies true if the output tensor must be zero initialized
        first, else false if the output tensor is accumulated into. This flag is
        used to implement Conv/Sum fusion.

    ThreadPool - Supplies the thread pool object to use, else nullptr if the
        base library threading support should be used.

Return Value:

    None.

--*/
{
    MLAS_NCHWC_CONV_WORK_BLOCK WorkBlock;

    //
    // Capture the convolution specific parameters to the work block.
    //

    WorkBlock.Input = Input;
    WorkBlock.Output = Output;
    WorkBlock.GroupCount = GroupCount;
    WorkBlock.Filter = Filter;
    WorkBlock.Bias = Bias;
    WorkBlock.Activation = Activation;
    WorkBlock.ZeroMode = ZeroMode;

    //
    // Capture the generic shape parameters to the work block.
    //

    MlasNchwcPrepareWorkBlock(&WorkBlock, InputShape, KernelShape,
        DilationShape, Padding, StrideShape, OutputShape);

    WorkBlock.InputChannels /= GroupCount;
    WorkBlock.OutputChannels /= GroupCount;

    //
    // Determine the type of convolution to perform based on the shape
    // parameters.
    //
    // N.B. The caller must be aware of the selection algorithm in order to
    // reorder the filter tensor in the expected format for the given algorithm.
    //

    MLAS_THREADED_ROUTINE* ThreadedRoutine;

    if (WorkBlock.InputChannels >= MlasNchwcGetBlockSize()) {
        if (WorkBlock.KernelShape[0] == 1 && WorkBlock.KernelShape[1] == 1 &&
            WorkBlock.Padding[0] == 0 && WorkBlock.Padding[1] == 0 &&
            WorkBlock.Padding[2] == 0 && WorkBlock.Padding[3] == 0) {
            ThreadedRoutine = MlasNchwcThreaded<MLAS_NCHWC_CONV_POINTWISE_ALGORITHM>;
        } else {
            ThreadedRoutine = MlasNchwcThreaded<MLAS_NCHWC_CONV_NCHWC_ALGORITHM>;
        }
    } else if (WorkBlock.InputChannels == 1 && WorkBlock.OutputChannels == 1) {
        ThreadedRoutine = MlasNchwcThreaded<MLAS_NCHWC_CONV_DEPTHWISE_ALGORITHM>;
    } else {
        ThreadedRoutine = MlasNchwcThreaded<MLAS_NCHWC_CONV_NCHW_ALGORITHM>;
    }

    //
    // Schedule the operation across a set of worker threads.
    //

    WorkBlock.tids = MlasGetMaximumThreadCount(ThreadPool);

    MlasExecuteThreaded(ThreadedRoutine, &WorkBlock, WorkBlock.tids, ThreadPool);
}

void
MLASCALL
MlasNchwcPool(
    MLAS_POOLING_KIND PoolingKind,
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
/*++

Routine Description:

    This routine implements the NCHWc pooling operation.

Arguments:

    PoolingKind - Supplies the kind of pooling operation to perform.

    InputShape - Supplies the shape of the input tensor.

    KernelShape - Supplies the shape of the kernel transform.

    DilationShape - Supplies the shape of the dilation.

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
    MLAS_NCHWC_POOL_WORK_BLOCK WorkBlock;

    //
    // Capture the pooling specific parameters to the work block.
    //

    WorkBlock.Input = Input;
    WorkBlock.Output = Output;
    WorkBlock.PoolingKind = PoolingKind;

    //
    // Capture the generic shape parameters to the work block.
    //

    MlasNchwcPrepareWorkBlock(&WorkBlock, InputShape, KernelShape,
        DilationShape, Padding, StrideShape, OutputShape);

    //
    // Schedule the operation across a set of worker threads.
    //

    WorkBlock.tids = MlasGetMaximumThreadCount(ThreadPool);

    MlasExecuteThreaded(MlasNchwcThreaded<MLAS_NCHWC_POOL_ALGORITHM>, &WorkBlock, WorkBlock.tids, ThreadPool);
}

void
MLASCALL
MlasNchwcUpsampleNearest(
    const int64_t* InputShape,
    const int64_t* Scales,
    const float* Input,
    float* Output
    )
/*++

Routine Description:

    This routine implements the NCHWc upsample nearest operation.

Arguments:

    InputShape - Supplies the shape of the input tensor.

    Scales - Supplies the shape of the spatial scaling.

    Input - Supplies the input tensor.

    Output - Supplies the output tensor.

Return Value:

    None.

--*/
{
    const size_t BlockSize = MlasNchwcGetBlockSize();

    const size_t BatchCount = size_t(InputShape[0]);
    const size_t ChannelCount = size_t(InputShape[1]);
    const size_t InputHeight = size_t(InputShape[2]);
    const size_t InputWidth = size_t(InputShape[3]);

    const size_t TotalInputHeight = BatchCount * ChannelCount * InputHeight;

    const size_t ScaleHeight = size_t(Scales[0]);
    const size_t ScaleWidth = size_t(Scales[1]);

    const size_t OutputWidth = InputWidth * ScaleWidth;

    //
    // Iterate over each line of the input tensor.
    //

    for (size_t h = 0; h < TotalInputHeight; h += BlockSize) {

        float* OutputBaseRow = Output;

        //
        // Scale the input tensor across the width dimension.
        //

        for (size_t w = 0; w < InputWidth; w++) {

            if (BlockSize == 16) {

                MLAS_FLOAT32X4 v0 = MlasLoadFloat32x4(Input);
                MLAS_FLOAT32X4 v1 = MlasLoadFloat32x4(Input + 4);
                MLAS_FLOAT32X4 v2 = MlasLoadFloat32x4(Input + 8);
                MLAS_FLOAT32X4 v3 = MlasLoadFloat32x4(Input + 12);

                for (size_t sw = 0; sw < ScaleWidth; sw++) {

                    MlasStoreFloat32x4(Output, v0);
                    MlasStoreFloat32x4(Output + 4, v1);
                    MlasStoreFloat32x4(Output + 8, v2);
                    MlasStoreFloat32x4(Output + 12, v3);

                    Output += BlockSize;
                }

            } else {

                MLAS_FLOAT32X4 v0 = MlasLoadFloat32x4(Input);
                MLAS_FLOAT32X4 v1 = MlasLoadFloat32x4(Input + 4);

                for (size_t sw = 0; sw < ScaleWidth; sw++) {

                    MlasStoreFloat32x4(Output, v0);
                    MlasStoreFloat32x4(Output + 4, v1);

                    Output += BlockSize;
                }
            }

            Input += BlockSize;
        }

        //
        // Scale the input tensor across the height dimension by duplicating
        // the first output line.
        //

        for (size_t sh = 1; sh < ScaleHeight; sh++) {
            Output = std::copy_n(OutputBaseRow, OutputWidth * BlockSize, Output);
        }
    }
}

MLAS_FORCEINLINE
void
MlasNchwcExtractInterpolation(
    float InterpolationValue,
    size_t InputLimit,
    ptrdiff_t InputIndex[2],
    MLAS_FLOAT32X4 Multipliers[2]
    )
{
    InputIndex[0] = ptrdiff_t(InterpolationValue);
    InputIndex[1] = std::min<ptrdiff_t>(InputIndex[0] + 1, ptrdiff_t(InputLimit - 1));

    float ScalarMultiplier0 = InterpolationValue - float(InputIndex[0]);
    float ScalarMultiplier1 = 1.0f - ScalarMultiplier0;

    Multipliers[0] = MlasBroadcastFloat32x4(ScalarMultiplier0);
    Multipliers[1] = MlasBroadcastFloat32x4(ScalarMultiplier1);
}

void
MLASCALL
MlasNchwcUpsampleLinear(
    size_t InputHeight,
    size_t InputWidth,
    size_t OutputWidth,
    float InterpolationHeight,
    const float* InterpolationWidth,
    const float* Input,
    float* Output
    )
/*++

Routine Description:

    This routine implements the NCHWc upsample linear operation for a single row.

    The integer portion of each interpolation float supplies the mapping from
    output element to input element. The fractional portion supplies the relative
    weights for the four points of the interpolation.

Arguments:

    InputHeight - Supplies the input height.

    InputWidth - Supplies the input width.

    OutputWidth - Supplies the output width.

    InterpolationHeight - Supplies the height interpolation values for the target
        row.

    InterpolationWidth - Supplies an array of computed interpolation values of
        length OutputWidth.

    Input - Supplies the input spatial buffer.

    Output - Supplies the output row buffer.

Return Value:

    None.

--*/
{
    const size_t BlockSize = MlasNchwcGetBlockSize();

    ptrdiff_t InputIndexY[2];
    MLAS_FLOAT32X4 MultipliersY[2];

    MlasNchwcExtractInterpolation(InterpolationHeight, InputHeight, InputIndexY, MultipliersY);

    const float* InputRowY0 = Input + InputIndexY[0] * InputWidth * BlockSize;
    const float* InputRowY1 = Input + InputIndexY[1] * InputWidth * BlockSize;

    for (size_t ow = 0; ow < OutputWidth; ow++) {

        ptrdiff_t InputIndexX[2];
        MLAS_FLOAT32X4 MultipliersX[2];

        MlasNchwcExtractInterpolation(InterpolationWidth[ow], InputWidth, InputIndexX, MultipliersX);

        MLAS_FLOAT32X4 MultiplierY0X0 = MlasMultiplyFloat32x4(MultipliersY[0], MultipliersX[0]);
        MLAS_FLOAT32X4 MultiplierY0X1 = MlasMultiplyFloat32x4(MultipliersY[0], MultipliersX[1]);
        MLAS_FLOAT32X4 MultiplierY1X0 = MlasMultiplyFloat32x4(MultipliersY[1], MultipliersX[0]);
        MLAS_FLOAT32X4 MultiplierY1X1 = MlasMultiplyFloat32x4(MultipliersY[1], MultipliersX[1]);

        for (size_t bc = 0; bc < BlockSize; bc += 4) {

            MLAS_FLOAT32X4 v00 = MlasLoadFloat32x4(InputRowY0 + InputIndexX[0] * BlockSize + bc);
            MLAS_FLOAT32X4 v01 = MlasLoadFloat32x4(InputRowY0 + InputIndexX[1] * BlockSize + bc);
            MLAS_FLOAT32X4 v10 = MlasLoadFloat32x4(InputRowY1 + InputIndexX[0] * BlockSize + bc);
            MLAS_FLOAT32X4 v11 = MlasLoadFloat32x4(InputRowY1 + InputIndexX[1] * BlockSize + bc);

            v00 = MlasMultiplyFloat32x4(MultiplierY1X1, v00);
            v01 = MlasMultiplyFloat32x4(MultiplierY1X0, v01);
            v10 = MlasMultiplyFloat32x4(MultiplierY0X1, v10);
            v11 = MlasMultiplyFloat32x4(MultiplierY0X0, v11);

            MLAS_FLOAT32X4 Reduction0 = MlasAddFloat32x4(v00, v01);
            MLAS_FLOAT32X4 Reduction1 = MlasAddFloat32x4(v10, v11);

            MLAS_FLOAT32X4 Reduction = MlasAddFloat32x4(Reduction0, Reduction1);

            MlasStoreFloat32x4(&Output[bc], Reduction);
        }

         Output += BlockSize;
    }
}

#if !defined(MLAS_TARGET_AMD64) && !defined(MLAS_TARGET_LARCH64)

//
// Convolution and pooling kernel stubs for architectures that do not yet have
// native support.
//

void
MLASCALL
MlasConvNchwFloatKernel(
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
    unsigned Flags
    )
{
    MLAS_UNREFERENCED_PARAMETER(Input);
    MLAS_UNREFERENCED_PARAMETER(Filter);
    MLAS_UNREFERENCED_PARAMETER(Output);
    MLAS_UNREFERENCED_PARAMETER(StrideWidth);
    MLAS_UNREFERENCED_PARAMETER(DilationWidth);
    MLAS_UNREFERENCED_PARAMETER(FilterCount);
    MLAS_UNREFERENCED_PARAMETER(InputStride);
    MLAS_UNREFERENCED_PARAMETER(FilterStride);
    MLAS_UNREFERENCED_PARAMETER(OutputStride);
    MLAS_UNREFERENCED_PARAMETER(KernelHeight);
    MLAS_UNREFERENCED_PARAMETER(KernelWidth);
    MLAS_UNREFERENCED_PARAMETER(InputBase);
    MLAS_UNREFERENCED_PARAMETER(InputWidth);
    MLAS_UNREFERENCED_PARAMETER(DilatedInputWidth);
    MLAS_UNREFERENCED_PARAMETER(OutputCountLeftPad);
    MLAS_UNREFERENCED_PARAMETER(OutputCount);
    MLAS_UNREFERENCED_PARAMETER(OutputCountRightPad);
    MLAS_UNREFERENCED_PARAMETER(Bias);
    MLAS_UNREFERENCED_PARAMETER(Flags);
}

void
MLASCALL
MlasConvNchwcFloatKernel(
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
    unsigned Flags
    )
{
    MLAS_UNREFERENCED_PARAMETER(Input);
    MLAS_UNREFERENCED_PARAMETER(Filter);
    MLAS_UNREFERENCED_PARAMETER(Output);
    MLAS_UNREFERENCED_PARAMETER(StrideWidth);
    MLAS_UNREFERENCED_PARAMETER(DilationWidth);
    MLAS_UNREFERENCED_PARAMETER(FilterCount);
    MLAS_UNREFERENCED_PARAMETER(InputStride);
    MLAS_UNREFERENCED_PARAMETER(FilterStride);
    MLAS_UNREFERENCED_PARAMETER(OutputStride);
    MLAS_UNREFERENCED_PARAMETER(KernelHeight);
    MLAS_UNREFERENCED_PARAMETER(KernelWidth);
    MLAS_UNREFERENCED_PARAMETER(InputBase);
    MLAS_UNREFERENCED_PARAMETER(InputWidth);
    MLAS_UNREFERENCED_PARAMETER(DilatedInputWidth);
    MLAS_UNREFERENCED_PARAMETER(OutputCountLeftPad);
    MLAS_UNREFERENCED_PARAMETER(OutputCount);
    MLAS_UNREFERENCED_PARAMETER(OutputCountRightPad);
    MLAS_UNREFERENCED_PARAMETER(Bias);
    MLAS_UNREFERENCED_PARAMETER(Flags);
}

void
MLASCALL
MlasConvDepthwiseFloatKernel(
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
    unsigned Flags
    )
{
    MLAS_UNREFERENCED_PARAMETER(Input);
    MLAS_UNREFERENCED_PARAMETER(Filter);
    MLAS_UNREFERENCED_PARAMETER(Output);
    MLAS_UNREFERENCED_PARAMETER(StrideWidth);
    MLAS_UNREFERENCED_PARAMETER(DilationWidth);
    MLAS_UNREFERENCED_PARAMETER(InputStride);
    MLAS_UNREFERENCED_PARAMETER(KernelHeight);
    MLAS_UNREFERENCED_PARAMETER(KernelWidth);
    MLAS_UNREFERENCED_PARAMETER(InputBase);
    MLAS_UNREFERENCED_PARAMETER(InputWidth);
    MLAS_UNREFERENCED_PARAMETER(DilatedInputWidth);
    MLAS_UNREFERENCED_PARAMETER(OutputCountLeftPad);
    MLAS_UNREFERENCED_PARAMETER(OutputCount);
    MLAS_UNREFERENCED_PARAMETER(OutputCountRightPad);
    MLAS_UNREFERENCED_PARAMETER(Bias);
    MLAS_UNREFERENCED_PARAMETER(Flags);
}

void
MLASCALL
MlasConvPointwiseFloatKernel(
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
    unsigned Flags
    )
{
    MLAS_UNREFERENCED_PARAMETER(Input);
    MLAS_UNREFERENCED_PARAMETER(Filter);
    MLAS_UNREFERENCED_PARAMETER(Output);
    MLAS_UNREFERENCED_PARAMETER(StrideWidth);
    MLAS_UNREFERENCED_PARAMETER(InputChannels);
    MLAS_UNREFERENCED_PARAMETER(FilterCount);
    MLAS_UNREFERENCED_PARAMETER(InputStride);
    MLAS_UNREFERENCED_PARAMETER(FilterStride);
    MLAS_UNREFERENCED_PARAMETER(OutputStride);
    MLAS_UNREFERENCED_PARAMETER(OutputCount);
    MLAS_UNREFERENCED_PARAMETER(Bias);
    MLAS_UNREFERENCED_PARAMETER(Flags);
}

void
MLASCALL
MlasPoolMaximumFloatKernel(
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
    MLAS_UNREFERENCED_PARAMETER(Input);
    MLAS_UNREFERENCED_PARAMETER(Output);
    MLAS_UNREFERENCED_PARAMETER(StrideWidth);
    MLAS_UNREFERENCED_PARAMETER(DilationWidth);
    MLAS_UNREFERENCED_PARAMETER(InputStride);
    MLAS_UNREFERENCED_PARAMETER(ActualKernelSize);
    MLAS_UNREFERENCED_PARAMETER(KernelHeight);
    MLAS_UNREFERENCED_PARAMETER(KernelWidth);
    MLAS_UNREFERENCED_PARAMETER(InputBase);
    MLAS_UNREFERENCED_PARAMETER(InputWidth);
    MLAS_UNREFERENCED_PARAMETER(DilatedInputWidth);
    MLAS_UNREFERENCED_PARAMETER(OutputCountLeftPad);
    MLAS_UNREFERENCED_PARAMETER(OutputCount);
    MLAS_UNREFERENCED_PARAMETER(OutputCountRightPad);
}

void
MLASCALL
MlasPoolAverageExcludePadFloatKernel(
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
    MLAS_UNREFERENCED_PARAMETER(Input);
    MLAS_UNREFERENCED_PARAMETER(Output);
    MLAS_UNREFERENCED_PARAMETER(StrideWidth);
    MLAS_UNREFERENCED_PARAMETER(DilationWidth);
    MLAS_UNREFERENCED_PARAMETER(InputStride);
    MLAS_UNREFERENCED_PARAMETER(ActualKernelSize);
    MLAS_UNREFERENCED_PARAMETER(KernelHeight);
    MLAS_UNREFERENCED_PARAMETER(KernelWidth);
    MLAS_UNREFERENCED_PARAMETER(InputBase);
    MLAS_UNREFERENCED_PARAMETER(InputWidth);
    MLAS_UNREFERENCED_PARAMETER(DilatedInputWidth);
    MLAS_UNREFERENCED_PARAMETER(OutputCountLeftPad);
    MLAS_UNREFERENCED_PARAMETER(OutputCount);
    MLAS_UNREFERENCED_PARAMETER(OutputCountRightPad);
}

void
MLASCALL
MlasPoolAverageIncludePadFloatKernel(
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
    MLAS_UNREFERENCED_PARAMETER(Input);
    MLAS_UNREFERENCED_PARAMETER(Output);
    MLAS_UNREFERENCED_PARAMETER(StrideWidth);
    MLAS_UNREFERENCED_PARAMETER(DilationWidth);
    MLAS_UNREFERENCED_PARAMETER(InputStride);
    MLAS_UNREFERENCED_PARAMETER(ActualKernelSize);
    MLAS_UNREFERENCED_PARAMETER(KernelHeight);
    MLAS_UNREFERENCED_PARAMETER(KernelWidth);
    MLAS_UNREFERENCED_PARAMETER(InputBase);
    MLAS_UNREFERENCED_PARAMETER(InputWidth);
    MLAS_UNREFERENCED_PARAMETER(DilatedInputWidth);
    MLAS_UNREFERENCED_PARAMETER(OutputCountLeftPad);
    MLAS_UNREFERENCED_PARAMETER(OutputCount);
    MLAS_UNREFERENCED_PARAMETER(OutputCountRightPad);
}

#endif
