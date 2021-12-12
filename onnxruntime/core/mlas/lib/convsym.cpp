/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    convsym.cpp

Abstract:

    This module implements the symmetric quantized integer convolution
    operation.

--*/

#include "mlasi.h"

//
// Define the prototypes of the platform optimized routines.
//

typedef
void
(MLASCALL MLAS_CONV_SYM_KERNEL)(
    const void* Input,
    const void* Filter,
    void* Output,
    size_t KernelSize,
    size_t InputChannels,
    size_t OutputChannels,
    unsigned ChannelCount,
    unsigned OutputCount,
    const struct MLAS_CONV_SYM_POST_PROCESS_PARAMS* PostProcessParams,
    unsigned KernelFlags
    );

typedef
void
(MLASCALL MLAS_CONV_SYM_DEPTHWISE_KERNEL)(
    const void* Input,
    const void* Filter,
    void* Output,
    size_t KernelSize,
    size_t Channels,
    size_t ChannelOffset,
    unsigned ChannelCount,
    unsigned OutputCount,
    const struct MLAS_CONV_SYM_POST_PROCESS_PARAMS* PostProcessParams,
    unsigned KernelFlags
    );


extern "C" {

#if defined(MLAS_TARGET_AMD64)
    MLAS_CONV_SYM_KERNEL MlasConvSymKernelAvx2;
    MLAS_CONV_SYM_DEPTHWISE_KERNEL MlasConvSymDepthwiseKernelAvx2;
    MLAS_CONV_SYM_KERNEL MlasConvSymKernelAvxVnni;
    MLAS_CONV_SYM_DEPTHWISE_KERNEL MlasConvSymDepthwiseKernelAvxVnni;
    MLAS_CONV_SYM_KERNEL MlasConvSymKernelAvx512Core;
    MLAS_CONV_SYM_DEPTHWISE_KERNEL MlasConvSymDepthwiseKernelAvx512Core;
    MLAS_CONV_SYM_KERNEL MlasConvSymKernelAvx512Vnni;
    MLAS_CONV_SYM_DEPTHWISE_KERNEL MlasConvSymDepthwiseKernelAvx512Vnni;
#elif defined(MLAS_TARGET_ARM64)
    MLAS_CONV_SYM_DEPTHWISE_KERNEL MlasConvSymDepthwiseKernelNeon;
    MLAS_CONV_SYM_DEPTHWISE_ROUTINE_KERNELSIZE MlasConvSymDepthwiseKernelSize9Arm64;
    MLAS_CONV_SYM_DEPTHWISE_ROUTINE_KERNELSIZE MlasConvSymDepthwiseKernelSize25Arm;
#endif

}

//
//
//

struct MLAS_CONV_SYM_DISPATCH {
    MLAS_CONV_SYM_KERNEL* Kernel;
    MLAS_CONV_SYM_DEPTHWISE_KERNEL* DepthwiseKernel;
    uint8_t FilterInputChannelPackCount;
    uint8_t FilterOutputChannelPackCount;
    uint8_t KernelChannelCount;
    uint8_t KernelOutputCount;
    uint8_t KernelInputChannelAlignment;
    uint8_t KernelOutputChannelAlignment;
    uint8_t KernelDepthwiseChannelCount;
    uint8_t KernelDepthwiseOutputCount;
    bool FixupInputZeroPoint;
};

#if defined(MLAS_TARGET_AMD64)

const MLAS_CONV_SYM_DISPATCH MlasConvSymDispatchAvx2 = {
    MlasConvSymKernelAvx2,
    MlasConvSymDepthwiseKernelAvx2,
    4,                                      // FilterInputChannelPackCount
    16,                                     // FilterOutputChannelPackCount
    16,                                     // KernelChannelCount
    4,                                      // KernelOutputCount
    4,                                      // KernelInputChannelAlignment
    8,                                      // KernelOutputChannelAlignment
    16,                                     // KernelDepthwiseChannelCount
    4,                                      // KernelDepthwiseOutputCount
    false,                                  // FixupInputZeroPoint
};

const MLAS_CONV_SYM_DISPATCH MlasConvSymDispatchAvxVnni = {
    MlasConvSymKernelAvxVnni,
    MlasConvSymDepthwiseKernelAvxVnni,
    4,                                      // FilterInputChannelPackCount
    16,                                     // FilterOutputChannelPackCount
    16,                                     // KernelChannelCount
    6,                                      // KernelOutputCount
    4,                                      // KernelInputChannelAlignment
    8,                                      // KernelOutputChannelAlignment
    16,                                     // KernelDepthwiseChannelCount
    4,                                      // KernelDepthwiseOutputCount
    false,                                  // FixupInputZeroPoint
};

#if !defined(ORT_MINIMAL_BUILD)

const MLAS_CONV_SYM_DISPATCH MlasConvSymDispatchAvx512Core = {
    MlasConvSymKernelAvx512Core,
    MlasConvSymDepthwiseKernelAvx512Core,
    4,                                      // FilterInputChannelPackCount
    16,                                     // FilterOutputChannelPackCount
    64,                                     // KernelChannelCount
    6,                                      // KernelOutputCount
    4,                                      // KernelInputChannelAlignment
    4,                                      // KernelOutputChannelAlignment
    64,                                     // KernelDepthwiseChannelCount
    6,                                      // KernelDepthwiseOutputCount
    false,                                  // FixupInputZeroPoint
};

const MLAS_CONV_SYM_DISPATCH MlasConvSymDispatchAvx512Vnni = {
    MlasConvSymKernelAvx512Vnni,
    MlasConvSymDepthwiseKernelAvx512Vnni,
    4,                                      // FilterInputChannelPackCount
    16,                                     // FilterOutputChannelPackCount
    64,                                     // KernelChannelCount
    6,                                      // KernelOutputCount
    4,                                      // KernelInputChannelAlignment
    4,                                      // KernelOutputChannelAlignment
    64,                                     // KernelDepthwiseChannelCount
    6,                                      // KernelDepthwiseOutputCount
    false,                                  // FixupInputZeroPoint
};

#endif // ORT_MINIMAL_BUILD

#elif defined(MLAS_TARGET_ARM64)
const MLAS_CONV_SYM_DISPATCH MlasConvSymDispatchNeon = {
    nullptr,
    MlasConvSymDepthwiseKernelNeon,
    4,   // FilterInputChannelPackCount
    16,  // FilterOutputChannelPackCount
    8,   // KernelChannelCount
    8,   // KernelOutputCount
    4,   // KernelInputChannelAlignment
    8,   // KernelOutputChannelAlignment
    16,  // KernelDepthwiseChannelCount
    4,   // KernelDepthwiseOutputCount
    true
};
#endif // MLAS_TARGET_AMD64

MLAS_FORCEINLINE
void
MlasConvSymSetOutputZeroPoint(
    MLAS_CONV_SYM_POST_PROCESS_PARAMS& PostProcessParams,
    int32_t OutputZeroPoint,
    bool InputIsSigned
    )
{
    int32_t minimum = InputIsSigned ? std::numeric_limits<int8_t>::lowest()
                                    : std::numeric_limits<uint8_t>::lowest();
    int32_t maximum = InputIsSigned ? std::numeric_limits<int8_t>::max()
                                    : std::numeric_limits<uint8_t>::max();
    PostProcessParams.MinimumValue = static_cast<float>(minimum - OutputZeroPoint);
    PostProcessParams.MaximumValue = static_cast<float>(maximum - OutputZeroPoint);
    PostProcessParams.OutputZeroPoint = OutputZeroPoint;
}

MLAS_FORCEINLINE
const
MLAS_CONV_SYM_DISPATCH*
GetConvSymDispatch(bool InputIsSigned){
    return InputIsSigned ? MlasPlatform.ConvSymS8S8Dispatch : MlasPlatform.ConvSymU8S8Dispatch;
}

size_t
MlasConvSymPackWSize(
    size_t GroupCount,
    size_t InputChannels,
    size_t OutputChannels,
    size_t KernelSize,
    bool InputIsSigned
    )
{
    const MLAS_CONV_SYM_DISPATCH* ConvSymDispatch = GetConvSymDispatch(InputIsSigned);

    if (ConvSymDispatch == nullptr) {
        return 0;
    }

    if (GroupCount > 1) {

        if (ConvSymDispatch->DepthwiseKernel != nullptr &&
            InputChannels == 1 && OutputChannels == 1) {
#ifdef MLAS_TARGET_ARM64
            constexpr size_t GroupAlign = 8;
#else
            constexpr size_t GroupAlign = 16;
#endif
            size_t AlignedGroupCount = (GroupCount + GroupAlign - 1) & ~(GroupAlign - 1);

            if (AlignedGroupCount != GroupCount) {
                return 0;
            }

            return AlignedGroupCount * KernelSize;

        } else {
            return 0;
        }

    } else {

        size_t OutputChannelPackCount = ConvSymDispatch->FilterOutputChannelPackCount;

        if (ConvSymDispatch->Kernel == nullptr ||
            OutputChannels < OutputChannelPackCount ||
            (InputChannels % ConvSymDispatch->KernelInputChannelAlignment) != 0 ||
            (OutputChannels % ConvSymDispatch->KernelOutputChannelAlignment) != 0
            ) {
            return 0;
        }

        size_t AlignedOutputChannels = (OutputChannels + OutputChannelPackCount - 1) / OutputChannelPackCount * OutputChannelPackCount;
        return AlignedOutputChannels * InputChannels * KernelSize;
    }
}

void
MlasConvSymPackW(
    size_t GroupCount,
    size_t InputChannels,
    size_t OutputChannels,
    size_t KernelSize,
    const int8_t* W,
    int8_t* PackedW,
    size_t PackedWSize,
    bool InputIsSigned
    )
{
    memset(PackedW, 0, PackedWSize);

    if (GroupCount > 1) {

        for (size_t gc = 0; gc < GroupCount; gc++) {

            for (size_t k = 0; k < KernelSize; k++) {

                PackedW[k * GroupCount + gc] = W[gc * KernelSize + k];

            }
        }

    } else {

        const MLAS_CONV_SYM_DISPATCH* ConvSymDispatch = GetConvSymDispatch(InputIsSigned);
        size_t InputChannelPackCount = ConvSymDispatch->FilterInputChannelPackCount;
        size_t OutputChannelPackCount = ConvSymDispatch->FilterOutputChannelPackCount;

        size_t kernel_dim = InputChannels * KernelSize;

        for (size_t oc = 0; oc < OutputChannels; oc += OutputChannelPackCount) {

            const size_t oc_pack_size = std::min(OutputChannels - oc, OutputChannelPackCount);

            for (size_t ki = 0; ki < KernelSize; ki++) {

                for (size_t ic = 0; ic < InputChannels; ic += InputChannelPackCount) {

                    const size_t ic_pack_size = std::min(InputChannels - ic, InputChannelPackCount);

                    for (size_t oc_pack = 0; oc_pack < oc_pack_size; oc_pack++) {

                        for (size_t ic_pack = 0; ic_pack < ic_pack_size; ic_pack++) {

                            *(PackedW++) = W[(oc + oc_pack) * kernel_dim + (ic + ic_pack) * KernelSize + ki];

                        }

                        PackedW += InputChannelPackCount - ic_pack_size;

                    }

                    PackedW += (OutputChannelPackCount - oc_pack_size) * InputChannelPackCount;

                }
            }
        }

    }
}

int32_t
MlasConvSymFixupInputZeroPoint(
    int32_t zero_point_value,
    bool InputIsSigned
    )
{
    const MLAS_CONV_SYM_DISPATCH* ConvSymDispatch = GetConvSymDispatch(InputIsSigned);

    if (ConvSymDispatch != nullptr && ConvSymDispatch->FixupInputZeroPoint) {
        return zero_point_value - 128;
    }
    return zero_point_value;
}


void
MlasConvSym(
    const MLAS_CONV_SYM_PARAMS& Params
    )
{
    const MLAS_CONV_SYM_DISPATCH* ConvSymDispatch = GetConvSymDispatch(Params.InputIsSigned);

    int32_t KernelFlags = 0;

    if (Params.PerChannelScale) {
        KernelFlags |= MLAS_CONV_SYM_FLAG_PER_CHANNEL_SCALE;
    }

    if (Params.InputIndirection == nullptr) {
        KernelFlags |= MLAS_CONV_SYM_FLAG_INPUT_DIRECT;
    }

    MLAS_CONV_SYM_POST_PROCESS_PARAMS PostProcessParams = {};

    MlasConvSymSetOutputZeroPoint(PostProcessParams, Params.OutputZeroPoint, Params.InputIsSigned);

    const size_t KernelChannelCount = ConvSymDispatch->KernelChannelCount;
    const size_t KernelOutputCount = ConvSymDispatch->KernelOutputCount;

    const size_t KernelSize = Params.KernelSize;
    const size_t InputChannels = Params.InputChannels;
    const size_t OutputChannels = Params.OutputChannels;

    for (size_t oc_outside = 0; oc_outside < Params.OutputCount;) {

        const size_t oc_outside_block_size = std::min<size_t>(Params.OutputCount - oc_outside, 240);
        const int8_t* pwb = static_cast<const int8_t*>(Params.Filter);

        for (size_t co = 0; co < OutputChannels;) {

            const size_t ChannelCount = std::min<size_t>(OutputChannels - co, KernelChannelCount);
            void* conv_out = static_cast<int8_t*>(Params.Output) + (oc_outside * OutputChannels) + co;

            PostProcessParams.Bias = Params.Bias + co;
            PostProcessParams.Scale = Params.Scale + (Params.PerChannelScale ? co : 0);

            for (size_t oc = 0; oc < oc_outside_block_size;) {

                const void* Input;
                if (Params.InputIndirection) {
                    Input = Params.InputIndirection + (oc_outside + oc) * KernelSize;
                } else {
                    Input = static_cast<const int8_t*>(Params.InputDirect) + (oc_outside + oc) * InputChannels;
                }
                size_t OutputCount = std::min<size_t>(oc_outside_block_size - oc, KernelOutputCount);

                ConvSymDispatch->Kernel(
                    Input,
                    pwb,
                    conv_out,
                    KernelSize,
                    InputChannels,
                    OutputChannels,
                    static_cast<unsigned>(ChannelCount),
                    static_cast<unsigned>(OutputCount),
                    &PostProcessParams,
                    KernelFlags);
                oc += OutputCount;
                conv_out = static_cast<int8_t*>(conv_out) + OutputCount * OutputChannels;
            }

            co += ChannelCount;
            pwb += ChannelCount * InputChannels * KernelSize;
        }

        oc_outside += oc_outside_block_size;
    }
}

void
MlasConvSymDepthwise(
    const MLAS_CONV_SYM_PARAMS& Params
    )
{
    const MLAS_CONV_SYM_DISPATCH* ConvSymDispatch = GetConvSymDispatch(Params.InputIsSigned);

    unsigned KernelFlags = 0;

    if (Params.PerChannelScale) {
        KernelFlags |= MLAS_CONV_SYM_FLAG_PER_CHANNEL_SCALE;
    }

    MLAS_CONV_SYM_POST_PROCESS_PARAMS PostProcessParams = {};

    MlasConvSymSetOutputZeroPoint(PostProcessParams, Params.OutputZeroPoint, Params.InputIsSigned);

#if defined(MLAS_TARGET_ARM64)

    if ((Params.KernelSize == 9 || Params.KernelSize == 25) && (Params.OutputChannels & 15) == 0) {
        PostProcessParams.Bias = Params.Bias;
        PostProcessParams.Scale = Params.Scale;
        if (Params.KernelSize == 9) {
            MlasConvSymDepthwiseKernelSize9Arm64(
                (const uint8_t *const *)Params.InputIndirection, (int8_t const*)Params.Filter, Params.OutputChannels,
                (uint8_t*)Params.Output, Params.OutputCount, &PostProcessParams, KernelFlags
            );
        } else {
            MlasConvSymDepthwiseKernelSize25Arm(
                (const uint8_t *const *)Params.InputIndirection, (int8_t const*)Params.Filter, Params.OutputChannels,
                (uint8_t*)Params.Output, Params.OutputCount, &PostProcessParams, KernelFlags
            );
        }
        return;
    }

#endif

    const size_t KernelChannelCount = ConvSymDispatch->KernelDepthwiseChannelCount;
    const size_t KernelOutputCount = ConvSymDispatch->KernelDepthwiseOutputCount;

    const size_t KernelSize = Params.KernelSize;
    const size_t OutputChannels = Params.OutputChannels;

    const auto* InputIndirection = Params.InputIndirection;
    void* Output = Params.Output;

    for (size_t OutputCountRemaining = Params.OutputCount; OutputCountRemaining > 0;) {

        const size_t OutputCount = std::min(OutputCountRemaining, KernelOutputCount);

        for (size_t ChannelOffset = 0; ChannelOffset < OutputChannels;) {

            const size_t ChannelCount = std::min(OutputChannels - ChannelOffset, KernelChannelCount);

            PostProcessParams.Bias = Params.Bias + ChannelOffset;
            PostProcessParams.Scale = Params.Scale + (Params.PerChannelScale ? ChannelOffset : 0);

            ConvSymDispatch->DepthwiseKernel(
                InputIndirection,
                static_cast<const uint8_t*>(Params.Filter) + ChannelOffset,
                static_cast<int8_t*>(Output) + ChannelOffset,
                KernelSize,
                OutputChannels,
                ChannelOffset,
                static_cast<unsigned>(ChannelCount),
                static_cast<unsigned>(OutputCount),
                &PostProcessParams,
                KernelFlags);

            ChannelOffset += ChannelCount;
        }

        InputIndirection += OutputCount * KernelSize;
        Output = static_cast<int8_t*>(Output) + OutputCount * OutputChannels;
        OutputCountRemaining -= OutputCount;
    }
}
