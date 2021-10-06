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
#include "convsym.h"

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
#endif

}


#if defined(MLAS_TARGET_AMD64)

const MLAS_CONV_SYM_DISPATCH MlasConvSymDispatchAvx2 = {
    MlasConvSymKernelAvx2,
    MlasConvSymDepthwiseKernelAvx2,
    16,                                     // KernelChannelCount
    4,                                      // KernelOutputCount
    16,                                     // KernelDepthwiseChannelCount
    4,                                      // KernelDepthwiseOutputCount
};

const MLAS_CONV_SYM_DISPATCH MlasConvSymDispatchAvxVnni = {
    MlasConvSymKernelAvxVnni,
    MlasConvSymDepthwiseKernelAvxVnni,
    16,                                     // KernelChannelCount
    6,                                      // KernelOutputCount
    16,                                     // KernelDepthwiseChannelCount
    4,                                      // KernelDepthwiseOutputCount
};

#if !defined(ORT_MINIMAL_BUILD)

const MLAS_CONV_SYM_DISPATCH MlasConvSymDispatchAvx512Core = {
    MlasConvSymKernelAvx512Core,
    MlasConvSymDepthwiseKernelAvx512Core,
    64,                                     // KernelChannelCount
    6,                                      // KernelOutputCount
    64,                                     // KernelDepthwiseChannelCount
    6,                                      // KernelDepthwiseOutputCount
};

const MLAS_CONV_SYM_DISPATCH MlasConvSymDispatchAvx512Vnni = {
    MlasConvSymKernelAvx512Vnni,
    MlasConvSymDepthwiseKernelAvx512Vnni,
    64,                                     // KernelChannelCount
    6,                                      // KernelOutputCount
    64,                                     // KernelDepthwiseChannelCount
    6,                                      // KernelDepthwiseOutputCount
};

#endif // ORT_MINIMAL_BUILD

#endif // MLAS_TARGET_AMD64

MLAS_FORCEINLINE
void
MlasConvSymSetOutputZeroPoint(
    MLAS_CONV_SYM_POST_PROCESS_PARAMS& PostProcessParams,
    int32_t OutputZeroPoint
    )
{
    PostProcessParams.MinimumValue = static_cast<float>(0 - OutputZeroPoint);
    PostProcessParams.MaximumValue = static_cast<float>(255 - OutputZeroPoint);
    PostProcessParams.OutputZeroPoint = OutputZeroPoint;
}

size_t
MlasConvSymPackWSize(
    size_t GroupCount,
    size_t InputChannels,
    size_t OutputChannels,
    size_t KernelSize
    )
{
    const MLAS_CONV_SYM_DISPATCH* ConvSymDispatch = MlasPlatform.ConvSymDispatch;

    if (ConvSymDispatch == nullptr) {
        return 0;
    }

    if (GroupCount > 1) {

        if (InputChannels == 1 && OutputChannels == 1) {

            size_t AlignedGroupCount = (GroupCount + 15) & ~15;

            if (AlignedGroupCount != GroupCount) {
                return 0;
            }

            return AlignedGroupCount * KernelSize;

        } else {
            return 0;
        }

    } else {

        if ((InputChannels % 4) != 0) {
            return 0;
        }

        if (OutputChannels < 16) {
            return 0;
        }

#if 0
        if ((OutputChannels % 4) != 0) {
            return 0;
        }
#else
        if ((OutputChannels % 8) != 0) {
            return 0;
        }
#endif

        // BUGBUG: assumes aligned input channels, align to 4

        size_t AlignedOutputChannels = (OutputChannels + 15) & ~15;

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
    int8_t* PackedW
    )
{
    if (GroupCount > 1) {

        // W: [k]

        for (size_t gc = 0; gc < GroupCount; gc++) {

            for (size_t k = 0; k < KernelSize; k++) {

                PackedW[k * GroupCount + gc] = W[gc * KernelSize + k];

            }
        }

    } else {

        for (size_t oc = 0; oc < OutputChannels; oc++) {

            int8_t* pw = PackedW + (oc / 16) * 16 * InputChannels * KernelSize;
            pw += (oc % 16) * 4;

            const int8_t* pW = W;

            int8_t* pww = pw;

            for (size_t k = 0; k < KernelSize; k++) {

                const int8_t* pWW = pW;

                for (size_t ic = 0; ic < InputChannels; ic += 4) {

                    int8_t f0 = pWW[KernelSize * 0];
                    int8_t f1 = pWW[KernelSize * 1];
                    int8_t f2 = pWW[KernelSize * 2];
                    int8_t f3 = pWW[KernelSize * 3];

                    pww[0] = f0;
                    pww[1] = f1;
                    pww[2] = f2;
                    pww[3] = f3;

                    pWW += KernelSize * 4;
                    pww += 64;
                }

                pW += 1;
            }

            W += InputChannels * KernelSize;
        }

        // ugly code to zero pad

        size_t AlignedOutputChannels = (OutputChannels + 15) & ~15;

        for (size_t oc = OutputChannels; oc < AlignedOutputChannels; oc++) {

            int8_t* pw = PackedW + (oc / 16) * 16 * InputChannels * KernelSize;
            pw += (oc % 16) * 4;

            int8_t* pww = pw;

            for (size_t k = 0; k < KernelSize; k++) {

                for (size_t ic = 0; ic < InputChannels; ic += 4) {

                    pww[0] = 0;
                    pww[1] = 0;
                    pww[2] = 0;
                    pww[3] = 0;

                    pww += 64;
                }
            }
        }
    }
}

void
MlasConvSym(
    const MLAS_CONV_SYM_PARAMS& Params
    )
{
    const MLAS_CONV_SYM_DISPATCH* ConvSymDispatch = MlasPlatform.ConvSymDispatch;

    int32_t KernelFlags = 0;

    if (Params.PerChannelScale) {
        KernelFlags |= MLAS_CONV_SYM_FLAG_PER_CHANNEL_SCALE;
    }

    if (Params.InputIndirection == nullptr) {
        KernelFlags |= MLAS_CONV_SYM_FLAG_INPUT_DIRECT;
    }

    MLAS_CONV_SYM_POST_PROCESS_PARAMS PostProcessParams = {};

    MlasConvSymSetOutputZeroPoint(PostProcessParams, Params.OutputZeroPoint);

    const size_t KernelChannelCount = ConvSymDispatch->KernelChannelCount;
    const size_t KernelOutputCount = ConvSymDispatch->KernelOutputCount;

    const size_t KernelSize = Params.KernelSize;
    const size_t InputChannels = Params.InputChannels;
    const size_t OutputChannels = Params.OutputChannels;

    for (size_t oc0 = 0; oc0 < Params.OutputCount;) {

        const size_t oc0_this_pass = std::min<size_t>(Params.OutputCount - oc0, 240);
        const int8_t* pwb = static_cast<const int8_t*>(Params.Filter);

        for (size_t co = 0; co < OutputChannels;) {

            const size_t ChannelCount = std::min<size_t>(OutputChannels - co, KernelChannelCount);
            auto* gemm_out = Params.Output + (oc0 * OutputChannels) + co;

            PostProcessParams.Bias = Params.Bias + co;
            PostProcessParams.Scale = Params.Scale + (Params.PerChannelScale ? co : 0);

            for (size_t oc = 0; oc < oc0_this_pass;) {

                const void* gg;
                if (Params.InputIndirection) {
                    gg = Params.InputIndirection + (oc0 + oc) * KernelSize;
                } else {
                    gg = Params.InputDirect + (oc0 + oc) * InputChannels;
                }
                size_t OutputCount = std::min<size_t>(oc0_this_pass - oc, KernelOutputCount);

                ConvSymDispatch->Kernel(
                    gg,
                    pwb,
                    gemm_out,
                    KernelSize,
                    InputChannels,
                    OutputChannels,
                    static_cast<unsigned>(ChannelCount),
                    static_cast<unsigned>(OutputCount),
                    &PostProcessParams,
                    KernelFlags);
                oc += OutputCount;
                gemm_out += OutputCount * OutputChannels;
            }

            co += ChannelCount;
            pwb += ChannelCount * InputChannels * KernelSize;
        }

        oc0 += oc0_this_pass;
    }
}

void
MlasConvSymDepthwise(
    const MLAS_CONV_SYM_PARAMS& Params
    )
{
    const MLAS_CONV_SYM_DISPATCH* ConvSymDispatch = MlasPlatform.ConvSymDispatch;

    unsigned KernelFlags = 0;

    if (Params.PerChannelScale) {
        KernelFlags |= MLAS_CONV_SYM_FLAG_PER_CHANNEL_SCALE;
    }

    //
    //
    //

    MLAS_CONV_SYM_POST_PROCESS_PARAMS PostProcessParams = {};

    MlasConvSymSetOutputZeroPoint(PostProcessParams, Params.OutputZeroPoint);

    const size_t KernelChannelCount = ConvSymDispatch->KernelDepthwiseChannelCount;
    const size_t KernelOutputCount = ConvSymDispatch->KernelDepthwiseOutputCount;

    const size_t KernelSize = Params.KernelSize;
    const size_t OutputChannels = Params.OutputChannels;

    const auto* InputIndirection = Params.InputIndirection;
    auto* Output = Params.Output;

    for (size_t OutputCountRemaining = Params.OutputCount; OutputCountRemaining > 0;) {

        const size_t OutputCount = std::min(OutputCountRemaining, KernelOutputCount);

        for (size_t ChannelOffset = 0; ChannelOffset < OutputChannels;) {

            const size_t ChannelCount = std::min(OutputChannels - ChannelOffset, KernelChannelCount);

            PostProcessParams.Bias = Params.Bias + ChannelOffset;
            PostProcessParams.Scale = Params.Scale + (Params.PerChannelScale ? ChannelOffset : 0);

            ConvSymDispatch->DepthwiseKernel(
                InputIndirection,
                static_cast<const uint8_t*>(Params.Filter) + ChannelOffset,
                Output + ChannelOffset,
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
        Output += OutputCount * OutputChannels;
        OutputCountRemaining -= OutputCount;
    }
}