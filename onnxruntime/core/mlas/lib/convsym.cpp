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
//
//

#define MLAS_CONV_SYM_FLAG_INPUT_DIRECT             0x00000001
#define MLAS_CONV_SYM_FLAG_PER_CHANNEL_SCALE        0x00000002

//
//
//

struct MLAS_CONV_SYM_POST_PROCESS_PARAMS {
    const int32_t* Bias;
    const float* Scale;
    float MinimumValue;
    float MaximumValue;
    int32_t OutputZeroPoint;
};

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
    if (GroupCount > 1) {

        if (MlasPlatform.ConvSymDepthwiseKernel == nullptr) {
            return 0;
        }

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

        if (MlasPlatform.ConvSymKernel == nullptr) {
            return 0;
        }

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
    int32_t KernelFlags = 0;

    if (Params.PerChannelScale) {
        KernelFlags |= MLAS_CONV_SYM_FLAG_PER_CHANNEL_SCALE;
    }

    if (Params.InputIndirection == nullptr) {
        KernelFlags |= MLAS_CONV_SYM_FLAG_INPUT_DIRECT;
    }

    MLAS_CONV_SYM_POST_PROCESS_PARAMS PostProcessParams = {};

    MlasConvSymSetOutputZeroPoint(PostProcessParams, Params.OutputZeroPoint);

    size_t co_kernel = MlasPlatform.MaximumConvSymChannelCount;
    size_t oc_kernel = MlasPlatform.MaximumConvSymOutputCount;

    for (size_t oc0 = 0; oc0 < Params.OutputCount;) {
      const size_t oc0_this_pass = std::min<size_t>(Params.OutputCount - oc0, 240);
      const int8_t* pwb = static_cast<const int8_t*>(Params.Filter);
      for (size_t co = 0; co < Params.OutputChannels;) {
        const size_t co_this_pass = std::min<size_t>(Params.OutputChannels - co, co_kernel);
        auto* gemm_out = Params.Output + (oc0 * Params.OutputChannels) + co;
        for (size_t oc = 0; oc < oc0_this_pass;) {
          const void* gg;
          if (Params.InputIndirection) {
            gg = Params.InputIndirection + (oc0 + oc) * Params.KernelSize;
          } else {
            gg = Params.InputDirect + (oc0 + oc) * Params.InputChannels;
          }
          size_t oc_this_pass = std::min<size_t>(oc0_this_pass - oc, oc_kernel);

          PostProcessParams.Bias = Params.Bias + co;
          PostProcessParams.Scale = Params.Scale + (Params.PerChannelScale ? co : 0);

          MlasPlatform.ConvSymKernel(
              gg,
              pwb,
              gemm_out,
              Params.KernelSize,
              Params.InputChannels,
              Params.OutputChannels,
              static_cast<unsigned>(co_this_pass),
              static_cast<unsigned>(oc_this_pass),
              &PostProcessParams,
              KernelFlags);
          oc += oc_this_pass;
          gemm_out += oc_this_pass * Params.OutputChannels;
        }
        co += co_this_pass;
        pwb += co_this_pass * Params.InputChannels * Params.KernelSize;
      }
      oc0 += oc0_this_pass;
    }
}

void
MlasConvSymDepthwise(
    const MLAS_CONV_SYM_PARAMS& Params
    )
{
    unsigned KernelFlags = 0;

    if (Params.PerChannelScale) {
        KernelFlags |= MLAS_CONV_SYM_FLAG_PER_CHANNEL_SCALE;
    }

    //
    //
    //

    MLAS_CONV_SYM_POST_PROCESS_PARAMS PostProcessParams = {};

    MlasConvSymSetOutputZeroPoint(PostProcessParams, Params.OutputZeroPoint);

    size_t co_kernel = MlasPlatform.MaximumConvSymChannelCount;
    size_t oc_kernel = MlasPlatform.MaximumConvSymDepthwiseOutputCount;

    auto* gemm_out = Params.Output;
    auto* conv_input = Params.InputIndirection;
    for (size_t oc = 0; oc < Params.OutputCount;) {
        size_t oc_this_pass = std::min<size_t>(Params.OutputCount - oc, oc_kernel);
        for (size_t co = 0; co < Params.OutputChannels;) {
            const size_t co_this_pass = std::min<size_t>(Params.OutputChannels - co, co_kernel);

            PostProcessParams.Bias = Params.Bias + co;
            PostProcessParams.Scale = Params.Scale + (Params.PerChannelScale ? co : 0);

            MlasPlatform.ConvSymDepthwiseKernel(
                conv_input,
                static_cast<const int8_t*>(Params.Filter) + co,
                gemm_out + co,
                Params.KernelSize,
                Params.OutputChannels,
                co,
                static_cast<unsigned>(co_this_pass),
                static_cast<unsigned>(oc_this_pass),
                &PostProcessParams,
                KernelFlags);
            co += co_this_pass;
        }

        oc += oc_this_pass;
        gemm_out += oc_this_pass * Params.OutputChannels;
        conv_input += Params.KernelSize * oc_this_pass;
    }
}
