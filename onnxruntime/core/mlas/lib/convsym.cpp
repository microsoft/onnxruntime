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
#include <iostream>

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

//
// Define the prototypes of the platform optimized routines.
//

typedef
void
(MLASCALL MLAS_CONV_SYM_KERNEL)(
    const void* Input,
    const void* Filter,
    uint8_t* Output,
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
    uint8_t* Output,
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
#endif

}

//
//
//

struct MLAS_CONV_SYM_DISPATCH {
    MLAS_CONV_SYM_KERNEL* Kernel;
    MLAS_CONV_SYM_DEPTHWISE_KERNEL* DepthwiseKernel;
    uint8_t KernelChannelCount;
    uint8_t KernelOutputCount;
    uint8_t KernelDepthwiseChannelCount;
    uint8_t KernelDepthwiseOutputCount;
};

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



#if defined(MLAS_TARGET_ARM64)
void
MLASCALL MlasConvSymKernelArm64(
    const void* Input,
    const void* Filter,
    uint8_t* Output,
    size_t KernelSize,
    size_t InputChannels,
    size_t OutputChannels,
    unsigned ChannelCount,
    unsigned OutputCount,
    const struct MLAS_CONV_SYM_POST_PROCESS_PARAMS* PostProcessParams,
    unsigned KernelFlags
)
{
    std::cout<<"Calling MlasConvSymKernelArm64"<<std::endl;
    char c_input;
    std::cin>> c_input;
    const int32_t* bias = PostProcessParams->Bias;
    int32x4_t ACC0 = vdupq_n_u32(0);
    int32x4_t ACC1 = vdupq_n_u32(0);
    int32x4_t ACC2 = vdupq_n_u32(0);
    int32x4_t ACC3 = vdupq_n_u32(0);
    int32x4_t ACC4 = vdupq_n_u32(0);
    int32x4_t ACC5 = vdupq_n_u32(0);
    int32x4_t ACC6 = vdupq_n_u32(0);
    int32x4_t ACC7 = vdupq_n_u32(0);
    int8x8_t BitFlip = vdup_n_s8(-128);


    bool IsIndirect = KernelFlags & MLAS_CONV_SYM_FLAG_INPUT_DIRECT;
    const int8_t* a0 = static_cast<const int8_t*> (Input);
    const int8_t* w = static_cast<const int8_t*> (Filter);
    // const uint8_t* a1 = static_cast<const uint8_t*> (Input) + InputChannels;
    for(size_t k = 0; k < KernelSize; k++){
        if (IsIndirect) {
            int8_t* const* InputIndirect = reinterpret_cast<int8_t* const *>(Input);
            a0 = InputIndirect[k];
        }

        for (size_t c = 0; c < InputChannels; c += 4, a0+=4) {
            int8x8_t Row0C = vreinterpret_s8_s32(vdup_n_s32(*(reinterpret_cast<const int32_t*>(a0 + c))));
            Row0C = veor_s8(Row0C, BitFlip);
            // int8x8_t Row1C = vreinterpret_s8_s32(vdup_n_s32(*(reinterpret_cast<const int32_t*>(a1 + c))));
            
            vpadalq_s16(ACC0, vmull_s8(Row0C, vld1_s8(w))); w += 8;
            vpadalq_s16(ACC1, vmull_s8(Row0C, vld1_s8(w))); w += 8;
            vpadalq_s16(ACC2, vmull_s8(Row0C, vld1_s8(w))); w += 8;
            vpadalq_s16(ACC3, vmull_s8(Row0C, vld1_s8(w))); w += 8;
            vpadalq_s16(ACC4, vmull_s8(Row0C, vld1_s8(w))); w += 8;
            vpadalq_s16(ACC5, vmull_s8(Row0C, vld1_s8(w))); w += 8;
            vpadalq_s16(ACC6, vmull_s8(Row0C, vld1_s8(w))); w += 8;
            vpadalq_s16(ACC7, vmull_s8(Row0C, vld1_s8(w))); w += 8;
        }
    }

    int32x4x2_t ACC01 = vuzpq_s32(ACC0, ACC1);
    int32x4x2_t ACC23 = vuzpq_s32(ACC2, ACC3);
    int32x4x2_t ACC45 = vuzpq_s32(ACC4, ACC5);
    int32x4x2_t ACC67 = vuzpq_s32(ACC6, ACC7);

    ACC0 = vaddq_s32(ACC01.val[0], ACC01.val[1]);
    ACC2 = vaddq_s32(ACC23.val[0], ACC23.val[1]);
    ACC4 = vaddq_s32(ACC45.val[0], ACC45.val[1]);
    ACC6 = vaddq_s32(ACC67.val[0], ACC67.val[1]);

    // post processing
    float32x4_t scale = vdupq_n_f32(*(PostProcessParams->Scale));
    float32x4_t MaxValue = vdupq_n_f32(PostProcessParams->MaximumValue);
    float32x4_t MinValue = vdupq_n_f32(PostProcessParams->MinimumValue);
    int32x4_t ZeroPoint = vdupq_n_s32(PostProcessParams->OutputZeroPoint);

    float32x4_t ACCScaled0 = vmulq_f32(scale, vcvtq_f32_s32(vaddq_s32(ACC0, vld1q_s32(bias))));
    float32x4_t ACCScaled1 = vmulq_f32(scale, vcvtq_f32_s32(vaddq_s32(ACC0, vld1q_s32(bias + 4))));
    float32x4_t ACCScaled2 = vmulq_f32(scale, vcvtq_f32_s32(vaddq_s32(ACC0, vld1q_s32(bias + 8))));
    float32x4_t ACCScaled3 = vmulq_f32(scale, vcvtq_f32_s32(vaddq_s32(ACC0, vld1q_s32(bias + 12))));

    ACCScaled0 = vmaxq_f32(vminq_f32(ACCScaled0, MaxValue), MinValue);
    ACCScaled1 = vmaxq_f32(vminq_f32(ACCScaled1, MaxValue), MinValue);
    ACCScaled2 = vmaxq_f32(vminq_f32(ACCScaled2, MaxValue), MinValue);
    ACCScaled3 = vmaxq_f32(vminq_f32(ACCScaled3, MaxValue), MinValue);

    int32x4_t output0 = vaddq_s32(vcvtq_s32_f32(ACCScaled0), ZeroPoint);
    int32x4_t output1 = vaddq_s32(vcvtq_s32_f32(ACCScaled0), ZeroPoint);
    int32x4_t output2 = vaddq_s32(vcvtq_s32_f32(ACCScaled0), ZeroPoint);
    int32x4_t output3 = vaddq_s32(vcvtq_s32_f32(ACCScaled0), ZeroPoint);

    uint8x8_t vout_8x8_0 = vqmovun_s16(vcombine_s16(vqmovn_s32(output0), vqmovn_s32(output1)));
    uint8x8_t vout_8x8_1 = vqmovun_s16(vcombine_s16(vqmovn_s32(output2), vqmovn_s32(output3)));
    uint8x16_t vout = vcombine_u8(vout_8x8_0, vout_8x8_1);
    vst1q_u8(Output, vout);
}

void
MLASCALL MlasConvSymDepthwiseKernelArm64(
    const void* /*Input*/,
    const void* /*Filter*/,
    uint8_t* /*Output*/,
    size_t /*KernelSize*/,
    size_t /*Channels*/,
    size_t /*ChannelOffset*/,
    unsigned /*ChannelCount*/,
    unsigned /*OutputCount*/,
    const struct MLAS_CONV_SYM_POST_PROCESS_PARAMS* /*PostProcessParams*/,
    unsigned /*KernelFlags*/
)
{
    return;
}

const MLAS_CONV_SYM_DISPATCH MlasConvSymDispatchArm64 = {
    MlasConvSymKernelArm64,
    MlasConvSymDepthwiseKernelArm64,
    16,                                     // KernelChannelCount
    1,                                      // KernelOutputCount
    16,                                     // KernelDepthwiseChannelCount
    4,                                      // KernelDepthwiseOutputCount
};

#endif