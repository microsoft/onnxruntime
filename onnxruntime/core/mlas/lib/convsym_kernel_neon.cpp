#include "mlasi.h"
#include "convsym.h"

extern "C" {
MLAS_CONV_SYM_KERNEL MlasConvSymKernelNeon;
}

void
MLASCALL MlasConvSymKernelArm64(
    const void* Input,
    const void* Filter,
    uint8_t* Output,
    size_t KernelSize,
    size_t InputChannels,
    size_t /*OutputChannels*/,
    unsigned /*ChannelCount*/,
    unsigned /*OutputCount*/,
    const struct MLAS_CONV_SYM_POST_PROCESS_PARAMS* PostProcessParams,
    unsigned KernelFlags
)
{
    const int32_t* bias = PostProcessParams->Bias;
    int32x4_t ACC0 = vdupq_n_s32(0);
    int32x4_t ACC1 = vdupq_n_s32(0);
    int32x4_t ACC2 = vdupq_n_s32(0);
    int32x4_t ACC3 = vdupq_n_s32(0);
    int32x4_t ACC4 = vdupq_n_s32(0);
    int32x4_t ACC5 = vdupq_n_s32(0);
    int32x4_t ACC6 = vdupq_n_s32(0);
    int32x4_t ACC7 = vdupq_n_s32(0);
    int8x8_t BitFlip = vdup_n_s8(-128);


    bool IsIndirect = !(KernelFlags & MLAS_CONV_SYM_FLAG_INPUT_DIRECT);
    const int8_t* a0 = static_cast<const int8_t*> (Input);
    const int8_t* w = static_cast<const int8_t*> (Filter);
    for (size_t k = 0; k < KernelSize; k++) {
        if (IsIndirect) {
            int8_t* const* InputIndirect = reinterpret_cast<int8_t* const*>(Input);
            a0 = InputIndirect[k];
        }

        for (size_t c = 0; c < InputChannels; c += 4, a0 += 4) {
            int8x8_t Row0C = vreinterpret_s8_s32(vdup_n_s32(*(reinterpret_cast<const int32_t*>(a0 + c))));
            Row0C = veor_s8(Row0C, BitFlip);
            ACC0 = vpadalq_s16(ACC0, vmull_s8(Row0C, vld1_s8(w))); w += 8;
            ACC1 = vpadalq_s16(ACC1, vmull_s8(Row0C, vld1_s8(w))); w += 8;
            ACC2 = vpadalq_s16(ACC2, vmull_s8(Row0C, vld1_s8(w))); w += 8;
            ACC3 = vpadalq_s16(ACC3, vmull_s8(Row0C, vld1_s8(w))); w += 8;
            ACC4 = vpadalq_s16(ACC4, vmull_s8(Row0C, vld1_s8(w))); w += 8;
            ACC5 = vpadalq_s16(ACC5, vmull_s8(Row0C, vld1_s8(w))); w += 8;
            ACC6 = vpadalq_s16(ACC6, vmull_s8(Row0C, vld1_s8(w))); w += 8;
            ACC7 = vpadalq_s16(ACC7, vmull_s8(Row0C, vld1_s8(w))); w += 8;
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

    ACC0 = vaddq_s32(ACC0, vld1q_s32(bias));
    ACC2 = vaddq_s32(ACC2, vld1q_s32(bias + 4));
    ACC4 = vaddq_s32(ACC4, vld1q_s32(bias + 8));
    ACC6 = vaddq_s32(ACC6, vld1q_s32(bias + 12));

    float32x4_t ACCScaled0 = vmulq_f32(scale, vcvtq_f32_s32(ACC0));
    float32x4_t ACCScaled1 = vmulq_f32(scale, vcvtq_f32_s32(ACC2));
    float32x4_t ACCScaled2 = vmulq_f32(scale, vcvtq_f32_s32(ACC4));
    float32x4_t ACCScaled3 = vmulq_f32(scale, vcvtq_f32_s32(ACC6));

    ACCScaled0 = vmaxq_f32(vminq_f32(ACCScaled0, MaxValue), MinValue);
    ACCScaled1 = vmaxq_f32(vminq_f32(ACCScaled1, MaxValue), MinValue);
    ACCScaled2 = vmaxq_f32(vminq_f32(ACCScaled2, MaxValue), MinValue);
    ACCScaled3 = vmaxq_f32(vminq_f32(ACCScaled3, MaxValue), MinValue);

    int32x4_t output0 = vaddq_s32(vcvtnq_s32_f32(ACCScaled0), ZeroPoint);
    int32x4_t output1 = vaddq_s32(vcvtnq_s32_f32(ACCScaled1), ZeroPoint);
    int32x4_t output2 = vaddq_s32(vcvtnq_s32_f32(ACCScaled2), ZeroPoint);
    int32x4_t output3 = vaddq_s32(vcvtnq_s32_f32(ACCScaled3), ZeroPoint);

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

// intrinsics
// const MLAS_CONV_SYM_DISPATCH MlasConvSymDispatchArm64 = {
//     //MlasConvSymKernelArm64,
//     MlasConvSymKernelNeon,
//     MlasConvSymDepthwiseKernelArm64,
//     16,                                      // KernelChannelCount
//     1,                                      // KernelOutputCount
//     16,                                     // KernelDepthwiseChannelCount
//     4,                                      // KernelDepthwiseOutputCount
// };

const MLAS_CONV_SYM_DISPATCH MlasConvSymDispatchArm64 = {
    MlasConvSymKernelNeon,
    MlasConvSymDepthwiseKernelArm64,
    8,                                      // KernelChannelCount
    2,                                      // KernelOutputCount
    16,                                     // KernelDepthwiseChannelCount
    4,                                      // KernelDepthwiseOutputCount
};