#include "mlasi.h"

#if defined(MLAS_TARGET_ARM64)

extern "C" {

void
MLASCALL
MlasConvSymDepthwiseKernelSize25Arm(
    uint8_t const* const* InputIndirection,
    int8_t const* Filter,
    size_t Channels,
    uint8_t* Output,
    size_t OutputCount,
    MLAS_CONV_SYM_POST_PROCESS_PARAMS const* PostProcessParams,
    unsigned KernelFlags
    )
{
    const uint8x16_t vu128 = vdupq_n_u8(128);
    const int16x8_t voutput_zero_point = vld1q_dup_s16((int16_t const*)&PostProcessParams->OutputZeroPoint);
    float32x4_t vscale_0123, vscale_4567, vscale_89AB, vscale_CDEF;
    const bool is_per_channel = ((KernelFlags & MLAS_CONV_SYM_FLAG_PER_CHANNEL_SCALE) != 0);
    // Init them anyway due to some compiler will generate uninitialized warnings.
    vscale_0123 = vscale_4567 = vscale_89AB = vscale_CDEF = vld1q_dup_f32(PostProcessParams->Scale);
    while (OutputCount-- > 0) {
        const uint8_t* i00 = InputIndirection[0];
        const uint8_t* i01 = InputIndirection[1];
        const uint8_t* i02 = InputIndirection[2];
        const uint8_t* i03 = InputIndirection[3];
        const uint8_t* i04 = InputIndirection[4];
        const uint8_t* i05 = InputIndirection[5];
        const uint8_t* i06 = InputIndirection[6];
        const uint8_t* i07 = InputIndirection[7];
        const uint8_t* i08 = InputIndirection[8];
        const uint8_t* i09 = InputIndirection[9];

        const uint8_t* i10 = InputIndirection[10];
        const uint8_t* i11 = InputIndirection[11];
        const uint8_t* i12 = InputIndirection[12];
        const uint8_t* i13 = InputIndirection[13];
        const uint8_t* i14 = InputIndirection[14];
        const uint8_t* i15 = InputIndirection[15];
        const uint8_t* i16 = InputIndirection[16];
        const uint8_t* i17 = InputIndirection[17];
        const uint8_t* i18 = InputIndirection[18];
        const uint8_t* i19 = InputIndirection[19];

        const uint8_t* i20 = InputIndirection[20];
        const uint8_t* i21 = InputIndirection[21];
        const uint8_t* i22 = InputIndirection[22];
        const uint8_t* i23 = InputIndirection[23];
        const uint8_t* i24 = InputIndirection[24];

        InputIndirection += 25;
        int32_t const* bias = PostProcessParams->Bias;
        float const* scale = PostProcessParams->Scale;
        for (size_t c = 0; c < Channels; c += 16) {
            int8_t const* w = Filter + c;
            int32x4_t vacc_0123 = vld1q_s32(bias); bias += 4;
            int32x4_t vacc_4567 = vld1q_s32(bias); bias += 4;
            int32x4_t vacc_89AB = vld1q_s32(bias); bias += 4;
            int32x4_t vacc_CDEF = vld1q_s32(bias); bias += 4;

            // kernel pixel 0, 1
            const int8x16_t vi00 = vreinterpretq_s8_u8(veorq_u8(vu128, vld1q_u8(i00))); i00 += 16;
            const int8x16_t vk00 = vld1q_s8(w); w += Channels;
            int16x8_t vprod_01234567 = vmull_s8(vget_low_s8(vi00), vget_low_s8(vk00));
            int16x8_t vprod_89ABCDEF = vmull_s8(vget_high_s8(vi00), vget_high_s8(vk00));

            const int8x16_t vi01 = vreinterpretq_s8_u8(veorq_u8(vu128, vld1q_u8(i01))); i01 += 16;
            const int8x16_t vk01 = vld1q_s8(w); w += Channels;
            vprod_01234567 = vmlal_s8(vprod_01234567, vget_low_s8(vi01), vget_low_s8(vk01));
            vprod_89ABCDEF = vmlal_s8(vprod_89ABCDEF, vget_high_s8(vi01), vget_high_s8(vk01));

            vacc_0123 = vaddw_s16(vacc_0123, vget_low_s16(vprod_01234567));
            vacc_4567 = vaddw_s16(vacc_4567, vget_high_s16(vprod_01234567));
            vacc_89AB = vaddw_s16(vacc_89AB, vget_low_s16(vprod_89ABCDEF));
            vacc_CDEF = vaddw_s16(vacc_CDEF, vget_high_s16(vprod_89ABCDEF));

            // kernel pixel 2, 3
            const int8x16_t vi02 = vreinterpretq_s8_u8(veorq_u8(vu128, vld1q_u8(i02))); i02 += 16;
            const int8x16_t vk02 = vld1q_s8(w); w += Channels;
            vprod_01234567 = vmull_s8(vget_low_s8(vi02), vget_low_s8(vk02));
            vprod_89ABCDEF = vmull_s8(vget_high_s8(vi02), vget_high_s8(vk02));

            const int8x16_t vi03 = vreinterpretq_s8_u8(veorq_u8(vu128, vld1q_u8(i03))); i03 += 16;
            const int8x16_t vk03 = vld1q_s8(w); w += Channels;
            vprod_01234567 = vmlal_s8(vprod_01234567, vget_low_s8(vi03), vget_low_s8(vk03));
            vprod_89ABCDEF = vmlal_s8(vprod_89ABCDEF, vget_high_s8(vi03), vget_high_s8(vk03));

            vacc_0123 = vaddw_s16(vacc_0123, vget_low_s16(vprod_01234567));
            vacc_4567 = vaddw_s16(vacc_4567, vget_high_s16(vprod_01234567));
            vacc_89AB = vaddw_s16(vacc_89AB, vget_low_s16(vprod_89ABCDEF));
            vacc_CDEF = vaddw_s16(vacc_CDEF, vget_high_s16(vprod_89ABCDEF));

            // kernel pixel 4, 5
            const int8x16_t vi04 = vreinterpretq_s8_u8(veorq_u8(vu128, vld1q_u8(i04))); i04 += 16;
            const int8x16_t vk04 = vld1q_s8(w); w += Channels;
            vprod_01234567 = vmull_s8(vget_low_s8(vi04), vget_low_s8(vk04));
            vprod_89ABCDEF = vmull_s8(vget_high_s8(vi04), vget_high_s8(vk04));

            const int8x16_t vi05 = vreinterpretq_s8_u8(veorq_u8(vu128, vld1q_u8(i05))); i05 += 16;
            const int8x16_t vk05 = vld1q_s8(w); w += Channels;
            vprod_01234567 = vmlal_s8(vprod_01234567, vget_low_s8(vi05), vget_low_s8(vk05));
            vprod_89ABCDEF = vmlal_s8(vprod_89ABCDEF, vget_high_s8(vi05), vget_high_s8(vk05));

            vacc_0123 = vaddw_s16(vacc_0123, vget_low_s16(vprod_01234567));
            vacc_4567 = vaddw_s16(vacc_4567, vget_high_s16(vprod_01234567));
            vacc_89AB = vaddw_s16(vacc_89AB, vget_low_s16(vprod_89ABCDEF));
            vacc_CDEF = vaddw_s16(vacc_CDEF, vget_high_s16(vprod_89ABCDEF));

            // kernel pixel 6, 7
            const int8x16_t vi06 = vreinterpretq_s8_u8(veorq_u8(vu128, vld1q_u8(i06))); i06 += 16;
            const int8x16_t vk06 = vld1q_s8(w); w += Channels;
            vprod_01234567 = vmull_s8(vget_low_s8(vi06), vget_low_s8(vk06));
            vprod_89ABCDEF = vmull_s8(vget_high_s8(vi06), vget_high_s8(vk06));

            const int8x16_t vi07 = vreinterpretq_s8_u8(veorq_u8(vu128, vld1q_u8(i07))); i07 += 16;
            const int8x16_t vk07 = vld1q_s8(w); w += Channels;
            vprod_01234567 = vmlal_s8(vprod_01234567, vget_low_s8(vi07), vget_low_s8(vk07));
            vprod_89ABCDEF = vmlal_s8(vprod_89ABCDEF, vget_high_s8(vi07), vget_high_s8(vk07));

            vacc_0123 = vaddw_s16(vacc_0123, vget_low_s16(vprod_01234567));
            vacc_4567 = vaddw_s16(vacc_4567, vget_high_s16(vprod_01234567));
            vacc_89AB = vaddw_s16(vacc_89AB, vget_low_s16(vprod_89ABCDEF));
            vacc_CDEF = vaddw_s16(vacc_CDEF, vget_high_s16(vprod_89ABCDEF));

            // kernel pixel 8, 9
            const int8x16_t vi08 = vreinterpretq_s8_u8(veorq_u8(vu128, vld1q_u8(i08))); i08 += 16;
            const int8x16_t vk08 = vld1q_s8(w); w += Channels;
            vprod_01234567 = vmull_s8(vget_low_s8(vi08), vget_low_s8(vk08));
            vprod_89ABCDEF = vmull_s8(vget_high_s8(vi08), vget_high_s8(vk08));

            const int8x16_t vi09 = vreinterpretq_s8_u8(veorq_u8(vu128, vld1q_u8(i09))); i09 += 16;
            const int8x16_t vk09 = vld1q_s8(w); w += Channels;
            vprod_01234567 = vmlal_s8(vprod_01234567, vget_low_s8(vi09), vget_low_s8(vk09));
            vprod_89ABCDEF = vmlal_s8(vprod_89ABCDEF, vget_high_s8(vi09), vget_high_s8(vk09));

            vacc_0123 = vaddw_s16(vacc_0123, vget_low_s16(vprod_01234567));
            vacc_4567 = vaddw_s16(vacc_4567, vget_high_s16(vprod_01234567));
            vacc_89AB = vaddw_s16(vacc_89AB, vget_low_s16(vprod_89ABCDEF));
            vacc_CDEF = vaddw_s16(vacc_CDEF, vget_high_s16(vprod_89ABCDEF));

            // kernel pixel 10, 11
            const int8x16_t vi10 = vreinterpretq_s8_u8(veorq_u8(vu128, vld1q_u8(i10))); i10 += 16;
            const int8x16_t vk10 = vld1q_s8(w); w += Channels;
            vprod_01234567 = vmull_s8(vget_low_s8(vi10), vget_low_s8(vk10));
            vprod_89ABCDEF = vmull_s8(vget_high_s8(vi10), vget_high_s8(vk10));

            const int8x16_t vi11 = vreinterpretq_s8_u8(veorq_u8(vu128, vld1q_u8(i11))); i11 += 16;
            const int8x16_t vk11 = vld1q_s8(w); w += Channels;
            vprod_01234567 = vmlal_s8(vprod_01234567, vget_low_s8(vi11), vget_low_s8(vk11));
            vprod_89ABCDEF = vmlal_s8(vprod_89ABCDEF, vget_high_s8(vi11), vget_high_s8(vk11));

            vacc_0123 = vaddw_s16(vacc_0123, vget_low_s16(vprod_01234567));
            vacc_4567 = vaddw_s16(vacc_4567, vget_high_s16(vprod_01234567));
            vacc_89AB = vaddw_s16(vacc_89AB, vget_low_s16(vprod_89ABCDEF));
            vacc_CDEF = vaddw_s16(vacc_CDEF, vget_high_s16(vprod_89ABCDEF));

            // kernel pixel 12, 13
            const int8x16_t vi12 = vreinterpretq_s8_u8(veorq_u8(vu128, vld1q_u8(i12))); i12 += 16;
            const int8x16_t vk12 = vld1q_s8(w); w += Channels;
            vprod_01234567 = vmull_s8(vget_low_s8(vi12), vget_low_s8(vk12));
            vprod_89ABCDEF = vmull_s8(vget_high_s8(vi12), vget_high_s8(vk12));

            const int8x16_t vi13 = vreinterpretq_s8_u8(veorq_u8(vu128, vld1q_u8(i13))); i13 += 16;
            const int8x16_t vk13 = vld1q_s8(w); w += Channels;
            vprod_01234567 = vmlal_s8(vprod_01234567, vget_low_s8(vi13), vget_low_s8(vk13));
            vprod_89ABCDEF = vmlal_s8(vprod_89ABCDEF, vget_high_s8(vi13), vget_high_s8(vk13));

            vacc_0123 = vaddw_s16(vacc_0123, vget_low_s16(vprod_01234567));
            vacc_4567 = vaddw_s16(vacc_4567, vget_high_s16(vprod_01234567));
            vacc_89AB = vaddw_s16(vacc_89AB, vget_low_s16(vprod_89ABCDEF));
            vacc_CDEF = vaddw_s16(vacc_CDEF, vget_high_s16(vprod_89ABCDEF));

            // kernel pixel 14, 15
            const int8x16_t vi14 = vreinterpretq_s8_u8(veorq_u8(vu128, vld1q_u8(i14))); i14 += 16;
            const int8x16_t vk14 = vld1q_s8(w); w += Channels;
            vprod_01234567 = vmull_s8(vget_low_s8(vi14), vget_low_s8(vk14));
            vprod_89ABCDEF = vmull_s8(vget_high_s8(vi14), vget_high_s8(vk14));

            const int8x16_t vi15 = vreinterpretq_s8_u8(veorq_u8(vu128, vld1q_u8(i15))); i15 += 16;
            const int8x16_t vk15 = vld1q_s8(w); w += Channels;
            vprod_01234567 = vmlal_s8(vprod_01234567, vget_low_s8(vi15), vget_low_s8(vk15));
            vprod_89ABCDEF = vmlal_s8(vprod_89ABCDEF, vget_high_s8(vi15), vget_high_s8(vk15));

            vacc_0123 = vaddw_s16(vacc_0123, vget_low_s16(vprod_01234567));
            vacc_4567 = vaddw_s16(vacc_4567, vget_high_s16(vprod_01234567));
            vacc_89AB = vaddw_s16(vacc_89AB, vget_low_s16(vprod_89ABCDEF));
            vacc_CDEF = vaddw_s16(vacc_CDEF, vget_high_s16(vprod_89ABCDEF));

            // kernel pixel 16, 17
            const int8x16_t vi16 = vreinterpretq_s8_u8(veorq_u8(vu128, vld1q_u8(i16))); i16 += 16;
            const int8x16_t vk16 = vld1q_s8(w); w += Channels;
            vprod_01234567 = vmull_s8(vget_low_s8(vi16), vget_low_s8(vk16));
            vprod_89ABCDEF = vmull_s8(vget_high_s8(vi16), vget_high_s8(vk16));

            const int8x16_t vi17 = vreinterpretq_s8_u8(veorq_u8(vu128, vld1q_u8(i17))); i17 += 16;
            const int8x16_t vk17 = vld1q_s8(w); w += Channels;
            vprod_01234567 = vmlal_s8(vprod_01234567, vget_low_s8(vi17), vget_low_s8(vk17));
            vprod_89ABCDEF = vmlal_s8(vprod_89ABCDEF, vget_high_s8(vi17), vget_high_s8(vk17));

            vacc_0123 = vaddw_s16(vacc_0123, vget_low_s16(vprod_01234567));
            vacc_4567 = vaddw_s16(vacc_4567, vget_high_s16(vprod_01234567));
            vacc_89AB = vaddw_s16(vacc_89AB, vget_low_s16(vprod_89ABCDEF));
            vacc_CDEF = vaddw_s16(vacc_CDEF, vget_high_s16(vprod_89ABCDEF));

            // kernel pixel 18, 19
            const int8x16_t vi18 = vreinterpretq_s8_u8(veorq_u8(vu128, vld1q_u8(i18))); i18 += 16;
            const int8x16_t vk18 = vld1q_s8(w); w += Channels;
            vprod_01234567 = vmull_s8(vget_low_s8(vi18), vget_low_s8(vk18));
            vprod_89ABCDEF = vmull_s8(vget_high_s8(vi18), vget_high_s8(vk18));

            const int8x16_t vi19 = vreinterpretq_s8_u8(veorq_u8(vu128, vld1q_u8(i19))); i19 += 16;
            const int8x16_t vk19 = vld1q_s8(w); w += Channels;
            vprod_01234567 = vmlal_s8(vprod_01234567, vget_low_s8(vi19), vget_low_s8(vk19));
            vprod_89ABCDEF = vmlal_s8(vprod_89ABCDEF, vget_high_s8(vi19), vget_high_s8(vk19));

            vacc_0123 = vaddw_s16(vacc_0123, vget_low_s16(vprod_01234567));
            vacc_4567 = vaddw_s16(vacc_4567, vget_high_s16(vprod_01234567));
            vacc_89AB = vaddw_s16(vacc_89AB, vget_low_s16(vprod_89ABCDEF));
            vacc_CDEF = vaddw_s16(vacc_CDEF, vget_high_s16(vprod_89ABCDEF));

            // kernel pixel 20, 21
            const int8x16_t vi20 = vreinterpretq_s8_u8(veorq_u8(vu128, vld1q_u8(i20))); i20 += 16;
            const int8x16_t vk20 = vld1q_s8(w); w += Channels;
            vprod_01234567 = vmull_s8(vget_low_s8(vi20), vget_low_s8(vk20));
            vprod_89ABCDEF = vmull_s8(vget_high_s8(vi20), vget_high_s8(vk20));

            const int8x16_t vi21 = vreinterpretq_s8_u8(veorq_u8(vu128, vld1q_u8(i21))); i21 += 16;
            const int8x16_t vk21 = vld1q_s8(w); w += Channels;
            vprod_01234567 = vmlal_s8(vprod_01234567, vget_low_s8(vi21), vget_low_s8(vk21));
            vprod_89ABCDEF = vmlal_s8(vprod_89ABCDEF, vget_high_s8(vi21), vget_high_s8(vk21));

            vacc_0123 = vaddw_s16(vacc_0123, vget_low_s16(vprod_01234567));
            vacc_4567 = vaddw_s16(vacc_4567, vget_high_s16(vprod_01234567));
            vacc_89AB = vaddw_s16(vacc_89AB, vget_low_s16(vprod_89ABCDEF));
            vacc_CDEF = vaddw_s16(vacc_CDEF, vget_high_s16(vprod_89ABCDEF));

            // kernel pixel 22, 23
            const int8x16_t vi22 = vreinterpretq_s8_u8(veorq_u8(vu128, vld1q_u8(i22))); i22 += 16;
            const int8x16_t vk22 = vld1q_s8(w); w += Channels;
            vprod_01234567 = vmull_s8(vget_low_s8(vi22), vget_low_s8(vk22));
            vprod_89ABCDEF = vmull_s8(vget_high_s8(vi22), vget_high_s8(vk22));

            const int8x16_t vi23 = vreinterpretq_s8_u8(veorq_u8(vu128, vld1q_u8(i23))); i23 += 16;
            const int8x16_t vk23 = vld1q_s8(w); w += Channels;
            vprod_01234567 = vmlal_s8(vprod_01234567, vget_low_s8(vi23), vget_low_s8(vk23));
            vprod_89ABCDEF = vmlal_s8(vprod_89ABCDEF, vget_high_s8(vi23), vget_high_s8(vk23));

            vacc_0123 = vaddw_s16(vacc_0123, vget_low_s16(vprod_01234567));
            vacc_4567 = vaddw_s16(vacc_4567, vget_high_s16(vprod_01234567));
            vacc_89AB = vaddw_s16(vacc_89AB, vget_low_s16(vprod_89ABCDEF));
            vacc_CDEF = vaddw_s16(vacc_CDEF, vget_high_s16(vprod_89ABCDEF));

            // kernel pixel 24
            const int8x16_t vi24 = vreinterpretq_s8_u8(veorq_u8(vu128, vld1q_u8(i24))); i24 += 16;
            const int8x16_t vk24 = vld1q_s8(w);  // w += Channels; no need to add
            vprod_01234567 = vmull_s8(vget_low_s8(vi24), vget_low_s8(vk24));
            vprod_89ABCDEF = vmull_s8(vget_high_s8(vi24), vget_high_s8(vk24));

            vacc_0123 = vaddw_s16(vacc_0123, vget_low_s16(vprod_01234567));
            vacc_4567 = vaddw_s16(vacc_4567, vget_high_s16(vprod_01234567));
            vacc_89AB = vaddw_s16(vacc_89AB, vget_low_s16(vprod_89ABCDEF));
            vacc_CDEF = vaddw_s16(vacc_CDEF, vget_high_s16(vprod_89ABCDEF));

            if (is_per_channel) {
                vscale_0123 = vld1q_f32(scale); scale += 4;
                vscale_4567 = vld1q_f32(scale); scale += 4;
                vscale_89AB = vld1q_f32(scale); scale += 4;
                vscale_CDEF = vld1q_f32(scale); scale += 4;
            }

            // requantize
            vacc_0123 = vcvtnq_s32_f32(vmulq_f32(vcvtq_f32_s32(vacc_0123), vscale_0123));
            vacc_4567 = vcvtnq_s32_f32(vmulq_f32(vcvtq_f32_s32(vacc_4567), vscale_4567));
            vacc_89AB = vcvtnq_s32_f32(vmulq_f32(vcvtq_f32_s32(vacc_89AB), vscale_89AB));
            vacc_CDEF = vcvtnq_s32_f32(vmulq_f32(vcvtq_f32_s32(vacc_CDEF), vscale_CDEF));

            const int16x8_t vacc_01234567 = vqaddq_s16(vqmovn_high_s32(vqmovn_s32(vacc_0123), vacc_4567), voutput_zero_point);
            const int16x8_t vacc_89ABCDEF = vqaddq_s16(vqmovn_high_s32(vqmovn_s32(vacc_89AB), vacc_CDEF), voutput_zero_point);
            uint8x16_t vout = vqmovun_high_s16(vqmovun_s16(vacc_01234567), vacc_89ABCDEF);

            vst1q_u8(Output, vout);
            Output += 16;
        }
    }
}

}

#endif
