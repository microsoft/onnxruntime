/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    CastPowerVSX.cpp

Abstract:

    FP16 <-> FP32 conversion kernels for POWER architectures using Power VSX instructions.

--*/

#include "mlasi.h"
#include "mlas_float16.h"

#if defined(__powerpc__) || defined(__ppc__) || defined(_ARCH_PPC)
#include <altivec.h>
#undef vector
#undef pixel
#undef bool

#define PREFETCH_READ(addr) \
    asm volatile("dcbt 0, %0" ::"r"(addr) : "memory");
#endif

void
MLASCALL
MlasCastF16ToF32KernelPowerVSX(
    const unsigned short* src,
    float* dest,
    size_t count
    )
/*++

Routine Description:

    Convert an array of FP16 (half-precision) values to FP32 using Power VSX instructions.

Arguments:

    src   - Source buffer of FP16 values stored as uint16_t.
    dest  - Destination buffer of FP32 values.
    count - Number of elements to convert.

--*/
{
    size_t i = 0;

#if defined(__powerpc__) || defined(__ppc__) || defined(_ARCH_PPC)
    // 64-element unrolled loop:
    // Reads 128 bytes (1 full POWER10 cache line from src)
    // Writes 256 bytes (2 full POWER10 cache lines to dest)
    for (; i + 64 <= count; i += 64) {
        PREFETCH_READ(src + i + 128);

        __vector unsigned short in0 = vec_xl(0,   src + i);
        __vector unsigned short in1 = vec_xl(16,  src + i);
        __vector unsigned short in2 = vec_xl(32,  src + i);
        __vector unsigned short in3 = vec_xl(48,  src + i);
        __vector unsigned short in4 = vec_xl(64,  src + i);
        __vector unsigned short in5 = vec_xl(80,  src + i);
        __vector unsigned short in6 = vec_xl(96,  src + i);
        __vector unsigned short in7 = vec_xl(112, src + i);

        __vector float f0  = vec_extract_fp32_from_shortl(in0);
        __vector float f1  = vec_extract_fp32_from_shorth(in0);
        __vector float f2  = vec_extract_fp32_from_shortl(in1);
        __vector float f3  = vec_extract_fp32_from_shorth(in1);
        __vector float f4  = vec_extract_fp32_from_shortl(in2);
        __vector float f5  = vec_extract_fp32_from_shorth(in2);
        __vector float f6  = vec_extract_fp32_from_shortl(in3);
        __vector float f7  = vec_extract_fp32_from_shorth(in3);
        __vector float f8  = vec_extract_fp32_from_shortl(in4);
        __vector float f9  = vec_extract_fp32_from_shorth(in4);
        __vector float f10 = vec_extract_fp32_from_shortl(in5);
        __vector float f11 = vec_extract_fp32_from_shorth(in5);
        __vector float f12 = vec_extract_fp32_from_shortl(in6);
        __vector float f13 = vec_extract_fp32_from_shorth(in6);
        __vector float f14 = vec_extract_fp32_from_shortl(in7);
        __vector float f15 = vec_extract_fp32_from_shorth(in7);

        vec_xst(f0,  0,   dest + i);
        vec_xst(f1,  16,  dest + i);
        vec_xst(f2,  32,  dest + i);
        vec_xst(f3,  48,  dest + i);
        vec_xst(f4,  64,  dest + i);
        vec_xst(f5,  80,  dest + i);
        vec_xst(f6,  96,  dest + i);
        vec_xst(f7,  112, dest + i);
        vec_xst(f8,  128, dest + i);
        vec_xst(f9,  144, dest + i);
        vec_xst(f10, 160, dest + i);
        vec_xst(f11, 176, dest + i);
        vec_xst(f12, 192, dest + i);
        vec_xst(f13, 208, dest + i);
        vec_xst(f14, 224, dest + i);
        vec_xst(f15, 240, dest + i);
    }

    // 16-element loop
    for (; i + 16 <= count; i += 16) {
        __vector unsigned short in0 = vec_xl(0,  src + i);
        __vector unsigned short in1 = vec_xl(16, src + i);

        __vector float f0 = vec_extract_fp32_from_shortl(in0);
        __vector float f1 = vec_extract_fp32_from_shorth(in0);
        __vector float f2 = vec_extract_fp32_from_shortl(in1);
        __vector float f3 = vec_extract_fp32_from_shorth(in1);

        vec_xst(f0, 0,  dest + i);
        vec_xst(f1, 16, dest + i);
        vec_xst(f2, 32, dest + i);
        vec_xst(f3, 48, dest + i);
    }

    // 8-element loop
    for (; i + 8 <= count; i += 8) {
        __vector unsigned short in = vec_xl(0, src + i);

        __vector float f0 = vec_extract_fp32_from_shortl(in);
        __vector float f1 = vec_extract_fp32_from_shorth(in);

        vec_xst(f0, 0,  dest + i);
        vec_xst(f1, 16, dest + i);
    }
#endif

    // Scalar fallback for remaining elements (< 8)
    for (; i < count; ++i) {
        dest[i] = MLAS_Half2Float(src[i]);
    }
}

void
MLASCALL
MlasCastF32ToF16KernelPowerVSX(
    const float* src,
    unsigned short* dest,
    size_t count
    )
/*++

Routine Description:

    Convert an array of FP32 values to FP16 (half-precision) using Power VSX instructions.

Arguments:

    src   - Source buffer of FP32 values.
    dest  - Destination buffer of FP16 values (uint16_t).
    count - Number of elements to convert.

--*/
{
    size_t i = 0;

#if defined(__powerpc__) || defined(__ppc__) || defined(_ARCH_PPC)
    // 64-element unrolled loop:
    // Reads 256 bytes (2 full POWER10 cache lines from src)
    // Writes 128 bytes (1 full POWER10 cache line to dest)
    for (; i + 64 <= count; i += 64) {
        PREFETCH_READ(src + i + 128);

        __vector float f0  = vec_xl(0,   src + i);
        __vector float f1  = vec_xl(16,  src + i);
        __vector float f2  = vec_xl(32,  src + i);
        __vector float f3  = vec_xl(48,  src + i);
        __vector float f4  = vec_xl(64,  src + i);
        __vector float f5  = vec_xl(80,  src + i);
        __vector float f6  = vec_xl(96,  src + i);
        __vector float f7  = vec_xl(112, src + i);
        __vector float f8  = vec_xl(128, src + i);
        __vector float f9  = vec_xl(144, src + i);
        __vector float f10 = vec_xl(160, src + i);
        __vector float f11 = vec_xl(176, src + i);
        __vector float f12 = vec_xl(192, src + i);
        __vector float f13 = vec_xl(208, src + i);
        __vector float f14 = vec_xl(224, src + i);
        __vector float f15 = vec_xl(240, src + i);

        __vector unsigned short out0 = vec_pack_to_short_fp32(f0,  f1);
        __vector unsigned short out1 = vec_pack_to_short_fp32(f2,  f3);
        __vector unsigned short out2 = vec_pack_to_short_fp32(f4,  f5);
        __vector unsigned short out3 = vec_pack_to_short_fp32(f6,  f7);
        __vector unsigned short out4 = vec_pack_to_short_fp32(f8,  f9);
        __vector unsigned short out5 = vec_pack_to_short_fp32(f10, f11);
        __vector unsigned short out6 = vec_pack_to_short_fp32(f12, f13);
        __vector unsigned short out7 = vec_pack_to_short_fp32(f14, f15);

        vec_xst(out0, 0,   dest + i);
        vec_xst(out1, 16,  dest + i);
        vec_xst(out2, 32,  dest + i);
        vec_xst(out3, 48,  dest + i);
        vec_xst(out4, 64,  dest + i);
        vec_xst(out5, 80,  dest + i);
        vec_xst(out6, 96,  dest + i);
        vec_xst(out7, 112, dest + i);
    }

    // 16-element loop
    for (; i + 16 <= count; i += 16) {
        __vector float f0 = vec_xl(0,  src + i);
        __vector float f1 = vec_xl(16, src + i);
        __vector float f2 = vec_xl(32, src + i);
        __vector float f3 = vec_xl(48, src + i);

        __vector unsigned short out0 = vec_pack_to_short_fp32(f0, f1);
        __vector unsigned short out1 = vec_pack_to_short_fp32(f2, f3);

        vec_xst(out0, 0,  dest + i);
        vec_xst(out1, 16, dest + i);
    }

    // 8-element loop
    for (; i + 8 <= count; i += 8) {
        __vector float f0 = vec_xl(0,  src + i);
        __vector float f1 = vec_xl(16, src + i);

        __vector unsigned short out = vec_pack_to_short_fp32(f0, f1);

        vec_xst(out, 0, dest + i);
    }
#endif

    // Scalar fallback for remaining elements (< 8)
    for (; i < count; ++i) {
        dest[i] = MLAS_Float2Half(src[i]);
    }
}
