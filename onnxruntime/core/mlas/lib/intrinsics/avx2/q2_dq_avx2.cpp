/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    q2_dq_avx2.cpp

Abstract:

    This module implements the AVX2 kernel for blockwise 2-bit dequantization
    with floating point zero points.

    Every uint32 of packed data holds 16 consecutive 2-bit elements, least
    significant bits first. The word is broadcast across all lanes and each
    lane extracts its element with a variable shift. The scale is applied as
    a separate multiply and add (not fused) so the results stay bit-identical
    to the scalar kernel. GCC and Clang can fuse the two into an FMA in their
    default configurations, which would change the result, so the build passes
    -ffp-contract=off for this TU; MSVC does not contract under its default
    /fp:precise. The scalar kernel's TU is compiled without FMA in the default
    build and cannot contract; a build that enables FMA globally (for example
    -march=x86-64-v3) can still fuse there and drift from this kernel by one
    ulp.

--*/

#include "../../mlasi.h"

#include <cstring>

void
MLASCALL
MlasDequantizeBlockwise2BitsKernelAvx2(
    float* Output,
    const uint8_t* PackedData,
    size_t N,
    float Scale,
    float ZeroPointAdjust
)
{
    const __m256i lo_shifts = _mm256_setr_epi32(0, 2, 4, 6, 8, 10, 12, 14);
    const __m256i hi_shifts = _mm256_setr_epi32(16, 18, 20, 22, 24, 26, 28, 30);
    const __m256i mask3 = _mm256_set1_epi32(3);
    const __m256 scale = _mm256_set1_ps(Scale);
    const __m256 zp_adjust = _mm256_set1_ps(ZeroPointAdjust);

    size_t i = 0;
    for (; i + 16 <= N; i += 16) {
        uint32_t bits;
        memcpy(&bits, PackedData + i / 4, sizeof(bits));
        const __m256i vbits = _mm256_set1_epi32(static_cast<int>(bits));
        const __m256i qlo = _mm256_and_si256(_mm256_srlv_epi32(vbits, lo_shifts), mask3);
        const __m256i qhi = _mm256_and_si256(_mm256_srlv_epi32(vbits, hi_shifts), mask3);
        const __m256 flo = _mm256_add_ps(_mm256_mul_ps(_mm256_cvtepi32_ps(qlo), scale), zp_adjust);
        const __m256 fhi = _mm256_add_ps(_mm256_mul_ps(_mm256_cvtepi32_ps(qhi), scale), zp_adjust);
        _mm256_storeu_ps(Output + i, flo);
        _mm256_storeu_ps(Output + i + 8, fhi);
    }

    for (; i < N; i++) {
        const uint8_t packed = PackedData[i >> 2];
        const float q = static_cast<float>((packed >> (2 * (i & 3))) & 0x3);
        Output[i] = q * Scale + ZeroPointAdjust;
    }
}
