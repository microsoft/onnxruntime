/*++

Copyright 2025 FUJITSU LIMITED

Module Name:

   Gelu.cpp

Abstract:

    This module contains  Gelu helper functions .

--*/
#include "gelu.h"

#if defined(__ARM_NEON) && defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)

void
MLASCALL
MlasNeonGeluF16Kernel(const MLAS_FP16* input, MLAS_FP16* output, MLAS_FP16* temp, int64_t count, const std::string& algo)
{
    const float16_t v_half1 = 0.5f;
    const float16_t v_one1 = 1.0f;
    const float16_t v_sqrt1_21 = static_cast<float>(M_SQRT1_2);
    const float16_t v_B1 = 0.7978845608028654f;
    const float16_t v_C1 = 0.035677408136300125f;
    const float16_t c1 = 5.0f;
    const float16_t c2 = -5.0f;
    const MLAS_FLOAT16X8 v_half = MlasBroadcastF16Float16x8(v_half1);
    const MLAS_FLOAT16X8 v_one = MlasBroadcastF16Float16x8(v_one1);
    const MLAS_FLOAT16X8 v_sqrt1_2 = MlasBroadcastF16Float16x8(v_sqrt1_21);
    const MLAS_FLOAT16X8 v_B = MlasBroadcastF16Float16x8(v_B1);
    const MLAS_FLOAT16X8 v_C = MlasBroadcastF16Float16x8(v_C1);

    int64_t i = 0;

    if (algo == "tanh") {
        // Preprocess input into temp[] for tanh
        for (; i + 7 < count; i += 8) {
            MLAS_FLOAT16X8 x = MlasLoadf16Float16x8(reinterpret_cast<const float16_t*>(input + i));
            MLAS_FLOAT16X8 x2 = MlasMultiplyFloat16(x, x);
            MLAS_FLOAT16X8 inner = MlasMultiplyAddFloat16(v_C, x2, v_B);  // B + C * x^2
            MLAS_FLOAT16X8 tanh_arg = MlasMultiplyFloat16(x, inner);      // x * (B + C * x^2)
            tanh_arg = MlasMaximumFloat16(MlasBroadcastF16Float16x8(c2), MlasMinimumFloat16(tanh_arg, MlasBroadcastF16Float16x8(c1)));
            MlasStoref16Float16x8(reinterpret_cast<float16_t*>(temp + i), tanh_arg);
        }

        // Tail
        for (; i < count; ++i) {
            float x = static_cast<float>(input[i]);
            float inner = x * (0.7979f + 0.03568f * x * x);
            inner = std::max(-5.0f, std::min(5.0f, inner));
            temp[i] = static_cast<MLAS_FP16>(inner);
        }

        // Tanh processing
        MlasComputeTanh<MLAS_FP16>(temp, temp, count);

    } else if (algo == "none") {
        // Preprocess input into temp[] for erf
        for (i = 0; i + 7 < count; i += 8) {
            MLAS_FLOAT16X8 x = MlasLoadf16Float16x8(reinterpret_cast<const float16_t*>(input + i));
            MLAS_FLOAT16X8 scaled = MlasMultiplyFloat16(x, v_sqrt1_2);
            MlasStoref16Float16x8(reinterpret_cast<float16_t*>(temp + i), scaled);
        }

        // Tail
        for (; i < count; ++i) {
            float x = static_cast<float>(input[i]);
            temp[i] = static_cast<MLAS_FP16>(x * 0.70710678f);
        }

        // Erf processing
        MlasNeonErfF16Kernel(reinterpret_cast<const _mlas_fp16_*>(temp), reinterpret_cast<_mlas_fp16_*>(temp), count);
    }

    // Final GELU output = 0.5 * x * (1 + tanh|erf)
    i = 0;
    for (; i + 7 < count; i += 8) {
        MLAS_FLOAT16X8 x = MlasLoadf16Float16x8(reinterpret_cast<const float16_t*>(input + i));
        MLAS_FLOAT16X8 t = MlasLoadf16Float16x8(reinterpret_cast<const float16_t*>(temp + i));
        MLAS_FLOAT16X8 result = MlasMultiplyFloat16(v_half, MlasMultiplyFloat16(x, MlasAddFloat16(v_one, t)));
        MlasStoref16Float16x8(reinterpret_cast<float16_t*>(output + i), result);
    }

    for (; i < count; ++i) {
        float x = static_cast<float>(input[i]);
        float t = static_cast<float>(temp[i]);
        float gelu = 0.5f * x * (1.0f + t);
        output[i] = static_cast<MLAS_FP16>(gelu);
    }
}
#endif