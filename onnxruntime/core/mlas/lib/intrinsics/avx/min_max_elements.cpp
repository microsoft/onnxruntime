/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    min_max_elements.cpp

Abstract:

    This module implements the logic to find min and max elements with AVX instructions.

--*/

#include "mlasi.h"

void
MLASCALL
MlasReduceMinimumMaximumF32KernelAvx(
    const float* Input,
    float* Min,
    float* Max,
    size_t N
    )
{
    float tmp_min = std::numeric_limits<float>::max();
    float tmp_max = std::numeric_limits<float>::lowest();

    if (N >= 8) {

        __m256 MaximumVector0 = _mm256_set1_ps(tmp_max);
        __m256 MinimumVector0 = _mm256_set1_ps(tmp_min);

        if (N >= 32) {

            __m256 MaximumVector1 = MaximumVector0;
            __m256 MaximumVector2 = MaximumVector0;
            __m256 MaximumVector3 = MaximumVector0;

            __m256 MinimumVector1 = MinimumVector0;
            __m256 MinimumVector2 = MinimumVector0;
            __m256 MinimumVector3 = MinimumVector0;

            while (N >= 32) {

                __m256 InputVector0 = _mm256_loadu_ps(Input);
                __m256 InputVector1 = _mm256_loadu_ps(Input + 8);
                __m256 InputVector2 = _mm256_loadu_ps(Input + 16);
                __m256 InputVector3 = _mm256_loadu_ps(Input + 24);

                MaximumVector0 = _mm256_max_ps(MaximumVector0, InputVector0);
                MaximumVector1 = _mm256_max_ps(MaximumVector1, InputVector1);
                MaximumVector2 = _mm256_max_ps(MaximumVector2, InputVector2);
                MaximumVector3 = _mm256_max_ps(MaximumVector3, InputVector3);

                MinimumVector0 = _mm256_min_ps(MinimumVector0, InputVector0);
                MinimumVector1 = _mm256_min_ps(MinimumVector1, InputVector1);
                MinimumVector2 = _mm256_min_ps(MinimumVector2, InputVector2);
                MinimumVector3 = _mm256_min_ps(MinimumVector3, InputVector3);

                Input += 32;
                N -= 32;
            }

            MaximumVector0 = _mm256_max_ps(MaximumVector0, MaximumVector1);
            MaximumVector2 = _mm256_max_ps(MaximumVector2, MaximumVector3);
            MaximumVector0 = _mm256_max_ps(MaximumVector0, MaximumVector2);

            MinimumVector0 = _mm256_min_ps(MinimumVector0, MinimumVector1);
            MinimumVector2 = _mm256_min_ps(MinimumVector2, MinimumVector3);
            MinimumVector0 = _mm256_min_ps(MinimumVector0, MinimumVector2);
        }

        while (N >= 8) {

            __m256 InputVector0 = _mm256_loadu_ps(Input);
            MaximumVector0 = _mm256_max_ps(MaximumVector0, InputVector0);
            MinimumVector0 = _mm256_min_ps(MinimumVector0, InputVector0);

            Input += 8;
            N -= 8;
        }

        __m128 low = _mm256_castps256_ps128(MaximumVector0);
        __m128 high = _mm256_extractf128_ps(MaximumVector0, 1);
        tmp_max = MlasReduceMaximumFloat32x4(MlasMaximumFloat32x4(low, high));

        low = _mm256_castps256_ps128(MinimumVector0);
        high = _mm256_extractf128_ps(MinimumVector0, 1);
        tmp_min = MlasReduceMinimumFloat32x4(MlasMinimumFloat32x4(low, high));
    }

    while (N > 0) {

        tmp_max = std::max(tmp_max, *Input);
        tmp_min = std::min(tmp_min, *Input);

        Input += 1;
        N -= 1;
    }

    *Min = tmp_min;
    *Max = tmp_max;
}
