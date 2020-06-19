/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    min_max_elements.cpp

Abstract:

    This module implements the logic to find min and max elements with avx2 instructions.

--*/

#include "../mlasi.h"

void
MLASCALL
MlasMinMaxF32KernelAvx2(
    const float* Input,
    float* min,
    float* max,
    size_t N)
{
    if (N <= 0) {
        *min = 0.0f;
        *max = 0.0f;
        return;
    }

    *min = *max = *Input;

    if (N >= 8) {

        __m256 MaximumVector0 = _mm256_set1_ps(*Input);
        __m256 MinimumVector0 = _mm256_set1_ps(*Input);

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

        float min_buf[8], max_buf[8];
        _mm256_storeu_ps(min_buf, MinimumVector0);
        _mm256_storeu_ps(max_buf, MaximumVector0);
        for (int j = 0; j < 8; ++j) {
            *min = std::min(*min, min_buf[j]);
            *max = std::max(*max, max_buf[j]);
        }
    }

    while (N > 0) {

        *max = std::max(*max, *Input);
        *min = std::min(*min, *Input);

        Input += 1;
        N -= 1;
    }
}