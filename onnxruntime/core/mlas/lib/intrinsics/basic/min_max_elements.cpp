/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    min_max_elements.cpp

Abstract:

    This module implements the logic to find min and max elements with basic extension instructions.

--*/

#include "../mlasi.h"

void
MLASCALL
MlasMinMaxF32Kernel(
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

    *min = *max = Input[0];

    if (N >= 4) {

        MLAS_FLOAT32X4 MaximumVector0 = MlasBroadcastFloat32x4(*Input);
        MLAS_FLOAT32X4 MinimumVector0 = MlasBroadcastFloat32x4(*Input);

        if (N >= 16) {

            MLAS_FLOAT32X4 MaximumVector1 = MaximumVector0;
            MLAS_FLOAT32X4 MaximumVector2 = MaximumVector0;
            MLAS_FLOAT32X4 MaximumVector3 = MaximumVector0;

            MLAS_FLOAT32X4 MinimumVector1 = MinimumVector0;
            MLAS_FLOAT32X4 MinimumVector2 = MinimumVector0;
            MLAS_FLOAT32X4 MinimumVector3 = MinimumVector0;

            while (N >= 16) {

                MLAS_FLOAT32X4 InputVector0 = MlasLoadFloat32x4(Input);
                MLAS_FLOAT32X4 InputVector1 = MlasLoadFloat32x4(Input + 4);
                MLAS_FLOAT32X4 InputVector2 = MlasLoadFloat32x4(Input + 8);
                MLAS_FLOAT32X4 InputVector3 = MlasLoadFloat32x4(Input + 12);

                MaximumVector0 = MlasMaximumFloat32x4(MaximumVector0, InputVector0);
                MaximumVector1 = MlasMaximumFloat32x4(MaximumVector1, InputVector1);
                MaximumVector2 = MlasMaximumFloat32x4(MaximumVector2, InputVector2);
                MaximumVector3 = MlasMaximumFloat32x4(MaximumVector3, InputVector3);

                MinimumVector0 = MlasMinimumFloat32x4(MinimumVector0, InputVector0);
                MinimumVector1 = MlasMinimumFloat32x4(MinimumVector1, InputVector1);
                MinimumVector2 = MlasMinimumFloat32x4(MinimumVector2, InputVector2);
                MinimumVector3 = MlasMinimumFloat32x4(MinimumVector3, InputVector3);

                Input += 16;
                N -= 16;
            }

            MaximumVector0 = MlasMaximumFloat32x4(MaximumVector0, MaximumVector1);
            MaximumVector2 = MlasMaximumFloat32x4(MaximumVector2, MaximumVector3);
            MaximumVector0 = MlasMaximumFloat32x4(MaximumVector0, MaximumVector2);

            MinimumVector0 = MlasMinimumFloat32x4(MinimumVector0, MinimumVector1);
            MinimumVector2 = MlasMinimumFloat32x4(MinimumVector2, MinimumVector3);
            MinimumVector0 = MlasMinimumFloat32x4(MinimumVector0, MinimumVector2);
        }

        while (N >= 4) {

            MLAS_FLOAT32X4 InputVector0 = MlasLoadFloat32x4(Input);
            MaximumVector0 = MlasMaximumFloat32x4(MaximumVector0, InputVector0);
            MinimumVector0 = MlasMinimumFloat32x4(MinimumVector0, InputVector0);

            Input += 4;
            N -= 4;
        }

        *min = MlasReduceMinimumFloat32x4(MinimumVector0);
        *max = MlasReduceMaximumFloat32x4(MaximumVector0);
    }

    while (N > 0) {

        *max = std::max(*max, *Input);
        *min = std::min(*min, *Input);

        Input += 1;
        N -= 1;
    }
}