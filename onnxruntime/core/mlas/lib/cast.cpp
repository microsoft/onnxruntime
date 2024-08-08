/*++

Copyright (c) Intel Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    cast.cpp

Abstract:

    This module implements Half (F16) to Single (F32) precision casting.

--*/
#include "mlasi.h"

union fp32_bits {
    uint32_t u;
    float f;
};

void
MLASCALL
MlasConvertHalfToFloatBuffer(
    const unsigned short* Source,
    float* Destination,
    size_t Count
)
{

    if (GetMlasPlatform().CastF16ToF32Kernel == nullptr) {
        // If there is no kernel use the reference implementation, adapted from mlas_float16.h.
        constexpr fp32_bits magic = {113 << 23};
        constexpr uint32_t shifted_exp = 0x7c00 << 13;  // exponent mask after shift

        for (size_t i = 0; i < Count; ++i) {
            fp32_bits o;
            o.u = (Source[i] & 0x7fff) << 13;  // exponent/mantissa bits
            uint32_t exp = shifted_exp & o.u;  // just the exponent
            o.u += (127 - 15) << 23;           // exponent adjust

            // handle exponent special cases
            if (exp == shifted_exp) {     // Inf/NaN?
                o.u += (128 - 16) << 23;  // extra exp adjust
            } else if (exp == 0) {        // Zero/Denormal?
                o.u += 1 << 23;           // extra exp adjust
                o.f -= magic.f;           // renormalize
            }

            o.u |= (Source[i] & 0x8000) << 16;  // sign bit
            Destination[i] = o.f;
        }
        
    } else {
        // If the kernel is available, use it to perform the conversion.
        GetMlasPlatform().CastF16ToF32Kernel(Source, Destination, Count);
    }
}
