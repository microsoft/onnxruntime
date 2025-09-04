/*++

// TODO: finish filling this out

module includes kernel functions for generating LUT for T-MAC GEMM optimization strategy.
*/

#include "qlutgemm.h"

bool MLASCALL MlasIsTMACAvailable(
    size_t /*BlkBitWidth*/,
    size_t BlkLen
)
{
    const auto* Dispatch = GetMlasPlatform().LUTGenKernel;
    return Dispatch != nullptr && BlkLen == 4; // only support group sizes of 4 for now
}

bool MLASCALL MlasTmacInitializeTable(
    size_t BlkLen,
    const void* QuantBData,     // B in MLFloat16 (per your layout)
    const float* QuantBScale,        // scale(s) in float
    void* qlut
) {
    const auto* Dispatch = GetMlasPlatform().LUTGenKernel;
    if (!Dispatch || !Dispatch->GenerateLUT) return false;

    // Cast target LUT buffer to int8, and prepare half-precision inputs
    auto* lut_i8 = reinterpret_cast<int8_t*>(qlut);
    auto* b_half = const_cast<onnxruntime::MLFloat16*>(
        reinterpret_cast<const onnxruntime::MLFloat16*>(QuantBData));

    // Convert the first float scale to half (adjust if you have more)
    onnxruntime::MLFloat16 s16(QuantBScale[0]);
    onnxruntime::MLFloat16 b16(0.0f);  // output bias goes here // TODO: pass the biases here

    // Call the dispatch
    Dispatch->GenerateLUT(static_cast<int32_t>(BlkLen), lut_i8, b_half, &s16, &b16);

    // If you need the bias value elsewhere, read it from b16
    // float bias_f = static_cast<float>(b16);

    return true;
}
