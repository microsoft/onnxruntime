/*++

// TODO: finish filling this out

module includes kernel functions for generating LUT for T-MAC GEMM optimization strategy.
*/

#include "qlutgemm.h"

bool MLASCALL MlasIsTMACAvailable(
    size_t /*BlkBitWidth*/,
    size_t /*BlkLen*/
)
{
    const auto* Dispatch = GetMlasPlatform().LUTGenKernel;
    return Dispatch != nullptr;
    // return Dispatch != nullptr && BlkLen == 4; // only support group sizes of 4 for now
}

// TODO: also pass in a biases reference
bool MLASCALL MlasTmacInitializeTable(
    size_t BlkLen,
    void* QuantBData,     // B in MLFloat16 (per your layout)
    float* QuantBScale,        // scale(s) in float
    int K,
    void* qlut
) {
    // base on lut_ctor_int8_g4
    const auto* Dispatch = GetMlasPlatform().LUTGenKernel;
    if (!Dispatch || !Dispatch->GenerateLUT) return false;

    // Cast target LUT buffer to int8, and prepare half-precision inputs
    auto* lut_i8 = reinterpret_cast<int8_t*>(qlut);
    auto* b_float = reinterpret_cast<float*>(QuantBData);

    const int num_groups = static_cast<int>(K / BlkLen);

    float* biases = new float[num_groups]();

    // Call the dispatch
    Dispatch->GenerateLUT(static_cast<int32_t>(BlkLen), lut_i8, b_float, QuantBScale, biases, K);

    // If you need the bias value elsewhere, read it from b16
    // float bias_f = static_cast<float>(b16);

    return true;
}
