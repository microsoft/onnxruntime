/*++

// TODO: finish filling this out

module includes kernel functions for generating LUT for T-MAC GEMM optimization strategy.
*/

#include "qlutgemm.h"

bool MLASCALL MlasIsTMACAvailable(
    size_t /*BlkBitWidth*/,
    size_t /*BlkLen*/,
    MLAS_QNBIT_GEMM_COMPUTE_TYPE /*ComputeType*/
)
{
    const auto* Dispatch = GetMlasPlatform().LUTGenKernel;
    return Dispatch != nullptr;
    // TODO: once you add the kernel for lut matmul itself, add switch case that handles the variant
    // and checks that the variant exists
}

bool MLASCALL MlasTmacInitializeTable(
    size_t BlkLen,
    const std::byte* QuantBData,
    const float* QuantBScale,
    const std::byte* QuantBZeroPoint,
    float* qlut,
    size_t CountN,
    size_t countK,
    size_t BlockStrideQuantB,
    const float* Bias
) {
    const auto* Dispatch = GetMlasPlatform().LUTGenKernel;

    return false;
}
