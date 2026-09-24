/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    sqnbitgemm_kernel_avx2_2bit.h

Abstract:

    Declarations for the AVX2 / AVX2-VNNI W2 CompInt8 BlkLen-routing dispatch
    forwarders. The forwarders are defined in sqnbitgemm_kernel_avx2.cpp and the
    per-BlkLen kernels live in sqnbitgemm_kernel_avx2_2bit_blklen{32,64,128}.h.
    Declared here so the dispatch tables and the unit tests can reference them,
    mirroring the AVX-512 forwarder declarations in
    sqnbitgemm_kernel_avx512_2bit.h.

--*/

#pragma once

#include <cstddef>

#include "mlas.h"

namespace onnxruntime {
namespace mlas {
namespace sq2bit_avx2 {

size_t MLASCALL
SQ2BitGemmKernel_BlkSum_CompInt8_Avx2_Dispatch(
    size_t BlkLen,
    const std::byte* QuantA,
    const float* QuantAScale,
    const std::byte* QuantBData,
    const float* QuantBScale,
    const std::byte* QuantBZeroPoint,
    float* C,
    size_t CountM,
    size_t CountN,
    size_t CountK,
    size_t BlockCountK,
    const float* Bias,
    size_t ldc,
    const float* ABlockSum,
    const float* QuantBBlkSum
);

size_t MLASCALL
SQ2BitGemmKernel_BlkSum_CompInt8_Avx2Vnni_Dispatch(
    size_t BlkLen,
    const std::byte* QuantA,
    const float* QuantAScale,
    const std::byte* QuantBData,
    const float* QuantBScale,
    const std::byte* QuantBZeroPoint,
    float* C,
    size_t CountM,
    size_t CountN,
    size_t CountK,
    size_t BlockCountK,
    const float* Bias,
    size_t ldc,
    const float* ABlockSum,
    const float* QuantBBlkSum
);

}  // namespace sq2bit_avx2
}  // namespace mlas
}  // namespace onnxruntime
