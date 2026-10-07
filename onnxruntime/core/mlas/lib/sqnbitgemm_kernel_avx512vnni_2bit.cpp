/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    sqnbitgemm_kernel_avx512vnni_2bit.cpp

Abstract:

    This module implements the BlkLen-routing wrapper for the 2-bit (W2)
    CompInt8 AVX-512-VNNI kernels.

    It is deliberately kept in its own translation unit, separate from
    sqnbitgemm_kernel_avx512vnni.cpp: the W2 BlkLen32/64/128 kernel headers
    are large, fully force-inlined AVX-512-VNNI intrinsic bodies, and
    combining them with the W4/W8 kernels in a single TU makes MSVC (19.44,
    VS 2022 17.14) hit a fatal C1001 internal compiler error during Release
    code generation.

--*/

#include <cassert>
#include <cstddef>

#include "qnbitgemm.h"
#include "sqnbitgemm_kernel_avx512_2bit.h"
#include "sqnbitgemm_kernel_avx512_2bit_blklen64.h"
#include "sqnbitgemm_kernel_avx512_2bit_blklen128.h"
#include "sqnbitgemm_kernel_avx512_2bit_blklen32.h"

//
// BlkLen-routing wrapper for the W2 CompInt8 AVX-512-VNNI dispatch entry
// (sqnbitgemm_kernel_avx512_2bit_blklen64.h and friends). Production code
// reaches this via the MLAS dispatch table; tests call it directly via the
// namespace.
//
namespace onnxruntime::mlas::sq2bit_avx512 {
size_t MLASCALL
SQ2BitGemmKernel_BlkSum_CompInt8_Avx512Vnni_Dispatch(
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
    const float* QuantBBlkSum)
{
    if (BlkLen == 128) {
        return SQ2BitGemmKernel_BlkSum_CompInt8_BlkLen128_Avx512Vnni(
            QuantA, QuantAScale, QuantBData, QuantBScale,
            C, CountM, CountN, BlockCountK, Bias, ldc, ABlockSum, QuantBBlkSum);
    }
    if (BlkLen == 32) {
        return SQ2BitGemmKernel_BlkSum_CompInt8_BlkLen32_Avx512Vnni(
            QuantA, QuantAScale, QuantBData, QuantBScale,
            C, CountM, CountN, BlockCountK, Bias, ldc, ABlockSum, QuantBBlkSum);
    }
    return SQ2BitGemmKernel_BlkSum_CompInt8_Avx512Vnni(
        BlkLen, QuantA, QuantAScale, QuantBData, QuantBScale, QuantBZeroPoint,
        C, CountM, CountN, CountK, BlockCountK, Bias, ldc, ABlockSum, QuantBBlkSum);
}
}  // namespace onnxruntime::mlas::sq2bit_avx512
