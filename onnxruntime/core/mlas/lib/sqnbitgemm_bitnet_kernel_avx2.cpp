/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    sqnbitgemm_kernel_avx2.cpp.h

Abstract:

    This module implements the float/quantized n-bit integer matrix
    multiplication kernels for x64 avx2.

--*/

#include <algorithm>
#include <cassert>
#include <utility>

#include "qnbitgemm.h"

size_t
Q2BitGemmPackQuantBDataSize(
    size_t /*N*/,
    size_t /*K*/,
    size_t /*BlkLen*/,
    MLAS_QNBIT_GEMM_COMPUTE_TYPE /*ComputeType*/
)
{
  return 0;
}

void SQ2BitGemmPackQuantBData(
  size_t /*N*/,
  size_t /*K*/,
  size_t /*BlkLen*/,
  MLAS_QNBIT_GEMM_COMPUTE_TYPE /* ComputeType*/,
  const std::byte* /*QuantBDataBegin*/,
  std::byte* /*PackedQuantBDataBegin*/,
  MLAS_THREADPOOL* /*ThreadPool*/
) 
{
}

size_t
Q2BitGemmPerGemmWorkspaceSize(
    size_t /*M*/,
    size_t /*N*/,
    size_t /*K*/,
    size_t /*BlkLen*/,
    MLAS_QNBIT_GEMM_COMPUTE_TYPE /*ComputeType*/
)
{
    return 0;
}

size_t
SQ2BitGemmKernel_CompInt8_avx2(
    size_t /*BlkLen*/,
    const std::byte* /*QuantA*/,
    const std::byte* /*QuantBData*/,
    const float* /*QuantBScale*/,
    const std::byte* /*QuantBZeroPoint*/,
    float* /*C*/,
    size_t /*CountM*/,
    size_t /*CountN*/,
    size_t /*CountK*/,
    size_t /*BlockCountK*/,
    size_t /*ldc*/,
    const float* /*Bias*/
)
{
    return 0;
}

void
QuantizeARow_CompInt8(
    size_t /*BlkLen*/,
    const float* /*A*/,
    size_t /*CountK*/,
    std::byte* /*QuantA*/
)
{
}
