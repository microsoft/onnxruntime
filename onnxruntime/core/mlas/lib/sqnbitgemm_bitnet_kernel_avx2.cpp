/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    sqnbitgemm_kernel_avx2.cpp.h

Abstract:

    This module implements the float/quantized n-bit integer matrix
    multiplication kernels for x64 avx2.

--*/

#include "qnbitgemm.h"
#include "sqnbitgemm_q8_block.h"

size_t
Q2BitGemmPackQuantBDataSize(
    size_t N,
    size_t K,
    size_t BlkLen,
    MLAS_QNBIT_GEMM_COMPUTE_TYPE ComputeType
)
{
  // TODO: This code shall change according to T-Mac.
    MLAS_UNREFERENCED_PARAMETER(ComputeType);  // same size regardless of ComputeType

    constexpr size_t BlkBitWidth = 2;

    const size_t BlockCountK = MlasDivRoundup(K, BlkLen);
    const size_t PackedQuantBDataSize = N * BlockCountK * MlasQNBitBlkDataSizeInBytes(BlkBitWidth, BlkLen);
    return PackedQuantBDataSize;
}

void SQ2BitGemmPackQuantBData(
  size_t /*N*/,
  size_t /*K*/,
  size_t /*BlkLen*/,
  MLAS_QNBIT_GEMM_COMPUTE_TYPE /*ComputeType*/,
  const std::byte* /*QuantBDataBegin*/,
  std::byte* /*PackedQuantBDataBegin*/,
  MLAS_THREADPOOL* /*ThreadPool*/
) 
{
  // TODO: need implementation
}

size_t
Q2BitGemmPerGemmWorkspaceSize(
    size_t M,
    size_t N,
    size_t K,
    size_t BlkLen,
    MLAS_QNBIT_GEMM_COMPUTE_TYPE ComputeType
)
{
    MLAS_UNREFERENCED_PARAMETER(N);

    switch (ComputeType) {
        case SQNBIT_CompInt8: {
            // workspace buffer is used for block quantization of A to int8
            const size_t BlockCountK = MlasDivRoundup(K, BlkLen);
            // QuantData + Scale
            const size_t PerGemmWorkspaceSize = M * BlockCountK * Q8BlkSize(BlkLen);
            return PerGemmWorkspaceSize;
        }
        default: {
            return 0;
        }
    }
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
  // reference SQ4BitGemmKernel_CompInt8_avx2
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
  // shall be similar to QuantizeARow_CompInt8_avx2 without blksum related code.
}
