/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    sqnbitgemm.h

Abstract:

    This module includes kernel function prototypes and helper functions for
    implementing SQNBitGemm.

    SQNBitGemm is a matrix/matrix multiplication, A*B, where A is a float
    matrix and B is a n-bit quantized integer matrix. B is block quantized,
    meaning values of B are divided into blocks and each block has its own
    scale and optional zero point.

--*/

#pragma once

#include "mlas_qnbit.h"
#include "mlasi.h"

constexpr MLAS_FORCEINLINE size_t
MlasQNBitQuantBBlkSumAlignment()
{
    // 16 floats. this alignment is required by GemmFloatKernel
    return 16 * sizeof(float);
}

constexpr MLAS_FORCEINLINE size_t
MlasQNBitBlkDataSizeInBytes(size_t BlkBitWidth, size_t BlkLen)
{
    return BlkLen * BlkBitWidth / 8;
}

MLAS_FORCEINLINE void*
MlasAlignAddress(void* addr, const size_t alignment)
{
    const uintptr_t QuantBBlkSumAddr = reinterpret_cast<uintptr_t>(addr);
    addr = (void*)((QuantBBlkSumAddr + alignment - 1) & (~(alignment - 1)));
    return addr;
}

struct PackedQuantBDataStruct {
    PackedQuantBDataStruct(void* PackedQuantBWorkspace, size_t N, size_t BlockCountK, size_t BlkLen)
        : QuantBWorkspace_(PackedQuantBWorkspace), N_(N), BlockCountK_(BlockCountK), BlkLen_(BlkLen)
    {
      // TODO: duplicate code from SQ4BitGemmPackQuantBDataSize
        constexpr size_t BlkBitWidth = 4;
        const size_t PackedQuantBDataSize = N * BlockCountK * MlasQNBitBlkDataSizeInBytes(BlkBitWidth, BlkLen);
        size_t BlkSumSize = MlasDivRoundup(N, 16) * BlockCountK * 16 * sizeof(float);

        // _mm256_load_si256 requires alignment on a 32-byte boundary
        PackedQuantBData = (std::byte*)MlasAlignAddress(PackedQuantBWorkspace, 32);
        QuantBBlkSum = (float*)(PackedQuantBData + PackedQuantBDataSize);
        QuantBBlkSum = (float*)MlasAlignAddress(QuantBBlkSum, MlasQNBitQuantBBlkSumAlignment());
        PackedQuantBScale = (float*)((std::byte*)QuantBBlkSum + BlkSumSize);
    }
    std::byte* PackedQuantBData;
    float* PackedQuantBScale;
    float* QuantBBlkSum;

    void* QuantBWorkspace_;
    size_t N_, BlockCountK_, BlkLen_;
};

template <size_t BlkBitWidth>
constexpr MLAS_FORCEINLINE size_t
MlasQNBitZeroPointsForBlksSizeInBytes(size_t BlkCount)
{
    if constexpr (BlkBitWidth <= 4) {
        return MlasDivRoundup(BlkCount, 2);  // 2 blocks per byte
    } else {
        return BlkCount;
    }
}

//
// Kernel dispatch structure.
//

struct MLAS_SQNBIT_GEMM_DISPATCH {
    //
    // Quantized B data packing function prototypes.
    //

    /** Gets size of packed quantized B data containing 4-bit integers. See MlasSQNBitGemmPackQuantBDataSize(). */
    typedef size_t(SQ4BitGemmPackQuantBDataSize_Fn)(
        size_t N,
        size_t K,
        size_t BlkLen,
        MLAS_SQNBIT_GEMM_COMPUTE_TYPE ComputeType
    );

    SQ4BitGemmPackQuantBDataSize_Fn* SQ4BitGemmPackQuantBDataSize = nullptr;

    /** Packs quantized B data containing 4-bit integers. See MlasSQNBitGemmPackQuantBData(). */
    typedef void(SQ4BitGemmPackQuantBData_Fn)(
        size_t N,
        size_t K,
        size_t BlkLen,
        MLAS_SQNBIT_GEMM_COMPUTE_TYPE ComputeType,
        const std::byte* QuantBDataBegin,
        std::byte* PackedQuantBDataBegin,
        MLAS_THREADPOOL* ThreadPool
    );

    SQ4BitGemmPackQuantBData_Fn* SQ4BitGemmPackQuantBData = nullptr;

    typedef void(SQ4BitGemmPackQuantBDataAndSumBlk_Fn)(
        size_t N,
        size_t K,
        size_t BlkLen,
        MLAS_SQNBIT_GEMM_COMPUTE_TYPE ComputeType,
        const std::byte* QuantBDataBegin,
        const float* QuantBScaleBegin,
        bool has_zp_input,
        const std::byte* QuantBZPBegin,
        PackedQuantBDataStruct& packed_quant_b,
        MLAS_THREADPOOL* ThreadPool
    );

    SQ4BitGemmPackQuantBDataAndSumBlk_Fn* SQ4BitGemmPackQuantBDataAndBlkSum = nullptr;

    //
    // Workspace size calculation function prototypes.
    //

    /**
     * @brief Gets the required size in bytes of the per-GEMM intermediate workspace.
     *        Returns a size of zero if no intermediate workspace is needed.
     *
     * @param[in]   M               row size of matrix A and C
     * @param[in]   N               column size of matrix B and C
     * @param[in]   K               column size of matrix A and row size of matrix B
     * @param[in]   BlkLen          number of quantized values per block
     * @param[in]   ComputeType     GEMM compute type (e.g., multiplying float or int8 values)
     */
    typedef size_t(SQ4BitGemmPerGemmWorkspaceSize_Fn)(
        size_t M,
        size_t N,
        size_t K,
        size_t BlkLen,
        MLAS_SQNBIT_GEMM_COMPUTE_TYPE ComputeType
    );

    SQ4BitGemmPerGemmWorkspaceSize_Fn* SQ4BitGemmPerGemmWorkspaceSize = nullptr;

    /**
     * @brief Gets the required byte alignment of the per-GEMM intermediate workspace.
     *
     * @param[in]   BlkLen          number of quantized values per block
     * @param[in]   ComputeType     GEMM compute type (e.g., multiplying float or int8 values)
     */
    typedef size_t(SQ4BitGemmPerGemmWorkspaceAlignment_Fn)(
        size_t BlkLen,
        MLAS_SQNBIT_GEMM_COMPUTE_TYPE ComputeType
    );

    SQ4BitGemmPerGemmWorkspaceAlignment_Fn* SQ4BitGemmPerGemmWorkspaceAlignment = nullptr;

    //
    // CompFp32 kernel function prototypes.
    //

    /**
     * @brief Multiply float matrix A with quantized 4-bit integer matrix B.
     *        B is block quantized and column major.
     *        This kernel handles the special case where M, the number of rows of A and C, is 1.
     *
     * @param       BlkLen              Number of values in a block.
     * @param       A                   Supplies the A matrix.
     * @param       QuantBData          Supplies the quantized B matrix block data.
     * @param       QuantBScale         Supplies the quantized B matrix block scale values.
     * @param       QuantBZeroPoint     Supplies the quantized B matrix block zero point values. Optional.
     * @param[out]  C                   Supplies the output C matrix.
     * @param       CountN              Number of columns of B and C.
     * @param       CountK              Number of columns of A and rows of B.
     * @param       BlockStrideQuantB   Number of blocks between adjacent columns of the quantized B matrix.
     * @param       Bias                Bias vector of length N.
     */
    template<typename AType>
    using SQ4BitGemmM1Kernel_CompFp32_Fn = void(
        size_t BlkLen,
        const AType* A,
        const std::byte* QuantBData,
        const float* QuantBScale,
        const std::byte* QuantBZeroPoint,
        float* C,
        size_t CountN,
        size_t CountK,
        size_t BlockStrideQuantB,
        const float* Bias
    );

    SQ4BitGemmM1Kernel_CompFp32_Fn<float>* SQ4BitGemmM1Kernel_CompFp32_ATypeFp32 = nullptr;
    SQ4BitGemmM1Kernel_CompFp32_Fn<MLAS_FP16>* SQ4BitGemmM1Kernel_CompFp32_ATypeFp16 = nullptr;

    template <typename AType>
        SQ4BitGemmM1Kernel_CompFp32_Fn<AType>*
    GetSQ4BitGemmM1Kernel_CompFp32_Fn()
    {
        if constexpr (std::is_same<AType, MLAS_FP16>::value) {
            return SQ4BitGemmM1Kernel_CompFp32_ATypeFp16;
        } else {
            return SQ4BitGemmM1Kernel_CompFp32_ATypeFp32;
        }
    }

    template <typename AType>
    void CallSQ4BitGemmM1Kernel_CompFp32_Fn(
      size_t BlkLen,
      const AType* A,
      const std::byte* QuantBData,
      const float* QuantBScale,
      const std::byte* QuantBZeroPoint,
      float* C,
      size_t CountN,
      size_t CountK,
      size_t BlockStrideQuantB,
      const float* Bias
    ) const {
        if constexpr (std::is_same<AType, MLAS_FP16>::value) {
            SQ4BitGemmM1Kernel_CompFp32_ATypeFp16(
                BlkLen,
                A,
                QuantBData,
                QuantBScale,
                QuantBZeroPoint,
                C,
                CountN,
                CountK,
                BlockStrideQuantB,
                Bias
            );
        } else {
            SQ4BitGemmM1Kernel_CompFp32_ATypeFp32(
                BlkLen,
                A,
                QuantBData,
                QuantBScale,
                QuantBZeroPoint,
                C,
                CountN,
                CountK,
                BlockStrideQuantB,
                Bias
            );
        }
    }

    /**
     * @brief Dequantize B into the format expected by the Sgemm kernel.
     *        B is a quantized 4-bit integer matrix that is block quantized and column major.
     *        This is equivalent to dequantizing B and then running MlasSgemmCopyPackB.
     *
     * @param       BlkLen              Number of values in a block.
     * @param[out]  FpData              Supplies the output buffer for the dequantized B float data.
     *                                  It should have enough space for
     *                                      (CountN + 16 - 1) / 16 * 16 * (CountK + BlkLen - 1) / BlkLen * BlkLen
     *                                  elements. Only the first (CountN + 16 - 1) / 16 * 16 * CountK elements are
     *                                  useful, but the kernel implementation can be simplified with the extra space.
     * @param       QuantBData          Supplies the quantized B matrix block data.
     * @param       QuantBScale         Supplies the quantized B matrix block scale values.
     * @param       QuantBZeroPoint     Supplies the quantized B matrix block zero point values. Optional.
     * @param       CountN              Number of columns of B.
     * @param       CountK              Number of rows of B.
     * @param       BlockStrideQuantB   Number of blocks between adjacent columns of the quantized B matrix.
     */
    typedef void(Q4BitBlkDequantBForSgemm_CompFp32_Fn)(
        size_t BlkLen,
        float* FpData,
        const std::byte* QuantBData,
        const float* QuantBScale,
        const std::byte* QuantBZeroPoint,
        size_t CountN,
        size_t CountK,
        size_t BlockStrideQuantB
    );

    Q4BitBlkDequantBForSgemm_CompFp32_Fn* Q4BitBlkDequantBForSgemm_CompFp32 = nullptr;

    //
    // CompInt8 kernel function prototypes.
    //

    /**
     * @brief Multiply quantized 8-bit integer matrix A with quantized 4-bit integer matrix B.
     *        A and B are block quantized and B is column major.
     *
     * @param       BlkLen              Number of values in a block.
     * @param       QuantA              Supplies the quantized A matrix.
                                        Binary data containing block quantized int8 data and scale values.
     * @param       QuantBData          Supplies the quantized B matrix block data.
     * @param       QuantBScale         Supplies the quantized B matrix block scale values.
     * @param       QuantBZeroPoint     Supplies the quantized B matrix block zero point values. Optional.
     * @param[out]  C                   Supplies the output C matrix.
     * @param       CountN              Number of columns of B and C.
     * @param       CountK              Number of columns of A and rows of B.
     * @param       BlockCountK         Number of blocks between adjacent columns of the quantized B matrix.
     * @param       Bias                Bias vector of length N.
     * @param       ldc                 Number of elements between adjacent rows of C..
     * @param       ABlockSum           Supplies the blksum of A.
     * @param       QuantBBlkSum        Supplies the blksum of B.
     */
    typedef size_t(SQ4BitGemmKernel_BlkSum_CompInt8_Fn)(
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

    SQ4BitGemmKernel_BlkSum_CompInt8_Fn* SQ4BitGemmKernel_BlkSum_CompInt8 = nullptr;

    /**
     * @brief Multiply quantized 8-bit integer matrix A with quantized 4-bit integer matrix B.
     *        A and B are block quantized and B is column major.
     *
     * @param       BlkLen              Number of values in a block.
     * @param       QuantA              Supplies the quantized A matrix.
                                        Binary data containing block quantized int8 data and scale values.
     * @param       QuantBData          Supplies the quantized B matrix block data.
     * @param       QuantBScale         Supplies the quantized B matrix block scale values.
     * @param       QuantBZeroPoint     Supplies the quantized B matrix block zero point values. Optional.
     * @param[out]  C                   Supplies the output C matrix.
     * @param       CountM              Number of rows of A and C to process, an upper bound.
     * @param       CountN              Number of columns of B and C to process.
     * @param       CountK              Number of columns of A and rows of B.
     * @param       BlockCountK         Number of blocks in one row of A and one column of B.
     * @param       ldc                 Number of elements between adjacent rows of C.
     * @param       Bias                Bias vector of length N.
     *
     * @return                          The number of rows of A and C that were processed, at most CountM.
     */
    typedef size_t(SQ4BitGemmKernel_CompInt8_Fn)(
        size_t BlkLen,
        const std::byte* QuantA,
        const std::byte* QuantBData,
        const float* QuantBScale,
        const std::byte* QuantBZeroPoint,
        float* C,
        size_t CountM,
        size_t CountN,
        size_t CountK,
        size_t BlockCountK,
        size_t ldc,
        const float* Bias
    );

    SQ4BitGemmKernel_CompInt8_Fn* SQ4BitGemmKernel_CompInt8 = nullptr;

    /**
     * @brief Block quantize values from one row of matrix A from floats to quantized 8-bit integers.
     *
     * @param       BlkLen  Number of values in a block.
     * @param       A       Supplies the A matrix.
     * @param       CountK  Number of columns of A.
     * @param[out]  QuantA  Supplies the output quantized A matrix.
     *                      Binary data containing block quantized int8 data and scale values.
     */
    typedef void(QuantizeARow_CompInt8_Fn)(
        size_t BlkLen,
        const float* A,
        size_t CountK,
        std::byte* QuantA
    );

    QuantizeARow_CompInt8_Fn* QuantizeARow_CompInt8 = nullptr;

    template<typename T>
    using QuantizeARowComputeBlkSum_CompInt8_Fn = void(
        size_t BlkLen,
        const T* A,
        size_t CountK,
        std::byte* QuantA,
        float* QuantAScale,
        float* AScaledGroupSum  // scale_k * Sum_blklen(a_i)
    );

    QuantizeARowComputeBlkSum_CompInt8_Fn<float>* QuantizeARowComputeBlkSum_CompInt8_ATypeFp32 = nullptr;
    QuantizeARowComputeBlkSum_CompInt8_Fn<MLAS_FP16>* QuantizeARowComputeBlkSum_CompInt8_ATypeFp16 = nullptr;
    template <typename AType>
    QuantizeARowComputeBlkSum_CompInt8_Fn<AType>*
    GetQuantizeARowComputeBlkSum_CompInt8_Fn() const
    {
        if constexpr (std::is_same<AType, MLAS_FP16>::value) {
            return QuantizeARowComputeBlkSum_CompInt8_ATypeFp16;
        } else {
            return QuantizeARowComputeBlkSum_CompInt8_ATypeFp32;
        }
    }
};
