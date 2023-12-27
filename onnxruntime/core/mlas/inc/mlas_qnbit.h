/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    mlas_qnbit.h

Abstract:

    This module contains the public data structures and procedure prototypes
    for blocked n-bit quantized GEMM.

    N-bit block quantization is used to compress weight tensors of large
    language models.

--*/

#pragma once

#include "mlas.h"
#include "mlas_gemm_postprocessor.h"

/**
 * @brief Define compute types of block quantization
 */
typedef enum {
    CompUndef = 0, /*!< undef */
    CompFp32 = 1,  /*!< input fp32, accumulator fp32 */
    CompFp16 = 2,  /*!< input fp16, accumulator fp16 */
    CompBf16 = 3,  /*!< input bf16, accumulator fp32 */
    CompInt8 = 4   /*!< input int8, accumulator int32 */
} MLAS_SQNBIT_COMPUTE_TYPE;

using MLAS_SQNBITGEMM_COMPUTE_TYPE = MLAS_SQNBIT_COMPUTE_TYPE;  // TODO consolidate these

/**
 * @brief Data parameters for float/n-bit quantized int GEMM routine.
 */
struct MLAS_SQNBIT_GEMM_DATA_PARAMS {
    const float* A = nullptr;               ///< address of A (float32 matrix)
    size_t lda = 0;                         ///< leading dimension of A
    const void* QuantBData = nullptr;       ///< address of quantized B (quantized n-bit int values)
    const float* QuantBScale = nullptr;     ///< address of scale values of quantized B, one per block
    const void* QuantBZeroPoint = nullptr;  ///< optional address of zero point values of quantized B, one per block
    const float* Bias = nullptr;            ///< optional address of Bias, vector size N
    float* C = nullptr;                     ///< address of result matrix
    size_t ldc = 0;                         ///< leading dimension of C

    ///< optional post processing to apply to result matrix
    MLAS_GEMM_POSTPROCESSOR<float>* PostProcessor = nullptr;
};

/**
 * @brief Batched GEMM:  C = A * B + Bias
 *        A must be a float32 matrix
 *        B must be a quantized and packed n-bit int matrix
 *
 * @param[in]       M               row size of matrix A and C
 * @param[in]       N               column size of matrix B and C
 * @param[in]       K               column size of matrix A and row size of matrix B
 * @param[in]       BatchN          number of batches
 * @param[in]       BlkBitWidth     quantized value bit width (e.g., 4 means 4 bit ints)
 * @param[in]       BlkLen          number of quantized values per block
 * @param[in]       ComputeType     GEMM compute type (e.g., multiplying float or int8 values)
 * @param[inout]    DataParams      An array (size BatchN) of parameter blocks
 * @param[in]       Workspace       Address of intermediate workspace buffer.
                                    If MlasSQNBitGemmBatchWorkspaceSize() returns a non-zero value, this must be a
                                    buffer with at least that many bytes. Otherwise, it may be nullptr.
 * @param[in]       ThreadPool      optional thread pool to use
 */
void MLASCALL
MlasSQNBitGemmBatch(
    size_t M,
    size_t N,
    size_t K,
    size_t BatchN,
    size_t BlkBitWidth,
    size_t BlkLen,
    MLAS_SQNBITGEMM_COMPUTE_TYPE ComputeType,
    const MLAS_SQNBIT_GEMM_DATA_PARAMS* DataParams,
    void* Workspace,
    MLAS_THREADPOOL* ThreadPool = nullptr
);

/**
 * @brief Determines whether a float32/quantized n-bit int GEMM implementation is available on the current platform.
 *        Ensure that this returns true before calling MlasSQNBitGemmBatch().
 *
 * @param[in]   M               row size of matrix A and C
 * @param[in]   N               column size of matrix B and C
 * @param[in]   K               column size of matrix A and row size of matrix B
 * @param[in]   BlkBitWidth     quantized value bit width (e.g., 4 means 4 bit ints)
 * @param[in]   BlkLen          number of quantized values per block
 * @param[in]   ComputeType     GEMM compute type (e.g., multiplying float or int8 values)
 */
bool MLASCALL
MlasIsSQNBitGemmAvailable(
    size_t M,
    size_t N,
    size_t K,
    size_t BlkBitWidth,
    size_t BlkLen,
    MLAS_SQNBITGEMM_COMPUTE_TYPE ComputeType
);

/**
 * @brief Gets the size in bytes of the intermediate workspace buffer required by the float32/quantized n-bit int GEMM
 * implementation. If zero, no intermediate workspace is required.
 * @param[in]   M               row size of matrix A and C
 * @param[in]   N               column size of matrix B and C
 * @param[in]   K               column size of matrix A and row size of matrix B
 * @param[in]   BatchN          number of batches
 * @param[in]   BlkBitWidth     quantized value bit width (e.g., 4 means 4 bit ints)
 * @param[in]   BlkLen          number of quantized values per block
 * @param[in]   ComputeType     GEMM compute type (e.g., multiplying float or int8 values)
 */
size_t MLASCALL
MlasSQNBitGemmBatchWorkspaceSize(
    size_t M,
    size_t N,
    size_t K,
    size_t BatchN,
    size_t BlkBitWidth,
    size_t BlkLen,
    MLAS_SQNBITGEMM_COMPUTE_TYPE ComputeType
);

/**
 * @brief Data parameters for NBits GEMM routine
 *        C = A * B
 *        A, C must be a float32 matrix
 *        B must be a packed nbits blob
 *        All except C are [in] parameters
 */
struct MLAS_SQNBITS_GEMM_DATA_PACKED_PARAMS {
    const float* A = nullptr; /**< address of A (float32 matrix)*/
    const void* B = nullptr;  /**< address of B (packed nbits blob)*/
    float* C = nullptr;       /**< address of result matrix */
    size_t lda = 0;           /**< leading dimension of A */
    size_t ldc = 0;           /**< leading dimension of C*/
};

/**
 * @brief Compute the byte size of the parameter combination
 *
 * @param N      the number of columns of matrix B.
 * @param K      the number of rows of matrix B.
 * @param block_size    size of the block to quantize, elements from the same block share the same
 * scale and zero point
 * @param nbits  number of bits used for weight quantization
 * @param is_asym  flag for asymmetric quantization
 * @param comp_type  specify input data type and accumulator data type
 * @return size of the packing buffer, 0 if the operation is not yet supported.
 */
size_t MLASCALL
MlasNBitsGemmPackBSize(
    size_t N, size_t K, size_t block_size, int nbits, bool is_asym, MLAS_SQNBIT_COMPUTE_TYPE comp_type
);

/**
 * @brief Prepack tensor data from n-bit quantized data, scale and zero point buffers.
 *
 * @param PackedBuf     packed data buffer
 * @param QData         quantized data buffer
 * @param Scale         scale pointer
 * @param Zp            zero point pointer
 * @param N             the number of columns of matrix B.
 * @param K             the number of rows of matrix B.
 * @param ldb           leading dimension of B
 * @param block_size    size of the block to quantize, elements from the same block share the same
 * scale and zero point
 * @param nbits         number of bits used for weight quantization (default 4)
 * @param is_asym       flag for asymmetric quantization
 * @param comp_type     specify input data type and accumulator data type
 * @param last_call     flag to activate the epilogue process of packB. OpKernel::PrePack will query input tensor
 * one by one: QData, Scale, Zp (if is_asym is true). But kernel prefers to pack all tensors into one blob data where
 * they can share the common attributes like: block_size. Meanwhile, kernel has some pre-computations to speed up
 * inference which require that all blob data are ready. So, you need to set this flag to true when passing Scale
 * (is_asym is false) and Zp(is_asym is true).
 * @param thread_pool
 */
void MLASCALL
MlasNBitsGemmPackB(
    void* PackedBuf,
    const uint8_t* QData,
    const float* Scale,
    const uint8_t* Zp,
    size_t N,
    size_t K,
    size_t ldb,
    size_t block_size,
    int nbits,
    bool is_asym,
    bool last_call,
    MLAS_SQNBIT_COMPUTE_TYPE comp_type,
    MLAS_THREADPOOL* thread_pool
);

/**
 * @brief Unpack and dequantize to fp32
 *
 * @param FpData     unpacked float32 data
 * @param PackedBuf  quantized and packed data
 * @param N          the number of columns of matrix B.
 * @param K          the number of rows of matrix B.
 * @param ldb        leading dimension of B
 * @param thread_pool
 */
void MLASCALL
MlasNBitsGemmUnPackB(
    float* FpData, const void* PackedBuf, size_t N, size_t K, size_t ldb, MLAS_THREADPOOL* thread_pool
);

/**
 * @brief Get the workspace size required by computation.
 *
 * @param[in]  M       row size of matrix A and C
 * @param[in]  N       column size of matrix B and C
 * @param[in]  K       column size of matrix A and row size of matrix B
 * @param[in]  BatchN  number of batches
 * @param[inout]  DataParams  An array (size BatchN) of parameter blocks
 * @return     Workspace size in bytes
 */
size_t MLASCALL
MlasSQNBitsGemmBatchPackedBWorkspaceSize(
    const size_t M,
    const size_t N,
    const size_t K,
    const size_t BatchN,
    const MLAS_SQNBITS_GEMM_DATA_PACKED_PARAMS* DataParams
);

/**
 * @brief Batched GEMM:  C = A * B
 *        A, C must be a float32 matrix
 *        B must be a packed nbits blob
 *
 * @param[in]  M       row size of matrix A and C
 * @param[in]  N       column size of matrix B and C
 * @param[in]  K       column size of matrix A and row size of matrix B
 * @param[in]  BatchN  number of batches
 * @param[inout]  DataParams  An array (size BatchN) of parameter blocks
 * @param[in]  WorkSpace  temporary buffer
 * @param[in]  ThreadPool
 * @return
 */
void MLASCALL
MlasSQNBitsGemmBatchPackedB(
    const size_t M,
    const size_t N,
    const size_t K,
    const size_t BatchN,
    const MLAS_SQNBITS_GEMM_DATA_PACKED_PARAMS* DataParams,
    void* WorkSpace,
    MLAS_THREADPOOL* ThreadPool = nullptr
);
