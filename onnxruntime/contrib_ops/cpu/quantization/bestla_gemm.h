/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    bestla_gemm.h

Abstract:

    Currently only support Q4 gemm.
--*/

#pragma once

#include <stdint.h>
#include <cstddef>

/**
 * @brief Define compute types of block quantization
 */
enum NS_SQNBIT_COMPUTE_TYPE {
  CompUndef = 0, /*!< undef */
  CompFp32 = 1,  /*!< input fp32, accumulator fp32 */
  CompFp16 = 2,  /*!< input fp16, accumulator fp16 */
  CompBf16 = 3,  /*!< input bf16, accumulator fp32 */
  CompInt8 = 4   /*!< input int8, accumulator int32 */
};

/**
 * @brief Data parameters for NBits GEMM routine
 *        C = A * B
 *        A, C must be a float32 matrix
 *        B must be a packed nbits blob
 *        All except C are [in] parameters
 */
struct NS_SQNBITS_GEMM_DATA_PACKED_PARAMS {
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
size_t NSNBitsGemmPackBSize(size_t N, size_t K, size_t block_size, int nbits, bool is_asym,
                            NS_SQNBIT_COMPUTE_TYPE comp_type);

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
void NSNBitsGemmPackB(void* PackedBuf, const uint8_t* QData, const float* Scale, const uint8_t* Zp, size_t N, size_t K,
                      size_t ldb, size_t block_size, int nbits, bool is_asym, bool last_call,
                      NS_SQNBIT_COMPUTE_TYPE comp_type, void* thread_pool);

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
void NSNBitsGemmUnPackB(float* FpData, const void* PackedBuf, size_t N, size_t K, size_t ldb, void* thread_pool);

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
size_t NSSQNBitsGemmBatchWorkspaceSize(const size_t M, const size_t N, const size_t K, const size_t BatchN,
                                       const NS_SQNBITS_GEMM_DATA_PACKED_PARAMS* DataParams);

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
void NSSQNBitsGemmBatchPackedB(const size_t M, const size_t N, const size_t K, const size_t BatchN,
                               const NS_SQNBITS_GEMM_DATA_PACKED_PARAMS* DataParams, void* WorkSpace,
                               void* ThreadPool = nullptr);
