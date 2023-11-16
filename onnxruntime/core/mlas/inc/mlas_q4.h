/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    mlas_q4.h

Abstract:

    This module contains the public data structures and procedure prototypes
    for blocked int4 quantization and dequantization.

    Int4 block quantization is used to compress weight tensors of large
    language models.

--*/

#pragma once

#include <math.h>

#include <algorithm>

#include "mlas.h"
#include "mlas_gemm_postprocessor.h"

/**
 * @brief Define types of block quantization
 */
typedef enum {
    BlkQ4Sym = 0,   /*!< int4 Symmetric Block Quantization, zero_point = 0 */
    BlkQ4Zp8 = 1,   /*!< int4 Block Quantization, zero_point is int8 type */
    BlkQ4Sym64 = 2, /*!< int4 Symmetric Block Quantization, 64 values per block*/
    BlkQ4Sym128 = 4 /*!< int4 Symmetric Block Quantization, 128 values per block*/
} MLAS_BLK_QUANT_TYPE;

/**
 * @brief Define compute types of block quantization
 */
typedef enum {
    CompUndef = 0, /*!< undef */
    CompFp32 = 1,  /*!< input fp32, accumulator fp32 */
    CompFp16 = 2,  /*!< input fp16, accumulator fp16 */
    CompBf16 = 3,  /*!< input bf16, accumulator fp32 */
    CompInt8 = 4   /*!< input int8, accumulator int32 */
} MLAS_COMPUTE_TYPE;

/**
 * @brief Computes the number of bytes required to pack and int4-quantize
 *        a weight matrix
 * @param QType  type of block quantization
 * @param N      the number of columns of matrix B.
 * @param K      the number of rows of matrix B.
 * @return size of the packing buffer, 0 if the operation is not yet supported.
 */
size_t MLASCALL
MlasQ4GemmPackBSize(MLAS_BLK_QUANT_TYPE QType, size_t N, size_t K);

/**
 * @brief Prepack and Quantize fp32 weight tensor to int4 blocks
 *
 * @param QType      type of block quantization
 * @param PackedBuf  destination buffer
 * @param FpData     the pointer to fp32 matrix
 * @param N          the number of columns of matrix B.
 * @param K          the number of rows of matrix B.
 * @param ldb        leading dimension of B
 */
void MLASCALL
MlasQ4GemmPackB(MLAS_BLK_QUANT_TYPE QType, void* PackedBuf, const float* FpData, size_t N, size_t K, size_t ldb);

/**
 * @brief Unpack and dequantize from int4 to fp32, reverse operation of
 *        MlasQ4GemmPackB
 * @param QType      type of block quantization
 * @param FpData     destination buffer, the fp32 matrix
 * @param PackedBuf  int4 quantized and packed data
 * @param N          the number of columns of matrix B.
 * @param K          the number of rows of matrix B.
 * @param ldb        leading dimension of B
 */
void MLASCALL
MlasQ4GemmUnPackB(MLAS_BLK_QUANT_TYPE QType, float* FpData, const void* PackedBuf, size_t N, size_t K, size_t ldb);

/**
 * @brief Data parameters for Q4 GEMM routine
 *        C = A * B + Bias
 *        A must be a float32 matrix
 *        B must be a quantized and packed int4 blob
 *        All except C are [in] parameters
 */
struct MLAS_Q4_GEMM_DATA_PARAMS {
    const float* A = nullptr;    /**< address of A (float32 matrix)*/
    const void* B = nullptr;     /**< address of B (quantized and packed int4 blob)*/
    const float* Bias = nullptr; /**< address of Bias, vector size N */
    float* C = nullptr;          /**< address of result matrix */
    size_t lda = 0;              /**< leading dimension of A */
    size_t ldc = 0;              /**< leading dimension of C*/
    const MLAS_GEMM_POSTPROCESSOR<float>* OutputProcessor = nullptr;
};

/**
 * @brief Batched GEMM:  C = A * B + Bias
 *        A must be a float32 matrix
 *        B must be a quantized and packed int4 blob
 *
 * @param[in]  QType   type of block quantization used in B
 * @param[in]  M       row size of matrix A and C
 * @param[in]  N       column size of matrix B and C
 * @param[in]  K       column size of matrix A and row size of matrix B
 * @param[in]  BatchN  number of batches
 * @param[inout]  DataParams  An array (size BatchN) of parameter blocks
 * @param[in]  ThreadPool
 * @return
 */
void MLASCALL
MlasQ4GemmBatch(MLAS_BLK_QUANT_TYPE QType, const size_t M, const size_t N, const size_t K, const size_t BatchN, const MLAS_Q4_GEMM_DATA_PARAMS* DataParams, MLAS_THREADPOOL* ThreadPool = nullptr);

/**
 * @brief Calculate the buffer size needed for int8 block quantize
 * @param[in]  QType   Type of block quantization used
 * @param[in]  M       Number of rows of the input matrix
 * @param[in]  K       Number of columns of the input matrix
 * @return    buffer size (in bytes) needed, 0 if not yet supported on current hardware
 */

size_t MLASCALL
MlasQ80BlkQuantSize(MLAS_BLK_QUANT_TYPE QType, size_t M, size_t K);

/**
 * @brief Given an input float 2-D matrix, perform blocked int8 quantize
 *
 * @param QType     Type of block quantization used
 * @param Qblob     Pointer to the output buffer
 * @param A         Pointer to the float matrix
 * @param M         Number of rows of the input matrix
 * @param K         Number of columns of the input matrix
 * @param lda       leading dimension of the input matrix
 * @param ThreadPool
 */
void MLASCALL
MlasQ80BlkQuant(MLAS_BLK_QUANT_TYPE QType, void* Qblob, const float* A, size_t M, size_t K, size_t lda, MLAS_THREADPOOL* ThreadPool);

/**
 * @brief Data parameters for Q8Q4 GEMM routine
 *        C = A * B + Bias
 *        A must be a block quantized int8 matrix
 *        B must be a block quantized and packed int4 blob
 *        All except C are [in] parameters
 */
struct MLAS_Q8Q4_GEMM_DATA_PARAMS {
    const void* A = nullptr;     /**< address of A (quantized int8 blob)*/
    const void* B = nullptr;     /**< address of B (quantized and packed int4 blob)*/
    const float* Bias = nullptr; /**< address of Bias, vector size N */
    float* C = nullptr;          /**< address of result matrix */
    size_t ldc = 0;              /**< leading dimension of C*/
    const MLAS_GEMM_POSTPROCESSOR<float>* OutputProcessor = nullptr;
};

/**
 * @brief Batched GEMM:  C = A * B + Bias
 *        A must be a quantized int8 blob
 *        B must be a quantized and packed int4 blob
 *
 * @param[in]  QType   type of block quantization used in B
 * @param[in]  M       row size of matrix A and C
 * @param[in]  N       column size of matrix B and C
 * @param[in]  K       column size of matrix A and row size of matrix B
 * @param[in]  BatchN  number of batches
 * @param[inout]  DataParams  An array (size BatchN) of parameter blocks
 * @param[in]  ThreadPool
 * @return
 */
void MLASCALL
MlasQ8Q4GemmBatch(MLAS_BLK_QUANT_TYPE QType, const size_t M, const size_t N, const size_t K, const size_t BatchN, const MLAS_Q8Q4_GEMM_DATA_PARAMS* DataParams, MLAS_THREADPOOL* ThreadPool);

////////////////////////////////////////////////////////////
// Blockwise quantization and dequantization where quantization
// parameters are packed into separate buffers.
//

/**
 * @brief For quantization type <T, block_size, columnwise>, and
 *        matrix shape [rows, columns], compute the shape of the
 *        quantization parameter matrix [meta_rows, meta_cols]
 */
template <typename T, int qbits>
void
MlasBlockwiseQuantMetaShape(
    int block_size, bool columnwise, int rows, int columns, int& meta_rows, int& meta_cols
);

/**
 * @brief For quantization type <T, block_size, columnwise>, and
 * matrix shape [rows, columns], compute the shape of the
 * quantized matrix [q_rows, q_cols]. The quantized matrix
 * is in column major layout, with bits packed on the column.
 *
 * @tparam T
 * @tparam qbits
 * @param block_size
 * @param columnwise
 * @param rows
 * @param columns
 * @param q_rows
 * @param q_cols
 */
template <typename T, int qbits>
void
MlasBlockwiseQuantizedShape(
    int block_size, bool columnwise, int rows, int columns, int& q_rows, int& q_cols
);

/**
 * @brief Compute the sizes of the quantized data and quantization parameter buffers.
 *
 * @param qbits                             The bit width of each quantized value.
 * @param block_size                        The number of quantized values in a block.
 * @param columnwise                        Whether a block contains values from a matrix column (true) or row (false).
 * @param rows                              Number of matrix rows.
 * @param columns                           Number of matrix columns.
 * @param[out] q_data_size_in_bytes         The size in bytes of the quantized data.
 * @param[out] q_scale_num_elements         The size in elements of the scale quantization parameters.
 * @param[out] q_zero_point_size_in_bytes   The size in bytes of the zero point quantization parameters. Optional.
 *
 * If the qbits or block_size values are unsupported the output sizes will be zero.
 */
void MLASCALL
MlasBlockwiseQuantizedBufferSizes(
    int qbits,
    int block_size,
    bool columnwise,
    int rows,
    int columns,
    size_t& q_data_size_in_bytes,
    size_t& q_scale_num_elements,
    size_t* q_zero_point_size_in_bytes
);

/**
 * @brief Blockwise 4 bits quantization, resulting elements and quantization
 *        parameters (scales, zero points) are packed into separate matrices
 *        all in column major layout for faster access during subsequent matrix
 *        multiplication.
 *
 * @tparam ElementT             type of the input matrix element, usually floating point
 * @tparam qbits                number of bits used for quantization, 4 for int4
 *
 * @param dst                   points to the quantized matrix, shape [rows, columns] column major
 * @param scales                points to the scales matrix, column major
 * @param zero_points           points to the zero_points matrix, column major
 * @param src                   points to the floating point matrix, to be quantized, row major
 * shape [rows, columns]
 * @param block_size            size of the block to quantize, elements from the same block share
 * the same scale and zero point
 * @param columnwise            true when elements in a block are from the same column, false when
 * elements in a block are from the same row
 * @param rows
 * @param columns
 * @param leading_dimension
 * @param thread_pool
 */
template <typename ElementT, int qbits>
void
MlasQuantizeBlockwise(uint8_t* dst, ElementT* scales, uint8_t* zero_points, const ElementT* src, int block_size, bool columnwise, int rows, int columns, int leading_dimension, MLAS_THREADPOOL* thread_pool);

/**
 * @brief Blockwise 4 bits dequantization, quantized elements and quantization
 *        parameters (scales, zero points) are from separate matrices packed
 *        in column major layout.  Output is a floating point matrix in column
 *        major layout for faster access during subsequent matrix multiplication.
 *
 * @tparam ElementT     type of the dequantized matrix element, usually floating point
 * @tparam qbits        number of bits used for quantization, 4 for int4
 *
 * @param dst           points to dequantized matrix shape [rows, columns] column major
 * @param src           points to quantized matrix, column major
 * @param scales        points to quantization scales, column major
 * @param zero_points   points to quantization zero points, column major
 * @param block_size    size of the block to quantize, elements from the same block share the same
 * scale and zero point
 * @param columnwise    true when elements in a block are from the same column, false when elements
 * in a block are from the same row
 * @param rows
 * @param columns
 * @param thread_pool
 */
template <typename ElementT, int qbits>
void
MlasDequantizeBlockwise(ElementT* dst, const uint8_t* src, const ElementT* scales, const uint8_t* zero_points, int block_size, bool columnwise, int rows, int columns, MLAS_THREADPOOL* thread_pool);

/**
 * @brief Check if the parameter combination is supported
 *
 * @param N      the number of columns of matrix B.
 * @param K      the number of rows of matrix B.
 * @param block_size    size of the block to quantize, elements from the same block share the same
 * scale and zero point
 * @param nbits  number of bits used for weight quantization (default 4)
 * @param is_asym  flag for asymmetric quantization
 * @param comp_type  specify input data type and accumulator data type
 * @return support flag, true if the combination is supported.
 */
bool MLASCALL
MlasNBitsGemmPackBSupport(
    size_t N, size_t K, size_t block_size, int nbits, bool is_asym, MLAS_COMPUTE_TYPE comp_type
);

/**
 * @brief Compute the byte size of the parameter combination
 *
 * @param N      the number of columns of matrix B.
 * @param K      the number of rows of matrix B.
 * @param block_size    size of the block to quantize, elements from the same block share the same
 * scale and zero point
 * @param nbits  number of bits used for weight quantization (default 4)
 * @param is_asym  flag for asymmetric quantization
 * @param comp_type  specify input data type and accumulator data type
 * @return size of the packing buffer, 0 if the operation is not yet supported.
 */
size_t MLASCALL
MlasNBitsGemmPackBSize(
    size_t N, size_t K, size_t block_size, int nbits, bool is_asym, MLAS_COMPUTE_TYPE comp_type
);

/**
 * @brief Prepack tensor data from MatMulNBits operator
 *
 * @param PackedBuf     pakced data buffer
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
 * @param last_call     flag to activate the epilogue process of packB
 * @param thread_pool
 */
void MLASCALL
MlasNBitsGemmPackB(void* PackedBuf, const uint8_t* QData, const float* Scale, const uint8_t* Zp, size_t N, size_t K, size_t ldb, size_t block_size, int nbits, bool is_asym, bool last_call, MLAS_COMPUTE_TYPE comp_type, MLAS_THREADPOOL* thread_pool);
/**
 * @brief Unpack and dequantize to fp32
 *
 * @param FpData     unpakced float32 data
 * @param PackedBuf  int4 quantized and packed data
 * @param N          the number of columns of matrix B.
 * @param K          the number of rows of matrix B.
 * @param ldb        leading dimension of B
 * @param thread_pool
 */
void MLASCALL
MlasNBitsGemmUnPackB(float* FpData, const void* PackedBuf, size_t N, size_t K, size_t ldb, MLAS_THREADPOOL* thread_pool);

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
MlasNBitsGemmBatch(const size_t M, const size_t N, const size_t K, const size_t BatchN, const MLAS_Q4_GEMM_DATA_PARAMS* DataParams, int8_t* WorkSpace, MLAS_THREADPOOL* ThreadPool = nullptr);