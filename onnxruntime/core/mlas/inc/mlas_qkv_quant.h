/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    mlas_qkv_quant.h

Abstract:

    Public API for symmetric INT4 / INT8 quantized KV-cache GEMMs used by the
    CPU GroupQueryAttention contrib operator.

    The "B" matrix in these GEMMs is the K or V cache, which is updated every
    decoding step. Unlike the weight-quantization kernels in mlas_qnbit.h /
    mlas_q4.h, B is therefore NOT prepacked: the kernels operate directly on a
    runtime-quantized buffer.

    Layout, packing, and scaling conventions follow the CUDA implementation in
    onnxruntime/contrib_ops/cuda/bert/group_query_attention_qdq.cuh:

      INT8 (signed):
        - Range  : [-128, 127]
        - Storage: int8_t, 1 byte per element
        - Formula: q = clamp(round(x / scale), -128, 127)

      INT4 (signed, biased nibble packing):
        - Range  : [-8, 7]
        - Storage: uint8_t, 2 elements per byte
            packed_byte = ((q0 + 8) & 0x0F) | (((q1 + 8) & 0x0F) << 4)
          q0 (even index)  --> low nibble  (bits 0-3)
          q1 (odd  index)  --> high nibble (bits 4-7)
        - Dequant: x = (nibble - 8) * scale
        - For odd column count, the last element pairs with q1 = 0.

    Scale granularity:
        PER_TENSOR : a single float scalar applies to the whole B slice.
        PER_CHANNEL: a float vector of length `cols` (i.e. head_size, the
                     innermost dim of B) applies along that axis.

    Symmetric only; no zero points.

--*/

#pragma once

#include "mlas.h"

#include <cstddef>

/**
 * @brief Quantization mode for the KV-cache GEMM kernels.
 *
 * - S8_PerTensor : INT8 symmetric, single scalar scale.
 * - S8_PerChannel: INT8 symmetric, one scale per innermost column of B.
 * - S4_PerTensor : INT4 symmetric (packed two-per-byte), single scalar scale.
 * - S4_PerChannel: INT4 symmetric (packed two-per-byte), one scale per
 *                  innermost column of B.
 */
enum class MLAS_KV_QUANT_TYPE {
    S8_PerTensor = 0,
    S8_PerChannel = 1,
    S4_PerTensor = 2,
    S4_PerChannel = 3,
};

/**
 * @brief Returns true if MlasKVQuantize / MlasQKGemm / MlasSVGemm /
 *        MlasKVDequantize are available for the requested mode on the current
 *        platform. Always true for the portable reference implementation.
 */
bool
MLASCALL
MlasIsKVQuantGemmSupported(
    MLAS_KV_QUANT_TYPE QuantType
    );

/**
 * @brief Returns the number of storage bytes per row of a quantized B with
 *        `cols` columns under the given quantization mode.
 *
 *        INT8: cols bytes per row.
 *        INT4: (cols + 1) / 2 bytes per row (low nibble = even element).
 */
size_t
MLASCALL
MlasKVQuantPackedRowBytes(
    MLAS_KV_QUANT_TYPE QuantType,
    size_t Cols
    );

/**
 * @brief Symmetrically quantize a row-major FP32 matrix into the packed
 *        INT8 / INT4 layout used by the KV-cache GEMMs.
 *
 * @param Src        Source FP32 buffer of shape [Rows, Cols], stride lda.
 * @param Dst        Destination buffer. Must be at least
 *                   Rows * MlasKVQuantPackedRowBytes(QuantType, Cols) bytes.
 *                   Each output row is contiguous (no row padding).
 * @param Rows       Number of rows.
 * @param Cols       Number of columns (un-packed element count, i.e. head_size).
 * @param lda        Source leading dimension (stride between rows of Src) in
 *                   elements. Typically equal to Cols.
 * @param QuantType  Quantization mode.
 * @param Scales     For PER_TENSOR : pointer to a single float.
 *                   For PER_CHANNEL: pointer to a float vector of length Cols.
 *                   Must not be null. A zero scale is treated as 1.0 to avoid
 *                   division by zero; callers should ensure scales > 0.
 * @param ThreadPool Optional thread pool. May be null.
 */
void
MLASCALL
MlasKVQuantize(
    const float* Src,
    void* Dst,
    size_t Rows,
    size_t Cols,
    size_t lda,
    MLAS_KV_QUANT_TYPE QuantType,
    const float* Scales,
    MLAS_THREADPOOL* ThreadPool
    );

/**
 * @brief Dequantize a packed INT8 / INT4 buffer back to row-major FP32.
 *        Inverse of MlasKVQuantize. Primarily useful for tests and for
 *        fallback / reference paths.
 *
 * @param Src        Packed quantized buffer of shape [Rows, packed_row_bytes].
 * @param Dst        Destination FP32 buffer of shape [Rows, Cols], stride ldb.
 * @param Rows       Number of rows.
 * @param Cols       Number of unpacked columns.
 * @param ldb        Destination leading dimension (elements). Typically Cols.
 * @param QuantType  Quantization mode.
 * @param Scales     Same shape rules as MlasKVQuantize.
 * @param ThreadPool Optional thread pool.
 */
void
MLASCALL
MlasKVDequantize(
    const void* Src,
    float* Dst,
    size_t Rows,
    size_t Cols,
    size_t ldb,
    MLAS_KV_QUANT_TYPE QuantType,
    const float* Scales,
    MLAS_THREADPOOL* ThreadPool
    );

/**
 * @brief QK^T GEMM with a quantized K cache.
 *
 *   C[M, N] = Alpha * A[M, K] * B^T[K, N]
 *
 * where:
 *   - A is FP32 row-major, shape [M, K], stride lda (>= K).
 *   - B is the quantized K cache, logically shape [N, K] (BNSH per-head slice
 *     with N = total_sequence_length, K = head_size), row-major in packed
 *     form. Storage is contiguous over rows; each row occupies
 *     MlasKVQuantPackedRowBytes(QuantType, K) bytes.
 *   - C is FP32 row-major, shape [M, N], stride ldc (>= N). The kernel
 *     overwrites C (no accumulate).
 *   - Scales follow the conventions described for MlasKVQuantize, applied
 *     along the K (head_size) axis when PER_CHANNEL.
 *
 * @param M          Query token count for this (batch, head) tile.
 * @param N          Total sequence length of the K cache.
 * @param K          head_size.
 * @param Alpha      Scalar multiplier (e.g. 1/sqrt(head_size)).
 * @param A          FP32 query.
 * @param lda        Leading dimension of A in elements.
 * @param B          Packed quantized K cache.
 * @param QuantType  Quantization mode.
 * @param Scales     Scale buffer (single scalar or length-K vector).
 * @param C          Output buffer (FP32).
 * @param ldc        Leading dimension of C in elements.
 * @param ThreadPool Optional thread pool.
 */
void
MLASCALL
MlasQKGemm(
    size_t M,
    size_t N,
    size_t K,
    float Alpha,
    const float* A,
    size_t lda,
    const void* B,
    MLAS_KV_QUANT_TYPE QuantType,
    const float* Scales,
    float* C,
    size_t ldc,
    MLAS_THREADPOOL* ThreadPool
    );

/**
 * @brief Softmax-times-V GEMM with a quantized V cache.
 *
 *   C[M, N] = Beta * C[M, N] + A[M, K] * B[K, N]
 *
 * where:
 *   - A is FP32 row-major, shape [M, K] (attention probabilities), stride lda.
 *   - B is the quantized V cache, logically shape [K, N] (BNSH per-head slice
 *     with K = total_sequence_length, N = head_size), packed row-major over
 *     rows. Each row occupies
 *     MlasKVQuantPackedRowBytes(QuantType, N) bytes.
 *   - C is FP32 row-major, shape [M, N], stride ldc (>= N).
 *     When Beta == 0, C is overwritten. When Beta != 0, C is accumulated.
 *   - PER_CHANNEL scales are length N and apply along the N (head_size) axis.
 *
 * @param M          Query token count.
 * @param N          head_size.
 * @param K          Total sequence length of the V cache.
 * @param A          FP32 attention probabilities.
 * @param lda        Leading dimension of A in elements.
 * @param B          Packed quantized V cache.
 * @param QuantType  Quantization mode.
 * @param Scales     Scale buffer (single scalar or length-N vector).
 * @param C          Output buffer (FP32).
 * @param ldc        Leading dimension of C in elements.
 * @param Beta       Scalar multiplier for existing C values. 0 = overwrite.
 * @param ThreadPool Optional thread pool.
 */
void
MLASCALL
MlasSVGemm(
    size_t M,
    size_t N,
    size_t K,
    const float* A,
    size_t lda,
    const void* B,
    MLAS_KV_QUANT_TYPE QuantType,
    const float* Scales,
    float* C,
    size_t ldc,
    float Beta,
    MLAS_THREADPOOL* ThreadPool
    );

/**
 * @brief Arguments for the Flash Attention kernel with quantized KV cache.
 *
 * This kernel implements the online-softmax tiled Flash Attention algorithm
 * operating directly on INT8/INT4 quantized K and V cache buffers.
 * It avoids materializing the full [S, T] attention probability matrix.
 */
struct MlasFlashAttentionQuantizedKVArgs {
    int batch_size = 0;
    int num_heads = 0;           // Q heads
    int kv_num_heads = 0;        // KV heads (for GQA sharing)
    int sequence_length = 0;     // Q sequence length (new tokens)
    int total_seqlen = 0;        // Total KV sequence length (past + new)
    int head_size = 0;
    int past_seqlen = 0;         // For computing causal positions
    int local_window_size = -1;  // -1 = disabled
    int seqlen_present_kv = 0;   // Buffer dimension for present KV (may be > total_seqlen)
    int q_block_size = 0;        // Br (query block size)
    int kv_block_size = 0;       // Bc (KV block size)
    float scale = 0.0f;          // 1/sqrt(head_size) or user-specified

    MLAS_KV_QUANT_TYPE quant_type = MLAS_KV_QUANT_TYPE::S8_PerTensor;
    bool per_channel_k = false;  // Whether K uses per-channel scales
    bool per_channel_v = false;  // Whether V uses per-channel scales

    int thread_count = 1;
    float* buffer = nullptr;
    size_t buffer_size_per_thread = 0;

    const float* query = nullptr;      // [B, N, S, H] FP32
    size_t q_batch_stride = 0;         // element stride between consecutive batches in `query`
                                       // (num_heads*S*H for unpacked, (num_heads+2*kv_num_heads)*S*H for packed QKV)
    const uint8_t* k_cache = nullptr;  // [B, kv_N, seqlen_present, packed_row_bytes] quantized
    const uint8_t* v_cache = nullptr;  // [B, kv_N, seqlen_present, packed_row_bytes] quantized
    const float* k_scale = nullptr;    // Scalar or per-channel scales for K
    const float* v_scale = nullptr;    // Scalar or per-channel scales for V
    float* output = nullptr;           // [B, S, N, H] FP32

    // Attention bias (additive, applied after QK GEMM before masking/softmax).
    // Shape: [B|1, N|1, S, T] where dimensions of size 1 are broadcast.
    const float* attention_bias = nullptr;            // nullptr if no bias
    int attention_bias_seqlen_stride = 0;             // stride along the T (total_seqlen) dimension = shape[3]
    bool attention_bias_broadcast_batch = true;       // true if shape[0] == 1
    bool attention_bias_broadcast_head = true;        // true if shape[1] == 1

    // Flash decoding fields (used when sequence_length == 1 and KV is split across threads).
    // Partials buffer stores per-(batch, head, kv_chunk) intermediate results:
    //   [m_partial, l_partial, output_partial[head_size]] for each chunk.
    float* flash_decoding_partials = nullptr;  // nullptr to disable flash decoding
    int kv_chunk_count = 0;                    // number of KV chunks = ceil(total_seqlen / kv_block_size)
};

/**
 * @brief Flash Attention with quantized KV cache.
 *
 * Implements tiled attention with online softmax, processing KV in blocks
 * to avoid materializing the full attention matrix. Supports causal masking
 * and local window attention.
 *
 * @param args       Pointer to argument structure.
 * @param ThreadPool Optional thread pool for parallelization.
 */
void
MLASCALL
MlasFlashAttentionQuantizedKV(
    MlasFlashAttentionQuantizedKVArgs* args,
    MLAS_THREADPOOL* ThreadPool
    );
