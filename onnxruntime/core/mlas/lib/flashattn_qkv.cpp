/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    flashattn_qkv.cpp

Abstract:

    Flash Attention kernel for quantized KV cache (INT8/INT4).

    Adapts the online-softmax tiled algorithm from flashattn.cpp to operate
    on quantized K/V buffers using MlasQKGemm (for Q×K^T) and
    MlasSVGemm with Beta=1.0 (for fused dequant + S×V accumulation).

    Supports causal masking and local window attention.

--*/

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>

#include "mlasi.h"
#include "mlas_qkv_quant.h"

void
MlasFlashAttentionQuantizedKVThreaded(
    void* argptr,
    std::ptrdiff_t thread_id
)
{
    const MlasFlashAttentionQuantizedKVArgs* args =
        reinterpret_cast<MlasFlashAttentionQuantizedKVArgs*>(argptr);

    const ptrdiff_t q_block_size = static_cast<ptrdiff_t>(args->q_block_size);
    const ptrdiff_t kv_block_size = static_cast<ptrdiff_t>(args->kv_block_size);
    const ptrdiff_t batch_size = static_cast<ptrdiff_t>(args->batch_size);
    const ptrdiff_t num_heads = static_cast<ptrdiff_t>(args->num_heads);
    const ptrdiff_t kv_num_heads = static_cast<ptrdiff_t>(args->kv_num_heads);
    const ptrdiff_t sequence_length = static_cast<ptrdiff_t>(args->sequence_length);
    const ptrdiff_t total_seqlen = static_cast<ptrdiff_t>(args->total_seqlen);
    const ptrdiff_t head_size = static_cast<ptrdiff_t>(args->head_size);
    const ptrdiff_t past_seqlen = static_cast<ptrdiff_t>(args->past_seqlen);
    const ptrdiff_t local_window_size = static_cast<ptrdiff_t>(args->local_window_size);
    const float scale = args->scale;
    const MLAS_KV_QUANT_TYPE quant_type = args->quant_type;

    float* buffer = args->buffer;
    const ptrdiff_t buffer_size_per_thread = static_cast<ptrdiff_t>(args->buffer_size_per_thread);
    const ptrdiff_t thread_count = static_cast<ptrdiff_t>(args->thread_count);

    const size_t packed_row_bytes = MlasKVQuantPackedRowBytes(quant_type, static_cast<size_t>(head_size));
    const size_t kv_num_heads_factor = static_cast<size_t>(num_heads / kv_num_heads);

#if defined(MLAS_TARGET_AMD64) || defined(MLAS_TARGET_LARCH64)
    auto&& mlas_platform = GetMlasPlatform();
#endif

    // Total tasks: one per (batch, head, q_block)
    const ptrdiff_t q_chunk_count = (sequence_length + q_block_size - 1) / q_block_size;
    const ptrdiff_t total_task_count = batch_size * num_heads * q_chunk_count;

    ptrdiff_t task_start = 0;
    ptrdiff_t task_end = 0;
    ptrdiff_t quotient = total_task_count / thread_count;
    ptrdiff_t remainder = total_task_count % thread_count;
    if (thread_id < remainder) {
        task_start = (quotient + 1) * thread_id;
        task_end = task_start + quotient + 1;
    } else {
        task_start = quotient * thread_id + remainder;
        task_end = task_start + quotient;
    }

    for (ptrdiff_t task_index = task_start; task_index < task_end; ++task_index) {
        ptrdiff_t batch_idx = task_index;
        ptrdiff_t q_idx = (batch_idx % q_chunk_count) * q_block_size;
        batch_idx /= q_chunk_count;
        ptrdiff_t head_idx = batch_idx % num_heads;
        batch_idx /= num_heads;

        // Per-thread buffer layout:
        //   l[q_block_size]             - running sum for online softmax
        //   m[q_block_size]             - running max for online softmax
        //   scores[q_block_size * kv_block_size] - QK scores (S)
        //   temp_output[q_block_size * head_size] - accumulated output
        char* buffer_ptr = reinterpret_cast<char*>(buffer) + thread_id * buffer_size_per_thread;
        float* l = reinterpret_cast<float*>(buffer_ptr);
        float* m = l + q_block_size;
        float* scores = m + q_block_size;
        float* temp_output = scores + q_block_size * kv_block_size;

        // Initialize running state
        for (ptrdiff_t t = 0; t < q_block_size; ++t) {
            m[t] = std::numeric_limits<float>::lowest();
            l[t] = 0.0f;
        }
        memset(temp_output, 0, static_cast<size_t>(q_block_size * head_size) * sizeof(float));

        const size_t row_size_q = static_cast<size_t>(std::min(q_block_size, sequence_length - q_idx));

        // Determine KV head index for GQA head sharing
        const size_t kv_head_idx = static_cast<size_t>(head_idx) / kv_num_heads_factor;

        // Pointers into quantized K/V caches
        // K cache layout: [batch, kv_num_heads, seqlen_present, packed_head_bytes]
        const size_t k_batch_head_offset =
            (static_cast<size_t>(batch_idx) * static_cast<size_t>(kv_num_heads) + kv_head_idx) *
            static_cast<size_t>(args->seqlen_present_kv) * packed_row_bytes;
        const uint8_t* k_cache_head = args->k_cache + k_batch_head_offset;

        const size_t v_batch_head_offset =
            (static_cast<size_t>(batch_idx) * static_cast<size_t>(kv_num_heads) + kv_head_idx) *
            static_cast<size_t>(args->seqlen_present_kv) * packed_row_bytes;
        const uint8_t* v_cache_head = args->v_cache + v_batch_head_offset;

        // K/V scale pointers
        const float* head_k_scale = args->per_channel_k
            ? args->k_scale + kv_head_idx * static_cast<size_t>(head_size)
            : args->k_scale;
        const float* head_v_scale = args->per_channel_v
            ? args->v_scale + kv_head_idx * static_cast<size_t>(head_size)
            : args->v_scale;

        // Q pointer: layout [batch, num_heads, seq, head_size]. The batch stride is
        // supplied separately (args->q_batch_stride) so the kernel works with both the
        // standard BNSH layout and packed-QKV input where Q/K/V are interleaved per batch.
        const float* q_ptr = args->query +
            static_cast<size_t>(batch_idx) * args->q_batch_stride +
            static_cast<size_t>(head_idx) * static_cast<size_t>(sequence_length) * static_cast<size_t>(head_size) +
            static_cast<size_t>(q_idx) * static_cast<size_t>(head_size);

        // Iterate over KV blocks
        for (ptrdiff_t ir = 0; ir < total_seqlen; ir += kv_block_size) {
            const size_t row_size_kv = static_cast<size_t>(std::min(kv_block_size, total_seqlen - ir));

            // Step 1: QK^T GEMM with quantized K block
            // K cache at row offset ir: pointer arithmetic on packed rows
            const uint8_t* k_block = k_cache_head + static_cast<size_t>(ir) * packed_row_bytes;

            MlasQKGemm(
                row_size_q,                         // M
                row_size_kv,                        // N
                static_cast<size_t>(head_size),     // K
                scale,                              // Alpha
                q_ptr,                              // A (FP32 query)
                static_cast<size_t>(head_size),     // lda
                k_block,                            // B (quantized K block)
                quant_type,
                head_k_scale,
                scores,                             // C (output scores)
                row_size_kv,                        // ldc
                nullptr                             // no thread pool (already threaded)
            );

            // Step 1b: Apply attention bias (additive) if present
            if (args->attention_bias != nullptr) {
                const ptrdiff_t bias_seqlen_stride =
                    static_cast<ptrdiff_t>(args->attention_bias_seqlen_stride);
                const ptrdiff_t bias_matrix_size =
                    static_cast<ptrdiff_t>(sequence_length) * bias_seqlen_stride;
                // The bias tensor has shape [batch|1, num_heads|1, S, T]; the batch
                // stride uses the actual head extent (1 when the head dim is broadcast).
                const ptrdiff_t bias_head_extent =
                    args->attention_bias_broadcast_head ? 1 : static_cast<ptrdiff_t>(num_heads);
                ptrdiff_t bias_offset = 0;
                if (!args->attention_bias_broadcast_batch) {
                    bias_offset += static_cast<ptrdiff_t>(batch_idx) *
                                   bias_head_extent * bias_matrix_size;
                }
                if (!args->attention_bias_broadcast_head) {
                    bias_offset += static_cast<ptrdiff_t>(head_idx) * bias_matrix_size;
                }
                // Add bias tile: bias[q_idx + irow, ir + jcol]
                for (ptrdiff_t irow = 0; irow < static_cast<ptrdiff_t>(row_size_q); ++irow) {
                    const float* bias_row = args->attention_bias + bias_offset +
                        (q_idx + irow) * bias_seqlen_stride + ir;
                    float* s_row = scores + irow * static_cast<ptrdiff_t>(row_size_kv);
                    for (ptrdiff_t jcol = 0; jcol < static_cast<ptrdiff_t>(row_size_kv); ++jcol) {
                        s_row[jcol] += bias_row[jcol];
                    }
                }
            }

            // Step 2: Apply causal mask and Step 3: Online softmax update
            for (ptrdiff_t irow = 0; irow < static_cast<ptrdiff_t>(row_size_q); ++irow) {
                float* p = scores + irow * static_cast<ptrdiff_t>(row_size_kv);
                const ptrdiff_t global_q_pos = past_seqlen + q_idx + irow;
                const ptrdiff_t causal_limit = global_q_pos + 1;  // can attend to positions [0, causal_limit)

                // Apply causal masking
                for (ptrdiff_t jcol = 0; jcol < static_cast<ptrdiff_t>(row_size_kv); ++jcol) {
                    ptrdiff_t kv_pos = ir + jcol;
                    if (kv_pos >= causal_limit) {
                        p[jcol] = std::numeric_limits<float>::lowest();
                    }
                }

                // Apply local window masking if enabled
                if (local_window_size >= 0) {
                    const ptrdiff_t window_start =
                        (causal_limit > local_window_size) ? (causal_limit - local_window_size) : 0;
                    for (ptrdiff_t jcol = 0; jcol < static_cast<ptrdiff_t>(row_size_kv); ++jcol) {
                        ptrdiff_t kv_pos = ir + jcol;
                        if (kv_pos < window_start) {
                            p[jcol] = std::numeric_limits<float>::lowest();
                        }
                    }
                }

                // Online softmax: find row max, update running max
#if defined(MLAS_TARGET_AMD64) || defined(MLAS_TARGET_LARCH64)
                float rowmax = mlas_platform.ReduceMaximumF32Kernel(p, row_size_kv);
#else
                float rowmax = MlasReduceMaximumF32Kernel(p, row_size_kv);
#endif

                // If the entire row is masked (all scores are -inf), zero the scores
                // so SVGemm contributes nothing and skip the softmax state update.
                if (rowmax == std::numeric_limits<float>::lowest()) {
                    memset(p, 0, row_size_kv * sizeof(float));
                    continue;
                }

                float m_old = m[irow];
                m[irow] = std::max(m[irow], rowmax);
                float m_diff = m_old - m[irow];  // <= 0

                // Compute exp(score - m_new) for each element
                float negmax = -m[irow];
#if defined(MLAS_TARGET_AMD64)
                float rowsum = mlas_platform.ComputeSumExpF32Kernel(p, p, row_size_kv, &negmax);
#else
                float rowsum = MlasComputeSumExpF32Kernel(p, p, row_size_kv, &negmax);
#endif

                // Rescale previous state
                if (ir != 0) {
                    float exp_diff = std::exp(m_diff);
                    l[irow] = exp_diff * l[irow] + rowsum;

                    // Rescale accumulated output
                    float* out_row = temp_output + irow * head_size;
                    for (ptrdiff_t icol = 0; icol < head_size; ++icol) {
                        out_row[icol] *= exp_diff;
                    }
                } else {
                    l[irow] = rowsum;
                }
            }

            // Step 4: Accumulate O += S_exp * V_block using fused dequant+GEMM
            const uint8_t* v_block = v_cache_head + static_cast<size_t>(ir) * packed_row_bytes;
            MlasSVGemm(
                row_size_q,                         // M
                static_cast<size_t>(head_size),     // N
                row_size_kv,                        // K
                scores,                             // A (exp softmax scores)
                row_size_kv,                        // lda
                v_block,                            // B (quantized V block)
                quant_type,
                head_v_scale,
                temp_output,                        // C (accumulated output)
                static_cast<size_t>(head_size),     // ldc
                1.0f,                               // Beta (accumulate)
                nullptr                             // no thread pool (already threaded)
            );
        }

        // Final: normalize output by l (softmax denominator)
        // Output layout: [batch, sequence_length, num_heads, head_size]
        float* output_row = args->output +
            (static_cast<size_t>(batch_idx) * static_cast<size_t>(sequence_length) +
             static_cast<size_t>(q_idx)) * static_cast<size_t>(num_heads) * static_cast<size_t>(head_size) +
            static_cast<size_t>(head_idx) * static_cast<size_t>(head_size);
        const ptrdiff_t output_row_stride = num_heads * head_size;

        for (ptrdiff_t irow = 0; irow < static_cast<ptrdiff_t>(row_size_q); ++irow) {
            float inv_l = (l[irow] > 0.0f) ? (1.0f / l[irow]) : 0.0f;
            float* src = temp_output + irow * head_size;
            for (ptrdiff_t icol = 0; icol < head_size; ++icol) {
                output_row[icol] = src[icol] * inv_l;
            }
            output_row += output_row_stride;
        }
    }
}

//
// Flash Decoding: Phase 1 - parallel partial attention over (batch, head, kv_chunk).
// Each task computes attention for one KV chunk and stores (m, l, partial_output)
// into the partials buffer.
//
void
MlasFlashDecodingQuantizedKVThreaded(
    void* argptr,
    std::ptrdiff_t thread_id
)
{
    const MlasFlashAttentionQuantizedKVArgs* args =
        reinterpret_cast<MlasFlashAttentionQuantizedKVArgs*>(argptr);

    const ptrdiff_t kv_block_size = static_cast<ptrdiff_t>(args->kv_block_size);
    const ptrdiff_t batch_size = static_cast<ptrdiff_t>(args->batch_size);
    const ptrdiff_t num_heads = static_cast<ptrdiff_t>(args->num_heads);
    const ptrdiff_t kv_num_heads = static_cast<ptrdiff_t>(args->kv_num_heads);
    const ptrdiff_t total_seqlen = static_cast<ptrdiff_t>(args->total_seqlen);
    const ptrdiff_t head_size = static_cast<ptrdiff_t>(args->head_size);
    const ptrdiff_t past_seqlen = static_cast<ptrdiff_t>(args->past_seqlen);
    const ptrdiff_t local_window_size = static_cast<ptrdiff_t>(args->local_window_size);
    const float scale = args->scale;
    const MLAS_KV_QUANT_TYPE quant_type = args->quant_type;

    float* buffer = args->buffer;
    const ptrdiff_t buffer_size_per_thread = static_cast<ptrdiff_t>(args->buffer_size_per_thread);
    const ptrdiff_t thread_count = static_cast<ptrdiff_t>(args->thread_count);

    const size_t packed_row_bytes = MlasKVQuantPackedRowBytes(quant_type, static_cast<size_t>(head_size));
    const size_t kv_num_heads_factor = static_cast<size_t>(num_heads / kv_num_heads);

    const ptrdiff_t kv_chunk_count = static_cast<ptrdiff_t>(args->kv_chunk_count);
    // Partials layout per entry: [m, l, output[head_size]]
    const ptrdiff_t partial_stride = 2 + head_size;

#if defined(MLAS_TARGET_AMD64) || defined(MLAS_TARGET_LARCH64)
    auto&& mlas_platform = GetMlasPlatform();
#endif

    // Total tasks: (batch, head, kv_chunk)
    const ptrdiff_t total_task_count = batch_size * num_heads * kv_chunk_count;

    ptrdiff_t task_start = 0;
    ptrdiff_t task_end = 0;
    ptrdiff_t quotient = total_task_count / thread_count;
    ptrdiff_t remainder = total_task_count % thread_count;
    if (thread_id < remainder) {
        task_start = (quotient + 1) * thread_id;
        task_end = task_start + quotient + 1;
    } else {
        task_start = quotient * thread_id + remainder;
        task_end = task_start + quotient;
    }

    for (ptrdiff_t task_index = task_start; task_index < task_end; ++task_index) {
        // Decompose task_index into (batch_idx, head_idx, kv_chunk_idx)
        ptrdiff_t tmp = task_index;
        ptrdiff_t kv_chunk_idx = tmp % kv_chunk_count;
        tmp /= kv_chunk_count;
        ptrdiff_t head_idx = tmp % num_heads;
        ptrdiff_t batch_idx = tmp / num_heads;

        // Per-thread scratch buffer: just scores[kv_block_size]
        char* buffer_ptr = reinterpret_cast<char*>(buffer) + thread_id * buffer_size_per_thread;
        float* scores = reinterpret_cast<float*>(buffer_ptr);

        // KV block range for this chunk
        const ptrdiff_t ir = kv_chunk_idx * kv_block_size;
        const size_t row_size_kv = static_cast<size_t>(std::min(kv_block_size, total_seqlen - ir));

        // Determine KV head index for GQA head sharing
        const size_t kv_head_idx = static_cast<size_t>(head_idx) / kv_num_heads_factor;

        // K/V cache pointers
        const size_t k_batch_head_offset =
            (static_cast<size_t>(batch_idx) * static_cast<size_t>(kv_num_heads) + kv_head_idx) *
            static_cast<size_t>(args->seqlen_present_kv) * packed_row_bytes;
        const uint8_t* k_cache_head = args->k_cache + k_batch_head_offset;

        const size_t v_batch_head_offset =
            (static_cast<size_t>(batch_idx) * static_cast<size_t>(kv_num_heads) + kv_head_idx) *
            static_cast<size_t>(args->seqlen_present_kv) * packed_row_bytes;
        const uint8_t* v_cache_head = args->v_cache + v_batch_head_offset;

        // K/V scale pointers
        const float* head_k_scale = args->per_channel_k
            ? args->k_scale + kv_head_idx * static_cast<size_t>(head_size)
            : args->k_scale;
        const float* head_v_scale = args->per_channel_v
            ? args->v_scale + kv_head_idx * static_cast<size_t>(head_size)
            : args->v_scale;

        // Q pointer: layout [batch, num_heads, 1, head_size] (sequence_length=1).
        // The batch stride is supplied separately to support packed-QKV input.
        const float* q_ptr = args->query +
            static_cast<size_t>(batch_idx) * args->q_batch_stride +
            static_cast<size_t>(head_idx) * static_cast<size_t>(head_size);

        // Step 1: QK^T GEMM for this KV chunk
        const uint8_t* k_block = k_cache_head + static_cast<size_t>(ir) * packed_row_bytes;
        MlasQKGemm(
            1,                                  // M (single query row)
            row_size_kv,                        // N
            static_cast<size_t>(head_size),     // K
            scale,                              // Alpha
            q_ptr,                              // A (FP32 query)
            static_cast<size_t>(head_size),     // lda
            k_block,                            // B (quantized K block)
            quant_type,
            head_k_scale,
            scores,                             // C (output scores)
            row_size_kv,                        // ldc
            nullptr
        );

        // Step 1b: Apply attention bias if present
        if (args->attention_bias != nullptr) {
            const ptrdiff_t bias_seqlen_stride =
                static_cast<ptrdiff_t>(args->attention_bias_seqlen_stride);
            const ptrdiff_t bias_matrix_size = bias_seqlen_stride;  // S=1
            // The bias tensor has shape [batch|1, num_heads|1, S, T]; the batch stride
            // uses the actual head extent (1 when the head dim is broadcast).
            const ptrdiff_t bias_head_extent =
                args->attention_bias_broadcast_head ? 1 : static_cast<ptrdiff_t>(num_heads);
            ptrdiff_t bias_offset = 0;
            if (!args->attention_bias_broadcast_batch) {
                bias_offset += static_cast<ptrdiff_t>(batch_idx) *
                               bias_head_extent * bias_matrix_size;
            }
            if (!args->attention_bias_broadcast_head) {
                bias_offset += static_cast<ptrdiff_t>(head_idx) * bias_matrix_size;
            }
            const float* bias_row = args->attention_bias + bias_offset + ir;
            for (ptrdiff_t jcol = 0; jcol < static_cast<ptrdiff_t>(row_size_kv); ++jcol) {
                scores[jcol] += bias_row[jcol];
            }
        }

        // Step 2: Apply causal mask
        const ptrdiff_t global_q_pos = past_seqlen;  // sequence_length=1, q_idx=0
        const ptrdiff_t causal_limit = global_q_pos + 1;
        for (ptrdiff_t jcol = 0; jcol < static_cast<ptrdiff_t>(row_size_kv); ++jcol) {
            ptrdiff_t kv_pos = ir + jcol;
            if (kv_pos >= causal_limit) {
                scores[jcol] = std::numeric_limits<float>::lowest();
            }
        }

        // Apply local window masking if enabled
        if (local_window_size >= 0) {
            const ptrdiff_t window_start =
                (causal_limit > local_window_size) ? (causal_limit - local_window_size) : 0;
            for (ptrdiff_t jcol = 0; jcol < static_cast<ptrdiff_t>(row_size_kv); ++jcol) {
                ptrdiff_t kv_pos = ir + jcol;
                if (kv_pos < window_start) {
                    scores[jcol] = std::numeric_limits<float>::lowest();
                }
            }
        }

        // Step 3: Compute local softmax statistics (m, l) and exp scores
#if defined(MLAS_TARGET_AMD64) || defined(MLAS_TARGET_LARCH64)
        float rowmax = mlas_platform.ReduceMaximumF32Kernel(scores, row_size_kv);
#else
        float rowmax = MlasReduceMaximumF32Kernel(scores, row_size_kv);
#endif

        // Pointer to this task's partial in the partials buffer
        const ptrdiff_t partial_index =
            (batch_idx * num_heads + head_idx) * kv_chunk_count + kv_chunk_idx;
        float* partial = args->flash_decoding_partials + partial_index * partial_stride;
        float* partial_m = partial;
        float* partial_l = partial + 1;
        float* partial_output = partial + 2;

        if (rowmax == std::numeric_limits<float>::lowest()) {
            // Entire chunk is masked: store sentinel
            *partial_m = std::numeric_limits<float>::lowest();
            *partial_l = 0.0f;
            memset(partial_output, 0, static_cast<size_t>(head_size) * sizeof(float));
            continue;
        }

        *partial_m = rowmax;
        float negmax = -rowmax;
#if defined(MLAS_TARGET_AMD64)
        float rowsum = mlas_platform.ComputeSumExpF32Kernel(scores, scores, row_size_kv, &negmax);
#else
        float rowsum = MlasComputeSumExpF32Kernel(scores, scores, row_size_kv, &negmax);
#endif
        *partial_l = rowsum;

        // Step 4: S_exp * V_block -> partial_output
        const uint8_t* v_block = v_cache_head + static_cast<size_t>(ir) * packed_row_bytes;
        memset(partial_output, 0, static_cast<size_t>(head_size) * sizeof(float));
        MlasSVGemm(
            1,                                  // M
            static_cast<size_t>(head_size),     // N
            row_size_kv,                        // K
            scores,                             // A (exp softmax scores)
            row_size_kv,                        // lda
            v_block,                            // B (quantized V block)
            quant_type,
            head_v_scale,
            partial_output,                     // C (output for this chunk)
            static_cast<size_t>(head_size),     // ldc
            0.0f,                               // Beta=0 (overwrite)
            nullptr
        );
    }
}

//
// Flash Decoding: Phase 2 - reduce partials for each (batch, head) into final output.
//
void
MlasFlashDecodingReduceThreaded(
    void* argptr,
    std::ptrdiff_t thread_id
)
{
    const MlasFlashAttentionQuantizedKVArgs* args =
        reinterpret_cast<MlasFlashAttentionQuantizedKVArgs*>(argptr);

    const ptrdiff_t batch_size = static_cast<ptrdiff_t>(args->batch_size);
    const ptrdiff_t num_heads = static_cast<ptrdiff_t>(args->num_heads);
    const ptrdiff_t head_size = static_cast<ptrdiff_t>(args->head_size);
    const ptrdiff_t kv_chunk_count = static_cast<ptrdiff_t>(args->kv_chunk_count);
    const ptrdiff_t thread_count = static_cast<ptrdiff_t>(args->thread_count);
    const ptrdiff_t partial_stride = 2 + head_size;

    // Total reduction tasks: one per (batch, head)
    const ptrdiff_t total_task_count = batch_size * num_heads;

    ptrdiff_t task_start = 0;
    ptrdiff_t task_end = 0;
    ptrdiff_t quotient = total_task_count / thread_count;
    ptrdiff_t remainder = total_task_count % thread_count;
    if (thread_id < remainder) {
        task_start = (quotient + 1) * thread_id;
        task_end = task_start + quotient + 1;
    } else {
        task_start = quotient * thread_id + remainder;
        task_end = task_start + quotient;
    }

    for (ptrdiff_t task_index = task_start; task_index < task_end; ++task_index) {
        ptrdiff_t head_idx = task_index % num_heads;
        ptrdiff_t batch_idx = task_index / num_heads;

        // Pointer to this (batch, head)'s partials: kv_chunk_count entries
        const float* partials_base = args->flash_decoding_partials +
            task_index * kv_chunk_count * partial_stride;

        // Find global max across all chunks
        float global_m = std::numeric_limits<float>::lowest();
        for (ptrdiff_t c = 0; c < kv_chunk_count; ++c) {
            float chunk_m = partials_base[c * partial_stride];
            global_m = std::max(global_m, chunk_m);
        }

        // If all chunks are masked, output zeros
        if (global_m == std::numeric_limits<float>::lowest()) {
            float* output_ptr = args->output +
                static_cast<size_t>(batch_idx) * static_cast<size_t>(num_heads) * static_cast<size_t>(head_size) +
                static_cast<size_t>(head_idx) * static_cast<size_t>(head_size);
            memset(output_ptr, 0, static_cast<size_t>(head_size) * sizeof(float));
            continue;
        }

        // Accumulate rescaled outputs and l values
        float global_l = 0.0f;
        // Use the output location directly for accumulation
        // Output layout: [batch, sequence_length=1, num_heads, head_size]
        float* output_ptr = args->output +
            static_cast<size_t>(batch_idx) * static_cast<size_t>(num_heads) * static_cast<size_t>(head_size) +
            static_cast<size_t>(head_idx) * static_cast<size_t>(head_size);
        memset(output_ptr, 0, static_cast<size_t>(head_size) * sizeof(float));

        for (ptrdiff_t c = 0; c < kv_chunk_count; ++c) {
            const float* partial = partials_base + c * partial_stride;
            float chunk_m = partial[0];
            float chunk_l = partial[1];
            const float* chunk_output = partial + 2;

            if (chunk_l <= 0.0f) {
                continue;  // masked chunk contributes nothing
            }

            float rescale = std::exp(chunk_m - global_m);
            global_l += rescale * chunk_l;

            // partial_output = S_exp * V where sum(S_exp) = l_c (unnormalized).
            // Rescale by exp(m_c - global_m) to align all chunks to the same max.
            for (ptrdiff_t i = 0; i < head_size; ++i) {
                output_ptr[i] += rescale * chunk_output[i];
            }
        }

        // output = sum_c(rescale_c * partial_output_c) / global_l
        float inv_l = (global_l > 0.0f) ? (1.0f / global_l) : 0.0f;
        for (ptrdiff_t i = 0; i < head_size; ++i) {
            output_ptr[i] *= inv_l;
        }
    }
}

void
MLASCALL
MlasFlashAttentionQuantizedKV(
    MlasFlashAttentionQuantizedKVArgs* args,
    MLAS_THREADPOOL* ThreadPool
)
{
    if (args->flash_decoding_partials != nullptr && args->sequence_length == 1) {
        // Flash decoding: two-phase approach.
        // Phase 1: parallel partial computation over (batch, head, kv_chunk).
        MlasExecuteThreaded(
            MlasFlashDecodingQuantizedKVThreaded,
            static_cast<void*>(args),
            static_cast<std::ptrdiff_t>(args->thread_count),
            ThreadPool
        );
        // Phase 2: reduce partials into final output (parallel over batch*heads).
        MlasExecuteThreaded(
            MlasFlashDecodingReduceThreaded,
            static_cast<void*>(args),
            static_cast<std::ptrdiff_t>(args->thread_count),
            ThreadPool
        );
    } else {
        MlasExecuteThreaded(
            MlasFlashAttentionQuantizedKVThreaded,
            static_cast<void*>(args),
            static_cast<std::ptrdiff_t>(args->thread_count),
            ThreadPool
        );
    }
}
