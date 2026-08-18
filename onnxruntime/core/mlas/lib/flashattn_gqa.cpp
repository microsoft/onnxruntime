/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    flashattn_gqa.cpp

Abstract:

    Flash Attention kernel for the non-quantized (FP32) GroupQueryAttention
    KV cache.

    Adapts the online-softmax tiled algorithm from flashattn.cpp to operate on
    an FP32 present K/V cache laid out as BNSH
    ([batch, kv_num_heads, seqlen_present, head_size]) and to support GQA head
    grouping (num_heads % kv_num_heads == 0), causal masking, local window
    attention, additive attention bias, and an optional flash-decoding split
    over the KV sequence dimension for single-token decode.

    For multi-token prefill (sequence_length > 1) QK^T and S*V use the
    single-threaded SGEMM primitive MlasSgemmOperation. For single-token decode
    (sequence_length == 1, including the flash-decoding KV split) the M == 1
    GEMVs use the local MlasGQADecodeQK / MlasGQADecodeSV helpers to avoid SGEMM
    packing overhead. The outer parallelism is provided by MlasExecuteThreaded.

--*/

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>

#include "mlasi.h"

//
// Decode (sequence_length == 1) GEMV helpers.
//
// With a single query token the QK^T and S*V products degenerate into
// matrix-vector products. Computing them directly streams the K and V cache
// exactly once and avoids the SGEMM B-packing overhead that otherwise dominates
// the tiny M = 1 GEMMs. These helpers live in the baseline-ISA MLAS translation
// unit, so the inner loops are written with independent accumulator lanes and a
// map-style update so the compiler can vectorize them without -ffast-math
// (which would be required to reassociate a plain scalar float reduction).
//

// QK^T GEMV: scores[t] = scale * dot(q[0..H), K[t*H .. t*H+H))  for t in [0, n_kv).
static void
MlasGQADecodeQK(
    const float* q,
    const float* k_cache,
    std::ptrdiff_t n_kv,
    std::ptrdiff_t head_size,
    float scale,
    float* scores
)
{
    constexpr std::ptrdiff_t kLanes = 8;
    for (std::ptrdiff_t t = 0; t < n_kv; ++t) {
        const float* krow = k_cache + t * head_size;
        float acc[kLanes] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
        std::ptrdiff_t h = 0;
        for (; h + kLanes <= head_size; h += kLanes) {
            for (std::ptrdiff_t j = 0; j < kLanes; ++j) {
                acc[j] += q[h + j] * krow[h + j];
            }
        }
        float sum = ((acc[0] + acc[1]) + (acc[2] + acc[3])) +
                    ((acc[4] + acc[5]) + (acc[6] + acc[7]));
        for (; h < head_size; ++h) {
            sum += q[h] * krow[h];
        }
        scores[t] = sum * scale;
    }
}

// S*V GEMV (accumulate): out[h] = sum_t probs[t] * V[t*H + h]  for h in [0, head_size).
// `out` is overwritten (initialized to zero) before accumulation.
static void
MlasGQADecodeSV(
    const float* probs,
    const float* v_cache,
    std::ptrdiff_t n_kv,
    std::ptrdiff_t head_size,
    float* out
)
{
    for (std::ptrdiff_t h = 0; h < head_size; ++h) {
        out[h] = 0.0f;
    }
    for (std::ptrdiff_t t = 0; t < n_kv; ++t) {
        const float p = probs[t];
        const float* vrow = v_cache + t * head_size;
        for (std::ptrdiff_t h = 0; h < head_size; ++h) {
            out[h] += p * vrow[h];
        }
    }
}

void
MlasFlashAttentionGQAThreaded(
    void* argptr,
    std::ptrdiff_t thread_id
)
{
    const MlasFlashAttentionGQAArgs* args =
        reinterpret_cast<MlasFlashAttentionGQAArgs*>(argptr);

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

    float* buffer = args->buffer;
    const ptrdiff_t buffer_size_per_thread = static_cast<ptrdiff_t>(args->buffer_size_per_thread);
    const ptrdiff_t thread_count = static_cast<ptrdiff_t>(args->thread_count);

    const size_t kv_num_heads_factor = static_cast<size_t>(num_heads / kv_num_heads);
    const size_t kv_head_stride =
        static_cast<size_t>(args->seqlen_present_kv) * static_cast<size_t>(head_size);

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
        //   l[q_block_size]                       - running sum for online softmax
        //   m[q_block_size]                       - running max for online softmax
        //   scores[q_block_size * kv_block_size]  - QK scores (S)
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

        // K/V cache pointers. Layout: [batch, kv_num_heads, seqlen_present, head_size]
        const size_t kv_batch_head_offset =
            (static_cast<size_t>(batch_idx) * static_cast<size_t>(kv_num_heads) + kv_head_idx) *
            kv_head_stride;
        const float* k_cache_head = args->k_cache + kv_batch_head_offset;
        const float* v_cache_head = args->v_cache + kv_batch_head_offset;

        // Q pointer: layout [batch, num_heads, seq, head_size]. The batch stride is
        // supplied separately (args->q_batch_stride) so the kernel works with both the
        // standard BNSH layout and packed-QKV input where Q/K/V are interleaved per batch.
        const float* q_ptr = args->query +
            static_cast<size_t>(batch_idx) * args->q_batch_stride +
            static_cast<size_t>(head_idx) * static_cast<size_t>(sequence_length) * static_cast<size_t>(head_size) +
            static_cast<size_t>(q_idx) * static_cast<size_t>(head_size);

        // Causal early-termination bound: the largest global query position in this
        // q_block is (past_seqlen + q_idx + row_size_q - 1), so it can attend to KV
        // positions up to that index inclusive. Any KV block starting at or beyond
        // (past_seqlen + q_idx + row_size_q) is fully causally masked for every row in
        // the block, so it contributes nothing and can be skipped. This avoids the
        // wasted QK/SV GEMMs over the causal upper triangle during prefill.
        const ptrdiff_t kv_causal_limit =
            past_seqlen + q_idx + static_cast<ptrdiff_t>(row_size_q);

        // Iterate over KV blocks
        for (ptrdiff_t ir = 0; ir < total_seqlen; ir += kv_block_size) {
            if (ir >= kv_causal_limit) {
                break;
            }
            const size_t row_size_kv = static_cast<size_t>(std::min(kv_block_size, total_seqlen - ir));

            // Step 1: QK^T GEMM with FP32 K block
            const float* k_block = k_cache_head + static_cast<size_t>(ir) * static_cast<size_t>(head_size);
            MlasSgemmOperation(
                CblasNoTrans,
                CblasTrans,
                row_size_q,                         // M
                row_size_kv,                        // N
                static_cast<size_t>(head_size),     // K
                scale,                              // alpha
                q_ptr,                              // A (FP32 query)
                static_cast<size_t>(head_size),     // lda
                k_block,                            // B (FP32 K block)
                static_cast<size_t>(head_size),     // ldb
                0.0f,                               // beta
                scores,                             // C (output scores)
                row_size_kv                         // ldc
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
                // so the S*V GEMM contributes nothing and skip the softmax state update.
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

            // Step 4: Accumulate O += S_exp * V_block
            const float* v_block = v_cache_head + static_cast<size_t>(ir) * static_cast<size_t>(head_size);
            MlasSgemmOperation(
                CblasNoTrans,
                CblasNoTrans,
                row_size_q,                         // M
                static_cast<size_t>(head_size),     // N
                row_size_kv,                        // K
                1.0f,                               // alpha
                scores,                             // A (exp softmax scores)
                row_size_kv,                        // lda
                v_block,                            // B (FP32 V block)
                static_cast<size_t>(head_size),     // ldb
                ir == 0 ? 0.0f : 1.0f,              // beta (accumulate after first block)
                temp_output,                        // C (accumulated output)
                static_cast<size_t>(head_size)      // ldc
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
MlasFlashDecodingGQAThreaded(
    void* argptr,
    std::ptrdiff_t thread_id
)
{
    const MlasFlashAttentionGQAArgs* args =
        reinterpret_cast<MlasFlashAttentionGQAArgs*>(argptr);

    const ptrdiff_t kv_block_size = static_cast<ptrdiff_t>(args->kv_block_size);
    const ptrdiff_t batch_size = static_cast<ptrdiff_t>(args->batch_size);
    const ptrdiff_t num_heads = static_cast<ptrdiff_t>(args->num_heads);
    const ptrdiff_t kv_num_heads = static_cast<ptrdiff_t>(args->kv_num_heads);
    const ptrdiff_t total_seqlen = static_cast<ptrdiff_t>(args->total_seqlen);
    const ptrdiff_t head_size = static_cast<ptrdiff_t>(args->head_size);
    const ptrdiff_t past_seqlen = static_cast<ptrdiff_t>(args->past_seqlen);
    const ptrdiff_t local_window_size = static_cast<ptrdiff_t>(args->local_window_size);
    const float scale = args->scale;

    float* buffer = args->buffer;
    const ptrdiff_t buffer_size_per_thread = static_cast<ptrdiff_t>(args->buffer_size_per_thread);
    const ptrdiff_t thread_count = static_cast<ptrdiff_t>(args->thread_count);

    const size_t kv_num_heads_factor = static_cast<size_t>(num_heads / kv_num_heads);
    const size_t kv_head_stride =
        static_cast<size_t>(args->seqlen_present_kv) * static_cast<size_t>(head_size);

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
        const size_t kv_batch_head_offset =
            (static_cast<size_t>(batch_idx) * static_cast<size_t>(kv_num_heads) + kv_head_idx) *
            kv_head_stride;
        const float* k_cache_head = args->k_cache + kv_batch_head_offset;
        const float* v_cache_head = args->v_cache + kv_batch_head_offset;

        // Q pointer: layout [batch, num_heads, 1, head_size] (sequence_length=1).
        // The batch stride is supplied separately to support packed-QKV input.
        const float* q_ptr = args->query +
            static_cast<size_t>(batch_idx) * args->q_batch_stride +
            static_cast<size_t>(head_idx) * static_cast<size_t>(head_size);

        // Step 1: QK^T GEMV for this KV chunk (M = 1)
        const float* k_block = k_cache_head + static_cast<size_t>(ir) * static_cast<size_t>(head_size);
        MlasGQADecodeQK(q_ptr, k_block, static_cast<std::ptrdiff_t>(row_size_kv), head_size, scale, scores);

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

        // Step 4: S_exp * V_block -> partial_output (M = 1)
        const float* v_block = v_cache_head + static_cast<size_t>(ir) * static_cast<size_t>(head_size);
        MlasGQADecodeSV(scores, v_block, static_cast<std::ptrdiff_t>(row_size_kv), head_size, partial_output);
    }
}

//
// Flash Decoding: Phase 2 - reduce partials for each (batch, head) into final output.
//
void
MlasFlashDecodingGQAReduceThreaded(
    void* argptr,
    std::ptrdiff_t thread_id
)
{
    const MlasFlashAttentionGQAArgs* args =
        reinterpret_cast<MlasFlashAttentionGQAArgs*>(argptr);

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

        // Output layout: [batch, sequence_length=1, num_heads, head_size]
        float* output_ptr = args->output +
            static_cast<size_t>(batch_idx) * static_cast<size_t>(num_heads) * static_cast<size_t>(head_size) +
            static_cast<size_t>(head_idx) * static_cast<size_t>(head_size);

        // If all chunks are masked, output zeros
        if (global_m == std::numeric_limits<float>::lowest()) {
            memset(output_ptr, 0, static_cast<size_t>(head_size) * sizeof(float));
            continue;
        }

        // Accumulate rescaled outputs and l values
        float global_l = 0.0f;
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

//
// Decode kernel for sequence_length == 1 without KV-split (batch * heads >=
// thread_count). Parallelizes over (batch, head); each task attends the single
// query token to the whole KV cache with a pair of GEMVs and a two-pass softmax.
// Decode needs no causal masking (the single new token is the most recent
// position and attends to every cached token); only optional local-window
// masking and additive bias are applied.
//
void
MlasGQADecodeGQAThreaded(
    void* argptr,
    std::ptrdiff_t thread_id
)
{
    const MlasFlashAttentionGQAArgs* args =
        reinterpret_cast<MlasFlashAttentionGQAArgs*>(argptr);

    const ptrdiff_t batch_size = static_cast<ptrdiff_t>(args->batch_size);
    const ptrdiff_t num_heads = static_cast<ptrdiff_t>(args->num_heads);
    const ptrdiff_t kv_num_heads = static_cast<ptrdiff_t>(args->kv_num_heads);
    const ptrdiff_t total_seqlen = static_cast<ptrdiff_t>(args->total_seqlen);
    const ptrdiff_t head_size = static_cast<ptrdiff_t>(args->head_size);
    const ptrdiff_t local_window_size = static_cast<ptrdiff_t>(args->local_window_size);
    const float scale = args->scale;

    float* buffer = args->buffer;
    const ptrdiff_t buffer_size_per_thread = static_cast<ptrdiff_t>(args->buffer_size_per_thread);
    const ptrdiff_t thread_count = static_cast<ptrdiff_t>(args->thread_count);

    const size_t kv_num_heads_factor = static_cast<size_t>(num_heads / kv_num_heads);
    const size_t kv_head_stride =
        static_cast<size_t>(args->seqlen_present_kv) * static_cast<size_t>(head_size);

#if defined(MLAS_TARGET_AMD64) || defined(MLAS_TARGET_LARCH64)
    auto&& mlas_platform = GetMlasPlatform();
#endif

    // One task per (batch, head).
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

    // Local-window low bound: decode can attend to KV positions [window_start, total_seqlen).
    // causal_limit == past_seqlen + 1 == total_seqlen for the single new token.
    const ptrdiff_t window_start =
        (local_window_size >= 0 && total_seqlen > local_window_size) ? (total_seqlen - local_window_size) : 0;

    // Per-thread scratch: scores[total_seqlen] followed by temp_output[head_size].
    char* buffer_ptr = reinterpret_cast<char*>(buffer) + thread_id * buffer_size_per_thread;
    float* scores = reinterpret_cast<float*>(buffer_ptr);
    float* temp_output = scores + total_seqlen;

    for (ptrdiff_t task_index = task_start; task_index < task_end; ++task_index) {
        const ptrdiff_t head_idx = task_index % num_heads;
        const ptrdiff_t batch_idx = task_index / num_heads;

        // KV head index for GQA head sharing.
        const size_t kv_head_idx = static_cast<size_t>(head_idx) / kv_num_heads_factor;
        const size_t kv_batch_head_offset =
            (static_cast<size_t>(batch_idx) * static_cast<size_t>(kv_num_heads) + kv_head_idx) *
            kv_head_stride;
        const float* k_cache_head = args->k_cache + kv_batch_head_offset;
        const float* v_cache_head = args->v_cache + kv_batch_head_offset;

        // Q pointer: layout [batch, num_heads, 1, head_size]; batch stride supplied
        // separately to support packed-QKV input.
        const float* q_ptr = args->query +
            static_cast<size_t>(batch_idx) * args->q_batch_stride +
            static_cast<size_t>(head_idx) * static_cast<size_t>(head_size);

        // Step 1: QK^T GEMV -> scores[0..T)
        MlasGQADecodeQK(q_ptr, k_cache_head, total_seqlen, head_size, scale, scores);

        // Step 1b: additive attention bias (shape [batch|1, num_heads|1, S=1, T]).
        if (args->attention_bias != nullptr) {
            const ptrdiff_t bias_matrix_size =
                static_cast<ptrdiff_t>(args->attention_bias_seqlen_stride);  // S == 1
            const ptrdiff_t bias_head_extent =
                args->attention_bias_broadcast_head ? 1 : static_cast<ptrdiff_t>(num_heads);
            ptrdiff_t bias_offset = 0;
            if (!args->attention_bias_broadcast_batch) {
                bias_offset += static_cast<ptrdiff_t>(batch_idx) * bias_head_extent * bias_matrix_size;
            }
            if (!args->attention_bias_broadcast_head) {
                bias_offset += static_cast<ptrdiff_t>(head_idx) * bias_matrix_size;
            }
            const float* bias_row = args->attention_bias + bias_offset;
            for (ptrdiff_t t = 0; t < total_seqlen; ++t) {
                scores[t] += bias_row[t];
            }
        }

        // Step 2: local-window masking (no causal mask needed for decode).
        if (window_start > 0) {
            for (ptrdiff_t t = 0; t < window_start; ++t) {
                scores[t] = std::numeric_limits<float>::lowest();
            }
        }

        // Step 3: softmax over scores[0..T).
#if defined(MLAS_TARGET_AMD64) || defined(MLAS_TARGET_LARCH64)
        float rowmax = mlas_platform.ReduceMaximumF32Kernel(scores, total_seqlen);
#else
        float rowmax = MlasReduceMaximumF32Kernel(scores, total_seqlen);
#endif

        // Output layout: [batch, sequence_length=1, num_heads, head_size]
        float* output_ptr = args->output +
            (static_cast<size_t>(batch_idx) * static_cast<size_t>(num_heads) +
             static_cast<size_t>(head_idx)) * static_cast<size_t>(head_size);

        if (rowmax == std::numeric_limits<float>::lowest()) {
            memset(output_ptr, 0, static_cast<size_t>(head_size) * sizeof(float));
            continue;
        }

        float negmax = -rowmax;
#if defined(MLAS_TARGET_AMD64)
        float rowsum = mlas_platform.ComputeSumExpF32Kernel(scores, scores, total_seqlen, &negmax);
#else
        float rowsum = MlasComputeSumExpF32Kernel(scores, scores, total_seqlen, &negmax);
#endif

        // Step 4: S_exp * V GEMV -> temp_output, then normalize by 1/l.
        MlasGQADecodeSV(scores, v_cache_head, total_seqlen, head_size, temp_output);

        const float inv_l = (rowsum > 0.0f) ? (1.0f / rowsum) : 0.0f;
        for (ptrdiff_t h = 0; h < head_size; ++h) {
            output_ptr[h] = temp_output[h] * inv_l;
        }
    }
}

void
MLASCALL
MlasFlashAttentionGQA(
    MlasFlashAttentionGQAArgs* args,
    MLAS_THREADPOOL* ThreadPool
)
{
    if (args->sequence_length == 1) {
        // Decode: M = 1, use the GEMV kernels (no SGEMM packing overhead).
        if (args->flash_decoding_partials != nullptr) {
            // Flash decoding: two-phase approach when KV is partitioned across threads.
            // Phase 1: parallel partial computation over (batch, head, kv_chunk).
            MlasExecuteThreaded(
                MlasFlashDecodingGQAThreaded,
                static_cast<void*>(args),
                static_cast<std::ptrdiff_t>(args->thread_count),
                ThreadPool
            );
            // Phase 2: reduce partials into final output (parallel over batch*heads).
            MlasExecuteThreaded(
                MlasFlashDecodingGQAReduceThreaded,
                static_cast<void*>(args),
                static_cast<std::ptrdiff_t>(args->thread_count),
                ThreadPool
            );
        } else {
            // Single-pass decode parallelized over (batch, head).
            MlasExecuteThreaded(
                MlasGQADecodeGQAThreaded,
                static_cast<void*>(args),
                static_cast<std::ptrdiff_t>(args->thread_count),
                ThreadPool
            );
        }
    } else {
        // Prefill (sequence_length > 1): tiled online-softmax SGEMM kernel.
        MlasExecuteThreaded(
            MlasFlashAttentionGQAThreaded,
            static_cast<void*>(args),
            static_cast<std::ptrdiff_t>(args->thread_count),
            ThreadPool
        );
    }
}

