/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    linear_attention.cpp

Abstract:

    This module implements the FP32 linear (recurrent) attention kernels used
    by the LinearAttention contrib operator.

    Unlike softmax attention, the recurrence is strictly sequential in the
    token dimension, so parallelism comes entirely from the (batch, kv head)
    dimensions - each pair owns a private state matrix and is fully
    independent.

--*/

#include "linear_attention.h"

#include <algorithm>
#include <cassert>
#include <cmath>

//
// Below this state size the SGEMM call overhead (argument marshalling, packing
// decisions, kernel selection) dominates the ~2 * d_k * d_v flops of the M=1 /
// K=1 shapes involved, so scalar loops win.
//
constexpr size_t MlasLinearAttentionSgemmThreshold = 4096;

size_t
MLASCALL
MlasLinearAttentionOutputHiddenSize(
    int q_num_heads,
    int kv_num_heads,
    int v_head_size
)
{
    return static_cast<size_t>(std::max(q_num_heads, kv_num_heads)) *
           static_cast<size_t>(v_head_size);
}

size_t
MLASCALL
MlasLinearAttentionBufferSizePerThread(
    int k_head_size,
    int v_head_size
)
{
    //
    // v_head_size floats hold the retrieval / delta vector. k_head_size floats
    // are reserved so a fused kernel can materialize exp(g_t) without growing
    // the contract. Round up so each thread's slice starts on the preferred
    // buffer alignment, which is queried rather than assumed to be 64.
    //
    const size_t Bytes = (static_cast<size_t>(k_head_size) +
                          static_cast<size_t>(v_head_size)) * sizeof(float);

    const size_t BufferAlignment = MlasGetPreferredBufferAlignment();

    return (Bytes + BufferAlignment - 1) & ~(BufferAlignment - 1);
}

void
MlasLinearAttentionProcessHead(
    const MLAS_LINEAR_ATTENTION_WORK* Work
)
{
    const size_t dk = Work->KHeadSize;
    const size_t dv = Work->VHeadSize;
    const size_t heads_per_group = Work->HeadsPerGroup;
    const float scale = Work->Scale;

    const bool needs_decay = (Work->Rule == MlasLinearAttentionRuleGated ||
                              Work->Rule == MlasLinearAttentionRuleGatedDelta);
    const bool needs_beta = (Work->Rule == MlasLinearAttentionRuleDelta ||
                             Work->Rule == MlasLinearAttentionRuleGatedDelta);
    const bool needs_retrieval = needs_beta;
    const bool decay_per_key_dim = (Work->DecayLayout == MlasLinearAttentionDecayPerKeyDim);

    const bool use_sgemm = (dk * dv >= MlasLinearAttentionSgemmThreshold);

    float* S = Work->State;
    float* retrieved_buf = Work->Scratch;

    for (size_t t = 0; t < Work->SequenceLength; ++t) {
        const float* kt = Work->Key + t * Work->KeyTokenStride;
        const float* vt = Work->Value + t * Work->ValueTokenStride;

        // ---- Step 1: Apply decay S *= exp(g_t) ----
        if (needs_decay) {
            const float* gt = Work->Decay + t * Work->DecayTokenStride;
            if (decay_per_key_dim) {
                for (size_t i = 0; i < dk; ++i) {
                    float exp_g = std::exp(gt[i]);
                    // Scale row i of S by exp_g
                    for (size_t j = 0; j < dv; ++j) {
                        S[i * dv + j] *= exp_g;
                    }
                }
            } else {
                float exp_g = std::exp(gt[0]);
                for (size_t i = 0; i < dk * dv; ++i) {
                    S[i] *= exp_g;
                }
            }
        }

        // ---- Step 2: Retrieval = S^T @ k_t ----
        if (needs_retrieval) {
            if (use_sgemm) {
                MlasSgemmOperation(CblasNoTrans, CblasNoTrans, 1, dv, dk,
                                   1.0f, kt, dk, S, dv, 0.0f, retrieved_buf, dv);
            } else {
                for (size_t j = 0; j < dv; ++j) {
                    float acc = 0.0f;
                    for (size_t i = 0; i < dk; ++i) {
                        acc += S[i * dv + j] * kt[i];
                    }
                    retrieved_buf[j] = acc;
                }
            }
        }

        // ---- Step 3: State update ----
        if (needs_beta) {
            const float bt = Work->Beta[t * Work->BetaTokenStride];
            // Compute delta = beta * (v_t - retrieved) in-place into retrieved_buf
            for (size_t j = 0; j < dv; ++j) {
                retrieved_buf[j] = bt * (vt[j] - retrieved_buf[j]);
            }
            // S += k_t outer delta
            if (use_sgemm) {
                MlasSgemmOperation(CblasNoTrans, CblasNoTrans, dk, dv, 1,
                                   1.0f, kt, 1, retrieved_buf, dv, 1.0f, S, dv);
            } else {
                for (size_t i = 0; i < dk; ++i) {
                    float* s_row = S + i * dv;
                    const float ki = kt[i];
                    for (size_t j = 0; j < dv; ++j) {
                        s_row[j] += ki * retrieved_buf[j];
                    }
                }
            }
        } else {
            // linear/gated: S += k_t outer v_t
            if (use_sgemm) {
                MlasSgemmOperation(CblasNoTrans, CblasNoTrans, dk, dv, 1,
                                   1.0f, kt, 1, vt, dv, 1.0f, S, dv);
            } else {
                for (size_t i = 0; i < dk; ++i) {
                    float* s_row = S + i * dv;
                    const float ki = kt[i];
                    for (size_t j = 0; j < dv; ++j) {
                        s_row[j] += ki * vt[j];
                    }
                }
            }
        }

        // ---- Step 4: Query readout, o_t = scale * q_t^T @ S -> [1, d_v] ----
        //
        // Standard GQA: HeadsPerGroup consecutive query heads share this state.
        // Inverse GQA: HeadsPerGroup is 1 and Query already points at the
        // shared query head for this kv head.
        //
        const float* qt_base = Work->Query + t * Work->QueryTokenStride;
        float* ot_base = Work->Output + t * Work->OutputTokenStride;

        for (size_t g = 0; g < heads_per_group; ++g) {
            const float* qt = qt_base + g * dk;
            float* ot = ot_base + g * dv;

            if (use_sgemm) {
                // Use alpha=1.0 to hit the MLAS M=1 gemv fast path, then scale output.
                MlasSgemmOperation(CblasNoTrans, CblasNoTrans, 1, dv, dk,
                                   1.0f, qt, dk, S, dv, 0.0f, ot, dv);
                if (scale != 1.0f) {
                    for (size_t j = 0; j < dv; ++j) {
                        ot[j] *= scale;
                    }
                }
            } else {
                for (size_t j = 0; j < dv; ++j) {
                    float acc = 0.0f;
                    for (size_t i = 0; i < dk; ++i) {
                        acc += qt[i] * S[i * dv + j];
                    }
                    ot[j] = scale * acc;
                }
            }
        }
    }
}

const MLAS_LINEAR_ATTENTION_DISPATCH MlasLinearAttentionDispatchDefault = {
    MlasLinearAttentionProcessHead
};

//
// Number of independent work items: one per (batch, kv head) pair.
//
static
std::ptrdiff_t
MlasLinearAttentionTaskCount(
    const MlasLinearAttentionArgs* args
)
{
    return static_cast<std::ptrdiff_t>(args->batch_size) *
           static_cast<std::ptrdiff_t>(args->kv_num_heads);
}

static
std::ptrdiff_t
MlasLinearAttentionPartitionCount(
    const MlasLinearAttentionArgs* args
)
{
    const std::ptrdiff_t total_task_count = MlasLinearAttentionTaskCount(args);

    std::ptrdiff_t thread_count = static_cast<std::ptrdiff_t>(args->thread_count);
    if (thread_count > total_task_count) {
        thread_count = total_task_count;
    }
    if (thread_count < 1) {
        thread_count = 1;
    }

    return thread_count;
}

void
MlasLinearAttentionThreaded(
    void* argptr,
    std::ptrdiff_t thread_id
)
{
    const MlasLinearAttentionArgs* args = reinterpret_cast<MlasLinearAttentionArgs*>(argptr);

    const std::ptrdiff_t total_task_count = MlasLinearAttentionTaskCount(args);
    const std::ptrdiff_t thread_count = MlasLinearAttentionPartitionCount(args);

    if (thread_id >= thread_count) {
        return;
    }

    std::ptrdiff_t quotient = total_task_count / thread_count;
    std::ptrdiff_t remainder = total_task_count % thread_count;
    std::ptrdiff_t task_start;
    std::ptrdiff_t task_end;
    if (thread_id < remainder) {
        task_start = (quotient + 1) * thread_id;
        task_end = task_start + quotient + 1;
    } else {
        task_start = quotient * thread_id + remainder;
        task_end = task_start + quotient;
    }

    //
    // Loop invariants. All offsets are computed in size_t: a packed query
    // tensor at B=8, T=8192, H_q=32, d_k=128 already exceeds INT32_MAX.
    //
    const std::ptrdiff_t kv_num_heads = static_cast<std::ptrdiff_t>(args->kv_num_heads);
    const size_t dk = static_cast<size_t>(args->k_head_size);
    const size_t dv = static_cast<size_t>(args->v_head_size);
    const size_t seq_len = static_cast<size_t>(args->sequence_length);
    const size_t state_per_head = dk * dv;

    const size_t q_token_stride = static_cast<size_t>(args->q_num_heads) * dk;
    const size_t k_token_stride = static_cast<size_t>(args->k_num_heads) * dk;
    const size_t v_token_stride = static_cast<size_t>(args->kv_num_heads) * dv;
    const size_t o_token_stride = MlasLinearAttentionOutputHiddenSize(
        args->q_num_heads, args->kv_num_heads, args->v_head_size);

    const std::ptrdiff_t kv_per_k_head = kv_num_heads / static_cast<std::ptrdiff_t>(args->k_num_heads);

    //
    // Standard GQA (H_q >= H_kv) gives each kv head H_q / H_kv consecutive
    // query heads; inverse GQA (H_q < H_kv) gives it a single shared one. Both
    // reduce to "HeadsPerGroup consecutive query heads from h_q0, written to
    // HeadsPerGroup consecutive output heads from h_out0".
    //
    const bool inverse_gqa = (args->q_num_heads < args->kv_num_heads);
    const std::ptrdiff_t heads_per_group =
        inverse_gqa ? 1 : (static_cast<std::ptrdiff_t>(args->q_num_heads) / kv_num_heads);

    size_t decay_token_stride = 0;
    if (args->decay_layout == MlasLinearAttentionDecayPerKeyDim) {
        decay_token_stride = static_cast<size_t>(args->kv_num_heads) * dk;
    } else if (args->decay_layout == MlasLinearAttentionDecayPerHead) {
        decay_token_stride = static_cast<size_t>(args->kv_num_heads);
    }

    size_t beta_token_stride = 0;
    if (args->beta_layout == MlasLinearAttentionBetaPerHead) {
        beta_token_stride = static_cast<size_t>(args->kv_num_heads);
    } else if (args->beta_layout == MlasLinearAttentionBetaShared) {
        beta_token_stride = 1;
    }

    float* scratch = reinterpret_cast<float*>(
        reinterpret_cast<char*>(args->buffer) + thread_id * args->buffer_size_per_thread);

    const MLAS_LINEAR_ATTENTION_DISPATCH* dispatch = GetMlasPlatform().LinearAttentionDispatch;

    for (std::ptrdiff_t task_index = task_start; task_index < task_end; ++task_index) {
        const std::ptrdiff_t b = task_index / kv_num_heads;
        const std::ptrdiff_t h_kv = task_index % kv_num_heads;
        const std::ptrdiff_t h_k = h_kv / kv_per_k_head;

        const std::ptrdiff_t h_q0 =
            inverse_gqa ? (h_kv * static_cast<std::ptrdiff_t>(args->q_num_heads) / kv_num_heads)
                        : (h_kv * heads_per_group);
        const std::ptrdiff_t h_out0 = inverse_gqa ? h_kv : (h_kv * heads_per_group);

        // Token 0 of this batch item.
        const size_t seq_base = static_cast<size_t>(b) * seq_len;

        MLAS_LINEAR_ATTENTION_WORK work;

        work.State = args->state +
                     (static_cast<size_t>(b) * static_cast<size_t>(args->kv_num_heads) +
                      static_cast<size_t>(h_kv)) * state_per_head;
        work.Query = args->query + seq_base * q_token_stride + static_cast<size_t>(h_q0) * dk;
        work.Key = args->key + seq_base * k_token_stride + static_cast<size_t>(h_k) * dk;
        work.Value = args->value + seq_base * v_token_stride + static_cast<size_t>(h_kv) * dv;
        work.Output = args->output + seq_base * o_token_stride + static_cast<size_t>(h_out0) * dv;

        if (args->decay != nullptr && args->decay_layout != MlasLinearAttentionDecayNone) {
            const size_t head_offset =
                (args->decay_layout == MlasLinearAttentionDecayPerKeyDim)
                    ? static_cast<size_t>(h_kv) * dk
                    : static_cast<size_t>(h_kv);
            work.Decay = args->decay + seq_base * decay_token_stride + head_offset;
        } else {
            work.Decay = nullptr;
        }

        if (args->beta != nullptr && args->beta_layout != MlasLinearAttentionBetaNone) {
            const size_t head_offset = (args->beta_layout == MlasLinearAttentionBetaPerHead)
                                           ? static_cast<size_t>(h_kv)
                                           : 0;
            work.Beta = args->beta + seq_base * beta_token_stride + head_offset;
        } else {
            work.Beta = nullptr;
        }

        work.QueryTokenStride = q_token_stride;
        work.KeyTokenStride = k_token_stride;
        work.ValueTokenStride = v_token_stride;
        work.DecayTokenStride = (work.Decay != nullptr) ? decay_token_stride : 0;
        work.BetaTokenStride = (work.Beta != nullptr) ? beta_token_stride : 0;
        work.OutputTokenStride = o_token_stride;

        work.SequenceLength = seq_len;
        work.KHeadSize = dk;
        work.VHeadSize = dv;
        work.HeadsPerGroup = static_cast<size_t>(heads_per_group);
        work.Scale = args->scale;
        work.Rule = args->rule;
        work.DecayLayout = args->decay_layout;
        work.Scratch = scratch;

        if (dispatch != nullptr && dispatch->ProcessHead != nullptr) {
            dispatch->ProcessHead(&work);
        } else {
            MlasLinearAttentionProcessHead(&work);
        }
    }
}

void
MLASCALL
MlasLinearAttention(
    MlasLinearAttentionArgs* args,
    MLAS_THREADPOOL* ThreadPool
)
{
    //
    // Preconditions, documented on MlasLinearAttentionArgs. Like the other MLAS
    // attention entry points this trusts its caller in release builds - the CPU
    // EP validates and reports a Status - but the invariants are asserted here
    // so a mis-filled args struct fails loudly in a debug build rather than
    // dividing by zero or dereferencing null deep inside a kernel.
    //
    assert(args->k_num_heads > 0);
    assert(args->q_num_heads > 0 && args->kv_num_heads > 0);
    assert(args->kv_num_heads % args->k_num_heads == 0);
    assert(args->q_num_heads >= args->kv_num_heads
               ? args->q_num_heads % args->kv_num_heads == 0
               : args->kv_num_heads % args->q_num_heads == 0);
    assert(args->k_head_size > 0 && args->v_head_size > 0);
    assert((args->rule != MlasLinearAttentionRuleGated &&
            args->rule != MlasLinearAttentionRuleGatedDelta) ||
           (args->decay != nullptr && args->decay_layout != MlasLinearAttentionDecayNone));
    assert((args->rule != MlasLinearAttentionRuleDelta &&
            args->rule != MlasLinearAttentionRuleGatedDelta) ||
           (args->beta != nullptr && args->beta_layout != MlasLinearAttentionBetaNone));

    if (args->sequence_length <= 0 || MlasLinearAttentionTaskCount(args) <= 0) {
        return;
    }

    //
    // Clamping the partition count to the task count keeps threads from being
    // scheduled with nothing to do, and lets the common B=1, H_kv=1 decode
    // case take the inline path in MlasExecuteThreaded.
    //
    MlasExecuteThreaded(
        MlasLinearAttentionThreaded,
        static_cast<void*>(args),
        MlasLinearAttentionPartitionCount(args),
        ThreadPool);
}
