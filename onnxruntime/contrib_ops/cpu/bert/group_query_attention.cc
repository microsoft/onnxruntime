// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "contrib_ops/cpu/bert/group_query_attention.h"
#include "contrib_ops/cpu/bert/group_query_attention_helper.h"
#include "contrib_ops/cpu/bert/rotary_helper.h"
#include "contrib_ops/cpu/bert/attention_utils.h"
#include "contrib_ops/cpu/bert/rotary_embedding.h"
#include "contrib_ops/cpu/bert/rotary_embedding_helper.h"

#include "core/framework/tensorprotoutils.h"
#include "core/graph/onnx_protobuf.h"
#include "core/common/safeint.h"
#include "core/platform/threadpool.h"

#include <unsupported/Eigen/SpecialFunctions>
#include <algorithm>
#include <cstring>
#include <optional>
#include <vector>

using onnxruntime::concurrency::ThreadPool;

namespace onnxruntime {
namespace contrib {
namespace {

// ---------------------------------------------------------------------------------------------
// Windowed ("sliding window") KV cache.
//
// The cache holds only the most recent positions of the sequence, at cache indices [0, L) with
// W <= L <= C, where W is the attention window and C the buffer capacity. Both the causal mask and
// the local-window mask depend only on the distance between a query and a key, so translating the
// cache coordinate system leaves them unchanged: running the existing attention paths against
// cache-relative sequence lengths is exact. RoPE is the only position-absolute quantity, and it is
// applied to Q/K from the absolute positions before any of this happens.
//
// Resident entries live at [0, end) and new entries are appended at `end`, so a step only has to
// move memory when the append would run past C. The oldest resident entries are usually already
// older than the window, but the local-window mask hides them, so they can be left alone instead
// of being shifted out one row at a time: compacting reclaims `gap = C - W + 1` rows at once and
// therefore runs once every `gap` decode steps instead of on every step. Slack above the window is
// what buys those free steps; at C == W, gap == 1 and every step compacts.
//
// `end` is a pure function of the absolute past length P, so the op stays stateless across Run()
// calls -- the cache buffer remains the only state carried between steps.
//
// A step whose compaction would drop rows its own earliest query still reads runs against a
// staging cache of `end + S` entries instead, and only the surviving tail is written back.
// ---------------------------------------------------------------------------------------------

// Per-batch row geometry of one windowed step.
struct WindowedStep {
  int seed_offset = 0;   // first resident row read out of the real cache (in-place shift distance)
  int seed_rows = 0;     // resident rows carried into the buffer that attention runs against
  int write_offset = 0;  // first row of the staging buffer written back into the real cache
  int write_rows = 0;    // rows written back (0 when the step ran in place)
};

// Offset one past the last resident row, i.e. where the next entry is appended. Whole `gap`-sized
// blocks are reclaimed at once, which keeps `end` in [C - gap + 1, C] once the cache has filled.
//
// The intermediate math runs in int64_t: `past_sequence_length` is derived from the caller-supplied
// seqlens_k input, so `overflow + gap - 1` and `gap * blocks` would be able to overflow for values
// near INT32_MAX -- signed overflow is UB, so it has to be avoided even for inputs that later
// validation would reject. The result is bounded by `capacity`, so narrowing back to int is safe.
int WindowedCacheEnd(int64_t past_sequence_length, int64_t capacity, int64_t gap) {
  if (past_sequence_length <= capacity) {
    return static_cast<int>(past_sequence_length);  // still filling: nothing has been reclaimed yet
  }
  const int64_t overflow = past_sequence_length - capacity;
  const int64_t reclaimed = gap * ((overflow + gap - 1) / gap);
  return static_cast<int>(past_sequence_length - reclaimed);
}

// Fills `steps` and the cache-relative seqlens_k, and decides whether the step needs a staging
// cache. `seqlens_k[b]` is the absolute total length minus one, i.e. T - 1 = P + S - 1.
void PlanWindowedKvCache(const int32_t* seqlens_k,
                         int batch_size,
                         int sequence_length,
                         int capacity,
                         int local_window_size,
                         std::vector<WindowedStep>& steps,
                         std::vector<int32_t>& cache_seqlens_k,
                         bool& use_staging,
                         int& staged_capacity) {
  steps.assign(batch_size, WindowedStep{});
  cache_seqlens_k.resize(batch_size);

  // CheckInputs validates capacity >= local_window_size, so this is at least 1.
  const int gap = capacity - local_window_size + 1;

  int max_resident = 0;
  use_staging = false;
  for (int b = 0; b < batch_size; ++b) {
    // Computed in int64_t for the same reason as in WindowedCacheEnd: seqlens_k is caller-supplied
    // and `seqlens_k[b] + 1` would overflow at INT32_MAX. Clamping at zero keeps a bogus (too
    // small) seqlens_k from turning into a negative row count downstream.
    const int64_t past_sequence_length =
        std::max<int64_t>(static_cast<int64_t>(seqlens_k[b]) + 1 - sequence_length, 0);  // P
    const int end_before = WindowedCacheEnd(past_sequence_length, capacity, gap);
    const int end_after = WindowedCacheEnd(past_sequence_length + sequence_length, capacity, gap);
    const int kept = end_after - sequence_length;  // resident rows that survive this step

    // The first new row of the step is read by a query that also reaches local_window_size - 1
    // rows further back, so compacting down to `kept` rows in place is only valid while that much
    // history survives (all of it while the sequence is still shorter than the window).
    const int64_t required = std::min<int64_t>(past_sequence_length, local_window_size - 1);
    use_staging = use_staging || kept < required;

    steps[b].seed_rows = end_before;
    steps[b].seed_offset = end_before - kept;  // rows dropped off the front, 0 on a drifting step
    max_resident = std::max(max_resident, end_before);
  }

  staged_capacity = use_staging ? max_resident + sequence_length : capacity;

  for (int b = 0; b < batch_size; ++b) {
    WindowedStep& step = steps[b];
    if (use_staging) {
      // Seed the staging cache with everything resident, then write back only the tail that the
      // post-step layout keeps.
      step.write_offset = step.seed_offset;
      step.write_rows = step.seed_rows + sequence_length - step.seed_offset;
      step.seed_offset = 0;
    } else {
      step.seed_rows -= step.seed_offset;  // becomes `kept`; 0 shift on a drifting step
    }
    cache_seqlens_k[b] = static_cast<int32_t>(step.seed_rows + sequence_length - 1);
  }
}

// Moves whole KV rows per (batch, kv_head) between two BNSH caches whose sequence capacities may
// differ. Rows are copied verbatim, so this is independent of the cache element type and of any
// quantization packing. Each (batch, kv_head) slice is disjoint, so the moves are parallelized;
// this is pure memory traffic, which is why the caller skips it on a drifting step.
void MoveKvCacheRows(const uint8_t* src,
                     uint8_t* dst,
                     int batch_size,
                     int kv_num_heads,
                     int src_capacity,
                     int dst_capacity,
                     size_t row_bytes,
                     const std::vector<WindowedStep>& steps,
                     bool write_back,
                     ThreadPool* thread_pool) {
  int max_rows = 0;
  for (int b = 0; b < batch_size; ++b) {
    max_rows = std::max(max_rows, write_back ? steps[b].write_rows : steps[b].seed_rows);
  }
  if (max_rows <= 0) {
    return;
  }

  const double moved_bytes = static_cast<double>(max_rows) * static_cast<double>(row_bytes);
  const TensorOpCost cost{moved_bytes, moved_bytes, moved_bytes};
  ThreadPool::TryParallelFor(
      thread_pool, static_cast<std::ptrdiff_t>(batch_size) * kv_num_heads, cost,
      [&](std::ptrdiff_t begin, std::ptrdiff_t end) {
        for (std::ptrdiff_t unit = begin; unit < end; ++unit) {
          const int b = static_cast<int>(unit / kv_num_heads);
          const int rows = write_back ? steps[b].write_rows : steps[b].seed_rows;
          if (rows <= 0) {
            continue;
          }
          const int src_offset = write_back ? steps[b].write_offset : steps[b].seed_offset;
          const SafeInt<size_t> head = SafeInt<size_t>(unit);
          const uint8_t* src_rows = src + static_cast<size_t>((head * src_capacity + src_offset) * row_bytes);
          uint8_t* dst_rows = dst + static_cast<size_t>(head * dst_capacity * row_bytes);
          if (src_rows != dst_rows) {
            // memmove, not memcpy: the in-place left shift of the real cache overlaps.
            std::memmove(dst_rows, src_rows, static_cast<size_t>(SafeInt<size_t>(rows) * row_bytes));
          }
        }
      });
}

}  // namespace

// These ops are internal-only, so register outside of onnx
#define REGISTER_KERNEL_TYPED(T)                                              \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                              \
      GroupQueryAttention,                                                    \
      kMSDomain,                                                              \
      1,                                                                      \
      T,                                                                      \
      kCpuExecutionProvider,                                                  \
      KernelDefBuilder()                                                      \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>())              \
          .TypeConstraint("T_CACHE", {DataTypeImpl::GetTensorType<T>(),       \
                                      DataTypeImpl::GetTensorType<uint8_t>(), \
                                      DataTypeImpl::GetTensorType<int8_t>()}) \
          .TypeConstraint("T_KV_SCALE", DataTypeImpl::GetTensorType<float>()) \
          .TypeConstraint("M", DataTypeImpl::GetTensorType<int32_t>()),       \
      GroupQueryAttention<T>);

REGISTER_KERNEL_TYPED(float)
REGISTER_KERNEL_TYPED(MLFloat16)

template <typename T>
GroupQueryAttention<T>::GroupQueryAttention(const OpKernelInfo& info)
    : OpKernel(info), GQAAttentionBase(info, true) {
  sliding_window_cache_ = info.GetAttrOrDefault<int64_t>("sliding_window_cache", 0) == 1;
  ORT_ENFORCE(!sliding_window_cache_ || local_window_size_ > 0,
              "GroupQueryAttention (CPU): sliding_window_cache=1 requires local_window_size > 0.");
}

template <typename T>
Status GroupQueryAttention<T>::Compute(OpKernelContext* context) const {
  const Tensor* query = context->Input<Tensor>(0);
  const Tensor* key = context->Input<Tensor>(1);
  const Tensor* value = context->Input<Tensor>(2);
  const Tensor* past_key = context->Input<Tensor>(3);
  const Tensor* past_value = context->Input<Tensor>(4);
  const Tensor* seqlens_k = context->Input<Tensor>(5);
  const Tensor* total_seqlen_tensor = context->Input<Tensor>(6);
  const Tensor* cos_cache = context->Input<Tensor>(7);
  const Tensor* sin_cache = context->Input<Tensor>(8);
  const Tensor* position_ids = context->Input<Tensor>(9);
  const Tensor* attention_bias = context->Input<Tensor>(10);
  const Tensor* head_sink = context->Input<Tensor>(11);
  const Tensor* k_scale = context->Input<Tensor>(12);
  const Tensor* v_scale = context->Input<Tensor>(13);

  // Validate quantization configuration.
  if (kv_quant_enabled_) {
    ORT_RETURN_IF(k_quant_type_ != v_quant_type_,
                  "CPU GroupQueryAttention requires k_quant_type == v_quant_type, got different types");
    ORT_RETURN_IF(kv_cache_bit_width_ != 4 && kv_cache_bit_width_ != 8,
                  "kv_cache_bit_width must be 4 or 8 when quantization is enabled, got ", kv_cache_bit_width_);
    ORT_RETURN_IF(k_scale == nullptr,
                  "k_scale must be provided when k_quant_type is not NONE");
    ORT_RETURN_IF(v_scale == nullptr,
                  "v_scale must be provided when v_quant_type is not NONE");
    ORT_RETURN_IF(k_scale->DataType() != DataTypeImpl::GetType<float>(),
                  "k_scale must be float tensor");
    ORT_RETURN_IF(v_scale->DataType() != DataTypeImpl::GetType<float>(),
                  "v_scale must be float tensor");
  } else {
    ORT_RETURN_IF(kv_cache_bit_width_ != 0,
                  "kv_cache_bit_width must be 0 when quantization is disabled, got ", kv_cache_bit_width_);
  }

  // q_norm_weight (input 14) / k_norm_weight (input 15) are populated by the CUDA/WebGPU
  // GroupQueryAttentionPreNormFusion optimizer pass. The CPU kernel does not implement
  // the fused per-head Q/K RMS normalization prologue, so reject the node if either input
  // is present rather than silently dropping the normalization.
  if ((context->InputCount() > 14 && context->Input<Tensor>(14) != nullptr) ||
      (context->InputCount() > 15 && context->Input<Tensor>(15) != nullptr)) {
    return ORT_MAKE_STATUS(
        ONNXRUNTIME, INVALID_ARGUMENT,
        "GroupQueryAttention (CPU): q_norm_weight / k_norm_weight inputs are not supported. "
        "The per-head Q/K RMS normalization prologue is implemented only on the CUDA and WebGPU EPs.");
  }

  GroupQueryAttentionParameters parameters = {};
  ORT_RETURN_IF_ERROR(group_query_attention_helper::CheckInputs(query,
                                                                key,
                                                                value,
                                                                past_key,
                                                                past_value,
                                                                cos_cache,
                                                                sin_cache,
                                                                &parameters,
                                                                num_heads_,
                                                                kv_num_heads_,
                                                                seqlens_k,
                                                                total_seqlen_tensor,
                                                                scale_,
                                                                softcap_,
                                                                kv_cache_bit_width_,
                                                                /*max_threads_per_block=*/0,
                                                                /*kv_cache_extra_bits=*/0,
                                                                sliding_window_cache_,
                                                                local_window_size_));

  ORT_RETURN_IF_ERROR(group_query_attention_helper::CheckCustomAttentionInputs(position_ids,
                                                                               attention_bias,
                                                                               head_sink,
                                                                               parameters));

  // Populate quantization fields in parameters.
  parameters.k_quant_type = k_quant_type_;
  parameters.v_quant_type = v_quant_type_;
  parameters.kv_cache_bit_width = kv_cache_bit_width_;

  const int batch_size = parameters.batch_size;
  const int sequence_length = parameters.sequence_length;
  const int present_kv_seqlen = parameters.seqlen_present_kv_cache;
  int head_size = parameters.head_size;

  // Validate scale tensor shapes after CheckInputs (which validates query rank).
  if (kv_quant_enabled_) {
    const bool per_channel = (k_quant_type_ == KVQuantizationType::PER_CHANNEL);
    const int64_t expected_scale_size = per_channel
                                            ? static_cast<int64_t>(kv_num_heads_) * head_size
                                            : 1;
    ORT_RETURN_IF(k_scale->Shape().Size() != expected_scale_size,
                  "k_scale shape mismatch: expected ", expected_scale_size,
                  " elements, got ", k_scale->Shape().Size());
    ORT_RETURN_IF(v_scale->Shape().Size() != expected_scale_size,
                  "v_scale shape mismatch: expected ", expected_scale_size,
                  " elements, got ", v_scale->Shape().Size());
  }

  // Validate seqlens_k values before they are used as GEMM dimensions to prevent OOB access.
  {
    const int32_t* seqlens_k_data = seqlens_k->Data<int32_t>();
    const int past_kv_seqlen = parameters.seqlen_past_kv_cache;
    const bool windowed = parameters.is_windowed_kv_cache;
    for (int b = 0; b < batch_size; b++) {
      // A windowed cache is deliberately shorter than the sequence it serves, so seqlens_k (which
      // stays absolute) is not bounded by the buffer. The resident row count is clamped to the
      // capacity instead, when the step is planned.
      if (seqlens_k_data[b] < 0 || (!windowed && seqlens_k_data[b] >= present_kv_seqlen)) {
        return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                               "seqlens_k[", b, "] = ", seqlens_k_data[b],
                               " is out of range [0, ", present_kv_seqlen, ")");
      }
      if ((windowed || !parameters.is_first_prompt) &&
          static_cast<int64_t>(seqlens_k_data[b]) + 1 < sequence_length) {
        return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                               "seqlens_k[", b, "] = ", seqlens_k_data[b],
                               " is too small for sequence_length ", sequence_length);
      }
      // Bound the number of past KV rows copied out of the past key/value buffers during
      // token generation (decode). ConcatStateChunkGQA copies (seqlens_k + 1 - sequence_length)
      // rows from the past buffer (sized past_kv_seqlen). The present-buffer check above does
      // not bound this past-side read, because the present buffer can be larger than the past
      // buffer when total_sequence_length exceeds the past sequence length. A large seqlens_k
      // combined with a small past buffer would otherwise read past the end of the past tensors.
      // Shared KV (kv_sequence_length == 0) appends no new KV and its past read is already
      // bounded by the present-buffer check together with the total_sequence_length <=
      // seqlen_past_kv_cache enforcement in the apply-attention paths, so it needs no check here.
      if (!windowed && past_key != nullptr && past_value != nullptr && parameters.kv_sequence_length != 0 &&
          !parameters.is_first_prompt) {
        const int64_t past_rows = static_cast<int64_t>(seqlens_k_data[b]) + 1 - sequence_length;
        if (past_rows > past_kv_seqlen) {
          return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                                 "seqlens_k[", b, "] = ", seqlens_k_data[b], " requires ", past_rows,
                                 " past KV rows, which exceeds the past buffer sequence length ",
                                 past_kv_seqlen, ".");
        }
      }
    }
  }
  int q_hidden_size = parameters.hidden_size;
  const bool packed_qkv = parameters.is_packed_qkv;

  std::vector<int64_t> output_shape(3);
  output_shape[0] = static_cast<int64_t>(batch_size);
  output_shape[1] = static_cast<int64_t>(sequence_length);
  output_shape[2] = static_cast<int64_t>(q_hidden_size);
  Tensor* output = context->Output(0, output_shape);

  const int packed_head_size = (kv_cache_bit_width_ == 4) ? ((head_size + 1) / 2) : head_size;
  std::vector<int64_t> present_k_shape({static_cast<int64_t>(batch_size), static_cast<int64_t>(kv_num_heads_), static_cast<int64_t>(present_kv_seqlen), static_cast<int64_t>(packed_head_size)});
  std::vector<int64_t> present_v_shape({static_cast<int64_t>(batch_size), static_cast<int64_t>(kv_num_heads_), static_cast<int64_t>(present_kv_seqlen), static_cast<int64_t>(packed_head_size)});
  Tensor* present_k = context->Output(1, present_k_shape);
  Tensor* present_v = context->Output(2, present_v_shape);

  std::vector<int64_t> output_qk_shape{static_cast<int64_t>(batch_size), static_cast<int64_t>(num_heads_), static_cast<int64_t>(parameters.sequence_length), static_cast<int64_t>(parameters.total_sequence_length)};
  Tensor* output_qk = context->Output(3, output_qk_shape);

  ORT_RETURN_IF_ERROR(group_query_attention_helper::CheckOutputs(output_qk, qk_output_));

  // The QK output is laid out over the absolute total_sequence_length, while a windowed cache
  // computes scores over cache-relative positions only. Reject the combination rather than
  // emitting a buffer whose columns do not mean what the shape says.
  if (parameters.is_windowed_kv_cache && output_qk != nullptr) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, NOT_IMPLEMENTED,
                           "GroupQueryAttention (CPU): qk_output is not implemented with sliding_window_cache=1.");
  }

  AllocatorPtr allocator;
  ORT_RETURN_IF_ERROR(context->GetTempSpaceAllocator(&allocator));

  auto element_type = DataTypeImpl::GetType<T>();
  OrtValue Q;
  OrtValue K;
  OrtValue V;
  const int kv_sequence_length = parameters.kv_sequence_length;
  if (packed_qkv) {
    ORT_RETURN_IF_ERROR(MaybeTransposeToBNSH<T>(
        allocator, batch_size, num_heads_ + 2 * kv_num_heads_, sequence_length, head_size, query, Q));
  } else {
    ORT_RETURN_IF_ERROR(MaybeTransposeToBNSH<T>(
        allocator, batch_size, num_heads_, sequence_length, head_size, query, Q));
    ORT_RETURN_IF_ERROR(MaybeTransposeToBNSH<T>(
        allocator, batch_size, kv_num_heads_, kv_sequence_length, head_size, key, K));
    ORT_RETURN_IF_ERROR(MaybeTransposeToBNSH<T>(
        allocator, batch_size, kv_num_heads_, kv_sequence_length, head_size, value, V));
  }

  OrtValue RotaryQKV;
  OrtValue RotaryQ;
  OrtValue RotaryK;
  T* q_rotary = Q.GetMutable<Tensor>()->MutableData<T>();
  T* k_rotary = packed_qkv ? nullptr : K.GetMutable<Tensor>()->MutableData<T>();
  if (do_rotary_) {
    // When kv_sequence_length == 0 (shared KV), only Q needs RoPE — K is skipped below.
    ORT_ENFORCE(cos_cache != nullptr && sin_cache != nullptr, "cos_cache and sin_cache must be provided when do_rotary is true");
    // Validation of seqlens_k against rotary cache size is performed in CheckInputs()
    // when seqlens_k is on CPU. GPU EPs where seqlens_k resides on device rely on
    // RunRotaryEmbedding's position_ids validation for OOB protection.

    // Initialize rotary parameters
    rotary_embedding_helper::RotaryParameters rotary_params = {};
    rotary_params.batch_size = batch_size;
    rotary_params.sequence_length = sequence_length;
    rotary_params.hidden_size = q_hidden_size;
    rotary_params.head_size = head_size;
    rotary_params.rotary_embedding_dim = parameters.rotary_dim;
    rotary_params.num_heads = num_heads_;
    rotary_params.max_sequence_length = static_cast<int>(cos_cache->Shape().GetDims()[0]);
    rotary_params.seq_stride = head_size;
    rotary_params.head_stride = sequence_length * rotary_params.seq_stride;
    rotary_params.batch_stride = (packed_qkv ? (num_heads_ + 2 * kv_num_heads_) : num_heads_) * rotary_params.head_stride;
    rotary_params.position_ids_format = !parameters.is_first_prompt ? 1 : 0;
    rotary_params.transposed = true;
    auto* tp = context->GetOperatorThreadPool();
    // Generate position ids
    const int pos_ids_size = parameters.is_first_prompt ? 1 : batch_size * sequence_length;
    std::vector<int64_t> default_pos_ids(pos_ids_size);
    const int64_t* pos_ids_data = default_pos_ids.data();

    if (position_ids != nullptr) {
      pos_ids_data = position_ids->Data<int64_t>();
    } else if (parameters.is_first_prompt) {
      default_pos_ids[0] = static_cast<int64_t>(0);
    } else {
      // Note: As of now, continuous decoding supports only batch size 1 and token generation supports only sequence length 1.
      for (int b = 0; b < batch_size; b++) {
        const int total_seqlen = seqlens_k->Data<int32_t>()[b] + 1;
        const int past_seqlen = total_seqlen - sequence_length;
        for (int s = 0; s < sequence_length; s++) {
          if (past_seqlen + s < total_seqlen) {
            default_pos_ids[b * sequence_length + s] = static_cast<int64_t>(past_seqlen) + s;
          } else {
            default_pos_ids[b * sequence_length + s] = static_cast<int64_t>(1);
          }
        }
      }
    }

    // Initialize separate buffers for rotary embeddings
    const T* q_input;
    const T* k_input;
    if (packed_qkv) {
      Tensor::InitOrtValue(element_type, TensorShape({batch_size, num_heads_ + 2 * kv_num_heads_, sequence_length, head_size}), allocator, RotaryQKV);
      q_input = Q.Get<Tensor>().Data<T>();
      k_input = q_input + num_heads_ * sequence_length * head_size;
      q_rotary = RotaryQKV.GetMutable<Tensor>()->MutableData<T>();
      k_rotary = q_rotary + num_heads_ * sequence_length * head_size;
    } else {
      Tensor::InitOrtValue(element_type, TensorShape({batch_size, num_heads_, sequence_length, head_size}), allocator, RotaryQ);
      Tensor::InitOrtValue(element_type, TensorShape({batch_size, kv_num_heads_, sequence_length, head_size}), allocator, RotaryK);
      q_input = Q.Get<Tensor>().Data<T>();
      k_input = K.Get<Tensor>().Data<T>();
      q_rotary = RotaryQ.GetMutable<Tensor>()->MutableData<T>();
      k_rotary = RotaryK.GetMutable<Tensor>()->MutableData<T>();
    }
    // Run rotary embedding for Q
    ORT_RETURN_IF_ERROR(RunRotaryEmbedding<T>(tp, rotary_params, q_input,
                                              pos_ids_data, cos_cache->Data<T>(),
                                              sin_cache->Data<T>(), q_rotary, rotary_interleaved_));

    // Run rotary embedding for K (skip when kv_sequence_length == 0, i.e. shared KV with no new tokens)
    if (kv_sequence_length > 0) {
      rotary_params.num_heads = kv_num_heads_;
      rotary_params.hidden_size = parameters.kv_hidden_size;
      if (!packed_qkv) {
        rotary_params.batch_stride = kv_num_heads_ * rotary_params.head_stride;
      }
      ORT_RETURN_IF_ERROR(RunRotaryEmbedding<T>(tp, rotary_params, k_input,
                                                pos_ids_data, cos_cache->Data<T>(),
                                                sin_cache->Data<T>(), k_rotary, rotary_interleaved_));
    }
    // Pack V into rotary QKV buffer
    if (packed_qkv) {
      const T* v_input = k_input + kv_num_heads_ * sequence_length * head_size;
      T* v_rotary = k_rotary + kv_num_heads_ * sequence_length * head_size;
      ORT_RETURN_IF_ERROR(rotary_helper::PackVIntoRotaryQKV<T>(tp,
                                                               parameters.batch_size,
                                                               parameters.sequence_length,
                                                               parameters.num_heads,
                                                               parameters.kv_num_heads,
                                                               parameters.head_size,
                                                               v_input,
                                                               v_rotary));
    }
  }

  ORT_RETURN_IF_ERROR(context->GetTempSpaceAllocator(&allocator));

  const T* head_sink_data = (head_sink != nullptr) ? head_sink->Data<T>() : nullptr;

  // ---- Windowed KV cache: switch to cache-relative coordinates ----
  // RoPE above is the last consumer of absolute positions, so everything from here on can run
  // against a cache that holds only the most recent min(T, C) entries. The attention paths derive
  // the causal offset and the mask from seqlens_k alone, so redirecting them at a cache-relative
  // seqlens_k (and, when the step would evict entries it still needs, at a staging cache) reuses
  // them unchanged.
  const Tensor* attention_past_key = past_key;
  const Tensor* attention_past_value = past_value;
  Tensor* attention_present_key = present_k;
  Tensor* attention_present_value = present_v;
  const Tensor* attention_seqlens_k = seqlens_k;

  std::vector<WindowedStep> windowed_steps;
  std::vector<int32_t> windowed_cache_seqlens;
  std::optional<Tensor> windowed_seqlens_tensor;
  std::optional<Tensor> staged_key_tensor;
  std::optional<Tensor> staged_value_tensor;
  BufferUniquePtr staged_key_buffer;
  BufferUniquePtr staged_value_buffer;
  bool windowed_use_staging = false;
  int windowed_capacity = 0;
  int windowed_staged_capacity = 0;
  size_t windowed_row_bytes = 0;

  ThreadPool* windowed_thread_pool = context->GetOperatorThreadPool();

  if (parameters.is_windowed_kv_cache) {
    ORT_RETURN_IF(present_k == nullptr || present_v == nullptr,
                  "sliding_window_cache=1 requires the present_key and present_value outputs.");
    windowed_capacity = parameters.kv_cache_capacity;
    PlanWindowedKvCache(seqlens_k->Data<int32_t>(), batch_size, sequence_length, windowed_capacity,
                        local_window_size_, windowed_steps, windowed_cache_seqlens, windowed_use_staging,
                        windowed_staged_capacity);

    // Rows are moved verbatim, so the row size is taken from the cache tensor itself and covers
    // fp32/fp16 as well as the int8 and (nibble-packed) int4 quantized layouts.
    windowed_row_bytes = SafeInt<size_t>(past_key->Shape().GetDims()[3]) * past_key->DataType()->Size();
    const uint8_t* past_key_bytes = static_cast<const uint8_t*>(past_key->DataRaw());
    const uint8_t* past_value_bytes = static_cast<const uint8_t*>(past_value->DataRaw());

    if (windowed_use_staging) {
      const size_t staged_bytes = SafeInt<size_t>(batch_size) * kv_num_heads_ *
                                  windowed_staged_capacity * windowed_row_bytes;
      staged_key_buffer = BufferUniquePtr(allocator->Alloc(staged_bytes), BufferDeleter(allocator));
      staged_value_buffer = BufferUniquePtr(allocator->Alloc(staged_bytes), BufferDeleter(allocator));
      ORT_RETURN_IF(staged_key_buffer.get() == nullptr || staged_value_buffer.get() == nullptr,
                    "Failed to allocate the sliding_window_cache staging buffers.");

      const TensorShape staged_shape({static_cast<int64_t>(batch_size), static_cast<int64_t>(kv_num_heads_),
                                      static_cast<int64_t>(windowed_staged_capacity),
                                      past_key->Shape().GetDims()[3]});
      staged_key_tensor.emplace(past_key->DataType(), staged_shape, staged_key_buffer.get(), allocator->Info());
      staged_value_tensor.emplace(past_value->DataType(), staged_shape, staged_value_buffer.get(), allocator->Info());

      MoveKvCacheRows(past_key_bytes, static_cast<uint8_t*>(staged_key_buffer.get()), batch_size, kv_num_heads_,
                      windowed_capacity, windowed_staged_capacity, windowed_row_bytes, windowed_steps,
                      /*write_back=*/false, windowed_thread_pool);
      MoveKvCacheRows(past_value_bytes, static_cast<uint8_t*>(staged_value_buffer.get()), batch_size, kv_num_heads_,
                      windowed_capacity, windowed_staged_capacity, windowed_row_bytes, windowed_steps,
                      /*write_back=*/false, windowed_thread_pool);

      attention_past_key = &staged_key_tensor.value();
      attention_past_value = &staged_value_tensor.value();
      attention_present_key = &staged_key_tensor.value();
      attention_present_value = &staged_value_tensor.value();
    } else {
      // Compact in place: the surviving window moves to the front of the real cache, which then
      // doubles as the past buffer so the append below does not copy it a second time.
      uint8_t* present_key_bytes = static_cast<uint8_t*>(present_k->MutableDataRaw());
      uint8_t* present_value_bytes = static_cast<uint8_t*>(present_v->MutableDataRaw());

      // A drifting step appends inside the buffer and reclaims nothing, so it moves no memory at
      // all -- this is the common decode step. The copy is still needed when past and present are
      // separate buffers, since the append below only writes the new rows.
      const bool compacts = std::any_of(windowed_steps.begin(), windowed_steps.end(),
                                        [](const WindowedStep& step) { return step.seed_offset > 0; });
      if (compacts || past_key_bytes != present_key_bytes || past_value_bytes != present_value_bytes) {
        MoveKvCacheRows(past_key_bytes, present_key_bytes, batch_size, kv_num_heads_,
                        windowed_capacity, windowed_capacity, windowed_row_bytes, windowed_steps,
                        /*write_back=*/false, windowed_thread_pool);
        MoveKvCacheRows(past_value_bytes, present_value_bytes, batch_size, kv_num_heads_,
                        windowed_capacity, windowed_capacity, windowed_row_bytes, windowed_steps,
                        /*write_back=*/false, windowed_thread_pool);
      }

      attention_past_key = present_k;
      attention_past_value = present_v;
    }

    windowed_seqlens_tensor.emplace(DataTypeImpl::GetType<int32_t>(),
                                    TensorShape({static_cast<int64_t>(batch_size)}),
                                    windowed_cache_seqlens.data(), allocator->Info());
    attention_seqlens_k = &windowed_seqlens_tensor.value();

    // Cache-relative lengths always have a non-empty past (possibly of length zero), so the
    // first-prompt shortcut must be off: past_seqlen = total_seqlen - sequence_length already
    // yields 0 when nothing is resident.
    parameters.is_first_prompt = false;
    parameters.is_subsequent_prompt = false;
    parameters.seqlen_past_kv_cache = windowed_staged_capacity;
    parameters.seqlen_present_kv_cache = windowed_staged_capacity;
    parameters.kv_cache_capacity = windowed_staged_capacity;
    parameters.total_sequence_length =
        *std::max_element(windowed_cache_seqlens.begin(), windowed_cache_seqlens.end()) + 1;
  }

  auto run_attention = [&]() -> Status {
    // Quantized KV cache path: quantize-on-write + direct MLAS QK/SV.
    if (kv_quant_enabled_) {
      const T* k_data_q = packed_qkv ? nullptr : k_rotary;
      const T* v_data_q = packed_qkv ? nullptr : V.Get<Tensor>().Data<T>();
      auto mlas_quant_type = ToMlasKVQuantType(k_quant_type_, kv_cache_bit_width_);

      // Use flash attention path when:
      // 1. Total sequence length is long enough to benefit from tiling
      // 2. No features that flash path doesn't support (softcap, smooth softmax, output_qk)
      const bool use_flash = !disable_gqa_flash_ &&
                             parameters.total_sequence_length > 1 &&
                             softcap_ == 0.0f &&
                             !use_smooth_softmax_ &&
                             head_sink_data == nullptr &&
                             output_qk == nullptr;

      if (use_flash) {
        return ApplyAttentionQuantizedFlash(
            q_rotary, k_data_q, v_data_q,
            attention_bias,
            attention_past_key, attention_past_value,
            output, attention_present_key, attention_present_value, attention_seqlens_k,
            k_scale->Data<float>(), v_scale->Data<float>(),
            mlas_quant_type, parameters, allocator, context);
      }

      return ApplyAttentionQuantized(
          q_rotary, k_data_q, v_data_q, head_sink_data,
          attention_bias, attention_past_key, attention_past_value,
          output, attention_present_key, attention_present_value, output_qk, attention_seqlens_k,
          k_scale->Data<float>(), v_scale->Data<float>(),
          mlas_quant_type, parameters, allocator, context);
    }

    // Compute the attention score and apply the score to V
    const T* k_data = packed_qkv ? nullptr : k_rotary;
    const T* v_data = packed_qkv ? nullptr : V.Get<Tensor>().Data<T>();

    // Non-quantized flash attention path (float only). Uses the tiled online-softmax
    // kernel to avoid materializing the full attention score matrix. Falls back to the
    // naive path when an unsupported feature is requested (softcap, smooth softmax,
    // head sink, or QK output).
    //
    // Prefill (sequence_length > 1) uses the tiled kernel; single-token decode
    // (sequence_length == 1 with total_sequence_length > 1) uses the dedicated GEMV
    // decode kernel. Both are reached when total_sequence_length > 1.
    if constexpr (std::is_same_v<T, float>) {
      const bool use_flash = !disable_gqa_flash_ &&
                             parameters.total_sequence_length > 1 &&
                             softcap_ == 0.0f &&
                             !use_smooth_softmax_ &&
                             head_sink_data == nullptr &&
                             output_qk == nullptr &&
                             attention_present_key != nullptr && attention_present_value != nullptr;
      if (use_flash) {
        return ApplyAttentionFlash(q_rotary, k_data, v_data,
                                   attention_bias, attention_past_key, attention_past_value,
                                   output, attention_present_key, attention_present_value, attention_seqlens_k,
                                   parameters, allocator, context);
      }
    }

    return ApplyAttention(q_rotary, k_data, v_data,
                          head_sink_data, attention_bias, attention_past_key, attention_past_value, output,
                          attention_present_key, attention_present_value,
                          output_qk, attention_seqlens_k, parameters, allocator, context);
  };

  ORT_RETURN_IF_ERROR(run_attention());

  if (windowed_use_staging) {
    // Attention ran against the staging cache; keep only the entries the sliding window can still
    // reach. Everything before write_offset is older than the oldest key any future query can see.
    MoveKvCacheRows(static_cast<const uint8_t*>(staged_key_buffer.get()),
                    static_cast<uint8_t*>(present_k->MutableDataRaw()), batch_size, kv_num_heads_,
                    windowed_staged_capacity, windowed_capacity, windowed_row_bytes, windowed_steps,
                    /*write_back=*/true, windowed_thread_pool);
    MoveKvCacheRows(static_cast<const uint8_t*>(staged_value_buffer.get()),
                    static_cast<uint8_t*>(present_v->MutableDataRaw()), batch_size, kv_num_heads_,
                    windowed_staged_capacity, windowed_capacity, windowed_row_bytes, windowed_steps,
                    /*write_back=*/true, windowed_thread_pool);
  }

  return Status::OK();
}
}  // namespace contrib
}  // namespace onnxruntime
