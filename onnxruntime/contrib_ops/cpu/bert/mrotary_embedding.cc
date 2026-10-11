// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "contrib_ops/cpu/bert/mrotary_embedding.h"
#include "contrib_ops/cpu/bert/mrotary_embedding_helper.h"
#include "core/providers/cpu/llm/rotary_embedding_int32_utils.h"

#include <algorithm>
#include <limits>
#include <vector>

#include "core/mlas/inc/mlas.h"
#include "core/platform/threadpool.h"

using onnxruntime::concurrency::ThreadPool;
using namespace onnxruntime::contrib::mrotary_embedding_helper;

namespace onnxruntime {
namespace contrib {

// These ops are internal-only, so register outside of onnx
#define REGISTER_KERNEL_TYPED(T)                                        \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                        \
      MRotaryEmbedding,                                                 \
      kMSDomain,                                                        \
      1,                                                                \
      T,                                                                \
      kCpuExecutionProvider,                                            \
      KernelDefBuilder()                                                \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>())        \
          .TypeConstraint("M", DataTypeImpl::GetTensorType<int64_t>()), \
      MRotaryEmbedding<T>);

REGISTER_KERNEL_TYPED(float)
REGISTER_KERNEL_TYPED(MLFloat16)

template <typename T>
MRotaryEmbedding<T>::MRotaryEmbedding(const OpKernelInfo& info) : OpKernel(info) {
  scale = info.GetAttrOrDefault<float>("scale", 1.0);
  const int64_t rotary_embedding_dim_attr = info.GetAttrOrDefault<int64_t>("rotary_embedding_dim", 0);
  const int64_t num_heads_attr = info.GetAttrOrDefault<int64_t>("num_heads", 0);
  ORT_ENFORCE(rotary_embedding_dim_attr >= 0 && rotary_embedding_dim_attr <= std::numeric_limits<int>::max(),
              "rotary_embedding_dim must be in range [0, ", std::numeric_limits<int>::max(),
              "]. Actual value: ", rotary_embedding_dim_attr);
  ORT_ENFORCE(num_heads_attr >= 0 && num_heads_attr <= std::numeric_limits<int>::max(),
              "num_heads must be in range [0, ", std::numeric_limits<int>::max(),
              "]. Actual value: ", num_heads_attr);
  rotary_embedding_dim = static_cast<int>(rotary_embedding_dim_attr);
  num_heads = static_cast<int>(num_heads_attr);
  interleaved = (info.GetAttrOrDefault<int64_t>("interleaved", 0) == 1);
  is_packed_batching = (info.GetAttrOrDefault<int64_t>("is_packed_batching", 0) == 1);
  mrope_layout = info.GetAttrOrDefault<int64_t>("mrope_layout", 0);
  ORT_ENFORCE(info.GetAttrs<int64_t>("mrope_section", mrope_section).IsOK(),
              "MRotaryEmbedding: 'mrope_section' attribute is required");

  if (rotary_embedding_dim > 0) {
    ORT_ENFORCE(num_heads > 0, "num_heads must be provided if rotary_embedding_dim is specified");
  }
}

template <typename T>
static Status RunMRotaryEmbedding(concurrency::ThreadPool* tp, const MRotaryParameters& parameters,
                                  const T* input, const int64_t* position_ids, const T* cos_cache,
                                  const T* sin_cache, T* output, bool interleaved, float scale) {
  const int batch_size = parameters.batch_size;
  const int sequence_length = parameters.sequence_length;
  const int n_heads = parameters.num_heads;
  const int head_size = parameters.head_size;
  const int head_stride = parameters.head_stride;
  const int seq_stride = parameters.seq_stride;
  const int batch_stride = parameters.batch_stride;
  const int max_sequence_length = parameters.max_sequence_length;
  const int rotary_emb_dim = parameters.rotary_embedding_dim;
  const int half_rotary_emb_dim = rotary_emb_dim / 2;

  // position_ids is (3, batch_size, sequence_length): T, H, W streams.
  const std::ptrdiff_t stream_stride = static_cast<std::ptrdiff_t>(batch_size) * sequence_length;

  // Validate position_ids values are within cos/sin cache bounds.
  for (int stream = 0; stream < 3; ++stream) {
    const int64_t* stream_ids = position_ids + static_cast<std::ptrdiff_t>(stream) * stream_stride;
    for (std::ptrdiff_t i = 0; i < stream_stride; ++i) {
      int64_t pos = stream_ids[i];
      if (pos < 0 || pos >= static_cast<int64_t>(max_sequence_length)) {
        return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                               "MRotaryEmbedding: position_ids value ", pos, " at stream ", stream,
                               ", index ", i, " is out of range [0, ", max_sequence_length, ")");
      }
    }
  }

  // Precompute which (T=0, H=1, W=2) position stream contributes each cos/sin cache column.
  std::vector<int8_t> dim_assignment;
  ComputeDimAssignments(parameters.mrope_section, parameters.mrope_layout, half_rotary_emb_dim, dim_assignment);

  std::ptrdiff_t loop_len = 0;
  ORT_RETURN_IF_ERROR(onnxruntime::rotary_embedding_int32_utils::CheckedMulToPtrdiff(
      batch_size, sequence_length, n_heads, "total_elements", loop_len));

  const double cost = static_cast<double>(head_size * sizeof(T) * 2 + rotary_emb_dim * 32);
  ThreadPool::TryParallelFor(tp, loop_len, cost, [&](std::ptrdiff_t begin, std::ptrdiff_t end) {
    // Per-token scratch buffer for the combined cos/sin vector (size half_rotary_emb_dim).
    std::vector<T> cos_buffer(static_cast<size_t>(half_rotary_emb_dim));
    std::vector<T> sin_buffer(static_cast<size_t>(half_rotary_emb_dim));

    for (std::ptrdiff_t ptr = begin; ptr != end; ++ptr) {
      const int b = static_cast<int>((ptr / n_heads) / sequence_length);
      const int s = static_cast<int>((ptr / n_heads) % sequence_length);
      const int n = static_cast<int>(ptr % n_heads);

      const std::ptrdiff_t block_offset = static_cast<std::ptrdiff_t>(b) * batch_stride +
                                          static_cast<std::ptrdiff_t>(s) * seq_stride +
                                          static_cast<std::ptrdiff_t>(n) * head_stride;

      const T* input_data = input + block_offset;
      T* output_data = output + block_offset;

      // Gather the position id for each of the 3 streams for this (b, s) token.
      const std::ptrdiff_t bs_index = static_cast<std::ptrdiff_t>(b) * sequence_length + s;
      int64_t stream_positions[3];
      const T* stream_cos[3];
      const T* stream_sin[3];
      for (int stream = 0; stream < 3; ++stream) {
        stream_positions[stream] = position_ids[static_cast<std::ptrdiff_t>(stream) * stream_stride + bs_index];
        const std::ptrdiff_t cache_offset = static_cast<std::ptrdiff_t>(stream_positions[stream]) *
                                            half_rotary_emb_dim;
        stream_cos[stream] = cos_cache + cache_offset;
        stream_sin[stream] = sin_cache + cache_offset;
      }

      // Build the combined per-token cos/sin vectors by selecting, for each column,
      // the value from the (T/H/W) stream that owns that column per mrope_section/mrope_layout.
      for (int col = 0; col < half_rotary_emb_dim; ++col) {
        const int stream = dim_assignment[static_cast<size_t>(col)];
        cos_buffer[static_cast<size_t>(col)] = static_cast<T>(static_cast<float>(stream_cos[stream][col]) * scale);
        sin_buffer[static_cast<size_t>(col)] = static_cast<T>(static_cast<float>(stream_sin[stream][col]) * scale);
      }

      MlasRotaryEmbedOneRow<T>(input_data, sin_buffer.data(), cos_buffer.data(), rotary_emb_dim, interleaved,
                               output_data);

      if (rotary_emb_dim < head_size) {
        std::memcpy(output_data + rotary_emb_dim,
                    input_data + rotary_emb_dim,
                    (head_size - rotary_emb_dim) * sizeof(T));
      }
    }
  });

  return Status::OK();
}

template <typename T>
Status MRotaryEmbedding<T>::Compute(OpKernelContext* context) const {
  const Tensor* input = context->Input<Tensor>(0);
  const Tensor* position_ids = context->Input<Tensor>(1);
  const Tensor* cos_cache = context->Input<Tensor>(2);
  const Tensor* sin_cache = context->Input<Tensor>(3);

  MRotaryParameters parameters = {};
  ORT_RETURN_IF_ERROR(CheckInputs<Tensor>(input,
                                          position_ids,
                                          cos_cache,
                                          sin_cache,
                                          num_heads,
                                          rotary_embedding_dim,
                                          mrope_section,
                                          mrope_layout,
                                          &parameters));

  Tensor* output = context->Output(0, input->Shape());

  if (is_packed_batching == false && parameters.sequence_length > parameters.max_sequence_length) {
    ORT_NOT_IMPLEMENTED("Updating cos_cache and sin_cache in MRotaryEmbedding is not currently supported");
  }

  const T* input_src = input->Data<T>();
  const int64_t* pos_ids_data = position_ids->Data<int64_t>();
  const T* cos_cache_data = cos_cache->Data<T>();
  const T* sin_cache_data = sin_cache->Data<T>();
  T* output_dest = output->MutableData<T>();

  auto* tp = context->GetOperatorThreadPool();

  return RunMRotaryEmbedding<T>(tp, parameters, input_src, pos_ids_data, cos_cache_data, sin_cache_data, output_dest,
                                interleaved, scale);
}

}  // namespace contrib
}  // namespace onnxruntime
