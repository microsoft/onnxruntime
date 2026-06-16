// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cpu/llm/rotary_embedding.h"
#include "core/providers/cpu/llm/rotary_embedding_helper.h"
#include "core/providers/cpu/llm/rotary_embedding_int32_utils.h"

#include <algorithm>
#include <limits>

#include "core/mlas/inc/mlas.h"
#include "core/platform/threadpool.h"

using onnxruntime::concurrency::ThreadPool;
using namespace onnxruntime::rotary_embedding_helper;
using onnxruntime::rotary_embedding_int32_utils::CheckedAddToPtrdiff;

using onnxruntime::rotary_embedding_int32_utils::CheckedPtrdiffMulToPtrdiff;

namespace onnxruntime {

#define REGISTER_ONNX_KERNEL_TYPED(T)                                   \
  ONNX_CPU_OPERATOR_TYPED_KERNEL(                                       \
      RotaryEmbedding,                                                  \
      23,                                                               \
      T,                                                                \
      KernelDefBuilder()                                                \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>())        \
          .TypeConstraint("M", DataTypeImpl::GetTensorType<int64_t>()), \
      RotaryEmbedding<T>);

REGISTER_ONNX_KERNEL_TYPED(float)
REGISTER_ONNX_KERNEL_TYPED(MLFloat16)

template <typename T>
RotaryEmbedding<T>::RotaryEmbedding(const OpKernelInfo& info) : OpKernel(info) {
  const int64_t num_heads_attr = info.GetAttrOrDefault<int64_t>("num_heads", 0);
  const int64_t rotary_embedding_dim_attr = info.GetAttrOrDefault<int64_t>("rotary_embedding_dim", 0);
  ORT_ENFORCE(num_heads_attr >= 0 && num_heads_attr <= std::numeric_limits<int>::max(),
              "num_heads must be in range [0, ", std::numeric_limits<int>::max(),
              "]. Actual value: ", num_heads_attr);
  ORT_ENFORCE(rotary_embedding_dim_attr >= 0 && rotary_embedding_dim_attr <= std::numeric_limits<int>::max(),
              "rotary_embedding_dim must be in range [0, ", std::numeric_limits<int>::max(),
              "]. Actual value: ", rotary_embedding_dim_attr);
  num_heads = static_cast<int>(num_heads_attr);
  rotary_embedding_dim = static_cast<int>(rotary_embedding_dim_attr);
  interleaved = (info.GetAttrOrDefault<int64_t>("interleaved", 0) == 1);  // Turn 0/1 into bool
}

// TODO: rotary embedding in place
template <typename T>
Status RunRotaryEmbedding(concurrency::ThreadPool* tp, RotaryParameters parameters, const T* input,
                          const int64_t* position_ids, const T* cos_cache, const T* sin_cache, T* output,
                          bool interleaved) {
  const int batch_size = parameters.batch_size;
  const int sequence_length = parameters.sequence_length;
  const int n_heads = parameters.num_heads;
  const int head_size = parameters.head_size;
  const int head_stride = parameters.head_stride;
  const int seq_stride = parameters.seq_stride;
  const int batch_stride = parameters.batch_stride;
  const int position_ids_format = parameters.position_ids_format;
  const int max_sequence_length = parameters.max_sequence_length;
  const int rotary_emb_dim = parameters.rotary_embedding_dim;
  const int half_rotary_emb_dim = rotary_emb_dim / 2;

  // Validate position_ids values are within cos/sin cache bounds
  if (position_ids_format != 0) {
    std::ptrdiff_t position_count = 0;
    ORT_RETURN_IF_ERROR(rotary_embedding_int32_utils::CheckedMulToPtrdiff(
        batch_size, sequence_length, "position_ids element count", position_count));

    for (std::ptrdiff_t i = 0; i < position_count; ++i) {
      int64_t pos = position_ids[i];
      if (pos < 0 || pos >= static_cast<int64_t>(max_sequence_length)) {
        return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                               "position_ids value ", pos, " at index ", i,
                               " is out of range [0, ", max_sequence_length, ")");
      }
    }
  }

  // Parallel to calculate based on head_size
  std::ptrdiff_t loop_len = 0;
  ORT_RETURN_IF_ERROR(rotary_embedding_int32_utils::CheckedMulToPtrdiff(
      batch_size, sequence_length, n_heads, "total_elements", loop_len));

  std::ptrdiff_t max_batch_offset = 0;
  std::ptrdiff_t max_seq_offset = 0;
  std::ptrdiff_t max_head_offset = 0;
  std::ptrdiff_t max_block_offset = 0;
  std::ptrdiff_t max_b_s_index = 0;
  [[maybe_unused]] std::ptrdiff_t max_cache_offset = 0;
  ORT_RETURN_IF_ERROR(rotary_embedding_int32_utils::CheckedMulToPtrdiff(
      std::max(batch_size - 1, 0), batch_stride, "max_batch_offset", max_batch_offset));
  ORT_RETURN_IF_ERROR(rotary_embedding_int32_utils::CheckedMulToPtrdiff(
      std::max(sequence_length - 1, 0), seq_stride, "max_seq_offset", max_seq_offset));
  ORT_RETURN_IF_ERROR(rotary_embedding_int32_utils::CheckedMulToPtrdiff(
      std::max(n_heads - 1, 0), head_stride, "max_head_offset", max_head_offset));
  ORT_RETURN_IF_ERROR(rotary_embedding_int32_utils::CheckedAddToPtrdiff(
      max_batch_offset, max_seq_offset, "max_block_offset", max_block_offset));
  ORT_RETURN_IF_ERROR(rotary_embedding_int32_utils::CheckedAddToPtrdiff(
      max_block_offset, max_head_offset, "max_block_offset", max_block_offset));
  if (position_ids_format == 0) {
    std::ptrdiff_t total_b_s_count = 0;
    ORT_RETURN_IF_ERROR(rotary_embedding_int32_utils::CheckedMulToPtrdiff(
        batch_size, sequence_length, "total_b_s_count", total_b_s_count));
    max_b_s_index = total_b_s_count > 0 ? total_b_s_count - 1 : 0;
  } else {
    max_b_s_index = std::max(max_sequence_length - 1, 0);
  }
  ORT_RETURN_IF_ERROR(rotary_embedding_int32_utils::CheckedPtrdiffMulToPtrdiff(
      max_b_s_index, half_rotary_emb_dim, "max_cache_offset", max_cache_offset));

  // The cost is calculated as:
  //   - head_size * sizeof(T) for reading input
  //   - head_size * sizeof(T) for writing output
  //   - rotary_emb_dim * 32 for the rotary embedding operations (32 is an approximation of the number of CPU cycles)
  const double cost = static_cast<double>(head_size * sizeof(T) * 2 + rotary_emb_dim * 32);
  ThreadPool::TryParallelFor(tp, loop_len, cost, [&](std::ptrdiff_t begin, std::ptrdiff_t end) {
    for (std::ptrdiff_t ptr = begin; ptr != end; ++ptr) {
      const int b = static_cast<int>((ptr / n_heads) / sequence_length);
      const int s = static_cast<int>((ptr / n_heads) % sequence_length);
      const int n = static_cast<int>(ptr % n_heads);
      // Identify the index of batch, sequence, and head (specific range) in the input/output tensor
      // for read/write
      const std::ptrdiff_t block_offset = static_cast<std::ptrdiff_t>(b) * batch_stride +
                                          static_cast<std::ptrdiff_t>(s) * seq_stride +
                                          static_cast<std::ptrdiff_t>(n) * head_stride;
      const T* input_data = input + block_offset;
      T* output_data = output + block_offset;

      const T* cos_data;
      const T* sin_data;
      // position_ids_format == 0 means position_ids is nullptr
      // position_ids_format == 1 means position_ids is a 2D array of size (batch_size, sequence_length)
      std::ptrdiff_t b_s_index = static_cast<std::ptrdiff_t>(b) * sequence_length + s;
      if (position_ids_format != 0) {
        b_s_index = static_cast<std::ptrdiff_t>(position_ids[b_s_index]);
      }
      const std::ptrdiff_t cache_offset = b_s_index * half_rotary_emb_dim;
      cos_data = cos_cache + cache_offset;
      sin_data = sin_cache + cache_offset;

      MlasRotaryEmbedOneRow<T>(input_data, sin_data, cos_data, rotary_emb_dim, interleaved, output_data);

      if (rotary_emb_dim < head_size) {
        std::memcpy(output_data + rotary_emb_dim,
                    input_data + rotary_emb_dim,
                    (head_size - rotary_emb_dim) * sizeof(T));
      }
    }
  });

  return Status::OK();
}

template Status RunRotaryEmbedding<float>(concurrency::ThreadPool* tp, RotaryParameters parameters, const float* input,
                                          const int64_t* position_ids, const float* cos_cache, const float* sin_cache, float* output,
                                          bool interleaved);

template Status RunRotaryEmbedding<MLFloat16>(concurrency::ThreadPool* tp, RotaryParameters parameters, const MLFloat16* input,
                                              const int64_t* position_ids, const MLFloat16* cos_cache, const MLFloat16* sin_cache,
                                              MLFloat16* output, bool interleaved);

template <typename T>
Status RotaryEmbedding<T>::Compute(OpKernelContext* context) const {
  const Tensor* X = context->Input<Tensor>(0);
  const Tensor* cos_cache = context->Input<Tensor>(1);
  const Tensor* sin_cache = context->Input<Tensor>(2);
  // Optional position_ids input, can be nullptr
  const Tensor* position_ids = context->Input<Tensor>(3);

  // If rotary_embedding_dim is set (>0) and num_heads attribute not provided (==0),
  // we can only proceed if input is 4D (B, num_heads, S, head_size) so num_heads can be inferred.
  if (rotary_embedding_dim > 0 && num_heads <= 0) {
    const auto& dims = X->Shape().GetDims();
    ORT_ENFORCE(dims.size() == 4,
                "Attribute 'num_heads' must be provided when 'rotary_embedding_dim' is specified "
                "and input is not rank-4 (batch, num_heads, sequence, head).");
  }

  RotaryParameters parameters = {};
  ORT_RETURN_IF_ERROR(rotary_embedding_helper::CheckInputs<Tensor>(X,
                                                                   position_ids,
                                                                   cos_cache,
                                                                   sin_cache,
                                                                   num_heads,
                                                                   rotary_embedding_dim,
                                                                   &parameters));

  Tensor* output = context->Output(0, X->Shape());

  const T* x_src = X->Data<T>();
  const int64_t* pos_ids_data = (nullptr == position_ids) ? nullptr : position_ids->Data<int64_t>();
  const T* cos_cache_data = cos_cache->Data<T>();
  const T* sin_cache_data = sin_cache->Data<T>();
  T* output_dest = output->MutableData<T>();

  AllocatorPtr allocator;
  ORT_RETURN_IF_ERROR(context->GetTempSpaceAllocator(&allocator));
  auto* tp = context->GetOperatorThreadPool();

  return RunRotaryEmbedding<T>(tp, parameters, x_src, pos_ids_data, cos_cache_data, sin_cache_data, output_dest,
                               interleaved);
}

}  // namespace onnxruntime
