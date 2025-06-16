// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cpu/llm/rotary_embedding.h"
#include "core/providers/cpu/llm/rotary_embedding_helper.h"

#include "core/mlas/inc/mlas.h"
#include "core/platform/threadpool.h"

using onnxruntime::concurrency::ThreadPool;
using namespace onnxruntime::rotary_embedding_helper;

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
  num_heads = static_cast<int>(info.GetAttrOrDefault<int64_t>("num_heads", 0));
  rotary_embedding_dim = static_cast<int>(info.GetAttrOrDefault<int64_t>("rotary_embedding_dim", 0));
  interleaved = (info.GetAttrOrDefault<int64_t>("interleaved", 0) == 1);  // Turn 0/1 into bool

  if (rotary_embedding_dim > 0) {
    ORT_ENFORCE(num_heads > 0, "num_heads must be provided if rotary_embedding_dim is specified");
  }
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
  const int rotary_emb_dim = parameters.rotary_embedding_dim;
  const int half_rotary_emb_dim = rotary_emb_dim / 2;

  const int loop_len = batch_size * sequence_length * n_heads;
  const double cost = static_cast<double>(rotary_emb_dim);
  ThreadPool::TryParallelFor(tp, loop_len, cost, [&](std::ptrdiff_t begin, std::ptrdiff_t end) {
    for (std::ptrdiff_t ptr = begin; ptr != end; ++ptr) {
      const int b = static_cast<int>((ptr / n_heads) / sequence_length);
      const int s = static_cast<int>((ptr / n_heads) % sequence_length);
      const int n = static_cast<int>(ptr % n_heads);

      const int block_offset = b * batch_stride + s * seq_stride + n * head_stride;

      const T* input_data = input + block_offset;
      T* output_data = output + block_offset;

      const T* cos_data;
      const T* sin_data;
      int cache_offset;
      if (position_ids_format == -1) {
        cache_offset = (b * sequence_length + s) * half_rotary_emb_dim;
      } else {
        // Cache is (M, H/2) or (M, rotary_embedding_dim/2)
        const int position_id = (position_ids_format == 0)
                                    ? static_cast<int>(position_ids[0]) + s
                                    : static_cast<int>(position_ids[b * sequence_length + s]);
        cache_offset = position_id * half_rotary_emb_dim;
      }
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
  const Tensor* position_ids = context->Input<Tensor>(3);  // Optional, can be nullptr

  RotaryParameters parameters = {};
  ORT_RETURN_IF_ERROR(rotary_embedding_helper::CheckInputs<Tensor>(X,
                                                                   position_ids,
                                                                   cos_cache,
                                                                   sin_cache,
                                                                   num_heads,
                                                                   rotary_embedding_dim,
                                                                   &parameters));

  Tensor* output = context->Output(0, X->Shape());

  if (parameters.sequence_length > parameters.max_sequence_length) {
    // Launch update_cos_sin_cache kernel with scale
    ORT_NOT_IMPLEMENTED("Updating cos_cache and sin_cache in RotaryEmbedding is not currently supported");
  }

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
