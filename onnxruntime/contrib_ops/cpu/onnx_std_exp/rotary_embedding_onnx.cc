// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "contrib_ops/cpu/onnx_std_exp/rotary_embedding_onnx.h"
#include "contrib_ops/cpu/onnx_std_exp/rotary_embedding_onnx_helper.h"

#include "core/mlas/inc/mlas.h"
#include "core/platform/threadpool.h"

using onnxruntime::concurrency::ThreadPool;
using namespace onnxruntime::contrib::rotary_embedding_onnx_helper;

namespace onnxruntime {
namespace contrib {

// These ops are internal-only, so register outside of onnx
#define REGISTER_KERNEL_TYPED(T)                                        \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                        \
      RotaryEmbeddingONNX,                                                  \
      kMSDomain,                                                        \
      1,                                                                \
      T,                                                                \
      kCpuExecutionProvider,                                            \
      KernelDefBuilder()                                                \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>())        \
          .TypeConstraint("M", DataTypeImpl::GetTensorType<int64_t>()), \
      RotaryEmbeddingONNX<T>);

REGISTER_KERNEL_TYPED(float)
REGISTER_KERNEL_TYPED(MLFloat16)

template <typename T>
RotaryEmbeddingONNX<T>::RotaryEmbeddingONNX(const OpKernelInfo& info) : OpKernel(info) {
  interleaved = (info.GetAttrOrDefault<int64_t>("interleaved", 0) == 1);
  num_heads = static_cast<int>(info.GetAttrOrDefault<int64_t>("num_heads", 0));
  rotary_embedding_dim = static_cast<int>(info.GetAttrOrDefault<int64_t>("rotary_embedding_dim", 0));

  if (rotary_embedding_dim > 0) {
    ORT_ENFORCE(num_heads > 0, "num_heads must be provided if rotary_embedding_dim is specified");
  }
}

// TODO: rotary embedding in place
template <typename T>
Status RunRotaryEmbeddingONNX(concurrency::ThreadPool* tp, RotaryParameters parameters, const T* input,
                          const T* cos_cache, const T* sin_cache, const int64_t* position_ids, T* output,
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

template Status RunRotaryEmbeddingONNX<float>(concurrency::ThreadPool* tp, RotaryParameters parameters, const float* input,
                                          const float* cos_cache, const float* sin_cache, const int64_t* position_ids, float* output,
                                          bool interleaved);

template Status RunRotaryEmbeddingONNX<MLFloat16>(concurrency::ThreadPool* tp, RotaryParameters parameters, const MLFloat16* input,
                                              const MLFloat16* cos_cache, const MLFloat16* sin_cache, const int64_t* position_ids,
                                              MLFloat16* output, bool interleaved);

template <typename T>
Status RotaryEmbeddingONNX<T>::Compute(OpKernelContext* context) const {
  const Tensor* input = context->Input<Tensor>(0);
  const Tensor* cos_cache = context->Input<Tensor>(1);
  const Tensor* sin_cache = context->Input<Tensor>(2);
  const Tensor* position_ids = context->Input<Tensor>(3);  // position_ids are optional

  RotaryParameters parameters = {};
  ORT_RETURN_IF_ERROR(rotary_embedding_onnx_helper::CheckInputs<Tensor>(input,
                                                                   cos_cache,
                                                                   sin_cache,
                                                                   position_ids,
                                                                   num_heads,
                                                                   rotary_embedding_dim,
                                                                   &parameters));

  Tensor* output = context->Output(0, input->Shape());

  if (parameters.sequence_length > parameters.max_sequence_length) {
    // Launch update_cos_sin_cache kernel with scale
    ORT_NOT_IMPLEMENTED("Updating cos_cache and sin_cache in RotaryEmbedding is not currently supported");
  }

  const T* input_src = input->Data<T>();
  const T* cos_cache_data = cos_cache->Data<T>();
  const T* sin_cache_data = sin_cache->Data<T>();
  const int64_t* pos_ids_data = (nullptr == position_ids) ? nullptr : position_ids->Data<int64_t>();
  T* output_dest = output->MutableData<T>();

  AllocatorPtr allocator;
  ORT_RETURN_IF_ERROR(context->GetTempSpaceAllocator(&allocator));
  auto* tp = context->GetOperatorThreadPool();

  return RunRotaryEmbeddingONNX<T>(tp, parameters, input_src, cos_cache_data, sin_cache_data, pos_ids_data, output_dest,
                               interleaved);
}

}  // namespace contrib
}  // namespace onnxruntime
