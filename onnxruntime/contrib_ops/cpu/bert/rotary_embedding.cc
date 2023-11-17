// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "contrib_ops/cpu/bert/rotary_embedding.h"
#include "contrib_ops/cpu/bert/rotary_embedding_helper.h"

#include "core/platform/threadpool.h"

using onnxruntime::concurrency::ThreadPool;
using namespace onnxruntime::contrib::rotary_embedding_helper;

namespace onnxruntime {
namespace contrib {

// These ops are internal-only, so register outside of onnx
ONNX_OPERATOR_TYPED_KERNEL_EX(
    RotaryEmbedding,
    kMSDomain,
    1,
    float,
    kCpuExecutionProvider,
    KernelDefBuilder()
        .TypeConstraint("T", DataTypeImpl::GetTensorType<float>())
        .TypeConstraint("M", DataTypeImpl::GetTensorType<int64_t>()),
    RotaryEmbedding<float>);

template <typename T>
RotaryEmbedding<T>::RotaryEmbedding(const OpKernelInfo& info) : OpKernel(info) {
  scale = info.GetAttrOrDefault<float>("scale", 1.0);
  interleaved = (info.GetAttrOrDefault<int64_t>("interleaved", 0) == 1);
}

template <typename T>
Status RotaryEmbedding<T>::Compute(OpKernelContext* context) const {
  const Tensor* input = context->Input<Tensor>(0);
  const Tensor* position_ids = context->Input<Tensor>(1);
  const Tensor* cos_cache = context->Input<Tensor>(2);
  const Tensor* sin_cache = context->Input<Tensor>(3);

  RotaryParameters parameters = {};
  ORT_RETURN_IF_ERROR(rotary_embedding_helper::CheckInputs<Tensor>(input,
                                                                   position_ids,
                                                                   cos_cache,
                                                                   sin_cache,
                                                                   &parameters));

  Tensor* output = context->Output(0, input->Shape());

  if (parameters.sequence_length > parameters.max_sequence_length) {
    // Launch update_cos_sin_cache kernel with scale
    ORT_NOT_IMPLEMENTED("Updating cos_cache and sin_cache in RotaryEmbedding is not currently supported");
  }

  const T* input_src = input->Data<T>();
  const int64_t* pos_ids_data = position_ids->Data<int64_t>();
  const T* cos_cache_data = cos_cache->Data<T>();
  const T* sin_cache_data = sin_cache->Data<T>();
  T* output_dest = output->MutableData<T>();

  const int batch_size = parameters.batch_size;
  const int sequence_length = parameters.sequence_length;
  const int num_heads = parameters.num_heads;
  const int head_size = parameters.head_size;
  const int position_ids_format = parameters.position_ids_format;
  const int half_head_size = head_size / 2;
  // Default input tensor shape is [batch, seq_len, hidden_size]
  int head_stride = head_size;
  int seq_stride = num_heads * head_stride;
  int batch_stride = sequence_length * seq_stride;
  if (parameters.transposed) {
    // Transposed input tensor shape is [batch, num_heads, seq_len, head_size]
    seq_stride = head_size;
    head_stride = sequence_length * seq_stride;
    batch_stride = num_heads * head_stride;
  }

  AllocatorPtr allocator;
  ORT_RETURN_IF_ERROR(context->GetTempSpaceAllocator(&allocator));
  auto* tp = context->GetOperatorThreadPool();

  const int loop_len = batch_size * sequence_length * num_heads;
  const double cost = static_cast<double>(head_size);
  ThreadPool::TryParallelFor(tp, loop_len, cost, [&](std::ptrdiff_t begin, std::ptrdiff_t end) {
    for (std::ptrdiff_t ptr = begin; ptr != end; ++ptr) {
      const int b = static_cast<int>((ptr / num_heads) / sequence_length);
      const int s = static_cast<int>((ptr / num_heads) % sequence_length);
      const int n = static_cast<int>(ptr % num_heads);

      const int block_offset = b * batch_stride + s * seq_stride + n * head_stride;

      const T* input_data = input_src + block_offset;
      T* output_data = output_dest + block_offset;

      // Cache is (M, H/2)
      const int position_id = (position_ids_format == 0)
                                  ? static_cast<int>(pos_ids_data[0]) + s
                                  : static_cast<int>(pos_ids_data[b * sequence_length + s]);
      const int cache_offset = position_id * half_head_size;
      const T* cos_data = cos_cache_data + cache_offset;
      const T* sin_data = sin_cache_data + cache_offset;

      int cache_idx = 0;
      T sign = 0;
      int j = 0;
      for (int i = 0; i < head_size; i++) {
        if (interleaved) {
          cache_idx = (i / 2) % half_head_size;
          sign = (i % 2 == 0) ? static_cast<T>(-1) : static_cast<T>(1);
          j = (i % 2 == 0) ? i + 1 : i - 1;  // i - sign
        } else {
          cache_idx = i % half_head_size;
          sign = (i < half_head_size) ? static_cast<T>(-1) : static_cast<T>(1);
          j = (i + half_head_size) % head_size;
        }
        output_data[i] = input_data[i] * cos_data[cache_idx] + sign * input_data[j] * sin_data[cache_idx];
      }
    }
  });

  return Status::OK();
}

}  // namespace contrib
}  // namespace onnxruntime
