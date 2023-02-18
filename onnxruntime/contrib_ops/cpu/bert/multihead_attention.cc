// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "attention_cpu_base.h"
// #include "attention_helper.h"
#include "multihead_attention.h"
#include "multihead_attention_helper.h"

#include "core/framework/tensorprotoutils.h"
#include "core/graph/onnx_protobuf.h"
// #include "core/util/math.h"
// #include "core/util/math_cpuonly.h"
// #include "core/common/safeint.h"
// #include "core/platform/threadpool.h"

// using onnxruntime::narrow;
// using onnxruntime::concurrency::ThreadPool;

namespace onnxruntime {
namespace contrib {

// These ops are internal-only, so register outside of onnx
ONNX_OPERATOR_TYPED_KERNEL_EX(
    MultiHeadAttention,
    kMSDomain,
    1,
    float,
    kCpuExecutionProvider,
    KernelDefBuilder()
        .TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    MultiHeadAttention<float>);

template <typename T>
MultiHeadAttention<T>::MultiHeadAttention(const OpKernelInfo& info) : OpKernel(info), AttentionCPUBase(info, false) {
  int64_t num_heads = 0;
  ORT_ENFORCE(info.GetAttr("num_heads", &num_heads).IsOK() && num_heads > 0);
  num_heads_ = static_cast<int>(num_heads);

  mask_filter_value_ = info.GetAttrOrDefault<float>("mask_filter_value", -10000.0f);
}

template <typename T>
Status MultiHeadAttention<T>::Compute(OpKernelContext* context) const {
  const Tensor* query = context->Input<Tensor>(0);
  const Tensor* key = context->Input<Tensor>(1);
  const Tensor* value = context->Input<Tensor>(2);
  const Tensor* bias = context->Input<Tensor>(3);
  const Tensor* key_padding_mask = context->Input<Tensor>(4);

  if (bias != nullptr) {
    ORT_NOT_IMPLEMENTED("Q/K/V bias in multihead attention cpu kernel is not supported");
  }
  AttentionParameters parameters = {};
  ORT_RETURN_IF_ERROR(multihead_attention_helper::CheckInputs4D<Tensor>(query,
                                                                        key,
                                                                        value,
                                                                        key_padding_mask,
                                                                        &parameters,
                                                                        num_heads_,
                                                                        mask_filter_value_));

  const int batch_size = parameters.batch_size;
  const int sequence_length = parameters.sequence_length;
  const int kv_sequence_length = parameters.kv_sequence_length;
  int qk_head_size = parameters.head_size;
  int v_head_size = parameters.v_head_size;
  int v_hidden_size = parameters.v_hidden_size;

  std::vector<int64_t> output_shape(3);
  output_shape[0] = static_cast<int64_t>(batch_size);
  output_shape[1] = static_cast<int64_t>(sequence_length);
  output_shape[2] = static_cast<int64_t>(parameters.v_hidden_size);
  Tensor* output = context->Output(0, output_shape);

  // // Compute Q, K, V
  // constexpr size_t element_size = sizeof(T);

  // AllocatorPtr q_allocator;
  // ORT_RETURN_IF_ERROR(context->GetTempSpaceAllocator(&q_allocator));
  // auto q_size = SafeInt<size_t>(batch_size) * num_heads_ * sequence_length * head_size * element_size;
  // auto q_data = q_allocator->Alloc(q_size);
  // BufferUniquePtr q_buffer(q_data, BufferDeleter(std::move(q_allocator)));
  // auto Q = reinterpret_cast<T*>(q_data);

  // AllocatorPtr k_allocator;
  // ORT_RETURN_IF_ERROR(context->GetTempSpaceAllocator(&k_allocator));
  // auto k_size = SafeInt<size_t>(batch_size) * num_heads_ * kv_sequence_length * head_size * element_size;
  // auto k_data = k_allocator->Alloc(k_size);
  // BufferUniquePtr k_buffer(k_data, BufferDeleter(std::move(k_allocator)));
  // auto K = reinterpret_cast<T*>(k_data);

  // AllocatorPtr v_allocator;
  // ORT_RETURN_IF_ERROR(context->GetTempSpaceAllocator(&v_allocator));
  // auto v_size = SafeInt<size_t>(batch_size) * num_heads_ * kv_sequence_length * v_head_size * element_size;
  // auto v_data = k_allocator->Alloc(v_size);
  // BufferUniquePtr k_buffer(v_data, BufferDeleter(std::move(v_allocator)));
  // auto V = reinterpret_cast<T*>(v_data);

  // const T* Q[parameters.batch_size][num_heads_][parameters.sequence_length][parameters.head_size] = reinterpret_cast<const T*>(query->Data<T>());
  // const T* K[parameters.batch_size][num_heads_][parameters.kv_sequence_length][parameters.head_size] = reinterpret_cast<const T*>(key->Data<T>());
  // const T* V[parameters.batch_size][num_heads_][parameters.kv_sequence_length][parameters.v_head_size] = reinterpret_cast<const T*>(value->Data<T>());

  const Tensor* past = nullptr;
  const Tensor* extra_add_qk = nullptr;

  const T* Q = query->Data<T>();
  const T* K = key->Data<T>();
  const T* V = value->Data<T>();

  // Compute the attention score and apply the score to V
  return ApplyAttention(Q, K, V, key_padding_mask, past, output,
                        batch_size, sequence_length, kv_sequence_length, 
                        qk_head_size, v_head_size, v_hidden_size, 
                        extra_add_qk, context);
  // return ApplyAttention(query->Data<T>, key->Data<T>, value->Data<T>, key_padding_mask, past, output,
  //                       batch_size, sequence_length, kv_sequence_length,
  //                       parameters.head_size, parameters.v_head_size, parameters.v_hidden_size,
  //                       extra_add_qk, context);
}
}  // namespace contrib
}  // namespace onnxruntime
