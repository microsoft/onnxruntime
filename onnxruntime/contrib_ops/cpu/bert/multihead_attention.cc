// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "attention_cpu_base.h"
#include "multihead_attention.h"
#include "multihead_attention_helper.h"

#include "core/common/common.h"
#include "core/framework/tensorprotoutils.h"
#include "core/framework/transpose_helper.h"
#include "core/graph/onnx_protobuf.h"
#include "core/common/safeint.h"
#include "core/platform/threadpool.h"
#include "core/providers/cpu/math/element_wise_ops.h"
#include "core/providers/cpu/tensor/reshape_helper.h"

#include <unsupported/Eigen/SpecialFunctions>
#include <vector>

using onnxruntime::concurrency::ThreadPool;

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

// Add bias + transpose for each of Q/K/V
template <typename T>
Status AddBiasTranspose(const Tensor* qkv,                  // Input: Q/K/V data - query is BxSxD, key is BxLxD, value is BxLxD_v
                        const T* qkv_bias,                  // Input: Q/K/V bias - bias is (D + D + D_v)
                        OrtValue& qkv_with_bias_transposed, // Output: Q/K/V data - query is BxNxSxH, key is BxNxLxH, value is BxNxLxH_v
                        int bias_offset,                    // bias offset to enter qkv_bias
                        int batch_size,                     // batch size
                        int sequence_length,                // sequence_length for Q, kv_sequence_length for K/V
                        int num_heads,                      // num heads
                        int head_size,                      // head_size for Q/K, v_head_size for V
                        int hidden_size,                    // hidden_size for Q/K, v_hidden_size for V
                        OpKernelContext* context) {
  // Note: the comments below will refer to Q's dimensions for simplicity
  auto element_type = DataTypeImpl::GetType<T>();
  constexpr size_t element_size = sizeof(T);
  ProcessBroadcastSpanFuncs add_funcs{
      [](BroadcastHelper& per_iter_bh) {
        per_iter_bh.OutputEigen<T>() = per_iter_bh.ScalarInput0<T>() + per_iter_bh.EigenInput1<T>().array();
      },
      [](BroadcastHelper& per_iter_bh) {
        per_iter_bh.OutputEigen<T>() = per_iter_bh.EigenInput0<T>().array() + per_iter_bh.ScalarInput1<T>();
      },
      [](BroadcastHelper& per_iter_bh) {
        per_iter_bh.OutputEigen<T>() = per_iter_bh.EigenInput0<T>() + per_iter_bh.EigenInput1<T>();
      }}; // For element-wise add

  // Allocate space for output of Q(BS, D) + bias(D)
  AllocatorPtr allocator;
  ORT_RETURN_IF_ERROR(context->GetTempSpaceAllocator(&allocator));
  std::vector<int64_t> old_dims({batch_size, sequence_length, hidden_size});
  gsl::span<const int64_t> old_dims_span{old_dims};
  TensorShape qkv_with_bias_shape(old_dims_span);
  OrtValue qkv_with_bias;
  Tensor::InitOrtValue(element_type, qkv_with_bias_shape, allocator, qkv_with_bias);

  // Get Q's bias from combined bias
  std::vector<int64_t> bias_dims({num_heads * head_size});
  gsl::span<const int64_t> bias_dims_span{bias_dims};
  TensorShape bias_shape(bias_dims_span);
  OrtValue bias;
  Tensor::InitOrtValue(element_type, bias_shape, allocator, bias);
  memcpy(bias.GetMutable<Tensor>()->MutableData<T>(), qkv_bias + bias_offset, hidden_size * element_size);
  
  // Compute Q(BS, D) + bias(D) as broadcasted element-wise add
  {
    InputBroadcaster input_broadcaster(*bias.GetMutable<Tensor>(), *qkv);
    const InputBroadcaster& const_input_broadcaster = input_broadcaster;
    Tensor& output_tensor = *qkv_with_bias.GetMutable<Tensor>();

    size_t span_size = input_broadcaster.GetSpanSize();
    size_t output_size = static_cast<ptrdiff_t>(output_tensor.Shape().Size());
    void* user_data = nullptr;

    const int loop_len = output_size / span_size;
    double unit_cost = 1.0f;
    const auto cost = TensorOpCost{static_cast<double>(input_broadcaster.Input0ElementSize()) * span_size,
                                   static_cast<double>(output_tensor.DataType()->Size()) * span_size,
                                   unit_cost * span_size};
    auto tp = context->GetOperatorThreadPool();
    ThreadPool::TryParallelFor(tp, loop_len, cost, \
      [span_size, &const_input_broadcaster, &output_tensor, &add_funcs, user_data](std::ptrdiff_t first_span,
                                                                                   std::ptrdiff_t last_span) {
        InputBroadcaster segment_input_broadcaster(const_input_broadcaster);
        segment_input_broadcaster.AdvanceBy(first_span * span_size);

        OutputBroadcaster segment_output_broadcaster(span_size, output_tensor,
                                                     first_span * span_size, last_span * span_size);

        BroadcastHelper segment_helper(segment_input_broadcaster, segment_output_broadcaster, user_data);
        BroadcastLooper(segment_helper, add_funcs);
    });
  }

  // Reshape Q from BxSxD to BxSxNxH
  std::vector<int64_t> reshape_dims({batch_size, sequence_length, num_heads, head_size});
  gsl::span<const int64_t> reshape_dims_span{reshape_dims};
  TensorShape qkv_bsnh(reshape_dims_span);
  qkv_with_bias.GetMutable<Tensor>()->Reshape(qkv_bsnh);

  // Transpose Q from BxSxNxH to BxNxSxH
  std::vector<size_t> permutations({0, 2, 1, 3});
  gsl::span<const size_t> permutations_span{permutations};
  size_t from = 2, to = 1;
  SingleAxisTranspose(permutations_span, *qkv_with_bias.GetMutable<Tensor>(), *qkv_with_bias_transposed.GetMutable<Tensor>(), from, to);
  return Status::OK();
}

template <typename T>
Status MultiHeadAttention<T>::Compute(OpKernelContext* context) const {
  const Tensor* query = context->Input<Tensor>(0);
  const Tensor* key = context->Input<Tensor>(1);
  const Tensor* value = context->Input<Tensor>(2);
  const Tensor* bias = context->Input<Tensor>(3);
  const Tensor* key_padding_mask = context->Input<Tensor>(4);
  const Tensor* extra_add_qk = context->Input<Tensor>(5);
  const Tensor* past_key = context->Input<Tensor>(6);
  const Tensor* past_value = context->Input<Tensor>(7);

  if (query->Shape().GetDims().size() > 3) {
    ORT_NOT_IMPLEMENTED("Packed QKV not implemented for CPU");
  }
  if (key != nullptr && key->Shape().GetDims().size() == 5) {
    ORT_NOT_IMPLEMENTED("Packed KV not implemented for CPU");
  }
  if (key_padding_mask != nullptr) {
    ORT_NOT_IMPLEMENTED("Key padding mask not implemented for CPU");
  }
  if (extra_add_qk != nullptr) {
    ORT_NOT_IMPLEMENTED("Addition after QxK' not implemented for CPU");
  }
  if (key != nullptr && past_key != nullptr) {
    ORT_NOT_IMPLEMENTED("MHA with both key and past key not implemented for CPU");
  }
  if (value != nullptr && past_value != nullptr) {
    ORT_NOT_IMPLEMENTED("MHA with both value and past value not implemented for CPU");
  }

  const Tensor* past = nullptr;
  
  // Check that K/V have data
  ORT_ENFORCE(key != nullptr && value != nullptr);
  
  AttentionParameters parameters = {};
  ORT_RETURN_IF_ERROR(multihead_attention_helper::CheckInputs<Tensor>(query,
                                                                      key,
                                                                      value,
                                                                      bias,
                                                                      key_padding_mask,
                                                                      &parameters,
                                                                      num_heads_,
                                                                      mask_filter_value_));

  const int batch_size = parameters.batch_size;
  const int q_sequence_length = parameters.sequence_length;
  const int kv_sequence_length = parameters.kv_sequence_length;
  int qk_head_size = parameters.head_size;
  int v_head_size = parameters.v_head_size;
  int qk_hidden_size = parameters.hidden_size;
  int v_hidden_size = parameters.v_hidden_size;

  std::vector<int64_t> output_shape(3);
  output_shape[0] = static_cast<int64_t>(batch_size);
  output_shape[1] = static_cast<int64_t>(q_sequence_length);
  output_shape[2] = static_cast<int64_t>(parameters.v_hidden_size);
  Tensor* output = context->Output(0, output_shape);

  auto element_type = DataTypeImpl::GetType<T>();
  const auto* qkv_bias = bias->Data<T>();
  const int q_bias_offset = 0;
  const int k_bias_offset = qk_hidden_size;
  const int v_bias_offset = 2 * qk_hidden_size;

  OrtValue Q;
  {
    AllocatorPtr allocator;
    ORT_RETURN_IF_ERROR(context->GetTempSpaceAllocator(&allocator));
    std::vector<int64_t> new_dims({batch_size, num_heads_, q_sequence_length, qk_head_size});
    gsl::span<const int64_t> new_dims_span{new_dims};
    TensorShape q_BNSH(new_dims_span);
    Tensor::InitOrtValue(element_type, q_BNSH, allocator, Q);
    AddBiasTranspose(query, qkv_bias, Q, q_bias_offset, batch_size, q_sequence_length, num_heads_, qk_head_size, qk_hidden_size, context);
  }

  if (key->Shape().GetDims().size() == 4 && value->Shape().GetDims().size() == 4) {
    std::vector<int64_t> present_k_shape({static_cast<int64_t>(batch_size), static_cast<int64_t>(num_heads_), static_cast<int64_t>(kv_sequence_length), static_cast<int64_t>(qk_head_size)});
    std::vector<int64_t> present_v_shape({static_cast<int64_t>(batch_size), static_cast<int64_t>(num_heads_), static_cast<int64_t>(kv_sequence_length), static_cast<int64_t>(v_head_size)});
    Tensor* present_k = context->Output(1, present_k_shape);
    Tensor* present_v = context->Output(2, present_v_shape);
    // No bias add needed for K/V, key already of shape BxNxLxH, value already of shape BxNxLxH_v
    ORT_RETURN_IF_ERROR(ApplyAttention(Q.GetMutable<Tensor>()->MutableData<T>(), key->Data<T>(), value->Data<T>(),
                                       key_padding_mask, past, output, present_k, present_v, 
                                       batch_size, q_sequence_length, kv_sequence_length, 
                                       qk_head_size, v_head_size, v_hidden_size, 
                                       extra_add_qk, true, context));
    return Status::OK();
  }

  OrtValue K;
  {
    AllocatorPtr allocator;
    ORT_RETURN_IF_ERROR(context->GetTempSpaceAllocator(&allocator));
    std::vector<int64_t> new_dims({batch_size, num_heads_, kv_sequence_length, qk_head_size});
    gsl::span<const int64_t> new_dims_span{new_dims};
    TensorShape k_BNLH(new_dims_span);
    Tensor::InitOrtValue(element_type, k_BNLH, allocator, K);
    AddBiasTranspose(key, qkv_bias, K, k_bias_offset, batch_size, kv_sequence_length, num_heads_, qk_head_size, qk_hidden_size, context);
  }

  OrtValue V;
  {
    AllocatorPtr allocator;
    ORT_RETURN_IF_ERROR(context->GetTempSpaceAllocator(&allocator));
    std::vector<int64_t> new_dims({batch_size, num_heads_, kv_sequence_length, v_head_size});
    gsl::span<const int64_t> new_dims_span{new_dims};
    TensorShape v_BNLH(new_dims_span);
    Tensor::InitOrtValue(element_type, v_BNLH, allocator, V);
    AddBiasTranspose(value, qkv_bias, V, v_bias_offset, batch_size, kv_sequence_length, num_heads_, v_head_size, v_hidden_size, context);
  }

  // Compute the attention score and apply the score to V
  return ApplyAttention(Q.GetMutable<Tensor>()->MutableData<T>(), K.GetMutable<Tensor>()->MutableData<T>(), V.GetMutable<Tensor>()->MutableData<T>(), 
                        key_padding_mask, past, output, nullptr, nullptr,
                        batch_size, q_sequence_length, kv_sequence_length, 
                        qk_head_size, v_head_size, v_hidden_size, 
                        extra_add_qk, false, context);
}
}  // namespace contrib
}  // namespace onnxruntime
